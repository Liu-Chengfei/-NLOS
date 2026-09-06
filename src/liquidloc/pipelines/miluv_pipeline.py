"""MILUV 外部验证流水线。

这个模块负责读取 MILUV 数据集的原始序列、执行字段映射、构建事件，
并把结果路由到 CorePipeline 进行推理。
它还处理 MILUV 特有的锚点布局解析（包括 3D→2D 投影），
确保 teacher 和估计器能拿到正确的 2D 锚点几何信息。

上游依赖:
- liquidloc.dataio.readers.miluv_reader: MILUV 原始数据读取器
- liquidloc.dataio.adapters.field_mapper: 外部字段到内部字段的映射
- liquidloc.dataio.adapters.event_builder: 事件构建与合并
- liquidloc.pipelines.core_pipeline: 核心推理流水线

下游调用者:
- liquidloc.pipelines.public_benchmark_pipeline: 公开基准路由
- 外部 MILUV 验证脚本

核心变量:
- MILUV_OFFICIAL_ANCHOR_METADATA_SOURCE: MILUV 官方锚点元数据来源标识（从 common.constants 单源导入）
"""

from __future__ import annotations  # 允许在类型标注里引用尚未定义的类名。

from collections.abc import Mapping  # 用于判断映射型输入，替代 dict 做类型检查。
from pathlib import Path  # 用于路径拼接和解析。
from typing import Any  # 用于给动态结构写宽松类型注解。

from liquidloc.common.constants import (  # 单源真相常量（D9 漂移根因修复）。
    DATASET_NAME_MILUV,  # 数据集名字常量。
    MILUV_OFFICIAL_ANCHOR_METADATA_SOURCE,  # MILUV 官方锚点元数据来源标识。
)
from liquidloc.common.io_utils import write_json
from liquidloc.common.paths import get_standard_dirs, resolve_output_root  # 解析项目标准目录结构和统一输出根目录。
from liquidloc.common.types import StageResult  # 流水线阶段结果统一返回结构。
from liquidloc.common.validation import is_string_like, validate_path_component  # 统一判断字符串类型和路径组件校验。
from liquidloc.dataio.adapters.event_builder import (  # 事件构建器。
    build_imu_events,  # 构建 IMU 事件列表。
    build_uwb_events,  # 构建 UWB 事件列表。
    build_vio_events,  # 构建 VIO 事件列表。
    merge_and_finalize_events,  # 合并并排序所有模态事件，生成最终事件序列。
)
from liquidloc.dataio.adapters.field_mapper import map_external_fields  # 外部字段映射到内部字段名。
from liquidloc.dataio.manifests.dataset_checks import REQUIRED_RAW_KEYS  # 标准数据集必须有的原始键。
from liquidloc.dataio.manifests.dataset_checks import run_dataset_checks  # 运行数据集合法性检查（MILUV 等价合同校验）。
from liquidloc.dataio.readers.miluv_reader import read_miluv_sequence  # 读取 MILUV 序列 bundle。
from liquidloc.interfaces.pipeline_api import PipelineAPI, normalize_pipeline_cfg  # 流水线接口基类与统一配置规整 helper。
from liquidloc.pipelines.core_pipeline import CorePipeline  # 核心推理/评测调度流水线。
from liquidloc.sensors.anchor_model import project_anchor_layout_xy  # 3D→2D 锚点布局投影。


def _normalize_seq_ids(seq_ids: Any) -> list[str]:
    """归一化序列 ID 列表，去重并拒绝空白、非字符串值和路径穿越字符。

    Args:
        seq_ids: 原始序列 ID 输入，必须是可迭代对象。

    Returns:
        去重后的非空字符串序列 ID 列表。

    Raises:
        TypeError: 输入为字符串/字节、不可迭代，或包含非字符串元素。
        ValueError: 包含空白项或路径穿越字符（..、/、\\、null 字节）。
    """
    if isinstance(seq_ids, (str, bytes)):  # 单个字符串或字节不是合法的序列列表。
        raise TypeError("seq_ids must be a non-empty list")
    try:  # 非可迭代对象（如 int/bool）调用 list() 会抛 TypeError。
        seq_id_items = list(seq_ids or [])  # 逐个处理输入序列。
    except TypeError:  # 统一错误消息，避免暴露底层 "object is not iterable"。
        raise TypeError("seq_ids must be a non-empty list") from None
    normalized: list[str] = []  # 收集归一化后的序列 ID。
    seen: set[str] = set()  # 用于去重。
    for seq_id in seq_id_items:  # 逐个处理输入序列。
        if not is_string_like(seq_id):  # 每个元素必须是字符串。
            raise TypeError("seq_ids must contain only strings")
        normalized_seq_id = str(seq_id).strip()  # 去掉首尾空白。
        if not normalized_seq_id:  # 空白字符串视为非法。
            raise ValueError("seq_ids must not contain blank items")
        validate_path_component(normalized_seq_id, name="seq_id")  # 校验不含路径穿越字符，防止下游 read_miluv_sequence 拼接 Path(raw_root)/seq_id 时路径穿越，与 prepare_pipeline/core_pipeline/eval_pipeline 模式一致。
        if normalized_seq_id in seen:  # 重复项跳过，保证输出唯一。
            continue
        seen.add(normalized_seq_id)  # 记录已见序列。
        normalized.append(normalized_seq_id)  # 加入输出列表。
    return normalized


def _normalize_seq_id_key(seq_id: Any, *, field_name: str) -> str:
    """归一化单个序列 ID 键，拒绝空白和非字符串值。

    Args:
        seq_id: 待归一化的序列 ID。
        field_name: 字段名，用于错误消息定位。

    Returns:
        去空白后的序列 ID 字符串。

    Raises:
        TypeError: 输入不是字符串。
        ValueError: 输入为空白字符串。
    """
    if not is_string_like(seq_id):  # 必须是字符串类型。
        raise TypeError(f"{field_name} must be a string")
    normalized_seq_id = str(seq_id).strip()  # 去掉首尾空白。
    if not normalized_seq_id:  # 空白字符串视为非法。
        raise ValueError(f"{field_name} must not be blank")
    return normalized_seq_id


def _normalize_scene_id_value(scene_id: Any, *, field_name: str) -> str:
    """归一化场景 ID 字符串，拒绝空白和非字符串值。

    Args:
        scene_id: 待归一化的场景 ID。
        field_name: 字段名，用于错误消息定位。

    Returns:
        去空白后的场景 ID 字符串。

    Raises:
        TypeError: 输入不是字符串。
        ValueError: 输入为 None（字段缺失）或空白字符串。
    """
    if scene_id is None:  # None 表示字段缺失，应抛 ValueError 而非 TypeError，与 raw_root/seq_ids 等必填项的 value contract 错误类型一致。
        raise ValueError(f"{field_name} is required")
    if not is_string_like(scene_id):  # 必须是字符串类型。
        raise TypeError(f"{field_name} must be a string")
    normalized_scene_id = str(scene_id).strip()  # 去掉首尾空白。
    if not normalized_scene_id:  # 空白字符串视为非法。
        raise ValueError(f"{field_name} must not be blank")
    return normalized_scene_id


def _resolve_scene_id(cfg: dict[str, Any], seq_id: str) -> str:
    """解析某条序列的场景 ID。

    优先使用 scene_id_by_seq 中的显式映射，其次回退到统一 scene_id，
    最后用默认的 miluv:seq_id 格式。

    Args:
        cfg: 流水线配置，可能包含 scene_id_by_seq 或 scene_id。
        seq_id: 当前序列 ID。

    Returns:
        解析后的场景 ID 字符串。

    Raises:
        ValueError: 显式映射值为空字符串时。
    """
    raw_by_seq_id = cfg.get("scene_id_by_seq") or {}  # 先看是否有按序列定制的映射。
    by_seq_id = {  # 归一化映射键，确保键值都是干净字符串。
        _normalize_seq_id_key(raw_seq_id, field_name="scene_id_by_seq key"): scene_id
        for raw_seq_id, scene_id in dict(raw_by_seq_id).items()
    }
    if seq_id in by_seq_id:  # 显式映射优先。
        scene_id = by_seq_id[seq_id]
        if not is_string_like(scene_id) or not str(scene_id).strip():  # 映射值必须是非空字符串。
            raise ValueError(f"scene_id_by_seq[{seq_id!r}] must be a non-empty string")
        return str(scene_id).strip()  # 返回规范化后的场景 ID。
    scene_id = cfg.get("scene_id")  # 再看是否有统一的场景 ID。
    if is_string_like(scene_id) and str(scene_id).strip():  # 非空字符串才算合法。
        return str(scene_id).strip()  # 返回规范化后的统一场景 ID。
    return f"{DATASET_NAME_MILUV}:{seq_id}"  # 默认用 miluv:seq_id 格式，数据集名字走单源常量（D9 漂移根因修复）。


def _resolve_output_root(cfg: dict[str, Any], default_name: str) -> Path:
    """解析输出根目录，委托给 common.paths.resolve_output_root 统一处理。"""
    return resolve_output_root(cfg, default_name)


def _resolve_sequence_anchor_layout(
    raw_bundle: dict[str, Any],
    read_report: dict[str, Any],
) -> dict[str, Any] | None:
    """解析序列的锚点布局，优先使用原始 bundle 中的布局，其次从官方元数据投影。

    MILUV 数据可能直接提供 2D 锚点布局（anchor_layout_raw），
    也可能只提供 3D 元数据（anchor_layout_metadata_raw）需要投影。

    Args:
        raw_bundle: 原始数据 bundle，可能包含 anchor_layout_raw 或 anchor_layout_metadata_raw。
        read_report: 读取报告，包含锚点元数据的可用性和来源信息。

    Returns:
        解析后的 2D 锚点布局字典；无法解析时返回 None。
    """
    source_anchor_layout = raw_bundle.get("anchor_layout_raw")  # 优先看原始 bundle 是否直接提供布局。
    if isinstance(source_anchor_layout, Mapping):  # 直接提供时直接返回副本。
        return dict(source_anchor_layout)

    anchor_layout_metadata = raw_bundle.get("anchor_layout_metadata_raw")  # 其次看是否提供 3D 元数据。
    if isinstance(anchor_layout_metadata, Mapping):  # 有元数据时检查是否需要投影。
        if bool(read_report.get("anchor_layout_metadata_available")):  # 元数据确实可用时才处理。
            if read_report.get("anchor_layout_metadata_source") == MILUV_OFFICIAL_ANCHOR_METADATA_SOURCE:  # 只处理官方来源的元数据（单源常量，D9 漂移根因修复）。
                if read_report.get("anchor_layout_position_dim") == 3:  # 3D 坐标需要投影为 2D。
                    return project_anchor_layout_xy(anchor_layout_metadata)  # 执行 3D→2D 投影。
    return None  # 无法解析时返回 None。


class MiluvPipeline(PipelineAPI):
    """MILUV 外部验证流水线。

    读取 MILUV 序列，执行字段映射和事件构建，
    解析锚点布局（含 3D→2D 投影），然后路由到 CorePipeline 进行推理。
    """

    def run(
        self,
        pipeline_cfg: dict | None = None,
        runtime_context: dict | None = None,
    ) -> StageResult:
        """执行 MILUV 验证流水线。

        Args:
            pipeline_cfg: 运行配置，须包含 seq_ids、field_mapping、raw_root。
            runtime_context: 运行时上下文，当前未使用。

        Returns:
            StageResult，包含 CorePipeline 产物和映射报告。

        Raises:
            ValueError: seq_ids 为空、field_mapping 缺失或 raw_root 缺失。
        """
        del runtime_context  # 当前未使用，显式删除避免误用。

        cfg = normalize_pipeline_cfg(pipeline_cfg)  # 统一转成普通字典，并拒绝非映射配置。
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "seq_ids": cfg.get("seq_ids"),
            "raw_root": str(cfg.get("raw_root")) if cfg.get("raw_root") else None,
            "methods": cfg.get("methods"),
            "output_root": str(cfg.get("output_root")) if cfg.get("output_root") else None,
            "scene_tasks_count": len(cfg.get("scene_tasks") or []),
        }, "MiluvPipeline.run 入口参数")
        seq_ids = _normalize_seq_ids(cfg.get("seq_ids"))  # 归一化并去重序列列表。
        if not seq_ids:  # 序列列表不能为空。
            raise ValueError("seq_ids must be a non-empty list")

        field_mapping = cfg.get("field_mapping")  # 读取字段映射配置。
        if not isinstance(field_mapping, Mapping) or not field_mapping:  # MILUV 必须提供非空字段映射，因为原始字段名与内部不一致。
            raise ValueError("field_mapping must be a non-empty mapping for MILUV")

        raw_root = cfg.get("raw_root")  # 原始数据根目录。
        if raw_root is None:  # raw_root 是必需的。
            raise ValueError("raw_root is required")

        output_root = _resolve_output_root(cfg, "miluv_pipeline")  # 解析输出根目录。
        reports_dir = output_root / "reports"  # 报告输出目录。
        reports_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在。

        events_by_seq_id: dict[str, list[dict[str, Any]]] = {}  # 按序列编号索引的事件序列。
        mapping_reports: dict[str, Any] = {}  # 按序列编号索引的映射报告。
        scene_tasks: list[dict[str, Any]] = []  # 场景任务列表，供 CorePipeline 消费。
        provided_scene_tasks = list(cfg.get("scene_tasks") or [])  # 上游可能已经提供了场景任务。
        provided_task_by_seq_id = {  # 按序列编号索引已提供的场景任务，方便后续查找。
            _normalize_seq_id_key(task["seq_id"], field_name="scene_tasks[].seq_id"): {
                **dict(task),  # 复制任务内容。
                "seq_id": _normalize_seq_id_key(task["seq_id"], field_name="scene_tasks[].seq_id"),  # 归一化序列 ID。
                "scene_id": _normalize_scene_id_value(task.get("scene_id"), field_name="scene_tasks[].scene_id"),  # 归一化场景 ID；用 .get 避免 scene_id 缺失时抛 KeyError，由 _normalize_scene_id_value 统一抛 ValueError。
            }
            for task in provided_scene_tasks  # 遍历所有已提供的任务。
            if isinstance(task, dict) and task.get("seq_id")  # 只处理有效任务。
        }

        for index, seq_id in enumerate(seq_ids):  # 逐个序列处理。
            scene_task = dict(provided_task_by_seq_id.get(seq_id) or {})  # 先看是否有上游提供的任务。
            if not scene_task:  # 没有上游任务时构造默认场景任务。
                scene_task = {
                    "task_id": f"miluv_{index:04d}",  # 默认任务编号，4 位数对齐 scene_sampler 和 prepared_inputs。
                    "scene_id": _resolve_scene_id(cfg, seq_id),  # 解析场景 ID。
                    "seq_id": seq_id,  # 序列编号。
                    "dataset_name": DATASET_NAME_MILUV,  # 数据集名称（单源常量，D9 漂移根因修复）。
                    "axes": dict(cfg.get("axes") or {"dataset": DATASET_NAME_MILUV}),  # 轴配置，默认标记为 miluv 数据集（D9 漂移根因修复）。
                }

            normalized_field_mapping = dict(field_mapping)  # 复制一份映射表，避免后续链路改写调用方对象。
            raw_bundle, read_report = read_miluv_sequence(seq_id, raw_root)  # 读取 MILUV 原始数据。
            internal_bundle, mapping_report = map_external_fields(raw_bundle, normalized_field_mapping)  # 执行字段映射。
            # MILUV 等价合同校验：与 PreparePipeline 走同一套 run_dataset_checks 闸门，
            # 保证 MILUV 路径不绕过统一数据合同（用户审查标准 #6/#7/#8 公开对齐面/公平同口径）。
            miluv_check_report = run_dataset_checks(internal_bundle, required_raw_keys=REQUIRED_RAW_KEYS)
            if not miluv_check_report.get('is_valid'):
                raise ValueError(
                    f'MILUV sequence {seq_id!r} failed dataset checks: '
                    f"missing_streams={miluv_check_report.get('missing_streams')}, "
                    f"empty_streams={miluv_check_report.get('empty_streams')}, "
                    f"bad_streams={miluv_check_report.get('bad_streams')}"
                )
            sequence_anchor_layout = _resolve_sequence_anchor_layout(raw_bundle, read_report)  # 解析锚点布局（含 3D→2D 投影）。
            if sequence_anchor_layout is not None:  # 有锚点布局时写入场景任务。
                scene_task["anchor_layout"] = sequence_anchor_layout  # 供 CorePipeline 的估计器初始化使用。

            imu_events = build_imu_events(internal_bundle.get("imu_raw", []), scene_task["scene_id"], seq_id)  # 构建 IMU 事件。
            uwb_events = build_uwb_events(internal_bundle.get("uwb_raw", []), scene_task["scene_id"], seq_id)  # 构建 UWB 事件。
            vio_events = build_vio_events(internal_bundle.get("vio_raw", []), scene_task["scene_id"], seq_id)  # 构建 VIO 事件。
            events = merge_and_finalize_events([imu_events, uwb_events, vio_events])  # 合并并排序所有模态事件。

            events_by_seq_id[seq_id] = events  # 按序列索引存储事件。
            mapping_reports[seq_id] = {  # 记录当前序列的映射报告。
                "read_report": read_report,  # 读取报告。
                "mapping_report": mapping_report,  # 映射报告。
                "event_count": len(events),  # 事件数量。
                "anchor_layout": sequence_anchor_layout,  # 解析后的锚点布局。
            }
            scene_tasks.append(scene_task)  # 加入场景任务列表。

        core_output_root = output_root / "core"  # CorePipeline 的输出子目录。
        if provided_scene_tasks:  # 上游提供了场景任务时，清理可能残留的旧预测文件。
            predictions_dir = core_output_root / "predictions"  # 预测文件目录。
            if predictions_dir.is_dir():  # 目录存在时才清理。
                for stale_prediction in predictions_dir.glob("*.json"):  # 删除所有旧预测文件。
                    stale_prediction.unlink()

        core_input = {  # 构造 CorePipeline 的输入配置。
            "scene_tasks": scene_tasks,  # 场景任务列表。
            "events_by_seq_id": events_by_seq_id,  # 按序列索引的事件。
            "methods": list(cfg.get("methods") or []),  # 方法列表。
            "estimator_cfgs": cfg.get("estimator_cfgs") or {},  # 估计器配置覆盖。
            "model_cfgs": cfg.get("model_cfgs") or {},  # 模型配置覆盖。
            "output_root": core_output_root,  # CorePipeline 输出目录。
            "project_root": cfg.get("project_root"),  # 项目根目录。
        }
        core_result = CorePipeline().run(core_input)  # 执行核心推理流水线。

        mapping_path = reports_dir / "mapping_reports.json"  # 映射报告文件路径。
        write_json(mapping_path, mapping_reports)  # 写出映射报告。

        metadata = dict(core_result.metadata)  # 复制 CorePipeline 的元数据。
        metadata["mapping_reports"] = mapping_reports  # 补充映射报告。
        return StageResult(
            stage_name="miluv_pipeline",  # 阶段名。
            artifacts=list(core_result.artifacts) + [str(mapping_path)],  # 合并 CorePipeline 产物和映射报告。
            metadata=metadata,  # 完整元数据。
        )


def run(pipeline_cfg):
    """兼容旧入口的便捷函数，直接转发给 MiluvPipeline.run。

    Args:
        pipeline_cfg: 传给 MiluvPipeline.run 的配置字典。

    Returns:
        MiluvPipeline().run(...) 的原始返回值。
    """
    return MiluvPipeline().run(pipeline_cfg)
