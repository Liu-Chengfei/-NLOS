"""公开基准路由流水线。

这个模块负责把公开数据集的基准评测请求路由到正确的子流水线。
它读取数据集注册表、验证协议门控、构造场景任务，
然后根据数据集名称选择 PreparePipeline + CorePipeline 或 MiluvPipeline 执行，
最后汇总报告和审计产物。

上游依赖:
- liquidloc.dataio.registry.public_dataset_registry: 公开数据集注册表
- liquidloc.protocol.experiment_gates: 实验协议门控
- liquidloc.common.prepared_inputs: 预处理输入加载工具

下游调用者:
- 外部基准评测脚本
- CI/CD 基准验证流程

核心变量:
- dataset_name: 数据集名称，决定路由到哪条子流水线
- registry_cfg: 数据集注册表配置
- gate_report: 协议门控报告
"""

from __future__ import annotations  # 允许在类型标注里引用尚未定义的类名。

from collections.abc import Mapping  # 用于校验 field_mapping 覆盖时仍遵守映射合同。
from copy import deepcopy  # 深拷贝 field_mapping 嵌套结构，避免路由层污染源对象。
from pathlib import Path  # 用于路径拼接和解析。

from liquidloc.common.config_utils import load_dataset_config  # 读取数据集配置，补齐 raw_root 和 field_mapping。
from liquidloc.common.constants import DATASET_NAME_MILUV  # 数据集名字常量（单源真相，D9 漂移根因修复）。
from liquidloc.common.io_utils import dumps_json_text, read_json  # 严格 JSON 读写工具，拒绝 NaN/Infinity。
from liquidloc.common.paths import get_standard_dirs, resolve_output_root  # 解析项目标准目录结构和统一输出根目录解析。
from liquidloc.common.prepared_inputs import (  # 预处理输入加载工具。
    load_ground_truth_by_seq_id,  # 按序列编号加载真值映射。
    load_prepare_manifest,  # 加载 prepare 阶段产出的清单文件。
    load_prepared_events_by_seq_id,  # 按序列编号加载预处理后的事件（G7.1：懒加载 .pkl.gz）。
    load_source_report_by_seq_id,  # 按序列编号加载数据源报告。
)
from liquidloc.common.types import StageResult  # 流水线阶段结果统一返回结构。
from liquidloc.common.validation import is_string_like  # 统一判断字符串类型。
from liquidloc.dataio.registry.public_dataset_registry import (  # 公开数据集注册表相关工具。
    get_dataset_entry,  # 按数据集名称获取注册条目。
    load_public_dataset_registry,  # 加载公开数据集注册表配置。
    normalize_public_dataset_name,  # 归一化数据集名称（大小写/别名统一）。
)
from liquidloc.interfaces.pipeline_api import PipelineAPI, normalize_pipeline_cfg  # 流水线接口基类与统一配置规整 helper。
from liquidloc.pipelines.core_pipeline import CorePipeline  # 核心推理/评测调度流水线。
from liquidloc.pipelines.miluv_pipeline import MiluvPipeline  # MILUV 专用外部验证流水线。
from liquidloc.pipelines.prepare_pipeline import PreparePipeline  # 原始数据准备流水线。
from liquidloc.protocol.experiment_gates import (  # 实验协议门控工具。
    load_experiment_protocol,  # 加载实验协议配置。
    normalize_public_benchmark_request,  # 规范化公开基准请求是否满足协议要求。
)


def _resolve_optional_project_path(path_value, *, field_name: str, project_root: Path) -> Path | None:
    """解析可选路径，支持绝对路径和相对路径，拒绝空白字符串。

    当 path_value 为 None 时返回 None；为字符串时先去空白再判断是否为空；
    相对路径按 project_root 解析为绝对路径。支持 ``~`` 展开到用户主目录，
    与 ``paths._resolve_project_root_arg`` / ``resolve_output_root`` 保持一致。

    Args:
        path_value: 待解析的路径值，可为 None、字符串或 Path。
        field_name: 字段名，用于错误消息定位。
        project_root: 项目根目录，用于解析相对路径。

    Returns:
        解析后的绝对路径；输入为 None 时返回 None。

    Raises:
        ValueError: 路径为空白字符串或空白 Path 时。
        TypeError: path_value 不是 None、字符串或 Path 类型时。
    """
    if path_value is None:  # None 表示未指定，直接返回。
        return None
    if is_string_like(path_value):  # 字符串类型需要额外检查空白。
        path_str = str(path_value)  # 统一转 Python str，避免 numpy.str_ 在某些版本中不再是 str 子类时 Path() 失败。
        if not path_str.strip():  # 空白字符串视为非法输入。
            raise ValueError(f"{field_name} must not be blank")
        candidate = Path(path_str)  # 用转换后的 str 构造 Path。
    else:  # 其他类型只允许 Path 对象，与 resolve_output_root 的类型门控保持一致。
        if not isinstance(path_value, Path):  # 拒绝非 str/Path/None 类型，给出清晰错误而非 Path() 的隐式 TypeError。
            raise TypeError(f"{field_name} must be str, Path or None, got {type(path_value).__name__}")
        candidate = Path(path_value)
        if not str(candidate).strip():  # 空白 Path（如 Path("  ")）没有路径意义，与 _resolve_project_root_arg 保持一致。
            raise ValueError(f"{field_name} must not be blank, got {path_value!r}")
    candidate = candidate.expanduser()  # 展开 ~ 为用户主目录，与 paths.py 中的 _resolve_project_root_arg / resolve_output_root 保持一致。
    if not candidate.is_absolute():  # 相对路径按 project_root 解析。
        candidate = project_root / candidate  # 拼接成绝对路径。
    return candidate.resolve()  # 返回规范化后的绝对路径。


def _resolve_dataset_runtime_contract(
    *,
    cfg: dict,
    dataset_name: str,
    project_root: Path,
) -> dict[str, object]:
    """补齐公开数据集运行所需的 raw_root 和 field_mapping 合同。"""
    dataset_cfg_path = project_root / "configs" / "datasets" / f"{dataset_name}.yaml"
    dataset_cfg = load_dataset_config(dataset_cfg_path, required_keys=("raw_root",))

    resolved_raw_root = cfg.get("raw_root")
    if resolved_raw_root is None:
        resolved_raw_root = dataset_cfg.get("raw_root")
        if resolved_raw_root is None or str(resolved_raw_root).strip() == "":
            raise ValueError(f"{dataset_cfg_path} must define a non-empty raw_root")
    # 统一对调用方和数据集配置的 raw_root 走相同的路径解析与空白校验：
    # 调用方显式提供相对路径或空白字符串时也必须经 project_root 解析，
    # 避免 D7 公平性/可复现 与 D3 数据合同保真 被绕过。
    resolved_raw_root = str(_resolve_optional_project_path(
        resolved_raw_root,
        field_name=f"{dataset_name}.raw_root",
        project_root=project_root,
    ))

    resolved_field_mapping = cfg.get("field_mapping")
    if resolved_field_mapping is None:
        dataset_field_mapping = dataset_cfg.get("field_mapping") or {}
        # 校验数据集配置的 field_mapping 类型，避免 list/str 等非映射类型被 dict() 静默强转，
        # 破坏 D3 数据合同保真。
        if not isinstance(dataset_field_mapping, Mapping):
            raise ValueError(f"{dataset_cfg_path} field_mapping must be a mapping when present")
        resolved_field_mapping = deepcopy(dataset_field_mapping)  # 深拷贝，避免路由层污染数据集配置对象。
    elif not isinstance(resolved_field_mapping, Mapping) or not resolved_field_mapping:  # 调用方显式覆盖时也必须保持非空映射合同。
        raise ValueError("field_mapping must be a non-empty mapping")
    else:
        resolved_field_mapping = deepcopy(resolved_field_mapping)  # 深拷贝，避免路由层污染调用方对象。

    return {
        "raw_root": resolved_raw_root,
        "field_mapping": resolved_field_mapping,
    }


class PublicBenchmarkPipeline(PipelineAPI):
    """公开基准路由流水线。

    根据数据集名称把基准评测请求路由到对应的子流水线：
    - miluv 数据集走 MiluvPipeline（专用读取+核心推理）
    - 其他数据集走 PreparePipeline + CorePipeline（通用准备+核心推理）

    路由完成后汇总报告和审计产物，统一返回 StageResult。
    """

    def run(
        self,
        pipeline_cfg: dict | None = None,
        runtime_context: dict | None = None,
    ) -> StageResult:
        """执行公开基准路由流水线。

        Args:
            pipeline_cfg: 运行配置，须包含 dataset_name，可选包含
                registry_path、experiment_protocol_path、methods 等。
            runtime_context: 运行时上下文，当前未使用。

        Returns:
            StageResult，包含子流水线产物路径、基准报告和协议门控审计。

        Raises:
            ValueError: dataset_name 缺失或 output_root 为空白字符串。
        """
        del runtime_context  # 当前未使用，显式删除避免误用。

        cfg = normalize_pipeline_cfg(pipeline_cfg)  # 统一转成普通字典，并拒绝非映射输入。
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "dataset_name": cfg.get("dataset_name"),
            "registry_path": str(cfg.get("registry_path")) if cfg.get("registry_path") else None,
            "methods": cfg.get("methods"),
            "output_root": str(cfg.get("output_root")) if cfg.get("output_root") else None,
            "experiment_protocol_path": str(cfg.get("experiment_protocol_path")) if cfg.get("experiment_protocol_path") else None,
        }, "PublicBenchmarkPipeline.run 入口参数")
        dataset_name = normalize_public_dataset_name(cfg.get("dataset_name"))  # 归一化数据集名称，统一大小写和别名。
        cfg["dataset_name"] = dataset_name  # 把归一化后的名称写回配置，供下游使用。

        dirs = get_standard_dirs(cfg.get("project_root"))  # 解析项目标准目录。
        registry_path = _resolve_optional_project_path(  # 解析注册表路径，支持相对路径。
            cfg.get("registry_path"),
            field_name="registry_path",
            project_root=dirs["project_root"],
        )
        protocol_path = _resolve_optional_project_path(  # 解析实验协议路径，支持相对路径。
            cfg.get("experiment_protocol_path"),
            field_name="experiment_protocol_path",
            project_root=dirs["project_root"],
        )

        registry_cfg = load_public_dataset_registry(registry_path)  # 加载数据集注册表配置。
        dataset_entry = get_dataset_entry(dataset_name, registry_cfg)  # 按名称获取数据集条目，包含路径和元信息。
        protocol_cfg = load_experiment_protocol(protocol_path)  # 加载实验协议配置。
        gate_report = normalize_public_benchmark_request(cfg, dataset_entry, protocol_cfg)  # 规范化请求是否满足协议门控。
        runtime_contract = _resolve_dataset_runtime_contract(  # 从数据集配置补齐运行合同，避免路由层和脚本层脱钩。
            cfg=cfg,
            dataset_name=dataset_name,
            project_root=dirs["project_root"],
        )
        normalized_cfg = {  # 用门控归一化后的字段构建标准化配置。
            **cfg,
            "dataset_name": gate_report["dataset_name"],  # 门控归一化后的数据集名称。
            "mode": gate_report["mode"],  # 门控归一化后的运行模式。
            "split": gate_report["split"],  # 门控归一化后的数据切分。
            "seq_ids": list(gate_report.get("seq_ids") or []),  # 门控归一化后的序列列表。
            "raw_root": runtime_contract["raw_root"],  # 若调用方未显式提供，则从数据集配置补齐。
            "field_mapping": runtime_contract["field_mapping"],  # 若调用方未显式提供，则复用冻结字段合同。
        }

        output_root = resolve_output_root(cfg, "public_benchmark_pipeline")  # 统一解析输出根目录：拒绝空白字符串/Path(".")/非 str/Path 类型，支持 ~ 展开和相对路径解析，与其他 pipeline 保持一致。

        reports_dir = output_root / "reports"  # 报告输出目录。
        audits_dir = output_root / "audits"  # 审计输出目录。
        reports_dir.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
        audits_dir.mkdir(parents=True, exist_ok=True)  # 确保审计目录存在。

        seq_ids = list(gate_report.get("seq_ids") or [])  # 从门控报告取出序列列表。
        public_scene_tasks: list[dict] = []  # 构造公开基准专用的场景任务列表。
        for idx, seq_id in enumerate(seq_ids):  # 逐个序列构造场景任务。
            public_scene_tasks.append(
                {
                    "task_id": f"public_{idx:04d}",  # 任务编号按序递增，4 位数对齐 scene_sampler 和 prepared_inputs。
                    "scene_id": f"{dataset_name}:{seq_id}",  # 场景编号由数据集名和序列名拼接。
                    "seq_id": seq_id,  # 序列编号。
                    "dataset_name": dataset_name,  # 数据集名称。
                    "axes": {  # 轴配置标记数据集来源和切分。
                        "dataset": dataset_name,
                        "split": gate_report["split"],
                    },
                }
            )

        if dataset_name == DATASET_NAME_MILUV:  # MILUV 数据集走专用流水线。
            miluv_input = {  # 构造 MiluvPipeline 的输入配置。
                **normalized_cfg,
                "output_root": output_root / DATASET_NAME_MILUV,  # MILUV 子目录。
                "scene_tasks": public_scene_tasks,  # 传入构造好的场景任务。
            }
            dataset_result = MiluvPipeline().run(miluv_input)  # 执行 MILUV 专用流水线。
        else:  # 其他数据集走通用 Prepare + Core 流水线。
            prepare_output_root = output_root / "prepare"  # prepare 子目录。
            # 公共基准烟雾路径: 显式传递 quick_full_rule 让 PreparePipeline B04 豁免单轨调通
            prepare_input = {
                **normalized_cfg,
                "output_root": prepare_output_root,
                "quick_full_rule": normalized_cfg.get(
                    "quick_full_rule",
                    "quick_smoke_scale__full_real_execution_required",
                ),
            }  # 构造 PreparePipeline 输入。
            prepare_result = PreparePipeline().run(prepare_input)  # 执行数据准备流水线。
            prepare_manifest = load_prepare_manifest(prepare_output_root)  # 加载准备阶段产出的清单。

            ground_truth_by_seq_id = load_ground_truth_by_seq_id(normalized_cfg["raw_root"], seq_ids)  # 按序列加载真值。
            source_report_by_seq_id = load_source_report_by_seq_id(  # 按序列加载数据源报告。
                normalized_cfg["raw_root"],
                prepare_manifest,
                seq_ids,
                default_source=f"{dataset_name}_prepare_bridge",  # 默认来源标识。
            )
            # 加载准备阶段事件（使用统一惰性加载器，支持 .pkl.gz 和 .json）。
            events_by_seq_id = load_prepared_events_by_seq_id(prepare_output_root, seq_ids)

            core_input = {  # 构造 CorePipeline 的输入配置。
                "scene_tasks": public_scene_tasks,  # 场景任务列表。
                "events_by_seq_id": events_by_seq_id,  # 按序列索引的事件。
                "ground_truth_by_seq_id": ground_truth_by_seq_id,  # 按序列索引的真值。
                "source_report_by_seq_id": source_report_by_seq_id,  # 按序列索引的数据源报告。
                "methods": list(normalized_cfg.get("methods") or []),  # 方法列表。
                "estimator_cfgs": normalized_cfg.get("estimator_cfgs") or {},  # 估计器配置覆盖。
                "model_cfgs": normalized_cfg.get("model_cfgs") or {},  # 模型配置覆盖。
                "output_root": output_root / "core",  # core 子目录。
                "project_root": normalized_cfg.get("project_root"),  # 项目根目录。
                # 与 MILUV 路径保持一致：CorePipeline 默认 retain_prediction_bundles=True，
                # 这里显式默认 True，避免 NTU VIRAL 路径 prediction_bundles 恒空导致
                # 18_run_public_benchmarks.py 评估交接必然抛 ValueError。
                "retain_prediction_bundles": bool(normalized_cfg.get("retain_prediction_bundles", True)),
            }
            prediction_result = CorePipeline().run(core_input)  # 执行核心推理流水线。
            dataset_result = StageResult(  # 合并 prepare 和 core 的产物。
                stage_name="public_dataset_route",  # 阶段名标记为公开数据集路由。
                artifacts=list(prepare_result.artifacts) + list(prediction_result.artifacts),  # 合并产物路径。
                metadata={  # 合并元数据，prepare 的键优先，避免被 core 同名键覆盖。
                    **prediction_result.metadata,
                    "prepare_result": prepare_result.metadata,
                },
            )

        routed_scene_tasks = list(dataset_result.metadata.get("scene_tasks") or public_scene_tasks)  # 取出实际执行的场景任务。
        prediction_bundle_count = len(routed_scene_tasks)  # 默认按场景任务数统计。
        prediction_index = dataset_result.metadata.get("prediction_index")  # 优先按落盘索引统计。
        if isinstance(prediction_index, list):  # 索引存在时，说明预测 bundle 已落盘可重建。
            prediction_bundle_count = len(prediction_index)
        elif "prediction_bundles" in dataset_result.metadata:  # 兼容旧路径。
            prediction_bundle_count = len(dataset_result.metadata["prediction_bundles"])

        report = {  # 构造基准报告。
            "dataset_name": dataset_name,  # 数据集名称。
            "dataset_entry": dataset_entry,  # 数据集注册条目。
            "method_count": len(normalized_cfg.get("methods") or []),  # 方法数量。
            "seq_ids": [task["seq_id"] for task in routed_scene_tasks],  # 实际执行的序列列表。
            "prediction_bundle_count": prediction_bundle_count,  # 预测 bundle 数量。
        }

        report_path = reports_dir / "public_benchmark_report.json"  # 报告文件路径。
        report_path.write_text(dumps_json_text(report), encoding="utf-8")  # 以严格 JSON 写出报告。
        gate_path = audits_dir / "public_protocol_gate.json"  # 协议门控审计文件路径。
        gate_path.write_text(dumps_json_text(gate_report), encoding="utf-8")  # 以严格 JSON 写出门控审计。

        metadata = dict(dataset_result.metadata)  # 复制子流水线元数据。
        metadata["scene_tasks"] = routed_scene_tasks  # 补充实际执行的场景任务。
        metadata["public_benchmark_report"] = report  # 补充基准报告。
        metadata["dataset_entry"] = dataset_entry  # 补充数据集注册条目。
        metadata["protocol_gate"] = gate_report  # 补充协议门控报告。
        return StageResult(
            stage_name="public_benchmark_pipeline",  # 阶段名。
            artifacts=list(dataset_result.artifacts) + [str(report_path), str(gate_path)],  # 合并子流水线产物和本阶段产物。
            metadata=metadata,  # 完整元数据。
        )


def run(pipeline_cfg):
    """兼容旧入口的便捷函数，直接转发给 PublicBenchmarkPipeline.run。

    Args:
        pipeline_cfg: 传给 PublicBenchmarkPipeline.run 的配置字典。

    Returns:
        PublicBenchmarkPipeline().run(...) 的原始返回值。
    """
    return PublicBenchmarkPipeline().run(pipeline_cfg)
