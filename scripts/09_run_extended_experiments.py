"""运行最小的扩展实验入口。"""

from __future__ import annotations  # 让类型注解可以直接引用当前模块定义。

import argparse  # 解析命令行参数。
import copy  # 复制配置和任务对象，避免共享可变引用。
import json  # 输出摘要报告。
import os  # 文件系统路径判断。
import sys  # 调整导入路径。
from pathlib import Path  # 处理路径。
from typing import Any  # 标注嵌套结构。
from collections.abc import Mapping  # 阶段 12 全面审计修复 (2026-09-06): BUG-011 Mapping 未导入导致 L122 isinstance 失败.

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录。
SRC = ROOT / "src"  # 项目源码目录。
if str(SRC) not in sys.path:  # 如果源码目录还没加进来。
    sys.path.insert(0, str(SRC))  # 把源码目录插到前面。

# 在所有导入之前清除协议快照缓存，确保 YAML 修改后立即生效。
from liquidloc.protocol.scene_axis_protocol import _load_frozen_scene_axis_protocol_snapshot
# 先调用一次触发 lru_cache 缓存旧值，再清除，这样后续调用才会重新加载。
_load_frozen_scene_axis_protocol_snapshot()
_load_frozen_scene_axis_protocol_snapshot.cache_clear()

from liquidloc.common.config_utils import load_yaml_config  # 读取 YAML 配置。
from liquidloc.common.io_utils import dumps_json_text  # 严格 JSON 序列化，拒绝 NaN/Infinity。
from liquidloc.common.prepared_inputs import load_ground_truth_by_seq_id, load_prepare_manifest, load_prepared_events_by_seq_id, load_source_report_by_seq_id  # 读取真值、prepare 清单、事件和来源报告。
from liquidloc.dataio.registry.public_dataset_registry import get_dataset_entry, load_public_dataset_registry, normalize_public_dataset_name, resolve_public_eval_seq_ids  # 统一公开数据集名并读取注册表。
from liquidloc.pipelines.core_pipeline import CorePipeline  # 核心实验流水线。
from liquidloc.pipelines.prepare_pipeline import PreparePipeline  # 通用准备流水线。
from liquidloc.pipelines.public_benchmark_pipeline import PublicBenchmarkPipeline  # 公开 benchmark 流水线。
from liquidloc.protocol.experiment_gates import _PUBLIC_BENCHMARK_FROZEN_EVAL_SPLIT, get_public_benchmark_allowed_datasets  # 读取协议定义的官方公开 benchmark 白名单和冻结评测分割名。
from liquidloc.scenarios.scene_sampler import sample_scenes  # 采样场景任务。


def _build_sim_scene_id_by_seq(
    seq_ids: list[str],
    raw_root: Path,
    experiment_cfg: dict[str, Any] | None = None,
) -> dict[str, str]:
    """从 sim_meta.json 的 axes_override 生成 scene_id_by_seq。

    scene_id 格式必须与 encode_scene / decode_scene 的 S(A,N,V,K,M) 五轴格式一致，
    禁止包含 G 轴（几何条件由 K 轴自身编码，不单独出现在 scene_code 中）。
    缺失时回退主表欠定默认 S(A0,N0,V0,K3,M0)（K 轴协议仅 K0/K1/K3，锚数全档固定 4）。

    BUG-007 修复 (2026-09-06 §10 阶段 12 审计): 早期物化 sim_meta.json 的 axes_override
    缺 M 字段. 这里接受 experiment_cfg['frozen_axes'] 覆盖, 保证 e9 yaml M:M1
    能在 scene_id 编码里出现, 让 _resolve_scene_parameters 走 5 轴路径.

    参数
    ----------
    seq_ids : list[str]
        需要准备或运行的序列 ID 列表。
    raw_root : Path
        仿真原始数据根目录。

    返回
    -------
    dict[str, str]
        序列 ID 到场景 ID 的映射。
    """
    scene_id_by_seq: dict[str, str] = {}
    frozen_axes = (experiment_cfg or {}).get("frozen_axes") or {}
    # e9 yaml 的 M 轴通常为单值 (M:M1), 但允许 [M1, M2] 列表形式, 取第一项
    yaml_m_override = None
    if isinstance(frozen_axes.get("M"), list) and frozen_axes["M"]:
        yaml_m_override = str(frozen_axes["M"][0]).strip()
    elif isinstance(frozen_axes.get("M"), str):
        yaml_m_override = str(frozen_axes["M"]).strip()
    for seq_id in seq_ids:
        # BUG-005 链路: 09 seq_id='seed0__sim_curve_01', 实际路径 seed0/sim_curve_01/sim_meta.json
        resolved_seq_id = seq_id.replace('__', '/') if '__' in seq_id else seq_id
        sim_meta_path = raw_root / resolved_seq_id / "sim_meta.json"
        if sim_meta_path.is_file():
            try:
                sim_meta = json.loads(sim_meta_path.read_text(encoding="utf-8"))
                axes_override = sim_meta.get("axes_override") or {}
                if axes_override:
                    # encode_scene 顺序：五轴 S(A,N,V,K,M)，不含 G。
                    # G 轴编码进 K 轴自身（G1→K3，G2→K4 等），不在 scene_code 中独立出现。
                    axis_order = ["A", "N", "V", "K", "M"]
                    parts = [axes_override.get(ax, _AXIS_DEFAULTS.get(ax, ax + "0")) for ax in axis_order]
                    if parts:
                        scene_id_by_seq[seq_id] = f"S({','.join(parts)})"
                        continue
            except Exception:
                pass  # 解析失败时回退到主表欠定默认
        # 缺 axes_override 或解析失败时回退. e9 yaml 的 M 覆盖 (e9 frozen_axes.M:M1).
        if yaml_m_override is not None:
            scene_id_by_seq[seq_id] = f"S(A0,N0,V0,K3,{yaml_m_override})"
        else:
            scene_id_by_seq[seq_id] = "S(A0,N0,V0,K3,M0)"  # 五轴档位协议：主表欠定默认 K3（K 轴仅 K0/K1/K3）
    return scene_id_by_seq


def _build_sim_axes_by_seq(
    seq_ids: list[str],
    raw_root: Path,
    experiment_cfg: dict[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    """从 sim_meta.json 的 axes_override 为每个序列收集实际的场景轴。

    这对于从 supplementary_public_route 运行的实验至关重要：
    core_pipeline._apply_scene_task 依赖 task['axes'] 中的 G/K 轴来构建
    anchor_layout；若任务['axes'] 只含 dataset/split（不完整），则
    anchor_layout 为 None，导致后续 fused_estimator 查不到 anchor_id。

    参数
    ----------
    seq_ids : list[str]
        序列 ID 列表。
    raw_root : Path
        仿真数据根目录。

    返回
    -------
    dict[str, dict[str, str]]
        seq_id -> {A: ..., N: ..., V: ..., G: ..., K: ..., M: ...} 的映射。
        若 sim_meta.json 不存在或 axes_override 不完整，回退到主表默认值。
    """
    yaml_m_override = None
    if isinstance(experiment_cfg, Mapping):
        frozen_axes = (experiment_cfg or {}).get("frozen_axes") or {}
        if isinstance(frozen_axes.get("M"), list) and frozen_axes["M"]:
            yaml_m_override = str(frozen_axes["M"][0]).strip()
        elif isinstance(frozen_axes.get("M"), str):
            yaml_m_override = str(frozen_axes["M"]).strip()
    axes_by_seq: dict[str, dict[str, str]] = {}
    for seq_id in seq_ids:
        # BUG-005 链路修复: 09 脚本 seq_id='seed0__sim_curve_01', 但 sim_meta.json 在
        # raw_root / 'seed0' / 'sim_curve_01' / 'sim_meta.json' (嵌套 seed/seq 结构).
        # read_sim_sequence 已接受 __ 并还原 /; 这里对 _build_sim_axes_by_seq 同样处理.
        resolved_seq_id = seq_id.replace('__', '/') if '__' in seq_id else seq_id
        sim_meta_path = raw_root / resolved_seq_id / "sim_meta.json"
        if sim_meta_path.is_file():
            try:
                sim_meta = json.loads(sim_meta_path.read_text(encoding="utf-8"))
                axes_override = sim_meta.get("axes_override") or {}
                if axes_override:
                    # BUG-007 修复 (2026-09-06 §10 阶段 12 审计): 早期物化 sim_meta.json 的
                    # axes_override 只有 4 字段 (A/N/V/K), 缺 M. 5 轴退化 (A/N/V/K/M) 协议要求 axes 必
                    # 须齐全才能调 attach_scene_parameters 展开 flat 字段, 否则 _apply_scene_task
                    # 的 M 轴 block 不会执行, 缺失注入跳过. 这里补 M1 (e9 主档 UWB 5% 成簇丢包).
                    if "M" not in axes_override and yaml_m_override is not None:
                        axes_override = dict(axes_override)
                        axes_override["M"] = yaml_m_override
                    elif "M" not in axes_override:
                        axes_override = dict(axes_override)
                        axes_override["M"] = "M1"
                    axes_by_seq[seq_id] = axes_override
                    continue
            except Exception:
                pass
        # 回退到主表欠定默认（S(A0,N0,V0,K3,M0) 与 scene_id_by_seq 的 fallback 一致）.
        # e9 yaml frozen_axes.M 覆盖 (M:M1) 走主表 + yaml 注入.
        fallback_m = yaml_m_override or "M0"
        axes_by_seq[seq_id] = {"A": "A0", "N": "N0", "V": "V0", "K": "K3", "M": fallback_m}
    return axes_by_seq

_AXIS_DEFAULTS = {"A": "A0", "N": "N0", "V": "V0", "K": "K3", "M": "M0"}  # 五轴默认值，scene_id 不含 G（K 轴协议仅 K0/K1/K3）。
_DEFAULT_CONFIG = None  # 实验配置必须由 --config 显式传入（不再隐式绑定默认实验）。
_DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "extended_script_smoke"  # 默认输出目录。
_DEFAULT_FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "datasets"  # 默认 fixture 数据目录。
_DEFAULT_CORE_SEQ_ID = "mini_seq"  # 默认冒烟序列 ID。
_OFFICIAL_PUBLIC_DATASETS = None


_UNIMPLEMENTED_PRIMARY_AXES = {"modality_recovery_profile", "phase_switch_profile"}


def _get_official_public_datasets():
    global _OFFICIAL_PUBLIC_DATASETS
    if _OFFICIAL_PUBLIC_DATASETS is None:
        _OFFICIAL_PUBLIC_DATASETS = frozenset(get_public_benchmark_allowed_datasets())
    return _OFFICIAL_PUBLIC_DATASETS
# 主表核心实验面：EKF 基线 + LSTM/Transformer/Liquid 神经网络增强 EKF（不含 FGO）。
_ALL_MODEL_CORE_SURFACE = {"ekf", "lstm_ekf", "transformer_ekf", "liquid_ekf"}


def _resolve_non_empty_string(raw_value: str | None, *, flag_name: str, allow_none: bool = False) -> str | None:
    """把命令行字符串参数规范成非空字符串。"""
    if raw_value is None:  # 没传值时看是否允许空。
        if allow_none:  # 允许空就直接返回 None。
            return None
        raise ValueError(f"{flag_name} must be a non-empty string")  # 不允许空就报错。
    value = str(raw_value).strip()  # 先去掉首尾空白。
    if not value:  # 空字符串不合法。
        raise ValueError(f"{flag_name} must be a non-empty string")  # 明确指出参数有问题。
    return value  # 返回规范化后的字符串。


def _resolve_path(raw_value: str | None, *, flag_name: str, default: Path) -> Path:
    """把命令行路径参数规范成绝对路径。"""
    if raw_value is None:  # 没传就用默认值。
        return default.resolve()  # 默认路径统一转绝对路径。
    value = str(raw_value).strip()  # 去掉空白。
    if not value:  # 空字符串不合法。
        raise ValueError(f"{flag_name} must be a non-empty string")  # 说明哪个参数有问题。
    path = Path(value)  # 转成路径对象。
    if not path.is_absolute():  # 相对路径按仓库根目录解释。
        path = (ROOT / path).resolve()  # 转成绝对路径。
    return path.resolve()  # 返回最终路径。


def _default_core_events(scene_id: str) -> list[dict[str, Any]]:
    """构造核心实验使用的固定事件序列。"""
    return [  # 返回三条事件，分别覆盖 UWB、VIO 和 IMU。
        {  # 第一条是 UWB。
            "t": 0.028,  # 事件时间戳。
            "dt": 0.0,  # 相对上一条的时间差。
            "modality": "uwb",  # 模态名称。
            "meta": {"scene_id": scene_id, "seq_id": _DEFAULT_CORE_SEQ_ID},  # 场景和序列信息。
            "imu_payload": None,  # 不携带 IMU 数据。
            "uwb_payload": {"anchor_id": 0, "range": 2.0, "valid": True, "quality": 0.95},  # UWB 原始数据。
            "vio_payload": None,  # 不携带 VIO 数据。
        },  # UWB 事件结束。
        {  # 第二条是 VIO。
            "t": 0.038000000000000006,  # 事件时间戳。
            "dt": 0.010000000000000005,  # 相对上一条的时间差。
            "modality": "vio",  # 模态名称。
            "meta": {"scene_id": scene_id, "seq_id": _DEFAULT_CORE_SEQ_ID},  # 场景和序列信息。
            "imu_payload": None,  # 不携带 IMU 数据。
            "uwb_payload": None,  # 不携带 UWB 数据。
            "vio_payload": {  # VIO 原始数据。
                "dx": 0.03,  # x 位移增量。
                "dy": 0.0,  # y 位移增量。
                "dyaw": 0.0,  # 航向增量。
                "quality": 0.85,  # 视觉质量分数。
                "tracked_features": 60,  # 跟踪特征数。
                "reproj_err": 0.4,  # 重投影误差。
            },  # VIO 数据结束。
        },  # VIO 事件结束。
        {  # 第三条是 IMU。
            "t": 0.075,  # 事件时间戳。
            "dt": 0.037,  # 相对上一条的时间差。
            "modality": "imu",  # 模态名称。
            "meta": {"scene_id": scene_id, "seq_id": _DEFAULT_CORE_SEQ_ID},  # 场景和序列信息。
            "imu_payload": {"ax": 0.1, "ay": 0.0, "gz": 0.01},  # IMU 原始数据。
            "uwb_payload": None,  # 不携带 UWB 数据。
            "vio_payload": None,  # 不携带 VIO 数据。
        },  # IMU 事件结束。
    ]  # 事件列表结束。


def _resolve_core_surface_methods(methods: list[str]) -> list[str]:
    """规范核心实验里要跑的方法列表。"""
    normalized_methods = [str(method_name).strip() for method_name in methods if str(method_name).strip()]  # 去空白并过滤空项。
    if not normalized_methods:  # 没有方法就不能继续。
        raise ValueError("core experiment route requires a non-empty methods list")  # 明确报错。
    return normalized_methods  # 返回规范化方法列表。


def _resolve_public_dataset_name(experiment_cfg: dict[str, Any], raw_dataset_name: str | None) -> str:
    """确定公开 benchmark 使用的数据集名。"""
    # 优先：命令行 > frozen_axes.dataset > experiment_cfg.dataset_name。
    cfg_dataset = experiment_cfg.get("dataset_name")
    frozen_dataset = (experiment_cfg.get("frozen_axes") or {}).get("dataset")
    fallback = frozen_dataset if frozen_dataset else cfg_dataset
    dataset_name = normalize_public_dataset_name(_resolve_non_empty_string(
        raw_dataset_name if raw_dataset_name is not None else fallback,
        flag_name="--dataset-name",
    ))
    expected_dataset_name = str(experiment_cfg.get("dataset_name") or "").strip().lower()
    experiment_id = str(experiment_cfg.get("experiment_id") or "").strip()
    if expected_dataset_name and dataset_name != expected_dataset_name:
        raise ValueError(f"{experiment_id} requires dataset_name={expected_dataset_name}")
    registry_cfg = load_public_dataset_registry()
    try:
        get_dataset_entry(dataset_name, registry_cfg)
    except KeyError as exc:
        available = ", ".join(sorted((registry_cfg.get("datasets") or {}).keys()))
        raise ValueError(f"registered public dataset required; available: {available}") from exc
    return dataset_name


def _resolve_public_seq_ids(
    *,
    dataset_name: str,
    experiment_cfg: dict[str, Any],
    raw_seq_ids: list[str] | None,
    raw_root: Path | None = None,
) -> list[str]:
    """规范公开 benchmark 需要跑的序列列表。

    优先级（与 B 协议 §0.2 一致）：
    1. 命令行 --seq-ids (raw_seq_ids)
    2. experiment_cfg['seq_ids'] (非空)
    3. public_dataset_registry.datasets[<name>].frozen_public_eval_seq_ids (非空)
    4. sim 数据集专属回退：从 raw_root 自动扫描所有 seed/*/*/ 序列（解决 2026-09-06
       第 10 阶段发现：sim 真实物化目录 data/raw/sim_e9_main/seed*/<seq>/
       有 600 文件，但 sim registry 的 frozen_public_eval_seq_ids=null, e9 yaml 的
       seq_ids=null, 09 脚本会 raise ValueError. 现在 sim + frozen=null + raw_root 存在
       时自动从 raw_root/seed*/*/ 扫出来.)
    5. 仍找不到 → raise ValueError
    """
    if raw_seq_ids is not None:  # 优先级 1: 命令行 --seq-ids
        normalized = [str(seq_id).strip() for seq_id in raw_seq_ids if str(seq_id).strip()]  # 去空白。
        if not normalized:  # 空列表不允许。
            raise ValueError("--seq-ids must be a non-empty list of non-empty strings")
        return normalized
    cfg_seq_ids = experiment_cfg.get("seq_ids")  # 优先级 2: yaml 配置
    if cfg_seq_ids:
        normalized = [str(seq_id).strip() for seq_id in cfg_seq_ids if str(seq_id).strip()]
        if normalized:
            return normalized
    # 优先级 3+4: 注册表 frozen + sim 自动扫
    try:
        resolved = resolve_public_eval_seq_ids(
            dataset_name,
            experiment_cfg,
            str(experiment_cfg.get("mode") or "quick"),
        )
        if resolved:
            return resolved
    except (ValueError, KeyError):
        pass  # sim 数据集 frozen_public_eval_seq_ids=null 时会 raise, 落入优先级 4
    # 优先级 4: sim 自动从 raw_root 扫
    if dataset_name == "sim" and raw_root is not None and raw_root.is_dir():
        sim_seq_ids: list[str] = []
        for seed_dir in sorted(raw_root.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                continue
            for seq_dir in sorted(seed_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                # BUG-010 修复 (2026-09-06 §10 阶段 12 审计): sim 序列名采用嵌套路径格式
                # seed0/sim_curve_01 (正斜杠). 之前错误地使用 '__' 双下划线 (为绕过
                # validate_path_component), 但 build_manifests 生成的 record.seq_id
                # 也用正斜杠, 两者不一致导致 PreparePipeline 报 'missing from dataset manifest'.
                # prepare_pipeline 已在 sim 路径下跳过 validate_path_component, 这里
                # 直接用正斜杠让两边一致.
                sim_seq_ids.append(f"{seed_dir.name}/{seq_dir.name}")
        if sim_seq_ids:
            return sim_seq_ids
    raise ValueError(
        f"public_sequence_category requires explicit seq_ids in experiment_cfg "
        f"or frozen_public_eval_seq_ids in the public dataset registry or "
        f"raw_root auto-scan for sim dataset (dataset_name={dataset_name!r}, raw_root={raw_root})"
    )


def _build_public_scene_tasks(
    dataset_name: str,
    seq_ids: list[str],
    split_name: str,
    frozen_axes: dict[str, Any] | None = None,
    axes_by_seq: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """为公开或补充公开路线构造稳定场景任务。

    参数
    ----------
    dataset_name : str
        数据集标识（用于 scene_id 前缀）。
    seq_ids : list[str]
        序列 ID 列表。
    split_name : str
        数据集划分名称（如 "public_benchmark"）。
    frozen_axes : dict | None
        实验配置中固定的场景轴配置（A/N/V/G/K/M），来自
        experiment_cfg['frozen_axes']。这些轴会合并到每个任务的 axes
        字段中，确保 downstream _apply_scene_task 能基于 G/K 轴构建
        正确的 anchor_layout。注意：只挑出合法的场景轴（A/N/V/G/K/M），
        dataset / split 等非场景键不进入 axes 字典，否则
        attach_scene_parameters 会报 "Unknown axis"。
    axes_by_seq : dict[str, dict[str, str]] | None
        每个序列的实际场景轴（来自 sim_meta.json 的 axes_override）。
        若提供，则其优先级高于 frozen_axes —— 这对于 sim_e9 这类
        数据集至关重要：sim_e9 实际为 K3，但 yaml 默认可能为其他档，
        anchor_layout 与实际锚点不匹配时 NIS 爆炸触发 §19.1 拒识。

    返回
    -------
    list[dict[str, Any]]
    任务列表，每个任务的 axes 字段只含合法场景轴（A/N/V/K/M），
    dataset / split 等元数据放在 task 顶层。
    """
    tasks: list[dict[str, Any]] = []
    for index, seq_id in enumerate(seq_ids):
        # 优先用 axes_by_seq（实际数据集的轴），其次回退到 frozen_axes（实验配置）。
        seq_axes: dict[str, str] = {}
        if axes_by_seq and seq_id in axes_by_seq:
            seq_axes = dict(axes_by_seq[seq_id])
        elif frozen_axes:
            for axis_name, axis_value in dict(frozen_axes).items():
                if axis_name in ("A", "N", "V", "K", "M"):
                    seq_axes[axis_name] = axis_value
        # 只挑合法的五场景轴（A/N/V/K/M），G 轴已并入 K。
        clean_axes = {k: v for k, v in seq_axes.items() if k in ("A", "N", "V", "K", "M")}
        tasks.append({
            "task_id": f"public_{index:02d}",
            "scene_id": f"{dataset_name}:{seq_id}",
            "seq_id": seq_id,
            "dataset_name": dataset_name,
            "axes": clean_axes,
        })
    return tasks


def _run_supplementary_public_route(
    *,
    experiment_cfg: dict[str, Any],
    dataset_name: str,
    raw_root: Path,
    seq_ids: list[str],
    output_root: Path,
) -> dict[str, Any]:
    """运行已注册但不在官方公开 benchmark 面上的补充公开路线。"""
    dataset_cfg = dict(load_yaml_config(ROOT / "configs" / "datasets" / f"{dataset_name}.yaml"))
    field_mapping = dict(dataset_cfg.get("field_mapping") or {})
    split_name = str((experiment_cfg.get("frozen_axes") or {}).get("split") or _PUBLIC_BENCHMARK_FROZEN_EVAL_SPLIT)
    prepare_output_root = output_root / "prepare"

    # 仿真数据集需要从 sim_meta.json 构建 scene_id_by_seq；其他数据集可省略。
    scene_id_by_seq = None
    if dataset_name == "sim":
        scene_id_by_seq = _build_sim_scene_id_by_seq(seq_ids, raw_root, experiment_cfg=experiment_cfg)

    # 读取 registry 中的仿真数据合同参数（K 档 / 锚点数），用于 prepare 阶段强校验。
    registry_cfg = load_public_dataset_registry()
    registry_entry = get_dataset_entry(dataset_name, registry_cfg)
    allowed_k_levels = registry_entry.get("allowed_k_levels")
    allowed_anchor_counts = registry_entry.get("allowed_anchor_counts")

    prepare_result = PreparePipeline().run(
        {
            "dataset_name": dataset_name,
            "raw_root": str(raw_root),
            "seq_ids": seq_ids,
            "field_mapping": field_mapping,
            "output_root": str(prepare_output_root),
            # 阶段 12 全面审计修复 (2026-09-06 §10): e9 yaml 声明 quick_full_rule='...'
            # 让 PreparePipeline 跳过 B04 ≥20 硬门 (smoke 模式). 真实论文级 10 seed × 100 trajs
            # 移除此字段即恢复硬门. 同时让 e9 quick 模式能跑 < 20 seqs 测试.
            "quick_full_rule": experiment_cfg.get("quick_full_rule"),
            **({"scene_id_by_seq": scene_id_by_seq} if scene_id_by_seq else {}),
            **({"allowed_k_levels": allowed_k_levels} if allowed_k_levels else {}),
            **({"allowed_anchor_counts": allowed_anchor_counts} if allowed_anchor_counts else {}),
        }
    )
    prepare_manifest = load_prepare_manifest(prepare_output_root)
    # 为每个任务注入实际的场景轴：sim_e9 等仿真数据集的 scene_id（sim_curve_01_seed0）
    # 不符合 S(A,N,V,G,K) 格式，core_pipeline 无法从 scene_id 解码出轴信息；
    # 必须从 sim_meta.json 的 axes_override 注入实际轴（如 K3），
    # 否则 anchor_layout 与真实锚点数不匹配，NIS 会爆炸触发 §19.1 永久拒识。
    axes_by_seq = None
    if dataset_name == "sim":
        axes_by_seq = _build_sim_axes_by_seq(seq_ids, raw_root, experiment_cfg=experiment_cfg)
    scene_tasks = _build_public_scene_tasks(
        dataset_name,
        seq_ids,
        split_name,
        frozen_axes=experiment_cfg.get("frozen_axes"),
        axes_by_seq=axes_by_seq,
    )
    events_by_seq_id = load_prepared_events_by_seq_id(prepare_output_root, seq_ids)  # 用集中函数读取事件文件，不再硬编码文件名模式。
    ground_truth_by_seq_id = load_ground_truth_by_seq_id(raw_root, seq_ids)
    source_report_by_seq_id = load_source_report_by_seq_id(
        raw_root,
        prepare_manifest,
        seq_ids,
        default_source=f"{dataset_name}_supplementary_public_route",
    )
    core_result = CorePipeline().run(
        {
            "scene_tasks": scene_tasks,
            "events_by_seq_id": events_by_seq_id,
            "ground_truth_by_seq_id": ground_truth_by_seq_id,
            "source_report_by_seq_id": source_report_by_seq_id,
            "methods": list(experiment_cfg.get("methods") or []),
            "output_root": str(output_root / "core"),
        }
    )
    _ensure_requested_methods_present(list(experiment_cfg.get("methods") or []), core_result.metadata)
    return {
        "route": "public_supplementary",
        "stage_name": core_result.stage_name,
        "artifacts": list(prepare_result.artifacts) + list(core_result.artifacts),
        "bundle_count": len(core_result.metadata.get("prediction_bundles") or []),
    }


def _resolve_core_scene_tasks(experiment_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    _validate_extended_route_contract(experiment_cfg)  # 进入 scene_sampler 之前先拦住未实现的扩展主轴。
    """根据实验配置构造核心场景任务列表。

    调用 sample_scenes 前先剥离 frozen_axes 中的非场景轴键（dataset/split 等），
    避免 scene_sampler 将其误作场景轴并覆盖方法的默认参数，
    导致 scene_id 与配置意图不符（Fix for HIGH-24: e9 config 的 dataset: sim
    在 frozen_axes 中被 sample_scenes 忽略，从而回退到 A0/N0/V0/K3，
    但测试期望 S(A2,N2,V2,K3)）。
    """
    cfg = copy.deepcopy(experiment_cfg)
    frozen = cfg.get("frozen_axes") or {}
    scene_axis_keys = {"A", "N", "V", "G", "K", "M"}
    cfg["frozen_axes"] = {k: v for k, v in frozen.items() if k in scene_axis_keys}
    return copy.deepcopy(sample_scenes(cfg))  # 直接从采样器拿任务并复制一份。


def _validate_extended_route_contract(experiment_cfg: dict[str, Any], *, config_path: Path | None = None) -> None:
    """Fail fast when the config declares an extended-experiment profile with no executable route."""
    primary_axis = str(experiment_cfg.get("primary_axis") or "").strip()
    if primary_axis not in _UNIMPLEMENTED_PRIMARY_AXES:
        return
    experiment_id = str(experiment_cfg.get("experiment_id") or "").strip()
    if not experiment_id and config_path is not None:
        experiment_id = config_path.stem
    if not experiment_id:
        experiment_id = "extended_experiment"
    raise ValueError(
        f"{experiment_id} declares primary_axis={primary_axis}, but the current extended experiment route has no "
        "event-level execution contract for that profile"
    )


def _build_events_by_scene_id(scene_tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """把每个场景任务映射到固定事件序列。"""
    return {  # 返回场景到事件列表的映射。
        str(task["scene_id"]): _default_core_events(str(task["scene_id"]))  # 每个场景都用固定冒烟事件。
        for task in scene_tasks  # 遍历所有场景任务。
    }  # 映射结束。


def _load_default_geometry_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    """加载默认几何输入，用于核心路线冒烟测试。"""
    raw_root = (_DEFAULT_FIXTURE_ROOT / "miluv").resolve()  # 默认 fixture 数据目录。
    ground_truth_by_seq_id = load_ground_truth_by_seq_id(raw_root, [_DEFAULT_CORE_SEQ_ID])  # 读取默认序列真值。
    source_report_by_seq_id = load_source_report_by_seq_id(  # 读取默认来源报告。
        raw_root,  # 原始数据目录。
        {"sequences": {_DEFAULT_CORE_SEQ_ID: {}}},  # 最小序列索引结构。
        [_DEFAULT_CORE_SEQ_ID],  # 要读取的序列 ID。
        default_source="extended_script_smoke",  # 默认来源标签。
    )  # 来源报告读取结束。
    return ground_truth_by_seq_id, source_report_by_seq_id  # 返回真值和来源报告。


def _ensure_requested_methods_present(requested_methods: list[str], result_metadata: dict[str, Any]) -> None:
    """检查流水线返回的方法列表没有丢失请求的方法。"""
    prediction_bundles = list(result_metadata.get("prediction_bundles") or [])  # 取出预测 bundle 列表。
    returned_methods = {  # 收集实际返回的方法名集合。
        str(bundle.get("method_name")).strip()  # 提取方法名。
        for bundle in prediction_bundles  # 遍历所有 bundle。
        if str(bundle.get("method_name") or "").strip()  # 过滤空方法名。
    }  # 方法集合结束。
    if not returned_methods:  # 如果没有返回方法名，就不做丢失检查。
        return  # 直接结束。
    missing_methods = sorted(method_name for method_name in requested_methods if method_name not in returned_methods)  # 找出缺失方法。
    if missing_methods:  # 如果有缺失。
        raise ValueError("requested methods disappeared from prediction surface: " + ", ".join(missing_methods))  # 直接报错。


def _run_core_route(
    *,
    experiment_cfg: dict[str, Any],  # 核心实验配置。
    output_root: Path,  # 输出目录。
) -> dict[str, Any]:
    """运行核心实验路线。"""
    methods = _resolve_core_surface_methods(list(experiment_cfg.get("methods") or []))  # 规范方法列表。
    scene_tasks = _resolve_core_scene_tasks(experiment_cfg)  # 生成场景任务。
    ground_truth_by_seq_id, source_report_by_seq_id = _load_default_geometry_inputs()  # 加载默认几何输入。
    payload: dict[str, Any] = {  # 组装核心流水线输入。
        "experiment_cfg": experiment_cfg,  # 实验配置。
        "scene_tasks": scene_tasks,  # 场景任务。
        "methods": methods,  # 方法列表。
        "output_root": str(output_root),  # 输出目录。
        "ground_truth_by_seq_id": ground_truth_by_seq_id,  # 真值数据。
        "source_report_by_seq_id": source_report_by_seq_id,  # 来源报告。
    }  # 核心输入结束。
    payload["events_by_scene_id"] = _build_events_by_scene_id(scene_tasks)  # 组装场景到事件映射。

    result = CorePipeline().run(payload)  # 执行核心流水线。
    _ensure_requested_methods_present(methods, result.metadata)  # 检查输出没有丢方法。
    return {  # 返回摘要。
        "route": "core",  # 路由标识。
        "stage_name": result.stage_name,  # 阶段名称。
        "artifacts": list(result.artifacts),  # 产物列表。
        "bundle_count": len(result.metadata.get("prediction_bundles") or []),  # 预测 bundle 数量。
    }  # 摘要结束。


def _run_public_route(
    *,
    experiment_cfg: dict[str, Any],  # 实验配置。
    output_root: Path,  # 输出目录。
    dataset_name_arg: str | None,  # 命令行数据集名。
    raw_root_arg: str | None,  # 命令行原始数据目录。
    seq_ids_arg: list[str] | None,  # 命令行序列列表。
) -> dict[str, Any]:
    """运行公开 benchmark 路线。"""
    dataset_name = _resolve_public_dataset_name(experiment_cfg, dataset_name_arg)  # 解析数据集名。
    # Prefer --raw-root, then registry landed_raw_root, else fixture fallback.
    registry_cfg = load_public_dataset_registry()
    dataset_entry = get_dataset_entry(dataset_name, registry_cfg)
    landed_root = dataset_entry.get("landed_raw_root")
    fallback_default = (_DEFAULT_FIXTURE_ROOT / dataset_name)
    if landed_root:
        fallback_default = ROOT / landed_root if not os.path.isabs(landed_root) else Path(landed_root)
    raw_root = _resolve_path(  # 解析原始数据目录。
        raw_root_arg,  # 命令行原始数据目录。
        flag_name="--raw-root",  # 参数名用于报错。
        default=fallback_default,  # 使用 registry landed_raw_root 作为默认。
    )  # 原始数据目录解析结束。
    # 第 8 阶段修复 HIGH-21: experiment_cfg['raw_root'] 优先级最高 (yaml 配置), 命令行第二, registry 第三.
    cfg_raw_root = experiment_cfg.get('raw_root')
    if cfg_raw_root:
        raw_root = _resolve_path(
            cfg_raw_root, flag_name='experiment_cfg.raw_root', default=raw_root,
        )
    seq_ids = _resolve_public_seq_ids(  # 解析序列列表。
        dataset_name=dataset_name,
        experiment_cfg=experiment_cfg,
        raw_seq_ids=seq_ids_arg,
        raw_root=raw_root,  # 第 10 阶段修复: sim 数据集自动扫描 raw_root 序列
    )
    if dataset_name not in _get_official_public_datasets():  # 补充公开路线不走官方 public benchmark 门控。
        return _run_supplementary_public_route(
            experiment_cfg=experiment_cfg,
            dataset_name=dataset_name,
            raw_root=raw_root,
            seq_ids=seq_ids,
            output_root=output_root,
        )
    payload = {  # 组装公开 benchmark 输入。
        "dataset_name": dataset_name,  # 数据集名。
        "raw_root": str(raw_root),  # 原始数据目录。
        "seq_ids": seq_ids,  # 序列列表。
        "methods": list(experiment_cfg.get("methods") or []),  # 方法列表。
        "mode": experiment_cfg.get("mode", "quick"),  # 运行模式。
        "output_root": str(output_root),  # 输出目录。
        "field_mapping": dict(load_yaml_config(ROOT / "configs" / "datasets" / f"{dataset_name}.yaml").get("field_mapping") or {}),  # 字段映射。
    }  # 输入组装结束。
    result = PublicBenchmarkPipeline().run(payload)  # 执行公开 benchmark 流水线。
    return {  # 返回摘要。
        "route": "public",  # 路由标识。
        "stage_name": result.stage_name,  # 阶段名称。
        "artifacts": list(result.artifacts),  # 产物列表。
        "bundle_count": len(result.metadata.get("prediction_bundles") or []),  # 预测 bundle 数量。
    }  # 摘要结束。


def main(argv: list[str] | None = None) -> int:
    """脚本主入口，按实验类型选择核心或公开路线。"""
    print("[09_ext_exp] 开始 | awaiting args", flush=True)
    parser = argparse.ArgumentParser(description="Run minimal extended experiments")  # 创建参数解析器。
    parser.add_argument("--config", default=None)  # 实验配置路径（必须显式指定）。
    parser.add_argument("--output-root", default=None)  # 输出目录。
    parser.add_argument("--dataset-name", default=None)  # 公开 benchmark 的数据集名。
    parser.add_argument("--raw-root", default=None)  # 公开 benchmark 的原始数据目录。
    parser.add_argument("--seq-ids", nargs="*", default=None)  # 公开 benchmark 的序列列表。
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")  # 运行模式。
    args = parser.parse_args(argv)  # 解析命令行参数。
    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "09_run_extended_experiments")

    config_path = args.config
    if not config_path:
        print("[09_ext_exp] 错误: 需要 --config 指定实验配置文件", flush=True)
        return 1
    output_root = _resolve_path(args.output_root, flag_name="--output-root", default=_DEFAULT_OUTPUT_ROOT)  # 解析输出路径。
    output_root.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    experiment_cfg = copy.deepcopy(dict(load_yaml_config(config_path)))  # 读取并复制实验配置。
    experiment_cfg["mode"] = args.mode  # 覆盖运行模式。
    print_dict(experiment_cfg, "实验配置")
    primary_axis = str(experiment_cfg.get("primary_axis") or "").strip()  # 读取主轴类型。
    print_dict(
        {
            "primary_axis": primary_axis,
            "config_path": str(config_path),
            "output_root": str(output_root),
            "dataset_name": args.dataset_name,
            "raw_root": str(args.raw_root) if args.raw_root else None,
            "seq_ids": args.seq_ids,
            "mode": args.mode,
        },
        "派生参数",
    )
    if primary_axis == "public_sequence_category":  # 如果这是公开数据集路线。
        print(f"[09_ext_exp] 运行公开数据集路径 | dataset_name_arg={args.dataset_name}", flush=True)
        summary = _run_public_route(  # 运行公开 benchmark 路线。
            experiment_cfg=experiment_cfg,  # 实验配置。
            output_root=output_root,  # 输出目录。
            dataset_name_arg=args.dataset_name,  # 数据集名参数。
            raw_root_arg=args.raw_root,  # 原始数据目录参数。
            seq_ids_arg=args.seq_ids,  # 序列列表参数。
        )  # 公开路线结束。
    else:  # 否则走核心路线。
        print(f"[09_ext_exp] 运行核心路径 | experiment_id={experiment_cfg.get('experiment_id')}", flush=True)
        summary = _run_core_route(  # 运行核心实验路线。
            experiment_cfg=experiment_cfg,  # 实验配置。
            output_root=output_root,  # 输出目录。
        )  # 核心路线结束。

    print(dumps_json_text(summary))  # 打印严格 JSON 摘要。
    print("[09_ext_exp] 完成 | 返回码=0", flush=True)
    return 0  # 正常退出。


if __name__ == "__main__":  # 直接执行脚本时走这里。
    raise SystemExit(main())  # 用 main 返回码结束进程。
