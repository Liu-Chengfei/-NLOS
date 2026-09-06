"""Run minimal classical baselines smoke test — E9 multi-pathway (miluv + sim)."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.common.io_utils import dumps_json_text
from liquidloc.common.prepared_inputs import (
    load_ground_truth_by_seq_id,
    load_prepared_events_by_seq_id,
    load_source_report_by_seq_id,
)
from liquidloc.dataio.adapters.event_builder import (
    build_imu_events,
    build_uwb_events,
    build_vio_events,
    merge_and_finalize_events,
)
from liquidloc.dataio.adapters.field_mapper import map_external_fields
from liquidloc.dataio.readers.miluv_reader import read_miluv_sequence
from liquidloc.pipelines.core_pipeline import CorePipeline
from liquidloc.scenarios.scene_sampler import sample_scenes

_DEFAULT_RAW_ROOT = ROOT / "tests" / "fixtures" / "datasets" / "miluv"
_DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "baseline_script_smoke"
_SMOKE_SEQ_ID = "mini_seq"
_SMOKE_SCENE_ID = "S(A3,N3,V2,K0,M0)"  # 五轴档位协议：G 已并入 K，K 轴仅 K0/K1/K3。
_BASELINE_METHODS = ("ekf",)  # 基线：仅 EKF（无 FGO）。
_MILUV_CONFIG = ROOT / "configs" / "datasets" / "miluv.yaml"
_SIM_DEFAULT_PREPARE_ROOT = Path("outputs/prepare_sim_e9_main")


def _resolve_non_empty_path(raw_value, *, flag_name, default):
    if raw_value is None:
        # 默认值可能为 None（当调用方只想检查 raw_value 是否被显式提供时），
        # 此时返回 None 而非对 None 调用 .resolve()。调用方负责 None 分支处理。
        if default is None:
            return None
        return default.resolve()
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"{flag_name} must be a non-empty path")
    return Path(value).resolve()


def _load_sim_prepared_events(prepare_root: Path, split_ids: list[str]) -> list[dict[str, Any]]:
 """加载多序列事件并进行时间戳偏移，保证合并后时间单调。"""
 from liquidloc.common.prepared_inputs import load_prepared_events_by_seq_id
 events_by_seq_id = load_prepared_events_by_seq_id(prepare_root, split_ids)
 all_events: list[dict[str, Any]] = []
 time_offset: float = 0.0
 for sid in split_ids:
     seq_events = events_by_seq_id.get(sid, [])
     if not seq_events:
         continue
     for e in seq_events:
         e = dict(e)
         e["t"] = e.get("t", 0.0) + time_offset
         all_events.append(e)
     max_t = max(e.get("t", 0.0) for e in seq_events)
     time_offset = max_t + 0.02
 return all_events


def _load_real_smoke_events(
    raw_root: Path,
    scene_id: str,
    *,
    dataset_name: str = "miluv",
    seq_id: str = _SMOKE_SEQ_ID,
) -> list[dict[str, Any]]:
    """加载事件流：miluv 走 reader+adapter 链，sim 走 prepare 输出目录。"""
    if dataset_name == "sim":
        prepare_root = raw_root if (raw_root / "prepare_manifest.json").is_file() else _SIM_DEFAULT_PREPARE_ROOT
        return _load_sim_prepared_events(prepare_root, [seq_id])
    # MILUV 路径
    dataset_cfg = dict(load_yaml_config(_MILUV_CONFIG))
    field_mapping = copy.deepcopy(dict(dataset_cfg.get("field_mapping") or {}))
    raw_bundle, _ = read_miluv_sequence(seq_id, raw_root, field_mapping)
    internal_bundle, _ = map_external_fields(raw_bundle, field_mapping)
    imu_events = build_imu_events(internal_bundle.get("imu_raw", []), scene_id, seq_id)
    uwb_events = build_uwb_events(internal_bundle.get("uwb_raw", []), scene_id, seq_id)
    vio_events = build_vio_events(internal_bundle.get("vio_raw", []), scene_id, seq_id)
    return merge_and_finalize_events([imu_events, uwb_events, vio_events])


def _select_baseline_methods(methods: list[str]) -> list[str]:
    return [m for m in methods if m in _BASELINE_METHODS]


def _find_smoke_scene(scene_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    for task in scene_tasks:
        if task.get("scene_id") == _SMOKE_SCENE_ID:
            return {
                "task_id": "scene_00",
                "scene_id": _SMOKE_SCENE_ID,
                "axes": copy.deepcopy(dict(task.get("axes") or {})),
                "seq_id": _SMOKE_SEQ_ID,
            }
    raise ValueError("configured scene surface does not include the baseline smoke scene")


def _default_smoke_task() -> dict[str, Any]:
    return {
        "task_id": "scene_00",
        "scene_id": _SMOKE_SCENE_ID,
        "axes": {"A": "A3", "N": "N3", "V": "V2", "K": "K0"},  # 五轴协议：G 并入 K。
        "seq_id": _SMOKE_SEQ_ID,
    }


def main(argv: list[str] | None = None) -> int:
    print("[07_baselines] 开始 | E9 multi-pathway", flush=True)
    parser = argparse.ArgumentParser(description="Baseline smoke — E9 multi-pathway (miluv or sim)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--split-ids", default=None, help="Comma-separated seq IDs (default: mini_seq)")
    parser.add_argument("--scene-id", default=None, help="Scene ID (default: S(A3,N3,V2,K0,M0))")
    parser.add_argument("--dataset-name", default=None, help="Dataset: miluv (default) or sim")
    parser.add_argument("--method", default=None, help="Run single method only (overrides config methods list)")
    args = parser.parse_args(argv)
    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "07_run_baselines")

    config_path = _resolve_non_empty_path(args.config, flag_name="--config", default=None)
    if config_path is None:
        print("[07_baselines] 错误: 需要 --config 指定实验配置文件", flush=True)
        return 1
    raw_root = _resolve_non_empty_path(args.raw_root, flag_name="--raw-root", default=_DEFAULT_RAW_ROOT)
    output_root = _resolve_non_empty_path(args.output_root, flag_name="--output-root", default=_DEFAULT_OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    print("[07_baselines] 加载实验配置", flush=True)
    experiment_cfg = copy.deepcopy(dict(load_yaml_config(config_path)))
    experiment_cfg["mode"] = experiment_cfg.get("mode", "quick")
    # E9 方法覆盖：--method 可单跑一个方法，不限制传统基线。
    experiment_cfg["methods"] = list(experiment_cfg.get("methods") or [])
    if args.method is not None:
        _single = args.method.strip().lower()
        # 始终覆盖为单方法，无论该方法是否已在 config methods 列表里。
        experiment_cfg["methods"] = [_single]
    print_dict(experiment_cfg, "实验配置")

    # E9 多序列 + 多数据集支持
    _ds_name = (args.dataset_name or "miluv") if args.dataset_name else "miluv"
    _split_ids_list = [s.strip() for s in (args.split_ids or _SMOKE_SEQ_ID).split(",") if s.strip()]
    _scene_id_value = args.scene_id or _SMOKE_SCENE_ID
    axes_default = {"A": "A3", "N": "N3", "V": "V0", "K": "K1"}  # 五轴协议：G 并入 K，K1=中等几何。
    # sim 数据需要走 prepare_sim 输出目录作为 raw_root
    _effective_raw_root = raw_root
    if _ds_name == "sim" and not (raw_root / "prepare_manifest.json").is_file():
        _effective_raw_root = _SIM_DEFAULT_PREPARE_ROOT

    scene_tasks = []
    for _idx, _sid in enumerate(_split_ids_list):
        scene_tasks.append({
            "task_id": f"scene_{_idx:02d}",
            "scene_id": _scene_id_value,
            "axes": copy.deepcopy(axes_default),
            "seq_id": _sid,
        })
    smoke_task = scene_tasks[0]
    print_dict({
        "scene_tasks": scene_tasks,
        "split_ids": _split_ids_list,
        "dataset_name": _ds_name,
        "raw_root": str(_effective_raw_root),
        "output_root": str(output_root),
    }, "派生路径与任务")

    # 加载真值和来源报告
    if _ds_name == "sim":
        gt_root = ROOT / "data/raw/sim_e9_main"
    else:
        gt_root = _effective_raw_root
    ground_truth_by_seq_id = load_ground_truth_by_seq_id(gt_root, _split_ids_list)
    source_report_by_seq_id = load_source_report_by_seq_id(
        _effective_raw_root,
        {"sequences": {sid: {} for sid in _split_ids_list}},
        _split_ids_list,
        default_source="baseline_script_smoke",
    )

    # 加载事件流：按 seq_id 分派，避免多任务批次下每个任务都跑合并流。
    # 仍构造一份合并的 _all_events 用作单任务调用回退，但优先传 events_by_seq_id。
    _all_events: list[dict[str, Any]] = []
    _events_by_seq_id: dict[str, list[dict[str, Any]]] = {}
    _time_offset: float = 0.0
    for _task in scene_tasks:
        _task_events = _load_real_smoke_events(
            _effective_raw_root,
            _task["scene_id"],
            dataset_name=_ds_name,
            seq_id=_task.get("seq_id", _SMOKE_SEQ_ID),
        )
        # 每个 seq_id 独立存一份，时间戳不偏移，供 events_by_seq_id 路径使用。
        _events_by_seq_id[_task["seq_id"]] = list(_task_events)
        # 合并流保留时间偏移，用作回退兼容。
        for _i, _e in enumerate(_task_events):
            _e = dict(_e)
            _e["t"] = _e.get("t", 0.0) + _time_offset
            # 跨序列边界：首个事件的 dt 需反映与前一序列末尾的时间间隔
            if _i == 0 and _all_events:
                _e["dt"] = round(_e["t"] - _all_events[-1]["t"], 6)
            _all_events.append(_e)
        if _task_events:
            _max_t = max(_e.get("t", 0.0) for _e in _all_events[-len(_task_events):])
            _time_offset = _max_t + 0.02

    print(f"[07_baselines] 运行核心流水线 | methods={experiment_cfg['methods']}", flush=True)
    result = CorePipeline().run({
        "experiment_cfg": experiment_cfg,
        "scene_tasks": scene_tasks,
        "events": _all_events,
        "events_by_seq_id": _events_by_seq_id,
        "methods": list(experiment_cfg["methods"]),
        "ground_truth_by_seq_id": ground_truth_by_seq_id,
        "source_report_by_seq_id": source_report_by_seq_id,
        "output_root": str(output_root),
    })

    print(dumps_json_text({
        "stage_name": result.stage_name,
        "artifacts": list(result.artifacts),
        "bundle_count": len(result.metadata.get("prediction_bundles") or []),
    }))
    print("[07_baselines] 完成 | 返回码=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
