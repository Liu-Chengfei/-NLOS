"""Run the minimal core experiment orchestration smoke path."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.common.io_utils import dumps_json_text
from liquidloc.pipelines.core_pipeline import CorePipeline
from liquidloc.protocol.scene_schema import decode_scene


def _default_events() -> list[dict]:
    """Construct a tiny event sequence for core-pipeline smoke tests."""
    return [
        {
            "t": 0.028,
            "dt": 0.0,
            "modality": "uwb",
            "meta": {"scene_id": "S(A2,N2,V2,K0,M0)", "seq_id": "mini_seq"},
            "imu_payload": None,
            "uwb_payload": {"anchor_id": 0, "range": 2.0, "valid": True, "quality": 0.95},
            "vio_payload": None,
        },
        {
            "t": 0.038000000000000006,
            "dt": 0.010000000000000005,
            "modality": "vio",
            "meta": {"scene_id": "S(A2,N2,V2,K0,M0)", "seq_id": "mini_seq"},
            "imu_payload": None,
            "uwb_payload": None,
            "vio_payload": {
                "dx": 0.03,
                "dy": 0.0,
                "dyaw": 0.0,
                "quality": 0.85,
                "tracked_features": 60,
                "reproj_err": 0.4,
            },
        },
        {
            "t": 0.075,
            "dt": 0.037,
            "modality": "imu",
            "meta": {"scene_id": "S(A2,N2,V2,K0,M0)", "seq_id": "mini_seq"},
            "imu_payload": {"ax": 0.1, "ay": 0.0, "gz": 0.01},
            "uwb_payload": None,
            "vio_payload": None,
        },
    ]


def _scene_tasks_from_events(events: list[dict]) -> list[dict]:
    """Derive one minimal scene task from the smoke events."""
    if not events:
        raise ValueError("events must be non-empty")
    first_event = events[0]
    meta = dict(first_event.get("meta") or {})
    scene_id = str(meta.get("scene_id") or "S(A0,N0,V0,K0,M0)")
    seq_id = str(meta.get("seq_id") or "mini_seq")
    scene_spec = decode_scene(scene_id)
    axes = {
        "A": scene_spec.A_level,
        "N": scene_spec.N_level,
        "V": scene_spec.V_level,
        "K": scene_spec.K_value,
    }
    return [
        {
            "task_id": "toy_scene_00",
            "scene_id": scene_id,
            "seq_id": seq_id,
            "axes": axes,
        }
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the minimal core experiment path and print a summary."""
    print("[08_core_exp] 开始 | mode=quick scene=toy_scene_00", flush=True)
    parser = argparse.ArgumentParser(description="Run minimal core experiments")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "08_run_core_experiments")

    print("[08_core_exp] 加载配置", flush=True)
    if not args.config:
        print("[08_core_exp] 错误: 需要 --config 指定实验配置文件", flush=True)
        return 1
    experiment_cfg = copy.deepcopy(load_yaml_config(args.config))
    experiment_cfg.setdefault("mode", "quick")
    print_dict(experiment_cfg, "实验配置")
    events = _default_events()
    scene_tasks = _scene_tasks_from_events(events)
    output_root = args.output_root or str(ROOT / "outputs" / "core_script_smoke")
    print_dict(
        {
            "scene_tasks_count": len(scene_tasks),
            "events_count": len(events),
            "output_root": str(output_root),
        },
        "派生参数",
    )
    print_dict({"scene_tasks": scene_tasks}, "场景任务详情")
    print(f"[08_core_exp] 运行核心流水线 | methods={experiment_cfg['methods']}", flush=True)
    result = CorePipeline().run(
        {
            "experiment_cfg": experiment_cfg,
            "scene_tasks": scene_tasks,
            "events": events,
            "methods": experiment_cfg["methods"],
            "output_root": output_root,
        }
    )
    prediction_bundles = result.metadata["prediction_bundles"]
    summary = {
        "stage_name": result.stage_name,
        "artifacts": result.artifacts,
        "num_bundles": len(prediction_bundles),
    }
    print(dumps_json_text(summary))
    print("[08_core_exp] 完成 | 返回码=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
