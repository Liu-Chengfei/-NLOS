"""Run the minimal MILUV evaluation smoke pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from liquidloc.common.config_utils import load_dataset_config
from liquidloc.common.io_utils import dumps_json_text
from liquidloc.pipelines.eval_pipeline import EvalPipeline
from liquidloc.pipelines.miluv_pipeline import MiluvPipeline


def _resolve_non_empty_path(
    raw_value: str | None,
    *,
    flag_name: str,
    default: Path | None = None,
) -> Path:
    """Resolve a CLI path against the repo root and reject blank values."""
    if raw_value is None:
        if default is None:
            raise ValueError(f"{flag_name} must be a non-empty path")
        return default.resolve()
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"{flag_name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path.resolve()


def _default_raw_root() -> Path:
    """Return the fixture-backed MILUV raw root used by smoke runs."""
    return ROOT / "tests" / "fixtures" / "datasets" / "miluv"


def _default_output_root() -> Path:
    """Return the default script smoke output root."""
    return ROOT / "outputs" / "miluv_script_smoke"


def _print_summary(payload: dict[str, object]) -> int:
    """Emit a structured JSON summary and return the exit code."""
    print(dumps_json_text(payload, indent=None))
    return int(payload["exit_code"])


def main(argv: list[str] | None = None) -> int:
    """Build the minimal MILUV input and run prediction plus evaluation."""
    print("[10_miluv_eval] 开始 | 最小 MILUV 评估冒烟", flush=True)
    parser = argparse.ArgumentParser(description="Run minimal MILUV evaluation")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "10_run_miluv_eval")

    try:
        print("[10_miluv_eval] 解析路径并加载数据集配置", flush=True)
        raw_root = _resolve_non_empty_path(
            args.raw_root,
            flag_name="--raw-root",
            default=_default_raw_root(),
        )
        output_root = _resolve_non_empty_path(
            args.output_root,
            flag_name="--output-root",
            default=_default_output_root(),
        )
        dataset_cfg = load_dataset_config(
            ROOT / "configs" / "datasets" / "miluv.yaml",
            required_keys=["field_mapping"],
        )
        field_mapping = dataset_cfg["field_mapping"]
        print_dict(dataset_cfg, "数据集配置 (miluv.yaml)")

        print("[10_miluv_eval] 运行 MiluvPipeline 预测", flush=True)
        # E7 真实数据评估：4 个方法对比（1 传统基线 + 3 神经网络）
        # 传统基线: EKF
        # 神经网络: lstm_ekf, liquid_ekf, transformer_ekf
        miluv_payload = {
            "raw_root": str(raw_root),
            "field_mapping": field_mapping,
            "seq_ids": ["mini_seq"],
            "methods": ["ekf", "lstm_ekf", "liquid_ekf", "transformer_ekf"],
            "output_root": str(output_root),
        }
        print_dict(miluv_payload, "评估 payload")
        prediction_result = MiluvPipeline().run(miluv_payload)
        prediction_bundles = prediction_result.metadata["prediction_bundles"]
        eval_output_root = (output_root / "eval").resolve()
        print("[10_miluv_eval] 运行 EvalPipeline 评估", flush=True)
        eval_payload = {
            "prediction_bundles": prediction_bundles,
            "ground_truth_root": str(raw_root),
            "mode": "quick",
            "output_root": str(eval_output_root),
        }
        print_dict(eval_payload, "评估执行 payload")
        eval_result = EvalPipeline().run(eval_payload)
    except Exception as exc:
        _rc = _print_summary(
            {
                "status": "failed",
                "stage": "run_miluv_eval",
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "exit_code": 1,
            }
        )
        print(f"[10_miluv_eval] 完成 | 返回码={_rc}", flush=True)
        return _rc

    _rc = _print_summary(
        {
            "status": "ok",
            "stage": "run_miluv_eval",
            "output_root": str(output_root),
            "eval_output_root": str(eval_output_root),
            "pipeline_stage": eval_result.stage_name,
            "artifact_count": len(eval_result.artifacts),
            "num_bundles": len(prediction_bundles),
            "exit_code": 0,
        }
    )
    print(f"[10_miluv_eval] 完成 | 返回码={_rc}", flush=True)
    return _rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
