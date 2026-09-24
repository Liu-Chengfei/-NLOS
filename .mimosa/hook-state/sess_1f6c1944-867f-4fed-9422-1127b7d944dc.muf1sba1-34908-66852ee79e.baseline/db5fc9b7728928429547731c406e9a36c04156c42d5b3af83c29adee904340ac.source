"""Generate figures and a figure manifest from frozen plotting inputs."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from liquidloc.common.io_utils import dumps_json_text, read_json, write_json
from liquidloc.plotting.plot_calibration import render_calibration_figure
from liquidloc.plotting.plot_cases import render_case_figures
from liquidloc.plotting.plot_main_table import render_main_table_figure
from liquidloc.plotting.plot_runtime import render_runtime_figure
from liquidloc.plotting.plot_sweeps import render_sweep_figure
from liquidloc.plotting.plot_training_trends import render_training_trend_figure
from liquidloc.plotting.plot_trajectories import render_trajectory_figure


def _resolve_non_empty_path(raw_value: str | None, *, flag_name: str, default: Path | None = None) -> Path:
    """Normalize a CLI path argument into an absolute path."""
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


def _read_json(path: Path) -> Any:
    """Read one JSON payload from disk."""
    return read_json(path, encoding="utf-8-sig")


def _default_output_root() -> Path:
    """Return the default figure output root."""
    return ROOT / "outputs" / "script_smoke" / "figures"


def _unwrap_named_payload(root_payload: Mapping[str, Any], key: str) -> Any:
    """Unwrap an optionally double-wrapped named payload."""
    if key not in root_payload:
        return None
    current: Any = root_payload[key]
    if isinstance(current, Mapping) and key in current:
        current = current[key]
    return current


def _merged_figure_cfg(
    *,
    output_root: Path,
    default_filename: str,
    global_cfg: Any,
    local_cfg: Any,
) -> dict[str, Any]:
    """Merge global and local plotting config into one figure config."""
    merged: dict[str, Any] = {}
    if isinstance(global_cfg, Mapping):
        merged.update(dict(global_cfg))
    if isinstance(local_cfg, Mapping):
        merged.update(dict(local_cfg))
    merged.setdefault("figure_path", str((output_root / default_filename).resolve()))
    return merged


def _normalize_json_value(value: Any) -> Any:
    """Convert nested payloads into JSON-friendly values."""
    if isinstance(value, os.PathLike):
        return str(Path(value))
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_value(item) for item in value]
    return value


def _normalize_figure_result(result: Any) -> dict[str, Any]:
    """Normalize one renderer result into a manifest entry."""
    if isinstance(result, (str, os.PathLike)):
        return {"figure_path": str(Path(result))}
    if isinstance(result, Mapping) and result.get("skipped"):
        return dict(result)
    normalized = _normalize_json_value(result)
    if not isinstance(normalized, Mapping):
        raise TypeError("figure renderer result must be a mapping or path-like")
    return dict(normalized)


def _print_summary(payload: dict[str, Any]) -> int:
    """Print a structured summary and return its exit code."""
    print(dumps_json_text(payload, indent=None))
    return int(payload["exit_code"])


def _safe_write_manifest(manifest_path: Path, figure_manifest: Mapping[str, Any]) -> tuple[bool, str | None, str | None]:
    """Write the figure manifest, returning structured write errors."""
    try:
        write_json(manifest_path, {"figure_manifest": figure_manifest})
    except OSError as exc:
        return False, type(exc).__name__, str(exc)
    return True, None, None


def main(argv: list[str] | None = None) -> int:
    """Read plotting inputs, render figures, and write a manifest."""
    print("[13_figures] 开始 | 从冻结绘图输入生成图表", flush=True)
    parser = argparse.ArgumentParser(description="Generate figures from frozen plotting inputs")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "13_generate_figures")

    try:
        print("[13_figures] 解析路径并读取图表输入", flush=True)
        input_path = _resolve_non_empty_path(args.input_path, flag_name="--input-path")
        output_root = _resolve_non_empty_path(
            args.output_root,
            flag_name="--output-root",
            default=_default_output_root(),
        )
        output_root.mkdir(parents=True, exist_ok=True)
        payload = _read_json(input_path)
        if not isinstance(payload, Mapping):
            raise TypeError("figure input payload must be a mapping")

        print_dict({"input_path": str(input_path), "output_root": str(output_root), "payload_keys": list(payload.keys()) if isinstance(payload, dict) else "N/A"}, "路径与 payload 概览")

        figure_cfg = payload.get("figure_cfg")
        if figure_cfg is not None:
            print_dict(figure_cfg, "图表配置 (figure_cfg)")
        main_table = _unwrap_named_payload(payload, "main_table")
        metric_table = _unwrap_named_payload(payload, "metric_table")
        runtime_table = _unwrap_named_payload(payload, "runtime_table")
        calibration_report = _unwrap_named_payload(payload, "calibration_report")
        selected_cases = _unwrap_named_payload(payload, "selected_cases")
        sweep_table = _unwrap_named_payload(payload, "sweep_table")
        training_trend_report = _unwrap_named_payload(payload, "training_trend_report")
        trajectory_bundle = _unwrap_named_payload(payload, "trajectory_bundle")

        print("[13_figures] 渲染图表", flush=True)
        figure_manifest: dict[str, dict[str, Any]] = {}
        if main_table is not None:
            figure_manifest["main_table"] = _normalize_figure_result(
                render_main_table_figure(
                    main_table,
                    _merged_figure_cfg(
                        output_root=output_root,
                        default_filename="main_table.png",
                        global_cfg=figure_cfg,
                        local_cfg=payload.get("main_table_figure_cfg"),
                    ),
                )
            )
        if runtime_table is not None:
            figure_manifest["runtime"] = _normalize_figure_result(
                render_runtime_figure(
                    runtime_table,
                    _merged_figure_cfg(
                        output_root=output_root,
                        default_filename="runtime.svg",
                        global_cfg=figure_cfg,
                        local_cfg=payload.get("runtime_figure_cfg"),
                    ),
                )
            )
        if calibration_report is not None:
            figure_manifest["calibration"] = _normalize_figure_result(
                render_calibration_figure(
                    calibration_report,
                    _merged_figure_cfg(
                        output_root=output_root,
                        default_filename="calibration.svg",
                        global_cfg=figure_cfg,
                        local_cfg=payload.get("calibration_figure_cfg"),
                    ),
                )
            )
        if selected_cases is not None:
            figure_manifest["cases"] = _normalize_figure_result(
                render_case_figures(
                    selected_cases,
                    _merged_figure_cfg(
                        output_root=output_root,
                        default_filename="cases.png",
                        global_cfg=figure_cfg,
                        local_cfg=payload.get("cases_figure_cfg"),
                    ),
                )
            )
        if sweep_table is not None:
            figure_manifest["sweep"] = _normalize_figure_result(
                render_sweep_figure(
                    sweep_table,
                    _merged_figure_cfg(
                        output_root=output_root,
                        default_filename="sweep.svg",
                        global_cfg=figure_cfg,
                        local_cfg=payload.get("sweep_figure_cfg"),
                    ),
                )
            )
        if training_trend_report is not None:
            figure_manifest["training_trend"] = _normalize_figure_result(
                render_training_trend_figure(
                    training_trend_report,
                    _merged_figure_cfg(
                        output_root=output_root,
                        default_filename="training_trends.svg",
                        global_cfg=figure_cfg,
                        local_cfg=payload.get("training_trend_figure_cfg"),
                    ),
                )
            )
        if trajectory_bundle is not None:
            figure_manifest["trajectory"] = _normalize_figure_result(
                render_trajectory_figure(
                    trajectory_bundle,
                    _merged_figure_cfg(
                        output_root=output_root,
                        default_filename="trajectory.png",
                        global_cfg=figure_cfg,
                        local_cfg=payload.get("trajectory_figure_cfg"),
                    ),
                )
            )

        manifest_path = (output_root / "figure_manifest.json").resolve()
    except Exception as exc:
        _rc = _print_summary(
            {
                "status": "failed",
                "stage": "generate_figures",
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "exit_code": 1,
            }
        )
        print(f"[13_figures] 完成 | 返回码={_rc}", flush=True)
        return _rc

    print("[13_figures] 写入图表清单", flush=True)
    write_ok, error_type, error_detail = _safe_write_manifest(manifest_path, figure_manifest)
    if not write_ok:
        _rc = _print_summary(
            {
                "status": "failed",
                "stage": "generate_figures",
                "manifest_path": str(manifest_path),
                "error_type": error_type,
                "detail": error_detail,
                "exit_code": 1,
            }
        )
        print(f"[13_figures] 完成 | 返回码={_rc}", flush=True)
        return _rc

    _rc = _print_summary(
        {
            "status": "ok",
            "stage": "generate_figures",
            "manifest_path": str(manifest_path),
            "figure_count": len(figure_manifest),
            "exit_code": 0,
        }
    )
    print(f"[13_figures] 完成 | 返回码={_rc}", flush=True)
    return _rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
