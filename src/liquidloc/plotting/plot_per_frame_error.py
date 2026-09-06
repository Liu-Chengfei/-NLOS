"""Per-frame error time-series plotter (异步高NLOS实验全流程保障手册 Part 3 §0-V2).

Renders per-frame (x, y, yaw) error curves versus time for one method on one trajectory.
This is the second of six "pre-flight visualization" checks (§0-V1..§0-V6). The figure
must surface three failure modes:
1. Linear drift in any channel (initial pose / yaw initialization still leaking in)
2. Directional bias consistent with a yaw mirror or sign flip
3. Unit mismatch (e.g., x1000) showing as obvious scale jumps

The function does NOT do Sim(3) alignment; the caller is expected to have aligned
trajectories already (P9). It only renders the aligned per-frame error into a figure.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — required for headless rendering.

import matplotlib.pyplot as plt  # plotting only.
import numpy as np  # vectorized numeric ops.

from liquidloc.common.validation import coerce_finite_scalar, is_integer  # D9 validation contract.


__all__ = [
    "build_per_frame_error_figure_spec",
    "render_per_frame_error_figure",
]


def _coerce_1d_array(values: Any, *, name: str) -> np.ndarray:
    """Convert a sequence-like input into a 1-D float64 numpy array.

    The value contract here matches `metrics_quality._coerce_*`: NaN/Inf raises
    ValueError, non-numeric raises TypeError. Empty arrays are rejected so the
    downstream plot cannot silently render a blank axis.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN/Inf).")
    return arr


def build_per_frame_error_figure_spec(
    *,
    t: Any,
    err_x: Any,
    err_y: Any,
    err_yaw: Any,
    method_name: str,
    seq_id: str,
) -> dict[str, Any]:
    """Build the figure specification for the per-frame error plot (§0-V2).

    Returns a JSON-serialisable dict mirroring the layout used by the other
    plotting modules so downstream figure runners can ingest it uniformly.
    """
    t_arr = _coerce_1d_array(t, name="t")
    ex = _coerce_1d_array(err_x, name="err_x")
    ey = _coerce_1d_array(err_y, name="err_y")
    eyaw = _coerce_1d_array(err_yaw, name="err_yaw")
    if not (t_arr.size == ex.size == ey.size == eyaw.size):
        raise ValueError(
            "t / err_x / err_y / err_yaw must have equal length."
        )
    return {
        "kind": "per_frame_error",
        "seq_id": str(seq_id),
        "method_name": str(method_name),
        "n_points": int(t_arr.size),
        "summary": {
            "abs_max_x_m": float(np.max(np.abs(ex))),
            "abs_max_y_m": float(np.max(np.abs(ey))),
            "abs_max_yaw_rad": float(np.max(np.abs(eyaw))),
            "drift_x_m": float(ex[-1] - ex[0]),
            "drift_y_m": float(ey[-1] - ey[0]),
            "drift_yaw_rad": float(eyaw[-1] - eyaw[0]),
        },
    }


def render_per_frame_error_figure(
    *,
    t: Any,
    err_x: Any,
    err_y: Any,
    err_yaw: Any,
    method_name: str,
    seq_id: str,
    output_path: str | Path,
    title: str | None = None,
) -> dict[str, Any]:
    """Render the per-frame error time-series figure to ``output_path``.

    Returns a dict with `figure_path` (str), `figure_spec` (the spec produced by
    ``build_per_frame_error_figure_spec``) and `render_status`. On disk-write
    failure the dict includes `error_type` / `error_detail` instead of throwing,
    matching the safe-write pattern used by the other plotting modules.
    """
    spec = build_per_frame_error_figure_spec(
        t=t,
        err_x=err_x,
        err_y=err_y,
        err_yaw=err_yaw,
        method_name=method_name,
        seq_id=seq_id,
    )
    t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
    ex = np.asarray(err_x, dtype=np.float64).reshape(-1)
    ey = np.asarray(err_y, dtype=np.float64).reshape(-1)
    eyaw = np.asarray(err_yaw, dtype=np.float64).reshape(-1)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.5), sharex=True)
    axes[0].plot(t_arr, ex, color="#1f77b4", linewidth=1.0)
    axes[0].axhline(0.0, color="black", linewidth=0.5)
    axes[0].set_ylabel("x error [m]")
    axes[0].set_title(title or f"Per-frame error — {method_name} / {seq_id}")

    axes[1].plot(t_arr, ey, color="#2ca02c", linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_ylabel("y error [m]")

    axes[2].plot(t_arr, eyaw, color="#d62728", linewidth=1.0)
    axes[2].axhline(0.0, color="black", linewidth=0.5)
    axes[2].set_ylabel("yaw error [rad]")
    axes[2].set_xlabel("t [s]")

    fig.tight_layout()

    out_path = Path(output_path).resolve()
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    except OSError as exc:
        plt.close(fig)
        return {
            "figure_path": str(out_path),
            "figure_spec": spec,
            "render_status": "failed",
            "error_type": type(exc).__name__,
            "error_detail": str(exc),
        }
    plt.close(fig)
    return {
        "figure_path": str(out_path),
        "figure_spec": spec,
        "render_status": "ok",
    }


if __name__ == "__main__":
    # Manual smoke: 30-frame synthetic error, save to /tmp.
    _t = np.linspace(0.0, 3.0, 30)
    _ex = 0.1 * np.sin(_t) + 0.01 * _t
    _ey = 0.05 * np.cos(_t)
    _eyaw = 0.02 * np.ones_like(_t)
    out = render_per_frame_error_figure(
        t=_t,
        err_x=_ex,
        err_y=_ey,
        err_yaw=_eyaw,
        method_name="lnn_smoke",
        seq_id="smoke_seq",
        output_path="outputs/figures/_smoke_per_frame_error.png",
    )
    print("render_per_frame_error_figure:", out["render_status"], out["figure_path"])