"""Heading / yaw-arrow plotter (异步高NLOS实验全流程保障手册 Part 3 §0-V4).

Renders a 2-D trajectory with a heading arrow drawn every N steps. The arrow length
is fixed in metres; the arrow direction encodes the per-step yaw. This is the
fourth of six pre-flight visualizations (§0-V1..§0-V6) and is the cheapest way
to spot a yaw mirror / sign-flip / 180°-jump — all three are immediately
obvious as an arrow fan that points the wrong way.

The plotter does NOT do Sim(3) alignment — the caller must pass already-aligned
trajectories (P9). Yaw is assumed to be in radians and to use the enu / FLU
convention established in S1.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend.

import matplotlib.pyplot as plt  # plotting only.
import numpy as np  # vectorized numeric ops.

from liquidloc.common.validation import coerce_finite_scalar  # D9 validation contract.


__all__ = [
    "build_heading_arrow_figure_spec",
    "render_heading_arrow_figure",
]


def _coerce_1d_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN/Inf).")
    return arr


def build_heading_arrow_figure_spec(
    *,
    px: Any,
    py: Any,
    yaw: Any,
    seq_id: str,
    arrow_stride: int = 20,
) -> dict[str, Any]:
    """Build the spec for the heading-arrow figure (§0-V4)."""
    x = _coerce_1d_array(px, name="px")
    y = _coerce_1d_array(py, name="py")
    yw = _coerce_1d_array(yaw, name="yaw")
    if not (x.size == y.size == yw.size):
        raise ValueError("px / py / yaw must have equal length.")
    if not isinstance(arrow_stride, int) or arrow_stride < 1:
        raise ValueError("arrow_stride must be a positive int.")
    sampled_idx = np.arange(0, x.size, int(arrow_stride))
    if sampled_idx.size == 0:
        sampled_idx = np.array([0])
    return {
        "kind": "heading_arrows",
        "seq_id": str(seq_id),
        "n_points": int(x.size),
        "n_arrows": int(sampled_idx.size),
        "arrow_stride": int(arrow_stride),
        "yaw_range_rad": [float(yw.min()), float(yw.max())],
    }


def render_heading_arrow_figure(
    *,
    px: Any,
    py: Any,
    yaw: Any,
    seq_id: str,
    output_path: str | Path,
    arrow_stride: int = 20,
    arrow_length_m: float = 0.6,
    title: str | None = None,
) -> dict[str, Any]:
    """Render the trajectory + heading-arrow figure to ``output_path``."""
    spec = build_heading_arrow_figure_spec(
        px=px,
        py=py,
        yaw=yaw,
        seq_id=seq_id,
        arrow_stride=arrow_stride,
    )
    x = np.asarray(px, dtype=np.float64).reshape(-1)
    y = np.asarray(py, dtype=np.float64).reshape(-1)
    yw = np.asarray(yaw, dtype=np.float64).reshape(-1)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(x, y, color="#1f77b4", linewidth=1.0, alpha=0.6, label="trajectory")
    sampled_idx = np.arange(0, x.size, int(arrow_stride))
    if sampled_idx.size == 0:
        sampled_idx = np.array([0])
    for i in sampled_idx:
        dx = arrow_length_m * math.cos(float(yw[i]))
        dy = arrow_length_m * math.sin(float(yw[i]))
        ax.arrow(
            float(x[i]),
            float(y[i]),
            dx,
            dy,
            head_width=0.15,
            head_length=0.15,
            fc="#d62728",
            ec="#d62728",
            alpha=0.8,
        )
    ax.scatter([x[0]], [y[0]], color="green", s=40, label="start", zorder=3)
    ax.scatter([x[-1]], [y[-1]], color="black", s=40, label="end", zorder=3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title or f"Heading arrows — {seq_id}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
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
    _t = np.linspace(0.0, 6.0, 60)
    _x = np.cumsum(0.1 * np.cos(_t))
    _y = np.cumsum(0.1 * np.sin(_t))
    _yaw = _t
    out = render_heading_arrow_figure(
        px=_x,
        py=_y,
        yaw=_yaw,
        seq_id="smoke_seq",
        output_path="outputs/figures/_smoke_heading_arrows.png",
    )
    print("render_heading_arrow_figure:", out["render_status"], out["figure_path"])