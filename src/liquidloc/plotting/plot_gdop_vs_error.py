"""GDOP-vs-error dual-curve plotter (异步高NLOS实验全流程保障手册 Part 3 §0-V5).

Renders a dual-axis time-series: GDOP on the left axis (geometry quality) and
the per-frame positioning error on the right axis (one method). The §0-V5
diagnostic is the regression-by-eye: high-GDOP regions should generally produce
higher error. A perfectly inverted correlation (low GDOP → high error) is the
§0-V5 / P4 red-flag — typically caused by mis-aligned anchor coordinates or a
GDOP computation that used the wrong reference trajectory.

Inputs:
- ``t`` — 1-D array of frame timestamps in seconds.
- ``gdop`` — 1-D array of GDOP values (>=1.0, dimensionless).
- ``err`` — 1-D array of per-frame positioning errors in metres.
- ``method_name`` — label for the right-axis series.

The plotter does NOT compute GDOP itself — callers should source it from
``scenarios/geometry_levels.py`` or `` scripts/s9_validate_seeds.py``.
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
    "build_gdop_vs_error_figure_spec",
    "render_gdop_vs_error_figure",
]


def _coerce_1d_array(values: Any, *, name: str, min_value: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN/Inf).")
    if min_value is not None and float(arr.min()) < min_value:
        raise ValueError(f"{name} must be >= {min_value} (got min={float(arr.min())}).")
    return arr


def build_gdop_vs_error_figure_spec(
    *,
    t: Any,
    gdop: Any,
    err: Any,
    method_name: str,
    seq_id: str,
) -> dict[str, Any]:
    """Build the spec for the GDOP-vs-error figure (§0-V5)."""
    t_arr = _coerce_1d_array(t, name="t")
    g_arr = _coerce_1d_array(gdop, name="gdop", min_value=1.0)
    e_arr = _coerce_1d_array(err, name="err", min_value=0.0)
    if not (t_arr.size == g_arr.size == e_arr.size):
        raise ValueError("t / gdop / err must have equal length.")
    # Pearson correlation is a coarse §0-V5 sanity check: large negative correlation
    # indicates the geometry-error amplifier has been broken (or anchor coords flipped).
    if g_arr.std() > 1e-12 and e_arr.std() > 1e-12:
        corr = float(np.corrcoef(g_arr, e_arr)[0, 1])
    else:
        corr = 0.0
    return {
        "kind": "gdop_vs_error",
        "seq_id": str(seq_id),
        "method_name": str(method_name),
        "n_points": int(t_arr.size),
        "gdop_range": [float(g_arr.min()), float(g_arr.max())],
        "err_range_m": [float(e_arr.min()), float(e_arr.max())],
        "pearson_gdop_err": corr,
    }


def render_gdop_vs_error_figure(
    *,
    t: Any,
    gdop: Any,
    err: Any,
    method_name: str,
    seq_id: str,
    output_path: str | Path,
    title: str | None = None,
) -> dict[str, Any]:
    """Render the dual-axis GDOP-vs-error figure to ``output_path``."""
    spec = build_gdop_vs_error_figure_spec(
        t=t,
        gdop=gdop,
        err=err,
        method_name=method_name,
        seq_id=seq_id,
    )
    t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
    g_arr = np.asarray(gdop, dtype=np.float64).reshape(-1)
    e_arr = np.asarray(err, dtype=np.float64).reshape(-1)

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    color_g = "#1f77b4"
    ax1.plot(t_arr, g_arr, color=color_g, linewidth=1.0, label="GDOP")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("GDOP", color=color_g)
    ax1.tick_params(axis="y", labelcolor=color_g)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_e = "#d62728"
    ax2.plot(t_arr, e_arr, color=color_e, linewidth=0.9, alpha=0.8, label="error")
    ax2.set_ylabel(f"{method_name} error [m]", color=color_e)
    ax2.tick_params(axis="y", labelcolor=color_e)

    corr = spec["pearson_gdop_err"]
    fig.suptitle(
        title or f"GDOP vs error — {method_name} / {seq_id} (r={corr:+.2f})"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(output_path).resolve()
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)  # J-8: ≥300 dpi per handbook J-8
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
    _t = np.linspace(0.0, 4.0, 40)
    _gdop = 1.5 + 0.5 * np.sin(_t / 2.0)
    _err = 0.5 + 0.3 * np.sin(_t / 2.0 + 0.4) + 0.05 * np.random.RandomState(0).randn(40)
    _err = np.clip(_err, 0.0, None)
    out = render_gdop_vs_error_figure(
        t=_t,
        gdop=_gdop,
        err=_err,
        method_name="lnn_smoke",
        seq_id="smoke_seq",
        output_path="outputs/figures/_smoke_gdop_vs_error.png",
    )
    print("render_gdop_vs_error_figure:", out["render_status"], out["figure_path"])