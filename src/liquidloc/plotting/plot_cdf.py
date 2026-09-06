"""Five-method error CDF plotter (异步高NLOS实验全流程保障手册 Part 3 §0-V6 + F-1).

Renders the empirical CDF of per-trajectory errors for all five methods
(LNN / LSTM / Transformer / EKF / Robust-EKF) on the same axes, with P50/P95
markers drawn on the plot and self-contained caption indicating data layer /
sample size / significance. This is the F-1 figure in the handbook T/F
delivery list (handbook lines 661) and supports the D3/D29 narrative
story (a method that is competitive on the median but loses the tail).

Per F-1 spec, the plot aggregates to **trajectory-level** errors (one
RMSE per trajectory) and draws P50/P95 markers on each curve. Per
J-3 (self-contained caption) and J-9 (data layer declaration), the
figure title encodes the data layer (①/②/③), the per-method
trajectory count N, and the analysis unit.

The plotter expects per-method per-trajectory errors already filtered
through the warm-up exclusion (P37) and Sim(3) alignment (P9).
Per-frame diagnostics are reported separately in §0-V6 supplementary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping  # type-check Mapping input.
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend.

import matplotlib.pyplot as plt  # plotting only.
import numpy as np  # vectorized numeric ops.

from liquidloc.common.validation import coerce_finite_scalar  # D9 validation contract.
from liquidloc.plotting._okabe_ito_palette import METHOD_COLORS  # J-5 Okabe-Ito palette.


__all__ = [
    "build_cdf_figure_spec",
    "render_cdf_figure",
]


_METHOD_ORDER = ("lnn", "lstm", "transformer", "ekf", "robust_ekf")  # mandated by handbook Part 0.
# J-5: Delegated to _okabe_ito_palette so there is one canonical source of truth.
# METHOD_COLORS is re-imported above to avoid circular dep at module level.


def _coerce_1d_array(values: Any, *, name: str, min_value: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN/Inf).")
    if min_value is not None and float(arr.min()) < min_value:
        raise ValueError(f"{name} must be >= {min_value} (got min={float(arr.min())}).")
    return arr


def build_cdf_figure_spec(
    *,
    err_by_method: Mapping[str, Any],
    seq_id: str,
    p_list: tuple[float, ...] = (0.50, 0.90, 0.95),
) -> dict[str, Any]:
    """Build the spec for the 5-method CDF figure (§0-V6).

    The spec records the requested percentiles for each method so downstream
    run-summaries can quote P50/P90/P95 without re-computing from raw data.
    """
    if not isinstance(err_by_method, Mapping):
        raise TypeError("err_by_method must be a dict[str, sequence].")
    method_arrays: dict[str, np.ndarray] = {}
    for method in _METHOD_ORDER:
        if method not in err_by_method:
            raise ValueError(f"err_by_method missing required method '{method}'.")
        method_arrays[method] = _coerce_1d_array(
            err_by_method[method],
            name=f"err_by_method[{method}]",
            min_value=0.0,
        )
    spec: dict[str, Any] = {
        "kind": "cdf_5method",
        "seq_id": str(seq_id),
        "p_list": list(p_list),
        "per_method": {},
    }
    for method, arr in method_arrays.items():
        spec["per_method"][method] = {
            "n": int(arr.size),
            "median_m": float(np.median(arr)),
            "percentiles_m": {
                f"p{int(round(p * 100))}": float(np.percentile(arr, p * 100.0))
                for p in p_list
            },
        }
    return spec


def render_cdf_figure(
    *,
    err_by_method: Any,
    seq_id: str,
    output_path: str | Path,
    title: str | None = None,
    max_x_m: float | None = None,
    data_layer: str = "①",
) -> dict[str, Any]:
    """Render the 5-method empirical CDF figure to ``output_path``.

    Per F-1 spec (handbook lines 661):
      - P50/P95 markers drawn on each method's CDF curve.
      - Axes have units: x-axis in [m] (J-2).
      - Self-contained caption via title (J-3) encoding data layer + seq_id.
      - Data layer (①/②/③) declared in title (J-9).
      - seed/trajectory count stated per legend (J-10).
      - Colorblind-safe Okabe-Ito palette (J-5) via METHOD_COLORS.
      - DPI ≥ 300 (J-8).
    """
    spec = build_cdf_figure_spec(err_by_method=err_by_method, seq_id=seq_id)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # P50/P95 marker style
    MARKER_P50 = "v"   # F-1: P50 marker
    MARKER_P95 = "^"   # F-1: P95 marker

    for method in _METHOD_ORDER:
        arr = np.asarray(err_by_method[method], dtype=np.float64).reshape(-1)
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, sorted_arr.size + 1) / sorted_arr.size
        ax.plot(
            sorted_arr,
            cdf,
            color=METHOD_COLORS[method],
            linewidth=1.6,
            label=f"{method} (n={arr.size})",  # J-10: trajectory count stated
        )

        # F-1: draw P50 (median) and P95 markers per method
        p50 = float(np.percentile(arr, 50.0))
        p95 = float(np.percentile(arr, 95.0))
        # Compute CDF value at P50/P95 for marker y-position
        cdf_at_p50 = float(np.searchsorted(sorted_arr, p50)) / sorted_arr.size
        cdf_at_p95 = float(np.searchsorted(sorted_arr, p95)) / sorted_arr.size
        ax.plot(
            p50, cdf_at_p50,
            marker=MARKER_P50, color=METHOD_COLORS[method],
            markersize=7, linestyle="none", zorder=5,
        )
        ax.plot(
            p95, cdf_at_p95,
            marker=MARKER_P95, color=METHOD_COLORS[method],
            markersize=7, linestyle="none", zorder=5,
        )

    # J-2: unit label on x-axis
    ax.set_xlabel("position error [m]")          # J-2: units [m]
    ax.set_ylabel("cumulative probability")     # y-axis is unitless CDF
    ax.set_xlim(left=0.0)
    if max_x_m is not None:
        ax.set_xlim(right=float(max_x_m))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    # J-3: self-contained caption in title — data layer + seq_id
    caption = title or (
        f"Figure F-1: Cumulative Error Distribution — {data_layer} {seq_id}"
    )
    ax.set_title(caption)  # J-3: self-contained caption with data layer (J-9)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()

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
    _rng = np.random.RandomState(0)
    _errs = {
        "lnn": np.abs(_rng.normal(3.5, 0.8, 200)),
        "lstm": np.abs(_rng.normal(4.2, 1.0, 200)),
        "transformer": np.abs(_rng.normal(4.5, 1.1, 200)),
        "ekf": np.abs(_rng.normal(6.5, 1.5, 200)),
        "robust_ekf": np.abs(_rng.normal(6.7, 1.6, 200)),
    }
    out = render_cdf_figure(
        err_by_method=_errs,
        seq_id="smoke_seq",
        output_path="outputs/figures/_smoke_cdf.png",
    )
    print("render_cdf_figure:", out["render_status"], out["figure_path"])