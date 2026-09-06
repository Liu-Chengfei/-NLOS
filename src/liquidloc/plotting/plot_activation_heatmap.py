"""Activation heatmap plotter for the 4 heads (异步高NLOS实验全流程保障手册 Part 3 §0-V3).

Renders 4 stacked heatmap rows (risk / bias / uwb_scaling / vio_scaling) with a
bottom NLOS-band row showing the `nl_flag` (or `scene_mask`) binary signal. This
is the third of six pre-flight visualizations (§0-V1..§0-V6). The intent is
that risk / bias heads visibly activate inside the NLOS band while uwb_scaling
and vio_scaling stay relatively flat — failing to see that pattern is the D20
diagnostic action.

Inputs are expected to be aligned time-series of equal length: per-head
activations are continuous scalars in [0, 1]; nl_flag is a binary 0/1 vector.
"""

from __future__ import annotations

import math
from collections.abc import Mapping  # type-check Mapping input.
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless rendering.

import matplotlib.pyplot as plt  # plotting only.
import numpy as np  # vectorized numeric ops.

from liquidloc.common.validation import coerce_finite_scalar  # D9 validation contract.


__all__ = [
    "build_activation_heatmap_figure_spec",
    "render_activation_heatmap_figure",
]


_HEAD_ORDER = ("risk", "bias", "uwb_scaling", "vio_scaling")  # fixed ordering matching MODEL_INTERMEDIATE_KEYS layout convention.


def _coerce_1d_array(values: Any, *, name: str, value_range: tuple[float, float] | None = None) -> np.ndarray:
    """Convert to a 1-D float64 array and validate finiteness / value range."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN/Inf).")
    if value_range is not None:
        lo, hi = value_range
        if float(arr.min()) < lo or float(arr.max()) > hi:
            raise ValueError(f"{name} values must lie in [{lo}, {hi}].")
    return arr


def build_activation_heatmap_figure_spec(
    *,
    per_head_activations: Mapping[str, Any],
    nl_flag: Any,
    t: Any,
    seq_id: str,
) -> dict[str, Any]:
    """Build the spec for the activation-heatmap figure (§0-V3).

    The spec dict records per-head mean / p95 activation values plus the NLOS
    band length, so downstream runs can quantify how well the heads light up on
    NLOS segments without re-loading the raw arrays.
    """
    if not isinstance(per_head_activations, dict):
        raise TypeError("per_head_activations must be a dict[str, sequence].")
    head_arrays: dict[str, np.ndarray] = {}
    for head_name in _HEAD_ORDER:
        if head_name not in per_head_activations:
            raise ValueError(f"per_head_activations missing required head '{head_name}'.")
        head_arrays[head_name] = _coerce_1d_array(
            per_head_activations[head_name],
            name=f"per_head_activations[{head_name}]",
            value_range=(0.0, 1.0),
        )
    nl = _coerce_1d_array(nl_flag, name="nl_flag", value_range=(0.0, 1.0))
    t_arr = _coerce_1d_array(t, name="t")
    sizes = {t_arr.size, nl.size, *[v.size for v in head_arrays.values()]}
    if len(sizes) != 1:
        raise ValueError("t / nl_flag / per-head activations must have equal length.")
    # Quantize NLOS to a strict 0/1 band so heat-mask alignment is binary.
    nl_unique = {float(v) for v in np.unique(nl)}
    if not nl_unique.issubset({0.0, 1.0}):
        raise ValueError("nl_flag must be binary (0/1).")
    spec: dict[str, Any] = {
        "kind": "activation_heatmap",
        "seq_id": str(seq_id),
        "n_points": int(t_arr.size),
        "per_head": {
            h: {"mean": float(head_arrays[h].mean()), "p95": float(np.percentile(head_arrays[h], 95))}
            for h in _HEAD_ORDER
        },
        "nlos_band_fraction": float(nl.mean()),
    }
    return spec


def render_activation_heatmap_figure(
    *,
    per_head_activations: Any,
    nl_flag: Any,
    t: Any,
    seq_id: str,
    output_path: str | Path,
    method_name: str | None = None,
) -> dict[str, Any]:
    """Render the 4-head activation heatmap + NLOS band figure to ``output_path``."""
    spec = build_activation_heatmap_figure_spec(
        per_head_activations=per_head_activations,
        nl_flag=nl_flag,
        t=t,
        seq_id=seq_id,
    )
    t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
    nl = np.asarray(nl_flag, dtype=np.float64).reshape(-1)
    head_arrays = {
        h: np.asarray(per_head_activations[h], dtype=np.float64).reshape(-1)
        for h in _HEAD_ORDER
    }

    fig, axes = plt.subplots(
        len(_HEAD_ORDER) + 1,
        1,
        figsize=(9.0, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.4]},
    )
    extent = [float(t_arr[0]), float(t_arr[-1]), 0.0, 1.0]
    for idx, head_name in enumerate(_HEAD_ORDER):
        axes[idx].imshow(
            head_arrays[head_name][np.newaxis, :],
            aspect="auto",
            cmap="viridis",
            extent=extent,
            vmin=0.0,
            vmax=1.0,
        )
        axes[idx].set_yticks([0.5])
        axes[idx].set_yticklabels([head_name])
        axes[idx].set_ylabel(head_name, rotation=0, labelpad=22, ha="right")

    axes[-1].fill_between(t_arr, 0.0, nl, color="#d62728", alpha=0.6)
    axes[-1].set_ylim(0.0, 1.05)
    axes[-1].set_yticks([0.5])
    axes[-1].set_yticklabels(["nl_flag"])
    axes[-1].set_ylabel("nl_flag", rotation=0, labelpad=22, ha="right")
    axes[-1].set_xlabel("t [s]")

    title_method = method_name or "model"
    fig.suptitle(f"4-head activations & NLOS band — {title_method} / {seq_id}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

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
    _t = np.linspace(0.0, 5.0, 50)
    _nl = np.zeros(50)
    _nl[20:35] = 1.0  # NLOS burst at t = 2.0-3.5s
    _activ = {
        "risk": np.clip(0.5 + 0.4 * _nl, 0.0, 1.0),
        "bias": np.clip(0.4 + 0.3 * _nl, 0.0, 1.0),
        "uwb_scaling": np.full_like(_nl, 0.5),
        "vio_scaling": np.full_like(_nl, 0.5),
    }
    out = render_activation_heatmap_figure(
        per_head_activations=_activ,
        nl_flag=_nl,
        t=_t,
        seq_id="smoke_seq",
        method_name="lnn_smoke",
        output_path="outputs/figures/_smoke_activation_heatmap.png",
    )
    print("render_activation_heatmap_figure:", out["render_status"], out["figure_path"])