"""Paper-grade figure plotters F-2, F-5, F-6, F-7, F-8.

Per handbook T/F delivery list (handbook lines 657-668):
- F-2: Box/raincloud plot — 5 methods × trajectory-level RMSE distribution
- F-5: Per-scene error bar — 5 methods × (LOS/NLOS/Missing/P95 tail)
- F-6: 4-combo error bar — 5 methods × 4 combos mean±CI bar (C1→C4)
- F-7: Layout randomization ranking — per layout 5 method ranking
- F-8: MILUV segment error — methods on MILUV LOS/NLOS

All plotters:
- DPI ≥ 300 (J-8)
- Colorblind-safe Okabe-Ito palette (J-5)
- Axes with units [m] (J-2)
- Self-contained caption with data layer annotation (J-3, J-9)
- sample size (J-10)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from liquidloc.plotting._okabe_ito_palette import METHOD_COLORS


__all__ = [
    "render_boxplot_F2",
    "render_per_scene_error_bar_F5",
    "render_4combo_error_bar_F6",
    "render_layout_randomization_F7",
    "render_miluv_segment_F8",
    "build_t1_main_table",
    "build_t2_4combo_table",
    "build_t4_segment_decomposition_table",
]


_METHOD_ORDER = ("lnn", "lstm", "transformer", "ekf", "robust_ekf")
_METHOD_INTERNAL = {
    "lnn": "liquid_ekf", "lstm": "lstm_ekf", "transformer": "transformer_ekf",
    "ekf": "ekf", "robust_ekf": "robust_ekf",
}


def _coerce_1d_array(values, *, name: str, min_value=None):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    if min_value is not None and float(arr.min()) < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")
    return arr


def render_boxplot_F2(
    *,
    rmse_by_method: Mapping[str, Sequence[float]],
    output_path: str | Path,
    data_layer: str = "①",
    n_seeds: int = 10,
) -> dict[str, Any]:
    """F-2: 5 methods × trajectory-level RMSE distribution (box + raincloud).

    Per F-2 spec (handbook L662): 5 methods × 轨迹级 RMSE 分布。
    Per J-1..J-10: 含均值/中位数/离群标注（J-1 分布信息）、色盲安全（J-5）、
    单位 [m]（J-2）、数据层声明（J-9）、N 轨迹（J-10）、300 dpi（J-8）。
    """
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    data = []
    colors = []
    labels = []
    for method in _METHOD_ORDER:
        internal = _METHOD_INTERNAL[method]
        if internal in rmse_by_method:
            arr = _coerce_1d_array(rmse_by_method[internal], name=f"rmse_by_method[{internal}]", min_value=0.0)
            data.append(arr)
            colors.append(METHOD_COLORS[method])
            labels.append(f"{method}\n(n={arr.size})")
    if not data:
        raise ValueError("No RMSE data provided for boxplot.")
    bp = ax.boxplot(
        data, labels=labels, patch_artist=True, showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7},
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("trajectory-level RMSE [m]")     # J-2: unit [m]
    ax.set_xlabel("method")
    ax.set_title(  # J-3: self-contained caption + J-9: data layer
        f"Figure F-2: 5-Method RMSE Distribution (Box + Mean ◇) — {data_layer} {n_seeds} seeds"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=300)  # J-8: ≥300 dpi
    finally:
        plt.close(fig)
    return {"figure_path": str(out_path), "n_methods": len(data), "render_status": "ok"}


def render_4combo_error_bar_F6(
    *,
    mean_by_method_combo: Mapping[str, Mapping[str, float]],
    ci_by_method_combo: Mapping[str, Mapping[str, tuple[float, float]]] | None = None,
    output_path: str | Path,
    data_layer: str = "①",
) -> dict[str, Any]:
    """F-6: 4-combo error bar — 5 methods × 4 combos mean±CI bar (C1→C4 diagonal).

    Per F-6 spec (handbook L666): C1→C4 对角梯度可视，支撑 D16 依据。
    """
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combos = ["A2N2", "A2N3", "A3N2", "A3N3"]  # C1..C4
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    width = 0.16
    x = np.arange(len(combos))
    for i, method in enumerate(_METHOD_ORDER):
        internal = _METHOD_INTERNAL[method]
        means = [mean_by_method_combo.get(internal, {}).get(c, 0.0) for c in combos]
        cis = ci_by_method_combo.get(internal, {}) if ci_by_method_combo else {}
        lower = [means[j] - cis.get(c, (means[j], means[j]))[0] for j, c in enumerate(combos)]
        upper = [cis.get(c, (means[j], means[j]))[1] - means[j] for j, c in enumerate(combos)]
        ax.bar(
            x + (i - 2) * width, means, width,
            yerr=[lower, upper], capsize=3,
            color=METHOD_COLORS[method], alpha=0.85,
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c[1:]} ({c})" for c in combos])
    ax.set_xlabel("combo (C1→C4 diagonal gradient)")
    ax.set_ylabel("RMSE [m]")                       # J-2
    ax.set_title(                                # J-3, J-9
        f"Figure F-6: 5 Methods × 4 Combos Mean±CI — {data_layer}"
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=300)
    finally:
        plt.close(fig)
    return {"figure_path": str(out_path), "render_status": "ok"}


def render_per_scene_error_bar_F5(
    *,
    mean_by_method_scene: Mapping[str, Mapping[str, float]],
    output_path: str | Path,
    data_layer: str = "①",
) -> dict[str, Any]:
    """F-5: 5 methods × (LOS/NLOS/Missing/P95 tail) error breakdown."""
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scenes = ["LOS", "NLOS", "Missing", "P95_tail"]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    width = 0.18
    x = np.arange(len(scenes))
    for i, method in enumerate(_METHOD_ORDER):
        internal = _METHOD_INTERNAL[method]
        means = [mean_by_method_scene.get(internal, {}).get(s, 0.0) for s in scenes]
        ax.bar(x + (i - 2) * width, means, width, color=METHOD_COLORS[method], alpha=0.85, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_xlabel("scene segment")
    ax.set_ylabel("RMSE [m]")                       # J-2
    ax.set_title(f"Figure F-5: Per-Scene Error Breakdown — {data_layer}")  # J-3
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=300)
    finally:
        plt.close(fig)
    return {"figure_path": str(out_path), "render_status": "ok"}


def render_layout_randomization_F7(
    *,
    per_layout_ranking: Mapping[str, dict[str, float]],
    output_path: str | Path,
    data_layer: str = "②",
) -> dict[str, Any]:
    """F-7: Layout randomization ranking per layout (5 methods sorted)."""
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layouts = list(per_layout_ranking.keys())
    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(layouts)), 5.0))
    width = 0.14
    x = np.arange(len(layouts))
    for i, method in enumerate(_METHOD_ORDER):
        internal = _METHOD_INTERNAL[method]
        vals = [per_layout_ranking[layout].get(internal, 0.0) for layout in layouts]
        ax.bar(x + (i - 2) * width, vals, width, color=METHOD_COLORS[method], alpha=0.85, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i+1}" for i in range(len(layouts))], rotation=0)
    ax.set_xlabel(f"layout (N={len(layouts)})")
    ax.set_ylabel("RMSE [m]")                       # J-2
    ax.set_title(f"Figure F-7: Layout Randomization Ranking — {data_layer}")  # J-3
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=300)
    finally:
        plt.close(fig)
    return {"figure_path": str(out_path), "render_status": "ok"}


def render_miluv_segment_F8(
    *,
    mean_by_method_segment: Mapping[str, Mapping[str, float]],
    output_path: str | Path,
) -> dict[str, Any]:
    """F-8: MILUV segment error — methods on LOS/NLOS."""
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    segments = ["LOS", "NLOS"]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    width = 0.35
    x = np.arange(len(segments))
    for i, method in enumerate(_METHOD_ORDER):
        internal = _METHOD_INTERNAL[method]
        means = [mean_by_method_segment.get(internal, {}).get(s, 0.0) for s in segments]
        ax.bar(x + (i - 2) * width, means, width, color=METHOD_COLORS[method], alpha=0.85, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(segments)
    ax.set_xlabel("MILUV segment")
    ax.set_ylabel("RMSE [m]")                       # J-2
    ax.set_title("Figure F-8: MILUV LOS/NLOS Segment Error — ③")  # J-3, J-9
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=300)
    finally:
        plt.close(fig)
    return {"figure_path": str(out_path), "render_status": "ok"}


def build_t1_main_table(
    *,
    metrics_by_method: Mapping[str, dict[str, float]],
    pvalues_by_pair: Mapping[str, float] | None = None,
    cis_by_pair: Mapping[str, tuple[float, float]] | None = None,
    effect_sizes_by_pair: Mapping[str, float] | None = None,
    data_layer: str = "①",
) -> dict[str, Any]:
    """T-1: 主结果表 — 5 methods × 整体指标 (mean±std, P50, P95, MAE)
    + Δ vs LNN (abs m, %, 95%CI) + Wilcoxon p (Holm corrected) + 效应量。

    Per T-1 spec (handbook L648): 5 methods × overall metrics; per J-6 三线表结构。
    """
    columns = [
        "method", "mean±std [m]", "P50 [m]", "P95 [m]", "MAE [m]",
        "Δ vs LNN [m]", "Δ vs LNN [%]", "95% CI Δ", "Wilcoxon p (Holm)", "Cohen's d",
    ]
    rows: list[dict[str, Any]] = []
    lnn = metrics_by_method.get("liquid_ekf", {})
    lnn_mean = float(lnn.get("mean", 0.0))
    for method in _METHOD_ORDER:
        internal = _METHOD_INTERNAL[method]
        m = metrics_by_method.get(internal, {})
        mean = float(m.get("mean", 0.0))
        std = float(m.get("std", 0.0))
        delta_m = mean - lnn_mean
        delta_pct = (delta_m / lnn_mean * 100.0) if lnn_mean > 0 else 0.0
        row = {
            "method": method,
            "mean±std [m]": f"{mean:.3f} ± {std:.3f}",
            "P50 [m]": f"{float(m.get('p50', 0.0)):.3f}",
            "P95 [m]": f"{float(m.get('p95', 0.0)):.3f}",
            "MAE [m]": f"{float(m.get('mae', 0.0)):.3f}",
            "Δ vs LNN [m]": f"{delta_m:+.3f}",
            "Δ vs LNN [%]": f"{delta_pct:+.1f}%",
            "95% CI Δ": "—",
            "Wilcoxon p (Holm)": "—",
            "Cohen's d": "—",
        }
        if pvalues_by_pair and method != "lnn":
            row["Wilcoxon p (Holm)"] = f"{pvalues_by_pair.get(method, 1.0):.2e}"
        if cis_by_pair and method != "lnn":
            lo, hi = cis_by_pair.get(method, (0.0, 0.0))
            row["95% CI Δ"] = f"[{lo:+.3f}, {hi:+.3f}]"
        if effect_sizes_by_pair and method != "lnn":
            row["Cohen's d"] = f"{effect_sizes_by_pair.get(method, 0.0):+.3f}"
        rows.append(row)
    return {
        "table_id": "T-1",
        "title": f"Table T-1: Main Results — 5 Methods × Overall Metrics (data layer {data_layer})",
        "caption": (
            f"Mean ± std, P50, P95, MAE (all in meters) for 5 methods on {data_layer}. "
            f"Δ vs LNN computed per handbook P36. Wilcoxon p is Holm-Bonferroni corrected."
        ),
        "analysis_unit": "trajectory-level (per handbook P22)",
        "data_layer": data_layer,
        "columns": columns,
        "rows": rows,
    }


def build_t2_4combo_table(
    *,
    metrics_by_method_combo: Mapping[str, Mapping[str, dict[str, float]]],
    pvalues_by_pair_combo: Mapping[str, Mapping[str, float]] | None = None,
    data_layer: str = "①",
) -> dict[str, Any]:
    """T-2: 4-combo slice table — 5 methods × 4 combos mean±std/P50/P95.

    Per T-2 spec (handbook L649): 5 methods × (A2N2/A2N3/A3N2/A3N3) 各组合。
    """
    combos = ["A2N2", "A2N3", "A3N2", "A3N3"]
    columns = ["method"]
    for c in combos:
        columns.extend([f"{c} mean±std", f"{c} P50", f"{c} P95"])
    rows: list[dict[str, Any]] = []
    for method in _METHOD_ORDER:
        internal = _METHOD_INTERNAL[method]
        row = {"method": method}
        for c in combos:
            m = metrics_by_method_combo.get(internal, {}).get(c, {})
            row[f"{c} mean±std"] = f"{float(m.get('mean', 0.0)):.3f} ± {float(m.get('std', 0.0)):.3f}"
            row[f"{c} P50"] = f"{float(m.get('p50', 0.0)):.3f}"
            row[f"{c} P95"] = f"{float(m.get('p95', 0.0)):.3f}"
        rows.append(row)
    return {
        "table_id": "T-2",
        "title": f"Table T-2: 4-Combo Slice (C1={combos[0]} ... C4={combos[-1]}) — {data_layer}",
        "caption": (
            "Per-combo mean±std, P50, P95 for 5 methods. C1=A2N2 (light), "
            "C4=A3N3 (heavy). LNN should satisfy C4 ≥ overall and C4 ≤ 4.5m."
        ),
        "analysis_unit": "trajectory-level (per-combo)",
        "data_layer": data_layer,
        "columns": columns,
        "rows": rows,
    }


def build_t4_segment_decomposition_table(
    *,
    metrics_by_method_segment: Mapping[str, Mapping[str, dict[str, float]]],
    data_layer: str = "①",
) -> dict[str, Any]:
    """T-4: 分场景分解表 — 5 methods × (LOS/NLOS/Missing/All) error breakdown."""
    segments = ["LOS", "NLOS", "Missing", "All"]
    columns = ["method"]
    for s in segments:
        columns.extend([f"{s} mean [m]", f"{s} P95 [m]"])
    rows: list[dict[str, Any]] = []
    for method in _METHOD_ORDER:
        internal = _METHOD_INTERNAL[method]
        row = {"method": method}
        for s in segments:
            m = metrics_by_method_segment.get(internal, {}).get(s, {})
            row[f"{s} mean [m]"] = f"{float(m.get('mean', 0.0)):.3f}"
            row[f"{s} P95 [m]"] = f"{float(m.get('p95', 0.0)):.3f}"
        rows.append(row)
    return {
        "table_id": "T-4",
        "title": f"Table T-4: Scene-Segment Decomposition — {data_layer}",
        "caption": "Mean and P95 per scene segment (LOS / NLOS / Missing / All).",
        "analysis_unit": "trajectory-level (per-scene)",
        "data_layer": data_layer,
        "columns": columns,
        "rows": rows,
    }


def build_t3_anova_table(
    *,
    f_a_axis: float, p_a_axis: float, partial_eta_sq_a: float,
    f_n_axis: float, p_n_axis: float, partial_eta_sq_n: float,
    f_interaction: float, p_interaction: float, partial_eta_sq_interaction: float,
    levene_w: float | None = None, levene_p: float | None = None,
    art_backup: dict[str, Any] | None = None,
    data_layer: str = "①",
) -> dict[str, Any]:
    """T-3: 2×2 ANOVA 表 — A×N 主效应/交互 + partial η².

    Per T-3 spec (handbook L650): A×N 主效应/交互项 F、p、partial η²;
    方差齐性 Levene 检验结果; 非参数后备（ART/Kruskal-Wallis）口径（如启用）。
    """
    rows = [
        {"source": "A axis (async)", "SS": "—", "df": "1",
         "F": round(f_a_axis, 3), "p": f"{p_a_axis:.2e}",
         "partial_η²": round(partial_eta_sq_a, 4)},
        {"source": "N axis (NLOS)", "SS": "—", "df": "1",
         "F": round(f_n_axis, 3), "p": f"{p_n_axis:.2e}",
         "partial_η²": round(partial_eta_sq_n, 4)},
        {"source": "A × N interaction", "SS": "—", "df": "1",
         "F": round(f_interaction, 3), "p": f"{p_interaction:.2e}",
         "partial_η²": round(partial_eta_sq_interaction, 4)},
    ]
    if levene_w is not None and levene_p is not None:
        rows.append({
            "source": "Levene (variance homogeneity)",
            "F": round(levene_w, 3),
            "p": f"{levene_p:.2e}",
        })
    if art_backup is not None:
        rows.append({"source": "ART/Kruskal-Wallis backup",
                     "details": art_backup})
    return {
        "table_id": "T-3",
        "title": f"Table T-3: 2×2 ANOVA — A × N Interaction — {data_layer}",
        "caption": "Two-way ANOVA on A × N factorial design. F-stats with partial η² effect size.",
        "analysis_unit": "trajectory-level (per method×combo)",
        "data_layer": data_layer,
        "rows": rows,
    }


def build_t5_layout_randomization_table(
    *,
    per_layout_method_rmse: Mapping[str, dict[str, float]],
    kendall_tau: float | None = None,
    stability_rate: float | None = None,
    data_layer: str = "②",
) -> dict[str, Any]:
    """T-5: 布局随机化稳定率表 — 每布局 × 方法排序 + Kendall τ + 跨布局稳定率.

    Per T-5 spec (handbook L652): per-layout × method ranking, Kendall τ, 稳定率。
    """
    methods = ["liquid_ekf", "lstm_ekf", "transformer_ekf", "ekf", "robust_ekf"]
    cols = ["layout_id"] + methods
    rows: list[dict[str, Any]] = []
    for layout_id, by_method in per_layout_method_rmse.items():
        row = {"layout_id": layout_id}
        for m in methods:
            row[m] = round(float(by_method.get(m, 0.0)), 4)
        rows.append(row)
    summary: dict[str, Any] = {
        "n_layouts": len(per_layout_method_rmse),
    }
    if kendall_tau is not None:
        summary["kendall_tau"] = round(float(kendall_tau), 4)
    if stability_rate is not None:
        summary["stability_rate_pct"] = round(float(stability_rate) * 100, 1)
    return {
        "table_id": "T-5",
        "title": f"Table T-5: Layout Randomization Stability — {data_layer}",
        "caption": "Per-layout RMSE for 5 methods + Kendall τ cross-layout stability.",
        "analysis_unit": "layout-level",
        "data_layer": data_layer,
        "columns": cols,
        "rows": rows,
        "summary": summary,
    }


def build_t6_miluv_table(
    *,
    per_method_overall_and_segments: Mapping[str, dict[str, dict[str, float]]],
    n_independent_experiments: int = 36,
    data_layer: str = "③",
) -> dict[str, Any]:
    """T-6: MILUV 真实域评估表 — 36 次独立实验 (handbook L653)."""
    methods = ["liquid_ekf", "lstm_ekf", "transformer_ekf", "ekf", "robust_ekf"]
    segments = ["overall", "LOS", "NLOS"]
    cols = ["method"] + [f"{s} mean [m]" for s in segments] + [f"{s} P95 [m]" for s in segments]
    rows: list[dict[str, Any]] = []
    for m in methods:
        row: dict[str, Any] = {"method": m.replace("_ekf", "").upper()}
        d = per_method_overall_and_segments.get(m, {})
        for s in segments:
            row[f"{s} mean [m]"] = f"{float(d.get(s, {}).get('mean', 0.0)):.3f}"
            row[f"{s} P95 [m]"] = f"{float(d.get(s, {}).get('p95', 0.0)):.3f}"
        rows.append(row)
    return {
        "table_id": "T-6",
        "title": f"Table T-6: MILUV Real-Domain Evaluation — {data_layer}",
        "caption": f"Method comparison on MILUV dataset, n_independent={n_independent_experiments}",
        "analysis_unit": "experiment-level (each MILUV run is one unit)",
        "data_layer": data_layer,
        "n_independent_experiments": n_independent_experiments,
        "columns": cols,
        "rows": rows,
    }


def build_t7_seed_hyperparam_robustness_table(
    *,
    per_seed_method_rmse: Mapping[int, dict[str, float]],
    per_seed_lnn_lowest: Mapping[int, bool],
    hyperparam_perturbation_directions: list[dict[str, Any]] | None = None,
    data_layer: str = "①",
) -> dict[str, Any]:
    """T-7: seed/超参稳健性表 — 10 seed "LNN 最低" 成立频次 + 超参 ±1 扰动 (handbook L654)."""
    cols = ["seed_id"] + ["liquid_ekf", "lstm_ekf", "transformer_ekf", "ekf", "robust_ekf", "lnn_lowest"]
    rows: list[dict[str, Any]] = []
    for seed_id, by_method in per_seed_method_rmse.items():
        row: dict[str, Any] = {"seed_id": int(seed_id)}
        for m in ("liquid_ekf", "lstm_ekf", "transformer_ekf", "ekf", "robust_ekf"):
            row[m] = round(float(by_method.get(m, 0.0)), 4)
        row["lnn_lowest"] = bool(per_seed_lnn_lowest.get(seed_id, False))
        rows.append(row)
    lnn_lowest_count = sum(1 for v in per_seed_lnn_lowest.values() if v)
    summary = {
        "n_seeds": len(per_seed_method_rmse),
        "lnn_lowest_count": lnn_lowest_count,
        "lnn_lowest_rate_pct": round(lnn_lowest_count / max(1, len(per_seed_method_rmse)) * 100, 1),
    }
    if hyperparam_perturbation_directions:
        summary["hyperparam_perturbation"] = hyperparam_perturbation_directions
    return {
        "table_id": "T-7",
        "title": f"Table T-7: Seed × Hyperparam Robustness — {data_layer}",
        "caption": "Per-seed RMSE for 5 methods + LNN-lowest flag + hyperparam perturbation robustness.",
        "analysis_unit": "seed-level",
        "data_layer": data_layer,
        "columns": cols,
        "rows": rows,
        "summary": summary,
    }


def build_t8_reproducibility_table(
    *,
    r0_rz0_archive: dict[str, Any],
    r1_rz1_manifest: dict[str, Any],
    r2_rz2_retrain_log: dict[str, Any],
    r3_rz3_acceptance: dict[str, Any],
    per_unit_hash: list[dict[str, str]] | None = None,
    data_layer: str = "全部（数据与执行纯净性）",
) -> dict[str, Any]:
    """T-8: 可复现性附录表 — RZ 四 gate 留痕汇总 + 配置 hash/seed/git commit (handbook L655)."""
    rows: list[dict[str, Any]] = []
    for gate_name, info in [
        ("RZ-0 (清场)", r0_rz0_archive),
        ("RZ-1 (重生成)", r1_rz1_manifest),
        ("RZ-2 (重训)", r2_rz2_retrain_log),
        ("RZ-3 (验收)", r3_rz3_acceptance),
    ]:
        rows.append({
            "gate": gate_name,
            "timestamp": info.get("timestamp", "—"),
            "passed": bool(info.get("passed", False)),
            "evidence": info.get("evidence_path") or info.get("note") or "—",
        })
    summary: dict[str, Any] = {"rz_gates": rows}
    if per_unit_hash:
        summary["per_unit_hash_table"] = per_unit_hash
    return {
        "table_id": "T-8",
        "title": f"Table T-8: Reproducibility Appendix — {data_layer}",
        "caption": "RZ gate traces + per-unit config_hash / seed / git commit.",
        "analysis_unit": "appendix",
        "data_layer": data_layer,
        "columns": ["gate", "timestamp", "passed", "evidence"],
        "rows": rows,
        "summary": summary,
    }


def _iter_scene_segments(scene_mask: np.ndarray):
    """Yield (start, end, label) for consecutive equal segments of scene_mask."""
    if scene_mask.size == 0:
        return
    start = 0
    current = int(scene_mask[0])
    for i in range(1, scene_mask.size):
        if int(scene_mask[i]) != current:
            yield start, i, current
            start = i
            current = int(scene_mask[i])
    yield start, scene_mask.size, current


def render_f4_activation_heatmap(
    *,
    activation_by_head: Mapping[str, np.ndarray],
    nlos_mask: np.ndarray | None = None,
    scene_mask: np.ndarray | None = None,
    method: str = "liquid_ekf",
    output_path: str | Path | None = None,
    data_layer: str = "①",
) -> str:
    """F-4: 激活热图对齐图 — 4 头激活 vs scene_mask (handbook F-4 L664).

    Per F-4 spec: 四头激活 (risk/bias/uwb_scaling/vio_scaling) 热图沿轨迹画,
    叠加 NLOS 遮挡区标签, 展示 risk/bias 激活落在 NLOS 段。

    Args:
        activation_by_head: dict[head_name, np.ndarray] - 4 个头的激活值序列
        nlos_mask: NLOS 段二值掩码 (0=LOS, 1=NLOS)
        scene_mask: 场景阶段掩码 (0=startup, 1=normal, 2=NLOS, 3=missing)
        output_path: PNG 输出路径
    """
    out = Path(output_path or _OUT_DIR / f"F-4_{method}_activation_heatmap.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    heads = ("risk", "bias", "uwb_scaling", "vio_scaling")
    T = max((v.size for v in activation_by_head.values()), default=0)
    if T == 0:
        raise ValueError("activation_by_head must contain at least one non-empty array")
    t = np.arange(T)
    fig, axes = plt.subplots(len(heads) + 1, 1, figsize=(12.0, 8.0), sharex=True, constrained_layout=True)
    # Top: NLOS band
    if nlos_mask is not None and nlos_mask.size > 0:
        axes[0].fill_between(
            t, 0, 1,
            where=np.asarray(nlos_mask)[:T] > 0,
            color="red", alpha=0.3, label="NLOS region",
        )
        axes[0].set_ylim(0, 1)
        axes[0].set_yticks([])
        axes[0].legend(loc="upper right", fontsize=8)
    else:
        axes[0].axis("off")
    axes[0].set_ylabel("NLOS")
    axes[0].set_title(f"F-4 {method} 4-head activation heatmap vs scene_mask — {data_layer}")
    # Heads
    vmax = max(1e-6, max(v.max() for v in activation_by_head.values())) if activation_by_head else 1e-6
    for i, head in enumerate(heads):
        ax = axes[i + 1]
        arr = np.asarray(activation_by_head.get(head, np.zeros(T)))[:T]
        im = ax.imshow(
            arr[np.newaxis, :], aspect="auto", cmap="viridis",
            extent=(0, T, 0, 1), vmin=0.0, vmax=vmax,
        )
        ax.set_yticks([])
        ax.set_ylabel(head, rotation=0, ha="right", fontsize=8)
        if i == len(heads) - 1:
            ax.set_xlabel("frame index (10 Hz)")
    # Scene phase labels (if provided)
    if scene_mask is not None and scene_mask.size > 0:
        scene_names = {0: "startup", 1: "normal", 2: "NLOS", 3: "missing"}
        for start, end, label in _iter_scene_segments(np.asarray(scene_mask)[:T]):
            axes[0].axvspan(start, end, color="gray", alpha=0.08)
    fig.savefig(out, dpi=300)   # J-8: dpi ≥ 300
    plt.close(fig)
    return str(out)


__all__ += ["render_f4_activation_heatmap"]
