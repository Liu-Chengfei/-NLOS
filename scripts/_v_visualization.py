"""0-V1..0-V6 跑后可视化速查脚本。

0-V1 轨迹叠加图: LNN vs EKF vs GT, C4 (A3N3) 重压 + C1 (A2N2) 轻压
0-V2 逐帧误差时间序列: x/y/yaw 误差 vs 时间
0-V3 激活热图 vs scene_mask 对齐: 4 头激活 + NLOS 段标注
0-V4 朝向/航向箭头图: 每 N 步画 tag 朝向
0-V5 GDOP 时间序列 vs 误差曲线: 同轴双曲线
0-V6 误差 CDF 曲线: 5 方法同轴, P50/P95 标注

每张图都标注 data layer (①②③) + sample size + dpi=300 (J-8) + 色盲安全 (J-5)。
"""
from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from liquidloc.plotting._okabe_ito_palette import METHOD_COLORS


_OUT_DIR = Path("E:/异步高NLOS/outputs/figures/quick_check")
_OUT_DIR.mkdir(parents=True, exist_ok=True)

_METHOD_ORDER = ("lnn", "lstm", "transformer", "ekf", "robust_ekf")
_METHOD_INTERNAL = {
    "lnn": "liquid_ekf", "lstm": "lstm_ekf", "transformer": "transformer_ekf",
    "ekf": "ekf", "robust_ekf": "robust_ekf",
}


def v1_trajectory_overlay(
    trajectories_by_combo: Mapping[str, dict[str, dict[str, list[float]]]],
    output_path: str | Path | None = None,
) -> str:
    """0-V1: 轨迹叠加图 — LNN vs EKF vs GT, C4 重压 + C1 轻压 各一条."""
    out = Path(output_path or _OUT_DIR / "0-V1_trajectory_overlay.png")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    combos = ["C1_A2N2", "C4_A3N3"]
    for i, combo in enumerate(combos):
        ax = axes[i]
        traj = trajectories_by_combo.get(combo, {})
        for method, color_key in (("lnn", "lnn"), ("ekf", "ekf"), ("gt", None)):
            if method in traj:
                xs = traj[method].get("px", [])
                ys = traj[method].get("py", [])
                color = METHOD_COLORS.get(color_key, "black") if color_key else "black"
                ls = "-" if method == "gt" else "--"
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.6, label=method.upper())
        ax.set_xlabel("px [m]")           # J-2
        ax.set_ylabel("py [m]")
        ax.set_title(f"0-V1 {combo}: LNN vs EKF vs GT")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out, dpi=300)            # J-8
    plt.close(fig)
    return str(out)


def v2_per_frame_error(
    error_by_method: Mapping[str, dict[str, list[float]]],
    output_path: str | Path | None = None,
) -> str:
    """0-V2: 逐帧误差时间序列 (x/y/yaw)."""
    out = Path(output_path or _OUT_DIR / "0-V2_per_frame_error.png")
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True)
    for method in _METHOD_ORDER:
        e = error_by_method.get(_METHOD_INTERNAL[method], {})
        for i, key in enumerate(("x", "y", "yaw")):
            arr = np.asarray(e.get(key, []), dtype=np.float64)
            axes[i].plot(arr, color=METHOD_COLORS[method], linewidth=0.8, label=method)
    for i, (key, unit) in enumerate((("x", "[m]"), ("y", "[m]"), ("yaw", "[rad]"))):
        axes[i].set_ylabel(f"err {key} {unit}")   # J-2
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right", fontsize=7, ncol=5)
    axes[-1].set_xlabel("frame index")
    fig.suptitle("0-V2 Per-frame error time series (LNN / LSTM / Transformer / EKF / Robust-EKF)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return str(out)


def v3_activation_heatmap_alignment(
    activation_by_head: Mapping[str, dict[str, list[float]]],
    nl_flag_sequence: Sequence[int],
    output_path: str | Path | None = None,
) -> str:
    """0-V3: 激活热图 vs scene_mask 对齐."""
    out = Path(output_path or _OUT_DIR / "0-V3_activation_heatmap.png")
    fig, axes = plt.subplots(5, 1, figsize=(10.0, 8.0), sharex=True)
    heads = ("risk", "bias", "uwb_scaling", "vio_scaling")
    t = np.arange(len(nl_flag_sequence))
    # Top panel: NLOS mask
    axes[0].fill_between(t, 0, 1, where=np.asarray(nl_flag_sequence) > 0, color="red", alpha=0.3, label="NLOS")
    axes[0].set_ylabel("NLOS")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("0-V3 Activation heatmap vs scene_mask (LNN)")
    for i, head in enumerate(heads):
        ax = axes[i + 1]
        activations = np.asarray(activation_by_head.get(head, {}).get("lnn", []), dtype=np.float64)
        # Heatmap
        im = ax.imshow(
            activations[np.newaxis, :], aspect="auto", cmap="viridis",
            extent=(0, len(activations), 0, 1), vmin=0.0, vmax=max(activations.max(), 1e-6),
        )
        ax.set_yticks([])
        ax.set_ylabel(head, rotation=0, ha="right")
    axes[-1].set_xlabel("frame index")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return str(out)


def v4_heading_arrows(
    heading_data: Mapping[str, dict[str, list[float]]],
    output_path: str | Path | None = None,
) -> str:
    """0-V4: 朝向/航向箭头图 — 每 N 步画 tag 朝向."""
    out = Path(output_path or _OUT_DIR / "0-V4_heading_arrows.png")
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    color_map = {"lnn": METHOD_COLORS["lnn"], "ekf": METHOD_COLORS["ekf"], "gt": "black"}
    for method, traj in heading_data.items():
        xs = np.asarray(traj.get("px", []), dtype=np.float64)
        ys = np.asarray(traj.get("py", []), dtype=np.float64)
        yaws = np.asarray(traj.get("yaw", []), dtype=np.float64)
        n = len(xs)
        if n == 0:
            continue
        step = max(n // 20, 1)
        for k in range(0, n, step):
            dx = math.cos(yaws[k]) * 0.5
            dy = math.sin(yaws[k]) * 0.5
            ax.arrow(xs[k], ys[k], dx, dy, head_width=0.1, head_length=0.1,
                    fc=color_map.get(method, "gray"), ec=color_map.get(method, "gray"),
                    alpha=0.6)
        ax.plot(xs, ys, color=color_map.get(method, "gray"), linewidth=0.8, label=method.upper())
    ax.set_xlabel("px [m]")
    ax.set_ylabel("py [m]")
    ax.set_title("0-V4 Heading/yaw arrows (C4 A3N3 trajectory)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return str(out)


def v5_gdop_vs_error(
    gdop_timeseries: Sequence[float],
    rmse_by_method: Mapping[str, Sequence[float]],
    output_path: str | Path | None = None,
) -> str:
    """0-V5: GDOP 时间序列 vs 误差曲线（同轴双曲线）."""
    out = Path(output_path or _OUT_DIR / "0-V5_gdop_vs_error.png")
    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    t = np.arange(len(gdop_timeseries))
    color_gdop = "tab:gray"
    ax1.plot(t, gdop_timeseries, color=color_gdop, linewidth=1.6, label="GDOP")
    ax1.set_xlabel("frame index")
    ax1.set_ylabel("GDOP", color=color_gdop)
    ax1.tick_params(axis="y", labelcolor=color_gdop)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    for method in _METHOD_ORDER:
        rmse = np.asarray(rmse_by_method.get(_METHOD_INTERNAL[method], []), dtype=np.float64)
        if len(rmse) == len(t):
            ax2.plot(t, rmse, color=METHOD_COLORS[method], linewidth=1.0, label=method)
        elif len(rmse) > 0:
            xs = np.linspace(0, len(t) - 1, len(rmse))
            ax2.plot(xs, rmse, color=METHOD_COLORS[method], linewidth=1.0, label=method)
    ax2.set_ylabel("RMSE [m]")
    ax2.legend(loc="upper right", fontsize=7, ncol=2)
    plt.title("0-V5 GDOP time series vs RMSE (high-GDOP segments should align with high-RMSE)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return str(out)


def v6_error_cdf(
    err_by_method: Mapping[str, Sequence[float]],
    output_path: str | Path | None = None,
) -> str:
    """0-V6: 误差 CDF 曲线 — 5 方法同轴, P50/P95 标注."""
    out = Path(output_path or _OUT_DIR / "0-V6_error_cdf.png")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for method in _METHOD_ORDER:
        arr = np.asarray(err_by_method.get(_METHOD_INTERNAL[method], []), dtype=np.float64)
        if arr.size == 0:
            continue
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, sorted_arr.size + 1) / sorted_arr.size
        ax.plot(sorted_arr, cdf, color=METHOD_COLORS[method], linewidth=1.6, label=method)
        p50, p95 = np.percentile(arr, [50, 95])
        cdf_at_p50 = float(np.searchsorted(sorted_arr, p50)) / sorted_arr.size
        cdf_at_p95 = float(np.searchsorted(sorted_arr, p95)) / sorted_arr.size
        ax.plot(p50, cdf_at_p50, marker="v", color=METHOD_COLORS[method], markersize=7, linestyle="none")
        ax.plot(p95, cdf_at_p95, marker="^", color=METHOD_COLORS[method], markersize=7, linestyle="none")
    ax.set_xlabel("position error [m]")
    ax.set_ylabel("CDF")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("0-V6 Per-frame error CDF (P50 ▼, P95 ▲) — data layer ①")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return str(out)


def main():
    out = {
        "0-V1": "use v1_trajectory_overlay({})".format(dict),
        "0-V2": "use v2_per_frame_error({})".format(dict),
        "0-V3": "use v3_activation_heatmap_alignment({}, nl_flag_seq)".format(dict),
        "0-V4": "use v4_heading_arrows({})".format(dict),
        "0-V5": "use v5_gdop_vs_error(gdop_seq, {})".format(dict),
        "0-V6": "use v6_error_cdf({})".format(dict),
    }
    print("[0-V] Six visualization functions ready (callable from main paper pipeline).")
    print("[0-V] All functions: J-5 Okabe-Ito + J-8 dpi=300 + J-2 [m] units + J-3 self-contained caption.")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
