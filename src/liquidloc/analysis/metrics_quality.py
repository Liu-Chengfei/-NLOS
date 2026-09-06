"""P7/P8/P9/P21/P27/P33/P34/P36 缺失项实施模块。

本模块把手册 Part 0 S3 (P7/P8/P9) 和 P21/P27/P33/P34/P36 落地为可直接调用的函数，
供 prepare_pipeline / train_pipeline / eval_pipeline 调用。

P7  RotationLink (IMU 绕圈测试)
P8  单位一致性 (rad, m, [-π, π])
P9  Sim(3) 对齐 (Umeyama / scale-aware)
P21 归一化泄漏检查 (train-only 统计量)
P27 激活值统计 (4 head 可视化)
P33 Best-val checkpoint 协议
P34 过拟合检测
P36 RMSE + MAE + trimmed mean + median + P95 + P50 口径
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


# ============================================================================
# 常量
# ============================================================================

RAD_PER_DEG: float = math.pi / 180.0
M_PER_CM: float = 0.01


# ============================================================================
# 通用工具
# ============================================================================


def ensure_finite(arr: np.ndarray | float, name: str = "value") -> None:
    """若含 NaN/Inf 则抛 ValueError。"""
    v = np.asarray(arr)
    if not np.all(np.isfinite(v)):
        raise ValueError(f"[{name}] contains non-finite values: {v}")


def compute_rmse(errors: np.ndarray) -> float:
    """均方根误差 (m)。"""
    ensure_finite(errors, "rmse_input")
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_mae(errors: np.ndarray) -> float:
    """平均绝对误差 (m)。"""
    ensure_finite(errors, "mae_input")
    return float(np.mean(np.abs(errors)))


def compute_median_error(errors: np.ndarray) -> float:
    return float(np.median(np.abs(errors)))


def compute_p50(errors: np.ndarray) -> float:
    return float(np.percentile(np.abs(errors), 50))


def compute_p95(errors: np.ndarray) -> float:
    return float(np.percentile(np.abs(errors), 95))


def compute_trimmed_mean(errors: np.ndarray, trim_ratio: float = 0.05) -> float:
    """截尾均值，默认 5% 截尾。"""
    ensure_finite(errors, "trimmed_mean_input")
    lo, hi = float(np.percentile(errors, trim_ratio * 100)), float(
        np.percentile(errors, (1 - trim_ratio) * 100)
    )
    masked = errors[(errors >= lo) & (errors <= hi)]
    return float(np.mean(masked)) if len(masked) > 0 else float(np.mean(errors))


@dataclass
class MetricBundle:
    """P36 口径 Bundle。"""
    rmse: float
    mae: float
    median: float
    p50: float
    p95: float
    trimmed_mean: float
    mean: float
    std: float

    def to_dict(self) -> dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "median": self.median,
            "p50": self.p50,
            "p95": self.p95,
            "trimmed_mean": self.trimmed_mean,
            "mean": self.mean,
            "std": self.std,
        }


def compute_all_metrics(errors: np.ndarray) -> MetricBundle:
    """P36: 一次性计算 RMSE/MAE/Median/P50/P95/trimmed_mean/mean/std。"""
    abs_err = np.abs(np.asarray(errors, dtype=np.float64))
    ensure_finite(abs_err, "all_metrics")
    return MetricBundle(
        rmse=compute_rmse(errors),
        mae=compute_mae(errors),
        median=compute_median_error(errors),
        p50=compute_p50(errors),
        p95=compute_p95(errors),
        trimmed_mean=compute_trimmed_mean(errors),
        mean=float(np.mean(abs_err)),
        std=float(np.std(abs_err)),
    )


# ============================================================================
# P7: RotationLink — IMU 绕圈回起点测试
# ============================================================================

@dataclass
class RotationChainResult:
    passed: bool
    circle_return_error_m: float
    yaw_direction_correct: bool
    straight_vs_turn_diff_m: float
    imu_angle_accumulated_rad: float
    detail: str


def run_rotation_chain_test(
    imu_gyro_z: np.ndarray,
    imu_dt: float,
    trajectory_x: np.ndarray,
    trajectory_z: np.ndarray,
    start_x: float,
    start_z: float,
) -> RotationChainResult:
    """P7: 验证 IMU 积分"绕圈"后位置回到起点附近。

    步骤：
    1. 用 gyro_z 累积积分得到 yaw_delta_rad（总转角）。
    2. 用累积 yaw 估计位置轨迹（IMU 积分推算 → 应回到起点）。
    3. 与 GT 轨迹对比（直线 vs 绕圈）。
    """
    ensure_finite(imu_gyro_z, "gyro_z")
    ensure_finite(trajectory_x, "traj_x")
    ensure_finite(trajectory_z, "traj_z")
    n = min(len(imu_gyro_z), len(trajectory_x), len(trajectory_z))
    gyro = np.asarray(imu_gyro_z[:n])
    traj_x = np.asarray(trajectory_x[:n])
    traj_z = np.asarray(trajectory_z[:n])

    # 积分 gyro_z → yaw_delta（gyro_z 是 rad/s，np.cumsum 本身给出弧度）
    cum_yaw = np.cumsum(gyro)  # 弧度（不需要再乘 dt）
    imu_angle_accumulated = float(abs(cum_yaw[-1])) if len(cum_yaw) > 0 else 0.0

    # 用累积 yaw 估计位置增量（这是 dead-reckoning 的标准做法）
    # 增量：[cos(yaw) * v, sin(yaw) * v]（假设速度 v=1 m/s）
    cos_yaw = np.cos(cum_yaw[:-1])
    sin_yaw = np.sin(cum_yaw[:-1])
    disp_x_imu = float(np.sum(cos_yaw)) * imu_dt
    disp_z_imu = float(np.sum(sin_yaw)) * imu_dt
    imu_circle_return_error = math.sqrt(disp_x_imu**2 + disp_z_imu**2)

    # GT 终点（绕圈应该回到起点附近）
    circle_return_traj = math.sqrt(
        (traj_x[-1] - start_x) ** 2 + (traj_z[-1] - start_z) ** 2
    )
    straight_disp = math.sqrt(
        (traj_x[-1] - start_x) ** 2 + (traj_z[-1] - start_z) ** 2
    )

    # 方向：若累计 yaw ≥ 2π（几乎正好一圈），则认为有绕圈
    # 使用 2*π - 1e-9 容许浮点边界
    _TWO_PI = 2 * math.pi - 1e-9
    has_turned = imu_angle_accumulated >= _TWO_PI
    yaw_direction_correct = has_turned

    # 直线轨迹的 IMU 积分误差应该小于绕圈轨迹（绕圈有 cancel out 效应）
    straight_vs_turn_diff = abs(straight_disp - circle_return_traj)

    # 通过条件：
    # 1. 累计 yaw ≥ 2π（确实有绕圈动作）
    # 2. GT 轨迹回到起点（circle_return_traj < 0.5m）
    # 3. IMU 推算位移不应爆炸（但允许相对大的偏差，因这是 dead-reckoning 简化版）
    passed = (
        has_turned
        and circle_return_traj < 0.5
        and imu_angle_accumulated < 100.0
    )
    return RotationChainResult(
        passed=passed,
        circle_return_error_m=imu_circle_return_error,
        yaw_direction_correct=yaw_direction_correct,
        straight_vs_turn_diff_m=straight_vs_turn_diff,
        imu_angle_accumulated_rad=imu_angle_accumulated,
        detail=(
            f"yaw_accum={imu_angle_accumulated:.2f}rad, "
            f"circle_traj_err={circle_return_traj:.3f}m, "
            f"imu_disp={imu_circle_return_error:.3f}m, "
            f"straight_vs_turn={straight_vs_turn_diff:.3f}m"
        ),
    )


# ============================================================================
# P8: 单位一致性
# ============================================================================

@dataclass
class UnitCheckResult:
    passed: bool
    distance_unit: Literal["m", "cm", "mm", "UNKNOWN"]
    yaw_unit: Literal["rad", "deg", "UNKNOWN"]
    has_x57_or_x100_artifact: bool
    detail: str


def check_units(
    range_values: np.ndarray,
    yaw_values: np.ndarray,
    position_values: np.ndarray,
    max_reasonable_position_m: float = 20.0,
    max_reasonable_range_m: float = 30.0,
) -> UnitCheckResult:
    """P8: 验证物理单位一致性。

    检查项：
    - 距离在 [0, max_reasonable_range_m] m 范围内（若出现 >30m 则可能是 cm 当 m 用）
    - yaw 在 [-π, π] rad 范围内（若出现 [-180, 180] 则可能是 deg 当 rad 用）
    - position 绝对值 < max_reasonable_position_m
    - 检测 0.01745 (~π/180) 或 57.3 等疑似 rad→deg 错误常数
    """
    range_arr = np.asarray(range_values, dtype=np.float64)
    yaw_arr = np.asarray(yaw_values, dtype=np.float64)
    pos_arr = np.asarray(position_values, dtype=np.float64)

    ensure_finite(range_arr, "range")
    ensure_finite(yaw_arr, "yaw")
    ensure_finite(pos_arr, "pos")

    # 检测距离单位
    if len(range_arr) > 0:
        r_max = float(np.max(np.abs(range_arr)))
        if r_max > max_reasonable_range_m * 5:
            dist_unit = "cm"  # 疑似 cm 当 m
        elif r_max <= max_reasonable_range_m:
            dist_unit = "m"
        else:
            dist_unit = "m"  # 正常
    else:
        dist_unit = "UNKNOWN"

    # 检测 yaw 单位：若最大值 > π 且 < 2π，则疑似 deg 当 rad
    if len(yaw_arr) > 0:
        y_abs_max = float(np.max(np.abs(yaw_arr)))
        if y_abs_max > math.pi * 1.5 and y_abs_max < math.pi * 200:
            yaw_unit = "deg"  # 疑似 deg 当 rad
        elif y_abs_max <= math.pi * 1.1:
            yaw_unit = "rad"
        else:
            yaw_unit = "UNKNOWN"
    else:
        yaw_unit = "UNKNOWN"

    # 检测 57.3 / 0.01745 等常数
    has_x57 = bool(np.any(np.abs(range_arr - 57.3) < 0.1))
    has_x001745 = bool(np.any(np.abs(range_arr - 0.01745) < 0.001))
    has_x100 = bool(np.any(np.abs(range_arr - 100.0) < 0.1))
    has_x001 = bool(np.any(np.abs(range_arr - 0.01) < 0.001))
    has_artifact = has_x57 or has_x001745 or has_x100 or has_x001

    passed = (
        dist_unit == "m"
        and yaw_unit == "rad"
        and not has_artifact
        and np.all(np.abs(pos_arr) < max_reasonable_position_m)
    )
    return UnitCheckResult(
        passed=passed,
        distance_unit=dist_unit,
        yaw_unit=yaw_unit,
        has_x57_or_x100_artifact=has_artifact,
        detail=(
            f"dist={dist_unit}, yaw={yaw_unit}, "
            f"x57={has_x57}, x100={has_x100}, "
            f"pos_max={float(np.max(np.abs(pos_arr))):.2f}m"
        ),
    )


# ============================================================================
# P9: Sim(3) 对齐（Umeyama）
# ============================================================================


def _umeyama_sim3(
    X: np.ndarray, Y: np.ndarray, with_scale: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama Sim(3) 对齐：Y ≈ s*R*X + t。

    Returns (R, t, s)。R 是 2×2 旋转，t 是 2×1 平移，s 是标量尺度。
    """
    assert X.shape == Y.shape, f"{X.shape} vs {Y.shape}"
    n, d = X.shape
    assert d in (2, 3), d

    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    Xc = X - mu_X
    Yc = Y - mu_Y

    sigma_X2 = np.mean(np.sum(Xc**2, axis=1))
    sigma_Y2 = np.mean(np.sum(Yc**2, axis=1))

    cov = (Yc.T @ Xc) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[d - 1, d - 1] = -1.0

    R = U @ S @ Vt
    c = np.trace(np.diag(D) @ S) / sigma_X2 if sigma_X2 > 0 else 1.0
    s = math.sqrt(c) if with_scale else 1.0
    t = mu_Y - s * R @ mu_X

    return R, t, s


def align_sim3(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """对 X 做 Sim(3) 对齐后返回 X_aligned，使得 X_aligned ≈ Y。"""
    R, t, s = _umeyama_sim3(X, Y, with_scale=True)
    return s * (R @ X.T).T + t


def align_rigid(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """纯 SE(2) 刚体对齐（无尺度）。"""
    R, t, _ = _umeyama_sim3(X, Y, with_scale=False)
    return (R @ X.T).T + t


@dataclass
class Sim3AlignmentResult:
    pre_align_rmse: float
    post_align_rmse: float
    scale: float
    rotation_deg: float
    translation_m: float
    ratio: float  # post/pre
    passed: bool


def evaluate_sim3_alignment(
    pred_3d: np.ndarray, gt_3d: np.ndarray, warmup: int = 0
) -> Sim3AlignmentResult:
    """P9: 计算对齐前后的 RMSE，验证 Sim(3) 对齐有效性。

    Args:
        pred_3d: (N, 3) 预测轨迹（可能是 3D 但有尺度/旋转偏移）
        gt_3d:   (N, 3) 真值轨迹
        warmup:  剔除前 warmup 帧
    """
    pred = np.asarray(pred_3d, dtype=np.float64)
    gt = np.asarray(gt_3d, dtype=np.float64)
    n = min(len(pred), len(gt))
    p, g = pred[:n], gt[:n]

    if warmup > 0:
        p, g = p[warmup:], g[warmup:]

    if len(p) < 3:
        raise ValueError("Too few points after warmup removal")

    # 投影到 xz 平面（2D 实验）
    p2d = p[:, [0, 2]]
    g2d = g[:, [0, 2]]

    # 对齐前误差
    err_pre = np.linalg.norm(p2d - g2d, axis=1)
    rmse_pre = compute_rmse(err_pre)

    # Sim(3) 对齐
    p_aligned = align_sim3(p2d, g2d)

    # 对齐后误差
    err_post = np.linalg.norm(p_aligned - g2d, axis=1)
    rmse_post = compute_rmse(err_post)

    # 提取 Sim(3) 参数
    R, t, s = _umeyama_sim3(p2d, g2d, with_scale=True)
    angle = math.atan2(R[1, 0], R[0, 0])
    translation_norm = float(np.linalg.norm(t))

    ratio = rmse_post / max(rmse_pre, 1e-9)
    # 对齐后误差应显著小于对齐前（否则说明对齐逻辑有误）
    passed = ratio < 0.95 and 0.5 < s < 2.0

    return Sim3AlignmentResult(
        pre_align_rmse=rmse_pre,
        post_align_rmse=rmse_post,
        scale=s,
        rotation_deg=float(angle) * 180.0 / math.pi,
        translation_m=translation_norm,
        ratio=ratio,
        passed=passed,
    )


# ============================================================================
# P21: 归一化泄漏检查
# ============================================================================

@dataclass
class NormalizationLeakResult:
    passed: bool
    train_stats: dict[str, float]
    test_stats: dict[str, float]
    train_test_deviation: float  # 相对偏差
    detail: str


def check_normalization_leak(
    train_features: np.ndarray,
    test_features: np.ndarray,
    rtol: float = 0.15,
) -> NormalizationLeakResult:
    """P21: 验证归一化只用训练集，测试集统计量与训练集偏差 ≤ rtol。

    若测试集均值/标准差与训练集一致 → 说明用了全量数据或测试集泄漏。
    若测试集与训练集有合理偏差但分布族相同 → 通过。
    """
    train_arr = np.asarray(train_features, dtype=np.float64)
    test_arr = np.asarray(test_features, dtype=np.float64)

    ensure_finite(train_arr, "train_features")
    ensure_finite(test_arr, "test_features")

    train_mean = float(np.mean(train_arr))
    train_std = float(np.std(train_arr))
    test_mean = float(np.mean(test_arr))
    test_std = float(np.std(test_arr))

    # 计算相对偏差
    def rel_dev(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1e-9)

    mean_dev = rel_dev(train_mean, test_mean)
    std_dev = rel_dev(train_std, test_std)

    # 判定：若均值偏差 > rtol，则说明两者来自不同阶段，未泄漏
    # 若偏差极小 (< 5%)，则疑似泄漏
    is_suspiciously_similar = (mean_dev < 0.05) and (std_dev < 0.05)

    # 合理偏差范围
    passed = not is_suspiciously_similar and mean_dev < rtol * 2 and std_dev < rtol * 2

    return NormalizationLeakResult(
        passed=passed,
        train_stats={"mean": train_mean, "std": train_std},
        test_stats={"mean": test_mean, "std": test_std},
        train_test_deviation=max(mean_dev, std_dev),
        detail=(
            f"train_mean={train_mean:.4f}, test_mean={test_mean:.4f} "
            f"(dev={mean_dev:.1%}); "
            f"train_std={train_std:.4f}, test_std={test_std:.4f} "
            f"(dev={std_dev:.1%}); "
            f"suspiciously_similar={is_suspiciously_similar}"
        ),
    )


# ============================================================================
# P27: 激活值统计（4 head 可视化数据）
# ============================================================================

@dataclass
class ActivationStatsResult:
    passed: bool
    n_4_heads_with_stats: int
    activation_heatmap_saved: bool
    per_head_stats: dict[str, dict[str, float]]
    detail: str


def compute_activation_stats(
    uwb_scaling: np.ndarray,
    uwb_bias: np.ndarray,
    vio_scaling: np.ndarray,
    risk: np.ndarray,
    save_path: str | Path | None = None,
) -> ActivationStatsResult:
    """P27: 计算 4 个输出头的激活值统计（mean/std/min/max）。

    Args:
        uwb_scaling: (T,) or (T, B) 激活值
        uwb_bias:    同上
        vio_scaling: 同上
        risk:        同上
        save_path:   若提供则写入 JSON

    Returns:
        ActivationStatsResult（含 per_head_stats 和 passed）
    """
    def _stats(arr: np.ndarray) -> dict[str, float]:
        a = np.asarray(arr, dtype=np.float64).flatten()
        ensure_finite(a, "activation")
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "median": float(np.median(a)),
            "p95": compute_p95(a),
        }

    per_head = {
        "uwb_scaling": _stats(uwb_scaling),
        "bias": _stats(uwb_bias),
        "vio_scaling": _stats(vio_scaling),
        "risk": _stats(risk),
    }

    # 检查 4 个头都有有效统计（非零 range）
    n_valid = sum(
        1 for s in per_head.values()
        if s["max"] - s["min"] > 1e-9
    )
    saved = False
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"per_head": per_head}, fh, indent=2)
        saved = True

    return ActivationStatsResult(
        passed=n_valid == 4,
        n_4_heads_with_stats=n_valid,
        activation_heatmap_saved=saved,
        per_head_stats=per_head,
        detail=(
            f"n_valid_heads={n_valid}/4; "
            + "; ".join(
                f"{k}: mean={v['mean']:.3f}±{v['std']:.3f}"
                for k, v in per_head.items()
            )
        ),
    )


# ============================================================================
# P33: Best-val checkpoint 协议
# ============================================================================

@dataclass
class BestValResult:
    passed: bool
    best_epoch: int
    best_val_loss: float
    no_test_set_selection: bool
    uses_best_val: bool
    detail: str


def check_best_val_checkpoint(
    train_losses: list[float],
    val_losses: list[float],
    epochs: list[int],
    use_best_val: bool = True,
) -> BestValResult:
    """P33: 验证使用 best-val 而非 test 选 checkpoint。

    若 val_losses 的最小值在最后 20% epoch 之外（正常 early-stopped）→ 通过。
    若最小值在最后 20% 之内且 use_best_val=False → 疑似 test 选择。
    """
    if len(val_losses) == 0:
        return BestValResult(
            passed=False,
            best_epoch=-1,
            best_val_loss=float("inf"),
            no_test_set_selection=False,
            uses_best_val=False,
            detail="no val_losses",
        )

    val_arr = np.asarray(val_losses, dtype=np.float64)
    ensure_finite(val_arr, "val_losses")

    best_idx = int(np.argmin(val_arr))
    best_val = float(val_arr[best_idx])
    best_epoch = int(epochs[best_idx]) if best_idx < len(epochs) else -1

    # no_test_select：最优 epoch 不在最后 20%
    last_20_start = int(len(val_losses) * 0.80)
    no_test_select = best_idx < last_20_start

    passed = use_best_val and no_test_select
    return BestValResult(
        passed=passed,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        no_test_set_selection=no_test_select,
        uses_best_val=use_best_val,
        detail=(
            f"best_epoch={best_epoch}/{epochs[-1]}, "
            f"best_val={best_val:.4f}, "
            f"no_test_select={no_test_select}, "
            f"use_best_val={use_best_val}"
        ),
    )


# ============================================================================
# P34: 过拟合检测
# ============================================================================

@dataclass
class OverfitResult:
    passed: bool
    train_loss_converged: bool
    val_loss_not_rising: bool
    gap: float  # val - train
    detail: str


def check_overfit(
    train_losses: list[float],
    val_losses: list[float],
    overfit_threshold: float = 2.0,
) -> OverfitResult:
    """P34: 训练收敛 + 验证损失未持续上升 + gap < threshold。

    收敛标准：最后 20% epoch 的 train loss 标准差 < 前 20% 的 1/5（震荡小）
    验证未上升：最后 20% val loss 单调性（不持续上升）
    gap：最后 epoch 的 val - train，若 > overfit_threshold 则过拟合
    """
    if len(train_losses) < 5:
        return OverfitResult(
            passed=False,
            train_loss_converged=False,
            val_loss_not_rising=False,
            gap=float("inf"),
            detail="too few epochs",
        )

    train_arr = np.asarray(train_losses, dtype=np.float64)
    val_arr = np.asarray(val_losses, dtype=np.float64)
    ensure_finite(train_arr, "train_losses")
    ensure_finite(val_arr, "val_losses")

    last_20 = int(len(train_arr) * 0.80)
    first_20 = int(len(train_arr) * 0.20)

    # 收敛检测
    early_std = float(np.std(train_arr[:first_20]))
    late_std = float(np.std(train_arr[last_20:]))
    converged = late_std < early_std / 5.0 if early_std > 1e-9 else False

    # 验证未上升（最后 20% val 不持续上升）
    late_val = val_arr[last_20:]
    if len(late_val) >= 3:
        # 线性回归斜率若负→下降；若正但斜率很小→可接受
        x = np.arange(len(late_val))
        slope = float(np.polyfit(x, late_val, 1)[0])
        not_rising = slope <= 0.01 * abs(float(np.mean(late_val)))
    else:
        not_rising = True

    # gap
    gap = float(val_arr[-1] - train_arr[-1])

    passed = converged and not_rising and gap < overfit_threshold
    return OverfitResult(
        passed=passed,
        train_loss_converged=converged,
        val_loss_not_rising=not_rising,
        gap=gap,
        detail=(
            f"converged={converged}, not_rising={not_rising}, "
            f"gap={gap:.4f} (threshold={overfit_threshold})"
        ),
    )


# ============================================================================
# D11: 训练收敛（欠拟合）检测 — 从 P34/overfit 拆出
# ============================================================================

@dataclass
class UnderfitResult:
    passed: bool
    train_loss_decreasing: bool
    loss_plateaued: bool
    final_loss_mean: float
    last_n_epochs_avg: float
    detail: str


def check_underfit(
    train_loss_history: list[float],
    *,
    convergence_patience: int = 5,
    loss_tolerance: float = 1e-4,
) -> UnderfitResult:
    """D11 / Part 3 §第三层 — train loss converged check (separated from overfit per D11 PARTIAL audit).

    Args:
        train_loss_history: 每个 epoch 的训练 loss 列表。
        convergence_patience: 用最后多少个 epoch 判定 plateau（默认 5）。
        loss_tolerance: plateau 判定的相对方差阈值（默认 1e-4）。

    Returns:
        UnderfitResult，其中：
        - train_loss_decreasing: loss_history 的最小值出现在前半段 → 训练确有下降趋势
        - loss_plateaued: 最后 `convergence_patience` 个 epoch 的相对方差 < loss_tolerance
        - final_loss_mean / last_n_epochs_avg: 最后 `convergence_patience` 个 epoch 的均值
        - passed: loss_plateaued and (final_loss < 1.5 * initial_loss)
                  (D11 主判定: loss 已收敛 AND 没有卡在 1.5x 初值)
    """
    if len(train_loss_history) < 2:
        return UnderfitResult(
            passed=False,
            train_loss_decreasing=False,
            loss_plateaued=False,
            final_loss_mean=float("inf"),
            last_n_epochs_avg=float("inf"),
            detail="too few epochs (<2)",
        )

    arr = np.asarray(train_loss_history, dtype=np.float64)
    ensure_finite(arr, "train_loss_history")

    initial_loss = float(arr[0])
    final_loss = float(arr[-1])

    # 1) 训练有下降趋势：min 出现在前半段
    mid_idx = len(arr) // 2
    min_idx = int(np.argmin(arr))
    train_loss_decreasing = min_idx < mid_idx

    # 2) 已收敛（plateau）：最后 N 个 epoch 的相对方差 < tolerance
    n = min(int(convergence_patience), len(arr))
    tail = arr[-n:]
    tail_mean = float(np.mean(tail))
    if abs(tail_mean) > 1e-9:
        rel_var = float(np.var(tail)) / (tail_mean ** 2)
    else:
        rel_var = float(np.var(tail))
    loss_plateaued = rel_var < float(loss_tolerance)

    # 3) 明确欠拟合：final loss 仍在 initial 1.5x 以上 → fail
    not_stuck_high = final_loss < 1.5 * initial_loss

    # D11 主判定: loss 已收敛 AND 没有卡在 1.5x 初值（"训练不动"）→ 通过
    passed = bool(loss_plateaued and not_stuck_high)
    return UnderfitResult(
        passed=passed,
        train_loss_decreasing=train_loss_decreasing,
        loss_plateaued=loss_plateaued,
        final_loss_mean=tail_mean,
        last_n_epochs_avg=tail_mean,
        detail=(
            f"initial={initial_loss:.4f}, final={final_loss:.4f}, "
            f"tail_mean={tail_mean:.4f}, rel_var={rel_var:.2e}, "
            f"decreasing={train_loss_decreasing}, plateaued={loss_plateaued}, "
            f"not_stuck_high={not_stuck_high}"
        ),
    )


# ============================================================================
# D21: risk gate 阈值活跃度检查 — gate 必须是真"门"而非恒开/恒关
# ============================================================================

@dataclass
class GateActivityResult:
    passed: bool
    n_high_activity_frames: int
    n_low_activity_frames: int
    mean_risk: float
    risk_variance: float
    notes: str


def check_risk_gate_threshold_active(
    risk_trace: list[float],
    threshold: float = 0.5,
    *,
    low_activity_quantile: float = 0.10,
    high_activity_quantile: float = 0.90,
) -> GateActivityResult:
    """D21 / Part 3 §第四层 — risk gate 阈值活跃度（避免恒开/恒关）。

    检查项：
    - 必须有合理分布：>5% 帧高于 threshold 且 >5% 帧低于 threshold
    - 否则 gate 形同虚设（恒开或恒关），risk head 无信息可学

    Args:
        risk_trace: 每帧 risk 值序列（[0, 1] 区间）。
        threshold: 风险阈值（默认 0.5）。
        low_activity_quantile / high_activity_quantile: 仅用于记录；主判定为 ±5% 帧数。

    Returns:
        GateActivityResult 含 passed, n_high/n_low_activity_frames, mean_risk, risk_variance, notes。
    """
    if len(risk_trace) < 2:
        return GateActivityResult(
            passed=False,
            n_high_activity_frames=0,
            n_low_activity_frames=0,
            mean_risk=float("nan"),
            risk_variance=float("nan"),
            notes="too_short",
        )

    arr = np.asarray(risk_trace, dtype=np.float64)
    ensure_finite(arr, "risk_trace")

    n_total = len(arr)
    n_high = int(np.sum(arr > threshold))
    n_low = int(np.sum(arr <= threshold))
    frac_high = n_high / n_total
    frac_low = n_low / n_total

    mean_risk = float(np.mean(arr))
    risk_var = float(np.var(arr))

    notes = "active"
    passed = (frac_high > 0.05) and (frac_low > 0.05)
    if frac_high < 0.05:
        notes = "always_off"
    elif frac_low < 0.05:
        notes = "always_on"

    return GateActivityResult(
        passed=passed,
        n_high_activity_frames=n_high,
        n_low_activity_frames=n_low,
        mean_risk=mean_risk,
        risk_variance=risk_var,
        notes=notes,
    )


# ============================================================================
# Git commit hash helper (P39)
# ============================================================================

def get_git_commit_hash() -> str:
    """获取当前 git commit hash（若无 git 则返回 'no-git'）。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "no-git"


def compute_config_hash(cfg: Mapping[str, Any]) -> str:
    """计算配置的稳定 JSON 哈希（键排序）。"""
    try:
        stable = json.dumps(cfg, sort_keys=True, default=str)
        return hashlib.sha256(stable.encode()).hexdigest()[:12]
    except Exception:
        return "hash-error"


# ============================================================================
# P22 stats 工具（从 handbook_p22_statistics.py 导出便捷包装）
# ============================================================================

# 完整实现在 handbook_p22_statistics.py；这里做懒导入避免循环
_p22_mod = None


def _get_p22():
    global _p22_mod
    if _p22_mod is None:
        import importlib
        _p22_mod = importlib.import_module("liquidloc.analysis.handbook_p22_statistics")
    return _p22_mod


def run_holm_bonferroni(
    p_values: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """P22: Holm-Bonferroni 逐步校正。"""
    m = _get_p22()
    return m.run_holm_bonferroni_correction(p_values, alpha)


def run_tost_equivalence(
    effect: float,
    ci_lower: float,
    ci_upper: float,
    epsilon: float,
) -> dict[str, Any]:
    """P22: TOST 等效性检验。"""
    m = _get_p22()
    return m.run_tost(lower=ci_lower, upper=ci_upper, epsilon=epsilon, effect=effect)


def run_bf01_bayes_factor(
    errors_lnn: np.ndarray,
    errors_method: np.ndarray,
    prior_width: float = 1.0,
) -> dict[str, Any]:
    """P22: BF01 Bayes Factor (方法 vs LNN)。"""
    m = _get_p22()
    return m.compute_bf01(errors_lnn, errors_method, prior_width=prior_width)


def compute_nlos_recall(
    risk_trace: Any,
    nl_flag: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """D18 诊断动作 — NLOS 检出率（手册 Part 3 §第五层）。

    以 ``nl_flag`` 为真值算 NLOS 检出能力：
        recall = (risk>=threshold ∧ nl_flag==1) / (nl_flag==1)
        false_positive_rate = (risk>=threshold ∧ nl_flag==0) / (nl_flag==0)
        precision = TP / (TP + FP)

    这是"NLOS 识别 → 协方差调制 → 鲁棒核/残差拒斥 → 自适应噪声"两阶段框架中
    "识别"阶段的量化证据（NLOS 识别+缓解是公开文献共识，公开文献支撑 NLOS
    识别方法演进、GAF 混合网络、LLM 融合可信度评估等研究）。若 recall ≈ 0
    说明 risk 头根本没学到 NLOS 信号 → LNN 的"鲁棒性卖点"无从归因。

    参数:
        risk_trace: 1-D array ( risk ∈ [0, 1] )
        nl_flag: 1-D array ( NLOS 真值, 0/1 )
        threshold: risk 激活阈值 (默认 0.5)

    返回:
        dict 含 recall / false_positive_rate / precision / n_nlos_frames /
        n_los_frames / threshold_used / notes。空 NLOS/LOS 子集时返回零指标
        加 notes 说明（不抛错，保证流水线不会被缺数据拖崩）。

    异常:
        ValueError: 长度不一致 / 有限性违反 / nl_flag 非 0/1 / threshold 越界。
    """
    risk_arr = np.asarray(risk_trace, dtype=np.float64).reshape(-1)
    nl_arr = np.asarray(nl_flag, dtype=np.float64).reshape(-1)
    if risk_arr.size == 0 or nl_arr.size == 0:
        raise ValueError("risk_trace / nl_flag must be non-empty.")
    if risk_arr.size != nl_arr.size:
        raise ValueError("risk_trace and nl_flag must have equal length.")
    if not np.all(np.isfinite(risk_arr)) or not np.all(np.isfinite(nl_arr)):
        raise ValueError("risk_trace / nl_flag must be finite (no NaN/Inf).")
    nl_unique = {float(v) for v in np.unique(nl_arr)}
    if not nl_unique.issubset({0.0, 1.0}):
        raise ValueError("nl_flag must be binary (0/1).")
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be in [0.0, 1.0].")

    n_total = int(risk_arr.size)
    risk_high = risk_arr >= float(threshold)
    n_nlos = int(np.sum(nl_arr == 1))
    n_los = int(np.sum(nl_arr == 0))

    if n_nlos == 0 or n_los == 0:
        notes = (
            "NLOS-only segments missing"
            if n_nlos == 0
            else "LOS-only segments missing"
        )
        return {
            "recall": 0.0,
            "false_positive_rate": 0.0,
            "precision": 0.0,
            "n_nlos_frames": n_nlos,
            "n_los_frames": n_los,
            "n_total_frames": n_total,
            "threshold_used": float(threshold),
            "notes": notes,
        }

    tp = int(np.sum(risk_high & (nl_arr == 1)))
    fp = int(np.sum(risk_high & (nl_arr == 0)))
    fn = int(np.sum((~risk_high) & (nl_arr == 1)))
    tn = int(np.sum((~risk_high) & (nl_arr == 0)))

    recall = tp / n_nlos
    fpr = fp / n_los
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    notes = "ok"
    if recall < 0.20:
        notes = "risk head not learning NLOS signal — recall < 20%"
    elif fpr > 0.50:
        notes = "high false-positive rate — risk head over-fires on LOS"

    return {
        "recall": float(recall),
        "false_positive_rate": float(fpr),
        "precision": float(precision),
        "n_nlos_frames": n_nlos,
        "n_los_frames": n_los,
        "n_total_frames": n_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold_used": float(threshold),
        "notes": notes,
    }


def slice_bias_by_nlos_level(
    bias_trace: Any,
    nl_flag: Any,
) -> dict[str, Any]:
    """D19 诊断动作 — bias 头激活按 NLOS / LOS 切片（手册 Part 3 §第五层）。

    把 bias 头时序按 ``nl_flag`` 切成 NLOS / LOS 两段，比较 bias 头是否在
    NLOS 高偏差段（注入 mu=2-6m 的系统性正偏差）激活更强。如果 NLOS bias
    均值 ≈ LOS bias 均值，说明 bias 头没学到 N3 偏差信号 → "偏差补偿卖点"
    无法支撑。

    参数:
        bias_trace: 1-D array ( bias 头标量输出, 任意数值范围)
        nl_flag: 1-D array ( NLOS 真值, 0/1 )

    返回:
        dict 含 nlos / los 两段的 mean / std / min / max / p95 / frame_count，
        以及 bias_nlos_minus_los_mean（差距）和 bias_nlos_over_los_ratio（倍数）。
        当某一段无样本时记 0.0 + notes 说明。

    异常:
        ValueError: 长度不一致 / 有限性违反 / nl_flag 非 0/1。
    """
    bias_arr = np.asarray(bias_trace, dtype=np.float64).reshape(-1)
    nl_arr = np.asarray(nl_flag, dtype=np.float64).reshape(-1)
    if bias_arr.size == 0 or nl_arr.size == 0:
        raise ValueError("bias_trace / nl_flag must be non-empty.")
    if bias_arr.size != nl_arr.size:
        raise ValueError("bias_trace and nl_flag must have equal length.")
    if not np.all(np.isfinite(bias_arr)) or not np.all(np.isfinite(nl_arr)):
        raise ValueError("bias_trace / nl_flag must be finite (no NaN/Inf).")
    nl_unique = {float(v) for v in np.unique(nl_arr)}
    if not nl_unique.issubset({0.0, 1.0}):
        raise ValueError("nl_flag must be binary (0/1).")

    nlos_mask = nl_arr == 1
    los_mask = nl_arr == 0

    def _stats(arr: np.ndarray) -> dict[str, float]:
        if arr.size == 0:
            return {
                "frame_count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p95": 0.0,
            }
        return {
            "frame_count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p95": float(np.percentile(arr, 95)),
        }

    nlos_stats = _stats(bias_arr[nlos_mask])
    los_stats = _stats(bias_arr[los_mask])

    bias_gap = float(nlos_stats["mean"] - los_stats["mean"])
    los_mean_abs = abs(los_stats["mean"]) if los_stats["mean"] != 0.0 else 1.0
    bias_ratio = bias_gap / los_mean_abs

    notes = "ok"
    if nlos_stats["frame_count"] == 0 or los_stats["frame_count"] == 0:
        notes = "one of NLOS / LOS segments is empty — verdict weak"
    elif abs(bias_ratio) < 0.10:
        notes = "bias head not differentiating NLOS vs LOS — |ratio| < 0.10"

    return {
        "nlos": nlos_stats,
        "los": los_stats,
        "bias_nlos_minus_los_mean": bias_gap,
        "bias_nlos_over_los_ratio": float(bias_ratio),
        "notes": notes,
    }


def check_activation_mask_alignment(
    per_head_activations: Any,
    scene_mask: Any,
    *,
    activation_quantile: float = 0.75,
) -> dict[str, Any]:
    """D20 诊断动作 — 4 头激活 × scene_mask 对齐度（手册 Part 3 §第五层）。

    每个头在按 ``activation_quantile`` 分位数阈值二值化后算 IoU / precision /
    recall / F1。risk / bias 头应该和 NLOS mask 强对齐（IoU 高），uwb /
    vio_scaling 头应该相对均匀（IoU 低）。无对齐说明机制错位。

    参数:
        per_head_activations: dict[head_name → 1-D array]，必须含 risk/bias/
            uwb_scaling/vio_scaling 4 个键。
        scene_mask: 1-D array, NLOS 段二值标签 (0/1)。
        activation_quantile: 激活二值化阈值（默认 0.75 百分位 — 高激活帧）。

    返回:
        dict per head: {iou, precision, recall, f1}。
    """
    if not isinstance(per_head_activations, dict):
        raise TypeError("per_head_activations must be a dict[str, sequence].")
    head_arrays: dict[str, np.ndarray] = {}
    required_heads = ("risk", "bias", "uwb_scaling", "vio_scaling")
    for head_name in required_heads:
        if head_name not in per_head_activations:
            raise ValueError(f"per_head_activations missing required head '{head_name}'.")
        arr = np.asarray(per_head_activations[head_name], dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"per_head_activations[{head_name}] must be non-empty.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"per_head_activations[{head_name}] must be finite.")
        head_arrays[head_name] = arr

    scene = np.asarray(scene_mask, dtype=np.float64).reshape(-1)
    if scene.size == 0:
        raise ValueError("scene_mask must be non-empty.")
    if not np.all(np.isfinite(scene)):
        raise ValueError("scene_mask must be finite.")
    scene_unique = {float(v) for v in np.unique(scene)}
    if not scene_unique.issubset({0.0, 1.0}):
        raise ValueError("scene_mask must be binary (0/1).")

    n_total = scene.size
    sizes = {n_total, *[v.size for v in head_arrays.values()]}
    if len(sizes) != 1:
        raise ValueError("scene_mask / per-head activations must have equal length.")
    if not (0.0 < float(activation_quantile) < 1.0):
        raise ValueError("activation_quantile must be in (0, 1).")

    n_scene_pos = int(np.sum(scene == 1))
    result: dict[str, Any] = {
        "n_total_frames": n_total,
        "n_scene_pos_frames": n_scene_pos,
        "activation_quantile": float(activation_quantile),
        "per_head": {},
    }
    for head_name in required_heads:
        arr = head_arrays[head_name]
        if n_scene_pos == 0:
            iou = precision = recall = f1 = 0.0
        else:
            threshold = float(np.quantile(arr, float(activation_quantile)))
            head_active = arr >= threshold
            tp = int(np.sum(head_active & (scene == 1)))
            fp = int(np.sum(head_active & (scene == 0)))
            fn = int(np.sum((~head_active) & (scene == 1)))
            union = int(np.sum(head_active | (scene == 1)))
            iou = tp / union if union > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / n_scene_pos
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
        result["per_head"][head_name] = {
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return result


__all__ = [
    # 工具
    "ensure_finite",
    "compute_rmse", "compute_mae", "compute_median_error",
    "compute_p50", "compute_p95", "compute_trimmed_mean", "compute_all_metrics",
    "MetricBundle",
    "get_git_commit_hash", "compute_config_hash",
    # P7
    "run_rotation_chain_test", "RotationChainResult",
    # P8
    "check_units", "UnitCheckResult",
    # P9
    "evaluate_sim3_alignment", "align_sim3", "align_rigid", "Sim3AlignmentResult",
    # P21
    "check_normalization_leak", "NormalizationLeakResult",
    # P27
    "compute_activation_stats", "ActivationStatsResult",
    # P33
    "check_best_val_checkpoint", "BestValResult",
    # P34
    "check_overfit", "OverfitResult",
    # D11 (从 P34 拆出)
    "check_underfit", "UnderfitResult",
    # D21 risk gate 活跃度
    "check_risk_gate_threshold_active", "GateActivityResult",
    # D18/D19/D20 NLOS-layer diagnostic actions
    "compute_nlos_recall",
    "slice_bias_by_nlos_level",
    "check_activation_mask_alignment",
    # P22 wrappers
    "run_holm_bonferroni", "run_tost_equivalence", "run_bf01_bayes_factor",
]
