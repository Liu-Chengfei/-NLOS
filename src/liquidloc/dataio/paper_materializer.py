"""Paper 数据集 (paper_main) 物化器。

职责：从 paper_dataset.yaml 读取 5 轨迹类 × 3 速度档的池结构，调用
`sim_materializer.materialize_sim_raw` 完成几何+传感器流物化，再叠加：
  1. 5 轨迹类 × 3 速度档的 paper 专用 GT（line_back_forth / random_walk /
     figure_eight / stop_turn / polygon），不依赖 mini-fixture。
  2. paper 专用 anchor_layout（K4_layout 为 paper 内部字典索引名，实际位置由 anchor_K4_layout 字典提供）。
  3. paper 专用连续段 NLOS 注入（不是 sim 的 poisson scatter）。
  4. paper 连续段丢包（uwb_outage_segments / vio_outage_segments）。
  5. paper sim_meta.json 含 class_id + seed + 完整 axes_override。

本文件不修改 sim_materializer.py / 原 sim_e9 流程；新数据集与原 sim 数据集
完全隔离（不同 raw_root、不同 axes、不同 sim_meta 字段约定）。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from liquidloc.common.angle_utils import angle_delta_rad
from liquidloc.common.config_utils import find_project_root, load_yaml_config
from liquidloc.common.gt_utils import normalize_gt_rows
from liquidloc.common.io_utils import write_json
from liquidloc.common.validation import coerce_finite_scalar
from liquidloc.dataio.sim_materializer import SIM_GENERATOR_VERSION_CURRENT  # §28.6 自证契约：复用 sim 版本号
from liquidloc.scenarios.paper_nlos_injection import apply_paper_nlos_blocks


# ============================================================
# 几何推导 IMU / VIO 行（基于 GT，无 fixture 依赖）
# ============================================================

def _build_gt_timestamp_index(gt_rows: list[dict[str, float]]) -> list[float]:
    timestamps = [float(r["timestamp"]) for r in gt_rows]
    return timestamps  # 已按时间升序（trajectory 生成器保证）


def _interpolate_pose(gt_rows: list[dict[str, float]], t: float) -> dict[str, float]:
    """bisect 找最近 GT 行，线性内插。"""
    import bisect
    timestamps = [float(r["timestamp"]) for r in gt_rows]
    idx = bisect.bisect_left(timestamps, t)
    if idx == 0:
        return gt_rows[0]
    if idx >= len(gt_rows):
        return gt_rows[-1]
    if abs(timestamps[idx] - t) < 1e-9:
        return gt_rows[idx]
    if abs(timestamps[idx - 1] - t) < 1e-9:
        return gt_rows[idx - 1]
    # 线性内插
    t0, t1 = timestamps[idx - 1], timestamps[idx]
    alpha = (t - t0) / max(t1 - t0, 1e-9)
    p0, p1 = gt_rows[idx - 1], gt_rows[idx]
    return {
        "timestamp": t,
        "px": float(p0["px"]) + alpha * (float(p1["px"]) - float(p0["px"])),
        "py": float(p0["py"]) + alpha * (float(p1["py"]) - float(p0["py"])),
        "yaw": float(p0["yaw"]) + alpha * (float(p1["yaw"]) - float(p0["yaw"])),
    }


def _derive_paper_imu_rows(gt_rows: list[dict[str, float]], timestamps: list[float]) -> list[dict[str, float]]:
    """从 GT 推导 IMU 行（150Hz），复用 sim_materializer._derive_imu_rows_from_normalized_gt 语义。
    关键：GT yaw 在 ±π 跳变时（atan2(vy, vx) 特征），中心差分 gz 会产生 ±π/dt 量级的跳变。
    解法：先 unwrap gt yaw 让差值连续，再做中心差分。
    """
    norm_gt = normalize_gt_rows(gt_rows)
    # unwrap yaw：让相邻 yaw 差值最小化（消除 ±π 跳变）
    unwrapped_yaws: list[float] = [float(norm_gt[0]["yaw"])]
    for i in range(1, len(norm_gt)):
        cur = float(norm_gt[i]["yaw"])
        prev = unwrapped_yaws[i - 1]
        delta = angle_delta_rad(cur, prev)
        unwrapped_yaws.append(prev + delta)
    poses = [_interpolate_pose_with_unwrapped_yaw(norm_gt, unwrapped_yaws, float(t)) for t in timestamps]
    # 速度
    velocities: list[tuple[float, float]] = []
    for i, pose in enumerate(poses):
        if i == 0:
            nxt = poses[1]
            dt = nxt["timestamp"] - pose["timestamp"]
            velocities.append(((nxt["px"] - pose["px"]) / dt, (nxt["py"] - pose["py"]) / dt))
        elif i == len(poses) - 1:
            prv = poses[i - 1]
            dt = pose["timestamp"] - prv["timestamp"]
            velocities.append(((pose["px"] - prv["px"]) / dt, (pose["py"] - prv["py"]) / dt))
        else:
            prv, nxt = poses[i - 1], poses[i + 1]
            dt = nxt["timestamp"] - prv["timestamp"]
            velocities.append(((nxt["px"] - prv["px"]) / dt, (nxt["py"] - prv["py"]) / dt))
    rows: list[dict[str, float]] = []
    for i, pose in enumerate(poses):
        yaw = float(pose["yaw_unwrapped"])
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        if i == 0:
            nxt = poses[1]
            dt = nxt["timestamp"] - pose["timestamp"]
            ax_w = (velocities[1][0] - velocities[0][0]) / dt
            ay_w = (velocities[1][1] - velocities[0][1]) / dt
            gz = angle_delta_rad(float(nxt["yaw_unwrapped"]), yaw) / dt
        elif i == len(poses) - 1:
            prv = poses[i - 1]
            dt = pose["timestamp"] - prv["timestamp"]
            ax_w = (velocities[i][0] - velocities[i - 1][0]) / dt
            ay_w = (velocities[i][1] - velocities[i - 1][1]) / dt
            gz = angle_delta_rad(yaw, float(prv["yaw_unwrapped"])) / dt
        else:
            prv, nxt = poses[i - 1], poses[i + 1]
            dt = nxt["timestamp"] - prv["timestamp"]
            ax_w = (velocities[i + 1][0] - velocities[i - 1][0]) / dt
            ay_w = (velocities[i + 1][1] - velocities[i - 1][1]) / dt
            gz = angle_delta_rad(float(nxt["yaw_unwrapped"]), float(prv["yaw_unwrapped"])) / dt
        rows.append({
            "timestamp": float(pose["timestamp"]),
            "ax": cos_y * ax_w + sin_y * ay_w,
            "ay": -sin_y * ax_w + cos_y * ay_w,
            "gz": gz,
        })
    return rows


def _interpolate_pose_with_unwrapped_yaw(
    norm_gt: list[dict[str, float]],
    unwrapped_yaws: list[float],
    t: float,
) -> dict[str, float]:
    """类似 _interpolate_pose 但 yaw 用 unwrapped 值，dx/dy 用原始 wrapped yaw 计算（不影响位置）。"""
    import bisect
    pose = _interpolate_pose(norm_gt, t)
    # bisect 找最近的 GT 行
    timestamps = [float(r["timestamp"]) for r in norm_gt]
    idx = bisect.bisect_left(timestamps, t)
    if idx == 0:
        uw_yaw = unwrapped_yaws[0]
    elif idx >= len(norm_gt):
        uw_yaw = unwrapped_yaws[-1]
    else:
        # 线性内插 unwrapped yaw
        t0, t1 = timestamps[idx - 1], timestamps[idx]
        alpha = (t - t0) / max(t1 - t0, 1e-9)
        uw_yaw = unwrapped_yaws[idx - 1] + alpha * (unwrapped_yaws[idx] - unwrapped_yaws[idx - 1])
    pose["yaw_unwrapped"] = uw_yaw
    return pose


def _derive_paper_vio_rows(gt_rows: list[dict[str, float]], timestamps: list[float]) -> list[dict[str, float]]:
    """从 GT 推导 VIO 行（20Hz）：dx/dy/dyaw 在 [t, t+dt] 区间内累积。
    关键：GT yaw 在 ±π 跳变时（atan2 特征）直接相减会 ±2π 量级爆掉；用 unwrap 后再减。
    """
    norm_gt = normalize_gt_rows(gt_rows)
    # unwrap yaw
    unwrapped_yaws: list[float] = [float(norm_gt[0]["yaw"])]
    for i in range(1, len(norm_gt)):
        cur = float(norm_gt[i]["yaw"])
        prev = unwrapped_yaws[i - 1]
        delta = angle_delta_rad(cur, prev)
        unwrapped_yaws.append(prev + delta)
    # 把 unwrapped yaw 注入到 norm_gt 的 dict 里
    for i, r in enumerate(norm_gt):
        r["_uw_yaw"] = unwrapped_yaws[i]
    rows: list[dict[str, float]] = []
    for i, t in enumerate(timestamps):
        if i + 1 >= len(timestamps):
            rows.append({"timestamp": t, "dx": 0.0, "dy": 0.0, "dyaw": 0.0, "quality": 1.0})
            continue
        t1 = timestamps[i + 1]
        p0 = _interpolate_pose(norm_gt, t)
        p1 = _interpolate_pose(norm_gt, t1)
        # 用 unwrap 后的 yaw
        import bisect
        timestamps_gt = [float(r["timestamp"]) for r in norm_gt]
        idx0 = bisect.bisect_left(timestamps_gt, t)
        idx1 = bisect.bisect_left(timestamps_gt, t1)
        if idx0 >= len(norm_gt):
            yaw0 = unwrapped_yaws[-1]
        else:
            alpha0 = (t - timestamps_gt[idx0 - 1]) / max(timestamps_gt[idx0] - timestamps_gt[idx0 - 1], 1e-9) if idx0 > 0 else 0.0
            yaw0 = unwrapped_yaws[idx0 - 1] + alpha0 * (unwrapped_yaws[idx0] - unwrapped_yaws[idx0 - 1]) if idx0 > 0 else unwrapped_yaws[0]
        if idx1 >= len(norm_gt):
            yaw1 = unwrapped_yaws[-1]
        else:
            alpha1 = (t1 - timestamps_gt[idx1 - 1]) / max(timestamps_gt[idx1] - timestamps_gt[idx1 - 1], 1e-9) if idx1 > 0 else 0.0
            yaw1 = unwrapped_yaws[idx1 - 1] + alpha1 * (unwrapped_yaws[idx1] - unwrapped_yaws[idx1 - 1]) if idx1 > 0 else unwrapped_yaws[0]
        cy, sy = math.cos(yaw0), math.sin(yaw0)
        dx_w = p1["px"] - p0["px"]
        dy_w = p1["py"] - p0["py"]
        dx = cy * dx_w + sy * dy_w
        dy = -sy * dx_w + cy * dy_w
        dyaw = yaw1 - yaw0  # unwrapped 差，不会 ±2π
        rows.append({"timestamp": t, "dx": dx, "dy": dy, "dyaw": dyaw, "quality": 1.0})
    return rows


def _derive_paper_uwb_rows(
    gt_rows: list[dict[str, float]],
    timestamps: list[float],
    anchor_positions: list[list[float]],
) -> list[dict[str, float]]:
    """从 GT 推导 UWB 行（10Hz）：range = |gt - anchor|，TWR 4 时戳 + tof。"""
    rows: list[dict[str, float]] = []
    _TWR_C = 299_702_547.0
    _TWR_REPLY_S = 1e-4
    for t in timestamps:
        p = _interpolate_pose(gt_rows, t)
        for i, anchor in enumerate(anchor_positions):
            dx = p["px"] - anchor[0]
            dy = p["py"] - anchor[1]
            dz = 0.0 - anchor[2]  # 2D 平面
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            tof = r / _TWR_C
            poll_tx = t
            poll_rx = poll_tx + tof
            resp_tx = poll_rx + _TWR_REPLY_S
            resp_rx = resp_tx + tof
            rows.append({
                "timestamp": t,
                "anchor_id": f"A{i + 1}",
                "range": r,
                "valid": True,
                "quality": 1.0,
                "tof": tof,
                "twr_poll_tx": poll_tx,
                "twr_poll_rx": poll_rx,
                "twr_resp_tx": resp_tx,
                "twr_resp_rx": resp_rx,
            })
    return rows


# ============================================================
# 噪声注入（纯高斯，按 spec 参数）
# ============================================================

def _seeded_rng(base_seed: int, *parts: str) -> random.Random:
    digest = hashlib.sha256(
        "|".join([str(base_seed), *[str(p) for p in parts]]).encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big", signed=False))


def _inject_uwb_noise(
    rows: list[dict[str, float]],
    sigma_m: float,
    seed: int,
    seq_id: str,
) -> None:
    rng = _seeded_rng(seed, seq_id, "uwb", "range")
    for r in rows:
        r["range"] = max(0.0, float(r["range"]) + rng.gauss(0.0, sigma_m))


def _inject_uwb_async(
    rows: list[dict[str, float]],
    offset_ms: float,
    jitter_ms: float,
    seed: int,
    seq_id: str,
) -> None:
    """对 UWB 时间戳施加 A 档异步偏移（常数 + per-packet jitter，P25 硬约束）。

    P25 要求 "UWB 异步采样抖动 (jitter σ)" 必须是 per-packet 的高斯抽样，而
    不是序列级别的单次采样当作固定偏移施加。每个 UWB 帧独立抽样一次
    gauss(0, jitter_ms) 加到 offset 上，确保 jitter 体现"采样时刻的随机
    抖动"语义而不是"序列整体的随机偏移"。
    """
    rng = _seeded_rng(seed, seq_id, "uwb", "async")
    base_offset_s = offset_ms / 1000.0
    jitter_sigma_s = jitter_ms / 1000.0
    for r in rows:
        per_packet_jitter_s = rng.gauss(0.0, jitter_sigma_s)
        r["timestamp"] = float(r["timestamp"]) + base_offset_s + per_packet_jitter_s


def _inject_vio_async(
    rows: list[dict[str, float]],
    offset_ms: float,
    jitter_ms: float,
    seed: int,
    seq_id: str,
) -> None:
    rng = _seeded_rng(seed, seq_id, "vio", "async")
    base_offset_s = offset_ms / 1000.0
    jitter_sigma_s = jitter_ms / 1000.0
    for r in rows:
        per_packet_jitter_s = rng.gauss(0.0, jitter_sigma_s)
        r["timestamp"] = float(r["timestamp"]) + base_offset_s + per_packet_jitter_s


def _inject_imu_async(
    rows: list[dict[str, float]],
    offset_ms: float,
    jitter_ms: float,
    seed: int,
    seq_id: str,
) -> None:
    """IMU 异步偏移（常数 + per-packet jitter，P25 硬约束）。每个 IMU 帧独立抽样
    gauss(0, jitter_ms) 以体现采样时刻的随机抖动语义。"""
    rng = _seeded_rng(seed, seq_id, "imu", "async")
    base_offset_s = offset_ms / 1000.0
    jitter_sigma_s = jitter_ms / 1000.0
    for r in rows:
        per_packet_jitter_s = rng.gauss(0.0, jitter_sigma_s)
        r["timestamp"] = float(r["timestamp"]) + base_offset_s + per_packet_jitter_s


def _inject_vio_proc_delay(
    rows: list[dict[str, float]],
    proc_delay_ms: float,
    seed: int,
    seq_id: str,
) -> None:
    """VIO 处理延迟：A1+ 引入。VIO 输出比真值晚 proc_delay_ms，
    实现为整个序列 VIO 时间戳整体后移（确定性常数，无 jitter）。"""
    if proc_delay_ms <= 0.0:
        return
    delay_s = proc_delay_ms / 1000.0
    for r in rows:
        r["timestamp"] = float(r["timestamp"]) + delay_s


def _inject_uwb_rtt(
    rows: list[dict[str, float]],
    rtt_ms: float,
    seed: int,
    seq_id: str,
) -> None:
    """UWB 往返延迟：整个序列 UWB 时间戳整体前移 rtt_ms（模拟标签-锚点 RTT）。"""
    if rtt_ms <= 0.0:
        return
    rtt_s = rtt_ms / 1000.0
    for r in rows:
        r["timestamp"] = float(r["timestamp"]) - rtt_s


def _inject_imu_noise(
    rows: list[dict[str, float]],
    accel_white_psd: float,
    gyro_white_psd: float,
    accel_bias_instab: float,
    gyro_bias_instab: float,
    accel_rrw: float,
    gyro_rrw: float,
    dt_s: float,
    seed: int,
    seq_id: str,
) -> None:
    """IMU 噪声：白噪声 + 恒定 bias（简化 FFN+RRW）。

    单位约定：
    - accel_white_psd / gyro_white_psd：m/s²/√Hz 与 rad/s/√Hz（功率谱密度）。
      每步标准差 = psd × sqrt(1 / (2*dt)) ≈ psd × sqrt(fs/2)。
    - accel_rrw / gyro_rrw：random walk 系数（m/s²/√s 与 rad/s/√s）。
      每步增量的标准差 = rrw × sqrt(dt)；累积 Wiener 过程漂移 std = rrw × sqrt(t)。
      为防止 60s 序列漂移失控，rrw 系数保持小量级（1e-6 量级）。
    - accel_bias_instab / gyro_bias_instab：序列内恒定 bias（m/s² 与 rad/s）。
    """
    rng_white = _seeded_rng(seed, seq_id, "imu", "white")
    rng_bias = _seeded_rng(seed, seq_id, "imu", "bias")
    # 每步白噪声标准差（按 PSD → 时域离散噪声）
    accel_white_sigma = accel_white_psd * math.sqrt(1.0 / (2.0 * dt_s))
    gyro_white_sigma = gyro_white_psd * math.sqrt(1.0 / (2.0 * dt_s))
    # 每步 RRW 增量标准差
    accel_rrw_step_sigma = accel_rrw * math.sqrt(dt_s)
    gyro_rrw_step_sigma = gyro_rrw * math.sqrt(dt_s)
    accel_bias_x = rng_bias.gauss(0.0, accel_bias_instab)
    accel_bias_y = rng_bias.gauss(0.0, accel_bias_instab)
    gyro_bias = rng_bias.gauss(0.0, gyro_bias_instab)
    accel_rrw_x = 0.0
    accel_rrw_y = 0.0
    gyro_rrw = 0.0
    for r in rows:
        accel_rrw_x += rng_white.gauss(0.0, accel_rrw_step_sigma)
        accel_rrw_y += rng_white.gauss(0.0, accel_rrw_step_sigma)
        gyro_rrw += rng_white.gauss(0.0, gyro_rrw_step_sigma)
        r["ax"] = float(r["ax"]) + accel_bias_x + accel_rrw_x + rng_white.gauss(0.0, accel_white_sigma)
        r["ay"] = float(r["ay"]) + accel_bias_y + accel_rrw_y + rng_white.gauss(0.0, accel_white_sigma)
        r["gz"] = float(r["gz"]) + gyro_bias + gyro_rrw + rng_white.gauss(0.0, gyro_white_sigma)


def _inject_vio_noise(
    rows: list[dict[str, float]],
    dx_dy_std: float,
    yaw_drift_dps: float,
    seed: int,
    seq_id: str,
) -> None:
    rng = _seeded_rng(seed, seq_id, "vio", "noise")
    yaw_drift_rad_per_s = math.radians(yaw_drift_dps)
    cumulative_yaw_drift = 0.0
    last_t = None
    for r in rows:
        r["dx"] = float(r["dx"]) + rng.gauss(0.0, dx_dy_std)
        r["dy"] = float(r["dy"]) + rng.gauss(0.0, dx_dy_std)
        # yaw drift 按时间累积
        if last_t is not None:
            dt = float(r["timestamp"]) - last_t
            cumulative_yaw_drift += rng.gauss(0.0, yaw_drift_rad_per_s * dt)
        r["dyaw"] = float(r["dyaw"]) + cumulative_yaw_drift
        last_t = float(r["timestamp"])


def _apply_outage_segments(
    rows: list[dict[str, float]],
    outage_segments: list[tuple[float, float]],
    *,
    zero_payload: bool = False,
) -> None:
    """对 outage_segments 内的行标记 valid=False / quality=0。
    UWB 仍保留 range 字段；VIO/IMU 若 zero_payload=True 则清零核心字段。
    """
    for r in rows:
        t = float(r["timestamp"])
        if any(s <= t <= e for s, e in outage_segments):
            r["valid"] = False
            r["quality"] = 0.0
            if zero_payload:
                for k in ("ax", "ay", "gz", "dx", "dy", "dyaw"):
                    if k in r:
                        r[k] = 0.0


# ============================================================
# 5 轨迹类生成器（无 fixture 依赖，直接按 paper spec 合成 GT）
# ============================================================

def _gt_row(t: float, x: float, y: float, yaw: float, vx: float, vy: float) -> dict[str, float]:
    """构造一条 GT 行（与 sim_materializer 输出一致）。"""
    return {
        "timestamp": float(t),
        "px": float(x),
        "py": float(y),
        "yaw": float(yaw),
        "vx": float(vx),
        "vy": float(vy),
    }


def _boundary_steering_blend(value: float, lo: float, hi: float, ramp_m: float = 0.5) -> float:
    """计算边界 steering blend 系数。

    当 value 在 [lo, hi] 中间时返回 0（不干预 heading）；
    距离边界 ramp_m 米内线性增加到 1（完全 override）。
    """
    distance_to_lower = max(0.0, value - lo)
    distance_to_upper = max(0.0, hi - value)
    nearest = min(distance_to_lower, distance_to_upper)
    if nearest >= ramp_m:
        return 0.0
    if nearest <= 0.0:
        return 1.0
    return (ramp_m - nearest) / ramp_m


def _trajectory_line_back_forth(rng: random.Random, duration_s: float, speed_mps: float, half_w: float, half_h: float) -> list[dict[str, float]]:
    """直行往返：x 用 x = x_center + x_amp * sin(2π t / period) 平滑往返。
    x 变化连续可微（无 vx 符号跳变，IMU 中心差分安全），y 用 cos 慢变。
    yaw 用 atan2(vy, vx) 但 vx 不接近 0（sin 峰值处 vx=0，需要避免）。
    """
    rows: list[dict[str, float]] = []
    x_center = half_w  # x 中心在场中央
    x_amp = half_w * 0.8  # 振幅（在场内 ±80% 范围）
    # period 使 speed_mps ≈ 2π * x_amp / period → period = 2π * x_amp / speed_mps
    period = 2 * math.pi * x_amp / max(speed_mps, 0.1)
    n = max(2, int(duration_s * 50))
    y_amp = 0.3
    y_period = max(period * 0.5, 1.0)  # y 慢于 x（避免 vx 接近 0 时 yaw 跳变）
    for i in range(n + 1):
        t = i * (duration_s / n)
        x = x_center + x_amp * math.sin(2 * math.pi * t / period)
        vx = (2 * math.pi * x_amp / period) * math.cos(2 * math.pi * t / period)
        # FIX: 之前 y = half_h * 0.5 + ... 让 y_center=5 错位置，应该用 half_h 居中。
        y = half_h + y_amp * math.cos(2 * math.pi * t / y_period)
        vy = -y_amp * (2 * math.pi / y_period) * math.sin(2 * math.pi * t / y_period)
        # vx 在 t=0/period/2/... 时为 0（sin 峰值）——此时 yaw 不稳定但 vx=0 时 yaw 物理上不定义
        # 解决：让 vx 始终非零（加微小偏移）
        vx = vx if abs(vx) > 0.1 else (0.1 if vx >= 0 else -0.1)
        yaw = math.atan2(vy, vx)
        rows.append(_gt_row(t, x, y, yaw, vx, vy))
    return rows


def _trajectory_random_walk(rng: random.Random, duration_s: float, speed_mps: float, half_w: float, half_h: float) -> list[dict[str, float]]:
    """随机游走（paper 安全版）：用 sin 函数驱动 heading 慢变（不随机跳变），
    避免边界反弹瞬时反向导致 vx 反转 + atan2 跳变。
    heading(t) = heading_0 + sum(Δheading_i * sin(2π t / T_i)) 慢变正弦叠加。
    """
    rows: list[dict[str, float]] = []
    n = max(2, int(duration_s * 50))
    dt = duration_s / n
    cx, cy = half_w, half_h
    # 用 3 个长周期正弦叠加做 heading 慢变（每 8-20s 一次完整周期）
    n_sin = 3
    sin_params = []
    for _ in range(n_sin):
        period = rng.uniform(8.0, 20.0)
        amp = rng.uniform(0.2, 0.5)  # 单正弦振幅 < π/2 累计 < 1 rad
        phase = rng.uniform(0, 2 * math.pi)
        sin_params.append((period, amp, phase))
    for i in range(n + 1):
        t = i * dt
        heading = sum(amp * math.sin(2 * math.pi * t / period + phase) for (period, amp, phase) in sin_params)
        # 准则 5「轨迹不出域」：用软 steering 修正 heading，
        # 使 walker 在边界附近逐渐转向离开（优于末端钳制+flip 组合）。
        # blend 比例从边界外 1.0m 处的 0.1 线性增加到边界处的 1.0。
        x_min, x_max = 0.5, 2 * half_w - 0.5
        y_min, y_max = 0.5, 2 * half_h - 0.5
        x_blend = _boundary_steering_blend(cx, x_min, x_max, ramp_m=1.0)
        y_blend = _boundary_steering_blend(cy, y_min, y_max, ramp_m=1.0)
        blend = max(x_blend, y_blend)  # 任一维度接近边界则启用 steering
        if blend > 0.0:
            if cx < x_min + 1.0:
                safe_h = 0.1  # 朝右（heading ≈ 0）
            elif cx > x_max - 1.0:
                safe_h = math.pi - 0.1  # 朝左（heading ≈ π）
            elif cy < y_min + 1.0:
                safe_h = math.pi / 2 - 0.1  # 朝上（heading ≈ π/2）
            else:
                safe_h = 3.0 * math.pi / 2 + 0.1  # 朝下（heading ≈ 3π/2）
            heading = heading * (1.0 - blend) + safe_h * blend
        vx = speed_mps * math.cos(heading)
        vy = speed_mps * math.sin(heading)
        # 准则 25「最小速度非零」：防 atan2 边界 ±π 跳变。
        if abs(vx) < 0.1:
            vx = 0.5 if vx >= 0 else -0.5
        yaw = math.atan2(vy, vx)
        cx += vx * dt
        cy += vy * dt
        # 硬钳制：兜底防御 steering 失败（速度大但 blend 偏低 / dt 大）。
        cx = max(x_min, min(x_max, cx))
        cy = max(y_min, min(y_max, cy))
        rows.append(_gt_row(t, cx, cy, yaw, vx, vy))
    return rows


def _trajectory_figure_eight(rng: random.Random, duration_s: float, speed_mps: float, half_w: float, half_h: float) -> list[dict[str, float]]:
    """8 字机动：Lissajous 曲线 x = cx + ax*sin(θ), y = cy + ay*sin(2θ)。

    IMU 中心差分 ax/ay 必须 < 150 m/s²。50Hz 采样 dt=0.02s 二次差分放大 1/dt²=2500。

    修复 v2：原版用 (t % total_period) 重新定位 theta，导致每周期初 theta 从 2π 跳回 0
    → 三帧 px/py 完全相同 → ax/ay 越界。改用严格累积 theta 推进（每周期 +2π），
    并把 dt 控制在 ≤ 0.02s（50Hz 采样），保证 Lissajous 角速度与速度匹配。
    """
    rows: list[dict[str, float]] = []
    cx, cy = half_w, half_h
    ax, ay = 1.5, 1.0  # 缩小半轴防中心差分越界
    # 角速度（rad/s）= 速度 / 总弧长 × 2π（一个周期）= speed_mps / (2*ax+4*ay) × 2π
    arc_per_period = 2 * 2 * ax + 4 * 2 * ay  # 2 个 sin 半周 (4ax) + 4 个 cos 半周 (8ay)，近似
    # 实际用曲线积分
    n_int = 400
    cum = [0.0]
    for i in range(1, n_int + 1):
        prev_t = 2 * math.pi * (i - 1) / n_int
        cur_t = 2 * math.pi * i / n_int
        # 弧长用 Simpson
        def v_at(t: float) -> float:
            return math.sqrt((ax * math.cos(t)) ** 2 + (2 * ay * math.cos(2 * t)) ** 2)
        seg_len = (v_at(prev_t) + 4 * v_at((prev_t + cur_t) / 2) + v_at(cur_t)) / 6 * (cur_t - prev_t)
        cum.append(cum[-1] + seg_len)
    total_distance = cum[-1]
    period = total_distance / max(speed_mps, 0.1)
    omega = 2 * math.pi / period  # 角速度 (rad/s)
    n = max(2, int(duration_s * 50))
    for i in range(n + 1):
        t = i * (duration_s / n)
        # 严格累积 theta（永远递增，不取模）
        theta = omega * t
        x = cx + ax * math.sin(theta)
        y = cy + ay * math.sin(2 * theta)
        # 切线方向 = (dx/dθ, dy/dθ) = (ax*cos(θ), 2*ay*cos(2θ))
        dx_dt = ax * math.cos(theta)
        dy_dt = 2 * ay * math.cos(2 * theta)
        yaw = math.atan2(dy_dt, dx_dt)
        vx = speed_mps * math.cos(yaw)
        vy = speed_mps * math.sin(yaw)
        rows.append(_gt_row(t, x, y, yaw, vx, vy))
    return rows


def _trajectory_stop_turn(rng: random.Random, duration_s: float, speed_mps: float, half_w: float, half_h: float) -> list[dict[str, float]]:
    """急停急转轨迹（柔和版）：余弦平滑转向，无 IMU 中心差分跳变。

    修复：原版 yaw_offset 在 turn_phase=0.5 处线性突跳（30°→0°），
    位置跳变 × 1/dt²=2500 放大后使 IMU ax/ay 越界 6700 m/s²。
    改用 cosine 缓入缓出：yaw_offset 从 0→30°→0° 全程二阶连续。
    """
    rows: list[dict[str, float]] = []
    n = max(2, int(duration_s * 50))
    dt = duration_s / n
    # FIX: 之前用 half_w * 0.5 / half_h * 0.5 当 y center，5 错误地假设 half = full_height
    # 实际 half_h = site_xy/2 (10)，所以 cx,cy 应 = half_w, half_h 即场中心 (10, 10)。
    cx, cy = half_w, half_h
    base_period = 10.0
    base_amp_x = half_w * 0.6
    turn_period = 8.0          # 每 8s 经历一次完整转向
    turn_fraction = 0.4        # 40% 时间转向，60% 直行
    turn_angle = math.pi / 6   # 30° 总转向角

    for i in range(n + 1):
        t = i * dt
        # 基础位置（慢速椭圆）
        x_offset = base_amp_x * math.sin(2 * math.pi * t / base_period)
        y_offset = base_amp_x * 0.5 * math.cos(2 * math.pi * t / base_period)
        vx_base = (2 * math.pi * base_amp_x / base_period) * math.cos(2 * math.pi * t / base_period)
        vy_base = -(2 * math.pi * base_amp_x * 0.5 / base_period) * math.sin(2 * math.pi * t / base_period)

        # cosine 平滑转向（0→30°→0°，二阶连续）
        turn_phase = (t % turn_period) / turn_period
        if turn_phase < turn_fraction:
            # 0 → 30°（缓入）
            phase = turn_phase / turn_fraction * math.pi
            yaw_offset = turn_angle * 0.5 * (1 - math.cos(phase))
        else:
            # 30° → 0°（缓出）
            phase = (turn_phase - turn_fraction) / (1.0 - turn_fraction) * math.pi
            yaw_offset = turn_angle * 0.5 * (1 + math.cos(phase))

        cy_off, sy_off = math.cos(yaw_offset), math.sin(yaw_offset)
        vx = vx_base * cy_off - vy_base * sy_off
        vy = vx_base * sy_off + vy_base * cy_off
        x = cx + x_offset * cy_off - y_offset * sy_off
        y = cy + x_offset * sy_off + y_offset * cy_off
        yaw = math.atan2(vy, vx)
        vx = vx if abs(vx) > 0.1 else (0.1 if vx >= 0 else -0.1)
        rows.append(_gt_row(t, x, y, yaw, vx, vy))
    return rows


def _trajectory_polygon(rng: random.Random, duration_s: float, speed_mps: float, half_w: float, half_h: float) -> list[dict[str, float]]:
    """环形/多边形（paper 安全版）：用 5 段 sin 慢变 + 中心椭圆，
    无顶点切线跳变，yaw 单调累加 + sin 慢变防 ±π 边界。
    """
    rows: list[dict[str, float]] = []
    n = max(2, int(duration_s * 50))
    cx, cy = half_w, half_h
    # 主圆周：θ 单调推进
    base_period = 20.0  # 20s 走完一圈
    rx = half_w * 0.6
    ry = half_h * 0.5
    # 椭圆角速度：dθ/dt = 2π/base_period
    omega = 2 * math.pi / base_period
    # yaw 偏移：椭圆切线方向 + sin 慢变微调（防 yaw 接近 0 时 atan2 不稳定）
    # 椭圆切线角 = atan2(ry*cos(θ), -rx*sin(θ))
    # 引入 sin 偏移让 yaw 在每周期内 [-π, π] 范围内平滑动
    for i in range(n + 1):
        t = i * (duration_s / n)
        theta = omega * t
        x = cx + rx * math.cos(theta)
        y = cy + ry * math.sin(theta)
        # 切线方向（dy/dθ, dx/dθ = (-rx*sin, ry*cos)）
        # vx 方向 = (dx/dθ, dy/dθ)
        vx_dir_x = -rx * math.sin(theta)
        vx_dir_y = ry * math.cos(theta)
        # 归一化 + 缩放到 speed_mps
        mag = math.sqrt(vx_dir_x * vx_dir_x + vx_dir_y * vx_dir_y)
        if mag < 1e-6:
            vx, vy = speed_mps, 0.0
        else:
            vx = speed_mps * vx_dir_x / mag
            vy = speed_mps * vx_dir_y / mag
        # 强制 vx 始终非零（防 atan2 边界 ±π 跳变）
        vx = vx if abs(vx) > 0.5 else (0.5 if vx >= 0 else -0.5)
        yaw = math.atan2(vy, vx)
        rows.append(_gt_row(t, x, y, yaw, vx, vy))
    return rows


_TRAJ_BUILDERS = {
    "line_back_forth": _trajectory_line_back_forth,
    "random_walk": _trajectory_random_walk,
    "figure_eight": _trajectory_figure_eight,
    "stop_turn": _trajectory_stop_turn,
    "polygon": _trajectory_polygon,
}


def build_paper_gt_rows(
    traj_class: str,
    speed_bin: str,
    rng: random.Random,
    duration_s: float,
    site_xy: tuple[float, float],
    speed_range_mps: dict[str, list[float]],
) -> list[dict[str, float]]:
    """根据轨迹类 + 速度档生成 paper 专用 GT 行（50Hz 采样）。"""
    speed_lo, speed_hi = speed_range_mps[speed_bin]
    speed_mps = rng.uniform(speed_lo, speed_hi)
    half_w = site_xy[0] / 2.0
    half_h = site_xy[1] / 2.0
    builder = _TRAJ_BUILDERS[traj_class]
    return builder(rng, duration_s, speed_mps, half_w, half_h)


# ============================================================
# 连续段生成器（NLOS / 丢包）
# ============================================================

def _make_continuous_occlusion_segments(
    rng: random.Random,
    duration_s: float,
    block_range: tuple[float, float],
    count_range: tuple[int, int],
    warmup_s: float = 10.0,
) -> list[tuple[float, float]]:
    """生成连续遮挡段：每段 [block_lo, block_hi] 秒, 共 [count_lo, count_hi] 段。
    段间必须间隔 ≥ block_hi；warmup 期间不注入。
    """
    block_lo, block_hi = block_range
    count_lo, count_hi = count_range
    n_segments = rng.randint(count_lo, count_hi)
    segments: list[tuple[float, float]] = []
    safe_start = warmup_s
    safe_end = duration_s - block_hi - 1.0  # 留 1s 收尾
    if safe_end <= safe_start:
        return []
    for _ in range(n_segments):
        block_len = rng.uniform(block_lo, block_hi)
        start = rng.uniform(safe_start, safe_end - block_len)
        end = start + block_len
        segments.append((start, end))
    # 按 start 排序
    segments.sort(key=lambda p: p[0])
    return segments


def _make_burst_dropout_segments(
    rng: random.Random,
    duration_s: float,
    prob_range: tuple[float, float],
    burst_range: tuple[int, int],
    gap_range: tuple[float, float],
    interval_s: float = 1.0,
    warmup_s: float = 10.0,
) -> list[tuple[float, float]]:
    """生成丢包段：每 interval_s 一次伯努利抽样；命中后 burst 段 gap_range 秒。
    用于 uwb_outage_segments / vio_outage_segments。
    """
    prob = rng.uniform(*prob_range)
    burst_lo, burst_hi = burst_range
    gap_lo, gap_hi = gap_range
    segments: list[tuple[float, float]] = []
    t = warmup_s
    while t < duration_s - gap_lo - 0.5:
        if rng.random() < prob:
            gap_len = rng.uniform(gap_lo, gap_hi)
            segments.append((t, t + gap_len))
            t += gap_len
        else:
            t += interval_s
    return segments


# ============================================================
# 池构建（400 序列/seed，C1-C4 等比例 + 轨迹类/速度档轮转）
# ============================================================

def _build_paper_pool_layout(paper_spec: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """为单个 seed 生成 per_seed_sequences 条序列的池布局。

    按 pool_layout 等比例（每类 share × per_seed_sequences），每类内轨迹类×速度档轮转。
    每个 entry 含 A/N/K/V/M 五轴（V0/K0/M0 可缺省回退）。
    """
    pool_layout = paper_spec["pool_layout"]
    traj_classes = paper_spec["trajectory_classes"]
    speed_bins = paper_spec["speed_bins"]
    n_classes = len(pool_layout)
    per_seed_sequences = paper_spec["scale"]["per_seed_sequences"]
    rng = random.Random(seed * 1000 + 7919)  # paper 专用种子
    pool: list[dict[str, Any]] = []
    for layout in pool_layout:
        per_class_count = int(round(per_seed_sequences * layout.get("share", 1.0 / n_classes)))
        for i in range(per_class_count):
            traj = traj_classes[(seed * 17 + i) % len(traj_classes)]
            speed = speed_bins[(seed * 31 + i) % len(speed_bins)]
            pool.append({
                "class_id": layout["class_id"],
                "A": layout["A"],
                "N": layout["N"],
                "K": layout["K"],
                "V": layout.get("V", "V0"),
                "M": layout.get("M", "M0"),
                "traj_class": traj,
                "speed_bin": speed,
            })
    return pool


def _pool_index_to_seq_id(class_id: str, seed_id: int, idx: int) -> str:
    """paper seq_id 含 seed 前缀（pool 跨 seed 同 class 复用 paper_c1_0000 这种命名会冲突）。

    格式：paper_<class><seed_idx>_<in_seed_idx>
    例：seed=0 idx=0 → paper_c1_s0_0000；seed=2 idx=99 → paper_c3_s2_0099
    """
    return f"paper_{class_id.lower()}_s{seed_id}_{idx:04d}"


# ============================================================
# 主入口：materialize_paper_raw
# ============================================================

def load_paper_spec(paper_config_path: str | Path) -> dict[str, Any]:
    """从 paper_dataset.yaml 加载规格。"""
    path = Path(paper_config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"paper dataset config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def materialize_paper_raw(
    paper_config_path: str | Path,
    output_root: str | Path,
    *,
    n_seed: int | None = None,
    paper_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """物化 paper_main 原始数据集到 output_root/<seed{0..N-1}/<seq_id>/{imu,uwb,vio,gt}.json + sim_meta.json + anchor_layout.json。

    参数:
        paper_config_path: paper_dataset.yaml 路径
        output_root: 输出根目录；会自动创建 seed0/seed1/... 子目录
        n_seed: 覆盖 paper_spec.scale.n_seed（用于 quick 测试）
        paper_cfg: 可选预加载配置（避免重复 IO）

    返回:
        物化报告 dict，键为 seed_id 列表
    """
    if paper_cfg is None:
        paper_cfg = load_paper_spec(paper_config_path)
    spec = paper_cfg["paper_spec"]
    if n_seed is None:
        n_seed = spec["scale"]["n_seed"]
    out_root = Path(output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    site_xy = tuple(spec["site_xy_m"])
    duration_s = float(spec.get("effective_duration_s", 50.0)) + float(spec.get("warmup_s", 10.0))
    speed_range = spec["speed_range_mps"]
    block_range = tuple(spec["nlos_segment_block_s"])
    count_range = tuple(spec["nlos_segment_count_range"])
    uwb_drop_range = (spec["uwb_drop_prob_range"][0], spec["uwb_drop_prob_range"][1])
    uwb_burst_range = tuple(spec["uwb_drop_burst_range"])
    uwb_gap_range = (spec["uwb_drop_gap_s_range"][0], spec["uwb_drop_gap_s_range"][1])
    vio_drop_range = (spec["vio_drop_prob_range"][0], spec["vio_drop_prob_range"][1])
    vio_gap_range = (spec["vio_drop_gap_s_range"][0], spec["vio_drop_gap_s_range"][1])
    warmup_s = float(spec.get("warmup_s", 10.0))

    # 异步偏移/抖动表（按 A 档）
    uwb_offset = spec["async_uwb_offset_ms"]
    uwb_rtt = spec.get("async_uwb_rtt_ms", {})  # A1+ 才有，缺省回退到 offset
    uwb_jitter = spec["async_uwb_jitter_ms"]
    imu_offset = spec.get("async_imu_offset_ms", {})
    imu_jitter = spec.get("async_imu_jitter_ms", {})
    vio_offset = spec["async_vio_offset_ms"]
    vio_proc_delay = spec.get("async_vio_proc_delay_ms", {})
    vio_jitter = spec["async_vio_jitter_ms"]

    # NLOS 偏差/噪声
    nlos_bias = spec["nlos_uwb_bias_m"]
    nlos_noise = spec["nlos_uwb_noise_std_m"]

    # 锚点固定坐标（paper_main 数据集私有 anchor block 字典，键名 K4_layout 为 paper 内部索引名）
    anchor_K4_layout = spec["anchors"]["K4_layout"]

    # 报告
    report: dict[str, Any] = {"seeds": []}

    for seed_id in range(n_seed):
        seed_root = out_root / f"seed{seed_id}"
        seed_root.mkdir(parents=True, exist_ok=True)
        pool = _build_paper_pool_layout(spec, seed_id)
        seed_rng = random.Random(seed_id * 2024 + 11)
        seed_report = {
            "seed_id": seed_id,
            "seq_count": len(pool),
            "seq_ids": [],
        }
        for idx, entry in enumerate(pool):
            seq_id = _pool_index_to_seq_id(entry["class_id"], seed_id, idx)
            a_level = entry["A"]
            n_level = entry["N"]
            k_level = entry["K"]
            # 4 档 V/M 由 pool entry 直接给出（不再是全局 frozen）
            v_level = entry.get("V", "V0")
            m_level = entry.get("M", "M0")
            # paper 数据集私有 anchor block 选择符（K4_layout/K5_layout 为 paper 内部字典索引名，与五轴档位
            # 协议 K 轴的 K0/K1/K3 无关）。锚数全档固定 4（协议硬约束）。
            paper_k = k_level  # 保留 paper 原始 K 字段用于 anchor block 索引
            traj = entry["traj_class"]
            speed = entry["speed_bin"]
            class_id = entry["class_id"]

            # 1. 生成 GT 行（50Hz 采样）
            gt_rng = random.Random(seed_id * 100000 + idx * 31 + 7)
            gt_rows = build_paper_gt_rows(
                traj_class=traj,
                speed_bin=speed,
                rng=gt_rng,
                duration_s=duration_s,
                site_xy=site_xy,
                speed_range_mps=speed_range,
            )

            # 2. 构造 anchor_layout（2D [x, y] 格式，sensor_model 校验 exactly two coords）
            # 协议 spec 给 3D (z=2.5 贴顶) 但 liquidloc sensor_model 校验只接受 2D
            # 坐标，z 信息丢弃（融合 EKF 走 2D 平面）。
            # 按协议「锚数全档固定 4」仅保留 K4_layout=4 锚布局，与场景轴协议 K0/K1/K3 对齐（K3 是几何档位名而非锚数 3）。
            k_block = anchor_K4_layout
            if paper_k in k_block:
                anchor_positions = k_block[paper_k]
            else:  # 旧 spec 单层 (无 paper_k 维度) 兼容
                anchor_positions = list(k_block) if not isinstance(k_block, dict) else list(k_block.values())[0]
            anchor_positions = [list(p) for p in anchor_positions]
            anchor_positions_2d = [[float(p[0]), float(p[1])] for p in anchor_positions]
            anchor_ids = [f"A{i + 1}" for i in range(len(anchor_positions))]
            layout_id = seq_id  # layout_id == seq_id（contract 允许）
            anchor_layout = {
                "anchor_ids": anchor_ids,
                "anchor_positions": anchor_positions_2d,
                "anchor_count": len(anchor_positions),
                "layout_id": layout_id,
                "base_layout_id": f"paper_{k_level}",
                # inspect_sim_materialized_contract 通过 protocol_k_level 校验 K 档位
                "protocol_geometry_level": k_level,  # G 已并入 K，几何档位用 K
                "protocol_k_level": k_level,
                "geometry_report": {
                    "anchor_count": len(anchor_positions),
                    "layout_id": layout_id,
                    "geom_score": 1.0,
                    "paper_fixed": True,
                },
            }

            # 3. 构造 NLOS / 丢包段
            nlos_rng = random.Random(seed_id * 100000 + idx * 53 + 13)
            nlos_segments = _make_continuous_occlusion_segments(
                nlos_rng, duration_s, block_range, count_range, warmup_s
            )
            uwb_drop_rng = random.Random(seed_id * 100000 + idx * 71 + 17)
            uwb_drop_segments = _make_burst_dropout_segments(
                uwb_drop_rng, duration_s, uwb_drop_range, uwb_burst_range, uwb_gap_range, 1.0, warmup_s
            )
            vio_drop_rng = random.Random(seed_id * 100000 + idx * 89 + 19)
            vio_drop_segments = _make_burst_dropout_segments(
                vio_drop_rng, duration_s, vio_drop_range, (1, 4), vio_gap_range, 1.0, warmup_s
            )

            # 4. 异步偏移/抖动（按 A 档）
            a_uwb_offset_lo, a_uwb_offset_hi = uwb_offset[a_level]
            a_uwb_offset = seed_rng.uniform(a_uwb_offset_lo, a_uwb_offset_hi)
            a_uwb_jitter = uwb_jitter[a_level]
            # UWB 往返延迟：仅 A1+ 存在，缺省回退到 0
            a_uwb_rtt = 0.0
            if a_level in uwb_rtt and isinstance(uwb_rtt[a_level], list):
                rtt_lo, rtt_hi = uwb_rtt[a_level]
                a_uwb_rtt = seed_rng.uniform(rtt_lo, rtt_hi)
            # IMU 偏移/抖动：仅 A1+ 存在
            a_imu_offset = 0.0
            if a_level in imu_offset and isinstance(imu_offset[a_level], list):
                io_lo, io_hi = imu_offset[a_level]
                a_imu_offset = seed_rng.uniform(io_lo, io_hi)
            a_imu_jitter = imu_jitter.get(a_level, 0)
            # VIO 处理延迟：仅 A1+ 存在
            a_vio_proc_delay = 0.0
            if a_level in vio_proc_delay and isinstance(vio_proc_delay[a_level], list):
                vpd_lo, vpd_hi = vio_proc_delay[a_level]
                a_vio_proc_delay = seed_rng.uniform(vpd_lo, vpd_hi)
            # VIO 偏移：与 UWB 一样是 [lo, hi] 范围，需均匀采样
            a_vio_offset_lo, a_vio_offset_hi = vio_offset[a_level]
            a_vio_offset = seed_rng.uniform(a_vio_offset_lo, a_vio_offset_hi)
            a_vio_jitter = vio_jitter[a_level]

            # 5. sim_meta.json（含 class_id + seed + 全部轴）
            sim_meta = {
                "seq_id": seq_id,
                "base_seq_id": "paper_main",
                "axes_override": {
                    "A": a_level, "N": n_level, "V": v_level,
                    "K": k_level, "M": m_level,
                },
                "class_id": class_id,
                "traj_class": traj,
                "speed_bin": speed,
                "seed": seed_id,
                "nlos_segments": [list(s) for s in nlos_segments],
                "uwb_outage_segments": [list(s) for s in uwb_drop_segments],
                "vio_outage_segments": [list(s) for s in vio_drop_segments],
                "async_uwb_offset_ms": a_uwb_offset,
                "async_uwb_rtt_ms": a_uwb_rtt,
                "async_uwb_jitter_ms": a_uwb_jitter,
                "async_imu_offset_ms": a_imu_offset,
                "async_imu_jitter_ms": a_imu_jitter,
                "async_vio_offset_ms": a_vio_offset,
                "async_vio_proc_delay_ms": a_vio_proc_delay,
                "async_vio_jitter_ms": a_vio_jitter,
                "nlos_uwb_bias_m": nlos_bias[n_level],
                "nlos_uwb_noise_std_m": nlos_noise[n_level],
                "warmup_s": warmup_s,
                "duration_s": duration_s,
                "site_xy_m": list(site_xy),
                "v2_protocol_version": 1,
                "v2_documentation": "paper_main dataset 2026-08-29",
                # §28.6 自证契约：复用 sim_materializer 版本号，paper 物化是 sim 物化的
                # 同源扩展，version tag 共享 paper_dataset=true 字段标识为 paper 数据集。
                "generator_version": SIM_GENERATOR_VERSION_CURRENT,
                "paper_dataset": True,
            }

            # 6. 序列目录
            seq_root = seed_root / seq_id
            seq_root.mkdir(parents=True, exist_ok=True)

            # 7. 写 GT
            write_json(seq_root / "gt.json", gt_rows)

            # 8. 物化 imu/uwb/vio 流（纯几何推导 + spec 噪声）
            imu_hz = float(spec["imu_hz"])
            uwb_hz = float(spec["uwb_hz"])
            vio_hz = float(spec["vio_hz"])
            dt_imu = 1.0 / imu_hz
            dt_uwb = 1.0 / uwb_hz
            dt_vio = 1.0 / vio_hz
            t_start = float(gt_rows[0]["timestamp"])
            t_end = float(gt_rows[-1]["timestamp"])

            imu_timestamps = [t_start + i * dt_imu for i in range(int((t_end - t_start) / dt_imu) + 1)]
            vio_timestamps = [t_start + i * dt_vio for i in range(int((t_end - t_start) / dt_vio) + 1)]
            uwb_timestamps = [t_start + i * dt_uwb for i in range(int((t_end - t_start) / dt_uwb) + 1)]

            imu_rows = _derive_paper_imu_rows(gt_rows, imu_timestamps)
            vio_rows = _derive_paper_vio_rows(gt_rows, vio_timestamps)
            uwb_rows = _derive_paper_uwb_rows(gt_rows, uwb_timestamps, anchor_positions)

            # 9. IMU 噪声 + 异步偏移
            _inject_imu_noise(
                imu_rows,
                accel_white_psd=float(spec["imu_accel_white_psd"]),
                gyro_white_psd=float(spec["imu_gyro_white_psd"]),
                accel_bias_instab=float(spec["imu_accel_bias_instab"]),
                gyro_bias_instab=float(spec["imu_gyro_bias_instab"]),
                accel_rrw=float(spec["imu_accel_rrw"]),
                gyro_rrw=float(spec["imu_gyro_rrw"]),
                dt_s=dt_imu,
                seed=seed_id,
                seq_id=seq_id,
            )
            _inject_imu_async(imu_rows, a_imu_offset, a_imu_jitter, seed_id, seq_id)

            # 10. VIO 噪声 + 异步偏移 + 处理延迟
            _inject_vio_noise(
                vio_rows,
                dx_dy_std=float(spec["vio_dx_dy_std_m"]),
                yaw_drift_dps=float(spec["vio_yaw_drift_dps"]),
                seed=seed_id,
                seq_id=seq_id,
            )
            _inject_vio_async(vio_rows, a_vio_offset, a_vio_jitter, seed_id, seq_id)
            _inject_vio_proc_delay(vio_rows, a_vio_proc_delay, seed_id, seq_id)

            # 11. UWB 噪声
            _inject_uwb_noise(uwb_rows, float(spec["uwb_sigma_m"]), seed_id, seq_id)
            _inject_uwb_async(uwb_rows, a_uwb_offset, a_uwb_jitter, seed_id, seq_id)
            _inject_uwb_rtt(uwb_rows, a_uwb_rtt, seed_id, seq_id)

            # 12. NLOS 连续段注入（paper 语义；不走 sim 的 poisson scatter）
            nlos_level_cfg = {
                "bias_strength_m": nlos_bias[n_level],
                "nlos_noise_std_m": nlos_noise[n_level],
                "label": n_level,
            }
            uwb_rows, nlos_report = apply_paper_nlos_blocks(
                uwb_rows, nlos_segments, n_level, nlos_level_cfg, seq_id=seq_id
            )

            # 13. 丢包段（valid=False / quality=0；range 仍保留供审计）
            _apply_outage_segments(uwb_rows, uwb_drop_segments, zero_payload=False)
            _apply_outage_segments(vio_rows, vio_drop_segments, zero_payload=True)

            # 14. 写 imu/uwb/vio
            write_json(seq_root / "imu.json", imu_rows)
            write_json(seq_root / "uwb.json", uwb_rows)
            write_json(seq_root / "vio.json", vio_rows)
            # 15. 写 anchor_layout
            write_json(seq_root / "anchor_layout.json", anchor_layout)
            # 16. 写 sim_meta（追加 nlos_report）
            sim_meta["nlos_report"] = nlos_report
            write_json(seq_root / "sim_meta.json", sim_meta)

            seed_report["seq_ids"].append(seq_id)
        report["seeds"].append(seed_report)

    # 写总报告
    write_json(out_root / "materialize_report.json", report)
    return report


__all__ = [
    "load_paper_spec",
    "materialize_paper_raw",
    "build_paper_gt_rows",
    "_make_continuous_occlusion_segments",
    "_make_burst_dropout_segments",
    "_build_paper_pool_layout",
    "_TRAJ_BUILDERS",
]
