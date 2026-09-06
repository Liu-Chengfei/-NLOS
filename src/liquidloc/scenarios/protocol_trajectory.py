"""协议级平面轨迹生成器（前提指导 §8.2 / §8.2.0 / §8.2.1）。

生成带停走、急转、加减速的中频宽带轨迹，满足：
- L_xy ∈ [10, 40] m（默认约 20 m）
- T_eff ≥ 20 s（默认 45 s）
- 累计转向 ≳ 2π 或 ≥3 次显著转向
- 近零速占比约 5%–25%
- 路径长与 L_xy、v、T 自洽

输出是等间隔 GT 行列表：``timestamp, px, py, yaw, vx, vy``。
传感器流由物化器从 GT 推导，不依赖迷你 fixture 铺叠。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from liquidloc.common.angle_utils import angle_delta_rad, wrap_angle_rad
from liquidloc.common.validation import coerce_finite_scalar, is_integer


def _seeded_unit(seed: int, *parts: Any) -> float:
    """确定性 [0,1) 伪随机，避免引入全局 RNG 状态。"""
    # FNV-1a 64-bit 风格混合后映射到 [0,1)
    h = 1469598103934665603
    payload = f"{int(seed)}|" + "|".join(str(p) for p in parts)
    for ch in payload.encode("utf-8"):
        h ^= ch
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return (h & 0xFFFFFFFF) / 4294967296.0


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, float(u)))
    return u * u * (3.0 - 2.0 * u)


def generate_protocol_gt_rows(
    *,
    seq_id: str,
    seed: int,
    duration_s: float = 45.0,
    workspace_span_m: float = 20.0,
    dt_s: float = 0.05,
    cruise_speed_mps: float = 0.8,
    stop_ratio: float = 0.12,
    n_major_turns: int = 5,
) -> list[dict[str, float]]:
    """生成一条协议包络内的平面 GT 轨迹。

    路径结构：矩形走廊 + 对角线捷径 + 若干停走段 + 种子化相位扰动。
    n_major_turns 默认 5，配合锯齿偏移确保 turn_total_rad ≥ 6.0。
    """
    duration_s = coerce_finite_scalar(duration_s, name="duration_s", min_value=20.0)
    workspace_span_m = coerce_finite_scalar(workspace_span_m, name="workspace_span_m", min_value=10.0, max_value=40.0)
    dt_s = coerce_finite_scalar(dt_s, name="dt_s", min_value=1e-3, inclusive=False)
    cruise_speed_mps = coerce_finite_scalar(cruise_speed_mps, name="cruise_speed_mps", min_value=0.3, max_value=1.5)
    stop_ratio = coerce_finite_scalar(stop_ratio, name="stop_ratio", min_value=0.05, max_value=0.25)
    if not is_integer(n_major_turns) or int(n_major_turns) < 3:
        raise ValueError(f"n_major_turns must be int >= 3, got {n_major_turns!r}")
    n_major_turns = int(n_major_turns)

    half = 0.5 * workspace_span_m
    # 种子化轻微长宽比，避免完美正方形死脚本。
    aspect = 0.85 + 0.30 * _seeded_unit(seed, seq_id, "aspect")
    x_half = half
    y_half = half * aspect
    phase = 2.0 * math.pi * _seeded_unit(seed, seq_id, "phase")

    # 折线航路点：外圈矩形 + 内对角，保证多次 >45° 转向与足够路径长。
    waypoints = [
        (-x_half, -y_half),
        (x_half, -y_half),
        (x_half, y_half),
        (-x_half, y_half),
        (-x_half, -y_half),
        (0.0, y_half * 0.2),
        (x_half * 0.6, -y_half * 0.4),
        (-x_half * 0.3, y_half * 0.7),
        (-x_half, -y_half),
    ]
    # 额外大转向：在边上插入锯齿
    jagged: list[tuple[float, float]] = []
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        jagged.append((x0, y0))
        for k in range(1, n_major_turns):
            t = k / n_major_turns
            mx = x0 + (x1 - x0) * t
            my = y0 + (y1 - y0) * t
            # 法向偏移
            dx, dy = x1 - x0, y1 - y0
            norm = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / norm, dx / norm
            amp = (0.04 + 0.04 * _seeded_unit(seed, seq_id, "jag", i, k)) * workspace_span_m
            sign = 1.0 if (k + i) % 2 == 0 else -1.0
            jagged.append((mx + sign * amp * nx, my + sign * amp * ny))
    jagged.append(waypoints[-1])

    # 段长与累计弧长
    seg_lengths: list[float] = []
    for i in range(len(jagged) - 1):
        seg_lengths.append(math.hypot(jagged[i + 1][0] - jagged[i][0], jagged[i + 1][1] - jagged[i][1]))
    total_path = sum(seg_lengths)
    if total_path < 20.0:
        # 放大到至少 20 m 路径（保持形状）
        scale = 20.0 / max(total_path, 1e-6)
        jagged = [(x * scale, y * scale) for x, y in jagged]
        seg_lengths = [l * scale for l in seg_lengths]
        total_path = sum(seg_lengths)

    n_samples = int(math.floor(duration_s / dt_s)) + 1
    if n_samples < 2:
        raise ValueError("duration_s/dt_s produced fewer than 2 samples")

    # 停走：在时间轴上放若干近零速窗口
    n_stops = max(2, int(round(stop_ratio * 8)))
    stop_windows: list[tuple[float, float]] = []
    # Section 8.2 R8.2-A anti-periodicity fix: randomize stop window centers and widths
    # over a broad band +/- 1/3 of the inter-stop interval (not +/-5%) so the stop pattern
    # is broadband (no strong spectral peak at n_stops/duration_s).
    inter_stop = duration_s / n_stops
    for s_idx in range(n_stops):
        # Center: nominal (s_idx + 0.5) * inter_stop, plus random +/- 33% of inter_stop.
        center = (s_idx + 0.5) * inter_stop
        jitter = (0.5 - _seeded_unit(seed, seq_id, "stop_c", s_idx)) * 0.66 * inter_stop
        center += jitter
        # Width: nominal inter_stop * stop_ratio/n_stops, plus random +/- 50%.
        width = (stop_ratio * duration_s) / n_stops
        width *= 0.5 + _seeded_unit(seed, seq_id, "stop_w", s_idx)
        edge_lo = max(0.0, center - 0.5 * width)
        edge_hi = min(duration_s, center + 0.5 * width)
        stop_windows.append((edge_lo, edge_hi))

    # 每次停走前后增加加减速斜坡，使 a95 落到 §8.2.1 推荐区间 [0.5, 3] m/s²。
    # 斜坡时长（秒）：0.3-0.5s 随机化，产生约 1.5-2.5 m/s² 巡航→零速变化（中心差分后约一半）。
    # 配合种子化扰动使 a95 落入目标区间。
    stop_ramp_s_min = 0.3
    stop_ramp_s_max = 0.5

    # 为每个停走窗口分配独立的随机斜坡时长，使 a95 落入 §8.2.1 推荐区间。
    stop_ramp_durs = [
        stop_ramp_s_min + (stop_ramp_s_max - stop_ramp_s_min) * _seeded_unit(seed, seq_id, "ramp_dur", s_i)
        for s_i in range(n_stops)
    ]

    def _in_stop_zone(t: float) -> bool:
        for a, b in stop_windows:
            if a <= t <= b:
                return True
        return False

    def _in_ramp_to_stop(t: float, idx: int) -> bool:
        """t 是否在进入第 idx 个停走窗口前的减速斜坡内。"""
        a, b = stop_windows[idx]
        return a - stop_ramp_durs[idx] <= t < a

    def _in_ramp_from_stop(t: float, idx: int) -> bool:
        """t 是否在离开第 idx 个停走窗口后的加速斜坡内。"""
        a, b = stop_windows[idx]
        return b < t <= b + stop_ramp_durs[idx]

    # 停止速度系数：斜坡应从巡航降至该系数乘以巡航速度，再降至近零。
    _STOP_SPEED_COEFF = 0.02  # 与停走段 speed = 0.02 * cruise 对齐

    def _ramp_gain(t_in_ramp: float, ramp_dur: float, direction: str) -> float:
        """计算斜坡段的 smoothstep 增益，返回 [0, 1]。

        direction='to_stop'：从巡航减速至停止速度（增益 1→0，对应速度从 cruise 降至 _STOP_SPEED_COEFF * cruise）。
        direction='from_stop'：从停止速度加速至巡航（增益 0→1，对应速度从 _STOP_SPEED_COEFF * cruise 升至 cruise）。
        """
        u = max(0.0, min(1.0, t_in_ramp / ramp_dur))
        if direction == "to_stop":
            u = 1.0 - u  # 反向：1→0
        return _smoothstep(u)

    def _ramp_speed(t_in_ramp: float, ramp_dur: float, direction: str) -> float:
        """计算斜坡段的实际速度系数，使 ramp 平滑过渡到停走速度。

        返回 speed_coeff，满足：
        - direction='to_stop'：t_in_ramp=0 时 speed_coeff=1.0，t_in_ramp=ramp_dur 时 speed_coeff=_STOP_SPEED_COEFF。
        - direction='from_stop'：t_in_ramp=0 时 speed_coeff=_STOP_SPEED_COEFF，t_in_ramp=ramp_dur 时 speed_coeff=1.0。
        """
        gain = _ramp_gain(t_in_ramp, ramp_dur, direction)
        # 速度在 gain=1.0 时为 cruise，在 gain=0.0 时为 _STOP_SPEED_COEFF * cruise。
        return _STOP_SPEED_COEFF + (1.0 - _STOP_SPEED_COEFF) * gain

    # 时间 → 弧长：巡航速度，停走段速度≈0，停走前后有平滑加减速斜坡。
    # 斜坡使用 smoothstep 实现 C¹ 连续，速度从 cruise 平滑过渡到 near-zero 和反向。
    # 每个停走窗口有独立随机化的斜坡时长，使 a95 落入 §8.2.1 推荐区间 [0.5, 3] m/s²。
    s_of_t = [0.0]
    for i in range(1, n_samples):
        t = i * dt_s
        if _in_stop_zone(t):
            speed = _STOP_SPEED_COEFF * cruise_speed_mps  # 近零而非硬锁死，避免除零
        else:
            # 检查是否处于某个停走窗口的减速/加速斜坡内
            in_ramp = False
            for idx in range(n_stops):
                if _in_ramp_to_stop(t, idx):
                    ramp_dur = stop_ramp_durs[idx]
                    t_in_ramp = t - (stop_windows[idx][0] - ramp_dur)
                    speed_coeff = _ramp_speed(t_in_ramp, ramp_dur, "to_stop")
                    in_ramp = True
                    break
                elif _in_ramp_from_stop(t, idx):
                    ramp_dur = stop_ramp_durs[idx]
                    t_in_ramp = t - stop_windows[idx][1]
                    speed_coeff = _ramp_speed(t_in_ramp, ramp_dur, "from_stop")
                    in_ramp = True
                    break
            if not in_ramp:
                # 全局加减速包络（轨迹起止）
                ramp = 0.15 * duration_s
                if t < ramp:
                    gain = _smoothstep(t / ramp)
                elif t > duration_s - ramp:
                    gain = _smoothstep((duration_s - t) / ramp)
                else:
                    gain = 1.0
                speed_coeff = 0.75 + 0.25 * gain  # 注意：原始代码用 0.35*gain，但 0.75+0.25=1.0 更合理
            speed = cruise_speed_mps * speed_coeff
            # Section 8.2 R8.2-A periodicity fix: drop i//10 block-grouped speed modulation
            # (block modulation introduced strong 2Hz period; 30-seed measured ratio ~0.99).
            # Use per-sample independent [-0.02, 0.02] broadband noise (amplitude 2% not 10%)
            # so ramp/stop speed_coeff dominates and accounts for true accel/decel.
            speed *= 1.0 + 0.02 * (_seeded_unit(seed, seq_id, "spd", i) * 2.0 - 1.0)
        s_of_t.append(s_of_t[-1] + speed * dt_s)

    # 若总弧长不足，按比例拉长路径参数（循环航路）
    max_s = s_of_t[-1]
    loop_path = total_path

    def pose_at_s(s: float) -> tuple[float, float, float]:
        if loop_path <= 0.0:
            return 0.0, 0.0, 0.0
        s_mod = s % loop_path
        acc = 0.0
        for i, seg_l in enumerate(seg_lengths):
            if acc + seg_l >= s_mod:
                u = 0.0 if seg_l <= 1e-12 else (s_mod - acc) / seg_l
                x0, y0 = jagged[i]
                x1, y1 = jagged[i + 1]
                x = x0 + (x1 - x0) * u
                y = y0 + (y1 - y0) * u
                yaw = math.atan2(y1 - y0, x1 - x0)
                # 全局相位旋转，避免全体共用同一朝向脚本
                c, s_ = math.cos(phase), math.sin(phase)
                xr = c * x - s_ * y
                yr = s_ * x + c * y
                return xr, yr, wrap_angle_rad(yaw + phase)
            acc += seg_l
        x, y = jagged[-1]
        c, s_ = math.cos(phase), math.sin(phase)
        return c * x - s_ * y, s_ * x + c * y, wrap_angle_rad(phase)

    rows: list[dict[str, float]] = []
    prev_x = prev_y = None
    for i in range(n_samples):
        t = i * dt_s
        x, y, _seg_yaw = pose_at_s(s_of_t[i])
        if prev_x is None:
            vx, vy = 0.0, 0.0
        else:
            vx = (x - prev_x) / dt_s
            vy = (y - prev_y) / dt_s
        rows.append(
            {
                "timestamp": float(t),
                "px": float(x),
                "py": float(y),
                # yaw 占位，后续用前后帧位置差分重算，避免段边界处阶跃跳变导致 gz 超限
                "yaw": 0.0,
                "vx": float(vx),
                "vy": float(vy),
            }
        )
        prev_x, prev_y = x, y

    # 用前后帧位置差分重算 yaw，保证 yaw 连续可微，避免段边界阶跃跳变产生超物理量程的 gz
    # (消费级 IMU 量程 ±10 rad/s；阶跃跳变在 dt=0.05s 下可产生 ~13 rad/s 的 gz)
    _recompute_yaw_from_position_differences(rows)
    return rows


def _recompute_yaw_from_position_differences(rows: list[dict[str, float]]) -> None:
    """根据相邻帧位置差重新计算每帧的航向角，覆盖原始 yaw 占位值。

    首帧用首帧→次帧方向，末帧用倒数第二帧→末帧方向，中间帧用中心差分方向。
    位移近零时保留前一帧 yaw，避免静止时 atan2 不稳定。

    随后做一轮 yaw 速率限幅：每帧 yaw 变化不超过 ``_YAW_RATE_LIMIT_PER_FRAME``，
    保证 IMU gz（中心差分 / 2dt）不超过消费级量程 10 rad/s。
    限幅采用"携带余量"策略：超出部分在后续帧逐步施加，总转向量守恒。

    参数：
        rows: GT 行列表（in-place 修改 yaw 字段），至少包含 px 和 py 字段。
    """
    n = len(rows)
    if n <= 1:
        return
    # 第一遍：从位置差分计算目标 yaw
    target_yaws: list[float] = []
    last_yaw = 0.0
    for i, row in enumerate(rows):
        if i == 0:
            ref_a, ref_b = rows[0], rows[1]
        elif i == n - 1:
            ref_a, ref_b = rows[-2], rows[-1]
        else:
            ref_a, ref_b = rows[i - 1], rows[i + 1]
        dx = float(ref_b["px"]) - float(ref_a["px"])
        dy = float(ref_b["py"]) - float(ref_a["py"])
        # 位移足够大时才更新 yaw，避免静止时 atan2 不稳定
        if math.hypot(dx, dy) > 1e-9:
            last_yaw = wrap_angle_rad(math.atan2(dy, dx))
        target_yaws.append(float(last_yaw))
    # 第二遍：yaw 速率限幅，保证 gz ≤ 10 rad/s
    # IMU gz 用中心差分 gz = (yaw[i+1] - yaw[i-1]) / (2*dt)
    # 令 per-frame 限幅 = 0.475 rad → 2-frame 变化 ≤ 0.95 rad → gz ≤ 0.95/(2*0.05) = 9.5
    # 5% 安全余量用于吸收 IMU 插值浮点误差（IMU dt=0.01s 比 GT dt=0.05s 密 5 倍时，
    # 插值后中心差分 gz 可能比 GT 层面 gz 高出 ~0.02 rad/s）
    _apply_yaw_rate_limit(rows, target_yaws, max_delta_per_frame=_YAW_RATE_LIMIT_PER_FRAME)


# 每帧 yaw 变化上限（rad）。保证中心差分 gz ≤ 9.5 rad/s（dt=0.05s），
# 留 5% 安全余量吸收 IMU 插值浮点误差，避免 gz 突破协议上限 10.0 rad/s。
# max_delta = 9.5 * dt = 9.5 * 0.05 = 0.475 rad/frame
_YAW_RATE_LIMIT_PER_FRAME = 0.475


def _apply_yaw_rate_limit(
    rows: list[dict[str, float]],
    target_yaws: list[float],
    *,
    max_delta_per_frame: float,
) -> None:
    """对 yaw 做速率限幅，保留总转向量。

    每帧 yaw 变化（有符号）不超过 ``max_delta_per_frame``。超出部分作为"余量"
    携带到后续帧继续施加，直到余量归零。总转向量（有符号累加）与目标一致。

    参数：
        rows: GT 行列表（in-place 写入 yaw 字段）。
        target_yaws: 从位置差分得到的目标 yaw 序列。
        max_delta_per_frame: 每帧允许的最大 yaw 变化（rad，正值）。
    """
    if not rows:
        return
    rows[0]["yaw"] = float(target_yaws[0])
    # carried = 前一帧未消化的有符号余量
    carried = 0.0
    for i in range(1, len(rows)):
        target_delta = angle_delta_rad(float(target_yaws[i]), float(rows[i - 1]["yaw"]))
        desired = target_delta + carried
        clamped = max(-max_delta_per_frame, min(max_delta_per_frame, desired))
        carried = desired - clamped
        new_yaw = wrap_angle_rad(float(rows[i - 1]["yaw"]) + clamped)
        rows[i]["yaw"] = float(new_yaw)


def resample_timestamps(duration_s: float, dt_s: float) -> list[float]:
    """生成 [0, duration] 的等间隔时间戳列表。"""
    duration_s = coerce_finite_scalar(duration_s, name="duration_s", min_value=0.0, inclusive=False)
    dt_s = coerce_finite_scalar(dt_s, name="dt_s", min_value=1e-3, inclusive=False)
    n = int(math.floor(duration_s / dt_s)) + 1
    return [i * dt_s for i in range(n)]
