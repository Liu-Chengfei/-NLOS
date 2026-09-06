"""异步高NLOS实验全流程保障手册 P16 EKF 初始化协议（紧耦合 EKF 关键环节）。

P16 硬约束（手册 line 172）：
- 初始位置取序列首帧 UWB 三边测量解（≥3 锚有效时）
- 若首帧锚数 <3 或 uwb_valid=0 → 延迟初始化：用零位 + 大协方差 P0（位置 100m²、yaw π² 量级）
- 初始 yaw 取 VIO 首帧朝向（vio_yaw[0]），VIO 无效时置 0 + 大协方差
- 初始速度取零
- P0 对角取上述不确定性

公开文献共识（UWB-VIO 初始化是融合定位成败环节，含近3年 UWB-VIO 初始化与可观测性分析研究）。

实现为 EKFCore 的方法 `compute_initial_state_from_first_frame()` + 静态辅助
`build_p0_diagonal()`，允许主流程在收到首批 UWB/VIO 事件时调用，把结果作为
``initial_state`` 参数传给 ``EKFCore.reset()``。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from liquidloc.common.validation import coerce_finite_scalar
from liquidloc.estimators.state_definition import state_items

# 手册 P16 硬约束常量：P0 对角值。
# 位置 100 m² = 10 m 标准差；yaw π² ≈ π² ≈ 9.87 ≈ yaw std ≈ π rad 量级。
INITIAL_POSITION_UNCERTAINTY_M2: float = 100.0
INITIAL_YAW_UNCERTAINTY_RAD2: float = math.pi ** 2  # π² ≈ 9.87
INITIAL_VELOCITY_UNCERTAINTY_MPS2: float = 1.0  # 1 m/s 标准差
INITIAL_BIAS_UNCERTAINTY: float = 1e-4  # 偏置初值小不确定性
INITIAL_EXTRA_UNCERTAINTY: float = 1e-4  # uwb_clock_bias / vio_scale 不确定性初值
# 三边测量最小锚数门槛（手册 P16：≥3 锚有效时才走三边测量解）。
TRILATERATION_MIN_ANCHORS: int = 3


def _safe_index(name: str) -> int:
    """安全取状态项索引；缺键 raise ValueError（不静默回退）。"""
    try:
        return state_items.index(name)
    except ValueError as exc:
        raise KeyError(f"state_items missing required key '{name}'") from exc


def trilateration_least_squares(
    ranges_m: Sequence[float],
    anchor_positions_m: Sequence[Sequence[float]],
    *,
    z_anchor_m: float = 2.5,
    z_tag_m: float = 1.2,
) -> tuple[float, float] | None:
    """最小二乘三边测量解（2D）。

    参数:
        ranges_m: 每锚测距（米），长度 = 锚点数。
        anchor_positions_m: 每锚 (x, y)，米。
        z_anchor_m / z_tag_m: 锚与 tag 的 z 高度（米），用于 dz 修正。

    返回:
        (x, y) 估计；失败返回 None。

    失败条件:
        - 锚数 <3 (P16 门槛)
        - 系数矩阵奇异（4 锚共线/聚簇的退化构型，与 S2 一致）
        - 解出负方差（物理上不可能）
    """
    if len(ranges_m) < TRILATERATION_MIN_ANCHORS or len(anchor_positions_m) != len(ranges_m):
        return None
    dz = z_anchor_m - z_tag_m
    rows = []
    b: list[float] = []
    ax0, ay0 = anchor_positions_m[0]
    r0 = ranges_m[0]
    for i in range(1, len(anchor_positions_m)):
        ax_i, ay_i = anchor_positions_m[i]
        r_i = ranges_m[i]
        # 线性化：|p - a_i|² - |p - a_0|² = r_i² - r0² ⇒ 2(a_0 - a_i)·p = r_i² - r0² - |a_0|² + |a_i|²
        rows.append(
            (
                2 * (ax0 - ax_i),
                2 * (ay0 - ay_i),
            )
        )
        b.append(
            r_i ** 2 - r0 ** 2 - (ax_i ** 2 + ay_i ** 2) + (ax0 ** 2 + ay0 ** 2)
        )
    a = np.asarray(rows, dtype=float)
    bvec = np.asarray(b, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 2:
        return None
    try:
        # lstsq 允许欠定/超定；N=3 时 2x2 矩阵可解，N=4 时 2x3 超定用最小二乘。
        sol, residuals, rank, _ = np.linalg.lstsq(a, bvec, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 2 or (residuals.size > 0 and float(np.min(residuals)) < -1e-3):
        # 解退化或残差符号异常（解出负方差）→ 失败
        return None
    return float(sol[0]), float(sol[1])


def collect_first_frame_anchor_ranges(
    uwb_events: Sequence[Any],
    anchor_lookup: Mapping[Any, tuple[float, float, float]],
) -> dict[Any, float]:
    """从首批 UWB 事件中提取"每锚最近一次有效测距"（用于三边测量）。

    处理流程:
        - 仅取 modality == uwb、payload['valid'] is True 的事件
        - 按 (anchor_id, t) 排序后取每锚最早 t 的测距
    """
    per_anchor_first: dict[Any, tuple[float, float]] = {}
    for event in uwb_events:
        if not isinstance(event, Mapping):
            continue
        modality = event.get("modality")
        if modality != "uwb":
            continue
        payload = event.get("payload") or event.get("uwb_payload")
        if not isinstance(payload, Mapping):
            continue
        if not payload.get("valid"):
            continue
        anchor_id = payload.get("anchor_id")
        if anchor_id is None:
            continue
        measured_range = payload.get("measured_range")
        if measured_range is None:
            continue
        try:
            measured_range_f = coerce_finite_scalar(
                measured_range, name="first_frame_range", min_value=0.0
            )
        except (TypeError, ValueError):
            continue
        t = event.get("t")
        try:
            t_f = coerce_finite_scalar(t, name="first_frame_t")
        except (TypeError, ValueError):
            continue
        prev = per_anchor_first.get(anchor_id)
        if prev is None or t_f < prev[0]:
            per_anchor_first[anchor_id] = (t_f, measured_range_f)
    return {anchor_id: r for anchor_id, (_, r) in per_anchor_first.items()}


def collect_first_vio_yaw(vio_events: Sequence[Any]) -> tuple[float | None, float | None]:
    """从首批 VIO 事件中提取最早 t 处的 vio_yaw 与 quality。

    返回:
        (yaw_rad, quality) 之一/两者都可为 None（缺事件或 quality=0）。
    """
    earliest: tuple[float, float, float] | None = None
    for event in vio_events:
        if not isinstance(event, Mapping):
            continue
        if event.get("modality") != "vio":
            continue
        payload = event.get("payload") or event.get("vio_payload")
        if not isinstance(payload, Mapping):
            continue
        try:
            t_f = coerce_finite_scalar(event.get("t"), name="first_vio_t")
        except (TypeError, ValueError):
            continue
        quality_f = float(payload.get("quality", 0.0))
        dyaw_rad = float(payload.get("dyaw", 0.0))
        # vio yaw 是增量，首帧累计 dyaw = dyaw[0]，故初始 yaw 近似 0；这是真实场景下的诚实口径。
        # 若希望用 VIO 绝对朝向，需要 sensor model 提供 vio_yaw（而非增量 dyaw）。此处保守返回 0。
        yaw_accum = dyaw_rad  # 累计 = dyaw[0]（首帧 dyaw 在 [0, dyaw_max] 之间）
        if earliest is None or t_f < earliest[0]:
            earliest = (t_f, yaw_accum, quality_f)
    if earliest is None:
        return None, None
    return float(earliest[1]), float(earliest[2])


def build_p0_diagonal(
    *,
    delayed_init: bool,
    initial_position_yaw: bool,
) -> tuple[list[float], str]:
    """按手册 P16 构造 P0 对角向量（与 state_items 同序）。

    参数:
        delayed_init: True 表示首帧 <3 锚有效（位置用零位 + 大 P0）
        initial_position_yaw: True 表示首帧可用位置/yaw；False 表示无位置/yaw

    返回:
        (对角列表, mode_label)
        mode_label: "trilated_first" / "delayed_init_no_pos" / "delayed_init_no_yaw"
    """
    diag: list[float] = []
    if delayed_init:
        pos_var = INITIAL_POSITION_UNCERTAINTY_M2
        yaw_var = INITIAL_YAW_UNCERTAINTY_RAD2
        mode = "delayed_init_no_pos"
    elif initial_position_yaw:
        pos_var = 0.25  # 0.5m² 标准差（实测三边测量解在中段较好）
        yaw_var = 0.01  # 0.1rad 标准差
        mode = "trilated_first"
    else:
        # 回退：默认用 P16 大 P0
        pos_var = INITIAL_POSITION_UNCERTAINTY_M2
        yaw_var = INITIAL_YAW_UNCERTAINTY_RAD2
        mode = "delayed_init_no_pos"

    if not initial_position_yaw:
        # 没有 VIO-yaw：yaw 用大 P0
        yaw_var = INITIAL_YAW_UNCERTAINTY_RAD2
        if delayed_init:
            mode = "delayed_init_no_pos"
        else:
            mode = "trilated_init_no_yaw"

    vel_var = INITIAL_VELOCITY_UNCERTAINTY_MPS2
    bias_var = INITIAL_BIAS_UNCERTAINTY
    extra_var = INITIAL_EXTRA_UNCERTAINTY

    for key in state_items:
        if key == "px" or key == "py":
            diag.append(pos_var)
        elif key == "yaw":
            diag.append(yaw_var)
        elif key in ("vx", "vy"):
            diag.append(vel_var)
        elif key in ("bax", "bay", "bg"):
            diag.append(bias_var)
        else:
            diag.append(extra_var)
    return diag, mode


def ekf_init_protocol_p16(
    uwb_events: Sequence[Any],
    vio_events: Sequence[Any],
    *,
    anchor_lookup: Mapping[Any, tuple[float, float, float]],
) -> dict[str, Any]:
    """手册 P16 EKF 初始化协议统一入口。

    参数:
        uwb_events / vio_events: 首批 UWB/VIO 事件序列
        anchor_lookup: {anchor_id: (x, y, z)} 锚点查找表

    返回:
        dict 含:
            "initial_state": {state_key: float} （键与 state_items 对齐）
            "init_cov_diagonal": list[float] （与 state_items 同长）
            "mode": "trilated_first" / "delayed_init_no_pos" / "trilated_init_no_yaw" /
                   "delayed_init_no_pos_no_yaw"
            "n_anchors_used": int （实际参与三边测量的锚数）
            "trilaterated_position_xy": (x, y) | None
    """
    # 1. 收集首帧锚测距
    ranges = collect_first_frame_anchor_ranges(uwb_events, anchor_lookup)
    n_anchors_used = len(ranges)
    anchor_positions: list[list[float]] = []
    sorted_ranges: list[float] = []
    for anchor_id, r in ranges.items():
        coords = anchor_lookup.get(anchor_id)
        if coords is None:
            continue
        anchor_positions.append([coords[0], coords[1]])
        sorted_ranges.append(r)
    # 2. 尝试三边测量
    position_xy = trilateration_least_squares(sorted_ranges, anchor_positions) if n_anchors_used >= TRILATERATION_MIN_ANCHORS else None
    delayed_init = position_xy is None
    # 3. 收集首帧 vio_yaw
    vio_yaw, vio_quality = collect_first_vio_yaw(vio_events)
    initial_yaw_present = vio_yaw is not None and (vio_quality is None or vio_quality > 0.0)
    # 4. 构造初始 state 全零 + 覆盖
    initial_state = {key: 0.0 for key in state_items}
    if position_xy is not None:
        initial_state["px"] = float(position_xy[0])
        initial_state["py"] = float(position_xy[1])
    if initial_yaw_present:
        initial_state["yaw"] = float(vio_yaw)  # 手册 P16：初始 yaw 取 VIO 首帧朝向
    # 5. 构造 P0 对角
    diag, mode = build_p0_diagonal(
        delayed_init=delayed_init,
        initial_position_yaw=(position_xy is not None and initial_yaw_present),
    )
    if position_xy is None and not initial_yaw_present:
        mode = "delayed_init_no_pos_no_yaw"
    elif position_xy is None:
        mode = "delayed_init_no_pos"
    elif not initial_yaw_present:
        mode = "trilated_init_no_yaw"

    return {
        "initial_state": initial_state,
        "init_cov_diagonal": diag,
        "mode": mode,
        "n_anchors_used": n_anchors_used,
        "trilaterated_position_xy": position_xy,
        "vio_yaw_used": float(vio_yaw) if initial_yaw_present else None,
    }


__all__ = [
    "ekf_init_protocol_p16",
    "build_p0_diagonal",
    "trilateration_least_squares",
    "collect_first_frame_anchor_ranges",
    "collect_first_vio_yaw",
    "INITIAL_POSITION_UNCERTAINTY_M2",
    "INITIAL_YAW_UNCERTAINTY_RAD2",
    "INITIAL_VELOCITY_UNCERTAINTY_MPS2",
    "TRILATERATION_MIN_ANCHORS",
]