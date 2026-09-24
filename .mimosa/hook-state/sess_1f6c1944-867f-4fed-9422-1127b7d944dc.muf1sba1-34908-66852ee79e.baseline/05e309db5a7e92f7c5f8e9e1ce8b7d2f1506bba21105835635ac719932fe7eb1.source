"""回归测试：paper_materializer._trajectory_random_walk 轨迹出域防护（避免 RMSE 过大准则 §5）。

测试覆盖：
1.  随机游走轨迹在长时运行下，所有真值点 (px, py) 必须落在锚区凸包内。
2.  位置钳制后速度分量必须反向（防钉边界振荡）。
3.  yaw 单调推进 + sin 驱动 heading 慢变：避免 yaw ±π 边界跳变。

设计意图：
- `_trajectory_random_walk` 原版在 (cx < 0.5 || cx > 2*half_w - 0.5) 时只钳制位置，不翻转
  速度，导致速度方向持续指向边界外、位置卡死在边界上，是 RMSE 膨胀的典型诱因。
- 修复后 clamp 同步翻转对应速度分量 (vx 或 vy)，保证轨迹在凸包内不抖动。
"""

from __future__ import annotations

import math
import random

import pytest

from liquidloc.dataio.paper_materializer import _trajectory_random_walk


def _in_anchor_convex_hull(px: float, py: float, half_w: float, half_h: float, margin: float = 0.5) -> bool:
    """判定位置是否在锚区凸包内（含 0.5m 边距缓冲）。"""
    return (margin <= px <= 2.0 * half_w - margin) and (margin <= py <= 2.0 * half_h - margin)


def test_random_walk_stays_in_anchor_convex_hull_long_horizon() -> None:
    """长时随机游走（120s）应全程不超出锚区凸包。"""
    rng = random.Random(2026)
    rows = _trajectory_random_walk(
        rng,
        duration_s=120.0,
        speed_mps=1.5,
        half_w=10.0,  # site_xy = 20m → 半宽 10m
        half_h=10.0,
    )
    # 检查所有真值点都在 [0.5, 19.5] 凸包内（margin=0.5m）。
    out_of_domain = []
    for row in rows:
        px = row["px"]
        py = row["py"]
        if not _in_anchor_convex_hull(px, py, half_w=10.0, half_h=10.0, margin=0.5):
            out_of_domain.append((px, py))
    assert not out_of_domain, (
        f"出域点数={len(out_of_domain)}（前 3 个={out_of_domain[:3]}），"
        f"违反避免 RMSE 过大准则 §5「轨迹不出域」"
    )


def test_random_walk_does_not_stick_to_boundary() -> None:
    """修复：边界 clamp 后必须翻转速度方向，不能持续卡死在边界。"""
    rng = random.Random(7)
    rows = _trajectory_random_walk(
        rng,
        duration_s=60.0,
        speed_mps=2.0,
        half_w=10.0,
        half_h=10.0,
    )
    # 检查边界附近的位置是否来回振荡（而非卡死）。
    boundary_count_left = 0
    boundary_count_right = 0
    consecutive_at_boundary = 0
    max_consecutive_at_boundary = 0
    for row in rows:
        px = row["px"]
        if px <= 0.55:
            boundary_count_left += 1
            consecutive_at_boundary += 1
            max_consecutive_at_boundary = max(max_consecutive_at_boundary, consecutive_at_boundary)
        elif px >= 19.45:
            boundary_count_right += 1
            consecutive_at_boundary += 1
            max_consecutive_at_boundary = max(max_consecutive_at_boundary, consecutive_at_boundary)
        else:
            consecutive_at_boundary = 0
    # 钳制翻转后，轨迹不应在边界连续停留超过 3 帧（at 50Hz → 60ms）。
    assert max_consecutive_at_boundary <= 3, (
        f"轨迹在边界连续停留 {max_consecutive_at_boundary} 帧，"
        f"违反避免 RMSE 过大准则 §5「轨迹不出域」（应被钳制+速度翻转）"
    )


def test_random_walk_yaw_no_pi_boundary_jump() -> None:
    """heading 慢变 sin 叠加：yaw 不应有 ±π 跳变。"""
    rng = random.Random(99)
    rows = _trajectory_random_walk(
        rng,
        duration_s=30.0,
        speed_mps=1.0,
        half_w=10.0,
        half_h=10.0,
    )
    prev_yaw = rows[0]["yaw"]
    for row in rows[1:]:
        yaw = row["yaw"]
        delta = abs(_shortest_angle_delta(prev_yaw, yaw))
        # 单帧 delta 不应接近 π（防止 atan2(vy, vx) 边界跳变）
        assert delta < math.pi - 0.01, (
            f"yaw 跳变 {delta:.4f} rad 接近 ±π 边界，"
            f"违反避免 RMSE 过大准则 §7「旋转链路正确」"
        )
        prev_yaw = yaw


def test_random_walk_finite_outputs() -> None:
    """所有输出必须有限（避免 RMSE 过大准则 §24「无 NaN/Inf」）。"""
    rng = random.Random(1234)
    rows = _trajectory_random_walk(
        rng,
        duration_s=60.0,
        speed_mps=1.0,
        half_w=10.0,
        half_h=10.0,
    )
    for idx, row in enumerate(rows):
        for key in ("timestamp", "px", "py", "yaw", "vx", "vy"):
            value = row[key]
            assert math.isfinite(value), (
                f"行 {idx} 字段 {key} 非有限值={value}，违反避免 RMSE 过大准则 §24"
            )


def _shortest_angle_delta(lhs: float, rhs: float) -> float:
    """ 辅助：最短环形角差。"""
    delta = lhs - rhs
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta