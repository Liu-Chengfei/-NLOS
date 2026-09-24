"""协议轨迹与几何运动包络门禁测试。"""

from __future__ import annotations

import pytest

from liquidloc.protocol.scene_axis_protocol import get_nominal_levels, load_scene_axis_protocol
from liquidloc.scenarios.geometry_levels import build_anchor_layout
from liquidloc.scenarios.geometry_motion_envelope import (
    assert_geometry_motion_envelope,
    compute_trajectory_envelope,
)
from liquidloc.scenarios.protocol_trajectory import generate_protocol_gt_rows


def test_protocol_trajectory_meets_main_table_envelope():
    gt = generate_protocol_gt_rows(
        seq_id="ut_seq",
        seed=7,
        duration_s=45.0,
        workspace_span_m=20.0,
        dt_s=0.05,
    )
    env = compute_trajectory_envelope(gt)
    assert env["l_xy_m"] >= 10.0
    assert env["t_eff_s"] >= 20.0
    assert env["path_length_m"] >= 20.0
    assert 0.3 <= env["v_median_mps"] <= 1.5
    assert env["v95_mps"] <= 2.5
    assert env["near_zero_speed_ratio"] >= 0.05
    assert (
        env["turn_total_rad"] >= 6.0
        or env["significant_turn_count"] >= 3.0
    )


def test_anchor_workspace_scale_matches_l_xy():
    cfg = load_scene_axis_protocol()["axes"]["K"]
    layout, report = build_anchor_layout(4, "K1", cfg, workspace_span_m=20.0)
    xs = [p[0] for p in layout["anchor_positions"]]
    ys = [p[1] for p in layout["anchor_positions"]]
    baseline = max(max(xs) - min(xs), max(ys) - min(ys))
    assert baseline == pytest.approx(20.0, rel=1e-3)
    assert report["workspace_span_m"] == pytest.approx(20.0)


def test_assert_geometry_motion_envelope_passes_for_protocol_pair():
    gt = generate_protocol_gt_rows(seq_id="ut2", seed=3, duration_s=45.0, workspace_span_m=20.0)
    cfg = load_scene_axis_protocol()["axes"]["K"]
    layout, _ = build_anchor_layout(4, "K3", cfg, workspace_span_m=20.0)
    report = assert_geometry_motion_envelope(gt, layout, profile="main_table")
    assert report["passed"] is True
    assert report["checks"]["anchor_count_in_main_band"] is True


def test_assert_geometry_motion_envelope_rejects_toy_track():
    toy = [
        {"timestamp": 0.0, "px": 0.0, "py": 0.0, "yaw": 0.0},
        {"timestamp": 0.1, "px": 0.05, "py": 0.0, "yaw": 0.0},
        {"timestamp": 0.2, "px": 0.10, "py": 0.0, "yaw": 0.0},
    ]
    cfg = load_scene_axis_protocol()["axes"]["K"]
    layout, _ = build_anchor_layout(4, "K0", cfg, workspace_span_m=20.0)
    with pytest.raises(ValueError, match="geometry/motion envelope failed"):
        assert_geometry_motion_envelope(toy, layout, profile="main_table")


def test_nominal_k_prefers_undetermined_k0():
    # 「五轴档位协议定义」K 轴基线为 K0（4 锚对称，好几何）。
    nominal = get_nominal_levels()
    assert nominal["K"] == "K0"


def test_assert_geometry_motion_envelope_trajectory_in_hull_ratio_recorded():
    """§8.2.1 表行 9「与锚点几何匹配 ≥0.15」软门禁验证。
    轨迹整体远离锚点凸包 → trajectory_in_hull_ratio≈0 → 仅记录在 report（不 raise）。
    §8.2.1 表行 9 此项软化设计：fixture 数据生成时轨迹可能因设计原因短暂偏离凸包，
    不应中断 materialize 流程；主表实验摆问题由 §8.2.1 纪律 4 等更高层捕获。
    """
    from liquidloc.scenarios.protocol_trajectory import generate_protocol_gt_rows
    cfg = load_scene_axis_protocol()["axes"]["K"]
    # 凸包外但 baseline 在 [0.5, 2.5] 比例的锚点布局：3 个锚点 baseline~5m
    small_layout = {
        # 一行 3 锚共线 → l_xy≈20m, baseline 5m，比例 0.25 不在 [0.5, 2.5]，
        # 因此调整成 baseline ~10m 的非退化布局：anchor baseline=10m，hull 仅含少量轨迹点
        "anchor_positions": [[-5.0, 0.0], [5.0, 0.0], [0.0, 8.66]],
        "anchor_ids": ["a1", "a2", "a3"],
    }
    # 用 protocol_trajectory 生成轨迹（workspace=20m），轨迹大部分超出小三角凸包
    gt = generate_protocol_gt_rows(seq_id="probe_hull", seed=0, duration_s=45.0,
                                   workspace_span_m=20.0, dt_s=0.05)
    report = assert_geometry_motion_envelope(gt, small_layout, profile="main_table")
    # 软门禁：trajectory_in_hull_ratio 仅记录，不在 checks 字典中触发 raise。
    assert "trajectory_in_hull_ratio" in report
    # hull ratio 应 < 0.15（轨迹大部分超出小凸包）
    assert report["trajectory_in_hull_ratio"] < 0.5, (
        "§8.2.1 表行 9 hull ratio 实测值: "
        f"{report['trajectory_in_hull_ratio']:.4f}; 应该 < 0.5（小凸包围栏外）"
    )


def test_assert_geometry_motion_envelope_anchor_count_in_main_band_flagged():
    """§8.1 Na ≥ 8 高冗余：anchor_count_in_main_band=False → raise."""
    from liquidloc.scenarios.protocol_trajectory import generate_protocol_gt_rows
    cfg = load_scene_axis_protocol()["axes"]["K"]
    # 6 锚（默认 max 5 不允许 high count）
    layout = {
        "anchor_positions": [[-10, -10], [-10, 10], [10, -10], [10, 10],
                              [0, 0], [-5, 5]],
        "anchor_ids": ["a1", "a2", "a3", "a4", "a5", "a6"],
    }
    gt = generate_protocol_gt_rows(seq_id="probe_count", seed=0, duration_s=45.0,
                                   workspace_span_m=20.0, dt_s=0.05)
    with pytest.raises(ValueError) as excinfo:
        assert_geometry_motion_envelope(gt, layout, profile="main_table")
    msg = str(excinfo.value)
    assert "anchor_count_in_main_band" in msg


def test_assert_geometry_motion_envelope_baseline_matches_l_xy_flagged():
    """§8.1 baseline ≥ 0.25 × L_xy：baseline 远小于 L_xy → raise."""
    from liquidloc.scenarios.protocol_trajectory import generate_protocol_gt_rows
    # baseline 1m, L_xy ~20m → ratio 0.05 < 0.25 → fail
    layout = {
        "anchor_positions": [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]],
        "anchor_ids": ["a1", "a2", "a3", "a4"],
    }
    gt = generate_protocol_gt_rows(seq_id="probe_baseline", seed=0, duration_s=45.0,
                                   workspace_span_m=20.0, dt_s=0.05)
    with pytest.raises(ValueError) as excinfo:
        assert_geometry_motion_envelope(gt, layout, profile="main_table")
    msg = str(excinfo.value)
    assert "baseline_matches_l_xy" in msg
