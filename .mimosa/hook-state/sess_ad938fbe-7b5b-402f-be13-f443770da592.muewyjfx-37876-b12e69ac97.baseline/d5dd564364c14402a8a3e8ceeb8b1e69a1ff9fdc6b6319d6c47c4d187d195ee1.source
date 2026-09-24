
"""轨迹指标（trajectory_metrics）测试模块。

文件职责：验证 compute_trajectory_metrics 能正确计算
RMSE、MAE、ATE、RPE 等轨迹级指标。

测试覆盖范围：
- 正常场景：3 点轨迹的完整指标
- 边界场景：单点轨迹
- 异常场景：坐标维度不一致、重复时间戳、布尔坐标、非有限坐标
- 对齐保留预测顺序以正确计算 RPE

被测模块：liquidloc.metrics.trajectory_metrics"""

import math

import pytest

from liquidloc.metrics.trajectory_metrics import compute_trajectory_metrics


def test_normal_case():
    pred_traj = [
        {"px": 0.0, "py": 0.0},
        {"px": 2.0, "py": 0.0},
        {"px": 4.0, "py": 0.0},
    ]
    gt_traj = [
        {"px": 0.0, "py": 0.0},
        {"px": 0.0, "py": 0.0},
        {"px": 0.0, "py": 0.0},
    ]

    result = compute_trajectory_metrics(pred_traj, gt_traj)

    # 实现额外返回 rmse_raw/mae_raw/align_degraded 作为诊断输出（文档化的辅助键）
    assert set(result).issuperset({"rmse", "mae", "ate", "ate_degraded", "rpe"})
    assert result["rmse"] == pytest.approx(math.sqrt((0.0**2 + 2.0**2 + 4.0**2) / 3.0))
    assert result["mae"] == pytest.approx(2.0)
    # ATE 现在做 SE3 对齐，消除全局坐标系偏移后计算 RMSE。
    # 预测轨迹 (0,0),(0,0),(0,0) 对齐到真值 (0,0),(2,0),(4,0) 后，
    # 最优刚体变换将所有预测点映射到质心 (2,0)，ATE 衡量对齐后的残差。
    assert result["ate"] < result["rmse"]  # SE3 对齐后 ATE 必然 <= RMSE。
    # RPE 现在衡量相邻帧相对运动差异，而非误差波动。
    # pred 的相邻运动为 (0,0), gt 的相邻运动为 (2,0),(2,0)，
    # RPE = (||(0,0)-(2,0)|| + ||(0,0)-(2,0)||) / 2 = 2.0
    assert result["rpe"] == pytest.approx(2.0)


def test_boundary_case():
    pred_traj = [{"t": 10, "px": 1.0, "py": 2.0, "pz": 3.0}]
    gt_traj = [{"t": 10, "px": 1.0, "py": 1.0, "pz": 5.0}]

    result = compute_trajectory_metrics(pred_traj, gt_traj)

    assert result["rmse"] == pytest.approx(math.sqrt(5.0))
    assert result["mae"] == pytest.approx(math.sqrt(5.0))
    # 单点时 ATE 退化为平移对齐后的 RMSE，即 0.0（减去质心偏移后残差为0）。
    assert result["ate"] == pytest.approx(0.0)
    assert result["rpe"] == pytest.approx(0.0)


def test_invalid_case():
    with pytest.raises(ValueError, match="same coordinate fields"):
        compute_trajectory_metrics(
            [{"px": 0.0, "py": 0.0}],
            [{"px": 0.0, "py": 0.0, "pz": 0.0}],
        )


def test_rejects_inconsistent_coordinate_fields_within_trajectory():
    with pytest.raises(ValueError, match="consistent coordinate fields"):
        compute_trajectory_metrics(
            [
                {"px": 0.0, "py": 0.0, "pz": 0.0},
                {"px": 1.0, "py": 1.0},
            ],
            [
                {"px": 0.0, "py": 0.0, "pz": 0.0},
                {"px": 1.0, "py": 1.0, "pz": 1.0},
            ],
        )


def test_rejects_duplicate_alignment_keys():
    with pytest.raises(ValueError, match="duplicate 't' values"):
        compute_trajectory_metrics(
            [
                {"t": 1, "px": 0.0, "py": 0.0},
                {"t": 1, "px": 1.0, "py": 1.0},
            ],
            [
                {"t": 1, "px": 0.0, "py": 0.0},
                {"t": 2, "px": 1.0, "py": 1.0},
            ],
        )


def test_alignment_preserves_prediction_order_for_rpe_computation():
    pred_traj = [
        {"t": 2, "px": 4.0, "py": 0.0},
        {"t": 1, "px": 1.0, "py": 0.0},
        {"t": 3, "px": 9.0, "py": 0.0},
    ]
    gt_traj = [
        {"t": 1, "px": 0.0, "py": 0.0},
        {"t": 2, "px": 0.0, "py": 0.0},
        {"t": 3, "px": 0.0, "py": 0.0},
    ]

    result = compute_trajectory_metrics(pred_traj, gt_traj)

    assert result["rpe"] == pytest.approx(5.5)


def test_rejects_bool_coordinates():
    with pytest.raises(ValueError, match="must be numeric"):
        compute_trajectory_metrics(
            [{"px": True, "py": 0.0}],
            [{"px": 0.0, "py": 0.0}],
        )


def test_rejects_non_finite_coordinates():
    with pytest.raises(ValueError, match="must be finite"):
        compute_trajectory_metrics(
            [{"px": float("nan"), "py": 0.0}],
            [{"px": 0.0, "py": 0.0}],
        )
