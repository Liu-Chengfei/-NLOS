"""轨迹绘图模块测试。

本模块验证 build_trajectory_figure_spec 函数的正确性，该函数
根据预测轨迹和真值轨迹构建图表规格说明。

测试覆盖范围：
  - 正常情况：2D 轨迹的图表规格构建（figure_path、dimension、coord_keys、sequence_specs）
  - 异常情况：缺少 figure_path 时抛出 KeyError
  - 异常情况：布尔类型坐标值被拒绝
  - 异常情况：部分点包含 pz 而部分不包含时被拒绝

被测模块：
  - liquidloc.plotting.plot_trajectories
"""

from __future__ import annotations

import pytest

from liquidloc.plotting.plot_trajectories import build_trajectory_figure_spec


def test_normal_case(tmp_path):
    """验证 2D 轨迹的图表规格正确构建。

    测试场景：传入预测轨迹和真值轨迹，各包含 2 个状态点，
    指定 figure_path。
    预期行为：figure_spec 包含正确的 figure_path、dimension=2、
    coord_keys=("px", "py")，sequence_specs 中包含 seq_label。
    """
    figure_spec = build_trajectory_figure_spec(
        {
            "prediction_bundle": {
                "seq_id": "seq_a",
                "states": [
                    {"px": 0, "py": 0},
                    {"px": 1, "py": 1},
                ],
            },
            "gt_bundle": {
                "seq_id": "seq_a",
                "states": [
                    {"px": 0, "py": 0},
                    {"px": 1, "py": 0.5},
                ],
            },
        },
        {"figure_path": tmp_path / "trajectory.png"},
    )

    assert figure_spec["figure_path"] == tmp_path / "trajectory.png"
    # 2D 轨迹，维度为 2
    assert figure_spec["dimension"] == 2
    # 坐标键为 px 和 py
    assert figure_spec["coord_keys"] == ("px", "py")
    # 序列规格中应包含 seq_label
    assert figure_spec["sequence_specs"][0]["seq_label"] == "seq_a"


def test_invalid_case():
    """验证缺少 figure_path 时抛出 ValueError。

    测试场景：传入空的绘图选项字典（缺少 figure_path）。
    预期行为：抛出 ValueError，提示缺少 figure_path（按项目硬约束，值合同违规用 ValueError 而非 KeyError）。
    """
    with pytest.raises(ValueError, match="figure_path"):
        build_trajectory_figure_spec(
            {
                "prediction_bundle": {
                    "seq_id": "seq_a",
                    "states": [{"px": 0, "py": 0}],
                },
                "gt_bundle": {
                    "seq_id": "seq_a",
                    "states": [{"px": 0, "py": 0}],
                },
            },
            {},
        )


def test_rejects_bool_coordinate(tmp_path):
    """验证布尔类型坐标值被拒绝。

    测试场景：预测轨迹的第一个状态点的 px 为 True（布尔类型）。
    预期行为：抛出 ValueError，提示坐标值必须为数值类型。
    """
    with pytest.raises(ValueError, match=r"prediction_bundle\[0\]\.states\[0\]\.px must be numeric\."):
        build_trajectory_figure_spec(
            {
                "prediction_bundle": {
                    "seq_id": "seq_a",
                    "states": [{"px": True, "py": 0}],
                },
                "gt_bundle": {
                    "seq_id": "seq_a",
                    "states": [{"px": 0, "py": 0}],
                },
            },
            {"figure_path": tmp_path / "trajectory.png"},
        )


def test_rejects_partial_pz_presence(tmp_path):
    """验证部分点包含 pz 而部分不包含时被拒绝。

    测试场景：预测轨迹的第一个状态点包含 pz，第二个不包含。
    预期行为：抛出 ValueError，提示所有点必须要么都包含 pz，
    要么都不包含。
    """
    with pytest.raises(ValueError, match=r"prediction_bundle\[0\] points must either all include 'pz' or all omit it\."):
        build_trajectory_figure_spec(
            {
                "prediction_bundle": {
                    "seq_id": "seq_a",
                    "states": [
                        {"px": 0, "py": 0, "pz": 0},
                        {"px": 1, "py": 1},
                    ],
                },
                "gt_bundle": {
                    "seq_id": "seq_a",
                    "states": [
                        {"px": 0, "py": 0, "pz": 0},
                        {"px": 1, "py": 1, "pz": 1},
                    ],
                },
            },
            {"figure_path": tmp_path / "trajectory.png"},
        )
