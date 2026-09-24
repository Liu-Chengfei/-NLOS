from __future__ import annotations

"""协方差适配器（covariance_adapter）测试模块。

文件职责：验证 build_effective_cov 能根据缩放因子
正确调整协方差矩阵。

测试覆盖范围：
- 正常场景：字典和标量协方差缩放
- 边界场景：缩放因子小于 1
- 异常场景：零缩放因子拒绝
- 序列形状区分错误
- 输入类型保留
- 布尔输入拒绝
- 非有限 base_cov 拒绝

被测模块：liquidloc.fusion.covariance_adapter"""


import numpy as np
import pytest

from liquidloc.fusion.covariance_adapter import build_effective_cov


def test_normal_case():
    base_cov = {"uwb": 2.0, "vio": {"pos": [1.0, 2.0], "yaw": 0.5}, "other": 3.0}

    effective_cov, cov_report = build_effective_cov(
        base_cov,
        uwb_scaling=1.5,
        vio_scaling=2.0,
    )

    # risk 已由 _compose_noise_multiplier 处理，本模块只消费 scaling。
    assert effective_cov["uwb"] == pytest.approx(3.0)
    assert effective_cov["vio"]["pos"] == [pytest.approx(2.0), pytest.approx(4.0)]
    assert effective_cov["vio"]["yaw"] == pytest.approx(1.0)
    assert effective_cov["other"] == pytest.approx(3.0)
    assert cov_report["uwb_scaling"] == pytest.approx(1.5)
    assert cov_report["vio_scaling"] == pytest.approx(2.0)
    assert base_cov["uwb"] == 2.0
    assert base_cov["vio"]["pos"] == [1.0, 2.0]


def test_named_scaling_applies_per_channel():
    # 仅 scaling 有效，risk 由上游 _compose_noise_multiplier 处理。
    scaled_cov, _ = build_effective_cov(2.0, uwb_scaling=1.5)
    combined_cov, _ = build_effective_cov(2.0, uwb_scaling=1.5, vio_scaling=1.5)

    assert scaled_cov == pytest.approx(3.0)
    assert combined_cov == pytest.approx(3.0)  # 标量时 uwb_scaling 和 vio_scaling 相同等价


def test_boundary_case():
    effective_cov, cov_report = build_effective_cov(3.0, uwb_scaling=0.2, vio_scaling=0.2)

    # 协方差仅由 scaling 决定：3.0 * 0.2 = 0.6
    assert effective_cov == pytest.approx(0.6)
    assert cov_report["effective_cov"] == pytest.approx(0.6)
    assert cov_report["uwb_scaling"] == pytest.approx(0.2)
    assert cov_report["vio_scaling"] == pytest.approx(0.2)


def test_invalid_scaling_rejected():
    with pytest.raises(ValueError, match="finite and > 0"):
        build_effective_cov(3.0, uwb_scaling=0.0)


def test_sequence_shape_distinction_error():
    with pytest.raises(ValueError, match="sequence base_cov"):
        build_effective_cov([1.0, 2.0], uwb_scaling=1.2, vio_scaling=1.4)


def test_invalid_case():
    # risk 参数已由上游 _compose_noise_multiplier 处理，本模块不再接受。
    with pytest.raises(TypeError, match="risk"):
        build_effective_cov(3.0, risk="bad")


@pytest.mark.parametrize(
    "base_cov, kwargs, expected_type",
    [
        ((1.0, 2.0), {"uwb_scaling": 1.5}, tuple),
        (np.array([1.0, 2.0]), {"vio_scaling": 2.0}, np.ndarray),
    ],
)
def test_report_preserves_input_type_and_copy(base_cov, kwargs, expected_type):
    effective_cov, cov_report = build_effective_cov(base_cov, **kwargs)

    assert isinstance(effective_cov, expected_type)
    assert isinstance(cov_report["base_cov"], expected_type)
    assert isinstance(cov_report["effective_cov"], expected_type)
    if isinstance(base_cov, np.ndarray):
        np.testing.assert_allclose(base_cov, np.array([1.0, 2.0]))
    else:
        assert base_cov == (1.0, 2.0)


@pytest.mark.parametrize("base_cov", [True, [1.0, True]])
def test_boolean_input_rejected(base_cov):
    with pytest.raises(TypeError, match="numeric"):
        build_effective_cov(base_cov)


def test_nonfinite_base_cov_rejected():
    with pytest.raises(ValueError, match=r"base_cov(\.uwb)? must contain finite values"):
        build_effective_cov({"uwb": float("inf")})


def test_vio_cov_mapping_pos_yaw_restores_uwb_keys():
    # 顶层含 pos+yaw 时走 VIO 协方差结构识别分支：
    # 整体按 vio_scaling 缩放，UWB 键恢复为仅 uwb_scaling 缩放后的值，避免二次放大。
    base_cov = {"uwb": 2.0, "pos": [1.0, 2.0], "yaw": 0.5, "other": 3.0}

    effective_cov, _ = build_effective_cov(
        base_cov, uwb_scaling=1.5, vio_scaling=2.0
    )

    # uwb 只受 uwb_scaling 影响：2.0 * 1.5 = 3.0，未被 vio_scaling 二次放大
    assert effective_cov["uwb"] == pytest.approx(3.0)
    # VIO 结构整体按 vio_scaling 缩放
    assert effective_cov["pos"] == [pytest.approx(2.0), pytest.approx(4.0)]
    assert effective_cov["yaw"] == pytest.approx(1.0)
    # 非通道键也参与 vio 整体缩放
    assert effective_cov["other"] == pytest.approx(6.0)
    # 原对象未被改写
    assert base_cov["uwb"] == 2.0
    assert base_cov["pos"] == [1.0, 2.0]


def test_vio_cov_mapping_dxy_dyaw_restores_uwb_keys():
    # 顶层含 dxy+dyaw（VIO_MEASUREMENT_ITEMS）时同样走 VIO 整体缩放分支。
    base_cov = {"uwb": 2.0, "dx": 1.0, "dy": 2.0, "dyaw": 0.5, "other": 3.0}

    effective_cov, _ = build_effective_cov(
        base_cov, uwb_scaling=1.5, vio_scaling=2.0
    )

    assert effective_cov["uwb"] == pytest.approx(3.0)
    assert effective_cov["dx"] == pytest.approx(2.0)
    assert effective_cov["dy"] == pytest.approx(4.0)
    assert effective_cov["dyaw"] == pytest.approx(1.0)
    assert effective_cov["other"] == pytest.approx(6.0)


def test_scalar_base_cov_with_all_none_scaling_returns_deepcopy():
    effective_cov, cov_report = build_effective_cov(3.0)

    assert effective_cov == pytest.approx(3.0)
    assert cov_report["uwb_scaling"] is None
    assert cov_report["vio_scaling"] is None
    assert cov_report["effective_cov"] == pytest.approx(3.0)


def test_scalar_base_cov_distinct_scaling_rejected():
    with pytest.raises(ValueError, match="scalar base_cov"):
        build_effective_cov(3.0, uwb_scaling=1.2, vio_scaling=1.4)


def test_list_base_cov_with_equal_scaling():
    effective_cov, _ = build_effective_cov(
        [1.0, 2.0], uwb_scaling=2.0, vio_scaling=2.0
    )

    assert effective_cov == [pytest.approx(2.0), pytest.approx(4.0)]


def test_list_base_cov_with_all_none_scaling_returns_deepcopy():
    effective_cov, cov_report = build_effective_cov([1.0, 2.0])

    assert effective_cov == [1.0, 2.0]
    assert cov_report["uwb_scaling"] is None
    assert cov_report["vio_scaling"] is None


@pytest.mark.parametrize("bad_scaling", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_scaling_rejected(bad_scaling):
    with pytest.raises(ValueError, match="finite and > 0"):
        build_effective_cov(3.0, uwb_scaling=bad_scaling)


@pytest.mark.parametrize(
    "base_cov",
    [np.array([1.0, 2.0, 3.0]), np.eye(3) * 2.0],
)
def test_vio_shaped_array_applies_vio_scaling_only(base_cov):
    # shape (3,) 或 (3,3) 的数组仅消费 vio_scaling，忽略 uwb_scaling。
    effective_cov, _ = build_effective_cov(
        base_cov, uwb_scaling=None, vio_scaling=2.0
    )

    expected = np.asarray(base_cov, dtype=float) * 2.0
    np.testing.assert_allclose(effective_cov, expected)


def test_other_array_shape_distinct_scaling_rejected():
    with pytest.raises(ValueError, match="base_cov shape"):
        build_effective_cov(
            np.array([1.0, 2.0, 3.0, 4.0]), uwb_scaling=1.2, vio_scaling=1.4
        )
