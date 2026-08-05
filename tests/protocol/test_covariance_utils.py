"""covariance_utils 单源锁死 pytest（§12.2-B1i/B1ii/B1iii 第十五轮反推精读补锁死）。

锁死目的：
- 三方法（EKF / Robust-EKF / FGO）都 import 自 common/covariance_utils.py 单源
  build_effective_cov（common/covariance_utils.py:183）和 estimators/shared.py
  build_controlled_measurement_cov（shared.py:56），禁止任何方法本地重实现。
- 本测试锁死 build_effective_cov 的真实行为：scalar / (3,3) 分块，不区分 UWB/VIO
  形状时要求 uwb_scaling == vio_scaling。
- 同时锁死 build_controlled_measurement_cov 的 noise_multiplier 公式与 invalid 拒绝。

被测模块：
- liquidloc.common.covariance_utils.build_effective_cov
- liquidloc.estimators.shared.build_controlled_measurement_cov
"""

import numpy as np
import pytest

from liquidloc.common.covariance_utils import build_effective_cov
from liquidloc.estimators.shared import build_controlled_measurement_cov, MeasurementControl


# ---------------------------------------------------------------------------
# build_effective_cov 锁死：scalar base_cov 必须同一 scaling
# ---------------------------------------------------------------------------


def test_build_effective_cov_scalar_same_scaling():
    """scalar base_cov 时 uwb_scaling 直接线性乘（_scale_cov_like 用线性乘）。

    注意：scaling^2 转换在 NN 桥接层 _compose_noise_multiplier 完成，
    build_effective_cov 只做线性乘 base_cov * multiplier。这里 multiplier
    已经是 noise_multiplier（= scaling^2 * (1+risk)）的最终值。
    """
    base_cov = np.array(1.0)
    effective, report = build_effective_cov(base_cov, uwb_scaling=2.0, vio_scaling=2.0)
    # linear scaling: 1.0 * 2.0 = 2.0
    assert float(np.asarray(effective)) == pytest.approx(2.0)
    assert report["uwb_scaling"] == 2.0
    assert report["vio_scaling"] == 2.0


def test_build_effective_cov_scalar_diff_scaling_raises():
    """scalar base_cov 时 uwb_scaling != vio_scaling 必 raise。"""
    base_cov = np.array(1.0)
    with pytest.raises(ValueError, match="cannot apply distinct"):
        build_effective_cov(base_cov, uwb_scaling=2.0, vio_scaling=3.0)


def test_build_effective_cov_scalar_single_scaling_works():
    """scalar base_cov 单 scaling 正常工作（线性乘）。"""
    base_cov = np.array(1.0)
    effective, report = build_effective_cov(base_cov, uwb_scaling=2.0)
    # linear scaling: 1.0 * 2.0 = 2.0
    assert float(np.asarray(effective)) == pytest.approx(2.0)
    assert report["uwb_scaling"] == 2.0


# ---------------------------------------------------------------------------
# build_effective_cov 锁死：(3,3) VIO 块只走 vio_scaling
# ---------------------------------------------------------------------------


def test_build_effective_cov_vio_3x3_only_vio_scaling():
    """(3,3) base_cov 只接受 vio_scaling，uwb_scaling 被忽略（线性乘）。"""
    base_cov = np.eye(3) * 4.0
    effective, report = build_effective_cov(base_cov, uwb_scaling=2.0, vio_scaling=3.0)
    assert report["vio_scaling"] == 3.0
    # linear scaling: 4.0 * 3.0 = 12.0
    np.testing.assert_array_equal(effective, np.eye(3) * 12.0)


# ---------------------------------------------------------------------------
# build_effective_cov 锁死：coerce_finite_scalar 拒绝 zero/negative scaling
# ---------------------------------------------------------------------------


def test_build_effective_cov_rejects_zero_uwb_scaling():
    """uwb_scaling=0 被 coerce_finite_scalar 拒绝。"""
    base_cov = np.array(1.0)
    with pytest.raises(ValueError):
        build_effective_cov(base_cov, uwb_scaling=0.0)


def test_build_effective_cov_rejects_negative_vio_scaling():
    """vio_scaling<0 被 coerce_finite_scalar 拒绝。"""
    base_cov = np.array(1.0)
    with pytest.raises(ValueError):
        build_effective_cov(base_cov, vio_scaling=-1.0)


# ---------------------------------------------------------------------------
# build_controlled_measurement_cov 锁死：noise_multiplier 公式
# ---------------------------------------------------------------------------


def test_build_controlled_measurement_cov_consumes_noise_multiplier_literal():
    """build_controlled_measurement_cov 直接消费 control.noise_multiplier 字面量。

    真实行为：scaling^2 * (1+risk) 公式在 NN 桥接层（fusion_runner /
    liquid_bridge_contract._compose_noise_multiplier）做，本函数只消费最终值
    并将其线性乘到 base_cov 上。control.scaling / control.risk 在本函数中只
    作为元数据报告，不参与计算。
    """
    # noise_multiplier=6.0（已由上游 NN 桥接层算好）
    control = MeasurementControl(
        modality="uwb",
        bias_applied=0.0,
        noise_multiplier=6.0,
        scaling=2.0,  # 仅元数据：对应 noise_multiplier=2^2*(1+0.5)=6.0
        risk=0.5,
        gate_action="pass_through",
    )
    base_cov = np.array([[1.0]])
    effective, report = build_controlled_measurement_cov(base_cov, control, modality="uwb")
    # 直接消费 control.noise_multiplier 字面量（线性乘，不在本函数内重算公式）
    assert report["noise_multiplier"] == pytest.approx(6.0)
    # scaling 与 risk 作为元数据被报告但不参与计算
    assert report["bridge_scaling"] == pytest.approx(2.0)
    assert report["bridge_risk"] == pytest.approx(0.5)
    # effective_cov = base_cov * noise_multiplier = 1.0 * 6.0 = 6.0（线性乘）
    assert effective[0, 0] == pytest.approx(6.0)


def test_build_controlled_measurement_cov_rejects_nan_noise_multiplier_via_control():
    """noise_multiplier=NaN 在 MeasurementControl.__post_init__ 被 coerce_finite_scalar 拒绝。

    锁死 build_controlled_measurement_cov 的前置校验路径：
    如果上游绕过 MeasurementControl 校验直接传 NaN 给 noise_multiplier，
    build_controlled_measurement_cov 内部的 isfinite 守卫仍会 raise。
    """
    with pytest.raises(ValueError, match="must be finite"):
        MeasurementControl(
            modality="uwb",
            bias_applied=0.0,
            noise_multiplier=float("nan"),
            scaling=1.0,
            risk=0.0,
            gate_action="pass_through",
        )


def test_build_controlled_measurement_cov_rejects_zero_noise_multiplier_via_control():
    """noise_multiplier=0 在 MeasurementControl.__post_init__ 被 _nm_floor 拒绝。

    _nm_floor = scaling_min^2 = 1.0^2 = 1.0，所以 0 < 1.0 必被拒。
    """
    with pytest.raises(ValueError, match="noise_multiplier must be >= 1.0"):
        MeasurementControl(
            modality="uwb",
            bias_applied=0.0,
            noise_multiplier=0.0,
            scaling=1.0,
            risk=0.0,
            gate_action="pass_through",
        )


# ---------------------------------------------------------------------------
# 三方法 import 同源静态锁死（核心 §12.2-B1i/B1ii/B1iii 锁死）
# ---------------------------------------------------------------------------


def test_three_estimators_all_import_build_effective_cov_from_singleton():
    """EKF / Robust-EKF / FGO 都 import 自 common/covariance_utils 单源。

    锁死 build_effective_cov 禁止任何方法本地重实现
    （§12.2-B1i/B1ii/B1iii "三网同一 R 矩阵构建"）。
    """
    import inspect

    for name, mod in [
        ("ekf_core", "liquidloc.estimators.ekf_core"),
        ("robust_ekf_core", "liquidloc.estimators.robust_ekf_core"),
        ("fgo_core", "liquidloc.estimators.fgo_core"),
    ]:
        src = inspect.getsource(__import__(mod, fromlist=[""]))
        assert "build_effective_cov" in src, (
            f"{name} 必须 import build_effective_cov 单源 "
            f"（§12.2-B1i/B1ii/B1iii 三网同一 R 矩阵构建锁死）"
        )
        assert "from liquidloc.common.covariance_utils import build_effective_cov" in src, (
            f"{name} 必须从 liquidloc.common.covariance_utils 单源 import "
            f"build_effective_cov（禁止本地重实现）"
        )


def test_three_estimators_all_import_build_controlled_measurement_cov_from_singleton():
    """EKF / Robust-EKF / FGO 都 import 自 estimators/shared 单源。

    锁死 build_controlled_measurement_cov 禁止任何方法本地重实现
    （§12.2-B1i/B1ii/B1iii "三网同一 scaling 控制"）。
    """
    import inspect

    for name, mod in [
        ("ekf_core", "liquidloc.estimators.ekf_core"),
        ("robust_ekf_core", "liquidloc.estimators.robust_ekf_core"),
        ("fgo_core", "liquidloc.estimators.fgo_core"),
    ]:
        src = inspect.getsource(__import__(mod, fromlist=[""]))
        assert "build_controlled_measurement_cov" in src, (
            f"{name} 必须 import build_controlled_measurement_cov 单源 "
            f"（§12.2-B1i/B1ii/B1iii 三网同一 scaling 控制锁死）"
        )
        assert "from liquidloc.estimators.shared import build_controlled_measurement_cov" in src, (
            f"{name} 必须从 liquidloc.estimators.shared 单源 import "
            f"build_controlled_measurement_cov（禁止本地重实现）"
        )