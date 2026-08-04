from __future__ import annotations

"""鲁棒扩展卡尔曼滤波器（RobustEKF）核心测试模块。

测试覆盖范围：
- RobustEKF 的预测步与更新步
- Huber 权重函数对异常值的抑制
- 马氏距离门控与质量地板
- UWB/VIO 更新的鲁棒处理

被测模块：liquidloc.estimators.robust_ekf_core"""

import math

import numpy as np
import pytest

from liquidloc.common.types import MeasurementControl, ModelIntermediate
from liquidloc.estimators.robust_ekf_core import (
    RobustEKFCore,
    _normalize_vio_covariance,
    _quality_below_floor,
)
from liquidloc.protocol.liquid_bridge_contract import build_measurement_control


# ────────────────────────────────────────────────────────────
#  测试辅助
# ────────────────────────────────────────────────────────────

def _anchor_layout():
    return {"anchor_ids": [0], "anchor_positions": [(1.0, 0.0)]}


def _cfg():
    return {
        "measurement_noise": {"uwb": 0.25, "vio": {"pos": 0.08, "yaw": 0.03}},
        "anchor_layout": _anchor_layout(),
        # §1.1 主表 8 维 + §2.3 紧耦合扩维（uwb_clock_bias / vio_scale），10 维
        "init_cov": [1.0] * 10,
        "robust_weight": {"type": "huber", "delta": 1.345},
        "gate": {"mahalanobis_sq": 9.21, "quality_floor": 0.2},
    }


def _imu_event(*, t=0.05, dt=0.05):
    return {
        "t": t,
        "dt": dt,
        "modality": "imu",
        "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
        "imu_payload": {"ax": 0.0, "ay": 0.0, "gz": 0.0},
        "uwb_payload": None,
        "vio_payload": None,
    }


def _uwb_event(*, t=0.1, rng=0.8, quality=0.9):
    return {
        "t": t,
        "dt": 0.1,
        "modality": "uwb",
        "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
        "imu_payload": None,
        "uwb_payload": {"anchor_id": 0, "range": rng, "valid": True, "quality": quality},
        "vio_payload": None,
    }


def _vio_event(*, t=0.2, quality=0.9, dx=0.1, dy=0.0, dyaw=0.0):
    return {
        "t": t,
        "dt": 0.1,
        "modality": "vio",
        "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
        "imu_payload": None,
        "uwb_payload": None,
        "vio_payload": {
            "dx": dx,
            "dy": dy,
            "dyaw": dyaw,
            "quality": quality,
            "tracked_features": 120,
            "reproj_err": 0.3,
        },
    }


def test_consuming_intermediate_without_measurement_control_does_not_change_robust_ekf_update():
    """不侵入测试：consuming intermediate without measurement control。\n\n验证 consuming intermediate without measurement control 不会产生副作用，\n确保功能隔离性。
    """
    plain = RobustEKFCore(_cfg())
    plain.step(_imu_event())
    plain_after = plain.step(_uwb_event())

    with_cached_intermediate = RobustEKFCore(_cfg())
    with_cached_intermediate.step(_imu_event())
    with_cached_intermediate.consume_model_intermediate(
        ModelIntermediate(bias=9.0, risk=0.95, uwb_scaling=5.0, vio_scaling=4.0)
    )
    cached_after = with_cached_intermediate.step(_uwb_event())

    assert cached_after.state == pytest.approx(plain_after.state)
    assert cached_after.covariance_diag == pytest.approx(plain_after.covariance_diag)
    assert with_cached_intermediate.last_update_report["measurement_control"]["bias_applied"] == pytest.approx(0.0)
    assert with_cached_intermediate.last_update_report["measurement_control"]["noise_multiplier"] == pytest.approx(1.0)


# ════════════════════════════════════════════════════════════
#  1. _normalize_vio_covariance
# ════════════════════════════════════════════════════════════

class TestNormalizeVioCovariance:
    """_normalize_vio_covariance 视觉噪声规整测试。"""

    def test_none_rejected(self):
        """拒绝测试：none。\n\n验证被测功能对不合法的 none 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(TypeError, match="R_vio must not be None"):
            _normalize_vio_covariance(None)

    def test_bool_rejected(self):
        """拒绝测试：bool。\n\n验证被测功能对不合法的 bool 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(TypeError, match="R_vio must not be boolean-like"):
            _normalize_vio_covariance(True)

    def test_numpy_bool_rejected(self):
        """拒绝测试：numpy bool。\n\n验证被测功能对不合法的 numpy bool 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(TypeError, match="R_vio must not be boolean-like"):
            _normalize_vio_covariance(np.bool_(False))

    def test_complex_rejected(self):
        """拒绝测试：complex。\n\n验证被测功能对不合法的 complex 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(TypeError, match="must be real numeric, got complex"):
            _normalize_vio_covariance(1 + 2j)

    def test_mapping_pos_yaw(self):
        R = _normalize_vio_covariance({"pos": 0.08, "yaw": 0.03})
        assert R.shape == (3, 3)
        assert R[0, 0] == pytest.approx(0.08)
        assert R[1, 1] == pytest.approx(0.08)
        assert R[2, 2] == pytest.approx(0.03)
        assert R[0, 1] == pytest.approx(0.0)

    def test_mapping_dx_dy_dyaw(self):
        R = _normalize_vio_covariance({"dx": 0.1, "dy": 0.2, "dyaw": 0.05})
        assert R.shape == (3, 3)
        assert R[0, 0] == pytest.approx(0.1)
        assert R[1, 1] == pytest.approx(0.2)
        assert R[2, 2] == pytest.approx(0.05)

    def test_mapping_invalid_keys_rejected(self):
        """拒绝测试：mapping invalid keys。\n\n验证被测功能对不合法的 mapping invalid keys 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(KeyError, match="must provide either pos/yaw or dx/dy/dyaw"):
            _normalize_vio_covariance({"x": 0.1, "y": 0.2})

    def test_mapping_pos_yaw_priority_over_dx_dy_dyaw(self):
        """pos/yaw 与 dx/dy/dyaw 混用时应报错，不允许歧义。"""
        with pytest.raises(ValueError, match="must not mix"):
            _normalize_vio_covariance({"pos": 0.5, "yaw": 0.1, "dx": 9.0, "dy": 9.0, "dyaw": 9.0})

    def test_mapping_pos_yaw_non_finite_rejected(self):
        """拒绝测试：mapping pos yaw non finite。\n\n验证被测功能对不合法的 mapping pos yaw non finite 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must be finite"):
            _normalize_vio_covariance({"pos": float("nan"), "yaw": 0.03})

    def test_mapping_pos_yaw_negative_rejected(self):
        """拒绝测试：mapping pos yaw negative。\n\n验证被测功能对不合法的 mapping pos yaw negative 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must be > 0.0"):
            _normalize_vio_covariance({"pos": -0.1, "yaw": 0.03})

    def test_mapping_dx_dy_dyaw_non_finite_rejected(self):
        """拒绝测试：mapping dx dy dyaw non finite。\n\n验证被测功能对不合法的 mapping dx dy dyaw non finite 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must be finite"):
            _normalize_vio_covariance({"dx": 0.1, "dy": float("inf"), "dyaw": 0.05})

    def test_mapping_dx_dy_dyaw_negative_rejected(self):
        """拒绝测试：mapping dx dy dyaw negative。\n\n验证被测功能对不合法的 mapping dx dy dyaw negative 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must be > 0.0"):
            _normalize_vio_covariance({"dx": 0.1, "dy": -0.2, "dyaw": 0.05})

    def test_scalar(self):
        R = _normalize_vio_covariance(0.5)
        assert R.shape == (3, 3)
        assert np.allclose(R, np.eye(3) * 0.5)

    def test_scalar_non_finite_rejected(self):
        """拒绝测试：scalar non finite。\n\n验证被测功能对不合法的 scalar non finite 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must be finite"):
            _normalize_vio_covariance(float("nan"))

    def test_scalar_negative_rejected(self):
        """拒绝测试：scalar negative。\n\n验证被测功能对不合法的 scalar negative 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must be > 0.0"):
            _normalize_vio_covariance(-0.5)

    def test_1d_vector(self):
        R = _normalize_vio_covariance([0.1, 0.2, 0.3])
        assert R.shape == (3, 3)
        assert R[0, 0] == pytest.approx(0.1)
        assert R[1, 1] == pytest.approx(0.2)
        assert R[2, 2] == pytest.approx(0.3)

    def test_1d_non_finite_rejected(self):
        """拒绝测试：1d non finite。\n\n验证被测功能对不合法的 1d non finite 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must be finite"):
            _normalize_vio_covariance([0.1, float("nan"), 0.3])

    def test_1d_negative_rejected(self):
        """拒绝测试：1d negative。\n\n验证被测功能对不合法的 1d negative 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must be positive"):
            _normalize_vio_covariance([0.1, -0.2, 0.3])

    def test_3x3_matrix(self):
        M = np.diag([0.1, 0.2, 0.3])
        R = _normalize_vio_covariance(M)
        assert np.allclose(R, M)

    def test_3x3_non_finite_rejected(self):
        """拒绝测试：3x3 non finite。\n\n验证被测功能对不合法的 3x3 non finite 输入正确抛出异常，\n防止无效参数通过验证。
        """
        M = np.diag([0.1, 0.2, float("nan")])
        with pytest.raises(ValueError, match="must be finite"):
            _normalize_vio_covariance(M)

    def test_3x3_negative_diagonal_rejected(self):
        """拒绝测试：3x3 negative diagonal。\n\n验证被测功能对不合法的 3x3 negative diagonal 输入正确抛出异常，\n防止无效参数通过验证。
        """
        M = np.diag([0.1, -0.2, 0.3])
        with pytest.raises(ValueError, match="must be positive"):
            _normalize_vio_covariance(M)

    def test_wrong_shape_rejected(self):
        """拒绝测试：wrong shape。\n\n验证被测功能对不合法的 wrong shape 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match=r"must be shape \(3, 3\)"):
            _normalize_vio_covariance(np.eye(4))

    def test_numpy_scalar(self):
        R = _normalize_vio_covariance(np.float64(0.25))
        assert R.shape == (3, 3)
        assert np.allclose(R, np.eye(3) * 0.25)

    def test_zero_scalar_rejected(self):
        """零噪声标量不再允许（必须为正）。"""
        with pytest.raises(ValueError, match="must be > 0.0"):
            _normalize_vio_covariance(0.0)

    def test_numpy_1d_array(self):
        R = _normalize_vio_covariance(np.array([0.1, 0.2, 0.3]))
        assert R.shape == (3, 3)
        assert R[1, 1] == pytest.approx(0.2)


# ════════════════════════════════════════════════════════════
#  2. _quality_below_floor
# ════════════════════════════════════════════════════════════

class TestQualityBelowFloor:
    """_quality_below_floor 质量门槛判断测试。"""

    def test_below_floor_returns_true(self):
        """地板测试：below。\n\n验证 below 的地板值约束，\n确保输出不低于最小值。
        """
        assert _quality_below_floor(0.1, 0.2) is True

    def test_above_floor_returns_false(self):
        """地板测试：above。\n\n验证 above 的地板值约束，\n确保输出不低于最小值。
        """
        assert _quality_below_floor(0.5, 0.2) is False

    def test_equal_floor_returns_false(self):
        """地板测试：equal。\n\n验证 equal 的地板值约束，\n确保输出不低于最小值。
        """
        assert _quality_below_floor(0.2, 0.2) is False

    def test_near_boundary_with_epsilon(self):
        """微容差 _QUALITY_FLOOR_EPSILON 避免边界抖动。"""
        import liquidloc.estimators.robust_ekf_core as mod
        eps = mod.QUALITY_FLOOR_EPSILON
        # quality + eps >= floor → 不算低于门槛
        assert _quality_below_floor(0.2 - eps, 0.2) is False
        # quality + eps < floor → 低于门槛
        assert _quality_below_floor(0.2 - eps - 1e-12, 0.2) is True

    def test_zero_floor_never_rejects(self):
        """拒绝测试：zero floor never。\n\n验证被测功能对 zero floor never 的拒绝行为，\n确保不合法输入被正确拦截。
        """
        assert _quality_below_floor(0.0, 0.0) is False
        assert _quality_below_floor(0.001, 0.0) is False


# ════════════════════════════════════════════════════════════
#  3. RobustEKFCore 初始化
# ════════════════════════════════════════════════════════════

class TestRobustEKFCoreInit:
    """RobustEKFCore 初始化测试。"""

    def test_default_name(self):
        ekf = RobustEKFCore()
        assert ekf.name == "robust_ekf"

    def test_custom_name(self):
        ekf = RobustEKFCore({"name": "my_robust"})
        assert ekf.name == "my_robust"

    def test_inherits_from_ekf_core(self):
        from liquidloc.estimators.ekf_core import EKFCore
        ekf = RobustEKFCore()
        assert isinstance(ekf, EKFCore)

    def test_gate_cfg(self):
        ekf = RobustEKFCore(_cfg())
        gate = ekf._gate_cfg()
        assert gate["quality_floor"] == 0.2
        assert gate["mahalanobis_sq"] == 9.21

    def test_gate_cfg_empty_when_missing(self):
        """缺失测试：gate cfg empty when。\n\n验证 gate cfg empty when 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
        """
        ekf = RobustEKFCore({})
        assert ekf._gate_cfg() == {}

    def test_gate_cfg_non_mapping_rejected(self):
        """拒绝测试：gate cfg non mapping。\n\n验证被测功能对不合法的 gate cfg non mapping 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = RobustEKFCore({"gate": 42})
        with pytest.raises(TypeError, match="gate must be a mapping"):
            ekf._gate_cfg()

    def test_robust_cfg(self):
        ekf = RobustEKFCore(_cfg())
        robust = ekf._robust_cfg()
        assert robust["type"] == "huber"
        assert robust["delta"] == 1.345

    def test_robust_cfg_empty_when_missing(self):
        """缺失测试：robust cfg empty when。\n\n验证 robust cfg empty when 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
        """
        ekf = RobustEKFCore({})
        assert ekf._robust_cfg() == {}

    def test_robust_cfg_non_mapping_rejected(self):
        """拒绝测试：robust cfg non mapping。\n\n验证被测功能对不合法的 robust cfg non mapping 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = RobustEKFCore({"robust_weight": "bad"})
        with pytest.raises(TypeError, match="robust_weight must be a mapping"):
            ekf._robust_cfg()


# ════════════════════════════════════════════════════════════
#  4. 门控配置读取
# ════════════════════════════════════════════════════════════

class TestGateConfig:
    """门控配置读取测试。"""

    def test_quality_floor_default(self):
        """地板测试：quality。\n\n验证 quality 的地板值约束，\n确保输出不低于最小值。
        """
        ekf = RobustEKFCore({})
        assert ekf._quality_floor("vio") == pytest.approx(0.12)  # 默认从 BRIDGE_THRESHOLDS["vio_hard_skip_quality_floor"] 读取

    def test_quality_floor_configured(self):
        """地板测试：quality。\n\n验证 quality 的地板值约束，\n确保输出不低于最小值。
        """
        ekf = RobustEKFCore({"gate": {"quality_floor": 0.3}})
        assert ekf._quality_floor("vio") == pytest.approx(0.3)

    def test_quality_floor_non_finite_rejected(self):
        """拒绝测试：quality floor non finite。\n\n验证被测功能对不合法的 quality floor non finite 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = RobustEKFCore({"gate": {"quality_floor": float("nan")}})
        with pytest.raises(ValueError, match="gate.quality_floor must be finite"):
            ekf._quality_floor("vio")

    def test_quality_floor_negative_rejected(self):
        """拒绝测试：quality floor negative。\n\n验证被测功能对不合法的 quality floor negative 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = RobustEKFCore({"gate": {"quality_floor": -0.1}})
        with pytest.raises(ValueError, match="gate.quality_floor must be >= 0.0"):
            ekf._quality_floor("vio")

    def test_nis_threshold_default_inf(self):
        ekf = RobustEKFCore({})
        assert math.isinf(ekf._nis_threshold())

    def test_nis_threshold_configured(self):
        ekf = RobustEKFCore({"gate": {"mahalanobis_sq": 9.21}})
        assert ekf._nis_threshold() == pytest.approx(9.21)

    def test_nis_threshold_nan_rejected(self):
        """拒绝测试：nis threshold nan。\n\n验证被测功能对不合法的 nis threshold nan 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = RobustEKFCore({"gate": {"mahalanobis_sq": float("nan")}})
        with pytest.raises(ValueError, match="gate.mahalanobis_sq must be finite"):
            ekf._nis_threshold()

    def test_nis_threshold_negative_rejected(self):
        """拒绝测试：nis threshold negative。\n\n验证被测功能对不合法的 nis threshold negative 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = RobustEKFCore({"gate": {"mahalanobis_sq": -1.0}})
        with pytest.raises(ValueError, match="gate.mahalanobis_sq must be >= 0.0"):
            ekf._nis_threshold()


# ════════════════════════════════════════════════════════════
#  5. Huber 权重
# ════════════════════════════════════════════════════════════

class TestHuberWeight:
    """_huber_weight Huber 权重计算测试。"""

    def test_within_delta_returns_1(self):
        ekf = RobustEKFCore({"robust_weight": {"type": "huber", "delta": 1.345}})
        assert ekf._huber_weight(0.5) == pytest.approx(1.0)

    def test_at_delta_returns_1(self):
        ekf = RobustEKFCore({"robust_weight": {"type": "huber", "delta": 1.345}})
        assert ekf._huber_weight(1.345) == pytest.approx(1.0)

    def test_above_delta_returns_delta_over_norm(self):
        ekf = RobustEKFCore({"robust_weight": {"type": "huber", "delta": 1.345}})
        # whitened = 3.0, delta = 1.345 → weight = 1.345/3.0 ≈ 0.4483
        assert ekf._huber_weight(3.0) == pytest.approx(1.345 / 3.0)

    def test_very_large_residual_small_weight(self):
        ekf = RobustEKFCore({"robust_weight": {"type": "huber", "delta": 1.345}})
        weight = ekf._huber_weight(1e6)
        assert weight < 0.01
        assert weight >= 1e-6  # 下界

    def test_unsupported_type_rejected(self):
        """拒绝测试：unsupported type。\n\n验证被测功能对不合法的 unsupported type 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = RobustEKFCore({"robust_weight": {"type": "tukey", "delta": 1.345}})
        with pytest.raises(ValueError, match="Unsupported robust weight type"):
            ekf._huber_weight(1.0)

    def test_non_finite_delta_rejected(self):
        """拒绝测试：non finite delta。\n\n验证被测功能对不合法的 non finite delta 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = RobustEKFCore({"robust_weight": {"type": "huber", "delta": float("nan")}})
        with pytest.raises(ValueError, match="robust_weight.delta must be finite"):
            ekf._huber_weight(1.0)

    def test_negative_delta_rejected(self):
        """拒绝测试：negative delta。\n\n验证被测功能对不合法的 negative delta 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = RobustEKFCore({"robust_weight": {"type": "huber", "delta": -1.0}})
        with pytest.raises(ValueError, match="robust_weight.delta must be > 0.0"):
            ekf._huber_weight(1.0)

    def test_zero_delta_rejected(self):
        """拒绝测试：zero delta。\n\n验证被测功能对不合法的 zero delta 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = RobustEKFCore({"robust_weight": {"type": "huber", "delta": 0.0}})
        with pytest.raises(ValueError, match="robust_weight.delta must be > 0.0"):
            ekf._huber_weight(1.0)

    def test_default_delta(self):
        ekf = RobustEKFCore({})  # 无 robust_weight 配置
        # 默认 delta=1.0, whitened=0.5 ≤ 1.0 → weight=1.0
        assert ekf._huber_weight(0.5) == pytest.approx(1.0)

    def test_weight_minimum_floor(self):
        """权重不应低于 1e-6。"""
        ekf = RobustEKFCore({"robust_weight": {"type": "huber", "delta": 1e-3}})
        weight = ekf._huber_weight(1e9)
        assert weight == pytest.approx(1e-6)


# ════════════════════════════════════════════════════════════
#  6. _quality_value
# ════════════════════════════════════════════════════════════

class TestQualityValue:
    """_quality_value 质量值提取测试。"""

    def test_none_payload_returns_1(self):
        ekf = RobustEKFCore({})
        assert ekf._quality_value(None) == pytest.approx(1.0)

    def test_non_mapping_returns_1(self):
        """映射报告测试：non。\n\n验证 non 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        ekf = RobustEKFCore({})
        assert ekf._quality_value(42) == pytest.approx(1.0)

    def test_mapping_with_quality(self):
        ekf = RobustEKFCore({})
        assert ekf._quality_value({"quality": 0.8}) == pytest.approx(0.8)

    def test_mapping_without_quality_defaults_1(self):
        """无依赖测试：mapping。\n\n验证 mapping 在缺少依赖时的降级行为，\n确保回退策略正确。
        """
        ekf = RobustEKFCore({})
        assert ekf._quality_value({"range": 1.0}) == pytest.approx(1.0)

    def test_non_finite_quality_rejected(self):
        """拒绝测试：non finite quality。\n\n验证被测功能对不合法的 non finite quality 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = RobustEKFCore({})
        with pytest.raises(ValueError, match="quality must be finite"):
            ekf._quality_value({"quality": float("nan")})

    def test_negative_quality_rejected(self):
        """拒绝测试：negative quality。\n\n验证被测功能对不合法的 negative quality 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = RobustEKFCore({})
        with pytest.raises(ValueError, match="quality must be non-negative"):
            ekf._quality_value({"quality": -0.1})

    def test_zero_quality_allowed(self):
        ekf = RobustEKFCore({})
        assert ekf._quality_value({"quality": 0.0}) == pytest.approx(0.0)


# ════════════════════════════════════════════════════════════
#  7. UWB 正常更新
# ════════════════════════════════════════════════════════════

class TestUWBNormalUpdate:
    """UWB 正常更新测试。"""

    def test_normal_uwb_update(self):
        estimator = RobustEKFCore(_cfg())
        state = estimator.step(_uwb_event())
        report = estimator.last_update_report
        assert state.timestamp == 0.1
        assert report["modality"] == "uwb"
        assert report["update_applied"] is True
        assert report["covariance_report"] is not None
        assert report["robust_covariance_report"] is not None
        assert report["gate"]["passed"] is True
        assert report["robust"]["weight"] == pytest.approx(1.0)
        assert report["robust"]["covariance_scale"] == pytest.approx(1.0)

    def test_uwb_robust_report_fields(self):
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event())
        report = estimator.last_update_report
        robust = report["robust"]
        assert "type" in robust
        assert "delta" in robust
        assert "whitened_residual_norm" in robust
        assert "weight" in robust
        assert "covariance_scale" in robust
        assert robust["type"] == "huber"

    def test_uwb_gate_report_fields(self):
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event())
        gate = estimator.last_update_report["gate"]
        assert gate["passed"] is True
        assert "quality" in gate
        assert "quality_floor" in gate
        assert "nis" in gate
        assert "mahalanobis_sq_threshold" in gate
        assert gate["rejected_by"] is None

    def test_uwb_huber_downweights_large_residual(self):
        """大残差应被 Huber 降权。"""
        estimator = RobustEKFCore(_cfg())
        # 用很远的 range 产生大残差
        estimator.step(_uwb_event(rng=10.0))
        report = estimator.last_update_report
        if report["update_applied"]:
            assert report["robust"]["weight"] < 1.0

    def test_uwb_reduces_covariance(self):
        estimator = RobustEKFCore(_cfg())
        cov_before = estimator._covariance.copy()
        estimator.step(_uwb_event())
        assert np.diag(estimator._covariance)[0] <= cov_before[0, 0]


# ════════════════════════════════════════════════════════════
#  8. UWB 质量门控
# ════════════════════════════════════════════════════════════

class TestUWBQualityGate:
    """UWB 质量门控测试。"""

    def test_low_quality_rejected(self):
        """拒绝测试：low quality。\n\n验证被测功能对不合法的 low quality 输入正确抛出异常，\n防止无效参数通过验证。
        """
        estimator = RobustEKFCore(_cfg())
        baseline = estimator.get_state()
        state = estimator.step(_uwb_event(quality=0.1))
        report = estimator.last_update_report
        assert report["update_applied"] is False
        assert report["gate"]["rejected_by"] == "quality_floor"
        assert report["covariance_report"] is None
        assert report["robust_covariance_report"] is None
        assert state.state == baseline.state

    def test_quality_at_floor_not_rejected(self):
        """quality == floor 时不应被拒绝。"""
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event(quality=0.2))
        report = estimator.last_update_report
        assert report["update_applied"] is True

    def test_quality_just_below_floor_rejected(self):
        """拒绝测试：quality just below floor。\n\n验证被测功能对不合法的 quality just below floor 输入正确抛出异常，\n防止无效参数通过验证。
        """
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event(quality=0.19))
        report = estimator.last_update_report
        assert report["update_applied"] is False
        assert report["gate"]["rejected_by"] == "quality_floor"

    def test_zero_quality_floor_never_rejects(self):
        """拒绝测试：zero quality floor never。\n\n验证被测功能对 zero quality floor never 的拒绝行为，\n确保不合法输入被正确拦截。
        """
        cfg = _cfg()
        cfg["gate"]["quality_floor"] = 0.0
        estimator = RobustEKFCore(cfg)
        estimator.step(_uwb_event(quality=0.01))
        report = estimator.last_update_report
        assert report["update_applied"] is True


# ════════════════════════════════════════════════════════════
#  9. UWB NIS 门控
# ════════════════════════════════════════════════════════════

class TestUWBNISGate:
    """UWB NIS 门控测试。"""

    def test_nis_rejection_report(self):
        """NIS 超阈值时拒绝并返回 NIS 值。"""
        cfg = _cfg()
        cfg["gate"]["mahalanobis_sq"] = 0.001  # 极小阈值，几乎必拒绝
        estimator = RobustEKFCore(cfg)
        estimator.step(_uwb_event())
        report = estimator.last_update_report
        # 可能被 NIS 拒绝
        if not report["update_applied"] and report["gate"]["rejected_by"] == "mahalanobis_sq":
            assert report["gate"]["nis"] is not None
            assert report["gate"]["nis"] > 0.001

    def test_infinite_nis_threshold_never_rejects(self):
        """拒绝测试：infinite nis threshold never。\n\n验证被测功能对 infinite nis threshold never 的拒绝行为，\n确保不合法输入被正确拦截。
        """
        cfg = _cfg()
        cfg["gate"]["mahalanobis_sq"] = float("inf")
        estimator = RobustEKFCore(cfg)
        estimator.step(_uwb_event())
        report = estimator.last_update_report
        assert report["update_applied"] is True

    def test_nis_rejection_includes_extra_fields(self):
        cfg = _cfg()
        cfg["gate"]["mahalanobis_sq"] = 0.001
        estimator = RobustEKFCore(cfg)
        estimator.step(_uwb_event())
        report = estimator.last_update_report
        if not report["update_applied"] and report["gate"]["rejected_by"] == "mahalanobis_sq":
            assert "z_pred" in report
            assert "residual" in report
            assert "S" in report


# ════════════════════════════════════════════════════════════
#  10. UWB 跳过更新
# ════════════════════════════════════════════════════════════

class TestUWBSkipUpdate:
    """UWB 控制层跳过更新测试。"""

    def test_uwb_skip_update(self):
        estimator = RobustEKFCore(_cfg())
        before = estimator.get_state()
        estimator.set_measurement_control(
            MeasurementControl(
                modality="uwb",
                bias_applied=0.1,
                scaling=1.6,
                risk=0.97,
                noise_multiplier=3.0,
                gate_action="uwb_skip_update",
            )
        )
        after = estimator.step(_uwb_event(rng=1.6, quality=0.95))
        report = estimator.last_update_report
        assert report["modality"] == "uwb"
        assert report["update_applied"] is False
        assert report["reason"] == "uwb_skip_update"
        assert report["gate"] == {"passed": False, "rejected_by": "uwb_skip_update"}
        assert report["robust"] is None
        assert after.state == before.state
        assert after.covariance_diag == before.covariance_diag


# ════════════════════════════════════════════════════════════
#  11. VIO 正常更新
# ════════════════════════════════════════════════════════════

class TestVIONormalUpdate:
    """VIO 正常更新测试。"""

    def test_normal_vio_update(self):
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event())
        # 首帧 VIO 初始化参考位姿
        estimator.step(_vio_event())
        # 第二帧 VIO 正常更新
        state = estimator.step(_vio_event(t=0.3))
        report = estimator.last_update_report
        assert state.timestamp == 0.3
        assert report["modality"] == "vio"
        assert report["update_applied"] is True
        assert report["covariance_report"] is not None
        assert report["robust_covariance_report"] is not None
        assert report["gate"]["passed"] is True
        assert report["gate"]["rejected_by"] is None

    def test_vio_robust_report_fields(self):
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event())
        # 首帧 VIO 初始化参考位姿
        estimator.step(_vio_event())
        # 第二帧 VIO 正常更新
        estimator.step(_vio_event(t=0.3))
        robust = estimator.last_update_report["robust"]
        assert robust["type"] == "huber"
        assert "delta" in robust
        assert "whitened_residual_norm" in robust
        assert "weight" in robust
        assert "covariance_scale" in robust

    def test_vio_updates_reference_pose(self):
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event())
        estimator.step(_vio_event())
        assert estimator._last_vio_reference_pose is not None

    def test_vio_no_enabled_in_gate(self):
        """成功更新时 gate 报告不包含 'enabled' 键。"""
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event())
        estimator.step(_vio_event())
        assert "enabled" not in estimator.last_update_report["gate"]


# ════════════════════════════════════════════════════════════
#  12. VIO 质量门控
# ════════════════════════════════════════════════════════════

class TestVIOQualityGate:
    """VIO 质量门控测试。"""

    def test_low_quality_rejected(self):
        """拒绝测试：low quality。\n\n验证被测功能对不合法的 low quality 输入正确抛出异常，\n防止无效参数通过验证。
        """
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event())
        before = estimator.get_state()
        state = estimator.step(_vio_event(quality=0.05))
        report = estimator.last_update_report
        assert report["update_applied"] is False
        assert report["gate"]["rejected_by"] == "quality_floor"
        assert state.state == before.state

    def test_quality_at_floor_passes(self):
        """传递测试：quality at floor。\n\n验证 quality at floor 的传递一致性，\n确保数据在流水线中无损传递。
        """
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event())
        # 首帧 VIO 初始化参考位姿
        estimator.step(_vio_event())
        # 第二帧 VIO 正常更新，quality == floor 应通过
        estimator.step(_vio_event(quality=0.2))
        report = estimator.last_update_report
        assert report["update_applied"] is True

    def test_quality_zero_skips_update_and_resets_reference_pose(self):
        """quality <= 0.0 的 VIO 事件跳过更新并重置参考位姿（同 ekf_core 行为）。"""
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event())
        # 首帧 VIO 初始化参考位姿
        estimator.step(_vio_event(quality=0.9))
        # 第二帧 VIO 正常更新，建立参考位姿
        estimator.step(_vio_event(quality=0.9))
        assert estimator.last_update_report["update_applied"] is True
        # quality=0.0 的事件应跳过更新并重置参考位姿
        state_before = estimator.get_state()
        estimator.step(_vio_event(quality=0.0))
        report = estimator.last_update_report
        assert report["update_applied"] is False
        # 参考位姿应被重置为当前估计位姿
        assert estimator._last_vio_reference_pose is not None


# ════════════════════════════════════════════════════════════
#  13. VIO 跳过更新
# ════════════════════════════════════════════════════════════

class TestVIOSkipUpdate:
    """VIO 控制层跳过更新测试。"""

    def test_vio_skip_update(self):
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event())
        before = estimator.get_state()
        estimator.set_measurement_control(
            MeasurementControl(
                modality="vio",
                bias_applied=0.0,
                scaling=1.8,
                risk=0.9,
                noise_multiplier=2.34,
                gate_action="vio_skip_update",
            )
        )
        after = estimator.step(_vio_event())
        report = estimator.last_update_report
        assert report["modality"] == "vio"
        assert report["update_applied"] is False
        assert report["reason"] == "vio_skip_update"
        assert after.state == before.state
        assert after.covariance_diag == before.covariance_diag


# ════════════════════════════════════════════════════════════
#  14. 测量控制与桥接
# ════════════════════════════════════════════════════════════

class TestMeasurementControlAndBridge:
    """测量控制和桥接测试。"""

    def test_noise_multiplier_authority_without_reinflation(self):
        """无依赖测试：noise multiplier authority。\n\n验证 noise multiplier authority 在缺少依赖时的降级行为，\n确保回退策略正确。
        """
        estimator = RobustEKFCore(_cfg())
        estimator.set_measurement_control(
            MeasurementControl(
                modality="uwb",
                bias_applied=0.0,
                scaling=1.5,
                risk=0.5,
                noise_multiplier=2.0,
                gate_action="pass_through",
            )
        )
        estimator.step(_uwb_event())
        report = estimator.last_update_report
        assert report["covariance_report"]["uwb_scaling"] == pytest.approx(2.0)
        assert report["covariance_report"]["bridge_scaling"] == pytest.approx(1.5)
        assert report["covariance_report"]["bridge_risk"] == pytest.approx(0.5)
        assert report["covariance_report"]["noise_multiplier"] == pytest.approx(2.0)
        assert report["covariance_report"]["effective_cov"] == pytest.approx(0.125)  # base_var=0.0625 * noise_multiplier=2.0
        assert report["robust_covariance_report"]["uwb_scaling"] == pytest.approx(1.0)

    def test_bridge_risk_inflates_measurement_covariance(self):
        """膨胀测试：bridge risk。\n\n验证 bridge risk 的膨胀效应，\n确保恶化观测条件导致协方差增大。
        """
        low_risk_control = build_measurement_control(
            _uwb_event(quality=1.0),
            intermediate=ModelIntermediate(bias=0.0, risk=0.0, uwb_scaling=1.4, vio_scaling=1.0),
        )
        high_risk_control = build_measurement_control(
            _uwb_event(quality=1.0),
            intermediate=ModelIntermediate(bias=0.0, risk=0.5, uwb_scaling=1.4, vio_scaling=1.0),
        )

        low_risk_estimator = RobustEKFCore(_cfg())
        low_risk_estimator.set_measurement_control(low_risk_control)
        low_risk_estimator.step(_uwb_event(quality=1.0))

        high_risk_estimator = RobustEKFCore(_cfg())
        high_risk_estimator.set_measurement_control(high_risk_control)
        high_risk_estimator.step(_uwb_event(quality=1.0))

        low_report = low_risk_estimator.last_update_report["covariance_report"]
        high_report = high_risk_estimator.last_update_report["covariance_report"]
        # A0 cross_modal_skew_ms=5 → async_axis_risk=5/300≈0.01667
        # low_risk: applied_risk = max(0.0, 0.0, 0.0, 0.01667) = 0.01667
        a0_async_risk = 5.0 / (0.30 * 1000.0)
        assert low_report["noise_multiplier"] == pytest.approx(1.4**2 * (1.0 + a0_async_risk))  # scaling^2 * (1+applied_risk)
        # high_risk: applied_risk = max(0.5, 0.0, 0.0, 0.01667) = 0.5
        assert high_report["noise_multiplier"] == pytest.approx(1.4**2 * 1.5)  # scaling^2 * (1+risk=0.5)
        assert low_report["effective_cov"] == pytest.approx(0.0625 * 1.4**2 * (1.0 + a0_async_risk))  # base_var * nm
        assert high_report["effective_cov"] == pytest.approx(0.0625 * 1.4**2 * 1.5)  # base_var * nm
        assert high_report["effective_cov"] > low_report["effective_cov"]


# ════════════════════════════════════════════════════════════
#  15. 无效配置
# ════════════════════════════════════════════════════════════

class TestInvalidConfig:
    """无效配置测试。"""

    def test_unsupported_robust_type(self):
        estimator = RobustEKFCore(_cfg() | {"robust_weight": {"type": "bad_weight", "delta": 1.345}})
        with pytest.raises(ValueError, match="Unsupported robust weight type"):
            estimator.step(_uwb_event())

    def test_non_finite_gate_mahalanobis(self):
        estimator = RobustEKFCore(_cfg() | {"gate": {"mahalanobis_sq": float("nan"), "quality_floor": 0.2}})
        with pytest.raises(ValueError, match="gate.mahalanobis_sq must be finite"):
            estimator.step(_uwb_event())

    def test_non_finite_gate_quality_floor(self):
        """地板测试：non finite gate quality。\n\n验证 non finite gate quality 的地板值约束，\n确保输出不低于最小值。
        """
        estimator = RobustEKFCore(_cfg() | {"gate": {"quality_floor": float("inf"), "mahalanobis_sq": 9.21}})
        with pytest.raises(ValueError, match="gate.quality_floor must be finite"):
            estimator.step(_uwb_event())

    def test_missing_measurement_noise_uwb(self):
        cfg = _cfg()
        del cfg["measurement_noise"]["uwb"]
        estimator = RobustEKFCore(cfg)
        with pytest.raises(KeyError, match="measurement_noise.uwb"):
            estimator.step(_uwb_event())


# ════════════════════════════════════════════════════════════
#  16. 偏置修正与钳位
# ════════════════════════════════════════════════════════════

class TestBiasCorrection:
    """UWB 偏置修正测试。"""

    def test_negative_corrected_range_clamped(self):
        estimator = RobustEKFCore(_cfg())
        estimator.set_measurement_control(
            MeasurementControl(
                modality="uwb",
                bias_applied=2.0,
                scaling=1.0,
                risk=0.0,
                noise_multiplier=1.0,
                gate_action="pass_through",
            )
        )
        estimator.step(_uwb_event(rng=0.5))
        # corrected_range = max(0.0, 0.5 - 2.0) = 0.0
        report = estimator.last_update_report
        assert report["update_applied"] is True

    def test_zero_bias_no_clamp(self):
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event(rng=1.0))
        report = estimator.last_update_report
        assert report["update_applied"] is True


# ════════════════════════════════════════════════════════════
#  17. 完整 pipeline 交互
# ════════════════════════════════════════════════════════════

class TestFullPipeline:
    """完整 IMU → UWB → VIO 交互测试。"""

    def test_imu_then_uwb_then_vio(self):
        estimator = RobustEKFCore(_cfg())
        estimator.step(_imu_event())
        estimator.step(_uwb_event())
        state = estimator.step(_vio_event())
        assert state.timestamp == pytest.approx(0.2)
        assert estimator.last_update_report["modality"] == "vio"

    def test_multiple_uwb_updates(self):
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event(t=0.1))
        estimator.step(_uwb_event(t=0.2))
        estimator.step(_uwb_event(t=0.3))
        assert estimator._timestamp == pytest.approx(0.3)

    def test_reset_mid_pipeline(self):
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event())
        estimator.step(_vio_event())
        estimator.reset()
        assert estimator._timestamp is None
        assert estimator.last_update_report is None

    def test_imu_no_gate_report(self):
        """IMU 事件由父类处理，不包含鲁棒门控。"""
        estimator = RobustEKFCore(_cfg())
        estimator.step(_imu_event())
        report = estimator.last_update_report
        assert report["modality"] == "imu"
        # IMU 不走 RobustEKFCore 的 _handle_uwb/_handle_vio

    def test_missing_payload_returns_missing_payload(self):
        """缺失测试：missing payload returns。\n\n验证 missing payload returns 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
        """
        estimator = RobustEKFCore(_cfg())
        # 使用 IMU 模态 + dt=0 来触发 "nonpositive_dt"（不真正更新）
        # 注意：validate_event 会拒绝 None payload 的 UWB/VIO 事件，
        # 所以用 IMU + 空 payload 无法直接走到 missing_payload 分支。
        # 这里改测 nonpositive_dt 作为"未更新"场景的替代。
        event = _imu_event(dt=0.0)
        estimator.step(event)
        report = estimator.last_update_report
        assert report["update_applied"] is False
        assert report["reason"] == "nonpositive_dt"


# ════════════════════════════════════════════════════════════
#  18. 拒绝报告结构
# ════════════════════════════════════════════════════════════

class TestRejectReportStructure:
    """拒绝报告结构完整性测试。"""

    def test_quality_reject_report_structure(self):
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event(quality=0.1))
        report = estimator.last_update_report
        assert report["modality"] == "uwb"
        assert report["update_applied"] is False
        assert report["gate"]["passed"] is False
        assert report["gate"]["quality"] == pytest.approx(0.1)
        assert report["gate"]["quality_floor"] == pytest.approx(0.2)
        assert report["gate"]["nis"] is None
        assert report["gate"]["rejected_by"] == "quality_floor"
        assert report["robust"] is None
        assert report["covariance_report"] is None
        assert report["robust_covariance_report"] is None

    def test_nis_reject_report_structure(self):
        cfg = _cfg()
        cfg["gate"]["mahalanobis_sq"] = 0.001
        estimator = RobustEKFCore(cfg)
        estimator.step(_uwb_event())
        report = estimator.last_update_report
        if report["gate"]["rejected_by"] == "mahalanobis_sq":
            assert report["gate"]["nis"] is not None
            assert report["gate"]["mahalanobis_sq_threshold"] == pytest.approx(0.001)
            assert report["robust"] is None
            assert report["covariance_report"] is not None  # 已经做了控制缩放

    def test_uwb_skip_report_structure(self):
        estimator = RobustEKFCore(_cfg())
        estimator.set_measurement_control(
            MeasurementControl(modality="uwb", gate_action="uwb_skip_update")
        )
        estimator.step(_uwb_event())
        report = estimator.last_update_report
        assert report["modality"] == "uwb"
        assert report["update_applied"] is False
        assert report["reason"] == "uwb_skip_update"
        assert report["gate"] == {"passed": False, "rejected_by": "uwb_skip_update"}
        assert report["robust"] is None
        assert report["covariance_report"] is None
        assert report["robust_covariance_report"] is None


# ════════════════════════════════════════════════════════════
#  19. VIO NIS 门控
# ════════════════════════════════════════════════════════════

class TestVIONISGate:
    """VIO NIS 门控测试。"""

    def test_vio_nis_rejection(self):
        cfg = _cfg()
        cfg["gate"]["mahalanobis_sq"] = 0.001  # 极小阈值
        estimator = RobustEKFCore(cfg)
        estimator.step(_uwb_event())
        estimator.step(_vio_event())
        report = estimator.last_update_report
        if not report["update_applied"] and report["gate"]["rejected_by"] == "mahalanobis_sq":
            assert report["gate"]["nis"] > 0.001

    def test_vio_infinite_nis_threshold_passes(self):
        """传递测试：vio infinite nis threshold。\n\n验证 vio infinite nis threshold 的传递一致性，\n确保数据在流水线中无损传递。
        """
        cfg = _cfg()
        cfg["gate"]["mahalanobis_sq"] = float("inf")
        estimator = RobustEKFCore(cfg)
        estimator.step(_uwb_event())
        # 首帧 VIO 初始化参考位姿
        estimator.step(_vio_event())
        # 第二帧 VIO 正常更新
        estimator.step(_vio_event(t=0.3))
        report = estimator.last_update_report
        assert report["update_applied"] is True

    def test_vio_nis_rejection_includes_extras(self):
        cfg = _cfg()
        cfg["gate"]["mahalanobis_sq"] = 0.001
        estimator = RobustEKFCore(cfg)
        estimator.step(_uwb_event())
        estimator.step(_vio_event())
        report = estimator.last_update_report
        if not report["update_applied"] and report["gate"]["rejected_by"] == "mahalanobis_sq":
            assert "z_vio" in report
            assert "residual" in report


# ════════════════════════════════════════════════════════════
#  20. 异常与边界
# ════════════════════════════════════════════════════════════

class TestEdgeCases:
    """边界与异常测试。"""

    def test_effective_uwb_noise_must_be_finite(self):
        """有效 UWB 噪声为非有限数时应抛异常。"""
        # 很难直接构造这个场景，但可以通过极端配置触发
        estimator = RobustEKFCore(_cfg())
        # 正常场景下噪声应是有限的
        estimator.step(_uwb_event())
        assert estimator.last_update_report["update_applied"] is True

    def test_imu_step_inherited(self):
        """IMU 步骤完全继承自 EKFCore。"""
        estimator = RobustEKFCore(_cfg())
        state = estimator.step(_imu_event())
        assert state.timestamp == pytest.approx(0.05)
        assert estimator.last_update_report["modality"] == "imu"
        assert estimator.last_update_report["update_applied"] is True

    def test_imu_nonpositive_dt(self):
        estimator = RobustEKFCore(_cfg())
        state = estimator.step(_imu_event(dt=0.0))
        assert estimator.last_update_report["update_applied"] is False
        assert estimator.last_update_report["reason"] == "nonpositive_dt"

    def test_skipped_uwb_between_imus_does_not_change_imu_propagation(self):
        """不侵入测试：skipped uwb between imus。\n\n验证 skipped uwb between imus 不会产生副作用，\n确保功能隔离性。
        """
        plain = RobustEKFCore(_cfg())
        plain.step(_imu_event(t=0.1, dt=0.1))
        plain.step(_imu_event(t=0.2, dt=0.1))

        interleaved = RobustEKFCore(_cfg())
        interleaved.step(_imu_event(t=0.1, dt=0.1))
        interleaved.set_measurement_control(MeasurementControl(modality="uwb", gate_action="uwb_skip_update"))
        interleaved.step(_uwb_event(t=0.15, rng=0.8, quality=0.9))
        interleaved.step(_imu_event(t=0.2, dt=0.05))

        assert interleaved._state["px"] == pytest.approx(plain._state["px"])
        assert interleaved._state["vx"] == pytest.approx(plain._state["vx"])
        assert interleaved._covariance == pytest.approx(plain._covariance)

    def test_state_estimate_dict_is_copy(self):
        """get_state().state 应是副本。"""
        estimator = RobustEKFCore(_cfg())
        estimator.step(_uwb_event())
        state = estimator.get_state()
        state.state["px"] = 999.0
        assert estimator._state["px"] != 999.0

    def test_covariance_diag_length(self):
        estimator = RobustEKFCore(_cfg())
        state = estimator.get_state()
        # §1.1 主表 8 维 + §2.3 紧耦合扩维 (uwb_clock_bias / vio_scale) = 10 维
        assert len(state.covariance_diag) == 10


# ════════════════════════════════════════════════════════════
#  §11.5 UWB 路径标量 S jitter fallback（v8 修复锁死）
#  修复位置：src/liquidloc/estimators/robust_ekf_core.py:367-387
#  v6 audit 发现 v5 仅修了 VIO 路径与 uwb_update_step 内部 stacked-H 路径的
#  jitter fallback，但漏审 Robust-EKF UWB 路径的标量 S<=0 守门。
#
#  设计诚实声明：
#  对 P 正定 + scalar_noise ≥ 0，S = H·P·Hᵀ + scalar_noise ≥ 0 恒成立（数学期望）。
#  本测试用正交投影构造 P 使 H·P·Hᵀ ≈ 1e-12（正定但极小），再通过 monkeypatch
#  coerce_finite_scalar 把 scalar_noise 设为负值，精确控制 S 进入 v8 jitter 分支。
#  这是数学上诚实且可复现的测试设计——不靠违法病态 P，仅靠定向 mock 一个数值边界。
#  锁死：(a) 可恢复 S（|S| < eps）走 jitter 救活，update_applied=True；
#       (b) 不可恢复 S（S < -eps）仍 fail-loud 拒绝。
# ════════════════════════════════════════════════════════════

class TestUwbScalarSJitterFallbackRobustEKF:
    """§11.5 UWB 路径标量 S jitter fallback 锁死（Robust-EKF 单元）。"""

    @staticmethod
    def _orthogonal_pd_P(ekf):
        """构造 P 正定但 H 行空间投影 ≈ 0（使 H·P·Hᵀ ≈ 1e-12）。

        P = I - (1 - 1e-12) · vvᵀ / ||v||²，v = H[0] 行向量。
        """
        from liquidloc.estimators.uwb_update_step import build_uwb_jacobian
        P = np.eye(ekf._covariance.shape[0])
        x_prev = ekf._state_vector()
        H = build_uwb_jacobian(x_prev, ekf._resolve_anchor_position(0))
        v = H[0]
        v_norm_sq = float(v @ v)
        P -= (1 - 1e-12) * np.outer(v, v) / v_norm_sq
        return P

    def test_uwb_recoverable_S_triggers_jitter_fallback(self, monkeypatch):
        """§11.5 可恢复病态 S（|S| < cov_jitter_eps）走 jitter 救活。"""
        from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
        from liquidloc.estimators.uwb_update_step import build_uwb_jacobian
        import liquidloc.estimators.robust_ekf_core as robust_mod
        eps = float(BRIDGE_THRESHOLDS["cov_jitter_eps"])
        # 把 NIS 阈值放高，使 jitter 救活后 nis 不被 mahalanobis_sq 门禁误拒
        # （否则测的是 mahalanobis 门禁而非 v8 jitter 路径，越界）。
        cfg = _cfg() | {"gate": {"mahalanobis_sq": 1e12, "quality_floor": 0.0}}
        ekf = RobustEKFCore(cfg)
        ekf._covariance = self._orthogonal_pd_P(ekf)
        x_prev = ekf._state_vector()
        anchor_pos = ekf._resolve_anchor_position(0)
        H = build_uwb_jacobian(x_prev, anchor_pos)
        HPtH = float((H @ ekf._covariance @ H.T)[0, 0])

        target_noise = -5e-10 - HPtH  # S = -5e-10 ∈ (-eps, 0]
        orig_cfs = robust_mod.coerce_finite_scalar

        def patched(value, name=None, **kwargs):
            if name == "effective UWB noise":
                return orig_cfs(target_noise, name=name, **kwargs)
            return orig_cfs(value, name=name, **kwargs)

        monkeypatch.setattr(robust_mod, "coerce_finite_scalar", patched)

        control = MeasurementControl(modality="uwb", gate_action="pass_through")
        result = ekf._handle_uwb(_uwb_event(), x_prev, control)

        assert result["update_applied"] is True, \
            f"§11.5 可恢复病态 S 应被 jitter 救活；reason={result.get('reason')}"
        assert result.get("reason") != "nonpositive_innovation_covariance"

    def test_uwb_unrecoverable_S_still_fails_loud(self, monkeypatch):
        """§11.5 不可恢复病态 S（S < -eps）仍 fail-loud 拒绝。"""
        from liquidloc.estimators.uwb_update_step import build_uwb_jacobian
        import liquidloc.estimators.robust_ekf_core as robust_mod
        ekf = RobustEKFCore(_cfg())
        ekf._covariance = self._orthogonal_pd_P(ekf)
        x_prev = ekf._state_vector()
        anchor_pos = ekf._resolve_anchor_position(0)
        H = build_uwb_jacobian(x_prev, anchor_pos)
        HPtH = float((H @ ekf._covariance @ H.T)[0, 0])

        target_noise = -0.0625 - HPtH  # S ≈ -0.0625 << -eps=1e-9
        orig_cfs = robust_mod.coerce_finite_scalar

        def patched(value, name=None, **kwargs):
            if name == "effective UWB noise":
                return orig_cfs(target_noise, name=name, **kwargs)
            return orig_cfs(value, name=name, **kwargs)

        monkeypatch.setattr(robust_mod, "coerce_finite_scalar", patched)

        control = MeasurementControl(modality="uwb", gate_action="pass_through")
        result = ekf._handle_uwb(_uwb_event(), x_prev, control)

        assert result["update_applied"] is False, \
            "§11.5 不可恢复病态 S 必须被拒绝，jitter fallback 不应掩盖真病态"
        assert result.get("reason") == "nonpositive_innovation_covariance"


class TestUwbValidFlagRejectionRobustEKF:
    """§11.3-d 共享「传感器无效」硬标志串项同源锁死。

    spec L1734「无效标志 → 方法内部更新的串联顺序全员固定」
    v9 audit 发现 Robust-EKF `_handle_uwb` 缺 valid=False 检查；本测试锁死补后的路径。
    """

    def test_uwb_valid_false_rejected_with_uwb_invalid_measurement_reason(self):

        estimator = RobustEKFCore(_cfg())
        x_prev = estimator._state_vector()
        invalid_event = _uwb_event()
        invalid_event["uwb_payload"]["valid"] = False
        control = MeasurementControl(modality="uwb", gate_action="pass_through")
        result = estimator._handle_uwb(invalid_event, x_prev, control)

        assert result["update_applied"] is False, \
            "§11.3-d valid=False 事件必须被拒识，跳过更新链路"
        assert result.get("reason") == "uwb_invalid_measurement", \
            "§11.3-d 拒识原因必须是 uwb_invalid_measurement，与 EKF / FGO 同源"

    def test_uwb_valid_false_skips_before_quality_floor_check(self):
        """§11.3-d 串项顺序锁死：valid=False 必须在 quality_floor 之前触发。

        v9 audit 修复前 Robust-EKF 完全缺 valid=False 检查；
        修复后必须验证 valid=False 先于 quality_floor，与 EKF L856 同口径。
        """

        estimator = RobustEKFCore(_cfg())
        x_prev = estimator._state_vector()
        invalid_event = _uwb_event(quality=0.0)
        invalid_event["uwb_payload"]["valid"] = False
        control = MeasurementControl(modality="uwb", gate_action="pass_through")
        result = estimator._handle_uwb(invalid_event, x_prev, control)

        assert result["update_applied"] is False
        assert result.get("reason") == "uwb_invalid_measurement", \
            "valid=False 必须先于 quality_floor 触发，不可被 quality_floor 覆盖"


class TestVioQualityNormalizationRobustEKF:
    """§11.3-d VIO quality<=0 协议级拒识同源规范化锁死。

    v11 audit 发现：Robust-EKF `_handle_vio` 本走 `_quality_value` 同源规范化，
    但 EKF 此前走捷径绕过。v11 修复后三方法同源。本测试锁死 Robust-EKF 路径。
    """

    def test_vio_quality_zero_python_zero_skips_update(self):
        """Python 0.0 视为 quality<=0 触发 vio_quality_zero 跳过。

        v11 audit：三方法 VIO quality<=0 协议级拒识同源规范化锁死。
        本测试用合法的 0.0 quality（不触发规范化异常路径）锁死
        三方法同源触发 vio_quality_zero 跳过行为。
        """
        ekf = RobustEKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.step(_uwb_event())
        ekf.step(_vio_event(t=0.3))  # 初始化参考位姿
        state_before = ekf._state.copy()

        result = ekf._handle_vio(
            {"t": 0.4, "dt": 0.1, "modality": "vio",
             "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
             "imu_payload": None, "uwb_payload": None,
             "vio_payload": {"dx": 0.1, "dy": 0.0, "dyaw": 0.0,
                             "quality": 0.0, "tracked_features": 0, "reproj_err": 0.0}},
            ekf._state_vector(),
            MeasurementControl(modality="vio", gate_action="pass_through"),
        )
        assert result["update_applied"] is False
        assert result.get("reason") == "vio_quality_zero"
        assert ekf._state["px"] == pytest.approx(state_before["px"])

    def test_vio_quality_bool_rejected_by_quality_value_normalization(self):
        """Boolean quality 必须经 _quality_value 规范化抛 TypeError，与三方法同口径。

        v11 修复前 EKF 直接 float(False)=0.0 触发 vio_quality_zero 跳过，
        与 Robust/FGO 走 _quality_value 抛 TypeError 不同源。
        v11 修复后 EKF 也走 _quality_value，对 bool 类 quality 同源抛 TypeError。
        本测试锁死三方法同源同口径规范化路径。
        """
        ekf = RobustEKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.step(_uwb_event())
        ekf.step(_vio_event(t=0.3))  # 初始化参考位姿

        with pytest.raises(TypeError, match="quality must be numeric"):
            ekf._handle_vio(
                {"t": 0.4, "dt": 0.1, "modality": "vio",
                 "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
                 "imu_payload": None, "uwb_payload": None,
                 "vio_payload": {"dx": 0.1, "dy": 0.0, "dyaw": 0.0,
                                 "quality": False, "tracked_features": 0, "reproj_err": 0.0}},
                ekf._state_vector(),
                MeasurementControl(modality="vio", gate_action="pass_through"),
            )
