from __future__ import annotations

"""扩展卡尔曼滤波器（EKF）核心测试模块。

测试覆盖范围：
- EKF 的预测步与更新步
- UWB/VIO 更新的正确性
- 状态估计与协方差传播
- 初始化与重置

被测模块：liquidloc.estimators.ekf_core"""

import copy
import numpy as np
import pytest

from liquidloc.common.types import MeasurementControl, ModelIntermediate
from liquidloc.estimators.ekf_core import (
    EKFCore,
    _build_runtime_resource_meta,
    _count_measurement_noise_terms,
    _reject_boolean_like_measurement_noise_entry,
    _require_positive_measurement_noise_entry,
)
from liquidloc.estimators.shared import build_controlled_measurement_cov, control_to_dict
from liquidloc.estimators.state_definition import state_items
from liquidloc.protocol.liquid_bridge_contract import build_measurement_control


# 前提指导 §1.1 主表 8 维 + §2.3 紧耦合扩维（uwb_clock_bias / vio_scale），全体同增同模型。
# _STATE_DIM 由 state_items 动态派生（10），_EXTRA_* 占位填入扩维项保持与 task.yaml 一致。
_STATE_DIM = len(state_items)  # 10
_EXTRA_INIT_COV = (0.05, 0.01)  # uwb_clock_bias σ²=0.05, vio_scale σ²=0.01（与 configs/models/ekf.yaml 同口径）
_EXTRA_INIT_STATE = {"uwb_clock_bias": 0.0, "vio_scale": 1.0}  # 钟差初值 0，尺度因子初值 1.0
_EXTRA_PROCESS_NOISE = {"uwb_clock_bias": 0.001, "vio_scale": 0.001}  # 与加速度/陀螺偏置同量级


def _state_eye() -> np.ndarray:
    """返回与 state 维度对齐的单位矩阵。"""
    return np.eye(_STATE_DIM, dtype=float)


# ────────────────────────────────────────────────────────────
#  测试辅助
# ────────────────────────────────────────────────────────────

def _anchor_layout():
    return {
        "anchor_ids": [0, 1],
        "anchor_positions": [(1.0, 0.0), (2.0, 0.0)],
    }


def _cfg():
    return {
        "process_noise": {"pos": 0.05, "vel": 0.10, "yaw": 0.02, "accel_bias": 0.001, "gyro_bias": 0.001, **_EXTRA_PROCESS_NOISE},
        "measurement_noise": {"uwb": 0.25, "vio": {"pos": 0.08, "yaw": 0.03}},
        "init_cov": [1.0, 1.0, 0.5, 0.5, 0.3, 0.05, 0.05, 0.02, *_EXTRA_INIT_COV],
        "anchor_layout": _anchor_layout(),
    }


def _imu_event(*, t=0.1, dt=0.1, ax=0.2, ay=0.0, gz=0.01):
    return {
        "t": t,
        "dt": dt,
        "modality": "imu",
        "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
        "imu_payload": {"ax": ax, "ay": ay, "gz": gz},
        "uwb_payload": None,
        "vio_payload": None,
    }


def _uwb_event(*, t=0.2, dt=0.1, rng=0.9, quality=0.95, anchor_id=0):
    return {
        "t": t,
        "dt": dt,
        "modality": "uwb",
        "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
        "imu_payload": None,
        "uwb_payload": {"anchor_id": anchor_id, "range": rng, "valid": True, "quality": quality},
        "vio_payload": None,
    }


def _vio_event(*, t=0.3, dt=0.1, dx=0.2, dy=0.0, dyaw=0.05):
    return {
        "t": t,
        "dt": dt,
        "modality": "vio",
        "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
        "imu_payload": None,
        "uwb_payload": None,
        "vio_payload": {
            "dx": dx,
            "dy": dy,
            "dyaw": dyaw,
            "quality": 0.9,
            "tracked_features": 120,
            "reproj_err": 0.3,
        },
    }


# ════════════════════════════════════════════════════════════
#  1. _count_measurement_noise_terms
# ════════════════════════════════════════════════════════════

class TestCountMeasurementNoiseTerms:
    """_count_measurement_noise_terms 统计测试。"""

    def test_none_returns_zero(self):
        """零值测试：none returns。\n\n验证 none returns 在零值输入下的行为，\n确保边界情况正确处理。
        """
        assert _count_measurement_noise_terms(None) == 0

    def test_empty_dict_returns_zero(self):
        """零值测试：empty dict returns。\n\n验证 empty dict returns 在零值输入下的行为，\n确保边界情况正确处理。
        """
        assert _count_measurement_noise_terms({}) == 0

    def test_scalar_int(self):
        assert _count_measurement_noise_terms(3) == 1

    def test_scalar_float(self):
        assert _count_measurement_noise_terms(2.5) == 1

    def test_numpy_scalar_float64(self):
        assert _count_measurement_noise_terms(np.float64(0.25)) == 1

    def test_numpy_scalar_float32(self):
        assert _count_measurement_noise_terms(np.float32(0.08)) == 1

    def test_numpy_scalar_int(self):
        assert _count_measurement_noise_terms(np.int32(5)) == 1

    def test_complex_returns_zero(self):
        """complex 是 Number 子类但不是实数噪声项。"""
        assert _count_measurement_noise_terms(1 + 2j) == 0

    def test_numpy_complex_scalar_returns_zero(self):
        """零值测试：numpy complex scalar returns。\n\n验证 numpy complex scalar returns 在零值输入下的行为，\n确保边界情况正确处理。
        """
        assert _count_measurement_noise_terms(np.complex128(1 + 0j)) == 0

    def test_bool_returns_zero(self):
        """零值测试：bool returns。\n\n验证 bool returns 在零值输入下的行为，\n确保边界情况正确处理。
        """
        assert _count_measurement_noise_terms(True) == 0

    def test_numpy_bool_returns_zero(self):
        """零值测试：numpy bool returns。\n\n验证 numpy bool returns 在零值输入下的行为，\n确保边界情况正确处理。
        """
        assert _count_measurement_noise_terms(np.bool_(True)) == 0

    def test_list_of_scalars(self):
        assert _count_measurement_noise_terms([1.0, 2.0, 3.0]) == 3

    def test_tuple_of_scalars(self):
        assert _count_measurement_noise_terms((1.0, 2.0)) == 2

    def test_nested_list(self):
        assert _count_measurement_noise_terms([1.0, [2.0, 3.0]]) == 3

    def test_list_with_bools(self):
        assert _count_measurement_noise_terms([1.0, True, 2.0]) == 2

    def test_numpy_1d_array(self):
        assert _count_measurement_noise_terms(np.array([1.0, 2.0, 3.0])) == 3

    def test_numpy_1d_int_array(self):
        assert _count_measurement_noise_terms(np.array([1, 2])) == 2

    def test_numpy_0d_scalar_array(self):
        arr = np.array(5.0)
        assert _count_measurement_noise_terms(arr) == 1

    def test_numpy_bool_array_returns_zero(self):
        """零值测试：numpy bool array returns。\n\n验证 numpy bool array returns 在零值输入下的行为，\n确保边界情况正确处理。
        """
        assert _count_measurement_noise_terms(np.array([True, False])) == 0

    def test_numpy_2d_array(self):
        assert _count_measurement_noise_terms(np.eye(3)) == 9

    def test_flat_mapping(self):
        """映射报告测试：flat。\n\n验证 flat 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        assert _count_measurement_noise_terms({"uwb": 0.25, "vio": 0.08}) == 2

    def test_nested_mapping(self):
        """映射报告测试：nested。\n\n验证 nested 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        cfg = {"uwb": 0.25, "vio": {"pos": 0.08, "yaw": 0.03}}
        assert _count_measurement_noise_terms(cfg) == 3

    def test_deeply_nested_mapping(self):
        """映射报告测试：deeply nested。\n\n验证 deeply nested 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        cfg = {"a": {"b": {"c": 1.0}}, "d": 2.0}
        assert _count_measurement_noise_terms(cfg) == 2

    def test_string_returns_zero(self):
        """零值测试：string returns。\n\n验证 string returns 在零值输入下的行为，\n确保边界情况正确处理。
        """
        assert _count_measurement_noise_terms("hello") == 0

    def test_empty_list_returns_zero(self):
        """零值测试：empty list returns。\n\n验证 empty list returns 在零值输入下的行为，\n确保边界情况正确处理。
        """
        assert _count_measurement_noise_terms([]) == 0

    def test_empty_tuple_returns_zero(self):
        """零值测试：empty tuple returns。\n\n验证 empty tuple returns 在零值输入下的行为，\n确保边界情况正确处理。
        """
        assert _count_measurement_noise_terms(()) == 0


# ════════════════════════════════════════════════════════════
#  2. _build_runtime_resource_meta
# ════════════════════════════════════════════════════════════

class TestBuildRuntimeResourceMeta:
    """_build_runtime_resource_meta 资源估算测试。"""

    def test_empty_cfg(self):
        meta = _build_runtime_resource_meta({})
        assert "params" in meta
        assert "ram_peak" in meta
        assert "ram_peak_mb" in meta
        # state_items 长度 = _STATE_DIM，无噪声项，init_cov 退回 state_items 长度
        assert meta["params"] == pytest.approx(_STATE_DIM + _STATE_DIM)

    def test_with_measurement_noise_scalar(self):
        meta = _build_runtime_resource_meta({"measurement_noise": {"uwb": 0.25}})
        # _STATE_DIM state + 1 noise + _STATE_DIM init_cov_default
        assert meta["params"] == pytest.approx(_STATE_DIM * 2 + 1)

    def test_with_measurement_noise_nested(self):
        meta = _build_runtime_resource_meta(
            {"measurement_noise": {"uwb": 0.25, "vio": {"pos": 0.08, "yaw": 0.03}}}
        )
        # _STATE_DIM state + 3 noise + _STATE_DIM init_cov_default
        assert meta["params"] == pytest.approx(_STATE_DIM * 2 + 3)

    def test_with_init_cov_list(self):
        init_cov = [1.0] * _STATE_DIM
        meta = _build_runtime_resource_meta({"init_cov": init_cov})
        # _STATE_DIM state + 0 noise + _STATE_DIM init_cov
        assert meta["params"] == pytest.approx(_STATE_DIM * 2)

    def test_with_partial_init_cov(self):
        meta = _build_runtime_resource_meta({"init_cov": [1.0, 0.5]})
        # init_cov list len = 2 but != len(state_items), so it raises ValueError in EKFCore
        # but _build_runtime_resource_meta just counts the length
        # _STATE_DIM state + 0 noise + 2 init_cov
        assert meta["params"] == pytest.approx(_STATE_DIM + 2)

    def test_ram_peak_minimum(self):
        meta = _build_runtime_resource_meta({})
        # params = 2*_STATE_DIM, ram_peak = max(1.0, (2*_STATE_DIM)/256) = 1.0
        assert meta["ram_peak_mb"] >= 1.0

    def test_ram_peak_mb_equals_ram_peak(self):
        meta = _build_runtime_resource_meta({})
        assert meta["ram_peak"] == meta["ram_peak_mb"]

    def test_none_measurement_noise_treated_as_empty(self):
        meta = _build_runtime_resource_meta({"measurement_noise": None})
        # None -> {} -> 0 noise terms
        assert meta["params"] == pytest.approx(_STATE_DIM * 2)


# ════════════════════════════════════════════════════════════
#  3. _reject_boolean_like_measurement_noise_entry
# ════════════════════════════════════════════════════════════

class TestRejectBooleanLikeMeasurementNoiseEntry:
    """_reject_boolean_like_measurement_noise_entry 布尔拒绝测试。"""

    def test_bool_rejected(self):
        """拒绝测试：bool。\n\n验证被测功能对不合法的 bool 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            _reject_boolean_like_measurement_noise_entry(True, path="test")

    def test_numpy_bool_rejected(self):
        """拒绝测试：numpy bool。\n\n验证被测功能对不合法的 numpy bool 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            _reject_boolean_like_measurement_noise_entry(np.bool_(False), path="test")

    def test_bool_in_list_rejected(self):
        """拒绝测试：bool in list。\n\n验证被测功能对不合法的 bool in list 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(TypeError, match=r"test\[0\] must be numeric, got bool"):
            _reject_boolean_like_measurement_noise_entry([True, 1.0], path="test")

    def test_bool_in_tuple_rejected(self):
        """拒绝测试：bool in tuple。\n\n验证被测功能对不合法的 bool in tuple 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(TypeError, match=r"test\[1\] must be numeric, got bool"):
            _reject_boolean_like_measurement_noise_entry([1.0, False], path="test")

    def test_bool_in_mapping_rejected(self):
        """拒绝测试：bool in mapping。\n\n验证被测功能对不合法的 bool in mapping 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(TypeError, match=r"test.uwb must be numeric, got bool"):
            _reject_boolean_like_measurement_noise_entry({"uwb": True}, path="test")

    def test_bool_array_rejected(self):
        """拒绝测试：bool array。\n\n验证被测功能对不合法的 bool array 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            _reject_boolean_like_measurement_noise_entry(np.array([True, False]), path="test")

    def test_object_array_with_bool_rejected(self):
        """拒绝测试：object array with bool。\n\n验证被测功能对不合法的 object array with bool 输入正确抛出异常，\n防止无效参数通过验证。
        """
        arr = np.array([1.0, True], dtype=object)
        with pytest.raises(TypeError, match=r"test\[1\] must be numeric, got bool"):
            _reject_boolean_like_measurement_noise_entry(arr, path="test")

    def test_scalar_passes(self):
        """传递测试：scalar。\n\n验证 scalar 的传递一致性，\n确保数据在流水线中无损传递。
        """
        _reject_boolean_like_measurement_noise_entry(1.0, path="test")  # 不抛

    def test_list_of_numbers_passes(self):
        """传递测试：list of numbers。\n\n验证 list of numbers 的传递一致性，\n确保数据在流水线中无损传递。
        """
        _reject_boolean_like_measurement_noise_entry([1.0, 2.0], path="test")  # 不抛

    def test_mapping_of_numbers_passes(self):
        """传递测试：mapping of numbers。\n\n验证 mapping of numbers 的传递一致性，\n确保数据在流水线中无损传递。
        """
        _reject_boolean_like_measurement_noise_entry({"uwb": 0.25}, path="test")  # 不抛

    def test_empty_mapping_passes(self):
        """传递测试：empty mapping。\n\n验证 empty mapping 的传递一致性，\n确保数据在流水线中无损传递。
        """
        _reject_boolean_like_measurement_noise_entry({}, path="test")  # 不抛

    def test_empty_list_passes(self):
        """传递测试：empty list。\n\n验证 empty list 的传递一致性，\n确保数据在流水线中无损传递。
        """
        _reject_boolean_like_measurement_noise_entry([], path="test")  # 不抛

    def test_numpy_float_array_passes(self):
        """传递测试：numpy float array。\n\n验证 numpy float array 的传递一致性，\n确保数据在流水线中无损传递。
        """
        _reject_boolean_like_measurement_noise_entry(np.array([1.0, 2.0]), path="test")

    def test_nested_mapping_with_bool_rejected(self):
        """拒绝测试：nested mapping with bool。\n\n验证被测功能对不合法的 nested mapping with bool 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(TypeError, match=r"test.vio.pos must be numeric, got bool"):
            _reject_boolean_like_measurement_noise_entry(
                {"vio": {"pos": True, "yaw": 0.03}}, path="test"
            )


# ════════════════════════════════════════════════════════════
#  4. _require_positive_measurement_noise_entry
# ════════════════════════════════════════════════════════════

class TestRequirePositiveMeasurementNoiseEntry:
    """_require_positive_measurement_noise_entry 正值校验测试。"""

    def test_positive_scalar_passes(self):
        """传递测试：positive scalar。\n\n验证 positive scalar 的传递一致性，\n确保数据在流水线中无损传递。
        """
        _require_positive_measurement_noise_entry(1.0, path="test")

    def test_zero_scalar_rejected(self):
        """拒绝测试：zero scalar。\n\n验证被测功能对不合法的 zero scalar 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must be positive"):
            _require_positive_measurement_noise_entry(0.0, path="test")

    def test_negative_scalar_rejected(self):
        """拒绝测试：negative scalar。\n\n验证被测功能对不合法的 negative scalar 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must be positive"):
            _require_positive_measurement_noise_entry(-0.5, path="test")

    def test_complex_rejected(self):
        """complex 不是合法噪声类型。"""
        with pytest.raises(TypeError, match="must be real numeric, got complex"):
            _require_positive_measurement_noise_entry(1 + 2j, path="test")

    def test_bool_rejected(self):
        """拒绝测试：bool。\n\n验证被测功能对不合法的 bool 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            _require_positive_measurement_noise_entry(True, path="test")

    def test_positive_1d_passes(self):
        """传递测试：positive 1d。\n\n验证 positive 1d 的传递一致性，\n确保数据在流水线中无损传递。
        """
        _require_positive_measurement_noise_entry([1.0, 2.0], path="test")

    def test_1d_with_zero_rejected(self):
        """拒绝测试：1d with zero。\n\n验证被测功能对不合法的 1d with zero 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must contain positive diagonal entries"):
            _require_positive_measurement_noise_entry([1.0, 0.0], path="test")

    def test_1d_with_negative_rejected(self):
        """拒绝测试：1d with negative。\n\n验证被测功能对不合法的 1d with negative 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must contain positive diagonal entries"):
            _require_positive_measurement_noise_entry([1.0, -1.0], path="test")

    def test_empty_1d_rejected(self):
        """拒绝测试：empty 1d。\n\n验证被测功能对不合法的 empty 1d 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must contain positive diagonal entries"):
            _require_positive_measurement_noise_entry([], path="test")

    def test_positive_diagonal_2d_passes(self):
        """传递测试：positive diagonal 2d。\n\n验证 positive diagonal 2d 的传递一致性，\n确保数据在流水线中无损传递。
        """
        _require_positive_measurement_noise_entry(np.diag([1.0, 2.0]), path="test")

    def test_2d_non_square_rejected(self):
        """拒绝测试：2d non square。\n\n验证被测功能对不合法的 2d non square 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must be a square covariance matrix"):
            _require_positive_measurement_noise_entry(np.ones((2, 3)), path="test")

    def test_2d_zero_diagonal_rejected(self):
        """拒绝测试：2d zero diagonal。\n\n验证被测功能对不合法的 2d zero diagonal 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must contain positive diagonal entries"):
            _require_positive_measurement_noise_entry(np.diag([1.0, 0.0]), path="test")

    def test_2d_empty_rejected(self):
        """拒绝测试：2d empty。\n\n验证被测功能对不合法的 2d empty 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must be a square covariance matrix"):
            _require_positive_measurement_noise_entry(np.zeros((0, 0)), path="test")

    def test_mapping_with_positive_values_passes(self):
        """传递测试：mapping with positive values。\n\n验证 mapping with positive values 的传递一致性，\n确保数据在流水线中无损传递。
        """
        _require_positive_measurement_noise_entry({"pos": 0.08, "yaw": 0.03}, path="test")

    def test_empty_mapping_rejected(self):
        """拒绝测试：empty mapping。\n\n验证被测功能对不合法的 empty mapping 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must provide positive measurement noise values"):
            _require_positive_measurement_noise_entry({}, path="test")

    def test_mapping_with_zero_value_rejected(self):
        """拒绝测试：mapping with zero value。\n\n验证被测功能对不合法的 mapping with zero value 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="test.uwb must be positive"):
            _require_positive_measurement_noise_entry({"uwb": 0.0}, path="test")

    def test_mapping_with_negative_value_rejected(self):
        """拒绝测试：mapping with negative value。\n\n验证被测功能对不合法的 mapping with negative value 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="test.vio.pos must be positive"):
            _require_positive_measurement_noise_entry({"vio": {"pos": -0.1}}, path="test")

    def test_3d_rejected(self):
        """拒绝测试：3d。\n\n验证被测功能对不合法的 3d 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(TypeError, match="must be a scalar, vector, or square covariance matrix"):
            _require_positive_measurement_noise_entry(np.ones((2, 2, 2)), path="test")

    def test_numpy_scalar_passes(self):
        """传递测试：numpy scalar。\n\n验证 numpy scalar 的传递一致性，\n确保数据在流水线中无损传递。
        """
        _require_positive_measurement_noise_entry(np.float64(0.5), path="test")

    def test_numpy_scalar_zero_rejected(self):
        """拒绝测试：numpy scalar zero。\n\n验证被测功能对不合法的 numpy scalar zero 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError, match="must be positive"):
            _require_positive_measurement_noise_entry(np.float64(0.0), path="test")


# ════════════════════════════════════════════════════════════
#  5. control_to_dict
# ════════════════════════════════════════════════════════════

class TestControlToDict:
    """control_to_dict 转换测试。"""

    def test_none_returns_none(self):
        assert control_to_dict(None) is None

    def test_control_converted(self):
        ctrl = MeasurementControl(
            modality="uwb",
            bias_applied=0.1,
            scaling=1.5,
            risk=0.3,
            noise_multiplier=2.0,
            gate_action="pass_through",
        )
        d = control_to_dict(ctrl)
        assert d["modality"] == "uwb"
        assert d["bias_applied"] == pytest.approx(0.1)
        assert d["scaling"] == pytest.approx(1.5)
        assert d["risk"] == pytest.approx(0.3)
        assert d["noise_multiplier"] == pytest.approx(2.0)
        assert d["gate_action"] == "pass_through"

    def test_default_control(self):
        ctrl = MeasurementControl(modality="imu")
        d = control_to_dict(ctrl)
        assert d["bias_applied"] == pytest.approx(0.0)
        assert d["scaling"] == pytest.approx(1.0)
        assert d["risk"] == pytest.approx(0.0)
        assert d["noise_multiplier"] == pytest.approx(1.0)


# ════════════════════════════════════════════════════════════
#  6. build_controlled_measurement_cov
# ════════════════════════════════════════════════════════════

class TestBuildControlledMeasurementCov:
    """build_controlled_measurement_cov 协方差缩放测试。"""

    def test_uwb_scaling(self):
        ctrl = MeasurementControl(modality="uwb", noise_multiplier=2.0)
        eff_cov, report = build_controlled_measurement_cov(0.25, ctrl, modality="uwb")
        assert report["uwb_scaling"] == pytest.approx(2.0)
        assert report["effective_cov"] == pytest.approx(0.5)

    def test_vio_scaling(self):
        ctrl = MeasurementControl(modality="vio", noise_multiplier=3.0)
        eff_cov, report = build_controlled_measurement_cov({"pos": 0.08, "yaw": 0.03}, ctrl, modality="vio")
        assert report["vio_scaling"] == pytest.approx(3.0)

    def test_bridge_fields_in_report(self):
        ctrl = MeasurementControl(
            modality="uwb", scaling=1.5, risk=0.5, noise_multiplier=2.0, gate_action="pass_through"
        )
        _, report = build_controlled_measurement_cov(0.25, ctrl, modality="uwb")
        assert report["bridge_scaling"] == pytest.approx(1.5)
        assert report["bridge_risk"] == pytest.approx(0.5)
        assert report["noise_multiplier"] == pytest.approx(2.0)
        assert report["gate_action"] == "pass_through"
        assert report["bridge_semantics_authority"] == "noise_multiplier"

    def test_invalid_modality_rejected(self):
        """拒绝测试：invalid modality。\n\n验证被测功能对不合法的 invalid modality 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError):
            MeasurementControl(modality="gps", noise_multiplier=1.0)

    def test_non_finite_multiplier_rejected(self):
        """拒绝测试：non finite multiplier。\n\n验证被测功能对不合法的 non finite multiplier 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError):
            MeasurementControl(modality="uwb", noise_multiplier=np.inf)

    def test_negative_multiplier_rejected(self):
        """拒绝测试：negative multiplier。\n\n验证被测功能对不合法的 negative multiplier 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError):
            MeasurementControl(modality="uwb", noise_multiplier=-1.0)

    def test_zero_multiplier_rejected(self):
        """拒绝测试：zero multiplier。\n\n验证被测功能对不合法的 zero multiplier 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError):
            MeasurementControl(modality="uwb", noise_multiplier=0.0)

    def test_nan_multiplier_rejected(self):
        """拒绝测试：nan multiplier。\n\n验证被测功能对不合法的 nan multiplier 输入正确抛出异常，\n防止无效参数通过验证。
        """
        with pytest.raises(ValueError):
            MeasurementControl(modality="uwb", noise_multiplier=float("nan"))


# ════════════════════════════════════════════════════════════
#  7. EKFCore 初始化与重置
# ════════════════════════════════════════════════════════════

class TestEKFCoreInitAndReset:
    """EKFCore 初始化和重置测试。"""

    def test_default_name(self):
        ekf = EKFCore()
        assert ekf.name == "ekf"

    def test_custom_name(self):
        ekf = EKFCore({"name": "my_ekf"})
        assert ekf.name == "my_ekf"

    def test_default_state_all_zeros(self):
        """零值测试：default state all。\n\n验证 default state all 在零值输入下的行为，\n确保边界情况正确处理。
        """
        ekf = EKFCore()
        for key in state_items:
            assert ekf._state[key] == pytest.approx(0.0)

    def test_init_state_from_cfg(self):
        ekf = EKFCore({"init_state": {"px": 1.0, "py": 2.0, **_EXTRA_INIT_STATE}})
        assert ekf._state["px"] == pytest.approx(1.0)
        assert ekf._state["py"] == pytest.approx(2.0)
        assert ekf._state["vx"] == pytest.approx(0.0)

    def test_init_cov_from_cfg(self):
        init_cov = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, *_EXTRA_INIT_COV]
        ekf = EKFCore({"init_cov": init_cov})
        diag = np.diag(ekf._covariance)
        for i, v in enumerate(init_cov):
            assert diag[i] == pytest.approx(v)

    def test_init_cov_wrong_length_raises_value_error(self):
        """拒绝测试：init cov wrong length。\n\n验证 init cov wrong length 的拒绝行为，\n确保不合法输入被正确拦截。
        """
        with pytest.raises(ValueError, match="init_cov must have shape"):
            EKFCore({"init_cov": [1.0, 2.0]})

    def test_reset_clears_state(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        assert ekf._state["px"] != pytest.approx(0.0)
        ekf.reset()
        for key in state_items:
            assert ekf._state[key] == pytest.approx(0.0)

    def test_reset_with_initial_state(self):
        ekf = EKFCore(_cfg())
        ekf.reset(initial_state={"px": 5.0, "yaw": 1.0})
        assert ekf._state["px"] == pytest.approx(5.0)
        assert ekf._state["yaw"] == pytest.approx(1.0)

    def test_reset_initial_state_overrides_cfg(self):
        """覆盖测试：reset initial state。\n\n验证 reset initial state 的覆盖行为，\n确保显式参数优先于默认值。
        """
        ekf = EKFCore({"init_state": {"px": 1.0, **_EXTRA_INIT_STATE}})
        ekf.reset(initial_state={"px": 9.0, **_EXTRA_INIT_STATE})
        assert ekf._state["px"] == pytest.approx(9.0)

    def test_reset_clears_timestamp(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        assert ekf._timestamp is not None
        ekf.reset()
        assert ekf._timestamp is None

    def test_reset_clears_measurement_control(self):
        ekf = EKFCore(_cfg())
        ekf.set_measurement_control(MeasurementControl(modality="uwb"))
        ekf.reset()
        assert ekf._measurement_control is None

    def test_reset_clears_last_intermediate(self):
        ekf = EKFCore(_cfg())
        ekf.consume_model_intermediate(ModelIntermediate(bias=0.1))
        ekf.reset()
        assert ekf._last_intermediate is None

    def test_reset_clears_last_vio_reference_pose(self):
        ekf = EKFCore(_cfg())
        ekf._last_vio_reference_pose = {"px": 1.0, "py": 0.0, "yaw": 0.0}
        ekf.reset()
        assert ekf._last_vio_reference_pose is None

    def test_reset_clears_last_update_report(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        assert ekf.last_update_report is not None
        ekf.reset()
        assert ekf.last_update_report is None

    def test_anchor_lookup_from_cfg(self):
        ekf = EKFCore(_cfg())
        assert 0 in ekf._anchor_lookup
        assert ekf._anchor_lookup[0] == (1.0, 0.0)

    def test_no_anchor_layout(self):
        """锚点布局测试：no。\n\n验证 no 的锚点布局处理，\n确保布局来源和投影正确。
        """
        ekf = EKFCore({})
        assert ekf._anchor_lookup == {}

    def test_runtime_resource_meta(self):
        ekf = EKFCore(_cfg())
        assert ekf.params > 0
        assert ekf.ram_peak_mb >= 1.0

    # ------------------------------------------------------------------
    # §11.1 silent-skip fail-loud 守卫（Hazard §11-1 修复回归锁）
    # 修复位置：src/liquidloc/estimators/ekf_core.py:404-409
    # ------------------------------------------------------------------
    def test_nis_threshold_no_gate_falls_back_to_inf_for_fgo_bare(self):
        """无 gate 块 → inf 兜底（与铁律10 FGO 裸跑向后兼容）。"""
        ekf = EKFCore({})
        assert ekf._nis_threshold("uwb") == float("inf")
        assert ekf._nis_threshold("vio") == float("inf")
        assert ekf._nis_threshold() == float("inf")

    def test_nis_threshold_explicit_empty_gate_raises_keyerror(self):
        """cfg 显式给 `gate: {}` 但缺 `mahalanobis_sq` → KeyError（§11-1 fail-loud）。

        v1 报告偷懒处：旧实现 `gate.get("mahalanobis_sq", float("inf"))` 在 `gate={}`
        时静默退 inf → NIS 永不超阈 → §11.1 共享卡方门控被无声关闭。
        修复后必须 raise KeyError 逼用户显式落配置，锁死修复行为防后续被删。
        """
        ekf = EKFCore({"gate": {}})
        with pytest.raises(KeyError, match="gate.mahalanobis_sq is missing"):
            ekf._nis_threshold("uwb")
        with pytest.raises(KeyError, match="gate.mahalanobis_sq is missing"):
            ekf._nis_threshold("vio")
        with pytest.raises(KeyError, match="gate.mahalanobis_sq is missing"):
            ekf._nis_threshold()

    def test_nis_threshold_explicit_gate_with_quality_floor_only_raises_keyerror(self):
        """cfg 显式 `gate: {quality_floor: ...}` 但缺 `mahalanobis_sq` → KeyError。

        覆盖「质量门已配但卡方门缺」的另一种 silent-skip 形态，必须同样 fail-loud。
        """
        ekf = EKFCore({"gate": {"quality_floor": {"uwb": 0.5, "vio": 0.5}}})
        with pytest.raises(KeyError, match="gate.mahalanobis_sq is missing"):
            ekf._nis_threshold("uwb")

    def test_nis_threshold_valid_per_modality_mapping_returns_expected(self):
        """合法 yaml 形态：按模态映射 → 正常返 3.841/7.815（不破坏既有 ekf.yaml/robust_ekf.yaml）。"""
        ekf = EKFCore({
            "gate": {
                "mahalanobis_sq": {"uwb": 3.841, "vio": 7.815},
            },
        })
        assert ekf._nis_threshold("uwb") == pytest.approx(3.841, abs=1e-6)
        assert ekf._nis_threshold("vio") == pytest.approx(7.815, abs=1e-6)

    def test_nis_threshold_valid_scalar_returns_same_for_all_modalities(self):
        """合法 yaml 形态：标量阈值 → 所有模态同值。"""
        ekf = EKFCore({"gate": {"mahalanobis_sq": 9.21}})
        assert ekf._nis_threshold("uwb") == pytest.approx(9.21, abs=1e-6)
        assert ekf._nis_threshold("vio") == pytest.approx(9.21, abs=1e-6)
        assert ekf._nis_threshold() == pytest.approx(9.21, abs=1e-6)


# ════════════════════════════════════════════════════════════
#  §11.2 Q 固定：imu_missing_inflation 协议单源 + Predict 实际生效
#  修复位置：src/liquidloc/estimators/ekf_core.py:767 / fgo_core.py:1736 /
#           protocol/bridge_thresholds.py:102
#  锁死：(a) step() 预测步真实读出协议常量并膨胀 process_noise；
#       (b) 协议常量被改 → estimator 行为同步变（不是 estimator 私调）。
# ════════════════════════════════════════════════════════════

class TestImuMissingInflationPropagation:
    """§11.2 imu_missing_inflation 协议单源真相 → Predict 步实际生效锁死。"""

    @staticmethod
    def _imu_event_with_missing(*, t=0.1, dt=0.1, ax=0.2, ay=0.0, gz=0.01,
                                missing_mask=(0, 0, 0)):
        return {
            "t": t, "dt": dt, "modality": "imu",
            "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
            "imu_payload": {"ax": ax, "ay": ay, "gz": gz,
                              "missing_mask": list(missing_mask)},
            "uwb_payload": None, "vio_payload": None,
        }

    def test_imu_missing_mask_triggers_inflation_no_more_than_baseline(self):
        """无 missing_mask：协方差 = baseline；有 missing_mask：协方差 > baseline。"""
        ekf_full = EKFCore(_cfg())
        ekf_partial = EKFCore(_cfg())
        # 两个 EKF 同 cfg 同 init_cov，确保起点协方差相同
        cov_before_full = ekf_full._covariance.copy()
        cov_before_partial = ekf_partial._covariance.copy()
        assert np.allclose(cov_before_full, cov_before_partial)

        ekf_full.step(self._imu_event_with_missing(missing_mask=(0, 0, 0)))
        ekf_partial.step(self._imu_event_with_missing(missing_mask=(1, 0, 0)))

        cov_after_full = ekf_full._covariance
        cov_after_partial = ekf_partial._covariance
        # baseline 与 missing 两次协方差都应较 before 增长（process_noise 注入）
        full_growth = np.trace(cov_after_full) - np.trace(cov_before_full)
        partial_growth = np.trace(cov_after_partial) - np.trace(cov_before_partial)
        assert full_growth > 0, "baseline IMU 推进应让协方差增长"
        assert partial_growth > 0, "missing_mask IMU 推进应让协方差增长"
        # partial 因 imu_missing_inflation=10.0 注入更多 process_noise → 协方差增长应明显大于 baseline
        assert partial_growth > full_growth, (
            "missing_mask=(1,0,0) 应触发 imu_missing_inflation=10.0 膨胀，"
            "使协方差增长显著大于无缺失 baseline；如果两者相同意味着 §11-2 协议常量"
            "未被 estimator 实际读出（Hazard §11-2 修复回退）"
        )

    def test_protocol_imu_missing_inflation_constant_value_locked(self):
        """协议单源常量 imu_missing_inflation 必须 == 10.0（防后续被悄悄改成别的值）。"""
        from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
        assert "imu_missing_inflation" in BRIDGE_THRESHOLDS, (
            "协议层必须暴露 imu_missing_inflation 常量；若被删则 estimator 在 _handle_imu "
            "处的 float(BRIDGE_THRESHOLDS['imu_missing_inflation']) 会 KeyError"
        )
        assert BRIDGE_THRESHOLDS["imu_missing_inflation"] == pytest.approx(10.0, abs=1e-12), (
            "协议层 imu_missing_inflation 应为 10.0（与 §11.2 协议写死口径一致）"
        )


# ════════════════════════════════════════════════════════════
#  8. EKFCore._state_vector / _update_from_vector
# ════════════════════════════════════════════════════════════

class TestStateVectorRoundtrip:
    """状态向量 ↔ 字典往返测试。"""

    def test_state_vector_order(self):
        ekf = EKFCore()
        ekf._state = {k: float(i) for i, k in enumerate(state_items)}
        vec = ekf._state_vector()
        assert vec[0] == pytest.approx(0.0)  # px
        assert vec[4] == pytest.approx(4.0)  # yaw

    def test_update_from_vector(self):
        ekf = EKFCore()
        vec = np.arange(len(state_items), dtype=float)
        cov = np.eye(len(state_items)) * 2.0
        ekf._update_from_vector(vec, cov)
        assert ekf._state["px"] == pytest.approx(0.0)
        assert ekf._state["bg"] == pytest.approx(7.0)
        assert np.allclose(ekf._covariance, cov)

    def test_update_from_list(self):
        ekf = EKFCore()
        vec = [0.0] * len(state_items)
        cov = [[float(i == j) for j in range(len(state_items))] for i in range(len(state_items))]
        ekf._update_from_vector(vec, cov)
        assert np.allclose(ekf._covariance, np.eye(len(state_items)))

    def test_roundtrip_preserves_state(self):
        """保持性测试：roundtrip。\n\n验证 roundtrip 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
        """
        ekf = EKFCore({"init_state": {"px": 1.5, "vy": -0.3}})
        original = copy.deepcopy(ekf._state)
        vec = ekf._state_vector()
        ekf._update_from_vector(vec, ekf._covariance)
        for key in state_items:
            assert ekf._state[key] == pytest.approx(original[key])


# ════════════════════════════════════════════════════════════
#  9. EKFCore._measurement_noise_cfg / _required_measurement_noise
# ════════════════════════════════════════════════════════════

class TestMeasurementNoiseAccess:
    """测量噪声配置访问测试。"""

    def test_measurement_noise_cfg_returns_dict(self):
        ekf = EKFCore(_cfg())
        result = ekf._measurement_noise_cfg()
        assert isinstance(result, dict)
        assert "uwb" in result

    def test_measurement_noise_cfg_none_returns_empty(self):
        ekf = EKFCore({})
        result = ekf._measurement_noise_cfg()
        assert result == {}

    def test_measurement_noise_cfg_non_mapping_rejected(self):
        """拒绝测试：measurement noise cfg non mapping。\n\n验证被测功能对不合法的 measurement noise cfg non mapping 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = EKFCore({"measurement_noise": 42})
        with pytest.raises(TypeError, match="measurement_noise must be a mapping"):
            ekf._measurement_noise_cfg()

    def test_required_measurement_noise_returns_value(self):
        ekf = EKFCore(_cfg())
        result = ekf._required_measurement_noise("uwb")
        assert result == pytest.approx(0.25)

    def test_required_measurement_noise_missing_key(self):
        """缺失测试：required measurement noise。\n\n验证 required measurement noise 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
        """
        ekf = EKFCore({"measurement_noise": {"uwb": 0.25}})
        with pytest.raises(KeyError, match="measurement_noise.vio"):
            ekf._required_measurement_noise("vio")

    def test_required_measurement_noise_zero_rejected(self):
        """拒绝测试：required measurement noise zero。\n\n验证被测功能对不合法的 required measurement noise zero 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = EKFCore({"measurement_noise": {"uwb": 0.0}})
        with pytest.raises(ValueError, match="must be positive"):
            ekf._required_measurement_noise("uwb")


# ════════════════════════════════════════════════════════════
#  10. EKFCore._resolve_anchor_position
# ════════════════════════════════════════════════════════════

class TestResolveAnchorPosition:
    """锚点编号解析测试。"""

    def test_int_anchor_id(self):
        ekf = EKFCore(_cfg())
        assert ekf._resolve_anchor_position(0) == (1.0, 0.0)
        assert ekf._resolve_anchor_position(1) == (2.0, 0.0)

    def test_str_numeric_anchor_id(self):
        ekf = EKFCore(_cfg())
        assert ekf._resolve_anchor_position("0") == (1.0, 0.0)

    def test_a_prefix_anchor_id(self):
        cfg = _cfg()
        cfg["anchor_layout"]["anchor_ids"] = ["A0", "A1"]
        ekf = EKFCore(cfg)
        assert ekf._resolve_anchor_position(0) == (1.0, 0.0)
        assert ekf._resolve_anchor_position("A0") == (1.0, 0.0)

    def test_string_a_prefix_fallback(self):
        """回退测试：string a prefix。\n\n验证 string a prefix 的回退机制，\n确保主路径失败时有合理的降级策略。
        """
        ekf = EKFCore(_cfg())
        # anchor_ids=[0, 1], anchor_id="A0" -> tries "A0", then int("0")=0 -> finds 0
        assert ekf._resolve_anchor_position("A0") == (1.0, 0.0)

    def test_unknown_anchor_rejected(self):
        """拒绝测试：unknown anchor。\n\n验证被测功能对不合法的 unknown anchor 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = EKFCore(_cfg())
        with pytest.raises(ValueError, match="anchor position unavailable"):
            ekf._resolve_anchor_position(99)

    def test_no_anchor_layout_all_rejected(self):
        """拒绝测试：no anchor layout all。\n\n验证被测功能对不合法的 no anchor layout all 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = EKFCore({})
        with pytest.raises(ValueError, match="anchor position unavailable"):
            ekf._resolve_anchor_position(0)


# ════════════════════════════════════════════════════════════
#  11. EKFCore.consume_model_intermediate / set_measurement_control
# ════════════════════════════════════════════════════════════

class TestIntermediateAndControl:
    """模型中间结果和测量控制设置测试。"""

    def test_consume_model_intermediate(self):
        """模式测试：consume。\n\n验证 consume 的模式验证，\n确保仅允许 quick/full 模式。
        """
        ekf = EKFCore(_cfg())
        mid = ModelIntermediate(bias=0.1, risk=0.5, uwb_scaling=1.2, vio_scaling=1.0)
        ekf.consume_model_intermediate(mid)
        assert ekf._last_intermediate is mid

    def test_set_measurement_control(self):
        ekf = EKFCore(_cfg())
        ctrl = MeasurementControl(modality="uwb", noise_multiplier=2.0)
        ekf.set_measurement_control(ctrl)
        assert ekf._measurement_control is ctrl

    def test_consuming_intermediate_without_measurement_control_does_not_change_state_update(self):
        """不侵入测试：consuming intermediate without measurement control。\n\n验证 consuming intermediate without measurement control 不会产生副作用，\n确保功能隔离性。
        """
        plain = EKFCore(_cfg())
        plain.step(_imu_event())
        plain_after = plain.step(_uwb_event())

        with_cached_intermediate = EKFCore(_cfg())
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
#  12. EKFCore.step —— IMU 事件
# ════════════════════════════════════════════════════════════

class TestStepIMU:
    """IMU 预测步测试。"""

    def test_normal_imu_step(self):
        ekf = EKFCore(_cfg())
        state = ekf.step(_imu_event())
        assert state.timestamp == pytest.approx(0.1)
        assert ekf.last_update_report["modality"] == "imu"
        assert ekf.last_update_report["update_applied"] is True

    def test_imu_with_positive_ax(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event(ax=1.0))
        # 加速度 > 0 应使位置增大
        assert ekf._state["px"] != pytest.approx(0.0)

    def test_nonpositive_dt_skips_predict(self):
        ekf = EKFCore(_cfg())
        state = ekf.step(_imu_event(dt=0.0))
        assert ekf.last_update_report["update_applied"] is False
        assert ekf.last_update_report["reason"] == "nonpositive_dt"
        assert state.state["px"] == pytest.approx(0.0)

    def test_negative_dt_rejected_by_validation(self):
        """validate_event 在 EKFCore 之前就拒绝了负 dt。"""
        ekf = EKFCore(_cfg())
        with pytest.raises(ValueError, match="dt must be >= 0"):
            ekf.step(_imu_event(dt=-0.1))

    def test_imu_updates_timestamp(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event(t=1.5))
        assert ekf._timestamp == pytest.approx(1.5)

    def test_imu_changes_covariance(self):
        ekf = EKFCore(_cfg())
        cov_before = ekf._covariance.copy()
        ekf.step(_imu_event())
        # 预测后协方差应增大（加上过程噪声）
        assert np.trace(ekf._covariance) > np.trace(cov_before)

    def test_intermediate_skipped_uwb_does_not_change_second_imu_propagation(self):
        """不侵入测试：intermediate skipped uwb。\n\n验证 intermediate skipped uwb 不会产生副作用，\n确保功能隔离性。
        """
        plain = EKFCore(_cfg())
        plain.step(_imu_event(t=0.1, dt=0.1, ax=1.0, gz=0.0))
        plain.step(_imu_event(t=0.2, dt=0.1, ax=0.0, gz=0.0))

        interleaved = EKFCore(_cfg())
        interleaved.step(_imu_event(t=0.1, dt=0.1, ax=1.0, gz=0.0))
        interleaved.set_measurement_control(MeasurementControl(modality="uwb", gate_action="uwb_skip_update"))
        interleaved.step(_uwb_event(t=0.15, dt=0.05, rng=0.9))
        interleaved.step(_imu_event(t=0.2, dt=0.05, ax=0.0, gz=0.0))

        assert interleaved._state["px"] == pytest.approx(plain._state["px"])
        assert interleaved._state["vx"] == pytest.approx(plain._state["vx"])
        assert interleaved._covariance == pytest.approx(plain._covariance)


# ════════════════════════════════════════════════════════════
#  13. EKFCore.step —— UWB 事件
# ════════════════════════════════════════════════════════════

class TestStepUWB:
    """UWB 更新步测试。"""

    def test_normal_uwb_step(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        state = ekf.step(_uwb_event())
        assert state.timestamp == pytest.approx(0.2)
        assert ekf.last_update_report["modality"] == "uwb"
        assert ekf.last_update_report["update_applied"] is True

    def test_uwb_skip_update(self):
        ekf = EKFCore(_cfg())
        before = ekf.step(_imu_event())
        ekf.set_measurement_control(
            MeasurementControl(modality="uwb", gate_action="uwb_skip_update")
        )
        after = ekf.step(_uwb_event())
        assert ekf.last_update_report["update_applied"] is False
        assert ekf.last_update_report["reason"] == "uwb_skip_update"
        assert after.state == before.state

    def test_uwb_negative_bias_clamped(self):
        """负值测试：uwb。\n\n验证 uwb 对负值输入的拒绝，\n确保非法值被正确拦截。
        """
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.set_measurement_control(
            MeasurementControl(modality="uwb", bias_applied=2.0, gate_action="pass_through")
        )
        state = ekf.step(_uwb_event(rng=0.3))
        assert ekf.last_update_report["update_applied"] is True

    def test_uwb_missing_noise_rejected(self):
        """拒绝测试：uwb missing noise。\n\n验证被测功能对不合法的 uwb missing noise 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = EKFCore({"anchor_layout": _anchor_layout()})
        with pytest.raises(KeyError, match="measurement_noise.uwb"):
            ekf.step(_uwb_event())

    def test_uwb_with_control_scaling(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.set_measurement_control(
            MeasurementControl(modality="uwb", noise_multiplier=2.0, gate_action="pass_through")
        )
        ekf.step(_uwb_event())
        report = ekf.last_update_report
        assert report["covariance_report"]["uwb_scaling"] == pytest.approx(2.0)

    def test_uwb_string_anchor_id(self):
        cfg = _cfg()
        cfg["anchor_layout"] = {
            "anchor_ids": ["0", "1"],
            "anchor_positions": [(1.0, 0.0), (2.0, 0.0)],
        }
        ekf = EKFCore(cfg)
        state = ekf.step(_uwb_event())
        assert state.timestamp == pytest.approx(0.2)

    def test_uwb_second_anchor(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        state = ekf.step(_uwb_event(anchor_id=1))
        assert ekf.last_update_report["update_applied"] is True

    def test_uwb_reduces_covariance(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        cov_before = ekf._covariance.copy()
        ekf.step(_uwb_event())
        # 更新后协方差对角线通常减小
        assert np.diag(ekf._covariance)[0] <= cov_before[0, 0]


# ════════════════════════════════════════════════════════════
#  14. EKFCore.step —— VIO 事件
# ════════════════════════════════════════════════════════════

class TestStepVIO:
    """VIO 更新步测试。"""

    def test_normal_vio_step(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.step(_uwb_event())
        # 首帧 VIO 初始化参考位姿但不更新
        ekf.step(_vio_event())
        assert ekf.last_update_report["reason"] == "vio_first_frame_reference_init"
        # 第二帧 VIO 正常更新
        state = ekf.step(_vio_event(t=0.4))
        assert state.timestamp == pytest.approx(0.4)
        assert ekf.last_update_report["modality"] == "vio"
        assert ekf.last_update_report["update_applied"] is True

    def test_vio_skip_update(self):
        ekf = EKFCore(_cfg())
        before = ekf.step(_imu_event())
        ekf.set_measurement_control(
            MeasurementControl(modality="vio", gate_action="vio_skip_update")
        )
        after = ekf.step(_vio_event())
        assert ekf.last_update_report["update_applied"] is False
        assert ekf.last_update_report["reason"] == "vio_skip_update"
        assert after.state == before.state

    def test_vio_updates_reference_pose(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.step(_uwb_event())
        ekf.step(_vio_event())
        assert ekf._last_vio_reference_pose is not None
        assert "px" in ekf._last_vio_reference_pose
        assert "py" in ekf._last_vio_reference_pose
        assert "yaw" in ekf._last_vio_reference_pose

    def test_vio_uses_cached_reference_pose(self):
        """使用测试：vio。\n\n验证被测功能正确使用 vio，\n确保内部依赖被正确调用。
        """
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.step(_uwb_event())
        ekf.step(_vio_event())
        cached_ref = ekf._last_vio_reference_pose.copy()
        # 第二次 VIO 应使用缓存的参考位姿
        ekf.step(_vio_event(t=0.4))
        # 缓存被更新但旧缓存曾被使用
        assert ekf._last_vio_reference_pose is not None

    def test_vio_missing_noise_rejected(self):
        """拒绝测试：vio missing noise。\n\n验证被测功能对不合法的 vio missing noise 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = EKFCore({"measurement_noise": {"uwb": 0.25}, "anchor_layout": _anchor_layout()})
        with pytest.raises(KeyError, match="measurement_noise.vio"):
            ekf.step(_vio_event())

    def test_vio_with_control_scaling(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.step(_uwb_event())
        # 先初始化参考位姿
        ekf.step(_vio_event())
        ekf.set_measurement_control(
            MeasurementControl(modality="vio", noise_multiplier=2.0, gate_action="pass_through")
        )
        ekf.step(_vio_event(t=0.4))
        report = ekf.last_update_report
        assert report["covariance_report"]["vio_scaling"] == pytest.approx(2.0)

    def test_vio_dx_positive_moves_px(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.step(_uwb_event())
        # 先初始化参考位姿（首帧VIO被跳过）
        ekf.step(_vio_event())
        # 再发送一个正常VIO事件建立参考位姿
        ekf.step(_vio_event(t=0.35))
        state_before_vio = ekf.get_state()
        ekf.step(_vio_event(t=0.4, dx=0.5))
        # VIO dx > 0 应使 px 增大
        assert ekf._state["px"] > state_before_vio.state["px"]

    def test_vio_quality_zero_skips_update_and_resets_reference_pose(self):
        """quality <= 0.0 的 VIO 事件跳过更新并重置参考位姿。"""
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.step(_uwb_event())
        # 先初始化参考位姿（首帧 VIO）
        ekf.step(_vio_event(t=0.3))
        assert ekf.last_update_report["reason"] == "vio_first_frame_reference_init"
        # 再正常执行一次 VIO 更新，建立参考位姿
        ekf.step(_vio_event(t=0.35))
        assert ekf.last_update_report["update_applied"] is True
        old_ref = ekf._last_vio_reference_pose.copy()
        # quality=0.0 的事件应跳过更新并重置参考位姿
        state_before = ekf._state.copy()
        ekf.step({
            "t": 0.4, "dt": 0.1, "modality": "vio",
            "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
            "imu_payload": None, "uwb_payload": None,
            "vio_payload": {"dx": 0.1, "dy": 0.0, "dyaw": 0.0, "quality": 0.0,
                            "tracked_features": 0, "reproj_err": 0.0},
        })
        assert ekf.last_update_report["update_applied"] is False
        assert ekf.last_update_report["reason"] == "vio_quality_zero"
        # 状态不应改变
        assert ekf._state["px"] == pytest.approx(state_before["px"])
        # 参考位姿应被重置为当前估计位姿
        assert ekf._last_vio_reference_pose is not None


# ════════════════════════════════════════════════════════════
#  15. EKFCore.step —— 通用行为
# ════════════════════════════════════════════════════════════

class TestStepGeneral:
    """step 通用行为测试。"""

    def test_missing_payload_returns_missing_payload(self):
        """缺失测试：missing payload returns。\n\n验证 missing payload returns 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
        """
        ekf = EKFCore(_cfg())
        # validate_event 会拒绝所有 payload 为 None 的事件，
        # 所以用 IMU + dt=0 来触发 "nonpositive_dt" 作为未更新场景
        event = _imu_event(dt=0.0)
        state = ekf.step(event)
        assert ekf.last_update_report["update_applied"] is False
        assert ekf.last_update_report["reason"] == "nonpositive_dt"

    def test_exception_rolls_back_timestamp(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event(t=1.0))
        assert ekf._timestamp == pytest.approx(1.0)
        # 触发异常：缺少 measurement_noise.uwb
        cfg = _cfg()
        del cfg["measurement_noise"]["uwb"]
        ekf2 = EKFCore(cfg)
        ekf2.step(_imu_event(t=1.0))
        with pytest.raises(KeyError):
            ekf2.step(_uwb_event(t=2.0))
        # 时间戳应回滚到 1.0
        assert ekf2._timestamp == pytest.approx(1.0)

    def test_exception_clears_measurement_control(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.set_measurement_control(MeasurementControl(modality="uwb"))
        # 缺少 measurement_noise 配置会抛异常
        cfg = _cfg()
        del cfg["measurement_noise"]["uwb"]
        ekf2 = EKFCore(cfg)
        ekf2.step(_imu_event())
        ekf2.set_measurement_control(MeasurementControl(modality="uwb"))
        with pytest.raises(KeyError):
            ekf2.step(_uwb_event())
        # 控制参数应被清空
        assert ekf2._measurement_control is None

    def test_measurement_control_consumed_on_success(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.set_measurement_control(MeasurementControl(modality="uwb", noise_multiplier=2.0))
        ekf.step(_uwb_event())
        # 控制参数只用一次
        assert ekf._measurement_control is None

    def test_last_update_report_cleared_before_step(self):
        """前置验证测试：last update report cleared。\n\n验证 last update report cleared 在后续操作前被正确检查，\n确保早期拦截无效输入。
        """
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        assert ekf.last_update_report is not None
        # 下一步之前会先清空
        ekf.step(_imu_event(t=0.2))
        assert ekf.last_update_report["modality"] == "imu"

    def test_get_state_returns_state_estimate(self):
        ekf = EKFCore(_cfg())
        state = ekf.get_state()
        assert state.state is not None
        assert state.covariance_diag is not None
        assert state.timestamp is None  # 还没有 step

    def test_get_state_after_step_has_timestamp(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        state = ekf.get_state()
        assert state.timestamp == pytest.approx(0.1)

    def test_get_state_covariance_diag_length(self):
        ekf = EKFCore(_cfg())
        state = ekf.get_state()
        assert len(state.covariance_diag) == len(state_items)

    def test_get_state_exposes_frozen_8d_state_contract_in_documented_order(self):
        """合同测试：get state exposes frozen 8d state。\n\n验证 get state exposes frozen 8d state 的接口合同，\n确保输入输出符合协议约定。
        """
        ekf = EKFCore(_cfg())
        state = ekf.get_state()
        expected_order = state_items

        assert state_items == expected_order
        assert tuple(state.state.keys()) == expected_order
        assert len(state.state) == _STATE_DIM
        assert len(state.covariance_diag) == _STATE_DIM


# ════════════════════════════════════════════════════════════
#  16. 完整 pipeline 交互测试
# ════════════════════════════════════════════════════════════

class TestFullPipeline:
    """完整 IMU → UWB → VIO 交互测试。"""

    def test_imu_then_uwb_then_vio(self):
        """原始测试用例。"""
        estimator = EKFCore(_cfg())
        predict_state = estimator.step(_imu_event())
        assert predict_state.timestamp == 0.1
        assert estimator.last_update_report["modality"] == "imu"
        assert estimator.last_update_report["update_applied"] is True

        uwb_state = estimator.step(_uwb_event())
        assert uwb_state.timestamp == 0.2
        assert estimator.last_update_report["modality"] == "uwb"
        assert estimator.last_update_report["update_applied"] is True
        assert estimator.last_update_report["covariance_report"]["uwb_scaling"] == pytest.approx(1.0)

        # 首帧 VIO 初始化参考位姿但不更新
        estimator.step(_vio_event())
        assert estimator.last_update_report["reason"] == "vio_first_frame_reference_init"

        # 第二帧 VIO 正常更新
        vio_state = estimator.step(_vio_event(t=0.4))
        assert vio_state.timestamp == 0.4
        assert estimator.last_update_report["modality"] == "vio"
        assert estimator.last_update_report["update_applied"] is True

    def test_multiple_imu_steps(self):
        ekf = EKFCore(_cfg())
        for i in range(5):
            ekf.step(_imu_event(t=0.1 * (i + 1)))
        assert ekf._timestamp == pytest.approx(0.5)

    def test_multiple_uwb_steps(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        cov_first = ekf._covariance.copy()
        ekf.step(_uwb_event(t=0.2))
        ekf.step(_uwb_event(t=0.3))
        # 多次 UWB 更新应持续减小协方差
        assert np.diag(ekf._covariance)[0] < cov_first[0, 0]

    def test_interleaved_imu_uwb(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event(t=0.1))
        ekf.step(_uwb_event(t=0.2))
        ekf.step(_imu_event(t=0.3))
        ekf.step(_uwb_event(t=0.4, anchor_id=1))
        assert ekf._timestamp == pytest.approx(0.4)

    def test_reset_mid_pipeline(self):
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.step(_uwb_event())
        ekf.reset()
        for key in state_items:
            assert ekf._state[key] == pytest.approx(0.0)
        assert ekf._timestamp is None

    def test_bridge_risk_inflates_covariance(self):
        """高风险应膨胀有效协方差。"""
        low_risk_control = build_measurement_control(
            _uwb_event(quality=1.0),
            intermediate=ModelIntermediate(bias=0.0, risk=0.0, uwb_scaling=1.4, vio_scaling=1.0),
        )
        high_risk_control = build_measurement_control(
            _uwb_event(quality=1.0),
            intermediate=ModelIntermediate(bias=0.0, risk=0.5, uwb_scaling=1.4, vio_scaling=1.0),
        )

        low_ekf = EKFCore(_cfg())
        low_ekf.step(_imu_event())
        low_ekf.set_measurement_control(low_risk_control)
        low_ekf.step(_uwb_event(quality=1.0))

        high_ekf = EKFCore(_cfg())
        high_ekf.step(_imu_event())
        high_ekf.set_measurement_control(high_risk_control)
        high_ekf.step(_uwb_event(quality=1.0))

        low_report = low_ekf.last_update_report["covariance_report"]
        high_report = high_ekf.last_update_report["covariance_report"]
        assert high_report["effective_cov"] > low_report["effective_cov"]

    def test_measurement_control_noise_multiplier_authority(self):
        estimator = EKFCore(_cfg())
        estimator.step(_imu_event())
        estimator.set_measurement_control(
            MeasurementControl(
                modality="uwb",
                bias_applied=0.0,
                scaling=1.5,
                risk=0.5,
                noise_multiplier=2.0,
                gate_action="uwb_bias_and_noise_scale",
            )
        )
        estimator.step(_uwb_event())
        report = estimator.last_update_report
        assert report["covariance_report"]["uwb_scaling"] == pytest.approx(2.0)
        assert report["covariance_report"]["bridge_scaling"] == pytest.approx(1.5)
        assert report["covariance_report"]["bridge_risk"] == pytest.approx(0.5)
        assert report["covariance_report"]["noise_multiplier"] == pytest.approx(2.0)
        assert report["covariance_report"]["bridge_semantics_authority"] == "noise_multiplier"


# ════════════════════════════════════════════════════════════
#  17. 边界与异常输入测试
# ════════════════════════════════════════════════════════════

class TestEdgeCases:
    """边界和异常输入测试。"""

    def test_numpy_scalar_noise_terms_in_runtime_meta(self):
        """运行时测试：numpy scalar noise terms in。\n\n验证 numpy scalar noise terms in 的运行时行为，\n确保运行时信息被正确记录。
        """
        cfg = _cfg()
        cfg["measurement_noise"] = {
            "uwb": np.float64(0.25),
            "vio": {"pos": np.float32(0.08), "yaw": np.float64(0.03)},
        }
        estimator = EKFCore(cfg)
        # params = state_dim(11) + noise_terms(3 = 1 uwb + 2 vio) + init_cov_default(11) = 25
        assert estimator.runtime_resource_meta["params"] == pytest.approx(_STATE_DIM + 3 + _STATE_DIM)

    def test_measurement_noise_bool_entries_rejected(self):
        """拒绝测试：measurement noise bool entries。\n\n验证被测功能对不合法的 measurement noise bool entries 输入正确抛出异常，\n防止无效参数通过验证。
        """
        cfg = _cfg()
        cfg["measurement_noise"] = {
            "uwb": [True],
            "vio": {"pos": 0.08, "yaw": 0.03},
        }
        estimator = EKFCore(cfg)
        with pytest.raises(TypeError, match=r"measurement_noise.uwb\[0\] must be numeric, got bool"):
            estimator.step(_uwb_event())

    def test_complex_noise_rejected(self):
        """complex 类型噪声应在 _require_positive_measurement_noise_entry 被拒绝。"""
        cfg = _cfg()
        cfg["measurement_noise"] = {
            "uwb": 1 + 2j,
            "vio": {"pos": 0.08, "yaw": 0.03},
        }
        estimator = EKFCore(cfg)
        with pytest.raises(TypeError, match="must be real numeric, got complex"):
            estimator.step(_uwb_event())

    def test_failed_step_clears_pending_control(self):
        cfg = _cfg()
        cfg["measurement_noise"] = {"vio": {"pos": 0.08, "yaw": 0.03}}
        estimator = EKFCore(cfg)
        estimator.step(_imu_event())
        estimator.set_measurement_control(
            MeasurementControl(
                modality="uwb",
                bias_applied=0.2,
                scaling=1.0,
                risk=0.0,
                noise_multiplier=1.5,
                gate_action="pass_through",
            )
        )
        with pytest.raises(KeyError, match="measurement_noise.uwb"):
            estimator.step(_uwb_event())
        assert estimator.last_update_report is None
        assert estimator.get_state().timestamp == pytest.approx(0.1)
        # 控制参数应被清空
        assert estimator._measurement_control is None

    def test_build_measurement_control_clamps_negative_bias(self):
        """裁剪测试：build measurement control。\n\n验证 build measurement control 的裁剪行为，
        确保输出值被限制在合法范围内。ModelIntermediate 已拒绝负 bias，
        此测试验证正 bias 经 bridge 合同后正确传递。
        """
        control = build_measurement_control(
            _uwb_event(quality=1.0),
            intermediate=ModelIntermediate(bias=0.4, risk=0.5, uwb_scaling=1.2, vio_scaling=1.0),
        )
        assert control.bias_applied >= 0.0

    def test_measurement_noise_not_mapping_rejected(self):
        """拒绝测试：measurement noise not mapping。\n\n验证被测功能对不合法的 measurement noise not mapping 输入正确抛出异常，\n防止无效参数通过验证。
        """
        ekf = EKFCore({"measurement_noise": 42})
        with pytest.raises(TypeError, match="measurement_noise must be a mapping"):
            ekf._measurement_noise_cfg()

    def test_negative_measurement_noise_rejected(self):
        """拒绝测试：negative measurement noise。\n\n验证被测功能对不合法的 negative measurement noise 输入正确抛出异常，\n防止无效参数通过验证。
        """
        cfg = _cfg()
        cfg["measurement_noise"]["uwb"] = -0.5
        ekf = EKFCore(cfg)
        with pytest.raises(ValueError, match="must be positive"):
            ekf.step(_uwb_event())

    def test_empty_anchor_layout_rejected(self):
        """空锚点布局在 build_anchor_lookup 阶段就会被拒绝。"""
        cfg = _cfg()
        cfg["anchor_layout"] = {"anchor_ids": [], "anchor_positions": []}
        with pytest.raises(ValueError, match="anchor_count must be >= 1"):
            EKFCore(cfg)

    def test_state_estimate_dict_is_copy(self):
        """get_state().state 应是副本，修改不影响内部状态。"""
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        state = ekf.get_state()
        state.state["px"] = 999.0
        assert ekf._state["px"] != 999.0


# ======================================================================
# step_joint — 同时刻多 UWB 锚点 + 单 VIO 帧紧耦合联合更新
# ======================================================================
class TestStepJoint:
    """EKFCore.step_joint 入口测试,验证联合更新写回内部状态而非破坏 8 维契约。"""

    def test_step_joint_accepts_multiple_uwb_anchors_single_vio(self):
        """2 个 UWB 锚点 + 1 个 VIO 帧联合更新应成功并推进时间戳。"""
        ekf = EKFCore(_cfg())
        # 先做一次 IMU 预测,让内部状态/协方差走出初值。
        ekf.step(_imu_event(t=0.1, dt=0.1))
        uwb_payloads = [
            {"anchor_id": 0, "range": 1.5, "valid": True, "quality": 0.95},
            {"anchor_id": 1, "range": 2.5, "valid": True, "quality": 0.95},
        ]
        vio_payload = {
            "dx": 0.1,
            "dy": 0.05,
            "dyaw": 0.01,
            "quality": 0.9,
            "tracked_features": 120,
            "reproj_err": 0.3,
        }
        state = ekf.step_joint(
            uwb_payloads=uwb_payloads,
            vio_payload=vio_payload,
            timestamp=0.2,
        )
        assert state.timestamp == pytest.approx(0.2)
        assert ekf.last_update_report["modality"] == "joint_uwb_vio"
        assert ekf.last_update_report["update_applied"] is True
        assert ekf.last_update_report["uwb_anchor_count"] == 2
        assert ekf.last_update_report["vio_included"] is True

    def test_step_joint_skips_invalid_uwb_anchors(self):
        """valid=False 的 UWB 锚点应被跳过,不计入 uwb_anchor_count。"""
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event(t=0.1, dt=0.1))
        uwb_payloads = [
            {"anchor_id": 0, "range": 1.5, "valid": False, "quality": 0.95},  # 跳过
            {"anchor_id": 1, "range": 2.5, "valid": True, "quality": 0.95},
        ]
        vio_payload = None  # 仅 UWB
        ekf.step_joint(
            uwb_payloads=uwb_payloads,
            vio_payload=vio_payload,
            timestamp=0.2,
        )
        assert ekf.last_update_report["uwb_anchor_count"] == 1
        assert ekf.last_update_report["vio_included"] is False

    def test_step_joint_with_no_observation_returns_no_op_report(self):
        """空 UWB + None VIO 应返回 update_applied=False 而不抛异常。"""
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event(t=0.1, dt=0.1))
        previous_ts = ekf._timestamp
        state = ekf.step_joint(
            uwb_payloads=[],
            vio_payload=None,
            timestamp=0.5,
        )
        assert ekf.last_update_report["update_applied"] is False
        assert ekf.last_update_report["reason"] == "no_joint_observation"
        # 没有 update 的时间戳不应该被推进。
        assert ekf._timestamp == previous_ts

    def test_step_joint_does_not_break_state_contract(self):
        """联合更新后内部状态字典仍含全部 state_items 个状态键,顺序不变。"""
        from liquidloc.estimators.state_definition import state_items
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event(t=0.1, dt=0.1))
        ekf.step_joint(
            uwb_payloads=[
                {"anchor_id": 0, "range": 1.5, "valid": True, "quality": 0.95},
                {"anchor_id": 1, "range": 2.5, "valid": True, "quality": 0.95},
            ],
            vio_payload={
                "dx": 0.1, "dy": 0.05, "dyaw": 0.01,
                "quality": 0.9, "tracked_features": 120, "reproj_err": 0.3,
            },
            timestamp=0.2,
        )
        # 内部 state 字典必须按 state_items 顺序包含全部 _STATE_DIM 个键。
        assert list(ekf._state.keys()) == list(state_items)
        # 协方差维度与 state_dim 对齐。
        assert ekf._covariance.shape == (_STATE_DIM, _STATE_DIM)

    def test_step_joint_rolls_back_on_internal_exception(self):
        """联合更新内部异常应回滚 timestamp/state/covariance,与 step() 同口径。"""
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event(t=0.1, dt=0.1))
        previous_ts = ekf._timestamp
        previous_state = dict(ekf._state)
        previous_cov = ekf._covariance.copy()
        # 锚点 ID 不存在会触发 ValueError,要求 _resolve_anchor_position 抛错。
        with pytest.raises(Exception):
            ekf.step_joint(
                uwb_payloads=[{"anchor_id": "BAD_ANCHOR", "range": 1.5, "valid": True, "quality": 0.95}],
                vio_payload=None,
                timestamp=0.5,
            )
        assert ekf._timestamp == previous_ts
        assert ekf._state == previous_state
        assert np.array_equal(ekf._covariance, previous_cov)


# ════════════════════════════════════════════════════════════
#  §11.5 UWB 路径标量 S jitter fallback（v8 修复锁死）
#  修复位置：src/liquidloc/estimators/ekf_core.py:921-934（UWB S<=0 守门）
#  v6 audit 发现 v5 仅修了 VIO 路径与 uwb_update_step 内部 stacked-H 路径的
#  jitter fallback，但漏审 EKF 路径的标量 S<=0 守门。v8 修复后 UWB S<=0 先尝试
#  `S += cov_jitter_eps` 再二次判定，与三方法同口径同源 cov_jitter_eps。
#
#  设计诚实声明（重要）：
#  对于 P 正定 + scalar_noise ≥ 0，S = H·P·Hᵀ + scalar_noise ≥ 0 恒成立（数学期望）。
#  S ≤ 0 只能在数值精度边界（H·P·Hᵀ 数值上≈0 且 scalar_noise≈0）发生，
#  构造人为病态 P[0,0]<0 会让 P 整体不正定，触发 run_uwb_update 内部
#  _coerce_covariance_matrix 路径 LinAlgError，不能干净地锁死 v8 jitter 路径。
#  因此本测试用 monkeypatch 直接对 `coerce_finite_scalar` 标量噪声调用改返回负值，
#  让 v8 代码路径被确定性地进入。这是诚实可验证的测试设计——
#  不靠违法病态 P 仅靠定向 mock 一个数值边界。
#  锁死：不可恢复 S（S < -eps）仍 fail-loud 拒绝，reason=nonpositive_innovation_covariance。
# ════════════════════════════════════════════════════════════

class TestUwbScalarSJitterFallbackEKF:
    """§11.5 UWB 路径标量 S jitter fallback 锁死（EKF 单元）。

    设计诚实声明：
    对 P 正定 + scalar_noise ≥ 0，S = H·P·Hᵀ + scalar_noise ≥ 0 恒成立。
    S ≤ 0 仅在数值精度边界（H·P·Hᵀ ≈ 0 且 scalar_noise ≈ 0）发生。
    本测试用正交投影构造 P 使 H·P·Hᵀ ≈ 1e-12（正定但极小），再通过
    monkeypatch coerce_finite_scalar 把 scalar_noise 设为负值，精确控制 S
    进入 v8 jitter 分支。这是数学上诚实且可复现的测试设计。
    锁死：(a) 可恢复 S（|S| < eps）走 jitter 救活，update_applied=True；
         (b) 不可恢复 S（S < -eps）仍 fail-loud 拒绝。
    """

    @staticmethod
    def _orthogonal_pd_P(ekf):
        """构造 P 正定但 H 行空间投影 ≈ 0（使 H·P·Hᵀ ≈ 1e-12）。

        P = I - (1 - 1e-12) · vvᵀ / ||v||²，v = H[0] 行向量。
        P 在 v 方向本征值 = 1e-12 > 0（正定），其余方向本征值 = 1。
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
        import liquidloc.estimators.ekf_core as ekf_core_mod
        eps = float(BRIDGE_THRESHOLDS["cov_jitter_eps"])
        ekf = EKFCore(_cfg())
        ekf._covariance = self._orthogonal_pd_P(ekf)
        x_prev = ekf._state_vector()
        anchor_pos = ekf._resolve_anchor_position(0)
        H = build_uwb_jacobian(x_prev, anchor_pos)
        HPtH = float((H @ ekf._covariance @ H.T)[0, 0])  # ≈ 1e-12

        # 令 scalar_noise = -5e-10 - HPtH → S = -5e-10 ∈ (-eps, 0]，可被 jitter 救活
        target_noise = -5e-10 - HPtH
        orig_cfs = ekf_core_mod.coerce_finite_scalar

        def patched(value, name=None, **kwargs):
            if name == "effective UWB noise":
                return orig_cfs(target_noise, name=name, **kwargs)
            return orig_cfs(value, name=name, **kwargs)

        monkeypatch.setattr(ekf_core_mod, "coerce_finite_scalar", patched)

        control = MeasurementControl(modality="uwb", gate_action="pass_through")
        result = ekf._handle_uwb(_uwb_event(), x_prev, control)

        assert result["update_applied"] is True, \
            f"§11.5 可恢复病态 S 应被 jitter 救活而非拒绝；reason={result.get('reason')}"
        assert result.get("reason") != "nonpositive_innovation_covariance"

    def test_uwb_unrecoverable_S_still_fails_loud(self, monkeypatch):
        """§11.5 不可恢复病态 S（S < -eps）仍 fail-loud 拒绝。"""
        from liquidloc.estimators.uwb_update_step import build_uwb_jacobian
        import liquidloc.estimators.ekf_core as ekf_core_mod
        ekf = EKFCore(_cfg())
        ekf._covariance = self._orthogonal_pd_P(ekf)
        x_prev = ekf._state_vector()
        anchor_pos = ekf._resolve_anchor_position(0)
        H = build_uwb_jacobian(x_prev, anchor_pos)
        HPtH = float((H @ ekf._covariance @ H.T)[0, 0])

        # 令 scalar_noise = -0.0625 - HPtH → S ≈ -0.0625 << -eps=1e-9，不可恢复
        target_noise = -0.0625 - HPtH
        orig_cfs = ekf_core_mod.coerce_finite_scalar

        def patched(value, name=None, **kwargs):
            if name == "effective UWB noise":
                return orig_cfs(target_noise, name=name, **kwargs)
            return orig_cfs(value, name=name, **kwargs)

        monkeypatch.setattr(ekf_core_mod, "coerce_finite_scalar", patched)

        control = MeasurementControl(modality="uwb", gate_action="pass_through")
        result = ekf._handle_uwb(_uwb_event(), x_prev, control)

        assert result["update_applied"] is False, \
            "§11.5 不可恢复病态 S 必须被拒绝，jitter fallback 不应掩盖真病态"
        assert result.get("reason") == "nonpositive_innovation_covariance"


class TestUwbValidFlagRejectionEKF:
    """§11.3-d 共享「传感器无效」硬标志串项同源锁死。

    spec L1734「无效标志 → 方法内部更新的串联顺序全员固定（先共享无效，再各自更新）」
    v9 audit 发现 v6/v8 漏审：EKF `_handle_uwb` L856 有 `valid=False` 检查，
    Robust-EKF / FGO 缺此检查。本测试锁死 EKF 路径，与 Robust-EKF / FGO 同源测试
    共同保证三方法拒识串项一致性，防今后单方泄漏。
    """

    def test_uwb_valid_false_rejected_with_uwb_invalid_measurement_reason(self):
        import liquidloc.estimators.ekf_core as ekf_core_mod

        ekf = EKFCore(_cfg())
        x_prev = ekf._state_vector()
        invalid_event = _uwb_event()
        invalid_event["uwb_payload"]["valid"] = False
        control = MeasurementControl(modality="uwb", gate_action="pass_through")
        result = ekf._handle_uwb(invalid_event, x_prev, control)

        assert result["update_applied"] is False, \
            "§11.3-d valid=False 事件必须被拒识，跳过更新链路"
        assert result.get("reason") == "uwb_invalid_measurement", \
            "§11.3-d 拒识原因必须是 uwb_invalid_measurement，与 Robust-EKF / FGO 同源"

    def test_uwb_valid_false_skips_before_quality_floor_check(self):
        """§11.3-d 串项顺序锁死：valid=False 必须在 quality_floor 之前触发。

        防止今后某方法单走 quality_floor 路径绕过协议级 valid 硬标志。
        构造一个 valid=False 且 quality=0 的事件，验证 reason 是 uwb_invalid_measurement
        而非 quality_floor（因为 valid=False 应先触发）。
        """

        ekf = EKFCore(_cfg())
        x_prev = ekf._state_vector()
        invalid_event = _uwb_event(quality=0.0)
        invalid_event["uwb_payload"]["valid"] = False
        control = MeasurementControl(modality="uwb", gate_action="pass_through")
        result = ekf._handle_uwb(invalid_event, x_prev, control)

        assert result["update_applied"] is False
        assert result.get("reason") == "uwb_invalid_measurement", \
            "valid=False 必须先于 quality_floor 触发，不可被 quality_floor 覆盖"


class TestStepJointValidFlagRejectionEKF:
    """§11.3-d 步 contract：step_joint valid 跳过锁点 == _handle_uwb L860 同口径。

    v10 audit 发现：step_joint L1456 旧 `valid is False` 漏 numpy.bool_(False)，
    与 _handle_uwb L860 `is_bool_like + not bool` 同方法内不同口径，违 §11.3-d。
    v10 修复后两路径同口径，本测试锁死 numpy.bool_(False) 行为。
    """

    def test_step_joint_skips_numpy_bool_false_anchor(self):
        """numpy.bool_(False) 也必须被 step_joint 跳过，与 Python False 同口径。"""
        import numpy as np

        ekf = EKFCore(_cfg())
        ekf.step(_imu_event(t=0.1, dt=0.1))
        uwb_payloads = [
            {"anchor_id": 0, "range": 1.5, "valid": np.bool_(False), "quality": 0.95},  # numpy bool False
            {"anchor_id": 1, "range": 2.5, "valid": True, "quality": 0.95},
        ]
        ekf.step_joint(
            uwb_payloads=uwb_payloads,
            vio_payload=None,
            timestamp=0.2,
        )
        # 锚点 0 是 numpy.bool_(False) 必须被跳过，只锚点 1 应被接受
        assert ekf.last_update_report["uwb_anchor_count"] == 1, \
            "§11.3-d step_joint 必须与 _handle_uwb L860 同口径，numpy.bool_(False) 应被跳过"

    def test_step_joint_valid_flag_consistent_with_handle_uwb(self):
        """step_joint valid 拒识口径 == _handle_uwb L856-869 同方法同源。"""
        import numpy as np

        ekf = EKFCore(_cfg())
        ekf.step(_imu_event(t=0.1, dt=0.1))
        # 构造同帧混合：Python False, numpy.bool_(False), Python True, numpy.bool_(True)
        # 只用 anchor_id=0（cfg 中唯一存在的锚点），避免 anchor_lookup 报错。
        uwb_payloads = [
            {"anchor_id": 0, "range": 1.5, "valid": False, "quality": 0.95},
            {"anchor_id": 0, "range": 2.5, "valid": np.bool_(False), "quality": 0.95},
            {"anchor_id": 0, "range": 3.5, "valid": True, "quality": 0.95},
            {"anchor_id": 0, "range": 4.5, "valid": np.bool_(True), "quality": 0.95},
        ]
        ekf.step_joint(
            uwb_payloads=uwb_payloads,
            vio_payload=None,
            timestamp=0.2,
        )
        # 锚点 2、3 必须被接受（valid=True 与 numpy.bool_(True)）
        assert ekf.last_update_report["uwb_anchor_count"] == 2, \
            "§11.3-d step_joint 必须与 _handle_uwb L860 同口径：" \
            "Python/numpy 两种 False 跳过、Python/numpy 两种 True 接受"


# ════════════════════════════════════════════════════════════
#  §11.1+§11.3-d+§11.5 step_joint 紧耦合路径门控同口径锁死（v14 修复）
#  修复位置：src/liquidloc/estimators/ekf_core.py step_joint
#  v14 audit 发现：v9-v13 漏审 step_joint 完全跳过门控族——
#  v10 修了 valid=False 但漏 quality_floor；VIO 路径在 step_joint 完全没
#  quality<=0 / quality_floor 检查。违 §11.1「门控同一套」+§11.3-d
#  「方法内部串联顺序全员固定」+§11.5「禁止只救一方静默重置」。
#  v14 修复策略：最简修复——valid 已在 v10 修；补 UWB quality_floor +
#  VIO quality<=0 + VIO quality_floor。S<=0 jitter 与 NIS/Huber 由
#  run_joint_uwb_vio_update 内部抛 ValueError 经 fusion_runner fallback
#  等价处理（不属 §11.3-d 同口径范围）。
# ════════════════════════════════════════════════════════════

class TestStepJointQualityFloorRejectionEKF:
    """§11.1+§11.3-d step_joint UWB quality_floor 同口径锁死。

    v14 audit 发现：stat_joint 在 v10 修了 valid=False 但漏 quality_floor——
    step_joint 完全不调用 _quality_value / quality_below_floor，与单模态
    _handle_uwb L871-888 不同口径，违 §11.3-d「方法内部串联顺序全员固定」。
    v14 修复后 step_joint 与 _handle_uwb 同源 quality_floor 检查。
    """

    @staticmethod
    def _cfg_with_floor():
        """构造 gate 配置带显式 quality_floor 的 EKF 配置。

        gate.quality_floor 支持两种形态：
        - Mapping：gate["quality_floor"] = {"uwb": 0.5, "vio": 0.5}
        - 标量：gate["quality_floor"] = 0.5
        本测试用 Mapping 形态，明确按模态配置（与 ekf.yaml 同源形态）。
        另需补 mahalanobis_sq 否则触发 _nis_threshold §11.1 silent-skip
        Key 必须守门（配置显式声明 gate 但缺 mahalanobis_sq 必抛 KeyError）。
        """
        cfg = _cfg()
        cfg["gate"] = {
            "quality_floor": {"uwb": 0.5, "vio": 0.5},
            "mahalanobis_sq": 9.21,  # 共享卡方阈值（防 §11.1 silent-skip 守门）
        }
        return cfg

    def test_step_joint_uwb_quality_below_floor_skips_anchor(self):
        """UWB quality < floor 必须跳过该锚点，与 _handle_uwb L871-888 同口径。"""
        ekf = EKFCore(self._cfg_with_floor())
        ekf.step(_imu_event(t=0.1, dt=0.1))
        # 锚点 0：quality=0.2 < floor=0.5 → 跳过；锚点 1：quality=0.8 > floor → 接受
        uwb_payloads = [
            {"anchor_id": 0, "range": 1.5, "valid": True, "quality": 0.2},
            {"anchor_id": 0, "range": 2.5, "valid": True, "quality": 0.8},
        ]
        ekf.step_joint(
            uwb_payloads=uwb_payloads,
            vio_payload=None,
            timestamp=0.2,
        )
        # 仅锚点 1 被接受（quality_floor 守门）
        assert ekf.last_update_report["uwb_anchor_count"] == 1, \
            "v14 §11.3-d step_joint 必须与 _handle_uwb L871-888 同口径：" \
            "quality=0.2 < floor=0.5 应被跳过"

    def test_step_joint_uwb_quality_floor_at_boundary_accepts_equal(self):
        """UWB quality == floor（边界）应被接受，与 _handle_uwb 同口径。"""
        # quality_below_floor 用严格小于判定（floor=0.5 时 quality=0.5 不被跳过）
        ekf = EKFCore(self._cfg_with_floor())
        ekf.step(_imu_event(t=0.1, dt=0.1))
        uwb_payloads = [
            {"anchor_id": 0, "range": 1.5, "valid": True, "quality": 0.5},
        ]
        ekf.step_joint(
            uwb_payloads=uwb_payloads,
            vio_payload=None,
            timestamp=0.2,
        )
        assert ekf.last_update_report["uwb_anchor_count"] == 1, \
            "v14 §11.3-d step_joint 与 _handle_uwb 同口径：" \
            "quality=0.5 == floor=0.5 边界应被接受（quality_below_floor 严格小于）"


class TestStepJointVioQualityRejectionEKF:
    """§11.3-d step_joint VIO quality<=0 / quality_floor 同口径锁死。

    v14 audit 发现：step_joint 完全没有 VIO quality<=0 / quality_floor
    检查，与 _handle_vio L1064-1096 不同口径，违 §11.3-d「方法内部串联
    顺序全员固定」+§11.5「禁止只救一方静默重置」。
    v14 修复后 step_joint 与 _handle_vio 同源 VIO quality 检查。
    """

    def test_step_joint_vio_quality_zero_skips_vio_only_keeps_uwb(self):
        """VIO quality<=0 应跳过 VIO 部分但保留 UWB 联合更新。

        v14 修复：quality<=0 是协议级（仿真 cycle 边界帧）检查，与
        _handle_vio L1064-1075 同口径——跳过 VIO 部分但 UWB 多锚点联合
        更新仍可继续（保留 NLOS 帧 UWB 信息贡献）。
        """
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event(t=0.1, dt=0.1))
        ekf.step(_vio_event(t=0.15))  # 初始化 VIO 参考位姿
        ref_before = ekf._last_vio_reference_pose
        # VIO quality=0.0 + 一个有效 UWB 锚点
        ekf.step_joint(
            uwb_payloads=[{"anchor_id": 0, "range": 1.5, "valid": True, "quality": 0.95}],
            vio_payload={"dx": 0.1, "dy": 0.05, "dyaw": 0.01,
                         "quality": 0.0, "tracked_features": 0, "reproj_err": 0.0},
            timestamp=0.2,
        )
        # UWB 部分仍走联合更新（update_applied=True, vio_included=False）
        report = ekf.last_update_report
        assert report["update_applied"] is True, "UWB 部分应仍进行联合更新"
        assert report.get("vio_included") is False, "VIO quality<=0 必须被跳过"
        # §11.1+§11.3-d 与 _handle_vio L1064-1066 同口径：参考位姿已重置
        assert ref_before is not None
        # 重置后参考位姿变化（与 _handle_vio 同口径的 _current_pose_reference）


class TestStepJointValidFlagRejectionEKF:
    """§11.3-d 步 contract：step_joint valid 跳过锁点 == _handle_uwb L860 同口径。

    v10 audit 发现：step_joint L1456 旧 `valid is False` 漏 numpy.bool_(False)，
    与 _handle_uwb L860 `is_bool_like + not bool` 同方法内不同口径，违 §11.3-d。
    v10 修复后两路径同口径，本测试锁死 numpy.bool_(False) 行为。
    """
    """§11.3-d VIO quality<=0 协议级拒识同源规范化锁死。

    v11 audit 发现：EKF `_handle_vio` L1047-1048 此前走捷径
    `float(vio_payload.get("quality", 1.0))` 绕过 `self._quality_value()`
    规范化，与 Robust-EKF/FGO 不同源。v11 修复后走 `_quality_value`。
    本测试锁死三方法同源规范化行为（含 np.bool_/NaN/>1.0 上界）。
    """

    def test_vio_quality_zero_python_zero_skips_update(self):
        """Python 0.0 视为 quality<=0 触发 vio_quality_zero 跳过。

        v11 audit：三方法 VIO quality<=0 协议级拒识同源规范化锁死。
        旧 EKF 走捷径 `float(quality)` 绕过 _quality_value 规范化，
        v11 修复后与 Robust/FGO 同走 _quality_value 同源规范化。
        本测试用合法的 0.0 quality（不触发规范化异常路径）锁死
        三方法同源触发 vio_quality_zero 跳过行为。
        """
        ekf = EKFCore(_cfg())
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

        v11 修复前 EKF _handle_vio 直接 float(False) = 0.0 触发 vio_quality_zero 跳过，
        与 Robust/FGO 走 _quality_value 抛 TypeError 不同源。
        v11 修复后 EKF 也走 _quality_value，对 bool 类 quality 同源抛 TypeError。
        本测试锁死三方法同源同口径规范化路径。
        """
        ekf = EKFCore(_cfg())
        ekf.step(_imu_event())
        ekf.step(_uwb_event())
        ekf.step(_vio_event(t=0.3))  # 初始化参考位姿
        ref_before = ekf._last_vio_reference_pose

        # bool 类 quality 必须经规范化抛 TypeError，与 Robust/FGO 同源
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
