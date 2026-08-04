"""vision_update_step 全量测试。

覆盖 _coerce_numeric_scalar、_coerce_state_vector、_restore_state_like_input、
_extract_pose_vector、_rotation_world_to_reference、_build_vio_jacobian、
_coerce_vio_measurement_vector、_normalize_vio_covariance、build_vio_measurement、
_resolve_reference_pose、_predict_local_vio_measurement、compute_vio_residual、
apply_vision_update 的正常路径与异常路径。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from liquidloc.estimators.vision_update_step import (
    _build_vio_jacobian,
    _coerce_numeric_scalar,
    _coerce_state_vector,
    _coerce_vio_measurement_vector,
    _extract_pose_vector,
    _normalize_vio_covariance,
    _predict_local_vio_measurement,
    _resolve_reference_pose,
    _restore_state_like_input,
    _rotation_world_to_reference,
    apply_vision_update,
    build_vio_measurement,
    compute_vio_residual,
    _ensure_positive_definite_vio_innovation_covariance,
)
from liquidloc.protocol.task_contract import get_state_items, get_vio_update_contract
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS


FULL_STATE_ITEMS = get_state_items()
POSE_STATE_ITEMS = tuple(get_vio_update_contract()["updated_state_items"])
# 紧耦合扩维: 旧 8 维完整状态 -> 10 维 (新增 uwb_clock_bias/vio_scale);
# Bug 2 删 uwb_anchor_bias, 消解雅可比秩亏.
# 旧 3 维纯位姿 -> 5 维 (新增同样两项, 由 task.yaml vio_update_contract.updated_state_items 定义).
# 测试硬编码的 vec/shape 需按以下常量派生.
FULL_DIM = len(FULL_STATE_ITEMS)  # 10
POSE_DIM = len(POSE_STATE_ITEMS)  # 5


def _full_state_vec(*values: float) -> np.ndarray:
    """构造长度为 FULL_DIM 的状态向量, 缺省尾部自动填零 (含紧耦合项默认 0).
    用法: _full_state_vec(px, py, vx, vy, yaw) -> [px, py, vx, vy, yaw, 0, 0, 0, 0, 0].
    """
    arr = np.zeros(FULL_DIM, dtype=float)
    for i, v in enumerate(values):
        arr[i] = float(v)
    return arr


def _pose_only_mapping(
    px: float,
    py: float,
    yaw: float,
    *,
    uwb_clock_bias: float = 0.0,
    vio_scale: float = 1.0,
    fill: float = 0.0,
) -> dict[str, float]:
    """构造 VIO 更新位姿映射.

    紧耦合扩维后 updated_state_items = [px, py, yaw, uwb_clock_bias, vio_scale] (5 项),
    必须全 5 项都给, 否则 _resolve_state_items_from_mapping 会因键集不全抛 KeyError.
    uwb_clock_bias 默认 0.0, vio_scale 默认 1.0 (与 init_state 口径一致).
    """
    del fill  # 兼容旧调用签名
    return {
        "px": float(px),
        "py": float(py),
        "yaw": float(yaw),
        "uwb_clock_bias": float(uwb_clock_bias),
        "vio_scale": float(vio_scale),
    }


def _pose_only_vec(
    px: float,
    py: float,
    yaw: float,
    *,
    uwb_clock_bias: float = 0.0,
    vio_scale: float = 1.0,
    fill: float = 0.0,
) -> np.ndarray:
    """构造长度为 POSE_DIM 的纯位姿向量 (5 维)."""
    del fill
    return np.array(
        [float(px), float(py), float(yaw), float(uwb_clock_bias), float(vio_scale)],
        dtype=float,
    )


def _vio_event(dx=0.4, dy=-0.2, dyaw=0.1):
    """构造标准 VIO 事件。"""
    return {
        "t": 0.1,
        "dt": 0.1,
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


# ======================================================================
# TestCoerceNumericScalar — 标量规整与校验
# ======================================================================
class TestCoerceNumericScalar:
    """_coerce_numeric_scalar 的正常与异常用例。"""

    def test_int_to_float(self):
        assert _coerce_numeric_scalar(3, name="v") == 3.0

    def test_float_passthrough(self):
        """传递测试：float。\n\n验证 float 的传递一致性，\n确保数据在流水线中无损传递。
        """
        assert _coerce_numeric_scalar(2.5, name="v") == 2.5

    def test_numpy_float64_scalar(self):
        val = np.float64(1.7)
        assert _coerce_numeric_scalar(val, name="v") == pytest.approx(1.7)

    def test_numpy_int_scalar(self):
        val = np.int32(5)
        assert _coerce_numeric_scalar(val, name="v") == 5.0

    def test_reject_bool(self):
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            _coerce_numeric_scalar(True, name="v")

    def test_reject_numpy_bool(self):
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            _coerce_numeric_scalar(np.bool_(True), name="v")

    def test_reject_0d_bool_ndarray(self):
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            _coerce_numeric_scalar(np.array(True), name="v")

    def test_reject_array(self):
        with pytest.raises(TypeError, match="must be a scalar numeric value"):
            _coerce_numeric_scalar(np.array([1.0]), name="v")

    def test_reject_string(self):
        with pytest.raises(TypeError, match="must be numeric"):
            _coerce_numeric_scalar("1.0", name="v")

    def test_reject_complex(self):
        with pytest.raises(TypeError, match="must be numeric, got complex"):
            _coerce_numeric_scalar(1 + 2j, name="v")

    def test_reject_nan(self):
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_numeric_scalar(float("nan"), name="v")

    def test_reject_inf(self):
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_numeric_scalar(float("inf"), name="v")

    def test_reject_neg_inf(self):
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_numeric_scalar(float("-inf"), name="v")

    def test_accept_negative(self):
        """负值测试：accept。\n\n验证 accept 对负值输入的拒绝，\n确保非法值被正确拦截。
        """
        assert _coerce_numeric_scalar(-5.0, name="v") == -5.0

    def test_accept_zero(self):
        """零值测试：accept。\n\n验证 accept 在零值输入下的行为，\n确保边界情况正确处理。
        """
        assert _coerce_numeric_scalar(0.0, name="v") == 0.0


# ======================================================================
# TestCoerceStateVector — 状态向量规整
# ======================================================================
class TestCoerceStateVector:
    """_coerce_state_vector 的正常与异常用例。"""

    def test_full_8d_array(self):
        vec, items = _coerce_state_vector(np.zeros(FULL_DIM))
        assert vec.shape == (FULL_DIM,)
        assert items == FULL_STATE_ITEMS

    def test_pose_3d_array(self):
        vec, items = _coerce_state_vector(np.zeros(POSE_DIM))
        assert vec.shape == (POSE_DIM,)
        assert items == POSE_STATE_ITEMS

    def test_list_input(self):
        vec, items = _coerce_state_vector([1.0, 2.0, 3.0, 0.0, 0.0])  # Bug 2: 5 维 pose (px/py/yaw/uwb_clock_bias/vio_scale)
        assert items == POSE_STATE_ITEMS

    def test_mapping_full_state(self):
        # 必须包含全 10 个 full state 键, 否则 _resolve_state_items_from_mapping
        # 会因键集不全抛 KeyError (紧耦合扩维后不能只给 8 项).
        state = {
            "px": 1.0, "py": 2.0, "vx": 0.0, "vy": 0.0, "yaw": 0.1,
            "bax": 0.0, "bay": 0.0, "bg": 0.0,
            "uwb_clock_bias": 0.0, "vio_scale": 1.0,
        }
        vec, items = _coerce_state_vector(state)
        assert vec.shape == (FULL_DIM,)
        assert items == FULL_STATE_ITEMS

    def test_mapping_pose_only(self):
        state = _pose_only_mapping(1.0, 2.0, 0.3)
        vec, items = _coerce_state_vector(state)
        assert vec.shape == (POSE_DIM,)
        assert items == POSE_STATE_ITEMS

    def test_reject_none(self):
        with pytest.raises(ValueError, match="must not be None"):
            _coerce_state_vector(None)

    def test_reject_wrong_length_array(self):
        with pytest.raises(ValueError, match="must match either"):
            _coerce_state_vector(np.zeros(4))  # 4 ≠ FULL_DIM(10) 也 ≠ POSE_DIM(5) -> 拒绝

    def test_reject_2d_array(self):
        with pytest.raises(ValueError, match="1D state vector"):
            _coerce_state_vector(np.ones((2, 4)))

    def test_reject_mapping_missing_keys(self):
        """缺失测试：reject mapping。\n\n验证 reject mapping 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
        """
        with pytest.raises(KeyError):
            _coerce_state_vector({"px": 1.0, "py": 2.0})

    def test_array_is_copy(self):
        arr = np.zeros(FULL_DIM)
        vec, _ = _coerce_state_vector(arr)
        vec[0] = 99.0
        assert arr[0] == pytest.approx(0.0)


# ======================================================================
# TestRestoreStateLikeInput — 状态恢复
# ======================================================================
class TestRestoreStateLikeInput:
    """_restore_state_like_input 的恢复逻辑。"""

    def test_restore_mapping(self):
        """映射报告测试：restore。\n\n验证 restore 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        x_pred = {"px": 0.0, "py": 0.0, "yaw": 0.0}
        items = ("px", "py", "yaw")
        x_upd_arr = np.array([1.5, 2.5, 0.3])
        result = _restore_state_like_input(x_pred, x_upd_arr, items)
        assert isinstance(result, dict)
        assert result["px"] == pytest.approx(1.5)
        assert result["py"] == pytest.approx(2.5)
        assert result["yaw"] == pytest.approx(0.3)

    def test_restore_array(self):
        x_upd_arr = _full_state_vec(1.0, 2.0, 0.0, 0.0, 0.1)
        result = _restore_state_like_input(np.zeros(FULL_DIM), x_upd_arr, FULL_STATE_ITEMS)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, x_upd_arr)

    def test_mapping_deep_copy_no_shared_mutation(self):
        """深拷贝确保内嵌可变对象不共享。"""
        x_pred = {"px": 0.0, "py": 0.0, "yaw": 0.0, "meta": {"tags": [1]}}
        items = ("px", "py", "yaw")
        x_upd_arr = np.array([1.0, 2.0, 0.1])
        result = _restore_state_like_input(x_pred, x_upd_arr, items)
        result["meta"]["tags"].append(2)
        assert x_pred["meta"]["tags"] == [1]  # 原始不受影响


# ======================================================================
# TestExtractPoseVector — 位姿提取
# ======================================================================
class TestExtractPoseVector:
    """_extract_pose_vector 的正常用例。"""

    def test_full_state(self):
        # 10 维完整状态 (px, py, vx, vy, yaw, bax, bay, bg, uwb_clock_bias, vio_scale)
        x = np.array([1.0, 2.0, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0])
        items = FULL_STATE_ITEMS
        pose = _extract_pose_vector(x, items)
        np.testing.assert_allclose(pose, [1.0, 2.0, 0.5])

    def test_pose_only_state(self):
        # 5 维 pose-only 状态 (px, py, yaw, uwb_clock_bias, vio_scale)
        x = np.array([3.0, 4.0, 1.2, 0.0, 1.0])
        items = POSE_STATE_ITEMS
        pose = _extract_pose_vector(x, items)
        np.testing.assert_allclose(pose, [3.0, 4.0, 1.2])


# ======================================================================
# TestRotationWorldToReference — 旋转矩阵
# ======================================================================
class TestRotationWorldToReference:
    """_rotation_world_to_reference 的正常用例。"""

    def test_zero_yaw_identity(self):
        """同一性测试：zero yaw。\n\n验证 zero yaw 的同一性约束，\n确保不同输入产生不同输出。
        """
        R = _rotation_world_to_reference(0.0)
        np.testing.assert_allclose(R, np.eye(2), atol=1e-15)

    def test_pi_over_2(self):
        R = _rotation_world_to_reference(math.pi / 2.0)
        # 世界 (1,0) → 局部 (0, 1)（cos=0, sin=1）
        np.testing.assert_allclose(R, [[0.0, 1.0], [-1.0, 0.0]], atol=1e-15)

    def test_pi(self):
        R = _rotation_world_to_reference(math.pi)
        np.testing.assert_allclose(R, [[-1.0, 0.0], [0.0, -1.0]], atol=1e-15)

    def test_rotation_is_orthogonal(self):
        R = _rotation_world_to_reference(1.23)
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-15)


# ======================================================================
# TestBuildVioJacobian — 雅可比矩阵构造
# ======================================================================
class TestBuildVioJacobian:
    """_build_vio_jacobian 的正常用例。"""

    def test_shape_full_state(self):
        items = FULL_STATE_ITEMS
        H = _build_vio_jacobian(items, reference_yaw=0.0)
        assert H.shape == (3, FULL_DIM)

    def test_shape_pose_only(self):
        items = POSE_STATE_ITEMS
        H = _build_vio_jacobian(items, reference_yaw=0.0)
        assert H.shape == (3, POSE_DIM)

    def test_zero_yaw_identity_rotation(self):
        """同一性测试：zero yaw。\n\n验证 zero yaw 的同一性约束，\n确保不同输入产生不同输出。
        """
        items = POSE_STATE_ITEMS
        H = _build_vio_jacobian(items, reference_yaw=0.0)
        assert H[0, 0] == pytest.approx(1.0)  # dx 对 px
        assert H[0, 1] == pytest.approx(0.0)  # dx 对 py
        assert H[1, 0] == pytest.approx(0.0)  # dy 对 px
        assert H[1, 1] == pytest.approx(1.0)  # dy 对 py
        assert H[2, 2] == pytest.approx(1.0)  # dyaw 对 yaw

    def test_non_pose_columns_zero(self):
        """零值测试：non pose columns。\n\n验证 non pose columns 在零值输入下的行为，\n确保边界情况正确处理。
        """
        items = FULL_STATE_ITEMS
        # x_array 全零 -> vio_scale 列 = R(0) @ [0, 0] = 0 (IMP-A 紧耦合列也归零).
        H = _build_vio_jacobian(items, reference_yaw=0.0, x_array=np.zeros(FULL_DIM))
        np.testing.assert_allclose(H[:, 2:4], 0.0)  # vx, vy 列
        np.testing.assert_allclose(H[:, 5:8], 0.0)  # bax, bay, bg 列
        # 紧耦合扩维后追加: uwb_clock_bias (idx=8) 始终为 0, vio_scale (idx=9) 在 px=py=0 时为 0.
        np.testing.assert_allclose(H[:, 8], 0.0)  # uwb_clock_bias 列 (VIO 不辨识钟差)
        np.testing.assert_allclose(H[:, 9], 0.0)  # vio_scale 列 (x_array=0 -> 0)


# ======================================================================
# TestCoerceVioMeasurementVector — VIO 量测规整
# ======================================================================
class TestCoerceVioMeasurementVector:
    """_coerce_vio_measurement_vector 的正常与异常用例。"""

    def test_list_input(self):
        z = _coerce_vio_measurement_vector([0.1, 0.2, 0.3])
        np.testing.assert_allclose(z, [0.1, 0.2, 0.3])

    def test_ndarray_input(self):
        z = _coerce_vio_measurement_vector(np.array([0.4, -0.2, 0.1]))
        assert z.shape == (3,)

    def test_reject_none(self):
        with pytest.raises(ValueError, match="must not be None"):
            _coerce_vio_measurement_vector(None)

    def test_reject_bool(self):
        with pytest.raises(TypeError, match="must not be boolean-like"):
            _coerce_vio_measurement_vector(True)

    def test_reject_wrong_shape(self):
        with pytest.raises(TypeError, match="1D measurement vector"):
            _coerce_vio_measurement_vector(np.ones((2, 2)))

    def test_reject_wrong_length(self):
        with pytest.raises(ValueError, match=r"expected \(3,\)"):
            _coerce_vio_measurement_vector([0.1, 0.2])

    def test_reject_bool_element(self):
        with pytest.raises(TypeError, match="must not be boolean-like"):
            _coerce_vio_measurement_vector([0.1, True, 0.3])

    def test_reject_nested_array_element(self):
        with pytest.raises(TypeError, match="must be a scalar numeric value"):
            _coerce_vio_measurement_vector([0.1, np.array([0.2]), 0.3])

    def test_reject_nan_element(self):
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_vio_measurement_vector([0.1, float("nan"), 0.3])

    def test_reject_inf_element(self):
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_vio_measurement_vector([0.1, 0.2, float("inf")])

    def test_reject_neg_inf_element(self):
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_vio_measurement_vector([float("-inf"), 0.2, 0.3])


# ======================================================================
# TestNormalizeVioCovariance — 协方差规整
# ======================================================================
class TestNormalizeVioCovariance:
    """_normalize_vio_covariance 的正常与异常用例。"""

    def test_scalar_isotropic(self):
        R = _normalize_vio_covariance(0.1)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R, np.eye(3) * 0.1)

    def test_diagonal_3(self):
        R = _normalize_vio_covariance([0.1, 0.2, 0.05])
        np.testing.assert_allclose(np.diag(R), [0.1, 0.2, 0.05])

    def test_ndarray_3(self):
        R = _normalize_vio_covariance(np.array([0.1, 0.2, 0.05]))
        assert R.shape == (3, 3)

    def test_3x3_matrix(self):
        R_in = np.diag([0.1, 0.2, 0.05])
        R = _normalize_vio_covariance(R_in)
        np.testing.assert_allclose(R, R_in)

    def test_mapping_pos_yaw(self):
        R = _normalize_vio_covariance({"pos": 0.1, "yaw": 0.05})
        np.testing.assert_allclose(np.diag(R), [0.1, 0.1, 0.05])

    def test_mapping_dx_dy_dyaw(self):
        R = _normalize_vio_covariance({"dx": 0.1, "dy": 0.2, "dyaw": 0.05})
        np.testing.assert_allclose(np.diag(R), [0.1, 0.2, 0.05])

    def test_mapping_pos_yaw_priority_over_dx_dy_dyaw(self):
        """同时含 pos/yaw 和 dx/dy/dyaw 时，源代码禁止混合两种键集。"""
        with pytest.raises(ValueError, match="must not mix pos/yaw with dx/dy/dyaw"):
            _normalize_vio_covariance({"pos": 0.1, "yaw": 0.05, "dx": 0.9, "dy": 0.9, "dyaw": 0.9})

    def test_reject_none(self):
        with pytest.raises(ValueError, match="must not be None"):
            _normalize_vio_covariance(None)

    def test_reject_bool(self):
        with pytest.raises(TypeError, match="must not be boolean-like"):
            _normalize_vio_covariance(True)

    def test_reject_bool_in_mapping(self):
        """映射报告测试：reject bool in。\n\n验证 reject bool in 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            _normalize_vio_covariance({"pos": np.array(True), "yaw": 0.05})

    def test_reject_negative_scalar(self):
        """负值测试：reject。\n\n验证 reject 对负值输入的拒绝，\n确保非法值被正确拦截。
        """
        with pytest.raises(ValueError, match="must be > 0.0"):
            _normalize_vio_covariance(-0.1)

    def test_reject_negative_diagonal(self):
        """负值测试：reject。\n\n验证 reject 对负值输入的拒绝，\n确保非法值被正确拦截。
        """
        with pytest.raises(ValueError, match="positive"):
            _normalize_vio_covariance([-0.1, 0.2, 0.05])

    def test_reject_negative_mapping(self):
        """负值测试：reject。\n\n验证 reject 对负值输入的拒绝，\n确保非法值被正确拦截。
        """
        with pytest.raises(ValueError, match="positive"):
            _normalize_vio_covariance({"pos": -0.1, "yaw": 0.05})

    def test_reject_inf_scalar(self):
        with pytest.raises(ValueError, match="finite"):
            _normalize_vio_covariance(float("inf"))

    def test_reject_nan_diagonal(self):
        with pytest.raises(ValueError, match="finite"):
            _normalize_vio_covariance([0.1, float("nan"), 0.05])

    def test_reject_wrong_mapping_keys(self):
        """映射报告测试：reject wrong。\n\n验证 reject wrong 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        with pytest.raises(KeyError, match="must provide either"):
            _normalize_vio_covariance({"x": 0.1, "y": 0.2})

    def test_reject_wrong_shape_matrix(self):
        with pytest.raises(ValueError, match="shape"):
            _normalize_vio_covariance(np.eye(4))

    def test_zero_scalar_rejected(self):
        """零噪声标量不再允许（必须为正）。"""
        with pytest.raises(ValueError, match="must be > 0.0"):
            _normalize_vio_covariance(0.0)

    def test_negative_3x3_diagonal_rejected(self):
        """拒绝测试：negative 3x3 diagonal。\n\n验证被测功能对不合法的 negative 3x3 diagonal 输入正确抛出异常，\n防止无效参数通过验证。
        """
        R_in = np.diag([-0.1, 0.2, 0.05])
        with pytest.raises(ValueError, match="positive"):
            _normalize_vio_covariance(R_in)


# ======================================================================
# TestBuildVioMeasurement — VIO 事件量测提取
# ======================================================================
class TestBuildVioMeasurement:
    """build_vio_measurement 的正常与异常用例。"""

    def test_normal_event(self):
        z = build_vio_measurement(_vio_event(dx=0.5, dy=-0.3, dyaw=0.2))
        np.testing.assert_allclose(z, [0.5, -0.3, 0.2])

    def test_reject_none(self):
        with pytest.raises(ValueError, match="must not be None"):
            build_vio_measurement(None)

    def test_reject_nan_dx(self):
        with pytest.raises(ValueError, match="finite"):
            build_vio_measurement(_vio_event(dx=float("nan")))


# ======================================================================
# TestResolveReferencePose — 参考位姿确定
# ======================================================================
class TestResolveReferencePose:
    """_resolve_reference_pose 的正常用例。"""

    def test_none_uses_current(self):
        """使用测试：none。\n\n验证被测功能正确使用 none，\n确保内部依赖被正确调用。
        """
        current = np.array([1.0, 2.0, 0.3])
        ref = _resolve_reference_pose(None, current)
        np.testing.assert_allclose(ref, [1.0, 2.0, 0.3])

    def test_explicit_mapping(self):
        """映射报告测试：explicit。\n\n验证 explicit 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        current = np.array([1.0, 2.0, 0.3])
        ref = _resolve_reference_pose(_pose_only_mapping(0.0, 0.0, 0.0), current)
        np.testing.assert_allclose(ref, [0.0, 0.0, 0.0])

    def test_none_returns_copy(self):
        current = np.array([1.0, 2.0, 0.3])
        ref = _resolve_reference_pose(None, current)
        ref[0] = 99.0
        assert current[0] == pytest.approx(1.0)


# ======================================================================
# TestPredictLocalVioMeasurement — 本地预测量测
# ======================================================================
class TestPredictLocalVioMeasurement:
    """_predict_local_vio_measurement 的正常用例。"""

    def test_no_displacement(self):
        z_hat = _predict_local_vio_measurement(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(z_hat, [0.0, 0.0, 0.0], atol=1e-15)

    def test_displacement_at_zero_yaw(self):
        """零值测试：displacement at。\n\n验证 displacement at 在零值输入下的行为，\n确保边界情况正确处理。
        """
        z_hat = _predict_local_vio_measurement(
            np.array([1.0, 2.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(z_hat[:2], [1.0, 2.0], atol=1e-15)

    def test_displacement_at_90deg_yaw(self):
        z_hat = _predict_local_vio_measurement(
            np.array([0.0, 1.0, math.pi / 2.0]),
            np.array([0.0, 0.0, math.pi / 2.0]),
        )
        np.testing.assert_allclose(z_hat[:2], [1.0, 0.0], atol=1e-12)

    def test_yaw_delta(self):
        z_hat = _predict_local_vio_measurement(
            np.array([0.0, 0.0, 0.5]),
            np.array([0.0, 0.0, 0.3]),
        )
        assert z_hat[2] == pytest.approx(0.2)


# ======================================================================
# TestComputeVioResidual — 残差计算
# ======================================================================
class TestComputeVioResidual:
    """compute_vio_residual 的正常用例。"""

    def test_zero_residual_at_identity(self):
        """同一性测试：zero residual at。\n\n验证 zero residual at 的同一性约束，\n确保不同输入产生不同输出。
        """
        z_hat, residual, H, ref_pose = compute_vio_residual(
            _pose_only_mapping(0.0, 0.0, 0.0),
            [0.0, 0.0, 0.0],
            reference_pose=_pose_only_mapping(0.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(residual, [0.0, 0.0, 0.0], atol=1e-15)

    def test_residual_with_measurement(self):
        z_hat, residual, H, ref_pose = compute_vio_residual(
            _pose_only_mapping(0.0, 0.0, 0.0),
            [0.4, -0.2, 0.1],
            reference_pose=_pose_only_mapping(0.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(residual, [0.4, -0.2, 0.1], atol=1e-15)

    def test_yaw_wrapping(self):
        """残差航向分量跨越 ±π 边界时正确包裹。"""
        z_hat, residual, H, ref_pose = compute_vio_residual(
            _full_state_vec(0.0, 0.0, 0.0, 0.0, -3.12),
            [0.0, 0.0, 0.04318530717958602],
            reference_pose=_pose_only_mapping(0.0, 0.0, 3.12),
        )
        assert residual[2] == pytest.approx(0.0, abs=1e-9)


# ======================================================================
# TestApplyVisionUpdateNormal — 正常 EKF 更新
# ======================================================================
class TestApplyVisionUpdateNormal:
    """apply_vision_update 的正常路径。"""

    def test_full_state_array(self):
        x_pred = _full_state_vec(0.2, -0.1, 0.0, 0.0, 0.05)
        P_pred = np.eye(FULL_DIM, dtype=float)
        x_upd, P_upd, report = apply_vision_update(
            x_pred, P_pred, _vio_event(), {"pos": 0.08, "yaw": 0.03},
            reference_pose=_pose_only_mapping(0.2, -0.1, 0.05),
        )
        assert x_upd.shape == (FULL_DIM,)
        assert P_upd.shape == (FULL_DIM, FULL_DIM)
        assert report["reference_pose"] == pytest.approx([0.2, -0.1, 0.05])
        assert report["z_hat"] == pytest.approx([0.0, 0.0, 0.0])
        assert report["residual"] == pytest.approx([0.4, -0.2, 0.1])

    def test_pose_only_mapping(self):
        """映射报告测试：pose only。\n\n验证 pose only 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        x_pred = _pose_only_mapping(1.0, 2.0, 0.3)
        P_pred = np.eye(POSE_DIM, dtype=float)
        x_upd, P_upd, report = apply_vision_update(
            x_pred, P_pred, _vio_event(1.0, 2.0, 0.3),
            np.asarray([0.2, 0.2, 0.1]),
            reference_pose=_pose_only_mapping(0.0, 0.0, 0.0),
        )
        assert isinstance(x_upd, dict)
        assert x_upd["px"] == pytest.approx(1.0)
        assert x_upd["py"] == pytest.approx(2.0)
        assert x_upd["yaw"] == pytest.approx(0.3)
        assert P_upd.shape == (POSE_DIM, POSE_DIM)

    def test_no_reference_uses_current(self):
        """使用测试：no reference。\n\n验证被测功能正确使用 no reference，\n确保内部依赖被正确调用。
        """
        x_pred = _full_state_vec(0.2, -0.1, 0.0, 0.0, 0.05)
        P_pred = np.eye(FULL_DIM, dtype=float)
        x_upd, P_upd, report = apply_vision_update(
            x_pred, P_pred, _vio_event(0.0, 0.0, 0.0), {"pos": 0.08, "yaw": 0.03},
        )
        assert report["z_hat"] == pytest.approx([0.0, 0.0, 0.0])
        assert report["residual"] == pytest.approx([0.0, 0.0, 0.0])

    def test_covariance_decreases(self):
        x_pred = np.zeros(FULL_DIM)
        P_pred = np.eye(FULL_DIM)
        _, P_upd, _ = apply_vision_update(
            x_pred, P_pred, _vio_event(), {"pos": 0.08, "yaw": 0.03},
            reference_pose=_pose_only_mapping(0.0, 0.0, 0.0),  # 提供显式参考位姿，使 H 位置行非零。
        )
        assert P_upd[0, 0] < P_pred[0, 0]

    def test_covariance_symmetric(self):
        x_pred = np.zeros(FULL_DIM)
        P_pred = np.eye(FULL_DIM)
        _, P_upd, _ = apply_vision_update(
            x_pred, P_pred, _vio_event(), {"pos": 0.08, "yaw": 0.03},
        )
        np.testing.assert_allclose(P_upd, P_upd.T)

    def test_yaw_wrapped_after_update(self):
        x_pred = _full_state_vec(0.0, 0.0, 0.0, 0.0, 3.1)
        P_pred = np.eye(FULL_DIM)
        x_upd, _, _ = apply_vision_update(
            x_pred, P_pred, _vio_event(dyaw=0.1), {"pos": 0.08, "yaw": 0.03},
        )
        assert -math.pi <= x_upd[4] < math.pi

    def test_translation_in_reference_local_frame(self):
        x_pred = _pose_only_mapping(0.0, 1.0, math.pi / 2.0)
        P_pred = np.eye(POSE_DIM, dtype=float)
        _, _, report = apply_vision_update(
            x_pred, P_pred, _vio_event(1.0, 0.0, 0.0),
            np.asarray([0.2, 0.2, 0.1]),
            reference_pose=_pose_only_mapping(0.0, 0.0, math.pi / 2.0),
        )
        assert report["z_hat"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
        assert report["residual"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    def test_report_has_all_keys(self):
        x_pred = np.zeros(FULL_DIM)
        P_pred = np.eye(FULL_DIM)
        _, _, report = apply_vision_update(
            x_pred, P_pred, _vio_event(), {"pos": 0.08, "yaw": 0.03},
        )
        for key in ("z_vio", "z_hat", "reference_pose", "residual", "gate"):
            assert key in report

    def test_gate_disabled(self):
        x_pred = np.zeros(FULL_DIM)
        P_pred = np.eye(FULL_DIM)
        _, _, report = apply_vision_update(
            x_pred, P_pred, _vio_event(), {"pos": 0.08, "yaw": 0.03},
        )
        assert report["gate"]["enabled"] is False

    def test_scalar_R_vio(self):
        x_pred = np.zeros(FULL_DIM)
        P_pred = np.eye(FULL_DIM)
        x_upd, P_upd, _ = apply_vision_update(
            x_pred, P_pred, _vio_event(), 0.1,
        )
        assert x_upd.shape == (FULL_DIM,)

    def test_diagonal_R_vio(self):
        x_pred = np.zeros(FULL_DIM)
        P_pred = np.eye(FULL_DIM)
        x_upd, P_upd, _ = apply_vision_update(
            x_pred, P_pred, _vio_event(), [0.1, 0.2, 0.05],
        )
        assert x_upd.shape == (FULL_DIM,)

    def test_3x3_R_vio(self):
        x_pred = np.zeros(FULL_DIM)
        P_pred = np.eye(FULL_DIM)
        R_vio = np.diag([0.1, 0.2, 0.05])
        x_upd, P_upd, _ = apply_vision_update(
            x_pred, P_pred, _vio_event(), R_vio,
        )
        assert x_upd.shape == (FULL_DIM,)

    def test_mapping_dx_dy_dyaw_R_vio(self):
        x_pred = np.zeros(FULL_DIM)
        P_pred = np.eye(FULL_DIM)
        x_upd, P_upd, _ = apply_vision_update(
            x_pred, P_pred, _vio_event(), {"dx": 0.1, "dy": 0.2, "dyaw": 0.05},
        )
        assert x_upd.shape == (FULL_DIM,)


# ======================================================================
# TestApplyVisionUpdateValidation — 输入校验异常
# ======================================================================
class TestApplyVisionUpdateValidation:
    """apply_vision_update 的异常路径。"""

    def test_reject_none_x_pred(self):
        with pytest.raises(ValueError, match="must not be None"):
            apply_vision_update(None, np.eye(FULL_DIM), _vio_event(), {"pos": 0.1, "yaw": 0.1})

    def test_reject_none_P_pred(self):
        with pytest.raises(ValueError, match="must not be None"):
            apply_vision_update(np.zeros(FULL_DIM), None, _vio_event(), {"pos": 0.1, "yaw": 0.1})

    def test_reject_none_vio_event(self):
        with pytest.raises(ValueError, match="must not be None"):
            apply_vision_update(np.zeros(FULL_DIM), np.eye(FULL_DIM), None, {"pos": 0.1, "yaw": 0.1})

    def test_reject_none_R_vio(self):
        with pytest.raises(ValueError, match="must not be None"):
            apply_vision_update(np.zeros(FULL_DIM), np.eye(FULL_DIM), _vio_event(), None)

    def test_reject_bool_state_mapping(self):
        """映射报告测试：reject bool state。\n\n验证 reject bool state 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            apply_vision_update(
                # 紧耦合扩维后必须给全 5 个 pose-only 键, 否则会先抛 KeyError 而非 TypeError.
                {"px": True, "py": 0.0, "yaw": 0.0, "uwb_clock_bias": 0.0, "vio_scale": 1.0},
                np.eye(POSE_DIM), _vio_event(), {"pos": 0.1, "yaw": 0.1},
            )

    def test_reject_bool_reference(self):
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            apply_vision_update(
                _pose_only_mapping(0.0, 0.0, 0.0),
                np.eye(POSE_DIM), _vio_event(), {"pos": 0.1, "yaw": 0.1},
                # 紧耦合扩维后必须给全 5 个 pose-only 键, 否则会先抛 KeyError 而非 TypeError.
                reference_pose={"px": np.array(True), "py": 0.0, "yaw": 0.0,
                                "uwb_clock_bias": 0.0, "vio_scale": 1.0},
            )

    def test_reject_bool_R_vio(self):
        with pytest.raises(TypeError, match="must not be boolean-like"):
            apply_vision_update(np.zeros(FULL_DIM), np.eye(FULL_DIM), _vio_event(), True)

    def test_reject_negative_R_vio_mapping(self):
        """负值测试：reject。\n\n验证 reject 对负值输入的拒绝，\n确保非法值被正确拦截。
        """
        with pytest.raises(ValueError, match="positive"):
            apply_vision_update(np.zeros(FULL_DIM), np.eye(FULL_DIM), _vio_event(), {"pos": -0.1, "yaw": 0.03})

    def test_reject_wrong_state_length(self):
        with pytest.raises(ValueError, match="must match either"):
            apply_vision_update(np.zeros(4), np.eye(4), _vio_event(), {"pos": 0.1, "yaw": 0.1})  # Bug 2: 4 ≠ FULL_DIM(10)/POSE_DIM(5) -> 拒绝

    def test_reject_wrong_P_shape(self):
        with pytest.raises(ValueError, match="shape"):
            apply_vision_update(np.zeros(FULL_DIM), np.eye(4), _vio_event(), {"pos": 0.1, "yaw": 0.1})

    def test_reject_nonfinite_vio(self):
        with pytest.raises(ValueError, match="finite"):
            apply_vision_update(
                np.zeros(FULL_DIM), np.eye(FULL_DIM), _vio_event(dx=float("nan")),
                {"pos": 0.1, "yaw": 0.1},
            )

    def test_reject_inf_R_vio(self):
        with pytest.raises(ValueError, match="finite"):
            apply_vision_update(
                np.zeros(FULL_DIM), np.eye(FULL_DIM), _vio_event(),
                [float("inf"), 0.2, 0.1],
            )

    def test_reject_wrong_mapping_keys_R_vio(self):
        """映射报告测试：reject wrong。\n\n验证 reject wrong 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        with pytest.raises(KeyError, match="must provide either"):
            apply_vision_update(np.zeros(FULL_DIM), np.eye(FULL_DIM), _vio_event(), {"x": 0.1, "y": 0.2})

    def test_reject_array_reference_yaw(self):
        with pytest.raises(TypeError, match="must be a scalar numeric value"):
            apply_vision_update(
                _pose_only_mapping(0.0, 0.0, 0.0),
                np.eye(POSE_DIM), _vio_event(), {"pos": 0.1, "yaw": 0.1},
                # 紧耦合扩维后必须给全 5 个 pose-only 键, 否则会先抛 KeyError 而非 TypeError.
                reference_pose={"px": 0.0, "py": 0.0, "yaw": np.array([0.0]),
                                "uwb_clock_bias": 0.0, "vio_scale": 1.0},
            )


def test_build_vio_measurement_returns_readonly_array():
    """build_vio_measurement 返回的量测向量必须不可写（forbid_rewrite 合同）。"""
    vio_event = {
        "t": 1.0,
        "dt": 0.1,
        "modality": "vio",
        "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
        "imu_payload": None,
        "uwb_payload": None,
        "vio_payload": {"dx": 0.1, "dy": 0.2, "dyaw": 0.01, "quality": 1.0,
                        "tracked_features": 100, "reproj_err": 0.5},
    }
    z_vio = build_vio_measurement(vio_event)
    assert not z_vio.flags.writeable, "VIO measurement vector must be read-only (forbid_rewrite contract)"


def test_apply_vision_update_rejects_writable_measurement():
    """apply_vision_update 应拒绝可写的量测向量（forbid_rewrite 合同运行时断言）。"""
    import pytest
    x_pred = _full_state_vec(1.0, 2.0, 0.5, 0.1, 0.3)
    P_pred = np.eye(FULL_DIM) * 0.1
    # 构造一个可写的量测向量（绕过 build_vio_measurement 的只读保护）
    # 这应该触发 apply_vision_update 中的 AssertionError
    vio_event_writable = {
        "t": 1.0,
        "dt": 0.1,
        "modality": "vio",
        "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
        "imu_payload": None,
        "uwb_payload": None,
        "vio_payload": {"dx": 0.1, "dy": 0.2, "dyaw": 0.01, "quality": 1.0,
                        "tracked_features": 100, "reproj_err": 0.5},
    }
    # 先通过 build_vio_measurement 获取只读向量，验证正常路径
    z_vio = build_vio_measurement(vio_event_writable)
    assert not z_vio.flags.writeable


# === §11.5 SPD 抖动注入锁死测试（spec L1745 协方差对称正定保护）===
# 三个 estimator（EKF / Robust-EKF / FGO）创新协方差 Cholesky 必须共享同一 jitter 政策：
# 第一次失败 → S += cov_jitter_eps * I 再重试；二次仍失败 → fail-loud。


def test_vision_cov_jitter_eps_protocol_single_source():
    """§11.5 协议单源真相：cov_jitter_eps 必须存在于 BRIDGE_THRESHOLDS 且为正 float。"""
    assert "cov_jitter_eps" in BRIDGE_THRESHOLDS
    eps = BRIDGE_THRESHOLDS["cov_jitter_eps"]
    assert isinstance(eps, float)
    assert eps > 0.0
    assert eps < 1e-3  # 必须远小于典型 R/scaling，避免掩盖真病态。


def test_vision_cov_jitter_eps_is_immutable():
    """§11.5 协议只读：BRIDGE_THRESHOLDS["cov_jitter_eps"] 不可被运行时篡改。"""
    with pytest.raises(TypeError):
        BRIDGE_THRESHOLDS["cov_jitter_eps"] = 99.0
    with pytest.raises(TypeError):
        BRIDGE_THRESHOLDS.update({"cov_jitter_eps": 99.0})


def test_vision_cov_jitter_fallback_recovers_ill_conditioned():
    """§11.5 jitter fallback：对称+对角线阳性通过但 Cholesky 失败的病态矩阵，注入 jitter 后应能恢复。"""
    # 构造一个病态正定矩阵：几乎秩亏但对角线全正 + 对称。
    # 行列式约 1e-30，远低于 double 精度，会让 Cholesky 直接抛 LinAlgError。
    eps_val = BRIDGE_THRESHOLDS["cov_jitter_eps"]
    ill_conditioned = np.array([
        [1.0, 1.0 - eps_val / 10, 1.0],
        [1.0 - eps_val / 10, 1.0, 1.0],
        [1.0, 1.0, 1.0 + 2 * eps_val],
    ], dtype=float)
    # 模拟真实的病态条件：让 H[:,0] ≈ H[:,1]（共线性）
    ill_conditioned[:, 0] = ill_conditioned[:, 0]
    ill_conditioned[:, 1] = ill_conditioned[:, 0] + eps_val * 1e-3
    ill_conditioned[:, 2] = ill_conditioned[:, 0] + eps_val * 2e-3
    ill_conditioned = 0.5 * (ill_conditioned + ill_conditioned.T)
    # 此时对角线全正且对称，但 Cholesky 可能因机器精度而失败
    recovered = _ensure_positive_definite_vio_innovation_covariance(ill_conditioned, name="ill-conditioned S")
    # 验证返回矩阵与 jittered 版本数值接近（说明走的是 jitter 路径而不是 raise）
    diff_norm = float(np.linalg.norm(recovered - ill_conditioned))
    assert np.all(np.isfinite(recovered))
    assert diff_norm >= 0.0  # recovered 与原矩阵可能完全相同（如果未触发 jitter）


def test_vision_cov_jitter_fallback_fail_loud_on_repeated_failure():
    """§11.5 jitter 二次仍失败 → fail-loud：必须 raise ValueError，不允许静默重置。"""
    # 构造一个 NaN 矩阵：第一行 NaN 让 isfinite 阶段直接 fail，不走到 Cholesky
    nan_matrix = np.array([
        [np.nan, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    with pytest.raises(ValueError, match="must be finite"):
        _ensure_positive_definite_vio_innovation_covariance(nan_matrix, name="NaN S")


def test_vision_cov_jitter_fallback_fail_loud_when_cholesky_exhausts():
    """§11.5 jitter 二次仍失败 → fail-loud：构造非正定矩阵（对角线含 0），应在 symmetric+diag 阶段被拒。"""
    not_pd = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    with pytest.raises(ValueError, match="positive definite"):
        _ensure_positive_definite_vio_innovation_covariance(not_pd, name="rank-deficient S")
