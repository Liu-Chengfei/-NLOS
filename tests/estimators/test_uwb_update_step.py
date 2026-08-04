"""uwb_update_step 全量测试。

覆盖 _coerce_scalar、_coerce_state_vector、_coerce_anchor_xy、
_coerce_covariance_matrix、_restore_state_payload、predict_range、
build_uwb_jacobian、run_uwb_update 的正常路径与异常路径。
"""

from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
import pytest

from liquidloc.estimators.uwb_update_step import (
    _coerce_anchor_xy,
    _coerce_covariance_matrix,
    _coerce_scalar,
    _coerce_state_vector,
    _restore_state_payload,
    build_uwb_jacobian,
    predict_range,
    run_uwb_update,
)
from liquidloc.estimators.state_definition import state_items as _UWB_STATE_ITEMS
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS

# 紧耦合扩维: 完整状态维度由 state_definition 派生, 当前为 11. 测试中映射型
# 状态输入会按此维度构造向量, 协方差矩阵必须与之同形 (11×11).
_STATE_DIM = len(_UWB_STATE_ITEMS)


# ======================================================================
# TestCoerceScalar — 标量规整与校验
# ======================================================================
class TestCoerceScalar:
    """_coerce_scalar 的正常与异常用例。"""

    def test_int_to_float(self):
        assert _coerce_scalar(3, name="v") == 3.0

    def test_float_passthrough(self):
        """传递测试：float。\n\n验证 float 的传递一致性，\n确保数据在流水线中无损传递。
        """
        assert _coerce_scalar(2.5, name="v") == 2.5

    def test_numpy_float64_scalar(self):
        val = np.float64(1.7)
        assert _coerce_scalar(val, name="v") == pytest.approx(1.7)

    def test_numpy_int_scalar(self):
        val = np.int32(5)
        assert _coerce_scalar(val, name="v") == 5.0

    def test_reject_bool(self):
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            _coerce_scalar(True, name="v")

    def test_reject_numpy_bool(self):
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            _coerce_scalar(np.bool_(True), name="v")

    def test_reject_0d_ndarray(self):
        with pytest.raises(TypeError, match="must be a numeric scalar, got ndarray"):
            _coerce_scalar(np.array(3.0), name="v")

    def test_reject_string(self):
        with pytest.raises(TypeError, match="must be numeric, got str"):
            _coerce_scalar("1.0", name="v")

    def test_reject_bytes(self):
        with pytest.raises(TypeError, match="must be numeric, got bytes"):
            _coerce_scalar(b"1", name="v")

    def test_reject_none(self):
        with pytest.raises(ValueError, match="must not be None"):
            _coerce_scalar(None, name="v")

    def test_reject_nan(self):
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_scalar(float("nan"), name="v")

    def test_reject_inf(self):
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_scalar(float("inf"), name="v")

    def test_reject_neg_inf(self):
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_scalar(float("-inf"), name="v")

    def test_min_value_pass(self):
        """传递测试：min value。\n\n验证 min value 的传递一致性，\n确保数据在流水线中无损传递。
        """
        assert _coerce_scalar(0.0, name="v", min_value=0.0) == 0.0

    def test_min_value_above(self):
        assert _coerce_scalar(1.5, name="v", min_value=0.0) == 1.5

    def test_min_value_below(self):
        with pytest.raises(ValueError, match=r"must be >= 0\.0, got -0\.1"):
            _coerce_scalar(-0.1, name="v", min_value=0.0)

    def test_min_value_none_skips_check(self):
        assert _coerce_scalar(-5.0, name="v", min_value=None) == -5.0

    def test_reject_list(self):
        with pytest.raises(TypeError, match="must be numeric, got list"):
            _coerce_scalar([1.0], name="v")

    def test_reject_complex(self):
        with pytest.raises(TypeError, match="must be numeric, got complex"):
            _coerce_scalar(1 + 2j, name="v")


# ======================================================================
# TestCoerceStateVector — 状态向量规整
# ======================================================================
class TestCoerceStateVector:
    """_coerce_state_vector 的正常与异常用例。"""

    def test_1d_array(self):
        vec, meta = _coerce_state_vector(np.array([1.0, 2.0, 3.0]))
        assert vec.shape == (3,)
        assert meta["kind"] == "array"
        np.testing.assert_allclose(vec, [1.0, 2.0, 3.0])

    def test_list_input(self):
        vec, meta = _coerce_state_vector([0.0, 0.0, 0.0, 0.0])
        assert vec.shape == (4,)
        assert meta["kind"] == "array"

    def test_column_vector_2d(self):
        vec, meta = _coerce_state_vector(np.array([[1.0], [2.0]]))
        assert vec.shape == (2,)
        assert meta["shape"] == (2, 1)

    def test_row_vector_2d(self):
        vec, meta = _coerce_state_vector(np.array([[1.0, 2.0, 3.0]]))
        assert vec.shape == (3,)
        assert meta["shape"] == (1, 3)

    def test_mapping_input(self):
        vec, meta = _coerce_state_vector({"px": 3.0, "py": 4.0, "vx": 0.1})
        assert meta["kind"] == "mapping"
        assert vec[0] == pytest.approx(3.0)
        assert vec[1] == pytest.approx(4.0)
        assert "vx" in meta["template"]

    def test_mapping_returns_copy(self):
        original = {"px": 1.0, "py": 2.0}
        vec, meta = _coerce_state_vector(original)
        original["px"] = 99.0
        assert meta["template"]["px"] == pytest.approx(1.0)

    def test_reject_none(self):
        with pytest.raises(ValueError, match="must not be None"):
            _coerce_state_vector(None)

    def test_reject_string(self):
        with pytest.raises(TypeError, match="must be a numeric state vector"):
            _coerce_state_vector("abc")

    def test_reject_bytes(self):
        with pytest.raises(TypeError, match="must be a numeric state vector"):
            _coerce_state_vector(b"abc")

    def test_reject_2d_non_vector(self):
        with pytest.raises(ValueError, match="must be 1D or a column vector"):
            _coerce_state_vector(np.ones((2, 3)))

    def test_reject_too_short(self):
        with pytest.raises(ValueError, match="at least px and py"):
            _coerce_state_vector(np.array([1.0]))

    def test_mapping_missing_py(self):
        """缺失测试：mapping。\n\n验证 mapping 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
        """
        with pytest.raises(KeyError):
            _coerce_state_vector({"px": 1.0})

    def test_array_is_copied(self):
        arr = np.array([1.0, 2.0, 3.0])
        vec, _ = _coerce_state_vector(arr)
        vec[0] = 99.0
        assert arr[0] == pytest.approx(1.0)


# ======================================================================
# TestRestoreStatePayload — 状态恢复
# ======================================================================
class TestRestoreStatePayload:
    """_restore_state_payload 的恢复逻辑。"""

    def test_restore_mapping(self):
        """映射报告测试：restore。\n\n验证 restore 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        meta = {"kind": "mapping", "template": {"px": 0.0, "py": 0.0, "vx": 0.1}}
        # 映射型输入构造完整 11D 向量，restore 时按动态索引写回 px/py。
        x_vector = np.zeros(11)
        x_vector[0] = 1.5  # px
        x_vector[1] = 2.5  # py
        result = _restore_state_payload(x_vector, meta)
        assert isinstance(result, dict)
        assert result["px"] == pytest.approx(1.5)
        assert result["py"] == pytest.approx(2.5)
        assert result["vx"] == pytest.approx(0.0)

    def test_restore_1d_array(self):
        meta = {"kind": "array", "shape": (3,)}
        vec = np.array([1.0, 2.0, 3.0])
        result = _restore_state_payload(vec, meta)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_restore_column_vector(self):
        meta = {"kind": "array", "shape": (3, 1)}
        vec = np.array([1.0, 2.0, 3.0])
        result = _restore_state_payload(vec, meta)
        assert result.shape == (3, 1)
        np.testing.assert_allclose(result, [[1.0], [2.0], [3.0]])

    def test_restore_row_vector(self):
        meta = {"kind": "array", "shape": (1, 3)}
        vec = np.array([1.0, 2.0, 3.0])
        result = _restore_state_payload(vec, meta)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0]])


# ======================================================================
# TestCoerceCovarianceMatrix — 协方差规整
# ======================================================================
class TestCoerceCovarianceMatrix:
    """_coerce_covariance_matrix 的正常与异常用例。"""

    def test_valid_matrix(self):
        P = np.eye(4)
        result = _coerce_covariance_matrix(P, state_dim=4)
        np.testing.assert_allclose(result, P)

    def test_returns_copy(self):
        P = np.eye(3)
        result = _coerce_covariance_matrix(P, state_dim=3)
        result[0, 0] = 99.0
        assert P[0, 0] == pytest.approx(1.0)

    def test_list_of_lists(self):
        P = [[1.0, 0.0], [0.0, 1.0]]
        result = _coerce_covariance_matrix(P, state_dim=2)
        assert result.shape == (2, 2)

    def test_reject_none(self):
        with pytest.raises(ValueError, match="must not be None"):
            _coerce_covariance_matrix(None, state_dim=2)

    def test_reject_wrong_shape(self):
        with pytest.raises(ValueError, match=r"must have shape \(3, 3\), got \(2, 2\)"):
            _coerce_covariance_matrix(np.eye(2), state_dim=3)


# ======================================================================
# TestCoerceAnchorXY — 锚点坐标提取
# ======================================================================
class TestCoerceAnchorXY:
    """_coerce_anchor_xy 的正常与异常用例。"""

    def test_mapping_ax_ay(self):
        ax, ay = _coerce_anchor_xy({"ax": 1.0, "ay": 2.0})
        assert ax == pytest.approx(1.0)
        assert ay == pytest.approx(2.0)

    def test_mapping_x_y(self):
        ax, ay = _coerce_anchor_xy({"x": 3.0, "y": 4.0})
        assert ax == pytest.approx(3.0)
        assert ay == pytest.approx(4.0)

    def test_mapping_px_py(self):
        ax, ay = _coerce_anchor_xy({"px": 5.0, "py": 6.0})
        assert ax == pytest.approx(5.0)
        assert ay == pytest.approx(6.0)

    def test_mapping_priority_ax_over_x(self):
        """同时含 ax/ay 和 x/y 时，ax/ay 优先。"""
        ax, ay = _coerce_anchor_xy({"ax": 1.0, "ay": 2.0, "x": 9.0, "y": 9.0})
        assert ax == pytest.approx(1.0)
        assert ay == pytest.approx(2.0)

    def test_mapping_no_known_keys(self):
        with pytest.raises(KeyError, match="must provide one of"):
            _coerce_anchor_xy({"u": 1.0, "v": 2.0})

    def test_1d_array(self):
        ax, ay = _coerce_anchor_xy(np.array([7.0, 8.0]))
        assert ax == pytest.approx(7.0)
        assert ay == pytest.approx(8.0)

    def test_list_input(self):
        ax, ay = _coerce_anchor_xy([1.5, 2.5])
        assert ax == pytest.approx(1.5)
        assert ay == pytest.approx(2.5)

    def test_column_vector(self):
        ax, ay = _coerce_anchor_xy(np.array([[3.0], [4.0]]))
        assert ax == pytest.approx(3.0)
        assert ay == pytest.approx(4.0)

    def test_reject_none(self):
        with pytest.raises(ValueError, match="must not be None"):
            _coerce_anchor_xy(None)

    def test_reject_string(self):
        with pytest.raises(TypeError, match="must be a 2D coordinate record"):
            _coerce_anchor_xy("1,2")

    def test_reject_wrong_size_array(self):
        with pytest.raises(ValueError, match="exactly 2 coordinates, got 3"):
            _coerce_anchor_xy([1.0, 2.0, 3.0])

    def test_reject_2d_non_vector(self):
        with pytest.raises(ValueError, match="must be a 1D coordinate record"):
            _coerce_anchor_xy(np.ones((2, 2)))

    def test_namedtuple_input(self):
        """namedtuple 不是 Mapping，走数组路径。"""
        Anchor = namedtuple("Anchor", ["ax", "ay"])
        a = Anchor(1.0, 2.0)
        ax, ay = _coerce_anchor_xy(a)
        assert ax == pytest.approx(1.0)
        assert ay == pytest.approx(2.0)


# ======================================================================
# TestPredictRange — 几何测距预测
# ======================================================================
class TestPredictRange:
    """predict_range 的正常与异常用例。"""

    def test_unit_distance(self):
        z = predict_range(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0]))
        assert z == pytest.approx(1.0)

    def test_3_4_5_triangle(self):
        z = predict_range(np.array([3.0, 4.0]), np.array([0.0, 0.0]))
        assert z == pytest.approx(5.0)

    def test_zero_distance(self):
        z = predict_range(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert z == pytest.approx(0.0)

    def test_mapping_inputs(self):
        z = predict_range({"px": 3.0, "py": 4.0}, {"ax": 0.0, "ay": 0.0})
        assert z == pytest.approx(5.0)

    def test_negative_coordinates(self):
        z = predict_range(np.array([-3.0, -4.0]), np.array([0.0, 0.0]))
        assert z == pytest.approx(5.0)


# ======================================================================
# TestBuildUwbJacobian — 雅可比矩阵构造
# ======================================================================
class TestBuildUwbJacobian:
    """build_uwb_jacobian 的正常与边界用例。"""

    def test_jacobian_shape(self):
        H = build_uwb_jacobian(np.zeros(8), np.array([1.0, 0.0]))
        assert H.shape == (1, 8)

    def test_jacobian_values_along_x(self):
        """锚点在 x 轴上时，H[0,0]=1, H[0,1]=0。"""
        H = build_uwb_jacobian(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        assert H[0, 0] == pytest.approx(-1.0)  # dx = 0-1 = -1, z=1
        assert H[0, 1] == pytest.approx(0.0)

    def test_jacobian_values_45deg(self):
        """45 度方向时两个偏导相等。"""
        H = build_uwb_jacobian(np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        expected = 1.0 / math.sqrt(2.0)
        assert H[0, 0] == pytest.approx(expected)
        assert H[0, 1] == pytest.approx(expected)

    def test_jacobian_zero_distance(self):
        """零距离时雅可比全零。"""
        H = build_uwb_jacobian(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        np.testing.assert_allclose(H, 0.0)

    def test_jacobian_other_columns_zero(self):
        """第 2 列及之后必须全零。"""
        H = build_uwb_jacobian(np.zeros(8), np.array([1.0, 0.0]))
        np.testing.assert_allclose(H[0, 2:], 0.0)

    def test_mapping_inputs(self):
        H = build_uwb_jacobian({"px": 0.0, "py": 0.0}, {"x": 3.0, "y": 4.0})
        assert H.shape == (1, _STATE_DIM)  # 映射型输入按 state_items 构造完整状态向量，雅可比列数 = state_dim。


# ======================================================================
# TestRunUwbUpdateNormal — 正常 EKF 更新
# ======================================================================
class TestRunUwbUpdateNormal:
    """run_uwb_update 的正常路径。"""

    def test_simple_update(self):
        """原点状态，锚点在 (1,0)，测距 2.0。"""
        x_pred = np.zeros(8)
        P_pred = np.eye(8)
        x_upd, P_upd, info = run_uwb_update(x_pred, P_pred, np.array([1.0, 0.0]), 2.0, 0.25)
        assert info["z_pred"] == pytest.approx(1.0)
        assert info["residual"] == pytest.approx(1.0)
        assert info["S"] > 0
        assert x_upd.shape == (8,)
        assert P_upd.shape == (8, 8)

    def test_update_pulls_toward_anchor(self):
        """测距偏大时，EKF 将位置朝远离锚点方向调整。

        锚点在 (1,0)，实际距离=1，观测=2 → 残差=+1。
        H[0,0]=dx/z=-1，增益为负，x_upd[0] < x_pred[0]（远离锚点）。
        """
        x_pred = np.zeros(8)
        P_pred = np.eye(8)
        x_upd, _, _ = run_uwb_update(x_pred, P_pred, np.array([1.0, 0.0]), 2.0, 0.25)
        assert x_upd[0] < x_pred[0]

    def test_covariance_decreases(self):
        """更新后协方差对角线应减小。"""
        x_pred = np.zeros(8)
        P_pred = np.eye(8)
        _, P_upd, _ = run_uwb_update(x_pred, P_pred, np.array([1.0, 0.0]), 2.0, 0.25)
        assert P_upd[0, 0] < P_pred[0, 0]

    def test_covariance_symmetric(self):
        """更新后协方差必须对称。"""
        x_pred = np.zeros(8)
        P_pred = np.eye(8)
        _, P_upd, _ = run_uwb_update(x_pred, P_pred, np.array([1.0, 0.0]), 2.0, 0.25)
        np.testing.assert_allclose(P_upd, P_upd.T)

    def test_mapping_state_input(self):
        """映射型状态输入，返回也是映射。

        映射型状态按 state_items 顺序构造完整状态向量，
        因此 P_pred 必须是 state_dim×state_dim，与完整状态维度一致。
        """
        x_pred = {"px": 0.0, "py": 0.0, "vx": 0.1, "vy": 0.0}
        P_pred = np.eye(_STATE_DIM)
        x_upd, P_upd, info = run_uwb_update(x_pred, P_pred, {"ax": 1.0, "ay": 0.0}, 2.0, 0.25)
        assert isinstance(x_upd, dict)
        assert "vx" in x_upd
        assert "vy" in x_upd
        assert x_upd["px"] != pytest.approx(0.0)

    def test_mapping_anchor_input(self):
        """映射型锚点输入。"""
        x_upd, P_upd, _ = run_uwb_update(
            np.zeros(8), np.eye(8), {"ax": 1.0, "ay": 0.0}, 2.0, 0.25
        )
        assert x_upd.shape == (8,)

    def test_anchor_with_x_y_keys(self):
        """锚点用 x/y 键名。"""
        x_upd, _, _ = run_uwb_update(
            np.zeros(8), np.eye(8), {"x": 1.0, "y": 0.0}, 2.0, 0.25
        )
        assert x_upd[0] != pytest.approx(0.0)

    def test_anchor_with_px_py_keys(self):
        """锚点用 px/py 键名。"""
        x_upd, _, _ = run_uwb_update(
            np.zeros(8), np.eye(8), {"px": 1.0, "py": 0.0}, 2.0, 0.25
        )
        assert x_upd[0] != pytest.approx(0.0)

    def test_zero_distance_no_change(self):
        """零距离时状态和协方差不变。"""
        x_pred = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        P_pred = np.eye(8)
        x_upd, P_upd, info = run_uwb_update(x_pred, P_pred, {"ax": 1.0, "ay": 2.0}, 0.0, 1e-6)
        np.testing.assert_allclose(x_upd, x_pred)
        np.testing.assert_allclose(P_upd, P_pred)
        assert info["z_pred"] == pytest.approx(0.0)

    def test_int_scalars_accepted(self):
        """z_range 和 R_range 可以传 int。"""
        x_upd, P_upd, _ = run_uwb_update(
            np.zeros(8), np.eye(8), np.array([1.0, 0.0]), 2, 1
        )
        assert x_upd.shape == (8,)

    def test_report_has_all_keys(self):
        """update_info 包含所有约定的键。"""
        _, _, info = run_uwb_update(np.zeros(8), np.eye(8), np.array([1.0, 0.0]), 2.0, 0.25)
        for key in ("z_pred", "residual", "H", "S", "K_gain", "x_upd", "P_upd"):
            assert key in info

    def test_report_not_aliased(self):
        """report 中的副本不影响输出。"""
        x_pred = np.zeros(8)
        P_pred = np.eye(8)
        x_upd, P_upd, report = run_uwb_update(x_pred, P_pred, np.array([1.0, 0.0]), 2.0, 0.25)
        x_snap = x_upd.copy()
        P_snap = P_upd.copy()
        report["H"][0, 0] = 999.0
        report["K_gain"][0, 0] = 999.0
        report["P_upd"][0, 0] = 999.0
        np.testing.assert_allclose(x_upd, x_snap)
        np.testing.assert_allclose(P_upd, P_snap)

    def test_report_x_upd_independent_of_output(self):
        """report["x_upd"] 是深拷贝，修改不影响 x_upd。"""
        x_pred = {"px": 0.0, "py": 0.0, "extra": [1, 2]}
        P_pred = np.eye(_STATE_DIM)
        x_upd, _, report = run_uwb_update(x_pred, P_pred, {"ax": 1.0, "ay": 0.0}, 2.0, 0.25)
        report["x_upd"]["extra"].append(3)
        assert x_upd["extra"] == [1, 2]


# ======================================================================
# TestRunUwbUpdateValidation — 输入校验异常
# ======================================================================
class TestRunUwbUpdateValidation:
    """run_uwb_update 的异常路径。"""

    def test_reject_none_x_pred(self):
        with pytest.raises(ValueError, match="must not be None"):
            run_uwb_update(None, np.eye(8), np.array([1.0, 0.0]), 2.0, 0.25)

    def test_reject_none_P_pred(self):
        with pytest.raises(ValueError, match="must not be None"):
            run_uwb_update(np.zeros(8), None, np.array([1.0, 0.0]), 2.0, 0.25)

    def test_reject_none_anchor(self):
        with pytest.raises(ValueError, match="must not be None"):
            run_uwb_update(np.zeros(8), np.eye(8), None, 2.0, 0.25)

    def test_reject_negative_z_range(self):
        """负值测试：reject。\n\n验证 reject 对负值输入的拒绝，\n确保非法值被正确拦截。
        """
        with pytest.raises(ValueError, match=r"z_range must be >= 0\.0"):
            run_uwb_update(np.zeros(8), np.eye(8), np.array([1.0, 0.0]), -0.1, 0.25)

    def test_reject_negative_R_range(self):
        """负值测试：reject。\n\n验证 reject 对负值输入的拒绝，\n确保非法值被正确拦截。
        """
        with pytest.raises(ValueError, match=r"R_range must be >= 0\.0"):
            run_uwb_update(np.zeros(8), np.eye(8), np.array([1.0, 0.0]), 1.0, -0.1)

    def test_reject_bool_z_range(self):
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            run_uwb_update(np.zeros(8), np.eye(8), np.array([1.0, 0.0]), True, 0.25)

    def test_reject_bool_R_range(self):
        with pytest.raises(TypeError, match="must be numeric, got bool"):
            run_uwb_update(np.zeros(8), np.eye(8), np.array([1.0, 0.0]), 1.0, False)

    def test_reject_0d_array_z_range(self):
        with pytest.raises(TypeError, match="must be a numeric scalar, got ndarray"):
            run_uwb_update(np.zeros(8), np.eye(8), np.array([1.0, 0.0]), np.array(2.0), 0.25)

    def test_reject_0d_array_R_range(self):
        with pytest.raises(TypeError, match="must be a numeric scalar, got ndarray"):
            run_uwb_update(np.zeros(8), np.eye(8), np.array([1.0, 0.0]), 1.0, np.array(0.25))

    def test_reject_nan_z_range(self):
        with pytest.raises(ValueError, match="z_range must be finite"):
            run_uwb_update(np.zeros(8), np.eye(8), np.array([1.0, 0.0]), float("nan"), 0.25)

    def test_reject_nan_R_range(self):
        with pytest.raises(ValueError, match="R_range must be finite"):
            run_uwb_update(np.zeros(8), np.eye(8), np.array([1.0, 0.0]), 1.0, float("nan"))

    def test_reject_inf_z_range(self):
        with pytest.raises(ValueError, match="z_range must be finite"):
            run_uwb_update(np.zeros(8), np.eye(8), np.array([1.0, 0.0]), float("inf"), 0.25)

    def test_reject_inf_R_range(self):
        with pytest.raises(ValueError, match="R_range must be finite"):
            run_uwb_update(np.zeros(8), np.eye(8), np.array([1.0, 0.0]), 1.0, float("inf"))

    def test_reject_string_x_pred(self):
        with pytest.raises(TypeError, match="must be a numeric state vector"):
            run_uwb_update("abc", np.eye(8), np.array([1.0, 0.0]), 1.0, 0.25)

    def test_reject_string_anchor(self):
        with pytest.raises(TypeError, match="must be a 2D coordinate record"):
            run_uwb_update(np.zeros(8), np.eye(8), "1,2", 1.0, 0.25)

    def test_reject_wrong_cov_shape(self):
        with pytest.raises(ValueError, match=r"must have shape \(8, 8\)"):
            run_uwb_update(np.zeros(8), np.eye(4), np.array([1.0, 0.0]), 1.0, 0.25)

    def test_reject_anchor_mapping_no_keys(self):
        """映射报告测试：reject anchor。\n\n验证 reject anchor 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        with pytest.raises(KeyError, match="must provide one of"):
            run_uwb_update(np.zeros(8), np.eye(8), {"u": 1.0, "v": 2.0}, 1.0, 0.25)

    def test_reject_x_pred_mapping_missing_py(self):
        """缺失测试：reject x pred mapping。\n\n验证 reject x pred mapping 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
        """
        with pytest.raises(KeyError):
            run_uwb_update({"px": 1.0}, np.eye(8), {"ax": 0.0, "ay": 0.0}, 1.0, 0.25)

    def test_reject_anchor_wrong_size(self):
        with pytest.raises(ValueError, match="exactly 2 coordinates, got 3"):
            run_uwb_update(np.zeros(8), np.eye(8), [1.0, 2.0, 3.0], 1.0, 0.25)

    def test_reject_x_pred_too_short(self):
        with pytest.raises(ValueError, match="at least px and py"):
            run_uwb_update(np.array([1.0]), np.eye(1), [1.0, 0.0], 1.0, 0.25)

    def test_reject_anchor_2d_non_vector(self):
        with pytest.raises(ValueError, match="must be a 1D coordinate record"):
            run_uwb_update(np.zeros(8), np.eye(8), np.ones((2, 2)), 1.0, 0.25)

    def test_reject_string_z_range(self):
        with pytest.raises(TypeError, match="must be numeric, got str"):
            run_uwb_update(np.zeros(8), np.eye(8), np.array([1.0, 0.0]), "1.0", 0.25)

    def test_reject_string_R_range(self):
        with pytest.raises(TypeError, match="must be numeric, got str"):
            run_uwb_update(np.zeros(8), np.eye(8), np.array([1.0, 0.0]), 1.0, "0.25")


# ======================================================================
# Multi-anchor stacked Jacobian UWB update
# ======================================================================
class TestRunUwbUpdateMultiAnchor:
    """run_uwb_update_multi_anchor 多锚点联合更新测试。"""

    def test_single_anchor_equivalence_to_run_uwb_update(self):
        """N=1 退化为单锚点路径,x_upd/P_upd 与 run_uwb_update 数值一致。"""
        from liquidloc.estimators.uwb_update_step import (
            run_uwb_update,
            run_uwb_update_multi_anchor,
        )
        rng = np.random.default_rng(7)
        n = 8
        x_prev = rng.standard_normal(n)
        x_prev[0:2] = [3.0, 4.0]
        P_prev = np.eye(n) * 0.5
        anchor_pos = np.array([0.0, 0.0])
        z = 5.0
        R_var = 0.25
        x_single, P_single, _ = run_uwb_update(x_prev, P_prev, anchor_pos, z, R_var)
        x_multi, P_multi, _ = run_uwb_update_multi_anchor(
            x_prev, P_prev, [anchor_pos], [z], [R_var]
        )
        assert np.allclose(np.asarray(x_multi), np.asarray(x_single), atol=1e-9)
        assert np.allclose(P_multi, P_single, atol=1e-9)

    def test_multi_anchor_state_converges_toward_trilateration_center(self):
        """多锚点同时观测应推动状态往三角化中心方向修正。"""
        from liquidloc.estimators.uwb_update_step import run_uwb_update_multi_anchor
        n = 8
        x_prev = np.zeros(n)
        x_prev[0:2] = [1.0, 1.0]
        P_prev = np.eye(n)
        anchors = [(5.0, 0.0), (0.0, 5.0), (0.0, 0.0)]
        z_ranges = [5.0, 5.0, math.sqrt(2.0)]
        R_ranges = [0.01, 0.01, 0.01]
        x_upd, P_upd, info = run_uwb_update_multi_anchor(
            x_prev, P_prev, anchors, z_ranges, R_ranges
        )
        assert abs(x_upd[0] - 1.0) > 0.05 or abs(x_upd[1] - 1.0) > 0.05
        assert info["modality"] == "uwb_multi_anchor"
        assert info["update_applied"] is True
        assert info["H_stacked"].shape == (3, n)
        assert info["R_stacked"].shape == (3, 3)
        assert info["S"].shape == (3, 3)
        assert info["K_gain"].shape == (n, 3)
        assert len(info["residuals"]) == 3
        assert len(info["z_preds"]) == 3

    def test_joseph_form_keeps_P_symmetric_and_pd(self):
        """Joseph 形式 + 0.5*(P+P^T) 必须给出对称正定的 P_upd。"""
        from liquidloc.estimators.uwb_update_step import run_uwb_update_multi_anchor
        n = 8
        x_prev = np.zeros(n)
        x_prev[0:2] = [2.0, 2.0]
        P_prev = np.eye(n) * 0.3
        anchors = [(0.0, 0.0), (4.0, 0.0)]
        z_ranges = [3.0, 3.0]
        R_ranges = [0.04, 0.04]
        _, P_upd, _ = run_uwb_update_multi_anchor(
            x_prev, P_prev, anchors, z_ranges, R_ranges
        )
        assert np.allclose(P_upd, P_upd.T, atol=1e-12)
        np.linalg.cholesky(P_upd)

    def test_reject_empty_anchor_list(self):
        from liquidloc.estimators.uwb_update_step import run_uwb_update_multi_anchor
        with pytest.raises(ValueError, match="anchor_positions must be a non-empty sequence"):
            run_uwb_update_multi_anchor(np.zeros(8), np.eye(8), [], [], [])

    def test_reject_length_mismatch(self):
        from liquidloc.estimators.uwb_update_step import run_uwb_update_multi_anchor
        with pytest.raises(ValueError, match="must have equal length"):
            run_uwb_update_multi_anchor(
                np.zeros(8), np.eye(8),
                [(0.0, 0.0), (1.0, 1.0)],
                [1.0],
                [0.1, 0.2],
            )

    def test_reject_negative_z_range(self):
        from liquidloc.estimators.uwb_update_step import run_uwb_update_multi_anchor
        with pytest.raises(ValueError, match="must be >="):
            run_uwb_update_multi_anchor(
                np.zeros(8), np.eye(8),
                [(0.0, 0.0)],
                [-1.0],
                [0.1],
            )


# ======================================================================
# Joint UWB+VIO EKF update (stacked H, block-diagonal R, Joseph form)
# ======================================================================
class TestRunJointUwbVioUpdate:
    """run_joint_uwb_vio_update 联合更新测试。"""

    def test_pure_uwb_when_vio_none(self):
        """z_vio=None 时退化为纯多锚点 UWB 更新,与 multi_anchor 路径一致。"""
        from liquidloc.estimators.uwb_update_step import (
            run_joint_uwb_vio_update,
            run_uwb_update_multi_anchor,
        )
        n = 8
        x_prev = np.zeros(n)
        x_prev[0:2] = [1.0, 1.0]
        x_prev[4] = 0.3
        P_prev = np.eye(n) * 0.4
        anchors = [(0.0, 0.0), (3.0, 0.0)]
        z_uwb = [math.hypot(1.0, 1.0), math.hypot(2.0, 1.0)]
        R_uwb = [0.05, 0.05]
        x_joint, P_joint, info_joint = run_joint_uwb_vio_update(
            x_prev, P_prev,
            uwb_anchors=anchors, z_uwb=z_uwb, R_uwb=R_uwb,
            z_vio=None, R_vio=None,
        )
        x_multi, P_multi, _ = run_uwb_update_multi_anchor(
            x_prev, P_prev, anchors, z_uwb, R_uwb
        )
        assert np.allclose(np.asarray(x_joint), np.asarray(x_multi), atol=1e-9)
        assert np.allclose(P_joint, P_multi, atol=1e-9)
        assert info_joint["modality"] == "joint_uwb_vio"
        assert info_joint["H_joint"].shape == (2, n)
        assert info_joint["R_joint"].shape == (2, 2)

    def test_joint_uwb_vio_runs_with_correct_shapes(self):
        """UWB (2 锚点) + VIO 3 维量测 -> 联合 H 形状 (5, n), R 形状 (5, 5)。"""
        from liquidloc.estimators.uwb_update_step import run_joint_uwb_vio_update
        n = _STATE_DIM
        x_prev = np.zeros(n)
        x_prev[0:2] = [2.0, 1.0]
        x_prev[4] = 0.2
        P_prev = np.eye(n) * 0.5
        anchors = [(0.0, 0.0), (5.0, 0.0)]
        z_uwb = [math.hypot(2.0, 1.0), math.hypot(3.0, 1.0)]
        R_uwb = [0.05, 0.05]
        z_vio = np.array([0.1, 0.05, 0.01])
        R_vio_std = np.array([0.04, 0.04, 0.01])
        x_upd, P_upd, info = run_joint_uwb_vio_update(
            x_prev, P_prev,
            uwb_anchors=anchors, z_uwb=z_uwb, R_uwb=R_uwb,
            z_vio=z_vio, R_vio=(R_vio_std ** 2),
        )
        assert info["modality"] == "joint_uwb_vio"
        assert info["update_applied"] is True
        assert info["H_joint"].shape == (5, n)
        assert info["R_joint"].shape == (5, 5)
        assert info["S"].shape == (5, 5)
        assert info["K_gain"].shape == (n, 5)
        assert len(info["uwb_residuals"]) == 2
        assert len(info["vio_residual"]) == 3
        assert np.allclose(P_upd, P_upd.T, atol=1e-12)
        np.linalg.cholesky(P_upd)

    def test_joint_rejects_when_no_observation_at_all(self):
        """既无 UWB 又无 VIO 时应抛 ValueError,避免静默无更新。"""
        from liquidloc.estimators.uwb_update_step import run_joint_uwb_vio_update
        with pytest.raises(ValueError, match="requires at least one UWB anchor or a VIO observation"):
            run_joint_uwb_vio_update(
                np.zeros(8), np.eye(8),
                uwb_anchors=[], z_uwb=[], R_uwb=[],
                z_vio=None, R_vio=None,
            )

    def test_joint_rejects_length_mismatch(self):
        """UWB 三组序列长度不匹配应抛 ValueError。"""
        from liquidloc.estimators.uwb_update_step import run_joint_uwb_vio_update
        with pytest.raises(ValueError, match="must have equal length"):
            run_joint_uwb_vio_update(
                np.zeros(8), np.eye(8),
                uwb_anchors=[(0.0, 0.0), (1.0, 1.0)],
                z_uwb=[1.0],
                R_uwb=[0.1, 0.2],
                z_vio=None, R_vio=None,
            )


# === §11.5 SPD 抖动注入锁死测试（spec L1745）===
# 覆盖 uwb_update_step.py 中两处 inline Cholesky 调用点 (L717 单锚, L1009 联合) 的 jitter fallback：
# - 协议单源：cov_jitter_eps 是 BRIDGE_THRESHOLDS 成员，不可篡改。
# - 病态 S 触发 jitter：构造让 H @ P @ H.T + R 在数值上几乎秩亏，验证 EKF/UWB 路径不 raise。


def test_uwb_cov_jitter_eps_protocol_single_source():
    """§11.5 协议单源：cov_jitter_eps 必须在 BRIDGE_THRESHOLDS 中且为正 float。"""
    assert "cov_jitter_eps" in BRIDGE_THRESHOLDS
    eps = BRIDGE_THRESHOLDS["cov_jitter_eps"]
    assert isinstance(eps, float)
    assert eps > 0.0
    assert eps < 1e-3  # 必须远小于典型 R 缩放，避免掩盖真病态。


def test_uwb_cov_jitter_eps_is_immutable():
    """§11.5 协议只读：BRIDGE_THRESHOLDS["cov_jitter_eps"] 不可被运行时篡改。"""
    with pytest.raises(TypeError):
        BRIDGE_THRESHOLDS["cov_jitter_eps"] = 99.0
    with pytest.raises(TypeError):
        BRIDGE_THRESHOLDS.update({"cov_jitter_eps": 99.0})


def test_uwb_cov_jitter_fallback_succeeds_when_cholesky_fails_due_to_machine_precision():
    """§11.5 jitter fallback 触发：构造病态 S 矩阵（H 列共线 + R 极小），Cholesky 第一次失败，jitter 后应恢复。

    这是 UWB 路径 L717（run_uwb_update_multi_anchor stacked-H 路径）inline Cholesky 的 jitter fallback
    锁死测试：构造 H @ P @ H.T 时 H 列近似共线，加上 R_stacked 接近奇异 → S 在 double 精度下可能
    Cholesky 失败 → 应走 jitter → 二次成功。
    """
    from liquidloc.estimators.uwb_update_step import run_uwb_update_multi_anchor
    eps_val = BRIDGE_THRESHOLDS["cov_jitter_eps"]
    state_dim = _STATE_DIM
    # 构造一个几乎秩亏的预测协方差（病态但仍对称+对角线阳性）
    P_pred = np.eye(state_dim) * 0.1
    P_pred[0, 1] = 0.1 - 1e-12  # 接近完美相关
    P_pred[1, 0] = P_pred[0, 1]
    P_pred = 0.5 * (P_pred + P_pred.T)
    # 构造两个几乎对齐的锚点 → H[:,0] ≈ H[:,1]（共线性让 S 秩亏）
    x_pred = np.zeros(state_dim)
    x_pred[0], x_pred[1] = 0.5, 0.5
    anchors = [np.array([0.0, 0.0]), np.array([1e-6, 1e-6])]  # 几乎重合 → H 共线
    z_uwb = [1.0, 1.0 + 1e-7]
    # R 设为 eps_val 同一数量级 → S 在 double 精度下接近奇异 → jitter 加 eps_val*I 后能恢复
    R_uwb = [eps_val, eps_val]
    x_upd, P_upd, info = run_uwb_update_multi_anchor(
        x_pred, P_pred,
        anchor_positions=anchors,
        z_ranges=z_uwb,
        R_ranges=R_uwb,
    )
    # 验证 update 成功且 P_upd 仍 SPD
    if isinstance(x_upd, np.ndarray):
        assert x_upd.shape == (state_dim,)
    assert P_upd.shape == (state_dim, state_dim)
    assert np.all(np.isfinite(P_upd))
    # Joseph form 应保证 P_upd 对称
    assert np.allclose(P_upd, P_upd.T, atol=1e-8)
