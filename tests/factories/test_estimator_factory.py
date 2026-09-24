from __future__ import annotations

"""估计器工厂（estimator_factory）测试模块。

测试覆盖范围：
- EKF 估计器的创建
- 未知估计器类型的拒绝
- 配置参数的传递与验证

被测模块：liquidloc.factories.estimator_factory"""

import numpy as np
import pytest

from liquidloc.common.constants import STATE_ITEMS as _STATE_ITEMS
from liquidloc.estimators.ekf_core import EKFCore
from liquidloc.factories.estimator_factory import (
    _require_mapping,
    _require_numeric_scalar,
    create_estimator,
)


# 紧耦合扩维后 8→10 维 (px, py, vx, vy, yaw, bax, bay, bg, uwb_clock_bias, vio_scale),
# Bug 2 删 uwb_anchor_bias. 用 len(_STATE_ITEMS) 自动对齐 src/ 維度, 避免再硬编码 8.
_STATE_DIM = len(_STATE_ITEMS)


def _anchor_layout():
    return {
        "anchor_ids": [0, 1],
        "anchor_positions": [(1.0, 0.0), (2.0, 0.0)],
    }


def _measurement_noise_cfg():
    return {
        "uwb": 0.25,
        "vio": {"pos": 0.08, "yaw": 0.03},
    }


def _cfg(estimator_name: str):
    cfg = {
        "process_noise": {
            "pos": 0.05,
            "vel": 0.10,
            "yaw": 0.02,
            "accel_bias": 0.001,
            "gyro_bias": 0.001,
            # 紧耦合扩维新增的过程噪声键 (8→10, Bug 2 删 uwb_anchor_bias).
            # 必须同时给 uwb_clock_bias 与 vio_scale, 否则 estimator 校验会
            # 抛 "missing required keys" (与 configs/models/*.yaml init_state 对齐).
            "uwb_clock_bias": 0.001,
            "vio_scale": 0.001,
        },
        "measurement_noise": _measurement_noise_cfg(),
        "anchor_layout": _anchor_layout(),
        "init_state": {
            "px": 0.0,
            "py": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "yaw": 0.0,
            "bax": 0.0,
            "bay": 0.0,
            "bg": 0.0,
            # 紧耦合扩维新增状态项 (8→10, Bug 2 删 uwb_anchor_bias), 与 configs/base/task.yaml state_items 一致.
            # vio_scale 初值取 1.0 (与 configs/models/*.yaml init_state 对齐, 无量纲因子默认 1).
            "uwb_clock_bias": 0.0,
            "vio_scale": 1.0,
        },
        "init_cov": [1.0] * _STATE_DIM,
    }
    return cfg


def _uwb_event(quality=0.55):
    return {
        "t": 0.05,
        "dt": 0.05,
        "modality": "uwb",
        "meta": {"scene_id": "S(A2,N2,V2,K0,M0)", "seq_id": "mini_seq"},
        "imu_payload": None,
        "uwb_payload": {"anchor_id": 0, "range": 0.8, "valid": True, "quality": quality},
        "vio_payload": None,
    }


@pytest.mark.parametrize(
    ("estimator_name", "expected_type"),
    [
        ("ekf", EKFCore),
    ],
)
def test_normal_case(estimator_name, expected_type):
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    estimator = create_estimator(estimator_name, _cfg(estimator_name))
    assert isinstance(estimator, expected_type)
    state = estimator.step(_uwb_event(quality=0.9))
    assert state.timestamp == 0.05


def test_lstm_ekf_is_rejected_by_closed_competitor_set():
    """§16.1 封闭对手集：create_estimator 仅承认 ekf / sgpr。

    lstm_ekf 已从工厂封闭集移除（纯 SGPR 占位保持集合封闭），此处显式断言
    'lstm_ekf' 不再是可创建的估计器身份，防止游离方法名混入主表。
    """
    with pytest.raises(ValueError, match="Unknown estimator: lstm_ekf"):
        create_estimator("lstm_ekf", _cfg("lstm_ekf"))


def test_boundary_case():
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    # 这里测试标准EKF的边界情况
    estimator = create_estimator("ekf", _cfg("ekf"))
    state = estimator.step(_uwb_event(quality=0.9))
    assert state.timestamp == 0.05


def test_invalid_case():
    """无效输入测试。

    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    with pytest.raises(ValueError, match="Unknown estimator"):
        create_estimator("bad_name", {})


# ---- _require_numeric_scalar 复数拒绝 ----


def test_require_numeric_scalar_rejects_python_complex():
    """拒绝测试：require numeric scalar。\n\n验证被测功能对 require numeric scalar 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(TypeError, match="must be numeric"):
        _require_numeric_scalar(complex(3.0), name="test_val")


def test_require_numeric_scalar_rejects_numpy_complex():
    """拒绝测试：require numeric scalar。\n\n验证被测功能对 require numeric scalar 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(TypeError, match="must be numeric"):
        _require_numeric_scalar(np.complex128(3.0), name="test_val")


def test_require_numeric_scalar_rejects_bool():
    """拒绝测试：require numeric scalar。\n\n验证被测功能对 require numeric scalar 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(TypeError, match="must be numeric"):
        _require_numeric_scalar(True, name="test_val")


def test_require_numeric_scalar_rejects_string():
    """拒绝测试：require numeric scalar。\n\n验证被测功能对 require numeric scalar 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(TypeError, match="must be numeric"):
        _require_numeric_scalar("1.0", name="test_val")


def test_require_numeric_scalar_accepts_int():
    """接受测试：require numeric scalar。\n\n验证 require numeric scalar 的接受行为，\n确保合法输入被正确处理。
    """
    assert _require_numeric_scalar(3, name="test_val") == 3.0


def test_require_numeric_scalar_accepts_float():
    """接受测试：require numeric scalar。\n\n验证 require numeric scalar 的接受行为，\n确保合法输入被正确处理。
    """
    assert _require_numeric_scalar(3.5, name="test_val") == 3.5


def test_require_numeric_scalar_accepts_numpy_float():
    """接受测试：require numeric scalar。\n\n验证 require numeric scalar 的接受行为，\n确保合法输入被正确处理。
    """
    assert _require_numeric_scalar(np.float64(2.5), name="test_val") == 2.5


# ---- _require_mapping 边界 ----


def test_require_mapping_rejects_list():
    """拒绝测试：require mapping。\n\n验证被测功能对 require mapping 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(TypeError, match="must be a mapping"):
        _require_mapping([1, 2, 3], name="test_val")


def test_require_mapping_accepts_dict():
    """接受测试：require mapping。\n\n验证 require mapping 的接受行为，\n确保合法输入被正确处理。
    """
    result = _require_mapping({"a": 1}, name="test_val")
    assert result == {"a": 1}


# ---- 配置中复数值渗透到 process_noise / init_state / init_cov ----


def test_process_noise_rejects_complex():
    """拒绝测试：process noise。\n\n验证被测功能对 process noise 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    cfg = _cfg("ekf")
    cfg["process_noise"]["pos"] = complex(0.05)
    with pytest.raises(TypeError, match="must be numeric"):
        create_estimator("ekf", cfg)


def test_init_state_rejects_complex():
    """拒绝测试：init state。\n\n验证被测功能对 init state 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    cfg = _cfg("ekf")
    cfg["init_state"]["px"] = complex(0.0)
    with pytest.raises(TypeError, match="must be numeric"):
        create_estimator("ekf", cfg)


def test_init_cov_rejects_complex():
    """拒绝测试：init cov。\n\n验证被测功能对 init cov 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    cfg = _cfg("ekf")
    cfg["init_cov"][0] = complex(1.0)
    with pytest.raises(TypeError, match="must be numeric"):
        create_estimator("ekf", cfg)


def test_measurement_noise_uwb_rejects_complex():
    """拒绝测试：measurement noise uwb。\n\n验证被测功能对 measurement noise uwb 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    cfg = _cfg("ekf")
    cfg["measurement_noise"]["uwb"] = complex(0.25)
    with pytest.raises(TypeError, match="must be numeric"):
        create_estimator("ekf", cfg)


def test_measurement_noise_vio_dx_dy_dyaw_is_allowed():
    cfg = _cfg("ekf")
    cfg["measurement_noise"]["vio"] = {"dx": 0.08, "dy": 0.08, "dyaw": 0.03}
    estimator = create_estimator("ekf", cfg)
    assert isinstance(estimator, EKFCore)


def test_measurement_noise_vio_missing_contract_key_rejected():
    """拒绝测试：measurement noise vio missing contract key。\n\n验证被测功能对不合法的 measurement noise vio missing contract key 输入正确抛出异常，\n防止无效参数通过验证。
    """
    cfg = _cfg("ekf")
    cfg["measurement_noise"]["vio"] = {"dx": 0.08, "dy": 0.08}
    with pytest.raises(ValueError, match="must provide exactly one complete VIO noise scheme"):
        create_estimator("ekf", cfg)


# ---- estimator_cfg 非 mapping ----


def test_estimator_cfg_rejects_non_mapping():
    """拒绝测试：estimator cfg。\n\n验证被测功能对 estimator cfg 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(TypeError, match="must be a mapping"):
        create_estimator("ekf", "not_a_dict")