"""IMU 事件提取与校验模块测试。

覆盖范围：
    - coerce_finite_scalar：类型校验、有限性校验、边界值
    - _as_imu_event_dict：Event/dict 输入、非法类型
    - validate_imu_event：正常校验、模态不匹配、payload 缺失、字段缺失、字段非法
    - extract_imu_measurement：正常提取、返回值类型、与 validate 一致性
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from liquidloc.protocol.event_schema import Event
from liquidloc.common.validation import coerce_finite_scalar
from liquidloc.sensors.imu_model import (
    _as_imu_event_dict,
    extract_imu_measurement,
    validate_imu_event,
)


# ── 辅助工具 ──────────────────────────────────────────────


def _valid_imu_dict(**overrides) -> dict:
    """构造一个合法的 IMU 事件字典。"""
    event = {
        "t": 0.0,
        "dt": 0.01,
        "modality": "imu",
        "meta": {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "seq_001"},
        "imu_payload": {"ax": 0.1, "ay": -0.2, "gz": 0.03},
    }
    event.update(overrides)
    return event


def _valid_imu_event(**overrides) -> Event:
    """构造一个合法的 IMU Event 对象。"""
    payload = {"ax": 0.1, "ay": -0.2, "gz": 0.03}
    payload.update(overrides.pop("imu_payload", {}))
    return Event(
        t=overrides.pop("t", 0.0),
        dt=overrides.pop("dt", 0.01),
        modality="imu",
        meta=overrides.pop("meta", {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "seq_001"}),
        imu_payload=payload,
    )


# ── coerce_finite_scalar ────────────────────────────


class TestCoerceFiniteImuScalar:
    """coerce_finite_scalar 单元测试。"""

    def test_int_input_returns_float(self):
        """整数输入返回浮点数。"""
        assert coerce_finite_scalar(1, name="test") == 1.0
        assert isinstance(coerce_finite_scalar(1, name="test"), float)

    def test_float_input_returns_same(self):
        """浮点输入原样返回。"""
        assert coerce_finite_scalar(3.14, name="test") == pytest.approx(3.14)

    def test_numpy_scalar_accepted(self):
        """numpy 标量（np.float64）被接受。"""
        assert coerce_finite_scalar(np.float64(2.5), name="test") == pytest.approx(2.5)

    def test_numpy_int_accepted(self):
        """numpy 整数标量被接受。"""
        assert coerce_finite_scalar(np.int64(3), name="test") == 3.0

    def test_negative_value_accepted(self):
        """负数值被接受（IMU 加速度可以为负）。"""
        assert coerce_finite_scalar(-9.8, name="test") == pytest.approx(-9.8)

    def test_zero_accepted(self):
        """零被接受。"""
        assert coerce_finite_scalar(0.0, name="test") == 0.0

    def test_rejects_bool_true(self):
        """bool True 被拒绝。"""
        with pytest.raises(TypeError, match="must be numeric"):
            coerce_finite_scalar(True, name="test")

    def test_rejects_bool_false(self):
        """bool False 被拒绝。"""
        with pytest.raises(TypeError, match="must be numeric"):
            coerce_finite_scalar(False, name="test")

    def test_rejects_string(self):
        """字符串被拒绝。"""
        with pytest.raises(TypeError, match="must be numeric"):
            coerce_finite_scalar("1.0", name="test")

    def test_rejects_list(self):
        """列表被拒绝。"""
        with pytest.raises(TypeError, match="must be numeric"):
            coerce_finite_scalar([1.0], name="test")

    def test_rejects_nan(self):
        """NaN 被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            coerce_finite_scalar(float("nan"), name="test")

    def test_rejects_inf(self):
        """正无穷被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            coerce_finite_scalar(float("inf"), name="test")

    def test_rejects_neg_inf(self):
        """负无穷被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            coerce_finite_scalar(float("-inf"), name="test")

    def test_rejects_numpy_nan(self):
        """numpy NaN 被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            coerce_finite_scalar(np.float64("nan"), name="test")

    def test_rejects_numpy_inf(self):
        """numpy inf 被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            coerce_finite_scalar(np.float64("inf"), name="test")

    def test_large_value_accepted(self):
        """大数值被接受（IMU 理论上可以有极端加速度）。"""
        assert coerce_finite_scalar(1e10, name="test") == pytest.approx(1e10)


# ── _as_imu_event_dict ──────────────────────────────────


class TestAsImuEventDict:
    """_as_imu_event_dict 单元测试。"""

    def test_event_returns_dict(self):
        """Event 对象返回字典。"""
        event = _valid_imu_event()
        result = _as_imu_event_dict(event)
        assert isinstance(result, dict)
        assert result["modality"] == "imu"

    def test_dict_returns_same_reference(self):
        """dict 输入返回原引用（不做拷贝）。"""
        d = _valid_imu_dict()
        result = _as_imu_event_dict(d)
        assert result is d  # 同一对象。

    def test_event_dict_has_payload(self):
        """Event 转换后的字典包含 imu_payload。"""
        event = _valid_imu_event()
        result = _as_imu_event_dict(event)
        assert "imu_payload" in result
        assert result["imu_payload"]["ax"] == pytest.approx(0.1)

    def test_rejects_string_input(self):
        """字符串输入被拒绝。"""
        with pytest.raises(TypeError, match="must be an Event or dict"):
            _as_imu_event_dict("not_an_event")

    def test_rejects_int_input(self):
        """整数输入被拒绝。"""
        with pytest.raises(TypeError, match="must be an Event or dict"):
            _as_imu_event_dict(42)

    def test_rejects_none_input(self):
        """None 输入被拒绝。"""
        with pytest.raises(TypeError, match="must be an Event or dict"):
            _as_imu_event_dict(None)

    def test_rejects_list_input(self):
        """列表输入被拒绝。"""
        with pytest.raises(TypeError, match="must be an Event or dict"):
            _as_imu_event_dict([1, 2, 3])


# ── validate_imu_event ──────────────────────────────────


class TestValidateImuEvent:
    """validate_imu_event 单元测试。"""

    def test_valid_dict_passes(self):
        """合法字典事件通过校验。"""
        validate_imu_event(_valid_imu_dict())  # 不抛异常即可。

    def test_valid_event_passes(self):
        """合法 Event 对象通过校验。"""
        validate_imu_event(_valid_imu_event())  # 不抛异常即可。

    def test_wrong_modality_rejected(self):
        """模态不是 'imu' 时被拒绝。"""
        event = _valid_imu_dict(modality="uwb", imu_payload=None, uwb_payload={"anchor_id": "A1", "range": 1.0, "quality": 0.9, "valid": True})
        with pytest.raises(ValueError, match="modality must be 'imu'"):
            validate_imu_event(event)

    def test_missing_payload_rejected(self):
        """缺少 imu_payload 时被拒绝。"""
        event = _valid_imu_dict()
        del event["imu_payload"]
        with pytest.raises(ValueError, match="must not be None"):
            validate_imu_event(event)

    def test_none_payload_rejected(self):
        """imu_payload 为 None 时被拒绝。"""
        event = _valid_imu_dict(imu_payload=None)
        with pytest.raises(ValueError, match="must not be None"):
            validate_imu_event(event)

    def test_missing_ax_rejected(self):
        """缺少 ax 字段时被拒绝。"""
        event = _valid_imu_dict(imu_payload={"ay": 0.1, "gz": 0.0})
        with pytest.raises(KeyError, match="ax"):
            validate_imu_event(event)

    def test_missing_ay_rejected(self):
        """缺少 ay 字段时被拒绝。"""
        event = _valid_imu_dict(imu_payload={"ax": 0.1, "gz": 0.0})
        with pytest.raises(KeyError, match="ay"):
            validate_imu_event(event)

    def test_missing_gz_rejected(self):
        """缺少 gz 字段时被拒绝。"""
        event = _valid_imu_dict(imu_payload={"ax": 0.1, "ay": 0.0})
        with pytest.raises(KeyError, match="gz"):
            validate_imu_event(event)

    def test_bool_ax_rejected(self):
        """ax 为 bool 时被拒绝。"""
        event = _valid_imu_dict(imu_payload={"ax": True, "ay": 0.0, "gz": 0.0})
        with pytest.raises(TypeError, match="must be numeric"):
            validate_imu_event(event)

    def test_nan_ax_rejected(self):
        """ax 为 NaN 时被拒绝。"""
        event = _valid_imu_dict(imu_payload={"ax": float("nan"), "ay": 0.0, "gz": 0.0})
        with pytest.raises(ValueError, match="must be finite"):
            validate_imu_event(event)

    def test_inf_ay_rejected(self):
        """ay 为 inf 时被拒绝。"""
        event = _valid_imu_dict(imu_payload={"ax": 0.0, "ay": float("inf"), "gz": 0.0})
        with pytest.raises(ValueError, match="must be finite"):
            validate_imu_event(event)

    def test_string_gz_rejected(self):
        """gz 为字符串时被拒绝。"""
        event = _valid_imu_dict(imu_payload={"ax": 0.0, "ay": 0.0, "gz": "0.03"})
        with pytest.raises(TypeError, match="must be numeric"):
            validate_imu_event(event)

    def test_numpy_scalar_ax_accepted(self):
        """ax 为 numpy 标量时通过校验。"""
        event = _valid_imu_dict(imu_payload={"ax": np.float64(0.1), "ay": 0.0, "gz": 0.0})
        validate_imu_event(event)  # 不抛异常即可。

    def test_negative_values_accepted(self):
        """负数值通过校验（IMU 加速度/角速度可以为负）。"""
        event = _valid_imu_dict(imu_payload={"ax": -9.8, "ay": -0.5, "gz": -3.14})
        validate_imu_event(event)  # 不抛异常即可。


# ── extract_imu_measurement ─────────────────────────────


class TestExtractImuMeasurement:
    """extract_imu_measurement 单元测试。"""

    def test_dict_input_returns_correct_dict(self):
        """字典输入返回正确的测量字典。"""
        event = _valid_imu_dict()
        result = extract_imu_measurement(event)
        assert result == {"ax": pytest.approx(0.1), "ay": pytest.approx(-0.2), "gz": pytest.approx(0.03)}

    def test_event_input_returns_correct_dict(self):
        """Event 对象输入返回正确的测量字典。"""
        event = _valid_imu_event()
        result = extract_imu_measurement(event)
        assert result == {"ax": pytest.approx(0.1), "ay": pytest.approx(-0.2), "gz": pytest.approx(0.03)}

    def test_return_values_are_float(self):
        """返回值的类型是 float。"""
        event = _valid_imu_dict(imu_payload={"ax": 1, "ay": -2, "gz": 3})
        result = extract_imu_measurement(event)
        assert isinstance(result["ax"], float)
        assert isinstance(result["ay"], float)
        assert isinstance(result["gz"], float)

    def test_numpy_scalar_converted_to_float(self):
        """numpy 标量被转换为 Python float。"""
        event = _valid_imu_dict(imu_payload={"ax": np.float64(1.5), "ay": np.float32(-0.5), "gz": np.int64(2)})
        result = extract_imu_measurement(event)
        assert isinstance(result["ax"], float)
        assert isinstance(result["ay"], float)
        assert isinstance(result["gz"], float)
        assert result["ax"] == pytest.approx(1.5)
        assert result["gz"] == 2.0

    def test_invalid_event_raises(self):
        """非法事件抛出异常。"""
        event = _valid_imu_dict(modality="uwb", imu_payload=None, uwb_payload={"anchor_id": "A1", "range": 1.0, "quality": 0.9, "valid": True})
        with pytest.raises(ValueError, match="modality must be 'imu'"):
            extract_imu_measurement(event)

    def test_missing_gz_raises(self):
        """缺少 gz 字段抛出 KeyError。"""
        event = _valid_imu_dict(imu_payload={"ax": 0.1, "ay": -0.2})
        with pytest.raises(KeyError, match="gz"):
            extract_imu_measurement(event)

    def test_nan_field_raises(self):
        """字段为 NaN 抛出 ValueError。"""
        event = _valid_imu_dict(imu_payload={"ax": float("nan"), "ay": 0.0, "gz": 0.0})
        with pytest.raises(ValueError, match="must be finite"):
            extract_imu_measurement(event)

    def test_negative_values_extracted_correctly(self):
        """负数值正确提取。"""
        event = _valid_imu_dict(imu_payload={"ax": -9.8, "ay": -0.5, "gz": -3.14})
        result = extract_imu_measurement(event)
        assert result["ax"] == pytest.approx(-9.8)
        assert result["ay"] == pytest.approx(-0.5)
        assert result["gz"] == pytest.approx(-3.14)

    def test_zero_values_extracted_correctly(self):
        """零值正确提取。"""
        event = _valid_imu_dict(imu_payload={"ax": 0.0, "ay": 0.0, "gz": 0.0})
        result = extract_imu_measurement(event)
        assert result["ax"] == 0.0
        assert result["ay"] == 0.0
        assert result["gz"] == 0.0

    def test_large_values_extracted_correctly(self):
        """大数值正确提取（在物理范围约束内）。"""
        event = _valid_imu_dict(imu_payload={"ax": 40.0, "ay": -40.0, "gz": 5.0})
        result = extract_imu_measurement(event)
        assert result["ax"] == pytest.approx(40.0)
        assert result["ay"] == pytest.approx(-40.0)
        assert result["gz"] == pytest.approx(5.0)

    def test_return_dict_has_exactly_three_keys(self):
        """返回字典恰好有三个键。"""
        event = _valid_imu_dict()
        result = extract_imu_measurement(event)
        assert set(result.keys()) == {"ax", "ay", "gz"}

    def test_does_not_modify_input_dict(self):
        """不修改输入字典中的原始值。"""
        original = _valid_imu_dict()
        original_payload = dict(original["imu_payload"])
        extract_imu_measurement(original)
        assert original["imu_payload"] == original_payload  # 原始数据未被修改。
