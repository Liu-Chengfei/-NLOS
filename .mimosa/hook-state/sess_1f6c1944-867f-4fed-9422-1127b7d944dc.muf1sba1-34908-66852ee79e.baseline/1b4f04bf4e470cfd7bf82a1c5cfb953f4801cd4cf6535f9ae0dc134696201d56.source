"""UWB 事件提取与几何预测模块测试。

覆盖范围：
    - _coerce_nonnegative_finite_scalar：类型校验、有限性校验、非负校验
    - extract_uwb_measurement：正常提取、模态不匹配、payload 缺失/缺字段、
      range 非法、valid 类型非法、quality 归一化
    - predict_range_to_anchor：正常预测、零距离、坐标类型兼容、维度检查、
      NaN/inf 边界行为
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from liquidloc.protocol.event_schema import Event
from liquidloc.sensors.uwb_model import (
    _coerce_nonnegative_finite_scalar,
    extract_uwb_measurement,
    predict_range_to_anchor,
)


# ── 辅助工具 ──────────────────────────────────────────────


def _valid_uwb_dict(**overrides) -> dict:
    """构造一个合法的 UWB 事件字典。"""
    event = {
        "t": 0.2,
        "dt": 0.1,
        "modality": "uwb",
        "meta": {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "seq_001"},
        "uwb_payload": {
            "anchor_id": "a1",
            "range": 5.0,
            "valid": True,
            "quality": 0.75,
        },
    }
    event.update(overrides)
    return event


def _valid_uwb_event(**overrides) -> Event:
    """构造一个合法的 UWB Event 对象。"""
    payload = {
        "anchor_id": "a1",
        "range": 5.0,
        "valid": True,
        "quality": 0.75,
    }
    payload.update(overrides.pop("uwb_payload", {}))
    return Event(
        t=overrides.pop("t", 0.2),
        dt=overrides.pop("dt", 0.1),
        modality="uwb",
        meta=overrides.pop("meta", {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "seq_001"}),
        uwb_payload=payload,
    )


# ── _coerce_nonnegative_finite_scalar ────────────────────


class TestCoerceNonnegativeFiniteScalar:
    """_coerce_nonnegative_finite_scalar 单元测试。"""

    def test_int_returns_float(self):
        """整数输入返回浮点数。"""
        assert _coerce_nonnegative_finite_scalar(1, name="test") == 1.0
        assert isinstance(_coerce_nonnegative_finite_scalar(1, name="test"), float)

    def test_float_returns_same(self):
        """浮点输入原样返回。"""
        assert _coerce_nonnegative_finite_scalar(3.14, name="test") == pytest.approx(3.14)

    def test_numpy_scalar_accepted(self):
        """numpy 标量被接受。"""
        assert _coerce_nonnegative_finite_scalar(np.float64(2.5), name="test") == pytest.approx(2.5)

    def test_zero_accepted(self):
        """零被接受。"""
        assert _coerce_nonnegative_finite_scalar(0.0, name="test") == 0.0

    def test_rejects_negative(self):
        """负值被拒绝。"""
        with pytest.raises(ValueError, match="must be >= 0.0"):
            _coerce_nonnegative_finite_scalar(-0.1, name="test")

    def test_rejects_bool_true(self):
        """bool True 被拒绝。"""
        with pytest.raises(TypeError, match="must be numeric"):
            _coerce_nonnegative_finite_scalar(True, name="test")

    def test_rejects_bool_false(self):
        """bool False 被拒绝。"""
        with pytest.raises(TypeError, match="must be numeric"):
            _coerce_nonnegative_finite_scalar(False, name="test")

    def test_rejects_string(self):
        """字符串被拒绝。"""
        with pytest.raises(TypeError, match="must be numeric"):
            _coerce_nonnegative_finite_scalar("5.0", name="test")

    def test_rejects_nan(self):
        """NaN 被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_nonnegative_finite_scalar(float("nan"), name="test")

    def test_rejects_inf(self):
        """正无穷被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_nonnegative_finite_scalar(float("inf"), name="test")

    def test_rejects_neg_inf(self):
        """负无穷被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_nonnegative_finite_scalar(float("-inf"), name="test")

    def test_rejects_numpy_nan(self):
        """numpy NaN 被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_nonnegative_finite_scalar(np.float64("nan"), name="test")

    def test_rejects_numpy_inf(self):
        """numpy inf 被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_nonnegative_finite_scalar(np.float64("inf"), name="test")

    def test_large_value_accepted(self):
        """大数值被接受。"""
        assert _coerce_nonnegative_finite_scalar(1e10, name="test") == pytest.approx(1e10)


# ── extract_uwb_measurement ─────────────────────────────


class TestExtractUwbMeasurement:
    """extract_uwb_measurement 单元测试。"""

    def test_dict_input_returns_correct_dict(self):
        """字典输入返回正确的测量字典。"""
        event = _valid_uwb_dict()
        result = extract_uwb_measurement(event)
        assert result["anchor_id"] == "a1"
        assert result["measured_range"] == pytest.approx(5.0)
        assert result["valid"] is True
        assert result["quality"] == pytest.approx(0.75)

    def test_event_input_returns_correct_dict(self):
        """Event 对象输入返回正确的测量字典。"""
        event = _valid_uwb_event()
        result = extract_uwb_measurement(event)
        assert result["anchor_id"] == "a1"
        assert result["measured_range"] == pytest.approx(5.0)
        assert result["valid"] is True
        assert result["quality"] == pytest.approx(0.75)

    def test_return_dict_has_exactly_four_keys(self):
        """返回字典恰好有四个键。"""
        event = _valid_uwb_dict()
        result = extract_uwb_measurement(event)
        assert set(result.keys()) == {"anchor_id", "measured_range", "valid", "quality"}

    def test_measured_range_is_float(self):
        """measured_range 是 float 类型。"""
        event = _valid_uwb_dict(uwb_payload={"anchor_id": "a1", "range": 5, "valid": True, "quality": 0.75})
        result = extract_uwb_measurement(event)
        assert isinstance(result["measured_range"], float)

    def test_quality_is_float(self):
        """quality 是 float 类型。"""
        event = _valid_uwb_dict(uwb_payload={"anchor_id": "a1", "range": 5.0, "valid": True, "quality": 1})
        result = extract_uwb_measurement(event)
        assert isinstance(result["quality"], float)

    def test_valid_false_preserved(self):
        """valid=False 被保留。"""
        event = _valid_uwb_event(uwb_payload={"anchor_id": "a1", "range": 5.0, "valid": False, "quality": 0.75})
        result = extract_uwb_measurement(event)
        assert result["valid"] is False

    def test_zero_range_accepted(self):
        """range=0.0 被接受。"""
        event = _valid_uwb_event(uwb_payload={"anchor_id": "a1", "range": 0.0, "valid": True, "quality": 0.5})
        result = extract_uwb_measurement(event)
        assert result["measured_range"] == 0.0

    def test_none_input_rejected(self):
        """None 输入被拒绝。"""
        with pytest.raises(ValueError, match="must not be None"):
            extract_uwb_measurement(None)

    def test_wrong_modality_rejected(self):
        """模态不是 'uwb' 时被拒绝。"""
        event = _valid_uwb_dict(modality="imu")
        with pytest.raises(ValueError, match="must not be None|Unsupported modality|modality must be 'uwb'"):
            extract_uwb_measurement(event)

    def test_missing_payload_rejected(self):
        """缺少 uwb_payload 时被拒绝。"""
        event = _valid_uwb_dict()
        del event["uwb_payload"]
        with pytest.raises((ValueError, KeyError)):
            extract_uwb_measurement(event)

    def test_missing_anchor_id_rejected(self):
        """缺少 anchor_id 时被拒绝。"""
        payload = {"range": 5.0, "valid": True, "quality": 0.75}
        event = _valid_uwb_dict(uwb_payload=payload)
        with pytest.raises(KeyError, match="anchor_id"):
            extract_uwb_measurement(event)

    def test_missing_range_rejected(self):
        """缺少 range 时被拒绝。"""
        payload = {"anchor_id": "a1", "valid": True, "quality": 0.75}
        event = _valid_uwb_dict(uwb_payload=payload)
        with pytest.raises(KeyError, match="range"):
            extract_uwb_measurement(event)

    def test_missing_valid_rejected(self):
        """缺少 valid 时被拒绝。"""
        payload = {"anchor_id": "a1", "range": 5.0, "quality": 0.75}
        event = _valid_uwb_dict(uwb_payload=payload)
        with pytest.raises(KeyError, match="valid"):
            extract_uwb_measurement(event)

    def test_missing_quality_rejected(self):
        """缺少 quality 时被拒绝。"""
        payload = {"anchor_id": "a1", "range": 5.0, "valid": True}
        event = _valid_uwb_dict(uwb_payload=payload)
        with pytest.raises(KeyError, match="quality"):
            extract_uwb_measurement(event)

    def test_negative_range_rejected(self):
        """负 range 被拒绝。"""
        event = _valid_uwb_dict(uwb_payload={"anchor_id": "a1", "range": -1.0, "valid": True, "quality": 0.5})
        with pytest.raises(ValueError, match="must be >= 0.0"):
            extract_uwb_measurement(event)

    def test_inf_range_rejected(self):
        """inf range 被拒绝。"""
        event = _valid_uwb_event()
        event.uwb_payload["range"] = float("inf")
        with pytest.raises(ValueError, match="must be finite"):
            extract_uwb_measurement(event)

    def test_nan_range_rejected(self):
        """NaN range 被拒绝。"""
        event = _valid_uwb_event()
        event.uwb_payload["range"] = float("nan")
        with pytest.raises(ValueError, match="must be finite"):
            extract_uwb_measurement(event)

    def test_string_valid_rejected(self):
        """字符串 valid 被拒绝（validate_event 或双重防线均可拦截）。"""
        event = _valid_uwb_event(uwb_payload={"anchor_id": "a1", "range": 5.0, "valid": "yes", "quality": 0.75})
        with pytest.raises(TypeError, match="valid must be a bool|valid must be bool"):
            extract_uwb_measurement(event)

    def test_int_valid_rejected(self):
        """整数 valid 被拒绝。"""
        event = _valid_uwb_dict(uwb_payload={"anchor_id": "a1", "range": 5.0, "valid": 1, "quality": 0.75})
        with pytest.raises(TypeError, match="valid must be a bool|valid must be bool"):
            extract_uwb_measurement(event)

    def test_quality_out_of_range_rejected_by_validate_event(self):
        """quality 超过 [0,1] 时被 validate_event 严格模式拒绝（不会走到归一化）。"""
        event = _valid_uwb_dict(uwb_payload={"anchor_id": "a1", "range": 5.0, "valid": True, "quality": 1.5})
        with pytest.raises(ValueError, match="quality must be <= 1.0"):
            extract_uwb_measurement(event)

    def test_negative_quality_rejected_by_validate_event(self):
        """负 quality 被 validate_event 严格模式拒绝（不会走到归一化）。"""
        event = _valid_uwb_dict(uwb_payload={"anchor_id": "a1", "range": 5.0, "valid": True, "quality": -0.5})
        with pytest.raises(ValueError, match="quality must be >= 0.0"):
            extract_uwb_measurement(event)

    def test_numpy_range_accepted(self):
        """numpy 标量 range 被接受。"""
        event = _valid_uwb_dict(uwb_payload={"anchor_id": "a1", "range": np.float64(5.0), "valid": True, "quality": 0.75})
        result = extract_uwb_measurement(event)
        assert result["measured_range"] == pytest.approx(5.0)

    def test_anchor_id_preserved_as_int(self):
        """anchor_id 为整数时原样保留。"""
        event = _valid_uwb_dict(uwb_payload={"anchor_id": 42, "range": 5.0, "valid": True, "quality": 0.75})
        result = extract_uwb_measurement(event)
        assert result["anchor_id"] == 42


# ── predict_range_to_anchor ─────────────────────────────


class TestPredictRangeToAnchor:
    """predict_range_to_anchor 单元测试。"""

    def test_3_4_5_triangle(self):
        """3-4-5 直角三角形距离为 5.0。"""
        assert predict_range_to_anchor({"px": 0.0, "py": 0.0}, (3.0, 4.0)) == pytest.approx(5.0)

    def test_same_position_zero_distance(self):
        """同一位置距离为 0。"""
        assert predict_range_to_anchor({"px": 1.0, "py": 2.0}, (1.0, 2.0)) == pytest.approx(0.0)

    def test_negative_coordinates(self):
        """负坐标正确计算。"""
        assert predict_range_to_anchor({"px": -1.0, "py": -1.0}, (2.0, 3.0)) == pytest.approx(5.0)

    def test_tuple_anchor(self):
        """元组 anchor_pos 正确。"""
        assert predict_range_to_anchor({"px": 0.0, "py": 0.0}, (3.0, 4.0)) == pytest.approx(5.0)

    def test_list_anchor(self):
        """列表 anchor_pos 正确。"""
        assert predict_range_to_anchor({"px": 0.0, "py": 0.0}, [3.0, 4.0]) == pytest.approx(5.0)

    def test_numpy_array_anchor(self):
        """numpy 数组 anchor_pos 正确。"""
        assert predict_range_to_anchor({"px": 0.0, "py": 0.0}, np.array([3.0, 4.0])) == pytest.approx(5.0)

    def test_none_state_rejected(self):
        """None state 被拒绝。"""
        with pytest.raises(ValueError, match="must not be None"):
            predict_range_to_anchor(None, (3.0, 4.0))

    def test_none_anchor_rejected(self):
        """None anchor_pos 被拒绝。"""
        with pytest.raises(ValueError, match="must not be None"):
            predict_range_to_anchor({"px": 0.0, "py": 0.0}, None)

    def test_string_anchor_rejected(self):
        """字符串 anchor_pos 被拒绝。"""
        with pytest.raises(TypeError, match="must be a 2D coordinate record"):
            predict_range_to_anchor({"px": 0.0, "py": 0.0}, "ab")

    def test_scalar_anchor_rejected(self):
        """标量 anchor_pos 被拒绝。"""
        with pytest.raises(TypeError, match="must be a 2D coordinate record"):
            predict_range_to_anchor({"px": 0.0, "py": 0.0}, 5.0)

    def test_1d_anchor_rejected(self):
        """1 维 anchor_pos 被拒绝。"""
        with pytest.raises(ValueError, match="exactly 2 coordinates, got 1"):
            predict_range_to_anchor({"px": 0.0, "py": 0.0}, [3.0])

    def test_3d_anchor_rejected(self):
        """3 维 anchor_pos 被拒绝。"""
        with pytest.raises(ValueError, match="exactly 2 coordinates, got 3"):
            predict_range_to_anchor({"px": 0.0, "py": 0.0}, [1.0, 2.0, 3.0])

    def test_missing_px_rejected(self):
        """缺少 px 键被拒绝。"""
        with pytest.raises(KeyError, match="px"):
            predict_range_to_anchor({"py": 0.0}, (3.0, 4.0))

    def test_missing_py_rejected(self):
        """缺少 py 键被拒绝。"""
        with pytest.raises(KeyError, match="py"):
            predict_range_to_anchor({"px": 0.0}, (3.0, 4.0))

    def test_nan_px_rejected(self):
        """NaN px 会被拒绝，避免非法几何量继续流入下游。"""
        with pytest.raises(ValueError, match="state.px must be finite"):
            predict_range_to_anchor({"px": float("nan"), "py": 0.0}, (3.0, 4.0))

    def test_inf_anchor_rejected(self):
        """inf anchor 坐标会被拒绝，避免非法几何量继续流入下游。"""
        with pytest.raises(ValueError, match=r"anchor_pos\[0\] must be finite"):
            predict_range_to_anchor({"px": 0.0, "py": 0.0}, (float("inf"), 0.0))

    def test_large_coordinates(self):
        """大坐标正确计算。"""
        result = predict_range_to_anchor({"px": 1e6, "py": 0.0}, (0.0, 1e6))
        assert result == pytest.approx(math.hypot(1e6, 1e6))

    def test_int_coordinates_accepted(self):
        """整数坐标被接受。"""
        assert predict_range_to_anchor({"px": 0, "py": 0}, (3, 4)) == pytest.approx(5.0)
