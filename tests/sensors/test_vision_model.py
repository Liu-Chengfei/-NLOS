from __future__ import annotations

"""视觉模型（vision_model）测试模块。

文件职责：验证 VIO 测量提取和标量校验函数的正确性。

测试覆盖范围：
- coerce_finite_scalar：类型校验、有限性、min_value 参数
- _coerce_nonnegative_int：整数校验、非负校验、类型拒绝
- extract_vio_measurement：正常提取、模态校验、payload 校验、
  必需字段缺失、字段值非法、quality 归一化、不修改输入

被测模块：liquidloc.sensors.vision_model"""


import math

import numpy as np
import pytest

from liquidloc.protocol.event_schema import Event
from liquidloc.common.validation import coerce_finite_scalar
from liquidloc.sensors.vision_model import (
    _coerce_nonnegative_int,
    extract_vio_measurement,
)


# ---------------------------------------------------------------------------
# 辅助工厂
# ---------------------------------------------------------------------------

def _valid_vio_payload(**overrides):
    """构造合法的 vio_payload 字典。

    铁律 3 (Stage A1 下游修复, 2026-07-23): VIO 4 字段新协议
    (dx, dy, dyaw, quality) — tracked_features / reproj_err 已删除.
    """
    payload = {
        "dx": 0.5,
        "dy": -0.1,
        "dyaw": 0.02,
        "quality": 0.9,
    }
    payload.update(overrides)
    return payload


def _valid_vio_dict(**overrides):
    """构造合法的 VIO 事件字典。"""
    event = {
        "t": 0.3,
        "dt": 0.1,
        "modality": "vio",
        "meta": {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "seq_001"},
        "imu_payload": None,
        "uwb_payload": None,
        "vio_payload": _valid_vio_payload(),
    }
    event.update(overrides)
    return event


def _valid_vio_event(**overrides):
    """构造合法的 Event 对象。"""
    event_kwargs = {
        "t": 0.3,
        "dt": 0.1,
        "modality": "vio",
        "meta": {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "seq_001"},
        "vio_payload": _valid_vio_payload(),
    }
    event_kwargs.update(overrides)
    return Event(**event_kwargs)


# ===========================================================================
# TestCoerceFiniteVioScalar
# ===========================================================================

class TestCoerceFiniteVioScalar:
    """coerce_finite_scalar 的单元测试。"""

    def test_int_input(self):
        """整数输入被转换为 float。"""
        assert coerce_finite_scalar(3, name="x") == 3.0

    def test_float_input(self):
        """float 输入原样返回。"""
        assert coerce_finite_scalar(2.5, name="x") == 2.5

    def test_numpy_scalar(self):
        """numpy 标量被转换为 float。"""
        assert coerce_finite_scalar(np.float64(1.5), name="x") == 1.5

    def test_negative_value(self):
        """负值通过（不像 uwb_model 那样要求非负）。"""
        assert coerce_finite_scalar(-3.0, name="x") == -3.0

    def test_zero_value(self):
        """零值通过。"""
        assert coerce_finite_scalar(0.0, name="x") == 0.0

    def test_large_value(self):
        """大数值通过。"""
        assert coerce_finite_scalar(1e10, name="x") == 1e10

    def test_bool_rejected(self):
        """bool 被拒绝。"""
        with pytest.raises(TypeError, match="must be numeric"):
            coerce_finite_scalar(True, name="x")

    def test_string_rejected(self):
        """字符串被拒绝。"""
        with pytest.raises(TypeError, match="must be numeric"):
            coerce_finite_scalar("1.0", name="x")

    def test_list_rejected(self):
        """列表被拒绝。"""
        with pytest.raises(TypeError, match="must be numeric"):
            coerce_finite_scalar([1.0], name="x")

    def test_nan_rejected(self):
        """NaN 被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            coerce_finite_scalar(float("nan"), name="x")

    def test_inf_rejected(self):
        """正无穷被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            coerce_finite_scalar(float("inf"), name="x")

    def test_neg_inf_rejected(self):
        """负无穷被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            coerce_finite_scalar(float("-inf"), name="x")

    def test_numpy_nan_rejected(self):
        """numpy NaN 被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            coerce_finite_scalar(np.float64("nan"), name="x")

    def test_numpy_inf_rejected(self):
        """numpy inf 被拒绝。"""
        with pytest.raises(ValueError, match="must be finite"):
            coerce_finite_scalar(np.float64("inf"), name="x")

    # -- min_value 参数测试 --

    def test_min_value_passes(self):
        """值等于 min_value 时通过。"""
        assert coerce_finite_scalar(0.0, name="x", min_value=0.0) == 0.0

    def test_min_value_above_passes(self):
        """值大于 min_value 时通过。"""
        assert coerce_finite_scalar(1.5, name="x", min_value=0.0) == 1.5

    def test_min_value_below_rejected(self):
        """值小于 min_value 时被拒绝。"""
        with pytest.raises(ValueError, match="must be >= 0.0"):
            coerce_finite_scalar(-0.1, name="x", min_value=0.0)

    def test_min_value_none_skips_check(self):
        """min_value=None 时不检查下界（默认行为）。"""
        assert coerce_finite_scalar(-1.0, name="x", min_value=None) == -1.0


# ===========================================================================
# TestCoerceNonnegativeInt
# ===========================================================================

class TestCoerceNonnegativeInt:
    """_coerce_nonnegative_int 的单元测试。"""

    def test_int_input(self):
        """整数输入原样返回。"""
        assert _coerce_nonnegative_int(5, name="x") == 5

    def test_zero_input(self):
        """零值通过。"""
        assert _coerce_nonnegative_int(0, name="x") == 0

    def test_numpy_int(self):
        """numpy 整数被转换为 int。"""
        assert _coerce_nonnegative_int(np.int64(10), name="x") == 10

    def test_negative_rejected(self):
        """负整数被拒绝。"""
        with pytest.raises(ValueError, match="must be >= 0"):
            _coerce_nonnegative_int(-1, name="x")

    def test_bool_rejected(self):
        """bool 被拒绝（虽然 bool 是 Integral 的子类）。"""
        with pytest.raises(TypeError, match="must be an integer"):
            _coerce_nonnegative_int(True, name="x")

    def test_float_rejected(self):
        """浮点数被拒绝（不是 Integral）。"""
        with pytest.raises(TypeError, match="must be an integer"):
            _coerce_nonnegative_int(3.0, name="x")

    def test_string_rejected(self):
        """字符串被拒绝。"""
        with pytest.raises(TypeError, match="must be an integer"):
            _coerce_nonnegative_int("5", name="x")


# ===========================================================================
# TestExtractVioMeasurement
# ===========================================================================

class TestExtractVioMeasurement:
    """extract_vio_measurement 的单元测试。"""

    # -- 正常提取 --

    def test_dict_input(self):
        """字典输入正常提取。"""
        event = _valid_vio_dict()
        result = extract_vio_measurement(event)
        assert result["dx"] == pytest.approx(0.5)
        assert result["dy"] == pytest.approx(-0.1)
        assert result["dyaw"] == pytest.approx(0.02)
        assert result["quality"] == pytest.approx(0.9)

    def test_event_input(self):
        """Event 对象输入正常提取。"""
        event = _valid_vio_event()
        result = extract_vio_measurement(event)
        assert result["dx"] == pytest.approx(0.5)
        assert result["dy"] == pytest.approx(-0.1)
        assert result["dyaw"] == pytest.approx(0.02)
        assert result["quality"] == pytest.approx(0.9)

    def test_return_keys(self):
        """返回字典包含 4 个固定键 (新协议 dx/dy/dyaw/quality)。"""
        event = _valid_vio_dict()
        result = extract_vio_measurement(event)
        assert set(result.keys()) == {"dx", "dy", "dyaw", "quality"}

    def test_return_types(self):
        """返回值的类型正确。"""
        event = _valid_vio_dict()
        result = extract_vio_measurement(event)
        assert isinstance(result["dx"], float)
        assert isinstance(result["dy"], float)
        assert isinstance(result["dyaw"], float)
        assert isinstance(result["quality"], float)

    def test_numpy_scalars_converted(self):
        """numpy 标量输入被转换为 Python 原生类型。"""
        event = _valid_vio_dict(vio_payload=_valid_vio_payload(
            dx=np.float64(1.0),
            dy=np.float64(-0.5),
            dyaw=np.float64(0.01),
        ))
        result = extract_vio_measurement(event)
        assert isinstance(result["dx"], float)
        assert isinstance(result["dy"], float)
        assert isinstance(result["dyaw"], float)
        assert isinstance(result["quality"], float)

    # -- 负值 / 零值 / 大值 --

    def test_negative_dx_dy(self):
        """负 dx/dy 允许（不像 UWB 的 range）。"""
        event = _valid_vio_dict(vio_payload=_valid_vio_payload(dx=-5.0, dy=-3.0))
        result = extract_vio_measurement(event)
        assert result["dx"] == pytest.approx(-5.0)
        assert result["dy"] == pytest.approx(-3.0)

    def test_negative_dyaw(self):
        """负 dyaw 允许。"""
        event = _valid_vio_dict(vio_payload=_valid_vio_payload(dyaw=-0.5))
        result = extract_vio_measurement(event)
        assert result["dyaw"] == pytest.approx(-0.5)

    def test_zero_values(self):
        """所有数值字段为零时通过。"""
        event = _valid_vio_dict(vio_payload=_valid_vio_payload(
            dx=0.0, dy=0.0, dyaw=0.0, quality=0.0,
        ))
        result = extract_vio_measurement(event)
        assert result["dx"] == 0.0
        assert result["dy"] == 0.0
        assert result["dyaw"] == 0.0
        assert result["quality"] == 0.0

    def test_large_values(self):
        """大数值通过（在物理范围约束内）。"""
        event = _valid_vio_dict(vio_payload=_valid_vio_payload(
            dx=9.5, dy=-9.5, dyaw=3.14,
        ))
        result = extract_vio_measurement(event)
        assert result["dx"] == pytest.approx(9.5)

    # -- 模态校验 --

    def test_wrong_modality_rejected(self):
        """模态不是 'vio' 时被拒绝。"""
        event = _valid_vio_dict(modality="imu")
        with pytest.raises(ValueError):
            extract_vio_measurement(event)

    def test_missing_modality_rejected(self):
        """缺少模态字段时被拒绝。"""
        event = _valid_vio_dict()
        del event["modality"]
        with pytest.raises(KeyError, match="modality"):
            extract_vio_measurement(event)

    # -- payload 校验 --

    def test_missing_payload_rejected(self):
        """缺少 vio_payload 时被拒绝。"""
        event = _valid_vio_dict(vio_payload=None)
        with pytest.raises(ValueError):
            extract_vio_measurement(event)

    def test_non_dict_payload_rejected(self):
        """vio_payload 不是字典时被拒绝。"""
        event = _valid_vio_dict()
        event["vio_payload"] = "not_a_dict"
        with pytest.raises((TypeError, ValueError)):
            extract_vio_measurement(event)

    # -- 必需字段缺失 --

    def test_missing_dx_rejected(self):
        """缺少 dx 字段时被拒绝。"""
        payload = _valid_vio_payload()
        del payload["dx"]
        event = _valid_vio_dict(vio_payload=payload)
        with pytest.raises(KeyError, match="dx"):
            extract_vio_measurement(event)

    def test_missing_dy_rejected(self):
        """缺少 dy 字段时被拒绝。"""
        payload = _valid_vio_payload()
        del payload["dy"]
        event = _valid_vio_dict(vio_payload=payload)
        with pytest.raises(KeyError, match="dy"):
            extract_vio_measurement(event)

    def test_missing_dyaw_rejected(self):
        """缺少 dyaw 字段时被拒绝。"""
        payload = _valid_vio_payload()
        del payload["dyaw"]
        event = _valid_vio_dict(vio_payload=payload)
        with pytest.raises(KeyError, match="dyaw"):
            extract_vio_measurement(event)

    def test_missing_quality_rejected(self):
        """缺少 quality 字段时被拒绝。"""
        payload = _valid_vio_payload()
        del payload["quality"]
        event = _valid_vio_dict(vio_payload=payload)
        with pytest.raises(KeyError, match="quality"):
            extract_vio_measurement(event)

    # 铁律 3 (Stage A1 下游修复, 2026-07-23): tracked_features / reproj_err 已从
    # VIO 协议删除 — 缺失/负/类型不匹配测试均不再适用, 已删除.

    # -- 字段值非法 --

    def test_nan_dx_rejected(self):
        """NaN dx 被拒绝。"""
        event = _valid_vio_dict(vio_payload=_valid_vio_payload(dx=float("nan")))
        with pytest.raises(ValueError, match="must be finite"):
            extract_vio_measurement(event)

    def test_inf_dy_rejected(self):
        """inf dy 被拒绝。"""
        event = _valid_vio_dict(vio_payload=_valid_vio_payload(dy=float("inf")))
        with pytest.raises(ValueError, match="must be finite"):
            extract_vio_measurement(event)

    def test_bool_dx_rejected(self):
        """bool dx 被拒绝。"""
        event = _valid_vio_dict(vio_payload=_valid_vio_payload(dx=True))
        with pytest.raises(TypeError, match="must be numeric"):
            extract_vio_measurement(event)

    def test_string_dyaw_rejected(self):
        """字符串 dyaw 被拒绝。"""
        event = _valid_vio_dict(vio_payload=_valid_vio_payload(dyaw="0.02"))
        with pytest.raises(TypeError, match="must be numeric"):
            extract_vio_measurement(event)

    # 铁律 3 (Stage A1 下游修复, 2026-07-23): tracked_features / reproj_err 校验
    # 已从 extract_vio_measurement 移除 — 负值/bool/float/NaN/inf 测试均不
    # 再适用, 已删除.

    # -- quality 归一化 --

    def test_quality_out_of_range_rejected(self):
        """quality 超过 [0,1] 时被严格拒绝（不裁剪）。"""
        event = _valid_vio_dict(vio_payload=_valid_vio_payload(quality=1.5))
        with pytest.raises(ValueError, match="must be <= 1.0"):
            extract_vio_measurement(event)

    def test_negative_quality_rejected(self):
        """负 quality 被严格拒绝（不裁剪）。"""
        event = _valid_vio_dict(vio_payload=_valid_vio_payload(quality=-0.5))
        with pytest.raises(ValueError, match="must be >= 0.0"):
            extract_vio_measurement(event)

    def test_quality_none_rejected(self):
        """quality 为 None 时被校验拒绝。"""
        event = _valid_vio_dict(vio_payload=_valid_vio_payload(quality=None))
        with pytest.raises(ValueError, match="must not be None"):
            extract_vio_measurement(event)

    # -- 不修改输入 --

    def test_dict_input_not_modified(self):
        """提取操作不修改原始字典。"""
        payload = _valid_vio_payload()
        original = dict(payload)
        event = _valid_vio_dict(vio_payload=payload)
        extract_vio_measurement(event)
        assert payload == original

    # -- 事件对象无属性时 --

    def test_object_without_modality_rejected(self):
        """没有 modality 属性的普通对象被拒绝。"""
        class NoModality:
            pass
        with pytest.raises((ValueError, TypeError)):
            extract_vio_measurement(NoModality())

    def test_object_without_payload_rejected(self):
        """没有 vio_payload 属性的对象被拒绝。"""
        class NoPayload:
            modality = "vio"
        with pytest.raises((ValueError, TypeError)):
            extract_vio_measurement(NoPayload())
