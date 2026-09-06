from __future__ import annotations

"""偏差适配器（bias_adapter）测试模块。

文件职责：验证 apply_bias 能正确对 UWB 测距值
施加偏差校正，并将校正后距离裁剪到非负。

测试覆盖范围：
- 正常场景：施加偏差校正
- 边界场景：校正后距离裁剪到 0
- 异常场景：缺失 uwb_payload 拒绝
- 布尔输入拒绝

被测模块：liquidloc.fusion.bias_adapter"""


import pytest

from liquidloc.fusion.bias_adapter import apply_bias
from liquidloc.protocol.event_schema import Event


def _uwb_event(range_value: float):
    return Event(
        t=0.0,
        dt=0.0,
        modality="uwb",
        meta={"scene_id": "S(A2,N2,V2,K0,M0)", "seq_id": "mini_seq"},
        uwb_payload={"anchor_id": 0, "range": range_value, "valid": True, "quality": 0.9},
    )


def test_normal_case():
    corrected_range, adapter_report = apply_bias(_uwb_event(2.5), 0.5)

    assert corrected_range == pytest.approx(2.0)
    assert adapter_report == {
        "raw_range": pytest.approx(2.5),
        "bias_value": pytest.approx(0.5),
        "corrected_range": pytest.approx(2.0),
    }


def test_boundary_case():
    corrected_range, adapter_report = apply_bias(_uwb_event(0.4), 1.2)

    # bias 被截断到 raw_range * uwb_bias_max_ratio = 0.4 * 0.5 = 0.2
    assert corrected_range == pytest.approx(0.2)
    assert adapter_report["raw_range"] == pytest.approx(0.4)
    assert adapter_report["bias_value"] == pytest.approx(0.2)
    assert adapter_report["corrected_range"] == pytest.approx(0.2)


def test_invalid_case():
    bad_event = {
        "t": 0.0,
        "dt": 0.0,
        "modality": "uwb",
        "meta": {"scene_id": "S(A2,N2,V2,K0,M0)", "seq_id": "mini_seq"},
        "uwb_payload": None,
    }

    with pytest.raises(ValueError, match="uwb_payload"):
        apply_bias(bad_event, 0.5)


@pytest.mark.parametrize(
    "range_value,bias_value",
    [
        (True, 0.5),
        (1.0, True),
    ],
)
def test_boolean_input_rejected(range_value, bias_value):
    event = _uwb_event(1.0)
    event.uwb_payload["range"] = range_value

    # validate_event 在 _coerce_real_number 之前执行，
    # 布尔值可能被 validate_event 或 _coerce_real_number 拒绝。
    with pytest.raises((TypeError, ValueError)):  # 接受两种拒绝方式。
        apply_bias(event, bias_value)


def test_none_uwb_event_rejected():
    with pytest.raises(ValueError, match="uwb_event must not be None"):
        apply_bias(None, 0.5)


def test_none_bias_value_rejected():
    event = _uwb_event(2.5)
    with pytest.raises(ValueError, match="bias_value must not be None"):
        apply_bias(event, None)


def test_unsupported_event_type_rejected():
    with pytest.raises(TypeError, match="uwb_event must be an Event or dict"):
        apply_bias([1, 2], 0.5)


def test_empty_uwb_payload_rejected():
    # 非当前模态事件允许 uwb_payload 为空字典通过 validate_event，
    # apply_bias 自身守卫在此处阻断空负载。
    imu_event_with_empty_uwb = {
        "t": 0.0,
        "dt": 0.0,
        "modality": "imu",
        "meta": {"scene_id": "S(A2,N2,V2,K0,M0)", "seq_id": "mini_seq"},
        "imu_payload": {"ax": 0.2, "ay": 0.0, "gz": 0.01},
        "uwb_payload": {},
    }

    with pytest.raises(ValueError, match="uwb_payload"):
        apply_bias(imu_event_with_empty_uwb, 0.5)
