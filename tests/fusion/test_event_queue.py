from __future__ import annotations

"""事件队列（event_queue）测试模块。

文件职责：验证 EventQueue 能正确管理事件序列的
游标、弹出、窥视和重置操作。

测试覆盖范围：
- 正常场景：弹出、窥视、重置
- 边界场景：空序列拒绝
- 异常场景：dt 不匹配拒绝
- 可迭代输入单次消费
- 返回隔离的事件快照
- Event 输入保留字典快照合同

被测模块：liquidloc.fusion.event_queue"""


import pytest

from liquidloc.protocol.event_schema import Event
from liquidloc.fusion.event_queue import EventQueue


def _imu_event(*, t=0.0, dt=0.0, ax=0.2, ay=0.0, gz=0.01):
    return {
        "t": t,
        "dt": dt,
        "modality": "imu",
        "meta": {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "mini_seq"},
        "imu_payload": {"ax": ax, "ay": ay, "gz": gz},
        "uwb_payload": None,
        "vio_payload": None,
    }


def _uwb_event(*, t=0.1, dt=0.1, rng=0.9, quality=0.95):
    return {
        "t": t,
        "dt": dt,
        "modality": "uwb",
        "meta": {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "mini_seq"},
        "imu_payload": None,
        "uwb_payload": {"anchor_id": 0, "range": rng, "valid": True, "quality": quality},
        "vio_payload": None,
    }


def test_normal_case():
    events = [_imu_event(), _uwb_event()]

    queue = EventQueue(events)

    assert queue.queue_length == 2
    assert queue.cursor == 0
    assert queue.current_event == events[0]
    assert queue.peek() == events[0]
    assert queue.cursor == 0

    assert queue.pop() == events[0]
    assert queue.cursor == 1
    assert queue.current_event == events[1]

    assert queue.pop() == events[1]
    assert queue.cursor == 2
    assert queue.has_next() is False
    assert queue.current_event is None
    assert queue.peek() is None
    assert queue.pop() is None

    queue.reset()
    assert queue.cursor == 0
    assert queue.current_event == events[0]


def test_boundary_case():
    with pytest.raises(ValueError, match="event sequence must not be empty"):
        EventQueue([])


def test_invalid_case():
    events = [_imu_event(), _uwb_event(dt=0.2)]

    with pytest.raises(ValueError, match="event.dt mismatch"):
        EventQueue(events)


def test_iterable_input_is_consumed_once():
    queue = EventQueue(event for event in [_imu_event(), _uwb_event()])

    assert queue.queue_length == 2
    assert queue.peek() == _imu_event()
    assert queue.pop() == _imu_event()
    assert queue.pop() == _uwb_event()
    assert queue.pop() is None


def test_queue_returns_isolated_event_snapshots():
    events = [_imu_event(), _uwb_event()]

    queue = EventQueue(events)
    peeked = queue.peek()
    peeked["meta"]["scene_id"] = "mutated"
    peeked["imu_payload"]["ax"] = 9.9

    assert events[0]["meta"]["scene_id"] == "S(A0,N0,V0,K0,M0)"
    assert events[0]["imu_payload"]["ax"] == 0.2
    assert queue.current_event["meta"]["scene_id"] == "S(A0,N0,V0,K0,M0)"

    popped = queue.pop()
    popped["meta"]["seq_id"] = "changed"
    popped["imu_payload"]["ax"] = 8.8

    assert queue.current_event["meta"]["seq_id"] == "mini_seq"
    assert queue.peek()["meta"]["seq_id"] == "mini_seq"


def test_event_input_preserves_dict_snapshot_contract():
    queue = EventQueue(
        [
            Event(
                t=0.0,
                dt=0.0,
                modality="imu",
                meta={"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "mini_seq"},
                imu_payload={"ax": 0.2, "ay": 0.0, "gz": 0.01},
            )
        ]
    )

    current = queue.current_event

    assert isinstance(current, dict)
    assert current["meta"]["seq_id"] == "mini_seq"
    assert queue.pop()["imu_payload"]["ax"] == 0.2


def test_none_input_rejected():
    with pytest.raises(ValueError, match="events must not be None"):
        EventQueue(None)


@pytest.mark.parametrize("bad_input", ["abc", b"abc"])
def test_string_or_bytes_input_rejected(bad_input):
    with pytest.raises(TypeError, match="not a string or bytes"):
        EventQueue(bad_input)


def test_mapping_input_rejected():
    with pytest.raises(TypeError, match="not a single mapping"):
        EventQueue({"t": 0.0, "dt": 0.0})


def test_events_property_returns_isolated_full_snapshot():
    events = [_imu_event(), _uwb_event()]
    queue = EventQueue(events)

    # 推进游标，events 属性应不受游标位置影响
    queue.pop()

    snapshot = queue.events
    assert isinstance(snapshot, tuple)
    assert len(snapshot) == 2
    # 游标已前进到 1，但 events 仍返回全部事件
    assert snapshot[0]["modality"] == "imu"
    assert snapshot[1]["modality"] == "uwb"

    # 修改返回快照不影响内部缓存
    snapshot[0]["meta"]["scene_id"] = "mutated"
    snapshot[0]["imu_payload"]["ax"] = 9.9
    assert queue.events[0]["meta"]["scene_id"] == "S(A0,N0,V0,K0,M0)"
    assert queue.events[0]["imu_payload"]["ax"] == 0.2
    # current_event 也未受污染
    assert queue.current_event["meta"]["scene_id"] == "S(A0,N0,V0,K0,M0)"
