from __future__ import annotations

"""模态丢失（missing_modalities）测试模块。

文件职责：验证 apply_modality_drop 能在指定时间段内
丢弃指定模态的事件，并正确更新 dt 和恢复点。

测试覆盖范围：
- 正常场景：在指定时间段丢弃 VIO 事件
- 边界场景：丢弃段不匹配任何事件时无变化
- 重叠丢弃段合并
- 异常场景：不支持的模态名称

被测模块：liquidloc.scenarios.missing_modalities"""


import pytest

from liquidloc.protocol.event_schema import validate_event_sequence
from liquidloc.scenarios.missing_modalities import apply_modality_drop


def _events():
    return [
        {'t': 0.0, 'dt': 0.0, 'modality': 'imu', 'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.1, 'ay': 0.0, 'gz': 0.0}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.1, 'dt': 0.1, 'modality': 'vio', 'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.01, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.9, 'tracked_features': 120, 'reproj_err': 0.2}},
        {'t': 0.2, 'dt': 0.1, 'modality': 'uwb', 'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 0.95}, 'vio_payload': None},
        {'t': 0.3, 'dt': 0.1, 'modality': 'vio', 'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.02, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.85, 'tracked_features': 118, 'reproj_err': 0.3}},
        {'t': 0.4, 'dt': 0.1, 'modality': 'vio', 'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.03, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.82, 'tracked_features': 116, 'reproj_err': 0.4}},
        {'t': 0.5, 'dt': 0.1, 'modality': 'imu', 'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.0, 'ay': 0.1, 'gz': 0.01}, 'uwb_payload': None, 'vio_payload': None},
    ]


def test_normal_case():
    source_events = _events()
    new_events, drop_report = apply_modality_drop(source_events, 'vio', [(0.25, 0.35)])

    assert [event['t'] for event in new_events] == pytest.approx([0.0, 0.1, 0.2, 0.4, 0.5])
    assert [event['dt'] for event in new_events] == pytest.approx([0.0, 0.1, 0.1, 0.2, 0.1])
    assert [event['t'] for event in drop_report['dropped_events']] == pytest.approx([0.3])
    assert [event['t'] for event in drop_report['recover_points']] == pytest.approx([0.4])
    assert source_events[4]['dt'] == pytest.approx(0.1)
    validate_event_sequence(new_events)


def test_boundary_case():
    source_events = _events()
    new_events, drop_report = apply_modality_drop(source_events, 'uwb', [(1.0, 1.1)])

    assert new_events is not source_events
    assert new_events[2] is not source_events[2]
    assert [event['dt'] for event in new_events] == pytest.approx([0.0, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert drop_report['dropped_events'] == []
    assert drop_report['recover_points'] == []
    validate_event_sequence(new_events)


def test_overlapping_drop_segments_are_merged_without_duplicate_recovery_points():
    source_events = _events()
    new_events, drop_report = apply_modality_drop(source_events, 'vio', [(0.25, 0.35), (0.30, 0.45)])

    assert [event['t'] for event in new_events] == pytest.approx([0.0, 0.1, 0.2, 0.5])
    assert [event['t'] for event in drop_report['dropped_events']] == pytest.approx([0.3, 0.4])
    assert drop_report['drop_segments'] == [(0.25, 0.45)]
    assert drop_report['recover_points'] == []
    validate_event_sequence(new_events)


def test_invalid_case():
    with pytest.raises(ValueError, match='Unsupported modality'):
        apply_modality_drop(_events(), 'gps', [(0.0, 0.1)])
