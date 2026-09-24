
"""事件构建器（event_builder）测试模块。

文件职责：验证事件构建器能将原始数据行转换为
标准化事件字典，并正确合并、排序和计算 dt。

测试覆盖范围：
- 正常场景：IMU/UWB/VIO 事件合并和排序
- 边界场景：同时间戳事件的 dt=0
- source_t 默认值和显式值保留
- UWB anchor_id 类型保留
- 异常场景：非法 VIO quality 值
- 同时间戳事件保持输入分组和组内顺序

被测模块：liquidloc.dataio.adapters.event_builder"""

from liquidloc.dataio.adapters.event_builder import (
    build_imu_events,
    build_uwb_events,
    build_vio_events,
    merge_and_finalize_events,
)


def test_normal_case():
    imu_rows = [{'timestamp': 0.0, 'ax': 0.1, 'ay': 0.0, 'gz': 0.0}]
    uwb_rows = [{'timestamp': 0.1, 'anchor_id': 1, 'range': 2.0, 'valid': True, 'quality': 0.9}]
    # Stage A1 下游修复 (2026-07-23): VIO 紧耦合仅输出 dx/dy/dyaw/quality,
    # tracked_features / reproj_err 已从 sensors.yaml vio_fields 删除 (4 字段).
    vio_rows = [{'timestamp': 0.2, 'dx': 0.05, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.8}]

    events = merge_and_finalize_events([
        build_imu_events(imu_rows, 'S(A0,N0,V0,K0,M0)', 'seq0'),
        build_uwb_events(uwb_rows, 'S(A0,N0,V0,K0,M0)', 'seq0'),
        build_vio_events(vio_rows, 'S(A0,N0,V0,K0,M0)', 'seq0'),
    ])

    assert [event['modality'] for event in events] == ['imu', 'uwb', 'vio']
    assert [event['dt'] for event in events] == [0.0, 0.1, 0.1]


def test_boundary_case():
    events = merge_and_finalize_events([
        build_uwb_events([{'timestamp': 0.1, 'anchor_id': 1, 'range': 2.0, 'valid': True, 'quality': 0.9}], 'S(A0,N0,V0,K0,M0)', 'seq0'),
        build_imu_events([{'timestamp': 0.1, 'ax': 0.1, 'ay': 0.0, 'gz': 0.0}], 'S(A0,N0,V0,K0,M0)', 'seq0'),
    ])
    assert len(events) == 2
    assert events[0]['modality'] == 'uwb'
    assert events[1]['dt'] == 0.0


def test_source_t_defaults_to_timestamp_and_preserves_explicit_source_t():
    events = merge_and_finalize_events([
        build_imu_events([{'timestamp': 0.0, 'source_t': 0.25, 'ax': 0.1, 'ay': 0.0, 'gz': 0.0}], 'S(A0,N0,V0,K0,M0)', 'seq0'),
        build_uwb_events([{'timestamp': 0.1, 'anchor_id': 1, 'range': 2.0, 'valid': True, 'quality': 0.9}], 'S(A0,N0,V0,K0,M0)', 'seq0'),
        build_vio_events([{'timestamp': 0.2, 'dx': 0.05, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.8}], 'S(A0,N0,V0,K0,M0)', 'seq0'),
    ])

    assert events[0]['meta']['source_t'] == 0.25
    assert events[1]['meta']['source_t'] == 0.1
    assert events[2]['meta']['source_t'] == 0.2


def test_uwb_anchor_id_preserves_original_type():
    events = build_uwb_events(
        [{'timestamp': 0.1, 'anchor_id': 'A01', 'range': 2.0, 'valid': True, 'quality': 0.9}],
        'S(A0,N0,V0,K0,M0)',
        'seq0',
    )

    assert events[0]['uwb_payload']['anchor_id'] == 'A01'


def test_invalid_case():
    bad_vio_rows = [{'timestamp': 0.2, 'dx': 0.05, 'dy': 0.0, 'dyaw': 0.0, 'quality': 1.2}]
    try:
        merge_and_finalize_events([
            build_vio_events(bad_vio_rows, 'S(A0,N0,V0,K0,M0)', 'seq0'),
        ])
    except ValueError as exc:
        assert 'quality' in str(exc)
    else:
        raise AssertionError('Expected invalid VIO quality to raise ValueError')


def test_same_timestamp_events_preserve_input_group_and_group_internal_order():
    imu_events = build_imu_events(
        [
            {'timestamp': 0.1, 'ax': 0.1, 'ay': 0.0, 'gz': 0.0},
            {'timestamp': 0.1, 'ax': 0.2, 'ay': 0.0, 'gz': 0.0},
        ],
        'S(A0,N0,V0,K0,M0)',
        'seq0',
    )
    uwb_events = build_uwb_events(
        [{'timestamp': 0.1, 'anchor_id': 1, 'range': 2.0, 'valid': True, 'quality': 0.9}],
        'S(A0,N0,V0,K0,M0)',
        'seq0',
    )
    vio_events = build_vio_events(
        [{'timestamp': 0.1, 'dx': 0.05, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.8}],
        'S(A0,N0,V0,K0,M0)',
        'seq0',
    )

    events = merge_and_finalize_events([vio_events, uwb_events, imu_events])

    assert [event['modality'] for event in events] == ['vio', 'uwb', 'imu', 'imu']
    assert [event['imu_payload']['ax'] if event['imu_payload'] else None for event in events] == [None, None, 0.1, 0.2]
    assert all('_merge_group_order' not in event and '_merge_event_order' not in event for event in events)
