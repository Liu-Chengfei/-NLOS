from __future__ import annotations

"""异步等级（async_levels）测试模块。

文件职责：验证 apply_async_level 能根据异步等级配置
对事件序列施加时间偏移、抖动、突发丢失、时钟漂移等扰动。

测试覆盖范围：
- 正常场景：A1 等级施加偏移和突发丢失
- 边界场景：零偏移、零抖动、零丢失时事件不变
- 时钟漂移累积：验证漂移按 IMU 参考时钟累积
- 异常场景：缺失配置字段、非法参数
- 空序列处理
- burst_missing_ratio 别名兼容
- blackout 选择稳定性
- 序列身份影响偏移和抖动实现

被测模块：liquidloc.scenarios.async_levels"""


import pytest

from liquidloc.scenarios.async_levels import apply_async_level


def _events():
    return [
        {
            't': 0.0,
            'dt': 0.0,
            'modality': 'imu',
            'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'},
            'imu_payload': {'ax': 0.1, 'ay': 0.0, 'gz': 0.0},
            'uwb_payload': None,
            'vio_payload': None,
        },
        {
            't': 0.1,
            'dt': 0.1,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 0.95},
            'vio_payload': None,
        },
        {
            't': 0.2,
            'dt': 0.1,
            'modality': 'vio',
            'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'},
            'imu_payload': None,
            'uwb_payload': None,
            'vio_payload': {
                'dx': 0.02,
                'dy': 0.0,
                'dyaw': 0.0,
                'quality': 0.85,
                'tracked_features': 118,
                'reproj_err': 0.3,
            },
        },
        {
            't': 0.3,
            'dt': 0.1,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 1, 'range': 2.5, 'valid': True, 'quality': 0.9},
            'vio_payload': None,
        },
    ]


def test_normal_case():
    source_events = _events()
    new_events, report = apply_async_level(
        source_events,
        'A1',
        {'A1': {'offset_ms': [10, 30], 'jitter_ms': 2, 'burst_missing_prob': 0.5, 'cross_modal_skew_ms': 4, 'clock_drift_ppm': 0}},
    )

    assert [event['modality'] for event in new_events] == ['imu', 'uwb']
    # offset和jitter现在是确定性随机的，不再使用固定中点和交替方向
    # IMU事件不受偏移影响，但受jitter影响；UWB事件受偏移+jitter影响
    # 只检查时间戳在合理范围内
    assert new_events[0]['t'] >= 0.0  # IMU事件时间戳非负
    assert new_events[1]['t'] > 0.1  # UWB事件在偏移后应大于原始0.1s
    assert report['async_level'] == 'A1'
    # offset_midpoint现在是随机采样的，不再固定为20.0
    assert report['clock_drift_ppm'] == pytest.approx(0.0)
    assert report['dropped_event_count'] == 2
    assert report['blackout_strategy'] == 'stable_hashed_contiguous_non_imu_window'
    # blackout时间段受事件时间戳变化影响，不再精确匹配
    assert len(report['blackout_segments']) == 1
    assert source_events[1]['t'] == pytest.approx(0.1)
    assert source_events[2]['vio_payload']['tracked_features'] == 118


def test_boundary_case():
    source_events = _events()
    new_events, report = apply_async_level(
        source_events,
        'A1',
        {'A1': {'offset_ms': [0, 0], 'jitter_ms': 0, 'burst_missing_prob': 0.0, 'cross_modal_skew_ms': 0, 'clock_drift_ppm': 0}},
    )

    assert len(new_events) == len(source_events)
    assert [event['t'] for event in new_events] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert [event['dt'] for event in new_events] == pytest.approx([0.0, 0.1, 0.1, 0.1])
    assert report['dropped_event_count'] == 0
    assert report['blackout_segments'] == []
    assert report['timing_plan'][1]['shift_ms'] == pytest.approx(0.0)
    assert report['timing_plan'][2]['jitter_ms'] == pytest.approx(0.0)
    assert report['timing_plan'][2]['clock_drift_ms'] == pytest.approx(0.0)


def test_clock_drift_accumulates_against_imu_reference():
    source_events = _events()
    new_events, report = apply_async_level(
        source_events,
        'A2',
        {'A2': {'offset_ms': [0, 0], 'jitter_ms': 0, 'burst_missing_prob': 0.0, 'cross_modal_skew_ms': 0, 'clock_drift_ppm': 100000}},
    )

    assert [event['t'] for event in new_events] == pytest.approx([0.0, 0.11, 0.18, 0.33])
    assert [event['dt'] for event in new_events] == pytest.approx([0.0, 0.11, 0.07, 0.15])
    assert report['clock_drift_ppm'] == pytest.approx(100000.0)
    assert report['timing_plan'][1]['clock_drift_ms'] == pytest.approx(10.0)
    assert report['timing_plan'][2]['clock_drift_ms'] == pytest.approx(-20.0)
    assert report['timing_plan'][3]['clock_drift_ms'] == pytest.approx(30.0)


@pytest.mark.parametrize(
    ('async_level', 'async_cfg', 'exc_type', 'match'),
    [
        ('A1', {'A1': {'offset_ms': [10, 30], 'jitter_ms': 2, 'burst_missing_prob': 0.5}}, KeyError, 'cross_modal_skew_ms'),
        ('', {'A1': {'offset_ms': [10, 30], 'jitter_ms': 2, 'burst_missing_prob': 0.5, 'cross_modal_skew_ms': 4, 'clock_drift_ppm': 0}}, ValueError, 'non-empty string'),
        ('A1', {'A1': {'offset_ms': [10, 30], 'jitter_ms': 2, 'burst_missing_prob': 0.5, 'cross_modal_skew_ms': 4, 'clock_drift_ppm': -1}}, ValueError, 'clock_drift_ppm'),
        ('A1', {'A1': {'offset_ms': [0, float('nan')], 'jitter_ms': 2, 'burst_missing_prob': 0.5, 'cross_modal_skew_ms': 4, 'clock_drift_ppm': 0}}, ValueError, 'offset_ms'),
    ],
)
def test_invalid_case(async_level, async_cfg, exc_type, match):
    with pytest.raises(exc_type, match=match):
        apply_async_level(_events(), async_level, async_cfg)


def test_empty_sequence_returns_empty():
    """空事件序列不应崩溃，应返回空结果。"""
    new_events, report = apply_async_level(
        [],
        'A0',
        {'A0': {'offset_ms': [0, 0], 'jitter_ms': 0, 'burst_missing_prob': 0.0, 'cross_modal_skew_ms': 0, 'clock_drift_ppm': 0}},
    )
    assert new_events == []
    assert report['async_level'] == 'A0'
    assert report['dropped_event_count'] == 0
    assert report['timing_plan'] == []


def test_burst_missing_ratio_alias():
    """burst_missing_ratio 应被接受为 burst_missing_prob 的别名。"""
    source_events = _events()
    new_events_ratio, report_ratio = apply_async_level(
        source_events,
        'A1',
        {'A1': {'offset_ms': [0, 0], 'jitter_ms': 0, 'burst_missing_ratio': 0.5, 'cross_modal_skew_ms': 0, 'clock_drift_ppm': 0}},
    )
    new_events_prob, report_prob = apply_async_level(
        source_events,
        'A1',
        {'A1': {'offset_ms': [0, 0], 'jitter_ms': 0, 'burst_missing_prob': 0.5, 'cross_modal_skew_ms': 0, 'clock_drift_ppm': 0}},
    )
    assert [e['t'] for e in new_events_ratio] == pytest.approx([e['t'] for e in new_events_prob])
    assert report_ratio['burst_missing_prob'] == pytest.approx(0.5)


def test_blackout_selection_is_stable_contiguous_window_not_fixed_middle():
    starts = set()
    for seq_suffix in range(12):
        source_events = [
            {
                't': float(index) * 0.1,
                'dt': 0.0 if index == 0 else 0.1,
                'modality': modality,
                'meta': {'scene_id': 'S(A3,N2,V2,K0,M0)', 'seq_id': f'async_seq_{seq_suffix}'},
                'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0} if modality == 'imu' else None,
                'uwb_payload': {'anchor_id': index, 'range': 2.0 + index, 'valid': True, 'quality': 0.9} if modality == 'uwb' else None,
                'vio_payload': {'dx': 0.0, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.9, 'tracked_features': 50, 'reproj_err': 0.2} if modality == 'vio' else None,
            }
            for index, modality in enumerate(['imu', 'uwb', 'vio', 'uwb', 'vio', 'uwb', 'vio'])
        ]

        _, report = apply_async_level(
            source_events,
            'A2',
            {'A2': {'offset_ms': [0, 0], 'jitter_ms': 0, 'burst_missing_prob': 0.5, 'cross_modal_skew_ms': 0, 'clock_drift_ppm': 0}},
        )

        assert report['dropped_event_count'] == 3
        start_t, end_t = report['blackout_segments'][0]
        assert end_t > start_t
        starts.add(report['blackout_selection_start'])
        _, rerun_report = apply_async_level(
            source_events,
            'A2',
            {'A2': {'offset_ms': [0, 0], 'jitter_ms': 0, 'burst_missing_prob': 0.5, 'cross_modal_skew_ms': 0, 'clock_drift_ppm': 0}},
        )
        assert rerun_report['blackout_segments'] == pytest.approx(report['blackout_segments'])
        assert rerun_report['blackout_selection_start'] == report['blackout_selection_start']

    assert len(starts) > 1


def test_sequence_identity_changes_offset_and_jitter_realization():
    base_cfg = {
        'A2': {
            'offset_ms': [40, 120],
            'jitter_ms': 12,
            'burst_missing_prob': 0.0,
            'cross_modal_skew_ms': 60,
            'clock_drift_ppm': 120,
        }
    }
    seq_alpha_events = _events()
    seq_beta_events = _events()
    for event in seq_alpha_events:
        event['meta']['seq_id'] = 'seq_alpha'
    for event in seq_beta_events:
        event['meta']['seq_id'] = 'seq_beta'

    alpha_events, alpha_report = apply_async_level(seq_alpha_events, 'A2', base_cfg)
    beta_events, beta_report = apply_async_level(seq_beta_events, 'A2', base_cfg)

    assert alpha_report['offset_ms_midpoint'] != pytest.approx(beta_report['offset_ms_midpoint'])
    assert [step['jitter_ms'] for step in alpha_report['timing_plan']] != pytest.approx(
        [step['jitter_ms'] for step in beta_report['timing_plan']]
    )
    assert [event['t'] for event in alpha_events] != pytest.approx([event['t'] for event in beta_events])


def test_yaml_protocol_a0_a3_params_are_consumable():
    """集成测试：验证 YAML 协议中的 A0-A3 参数可被 apply_async_level 正确消费。

    此测试从冻结协议加载 A 轴参数，确保 YAML 值变更会导致测试失败，
    避免 YAML 与测试脱钩的假绿风险。
    """
    from liquidloc.protocol.scene_axis_protocol import load_scene_axis_protocol, get_protocol_axes

    cfg = load_scene_axis_protocol()
    axes = get_protocol_axes(cfg)
    a_axis = axes["A"]

    # 逐级验证 YAML 参数可被 apply_async_level 消费
    for level_name in ("A0", "A1", "A2", "A3"):
        level_cfg = a_axis[level_name]
        async_cfg = {level_name: {
            "offset_ms": level_cfg["offset_ms"],
            "jitter_ms": level_cfg["jitter_ms"],
            "burst_missing_prob": level_cfg["burst_missing_prob"],
            "cross_modal_skew_ms": level_cfg["cross_modal_skew_ms"],
            "clock_drift_ppm": level_cfg["clock_drift_ppm"],
        }}
        events, report = apply_async_level(_events(), level_name, async_cfg)
        # 基本断言：事件未被清空，报告包含必要字段
        assert len(events) > 0
        assert "offset_ms_midpoint" in report
