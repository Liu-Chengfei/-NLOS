from __future__ import annotations

"""视觉退化等级（visual_levels）测试模块。

文件职责：验证 apply_visual_level 能根据视觉退化等级配置
对 VIO 事件施加质量退化、特征丢失等扰动。

测试覆盖范围：
- 正常场景：V1 等级对 VIO 事件施加退化
- 边界场景：V0 等级不施加任何退化
- 异常场景：非法等级名称或缺失配置字段
- 锚点重映射的确定性

被测模块：liquidloc.scenarios.visual_levels"""


import random  # 2026-09-03：blackout_identity 测试用 random/numpy 种子。
import numpy as np  # noqa: F401  # 2026-09-03：同一测试需要。
import pytest

from liquidloc.scenarios.visual_levels import apply_visual_level


def _events():
    """6 个 VIO 事件（足够让不同 seq_id 的 stable_hash 起点分散到不同窗口）。"""
    base_time = 0.0
    events = [{
        't': 0.0,
        'dt': 0.0,
        'modality': 'imu',
        'meta': {'scene_id': 'S(A1,N0,V1,K0,M0)', 'seq_id': 'mini_seq'},
        'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0},
        'uwb_payload': None,
        'vio_payload': None,
    }]
    vio_payloads = [
        {'dx': 0.5, 'dy': -0.2, 'dyaw': 0.1, 'quality': 0.9, 'tracked_features': 150, 'reproj_err': 1.4},
        {'dx': 0.1, 'dy': 0.0, 'dyaw': 0.05, 'quality': 0.8, 'tracked_features': 50, 'reproj_err': 0.2},
        {'dx': -0.2, 'dy': 0.3, 'dyaw': -0.04, 'quality': 0.7, 'tracked_features': 80, 'reproj_err': 5.0},
    ]
    for i, vp in enumerate(vio_payloads):
        events.append({
            't': 0.1 * (i + 1),
            'dt': 0.1,
            'modality': 'vio',
            'meta': {'scene_id': 'S(A1,N0,V1,K0,M0)', 'seq_id': 'mini_seq'},
            'imu_payload': None,
            'uwb_payload': None,
            'vio_payload': dict(vp),
        })
    return events


def _visual_cfg():
    return {
        'V1': {
            'tracked_features_range': [60, 120],
            'reproj_err_max': 1.0,
            'blackout_prob': 0.0,
            'drift_bias_m': 0.05,
            # 2026-09-03 补全：区间化字段（测试场景 V1 无黑屏/无关键帧，设为 baseline 空值）。
            'blackout_duration_range_s': [0.0, 0.0],
            'blackout_fraction': [0.0, 0.0],
            'keyframe_interruptions_per_seq': 0,
            'keyframe_burst_duration_range_s': [0.0, 0.0],
            'keyframe_burst_fraction': [0.0, 0.0],
            'keyframe_recovery_jump_m': 0.0,
        },
        'V2': {
            'tracked_features_range': [20, 80],
            'reproj_err_max': 3.0,
            'blackout_prob': 0.3,
            'drift_bias_m': 0.05,
            'blackout_duration_range_s': [0.5, 2.0],
            'blackout_fraction': [0.2, 0.6],
            'keyframe_interruptions_per_seq': 1,
            'keyframe_burst_duration_range_s': [0.5, 2.0],
            'keyframe_burst_fraction': [0.2, 0.6],
            'keyframe_recovery_jump_m': 0.3,
        },
    }


def test_normal_case():
    degraded_events, report = apply_visual_level(_events(), 'V1', _visual_cfg())
    vio_payloads = [event['vio_payload'] for event in degraded_events if event['modality'] == 'vio']

    assert [payload['tracked_features'] for payload in vio_payloads] == [120, 50, 80]
    assert [payload['reproj_err'] for payload in vio_payloads] == pytest.approx([1.0, 0.218231187168, 1.0])
    assert [payload['quality'] for payload in vio_payloads] == pytest.approx([0.801, 0.791978, 0.691127])
    # VIO drift 改为随机游走模型后，第一个 VIO 事件无前驱不累积 bias，
    # dx_after = dx_before；后续事件按时间累积 Wiener 过程，bias 非零。
    # 验证：第一个事件 dx 不变，第二个事件 bias_x 非零，第三个事件 dyaw 受 bias 影响。
    assert vio_payloads[0]['dx'] == pytest.approx(0.5)  # 第一个 VIO 事件无 drift
    assert vio_payloads[2]['dy'] == pytest.approx(0.342903356824)  # 受 bias_y 累积影响
    assert report['tracked_features_range_after'] == [50, 120]
    assert report['reproj_err_max_after'] == pytest.approx(1.0)
    assert report['blackout_count_expected'] == 0
    assert report['reproj_err_plan'][0]['reproj_err_after'] == pytest.approx(1.0)
    assert report['reproj_err_plan'][1]['reproj_err_after'] == pytest.approx(0.218231187168)
    assert report['quality_plan'][0]['quality_after'] == pytest.approx(0.801)
    assert report['quality_plan'][2]['quality_after'] == pytest.approx(0.691127)
    # 随机游走模型：第一个事件 drift_bias_x=0（无前驱），第二个事件非零。
    assert report['drift_transform_plan'][0]['dx_after'] == pytest.approx(0.5)
    assert report['drift_transform_plan'][0]['drift_bias_x'] == pytest.approx(0.0)
    assert report['drift_transform_plan'][1]['drift_bias_x'] != pytest.approx(0.0)
    assert report['drift_transform_plan'][2]['dyaw_after'] == pytest.approx(-0.040270196496)
    # 一致性：dx_after - dx_before == drift_dx（随机游走 bias × sign(dx_before)）。
    for step in report['drift_transform_plan']:
        assert step['dx_after'] == pytest.approx(step['dx_before'] + step['drift_dx'])
        assert step['dy_after'] == pytest.approx(step['dy_before'] + step['drift_dy'])
    assert report['consistency_checks'] == {
        'tracked_features_range': True,
        'reproj_err_max': True,
        'blackout': True,
        'drift_bias_m': True,
    }
    assert report['protocol_consistent'] is True


def test_boundary_case():
    visual_cfg = _visual_cfg()
    visual_cfg['V1']['blackout_prob'] = 0.5
    degraded_events, report = apply_visual_level(_events(), 'V1', visual_cfg)

    vio_events = [event for event in degraded_events if event['modality'] == 'vio']
    assert len(vio_events) == 1
    assert report['blackout_strategy'] == 'stable_hashed_contiguous_vio_window'
    assert report['blackout_selection_start'] == 0
    assert vio_events[0]['vio_payload']['dx'] == pytest.approx(-0.2)
    assert report['blackout_count'] == 2
    assert report['blackout_count_expected'] == 2
    assert report['blackout_segments']
    assert report['consistency_checks']['blackout'] is True
    assert report['consistency_checks']['drift_bias_m'] is True
    assert report['protocol_consistent'] is True


def test_dyaw_is_wrapped_after_visual_drift_bias():
    events = _events()
    # 把第二个 VIO 事件的 dyaw 设为接近 π 的值，使 drift 叠加后需要 wrap。
    events[2]['vio_payload']['dyaw'] = 3.2
    visual_cfg = _visual_cfg()
    visual_cfg['V1']['drift_bias_m'] = 0.5

    degraded_events, report = apply_visual_level(events, 'V1', visual_cfg)

    second_vio = [event for event in degraded_events if event['modality'] == 'vio'][1]['vio_payload']
    # 随机游走模型：第二个 VIO 事件 bias_x 非零，dyaw_bias = bias_x * 0.01 * sign(dyaw_before)。
    # dyaw_after = wrap(dyaw_before + dyaw_bias)，必须在 [-π, π) 区间内。
    import math
    assert -math.pi <= second_vio['dyaw'] < math.pi
    # 一致性：dyaw_after == dyaw_wrapped_from_raw。
    assert report['drift_transform_plan'][1]['dyaw_after'] == pytest.approx(
        report['drift_transform_plan'][1]['dyaw_wrapped_from_raw']
    )
    assert report['consistency_checks']['drift_bias_m'] is True


def test_invalid_case():
    with pytest.raises(ValueError):
        apply_visual_level(
            _events(),
            'V1',
            {
                'V1': {
                    'tracked_features_range': [120, 60],
                    'reproj_err_max': 1.0,
                    'blackout_prob': 0.0,
                    'drift_bias_m': 0.05,
                }
            },
        )


@pytest.mark.parametrize(
    ('field_name', 'value', 'exc_type', 'match'),
    [
        ('tracked_features_range', [60, float('nan')], ValueError, 'tracked_features_range'),
        ('reproj_err_max', float('inf'), ValueError, 'reproj_err_max'),
        ('blackout_prob', float('nan'), ValueError, 'blackout_prob'),
        ('drift_bias_m', float('inf'), ValueError, 'drift_bias_m'),
    ],
)
def test_invalid_case_rejects_non_finite_visual_parameters(field_name, value, exc_type, match):
    visual_cfg = _visual_cfg()
    visual_cfg['V1'][field_name] = value
    with pytest.raises(exc_type, match=match):
        apply_visual_level(_events(), 'V1', visual_cfg)


def test_blackout_ratio_alias():
    """blackout_ratio 应被接受为 blackout_prob 的别名。"""
    source_events = _events()
    new_events_ratio, report_ratio = apply_visual_level(
        source_events,
        'V1',
        {'V1': {'tracked_features_range': [60, 120], 'reproj_err_max': 1.0, 'blackout_ratio': 0.0, 'drift_bias_m': 0.05}},
    )
    new_events_prob, report_prob = apply_visual_level(
        source_events,
        'V1',
        {'V1': {'tracked_features_range': [60, 120], 'reproj_err_max': 1.0, 'blackout_prob': 0.0, 'drift_bias_m': 0.05}},
    )
    assert [e['t'] for e in new_events_ratio] == pytest.approx([e['t'] for e in new_events_prob])
    assert report_ratio['blackout_prob'] == pytest.approx(0.0)


def test_drift_bias_m_zero_allowed():
    """drift_bias_m=0.0 应被允许，表示无漂移。"""
    degraded_events, report = apply_visual_level(
        _events(),
        'V1',
        {'V1': {'tracked_features_range': [60, 120], 'reproj_err_max': 1.0, 'blackout_prob': 0.0, 'drift_bias_m': 0.0}},
    )
    vio_payloads = [e['vio_payload'] for e in degraded_events if e['modality'] == 'vio']
    for payload in vio_payloads:
        # drift_bias_m=0.0 means no drift added, dx/dy/dyaw unchanged
        assert payload['quality'] >= 0.0


def test_visual_quality_monotonically_degrades_with_visual_corruption():
    degraded_events, report = apply_visual_level(_events(), 'V1', _visual_cfg())
    vio_payloads = [event['vio_payload'] for event in degraded_events if event['modality'] == 'vio']

    source_qualities = [0.9, 0.8, 0.7]
    degraded_qualities = [payload['quality'] for payload in vio_payloads]

    assert len(report['quality_plan']) == 3
    for before, after in zip(source_qualities, degraded_qualities, strict=True):
        assert 0.0 <= after <= before


def test_drift_bias_m_negative_rejected():
    """drift_bias_m 为负数应被拒绝。"""
    with pytest.raises(ValueError, match='drift_bias_m'):
        apply_visual_level(
            _events(),
            'V1',
            {'V1': {'tracked_features_range': [60, 120], 'reproj_err_max': 1.0, 'blackout_prob': 0.0, 'drift_bias_m': -0.5}},
        )


def test_blackout_window_depends_on_sequence_identity():
    cfg = {
        'V2': {
            'tracked_features_range': [30, 80],
            'reproj_err_max': 2.0,
            'blackout_prob': 0.4,
            'drift_bias_m': 0.15,
            'blackout_duration_range_s': [0.1, 0.3],
            'blackout_fraction': [0.1, 0.4],
            'keyframe_interruptions_per_seq': 1,
            'keyframe_burst_duration_range_s': [0.5, 2.0],
            'keyframe_burst_fraction': [0.2, 0.6],
            'keyframe_recovery_jump_m': 0.3,
        }
    }
    # 用 10 个 VIO 事件以确保不同 seq_id 的稳定哈希起点分散到不同窗口
    # （3 事件选 1 时哈希碰撞率高，10 个候选 1 个选择足以分离大多数 seq_id）。
    def _make_n_vio_events(n_vio: int, seq_id: str):
        events = [{
            't': 0.0, 'dt': 0.0, 'modality': 'imu',
            'meta': {'scene_id': 'S(A1,N0,V2,K0,M0)', 'seq_id': seq_id},
            'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0},
            'uwb_payload': None, 'vio_payload': None,
        }]
        for i in range(n_vio):
            events.append({
                't': 0.1 * (i + 1), 'dt': 0.1, 'modality': 'vio',
                'meta': {'scene_id': 'S(A1,N0,V2,K0,M0)', 'seq_id': seq_id},
                'imu_payload': None, 'uwb_payload': None,
                'vio_payload': {
                    'dx': 0.5, 'dy': -0.2, 'dyaw': 0.1, 'quality': 0.9,
                    'tracked_features': 150, 'reproj_err': 1.4,
                },
            })
        return events

    alpha_events = _make_n_vio_events(10, 'seq_alpha')
    beta_events = _make_n_vio_events(10, 'seq_beta')

    # 不同 seed 产生不同 blackout 时长；不同 seq_id 决定不同稳定哈希窗口。
    random.seed(42); np.random.seed(42)
    _, alpha_report = apply_visual_level(alpha_events, 'V2', cfg)
    random.seed(99); np.random.seed(99)
    _, beta_report = apply_visual_level(beta_events, 'V2', cfg)

    assert alpha_report['blackout_strategy'] == 'stable_hashed_contiguous_vio_window'
    assert beta_report['blackout_strategy'] == 'stable_hashed_contiguous_vio_window'
    # 同一序列两次调用（相同 seed）结果相同；不同序列（不同 seed）结果不同。
    random.seed(42); np.random.seed(42)
    _, alpha_report_repeat = apply_visual_level(alpha_events, 'V2', cfg)
    assert alpha_report['blackout_segments'] == pytest.approx(alpha_report_repeat['blackout_segments'])
    # 不同序列产生不同 blackout 行为（稳定哈希 + 不同 RNG seed）。
    assert alpha_report['blackout_segments'] != pytest.approx(beta_report['blackout_segments'])
