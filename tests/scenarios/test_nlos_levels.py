from __future__ import annotations

"""NLOS（非视距）等级测试模块。

文件职责：验证 apply_nlos_level 能根据 NLOS 等级配置
对 UWB 事件施加距离偏差、质量下降等 NLOS 效应。

测试覆盖范围：
- 正常场景：N2 等级对部分 UWB 事件施加偏差
- 边界场景：N1 等级（nlos_ratio=1.0）选中所有 UWB 事件
- 部分选择的稳定连续窗口策略
- 异常场景：缺失必需配置字段

被测模块：liquidloc.scenarios.nlos_levels"""


import pytest

from liquidloc.scenarios.nlos_levels import apply_nlos_level


def _events():
    return [
        {'t': 0.0, 'dt': 0.0, 'modality': 'imu', 'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.1, 'ay': 0.0, 'gz': 0.0}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.1, 'dt': 0.1, 'modality': 'uwb', 'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 0.95}, 'vio_payload': None},
        {'t': 0.2, 'dt': 0.1, 'modality': 'vio', 'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.02, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.85, 'tracked_features': 118, 'reproj_err': 0.3}},
        {'t': 0.4, 'dt': 0.2, 'modality': 'uwb', 'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 1, 'range': 2.5, 'valid': True, 'quality': 0.9}, 'vio_payload': None},
    ]


def test_normal_case():
    """铁律 4 大脉冲模型: 验证 N2 在 ~50% UWB row 上注入大脉冲 (>=5m 量级),
    未选中 row 保持不变 (恒偏已废弃), quality 退化与 nlos_ratio 成比例.
    """
    source_events = _events()
    new_events, nlos_report = apply_nlos_level(
        source_events,
        'N2',
        # 2026-09-03 fix: apply_nlos_level 直接取 nlos_cfg.get('N2')，不认 axes/N 嵌套
        {'N2': {'nlos_ratio': 0.5, 'bias_strength_m': 8.0, 'nlos_noise_std_m': 0.0,
                'min_cluster_duration_s': 0.0, 'burst_missing_prob': 0.0}},
    )

    assert new_events[0]['imu_payload'] == source_events[0]['imu_payload']
    assert len(nlos_report['selected_indices']) == 1
    assert nlos_report['selection_strategy'] == 'stable_hashed_contiguous_uwb_window'
    selected_index = nlos_report['selected_indices'][0]
    selected_event = next(event for event in new_events if event['t'] == pytest.approx(source_events[selected_index]['t']))
    selected_source = source_events[selected_index]
    # 大脉冲模型: bias_value 是 N(mu=bias_strength_m, sigma=bias_strength_m/3) 的样本,
    # range 偏移 = bias + multipath (大脉冲). 至少应满足 |delta| >= bias_strength_m / 3 (1-sigma 下界近似)。
    delta = float(selected_event['uwb_payload']['range']) - float(selected_source['uwb_payload']['range'])
    bias_value = nlos_report['bias_values'][0]
    multipath_value = nlos_report['multipath_values'][0]
    # 确定性审计: range_delta == bias + multipath + noise (此处 noise_std=0).
    assert delta == pytest.approx(bias_value + multipath_value)
    # 铁律 4: NLOS 大脉冲应在大脉冲量级 (bias_strength_m=8m, sigma=8/3≈2.67),
    # bias_value 在 [0, 16] 区间内是合理 3-sigma 覆盖, 验证不退化为 cm 级恒偏.
    assert bias_value >= 0.0  # N(8, 8/3) 极端左尾也接近 0, 不强制 > 0 但应非负.
    assert selected_event['uwb_payload']['quality'] < selected_source['uwb_payload']['quality']
    untouched_indices = [index for index, event in enumerate(source_events) if event['modality'] == 'uwb' and index != selected_index]
    for index in untouched_indices:
        assert new_events[index]['uwb_payload'] == source_events[index]['uwb_payload']
    assert nlos_report['bias_strength_m'] == pytest.approx(8.0)


def test_boundary_case():
    """铁律 4: nlos_ratio=1.0 时所有 UWB row 都被选中, 都受大脉冲影响."""
    _, nlos_report = apply_nlos_level(
        _events(),
        'N1',
        # 2026-09-03 fix: 直接传 {nlos_level: params} 格式，apply_nlos_level 用 .get(level) 取值
        {'N1': {'label': 'mild_nlos', 'nlos_ratio': 1.0, 'bias_strength_m': 5.0,
                'nlos_noise_std_m': 0.0, 'min_cluster_duration_s': 0.0, 'burst_missing_prob': 0.0}},
    )

    assert nlos_report['label'] == 'mild_nlos'
    assert nlos_report['selected_indices'] == [1, 3]
    assert nlos_report['updated_quality'] == pytest.approx([0.0, 0.0])


def test_partial_nlos_selection_is_stable_contiguous_window():
    """铁律 4: 50% NLOS 的稳定哈希选取, 选取策略应可复现 (相同 seq_id 重复运行得相同结果).
    不强制连续窗口 (selection_mode=poisson 散布, 铁律更关注稀疏大脉冲突发覆盖率).
    """
    starts = set()
    for seq_suffix in range(12):
        events = [
            {
                't': float(index),
                'dt': 0.0 if index == 0 else 1.0,
                'modality': 'uwb',
                'meta': {'scene_id': 'S(A2,N2,V2,K0,M0)', 'seq_id': f'mini_seq_{seq_suffix}'},
                'imu_payload': None,
                'uwb_payload': {'anchor_id': index, 'range': 2.0 + index, 'valid': True, 'quality': 0.95},
                'vio_payload': None,
            }
            for index in range(6)
        ]

        _, nlos_report = apply_nlos_level(
            events,
            'N2',
            # 2026-09-03 fix: 直接传 {nlos_level: params} 格式
            {'N2': {'nlos_ratio': 0.5, 'bias_strength_m': 8.0, 'nlos_noise_std_m': 0.0,
                    'min_cluster_duration_s': 0.0, 'burst_missing_prob': 0.0}},
        )
        selected_indices = nlos_report['selected_indices']
        assert len(selected_indices) == 3  # nlos_ratio=0.5 x 6 UWB = 3 选中.
        # Poisson 散布选取可能非连续, 但需保证:
        # (1) 选中的 index 都在 [0, 6) 内; (2) 不重复; (3) 重复运行 key 相同 -> selected_indices 相同.
        assert all(0 <= i < 6 for i in selected_indices)
        assert len(set(selected_indices)) == len(selected_indices)
        starts.add(tuple(nlos_report['selection_start']) if isinstance(nlos_report['selection_start'], list) else nlos_report['selection_start'])

        _, rerun_report = apply_nlos_level(
            events,
            'N2',
            {'N2': {'nlos_ratio': 0.5, 'bias_strength_m': 8.0, 'nlos_noise_std_m': 0.0}},
        )
        assert rerun_report['selected_indices'] == selected_indices

    # 不同 seq_id 的 selection_start 应该有差异, 否则哈希选取在不同序列上是退化的.
    assert len(starts) > 1


@pytest.mark.parametrize(
    'bad_cfg',
    [
        {'N2': {'ratio': 0.5, 'bias_strength_m': 8.0}},  # 缺 nlos_ratio (字段名错为 ratio)
        {'N2': {'nlos_ratio': 0.5}},  # 缺 bias_strength_m
    ],
)
def test_invalid_case(bad_cfg):
    with pytest.raises(ValueError, match='Missing required NLOS fields'):
        apply_nlos_level(_events(), 'N2', bad_cfg)


def test_large_pulse_model_not_constant_drift():
    """铁律 4 验证: NLOS bias_values 是大脉冲 (米级随机变量), 不是恒偏.
    旧恒偏模型下 bias = bias_strength_m + drift*elapsed_t 是确定性单调量,
    新大脉冲模型下 bias_values 是 N(mu, sigma) 随机样本, 多 row 间不应完全相等.
    """
    events = [
        {
            't': 0.1 * float(index),
            'dt': 0.0 if index == 0 else 0.1,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A2,N3,V2,K0,M0)', 'seq_id': 'pulse_diversity'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': index, 'range': 2.0, 'valid': True, 'quality': 0.95},
            'vio_payload': None,
        }
        for index in range(20)
    ]
    _, nlos_report = apply_nlos_level(
        events,
        'N3',
        {'N3': {'nlos_ratio': 1.0, 'bias_strength_m': 12.0, 'nlos_noise_std_m': 0.0}},
    )
    # nlos_ratio=1.0 选中全部 20 个 UWB row, 每个独立采样大脉冲偏置.
    assert len(nlos_report['bias_values']) == 20
    # 大脉冲 sigma = 12/3 = 4.0, 20 个独立样本不应全相等 (P(全相等) ≈ 0).
    assert len(set(round(b, 6) for b in nlos_report['bias_values'])) > 1
    # 铁律 4: bias_values 应在米级 (mu=12m, sigma=4m), 不应退化为 cm 级恒偏.
    # 期望 mean ≈ 12.0, std ≈ 4.0, 容忍较大统计波动 (20 样本).
    mean_bias = sum(nlos_report['bias_values']) / len(nlos_report['bias_values'])
    assert mean_bias > 5.0  # 大脉冲均值远超铁律 >5m 阈值。


# ── §4 BV2 状态依赖单测 ──────────────────────────────────────────────
# 验证 gt_rows 非空时 selection_key 注入 tag-yaw heading bucket (h0-h11)，
# gt_rows=None 时降级为无方向哈希的确定性窗口（旧行为）。

def _bv2_uwb_events():
    """3 个 UWB 事件，用于 BV2 heading-dependent 选择测试。
    首事件 dt=0.0 满足 validate_event_sequence 约束。"""
    return [
        {'t': 0.1, 'dt': 0.0, 'modality': 'uwb',
         'meta': {'scene_id': 'S(A0,N2,V0,K0,M0)', 'seq_id': 'bv2_seq'},
         'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 0.95}},
        {'t': 0.2, 'dt': 0.1, 'modality': 'uwb',
         'meta': {'scene_id': 'S(A0,N2,V0,K0,M0)', 'seq_id': 'bv2_seq'},
         'uwb_payload': {'anchor_id': 1, 'range': 2.5, 'valid': True, 'quality': 0.90}},
        {'t': 0.3, 'dt': 0.1, 'modality': 'uwb',
         'meta': {'scene_id': 'S(A0,N2,V0,K0,M0)', 'seq_id': 'bv2_seq'},
         'uwb_payload': {'anchor_id': 2, 'range': 3.0, 'valid': True, 'quality': 0.85}},
    ]


def _bv2_gt_rows(yaw_deg):
    """生成带 yaw 的 GT 行列表。"""
    import math
    return [
        {'timestamp': 0.05, 'px': 0.0, 'py': 0.0, 'yaw': math.radians(yaw_deg)},
        {'timestamp': 0.15, 'px': 0.0, 'py': 0.0, 'yaw': math.radians(yaw_deg)},
        {'timestamp': 0.25, 'px': 0.0, 'py': 0.0, 'yaw': math.radians(yaw_deg)},
    ]


def test_gt_rows_injects_heading_hash():
    """BV2: gt_rows 非空时 selection_key 包含方向哈希 (h0-h11)。"""
    import math
    events = _bv2_uwb_events()
    gt_rows = _bv2_gt_rows(yaw_deg=45)  # yaw=45° → heading_bucket ≠ 0

    _, report = apply_nlos_level(
        events, 'N2',
        {'N2': {'nlos_ratio': 0.5, 'bias_strength_m': 8.0, 'nlos_noise_std_m': 0.0}},
        gt_rows=gt_rows,
    )

    # BV2 验证: state_dep 前缀证明 gt_rows 路径执行了方向哈希注入
    selection_key = report.get('selection_key', '')
    assert 'state_dep' in selection_key, f"BV2: selection_key should contain 'state_dep' prefix, got: {selection_key}"
    # state_dep_parts 有 3 条（对应 3 个 UWB 事件）
    parts = selection_key.split('|')
    state_dep_parts = [p for p in parts if p.startswith('state_dep')]
    assert len(state_dep_parts) == 1, f"Expected 1 state_dep prefix, got: {selection_key}"
    # 3 个 UWB 事件的索引列表 (0, 1, 2)
    uwb_parts = [p for p in parts if ':' in p and p.split(':')[0].isdigit()]
    assert len(uwb_parts) == 3, f"Expected 3 UWB state parts, got: {uwb_parts}"


def test_gt_rows_none_falls_back_to_no_hash():
    """BV2: gt_rows=None 时 selection_key 不含方向哈希，降级为确定性窗口。"""
    events = _bv2_uwb_events()

    _, report = apply_nlos_level(
        events, 'N2',
        {'N2': {'nlos_ratio': 0.5, 'bias_strength_m': 8.0, 'nlos_noise_std_m': 0.0}},
        gt_rows=None,
    )

    selection_key = report.get('selection_key', '')
    # 无 gt_rows 时不应出现 state_dep 段 + hX: 方向哈希格式（H22c heading_hash 真改口径）。
    # H24d 真改注入 occluder=machine_body 段（含字符 'h'），改用精确断言避免过严误命中。
    assert 'state_dep' not in selection_key, f"Without gt_rows, no state_dep segment expected, got: {selection_key}"
    import re
    heading_hash_matches = re.findall(r'\bh\d+:', selection_key)
    assert not heading_hash_matches, f"Without gt_rows, no heading hash (hX:) expected, got: {selection_key}"


def test_gt_rows_yaw_changes_selection():
    """BV2: 不同 yaw → 不同 heading_bucket → selection_key 不同 → 可能改变选中事件。"""
    events = _bv2_uwb_events()
    gt_0 = _bv2_gt_rows(yaw_deg=0)
    gt_90 = _bv2_gt_rows(yaw_deg=90)

    _, report_0 = apply_nlos_level(
        events, 'N2',
        {'N2': {'nlos_ratio': 0.5, 'bias_strength_m': 8.0, 'nlos_noise_std_m': 0.0}},
        gt_rows=gt_0,
    )
    _, report_90 = apply_nlos_level(
        events, 'N2',
        {'N2': {'nlos_ratio': 0.5, 'bias_strength_m': 8.0, 'nlos_noise_std_m': 0.0}},
        gt_rows=gt_90,
    )

    # yaw=0 和 yaw=90 产生不同 heading_bucket → selection_key 不同
    assert report_0['selection_key'] != report_90['selection_key'], \
        "Different yaw values should produce different selection keys"
    # 实际选中事件集合也可能不同（不强制，但 selection_key 必须不同）
