from __future__ import annotations

"""特征构建器（feature_builder）测试模块。

测试覆盖范围：
- 特征窗口的构建逻辑
- 特征顺序与缺失掩码
- 时间可靠性特征的提取

被测模块：liquidloc.models.feature_builder"""

from pathlib import Path

import pytest

from liquidloc.models.features.feature_builder import build_feature_state_history, build_feature_vector


def _uwb_event():
    return {
        't': 0.1,
        'dt': 0.0,
        'modality': 'uwb',
        'meta': {'scene_id': 'S(A0,N0,V0,K0,M0)', 'seq_id': 'mini_seq'},
        'imu_payload': None,
        'uwb_payload': {'anchor_id': 0, 'range': 0.0, 'valid': False, 'quality': 0.0},
        'vio_payload': None,
    }


def _imu_event():
    return {
        't': 0.2,
        'dt': 0.1,
        'modality': 'imu',
        'meta': {'scene_id': 'S(A0,N0,V0,K0,M0)', 'seq_id': 'mini_seq'},
        'imu_payload': {'ax': 0.0, 'ay': 0.1, 'gz': 0.0},
        'uwb_payload': None,
        'vio_payload': None,
    }


def _vio_event_missing_required_field():
    return {
        't': 0.3,
        'dt': 0.1,
        'modality': 'vio',
        'meta': {'scene_id': 'S(A0,N0,V0,K0,M0)', 'seq_id': 'mini_seq'},
        'imu_payload': None,
        'uwb_payload': None,
        'vio_payload': {'dx': 0.2, 'dy': -0.1, 'dyaw': 0.0, 'quality': 0.8, 'reproj_err': 1.2},
    }


def _uwb_event_with_state():
    event = _uwb_event()
    event['uwb_payload'] = {'anchor_id': 0, 'range': 0.0, 'valid': False, 'quality': 0.25}
    return event


def _vio_event_with_state():
    event = _vio_event_missing_required_field()
    event['vio_payload'] = {'dx': 0.2, 'dy': -0.1, 'dyaw': 0.0, 'quality': 0.8, 'tracked_features': 11, 'reproj_err': 1.2}
    return event


def _uwb_event_with_unrelated_vio_payload():
    event = _uwb_event()
    event['vio_payload'] = {'dx': 0.0, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.7, 'tracked_features': 77, 'reproj_err': 0.9}
    return event


def test_normal_case():
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    features = build_feature_vector(_uwb_event(), {'px': 1.5}, ['dt', 'range', 'valid', 'quality', 'px'])
    assert features['feature_values'] == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.5])
    assert features['missing_mask'] == [False, False, False, False, False]


def test_bool_payload_fields_are_converted_to_numeric():
    """布尔载荷字段应转为 0.0/1.0 数值特征，且 missing_mask=False 表示已观测。"""
    features = build_feature_vector(_uwb_event(), {}, ['valid'])
    assert features['feature_values'] == pytest.approx([0.0])
    assert features['missing_mask'] == [False]


def test_bool_payload_true_is_converted_to_one():
    """valid=True 应转为 1.0，missing_mask=False。"""
    event = _uwb_event()
    event['uwb_payload'] = {'anchor_id': 0, 'range': 1.0, 'valid': True, 'quality': 0.5}
    features = build_feature_vector(event, {}, ['valid'])
    assert features['feature_values'] == pytest.approx([1.0])
    assert features['missing_mask'] == [False]


def test_boundary_case():
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    features = build_feature_vector(
        _vio_event_missing_required_field(),
        {'px': 2.0, 'yaw': 1.25},
        ['dt', 'tracked_features', 'reproj_err', 'yaw', 'px'],
    )
    assert features['feature_values'] == pytest.approx([0.1, 0.0, 1.2, 1.25, 2.0])
    assert features['missing_mask'] == [False, True, False, False, False]


def test_unrelated_payload_is_ignored():
    features = build_feature_vector(_uwb_event_with_unrelated_vio_payload(), {}, ['tracked_features'])
    assert features['feature_values'] == pytest.approx([0.0])
    assert features['missing_mask'] == [True]


def test_temporal_reliability_pack_is_derived_from_state_ctx():
    state_ctx = {
        'modality_gap_dt': 0.4,
        'uwb_quality_min': 0.2,
        'uwb_invalid_rate': 0.75,
        'vio_reproj_err_slope': 0.3,
        'tracked_features_drop': 6.0,
    }
    features = build_feature_vector(
        _uwb_event_with_state(),
        state_ctx,
        ['modality_gap_dt', 'uwb_quality_min', 'uwb_invalid_rate', 'vio_reproj_err_slope', 'tracked_features_drop'],
    )
    assert features['feature_values'] == pytest.approx([0.4, 0.2, 0.75, 0.3, 6.0])
    assert features['missing_mask'] == [False, False, False, False, False]


def test_temporal_reliability_pack_falls_back_to_event_fields_when_state_missing():
    """缺失测试：temporal reliability pack falls back to event fields when state。\n\n验证 temporal reliability pack falls back to event fields when state 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    features = build_feature_vector(
        _vio_event_with_state(),
        {},
        ['modality_gap_dt', 'uwb_quality_min', 'uwb_invalid_rate', 'vio_reproj_err_slope', 'tracked_features_drop'],
    )
    assert features['feature_values'] == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])  # modality_gap_dt 回退为 None（缺失）
    assert features['missing_mask'] == [True, True, True, True, True]  # 全部缺失


def test_feature_state_history_tracks_temporal_reliability_context():
    """追踪测试：feature state history。\n\n验证 feature state history 的追踪机制，\n确保状态变化被正确记录。
    """
    history = [
        {
            't': 0.0,
            'dt': 0.0,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A3,N3,V2,K0,M0)', 'seq_id': 'mini_seq'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 0, 'range': 1.0, 'valid': True, 'quality': 0.6},
            'vio_payload': None,
        },
        {
            't': 0.05,
            'dt': 0.05,
            'modality': 'imu',
            'meta': {'scene_id': 'S(A3,N3,V2,K0,M0)', 'seq_id': 'mini_seq'},
            'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0},
            'uwb_payload': None,
            'vio_payload': None,
        },
        {
            't': 0.30,
            'dt': 0.25,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A3,N3,V2,K0,M0)', 'seq_id': 'mini_seq'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 0, 'range': 1.4, 'valid': False, 'quality': 0.2},
            'vio_payload': None,
        },
        {
            't': 0.50,
            'dt': 0.20,
            'modality': 'vio',
            'meta': {'scene_id': 'S(A3,N3,V2,K0,M0)', 'seq_id': 'mini_seq'},
            'imu_payload': None,
            'uwb_payload': None,
            # 铁律 3: VIO 紧耦合不再输出 reproj_err / tracked_features 字段。
            # 测试保留这两个字段在 payload 中，验证实现确实忽略它们（不再派生 vio_reproj_err_slope / tracked_features_drop）。
            'vio_payload': {'dx': 0.2, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.7, 'tracked_features': 40, 'reproj_err': 0.8},
        },
        {
            't': 0.80,
            'dt': 0.30,
            'modality': 'vio',
            'meta': {'scene_id': 'S(A3,N3,V2,K0,M0)', 'seq_id': 'mini_seq'},
            'imu_payload': None,
            'uwb_payload': None,
            'vio_payload': {'dx': 0.1, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.6, 'tracked_features': 20, 'reproj_err': 1.4},
        },
    ]
    state_history = [{'px': 0.0, 'py': 0.0} for _ in history]

    feature_state_history = build_feature_state_history(history, state_history)

    assert feature_state_history[2]['modality_gap_dt'] == pytest.approx(0.30)
    assert feature_state_history[2]['uwb_quality_min'] == pytest.approx(0.2)
    assert feature_state_history[2]['uwb_invalid_rate'] == pytest.approx(0.5)
    # 铁律 3 (Stage A1 下游修复, 2026-07-23): VIO 紧耦合不再输出 reproj_err / tracked_features，
    # 因此 vio_reproj_err_slope / tracked_features_drop 不再被计算。
    # 对所有 VIO 事件索引都断言这两个键不存在，验证设计决策被严格执行。
    assert 'vio_reproj_err_slope' not in feature_state_history[3]
    assert 'tracked_features_drop' not in feature_state_history[3]
    assert 'vio_reproj_err_slope' not in feature_state_history[4]
    assert 'tracked_features_drop' not in feature_state_history[4]
    assert feature_state_history[4]['modality_gap_dt'] == pytest.approx(0.30)
    assert feature_state_history[4]['uwb_quality_min'] == pytest.approx(0.2)
    assert feature_state_history[4]['uwb_invalid_rate'] == pytest.approx(0.5)


def test_uwb_geometry_features_are_derived_from_state_anchor_lookup():
    """几何测试：uwb。\n\n验证 uwb 的几何偏置计算，\n确保锚点-目标几何关系正确。
    """
    state_ctx = {
        'px': 1.0,
        'py': 1.5,
        'anchor_lookup': {0: (4.0, 5.5)},
    }
    features = build_feature_vector(
        _uwb_event(),
        state_ctx,
        ['anchor_dx', 'anchor_dy', 'uwb_range_residual'],
    )
    assert features['feature_values'] == pytest.approx([3.0, 4.0, -5.0])
    assert features['missing_mask'] == [False, False, False]


def test_geom_score_is_derived_from_state_anchor_lookup():
    state_ctx = {
        'anchor_lookup': {
            0: (0.0, 0.0),
            1: (2.0, 0.0),
            2: (2.0, 2.0),
            3: (0.0, 2.0),
        },
    }
    features = build_feature_vector(_uwb_event(), state_ctx, ['geom_score'])
    assert features['feature_values'] == pytest.approx([1.0])
    assert features['missing_mask'] == [False]


def test_geom_score_is_missing_for_two_anchor_layouts():
    """缺失测试：geom score is。\n\n验证 geom score is 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    state_ctx = {
        'anchor_lookup': {
            0: (0.0, 0.0),
            1: (2.0, 0.0),
        },
    }
    features = build_feature_vector(_uwb_event(), state_ctx, ['geom_score'])
    assert features['feature_values'] == pytest.approx([0.0])
    assert features['missing_mask'] == [True]


def test_uwb_geometry_features_are_missing_without_anchor_context():
    """无依赖测试：uwb geometry features are missing。\n\n验证 uwb geometry features are missing 在缺少依赖时的降级行为，\n确保回退策略正确。
    """
    features = build_feature_vector(
        _uwb_event(),
        {'px': 1.0, 'py': 1.5},
        ['anchor_dx', 'anchor_dy', 'uwb_range_residual'],
    )
    assert features['feature_values'] == pytest.approx([0.0, 0.0, 0.0])
    assert features['missing_mask'] == [True, True, True]


def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    with pytest.raises(TypeError):
        build_feature_vector(_uwb_event(), {}, 'meta')


def test_model_feature_orders_match_yaml_contract():
    import yaml

    liquid_cfg = yaml.safe_load(Path('configs/models/liquid_ekf.yaml').read_text(encoding='utf-8'))
    lstm_cfg = yaml.safe_load(Path('configs/models/lstm_ekf.yaml').read_text(encoding='utf-8'))

    # 2026-09 契约：feature_order 收敛为 8 维（yaw 不再作为输入特征，
    # yaw 语义由估计器状态承载），网络 input_dim 同步为 8。
    expected_order = [
        'dt', 'ax', 'ay', 'gz', 'range', 'dx', 'dy', 'dyaw',
    ]
    assert liquid_cfg["feature_order"] == expected_order
    assert lstm_cfg["feature_order"] == expected_order
    assert liquid_cfg["network"]["input_dim"] == 8
    assert lstm_cfg["network"]["input_dim"] == 8
