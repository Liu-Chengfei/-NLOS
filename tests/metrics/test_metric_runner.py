from __future__ import annotations

"""指标运行器（metric_runner）测试模块。

文件职责：验证 compute_metrics 能正确计算全部指标，
包括轨迹指标、尾部指标、运行时指标和可靠性指标，
并正确处理多序列聚合、对齐、阈值等。

测试覆盖范围：
- 正常场景：单序列全指标计算
- 缩放相关性优先使用 applied 轨迹
- 风险/偏差相关性优先使用 applied 轨迹
- 边界场景：不同失败阈值
- 默认失败阈值跟随协议
- 失败阈值边界不算失败
- 重叠不足报告
- 真值对齐复用训练插值语义
- 多序列池化聚合
- 多序列参数一致性检查
- 混合坐标维度拒绝
- 布尔坐标拒绝
- 重复预测时间戳允许
- 重复真值时间戳拒绝
- 缺失 runtime_log 拒绝
- ground_truth_length 不被放大
- yaw 指标：零误差、已知误差、角度环绕、缺失 yaw
- corr_scaling_error：零缩放、负相关、常数缩放

被测模块：liquidloc.metrics.metric_runner"""


import pytest

from liquidloc.metrics.metric_runner import compute_metrics
from liquidloc.protocol.metric_schema import get_metric_order


def _prediction_bundle(second_px: float = 1.5) -> dict:
    return {
        'seq_id': 'mini_seq',
        'scene_id': 'S(A1,N0,V1,K0,M0)',
        'states': [{'timestamp': 0.0, 'px': 0.0, 'py': 0.0}, {'timestamp': 0.1, 'px': second_px, 'py': 0.0}],
        'timestamps': [0.0, 0.1],
        'diagnostics': {
            'risk_trace': [0.1, 0.9],
            'bias_trace': [0.0, 1.0],
            'modalities': ['uwb', 'vio'],
            'uwb_scaling_trace': [0.0, 0.0],
            'vio_scaling_trace': [0.0, second_px],
        },
        'runtime_log': {'latency': [1.0, 2.0], 'params': 0.0, 'ram_peak': 0.0},
    }


def _gt_bundle() -> dict:
    return {
        'seq_id': 'mini_seq',
        'states': [{'timestamp': 0.0, 'px': 0.0, 'py': 0.0}, {'timestamp': 0.1, 'px': 0.0, 'py': 0.0}],
    }


def _partial_overlap_gt_bundle() -> dict:
    return {
        'seq_id': 'mini_seq',
        'states': [{'timestamp': 0.1, 'px': 0.0, 'py': 0.0}, {'timestamp': 0.2, 'px': 0.0, 'py': 0.0}],
    }


def _bundle_from_errors(
    seq_id: str,
    errors: list[float],
    *,
    latency: list[float],
    params: float = 11.0,
    ram_peak: float = 1.0,
) -> tuple[dict, dict]:
    timestamps = [index * 0.1 for index in range(len(errors))]
    prediction = {
        'seq_id': seq_id,
        'scene_id': f'scene:{seq_id}',
        'states': [{'timestamp': timestamp, 'px': error, 'py': 0.0} for timestamp, error in zip(timestamps, errors)],
        'timestamps': timestamps,
        'diagnostics': {
            'risk_trace': [0.0 for _ in errors],
            'bias_trace': [0.0 for _ in errors],
        },
        'runtime_log': {'latency': latency, 'params': params, 'ram_peak': ram_peak},
    }
    gt = {
        'seq_id': seq_id,
        'states': [{'timestamp': timestamp, 'px': 0.0, 'py': 0.0} for timestamp in timestamps],
        'timestamps': timestamps,
    }
    return prediction, gt


def test_normal_case():
    metrics, support = compute_metrics(_prediction_bundle(), _gt_bundle(), return_support=True)
    assert list(metrics) == get_metric_order()
    assert support['prediction_length'] == 2
    assert support['ground_truth_length'] == 2
    assert support['aligned_length'] == 2
    assert support['valid_pair_count'] == 2
    assert support['overlap_ratio'] == pytest.approx(1.0)
    assert support['reliability_status'] == 'ok'
    assert metrics['corr_scaling_error'] == pytest.approx(1.0)


def test_scaling_correlation_prefers_applied_modality_specific_traces():
    prediction = _prediction_bundle(second_px=2.0)
    prediction['diagnostics'].update(
        {
            'uwb_scaling_trace': [1.0, 1.0],
            'vio_scaling_trace': [1.0, 0.0],
            'applied_uwb_scaling_trace': [0.0, 0.0],
            'applied_vio_scaling_trace': [0.0, 2.0],
        }
    )

    metrics = compute_metrics(prediction, _gt_bundle())
    assert metrics['corr_scaling_error'] == pytest.approx(1.0)


def test_risk_and_bias_correlations_prefer_applied_traces_over_raw_heads():
    prediction = _prediction_bundle(second_px=2.0)
    prediction['diagnostics'].update(
        {
            'risk_trace': [1.0, 0.0],
            'bias_trace': [1.0, 0.0],
            'applied_risk_trace': [0.0, 1.0],
            'applied_bias_trace': [0.0, 1.0],
        }
    )

    metrics = compute_metrics(prediction, _gt_bundle())
    assert metrics['risk_error_corr'] == pytest.approx(1.0)
    assert metrics['bias_alignment'] == pytest.approx(1.0)


def test_boundary_case():
    bundle = _prediction_bundle()
    gt_bundle = _gt_bundle()
    strict_metrics = compute_metrics(bundle, gt_bundle, failure_threshold=1.0)
    loose_metrics = compute_metrics(bundle, gt_bundle, failure_threshold=2.0)
    assert strict_metrics['failure_rate'] == 0.5
    assert loose_metrics['failure_rate'] == 0.0


def test_compute_metrics_default_failure_threshold_can_follow_protocol(monkeypatch):
    monkeypatch.setattr('liquidloc.metrics.metric_runner.get_default_failure_threshold_m', lambda: 0.5)

    metrics = compute_metrics(_prediction_bundle(), _gt_bundle())

    assert metrics['failure_rate'] == pytest.approx(0.5)


def test_failure_threshold_boundary_is_not_counted_as_failure():
    prediction, gt_bundle = _bundle_from_errors(
        'boundary_seq',
        [1.0, 1.0],
        latency=[1.0, 1.0],
        params=11.0,
        ram_peak=1.0,
    )

    metrics = compute_metrics(prediction, gt_bundle, failure_threshold=1.0)
    assert metrics['failure_rate'] == pytest.approx(0.0)


def test_one_point_overlap_reports_insufficient_support():
    metrics, support = compute_metrics(_prediction_bundle(), _partial_overlap_gt_bundle(), return_support=True)
    assert support['prediction_length'] == 2
    assert support['ground_truth_length'] == 2
    assert support['aligned_length'] == 1
    assert support['valid_pair_count'] == 1
    assert support['overlap_ratio'] == pytest.approx(0.5)
    assert support['reliability_status'] == 'insufficient_overlap'
    assert metrics['coverage'] == pytest.approx(1.0)  # 机制指标使用 measurement-level 支持口径
    assert metrics['risk_error_corr'] == pytest.approx(0.0)


def test_ground_truth_alignment_reuses_train_interpolation_semantics():
    prediction = _prediction_bundle(second_px=0.5)
    prediction['states'] = [
        {'timestamp': 0.0, 'px': 0.0, 'py': 0.0},
        {'timestamp': 0.05, 'px': 0.5, 'py': 0.0},
        {'timestamp': 0.2, 'px': 1.0, 'py': 0.0},
    ]
    prediction['timestamps'] = [0.0, 0.05, 0.2]
    prediction['diagnostics'] = {
        'risk_trace': [0.1, 0.2, 0.3],
        'bias_trace': [0.0, 0.0, 0.0],
        'scaling_trace': [0.0, 0.0, 0.0],
    }
    prediction['runtime_log'] = {'latency': [1.0, 1.0, 1.0], 'params': 0.0, 'ram_peak': 0.0}
    gt_bundle = {
        'seq_id': 'mini_seq',
        'states': [
            {'timestamp': 0.0, 'px': 0.0, 'py': 0.0},
            {'timestamp': 0.1, 'px': 1.0, 'py': 0.0},
        ],
    }

    metrics, support = compute_metrics(prediction, gt_bundle, return_support=True)
    assert support['aligned_length'] == 3
    assert support['valid_pair_count'] == 3
    assert metrics['rmse'] == pytest.approx(0.0)
    assert metrics['mae'] == pytest.approx(0.0)


def test_multi_sequence_uses_pooled_aggregation():
    short_pred, short_gt = _bundle_from_errors(
        'short_seq',
        [0.0, 2.0],
        latency=[1.0, 100.0],
        params=11.0,
        ram_peak=3.0,
    )
    long_pred, long_gt = _bundle_from_errors(
        'long_seq',
        [0.0] * 100,
        latency=[10.0] * 100,
        params=11.0,
        ram_peak=7.0,
    )

    metrics, support = compute_metrics(
        [short_pred, long_pred],
        [short_gt, long_gt],
        failure_threshold=1.0,
        return_support=True,
    )

    assert list(metrics) == get_metric_order()
    assert metrics['failure_rate'] == pytest.approx(1 / 102)
    assert metrics['p95'] == pytest.approx(0.0)
    assert metrics['p99'] == pytest.approx(0.0)
    assert metrics['rpe'] == pytest.approx(0.02)
    assert metrics['latency_p95'] == pytest.approx(10.0)
    assert metrics['params'] == pytest.approx(11.0)
    assert metrics['ram_peak'] == pytest.approx(7.0)
    assert support['prediction_length'] == 102
    assert support['ground_truth_length'] == 102
    assert support['aligned_length'] == 102
    assert support['valid_pair_count'] == 102
    assert support['overlap_ratio'] == pytest.approx(1.0)
    assert support['reliability_status'] == 'ok'


def test_multi_sequence_requires_consistent_params():
    first_pred, first_gt = _bundle_from_errors('seq_a', [0.0, 1.0], latency=[1.0, 2.0], params=11.0)
    second_pred, second_gt = _bundle_from_errors('seq_b', [0.0, 0.0], latency=[3.0, 4.0], params=12.0)

    with pytest.raises(ValueError, match='runtime params must match'):
        compute_metrics([first_pred, second_pred], [first_gt, second_gt])


def test_multi_sequence_rejects_mixed_coordinate_dimensionality():
    bad_pred = _prediction_bundle()
    bad_pred['seq_id'] = 'mixed_dim_pred'
    bad_pred['states'] = [
        {'timestamp': 0.0, 'px': 0.0, 'py': 0.0, 'pz': 0.0},
        {'timestamp': 0.1, 'px': 1.5, 'py': 0.0},
    ]
    bad_gt = {
        'seq_id': 'mixed_dim_pred',
        'states': [
            {'timestamp': 0.0, 'px': 0.0, 'py': 0.0, 'pz': 0.0},
            {'timestamp': 0.1, 'px': 0.0, 'py': 0.0, 'pz': 0.0},
        ],
    }
    good_pred, good_gt = _bundle_from_errors('seq_ok', [0.0, 0.0], latency=[1.0, 1.0])

    with pytest.raises(ValueError, match='consistent coordinate fields'):
        compute_metrics([bad_pred, good_pred], [bad_gt, good_gt])


def test_rejects_boolean_coordinates():
    prediction = _prediction_bundle()
    prediction['states'] = [
        {'timestamp': 0.0, 'px': True, 'py': 0.0},
        {'timestamp': 0.1, 'px': 1.5, 'py': 0.0},
    ]

    with pytest.raises(ValueError, match="trajectory coordinate 'px' must be numeric"):
        compute_metrics(prediction, _gt_bundle())


def test_allows_duplicate_prediction_timestamps_for_event_level_async_updates():
    prediction = _prediction_bundle()
    prediction['states'] = [
        {'timestamp': 0.0, 'px': 0.0, 'py': 0.0},
        {'timestamp': 0.0, 'px': 1.5, 'py': 0.0},
    ]
    prediction['timestamps'] = [0.0, 0.0]

    metrics, support = compute_metrics(prediction, _gt_bundle(), return_support=True)
    assert metrics['rmse'] == pytest.approx((1.5**2 / 2) ** 0.5)
    assert support['prediction_length'] == 2
    assert support['aligned_length'] == 2
    assert support['valid_pair_count'] == 2


def test_duplicate_prediction_timestamps_do_not_turn_aligned_ground_truth_into_false_duplicate_case():
    prediction = _prediction_bundle(second_px=0.5)
    prediction['states'] = [
        {'timestamp': 0.0, 'px': 0.0, 'py': 0.0},
        {'timestamp': 0.0, 'px': 0.5, 'py': 0.0},
        {'timestamp': 0.1, 'px': 1.0, 'py': 0.0},
    ]
    prediction['timestamps'] = [0.0, 0.0, 0.1]
    prediction['diagnostics'] = {
        'risk_trace': [0.1, 0.2, 0.3],
        'bias_trace': [0.0, 0.0, 0.0],
        'scaling_trace': [0.0, 0.0, 0.0],
    }
    prediction['runtime_log'] = {'latency': [1.0, 1.0, 1.0], 'params': 0.0, 'ram_peak': 0.0}
    gt_bundle = {
        'seq_id': 'mini_seq',
        'states': [
            {'timestamp': 0.0, 'px': 0.0, 'py': 0.0},
            {'timestamp': 0.1, 'px': 1.0, 'py': 0.0},
        ],
    }

    metrics, support = compute_metrics(prediction, gt_bundle, return_support=True)
    assert metrics['rmse'] == pytest.approx((0.5**2 / 3.0) ** 0.5)
    assert support['aligned_length'] == 3
    assert support['valid_pair_count'] == 3


def test_rejects_duplicate_ground_truth_timestamps():
    gt_bundle = _gt_bundle()
    gt_bundle['states'] = [
        {'timestamp': 0.0, 'px': 0.0, 'py': 0.0},
        {'timestamp': 0.0, 'px': 0.0, 'py': 0.0},
    ]

    with pytest.raises(ValueError, match="ground-truth trajectory contains duplicate 'timestamp' values"):
        compute_metrics(_prediction_bundle(), gt_bundle)


def test_invalid_case():
    with pytest.raises(KeyError):
        compute_metrics(
            {key: value for key, value in _prediction_bundle().items() if key != 'runtime_log'},
            _gt_bundle(),
        )


def test_ground_truth_length_is_raw_not_max_amplified():
    """ground_truth_length 不被 max(prediction_length, ground_truth_length) 放大。"""
    prediction = _prediction_bundle()
    # prediction 有 2 个点
    gt_bundle = {
        'seq_id': 'mini_seq',
        'states': [
            {'timestamp': 0.0, 'px': 0.0, 'py': 0.0},
            {'timestamp': 0.05, 'px': 0.0, 'py': 0.0},
            {'timestamp': 0.1, 'px': 0.0, 'py': 0.0},
        ],
    }
    # gt 有 3 个点，prediction 有 2 个点
    metrics, support = compute_metrics(prediction, gt_bundle, return_support=True)
    assert support['prediction_length'] == 2
    assert support['ground_truth_length'] == 3  # 原始值，不是 max(2,3)=3
    # overlap_ratio 分母仍用 max
    assert support['overlap_ratio'] == pytest.approx(support['aligned_length'] / max(2, 3))


# ── yaw 指标专项测试 ──────────────────────────────────────


def _prediction_bundle_with_yaw(yaw_values: list[float | None], *, px_values: list[float] | None = None) -> dict:
    """构造带 yaw 字段的预测 bundle。"""
    n = len(yaw_values)
    if px_values is None:
        px_values = [0.0] * n
    return {
        'seq_id': 'yaw_seq',
        'scene_id': 'S(A0,N0,V0,K0,M0)',
        'states': [
            {'timestamp': i * 0.1, 'px': px_values[i], 'py': 0.0, 'yaw': yaw_values[i]}
            for i in range(n)
        ],
        'timestamps': [i * 0.1 for i in range(n)],
        'diagnostics': {
            'risk_trace': [0.0] * n,
            'bias_trace': [0.0] * n,
        },
        'runtime_log': {'latency': [1.0] * n, 'params': 0.0, 'ram_peak': 0.0},
    }


def _gt_bundle_with_yaw(yaw_values: list[float]) -> dict:
    """构造带 yaw 字段的真值 bundle。"""
    n = len(yaw_values)
    return {
        'seq_id': 'yaw_seq',
        'states': [
            {'timestamp': i * 0.1, 'px': 0.0, 'py': 0.0, 'yaw': yaw_values[i]}
            for i in range(n)
        ],
    }


def test_yaw_metrics_zero_error():
    """预测 yaw 与真值完全一致时，yaw_rmse 和 yaw_p95 均为 0。"""
    import math
    yaw_vals = [0.0, 0.5, -0.3, math.pi - 0.1]
    pred = _prediction_bundle_with_yaw(yaw_vals, px_values=[0.0, 0.0, 0.0, 0.0])
    gt = _gt_bundle_with_yaw(yaw_vals)
    metrics = compute_metrics(pred, gt)
    assert metrics['yaw_rmse'] == pytest.approx(0.0)
    assert metrics['yaw_p95'] == pytest.approx(0.0)


def test_yaw_metrics_known_error():
    """已知航向误差下 yaw_rmse 和 yaw_p95 计算正确。"""
    pred_yaw = [0.0, 0.1, 0.0, 0.0]
    gt_yaw = [0.0, 0.0, 0.0, 0.0]
    metrics = compute_metrics(
        _prediction_bundle_with_yaw(pred_yaw, px_values=[0.0, 0.0, 0.0, 0.0]),
        _gt_bundle_with_yaw(gt_yaw),
    )
    # 误差序列: [0.0, 0.1, 0.0, 0.0]
    expected_rmse = (0.1 ** 2 / 4) ** 0.5
    assert metrics['yaw_rmse'] == pytest.approx(expected_rmse)
    assert metrics['yaw_p95'] >= 0.0


def test_yaw_metrics_angle_wrapping():
    """航向角环绕 [-π, π) 时 yaw 误差正确处理。"""
    import math
    pred_yaw = [math.pi - 0.01]  # 接近 π
    gt_yaw = [-math.pi + 0.01]  # 接近 -π
    metrics = compute_metrics(
        _prediction_bundle_with_yaw(pred_yaw),
        _gt_bundle_with_yaw(gt_yaw),
    )
    # 角度差应为 ~0.02 rad，而非 ~2π
    assert metrics['yaw_rmse'] < 0.1
    assert metrics['yaw_p95'] < 0.1


def test_yaw_metrics_missing_yaw_returns_zero():
    """预测或真值缺少 yaw 字段时，yaw_rmse 和 yaw_p95 返回 0。"""
    metrics = compute_metrics(_prediction_bundle(), _gt_bundle())
    assert metrics['yaw_rmse'] == pytest.approx(0.0)
    assert metrics['yaw_p95'] == pytest.approx(0.0)


# ── corr_scaling_error 零值/负相关专项测试 ──────────────────


def test_corr_scaling_error_zero_scaling():
    """缩放轨迹全为零时，方差为零，corr_scaling_error 返回 0。"""
    prediction = _prediction_bundle(second_px=1.5)
    prediction['diagnostics']['uwb_scaling_trace'] = [0.0, 0.0]
    prediction['diagnostics']['vio_scaling_trace'] = [0.0, 0.0]
    metrics = compute_metrics(prediction, _gt_bundle())
    assert metrics['corr_scaling_error'] == pytest.approx(0.0)


def test_corr_scaling_error_negative_correlation():
    """缩放与误差负相关时，corr_scaling_error 为负值。"""
    # 使用 applied_scaling_trace 统一缩放轨迹，绕过 modalities 分派。
    # 误差轨迹: [0.0, 1.5]（px=1.5 vs gt px=0.0）
    # 缩放轨迹: [1.0, 0.0] — 缩放高时误差低，缩放低时误差高 → 负相关
    prediction = _prediction_bundle(second_px=1.5)
    prediction['diagnostics']['applied_scaling_trace'] = [1.0, 0.0]
    metrics = compute_metrics(prediction, _gt_bundle())
    assert metrics['corr_scaling_error'] < 0.0


def test_corr_scaling_error_constant_scaling():
    """缩放轨迹为常数时方差为零，corr_scaling_error 返回 0。"""
    prediction = _prediction_bundle(second_px=1.5)
    prediction['diagnostics']['uwb_scaling_trace'] = [0.5, 0.5]
    prediction['diagnostics']['vio_scaling_trace'] = [0.5, 0.5]
    metrics = compute_metrics(prediction, _gt_bundle())
    assert metrics['corr_scaling_error'] == pytest.approx(0.0)


def _prediction_bundle_with_scenario_context(
    second_px: float = 1.5,
    takeoff_landing_mask: list[bool] | None = None,
    floor_transition_mask: list[bool] | None = None,
) -> dict:
    """构造含 scenario_context 的预测 bundle，用于 §9 场景制度测试."""
    bundle = _prediction_bundle(second_px=second_px)
    scenario_context = {}
    if takeoff_landing_mask is not None:
        scenario_context['takeoff_landing_mask'] = takeoff_landing_mask
    if floor_transition_mask is not None:
        scenario_context['floor_transition_mask'] = floor_transition_mask
    if scenario_context:
        bundle['scenario_context'] = scenario_context
    return bundle


def test_takeoff_landing_exclude_unified_filters_takeoff_frames():
    """takeoff_landing_policy=exclude_unified 时，takeoff_landing_mask=True 的帧从 measurement_mask 剔除."""
    pred = _prediction_bundle_with_scenario_context(
        takeoff_landing_mask=[True, False],  # 第 0 帧是起飞段，排除
    )
    gt = _gt_bundle()
    metrics, support = compute_metrics(pred, gt, protocol_cfg={'scene_scale': {'takeoff_landing_policy': 'exclude_unified'}}, return_support=True)
    # measurement_mask 反映制度段剔除：第 0 帧被剔除，第 1 帧保留
    assert support['measurement_mask'] == [False, True]


def test_takeoff_landing_include_unified_skips_filtering():
    """takeoff_landing_policy=include_unified 时，mask 不触发过滤，2 帧均保留."""
    pred = _prediction_bundle_with_scenario_context(
        takeoff_landing_mask=[True, False],
    )
    gt = _gt_bundle()
    metrics, support = compute_metrics(pred, gt, protocol_cfg={'scene_scale': {'takeoff_landing_policy': 'include_unified'}}, return_support=True)
    assert support['aligned_length'] == 2  # 2 帧均保留
    assert support['measurement_mask'] == [True, True]


def test_floor_transition_exclude_unified_filters_floor_frames():
    """floor_transition_policy=exclude_unified 时，floor_transition_mask=True 的帧从 measurement_mask 剔除."""
    pred = _prediction_bundle_with_scenario_context(
        floor_transition_mask=[False, True],  # 第 1 帧是楼层切换段，排除
    )
    gt = _gt_bundle()
    metrics, support = compute_metrics(pred, gt, protocol_cfg={'scene_scale': {'floor_transition_policy': 'exclude_unified'}}, return_support=True)
    # measurement_mask 反映制度段剔除：第 1 帧被剔除，第 0 帧保留
    assert support['measurement_mask'] == [True, False]


def test_missing_takeoff_mask_with_exclude_unified_skips_filtering():
    """缺 takeoff_landing_mask 且 policy=exclude_unified 时不报错，跳过过滤."""
    pred = _prediction_bundle_with_scenario_context()  # 无 takeoff_landing_mask
    gt = _gt_bundle()
    metrics, support = compute_metrics(pred, gt, protocol_cfg={'scene_scale': {'takeoff_landing_policy': 'exclude_unified'}}, return_support=True)
    assert support['aligned_length'] == 2  # 无 mask，不做过滤
    assert support['measurement_mask'] == [True, True]


def test_manual_guard_policy_skips_filtering():
    """takeoff_landing_policy=manual_guard 时，不自动过滤（需调用方自行审计）."""
    pred = _prediction_bundle_with_scenario_context(
        takeoff_landing_mask=[True, False],
    )
    gt = _gt_bundle()
    metrics, support = compute_metrics(pred, gt, protocol_cfg={'scene_scale': {'takeoff_landing_policy': 'manual_guard'}}, return_support=True)
    assert support['aligned_length'] == 2  # manual_guard 不自动过滤
    assert support['measurement_mask'] == [True, True]


def test_takeoff_landing_mask_length_mismatch_degrades_gracefully():
    """takeoff_landing_mask 长度与轨迹不匹配时降级为不过滤，不抛异常，发出警告."""
    pred = _prediction_bundle_with_scenario_context(
        takeoff_landing_mask=[True],  # 长度 1 ≠ 轨迹长度 2
    )
    gt = _gt_bundle()
    # 长度不匹配时发出警告并降级为不过滤
    metrics, support = compute_metrics(pred, gt, protocol_cfg={'scene_scale': {'takeoff_landing_policy': 'exclude_unified'}}, return_support=True)
    assert support['aligned_length'] == 2  # 降级为不过滤，保留全部 2 帧


def test_cold_start_global_enforced_false_emits_warning():
    """cold_start_global_enforced=False 且 cold_start_offset_s>0 时不报错，发出 §9.1 警告."""
    pred = _prediction_bundle()
    gt = _gt_bundle()
    protocol_cfg = {
        'scene_scale': {
            'cold_start_offset_s': 0.05,  # 启用冷启动
            'cold_start_global_enforced': False,  # 但全员不统一
        }
    }
    with pytest.warns(UserWarning, match='cold_start_global_enforced=False'):
        compute_metrics(pred, gt, protocol_cfg=protocol_cfg)


def test_t_eff_min_s_violation_emits_warning():
    """T_eff < t_eff_min_s 时发出 §9.1 警告（软约束不阻断）."""
    pred = _prediction_bundle()
    gt = _gt_bundle()
    protocol_cfg = {
        'scene_scale': {
            't_eff_min_s': 100.0,  # 远超实际段 0.1s
        }
    }
    with pytest.warns(UserWarning, match='T_eff 守门违反'):
        compute_metrics(pred, gt, protocol_cfg=protocol_cfg)
