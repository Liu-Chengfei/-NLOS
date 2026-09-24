from __future__ import annotations

"""可靠性指标（reliability_metrics）测试模块。

文件职责：验证 compute_reliability_metrics 能正确计算
覆盖率、风险-误差相关性、有效对数和可靠性状态。

测试覆盖范围：
- 正常场景：完整对齐的可靠性计算
- 边界场景：部分 None 值导致重叠不足
- 异常场景：长度不匹配、support_context 矛盾、布尔值拒绝

被测模块：liquidloc.metrics.reliability_metrics"""


import pytest

from liquidloc.metrics.reliability_metrics import compute_reliability_metrics


def test_normal_case():
    metrics, support = compute_reliability_metrics(
        [0.0, 1.0],
        [0.0, 1.0],
        support_context={'prediction_length': 4, 'ground_truth_length': 2, 'aligned_length': 2},
    )
    assert metrics['coverage'] == pytest.approx(0.5)
    assert metrics['risk_error_corr'] == pytest.approx(1.0)
    assert support['valid_pair_count'] == 2
    assert support['reliability_status'] == 'ok'


def test_boundary_case():
    metrics, support = compute_reliability_metrics(
        [0.5, None],
        [1.0, 2.0],
        support_context={'prediction_length': 2, 'ground_truth_length': 4, 'aligned_length': 2},
    )
    assert metrics['coverage'] == pytest.approx(0.25)
    assert metrics['risk_error_corr'] == pytest.approx(0.0)
    assert support['valid_pair_count'] == 1
    assert support['reliability_status'] == 'insufficient_overlap'


def test_invalid_case():
    with pytest.raises(ValueError, match='same length'):
        compute_reliability_metrics([0.1], [0.1, 0.2])


def test_support_context_rejects_shorter_source_lengths_than_alignment():
    with pytest.raises(ValueError, match='greater than or equal to the aligned trace length'):
        compute_reliability_metrics(
            [0.0, 1.0],
            [0.0, 1.0],
            support_context={'prediction_length': 1, 'ground_truth_length': 2, 'aligned_length': 2},
        )


def test_bool_trace_values_are_rejected():
    with pytest.raises(ValueError, match='numeric values or None'):
        compute_reliability_metrics([True, 0.5], [0.2, 0.4])
