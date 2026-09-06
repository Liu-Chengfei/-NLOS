from __future__ import annotations

"""运行时指标（runtime_metrics）测试模块。

文件职责：验证 compute_runtime_metrics 能正确计算
延迟均值、p50、p95 以及参数量和内存峰值。

测试覆盖范围：
- 正常场景：3 个延迟值的统计
- 边界场景：单次延迟
- 可迭代输入兼容
- 异常场景：缺失键、非有限值
- 标量输出归一化为 float

被测模块：liquidloc.metrics.runtime_metrics"""


from fractions import Fraction

import pytest

from liquidloc.metrics.runtime_metrics import compute_runtime_metrics


def test_normal_case():
    metrics = compute_runtime_metrics({'latency': [1.0, 2.0, 4.0], 'params': 12.0, 'ram_peak': 3.5})
    assert metrics['latency_mean'] == pytest.approx((1.0 + 2.0 + 4.0) / 3.0)
    assert metrics['latency_p50'] == pytest.approx(2.0)
    assert metrics['latency_p95'] == pytest.approx(3.8)
    assert metrics['params'] == pytest.approx(12.0)
    assert metrics['ram_peak'] == pytest.approx(3.5)


def test_boundary_case():
    metrics = compute_runtime_metrics({'latency': [7.5], 'params': 0.0, 'ram_peak': 1.0})
    assert metrics['latency_mean'] == pytest.approx(7.5)
    assert metrics['latency_p50'] == pytest.approx(7.5)
    assert metrics['latency_p95'] == pytest.approx(7.5)
    assert metrics['params'] == pytest.approx(0.0)
    assert metrics['ram_peak'] == pytest.approx(1.0)


def test_latency_iterable_case():
    metrics = compute_runtime_metrics({'latency': (value for value in [1.0, 2.0, 4.0]), 'params': 5.0, 'ram_peak': 6.0})
    assert metrics['latency_mean'] == pytest.approx((1.0 + 2.0 + 4.0) / 3.0)
    assert metrics['latency_p50'] == pytest.approx(2.0)
    assert metrics['latency_p95'] == pytest.approx(3.8)
    assert metrics['params'] == pytest.approx(5.0)
    assert metrics['ram_peak'] == pytest.approx(6.0)


def test_invalid_case():
    with pytest.raises(KeyError, match='missing required keys'):
        compute_runtime_metrics({'latency': [1.0], 'params': 1.0})


@pytest.mark.parametrize(
    ("runtime_log", "match"),
    [
        ({'latency': [1.0, float('nan')], 'params': 1.0, 'ram_peak': 2.0}, 'latency'),
        ({'latency': [1.0], 'params': float('inf'), 'ram_peak': 2.0}, 'params'),
        ({'latency': [1.0], 'params': 1.0, 'ram_peak': float('-inf')}, 'ram_peak'),
    ],
)
def test_non_finite_runtime_values(runtime_log, match):
    with pytest.raises(ValueError, match=match):
        compute_runtime_metrics(runtime_log)


def test_runtime_scalar_outputs_are_normalized_to_float():
    metrics = compute_runtime_metrics({'latency': [1, 2, 3], 'params': Fraction(12, 1), 'ram_peak': Fraction(7, 2)})
    assert isinstance(metrics['latency_mean'], float)
    assert isinstance(metrics['latency_p50'], float)
    assert isinstance(metrics['latency_p95'], float)
    assert isinstance(metrics['params'], float)
    assert isinstance(metrics['ram_peak'], float)
