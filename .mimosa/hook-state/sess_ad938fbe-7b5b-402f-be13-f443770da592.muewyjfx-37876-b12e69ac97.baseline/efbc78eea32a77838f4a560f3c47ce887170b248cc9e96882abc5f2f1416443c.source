
"""尾部指标（tail_metrics）测试模块。

文件职责：验证 compute_tail_metrics 能正确计算 p95、p99、
失败率和长失败段等尾部统计量。

测试覆盖范围：
- 正常场景：4 个误差值的尾部统计
- 边界场景：误差恰好在阈值上
- 默认失败阈值跟随协议
- 异常场景：空列表、布尔值、NaN、非法阈值

被测模块：liquidloc.metrics.tail_metrics"""

import pytest

from liquidloc.metrics.tail_metrics import compute_tail_metrics


def test_normal_case():
    errors = [0.2, 1.4, 1.3, 0.6]

    result, support = compute_tail_metrics(errors)

    assert set(result) == {"p95", "p99", "failure_rate"}
    assert "long_failure_segments" in support
    assert result["p95"] == pytest.approx(1.385)  # 线性插值: sorted[2]+0.85*(sorted[3]-sorted[2])=1.3+0.85*0.1=1.385
    assert result["p99"] == pytest.approx(1.397)  # 线性插值: sorted[2]+0.97*(sorted[3]-sorted[2])=1.3+0.97*0.1=1.397
    assert result["failure_rate"] == pytest.approx(0.5)
    assert support["long_failure_segments"] == [(1, 2)]


def test_boundary_case():
    errors = [1.0, 0.3, 1.0]

    result, support = compute_tail_metrics(errors)

    assert result["p95"] == pytest.approx(1.0)
    assert result["p99"] == pytest.approx(1.0)
    assert result["failure_rate"] == pytest.approx(0.0)
    assert support["long_failure_segments"] == []


def test_default_failure_threshold_can_follow_protocol(monkeypatch):
    monkeypatch.setattr('liquidloc.metrics.tail_metrics.get_default_failure_threshold_m', lambda: 0.5)

    result, _ = compute_tail_metrics([0.4, 0.6])

    assert result["failure_rate"] == pytest.approx(0.5)


def test_invalid_case():
    with pytest.raises(ValueError, match="non-empty"):
        compute_tail_metrics([])

    with pytest.raises(TypeError, match="errors\\[0\\] must be numeric"):
        compute_tail_metrics([True, 0.2])

    with pytest.raises(ValueError, match="errors\\[0\\] must be finite"):
        compute_tail_metrics([float("nan"), 0.2])

    with pytest.raises(TypeError, match="failure_threshold must be a real number"):
        compute_tail_metrics([0.2, 0.4], failure_threshold=True)

    with pytest.raises(ValueError, match="failure_threshold must be finite"):
        compute_tail_metrics([0.2, 0.4], failure_threshold=float("inf"))
