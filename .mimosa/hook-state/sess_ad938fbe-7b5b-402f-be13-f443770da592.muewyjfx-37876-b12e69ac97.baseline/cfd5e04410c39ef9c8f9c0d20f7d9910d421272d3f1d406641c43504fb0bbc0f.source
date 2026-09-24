from __future__ import annotations

"""风险适配器（risk_adapter）测试模块。

文件职责：验证 normalize_risk 能将风险值归一化到 [0, 1]，
负值裁剪到 0，并记录完整的归一化报告。

测试覆盖范围：
- 正常场景：合法风险值直接通过
- 边界场景：负值裁剪到 0
- 边界场景：超过上界裁剪到 1
- 异常场景：布尔值拒绝

被测模块：liquidloc.fusion.risk_adapter"""


import pytest

from liquidloc.fusion.risk_adapter import normalize_risk


def test_normal_case():
    normalized_risk, risk_report = normalize_risk(0.42)

    assert normalized_risk == pytest.approx(0.42)
    assert risk_report == {
        "raw_risk": pytest.approx(0.42),
        "clipped_risk": pytest.approx(0.42),
        "normalized_risk": pytest.approx(0.42),
    }


def test_boundary_case():
    normalized_risk, risk_report = normalize_risk(-0.3)

    assert normalized_risk == 0.0
    assert risk_report["raw_risk"] == pytest.approx(-0.3)
    assert risk_report["clipped_risk"] == 0.0
    assert risk_report["normalized_risk"] == 0.0


def test_invalid_case():
    with pytest.raises(TypeError, match="real number"):
        normalize_risk(True)


def test_normalize_risk_clamps_above_max_to_one():
    """risk 超过 risk_max(=1.0) 时被裁剪到 1.0。"""
    normalized_risk, risk_report = normalize_risk(1.5)

    assert normalized_risk == pytest.approx(1.0)
    assert risk_report["raw_risk"] == pytest.approx(1.5)
    assert risk_report["clipped_risk"] == pytest.approx(1.0)
    assert risk_report["normalized_risk"] == pytest.approx(1.0)
