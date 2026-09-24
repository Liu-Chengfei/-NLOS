from __future__ import annotations

"""公共类型（types）测试模块。

文件职责：验证公共数据类型（MetricRow、ModelIntermediate、
StageResult）的定义和默认值稳定性。

测试覆盖范围：
- summarize_types 暴露示例
- 默认值稳定性
- MetricRow 未知方向拒绝

被测模块：liquidloc.common.types"""


import pytest

from liquidloc.common.types import MetricRow, ModelIntermediate, StageResult, summarize_types



def test_summarize_types_exposes_examples():
    summary = summarize_types()
    assert "MetricRow" in summary["dataclasses"]
    assert summary["example_metric_row"]["metric"] == "rmse"
    assert summary["example_measurement_control"]["gate_action"] == "uwb_bias_and_noise_scale"
    assert summary["example_measurement_control"]["noise_multiplier"] == pytest.approx(3.15)  # scaling^2 * (1 + risk) = 1.5^2 * 1.4


def test_defaults_are_stable():
    stage = StageResult(stage_name="x")
    model = ModelIntermediate()
    assert stage.artifacts == []
    assert model.uwb_scaling == 1.0


def test_metric_row_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        MetricRow(
            metric="rmse",
            value=0.0,
            unit="m",
            direction="sideways",
            group="primary",
        )
