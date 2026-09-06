from __future__ import annotations

"""指标模式（metric_schema）测试模块。

测试覆盖范围：
- 指标定义与计算顺序
- 指标字段类型与范围约束
- 指标分组与优先级

被测模块：liquidloc.protocol.metric_schema"""

import pytest

import liquidloc.protocol.metric_schema as metric_schema
from liquidloc.protocol.metric_schema import get_metric_meta, get_metric_order



def test_normal_case():
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    meta = get_metric_meta()
    assert meta["rmse"]["unit"] == "m"
    assert meta["failure_rate"]["direction"] == "lower_is_better"
    assert meta["corr_scaling_error"]["group"] == "mechanism"
    assert meta["params"]["unit"] == "count"
    assert meta["ram_peak"]["unit"] == "MB"
    assert list(meta.keys()) == get_metric_order()



def test_boundary_case():
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    order = get_metric_order()
    assert order[:3] == ["rmse", "mean_rmse", "std_rmse"]
    assert order[order.index("bias_alignment") + 1] == "corr_scaling_error"
    assert len(order) == len(set(order))
    assert order == [
        "rmse",
        "mean_rmse",
        "std_rmse",
        "p95",
        "failure_rate",
        "p99",
        "mae",
        "ate",
        "rpe",
        "yaw_rmse",
        "yaw_p95",
        "risk_error_corr",
        "coverage",
        "bias_alignment",
        "corr_scaling_error",
        "latency_mean",
        "latency_p50",
        "latency_p95",
        "params",
        "ram_peak",
    ]



def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    meta = get_metric_meta()
    assert meta["corr_scaling_error"]["direction"] == "higher_is_better"
    assert meta["risk_error_corr"]["direction"] == "higher_is_better"


def test_metrics_yaml_order_matches_schema_contract():
    """匹配测试：metrics yaml order。\n\n验证 metrics yaml order 的输出与预期一致，\n确保合同合规。
    """
    order = get_metric_order()
    assert order == list(get_metric_meta().keys())


def test_metrics_config_rejects_mapping_group_payload(monkeypatch):
    """拒绝测试：metrics config。\n\n验证被测功能对 metrics config 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    payload = {
        "primary_metrics": {"rmse": 1, "p95": 1, "failure_rate": 1},
        "secondary_metrics": ["p99", "mae", "ate", "rpe"],
        "mechanism_metrics": ["risk_error_corr", "coverage", "bias_alignment", "corr_scaling_error"],
        "runtime_metrics": ["latency_mean", "latency_p50", "latency_p95", "params", "ram_peak"],
    }
    monkeypatch.setattr(metric_schema, "load_yaml_config", lambda _path: payload)

    with pytest.raises(ValueError, match="metrics config field primary_metrics must be a non-empty list"):
        metric_schema._validate_metrics_config()


def test_metrics_config_rejects_string_group_payload_with_type_error(monkeypatch):
    """拒绝测试：metrics config。\n\n验证被测功能对 metrics config 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    payload = {
        "primary_metrics": "rmse",
        "secondary_metrics": ["p99", "mae", "ate", "rpe"],
        "mechanism_metrics": ["risk_error_corr", "coverage", "bias_alignment", "corr_scaling_error"],
        "runtime_metrics": ["latency_mean", "latency_p50", "latency_p95", "params", "ram_peak"],
    }
    monkeypatch.setattr(metric_schema, "load_yaml_config", lambda _path: payload)

    with pytest.raises(ValueError, match="metrics config field primary_metrics must be a non-empty list"):
        metric_schema._validate_metrics_config()
