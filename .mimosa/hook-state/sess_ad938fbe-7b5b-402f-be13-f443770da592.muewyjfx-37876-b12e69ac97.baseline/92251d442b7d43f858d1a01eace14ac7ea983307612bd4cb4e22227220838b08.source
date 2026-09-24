from __future__ import annotations

"""H25c 真改 §4.2 L894「差 GDOP 占比协议层下界守门」测试模块。

文件职责：验证 _aggregate_gdop_occupancy / _compute_single/_multi_sequence_metrics /
compute_metrics 的 H25c 真改协议层显式冻结：
- protocol_cfg.evaluation.gdop_occupancy_min_ratio 设定时，ratio_above_floor < 阈值
  且 count_measured > 0 必须抛 ValueError 阻断评估
- protocol_cfg 为 None 或缺 evaluation.gdop_occupancy_min_ratio 字段时跳过守门（向后兼容）
- count_measured == 0 时不守门（向后兼容旧 bundle 无 geometry_report）
- min_ratio_threshold 显式回写到聚合报告，便于审计「阈值-观测值」对照
- compute_metrics 入口 protocol_cfg kwarg 透传到 _compute_single/_multi_sequence_metrics

被测模块：liquidloc.metrics.metric_runner._aggregate_gdop_occupancy
        liquidloc.metrics.metric_runner.compute_metrics
        liquidloc.metrics.metric_runner._compute_single_sequence_metrics
        liquidloc.metrics.metric_runner._compute_multi_sequence_metrics
"""

from typing import Any, Mapping

import pytest

from liquidloc.metrics.metric_runner import (
    _aggregate_gdop_occupancy,
    _collect_sequence_data,
    _compute_single_sequence_metrics,
    _compute_multi_sequence_metrics,
    compute_metrics,
)


def _seq_data_with_gdop(value: float | None) -> dict[str, Any]:
    """构造含 support_context.gdop_occupancy_above_floor 的 sequence_data（轻量）。"""
    return {
        "support_context": {"gdop_occupancy_above_floor": value},
    }


def _protocol_cfg(min_ratio: float | None) -> Mapping[str, Any]:
    """构造含 evaluation.gdop_occupancy_min_ratio 的协议层配置。"""
    if min_ratio is None:
        return {"evaluation": {}}  # 缺字段
    return {"evaluation": {"gdop_occupancy_min_ratio": min_ratio}}


def test_h25c_protocol_cfg_below_threshold_raises_value_error():
    """ratio_above_floor < gdop_occupancy_min_ratio 且 count_measured > 0 必须抛 ValueError。"""
    sequence_data_group = [
        _seq_data_with_gdop(1.0),  # above floor
        _seq_data_with_gdop(0.0),  # at/below floor → ratio_above_floor = 0.5
        _seq_data_with_gdop(0.0),
    ]
    # ratio_above_floor = 1/3 ≈ 0.333 < 0.5 阈值
    with pytest.raises(ValueError, match=r"gdop_occupancy ratio_above_floor.*<.*gdop_occupancy_min_ratio"):
        _aggregate_gdop_occupancy(
            sequence_data_group,
            protocol_cfg=_protocol_cfg(0.5),
        )


def test_h25c_protocol_cfg_above_threshold_passes():
    """ratio_above_floor >= gdop_occupancy_min_ratio 时正常返回聚合字典。"""
    sequence_data_group = [
        _seq_data_with_gdop(1.0),
        _seq_data_with_gdop(1.0),
    ]
    result = _aggregate_gdop_occupancy(
        sequence_data_group,
        protocol_cfg=_protocol_cfg(0.5),
    )
    assert result is not None
    assert result["ratio_above_floor"] == 1.0
    assert result["min_ratio_threshold"] == 0.5


def test_h25c_protocol_cfg_none_skips_gate():
    """protocol_cfg=None 时跳过守门（向后兼容旧调用方）。"""
    sequence_data_group = [_seq_data_with_gdop(0.0)]  # ratio_above_floor = 0.0
    result = _aggregate_gdop_occupancy(sequence_data_group, protocol_cfg=None)
    assert result is not None
    assert result["ratio_above_floor"] == 0.0
    assert result["min_ratio_threshold"] is None


def test_h25c_protocol_cfg_missing_field_skips_gate():
    """protocol_cfg.evaluation.gdop_occupancy_min_ratio 缺字段时跳过守门。"""
    sequence_data_group = [_seq_data_with_gdop(0.0)]
    result = _aggregate_gdop_occupancy(
        sequence_data_group,
        protocol_cfg=_protocol_cfg(None),  # 缺字段
    )
    assert result is not None
    assert result["min_ratio_threshold"] is None


def test_h25c_count_measured_zero_skips_gate():
    """所有序列无 geometry_report（count_measured=0）时跳过守门（向后兼容旧 bundle）。"""
    sequence_data_group = [
        _seq_data_with_gdop(None),
        _seq_data_with_gdop(None),
    ]
    # 不应抛错即使阈值设为 0.99
    result = _aggregate_gdop_occupancy(
        sequence_data_group,
        protocol_cfg=_protocol_cfg(0.99),
    )
    assert result is None  # 无测量值


def test_h25c_min_ratio_threshold_written_back_to_aggregation_report():
    """min_ratio_threshold 必须显式回写到聚合报告（便于审计「阈值-观测值」对照）。"""
    sequence_data_group = [_seq_data_with_gdop(1.0)]
    result = _aggregate_gdop_occupancy(
        sequence_data_group,
        protocol_cfg=_protocol_cfg(0.3),
    )
    assert result is not None
    assert "min_ratio_threshold" in result
    assert result["min_ratio_threshold"] == 0.3


def test_h25c_invalid_threshold_value_skips_gate():
    """协议层 min_ratio 为非数值/NaN/inf 时跳过守门（不让守门被脏配置误触）。"""
    sequence_data_group = [_seq_data_with_gdop(0.0)]
    # 非数值字符串
    result_str = _aggregate_gdop_occupancy(
        sequence_data_group,
        protocol_cfg={"evaluation": {"gdop_occupancy_min_ratio": "invalid"}},
    )
    assert result_str is not None
    assert result_str["min_ratio_threshold"] is None
    # inf
    result_inf = _aggregate_gdop_occupancy(
        sequence_data_group,
        protocol_cfg={"evaluation": {"gdop_occupancy_min_ratio": float("inf")}},
    )
    assert result_inf is not None
    # math.isfinite(inf) = False → 跳过守门
    assert result_inf["min_ratio_threshold"] is None or result_inf["min_ratio_threshold"] == float("inf")


def test_h25c_compute_metrics_passes_protocol_cfg_through_to_aggregation():
    """compute_metrics 入口 protocol_cfg kwarg 必须透传到 _aggregate_gdop_occupancy。"""
    # 构造 1 个序列 prediction_obj/gt_obj + 阈值 0.99 守门触发
    # gdop_above_floor_ratio = 0.0 (< 0.99) → 触发 ValueError
    pred = {
        "seq_id": "seq_a",
        "scene_id": "S(A0,N0,V0,K0,M0)",
        "states": [
            {"timestamp": 0.0, "px": 0.0, "py": 0.0},
            {"timestamp": 0.1, "px": 1.0, "py": 0.0},
        ],
        "timestamps": [0.0, 0.1],
        "diagnostics": {
            "risk_trace": [0.1, 0.2],
            "bias_trace": [0.0, 0.0],
            "modalities": ["uwb", "vio"],
            "uwb_scaling_trace": [0.0, 0.0],
            "vio_scaling_trace": [0.0, 1.0],
        },
        "runtime_log": {"latency": [1.0, 2.0], "params": 0.0, "ram_peak": 0.0},
        "scenario_context": {
            "geometry_report": {
                "gdop_above_floor_ratio": 0.0,  # below floor
            }
        },
    }
    gt = {
        "seq_id": "seq_a",
        "states": [
            {"timestamp": 0.0, "px": 0.0, "py": 0.0},
            {"timestamp": 0.1, "px": 1.0, "py": 0.0},
        ],
    }
    with pytest.raises(ValueError, match=r"gdop_occupancy ratio_above_floor.*<.*gdop_occupancy_min_ratio"):
        compute_metrics(
            pred, gt,
            return_support=True,
            protocol_cfg=_protocol_cfg(0.99),
        )


def test_h25c_compute_metrics_skips_gate_when_protocol_cfg_none():
    """compute_metrics 不传 protocol_cfg 时跳过守门（向后兼容旧调用方）。"""
    pred = {
        "seq_id": "seq_a",
        "scene_id": "S(A0,N0,V0,K0,M0)",
        "states": [
            {"timestamp": 0.0, "px": 0.0, "py": 0.0},
            {"timestamp": 0.1, "px": 1.0, "py": 0.0},
        ],
        "timestamps": [0.0, 0.1],
        "diagnostics": {
            "risk_trace": [0.1, 0.2],
            "bias_trace": [0.0, 0.0],
            "modalities": ["uwb", "vio"],
            "uwb_scaling_trace": [0.0, 0.0],
            "vio_scaling_trace": [0.0, 1.0],
        },
        "runtime_log": {"latency": [1.0, 2.0], "params": 0.0, "ram_peak": 0.0},
        "scenario_context": {
            "geometry_report": {
                "gdop_above_floor_ratio": 0.0,
            }
        },
    }
    gt = {
        "seq_id": "seq_a",
        "states": [
            {"timestamp": 0.0, "px": 0.0, "py": 0.0},
            {"timestamp": 0.1, "px": 1.0, "py": 0.0},
        ],
    }
    # 不传 protocol_cfg 不应抛 ValueError（守门跳过）
    metric_table, support_report = compute_metrics(pred, gt, return_support=True)
    agg = support_report["gdop_occupancy_aggregation"]
    assert agg is not None
    assert agg["min_ratio_threshold"] is None
