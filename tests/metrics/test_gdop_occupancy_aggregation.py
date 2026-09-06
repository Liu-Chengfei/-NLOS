from __future__ import annotations

"""H24e 真改 §4.2 L894「差 GDOP 占比聚合」测试模块。

文件职责：验证 _aggregate_gdop_occupancy 与 _collect_sequence_data / _compute_*_sequence_metrics
的 H24e 真改：
- _collect_sequence_data 从 prediction_obj.scenario_context.geometry_report 提取
  gdop_above_floor_ratio 写入 support_context['gdop_occupancy_above_floor']
- _aggregate_gdop_occupancy 跨序列聚合：mean/min/max/ratio_above_floor/measured_ratio
- 所有序列均无 geometry_report 时返回 None（视作"未测量"而非 0.0）
- _compute_single_sequence_metrics 与 _compute_multi_sequence_metrics 的 support_report
  均写回 gdop_occupancy_aggregation 字段（口径一致）

被测模块：liquidloc.metrics.metric_runner._aggregate_gdop_occupancy
        liquidloc.metrics.metric_runner._collect_sequence_data
        liquidloc.metrics.metric_runner._compute_single_sequence_metrics
        liquidloc.metrics.metric_runner._compute_multi_sequence_metrics
"""

import math
from typing import Any, Mapping

import pytest

from liquidloc.metrics.metric_runner import (
    _aggregate_gdop_occupancy,
    _collect_sequence_data,
    _compute_single_sequence_metrics,
    _compute_multi_sequence_metrics,
)


def _make_prediction_obj(
    *,
    seq_id: str = "seq_a",
    scene_id: str = "S(A0,N0,V0,K0,M0)",
    gdop_above_floor_ratio: float | None = None,
) -> dict[str, Any]:
    """构造含 scenario_context.geometry_report.gdop_above_floor_ratio 的 prediction_obj。

    None 表示该序列无 geometry_report，对应场景：旧 prediction bundle 未携带此字段。
    轨迹点字段口径：timestamp/px/py（与 test_metric_runner.py:38-52 _prediction_bundle 同口径）；
    diagnostics.risk_trace / bias_trace / modalities / uwb_scaling_trace / vio_scaling_trace
    用于 _extract_preferred_aligned_trace 严格校验。
    """
    pred: dict[str, Any] = {
        "seq_id": seq_id,
        "scene_id": scene_id,
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
    }
    if gdop_above_floor_ratio is not None:
        pred["scenario_context"] = {
            "geometry_report": {
                "gdop_above_floor_ratio": gdop_above_floor_ratio,
            }
        }
    return pred


def _make_gt_obj(seq_id: str = "seq_a") -> dict[str, Any]:
    return {
        "seq_id": seq_id,
        "states": [
            {"timestamp": 0.0, "px": 0.0, "py": 0.0},
            {"timestamp": 0.1, "px": 1.0, "py": 0.0},
        ],
    }


def test_aggregate_gdop_occupancy_empty_returns_none():
    """空序列组必须返回 None（视作"未测量"而非 0.0）。"""
    assert _aggregate_gdop_occupancy([]) is None


def test_aggregate_gdop_occupancy_all_none_returns_none():
    """所有序列无 gdop_occupancy_above_floor 时返回 None（向后兼容旧 bundle）。"""
    sequence_data_group = [
        {"support_context": {"gdop_occupancy_above_floor": None}},
        {"support_context": {"gdop_occupancy_above_floor": None}},
    ]
    assert _aggregate_gdop_occupancy(sequence_data_group) is None


def test_aggregate_gdop_occupancy_mixed_measured_and_none():
    """部分序列有 gdop_above_floor_ratio 时只聚合非 None 序列。"""
    sequence_data_group = [
        {"support_context": {"gdop_occupancy_above_floor": 1.0}},  # above floor
        {"support_context": {"gdop_occupancy_above_floor": 0.0}},  # at/below floor
        {"support_context": {"gdop_occupancy_above_floor": None}},  # 无 geometry_report
    ]
    result = _aggregate_gdop_occupancy(sequence_data_group)
    assert result is not None
    assert result["count_total"] == 3
    assert result["count_measured"] == 2
    assert result["count_above_floor"] == 1
    assert result["ratio_above_floor"] == 0.5
    assert result["measured_ratio"] == pytest.approx(2 / 3)
    assert result["mean"] == 0.5
    assert result["min"] == 0.0
    assert result["max"] == 1.0
    assert result["per_sequence_values"] == [1.0, 0.0]


def test_aggregate_gdop_occupancy_all_above_floor():
    """所有序列 gdop_above_floor_ratio >= 1.0 时 ratio_above_floor=1.0。"""
    sequence_data_group = [
        {"support_context": {"gdop_occupancy_above_floor": 1.0}},
        {"support_context": {"gdop_occupancy_above_floor": 1.0}},
    ]
    result = _aggregate_gdop_occupancy(sequence_data_group)
    assert result["count_above_floor"] == 2
    assert result["ratio_above_floor"] == 1.0
    assert result["measured_ratio"] == 1.0


def test_aggregate_gdop_occupancy_handles_invalid_values():
    """非数值 / NaN / inf 的 gdop_occupancy_above_floor 必须被跳过不污染聚合。"""
    sequence_data_group = [
        {"support_context": {"gdop_occupancy_above_floor": 1.0}},
        {"support_context": {"gdop_occupancy_above_floor": "invalid"}},  # 字符串
        {"support_context": {"gdop_occupancy_above_floor": float("nan")}},  # NaN
        {"support_context": {"gdop_occupancy_above_floor": float("inf")}},  # Inf
        {"support_context": {}},  # 缺字段
        "not a mapping",  # 非 Mapping
    ]
    result = _aggregate_gdop_occupancy(sequence_data_group)
    assert result is not None
    assert result["count_measured"] == 1
    assert result["count_above_floor"] == 1
    assert result["per_sequence_values"] == [1.0]


def test_collect_sequence_data_extracts_gdop_above_floor_ratio():
    """_collect_sequence_data 从 prediction_obj.scenario_context.geometry_report 提取 gdop_above_floor_ratio。"""
    pred = _make_prediction_obj(gdop_above_floor_ratio=1.0)
    gt = _make_gt_obj()
    seq_data = _collect_sequence_data(pred, gt, bundle_index=0)
    assert seq_data["support_context"]["gdop_occupancy_above_floor"] == 1.0


def test_collect_sequence_data_returns_none_when_no_geometry_report():
    """prediction_obj 无 scenario_context.geometry_report 时 gdop_occupancy_above_floor 为 None。"""
    pred = _make_prediction_obj(gdop_above_floor_ratio=None)
    gt = _make_gt_obj()
    seq_data = _collect_sequence_data(pred, gt, bundle_index=0)
    assert seq_data["support_context"]["gdop_occupancy_above_floor"] is None


def test_compute_single_sequence_metrics_writes_gdop_occupancy_aggregation():
    """_compute_single_sequence_metrics 的 support_report 必须含 gdop_occupancy_aggregation 字段。"""
    pred = _make_prediction_obj(gdop_above_floor_ratio=1.0)
    gt = _make_gt_obj()
    seq_data = _collect_sequence_data(pred, gt, bundle_index=0)
    metric_table, support_report = _compute_single_sequence_metrics(
        seq_data,
        metric_order=("rmse", "p95", "failure_rate", "p99", "mae", "ate", "rpe",
                      "yaw_rmse", "yaw_p95", "risk_error_corr", "coverage",
                      "bias_alignment", "corr_scaling_error",
                      "latency_mean", "latency_p50", "latency_p95",
                      "params", "ram_peak"),
        failure_threshold=2.0,
    )
    assert "gdop_occupancy_aggregation" in support_report
    agg = support_report["gdop_occupancy_aggregation"]
    assert agg is not None
    assert agg["count_measured"] == 1
    assert agg["count_above_floor"] == 1
    assert agg["ratio_above_floor"] == 1.0


def test_compute_multi_sequence_metrics_writes_gdop_occupancy_aggregation():
    """_compute_multi_sequence_metrics 的 support_report 必须含跨序列聚合字段。"""
    pred1 = _make_prediction_obj(seq_id="seq_a", gdop_above_floor_ratio=1.0)
    pred2 = _make_prediction_obj(seq_id="seq_b", gdop_above_floor_ratio=0.0)
    pred3 = _make_prediction_obj(seq_id="seq_c", gdop_above_floor_ratio=None)
    gt1 = _make_gt_obj(seq_id="seq_a")
    gt2 = _make_gt_obj(seq_id="seq_b")
    gt3 = _make_gt_obj(seq_id="seq_c")
    seq_data_group = [
        _collect_sequence_data(pred1, gt1, bundle_index=0),
        _collect_sequence_data(pred2, gt2, bundle_index=1),
        _collect_sequence_data(pred3, gt3, bundle_index=2),
    ]
    metric_table, support_report = _compute_multi_sequence_metrics(
        seq_data_group,
        metric_order=("rmse", "p95", "failure_rate", "p99", "mae", "ate", "rpe",
                      "yaw_rmse", "yaw_p95", "risk_error_corr", "coverage",
                      "bias_alignment", "corr_scaling_error",
                      "latency_mean", "latency_p50", "latency_p95",
                      "params", "ram_peak"),
        failure_threshold=2.0,
    )
    agg = support_report["gdop_occupancy_aggregation"]
    assert agg is not None
    assert agg["count_total"] == 3
    assert agg["count_measured"] == 2
    assert agg["count_above_floor"] == 1
    assert agg["ratio_above_floor"] == 0.5
    assert agg["measured_ratio"] == pytest.approx(2 / 3)


def test_compute_multi_sequence_metrics_returns_none_when_no_geometry_reports():
    """所有序列均无 geometry_report 时 support_report['gdop_occupancy_aggregation'] 为 None。"""
    pred = _make_prediction_obj(gdop_above_floor_ratio=None)
    gt = _make_gt_obj()
    seq_data_group = [
        _collect_sequence_data(pred, gt, bundle_index=0),
        _collect_sequence_data(pred, gt, bundle_index=1),
    ]
    _, support_report = _compute_multi_sequence_metrics(
        seq_data_group,
        metric_order=("rmse", "p95", "failure_rate", "p99", "mae", "ate", "rpe",
                      "yaw_rmse", "yaw_p95", "risk_error_corr", "coverage",
                      "bias_alignment", "corr_scaling_error",
                      "latency_mean", "latency_p50", "latency_p95",
                      "params", "ram_peak"),
        failure_threshold=2.0,
    )
    assert support_report["gdop_occupancy_aggregation"] is None
