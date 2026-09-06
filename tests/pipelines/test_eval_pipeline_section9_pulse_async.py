"""§9.3 pulse/async 量级门审计透传端到端测试.

测试覆盖:
- eval_pipeline.run 在 audit_payload['section9_pulse_async_audit'] 中聚合
  compute_metrics 返回的 support_report['section9_pulse_async_violations']
- 落盘 audits_dir/section9_pulse_async_audit.json 独立审计文件, 供 12_run_statistics
  与人工审计直接读取, 不依赖 audit_payload 嵌套解析
- §9 pulse_async 透传到 method/seq/case 维度的计数容器
- 无违规的 bundle 也被统计到 bundle_count

本测试与 metric_runner._emit_section9_pulse_async_warnings 一起形成
§9 pulse/async 量级门从单 bundle → 全 run 聚合的端到端调用链覆盖率.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from liquidloc.pipelines import eval_pipeline as eval_pipeline_module
from liquidloc.pipelines.eval_pipeline import run
from liquidloc.protocol.metric_schema import get_metric_order

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv'


def _tmp_output_root() -> Path:
    output_root = (
        Path(__file__).resolve().parents[2]
        / 'outputs'
        / 'test_eval_pipeline_section9_pytest'
        / uuid4().hex
    )
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root


def _make_bundle(method_name: str, *, second_px: float) -> dict:
    """构造一个最小可用 prediction bundle (与 test_eval_pipeline.py 同口径)."""
    bundle = {
        'seq_id': 'mini_seq',
        'scene_id': 'S(A2,N2,V2,K0,M0)',
        'method_name': method_name,
        'states': [{'px': 0.0, 'py': 0.0}, {'px': second_px, 'py': 0.0}],
        'timestamps': [0.0, 0.1],
        'diagnostics': {
            'risk_trace': [0.0, 0.0],
            'bias_trace': [0.0, 0.0],
            'scaling_trace': [0.0, second_px],
        },
        'runtime_log': {'latency': [1.0, 2.0], 'params': 0.0, 'ram_peak': 0.0},
        'scenario_context': {
            'scenario_reports': {
                'V': {
                    'protocol_consistent': True,
                    'consistency_checks': {
                        'tracked_features_range': True,
                        'reproj_err_max': True,
                        'blackout': True,
                        'drift_scale': True,
                    },
                }
            }
        },
    }
    return bundle


def _fake_compute_metrics_with_section9_violations(*args, **kwargs):
    """patch compute_metrics, 返回带 §9 pulse_async 违规的 support_report.

    真实 compute_metrics 在 support_report['section9_pulse_async_violations'] /
    ['section9_pulse_async_aggregation'] 中按单 bundle 生成 §9 违规信息.
    本 fake 直接注入违规条目, 让 eval_pipeline 的聚合逻辑被验证.
    """
    return_support = kwargs.get('return_support', False)
    # 构造最小可用 metric_values (与 test_eval_pipeline.py 同口径)
    metric_values = {
        'rmse': 0.05,
        'p95': 0.05,
        'failure_rate': 0.0,
        'p99': 0.05,
        'mae': 0.05,
        'ate': 0.05,
        'rpe': 0.05,
        'yaw_rmse': 0.05,
        'yaw_p95': 0.05,
        'risk_error_corr': 0.0,
        'coverage': 1.0,
        'bias_alignment': 0.0,
        'corr_scaling_error': 0.0,
        'latency_mean': 1.0,
        'latency_p50': 1.0,
        'latency_p95': 2.0,
        'params': 0.0,
        'ram_peak': 0.0,
    }
    support_report = {
        'prediction_length': 2,
        'ground_truth_length': 2,
        'aligned_length': 2,
        'valid_pair_count': 2,
        'overlap_ratio': 1.0,
        'reliability_status': 'ok',
        'long_failure_segments': [],
        'ate_degraded': False,
        'measurement_mask': [True, True],
        'gdop_occupancy_aggregation': {},
        # §9 pulse/async 违规条目 — 模拟 metric_runner._emit_section9_pulse_async_warnings 输出
        'section9_pulse_async_violations': [
            {
                'seq_id': 'mini_seq',
                'n_pulse': 5,        # < n_pulse_min=30 → pulse_violated=True
                'n_pulse_min': 30,
                'n_async': 2,        # < n_async_min=20 → async_violated=True
                'n_async_min': 20,
                'pulse_violated': True,
                'async_violated': True,
                'cmp1_cmp5_at_risk': True,
                'message': '§9.3 单轨 n_pulse=5 < n_pulse_min=30, n_async=2 < n_async_min=20 (seq_id=mini_seq)',
            }
        ],
        'section9_pulse_async_aggregation': {
            'n_seq_iter': 1,
            'total_n_pulse': 5,
            'total_n_async': 2,
            'pulse_violation_count': 1,
            'async_violation_count': 1,
        },
    }
    if return_support:
        return metric_values, support_report
    return metric_values


def test_section9_pulse_async_audit_in_audit_payload(tmp_path):
    """audit_payload['section9_pulse_async_audit'] 必须聚合 §9 pulse_async 违规."""
    output_root = _tmp_output_root()

    with patch.object(
        eval_pipeline_module, 'compute_metrics',
        side_effect=_fake_compute_metrics_with_section9_violations,
    ):
        result = run({
            'prediction_bundles': [
                _make_bundle('ekf', second_px=0.03),
                _make_bundle('robust_ekf', second_px=0.30),
            ],
            'ground_truth_root': FIXTURE_ROOT,
            'output_root': output_root / 'eval',
        })

    audit = result.metadata['audit_report']
    # §9 pulse_async 审计聚合键必须存在
    assert 'section9_pulse_async_audit' in audit, (
        "audit_payload['section9_pulse_async_audit'] 缺失 — §9 pulse_async 透传修复未生效"
    )
    aggr = audit['section9_pulse_async_audit']
    # 2 个 bundle, 每 bundle 注入 1 条违规 → 共 2 条
    assert aggr['bundle_count'] == 2
    assert aggr['violations_total'] == 2
    assert aggr['pulse_violated_count'] == 2
    assert aggr['async_violated_count'] == 2
    assert aggr['cmp1_cmp5_at_risk_count'] == 2
    # method 维度: 每方法 1 条违规
    assert aggr['violations_by_method'].get('ekf') == 1
    assert aggr['violations_by_method'].get('robust_ekf') == 1
    # seq 维度: 但 mini_seq 跨 2 bundle → 2 条违规
    assert aggr['violations_by_seq'].get('mini_seq') == 2
    # violations_detail 含 2 条完整副本
    assert len(aggr['violations_detail']) == 2
    detail0 = aggr['violations_detail'][0]
    assert detail0['pulse_violated'] is True
    assert detail0['async_violated'] is True
    assert detail0['n_pulse'] == 5
    assert detail0['n_pulse_min'] == 30
    # aggregation_by_bundle 含 2 条
    assert len(aggr['aggregation_by_bundle']) == 2
    assert aggr['aggregation_by_bundle'][0]['aggregation']['total_n_pulse'] == 5


def test_section9_pulse_async_audit_json_written_to_disk(tmp_path):
    """audits_dir/section9_pulse_async_audit.json 必须独立落盘, 不依赖 audit_payload 嵌套解析."""
    output_root = _tmp_output_root()

    with patch.object(
        eval_pipeline_module, 'compute_metrics',
        side_effect=_fake_compute_metrics_with_section9_violations,
    ):
        result = run({
            'prediction_bundles': [_make_bundle('ekf', second_px=0.03)],
            'ground_truth_root': FIXTURE_ROOT,
            'output_root': output_root / 'eval',
        })

    audit_path = output_root / 'eval' / 'audits' / 'section9_pulse_async_audit.json'
    assert audit_path.is_file(), (
        f"section9_pulse_async_audit.json 未落盘: {audit_path}"
    )
    payload = json.loads(audit_path.read_text(encoding='utf-8'))
    assert payload['bundle_count'] == 1
    assert payload['violations_total'] == 1
    assert payload['pulse_violated_count'] == 1
    assert payload['async_violated_count'] == 1
    assert payload['cmp1_cmp5_at_risk_count'] == 1
    assert len(payload['violations_detail']) == 1
    assert len(payload['aggregation_by_bundle']) == 1


def test_section9_pulse_async_audit_in_artifacts(tmp_path):
    """§9 脉冲/异步审计 JSON 路径必须出现在 StageResult.artifacts 列表中."""
    output_root = _tmp_output_root()

    with patch.object(
        eval_pipeline_module, 'compute_metrics',
        side_effect=_fake_compute_metrics_with_section9_violations,
    ):
        result = run({
            'prediction_bundles': [_make_bundle('ekf', second_px=0.03)],
            'ground_truth_root': FIXTURE_ROOT,
            'output_root': output_root / 'eval',
        })

    audit_artifact = next(
        (a for a in result.artifacts if a.endswith('section9_pulse_async_audit.json')),
        None,
    )
    assert audit_artifact is not None, (
        "StageResult.artifacts 缺 section9_pulse_async_audit.json — 上层编排看不到该产物"
    )
    assert Path(audit_artifact).is_file()


def test_section9_pulse_async_no_violations_when_support_report_clean():
    """当 compute_metrics 返回的 support_report 不含 §9 pulse_async 字段时, 聚合为零, 不报错."""
    output_root = _tmp_output_root()

    def _fake_clean_compute_metrics(*args, **kwargs):
        return_support = kwargs.get('return_support', False)
        metric_values = {
            'rmse': 0.05, 'p95': 0.05, 'failure_rate': 0.0,
            'p99': 0.05, 'mae': 0.05, 'ate': 0.05, 'rpe': 0.05,
            'yaw_rmse': 0.05, 'yaw_p95': 0.05,
            'risk_error_corr': 0.0, 'coverage': 1.0,
            'bias_alignment': 0.0, 'corr_scaling_error': 0.0,
            'latency_mean': 1.0, 'latency_p50': 1.0, 'latency_p95': 2.0,
            'params': 0.0, 'ram_peak': 0.0,
        }
        support_report = {
            'prediction_length': 2, 'ground_truth_length': 2,
            'aligned_length': 2, 'valid_pair_count': 2,
            'overlap_ratio': 1.0, 'reliability_status': 'ok',
            'long_failure_segments': [], 'ate_degraded': False,
            'measurement_mask': [True, True],
            'gdop_occupancy_aggregation': {},
        }
        if return_support:
            return metric_values, support_report
        return metric_values

    with patch.object(
        eval_pipeline_module, 'compute_metrics',
        side_effect=_fake_clean_compute_metrics,
    ):
        result = run({
            'prediction_bundles': [_make_bundle('ekf', second_px=0.03)],
            'ground_truth_root': FIXTURE_ROOT,
            'output_root': output_root / 'eval',
        })

    aggr = result.metadata['audit_report']['section9_pulse_async_audit']
    assert aggr['bundle_count'] == 1
    # 无违规时, total/by_method/by_seq 均为空, 不抛 KeyError
    assert aggr['violations_total'] == 0
    assert aggr['violations_by_method'] == {}
    assert aggr['violations_by_seq'] == {}
    assert aggr['pulse_violated_count'] == 0
    assert aggr['cmp1_cmp5_at_risk_count'] == 0
    assert aggr['violations_detail'] == []
    assert aggr['aggregation_by_bundle'] == []
