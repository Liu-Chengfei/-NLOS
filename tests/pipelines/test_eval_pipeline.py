from __future__ import annotations

"""评估流水线（eval_pipeline）测试模块。

测试覆盖范围：
- 指标计算与持久化（RMSE/MAE/ATE/RPE 等）
- 最佳方法选择与协议优先级（非仅 RMSE）
- 重复实验 ID（repeat_id）在评估产物中的传递
- 场景级聚合 vs 原始重复计数
- 配对 Wilcoxon 检验
- smoke 模式（无真值运行）
- 直接真值覆盖与绘图输入合同
- 协议版本校验

被测模块：liquidloc.pipelines.eval_pipeline"""

import csv
import json
from pathlib import Path
from uuid import uuid4

import pytest

from liquidloc.common.config_utils import find_project_root
from liquidloc.pipelines.eval_pipeline import run
from liquidloc.protocol.metric_schema import get_metric_order

_PROJECT_ROOT = find_project_root()


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv'
A3_VIO_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'a3_vio'


def _tmp_output_root() -> Path:
    output_root = Path(__file__).resolve().parents[2] / 'outputs' / 'test_eval_pipeline_pytest' / uuid4().hex
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root


def _bundle(
    method_name: str,
    *,
    second_px: float,
    task_id: str | None = None,
    repeat_id: str | None = None,
) -> dict:
    bundle = {
        'seq_id': 'mini_seq',
        'scene_id': 'S(A2,N2,V2,K0,M0)',
        'method_name': method_name,
        'states': [{'px': 0.0, 'py': 0.0}, {'px': second_px, 'py': 0.0}],
        'timestamps': [0.0, 0.1],
        'diagnostics': {'risk_trace': [0.0, 0.0], 'bias_trace': [0.0, 0.0], 'scaling_trace': [0.0, second_px]},
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
    if task_id is not None:
        bundle['task_id'] = task_id
    if repeat_id is not None:
        bundle['repeat_id'] = repeat_id
    return bundle


def test_normal_case():
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    output_root = _tmp_output_root()
    result = run({
        'prediction_bundles': [
            _bundle('ekf', second_px=0.03, task_id='scene_00'),
            _bundle('robust_ekf', second_px=0.30, task_id='scene_00'),
        ],
        'ground_truth_root': FIXTURE_ROOT,
        'output_root': output_root / 'eval',
    })
    assert result.stage_name == 'eval_pipeline'
    assert (output_root / 'eval' / 'metrics' / 'metric_table.csv').is_file()
    assert result.metadata['audit_report']['best_method'] == 'ekf'
    assert result.metadata['audit_report']['best_method_by_priority'] == 'ekf'
    assert result.metadata['audit_report']['protocol_gate']['aggregation_order'][0] == 'single_run'
    with (output_root / 'eval' / 'metrics' / 'metric_table.csv').open('r', encoding='utf-8', newline='') as handle:
        metric_rows = list(csv.DictReader(handle))
    support_columns = {
        'prediction_length',
        'ground_truth_length',
        'aligned_length',
        'valid_pair_count',
        'overlap_ratio',
        'reliability_status',
    }
    assert support_columns.issubset(metric_rows[0].keys())
    rows_by_case: dict[str, list[str]] = {}
    for row in metric_rows:
        rows_by_case.setdefault(row['case_ref'], []).append(row['metric'])
    assert set(rows_by_case) == {'scene_00::ekf::S(A2,N2,V2,K0,M0)', 'scene_00::robust_ekf::S(A2,N2,V2,K0,M0)'}
    assert all(metric_names == get_metric_order() for metric_names in rows_by_case.values())
    assert {row['metric'] for row in metric_rows if row['group'] == 'mechanism'} >= {'corr_scaling_error'}
    selected_case_refs = {
        case_obj['case_ref']
        for case_group in result.metadata['selected_cases'].values()
        for case_obj in case_group
    }
    selected_methods = {
        case_obj['method_name']
        for case_group in result.metadata['selected_cases'].values()
        for case_obj in case_group
    }
    assert selected_case_refs == {'scene_00::ekf::S(A2,N2,V2,K0,M0)', 'scene_00::robust_ekf::S(A2,N2,V2,K0,M0)'}
    assert selected_methods == {'ekf', 'robust_ekf'}
    runtime_table_path = output_root / 'eval' / 'plotting_inputs' / 'runtime_table.json'
    assert runtime_table_path.is_file()
    runtime_rows = json.loads(runtime_table_path.read_text(encoding='utf-8'))
    metric_runtime_by_case: dict[str, dict[str, float]] = {}
    for row in metric_rows:
        if row['group'] != 'runtime':
            continue
        metric_runtime_by_case.setdefault(row['case_ref'], {})[row['metric']] = float(row['value'])
    assert {row['case_ref'] for row in runtime_rows} == set(metric_runtime_by_case)
    for runtime_row in runtime_rows:
        case_metrics = metric_runtime_by_case[runtime_row['case_ref']]
        for metric_name in ('latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak'):
            assert float(runtime_row[metric_name]) == pytest.approx(case_metrics[metric_name])
    assert result.metadata['plotting_inputs']['runtime_table'] == runtime_rows
    assert 'calibration_report' not in result.metadata['plotting_inputs']
    assert not (output_root / 'eval' / 'plotting_inputs' / 'calibration_report.json').exists()


def test_selected_cases_tie_break_is_method_neutral():
    """平局打破测试：selected cases。\n\n验证 selected cases 的平局打破逻辑，\n确保方法顺序不影响结果。
    """
    output_root = _tmp_output_root()
    forward = run({
        'prediction_bundles': [
            _bundle('ekf', second_px=0.03, task_id='scene_00'),
            _bundle('robust_ekf', second_px=0.03, task_id='scene_00'),
        ],
        'ground_truth_root': FIXTURE_ROOT,
        'output_root': output_root / 'eval_forward',
    })
    reverse = run({
        'prediction_bundles': [
            _bundle('robust_ekf', second_px=0.03, task_id='scene_00'),
            _bundle('ekf', second_px=0.03, task_id='scene_00'),
        ],
        'ground_truth_root': FIXTURE_ROOT,
        'output_root': output_root / 'eval_reverse',
    })

    assert forward.metadata['selected_cases'] == reverse.metadata['selected_cases']
    assert [case_obj['case_ref'] for case_obj in forward.metadata['selected_cases']['main_cases']] == [
        'scene_00::ekf::S(A2,N2,V2,K0,M0)',
        'scene_00::robust_ekf::S(A2,N2,V2,K0,M0)',
    ]
    assert [case_obj['case_ref'] for case_obj in forward.metadata['selected_cases']['failure_cases']] == [
        'scene_00::ekf::S(A2,N2,V2,K0,M0)',
        'scene_00::robust_ekf::S(A2,N2,V2,K0,M0)',
    ]
    assert [case_obj['case_ref'] for case_obj in forward.metadata['selected_cases']['boundary_cases']] == [
        'scene_00::ekf::S(A2,N2,V2,K0,M0)',
        'scene_00::robust_ekf::S(A2,N2,V2,K0,M0)',
    ]


def test_eval_metric_rows_prefer_applied_risk_and_bias_traces():
    """指标测试：eval。\n\n验证 eval 的指标计算，\n确保指标值和分组正确。
    """
    output_root = _tmp_output_root()
    bundle = _bundle('liquid_ekf', second_px=2.0, task_id='scene_00')
    bundle['diagnostics'].update(
        {
            'risk_trace': [1.0, 0.0],
            'bias_trace': [1.0, 0.0],
            'applied_risk_trace': [0.0, 1.0],
            'applied_bias_trace': [0.0, 1.0],
            'scaling_trace': [0.0, 0.0],
        }
    )

    result = run({
        'prediction_bundles': [bundle],
        'ground_truth_root': FIXTURE_ROOT,
        'output_root': output_root / 'eval',
    })

    metric_rows = result.metadata['metric_rows']
    metric_by_name = {row['metric']: float(row['value']) for row in metric_rows}
    assert metric_by_name['risk_error_corr'] == pytest.approx(1.0)
    assert metric_by_name['bias_alignment'] == pytest.approx(1.0)

    with (output_root / 'eval' / 'metrics' / 'metric_table.csv').open('r', encoding='utf-8', newline='') as handle:
        persisted_rows = list(csv.DictReader(handle))
    persisted_metric_by_name = {row['metric']: float(row['value']) for row in persisted_rows}
    assert persisted_metric_by_name['risk_error_corr'] == pytest.approx(1.0)
    assert persisted_metric_by_name['bias_alignment'] == pytest.approx(1.0)


def test_repeat_id_flows_through_eval_artifacts():
    """传递测试：repeat id。\n\n验证 repeat id 在流水线各阶段间正确传递，\n确保端到端数据流完整性。
    """
    output_root = _tmp_output_root()
    result = run({
        'prediction_bundles': [
            _bundle('ekf', second_px=0.03, task_id='scene_00', repeat_id='repeat_00'),
            _bundle('ekf', second_px=0.05, task_id='scene_00', repeat_id='repeat_01'),
        ],
        'ground_truth_root': FIXTURE_ROOT,
        'output_root': output_root / 'eval',
    })

    with (output_root / 'eval' / 'metrics' / 'metric_table.csv').open('r', encoding='utf-8', newline='') as handle:
        metric_rows = list(csv.DictReader(handle))
    case_refs = {row['case_ref'] for row in metric_rows}
    assert case_refs == {'scene_00::ekf::S(A2,N2,V2,K0,M0)::repeat_00', 'scene_00::ekf::S(A2,N2,V2,K0,M0)::repeat_01'}
    assert {row['repeat_id'] for row in metric_rows} == {'repeat_00', 'repeat_01'}

    runtime_rows = result.metadata['plotting_inputs']['runtime_table']
    assert {row['repeat_id'] for row in runtime_rows} == {'repeat_00', 'repeat_01'}

    sweep_rows = result.metadata['plotting_inputs']['sweep_table']
    assert {row['repeat_id'] for row in sweep_rows} == {'repeat_00', 'repeat_01'}

    trajectory_bundle = result.metadata['plotting_inputs']['trajectory_bundle']
    assert {row['repeat_id'] for row in trajectory_bundle['prediction_bundle']} == {'repeat_00', 'repeat_01'}
    assert {row['repeat_id'] for row in trajectory_bundle['gt_bundle']} == {'repeat_00', 'repeat_01'}
    assert 'prediction_bundle' in trajectory_bundle
    assert 'gt_bundle' in trajectory_bundle

    gt_bundle = result.metadata['plotting_inputs']['gt_bundle']
    assert {row['repeat_id'] for row in gt_bundle} == {'repeat_00', 'repeat_01'}

    selected_case_refs = {
        case_obj['case_ref']
        for case_group in result.metadata['selected_cases'].values()
        for case_obj in case_group
    }
    assert selected_case_refs <= case_refs
    assert all(
        case_obj['repeat_id'] in {'repeat_00', 'repeat_01'}
        for case_group in result.metadata['selected_cases'].values()
        for case_obj in case_group
    )


def test_boundary_case():
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    output_root = _tmp_output_root()
    bundle = _bundle('ekf', second_px=0.03, task_id='scene_00')
    bundle['scenario_context']['scenario_reports']['V']['consistency_checks']['drift_scale'] = False
    with pytest.raises(ValueError):
        run({
            'prediction_bundles': [bundle],
            'ground_truth_root': FIXTURE_ROOT,
            'output_root': output_root / 'eval',
        })


def test_normalizes_explicit_axes_for_sweep_table():
    """显式测试：normalizes。\n\n验证 normalizes 的显式参数优先级，\n确保显式指定覆盖隐式推断。
    """
    output_root = _tmp_output_root()
    bundle = _bundle('ekf', second_px=0.03, task_id='scene_00')
    bundle['axes'] = {'A': 2, 'N': '2'}
    result = run({
        'prediction_bundles': [bundle],
        'ground_truth_root': FIXTURE_ROOT,
        'output_root': output_root / 'eval',
    })

    sweep_rows = result.metadata['plotting_inputs']['sweep_table']
    assert len(sweep_rows) == 1
    assert sweep_rows[0]['A'] == 'A2'
    assert sweep_rows[0]['N'] == 'N2'
    assert sweep_rows[0]['V'] == 'V2'
    # 五轴档位协议：G 轴已并入 K 轴；scene_id 'S(A2,N2,V2,K0,M0)' 解析 K=K0。
    assert sweep_rows[0]['K'] == 'K0'


def test_plotting_inputs_expose_trajectory_bundle_contract():
    """合同测试：plotting inputs expose trajectory bundle。\n\n验证 plotting inputs expose trajectory bundle 的接口合同，\n确保输入输出符合协议约定。
    """
    output_root = _tmp_output_root()
    result = run({
        'prediction_bundles': [_bundle('ekf', second_px=0.03, task_id='scene_00', repeat_id='repeat_00')],
        'ground_truth_root': FIXTURE_ROOT,
        'output_root': output_root / 'eval',
    })

    trajectory_bundle = result.metadata['plotting_inputs']['trajectory_bundle']
    assert 'prediction_bundle' in trajectory_bundle
    assert 'gt_bundle' in trajectory_bundle
    assert trajectory_bundle['prediction_bundle'][0]['repeat_id'] == 'repeat_00'
    assert trajectory_bundle['gt_bundle'][0]['repeat_id'] == 'repeat_00'
    assert trajectory_bundle['prediction_bundle'][0]['case_ref'] == 'scene_00::ekf::S(A2,N2,V2,K0,M0)::repeat_00'
    assert trajectory_bundle['gt_bundle'][0]['case_ref'] == 'scene_00::ekf::S(A2,N2,V2,K0,M0)::repeat_00'


def test_smoke_mode_runs_without_ground_truth_root():
    """无依赖测试：smoke mode runs。\n\n验证 smoke mode runs 在缺少依赖时的降级行为，\n确保回退策略正确。
    """
    output_root = _tmp_output_root()
    result = run({
        'prediction_bundles': [_bundle('ekf', second_px=0.03, task_id='scene_00')],
        'smoke_mode': True,
        'output_root': output_root / 'eval',
    })

    assert result.stage_name == 'eval_pipeline'
    assert (output_root / 'eval' / 'metrics' / 'metric_table.csv').is_file()
    assert (output_root / 'eval' / 'statistics' / 'statistics_table.json').is_file()
    assert (output_root / 'eval' / 'cases' / 'selected_cases.json').is_file()
    assert result.metadata['audit_report']['best_method'] is None
    assert result.metadata['audit_report']['best_method_by_priority'] is None
    assert result.metadata['audit_report']['comparison_eligible'] is False
    assert result.metadata['audit_report']['comparison_status'] == 'smoke_only_self_ground_truth'
    assert result.metadata['statistics_table']['method_summary'] == {}
    assert result.metadata['statistics_table']['pairwise_tests'] == []
    assert result.metadata['statistics_table']['pairing_keys'] == []
    assert result.metadata['statistics_table']['comparison_eligible'] is False
    assert result.metadata['statistics_table']['comparison_status'] == 'smoke_only_self_ground_truth'
    assert result.metadata['selected_cases'] == {
        'main_cases': [],
        'failure_cases': [],
        'boundary_cases': [],
    }


def test_eval_rejects_nonexistent_ground_truth_root_before_metric_computation():
    """拒绝测试：eval。\n\n验证被测功能对 eval 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    output_root = _tmp_output_root()
    with pytest.raises(ValueError, match='ground_truth_root must point to an existing directory'):
        run({
            'prediction_bundles': [_bundle('ekf', second_px=0.03, task_id='scene_00')],
            'ground_truth_root': 'C:/definitely/not/exist',
            'output_root': output_root / 'eval',
        })


def test_eval_pipeline_uses_gate_normalized_ground_truth_root():
    """使用测试：eval pipeline。\n\n验证被测功能正确使用 eval pipeline，\n确保内部依赖被正确调用。
    """
    output_root = _tmp_output_root()
    result = run({
        'prediction_bundles': [_bundle('ekf', second_px=0.03, task_id='scene_00')],
        'ground_truth_root': ' tests/fixtures/datasets/miluv ',
        'output_root': output_root / 'eval',
    })
    assert result.metadata['audit_report']['protocol_gate']['ground_truth_root'] == 'tests/fixtures/datasets/miluv'


def test_eval_rejects_mismatched_experiment_protocol_version(tmp_path):
    """拒绝测试：eval。\n\n验证被测功能对 eval 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    tmp_dir = _PROJECT_ROOT / "outputs" / "_test_tmp" / "eval_pipeline"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = tmp_dir / 'bad_experiment_protocol.yaml'
    protocol_path.write_text(
        '\n'.join([
            'protocol_version: 999',
            'quick_full_rule: quick_smoke_scale__full_real_execution_required',
            'failure_sample_policy: retain_and_audit',
            'aggregation_order: [single_run, repeat_summary, scene_summary, experiment_conclusion]',
            'conclusion_priority: [rmse, p95, failure_rate, mae]',
            'public_benchmark:',
            '  frozen_eval_split: frozen_public_eval',
            '  tuning_forbidden: true',
            '  shared_split_required: true',
            'training:',
            '  allowed_split_roles: [train, val]',
            '  forbidden_split_roles: [test, external, frozen_public_eval]',
            'evaluation:',
            '  require_prediction_bundles: true',
            '  default_failure_threshold_m: 1.0',
            '  require_ground_truth_unless_smoke: true',
        ]),
        encoding='utf-8',
    )
    output_root = _tmp_output_root()
    with pytest.raises(ValueError, match='experiment protocol version must be 2'):
        run({
            'prediction_bundles': [_bundle('ekf', second_px=0.03, task_id='scene_00')],
            'ground_truth_root': FIXTURE_ROOT,
            'experiment_protocol_path': protocol_path,
            'output_root': output_root / 'eval',
        })


def test_a3_vio_exact_timestamp_overlap_has_support():
    """重叠测试：a3 vio exact timestamp。\n\n验证 a3 vio exact timestamp 的时间戳重叠处理，\n确保精确时间戳匹配正确。
    """
    output_root = _tmp_output_root()
    bundle = _bundle('ekf', second_px=0.01, task_id='scene_00')
    bundle['scene_id'] = 'S(A3,N3,V2,K0,M0)'
    bundle['axes'] = {'A': 'A3', 'N': 'N3', 'V': 'V2', 'G': 'K0', 'K': 'K0'}
    bundle['states'] = [
        {'px': 0.0, 'py': 0.0, 'vx': 0.0, 'vy': 0.0, 'yaw': 0.0, 'bax': 0.0, 'bay': 0.0, 'bg': 0.0},
        {'px': 0.049999999999999996, 'py': 0.0, 'vx': 0.0, 'vy': 0.0, 'yaw': 0.0, 'bax': 0.0, 'bay': 0.0, 'bg': 0.0},
        {'px': 0.07163461538461538, 'py': 0.0, 'vx': 0.0, 'vy': 0.0, 'yaw': 0.0, 'bax': 0.0, 'bay': 0.0, 'bg': 0.0},
        {'px': 0.072, 'py': 0.0005, 'vx': 0.0, 'vy': 0.0, 'yaw': 0.0, 'bax': 0.0, 'bay': 0.0, 'bg': 0.0},
        {'px': 0.08, 'py': 0.001, 'vx': 0.0, 'vy': 0.0, 'yaw': 0.0, 'bax': 0.0, 'bay': 0.0, 'bg': 0.0},
        {'px': 0.083, 'py': 0.0015, 'vx': 0.0, 'vy': 0.0, 'yaw': 0.0, 'bax': 0.0, 'bay': 0.0, 'bg': 0.0},
        {'px': 0.09, 'py': 0.002, 'vx': 0.0, 'vy': 0.0, 'yaw': 0.0, 'bax': 0.0, 'bay': 0.0, 'bg': 0.0},
        {'px': 0.1, 'py': 0.0025, 'vx': 0.0, 'vy': 0.0, 'yaw': 0.0, 'bax': 0.0, 'bay': 0.0, 'bg': 0.0},
        {'px': 0.106, 'py': 0.003, 'vx': 0.0, 'vy': 0.0, 'yaw': 0.0, 'bax': 0.0, 'bay': 0.0, 'bg': 0.0},
        {'px': 0.112, 'py': 0.0035, 'vx': 0.0, 'vy': 0.0, 'yaw': 0.0, 'bax': 0.0, 'bay': 0.0, 'bg': 0.0},
    ]
    bundle['timestamps'] = [0.1, 0.183, 0.207, 0.24900000000000003, 0.29300000000000004, 0.313, 0.36, 0.395, 0.43100000000000005, 0.46799999999999997]
    bundle['diagnostics'] = {
        'risk_trace': [0.0] * 10,
        'bias_trace': [0.0] * 10,
        'scaling_trace': [1.0] * 10,
        'mechanism_contract': {
            'bridge_enabled': False,
            'liquid_controls_measurement_update': True,
            'controls': ['bias_applied', 'risk', 'scaling', 'noise_multiplier', 'gate_action'],
        },
        'uwb_scaling_trace': [1.0] * 10,
        'vio_scaling_trace': [1.0] * 10,
        'noise_multiplier_trace': [1.0] * 10,
        'gate_action_trace': [
            'pass_through',
            'vio_confidence_scale',
            'vio_confidence_scale',
            'vio_confidence_scale',
            'uwb_bias_and_noise_scale',
            'uwb_bias_and_noise_scale',
            'uwb_bias_and_noise_scale',
            'uwb_bias_and_noise_scale',
            'uwb_bias_and_noise_scale',
            'uwb_bias_and_noise_scale',
        ],
        'modalities': ['imu', 'vio', 'vio', 'vio', 'uwb', 'uwb', 'uwb', 'uwb', 'uwb', 'uwb'],
    }
    bundle['runtime_log'] = {
        'latency': [
            0.032300129532814026,
            3.071499988436699,
            0.14379993081092834,
            0.2511999011039734,
            0.4088000860065222,
            0.17760001122951508,
            0.2144000083208084,
            0.19360005855560303,
            0.16400004923343658,
            0.1510000228881836,
        ],
        'params': 19.0,
        'ram_peak': 1.0,
    }

    result = run({
        'prediction_bundles': [bundle],
        'ground_truth_root': A3_VIO_FIXTURE_ROOT,
        'output_root': output_root / 'eval',
    })

    support_row = next(row for row in result.metadata['metric_rows'] if row['metric'] == 'rmse')
    assert support_row['aligned_length'] == 10
    assert support_row['valid_pair_count'] == 9
    assert support_row['reliability_status'] == 'ok'


def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    output_root = _tmp_output_root()
    with pytest.raises(ValueError):
        run({'prediction_bundles': [], 'output_root': output_root / 'eval'})


def test_direct_ground_truth_overrides_root_and_flows_to_plotting_inputs():
    """传递测试：direct ground truth overrides root and。\n\n验证 direct ground truth overrides root and 在流水线各阶段间正确传递，\n确保端到端数据流完整性。
    """
    output_root = _tmp_output_root()
    direct_gt_rows = [
        {'timestamp': 0.0, 'px': 0.0, 'py': 0.0},
        {'timestamp': 0.1, 'px': 0.4, 'py': 0.0},
    ]
    bundle = _bundle('ekf', second_px=0.4, task_id='scene_00')

    result = run({
        'prediction_bundles': [bundle],
        'ground_truth_root': FIXTURE_ROOT,
        'ground_truth_by_task_id': {'scene_00': direct_gt_rows},
        'output_root': output_root / 'eval',
    })

    rmse_row = next(row for row in result.metadata['metric_rows'] if row['metric'] == 'rmse')
    assert rmse_row['value'] == pytest.approx(0.0)
    assert rmse_row['aligned_length'] == 2
    assert result.metadata['plotting_inputs']['gt_bundle'][0]['states'] == direct_gt_rows


def test_eval_accepts_duplicate_prediction_timestamps_for_async_event_level_states():
    """接受测试：eval。\n\n验证 eval 的接受行为，\n确保合法输入被正确处理。
    """
    output_root = _tmp_output_root()
    bundle = _bundle('ekf', second_px=0.4, task_id='scene_00')
    bundle['states'] = [
        {'px': 0.0, 'py': 0.0},
        {'px': 0.4, 'py': 0.0},
    ]
    bundle['timestamps'] = [0.1, 0.1]
    direct_gt_rows = [
        {'timestamp': 0.1, 'px': 0.0, 'py': 0.0},
        {'timestamp': 0.2, 'px': 0.0, 'py': 0.0},
    ]

    result = run({
        'prediction_bundles': [bundle],
        'ground_truth_by_task_id': {'scene_00': direct_gt_rows},
        'ground_truth_root': FIXTURE_ROOT,
        'output_root': output_root / 'eval',
    })

    rmse_row = next(row for row in result.metadata['metric_rows'] if row['metric'] == 'rmse')
    assert rmse_row['aligned_length'] == 2
    assert rmse_row['valid_pair_count'] == 2
    assert rmse_row['reliability_status'] == 'ok'


def test_relative_output_root_is_project_root_anchored(tmp_path):
    """项目根锚定测试：relative output root。\n\n验证 relative output root 的相对路径被正确锚定到项目根目录，\n而非当前工作目录。
    """
    project_root = tmp_path / 'repo_root'
    result = run({
        'prediction_bundles': [_bundle('ekf', second_px=0.03, task_id='scene_00')],
        'ground_truth_root': FIXTURE_ROOT,
        'project_root': project_root,
        'output_root': Path('outputs') / 'eval_relative',
    })

    assert (project_root / 'outputs' / 'eval_relative' / 'metrics' / 'metric_table.csv').is_file()
    assert result.stage_name == 'eval_pipeline'


def test_eval_best_method_respects_protocol_priority_not_rmse_only():
    """尊重测试：eval best method。\n\n验证被测功能尊重 eval best method 的规则，\n确保协议约束被正确执行。
    """
    output_root = _tmp_output_root()
    result = run({
        'prediction_bundles': [
            _bundle('ekf', second_px=0.80, task_id='scene_00'),
            _bundle('liquid_ekf', second_px=0.90, task_id='scene_00'),
        ],
        'ground_truth_root': FIXTURE_ROOT,
        'output_root': output_root / 'eval',
        'failure_threshold': 0.85,
    })
    assert result.metadata['audit_report']['conclusion_priority'] == ['rmse', 'p95', 'failure_rate', 'mae']
    assert result.metadata['audit_report']['best_method'] == 'ekf'
    assert {
        case_obj['method_name']
        for case_group in result.metadata['selected_cases'].values()
        for case_obj in case_group
    } == {'ekf', 'liquid_ekf'}


def test_eval_pipeline_uses_paired_wilcoxon_when_task_ids_align():
    """使用测试：eval pipeline。\n\n验证被测功能正确使用 eval pipeline，\n确保内部依赖被正确调用。
    """
    output_root = _tmp_output_root()
    run({
        'prediction_bundles': [
            _bundle('ekf', second_px=0.03, task_id='scene_00'),
            _bundle('robust_ekf', second_px=0.04, task_id='scene_00'),
            _bundle('ekf', second_px=0.02, task_id='scene_01'),
            _bundle('robust_ekf', second_px=0.01, task_id='scene_01'),
        ],
        'ground_truth_root': FIXTURE_ROOT,
        'output_root': output_root / 'eval',
    })

    statistics_path = output_root / 'eval' / 'statistics' / 'statistics_table.json'
    statistics_payload = json.loads(statistics_path.read_text(encoding='utf-8'))
    # scene_summary 合并了不同 task_id 的行后 task_id 不唯一，配对键回退到 scene_id + seq_id。
    assert statistics_payload['pairing_keys'] == ['scene_id', 'seq_id']
    assert statistics_payload['pairwise_tests']
    assert all(row['test_name'] == 'paired_wilcoxon' for row in statistics_payload['pairwise_tests'])
    assert all(row['pairing_fields'] == ['scene_id', 'seq_id'] for row in statistics_payload['pairwise_tests'])


def test_eval_best_method_uses_scene_level_aggregation_not_raw_repeat_count():
    """使用测试：eval best method。\n\n验证被测功能正确使用 eval best method，\n确保内部依赖被正确调用。
    """
    output_root = _tmp_output_root()
    result = run({
        'prediction_bundles': [
            _bundle('ekf', second_px=0.0, task_id='scene_00', repeat_id='repeat_00'),
            _bundle('ekf', second_px=0.0, task_id='scene_00', repeat_id='repeat_01'),
            _bundle('ekf', second_px=0.0, task_id='scene_00', repeat_id='repeat_02'),
            _bundle('ekf', second_px=2 ** 0.5, task_id='scene_01', repeat_id='repeat_00'),
            _bundle('liquid_ekf', second_px=0.4 * (2 ** 0.5), task_id='scene_00', repeat_id='repeat_00'),
            _bundle('liquid_ekf', second_px=0.4 * (2 ** 0.5), task_id='scene_01', repeat_id='repeat_00'),
        ],
        'ground_truth_root': FIXTURE_ROOT,
        'output_root': output_root / 'eval',
        'failure_threshold': 0.5,
    })

    method_summary = result.metadata['statistics_table']['method_summary']

    assert method_summary['ekf']['num_bundles'] == 4
    # 所有 ekf bundles 共享相同的 scene_id 和 seq_id，所以被合并成 1 个 repeat_summary。
    # task_id 是每次执行的唯一标识，不作为 repeat 分组键。
    assert method_summary['ekf']['num_repeat_summaries'] == 1
    assert method_summary['ekf']['num_scene_summaries'] == 1
    assert method_summary['liquid_ekf']['num_bundles'] == 2
    # 两个 liquid_ekf bundles 共享相同的 scene_id、seq_id 和 repeat_id，被合并成 1 个 repeat_summary。
    assert method_summary['liquid_ekf']['num_repeat_summaries'] == 1
    assert method_summary['liquid_ekf']['num_scene_summaries'] == 1
    # 验证 method_summary 包含 mean_rmse 字段且为正数。
    assert method_summary['ekf']['mean_rmse'] > 0.0
    assert method_summary['liquid_ekf']['mean_rmse'] > 0.0


def test_eval_rejects_string_smoke_mode_before_metric_computation():
    """冒烟测试：eval rejects string。\n\n快速验证 eval rejects string 的基本功能可用，\n不深入检查细节，仅确认流程不崩溃。
    """
    output_root = _tmp_output_root()
    with pytest.raises(TypeError, match='smoke_mode must be a boolean'):
        run({
            'prediction_bundles': [_bundle('ekf', second_px=0.03, task_id='scene_00')],
            'smoke_mode': 'true',
            'output_root': output_root / 'eval',
        })
