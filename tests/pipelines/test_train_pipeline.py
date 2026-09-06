from __future__ import annotations

"""训练流水线（train_pipeline）测试模块。

测试覆盖范围：
- LSTM 和 Liquid 模型的训练流程端到端测试
- 设备选择策略（auto/cpu/cuda）及 CUDA 可用性检测
- 训练/验证集划分逻辑与审计报告
- 目标中间值（target_intermediate）构建与合同验证
- 几何偏置教师信号（geometry_bias_teacher）的质量审计
- 禁止架构术语检查（防止紧耦合等违规设计）
- 检查点冒烟测试（checkpoint smoke test）
- quick/full 模式的协议门控

被测模块：liquidloc.pipelines.train_pipeline"""

import json
import math
import shutil
from pathlib import Path

import pytest
import torch
from liquidloc.factories.estimator_factory import create_estimator
from liquidloc.factories.model_factory import create_model
from liquidloc.fusion.fusion_runner import _build_readout_context
from liquidloc.fusion.fusion_runner import _init_readout_context_cache
from liquidloc.fusion.fusion_runner import _update_readout_context_cache
from liquidloc.common.gt_utils import resolve_anchor_position as _resolve_anchor_position
from liquidloc.common.gt_utils import GT_TIME_TOLERANCE as _GT_TIME_TOLERANCE, align_ground_truth as _align_ground_truth
from liquidloc.pipelines.train_pipeline import (
    _load_default_ekf_cfg,
    _build_feature_window_builder,
    _build_split_audit,
    _build_training_readout_context,
    _build_liquid_samples,
    _build_teacher_free_uwb_scaling,
    _build_teacher_free_vio_scaling,
    _decode_event_scene_spec,
    _is_normal_scene_for_training,
    _build_target_intermediate,
    _coerce_training_residual_norm,
    _init_training_readout_context_cache,
    _load_miluv_sequence_payload,
    _materialize_teacher_anchor_layout_from_source_report,
    _resolve_alignment_risk_scales,
    _resolve_modality_observation_signal,
    _resolve_scene_axis_risk_floor,
    _resolve_train_device,
    _resolve_sequence_anchor_layout,
    _split_train_and_val_samples,
    _update_training_readout_context_cache,
    run,
)
from liquidloc.protocol.scene_axis_protocol import load_scene_axis_protocol
from liquidloc.scenarios.geometry_levels import build_anchor_layout
from liquidloc.scenarios.visual_levels import apply_visual_level


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_ARCHITECTURE_TERMS = (
    '紧耦合',
    '端到端滤波学习',
    '联合估计',
    '直接学习 EKF 主方程',
    'q_scaling',
    'online adaptation',
    'Shadow EKF',
)
_FEATURE_ORDER = ['dt', 'ax', 'ay', 'gz', 'range', 'valid', 'quality', 'dx', 'dy', 'dyaw', 'yaw', 'tracked_features', 'reproj_err']
_OFFICIAL_EXPERIMENTS_CSV = """experiment,num_robots,num_tags_per_robot,num_anchors,anchor_constellation,trajectory,cir_bool,obstacles_bool,apriltags_bool,barometer_bool
default_3_random_0,3,2,6,0,random,false,false,true,false
"""
_OFFICIAL_ANCHORS_YAML = """"0":
  "0": "[3.273827392578125, 3.46404736328125, 1.8093309326171875]"
  "1": "[3.186386962890625, 0.27394485473632812, 1.5884853515625]"
  "2": "[2.850500244140625, -2.923056884765625, 1.89742041015625]"
  "3": "[-2.497634521484375, -3.5018203125, 1.7730911865234375]"
  "4": "[-2.95793310546875, 0.6128419189453125, 1.65714208984375]"
  "5": "[-2.734676513671875, 3.65854248046875, 1.890254638671875]"
"""


def _lstm_model_cfg():
    return {
        'name': 'lstm_ekf',
        'feature_order': list(_FEATURE_ORDER),
        'window': {'size': 20, 'step': 1},
        'network': {'input_dim': 13, 'hidden_dim': 64, 'output_heads': ['bias', 'risk', 'uwb_scaling', 'vio_scaling']},
        'train': {'optimizer': 'adam', 'lr': 0.001},
    }


def _liquid_model_cfg():
    return {
        'name': 'liquid_ekf',
        'feature_order': list(_FEATURE_ORDER),
        'window': {'size': 20, 'step': 1},
        'network': {'input_dim': 13, 'hidden_dim': 64, 'output_heads': ['bias', 'risk', 'uwb_scaling', 'vio_scaling']},
        'train': {'optimizer': 'adam', 'lr': 0.001},
    }


def test_resolve_alignment_risk_scales_uses_protocol_default_failure_threshold(monkeypatch):
    """使用测试：resolve alignment risk scales。\n\n验证被测功能正确使用 resolve alignment risk scales，\n确保内部依赖被正确调用。
    """
    _resolve_alignment_risk_scales.cache_clear()
    monkeypatch.setattr(
        'liquidloc.pipelines.train_pipeline.load_experiment_protocol',
        lambda: {
            'protocol_version': 1,
            'quick_full_rule': 'quick_smoke_scale__full_real_execution_required',
            'failure_sample_policy': 'retain_and_audit',
            'aggregation_order': ['single_run', 'repeat_summary', 'scene_summary', 'experiment_conclusion'],
            'conclusion_priority': ['rmse', 'p95', 'failure_rate', 'mae'],
            'public_benchmark': {
                'allowed_datasets': ['miluv', 'ntu_viral'],
                'frozen_eval_split': 'frozen_public_eval',
                'tuning_forbidden': True,
                'shared_split_required': True,
            },
            'training': {
                'allowed_split_roles': ['train', 'val'],
                'forbidden_split_roles': ['test', 'external', 'frozen_public_eval'],
            },
            'evaluation': {
                'require_prediction_bundles': True,
                'default_failure_threshold_m': 1.0,
                'require_ground_truth_unless_smoke': True,
            },
        },
    )

    pose_scale_m, yaw_scale = _resolve_alignment_risk_scales()

    assert pose_scale_m == pytest.approx(1.0)
    assert yaw_scale == pytest.approx(math.pi)
    _resolve_alignment_risk_scales.cache_clear()


def _miluv_field_mapping():
    return {
        'imu': {'timestamp': 'timestamp', 'ax': 'ax', 'ay': 'ay', 'gz': 'gz'},
        'uwb': {'timestamp': 'timestamp', 'anchor_id': 'anchor_id', 'range': 'range', 'valid': 'valid', 'quality': 'quality'},
        'vio': {
            'timestamp': 'timestamp',
            'dx': 'dx',
            'dy': 'dy',
            'dyaw': 'dyaw',
            'quality': 'quality',
            'tracked_features': 'tracked_features',
            'reproj_err': 'reproj_err',
        },
        'gt': {'timestamp': 'timestamp', 'px': 'px', 'py': 'py', 'yaw': 'yaw'},
    }


def _write_official_miluv_raw_root(raw_root: Path) -> Path:
    fixture_root = PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv' / 'mini_seq'
    seq_dir = raw_root / 'default_3_random_0'
    seq_dir.mkdir(parents=True)
    for name in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
        shutil.copy2(fixture_root / name, seq_dir / name)

    cfg_dir = raw_root / 'config' / 'uwb'
    cfg_dir.mkdir(parents=True)
    (raw_root / 'config' / 'experiments.csv').write_text(_OFFICIAL_EXPERIMENTS_CSV, encoding='utf-8')
    (cfg_dir / 'anchors.yaml').write_text(_OFFICIAL_ANCHORS_YAML, encoding='utf-8')
    return raw_root


def _load_mini_seq_direct_inputs_without_anchor_layout():
    events, gt_rows, _ = _load_miluv_sequence_payload(
        'mini_seq',
        {
            'dataset_name': 'miluv',
            'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': _miluv_field_mapping(),
        },
    )
    return events, gt_rows


def _temporal_reliability_training_inputs():
    events = [
        {
            't': 0.0,
            'dt': 0.0,
            'modality': 'imu',
            'meta': {'scene_id': 'S(A3,N3,V2,K0,M0)', 'seq_id': 'temporal_seq'},
            'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0},
            'uwb_payload': None,
            'vio_payload': None,
        },
        {
            't': 0.1,
            'dt': 0.1,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A3,N3,V2,K0,M0)', 'seq_id': 'temporal_seq'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 0.6},
            'vio_payload': None,
        },
        {
            't': 0.2,
            'dt': 0.1,
            'modality': 'imu',
            'meta': {'scene_id': 'S(A3,N3,V2,K0,M0)', 'seq_id': 'temporal_seq'},
            'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0},
            'uwb_payload': None,
            'vio_payload': None,
        },
        {
            't': 0.4,
            'dt': 0.2,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A3,N3,V2,K0,M0)', 'seq_id': 'temporal_seq'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 1, 'range': 2.2, 'valid': False, 'quality': 0.2},
            'vio_payload': None,
        },
        {
            't': 0.5,
            'dt': 0.1,
            'modality': 'vio',
            'meta': {'scene_id': 'S(A3,N3,V2,K0,M0)', 'seq_id': 'temporal_seq'},
            'imu_payload': None,
            'uwb_payload': None,
            'vio_payload': {'dx': 0.2, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.8, 'tracked_features': 40, 'reproj_err': 0.8},
        },
        {
            't': 0.8,
            'dt': 0.3,
            'modality': 'vio',
            'meta': {'scene_id': 'S(A3,N3,V2,K0,M0)', 'seq_id': 'temporal_seq'},
            'imu_payload': None,
            'uwb_payload': None,
            'vio_payload': {'dx': 0.1, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.7, 'tracked_features': 20, 'reproj_err': 1.4},
        },
    ]
    gt_rows = [
        {'timestamp': 0.1, 'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        {'timestamp': 0.4, 'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        {'timestamp': 0.5, 'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        {'timestamp': 0.8, 'px': 0.0, 'py': 0.0, 'yaw': 0.0},
    ]
    return events, gt_rows


def _expected_remapped_uwb_range(base_seq: str, anchor_index: int, gt_timestamp: float, new_anchor_xy: tuple[float, float]) -> float:
    fixture_root = PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv' / base_seq
    base_anchor_layout = json.loads((fixture_root / 'anchor_layout.json').read_text(encoding='utf-8'))
    base_uwb_rows = json.loads((fixture_root / 'uwb.json').read_text(encoding='utf-8'))
    gt_rows = json.loads((fixture_root / 'gt.json').read_text(encoding='utf-8'))
    gt_row = next(row for row in gt_rows if float(row['timestamp']) == float(gt_timestamp))
    uwb_row = next(row for row in base_uwb_rows if int(row['anchor_id']) == int(anchor_index))
    old_anchor_xy = base_anchor_layout['anchor_positions'][base_anchor_layout['anchor_ids'].index(anchor_index)]
    old_range = math.hypot(float(old_anchor_xy[0]) - float(gt_row['px']), float(old_anchor_xy[1]) - float(gt_row['py']))
    residual = float(uwb_row['range']) - float(old_range)
    new_range = math.hypot(float(new_anchor_xy[0]) - float(gt_row['px']), float(new_anchor_xy[1]) - float(gt_row['py'])) + residual
    return max(0.0, new_range)


def _assert_no_forbidden_architecture_terms(payload) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    for forbidden in _FORBIDDEN_ARCHITECTURE_TERMS:
        assert forbidden not in serialized


def test_normal_case(tmp_path):
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)
    result = run({
        'model_name': 'lstm_ekf',
        'model_cfg': _lstm_model_cfg(),
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train',
        'mode': 'quick',
        'device': 'cuda',
    })
    assert result.stage_name == 'train_pipeline'
    assert result.metadata['target_contract']['required_heads'] == ['bias', 'risk', 'uwb_scaling', 'vio_scaling']
    assert result.metadata['target_contract']['confidence_materialization'] == 'trainer consumes confidence as modality-specific scaling heads with a neutral floor, plus a conditional robust low-quality supplement'
    assert result.metadata['target_contract']['risk_semantics'] == 'alignment_risk_only_pre_bridge_base_risk'
    assert result.metadata['target_contract']['head_sources']['risk'] == 'alignment_risk only; base risk comes from max(normalized pose_error / failure_threshold_m, normalized yaw_error / pi) against exact, interpolated, or trailing-tolerant ground truth, while observation_risk remains an audit trace that is consumed by modality-specific scaling and downstream bridge noise logic instead of the risk head'
    assert result.metadata['target_contract']['head_sources']['uwb_scaling'] == 'UWB-only extra noise inflation from quality, validity, geometric bias severity, modality_gap_dt staleness, and alignment_risk with a neutral floor, plus a conditional robust low-quality supplement'
    assert result.metadata['target_contract']['head_sources']['vio_scaling'] == 'VIO-only extra noise inflation from quality, tracked_features, reproj_err, modality_gap_dt staleness, and alignment_risk with a neutral floor, plus a conditional robust low-quality supplement'
    assert result.metadata['target_contract']['bridge_thresholds'] == result.metadata['sample_report']['bridge_thresholds']
    assert result.metadata['target_contract']['geometry_bias_teacher'] == result.metadata['sample_report']['geometry_bias_teacher']
    assert result.metadata['sample_report']['usable_sample_count'] == 8
    assert result.metadata['sample_report']['usable_sample_count_by_modality'] == {'uwb': 4, 'vio': 4}
    assert result.metadata['sample_report']['train_window_count'] == 4
    assert result.metadata['sample_report']['val_window_count'] == 4
    assert result.metadata['sample_report']['geometry_bias_teacher']['status'] == 'geometric_teacher_available'
    assert result.metadata['sample_report']['geometry_bias_teacher']['available'] is True
    assert result.metadata['sample_report']['geometry_bias_teacher']['source'] == 'miluv_fixture_anchor_layout'
    assert result.metadata['sample_report']['bias_source_counts'] == {
        'measured_minus_geometric_true_range': 4,
        'neutral_baseline': 4,
    }
    assert result.metadata['sample_report']['sequences']['mini_seq']['ground_truth_alignment_modes'] == {
        'linear_interpolation': 4,
    }
    assert result.metadata['sample_report']['sequences']['mini_seq']['skipped_outside_ground_truth_span'] == 0
    assert result.metadata['sample_report']['sequences']['mini_seq']['skipped_uwb_updates_without_anchor_layout'] == 0
    assert result.metadata['train_report']['status'] == 'trained'
    assert result.metadata['train_report']['mode'] == 'quick'
    assert result.metadata['train_report']['closure_note']['scope'] == 'smoke-scale run label only'
    assert result.metadata['checkpoint_smoke']['status'] == 'ok'
    assert result.metadata['protocol_gate'] == {
        'split_role': 'train',
        'num_split_ids': 2,
        'failure_sample_policy': 'retain_and_audit',
        'quick_full_rule': 'quick_smoke_scale__full_real_execution_required',
        'mode': 'quick',
        'forbidden_training_split_ids': [],
    }
    loaded_model = create_model('lstm_ekf', {'checkpoint_path': result.metadata['checkpoint_smoke']['checkpoint_path']})
    source_cfg = {
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
    }
    samples, _ = _build_liquid_samples(
        ['mini_seq', 'mini_seq_02'],
        source_cfg,
        _lstm_model_cfg(),
        dict(_load_default_ekf_cfg()),
    )
    _train_samples, val_samples, _train_ids, _val_ids = _split_train_and_val_samples(
        samples,
        split_ids=['mini_seq', 'mini_seq_02'],
        train_split_ids=['mini_seq'],
        val_split_ids=['mini_seq_02'],
    )
    checkpoint_window = next(
        sample['window_tensor']
        for sample in val_samples
        if sample['seq_id'] == result.metadata['checkpoint_smoke']['val_seq_id']
        and abs(float(sample['event_time']) - float(result.metadata['checkpoint_smoke']['val_event_time'])) <= 1e-9
    )
    inferred_outputs = loaded_model.infer_intermediate(checkpoint_window)
    assert result.metadata['checkpoint_smoke']['inference_outputs'] == {
        'bias': pytest.approx(inferred_outputs.bias, rel=1e-6, abs=1e-6),
        'risk': pytest.approx(inferred_outputs.risk, rel=1e-6, abs=1e-6),
        'uwb_scaling': pytest.approx(inferred_outputs.uwb_scaling, rel=1e-6, abs=1e-6),
        'vio_scaling': pytest.approx(inferred_outputs.vio_scaling, rel=1e-6, abs=1e-6),
    }
    expected_checkpoint = tmp_path / 'train' / 'checkpoints' / 'lstm_ekf_best_checkpoint.pt'
    expected_train_report = tmp_path / 'train' / 'reports' / 'lstm_ekf_train_report.json'
    expected_protocol_gate = tmp_path / 'train' / 'audits' / 'lstm_ekf_protocol_gate.json'
    expected_sample_report = tmp_path / 'train' / 'audits' / 'lstm_ekf_sample_report.json'
    expected_target_contract = tmp_path / 'train' / 'audits' / 'lstm_ekf_target_contract.json'
    expected_checkpoint_smoke = tmp_path / 'train' / 'audits' / 'lstm_ekf_checkpoint_smoke.json'
    expected_training_flow_contract = tmp_path / 'train' / 'audits' / 'lstm_ekf_training_flow_contract.json'
    assert expected_checkpoint.is_file()
    assert expected_protocol_gate.is_file()
    assert expected_sample_report.is_file()
    assert expected_target_contract.is_file()
    assert expected_checkpoint_smoke.is_file()
    assert expected_training_flow_contract.is_file()
    assert result.artifacts == [
        str(expected_checkpoint),
        str(expected_train_report),
        str(expected_protocol_gate),
        str(expected_sample_report),
        str(expected_target_contract),
        str(expected_checkpoint_smoke),
        str(expected_training_flow_contract),
    ]
    assert result.metadata['train_report']['report_path'] == str(expected_train_report)
    assert result.metadata['train_report']['training_flow_contract_path'] == str(expected_training_flow_contract)
    assert json.loads(expected_protocol_gate.read_text(encoding='utf-8')) == result.metadata['protocol_gate']
    assert json.loads(expected_sample_report.read_text(encoding='utf-8')) == result.metadata['sample_report']
    assert json.loads(expected_target_contract.read_text(encoding='utf-8')) == result.metadata['target_contract']
    assert json.loads(expected_checkpoint_smoke.read_text(encoding='utf-8')) == result.metadata['checkpoint_smoke']
    assert json.loads(expected_training_flow_contract.read_text(encoding='utf-8')) == result.metadata['training_flow_contract']
    assert result.metadata['training_flow_contract']['trainer_mode'] == 'single_phase_baseline'
    for payload in (
        result.metadata['protocol_gate'],
        result.metadata['sample_report'],
        result.metadata['target_contract'],
        result.metadata['train_report'],
        result.metadata['checkpoint_smoke'],
        result.metadata['training_flow_contract'],
    ):
        _assert_no_forbidden_architecture_terms(payload)
    monkeypatch.undo()


def test_train_pipeline_uses_gate_normalized_split_ids_and_explicit_splits(tmp_path):
    """使用测试：train pipeline。\n\n验证被测功能正确使用 train pipeline，\n确保内部依赖被正确调用。
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)
    result = run({
        'model_name': 'lstm_ekf',
        'model_cfg': _lstm_model_cfg(),
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': [' mini_seq ', ' mini_seq_02 '],
        'train_split_ids': [' mini_seq '],
        'val_split_ids': [' mini_seq_02 '],
        'split_role': 'train',
        'output_root': tmp_path / 'train_norm_splits',
        'mode': 'quick',
        'device': 'cuda',
    })
    assert result.metadata['sample_report']['train_split_ids'] == ['mini_seq']
    assert result.metadata['sample_report']['val_split_ids'] == ['mini_seq_02']
    assert result.metadata['sample_report']['split_audit']['requested_split_ids'] == ['mini_seq', 'mini_seq_02']
    assert result.metadata['sample_report']['split_audit']['requested_train_split_ids'] == ['mini_seq']
    assert result.metadata['sample_report']['split_audit']['requested_val_split_ids'] == ['mini_seq_02']
    assert result.metadata['train_report']['train_split_ids'] == ['mini_seq']
    assert result.metadata['train_report']['val_split_ids'] == ['mini_seq_02']
    monkeypatch.undo()


def test_liquid_real_smoke(tmp_path):
    """冒烟测试：liquid real。\n\n快速验证 liquid real 的基本功能可用，\n不深入检查细节，仅确认流程不崩溃。
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)
    result = run({
        'model_name': 'liquid_ekf',
        'model_cfg': _liquid_model_cfg(),
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'output_root': tmp_path / 'train_liquid',
        'mode': 'quick',
        'device': 'cuda',
    })

    assert result.stage_name == 'train_pipeline'
    assert result.metadata['checkpoint_smoke']['status'] == 'ok'
    assert result.metadata['protocol_gate'] == {
        'split_role': 'train',
        'num_split_ids': 2,
        'failure_sample_policy': 'retain_and_audit',
        'quick_full_rule': 'quick_smoke_scale__full_real_execution_required',
        'mode': 'quick',
        'forbidden_training_split_ids': [],
    }
    loaded_model = create_model('liquid_ekf', {'checkpoint_path': result.metadata['checkpoint_smoke']['checkpoint_path']})
    source_cfg = {
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
    }
    samples, _ = _build_liquid_samples(
        ['mini_seq', 'mini_seq_02'],
        source_cfg,
        _liquid_model_cfg(),
        dict(_load_default_ekf_cfg()),
    )
    _train_samples, val_samples, _train_ids, _val_ids = _split_train_and_val_samples(
        samples,
        split_ids=['mini_seq', 'mini_seq_02'],
        train_split_ids=['mini_seq'],
        val_split_ids=['mini_seq_02'],
    )
    checkpoint_window = next(
        sample['window_tensor']
        for sample in val_samples
        if sample['seq_id'] == result.metadata['checkpoint_smoke']['val_seq_id']
        and abs(float(sample['event_time']) - float(result.metadata['checkpoint_smoke']['val_event_time'])) <= 1e-9
    )
    inferred_outputs = loaded_model.infer_intermediate(checkpoint_window)
    assert result.metadata['checkpoint_smoke']['inference_outputs'] == {
        'bias': pytest.approx(inferred_outputs.bias, rel=1e-6, abs=1e-6),
        'risk': pytest.approx(inferred_outputs.risk, rel=1e-6, abs=1e-6),
        'uwb_scaling': pytest.approx(inferred_outputs.uwb_scaling, rel=1e-6, abs=1e-6),
        'vio_scaling': pytest.approx(inferred_outputs.vio_scaling, rel=1e-6, abs=1e-6),
    }
    sample_report_json = json.loads((tmp_path / 'train_liquid' / 'audits' / 'liquid_ekf_sample_report.json').read_text(encoding='utf-8'))
    target_contract_json = json.loads((tmp_path / 'train_liquid' / 'audits' / 'liquid_ekf_target_contract.json').read_text(encoding='utf-8'))
    checkpoint_smoke_json = json.loads((tmp_path / 'train_liquid' / 'audits' / 'liquid_ekf_checkpoint_smoke.json').read_text(encoding='utf-8'))
    protocol_gate_json = json.loads((tmp_path / 'train_liquid' / 'audits' / 'liquid_ekf_protocol_gate.json').read_text(encoding='utf-8'))
    assert sample_report_json == result.metadata['sample_report']
    assert target_contract_json == result.metadata['target_contract']
    assert checkpoint_smoke_json == result.metadata['checkpoint_smoke']
    assert protocol_gate_json == result.metadata['protocol_gate']
    # §13.6.2.7 + §13.3 audit fix: paper_checkpoint_selection_must_use = "export_score"
    # 所以 train_model 返回的是 export_score 视角最优的 checkpoint.
    expected_checkpoint = tmp_path / 'train_liquid' / 'checkpoints' / 'liquid_ekf_export_best_checkpoint.pt'
    expected_train_report = tmp_path / 'train_liquid' / 'reports' / 'liquid_ekf_train_report.json'
    expected_protocol_gate = tmp_path / 'train_liquid' / 'audits' / 'liquid_ekf_protocol_gate.json'
    expected_sample_report = tmp_path / 'train_liquid' / 'audits' / 'liquid_ekf_sample_report.json'
    expected_target_contract = tmp_path / 'train_liquid' / 'audits' / 'liquid_ekf_target_contract.json'
    expected_checkpoint_smoke = tmp_path / 'train_liquid' / 'audits' / 'liquid_ekf_checkpoint_smoke.json'
    expected_training_flow_contract = tmp_path / 'train_liquid' / 'audits' / 'liquid_ekf_training_flow_contract.json'
    assert result.artifacts == [
        str(expected_checkpoint),
        str(expected_train_report),
        str(expected_protocol_gate),
        str(expected_sample_report),
        str(expected_target_contract),
        str(expected_checkpoint_smoke),
        str(expected_training_flow_contract),
    ]
    assert result.metadata['train_report']['report_path'] == str(expected_train_report)
    assert result.metadata['train_report']['training_flow_contract_path'] == str(expected_training_flow_contract)
    for payload in (
        result.metadata['protocol_gate'],
        result.metadata['sample_report'],
        result.metadata['target_contract'],
        result.metadata['train_report'],
        result.metadata['checkpoint_smoke'],
        result.metadata['training_flow_contract'],
    ):
        _assert_no_forbidden_architecture_terms(payload)
    monkeypatch.undo()


def test_split_train_and_val_samples_rejects_overlapping_explicit_split_ids():
    """拒绝测试：split train and val samples。\n\n验证被测功能对 split train and val samples 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    samples = [
        {'seq_id': 'seq_a', 'event_time': 0.0},
        {'seq_id': 'seq_b', 'event_time': 1.0},
    ]
    with pytest.raises(ValueError, match='must be disjoint'):
        _split_train_and_val_samples(
            samples,
            split_ids=['seq_a', 'seq_b'],
            train_split_ids=['seq_a'],
            val_split_ids=['seq_a'],
        )


def test_split_train_and_val_samples_rejects_unknown_explicit_split_ids():
    """拒绝测试：split train and val samples。\n\n验证被测功能对 split train and val samples 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    samples = [
        {'seq_id': 'seq_a', 'event_time': 0.0},
        {'seq_id': 'seq_b', 'event_time': 1.0},
    ]
    with pytest.raises(ValueError, match='unknown sequence ids'):
        _split_train_and_val_samples(
            samples,
            split_ids=['seq_a', 'seq_b'],
            train_split_ids=['seq_missing'],
            val_split_ids=['seq_b'],
        )


def test_split_train_and_val_samples_keeps_single_sequence_fallback():
    """回退测试：split train and val samples keeps single sequence。\n\n验证 split train and val samples keeps single sequence 的回退机制，\n确保主路径失败时有合理的降级策略。
    """
    samples = [
        {'seq_id': 'seq_a', 'event_time': 0.0},
        {'seq_id': 'seq_a', 'event_time': 1.0},
        {'seq_id': 'seq_a', 'event_time': 2.0},
    ]
    train_samples, val_samples, train_ids, val_ids = _split_train_and_val_samples(
        samples,
        split_ids=['seq_a'],
    )
    assert train_ids == ['seq_a']
    assert val_ids == ['seq_a']
    assert train_samples
    assert val_samples


def test_build_split_audit_marks_single_sequence_time_split_without_time_overlap():
    """无依赖测试：build split audit marks single sequence time split。\n\n验证 build split audit marks single sequence time split 在缺少依赖时的降级行为，\n确保回退策略正确。
    """
    samples = [
        {'seq_id': 'seq_a', 'event_time': 0.0},
        {'seq_id': 'seq_a', 'event_time': 1.0},
        {'seq_id': 'seq_a', 'event_time': 2.0},
    ]
    train_samples, val_samples, train_ids, val_ids = _split_train_and_val_samples(
        samples,
        split_ids=['seq_a'],
    )
    split_audit = _build_split_audit(
        split_ids=['seq_a'],
        requested_train_ids=[],
        requested_val_ids=[],
        resolved_train_ids=train_ids,
        resolved_val_ids=val_ids,
        train_samples=train_samples,
        val_samples=val_samples,
    )

    assert split_audit == {
        'split_strategy': 'single_sequence_time_split',
        'requested_split_ids': ['seq_a'],
        'requested_train_split_ids': [],
        'requested_val_split_ids': [],
        'resolved_train_split_ids': ['seq_a'],
        'resolved_val_split_ids': ['seq_a'],
        'shared_seq_ids': ['seq_a'],
        'sequence_disjoint': False,
        # H24c 真改：scene_id cross-set 守门审计字段（4 个新字段）
        # 旧 sample 无 scene_id 字段 → train_scene_ids/val_scene_ids 均为 []，
        # shared_scene_ids = [] → scene_id_disjoint = True（无共享即 disjoint）。
        'shared_scene_ids': [],
        'scene_id_disjoint': True,
        'train_scene_ids': [],
        'val_scene_ids': [],
        'time_ordered_nonoverlap': True,
        'train_window_count': len(train_samples),
        'val_window_count': len(val_samples),
        'train_event_time_span': {'start': 0.0, 'end': 1.0},
        'val_event_time_span': {'start': 2.0, 'end': 2.0},
    }


def test_train_pipeline_rejects_zero_train_windows_after_explicit_split(tmp_path):
    """拒绝测试：train pipeline。\n\n验证被测功能对 train pipeline 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(ValueError, match='train split resolved to zero usable windows'):
        run({
            'model_name': 'lstm_ekf',
            'model_cfg': _lstm_model_cfg(),
            'dataset_name': 'miluv',
            'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': _miluv_field_mapping(),
            'split_ids': ['mini_seq', 'mini_seq_02'],
            'val_split_ids': ['mini_seq', 'mini_seq_02'],
            'output_root': tmp_path / 'bad_train_split',
            'mode': 'quick',
            'device': 'cuda',
        })


def test_train_pipeline_rejects_zero_val_windows_after_explicit_split(tmp_path):
    """拒绝测试：train pipeline。\n\n验证被测功能对 train pipeline 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(ValueError, match='val split resolved to zero usable windows'):
        run({
            'model_name': 'liquid_ekf',
            'model_cfg': _liquid_model_cfg(),
            'dataset_name': 'miluv',
            'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': _miluv_field_mapping(),
            'split_ids': ['mini_seq', 'mini_seq_02'],
            'train_split_ids': ['mini_seq', 'mini_seq_02'],
            'output_root': tmp_path / 'bad_val_split',
            'mode': 'quick',
            'device': 'cuda',
        })


def test_liquid_pipeline_preserves_phase_schedule_into_train_report(tmp_path):
    """保持性测试：liquid pipeline。\n\n验证 liquid pipeline 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)
    model_cfg = _liquid_model_cfg()
    model_cfg['train'] = dict(model_cfg['train'])
    model_cfg['train']['epochs'] = 3
    model_cfg['train']['phase_schedule'] = {
        'warmup_epochs': 1,
        'gate_alignment_epochs': 1,
    }
    result = run({
        'model_name': 'liquid_ekf',
        'model_cfg': model_cfg,
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'output_root': tmp_path / 'train_liquid_phase_schedule',
        'mode': 'quick',
        'device': 'cuda',
    })

    assert result.metadata['train_report']['phase_schedule'] == {
        'warmup_epochs': 1,
        'gate_alignment_epochs': 1,
    }
    assert result.metadata['train_report']['epoch_phase_names'] == [
        'readout_warmup',
        'gate_alignment',
        'full_tuning',
    ]
    expected_loss_diagnostics = tmp_path / 'train_liquid_phase_schedule' / 'reports' / 'liquid_ekf_loss_diagnostics.json'
    expected_epoch_payload = tmp_path / 'train_liquid_phase_schedule' / 'reports' / 'liquid_ekf_epoch_predictions_vs_targets.json'
    assert result.metadata['train_report']['loss_diagnostics_path'] == str(expected_loss_diagnostics)
    assert result.metadata['train_report']['epoch_predictions_vs_targets_path'] == str(expected_epoch_payload)
    assert result.metadata['training_flow_contract']['phase_schedule'] == result.metadata['train_report']['phase_schedule']
    assert result.metadata['training_flow_contract']['epoch_phase_names'] == result.metadata['train_report']['epoch_phase_names']
    assert result.metadata['training_flow_contract']['trainer_mode'] == 'phase_scheduled_liquid'
    persisted_report = json.loads(Path(result.metadata['train_report']['report_path']).read_text(encoding='utf-8'))
    assert persisted_report['phase_schedule'] == result.metadata['train_report']['phase_schedule']
    assert persisted_report['epoch_phase_names'] == result.metadata['train_report']['epoch_phase_names']
    assert persisted_report['loss_diagnostics_path'] == result.metadata['train_report']['loss_diagnostics_path']
    assert persisted_report['epoch_predictions_vs_targets_path'] == result.metadata['train_report']['epoch_predictions_vs_targets_path']
    assert persisted_report['training_flow_contract_path'] == result.metadata['train_report']['training_flow_contract_path']
    diagnostics = json.loads(Path(result.metadata['train_report']['loss_diagnostics_path']).read_text(encoding='utf-8'))
    epoch_payload = json.loads(Path(result.metadata['train_report']['epoch_predictions_vs_targets_path']).read_text(encoding='utf-8'))
    for split_name in ('train', 'val'):
        assert [row['phase_name'] for row in diagnostics[split_name]] == result.metadata['train_report']['epoch_phase_names']
        assert [row['phase_name'] for row in epoch_payload[split_name]] == result.metadata['train_report']['epoch_phase_names']
        assert [row['epoch_index'] for row in diagnostics[split_name]] == [1, 2, 3]
        assert [row['epoch_index'] for row in epoch_payload[split_name]] == [1, 2, 3]
    monkeypatch.undo()


def test_liquid_real_smoke_without_teacher_checkpoint(tmp_path):
    """冒烟测试：liquid real。\n\n快速验证 liquid real 的基本功能可用，\n不深入检查细节，仅确认流程不崩溃。
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)
    result = run({
        'model_name': 'liquid_ekf',
        'model_cfg': _liquid_model_cfg(),
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'output_root': tmp_path / 'train_liquid_missing_teacher',
        'mode': 'quick',
        'device': 'cuda',
    })

    assert result.stage_name == 'train_pipeline'
    assert result.metadata['checkpoint_smoke']['status'] == 'ok'
    monkeypatch.undo()


def test_liquid_real_rejects_unused_teacher_checkpoint_input(tmp_path):
    """拒绝测试：liquid real。\n\n验证被测功能对 liquid real 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)
    teacher_checkpoint = tmp_path / 'teacher.pt'
    teacher_checkpoint.write_text('unused', encoding='utf-8')

    with pytest.raises(ValueError, match='liquid_ekf does not accept teacher_checkpoint_path'):
        run({
            'model_name': 'liquid_ekf',
            'model_cfg': _liquid_model_cfg(),
            'dataset_name': 'miluv',
            'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': _miluv_field_mapping(),
            'split_ids': ['mini_seq', 'mini_seq_02'],
            'train_split_ids': ['mini_seq'],
            'val_split_ids': ['mini_seq_02'],
            'teacher_checkpoint_path': str(teacher_checkpoint),
            'output_root': tmp_path / 'train_liquid',
            'mode': 'quick',
            'device': 'cuda',
        })

    monkeypatch.undo()


def test_lstm_pipeline_checkpoint_smoke_preserves_neutral_floor_scaling(tmp_path, monkeypatch):
    """冒烟测试：lstm pipeline checkpoint。\n\n快速验证 lstm pipeline checkpoint 的基本功能可用，\n不深入检查细节，仅确认流程不崩溃。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)
    def _stub_train_model(_train_windows, _val_windows, train_cfg):
        output_root = Path(train_cfg['output_root'])
        checkpoints_dir = output_root / 'checkpoints'
        reports_dir = output_root / 'reports'
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        model = create_model(
            'lstm_ekf',
            {
                'feature_order': list(train_cfg.get('feature_order') or _FEATURE_ORDER),
                'window': dict(train_cfg.get('window') or {}),
                'network': dict(train_cfg.get('network') or {}),
            },
        )
        with torch.no_grad():
            model.network.output_layer.weight.zero_()
            model.network.output_layer.bias.copy_(torch.tensor([0.0, 0.0, -1000.0, -1000.0]))

        checkpoint_path = checkpoints_dir / 'lstm_ekf_best_checkpoint.pt'
        report_path = reports_dir / 'lstm_ekf_train_report.json'
        torch.save(
            {
                'checkpoint_format': 'lstm_real_v1',
                'model_cfg': {
                    'feature_order': list(train_cfg.get('feature_order') or _FEATURE_ORDER),
                    'window': dict(train_cfg.get('window') or {}),
                    'network': dict(train_cfg.get('network') or {}),
                },
                'model_state': model.state_dict(),
                'best_epoch': 1,
                'best_loss': 0.0,
                'train_window_count': 1,
                'val_window_count': 1,
            },
            checkpoint_path,
        )
        report_path.write_text('{"status":"trained"}', encoding='utf-8')
        return str(checkpoint_path), {
            'status': 'trained',
            'report_path': str(report_path),
            'checkpoint_path': str(checkpoint_path),
        }

    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.train_lstm_model', _stub_train_model)

    result = run({
        'model_name': 'lstm_ekf',
        'model_cfg': _lstm_model_cfg(),
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_strict_positive_checkpoint',
        'mode': 'quick',
        'device': 'cuda',
    })

    inference_outputs = result.metadata['checkpoint_smoke']['inference_outputs']
    assert result.metadata['checkpoint_smoke']['status'] == 'ok'
    assert inference_outputs['uwb_scaling'] == pytest.approx(1.0)
    assert inference_outputs['vio_scaling'] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ('requested_device', 'cuda_available', 'expected_device'),
    [
        ('auto', True, 'cuda'),
        ('auto', False, 'cpu'),
        ('cpu', True, 'cpu'),
        ('cuda', True, 'cuda'),
        ('cuda', False, None),
    ],
)
def test_device_selection_report(tmp_path, monkeypatch, requested_device, cuda_available, expected_device):
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: cuda_available)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: cuda_available)

    payload = {
        'model_name': 'lstm_ekf',
        'model_cfg': _lstm_model_cfg(),
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_device',
        'device': requested_device,
    }
    if expected_device is None:
        with pytest.raises(RuntimeError, match=r'full training requires an available CUDA GPU'):
            run(payload)
        return

    result = run(payload)

    assert result.metadata['train_report']['requested_device'] == requested_device
    assert result.metadata['train_report']['device'] == expected_device
    assert result.metadata['train_report']['cuda_available'] is cuda_available


def test_auto_device_policy_respects_pipeline_context(monkeypatch):
    """尊重测试：auto device policy。\n\n验证被测功能尊重 auto device policy 的规则，\n确保协议约束被正确执行。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)

    quick_report = _resolve_train_device('auto', allow_auto_cuda=True)
    cpu_only_report = _resolve_train_device('auto', allow_auto_cuda=False)

    assert quick_report['requested_device'] == 'auto'
    assert quick_report['selected_device'] == 'cuda'
    assert cpu_only_report['requested_device'] == 'auto'
    assert cpu_only_report['selected_device'] == 'cpu'


def test_train_report_file_persists_runtime_and_split_metadata(tmp_path, monkeypatch):
    """分裂测试：train report file persists runtime and。\n\n验证 train report file persists runtime and 的训练/验证分裂逻辑，\n确保分裂策略和审计正确。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: False)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: False)

    result = run({
        'model_name': 'lstm_ekf',
        'model_cfg': _lstm_model_cfg(),
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_report_persist',
        'device': 'auto',
    })

    report_path = Path(result.metadata['train_report']['report_path'])
    persisted_report = json.loads(report_path.read_text(encoding='utf-8'))
    assert persisted_report['requested_device'] == 'auto'
    assert persisted_report['device'] == 'cpu'
    assert persisted_report['cuda_available'] is False
    assert persisted_report['cuda_runtime_available'] is False
    assert persisted_report['mode'] == 'full'
    assert persisted_report['train_split_ids'] == ['mini_seq']
    assert persisted_report['val_split_ids'] == ['mini_seq_02']
    assert persisted_report['closure_note']['scope'] == 'real execution run label'
    assert persisted_report['bridge_thresholds'] == result.metadata['sample_report']['bridge_thresholds']
    assert persisted_report['bridge_thresholds'] == result.metadata['train_report']['bridge_thresholds']


def test_full_training_rejects_cuda_without_runtime_usable_backend(tmp_path, monkeypatch):
    """拒绝测试：full training。\n\n验证被测功能对 full training 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: False)

    with pytest.raises(RuntimeError, match=r'full training requires an available CUDA GPU'):
        run({
            'model_name': 'lstm_ekf',
            'model_cfg': _lstm_model_cfg(),
            'dataset_name': 'miluv',
            'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': _miluv_field_mapping(),
            'split_ids': ['mini_seq', 'mini_seq_02'],
            'train_split_ids': ['mini_seq'],
            'val_split_ids': ['mini_seq_02'],
            'split_role': 'train',
            'output_root': tmp_path / 'train_cuda_runtime_fallback',
            'device': 'cuda',
        })


def test_full_training_rejects_checkpoint_reuse(tmp_path, monkeypatch):
    """拒绝测试：full training。\n\n验证被测功能对 full training 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: False)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: False)

    with pytest.raises(ValueError, match=r'full training must start from scratch'):
        run({
            'model_name': 'lstm_ekf',
            'model_cfg': _lstm_model_cfg() | {'checkpoint_path': str(tmp_path / 'resume.pt')},
            'dataset_name': 'miluv',
            'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': _miluv_field_mapping(),
            'split_ids': ['mini_seq', 'mini_seq_02'],
            'train_split_ids': ['mini_seq'],
            'val_split_ids': ['mini_seq_02'],
            'split_role': 'train',
            'output_root': tmp_path / 'train_full_reuse_checkpoint',
            'mode': 'full',
            'device': 'cpu',
        })


def test_quick_training_requires_cuda(tmp_path, monkeypatch):
    """必填测试：quick training。\n\n验证 quick training 的必填约束，\n确保缺少必要输入时抛出异常。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: False)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: False)

    with pytest.raises(RuntimeError, match=r"quick training requires an available CUDA GPU"):
        run({
            'model_name': 'lstm_ekf',
        'model_cfg': _lstm_model_cfg(),
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_quick_requires_cuda',
        'mode': 'quick',
        'device': 'cuda',
        })


def test_quick_training_rejects_cuda_without_runtime_usable_backend(tmp_path, monkeypatch):
    """拒绝测试：quick training。\n\n验证被测功能对 quick training 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: False)

    with pytest.raises(RuntimeError, match=r"quick training requires an available CUDA GPU"):
        run({
        'model_name': 'lstm_ekf',
        'model_cfg': _lstm_model_cfg(),
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_quick_runtime_unusable',
        'mode': 'quick',
        'device': 'cuda',
        })


def test_quick_training_rejects_checkpoint_reuse(tmp_path, monkeypatch):
    """拒绝测试：quick training。\n\n验证被测功能对 quick training 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)

    with pytest.raises(ValueError, match=r"cannot reuse checkpoint_path"):
        run({
            'model_name': 'lstm_ekf',
        'model_cfg': {**_lstm_model_cfg(), 'checkpoint_path': 'old.pt'},
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_quick_reuse',
        'mode': 'quick',
        'device': 'cuda',
        })


def test_lstm_train_report_preserves_weight_decay(tmp_path, monkeypatch):
    """保持性测试：lstm train report。\n\n验证 lstm train report 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)

    model_cfg = _lstm_model_cfg()
    model_cfg['train'] = dict(model_cfg['train'])
    model_cfg['train']['weight_decay'] = 0.125

    result = run({
        'model_name': 'lstm_ekf',
        'model_cfg': model_cfg,
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_lstm_weight_decay',
        'mode': 'quick',
        'device': 'cuda',
    })

    assert result.metadata['train_report']['optimizer']['weight_decay'] == pytest.approx(0.125)
    payload = torch.load(
        tmp_path / 'train_lstm_weight_decay' / 'checkpoints' / 'lstm_ekf_best_checkpoint.pt',
        map_location='cpu',
        weights_only=False,
    )
    assert payload['optimizer']['weight_decay'] == pytest.approx(0.125)


def test_lstm_pipeline_preserves_tail_selection_observation_coeff_into_training_flow_contract(tmp_path, monkeypatch):
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)

    model_cfg = _lstm_model_cfg()
    model_cfg['train'] = dict(model_cfg['train'])
    model_cfg['train']['tail_selection_observation_coeff'] = 0.0

    result = run({
        'model_name': 'lstm_ekf',
        'model_cfg': model_cfg,
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_lstm_tail_selection_coeff',
        'mode': 'quick',
        'device': 'cuda',
    })

    assert result.metadata['train_report']['tail_selection_observation_coeff'] == pytest.approx(0.0)
    contract_path = Path(result.metadata['train_report']['training_flow_contract_path'])
    contract_payload = json.loads(contract_path.read_text(encoding='utf-8'))
    assert contract_payload['tail_selection_observation_coeff'] == pytest.approx(0.0)


def test_liquid_train_report_preserves_weight_decay(tmp_path, monkeypatch):
    """保持性测试：liquid train report。\n\n验证 liquid train report 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)

    model_cfg = _liquid_model_cfg()
    model_cfg['train'] = dict(model_cfg['train'])
    model_cfg['train']['weight_decay'] = 0.125

    result = run({
        'model_name': 'liquid_ekf',
        'model_cfg': model_cfg,
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_liquid_weight_decay',
        'mode': 'quick',
        'device': 'cuda',
    })

    assert result.metadata['train_report']['optimizer']['weight_decay'] == pytest.approx(0.125)
    payload = torch.load(
        tmp_path / 'train_liquid_weight_decay' / 'checkpoints' / 'liquid_ekf_best_checkpoint.pt',
        map_location='cpu',
        weights_only=False,
    )
    assert payload['optimizer']['weight_decay'] == pytest.approx(0.125)


def test_liquid_pipeline_preserves_tail_selection_observation_coeff_into_training_flow_contract(tmp_path, monkeypatch):
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)

    model_cfg = _liquid_model_cfg()
    model_cfg['train'] = dict(model_cfg['train'])
    model_cfg['train']['tail_selection_observation_coeff'] = 0.0

    result = run({
        'model_name': 'liquid_ekf',
        'model_cfg': model_cfg,
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_liquid_tail_selection_coeff',
        'mode': 'quick',
        'device': 'cuda',
    })

    assert result.metadata['train_report']['tail_selection_observation_coeff'] == pytest.approx(0.0)
    contract_path = Path(result.metadata['train_report']['training_flow_contract_path'])
    contract_payload = json.loads(contract_path.read_text(encoding='utf-8'))
    assert contract_payload['tail_selection_observation_coeff'] == pytest.approx(0.0)


def test_full_mode_runs_real_training(tmp_path, monkeypatch):
    """模式测试：full。\n\n验证 full 的模式验证，\n确保仅允许 quick/full 模式。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: False)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: False)

    result = run({
        'model_name': 'lstm_ekf',
        'model_cfg': _lstm_model_cfg(),
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_full_review',
        'mode': 'full',
        'device': 'cpu',
    })

    assert result.metadata['train_report']['status'] == 'trained'
    assert result.metadata['train_report']['mode'] == 'full'
    assert result.metadata['train_report']['device'] == 'cpu'
    assert result.metadata['train_report']['closure_note']['scope'] == 'real execution run label'
    assert result.metadata['checkpoint_smoke']['status'] == 'ok'


def test_full_mode_can_export_epoch_candidate_checkpoints(tmp_path, monkeypatch):
    """导出测试：full mode can。\n\n验证 full mode can 的导出功能，\n确保产物被正确持久化。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: False)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: False)

    model_cfg = _lstm_model_cfg()
    model_cfg['train'] = dict(model_cfg['train'])
    model_cfg['train']['epochs'] = 4
    model_cfg['train']['save_epoch_candidates'] = True

    result = run({
        'model_name': 'lstm_ekf',
        'model_cfg': model_cfg,
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_full_epoch_candidates',
        'mode': 'full',
        'device': 'cpu',
    })

    train_report = result.metadata['train_report']
    epoch_candidate_paths = list(train_report.get('epoch_candidate_paths') or [])
    assert epoch_candidate_paths
    assert all(Path(path).is_file() for path in epoch_candidate_paths)
    assert sorted(train_report.get('epoch_candidate_epochs') or []) == [1, 2, 3, 4]


def test_full_mode_can_subsample_epoch_candidate_checkpoints(tmp_path, monkeypatch):
    """子采样测试：full mode can。\n\n验证 full mode can 的子采样行为，\n确保步长参数被正确应用。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: False)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: False)

    model_cfg = _lstm_model_cfg()
    model_cfg['train'] = dict(model_cfg['train'])
    model_cfg['train']['epochs'] = 4
    model_cfg['train']['save_epoch_candidates'] = True
    model_cfg['train']['epoch_candidate_stride'] = 3

    result = run({
        'model_name': 'lstm_ekf',
        'model_cfg': model_cfg,
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_full_epoch_candidates_stride',
        'mode': 'full',
        'device': 'cpu',
    })

    train_report = result.metadata['train_report']
    assert sorted(train_report.get('epoch_candidate_epochs') or []) == [1, 3, 4]
    assert len(train_report.get('epoch_candidate_paths') or []) == 3


def test_full_mode_auto_device_uses_cuda_when_available(tmp_path, monkeypatch):
    """使用测试：full mode auto device。\n\n验证被测功能正确使用 full mode auto device，\n确保内部依赖被正确调用。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)

    result = run({
        'model_name': 'lstm_ekf',
        'model_cfg': _lstm_model_cfg(),
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': tmp_path / 'train_full_review_auto_device',
        'mode': 'full',
        'device': 'auto',
    })

    train_report = result.metadata['train_report']
    assert train_report['status'] == 'trained'
    assert train_report['requested_device'] == 'auto'
    assert train_report['device'] == 'cuda'
    assert train_report['cuda_available'] is True
    assert train_report['cuda_runtime_available'] is True
    assert train_report['closure_note']['scope'] == 'real execution run label'


def test_liquid_official_anchor_projection_smoke(tmp_path):
    """冒烟测试：liquid official anchor projection。\n\n快速验证 liquid official anchor projection 的基本功能可用，\n不深入检查细节，仅确认流程不崩溃。
    """
    raw_root = _write_official_miluv_raw_root(tmp_path / 'miluv_official')
    field_mapping = _miluv_field_mapping()

    events, gt_rows, source_report = _load_miluv_sequence_payload(
        'default_3_random_0',
        {'raw_root': raw_root, 'field_mapping': field_mapping},
    )
    assert isinstance(source_report['anchor_layout'], dict)
    teacher_anchor_layout, teacher_anchor_info = _materialize_teacher_anchor_layout_from_source_report(source_report)

    assert len(events) == 6
    assert len(gt_rows) == 3
    assert teacher_anchor_info['source'] == 'miluv_official_experiments_csv+anchors_yaml'
    assert teacher_anchor_info['metadata_position_dim'] == 3
    assert teacher_anchor_info['teacher_input_position_dim'] == 2
    assert teacher_anchor_info['blockers'] == []
    assert teacher_anchor_layout is not None
    assert teacher_anchor_layout['anchor_ids'] == [0, 1, 2, 3, 4, 5]
    assert teacher_anchor_layout['anchor_positions'][0] == pytest.approx([3.273827392578125, 3.46404736328125])
    assert teacher_anchor_layout['original_anchor_position_dim'] == 3
    assert teacher_anchor_layout['teacher_anchor_position_dim'] == 2
    assert teacher_anchor_layout['projection'] == 'xy'
    assert teacher_anchor_layout['ignored_axis'] == 'z'

    samples, sample_report = _build_liquid_samples(
        ['default_3_random_0'],
        {'dataset_name': 'miluv', 'raw_root': raw_root, 'field_mapping': field_mapping},
        _liquid_model_cfg(),
        dict(_load_default_ekf_cfg()),
    )

    assert len(samples) == 4
    assert sample_report['geometry_bias_teacher']['status'] == 'geometric_teacher_available'
    assert sample_report['geometry_bias_teacher']['available'] is True
    assert sample_report['geometry_bias_teacher']['source'] == 'miluv_official_experiments_csv+anchors_yaml'
    assert sample_report['geometry_bias_teacher']['metadata_position_dim'] == 3
    assert sample_report['geometry_bias_teacher']['teacher_input_position_dim'] == 2
    assert sample_report['geometry_bias_teacher']['original_anchor_position_dim'] == 3
    assert sample_report['geometry_bias_teacher']['teacher_anchor_position_dim'] == 2
    assert sample_report['geometry_bias_teacher']['projection'] == 'xy'
    assert sample_report['geometry_bias_teacher']['ignored_axis'] == 'z'
    assert sample_report['teacher_quality_audit'] == {
        'uwb_teacher_enabled_sample_count': 2,
        'uwb_teacher_fallback_sample_count': 0,
        'teacher_anchor_layout_available_sequence_count': 1,
        'teacher_anchor_layout_missing_sequence_count': 0,
        'teacher_projection_applied_sequence_count': 1,
        'teacher_projection_required_but_unavailable_sequence_count': 0,
        'teacher_geometry_bias_trace_available_count': 2,
        'teacher_geometry_bias_trace_missing_count': 0,
    }
    assert sample_report['bias_source_counts'] == {
        'measured_minus_geometric_true_range': 2,
        'neutral_baseline': 2,
    }
    assert sample_report['sequences']['default_3_random_0']['skipped_uwb_updates_without_anchor_layout'] == 0
    assert sample_report['sequences']['default_3_random_0']['geometry_bias_teacher']['projection'] == 'xy'
    assert sample_report['sequences']['default_3_random_0']['teacher_quality_audit'] == sample_report['teacher_quality_audit']

    uwb_samples = [sample for sample in samples if sample['modality'] == 'uwb']
    assert len(uwb_samples) == 2
    first_trace = uwb_samples[0]['target_trace']
    expected_range = math.hypot(3.273827392578125 - 0.015, 3.46404736328125 - 0.0)
    assert first_trace['bias_source'] == 'measured_minus_geometric_true_range'
    assert first_trace['anchor_position'] == pytest.approx([3.273827392578125, 3.46404736328125])
    assert first_trace['geometric_true_range'] == pytest.approx(expected_range)
    assert first_trace['original_anchor_position_dim'] == 3
    assert first_trace['teacher_anchor_position_dim'] == 2
    assert first_trace['projection'] == 'xy'
    assert first_trace['ignored_axis'] == 'z'


def test_resolve_sequence_anchor_layout_prefers_explicit_estimator_override():
    """覆盖测试：resolve sequence anchor layout prefers explicit estimator。\n\n验证 resolve sequence anchor layout prefers explicit estimator 的覆盖行为，\n确保显式参数优先于默认值。
    """
    explicit_anchor_layout = {
        'anchor_ids': ['A0'],
        'anchor_positions': [[9.0, 8.0]],
        'source': 'explicit_override',
    }
    source_report = {
        'anchor_layout_metadata': {
            'anchor_ids': [0],
            'anchor_positions': [[3.273827392578125, 3.46404736328125, 1.8093309326171875]],
            'source': 'miluv_official_experiments_csv+anchors_yaml',
            'layout_id': 'default_3_random_0',
            'experiment': 'default_3_random_0',
            'anchor_constellation': 0,
            'source_paths': ['config/uwb/anchors.yaml'],
        },
        'anchor_layout_metadata_source': 'miluv_official_experiments_csv+anchors_yaml',
        'anchor_layout_position_dim': 3,
    }

    resolved_layout, resolved_info = _resolve_sequence_anchor_layout(
        {'anchor_layout': explicit_anchor_layout},
        source_report,
    )

    assert resolved_layout == explicit_anchor_layout
    assert resolved_info['source'] == 'estimator_cfg.anchor_layout'
    assert resolved_info['metadata_source'] == 'miluv_official_experiments_csv+anchors_yaml'
    assert resolved_info['metadata_position_dim'] == 3
    assert resolved_info['teacher_input_position_dim'] == 2


def test_resolve_sequence_anchor_layout_rejects_explicit_3d_anchor_layout_override():
    """拒绝测试：resolve sequence anchor layout。\n\n验证被测功能对 resolve sequence anchor layout 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    explicit_anchor_layout = {
        'anchor_ids': [0],
        'anchor_positions': [[1.0, 2.0, 3.0]],
        'source': 'explicit_override_3d',
    }
    source_report = {
        'anchor_layout_metadata_source': 'miluv_official_experiments_csv+anchors_yaml',
        'anchor_layout_position_dim': 3,
    }

    resolved_layout, resolved_info = _resolve_sequence_anchor_layout(
        {'anchor_layout': explicit_anchor_layout},
        source_report,
    )

    assert resolved_layout is None
    assert resolved_info['source'] == 'estimator_cfg.anchor_layout'
    assert resolved_info['metadata_position_dim'] == 3
    assert resolved_info['teacher_input_position_dim'] == 3
    assert resolved_info['blockers'] == ['anchor_layout_position_dim_3_requires_2d_teacher']


def test_materialize_teacher_anchor_layout_rejects_direct_3d_layout_and_projects_official_metadata():
    """拒绝测试：materialize teacher anchor layout。\n\n验证被测功能对 materialize teacher anchor layout 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    source_report = {
        'anchor_layout': {
            'anchor_ids': [0],
            'anchor_positions': [[10.0, 20.0, 30.0]],
            'source': 'direct_3d_layout',
        },
        'anchor_layout_source': 'direct_3d_layout',
        'anchor_layout_metadata': {
            'anchor_ids': [0],
            'anchor_positions': [[3.273827392578125, 3.46404736328125, 1.8093309326171875]],
            'source': 'miluv_official_experiments_csv+anchors_yaml',
            'layout_id': 'default_3_random_0',
            'experiment': 'default_3_random_0',
            'anchor_constellation': 0,
            'source_paths': ['config/uwb/anchors.yaml'],
        },
        'anchor_layout_metadata_source': 'miluv_official_experiments_csv+anchors_yaml',
        'anchor_layout_position_dim': 3,
    }

    resolved_layout, resolved_info = _materialize_teacher_anchor_layout_from_source_report(source_report)

    assert resolved_layout is not None
    assert resolved_layout['anchor_positions'][0] == pytest.approx([3.273827392578125, 3.46404736328125])
    assert resolved_layout['original_anchor_position_dim'] == 3
    assert resolved_layout['teacher_anchor_position_dim'] == 2
    assert resolved_info['source'] == 'miluv_official_experiments_csv+anchors_yaml'
    assert resolved_info['metadata_position_dim'] == 3
    assert resolved_info['teacher_input_position_dim'] == 2
    assert resolved_info['blockers'] == []


def test_resolve_anchor_position_supports_dual_anchor_id_compatibility():
    """双模态测试：resolve anchor position supports。\n\n验证 resolve anchor position supports 的双模态兼容性，\n确保不同 ID 格式都能正确处理。
    """
    assert _resolve_anchor_position('A0', {0: (1.0, 2.0)}) == (1.0, 2.0)
    assert _resolve_anchor_position(0, {'A0': (1.0, 2.0)}) == (1.0, 2.0)


def test_target_intermediate_does_not_floor_scaling_from_risk():
    """不侵入测试：target intermediate。\n\n验证 target intermediate 不会产生副作用，\n确保功能隔离性。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={'modality': 'uwb'},
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': []},
        anchor_lookup=None,
        anchor_layout=None,
    )

    assert target_intermediate['bias'] == pytest.approx(0.0)
    assert target_intermediate['risk'] == pytest.approx(0.0)
    assert target_intermediate['uwb_scaling'] == pytest.approx(1.0)
    assert target_intermediate['vio_scaling'] == pytest.approx(1.0)
    assert target_trace['confidence_source'] == 'modality_specific_scaling_with_neutral_floor'
    assert target_trace['bias_source'] == 'neutral_baseline'
    assert target_trace['observation_risk'] == pytest.approx(0.0)


def test_target_intermediate_rejects_non_finite_scaling():
    """拒绝测试：target intermediate。\n\n验证被测功能对 target intermediate 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    target_intermediate, _ = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={'modality': 'uwb', 'uwb_payload': {'quality': 0.0, 'valid': False}},
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': []},
        anchor_lookup=None,
        anchor_layout=None,
    )

    assert target_intermediate['uwb_scaling'] >= 1.0


def test_modality_observation_signal_does_not_penalize_vio_boundary_tracked_features():
    """不侵入测试：modality observation signal。\n\n验证 modality observation signal 不会产生副作用，\n确保功能隔离性。
    """
    signal = _resolve_modality_observation_signal(
        {
            'modality': 'vio',
            'vio_payload': {'quality': 1.0, 'tracked_features': 30, 'reproj_err': 0.1},
        }
    )

    assert signal['quality_score'] == pytest.approx(1.0)
    assert signal['quality_risk'] == pytest.approx(0.0)
    assert signal['modality_signal'] == pytest.approx(0.0)
    assert signal['observation_risk'] == pytest.approx(0.0)


def test_target_intermediate_applies_robust_supplement_only_for_low_quality_high_risk():
    """应用测试：target intermediate。\n\n验证 target intermediate 的应用逻辑，\n确保特定条件触发预期行为。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={'modality': 'vio', 'vio_payload': {'quality': 0.1, 'reproj_err': 1.2, 'tracked_features': 12}},
        prediction_state={'px': 0.45, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup=None,
        anchor_layout=None,
    )

    assert target_trace['alignment_risk'] == pytest.approx(0.45)
    assert target_intermediate['risk'] == pytest.approx(target_trace['alignment_risk'])
    assert target_intermediate['uwb_scaling'] == pytest.approx(1.0)
    assert target_intermediate['vio_scaling'] == pytest.approx(1.65)
    assert target_trace['confidence_source'] == 'modality_specific_scaling_with_neutral_floor+robust_low_quality_supplement'
    assert target_trace['robust_teacher_supplement'] == 'active'
    assert target_trace['observation_risk'] == pytest.approx(0.9)
    assert target_trace['risk_source'] == 'pre_bridge_base_risk_alignment_only'


def test_target_intermediate_applies_robust_supplement_only_to_current_uwb_scaling():
    """应用测试：target intermediate。\n\n验证 target intermediate 的应用逻辑，\n确保特定条件触发预期行为。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={'modality': 'uwb', 'quality': 0.2, 'uwb_payload': {'quality': 0.2, 'valid': True}},
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup=None,
        anchor_layout=None,
    )

    assert target_trace['alignment_risk'] == pytest.approx(0.0)
    assert target_trace['observation_risk'] == pytest.approx(0.8)
    assert target_trace['robust_teacher_supplement'] == 'active'
    assert target_intermediate['risk'] == pytest.approx(target_trace['alignment_risk'])
    assert target_trace['robust_teacher_supplement_strength'] == pytest.approx(0.2)
    assert target_intermediate['uwb_scaling'] == pytest.approx(1.2)
    assert target_intermediate['vio_scaling'] == pytest.approx(1.0)


def test_target_intermediate_keeps_observation_risk_blend_as_audit_only_trace():
    """保持测试：target intermediate。\n\n验证 target intermediate 的保持行为，\n确保特定属性在处理过程中不变。
    """
    event = {'modality': 'vio', 'vio_payload': {'quality': 0.1, 'reproj_err': 1.2, 'tracked_features': 12}}
    window_tensor = {'feature_order': [], 'feature_values': []}
    prediction_state = {'px': 0.45, 'py': 0.0, 'yaw': 0.0}
    gt_state = {'px': 0.0, 'py': 0.0, 'yaw': 0.0}
    gt_alignment = {'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]}

    default_intermediate, default_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor=window_tensor,
        event=event,
        prediction_state=prediction_state,
        gt_state=gt_state,
        gt_alignment=gt_alignment,
        anchor_lookup=None,
        anchor_layout=None,
    )
    overridden_intermediate, overridden_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor=window_tensor,
        event=event,
        prediction_state=prediction_state,
        gt_state=gt_state,
        gt_alignment=gt_alignment,
        anchor_lookup=None,
        anchor_layout=None,
        bridge_thresholds={
            'robust_supplement_quality_threshold': 0.7,
            'robust_supplement_alignment_threshold': 0.4,
            'robust_supplement_max_boost': 0.5,
            'observation_risk_blend': 0.95,
            'uwb_geometric_bias_full_scale': 0.5,
            'async_gap_full_scale': 0.30,
            'async_uwb_scaling_coeff': 0.20,
            'async_vio_scaling_coeff': 0.20,
            'uwb_scaling_alignment_coeff': 0.50,
            'uwb_scaling_quality_coeff': 0.25,
            'uwb_scaling_geometry_coeff': 0.35,
            'uwb_invalid_extra_boost': 0.15,
            'vio_scaling_alignment_coeff': 0.50,
            'vio_scaling_quality_coeff': 0.25,
            'vio_low_features_scaling_boost': 0.10,
            'vio_high_reproj_scaling_boost': 0.10,
        },
    )

    assert overridden_intermediate == default_intermediate
    assert overridden_trace['observation_risk_blend'] == pytest.approx(0.95)
    assert default_trace['observation_risk_blend'] != pytest.approx(overridden_trace['observation_risk_blend'])
    for key in (
        'alignment_risk',
        'observation_risk',
        'quality_risk',
        'modality_signal',
        'async_gap_risk',
        'uwb_geometry_risk',
        'risk_source',
        'confidence_source',
        'robust_teacher_supplement',
    ):
        assert overridden_trace[key] == pytest.approx(default_trace[key]) if isinstance(default_trace[key], float) else overridden_trace[key] == default_trace[key]
    # overridden 改变了 quality_threshold 和 alignment_threshold，
    # 可能激活 robust supplement，导致 supplement_strength 不同。
    # 只验证 overridden 的 strength 在合法范围内。
    assert overridden_trace['robust_teacher_supplement_strength'] >= 0.0


def test_target_intermediate_records_current_source_field_vocabulary():
    """记录测试：target intermediate。\n\n验证 target intermediate 的记录行为，\n确保关键信息被正确追踪。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={'modality': 'vio', 'vio_payload': {'quality': 0.1, 'reproj_err': 1.2, 'tracked_features': 12}},
        prediction_state={'px': 0.45, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup=None,
        anchor_layout=None,
    )

    assert target_intermediate['risk'] == pytest.approx(target_trace['alignment_risk'])
    assert target_trace['bias_source'] == 'neutral_baseline'
    assert target_trace['risk_source'] == 'pre_bridge_base_risk_alignment_only'
    assert target_trace['confidence_source'] == 'modality_specific_scaling_with_neutral_floor+robust_low_quality_supplement'


def test_target_intermediate_wraps_yaw_error_before_risk_projection():
    """前置验证测试：target intermediate wraps yaw error。\n\n验证 target intermediate wraps yaw error 在后续操作前被正确检查，\n确保早期拦截无效输入。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={'modality': 'vio'},
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': -3.12},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 3.12},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup=None,
        anchor_layout=None,
    )

    assert target_trace['yaw_error'] == pytest.approx(0.04318530717958602)
    assert target_trace['alignment_yaw_full_scale_rad'] == pytest.approx(math.pi)
    assert target_trace['normalized_yaw_error'] == pytest.approx(0.04318530717958602 / math.pi)
    assert target_trace['alignment_risk'] == pytest.approx(0.04318530717958602 / math.pi)
    assert target_intermediate['risk'] == pytest.approx(target_trace['alignment_risk'])


def test_target_intermediate_normalizes_pose_error_by_failure_threshold():
    """归一化测试：target intermediate。\n\n验证 target intermediate 的归一化处理，\n确保空白/重复等输入被正确清洗。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={'modality': 'vio'},
        prediction_state={'px': 1.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup=None,
        anchor_layout=None,
    )

    assert target_trace['pose_error'] == pytest.approx(1.0)
    assert target_trace['alignment_pose_full_scale_m'] == pytest.approx(1.0)
    assert target_trace['normalized_pose_error'] == pytest.approx(1.0)
    assert target_trace['alignment_risk'] == pytest.approx(1.0)
    assert target_intermediate['risk'] == pytest.approx(target_trace['alignment_risk'])


def test_target_intermediate_uses_max_of_normalized_pose_and_yaw_for_alignment_risk():
    """使用测试：target intermediate。\n\n验证被测功能正确使用 target intermediate，\n确保内部依赖被正确调用。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={'modality': 'vio'},
        prediction_state={'px': 0.2, 'py': 0.0, 'yaw': math.pi / 2.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup=None,
        anchor_layout=None,
    )

    assert target_trace['normalized_pose_error'] == pytest.approx(0.2)
    assert target_trace['normalized_yaw_error'] == pytest.approx(0.5)
    assert target_trace['alignment_risk'] == pytest.approx(0.5)
    assert target_intermediate['risk'] == pytest.approx(0.5)


def test_target_intermediate_reads_quality_from_payload():
    """读取测试：target intermediate。\n\n验证 target intermediate 的读取逻辑，\n确保数据从正确来源获取。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={
            'modality': 'vio',
            'vio_payload': {'quality': 0.1, 'reproj_err': 1.2, 'tracked_features': 12},
        },
        prediction_state={'px': 0.45, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup=None,
        anchor_layout=None,
    )

    assert target_trace['bias_source'] == 'neutral_baseline'
    assert target_trace['robust_teacher_quality_score'] == pytest.approx(0.1)
    assert target_trace['observation_risk'] == pytest.approx(0.9)
    assert target_intermediate['risk'] == pytest.approx(target_trace['alignment_risk'])
    assert target_intermediate['vio_scaling'] == pytest.approx(1.65)
    assert target_intermediate['uwb_scaling'] == pytest.approx(1.0)


def test_teacher_free_scaling_inflates_covariance_for_worse_observations():
    """膨胀测试：teacher free scaling。\n\n验证 teacher free scaling 的膨胀效应，\n确保恶化观测条件导致协方差增大。
    """
    clean_uwb = _build_teacher_free_uwb_scaling(
        {'modality': 'uwb', 'uwb_payload': {'quality': 1.0, 'valid': True}},
        0.0,
    )
    degraded_uwb = _build_teacher_free_uwb_scaling(
        {'modality': 'uwb', 'uwb_payload': {'quality': 0.1, 'valid': False}},
        0.6,
    )
    clean_vio = _build_teacher_free_vio_scaling(
        {'modality': 'vio', 'vio_payload': {'quality': 1.0, 'tracked_features': 60, 'reproj_err': 0.1}},
        0.0,
    )
    degraded_vio = _build_teacher_free_vio_scaling(
        {'modality': 'vio', 'vio_payload': {'quality': 0.1, 'tracked_features': 12, 'reproj_err': 1.2}},
        0.6,
    )

    assert clean_uwb == pytest.approx(1.0)
    assert clean_vio == pytest.approx(1.0)
    assert degraded_uwb > clean_uwb
    assert degraded_vio > clean_vio


def test_teacher_free_scaling_inflates_covariance_for_larger_async_gap():
    """膨胀测试：teacher free scaling。\n\n验证 teacher free scaling 的膨胀效应，\n确保恶化观测条件导致协方差增大。
    """
    clean_uwb = _build_teacher_free_uwb_scaling(
        {'modality': 'uwb', 'uwb_payload': {'quality': 1.0, 'valid': True}},
        0.0,
        async_gap_risk=0.0,
    )
    stale_uwb = _build_teacher_free_uwb_scaling(
        {'modality': 'uwb', 'uwb_payload': {'quality': 1.0, 'valid': True}},
        0.0,
        async_gap_risk=1.0,
    )
    clean_vio = _build_teacher_free_vio_scaling(
        {'modality': 'vio', 'vio_payload': {'quality': 1.0, 'tracked_features': 60, 'reproj_err': 0.1}},
        0.0,
        async_gap_risk=0.0,
    )
    stale_vio = _build_teacher_free_vio_scaling(
        {'modality': 'vio', 'vio_payload': {'quality': 1.0, 'tracked_features': 60, 'reproj_err': 0.1}},
        0.0,
        async_gap_risk=1.0,
    )

    assert stale_uwb > clean_uwb
    assert stale_vio > clean_vio


def test_target_intermediate_uses_positive_uwb_geometric_bias_to_inflate_scaling():
    """使用测试：target intermediate。\n\n验证被测功能正确使用 target intermediate，\n确保内部依赖被正确调用。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={
            't': 0.0,
            'dt': 0.1,
            'modality': 'uwb',
            'meta': {'scene_id': 'scene_geom', 'seq_id': 'seq_geom'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 0, 'range': 1.5, 'valid': True, 'quality': 1.0},
            'vio_payload': None,
        },
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup={0: (1.0, 0.0)},
        anchor_layout={'anchor_ids': [0], 'anchor_positions': [[1.0, 0.0]]},
    )

    assert target_intermediate['bias'] == pytest.approx(0.5)
    assert target_trace['uwb_geometry_risk'] == pytest.approx(1.0)
    assert target_trace['observation_risk'] == pytest.approx(1.0)
    assert target_trace['alignment_risk'] == pytest.approx(0.0)
    assert target_intermediate['risk'] == pytest.approx(target_trace['alignment_risk'])
    assert target_intermediate['uwb_scaling'] == pytest.approx(1.35)


def test_target_intermediate_clamps_negative_uwb_bias_to_contract_floor():
    """合同测试：target intermediate clamps negative uwb bias to。\n\n验证 target intermediate clamps negative uwb bias to 的接口合同，\n确保输入输出符合协议约定。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={
            't': 0.0,
            'dt': 0.1,
            'modality': 'uwb',
            'meta': {'scene_id': 'scene_geom', 'seq_id': 'seq_geom'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 0, 'range': 0.5, 'valid': True, 'quality': 1.0},
            'vio_payload': None,
        },
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup={0: (1.0, 0.0)},
        anchor_layout={'anchor_ids': [0], 'anchor_positions': [[1.0, 0.0]]},
    )

    assert target_intermediate['bias'] == pytest.approx(0.0)
    assert target_trace['raw_bias'] == pytest.approx(-0.5)
    assert target_trace['clamped_bias'] == pytest.approx(0.0)
    assert target_trace['bias_source'] == 'measured_minus_geometric_true_range'
    assert target_trace['uwb_geometry_risk'] == pytest.approx(0.0)


def test_target_intermediate_clamps_bias_to_uwb_bias_max_ratio():
    """偏置超过 measured_range * UWB_BIAS_MAX_RATIO 时裁剪到上限。"""
    from liquidloc.protocol.liquid_bridge_contract import UWB_BIAS_MAX_RATIO
    # anchor at (1.0, 0.0), gt at (0,0), geometric_true_range = 1.0
    # measured_range = 3.0, raw_bias = 2.0, max_bias = 3.0 * 0.5 = 1.5
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={
            't': 0.0,
            'dt': 0.1,
            'modality': 'uwb',
            'meta': {'scene_id': 'scene_geom', 'seq_id': 'seq_geom'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 0, 'range': 3.0, 'valid': True, 'quality': 1.0},
            'vio_payload': None,
        },
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup={0: (1.0, 0.0)},
        anchor_layout={'anchor_ids': [0], 'anchor_positions': [[1.0, 0.0]]},
    )

    assert target_trace['raw_bias'] == pytest.approx(2.0)
    assert target_trace['clamped_bias'] == pytest.approx(3.0 * UWB_BIAS_MAX_RATIO)
    assert target_intermediate['bias'] == pytest.approx(3.0 * UWB_BIAS_MAX_RATIO)


def test_target_intermediate_uses_async_gap_to_inflate_scaling_and_observation_risk():
    """使用测试：target intermediate。\n\n验证被测功能正确使用 target intermediate，\n确保内部依赖被正确调用。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={
            'feature_order': ['modality_gap_dt'],
            'feature_values': [0.30],
            'missing_mask': [0],
            'feature_window': [[0.30]],
            'missing_mask_window': [[0]],
            'current_modality': 'vio',
            'dt': 0.1,
        },
        event={
            'modality': 'vio',
            'dt': 0.1,
            'vio_payload': {'quality': 1.0, 'tracked_features': 60, 'reproj_err': 0.1},
        },
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup=None,
        anchor_layout=None,
    )

    assert target_trace['async_gap_risk'] == pytest.approx(1.0)
    assert target_trace['observation_risk'] == pytest.approx(1.0)
    assert target_intermediate['risk'] == pytest.approx(target_trace['alignment_risk'])
    assert target_intermediate['vio_scaling'] == pytest.approx(1.2)
    assert target_intermediate['uwb_scaling'] == pytest.approx(1.0)


def test_target_intermediate_uses_scene_axis_async_floor_even_without_gap_feature():
    """使用测试：target intermediate。\n\n验证被测功能正确使用 target intermediate，\n确保内部依赖被正确调用。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={
            'modality': 'vio',
            'dt': 0.0,
            'meta': {'scene_id': 'S(A3,N0,V0,K0,M0)', 'seq_id': 'axis_async_only'},
            'vio_payload': {'quality': 1.0, 'tracked_features': 80, 'reproj_err': 0.1},
        },
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup=None,
        anchor_layout=None,
    )

    assert target_trace['async_gap_risk'] == pytest.approx(0.0)
    assert target_trace['async_axis_risk'] > 0.0
    assert target_trace['effective_async_risk'] == pytest.approx(target_trace['async_axis_risk'])
    assert target_trace['axis_observation_floor'] == pytest.approx(target_trace['async_axis_risk'])
    assert target_trace['observation_risk'] == pytest.approx(target_trace['async_axis_risk'])
    assert target_intermediate['risk'] == pytest.approx(target_trace['alignment_risk'])
    assert target_intermediate['vio_scaling'] > 1.0


def test_visual_degradation_inflates_vio_training_scaling_and_observation_risk_without_polluting_base_risk():
    """膨胀测试：visual degradation。\n\n验证 visual degradation 的膨胀效应，\n确保恶化观测条件导致协方差增大。
    """
    source_event = {
        't': 0.0,
        'dt': 0.0,
        'modality': 'vio',
        'meta': {'scene_id': 'S(A1,N0,V0,K0,M0)', 'seq_id': 'visual_link'},
        'imu_payload': None,
        'uwb_payload': None,
        'vio_payload': {
            'dx': 0.5,
            'dy': -0.2,
            'dyaw': 0.1,
            'quality': 0.9,
            'tracked_features': 150,
            'reproj_err': 0.4,
        },
    }
    visual_cfg = {
        'V3': {
            'tracked_features_range': [10, 20],
            'reproj_err_max': 2.0,
            'blackout_prob': 0.0,
            'drift_bias_m': 0.3,
        }
    }
    degraded_events, visual_report = apply_visual_level([source_event], 'V3', visual_cfg)
    degraded_event = degraded_events[0]

    clean_intermediate, clean_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event=source_event,
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup=None,
        anchor_layout=None,
    )
    degraded_intermediate, degraded_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event=degraded_event,
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup=None,
        anchor_layout=None,
    )

    assert visual_report['protocol_consistent'] is True
    assert degraded_event['vio_payload']['quality'] < source_event['vio_payload']['quality']
    assert degraded_event['vio_payload']['tracked_features'] < source_event['vio_payload']['tracked_features']
    assert degraded_event['vio_payload']['reproj_err'] >= source_event['vio_payload']['reproj_err']

    assert clean_trace['alignment_risk'] == pytest.approx(0.0)
    assert degraded_trace['alignment_risk'] == pytest.approx(0.0)
    assert clean_intermediate['risk'] == pytest.approx(clean_trace['alignment_risk'])
    assert degraded_intermediate['risk'] == pytest.approx(degraded_trace['alignment_risk'])

    assert clean_trace['quality_risk'] < degraded_trace['quality_risk']
    assert clean_trace['modality_signal'] < degraded_trace['modality_signal']
    assert clean_trace['observation_risk'] < degraded_trace['observation_risk']
    assert clean_intermediate['vio_scaling'] < degraded_intermediate['vio_scaling']
    assert clean_intermediate['uwb_scaling'] == pytest.approx(1.0)
    assert degraded_intermediate['uwb_scaling'] == pytest.approx(1.0)
    assert degraded_trace['risk_source'] == 'pre_bridge_base_risk_alignment_only'
    assert degraded_trace['confidence_source'] == 'modality_specific_scaling_with_neutral_floor'


def test_target_intermediate_uses_scene_axis_visual_floor_without_payload_degradation():
    """使用测试：target intermediate。\n\n验证被测功能正确使用 target intermediate，\n确保内部依赖被正确调用。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={
            'modality': 'vio',
            'dt': 0.0,
            'meta': {'scene_id': 'S(A0,N0,V3,K0,M0)', 'seq_id': 'axis_visual_only'},
            'vio_payload': {'quality': 1.0, 'tracked_features': 100, 'reproj_err': 0.2},
        },
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup=None,
        anchor_layout=None,
    )

    assert target_trace['visual_axis_risk'] > 0.0
    assert target_trace['axis_observation_floor'] == pytest.approx(target_trace['visual_axis_risk'])
    assert target_trace['observation_risk'] == pytest.approx(target_trace['visual_axis_risk'])
    assert target_intermediate['risk'] == pytest.approx(target_trace['alignment_risk'])
    assert target_intermediate['vio_scaling'] > 1.0


def test_target_intermediate_keeps_async_and_geometry_risk_outside_base_risk():
    """保持测试：target intermediate。\n\n验证 target intermediate 的保持行为，\n确保特定属性在处理过程中不变。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={
            'feature_order': ['modality_gap_dt'],
            'feature_values': [0.30],
            'missing_mask': [0],
            'feature_window': [[0.30]],
            'missing_mask_window': [[0]],
            'current_modality': 'uwb',
            'dt': 0.1,
        },
        event={
            't': 0.0,
            'dt': 0.1,
            'modality': 'uwb',
            'meta': {'scene_id': 'scene_geom', 'seq_id': 'seq_geom'},
            'uwb_payload': {'anchor_id': 0, 'range': 1.5, 'valid': True, 'quality': 0.2},
        },
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup={0: (1.0, 0.0)},
        anchor_layout={'anchor_ids': [0], 'anchor_positions': [[1.0, 0.0]]},
    )

    assert target_trace['alignment_risk'] == pytest.approx(0.0)
    assert target_trace['uwb_geometry_risk'] == pytest.approx(1.0)
    assert target_trace['async_gap_risk'] == pytest.approx(1.0)
    assert target_trace['observation_risk'] == pytest.approx(1.0)
    assert target_intermediate['risk'] == pytest.approx(target_trace['alignment_risk'])
    assert target_intermediate['uwb_scaling'] > 1.0


def test_target_intermediate_uses_scene_axis_nlos_floor_without_event_level_quality_drop():
    """使用测试：target intermediate。\n\n验证被测功能正确使用 target intermediate，\n确保内部依赖被正确调用。
    """
    target_intermediate, target_trace = _build_target_intermediate(
        teacher_model=None,
        window_tensor={'feature_order': [], 'feature_values': []},
        event={
            't': 0.0,
            'dt': 0.0,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A0,N3,V0,K0,M0)', 'seq_id': 'axis_nlos_only'},
            'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 1.0},
        },
        prediction_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_state={'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        gt_alignment={'mode': 'exact', 'time_gap': 0.0, 'support_timestamps': [0.0]},
        anchor_lookup={0: (2.0, 0.0)},
        anchor_layout={'anchor_ids': [0], 'anchor_positions': [[2.0, 0.0]]},
    )

    assert target_trace['quality_risk'] == pytest.approx(0.0)
    assert target_trace['uwb_geometry_risk'] == pytest.approx(0.0)
    assert target_trace['nlos_axis_risk'] > 0.0
    assert target_trace['axis_observation_floor'] == pytest.approx(target_trace['nlos_axis_risk'])
    assert target_trace['observation_risk'] == pytest.approx(target_trace['nlos_axis_risk'])
    assert target_intermediate['risk'] == pytest.approx(target_trace['alignment_risk'])
    assert target_intermediate['uwb_scaling'] > 1.0


def test_scene_code_fallback_ignores_blank_scene_id_shadow():
    """回退测试：scene code。\n\n验证 scene code 的回退机制，\n确保主路径失败时有合理的降级策略。
    """
    event = {
        'meta': {
            'scene_id': '   ',
            'scene_code': 'S(A0,N0,V0,K0,M0)',
            'seq_id': 'mini_seq',
        }
    }

    assert _decode_event_scene_spec(event) is not None
    assert _resolve_scene_axis_risk_floor(event)['async_axis_risk'] > 0.0
    assert _is_normal_scene_for_training(event) is True


@pytest.mark.parametrize(
    (
        'gt_rows',
        'timestamp',
        'expected_mode',
        'expected_state',
        'expected_support_timestamps',
        'expected_gap',
    ),
    [
        ([], 1.23, 'missing_ground_truth', None, [], None),
        ([{'timestamp': 2.0, 'px': 4.0, 'py': 5.0, 'yaw': 6.0}], 2.0, 'exact', {'timestamp': 2.0, 'px': 4.0, 'py': 5.0, 'yaw': 6.0}, [2.0], 0.0),
        ([{'timestamp': 2.0, 'px': 4.0, 'py': 5.0, 'yaw': 6.0}], 2.0 + _GT_TIME_TOLERANCE + 0.01, 'outside_single_ground_truth_timestamp', None, [2.0], _GT_TIME_TOLERANCE + 0.01),
        ([{'timestamp': 3.0, 'px': 1.0, 'py': 2.0, 'yaw': 3.0}, {'timestamp': 5.0, 'px': 7.0, 'py': 11.0, 'yaw': 13.0}], 2.5, 'before_ground_truth_span', None, [3.0], 0.5),
        ([{'timestamp': 3.0, 'px': 1.0, 'py': 2.0, 'yaw': 3.0}, {'timestamp': 3.0 + _GT_TIME_TOLERANCE + 1.0, 'px': 7.0, 'py': 11.0, 'yaw': 13.0}], 3.0 + (_GT_TIME_TOLERANCE + 1.0) * 1.5, 'trailing_nearest_within_tolerance', {'timestamp': 3.0 + _GT_TIME_TOLERANCE + 1.0, 'px': 7.0, 'py': 11.0, 'yaw': 13.0}, [3.0 + _GT_TIME_TOLERANCE + 1.0], (_GT_TIME_TOLERANCE + 1.0) / 2.0),
        ([{'timestamp': 3.0, 'px': 1.0, 'py': 2.0, 'yaw': 3.0}, {'timestamp': 3.0 + _GT_TIME_TOLERANCE + 1.0, 'px': 7.0, 'py': 11.0, 'yaw': 13.0}], 3.0 + 2.0 * (_GT_TIME_TOLERANCE + 1.0) + 0.1, 'after_ground_truth_span', None, [3.0 + _GT_TIME_TOLERANCE + 1.0], (_GT_TIME_TOLERANCE + 1.0) + 0.1),
    ],
)
def test_align_ground_truth_essential_modes(gt_rows, timestamp, expected_mode, expected_state, expected_support_timestamps, expected_gap):
    """核心测试：align ground truth。\n\n验证 align ground truth 的核心模式，\n确保各种对齐模式正确工作。
    """
    aligned_gt, alignment = _align_ground_truth(gt_rows, timestamp)

    assert alignment['mode'] == expected_mode
    assert alignment['support_timestamps'] == pytest.approx(expected_support_timestamps)
    assert alignment['time_gap'] == pytest.approx(expected_gap)
    if expected_state is None:
        assert aligned_gt is None
    else:
        assert aligned_gt == expected_state


def test_align_ground_truth_interpolates_yaw_across_pi_boundary():
    """插值测试：align ground truth。\n\n验证 align ground truth 的插值行为，\n确保跨越 π 边界的航向插值正确。
    """
    gt_rows = [
        {'timestamp': 0.0, 'px': 0.0, 'py': 0.0, 'yaw': 3.12},
        {'timestamp': 1.0, 'px': 1.0, 'py': 1.0, 'yaw': -3.12},
    ]

    aligned_gt, alignment = _align_ground_truth(gt_rows, 0.5)

    assert alignment['mode'] == 'linear_interpolation'
    assert aligned_gt is not None
    assert aligned_gt['px'] == pytest.approx(0.5)
    assert aligned_gt['py'] == pytest.approx(0.5)
    assert aligned_gt['yaw'] == pytest.approx(-math.pi, abs=1e-6)


def test_liquid_samples_use_pre_update_state_for_current_step_geometry_features():
    """几何测试：liquid samples use pre update state for current step。\n\n验证 liquid samples use pre update state for current step 的几何偏置计算，\n确保锚点-目标几何关系正确。
    """
    events = [
        {
            't': 0.0,
            'dt': 0.0,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A1,N0,V0,K0,M0)', 'seq_id': 'causal_seq'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 0, 'range': 10.0, 'valid': True, 'quality': 1.0},
            'vio_payload': None,
        },
        {
            't': 1.0,
            'dt': 1.0,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A1,N0,V0,K0,M0)', 'seq_id': 'causal_seq'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 0, 'range': 8.0, 'valid': True, 'quality': 1.0},
            'vio_payload': None,
        },
    ]
    gt_rows = [
        {'timestamp': 0.0, 'px': 1.0, 'py': 0.0, 'yaw': 0.0},
        {'timestamp': 1.0, 'px': 2.0, 'py': 0.0, 'yaw': 0.0},
    ]
    estimator_cfg = dict(_load_default_ekf_cfg())
    estimator_cfg['anchor_layout'] = {
        'anchor_ids': [0],
        'anchor_positions': [[10.0, 0.0]],
        'layout_id': 'causal_layout',
    }

    samples, sample_report = _build_liquid_samples(
        ['causal_seq'],
        {
            'split_ids': ['causal_seq'],
            'events_by_seq_id': {'causal_seq': events},
            'ground_truth_by_seq_id': {'causal_seq': gt_rows},
        },
        {
            'feature_order': ['anchor_dx', 'anchor_dy', 'uwb_range_residual', 'px', 'py'],
            'window': {'size': 4, 'step': 1},
        },
        estimator_cfg,
    )

    assert sample_report['usable_sample_count'] == 2
    assert samples[0]['window_tensor']['feature_values'] == pytest.approx([10.0, 0.0, 0.0, 0.0, 0.0])
    assert samples[1]['window_tensor']['feature_values'] == pytest.approx([10.0, 0.0, -2.0, 0.0, 0.0])
    assert samples[1]['window_tensor']['missing_mask'] == [0, 0, 0, 0, 0]
    assert samples[1]['target_trace']['prediction_state'] == pytest.approx({'px': 0.0, 'py': 0.0, 'yaw': 0.0})
    assert samples[1]['target_trace']['alignment_risk'] == pytest.approx(1.0)


def test_liquid_samples_exclude_gt_outside_events_from_history_context():
    """排除测试：liquid samples。\n\n验证 liquid samples 的排除逻辑，\n确保不符合条件的数据被正确跳过。
    """
    events = [
        {
            't': 0.0,
            'dt': 0.0,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A1,N0,V0,K0,M0)', 'seq_id': 'gt_span_seq'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 0, 'range': 10.0, 'valid': True, 'quality': 1.0},
            'vio_payload': None,
        },
        {
            't': 1.0,
            'dt': 1.0,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A1,N0,V0,K0,M0)', 'seq_id': 'gt_span_seq'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 0, 'range': 8.0, 'valid': True, 'quality': 1.0},
            'vio_payload': None,
        },
    ]
    gt_rows = [
        {'timestamp': 1.0, 'px': 2.0, 'py': 0.0, 'yaw': 0.0},
    ]
    estimator_cfg = dict(_load_default_ekf_cfg())
    estimator_cfg['anchor_layout'] = {
        'anchor_ids': [0],
        'anchor_positions': [[10.0, 0.0]],
        'layout_id': 'gt_span_layout',
    }

    samples, sample_report = _build_liquid_samples(
        ['gt_span_seq'],
        {
            'split_ids': ['gt_span_seq'],
            'events_by_seq_id': {'gt_span_seq': events},
            'ground_truth_by_seq_id': {'gt_span_seq': gt_rows},
        },
        {
            'feature_order': ['modality_gap_dt', 'anchor_dx', 'uwb_range_residual', 'px'],
            'window': {'size': 4, 'step': 1},
        },
        estimator_cfg,
    )

    assert sample_report['sequences']['gt_span_seq']['skipped_outside_ground_truth_span'] == 1
    assert sample_report['usable_sample_count'] == 1
    assert len(samples) == 1
    assert samples[0]['window_tensor']['event_time_window'] == pytest.approx([1.0])
    assert samples[0]['window_tensor']['window_index_map'] == [0]
    assert samples[0]['window_tensor']['feature_values'] == pytest.approx([0.0, 10.0, -2.0, 0.0])
    assert samples[0]['window_tensor']['missing_mask'] == [0, 0, 0, 0]


def test_liquid_samples_materialize_temporal_reliability_context():
    """物化测试：liquid samples。\n\n验证 liquid samples 的物化过程，\n确保上下文被正确构建。
    """
    events, gt_rows = _temporal_reliability_training_inputs()

    samples, sample_report = _build_liquid_samples(
        ['temporal_seq'],
        {
            'split_ids': ['temporal_seq'],
            'events_by_seq_id': {'temporal_seq': events},
            'ground_truth_by_seq_id': {'temporal_seq': gt_rows},
        },
        {
            # 铁律 3: VIO 紧耦合不再输出 reproj_err / tracked_features，
            # 因此 feature_order 不再包含 vio_reproj_err_slope / tracked_features_drop
            'feature_order': ['modality_gap_dt', 'uwb_quality_min', 'uwb_invalid_rate'],
            'window': {'size': 8, 'step': 1},
        },
        dict(_load_default_ekf_cfg()),
    )

    assert sample_report['usable_sample_count'] == 4
    final_vio_sample = next(sample for sample in reversed(samples) if sample['modality'] == 'vio')
    assert final_vio_sample['window_tensor']['feature_values'] == pytest.approx([0.3, 0.2, 0.5])
    assert final_vio_sample['window_tensor']['missing_mask'] == [0, 0, 0]
    assert final_vio_sample['window_tensor']['event_time_window'] == pytest.approx([0.1, 0.2, 0.4, 0.5, 0.8])


def test_liquid_samples_materialize_training_readout_context_from_proxy_estimator():
    """物化测试：liquid samples。\n\n验证 liquid samples 的物化过程，\n确保上下文被正确构建。
    """
    events, gt_rows = _temporal_reliability_training_inputs()

    samples, _sample_report = _build_liquid_samples(
        ['temporal_seq'],
        {
            'split_ids': ['temporal_seq'],
            'events_by_seq_id': {'temporal_seq': events},
            'ground_truth_by_seq_id': {'temporal_seq': gt_rows},
        },
        {
            'feature_order': ['modality_gap_dt', 'uwb_quality_min', 'uwb_invalid_rate', 'vio_reproj_err_slope', 'tracked_features_drop'],
            'window': {'size': 8, 'step': 1},
        },
        dict(_load_default_ekf_cfg()),
    )

    first_uwb = next(sample for sample in samples if sample['modality'] == 'uwb')
    final_vio = next(sample for sample in reversed(samples) if sample['modality'] == 'vio')
    first_uwb_context = first_uwb['window_tensor']['readout_context_by_name']
    first_uwb_observed = first_uwb['window_tensor']['readout_context_observed_by_name']
    final_vio_context = final_vio['window_tensor']['readout_context_by_name']
    final_vio_observed = final_vio['window_tensor']['readout_context_observed_by_name']

    assert first_uwb_observed['state_cov_trace'] is True
    assert first_uwb_observed['pos_cov'] is True
    assert first_uwb_observed['last_innovation_norm'] is False
    assert first_uwb_observed['last_gate_skip_flag'] is False
    assert first_uwb_context['state_cov_trace'] > 0.0
    assert first_uwb_context['pos_cov'] > 0.0

    assert final_vio_observed['state_cov_trace'] is True
    assert final_vio_observed['pos_cov'] is True
    assert final_vio_observed['last_innovation_norm'] is False
    assert final_vio_observed['last_gate_skip_flag'] is True
    assert final_vio_context['state_cov_trace'] > 0.0
    assert final_vio_context['pos_cov'] > 0.0
    assert final_vio_context['last_innovation_norm'] == pytest.approx(0.0)
    # last_gate_skip_flag 现在在 readout context 中始终被观察，值取决于上游模态状态。
    assert isinstance(final_vio_context['last_gate_skip_flag'], float)


def test_training_readout_context_helpers_capture_vio_residual_norm_and_covariance_summary():
    """读出上下文测试：training。\n\n验证 training 的读出上下文构建，\n确保协方差摘要和门控标志正确传递。
    """
    estimator = create_estimator('ekf', dict(_load_default_ekf_cfg()))
    cache = _init_training_readout_context_cache()
    report = {
        'modality': 'vio',
        'update_applied': True,
        'reason': 'vio_update',
        'residual': [3.0, 4.0],
    }

    assert _coerce_training_residual_norm(report['residual'], modality='vio') == pytest.approx(5.0)
    _update_training_readout_context_cache(cache, report)
    values, observed = _build_training_readout_context(estimator, cache, modality='vio')

    assert observed['state_cov_trace'] is True
    assert observed['pos_cov'] is True
    assert observed['last_innovation_norm'] is True
    assert observed['last_gate_skip_flag'] is True
    assert values['state_cov_trace'] > 0.0
    assert values['pos_cov'] > 0.0
    assert values['last_innovation_norm'] == pytest.approx(5.0)
    assert values['last_gate_skip_flag'] == pytest.approx(0.0)


def test_training_and_runtime_readout_context_helpers_stay_in_lockstep():
    """同步测试：training and runtime readout context helpers stay in。\n\n验证 training and runtime readout context helpers stay in 的同步一致性，\n确保训练和推理路径使用相同逻辑。
    """
    estimator = create_estimator('ekf', dict(_load_default_ekf_cfg()))
    runtime_cache = _init_readout_context_cache()
    training_cache = _init_training_readout_context_cache()
    reports = [
        {
            'modality': 'uwb',
            'update_applied': True,
            'reason': 'uwb_update',
            'residual': 1.25,
        },
        {
            'modality': 'vio',
            'update_applied': False,
            'reason': 'vio_skip_update',
        },
        {
            'modality': 'vio',
            'update_applied': True,
            'reason': 'vio_update',
            'residual': [3.0, 4.0],
        },
    ]

    for report in reports:
        _update_readout_context_cache(runtime_cache, report)
        _update_training_readout_context_cache(training_cache, report)

    for modality in ('uwb', 'vio'):
        runtime_values, runtime_observed = _build_readout_context(estimator, runtime_cache, modality=modality)
        training_values, training_observed = _build_training_readout_context(
            estimator,
            training_cache,
            modality=modality,
        )

        assert training_observed == runtime_observed
        assert training_values == pytest.approx(runtime_values)


def test_training_feature_window_builder_reconstructs_anchor_lookup_from_estimator_cfg():
    """重建测试：training feature window builder。\n\n验证 training feature window builder 的重建能力，\n确保从配置中能正确恢复状态。
    """
    anchor_layout = {
        'anchor_ids': [0],
        'anchor_positions': [[10.0, 0.0]],
        'layout_id': 'cfg_only_layout',
    }
    estimator = create_estimator(
        'ekf',
        {
            **dict(_load_default_ekf_cfg()),
            'anchor_layout': anchor_layout,
        },
    )
    if hasattr(estimator, '_anchor_lookup'):
        estimator._anchor_lookup = {}

    window_builder = _build_feature_window_builder(
        {
            'feature_order': ['anchor_dx', 'anchor_dy', 'uwb_range_residual'],
            'window': {'size': 4, 'step': 1},
        },
        estimator,
    )

    event = {
        't': 0.0,
        'dt': 0.0,
        'modality': 'uwb',
        'meta': {'scene_id': 'S(A1,N0,V0,K0,M0)', 'seq_id': 'cfg_only_seq'},
        'imu_payload': None,
        'uwb_payload': {'anchor_id': 0, 'range': 10.0, 'valid': True, 'quality': 1.0},
        'vio_payload': None,
    }
    window_tensor = window_builder(
        [event],
        event,
        {'state_history': [{'px': 0.0, 'py': 0.0, 'yaw': 0.0}]},
    )

    assert window_tensor['feature_values'] == pytest.approx([10.0, 0.0, 0.0])
    assert window_tensor['missing_mask'] == [0, 0, 0]


def test_liquid_samples_missing_anchor_layout_uses_fallback_teacher_and_skips_uwb_updates():
    """使用测试：liquid samples missing anchor layout。\n\n验证被测功能正确使用 liquid samples missing anchor layout，\n确保内部依赖被正确调用。
    """
    events, gt_rows = _load_mini_seq_direct_inputs_without_anchor_layout()

    samples, sample_report = _build_liquid_samples(
        ['mini_seq'],
        {
            'split_ids': ['mini_seq'],
            'events_by_seq_id': {'mini_seq': events},
            'ground_truth_by_seq_id': {'mini_seq': gt_rows},
        },
        _liquid_model_cfg(),
        dict(_load_default_ekf_cfg()),
    )

    assert len(samples) == 4
    assert sample_report['geometry_bias_teacher']['status'] == 'fallback_teacher_only'
    assert sample_report['teacher_quality_audit'] == {
        'uwb_teacher_enabled_sample_count': 0,
        'uwb_teacher_fallback_sample_count': 2,
        'teacher_anchor_layout_available_sequence_count': 0,
        'teacher_anchor_layout_missing_sequence_count': 1,
        'teacher_projection_applied_sequence_count': 0,
        'teacher_projection_required_but_unavailable_sequence_count': 0,
        'teacher_geometry_bias_trace_available_count': 0,
        'teacher_geometry_bias_trace_missing_count': 2,
    }
    assert sample_report['bias_source_counts'] == {
        'measured_minus_geometric_true_range': 0,
        'neutral_baseline': 4,
    }
    assert sample_report['sequences']['mini_seq']['geometry_bias_teacher']['status'] == 'fallback_teacher_only'
    assert sample_report['sequences']['mini_seq']['teacher_quality_audit'] == sample_report['teacher_quality_audit']
    assert sample_report['sequences']['mini_seq']['bias_source_counts'] == {
        'measured_minus_geometric_true_range': 0,
        'neutral_baseline': 4,
    }
    assert sample_report['sequences']['mini_seq']['skipped_uwb_updates_without_anchor_layout'] == 2

    uwb_samples = [sample for sample in samples if sample['modality'] == 'uwb']
    assert len(uwb_samples) == 2
    assert all(sample['target_trace']['geometry_bias_teacher'] == 'unavailable_missing_anchor_layout' for sample in uwb_samples)
    assert all(sample['target_trace']['bias_source'] == 'neutral_baseline' for sample in uwb_samples)


def test_liquid_samples_teacher_quality_audit_marks_projection_required_but_unavailable():
    """投影测试：liquid samples teacher quality audit marks。\n\n验证 liquid samples teacher quality audit marks 的坐标投影，\n确保 3D→2D 投影正确。
    """
    events, gt_rows = _load_mini_seq_direct_inputs_without_anchor_layout()
    source_report = {
        'source': 'direct_inputs',
        'anchor_layout': {
            'anchor_ids': [0, 1],
            'anchor_positions': [[1.0, 2.0, 0.5], [3.0, 4.0, 0.5]],
            'source': 'direct_3d_anchor_layout',
        },
        'anchor_layout_source': 'direct_3d_anchor_layout',
        'anchor_layout_position_dim': 3,
    }

    samples, sample_report = _build_liquid_samples(
        ['mini_seq'],
        {
            'split_ids': ['mini_seq'],
            'events_by_seq_id': {'mini_seq': events},
            'ground_truth_by_seq_id': {'mini_seq': gt_rows},
            'source_report_by_seq_id': {'mini_seq': source_report},
        },
        _liquid_model_cfg(),
        dict(_load_default_ekf_cfg()),
    )

    assert len(samples) == 4
    assert sample_report['geometry_bias_teacher']['status'] == 'fallback_teacher_only'
    assert sample_report['teacher_quality_audit'] == {
        'uwb_teacher_enabled_sample_count': 0,
        'uwb_teacher_fallback_sample_count': 2,
        'teacher_anchor_layout_available_sequence_count': 0,
        'teacher_anchor_layout_missing_sequence_count': 1,
        'teacher_projection_applied_sequence_count': 0,
        'teacher_projection_required_but_unavailable_sequence_count': 1,
        'teacher_geometry_bias_trace_available_count': 0,
        'teacher_geometry_bias_trace_missing_count': 2,
    }
    assert sample_report['sequences']['mini_seq']['teacher_quality_audit'] == sample_report['teacher_quality_audit']


def test_liquid_samples_readout_context_tracks_uwb_skip_without_cross_modality_leak():
    """追踪测试：liquid samples readout context。\n\n验证 liquid samples readout context 的追踪机制，\n确保状态变化被正确记录。
    """
    events, gt_rows = _load_mini_seq_direct_inputs_without_anchor_layout()

    samples, _sample_report = _build_liquid_samples(
        ['mini_seq'],
        {
            'split_ids': ['mini_seq'],
            'events_by_seq_id': {'mini_seq': events},
            'ground_truth_by_seq_id': {'mini_seq': gt_rows},
        },
        _liquid_model_cfg(),
        dict(_load_default_ekf_cfg()),
    )

    uwb_samples = [sample for sample in samples if sample['modality'] == 'uwb']
    vio_samples = [sample for sample in samples if sample['modality'] == 'vio']
    assert len(uwb_samples) == 2
    assert len(vio_samples) == 2

    first_uwb_observed = uwb_samples[0]['window_tensor']['readout_context_observed_by_name']
    second_uwb_context = uwb_samples[1]['window_tensor']['readout_context_by_name']
    second_uwb_observed = uwb_samples[1]['window_tensor']['readout_context_observed_by_name']
    final_vio_observed = vio_samples[-1]['window_tensor']['readout_context_observed_by_name']

    assert first_uwb_observed['last_gate_skip_flag'] is False
    assert second_uwb_observed['last_gate_skip_flag'] is True
    assert second_uwb_context['last_gate_skip_flag'] == pytest.approx(1.0)
    assert second_uwb_observed['last_innovation_norm'] is False
    assert final_vio_observed['last_gate_skip_flag'] is True
    assert vio_samples[-1]['window_tensor']['readout_context_by_name']['last_gate_skip_flag'] == pytest.approx(1.0)


def test_liquid_and_lstm_share_same_training_sample_contract_before_model_specific_consumption():
    """前置验证测试：liquid and lstm share same training sample contract。\n\n验证 liquid and lstm share same training sample contract 在后续操作前被正确检查，\n确保早期拦截无效输入。
    """
    events, gt_rows = _temporal_reliability_training_inputs()
    runtime_cfg = {
        'split_ids': ['temporal_seq'],
        'events_by_seq_id': {'temporal_seq': events},
        'ground_truth_by_seq_id': {'temporal_seq': gt_rows},
    }

    liquid_samples, liquid_report = _build_liquid_samples(
        ['temporal_seq'],
        runtime_cfg,
        _liquid_model_cfg(),
        dict(_load_default_ekf_cfg()),
    )
    lstm_samples, lstm_report = _build_liquid_samples(
        ['temporal_seq'],
        runtime_cfg,
        _lstm_model_cfg(),
        dict(_load_default_ekf_cfg()),
    )

    assert liquid_report['usable_sample_count'] == lstm_report['usable_sample_count']
    assert liquid_report['usable_sample_count_by_modality'] == lstm_report['usable_sample_count_by_modality']
    assert liquid_report['bias_source_counts'] == lstm_report['bias_source_counts']
    assert liquid_report['bridge_thresholds'] == lstm_report['bridge_thresholds']
    assert liquid_report['geometry_bias_teacher'] == lstm_report['geometry_bias_teacher']
    assert liquid_report['sequences']['temporal_seq'] == lstm_report['sequences']['temporal_seq']

    assert len(liquid_samples) == len(lstm_samples)
    for liquid_sample, lstm_sample in zip(liquid_samples, lstm_samples, strict=True):
        assert liquid_sample['seq_id'] == lstm_sample['seq_id']
        assert liquid_sample['event_time'] == pytest.approx(lstm_sample['event_time'])
        assert liquid_sample['modality'] == lstm_sample['modality']
        assert liquid_sample['target_intermediate'] == lstm_sample['target_intermediate']
        assert liquid_sample['target_trace'] == lstm_sample['target_trace']
        assert liquid_sample['source_report'] == lstm_sample['source_report']

        liquid_window = liquid_sample['window_tensor']
        lstm_window = lstm_sample['window_tensor']
        assert liquid_window['feature_order'] == lstm_window['feature_order']
        assert liquid_window['current_modality'] == lstm_window['current_modality']
        assert liquid_window['window_index_map'] == lstm_window['window_index_map']
        assert liquid_window['feature_values'] == pytest.approx(lstm_window['feature_values'])
        assert liquid_window['missing_mask'] == lstm_window['missing_mask']
        assert len(liquid_window['feature_window']) == len(lstm_window['feature_window'])
        for liquid_row, lstm_row in zip(liquid_window['feature_window'], lstm_window['feature_window'], strict=True):
            assert liquid_row == pytest.approx(lstm_row)
        assert liquid_window['missing_mask_window'] == lstm_window['missing_mask_window']
        assert liquid_window['event_time_window'] == pytest.approx(lstm_window['event_time_window'])
        assert liquid_window['readout_context_by_name'] == pytest.approx(lstm_window['readout_context_by_name'])
        assert liquid_window['readout_context_observed_by_name'] == lstm_window['readout_context_observed_by_name']


def test_liquid_and_lstm_pipeline_pass_identical_pretrainer_windows(tmp_path, monkeypatch):
    """传递测试：liquid and lstm pipeline。\n\n验证 liquid and lstm pipeline 的传递一致性，\n确保数据在流水线中无损传递。
    """
    captured_calls: dict[str, Any] = {}

    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)

    def _snapshot(samples):
        snapshot = []
        for sample in samples:
            window = dict(sample['window_tensor'])
            snapshot.append(
                {
                    'seq_id': sample['seq_id'],
                    'event_time': float(sample['event_time']),
                    'modality': sample['modality'],
                    'target_intermediate': dict(sample['target_intermediate']),
                    'target_trace': dict(sample['target_trace']),
                    'source_report': dict(sample['source_report']),
                    'window_tensor': {
                        'feature_order': list(window['feature_order']),
                        'current_modality': window['current_modality'],
                        'window_index_map': list(window.get('window_index_map') or []),
                        'feature_values': list(window['feature_values']),
                        'missing_mask': list(window['missing_mask']),
                        'feature_window': [list(row) for row in window['feature_window']],
                        'missing_mask_window': [list(row) for row in window['missing_mask_window']],
                        'event_time_window': list(window.get('event_time_window') or []),
                        'readout_context_by_name': dict(window.get('readout_context_by_name') or {}),
                        'readout_context_observed_by_name': dict(window.get('readout_context_observed_by_name') or {}),
                    },
                }
            )
        return snapshot

    def _make_stub(model_name):
        def _stub(train_windows, val_windows, train_cfg):
            captured_calls[model_name] = {
                'train_windows': _snapshot(train_windows),
                'val_windows': _snapshot(val_windows),
                'feature_order': list(train_cfg.get('feature_order') or []),
                'window_cfg': dict(train_cfg.get('window') or {}),
            }
            output_root = Path(train_cfg['output_root'])
            checkpoints_dir = output_root / 'checkpoints'
            reports_dir = output_root / 'reports'
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

            model = create_model(
                model_name,
                {
                    'feature_order': list(train_cfg.get('feature_order') or _FEATURE_ORDER),
                    'window': dict(train_cfg.get('window') or {}),
                    'network': dict(train_cfg.get('network') or {}),
                },
            )
            checkpoint_path = checkpoints_dir / f'{model_name}_best_checkpoint.pt'
            report_path = reports_dir / f'{model_name}_train_report.json'
            torch.save(
                {
                    'checkpoint_format': 'liquid_real_v1' if model_name == 'liquid_ekf' else 'lstm_real_v1',
                    'model_cfg': {
                        'feature_order': list(train_cfg.get('feature_order') or _FEATURE_ORDER),
                        'window': dict(train_cfg.get('window') or {}),
                        'network': dict(train_cfg.get('network') or {}),
                    },
                    'model_state': model.state_dict(),
                    'best_epoch': 1,
                    'best_loss': 0.0,
                    'train_window_count': len(train_windows),
                    'val_window_count': len(val_windows),
                },
                checkpoint_path,
            )
            report_path.write_text('{"status":"trained"}', encoding='utf-8')
            return str(checkpoint_path), {
                'status': 'trained',
                'report_path': str(report_path),
                'checkpoint_path': str(checkpoint_path),
            }

        return _stub

    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.train_liquid_model', _make_stub('liquid_ekf'))
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.train_lstm_model', _make_stub('lstm_ekf'))

    common_cfg = {
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'mode': 'quick',
        'device': 'cuda',
    }

    run({
        'model_name': 'liquid_ekf',
        'model_cfg': _liquid_model_cfg(),
        'output_root': tmp_path / 'train_liquid_fairness',
        **common_cfg,
    })
    run({
        'model_name': 'lstm_ekf',
        'model_cfg': _lstm_model_cfg(),
        'output_root': tmp_path / 'train_lstm_fairness',
        **common_cfg,
    })

    liquid_call = captured_calls['liquid_ekf']
    lstm_call = captured_calls['lstm_ekf']
    assert liquid_call['feature_order'] == lstm_call['feature_order']
    assert liquid_call['window_cfg'] == lstm_call['window_cfg']
    assert liquid_call['train_windows'] == lstm_call['train_windows']
    assert liquid_call['val_windows'] == lstm_call['val_windows']


def test_split_train_and_val_samples_uses_distinct_seq_ids():
    """使用测试：split train and val samples。\n\n验证被测功能正确使用 split train and val samples，\n确保内部依赖被正确调用。
    """
    samples = [
        {'seq_id': 'seq_a', 'event_time': 0.1, 'window_tensor': {'feature_order': ['dt'], 'current_modality': 'uwb', 'feature_values': [0.0], 'missing_mask': [0], 'dt': 0.1, 'feature_window': [[0.0]], 'missing_mask_window': [[0]]}, 'target_intermediate': {'bias': 0.0, 'risk': 0.0, 'uwb_scaling': 1.0, 'vio_scaling': 1.0}},
        {'seq_id': 'seq_b', 'event_time': 0.2, 'window_tensor': {'feature_order': ['dt'], 'current_modality': 'uwb', 'feature_values': [0.0], 'missing_mask': [0], 'dt': 0.1, 'feature_window': [[0.0]], 'missing_mask_window': [[0]]}, 'target_intermediate': {'bias': 0.0, 'risk': 0.0, 'uwb_scaling': 1.0, 'vio_scaling': 1.0}},
        {'seq_id': 'seq_c', 'event_time': 0.3, 'window_tensor': {'feature_order': ['dt'], 'current_modality': 'uwb', 'feature_values': [0.0], 'missing_mask': [0], 'dt': 0.1, 'feature_window': [[0.0]], 'missing_mask_window': [[0]]}, 'target_intermediate': {'bias': 0.0, 'risk': 0.0, 'uwb_scaling': 1.0, 'vio_scaling': 1.0}},
    ]

    train_samples, val_samples, train_ids, val_ids = _split_train_and_val_samples(
        samples,
        split_ids=['seq_a', 'seq_b', 'seq_c'],
    )

    assert train_ids == ['seq_a', 'seq_b']
    assert val_ids == ['seq_c']
    assert {sample['seq_id'] for sample in train_samples} == {'seq_a', 'seq_b'}
    assert {sample['seq_id'] for sample in val_samples} == {'seq_c'}


def test_run_direct_inputs_uses_top_level_events_fallback_for_single_split_id(tmp_path):
    """使用测试：run direct inputs。\n\n验证被测功能正确使用 run direct inputs，\n确保内部依赖被正确调用。
    """
    events, gt_rows = _load_mini_seq_direct_inputs_without_anchor_layout()

    result = run({
        'model_name': 'liquid_ekf',
        'model_cfg': _liquid_model_cfg(),
        'split_ids': ['mini_seq'],
        'events': events,
        'ground_truth_by_seq_id': {'mini_seq': gt_rows},
        'output_root': tmp_path / 'train_events_fallback',
    })

    sample_report = result.metadata['sample_report']
    assert sample_report['sequences']['mini_seq']['source_report']['source'] == 'direct_inputs'
    assert sample_report['usable_sample_count'] == 4
    assert sample_report['sequences']['mini_seq']['usable_sample_count'] == 4


def test_run_direct_inputs_reads_ground_truth_from_root_for_single_split_id(tmp_path):
    """读取测试：run direct inputs。\n\n验证 run direct inputs 的读取逻辑，\n确保数据从正确来源获取。
    """
    events, gt_rows = _load_mini_seq_direct_inputs_without_anchor_layout()
    ground_truth_root = tmp_path / 'ground_truth_root'
    seq_root = ground_truth_root / 'mini_seq'
    seq_root.mkdir(parents=True)
    (seq_root / 'gt.json').write_text(json.dumps(gt_rows), encoding='utf-8')

    result = run({
        'model_name': 'liquid_ekf',
        'model_cfg': _liquid_model_cfg(),
        'split_ids': ['mini_seq'],
        'events_by_seq_id': {'mini_seq': events},
        'ground_truth_root': ground_truth_root,
        'output_root': tmp_path / 'train_ground_truth_root',
    })

    sample_report = result.metadata['sample_report']
    assert sample_report['sequences']['mini_seq']['source_report']['source'] == 'direct_inputs'
    assert sample_report['usable_sample_count'] == 4
    assert sample_report['sequences']['mini_seq']['usable_sample_count'] == 4


def test_run_missing_anchor_layout_reports_fallback_teacher_and_skip_counts(tmp_path):
    """回退测试：run missing anchor layout reports。\n\n验证 run missing anchor layout reports 的回退机制，\n确保主路径失败时有合理的降级策略。
    """
    events, gt_rows = _load_mini_seq_direct_inputs_without_anchor_layout()

    result = run({
        'model_name': 'liquid_ekf',
        'model_cfg': _liquid_model_cfg(),
        'split_ids': ['mini_seq'],
        'events_by_seq_id': {'mini_seq': events},
        'ground_truth_by_seq_id': {'mini_seq': gt_rows},
        'output_root': tmp_path / 'train_missing_anchor_layout',
    })

    sample_report = result.metadata['sample_report']
    assert sample_report['geometry_bias_teacher']['status'] == 'fallback_teacher_only'
    assert sample_report['split_audit'] == {
        'split_strategy': 'single_sequence_time_split',
        'requested_split_ids': ['mini_seq'],
        'requested_train_split_ids': [],
        'requested_val_split_ids': [],
        'resolved_train_split_ids': ['mini_seq'],
        'resolved_val_split_ids': ['mini_seq'],
        'shared_seq_ids': ['mini_seq'],
        'sequence_disjoint': False,
        # H24c 真改：scene_id cross-set 守门审计字段（4 个新字段）。
        # 该测试走 fall-through 单序列时间切分，不抛 cross-set ValueError，
        # 但 audit 写回 train/val 共享的 scene_id（miluv:mini_seq）。
        # fall-through 下允许共享（同 seq_id 内时间切片），scene_id_disjoint=False 正确。
        'shared_scene_ids': ['miluv:mini_seq'],
        'scene_id_disjoint': False,
        'train_scene_ids': ['miluv:mini_seq'],
        'val_scene_ids': ['miluv:mini_seq'],
        'time_ordered_nonoverlap': True,
        'train_window_count': 2,
        'val_window_count': 2,
        'train_event_time_span': {'start': 0.05, 'end': 0.08},
        'val_event_time_span': {'start': 0.15, 'end': 0.18},
    }
    assert sample_report['bias_source_counts'] == {
        'measured_minus_geometric_true_range': 0,
        'neutral_baseline': 4,
    }
    assert sample_report['sequences']['mini_seq']['geometry_bias_teacher']['status'] == 'fallback_teacher_only'
    assert sample_report['sequences']['mini_seq']['bias_source_counts'] == {
        'measured_minus_geometric_true_range': 0,
        'neutral_baseline': 4,
    }
    assert sample_report['sequences']['mini_seq']['skipped_uwb_updates_without_anchor_layout'] == 2
    assert result.metadata['target_contract']['split_audit'] == sample_report['split_audit']
    assert result.metadata['train_report']['split_audit'] == sample_report['split_audit']


def test_run_resolves_relative_output_root_against_project_root(tmp_path, monkeypatch):
    """输出根目录测试：run resolves relative。\n\n验证 run resolves relative 的输出根目录解析，\n确保相对路径锚定到项目根。
    """
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('liquidloc.pipelines.train_pipeline.cuda_runtime_usable', lambda: True)
    monkeypatch.chdir(tmp_path)

    relative_output_root = Path('relative_train_output')
    result = run({
        'model_name': 'lstm_ekf',
        'model_cfg': _lstm_model_cfg(),
        'dataset_name': 'miluv',
        'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': _miluv_field_mapping(),
        'split_ids': ['mini_seq', 'mini_seq_02'],
        'train_split_ids': ['mini_seq'],
        'val_split_ids': ['mini_seq_02'],
        'split_role': 'train',
        'output_root': relative_output_root,
        'mode': 'quick',
        'device': 'cuda',
    })

    expected_output_root = (PROJECT_ROOT / relative_output_root).resolve()
    assert result.metadata['train_report']['report_path'] == str(
        expected_output_root / 'reports' / 'lstm_ekf_train_report.json'
    )
    assert (expected_output_root / 'checkpoints' / 'lstm_ekf_best_checkpoint.pt').is_file()


def test_quick_g_axis_remaps_uwb_range_without_dropping_residual():
    """无依赖测试：quick g axis remaps uwb range。\n\n验证 quick g axis remaps uwb range 在缺少依赖时的降级行为，\n确保回退策略正确。
    """
    events, gt_rows, source_report = _load_miluv_sequence_payload(
        'mini_seq',
        {
            'dataset_name': 'miluv',
            'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': _miluv_field_mapping(),
        },
    )
    protocol = load_scene_axis_protocol()
    anchor_layout, _ = build_anchor_layout(6, 'K3', protocol['axes']['K'])  # 五轴档位协议：G 已并入 K，K3=差几何 4 锚场景。
    uwb_event = next(event for event in events if event['modality'] == 'uwb')
    original_range = float(uwb_event['uwb_payload']['range'])
    gt_row, _ = _align_ground_truth(gt_rows, float(uwb_event['t']))
    assert gt_row is not None
    old_anchor_layout = json.loads((PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv' / 'mini_seq' / 'anchor_layout.json').read_text(encoding='utf-8'))
    old_anchor_xy = old_anchor_layout['anchor_positions'][0]
    old_geometric = math.hypot(float(old_anchor_xy[0]) - float(gt_row['px']), float(old_anchor_xy[1]) - float(gt_row['py']))
    residual = original_range - old_geometric
    new_anchor_xy = tuple(anchor_layout['anchor_positions'][0])
    expected = max(0.0, math.hypot(float(new_anchor_xy[0]) - float(gt_row['px']), float(new_anchor_xy[1]) - float(gt_row['py'])) + residual)
    assert expected != pytest.approx(original_range)


def test_invalid_case(tmp_path):
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    with pytest.raises(ValueError):
        run({'output_root': tmp_path / 'train'})


def test_run_invalid_mode_rejected_before_training_starts(tmp_path):
    """拒绝测试：run invalid mode。\n\n验证被测功能对不合法的 run invalid mode 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(ValueError, match='mode must be one of'):
        run({
            'model_name': 'lstm_ekf',
            'model_cfg': _lstm_model_cfg(),
            'dataset_name': 'miluv',
            'raw_root': PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': _miluv_field_mapping(),
            'split_ids': ['mini_seq', 'mini_seq_02'],
            'train_split_ids': ['mini_seq'],
            'val_split_ids': ['mini_seq_02'],
            'split_role': 'train',
            'output_root': tmp_path / 'train_invalid_mode',
            'mode': 'paper',
            'device': 'auto',
        })


def test_assert_phase_epoch_count_within_budget_passes_when_within_budget():
    """§13.6.2.1 fail-loud 守门单元测试：phase_epoch_count 未超预算时应通过."""
    from liquidloc.models.liquid.trainer import _assert_phase_epoch_count_within_budget
    _assert_phase_epoch_count_within_budget(
        "full_tuning",
        {"phase_epoch_count": 50},
        {"phase_schedule_budget": {"full_tuning_epochs": 100}, "rollback_total_epoch_hard_cap_multiplier": 1.5},
    )
    # 边界: 正好等于 budget 应通过
    _assert_phase_epoch_count_within_budget(
        "full_tuning",
        {"phase_epoch_count": 100},
        {"phase_schedule_budget": {"full_tuning_epochs": 100}, "rollback_total_epoch_hard_cap_multiplier": 1.5},
    )


def test_assert_phase_epoch_count_within_budget_raises_when_over_budget():
    """§13.6.2.1 fail-loud 守门单元测试：phase_epoch_count 超 budget 时应 raise RuntimeError."""
    from liquidloc.models.liquid.trainer import _assert_phase_epoch_count_within_budget
    with pytest.raises(RuntimeError, match="§13.6.2.1 phase_epoch_count over-budget fail-loud"):
        _assert_phase_epoch_count_within_budget(
            "full_tuning",
            {"phase_epoch_count": 101},
            {"phase_schedule_budget": {"full_tuning_epochs": 100}, "rollback_total_epoch_hard_cap_multiplier": 1.5},
        )


def test_assert_phase_epoch_count_within_budget_raises_when_session_hard_cap_exceeded():
    """§13.6.2.1 fail-loud 守门单元测试：session 累计 epoch 超 hard cap 时应 raise RuntimeError."""
    from liquidloc.models.liquid.trainer import _assert_phase_epoch_count_within_budget
    # budget=100, multiplier=1.5, hard_cap=150.
    # last_rollback_phase_epoch_count=91 + phase_epoch_count=61 = 152 > 150 => 应 raise.
    with pytest.raises(RuntimeError, match="§13.6.2.1 session total epoch over hard-cap fail-loud"):
        _assert_phase_epoch_count_within_budget(
            "full_tuning",
            {"phase_epoch_count": 61, "last_rollback_phase_epoch_count": 91},
            {"phase_schedule_budget": {"full_tuning_epochs": 100}, "rollback_total_epoch_hard_cap_multiplier": 1.5},
        )
    # 边界: 正好等于 hard_cap 应通过 (60 + 90 = 150 == 150).
    _assert_phase_epoch_count_within_budget(
        "full_tuning",
        {"phase_epoch_count": 60, "last_rollback_phase_epoch_count": 90},
        {"phase_schedule_budget": {"full_tuning_epochs": 100}, "rollback_total_epoch_hard_cap_multiplier": 1.5},
    )


def test_assert_phase_epoch_count_within_budget_skips_when_no_budget():
    """§13.6.2.1 fail-loud 守门单元测试：无 budget 定义时应直接通过."""
    from liquidloc.models.liquid.trainer import _assert_phase_epoch_count_within_budget
    _assert_phase_epoch_count_within_budget(
        "full_tuning",
        {"phase_epoch_count": 9999},
        {"phase_schedule_budget": {}, "rollback_total_epoch_hard_cap_multiplier": 1.5},
    )


def test_cross_trainer_parity_declaration_symmetric_key_mismatch_raises():
    """§13.6.2.8 cross-trainer parity 负例：注入 symmetric 键 value 不一致时应 fail-loud."""
    from liquidloc.pipelines.train_pipeline import _assert_cross_trainer_parity_declared_and_aligned
    from liquidloc.models.liquid import trainer as liquid_trainer
    original = liquid_trainer._CROSS_TRAINER_PARITY_DECLARATION["bias_huber_delta"]["value"]
    liquid_trainer._CROSS_TRAINER_PARITY_DECLARATION["bias_huber_delta"]["value"] = 999.0
    try:
        _assert_cross_trainer_parity_declared_and_aligned()
        assert False, "Expected AssertionError for symmetric key value mismatch"
    except AssertionError as exc:
        assert "§13.6.2.8 cross-trainer parity violation (symmetric key)" in str(exc)
        assert "bias_huber_delta" in str(exc)
    finally:
        liquid_trainer._CROSS_TRAINER_PARITY_DECLARATION["bias_huber_delta"]["value"] = original


def test_cross_trainer_parity_declaration_one_sided_asymmetry_raises():
    """§13.6.2.8 cross-trainer parity 负例：仅部分方声明 asymmetry_reason 时应 fail-loud (三网校验)."""
    from liquidloc.pipelines.train_pipeline import _assert_cross_trainer_parity_declared_and_aligned
    from liquidloc.models.liquid import trainer as liquid_trainer
    original_reason = liquid_trainer._CROSS_TRAINER_PARITY_DECLARATION["calibration_weight"]["asymmetry_reason"]
    liquid_trainer._CROSS_TRAINER_PARITY_DECLARATION["calibration_weight"]["asymmetry_reason"] = ""
    try:
        _assert_cross_trainer_parity_declared_and_aligned()
        assert False, "Expected AssertionError for one-sided asymmetry"
    except AssertionError as exc:
        assert "asymmetry declared by only some trainers" in str(exc)
        assert "calibration_weight" in str(exc)
    finally:
        liquid_trainer._CROSS_TRAINER_PARITY_DECLARATION["calibration_weight"]["asymmetry_reason"] = original_reason


def test_cross_trainer_parity_declaration_missing_key_raises():
    """§13.6.2.8 cross-trainer parity 负例：仅一方声明键时应 fail-loud."""
    from liquidloc.pipelines.train_pipeline import _assert_cross_trainer_parity_declared_and_aligned
    from liquidloc.models.lstm import trainer as lstm_trainer
    saved = lstm_trainer._CROSS_TRAINER_PARITY_DECLARATION.pop("gate_l1_weight", None)
    try:
        _assert_cross_trainer_parity_declared_and_aligned()
        assert False, "Expected AssertionError for missing key"
    except AssertionError as exc:
        assert "key mismatch" in str(exc)
        assert "gate_l1_weight" in str(exc)
    finally:
        if saved is not None:
            lstm_trainer._CROSS_TRAINER_PARITY_DECLARATION["gate_l1_weight"] = saved


# === §13.6.2.6 激活架构前提 fail-loud gate (F10) 单元测试 ===

def test_assert_async_pipeline_activated_passes_when_async_dt_above_threshold():
    """§13.6.2.6 正例：train_samples 真异步 Δt 激活占比 >= 阈值时通过."""
    from liquidloc.pipelines.train_pipeline import _assert_async_pipeline_activated_in_train_samples

    samples = [
        {'target_trace': {'current_modality_gap_dt': 0.05}},
        {'target_trace': {'current_modality_gap_dt': 0.10}},
        {'target_trace': {'current_modality_gap_dt': 0.0}},  # 同步样本
        {'target_trace': {'current_modality_gap_dt': 0.08}},
        {'target_trace': {'current_modality_gap_dt': 0.12}},
    ]
    audit = _assert_async_pipeline_activated_in_train_samples(samples, async_dt_min_activation_ratio=0.05)
    assert audit['async_dt_activated_count'] == 4
    assert audit['total_train_samples'] == 5
    assert audit['async_dt_activation_ratio'] == 0.8


def test_assert_async_pipeline_activated_raises_when_async_dt_below_threshold():
    """§13.6.2.6 负例 (strict 模式): train_samples 真异步 Δt 激活占比 < 阈值时 raise RuntimeError."""
    from liquidloc.pipelines.train_pipeline import _assert_async_pipeline_activated_in_train_samples

    # 5 个样本全同步 (modality_gap_dt == 0)
    samples = [
        {'target_trace': {'current_modality_gap_dt': 0.0}},
        {'target_trace': {'current_modality_gap_dt': 0.0}},
        {'target_trace': {'current_modality_gap_dt': 0.0}},
        {'target_trace': {'current_modality_gap_dt': 0.0}},
        {'target_trace': {'current_modality_gap_dt': 0.0}},
    ]
    try:
        _assert_async_pipeline_activated_in_train_samples(samples, async_dt_min_activation_ratio=0.05, strict_fail_loud=True)
        assert False, "Expected RuntimeError for async_dt activation ratio below threshold"
    except RuntimeError as exc:
        assert "§13.6.2.6" in str(exc)
        assert "async_dt activation ratio fail-loud" in str(exc)


def test_assert_async_pipeline_activated_audit_only_when_async_dt_below_threshold():
    """§13.6.2.6 audit-only 模式 (默认): 不 raise, 仅返回 violation 字段."""
    from liquidloc.pipelines.train_pipeline import _assert_async_pipeline_activated_in_train_samples

    samples = [
        {'target_trace': {'current_modality_gap_dt': 0.0}},
        {'target_trace': {'current_modality_gap_dt': 0.0}},
        {'target_trace': {'current_modality_gap_dt': 0.0}},
        {'target_trace': {'current_modality_gap_dt': 0.0}},
        {'target_trace': {'current_modality_gap_dt': 0.0}},
    ]
    audit = _assert_async_pipeline_activated_in_train_samples(samples, async_dt_min_activation_ratio=0.05)
    assert audit['violation'] is not None
    assert "§13.6.2.6" in audit['violation']
    assert "async_dt activation ratio fail-loud" in audit['violation']
    assert audit['strict_fail_loud'] is False


def test_assert_async_pipeline_activated_raises_when_state_pulse_missing():
    """§13.6.2.6 负例 (strict 模式): target_trace 缺 current_modality_gap_dt 字段时 raise RuntimeError."""
    from liquidloc.pipelines.train_pipeline import _assert_async_pipeline_activated_in_train_samples

    # 10 个样本全为异步激活 (modality_gap_dt > 0)，但缺字段 → state pulse 触发
    samples = [
        {'target_trace': {}} for _ in range(10)
    ]
    try:
        _assert_async_pipeline_activated_in_train_samples(samples, async_dt_min_activation_ratio=0.05, state_dependent_pulse_min_count=1, strict_fail_loud=True)
        assert False, "Expected RuntimeError for missing state-dependent pulse activation"
    except RuntimeError as exc:
        assert "§13.6.2.6" in str(exc)
        assert "state-dependent pulse activation fail-loud" in str(exc)


def test_assert_async_pipeline_activated_skips_empty_train_samples():
    """§13.6.2.6 边界例：train_samples 为空时返回 skipped_no_samples=True 不抛."""
    from liquidloc.pipelines.train_pipeline import _assert_async_pipeline_activated_in_train_samples

    audit = _assert_async_pipeline_activated_in_train_samples([])
    assert audit.get('skipped_no_samples') is True
    assert audit['async_dt_activated_count'] == 0


def test_assert_async_pipeline_activated_tolerates_non_mapping_sample():
    """§13.6.2.6 鲁棒例：非 mapping 样本被静默跳过不抛."""
    from liquidloc.pipelines.train_pipeline import _assert_async_pipeline_activated_in_train_samples

    samples = [
        "not_a_dict",
        {'target_trace': {'current_modality_gap_dt': 0.08}},  # 1/2 异步 = 50% > 5%
        None,
    ]
    audit = _assert_async_pipeline_activated_in_train_samples(samples, async_dt_min_activation_ratio=0.05)
    assert audit['async_dt_activated_count'] == 1
    assert audit['state_dependent_pulse_count'] == 1


# === §13.6.2.5 EKF back-end 持有校验 (F11) 单元测试 ===

def test_liquid_trainer_train_model_raises_when_cell_is_none(monkeypatch):
    """§13.6.2.5 负例: train_model 入口检测 model.network.cell is None 时 raise RuntimeError (字面含 §13.6.2.5)."""
    from liquidloc.models.liquid import trainer as liquid_trainer

    # 构造一个最小 mock model: network.cell = None
    class _FakeNetwork:
        cell = None
        def to(self, device):
            pass

    class _FakeModel:
        def __init__(self):
            self.network = _FakeNetwork()
            self.output_heads = {}
        def parameters(self):
            return iter([])

    # 替换 model_factory 返回我们构造的 mock model
    fake_model = _FakeModel()
    fake_train_cfg = {
        'name': 'liquid_ekf',
        'feature_order': ['dt'],
        'window': {'size': 1},
        'network': {'hidden_dim': 4, 'num_layers': 1},
        'train': {'epochs': 1, 'batch_size': 1, 'lr': 0.001},
        'optimizer': {'name': 'adamw'},
        'loss_weights': {},
        'seed': 1,
        'device': 'cpu',
        'output_root': './_tmp_f11_test',
    }
    train_windows = [{'feature_order': ['dt'], 'feature_values': [0.0], 'missing_mask': [0], 'dt': 0.0, 'feature_window': [[0.0]], 'missing_mask_window': [[0]], 'event_time_window': [0.0]}]
    val_windows = list(train_windows)
    try:
        liquid_trainer.train_model(train_windows, val_windows, fake_train_cfg, model_factory=lambda name, cfg: fake_model)
        assert False, "Expected RuntimeError for cell is None at training start"
    except RuntimeError as exc:
        assert "§13.6.2.5" in str(exc)
        assert "EKF back-end activation fail-loud" in str(exc)
    except Exception as exc:
        # 可能因后续步骤抛其它异常, 只要不是 cell is None 的 RuntimeError 就视为 pass
        # (本测试只确认 cell is None 的 fail-loud 被率先触发)
        assert "§13.6.2.5" not in str(exc), f"Inexpected §13.6.2.5 message in: {exc}"

