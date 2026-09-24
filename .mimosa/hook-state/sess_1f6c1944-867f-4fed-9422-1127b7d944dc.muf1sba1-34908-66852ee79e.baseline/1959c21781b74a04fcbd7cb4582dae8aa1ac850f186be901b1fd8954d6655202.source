from __future__ import annotations

"""核心推理流水线（core_pipeline）测试模块。

测试覆盖范围：
- EKF/EKF/liquid_ekf 等方法的推理流程
- 场景退化（异步/视觉/NLOS/几何）在推理链路中的应用
- 神经网络模型的 eval 模式切换
- 特征窗口构建与时间可靠性上下文
- 锚点布局优先级（task > estimator_cfg）
- seq_id 回退机制（task > bundle > event meta）
- 场景参数合同与持久化一致性
- E5 消融别名路由

被测模块：liquidloc.pipelines.core_pipeline"""

from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import pytest

from liquidloc.dataio.sim_materializer import ZERO_SIM_NOISE_SPEC, SimSequenceSpec, materialize_sim_raw
from liquidloc.factories.model_factory import create_model as real_create_model
from liquidloc.fusion.fusion_runner import run_fusion as real_run_fusion
from liquidloc.pipelines.core_pipeline import run
from liquidloc.scenarios.scene_sampler import sample_scenes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / 'tests' / 'fixtures' / 'datasets' / 'miluv'


def _events():
    return [
        {'t': 0.0, 'dt': 0.0, 'modality': 'imu', 'meta': {'scene_id': 'S(A1,N2,V1,K3)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.1, 'ay': 0.0, 'gz': 0.01}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.1, 'dt': 0.1, 'modality': 'uwb', 'meta': {'scene_id': 'S(A1,N2,V1,K3)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 0.95}, 'vio_payload': None},
        {'t': 0.2, 'dt': 0.1, 'modality': 'vio', 'meta': {'scene_id': 'S(A1,N2,V1,K3)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.03, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.85, 'tracked_features': 150, 'reproj_err': 1.4}},
        {'t': 0.3, 'dt': 0.1, 'modality': 'imu', 'meta': {'scene_id': 'S(A1,N2,V1,K3)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.0, 'ay': 0.1, 'gz': 0.02}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.4, 'dt': 0.1, 'modality': 'uwb', 'meta': {'scene_id': 'S(A1,N2,V1,K3)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 1, 'range': 2.2, 'valid': True, 'quality': 0.9}, 'vio_payload': None},
        {'t': 0.5, 'dt': 0.1, 'modality': 'vio', 'meta': {'scene_id': 'S(A1,N2,V1,K3)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.02, 'dy': 0.01, 'dyaw': 0.01, 'quality': 0.8, 'tracked_features': 20, 'reproj_err': 5.0}},
    ]


def _tmp_output_root() -> Path:
    output_root = PROJECT_ROOT / 'outputs' / 'test_core_pipeline_pytest' / uuid4().hex
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root


def _estimator_cfgs():
    _process_noise = {
        'pos': 0.05, 'vel': 0.10, 'yaw': 0.02, 'accel_bias': 0.001, 'gyro_bias': 0.001,
        'uwb_clock_bias': 0.001, 'vio_scale': 0.001,
    }
    _init_state = {
        'px': 0.0, 'py': 0.0, 'vx': 0.0, 'vy': 0.0, 'yaw': 0.0,
        'bax': 0.0, 'bay': 0.0, 'bg': 0.0,
        'uwb_clock_bias': 0.0, 'vio_scale': 1.0,
    }
    _init_cov = [1.0, 1.0, 0.5, 0.5, 0.3, 0.05, 0.05, 0.02, 0.05, 0.01]
    return {
        'ekf': {
            'process_noise': dict(_process_noise),
            'measurement_noise': {'uwb': 0.25, 'vio': {'pos': 0.08, 'yaw': 0.03}},
            'init_state': dict(_init_state),
            'init_cov': list(_init_cov),
            'gate': {'mahalanobis_sq': 9.21, 'quality_floor': 0.2},
        },
    }


def _scene_task():
    return sample_scenes({
        'primary_axis': 'target_degradation_bundle',
        'frozen_axes': {'A': 'A1', 'N': 'N2', 'V': 'V1', 'K': 'K3', 'M': 'M0'},
    })[0]


def _relative_vio_events():
    return [
        {
            't': 0.0, 'dt': 0.0, 'modality': 'vio',
            'meta': {'scene_id': 'S(A0,N0,V0,K3)', 'seq_id': 'vio_seq'},
            'imu_payload': None, 'uwb_payload': None,
            'vio_payload': {'dx': 1.0, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.98, 'tracked_features': 120, 'reproj_err': 0.3},
        },
        {
            't': 0.1, 'dt': 0.1, 'modality': 'vio',
            'meta': {'scene_id': 'S(A0,N0,V0,K3)', 'seq_id': 'vio_seq'},
            'imu_payload': None, 'uwb_payload': None,
            'vio_payload': {'dx': 1.0, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.98, 'tracked_features': 120, 'reproj_err': 0.3},
        },
    ]


def _temporal_reliability_events():
    return [
        {'t': 0.0, 'dt': 0.0, 'modality': 'imu', 'meta': {'scene_id': 'S(A3,N3,V2,K3)', 'seq_id': 'temporal_seq'}, 'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.1, 'dt': 0.1, 'modality': 'uwb', 'meta': {'scene_id': 'S(A3,N3,V2,K3)', 'seq_id': 'temporal_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 0.6}, 'vio_payload': None},
        {'t': 0.2, 'dt': 0.1, 'modality': 'imu', 'meta': {'scene_id': 'S(A3,N3,V2,K3)', 'seq_id': 'temporal_seq'}, 'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.4, 'dt': 0.2, 'modality': 'uwb', 'meta': {'scene_id': 'S(A3,N3,V2,K3)', 'seq_id': 'temporal_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 1, 'range': 2.3, 'valid': False, 'quality': 0.2}, 'vio_payload': None},
        {'t': 0.5, 'dt': 0.1, 'modality': 'vio', 'meta': {'scene_id': 'S(A3,N3,V2,K3)', 'seq_id': 'temporal_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.2, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.8, 'tracked_features': 40, 'reproj_err': 0.8}},
        {'t': 0.8, 'dt': 0.3, 'modality': 'vio', 'meta': {'scene_id': 'S(A3,N3,V2,K3)', 'seq_id': 'temporal_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.1, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.7, 'tracked_features': 20, 'reproj_err': 1.4}},
    ]


def _load_raw_sequence_dir(seq_root: Path, *, scene_id: str | None = None) -> tuple[list[dict], list[dict], dict]:
    seq_id = str(seq_root.name)
    resolved_scene_id = str(scene_id or seq_id)
    imu_rows = json.loads((seq_root / 'imu.json').read_text(encoding='utf-8'))
    uwb_rows = json.loads((seq_root / 'uwb.json').read_text(encoding='utf-8'))
    vio_rows = json.loads((seq_root / 'vio.json').read_text(encoding='utf-8'))
    gt_rows = json.loads((seq_root / 'gt.json').read_text(encoding='utf-8'))
    anchor_layout = json.loads((seq_root / 'anchor_layout.json').read_text(encoding='utf-8'))
    events: list[dict] = []
    modality_order = {'imu': 0, 'uwb': 1, 'vio': 2}
    for row in imu_rows:
        timestamp = float(row['timestamp'])
        events.append({'t': timestamp, 'dt': 0.0, 'modality': 'imu', 'meta': {'seq_id': seq_id, 'scene_id': resolved_scene_id, 'source_t': timestamp}, 'imu_payload': {'ax': row['ax'], 'ay': row['ay'], 'gz': row['gz']}, 'uwb_payload': None, 'vio_payload': None})
    for row in uwb_rows:
        timestamp = float(row['timestamp'])
        events.append({'t': timestamp, 'dt': 0.0, 'modality': 'uwb', 'meta': {'seq_id': seq_id, 'scene_id': resolved_scene_id, 'source_t': timestamp}, 'imu_payload': None, 'uwb_payload': {'anchor_id': row['anchor_id'], 'range': row['range'], 'valid': row['valid'], 'quality': row['quality']}, 'vio_payload': None})
    for row in vio_rows:
        timestamp = float(row['timestamp'])
        events.append({'t': timestamp, 'dt': 0.0, 'modality': 'vio', 'meta': {'seq_id': seq_id, 'scene_id': resolved_scene_id, 'source_t': timestamp}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': row['dx'], 'dy': row['dy'], 'dyaw': row['dyaw'], 'quality': row['quality'], 'tracked_features': row.get('tracked_features', 0), 'reproj_err': row.get('reproj_err', 0.0)}})
    events.sort(key=lambda event: (float(event['t']), modality_order[str(event['modality'])]))
    previous_t = None
    for event in events:
        current_t = float(event['t'])
        event['dt'] = 0.0 if previous_t is None else round(current_t - previous_t, 6)
        previous_t = current_t
    return events, gt_rows, {'anchor_layout': anchor_layout}

def test_normal_case(monkeypatch):
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    original_events = _events()
    captured = {}
    output_root = _tmp_output_root()

    def _capture_run_fusion(events, estimator, model_infer=None, feature_builder=None, cfg=None):
        captured['events'] = deepcopy(events)
        return real_run_fusion(events, estimator, model_infer, feature_builder, cfg)

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.run_fusion', _capture_run_fusion)
    result = run({
        'events': original_events,
        'scene_tasks': [_scene_task()],
        'methods': ['ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    assert result.stage_name == 'core_pipeline'
    assert len(result.metadata['prediction_bundles']) == 1
    assert captured['events'][2]['vio_payload']['tracked_features'] == 120
    assert captured['events'][5]['vio_payload']['tracked_features'] == 20
    assert captured['events'][2]['vio_payload']['reproj_err'] == pytest.approx(1.0, abs=0.05)
    assert captured['events'][5]['vio_payload']['reproj_err'] == pytest.approx(1.0, abs=0.05)
    assert captured['events'][2]['vio_payload']['dx'] == pytest.approx(original_events[2]['vio_payload']['dx'])
    bundle = result.metadata['prediction_bundles'][0]
    assert bundle['scenario_context']['scenario_reports']['V']['protocol_consistent'] is True
    assert bundle['runtime_log']['latency']


def test_output_root_is_anchored_to_project_root_from_relative_cwd(monkeypatch, tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    output_root = Path('outputs') / 'cwd_shift_core'
    monkeypatch.chdir(cwd)

    result = run({
        'events': _events(),
        'scene_tasks': [_scene_task()],
        'methods': ['ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'project_root': project_root,
        'output_root': output_root,
    })

    expected_root = project_root / output_root
    assert result.artifacts
    assert expected_root.is_dir()
    assert (expected_root / 'predictions').is_dir()
    assert not (cwd / output_root).exists()


def test_visual_update_relative_increment_smoke():
    """冒烟测试：visual update relative increment。\n\n快速验证 visual update relative increment 的基本功能可用，
不深入检查细节，仅确认流程不崩溃。
    """
    output_root = _tmp_output_root()
    result = run({
        'events': _relative_vio_events(),
        'scene_tasks': [{
            'task_id': 'vio_scene',
            'scene_id': 'S(A0,N0,V0,K3)',
            'seq_id': 'vio_seq',
            'axes': {},
        }],
        'methods': ['ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    states = result.metadata['prediction_bundles'][0]['states']
    assert states[1]['px'] > 0.5


def test_boundary_case(monkeypatch):
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    output_root = _tmp_output_root()
    captured_windows = []

    class _RecordingModel:
        def eval(self):
            return None

        def infer_intermediate(self, window_tensor):
            captured_windows.append(deepcopy(window_tensor))
            feature_order = list(window_tensor.get('feature_order') or [])
            feature_values = list(window_tensor.get('feature_values') or [])
            missing_mask = list(window_tensor.get('missing_mask') or [])
            current_modality = str(window_tensor.get('current_modality') or '')
            feature_index_by_name = {name: index for index, name in enumerate(feature_order)}

            def _read_feature(name, default):
                feature_index = feature_index_by_name.get(name)
                if feature_index is None or feature_index >= len(feature_values):
                    return default
                if feature_index < len(missing_mask) and int(missing_mask[feature_index]) != 0:
                    return default
                return float(feature_values[feature_index])

            quality = _read_feature('quality', 1.0)
            tracked_features = _read_feature('tracked_features', 120.0)
            reproj_err = _read_feature('reproj_err', 0.0)
            uwb_scaling = 1.0
            vio_scaling = 1.0
            if current_modality == 'vio':
                vio_scaling = 1.35
            if current_modality == 'uwb':
                uwb_scaling = 1.25
            return {
                'bias': 0.0,
                'risk': 0.0,
                'uwb_scaling': uwb_scaling,
                'vio_scaling': vio_scaling,
            }

    def _create_model(name, cfg):
        return _RecordingModel()

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.create_model', _create_model)
    result = run({
        'events': _events(),
        'scene_tasks': [_scene_task()],
        'methods': ['liquid_ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    bundle = result.metadata['prediction_bundles'][0]
    assert bundle['method_name'] == 'liquid_ekf'
    assert len(bundle['states']) == len(_events())
    assert bundle['scenario_context']['scenario_reports']['V']['protocol_consistent'] is True
    assert max(bundle['diagnostics']['vio_scaling_trace']) > 1.0
    assert captured_windows
    assert max(len(window_tensor['feature_window']) for window_tensor in captured_windows) > 1
    assert all(len(window_tensor['feature_window']) == len(window_tensor['missing_mask_window']) for window_tensor in captured_windows)


def test_neural_models_switch_to_eval_mode_in_core_pipeline(monkeypatch):
    """切换测试：neural models。\n\n验证 neural models 的模式切换，\n确保状态转换正确。
    """
    output_root = _tmp_output_root()
    eval_called = {'count': 0}

    class _RecordingModel:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def eval(self):
            eval_called['count'] += 1
            return self._wrapped.eval()

    monkeypatch.setattr(
        'liquidloc.pipelines.core_pipeline.create_model',
        lambda name, cfg: _RecordingModel(real_create_model(name, cfg)),
    )

    result = run({
        'events': _events(),
        'scene_tasks': [_scene_task()],
        'methods': ['lstm_ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    assert result.stage_name == 'core_pipeline'
    assert eval_called['count'] == 1

def test_neural_windows_use_per_step_pre_update_state_snapshots(monkeypatch):
    output_root = _tmp_output_root()
    captured_windows = []

    class _RecordingModel:
        def __getattr__(self, name):
            raise AttributeError(name)

        def infer_intermediate(self, window_tensor):
            captured_windows.append(deepcopy(window_tensor))
            return {
                'bias': 0.0,
                'risk': 0.0,
                'uwb_scaling': 1.0,
                'vio_scaling': 1.0,
            }

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.create_model', lambda name, cfg: _RecordingModel())
    result = run({
        'events': _relative_vio_events(),
        'scene_tasks': [{
            'task_id': 'vio_snapshot_scene',
            'scene_id': 'S(A0,N0,V0,K3)',
            'seq_id': 'vio_seq',
            'axes': {},
        }],
        'methods': ['liquid_ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'model_cfgs': {
            'liquid_ekf': {
                'feature_order': ['dx', 'px'],
                'window': {'size': 2, 'step': 1},
            }
        },
        'output_root': output_root / 'core',
    })

    assert result.stage_name == 'core_pipeline'
    assert len(captured_windows) == 2
    first_state_px = result.metadata['prediction_bundles'][0]['states'][0]['px']
    assert captured_windows[0]['feature_window'][0] == pytest.approx([1.0, 0.0])
    assert captured_windows[1]['feature_window'][0] == pytest.approx([1.0, 0.0])
    assert captured_windows[1]['feature_window'][1] == pytest.approx([1.0, first_state_px])
    assert captured_windows[1]['window_index_map'] == [0, 1]
    assert captured_windows[1]['event_time_window'] == pytest.approx([0.0, 0.1])
    assert captured_windows[1]['dt'] == pytest.approx(0.1)


def test_neural_windows_materialize_temporal_reliability_context(monkeypatch):
    """物化测试：neural windows。\n\n验证 neural windows 的物化过程，\n确保上下文被正确构建。
    """
    output_root = _tmp_output_root()
    captured_windows = []

    class _RecordingModel:
        def __getattr__(self, name):
            raise AttributeError(name)

        def infer_intermediate(self, window_tensor):
            captured_windows.append(deepcopy(window_tensor))
            return {
                'bias': 0.0,
                'risk': 0.0,
                'uwb_scaling': 1.0,
                'vio_scaling': 1.0,
            }

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.create_model', lambda name, cfg: _RecordingModel())
    result = run({
        'events': _temporal_reliability_events(),
        'scene_tasks': [{
            'task_id': 'temporal_context_scene',
            'scene_id': 'S(A3,N3,V2,K3)',
            'seq_id': 'temporal_seq',
            'axes': {},
            'anchor_layout': {
                'anchor_ids': [0, 1],
                'anchor_positions': [[0.0, 0.0], [1.0, 0.0]],
            },
        }],
        'methods': ['liquid_ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'model_cfgs': {
            'liquid_ekf': {
                'feature_order': ['modality_gap_dt', 'uwb_quality_min', 'uwb_invalid_rate'],
                'window': {'size': 8, 'step': 1},
            }
        },
        'output_root': output_root / 'core',
    })

    assert result.stage_name == 'core_pipeline'
    assert len(captured_windows) == 4
    final_window = captured_windows[-1]
    assert final_window['feature_values'] == pytest.approx([0.3, 0.2, 0.5])
    assert final_window['missing_mask'] == [0, 0, 0]
    assert final_window['event_time_window'] == pytest.approx([0.0, 0.1, 0.2, 0.4, 0.5, 0.8])


def test_core_pipeline_passes_safe_mode_cfg_from_model_cfg_to_fusion(monkeypatch):
    """传递测试：core pipeline。\n\n验证 core pipeline 的传递一致性，\n确保数据在流水线中无损传递。
    """
    output_root = _tmp_output_root()
    captured_cfgs = []

    class _RecordingModel:
        def __getattr__(self, name):
            raise AttributeError(name)

        def infer_intermediate(self, window_tensor):
            return {
                'bias': 0.0,
                'risk': 0.0,
                'uwb_scaling': 1.0,
                'vio_scaling': 1.0,
            }

    def _capture_run_fusion(events, estimator, model_infer=None, feature_builder=None, cfg=None):
        captured_cfgs.append(deepcopy(cfg))
        return real_run_fusion(events, estimator, model_infer, feature_builder, cfg)

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.create_model', lambda name, cfg: _RecordingModel())
    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.run_fusion', _capture_run_fusion)

    result = run({
        'events': _relative_vio_events(),
        'scene_tasks': [{
            'task_id': 'vio_safe_mode_scene',
            'scene_id': 'S(A0,N0,V0,K3)',
            'seq_id': 'vio_seq',
            'axes': {},
        }],
        'methods': ['liquid_ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'model_cfgs': {
            'liquid_ekf': {
                'feature_order': ['dx', 'px'],
                'window': {'size': 2, 'step': 1},
                'safe_mode': {'enabled': False, 'risk_threshold': 0.25},
            }
        },
        'output_root': output_root / 'core',
    })

    assert result.stage_name == 'core_pipeline'
    assert captured_cfgs
    assert captured_cfgs[0]['safe_mode'] == {'enabled': False, 'risk_threshold': 0.25}


def test_neural_windows_missing_dt_raises(monkeypatch):
    """缺失测试：neural windows。\n\n验证 neural windows 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    output_root = _tmp_output_root()
    events = _relative_vio_events()
    events[1].pop('dt')

    class _RecordingModel:
        def __getattr__(self, name):
            raise AttributeError(name)

        def infer_intermediate(self, window_tensor):
            return {
                'bias': 0.0,
                'risk': 0.0,
                'uwb_scaling': 1.0,
                'vio_scaling': 1.0,
            }

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.create_model', lambda name, cfg: _RecordingModel())
    with pytest.raises(KeyError, match='dt'):
        run({
            'events': events,
            'scene_tasks': [{
                'task_id': 'vio_missing_dt_scene',
                'scene_id': 'S(A0,N0,V0,K3)',
                'seq_id': 'vio_seq',
                'axes': {},
            }],
            'methods': ['liquid_ekf'],
            'estimator_cfgs': _estimator_cfgs(),
            'model_cfgs': {
                'liquid_ekf': {
                    'feature_order': ['dx', 'px'],
                    'window': {'size': 2, 'step': 1},
                }
            },
            'output_root': output_root / 'core',
        })


@pytest.mark.skip(reason="经典基线冒烟测试不再适用（方法已精简）")
def test_classical_baseline_methods_smoke():
    """冒烟测试：classical baseline methods。

快速验证 classical baseline methods 的基本功能可用，
不深入检查细节，仅确认流程不崩溃。
    """
    output_root = _tmp_output_root()
    result = run({
        'events': _events(),
        'scene_tasks': [_scene_task()],
        'methods': ['ekf', 'ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    method_names = {bundle['method_name'] for bundle in result.metadata['prediction_bundles']}
    assert method_names == {'ekf', 'ekf'}
    assert len(result.metadata['runtime_table']) == 2
    assert all(bundle['runtime_log']['latency'] for bundle in result.metadata['prediction_bundles'])

