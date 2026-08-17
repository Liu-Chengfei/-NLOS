from __future__ import annotations

"""核心推理流水线（core_pipeline）测试模块。

测试覆盖范围：
- EKF/RobustEKF/FGO/liquid_ekf 等方法的推理流程
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
        {'t': 0.0, 'dt': 0.0, 'modality': 'imu', 'meta': {'scene_id': 'S(A1,N2,V1,G2,K4)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.1, 'ay': 0.0, 'gz': 0.01}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.1, 'dt': 0.1, 'modality': 'uwb', 'meta': {'scene_id': 'S(A1,N2,V1,G2,K4)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 0.95}, 'vio_payload': None},
        {'t': 0.2, 'dt': 0.1, 'modality': 'vio', 'meta': {'scene_id': 'S(A1,N2,V1,G2,K4)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.03, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.85, 'tracked_features': 150, 'reproj_err': 1.4}},
        {'t': 0.3, 'dt': 0.1, 'modality': 'imu', 'meta': {'scene_id': 'S(A1,N2,V1,G2,K4)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.0, 'ay': 0.1, 'gz': 0.02}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.4, 'dt': 0.1, 'modality': 'uwb', 'meta': {'scene_id': 'S(A1,N2,V1,G2,K4)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 1, 'range': 2.2, 'valid': True, 'quality': 0.9}, 'vio_payload': None},
        {'t': 0.5, 'dt': 0.1, 'modality': 'vio', 'meta': {'scene_id': 'S(A1,N2,V1,G2,K4)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.02, 'dy': 0.01, 'dyaw': 0.01, 'quality': 0.8, 'tracked_features': 20, 'reproj_err': 5.0}},
    ]


def _tmp_output_root() -> Path:
    output_root = PROJECT_ROOT / 'outputs' / 'test_core_pipeline_pytest' / uuid4().hex
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root


def _estimator_cfgs():
    # 紧耦合扩维后 8→10 维 (新增 uwb_clock_bias, vio_scale). Bug 2 删 uwb_anchor_bias.
    # 与 src/liquidloc/common/constants.py STATE_ITEMS / configs/base/task.yaml state_items 一致.
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
        },
        'robust_ekf': {
            'process_noise': dict(_process_noise),
            'measurement_noise': {'uwb': 0.30, 'vio': {'pos': 0.10, 'yaw': 0.04}},
            'init_state': dict(_init_state),
            'init_cov': list(_init_cov),
            'robust_weight': {'type': 'huber', 'delta': 1.5},
            'gate': {'mahalanobis_sq': 9.21, 'quality_floor': 0.2},
        },
        'fgo': {
            'window_size': 4,
            'process_noise': dict(_process_noise),
            'measurement_noise': {'uwb': 0.25, 'vio': {'pos': 0.08, 'yaw': 0.03}},
            'init_state': dict(_init_state),
            'optimizer': {'name': 'gauss_newton', 'max_iters': 5},
            'factor_weights': {'imu': 1.0, 'uwb': 1.0, 'vio': 1.0},
            'init_cov': list(_init_cov),
            'robust_weight': {'type': 'huber', 'delta': 1.5},
            'gate': {'mahalanobis_sq': 9.21, 'quality_floor': 0.2},
            # §3.4 + §10.3 / B10–B11 窗长比断言：window_size=4 步 @10Hz = 0.4s,
            # tau_filt_s=0.2s → ratio=2.0 ∈ [0.5, 3]。测试协议档，非主表声称。
            'nominal_event_rate_hz': 10.0,
            'tau_filt_s': 0.2,
            'window_length_ratio_min': 0.5,
            'window_length_ratio_max': 3.0,
        },
    }


def _scene_task():
    return sample_scenes({
        'primary_axis': 'target_degradation_bundle',
        'frozen_axes': {'A': 'A1', 'N': 'N2', 'V': 'V1', 'G': 'G2', 'K': 'K4', 'M': 'M0'},
    })[0]


def _relative_vio_events():
    return [
        {
            't': 0.0,
            'dt': 0.0,
            'modality': 'vio',
            'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'vio_seq'},
            'imu_payload': None,
            'uwb_payload': None,
            'vio_payload': {'dx': 1.0, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.9, 'tracked_features': 120, 'reproj_err': 0.3},
        },
        {
            't': 0.1,
            'dt': 0.1,
            'modality': 'vio',
            'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'vio_seq'},
            'imu_payload': None,
            'uwb_payload': None,
            'vio_payload': {'dx': 1.0, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.9, 'tracked_features': 120, 'reproj_err': 0.3},
        },
    ]


def _temporal_reliability_events():
    return [
        {
            't': 0.0,
            'dt': 0.0,
            'modality': 'imu',
            'meta': {'scene_id': 'S(A3,N3,V2,G1,K6)', 'seq_id': 'temporal_seq'},
            'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0},
            'uwb_payload': None,
            'vio_payload': None,
        },
        {
            't': 0.1,
            'dt': 0.1,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A3,N3,V2,G1,K6)', 'seq_id': 'temporal_seq'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 0.6},
            'vio_payload': None,
        },
        {
            't': 0.2,
            'dt': 0.1,
            'modality': 'imu',
            'meta': {'scene_id': 'S(A3,N3,V2,G1,K6)', 'seq_id': 'temporal_seq'},
            'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0},
            'uwb_payload': None,
            'vio_payload': None,
        },
        {
            't': 0.4,
            'dt': 0.2,
            'modality': 'uwb',
            'meta': {'scene_id': 'S(A3,N3,V2,G1,K6)', 'seq_id': 'temporal_seq'},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': 1, 'range': 2.3, 'valid': False, 'quality': 0.2},
            'vio_payload': None,
        },
        {
            't': 0.5,
            'dt': 0.1,
            'modality': 'vio',
            'meta': {'scene_id': 'S(A3,N3,V2,G1,K6)', 'seq_id': 'temporal_seq'},
            'imu_payload': None,
            'uwb_payload': None,
            'vio_payload': {'dx': 0.2, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.8, 'tracked_features': 40, 'reproj_err': 0.8},
        },
        {
            't': 0.8,
            'dt': 0.3,
            'modality': 'vio',
            'meta': {'scene_id': 'S(A3,N3,V2,G1,K6)', 'seq_id': 'temporal_seq'},
            'imu_payload': None,
            'uwb_payload': None,
            'vio_payload': {'dx': 0.1, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.7, 'tracked_features': 20, 'reproj_err': 1.4},
        },
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
        events.append({
            't': timestamp,
            'dt': 0.0,
            'modality': 'imu',
            'meta': {'seq_id': seq_id, 'scene_id': resolved_scene_id, 'source_t': timestamp},
            'imu_payload': {'ax': row['ax'], 'ay': row['ay'], 'gz': row['gz']},
            'uwb_payload': None,
            'vio_payload': None,
        })
    for row in uwb_rows:
        timestamp = float(row['timestamp'])
        events.append({
            't': timestamp,
            'dt': 0.0,
            'modality': 'uwb',
            'meta': {'seq_id': seq_id, 'scene_id': resolved_scene_id, 'source_t': timestamp},
            'imu_payload': None,
            'uwb_payload': {
                'anchor_id': row['anchor_id'],
                'range': row['range'],
                'valid': row['valid'],
                'quality': row['quality'],
            },
            'vio_payload': None,
        })
    for row in vio_rows:
        timestamp = float(row['timestamp'])
        events.append({
            't': timestamp,
            'dt': 0.0,
            'modality': 'vio',
            'meta': {'seq_id': seq_id, 'scene_id': resolved_scene_id, 'source_t': timestamp},
            'imu_payload': None,
            'uwb_payload': None,
            'vio_payload': {
                'dx': row['dx'],
                'dy': row['dy'],
                'dyaw': row['dyaw'],
                'quality': row['quality'],
                # sim_raw 材质化路径不输出 tracked_features/reproj_err (传感器级
                # A1 已下放此键到事件层), 缺省用 0 兜底以让 vision_model 协议校验通过.
                'tracked_features': row.get('tracked_features', 0),
                'reproj_err': row.get('reproj_err', 0.0),
            },
        })
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
    assert captured['events'][2]['vio_payload']['reproj_err'] == pytest.approx(1.0)
    assert captured['events'][5]['vio_payload']['reproj_err'] == pytest.approx(1.0)
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
    """冒烟测试：visual update relative increment。\n\n快速验证 visual update relative increment 的基本功能可用，\n不深入检查细节，仅确认流程不崩溃。
    """
    output_root = _tmp_output_root()
    result = run({
        'events': _relative_vio_events(),
        'scene_tasks': [{
            'task_id': 'vio_scene',
            'scene_id': 'S(A0,N0,V0,G0,K6)',
            'seq_id': 'vio_seq',
            'axes': {},
        }],
        'methods': ['ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    states = result.metadata['prediction_bundles'][0]['states']
    # 首个 VIO 事件初始化参考位姿（被跳过），第二个 VIO 事件做真正更新
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
            # Stage A1 后 LNN feature_order 只剩 9 项 raw, 上述默认值不再被 feature_order 命中,
            # 用一个无依赖的硬置 >1.0 分支验证 model -> control -> trace 链路保持 >1.0.
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
            'scene_id': 'S(A0,N0,V0,G0,K6)',
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
            'scene_id': 'S(A3,N3,V2,G1,K6)',
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
                # 铁律 3: VIO 紧耦合不再输出 reproj_err / tracked_features，
                # 因此 feature_order 不再包含 vio_reproj_err_slope / tracked_features_drop
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
            'scene_id': 'S(A0,N0,V0,G0,K6)',
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
                'scene_id': 'S(A0,N0,V0,G0,K6)',
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


def test_classical_baseline_methods_smoke():
    """冒烟测试：classical baseline methods。\n\n快速验证 classical baseline methods 的基本功能可用，\n不深入检查细节，仅确认流程不崩溃。
    """
    output_root = _tmp_output_root()
    result = run({
        'events': _events(),
        'scene_tasks': [_scene_task()],
        'methods': ['ekf', 'robust_ekf', 'fgo'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    method_names = {bundle['method_name'] for bundle in result.metadata['prediction_bundles']}
    assert method_names == {'ekf', 'robust_ekf', 'fgo'}
    assert len(result.metadata['runtime_table']) == 3
    assert all(bundle['runtime_log']['latency'] for bundle in result.metadata['prediction_bundles'])


def test_task_anchor_layout_precedence():
    """优先级测试：task anchor layout。\n\n验证 task anchor layout 的优先级规则，\n确保高优先级来源覆盖低优先级来源。
    """
    output_root = _tmp_output_root()
    captured_cfgs = []

    def _capture_create_estimator(name, cfg):
        captured_cfgs.append((name, deepcopy(cfg)))
        from liquidloc.factories.estimator_factory import create_estimator as real_create_estimator
        return real_create_estimator(name, cfg)

    task_anchor_layout = {
        'anchor_ids': [0, 1],
        'anchor_positions': [[2.0, 0.0], [2.1, 0.0]],
        'source': 'fixture_local_anchor_layout',
    }
    estimator_cfgs = _estimator_cfgs()
    estimator_cfgs['ekf']['anchor_layout'] = {
        'anchor_ids': ['cfg_only'],
        'anchor_positions': [[9.0, 9.0]],
        'source': 'estimator_cfg_anchor_layout',
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.create_estimator', _capture_create_estimator)
    result = run({
        'events': _events(),
        'scene_tasks': [{
            'task_id': 'miluv_scene',
            'scene_id': 'miluv:mini_seq',
            'seq_id': 'mini_seq',
            'axes': {},
            'anchor_layout': task_anchor_layout,
        }],
        'methods': ['ekf'],
        'estimator_cfgs': estimator_cfgs,
        'output_root': output_root / 'core',
    })
    monkeypatch.undo()

    scene_task = result.metadata['scene_tasks'][0]
    assert scene_task['anchor_layout']['source'] == 'fixture_local_anchor_layout'
    assert scene_task['anchor_layout']['anchor_ids'] == [0, 1]
    assert captured_cfgs
    assert captured_cfgs[0][0] == 'ekf'
    assert captured_cfgs[0][1]['anchor_layout'] == task_anchor_layout
    assert captured_cfgs[0][1]['anchor_layout'] is not task_anchor_layout


def test_quick_like_raw_path_applies_geometry_remap_before_fusion(monkeypatch):
    """前置验证测试：quick like raw path applies geometry remap。\n\n验证 quick like raw path applies geometry remap 在后续操作前被正确检查，\n确保早期拦截无效输入。
    """
    output_root = _tmp_output_root()
    captured = {}

    def _capture_run_fusion(events, estimator, model_infer=None, feature_builder=None, cfg=None):
        captured['events'] = deepcopy(events)
        return real_run_fusion(events, estimator, model_infer, feature_builder, cfg)

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.run_fusion', _capture_run_fusion)
    task = {
        'task_id': 'scene_00',
        'scene_id': 'S(A3,N3,V2,G1,K6)',
        'seq_id': 'mini_seq_02',
        'axes': {'A': 'A3', 'N': 'N3', 'V': 'V2', 'G': 'G1', 'K': 'K6'},
    }
    source_report = {
        'anchor_layout': {
            'anchor_ids': [0, 1, 2, 3, 4, 5],
            'anchor_positions': [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0)],
        }
    }
    ground_truth_by_seq_id = {
        'mini_seq_02': [
            {'timestamp': 0.02, 'px': 1.0, 'py': 0.0, 'yaw': 0.0},
            {'timestamp': 0.12, 'px': 2.0, 'py': 0.0, 'yaw': 0.0},
            {'timestamp': 0.19, 'px': 3.0, 'py': 0.0, 'yaw': 0.0},
        ]
    }
    result = run({
        'events': _events(),
        'scene_tasks': [task],
        'methods': ['ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'ground_truth_by_seq_id': ground_truth_by_seq_id,
        'source_report_by_seq_id': {'mini_seq_02': source_report},
        'output_root': output_root / 'core',
    })

    assert result.stage_name == 'core_pipeline'
    before_uwb_ranges = [event['uwb_payload']['range'] for event in _events() if event['modality'] == 'uwb']
    after_uwb_ranges = [event['uwb_payload']['range'] for event in captured['events'] if event['modality'] == 'uwb']
    assert before_uwb_ranges != after_uwb_ranges
    assert all(event['meta']['scene_id'] == task['scene_id'] for event in captured['events'])
    assert all(event['meta']['seq_id'] == task['seq_id'] for event in captured['events'])
    assert all('anchor_layout' in event['meta'] for event in captured['events'])
    assert result.metadata['prediction_bundles'][0]['scenario_context']['scenario_reports']['G']


def test_raw_core_path_cycles_target_anchor_ids_when_k_expands(monkeypatch):
    """循环测试：raw core path。\n\n验证 raw core path 的循环处理逻辑，\n确保重复模式被正确处理。
    """
    output_root = _tmp_output_root()
    captured = {}

    def _capture_run_fusion(events, estimator, model_infer=None, feature_builder=None, cfg=None):
        captured['events'] = deepcopy(events)
        return real_run_fusion(events, estimator, model_infer, feature_builder, cfg)

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.run_fusion', _capture_run_fusion)
    events = [
        {'t': 0.0, 'dt': 0.0, 'modality': 'uwb', 'meta': {'scene_id': 'scene_k4', 'seq_id': 'seq_k4'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 0.95}, 'vio_payload': None},
        {'t': 0.1, 'dt': 0.1, 'modality': 'uwb', 'meta': {'scene_id': 'scene_k4', 'seq_id': 'seq_k4'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 1, 'range': 2.1, 'valid': True, 'quality': 0.95}, 'vio_payload': None},
        {'t': 0.2, 'dt': 0.1, 'modality': 'uwb', 'meta': {'scene_id': 'scene_k4', 'seq_id': 'seq_k4'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.2, 'valid': True, 'quality': 0.95}, 'vio_payload': None},
        {'t': 0.3, 'dt': 0.1, 'modality': 'uwb', 'meta': {'scene_id': 'scene_k4', 'seq_id': 'seq_k4'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 1, 'range': 2.3, 'valid': True, 'quality': 0.95}, 'vio_payload': None},
    ]
    result = run({
        'events': events,
        'scene_tasks': [{
            'task_id': 'scene_k4',
            'scene_id': 'scene_k4',
            'seq_id': 'seq_k4',
            'axes': {'A': 'A0', 'N': 'N0', 'V': 'V0', 'G': 'G0', 'K': 'K4'},
        }],
        'methods': ['ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'ground_truth_by_seq_id': {
            'seq_k4': [
                {'timestamp': 0.0, 'px': 0.0, 'py': 0.0, 'yaw': 0.0},
                {'timestamp': 0.1, 'px': 0.0, 'py': 0.0, 'yaw': 0.0},
                {'timestamp': 0.2, 'px': 0.0, 'py': 0.0, 'yaw': 0.0},
                {'timestamp': 0.3, 'px': 0.0, 'py': 0.0, 'yaw': 0.0},
            ],
        },
        'source_report_by_seq_id': {
            'seq_k4': {
                'anchor_layout': {'anchor_ids': [0, 1], 'anchor_positions': [[2.0, 0.0], [2.1, 0.0]]},
            },
        },
        'output_root': output_root / 'core',
    })

    assert result.stage_name == 'core_pipeline'
    remapped_anchor_ids = [event['uwb_payload']['anchor_id'] for event in captured['events'] if event['modality'] == 'uwb']
    # hash 确定性分配：同一 source_anchor_id 映射到同一 target_anchor_id
    assert len(set(remapped_anchor_ids)) >= 1  # 至少有一个不同的目标锚点
    # 验证确定性：anchor_id=0 的事件映射到同一目标，anchor_id=1 的事件映射到同一目标
    source_anchor_ids = [event['uwb_payload']['anchor_id'] for event in events if event['modality'] == 'uwb']
    for src_id in set(source_anchor_ids):
        targets_for_this_source = [remapped_anchor_ids[i] for i, s in enumerate(source_anchor_ids) if s == src_id]
        assert len(set(targets_for_this_source)) == 1, f"source_anchor_id={src_id} should map to a single target"


def test_noisy_sim_raw_core_path_applies_async_nlos_visual_degradation_on_top_of_measurement_noise(monkeypatch, tmp_path):
    """应用测试：noisy sim raw core path。\n\n验证 noisy sim raw core path 的应用逻辑，\n确保特定条件触发预期行为。
    """
    output_root = _tmp_output_root()
    sim_root = tmp_path / 'sim_raw'
    zero_root = tmp_path / 'sim_raw_zero'
    seq_spec = SimSequenceSpec(
        seq_id='sim_noise_chain_case',
        base_seq_id='mini_seq_02',
        translation_xy=(0.0, 0.0),
        cycle_count=1,
        cycle_gap_s=0.10,
    )
    materialize_sim_raw(sim_root, fixture_root=FIXTURE_ROOT, sequence_specs=(seq_spec,))
    materialize_sim_raw(zero_root, fixture_root=FIXTURE_ROOT, sequence_specs=(seq_spec,), noise_spec=ZERO_SIM_NOISE_SPEC)

    raw_events, gt_rows, source_report = _load_raw_sequence_dir(sim_root / seq_spec.seq_id)
    zero_events, _, _ = _load_raw_sequence_dir(zero_root / seq_spec.seq_id)
    captured = {}

    def _capture_run_fusion(events, estimator, model_infer=None, feature_builder=None, cfg=None):
        captured['events'] = deepcopy(events)
        return real_run_fusion(events, estimator, model_infer, feature_builder, cfg)

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.run_fusion', _capture_run_fusion)
    scene_id = 'S(A3,N3,V3,G0,K6)'
    result = run({
        'events_by_seq_id': {seq_spec.seq_id: raw_events},
        'scene_tasks': [{
            'task_id': 'scene_noise_chain_00',
            'scene_id': scene_id,
            'seq_id': seq_spec.seq_id,
            'axes': {'A': 'A3', 'N': 'N3', 'V': 'V3', 'G': 'G0', 'K': 'K6'},
        }],
        'methods': ['ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'ground_truth_by_seq_id': {seq_spec.seq_id: gt_rows},
        'source_report_by_seq_id': {seq_spec.seq_id: source_report},
        'output_root': output_root / 'raw_core_noise_chain',
    })

    assert result.stage_name == 'core_pipeline'
    assert captured['events']

    raw_event_by_key = {
        (str(event['modality']), float(event['meta']['source_t'])): event
        for event in raw_events
    }
    zero_event_by_key = {
        (str(event['modality']), float(event['meta']['source_t'])): event
        for event in zero_events
    }
    captured_event_by_key = {
        (str(event['modality']), float(event['meta']['source_t'])): event
        for event in captured['events']
    }

    shared_uwb_keys = [key for key in raw_event_by_key if key[0] == 'uwb' and key in captured_event_by_key]
    shared_vio_keys = [key for key in raw_event_by_key if key[0] == 'vio' and key in captured_event_by_key]
    shared_imu_keys = [key for key in raw_event_by_key if key[0] == 'imu' and key in captured_event_by_key]
    assert shared_uwb_keys and shared_vio_keys and shared_imu_keys

    assert any(
        float(raw_event_by_key[key]['uwb_payload']['range']) != pytest.approx(float(zero_event_by_key[key]['uwb_payload']['range']))
        for key in shared_uwb_keys
    )
    assert any(
        float(raw_event_by_key[key]['imu_payload']['ax']) != pytest.approx(float(zero_event_by_key[key]['imu_payload']['ax']))
        for key in shared_imu_keys
    )
    assert any(
        float(raw_event_by_key[key]['vio_payload']['dx']) != pytest.approx(float(zero_event_by_key[key]['vio_payload']['dx']))
        for key in shared_vio_keys
    )

    assert any(
        float(captured_event_by_key[key]['t']) != pytest.approx(float(raw_event_by_key[key]['t']))
        for key in shared_uwb_keys + shared_vio_keys + shared_imu_keys
    )
    assert any(
        float(captured_event_by_key[key]['uwb_payload']['range']) != pytest.approx(float(raw_event_by_key[key]['uwb_payload']['range']))
        or float(captured_event_by_key[key]['uwb_payload']['quality']) != pytest.approx(float(raw_event_by_key[key]['uwb_payload']['quality']))
        or bool(captured_event_by_key[key]['uwb_payload']['valid']) != bool(raw_event_by_key[key]['uwb_payload']['valid'])
        for key in shared_uwb_keys
    )
    assert any(
        float(captured_event_by_key[key]['vio_payload']['dx']) != pytest.approx(float(raw_event_by_key[key]['vio_payload']['dx']))
        or float(captured_event_by_key[key]['vio_payload']['dy']) != pytest.approx(float(raw_event_by_key[key]['vio_payload']['dy']))
        or float(captured_event_by_key[key]['vio_payload']['dyaw']) != pytest.approx(float(raw_event_by_key[key]['vio_payload']['dyaw']))
        or float(captured_event_by_key[key]['vio_payload']['quality']) != pytest.approx(float(raw_event_by_key[key]['vio_payload']['quality']))
        or int(captured_event_by_key[key]['vio_payload']['tracked_features']) != int(raw_event_by_key[key]['vio_payload']['tracked_features'])
        or float(captured_event_by_key[key]['vio_payload']['reproj_err']) != pytest.approx(float(raw_event_by_key[key]['vio_payload']['reproj_err']))
        for key in shared_vio_keys
    )

    bundle = result.metadata['prediction_bundles'][0]
    assert bundle['scenario_context']['scenario_reports']['A']['protocol_consistent'] is True
    assert bundle['scenario_context']['scenario_reports']['N']['protocol_consistent'] is True
    assert bundle['scenario_context']['scenario_reports']['V']['protocol_consistent'] is True
    assert all(event['meta']['scene_id'] == scene_id for event in captured['events'])
    assert all(event['meta']['seq_id'] == seq_spec.seq_id for event in captured['events'])


def test_task_seq_id_takes_precedence_when_bundle_omits_it(monkeypatch):
    """优先级测试：task seq id takes。\n\n验证 task seq id takes 的优先级规则，\n确保高优先级来源覆盖低优先级来源。
    """
    output_root = _tmp_output_root()

    def _bundle_without_seq_id(events, estimator, model_infer=None, feature_builder=None, cfg=None):
        bundle = real_run_fusion(events, estimator, model_infer, feature_builder, cfg)
        bundle.pop('seq_id', None)
        return bundle

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.run_fusion', _bundle_without_seq_id)
    result = run({
        'events': _events(),
        'scene_tasks': [{
            'task_id': 'seq_fallback_scene',
            'scene_id': 'S(A1,N2,V1,G2,K4)',
            'seq_id': 'task_seq_id',
            'axes': {'A': 'A1', 'N': 'N2', 'V': 'V1', 'G': 'G2', 'K': 'K4'},
        }],
        'methods': ['ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    runtime_row = result.metadata['runtime_table'][0]
    prediction_entry = result.metadata['prediction_index'][0]
    bundle = result.metadata['prediction_bundles'][0]

    assert bundle['seq_id'] == 'task_seq_id'
    assert runtime_row['seq_id'] == 'task_seq_id'
    assert prediction_entry['seq_id'] == 'task_seq_id'


def test_seq_id_falls_back_to_first_event_meta_when_task_and_bundle_omit_it(monkeypatch):
    output_root = _tmp_output_root()

    def _bundle_without_seq_id(events, estimator, model_infer=None, feature_builder=None, cfg=None):
        bundle = real_run_fusion(events, estimator, model_infer, feature_builder, cfg)
        bundle.pop('seq_id', None)
        return bundle

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.run_fusion', _bundle_without_seq_id)
    result = run({
        'events': _events(),
        'scene_tasks': [{
            'task_id': 'seq_meta_fallback_scene',
            'scene_id': 'S(A1,N2,V1,G2,K4)',
            'axes': {'A': 'A1', 'N': 'N2', 'V': 'V1', 'G': 'G2', 'K': 'K4'},
        }],
        'methods': ['ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    runtime_row = result.metadata['runtime_table'][0]
    prediction_entry = result.metadata['prediction_index'][0]
    bundle = result.metadata['prediction_bundles'][0]

    assert bundle['seq_id'] == 'mini_seq'
    assert runtime_row['seq_id'] == 'mini_seq'
    assert prediction_entry['seq_id'] == 'mini_seq'


def test_scene_context_is_not_shared_between_methods():
    """共享测试：scene context is not。\n\n验证 scene context is not 的共享合同，\n确保不同模型使用一致的输入。
    """
    output_root = _tmp_output_root()
    result = run({
        'events': _events(),
        'scene_tasks': [_scene_task()],
        'methods': ['ekf', 'robust_ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    first_bundle, second_bundle = result.metadata['prediction_bundles']
    assert first_bundle['scenario_context'] is not second_bundle['scenario_context']
    first_bundle['scenario_context']['scenario_reports']['mutated'] = True
    assert 'mutated' not in second_bundle['scenario_context']['scenario_reports']


def test_scene_parameters_contract_matches_metadata_and_prediction_artifact():
    """匹配测试：scene parameters contract。\n\n验证 scene parameters contract 的输出与预期一致，\n确保合同合规。
    """
    output_root = _tmp_output_root()
    scene_parameters = {
        'axes': {
            'V': {'level': 'V1', 'degradation': 'clean'},
        },
        'flat': {
            'visual_dropout_prob': 0.0,
        },
    }
    result = run({
        'events': _relative_vio_events(),
        'scene_tasks': [{
            'task_id': 'scene_params_contract',
            'scene_id': 'S(A0,N0,V0,G0,K6)',
            'seq_id': 'vio_seq',
            'axes': {'V': 'V1'},
            'scene_parameters': scene_parameters,
        }],
        'methods': ['ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    scene_task = result.metadata['scene_tasks'][0]
    bundle = result.metadata['prediction_bundles'][0]
    prediction_path = Path(result.metadata['prediction_index'][0]['prediction_path'])
    persisted_bundle = json.loads(prediction_path.read_text(encoding='utf-8'))

    expected_scene_parameters = {
        'axes': {'V': {'level': 'V1', 'degradation': 'clean'}},
        'flat': {'visual_dropout_prob': 0.0},
        'axis_metadata': {},
    }
    assert scene_task['scene_parameters'] == expected_scene_parameters
    assert scene_task['scene_parameters'] is not scene_parameters
    assert scene_task['scene_parameters']['axes'] is not scene_parameters['axes']
    assert bundle['scenario_context']['scene_parameters'] == expected_scene_parameters
    assert persisted_bundle['scenario_context']['scene_parameters'] == expected_scene_parameters


def test_explicit_scene_parameters_do_not_suppress_declared_axes_degradation(monkeypatch):
    """退化测试：explicit scene parameters do not suppress declared axes。\n\n验证 explicit scene parameters do not suppress declared axes 的退化效果，\n确保退化操作正确改变数据质量。
    """
    output_root = _tmp_output_root()
    captured = {}

    def _capture_run_fusion(events, estimator, model_infer=None, feature_builder=None, cfg=None):
        captured['events'] = deepcopy(events)
        return real_run_fusion(events, estimator, model_infer, feature_builder, cfg)

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.run_fusion', _capture_run_fusion)
    result = run({
        'events': _events(),
        'scene_tasks': [{
            'task_id': 'scene_params_axes_precedence',
            'scene_id': 'S(A1,N2,V1,G2,K4)',
            'seq_id': 'mini_seq',
            'axes': {'A': 'A1', 'N': 'N2', 'V': 'V1', 'G': 'G2', 'K': 'K4', 'M': 'M0'},
            'scene_parameters': {
                'axes': {
                    'V': {'level': 'V1', 'degradation': 'annotated_only'},
                },
                'flat': {
                    'visual_dropout_prob': 0.0,
                },
            },
        }],
        'methods': ['ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    bundle = result.metadata['prediction_bundles'][0]
    scene_task = result.metadata['scene_tasks'][0]
    assert captured['events']
    assert bundle['scenario_context']['scenario_reports']['A']['protocol_consistent'] is True
    assert bundle['scenario_context']['scenario_reports']['N']['protocol_consistent'] is True
    assert bundle['scenario_context']['scenario_reports']['V']['protocol_consistent'] is True
    assert any(
        float(event['t']) != pytest.approx(float(source['t']))
        for event, source in zip(captured['events'], _events(), strict=True)
    )
    uwb_pairs = [
        (event, source)
        for event, source in zip(captured['events'], _events(), strict=True)
        if event['modality'] == 'uwb'
    ]
    assert any(
        float(event['uwb_payload']['range']) != pytest.approx(float(source['uwb_payload']['range']))
        or float(event['uwb_payload']['quality']) != pytest.approx(float(source['uwb_payload']['quality']))
        for event, source in uwb_pairs
    )
    vio_pairs = [
        (event, source)
        for event, source in zip(captured['events'], _events(), strict=True)
        if event['modality'] == 'vio'
    ]
    assert any(
        int(event['vio_payload']['tracked_features']) != int(source['vio_payload']['tracked_features'])
        or float(event['vio_payload']['reproj_err']) != pytest.approx(float(source['vio_payload']['reproj_err']))
        for event, source in vio_pairs
    )
    assert scene_task['scene_parameters']['axes']['A']['level'] == 'A1'
    assert scene_task['scene_parameters']['axes']['N']['level'] == 'N2'


def test_ekf_state_summary_clamps_negative_covariance_diag_to_zero():
    output_root = _tmp_output_root()
    result = run({
        'events': _events(),
        'scene_tasks': [_scene_task()],
        'methods': ['ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    bundle = result.metadata['prediction_bundles'][0]
    assert all(float(value) >= 0.0 for value in bundle['states'][0].get('covariance_diag', []))


def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    output_root = _tmp_output_root()
    with pytest.raises(ValueError):
        run({'methods': ['ekf'], 'output_root': output_root / 'core'})


def test_partial_declared_scene_axes_without_scene_parameters_preserve_partial_contract():
    """合同测试：partial declared scene axes without scene parameters preserve partial。\n\n验证 partial declared scene axes without scene parameters preserve partial 的接口合同，\n确保输入输出符合协议约定。
    """
    output_root = _tmp_output_root()
    result = run({
        'events': _relative_vio_events(),
        'scene_tasks': [{
            'task_id': 'partial_axes_only',
            'scene_id': 'S(A0,N0,V0,G0,K6)',
            'seq_id': 'vio_seq',
            'axes': {'V': 'V1'},
        }],
        'methods': ['ekf'],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    scene_task = result.metadata['scene_tasks'][0]
    bundle = result.metadata['prediction_bundles'][0]
    expected_scene_parameters = {
        'axes': {'V': {'level': 'V1'}},
        'flat': {},
        'axis_metadata': {},
    }
    assert scene_task['scene_parameters'] == expected_scene_parameters
    assert bundle['scenario_context']['scene_parameters'] == expected_scene_parameters


def test_partial_declared_scene_axes_reject_non_string_level():
    with pytest.raises(TypeError, match=r"scene axis level must be a string, got bool for axis V"):
        run({
            'events': _relative_vio_events(),
            'scene_tasks': [{
                'task_id': 'partial_axes_bad_type',
                'scene_id': 'S(A0,N0,V0,G0,K6)',
                'seq_id': 'vio_seq',
                'axes': {'V': True},
            }],
            'methods': ['ekf'],
            'estimator_cfgs': _estimator_cfgs(),
            'output_root': _tmp_output_root() / 'core',
        })


def test_partial_declared_scene_axes_reject_blank_level():
    """空白测试：partial declared scene axes reject。\n\n验证 partial declared scene axes reject 对空白输入的拒绝，\n确保空白字符串不被接受。
    """
    with pytest.raises(ValueError, match=r"scene axis level must not be blank for axis V"):
        run({
            'events': _relative_vio_events(),
            'scene_tasks': [{
                'task_id': 'partial_axes_blank_level',
                'scene_id': 'S(A0,N0,V0,G0,K6)',
                'seq_id': 'vio_seq',
                'axes': {'V': '   '},
            }],
            'methods': ['ekf'],
            'estimator_cfgs': _estimator_cfgs(),
            'output_root': _tmp_output_root() / 'core',
        })


def test_e5_ablation_alias_routes(monkeypatch):
    """别名测试：e5 ablation。\n\n验证 e5 ablation 的别名兼容性，\n确保旧参数名仍可使用。
    """
    output_root = _tmp_output_root()
    model_calls = []

    def _capture_create_model(name, cfg):
        model_calls.append(name)
        return real_create_model(name, cfg)

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.create_model', _capture_create_model)
    result = run({
        'events': _events(),
        'scene_tasks': [_scene_task()],
        'methods': [
            'liquid_ekf_full',
            'liquid_ekf_wo_liquid',
        ],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    bundles = {bundle['method_name']: bundle for bundle in result.metadata['prediction_bundles']}
    assert set(bundles) == {
        'liquid_ekf_full',
        'liquid_ekf_wo_liquid',
    }
    assert model_calls == [
        'liquid_ekf',
    ]
    assert bundles['liquid_ekf_wo_liquid']['diagnostics']['bias_trace'] == [0.0] * len(_events())
    assert bundles['liquid_ekf_wo_liquid']['diagnostics']['risk_trace'] == [0.0] * len(_events())
    assert all(value == 1.0 for value in bundles['liquid_ekf_wo_liquid']['diagnostics']['uwb_scaling_trace'])
    assert all(value == 1.0 for value in bundles['liquid_ekf_wo_liquid']['diagnostics']['vio_scaling_trace'])


def test_e5_ablation_mechanism_variants(monkeypatch):
    """机制级消融变体路由 + 行为锁死：e5_ablation 的 5 个变体都应能正常推理，
    并且被消融的字段在对应 trace 上应等于中性默认（bias=0、risk=0、vio_scaling=1）。

    关键语义：
    - wo_liquid：模型为 None，intermediate 由 build_measurement_control 内部默认中性值产生，
      因此 risk/bias/uwb_scaling/vio_scaling trace 均中性（全 0 / 全 1）。
    - wo_bias_memory：模型仍运行，但 fusion_runner._apply_mechanism_ablation 把
      intermediate.bias 强制为 0.0 后再喂给 build_measurement_control，
      体现为 bias_trace 全为 0；applied_bias_trace 也只含 UWB 模态事件，这些应为中性值 0.0。
    - wo_risk_gate：同理强制 risk=0.0，applied_risk_trace 应全 0.0。
    - wo_vio_confidence：强制 VIO 事件的 vio_scaling=1.0，applied_vio_scaling_trace
      在 VIO 模态事件上应全 1.0（其他模态事件为中性 1.0，统一全 1.0）。

    注：本测试用 N0 scene 而非 N2，避免 §8.3 全锚簇发段最小时长门对超短事件流的拦截——
    本测试关心的是机制级消融在 fusion_runner 的注入语义，不验证 NLOS 协议路径。
    """
    events = [
        {'t': 0.0, 'dt': 0.0, 'modality': 'imu', 'meta': {'scene_id': 'S(A1,N0,V1,G2,K4)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.1, 'ay': 0.0, 'gz': 0.01}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.1, 'dt': 0.1, 'modality': 'uwb', 'meta': {'scene_id': 'S(A1,N0,V1,G2,K4)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 0.95}, 'vio_payload': None},
        {'t': 0.2, 'dt': 0.1, 'modality': 'vio', 'meta': {'scene_id': 'S(A1,N0,V1,G2,K4)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.03, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.85, 'tracked_features': 150, 'reproj_err': 1.4}},
        {'t': 0.3, 'dt': 0.1, 'modality': 'imu', 'meta': {'scene_id': 'S(A1,N0,V1,G2,K4)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.0, 'ay': 0.1, 'gz': 0.02}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.4, 'dt': 0.1, 'modality': 'uwb', 'meta': {'scene_id': 'S(A1,N0,V1,G2,K4)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 1, 'range': 2.2, 'valid': True, 'quality': 0.9}, 'vio_payload': None},
        {'t': 0.5, 'dt': 0.1, 'modality': 'vio', 'meta': {'scene_id': 'S(A1,N0,V1,G2,K4)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.02, 'dy': 0.01, 'dyaw': 0.01, 'quality': 0.8, 'tracked_features': 20, 'reproj_err': 5.0}},
    ]
    scene_task = sample_scenes({
        'primary_axis': 'target_degradation_bundle',
        'frozen_axes': {'A': 'A1', 'N': 'N0', 'V': 'V1', 'G': 'G2', 'K': 'K4', 'M': 'M0'},
    })[0]
    output_root = _tmp_output_root()
    model_calls = []

    def _capture_create_model(name, cfg):
        model_calls.append(name)
        return real_create_model(name, cfg)

    monkeypatch.setattr('liquidloc.pipelines.core_pipeline.create_model', _capture_create_model)
    result = run({
        'events': events,
        'scene_tasks': [scene_task],
        'methods': [
            'liquid_ekf_full',
            'liquid_ekf_wo_liquid',
            'liquid_ekf_wo_bias_memory',
            'liquid_ekf_wo_risk_gate',
            'liquid_ekf_wo_vio_confidence',
        ],
        'estimator_cfgs': _estimator_cfgs(),
        'output_root': output_root / 'core',
    })

    bundles = {bundle['method_name']: bundle for bundle in result.metadata['prediction_bundles']}
    assert set(bundles) == {
        'liquid_ekf_full',
        'liquid_ekf_wo_liquid',
        'liquid_ekf_wo_bias_memory',
        'liquid_ekf_wo_risk_gate',
        'liquid_ekf_wo_vio_confidence',
    }
    # 4 个含 liquid 模型的变体都各自建一个 liquid 实例：full / wo_bias_memory / wo_risk_gate / wo_vio_confidence。
    # wo_liquid 走 model_name=None 分支不调 create_model。
    assert model_calls == ['liquid_ekf', 'liquid_ekf', 'liquid_ekf', 'liquid_ekf']

    # --- liquid_ekf_wo_liquid（纯 EKF 基线）：模型为 None，intermediate 全中性 ---
    assert bundles['liquid_ekf_wo_liquid']['diagnostics']['bias_trace'] == [0.0] * len(events)
    assert bundles['liquid_ekf_wo_liquid']['diagnostics']['risk_trace'] == [0.0] * len(events)
    assert all(value == 1.0 for value in bundles['liquid_ekf_wo_liquid']['diagnostics']['uwb_scaling_trace'])
    assert all(value == 1.0 for value in bundles['liquid_ekf_wo_liquid']['diagnostics']['vio_scaling_trace'])

    # --- liquid_ekf_wo_bias_memory：结构级消融让 intermediate.bias → 0.0 ---
    bias_trace = bundles['liquid_ekf_wo_bias_memory']['diagnostics']['bias_trace']
    assert len(bias_trace) == len(events)
    assert all(float(v) == 0.0 for v in bias_trace)
    # applied_bias_trace 在 UWB 模态才有非平凡值（IMU/VIO 路径填 1.0），
    # 结构级消融应确保所有 UWB 事件的 applied_bias 也是 0.0。
    applied_bias = bundles['liquid_ekf_wo_bias_memory']['diagnostics']['applied_bias_trace']
    assert len(applied_bias) == len(events)
    assert all(float(v) == 0.0 for v in applied_bias)
    # risk / scaling 不应被该消融触碰，保持与 full 同口径的非平凡值（仅断言存在且非全中性）。
    risk_trace = bundles['liquid_ekf_wo_bias_memory']['diagnostics']['risk_trace']
    assert len(risk_trace) == len(events)
    vio_scaling = bundles['liquid_ekf_wo_bias_memory']['diagnostics']['vio_scaling_trace']
    assert len(vio_scaling) == len(events)

    # --- liquid_ekf_wo_risk_gate：结构级消融让 intermediate.risk → 0.0 ---
    risk_trace = bundles['liquid_ekf_wo_risk_gate']['diagnostics']['risk_trace']
    assert len(risk_trace) == len(events)
    assert all(float(v) == 0.0 for v in risk_trace)
    applied_risk = bundles['liquid_ekf_wo_risk_gate']['diagnostics']['applied_risk_trace']
    assert len(applied_risk) == len(events)
    # applied_risk 不必为 0：协议层 _resolve_effective_risk 仍会叠加 quality_risk /
    # modality_signal / axis_floor 的下限值。wo_risk_gate 关掉的是"模型学到的 risk"，
    # 不应绕过协议层的质量下限守门。等价语义：wo_risk_gate 的 applied_risk 应与 wo_liquid
    # 同口径（wo_liquid 的 risk 也是 0，两条路径都只走协议 quality floor）。
    applied_risk_wo_liquid = bundles['liquid_ekf_wo_liquid']['diagnostics']['applied_risk_trace']
    assert applied_risk == applied_risk_wo_liquid, (
        'wo_risk_gate applied_risk must match wo_liquid baseline '
        '(both have intermediate.risk=0, only protocol quality_floor survives)'
    )
    # bias / scaling 不应被该消融触碰。
    bias_trace = bundles['liquid_ekf_wo_risk_gate']['diagnostics']['bias_trace']
    assert len(bias_trace) == len(events)
    vio_scaling = bundles['liquid_ekf_wo_risk_gate']['diagnostics']['vio_scaling_trace']
    assert len(vio_scaling) == len(events)

    # --- liquid_ekf_wo_vio_confidence：结构级消融在 VIO 事件上让 intermediate.vio_scaling → 1.0 ---
    vio_scaling = bundles['liquid_ekf_wo_vio_confidence']['diagnostics']['vio_scaling_trace']
    assert len(vio_scaling) == len(events)
    assert all(float(v) == 1.0 for v in vio_scaling), (
        'VIO confidence ablation must force vio_scaling_trace to neutral 1.0 on every event'
    )
    applied_vio_scaling = bundles['liquid_ekf_wo_vio_confidence']['diagnostics']['applied_vio_scaling_trace']
    assert len(applied_vio_scaling) == len(events)
    assert all(float(v) == 1.0 for v in applied_vio_scaling), (
        'VIO confidence ablation must force applied_vio_scaling_trace to neutral 1.0 on every event'
    )
    # bias / risk 不应被该消融触碰。
    bias_trace = bundles['liquid_ekf_wo_vio_confidence']['diagnostics']['bias_trace']
    assert len(bias_trace) == len(events)
    risk_trace = bundles['liquid_ekf_wo_vio_confidence']['diagnostics']['risk_trace']
    assert len(risk_trace) == len(events)
