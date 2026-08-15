from __future__ import annotations

"""公开基准测试流水线（public_benchmark_pipeline）测试模块。

测试覆盖范围：
- MILUV 和 NTU VIRAL 数据集的公开基准测试全链路
- 协议门控：数据集白名单、冻结评估集、调参禁止
- 序列 ID 和 split 的归一化处理
- 相对路径锚定到项目根目录
- 注册表和协议路径的项目根锚定
- 空白路径拒绝

被测模块：liquidloc.pipelines.public_benchmark_pipeline"""

from pathlib import Path
import json

import pytest

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.common.types import StageResult
from liquidloc.pipelines import public_benchmark_pipeline as public_pipeline
from liquidloc.pipelines.public_benchmark_pipeline import run


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[0]


def _patch_dataset_entry_frozen_seq_ids(monkeypatch, frozen_seq_ids):
    """Monkeypatch get_dataset_entry to include frozen_public_eval_seq_ids."""
    _real_fn = public_pipeline.get_dataset_entry

    def _wrapped(dataset_name, registry_cfg):
        entry = _real_fn(dataset_name, registry_cfg)
        entry['frozen_public_eval_seq_ids'] = list(frozen_seq_ids)
        return entry

    monkeypatch.setattr(public_pipeline, 'get_dataset_entry', _wrapped)


def test_normal_case(tmp_path, monkeypatch):
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    _patch_dataset_entry_frozen_seq_ids(monkeypatch, ['mini_seq'])
    result = run({
        'dataset_name': 'miluv',
        'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': field_mapping,
        'seq_ids': ['mini_seq'],
        'methods': ['ekf', 'liquid_ekf'],
        'split': 'frozen_public_eval',
        'output_root': tmp_path / 'public',
    })
    assert result.stage_name == 'public_benchmark_pipeline'
    assert result.metadata['public_benchmark_report']['dataset_name'] == 'miluv'
    assert result.metadata['scene_tasks'][0]['anchor_layout']['source'] == 'fixture_local_anchor_layout'
    assert (tmp_path / 'public' / 'reports' / 'public_benchmark_report.json').is_file()
    assert (tmp_path / 'public' / 'audits' / 'public_protocol_gate.json').is_file()


def test_report_bundle_count_matches_routed_scene_tasks(tmp_path, monkeypatch):
    """匹配测试：report bundle count。\n\n验证 report bundle count 的输出与预期一致，\n确保合同合规。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    _patch_dataset_entry_frozen_seq_ids(monkeypatch, ['mini_seq'])
    monkeypatch.setattr(
        public_pipeline.MiluvPipeline,
        'run',
        lambda self, pipeline_cfg=None, runtime_context=None: StageResult(
            stage_name='miluv_pipeline',
            artifacts=[],
            metadata={'scene_tasks': list((pipeline_cfg or {}).get('scene_tasks') or [])},
        ),
    )
    result = run({
        'dataset_name': 'miluv',
        'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': field_mapping,
        'seq_ids': ['mini_seq', 'mini_seq'],
        'methods': ['ekf'],
        'split': 'frozen_public_eval',
        'output_root': tmp_path / 'public',
    })
    assert result.metadata['public_benchmark_report']['prediction_bundle_count'] == len(result.metadata['scene_tasks'])


def test_pipeline_uses_gate_normalized_seq_ids_downstream(tmp_path, monkeypatch):
    """使用测试：pipeline。\n\n验证被测功能正确使用 pipeline，\n确保内部依赖被正确调用。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    _patch_dataset_entry_frozen_seq_ids(monkeypatch, ['mini_seq', 'mini_seq_02'])
    captured = {}

    monkeypatch.setattr(
        public_pipeline.MiluvPipeline,
        'run',
        lambda self, pipeline_cfg=None, runtime_context=None: (
            captured.setdefault('scene_tasks', list((pipeline_cfg or {}).get('scene_tasks') or [])),
            StageResult(
                stage_name='miluv_pipeline',
                artifacts=[],
                metadata={'scene_tasks': list((pipeline_cfg or {}).get('scene_tasks') or [])},
            )
        )[1],
    )
    result = run({
        'dataset_name': 'miluv',
        'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': field_mapping,
        'seq_ids': [' mini_seq ', 'mini_seq', 'mini_seq_02 '],
        'methods': ['ekf'],
        'split': 'frozen_public_eval',
        'output_root': tmp_path / 'public',
    })
    assert [task['seq_id'] for task in captured['scene_tasks']] == ['mini_seq', 'mini_seq_02']
    assert [task['scene_id'] for task in captured['scene_tasks']] == ['miluv:mini_seq', 'miluv:mini_seq_02']
    assert result.metadata['protocol_gate']['seq_ids'] == ['mini_seq', 'mini_seq_02']


def test_pipeline_uses_gate_normalized_split_downstream(tmp_path, monkeypatch):
    """使用测试：pipeline。\n\n验证被测功能正确使用 pipeline，\n确保内部依赖被正确调用。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    _patch_dataset_entry_frozen_seq_ids(monkeypatch, ['mini_seq'])
    captured = {}

    monkeypatch.setattr(
        public_pipeline.MiluvPipeline,
        'run',
        lambda self, pipeline_cfg=None, runtime_context=None: (
            captured.setdefault('scene_tasks', list((pipeline_cfg or {}).get('scene_tasks') or [])),
            StageResult(
                stage_name='miluv_pipeline',
                artifacts=[],
                metadata={'scene_tasks': list((pipeline_cfg or {}).get('scene_tasks') or [])},
            )
        )[1],
    )
    result = run({
        'dataset_name': 'miluv',
        'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': field_mapping,
        'seq_ids': ['mini_seq'],
        'methods': ['ekf'],
        'split': ' frozen_public_eval ',
        'output_root': tmp_path / 'public_split_norm',
    })
    assert captured['scene_tasks'][0]['axes']['split'] == 'frozen_public_eval'
    assert result.metadata['protocol_gate']['split'] == 'frozen_public_eval'


def test_pipeline_rejects_blank_split_before_routing(tmp_path):
    """拒绝测试：pipeline。\n\n验证被测功能对 pipeline 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    with pytest.raises(ValueError, match='public benchmark split must not be blank'):
        run({
            'dataset_name': 'miluv',
            'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': field_mapping,
            'seq_ids': ['mini_seq'],
            'methods': ['ekf'],
            'split': '   ',
            'output_root': tmp_path / 'public_blank_split',
        })


def test_non_miluv_report_uses_routed_scene_tasks(tmp_path, monkeypatch):
    """使用测试：non miluv report。\n\n验证被测功能正确使用 non miluv report，\n确保内部依赖被正确调用。
    """
    _patch_dataset_entry_frozen_seq_ids(monkeypatch, ['seq_a'])
    routed_scene_tasks = [
        {'task_id': 'public_00', 'scene_id': 'ntu_viral:seq_b', 'seq_id': 'seq_b', 'dataset_name': 'ntu_viral'},
        {'task_id': 'public_01', 'scene_id': 'ntu_viral:seq_c', 'seq_id': 'seq_c', 'dataset_name': 'ntu_viral'},
    ]
    captured = {}

    def _prepare_run(self, pipeline_cfg=None, runtime_context=None):
        output_root = Path((pipeline_cfg or {}).get('output_root'))
        output_root.mkdir(parents=True, exist_ok=True)
        import gzip
        import pickle
        event_path = output_root / 'seq_a_events.pkl.gz'
        with gzip.open(event_path, 'wb') as f:
            pickle.dump([], f)
        return StageResult(
            stage_name='prepare_pipeline',
            artifacts=[str(event_path)],
            metadata={
                'dataset_manifest': {'sequences': {'seq_a': {'seq_id': 'seq_a'}}},
                'scene_manifest': {'scenes': [{'scene_id': 'ntu_viral:seq_a', 'seq_ids': ['seq_a']}]},
            },
        )

    def _load_ground_truth(raw_root, seq_ids):
        captured['ground_truth'] = {'raw_root': str(raw_root), 'seq_ids': list(seq_ids)}
        return {'seq_a': [{'timestamp': 0.0, 'px': 0.0, 'py': 0.0, 'yaw': 0.0}]}

    def _load_source_report(raw_root, prepare_manifest, seq_ids, *, default_source):
        captured['source_report'] = {
            'raw_root': str(raw_root),
            'seq_ids': list(seq_ids),
            'default_source': default_source,
            'prepare_manifest': prepare_manifest,
        }
        return {'seq_a': {'source': default_source, 'prepare_sequence': {'seq_id': 'seq_a'}}}

    monkeypatch.setattr(
        public_pipeline.PreparePipeline,
        'run',
        _prepare_run,
    )
    monkeypatch.setattr(public_pipeline, 'load_prepare_manifest', lambda prepare_root: {'sequences': {'seq_a': {'seq_id': 'seq_a'}}})
    monkeypatch.setattr(public_pipeline, 'load_ground_truth_by_seq_id', _load_ground_truth)
    monkeypatch.setattr(public_pipeline, 'load_source_report_by_seq_id', _load_source_report)
    monkeypatch.setattr(
        public_pipeline.CorePipeline,
        'run',
        lambda self, pipeline_cfg=None, runtime_context=None: StageResult(
            stage_name='core_pipeline',
            artifacts=[],
            metadata={'scene_tasks': list(routed_scene_tasks), 'prediction_bundles': [{'seq_id': 'seq_b'}, {'seq_id': 'seq_c'}]},
        ),
    )
    result = run({
        'dataset_name': 'ntu_viral',
        'raw_root': tmp_path / 'ntu_viral_raw',
        'seq_ids': ['seq_a'],
        'methods': ['ekf'],
        'split': 'frozen_public_eval',
        'output_root': tmp_path / 'public',
    })
    assert result.metadata['scene_tasks'] == routed_scene_tasks
    assert result.metadata['public_benchmark_report']['seq_ids'] == ['seq_b', 'seq_c']
    assert result.metadata['public_benchmark_report']['prediction_bundle_count'] == len(routed_scene_tasks)
    assert captured['ground_truth'] == {'raw_root': str(tmp_path / 'ntu_viral_raw'), 'seq_ids': ['seq_a']}
    assert captured['source_report']['seq_ids'] == ['seq_a']
    assert captured['source_report']['default_source'] == 'ntu_viral_prepare_bridge'


def test_non_miluv_pipeline_uses_gate_normalized_seq_ids_and_split_downstream(tmp_path, monkeypatch):
    """使用测试：non miluv pipeline。\n\n验证被测功能正确使用 non miluv pipeline，\n确保内部依赖被正确调用。
    """
    _patch_dataset_entry_frozen_seq_ids(monkeypatch, ['seq_a', 'seq_b'])
    captured = {}

    def _prepare_run(self, pipeline_cfg=None, runtime_context=None):
        payload = dict(pipeline_cfg or {})
        captured['prepare_seq_ids'] = list(payload.get('seq_ids') or [])
        captured['prepare_split'] = payload.get('split')
        output_root = Path(payload['output_root'])
        output_root.mkdir(parents=True, exist_ok=True)
        import gzip
        import pickle
        artifact_paths = []
        for seq_id in captured['prepare_seq_ids']:
            event_path = output_root / f'{seq_id}_events.pkl.gz'
            with gzip.open(event_path, 'wb') as f:
                pickle.dump([], f)
            artifact_paths.append(str(event_path))
        return StageResult(
            stage_name='prepare_pipeline',
            artifacts=artifact_paths,
            metadata={
                'dataset_manifest': {'sequences': {seq_id: {'seq_id': seq_id} for seq_id in captured['prepare_seq_ids']}},
                'scene_manifest': {'scenes': []},
            },
        )

    monkeypatch.setattr(public_pipeline.PreparePipeline, 'run', _prepare_run)
    monkeypatch.setattr(public_pipeline, 'load_prepare_manifest', lambda prepare_root: {'sequences': {}})
    monkeypatch.setattr(
        public_pipeline,
        'load_ground_truth_by_seq_id',
        lambda raw_root, seq_ids: {seq_id: [] for seq_id in seq_ids},
    )
    monkeypatch.setattr(
        public_pipeline,
        'load_source_report_by_seq_id',
        lambda raw_root, prepare_manifest, seq_ids, *, default_source: {
            seq_id: {'source': default_source} for seq_id in seq_ids
        },
    )
    monkeypatch.setattr(
        public_pipeline.CorePipeline,
        'run',
        lambda self, pipeline_cfg=None, runtime_context=None: StageResult(
            stage_name='core_pipeline',
            artifacts=[],
            metadata={
                'scene_tasks': list((pipeline_cfg or {}).get('scene_tasks') or []),
                'prediction_bundles': [{'seq_id': task['seq_id']} for task in list((pipeline_cfg or {}).get('scene_tasks') or [])],
            },
        ),
    )

    result = run({
        'dataset_name': 'ntu_viral',
        'raw_root': tmp_path / 'ntu_viral_raw',
        'field_mapping': {'imu': {}, 'uwb': {}, 'vio': {}, 'gt': {}},
        'seq_ids': [' seq_a ', 'seq_a', 'seq_b '],
        'methods': ['ekf'],
        'split': ' frozen_public_eval ',
        'output_root': tmp_path / 'public_norm',
    })

    assert captured['prepare_seq_ids'] == ['seq_a', 'seq_b']
    assert captured['prepare_split'] == 'frozen_public_eval'
    assert result.metadata['protocol_gate']['seq_ids'] == ['seq_a', 'seq_b']
    assert result.metadata['protocol_gate']['split'] == 'frozen_public_eval'


def test_ntu_viral_real_chain_smoke(tmp_path, monkeypatch):
    """冒烟测试：ntu viral real chain。\n\n快速验证 ntu viral real chain 的基本功能可用，\n不深入检查细节，仅确认流程不崩溃。
    """
    anchor_layout = {'layout_id': 'ntu_seq_layout', 'anchor_ids': [1], 'anchor_positions': [[0.0, 0.0]]}
    raw_root = tmp_path / 'ntu_viral_raw'
    seq_dir = raw_root / 'ntu_seq'
    seq_dir.mkdir(parents=True)
    (seq_dir / 'anchor_layout.json').write_text(json.dumps({
        **anchor_layout,
    }), encoding='utf-8')
    (seq_dir / 'imu.json').write_text(json.dumps([
        {'timestamp': 0.0, 'ax': 0.0, 'ay': 0.0, 'gz': 0.0},
    ]), encoding='utf-8')
    (seq_dir / 'uwb.json').write_text(json.dumps([
        {'timestamp': 0.1, 'anchor_id': 1, 'range': 1.0, 'valid': True, 'quality': 0.9},
    ]), encoding='utf-8')
    (seq_dir / 'vio.json').write_text(json.dumps([
        {'timestamp': 0.2, 'dx': 0.05, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.8, 'tracked_features': 40, 'reproj_err': 0.5},
    ]), encoding='utf-8')
    (seq_dir / 'gt.json').write_text(json.dumps([
        {'timestamp': 0.0, 'px': 0.0, 'py': 0.0, 'yaw': 0.0},
        {'timestamp': 0.1, 'px': 0.1, 'py': 0.0, 'yaw': 0.0},
        {'timestamp': 0.2, 'px': 0.2, 'py': 0.0, 'yaw': 0.0},
        {'timestamp': 0.3, 'px': 0.3, 'py': 0.0, 'yaw': 0.0},
    ]), encoding='utf-8')

    real_core_pipeline_cls = public_pipeline.CorePipeline

    class _InjectAnchorLayoutCorePipeline:
        def run(self, payload, runtime_context=None):
            payload = dict(payload)
            payload['scene_tasks'] = [dict(task, anchor_layout=anchor_layout) for task in payload.get('scene_tasks', [])]
            return real_core_pipeline_cls().run(payload, runtime_context=runtime_context)

    _patch_dataset_entry_frozen_seq_ids(monkeypatch, ['ntu_seq'])
    monkeypatch.setattr(public_pipeline, 'CorePipeline', lambda: _InjectAnchorLayoutCorePipeline())

    result = run({
        'dataset_name': 'ntu_viral',
        'raw_root': raw_root,
        'field_mapping': {
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
        },
        'seq_ids': ['ntu_seq'],
        'methods': ['ekf'],
        'split': 'frozen_public_eval',
        'output_root': tmp_path / 'public',
    })

    report = result.metadata['public_benchmark_report']
    assert result.stage_name == 'public_benchmark_pipeline'
    assert report['dataset_name'] == 'ntu_viral'
    assert report['prediction_bundle_count'] == 1
    assert len(result.metadata['prediction_index']) == 1
    assert result.metadata['prediction_index'][0]['seq_id'] == 'ntu_seq'
    assert result.metadata['prediction_index'][0]['method_name'] == 'ekf'
    assert (tmp_path / 'public' / 'reports' / 'public_benchmark_report.json').is_file()
    assert (tmp_path / 'public' / 'audits' / 'public_protocol_gate.json').is_file()


def test_util_is_rejected_by_official_public_benchmark_gate(tmp_path):
    """拒绝测试：util is。\n\n验证被测功能对不合法的 util is 输入正确抛出异常，\n防止无效参数通过验证。
    """
    with pytest.raises(ValueError, match='dataset_name not allowed'):
        run({
            'dataset_name': 'util',
            'raw_root': tmp_path / 'util_raw',
            'seq_ids': ['util_seq'],
            'methods': ['ekf'],
            'split': 'frozen_public_eval',
            'output_root': tmp_path / 'public',
        })


def test_invalid_case(tmp_path):
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    with pytest.raises((ValueError, TypeError)):
        run({'raw_root': tmp_path})


def test_relative_output_root_is_project_root_anchored(tmp_path, monkeypatch):
    """项目根锚定测试：relative output root。\n\n验证 relative output root 的相对路径被正确锚定到项目根目录，\n而非当前工作目录。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    _patch_dataset_entry_frozen_seq_ids(monkeypatch, ['mini_seq'])
    project_root = tmp_path / 'repo_root'
    # _resolve_dataset_runtime_contract reads project_root / configs / datasets / miluv.yaml
    miluv_cfg_dir = project_root / 'configs' / 'datasets'
    miluv_cfg_dir.mkdir(parents=True)
    (miluv_cfg_dir / 'miluv.yaml').write_text(
        f"field_mapping:\n  imu: {{}}\n  uwb: {{}}\n  vio: {{}}\n  gt: {{}}\nraw_root: {ROOT / 'fixtures' / 'datasets' / 'miluv'}\n",
        encoding='utf-8',
    )
    result = run({
        'dataset_name': 'miluv',
        'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': field_mapping,
        'seq_ids': ['mini_seq'],
        'methods': ['ekf'],
        'split': 'frozen_public_eval',
        'project_root': project_root,
        'output_root': Path('outputs') / 'public_relative',
    })
    assert (project_root / 'outputs' / 'public_relative' / 'reports' / 'public_benchmark_report.json').is_file()
    assert result.metadata['public_benchmark_report']['dataset_name'] == 'miluv'


def test_relative_registry_and_protocol_paths_are_project_root_anchored(tmp_path, monkeypatch):
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    project_root = tmp_path / 'repo_root'
    registry_path = Path('configs') / 'datasets' / 'public_dataset_registry.yaml'
    protocol_path = Path('configs') / 'base' / 'experiment_protocol.yaml'
    captured = {}

    def _load_registry(path=None):
        captured['registry_path'] = path
        return {
            'datasets': {
                'miluv': {
                    'dataset_name': 'miluv',
                    'raw_root': str(ROOT / 'fixtures' / 'datasets' / 'miluv'),
                    'field_mapping': field_mapping,
                    'frozen_public_eval_seq_ids': ['mini_seq'],
                }
            }
        }

    def _load_protocol(path=None):
        captured['protocol_path'] = path
        return {
            'protocol_version': 2,
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
        }

    monkeypatch.setattr(public_pipeline, 'load_public_dataset_registry', _load_registry)
    monkeypatch.setattr(public_pipeline, 'load_experiment_protocol', _load_protocol)
    monkeypatch.setattr(
        public_pipeline.MiluvPipeline,
        'run',
        lambda self, pipeline_cfg=None, runtime_context=None: StageResult(
            stage_name='miluv_pipeline',
            artifacts=[],
            metadata={'scene_tasks': list((pipeline_cfg or {}).get('scene_tasks') or [])},
        ),
    )
    # _resolve_dataset_runtime_contract reads project_root / configs / datasets / miluv.yaml
    miluv_cfg_dir = project_root / 'configs' / 'datasets'
    miluv_cfg_dir.mkdir(parents=True, exist_ok=True)
    (miluv_cfg_dir / 'miluv.yaml').write_text(
        f"field_mapping:\n  imu: {{}}\n  uwb: {{}}\n  vio: {{}}\n  gt: {{}}\nraw_root: {ROOT / 'fixtures' / 'datasets' / 'miluv'}\n",
        encoding='utf-8',
    )

    run({
        'dataset_name': 'miluv',
        'registry_path': registry_path,
        'experiment_protocol_path': protocol_path,
        'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': field_mapping,
        'seq_ids': ['mini_seq'],
        'methods': ['ekf'],
        'split': 'frozen_public_eval',
        'project_root': project_root,
        'output_root': tmp_path / 'public',
    })

    assert captured['registry_path'] == project_root / registry_path
    assert captured['protocol_path'] == project_root / protocol_path


def test_blank_registry_path_rejected_before_resolution(tmp_path):
    """拒绝测试：blank registry path。\n\n验证被测功能对不合法的 blank registry path 输入正确抛出异常，\n防止无效参数通过验证。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    with pytest.raises(ValueError, match='registry_path must not be blank'):
        run({
            'dataset_name': 'miluv',
            'registry_path': '   ',
            'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': field_mapping,
            'seq_ids': ['mini_seq'],
            'methods': ['ekf'],
            'split': 'frozen_public_eval',
            'output_root': tmp_path / 'public',
        })


def test_blank_experiment_protocol_path_rejected_before_resolution(tmp_path):
    """拒绝测试：blank experiment protocol path。\n\n验证被测功能对不合法的 blank experiment protocol path 输入正确抛出异常，\n防止无效参数通过验证。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    with pytest.raises(ValueError, match='experiment_protocol_path must not be blank'):
        run({
            'dataset_name': 'miluv',
            'experiment_protocol_path': '   ',
            'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': field_mapping,
            'seq_ids': ['mini_seq'],
            'methods': ['ekf'],
            'split': 'frozen_public_eval',
            'output_root': tmp_path / 'public',
        })


def test_blank_output_root_rejected_before_resolution(tmp_path, monkeypatch):
    """拒绝测试：blank output root。\n\n验证被测功能对不合法的 blank output root 输入正确抛出异常，\n防止无效参数通过验证。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    _patch_dataset_entry_frozen_seq_ids(monkeypatch, ['mini_seq'])
    with pytest.raises(ValueError, match='output_root must not be blank'):
        run({
            'dataset_name': 'miluv',
            'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': field_mapping,
            'seq_ids': ['mini_seq'],
            'methods': ['ekf'],
            'split': 'frozen_public_eval',
            'output_root': '   ',
        })
