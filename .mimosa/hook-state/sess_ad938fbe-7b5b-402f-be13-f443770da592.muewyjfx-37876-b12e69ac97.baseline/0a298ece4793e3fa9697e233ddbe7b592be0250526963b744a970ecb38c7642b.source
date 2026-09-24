from __future__ import annotations

"""MILUV 数据集流水线（miluv_pipeline）测试模块。

测试覆盖范围：
- MILUV 数据集的端到端推理流程
- 序列 ID 归一化与去重
- 场景 ID 解析与归一化
- 官方锚点投影（3D→2D xy 平面）
- 输出根目录的项目根锚定
- 空白输入拒绝

被测模块：liquidloc.pipelines.miluv_pipeline"""

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.pipelines.miluv_pipeline import _resolve_scene_id, run


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[0]
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


def test_normal_case(tmp_path):
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    result = run({
        'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': field_mapping,
        'seq_ids': ['mini_seq'],
        'methods': ['ekf', 'liquid_ekf'],
        'output_root': tmp_path / 'miluv',
    })
    assert result.stage_name == 'miluv_pipeline'
    assert len(result.metadata['prediction_bundles']) == 2
    assert (tmp_path / 'miluv' / 'reports' / 'mapping_reports.json').is_file()
    assert all(task['scene_id'] == 'miluv:mini_seq' for task in result.metadata['scene_tasks'])
    assert all(bundle['scene_id'] == 'miluv:mini_seq' for bundle in result.metadata['prediction_bundles'])
    anchor_layout = result.metadata['scene_tasks'][0]['anchor_layout']
    assert anchor_layout['source'] == 'fixture_local_anchor_layout'
    assert anchor_layout['anchor_positions'][0] == [2.0, 0.0]
    assert result.metadata['mapping_reports']['mini_seq']['anchor_layout']['anchor_ids'] == [0, 1]


def test_output_root_is_anchored_to_project_root_from_relative_cwd(monkeypatch, tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    output_root = Path('outputs') / f'miluv_cwd_shift_{uuid4().hex}'
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    monkeypatch.chdir(cwd)

    result = run({
        'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': field_mapping,
        'seq_ids': ['mini_seq'],
        'methods': ['ekf'],
        'project_root': project_root,
        'output_root': output_root,
    })

    expected_root = project_root / output_root
    assert result.stage_name == 'miluv_pipeline'
    assert expected_root.is_dir()
    assert (expected_root / 'reports' / 'mapping_reports.json').is_file()
    assert not (cwd / output_root).exists()


def test_seq_ids_are_normalized_before_reading(tmp_path):
    """前置验证测试：seq ids are normalized。\n\n验证 seq ids are normalized 在后续操作前被正确检查，\n确保早期拦截无效输入。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    result = run({
        'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': field_mapping,
        'seq_ids': [' mini_seq ', 'mini_seq'],
        'methods': ['ekf'],
        'output_root': tmp_path / 'miluv_normalized',
    })
    assert [task['seq_id'] for task in result.metadata['scene_tasks']] == ['mini_seq']
    assert all(task['scene_id'] == 'miluv:mini_seq' for task in result.metadata['scene_tasks'])


def test_blank_seq_id_item_rejected():
    """拒绝测试：blank seq id item。\n\n验证被测功能对不合法的 blank seq id item 输入正确抛出异常，\n防止无效参数通过验证。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    with pytest.raises(ValueError, match='seq_ids must not contain blank items'):
        run({
            'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': field_mapping,
            'seq_ids': ['   '],
            'methods': ['ekf'],
        })


def test_blank_output_root_rejected():
    """拒绝测试：blank output root。\n\n验证被测功能对不合法的 blank output root 输入正确抛出异常，\n防止无效参数通过验证。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    with pytest.raises(ValueError, match='output_root must not be blank'):
        run({
            'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': field_mapping,
            'seq_ids': ['mini_seq'],
            'methods': ['ekf'],
            'output_root': '   ',
        })


def test_resolve_scene_id_strips_scene_id_by_seq_value():
    """场景 ID 测试：resolve scene id strips。\n\n验证 resolve scene id strips 的场景 ID 处理，\n确保归一化和解析正确。
    """
    assert _resolve_scene_id({'scene_id_by_seq': {'mini_seq': ' S(A0,N0,V0,K0,M0) '}}, 'mini_seq') == 'S(A0,N0,V0,K0,M0)'


def test_resolve_scene_id_strips_shared_scene_id_value():
    """共享测试：resolve scene id strips。\n\n验证 resolve scene id strips 的共享合同，\n确保不同模型使用一致的输入。
    """
    assert _resolve_scene_id({'scene_id': ' miluv:mini_seq '}, 'mini_seq') == 'miluv:mini_seq'


def test_provided_scene_task_seq_id_key_is_normalized(tmp_path):
    """序列 ID 测试：provided scene task。\n\n验证 provided scene task 的序列 ID 处理，\n确保归一化和去重正确。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    result = run({
        'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': field_mapping,
        'seq_ids': [' mini_seq '],
        'scene_tasks': [{
            'task_id': 'custom_00',
            'scene_id': 'S(A3,N3,V2,K0,M0)',
            'seq_id': ' mini_seq ',
            'dataset_name': 'miluv',
            'axes': {'dataset': 'miluv', 'split': 'frozen_public_eval'},
        }],
        'methods': ['ekf'],
        'output_root': tmp_path / 'miluv_scene_task_norm',
    })
    assert result.metadata['scene_tasks'][0]['task_id'] == 'custom_00'
    assert result.metadata['scene_tasks'][0]['scene_id'] == 'S(A3,N3,V2,K0,M0)'
    assert result.metadata['scene_tasks'][0]['seq_id'] == 'mini_seq'


def test_provided_scene_task_scene_id_is_normalized(tmp_path):
    """场景 ID 测试：provided scene task。\n\n验证 provided scene task 的场景 ID 处理，\n确保归一化和解析正确。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    result = run({
        'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': field_mapping,
        'seq_ids': ['mini_seq'],
        'scene_tasks': [{
            'task_id': 'custom_00',
            'scene_id': ' S(A3,N3,V2,K0,M0) ',
            'seq_id': 'mini_seq',
            'dataset_name': 'miluv',
            'axes': {'dataset': 'miluv', 'split': 'frozen_public_eval'},
        }],
        'methods': ['ekf'],
        'output_root': tmp_path / 'miluv_scene_task_scene_id_norm',
    })
    assert result.metadata['scene_tasks'][0]['scene_id'] == 'S(A3,N3,V2,K0,M0)'
    assert result.metadata['prediction_bundles'][0]['scene_id'] == 'S(A3,N3,V2,K0,M0)'


def test_blank_provided_scene_task_scene_id_rejected():
    """拒绝测试：blank provided scene task scene id。\n\n验证被测功能对不合法的 blank provided scene task scene id 输入正确抛出异常，\n防止无效参数通过验证。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    with pytest.raises(ValueError, match='scene_tasks\\[\\]\\.scene_id must not be blank'):
        run({
            'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
            'field_mapping': field_mapping,
            'seq_ids': ['mini_seq'],
            'scene_tasks': [{
                'task_id': 'custom_00',
                'scene_id': '   ',
                'seq_id': 'mini_seq',
                'dataset_name': 'miluv',
                'axes': {'dataset': 'miluv', 'split': 'frozen_public_eval'},
            }],
            'methods': ['ekf'],
        })


def test_scene_id_by_seq_key_is_normalized_with_seq_ids(tmp_path):
    """序列 ID 测试：scene id by seq key is normalized with。\n\n验证 scene id by seq key is normalized with 的序列 ID 处理，\n确保归一化和去重正确。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    result = run({
        'raw_root': ROOT / 'fixtures' / 'datasets' / 'miluv',
        'field_mapping': field_mapping,
        'seq_ids': [' mini_seq '],
        'scene_id_by_seq': {' mini_seq ': 'S(A3,N3,V2,K0,M0)'},
        'methods': ['ekf'],
        'output_root': tmp_path / 'miluv_scene_id_map_norm',
    })
    assert result.metadata['scene_tasks'][0]['scene_id'] == 'S(A3,N3,V2,K0,M0)'


def test_official_anchor_projection_case(tmp_path):
    """投影测试：official anchor。\n\n验证 official anchor 的坐标投影，\n确保 3D→2D 投影正确。
    """
    field_mapping = load_yaml_config(PROJECT_ROOT / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    raw_root = _write_official_miluv_raw_root(tmp_path / 'miluv_official')
    result = run({
        'raw_root': raw_root,
        'field_mapping': field_mapping,
        'seq_ids': ['default_3_random_0'],
        'methods': ['ekf'],
        'output_root': tmp_path / 'miluv_projected',
    })

    scene_task = result.metadata['scene_tasks'][0]
    anchor_layout = scene_task['anchor_layout']
    assert anchor_layout['anchor_ids'] == [0, 1, 2, 3, 4, 5]
    assert anchor_layout['anchor_positions'][0] == pytest.approx([3.273827392578125, 3.46404736328125])
    assert anchor_layout['original_anchor_position_dim'] == 3
    assert anchor_layout['teacher_anchor_position_dim'] == 2
    assert anchor_layout['projection'] == 'xy'
    assert anchor_layout['ignored_axis'] == 'z'
    assert result.metadata['mapping_reports']['default_3_random_0']['read_report']['anchor_layout_metadata_source'] == 'miluv_official_experiments_csv+anchors_yaml'


def test_invalid_case(tmp_path):
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    with pytest.raises(ValueError):
        run({'raw_root': tmp_path, 'field_mapping': {}, 'seq_ids': ['mini_seq'], 'methods': ['ekf']})
