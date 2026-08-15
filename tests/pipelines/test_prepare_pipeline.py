"""数据准备流水线（prepare_pipeline）测试模块。

测试覆盖范围：
- MILUV/SIM/UTIL 数据集的准备流程
- 场景 ID 解析与空白处理
- 流量数据（flow）到 VIO 格式的转换
- 输出清单（manifest）的序列过滤
- 重复序列 ID 拒绝
- 输出根目录的项目根锚定

被测模块：liquidloc.pipelines.prepare_pipeline"""

import json
import shutil
from uuid import uuid4
from pathlib import Path

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.pipelines.prepare_pipeline import _flow_rows_as_vio_rows
from liquidloc.pipelines.prepare_pipeline import _resolve_scene_id
from liquidloc.pipelines.prepare_pipeline import run


def test_normal_case(tmp_path):
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    root = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv'
    field_mapping = load_yaml_config(Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    result = run({
        'dataset_name': 'miluv',
        'raw_root': root,
        'seq_ids': ['mini_seq'],
        'field_mapping': field_mapping,
        'output_root': tmp_path / 'prepare',
    })
    assert result.stage_name == 'prepare_pipeline'
    assert any(path.endswith('mini_seq_events.pkl.gz') for path in result.artifacts)
    assert (tmp_path / 'prepare' / 'prepare_manifest.json').is_file()
    assert result.metadata['sequences']['mini_seq']['scene_id'] == 'miluv:mini_seq'
    assert result.metadata['scene_manifest']['scenes'][0]['scene_id'] == 'miluv:mini_seq'
    assert result.metadata['dataset_manifest']['sequences'][0]['scene_id'] == 'miluv:mini_seq'


def test_output_root_is_anchored_to_project_root_from_relative_cwd(monkeypatch, tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    output_root = Path('outputs') / f'prepare_cwd_shift_{uuid4().hex}'
    field_mapping = load_yaml_config(Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    monkeypatch.chdir(cwd)

    result = run({
        'dataset_name': 'miluv',
        'raw_root': project_root / 'tests' / 'fixtures' / 'datasets' / 'miluv',
        'seq_ids': ['mini_seq'],
        'field_mapping': field_mapping,
        'project_root': project_root,
        'output_root': output_root,
    })

    expected_root = project_root / output_root
    assert result.stage_name == 'prepare_pipeline'
    assert expected_root.is_dir()
    assert (expected_root / 'prepare_manifest.json').is_file()
    assert not (cwd / output_root).exists()


def test_manifest_is_limited_to_selected_sequences(tmp_path):
    source_root = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv'
    raw_root = tmp_path / 'raw'
    shutil.copytree(source_root / 'mini_seq', raw_root / 'mini_seq')
    shutil.copytree(source_root / 'mini_seq', raw_root / 'extra_seq')
    field_mapping = load_yaml_config(Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    result = run({
        'dataset_name': 'miluv',
        'raw_root': raw_root,
        'seq_ids': ['mini_seq'],
        'field_mapping': field_mapping,
        'output_root': tmp_path / 'prepare',
    })
    assert result.metadata['dataset_manifest']['sequence_count'] == 1
    assert [record['seq_id'] for record in result.metadata['dataset_manifest']['sequences']] == ['mini_seq']
    assert result.metadata['dataset_manifest']['sequences'][0]['scene_id'] == 'miluv:mini_seq'


def test_util_prepare_requires_flow_and_tof(tmp_path, monkeypatch):
    """必填测试：util prepare。\n\n验证 util prepare 的必填约束，\n确保 UTIL 序列包含 tof_raw 时被拒绝（防止静默丢弃测量数据）。
    """
    root = tmp_path / 'util_raw'
    seq_dir = root / 'util_seq'
    seq_dir.mkdir(parents=True)
    field_mapping = load_yaml_config(Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / 'util.yaml')['field_mapping']
    for filename, rows in {
        'imu.json': [{'timestamp': 0.0, 'ax': 0.1, 'ay': 0.2, 'gz': 0.3}],
        'uwb.json': [{'timestamp': 0.1, 'anchor_id': 1, 'range': 2.0, 'valid': True, 'quality': 0.9}],
        'flow.json': [{'timestamp': 0.2, 'dx': 0.0, 'dy': 0.0, 'quality': 1.0}],
        'tof.json': [{'timestamp': 0.3, 'range': 1.2, 'quality': 0.7}],
        'gt.json': [{'timestamp': 0.4, 'px': 0.0, 'py': 0.0, 'yaw': 0.0}],
    }.items():
        seq_dir.joinpath(filename).write_text(json.dumps(rows), encoding='utf-8')

    def _fake_read_util_sequence(seq_id, raw_root):
        return (
                {
                    'imu_raw': [{'timestamp': 0.0, 'ax': 0.1, 'ay': 0.2, 'gz': 0.3}],
                    'uwb_raw': [{'timestamp': 0.1, 'anchor_id': 1, 'range': 2.0, 'valid': True, 'quality': 0.9}],
                    'flow_raw': [{'timestamp': 0.2, 'dx': 0.0, 'dy': 0.0, 'quality': 1.0}],
                    'tof_raw': [{'timestamp': 0.3, 'range': 1.2, 'quality': 0.7}],
                    'gt_raw': [{'timestamp': 0.4, 'px': 0.0, 'py': 0.0, 'yaw': 0.0}],
                },
            {'dataset_name': 'util', 'seq_id': seq_id, 'seq_dir': str(Path(raw_root) / seq_id), 'streams': {}},
        )

    monkeypatch.setattr('liquidloc.pipelines.prepare_pipeline.read_util_sequence', _fake_read_util_sequence)

    try:
        run({
            'dataset_name': 'util',
            'raw_root': root,
            'seq_ids': ['util_seq'],
            'field_mapping': field_mapping,
            'output_root': tmp_path / 'prepare_util',
        })
    except ValueError as exc:
        assert 'tof_raw' in str(exc)
    else:
        raise AssertionError('Expected UTIL sequence with tof_raw to raise ValueError')


def test_flow_rows_as_vio_rows_preserves_source_t():
    """保持性测试：flow rows as vio rows。\n\n验证 flow rows as vio rows 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    rows = [{
        'timestamp': 0.25,
        'source_t': 0.2,
        'dx': 0.1,
        'dy': -0.05,
        'dyaw': 0.02,
        'quality': 0.9,
    }]

    converted = _flow_rows_as_vio_rows(rows)

    assert converted == [{
        'timestamp': 0.25,
        'dx': 0.1,
        'dy': -0.05,
        'dyaw': 0.02,
        'quality': 0.9,
        'tracked_features': 0,
        'reproj_err': 0.0,
        'missing_mask': [0, 0, 0, 0, 1, 1],
        'source_t': 0.2,
    }]


def test_resolve_scene_id_strips_surrounding_whitespace():
    """场景 ID 测试：resolve。\n\n验证 resolve 的场景 ID 处理，\n确保归一化和解析正确。
    """
    assert _resolve_scene_id({'scene_id': ' S(A0,N0,V0,G0,K4) '}, 'sim', 'mini_seq') == 'S(A0,N0,V0,G0,K4)'
    assert _resolve_scene_id(
        {'scene_id_by_seq': {'mini_seq': ' S(A0,N0,V0,G0,K4) '}},
        'sim',
        'mini_seq',
    ) == 'S(A0,N0,V0,G0,K4)'


def test_prepare_manifest_keeps_scene_parameters_for_whitespace_padded_scene_id(tmp_path, monkeypatch):
    """保持测试：prepare manifest。\n\n验证 prepare manifest 的保持行为，\n确保特定属性在处理过程中不变。
    """
    raw_root = tmp_path / 'sim_raw'
    (raw_root / 'mini_seq').mkdir(parents=True)

    def _fake_read_sim_sequence(seq_id, raw_root):
        bundle = {
            'imu_raw': [{'timestamp': 0.0, 'ax': 0.1, 'ay': 0.2, 'gz': 0.3}],
            'uwb_raw': [{'timestamp': 0.1, 'anchor_id': 1, 'range': 2.0, 'valid': True, 'quality': 0.9}],
            'vio_raw': [{'timestamp': 0.2, 'dx': 0.0, 'dy': 0.0, 'dyaw': 0.0, 'quality': 1.0, 'tracked_features': 8, 'reproj_err': 0.1}],
            'gt_raw': [{'timestamp': 0.3, 'px': 0.0, 'py': 0.0, 'yaw': 0.0}],
        }
        read_report = {
            'dataset_name': 'sim',
            'seq_id': seq_id,
            'seq_dir': str(raw_root),
            'streams': {key: len(value) for key, value in bundle.items() if key.endswith('_raw') and isinstance(value, list)},
            'is_complete': True,
        }
        return bundle, read_report

    monkeypatch.setattr('liquidloc.pipelines.prepare_pipeline.read_sim_sequence', _fake_read_sim_sequence)
    # K6/G0 物化合同对 tmp_path 内的 SIM raw 在 prepare 阶段强校验 anchor_layout.json
    # 等文件, 此测试只关心 scene_parameters 保留语义, 直接旁路合同闸门.
    monkeypatch.setattr(
        'liquidloc.pipelines.prepare_pipeline._enforce_sim_materialized_contract',
        lambda raw_root: None,
    )

    result = run({
        'dataset_name': 'sim',
        'raw_root': raw_root,
        'seq_ids': ['mini_seq'],
        'scene_id': ' S(A0,N0,V0,G0,K4) ',
        'output_root': tmp_path / 'prepare_sim',
    })

    seq_payload = result.metadata['sequences']['mini_seq']
    dataset_seq = result.metadata['dataset_manifest']['sequences'][0]
    import gzip
    import pickle
    with gzip.open(tmp_path / 'prepare_sim' / 'mini_seq_events.pkl.gz', 'rb') as f:
        events = pickle.load(f)
    assert seq_payload['scene_id'] == 'S(A0,N0,V0,G0,K4)'
    assert dataset_seq['scene_id'] == 'S(A0,N0,V0,G0,K4)'
    assert seq_payload['scene_parameters'] == dataset_seq['scene_parameters']
    assert events[0]['meta']['scene_parameters'] == seq_payload['scene_parameters']


def test_invalid_case(tmp_path, monkeypatch):
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    # 缺 scene_id 应触发 ValueError; 不应在更早的 K6/G0 物化合同上 abort.
    monkeypatch.setattr(
        'liquidloc.pipelines.prepare_pipeline._enforce_sim_materialized_contract',
        lambda raw_root: None,
    )
    try:
        run({'dataset_name': 'sim', 'raw_root': tmp_path, 'seq_ids': ['mini_seq']})
    except ValueError as exc:
        assert 'scene_id' in str(exc)
    else:
        raise AssertionError('Expected missing sim scene_id to raise ValueError')


def test_duplicate_seq_ids_rejected(tmp_path):
    """拒绝测试：duplicate seq ids。\n\n验证被测功能对不合法的 duplicate seq ids 输入正确抛出异常，\n防止无效参数通过验证。
    """
    root = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv'
    field_mapping = load_yaml_config(Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    try:
        run({
            'dataset_name': 'miluv',
            'raw_root': root,
            'seq_ids': ['mini_seq', 'mini_seq'],
            'field_mapping': field_mapping,
            'output_root': tmp_path / 'prepare',
        })
    except ValueError as exc:
        assert 'duplicates' in str(exc)
    else:
        raise AssertionError('Expected duplicate seq_ids to raise ValueError')