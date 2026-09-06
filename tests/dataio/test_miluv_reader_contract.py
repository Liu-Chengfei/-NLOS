
"""MILUV 数据读取器合同测试模块。

文件职责：验证 MILUV 读取器的数据合同，
包括本地锚点布局合同与官方元数据的分离。

测试覆盖范围：
- 本地 fixture 数据包合同
- 第二个 fixture 数据包合同
- 官方元数据不替代本地锚点布局合同
- 缺失原始序列的就绪检查
- 官方元数据不满足本地锚点布局合同
- 审计报告：本地合同满足
- 审计报告：官方元数据与本地合同分离

被测模块：liquidloc.dataio.readers.miluv_reader"""

import shutil
from pathlib import Path

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.dataio.readers.miluv_reader import (
    audit_miluv_official_sample,
    inspect_miluv_raw_readiness,
    read_miluv_sequence,
)


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


def test_read_miluv_sequence_contract():
    root = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv'
    field_mapping = load_yaml_config(Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    bundle, report = read_miluv_sequence('mini_seq', root, field_mapping)
    assert bundle['imu_raw'][0]['ax'] == 0.1
    assert any(row['timestamp'] == 0.265 for row in bundle['gt_raw'])
    assert report['streams']['gt_raw'] == 3
    assert report['is_complete'] is True


def test_read_miluv_sequence_contract_second_local_fixture():
    root = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv'
    field_mapping = load_yaml_config(Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    bundle, report = read_miluv_sequence('mini_seq_02', root, field_mapping)
    assert bundle['imu_raw'][0]['ax'] == 0.15
    assert [row['timestamp'] for row in bundle['gt_raw']] == [0.02, 0.12, 0.19]
    assert report['streams']['gt_raw'] == 3
    assert report['is_complete'] is True


def test_read_miluv_sequence_does_not_use_official_metadata_for_local_anchor_contract(tmp_path):
    fixture_root = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv' / 'mini_seq'
    raw_root = tmp_path / 'miluv'
    seq_dir = raw_root / 'default_3_random_0'
    seq_dir.mkdir(parents=True)
    for name in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
        shutil.copy2(fixture_root / name, seq_dir / name)

    cfg_dir = raw_root / 'config' / 'uwb'
    cfg_dir.mkdir(parents=True)
    (raw_root / 'config' / 'experiments.csv').write_text(_OFFICIAL_EXPERIMENTS_CSV, encoding='utf-8')
    (cfg_dir / 'anchors.yaml').write_text(_OFFICIAL_ANCHORS_YAML, encoding='utf-8')

    field_mapping = load_yaml_config(Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / 'miluv.yaml')['field_mapping']
    bundle, report = read_miluv_sequence('default_3_random_0', raw_root, field_mapping)

    assert report['anchor_layout_metadata_available'] is True
    assert report['anchor_layout_available'] is False
    assert report['anchor_layout_teacher_ready'] is False
    assert report['anchor_layout_metadata_source'] == 'miluv_official_experiments_csv+anchors_yaml'
    assert report['anchor_layout_position_dim'] == 3
    assert 'anchor_layout_raw' not in bundle
    assert bundle['anchor_layout_metadata_raw']['anchor_ids'] == [0, 1, 2, 3, 4, 5]


def test_inspect_miluv_raw_readiness_reports_missing_raw_sequence(tmp_path):
    raw_root = tmp_path / 'miluv_empty'
    raw_root.mkdir()
    (raw_root / '.gitkeep').write_text('', encoding='utf-8')

    report = inspect_miluv_raw_readiness(raw_root)

    assert report['status'] == 'not_ready'
    assert report['gate_action'] == 'skipped'
    assert report['ready_for_end_to_end_training_smoke'] is False
    assert report['sequence_count'] == 0
    assert report['ready_sequence_count'] == 0
    assert report['reasons'] == ['missing_raw_sequence']


def test_inspect_miluv_raw_readiness_does_not_accept_official_metadata_for_local_anchor_contract(tmp_path):
    fixture_root = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv' / 'mini_seq'
    raw_root = tmp_path / 'miluv'
    seq_dir = raw_root / 'default_3_random_0'
    seq_dir.mkdir(parents=True)
    for name in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
        shutil.copy2(fixture_root / name, seq_dir / name)

    cfg_dir = raw_root / 'config' / 'uwb'
    cfg_dir.mkdir(parents=True)
    (raw_root / 'config' / 'experiments.csv').write_text(_OFFICIAL_EXPERIMENTS_CSV, encoding='utf-8')
    (cfg_dir / 'anchors.yaml').write_text(_OFFICIAL_ANCHORS_YAML, encoding='utf-8')

    report = inspect_miluv_raw_readiness(raw_root)
    seq_report = report['sequences']['default_3_random_0']

    assert report['status'] == 'not_ready'
    assert report['gate_action'] == 'skipped'
    assert report['ready_for_end_to_end_training_smoke'] is False
    assert report['ready_sequence_count'] == 0
    assert seq_report['status'] == 'not_ready'
    assert seq_report['missing_streams'] == []
    assert seq_report['reasons'] == ['missing_anchor_layout']
    assert seq_report['anchor_layout'] == {
        'ready': False,
        'source': None,
        'source_paths': [],
        'metadata_available': False,
        'anchor_ids_available': False,
        'anchor_positions_available': False,
        'position_dim': None,
        'blockers': ['missing_anchor_layout_json'],
    }


def test_audit_miluv_official_sample_reports_local_contract_from_local_anchor_layout():
    raw_root = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv'

    report = audit_miluv_official_sample('mini_seq', raw_root)

    assert report['status'] == 'local_contract_satisfied'
    assert report['official_anchor_metadata']['metadata_available'] is False
    assert report['local_anchor_layout_contract']['ready'] is True
    assert report['local_anchor_layout_contract']['position_dim'] == 2
    assert report['official_metadata_used_for_local_contract'] is False
    assert report['local_contract_still_needed'] == []


def test_audit_miluv_official_sample_keeps_official_metadata_separate_from_local_contract(tmp_path):
    fixture_root = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv' / 'mini_seq'
    raw_root = tmp_path / 'miluv'
    seq_dir = raw_root / 'default_3_random_0'
    seq_dir.mkdir(parents=True)
    for name in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
        shutil.copy2(fixture_root / name, seq_dir / name)

    cfg_dir = raw_root / 'config' / 'uwb'
    cfg_dir.mkdir(parents=True)
    (raw_root / 'config' / 'experiments.csv').write_text(_OFFICIAL_EXPERIMENTS_CSV, encoding='utf-8')
    (cfg_dir / 'anchors.yaml').write_text(_OFFICIAL_ANCHORS_YAML, encoding='utf-8')

    report = audit_miluv_official_sample('default_3_random_0', raw_root)

    assert report['status'] == 'official_metadata_present_local_contract_missing'
    assert report['official_anchor_metadata']['metadata_available'] is True
    assert report['official_anchor_metadata']['position_dim'] == 3
    assert report['local_anchor_layout_contract']['ready'] is False
    assert report['local_anchor_layout_contract']['metadata_available'] is False
    assert report['official_metadata_used_for_local_contract'] is False
    assert report['local_contract_still_needed'] == ['missing_anchor_layout_json']
