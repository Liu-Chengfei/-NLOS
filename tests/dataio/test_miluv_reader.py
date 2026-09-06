
"""MILUV 数据读取器测试模块。

文件职责：验证 read_miluv_sequence 能正确读取
MILUV 数据集的原始数据包和锚点布局，
以及 inspect_miluv_raw_readiness 的就绪检查。

测试覆盖范围：
- 读取第一个 fixture 数据包和锚点布局
- 读取第二个 fixture 数据包
- 缺失锚点布局的就绪检查
- 空 field_mapping 拒绝

被测模块：liquidloc.dataio.readers.miluv_reader"""

import shutil
from pathlib import Path

import pytest

from liquidloc.common.config_utils import load_yaml_config
from liquidloc.dataio.readers.miluv_reader import inspect_miluv_raw_readiness, read_miluv_sequence


def _miluv_field_mapping():
    config_path = Path(__file__).resolve().parents[2] / 'configs' / 'datasets' / 'miluv.yaml'
    return load_yaml_config(config_path)['field_mapping']


def test_read_miluv_sequence_reads_fixture_bundle_and_anchor_layout():
    raw_root = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv'

    bundle, report = read_miluv_sequence('mini_seq', raw_root, _miluv_field_mapping())

    assert bundle['imu_raw'][0]['ax'] == 0.1
    assert bundle['uwb_raw'][0]['anchor_id'] == 0
    assert any(row['timestamp'] == 0.265 for row in bundle['gt_raw'])
    assert bundle['anchor_layout_raw']['anchor_ids'] == [0, 1]
    assert report['streams'] == {
        'imu_raw': 2,
        'uwb_raw': 2,
        'vio_raw': 2,
        'gt_raw': 3,
    }
    assert report['is_complete'] is True
    assert report['anchor_layout_available'] is True
    assert report['anchor_layout_teacher_ready'] is True


def test_read_miluv_sequence_reads_second_fixture_bundle_and_anchor_layout():
    raw_root = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv'

    bundle, report = read_miluv_sequence('mini_seq_02', raw_root, _miluv_field_mapping())

    assert bundle['imu_raw'][0]['ax'] == 0.15
    assert bundle['uwb_raw'][0]['anchor_id'] == 1
    assert [row['timestamp'] for row in bundle['gt_raw']] == [0.02, 0.12, 0.19]
    assert bundle['anchor_layout_raw']['anchor_ids'] == [0, 1]
    assert report['streams'] == {
        'imu_raw': 2,
        'uwb_raw': 2,
        'vio_raw': 2,
        'gt_raw': 3,
    }
    assert report['is_complete'] is True
    assert report['anchor_layout_available'] is True
    assert report['anchor_layout_teacher_ready'] is True


def test_inspect_miluv_raw_readiness_reports_missing_anchor_layout(tmp_path):
    fixture_root = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv' / 'mini_seq'
    raw_root = tmp_path / 'miluv'
    seq_dir = raw_root / 'seq_without_anchor_layout'
    seq_dir.mkdir(parents=True)
    for filename in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
        shutil.copy2(fixture_root / filename, seq_dir / filename)

    report = inspect_miluv_raw_readiness(raw_root)
    seq_report = report['sequences']['seq_without_anchor_layout']

    assert report['status'] == 'not_ready'
    assert report['gate_action'] == 'skipped'
    assert report['ready_sequence_count'] == 0
    assert seq_report['missing_streams'] == []
    assert seq_report['anchor_layout']['ready'] is False
    assert seq_report['reasons'] == ['missing_anchor_layout']


def test_read_miluv_sequence_rejects_empty_field_mapping():
    raw_root = Path(__file__).resolve().parents[1] / 'fixtures' / 'datasets' / 'miluv'

    with pytest.raises(ValueError, match='field_mapping must be a non-empty mapping'):
        read_miluv_sequence('mini_seq', raw_root, {})
