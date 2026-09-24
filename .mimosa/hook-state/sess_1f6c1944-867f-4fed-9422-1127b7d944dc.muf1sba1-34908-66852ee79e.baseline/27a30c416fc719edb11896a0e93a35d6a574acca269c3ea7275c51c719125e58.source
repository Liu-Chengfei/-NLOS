
"""数据集检查（dataset_checks）测试模块。

文件职责：验证 run_dataset_checks 和 inspect_layout_family_coverage
能正确检查数据完整性、缺失流和布局族覆盖。

测试覆盖范围：
- 完整原始数据包通过检查
- 空/缺失/非单调流报告
- 非列表必需流拒绝
- 序列目录缺失文件报告
- util 流程额外必需流
- 缺失路径拒绝
- 布局族覆盖报告
- 缺失/损坏/未解析的锚点布局报告

被测模块：liquidloc.dataio.manifests.dataset_checks"""

import json

import pytest

from liquidloc.dataio.manifests.dataset_checks import (
    REQUIRED_SEQ_FILES,
    UTIL_REQUIRED_RAW_KEYS,
    UTIL_REQUIRED_SEQ_FILES,
    inspect_layout_family_coverage,
    run_dataset_checks,
)


def _write_records(path, rows):
    path.write_text(json.dumps(rows), encoding='utf-8')


def test_run_dataset_checks_accepts_complete_raw_bundle():
    report = run_dataset_checks(
        {
            'imu_raw': [{'timestamp': 0.0, 'ax': 0.1}],
            'uwb_raw': [{'timestamp': 0.1, 'anchor_id': 1}],
            'vio_raw': [{'timestamp': 0.2, 'dx': 0.1}],
            'gt_raw': [{'timestamp': 0.3, 'px': 0.0}],
        }
    )

    assert report == {
        'kind': 'raw_bundle',
        'missing_streams': [],
        'empty_streams': [],
        'bad_streams': [],
        'is_valid': True,
    }


def test_run_dataset_checks_reports_empty_missing_and_non_monotonic_streams():
    report = run_dataset_checks(
        {
            'imu_raw': [{'timestamp': 0.1}, {'timestamp': 0.0}],
            'uwb_raw': [],
            'vio_raw': [{'timestamp': 0.2}],
        }
    )

    assert report['kind'] == 'raw_bundle'
    assert report['missing_streams'] == ['gt_raw']
    assert report['empty_streams'] == ['uwb_raw']
    assert report['bad_streams'] == ['imu_raw']
    assert report['is_valid'] is False


def test_run_dataset_checks_rejects_non_list_required_streams():
    report = run_dataset_checks(
        {
            'imu_raw': {'timestamp': 0.0},
            'uwb_raw': [{'timestamp': 0.1}],
            'vio_raw': [{'timestamp': 0.2}],
            'gt_raw': [{'timestamp': 0.3}],
        }
    )

    assert report['kind'] == 'raw_bundle'
    assert report['bad_streams'] == ['imu_raw']
    assert report['is_valid'] is False


def test_run_dataset_checks_reports_bad_sequence_directory(tmp_path):
    seq_dir = tmp_path / 'seq0'
    seq_dir.mkdir()
    _write_records(seq_dir / 'imu.json', [{'timestamp': 0.0}])
    _write_records(seq_dir / 'uwb.json', [{'timestamp': 0.1}])
    _write_records(seq_dir / 'gt.json', [{'timestamp': 0.2}])

    report = run_dataset_checks(tmp_path)

    assert report['kind'] == 'path'
    assert report['sequence_count'] == 1
    assert report['bad_sequences'] == [{'seq_id': 'seq0', 'missing_files': ['vio.json']}]
    assert report['is_valid'] is False


def test_run_dataset_checks_requires_util_flow_streams():
    report = run_dataset_checks(
        {
            'imu_raw': [{'timestamp': 0.0, 'ax': 0.1}],
            'uwb_raw': [{'timestamp': 0.1, 'anchor_id': 1}],
            'flow_raw': [{'timestamp': 0.2, 'dx': 0.1}],
            'gt_raw': [{'timestamp': 0.3, 'px': 0.0}],
        },
        required_raw_keys=UTIL_REQUIRED_RAW_KEYS,
    )

    assert report['missing_streams'] == []
    assert report['empty_streams'] == []
    assert report['is_valid'] is True


def test_run_dataset_checks_requires_util_sequence_files(tmp_path):
    seq_dir = tmp_path / 'util_seq'
    seq_dir.mkdir()
    for filename in REQUIRED_SEQ_FILES:
        _write_records(seq_dir / filename, [{'timestamp': 0.0}])

    report = run_dataset_checks(tmp_path, required_seq_files=UTIL_REQUIRED_SEQ_FILES)

    assert report['kind'] == 'path'
    assert report['sequence_count'] == 1
    assert report['bad_sequences'] == [{'seq_id': 'util_seq', 'missing_files': ['flow.json']}]
    assert report['is_valid'] is False


def test_run_dataset_checks_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match='Dataset path is not a directory'):
        run_dataset_checks(tmp_path / 'missing_root')


def test_inspect_layout_family_coverage_reports_distinct_base_layouts(tmp_path):
    for seq_id, family_id in (('seq_a', 'family_alpha'), ('seq_b', 'family_alpha'), ('seq_c', 'family_beta')):
        seq_dir = tmp_path / seq_id
        seq_dir.mkdir()
        (seq_dir / 'anchor_layout.json').write_text(
            json.dumps({'base_layout_id': family_id, 'layout_id': f'{family_id}__materialized'}),
            encoding='utf-8',
        )

    report = inspect_layout_family_coverage(tmp_path)

    assert report['sequence_count'] == 3
    assert report['family_count'] == 2
    assert report['sequence_family_map'] == {
        'seq_a': 'family_alpha',
        'seq_b': 'family_alpha',
        'seq_c': 'family_beta',
    }
    assert report['family_to_sequences'] == {
        'family_alpha': ['seq_a', 'seq_b'],
        'family_beta': ['seq_c'],
    }
    assert report['is_valid'] is True


def test_inspect_layout_family_coverage_reports_missing_and_bad_metadata(tmp_path):
    (tmp_path / 'seq_missing').mkdir()
    seq_bad = tmp_path / 'seq_bad'
    seq_bad.mkdir()
    (seq_bad / 'anchor_layout.json').write_text('{not-json}', encoding='utf-8')
    seq_unresolved = tmp_path / 'seq_unresolved'
    seq_unresolved.mkdir()
    (seq_unresolved / 'anchor_layout.json').write_text(json.dumps({'layout_id': ''}), encoding='utf-8')

    report = inspect_layout_family_coverage(tmp_path)

    assert report['family_count'] == 0
    assert report['missing_anchor_layout_seq_ids'] == ['seq_missing']
    assert report['bad_anchor_layout_seq_ids'] == ['seq_bad']
    assert report['unresolved_family_seq_ids'] == ['seq_unresolved']
    assert report['is_valid'] is False
