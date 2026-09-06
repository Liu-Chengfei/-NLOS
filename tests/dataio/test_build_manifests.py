
"""清单构建器（build_manifests）测试模块。

文件职责：验证 build_manifests 能正确扫描数据目录，
收集序列信息、缺失文件和场景清单。

测试覆盖范围：
- 正常场景：完整和不完整序列的清单
- 空数据目录
- util 流程的额外必需文件
- 异常场景：缺失根目录

被测模块：liquidloc.dataio.manifests.build_manifests"""

import json

import pytest

from liquidloc.dataio.manifests.build_manifests import REQUIRED_STREAMS, UTIL_REQUIRED_STREAMS, build_manifests


def _write_records(path, rows=None):
    path.write_text(json.dumps(rows if rows is not None else []), encoding='utf-8')


def test_build_manifests_collects_sequences_and_missing_files(tmp_path):
    complete_seq = tmp_path / 'S(A0,N0,V0,K0,M0)'
    complete_seq.mkdir()
    for filename in REQUIRED_STREAMS:
        _write_records(complete_seq / filename)

    incomplete_seq = tmp_path / 'seq_without_scene'
    incomplete_seq.mkdir()
    for filename in ('imu.json', 'uwb.json', 'gt.json'):
        _write_records(incomplete_seq / filename)

    dataset_manifest, scene_manifest = build_manifests(tmp_path)

    assert dataset_manifest['sequence_count'] == 2
    assert [record['seq_id'] for record in dataset_manifest['sequences']] == [
        'S(A0,N0,V0,K0,M0)',
        'seq_without_scene',
    ]
    assert dataset_manifest['sequences'][0]['is_complete'] is True
    assert dataset_manifest['sequences'][1]['missing_files'] == ['vio.json']
    assert scene_manifest['scene_count'] == 2
    assert scene_manifest['scenes'] == [
        {'scene_id': 'S(A0,N0,V0,K0,M0)', 'seq_ids': ['S(A0,N0,V0,K0,M0)']},
        {'scene_id': 'seq_without_scene', 'seq_ids': ['seq_without_scene']},
    ]


def test_build_manifests_handles_empty_dataset_root(tmp_path):
    dataset_manifest, scene_manifest = build_manifests(tmp_path)

    assert dataset_manifest['sequence_count'] == 0
    assert dataset_manifest['sequences'] == []
    assert scene_manifest == {'scene_count': 0, 'scenes': []}


def test_build_manifests_can_require_util_flow(tmp_path):
    seq_dir = tmp_path / 'util_seq'
    seq_dir.mkdir()
    for filename in REQUIRED_STREAMS:
        _write_records(seq_dir / filename)

    dataset_manifest, scene_manifest = build_manifests(tmp_path, required_streams=UTIL_REQUIRED_STREAMS)

    assert dataset_manifest['sequence_count'] == 1
    assert dataset_manifest['sequences'][0]['missing_files'] == ['flow.json']
    assert dataset_manifest['sequences'][0]['is_complete'] is False
    assert scene_manifest['scene_count'] == 1


def test_build_manifests_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError, match='Data root is not a directory'):
        build_manifests(tmp_path / 'missing_root')
