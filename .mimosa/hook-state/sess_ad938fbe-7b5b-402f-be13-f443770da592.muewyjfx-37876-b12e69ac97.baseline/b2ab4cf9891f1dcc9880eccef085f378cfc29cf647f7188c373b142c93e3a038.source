
"""Util 数据读取器测试模块。

文件职责：验证 read_util_sequence 能正确读取
util 数据集的 IMU、UWB、flow、GT 数据流，以及可选的 ToF 数据流。

测试覆盖范围：
- 正常场景：读取所有必需数据流
- 可选 ToF：tof.json 存在时读入，不存在时跳过
- 空数据流在报告中标记

被测模块：liquidloc.dataio.readers.util_reader"""

import json

from liquidloc.dataio.readers.util_reader import read_util_sequence


def _write_records(path, rows):
    path.write_text(json.dumps(rows), encoding='utf-8')


def test_read_util_sequence_reads_all_required_streams(tmp_path):
    seq_dir = tmp_path / 'util_seq'
    seq_dir.mkdir()
    _write_records(seq_dir / 'imu.json', [{'timestamp': 0.0, 'ax': 0.1}])
    _write_records(seq_dir / 'uwb.json', [{'timestamp': 0.1, 'range': 2.0}])
    _write_records(seq_dir / 'flow.json', [{'timestamp': 0.2, 'dx': 0.1}])
    _write_records(seq_dir / 'tof.json', [{'timestamp': 0.3, 'range': 1.2}])
    _write_records(seq_dir / 'gt.json', [{'timestamp': 0.4, 'px': 0.0}])

    bundle, report = read_util_sequence('util_seq', tmp_path)

    assert bundle['flow_raw'][0]['dx'] == 0.1
    assert bundle['tof_raw'][0]['range'] == 1.2
    assert report['streams'] == {
        'imu_raw': 1,
        'uwb_raw': 1,
        'flow_raw': 1,
        'tof_raw': 1,
        'gt_raw': 1,
    }
    assert report['is_complete'] is True
    assert report['tof_available'] is True


def test_read_util_sequence_works_without_tof(tmp_path):
    seq_dir = tmp_path / 'util_no_tof'
    seq_dir.mkdir()
    _write_records(seq_dir / 'imu.json', [{'timestamp': 0.0, 'ax': 0.1}])
    _write_records(seq_dir / 'uwb.json', [{'timestamp': 0.1, 'range': 2.0}])
    _write_records(seq_dir / 'flow.json', [{'timestamp': 0.2, 'dx': 0.1}])
    _write_records(seq_dir / 'gt.json', [{'timestamp': 0.4, 'px': 0.0}])

    bundle, report = read_util_sequence('util_no_tof', tmp_path)

    assert 'tof_raw' not in bundle
    assert report['is_complete'] is True
    assert report['tof_available'] is False


def test_read_util_sequence_accepts_empty_stream_lists(tmp_path):
    seq_dir = tmp_path / 'util_empty'
    seq_dir.mkdir()
    for filename in ('imu.json', 'uwb.json', 'flow.json', 'gt.json'):
        _write_records(seq_dir / filename, [])

    bundle, report = read_util_sequence('util_empty', tmp_path)

    assert bundle['imu_raw'] == []
    assert bundle['uwb_raw'] == []
    assert bundle['flow_raw'] == []
    assert bundle['gt_raw'] == []
    assert report['is_complete'] is False

