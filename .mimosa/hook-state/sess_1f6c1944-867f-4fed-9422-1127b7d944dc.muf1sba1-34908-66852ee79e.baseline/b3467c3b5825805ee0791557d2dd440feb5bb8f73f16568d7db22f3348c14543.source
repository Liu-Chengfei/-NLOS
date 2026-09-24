
"""仿真数据读取器（sim_reader）测试模块。

文件职责：验证 read_sim_sequence 能正确读取仿真序列的
IMU、UWB、VIO、GT 数据流和可选的锚点布局。

测试覆盖范围：
- 正常场景：读取所有必需数据流
- 空数据流返回空列表
- 可选锚点布局读取
- 异常场景：非法 payload 形状

被测模块：liquidloc.dataio.readers.sim_reader"""

import json

import pytest

from liquidloc.dataio.readers.sim_reader import read_sim_sequence


def _write_records(path, rows):
    path.write_text(json.dumps(rows), encoding='utf-8')


def test_read_sim_sequence_reads_all_required_streams(tmp_path):
    seq_dir = tmp_path / 'seq0'
    seq_dir.mkdir()
    _write_records(seq_dir / 'imu.json', [{'timestamp': 0.0, 'ax': 0.1}])
    _write_records(seq_dir / 'uwb.json', [{'timestamp': 0.1, 'range': 2.0}])
    _write_records(seq_dir / 'vio.json', [{'timestamp': 0.2, 'dx': 0.1}])
    _write_records(seq_dir / 'gt.json', [{'timestamp': 0.3, 'px': 0.0}])

    bundle, report = read_sim_sequence('seq0', tmp_path)

    assert bundle['imu_raw'][0]['ax'] == 0.1
    assert bundle['uwb_raw'][0]['range'] == 2.0
    assert bundle['vio_raw'][0]['dx'] == 0.1
    assert bundle['gt_raw'][0]['px'] == 0.0


def test_read_sim_sequence_accepts_empty_stream_lists(tmp_path):
    seq_dir = tmp_path / 'seq_empty'
    seq_dir.mkdir()
    for filename in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
        _write_records(seq_dir / filename, [])

    bundle, report = read_sim_sequence('seq_empty', tmp_path)

    assert bundle['imu_raw'] == []
    assert bundle['uwb_raw'] == []
    assert bundle['vio_raw'] == []
    assert bundle['gt_raw'] == []


def test_read_sim_sequence_reads_optional_anchor_layout(tmp_path):
    seq_dir = tmp_path / 'seq_with_anchor_layout'
    seq_dir.mkdir()
    for filename in ('imu.json', 'uwb.json', 'vio.json', 'gt.json'):
        _write_records(seq_dir / filename, [{'timestamp': 0.0}])
    (seq_dir / 'anchor_layout.json').write_text(
        json.dumps({'anchor_ids': [0], 'anchor_positions': [[1.0, 2.0]]}),
        encoding='utf-8',
    )

    bundle, report = read_sim_sequence('seq_with_anchor_layout', tmp_path)

    assert bundle['anchor_layout_raw'] == {'anchor_ids': [0], 'anchor_positions': [[1.0, 2.0]]}


def test_read_sim_sequence_rejects_invalid_payload_shape(tmp_path):
    seq_dir = tmp_path / 'seq_bad_payload'
    seq_dir.mkdir()
    (seq_dir / 'imu.json').write_text(json.dumps({'timestamp': 0.0}), encoding='utf-8')
    _write_records(seq_dir / 'uwb.json', [])
    _write_records(seq_dir / 'vio.json', [])
    _write_records(seq_dir / 'gt.json', [])

    with pytest.raises(TypeError, match='Expected a list\\[dict\\] payload'):
        read_sim_sequence('seq_bad_payload', tmp_path)
