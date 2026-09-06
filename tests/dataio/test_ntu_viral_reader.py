
"""NTU VIRAL 数据读取器测试模块。

文件职责：验证 read_ntu_viral_sequence 能正确读取
NTU VIRAL 数据集的 JSON 原始数据包，
以及 inspect_ntu_viral_raw_readiness 的就绪检查。

测试覆盖范围：
- 正常场景：读取完整 JSON 数据包
- 缺失文件的就绪检查
- 缺失必需字段的就绪检查

被测模块：liquidloc.dataio.readers.ntu_viral_reader"""

import json
from pathlib import Path

from liquidloc.dataio.readers.ntu_viral_reader import inspect_ntu_viral_raw_readiness, read_ntu_viral_sequence


def test_read_ntu_viral_sequence_reads_json_raw_bundle(tmp_path):
    seq_dir = tmp_path / 'ntu_seq'
    seq_dir.mkdir()
    for filename, rows in {
        'imu.json': [{'timestamp': 0.0, 'ax': 0.1, 'ay': 0.0, 'gz': 0.01}],
        'uwb.json': [{'timestamp': 0.1, 'anchor_id': 1, 'range': 2.0, 'valid': True, 'quality': 0.9}],
        'vio.json': [{'timestamp': 0.2, 'dx': 0.05, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.8, 'tracked_features': 40, 'reproj_err': 0.5}],
        'gt.json': [{'timestamp': 0.3, 'px': 0.0, 'py': 0.0, 'yaw': 0.0}],
    }.items():
        (seq_dir / filename).write_text(json.dumps(rows), encoding='utf-8')

    bundle, report = read_ntu_viral_sequence('ntu_seq', tmp_path)

    assert bundle['imu_raw'][0]['ax'] == 0.1
    assert report['dataset_name'] == 'ntu_viral'
    assert report['streams']['gt_raw'] == 1
    assert report['is_complete'] is True


def test_inspect_ntu_viral_raw_readiness_blocks_missing_files(tmp_path):
    seq_dir = tmp_path / 'ntu_seq'
    seq_dir.mkdir()
    (seq_dir / 'imu.json').write_text(json.dumps([{'timestamp': 0.0, 'ax': 0.1, 'ay': 0.0, 'gz': 0.01}]), encoding='utf-8')
    (seq_dir / 'uwb.json').write_text(json.dumps([{'timestamp': 0.1, 'anchor_id': 1, 'range': 2.0, 'valid': True, 'quality': 0.9}]), encoding='utf-8')

    report = inspect_ntu_viral_raw_readiness(tmp_path)
    seq_report = report['sequences']['ntu_seq']

    assert report['status'] == 'not_ready'
    assert report['gate_action'] == 'skipped'
    assert seq_report['status'] == 'not_ready'
    assert seq_report['reasons'] == ['missing_required_streams']
    assert 'vio' in seq_report['missing_streams']
    assert 'gt' in seq_report['missing_streams']


def test_inspect_ntu_viral_raw_readiness_blocks_missing_required_fields(tmp_path):
    seq_dir = tmp_path / 'ntu_seq'
    seq_dir.mkdir()
    (seq_dir / 'imu.json').write_text(json.dumps([{'timestamp': 0.0, 'ax': 0.1, 'ay': 0.0}]), encoding='utf-8')
    (seq_dir / 'uwb.json').write_text(json.dumps([{'timestamp': 0.1, 'anchor_id': 1, 'range': 2.0, 'valid': True, 'quality': 0.9}]), encoding='utf-8')
    (seq_dir / 'vio.json').write_text(json.dumps([{'timestamp': 0.2, 'dx': 0.05, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.8, 'tracked_features': 40, 'reproj_err': 0.5}]), encoding='utf-8')
    (seq_dir / 'gt.json').write_text(json.dumps([{'timestamp': 0.3, 'px': 0.0, 'py': 0.0, 'yaw': 0.0}]), encoding='utf-8')

    report = inspect_ntu_viral_raw_readiness(tmp_path)
    seq_report = report['sequences']['ntu_seq']

    assert report['status'] == 'not_ready'
    assert seq_report['status'] == 'not_ready'
    assert 'imu' in seq_report['missing_streams']
    assert seq_report['required_streams']['imu']['missing_fields'] == ['gz']

