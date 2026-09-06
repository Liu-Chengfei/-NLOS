
"""字段映射器（field_mapper）测试模块。

文件职责：验证 map_external_fields 能将外部字段名
映射为内部标准字段名。

测试覆盖范围：
- 正常场景：完整映射
- 边界场景：缺失源字段报告
- 异常场景：空原始数据包拒绝

被测模块：liquidloc.dataio.adapters.field_mapper"""

from liquidloc.dataio.adapters.field_mapper import map_external_fields


def test_normal_case():
    raw_bundle = {
        'imu_raw': [{'timestamp': 0.0, 'acc_x': 0.1, 'acc_y': 0.2, 'gyro_z': 0.3}],
        'uwb_raw': [{'timestamp': 0.1, 'aid': 1, 'dist': 2.0, 'is_valid': True, 'q': 0.9}],
    }
    mapping = {
        'imu': {'timestamp': 'timestamp', 'acc_x': 'ax', 'acc_y': 'ay', 'gyro_z': 'gz'},
        'uwb': {'timestamp': 'timestamp', 'aid': 'anchor_id', 'dist': 'range', 'is_valid': 'valid', 'q': 'quality'},
    }
    mapped, report = map_external_fields(raw_bundle, mapping)
    assert mapped['imu_raw'][0]['ax'] == 0.1
    assert mapped['uwb_raw'][0]['anchor_id'] == 1
    assert report['is_complete'] is True


def test_boundary_case():
    raw_bundle = {'imu_raw': [{'timestamp': 0.0, 'acc_x': 0.1}]}
    mapping = {'imu': {'timestamp': 'timestamp', 'acc_x': 'ax', 'missing': 'ay'}}
    mapped, report = map_external_fields(raw_bundle, mapping)
    assert mapped['imu_raw'][0]['ax'] == 0.1
    assert report['is_complete'] is False
    assert report['streams']['imu']['missing_source_fields']['missing'] == 1


def test_invalid_case():
    try:
        map_external_fields({}, {'imu': {'timestamp': 'timestamp'}})
    except ValueError as exc:
        assert 'non-empty' in str(exc)
    else:
        raise AssertionError('Expected empty raw bundle to raise ValueError')
