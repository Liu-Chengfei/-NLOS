from __future__ import annotations

"""H24d 真改 §4.3.1 L1002 遮挡物类型枚举测试模块。

文件职责：验证 apply_nlos_level 的 occluder_type 字段真生效：
- 不同 occluder_type 产生不同 selection_key → 不同 selected_indices 散布
- 缺省时回退 machine_body（向后兼容旧 yaml/旧测试）
- 非枚举值由协议层 _validate_axis_param_semantic_range 在 load_scene_axis_protocol
  阶段抛 ValueError（在本测试模块单独构造非法 cfg 时由 _resolve_level_cfg 触发）
- nlos_report 写回 occluder_type 字段供下游审计/测试消费

被测模块：liquidloc.scenarios.nlos_levels.apply_nlos_level
        liquidloc.protocol.scene_axis_protocol._validate_axis_param_semantic_range
"""

import pytest

from liquidloc.scenarios.nlos_levels import apply_nlos_level
from liquidloc.protocol.scene_axis_protocol import _NLOS_OCCLUDER_TYPES


def _events_with_occluder(scene_id: str, seq_id: str = "occluder_seq"):
    """构造 12 个 UWB 事件 + 12 个 IMU 事件交替的测试序列。

    用足够多事件让 selected_count 在 nlos_ratio=0.5 下达到 6 个，覆盖多锚点
    + 多时间戳，让 selection_key 的 occluder_type 段真影响 _stable_window_start
    散布结果。dt 严格等于与前一事件的时间差（apply_nlos_level 守门 dt == t - prev_t）。
    """
    events = []
    prev_t = 0.0
    for i in range(12):
        anchor_id = i % 4  # 4 个锚点轮换
        uwb_t = float(i) * 0.1
        imu_t = uwb_t + 0.05
        # UWB 事件
        events.append({
            't': uwb_t,
            'dt': uwb_t - prev_t,
            'modality': 'uwb',
            'meta': {'scene_id': scene_id, 'seq_id': seq_id},
            'imu_payload': None,
            'uwb_payload': {'anchor_id': anchor_id, 'range': 2.0 + 0.1 * i, 'valid': True, 'quality': 0.95},
            'vio_payload': None,
        })
        prev_t = uwb_t
        # IMU 事件（紧跟 UWB 后 0.05s）
        events.append({
            't': imu_t,
            'dt': imu_t - prev_t,
            'modality': 'imu',
            'meta': {'scene_id': scene_id, 'seq_id': seq_id},
            'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0},
            'uwb_payload': None,
            'vio_payload': None,
        })
        prev_t = imu_t
    return events


def _nlos_cfg(occluder_type: str | None, nlos_ratio: float = 0.5) -> dict:
    """构造 N2 等级 cfg，occluder_type=None 时省略该字段（测试缺省回退路径）。"""
    n2_cfg = {
        'nlos_ratio': nlos_ratio,
        'bias_strength_m': 8.0,
        'nlos_noise_std_m': 0.0,
        'min_cluster_duration_s': 0.0,  # 关闭簇发守门，让 selection_key 真决定散布
    }
    if occluder_type is not None:
        n2_cfg['occluder_type'] = occluder_type
    return {'N2': n2_cfg}


def test_occluder_type_default_falls_back_to_machine_body():
    """缺省 occluder_type 时回退 machine_body，nlos_report 写回该值。"""
    events = _events_with_occluder("S(A0,N2,V0,M0,K3)")
    _, nlos_report = apply_nlos_level(events, 'N2', _nlos_cfg(occluder_type=None))
    assert nlos_report['occluder_type'] == 'machine_body', (
        f"缺省 occluder_type 应回退 machine_body，实际为 {nlos_report['occluder_type']!r}"
    )


def test_occluder_type_explicit_value_written_back_to_report():
    """显式 occluder_type=concrete_wall 时 nlos_report 写回 concrete_wall。"""
    events = _events_with_occluder("S(A0,N2,V0,M0,K3)")
    _, nlos_report = apply_nlos_level(events, 'N2', _nlos_cfg(occluder_type='concrete_wall'))
    assert nlos_report['occluder_type'] == 'concrete_wall'


def test_different_occluder_types_produce_different_selection_keys():
    """同一 (scene_id, seq_id, nlos_level) 不同 occluder_type 必须产生不同 selection_key。

    这是 H24d 真改的核心机制：occluder_type 注入 selection_key_parts 让不同
    遮挡物类型进入不同 RNG 桶，从而 NLOS 候选事件选择按遮挡物类型分桶真生效。
    """
    events = _events_with_occluder("S(A0,N2,V0,M0,K3)")
    occluder_types_to_test = ['machine_body', 'human_body', 'concrete_wall', 'metal_wall', 'glass_wall']
    selection_keys = {}
    selected_indices_set = {}
    for occluder_type in occluder_types_to_test:
        _, nlos_report = apply_nlos_level(events, 'N2', _nlos_cfg(occluder_type=occluder_type))
        selection_keys[occluder_type] = nlos_report['selection_key']
        selected_indices_set[occluder_type] = tuple(nlos_report['selected_indices'])

    # 至少有 2 个不同 occluder_type 产生不同 selection_key（不强求全 5 个都不同，
    # 因为 _stable_window_start 仍可能存在哈希碰撞，但 5 类下应至少 2 类不同）。
    unique_keys = set(selection_keys.values())
    assert len(unique_keys) >= 2, (
        f"5 类 occluder_type 应至少产生 2 个不同 selection_key，实际只有 {len(unique_keys)} 个："
        f"{selection_keys}"
    )

    # 至少 2 类 occluder_type 产生不同 selected_indices 散布（核心机制：分桶真生效）
    unique_selected = set(selected_indices_set.values())
    assert len(unique_selected) >= 2, (
        f"5 类 occluder_type 应至少产生 2 种不同 selected_indices 散布，实际只有 "
        f"{len(unique_selected)} 种：{selected_indices_set}"
    )


def test_occluder_type_in_selection_key_string():
    """selection_key 字符串必须包含 occluder=<value> 段，验证注入真生效。"""
    events = _events_with_occluder("S(A0,N2,V0,M0,K3)")
    _, nlos_report = apply_nlos_level(events, 'N2', _nlos_cfg(occluder_type='metal_wall'))
    assert 'occluder=metal_wall' in nlos_report['selection_key'], (
        f"selection_key 必须包含 'occluder=metal_wall' 段，实际为 {nlos_report['selection_key']!r}"
    )


def test_occluder_type_enum_constants_complete():
    """_NLOS_OCCLUDER_TYPES 枚举集必须含 5 类，与协议层声明同口径。"""
    expected = {'machine_body', 'human_body', 'concrete_wall', 'metal_wall', 'glass_wall'}
    assert _NLOS_OCCLUDER_TYPES == expected, (
        f"_NLOS_OCCLUDER_TYPES 必须为 {expected}，实际为 {_NLOS_OCCLUDER_TYPES}"
    )


def test_occluder_type_empty_string_rejected():
    """空字符串 occluder_type 必须抛 ValueError，防止隐式缺省污染 selection_key。"""
    events = _events_with_occluder("S(A0,N2,V0,M0,K3)")
    with pytest.raises(ValueError, match="occluder_type must be a non-empty string"):
        apply_nlos_level(events, 'N2', _nlos_cfg(occluder_type=''))


def test_occluder_type_invalid_enum_rejected_at_protocol_layer():
    """非枚举值 occluder_type 必须由协议层 _validate_axis_param_semantic_range 拒绝。"""
    from liquidloc.protocol.scene_axis_protocol import _validate_axis_param_semantic_range
    with pytest.raises(ValueError, match="occluder_type must be one of"):
        _validate_axis_param_semantic_range('N', 'N2', {
            'nlos_ratio': 0.5,
            'bias_strength_m': 8.0,
            'nlos_noise_std_m': 0.0,
            'occluder_type': 'wood_wall',  # 不在 _NLOS_OCCLUDER_TYPES 枚举内
        })
