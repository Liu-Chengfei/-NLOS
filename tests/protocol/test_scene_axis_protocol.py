from __future__ import annotations

"""场景轴协议（scene_axis_protocol）测试模块。

测试覆盖范围：
- 场景轴定义与级别解析
- 协议加载与验证
- 轴级别参数的正确性

被测模块：liquidloc.protocol.scene_axis_protocol"""

import pytest

from liquidloc.common.config_utils import find_project_root
from liquidloc.protocol.scene_axis_protocol import attach_scene_parameters, load_scene_axis_protocol, resolve_axis_level

_PROJECT_ROOT = find_project_root()


def test_normal_case():
    """正常场景测试。

    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    cfg = load_scene_axis_protocol()
    axes = cfg['axes']
    # A0: near_sync 全参数断言（与文档「五轴档位协议定义」+ scene_axis_protocol.yaml 严格一致）
    assert axes["A"]["A0"]["label"] == "near_sync"
    assert axes["A"]["A0"]["offset_ms"] == [0, 0]
    assert axes["A"]["A0"]["jitter_ms"] == 0
    assert axes["A"]["A0"]["burst_missing_prob"] == 0.00
    assert axes["A"]["A0"]["cross_modal_skew_ms"] == 0
    assert axes["A"]["A0"]["clock_drift_ppm"] == 0
    # A1: mild_async 全参数断言
    assert axes["A"]["A1"]["label"] == "mild_async"
    assert axes["A"]["A1"]["offset_ms"] == [5, 20]
    assert axes["A"]["A1"]["jitter_ms"] == 1
    assert axes["A"]["A1"]["cross_modal_skew_ms"] == 30
    assert axes["A"]["A1"]["clock_drift_ppm"] == 10
    # A2: moderate_async 全参数断言
    assert axes["A"]["A2"]["label"] == "moderate_async"
    assert axes["A"]["A2"]["offset_ms"] == [20, 50]
    assert axes["A"]["A2"]["jitter_ms"] == 3
    assert axes["A"]["A2"]["cross_modal_skew_ms"] == 60
    assert axes["A"]["A2"]["clock_drift_ppm"] == 55
    assert cfg['axes']['A']['A3']['cross_modal_skew_ms'] == 185
    assert cfg['axes']['A']['A3']['clock_drift_ppm'] == 225
    # N2 协议层：bias_strength_m 是区间 [2.0, 3.0]（resolve_axis_level 返回原始列表）
    n2_bias = resolve_axis_level('N', 'N2', cfg)['bias_strength_m']
    assert tuple(n2_bias) == (2.0, 3.0), f'N2 bias_strength_m={n2_bias} 应为区间 [2.0, 3.0]'


def test_attach_scene_parameters_case():
    params = attach_scene_parameters({'A': 'A2', 'N': 'N3', 'V': 'V2', 'K': 'K1', 'M': 'M0'})
    assert params['axes']['A']['offset_ms'] == [20, 50]
    assert params['flat']['A_clock_drift_ppm'] == 55
    assert params['flat']['K_anchor_count'] == 4


def test_attach_scene_parameters_requires_all_axes():
    """必填测试：attach scene parameters。\n\n验证 attach scene parameters 的必填约束，\n确保缺少必要输入时抛出异常。
    """
    with pytest.raises(ValueError, match=r"missing scene axes"):
        attach_scene_parameters({'A': 'A2', 'N': 'N3', 'V': 'V2', 'K': 'K1'})


def test_resolve_axis_level_normalizes_whitespace_wrapped_level():
    """归一化测试：resolve axis level。\n\n验证 resolve axis level 的归一化处理，\n确保空白/重复等输入被正确清洗。
    """
    payload = resolve_axis_level('A', ' A0 ')
    assert payload['axis'] == 'A'
    assert payload['level'] == 'A0'
    assert payload['label'] == 'near_sync'


def test_attach_scene_parameters_normalizes_whitespace_wrapped_level():
    """归一化测试：attach scene parameters。\n\n验证 attach scene parameters 的归一化处理，\n确保空白/重复等输入被正确清洗。
    """
    params = attach_scene_parameters({'A': ' A2 ', 'N': 'N3', 'V': 'V2', 'K': 'K1', 'M': 'M0'})
    assert params['axes']['A']['level'] == 'A2'
    assert params['flat']['A_clock_drift_ppm'] == 55


def test_attach_scene_parameters_rejects_unknown_axes():
    """拒绝测试：attach scene parameters。\n\n验证被测功能对 attach scene parameters 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(KeyError, match=r"Unknown axis"):
        attach_scene_parameters({'A': 'A2', 'N': 'N3', 'V': 'V2', 'K': 'K1', 'X': 'X0'})


def test_resolve_axis_level_does_not_fallback_from_explicit_empty_protocol_cfg():
    """不侵入测试：resolve axis level。\n\n验证 resolve axis level 不会产生副作用，\n确保功能隔离性。
    """
    with pytest.raises(ValueError, match=r"scene axis protocol missing axes"):
        resolve_axis_level('N', 'N2', {'protocol_version': 27})


def test_attach_scene_parameters_does_not_fallback_from_explicit_empty_protocol_cfg():
    """不侵入测试：attach scene parameters。\n\n验证 attach scene parameters 不会产生副作用，\n确保功能隔离性。
    """
    with pytest.raises(ValueError, match=r"scene axis protocol missing axes"):
        attach_scene_parameters({'A': 'A2', 'N': 'N3', 'V': 'V2', 'K': 'K1'}, {'protocol_version': 27})


def test_resolve_axis_level_rejects_non_mapping_protocol_cfg():
    """拒绝测试：resolve axis level。\n\n验证被测功能对 resolve axis level 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(TypeError, match=r"scene axis protocol must be a mapping"):
        resolve_axis_level('N', 'N2', [])


def test_resolve_axis_level_rejects_falsey_non_mapping_axes_payload():
    """拒绝测试：resolve axis level。\n\n验证被测功能对 resolve axis level 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(TypeError, match=r"scene axis protocol axes must be a mapping"):
        resolve_axis_level('N', 'N2', {'protocol_version': 27, 'axes': []})


def test_resolve_axis_level_rejects_non_string_level():
    """拒绝测试：resolve axis level。\n\n验证被测功能对 resolve axis level 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(TypeError, match=r"level must be a string, got bool"):
        resolve_axis_level('A', True)


def test_attach_scene_parameters_rejects_non_string_axis_level():
    """拒绝测试：attach scene parameters。\n\n验证被测功能对 attach scene parameters 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(TypeError, match=r"level must be a string, got bool"):
        attach_scene_parameters({'A': True, 'N': 'N0', 'V': 'V0', 'K': 'K0', 'M': 'M0'})


def test_attach_scene_parameters_rejects_protocol_cfg_with_unknown_axes():
    """拒绝测试：attach scene parameters。\n\n验证被测功能对 attach scene parameters 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(ValueError, match=r"scene axis protocol contains unknown axes"):
        attach_scene_parameters(
            {'A': 'A0', 'N': 'N0', 'V': 'V0', 'K': 'K0'},
            {
                'protocol_version': 27,
                'axes': {
                    'A': {'A0': {'label': 'near_sync'}},
                    'N': {'N0': {'label': 'los_dominant'}},
                    'V': {'V0': {'label': 'nominal_vio'}},
                    'K': {'K0': {'label': 'anchor_count_4_symmetric', 'anchor_count': 4, 'geom_condition': 1.0}},
                    'X': {'X0': {'label': 'invalid_axis'}},
                }
            },
        )


def test_resolve_axis_level_rejects_non_mapping_axis_payload():
    """拒绝测试：resolve axis level。\n\n验证被测功能对 resolve axis level 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(TypeError, match=r"scene axis protocol axis payload must be a mapping"):
        resolve_axis_level(
            'A',
            'A0',
            {
                'protocol_version': 27,
                'axes': {
                    'A': [],
                    'N': {'N0': {'label': 'los_dominant'}},
                    'V': {'V0': {'label': 'nominal_vio'}},
                    'K': {'K0': {'label': 'anchor_count_4_symmetric', 'anchor_count': 4, 'geom_condition': 1.0}},
                    'M': {'M0': {'label': 'no_missing'}},
                }
            },
        )


def test_load_scene_axis_protocol_rejects_non_mapping_level_payload(tmp_path):
    """拒绝测试：load scene axis protocol。\n\n验证被测功能对 load scene axis protocol 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    tmp_dir = _PROJECT_ROOT / "outputs" / "_test_tmp" / "scene_axis_protocol"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = tmp_dir / 'scene_axis_protocol.yaml'
    protocol_path.write_text(
        '\n'.join(
            [
                'protocol_version: 27',
                'axes:',
                '  A:',
                '    A0: []',
                '  N:',
                "    N0: {label: los_dominant}",
                '  V:',
                "    V0: {label: nominal_vio}",
                '  K:',
                "    K0: {label: anchor_count_4_symmetric, anchor_count: 4, geom_condition: 1.0}",
                '  M:',
                "    M0: {label: no_missing}",
            ]
        ),
        encoding='utf-8',
    )

    with pytest.raises(TypeError, match=r"scene axis protocol level payload must be a mapping: A\.A0"):
        load_scene_axis_protocol(protocol_path)


def test_resolve_axis_level_rejects_unexpected_protocol_version():
    """拒绝测试：resolve axis level。\n\n验证被测功能对 resolve axis level 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(ValueError, match=r"scene axis protocol version must be 2"):
        resolve_axis_level(
            'A',
            'A0',
            {
                'protocol_version': 999,
                'axes': {
                    'A': {'A0': {'label': 'near_sync'}},
                    'N': {'N0': {'label': 'los_dominant'}},
                    'V': {'V0': {'label': 'nominal_vio'}},
                    'K': {'K0': {'label': 'anchor_count_4_symmetric', 'anchor_count': 4, 'geom_condition': 1.0}},
                },
            },
        )


def test_load_scene_axis_protocol_requires_frozen_protocol_version(tmp_path):
    """必填测试：load scene axis protocol。\n\n验证 load scene axis protocol 的必填约束，\n确保缺少必要输入时抛出异常。
    """
    tmp_dir = _PROJECT_ROOT / "outputs" / "_test_tmp" / "scene_axis_protocol"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = tmp_dir / 'scene_axis_protocol.yaml'
    protocol_path.write_text(
        '\n'.join(
            [
                'axes:',
                '  A:',
                "    A0: {label: near_sync}",
                '  N:',
                "    N0: {label: los_dominant}",
                '  V:',
                "    V0: {label: nominal_vio}",
                '  K:',
                "    K0: {label: anchor_count_4_symmetric, anchor_count: 4, geom_condition: 1.0}",
            ]
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match=r"scene axis protocol version is missing, expected 2"):
        load_scene_axis_protocol(protocol_path)


def test_load_scene_axis_protocol_rejects_blank_string_path():
    """拒绝测试：load scene axis protocol。\n\n验证被测功能对 load scene axis protocol 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(ValueError, match=r"scene axis protocol path must not be blank"):
        load_scene_axis_protocol("")
