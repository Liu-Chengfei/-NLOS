from __future__ import annotations

"""H26 真改 §4.3.1 L933 主表须事先写死生成族 测试模块。

文件职责：验证 §4.3.1 L933「主表须事先写死生成族」+ L935「允许的生成族（协议二选一或
组合，须声明）」字面要求落实：
- 默认 configs/base/scene_axis_protocol.yaml N 轴顶层存在 nlos_generation_family 字段
- 字段值属于 _ALLOWED_NLOS_GENERATION_FAMILIES 三选一（A_geometric_los /
  B_state_conditional_statistical / C_AB_hybrid），违反时由协议层 semantic 校验拒绝
- nlos_generation_family_doc 允许任意字符串（docstring 不参与枚举校验）
- 缺字段不校验（向后兼容旧 yaml）
- _AXIS_LEVEL_PARAM_KEYS['N'] 白名单已包含 nlos_generation_family + nlos_generation_family_doc，
  让 yaml load_scene_axis_protocol 不会因白名单缺失而拒绝该字段

被测模块：liquidloc.protocol.scene_axis_protocol.load_scene_axis_protocol
        liquidloc.protocol.scene_axis_protocol._validate_axis_param_semantic_range
        liquidloc.protocol.scene_axis_protocol._AXIS_LEVEL_PARAM_KEYS
        liquidloc.protocol.scene_axis_protocol._ALLOWED_NLOS_GENERATION_FAMILIES
"""

import pytest
import yaml

from liquidloc.common.config_utils import find_project_root
from liquidloc.protocol.scene_axis_protocol import (
    _ALLOWED_NLOS_GENERATION_FAMILIES,
    _AXIS_LEVEL_PARAM_KEYS,
    _validate_axis_param_semantic_range,
    load_scene_axis_protocol,
)

_PROJECT_ROOT = find_project_root()
_DEFAULT_PROTOCOL_YAML = _PROJECT_ROOT / "configs" / "base" / "scene_axis_protocol.yaml"


def test_default_protocol_yaml_declares_nlos_generation_family():
    """默认协议 yaml N 轴顶层必须显式声明 nlos_generation_family 字段。

    这是 §4.3.1 L933「主表须事先写死生成族」字面要求的最直接证据：缺此项 =
    主表未声明生成族 = §4.3.1 L933 真违规。
    """
    with _DEFAULT_PROTOCOL_YAML.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    n_axis = cfg["axes"]["N"]
    assert "nlos_generation_family" in n_axis, (
        "§4.3.1 L933 主表须事先写死生成族：默认协议 yaml N 轴顶层缺 nlos_generation_family 字段"
    )
    family = n_axis["nlos_generation_family"]
    assert isinstance(family, str) and family in _ALLOWED_NLOS_GENERATION_FAMILIES, (
        f"§4.3.1 L935 生成族须属于三选一枚举，got {family!r}"
    )


def test_default_protocol_yaml_nlos_generation_family_doc_is_string():
    """默认协议 yaml nlos_generation_family_doc 必须是字符串（docstring 不参与枚举校验）。"""
    with _DEFAULT_PROTOCOL_YAML.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    n_axis = cfg["axes"]["N"]
    doc = n_axis.get("nlos_generation_family_doc")
    assert isinstance(doc, str) and doc.strip(), (
        "nlos_generation_family_doc 须是非空字符串，描述所选生成族物理含义"
    )


def test_axis_level_param_keys_whitelist_includes_generation_family_fields():
    """_AXIS_LEVEL_PARAM_KEYS['N'] 白名单必须包含 nlos_generation_family + doc 字段。

    白名单缺这两个字段会让 yaml load_scene_axis_protocol 在协议加载阶段因
    "未知字段"拒绝，§4.3.1 L933 显式声明字段无法写入 yaml。
    """
    n_whitelist = _AXIS_LEVEL_PARAM_KEYS["N"]
    assert "nlos_generation_family" in n_whitelist, (
        "N 轴白名单缺 nlos_generation_family 字段，§4.3.1 L933 声明字段会被协议加载拒绝"
    )
    assert "nlos_generation_family_doc" in n_whitelist, (
        "N 轴白名单缺 nlos_generation_family_doc 字段"
    )


def test_allowed_nlos_generation_families_enumeration_is_locked():
    """_ALLOWED_NLOS_GENERATION_FAMILIES 必须是 A/B/C 三选一 frozenset。

    每个 entry 字面对应 §4.3.1 L937-941 表中允许的生成族名，禁止任何隐式扩展。
    """
    assert _ALLOWED_NLOS_GENERATION_FAMILIES == frozenset({
        "A_geometric_los",
        "B_state_conditional_statistical",
        "C_AB_hybrid",
    }), "§4.3.1 L937-941 允许的生成族固定为 A/B/C 三选一"


@pytest.mark.parametrize("family", sorted(_ALLOWED_NLOS_GENERATION_FAMILIES))
def test_validate_semantic_range_accepts_all_allowed_families(family):
    """_validate_axis_param_semantic_range 必须接受三个允许的生成族值。"""
    _validate_axis_param_semantic_range("N", "N0", {"nlos_generation_family": family})


def test_validate_semantic_range_rejects_unknown_family():
    """_validate_axis_param_semantic_range 必须拒绝枚举外的生成族值。"""
    with pytest.raises(ValueError, match="nlos_generation_family must be one of"):
        _validate_axis_param_semantic_range(
            "N", "N0", {"nlos_generation_family": "D_invalid_random_family"}
        )


def test_validate_semantic_range_rejects_non_string_family():
    """非字符串 nlos_generation_family 必须被协议层拒绝。"""
    with pytest.raises(ValueError, match="nlos_generation_family must be a string"):
        _validate_axis_param_semantic_range("N", "N0", {"nlos_generation_family": 123})


def test_validate_semantic_range_rejects_non_string_doc():
    """非字符串 nlos_generation_family_doc 必须被协议层拒绝。"""
    with pytest.raises(ValueError, match="nlos_generation_family_doc must be a string"):
        _validate_axis_param_semantic_range(
            "N", "N0", {"nlos_generation_family_doc": 456}
        )


def test_validate_semantic_range_skips_when_family_field_absent():
    """缺 nlos_generation_family 字段时协议层不校验（向后兼容旧 yaml）。"""
    # 不抛异常即通过
    _validate_axis_param_semantic_range("N", "N0", {"label": "los_dominant"})


def test_load_default_protocol_succeeds_with_generation_family():
    """load_scene_axis_protocol 必须真加载默认 yaml 而不因新字段拒绝。

    端到端验证：协议层 yaml + python 协议加载管线整链通畅，H26 真改字段成功
    贯穿 yaml → load → validation 三层。
    """
    cfg = load_scene_axis_protocol(str(_DEFAULT_PROTOCOL_YAML))
    n_axis = cfg["axes"]["N"]
    assert "nlos_generation_family" in n_axis, (
        "load_scene_axis_protocol 必须把 yaml 顶层 nlos_generation_family 字段透传给下游"
    )
    assert n_axis["nlos_generation_family"] in _ALLOWED_NLOS_GENERATION_FAMILIES
