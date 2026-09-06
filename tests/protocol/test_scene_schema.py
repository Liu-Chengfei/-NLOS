from __future__ import annotations

"""场景模式（scene_schema）测试模块。

测试覆盖范围：
- 场景 ID 的格式与解析
- 场景参数的结构验证
- 轴级别与退化参数的合同

被测模块：liquidloc.protocol.scene_schema"""

import pytest

import liquidloc.protocol.scene_schema as scene_schema_module
from liquidloc.protocol.scene_axis_protocol import AXIS_METADATA_KEYS, load_scene_axis_protocol
from liquidloc.protocol.scene_schema import SceneSpec, decode_scene, encode_scene


def _axis_levels() -> dict[str, list[str]]:
    cfg = load_scene_axis_protocol()
    return {
        axis_name: [k for k in axis_cfg.keys() if k not in AXIS_METADATA_KEYS]
        for axis_name, axis_cfg in cfg["axes"].items()
    }


def test_normal_case():
    """正常场景测试。

    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    axis_levels = _axis_levels()
    scene_spec = SceneSpec(
        A_level=axis_levels["A"][0],
        N_level=axis_levels["N"][0],
        V_level=axis_levels["V"][0],
        K_value=axis_levels["K"][0],
        M_level=axis_levels["M"][0],
    )

    scene_code = encode_scene(scene_spec)

    # 「五轴档位协议定义」文档：scene_code 格式 S(A,N,V,K,M) 五轴（G 已合并入 K）。
    assert scene_code == (
        f"S({scene_spec.A_level},{scene_spec.N_level},{scene_spec.V_level},"
        f"{scene_spec.K_value},{scene_spec.M_level})"
    )
    assert decode_scene(scene_code) == scene_spec


def test_boundary_case():
    """边界场景测试。

    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    axis_levels = _axis_levels()
    scene_spec = SceneSpec(
        A_level=axis_levels["A"][-1],
        N_level=axis_levels["N"][-1],
        V_level=axis_levels["V"][-1],
        K_value=axis_levels["K"][-1],
        M_level=axis_levels["M"][-1],
    )

    assert decode_scene(encode_scene(scene_spec)) == scene_spec


def test_invalid_case():
    """无效输入测试。

    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    axis_levels = _axis_levels()

    with pytest.raises(ValueError, match="K_value"):
        encode_scene(
            SceneSpec(
                A_level=axis_levels["A"][0],
                N_level=axis_levels["N"][0],
                V_level=axis_levels["V"][0],
                K_value="K999",
                M_level=axis_levels["M"][0],
            )
        )

    with pytest.raises(ValueError, match=r"must match S\(A,N,V,K,M\)"):
        decode_scene("S(A0,N0,V0,K0)")


def test_uses_current_scene_axis_protocol(monkeypatch):
    protocol_cfg = load_scene_axis_protocol()
    tightened_axes = {
        axis_name: dict(axis_cfg)
        for axis_name, axis_cfg in protocol_cfg["axes"].items()
    }
    removed_a_level = next(iter(tightened_axes["A"]))
    tightened_axes["A"].pop(removed_a_level)
    monkeypatch.setattr(
        scene_schema_module,
        "load_scene_axis_protocol",
        lambda: {"protocol_version": protocol_cfg.get("protocol_version"), "axes": tightened_axes},
    )

    with pytest.raises(ValueError, match="A_level"):
        encode_scene(
            SceneSpec(
                A_level=removed_a_level,
                N_level=next(iter(tightened_axes["N"])),
                V_level=next(iter(tightened_axes["V"])),
                K_value=next(iter(tightened_axes["K"])),
                M_level=next(iter(tightened_axes["M"])),
            )
        )


def test_encode_scene_uses_single_protocol_snapshot(monkeypatch):
    """使用测试：encode scene。\n\n验证被测功能正确使用 encode scene，\n确保内部依赖被正确调用。
    """
    protocol_cfg = load_scene_axis_protocol()
    original_axes = {
        axis_name: dict(axis_cfg)
        for axis_name, axis_cfg in protocol_cfg["axes"].items()
    }
    tightened_axes = {
        axis_name: dict(axis_cfg)
        for axis_name, axis_cfg in protocol_cfg["axes"].items()
    }
    # H26 真改：H26 在 N 轴顶层加了 nlos_generation_family 元数据键（与 N0-3 同级），
    # 后者已加入 AXIS_METADATA_KEYS，encode/decode 自动跳过；此处取 N 轴 first level 时
    # 也须跳过元数据键，保证 next() 取到真正的 level（N0/N1/N2/N3）而非元数据键。
    from liquidloc.protocol.scene_axis_protocol import AXIS_METADATA_KEYS
    removed_n_level = next(
        k for k in tightened_axes["N"] if k not in AXIS_METADATA_KEYS
    )
    tightened_axes["N"].pop(removed_n_level)

    payloads = iter(
        [
            {"protocol_version": protocol_cfg.get("protocol_version"), "axes": original_axes},
            {"protocol_version": protocol_cfg.get("protocol_version"), "axes": tightened_axes},
        ]
    )
    call_count = 0

    def _load_protocol():
        nonlocal call_count
        call_count += 1
        try:
            return next(payloads)
        except StopIteration:
            return {"protocol_version": protocol_cfg.get("protocol_version"), "axes": tightened_axes}

    monkeypatch.setattr(scene_schema_module, "load_scene_axis_protocol", _load_protocol)

    scene_spec = SceneSpec(
        A_level=next(k for k in original_axes["A"] if k not in AXIS_METADATA_KEYS),
        N_level=removed_n_level,
        V_level=next(k for k in original_axes["V"] if k not in AXIS_METADATA_KEYS),
        K_value=next(k for k in original_axes["K"] if k not in AXIS_METADATA_KEYS),
        M_level=next(k for k in original_axes["M"] if k not in AXIS_METADATA_KEYS),
    )

    assert encode_scene(scene_spec) == (
        f"S({scene_spec.A_level},{scene_spec.N_level},{scene_spec.V_level},"
        f"{scene_spec.K_value},{scene_spec.M_level})"
    )
    assert call_count == 1


def test_encode_scene_surfaces_current_protocol_structure_errors(monkeypatch):
    protocol_cfg = load_scene_axis_protocol()
    invalid_axes = {
        axis_name: dict(axis_cfg)
        for axis_name, axis_cfg in protocol_cfg["axes"].items()
        if axis_name != "K"
    }
    monkeypatch.setattr(
        scene_schema_module,
        "load_scene_axis_protocol",
        lambda: {"protocol_version": protocol_cfg.get("protocol_version"), "axes": invalid_axes},
    )

    with pytest.raises(ValueError, match="scene axis protocol missing axes"):
        encode_scene(
            SceneSpec(
                A_level=next(k for k in protocol_cfg["axes"]["A"] if k not in AXIS_METADATA_KEYS),
                N_level=next(k for k in protocol_cfg["axes"]["N"] if k not in AXIS_METADATA_KEYS),
                V_level=next(k for k in protocol_cfg["axes"]["V"] if k not in AXIS_METADATA_KEYS),
                K_value=next(k for k in protocol_cfg["axes"]["K"] if k not in AXIS_METADATA_KEYS),
                M_level=next(k for k in protocol_cfg["axes"]["M"] if k not in AXIS_METADATA_KEYS),
            )
        )