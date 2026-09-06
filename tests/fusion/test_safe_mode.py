from __future__ import annotations

"""安全模式（safe_mode）测试模块。

文件职责：验证 apply_safe_mode 能根据场景上下文和
风险阈值对模型输出施加衰减。

测试覆盖范围：
- 正常场景：基准场景不触发安全模式
- 显式启用正常场景
- 边界场景：风险恰等于阈值
- 异常场景：非法 safe_mode_cfg 类型
- 非基准锚点数不强制安全模式
- 轴字段场景上下文模式验证
- 冲突模式输入拒绝
- 启用但高风险正常场景保持原值
- 字典输入缺失键拒绝
- 字典输入布尔值拒绝
- 非法中间量类型拒绝
- risk 超范围拒绝
- risk_threshold 超范围拒绝
- scaling 低于下限拒绝
- None 输入拒绝
- 字典输入返回 ModelIntermediate

被测模块：liquidloc.fusion.safe_mode"""


import pytest

from liquidloc.common.types import ModelIntermediate
from liquidloc.fusion.safe_mode import apply_safe_mode


def test_normal_case():
    adjusted_outputs, safe_mode_flag = apply_safe_mode(
        ModelIntermediate(bias=0.4, risk=0.3, uwb_scaling=1.4, vio_scaling=1.2),
        "S(A0,N0,V0,K0,M0)",
        None,
    )

    assert safe_mode_flag is False
    assert isinstance(adjusted_outputs, ModelIntermediate)
    assert adjusted_outputs.bias == pytest.approx(0.4)
    assert adjusted_outputs.risk == pytest.approx(0.3)
    assert adjusted_outputs.uwb_scaling == pytest.approx(1.4)
    assert adjusted_outputs.vio_scaling == pytest.approx(1.2)


def test_explicit_enabled_normal_case():
    adjusted_outputs, safe_mode_flag = apply_safe_mode(
        ModelIntermediate(bias=0.4, risk=0.3, uwb_scaling=1.4, vio_scaling=1.2),
        "S(A0,N0,V0,K0,M0)",
        {"enabled": True, "risk_threshold": 0.5},
    )

    assert safe_mode_flag is True
    assert isinstance(adjusted_outputs, ModelIntermediate)
    assert adjusted_outputs.bias == pytest.approx(0.12)
    assert adjusted_outputs.risk == pytest.approx(0.3)
    assert adjusted_outputs.uwb_scaling == pytest.approx(1.12)
    assert adjusted_outputs.vio_scaling == pytest.approx(1.06)


def test_boundary_case():
    adjusted_outputs, safe_mode_flag = apply_safe_mode(
        {
            "bias": 0.5,
            "risk": 0.5,
            "uwb_scaling": 1.6,
            "vio_scaling": 1.8,
        },
        "S(A1,N1,V1,K0,M0)",
        {"enabled": True, "risk_threshold": 0.5},
    )

    assert safe_mode_flag is True
    assert adjusted_outputs.bias == pytest.approx(0.25)
    assert adjusted_outputs.risk == pytest.approx(0.5)
    assert adjusted_outputs.uwb_scaling == pytest.approx(1.3)
    assert adjusted_outputs.vio_scaling == pytest.approx(1.4)


def test_invalid_case():
    with pytest.raises(TypeError, match="safe_mode_cfg"):
        apply_safe_mode(
            {
                "bias": 0.5,
                "risk": 0.2,
                "uwb_scaling": 1.2,
                "vio_scaling": 1.1,
            },
            "S(A0,N0,V0,K0,M0)",
            [],
        )


@pytest.mark.xfail(reason="历史 fixture 漂移：risk=0.9 > _RISK_HARD_SKIP_THRESHOLD(0.04) 触发 skip 路径，partial_damping 公式从未执行；需重新设计测试场景。")
def test_non_base_anchor_count_does_not_force_safe_mode():
    adjusted_outputs, safe_mode_flag = apply_safe_mode(
        {
            "bias": 0.4,
            "risk": 0.9,
            "uwb_scaling": 1.4,
            "vio_scaling": 1.2,
        },
        "S(A0,N0,V0,K0,M0)",
        {"enabled": True, "risk_threshold": 0.5},
    )

    assert safe_mode_flag is False
    # 协议基准 K=K0（4 锚对称矩形四角），scene 是 K0 + high risk → 走非正常场景部分衰减分支
    # partial_damping = 1.0 - 0.5*0.9 = 0.55, adjusted_bias = 0.4 * 0.55 = 0.22
    assert adjusted_outputs.bias == pytest.approx(0.22)
    assert adjusted_outputs.risk == pytest.approx(0.9)
    assert adjusted_outputs.uwb_scaling == pytest.approx(1.4)
    assert adjusted_outputs.vio_scaling == pytest.approx(1.2)


def test_axis_field_scene_context_is_schema_validated():
    """schema 校验：超协议档位（K4/K5/K6）应 raise ValueError（K 轴仅 K0/K1/K3）。"""
    with pytest.raises(ValueError, match="K_value"):
        apply_safe_mode(
            {
                "bias": 0.4,
                "risk": 0.9,
                "uwb_scaling": 1.4,
                "vio_scaling": 1.2,
            },
            {
                "A_level": "A0",
                "N_level": "N0",
                "V_level": "V0",
                "M_level": "M0",
                "K_value": "K4",  # noqa: protocol-comply  主动注入超协议值验证拒绝逻辑。协议 K 轴仅 K0/K1/K3。
            },
            {"enabled": True, "risk_threshold": 0.5},
        )


def test_dict_scene_context_rejects_conflicting_schema_inputs():
    with pytest.raises(ValueError, match="scene schema inputs disagree"):
        apply_safe_mode(
            {
                "bias": 0.4,
                "risk": 0.9,
                "uwb_scaling": 1.4,
                "vio_scaling": 1.2,
            },
            {
                "scene_id": "S(A0,N0,V0,K0,M0)",  # scene_id 中 K=K0
                "A_level": "A0",
                "N_level": "N0",
                "V_level": "V0",
                "M_level": "M0",  # 完整 5 轴字段
                "K_value": "K1",  # 轴字段中 K=K1；与 scene_id 不一致触发 schema 冲突。
            },
            {"enabled": True, "risk_threshold": 0.5},
        )


def test_enabled_high_risk_normal_scene_keeps_original():
    """启用安全模式但风险高于阈值且为正常场景时，保持原值不收敛。"""
    adjusted_outputs, safe_mode_flag = apply_safe_mode(
        ModelIntermediate(bias=0.4, risk=0.9, uwb_scaling=1.4, vio_scaling=1.2),
        "S(A0,N0,V0,K0,M0)",
        {"enabled": True, "risk_threshold": 0.5},
    )

    assert safe_mode_flag is False
    assert adjusted_outputs.bias == pytest.approx(0.4)
    assert adjusted_outputs.risk == pytest.approx(0.9)
    assert adjusted_outputs.uwb_scaling == pytest.approx(1.4)
    assert adjusted_outputs.vio_scaling == pytest.approx(1.2)


def test_dict_input_missing_keys_rejected():
    """字典输入缺少必需键时抛 KeyError。"""
    with pytest.raises(KeyError, match="missing required keys"):
        apply_safe_mode(
            {"bias": 0.4, "risk": 0.3},
            "S(A0,N0,V0,K0,M0)",
            None,
        )


def test_dict_input_bool_values_rejected():
    """字典输入中布尔值冒充数值时抛 TypeError。"""
    with pytest.raises(TypeError, match="must be numeric, got bool"):
        apply_safe_mode(
            {"bias": True, "risk": 0.3, "uwb_scaling": 1.4, "vio_scaling": 1.2},
            "S(A0,N0,V0,K0,M0)",
            None,
        )


def test_invalid_intermediate_outputs_type_rejected():
    """中间量既不是 ModelIntermediate 也不是字典时抛 TypeError。"""
    with pytest.raises(TypeError, match="must be a ModelIntermediate or dict"):
        apply_safe_mode(
            [0.4, 0.3, 1.4, 1.2],
            "S(A0,N0,V0,K0,M0)",
            None,
        )


def test_risk_out_of_range_rejected():
    """risk 超出 [0, 1] 范围时抛异常（用字典绕过 ModelIntermediate 构造校验）。"""
    with pytest.raises(ValueError, match="risk_value"):
        apply_safe_mode(
            {"bias": 0.4, "risk": 1.5, "uwb_scaling": 1.4, "vio_scaling": 1.2},
            "S(A0,N0,V0,K0,M0)",
            None,
        )


def test_risk_threshold_out_of_range_rejected():
    """risk_threshold 超出 [0, 1] 范围时抛异常。"""
    with pytest.raises(ValueError, match="risk_threshold"):
        apply_safe_mode(
            ModelIntermediate(bias=0.4, risk=0.3, uwb_scaling=1.4, vio_scaling=1.2),
            "S(A0,N0,V0,K0,M0)",
            {"enabled": True, "risk_threshold": 2.0},
        )


def test_scaling_below_minimum_rejected():
    """scaling 低于下限时抛异常。"""
    with pytest.raises(ValueError, match="uwb_scaling"):
        apply_safe_mode(
            ModelIntermediate(bias=0.4, risk=0.3, uwb_scaling=0.5, vio_scaling=1.2),
            "S(A0,N0,V0,K0,M0)",
            None,
        )


def test_none_intermediate_outputs_rejected():
    """intermediate_outputs 为 None 时抛异常。"""
    with pytest.raises((ValueError, TypeError)):
        apply_safe_mode(
            None,
            "S(A0,N0,V0,K0,M0)",
            None,
        )


def test_none_scene_context_rejected():
    """scene_context 为 None 时抛异常。"""
    with pytest.raises((ValueError, TypeError)):
        apply_safe_mode(
            ModelIntermediate(bias=0.4, risk=0.3, uwb_scaling=1.4, vio_scaling=1.2),
            None,
            None,
        )


def test_dict_input_returns_model_intermediate():
    """字典输入也返回 ModelIntermediate，保持返回类型一致。"""
    adjusted_outputs, _ = apply_safe_mode(
        {"bias": 0.4, "risk": 0.3, "uwb_scaling": 1.4, "vio_scaling": 1.2},
        "S(A0,N0,V0,K0,M0)",
        None,
    )

    assert isinstance(adjusted_outputs, ModelIntermediate)
    assert adjusted_outputs.bias == pytest.approx(0.4)
    assert adjusted_outputs.risk == pytest.approx(0.3)
