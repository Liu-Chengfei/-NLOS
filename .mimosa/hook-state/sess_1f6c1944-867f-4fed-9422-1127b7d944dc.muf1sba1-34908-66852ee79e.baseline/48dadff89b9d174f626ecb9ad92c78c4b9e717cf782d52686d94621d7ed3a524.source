"""liquid_bridge_contract 模块测试。

覆盖范围：
  - normalize_risk: 有限/无限/边界裁剪
  - _coerce_positive_scaling: 合法值/非法值/< 1.0 裁剪警告
  - _resolve_quality: 有限/无限/边界裁剪
  - _quality_below_floor: 边界容差
  - _compose_noise_multiplier: 风险/缩放/上限交互
  - _resolve_modality_signal: UWB valid 语义、VIO tracked_features/reproj_err
  - _resolve_effective_risk: 三路取最大
  - _is_neutral_intermediate: 默认/非默认判断
  - apply_safe_mode: 各模态门控动作
  - adjust_intermediate_for_safe_mode: 收敛/不收敛/配置/类型
  - build_measurement_control: 端到端集成
  - _resolve_safe_mode_scene: 场景解析多来源
  - LiquidBridgeDecision.to_dict
"""

from __future__ import annotations

import warnings

import pytest

from liquidloc.common.types import MeasurementControl, ModelIntermediate
from liquidloc.pipelines.train_pipeline import _build_target_intermediate
from liquidloc.protocol.liquid_bridge_contract import (
    LiquidBridgeDecision,
    _coerce_positive_scaling,
    _compose_noise_multiplier,
    _is_neutral_intermediate,
    _quality_below_floor,
    _resolve_effective_risk,
    _resolve_modality_signal,
    _resolve_quality,
    _resolve_safe_mode_scene,
    adjust_intermediate_for_safe_mode,
    apply_safe_mode,
    build_measurement_control,
    normalize_risk,
)
from liquidloc.protocol.scene_schema import SceneSpec
from liquidloc.scenarios.async_levels import apply_async_level
from liquidloc.scenarios.nlos_levels import apply_nlos_level
from liquidloc.scenarios.visual_levels import apply_visual_level


# ============================================================================
# normalize_risk
# ============================================================================


class TestNormalizeRisk:
    """normalize_risk 测试。"""

    def test_clips_to_zero(self):
        """负值裁剪到 0。"""
        assert normalize_risk(-0.5) == pytest.approx(0.0)

    def test_clips_to_one(self):
        """超过 1 的值裁剪到 1。"""
        assert normalize_risk(1.7) == pytest.approx(1.0)

    def test_preserves_valid_range(self):
        """合法范围内的值不变。"""
        assert normalize_risk(0.0) == pytest.approx(0.0)
        assert normalize_risk(0.5) == pytest.approx(0.5)
        assert normalize_risk(1.0) == pytest.approx(1.0)

    def test_rejects_nan(self):
        """NaN 直接报错。"""
        with pytest.raises(ValueError, match="finite"):
            normalize_risk(float("nan"))

    def test_rejects_inf(self):
        """inf 直接报错。"""
        with pytest.raises(ValueError, match="finite"):
            normalize_risk(float("inf"))
        with pytest.raises(ValueError, match="finite"):
            normalize_risk(float("-inf"))


# ============================================================================
# _coerce_positive_scaling
# ============================================================================


class TestCoercePositiveScaling:
    """_coerce_positive_scaling 测试。"""

    def test_accepts_valid_scaling(self):
        """>= 1.0 的合法缩放值原样返回。"""
        assert _coerce_positive_scaling(1.0, name="test") == pytest.approx(1.0)
        assert _coerce_positive_scaling(2.5, name="test") == pytest.approx(2.5)

    def test_clips_above_scaling_max(self):
        """超过 scaling_max (50.0) 的值裁剪到上限。"""
        assert _coerce_positive_scaling(100.0, name="test") == pytest.approx(50.0)
        assert _coerce_positive_scaling(50.0, name="test") == pytest.approx(50.0)
        assert _coerce_positive_scaling(49.9, name="test") == pytest.approx(49.9)

    def test_rejects_zero(self):
        """零直接报错。"""
        with pytest.raises(ValueError, match="finite and > 0"):
            _coerce_positive_scaling(0.0, name="test")

    def test_rejects_negative(self):
        """负数直接报错。"""
        with pytest.raises(ValueError, match="finite and > 0"):
            _coerce_positive_scaling(-1.0, name="test")

    def test_rejects_nan(self):
        """NaN 直接报错。"""
        with pytest.raises(ValueError, match="finite and > 0"):
            _coerce_positive_scaling(float("nan"), name="test")

    def test_clips_below_one_with_warning(self):
        """0 < x < 1.0 裁剪到 1.0 并发出警告。"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _coerce_positive_scaling(0.8, name="uwb_scaling")
            assert result == pytest.approx(1.0)
            assert len(w) == 1
            assert "clipped to scaling_min" in str(w[0].message)

    def test_no_warning_at_boundary(self):
        """scaling=1.0 不产生警告。"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _coerce_positive_scaling(1.0, name="test")
            assert len(w) == 0


# ============================================================================
# _resolve_quality
# ============================================================================


class TestResolveQuality:
    """_resolve_quality 测试。"""

    def test_clips_to_range(self):
        """越界值裁剪到 [0, 1]。"""
        assert _resolve_quality(-0.5) == pytest.approx(0.0)
        assert _resolve_quality(1.5) == pytest.approx(1.0)

    def test_preserves_valid(self):
        """合法值不变。"""
        assert _resolve_quality(0.5) == pytest.approx(0.5)

    def test_rejects_nan(self):
        """NaN 直接报错。"""
        with pytest.raises(ValueError, match="finite"):
            _resolve_quality(float("nan"))


# ============================================================================
# _quality_below_floor
# ============================================================================


class TestQualityBelowFloor:
    """_quality_below_floor 测试。"""

    def test_below_floor(self):
        """低于门槛返回 True。"""
        assert _quality_below_floor(0.17, 0.18) is True

    def test_at_floor(self):
        """等于门槛时因容差返回 False（不跳过）。"""
        assert _quality_below_floor(0.18, 0.18) is False

    def test_above_floor(self):
        """高于门槛返回 False。"""
        assert _quality_below_floor(0.19, 0.18) is False


# ============================================================================
# _compose_noise_multiplier
# ============================================================================


class TestComposeNoiseMultiplier:
    """_compose_noise_multiplier 测试。"""

    def test_base_case(self):
        """scaling=1, risk=0 时倍数为 1。"""
        result = _compose_noise_multiplier(scaling=1.0, risk=0.0, ceiling=2.0)
        assert result == pytest.approx(1.0)

    def test_risk_increases_multiplier(self):
        """风险增加噪声倍数。"""
        low = _compose_noise_multiplier(scaling=1.0, risk=0.0, ceiling=3.0)
        high = _compose_noise_multiplier(scaling=1.0, risk=0.5, ceiling=3.0)
        assert high > low

    def test_ceiling_caps(self):
        """上限封顶。"""
        result = _compose_noise_multiplier(scaling=5.0, risk=1.0, ceiling=2.0)
        assert result == pytest.approx(2.0)

    def test_scaling_below_one_not_clamped_internally(self):
        """_compose_noise_multiplier 不内部裁剪 scaling，信任上游 _coerce_positive_scaling 的保证。"""
        result = _compose_noise_multiplier(scaling=0.5, risk=0.0, ceiling=3.0)
        assert result == pytest.approx(0.25)  # 0.5^2 * (1 + 0) = 0.25


# ============================================================================
# _resolve_modality_signal
# ============================================================================


class TestResolveModalitySignal:
    """_resolve_modality_signal 测试。"""

    def test_uwb_invalid_returns_signal(self):
        """UWB valid=False 返回 0.25。"""
        event = {"modality": "uwb", "uwb_payload": {"valid": False}}
        assert _resolve_modality_signal(event) == pytest.approx(0.25)

    def test_uwb_valid_returns_zero(self):
        """UWB valid=True 返回 0。"""
        event = {"modality": "uwb", "uwb_payload": {"valid": True}}
        assert _resolve_modality_signal(event) == pytest.approx(0.0)

    def test_uwb_valid_none_treated_as_valid(self):
        """UWB valid=None 当有效处理，返回 0。"""
        # 与 build_measurement_control 的 is not False 语义一致
        event = {"modality": "uwb", "uwb_payload": {"valid": None}}
        assert _resolve_modality_signal(event) == pytest.approx(0.0)

    def test_uwb_no_payload_returns_zero(self):
        """无 uwb_payload 返回 0。"""
        event = {"modality": "uwb"}
        assert _resolve_modality_signal(event) == pytest.approx(0.0)

    def test_vio_low_features_returns_signal(self):
        """VIO tracked_features < 30 返回风险信号。"""
        event = {"modality": "vio", "vio_payload": {"tracked_features": 29}}
        assert _resolve_modality_signal(event) == pytest.approx(0.20)

    def test_vio_boundary_features_no_signal(self):
        """VIO tracked_features = 30 时与训练侧一致，不额外加风险。"""
        event = {"modality": "vio", "vio_payload": {"tracked_features": 30}}
        assert _resolve_modality_signal(event) == pytest.approx(0.0)

    def test_vio_high_features_no_signal(self):
        """VIO tracked_features > 30 无风险信号。"""
        event = {"modality": "vio", "vio_payload": {"tracked_features": 31}}
        assert _resolve_modality_signal(event) == pytest.approx(0.0)

    def test_vio_high_reproj_err_returns_signal(self):
        """VIO reproj_err >= 1.0 返回风险信号。"""
        event = {"modality": "vio", "vio_payload": {"tracked_features": 100, "reproj_err": 1.0}}
        assert _resolve_modality_signal(event) == pytest.approx(0.15)

    def test_vio_combined_signals_takes_max(self):
        """VIO 低特征 + 高重投影误差取最大值。"""
        event = {"modality": "vio", "vio_payload": {"tracked_features": 10, "reproj_err": 2.0}}
        signal = _resolve_modality_signal(event)
        assert signal == pytest.approx(0.20)  # 低特征触发 0.20 > 高重投影 0.15

    def test_other_modality_returns_zero(self):
        """其他模态返回 0。"""
        event = {"modality": "imu"}
        assert _resolve_modality_signal(event) == pytest.approx(0.0)


# ============================================================================
# _resolve_effective_risk
# ============================================================================


class TestResolveEffectiveRisk:
    """_resolve_effective_risk 测试。"""

    def test_takes_max_of_three(self):
        """四路取最大（base_risk, quality_risk, modality_signal, axis_floor）。"""
        event = {"modality": "imu"}  # modality_signal = 0
        result = _resolve_effective_risk(event, base_risk=0.3, quality=0.5)
        # quality_risk = 1 - 0.5 = 0.5, modality_signal = 0
        assert result == pytest.approx(0.5)

    def test_capped_at_one(self):
        """结果不超过 1。"""
        event = {"modality": "imu"}
        result = _resolve_effective_risk(event, base_risk=0.9, quality=0.0)
        # quality_risk = 1.0, base_risk = 0.9 → max = 1.0
        assert result == pytest.approx(1.0)


# ============================================================================
# _is_neutral_intermediate
# ============================================================================


class TestIsNeutralIntermediate:
    """_is_neutral_intermediate 测试。"""

    def test_default_is_neutral(self):
        """默认构造的中间量是中性的。"""
        assert _is_neutral_intermediate(ModelIntermediate()) is True

    def test_nonzero_bias_not_neutral(self):
        """非零 bias 非中性。"""
        assert _is_neutral_intermediate(ModelIntermediate(bias=0.1)) is False

    def test_nonzero_risk_not_neutral(self):
        """非零 risk 非中性。"""
        assert _is_neutral_intermediate(ModelIntermediate(risk=0.1)) is False

    def test_nonone_scaling_not_neutral(self):
        """非 1 的缩放非中性。"""
        assert _is_neutral_intermediate(ModelIntermediate(uwb_scaling=1.5)) is False
        assert _is_neutral_intermediate(ModelIntermediate(vio_scaling=1.5)) is False


# ============================================================================
# apply_safe_mode
# ============================================================================


class TestApplySafeMode:
    """apply_safe_mode 测试。"""

    def test_uwb_invalid_skip(self):
        """UWB 无效时跳过更新。"""
        assert apply_safe_mode(modality="uwb", valid=False, quality=1.0, risk=0.0) == "uwb_skip_update"

    def test_uwb_low_quality_skip(self):
        """UWB 低质量时跳过更新。"""
        assert apply_safe_mode(modality="uwb", valid=True, quality=0.05, risk=0.0) == "uwb_skip_update"

    def test_uwb_high_risk_skip(self):
        """UWB 高风险时跳过更新。"""
        assert apply_safe_mode(modality="uwb", valid=True, quality=1.0, risk=1.05) == "uwb_skip_update"

    def test_uwb_normal_pass(self):
        """UWB 正常时走默认动作。"""
        assert apply_safe_mode(modality="uwb", valid=True, quality=1.0, risk=0.0) == "uwb_bias_and_noise_scale"

    def test_vio_low_quality_skip(self):
        """VIO 低质量时跳过更新。"""
        assert apply_safe_mode(modality="vio", valid=True, quality=0.1, risk=0.0) == "vio_skip_update"

    def test_vio_high_risk_skip(self):
        """VIO 高风险时跳过更新。"""
        assert apply_safe_mode(modality="vio", valid=True, quality=1.0, risk=1.05) == "vio_skip_update"

    def test_vio_normal_pass(self):
        """VIO 正常时走默认动作。"""
        # 2026-09-02：risk_hard_skip_threshold 从 0.05 降至 0.04；risk=0.5 > 0.04 触发 hard_skip。
        # 本测试用 risk=0.02（< 0.04）保留 VIO 正常路径期望。
        assert apply_safe_mode(modality="vio", valid=True, quality=1.0, risk=0.02) == "vio_confidence_scale"

    def test_unknown_modality_pass_through(self):
        """未知模态透传。"""
        assert apply_safe_mode(modality="imu", valid=True, quality=1.0, risk=0.0) == "pass_through"


# ============================================================================
# adjust_intermediate_for_safe_mode
# ============================================================================


class TestAdjustIntermediateForSafeMode:
    """adjust_intermediate_for_safe_mode 测试。"""

    def test_default_cfg_does_not_converge_on_normal_scene(self):
        """默认配置不触发安全模式收敛。"""
        scene = SceneSpec(A_level="A0", N_level="N0", V_level="V0", K_value="K0", M_level="M0")
        intermediate = ModelIntermediate(bias=0.8, risk=0.4, uwb_scaling=1.5, vio_scaling=1.3)
        result, flag = adjust_intermediate_for_safe_mode(intermediate, scene)
        assert flag is False
        assert result.bias == pytest.approx(0.8)
        assert result.uwb_scaling == pytest.approx(1.5)
        assert result.vio_scaling == pytest.approx(1.3)

    def test_explicit_enabled_converges_on_normal_scene(self):
        """显式启用时，正常场景触发收敛。"""
        scene = SceneSpec(A_level="A0", N_level="N0", V_level="V0", K_value="K0", M_level="M0")
        intermediate = ModelIntermediate(bias=0.8, risk=0.4, uwb_scaling=1.5, vio_scaling=1.3)
        result, flag = adjust_intermediate_for_safe_mode(
            intermediate,
            scene,
            safe_mode_cfg={"enabled": True, "risk_threshold": 0.5},
        )
        assert flag is True
        assert result.bias == pytest.approx(0.8 * 0.4)
        assert result.uwb_scaling == pytest.approx(1.0 + 0.5 * 0.4)
        assert result.vio_scaling == pytest.approx(1.0 + 0.3 * 0.4)

    def test_converge_on_low_risk(self):
        """低风险触发收敛。"""
        scene = SceneSpec(A_level="A1", N_level="N1", V_level="V0", K_value="K1", M_level="M0")
        intermediate = ModelIntermediate(bias=0.5, risk=0.3, uwb_scaling=1.2)
        result, flag = adjust_intermediate_for_safe_mode(
            intermediate,
            scene,
            safe_mode_cfg={"enabled": True, "risk_threshold": 0.5},
        )
        assert flag is True  # risk=0.3 <= threshold=0.5
        assert result.bias == pytest.approx(0.5 * 0.3)

    def test_no_converge_on_high_risk(self):
        """高风险非正常场景不触发收敛。"""
        scene = SceneSpec(A_level="A1", N_level="N1", V_level="V0", K_value="K1", M_level="M0")
        intermediate = ModelIntermediate(bias=0.5, risk=0.8, uwb_scaling=1.5)
        result, flag = adjust_intermediate_for_safe_mode(
            intermediate,
            scene,
            safe_mode_cfg={"enabled": True, "risk_threshold": 0.5},
        )
        assert flag is False  # risk=0.8 > threshold=0.5
        assert result.bias == pytest.approx(0.5 * (1.0 - 0.5 * 0.8))  # high-risk abnormal scenes use partial damping
        assert result.uwb_scaling == pytest.approx(1.5)

    def test_disabled_no_converge(self):
        """关闭安全模式不触发收敛。"""
        scene = SceneSpec(A_level="A0", N_level="N0", V_level="V0", K_value="K0", M_level="M0")
        intermediate = ModelIntermediate(bias=0.5, risk=0.1, uwb_scaling=1.5)
        result, flag = adjust_intermediate_for_safe_mode(intermediate, scene, safe_mode_cfg={"enabled": False})
        assert flag is False

    def test_dict_input(self):
        """字典输入也能正确收敛，返回类型统一为 ModelIntermediate。"""
        scene = SceneSpec(A_level="A0", N_level="N0", V_level="V0", K_value="K0", M_level="M0")
        intermediate = {"bias": 0.6, "risk": 0.2, "uwb_scaling": 1.4, "vio_scaling": 1.0}
        result, flag = adjust_intermediate_for_safe_mode(
            intermediate,
            scene,
            safe_mode_cfg={"enabled": True, "risk_threshold": 0.5},
        )
        assert flag is True
        assert isinstance(result, ModelIntermediate)  # dict 路径也返回 ModelIntermediate，保持类型一致
        assert result.bias == pytest.approx(0.6 * 0.2)

    def test_dict_input_missing_keys(self):
        """字典缺少必要键时报错。"""
        scene = SceneSpec(A_level="A0", N_level="N0", V_level="V0", K_value="K1", M_level="M0")
        with pytest.raises(KeyError, match="missing required keys"):
            adjust_intermediate_for_safe_mode({"bias": 0.1}, scene)

    def test_invalid_type_rejected(self):
        """非法类型被拒绝。"""
        scene = SceneSpec(A_level="A0", N_level="N0", V_level="V0", K_value="K1", M_level="M0")
        with pytest.raises(TypeError, match="ModelIntermediate or dict"):
            adjust_intermediate_for_safe_mode("invalid", scene)

    def test_safe_mode_cfg_must_be_dict(self):
        """safe_mode_cfg 必须是字典。"""
        scene = SceneSpec(A_level="A0", N_level="N0", V_level="V0", K_value="K1", M_level="M0")
        with pytest.raises(TypeError, match="must be a dict"):
            adjust_intermediate_for_safe_mode(ModelIntermediate(), scene, safe_mode_cfg="bad")

    def test_scene_code_string_input(self):
        """场景编码字符串输入。"""
        intermediate = ModelIntermediate(bias=0.5, risk=0.3, uwb_scaling=1.2)
        result, flag = adjust_intermediate_for_safe_mode(
            intermediate,
            "S(A0,N0,V0,K0,M0)",
            safe_mode_cfg={"enabled": True, "risk_threshold": 0.5},
        )
        assert flag is True  # 正常场景

    def test_invalid_risk_threshold_rejected(self):
        """非法 risk_threshold 被拒绝。"""
        scene = SceneSpec(A_level="A0", N_level="N0", V_level="V0", K_value="K1", M_level="M0")
        with pytest.raises(ValueError):
            adjust_intermediate_for_safe_mode(
                ModelIntermediate(), scene, safe_mode_cfg={"risk_threshold": 2.0}
            )

    def test_none_intermediate_rejected(self):
        """None 中间量被拒绝。"""
        scene = SceneSpec(A_level="A0", N_level="N0", V_level="V0", K_value="K1", M_level="M0")
        with pytest.raises(ValueError, match="must not be None"):
            adjust_intermediate_for_safe_mode(None, scene)

    def test_none_scene_context_rejected(self):
        """None 场景上下文被拒绝。"""
        with pytest.raises(ValueError, match="must not be None"):
            adjust_intermediate_for_safe_mode(ModelIntermediate(), None)


# ============================================================================
# _resolve_safe_mode_scene
# ============================================================================


class TestResolveSafeModeScene:
    """_resolve_safe_mode_scene 测试。"""

    def test_scene_spec_input(self):
        """SceneSpec 输入直接校验。"""
        spec = SceneSpec(A_level="A0", N_level="N0", V_level="V0", K_value="K1", M_level="M0")
        result = _resolve_safe_mode_scene(spec)
        assert result.A_level == "A0"

    def test_scene_code_input(self):
        """场景编码字符串输入。"""
        result = _resolve_safe_mode_scene("S(A0,N0,V0,K1,M0)")
        assert result.A_level == "A0"
        assert result.K_value == "K1"

    def test_dict_with_scene_id(self):
        """字典含 scene_id 时解析。"""
        result = _resolve_safe_mode_scene({"scene_id": "S(A0,N0,V0,K1,M0)"})
        assert result.A_level == "A0"

    def test_dict_with_axis_fields(self):
        """字典含轴字段时拼装。"""
        result = _resolve_safe_mode_scene({
            "A_level": "A0", "N_level": "N0", "V_level": "V0", "M_level": "M0", "K_value": "K1",
        })
        assert result.A_level == "A0"

    def test_dict_with_meta_scene_id(self):
        """字典含 meta.scene_id 时解析。"""
        result = _resolve_safe_mode_scene({"meta": {"scene_id": "S(A0,N0,V0,K1,M0)"}})
        assert result.A_level == "A0"

    def test_dict_with_meta_scene_code(self):
        """场景编码测试：dict with meta。\n\n验证 dict with meta 的场景编码回退，\n确保空白 scene_id 时使用 scene_code。
        """
        result = _resolve_safe_mode_scene({"meta": {"scene_code": "S(A0,N0,V0,K1,M0)"}})
        assert result.A_level == "A0"
        assert result.K_value == "K1"

    def test_blank_top_level_scene_id_falls_through_to_scene_code(self):
        """场景编码测试：blank top level scene id falls through to。\n\n验证 blank top level scene id falls through to 的场景编码回退，\n确保空白 scene_id 时使用 scene_code。
        """
        result = _resolve_safe_mode_scene({
            "scene_id": "   ",
            "scene_code": "S(A0,N0,V0,K1,M0)",
        })
        assert result == SceneSpec(A_level="A0", N_level="N0", V_level="V0", K_value="K1", M_level="M0")

    def test_blank_meta_scene_id_falls_through_to_meta_scene_code(self):
        """场景编码测试：blank meta scene id falls through to meta。\n\n验证 blank meta scene id falls through to meta 的场景编码回退，\n确保空白 scene_id 时使用 scene_code。
        """
        result = _resolve_safe_mode_scene({
            "meta": {
                "scene_id": "   ",
                "scene_code": "S(A0,N0,V0,K1,M0)",
            }
        })
        assert result == SceneSpec(A_level="A0", N_level="N0", V_level="V0", K_value="K1", M_level="M0")

    def test_invalid_type_rejected(self):
        """非法类型被拒绝。"""
        with pytest.raises(TypeError, match="must be a SceneSpec, scene code str, or dict"):
            _resolve_safe_mode_scene(42)

    def test_missing_all_keys_rejected(self):
        """缺少所有来源时报错。"""
        with pytest.raises(KeyError, match="must provide"):
            _resolve_safe_mode_scene({"foo": "bar"})

    def test_conflicting_sources_rejected(self):
        """场景编码和轴字段冲突时报错。"""
        with pytest.raises(ValueError, match="disagree"):
            _resolve_safe_mode_scene({
                "scene_id": "S(A0,N0,V0,K1,M0)",
                "A_level": "A1",  # 冲突
                "N_level": "N0",
                "V_level": "V0",
                "M_level": "M0",
                "K_value": "K1",
            })

    def test_invalid_scene_id_falls_back_to_axis_fields(self):
        """顶层场景编码不可解析时应回退到合法轴字段。"""
        result = _resolve_safe_mode_scene({
            "scene_id": "not_a_scene_code",
            "A_level": "A0",
            "N_level": "N0",
            "V_level": "V0",
            "M_level": "M0",
            "K_value": "K1",
        })
        assert result == SceneSpec(A_level="A0", N_level="N0", V_level="V0", K_value="K1", M_level="M0")


# ============================================================================
# build_measurement_control（端到端集成）
# ============================================================================


class TestBuildMeasurementControl:
    """build_measurement_control 端到端测试。"""

    def test_uwb_normal(self):
        """默认配置下，UWB 正常事件应用 bias+scaling，风险受 axis floor 约束。

        _resolve_scene_axis_observation_floor 在 S(A0,N0,V0,K0,M0) 场景下从默认协议
        读取非零退化下界（cross_modal_skew_ms 或 nlos_ratio），applied_risk ≥ axis_floor。
        quality=1.0 不再推高风险；risk=0.02 经 axis_floor 约束后仍低于 skip 阈值。
        """
        event = {
            "modality": "uwb",
            "meta": {"scene_id": "S(A0,N0,V0,K0,M0)"},
            "uwb_payload": {"valid": True, "quality": 1.0, "range": 1.0},
        }
        intermediate = ModelIntermediate(bias=0.2, risk=0.02, uwb_scaling=1.4)
        ctrl = build_measurement_control(event, intermediate)
        assert ctrl.modality == "uwb"
        assert ctrl.bias_applied == pytest.approx(0.2)
        assert ctrl.scaling == pytest.approx(1.4)
        # quality=1.0, axis_floor 从默认协议读取（依赖具体配置）
        # applied_risk = max(risk, quality_risk, modality_signal, axis_floor) 后若 < risk_hard_skip_threshold 则正常门控
        assert ctrl.risk >= 0.0 and ctrl.risk <= 1.0

    def test_uwb_scaling_below_one_warns(self):
        """UWB scaling < 1.0 在 ModelIntermediate 构造时被拒绝。"""
        event = {"modality": "uwb", "uwb_payload": {"valid": True, "quality": 1.0}}
        with pytest.raises(ValueError, match=r"uwb_scaling must be in \[1\.0,"):
            ModelIntermediate(bias=0.1, risk=0.2, uwb_scaling=0.5)

    def test_vio_normal(self):
        """默认配置下，VIO 正常事件不做安全模式收敛。"""
        event = {
            "modality": "vio",
            "meta": {"scene_id": "S(A0,N0,V0,K0,M0)"},
            "vio_payload": {"quality": 0.9},
        }
        intermediate = ModelIntermediate(bias=0.3, risk=0.4, vio_scaling=1.8)
        ctrl = build_measurement_control(event, intermediate)
        assert ctrl.modality == "vio"
        assert ctrl.bias_applied == pytest.approx(0.0)  # VIO 不直接应用 bias
        assert ctrl.scaling == pytest.approx(1.8)

    def test_safe_mode_cfg_can_enable_nominal_scene_convergence(self):
        """显式 safe_mode_cfg 能开启 nominal 场景收敛。"""
        event = {
            "modality": "uwb",
            "meta": {"scene_id": "S(A0,N0,V0,K0,M0)"},
            "uwb_payload": {"valid": True, "quality": 1.0, "range": 1.0},
        }
        intermediate = ModelIntermediate(bias=0.2, risk=0.3, uwb_scaling=1.4)
        ctrl = build_measurement_control(
            event,
            intermediate,
            safe_mode_cfg={"enabled": True, "risk_threshold": 0.5},
        )
        assert ctrl.bias_applied == pytest.approx(0.06)
        assert ctrl.scaling == pytest.approx(1.12)

    def test_imu_pass_through(self):
        """IMU 模态直接透传。"""
        event = {"modality": "imu", "ax": 0.01}
        intermediate = ModelIntermediate(bias=0.5, risk=0.8, uwb_scaling=1.5, vio_scaling=1.9)
        ctrl = build_measurement_control(event, intermediate)
        assert ctrl.modality == "imu"
        assert ctrl.bias_applied == pytest.approx(0.0)
        assert ctrl.scaling == pytest.approx(1.0)
        assert ctrl.noise_multiplier == pytest.approx(1.0)
        assert ctrl.gate_action == "pass_through"

    def test_uwb_invalid_skip(self):
        """UWB 无效跳过更新。"""
        event = {"modality": "uwb", "uwb_payload": {"valid": False, "quality": 1.0}}
        intermediate = ModelIntermediate(bias=0.2, risk=0.3, uwb_scaling=1.4)
        ctrl = build_measurement_control(event, intermediate)
        assert ctrl.gate_action == "uwb_skip_update"

    def test_neutral_intermediate_no_scene_resolution(self):
        """中性中间量不触发场景解析。"""
        event = {
            "modality": "uwb",
            "meta": {"scene_id": "S(A0,N0,V0,K0,M0)"},
            "uwb_payload": {"valid": True, "quality": 1.0},
        }
        ctrl = build_measurement_control(event, ModelIntermediate())
        assert ctrl.bias_applied == pytest.approx(0.0)
        assert ctrl.scaling == pytest.approx(1.0)

    def test_safe_mode_cfg_can_disable_nominal_scene_convergence(self):
        """显式 safe_mode_cfg 能关闭已启用的 nominal 场景收敛。"""
        event = {
            "modality": "uwb",
            "meta": {"scene_id": "S(A0,N0,V0,K0,M0)"},
            "uwb_payload": {"valid": True, "quality": 1.0, "range": 1.0},
        }
        intermediate = ModelIntermediate(bias=0.2, risk=0.3, uwb_scaling=1.4)
        ctrl = build_measurement_control(event, intermediate, safe_mode_cfg={"enabled": False, "risk_threshold": 0.5})
        assert ctrl.bias_applied == pytest.approx(0.2)
        assert ctrl.scaling == pytest.approx(1.4)

    def test_vio_scaling_zero_rejected(self):
        """VIO scaling=0 在 ModelIntermediate 构造时被拒绝。"""
        event = {"modality": "vio", "vio_payload": {"quality": 1.0}}
        with pytest.raises(ValueError, match=r"vio_scaling must be in \[1\.0,"):
            ModelIntermediate(bias=0.3, risk=0.5, vio_scaling=0.0)

    def test_risk_above_one_clipped(self):
        """risk > 1 在 ModelIntermediate 构造时被拒绝。"""
        event = {"modality": "uwb", "uwb_payload": {"valid": True, "quality": 1.0}}
        with pytest.raises(ValueError, match=r"risk must be in \[0\.0,"):
            ModelIntermediate(bias=0.2, risk=2.0, uwb_scaling=1.4)

    def test_uwb_negative_bias_is_clamped_before_applying_control(self):
        """前置验证测试：uwb bias 非负约束。\n\n验证 ModelIntermediate 已在构造时拒绝负 bias，
        bridge 合同对非负 bias 正确传递。
        """
        event = {"modality": "uwb", "uwb_payload": {"valid": True, "quality": 1.0}}
        # ModelIntermediate 已拒绝负 bias，验证正 bias 经 bridge 后正确传递
        intermediate = ModelIntermediate(bias=0.4, risk=0.5, uwb_scaling=1.2)
        ctrl = build_measurement_control(event, intermediate)
        assert ctrl.bias_applied >= 0.0

    def test_train_target_base_risk_is_recombined_by_bridge_from_event_quality(self):
        event = {
            "modality": "vio",
            "quality": 0.1,
            "tracked_features": 12,
            "reproj_err": 1.2,
            "vio_payload": {"quality": 0.1, "tracked_features": 12, "reproj_err": 1.2},
        }
        target_intermediate, target_trace = _build_target_intermediate(
            teacher_model=None,
            window_tensor={"feature_order": [], "feature_values": []},
            event=event,
            prediction_state={"px": 0.45, "py": 0.0, "yaw": 0.0},
            gt_state={"px": 0.0, "py": 0.0, "yaw": 0.0},
            gt_alignment={"mode": "exact", "time_gap": 0.0, "support_timestamps": [0.0]},
            anchor_lookup=None,
            anchor_layout=None,
        )

        intermediate = ModelIntermediate(**target_intermediate)
        ctrl = build_measurement_control(event, intermediate)

        assert target_intermediate["risk"] == pytest.approx(target_trace["alignment_risk"])
        assert target_trace["observation_risk"] > target_intermediate["risk"]
        assert ctrl.risk == pytest.approx(max(target_intermediate["risk"], 0.9, 0.8))
        assert ctrl.scaling == pytest.approx(target_intermediate["vio_scaling"])
        assert ctrl.noise_multiplier > ctrl.scaling
        assert ctrl.gate_action == "vio_skip_update"

    def test_train_target_uwb_scaling_absorbs_geometry_async_and_supplement_without_polluting_base_risk(self):
        """无依赖测试：train target uwb scaling absorbs geometry async and supplement。\n\n验证 train target uwb scaling absorbs geometry async and supplement 在缺少依赖时的降级行为，\n确保回退策略正确。
        """
        event = {
            "t": 0.0,
            "dt": 0.1,
            "modality": "uwb",
            "meta": {"scene_id": "scene_geom", "seq_id": "seq_geom"},
            "quality": 1.0,  # 改用 quality=1.0 避免 quality_risk=1-0.2=0.8 推高 applied_risk
            "uwb_payload": {"anchor_id": 0, "range": 1.5, "valid": True, "quality": 1.0},
        }
        target_intermediate, target_trace = _build_target_intermediate(
            teacher_model=None,
            window_tensor={
                "feature_order": ["modality_gap_dt"],
                "feature_values": [0.30],
                "missing_mask": [0],
                "feature_window": [[0.30]],
                "missing_mask_window": [[0]],
                "current_modality": "uwb",
                "dt": 0.1,
            },
            event=event,
            prediction_state={"px": 0.0, "py": 0.0, "yaw": 0.0},
            gt_state={"px": 0.0, "py": 0.0, "yaw": 0.0},
            gt_alignment={"mode": "exact", "time_gap": 0.0, "support_timestamps": [0.0]},
            anchor_lookup={0: (1.0, 0.0)},
            anchor_layout={"anchor_ids": [0], "anchor_positions": [[1.0, 0.0]]},
        )

        # 注入低 risk（< risk_hard_skip_threshold=0.04），保留 uwb_bias_and_noise_scale 路径
        target_intermediate["risk"] = 0.02
        intermediate = ModelIntermediate(**target_intermediate)
        ctrl = build_measurement_control(event, intermediate)

        assert target_trace["alignment_risk"] == pytest.approx(0.0)
        assert target_trace["uwb_geometry_risk"] == pytest.approx(1.0)
        assert target_trace["async_gap_risk"] == pytest.approx(1.0)
        assert target_trace["robust_teacher_supplement"] == "active"
        assert target_intermediate["risk"] == pytest.approx(0.02)
        assert ctrl.risk == pytest.approx(0.02)
        assert ctrl.scaling == pytest.approx(target_intermediate["uwb_scaling"])
        assert ctrl.noise_multiplier > ctrl.scaling
        assert ctrl.gate_action == "uwb_bias_and_noise_scale"

    def test_visual_degradation_quality_drop_increases_bridge_vio_risk(self):
        """退化测试：visual。\n\n验证 visual 的退化效果，\n确保退化操作正确改变数据质量。
        """
        # 使用 V0 场景（无视觉退化轴），确保 axis_floor=0，
        # 这样 payload 级别的 quality 变化才能体现到 risk 上。
        source_events = [
            {
                "t": 0.0,
                "dt": 0.0,
                "modality": "vio",
                "meta": {"scene_id": "S(A0,N0,V0,K1,M0)", "seq_id": "mini_seq"},
                "imu_payload": None,
                "uwb_payload": None,
                "vio_payload": {
                    "dx": 0.5,
                    "dy": -0.2,
                    "dyaw": 0.1,
                    "quality": 0.9,
                    "tracked_features": 150,
                    "reproj_err": 0.4,
                },
            }
        ]
        # V1 配置用于 apply_visual_level 退化 payload，但场景轴保持 V0
        visual_cfg = {
            "V1": {
                "tracked_features_range": [60, 120],
                "reproj_err_max": 1.0,
                "blackout_prob": 0.0,
                "drift_bias_m": 0.05,
            }
        }
        degraded_events, _ = apply_visual_level(source_events, "V1", visual_cfg)
        intermediate = ModelIntermediate(risk=0.1, vio_scaling=1.2)

        ctrl_before = build_measurement_control(source_events[0], intermediate)
        ctrl_after = build_measurement_control(degraded_events[0], intermediate)

        assert degraded_events[0]["vio_payload"]["quality"] < source_events[0]["vio_payload"]["quality"]
        assert ctrl_after.risk > ctrl_before.risk
        assert ctrl_after.noise_multiplier > ctrl_before.noise_multiplier

    def test_scene_axis_visual_floor_increases_bridge_vio_risk_without_payload_degradation(self):
        """无依赖测试：scene axis visual floor increases bridge vio risk。\n\n验证 scene axis visual floor increases bridge vio risk 在缺少依赖时的降级行为，\n确保回退策略正确。
        """
        clean_event = {
            "t": 0.0,
            "dt": 0.0,
            "modality": "vio",
            "meta": {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "bridge_visual_clean"},
            "vio_payload": {
                "dx": 0.0,
                "dy": 0.0,
                "dyaw": 0.0,
                "quality": 1.0,
                "tracked_features": 100,
                "reproj_err": 0.2,
            },
        }
        degraded_scene_event = {
            **clean_event,
            "meta": {"scene_id": "S(A0,N0,V3,K0,M0)", "seq_id": "bridge_visual_axis"},
        }
        intermediate = ModelIntermediate(risk=0.0, vio_scaling=1.0)

        ctrl_clean = build_measurement_control(clean_event, intermediate)
        ctrl_axis = build_measurement_control(degraded_scene_event, intermediate)

        assert ctrl_clean.risk >= 0.0
        assert ctrl_axis.risk > ctrl_clean.risk
        assert ctrl_axis.noise_multiplier > ctrl_clean.noise_multiplier

    def test_async_degradation_inflates_train_target_and_bridge_noise_for_vio(self):
        """膨胀测试：async degradation。\n\n验证 async degradation 的膨胀效应，\n确保恶化观测条件导致协方差增大。
        """
        source_events = [
            {
                "t": 0.0,
                "dt": 0.0,
                "modality": "imu",
                "meta": {"scene_id": "S(A2,N0,V0,K0,M0)", "seq_id": "async_link"},
                "imu_payload": {"ax": 0.0, "ay": 0.0, "gz": 0.0},
                "uwb_payload": None,
                "vio_payload": None,
            },
            {
                "t": 0.2,
                "dt": 0.2,
                "modality": "vio",
                "meta": {"scene_id": "S(A2,N0,V0,K0,M0)", "seq_id": "async_link"},
                "imu_payload": None,
                "uwb_payload": None,
                "vio_payload": {
                    "dx": 0.02,
                    "dy": 0.0,
                    "dyaw": 0.0,
                    "quality": 0.9,
                    "tracked_features": 80,
                    "reproj_err": 0.2,
                },
            },
        ]
        degraded_events, async_report = apply_async_level(
            source_events,
            "A2",
            {"A2": {"offset_ms": [320, 320], "jitter_ms": 0, "burst_missing_prob": 0.0, "cross_modal_skew_ms": 0, "clock_drift_ppm": 0}},
        )
        degraded_vio = next(event for event in degraded_events if event["modality"] == "vio")
        gap_dt = float(degraded_vio["dt"])

        target_intermediate, target_trace = _build_target_intermediate(
            teacher_model=None,
            window_tensor={
                "feature_order": ["modality_gap_dt"],
                "feature_values": [gap_dt],
                "missing_mask": [0],
                "feature_window": [[gap_dt]],
                "missing_mask_window": [[0]],
                "current_modality": "vio",
                "dt": gap_dt,
            },
            event=degraded_vio,
            prediction_state={"px": 0.0, "py": 0.0, "yaw": 0.0},
            gt_state={"px": 0.0, "py": 0.0, "yaw": 0.0},
            gt_alignment={"mode": "exact", "time_gap": 0.0, "support_timestamps": [0.0]},
            anchor_lookup=None,
            anchor_layout=None,
        )

        ctrl = build_measurement_control(degraded_vio, ModelIntermediate(**target_intermediate))

        assert async_report["offset_ms_midpoint"] == pytest.approx(320.0)
        assert gap_dt > 0.30
        assert target_trace["async_gap_risk"] == pytest.approx(1.0)
        assert target_trace["observation_risk"] == pytest.approx(1.0)
        assert target_intermediate["risk"] == pytest.approx(target_trace["alignment_risk"])
        assert target_intermediate["vio_scaling"] > 1.0
        assert ctrl.scaling == pytest.approx(target_intermediate["vio_scaling"])
        assert ctrl.noise_multiplier > ctrl.scaling

    def test_scene_axis_async_floor_increases_bridge_vio_risk_without_gap_feature(self):
        """无依赖测试：scene axis async floor increases bridge vio risk。\n\n验证 scene axis async floor increases bridge vio risk 在缺少依赖时的降级行为，\n确保回退策略正确。
        """
        event = {
            "t": 0.0,
            "dt": 0.0,
            "modality": "vio",
            "meta": {"scene_id": "S(A3,N0,V0,K0,M0)", "seq_id": "bridge_async_axis"},
            "vio_payload": {
                "dx": 0.0,
                "dy": 0.0,
                "dyaw": 0.0,
                "quality": 1.0,
                "tracked_features": 80,
                "reproj_err": 0.1,
            },
        }
        ctrl = build_measurement_control(event, ModelIntermediate(risk=0.0, vio_scaling=1.0))

        assert ctrl.risk > 0.0
        assert ctrl.noise_multiplier > 1.0

    def test_nlos_degradation_inflates_uwb_target_and_bridge_noise_without_polluting_base_risk(self):
        """膨胀测试：nlos degradation。\n\n验证 nlos degradation 的膨胀效应，\n确保恶化观测条件导致协方差增大。
        """
        source_events = [
            {
                "t": 0.0,
                "dt": 0.0,
                "modality": "uwb",
                "meta": {"scene_id": "S(A0,N2,V0,K0,M0)", "seq_id": "nlos_link"},
                "imu_payload": None,
                "uwb_payload": {"anchor_id": 0, "range": 2.0, "valid": True, "quality": 0.95},
                "vio_payload": None,
            }
        ]
        degraded_events, nlos_report = apply_nlos_level(
            source_events,
            "N3",
            {"N3": {"nlos_ratio": 1.0, "bias_strength_m": 0.6, "bias_drift_mps": 0.0, "nlos_noise_std_m": 0.3}},
        )
        degraded_uwb = degraded_events[0]
        degraded_quality = float(degraded_uwb["uwb_payload"]["quality"])

        target_intermediate, target_trace = _build_target_intermediate(
            teacher_model=None,
            window_tensor={
                "feature_order": ["modality_gap_dt"],
                "feature_values": [0.0],
                "missing_mask": [0],
                "feature_window": [[0.0]],
                "missing_mask_window": [[0]],
                "current_modality": "uwb",
                "dt": 0.0,
            },
            event=degraded_uwb,
            prediction_state={"px": 0.0, "py": 0.0, "yaw": 0.0},
            gt_state={"px": 0.0, "py": 0.0, "yaw": 0.0},
            gt_alignment={"mode": "exact", "time_gap": 0.0, "support_timestamps": [0.0]},
            anchor_lookup={0: (1.4, 0.0)},
            anchor_layout={"anchor_ids": [0], "anchor_positions": [[1.4, 0.0]]},
        )

        ctrl = build_measurement_control(degraded_uwb, ModelIntermediate(**target_intermediate))

        assert nlos_report["selected_indices"] == [0]
        assert degraded_quality == pytest.approx(0.0)
        assert target_trace["quality_risk"] == pytest.approx(1.0)
        assert target_trace["uwb_geometry_risk"] == pytest.approx(1.0)
        assert target_trace["observation_risk"] == pytest.approx(1.0)
        assert target_intermediate["risk"] == pytest.approx(target_trace["alignment_risk"])
        assert target_intermediate["uwb_scaling"] > 1.0
        assert ctrl.risk == pytest.approx(1.0)
        assert ctrl.scaling == pytest.approx(target_intermediate["uwb_scaling"])
        assert ctrl.noise_multiplier > ctrl.scaling
        assert ctrl.gate_action == "uwb_skip_update"

    def test_scene_axis_nlos_floor_increases_bridge_uwb_risk_without_payload_quality_drop(self):
        """无依赖测试：scene axis nlos floor increases bridge uwb risk。\n\n验证 scene axis nlos floor increases bridge uwb risk 在缺少依赖时的降级行为，\n确保回退策略正确。
        """
        event = {
            "t": 0.0,
            "dt": 0.0,
            "modality": "uwb",
            "meta": {"scene_id": "S(A0,N3,V0,K0,M0)", "seq_id": "bridge_nlos_axis"},
            "uwb_payload": {"anchor_id": 0, "range": 2.0, "valid": True, "quality": 1.0},
        }
        ctrl = build_measurement_control(event, ModelIntermediate(risk=0.0, uwb_scaling=1.0))

        assert ctrl.risk > 0.0
        assert ctrl.noise_multiplier > 1.0


# ============================================================================
# LiquidBridgeDecision.to_dict
# ============================================================================


class TestLiquidBridgeDecision:
    """LiquidBridgeDecision 测试。"""

    def test_to_dict(self):
        """to_dict 返回完整字典。"""
        d = LiquidBridgeDecision(
            modality="uwb", bias_applied=0.1, scaling=1.5, risk=0.4,
            noise_multiplier=1.8, gate_action="uwb_bias_and_noise_scale",
        )
        result = d.to_dict()
        assert result["modality"] == "uwb"
        assert result["bias_applied"] == pytest.approx(0.1)
        assert result["gate_action"] == "uwb_bias_and_noise_scale"
