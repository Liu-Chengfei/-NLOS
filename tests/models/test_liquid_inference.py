"""Liquid 推理（liquid_inference）测试模块。

测试覆盖范围：
- Liquid 模型的推理流程
- 中间输出的语义验证
- 检查点加载与设备兼容性

被测模块：liquidloc.models.liquid_inference"""

import math

import pytest
import torch
import torch.nn.functional as F
from liquidloc.factories.model_factory import create_model
from liquidloc.models.liquid.output_head import LIQUID_CONTEXT_FEATURE_KEYS as _LIQUID_CONTEXT_FEATURE_KEYS
from liquidloc.models.liquid.inference import infer_intermediate


def _uwb_window():
    return {
        "current_modality": "uwb",
        "feature_order": ["quality", "valid"],
        "feature_values": [0.2, 0.0],
        "missing_mask": [0, 1],
        "dt": 0.1,
        "feature_window": [[0.2, 0.0]],
        "missing_mask_window": [[0, 1]],
    }


def _vio_window():
    return {
        "current_modality": "vio",
        "feature_order": ["tracked_features", "reproj_err"],
        "feature_values": [12.0, 0.8],
        "missing_mask": [0, 0],
        "dt": 0.1,
        "feature_window": [[12.0, 0.8]],
        "missing_mask_window": [[0, 0]],
    }


class _PrecomputedNetwork:
    def __init__(self, raw_outputs):
        self._raw_outputs = dict(raw_outputs)

    def extract_shared_features(self, window_tensor):
        return dict(self._raw_outputs)


class _PrecomputedState:
    def __init__(self, raw_outputs):
        self.network = _PrecomputedNetwork(raw_outputs)
        self.output_heads = {
            "bias": lambda shared_features: torch.as_tensor(raw_outputs["bias"], dtype=torch.float32),
            "risk": lambda shared_features: torch.as_tensor(raw_outputs["risk"], dtype=torch.float32),
            "uwb_scaling": lambda shared_features: torch.as_tensor(raw_outputs["uwb_scaling"], dtype=torch.float32),
            "vio_scaling": lambda shared_features: torch.as_tensor(raw_outputs["vio_scaling"], dtype=torch.float32),
        }


def test_normal_case():
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    model = create_model("liquid_ekf", {})
    outputs = infer_intermediate(_uwb_window(), model_state=model)
    # infer_intermediate 返回 ModelIntermediate 对象，不是字典。
    from liquidloc.common.constants import BRIDGE_SCALING_MIN
    assert hasattr(outputs, "bias") and hasattr(outputs, "risk") and hasattr(outputs, "uwb_scaling") and hasattr(outputs, "vio_scaling")
    assert all(math.isfinite(getattr(outputs, k)) for k in ("bias", "risk", "uwb_scaling", "vio_scaling"))
    # v2: scaling 下界放宽至 BRIDGE_SCALING_MIN（0.5），配合 soft-mask 释放 Liquid 调节空间
    assert outputs.uwb_scaling >= BRIDGE_SCALING_MIN
    assert outputs.vio_scaling >= BRIDGE_SCALING_MIN


def test_boundary_case():
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    outputs = infer_intermediate(
        _uwb_window(),
        model_state=_PrecomputedState(
            {
                "bias": -1.0,
                "risk": 1.5,
                "uwb_scaling": -0.2,
                "vio_scaling": 1.25,
            }
        ),
    )
    assert outputs.bias == pytest.approx(0.0, rel=1e-6)  # bias 非负约束
    assert outputs.risk == pytest.approx(float(torch.sigmoid(torch.tensor(1.5))), rel=1e-6)
    assert outputs.uwb_scaling == pytest.approx(1.0, rel=1e-6)
    from liquidloc.common.constants import BRIDGE_SCALING_MIN
    assert outputs.vio_scaling >= BRIDGE_SCALING_MIN


def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    with pytest.raises(ValueError):
        infer_intermediate(_uwb_window(), model_state=None)


def test_model_predict_intermediate_tensors_applies_vio_uwb_neutral_floor():
    """应用测试：model predict intermediate tensors。

    验证 model predict intermediate tensors 的应用逻辑，
    确保特定条件触发预期行为。
    
    v3 改造:
    - scaling_ceiling 从 1.0 放宽到 2.5 (原 max=1.0 强制 clamp 到 1.0 现已放宽)
    - scaling_min = 1.0 (保持不变，回退到 1.0 是因为 v2 [0.5, 1.0] 在 e9 场景下让 Liquid 不当降权好测量)
    - uwb_scaling_head.projection.bias.fill_(2.0) → head 输出 2.0
    - neutral_floor_softplus(2.0, neutral_floor=1.0) = 1.0 + softplus(2.0) - softplus(0)
                                                  = 1.0 + 2.1269 - 0.6931
                                                  = 2.4338
    """
    model = create_model("liquid_ekf", {})
    with torch.no_grad():
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()
        model.uwb_scaling_head.projection.weight.zero_()
        model.uwb_scaling_head.projection.bias.fill_(2.0)  # softplus(2.0) ≈ 2.1269
        model.vio_scaling_head.projection.weight.zero_()
        model.vio_scaling_head.projection.bias.zero_()

    outputs = model.predict_intermediate_tensors(_vio_window())

    # v3: scaling_ceiling=2.5 放宽后，不再被 clamp 到 1.0
    # neutral_floor_softplus(2.0, neutral_floor=1.0) = 1.0 + 2.1269 - 0.6931 = 2.4338
    import math
    expected_uwb_scaling = 1.0 + math.log(math.exp(2.0) + 1.0) - math.log(2.0)  # neutral_floor=1.0
    assert outputs["uwb_scaling"].item() == pytest.approx(expected_uwb_scaling, rel=1e-6)
    assert outputs["risk"].item() == pytest.approx(float(torch.sigmoid(torch.tensor(0.0))), rel=1e-6)
    # v3: scaling_min = 1.0 (回退到 1.0 是 e9 场景的设计选择)
    assert outputs["vio_scaling"].item() >= 1.0


def test_model_predict_intermediate_tensors_applies_uwb_vio_neutral_floor():
    """应用测试：model predict intermediate tensors。

    验证 model predict intermediate tensors 的应用逻辑，
    确保特定条件触发预期行为。

    v3 改造:
    - scaling_ceiling 从 1.0 放宽到 2.5
    - scaling_min = 1.0 (v2 的 0.5 仅在 e9 不当降权已回退)
    - vio_scaling_head.projection.bias.fill_(2.0) → head 输出 2.0
    - neutral_floor_softplus(2.0, neutral_floor=1.0) = 1.0 + softplus(2.0) - softplus(0) ≈ 2.4338
    """
    model = create_model("liquid_ekf", {})
    with torch.no_grad():
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()
        model.uwb_scaling_head.projection.weight.zero_()
        model.uwb_scaling_head.projection.bias.zero_()
        model.vio_scaling_head.projection.weight.zero_()
        model.vio_scaling_head.projection.bias.fill_(2.0)  # softplus(2.0) ≈ 2.1269

    outputs = model.predict_intermediate_tensors(_uwb_window())

    # v3: scaling_ceiling=2.5 放宽后，vio_scaling = neutral_floor_softplus(2.0, 1.0) ≈ 2.4338
    import math
    expected_vio_scaling = 1.0 + math.log(math.exp(2.0) + 1.0) - math.log(2.0)
    assert outputs["vio_scaling"].item() == pytest.approx(expected_vio_scaling, rel=1e-6)
    assert outputs["risk"].item() == pytest.approx(float(torch.sigmoid(torch.tensor(0.0))), rel=1e-6)
    # v3: scaling_min = 1.0
    assert outputs["uwb_scaling"].item() >= 1.0


def test_liquid_inference_keeps_signed_bias_semantics():
    """保持测试：liquid inference。\n\n验证 liquid inference 的保持行为，\n确保特定属性在处理过程中不变。
    """
    outputs = infer_intermediate(
        _uwb_window(),
        model_state=_PrecomputedState(
            {
                "bias": -0.6,
                "risk": 0.0,
                "uwb_scaling": 0.0,
                "vio_scaling": 0.0,
            }
        ),
    )
    assert outputs.bias == pytest.approx(0.0, rel=1e-6)  # bias 非负约束


def test_liquid_output_head_uses_modality_aware_dynamic_branch_mixing():
    """使用测试：liquid output head。\n\n验证被测功能正确使用 liquid output head，\n确保内部依赖被正确调用。
    """
    model = create_model("liquid_ekf", {})
    hidden_dim = model.network.hidden_dim
    with torch.no_grad():
        model.bias_head.temporal_context_gate.weight.zero_()
        model.bias_head.temporal_context_gate.bias.zero_()
        model.bias_head.observation_context_gate.weight.zero_()
        model.bias_head.observation_context_gate.bias.zero_()
        model.bias_head.branch_mix_gate.weight.zero_()
        model.bias_head.branch_mix_gate.bias.zero_()
        model.bias_head.uwb_branch_mix_gate.weight.zero_()
        model.bias_head.uwb_branch_mix_gate.bias.fill_(-10.0)
        model.bias_head.vio_branch_mix_gate.weight.zero_()
        model.bias_head.vio_branch_mix_gate.bias.fill_(10.0)
        model.bias_head.filter_context_gate.weight.zero_()
        model.bias_head.filter_context_gate.bias.zero_()
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.projection.weight[0, 0] = 1.0

    uwb_shared = {
        "shared_vector": torch.zeros(hidden_dim, dtype=torch.float32),
        "final_hidden": torch.tensor([1.0] + ([0.0] * (hidden_dim - 1)), dtype=torch.float32),
        "pooled_hidden": torch.tensor([-1.0] + ([0.0] * (hidden_dim - 1)), dtype=torch.float32),
        "current_modality": "uwb",
        "current_feature_by_name": {"modality_gap_dt": 0.25},
        "current_observed_by_name": {"modality_gap_dt": True},
        "readout_context_by_name": {
            "state_cov_trace": 0.0,
            "pos_cov": 0.0,
            "last_innovation_norm": 0.0,
            "last_gate_skip_flag": 0.0,
        },
        "readout_context_observed_by_name": {
            "state_cov_trace": False,
            "pos_cov": False,
            "last_innovation_norm": False,
            "last_gate_skip_flag": False,
        },
    }
    vio_shared = dict(uwb_shared)
    vio_shared["current_modality"] = "vio"

    uwb_output = float(model.bias_head(uwb_shared).detach().item())
    vio_output = float(model.bias_head(vio_shared).detach().item())

    assert uwb_output > 0.5
    assert vio_output < -0.5


def test_liquid_output_head_uses_uwb_branch_mix_gate_for_uwb_modality():
    """使用测试：liquid output head。\n\n验证被测功能正确使用 liquid output head，\n确保 UWB 模态使用 uwb_branch_mix_gate。
    """
    model = create_model("liquid_ekf", {})
    hidden_dim = model.network.hidden_dim
    with torch.no_grad():
        model.bias_head.temporal_context_gate.weight.zero_()
        model.bias_head.temporal_context_gate.bias.zero_()
        model.bias_head.observation_context_gate.weight.zero_()
        model.bias_head.observation_context_gate.bias.zero_()
        model.bias_head.filter_context_gate.weight.zero_()
        model.bias_head.filter_context_gate.bias.zero_()
        model.bias_head.branch_mix_gate.weight.zero_()
        model.bias_head.branch_mix_gate.bias.fill_(10.0)
        model.bias_head.uwb_branch_mix_gate.weight.zero_()
        model.bias_head.uwb_branch_mix_gate.bias.fill_(-10.0)
        model.bias_head.vio_branch_mix_gate.weight.zero_()
        model.bias_head.vio_branch_mix_gate.bias.fill_(10.0)
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.projection.weight[0, 0] = 1.0

    shared = {
        "shared_vector": torch.zeros(hidden_dim, dtype=torch.float32),
        "final_hidden": torch.tensor([1.0] + ([0.0] * (hidden_dim - 1)), dtype=torch.float32),
        "pooled_hidden": torch.tensor([-1.0] + ([0.0] * (hidden_dim - 1)), dtype=torch.float32),
        "current_modality": "uwb",
        "current_feature_by_name": {"modality_gap_dt": 0.25},
        "current_observed_by_name": {"modality_gap_dt": True},
        "readout_context_by_name": {
            "state_cov_trace": 0.0,
            "pos_cov": 0.0,
            "last_innovation_norm": 0.0,
            "last_gate_skip_flag": 0.0,
        },
        "readout_context_observed_by_name": {
            "state_cov_trace": False,
            "pos_cov": False,
            "last_innovation_norm": False,
            "last_gate_skip_flag": False,
        },
    }

    output = float(model.bias_head(shared).detach().item())

    # uwb_branch_mix_gate.bias=-10 → sigmoid≈0 → fast权重≈1 → final_hidden[0]=1.0 主导
    assert output > 0.5


def test_liquid_inference_uses_monotonic_risk_calibration_when_available():
    """使用测试：liquid inference。\n\n验证被测功能正确使用 liquid inference，\n确保内部依赖被正确调用。
    """
    model = create_model("liquid_ekf", {})
    with torch.no_grad():
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()
        model.risk_calibration.a_raw.copy_(torch.log(torch.expm1(torch.tensor(1.0))))
        model.risk_calibration.b.fill_(0.3)

    outputs = infer_intermediate(_uwb_window(), model_state=model)

    expected = float(torch.sigmoid(torch.tensor(0.3)))
    assert outputs.risk == pytest.approx(expected, rel=1e-6)


def test_liquid_output_head_residual_projection_can_bypass_zero_main_projection():
    """零值测试：liquid output head residual projection can bypass。\n\n验证 liquid output head residual projection can bypass 在零值输入下的行为，\n确保边界情况正确处理。
    """
    model = create_model("liquid_ekf", {})
    hidden_dim = model.network.hidden_dim
    with torch.no_grad():
        model.bias_head.temporal_context_gate.weight.zero_()
        model.bias_head.temporal_context_gate.bias.zero_()
        model.bias_head.observation_context_gate.weight.zero_()
        model.bias_head.observation_context_gate.bias.zero_()
        model.bias_head.filter_context_gate.weight.zero_()
        model.bias_head.filter_context_gate.bias.zero_()
        model.bias_head.branch_mix_gate.weight.zero_()
        model.bias_head.branch_mix_gate.bias.zero_()
        model.bias_head.uwb_branch_mix_gate.weight.zero_()
        model.bias_head.uwb_branch_mix_gate.bias.zero_()
        model.bias_head.vio_branch_mix_gate.weight.zero_()
        model.bias_head.vio_branch_mix_gate.bias.zero_()
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.residual_projection.weight.zero_()
        model.bias_head.residual_projection.bias.zero_()
        model.bias_head.residual_projection.weight[0, 0] = 1.0

    shared = {
        "shared_vector": torch.tensor([0.75] + ([0.0] * (hidden_dim - 1)), dtype=torch.float32),
        # D4 v2 redesign: residual_projection 接收 final_hidden + pooled_hidden 拼接 (2*hidden_dim)
        # 测试让 final_hidden[0]=0.75 通过 residual_projection.weight[0,0]=1.0 投射到输出
        "final_hidden": torch.tensor([0.75] + ([0.0] * (hidden_dim - 1)), dtype=torch.float32),
        "pooled_hidden": torch.zeros(hidden_dim, dtype=torch.float32),
        "current_modality": "uwb",
        "current_feature_by_name": {},
        "current_observed_by_name": {},
        "readout_context_by_name": {
            "state_cov_trace": 0.0,
            "pos_cov": 0.0,
            "last_innovation_norm": 0.0,
            "last_gate_skip_flag": 0.0,
        },
        "readout_context_observed_by_name": {
            "state_cov_trace": False,
            "pos_cov": False,
            "last_innovation_norm": False,
            "last_gate_skip_flag": False,
        },
    }

    output = float(model.bias_head(shared).detach().item())
    assert output == pytest.approx(0.75, rel=1e-6)


def test_liquid_output_head_falls_back_to_shared_vector_when_final_and_pooled_hidden_are_missing():
    """共享测试：liquid output head falls back to。\n\n验证 liquid output head falls back to 的共享合同，\n确保不同模型使用一致的输入。
    """
    model = create_model("liquid_ekf", {})
    hidden_dim = model.network.hidden_dim
    with torch.no_grad():
        model.bias_head.temporal_context_gate.weight.zero_()
        model.bias_head.temporal_context_gate.bias.zero_()
        model.bias_head.observation_context_gate.weight.zero_()
        model.bias_head.observation_context_gate.bias.zero_()
        model.bias_head.filter_context_gate.weight.zero_()
        model.bias_head.filter_context_gate.bias.zero_()
        model.bias_head.branch_mix_gate.weight.zero_()
        model.bias_head.branch_mix_gate.bias.fill_(-10.0)
        model.bias_head.uwb_branch_mix_gate.weight.zero_()
        model.bias_head.uwb_branch_mix_gate.bias.fill_(-10.0)
        model.bias_head.vio_branch_mix_gate.weight.zero_()
        model.bias_head.vio_branch_mix_gate.bias.zero_()
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.projection.weight[0, 0] = 1.0
        model.bias_head.residual_projection.weight.zero_()
        model.bias_head.residual_projection.bias.zero_()

    shared = {
        "shared_vector": torch.tensor([0.6] + ([0.0] * (hidden_dim - 1)), dtype=torch.float32),
        "current_modality": "uwb",
        "current_feature_by_name": {},
        "current_observed_by_name": {},
        "readout_context_by_name": {
            "state_cov_trace": 0.0,
            "pos_cov": 0.0,
            "last_innovation_norm": 0.0,
            "last_gate_skip_flag": 0.0,
        },
        "readout_context_observed_by_name": {
            "state_cov_trace": False,
            "pos_cov": False,
            "last_innovation_norm": False,
            "last_gate_skip_flag": False,
        },
    }

    output = float(model.bias_head(shared).detach().item())

    assert output > 0.59
    assert output < 0.61


def test_liquid_output_head_applies_temporal_observation_and_filter_modulation_in_documented_order():
    """应用测试：liquid output head。\n\n验证 liquid output head 的应用逻辑，\n确保特定条件触发预期行为。
    """
    model = create_model("liquid_ekf", {})
    hidden_dim = model.network.hidden_dim
    context_dim = model.bias_head.context_dim
    filter_context_dim = model.bias_head.filter_context_dim
    context_scale = model.bias_head._CONTEXT_MODULATION_SCALE  # A3 根因修复：从实例属性改为类常量。
    filter_scale = model.bias_head._FILTER_CONTEXT_MODULATION_SCALE  # A3 根因修复：从实例属性改为类常量。

    with torch.no_grad():
        model.bias_head.temporal_context_gate.weight.zero_()
        model.bias_head.temporal_context_gate.bias.zero_()
        model.bias_head.temporal_context_gate.bias[0] = 1.0
        model.bias_head.observation_context_gate.weight.zero_()
        model.bias_head.observation_context_gate.bias.zero_()
        model.bias_head.observation_context_gate.bias[0] = 1.0
        model.bias_head.filter_context_gate.weight.zero_()
        model.bias_head.filter_context_gate.bias.zero_()
        model.bias_head.filter_context_gate.bias[0] = 1.0
        model.bias_head.branch_mix_gate.weight.zero_()
        model.bias_head.branch_mix_gate.bias.zero_()
        model.bias_head.uwb_branch_mix_gate.weight.zero_()
        model.bias_head.uwb_branch_mix_gate.bias.zero_()
        model.bias_head.vio_branch_mix_gate.weight.zero_()
        model.bias_head.vio_branch_mix_gate.bias.zero_()
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.projection.weight[0, 0] = 1.0
        model.bias_head.residual_projection.weight.zero_()
        model.bias_head.residual_projection.bias.zero_()

    explicit_context_vector = torch.zeros(context_dim, dtype=torch.float32)
    explicit_filter_context_vector = torch.zeros(filter_context_dim, dtype=torch.float32)

    shared = {
        "shared_vector": torch.zeros(hidden_dim, dtype=torch.float32),
        "final_hidden": torch.tensor([2.0] + ([0.0] * (hidden_dim - 1)), dtype=torch.float32),
        "pooled_hidden": torch.tensor([4.0] + ([0.0] * (hidden_dim - 1)), dtype=torch.float32),
        "current_modality": "uwb",
        "context_vector": explicit_context_vector,
        "filter_context_vector": explicit_filter_context_vector,
        "current_feature_by_name": {},
        "current_observed_by_name": {},
        "readout_context_by_name": {
            "state_cov_trace": 0.0,
            "pos_cov": 0.0,
            "last_innovation_norm": 0.0,
            "last_gate_skip_flag": 0.0,
        },
        "readout_context_observed_by_name": {
            "state_cov_trace": False,
            "pos_cov": False,
            "last_innovation_norm": False,
            "last_gate_skip_flag": False,
        },
    }

    output = float(model.bias_head(shared).detach().item())

    temporal_delta = float(torch.tanh(torch.tensor(1.0)).item())
    observation_delta = float(torch.tanh(torch.tensor(1.0)).item())
    filter_delta = float(torch.tanh(torch.tensor(1.0)).item())
    fast_shared = 2.0 * (1.0 + (context_scale * observation_delta))
    slow_shared = 4.0 * (1.0 + (context_scale * temporal_delta))
    # 修复后：滤波上下文统一在融合后做一次线性调制，不再对 slow_shared 单独调制。
    fused_shared = 0.5 * (fast_shared + slow_shared)
    expected = fused_shared * (1.0 + (filter_scale * filter_delta))

    assert output == pytest.approx(expected, rel=1e-6)


def test_liquid_output_head_masks_cross_modality_filter_context_features():
    model = create_model("liquid_ekf", {})
    hidden_dim = model.network.hidden_dim
    last_innovation_index = 4
    last_gate_skip_index = 6
    with torch.no_grad():
        model.bias_head.temporal_context_gate.weight.zero_()
        model.bias_head.temporal_context_gate.bias.zero_()
        model.bias_head.observation_context_gate.weight.zero_()
        model.bias_head.observation_context_gate.bias.zero_()
        model.bias_head.filter_context_gate.weight.zero_()
        model.bias_head.filter_context_gate.bias.zero_()
        model.bias_head.filter_context_gate.weight[0, last_innovation_index] = 1.0
        model.bias_head.filter_context_gate.weight[1, last_gate_skip_index] = 1.0
        model.bias_head.branch_mix_gate.weight.zero_()
        model.bias_head.branch_mix_gate.bias.zero_()
        model.bias_head.uwb_branch_mix_gate.weight.zero_()
        model.bias_head.uwb_branch_mix_gate.bias.zero_()
        model.bias_head.vio_branch_mix_gate.weight.zero_()
        model.bias_head.vio_branch_mix_gate.bias.zero_()
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.projection.weight[0, 0] = 1.0
        model.bias_head.projection.weight[0, 1] = 1.0
        model.bias_head.residual_projection.weight.zero_()
        model.bias_head.residual_projection.bias.zero_()

    base_shared = {
        "shared_vector": torch.zeros(hidden_dim, dtype=torch.float32),
        "final_hidden": torch.tensor([1.0, 1.0] + ([0.0] * (hidden_dim - 2)), dtype=torch.float32),
        "pooled_hidden": torch.tensor([1.0, 1.0] + ([0.0] * (hidden_dim - 2)), dtype=torch.float32),
        "current_feature_by_name": {"modality_gap_dt": 0.0},
        "current_observed_by_name": {"modality_gap_dt": True},
        "readout_context_by_name": {
            "state_cov_trace": 0.0,
            "pos_cov": 0.0,
            "last_innovation_norm": 0.0,
            "last_gate_skip_flag": 0.0,
        },
        "readout_context_observed_by_name": {
            "state_cov_trace": False,
            "pos_cov": False,
            "last_innovation_norm": False,
            "last_gate_skip_flag": False,
        },
    }

    uwb_base = dict(base_shared)
    uwb_base["current_modality"] = "uwb"
    uwb_with_vio_only_filter_change = dict(uwb_base)
    uwb_with_vio_only_filter_change["readout_context_by_name"] = dict(base_shared["readout_context_by_name"])
    uwb_with_vio_only_filter_change["readout_context_by_name"]["last_innovation_norm"] = 0.8
    uwb_with_vio_only_filter_change["readout_context_observed_by_name"] = dict(base_shared["readout_context_observed_by_name"])
    uwb_with_vio_only_filter_change["readout_context_observed_by_name"]["last_innovation_norm"] = True
    uwb_with_uwb_filter_change = dict(uwb_base)
    uwb_with_uwb_filter_change["readout_context_by_name"] = dict(base_shared["readout_context_by_name"])
    uwb_with_uwb_filter_change["readout_context_by_name"]["last_gate_skip_flag"] = 1.0
    uwb_with_uwb_filter_change["readout_context_observed_by_name"] = dict(base_shared["readout_context_observed_by_name"])
    uwb_with_uwb_filter_change["readout_context_observed_by_name"]["last_gate_skip_flag"] = True

    vio_base = dict(base_shared)
    vio_base["current_modality"] = "vio"
    vio_with_uwb_only_filter_change = dict(vio_base)
    vio_with_uwb_only_filter_change["readout_context_by_name"] = dict(base_shared["readout_context_by_name"])
    vio_with_uwb_only_filter_change["readout_context_by_name"]["last_gate_skip_flag"] = 1.0
    vio_with_uwb_only_filter_change["readout_context_observed_by_name"] = dict(base_shared["readout_context_observed_by_name"])
    vio_with_uwb_only_filter_change["readout_context_observed_by_name"]["last_gate_skip_flag"] = True
    vio_with_vio_filter_change = dict(vio_base)
    vio_with_vio_filter_change["readout_context_by_name"] = dict(base_shared["readout_context_by_name"])
    vio_with_vio_filter_change["readout_context_by_name"]["last_innovation_norm"] = 0.8
    vio_with_vio_filter_change["readout_context_observed_by_name"] = dict(base_shared["readout_context_observed_by_name"])
    vio_with_vio_filter_change["readout_context_observed_by_name"]["last_innovation_norm"] = True

    uwb_base_output = float(model.bias_head(uwb_base).detach().item())
    uwb_vio_only_output = float(model.bias_head(uwb_with_vio_only_filter_change).detach().item())
    uwb_own_filter_output = float(model.bias_head(uwb_with_uwb_filter_change).detach().item())
    vio_base_output = float(model.bias_head(vio_base).detach().item())
    vio_uwb_only_output = float(model.bias_head(vio_with_uwb_only_filter_change).detach().item())
    vio_own_filter_output = float(model.bias_head(vio_with_vio_filter_change).detach().item())

    assert uwb_vio_only_output == pytest.approx(uwb_base_output, abs=1e-6)
    assert uwb_own_filter_output > uwb_base_output + 1e-2
    assert vio_uwb_only_output == pytest.approx(vio_base_output, abs=1e-6)
    assert vio_own_filter_output > vio_base_output + 1e-2


def test_liquid_output_head_prefers_explicit_context_tensors_over_fallback_metadata():
    """回退测试：liquid output head prefers explicit context tensors over。\n\n验证 liquid output head prefers explicit context tensors over 的回退机制，\n确保主路径失败时有合理的降级策略。
    """
    model = create_model("liquid_ekf", {})
    hidden_dim = model.network.hidden_dim
    context_dim = model.bias_head.context_dim
    filter_context_dim = model.bias_head.filter_context_dim
    with torch.no_grad():
        model.bias_head.temporal_context_gate.weight.zero_()
        model.bias_head.temporal_context_gate.bias.zero_()
        model.bias_head.observation_context_gate.weight.zero_()
        model.bias_head.observation_context_gate.bias.zero_()
        model.bias_head.filter_context_gate.weight.zero_()
        model.bias_head.filter_context_gate.bias.zero_()
        model.bias_head.branch_mix_gate.weight.zero_()
        model.bias_head.branch_mix_gate.bias.zero_()
        model.bias_head.uwb_branch_mix_gate.weight.zero_()
        model.bias_head.uwb_branch_mix_gate.bias.zero_()
        model.bias_head.vio_branch_mix_gate.weight.zero_()
        model.bias_head.vio_branch_mix_gate.bias.zero_()
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.projection.weight[0, hidden_dim + 0] = 1.0
        model.bias_head.projection.weight[0, hidden_dim + context_dim + 0] = 1.0
        model.bias_head.residual_projection.weight.zero_()
        model.bias_head.residual_projection.bias.zero_()

    explicit_context_vector = torch.zeros(context_dim, dtype=torch.float32)
    explicit_filter_context_vector = torch.zeros(filter_context_dim, dtype=torch.float32)
    explicit_context_vector[0] = 0.4
    explicit_filter_context_vector[0] = 0.7

    shared = {
        "shared_vector": torch.zeros(hidden_dim, dtype=torch.float32),
        "final_hidden": torch.zeros(hidden_dim, dtype=torch.float32),
        "pooled_hidden": torch.zeros(hidden_dim, dtype=torch.float32),
        "current_modality": "uwb",
        "current_feature_by_name": {
            "quality": 0.9,
            "modality_gap_dt": 0.25,
        },
        "current_observed_by_name": {
            "quality": True,
            "modality_gap_dt": True,
        },
        "readout_context_by_name": {
            "state_cov_trace": 2.5,
            "pos_cov": 1.0,
            "last_innovation_norm": 0.2,
            "last_gate_skip_flag": 1.0,
        },
        "readout_context_observed_by_name": {
            "state_cov_trace": True,
            "pos_cov": True,
            "last_innovation_norm": True,
            "last_gate_skip_flag": True,
        },
        "context_vector": explicit_context_vector,
        "filter_context_vector": explicit_filter_context_vector,
    }

    output = float(model.bias_head(shared).detach().item())

    assert output == pytest.approx(0.7, abs=1e-6)  # 只有 filter_context_vector[0]=0.7 贡献；context_vector[0] 对应模态 one-hot，UWB 分支掩码不包含模态索引，被置零。


def test_liquid_output_head_keeps_full_filter_context_for_unknown_modality():
    """保持测试：liquid output head。\n\n验证 liquid output head 的保持行为，\n确保特定属性在处理过程中不变。
    """
    model = create_model("liquid_ekf", {})
    hidden_dim = model.network.hidden_dim
    with torch.no_grad():
        model.bias_head.temporal_context_gate.weight.zero_()
        model.bias_head.temporal_context_gate.bias.zero_()
        model.bias_head.observation_context_gate.weight.zero_()
        model.bias_head.observation_context_gate.bias.zero_()
        model.bias_head.filter_context_gate.weight.zero_()
        model.bias_head.filter_context_gate.bias.zero_()
        model.bias_head.filter_context_gate.weight[0, 4] = 1.0
        model.bias_head.filter_context_gate.weight[1, 6] = 1.0
        model.bias_head.branch_mix_gate.weight.zero_()
        model.bias_head.branch_mix_gate.bias.zero_()
        model.bias_head.uwb_branch_mix_gate.weight.zero_()
        model.bias_head.uwb_branch_mix_gate.bias.fill_(-10.0)
        model.bias_head.vio_branch_mix_gate.weight.zero_()
        model.bias_head.vio_branch_mix_gate.bias.fill_(-10.0)
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.projection.weight[0, 0] = 1.0
        model.bias_head.projection.weight[0, 1] = 1.0
        model.bias_head.residual_projection.weight.zero_()
        model.bias_head.residual_projection.bias.zero_()

    shared = {
        "shared_vector": torch.zeros(hidden_dim, dtype=torch.float32),
        "final_hidden": torch.tensor([1.0, 1.0] + ([0.0] * (hidden_dim - 2)), dtype=torch.float32),
        "pooled_hidden": torch.tensor([1.0, 1.0] + ([0.0] * (hidden_dim - 2)), dtype=torch.float32),
        "current_modality": "uwb",
        "current_feature_by_name": {"modality_gap_dt": 0.0},
        "current_observed_by_name": {"modality_gap_dt": True},
        "readout_context_by_name": {
            "state_cov_trace": 0.0,
            "pos_cov": 0.0,
            "last_innovation_norm": 0.8,
            "last_gate_skip_flag": 1.0,
        },
        "readout_context_observed_by_name": {
            "state_cov_trace": False,
            "pos_cov": False,
            "last_innovation_norm": True,
            "last_gate_skip_flag": True,
        },
    }

    output = float(model.bias_head(shared).detach().item())

    assert output > 2.0


def test_liquid_output_head_distinguishes_zero_observed_from_missing_observation_context():
    """缺失测试：liquid output head distinguishes zero observed from。\n\n验证 liquid output head distinguishes zero observed from 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    model = create_model("liquid_ekf", {})
    hidden_dim = model.network.hidden_dim
    # 铁律 9：探针改接 valid 观测标志位（layout: modal 0,1 | valid value 2, flag 3）。
    valid_flag_index = 2 + (2 * _LIQUID_CONTEXT_FEATURE_KEYS.index("valid")) + 1
    with torch.no_grad():
        model.bias_head.temporal_context_gate.weight.zero_()
        model.bias_head.temporal_context_gate.bias.zero_()
        model.bias_head.observation_context_gate.weight.zero_()
        model.bias_head.observation_context_gate.bias.zero_()
        model.bias_head.filter_context_gate.weight.zero_()
        model.bias_head.filter_context_gate.bias.zero_()
        model.bias_head.branch_mix_gate.weight.zero_()
        model.bias_head.branch_mix_gate.bias.zero_()
        model.bias_head.uwb_branch_mix_gate.weight.zero_()
        model.bias_head.uwb_branch_mix_gate.bias.zero_()
        model.bias_head.vio_branch_mix_gate.weight.zero_()
        model.bias_head.vio_branch_mix_gate.bias.zero_()
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.projection.weight[0, hidden_dim + valid_flag_index] = 1.0
        model.bias_head.residual_projection.weight.zero_()
        model.bias_head.residual_projection.bias.zero_()

    base_shared = {
        "shared_vector": torch.zeros(hidden_dim, dtype=torch.float32),
        "final_hidden": torch.zeros(hidden_dim, dtype=torch.float32),
        "pooled_hidden": torch.zeros(hidden_dim, dtype=torch.float32),
        "current_modality": "uwb",
        "current_feature_by_name": {
            "valid": 0.0,
            "uwb_range_residual": 0.0,
            "anchor_dx": 0.0,
            "anchor_dy": 0.0,
            "geom_score": 0.0,
            "modality_gap_dt": 0.0,
        },
        "readout_context_by_name": {
            "state_cov_trace": 0.0,
            "pos_cov": 0.0,
            "last_innovation_norm": 0.0,
            "last_gate_skip_flag": 0.0,
        },
        "readout_context_observed_by_name": {
            "state_cov_trace": False,
            "pos_cov": False,
            "last_innovation_norm": False,
            "last_gate_skip_flag": False,
        },
    }

    missing_shared = dict(base_shared)
    missing_shared["current_observed_by_name"] = {
        "valid": False,
        "uwb_range_residual": False,
        "anchor_dx": False,
        "anchor_dy": False,
        "geom_score": False,
        "modality_gap_dt": False,
    }
    observed_zero_shared = dict(base_shared)
    observed_zero_shared["current_observed_by_name"] = {
        "valid": True,
        "uwb_range_residual": False,
        "anchor_dx": False,
        "anchor_dy": False,
        "geom_score": False,
        "modality_gap_dt": False,
    }

    missing_output = float(model.bias_head(missing_shared).detach().item())
    observed_zero_output = float(model.bias_head(observed_zero_shared).detach().item())

    assert missing_output == pytest.approx(0.0, abs=1e-6)
    assert observed_zero_output == pytest.approx(1.0, abs=1e-6)


def test_liquid_output_head_distinguishes_zero_observed_from_missing_filter_context():
    """缺失测试：liquid output head distinguishes zero observed from。\n\n验证 liquid output head distinguishes zero observed from 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    model = create_model("liquid_ekf", {})
    hidden_dim = model.network.hidden_dim
    state_cov_trace_flag_index = 1
    with torch.no_grad():
        model.bias_head.temporal_context_gate.weight.zero_()
        model.bias_head.temporal_context_gate.bias.zero_()
        model.bias_head.observation_context_gate.weight.zero_()
        model.bias_head.observation_context_gate.bias.zero_()
        model.bias_head.filter_context_gate.weight.zero_()
        model.bias_head.filter_context_gate.bias.zero_()
        model.bias_head.branch_mix_gate.weight.zero_()
        model.bias_head.branch_mix_gate.bias.zero_()
        model.bias_head.uwb_branch_mix_gate.weight.zero_()
        model.bias_head.uwb_branch_mix_gate.bias.zero_()
        model.bias_head.vio_branch_mix_gate.weight.zero_()
        model.bias_head.vio_branch_mix_gate.bias.zero_()
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.projection.weight[0, hidden_dim + model.bias_head.context_dim + state_cov_trace_flag_index] = 1.0
        model.bias_head.residual_projection.weight.zero_()
        model.bias_head.residual_projection.bias.zero_()

    base_shared = {
        "shared_vector": torch.zeros(hidden_dim, dtype=torch.float32),
        "final_hidden": torch.zeros(hidden_dim, dtype=torch.float32),
        "pooled_hidden": torch.zeros(hidden_dim, dtype=torch.float32),
        "current_modality": "uwb",
        "current_feature_by_name": {},
        "current_observed_by_name": {},
        "readout_context_by_name": {
            "state_cov_trace": 0.0,
            "pos_cov": 0.0,
            "last_innovation_norm": 0.0,
            "last_gate_skip_flag": 0.0,
        },
    }

    missing_shared = dict(base_shared)
    missing_shared["readout_context_observed_by_name"] = {
        "state_cov_trace": False,
        "pos_cov": False,
        "last_innovation_norm": False,
        "last_gate_skip_flag": False,
    }
    observed_zero_shared = dict(base_shared)
    observed_zero_shared["readout_context_observed_by_name"] = {
        "state_cov_trace": True,
        "pos_cov": False,
        "last_innovation_norm": False,
        "last_gate_skip_flag": False,
    }

    missing_output = float(model.bias_head(missing_shared).detach().item())
    observed_zero_output = float(model.bias_head(observed_zero_shared).detach().item())

    assert missing_output == pytest.approx(0.0, abs=1e-6)
    assert observed_zero_output == pytest.approx(1.0, abs=1e-6)
