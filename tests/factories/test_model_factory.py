"""模型工厂（model_factory）测试模块。

测试覆盖范围：
- LSTM/Liquid 模型的创建
- 检查点加载与设备兼容性
- 未知模型类型的拒绝
- 配置参数的传递与验证

被测模块：liquidloc.factories.model_factory"""

import math
from pathlib import Path
from uuid import uuid4

import pytest
import torch
import torch.nn.functional as F
import yaml

import liquidloc.factories.model_factory as model_factory_module
from liquidloc.factories.model_factory import create_model


def _liquid_checkpoint_payload():
    model_cfg = {
        "feature_order": ["quality", "valid", "modality_gap_dt", "uwb_quality_min"],
        "window": {"size": 1, "step": 1},
        "network": {"input_dim": 4, "hidden_dim": 4},
    }
    model = create_model("liquid_ekf", dict(model_cfg))
    with torch.no_grad():
        model.network.shared_projection.weight.fill_(0.25)
    return model_cfg, {
        "checkpoint_format": "liquid_real_v1",
        "model_cfg": model_cfg,
        "model_state": model.state_dict(),
    }


def _liquid_checkpoint_payload_with_flat_model_cfg():
    checkpoint_model_cfg = {
        "feature_order": ["quality", "valid", "modality_gap_dt", "uwb_quality_min"],
        "window": {"size": 3, "step": 1},
        "hidden_dim": 8,
        "input_dim": 4,
        "network": {"output_heads": ["bias", "risk", "uwb_scaling", "vio_scaling"]},
    }
    model = create_model(
        "liquid_ekf",
        {
            "feature_order": checkpoint_model_cfg["feature_order"],
            "window": checkpoint_model_cfg["window"],
            "network": {
                "hidden_dim": checkpoint_model_cfg["hidden_dim"],
                "input_dim": checkpoint_model_cfg["input_dim"],
                "output_heads": checkpoint_model_cfg["network"]["output_heads"],
            },
        },
    )
    return checkpoint_model_cfg, {
        "checkpoint_format": "liquid_real_v1",
        "model_cfg": checkpoint_model_cfg,
        "model_state": model.state_dict(),
    }


def _lstm_checkpoint_payload():
    model_cfg = {
        "feature_order": ["dt", "ax", "modality_gap_dt", "vio_reproj_err_slope"],
        "window": {"size": 3, "step": 1},
        "network": {
            "input_dim": 4,
            "hidden_dim": 4,
            "output_heads": ["bias", "risk", "uwb_scaling", "vio_scaling"],
        },
    }
    model = create_model("lstm_ekf", dict(model_cfg))
    with torch.no_grad():
        model.network.input_projection.weight.fill_(0.125)
    return model_cfg, {
        "checkpoint_format": "lstm_real_v1",
        "model_cfg": model_cfg,
        "model_state": model.state_dict(),
    }


def _lstm_checkpoint_payload_with_flat_model_cfg():
    checkpoint_model_cfg = {
        "feature_order": ["dt", "ax", "modality_gap_dt", "vio_reproj_err_slope"],
        "window": {"size": 5, "step": 1},
        "hidden_dim": 6,
        "input_dim": 4,
        "network": {"output_heads": ["bias", "risk", "uwb_scaling", "vio_scaling"]},
    }
    model = create_model(
        "lstm_ekf",
        {
            "feature_order": checkpoint_model_cfg["feature_order"],
            "window": checkpoint_model_cfg["window"],
            "network": {
                "hidden_dim": checkpoint_model_cfg["hidden_dim"],
                "input_dim": checkpoint_model_cfg["input_dim"],
                "output_heads": checkpoint_model_cfg["network"]["output_heads"],
            },
        },
    )
    return checkpoint_model_cfg, {
        "checkpoint_format": "lstm_real_v1",
        "model_cfg": checkpoint_model_cfg,
        "model_state": model.state_dict(),
    }


def _lstm_structured_window(feature_order=None):
    resolved_feature_order = list(feature_order or ["dt", "ax", "modality_gap_dt", "vio_reproj_err_slope"])
    return {
        "current_modality": "uwb",
        "feature_order": resolved_feature_order,
        "feature_values": [0.0, 0.0, 0.0, 0.0],
        "missing_mask": [0, 0, 0, 0],
        "dt": 0.1,
        "feature_window": [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "missing_mask_window": [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    }


def test_normal_case():
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    model = create_model('liquid_ekf', {})
    outputs = model.infer_intermediate(
        {
            'current_modality': 'uwb',
            'feature_order': ['quality', 'valid', 'modality_gap_dt', 'uwb_quality_min'],
            'feature_values': [0.2, 0.0, 0.5, 0.2],
            'missing_mask': [0, 1, 0, 0],
            'dt': 0.1,
            'feature_window': [[0.2, 0.0, 0.5, 0.2]],
            'missing_mask_window': [[0, 1, 0, 0]],
        }
    )
    from liquidloc.common.constants import BRIDGE_SCALING_MIN
    assert all(hasattr(outputs, key) for key in ('bias', 'risk', 'uwb_scaling', 'vio_scaling'))
    assert torch.isfinite(torch.tensor(outputs.bias)).item()
    assert 0.0 <= outputs.risk <= 1.0
    # v2: scaling 下界放宽至 BRIDGE_SCALING_MIN（0.5），配合 soft-mask 释放 Liquid 调节空间
    assert outputs.uwb_scaling >= BRIDGE_SCALING_MIN
    assert outputs.vio_scaling >= BRIDGE_SCALING_MIN


def test_boundary_case():
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    from liquidloc.common.constants import BRIDGE_SCALING_MIN
    model = create_model('liquid_ekf', {})
    outputs = model.infer_intermediate(
        {
            'current_modality': 'vio',
            'feature_order': ['tracked_features', 'reproj_err', 'vio_reproj_err_slope', 'tracked_features_drop'],
            'feature_values': [0.0, 1.4, 0.2, 3.0],
            'missing_mask': [1, 0, 0, 0],
            'dt': 0.1,
            'feature_window': [[0.0, 1.4, 0.2, 3.0]],
            'missing_mask_window': [[1, 0, 0, 0]],
        }
    )
    # v2: scaling 下界放宽至 BRIDGE_SCALING_MIN（0.5），配合 soft-mask 释放 Liquid 调节空间
    assert outputs.vio_scaling >= BRIDGE_SCALING_MIN
    assert 0.0 <= outputs.risk <= 1.0


def test_liquid_factory_preserves_configured_reliability_profile():
    """保持性测试：liquid factory。\n\n验证 liquid factory 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    model = create_model(
        "liquid_ekf",
        {
            "feature_order": ["quality", "valid", "modality_gap_dt"],
            "window": {"size": 1, "step": 1},
            "network": {
                "input_dim": 3,
                "hidden_dim": 4,
                "pooling_logit_scale": 0.4,
                "cell_update_scale_floor": 0.52,
                "cell_update_scale_span": 0.33,
                "reliability_bias_init": 1.3,
                "bad_observation_floor": 0.18,
                "bad_observation_span": 0.82,
                "bad_observation_interaction_coeff": 0.9,
            },
        },
    )

    assert model.network.cell is not None
    assert model.network.pooling_logit_scale == pytest.approx(0.4)
    assert model.network.cell.update_scale_floor == pytest.approx(0.52)
    assert model.network.cell.update_scale_span == pytest.approx(0.33)
    assert model.network.cell.reliability_projection.bias.detach().cpu().item() == pytest.approx(1.3)
    assert model.network.cell.bad_observation_floor == pytest.approx(0.18)
    assert model.network.cell.bad_observation_span == pytest.approx(0.82)
    assert model.network.cell.bad_observation_interaction_coeff == pytest.approx(0.9)


def test_liquid_runtime_resource_meta_matches_actual_module_parameter_count():
    """匹配测试：liquid runtime resource meta。\n\n验证 liquid runtime resource meta 的输出与预期一致，\n确保合同合规。
    """
    model = create_model(
        "liquid_ekf",
        {
            "feature_order": ["quality", "valid", "modality_gap_dt", "uwb_quality_min"],
            "window": {"size": 1, "step": 1},
            "network": {
                "input_dim": 4,
                "hidden_dim": 4,
                "output_heads": ["bias", "risk", "uwb_scaling", "vio_scaling"],
            },
        },
    )

    actual_params = 0
    for module in (
        model.network,
        model.output_backbone,  # v3.1 head-shared: 共享 backbone
        model.bias_head,
        model.risk_head,
        model.uwb_scaling_head,
        model.vio_scaling_head,
        model.risk_calibration,
    ):
        actual_params += sum(int(parameter.numel()) for parameter in module.parameters())

    assert model.params == pytest.approx(float(actual_params))


def test_liquid_output_normalization_keeps_scaling_strictly_positive():
    """保持测试：liquid output normalization。\n\n验证 liquid output normalization 的保持行为，\n确保特定属性在处理过程中不变。
    """
    model = create_model("liquid_ekf", {})
    outputs = model.predict_intermediate_tensors(
        {
            "current_modality": "uwb",
            "feature_order": ["quality", "valid"],
            "feature_values": [0.2, 1.0],
            "missing_mask": [0, 0],
            "dt": 0.1,
            "feature_window": [[0.2, 1.0]],
            "missing_mask_window": [[0, 0]],
        }
    )

    # D5-1 v2 redesign: scaling_min 从 1.0 放宽到 0.5（BRIDGE_THRESHOLDS["scaling_min"]）
    assert outputs["uwb_scaling"].item() >= 0.5
    assert outputs["vio_scaling"].item() >= 0.5


def test_liquid_scaling_is_clamped_to_scaling_max():
    """_neutral_floor_scaling_softplus 的输出不超过 scaling_max (50.0)。"""
    from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
    scaling_max = float(BRIDGE_THRESHOLDS["scaling_max"])
    model = create_model("liquid_ekf", {})
    with torch.no_grad():
        model.uwb_scaling_head.projection.weight.zero_()
        model.uwb_scaling_head.projection.bias.fill_(100.0)  # 极大值触发上限裁剪
        model.vio_scaling_head.projection.weight.zero_()
        model.vio_scaling_head.projection.bias.fill_(100.0)

    outputs = model.predict_intermediate_tensors(
        {
            "current_modality": "uwb",
            "feature_order": ["quality", "valid"],
            "feature_values": [0.2, 1.0],
            "missing_mask": [0, 0],
            "dt": 0.1,
            "feature_window": [[0.2, 1.0]],
            "missing_mask_window": [[0, 0]],
        }
    )

    assert outputs["uwb_scaling"].item() <= scaling_max
    # vio_scaling 被 uwb 模态合同强制为 1.0，不测试上限


def test_liquid_output_normalization_matches_training_modality_contract_for_vio():
    """匹配测试：liquid output normalization。\n\n验证 liquid output normalization 的输出与预期一致，\n确保合同合规。
    """
    model = create_model("liquid_ekf", {})
    with torch.no_grad():
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()
        model.uwb_scaling_head.projection.weight.zero_()
        model.uwb_scaling_head.projection.bias.fill_(2.0)
        model.vio_scaling_head.projection.weight.zero_()
        model.vio_scaling_head.projection.bias.zero_()

    outputs = model.predict_intermediate_tensors(
        {
            "current_modality": "vio",
            "feature_order": ["quality", "valid"],
            "feature_values": [0.2, 1.0],
            "missing_mask": [0, 0],
            "dt": 0.1,
            "feature_window": [[0.2, 1.0]],
            "missing_mask_window": [[0, 0]],
        }
    )

    # v3 改造: 非当前模态 scaling 被 soft-mask clamp 到 [scaling_min=0.5, scaling_ceiling=2.5],
    # 不再硬性 == 1.0; uwb_scaling_head bias=2.0 经 sigmoid+normalize 落在 [0.5, 2.5] 区间.
    assert outputs["uwb_scaling"].item() <= 2.5
    assert outputs["uwb_scaling"].item() >= 0.5
    assert outputs["risk"].item() == pytest.approx(float(torch.sigmoid(torch.tensor(0.0))), rel=1e-6)
    # D5-1: scaling_min=0.5
    assert outputs["vio_scaling"].item() >= 0.5


def test_liquid_factory_state_dict_includes_risk_calibration():
    model = create_model("liquid_ekf", {})
    state = model.state_dict()

    assert "risk_calibration" in state
    assert "a_raw" in state["risk_calibration"]
    assert "b" in state["risk_calibration"]


def test_liquid_factory_exposes_current_documented_readout_modules():
    """读出上下文测试：liquid factory exposes current documented。\n\n验证 liquid factory exposes current documented 的读出上下文构建，\n确保协方差摘要和门控标志正确传递。
    """
    model = create_model("liquid_ekf", {})

    assert hasattr(model, "risk_calibration")
    assert hasattr(model, "output_backbone")
    backbone = model.output_backbone
    for gate in ("filter_context_gate", "uwb_branch_mix_gate", "vio_branch_mix_gate"):
        assert hasattr(backbone, gate)
    for head in model.output_heads.values():
        assert hasattr(head, "projection")
        assert hasattr(head, "residual_projection")


def test_liquid_factory_exposes_only_documented_four_intermediate_heads():
    model = create_model("liquid_ekf", {})
    hidden_dim = model.network.hidden_dim

    backbone = model.output_backbone
    with torch.no_grad():
        for head in model.output_heads.values():
            backbone.temporal_context_gate.weight.zero_()
            backbone.temporal_context_gate.bias.zero_()
            backbone.observation_context_gate.weight.zero_()
            backbone.observation_context_gate.bias.zero_()
            backbone.filter_context_gate.weight.zero_()
            backbone.filter_context_gate.bias.zero_()
            backbone.branch_mix_gate.weight.zero_()
            backbone.branch_mix_gate.bias.zero_()
            backbone.uwb_branch_mix_gate.weight.zero_()
            backbone.uwb_branch_mix_gate.bias.zero_()
            backbone.vio_branch_mix_gate.weight.zero_()
            backbone.vio_branch_mix_gate.bias.zero_()
            head.projection.weight.zero_()
            head.projection.bias.zero_()
            head.residual_projection.weight.zero_()
            head.residual_projection.bias.zero_()

    outputs = model.predict_intermediate_tensors(
        {
            "current_modality": "uwb",
            "feature_order": ["quality", "valid"],
            "feature_values": [0.2, 1.0],
            "missing_mask": [0, 0],
            "dt": 0.1,
            "feature_window": [[0.2, 1.0]],
            "missing_mask_window": [[0, 0]],
            "readout_context_by_name": {
                "state_cov_trace": 1.0,
                "pos_cov": 0.5,
                "last_innovation_norm": 0.25,
                "last_gate_skip_flag": 0.0,
            },
            "readout_context_observed_by_name": {
                "state_cov_trace": True,
                "pos_cov": True,
                "last_innovation_norm": True,
                "last_gate_skip_flag": True,
            },
        }
    )

    assert tuple(model.output_heads.keys()) == ("bias", "risk", "uwb_scaling", "vio_scaling")
    assert set(outputs.keys()) == {"bias", "risk", "uwb_scaling", "vio_scaling"}
    assert not ({"px", "py", "vx", "vy", "yaw", "bax", "bay", "bg"} & set(outputs.keys()))
    assert all(torch.is_tensor(outputs[key]) and outputs[key].reshape(()).numel() == 1 for key in outputs)
    assert model.network.shared_projection.weight.shape == (hidden_dim, hidden_dim * 2)


def test_liquid_factory_exposes_documented_context_dimensions_and_masks():
    model = create_model("liquid_ekf", {})

    # 铁律 9：LIQUID_CONTEXT_FEATURE_KEYS 6 键 → context_dim = 2 + 2*6 = 14；
    # filter context 仍 = 2 * len(LIQUID_READOUT_CONTEXT_KEYS)（读出键集合未变）。
    expected_context_dim = model_factory_module.LIQUID_CONTEXT_DIM
    assert expected_context_dim == 14
    expected_filter_context_dim = model_factory_module.LIQUID_FILTER_CONTEXT_DIM
    backbone = model.output_backbone
    assert backbone.context_dim == expected_context_dim
    assert backbone.filter_context_dim == expected_filter_context_dim
    assert tuple(backbone.uwb_fast_context_mask.shape) == (expected_context_dim,)
    assert tuple(backbone.vio_fast_context_mask.shape) == (expected_context_dim,)
    assert tuple(backbone.uwb_branch_context_mask.shape) == (expected_context_dim,)
    assert tuple(backbone.vio_branch_context_mask.shape) == (expected_context_dim,)
    assert tuple(backbone.uwb_filter_context_mask.shape) == (expected_filter_context_dim,)
    assert tuple(backbone.vio_filter_context_mask.shape) == (expected_filter_context_dim,)


def test_liquid_factory_uses_configured_context_modulation_scales():
    config_root = Path(__file__).resolve().parents[2] / "configs" / "models"
    liquid_cfg = yaml.safe_load((config_root / "liquid_ekf.yaml").read_text(encoding="utf-8"))
    model = create_model("liquid_ekf", liquid_cfg)
    backbone = model.output_backbone
    assert backbone.context_modulation_scale == pytest.approx(0.22)
    assert backbone.filter_context_modulation_scale == pytest.approx(0.22)


def test_liquid_output_head_reset_parameters_matches_documented_contract():
    """匹配测试：liquid output head reset parameters。\n\n验证 liquid output head reset parameters 的输出与预期一致，\n确保合同合规。
    """
    model = create_model("liquid_ekf", {})
    # 风险先验偏置 = log(RISK_PRIOR_PROB / (1 - RISK_PRIOR_PROB))；当前 RISK_PRIOR_PROB=0.5
    # ⇒ logit=0.0（与 src/liquidloc/common/constants.py:169 同步）。
    risk_bias = float(torch.tensor(0.50 / 0.50).log().item())
    backbone = model.output_backbone
    assert torch.count_nonzero(backbone.temporal_context_gate.weight).item() == 0
    assert torch.count_nonzero(backbone.temporal_context_gate.bias).item() == 0
    assert torch.count_nonzero(backbone.observation_context_gate.weight).item() == 0
    assert torch.count_nonzero(backbone.observation_context_gate.bias).item() == 0
    assert torch.count_nonzero(backbone.filter_context_gate.weight).item() == 0
    assert torch.count_nonzero(backbone.filter_context_gate.bias).item() == 0
    assert torch.count_nonzero(backbone.branch_mix_gate.weight).item() == 0
    assert torch.count_nonzero(backbone.branch_mix_gate.bias).item() == 0
    assert torch.count_nonzero(backbone.uwb_branch_mix_gate.weight).item() == 0
    assert torch.count_nonzero(backbone.uwb_branch_mix_gate.bias).item() == 0
    assert torch.count_nonzero(backbone.vio_branch_mix_gate.weight).item() == 0
    assert torch.count_nonzero(backbone.vio_branch_mix_gate.bias).item() == 0
    for key, head in model.output_heads.items():
        assert torch.count_nonzero(head.projection.weight).item() == 0
        assert torch.count_nonzero(head.residual_projection.weight).item() == 0
        assert torch.count_nonzero(head.residual_projection.bias).item() == 0
        if key == "risk":
            assert float(head.projection.bias.detach().item()) == pytest.approx(risk_bias)
        else:
            assert torch.count_nonzero(head.projection.bias).item() == 0


def test_liquid_risk_calibration_initialization_is_identity_sigmoid():
    """同一性测试：liquid risk calibration initialization is。\n\n验证 liquid risk calibration initialization is 的同一性约束，\n确保不同输入产生不同输出。
    """
    model = create_model("liquid_ekf", {})
    raw_risk = torch.tensor(0.7, dtype=torch.float32)
    calibrated = model.risk_calibration(raw_risk)

    assert float(F.softplus(model.risk_calibration.a_raw).detach().item()) == pytest.approx(1.0, rel=1e-6)
    assert float(model.risk_calibration.b.detach().item()) == pytest.approx(0.0, abs=1e-8)
    assert float(calibrated.detach().item()) == pytest.approx(float(torch.sigmoid(raw_risk).item()), abs=1e-6)


def test_cfg_only_liquid_preserves_optional_readout_context_after_feature_order_validation():
    """保持性测试：cfg only liquid。\n\n验证 cfg only liquid 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    model = create_model(
        "liquid_ekf",
        {
            "feature_order": ["quality", "valid", "modality_gap_dt"],
            "window": {"size": 1, "step": 1},
            "network": {"input_dim": 3, "hidden_dim": 4},
        },
    )

    captured_window: dict[str, object] = {}
    original_extract = model.network.extract_shared_features

    def _capture(window_tensor):
        captured_window.clear()
        if isinstance(window_tensor, dict):
            captured_window.update(window_tensor)
        return original_extract(window_tensor)

    model.network.extract_shared_features = _capture
    try:
        outputs = model.predict_intermediate_tensors(
            {
                "current_modality": "uwb",
                "feature_order": ["quality", "valid", "modality_gap_dt"],
                "feature_values": [0.3, 1.0, 0.2],
                "missing_mask": [0, 0, 0],
                "dt": 0.1,
                "feature_window": [[0.3, 1.0, 0.2]],
                "missing_mask_window": [[0, 0, 0]],
                "readout_context_by_name": {
                    "state_cov_trace": 4.0,
                    "pos_cov": 1.5,
                    "last_innovation_norm": 0.25,
                    "last_gate_skip_flag": 1.0,
                },
                "readout_context_observed_by_name": {
                    "state_cov_trace": True,
                    "pos_cov": True,
                    "last_innovation_norm": True,
                    "last_gate_skip_flag": True,
                },
            }
        )
    finally:
        model.network.extract_shared_features = original_extract

    assert torch.isfinite(outputs["risk"]).item()
    assert captured_window["readout_context_by_name"]["state_cov_trace"] == pytest.approx(4.0)
    assert captured_window["readout_context_by_name"]["pos_cov"] == pytest.approx(1.5)
    assert captured_window["readout_context_by_name"]["last_innovation_norm"] == pytest.approx(0.25)
    assert captured_window["readout_context_by_name"]["last_gate_skip_flag"] == pytest.approx(1.0)
    assert captured_window["readout_context_observed_by_name"]["state_cov_trace"] is True
    assert captured_window["readout_context_observed_by_name"]["pos_cov"] is True
    assert captured_window["readout_context_observed_by_name"]["last_innovation_norm"] is True
    assert captured_window["readout_context_observed_by_name"]["last_gate_skip_flag"] is True


def test_cfg_only_liquid_zeroes_unobserved_readout_context_after_feature_order_validation():
    """零值测试：cfg only liquid。\n\n验证 cfg only liquid 在零值输入下的行为，\n确保边界情况正确处理。
    """
    model = create_model(
        "liquid_ekf",
        {
            "feature_order": ["quality", "valid", "modality_gap_dt"],
            "window": {"size": 1, "step": 1},
            "network": {"input_dim": 3, "hidden_dim": 4},
        },
    )

    captured_window: dict[str, object] = {}
    original_extract = model.network.extract_shared_features

    def _capture(window_tensor):
        captured_window.clear()
        if isinstance(window_tensor, dict):
            captured_window.update(window_tensor)
        return original_extract(window_tensor)

    model.network.extract_shared_features = _capture
    try:
        outputs = model.predict_intermediate_tensors(
            {
                "current_modality": "uwb",
                "feature_order": ["quality", "valid", "modality_gap_dt"],
                "feature_values": [0.3, 1.0, 0.2],
                "missing_mask": [0, 0, 0],
                "dt": 0.1,
                "feature_window": [[0.3, 1.0, 0.2]],
                "missing_mask_window": [[0, 0, 0]],
                "readout_context_by_name": {
                    "state_cov_trace": 4.0,
                    "pos_cov": 1.5,
                },
                "readout_context_observed_by_name": {
                    "state_cov_trace": False,
                    "pos_cov": True,
                },
            }
        )
    finally:
        model.network.extract_shared_features = original_extract

    assert torch.isfinite(outputs["risk"]).item()
    assert captured_window["readout_context_by_name"]["state_cov_trace"] == pytest.approx(0.0)
    assert captured_window["readout_context_observed_by_name"]["state_cov_trace"] is False
    assert captured_window["readout_context_by_name"]["pos_cov"] == pytest.approx(1.5)
    assert captured_window["readout_context_observed_by_name"]["pos_cov"] is True


def test_cfg_only_liquid_preserves_explicit_context_tensors_after_feature_order_validation():
    """保持性测试：cfg only liquid。\n\n验证 cfg only liquid 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    model = create_model(
        "liquid_ekf",
        {
            "feature_order": ["quality", "valid", "modality_gap_dt"],
            "window": {"size": 1, "step": 1},
            "network": {"input_dim": 3, "hidden_dim": 4},
        },
    )
    captured_window = {}

    def _capture(window_tensor):
        captured_window.update(window_tensor)
        return {
            "bias": torch.tensor(0.0),
            "risk": torch.tensor(0.0),
            "uwb_scaling": torch.tensor(1.0),
            "vio_scaling": torch.tensor(1.0),
        }

    original_predict = model.predict_intermediate_tensors
    model.predict_intermediate_tensors = _capture
    explicit_context_vector = torch.tensor([0.1] * model.output_backbone.context_dim, dtype=torch.float32)
    explicit_filter_context_vector = torch.tensor([0.2] * model.output_backbone.filter_context_dim, dtype=torch.float32)
    try:
        outputs = model.infer_intermediate(
            {
                "current_modality": "uwb",
                "feature_order": ["quality", "valid", "modality_gap_dt"],
                "feature_values": [0.3, 1.0, 0.2],
                "missing_mask": [0, 0, 0],
                "dt": 0.1,
                "feature_window": [[0.3, 1.0, 0.2]],
                "missing_mask_window": [[0, 0, 0]],
                "context_vector": explicit_context_vector,
                "filter_context_vector": explicit_filter_context_vector,
            }
        )
    finally:
        model.predict_intermediate_tensors = original_predict

    assert math.isfinite(outputs.risk)
    assert torch.equal(captured_window["context_vector"], explicit_context_vector)
    assert torch.equal(captured_window["filter_context_vector"], explicit_filter_context_vector)


def test_liquid_predict_intermediate_tensors_consumes_explicit_context_tensors_end_to_end():
    """显式测试：liquid predict intermediate tensors consumes。\n\n验证 liquid predict intermediate tensors consumes 的显式参数优先级，\n确保显式指定覆盖隐式推断。
    """
    model = create_model(
        "liquid_ekf",
        {
            "feature_order": ["quality", "valid", "modality_gap_dt"],
            "window": {"size": 1, "step": 1},
            "network": {"input_dim": 3, "hidden_dim": 4},
        },
    )
    hidden_dim = model.network.hidden_dim
    context_dim = model.output_backbone.context_dim
    filter_context_dim = model.output_backbone.filter_context_dim
    with torch.no_grad():
        model.output_backbone.temporal_context_gate.weight.zero_()
        model.output_backbone.temporal_context_gate.bias.zero_()
        model.output_backbone.observation_context_gate.weight.zero_()
        model.output_backbone.observation_context_gate.bias.zero_()
        model.output_backbone.filter_context_gate.weight.zero_()
        model.output_backbone.filter_context_gate.bias.zero_()
        model.output_backbone.branch_mix_gate.weight.zero_()
        model.output_backbone.branch_mix_gate.bias.zero_()
        model.output_backbone.uwb_branch_mix_gate.weight.zero_()
        model.output_backbone.uwb_branch_mix_gate.bias.zero_()
        model.output_backbone.vio_branch_mix_gate.weight.zero_()
        model.output_backbone.vio_branch_mix_gate.bias.zero_()
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.projection.weight[0, hidden_dim + 0] = 1.0
        model.bias_head.projection.weight[0, hidden_dim + context_dim + 0] = 1.0
        model.bias_head.residual_projection.weight.zero_()
        model.bias_head.residual_projection.bias.zero_()

        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()
        model.uwb_scaling_head.projection.weight.zero_()
        model.uwb_scaling_head.projection.bias.zero_()
        model.vio_scaling_head.projection.weight.zero_()
        model.vio_scaling_head.projection.bias.zero_()

    base_window = {
        "current_modality": "uwb",
        "feature_order": ["quality", "valid", "modality_gap_dt"],
        "feature_values": [0.3, 1.0, 0.2],
        "missing_mask": [0, 0, 0],
        "dt": 0.1,
        "feature_window": [[0.3, 1.0, 0.2]],
        "missing_mask_window": [[0, 0, 0]],
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
            "consecutive_skip_count": True,
            "time_since_last_update": True,
        },
    }
    explicit_context_vector = torch.zeros(context_dim, dtype=torch.float32)
    explicit_filter_context_vector = torch.zeros(filter_context_dim, dtype=torch.float32)
    explicit_context_vector[0] = 0.4
    explicit_filter_context_vector[0] = 0.7
    explicit_window = dict(base_window)
    explicit_window["context_vector"] = explicit_context_vector
    explicit_window["filter_context_vector"] = explicit_filter_context_vector

    fallback_outputs = model.predict_intermediate_tensors(base_window)
    explicit_outputs = model.predict_intermediate_tensors(explicit_window)

    # 显式 context_vector 应该影响模型输出（与 fallback 不同）。
    assert float(fallback_outputs["bias"].detach().item()) != pytest.approx(
        float(explicit_outputs["bias"].detach().item()), abs=1e-6
    )


def test_liquid_output_head_context_groups_keep_modality_indicator_and_expected_feature_slices():
    # 铁律 9：上下文仅 valid / modality_gap_dt / 几何原始量；禁止 quality 教师标签。
    # 布局：modal(0,1) valid(2,3) modality_gap_dt(4,5) residual(6,7) anchor_dx(8,9)
    # anchor_dy(10,11) geom_score(12,13) → temporal = modality_gap_dt → (4, 5)。
    model = create_model(
        "liquid_ekf",
        {
            "feature_order": [
                "valid",
                "modality_gap_dt",
                "uwb_range_residual",
                "anchor_dx",
                "anchor_dy",
                "geom_score",
            ],
            "window": {"size": 1, "step": 1},
            "network": {"input_dim": 6, "hidden_dim": 4},
        },
    )
    head = model.bias_head
    backbone = model.output_backbone

    assert tuple(backbone.temporal_context_indices) == (4, 5)
    assert tuple(model_factory_module.LIQUID_CONTEXT_FEATURE_KEYS) == (
        "valid",
        "modality_gap_dt",
        "uwb_range_residual",
        "anchor_dx",
        "anchor_dy",
        "geom_score",
    )
    assert "quality" not in model_factory_module.LIQUID_CONTEXT_FEATURE_KEYS
    assert "uwb_quality_min" not in model_factory_module.LIQUID_CONTEXT_FEATURE_KEYS
    assert "uwb_invalid_rate" not in model_factory_module.LIQUID_CONTEXT_FEATURE_KEYS
    # 模态索引(0,1)不再包含在fast/branch上下文索引中
    assert 0 not in backbone.uwb_fast_context_indices
    assert 1 not in backbone.uwb_fast_context_indices
    assert 0 not in backbone.vio_fast_context_indices
    assert 1 not in backbone.vio_fast_context_indices
    assert 0 not in backbone.uwb_branch_context_indices
    assert 1 not in backbone.uwb_branch_context_indices
    assert 0 not in backbone.vio_branch_context_indices
    assert 1 not in backbone.vio_branch_context_indices
    # D2-3 v2 redesign: UWB filter context 7 键（dead placeholder nlos_indicator 已清，原 8 键→7 键），
    # state_cov_trace(0,1) pos_cov(2,3) last_gate_skip_flag(6,7) consecutive_skip_count(8,9)
    # time_since_last_update(10,11) uwb_residual_norm(16,17) geometry_dop(20,21)
    assert tuple(backbone.uwb_filter_context_indices) == (0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 16, 17, 20, 21)
    # D2-3 v2 redesign: VIO filter context 8 键（dead placeholder nlos_indicator 删除后 cross_modal_consistency
    # 索引从 (24,25)→(22,23)，读出键总数 15→14）
    # state_cov_trace(0,1) pos_cov(2,3) last_innovation_norm(4,5) consecutive_skip_count(8,9)
    # time_since_last_update(10,11) vel_cov_trace(14,15) vio_cov_summary(18,19) cross_modal_consistency(22,23)
    assert tuple(backbone.vio_filter_context_indices) == (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 18, 19, 22, 23)

    assert int(backbone.uwb_fast_context_mask.sum().item()) == len(backbone.uwb_fast_context_indices)
    assert int(backbone.vio_fast_context_mask.sum().item()) == len(backbone.vio_fast_context_indices)
    assert int(backbone.uwb_branch_context_mask.sum().item()) == len(backbone.uwb_branch_context_indices)
    assert int(backbone.vio_branch_context_mask.sum().item()) == len(backbone.vio_branch_context_indices)
    assert int(backbone.uwb_filter_context_mask.sum().item()) == len(backbone.uwb_filter_context_indices)
    assert int(backbone.vio_filter_context_mask.sum().item()) == len(backbone.vio_filter_context_indices)


def test_checkpoint_path_loads_legacy_liquid_checkpoint_without_new_readout_or_risk_calibration_keys(tmp_path):
    """无依赖测试：checkpoint path loads legacy liquid checkpoint。\n\n验证 checkpoint path loads legacy liquid checkpoint 在缺少依赖时的降级行为，\n确保回退策略正确。
    """
    # v3.1 head-shared 架构：旧 checkpoint 的 gates 在 head.X，risk_calibration 存在；
    # 当前格式下 gates 已迁到 output_backbone.X，head 仅含 projection/residual_projection。
    # 此处构造 v3.1 当前格式的 checkpoint，剥离 risk_calibration 与 output_backbone 上的
    # gate 后验证 create_model(load_checkpoint) 能恢复默认初始化的 risk_calibration 与 gates。
    model_cfg, payload = _liquid_checkpoint_payload()
    reference_model = create_model("liquid_ekf", dict(model_cfg))
    fresh_model = create_model("liquid_ekf", dict(model_cfg))
    legacy_state = dict(payload["model_state"])
    backbone_state = dict(legacy_state["output_backbone"])
    for gate_suffix in (
        "temporal_context_gate", "observation_context_gate", "filter_context_gate",
        "branch_mix_gate", "uwb_branch_mix_gate", "vio_branch_mix_gate",
    ):
        backbone_state.pop(f"{gate_suffix}.weight", None)
        backbone_state.pop(f"{gate_suffix}.bias", None)
    legacy_state["output_backbone"] = backbone_state
    # head 上不再有 gate，无需 pop（v3.1 LiquidOutputHeadLinear 仅 projection/residual_projection）。
    legacy_state.pop("risk_calibration", None)

    checkpoint_path = Path(tmp_path) / "legacy_liquid_readout_checkpoint.pt"
    torch.save(
        {
            "checkpoint_format": "liquid_real_v1",
            "model_cfg": model_cfg,
            "model_state": legacy_state,
        },
        checkpoint_path,
    )

    loaded_model = create_model("liquid_ekf", {"checkpoint_path": checkpoint_path})

    assert hasattr(loaded_model, "risk_calibration")
    assert torch.isfinite(loaded_model.risk_calibration.a_raw.detach()).item()
    assert loaded_model.output_backbone.filter_context_gate.weight.shape[0] == loaded_model.network.hidden_dim
    # D4 v2 redesign: residual_projection 接收 final_hidden+pooled_hidden 拼接 (2*hidden_dim)
    assert loaded_model.bias_head.residual_projection.weight.shape == (1, 2 * loaded_model.network.hidden_dim)
    assert torch.allclose(
        loaded_model.risk_calibration.a_raw.detach(),
        fresh_model.risk_calibration.a_raw.detach(),
    )
    assert torch.allclose(
        loaded_model.risk_calibration.b.detach(),
        fresh_model.risk_calibration.b.detach(),
    )
    # v3.1 head-shared: gates 全部位于 output_backbone，对比 fresh backbone 与 loaded backbone 的全部 gate 参数。
    loaded_backbone = loaded_model.output_backbone
    fresh_backbone = fresh_model.output_backbone
    for gate_suffix in (
        "temporal_context_gate", "observation_context_gate", "filter_context_gate",
        "branch_mix_gate", "uwb_branch_mix_gate", "vio_branch_mix_gate",
    ):
        assert torch.allclose(
            getattr(loaded_backbone, gate_suffix).weight.detach(),
            getattr(fresh_backbone, gate_suffix).weight.detach(),
        )
        assert torch.allclose(
            getattr(loaded_backbone, gate_suffix).bias.detach(),
            getattr(fresh_backbone, gate_suffix).bias.detach(),
        )
    for head_key in ("bias_head", "risk_head", "uwb_scaling_head", "vio_scaling_head"):
        loaded_head = getattr(loaded_model, head_key)
        fresh_head = getattr(fresh_model, head_key)
        assert torch.allclose(
            loaded_head.residual_projection.weight.detach(),
            fresh_head.residual_projection.weight.detach(),
        )
        assert torch.allclose(
            loaded_head.residual_projection.bias.detach(),
            fresh_head.residual_projection.bias.detach(),
        )
        assert loaded_head.projection.weight.shape == fresh_head.projection.weight.shape
    window = {
        "current_modality": "uwb",
        "feature_order": model_cfg["feature_order"],
        "feature_values": [0.3, 1.0, 0.2, 0.15],
        "missing_mask": [0, 0, 0, 0],
        "dt": 0.1,
        "feature_window": [[0.3, 1.0, 0.2, 0.15]],
        "missing_mask_window": [[0, 0, 0, 0]],
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
            "consecutive_skip_count": True,
            "time_since_last_update": True,
        },
    }
    reference_outputs = reference_model.predict_intermediate_tensors(window)
    loaded_outputs = loaded_model.predict_intermediate_tensors(window)
    for key in ("bias", "risk", "uwb_scaling", "vio_scaling"):
        assert float(loaded_outputs[key].detach().item()) == pytest.approx(
            float(reference_outputs[key].detach().item()),
            abs=1e-6,
        )


def test_coerce_scalar_tensor_preserves_tensor_gradients():
    """保持性测试：coerce scalar tensor。\n\n验证 coerce scalar tensor 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    from liquidloc.factories.model_factory import _coerce_scalar_tensor

    value = torch.tensor(1.23, dtype=torch.float32, requires_grad=True)
    scalar = _coerce_scalar_tensor(value, name="value")

    assert scalar.requires_grad is True
    assert scalar.device == value.device


def test_liquid_factory_patched_network_preserves_current_feature_context():
    """保持性测试：liquid factory patched network。\n\n验证 liquid factory patched network 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    model = create_model(
        "liquid_ekf",
        {
            "feature_order": ["quality", "valid"],
            "network": {"input_dim": 2, "hidden_dim": 4},
        },
    )
    shared = model.network.extract_shared_features(
        {
            "current_modality": "uwb",
            "feature_order": ["quality", "valid"],
            "feature_values": [0.2, 0.0],
            "missing_mask": [0, 1],
            "dt": 0.1,
            "feature_window": [[0.2, 0.0]],
            "missing_mask_window": [[0, 1]],
        }
    )

    assert shared["current_feature_by_name"]["quality"] == pytest.approx(0.2)
    assert shared["current_feature_by_name"]["valid"] == pytest.approx(0.0)
    assert shared["current_observed_by_name"]["quality"] is True
    assert shared["current_observed_by_name"]["valid"] is False


def test_liquid_factory_patched_network_uses_window_local_time_differences():
    """使用测试：liquid factory patched network。\n\n验证被测功能正确使用 liquid factory patched network，\n确保内部依赖被正确调用。
    """
    model = create_model(
        "liquid_ekf",
        {
            "feature_order": ["dt", "quality"],
            "network": {"input_dim": 2, "hidden_dim": 4},
        },
    )
    observed_dts: list[float] = []
    original_step = model.network.cell.step

    def _recording_step(inputs, hidden_state, *, dt, missing_mask=None):
        observed_dts.append(float(dt))
        return original_step(inputs, hidden_state, dt=dt, missing_mask=missing_mask)

    model.network.cell.step = _recording_step
    model.network.extract_shared_features(
        {
            "current_modality": "uwb",
            "feature_order": ["dt", "quality"],
            "feature_values": [0.1, 0.8],
            "missing_mask": [0, 0],
            "dt": 0.1,
            "feature_window": [[0.35, 0.7], [0.1, 0.8]],
            "missing_mask_window": [[0, 0], [0, 0]],
            "event_time_window": [8.0, 8.1],
        }
    )

    assert observed_dts == pytest.approx([0.1, 0.1])


def test_liquid_context_tensor_includes_current_uwb_geometry_features():
    """几何测试：liquid context tensor includes current uwb。\n\n验证 liquid context tensor includes current uwb 的几何偏置计算，\n确保锚点-目标几何关系正确。
    """
    from liquidloc.models.liquid.output_head import _build_liquid_context_tensor
    from liquidloc.models.liquid.output_head import LIQUID_CONTEXT_FEATURE_KEYS

    feature_keys = list(LIQUID_CONTEXT_FEATURE_KEYS)
    shared_features = {
        "current_modality": "uwb",
        "current_feature_by_name": {
            "uwb_range_residual": 0.45,
            "anchor_dx": 1.25,
            "anchor_dy": -0.75,
        },
        "current_observed_by_name": {
            "uwb_range_residual": True,
            "anchor_dx": True,
            "anchor_dy": False,
        },
    }

    context_tensor = _build_liquid_context_tensor(
        shared_features,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    context_values = context_tensor.tolist()
    residual_index = 2 + (feature_keys.index("uwb_range_residual") * 2)
    anchor_dx_index = 2 + (feature_keys.index("anchor_dx") * 2)
    anchor_dy_index = 2 + (feature_keys.index("anchor_dy") * 2)

    assert context_values[residual_index] == pytest.approx(0.45)
    assert context_values[residual_index + 1] == pytest.approx(1.0)
    assert context_values[anchor_dx_index] == pytest.approx(1.25)
    assert context_values[anchor_dx_index + 1] == pytest.approx(1.0)
    assert context_values[anchor_dy_index] == pytest.approx(0.0)
    assert context_values[anchor_dy_index + 1] == pytest.approx(0.0)


def test_liquid_filter_context_tensor_accepts_scalar_tensor_metadata():
    """接受测试：liquid filter context tensor。\n\n验证 liquid filter context tensor 的接受行为，\n确保合法输入被正确处理。
    """
    from liquidloc.models.liquid.output_head import _build_liquid_filter_context_tensor

    shared_features = {
        "readout_context_by_name": {
            "state_cov_trace": torch.tensor(4.0),
            "pos_cov": torch.tensor(1.25),
            "last_innovation_norm": torch.tensor(0.45),
            "last_gate_skip_flag": torch.tensor(1.0),
        },
        "readout_context_observed_by_name": {
            "state_cov_trace": True,
            "pos_cov": True,
            "last_innovation_norm": True,
            "last_gate_skip_flag": True,
            "consecutive_skip_count": True,
            "time_since_last_update": True,
        },
    }

    context_tensor = _build_liquid_filter_context_tensor(
        shared_features,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # LIQUID_READOUT_CONTEXT_KEYS 共 14 键 × 2(值+标志) = 28 维。
    # 测试只提供了 4 个键的值和 6 个键的 observed 标志，其余为默认值 0.0/False。
    # v2: 14 键（原 6 键 + 8 个扩展键），扩展键默认值 0.0, observed=False
    assert len(context_tensor.tolist()) == 28
    expected = [
        4.0, 1.0,   # state_cov_trace (value=4.0, observed=True)
        1.25, 1.0,  # pos_cov (value=1.25, observed=True)
        0.45, 1.0,  # last_innovation_norm (value=0.45, observed=True)
        1.0, 1.0,   # last_gate_skip_flag (value=1.0, observed=True)
        0.0, 1.0,   # consecutive_skip_count (value=0.0, observed=True)
        0.0, 1.0,   # time_since_last_update (value=0.0, observed=True)
        # 以下 8 个扩展键：未提供值且 observed 默认为 False，因此 value=0.0, observed=False
        0.0, 0.0,   # pos_cov_trace
        0.0, 0.0,   # vel_cov_trace
        0.0, 0.0,   # uwb_residual_norm
        0.0, 0.0,   # vio_cov_summary
        0.0, 0.0,   # geometry_dop
        0.0, 0.0,   # cross_modal_consistency
        0.0, 0.0,   # voxel_feature
        0.0, 0.0,   # last_update_skip_flag
    ]
    assert context_tensor.tolist() == pytest.approx(expected)


def test_lstm_context_tensor_includes_current_uwb_geometry_features():
    """几何测试：lstm context tensor includes current uwb。\n\n验证 lstm context tensor includes current uwb 的几何偏置计算，\n确保锚点-目标几何关系正确。
    """
    from liquidloc.models.lstm.network import _CONTEXT_FEATURE_KEYS
    from liquidloc.models.lstm.network import build_lstm_context_tensor_from_normalized_window

    shared_window = {
        "current_modality": "uwb",
        "feature_order": [
            "quality",
            "valid",
            "uwb_range_residual",
            "anchor_dx",
            "anchor_dy",
        ],
        "feature_values": [0.2, 1.0, 0.45, 1.25, -0.75],
        "missing_mask": [0, 0, 0, 0, 1],
    }

    context_tensor = build_lstm_context_tensor_from_normalized_window(
        shared_window,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    context_values = context_tensor.tolist()

    residual_index = 2 + (2 * _CONTEXT_FEATURE_KEYS.index("uwb_range_residual"))
    anchor_dx_index = 2 + (2 * _CONTEXT_FEATURE_KEYS.index("anchor_dx"))
    anchor_dy_index = 2 + (2 * _CONTEXT_FEATURE_KEYS.index("anchor_dy"))

    assert context_values[residual_index] == pytest.approx(0.45)
    assert context_values[residual_index + 1] == pytest.approx(1.0)
    assert context_values[anchor_dx_index] == pytest.approx(1.25)
    assert context_values[anchor_dx_index + 1] == pytest.approx(1.0)
    assert context_values[anchor_dy_index] == pytest.approx(0.0)
    assert context_values[anchor_dy_index + 1] == pytest.approx(0.0)


def test_default_liquid_and_lstm_configs_have_matched_compare_cost():
    config_root = Path(__file__).resolve().parents[2] / "configs" / "models"
    liquid_cfg = yaml.safe_load((config_root / "liquid_ekf.yaml").read_text(encoding="utf-8"))
    lstm_cfg = yaml.safe_load((config_root / "lstm_ekf.yaml").read_text(encoding="utf-8"))

    assert liquid_cfg["feature_order"] == lstm_cfg["feature_order"]
    assert liquid_cfg["network"]["input_dim"] == lstm_cfg["network"]["input_dim"]

    liquid_model = create_model("liquid_ekf", liquid_cfg)
    lstm_model = create_model("lstm_ekf", lstm_cfg)
    liquid_params = sum(int(parameter.numel()) for parameter in liquid_model.parameters())
    lstm_params = sum(int(parameter.numel()) for parameter in lstm_model.parameters())

    # 铁律 9 再削 quality 系上下文后 context_dim=14；主 yaml input_dim=9 对齐。
    # liquid 仍含多层 context gate + 4 头 readout；容忍度 4× 保持。
    assert liquid_params <= lstm_params * 4


def test_default_liquid_and_lstm_train_configs_match_paper_grade_budget_contract():
    """合同测试：default liquid and lstm train configs match paper grade budget。\n\n验证 default liquid and lstm train configs match paper grade budget 的接口合同，\n确保输入输出符合协议约定。
    """
    config_root = Path(__file__).resolve().parents[2] / "configs" / "models"
    liquid_cfg = yaml.safe_load((config_root / "liquid_ekf.yaml").read_text(encoding="utf-8"))
    lstm_cfg = yaml.safe_load((config_root / "lstm_ekf.yaml").read_text(encoding="utf-8"))

    liquid_train = liquid_cfg["train"]
    lstm_train = lstm_cfg["train"]

    assert liquid_train["epochs"] == 160
    assert lstm_train["epochs"] == 160
    assert liquid_train["batch_size"] == 32
    assert lstm_train["batch_size"] == 32
    assert liquid_train["eval_batch_size"] == 128
    assert lstm_train["eval_batch_size"] == 128
    assert liquid_train["amp_enabled"] == "auto"
    assert lstm_train["amp_enabled"] == "auto"
    assert liquid_train["amp_dtype"] == "bf16"
    assert lstm_train["amp_dtype"] == "bf16"
    assert liquid_train["save_epoch_candidates"] is True
    assert lstm_train["save_epoch_candidates"] is True
    assert liquid_train["epoch_candidate_stride"] == 10
    assert lstm_train["epoch_candidate_stride"] == 10
    assert liquid_train["deterministic"] is True
    assert lstm_train["deterministic"] is True
    assert liquid_train["weight_decay"] == pytest.approx(0.0)
    assert lstm_train["weight_decay"] == pytest.approx(0.0)
    assert liquid_train["lr"] == pytest.approx(1.5e-4)
    assert lstm_train["lr"] == pytest.approx(1.5e-4)
    assert liquid_train["phase_schedule"] == {
        "warmup_epochs": 20,
        "gate_alignment_epochs": 40,
        "full_tuning_epochs": 100,
    }
    assert lstm_train["patience"] == 160


def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    model = create_model('liquid_ekf', {})
    try:
        model.infer_intermediate(
            {
                'current_modality': 'uwb',
                'feature_order': ['quality', 'valid', 'modality_gap_dt', 'uwb_quality_min'],
                'feature_values': [0.2, 0.0, 0.5, 0.2],
                'missing_mask': [0, 0, 0, 0],
                'dt': 0.1,
            }
        )
    except (TypeError, ValueError) as exc:
        assert 'feature_window' in str(exc)
    else:
        raise AssertionError('Expected incomplete structured window to raise')


def test_checkpoint_path_loads_liquid_weights_only_payload(tmp_path):
    model_cfg, payload = _liquid_checkpoint_payload()
    checkpoint_path = Path(tmp_path) / "liquid_checkpoint.pt"
    torch.save(payload, checkpoint_path)

    loaded_model = create_model("liquid_ekf", {"checkpoint_path": checkpoint_path})

    assert loaded_model.expected_feature_order == model_cfg["feature_order"]
    assert loaded_model.checkpoint_meta["checkpoint_path"] == str(checkpoint_path)
    assert loaded_model.runtime_device == "cpu"
    assert torch.allclose(
        loaded_model.network.state_dict()["shared_projection.weight"],
        payload["model_state"]["network"]["shared_projection.weight"],
    )
    assert torch.allclose(
        loaded_model.bias_head.state_dict()["projection.weight"],
        payload["model_state"]["bias_head"]["projection.weight"],
    )


def test_checkpoint_path_overrides_liquid_flat_model_cfg(tmp_path):
    """覆盖测试：checkpoint path。\n\n验证 checkpoint path 的覆盖行为，\n确保显式参数优先于默认值。
    """
    model_cfg, payload = _liquid_checkpoint_payload_with_flat_model_cfg()
    checkpoint_path = Path(tmp_path) / "liquid_flat_checkpoint.pt"
    torch.save(payload, checkpoint_path)

    loaded_model = create_model(
        "liquid_ekf",
        {
            "feature_order": ["quality", "valid"],
            "window": {"size": 1, "step": 1},
            "network": {"hidden_dim": 64, "input_dim": 2},
            "checkpoint_path": checkpoint_path,
        },
    )

    assert loaded_model.cfg["feature_order"] == model_cfg["feature_order"]
    assert loaded_model.cfg["window"] == model_cfg["window"]
    assert loaded_model.network.hidden_dim == model_cfg["hidden_dim"]
    assert loaded_model.network.input_dim == model_cfg["input_dim"]
    assert loaded_model.expected_feature_order == model_cfg["feature_order"]
    assert loaded_model.network.shared_projection.weight.shape[0] == model_cfg["hidden_dim"]


def test_checkpoint_path_loads_lstm_weights_only_payload(tmp_path):
    model_cfg, payload = _lstm_checkpoint_payload()
    checkpoint_path = Path(tmp_path) / "lstm_checkpoint.pt"
    torch.save(payload, checkpoint_path)

    loaded_model = create_model("lstm_ekf", {"checkpoint_path": checkpoint_path})

    assert loaded_model.expected_feature_order == model_cfg["feature_order"]
    assert loaded_model.checkpoint_meta["checkpoint_path"] == str(checkpoint_path)
    assert loaded_model.runtime_device == "cpu"
    assert torch.allclose(
        loaded_model.network.state_dict()["input_projection.weight"],
        payload["model_state"]["network"]["input_projection.weight"],
    )
    outputs = loaded_model.predict_intermediate_tensors(_lstm_structured_window(model_cfg["feature_order"]))
    assert outputs["uwb_scaling"].item() >= 0.5
    assert outputs["vio_scaling"].item() >= 0.5


def test_checkpoint_path_overrides_lstm_flat_model_cfg(tmp_path):
    """覆盖测试：checkpoint path。\n\n验证 checkpoint path 的覆盖行为，\n确保显式参数优先于默认值。
    """
    model_cfg, payload = _lstm_checkpoint_payload_with_flat_model_cfg()
    checkpoint_path = Path(tmp_path) / "lstm_flat_checkpoint.pt"
    torch.save(payload, checkpoint_path)

    loaded_model = create_model(
        "lstm_ekf",
        {
            "feature_order": ["dt", "ax"],
            "window": {"size": 1, "step": 1},
            "network": {"hidden_dim": 64, "input_dim": 2},
            "checkpoint_path": checkpoint_path,
        },
    )

    assert loaded_model.cfg["feature_order"] == model_cfg["feature_order"]
    assert loaded_model.cfg["window"] == model_cfg["window"]
    assert loaded_model.network.hidden_dim == model_cfg["hidden_dim"]
    assert loaded_model.network.input_dim == model_cfg["input_dim"]
    assert loaded_model.expected_feature_order == model_cfg["feature_order"]
    # 铁律 9：_CONTEXT_FEATURE_KEYS 6 键 → _CONTEXT_DIM=14；
    # output_layer.in_features = hidden_dim(6) + context_dim(14) = 20。
    from liquidloc.models.lstm.network import _CONTEXT_DIM

    assert loaded_model.network.context_dim == _CONTEXT_DIM
    assert loaded_model.network.output_layer.in_features == model_cfg["hidden_dim"] + _CONTEXT_DIM


def test_checkpoint_path_loaded_lstm_rejects_feature_order_mismatch(tmp_path):
    """拒绝测试：checkpoint path loaded lstm。\n\n验证被测功能对 checkpoint path loaded lstm 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    model_cfg, payload = _lstm_checkpoint_payload()
    checkpoint_path = Path(tmp_path) / "lstm_checkpoint.pt"
    torch.save(payload, checkpoint_path)

    loaded_model = create_model("lstm_ekf", {"checkpoint_path": checkpoint_path})

    with pytest.raises(ValueError, match="window_tensor.feature_order must match the model expected_feature_order"):
        loaded_model.predict_intermediate_tensors(_lstm_structured_window(list(reversed(model_cfg["feature_order"]))))


def test_cfg_only_lstm_rejects_feature_order_mismatch():
    """拒绝测试：cfg only lstm。\n\n验证被测功能对 cfg only lstm 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    model_cfg = _lstm_checkpoint_payload()[0]
    model = create_model("lstm_ekf", dict(model_cfg))

    with pytest.raises(ValueError, match="window_tensor.feature_order must match the model expected_feature_order"):
        model.predict_intermediate_tensors(_lstm_structured_window(list(reversed(model_cfg["feature_order"]))))


def test_cfg_only_lstm_rejects_missing_dt():
    """拒绝测试：cfg only lstm。\n\n验证被测功能对 cfg only lstm 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    model = create_model("lstm_ekf", _lstm_checkpoint_payload()[0])
    window_tensor = _lstm_structured_window(model.expected_feature_order)
    window_tensor.pop("dt")

    with pytest.raises(ValueError, match="feature_window.dt must be provided explicitly"):
        model.predict_intermediate_tensors(window_tensor)


def test_checkpoint_path_loaded_lstm_rejects_unstructured_tensor_input(tmp_path):
    """拒绝测试：checkpoint path loaded lstm。\n\n验证被测功能对 checkpoint path loaded lstm 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    _, payload = _lstm_checkpoint_payload()
    checkpoint_path = Path(tmp_path) / "lstm_checkpoint.pt"
    torch.save(payload, checkpoint_path)

    loaded_model = create_model("lstm_ekf", {"checkpoint_path": checkpoint_path})

    with pytest.raises(
        TypeError,
        match="lstm_ekf with cfg/checkpoint expected_feature_order requires a structured feature window mapping",
    ):
        loaded_model.predict_intermediate_tensors(torch.zeros((1, 3, 4), dtype=torch.float32))


def test_cfg_only_lstm_rejects_unstructured_tensor_input_when_feature_order_is_configured():
    """拒绝测试：cfg only lstm。\n\n验证被测功能对 cfg only lstm 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    model = create_model("lstm_ekf", _lstm_checkpoint_payload()[0])

    with pytest.raises(
        TypeError,
        match="lstm_ekf with cfg/checkpoint expected_feature_order requires a structured feature window mapping",
    ):
        model.predict_intermediate_tensors(torch.zeros((1, 3, 4), dtype=torch.float32))


def test_lstm_output_normalization_matches_training_projection():
    """匹配测试：lstm output normalization。\n\n验证 lstm output normalization 的输出与预期一致，\n确保合同合规。
    """
    model = create_model("lstm_ekf", _lstm_checkpoint_payload()[0])

    def _raw_output(_window_tensor):
        return torch.tensor([-1.0, 1.5, -0.2, -0.3])

    model.network.forward = _raw_output

    outputs = model.infer_intermediate(_lstm_structured_window())

    assert outputs.bias == pytest.approx(0.0, rel=1e-6)  # bias非负约束将-1.0裁剪为0.0
    assert outputs.risk == pytest.approx(0.817574, rel=1e-6)
    # scaling_min=1.0 后，softplus(-0.2)+1.0-softplus(0)=0.905 被 clamp 到 floor=1.0
    # softplus(-0.3)+1.0-softplus(0)=0.799 被 clamp 到 floor=1.0
    assert outputs.uwb_scaling == pytest.approx(1.0, rel=1e-3)
    assert outputs.vio_scaling == pytest.approx(1.0, rel=1e-3)


def test_infer_intermediate_runs_without_grad_tracking():
    """追踪测试：infer intermediate runs without grad。\n\n验证 infer intermediate runs without grad 的追踪机制，\n确保状态变化被正确记录。
    """
    model = create_model("lstm_ekf", _lstm_checkpoint_payload()[0])

    def _raw_output(_window_tensor):
        return torch.tensor([-1.0, 1.5, -0.2, -0.3], requires_grad=True)

    model.network.forward = _raw_output
    outputs = model.infer_intermediate(_lstm_structured_window())

    assert outputs.bias == pytest.approx(0.0, rel=1e-6)  # bias非负约束将-1.0裁剪为0.0


def test_lstm_modality_output_contract_forces_cross_modality_scaling_to_neutral():
    """LSTM predict_intermediate_tensors 按模态合同约束：非当前模态缩放到 [scaling_min, scaling_ceiling]。"""
    model = create_model("lstm_ekf", _lstm_checkpoint_payload()[0])

    def _raw_output(_window_tensor):
        # bias=0.0, risk=0.0, uwb_scaling=2.0, vio_scaling=3.0
        return torch.tensor([0.0, 0.0, 2.0, 3.0])

    model.network.forward = _raw_output

    # v3 改造: 非当前模态 scaling 从硬 mask (==1.0) 改为 soft-mask clamp 到
    # [scaling_min=0.5, scaling_ceiling=2.5]; 与 apply_liquid_modality_output_contract 同口径.
    scaling_min, scaling_ceiling = 0.5, 2.5

    # UWB 模态: uwb_scaling 留在 [0.5, 2.5], vio_scaling clamp 到 [0.5, 2.5] (原值 3.0 → 2.5).
    uwb_outputs = model.predict_intermediate_tensors(_lstm_structured_window())
    assert scaling_min <= uwb_outputs["uwb_scaling"].item() <= scaling_ceiling
    assert uwb_outputs["vio_scaling"].item() <= scaling_ceiling
    assert uwb_outputs["vio_scaling"].item() >= scaling_min

    # VIO 模态: uwb_scaling clamp 到 [0.5, 2.5], 当前模态 vio_scaling 不受 cross-modality clamp.
    # 该值仅受归一化 (softplus + scaling_min) 限制, 不在 [scaling_min, scaling_ceiling] 内,
    # 顶层仍受 BRIDGE_THRESHOLDS["scaling_max"]=50 与 infer_intermediate clamp 限制.
    vio_window = dict(_lstm_structured_window())
    vio_window["current_modality"] = "vio"
    vio_outputs = model.predict_intermediate_tensors(vio_window)
    assert vio_outputs["uwb_scaling"].item() <= scaling_ceiling
    assert vio_outputs["uwb_scaling"].item() >= scaling_min
    # current modality vio_scaling 不被 cross-modality clamp, 但仍应是有限正数.
    assert vio_outputs["vio_scaling"].item() >= scaling_min
    assert torch.isfinite(torch.as_tensor(vio_outputs["vio_scaling"].item())).item()


def test_lstm_output_normalization_rejects_non_finite_scaling():
    """拒绝测试：lstm output normalization。\n\n验证被测功能对 lstm output normalization 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    model = create_model("lstm_ekf", _lstm_checkpoint_payload()[0])

    def _raw_output(_window_tensor):
        return torch.tensor([-1.0, 1.5, float("inf"), 0.7])

    model.network.forward = _raw_output

    with pytest.raises(ValueError, match="uwb_scaling must be finite"):
        model.infer_intermediate(
            _lstm_structured_window()
        )


def test_checkpoint_path_loads_legacy_lstm_output_layer_weights_only_payload(tmp_path):
    model_cfg, payload = _lstm_checkpoint_payload()
    legacy_network_state = dict(payload["model_state"]["network"])
    legacy_network_state["output_layer.weight"] = legacy_network_state["output_layer.weight"][
        :, : model_cfg["network"]["hidden_dim"]
    ].clone()
    checkpoint_path = Path(tmp_path) / "legacy_lstm_checkpoint.pt"
    torch.save(
        {
            "checkpoint_format": "lstm_real_v1",
            "model_cfg": model_cfg,
            "model_state": {"network": legacy_network_state},
        },
        checkpoint_path,
    )

    loaded_model = create_model("lstm_ekf", {"checkpoint_path": checkpoint_path})
    loaded_weight = loaded_model.network.state_dict()["output_layer.weight"]

    assert loaded_weight.shape[1] > legacy_network_state["output_layer.weight"].shape[1]
    assert torch.allclose(
        loaded_weight[:, : legacy_network_state["output_layer.weight"].shape[1]],
        legacy_network_state["output_layer.weight"],
    )
    assert torch.count_nonzero(
        loaded_weight[:, legacy_network_state["output_layer.weight"].shape[1] :]
    ).item() == 0


def test_lstm_factory_output_depends_on_current_step_context():
    """依赖测试：lstm factory output。\n\n验证 lstm factory output 的依赖关系，\n确保输出随输入变化。
    """
    # 上下文第一观测键是 valid（layout: modal 0,1 | valid 2,3 | ...）。
    # 探针接在 hidden_dim+2 = valid 数值位；feature_order 不得用 quality 教师标签。
    model = create_model(
        "lstm_ekf",
        {
            "feature_order": ["valid", "modality_gap_dt"],
            "network": {
                "input_dim": 4,
                "hidden_dim": 4,
                "output_heads": ["bias", "risk", "uwb_scaling", "vio_scaling"],
            },
        },
    )

    with torch.no_grad():
        for parameter in model.network.lstm.parameters():
            parameter.zero_()
        model.network.output_layer.weight.zero_()
        model.network.output_layer.bias.zero_()
        model.network.output_layer.weight[0, model.network.hidden_dim + 2] = 1.0

    base_window = {
        "current_modality": "uwb",
        "feature_order": ["valid", "modality_gap_dt"],
        "feature_values": [0.25, 0.1],
        "missing_mask": [0, 0],
        "dt": 0.1,
        "feature_window": [[0.25, 0.1]],
        "missing_mask_window": [[0, 0]],
    }
    richer_window = dict(base_window)
    richer_window["feature_values"] = [0.75, 0.1]
    richer_window["feature_window"] = [[0.75, 0.1]]

    low_bias = float(model.predict_intermediate_tensors(base_window)["bias"].detach().item())
    high_bias = float(model.predict_intermediate_tensors(richer_window)["bias"].detach().item())

    assert high_bias > low_bias


class _UnsupportedCheckpointObject:
    pass


def test_checkpoint_path_rejects_unsupported_serialized_objects(tmp_path):
    """拒绝测试：checkpoint path。\n\n验证被测功能对 checkpoint path 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    checkpoint_path = Path(tmp_path) / "unsafe_checkpoint.pt"
    torch.save({"unsafe": _UnsupportedCheckpointObject()}, checkpoint_path)

    with pytest.raises(ValueError, match="unsupported serialized objects"):
        create_model("liquid_ekf", {"checkpoint_path": checkpoint_path})


def test_relative_checkpoint_path_requires_project_root(tmp_path, monkeypatch):
    """必填测试：relative checkpoint path。\n\n验证 relative checkpoint path 的必填约束，\n确保缺少必要输入时抛出异常。
    """
    checkpoint_path = Path("checkpoints") / f"relative_{uuid4().hex}.pt"
    model_cfg, payload = _liquid_checkpoint_payload()
    project_root = Path(__file__).resolve().parents[2]
    abs_checkpoint_path = project_root / checkpoint_path
    abs_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, abs_checkpoint_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="relative checkpoint_path requires cfg.project_root or an absolute path"):
        create_model("liquid_ekf", {"checkpoint_path": checkpoint_path})


def test_relative_checkpoint_path_uses_project_root_anchor(tmp_path, monkeypatch):
    """使用测试：relative checkpoint path。\n\n验证被测功能正确使用 relative checkpoint path，\n确保内部依赖被正确调用。
    """
    checkpoint_path = Path("checkpoints") / f"anchored_{uuid4().hex}.pt"
    model_cfg, payload = _liquid_checkpoint_payload()
    project_root = Path(__file__).resolve().parents[2]
    abs_checkpoint_path = project_root / checkpoint_path
    abs_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, abs_checkpoint_path)
    monkeypatch.chdir(tmp_path)

    loaded_model = create_model("liquid_ekf", {
        "checkpoint_path": checkpoint_path,
        "project_root": project_root,
    })

    assert loaded_model.checkpoint_meta["checkpoint_path"] == str(abs_checkpoint_path)


def test_liquid_inference_device_cuda_request_uses_explicit_runtime_device(monkeypatch):
    """使用测试：liquid inference device cuda request。\n\n验证被测功能正确使用 liquid inference device cuda request，\n确保内部依赖被正确调用。
    """
    captured_devices = []
    monkeypatch.setattr(model_factory_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_factory_module, "cuda_runtime_usable", lambda: True)
    monkeypatch.setattr(
        model_factory_module,
        "_move_modules_to_device",
        lambda modules, device: captured_devices.append(device.type),
    )

    model = create_model(
        "liquid_ekf",
        {
            "feature_order": ["quality", "valid", "modality_gap_dt", "uwb_quality_min"],
            "window": {"size": 1, "step": 1},
            "network": {"input_dim": 4, "hidden_dim": 4},
            "inference_device": "cuda",
        },
    )

    assert captured_devices[-1] == "cuda"
    assert model.runtime_device == "cuda"


def test_lstm_inference_device_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    """无依赖测试：lstm inference device auto falls back to cpu。\n\n验证 lstm inference device auto falls back to cpu 在缺少依赖时的降级行为，\n确保回退策略正确。
    """
    monkeypatch.setattr(model_factory_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(model_factory_module, "cuda_runtime_usable", lambda: False)

    model = create_model(
        "lstm_ekf",
        {
            "feature_order": ["dt", "ax", "modality_gap_dt", "vio_reproj_err_slope"],
            "window": {"size": 3, "step": 1},
            "network": {"input_dim": 4, "hidden_dim": 4},
            "inference_device": "auto",
        },
    )

    assert model.runtime_device == "cpu"


def test_checkpoint_path_loads_legacy_liquid_checkpoint_without_pooling_or_reliability_gate(tmp_path):
    """无依赖测试：checkpoint path loads legacy liquid checkpoint。\n\n验证 checkpoint path loads legacy liquid checkpoint 在缺少依赖时的降级行为，\n确保回退策略正确。
    """
    model_cfg, payload = _liquid_checkpoint_payload()
    legacy_network_state = dict(payload["model_state"]["network"])
    legacy_network_state.pop("pooling_gate.weight", None)
    legacy_network_state.pop("pooling_gate.bias", None)
    legacy_network_state.pop("cell.reliability_projection.weight", None)
    legacy_network_state.pop("cell.reliability_projection.bias", None)
    checkpoint_path = Path(tmp_path) / "legacy_liquid_checkpoint.pt"
    torch.save(
        {
            "checkpoint_format": "liquid_real_v1",
            "model_cfg": model_cfg,
            "model_state": {
                **payload["model_state"],
                "network": legacy_network_state,
            },
        },
        checkpoint_path,
    )

    loaded_model = create_model("liquid_ekf", {"checkpoint_path": checkpoint_path})

    assert loaded_model.network.pooling_gate is not None
    assert loaded_model.network.cell is not None
    assert loaded_model.network.cell.reliability_projection is not None


def test_unknown_transformer_name_rejected_with_section10_2_bidirectional_guard():
    """§10.2 双向 TF 禁区前瞻守卫测试。

    spec 第 1638 行 / 第 1686 行：禁止双向 Transformer 作主表。model_factory.py L983
    `_TRANSFORMER_BIDIRECTIONAL_FORBIDDEN = True` + L2754-2766 前瞻守卫要求：
    cfg.network.bidirectional=True 时 fail-loud 拒绝（无论模型名是否已知）。

    本测试验证：传入 cfg.network.bidirectional=True 时必须 raise ValueError 含
    §10.2 双向 TF 禁区消息——防止未来开发者未审未补守卫即把双向 Transformer 路由
    偷渡进主表。

    注：`MODEL_NAME_TRANSFORMER` 已加入 _SUPPORTED 且已实现，create_model("transformer_ekf",
    {"window": {"size": 20}, "network": {"bidirectional": True}}) 会被前置守卫在
    build_modules 之前拦截，触发 §10.2 双向禁令 ValueError。
    参见 test_transformer_ekf_supported_and_implanted（无 bidirectional 时的成功路径）。
    """
    # cfg.network.bidirectional=True 时必须被前瞻守卫拒绝（无论 transformer 名是否在 _SUPPORTED）。
    with pytest.raises(ValueError, match="§10.2 双向 TF 禁区"):
        create_model("transformer_ekf", {
            "window": {"size": 20},
            "network": {"bidirectional": True},
        })


def test_unknown_transformer_name_guard_distinguishes_from_generic_unknown():
    """§10.2 守卫与通用 Unknown model 错误消息区分。

    非 Transformer 的未知模型名应走通用 'Unknown model:' 路径，
    不应误触 §10.2 双向 TF 禁区守卫——避免错误消息误导调用方。
    """
    with pytest.raises(ValueError, match="Unknown model:"):
        create_model("nonexistent_xyz", {"window": {"size": 20}})



def test_transformer_ekf_supported_and_implanted():
    """§10.2 真实现：MODEL_NAME_TRANSFORMER 在 _SUPPORTED 中且已实现。

    §10.2 表行2 要求 "Transformer+EKF ... 同上 EKF 递推 + 网络隐状态（有限维）"。
    spec §10.2 要求 "若实现则有限维隐状态 + 单向因果"。
    Transformer 模型现已实现（strict causal transformer encoder + risk calibration），
    此测试验证 create_model("transformer_ekf") 可以成功实例化。
    与 §10.2 双向 TF 禁区守卫 (_TRANSFORMER_BIDIRECTIONAL_FORBIDDEN) 互补：
    双向禁令仍防未来偷用双向；本 pass 测试确认当前走的是严格因果路径。
    """
    import torch
    from liquidloc.common.constants import MODEL_NAME_TRANSFORMER
    cfg = {
        "name": MODEL_NAME_TRANSFORMER,
        "feature_order": ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
        "window": {"size": 8},
        "network": {"hidden_size": 18, "num_layers": 1, "nhead": 3, "dim_feedforward": 72},
    }
    model = create_model(MODEL_NAME_TRANSFORMER, cfg)
    assert model.name == MODEL_NAME_TRANSFORMER
    assert hasattr(model, "network")
    assert hasattr(model, "risk_calibration")
    assert hasattr(model, "params")
    # 验证 forward 返回四个头
    feature_dim = len(cfg["feature_order"])
    seq_len = cfg["window"]["size"]
    window = {
        "current_modality": "uwb",
        "feature_order": cfg["feature_order"],
        "feature_values": torch.ones(feature_dim),
        "missing_mask": torch.zeros(feature_dim),
        "dt": 0.01,
        "feature_window": torch.zeros(seq_len, feature_dim),
        "missing_mask_window": torch.ones(seq_len, feature_dim),
    }
    window["feature_window"][-1] = window["feature_values"]
    window["missing_mask_window"][-1] = window["missing_mask"]
    with torch.no_grad():
        mi = model.infer_intermediate(window)
    assert mi.bias >= 0.0
    assert 0.0 <= mi.risk <= 1.05
    assert mi.uwb_scaling >= 1.0
    assert mi.vio_scaling >= 1.0


def test_transformer_ekf_bidirectional_true_rejected_by_section10_2_guard():
    """§10.2 双向 TF 禁区前瞻守卫：cfg.network.bidirectional=True 必须被拒。

    spec 第 1638 行 / 第 1686 行：禁止双向 Transformer 作主表。
    model_factory.py L2754-2766 守卫要求：未来 Transformer 模型必须强制
    bidirectional=False 后方可进主表；cfg.network.bidirectional=True 时
    fail-loud 拒绝。
    """
    from liquidloc.common.constants import MODEL_NAME_TRANSFORMER
    with pytest.raises(ValueError, match="§10.2 双向 TF 禁区"):
        create_model(MODEL_NAME_TRANSFORMER, {
            "window": {"size": 20},
            "network": {"bidirectional": True},
        })
