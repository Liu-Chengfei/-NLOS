"""Liquid 网络结构（liquid_network）测试模块。

测试覆盖范围：
- Liquid 神经网络的前向传播
- 网络参数与配置的一致性
- 输出头的维度验证

被测模块：liquidloc.models.liquid_network"""

import pytest
import torch

from liquidloc.common.seed_utils import cuda_runtime_usable
from liquidloc.models.liquid.network import LiquidNetwork, _normalize_window_tensor


def test_normal_case():
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    network = LiquidNetwork({"hidden_dim": 4})
    window_tensor = {
        "current_modality": "uwb",
        "feature_order": ["ax", "ay", "az"],
        "feature_values": [0.5, 1.0, 1.5],
        "missing_mask": [0, 0, 0],
        "dt": 0.1,
        "feature_window": [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]],
        "missing_mask_window": [[0, 0, 0], [0, 0, 0]],
        "window_index_map": [0, 1],
    }

    result = network.forward(window_tensor)

    assert "shared_features" in result
    shared = result["shared_features"]
    assert shared["shared_vector"].shape == (4,)
    assert shared["hidden_sequence"].shape == (2, 4)
    assert shared["feature_order"] == ["ax", "ay", "az"]
    assert torch.isfinite(shared["shared_vector"]).all()


def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    network = LiquidNetwork({"hidden_dim": 4})
    window_tensor = {
        "current_modality": "uwb",
        "feature_order": ["ax", "ay", "az"],
        "feature_values": [1.0, 2.0, 3.0],
        "missing_mask": [0, 0, 0],
        "dt": 0.1,
        "feature_window": [[1.0, 2.0, 3.0]],
        "missing_mask_window": [[0, 0, 0]],
        "window_index_map": [0],
    }

    with pytest.raises(ValueError, match="initial_state must have shape"):
        network.extract_shared_features(window_tensor, initial_state=[0.0, 0.0])


def test_step_dts_use_window_local_time_differences_when_event_times_are_available():
    normalized = _normalize_window_tensor(
        {
            "current_modality": "uwb",
            "feature_order": ["dt", "modality_gap_dt", "quality"],
            "feature_values": [0.10, 0.22, 0.8],
            "missing_mask": [0, 0, 0],
            "dt": 0.1,
            "feature_window": [
                [0.35, 0.20, 0.7],
                [0.10, 0.22, 0.8],
            ],
            "missing_mask_window": [
                [0, 0, 0],
                [0, 0, 0],
            ],
            "window_index_map": [0, 1],
            "event_time_window": [4.00, 4.10],
        }
    )

    assert normalized["step_dts"] == pytest.approx([0.1, 0.10])


def test_step_dts_fall_back_to_dt_feature_when_event_times_are_missing():
    """缺失测试：step dts fall back to dt feature when event times are。\n\n验证 step dts fall back to dt feature when event times are 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    normalized = _normalize_window_tensor(
        {
            "current_modality": "uwb",
            "feature_order": ["dt", "modality_gap_dt", "quality"],
            "feature_values": [0.10, 0.22, 0.8],
            "missing_mask": [0, 0, 0],
            "dt": 0.1,
            "feature_window": [
                [0.05, 0.20, 0.7],
                [0.10, 0.22, 0.8],
            ],
            "missing_mask_window": [
                [0, 0, 0],
                [0, 0, 0],
            ],
            "window_index_map": [0, 1],
        }
    )

    assert normalized["step_dts"] == pytest.approx([0.05, 0.10])


def test_step_dts_fall_back_to_explicit_dt_when_event_times_and_dt_feature_are_missing():
    """缺失测试：step dts fall back to explicit dt when event times and dt feature are。\n\n验证 step dts fall back to explicit dt when event times and dt feature are 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    normalized = _normalize_window_tensor(
        {
            "current_modality": "uwb",
            "feature_order": ["quality", "valid", "modality_gap_dt"],
            "feature_values": [0.8, 1.0, 0.22],
            "missing_mask": [0, 0, 0],
            "dt": 0.3,
            "feature_window": [
                [0.7, 1.0, 0.20],
                [0.8, 1.0, 0.22],
            ],
            "missing_mask_window": [
                [0, 0, 0],
                [0, 0, 0],
            ],
            "window_index_map": [0, 1],
        }
    )

    assert normalized["step_dts"] == pytest.approx([0.3, 0.3])


def test_normalize_window_tensor_rejects_current_step_mirror_mismatch():
    """拒绝测试：normalize window tensor。\n\n验证被测功能对 normalize window tensor 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(ValueError, match="current-step mirrors must match the last window row"):
        _normalize_window_tensor(
            {
                "current_modality": "uwb",
                "feature_order": ["quality", "valid"],
                "feature_values": [0.9, 1.0],
                "missing_mask": [0, 0],
                "dt": 0.1,
                "feature_window": [[0.7, 1.0], [0.8, 1.0]],
                "missing_mask_window": [[0, 0], [0, 0]],
                "window_index_map": [0, 1],
            }
        )


def test_zero_initialized_pooling_gate_starts_uniform_without_dt_bias():
    """无依赖测试：zero initialized pooling gate starts uniform。\n\n验证 zero initialized pooling gate starts uniform 在缺少依赖时的降级行为，\n确保回退策略正确。
    """
    network = LiquidNetwork({"input_dim": 3, "hidden_dim": 4})
    hidden_sequence = torch.zeros((1, 3, 4), dtype=torch.float32)
    feature_tensor = torch.tensor(
        [[[1.0, 0.0, 0.5], [0.5, 1.0, 0.0], [0.2, 0.3, 0.4]]],
        dtype=torch.float32,
    )
    missing_tensor = torch.zeros_like(feature_tensor)
    step_dts = torch.tensor([[0.20, 0.10, 0.40]], dtype=torch.float32)

    step_weights = network.compute_step_weights(
        hidden_sequence,
        feature_tensor,
        missing_tensor,
        step_dts,
    )
    expected = torch.full_like(step_dts, 1.0 / step_dts.shape[1])

    assert torch.allclose(step_weights, expected, atol=1e-6, rtol=1e-6)


def test_network_preserves_optional_readout_context_and_falls_back_to_neutral():
    """保持性测试：network。\n\n验证 network 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    network = LiquidNetwork({"input_dim": 3, "hidden_dim": 4})
    base_window = {
        "current_modality": "uwb",
        "feature_order": ["quality", "valid", "modality_gap_dt"],
        "feature_values": [0.8, 1.0, 0.12],
        "missing_mask": [0, 0, 0],
        "dt": 0.1,
        "feature_window": [[0.7, 1.0, 0.08], [0.8, 1.0, 0.12]],
        "missing_mask_window": [[0, 0, 0], [0, 0, 0]],
        "window_index_map": [0, 1],
    }
    neutral_shared = network.extract_shared_features(dict(base_window))
    assert neutral_shared["readout_context_by_name"]["state_cov_trace"] == pytest.approx(0.0)
    assert neutral_shared["readout_context_observed_by_name"]["state_cov_trace"] is False

    context_window = dict(base_window)
    context_window["readout_context_by_name"] = {
        "state_cov_trace": 5.0,
        "pos_cov": 1.5,
        "last_innovation_norm": 0.7,
        "last_gate_skip_flag": 1.0,
    }
    context_window["readout_context_observed_by_name"] = {
        "state_cov_trace": True,
        "pos_cov": True,
        "last_innovation_norm": True,
        "last_gate_skip_flag": True,
    }
    context_shared = network.extract_shared_features(context_window)
    assert context_shared["readout_context_by_name"]["state_cov_trace"] == pytest.approx(5.0)
    assert context_shared["readout_context_by_name"]["pos_cov"] == pytest.approx(1.5)
    assert context_shared["readout_context_observed_by_name"]["last_innovation_norm"] is True
    assert context_shared["readout_context_observed_by_name"]["last_gate_skip_flag"] is True


def test_network_zeroes_unobserved_readout_context_values():
    """零值测试：network。\n\n验证 network 在零值输入下的行为，\n确保边界情况正确处理。
    """
    normalized = _normalize_window_tensor(
        {
            "current_modality": "uwb",
            "feature_order": ["quality", "valid", "modality_gap_dt"],
            "feature_values": [0.8, 1.0, 0.12],
            "missing_mask": [0, 0, 0],
            "dt": 0.1,
            "feature_window": [[0.7, 1.0, 0.08], [0.8, 1.0, 0.12]],
            "missing_mask_window": [[0, 0, 0], [0, 0, 0]],
            "window_index_map": [0, 1],
            "readout_context_by_name": {
                "state_cov_trace": 7.0,
                "pos_cov": 1.5,
            },
            "readout_context_observed_by_name": {
                "state_cov_trace": False,
                "pos_cov": True,
            },
        }
    )

    assert normalized["readout_context_by_name"]["state_cov_trace"] == pytest.approx(0.0)
    assert normalized["readout_context_observed_by_name"]["state_cov_trace"] is False
    assert normalized["readout_context_by_name"]["pos_cov"] == pytest.approx(1.5)
    assert normalized["readout_context_observed_by_name"]["pos_cov"] is True


def test_network_accepts_scalar_tensor_readout_context_metadata():
    """接受测试：network。\n\n验证 network 的接受行为，\n确保合法输入被正确处理。
    """
    network = LiquidNetwork({"input_dim": 3, "hidden_dim": 4})
    context_shared = network.extract_shared_features(
        {
            "current_modality": "uwb",
            "feature_order": ["quality", "valid", "modality_gap_dt"],
            "feature_values": [torch.tensor(0.8), torch.tensor(1.0), torch.tensor(0.12)],
            "missing_mask": [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)],
            "dt": torch.tensor(0.1),
            "feature_window": [
                [torch.tensor(0.7), torch.tensor(1.0), torch.tensor(0.08)],
                [torch.tensor(0.8), torch.tensor(1.0), torch.tensor(0.12)],
            ],
            "missing_mask_window": [
                [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)],
                [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)],
            ],
            "window_index_map": [0, 1],
            "readout_context_by_name": {
                "state_cov_trace": torch.tensor(5.0),
                "pos_cov": torch.tensor(1.5),
                "last_innovation_norm": torch.tensor(0.7),
                "last_gate_skip_flag": torch.tensor(1.0),
            },
            "readout_context_observed_by_name": {
                "state_cov_trace": True,
                "pos_cov": True,
                "last_innovation_norm": True,
                "last_gate_skip_flag": True,
            },
        }
    )

    assert context_shared["readout_context_by_name"]["state_cov_trace"] == pytest.approx(5.0)
    assert context_shared["readout_context_by_name"]["pos_cov"] == pytest.approx(1.5)
    assert context_shared["current_feature_by_name"]["modality_gap_dt"] == pytest.approx(0.12)


def test_network_preserves_explicit_context_tensors_in_shared_features():
    """保持性测试：network。\n\n验证 network 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    network = LiquidNetwork({"input_dim": 3, "hidden_dim": 4})
    explicit_context_vector = torch.tensor([0.1] * 28, dtype=torch.float32)
    explicit_filter_context_vector = torch.tensor([0.2] * 8, dtype=torch.float32)

    shared = network.extract_shared_features(
        {
            "current_modality": "uwb",
            "feature_order": ["dt", "quality", "valid"],
            "feature_values": [0.1, 0.5, 1.0],
            "missing_mask": [0, 0, 0],
            "dt": 0.1,
            "feature_window": [[0.2, 0.4, 1.0], [0.1, 0.5, 1.0]],
            "missing_mask_window": [[0, 0, 0], [0, 0, 0]],
            "window_index_map": [0, 1],
            "context_vector": explicit_context_vector,
            "filter_context_vector": explicit_filter_context_vector,
        }
    )

    assert torch.equal(torch.as_tensor(shared["context_vector"]), explicit_context_vector)
    assert torch.equal(torch.as_tensor(shared["filter_context_vector"]), explicit_filter_context_vector)


def test_network_applies_configured_cell_and_pooling_settings():
    """应用测试：network。\n\n验证 network 的应用逻辑，\n确保特定条件触发预期行为。
    """
    network = LiquidNetwork(
        {
            "input_dim": 3,
            "feature_order": ["quality", "valid", "modality_gap_dt"],
            "hidden_dim": 4,
            "pooling_logit_scale": 0.5,
            "cell_update_scale_floor": 0.55,
            "cell_update_scale_span": 0.35,
            "reliability_bias_init": 1.4,
            "bad_observation_floor": 0.18,
            "bad_observation_span": 0.82,
            "bad_observation_interaction_coeff": 0.9,
        }
    )

    assert network.pooling_logit_scale == pytest.approx(0.5)
    assert network.cell is not None
    assert network.cell.update_scale_floor == pytest.approx(0.55)
    assert network.cell.update_scale_span == pytest.approx(0.35)
    assert network.cell.reliability_projection.bias.detach().cpu().item() == pytest.approx(1.4)
    assert network.cell.bad_observation_floor == pytest.approx(0.18)
    assert network.cell.bad_observation_span == pytest.approx(0.82)
    assert network.cell.bad_observation_interaction_coeff == pytest.approx(0.9)


def test_extract_shared_features_executes_documented_recurrent_pipeline_in_order(monkeypatch):
    """共享测试：extract。\n\n验证 extract 的共享合同，\n确保不同模型使用一致的输入。
    """
    network = LiquidNetwork({"input_dim": 2, "hidden_dim": 2})
    window_tensor = {
        "current_modality": "uwb",
        "feature_order": ["dt", "quality"],
        "feature_values": [0.3, 0.7],
        "missing_mask": [0, 0],
        "dt": 0.1,
        "feature_window": [[0.1, 0.5], [0.2, 0.6], [0.3, 0.7]],
        "missing_mask_window": [[0, 0], [0, 0], [0, 0]],
        "window_index_map": [0, 1, 2],
        "event_time_window": [1.0, 1.2, 1.5],
    }

    call_log: list[tuple[list[float], float]] = []

    def _fake_step(feature_row, hidden_state, *, dt, missing_mask):
        del hidden_state, missing_mask
        values = feature_row.squeeze(0)
        index = len(call_log)
        output = torch.tensor([[values[0].item() + dt, values[1].item() + index]], dtype=feature_row.dtype, device=feature_row.device)
        call_log.append((output.squeeze(0).tolist(), float(dt)))
        return output.clone(), output.clone()

    def _fake_weights(hidden_sequence, feature_tensor, missing_tensor, step_dts):
        del feature_tensor, missing_tensor
        assert hidden_sequence.shape == (1, 3, 2)
        assert step_dts.shape == (1, 3)
        assert step_dts.squeeze(0).tolist() == pytest.approx([0.1, 0.2, 0.3])
        return torch.tensor([[0.2, 0.3, 0.5]], dtype=hidden_sequence.dtype, device=hidden_sequence.device)

    monkeypatch.setattr(network.cell, "step", _fake_step)
    monkeypatch.setattr(network, "compute_step_weights", _fake_weights)

    with torch.no_grad():
        network.shared_projection.weight.zero_()
        network.shared_projection.bias.zero_()
        network.shared_projection.weight[0, 0] = 1.0
        network.shared_projection.weight[0, 2] = 1.0
        network.shared_projection.weight[1, 1] = 1.0
        network.shared_projection.weight[1, 3] = 1.0

    shared = network.extract_shared_features(window_tensor)

    expected_hidden_sequence = torch.tensor(
        [
            [0.2, 0.5],
            [0.4, 1.6],
            [0.6, 2.7],
        ],
        dtype=torch.float32,
    )
    expected_final_hidden = expected_hidden_sequence[-1]
    expected_pooled_hidden = (
        expected_hidden_sequence
        * torch.tensor([[0.2], [0.3], [0.5]], dtype=torch.float32)
    ).sum(dim=0)
    expected_shared_vector = torch.tanh(expected_final_hidden + expected_pooled_hidden)

    assert [dt for _, dt in call_log] == pytest.approx([0.1, 0.2, 0.3])
    assert torch.allclose(shared["hidden_sequence"], expected_hidden_sequence, atol=1e-6, rtol=1e-6)
    assert torch.allclose(shared["final_hidden"], expected_final_hidden, atol=1e-6, rtol=1e-6)
    assert torch.allclose(shared["pooled_hidden"], expected_pooled_hidden, atol=1e-6, rtol=1e-6)
    assert torch.allclose(shared["shared_vector"], expected_shared_vector, atol=1e-6, rtol=1e-6)


def test_network_shared_projection_matches_documented_concat_dimensions():
    """匹配测试：network shared projection。\n\n验证 network shared projection 的输出与预期一致，\n确保合同合规。
    """
    network = LiquidNetwork({"input_dim": 22, "hidden_dim": 44})

    assert network.shared_projection.in_features == 88
    assert network.shared_projection.out_features == 44


@pytest.mark.skipif(not cuda_runtime_usable(), reason="CUDA runtime unavailable")
def test_liquid_structured_window_forward_respects_module_device():
    """尊重测试：liquid structured window forward。\n\n验证被测功能尊重 liquid structured window forward 的规则，\n确保协议约束被正确执行。
    """
    network = LiquidNetwork({"input_dim": 3, "hidden_dim": 4}).cuda()
    window_tensor = {
        "current_modality": "uwb",
        "feature_order": ["quality", "valid", "modality_gap_dt"],
        "feature_values": [0.8, 1.0, 0.12],
        "missing_mask": [0, 0, 0],
        "dt": 0.1,
        "feature_window": [[0.7, 1.0, 0.08], [0.8, 1.0, 0.12]],
        "missing_mask_window": [[0, 0, 0], [0, 0, 0]],
        "window_index_map": [0, 1],
    }

    shared = network.forward(window_tensor)["shared_features"]

    assert shared["shared_vector"].device.type == "cuda"
    assert shared["hidden_sequence"].device.type == "cuda"
