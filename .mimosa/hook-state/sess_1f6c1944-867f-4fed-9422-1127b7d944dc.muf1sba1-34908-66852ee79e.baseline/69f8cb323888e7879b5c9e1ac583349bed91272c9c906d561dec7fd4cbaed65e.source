"""LSTM 网络结构（lstm_network）测试模块。

测试覆盖范围：
- LSTM 网络的前向传播
- 网络参数与配置的一致性
- 输出头的维度验证

被测模块：liquidloc.models.lstm_network"""

import pytest
import torch

from liquidloc.common.seed_utils import cuda_runtime_usable
from liquidloc.models.lstm.network import LSTMNetwork, build_lstm_context_tensor_from_normalized_window


def test_lstm_forward_from_structured_window_returns_four_heads():
    network = LSTMNetwork({"feature_order": ["quality", "valid"], "network": {"input_dim": 4, "hidden_dim": 8}})
    window_tensor = {
        "current_modality": "uwb",
        "feature_order": ["quality", "valid"],
        "feature_values": [0.6, 1.0],
        "missing_mask": [0, 0],
        "dt": 0.1,
        "feature_window": [[0.4, 1.0], [0.6, 1.0]],
        "missing_mask_window": [[0, 0], [0, 0]],
    }

    output = network.forward(window_tensor)

    assert output.shape == (4,)
    assert torch.isfinite(output).all()


def test_lstm_named_output_heads_are_parameterless_views():
    network = LSTMNetwork({"feature_order": ["quality", "valid"], "network": {"input_dim": 4, "hidden_dim": 8}})

    assert tuple(name for name, _ in network.named_parameters() if "output" in name) == (
        "output_layer.weight",
        "output_layer.bias",
    )
    assert all(tuple(head.parameters()) == () for head in network.output_heads.values())
    for index, (key, head) in enumerate(network.output_heads.items()):
        assert torch.equal(head.weight, network.output_layer.weight[index:index + 1])
        assert torch.equal(head.bias, network.output_layer.bias[index:index + 1])

    with torch.no_grad():
        network.output_layer.weight[2].fill_(0.25)
        network.output_layer.bias[2].fill_(0.75)
    assert torch.all(network.output_heads["uwb_scaling"].weight == 0.25)
    assert torch.all(network.output_heads["uwb_scaling"].bias == 0.75)


def test_lstm_named_output_head_gradient_updates_only_joint_output_layer():
    network = LSTMNetwork({"feature_order": ["quality", "valid"], "network": {"input_dim": 4, "hidden_dim": 8}})
    fused = torch.randn(3, network.hidden_dim + network.context_dim)

    output = torch.cat([head(fused) for head in network.output_heads.values()], dim=-1)
    output[:, 1].sum().backward()

    assert network.output_layer.weight.grad is not None
    assert network.output_layer.bias.grad is not None
    assert network.output_layer.weight.grad[1].abs().sum().item() > 0.0
    assert network.output_layer.weight.grad[[0, 2, 3]].abs().sum().item() == pytest.approx(0.0)
    assert network.output_layer.bias.grad[1].abs().item() > 0.0
    assert network.output_layer.bias.grad[[0, 2, 3]].abs().sum().item() == pytest.approx(0.0)


def test_lstm_load_state_dict_migrates_legacy_named_heads_and_prefers_joint_keys():
    source = LSTMNetwork({"feature_order": ["quality", "valid"], "network": {"input_dim": 4, "hidden_dim": 8}})
    with torch.no_grad():
        source.output_layer.weight.copy_(torch.arange(source.output_layer.weight.numel()).reshape_as(source.output_layer.weight))
        source.output_layer.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    canonical = {key: value.detach().clone() for key, value in source.state_dict().items()}
    legacy = {key: value for key, value in canonical.items() if not key.startswith("output_layer.")}
    for index, key in enumerate(source.output_heads):
        legacy[f"output_heads.{key}.weight"] = canonical["output_layer.weight"][index:index + 1].clone() + 100.0
        legacy[f"output_heads.{key}.bias"] = canonical["output_layer.bias"][index:index + 1].clone() + 100.0

    migrated = LSTMNetwork({"feature_order": ["quality", "valid"], "network": {"input_dim": 4, "hidden_dim": 8}})
    migrated.load_state_dict(legacy)
    assert torch.equal(migrated.output_layer.weight, canonical["output_layer.weight"] + 100.0)
    assert torch.equal(migrated.output_layer.bias, canonical["output_layer.bias"] + 100.0)

    mixed = dict(legacy)
    mixed["output_layer.weight"] = canonical["output_layer.weight"]
    mixed["output_layer.bias"] = canonical["output_layer.bias"]
    preferred = LSTMNetwork({"feature_order": ["quality", "valid"], "network": {"input_dim": 4, "hidden_dim": 8}})
    preferred.load_state_dict(mixed)
    assert torch.equal(preferred.output_layer.weight, canonical["output_layer.weight"])
    assert torch.equal(preferred.output_layer.bias, canonical["output_layer.bias"])


def test_lstm_context_tensor_accepts_scalar_tensor_metadata():
    """接受测试：lstm context tensor。\n\n验证 lstm context tensor 的接受行为，\n确保合法输入被正确处理。
    """
    context_tensor = build_lstm_context_tensor_from_normalized_window(
        {
            "current_modality": "uwb",
            "feature_order": ["quality", "valid", "uwb_range_residual"],
            "feature_values": [torch.tensor(0.6), torch.tensor(1.0), torch.tensor(0.45)],
            "missing_mask": [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)],
        },
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert context_tensor[0].item() == pytest.approx(1.0)
    assert context_tensor[1].item() == pytest.approx(0.0)
    assert torch.isfinite(context_tensor).all()


@pytest.mark.skipif(not cuda_runtime_usable(), reason="CUDA runtime unavailable")
def test_lstm_structured_window_forward_respects_module_device():
    """尊重测试：lstm structured window forward。\n\n验证被测功能尊重 lstm structured window forward 的规则，\n确保协议约束被正确执行。
    """
    network = LSTMNetwork({"feature_order": ["quality", "valid"], "network": {"input_dim": 4, "hidden_dim": 8}}).cuda()
    window_tensor = {
        "current_modality": "uwb",
        "feature_order": ["quality", "valid"],
        "feature_values": [0.6, 1.0],
        "missing_mask": [0, 0],
        "dt": 0.1,
        "feature_window": [[0.4, 1.0], [0.6, 1.0]],
        "missing_mask_window": [[0, 0], [0, 0]],
    }

    output = network.forward(window_tensor)

    assert output.device.type == "cuda"
