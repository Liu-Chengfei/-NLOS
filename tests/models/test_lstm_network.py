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
