"""Verify 3-network head structure and 4-element output."""
import sys
sys.path.insert(0, r"E:\Q4 - 副本\src")

import torch
from liquidloc.factories.model_factory import (
    LiquidOutputHead,
    LiquidOutputHeadLinear,
    _LiquidOutputHead,
)
from liquidloc.models.lstm.network import LSTMNetwork
from liquidloc.models.transformer.network import TransformerNetwork

print("=" * 60)
print("Item 14/35: Head structure verification")
print("=" * 60)

# Inspect LNN head class
lnn_head = LiquidOutputHead(key="bias", hidden_dim=18)
lnn_params = sum(p.numel() for p in lnn_head.parameters())
print(f"\nLNN LiquidOutputHead class: {type(lnn_head).__name__}")
print(f"LNN head parameter count (single head): {lnn_params}")

lnn_head_linear = LiquidOutputHeadLinear(key="bias", hidden_dim=18)
lnn_linear_params = sum(p.numel() for p in lnn_head_linear.parameters())
print(f"\nLNN LiquidOutputHeadLinear class: {type(lnn_head_linear).__name__}")
print(f"LNN head linear parameter count (single head): {lnn_linear_params}")
print(f"  - projection: {lnn_head_linear.projection.weight.numel() + lnn_head_linear.projection.bias.numel()}")
print(f"  - residual_projection: {lnn_head_linear.residual_projection.weight.numel() + lnn_head_linear.residual_projection.bias.numel()}")

# LNN uses 4 SEPARATE head instances
print("\nLNN: 4 separate LiquidOutputHeadLinear instances (one per head)")

# Inspect LSTM head
print("\n" + "-" * 60)
print("LSTM head structure:")
lstm_net = LSTMNetwork(model_cfg={
    "feature_order": ["range", "ax", "ay", "gz", "dx", "dy", "dyaw", "timestamp"],
    "network": {"input_dim": 16, "hidden_dim": 18, "num_layers": 1, "dropout": 0.0},
})
print(f"LSTM output_layer class: {type(lstm_net.output_layer).__name__}")
print(f"LSTM output_layer shape: weight={lstm_net.output_layer.weight.shape}, bias={lstm_net.output_layer.bias.shape if lstm_net.output_layer.bias is not None else None}")
print(f"  in_features: {lstm_net.output_layer.in_features}")
print(f"  out_features: {lstm_net.output_layer.out_features}")
lstm_params = sum(p.numel() for p in lstm_net.output_layer.parameters())
print(f"  LSTM total head parameters: {lstm_params}")

# Inspect Transformer head
print("\nTransformer head structure:")
trans_net = TransformerNetwork(model_cfg={
    "feature_order": ["range", "ax", "ay", "gz", "dx", "dy", "dyaw", "timestamp"],
    "network": {"input_dim": 16, "hidden_dim": 18, "num_layers": 1, "nhead": 3, "dropout": 0.1},
})
print(f"Transformer output_layer class: {type(trans_net.output_layer).__name__}")
print(f"Transformer output_layer shape: weight={trans_net.output_layer.weight.shape}, bias={trans_net.output_layer.bias.shape if trans_net.output_layer.bias is not None else None}")
print(f"  in_features: {trans_net.output_layer.in_features}")
print(f"  out_features: {trans_net.output_layer.out_features}")
trans_params = sum(p.numel() for p in trans_net.output_layer.parameters())
print(f"  Transformer total head parameters: {trans_params}")

print("\n" + "=" * 60)
print("Item 18: 4-element output verification (forward pass)")
print("=" * 60)

# Build a windowed input for both
seq_len = 5
feature_order = ["range", "ax", "ay", "gz", "dx", "dy", "dyaw", "timestamp"]
feature_dim = len(feature_order)

# Test LSTM
feature_window = torch.randn(1, seq_len, feature_dim * 2)  # LSTM batch form
modalities = ["uwb"]

with torch.no_grad():
    lstm_out = lstm_net.forward_sequence_batch(feature_window, modalities=modalities)
print(f"\nLSTM output shape: {lstm_out.shape}")
print(f"LSTM output values: {lstm_out}")

# Test Transformer
with torch.no_grad():
    trans_out = trans_net.forward_sequence_batch(feature_window, modalities=modalities)
print(f"\nTransformer output shape: {trans_out.shape}")
print(f"Transformer output values: {trans_out}")

# Test LNN
from liquidloc.factories.model_factory import _LiquidModel
print("\nLNN model:")
try:
    lnn_model = _LiquidModel(model_cfg={
        "feature_order": feature_order,
        "network": {"input_dim": feature_dim * 2, "hidden_dim": 18, "num_layers": 1},
    })
    seq_tensor = torch.randn(1, feature_dim * 2)
    lnn_out_dict = lnn_model.predict_intermediate_tensors(seq_tensor, modality="uwb")
    print(f"LNN output keys: {list(lnn_out_dict.keys())}")
    print(f"LNN output shapes: {[(k, v.shape if hasattr(v, 'shape') else v) for k, v in lnn_out_dict.items()]}")
    # Sum total head parameters (4 separate heads)
    total_lnn_head_params = sum(sum(p.numel() for p in head.parameters()) for head in lnn_model.output_heads.values())
    print(f"Total LNN head parameters (4 heads combined): {total_lnn_head_params}")
    print(f"Per-head parameter count: {total_lnn_head_params // 4}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"LNN forward test error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Comparison summary")
print("=" * 60)
print(f"LNN:    4 separate LiquidOutputHeadLinear instances, each ~{lnn_linear_params} params")
print(f"        Total LNN head params (4 heads): ~{4 * lnn_linear_params}")
print(f"LSTM:   1 single nn.Linear with {lstm_params} params total (single Linear)")
print(f"Trans:  1 single nn.Linear with {trans_params} params total (single Linear)")
print(f"\n*** VERDICT ***")
print("LNN:    4 SEPARATE heads")
print("LSTM:   SINGLE shared Linear (in=hidden+context, out=4)")
print("Trans:  SINGLE shared Linear (in=hidden+context, out=4)")
print("=> Item 14/35: NOT MET (LNN uses 4 separate heads; LSTM/Transformer use single shared Linear)")