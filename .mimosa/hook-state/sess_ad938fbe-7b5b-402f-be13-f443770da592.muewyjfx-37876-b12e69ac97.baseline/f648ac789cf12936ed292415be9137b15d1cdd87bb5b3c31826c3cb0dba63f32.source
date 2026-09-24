"""Transformer 网络与结构化窗口适配层。

这个模块把"结构化窗口"变成真正能喂给 causal Transformer 的张量，也把
Transformer 的输出头保持在固定协议里。核心设计：

- 使用 nn.TransformerEncoder + causal 掩码实现 strict causal 自注意力
  （符合 §10.2 禁止双向注意力的约束）。
- 输入: [batch, seq_len, input_dim]，序列长度固定为 window.size。
- 输出: [batch, 4] 四个输出头（bias, risk, uwb_scaling, vio_scaling）。
- 上下文向量拼接在最后一层编码器输出之后，由 output_layer 映射到四头。

上游通常来自工厂、训练脚本和推理脚本；下游则会把这里产出的网络输出接到
inference.py、trainer.py、pipelines 和更高层的融合逻辑。
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from liquidloc.common.constants import (
    CONTEXT_DIM,
    CONTEXT_FEATURE_KEYS,
    MAX_MODEL_DIM,
    MODEL_INTERMEDIATE_KEYS,
    MODALITY_UWB,
    MODALITY_VIO,
    RISK_PRIOR_LOGIT,
)
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_integer, is_real, is_string_like
from liquidloc.factories.model_factory import LiquidOutputHeadLinear  # D7 公平性：Transformer 共享 LNN 头类，参数结构一致 (Item 14/35)。

_OUTPUT_KEYS = MODEL_INTERMEDIATE_KEYS
_VALID_MODALITIES = frozenset({MODALITY_UWB, MODALITY_VIO})
_CONTEXT_FEATURE_KEYS = CONTEXT_FEATURE_KEYS
_CONTEXT_DIM = CONTEXT_DIM


def _coerce_supported_modality(value: Any, *, name: str) -> str:
    """把模态字段统一转成受支持的字符串，与 LSTM 侧行为一致。"""
    if not is_string_like(value):
        raise ValueError(f"{name} must be one of {sorted(_VALID_MODALITIES)}")
    modality = str(value).strip().lower()
    if modality not in _VALID_MODALITIES:
        raise ValueError(f"{name} must be one of {sorted(_VALID_MODALITIES)}")
    return modality


def _coerce_positive_int(
    value: Any,
    *,
    name: str,
    default: int,
) -> int:
    """把配置值转换成正整数，校验口径与 LSTM 侧对齐。"""
    if value is None:
        if not is_integer(default):
            raise TypeError(f"{name} default must be an integer, got {type(default).__name__}: {default!r}")
        if default < 1:
            raise ValueError(f"{name} default must be a positive integer, got {default!r}")
        if default > MAX_MODEL_DIM:
            raise ValueError(f"{name} default must be <= {MAX_MODEL_DIM} to avoid OOM, got {default}")
        return int(default)
    if not is_integer(value):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}: {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value!r}")
    if value > MAX_MODEL_DIM:
        raise ValueError(f"{name} must be <= {MAX_MODEL_DIM} to avoid OOM, got {value}")
    return int(value)


def resolve_transformer_network_cfg(model_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """把外部配置整理成网络内部配置，与 resolve_lstm_network_cfg 同口径。"""
    from liquidloc.common.tee_logger import print_dict
    _net_cfg = model_cfg.get("network") if isinstance(model_cfg, Mapping) else None
    net_cfg: dict[str, Any] = {}
    if isinstance(_net_cfg, Mapping):
        net_cfg = dict(_net_cfg)
    input_dim = _coerce_positive_int(net_cfg.get("input_dim"), name="network.input_dim", default=8)
    hidden_dim = _coerce_positive_int(net_cfg.get("hidden_dim") or net_cfg.get("hidden_size"), name="network.hidden_dim", default=18)
    num_layers = _coerce_positive_int(net_cfg.get("num_layers"), name="network.num_layers", default=1)
    nhead = _coerce_positive_int(net_cfg.get("nhead"), name="network.nhead", default=3)
    dropout = float(coerce_finite_scalar(net_cfg.get("dropout", 0.1), name="network.dropout", min_value=0.0, max_value=1.0))
    output_heads = list(net_cfg.get("output_heads") or ["bias", "risk", "uwb_scaling", "vio_scaling"])
    feature_order = model_cfg.get("feature_order") if isinstance(model_cfg, Mapping) else []
    if not isinstance(feature_order, (list, tuple)):
        raise TypeError("model_cfg.feature_order must be a list or tuple of strings")
    feature_dim = int(len(feature_order)) if feature_order else input_dim
    # §10.4 序列截断对等口径：pos_embedding 的最大序列长度可由 network.max_seq_len 配置；默认 20 与 window.size 历史口径一致，
    # 但允许下游训练器在遇到更长序列时按需扩容（见 TransformerNetwork._ensure_pos_embedding_capacity）。
    max_seq_len = _coerce_positive_int(net_cfg.get("max_seq_len"), name="network.max_seq_len", default=20)
    print_dict({
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "nhead": nhead,
        "dropout": dropout,
        "output_heads": output_heads,
        "feature_dim": feature_dim,
        "feature_order": list(feature_order) if feature_order else None,
        "max_seq_len": max_seq_len,
    }, "TransformerNetwork config resolved")
    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "nhead": nhead,
        "dropout": dropout,
        "output_heads": output_heads,
        "feature_dim": feature_dim,
        "feature_order": list(feature_order) if feature_order else [],
        "max_seq_len": max_seq_len,
    }


def _coerce_feature_order(  # 把特征顺序转成字符串列表。
    raw_feature_order: Any,  # 原始的 feature_order 配置。
) -> list[str]:  # 返回验证通过的特征名列表。
    """把 feature_order 统一成字符串列表，与 LSTM 侧完全对齐。"""
    if raw_feature_order is None:
        return []
    if (
        is_string_like(raw_feature_order)
        or isinstance(raw_feature_order, (bytes, bytearray))
        or isinstance(raw_feature_order, Mapping)
    ):
        raise TypeError("feature_order must be an iterable of strings.")
    try:
        feature_order = list(raw_feature_order)
    except TypeError as exc:
        raise TypeError("feature_order must be an iterable of strings.") from exc
    for index, feature_name in enumerate(feature_order):
        if not is_string_like(feature_name) or not feature_name:
            raise ValueError(f"feature_order[{index}] must be a non-empty string.")
    if len(set(feature_order)) != len(feature_order):
        raise ValueError("feature_order must not contain duplicate field names.")
    return feature_order


def _coerce_window_vector(  # 把一维数据转成窗口向量张量。
    raw_values: Any,
    *,
    name: str,
    feature_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """把一维窗口向量转成指定 dtype 的张量，并检查长度，与 LSTM 侧对齐。"""
    if raw_values is None:
        vector = torch.empty((0,), dtype=dtype)
    else:
        try:
            vector = torch.as_tensor(raw_values, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a numeric vector aligning with feature_order.") from exc
    if vector.ndim != 1 or vector.numel() != feature_dim:
        raise ValueError(f"{name} must be a 1D vector of length {feature_dim}")
    if vector.numel() > 0 and not torch.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
    return vector


def _coerce_window_matrix(  # 把二维数据转成窗口矩阵张量。
    raw_values: Any,
    *,
    name: str,
    feature_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """把二维窗口矩阵转成张量，并检查行列约束，与 LSTM 侧对齐。"""
    if raw_values is None:
        raise TypeError(f"{name} must be a 2D structured window")
    try:
        matrix = torch.as_tensor(raw_values, dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must be a 2D structured window") from exc
    if matrix.ndim != 2:
        raise TypeError(f"{name} must be a 2D structured window")
    if matrix.shape[0] < 1:
        raise ValueError(f"{name} must be non-empty")
    if int(matrix.shape[1]) != feature_dim:
        raise ValueError(f"{name} rows must align with feature_order")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
    return matrix


def normalize_structured_window(  # 把结构化窗口整理成统一格式。
    window_tensor: Any,  # 原始窗口对象。
    *,  # 下面的参数必须用关键字传入。
    expected_feature_order: list[str] | None = None,  # 可选的特征顺序强校验。
) -> dict[str, Any]:  # 返回标准化窗口字典。
    """校验并整理结构化窗口映射，保证后续输入形状一致。

    与 LSTM 侧 normalize_structured_window 完全对齐；确保 Transformer
    训练/推理走同一条结构化窗口解析路径（D7 公平性）。

    返回:
      - current_modality: str
      - feature_order: list[str]
      - feature_values: Tensor [feature_dim]
      - missing_mask: Tensor [feature_dim] (0/1)
      - dt: float
      - feature_window: Tensor [seq_len, feature_dim]
      - missing_mask_window: Tensor [seq_len, feature_dim] (0/1)
    """
    if not isinstance(window_tensor, Mapping):
        raise TypeError("window_tensor must be a structured feature window mapping.")

    feature_order = _coerce_feature_order(window_tensor.get("feature_order"))
    if expected_feature_order is not None and feature_order != list(expected_feature_order):
        raise ValueError(
            "window_tensor.feature_order must match the model feature_order; "
            f"expected {expected_feature_order}, got {feature_order}."
        )

    feature_dim = len(feature_order)
    current_modality = _coerce_supported_modality(
        window_tensor.get("current_modality"),
        name="feature_window.current_modality",
    )

    feature_values = _coerce_window_vector(
        window_tensor.get("feature_values"),
        name="feature_window.feature_values",
        feature_dim=feature_dim,
        dtype=torch.float32,
    )
    missing_mask = _coerce_window_vector(
        window_tensor.get("missing_mask"),
        name="feature_window.missing_mask",
        feature_dim=feature_dim,
        dtype=torch.float32,
    )

    dt = window_tensor.get("dt")
    if dt is None:
        raise ValueError("feature_window.dt must be provided explicitly")
    dt = coerce_finite_scalar(dt, name="feature_window.dt", min_value=0.0)

    feature_window = _coerce_window_matrix(
        window_tensor.get("feature_window"),
        name="feature_window.feature_window",
        feature_dim=feature_dim,
        dtype=torch.float32,
    )
    missing_mask_window = _coerce_window_matrix(
        window_tensor.get("missing_mask_window"),
        name="feature_window.missing_mask_window",
        feature_dim=feature_dim,
        dtype=torch.float32,
    )
    if not ((missing_mask == 0.0) | (missing_mask == 1.0)).all():
        raise ValueError("feature_window.missing_mask must contain only 0/1 values")
    if not ((missing_mask_window == 0.0) | (missing_mask_window == 1.0)).all():
        raise ValueError("feature_window.missing_mask_window must contain only 0/1 values")
    if feature_window.shape != missing_mask_window.shape:
        raise ValueError("feature_window.feature_window and missing_mask_window must have the same row count")
    if not torch.equal(feature_window[-1], feature_values):
        raise ValueError("feature_window current-step feature_values must match the last window row")
    if not torch.equal(missing_mask_window[-1], missing_mask):
        raise ValueError("feature_window current-step missing_mask must match the last window row")

    return {
        "current_modality": current_modality,
        "feature_order": feature_order,
        "feature_values": feature_values,
        "missing_mask": missing_mask,
        "dt": dt,
        "feature_window": feature_window,
        "missing_mask_window": missing_mask_window,
    }


def build_transformer_sequence_tensor(
    window_tensor: Any,
    *,
    expected_feature_order: list[str] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """把结构化窗口拼成 Transformer 序列张量，与 build_lstm_sequence_tensor 同口径。

    返回:
      sequence_tensor: shape [1, seq_len, input_dim*2]（特征+掩码拼接）
      normalized: 标准化后的窗口字典
    """
    normalized = normalize_structured_window(window_tensor, expected_feature_order=expected_feature_order)
    feature_window = normalized["feature_window"]  # shape [seq_len, feature_dim]
    missing_mask_window = normalized["missing_mask_window"]  # shape [seq_len, feature_dim]
    masked_features = feature_window * (1.0 - missing_mask_window)
    sequence_tensor = torch.cat([masked_features, missing_mask_window], dim=-1).unsqueeze(0)
    return sequence_tensor, normalized


def build_transformer_context_tensor_from_normalized_window(
    normalized_window: Mapping[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """从标准化窗口里提取上下文向量，供输出层拼接使用，与 LSTM 侧同口径。"""
    current_modality = _coerce_supported_modality(
        normalized_window.get("current_modality"),
        name="feature_window.current_modality",
    )
    modality_vector = {
        MODALITY_UWB: [1.0, 0.0],
        MODALITY_VIO: [0.0, 1.0],
    }[current_modality]
    feature_order = list(normalized_window.get("feature_order") or [])
    feature_index_by_name = {str(name): index for index, name in enumerate(feature_order)}
    feature_values = normalized_window.get("feature_values")
    missing_mask = normalized_window.get("missing_mask")
    context_values: list[float] = list(modality_vector)
    for feature_name in _CONTEXT_FEATURE_KEYS:
        feature_index = feature_index_by_name.get(feature_name)
        observed = False
        scalar_value = 0.0
        if feature_index is not None and feature_values is not None and missing_mask is not None:
            missing_value = coerce_finite_scalar(missing_mask[feature_index], name=f"missing_mask.{feature_name}")
            observed = missing_value < 0.5
            if observed:
                scalar_value = coerce_finite_scalar(feature_values[feature_index], name=f"feature_values.{feature_name}")
        context_values.append(scalar_value)
        context_values.append(1.0 if observed else 0.0)
    return torch.tensor(context_values, dtype=dtype, device=device)


def build_transformer_context_batch_from_sequence(
    sequence_tensor: Any,
    *,
    feature_order: Sequence[str],
    modalities: Sequence[str] | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """从一批序列张量里构造对应的上下文批次，与 LSTM 侧同口径。"""
    raw_sequence = torch.as_tensor(sequence_tensor, dtype=dtype, device=device)
    if raw_sequence.ndim == 2:
        raw_sequence = raw_sequence.unsqueeze(0)
    if raw_sequence.ndim != 3:
        raise ValueError("sequence_tensor must be a 2D or 3D sequence tensor")
    batch_size = int(raw_sequence.shape[0])
    feature_dim = len(list(feature_order))
    last_dim = int(raw_sequence.shape[-1])
    if last_dim == feature_dim:
        feature_values = raw_sequence[:, -1, :]
        missing_mask = torch.zeros_like(feature_values)
    elif last_dim == feature_dim * 2:
        feature_values = raw_sequence[:, -1, :feature_dim]
        missing_mask = raw_sequence[:, -1, feature_dim:]
    else:
        if feature_dim > 0:
            raise ValueError(
                "sequence_tensor last dimension must align with either feature_dim "
                "or feature_dim * 2 (features + missing mask)."
            )
        feature_values = None
        missing_mask = None
    if modalities is None:
        raise ValueError(
            "modalities must be provided for sequence batches because the Transformer context contract "
            "requires per-sample current_modality."
        )
    if len(modalities) != batch_size:
        raise ValueError("modalities must match the batch size")
    resolved_modalities = [
        _coerce_supported_modality(modality, name="modalities")
        for modality in modalities
    ]
    context_rows: list[torch.Tensor] = []
    for row_index in range(batch_size):
        normalized_window = {
            "current_modality": resolved_modalities[row_index],
            "feature_order": list(feature_order),
            "feature_values": (
                feature_values[row_index]
                if feature_values is not None
                else torch.zeros((feature_dim,), dtype=dtype, device=device)
            ),
            "missing_mask": (
                missing_mask[row_index]
                if missing_mask is not None
                else torch.ones((feature_dim,), dtype=dtype, device=device)
            ),
        }
        context_rows.append(
            build_transformer_context_tensor_from_normalized_window(
                normalized_window,
                dtype=dtype,
                device=device,
            )
        )
    return torch.stack(context_rows, dim=0)


class TransformerNetwork(nn.Module):
    """Causal Transformer-EKF 前端网络。

    保持与 LSTMNetwork 同口径的接口：
    - forward(window_tensor: Mapping) -> Tensor[4]
    - forward_sequence_batch(sequence_tensor, modalities) -> Tensor[B, 4]
    - predict_intermediate_tensors(window_tensor) -> dict[bias, risk, uwb_scaling, vio_scaling]

    关键约束：
    - strict causal：禁止双向注意力（符合 §10.2）
    - 固定序列长度（window.size）
    - 输出头顺序固定为 [bias, risk, uwb_scaling, vio_scaling]
    """

    def __init__(self, model_cfg: Mapping[str, Any] | None = None):
        super().__init__()
        resolved_cfg = resolve_transformer_network_cfg(model_cfg)
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "feature_dim": resolved_cfg.get("feature_dim"),
            "input_dim": resolved_cfg.get("input_dim"),
            "hidden_dim": resolved_cfg.get("hidden_dim"),
            "num_layers": resolved_cfg.get("num_layers"),
            "dropout": resolved_cfg.get("dropout"),
            "output_heads": resolved_cfg.get("output_heads"),
        }, "TransformerNetwork.__init__ 入口参数")
        self.feature_order = list(resolved_cfg["feature_order"])
        self.feature_dim = int(resolved_cfg["feature_dim"])
        self.input_dim = int(resolved_cfg["input_dim"])
        self.hidden_dim = int(resolved_cfg["hidden_dim"])
        self.num_layers = int(resolved_cfg["num_layers"])
        self.nhead = int(resolved_cfg.get("nhead", 3))
        self.dropout = float(resolved_cfg["dropout"])
        self.output_heads = tuple(resolved_cfg["output_heads"])
        self.context_dim = _CONTEXT_DIM
        # §10.4 序列截断对等口径：pos_embedding 大小由 network.max_seq_len 配置决定（默认 20，与历史 window.size 口径一致）。
        self.max_seq_len = int(resolved_cfg.get("max_seq_len", 20))
        # Transformer 输入维度 = 特征维度（含掩码拼接，由 build_transformer_sequence_tensor 保证）
        # 先投影到 input_dim（与 LSTM 同口径的"输入表征维度"），再投影到 hidden_dim 才能进入 encoder。
        # 实际进入 transformer encoder 的维度是 hidden_dim，所以需要第二个投影。
        self.input_proj = nn.Linear(self.feature_dim * 2, self.input_dim) if self.feature_dim * 2 != self.input_dim else None
        # hidden_proj: 把 input_dim 投到 hidden_dim，让 position embedding 的维度匹配。
        if self.input_dim != self.hidden_dim:
            self.hidden_proj = nn.Linear(self.input_dim, self.hidden_dim)
        else:
            self.hidden_proj = None
        # Position embedding: learned positional encoding，位置维度 = hidden_dim
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.hidden_dim)
        # Causal Transformer Encoder Layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,  # post-norm，稳定性更好
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=self.num_layers)
        # Item 14/35 修复：4 个独立 LiquidOutputHeadLinear（与 LNN 同类），共享 backbone 输出。
        # hidden_dim + context_dim 输入，每头投影 1 个标量（参数结构与 LNN 一致）。
        self.output_heads = nn.ModuleDict(
            {key: LiquidOutputHeadLinear(key, self.hidden_dim) for key in _OUTPUT_KEYS}
        )
        self._reset_output_layer_contract_biases()

    def _reset_output_layer_contract_biases(self) -> None:
        """Item 14/35: 初始化每个 head 的权重和偏置，委托给 LiquidOutputHeadLinear.reset_parameters()。"""
        for head in self.output_heads.values():
            head.reset_parameters()

    def reset_parameters(self) -> None:
        """重置所有可学习参数到初始状态，与 LSTMNetwork.reset_parameters 同名同口径。"""
        if self.input_proj is not None:
            self.input_proj.reset_parameters()
        if self.hidden_proj is not None:
            self.hidden_proj.reset_parameters()
        self.pos_embedding.reset_parameters()
        for layer in self.transformer_encoder.layers:
            for param in layer.parameters():
                if hasattr(param, "reset_parameters"):
                    param.reset_parameters()
        self._reset_output_layer_contract_biases()  # 每个 head 重置，保留协议偏置契约。

    def _ensure_pos_embedding_capacity(self, required_len: int, *, device: torch.device) -> None:
        """当 required_len 超过当前 pos_embedding 容量时，动态扩容并初始化新增位置编码。

        §10.4 序列截断对等口径：训练期样本可能因尾部积累事件数不足而产生短于 window.size 的序列，
        也可能因窗口截断策略调整而需要更长序列；pos_embedding 必须能覆盖任意 seq_len ≤ max_seq_len。
        """
        required_len = int(required_len)
        current_len = self.pos_embedding.num_embeddings
        if required_len <= current_len:
            return
        old_weights = self.pos_embedding.weight.data.detach().to(device=device)
        # 新建更大容量的 pos_embedding 并保持原有权重，新增位置编码用零初始化（稳定训练起始点）。
        new_pos_embedding = nn.Embedding(required_len, self.hidden_dim, device=device)
        new_pos_embedding.weight.data[:current_len] = old_weights
        self.pos_embedding = new_pos_embedding
        self.pos_embedding.requires_grad_(True)

    def _coerce_sequence_batch(self, window_tensor: Any) -> torch.Tensor:
        """把输入统一成三维序列批次张量，与 LSTMNetwork._coerce_sequence_batch 同口径。"""
        if isinstance(window_tensor, Mapping):
            sequence_tensor, _ = build_transformer_sequence_tensor(
                window_tensor,
                expected_feature_order=self.feature_order or None,
            )
            expected_last_dim = self.feature_dim * 2
            if int(sequence_tensor.shape[-1]) != expected_last_dim:
                raise ValueError(
                    f"window_tensor last dimension must be {expected_last_dim} "
                    f"(feature_dim={self.feature_dim} * 2 for features+mask); got {sequence_tensor.shape[-1]}."
                )
            return sequence_tensor
        try:
            sequence_tensor = torch.as_tensor(window_tensor, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError(
                "window_tensor must be a 2D/3D tensor or a structured feature window mapping."
            ) from exc
        if sequence_tensor.ndim == 2:
            sequence_tensor = sequence_tensor.unsqueeze(0)
        elif sequence_tensor.ndim != 3:
            raise ValueError("window_tensor must be a 2D/3D tensor or a structured feature window mapping.")
        if any(int(dim) <= 0 for dim in sequence_tensor.shape):
            raise ValueError("window_tensor must be non-empty in every dimension.")
        expected_last_dim = self.feature_dim * 2
        last_dim = int(sequence_tensor.shape[-1])
        if last_dim != expected_last_dim:
            raise ValueError(
                f"window_tensor last dimension must be {expected_last_dim}; got {last_dim}."
            )
        if not torch.isfinite(sequence_tensor).all():
            raise ValueError("window_tensor contains non-finite values (NaN or Inf)")
        return sequence_tensor

    def _prepare_inputs(self, window_tensor: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """准备序列输入和上下文输入，返回 (sequence, context)。"""
        sequence_tensor = self._coerce_sequence_batch(window_tensor)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        sequence_tensor = sequence_tensor.to(device=device, dtype=dtype)
        context_tensor = build_transformer_context_tensor_from_normalized_window(
            normalize_structured_window(window_tensor, expected_feature_order=self.feature_order or None),
            dtype=dtype,
            device=device,
        ).unsqueeze(0)
        return sequence_tensor, context_tensor

    def forward_sequence_batch(
        self,
        sequence_tensor: Any,
        *,
        modalities: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """对已经是序列批次的输入执行前向传播，与 LSTMNetwork.forward_sequence_batch 同口径。"""
        raw_sequence = self._coerce_sequence_batch(sequence_tensor)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        raw_sequence = raw_sequence.to(device=device, dtype=dtype)
        if raw_sequence.shape[-1] == self.input_dim:
            sequence_batch = raw_sequence
        else:
            sequence_batch = self.input_proj(raw_sequence)
        if self.hidden_proj is not None and sequence_batch.shape[-1] != self.hidden_dim:
            # 仅当维度仍不匹配时才投影；避免重复投影把已对齐到 hidden_dim 的张量再压一次（潜在数值漂移）。
            sequence_batch = self.hidden_proj(sequence_batch)
        batch_size = int(sequence_batch.shape[0])
        seq_len = int(sequence_batch.shape[1])
        # 位置编码：当实际 seq_len 超过当前 pos_embedding 容量时按需扩容（§10.4 序列截断对等口径，允许训练期出现更长窗口）。
        self._ensure_pos_embedding_capacity(seq_len, device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embed = self.pos_embedding(position_ids)
        x = sequence_batch + pos_embed
        # Causal mask：[batch_size*nhead, seq_len, seq_len]，供 PyTorch multi-head attention 校验
        causal_mask = self._generate_causal_mask(seq_len, self.nhead, device, batch_size)
        encoded_sequence = self.transformer_encoder(x, mask=causal_mask)
        # 取最后一个时间步的隐藏态
        last_hidden = encoded_sequence[:, -1, :]
        context_batch = build_transformer_context_batch_from_sequence(
            raw_sequence,
            feature_order=self.feature_order,
            modalities=modalities,
            dtype=dtype,
            device=device,
        )
        # Item 14/35: 4 个独立 head 共享 LiquidOutputHeadLinear 类（与 LNN 头类同构）。
        # joint_input 维度 = hidden_dim + context_dim (14)。
        # filter_context 维度 = LIQUID_FILTER_CONTEXT_DIM (28) 零张量占位（与 LNN 默认同口径）。
        joint_input = torch.cat([last_hidden, context_batch], dim=-1)  # (B, hidden_dim + context_dim)。
        filter_context_placeholder = joint_input.new_zeros((joint_input.shape[0], 28))
        head_outputs = []
        for key, head in self.output_heads.items():
            backbone_output = {
                "filter_modulated_shared": joint_input[:, :self.hidden_dim],
                "masked_branch_context": joint_input[:, self.hidden_dim:self.hidden_dim + self.context_dim],
                "masked_filter_context": filter_context_placeholder,
                "fast_source": joint_input[:, :self.hidden_dim],
                "slow_source": joint_input[:, :self.hidden_dim],
            }
            head_outputs.append(head(backbone_output))
        network_output = torch.stack(head_outputs, dim=-1)  # (B, 4)
        if network_output.shape[0] == 1:
            return network_output.squeeze(0)
        return network_output

    def forward(self, window_tensor: Any) -> torch.Tensor:
        """对结构化窗口映射执行一次完整前向传播，与 LSTMNetwork.forward 同口径。

        Item 14/35 修复：返回 4 个独立 head 标量拼接结果，按 output_heads 顺序。
        """
        if not isinstance(window_tensor, Mapping):
            raise TypeError(
                "TransformerNetwork.forward requires a structured feature window mapping (Mapping only); "
                "use forward_sequence_batch(sequence_tensor, modalities=...) for raw sequence batches."
            )
        sequence_tensor, context_tensor = self._prepare_inputs(window_tensor)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        sequence_tensor = sequence_tensor.to(device=device, dtype=dtype)
        if sequence_tensor.shape[-1] == self.input_dim:
            sequence_batch = sequence_tensor
        else:
            sequence_batch = self.input_proj(sequence_tensor)
        if sequence_batch.shape[-1] != self.hidden_dim:
            sequence_batch = self.hidden_proj(sequence_batch)
        seq_len = int(sequence_batch.shape[1])
        # 位置编码：当实际 seq_len 超过当前 pos_embedding 容量时按需扩容（§10.4 序列截断对等口径）。
        self._ensure_pos_embedding_capacity(seq_len, device=sequence_batch.device)
        position_ids = torch.arange(seq_len, device=sequence_batch.device).unsqueeze(0)
        pos_embed = self.pos_embedding(position_ids)
        x = sequence_batch + pos_embed
        # Causal mask：[batch_size*nhead, seq_len, seq_len]，供 PyTorch multi-head attention 校验
        causal_mask = self._generate_causal_mask(seq_len, self.nhead, sequence_batch.device, 1)
        encoded_sequence = self.transformer_encoder(x, mask=causal_mask)
        last_hidden = encoded_sequence[:, -1, :]
        # Item 14/35: 4 个独立 head 共享 LiquidOutputHeadLinear 类（与 LNN 头类同构）。
        joint_input = torch.cat([last_hidden, context_tensor], dim=-1)  # (1, hidden_dim + context_dim)。
        filter_context_placeholder = joint_input.new_zeros((1, 28))
        head_outputs = []
        for key, head in self.output_heads.items():
            backbone_output = {
                "filter_modulated_shared": joint_input[:, :self.hidden_dim],
                "masked_branch_context": joint_input[:, self.hidden_dim:self.hidden_dim + self.context_dim],
                "masked_filter_context": filter_context_placeholder,
                "fast_source": joint_input[:, :self.hidden_dim],
                "slow_source": joint_input[:, :self.hidden_dim],
            }
            head_outputs.append(head(backbone_output))
        network_output = torch.stack(head_outputs, dim=-1).squeeze(0)  # (4,)
        return network_output

    @staticmethod
    def _generate_causal_mask(seq_len: int, nhead: int, device: torch.device, batch_size: int = 1) -> torch.Tensor:
        """生成下三角 causal 掩码，shape [batch_size*nhead, seq_len, seq_len]，供 PyTorch ≥ 2.0 使用。
        值为 True 的位置被掩蔽（不可见），False 表示可以 attend。
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask.unsqueeze(0).expand(batch_size * nhead, -1, -1)  # [batch_size*nhead, seq_len, seq_len]
