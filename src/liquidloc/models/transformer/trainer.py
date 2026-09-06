"""Transformer 训练器模块。

【文件职责】
负责 Transformer 模型的完整训练流程，包括损失计算、优化器构建、训练循环、
验证评估、检查点保存和训练报告生成。设计目标是与 LSTM / Liquid trainer 在
监督损失、辅助损失、选样权重、校准权重等核心维度保持一致；只在前向构造
（因果 Transformer + 上下文拼接）与辅助损失路径上保持 Transformer 自身的语义。

【本文件绝对不负责】
不负责模型结构定义、推理和数据准备。

【上游依赖】
models/transformer/network.py、common/types.py、common/constants.py、
configs/models/transformer_ekf.yaml。
模型工厂通过 model_factory 参数注入，默认懒加载 factories/model_factory.py。

【下游调用者】
pipelines/train_pipeline.py、tests/models/test_transformer_trainer.py。

【核心变量定义】
- _OUTPUT_KEYS：固定输出头顺序（bias / risk / uwb_scaling / vio_scaling）。
- _ACTIVE_HEADS_BY_MODALITY：每种模态对应的激活输出头。
- _SELECTION_WEIGHTS：各输出头的选择权重。
- _BIAS_HUBER_DELTA：偏置 Huber 损失的 delta 参数。
- _GLOBAL_L2_WEIGHT：全局 L2 正则权重。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from liquidloc.common.config_utils import find_project_root
from liquidloc.common.constants import (
    MODALITY_UWB,
    MODALITY_VIO,
    BRIDGE_BIAS_MAX,
    RISK_LABEL_MODE_DEFAULT,
    UWB_SCALING_LABEL_MODE_DEFAULT,
    VIO_SCALING_LABEL_MODE_DEFAULT,
    DEVICE_AUTO,
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_CUDA_PREFIX,
)
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
from liquidloc.common.io_utils import dumps_json_text
from liquidloc.common.seed_utils import cuda_runtime_usable, set_global_seed
from liquidloc.common.types import ModelIntermediate
from liquidloc.common.validation import (
    coerce_finite_scalar,
    is_bool_like,
    is_integer,
    is_numeric,
    is_string_like,
)
from liquidloc.models.features.normalization import neutral_floor_softplus
from liquidloc.models.transformer.network import (
    build_transformer_sequence_tensor,
    normalize_structured_window,
    _OUTPUT_KEYS,
)

_ACTIVE_HEADS_BY_MODALITY = {
    MODALITY_UWB: ("bias", "risk", "uwb_scaling"),
    MODALITY_VIO: ("risk", "vio_scaling"),
}
_SELECTION_WEIGHTS = {
    "bias": 0.30,
    "risk": 0.05,
    "uwb_scaling": 0.45,
    "vio_scaling": 0.20,
}
_TAIL_SELECTION_MAX_WEIGHT = 2.0
_TAIL_SELECTION_TAIL_COEFF = 0.50
_TAIL_SELECTION_OBSERVATION_COEFF = 0.10
_SCALING_NEUTRAL_FLOOR = 1.0
_EPOCH_FIXED_PROBE_LIMIT = 8
_BATCH_DIAGNOSTIC_PROBE_LIMIT = 4
_EPOCH_ARTIFACT_BATCH_LIMIT = 1
_BIAS_HUBER_DELTA = 1.345  # 手册 §B15: Huber δ=1.345
_SCALING_LOG_EPS = 1e-6
_GLOBAL_L2_WEIGHT = 1e-4
_TRANSFORMER_CALIBRATION_WEIGHT = 0.08
_TRANSFORMER_MONO_WEIGHT = 1e-4
_TRANSFORMER_GATE_L1_WEIGHT = 1e-5
_RISK_CALIBRATION_BIN_COUNT = 10

# §13.6.2.8 cross-trainer parity declaration
# Transformer 与 LSTM 的校准 / mono 不对称性原因一致（back-end 架构不同，
# gradient 数值尺度不同），gate_l1 / global_l2 / selection / bin_count 等字段
# 严格字面对齐；详见 src/liquidloc/models/lstm/trainer.py 同名声明。
_CROSS_TRAINER_PARITY_DECLARATION: dict[str, dict[str, Any]] = {
    "calibration_weight": {
        "value": _TRANSFORMER_CALIBRATION_WEIGHT,
        "asymmetry_reason": "calibration_back_end_architecture_diff",
    },
    "mono_weight": {
        "value": _TRANSFORMER_MONO_WEIGHT,
        "asymmetry_reason": "calibration_back_end_architecture_diff",
    },
    "gate_l1_weight": {
        "value": _TRANSFORMER_GATE_L1_WEIGHT,
        "asymmetry_reason": "",
    },
    "global_l2_weight": {
        "value": _GLOBAL_L2_WEIGHT,
        "asymmetry_reason": "",
    },
    "selection_weights": {
        "value": dict(_SELECTION_WEIGHTS),
        "asymmetry_reason": "",
    },
    "risk_calibration_bin_count": {
        "value": _RISK_CALIBRATION_BIN_COUNT,
        "asymmetry_reason": "",
    },
    "scaling_neutral_floor": {
        "value": _SCALING_NEUTRAL_FLOOR,
        "asymmetry_reason": "",
    },
    "bias_huber_delta": {
        "value": _BIAS_HUBER_DELTA,
        "asymmetry_reason": "",
    },
    "scaling_log_eps": {
        "value": _SCALING_LOG_EPS,
        "asymmetry_reason": "",
    },
    "tail_selection_max_weight": {
        "value": _TAIL_SELECTION_MAX_WEIGHT,
        "asymmetry_reason": "",
    },
    "tail_selection_tail_coeff": {
        "value": _TAIL_SELECTION_TAIL_COEFF,
        "asymmetry_reason": "",
    },
    "phase_override_keys_supported": {
        "value": [
            "phase_aux_scale",
            "phase_gate_scale",
            "phase_calibration_scale",
            "phase_regularization_scale",
        ],
        "asymmetry_reason": "phase_override_only_applies_to_phase_scheduled_trainers",
    },
}


def _resolve_current_modality(window_tensor: Mapping[str, Any]) -> str:
    modality = str(window_tensor.get("current_modality", "")).strip().lower()
    if modality not in _ACTIVE_HEADS_BY_MODALITY:
        raise ValueError(
            f"current_modality must be one of {sorted(_ACTIVE_HEADS_BY_MODALITY)}, "
            f"got {modality!r}"
        )
    return modality


def _coerce_positive_sample_weight(value: Any, *, name: str) -> float:
    if not is_numeric(value):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    f = float(value)
    if f <= 0.0:
        raise ValueError(f"{name} must be positive, got {f}")
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite, got {f}")
    return f


def _clamp_unit_interval(value: Any, *, name: str) -> float:
    f = coerce_finite_scalar(value, name=name, min_value=0.0, max_value=1.0)
    return float(f)


def _build_head_mask_tensor(modalities: list[str], *, reference_tensor: torch.Tensor) -> torch.Tensor:
    batch_size = int(reference_tensor.shape[0])
    n_heads = int(reference_tensor.shape[1])
    mask = reference_tensor.new_zeros((batch_size, n_heads))
    for row_index, modality in enumerate(modalities):
        active_keys = _ACTIVE_HEADS_BY_MODALITY[modality]
        for key in active_keys:
            try:
                mask[row_index, _OUTPUT_KEYS.index(key)] = 1.0
            except ValueError:
                pass
    return mask


def _compute_transformer_overfit_audit(
    train_losses: list[float], val_losses: list[float]
) -> dict[str, Any]:
    """准则 34 Transformer 过拟合监控：与 liquid/lstm 同口径（保持 cross-trainer 一致）."""
    if not train_losses or not val_losses or len(train_losses) != len(val_losses):
        return {
            "best_epoch": -1,
            "final_gap": None,
            "consecutive_val_rise_count": 0,
            "overfit_risk": "insufficient_data",
            "early_stop_recommended": False,
            "val_train_ratio_final": None,
        }
    best_epoch = int(min(range(len(val_losses)), key=lambda i: val_losses[i]))
    final_train = float(train_losses[-1])
    final_val = float(val_losses[-1])
    val_train_ratio = final_val / max(final_train, 1e-9)
    consecutive_rise = 0
    for i in range(len(val_losses) - 1, 0, -1):
        if val_losses[i] > val_losses[i - 1] * 1.001:
            consecutive_rise += 1
        else:
            break
    if val_train_ratio > 1.5 and consecutive_rise >= 3:
        risk = "high"
    elif val_train_ratio > 1.2 and consecutive_rise >= 2:
        risk = "medium"
    elif val_train_ratio > 1.05:
        risk = "low"
    else:
        risk = "minimal"
    return {
        "best_epoch": best_epoch,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "final_gap": float(final_val - final_train),
        "val_train_ratio_final": float(val_train_ratio),
        "consecutive_val_rise_count": int(consecutive_rise),
        "overfit_risk": str(risk),
        "early_stop_recommended": bool(consecutive_rise >= 5 and val_train_ratio > 1.3),
    }


def _build_sample_weight_tensor(
    sample_weights: list[float] | None,
    *,
    reference_tensor: torch.Tensor,
) -> torch.Tensor:
    if sample_weights is None:
        return reference_tensor.new_ones((reference_tensor.shape[0],))
    if len(sample_weights) != reference_tensor.shape[0]:
        raise ValueError(
            f"sample_weights length ({len(sample_weights)}) must match "
            f"reference_tensor batch size ({reference_tensor.shape[0]})"
        )
    values = [
        coerce_finite_scalar(w, name="sample_weight")
        for w in sample_weights
    ]
    return reference_tensor.new_tensor(values)


def _project_train_outputs(raw_output_vector: Any) -> dict[str, torch.Tensor]:
    raw_tensor = torch.as_tensor(raw_output_vector, dtype=torch.float32)
    if raw_tensor.ndim == 2:
        if int(raw_tensor.shape[0]) != 1:
            raise ValueError("model output batch dimension must be 1 for trainer loss.")
        raw_tensor = raw_tensor.squeeze(0)
    raw_tensor = raw_tensor.reshape(-1)
    if raw_tensor.numel() != len(_OUTPUT_KEYS):
        raise ValueError(
            "model output must contain the fixed four heads ordered as "
            "bias, risk, uwb_scaling, vio_scaling."
        )
    if not torch.isfinite(raw_tensor).all():
        raise ValueError("model output contains non-finite values (NaN or Inf).")
    risk = raw_tensor[1]
    risk_span = BRIDGE_THRESHOLDS["risk_max"] - BRIDGE_THRESHOLDS["risk_min"]
    if risk_span != 1.0 or BRIDGE_THRESHOLDS["risk_min"] != 0.0:
        risk = risk * risk_span + BRIDGE_THRESHOLDS["risk_min"]
    return {
        "bias": torch.clamp(raw_tensor[0], min=0.0, max=BRIDGE_BIAS_MAX).reshape(()),
        "risk": risk.reshape(()),
        "uwb_scaling": neutral_floor_softplus(raw_tensor[2]).reshape(()),
        "vio_scaling": neutral_floor_softplus(raw_tensor[3]).reshape(()),
    }


def _semantic_loss_matrix(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> torch.Tensor:
    if not torch.isfinite(prediction_batch).all():
        raise ValueError("prediction_batch must be finite for semantic loss")
    if not torch.isfinite(target_batch).all():
        raise ValueError("target_batch must be finite for semantic loss")
    bias_loss = F.huber_loss(
        prediction_batch[:, 0],
        target_batch[:, 0],
        reduction="none",
        delta=bias_huber_delta,
    )
    risk_loss = (prediction_batch[:, 1] - target_batch[:, 1]).pow(2)
    predicted_scaling = prediction_batch[:, 2:] + scaling_log_eps
    target_scaling = torch.clamp_min(target_batch[:, 2:], _SCALING_NEUTRAL_FLOOR) + scaling_log_eps
    scaling_loss = (
        torch.log(predicted_scaling) - torch.log(target_scaling)
    ).pow(2)
    return torch.cat((bias_loss.unsqueeze(1), risk_loss.unsqueeze(1), scaling_loss), dim=1)


def _normalized_squared_error(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
) -> torch.Tensor:
    return (prediction_batch - target_batch).pow(2)


def _position_error_mse(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    position_scale: float = 1.0,
) -> torch.Tensor:
    predicted_pos_correction = prediction_batch[:, 0] / position_scale
    target_pos_correction = target_batch[:, 0] / position_scale
    position_mse = (predicted_pos_correction - target_pos_correction).pow(2)
    return position_mse.mean()


def _masked_supervised_loss_stats(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    modalities: list[str],
    sample_weights: list[float] | None = None,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _build_head_mask_tensor(modalities, reference_tensor=prediction_batch)
    sample_weight_tensor = _build_sample_weight_tensor(
        sample_weights,
        reference_tensor=prediction_batch,
    ).unsqueeze(1)
    semantic_loss = _semantic_loss_matrix(
        prediction_batch, target_batch,
        bias_huber_delta=bias_huber_delta,
        scaling_log_eps=scaling_log_eps,
    )
    weighted_mask = mask * sample_weight_tensor
    numerator = (semantic_loss * weighted_mask).sum()
    denominator = weighted_mask.sum()
    return numerator, denominator


def _masked_component_loss_stats(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    modalities: list[str],
    sample_weights: list[float] | None = None,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[str]]:
    mask = _build_head_mask_tensor(modalities, reference_tensor=prediction_batch)
    sample_weight_tensor = _build_sample_weight_tensor(
        sample_weights,
        reference_tensor=prediction_batch,
    ).unsqueeze(1)
    semantic_loss = _semantic_loss_matrix(
        prediction_batch, target_batch,
        bias_huber_delta=bias_huber_delta,
        scaling_log_eps=scaling_log_eps,
    )
    numerators: dict[str, torch.Tensor] = {}
    denominators: dict[str, torch.Tensor] = {}
    active_keys: list[str] = []
    for index, key in enumerate(_OUTPUT_KEYS):
        head_mask = mask[:, index : index + 1] * sample_weight_tensor
        if float(head_mask.sum().detach().item()) <= 0.0:
            continue
        head_loss = torch.where(
            head_mask > 0,
            semantic_loss[:, index : index + 1],
            torch.zeros_like(semantic_loss[:, index : index + 1]),
        )
        numerators[key] = (head_loss * head_mask).sum()
        denominators[key] = head_mask.sum()
        active_keys.append(key)
    _position_error_mse(prediction_batch, target_batch)
    return numerators, denominators, active_keys


def _risk_calibration_bin_weights(
    target_risk: torch.Tensor,
    *,
    bin_count: int = _RISK_CALIBRATION_BIN_COUNT,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> torch.Tensor:
    if target_risk.dim() != 1:
        raise ValueError(f"target_risk must be 1-D, got {target_risk.dim()}D")
    if not torch.isfinite(target_risk).all():
        raise ValueError("target_risk contains non-finite values (NaN or Inf)")
    sample_count = int(target_risk.numel())
    if sample_count <= 1:
        return torch.ones_like(target_risk)
    effective_bin_count = max(1, min(bin_count, sample_count))
    quantiles = torch.linspace(
        0.0, 1.0, steps=effective_bin_count + 1,
        device=target_risk.device, dtype=target_risk.dtype,
    )
    edges = torch.quantile(target_risk.detach(), quantiles)
    interior_edges = edges[1:-1]
    if interior_edges.numel() == 0:
        return torch.ones_like(target_risk)
    bucket_index = torch.bucketize(target_risk.detach(), interior_edges, right=True)
    bucket_count = torch.bincount(bucket_index, minlength=int(interior_edges.numel()) + 1).clamp_min(1)
    raw_weights = torch.reciprocal(bucket_count[bucket_index].to(dtype=target_risk.dtype))
    normalized_weights = raw_weights / torch.clamp_min(raw_weights.mean(), scaling_log_eps)
    return normalized_weights


def _risk_calibration_trainable(model: Any) -> bool:
    calibration = getattr(model, "risk_calibration", None)
    parameters = getattr(calibration, "parameters", None)
    if not callable(parameters):
        return False
    return any(parameter.requires_grad for parameter in parameters())


def _compute_transformer_auxiliary_loss_terms(
    model: Any,
    prediction_batch: torch.Tensor | None = None,
    target_batch: torch.Tensor | None = None,
    *,
    modalities: list[str] | None = None,
    sample_weights: list[float] | None = None,
    loss_weights: Mapping[str, float] | None = None,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> dict[str, torch.Tensor]:
    w_cal = (
        coerce_finite_scalar(loss_weights["w_cal"], name="loss_weights.w_cal", min_value=0.0)
        if loss_weights and "w_cal" in loss_weights
        else _TRANSFORMER_CALIBRATION_WEIGHT
    )
    lambda_l2 = (
        coerce_finite_scalar(loss_weights["lambda_l2"], name="loss_weights.lambda_l2", min_value=0.0)
        if loss_weights and "lambda_l2" in loss_weights
        else _GLOBAL_L2_WEIGHT
    )
    lambda_l1 = (
        coerce_finite_scalar(loss_weights["lambda_l1"], name="loss_weights.lambda_l1", min_value=0.0)
        if loss_weights and "lambda_l1" in loss_weights
        else _TRANSFORMER_GATE_L1_WEIGHT
    )
    lambda_mono = (
        coerce_finite_scalar(loss_weights["lambda_mono"], name="loss_weights.lambda_mono", min_value=0.0)
        if loss_weights and "lambda_mono" in loss_weights
        else _TRANSFORMER_MONO_WEIGHT
    )
    if prediction_batch is not None:
        zero = prediction_batch.new_zeros(())
    else:
        zero = next(model.parameters(), torch.zeros((), dtype=torch.float32)).new_zeros(())
    aux_terms = {
        "calibration": zero,
        "mono": zero,
        "reg_l2": zero,
        "gate_l1": zero,
    }
    if _risk_calibration_trainable(model):
        calibration = getattr(model, "risk_calibration", None)
        if prediction_batch is not None and target_batch is not None and modalities is not None:
            risk_mask = _build_head_mask_tensor(modalities, reference_tensor=prediction_batch)[:, 1]
            sample_weight_tensor = _build_sample_weight_tensor(
                sample_weights,
                reference_tensor=prediction_batch,
            )
            effective_weight = risk_mask * sample_weight_tensor
            observed = effective_weight > 0.0
            if bool(observed.any().item()):
                predicted_risk = prediction_batch[observed, 1]
                target_risk = target_batch[observed, 1]
                bin_weights = _risk_calibration_bin_weights(target_risk, scaling_log_eps=scaling_log_eps)
                calibration_weight = effective_weight[observed] * bin_weights
                calibration_loss = ((predicted_risk - target_risk).pow(2) * calibration_weight).sum()
                calibration_loss = calibration_loss / torch.clamp_min(calibration_weight.sum(), scaling_log_eps)
                aux_terms["calibration"] = calibration_loss * w_cal
        if hasattr(calibration, "a_raw"):
            aux_terms["mono"] = F.softplus(-calibration.a_raw).mean() * lambda_mono
    gate_penalty = zero
    for layer_name in ("input_proj", "output_layer"):
        layer = getattr(model.network, layer_name, None)
        if layer is None:
            continue
        parameters = getattr(layer, "parameters", None)
        if not callable(parameters):
            continue
        for parameter in parameters():
            if parameter.requires_grad:
                gate_penalty = gate_penalty + parameter.abs().sum()
    if float(gate_penalty.detach().item()) > 0.0:
        aux_terms["gate_l1"] = gate_penalty * lambda_l1
    l2_penalty = zero
    for parameter in model.parameters():
        if parameter.requires_grad:
            l2_penalty = l2_penalty + parameter.pow(2).sum()
    if float(l2_penalty.detach().item()) > 0.0:
        aux_terms["reg_l2"] = l2_penalty * lambda_l2
    aux_terms["total"] = (
        aux_terms["calibration"]
        + aux_terms["mono"]
        + aux_terms["reg_l2"]
        + aux_terms["gate_l1"]
    )
    return aux_terms


def _weighted_selection_score(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    selection_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    weights = selection_weights if selection_weights is not None else _SELECTION_WEIGHTS
    score = numerator.new_zeros(())
    for key_index, key in enumerate(_OUTPUT_KEYS):
        score = score + numerator[:, key_index : key_index + 1] * weights.get(key, 1.0)
    denom_sum = denominator.sum()
    if float(denom_sum.detach().item()) <= 0.0:
        return score
    return score / denom_sum


def _resolve_selection_sample_weight(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    modalities: list[str],
    observation_coeff: float = _TAIL_SELECTION_OBSERVATION_COEFF,
) -> torch.Tensor:
    mask = _build_head_mask_tensor(modalities, reference_tensor=prediction_batch)
    risk_residual = prediction_batch[:, 1] - target_batch[:, 1]
    risk_score = torch.sigmoid(risk_residual)
    observation_score = mask.any(dim=1).float()
    score = risk_score * (1.0 - observation_coeff) + observation_score * observation_coeff
    score = torch.clamp(score, min=1e-6, max=_TAIL_SELECTION_MAX_WEIGHT)
    return score


def _coerce_target_outputs(raw_target: Any) -> dict[str, float]:
    if not isinstance(raw_target, Mapping):
        raise TypeError(
            f"raw_target must be a mapping with keys {_OUTPUT_KEYS}, "
            f"got {type(raw_target).__name__}"
        )
    missing_keys = [key for key in _OUTPUT_KEYS if key not in raw_target]
    if missing_keys:
        raise ValueError(
            f"raw_target must contain keys {_OUTPUT_KEYS}; missing: {missing_keys}"
        )
    return {
        key: coerce_finite_scalar(raw_target[key], name=f"raw_target.{key}")
        for key in _OUTPUT_KEYS
    }


def _extract_window_and_target(sample: Any) -> tuple[dict[str, Any], dict[str, float]]:
    """从训练样本中分离标准化窗口和目标中间量。

    参数:
    `sample` 是训练样本映射，必须包含 window_tensor 或本身就是窗口，且必须包含 target_intermediate。

    返回值:
    返回 (标准化窗口字典, 目标字典) 元组。

    失败条件:
    样本不是映射、窗口不是映射或缺少目标时抛出异常。
    """
    if not isinstance(sample, Mapping):
        raise TypeError("train/val samples must be mappings containing a structured window")
    window_tensor = sample.get("window_tensor")
    if window_tensor is None:
        window_tensor = sample
    if not isinstance(window_tensor, Mapping):
        raise TypeError("window_tensor must be a structured feature window mapping.")
    normalized_window = normalize_structured_window(window_tensor)
    raw_target = sample.get("target_intermediate")
    if raw_target is None:
        raise ValueError("each train/val sample must contain target_intermediate")
    return normalized_window, _coerce_target_outputs(raw_target)


def _materialize_samples(
    windows: list[Any],
    *,
    tail_selection_observation_coeff: float = _TAIL_SELECTION_OBSERVATION_COEFF,
) -> list[tuple[dict[str, Any], torch.Tensor, int, float]]:
    materialized: list[tuple[dict[str, Any], torch.Tensor, int, float]] = []
    for sample in windows:
        window_tensor, target_outputs = _extract_window_and_target(sample)
        target_values = torch.tensor(
            [target_outputs[key] for key in _OUTPUT_KEYS],
            dtype=torch.float32,
        )
        # 与 LSTM trainer L824 同口径：从 feature_window 形状直接推导序列长度，避免依赖下游消费者
        # 不会写入的 `seq_len`/`window_size` 键（D9 漂移根因修复）。
        feature_window_tensor = window_tensor.get("feature_window")
        if feature_window_tensor is None:
            raise ValueError(
                "window_tensor must contain a feature_window tensor to derive seq_len"
            )
        try:
            feature_window_shape = tuple(int(dim) for dim in feature_window_tensor.shape)
        except AttributeError as exc:
            raise TypeError(
                "window_tensor.feature_window must be a tensor with a shape attribute"
            ) from exc
        if len(feature_window_shape) != 2 or feature_window_shape[0] <= 0:
            raise ValueError(
                f"window_tensor.feature_window must be a 2D tensor with non-zero rows; got shape {feature_window_shape}"
            )
        seq_len = int(feature_window_shape[0])
        sample_weight: float = 1.0
        observation_coeff = coerce_finite_scalar(
            tail_selection_observation_coeff,
            name="tail_selection_observation_coeff",
            min_value=0.0,
        )
        materialized.append((window_tensor, target_values, seq_len, sample_weight))
    return materialized


def _move_materialized_samples(
    samples: list[tuple[dict[str, Any], torch.Tensor, int, float]],
    device: torch.device,
) -> list[tuple[dict[str, Any], torch.Tensor, int, float]]:
    moved: list[tuple[dict[str, Any], torch.Tensor, int, float]] = []
    for window_tensor, target, seq_len, weight in samples:
        target = target.to(device=device, dtype=torch.float32)
        moved.append((window_tensor, target, seq_len, weight))
    return moved


def _group_samples_by_sequence_length(
    samples: list[tuple[dict[str, Any], torch.Tensor, int, float]],
) -> dict[int, list[tuple[dict[str, Any], torch.Tensor, int, float]]]:
    grouped: dict[int, list[tuple[dict[str, Any], torch.Tensor, int, float]]] = {}
    for sample in samples:
        seq_len = int(sample[2])
        grouped.setdefault(seq_len, []).append(sample)
    return {key: grouped[key] for key in sorted(grouped)}


def _resolve_feature_order(train_windows: list[Any], trainer_state: dict[str, Any]) -> list[str]:
    raw_feature_order = trainer_state["feature_order"]
    if not raw_feature_order:
        if not train_windows:
            raise ValueError("feature_order must be provided in train_cfg or in the training windows")
        window_tensor, _ = _extract_window_and_target(train_windows[0])
        raw_feature_order = window_tensor.get("feature_order")
    if (
        is_string_like(raw_feature_order)
        or isinstance(raw_feature_order, (bytes, bytearray))
        or isinstance(raw_feature_order, Mapping)
    ):
        raise TypeError("feature_order must be an iterable of feature names.")
    try:
        feature_order = list(raw_feature_order or [])
    except TypeError as exc:
        raise TypeError("feature_order must be an iterable of feature names.") from exc
    if not feature_order:
        raise ValueError("feature_order must be provided in train_cfg or in the training windows")
    for index, feature_name in enumerate(feature_order):
        if not is_string_like(feature_name) or not feature_name:
            raise ValueError(f"feature_order[{index}] must be a non-empty string.")
    if len(set(feature_order)) != len(feature_order):
        raise ValueError("feature_order must not contain duplicate field names.")
    return feature_order


def _build_optimizer(
    model: Any,
    optimizer_cfg: Mapping[str, Any],
    *,
    lr_layer: Mapping[str, Any] | None = None,
):
    optimizer_name = str(optimizer_cfg["name"]).lower()
    lr = coerce_finite_scalar(optimizer_cfg["lr"], name="optimizer_cfg.lr", min_value=0.0, inclusive=False)
    weight_decay = coerce_finite_scalar(optimizer_cfg.get("weight_decay", 0.0), name="optimizer_cfg.weight_decay", min_value=0.0)
    if lr_layer and bool(lr_layer):
        _SUPPORTED_LR_LAYER_KEYS = {"readout", "backbone", "gate_cal", "filter_context_gate", "backbone_unfreeze_ramp"}
        unsupported_keys = [key for key in lr_layer if key not in _SUPPORTED_LR_LAYER_KEYS]
        if unsupported_keys:
            raise ValueError(
                f"unsupported lr_layer keys: {unsupported_keys}; "
                f"allowed keys: {sorted(_SUPPORTED_LR_LAYER_KEYS)}"
            )
        readout_lr = coerce_finite_scalar(lr_layer.get("readout", lr), name="lr_layer.readout", min_value=0.0, inclusive=False)
        backbone_lr = coerce_finite_scalar(lr_layer.get("backbone", lr), name="lr_layer.backbone", min_value=0.0, inclusive=False)
        readout_params: list[torch.nn.Parameter] = []
        backbone_params: list[torch.nn.Parameter] = []
        readout_modules: list[torch.nn.Module] = []
        backbone_modules: list[torch.nn.Module] = []
        output_layer = getattr(getattr(model, "network", None), "output_layer", None)
        risk_calibration = getattr(model, "risk_calibration", None)
        if output_layer is not None:
            readout_modules.append(output_layer)
        if risk_calibration is not None:
            readout_modules.append(risk_calibration)
        network = getattr(model, "network", None)
        if network is not None:
            backbone_modules.append(network)
        readout_param_ids: set[int] = set()
        for module in readout_modules:
            for param in module.parameters():
                if param.requires_grad:
                    readout_params.append(param)
                    readout_param_ids.add(id(param))
        for module in backbone_modules:
            for param in module.parameters():
                if param.requires_grad and id(param) not in readout_param_ids:
                    backbone_params.append(param)
        for param in model.parameters():
            if param.requires_grad and id(param) not in readout_param_ids:
                already_in_backbone = any(id(param) == id(bp) for bp in backbone_params)
                if not already_in_backbone:
                    backbone_params.append(param)
        param_groups: list[dict[str, Any]] = []
        if readout_params:
            param_groups.append({"params": readout_params, "lr": readout_lr})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr})
        if not param_groups:
            raise ValueError("model does not expose trainable parameters")
        if optimizer_name == "adam":
            return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay, foreach=False, fused=False)
        if optimizer_name == "sgd":
            return torch.optim.SGD(param_groups, lr=lr, weight_decay=weight_decay)
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    params = list(model.parameters())
    if not params:
        raise ValueError("model does not expose trainable parameters")
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, foreach=False, fused=False)
    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer: {optimizer_name}")


def _resolve_runtime_device(device_request: str) -> tuple[torch.device, str]:
    if device_request is None or str(device_request).strip() == "":
        return torch.device("cpu"), "cpu"
    device_request = str(device_request).strip().lower()
    if device_request == DEVICE_AUTO or device_request == DEVICE_CUDA:
        if cuda_runtime_usable():
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"
    if device_request == DEVICE_CPU:
        return torch.device("cpu"), "cpu"
    if device_request.startswith(DEVICE_CUDA_PREFIX):
        if cuda_runtime_usable():
            idx = int(device_request.split(":")[-1])
            return torch.device(f"cuda:{idx}"), f"cuda:{idx}"
        return torch.device("cpu"), "cpu"
    try:
        return torch.device(device_request), device_request
    except RuntimeError as exc:
        raise ValueError(f"Unsupported runtime device request: {device_request!r}") from exc


def _resolve_amp_state(train_cfg: Mapping[str, Any], runtime_device: str) -> dict[str, Any]:
    requested = str(train_cfg.get("amp_enabled", "auto")).strip().lower()
    enabled = False
    if requested == "true" or requested == "1":
        enabled = True
    elif requested == "auto":
        enabled = runtime_device == "cuda" and cuda_runtime_usable()
    else:
        enabled = False
    dtype_name = str(train_cfg.get("amp_dtype", "bf16")).strip().lower()
    if dtype_name not in ("fp16", "bf16"):
        dtype_name = "bf16"
    return {"requested": requested, "enabled": enabled, "dtype_name": dtype_name}


def _predict_batch(
    model: Any,
    batch_or_metadata: list[dict[str, Any]] | torch.Tensor,
    *,
    modalities: list[str] | None = None,
) -> torch.Tensor:
    if torch.is_tensor(batch_or_metadata):
        sequence_batch = batch_or_metadata
        resolved_modalities = [str(modality) for modality in list(modalities or [])]
        if not resolved_modalities:
            raise ValueError("modalities must be provided for tensor batch prediction")
    else:
        metadata = list(batch_or_metadata)
        sequence_tensors: list[torch.Tensor] = []
        resolved_modalities = []
        expected_feature_order = list(getattr(model.network, "feature_order", []) or [])
        for sample_meta in metadata:
            sequence_tensor, normalized_window = build_transformer_sequence_tensor(
                sample_meta,
                expected_feature_order=expected_feature_order or None,
            )
            sequence_tensors.append(sequence_tensor.squeeze(0))
            resolved_modalities.append(_resolve_current_modality(normalized_window))
        sequence_batch = torch.stack(sequence_tensors, dim=0)
    raw_outputs = model.network.forward_sequence_batch(
        sequence_batch,
        modalities=resolved_modalities,
    )
    if raw_outputs.ndim == 1:
        raw_outputs = raw_outputs.unsqueeze(0)
    if len(resolved_modalities) != int(raw_outputs.shape[0]):
        raise ValueError(
            f"modalities count ({len(resolved_modalities)}) must match raw_outputs batch size ({int(raw_outputs.shape[0])})"
        )
    rows: list[torch.Tensor] = []
    for row_index in range(int(raw_outputs.shape[0])):
        projected = _project_train_outputs(raw_outputs[row_index])
        raw_risk = raw_outputs[row_index][1]
        if hasattr(model, "risk_calibration") and model.risk_calibration is not None:
            calibrated_risk = model.risk_calibration(raw_risk.reshape(()))
        else:
            calibrated_risk = torch.sigmoid(raw_risk.reshape(()))
        risk_span = BRIDGE_THRESHOLDS["risk_max"] - BRIDGE_THRESHOLDS["risk_min"]
        if risk_span != 1.0 or BRIDGE_THRESHOLDS["risk_min"] != 0.0:
            calibrated_risk = calibrated_risk * risk_span + BRIDGE_THRESHOLDS["risk_min"]
        projected["risk"] = calibrated_risk
        sample_modality = resolved_modalities[row_index]
        if sample_modality == MODALITY_UWB:
            projected["vio_scaling"] = projected["bias"].new_tensor(1.0)
        elif sample_modality == MODALITY_VIO:
            projected["uwb_scaling"] = projected["bias"].new_tensor(1.0)
        else:
            raise ValueError(f"unsupported modality in _predict_batch: {sample_modality!r}")
        rows.append(torch.stack([projected[key].reshape(()) for key in _OUTPUT_KEYS]))
    return torch.stack(rows, dim=0)


def _compute_loss(
    model: Any,
    window_tensor: Mapping[str, Any],
    target_outputs: Mapping[str, float],
    *,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> torch.Tensor:
    modality = _resolve_current_modality(window_tensor)
    prediction_batch = _predict_batch(model, [window_tensor])
    target_batch = prediction_batch.new_tensor(
        [[coerce_finite_scalar(target_outputs[key], name=f"target_outputs.{key}") for key in _OUTPUT_KEYS]]
    )
    numerator, denominator = _masked_supervised_loss_stats(
        prediction_batch,
        target_batch,
        modalities=[modality],
        sample_weights=None,
        bias_huber_delta=bias_huber_delta,
        scaling_log_eps=scaling_log_eps,
    )
    return numerator / denominator


def _compute_batch_loss(
    model: Any,
    target_batch: torch.Tensor,
    metadata: list[dict[str, Any]],
    *,
    sample_weights: list[float] | None = None,
    loss_weights: Mapping[str, float] | None = None,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> torch.Tensor:
    prediction_batch = _predict_batch(model, metadata)
    if target_batch.ndim != 2 or target_batch.shape[1] != len(_OUTPUT_KEYS):
        raise ValueError(
            f"target_batch must have shape (batch, {len(_OUTPUT_KEYS)}), got {tuple(target_batch.shape)}"
        )
    target_batch = target_batch.to(device=prediction_batch.device, dtype=prediction_batch.dtype)
    modalities = [_resolve_current_modality(sample_meta) for sample_meta in metadata]
    numerator, denominator = _masked_supervised_loss_stats(
        prediction_batch,
        target_batch,
        modalities=modalities,
        sample_weights=sample_weights,
        bias_huber_delta=bias_huber_delta,
        scaling_log_eps=scaling_log_eps,
    )
    supervised_loss = numerator / torch.clamp_min(denominator, scaling_log_eps)
    auxiliary_terms = _compute_transformer_auxiliary_loss_terms(
        model,
        prediction_batch,
        target_batch,
        modalities=modalities,
        sample_weights=sample_weights,
        loss_weights=loss_weights,
        scaling_log_eps=scaling_log_eps,
    )
    return supervised_loss + auxiliary_terms["total"]


def _loss_for_length_group(
    model: Any,
    samples: list[tuple[dict[str, Any], torch.Tensor, int, float]],
    *,
    loss_weights: Mapping[str, float] | None = None,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> torch.Tensor:
    if not samples:
        raise ValueError("_loss_for_length_group requires at least one sample, got empty list")
    sample_weights = [sample[3] for sample in samples]
    if not all(math.isfinite(float(weight)) for weight in sample_weights):
        raise ValueError(f"non-finite selection_weight detected in length group: {sample_weights}")
    target_batch = torch.stack([sample[1] for sample in samples], dim=0)
    return _compute_batch_loss(
        model,
        target_batch,
        [sample[0] for sample in samples],
        sample_weights=sample_weights,
        loss_weights=loss_weights,
        bias_huber_delta=bias_huber_delta,
        scaling_log_eps=scaling_log_eps,
    )


def _collect_batch_loss_diagnostics(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    modalities: list[str],
    sample_weights: list[float],
    seq_lens: list[int],
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> dict[str, Any]:
    mask = _build_head_mask_tensor(modalities, reference_tensor=prediction_batch)
    sample_weight_tensor = _build_sample_weight_tensor(
        sample_weights,
        reference_tensor=prediction_batch,
    ).unsqueeze(1)
    squared_error = _normalized_squared_error(prediction_batch, target_batch)
    semantic_loss = _semantic_loss_matrix(
        prediction_batch, target_batch,
        bias_huber_delta=bias_huber_delta,
        scaling_log_eps=scaling_log_eps,
    )
    weighted_mask = mask * sample_weight_tensor
    per_row_active_keys = [
        [
            key
            for key_index, key in enumerate(_OUTPUT_KEYS)
            if float(mask[row_index, key_index].item()) > 0.0
        ]
        for row_index in range(int(mask.shape[0]))
    ]
    component_loss_values = {
        key: [
            float(semantic_loss[row_index, key_index].detach().item())
            for row_index in range(int(semantic_loss.shape[0]))
            if float(mask[row_index, key_index].item()) > 0.0
        ]
        for key_index, key in enumerate(_OUTPUT_KEYS)
    }
    probe_row_count = min(int(prediction_batch.shape[0]), _BATCH_DIAGNOSTIC_PROBE_LIMIT)
    component_loss_sums = {
        key: float(sum(values))
        for key, values in component_loss_values.items()
    }
    component_loss_squared_sums = {
        key: float(sum(value * value for value in values))
        for key, values in component_loss_values.items()
    }
    component_loss_counts = {
        key: int(len(values))
        for key, values in component_loss_values.items()
    }
    return {
        "batch_size": int(prediction_batch.shape[0]),
        "loss_numerator": float((semantic_loss * weighted_mask).sum().detach().item()),
        "loss_denominator": float(weighted_mask.sum().detach().item()),
        "mean_loss": float(((semantic_loss * weighted_mask).sum() / weighted_mask.sum()).detach().item()) if weighted_mask.sum().item() > 0 else 0.0,
        "modalities": list(modalities[:probe_row_count]),
        "seq_lens": [int(seq_len) for seq_len in seq_lens[:probe_row_count]],
        "sample_weights": [float(weight) for weight in sample_weights[:probe_row_count]],
        "active_keys": per_row_active_keys[:probe_row_count],
        "head_mask": mask[:probe_row_count].detach().cpu().tolist(),
        "prediction": prediction_batch[:probe_row_count].detach().cpu().tolist(),
        "target": target_batch[:probe_row_count].detach().cpu().tolist(),
        "semantic_loss_by_head": (semantic_loss[:probe_row_count] * mask[:probe_row_count]).detach().cpu().tolist(),
        "squared_error_by_head": (squared_error[:probe_row_count] * mask[:probe_row_count]).detach().cpu().tolist(),
        "probe_row_count": probe_row_count,
        "truncated_row_count": max(0, int(prediction_batch.shape[0]) - probe_row_count),
        "modality_counts": {
            str(modality_key): int(sum(1 for modality in modalities if modality == modality_key))
            for modality_key in sorted(set(modalities))
        },
        "seq_len_stats": {
            "min": int(min(seq_lens)) if seq_lens else None,
            "max": int(max(seq_lens)) if seq_lens else None,
        },
        "sample_weight_stats": {
            "min": float(min(sample_weights)) if sample_weights else None,
            "max": float(max(sample_weights)) if sample_weights else None,
            "mean": (sum(float(weight) for weight in sample_weights) / float(len(sample_weights)))
            if sample_weights
            else None,
        },
        "component_loss_sums": component_loss_sums,
        "component_loss_squared_sums": component_loss_squared_sums,
        "component_loss_counts": component_loss_counts,
    }


def _summarize_component_losses(
    batches: list[dict[str, Any]],
    *,
    include_auxiliary: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for key in _OUTPUT_KEYS:
        value_count = 0
        value_sum = 0.0
        value_squared_sum = 0.0
        for batch in batches:
            counts = batch.get("component_loss_counts", {})
            sums = batch.get("component_loss_sums", {})
            squared_sums = batch.get("component_loss_squared_sums", {})
            if not isinstance(counts, Mapping) or not isinstance(sums, Mapping) or not isinstance(squared_sums, Mapping):
                continue
            key_count = int(counts.get(key, 0) or 0)
            if key_count <= 0:
                continue
            value_count += key_count
            raw_sum = float(sums.get(key, 0.0) or 0.0)
            raw_squared_sum = float(squared_sums.get(key, 0.0) or 0.0)
            if not (math.isfinite(raw_sum) and math.isfinite(raw_squared_sum)):
                continue
            value_sum += raw_sum
            value_squared_sum += raw_squared_sum
        if value_count > 0:
            mean_value = value_sum / float(value_count)
            variance = max(0.0, (value_squared_sum / float(value_count)) - (mean_value * mean_value))
            summary[key] = {"mean": mean_value, "variance": variance}
        else:
            summary[key] = {"mean": None, "variance": None}
    if include_auxiliary is not None:
        for key, value in include_auxiliary.items():
            if key in _OUTPUT_KEYS:
                raise ValueError(f"auxiliary key collides with output head: {key!r}")
            summary[key] = {"mean": float(value), "variance": 0.0}
    return summary


def _collect_epoch_loss_diagnostics(
    *,
    split: str,
    epoch_index: int,
    batches: list[dict[str, Any]],
    selection_score: float | None,
    auxiliary_summary: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    loss_numerator = math.fsum(float(batch["loss_numerator"]) for batch in batches)
    loss_denominator = math.fsum(float(batch["loss_denominator"]) for batch in batches)
    stored_batches = list(batches[:_EPOCH_ARTIFACT_BATCH_LIMIT])
    epoch_payload = {
        "split": split,
        "epoch_index": int(epoch_index),
        "mean_loss": loss_numerator / loss_denominator if loss_denominator > 0.0 else 0.0,
        "loss_numerator": loss_numerator,
        "loss_denominator": loss_denominator,
        "batch_count": len(batches),
        "stored_batch_count": len(stored_batches),
        "truncated_batch_count": max(0, len(batches) - len(stored_batches)),
        "batches": stored_batches,
    }
    if selection_score is not None:
        epoch_payload["selection_score"] = coerce_finite_scalar(selection_score, name="selection_score")
    if auxiliary_summary is not None:
        epoch_payload["auxiliary_losses"] = {
            str(key): coerce_finite_scalar(value, name=f"auxiliary_losses.{key}")
            for key, value in auxiliary_summary.items()
        }
    epoch_payload["component_loss_summary"] = _summarize_component_losses(
        batches,
        include_auxiliary=auxiliary_summary,
    )
    return epoch_payload


def _collect_epoch_prediction_target_snapshot(
    *,
    model: Any,
    split: str,
    epoch_index: int,
    cached_windows: list[tuple[dict[str, Any], torch.Tensor, int, float]],
    batch_diagnostics: list[dict[str, Any]],
    mean_loss: float,
    selection_score: float | None,
    prediction_target_summary: Mapping[str, Any],
) -> dict[str, Any]:
    sample_count = int(prediction_target_summary.get("sample_count", 0) or 0)
    active_sample_count_by_head = {
        key: int(
            (prediction_target_summary.get("active_sample_count_by_head", {}) or {}).get(key, 0) or 0
        )
        for key in _OUTPUT_KEYS
    }
    prediction_mean_active_samples = {
        key: (prediction_target_summary.get("prediction_mean_active_samples", {}) or {}).get(key)
        for key in _OUTPUT_KEYS
    }
    target_mean_active_samples = {
        key: (prediction_target_summary.get("target_mean_active_samples", {}) or {}).get(key)
        for key in _OUTPUT_KEYS
    }
    prediction_mean_all_samples = {
        key: (prediction_target_summary.get("prediction_mean_all_samples", {}) or {}).get(key)
        for key in _OUTPUT_KEYS
    }
    target_mean_all_samples = {
        key: (prediction_target_summary.get("target_mean_all_samples", {}) or {}).get(key)
        for key in _OUTPUT_KEYS
    }
    probe_batch = batch_diagnostics[0] if batch_diagnostics else {
        "modalities": [],
        "active_keys": [],
        "prediction": [],
        "target": [],
    }
    probe_batch_rows: list[dict[str, Any]] = []
    for row_index, (modality, active_keys, prediction_row, target_row) in enumerate(
        zip(
            probe_batch["modalities"],
            probe_batch["active_keys"],
            probe_batch["prediction"],
            probe_batch["target"],
            strict=True,
        )
    ):
        probe_batch_rows.append(
            {
                "row_index": row_index,
                "modality": modality,
                "active_keys": list(active_keys),
                "prediction_by_head": {
                    key: coerce_finite_scalar(
                        prediction_row[key_index], name=f"probe_batch.prediction.{key}"
                    )
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
                "target_by_head": {
                    key: coerce_finite_scalar(
                        target_row[key_index], name=f"probe_batch.target.{key}"
                    )
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
                "signed_error_by_head": {
                    key: coerce_finite_scalar(
                        prediction_row[key_index] - target_row[key_index],
                        name=f"probe_batch.signed_error.{key}",
                    )
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
            }
        )
    fixed_probe_count = min(len(cached_windows), _EPOCH_FIXED_PROBE_LIMIT)
    fixed_probe_rows = _collect_fixed_probe_rows(
        model=model,
        cached_windows=cached_windows[:fixed_probe_count],
    )
    payload = {
        "split": split,
        "epoch_index": int(epoch_index),
        "mean_loss": coerce_finite_scalar(mean_loss, name="mean_loss"),
        "selection_score": None if selection_score is None else coerce_finite_scalar(selection_score, name="selection_score"),
        "sample_count": sample_count,
        "active_sample_count_by_head": active_sample_count_by_head,
        "prediction_mean": dict(prediction_mean_active_samples),
        "target_mean": dict(target_mean_active_samples),
        "prediction_mean_active_samples": dict(prediction_mean_active_samples),
        "target_mean_active_samples": dict(target_mean_active_samples),
        "prediction_mean_all_samples": dict(prediction_mean_all_samples),
        "target_mean_all_samples": dict(target_mean_all_samples),
        "mean_scope": "fixed_split_active_sample_mean",
        "mean_scope_note": "target means can remain constant across epochs because the split is fixed and means are computed over active samples.",
        "probe_batch_modalities": list(probe_batch["modalities"]),
        "probe_batch_active_keys": [list(keys) for keys in probe_batch["active_keys"]],
        "probe_batch_prediction": probe_batch["prediction"],
        "probe_batch_target": probe_batch["target"],
        "probe_batch_rows": probe_batch_rows,
        "fixed_probe_source": "cached_split_prefix",
        "fixed_probe_sample_count": fixed_probe_count,
        "fixed_probe_rows": fixed_probe_rows,
    }
    return payload


def _collect_fixed_probe_rows(
    *,
    model: Any | None,
    cached_windows: list[tuple[dict[str, Any], torch.Tensor, int, float]],
) -> list[dict[str, Any]]:
    if not cached_windows:
        return []
    if model is None:
        raise ValueError("model is required to collect fixed probe rows")
    prediction_rows: list[list[float] | None] = [None] * len(cached_windows)
    grouped_indices: dict[int, list[int]] = {}
    for sample_index, sample in enumerate(cached_windows):
        grouped_indices.setdefault(int(sample[2]), []).append(sample_index)
    with torch.no_grad():
        for sample_indices in grouped_indices.values():
            batch_samples = [cached_windows[sample_index] for sample_index in sample_indices]
            prediction_batch = _predict_batch(model, [sample[0] for sample in batch_samples])
            for row_offset, sample_index in enumerate(sample_indices):
                prediction_rows[sample_index] = [
                    coerce_finite_scalar(
                        value,
                        name=f"fixed_probe_prediction[{sample_index}][{key_index}]",
                    )
                    for key_index, value in enumerate(prediction_batch[row_offset].detach().cpu().tolist())
                ]
    fixed_probe_rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(cached_windows):
        prediction_row = prediction_rows[sample_index]
        if prediction_row is None:
            raise RuntimeError("fixed probe prediction collection left an empty row")
        target_row = [
            coerce_finite_scalar(
                value, name=f"fixed_probe_target[{sample_index}][{key_index}]"
            )
            for key_index, value in enumerate(sample[1].detach().cpu().tolist())
        ]
        modality = _resolve_current_modality(sample[0])
        active_keys = list(_ACTIVE_HEADS_BY_MODALITY[modality])
        fixed_probe_rows.append(
            {
                "sample_index_in_split": sample_index,
                "modality": modality,
                "seq_len": int(sample[2]),
                "sample_weight": coerce_finite_scalar(
                    sample[3], name=f"fixed_probe_sample_weight[{sample_index}]"
                ),
                "active_keys": active_keys,
                "prediction_by_head": {
                    key: prediction_row[key_index]
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
                "target_by_head": {
                    key: target_row[key_index]
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
                "signed_error_by_head": {
                    key: prediction_row[key_index] - target_row[key_index]
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
            }
        )
    return fixed_probe_rows


def _evaluate_model(
    model: Any,
    val_windows: list[Any],
    *,
    epoch_index: int | None = None,
    split: str = "val",
    collect_diagnostics: bool = False,
    loss_weights: Mapping[str, float] | None = None,
    selection_weights: Mapping[str, float] | None = None,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> float | tuple[float, list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    cache_attr = "_cached_train_windows" if split == "train" else "_cached_val_windows"
    cached_windows = getattr(model, cache_attr, None)
    if cached_windows is None:
        cache_device = torch.device("cpu")
        cached_windows = _move_materialized_samples(
            _materialize_samples(
                val_windows,
                tail_selection_observation_coeff=coerce_finite_scalar(
                    getattr(model, "_tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
                    name="tail_selection_observation_coeff",
                    min_value=0.0,
                ),
            ),
            cache_device,
        )
        setattr(model, cache_attr, cached_windows)
    if split == "train":
        eval_batch_size = int(getattr(model, "_train_batch_size", 0) or len(cached_windows))
    else:
        eval_batch_size = int(getattr(model, "_eval_batch_size", 0) or len(cached_windows))
    component_numerator_sums = {key: 0.0 for key in _OUTPUT_KEYS}
    component_denominator_sums = {key: 0.0 for key in _OUTPUT_KEYS}
    loss_numerator_sum = 0.0
    loss_denominator_sum = 0.0
    batch_diagnostics: list[dict[str, Any]] = []
    sample_count = 0
    active_sample_count_by_head = {key: 0 for key in _OUTPUT_KEYS}
    prediction_sum_all_samples = {key: 0.0 for key in _OUTPUT_KEYS}
    target_sum_all_samples = {key: 0.0 for key in _OUTPUT_KEYS}
    prediction_sum_active_samples = {key: 0.0 for key in _OUTPUT_KEYS}
    target_sum_active_samples = {key: 0.0 for key in _OUTPUT_KEYS}
    accumulated_prediction_batches: list[torch.Tensor] = []
    accumulated_target_batches: list[torch.Tensor] = []
    accumulated_modalities: list[str] = []
    accumulated_sample_weights: list[float] = []
    with torch.no_grad():
        for sequence_group in _group_samples_by_sequence_length(cached_windows).values():
            for start_index in range(0, len(sequence_group), eval_batch_size):
                end_index = min(len(sequence_group), start_index + eval_batch_size)
                batch_samples = sequence_group[start_index:end_index]
                metadata = [sample[0] for sample in batch_samples]
                prediction_batch = _predict_batch(model, metadata)
                target_batch = torch.stack([sample[1] for sample in batch_samples], dim=0)
                target_batch = target_batch.to(device=prediction_batch.device, dtype=prediction_batch.dtype)
                modalities = [_resolve_current_modality(sample[0]) for sample in batch_samples]
                sample_weights = [sample[3] for sample in batch_samples]
                accumulated_prediction_batches.append(prediction_batch)
                accumulated_target_batches.append(target_batch)
                accumulated_modalities.extend(modalities)
                accumulated_sample_weights.extend(sample_weights)
                sample_count += int(prediction_batch.shape[0])
                for row_index, modality in enumerate(modalities):
                    active_keys = set(_ACTIVE_HEADS_BY_MODALITY[modality])
                    for key_index, key in enumerate(_OUTPUT_KEYS):
                        prediction_value = float(prediction_batch[row_index, key_index].detach().item())
                        target_value = float(target_batch[row_index, key_index].detach().item())
                        prediction_sum_all_samples[key] += prediction_value
                        target_sum_all_samples[key] += target_value
                        if key in active_keys:
                            active_sample_count_by_head[key] += 1
                            prediction_sum_active_samples[key] += prediction_value
                            target_sum_active_samples[key] += target_value
                numerators, denominators, active_keys = _masked_component_loss_stats(
                    prediction_batch,
                    target_batch,
                    modalities=modalities,
                    sample_weights=sample_weights,
                    bias_huber_delta=bias_huber_delta,
                    scaling_log_eps=scaling_log_eps,
                )
                for key in active_keys:
                    component_numerator_sums[key] += float(numerators[key].detach().item())
                    component_denominator_sums[key] += float(denominators[key].detach().item())
                loss_numerator, loss_denominator = _masked_supervised_loss_stats(
                    prediction_batch,
                    target_batch,
                    modalities=modalities,
                    sample_weights=sample_weights,
                    bias_huber_delta=bias_huber_delta,
                    scaling_log_eps=scaling_log_eps,
                )
                loss_numerator_sum += float(loss_numerator.detach().item())
                loss_denominator_sum += float(loss_denominator.detach().item())
                if collect_diagnostics:
                    batch_diagnostics.append(
                        _collect_batch_loss_diagnostics(
                            prediction_batch,
                            target_batch,
                            modalities=modalities,
                            sample_weights=sample_weights,
                            seq_lens=[sample[2] for sample in batch_samples],
                            bias_huber_delta=bias_huber_delta,
                            scaling_log_eps=scaling_log_eps,
                        )
                    )
    active_keys = [key for key in _OUTPUT_KEYS if component_denominator_sums[key] > 0.0]
    component_losses = {
        key: torch.tensor(
            component_numerator_sums[key] / component_denominator_sums[key],
            dtype=torch.float32,
        )
        for key in active_keys
    }
    supervised_loss = loss_numerator_sum / loss_denominator_sum if loss_denominator_sum > 0.0 else 0.0
    auxiliary_losses = {"calibration": 0.0, "mono": 0.0, "reg_l2": 0.0, "gate_l1": 0.0, "total": 0.0}
    if sample_count > 0:
        all_prediction_batch = torch.cat(accumulated_prediction_batches, dim=0)
        all_target_batch = torch.cat(accumulated_target_batches, dim=0)
        auxiliary_terms = _compute_transformer_auxiliary_loss_terms(
            model,
            all_prediction_batch,
            all_target_batch,
            modalities=accumulated_modalities,
            sample_weights=accumulated_sample_weights,
            loss_weights=loss_weights,
            scaling_log_eps=scaling_log_eps,
        )
        auxiliary_losses = {
            key: float(auxiliary_terms[key].detach().item())
            for key in ("calibration", "mono", "reg_l2", "gate_l1", "total")
        }
    selection_loss = supervised_loss + auxiliary_losses["total"]
    selection_score = float(selection_loss)
    model._selection_score = selection_score
    model._selection_loss = selection_loss
    if not collect_diagnostics:
        return selection_loss
    prediction_target_summary = {
        "sample_count": sample_count,
        "active_sample_count_by_head": dict(active_sample_count_by_head),
        "prediction_mean_active_samples": {
            key: (
                prediction_sum_active_samples[key] / float(active_sample_count_by_head[key])
                if active_sample_count_by_head[key] > 0
                else None
            )
            for key in _OUTPUT_KEYS
        },
        "target_mean_active_samples": {
            key: (
                target_sum_active_samples[key] / float(active_sample_count_by_head[key])
                if active_sample_count_by_head[key] > 0
                else None
            )
            for key in _OUTPUT_KEYS
        },
        "prediction_mean_all_samples": {
            key: (prediction_sum_all_samples[key] / float(sample_count))
            if sample_count > 0
            else None
            for key in _OUTPUT_KEYS
        },
        "target_mean_all_samples": {
            key: (target_sum_all_samples[key] / float(sample_count))
            if sample_count > 0
            else None
            for key in _OUTPUT_KEYS
        },
    }
    for batch_payload in batch_diagnostics:
        batch_payload["auxiliary_losses"] = dict(auxiliary_losses)
    epoch_payload = _collect_epoch_prediction_target_snapshot(
        model=model,
        split=split,
        epoch_index=int(epoch_index or 0),
        cached_windows=cached_windows,
        batch_diagnostics=batch_diagnostics,
        mean_loss=selection_loss,
        selection_score=selection_score,
        prediction_target_summary=prediction_target_summary,
    )
    epoch_payload["auxiliary_losses"] = dict(auxiliary_losses)
    return selection_loss, batch_diagnostics, epoch_payload


def _to_cpu_recursive(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, Mapping):
        return {key: _to_cpu_recursive(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)([_to_cpu_recursive(item) for item in obj])
    return obj


def _build_checkpoint_payload(
    *,
    model: Any,
    trainer_state: dict[str, Any],
    optimizer: Any,
    seed_report: dict[str, Any],
    best_epoch: int | None,
    best_loss: float | None,
    train_window_count: int,
    val_window_count: int,
) -> dict[str, Any]:
    return {
        "model_name": trainer_state["model_name"],
        "model_state": model.state_dict(),
        "model_cfg": {
            "feature_order": list(trainer_state.get("feature_order", [])),
            "window": dict(trainer_state.get("window", {})),
            "network": dict(trainer_state.get("network", {})),
        },
        "optimizer_state": optimizer.state_dict(),
        "seed_report": seed_report,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "train_window_count": train_window_count,
        "val_window_count": val_window_count,
        "checkpoint_format": "transformer_real_v1",
    }


class TransformerTrainer:
    """Transformer 训练器，封装训练配置读取和训练流程调用。

    从 train_cfg 的 train 子节读取 loss_weights、lr_layer、bias_huber_delta、
    scaling_log_eps、gradient_clip_max_norm 等配置，存为实例属性。
    对外暴露的方法使用实例属性替代模块级常量，确保训练流程受 YAML 配置驱动。
    """

    def __init__(self, train_section: Mapping[str, Any]) -> None:
        loss_weights_cfg = train_section.get("loss_weights") or {}
        self._loss_weights: dict[str, float] = {
            "w_cal": coerce_finite_scalar(
                loss_weights_cfg.get("w_cal", _TRANSFORMER_CALIBRATION_WEIGHT),
                name="loss_weights.w_cal", min_value=0.0,
            ),
            "lambda_l2": coerce_finite_scalar(
                loss_weights_cfg.get("lambda_l2", _GLOBAL_L2_WEIGHT),
                name="loss_weights.lambda_l2", min_value=0.0,
            ),
            "lambda_l1": coerce_finite_scalar(
                loss_weights_cfg.get("lambda_l1", _TRANSFORMER_GATE_L1_WEIGHT),
                name="loss_weights.lambda_l1", min_value=0.0,
            ),
            "lambda_mono": coerce_finite_scalar(
                loss_weights_cfg.get("lambda_mono", _TRANSFORMER_MONO_WEIGHT),
                name="loss_weights.lambda_mono", min_value=0.0,
            ),
        }
        self._selection_weights: dict[str, float] = dict(_SELECTION_WEIGHTS)
        _raw_lr_layer = train_section.get("lr_layer") or {}
        self._lr_layer: dict[str, Any] = {
            key: value
            for key, value in dict(_raw_lr_layer).items()
            if key in {"readout", "backbone"}
        }
        self._bias_huber_delta: float = coerce_finite_scalar(
            train_section.get("bias_huber_delta", _BIAS_HUBER_DELTA),
            name="bias_huber_delta",
            min_value=0.0,
        )
        self._scaling_log_eps: float = coerce_finite_scalar(
            train_section.get("scaling_log_eps", _SCALING_LOG_EPS),
            name="scaling_log_eps",
            min_value=0.0,
        )
        self._gradient_clip_max_norm: float = coerce_finite_scalar(
            train_section.get("gradient_clip_max_norm", 1.0),
            name="gradient_clip_max_norm",
            min_value=0.0,
            inclusive=False,
        )
        self._tail_selection_observation_coeff: float = coerce_finite_scalar(
            train_section.get("tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
            name="tail_selection_observation_coeff",
            min_value=0.0,
        )

    def _evaluate_model(
        self,
        model: Any,
        val_windows: list[Any],
        *,
        epoch_index: int | None = None,
        split: str = "val",
        collect_diagnostics: bool = False,
    ) -> float | tuple[float, list[dict[str, Any]], dict[str, Any]]:
        return _evaluate_model(
            model,
            val_windows,
            epoch_index=epoch_index,
            split=split,
            collect_diagnostics=collect_diagnostics,
            loss_weights=self._loss_weights,
            selection_weights=self._selection_weights,
            bias_huber_delta=self._bias_huber_delta,
            scaling_log_eps=self._scaling_log_eps,
        )

    def _build_optimizer(self, model: Any, optimizer_cfg: Mapping[str, Any]):
        return _build_optimizer(model, optimizer_cfg, lr_layer=self._lr_layer)

    def train_one_epoch(
        self,
        train_windows,
        val_windows,
        epoch_index: int,
        *,
        model=None,
        optimizer=None,
    ) -> float:
        return train_one_epoch(
            train_windows,
            val_windows,
            epoch_index,
            model=model,
            optimizer=optimizer,
            gradient_clip_max_norm=self._gradient_clip_max_norm,
            loss_weights=self._loss_weights,
            bias_huber_delta=self._bias_huber_delta,
            scaling_log_eps=self._scaling_log_eps,
        )


def build_trainer_state(train_cfg: dict) -> dict:
    """从训练配置字典构建训练器状态。"""
    from liquidloc.common.tee_logger import print_dict
    _train_section = train_cfg.get("train", {}) if isinstance(train_cfg, dict) else {}
    print_dict({
        "name": train_cfg.get("name") if isinstance(train_cfg, dict) else None,
        "optimizer": _train_section.get("optimizer"),
        "lr": _train_section.get("lr"),
        "epochs": _train_section.get("epochs"),
        "patience": _train_section.get("patience"),
        "device": train_cfg.get("device") if isinstance(train_cfg, dict) else None,
        "seed": train_cfg.get("seed") if isinstance(train_cfg, dict) else None,
        "deterministic": train_cfg.get("deterministic") if isinstance(train_cfg, dict) else None,
    }, "Transformer build_trainer_state 入口参数")
    if not isinstance(train_cfg, dict):
        raise TypeError("train_cfg must be a dictionary.")
    train_section = dict(train_cfg.get("train") or {})
    optimizer_name = train_section.get("optimizer", "adam")
    lr = train_section.get("lr", 0.001)
    weight_decay = train_section.get("weight_decay", 0.0)
    epochs = train_section.get("epochs", 1)
    patience = train_section.get("patience", epochs)
    epoch_candidate_stride = train_section.get("epoch_candidate_stride", 1)
    seed = train_cfg.get("seed", train_cfg.get("seed", 0))
    deterministic = train_section.get("deterministic", train_cfg.get("deterministic", True))
    if not is_string_like(optimizer_name) or not str(optimizer_name).strip():
        raise ValueError("train.optimizer must be a non-empty string.")
    if not is_numeric(lr):
        raise TypeError("train.lr must be numeric.")
    if not is_integer(epochs):
        raise TypeError("train.epochs must be an integer.")
    if epochs < 1:
        raise ValueError("train.epochs must be at least 1.")
    if not is_integer(patience):
        raise TypeError("train.patience must be an integer.")
    if patience < 1:
        raise ValueError("train.patience must be at least 1.")
    device = train_cfg.get("device", DEVICE_CPU)
    output_root = train_cfg.get("output_root")
    if output_root is None:
        project_root = find_project_root()
        output_root = project_root / "outputs" / "transformer_ekf"
    elif isinstance(output_root, str) and not output_root.strip():
        raise ValueError("train.output_root must be a non-empty string.")
    elif not isinstance(output_root, (str, Path)):
        raise TypeError(f"train.output_root must be a str or Path, got {type(output_root).__name__}")
    output_root = Path(output_root)
    return {
        "model_name": str(train_cfg.get("name", "transformer_ekf")),
        "train": train_section,
        "optimizer": {
            "name": optimizer_name,
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        },
        "epochs": int(epochs),
        "patience": int(patience),
        "epoch_candidate_stride": int(epoch_candidate_stride),
        "device": str(device),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "output_root": str(output_root),
        "network": dict(train_cfg.get("network", {}) or {}),
        "window": dict(train_cfg.get("window", {}) or {}),
        "feature_order": list(train_cfg.get("feature_order") or []),
        "save_epoch_candidates": bool(train_section.get("save_epoch_candidates", True)),
        "lr_scheduler": dict(train_section.get("lr_scheduler", {}) or {}),
        "tail_selection_observation_coeff": float(
            train_section.get("tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF)
        ),
        "_trainer_type": "transformer",
    }


def train_one_epoch(
    train_windows,
    val_windows,
    epoch_index: int,
    *,
    model=None,
    optimizer=None,
    gradient_clip_max_norm: float = 1.0,
    loss_weights: Mapping[str, float] | None = None,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> float:
    if not is_integer(epoch_index):
        raise TypeError("epoch_index must be an integer.")
    if epoch_index < 1:
        raise ValueError("epoch_index must be at least 1.")
    if model is None:
        raise ValueError("model must be provided for real Transformer training")
    if optimizer is None:
        raise ValueError("optimizer must be provided for real Transformer training")
    train_count = len(train_windows)
    val_count = len(val_windows)
    if train_count < 1:
        raise ValueError("train_windows must contain at least one window.")
    if val_count < 1:
        raise ValueError("val_windows must contain at least one window.")
    batch_size = int(getattr(model, "_train_batch_size", 0) or 0)
    if batch_size < 1:
        batch_size = train_count
    model.train()
    cached_materialized = getattr(model, "_cached_train_windows", None)
    if not cached_materialized:
        cache_device = torch.device("cpu")
        cached_materialized = _move_materialized_samples(
            _materialize_samples(
                train_windows,
                tail_selection_observation_coeff=coerce_finite_scalar(
                    getattr(model, "_tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
                    name="tail_selection_observation_coeff",
                    min_value=0.0,
                ),
            ),
            cache_device,
        )
        setattr(model, "_cached_train_windows", cached_materialized)
    loss_sum = 0.0
    loss_count = 0
    for sequence_group in _group_samples_by_sequence_length(cached_materialized).values():
        for start_index in range(0, len(sequence_group), batch_size):
            end_index = min(len(sequence_group), start_index + batch_size)
            batch_samples = sequence_group[start_index:end_index]
            optimizer.zero_grad(set_to_none=True)
            loss_value = _loss_for_length_group(
                model,
                batch_samples,
                loss_weights=loss_weights,
                bias_huber_delta=bias_huber_delta,
                scaling_log_eps=scaling_log_eps,
            )
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
            optimizer.step()
            loss_sum += float(loss_value.detach().item())
            loss_count += 1
    return loss_sum / float(loss_count)


def train_model(train_windows, val_windows, train_cfg, *, model_factory=None):
    """Transformer 训练入口：完成从配置到训练再到落盘的完整流程。

    参数:
    `train_windows` 是训练窗口列表。
    `val_windows` 是验证窗口列表。
    `train_cfg` 是训练配置字典。
    `model_factory` 是可选的模型工厂可调用对象。

    返回值:
    返回 (最佳检查点路径, 训练报告字典) 元组。
    """
    from liquidloc.common.tee_logger import print_dict
    if train_windows is None:
        raise ValueError("train_windows must not be None.")
    if val_windows is None:
        raise ValueError("val_windows must not be None.")
    try:
        train_windows = list(train_windows)
    except TypeError as exc:
        raise TypeError("train_windows must be an iterable of windows.") from exc
    try:
        val_windows = list(val_windows)
    except TypeError as exc:
        raise TypeError("val_windows must be an iterable of windows.") from exc
    if not train_windows:
        raise ValueError("train_windows must contain at least one window.")
    if not val_windows:
        raise ValueError("val_windows must contain at least one window.")
    _train_section = train_cfg.get("train", {}) if isinstance(train_cfg, dict) else {}
    print_dict({
        "name": train_cfg.get("name") if isinstance(train_cfg, dict) else None,
        "device": train_cfg.get("device") if isinstance(train_cfg, dict) else None,
        "output_root": str(train_cfg.get("output_root")) if isinstance(train_cfg, dict) and train_cfg.get("output_root") else None,
        "optimizer": _train_section.get("optimizer"),
        "lr": _train_section.get("lr"),
        "epochs": _train_section.get("epochs"),
        "patience": _train_section.get("patience"),
        "batch_size": _train_section.get("batch_size"),
        "feature_order_len": len(train_cfg.get("feature_order", [])) if isinstance(train_cfg, dict) else None,
        "window": train_cfg.get("window") if isinstance(train_cfg, dict) else None,
        "train_windows_count": len(train_windows) if isinstance(train_windows, (list, tuple)) else None,
        "val_windows_count": len(val_windows) if isinstance(val_windows, (list, tuple)) else None,
    }, "Transformer train_model 入口参数")

    trainer_state = build_trainer_state(train_cfg)
    trainer_state["feature_order"] = _resolve_feature_order(train_windows, trainer_state)
    seed_report = set_global_seed(
        trainer_state["seed"],
        deterministic=trainer_state["deterministic"],
    )
    model_cfg = {
        "feature_order": list(trainer_state["feature_order"]),
        "window": dict(trainer_state["window"]),
        "network": dict(trainer_state["network"]),
    }
    if model_factory is None:
        from liquidloc.factories.model_factory import create_model as _default_create_model
        model_factory = _default_create_model
    model = model_factory(trainer_state["model_name"], model_cfg)
    # 准则 27：Transformer 模型加载后挂载激活统计 hook，训练过程持续收集 5 头统计。
    activation_collector = None
    activation_hook_handles: list[Any] = []
    if bool(getattr(trainer_state["train"], "collect_activation_stats", False)) or bool(
        trainer_state.get("collect_activation_stats", False)
    ):
        from liquidloc.models.activation_stats import (
            ActivationStatsCollector,
            attach_activation_stats_hooks,
        )
        activation_collector = ActivationStatsCollector()
        activation_hook_handles = attach_activation_stats_hooks(model, activation_collector)
    transformer_trainer = TransformerTrainer(trainer_state["train"])
    optimizer = transformer_trainer._build_optimizer(model, trainer_state["optimizer"])
    device, runtime_device = _resolve_runtime_device(trainer_state["device"])
    model.network.to(device)
    if hasattr(model, "risk_calibration") and hasattr(model.risk_calibration, "to"):
        model.risk_calibration.to(device)
    model._train_batch_size = int(trainer_state["train"].get("batch_size") or len(train_windows))
    model._eval_batch_size = int(trainer_state["train"].get("eval_batch_size") or len(val_windows))
    model._tail_selection_observation_coeff = coerce_finite_scalar(
        trainer_state.get("tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
        name="tail_selection_observation_coeff",
        min_value=0.0,
    )
    cache_device = torch.device("cpu")
    model._cached_train_windows = _move_materialized_samples(
        _materialize_samples(
            train_windows,
            tail_selection_observation_coeff=model._tail_selection_observation_coeff,
        ),
        cache_device,
    )
    model._cached_val_windows = _move_materialized_samples(
        _materialize_samples(
            val_windows,
            tail_selection_observation_coeff=model._tail_selection_observation_coeff,
        ),
        cache_device,
    )
    amp_state = _resolve_amp_state(trainer_state["train"], runtime_device)

    output_root = Path(trainer_state["output_root"])
    checkpoints_dir = output_root / "checkpoints"
    reports_dir = output_root / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_epoch_losses: list[float] = []
    val_epoch_losses: list[float] = []
    val_selection_scores: list[float] = []
    diagnostics_payload = {
        "model_name": trainer_state["model_name"],
        "checkpoint_format": "transformer_real_v1",
        "train": [],
        "val": [],
    }
    epoch_predictions_payload = {
        "model_name": trainer_state["model_name"],
        "checkpoint_format": "transformer_real_v1",
        "output_heads": list(_OUTPUT_KEYS),
        "train": [],
        "val": [],
    }
    best_loss: float | None = None
    best_selection_score: float | None = None
    best_epoch: int | None = None
    best_ckpt = checkpoints_dir / f'{trainer_state["model_name"]}_best_checkpoint.pt'
    epoch_candidate_paths: list[str] = []
    epoch_candidate_epochs: list[int] = []
    save_epoch_candidates = bool(trainer_state["train"].get("save_epoch_candidates"))

    lr_scheduler_state = trainer_state.get("lr_scheduler", {})
    lr_scheduler_enabled = bool(lr_scheduler_state.get("enabled", False))
    lr_scheduler = None
    if lr_scheduler_enabled:
        lr_scheduler_t_max = int(lr_scheduler_state.get("T_max", trainer_state["epochs"]))
        lr_scheduler_eta_min = float(lr_scheduler_state.get("eta_min", 1e-6))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=lr_scheduler_t_max,
            eta_min=lr_scheduler_eta_min,
        )
        print(
            f"[Transformer训练][lr_scheduler] enabled CosineAnnealingLR "
            f"T_max={lr_scheduler_t_max} "
            f"eta_min={lr_scheduler_eta_min}",
            flush=True,
        )

    for epoch_index in range(1, trainer_state["epochs"] + 1):
        train_loss = transformer_trainer.train_one_epoch(
            train_windows,
            val_windows,
            epoch_index,
            model=model,
            optimizer=optimizer,
        )
        train_eval_loss, train_batches, train_epoch_snapshot = transformer_trainer._evaluate_model(
            model,
            train_windows,
            epoch_index=epoch_index,
            split="train",
            collect_diagnostics=True,
        )
        val_loss, val_batches, val_epoch_snapshot = transformer_trainer._evaluate_model(
            model,
            val_windows,
            epoch_index=epoch_index,
            split="val",
            collect_diagnostics=True,
        )
        selection_score = float(getattr(model, "_selection_score", val_loss))
        train_epoch_losses.append(float(train_loss))
        val_epoch_losses.append(float(val_loss))
        val_selection_scores.append(selection_score)
        diagnostics_payload["train"].append(
            _collect_epoch_loss_diagnostics(
                split="train",
                epoch_index=epoch_index,
                batches=train_batches,
                selection_score=None,
                auxiliary_summary=train_epoch_snapshot.get("auxiliary_losses"),
            )
        )
        diagnostics_payload["val"].append(
            _collect_epoch_loss_diagnostics(
                split="val",
                epoch_index=epoch_index,
                batches=val_batches,
                selection_score=selection_score,
                auxiliary_summary=val_epoch_snapshot.get("auxiliary_losses"),
            )
        )
        epoch_predictions_payload["train"].append(train_epoch_snapshot)
        epoch_predictions_payload["val"].append(val_epoch_snapshot)

        checkpoint_payload = _build_checkpoint_payload(
            model=model,
            trainer_state=trainer_state,
            optimizer=optimizer,
            seed_report=seed_report,
            best_epoch=epoch_index,
            best_loss=float(val_loss),
            train_window_count=len(train_windows),
            val_window_count=len(val_windows),
        )
        should_save_candidate = False
        if save_epoch_candidates:
            if epoch_index == 1 or epoch_index == trainer_state["epochs"]:
                should_save_candidate = True
            elif epoch_index % trainer_state["epoch_candidate_stride"] == 0:
                should_save_candidate = True
        if should_save_candidate:
            candidate_path = checkpoints_dir / f'{trainer_state["model_name"]}_epoch_{epoch_index:03d}.pt'
            torch.save(checkpoint_payload, candidate_path)
            epoch_candidate_paths.append(str(candidate_path))
            epoch_candidate_epochs.append(epoch_index)

        print(
            f"[Transformer训练] 第 {epoch_index}/{trainer_state['epochs']} 轮  "
            f"训练损失={train_loss:.6f}  验证损失={val_loss:.6f}  "
            f"选择分数={selection_score:.6f}",
            flush=True,
        )
        for split_name, snapshot in [("train", train_epoch_snapshot), ("val", val_epoch_snapshot)]:
            pred_mean = snapshot.get("prediction_mean_active_samples", {})
            tgt_mean = snapshot.get("target_mean_active_samples", {})
            split_label = "训练集" if split_name == "train" else "验证集"
            parts = [
                f"{k}: 预测={pred_mean.get(k):.4f} 真值={tgt_mean.get(k):.4f}"
                for k in _OUTPUT_KEYS
                if pred_mean.get(k) is not None
            ]
            print(f"  [{split_label}] " + " | ".join(parts), flush=True)

        if best_selection_score is None or selection_score < best_selection_score:
            best_selection_score = selection_score
            best_loss = float(val_loss)
            best_epoch = epoch_index
            checkpoint_payload["best_epoch"] = best_epoch
            checkpoint_payload["best_loss"] = best_loss
            torch.save(checkpoint_payload, best_ckpt)

        if lr_scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step()
            new_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Transformer训练][lr_scheduler] ep{epoch_index} lr 退火 "
                f"{current_lr:.2e} → {new_lr:.2e} (cos T_max={lr_scheduler.T_max} eta_min={lr_scheduler.eta_min:.2e})",
                flush=True,
            )

    loss_diagnostics_path = reports_dir / f'{trainer_state["model_name"]}_loss_diagnostics.json'
    epoch_predictions_vs_targets_path = reports_dir / f'{trainer_state["model_name"]}_epoch_predictions_vs_targets.json'
    finite_train_losses = all(math.isfinite(float(loss)) for loss in train_epoch_losses)
    finite_val_losses = all(math.isfinite(float(loss)) for loss in val_epoch_losses)
    finite_selection_scores = all(math.isfinite(float(score)) for score in val_selection_scores)
    training_stability_audit = {
        "status": "ok" if (finite_train_losses and finite_val_losses and finite_selection_scores) else "diverged",
        "finite_train_losses": finite_train_losses,
        "finite_val_losses": finite_val_losses,
        "finite_selection_scores": finite_selection_scores,
        "best_epoch_in_range": bool(
            best_epoch is not None and 1 <= int(best_epoch) <= int(trainer_state["epochs"])
        ),
        "best_epoch_matches_best_selection_score": bool(
            best_epoch is not None
            and best_selection_score is not None
            and int(best_epoch)
            == int(
                min(
                    range(1, len(val_selection_scores) + 1),
                    key=lambda idx: val_selection_scores[idx - 1],
                )
            )
        ),
        "checkpoint_path_exists": bool(best_ckpt.is_file()),
        "early_stop_patience_matches_config": bool(
            int(trainer_state["patience"]) == int(trainer_state["epochs"])
        ),
        "early_stopped": False,
    }
    loss_diagnostics_path.write_text(
        dumps_json_text(diagnostics_payload),
        encoding="utf-8",
    )
    epoch_predictions_vs_targets_path.write_text(
        dumps_json_text(epoch_predictions_payload),
        encoding="utf-8",
    )
    training_stability_audit["diagnostics_path_exists"] = bool(loss_diagnostics_path.is_file())
    training_stability_audit["epoch_predictions_path_exists"] = bool(
        epoch_predictions_vs_targets_path.is_file()
    )

    train_report = {
        "model_name": trainer_state["model_name"],
        "status": "trained",
        "checkpoint_format": "transformer_real_v1",
        "trainer_mode": "single_phase_baseline",
        "epochs": trainer_state["epochs"],
        "train_epoch_losses": train_epoch_losses,
        "val_epoch_losses": val_epoch_losses,
        "epoch_losses": list(val_epoch_losses),
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "best_selection_score": best_selection_score,
        "early_stop_patience": trainer_state["patience"],
        "early_stopped": False,
        "val_selection_scores": val_selection_scores,
        "optimizer": dict(trainer_state["optimizer"]),
        "seed_report": dict(seed_report),
        "cross_trainer_parity_declaration": deepcopy(_CROSS_TRAINER_PARITY_DECLARATION),
        "runtime_device": runtime_device,
        "amp_requested": amp_state["requested"],
        "amp_enabled": amp_state["enabled"],
        "amp_dtype": amp_state["dtype_name"],
        "train_window_count": len(train_windows),
        "val_window_count": len(val_windows),
        "checkpoint_path": str(best_ckpt),
        "epoch_candidate_paths": epoch_candidate_paths,
        "epoch_candidate_epochs": epoch_candidate_epochs,
        "loss_diagnostics_path": str(loss_diagnostics_path),
        "epoch_predictions_vs_targets_path": str(epoch_predictions_vs_targets_path),
        "optimizer_weight_decay_applied": float(
            trainer_state.get("optimizer", {}).get("weight_decay", 0.0)
        ),
        "risk_calibration_enabled": getattr(model, "risk_calibration", None) is not None,
        "tail_selection_observation_coeff": coerce_finite_scalar(
            trainer_state.get("tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
            name="tail_selection_observation_coeff",
            min_value=0.0,
        ),
        "risk_label_mode": RISK_LABEL_MODE_DEFAULT,
        "uwb_scaling_label_mode": UWB_SCALING_LABEL_MODE_DEFAULT,
        "vio_scaling_label_mode": VIO_SCALING_LABEL_MODE_DEFAULT,
        "training_stability_audit": training_stability_audit,
        "output_root": str(output_root),
        "best_ckpt": str(best_ckpt),
        # 准则 34：持久化 train/val loss 曲线 + 过拟合监控（与 liquid/lstm 同口径）.
        "train_loss_curve": list(train_epoch_losses),
        "val_loss_curve": list(val_epoch_losses),
        "overfit_audit": _compute_transformer_overfit_audit(train_epoch_losses, val_epoch_losses),
    }
    report_path = reports_dir / f'{trainer_state["model_name"]}_train_report.json'
    train_report["report_path"] = str(report_path)
    report_path.write_text(
        dumps_json_text(train_report),
        encoding="utf-8",
    )
    # 准则 27：脱钩激活统计 hook 并写入 train_report（与 liquid/lstm 同口径）.
    if activation_hook_handles:
        from liquidloc.models.activation_stats import detach_activation_stats_hooks
        detach_activation_stats_hooks(activation_hook_handles)
    if activation_collector is not None:
        train_report["activation_stats"] = activation_collector.summary()
        from liquidloc.models.activation_stats import DEFAULT_HEAD_NAMES
        train_report["activation_stats_meta"] = {
            "head_names": list(DEFAULT_HEAD_NAMES),
            "epochs_collected": int(trainer_state.get("epochs", 0) or 0),
            "method_name": str(trainer_state.get("model_name", "unknown")),
        }
        # 重新持久化（追加 activation_stats 后）
        report_path.write_text(dumps_json_text(train_report), encoding="utf-8")
    model.network.to("cpu")
    if hasattr(model, "risk_calibration") and hasattr(model.risk_calibration, "to"):
        model.risk_calibration.to("cpu")
    for attr in ("_cached_train_windows", "_cached_val_windows"):
        if hasattr(model, attr):
            delattr(model, attr)
    return str(best_ckpt), train_report
