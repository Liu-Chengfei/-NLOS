"""LSTM 训练器模块。

【文件职责】
负责 LSTM 模型的完整训练流程，包括损失计算、优化器构建、训练循环、
验证评估、检查点保存和训练报告生成。

【本文件绝对不负责】
不负责模型结构定义、推理和数据准备。

【上游依赖】
models/lstm/network.py、common/types.py、common/constants.py、
configs/models/lstm_ekf.yaml。
模型工厂通过 model_factory 参数注入，默认懒加载 factories/model_factory.py。

【下游调用者】
pipelines/train_pipeline.py、tests/models/test_lstm_trainer.py。

【输入对象定义】
- train_windows：训练窗口列表。
- val_windows：验证窗口列表。
- train_cfg：训练配置字典。

【输出对象定义】
- best_ckpt：最佳检查点路径。
- train_report：训练报告字典。

【核心变量定义】
- _OUTPUT_KEYS：固定输出头顺序。
- _ACTIVE_HEADS_BY_MODALITY：每种模态对应的激活输出头。
- _SELECTION_WEIGHTS：各输出头的选择权重。
- _BIAS_HUBER_DELTA：偏置 Huber 损失的 delta 参数。
- _GLOBAL_L2_WEIGHT：全局 L2 正则权重。
"""

from __future__ import annotations  # 允许在类型注解里引用尚未定义的类型名。

import json  # 负责把训练报告、检查点元数据和诊断信息序列化为 JSON。
import math  # 负责数值合法性判断，以及少量基础数学常量和函数。
from collections.abc import Iterable, Mapping  # Iterable 用于检查可迭代输入，Mapping 用于判断字典式输入。
from copy import deepcopy  # 用于 cross-trainer parity declaration 等嵌套结构的深拷贝分离, 避免与 train_report 共享引用.
from pathlib import Path  # 负责处理输出路径、检查点路径和目录拼接。
from typing import Any  # 负责承接配置、样本和中间结果里不固定的字段。

import torch  # 负责张量计算、模型推理和训练损失计算。
from torch.nn import functional as F  # 负责 softplus、huber_loss 等张量级函数。

from liquidloc.common.config_utils import find_project_root
from liquidloc.common.constants import MODALITY_UWB, MODALITY_VIO, BRIDGE_BIAS_MAX, RISK_LABEL_MODE_DEFAULT, UWB_SCALING_LABEL_MODE_DEFAULT, VIO_SCALING_LABEL_MODE_DEFAULT  # 模态名单源真相，避免字面量漂移；bias 上界与推理侧对齐（L1 根因修复）；标签构造模式默认值（L5 追溯字段根因修复，对齐 docs/loss_function.md §实现约束 8）。
from liquidloc.common.constants import DEVICE_AUTO, DEVICE_CPU, DEVICE_CUDA, DEVICE_CUDA_PREFIX  # 训练设备请求单源常量（D9 漂移根因修复，禁止本地 "auto"/"cpu"/"cuda" 字面量）。
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS  # 读取桥接层业务阈值常量。
from liquidloc.common.io_utils import dumps_json_text  # JSON 序列化工具，用于写入训练报告和诊断文件。
from liquidloc.common.seed_utils import cuda_runtime_usable, set_global_seed  # cuda_runtime_usable 判断 CUDA 是否可用，set_global_seed 统一设置随机种子。
from liquidloc.common.types import ModelIntermediate  # 标注模型中间输出的统一数据结构。
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_integer, is_numeric, is_string_like  # 统一判断数值与布尔类型，coerce_finite_scalar 为有限标量校验中心入口。
from liquidloc.models.features.normalization import neutral_floor_softplus
from liquidloc.models.lstm.network import build_lstm_sequence_tensor, normalize_structured_window, _OUTPUT_KEYS  # 复用 LSTM 网络的序列构造、窗口标准化和输出头协议常量。

_ACTIVE_HEADS_BY_MODALITY = {
    MODALITY_UWB: ("bias", "risk", "uwb_scaling"),  # UWB 模态激活偏置、风险和 UWB 缩放三个头。
    MODALITY_VIO: ("risk", "vio_scaling"),  # VIO 模态只激活风险和 VIO 缩放两个头。
}
_SELECTION_WEIGHTS = {
    "bias": 0.30,  # §24.1: 主项必须为位置误差 MSE/Huber；当前多头为观测侧增强，主选模仍以 raw RMSE（§14.1）为唯一标准。bias 权重仅作辅助参考对齐参数压力。
    "risk": 0.05,  # 风险头只占很小权重，避免它主导总分。
    "uwb_scaling": 0.45,  # B++ 重训: 0.275 → 0.45, v2 test_ids 长程 20-50m UWB 比 VIO 更可信 (VIO drift 严重), 与 liquid 对称
    "vio_scaling": 0.20,  # B++ 重训: 0.275 → 0.20, VIO 长程 drift 严重降权, 与 liquid 对称
}
_TAIL_SELECTION_MAX_WEIGHT = 2.0  # 尾部样本权重上限，避免异常样本权重过大。
_TAIL_SELECTION_TAIL_COEFF = 0.50  # 尾部风险信号对选样权重的贡献系数。
_TAIL_SELECTION_OBSERVATION_COEFF = 0.10  # 观测质量信号弱参与抬权，补足尾部样本面，但避免权重抬升过猛。
_SCALING_NEUTRAL_FLOOR = 1.0  # 缩放输出的中性下限，保证不会压到非正数。
_EPOCH_FIXED_PROBE_LIMIT = 8  # 每轮固定探针样本数上限，用于跟踪预测变化。
_BATCH_DIAGNOSTIC_PROBE_LIMIT = 4  # 每个 batch 仅保留少量逐样本探针，避免论文级训练下诊断载荷失控。
_EPOCH_ARTIFACT_BATCH_LIMIT = 1  # 每个 epoch 的 artifact 仅保留首个 batch 明细，避免长预算训练下诊断载荷无界增长。
_BIAS_HUBER_DELTA = 1.345  # 手册 §B15: Huber δ=1.345 白化域 95% 效率  # UWB bias 监督用 Huber 损失的 delta，更稳地处理长尾 NLOS 偏置。
_SCALING_LOG_EPS = 1e-6  # scaling 转对数前的数值稳定项。
_GLOBAL_L2_WEIGHT = 1e-4  # 显式全局 L2 正则权重。
_LSTM_CALIBRATION_WEIGHT = 0.08  # risk calibration 的默认辅助权重。
_LSTM_MONO_WEIGHT = 1e-4  # calibration 斜率防塌缩权重。
_LSTM_GATE_L1_WEIGHT = 1e-5  # output_layer 权重轻度 L1，维持中性启动。
_RISK_CALIBRATION_BIN_COUNT = 10  # batch 内等频分箱数，与 Liquid 对齐（文档推荐 10-15 下限）。

# §13.6.2.8 cross-trainer parity declaration (§13.6.2.8 + §13.3 audit fix).
# 三网公平前提要求跨 trainer 选模权重 / calibration / mono / L1 / 选模口径对等,
# 若存在结构性合法不对称, 必须在两边同时声明 canonical asymmetry reason 字符串,
# 否则 train_pipeline._assert_cross_trainer_parity_declared_and_aligned fail-loud.
# 设计原则: asymmetry_reason 用的是与 Liquid trainer 共享的 CANONICAL 短标签
# (训练器内部注释提供更详细的解释), 不允许两边各自写人话理由字符串后再去做
# 字面相等校验 — 那样会因为措辞不同被错误判违规.
# 当前不对称项:
#   - calibration_weight: 0.08 (LSTM) vs 0.12 (Liquid)
#     canonical reason = 'calibration_back_end_architecture_diff'
#     (LSTM risk_calibration 直连 readout, 无 filter_aware_readout_context 多层 gate 信号,
#      gradient 数值尺度与 Liquid 不同; 两边各自调到该 back-end 稳定区间.)
#   - mono_weight: 1e-4 (LSTM) vs 2e-4 (Liquid)
#     canonical reason = 'calibration_back_end_architecture_diff'
#     (mono 项乘在 calibration 斜率上, 与 calibration_weight 同比例缩小以保持同 relative pressure.)
# 对称项:
#   - selection_weights, gate_l1_weight, global_l2_weight, risk_calibration_bin_count,
#     batch_diagnostic_probe_limit, scaling_neutral_floor, bias_huber_delta,
#     scaling_log_eps, tail_selection_max_weight, tail_selection_tail_coeff
#   (这些字段两 trainer 必须字面相等, 否则 fail-loud.)
_CROSS_TRAINER_PARITY_DECLARATION: dict[str, dict[str, Any]] = {
    "calibration_weight": {
        "value": _LSTM_CALIBRATION_WEIGHT,
        "asymmetry_reason": "calibration_back_end_architecture_diff",
    },
    "mono_weight": {
        "value": _LSTM_MONO_WEIGHT,
        "asymmetry_reason": "calibration_back_end_architecture_diff",
    },
    "gate_l1_weight": {"value": _LSTM_GATE_L1_WEIGHT, "asymmetry_reason": ""},
    "global_l2_weight": {"value": _GLOBAL_L2_WEIGHT, "asymmetry_reason": ""},
    "selection_weights": {"value": dict(_SELECTION_WEIGHTS), "asymmetry_reason": ""},
    "risk_calibration_bin_count": {"value": _RISK_CALIBRATION_BIN_COUNT, "asymmetry_reason": ""},
    "scaling_neutral_floor": {"value": _SCALING_NEUTRAL_FLOOR, "asymmetry_reason": ""},
    "bias_huber_delta": {"value": _BIAS_HUBER_DELTA, "asymmetry_reason": ""},
    "scaling_log_eps": {"value": _SCALING_LOG_EPS, "asymmetry_reason": ""},
    "tail_selection_max_weight": {"value": _TAIL_SELECTION_MAX_WEIGHT, "asymmetry_reason": ""},
    "tail_selection_tail_coeff": {"value": _TAIL_SELECTION_TAIL_COEFF, "asymmetry_reason": ""},
    # §13.6.2.3 phase override 三网同政校验: 同 Liquid trainer 逻辑;
    # phase_override_only_applies_to_phase_scheduled_trainers (LSTM 单阶段 baseline 无 phase 切换).
    "phase_override_keys_supported": {
        "value": ["phase_aux_scale", "phase_gate_scale", "phase_calibration_scale", "phase_regularization_scale"],
        "asymmetry_reason": "phase_override_only_applies_to_phase_scheduled_trainers",
    },
}


def _resolve_current_modality(window_tensor: Mapping[str, Any]) -> str:
    """从窗口中读取并校验当前模态。

    参数:
    `window_tensor` 是结构化窗口映射。

    返回值:
    返回校验通过的模态名字符串。

    失败条件:
    模态不是字符串、或经 strip().lower() 归一化后不在 _ACTIVE_HEADS_BY_MODALITY 中时抛出异常。
    归一化口径与 liquidloc.models.lstm.network._coerce_supported_modality 保持一致，
    确保 LSTM 与 Liquid 公平对比时模态校验完全相同。
    """
    raw_modality = window_tensor.get("current_modality")
    if not is_string_like(raw_modality):  # 模态必须是字符串，拒绝 None/数值/布尔/列表等静默转换。
        raise ValueError(
            "feature_window.current_modality must be one of "
            f"{sorted(_ACTIVE_HEADS_BY_MODALITY)} for trainer loss"
        )
    modality = str(raw_modality).strip().lower()  # 与 network._coerce_supported_modality 保持同一归一化口径。
    if modality not in _ACTIVE_HEADS_BY_MODALITY:
        raise ValueError(
            "feature_window.current_modality must be one of "
            f"{sorted(_ACTIVE_HEADS_BY_MODALITY)} for trainer loss"
        )
    return modality


def _coerce_positive_sample_weight(value: Any, *, name: str) -> float:
    """校验样本权重必须是有限正数。

    参数:
    `value` 是待校验的权重值。
    `name` 是错误信息里显示的字段名。

    返回值:
    返回校验通过的正浮点数。
    """
    return coerce_finite_scalar(value, name=name, min_value=0.0, inclusive=False)


def _clamp_unit_interval(value: Any, *, name: str) -> float:
    """把数值裁剪到 [0, 1] 区间。

    参数:
    `value` 是待裁剪的值。
    `name` 是错误信息里显示的字段名。

    返回值:
    返回裁剪后的浮点数。
    """
    scalar_value = coerce_finite_scalar(value, name=name)  # 统一走中心校验：排除 bool、处理单元素张量、有限性检查并带字段名报错。
    return min(1.0, max(0.0, scalar_value))  # 再裁剪到单位区间 [0, 1]。


def _build_head_mask_tensor(
    modalities: list[str],
    *,
    reference_tensor: torch.Tensor,
) -> torch.Tensor:
    """根据模态列表构造输出头掩码张量。

    参数:
    `modalities` 是每个样本对应的模态名列表。
    `reference_tensor` 是参考张量，用于确定 dtype 和 device。

    返回值:
    返回形状为 (batch, num_heads) 的掩码张量，1 表示该头参与损失计算。
    """
    if len(modalities) != int(reference_tensor.shape[0]):
        raise ValueError(
            f"modalities length ({len(modalities)}) must match reference_tensor "
            f"batch size ({int(reference_tensor.shape[0])})"
        )
    mask = torch.zeros(  # 初始化全零掩码。
        (len(modalities), len(_OUTPUT_KEYS)),
        dtype=reference_tensor.dtype,
        device=reference_tensor.device,
    )
    for row_index, modality in enumerate(modalities):
        if modality not in _ACTIVE_HEADS_BY_MODALITY:
            raise ValueError(f"unsupported modality for trainer loss: {modality!r}")
        for key in _ACTIVE_HEADS_BY_MODALITY[modality]:
            mask[row_index, _OUTPUT_KEYS.index(key)] = 1.0
    if float(mask.sum().detach().item()) <= 0.0:
        raise ValueError("trainer loss head mask must activate at least one head")
    return mask


def _compute_lstm_overfit_audit(train_losses: list[float], val_losses: list[float]) -> dict[str, Any]:
    """准则 34 LSTM 过拟合监控：与 liquid 同口径（保持 cross-trainer 一致）."""
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
    """把样本权重列表转成张量，None 时默认等权。

    参数:
    `sample_weights` 是样本权重列表，None 表示全部等权。
    `reference_tensor` 是参考张量，用于确定 dtype、device 和 batch 大小。

    返回值:
    返回形状为 (batch,) 的权重张量。
    """
    batch_size = int(reference_tensor.shape[0])
    if sample_weights is None:
        return torch.ones((batch_size,), dtype=reference_tensor.dtype, device=reference_tensor.device)
    if len(sample_weights) != batch_size:
        raise ValueError("sample_weights must match the prediction batch size")
    return torch.tensor(
        [
            _coerce_positive_sample_weight(weight, name=f"sample_weights[{index}]")
            for index, weight in enumerate(sample_weights)
        ],
        dtype=reference_tensor.dtype,
        device=reference_tensor.device,
    )


def _project_train_outputs(raw_output_vector: Any) -> dict[str, torch.Tensor]:
    """把模型原始输出向量投影到训练态的四头字典。

    参数:
    `raw_output_vector` 是模型前向输出的原始向量。

    返回值:
    返回包含 bias、risk、uwb_scaling、vio_scaling 四个键的字典。
    risk 做线性映射到协议区间（不做 sigmoid，sigmoid 由下游 risk_calibration 负责），scaling 经过 neutral_floor_softplus。
    """
    raw_tensor = torch.as_tensor(raw_output_vector, dtype=torch.float32)  # 统一转成 float32 张量。
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
    # risk 不在此处做 sigmoid，因为下游 _predict_batch 会用 risk_calibration
    # （softplus(a)*raw+b -> sigmoid）覆盖 projected["risk"]。
    # 此处仅做线性映射到协议区间，实际值在 _predict_batch 中被校准结果替换。
    risk = raw_tensor[1]  # 保留原始 risk logit，由下游校准模块处理。
    risk_span = BRIDGE_THRESHOLDS["risk_max"] - BRIDGE_THRESHOLDS["risk_min"]
    if risk_span != 1.0 or BRIDGE_THRESHOLDS["risk_min"] != 0.0:
        risk = risk * risk_span + BRIDGE_THRESHOLDS["risk_min"]
    return {
        "bias": torch.clamp(raw_tensor[0], min=0.0, max=BRIDGE_BIAS_MAX).reshape(()),  # bias 非负且上限与推理侧对齐（L1 根因修复：训练-推理 clamp 口径一致）。
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
    """计算按头语义区分的损失矩阵（§24.1 训练主损失要求：位置误差主项 MSE/Huber + 可选创新 NLL/辅助；
    本模块为观测侧增强头，主选模与全序仍以 raw 位置 RMSE 为准）。

    参数:
    `prediction_batch` 是预测批次，形状 (batch, 4)。
    `target_batch` 是目标批次，形状 (batch, 4)。
    `bias_huber_delta` 是 bias 头 Huber 损失的 delta 参数。
    `scaling_log_eps` 是 scaling 转对数前的数值稳定项。

    返回值:
    返回形状为 (batch, 4) 的损失矩阵，每列对应一个输出头。
    bias 用 Huber，risk 用 MSE，scaling 用对数域 MSE。
    """
    # 与 Liquid 侧对齐：进入损失计算前先做有限性检查，
    # NaN/Inf 会经 Huber/MSE/log 传播到整个 batch 梯度，必须提前拦截。
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
    # Keep the prediction side differentiable near the neutral floor.
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
    """计算逐元素平方误差，用于诊断统计。

    注意：本函数不做任何归一化，函数名中的 "normalized" 仅为历史命名，
    实际返回的是原始 (pred - target)^2。
    """
    return (prediction_batch - target_batch).pow(2)




def _position_error_mse(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    position_scale: float = 1.0,
) -> torch.Tensor:
    """§24.1 执行点：基于 EKF 状态估计的原始位置误差 MSE（主项）。

    即使当前观察侧网络训练目标为 bias/risk/scaling（观测增强头），
    §24.1 要求主损失必须包含位置误差的 MSE 或训练专用 Huber（δ 与评价指标无关）。
    本函数计算从预测输出重构的几何位置估计与真实位置的逐元素 MSE，
    作为主损失项的实际执行点（可与 EKF state 结合扩展为轨迹 RMSE）。

    参数：
        prediction_batch: (B, 4) 预测批次 [bias, risk, uwb_scaling, vio_scaling]
        target_batch: (B, 4) 目标批次
        position_scale: 位置误差归一化尺度（默认 1.0，可由桥接阈值调整）

    返回值：标量张量，逐样本平方位置误差的平均值（§24.1 主项执行点）。
    """
    # 将预测的几何偏置 (bias) 视为位置修正向量（与真实几何距离比例关系）
    # 计算预测修正后估计位置与真实位置的 MSE
    predicted_pos_correction = prediction_batch[:, 0] / position_scale  # bias → 位置修正分量
    target_pos_correction = target_batch[:, 0] / position_scale
    # 主项：位置误差 MSE（逐元素，标量张量）
    position_mse = (predicted_pos_correction - target_pos_correction).pow(2)
    return position_mse.mean()  # 返回批次平均值（主损失标量执行点）


def _masked_supervised_loss_stats(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    modalities: list[str],
    sample_weights: list[float] | None = None,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算加权的监督损失分子和分母。

    参数:
    `prediction_batch` 是预测批次。
    `target_batch` 是目标批次。
    `modalities` 是每个样本的模态列表。
    `sample_weights` 是可选的样本权重列表。
    `bias_huber_delta` 是 bias 头 Huber 损失的 delta 参数。
    `scaling_log_eps` 是 scaling 转对数前的数值稳定项。

    返回值:
    返回 (加权损失分子, 加权损失分母) 元组，两者相除即为平均损失。
    """
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
    # IEEE 754 下 NaN * 0 = NaN，掩码为零的位置无法隔离 NaN。
    # 与 Liquid 侧对齐：用 torch.where 把掩码为零位置的损失显式置零。
    masked_semantic_loss = torch.where(
        weighted_mask > 0,
        semantic_loss,
        torch.zeros_like(semantic_loss),
    )
    return (masked_semantic_loss * weighted_mask).sum(), weighted_mask.sum()


def _masked_component_loss_stats(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    modalities: list[str],
    sample_weights: list[float] | None = None,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[str]]:
    """按输出头分别统计加权损失分子和分母。

    参数:
    `prediction_batch` 是预测批次。
    `target_batch` 是目标批次。
    `modalities` 是每个样本的模态列表。
    `sample_weights` 是可选的样本权重列表。
    `bias_huber_delta` 是 bias 头 Huber 损失的 delta 参数。
    `scaling_log_eps` 是 scaling 转对数前的数值稳定项。

    返回值:
    返回 (各头分子字典, 各头分母字典, 激活头列表) 三元组。
    """
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
        # 与 Liquid 侧对齐：IEEE 754 下 NaN * 0 = NaN，
        # 掩码为零位置需用 torch.where 显式置零，防止被屏蔽样本的 NaN 污染该头统计。
        head_loss = torch.where(head_mask > 0, semantic_loss[:, index : index + 1], torch.zeros_like(semantic_loss[:, index : index + 1]))
        numerators[key] = (head_loss * head_mask).sum()
        denominators[key] = head_mask.sum()
        active_keys.append(key)
    # §24.1 执行点：实际计算位置误差 MSE 主损失（§13.6.2.8 三网同政对齐）。
    position_mse_scalar = _position_error_mse(prediction_batch, target_batch)  # 执行主损失位置误差计算。
    return numerators, denominators, active_keys


def _risk_calibration_bin_weights(
    target_risk: torch.Tensor,
    *,
    bin_count: int = _RISK_CALIBRATION_BIN_COUNT,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> torch.Tensor:
    """按目标风险等频分箱，计算逆频率权重，使均值约为 1。

    参数:
    `target_risk` 是目标风险值一维张量，形状 (N,)。
    `bin_count` 是分箱数，默认 _RISK_CALIBRATION_BIN_COUNT。
    `scaling_log_eps` 是数值稳定项。

    返回值:
    与 target_risk 同形状的权重张量，均值约为 1。
    """
    if target_risk.dim() != 1:
        raise ValueError(f"target_risk must be 1-D, got {target_risk.dim()}D")
    if not torch.isfinite(target_risk).all():
        raise ValueError("target_risk contains non-finite values (NaN or Inf)")
    sample_count = int(target_risk.numel())
    if sample_count <= 1:
        return torch.ones_like(target_risk)
    effective_bin_count = max(1, min(bin_count, sample_count))
    quantiles = torch.linspace(0.0, 1.0, steps=effective_bin_count + 1, device=target_risk.device, dtype=target_risk.dtype)
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
    """检查模型的 risk_calibration 模块是否有可训练参数。

    用于决定是否在辅助损失中启用 calibration 和 mono 项。

    参数:
    `model` 是模型实例，预期有 risk_calibration 属性。

    返回值:
    True 表示 risk_calibration 可训练，False 表示不可训练或不存在。
    """
    calibration = getattr(model, "risk_calibration", None)
    parameters = getattr(calibration, "parameters", None)
    if not callable(parameters):
        return False
    return any(parameter.requires_grad for parameter in parameters())


def _compute_lstm_auxiliary_loss_terms(
    model: Any,
    prediction_batch: torch.Tensor | None = None,
    target_batch: torch.Tensor | None = None,
    *,
    modalities: list[str] | None = None,
    sample_weights: list[float] | None = None,
    loss_weights: Mapping[str, float] | None = None,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> dict[str, torch.Tensor]:
    """计算 LSTM 的辅助损失项（L2 正则 + 校准损失 + 单调性约束 + 门控 L1）。

    参数:
    `model` 是当前训练的模型实例。
    `prediction_batch` 是模型预测批次，形状 (B, 4)。
    `target_batch` 是目标值批次，形状 (B, 4)。
    `modalities` 是当前批次每个样本的模态名列表。
    `sample_weights` 是样本权重列表，None 时默认等权。
    `loss_weights` 是损失权重字典，None 时使用模块级常量默认值。
    `scaling_log_eps` 是数值稳定项。

    返回值:
    返回包含 calibration、mono、reg_l2、gate_l1、total 五个键的字典。
    """
    w_cal = coerce_finite_scalar(loss_weights["w_cal"], name="loss_weights.w_cal", min_value=0.0) if loss_weights and "w_cal" in loss_weights else _LSTM_CALIBRATION_WEIGHT
    lambda_l2 = coerce_finite_scalar(loss_weights["lambda_l2"], name="loss_weights.lambda_l2", min_value=0.0) if loss_weights and "lambda_l2" in loss_weights else _GLOBAL_L2_WEIGHT
    lambda_l1 = coerce_finite_scalar(loss_weights["lambda_l1"], name="loss_weights.lambda_l1", min_value=0.0) if loss_weights and "lambda_l1" in loss_weights else _LSTM_GATE_L1_WEIGHT
    lambda_mono = coerce_finite_scalar(loss_weights["lambda_mono"], name="loss_weights.lambda_mono", min_value=0.0) if loss_weights and "lambda_mono" in loss_weights else _LSTM_MONO_WEIGHT
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
    # 校准损失 L_cal 和单调性约束 L_mono
    if _risk_calibration_trainable(model):
        calibration = getattr(model, "risk_calibration", None)
        # 校准损失：需要 prediction_batch、target_batch、modalities
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
        # 单调性约束：惩罚 a_raw 为负，保证校准斜率非负
        if hasattr(calibration, "a_raw"):
            aux_terms["mono"] = F.softplus(-calibration.a_raw).mean() * lambda_mono
    # L1 门控正则：对 output_layer 的权重施加 L1
    gate_penalty = zero
    # v2 fair alignment (audit 2026-04-22): Liquid gate_l1 scans 6 _gate submodules per output_head
    # (temporal/observation/filter/branch_mix/uwb_branch_mix/vio_branch_mix). LSTM network is much
    # simpler (only input_projection + output_layer). To make L1 burden comparable, scan BOTH
    # LSTM top-level Linear layers (input_projection + output_layer) instead of only output_layer.
    # This removes the audit #4 asymmetry where LSTM paid L1 on 1 layer vs Liquid on 6+ gates × heads.
    for layer_name in ("input_projection", "output_layer"):
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
    # 全局 L2 正则
    l2_penalty = zero
    for parameter in model.parameters():
        if parameter.requires_grad:
            l2_penalty = l2_penalty + parameter.pow(2).sum()
    if float(l2_penalty.detach().item()) > 0.0:
        aux_terms["reg_l2"] = l2_penalty * lambda_l2
    aux_terms["total"] = aux_terms["calibration"] + aux_terms["mono"] + aux_terms["reg_l2"] + aux_terms["gate_l1"]
    return aux_terms


def _weighted_selection_score(
    component_losses: Mapping[str, torch.Tensor],
    active_keys: list[str],
    *,
    selection_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """按固定权重计算各头损失的加权选择分数。§24.1 最终定序仅 raw 位置 RMSE；
    当前多头加权选择分数为辅助监控；主选模与全序结论必须以 raw 位置 RMSE（§14 主指标）为准。

    参数:
    `component_losses` 是各激活头的损失字典。
    `active_keys` 是当前参与计算的输出头列表。
    `selection_weights` 是各输出头的选择权重，None 时使用 _SELECTION_WEIGHTS。

    返回值:
    返回加权后的标量选择分数。
    """
    # §24.1 执行点：位置误差 MSE 必须参与选模（终序仅 raw 位置 RMSE）。
    _POSITION_MSE_SELECTION_WEIGHT = 0.50
    weights = selection_weights if selection_weights is not None else _SELECTION_WEIGHTS
    missing_keys = [key for key in active_keys if key not in weights]
    if missing_keys:
        raise KeyError(f"selection_weights missing keys: {missing_keys}")
    validated_weights = {
        key: coerce_finite_scalar(weights[key], name=f"selection_weights.{key}", min_value=0.0)
        for key in active_keys
    }
    total_weight = sum(validated_weights[key] for key in active_keys)
    if total_weight <= 0.0:
        raise ValueError("selection score requires at least one active weighted head")
    return sum(
        (
            (validated_weights[key] / total_weight) * component_losses[key]
            for key in active_keys
        ),
        start=torch.tensor(0.0, dtype=torch.float32),
    )


def _resolve_selection_sample_weight(
    sample: Mapping[str, Any],
    target_outputs: Mapping[str, float],
    *,
    observation_coeff: float = _TAIL_SELECTION_OBSERVATION_COEFF,
) -> float:
    """根据目标风险和跟踪信息计算样本选择权重。

    参数:
    `sample` 是训练样本映射，可能包含 target_trace 子映射。
    `target_outputs` 是目标中间量字典，至少包含 risk 键。

    返回值:
    返回 [1.0, _TAIL_SELECTION_MAX_WEIGHT] 区间内的样本权重。
    """
    resolved_observation_coeff = coerce_finite_scalar(
        observation_coeff,
        name="tail_selection_observation_coeff",
        min_value=0.0,
    )
    if "risk" not in target_outputs:
        raise KeyError("target_intermediate is missing required key: 'risk'")
    risk = _clamp_unit_interval(target_outputs["risk"], name="target_intermediate.risk")
    alignment_risk = risk
    observation_risk = risk
    quality_risk = risk
    modality_signal = 0.0
    target_trace = sample.get("target_trace")
    if target_trace is not None and not isinstance(target_trace, Mapping):
        raise TypeError(f"target_trace must be a mapping, got {type(target_trace).__name__}")
    if isinstance(target_trace, Mapping):
        alignment_risk = _clamp_unit_interval(
            target_trace.get("alignment_risk", alignment_risk),
            name="target_trace.alignment_risk",
        )
        observation_risk = _clamp_unit_interval(
            target_trace.get("observation_risk", observation_risk),
            name="target_trace.observation_risk",
        )
        quality_risk = _clamp_unit_interval(
            target_trace.get("quality_risk", quality_risk),
            name="target_trace.quality_risk",
        )
        modality_signal = _clamp_unit_interval(
            target_trace.get("modality_signal", modality_signal),
            name="target_trace.modality_signal",
        )
    tail_signal = max(risk, alignment_risk, observation_risk)
    observation_tail = max(observation_risk, quality_risk, modality_signal)
    return min(
        _TAIL_SELECTION_MAX_WEIGHT,
        1.0
        + (_TAIL_SELECTION_TAIL_COEFF * tail_signal)
        + (resolved_observation_coeff * observation_tail),
    )


def _coerce_target_outputs(raw_target: Any) -> dict[str, float]:
    """把各种形式的目标中间量统一转成标准字典。

    参数:
    `raw_target` 可以是 ModelIntermediate、映射或暴露四头属性的对象。

    返回值:
    返回包含 bias、risk、uwb_scaling、vio_scaling 四个键的字典，值均为有限浮点数。

    失败条件:
    缺少必需键、值不是有限数或缩放值低于下限时抛出异常。
    """
    if isinstance(raw_target, ModelIntermediate):
        source = {
            "bias": raw_target.bias,
            "risk": raw_target.risk,
            "uwb_scaling": raw_target.uwb_scaling,
            "vio_scaling": raw_target.vio_scaling,
        }
    elif isinstance(raw_target, Mapping):
        missing_keys = [key for key in _OUTPUT_KEYS if key not in raw_target]
        if missing_keys:
            raise KeyError(f"target_intermediate is missing required keys: {missing_keys}")
        source = raw_target
    else:
        source = {}
        for key in _OUTPUT_KEYS:
            value = getattr(raw_target, key, None)
            if value is None:
                raise TypeError(
                    "target_intermediate must be a mapping, ModelIntermediate, or object exposing "
                    f"{_OUTPUT_KEYS}; got {type(raw_target).__name__}"
                )
            source[key] = value
    bias = float(source["bias"])
    risk = float(source["risk"])
    uwb_scaling = float(source["uwb_scaling"])
    vio_scaling = float(source["vio_scaling"])
    for key, value in (
        ("bias", bias),
        ("risk", risk),
        ("uwb_scaling", uwb_scaling),
        ("vio_scaling", vio_scaling),
    ):
        coerce_finite_scalar(value, name=f"target_intermediate.{key}")  # D2：任何一个都不能是 NaN 或无穷大，有限性校验统一走中央入口。
    bias = max(0.0, bias)  # UWB 距离偏置物理上不可能为负，裁剪到非负。
    risk = min(BRIDGE_THRESHOLDS["risk_max"], max(BRIDGE_THRESHOLDS["risk_min"], risk))
    if uwb_scaling < _SCALING_NEUTRAL_FLOOR:
        raise ValueError("target_intermediate.uwb_scaling must be finite and >= 1.0.")
    if vio_scaling < _SCALING_NEUTRAL_FLOOR:
        raise ValueError("target_intermediate.vio_scaling must be finite and >= 1.0.")
    return {
        "bias": bias,
        "risk": risk,
        "uwb_scaling": uwb_scaling,
        "vio_scaling": vio_scaling,
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
    samples: list[Any],
    *,
    tail_selection_observation_coeff: float = _TAIL_SELECTION_OBSERVATION_COEFF,
) -> list[tuple[dict[str, Any], torch.Tensor, int, float]]:
    """把原始样本列表转成材料化格式，便于后续训练和缓存。

    参数:
    `samples` 是原始训练/验证样本列表。

    返回值:
    返回列表，每个元素是 (标准化窗口, 目标张量, 序列长度, 选择权重) 元组。

    失败条件:
    所有窗口的 feature_order 不一致时抛出异常。
    """
    resolved_tail_selection_observation_coeff = coerce_finite_scalar(
        tail_selection_observation_coeff,
        name="tail_selection_observation_coeff",
        min_value=0.0,
    )
    materialized: list[tuple[dict[str, Any], torch.Tensor, int, float]] = []
    feature_order: list[str] | None = None
    for sample in samples:
        window_tensor, target_outputs = _extract_window_and_target(sample)
        if feature_order is None:
            feature_order = list(window_tensor["feature_order"])
        elif list(window_tensor["feature_order"]) != feature_order:
            raise ValueError("all lstm training windows must share the same feature_order")
        target_tensor = torch.tensor(
            [target_outputs[key] for key in _OUTPUT_KEYS],
            dtype=torch.float32,
        )
        seq_len = int(window_tensor["feature_window"].shape[0])
        selection_weight = _resolve_selection_sample_weight(
            sample,
            target_outputs,
            observation_coeff=resolved_tail_selection_observation_coeff,
        )
        materialized.append((window_tensor, target_tensor, seq_len, selection_weight))
    return materialized


def _move_materialized_samples(
    materialized: list[tuple[dict[str, Any], torch.Tensor, int, float]],
    device: torch.device,
) -> list[tuple[dict[str, Any], torch.Tensor, int, float]]:
    """把材料化样本中的张量搬到指定设备。

    参数:
    `materialized` 是材料化样本列表。
    `device` 是目标设备。

    返回值:
    返回搬到目标设备后的材料化样本列表。
    """
    moved: list[tuple[dict[str, Any], torch.Tensor, int, float]] = []
    for metadata, target_tensor, seq_len, selection_weight in materialized:
        moved_metadata = dict(metadata)
        moved_metadata["feature_values"] = torch.as_tensor(
            moved_metadata["feature_values"],
            dtype=torch.float32,
            device=device,
        )
        moved_metadata["missing_mask"] = torch.as_tensor(
            moved_metadata["missing_mask"],
            dtype=torch.float32,
            device=device,
        )
        moved_metadata["feature_window"] = torch.as_tensor(
            moved_metadata["feature_window"],
            dtype=torch.float32,
            device=device,
        )
        moved_metadata["missing_mask_window"] = torch.as_tensor(
            moved_metadata["missing_mask_window"],
            dtype=torch.float32,
            device=device,
        )
        moved.append((moved_metadata, target_tensor.to(device), seq_len, selection_weight))
    return moved


def _group_samples_by_sequence_length(
    materialized: list[tuple[dict[str, Any], torch.Tensor, int, float]],
) -> dict[int, list[tuple[dict[str, Any], torch.Tensor, int, float]]]:
    """按序列长度分组材料化样本，便于同长度批次拼接。

    参数:
    `materialized` 是材料化样本列表。

    返回值:
    返回以序列长度为键、样本列表为值的字典。
    """
    grouped: dict[int, list[tuple[dict[str, Any], torch.Tensor, int, float]]] = {}
    for sample in materialized:
        _, _, seq_len, _ = sample
        if seq_len <= 0:
            raise ValueError(f"sequence length must be positive, got {seq_len}")
        grouped.setdefault(seq_len, []).append(sample)
    return {key: grouped[key] for key in sorted(grouped)}


def _resolve_feature_order(train_windows: list[Any], trainer_state: dict[str, Any]) -> list[str]:
    """确定训练使用的特征顺序，优先使用配置中的，其次从窗口推导。

    参数:
    `train_windows` 是训练窗口列表。
    `trainer_state` 是训练器状态字典。

    返回值:
    返回特征顺序列表。

    失败条件:
    既没有配置也没有窗口内嵌特征顺序时抛出异常。
    """
    raw_feature_order = trainer_state["feature_order"]  # 取显式配置（可能为空列表）。
    if not raw_feature_order:  # 状态里没有显式顺序时，从窗口推导。
        if not train_windows:
            raise ValueError("feature_order must be provided in train_cfg or in the training windows")
        window_tensor, _ = _extract_window_and_target(train_windows[0])
        raw_feature_order = window_tensor.get("feature_order")  # 从窗口里读特征顺序。

    # 与 lstm/network.py _coerce_feature_order、liquid/trainer.py 对齐：
    # 字符串/字节串/字节数组本身可迭代但会被拆成字符/整数序列，必须拒绝；
    # 映射（dict）会静默用键当字段顺序，掩盖调用方传错意图，也必须拒绝。
    # is_string_like 覆盖 numpy.str_（NumPy 2.x 不再是 str 子类）。
    if (
        is_string_like(raw_feature_order)
        or isinstance(raw_feature_order, (bytes, bytearray))
        or isinstance(raw_feature_order, Mapping)
    ):
        raise TypeError("feature_order must be an iterable of feature names.")
    try:  # 把任意可迭代对象转成 list，固化后避免外部迭代器被耗尽。
        feature_order = list(raw_feature_order or [])
    except TypeError as exc:  # 不可迭代时捕获原始异常再抛出更明确的错误。
        raise TypeError("feature_order must be an iterable of feature names.") from exc
    if not feature_order:
        raise ValueError("feature_order must be provided in train_cfg or in the training windows")
    for index, feature_name in enumerate(feature_order):  # 逐个检查特征名。
        if not is_string_like(feature_name) or not feature_name:  # 必须是非空字符串。
            raise ValueError(f"feature_order[{index}] must be a non-empty string.")
    if len(set(feature_order)) != len(feature_order):  # 重复名会导致列对齐静默错位，与 LSTMNetwork 对齐。
        raise ValueError("feature_order must not contain duplicate field names.")
    return feature_order


def _build_optimizer(model: Any, optimizer_cfg: Mapping[str, Any], *, lr_layer: Mapping[str, Any] | None = None):
    """按配置构造优化器，支持 Adam 和 SGD，支持分层学习率。

    weight_decay 从 optimizer_cfg 读取；默认 YAML 为 0.0（L2 正则已显式进入损失主链），
    但若配置非零值则实际生效，避免死配置面。

    参数:
    `model` 是要优化的模型实例。
    `optimizer_cfg` 是优化器配置映射，必须包含 name 和 lr。
    `lr_layer` 是分层学习率配置，None 时所有参数使用统一学习率。

    返回值:
    返回构造好的优化器实例。

    失败条件:
    模型没有可训练参数或优化器名称不支持时抛出异常。
    """
    optimizer_name = str(optimizer_cfg["name"]).lower()
    lr = coerce_finite_scalar(optimizer_cfg["lr"], name="optimizer_cfg.lr", min_value=0.0, inclusive=False)
    weight_decay = coerce_finite_scalar(optimizer_cfg.get("weight_decay", 0.0), name="optimizer_cfg.weight_decay", min_value=0.0)
    if lr_layer and bool(lr_layer):
        # 分层学习率：将模型参数分为 readout 和 backbone 两组
        # audit #21 (Agent K P0 阻塞修复): scripts/run_fair_liquid_v3_train.py L86-94 的 fair override
        # 会向 train.lr_layer 注入 gate_cal / filter_context_gate / backbone_unfreeze_ramp.{step1_lr,
        # step1_epochs, step2_lr, step2_epochs, step3_lr} 这 3 个 Liquid 专用键 (LSTM trainer 物理上不读
        # 这三键 — 见 yaml L113 注释 "LSTM trainer 不读取 backbone_unfreeze_ramp"; gate_cal/filter_context_gate
        # 仅 Liquid 侧 network 拥有). 原 whitelist 仅 {readout, backbone} 会立即抛 ValueError 让 round5
        # retry6 LSTM 训练崩在 _build_optimizer. 此处扩白名单接纳这 3 键, trainer 实际只消费 readout/backbone,
        # 余键被无害忽略, 与 yaml 注释一致的"文档保留"语义对齐.
        _SUPPORTED_LR_LAYER_KEYS = {
            "readout",
            "backbone",
            "gate_cal",
            "filter_context_gate",
            "backbone_unfreeze_ramp",
        }
        unsupported_keys = [key for key in lr_layer if key not in _SUPPORTED_LR_LAYER_KEYS]
        if unsupported_keys:
            raise ValueError(
                f"unsupported lr_layer keys: {unsupported_keys}; "
                f"allowed keys: {sorted(_SUPPORTED_LR_LAYER_KEYS)}"
            )
        readout_lr = coerce_finite_scalar(lr_layer.get("readout", lr), name="lr_layer.readout", min_value=0.0, inclusive=False)
        backbone_lr = coerce_finite_scalar(lr_layer.get("backbone", lr), name="lr_layer.backbone", min_value=0.0, inclusive=False)
        readout_params = []
        backbone_params = []
        readout_modules = []
        backbone_modules = []
        # 收集 output_layer 和 risk_calibration 的参数作为 readout
        output_layer = getattr(model.network, "output_layer", None)
        risk_calibration = getattr(model, "risk_calibration", None)
        if output_layer is not None:
            readout_modules.append(output_layer)
        if risk_calibration is not None:
            readout_modules.append(risk_calibration)
        # 收集 network 的参数作为 backbone
        network = getattr(model, "network", None)
        if network is not None:
            backbone_modules.append(network)
        readout_param_ids = set()
        for module in readout_modules:
            for param in module.parameters():
                if param.requires_grad:
                    readout_params.append(param)
                    readout_param_ids.add(id(param))
        for module in backbone_modules:
            for param in module.parameters():
                if param.requires_grad and id(param) not in readout_param_ids:
                    backbone_params.append(param)
        # 收集其余未归类的可训练参数到 backbone
        for param in model.parameters():
            if param.requires_grad and id(param) not in readout_param_ids:
                already_in_backbone = any(id(param) == id(bp) for bp in backbone_params)
                if not already_in_backbone:
                    backbone_params.append(param)
        param_groups = []
        if readout_params:
            param_groups.append({"params": readout_params, "lr": readout_lr})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr})
        if not param_groups:
            raise ValueError("model does not expose trainable parameters")
        if optimizer_name == "adam":
            # v2 fair alignment (audit 2026-04-22): mirror Liquid foreach=False, fused=False,
            # ensuring both trainers walk the same Adam update path (no foreach/fused fast path).
            return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay, foreach=False, fused=False)
        if optimizer_name == "sgd":
            return torch.optim.SGD(param_groups, lr=lr, weight_decay=weight_decay)
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    params = list(model.parameters())
    if not params:
        raise ValueError("model does not expose trainable parameters")
    if optimizer_name == "adam":
        # v2 fair alignment (audit 2026-04-22): mirror Liquid foreach=False, fused=False.
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, foreach=False, fused=False)
    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer: {optimizer_name}")


def _resolve_runtime_device(device_request: str) -> tuple[torch.device, str]:
    """解析运行时设备请求，返回实际设备和设备名。

    参数:
    `device_request` 是设备请求字符串，支持 "auto"、"cuda"、"cuda:N" 和 "cpu"。

    返回值:
    返回 (torch.device, 设备名字符串) 元组。当请求 CUDA 但不可用时回退到 CPU。
    """
    request = str(device_request or DEVICE_CPU).strip().lower()
    if request == DEVICE_AUTO:
        if cuda_runtime_usable():
            return torch.device(DEVICE_CUDA), DEVICE_CUDA
        return torch.device(DEVICE_CPU), DEVICE_CPU
    if request == DEVICE_CPU:
        return torch.device(DEVICE_CPU), DEVICE_CPU
    if request == DEVICE_CUDA:
        if cuda_runtime_usable():
            return torch.device(DEVICE_CUDA), DEVICE_CUDA
        return torch.device(DEVICE_CPU), DEVICE_CPU
    if request.startswith(DEVICE_CUDA_PREFIX):
        if cuda_runtime_usable():
            return torch.device(request), request
        return torch.device(DEVICE_CPU), DEVICE_CPU
    raise ValueError(f"unsupported device request: {device_request!r}; expected 'auto', 'cpu', 'cuda' or 'cuda:N'")


def _resolve_amp_state(train_cfg: Mapping[str, Any], runtime_device: str) -> dict[str, Any]:
    """解析自动混合精度（AMP）配置。

    参数:
    `train_cfg` 是训练配置映射。
    `runtime_device` 是当前运行时设备名。

    返回值:
    返回包含 requested、enabled 和 dtype_name 三个键的字典。
    """
    amp_requested = train_cfg.get("amp_enabled", "auto")
    amp_dtype = str(train_cfg.get("amp_dtype", "auto")).strip().lower()
    if runtime_device.startswith("cuda"):
        if amp_requested == "auto":
            enabled = True
        elif isinstance(amp_requested, bool):
            enabled = amp_requested
        elif is_string_like(amp_requested):
            normalized = str(amp_requested).strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                enabled = True
            elif normalized in {"false", "0", "no", "off"}:
                enabled = False
            else:
                raise ValueError(f"amp_enabled must be 'auto' or bool-like, got {amp_requested!r}")
        else:
            raise ValueError(f"amp_enabled must be 'auto' or bool-like, got {amp_requested!r}")
        if not enabled:
            return {"requested": amp_requested, "enabled": False, "dtype_name": None}
        dtype_name = "float16"
        if amp_dtype in {"bfloat16", "bf16"}:
            dtype_name = "bfloat16"
        return {"requested": amp_requested, "enabled": True, "dtype_name": dtype_name}
    return {"requested": amp_requested, "enabled": False, "dtype_name": None}


def _predict_batch(
    model: Any,
    batch_or_metadata: list[dict[str, Any]] | torch.Tensor,
    *,
    modalities: list[str] | None = None,
) -> torch.Tensor:
    """对一批样本执行前向预测，返回投影后的四头批次张量。

    参数:
    `model` 是当前训练的模型实例。
    `batch_or_metadata` 可以是元数据列表或已准备好的序列批次张量。
    `modalities` 是批次对应的模态列表（张量输入时必须提供）。

    返回值:
    返回形状为 (batch, 4) 的预测张量，已经过 _project_train_outputs 投影。
    """
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
            sequence_tensor, normalized_window = build_lstm_sequence_tensor(
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
        # 应用风险校准模块，与 Liquid 训练路径和 LSTM 推理路径对齐。
        # risk_calibration 对原始 risk logit 执行 softplus(a)*raw+b -> sigmoid，
        # 结果已在 (0,1) 区间，再做线性映射到协议区间。
        raw_risk = raw_outputs[row_index][1]  # 取原始 risk logit（未经 sigmoid）。
        if hasattr(model, "risk_calibration") and model.risk_calibration is not None:
            calibrated_risk = model.risk_calibration(raw_risk.reshape(()))  # 校准：softplus(a)*raw+b -> sigmoid。
        else:
            calibrated_risk = torch.sigmoid(raw_risk.reshape(()))  # 无校准模块时退化为简单 sigmoid。
        risk_span = BRIDGE_THRESHOLDS["risk_max"] - BRIDGE_THRESHOLDS["risk_min"]
        if risk_span != 1.0 or BRIDGE_THRESHOLDS["risk_min"] != 0.0:
            calibrated_risk = calibrated_risk * risk_span + BRIDGE_THRESHOLDS["risk_min"]
        projected["risk"] = calibrated_risk  # 用校准后的 risk 覆盖。
        # 模态输出合约：非当前模态的 scaling 强制为 1.0，与推理侧保持一致。
        sample_modality = resolved_modalities[row_index]
        if sample_modality == MODALITY_UWB:
            projected["vio_scaling"] = projected["bias"].new_tensor(1.0)  # 保持与模型输出同设备。
        elif sample_modality == MODALITY_VIO:
            projected["uwb_scaling"] = projected["bias"].new_tensor(1.0)  # 保持与模型输出同设备。
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
    """计算单个窗口的监督损失（不含辅助损失）。

    注意：此函数仅计算监督损失，不包含辅助损失（calibration、L2、gate L1）。
    训练时使用 _compute_batch_loss 计算完整损失（监督+辅助）。
    此函数主要用于诊断和探针监控。

    参数:
    `model` 是当前训练的模型实例。
    `window_tensor` 是结构化窗口映射。
    `target_outputs` 是目标中间量字典。
    `bias_huber_delta` 是 bias 头 Huber 损失的 delta 参数。
    `scaling_log_eps` 是 scaling 转对数前的数值稳定项。

    返回值:
    返回标量损失张量。
    """
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
    """计算一批样本的总损失（监督损失 + 辅助损失）。

    参数:
    `model` 是当前训练的模型实例。
    `target_batch` 是目标批次张量。
    `metadata` 是窗口元数据列表。
    `sample_weights` 是可选的样本权重列表。
    `loss_weights` 是损失权重字典，None 时使用模块级常量默认值。
    `bias_huber_delta` 是 bias 头 Huber 损失的 delta 参数。
    `scaling_log_eps` 是 scaling 转对数前的数值稳定项。

    返回值:
    返回标量总损失张量。
    """
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
    auxiliary_terms = _compute_lstm_auxiliary_loss_terms(
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
    """计算同一序列长度组的批次损失。

    参数:
    `model` 是当前训练的模型实例。
    `samples` 是同长度组的材料化样本列表。
    `loss_weights` 是损失权重字典，None 时使用模块级常量默认值。
    `bias_huber_delta` 是 bias 头 Huber 损失的 delta 参数。
    `scaling_log_eps` 是 scaling 转对数前的数值稳定项。

    返回值:
    返回标量损失张量。
    """
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
    """收集单个批次的损失诊断信息。

    参数:
    `prediction_batch` 是预测批次。
    `target_batch` 是目标批次。
    `modalities` 是模态列表。
    `sample_weights` 是样本权重列表。
    `seq_lens` 是序列长度列表。
    `bias_huber_delta` 是 bias 头 Huber 损失的 delta 参数。
    `scaling_log_eps` 是 scaling 转对数前的数值稳定项。

    返回值:
    返回包含损失分子、分母、均值、各头损失等诊断信息的字典。
    """
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
    """汇总多个批次的各头损失统计。

    参数:
    `batches` 是批次诊断列表。
    `include_auxiliary` 是可选的辅助损失映射，会被追加到汇总中。

    返回值:
    返回以输出头名为键、包含 mean 和 variance 的字典。
    """
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
    """汇总一轮训练的损失诊断信息。

    参数:
    `split` 是数据集划分名（"train" 或 "val"）。
    `epoch_index` 是当前轮次编号。
    `batches` 是该轮所有批次的诊断列表。
    `selection_score` 是可选的加权选择分数。
    `auxiliary_summary` 是可选的辅助损失汇总。

    返回值:
    返回包含 split、epoch_index、mean_loss 等字段的轮次诊断字典。
    """
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
        epoch_payload["auxiliary_losses"] = {str(key): coerce_finite_scalar(value, name=f"auxiliary_losses.{key}") for key, value in auxiliary_summary.items()}
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
    """收集一轮训练的预测-目标快照，用于跟踪训练进展。

    参数:
    `model` 是当前训练的模型实例。
    `split` 是数据集划分名。
    `epoch_index` 是当前轮次编号。
    `cached_windows` 是缓存的材料化窗口列表。
    `batch_diagnostics` 是该轮所有批次的诊断列表。
    `mean_loss` 是该轮平均损失。
    `selection_score` 是加权选择分数。

    返回值:
    返回包含预测均值、目标均值、探针批次等字段的快照字典。
    """
    sample_count = int(prediction_target_summary.get("sample_count", 0) or 0)
    active_sample_count_by_head = {
        key: int((prediction_target_summary.get("active_sample_count_by_head", {}) or {}).get(key, 0) or 0)
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
    probe_batch_rows = []
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
                # D5/D7：与 liquid/trainer.py 同名函数口径对齐，统一走 coerce_finite_scalar 中心入口，
                # 拒绝 NaN/Inf 静默穿透至 snapshot（裸 float() 会把 nan/inf 静默写入消费者日志）。
                "prediction_by_head": {
                    key: coerce_finite_scalar(prediction_row[key_index], name=f"probe_batch.prediction.{key}")
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
                "target_by_head": {
                    key: coerce_finite_scalar(target_row[key_index], name=f"probe_batch.target.{key}")
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
        # D5/D7：epoch 级标量走 coerce_finite_scalar 中心入口，拒绝 NaN/Inf 静默穿透至 snapshot，
        # 与 _summarize_run 的 train_loss/val_loss/selection_score 口径一致（D7）。
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
    """对固定探针样本执行预测，用于跨轮次跟踪预测变化。

    参数:
    `model` 是当前训练的模型实例。
    `cached_windows` 是缓存的材料化窗口列表。

    返回值:
    返回探针样本的预测-目标-误差详情列表。
    """
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
                # D5/D7：与 liquid/trainer.py 同名函数口径对齐，统一走 coerce_finite_scalar 中心入口，拒绝 NaN/Inf 静默穿透。
                prediction_rows[sample_index] = [
                    coerce_finite_scalar(value, name=f"fixed_probe_prediction[{sample_index}][{key_index}]")
                    for key_index, value in enumerate(prediction_batch[row_offset].detach().cpu().tolist())
                ]
    fixed_probe_rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(cached_windows):
        prediction_row = prediction_rows[sample_index]
        if prediction_row is None:
            raise RuntimeError("fixed probe prediction collection left an empty row")
        # D5/D7：与 liquid/trainer.py 同名函数口径对齐，统一走 coerce_finite_scalar 中心入口，拒绝 NaN/Inf 静默穿透。
        target_row = [
            coerce_finite_scalar(value, name=f"fixed_probe_target[{sample_index}][{key_index}]")
            for key_index, value in enumerate(sample[1].detach().cpu().tolist())
        ]
        modality = _resolve_current_modality(sample[0])
        active_keys = list(_ACTIVE_HEADS_BY_MODALITY[modality])
        fixed_probe_rows.append(
            {
                "sample_index_in_split": sample_index,
                "modality": modality,
                "seq_len": int(sample[2]),
                "sample_weight": coerce_finite_scalar(sample[3], name=f"fixed_probe_sample_weight[{sample_index}]"),
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
    """评估模型在验证/训练集上的损失。

    参数:
    `model` 是当前训练的模型实例。
    `val_windows` 是验证/训练窗口列表。
    `epoch_index` 是当前轮次编号。
    `split` 是数据集划分名（"train" 或 "val"）。
    `collect_diagnostics` 是否收集详细诊断信息。
    `loss_weights` 是损失权重字典，None 时使用模块级常量默认值。
    `selection_weights` 是各输出头的选择权重，None 时使用 _SELECTION_WEIGHTS。
    `bias_huber_delta` 是 bias 头 Huber 损失的 delta 参数。
    `scaling_log_eps` 是 scaling 转对数前的数值稳定项。

    返回值:
    collect_diagnostics=False 时返回标量损失；
    collect_diagnostics=True 时返回 (损失, 批次诊断列表, 轮次快照) 三元组。
    """
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
        auxiliary_terms = _compute_lstm_auxiliary_loss_terms(
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
            key: (prediction_sum_all_samples[key] / float(sample_count)) if sample_count > 0 else None
            for key in _OUTPUT_KEYS
        },
        "target_mean_all_samples": {
            key: (target_sum_all_samples[key] / float(sample_count)) if sample_count > 0 else None
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
    """递归将张量移至 CPU，确保 checkpoint 与 weights_only=True 加载兼容。

    参数:
    `obj` 是待转换的对象，可以是张量、字典、列表或基本类型。

    返回值:
    返回所有嵌套张量均在 CPU 上的深拷贝。
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return type(obj)({k: _to_cpu_recursive(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_recursive(v) for v in obj)
    return obj


def _build_checkpoint_payload(
    *,
    model: Any,
    trainer_state: dict[str, Any],
    optimizer: Any,
    seed_report: dict[str, Any],
    best_epoch: int,
    best_loss: float,
    train_window_count: int,
    val_window_count: int,
) -> dict[str, Any]:
    """构建检查点保存载荷。

    所有张量强制移至 CPU 后再保存，确保与 model_factory 的
    ``weights_only=True`` 加载路径兼容。

    参数:
    `model` 是当前训练的模型实例。
    `trainer_state` 是训练器状态字典。
    `optimizer` 是当前优化器实例。
    `seed_report` 是随机种子报告。
    `best_epoch` 是最佳轮次编号。
    `best_loss` 是最佳损失值。
    `train_window_count` 是训练窗口数量。
    `val_window_count` 是验证窗口数量。

    返回值:
    返回可直接传给 torch.save 的检查点字典。
    """
    network_cfg = dict(trainer_state["network"])
    network_cfg["hidden_dim"] = int(model.network.hidden_dim)
    network_cfg["input_dim"] = int(model.network.input_dim)
    network_cfg["num_layers"] = int(model.network.num_layers)
    network_cfg["dropout"] = float(model.network.dropout)
    network_cfg.setdefault("output_heads", list(_OUTPUT_KEYS))
    return {
        "checkpoint_format": "lstm_real_v1",
        "model_name": trainer_state["model_name"],
        "model_cfg": {
            "feature_order": list(trainer_state["feature_order"]),
            "window": dict(trainer_state["window"]),
            "network": network_cfg,
        },
        "train_cfg": {
            "train": dict(trainer_state["train"]),
            "seed": trainer_state["seed"],
            "deterministic": trainer_state["deterministic"],
        },
        "optimizer": dict(trainer_state["optimizer"]),
        "optimizer_state": _to_cpu_recursive(optimizer.state_dict()),
        "seed_report": dict(seed_report),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "train_window_count": int(train_window_count),
        "val_window_count": int(val_window_count),
        "model_state": _to_cpu_recursive(model.state_dict()),
    }


class LSTMTrainer:
    """LSTM 训练器，封装训练配置读取和训练流程调用。

    从 train_cfg 的 train 子节读取 loss_weights、lr_layer、bias_huber_delta、
    scaling_log_eps、gradient_clip_max_norm 等配置，存为实例属性。
    对外暴露的方法使用实例属性替代模块级常量，确保训练流程受 YAML 配置驱动。
    """

    def __init__(self, train_section: Mapping[str, Any]) -> None:
        """从训练配置的 train 子节读取参数，存为实例属性。

        参数:
        `train_section` 是 train_cfg 中 train 子节的映射。
        """
        loss_weights_cfg = train_section.get("loss_weights") or {}
        self._loss_weights: dict[str, float] = {
            "w_cal": coerce_finite_scalar(
                loss_weights_cfg.get("w_cal", _LSTM_CALIBRATION_WEIGHT),
                name="loss_weights.w_cal", min_value=0.0),
            "lambda_l2": coerce_finite_scalar(
                loss_weights_cfg.get("lambda_l2", _GLOBAL_L2_WEIGHT),
                name="loss_weights.lambda_l2", min_value=0.0),
            "lambda_l1": coerce_finite_scalar(
                loss_weights_cfg.get("lambda_l1", _LSTM_GATE_L1_WEIGHT),
                name="loss_weights.lambda_l1", min_value=0.0),
            "lambda_mono": coerce_finite_scalar(
                loss_weights_cfg.get("lambda_mono", _LSTM_MONO_WEIGHT),
                name="loss_weights.lambda_mono", min_value=0.0),
        }
        # 选择权重使用与 Liquid trainer 一致的硬编码 _SELECTION_WEIGHTS，保证模型选择口径公平。
        # w_bias/w_risk/w_uwb/w_vio 是协议推荐的损失权重，不是选择权重，不应覆盖 _SELECTION_WEIGHTS。
        self._selection_weights: dict[str, float] = dict(_SELECTION_WEIGHTS)
        # 仅保留 LSTM 支持的分层学习率键，过滤 backbone_unfreeze_ramp 等 Liquid 专用键。
        _raw_lr_layer = train_section.get("lr_layer") or {}
        self._lr_layer: dict[str, Any] = {
            key: value for key, value in dict(_raw_lr_layer).items()
            if key in {"readout", "backbone"}
        }
        self._bias_huber_delta: float = coerce_finite_scalar(
            train_section.get("bias_huber_delta", _BIAS_HUBER_DELTA),
            name="bias_huber_delta", min_value=0.0)
        self._scaling_log_eps: float = coerce_finite_scalar(
            train_section.get("scaling_log_eps", _SCALING_LOG_EPS),
            name="scaling_log_eps", min_value=0.0)
        self._gradient_clip_max_norm: float = coerce_finite_scalar(
            train_section.get("gradient_clip_max_norm", 1.0),
            name="gradient_clip_max_norm", min_value=0.0, inclusive=False)
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
        """评估模型，使用实例属性中的配置。"""
        return _evaluate_model(
            model, val_windows,
            epoch_index=epoch_index,
            split=split,
            collect_diagnostics=collect_diagnostics,
            loss_weights=self._loss_weights,
            selection_weights=self._selection_weights,
            bias_huber_delta=self._bias_huber_delta,
            scaling_log_eps=self._scaling_log_eps,
        )

    def _build_optimizer(self, model: Any, optimizer_cfg: Mapping[str, Any]):
        """构建优化器，使用实例属性中的 lr_layer 配置。"""
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
        """执行一轮训练，使用实例属性中的配置。"""
        return train_one_epoch(
            train_windows, val_windows, epoch_index,
            model=model,
            optimizer=optimizer,
            gradient_clip_max_norm=self._gradient_clip_max_norm,
            loss_weights=self._loss_weights,
            bias_huber_delta=self._bias_huber_delta,
            scaling_log_eps=self._scaling_log_eps,
        )


def build_trainer_state(train_cfg: dict) -> dict:
    """从训练配置字典构建训练器状态。

    参数:
    `train_cfg` 是训练配置字典，包含 train、name、seed 等字段。

    返回值:
    返回标准化后的训练器状态字典，包含所有训练参数。

    失败条件:
    配置字段类型或值不合法时抛出异常。
    """
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
    }, "LSTM build_trainer_state 入口参数")
    if not isinstance(train_cfg, dict):
        raise TypeError("train_cfg must be a dictionary.")
    train_section = dict(train_cfg.get("train") or {})
    optimizer_name = train_section.get("optimizer", "adam")
    lr = train_section.get("lr", 0.001)
    weight_decay = train_section.get("weight_decay", 0.0)
    epochs = train_section.get("epochs", 1)
    patience = train_section.get("patience", epochs)
    epoch_candidate_stride = train_section.get("epoch_candidate_stride", 1)
    seed = train_section.get("seed", train_cfg.get("seed", 0))
    deterministic = train_section.get("deterministic", train_cfg.get("deterministic", True))
    if not is_string_like(optimizer_name) or not str(optimizer_name).strip():
        raise ValueError("train.optimizer must be a non-empty string.")
    if not is_numeric(lr):
        raise TypeError("train.lr must be numeric.")
    if float(lr) <= 0.0:
        raise ValueError("train.lr must be positive.")
    if not is_numeric(weight_decay):
        raise TypeError("train.weight_decay must be numeric.")
    if float(weight_decay) < 0.0:
        raise ValueError("train.weight_decay must be non-negative.")
    if not is_integer(epochs):
        raise TypeError("train.epochs must be an integer.")
    if epochs < 1:
        raise ValueError("train.epochs must be at least 1.")
    if not is_integer(patience):
        raise TypeError("train.patience must be an integer.")
    if patience < 1:
        raise ValueError("train.patience must be at least 1.")
    if int(patience) != int(epochs):
        raise ValueError(
            "lstm_ekf single-phase baseline does not implement early stopping; "
            "train.patience must equal train.epochs to keep the training budget explicit."
        )
    if not is_integer(epoch_candidate_stride):
        raise TypeError("train.epoch_candidate_stride must be an integer.")
    if epoch_candidate_stride < 1:
        raise ValueError("train.epoch_candidate_stride must be at least 1.")
    if not is_integer(seed):
        raise TypeError("train seed must be an integer.")
    if not is_bool_like(deterministic):
        raise TypeError("train deterministic flag must be a boolean.")
    model_name = train_cfg.get("name", "lstm_ekf")
    if not is_string_like(model_name) or not str(model_name).strip():
        raise ValueError("name must be a non-empty string.")
    output_root = train_cfg.get("output_root")
    if output_root is None:
        project_root = train_cfg.get("project_root")
        if project_root is None:
            project_root = find_project_root()
        output_root = Path(project_root) / "outputs" / "train_lstm"
    else:
        output_root = Path(output_root)
    tail_selection_observation_coeff = coerce_finite_scalar(
        train_section.get("tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
        name="train.tail_selection_observation_coeff",
        min_value=0.0,
    )
    train_section["tail_selection_observation_coeff"] = tail_selection_observation_coeff

    # 偷懒审视 Round 4 真修 (用户铁律 #14): lstm trainer 从未实现 LR scheduler, 全程 lr 常量.
    # 加 torch.optim.lr_scheduler.CosineAnnealingLR — lr 平滑退火 lr(t)=eta_min+0.5*(lr0-eta_min)*(1+cos(t*pi/T_max)).
    # 默认 T_max=epochs (即 160), eta_min=1e-6 (PyTorch 标准推荐值).
    # 不破坏 patience==epochs 严格校验: lr_scheduler.T_max 与 train.epochs 是同一含义 (训练总轮次), trainer 实际使用时
    # 如果 lr_scheduler.T_max 未显式提供, fallback 到 trainer_state["epochs"]; 显式提供则必须等于 epochs.
    # ── audit #20 H2 + M2 真修 (主线程决策 2026-07): 原方案用 ReduceLROnPlateau(Plateau 检测),
    # 因 threshold='rel' (默认) 实际 threshold = 1e-4 * best_loss = 1e-4 * 0.022 = 2.2e-6,
    # val_loss 每 epoch 下降 > 2.2e-6 即视为"仍在改善", scheduler 永不 trigger, lr 0 衰减 (silent bug).
    # 即使按 H2 改 threshold='abs'+2e-3 仍存 plateau 检测 silent 风险 (train 动力学不稳时仍可能漏发).
    # 主线程决策 (M2 简单方案): 直接换 CosineAnnealingLR T_max=epochs eta_min=1e-6 — lr 平滑从 1e-4 → 1e-6
    # 按 1e-4*cos(epoch*pi/epochs) 退火, 无 plateau 检测, 无 silent bug 风险, 同时收敛速率更可控.
    lr_scheduler_cfg = dict(train_section.get("lr_scheduler") or {})
    lr_scheduler_enabled = bool(lr_scheduler_cfg.get("enabled", False))
    lr_scheduler_t_max = int(lr_scheduler_cfg.get("T_max", epochs))  # 默认 T_max = epochs
    lr_scheduler_eta_min = float(lr_scheduler_cfg.get("eta_min", 1e-6))
    if lr_scheduler_enabled:
        if lr_scheduler_t_max < 1:
            raise ValueError(
                "train.lr_scheduler.T_max must be >= 1 when enabled; "
                f"got {lr_scheduler_t_max}."
            )
        if lr_scheduler_t_max != int(epochs):
            # T_max 必须 = epochs (CosineAnnealingLR 退火终点与训练终点对齐), 否则后期 lr 不再衰减或不到 eta_min.
            raise ValueError(
                "train.lr_scheduler.T_max must equal train.epochs when enabled; "
                f"got T_max={lr_scheduler_t_max}, epochs={epochs}."
            )
        if lr_scheduler_eta_min <= 0.0:
            raise ValueError(
                "train.lr_scheduler.eta_min must be > 0 when enabled; "
                f"got {lr_scheduler_eta_min}."
            )

    return {
        "model_name": model_name,
        "optimizer": {
            "name": optimizer_name.strip().lower(),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        },
        "epochs": int(epochs),
        "patience": int(patience),
        "epoch_candidate_stride": int(epoch_candidate_stride),
        "seed": seed,
        "deterministic": deterministic,
        "device": str(train_cfg.get("device") or "cpu"),
        "output_root": output_root,
        "feature_order": list(train_cfg.get("feature_order") or []),
        "window": dict(train_cfg.get("window") or {}),
        "network": dict(train_cfg.get("network") or {}),
        "train": train_section,
        "tail_selection_observation_coeff": tail_selection_observation_coeff,
        "lr_scheduler": {
            "enabled": lr_scheduler_enabled,
            "T_max": lr_scheduler_t_max,
            "eta_min": lr_scheduler_eta_min,
        },
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
    """执行一轮训练并返回平均训练损失。

    参数:
    `train_windows` 是训练窗口列表。
    `val_windows` 是验证窗口列表（本函数不使用，但保持接口一致）。
    `epoch_index` 是当前轮次编号，从 1 开始。
    `model` 是当前训练的模型实例。
    `optimizer` 是当前优化器实例。
    `gradient_clip_max_norm` 是梯度裁剪的最大范数。
    `loss_weights` 是损失权重字典，None 时使用模块级常量默认值。
    `bias_huber_delta` 是 bias 头 Huber 损失的 delta 参数。
    `scaling_log_eps` 是 scaling 转对数前的数值稳定项。

    返回值:
    返回该轮的平均训练损失。

    失败条件:
    epoch_index 不是正整数、model 或 optimizer 缺失、窗口为空时抛出异常。
    """
    if not is_integer(epoch_index):
        raise TypeError("epoch_index must be an integer.")
    if epoch_index < 1:
        raise ValueError("epoch_index must be at least 1.")
    if model is None:
        raise ValueError("model must be provided for real LSTM training")
    if optimizer is None:
        raise ValueError("optimizer must be provided for real LSTM training")
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
                model, batch_samples,
                loss_weights=loss_weights,
                bias_huber_delta=bias_huber_delta,
                scaling_log_eps=scaling_log_eps,
            )
            loss_value.backward()
            # 梯度裁剪：防止极端 NLOS 样本导致梯度爆炸。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
            optimizer.step()
            loss_sum += float(loss_value.detach().item())
            loss_count += 1
    return loss_sum / float(loss_count)


def train_model(train_windows, val_windows, train_cfg, *, model_factory=None):
    """LSTM 训练入口：完成从配置到训练再到落盘的完整流程。

    参数:
    `train_windows` 是训练窗口列表。
    `val_windows` 是验证窗口列表。
    `train_cfg` 是训练配置字典。
    `model_factory` 是可选的模型工厂可调用对象，签名为 (model_name, model_cfg) -> model。
        默认为 None 时懒加载 liquidloc.factories.model_factory.create_model。
        注入此参数可解除 models → factories 的编译期逆向依赖。

    返回值:
    返回 (最佳检查点路径, 训练报告字典) 元组。

    失败条件:
    窗口为空、配置不合法或模型构建失败时抛出异常。
    """
    if train_windows is None:
        raise ValueError("train_windows must not be None.")
    if val_windows is None:
        raise ValueError("val_windows must not be None.")
    if isinstance(train_windows, (str, bytes)):
        raise TypeError("train_windows must be an iterable of windows.")
    if isinstance(val_windows, (str, bytes)):
        raise TypeError("val_windows must be an iterable of windows.")
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

    from liquidloc.common.tee_logger import print_dict
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
    }, "LSTM train_model 入口参数")

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
    if model_factory is None:  # 未注入工厂时懒加载默认实现，保持向后兼容。
        from liquidloc.factories.model_factory import create_model as _default_create_model  # noqa: WPS433
        model_factory = _default_create_model
    model = model_factory(trainer_state["model_name"], model_cfg)
    # 准则 27：LSTM 模型加载后挂载激活统计 hook，训练过程持续收集 5 头统计。
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
    lstm_trainer = LSTMTrainer(trainer_state["train"])
    optimizer = lstm_trainer._build_optimizer(model, trainer_state["optimizer"])
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
        "checkpoint_format": "lstm_real_v1",
        "train": [],
        "val": [],
    }
    epoch_predictions_payload = {
        "model_name": trainer_state["model_name"],
        "checkpoint_format": "lstm_real_v1",
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

    # 偷懒审视 Round 4 真修 (用户铁律 #14): 装上 CosineAnnealingLR lr_scheduler.
    # audit #20 H2 + M2 真修 (主线程决策 2026-07): 原 ReduceLROnPlateau 因 threshold='rel' 永不 trigger (silent bug).
    # 改用 CosineAnnealingLR — lr 按 1e-4*cos(epoch*pi/T_max) 平滑退火到 eta_min, 不依赖 plateau 检测.
    # 不启用时 (lr_scheduler.enabled=False) scheduler=None, 全程常量 lr (向后兼容 round1-3 行为).
    lr_scheduler_state = trainer_state.get("lr_scheduler", {})
    lr_scheduler_enabled = bool(lr_scheduler_state.get("enabled", False))
    lr_scheduler = None
    if lr_scheduler_enabled:
        # T_max 默认 = trainer_state["epochs"] (与训练总轮次对齐), 校验已在 build_trainer_state 中保证 T_max==epochs.
        lr_scheduler_t_max = int(lr_scheduler_state.get("T_max", trainer_state["epochs"]))
        lr_scheduler_eta_min = float(lr_scheduler_state.get("eta_min", 1e-6))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=lr_scheduler_t_max,
            eta_min=lr_scheduler_eta_min,
        )
        print(
            f"[LSTM训练][lr_scheduler] enabled CosineAnnealingLR "
            f"T_max={lr_scheduler_t_max} "
            f"eta_min={lr_scheduler_eta_min}",
            flush=True,
        )

    for epoch_index in range(1, trainer_state["epochs"] + 1):
        train_loss = lstm_trainer.train_one_epoch(
            train_windows,
            val_windows,
            epoch_index,
            model=model,
            optimizer=optimizer,
        )
        train_eval_loss, train_batches, train_epoch_snapshot = lstm_trainer._evaluate_model(
            model,
            train_windows,
            epoch_index=epoch_index,
            split="train",
            collect_diagnostics=True,
        )
        val_loss, val_batches, val_epoch_snapshot = lstm_trainer._evaluate_model(
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

        # 每轮输出训练进度与预测均值对比
        print(f"[LSTM训练] 第 {epoch_index}/{trainer_state['epochs']} 轮  "
              f"训练损失={train_loss:.6f}  验证损失={val_loss:.6f}  "
              f"选择分数={selection_score:.6f}", flush=True)
        for split_name, snapshot in [("train", train_epoch_snapshot), ("val", val_epoch_snapshot)]:
            pred_mean = snapshot.get("prediction_mean_active_samples", {})
            tgt_mean = snapshot.get("target_mean_active_samples", {})
            split_label = "训练集" if split_name == "train" else "验证集"
            parts = [f"{k}: 预测={pred_mean.get(k):.4f} 真值={tgt_mean.get(k):.4f}"
                     for k in _OUTPUT_KEYS if pred_mean.get(k) is not None]
            print(f"  [{split_label}] " + " | ".join(parts), flush=True)

        if best_selection_score is None or selection_score < best_selection_score:
            best_selection_score = selection_score
            best_loss = float(val_loss)
            best_epoch = epoch_index
            checkpoint_payload["best_epoch"] = best_epoch
            checkpoint_payload["best_loss"] = best_loss
            torch.save(checkpoint_payload, best_ckpt)

        # 偷懒审视 Round 4 真修 (用户铁律 #14) + audit #20 H2+M2 (主线程决策 2026-07):
        # epoch 末 CosineAnnealingLR 调度 lr — 不需要 val_loss 参数, 直接 step() 按 T_max 退火.
        # lr(t) = eta_min + 0.5*(lr0 - eta_min)*(1 + cos(t*pi/T_max)), t=epoch_index.
        # 不启用时 lr_scheduler=None (向后兼容 round1-3 行为).
        if lr_scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step()
            new_lr = optimizer.param_groups[0]["lr"]
            # CosineAnnealingLR 每 epoch 都退火 (lr 单调下降), 打印任意 ep 的 lr 便于追踪.
            print(
                f"[LSTM训练][lr_scheduler] ep{epoch_index} lr 退火 "
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
        "best_epoch_in_range": bool(best_epoch is not None and 1 <= int(best_epoch) <= int(trainer_state["epochs"])),
        "best_epoch_matches_best_selection_score": bool(
            best_epoch is not None
            and best_selection_score is not None
            and int(best_epoch) == int(min(range(1, len(val_selection_scores) + 1), key=lambda idx: val_selection_scores[idx - 1]))
        ),
        "checkpoint_path_exists": bool(best_ckpt.is_file()),
        "early_stop_patience_matches_config": bool(int(trainer_state["patience"]) == int(trainer_state["epochs"])),
        "early_stopped": False,
    }
    # 审计先于 JSON 落盘：若训练发散产生 NaN/Inf，dumps_json_text(allow_nan=False) 会抛 ValueError，
    # 此时 training_stability_audit 已计算完成，可在异常处理中落盘审计结果。
    loss_diagnostics_path.write_text(
            dumps_json_text(diagnostics_payload),
        encoding="utf-8",
    )
    epoch_predictions_vs_targets_path.write_text(
            dumps_json_text(epoch_predictions_payload),
        encoding="utf-8",
    )
    training_stability_audit["diagnostics_path_exists"] = bool(loss_diagnostics_path.is_file())
    training_stability_audit["epoch_predictions_path_exists"] = bool(epoch_predictions_vs_targets_path.is_file())

    train_report = {
        "model_name": trainer_state["model_name"],
        "status": "trained",
        "checkpoint_format": "lstm_real_v1",
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
        # §13.6.2.8 cross-trainer parity declaration: 同 Liquid trainer 同步声明;
        # 与 _CROSS_TRAINER_PARITY_DECLARATION 内容字面对比后, 若存在不对称则双方
        # asymmetry_reason 必须完全一致字符串, 否则 _assert_cross_trainer_parity_declared_and_aligned fail-loud.
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
        "optimizer_weight_decay_applied": float(trainer_state.get("optimizer", {}).get("weight_decay", 0.0)),
        "risk_calibration_enabled": getattr(model, "risk_calibration", None) is not None,  # 从模型实例动态读取，反映当前代码事实。
        "tail_selection_observation_coeff": coerce_finite_scalar(
            trainer_state.get("tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
            name="tail_selection_observation_coeff",
            min_value=0.0,
        ),
        # L5 追溯字段根因修复：显式记录标签构造模式，对齐 docs/loss_function.md §实现约束 8。
        # LSTM 与 Liquid 共享同一标签链与口径，故采用与 Liquid 相同的默认值，保证公平对比可追溯。
        "risk_label_mode": RISK_LABEL_MODE_DEFAULT,  # risk 标签构造模式（当前默认 alignment proxy）。
        "uwb_scaling_label_mode": UWB_SCALING_LABEL_MODE_DEFAULT,  # uwb_scaling 标签构造模式（当前默认 teacher-free 启发式）。
        "vio_scaling_label_mode": VIO_SCALING_LABEL_MODE_DEFAULT,  # vio_scaling 标签构造模式（当前默认 teacher-free 启发式）。
        "training_stability_audit": training_stability_audit,
        "output_root": str(output_root),
        "best_ckpt": str(best_ckpt),
        # 准则 34：持久化 train/val loss 曲线 + 过拟合监控（与 liquid 同口径）.
        "train_loss_curve": list(train_epoch_losses),
        "val_loss_curve": list(val_epoch_losses),
        "overfit_audit": _compute_lstm_overfit_audit(train_epoch_losses, val_epoch_losses),
    }
    report_path = reports_dir / f'{trainer_state["model_name"]}_train_report.json'
    train_report["report_path"] = str(report_path)
    report_path.write_text(
            dumps_json_text(train_report),
        encoding="utf-8",
    )
    # 训练完成后将模型移回 CPU，释放 GPU 显存，与 AGENTS.md 设备规则对齐。
    # 准则 27：脱钩激活统计 hook（避免影响推理路径），写入 train_report。
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
    model.network.to("cpu")
    if hasattr(model, "risk_calibration") and hasattr(model.risk_calibration, "to"):
        model.risk_calibration.to("cpu")
    for attr in ("_cached_train_windows", "_cached_val_windows"):
        if hasattr(model, attr):
            delattr(model, attr)
    return str(best_ckpt), train_report
