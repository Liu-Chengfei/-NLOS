from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from typing import Any, Mapping

from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
from liquidloc.common.validation import coerce_finite_scalar
from liquidloc.models.liquid.output_head import apply_liquid_modality_output_contract

from ._constants import (  # type: ignore[import]
    _BIAS_HUBER_DELTA,
    _GLOBAL_L2_WEIGHT,
    _LIQUID_CALIBRATION_WEIGHT,
    _LIQUID_GATE_L1_WEIGHT,
    _LIQUID_MONO_WEIGHT,
    _OUTPUT_KEYS,
    _RISK_CALIBRATION_BIN_COUNT,
    _SCALING_LOG_EPS,
    _SCALING_NEUTRAL_FLOOR,
    _SELECTION_WEIGHTS,
    _TAIL_SELECTION_MAX_WEIGHT,
    _TAIL_SELECTION_OBSERVATION_COEFF,
    _TAIL_SELECTION_TAIL_COEFF,
)
from ._data_utils import (  # type: ignore[import,misc]
    _build_head_mask_tensor,
    _build_sample_weight_tensor,
    _normalize_liquid_train_outputs,
    _normalize_liquid_train_outputs_batch,
    _resolve_current_modality,
)

# Local constants used only within this module
_LIQUID_SMOOTH_WEIGHT = 1e-3  # Default weight for smooth regularization when enabled
_POSITION_MSE_SELECTION_WEIGHT = 0.50  # §24.1 位置误差主项选模权重（占 50%，高于各辅头）

"""
 Loss Compute submodule.
"""

def _project_train_risk(raw_risk: torch.Tensor, *, already_normalized: bool = False) -> torch.Tensor:
    """将原始风险输出投影到训练阶段实际使用的风险区间。

    训练risk 输出需要落到BRIDGE_THRESHOLDS["risk_min"] 。
    BRIDGE_THRESHOLDS["risk_max"] 的区间内。如risk 尚未归一。
    （即还是网络原始 logit），先做 sigmoid 压到 [0, 1]，再线性映。
    到目标区间；如果已经归一化（例如经过 risk_calibration 校准），
    则跳sigmoid，只做区间映射。

    协议层锚点（D2 分层边界）：本函数的线性映射+ clamp 公式与协议层
    ``liquidloc.protocol.risk_projection.project_risk_to_protocol_range`` 一致，
    该协议函数是公式唯一权威定义。本函数未直接委托协议层，原因有二：
    (1) 训练侧输入是张量，需要``torch.clamp`` 而非标量 ``min/max``。
    (2) 训练侧携带``already_normalized`` 分支持sigmoid 预归一化逻辑。
    属于训练链路特有语义。公式口径以协议层为锚点保持可追溯，不在此重述。
    公式定义。

    参数：
        raw_risk: 原始风险输出张量（标量）。
        already_normalized: 是否已经归一化到 [0, 1]。
            True 表示跳过 sigmoid，只做区间映射；
            False 表示先做 sigmoid 再映射。

    返回值：
        映射到训练区间的风险标量张量。
    """
    risk_span = BRIDGE_THRESHOLDS["risk_max"] - BRIDGE_THRESHOLDS["risk_min"]  # 先算训练区间跨度。
    risk = torch.as_tensor(raw_risk, dtype=torch.float32).reshape(())  # 统一成标量张量，便于后续映射。
    if not torch.isfinite(risk):  # NaN/Inf 会经 sigmoid 传播到loss，污染整个 batch 梯度。
        raise ValueError(f"raw_risk must be finite, got {risk.item()}")
    if not already_normalized:  # 未校准时按旧语义先做 sigmoid。
        risk = torch.sigmoid(risk) # 先把原始输出压到 [0, 1].
    if risk_span != 1.0 or BRIDGE_THRESHOLDS["risk_min"] != 0.0:  # 如果默认区间不是标准区间，就平移并缩放。
        risk = risk * risk_span + BRIDGE_THRESHOLDS["risk_min"]  # 把概率域映射到真实训练区间。
    # 与推理侧 _project_risk_to_protocol_range 保持一致：末尾 clamp 到协议区间。
    risk = torch.clamp(risk, BRIDGE_THRESHOLDS["risk_min"], BRIDGE_THRESHOLDS["risk_max"])
    return risk  # 返回训练阶段实际使用的风险。




def _semantic_loss_matrix(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> torch.Tensor:
    """按输出头语义分别计算逐元素损失矩阵。

    三个语义域使用不同的损失函数（§24.1 训练主损失要求：位置误差主项 MSE/Huber + 可选创新 NLL/辅助；本模块为观测侧增强头，主选模与全序仍以 raw 位置 RMSE 为准）。
    - bias: Huber 损失（δ 与评价主指标无关，训练专用），对极端 NLOS 偏置更鲁棒；
    - risk: 平方误差，做连续风险回归；
    - scaling (uwb_scaling + vio_scaling): 对数域平方误差，更符合乘性噪声语义。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
            列顺序为 bias, risk, uwb_scaling, vio_scaling。。
        target_batch: 目标值批次，形状态prediction_batch 相同。

    返回值：
        形状 (B, 4) 的、元素损失矩阵，每列对应一个输出头。
    """
    # ?LSTM 侧对齐：进入损失计算前先做有限、检查，
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
        ) # bias ?Huber，避免极端NLOS 样本主导。
    risk_loss = (prediction_batch[:, 1] - target_batch[:, 1]).pow(2)  # risk 继续做连续风险回归。
    # ?LSTM ?_semantic_loss_matrix 对齐：eps 放进 clamp_min 而非 log 内部。
    # 保证 LSTM ?Liquid 公平对对比scaling 损失数学等价（D7 公平性）。
    # Do not hard-clamp the prediction side here: Liquid scaling is already
    # projected to the neutral floor, and clamping at 1.0 would zero the
    # gradient exactly where the head initializes.
    predicted_scaling = prediction_batch[:, 2:] + scaling_log_eps
    target_scaling = torch.clamp_min(target_batch[:, 2:], _SCALING_NEUTRAL_FLOOR) + scaling_log_eps
    scaling_loss = (
        torch.log(predicted_scaling) - torch.log(target_scaling)
    ).pow(2)  # scaling 在对数域做回归，更符合乘性噪声语义。
    return torch.cat(
        (
            bias_loss.unsqueeze(1),
            risk_loss.unsqueeze(1),
            scaling_loss,
        ),
        dim=1,
    )




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
    """计算四头监督损失的加权分子和分母。

    先构造输出头掩码和样本权重，再与语义损失矩阵逐元素相乘，
    得到加权后的分子和分母，供外层做除法得到平均损失。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。
        modalities: 当前批次每个样本的模态名列表。
        sample_weights: 样本权重列表，None 时默认等权。

    返回值：
    二元（weighted_numerator, weighted_denominator)。
        - weighted_numerator: 加权损失分子标量张量。
        - weighted_denominator: 有效权重分母标量张量。
    """
    mask = _build_head_mask_tensor(modalities, reference_tensor=prediction_batch)  # 先构造输出头掩码。
    sample_weight_tensor = _build_sample_weight_tensor(sample_weights, reference_tensor=prediction_batch).unsqueeze(1)

    per_head_loss = _semantic_loss_matrix(prediction_batch, target_batch, bias_huber_delta=bias_huber_delta, scaling_log_eps=scaling_log_eps)

    weighted_mask = mask * sample_weight_tensor

    # IEEE 754 ?NaN * 0 = NaN，掩码为零的位置无法隔离 NaN。
    # ?torch.where 把掩码为零位置的 per_head_loss 显式置零。
    # 保证被屏蔽头（如 VIO 样本uwb_scaling）的 NaN/Inf 不会污染整个 batch 的损失。
    masked_per_head_loss = torch.where(
        weighted_mask > 0,
        per_head_loss,
        torch.zeros_like(per_head_loss),
    )
    return (masked_per_head_loss * weighted_mask).sum(), weighted_mask.sum()





def _masked_component_loss_stats(

    prediction_batch: torch.Tensor,

    target_batch: torch.Tensor,

    *,

    modalities: list[str],
    sample_weights: list[float] | None = None,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,

) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[str]]:
    """逐输出头统计加权损失分子、分母和濢、活头列表。

    ?_masked_supervised_loss_stats 不同，本函数按每个输出头
    单独统计，用于后续计算各头独立损失和选择分数。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。
        modalities: 当前批次每个样本的模态名列表。
        sample_weights: 样本权重列表，None 时默认等权。

    返回值：
    三元（component_numerators, component_denominators, active_keys)。
        - component_numerators: 每个濢、活头的加权损失分子字典；
        - component_denominators: 每个濢、活头的有效权重分母字典；
        - active_keys: 本批次真正参与损失的输出头名列表。
    """
    mask = _build_head_mask_tensor(modalities, reference_tensor=prediction_batch)

    sample_weight_tensor = _build_sample_weight_tensor(sample_weights, reference_tensor=prediction_batch)  # 再把样本权重转成张量。
    per_head_loss = _semantic_loss_matrix(prediction_batch, target_batch, bias_huber_delta=bias_huber_delta, scaling_log_eps=scaling_log_eps)  # 用按头语义一致的损失矩阵做统计。
    component_numerators: dict[str, torch.Tensor] = {}  # 每个输出头的加权损失分子。
    component_denominators: dict[str, torch.Tensor] = {}  # 每个输出头的有效权重分母。
    active_keys: list[str] = []  # 记录本批次真正参与损失的输出头。
    for index, key in enumerate(_OUTPUT_KEYS):  # 逐个输出头统计。
        head_mask = mask[:, index] * sample_weight_tensor  # 当前输出头的有效样本权重。
        if float(head_mask.sum().detach().item()) <= 0.0:  # 如果这个头没有有效样本，就跳过。
            continue  # 没有有效样本就不统计这个头。
        # 与_masked_supervised_loss_stats 一致：IEEE 754 中NaN * 0 = NaN锛。
        # 掩码为零位置霢torch.where 显式置零，防止被屏蔽样本NaN 污染该头统计。
        head_loss = torch.where(head_mask > 0, per_head_loss[:, index], torch.zeros_like(per_head_loss[:, index]))
        component_numerators[key] = (head_loss * head_mask).sum()  # 统计该头的加权分子。
        component_denominators[key] = head_mask.sum()  # 统计该头的有效分母。
        active_keys.append(key)  # 记录这个头确实参与了损失。
    # §24.1 执行点：实际计算位置误差 MSE 主损失（必须参与总损失与选模决策，非仅记录）。
    position_mse_scalar = _position_error_mse(prediction_batch, target_batch)  # 执行主损失位置误差计算。
    # 返回位置误差执行点值，供上游总损失与选模使用（§24.1 强制接入训练链）。
    return component_numerators, component_denominators, active_keys, position_mse_scalar  # 四元组：增加位置误差执行点。




def _masked_supervised_loss(
    prediction_batch: torch.Tensor,

    target_batch: torch.Tensor,

    *,

    modalities: list[str],
    sample_weights: list[float] | None = None,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,

) -> torch.Tensor:
    """返回整批四头监督项的加权语义损失。

    先调用_masked_supervised_loss_stats 得到分子和分母，
    再做除法得到平均损失。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。
        modalities: 当前批次每个样本的模态名列表。
        sample_weights: 样本权重列表，None 时默认等权。

    返回值：
        加权平均监督损失标量张量。
    """
    numerator, denominator = _masked_supervised_loss_stats(  # 先拿到分子和分母。
        prediction_batch,  # 预测批次。
        target_batch,  # 目标批次。
        modalities=modalities,  # 当前批次的模态列表。
        sample_weights=sample_weights,  # 当前批次的样本权重。
        bias_huber_delta=bias_huber_delta,
        scaling_log_eps=scaling_log_eps,
    )  # 统计结束。
    # ?LSTM ?_compute_batch_loss 对齐：分母做 clamp_min ?0 除，
    # 保证 LSTM ?Liquid 公平对比时除法数值稳定策略一致（D7 公平性）。
    return numerator / torch.clamp_min(denominator, scaling_log_eps)  # 用分子除以分母得到平均损失。




def _normalized_squared_error(prediction_batch: torch.Tensor, target_batch: torch.Tensor) -> torch.Tensor:
    """计算预测与目标之间的逐元素平方误差。

    ?_semantic_loss_matrix 不同，这里不做语义区分，
    统一用平方误差，用于诊断日志中的原始误差统计。

    注意：本函数不做任何归一化，函数名中\"normalized" 仅为历史命名。
    实际返回的是原始 (pred - target)^2。与 lstm/trainer.py 同名函数口径一致。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。

    返回值：
        形状 (B, 4) 的、元素平方误差张量。
    """
    return (prediction_batch - target_batch).pow(2) # 逐元 (pred - target)^2.


def _risk_calibration_bin_weights(target_risk: torch.Tensor, *, bin_count: int = _RISK_CALIBRATION_BIN_COUNT, scaling_log_eps: float = _SCALING_LOG_EPS) -> torch.Tensor:
    """risk calibration 损失计算等频分箱逆频率权重.

    batch 内目标风险按分位数分成若干等频桶, 每个样本的权重
    等于其所在桶的逆频率 (桶内样本越少权重越大), 最后归一化
    使均值约 1. 这样低风险区间 (样本多) 不会被过度主导,
    高风险区间 (样本少) 也能得到足够关注.

    参数:
        target_risk: 目标风险值一维张量, 形状 (N,).

    返回值:
        与 target_risk 同形状的权重张量, 均值约 1.
    """
    if target_risk.dim() != 1:  # 仅接受一维目标风险张量，LSTM 侧口径对齐。
        raise ValueError(f"target_risk must be 1-D, got {target_risk.dim()}D")
    if not torch.isfinite(target_risk).all():  # 拒绝 NaN/Inf，避免经 quantile/bucketize/bincount 静默穿。
        raise ValueError("target_risk contains non-finite values (NaN or Inf)")
    sample_count = int(target_risk.numel())  # 样本总数。
    if sample_count <= 1:  # 只有一个样本时无法分箱，直接等权。
        return torch.ones_like(target_risk)  # 返回全 1 权重。
    effective_bin_count = max(1, min(bin_count, sample_count))  # 实际分箱数，不超过样本数。
    quantiles = torch.linspace(0.0, 1.0, steps=effective_bin_count + 1, device=target_risk.device, dtype=target_risk.dtype)  # 等距分位点，按实际分箱数生成，与 LSTM 侧一致。
    edges = torch.quantile(target_risk.detach(), quantiles)  # 在目标风险上算出实际分位边界。
    interior_edges = edges[1:-1]  # 去掉首尾边界，只留内部切分点。
    if interior_edges.numel() == 0:  # 没有内部边界时无法分桶。
        return torch.ones_like(target_risk)  # 返回全 1 权重。
    bucket_index = torch.bucketize(target_risk.detach(), interior_edges, right=True)  # 每个样本落入哪个桶。
    bucket_count = torch.bincount(bucket_index, minlength=int(interior_edges.numel()) + 1).clamp_min(1)  # 每个桶的样本数，至少 1。
    raw_weights = torch.reciprocal(bucket_count[bucket_index].to(dtype=target_risk.dtype))  # 逆频率权重。
    normalized_weights = raw_weights / torch.clamp_min(raw_weights.mean(), scaling_log_eps)  # 归一化使均值约为 1。
    return normalized_weights  # 返回归一化后的分箱权重。


def _risk_calibration_trainable(model: Any) -> bool:
    """检查模型的 risk_calibration 模块是否有可训练参数。

    只有 risk_calibration 存在、有 parameters 方法、且至少一个
    参数 requires_grad 为 True 时，才认为它是可训练的。
    用于决定是否在辅助损失中启用 calibration 与 mono 项。

    参数：
        model: 液体模型实例，预期有 risk_calibration 属性。

    返回值：
        True 表示 risk_calibration 可训练，False 表示不可训练或不存在。
    """
    calibration = getattr(model, "risk_calibration", None)  # 安全地取 risk_calibration 属性。
    parameters = getattr(calibration, "parameters", None)  # 安全地取 parameters 方法。
    if not callable(parameters):  # 没有 parameters 方法说明不是 nn.Module。
        return False  # 不可训练。
    return any(parameter.requires_grad for parameter in parameters())  # 至少一个参数可训练就返回True。


def _compute_liquid_auxiliary_loss_terms(
    model: Any,
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    modalities: list[str],
    sample_weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """计算 Liquid 模型特有的辅助损失项。

    Liquid 模型在主监督损失之外还有四项辅助损失。
    - calibration: risk calibration 的加MSE，配合分箱、频率权重；
    - mono: calibration 斜率的单调、惩罚，防止 a_raw 塌缩为负。
    - gate_l1: 输出头中各种门控参数L1 惩罚，维持中性启动；
    - reg_l2: 全局 L2 正则，防止参数膨胢。

    参数：
        model: 液体模型实例。
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。
        modalities: 当前批次每个样本的模态名列表。
        sample_weights: 样本权重列表，None 时默认等权。

    返回值：
    字典，包含"calibration"?mono"?reg_l2"?gate_l1"?total"
        五个键，每个值都是标量张量。
    """
    zero = prediction_batch.new_zeros(())  # 零标量，用作各项默认值。
    # 从模型实例属性读取配置、，缺失时回逢、到模块级常量默认值。
    loss_weights = getattr(model, "_loss_weights", {})
    # D5/D7：与 lstm/trainer.py 同名函数口径对齐，统丢coerce_finite_scalar 中心入口。
    # float() 会把 NaN/Inf 与负值静默穿透到损失项，导致训练发散时无法定位字段名。
    scaling_log_eps = coerce_finite_scalar(  # 数稳定项必须有限且非负，LSTM 侧一致。
        getattr(model, "_scaling_log_eps", _SCALING_LOG_EPS),
        name="scaling_log_eps",
        min_value=0.0,
    )
    calibration_bin_count = int(  # 分箱数必须有限且 >=1，避免下界max(1, min(...)) 兜底掩盖配置错误。
        coerce_finite_scalar(
            getattr(model, "_calibration_bin_count", _RISK_CALIBRATION_BIN_COUNT),
            name="calibration_bin_count",
            min_value=1,
        )
    )
    w_cal = coerce_finite_scalar(  # 校准辅助权重，拒NaN/Inf/负。
        loss_weights.get("w_cal", _LIQUID_CALIBRATION_WEIGHT),
        name="loss_weights.w_cal",
        min_value=0.0,
    )
    lambda_mono = coerce_finite_scalar(  # 单调性权重，拒绝 NaN/Inf/负。
        loss_weights.get("lambda_mono", _LIQUID_MONO_WEIGHT),
        name="loss_weights.lambda_mono",
        min_value=0.0,
    )
    lambda_l1 = coerce_finite_scalar(  # 门控 L1 权重，拒NaN/Inf/负。
        loss_weights.get("lambda_l1", _LIQUID_GATE_L1_WEIGHT),
        name="loss_weights.lambda_l1",
        min_value=0.0,
    )
    lambda_l2 = coerce_finite_scalar(  # 全局 L2 权重，拒NaN/Inf/负。
        loss_weights.get("lambda_l2", _GLOBAL_L2_WEIGHT),
        name="loss_weights.lambda_l2",
        min_value=0.0,
    )
    # Read enable_smooth first so we can choose the correct default for lambda_smooth.
    phase_control_state = getattr(model, "_phase_control_state", {}) or {}
    enable_smooth = bool(
        phase_control_state.get("enable_smooth_regularization", False)
        or getattr(model, "_enable_smooth_regularization", False)
    )
    # 当启用 smooth 正则时，默认使用 _LIQUID_SMOOTH_WEIGHT（=1e-3）作为权重，
    # 否则权重为 0（保持向后兼容）。允许 loss_weights["lambda_smooth"] 显式覆盖。
    lambda_smooth = coerce_finite_scalar(  # L_smooth 输出平滑正则权重（默认 _LIQUID_SMOOTH_WEIGHT=1e-3 当启用时）。
        loss_weights.get("lambda_smooth", _LIQUID_SMOOTH_WEIGHT if enable_smooth else 0.0),
        name="loss_weights.lambda_smooth",
        min_value=0.0,
    )
    aux_terms = {  # v2：五项辅助损失（含 smooth 输出平滑正则）。
        "calibration": zero,  #  risk calibration 损失。
        "mono": zero,  #  calibration 单调性惩罚。
        "reg_l2": zero,  #  全局 L2 正则。
        "gate_l1": zero,  #  门控 L1 惩罚。
        "smooth": zero,  # L_smooth 输出平滑正则（v2 可选）。
    }
    aux_scale = coerce_finite_scalar(
        phase_control_state.get("aux_scale", 1.0),
        name="phase_control_state.aux_scale",
        min_value=0.0,
    )
    gate_scale = coerce_finite_scalar(
        phase_control_state.get("gate_scale", aux_scale),
        name="phase_control_state.gate_scale",
        min_value=0.0,
    )
    calibration_scale = coerce_finite_scalar(
        phase_control_state.get("calibration_scale", aux_scale),
        name="phase_control_state.calibration_scale",
        min_value=0.0,
    )
    regularization_scale = coerce_finite_scalar(
        phase_control_state.get("regularization_scale", 1.0),
        name="phase_control_state.regularization_scale",
        min_value=0.0,
    )
    if _risk_calibration_trainable(model) and calibration_scale > 0.0: # 只有 risk_calibration 可训练时才计 calibration/mono.
        risk_mask = _build_head_mask_tensor(modalities, reference_tensor=prediction_batch)[:, 1] # risk 列的掩码.
        sample_weight_tensor = _build_sample_weight_tensor(
            sample_weights,
            reference_tensor=prediction_batch,
        )
        effective_weight = risk_mask * sample_weight_tensor # risk 列有效权重= 头掩码× 样本权重。
        observed = effective_weight > 0.0  # 只有权重 > 0 的样本才参与 calibration 损失。
        if bool(observed.any().item()):  # 至少有一个有risk 样本时才计算。
            predicted_risk = prediction_batch[observed, 1]  # 取有效样本的预测 risk。
            target_risk = target_batch[observed, 1]  # 取有效样本的目标 risk。
            bin_weights = _risk_calibration_bin_weights(target_risk, bin_count=calibration_bin_count, scaling_log_eps=scaling_log_eps)  # 按目标风险分箱计算、频率权重。
            calibration_weight = effective_weight[observed] * bin_weights # 最终校准权= 有效权重 × 分箱权重。
            calibration_loss = ((predicted_risk - target_risk).pow(2) * calibration_weight).sum()  # 加权 MSE 分子。
            calibration_loss = calibration_loss / torch.clamp_min(calibration_weight.sum(), scaling_log_eps)  # 除以权重和得到均值。
            aux_terms["calibration"] = calibration_loss * w_cal * calibration_scale  # 乘以校准辅助权重。
        calibration = getattr(model, "risk_calibration", None)  # 安全地取 risk_calibration 模块。
        if hasattr(calibration, "a_raw"):  # 如果存在 a_raw 参数（校准斜率的原始值）。
            aux_terms["mono"] = F.softplus(-calibration.a_raw).mean() * lambda_mono * calibration_scale  # 惩罚 a_raw 为负，保证校准斜率非负。

    gate_penalty = zero  # 门控 L1 惩罚累加器。
    # v3.1 head-shared: 门控层在 output_backbone 上，不在 output_heads 中
    backbone = getattr(model, "output_backbone", None)
    if backbone is not None:
        for attribute_name in (  # 遍历有门控子模块名。
            "temporal_context_gate",  # 时间上下文门控。
            "observation_context_gate",  # 观测上下文门控。
            "filter_context_gate",  # 滤波上下文门控。
            "branch_mix_gate",  # 分支混合门控。
            "uwb_branch_mix_gate",  # UWB 分支混合门控。
            "vio_branch_mix_gate",  # VIO 分支混合门控。
        ):
            module = getattr(backbone, attribute_name, None)  # 安全地取门控子模块。
            parameters = getattr(module, "parameters", None)  # 安全地取 parameters 方法。
            if not callable(parameters):  # 不是 nn.Module 就跳过。
                continue  # 跳过不可调用的。
            for parameter in parameters():  # 遍历该门控的有参数。
                if parameter.requires_grad:  # 只对可训练参数做 L1。
                    gate_penalty = gate_penalty + parameter.abs().sum()  # 累加绝对值。
    if float(gate_penalty.detach().item()) > 0.0 and gate_scale > 0.0:  # 如果有非零门控惩罚。
        aux_terms["gate_l1"] = gate_penalty * lambda_l1 * gate_scale  # 乘以门控 L1 权重。
    l2_penalty = zero  # L2 正则累加器。
    for parameter in model.parameters():  # 遍历模型有参数。
        if parameter.requires_grad:  # 只对可训练参数做 L2。
            l2_penalty = l2_penalty + parameter.pow(2).sum()  # 累加参数平方和。
    if float(l2_penalty.detach().item()) > 0.0:  # 如果有非L2 惩罚。
        aux_terms["reg_l2"] = l2_penalty * lambda_l2 * regularization_scale  # 乘以阶段化正则接管权重。
    # v2: 输出平滑正则 L_smooth（可选）。
    # 惩罚相邻时间步测量的差异，促使模型输出更平滑。
    if enable_smooth and lambda_smooth > 0.0 and prediction_batch.shape[0] > 1:
        smooth_loss = (prediction_batch[1:] - prediction_batch[:-1]).pow(2).mean() * lambda_smooth * regularization_scale
        aux_terms["smooth"] = smooth_loss
    aux_terms["total"] = aux_terms["calibration"] + aux_terms["mono"] + aux_terms["reg_l2"] + aux_terms["gate_l1"] + aux_terms["smooth"]  # 四项辅助损失求和。
    return aux_terms  # 返回包含有辅助损失项的字典。


def _masked_component_losses(

    prediction_batch: torch.Tensor,

    target_batch: torch.Tensor,

    *,

    modalities: list[str],

    sample_weights: list[float] | None = None,

    bias_huber_delta: float = _BIAS_HUBER_DELTA,

    scaling_log_eps: float = _SCALING_LOG_EPS,

) -> tuple[dict[str, torch.Tensor], list[str]]:
    """逐输出头计算独立的加权平均损失。

    ?_masked_supervised_loss_stats 不同，本函数返回每个濢、活头
    独立做除法后的平均损失，用于选择分数计算和诊断日志。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。
        modalities: 当前批次每个样本的模态名列表。
        sample_weights: 样本权重列表，None 时默认等权。
        bias_huber_delta: bias ?Huber 损失 delta，默认_BIAS_HUBER_DELTA。
        scaling_log_eps: scaling 转对数前数、稳定项，默认_SCALING_LOG_EPS。
        两、必须转发模型配置、，_masked_supervised_loss 及评估管。
            同口径，避免选择分数漂移（D6 指标口径冻结、D7 公平性）。

    返回值：
    二元（component_losses, active_keys)。
        - component_losses: 每个濢、活头的独立平均损失字典；
        - active_keys: 本批次真正参与损失的输出头名列表。
    """

    component_numerators, component_denominators, active_keys, position_mse_scalar = _masked_component_loss_stats(  # 先拿各头统计量（含§24.1 位置误差执行点）。
        prediction_batch,  # 预测批次。
        target_batch,  # 目标批次。
        modalities=modalities,  # 当前批次的模态列表。
        sample_weights=sample_weights,  # 当前批次的样本权重。
        bias_huber_delta=bias_huber_delta,  # 转发模型配置bias Huber delta（D6/D7 口径一致）。
        scaling_log_eps=scaling_log_eps,  # 转发模型配置scaling 对数稳定项（D6/D7 口径一致）。
    )  # 统计结束。
    component_losses: dict[str, torch.Tensor] = {}  # 每个活头的独立损失。
    for key in active_keys:  # 只处理真正参与本批次损失的头。
        component_losses[key] = component_numerators[key] / component_denominators[key]  # 当前头的平均损失。
    return component_losses, active_keys  # 返回每个头的损失和激活头列表。




def _weighted_selection_score(component_losses: Mapping[str, torch.Tensor], active_keys: list[str], *, position_mse_scalar: torch.Tensor | None = None) -> torch.Tensor:
    """按预设权重计算多头的加权选择分数。§24.1 最终定序仅 raw 位置 RMSE；

    当前多头加权选择分数为辅助监控；主选模与全序结论必须以 raw 位置 RMSE（§14 主指标）为准。
    选择分数用于在验证阶段挑选最优检查点，各头的权重
    ?_SELECTION_WEIGHTS 中定义、只active_keys 中的。
    参与计算，权重按活头重新归一化。

    参数：
        component_losses: 每个活头的独立损失字典。
        active_keys: 本批次真正参与损失的输出头名列表。

    返回值：
        加权选择分数标量张量。

    失败条件。
    KeyError: ?active_keys 中存在未在_SELECTION_WEIGHTS 中定义的头时抛出。
        ValueError: 当权重、为 NaN/Inf/负数或所有激活头的权重之和为零时抛出。
    """
    missing_keys = [key for key in active_keys if key not in _SELECTION_WEIGHTS] # D9：校验active_keys ?_SELECTION_WEIGHTS（键名等价于 MODEL_INTERMEDIATE_KEYS）。
    if missing_keys:  # 存在未定义的输出头。
        raise KeyError(f"selection_weights missing keys: {missing_keys}")  # 报告缺失键名，便于定位。
    # §24.1 执行点：主损失必须包含位置误差 MSE（_position_error_mse 执行函数已定义并可用于扩展）。
    # §24.1 执行点：位置误差 MSE 必须参与选模（终序仅 raw 位置 RMSE，不可仅靠辅头）。
    _POSITION_MSE_SELECTION_WEIGHT = 0.50  # §24.1 位置误差主项选模权重（占 50%，高于各辅头）。
    validated_weights = {  # D5：经 coerce_finite_scalar 校验，拒NaN/Inf/bool/负数权重，与 lstm/trainer.py 同名函数口径致（D7）。
        key: coerce_finite_scalar(_SELECTION_WEIGHTS[key], name=f"selection_weights.{key}", min_value=0.0)
        for key in active_keys
    }
    total_weight = sum(validated_weights[key] for key in active_keys)  # 计算活头的权重。

    if total_weight <= 0.0:  # 总权重为零说明没有有效头。

        raise ValueError("selection score requires at least one active weighted head")  # 直接报错。

    # §24.1 执行点：位置误差 MSE 作为主项加入选模分数（强制参与最终定序）。
    base_score = sum(  # D7：与 lstm/trainer.py 同名函数对齐，使float32 起始张量保证结果类型一致。
        (
            (validated_weights[key] / total_weight) * component_losses[key]
            for key in active_keys
        ),
        start=torch.tensor(0.0, dtype=torch.float32),
    )  # 按归化权重加权求和。
    # §24.1 执行点：位置误差 MSE 列为选模主项（50% 权重），确保终序仅 raw 位置 RMSE 精神执行。
    if position_mse_scalar is not None:
        base_score = base_score + _POSITION_MSE_SELECTION_WEIGHT * position_mse_scalar  # 位置误差强制纳入选模。
    return base_score


def _compute_loss(model: Any, window_tensor: Mapping[str, Any], target_outputs: Mapping[str, float]) -> torch.Tensor:
    """计算单样本的监督损失（不含辅助项）。

    先做前向推理拿到预测输出，再按模态构造掩码，
    朢、后调用_masked_supervised_loss 计算加权损失。

    参数：
        model: 液体模型实例，必须有 predict_intermediate_tensors 方法。
        window_tensor: 结构化特征窗口映射。
        target_outputs: 目标输出字典。

    返回值：
        单样本的监督损失标量张量。
    """
    modality = _resolve_current_modality(window_tensor)  # 解析当前模。

    predicted_outputs = model.predict_intermediate_tensors(window_tensor)  # 前向推理拿到预测输出。

    prediction_vector = torch.stack(  # 按固定顺序拼成预测向量。

        [

            predicted_outputs[key].reshape(())  # 每个输出头取标量。

            for key in _OUTPUT_KEYS # ?_OUTPUT_KEYS?MODEL_INTERMEDIATE_KEYS）单源真相顺序，禁止字面量漂移。

        ]

    )

    target_vector = prediction_vector.new_tensor([coerce_finite_scalar(target_outputs[key], name=f"target_outputs.{key}") for key in _OUTPUT_KEYS])  # 目标也按相同顺序构、向量；lstm/trainer.py._compute_loss 口径对齐：拒NaN/Inf（D5/D7）。

    return _masked_supervised_loss(

        prediction_vector.unsqueeze(0),

        target_vector.unsqueeze(0),

        modalities=[modality],
        bias_huber_delta=coerce_finite_scalar(getattr(model, "_bias_huber_delta", _BIAS_HUBER_DELTA), name="model._bias_huber_delta", min_value=0.0),
        scaling_log_eps=coerce_finite_scalar(getattr(model, "_scaling_log_eps", _SCALING_LOG_EPS), name="model._scaling_log_eps", min_value=0.0),

    )





def _compute_batch_loss(

    model: Any,

    target_batch: torch.Tensor,

    metadata: list[dict[str, Any]],

    *,

    sample_weights: list[float] | None = None,

) -> torch.Tensor:
    """计算一个批次的监督损失加辅助损失。

    先预测当前批次，再计算四头监督损失和 Liquid 辅助损失。
    最后返回两者之和。

    参数：
        model: 液体模型实例。
        target_batch: 目标值批次，形状 (B, 4)。
        metadata: 当前批次每个样本的元数据列表。
        sample_weights: 样本权重列表，None 时默认等权。

    返回值：
        监督损失 + 辅助损失的标量张量。
    """

    prediction_batch = _predict_batch(model, metadata)  # 先预测当前批次。
    # D7/D3：与 lstm/trainer.py 同名函数口径对齐，先校验 target_batch 形状。
    # 避免错误形状静默穿、到 _masked_supervised_loss 后被广播掩盖。
    if target_batch.ndim != 2 or target_batch.shape[1] != len(_OUTPUT_KEYS):
        raise ValueError(
            f"target_batch must have shape (batch, {len(_OUTPUT_KEYS)}), got {tuple(target_batch.shape)}"
        )
    target_batch = target_batch.to(device=prediction_batch.device, dtype=prediction_batch.dtype)
    modalities = [_resolve_current_modality(sample_meta) for sample_meta in metadata]  # 逐个样本读模态。
    # D5/D7：与 lstm/trainer.py ?_compute_liquid_auxiliary_loss_terms 一致，
    # 统一coerce_finite_scalar 中心入口；float() 会把 NaN/Inf/负、静默穿透到损失项。
    supervised_loss = _masked_supervised_loss(  # 先算基础四头监督损失。
        prediction_batch,  # 预测批次。
        target_batch,  # 目标批次。
        modalities=modalities,  # 当前批次的模态列表。
        sample_weights=sample_weights,  # 当前批次的样本权重。
        bias_huber_delta=coerce_finite_scalar(
            getattr(model, "_bias_huber_delta", _BIAS_HUBER_DELTA),
            name="bias_huber_delta",
            min_value=0.0,
        ),
        scaling_log_eps=coerce_finite_scalar(
            getattr(model, "_scaling_log_eps", _SCALING_LOG_EPS),
            name="scaling_log_eps",
            min_value=0.0,
        ),
    )
    auxiliary_terms = _compute_liquid_auxiliary_loss_terms(
        model,
        prediction_batch,
        target_batch,
        modalities=modalities,
        sample_weights=sample_weights,
    )
    return supervised_loss + auxiliary_terms["total"]  # 把默认启用的 Liquid 辅助项一起并入损失。




def _predict_batch(model: Any, metadata: list[dict[str, Any]]) -> torch.Tensor:
    """对一个批次的样本做前向预测，返回标准化后的预测批次张量。

    批量提取共享特征（cell 递推做真正的批量矩阵运算），
    批量通过输出头（混合模、过模、指示器处理），
    再做 risk 校准、标准化和模态合约。

    参数：
        model: 液体模型实例。
        metadata: 当前批次每个样本的元数据列表。

    返回值：
        形状 (B, 4) 的预测批次张量，列顺序为 _OUTPUT_KEYS。
    """

    # 批量提取共享特征：cell 递推做真正的批量矩阵运算。
    shared_features_list = model.network.extract_shared_features_batch(
        metadata,
        minimal_output_head_features=True,
    )

    if len(shared_features_list) == 1:  # 单样本走逐样本路径。
        sf = shared_features_list[0]
        # v3.1 head-shared: 通过 _LiquidModel.run_head_forward 共享 backbone 再走 4 个 final-projection head
        if hasattr(model, "run_head_forward"):
            raw_outputs = model.run_head_forward(sf)
        else:
            raw_outputs = {key: head(sf) for key, head in model.output_heads.items()}
        risk_already_normalized = False
        calibration = getattr(model, "risk_calibration", None)
        if callable(calibration):
            # D5/D7：与 lstm/trainer.py _project_train_outputs（L222）口径一致。
            # 校准前先校验原始 risk 的有限、sigmoid(Inf)=1.0 / sigmoid(-Inf)=0.0 。
            # ?Inf 静默映射到有限、，掩盖模型前向的数值不稳定（如梯度爆炸），
            # 下游 _normalize_liquid_train_outputs 的有限、检查此时无法再捕获原始 Inf。
            _raw_risk_pre_calib = torch.as_tensor(raw_outputs["risk"], dtype=torch.float32).reshape(())
            if not torch.isfinite(_raw_risk_pre_calib):
                raise ValueError(f"raw risk output must be finite before calibration, got {_raw_risk_pre_calib.item()}")
            raw_outputs = dict(raw_outputs)
            raw_outputs["risk"] = calibration(raw_outputs["risk"])
            risk_already_normalized = True
        predicted_outputs = _normalize_liquid_train_outputs(
            raw_outputs, risk_already_normalized=risk_already_normalized,
        )
        predicted_outputs = apply_liquid_modality_output_contract(
            predicted_outputs, modality=_resolve_current_modality(sf),
        )
        return torch.stack([predicted_outputs[key].reshape(()) for key in _OUTPUT_KEYS]).unsqueeze(0)

    # 批量通过输出头：混合模、过模、指示器处理。
    raw_batch_outputs: dict[str, torch.Tensor] = {}  # 每个头的 (B,) 输出。
    if hasattr(model, "run_head_forward_batch"):
        # v3.1 head-shared: 共享 backbone + 4 个 final-projection head
        raw_batch_outputs = model.run_head_forward_batch(shared_features_list)
    else:
        for key, head in model.output_heads.items():
            raw_batch_outputs[key] = head.forward_batch(shared_features_list)  # (B,)

    # 批量 risk 标准分箱结果。
    risk_already_normalized = False
    calibration = getattr(model, "risk_calibration", None)
    if callable(calibration):
        # D5/D7：与标量版及 lstm/trainer.py _project_train_outputs（L222）口径一致。
        # 校准前先校验原始 risk 批次张量的有限、，避免 sigmoid ?Inf 静默映射。
        # 有限值（sigmoid(Inf)=1.0 / sigmoid(-Inf)=0.0），掩盖模型前向数、不稳定。
        # 下游 _normalize_liquid_train_outputs_batch 的有限、检查此时无法再捕获原始 Inf。
        _raw_risk_batch_pre_calib = torch.as_tensor(raw_batch_outputs["risk"], dtype=torch.float32)
        if not torch.isfinite(_raw_risk_batch_pre_calib).all():
            raise ValueError(f"raw risk batch output must be finite before calibration, got {_raw_risk_batch_pre_calib}")
        raw_batch_outputs = dict(raw_batch_outputs)
        raw_batch_outputs["risk"] = calibration(raw_batch_outputs["risk"])  # (B,) →(B,)
        risk_already_normalized = True

    # 批量标准化训练、输出。
    normalized_batch = _normalize_liquid_train_outputs_batch(
        raw_batch_outputs, risk_already_normalized=risk_already_normalized,
    )  # dict[str, (B,)]

    # 逐样本应用模态输出合约。
    modalities = [_resolve_current_modality(sf) for sf in shared_features_list]
    prediction_rows: list[torch.Tensor] = []
    for i in range(len(shared_features_list)):
        per_sample = {key: normalized_batch[key][i] for key in _OUTPUT_KEYS}
        per_sample = apply_liquid_modality_output_contract(per_sample, modality=modalities[i])
        prediction_rows.append(torch.stack([per_sample[key].reshape(()) for key in _OUTPUT_KEYS]))

    return torch.stack(prediction_rows, dim=0)  # (B, 4)




def _loss_for_length_group(model: Any, samples: list[tuple[dict[str, Any], torch.Tensor, int, float]]) -> torch.Tensor:
    """计算同一序列长度组内所有样本的批次损失。

    同组样本序列长度相同，可以直接堆叠成批次做前向和损失计算。

    参数：
        model: 液体模型实例。
        samples: 同组材料化样本列表，每个元素是四元组。

    返回值：
        该组的批次损失标量张量。
    """

    if not samples:  # D7：与 lstm/trainer.py 同名函数口径对齐，空组直接报错、非torch.stack 抛出误导性异常。
        raise ValueError("_loss_for_length_group requires at least one sample, got empty list")
    sample_weights = [sample[3] for sample in samples]  # 提前抽取权重，便NaN/Inf 预检与下游复用。
    if not all(math.isfinite(float(weight)) for weight in sample_weights):  # D5/D7：NaN/Inf 权重在进入批次前就拦截，lstm 侧口径一致。
        raise ValueError(f"non-finite selection_weight detected in length group: {sample_weights}")
    target_batch = torch.stack([sample[1] for sample in samples], dim=0)  # 把同组目标堆batch。
    return _compute_batch_loss(  # 直接复用批次损失计算。
        model,  # 当前模型。
        target_batch,  # 该组目标。
        [sample[0] for sample in samples],  # 该组元数据。
        sample_weights=sample_weights,  # 该组样本权重。
    )  # 长度组损失结束。




