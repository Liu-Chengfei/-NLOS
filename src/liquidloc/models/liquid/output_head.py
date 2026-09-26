"""Liquid 模型输出头、共享输出骨干和风险校准实现。

这些模块属于模型层；工厂只负责组装模型并为旧导入路径重导出公开符号。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from liquidloc.common.constants import (
    CONTEXT_DIM,
    CONTEXT_FEATURE_KEYS,
    LIQUID_READOUT_CONTEXT_KEYS,
    MODALITY_UWB,
    MODALITY_VIO,
    RISK_PRIOR_LOGIT,
)
from liquidloc.common.validation import coerce_finite_scalar, is_string_like
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS

# ===========================================================================
# 输出头与风险校准模块（原 liquidloc.models.liquid.output_head 实现，
# 为对齐 docs/liquid_architecture_current.md §6 已合入本模块）。
# 以下常量、函数与类保持原有行为不变，仅做物理位置迁移。
# ===========================================================================

# ---------------------------------------------------------------------------
# 模块级常量：Liquid 上下文特征键名，用于构建上下文张量。
# 每个特征在上下文向量中占两个位置：数值 + 观测标志位。
# ---------------------------------------------------------------------------
# 铁律 9 / §11.3：主网 feature_order 只吃 raw；上下文仅允许共享无效硬标志与几何/时间原始量。
# 禁止 quality / uwb_quality_min / uwb_invalid_rate 等 sim 派生质量标签进 NN 上下文
# （否则与「无 NLOS 标签 / 无质量教师」主路径纪律冲突）。
# §12.3-C2b 第十三轮消重：CONTEXT_FEATURE_KEYS 单源常量在 common.constants 中，
# 此处保留 LIQUID_CONTEXT_FEATURE_KEYS / LIQUID_CONTEXT_DIM 别名便于下游引用稳定。
LIQUID_CONTEXT_FEATURE_KEYS = CONTEXT_FEATURE_KEYS
LIQUID_CONTEXT_DIM = CONTEXT_DIM

# ---------------------------------------------------------------------------
# 模块级常量：模态上下文在上下文向量中的索引位置（前两个元素）。
# ---------------------------------------------------------------------------
LIQUID_MODALITY_CONTEXT_INDICES = (0, 1)

# ---------------------------------------------------------------------------
# 模块级常量：读出上下文键名（来自 liquidloc.common.constants 单源真相）。
# 公开名 LIQUID_READOUT_CONTEXT_KEYS 重新导出，供 model_factory 等下游引用。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 模块级常量：滤波上下文向量维度 = 2 × 读出键数（每个键占数值+标志位）。
# ---------------------------------------------------------------------------
LIQUID_FILTER_CONTEXT_DIM = 2 * len(LIQUID_READOUT_CONTEXT_KEYS)

# ---------------------------------------------------------------------------
# 模块级常量：时间类上下文特征键名，与观测类特征分离。
# ---------------------------------------------------------------------------
LIQUID_TEMPORAL_CONTEXT_FEATURE_KEYS = ("modality_gap_dt",)

# ---------------------------------------------------------------------------
# 模块级常量：观测类上下文特征键名 = 全部上下文特征 - 时间类特征。
# ---------------------------------------------------------------------------
LIQUID_OBSERVATION_CONTEXT_FEATURE_KEYS = tuple(
    key for key in LIQUID_CONTEXT_FEATURE_KEYS if key not in LIQUID_TEMPORAL_CONTEXT_FEATURE_KEYS
)

# ---------------------------------------------------------------------------
# 模块级常量：UWB 快速上下文特征键名，只包含与 UWB 相关的观测特征。
# ---------------------------------------------------------------------------
LIQUID_UWB_FAST_CONTEXT_FEATURE_KEYS = (
    "valid",                # 共享无效硬标志
    "uwb_range_residual",   # UWB 距离残差
    "anchor_dx",            # 锚点 x 差值
    "anchor_dy",            # 锚点 y 差值
    "geom_score",           # 几何一致性评分
)

# ---------------------------------------------------------------------------
# 模块级常量：VIO 快速上下文特征键名，只包含与 VIO 相关的观测特征。
# ---------------------------------------------------------------------------
LIQUID_VIO_FAST_CONTEXT_FEATURE_KEYS = (
    "valid",                # 共享无效硬标志（§11.3）
    # 铁律 3/9: 不进 quality / tracked_features / reproj_err 派生量。
)

# ---------------------------------------------------------------------------
# 模块级常量：UWB 分支上下文特征 = 时间特征 + UWB 快速特征。
# ---------------------------------------------------------------------------
LIQUID_UWB_BRANCH_CONTEXT_FEATURE_KEYS = LIQUID_TEMPORAL_CONTEXT_FEATURE_KEYS + LIQUID_UWB_FAST_CONTEXT_FEATURE_KEYS

# ---------------------------------------------------------------------------
# 模块级常量：VIO 分支上下文特征 = 时间特征 + VIO 快速特征。
# ---------------------------------------------------------------------------
LIQUID_VIO_BRANCH_CONTEXT_FEATURE_KEYS = LIQUID_TEMPORAL_CONTEXT_FEATURE_KEYS + LIQUID_VIO_FAST_CONTEXT_FEATURE_KEYS

# ---------------------------------------------------------------------------
# 模块级常量：UWB 滤波上下文键名，来自滤波器状态反馈。
# ---------------------------------------------------------------------------
LIQUID_UWB_FILTER_CONTEXT_KEYS = (
    "state_cov_trace", "pos_cov",
    "last_gate_skip_flag", "consecutive_skip_count", "time_since_last_update",
    # 扩展键：UWB 专属退化/几何上下文（v2 redesign D2-3）
    "uwb_residual_norm", "geometry_dop",
)

# ---------------------------------------------------------------------------
# 模块级常量：VIO 滤波上下文键名，来自滤波器状态反馈。
# ---------------------------------------------------------------------------
LIQUID_VIO_FILTER_CONTEXT_KEYS = (
    "state_cov_trace", "pos_cov",
    "last_innovation_norm", "consecutive_skip_count", "time_since_last_update",
    # 扩展键：VIO 专属退化/几何上下文（v2 redesign D2-3）
    "vel_cov_trace", "vio_cov_summary", "cross_modal_consistency",
)

# ---------------------------------------------------------------------------
# 模块级常量：softplus⁻¹(1) = ln(e¹ - 1)，用于 RiskCalibration 参数初始化。
# ---------------------------------------------------------------------------
SOFTPLUS_ONE_INVERSE = math.log(math.expm1(1.0))

_VALID_MODALITIES = frozenset({MODALITY_UWB, MODALITY_VIO})  # Liquid 合同只接受这两种有效模态，引用单源真相避免漂移。


def coerce_supported_modality(modality: Any, *, name: str) -> str:
    """把模态字段规范化为受支持的合同值。"""
    if not is_string_like(modality):
        raise ValueError(f"{name} must be one of {sorted(_VALID_MODALITIES)}")
    modality_name = str(modality).strip().lower()
    if modality_name not in _VALID_MODALITIES:
        raise ValueError(f"{name} must be one of {sorted(_VALID_MODALITIES)}")
    return modality_name


# ---------------------------------------------------------------------------
# 模块级常量：缩放输出的中性底值，引用协议层单源真相 BRIDGE_THRESHOLDS["scaling_min"]。
# D9：禁止本地硬编码 1.0，避免与 lstm/trainer.py、liquid/trainer.py 跨文件漂移。
# ---------------------------------------------------------------------------
_SCALING_NEUTRAL_FLOOR = float(BRIDGE_THRESHOLDS["scaling_min"])

# 模块级常量：非当前模态 scaling 上界，引用协议层单源真相 BRIDGE_THRESHOLDS["non_current_scaling_ceiling"]。
# 第十八轮穷举自审修复：model_factory.py 此前硬编码非当前模态 scaling 上界（违反 §12.3-C2a 三网同一写入口精神）。
# 现改为单源常量，禁止未来任何文件各自硬编码 scaling 上界。
_SCALING_CEILING = float(BRIDGE_THRESHOLDS["non_current_scaling_ceiling"])


def apply_liquid_modality_output_contract(
    normalized_outputs: Mapping[str, torch.Tensor],
    *,
    modality: str,
) -> dict[str, torch.Tensor]:
    """按模态合同约束输出：非当前模态的缩放因子被 clamp 到 [scaling_min, 1.0]。

    当 ``modality == "uwb"`` 时，``vio_scaling`` clamp 到 [scaling_min, 1.0]；
    当 ``modality == "vio"`` 时，``uwb_scaling`` clamp 到 [scaling_min, 1.0]。
    这保证了同一时刻只有一个模态的缩放因子作用在主值轴上，而非活跃头被允许
    在 [scaling_min, 1.0] 区间内取 LLM-512 之前 LearntCell 学习到的小幅残余信号，
    避免硬 mask 把 head 的输入当作常数后训练梯度对 head 完全失去学习信号。

    参数
    ----------
    normalized_outputs : Mapping[str, torch.Tensor]
        归一化后的输出字典。
    modality : str
        当前模态名称（"uwb" 或 "vio"）。

    返回
    -------
    dict[str, torch.Tensor]
        约束后的输出字典。
    """
    modality_name = coerce_supported_modality(modality, name="modality")  # 统一模态合同，禁止未知模态静默穿透。
    constrained = dict(normalized_outputs)  # 浅拷贝输出字典。
    scaling_floor = _SCALING_NEUTRAL_FLOOR  # 单源下界 BRIDGE_THRESHOLDS["scaling_min"]（v3：回退到 1.0；v2 曾放宽到 0.5 因 e9 场景不当降权被废弃，详见 bridge_thresholds.py:85）
    # v3 改造：放宽非当前模态 scaling 上界从 1.0 → 2.5，让 4 头在 NLOS 场景下可以
    # 把非当前模态的 R 矩阵放大（noise_multiplier = scaling^2*(1+risk)），允许 EKF
    # 对非当前模态观测做更强降权。仍受 bridge_thresholds.scaling_max=50 顶层封顶。
    # 第十八轮穷举自审修复：scaling_ceiling 已改为模块级常量 _SCALING_CEILING，
    # 单源真相在 BRIDGE_THRESHOLDS["non_current_scaling_ceiling"]（见 constants.py + bridge_thresholds.py）。
    # 禁止未来任何文件各自硬编码 scaling 上界（违反 §12.3-C2a 三网同一写入口精神）。
    scaling_ceiling = _SCALING_CEILING
    if modality_name == MODALITY_UWB:  # 当前是 UWB 模态。
        # D5+D10：clamp 保留到 vio_scaling_head 的梯度连接（grad not None），
        # 且在 [scaling_min, 1.0] 区间内梯度直通（torch.clamp 局部不被截断），
        # 在该区间之外梯度为 0（与其他文件中的 straight-through 估计合同一致）。
        # nan_to_num 先把 NaN/Inf 净化为 0，防止 clamp 在 NaN 上行为未定义。
        _safe_vio = torch.nan_to_num(constrained["vio_scaling"], nan=0.0, posinf=0.0, neginf=0.0)
        constrained["vio_scaling"] = torch.clamp(_safe_vio, min=scaling_floor, max=scaling_ceiling)
    if modality_name == MODALITY_VIO:  # 当前是 VIO 模态。
        # D5+D10：同上，保留到 uwb_scaling_head 的梯度连接，区间内梯度直通。
        _safe_uwb = torch.nan_to_num(constrained["uwb_scaling"], nan=0.0, posinf=0.0, neginf=0.0)
        constrained["uwb_scaling"] = torch.clamp(_safe_uwb, min=scaling_floor, max=scaling_ceiling)
    return constrained  # 返回合同约束后的输出。



def _liquid_context_indices_for_features(feature_keys: tuple[str, ...]) -> tuple[int, ...]:
    """计算指定特征键在上下文张量中的索引位置。

    每个特征在上下文向量中占两个位置（数值 + 观测标志），
    偏移量为 2（模态 one-hot 占前两位）。

    参数
    ----------
    feature_keys : tuple[str, ...]
        特征键名元组。

    返回
    -------
    tuple[int, ...]
        对应的索引位置元组（每个特征返回两个索引）。
    """
    indices: list[int] = []
    for feature_key in feature_keys:
        feature_index = LIQUID_CONTEXT_FEATURE_KEYS.index(feature_key)
        value_index = 2 + (2 * feature_index)
        indices.extend((value_index, value_index + 1))
    return tuple(indices)


def _liquid_filter_context_indices_for_keys(feature_keys: tuple[str, ...]) -> tuple[int, ...]:
    """计算指定滤波上下文键在滤波上下文张量中的索引位置。

    每个键在滤波上下文向量中占两个位置（数值 + 观测标志），
    无模态偏移。

    参数
    ----------
    feature_keys : tuple[str, ...]
        滤波上下文键名元组。

    返回
    -------
    tuple[int, ...]
        对应的索引位置元组。
    """
    indices: list[int] = []
    for feature_key in feature_keys:
        feature_index = LIQUID_READOUT_CONTEXT_KEYS.index(feature_key)
        value_index = 2 * feature_index
        indices.extend((value_index, value_index + 1))
    return tuple(indices)


def _build_index_mask(length: int, active_indices: tuple[int, ...]) -> torch.Tensor:
    """构建索引掩码张量：活跃位置为 1.0，其余为 0.0。

    用于从完整上下文向量中屏蔽不需要的特征维度。

    参数
    ----------
    length : int
        掩码总长度。
    active_indices : tuple[int, ...]
        活跃位置索引。

    返回
    -------
    torch.Tensor
        float32 掩码张量。
    """
    mask = torch.zeros((length,), dtype=torch.float32)
    for index in active_indices:
        mask[index] = 1.0
    return mask


def _slice_context_vector(context_vector: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    """从上下文向量中按索引切片出子向量。

    参数
    ----------
    context_vector : torch.Tensor
        2D 上下文张量 (batch, dim)。
    indices : tuple[int, ...]
        要提取的索引位置。

    返回
    -------
    torch.Tensor
        切片后的子向量 (batch, len(indices))。空索引时返回 (batch, 0)。
    """
    if not indices:
        return context_vector.new_zeros((int(context_vector.shape[0]), 0))
    index_tensor = torch.tensor(indices, dtype=torch.long, device=context_vector.device)
    return context_vector.index_select(dim=1, index=index_tensor)


def _apply_context_mask(context_vector: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对上下文向量施加掩码：逐元素乘以掩码。

    参数
    ----------
    context_vector : torch.Tensor
        2D 上下文张量 (batch, dim)。
    mask : torch.Tensor
        1D 掩码张量 (dim,)。

    返回
    -------
    torch.Tensor
        掩码后的上下文向量。

    异常
    ------
    ValueError
        当 context_vector 不是 2D 时抛出。
    """
    if context_vector.ndim != 2:
        raise ValueError("context_vector for masking must be batch-shaped")
    return context_vector * mask.to(device=context_vector.device, dtype=context_vector.dtype).unsqueeze(0)


def _build_liquid_context_tensor(
    shared_features: Mapping[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """从共享特征构建 Liquid 模型的上下文张量。

    上下文张量结构：
    [modality_onehot(2), feature1_value, feature1_observed, feature2_value, feature2_observed, ...]

    参数
    ----------
    shared_features : Mapping[str, Any]
        网络提取的共享特征字典。
    dtype : torch.dtype
        输出张量的数据类型。
    device : torch.device
        输出张量的设备。

    返回
    -------
    torch.Tensor
        1D 上下文张量，长度为 LIQUID_CONTEXT_DIM。
    """
    current_modality = coerce_supported_modality(
        shared_features["current_modality"],
        name="shared_features.current_modality",
    )
    modality_vector = {
        "uwb": [1.0, 0.0],
        "vio": [0.0, 1.0],
    }[current_modality]

    current_feature_by_name = dict(shared_features["current_feature_by_name"])
    current_observed_by_name = dict(shared_features["current_observed_by_name"])
    context_values: list[float] = list(modality_vector)
    for feature_name in LIQUID_CONTEXT_FEATURE_KEYS:
        observed = bool(current_observed_by_name.get(feature_name, False))
        raw_value = current_feature_by_name.get(feature_name, 0.0)
        scalar_value = 0.0
        if observed:
            # coerce_finite_scalar 契约：返回有限 float，或抛出 TypeError/ValueError；
            # 绝不返回 NaN/Inf，故无需再叠加 math.isfinite 守卫（其 else 分支不可达）。
            scalar_value = coerce_finite_scalar(raw_value, name=f"current_feature_by_name.{feature_name}")
        context_values.append(scalar_value)
        context_values.append(1.0 if observed else 0.0)

    return torch.tensor(context_values, dtype=dtype, device=device)


def _build_liquid_filter_context_tensor(
    shared_features: Mapping[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """从共享特征构建 Liquid 模型的滤波上下文张量。

    滤波上下文张量结构：
    [filter_key1_value, filter_key1_observed, filter_key2_value, filter_key2_observed, ...]

    参数
    ----------
    shared_features : Mapping[str, Any]
        网络提取的共享特征字典。
    dtype : torch.dtype
        输出张量的数据类型。
    device : torch.device
        输出张量的设备。

    返回
    -------
    torch.Tensor
        1D 滤波上下文张量，长度为 LIQUID_FILTER_CONTEXT_DIM。
    """
    readout_context_by_name = dict(shared_features["readout_context_by_name"])
    readout_context_observed_by_name = dict(shared_features["readout_context_observed_by_name"])
    context_values: list[float] = []
    for feature_name in LIQUID_READOUT_CONTEXT_KEYS:
        observed = bool(readout_context_observed_by_name.get(feature_name, False))
        raw_value = readout_context_by_name.get(feature_name, 0.0)
        scalar_value = 0.0
        if observed:
            # coerce_finite_scalar 契约：返回有限 float，或抛出 TypeError/ValueError；
            # 绝不返回 NaN/Inf，故无需再叠加 math.isfinite 守卫（其 else 分支不可达）。
            scalar_value = coerce_finite_scalar(raw_value, name=f"readout_context_by_name.{feature_name}")
        context_values.append(scalar_value)
        context_values.append(1.0 if observed else 0.0)
    return torch.tensor(context_values, dtype=dtype, device=device)


class LiquidOutputHead(nn.Module):
    """Liquid 模型的单输出头模块。

    每个输出头负责从共享隐层特征和上下文信息中计算一个标量输出。
    核心机制：
    1. 根据当前模态屏蔽不相关的上下文维度
    2. 分别计算时间上下文和观测上下文对隐层的调制量
    3. 通过分支混合门控融合快/慢路径特征
    4. 通过滤波上下文进一步调制融合结果
    5. 投影到标量输出，加上残差连接

    参数
    ----------
    key : str
        输出头键名（"bias"、"risk"、"uwb_scaling"、"vio_scaling"）。
    hidden_dim : int
        共享隐层维度。

    说明
    ----
    ``_CONTEXT_MODULATION_SCALE`` 与 ``_FILTER_CONTEXT_MODULATION_SCALE`` 提供
    向后兼容的默认值；实际实例可从 network 配置中读取同名可配置项。
    """

    # 向后兼容的默认值；实例可通过 network 配置覆盖。
    _CONTEXT_MODULATION_SCALE = 0.25
    _FILTER_CONTEXT_MODULATION_SCALE = 0.25

    def __init__(self, key: str, hidden_dim: int, network_cfg: Mapping[str, Any] | None = None):
        super().__init__()
        self.key = key
        self.hidden_dim = hidden_dim
        cfg = dict(network_cfg or {})
        self.context_dim = LIQUID_CONTEXT_DIM
        self.filter_context_dim = LIQUID_FILTER_CONTEXT_DIM
        self.context_modulation_scale = coerce_finite_scalar(
            cfg.get("context_modulation_scale", self._CONTEXT_MODULATION_SCALE),
            name="context_modulation_scale",
            min_value=0.0,
            max_value=1.0,
        )
        self.filter_context_modulation_scale = coerce_finite_scalar(
            cfg.get("filter_context_modulation_scale", self._FILTER_CONTEXT_MODULATION_SCALE),
            name="filter_context_modulation_scale",
            min_value=0.0,
            max_value=1.0,
        )
        # ---- 上下文索引 ----
        self.temporal_context_indices = _liquid_context_indices_for_features(LIQUID_TEMPORAL_CONTEXT_FEATURE_KEYS)
        self.observation_context_indices = _liquid_context_indices_for_features(LIQUID_OBSERVATION_CONTEXT_FEATURE_KEYS)
        self.temporal_context_dim = len(self.temporal_context_indices)
        self.observation_context_dim = len(self.observation_context_indices)
        self.uwb_fast_context_indices = _liquid_context_indices_for_features(LIQUID_UWB_FAST_CONTEXT_FEATURE_KEYS)
        self.vio_fast_context_indices = _liquid_context_indices_for_features(LIQUID_VIO_FAST_CONTEXT_FEATURE_KEYS)
        self.uwb_branch_context_indices = _liquid_context_indices_for_features(LIQUID_UWB_BRANCH_CONTEXT_FEATURE_KEYS)
        self.vio_branch_context_indices = _liquid_context_indices_for_features(LIQUID_VIO_BRANCH_CONTEXT_FEATURE_KEYS)
        self.uwb_filter_context_indices = _liquid_filter_context_indices_for_keys(LIQUID_UWB_FILTER_CONTEXT_KEYS)
        self.vio_filter_context_indices = _liquid_filter_context_indices_for_keys(LIQUID_VIO_FILTER_CONTEXT_KEYS)
        # ---- 门控线性层 ----
        self.temporal_context_gate = nn.Linear(self.temporal_context_dim, hidden_dim)
        self.observation_context_gate = nn.Linear(self.observation_context_dim, hidden_dim)
        self.filter_context_gate = nn.Linear(self.filter_context_dim, hidden_dim)
        self.branch_mix_gate = nn.Linear(self.context_dim, 1)
        self.uwb_branch_mix_gate = nn.Linear(self.context_dim + self.filter_context_dim, 1)
        self.vio_branch_mix_gate = nn.Linear(self.context_dim + self.filter_context_dim, 1)
        # ---- 投影层 ----
        self.projection = nn.Linear(hidden_dim + self.context_dim + self.filter_context_dim, 1)
        # D4 v2 redesign: residual_projection 接收 final_hidden + pooled_hidden 拼接 (2*hidden_dim)
        self.residual_projection = nn.Linear(2 * hidden_dim, 1)
        # ---- 掩码缓冲区（不参与梯度，不持久化到 state_dict）----
        self.register_buffer(
            "uwb_fast_context_mask",
            _build_index_mask(self.context_dim, self.uwb_fast_context_indices),
            persistent=False,
        )
        self.register_buffer(
            "vio_fast_context_mask",
            _build_index_mask(self.context_dim, self.vio_fast_context_indices),
            persistent=False,
        )
        self.register_buffer(
            "uwb_branch_context_mask",
            _build_index_mask(self.context_dim, self.uwb_branch_context_indices),
            persistent=False,
        )
        self.register_buffer(
            "vio_branch_context_mask",
            _build_index_mask(self.context_dim, self.vio_branch_context_indices),
            persistent=False,
        )
        self.register_buffer(
            "uwb_filter_context_mask",
            _build_index_mask(self.filter_context_dim, self.uwb_filter_context_indices),
            persistent=False,
        )
        self.register_buffer(
            "vio_filter_context_mask",
            _build_index_mask(self.filter_context_dim, self.vio_filter_context_indices),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """重置所有可学习参数。

        - risk 头的投影偏置初始化为 log(0.1/0.9)，使初始风险偏向低值。
        - 其余权重和偏置初始化为零，确保初始输出仅由残差路径决定。
        """
        if self.key == "risk":
            bias_value = RISK_PRIOR_LOGIT
        else:
            bias_value = 0.0

        with torch.no_grad():
            self.temporal_context_gate.weight.zero_()
            self.temporal_context_gate.bias.zero_()
            self.observation_context_gate.weight.zero_()
            self.observation_context_gate.bias.zero_()
            self.filter_context_gate.weight.zero_()
            self.filter_context_gate.bias.zero_()
            self.branch_mix_gate.weight.zero_()
            self.branch_mix_gate.bias.zero_()
            self.uwb_branch_mix_gate.weight.zero_()
            self.uwb_branch_mix_gate.bias.zero_()
            self.vio_branch_mix_gate.weight.zero_()
            self.vio_branch_mix_gate.bias.zero_()
            self.projection.weight.zero_()
            if self.projection.bias is not None:
                self.projection.bias.fill_(float(bias_value))
            self.residual_projection.weight.zero_()
            if self.residual_projection.bias is not None:
                self.residual_projection.bias.zero_()

    def forward(self, shared_features: Any) -> torch.Tensor:
        """前向传播：从共享特征计算标量输出。

        处理流程：
        1. 提取并校验 shared_vector、context_vector、filter_context_vector
        2. 提取快/慢路径隐层特征
        3. 按模态屏蔽上下文
        4. 计算时间/观测/滤波上下文调制量
        5. 分支混合门控融合快/慢路径
        6. 投影到标量输出 + 残差连接

        参数
        ----------
        shared_features : Any
            网络提取的共享特征字典，必须包含 shared_vector。

        返回
        -------
        torch.Tensor
            标量输出张量。

        异常
        ------
        TypeError
            当 shared_features 不是映射时抛出。
        ValueError
            当张量形状不匹配时抛出。
        """
        if not isinstance(shared_features, Mapping):
            raise TypeError("shared_features must be a mapping produced by LiquidNetwork")

        # ---- 1. 提取并校验 shared_vector ----
        shared_vector = shared_features.get("shared_vector")
        if shared_vector is None:
            raise ValueError("shared_features must expose shared_vector for residual output heads")
        shared_vector = torch.as_tensor(shared_vector, dtype=torch.float32)
        if shared_vector.ndim == 1:
            shared_vector = shared_vector.unsqueeze(0)
        if shared_vector.ndim != 2 or int(shared_vector.shape[1]) != self.hidden_dim:
            raise ValueError(
                f"shared_vector must have shape ({self.hidden_dim},) or (batch, {self.hidden_dim}), "
                f"got {tuple(shared_vector.shape)}"
            )

        # ---- 2. 提取并校验 context_vector ----
        context_vector = shared_features.get("context_vector")
        if context_vector is None:
            context_vector = _build_liquid_context_tensor(
                shared_features,
                dtype=shared_vector.dtype,
                device=shared_vector.device,
            ).unsqueeze(0)
        else:
            context_vector = torch.as_tensor(
                context_vector,
                dtype=shared_vector.dtype,
                device=shared_vector.device,
            )
            if context_vector.ndim == 1:
                context_vector = context_vector.unsqueeze(0)
        if context_vector.ndim != 2 or int(context_vector.shape[1]) != self.context_dim:
            raise ValueError(
                f"context_vector must have shape ({self.context_dim},) or (batch, {self.context_dim}), "
                f"got {tuple(context_vector.shape)}"
            )
        if int(context_vector.shape[0]) != int(shared_vector.shape[0]):
            raise ValueError("shared_vector and context_vector batch dimensions must match")

        # ---- 3. 提取并校验 filter_context_vector ----
        filter_context_vector = shared_features.get("filter_context_vector")
        if filter_context_vector is None:
            filter_context_vector = _build_liquid_filter_context_tensor(
                shared_features,
                dtype=shared_vector.dtype,
                device=shared_vector.device,
            ).unsqueeze(0)
        else:
            filter_context_vector = torch.as_tensor(
                filter_context_vector,
                dtype=shared_vector.dtype,
                device=shared_vector.device,
            )
            if filter_context_vector.ndim == 1:
                filter_context_vector = filter_context_vector.unsqueeze(0)
        if filter_context_vector.ndim != 2 or int(filter_context_vector.shape[1]) != self.filter_context_dim:
            raise ValueError(
                f"filter_context_vector must have shape ({self.filter_context_dim},) or "
                f"(batch, {self.filter_context_dim}), got {tuple(filter_context_vector.shape)}"
            )
        if int(filter_context_vector.shape[0]) != int(shared_vector.shape[0]):
            raise ValueError("shared_vector and filter_context_vector batch dimensions must match")

        # ---- 4. 提取快/慢路径隐层特征（缺失时回退到 shared_vector，支持独立 head 测试）----
        fast_source = shared_features.get("final_hidden")
        slow_source = shared_features.get("pooled_hidden")
        if fast_source is None:
            fast_source = shared_vector
        if slow_source is None:
            slow_source = shared_vector
        fast_source = torch.as_tensor(fast_source, dtype=shared_vector.dtype, device=shared_vector.device)
        slow_source = torch.as_tensor(slow_source, dtype=shared_vector.dtype, device=shared_vector.device)
        if fast_source.ndim == 1:
            fast_source = fast_source.unsqueeze(0)
        if slow_source.ndim == 1:
            slow_source = slow_source.unsqueeze(0)
        if tuple(fast_source.shape) != tuple(shared_vector.shape):
            raise ValueError("final_hidden must align with shared_vector shape")
        if tuple(slow_source.shape) != tuple(shared_vector.shape):
            raise ValueError("pooled_hidden must align with shared_vector shape")

        # ---- 5. 按模态屏蔽上下文 ----
        modality = coerce_supported_modality(
            shared_features["current_modality"],
            name="shared_features.current_modality",
        )
        masked_observation_context_source = context_vector
        masked_branch_context_source = context_vector
        masked_filter_context_source = filter_context_vector
        if modality == "uwb":
            masked_observation_context_source = _apply_context_mask(context_vector, self.uwb_fast_context_mask)
            masked_branch_context_source = _apply_context_mask(context_vector, self.uwb_branch_context_mask)
            masked_filter_context_source = _apply_context_mask(filter_context_vector, self.uwb_filter_context_mask)
        elif modality == "vio":
            masked_observation_context_source = _apply_context_mask(context_vector, self.vio_fast_context_mask)
            masked_branch_context_source = _apply_context_mask(context_vector, self.vio_branch_context_mask)
            masked_filter_context_source = _apply_context_mask(filter_context_vector, self.vio_filter_context_mask)

        # ---- 6. 计算上下文调制量 ----
        temporal_context = _slice_context_vector(masked_branch_context_source, self.temporal_context_indices)
        observation_context = _slice_context_vector(masked_observation_context_source, self.observation_context_indices)
        temporal_delta = torch.tanh(self.temporal_context_gate(temporal_context))
        observation_delta = torch.tanh(self.observation_context_gate(observation_context))
        fast_shared = fast_source * (1.0 + (self.context_modulation_scale * observation_delta))
        slow_shared = slow_source * (1.0 + (self.context_modulation_scale * temporal_delta))
        slow_filter_delta = torch.tanh(self.filter_context_gate(masked_filter_context_source))

        # ---- 7. 分支混合门控 ----
        # 对齐 docs/liquid_architecture_current.md §10.6：
        # uwb/vio 走模态专属 mix gate；其他模态走 branch_mix_gate(context_vector)。
        # 注：coerce_supported_modality 已限制 modality 仅 uwb/vio，else 分支为防御性回退。
        branch_gate_input = torch.cat((masked_branch_context_source, masked_filter_context_source), dim=-1)
        if modality == "uwb":
            slow_branch_weight = torch.sigmoid(self.uwb_branch_mix_gate(branch_gate_input))
        elif modality == "vio":
            slow_branch_weight = torch.sigmoid(self.vio_branch_mix_gate(branch_gate_input))
        else:
            slow_branch_weight = torch.sigmoid(self.branch_mix_gate(context_vector))
        fused_shared = ((1.0 - slow_branch_weight) * fast_shared) + (slow_branch_weight * slow_shared)

        # ---- 8. 滤波上下文统一调制 + 投影 ----
        filter_modulated_shared = fused_shared * (
            1.0 + (self.filter_context_modulation_scale * slow_filter_delta)
        )
        # D4 v2 redesign: residual_projection 接收 final_hidden + pooled_hidden 拼接
        dc_residual = torch.cat((fast_source, slow_source), dim=-1)
        output = self.projection(
            torch.cat((filter_modulated_shared, masked_branch_context_source, masked_filter_context_source), dim=-1)
        ) + self.residual_projection(dc_residual)
        output = output.reshape(-1)
        return output.squeeze(0) if int(output.shape[0]) == 1 else output

    def forward_batch(self, shared_features_list: list[Any]) -> torch.Tensor:
        """批量前向传播：从多个样本的共享特征计算批量标量输出。

        与逐个调用 forward 不同，此方法将多个样本的 shared_vector、
        context_vector、filter_context_vector 堆叠成批次张量，
        通过模态指示器处理混合模态的上下文掩码和分支门控选择，
        实现真正的批量矩阵运算。

        参数
        ----------
        shared_features_list : list[Any]
            共享特征字典列表，每个元素由 extract_shared_features_batch 产出。

        返回值
        -------
        torch.Tensor
            形状 (B,) 的标量输出张量。

        异常
        ------
        ValueError
            当列表为空或张量形状不匹配时抛出。
        """
        if not shared_features_list:
            raise ValueError("shared_features_list must be non-empty")
        if len(shared_features_list) == 1:
            # forward 对单样本返回 0-d 标量 ()；forward_batch 合同承诺 (B,) 形状，
            # 与下方批量路径 output.squeeze(-1) 对 B=1 返回 (1,) 保持一致，需展平为 1D。
            return self.forward(shared_features_list[0]).reshape(-1)

        batch_size = len(shared_features_list)

        # ---- 1. 堆叠 shared_vector (B, hidden_dim) ----
        shared_vectors = torch.stack([
            torch.as_tensor(sf["shared_vector"], dtype=torch.float32)
            for sf in shared_features_list
        ])
        if shared_vectors.ndim != 2 or int(shared_vectors.shape[1]) != self.hidden_dim:
            raise ValueError(
                f"shared_vector batch must have shape (B, {self.hidden_dim}), "
                f"got {tuple(shared_vectors.shape)}"
            )

        # ---- 2. 堆叠 context_vector (B, context_dim) ----
        context_vectors_list: list[torch.Tensor] = []
        for sf in shared_features_list:
            cv = sf.get("context_vector")
            if cv is None:
                cv = _build_liquid_context_tensor(sf, dtype=shared_vectors.dtype, device=shared_vectors.device)
            else:
                cv = torch.as_tensor(cv, dtype=shared_vectors.dtype, device=shared_vectors.device)
                if cv.ndim == 1:
                    cv = cv.unsqueeze(0)
            context_vectors_list.append(cv.squeeze(0) if cv.ndim == 2 and int(cv.shape[0]) == 1 else cv)
        context_vectors = torch.stack(context_vectors_list)
        if context_vectors.ndim != 2 or int(context_vectors.shape[1]) != self.context_dim:
            raise ValueError(
                f"context_vector batch must have shape (B, {self.context_dim}), "
                f"got {tuple(context_vectors.shape)}"
            )

        # ---- 3. 堆叠 filter_context_vector (B, filter_context_dim) ----
        filter_context_vectors_list: list[torch.Tensor] = []
        for sf in shared_features_list:
            fcv = sf.get("filter_context_vector")
            if fcv is None:
                fcv = _build_liquid_filter_context_tensor(sf, dtype=shared_vectors.dtype, device=shared_vectors.device)
            else:
                fcv = torch.as_tensor(fcv, dtype=shared_vectors.dtype, device=shared_vectors.device)
                if fcv.ndim == 1:
                    fcv = fcv.unsqueeze(0)
            filter_context_vectors_list.append(fcv.squeeze(0) if fcv.ndim == 2 and int(fcv.shape[0]) == 1 else fcv)
        filter_context_vectors = torch.stack(filter_context_vectors_list)
        if filter_context_vectors.ndim != 2 or int(filter_context_vectors.shape[1]) != self.filter_context_dim:
            raise ValueError(
                f"filter_context_vector batch must have shape (B, {self.filter_context_dim}), "
                f"got {tuple(filter_context_vectors.shape)}"
            )

        # ---- 4. 堆叠快/慢路径隐层特征 (B, hidden_dim)（缺失时回退到 shared_vector）----
        fast_sources = torch.stack([
            torch.as_tensor(
                sf.get("final_hidden") if sf.get("final_hidden") is not None else sf["shared_vector"],
                dtype=shared_vectors.dtype, device=shared_vectors.device,
            )
            for sf in shared_features_list
        ])
        slow_sources = torch.stack([
            torch.as_tensor(
                sf.get("pooled_hidden") if sf.get("pooled_hidden") is not None else sf["shared_vector"],
                dtype=shared_vectors.dtype, device=shared_vectors.device,
            )
            for sf in shared_features_list
        ])

        # ---- 5. 构建模态指示器 (B, 2): [is_uwb, is_vio] ----
        modality_flags = torch.zeros(batch_size, 2, dtype=shared_vectors.dtype, device=shared_vectors.device)
        for i, sf in enumerate(shared_features_list):
            modality = coerce_supported_modality(
                sf["current_modality"],
                name=f"shared_features_list[{i}].current_modality",
            )
            if modality == "uwb":
                modality_flags[i, 0] = 1.0
            elif modality == "vio":
                modality_flags[i, 1] = 1.0
        is_uwb = modality_flags[:, 0:1]  # (B, 1)
        is_vio = modality_flags[:, 1:2]  # (B, 1)

        # ---- 6. 按模态批量屏蔽上下文 ----
        uwb_fast_mask = self.uwb_fast_context_mask.to(device=context_vectors.device, dtype=context_vectors.dtype)
        vio_fast_mask = self.vio_fast_context_mask.to(device=context_vectors.device, dtype=context_vectors.dtype)
        obs_mask = (is_uwb * uwb_fast_mask.unsqueeze(0) + is_vio * vio_fast_mask.unsqueeze(0))
        is_other = 1.0 - is_uwb - is_vio  # (B, 1)
        obs_mask = obs_mask + is_other
        masked_observation_context = context_vectors * obs_mask.clamp(max=1.0)

        # 分支上下文掩码。
        uwb_branch_mask = self.uwb_branch_context_mask.to(device=context_vectors.device, dtype=context_vectors.dtype)
        vio_branch_mask = self.vio_branch_context_mask.to(device=context_vectors.device, dtype=context_vectors.dtype)
        branch_mask = (is_uwb * uwb_branch_mask.unsqueeze(0) + is_vio * vio_branch_mask.unsqueeze(0))
        branch_mask = branch_mask + is_other
        masked_branch_context = context_vectors * branch_mask.clamp(max=1.0)

        # 滤波上下文掩码。
        uwb_filter_mask = self.uwb_filter_context_mask.to(device=filter_context_vectors.device, dtype=filter_context_vectors.dtype)
        vio_filter_mask = self.vio_filter_context_mask.to(device=filter_context_vectors.device, dtype=filter_context_vectors.dtype)
        filter_mask = (is_uwb * uwb_filter_mask.unsqueeze(0) + is_vio * vio_filter_mask.unsqueeze(0))
        filter_mask = filter_mask + is_other
        masked_filter_context = filter_context_vectors * filter_mask.clamp(max=1.0)

        # ---- 7. 计算上下文调制量 ----
        temporal_context = _slice_context_vector(masked_branch_context, self.temporal_context_indices)
        observation_context = _slice_context_vector(masked_observation_context, self.observation_context_indices)
        temporal_delta = torch.tanh(self.temporal_context_gate(temporal_context))
        observation_delta = torch.tanh(self.observation_context_gate(observation_context))
        fast_shared = fast_sources * (1.0 + (self.context_modulation_scale * observation_delta))
        slow_shared = slow_sources * (1.0 + (self.context_modulation_scale * temporal_delta))
        slow_filter_delta = torch.tanh(self.filter_context_gate(masked_filter_context))

        # ---- 8. 分支混合门控（批量模态选择） ----
        branch_gate_input = torch.cat((masked_branch_context, masked_filter_context), dim=-1)
        uwb_gate = torch.sigmoid(self.uwb_branch_mix_gate(branch_gate_input))  # (B, 1)
        vio_gate = torch.sigmoid(self.vio_branch_mix_gate(branch_gate_input))  # (B, 1)
        generic_gate = torch.sigmoid(self.branch_mix_gate(context_vectors))  # (B, 1)
        # 按模态选择门控值。
        slow_branch_weight = (is_uwb * uwb_gate + is_vio * vio_gate + is_other * generic_gate)  # (B, 1)
        fused_shared = ((1.0 - slow_branch_weight) * fast_shared) + (slow_branch_weight * slow_shared)

        # ---- 9. 滤波上下文统一调制 + 投影 ----
        filter_modulated_shared = fused_shared * (
            1.0 + (self.filter_context_modulation_scale * slow_filter_delta)
        )
        # D4 v2 redesign: residual_projection 接收 final_hidden + pooled_hidden 拼接
        dc_residual = torch.cat((fast_sources, slow_sources), dim=-1)
        output = self.projection(
            torch.cat((filter_modulated_shared, masked_branch_context, masked_filter_context), dim=-1)
        ) + self.residual_projection(dc_residual)
        return output.squeeze(-1)  # (B,)


# ==============================================================================
# Head-Shared Architecture: backport of v3.1 head-shared Liquid model.
# ==============================================================================
# Original v3 used 4 independent LiquidOutputHead × ~865 params = ~3460 head params.
# The 4 heads shared all gate layers except their final projection+residual layers,
# so we extract the gate backbone (767 params) and keep 4 independent final
# projections (98 params each, key-dependent bias init for risk).
# ==============================================================================


class LiquidOutputBackbone(nn.Module):
    """共享的 Liquid 输出骨干网，仅含上下文门控层。

    与 LiquidOutputHead 共享所有 gate 层的结构（上下文索引、gate 权重、掩码缓冲区），
    但不含 projection 与 residual_projection——这两个保留给各自的 head。
    这样 4 个 head 可以共享同一套 backbone 参数（~767 参数），而不是 4 份副本（~3460 参数）。
    """

    _CONTEXT_MODULATION_SCALE = 0.25
    _FILTER_CONTEXT_MODULATION_SCALE = 0.25

    def __init__(self, network_cfg: Mapping[str, Any] | None = None):
        super().__init__()
        cfg = dict(network_cfg or {})
        self.hidden_dim = int(cfg.get("hidden_dim", 18))
        cfg = dict(network_cfg or {})
        self.context_dim = LIQUID_CONTEXT_DIM
        self.filter_context_dim = LIQUID_FILTER_CONTEXT_DIM
        self.context_modulation_scale = coerce_finite_scalar(
            cfg.get("context_modulation_scale", self._CONTEXT_MODULATION_SCALE),
            name="context_modulation_scale",
            min_value=0.0, max_value=1.0,
        )
        self.filter_context_modulation_scale = coerce_finite_scalar(
            cfg.get("filter_context_modulation_scale", self._FILTER_CONTEXT_MODULATION_SCALE),
            name="filter_context_modulation_scale",
            min_value=0.0, max_value=1.0,
        )
        # ---- 上下文索引 ----
        self.temporal_context_indices = _liquid_context_indices_for_features(LIQUID_TEMPORAL_CONTEXT_FEATURE_KEYS)
        self.observation_context_indices = _liquid_context_indices_for_features(LIQUID_OBSERVATION_CONTEXT_FEATURE_KEYS)
        self.temporal_context_dim = len(self.temporal_context_indices)
        self.observation_context_dim = len(self.observation_context_indices)
        self.uwb_fast_context_indices = _liquid_context_indices_for_features(LIQUID_UWB_FAST_CONTEXT_FEATURE_KEYS)
        self.vio_fast_context_indices = _liquid_context_indices_for_features(LIQUID_VIO_FAST_CONTEXT_FEATURE_KEYS)
        self.uwb_branch_context_indices = _liquid_context_indices_for_features(LIQUID_UWB_BRANCH_CONTEXT_FEATURE_KEYS)
        self.vio_branch_context_indices = _liquid_context_indices_for_features(LIQUID_VIO_BRANCH_CONTEXT_FEATURE_KEYS)
        self.uwb_filter_context_indices = _liquid_filter_context_indices_for_keys(LIQUID_UWB_FILTER_CONTEXT_KEYS)
        self.vio_filter_context_indices = _liquid_filter_context_indices_for_keys(LIQUID_VIO_FILTER_CONTEXT_KEYS)
        # ---- 门控线性层（所有 head 共享）----
        self.temporal_context_gate = nn.Linear(self.temporal_context_dim, self.hidden_dim)
        self.observation_context_gate = nn.Linear(self.observation_context_dim, self.hidden_dim)
        self.filter_context_gate = nn.Linear(self.filter_context_dim, self.hidden_dim)
        self.branch_mix_gate = nn.Linear(self.context_dim, 1)
        self.uwb_branch_mix_gate = nn.Linear(self.context_dim + self.filter_context_dim, 1)
        self.vio_branch_mix_gate = nn.Linear(self.context_dim + self.filter_context_dim, 1)
        # ---- 掩码缓冲区（不参与梯度，不持久化到 state_dict）----
        self.register_buffer(
            "uwb_fast_context_mask",
            _build_index_mask(self.context_dim, self.uwb_fast_context_indices),
            persistent=False,
        )
        self.register_buffer(
            "vio_fast_context_mask",
            _build_index_mask(self.context_dim, self.vio_fast_context_indices),
            persistent=False,
        )
        self.register_buffer(
            "uwb_branch_context_mask",
            _build_index_mask(self.context_dim, self.uwb_branch_context_indices),
            persistent=False,
        )
        self.register_buffer(
            "vio_branch_context_mask",
            _build_index_mask(self.context_dim, self.vio_branch_context_indices),
            persistent=False,
        )
        self.register_buffer(
            "uwb_filter_context_mask",
            _build_index_mask(self.filter_context_dim, self.uwb_filter_context_indices),
            persistent=False,
        )
        self.register_buffer(
            "vio_filter_context_mask",
            _build_index_mask(self.filter_context_dim, self.vio_filter_context_indices),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """重置所有骨干参数到零。"""
        with torch.no_grad():
            self.temporal_context_gate.weight.zero_()
            self.temporal_context_gate.bias.zero_()
            self.observation_context_gate.weight.zero_()
            self.observation_context_gate.bias.zero_()
            self.filter_context_gate.weight.zero_()
            self.filter_context_gate.bias.zero_()
            self.branch_mix_gate.weight.zero_()
            self.branch_mix_gate.bias.zero_()
            self.uwb_branch_mix_gate.weight.zero_()
            self.uwb_branch_mix_gate.bias.zero_()
            self.vio_branch_mix_gate.weight.zero_()
            self.vio_branch_mix_gate.bias.zero_()

    def forward_backbone(self, shared_features: Mapping[str, Any]) -> dict[str, Any]:
        """返回共享 backbone 计算后的中间张量，供各 head 的 projection 消费。

        返回字典包含:
            - fast_shared: (B, hidden_dim) 观测调制后的快路径特征
            - slow_shared: (B, hidden_dim) 时间调制后的慢路径特征
            - fused_shared: (B, hidden_dim) 门控融合后的特征
            - filter_modulated_shared: (B, hidden_dim) 滤波上下文调制后的特征
            - masked_branch_context: (B, context_dim) 分支上下文（已模态掩码）
            - masked_filter_context: (B, filter_context_dim) 滤波上下文（已模态掩码）
        """
        # 1. 提取 shared_vector
        shared_vector = shared_features.get("shared_vector")
        if shared_vector is None:
            raise ValueError("shared_features must expose shared_vector for residual output heads")
        shared_vector = torch.as_tensor(shared_vector, dtype=torch.float32)
        if shared_vector.ndim == 1:
            shared_vector = shared_vector.unsqueeze(0)
        if shared_vector.ndim != 2 or int(shared_vector.shape[1]) != self.hidden_dim:
            raise ValueError(
                f"shared_vector must have shape ({self.hidden_dim},) or (batch, {self.hidden_dim}), "
                f"got {tuple(shared_vector.shape)}"
            )

        # 2. 提取 context_vector
        context_vector = shared_features.get("context_vector")
        if context_vector is None:
            context_vector = _build_liquid_context_tensor(
                shared_features, dtype=shared_vector.dtype, device=shared_vector.device,
            ).unsqueeze(0)
        else:
            context_vector = torch.as_tensor(
                context_vector, dtype=shared_vector.dtype, device=shared_vector.device,
            )
            if context_vector.ndim == 1:
                context_vector = context_vector.unsqueeze(0)
        if context_vector.ndim != 2 or int(context_vector.shape[1]) != self.context_dim:
            raise ValueError(
                f"context_vector must have shape ({self.context_dim},) or (batch, {self.context_dim}), "
                f"got {tuple(context_vector.shape)}"
            )
        if int(context_vector.shape[0]) != int(shared_vector.shape[0]):
            raise ValueError("shared_vector and context_vector batch dimensions must match")

        # 3. 提取 filter_context_vector
        filter_context_vector = shared_features.get("filter_context_vector")
        if filter_context_vector is None:
            filter_context_vector = _build_liquid_filter_context_tensor(
                shared_features, dtype=shared_vector.dtype, device=shared_vector.device,
            ).unsqueeze(0)
        else:
            filter_context_vector = torch.as_tensor(
                filter_context_vector, dtype=shared_vector.dtype, device=shared_vector.device,
            )
            if filter_context_vector.ndim == 1:
                filter_context_vector = filter_context_vector.unsqueeze(0)
        if filter_context_vector.ndim != 2 or int(filter_context_vector.shape[1]) != self.filter_context_dim:
            raise ValueError(
                f"filter_context_vector must have shape ({self.filter_context_dim},) or "
                f"(batch, {self.filter_context_dim}), got {tuple(filter_context_vector.shape)}"
            )
        if int(filter_context_vector.shape[0]) != int(shared_vector.shape[0]):
            raise ValueError("shared_vector and filter_context_vector batch dimensions must match")

        # 4. 提取快/慢路径隐层特征
        fast_source = shared_features.get("final_hidden")
        slow_source = shared_features.get("pooled_hidden")
        if fast_source is None:
            fast_source = shared_vector
        if slow_source is None:
            slow_source = shared_vector
        fast_source = torch.as_tensor(fast_source, dtype=shared_vector.dtype, device=shared_vector.device)
        slow_source = torch.as_tensor(slow_source, dtype=shared_vector.dtype, device=shared_vector.device)
        if fast_source.ndim == 1:
            fast_source = fast_source.unsqueeze(0)
        if slow_source.ndim == 1:
            slow_source = slow_source.unsqueeze(0)
        if tuple(fast_source.shape) != tuple(shared_vector.shape):
            raise ValueError("final_hidden must align with shared_vector shape")
        if tuple(slow_source.shape) != tuple(shared_vector.shape):
            raise ValueError("pooled_hidden must align with shared_vector shape")

        # 5. 按模态屏蔽上下文
        modality = coerce_supported_modality(
            shared_features["current_modality"], name="shared_features.current_modality",
        )
        masked_observation_context_source = context_vector
        masked_branch_context_source = context_vector
        masked_filter_context_source = filter_context_vector
        if modality == "uwb":
            masked_observation_context_source = _apply_context_mask(context_vector, self.uwb_fast_context_mask)
            masked_branch_context_source = _apply_context_mask(context_vector, self.uwb_branch_context_mask)
            masked_filter_context_source = _apply_context_mask(filter_context_vector, self.uwb_filter_context_mask)
        elif modality == "vio":
            masked_observation_context_source = _apply_context_mask(context_vector, self.vio_fast_context_mask)
            masked_branch_context_source = _apply_context_mask(context_vector, self.vio_branch_context_mask)
            masked_filter_context_source = _apply_context_mask(filter_context_vector, self.vio_filter_context_mask)

        # 6. 计算上下文调制量
        temporal_context = _slice_context_vector(masked_branch_context_source, self.temporal_context_indices)
        observation_context = _slice_context_vector(masked_observation_context_source, self.observation_context_indices)
        temporal_delta = torch.tanh(self.temporal_context_gate(temporal_context))
        observation_delta = torch.tanh(self.observation_context_gate(observation_context))
        fast_shared = fast_source * (1.0 + (self.context_modulation_scale * observation_delta))
        slow_shared = slow_source * (1.0 + (self.context_modulation_scale * temporal_delta))
        slow_filter_delta = torch.tanh(self.filter_context_gate(masked_filter_context_source))

        # 7. 分支混合门控
        branch_gate_input = torch.cat((masked_branch_context_source, masked_filter_context_source), dim=-1)
        if modality == "uwb":
            slow_branch_weight = torch.sigmoid(self.uwb_branch_mix_gate(branch_gate_input))
        elif modality == "vio":
            slow_branch_weight = torch.sigmoid(self.vio_branch_mix_gate(branch_gate_input))
        else:
            slow_branch_weight = torch.sigmoid(self.branch_mix_gate(context_vector))
        fused_shared = ((1.0 - slow_branch_weight) * fast_shared) + (slow_branch_weight * slow_shared)

        # 8. 滤波上下文调制（不投影，只返回调制后的张量供 head 消费）
        filter_modulated_shared = fused_shared * (
            1.0 + (self.filter_context_modulation_scale * slow_filter_delta)
        )

        return {
            "fast_shared": fast_shared,
            "slow_shared": slow_shared,
            "fused_shared": fused_shared,
            "filter_modulated_shared": filter_modulated_shared,
            "masked_branch_context": masked_branch_context_source,
            "masked_filter_context": masked_filter_context_source,
            "fast_source": fast_source,
            "slow_source": slow_source,
        }

    def forward_backbone_batch(self, shared_features_list: list[Mapping[str, Any]]) -> dict[str, Any]:
        """批量版本的 forward_backbone。"""
        if not shared_features_list:
            raise ValueError("shared_features_list must be non-empty")
        if len(shared_features_list) == 1:
            return self.forward_backbone(shared_features_list[0])

        batch_size = len(shared_features_list)
        device = None

        # 1. 堆叠 shared_vector
        shared_vectors = torch.stack([
            torch.as_tensor(sf["shared_vector"], dtype=torch.float32)
            for sf in shared_features_list
        ])
        if device is None:
            device = shared_vectors.device
        if shared_vectors.ndim != 2 or int(shared_vectors.shape[1]) != self.hidden_dim:
            raise ValueError(
                f"shared_vector batch must have shape (B, {self.hidden_dim}), "
                f"got {tuple(shared_vectors.shape)}"
            )

        # 2. 堆叠 context_vector
        context_vectors_list: list[torch.Tensor] = []
        for sf in shared_features_list:
            cv = sf.get("context_vector")
            if cv is None:
                cv = _build_liquid_context_tensor(sf, dtype=shared_vectors.dtype, device=device)
            else:
                cv = torch.as_tensor(cv, dtype=shared_vectors.dtype, device=device)
                if cv.ndim == 1:
                    cv = cv.unsqueeze(0)
            context_vectors_list.append(cv.squeeze(0) if cv.ndim == 2 and int(cv.shape[0]) == 1 else cv)
        context_vectors = torch.stack(context_vectors_list)
        if context_vectors.ndim != 2 or int(context_vectors.shape[1]) != self.context_dim:
            raise ValueError(
                f"context_vector batch must have shape (B, {self.context_dim}), "
                f"got {tuple(context_vectors.shape)}"
            )

        # 3. 堆叠 filter_context_vector
        filter_context_vectors_list: list[torch.Tensor] = []
        for sf in shared_features_list:
            fcv = sf.get("filter_context_vector")
            if fcv is None:
                fcv = _build_liquid_filter_context_tensor(sf, dtype=shared_vectors.dtype, device=device)
            else:
                fcv = torch.as_tensor(fcv, dtype=shared_vectors.dtype, device=device)
                if fcv.ndim == 1:
                    fcv = fcv.unsqueeze(0)
            filter_context_vectors_list.append(fcv.squeeze(0) if fcv.ndim == 2 and int(fcv.shape[0]) == 1 else fcv)
        filter_context_vectors = torch.stack(filter_context_vectors_list)
        if filter_context_vectors.ndim != 2 or int(filter_context_vectors.shape[1]) != self.filter_context_dim:
            raise ValueError(
                f"filter_context_vector batch must have shape (B, {self.filter_context_dim}), "
                f"got {tuple(filter_context_vectors.shape)}"
            )

        # 4. 堆叠快/慢路径
        fast_sources = torch.stack([
            torch.as_tensor(
                sf.get("final_hidden") if sf.get("final_hidden") is not None else sf["shared_vector"],
                dtype=shared_vectors.dtype, device=device,
            )
            for sf in shared_features_list
        ])
        slow_sources = torch.stack([
            torch.as_tensor(
                sf.get("pooled_hidden") if sf.get("pooled_hidden") is not None else sf["shared_vector"],
                dtype=shared_vectors.dtype, device=device,
            )
            for sf in shared_features_list
        ])

        # 5. 模态指示器
        modality_flags = torch.zeros(batch_size, 2, dtype=shared_vectors.dtype, device=device)
        for i, sf in enumerate(shared_features_list):
            modality = coerce_supported_modality(sf["current_modality"], name=f"shared_features_list[{i}].current_modality")
            if modality == "uwb":
                modality_flags[i, 0] = 1.0
            elif modality == "vio":
                modality_flags[i, 1] = 1.0
        is_uwb = modality_flags[:, 0:1]
        is_vio = modality_flags[:, 1:2]
        is_other = 1.0 - is_uwb - is_vio

        # 6. 按模态批量屏蔽
        uwb_fast_mask = self.uwb_fast_context_mask.to(device=device, dtype=context_vectors.dtype)
        vio_fast_mask = self.vio_fast_context_mask.to(device=device, dtype=context_vectors.dtype)
        obs_mask = (is_uwb * uwb_fast_mask.unsqueeze(0) + is_vio * vio_fast_mask.unsqueeze(0))
        obs_mask = obs_mask + is_other
        masked_observation_context = context_vectors * obs_mask.clamp(max=1.0)

        uwb_branch_mask = self.uwb_branch_context_mask.to(device=device, dtype=context_vectors.dtype)
        vio_branch_mask = self.vio_branch_context_mask.to(device=device, dtype=context_vectors.dtype)
        branch_mask = (is_uwb * uwb_branch_mask.unsqueeze(0) + is_vio * vio_branch_mask.unsqueeze(0))
        branch_mask = branch_mask + is_other
        masked_branch_context = context_vectors * branch_mask.clamp(max=1.0)

        uwb_filter_mask = self.uwb_filter_context_mask.to(device=device, dtype=filter_context_vectors.dtype)
        vio_filter_mask = self.vio_filter_context_mask.to(device=device, dtype=filter_context_vectors.dtype)
        filter_mask = (is_uwb * uwb_filter_mask.unsqueeze(0) + is_vio * vio_filter_mask.unsqueeze(0))
        filter_mask = filter_mask + is_other
        masked_filter_context = filter_context_vectors * filter_mask.clamp(max=1.0)

        # 7. 批量门控计算
        temporal_context = _slice_context_vector(masked_branch_context, self.temporal_context_indices)
        observation_context = _slice_context_vector(masked_observation_context, self.observation_context_indices)
        temporal_delta = torch.tanh(self.temporal_context_gate(temporal_context))
        observation_delta = torch.tanh(self.observation_context_gate(observation_context))
        fast_shared = fast_sources * (1.0 + (self.context_modulation_scale * observation_delta))
        slow_shared = slow_sources * (1.0 + (self.context_modulation_scale * temporal_delta))
        slow_filter_delta = torch.tanh(self.filter_context_gate(masked_filter_context))

        branch_gate_input = torch.cat((masked_branch_context, masked_filter_context), dim=-1)
        uwb_gate = torch.sigmoid(self.uwb_branch_mix_gate(branch_gate_input))
        vio_gate = torch.sigmoid(self.vio_branch_mix_gate(branch_gate_input))
        generic_gate = torch.sigmoid(self.branch_mix_gate(context_vectors))
        slow_branch_weight = (is_uwb * uwb_gate + is_vio * vio_gate + is_other * generic_gate)
        fused_shared = ((1.0 - slow_branch_weight) * fast_shared) + (slow_branch_weight * slow_shared)

        filter_modulated_shared = fused_shared * (
            1.0 + (self.filter_context_modulation_scale * slow_filter_delta)
        )

        return {
            "fast_shared": fast_shared,
            "slow_shared": slow_shared,
            "fused_shared": fused_shared,
            "filter_modulated_shared": filter_modulated_shared,
            "masked_branch_context": masked_branch_context,
            "masked_filter_context": masked_filter_context,
            "fast_source": fast_sources,
            "slow_source": slow_sources,
        }


class LiquidOutputHeadLinear(nn.Module):
    """独立的最终投影层，与共享 backbone 配合使用。

    只包含 projection 和 residual_projection 两个线性层（各 ~98 参数），
    通过 key 初始化 projection 偏置（risk 头 log(0.1/0.9)，其余 0）。
    """

    def __init__(self, key: str, hidden_dim: int):
        super().__init__()
        self.key = key
        self.hidden_dim = hidden_dim
        # projection 输入 = filter_modulated_shared (hidden_dim) + masked_branch_context (context_dim) + masked_filter_context (filter_context_dim)
        proj_in_dim = hidden_dim + LIQUID_CONTEXT_DIM + LIQUID_FILTER_CONTEXT_DIM
        self.projection = nn.Linear(proj_in_dim, 1)
        self.residual_projection = nn.Linear(2 * hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """重置投影层。risk 头的 projection 偏置初始化为 log(0.1/0.9)。"""
        with torch.no_grad():
            self.projection.weight.zero_()
            bias_value = RISK_PRIOR_LOGIT if self.key == "risk" else 0.0
            if self.projection.bias is not None:
                self.projection.bias.fill_(float(bias_value))
            self.residual_projection.weight.zero_()
            if self.residual_projection.bias is not None:
                self.residual_projection.bias.zero_()

    def forward(self, backbone_output: dict[str, Any], shared_features: Mapping[str, Any] | None = None) -> torch.Tensor:
        """从 backbone 输出计算单个 head 的标量输出。

        v3.1 兼容性：当 backbone_output 为 shared_features 字典时调用 model.output_backbone
        forward_backbone 计算中间张量，但这需要传入 model reference。
        为简化 trainer.py 的迁移，调用方应通过 _LiquidModel.run_head_forward 提供 backbone_output。

        参数
        ----------
        backbone_output : dict
            forward_backbone 返回的中间张量字典。
        shared_features : Mapping, optional
            v3 legacy 调用方仍传 shared_features 字典。若提供且 backbone_output 为空 dict，
            需调用方通过 _LiquidModel.run_head_forward shim 来运行 backbone。

        返回
        -------
        torch.Tensor
            标量输出张量，形状 (B,) 或 ()。
        """
        filter_modulated_shared = backbone_output["filter_modulated_shared"]
        masked_branch_context = backbone_output["masked_branch_context"]
        masked_filter_context = backbone_output["masked_filter_context"]
        fast_source = backbone_output["fast_source"]
        slow_source = backbone_output["slow_source"]

        dc_residual = torch.cat((fast_source, slow_source), dim=-1)
        output = self.projection(
            torch.cat((filter_modulated_shared, masked_branch_context, masked_filter_context), dim=-1)
        ) + self.residual_projection(dc_residual)
        return output.squeeze(-1)

    def forward_batch(self, backbone_output: dict[str, Any], shared_features_list: list[Mapping[str, Any]] | None = None) -> torch.Tensor:
        """批量版 forward（等价，已接受 2D 张量）。"""
        return self.forward(backbone_output)

    def forward_legacy(self, shared_features: Mapping[str, Any], backbone) -> torch.Tensor:
        """v3 兼容入口：用传入的 backbone 计算 shared_features 的 backbone 输出再 forward。

        供 trainer.py 在不修改调用签名的情况下使用 head(sf, backbone=model.output_backbone)。
        """
        backbone_output = backbone.forward_backbone(shared_features)
        return self.forward(backbone_output)


_LiquidOutputBackbone = LiquidOutputBackbone
_LiquidOutputHeadLinear = LiquidOutputHeadLinear

class RiskCalibration(nn.Module):
    """风险校准模块：对原始 risk 输出施加可学习的 sigmoid 变换。

    通过可学习的斜率 a 和偏移 b，将原始 risk 值映射到 (0, 1) 区间，
    使风险分布更符合实际需求。

    参数
    ----------
    无显式参数，内部初始化 a_raw 和 b。
    """

    def __init__(self):
        super().__init__()
        self.a_raw = nn.Parameter(torch.tensor(SOFTPLUS_ONE_INVERSE, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, raw_risk: Any) -> torch.Tensor:
        """前向传播：对原始 risk 施加校准 sigmoid。

        支持标量和 (B,) 批量输入。

        参数
        ----------
        raw_risk : Any
            原始风险值，标量或 (B,) 张量。

        返回
        -------
        torch.Tensor
            校准后的风险值，范围 (0, 1)。
        """
        risk_tensor = torch.as_tensor(raw_risk, dtype=torch.float32, device=self.a_raw.device)
        if not torch.isfinite(risk_tensor).all():
            raise ValueError("raw_risk must be finite before risk calibration")
        slope = F.softplus(self.a_raw)
        return torch.sigmoid((slope * risk_tensor) + self.b)


# ---------------------------------------------------------------------------
# 文档 §6 引用别名：实现位于本模块的 _LiquidOutputHead 即 LiquidOutputHead。
# ---------------------------------------------------------------------------
_LiquidOutputHead = LiquidOutputHead


__all__ = [
    "LIQUID_CONTEXT_DIM",
    "LIQUID_CONTEXT_FEATURE_KEYS",
    "LIQUID_FILTER_CONTEXT_DIM",
    "LIQUID_MODALITY_CONTEXT_INDICES",
    "LIQUID_OBSERVATION_CONTEXT_FEATURE_KEYS",
    "LIQUID_READOUT_CONTEXT_KEYS",
    "LIQUID_TEMPORAL_CONTEXT_FEATURE_KEYS",
    "LIQUID_UWB_BRANCH_CONTEXT_FEATURE_KEYS",
    "LIQUID_UWB_FAST_CONTEXT_FEATURE_KEYS",
    "LIQUID_UWB_FILTER_CONTEXT_KEYS",
    "LIQUID_VIO_BRANCH_CONTEXT_FEATURE_KEYS",
    "LIQUID_VIO_FAST_CONTEXT_FEATURE_KEYS",
    "LIQUID_VIO_FILTER_CONTEXT_KEYS",
    "SOFTPLUS_ONE_INVERSE",
    "LiquidOutputBackbone",
    "LiquidOutputHead",
    "LiquidOutputHeadLinear",
    "RiskCalibration",
    "apply_liquid_modality_output_contract",
    "coerce_supported_modality",
]
