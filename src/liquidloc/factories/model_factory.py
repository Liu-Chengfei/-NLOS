"""模型工厂模块 —— 神经前端模型的创建、配置合并与推理入口。

职责
----
根据模型名字（``lstm_ekf`` 或 ``liquid_ekf``）和配置字典，创建对应的
模型实例（``_LSTMModel`` 或 ``_LiquidModel``），并负责：
- checkpoint 路径解析与安全加载（强制 CPU 反序列化）
- 运行时设备解析（默认 CPU，仅在显式请求且 CUDA 可用时上 GPU）
- 模型配置合并（基础配置 + checkpoint 内嵌配置）
- 输出张量归一化与协议合同约束
- 结构化特征窗口校验

输出头、共享输出骨干和风险校准实现位于模型层
``liquidloc.models.liquid.output_head``。本模块继续从该模型模块重导出同名符号，
保持旧版 ``liquidloc.factories.model_factory`` 导入路径兼容。

上游依赖
--------
- ``liquidloc.common.constants.DEFAULT_THRESHOLDS``  — 风险阈值常量
- ``liquidloc.common.paths.get_standard_dirs``       — 项目目录解析
- ``liquidloc.common.seed_utils.cuda_runtime_usable`` — CUDA 可用性检查
- ``liquidloc.common.types.ModelIntermediate``       — 推理中间结果数据类
- ``liquidloc.interfaces.model_api.ModelAPI``        — 模型接口协议
- ``liquidloc.models.lstm.network.LSTMNetwork``      — LSTM 网络实现
- ``liquidloc.models.liquid.network.LiquidNetwork``  — Liquid 网络实现

下游调用者
----------
- ``liquidloc.factories.__init__``  — 通过 ``create_model`` 统一导出
- ``liquidloc.pipelines.*``         — 各流水线通过工厂获取模型实例
- ``scripts/`` 下的训练 / 评估脚本
- ``tests/factories/*``             — 工厂层单元测试
- ``liquidloc.models.liquid.output_head`` — 输出头、共享输出骨干与风险校准实现

核心变量
--------
- ``_SUPPORTED``                        — 支持的模型名字集合
- ``_LIQUID_OUTPUT_KEYS``               — Liquid 模型输出头键名元组
- ``_SCALING_NEUTRAL_FLOOR``            — 缩放中性底值（1.0）
- ``_LIQUID_LEGACY_OPTIONAL_NETWORK_KEYS`` — 旧版 checkpoint 可选网络键
"""

from __future__ import annotations  # 允许延迟求值的类型注解，支持 str | Path 等前向引用。

from collections.abc import Iterator, Mapping  # Iterator 用于 parameters() 返回类型注解；Mapping 映射协议基类，用于判断配置是否为字典类对象。
from dataclasses import dataclass, field  # dataclass 定义轻量数据类；field 声明带默认值的字段。
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_real, is_string_like  # 统一判断标量数值和字符串类型。
from pathlib import Path  # 路径对象，用于统一处理文件路径的解析和拼接。
from typing import Any  # 任意类型标注，用于配置字段等灵活类型场景。

import torch  # PyTorch 核心库，提供张量运算和自动微分。
from torch import nn  # 神经网络模块基类，用于定义网络层和模型。

from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS  # 桥接层业务阈值字典，包含 risk_min/risk_max。
from liquidloc.common.seed_utils import cuda_runtime_usable  # CUDA 运行时可用性检查，区分"硬件有 GPU"和"运行时能用"。
from liquidloc.common.types import ModelIntermediate  # 模型推理中间结果数据类，包含 bias/risk/scaling 字段。
from liquidloc.common.constants import (  # 模型输出偏置上界常量；输出头键名单源真相（D9 根因修复）；模态名、模型名与运行时资源常量（D9 漂移根因修复）。
    BRIDGE_BIAS_MAX,
    DEVICE_CPU,
    MODEL_INTERMEDIATE_KEYS,
    MODEL_NAME_LIQUID,
    MODEL_NAME_LSTM,
    MODEL_NAME_TRANSFORMER,
    MODALITY_UWB,
    MODALITY_VIO,
    RAM_PEAK_FLOOR_MB,
)
from liquidloc.factories.model_runtime import (
    build_module_runtime_resource_meta as _shared_build_module_runtime_resource_meta,
    load_checkpoint_payload as _shared_load_checkpoint_payload,
    merge_checkpoint_model_cfg as _shared_merge_checkpoint_model_cfg,
    move_modules_to_device as _shared_move_modules_to_device,
    resolve_checkpoint_path as _shared_resolve_checkpoint_path,
    resolve_explicit_runtime_device as _shared_resolve_explicit_runtime_device,
    resolve_liquid_network_cfg as _shared_resolve_liquid_network_cfg,
)
from liquidloc.interfaces.model_api import ModelAPI  # 模型接口协议，定义 infer_intermediate / state_dict 等方法签名。
# 输出头实现位于模型层；工厂在此保留兼容性重导出。

# 输出头实现属于模型层；这里重导出旧工厂导入路径，保持现有 API。
from liquidloc.models.liquid.output_head import (  # noqa: F401
    LIQUID_CONTEXT_DIM,
    LIQUID_CONTEXT_FEATURE_KEYS,
    LIQUID_FILTER_CONTEXT_DIM,
    LIQUID_MODALITY_CONTEXT_INDICES,
    LIQUID_OBSERVATION_CONTEXT_FEATURE_KEYS,
    LIQUID_READOUT_CONTEXT_KEYS,
    LIQUID_TEMPORAL_CONTEXT_FEATURE_KEYS,
    LIQUID_UWB_BRANCH_CONTEXT_FEATURE_KEYS,
    LIQUID_UWB_FAST_CONTEXT_FEATURE_KEYS,
    LIQUID_UWB_FILTER_CONTEXT_KEYS,
    LIQUID_VIO_BRANCH_CONTEXT_FEATURE_KEYS,
    LIQUID_VIO_FAST_CONTEXT_FEATURE_KEYS,
    LIQUID_VIO_FILTER_CONTEXT_KEYS,
    SOFTPLUS_ONE_INVERSE,
    LiquidOutputBackbone,
    LiquidOutputHead,
    LiquidOutputHeadLinear,
    RiskCalibration,
    _LiquidOutputBackbone,
    _LiquidOutputHead,
    _LiquidOutputHeadLinear,
    _SCALING_NEUTRAL_FLOOR,
    _VALID_MODALITIES,
    _apply_context_mask,
    _build_index_mask,
    _build_liquid_context_tensor,
    _build_liquid_filter_context_tensor,
    _liquid_context_indices_for_features,
    _liquid_filter_context_indices_for_keys,
    _slice_context_vector,
    apply_liquid_modality_output_contract,
    coerce_supported_modality,
)

# 兼容旧版工厂模块的单源常量入口；模型层实现使用同一协议常量。
_SCALING_CEILING = float(BRIDGE_THRESHOLDS["non_current_scaling_ceiling"])

# ===========================================================================
# 以下为模型工厂原有常量与函数。
# ===========================================================================

# ---------------------------------------------------------------------------
# 模块级常量：Liquid 模型输出头的键名，顺序固定为 bias → risk → uwb_scaling → vio_scaling。
# ---------------------------------------------------------------------------
_LIQUID_OUTPUT_KEYS = MODEL_INTERMEDIATE_KEYS  # D9：引用单源真相常量，禁止本地重复定义；保留 _LIQUID_OUTPUT_KEYS 别名供本模块内部使用。

# ---------------------------------------------------------------------------
# 模块级常量：当前工厂支持的模型名字集合（引用 common.constants 单源真相，D9 漂移根因修复）。
# ---------------------------------------------------------------------------
_SUPPORTED = {MODEL_NAME_LSTM, MODEL_NAME_LIQUID, MODEL_NAME_TRANSFORMER}  # D9：引用单源常量，禁止硬编码字符串。
# §10.2 / §10.1.2 双向禁区：LSTM、Liquid、Transformer 均为单向因果模型。
# 若未来新增 Transformer 模型，必须在此处显式加入 MODEL_NAME_TRANSFORMER，
# 并在 create_model 中强制 bidirectional=False 或拒绝双向配置——禁止双向 TF
# 进入主表（spec 第 1638 行 / 第 1686 行）。
_TRANSFORMER_BIDIRECTIONAL_FORBIDDEN = True  # §10.2 双向 TF 禁区：True 表示任何 Transformer 双向配置均拒绝进主表。

# ---------------------------------------------------------------------------
# 模块级常量：旧版 checkpoint 中可选的网络层键名，加载时缺失不报错，用当前权重补齐。
# ---------------------------------------------------------------------------
_LIQUID_LEGACY_OPTIONAL_NETWORK_KEYS = (
    "pooling_gate.weight",                  # 池化门控权重
    "pooling_gate.bias",                    # 池化门控偏置
    "cell.reliability_projection.weight",   # 可靠性投影权重
    "cell.reliability_projection.bias",     # 可靠性投影偏置
    # B2 v4 (2026-07-26): cell-level state forget gate projection. 旧 checkpoint 没有这两个权重,
    # 加载时用当前模型 init 值补齐 (forget_root bias=0.0 → f_root=0.5 中性,
    # 2026-07-26 从旧 +0.5 改为 0.0 让 init 不偏向死保 prev_state, 对齐 LSTM 步级量级).
    # 注: 这让旧 epoch 111 ckpt 可加载到新 cell.py 中, inference-only 验证 v4 机制.
    "cell.forget_root_projection.weight",   # B2 v4: cell-level state forget gate 权重
    "cell.forget_root_projection.bias",      # B2 v4: cell-level state forget gate 偏置
)


# Runtime helpers live in model_runtime; these aliases preserve the factory's
# private compatibility seams for callers and tests.
_resolve_checkpoint_path = _shared_resolve_checkpoint_path
_load_checkpoint_payload = _shared_load_checkpoint_payload
_move_modules_to_device = _shared_move_modules_to_device
_merge_checkpoint_model_cfg = _shared_merge_checkpoint_model_cfg
_resolve_liquid_network_cfg = _shared_resolve_liquid_network_cfg
_build_module_runtime_resource_meta = _shared_build_module_runtime_resource_meta


def _resolve_explicit_runtime_device(cfg: Mapping[str, Any] | None) -> torch.device:
    return _shared_resolve_explicit_runtime_device(
        cfg,
        cuda_usable_fn=cuda_runtime_usable,
    )


def _coerce_scalar_tensor(value: Any, *, name: str) -> torch.Tensor:
    """将输入强制转换为标量张量（0 维 float32）。

    支持张量、Python 数值和带 ``.item()`` 方法的对象。
    输入必须是有限值。

    参数
    ----------
    value : Any
        待转换的值，可以是张量、数字或带 .item() 的对象。
    name : str
        参数名称，用于错误消息。

    返回
    -------
    torch.Tensor
        形状为 () 的 float32 标量张量。

    异常
    ------
    TypeError
        当值不是标量形状或不是数值类型时抛出。
    ValueError
        当值不是有限值时抛出。
    """
    if is_string_like(value) or isinstance(value, (bytes, bytearray)):  # 字符串(含 numpy.str_)/字节/字节数组不是数值，提前拒绝，与 _coerce_numeric_vector 口径一致。
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if torch.is_tensor(value):  # 输入已经是张量。
        if value.dtype == torch.bool:  # 布尔张量不是数值，与 is_real 排除 bool 的口径对齐。
            raise TypeError(f"{name} must be numeric, got bool tensor")
        if value.numel() != 1:  # 张量必须只含一个元素。
            raise TypeError(f"{name} must be scalar-shaped, got tensor with shape {tuple(value.shape)}")
        scalar = value.reshape(()).to(torch.float32)  # 重塑为 0 维标量张量并统一到 float32，与文档承诺和非张量路径口径一致。
    elif is_real(value):  # 输入是 Python 数值。
        scalar = torch.tensor(float(value), dtype=torch.float32)  # 转为 float32 标量张量。
    else:  # 尝试调用 .item() 方法（如 numpy 标量）。
        item = getattr(value, "item", None)  # 获取 .item 方法。
        if not callable(item):  # 没有 .item 方法，无法转为数值。
            raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
        scalar_value = item()  # 调用 .item() 取出 Python 数值。
        if not is_real(scalar_value):  # .item() 返回的不是数值。
            raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
        scalar = torch.tensor(float(scalar_value), dtype=torch.float32)  # 转为 float32 标量张量。

    if not torch.isfinite(scalar).item():  # 标量必须有限（非 NaN、非 Inf）。
        raise ValueError(f"{name} must be finite, got {scalar.item()}")  # 含实际值，便于定位非有限输入。
    return scalar  # 返回校验通过的标量张量。


def _coerce_numeric_vector(values: Any, *, name: str, cast) -> list[float] | list[int]:
    """将输入强制转换为数值列表。

    逐元素调用 ``coerce_finite_scalar`` 校验有限性，再用 ``cast``
    转换为 float 或 int。

    参数
    ----------
    values : Any
        待转换的可迭代对象。
    name : str
        参数名称，用于错误消息。
    cast : callable
        类型转换函数（float 或 int）。

    返回
    -------
    list[float] | list[int]
        转换后的数值列表。

    异常
    ------
    TypeError
        当输入为字符串(含 numpy.str_)/字节/字节数组/映射、不可迭代或元素不是数值时抛出。
    ValueError
        当元素为 inf 或 nan 等非有限值时抛出（由 ``coerce_finite_scalar`` 传播）。
    """
    if is_string_like(values) or isinstance(values, (bytes, bytearray, Mapping)):  # 字符串(含 numpy.str_)/字节/字节数组/映射不是合法数值向量，提前拒绝，防止 list() 把它们拆成字符静默穿透。
        raise TypeError(f"{name} must be a 1D numeric vector, not {type(values).__name__}")
    try:
        raw_list = list(values)  # 尝试转为列表。
    except TypeError as exc:  # 不可迭代。
        raise TypeError(f"{name} must be an iterable of numeric values, got {type(values).__name__}") from exc
    normalized = []  # 存放转换后的数值。
    for index, value in enumerate(raw_list):  # 逐元素校验和转换。
        normalized.append(cast(coerce_finite_scalar(value, name=f"{name}[{index}]")))  # 先校验有限性，再 cast。
    return normalized  # 返回转换后的数值列表。


def _coerce_mask_value(value: Any, *, name: str) -> int:
    """把外部缺失掩码统一成内部 0/1，兼容 bool 与 0/1 数值。

    与 ``liquidloc.models.liquid.network._coerce_mask_value`` 保持行为对齐：
    Python bool / numpy.bool_ / torch.bool 单元素张量统一转成 0/1 整型。
    """
    if isinstance(value, torch.Tensor) and value.dtype == torch.bool:  # torch.bool 单元素张量与 Liquid 侧 _coerce_mask_value 的 bool→0/1 语义对齐，避免被 coerce_finite_scalar 当作非法 bool 张量拒绝。
        if value.numel() != 1:
            raise TypeError(f"{name} must be a scalar bool tensor, got shape {tuple(value.shape)}")
        return int(bool(value.item()))
    if is_bool_like(value):
        return int(bool(value))
    scalar = coerce_finite_scalar(value, name=name)
    if scalar not in (0.0, 1.0):
        raise ValueError(f"{name} must be 0/1 or bool, got {scalar!r}")
    return int(scalar)


def _coerce_mask_vector(values: Any, *, name: str) -> list[int]:
    """把一维缺失掩码统一成内部 0/1 列表。

    与 ``liquidloc.models.liquid.network._coerce_mask_vector`` 保持行为对齐：
    字符串(含 numpy.str_)/字节/字节数组/映射不是合法掩码向量，提前拒绝，
    防止 ``list()`` 把它们拆成字符或整数静默穿透。
    """
    if is_string_like(values) or isinstance(values, (bytes, bytearray, Mapping)):  # 字符串(含 numpy.str_)/字节/字节数组/映射不是合法掩码向量，提前拒绝，防止 list() 把它们拆成字符或整数静默穿透。
        raise TypeError(f"{name} must be an iterable of 0/1 or bool values, not {type(values).__name__}")
    try:
        raw_list = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of 0/1 or bool values, got {type(values).__name__}") from exc
    return [_coerce_mask_value(value, name=f"{name}[{index}]") for index, value in enumerate(raw_list)]


def _neutral_floor_scaling_softplus(value: Any, *, name: str) -> torch.Tensor:
    """对缩放值施加"中性底值 softplus"变换。

    委托给 ``normalization.neutral_floor_softplus`` 的规范实现，
    此处仅负责输入校验（强制转为标量张量）并显式传入协议单源下限。

    参数
    ----------
    value : Any
        原始缩放值，会被强制转换为标量张量。
    name : str
        参数名称，用于错误消息。

    返回
    -------
    torch.Tensor
        变换后的标量张量，值 ∈ [BRIDGE_THRESHOLDS["scaling_min"], BRIDGE_THRESHOLDS["scaling_max"]]。
    """
    raw_value = _coerce_scalar_tensor(value, name=name)  # 强制转为标量张量。
    from liquidloc.models.features.normalization import neutral_floor_softplus  # 延迟导入，打破循环依赖。
    # D7/D9：下限显式取自 BRIDGE_THRESHOLDS["scaling_min"]（经 _SCALING_NEUTRAL_FLOOR），
    # 与 liquid/trainer.py::_project_train_scaling 口径一致，禁止本地字面量与协议单源真相漂移。
    return neutral_floor_softplus(raw_value, neutral_floor=_SCALING_NEUTRAL_FLOOR)


def _normalize_liquid_output_tensors(raw_outputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """归一化 Liquid 模型的原始输出张量。

    - ``bias``：直接强制为标量张量
    - ``risk``：线性映射到 [risk_min, risk_max] 区间
    - ``uwb_scaling`` / ``vio_scaling``：施加中性底值 softplus 变换

    参数
    ----------
    raw_outputs : Mapping[str, Any]
        原始输出字典，必须包含 bias、risk、uwb_scaling、vio_scaling 四个键。

    返回
    -------
    dict[str, torch.Tensor]
        归一化后的输出字典。
    """
    bias = _coerce_scalar_tensor(raw_outputs["bias"], name="bias")  # bias 直接取标量。
    bias = torch.clamp(bias, min=0.0, max=BRIDGE_BIAS_MAX)  # bias 非负约束且上限与 ModelIntermediate 验证范围对齐。
    risk = _coerce_scalar_tensor(raw_outputs["risk"], name="risk")  # risk 先取标量。
    risk_span = BRIDGE_THRESHOLDS["risk_max"] - BRIDGE_THRESHOLDS["risk_min"]  # 计算风险阈值跨度。
    if risk_span != 1.0 or BRIDGE_THRESHOLDS["risk_min"] != 0.0:  # 阈值不是 [0, 1] 时需要线性映射。
        risk = risk * risk_span + BRIDGE_THRESHOLDS["risk_min"]  # 线性映射到 [risk_min, risk_max]。
    risk = torch.clamp(risk, min=BRIDGE_THRESHOLDS["risk_min"], max=BRIDGE_THRESHOLDS["risk_max"])  # 防止浮点漂移越界。
    return {
        "bias": bias,  # 偏置标量。
        "risk": risk,  # 风险标量，已映射到阈值区间并 clamp。
        "uwb_scaling": _neutral_floor_scaling_softplus(raw_outputs["uwb_scaling"], name="uwb_scaling"),  # UWB 缩放因子，≥ 1.0。
        "vio_scaling": _neutral_floor_scaling_softplus(raw_outputs["vio_scaling"], name="vio_scaling"),  # VIO 缩放因子，≥ 1.0。
    }


def _resolve_modality_from_normalized_window(normalized_window: Any) -> str:
    """从标准化窗口中提取当前模态名称。

    LSTM 工厂推理必须与训练器和网络入口保持同一模态合同：
    只接受 "uwb" / "vio"，不能静默默认成 "uwb"。

    与 liquid/inference.py 的 _resolve_modality_from_window 对齐：
    只读窗口级 current_modality 字段（docs/liquid_architecture_current.md §4.1
    标准化结果），不回退到事件级 modality 字段（PRIMARY_EVENT_KEYS），
    避免跨层语义混淆与公平性分歧。缺失字段统一走 coerce_supported_modality
    硬报错，不静默兜底；不使用 ``or`` 短路，避免空串/0/False 等 falsy 值
    被误判为缺失而回退到事件级字段。
    """
    if isinstance(normalized_window, Mapping):
        modality = normalized_window.get("current_modality")
    elif hasattr(normalized_window, "current_modality"):
        modality = getattr(normalized_window, "current_modality")
    else:
        modality = None
    return coerce_supported_modality(modality, name="feature_window.current_modality")



def _normalize_lstm_output_tensors(raw_outputs: Mapping[str, Any], *, risk_already_calibrated: bool = False) -> dict[str, torch.Tensor]:
    """归一化 LSTM 模型的原始输出张量。

    当 ``risk_already_calibrated=False`` 时（默认，未经校准的原始 risk 输入），
    ``risk`` 使用 sigmoid 激活映射到 (0,1)。
    当 ``risk_already_calibrated=True`` 时（工厂推理路径，risk 已通过 RiskCalibration 校准），
    ``risk`` 只做线性映射，不再执行 sigmoid，与 Liquid 路径对齐。

    参数
    ----------
    raw_outputs : Mapping[str, Any]
        原始输出字典，必须包含 bias、risk、uwb_scaling、vio_scaling 四个键。
    risk_already_calibrated : bool
        risk 是否已经过 RiskCalibration 校准（含 sigmoid）。默认 False。

    返回
    -------
    dict[str, torch.Tensor]
        归一化后的输出字典。
    """
    bias = _coerce_scalar_tensor(raw_outputs["bias"], name="bias")  # bias 直接取标量。
    bias = torch.clamp(bias, min=0.0, max=BRIDGE_BIAS_MAX)  # bias 非负约束且上限与 ModelIntermediate 验证范围对齐。
    risk = _coerce_scalar_tensor(raw_outputs["risk"], name="risk")  # risk 先取标量。
    if not risk_already_calibrated:  # 未经校准的 risk：需要 sigmoid 映射到 (0, 1)。
        risk = torch.sigmoid(risk)  # sigmoid 映射到 (0, 1)。
    # 否则 risk 已被 RiskCalibration 处理过（含 sigmoid），只做线性映射。
    risk_span = BRIDGE_THRESHOLDS["risk_max"] - BRIDGE_THRESHOLDS["risk_min"]  # 计算风险阈值跨度。
    if risk_span != 1.0 or BRIDGE_THRESHOLDS["risk_min"] != 0.0:  # 阈值不是 [0, 1] 时需要线性映射。
        risk = risk * risk_span + BRIDGE_THRESHOLDS["risk_min"]  # 线性映射到 [risk_min, risk_max]。
    risk = torch.clamp(risk, min=BRIDGE_THRESHOLDS["risk_min"], max=BRIDGE_THRESHOLDS["risk_max"])  # 防止浮点漂移越界。
    return {
        "bias": bias,  # 偏置标量。
        "risk": risk,  # 风险标量，sigmoid 后映射到阈值区间并 clamp。
        "uwb_scaling": _neutral_floor_scaling_softplus(raw_outputs["uwb_scaling"], name="uwb_scaling"),  # UWB 缩放因子，≥ 1.0。
        "vio_scaling": _neutral_floor_scaling_softplus(raw_outputs["vio_scaling"], name="vio_scaling"),  # VIO 缩放因子，≥ 1.0。
    }


def _coerce_output_vector(raw_output: Any, *, output_keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """将 LSTM 模型的原始输出向量拆分为各输出头的标量张量。

    LSTM 网络输出一个向量，按固定顺序拆分为 bias、risk、uwb_scaling、vio_scaling。
    顺序与 ``MODEL_INTERMEDIATE_KEYS`` 单源真相一致（D9 配置表面漂移防护）。

    参数
    ----------
    raw_output : Any
        原始输出，可以是张量或可转换为张量的对象。
    output_keys : tuple[str, ...]
        输出头键名元组，决定拆分顺序和数量；调用方应传 ``MODEL_INTERMEDIATE_KEYS``。

    返回
    -------
    dict[str, torch.Tensor]
        键名到标量张量的映射；标量张量保留 ``raw_output`` 原设备，
        以便后续 ``risk_calibration`` 在同设备上工作（D4 设备一致性）。

    异常
    ------
    TypeError
        当 ``raw_output`` 是字符串/字节/字节数组/映射等非数值类型，
        或为布尔张量时抛出。
    ValueError
        当输出形状不符合要求、元素数与 ``output_keys`` 不匹配，
        或包含 NaN/Inf 时抛出。
    """
    # 字符串(含 numpy.str_)/字节/字节数组/映射不是合法数值向量，提前拒绝，
    # 防止 torch.as_tensor 把 bytes 拆成 uint8 静默穿透或对 Mapping 抛无意义错误；
    # 与 _coerce_scalar_tensor L1236、_coerce_numeric_vector L1287 口径一致。
    if is_string_like(raw_output) or isinstance(raw_output, (bytes, bytearray, Mapping)):
        raise TypeError(
            f"raw_output must be a numeric tensor or array-like, "
            f"got {type(raw_output).__name__}"
        )
    # 布尔张量不是合法数值输出，提前拒绝，与 _coerce_scalar_tensor L1239 口径一致；
    # 防止 torch.as_tensor(dtype=float32) 把 bool 静默转换为 0.0/1.0。
    if torch.is_tensor(raw_output) and raw_output.dtype == torch.bool:
        raise TypeError("raw_output must be numeric, got bool tensor")
    # 强制转为 float32 张量；torch.as_tensor 保留输入原设备（CPU 张量、GPU 张量均保持原设备），
    # 以便后续 risk_calibration 在同设备上工作（D4：避免设备静默漂移）。
    raw_tensor = torch.as_tensor(raw_output, dtype=torch.float32)
    if raw_tensor.ndim == 2:  # 如果是 2D（带 batch 维）。
        if int(raw_tensor.shape[0]) != 1:  # batch 维必须为 1。
            raise ValueError(
                "model output batch dimension must be 1 for infer_intermediate, "
                f"got shape {tuple(raw_tensor.shape)}"
            )
        raw_tensor = raw_tensor.squeeze(0)  # 去掉 batch 维。
    elif raw_tensor.ndim > 2:  # 超过 2D 视为合同违反：网络应输出 1D 向量或 (1, K) batch 形态。
        raise ValueError(
            f"model output must be 1D or 2D with batch=1, "
            f"got {raw_tensor.ndim}D tensor with shape {tuple(raw_tensor.shape)}"
        )
    raw_tensor = raw_tensor.reshape(-1)  # 展平为 1D；MATLAB reshape(X, []) 等价语义。
    if raw_tensor.numel() != len(output_keys):  # 元素数必须等于输出头数量。
        raise ValueError(
            "model output must contain the fixed four heads ordered as "
            "bias, risk, uwb_scaling, vio_scaling; "
            f"got tensor with shape {tuple(raw_tensor.shape)} and numel={raw_tensor.numel()}, "
            f"expected {len(output_keys)} (output_keys={output_keys})"
        )
    # NaN/Inf 拒绝：torch.as_tensor(NaN) 不拒绝 NaN，reshape 会传播 NaN（project_memory 已记录）；
    # 与 _coerce_scalar_tensor L1255-1256 和 models/lstm/inference.py L83-84 口径一致，
    # 防止非有限值流入 risk_calibration / _normalize_lstm_output_tensors 造成静默污染。
    # 按输出头顺序逐个检查，错误消息包含 head 名（与 _coerce_scalar_tensor 的 "{name} must be finite, got {value}" 格式对齐），
    # 保证下游测试 match="uwb_scaling must be finite" 等模式可命中。
    for index, key in enumerate(output_keys):
        value_tensor = raw_tensor[index]
        if not torch.isfinite(value_tensor).item():
            raise ValueError(f"{key} must be finite, got {value_tensor.item()}")
    return {key: raw_tensor[index].reshape(()) for index, key in enumerate(output_keys)}  # 按顺序拆分为标量。


def _coerce_window_rows(rows: Any, *, name: str, feature_dim: int, cast) -> list[list[Any]]:
    """将窗口行数据校验并转换为数值二维列表。

    每行的长度必须等于 ``feature_dim``，每个元素必须是有限数值。

    参数
    ----------
    rows : Any
        窗口行数据，必须是可迭代的可迭代对象。
    name : str
        参数名称，用于错误消息。
    feature_dim : int
        每行应有的特征维度。
    cast : callable
        类型转换函数（float 或 int）。

    返回
    -------
    list[list[Any]]
        校验后的二维数值列表。

    异常
    ------
    TypeError
        当输入不可迭代时抛出。
    ValueError
        当输入为空或行长度不匹配时抛出。
    """
    if is_string_like(rows) or isinstance(rows, (bytes, bytearray, Mapping)):  # 字符串(含 numpy.str_)/字节/字节数组/映射不是合法二维结构，提前拒绝，防止 list() 把它们拆成字符或整数静默穿透。
        raise TypeError(f"{name} must be a 2D structured window, not {type(rows).__name__}")
    try:
        row_list = list(rows)  # 尝试转为列表。
    except TypeError as exc:  # 不可迭代。
        raise TypeError(f"{name} must be a 2D structured window, got {type(rows).__name__}") from exc
    if not row_list:  # 空窗口不允许。
        raise ValueError(f"{name} must be non-empty")

    normalized_rows = []  # 存放校验后的行。
    for row_index, row in enumerate(row_list):  # 逐行校验，保留行号用于错误定位。
        if is_string_like(row) or isinstance(row, (bytes, bytearray, Mapping)):  # 字符串/字节/映射不是合法数值行，提前拒绝，防止 list() 把 bytes 拆成整数静默穿透。
            raise TypeError(f"{name}[{row_index}] must be a row of numeric values, not {type(row).__name__}")
        try:
            values = list(row)  # 将行转为列表。
        except TypeError as exc:  # 行不可迭代时直接报错，便于定位。
            raise TypeError(f"{name}[{row_index}] must be a row of numeric values, got {type(row).__name__}") from exc
        if len(values) != feature_dim:  # 行长度必须与特征维度对齐。
            raise ValueError(
                f"{name}[{row_index}] must align with feature_order, got len={len(values)}, expected={feature_dim}"
            )
        normalized_rows.append(
            [
                cast(coerce_finite_scalar(value, name=f"{name}[{row_index}][{col_index}]"))
                for col_index, value in enumerate(values)
            ]
        )  # 逐值校验和转换，错误信息携带行列索引便于定位。
    return normalized_rows  # 返回校验后的二维列表。


def _coerce_mask_rows(rows: Any, *, name: str, feature_dim: int) -> list[list[int]]:
    """把二维缺失掩码窗口统一成内部 0/1 列表。"""
    if is_string_like(rows) or isinstance(rows, (bytes, bytearray, Mapping)):  # 字符串(含 numpy.str_)/字节/字节数组/映射不是合法二维结构，提前拒绝，防止 list() 把它们拆成字符或整数静默穿透。
        raise TypeError(f"{name} must be a 2D structured window, not {type(rows).__name__}")
    try:
        row_list = list(rows)
    except TypeError as exc:
        raise TypeError(f"{name} must be a 2D structured window, got {type(rows).__name__}") from exc
    if not row_list:
        raise ValueError(f"{name} must be non-empty")

    normalized_rows: list[list[int]] = []
    for row_index, row in enumerate(row_list):
        values = _coerce_mask_vector(row, name=f"{name}[{row_index}]")
        if len(values) != feature_dim:
            raise ValueError(
                f"{name} rows must align with feature_order, "
                f"got row len {len(values)}, expected {feature_dim}"
            )
        normalized_rows.append(values)
    return normalized_rows


def _coerce_structured_window(window_tensor: Any) -> dict[str, Any]:
    """校验并转换结构化特征窗口映射。

    检查 ``feature_order``、``current_modality``、``feature_values``、
    ``missing_mask``、``dt``、``feature_window``、``missing_mask_window``
    等字段的存在性、类型和维度一致性。同时解析读出上下文和滤波上下文。

    参数
    ----------
    window_tensor : Any
        原始窗口数据，必须是映射。

    返回
    -------
    dict[str, Any]
        校验后的窗口字典，包含所有必要字段。

    异常
    ------
    TypeError
        当输入不是映射或字段类型错误时抛出。
    ValueError
        当字段缺失、维度不匹配或值不合法时抛出。
    """
    if not isinstance(window_tensor, Mapping):  # 窗口必须是映射。
        raise TypeError("window_tensor must be a structured feature window mapping")

    # feature_order 是必需字段，先用 `in` 检查存在性，区分“缺键”与“空列表”，
    # 避免 `or []` 把 None 静默洗成空列表后报“non-empty”的误导消息（D3 数据合同保真）。
    if "feature_order" not in window_tensor:
        raise ValueError("feature_window.feature_order must be provided explicitly")
    raw_feature_order = window_tensor["feature_order"]
    # 字符串(含 numpy.str_)/字节/字节数组/映射本身可迭代但会被 list() 拆成字符/整数/键序列，
    # 必须提前拒绝，与 lstm/network.py _coerce_feature_order 和 _coerce_numeric_vector 的拒绝口径对齐（D3/D8）。
    if (
        is_string_like(raw_feature_order)
        or isinstance(raw_feature_order, (bytes, bytearray))
        or isinstance(raw_feature_order, Mapping)
    ):
        raise TypeError(
            f"feature_window.feature_order must be an iterable of feature names, "
            f"got {type(raw_feature_order).__name__}"
        )
    try:
        feature_order = list(raw_feature_order)  # 固化成列表，后面不会再被外部迭代器耗尽。
    except TypeError as exc:
        raise TypeError(
            f"feature_window.feature_order must be an iterable of feature names, "
            f"got {type(raw_feature_order).__name__}"
        ) from exc
    if not feature_order:  # 特征顺序不能为空。
        raise ValueError("feature_window.feature_order must be non-empty")
    for index, name in enumerate(feature_order):  # 每个特征名必须是非空字符串。
        if not is_string_like(name) or not name:
            raise ValueError(
                f"feature_window.feature_order[{index}] must be a non-empty string, "
                f"got {type(name).__name__}: {name!r}"
            )
    if len(set(feature_order)) != len(feature_order):  # 特征名不允许重复，与 lstm/network.py _coerce_feature_order 对齐（D7 公平性）。
        duplicate_names = sorted({n for n in feature_order if feature_order.count(n) > 1})
        raise ValueError(
            f"feature_window.feature_order must not contain duplicate field names: {duplicate_names}"
        )

    current_modality = coerce_supported_modality(
        window_tensor.get("current_modality"),
        name="feature_window.current_modality",
    )  # 取出并校验当前模态，只允许协议支持的 UWB/VIO。

    feature_dim = len(feature_order)  # 特征维度 = 特征名数量。
    feature_values_raw = window_tensor.get("feature_values")  # 取出当前步特征值。
    missing_mask_raw = window_tensor.get("missing_mask")  # 取出当前步缺失掩码。
    if feature_values_raw is None:  # 特征值必须提供。
        raise ValueError("feature_window.feature_values must be provided explicitly")
    if missing_mask_raw is None:  # 缺失掩码必须提供。
        raise ValueError("feature_window.missing_mask must be provided explicitly")

    feature_values = _coerce_numeric_vector(feature_values_raw, name="feature_window.feature_values", cast=float)  # 校验特征值。
    missing_mask = _coerce_mask_vector(missing_mask_raw, name="feature_window.missing_mask")  # 校验缺失掩码。
    if len(feature_values) != feature_dim or len(missing_mask) != feature_dim:  # 长度必须与特征维度对齐。
        raise ValueError(
            "feature_window feature_order, feature_values, and missing_mask must align: "
            f"feature_dim={feature_dim}, len(feature_values)={len(feature_values)}, "
            f"len(missing_mask)={len(missing_mask)}"
        )

    dt = window_tensor.get("dt")  # 取出时间步长。
    if dt is None:  # dt 必须显式提供。
        raise ValueError("feature_window.dt must be provided explicitly")

    if "feature_window" not in window_tensor:  # 历史特征窗口必须提供。
        raise ValueError("feature_window.feature_window must be provided")
    if "missing_mask_window" not in window_tensor:  # 历史缺失掩码窗口必须提供。
        raise ValueError("feature_window.missing_mask_window must be provided")

    feature_window = _coerce_window_rows(  # 校验历史特征窗口。
        window_tensor.get("feature_window"),
        name="feature_window.feature_window",
        feature_dim=feature_dim,
        cast=float,
    )
    missing_mask_window = _coerce_mask_rows(  # 校验历史缺失掩码窗口。
        window_tensor.get("missing_mask_window"),
        name="feature_window.missing_mask_window",
        feature_dim=feature_dim,
    )
    if len(feature_window) != len(missing_mask_window):  # 两个窗口行数必须一致。
        raise ValueError(
            "feature_window.feature_window and missing_mask_window must have the same row count: "
            f"got feature_window rows={len(feature_window)}, missing_mask_window rows={len(missing_mask_window)}"
        )
    if feature_values != feature_window[-1]:  # 当前步特征值必须与窗口最后一行一致，与 lstm/network.py L297 拆分式校验对齐（D7 公平性）。
        raise ValueError(
            "feature_window current-step feature_values must match the last window row: "
            f"feature_values={feature_values!r}, feature_window[-1]={feature_window[-1]!r}"
        )
    if missing_mask != missing_mask_window[-1]:  # 当前步掩码必须与窗口最后一行一致，与 lstm/network.py L299 拆分式校验对齐（D7 公平性）。
        raise ValueError(
            "feature_window current-step missing_mask must match the last window row: "
            f"missing_mask={missing_mask!r}, missing_mask_window[-1]={missing_mask_window[-1]!r}"
        )

    event_time_window = window_tensor.get("event_time_window")  # 取出事件时间窗口（可选）。
    if event_time_window is not None:  # 如果提供了事件时间窗口。
        # _coerce_numeric_vector 内部已对每个元素调用 coerce_finite_scalar 校验有限性，
        # 并对 str/bytes/Mapping/不可迭代输入抛 TypeError、对 NaN/Inf 抛 ValueError，
        # 错误消息已带元素下标；此处不再二次包装，与 network.py _coerce_optional_time_window L474-478 单一真相源保持一致（D2 分层边界）。
        event_time_window = _coerce_numeric_vector(  # 校验为浮点列表。
            event_time_window,
            name="feature_window.event_time_window",
            cast=float,
        )
        if len(event_time_window) != len(feature_window):  # 时间窗口长度必须与特征窗口行数对齐。
            raise ValueError(
                "feature_window.event_time_window must align with the feature window row count: "
                f"got event_time_window len={len(event_time_window)}, feature_window rows={len(feature_window)}"
            )

    raw_readout_context = window_tensor.get("readout_context_by_name")  # 取出读出上下文映射。
    raw_readout_observed = window_tensor.get("readout_context_observed_by_name")  # 取出读出上下文观测标志映射。
    readout_context_by_name: dict[str, float] = {}  # 存放解析后的读出上下文值。
    readout_context_observed_by_name: dict[str, bool] = {}  # 存放解析后的读出上下文观测标志。
    for key in LIQUID_READOUT_CONTEXT_KEYS:  # 逐个解析读出上下文键。
        observed = bool(
            raw_readout_observed.get(key, False) if isinstance(raw_readout_observed, Mapping) else False
        )  # 先解析观测标志，未观测值必须回退中性值。
        has_raw_value = isinstance(raw_readout_context, Mapping) and key in raw_readout_context  # 显式区分“缺值”和“值非法”。
        raw_value = 0.0 if not has_raw_value else raw_readout_context.get(key, 0.0)  # 缺值时回退 0，但 observed 也要一并清掉。
        try:
            scalar = coerce_finite_scalar(raw_value, name=f"readout_context_by_name.{key}")  # 尝试转为有限浮点。
        except (TypeError, ValueError):  # 转换失败时回退为 0。
            scalar = 0.0
            observed = False  # 非法值不能继续当成“已观测”上下文，否则会污染读出门控。
        if not has_raw_value:
            observed = False  # 缺值统一降为未观测，与 Liquid 主链标准化保持一致。
        # coerce_finite_scalar 已保证 scalar 为有限 float；未观测值强制回退中性数值。
        # 与 network.py normalize_window_tensor L325 同口径，避免冗余 math.isfinite 检查（D2 单一真相源）。
        readout_context_by_name[key] = scalar if observed else 0.0
        readout_context_observed_by_name[key] = observed
    explicit_context_vector = window_tensor.get("context_vector")  # 取出显式上下文向量（可选）。
    explicit_filter_context_vector = window_tensor.get("filter_context_vector")  # 取出显式滤波上下文向量（可选）。
    # 校验显式上下文向量类型：拒绝 str/bytes/Mapping（虽有 __len__ 但不是合法数值向量），
    # 与 network.py normalize_window_tensor L327-343 同口径，避免非法类型静默穿透到读出层。
    if explicit_context_vector is not None:
        if isinstance(explicit_context_vector, (str, bytes, Mapping)) or (
            not isinstance(explicit_context_vector, (list, tuple))
            and not hasattr(explicit_context_vector, "__len__")
        ):
            raise TypeError(
                f"context_vector should be a sequence, got {type(explicit_context_vector).__name__}"
            )
    if explicit_filter_context_vector is not None:
        if isinstance(explicit_filter_context_vector, (str, bytes, Mapping)) or (
            not isinstance(explicit_filter_context_vector, (list, tuple))
            and not hasattr(explicit_filter_context_vector, "__len__")
        ):
            raise TypeError(
                f"filter_context_vector should be a sequence, got {type(explicit_filter_context_vector).__name__}"
            )

    # 显式区分 None 与空列表，避免 [] 被 `or` 静默替换为 range
    # （与 network.py normalize_window_tensor L345-349 同口径；project_memory: or 短路在 falsy 值上有问题）。
    raw_window_index_map = window_tensor.get("window_index_map")
    window_index_map = (
        list(raw_window_index_map) if raw_window_index_map is not None
        else list(range(len(feature_window)))
    )

    return {  # 返回校验后的完整窗口字典。
        "feature_order": feature_order,  # 特征顺序列表。
        "feature_values": feature_values,  # 当前步特征值。
        "missing_mask": missing_mask,  # 当前步缺失掩码。
        "dt": coerce_finite_scalar(dt, name="feature_window.dt", min_value=0.0),  # 时间步长，非负，与 lstm/network.py normalize_structured_window 和 liquid/network.py _resolve_step_dts 的 fallback_dt 非负口径对齐（D7 公平性）。
        "feature_window": feature_window,  # 历史特征窗口。
        "missing_mask_window": missing_mask_window,  # 历史缺失掩码窗口。
        "current_modality": current_modality,  # 当前模态。
        "window_index_map": window_index_map,  # 窗口索引映射，None 时回退为顺序索引。
        "event_time_window": event_time_window,  # 事件时间窗口（可能为 None）。
        "readout_context_by_name": readout_context_by_name,  # 读出上下文值字典。
        "readout_context_observed_by_name": readout_context_observed_by_name,  # 读出上下文观测标志字典。
        "context_vector": explicit_context_vector,  # 显式上下文向量（可能为 None）。
        "filter_context_vector": explicit_filter_context_vector,  # 显式滤波上下文向量（可能为 None）。
    }


# 未构建网络时仍需要稳定的默认资源元数据；资源估算实现统一位于 model_runtime。
_RUNTIME_RESOURCE_DEFAULT_PARAMS: float = 0.0  # 网络未构建时的默认参数量。


class _NeuralModelLifecycleMixin:
    """Share the lifecycle contract of the LSTM and Transformer backends.

    The two sequence backends differ only in network construction and legacy
    network-state preparation. Keeping those hooks local to each backend
    avoids duplicating checkpoint, device, and output-contract behavior.
    """

    def _initialize_model(self, *, network_type: str, log_label: str) -> None:
        from liquidloc.common.tee_logger import print_dict

        network_cfg = self.cfg.get("network") if isinstance(self.cfg, dict) else None
        print_dict(
            {
                "name": self.name,
                "network_type": network_type,
                "network_hidden_dim": (network_cfg or {}).get("hidden_dim")
                if isinstance(network_cfg, dict)
                else None,
                "network_input_dim": (network_cfg or {}).get("input_dim")
                if isinstance(network_cfg, dict)
                else None,
                "has_checkpoint_path": "checkpoint_path" in (self.cfg or {})
                if isinstance(self.cfg, dict)
                else False,
                "checkpoint_path": (self.cfg or {}).get("checkpoint_path")
                if isinstance(self.cfg, dict)
                else None,
                "window": (self.cfg or {}).get("window") if isinstance(self.cfg, dict) else None,
            },
            log_label,
        )
        raw_feature_order = self.cfg.get("feature_order")
        if raw_feature_order is None:
            self.expected_feature_order = []
        elif isinstance(raw_feature_order, (list, tuple)):
            self.expected_feature_order = list(raw_feature_order)
        else:
            raise ValueError(
                "feature_order must be a list or tuple of strings or None, "
                f"got {type(raw_feature_order).__name__}"
            )
        self.checkpoint_meta = {}
        self.expected_feature_order_required = bool(self.expected_feature_order)

        checkpoint_path = self.cfg.get("checkpoint_path")
        if checkpoint_path is not None:
            if isinstance(checkpoint_path, str) and not checkpoint_path.strip():
                raise ValueError("checkpoint_path must not be empty when present")
            self.load_checkpoint(checkpoint_path)
        else:
            self._build_modules()
            self._apply_runtime_device()
            self._refresh_runtime_resource_meta()

    def _apply_runtime_device(self) -> None:
        device = _resolve_explicit_runtime_device(self.cfg)
        _move_modules_to_device((self.network, self.risk_calibration), device)
        self.runtime_device = str(device)

    def _refresh_runtime_resource_meta(self) -> None:
        if self.network is None:
            self.runtime_resource_meta = {
                "params": _RUNTIME_RESOURCE_DEFAULT_PARAMS,
                "ram_peak": RAM_PEAK_FLOOR_MB,
                "ram_peak_mb": RAM_PEAK_FLOOR_MB,
            }
        else:
            self.runtime_resource_meta = _build_module_runtime_resource_meta(
                (self.network, self.risk_calibration)
            )
        self.params = coerce_finite_scalar(
            self.runtime_resource_meta["params"], name="runtime_resource_meta.params"
        )
        self.ram_peak = coerce_finite_scalar(
            self.runtime_resource_meta["ram_peak"], name="runtime_resource_meta.ram_peak"
        )
        self.ram_peak_mb = coerce_finite_scalar(
            self.runtime_resource_meta["ram_peak_mb"], name="runtime_resource_meta.ram_peak_mb"
        )

    def reset(self) -> None:
        """Sequence state is reset inside each backend's forward boundary."""
        return None

    def train(self) -> None:
        self.network.train()
        self.risk_calibration.train()

    def eval(self) -> None:
        self.network.eval()
        self.risk_calibration.eval()

    def parameters(self):
        yield from self.network.parameters()
        yield from self.risk_calibration.parameters()

    def _normalize_window(self, window_tensor: Any) -> Any:
        if not self.expected_feature_order_required:
            return window_tensor
        if not isinstance(window_tensor, Mapping):
            raise TypeError(
                f"{self.name} with cfg/checkpoint expected_feature_order requires a structured "
                "feature window mapping to validate expected_feature_order; "
                f"got {type(window_tensor).__name__}"
            )
        source = _coerce_structured_window(window_tensor)
        if source["feature_order"] != self.expected_feature_order:
            raise ValueError(
                "window_tensor.feature_order must match the model expected_feature_order; "
                f"expected {self.expected_feature_order}, got {source['feature_order']}"
            )
        return source

    def predict_intermediate_tensors(self, window_tensor: Any) -> dict[str, torch.Tensor]:
        normalized_window = self._normalize_window(window_tensor)
        raw_output = self.network(normalized_window)
        raw_outputs = _coerce_output_vector(raw_output, output_keys=_LIQUID_OUTPUT_KEYS)
        raw_outputs["risk"] = self.risk_calibration(raw_outputs["risk"])
        normalized = _normalize_lstm_output_tensors(raw_outputs, risk_already_calibrated=True)
        modality = _resolve_modality_from_normalized_window(normalized_window)
        if modality == MODALITY_UWB:
            safe_vio = torch.nan_to_num(normalized["vio_scaling"], nan=0.0, posinf=0.0, neginf=0.0)
            normalized["vio_scaling"] = torch.clamp(
                safe_vio, min=_SCALING_NEUTRAL_FLOOR, max=_SCALING_CEILING
            )
        elif modality == MODALITY_VIO:
            safe_uwb = torch.nan_to_num(normalized["uwb_scaling"], nan=0.0, posinf=0.0, neginf=0.0)
            normalized["uwb_scaling"] = torch.clamp(
                safe_uwb, min=_SCALING_NEUTRAL_FLOOR, max=_SCALING_CEILING
            )
        return normalized

    def infer_intermediate(self, window_tensor: Any) -> ModelIntermediate:
        with torch.no_grad():
            normalized_outputs = self.predict_intermediate_tensors(window_tensor)
            bias_key, risk_key, uwb_scaling_key, vio_scaling_key = MODEL_INTERMEDIATE_KEYS
            return ModelIntermediate(
                bias=round(float(normalized_outputs[bias_key].detach().item()), 6),
                risk=round(float(normalized_outputs[risk_key].detach().item()), 6),
                uwb_scaling=float(normalized_outputs[uwb_scaling_key].detach().item()),
                vio_scaling=float(normalized_outputs[vio_scaling_key].detach().item()),
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "network": self.network.state_dict(),
            "risk_calibration": self.risk_calibration.state_dict(),
        }

    def _prepare_network_state_dict(self, network_state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Allow a backend to migrate network-only legacy state."""
        return network_state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")
        network_state = state_dict.get("network", state_dict)
        if not isinstance(network_state, Mapping):
            raise ValueError("state_dict must contain a network state mapping")
        self.network.load_state_dict(self._prepare_network_state_dict(network_state))

        calibration_state = state_dict.get("risk_calibration")
        if isinstance(calibration_state, Mapping):
            patched_calibration_state = dict(calibration_state)
            current_calibration_state = self.risk_calibration.state_dict()
            for state_key, state_value in current_calibration_state.items():
                patched_calibration_state.setdefault(state_key, state_value)
            self.risk_calibration.load_state_dict(patched_calibration_state)

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        path = _resolve_checkpoint_path(checkpoint_path, self.cfg)
        if not path.is_file():
            import sys as _sys

            print(
                f"[load_checkpoint] WARNING: checkpoint file not found at {path}; "
                "falling back to default cfg + random initialization. "
                "Run scripts/05_train_lstm.py / 06_train_liquid.py / "
                "07_train_transformer.py to produce a real checkpoint first.",
                file=_sys.stderr,
                flush=True,
            )
            self._build_modules()
            self._apply_runtime_device()
            self._refresh_runtime_resource_meta()
            return {}

        payload = _load_checkpoint_payload(path)
        model_cfg = payload.get("model_cfg")
        if model_cfg is None:
            model_cfg = {}
        if not isinstance(model_cfg, Mapping):
            raise TypeError("checkpoint model_cfg must be a mapping")

        self.cfg = _merge_checkpoint_model_cfg(self.cfg, model_cfg)
        feature_order = self.cfg.get("feature_order")
        if feature_order is None:
            feature_order = payload.get("feature_order")
        self.expected_feature_order = list(feature_order or [])
        self.expected_feature_order_required = bool(self.expected_feature_order)
        self._build_modules()

        state_dict = payload.get("model_state")
        if state_dict is None:
            state_dict = payload.get("state_dict")
        if state_dict is None:
            raise ValueError("checkpoint payload must contain model_state")
        self.load_state_dict(state_dict)
        self.checkpoint_meta = {
            "checkpoint_path": str(path),
            "checkpoint_format": payload.get("checkpoint_format"),
            "best_epoch": payload.get("best_epoch"),
            "best_loss": payload.get("best_loss"),
            "train_window_count": payload.get("train_window_count"),
            "val_window_count": payload.get("val_window_count"),
        }
        self._apply_runtime_device()
        self._refresh_runtime_resource_meta()
        return dict(payload)


@dataclass(slots=True)
class _LSTMModel(_NeuralModelLifecycleMixin, ModelAPI):
    """LSTM-EKF 模型的工厂实例。

    封装 LSTM 网络的创建、checkpoint 加载、推理和状态管理。
    实现 ``ModelAPI`` 接口，提供统一的 ``infer_intermediate`` 方法。

    属性
    ----
    name : str
        模型名字（"lstm_ekf"）。
    cfg : dict[str, Any]
        模型配置字典。
    params : float
        模型参数量。
    ram_peak : float
        峰值内存估算（MB）。
    ram_peak_mb : float
        峰值内存估算（MB），与 ram_peak 相同。
    runtime_resource_meta : dict[str, float]
        运行时资源元数据。
    network : Any
        LSTM 网络实例。
    expected_feature_order : list[str]
        期望的特征顺序列表。
    checkpoint_meta : dict[str, Any]
        checkpoint 元数据。
    expected_feature_order_required : bool
        是否要求特征顺序校验。
    runtime_device : str
        运行时设备名称（"cpu"、"cuda" 或 "cuda:N"，与 trainer._resolve_runtime_device 返回的设备字符串对齐）。
    """

    name: str  # 模型名字。
    cfg: dict[str, Any] = field(default_factory=dict)  # 模型配置字典，默认为空。
    params: float = 0.0  # 模型参数量。
    ram_peak: float = 1.0  # 峰值内存估算（MB）。
    ram_peak_mb: float = 1.0  # 峰值内存估算（MB），与 ram_peak 相同。
    runtime_resource_meta: dict[str, float] = field(default_factory=dict)  # 运行时资源元数据。
    network: Any = field(init=False, repr=False, default=None)  # LSTM 网络实例，不参与构造和 repr。
    risk_calibration: Any = field(init=False, repr=False, default=None)  # 风险校准模块，与 Liquid 对齐。
    expected_feature_order: list[str] = field(init=False, repr=False, default_factory=list)  # 期望的特征顺序。
    checkpoint_meta: dict[str, Any] = field(init=False, repr=False, default_factory=dict)  # checkpoint 元数据。
    expected_feature_order_required: bool = field(init=False, repr=False, default=False)  # 是否要求特征顺序校验。
    runtime_device: str = field(init=False, default=DEVICE_CPU)  # 运行时设备，默认 CPU。

    def __post_init__(self) -> None:
        self._initialize_model(
            network_type="LSTMNetwork",
            log_label="_LSTMModel.__init__ 入口参数",
        )

    def _build_modules(self) -> None:
        """构建 LSTM 网络模块和风险校准模块。"""
        from liquidloc.models.lstm.network import LSTMNetwork  # 延迟导入，避免循环依赖。

        self.network = LSTMNetwork(dict(self.cfg))  # 用配置字典创建 LSTM 网络。
        self.risk_calibration = RiskCalibration()  # 创建风险校准模块，与 Liquid 对齐。

    def _apply_runtime_device(self) -> None:
        return _NeuralModelLifecycleMixin._apply_runtime_device(self)

    def _refresh_runtime_resource_meta(self) -> None:
        return _NeuralModelLifecycleMixin._refresh_runtime_resource_meta(self)

    def reset(self) -> None:
        """重置模型级状态（前提指导 §1.4 学习隐状态按轨重置）。

        本方法为 no-op 是合规而非疏漏：LSTM/Liquid 网络的 hidden_state 不在
        model 层持久化，而是在每次 ``network.forward`` 入口由 ``reset_state``
        自动重置为零状态（见 lstm/network.py 与 liquid/network.py 的
        ``if initial_state is None: hidden_state = self.reset_state(...)``
        分支）。fusion_runner.run_fusion 每轨开始时调 ``model_infer.reset()``
        是契约挂钩点而非实际清零位置——实际清零在 forward 入口发生，因此
        跨轨 hidden_state 不会泄漏，§1.4 「按轨重置、轨间不泄漏」合规。

        若未来在 _LSTMModel/_LiquidModel 增设跨 forward 持久状态字段
        （例如推理缓存、KV cache、轨迹级统计量等），必须在本方法里显式清零,
        否则 §1.4 被破。
        """
        return _NeuralModelLifecycleMixin.reset(self)

    def train(self) -> None:
        return _NeuralModelLifecycleMixin.train(self)

    def eval(self) -> None:
        return _NeuralModelLifecycleMixin.eval(self)

    def parameters(self):
        yield from _NeuralModelLifecycleMixin.parameters(self)

    def _normalize_window(self, window_tensor: Any) -> Any:
        """校验特征窗口的特征顺序是否与期望一致。

        如果不需要特征顺序校验，直接返回原始窗口。

        参数
        ----------
        window_tensor : Any
            原始窗口数据。

        返回
        -------
        Any
            校验后的窗口字典或原始窗口。

        异常
        ------
        TypeError
            当需要校验但窗口不是映射时抛出。
        ValueError
            当特征顺序不匹配时抛出。
        """
        return _NeuralModelLifecycleMixin._normalize_window(self, window_tensor)

    def predict_intermediate_tensors(self, window_tensor: Any) -> dict[str, torch.Tensor]:
        """前向推理，返回归一化后的中间张量字典。

        参数
        ----------
        window_tensor : Any
            特征窗口数据。

        返回
        -------
        dict[str, torch.Tensor]
            归一化后的输出字典，包含 bias、risk、uwb_scaling、vio_scaling。
        """
        return _NeuralModelLifecycleMixin.predict_intermediate_tensors(self, window_tensor)

    def infer_intermediate(self, window_tensor: Any) -> ModelIntermediate:
        """无梯度推理，返回 ModelIntermediate 数据类。

        参数
        ----------
        window_tensor : Any
            特征窗口数据。

        返回
        -------
        ModelIntermediate
            推理中间结果，包含 bias、risk、uwb_scaling、vio_scaling。
        """
        return _NeuralModelLifecycleMixin.infer_intermediate(self, window_tensor)

    def state_dict(self) -> dict[str, Any]:
        """返回模型状态字典，包含网络和风险校准模块的状态。

        返回
        -------
        dict[str, Any]
            包含 "network" 和 "risk_calibration" 键的状态字典。
        """
        return _NeuralModelLifecycleMixin.state_dict(self)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """加载模型状态字典，支持旧版 checkpoint 的输出层权重填充。

        当 checkpoint 的 output_layer 权重维度小于当前模型时，
        自动用零填充到正确维度。

        参数
        ----------
        state_dict : Mapping[str, Any]
            状态字典，必须包含 "network" 键（旧版裸网络状态字典除外）。

        异常
        ------
        TypeError
            当 state_dict 不是映射时抛出。
        ValueError
            当 network 状态不是映射时抛出。
        """
        return _NeuralModelLifecycleMixin.load_state_dict(self, state_dict)

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """从磁盘加载 checkpoint，合并配置并恢复模型状态。

        参数
        ----------
        checkpoint_path : str | Path
            checkpoint 文件路径。

        返回
        -------
        dict[str, Any]
            完整的 checkpoint payload 字典（缺失 checkpoint 时返回空字典 + warning）。

        异常
        ------
        TypeError
            当 checkpoint 的 model_cfg 不是映射时抛出。
        ValueError
            当 checkpoint 缺少 model_state 时抛出。

        说明
        ----
        阶段 12 全面审计修复 (2026-09-06): 当 checkpoint_path 解析的文件不存在时
        (训练脚本 05/06/07 仅跑 smoke 不存 ckpt)，不再 raise FileNotFoundError，
        改为 warning + 返回空 payload，模型继续以默认 cfg + 随机初始化运行。
        这允许 E9 实验在缺少预训练 ckpt 时仍能跑通（基线降级），
        并打印 warning 提醒用户先跑 05/06/07 真训练并存 ckpt 后重跑。
        """
        return _NeuralModelLifecycleMixin.load_checkpoint(self, checkpoint_path)


@dataclass(slots=True)
class _LiquidModel(ModelAPI):
    """Liquid-EKF 模型的工厂实例。

    封装 Liquid 网络和多个输出头的创建、checkpoint 加载、推理和状态管理。
    与 _LSTMModel 的区别：
    - 使用 LiquidNetwork + 独立输出头（bias/risk/uwb_scaling/vio_scaling）
    - 包含 RiskCalibration 校准模块
    - 支持 legacy checkpoint 的可选键补齐
    - 支持输出头的权重维度填充

    属性
    ----
    name : str
        模型名字（"liquid_ekf"）。
    cfg : dict[str, Any]
        模型配置字典。
    params : float
        模型参数量。
    ram_peak : float
        峰值内存估算（MB）。
    ram_peak_mb : float
        峰值内存估算（MB）。
    runtime_resource_meta : dict[str, float]
        运行时资源元数据。
    network : Any
        Liquid 网络实例。
    bias_head : Any
        bias 输出头。
    risk_head : Any
        risk 输出头。
    uwb_scaling_head : Any
        UWB 缩放输出头。
    vio_scaling_head : Any
        VIO 缩放输出头。
    risk_calibration : Any
        风险校准模块。
    output_heads : dict[str, Any]
        输出头字典。
    expected_feature_order : list[str]
        期望的特征顺序列表。
    checkpoint_meta : dict[str, Any]
        checkpoint 元数据。
    expected_feature_order_required : bool
        是否要求特征顺序校验。
    runtime_device : str
        运行时设备名称（"cpu"、"cuda" 或 "cuda:N"，与 trainer._resolve_runtime_device 返回的设备字符串对齐）。
    _resource_meta_dirty : bool
        资源元数据是否需要刷新标志。
    """

    name: str  # 模型名字。
    cfg: dict[str, Any] = field(default_factory=dict)  # 模型配置字典。
    params: float = 0.0  # 模型参数量。
    ram_peak: float = 1.0  # 峰值内存估算（MB）。
    ram_peak_mb: float = 1.0  # 峰值内存估算（MB）。
    runtime_resource_meta: dict[str, float] = field(default_factory=dict)  # 运行时资源元数据。
    network: Any = field(init=False, repr=False, default=None)  # Liquid 网络实例。
    bias_head: Any = field(init=False, repr=False, default=None)  # bias 输出头。
    risk_head: Any = field(init=False, repr=False, default=None)  # risk 输出头。
    uwb_scaling_head: Any = field(init=False, repr=False, default=None)  # UWB 缩放输出头。
    vio_scaling_head: Any = field(init=False, repr=False, default=None)  # VIO 缩放输出头。
    risk_calibration: Any = field(init=False, repr=False, default=None)  # 风险校准模块。
    output_heads: dict[str, Any] = field(init=False, repr=False, default_factory=dict)  # 输出头字典。
    expected_feature_order: list[str] = field(init=False, repr=False, default_factory=list)  # 期望的特征顺序。
    checkpoint_meta: dict[str, Any] = field(init=False, repr=False, default_factory=dict)  # checkpoint 元数据。
    expected_feature_order_required: bool = field(init=False, repr=False, default=False)  # 是否要求特征顺序校验。
    runtime_device: str = field(init=False, default="cpu")  # 运行时设备。
    _resource_meta_dirty: bool = field(init=False, repr=False, default=False)  # 资源元数据脏标志。

    def __post_init__(self) -> None:
        """数据类初始化后处理：解析配置、加载 checkpoint 或构建网络。"""
        from liquidloc.common.tee_logger import print_dict
        _network_cfg = self.cfg.get("network") if isinstance(self.cfg, dict) else None
        print_dict({
            "name": self.name,
            "network_hidden_dim": (_network_cfg or {}).get("hidden_dim") if isinstance(_network_cfg, dict) else None,
            "network_input_dim": (_network_cfg or {}).get("input_dim") if isinstance(_network_cfg, dict) else None,
            "has_checkpoint_path": "checkpoint_path" in (self.cfg or {}) if isinstance(self.cfg, dict) else False,
            "checkpoint_path": (self.cfg or {}).get("checkpoint_path") if isinstance(self.cfg, dict) else None,
            "window": (self.cfg or {}).get("window") if isinstance(self.cfg, dict) else None,
        }, "_LiquidModel.__init__ 入口参数")
        raw_feature_order = self.cfg.get("feature_order")  # 取出原始特征顺序（可能为 None）。
        if raw_feature_order is None:  # 键缺失或显式 None，视为无特征顺序要求。
            self.expected_feature_order = []
        elif isinstance(raw_feature_order, (list, tuple)):  # 列表或元组，拷贝为新的 list。
            self.expected_feature_order = list(raw_feature_order)
        else:  # 其它类型（字符串、数值等）违反 feature_order 数据合同，拒绝以防 list() 把字符串拆成字符列表静默穿透（与 _LSTMModel.__post_init__ 对齐）。
            raise ValueError(
                f"feature_order must be a list or tuple of strings or None, got {type(raw_feature_order).__name__}"
            )
        self.checkpoint_meta = {}  # 初始化 checkpoint 元数据为空。
        self.expected_feature_order_required = bool(self.expected_feature_order)  # 有特征顺序时要求校验。

        checkpoint_path = self.cfg.get("checkpoint_path")  # 取出 checkpoint 路径。
        if checkpoint_path is not None:  # 配置中指定了 checkpoint。
            if isinstance(checkpoint_path, str) and not checkpoint_path.strip():  # 空字符串或纯空白视为配置错误，避免 Path("") 解析为当前目录的误导性相对路径错误。
                raise ValueError("checkpoint_path must not be empty when present")
            self.load_checkpoint(checkpoint_path)  # 加载 checkpoint。
        else:  # 无 checkpoint，从零构建。
            self._build_modules()  # 构建网络模块。
            self._apply_runtime_device()  # 应用运行时设备。
            self._refresh_runtime_resource_meta()  # 刷新资源元数据。

    def _build_modules(self) -> None:
        """构建 Liquid 网络和所有输出头模块。

        v3.1 head-shared 架构：4 个 head 共享 1 个 LiquidOutputBackbone
        （含全部 context gate + branch mix gate, ~767 params）+ 4 个独立的
        LiquidOutputHeadLinear（各 ~98 params，仅含 projection + residual_projection）。
        相比 v3 (4 个独立 LiquidOutputHead × 865 = 3460 head params)，
        head-shared 总 head params = 767 + 4×98 = 1159，减少 66.5% (~2301 params)。
        """
        from liquidloc.models.liquid.network import LiquidNetwork  # 延迟导入，避免循环依赖。

        network_cfg = _resolve_liquid_network_cfg(self.cfg)  # 解析并校验网络配置。
        self.network = LiquidNetwork(network_cfg)  # 创建 Liquid 网络。
        hidden_dim = int(self.network.hidden_dim)  # 取出隐层维度。
        # 键名走 MODEL_INTERMEDIATE_KEYS 单源真相（D9 配置表面漂移防护），与
        # _resolve_liquid_network_cfg / liquid/network.py / trainer.py / inference.py 口径对齐。
        # 上下文调制缩放因子 0.25 由 LiquidOutputBackbone 类常量固定（文档 §10.4/§10.5/§10.7），
        # 现由 LiquidOutputBackbone 从网络配置读取同名项，保留类常量作为默认值。
        # v3.1 head-shared: 4 head 共享 backbone。
        # backbone 的 hidden_dim 必须与 network 一致：从 self.network.hidden_dim 取值，
        # 而非从 cfg.get("hidden_dim", 18) 取（cfg 中可能没有 hidden_dim 键，导致默认值与 network 不一致）。
        backbone_cfg = dict(self.cfg.get("network") or {})
        backbone_cfg["hidden_dim"] = hidden_dim  # 强制与 network.hidden_dim 一致
        self.output_backbone = _LiquidOutputBackbone(backbone_cfg)  # 共享 backbone。
        self.output_heads = {  # 组装 final-projection head 字典，插入顺序与 MODEL_INTERMEDIATE_KEYS 一致。
            key: _LiquidOutputHeadLinear(key, hidden_dim) for key in MODEL_INTERMEDIATE_KEYS
        }
        # 保留命名属性引用，对齐 _apply_runtime_device / _refresh_runtime_resource_meta 消费点（docs §17.1）；state_dict 走 output_heads 迭代（D9）。
        self.bias_head = self.output_heads[MODEL_INTERMEDIATE_KEYS[0]]  # bias 输出头。
        self.risk_head = self.output_heads[MODEL_INTERMEDIATE_KEYS[1]]  # risk 输出头。
        self.uwb_scaling_head = self.output_heads[MODEL_INTERMEDIATE_KEYS[2]]  # UWB 缩放输出头。
        self.vio_scaling_head = self.output_heads[MODEL_INTERMEDIATE_KEYS[3]]  # VIO 缩放输出头。
        self.risk_calibration = RiskCalibration()  # 创建风险校准模块。
        self._resource_meta_dirty = self.network.input_dim is None  # input_dim 未确定时标记为脏。

    def _apply_runtime_device(self) -> None:
        """根据配置将所有模块移动到运行时设备。"""
        device = _resolve_explicit_runtime_device(self.cfg)  # 解析运行时设备。
        _move_modules_to_device(
            (
                self.network,  # Liquid 网络。
                self.output_backbone,  # 共享 backbone。
                self.bias_head,  # bias 输出头。
                self.risk_head,  # risk 输出头。
                self.uwb_scaling_head,  # UWB 缩放输出头。
                self.vio_scaling_head,  # VIO 缩放输出头。
                self.risk_calibration,  # 风险校准模块。
            ),
            device,
        )
        self.runtime_device = str(device)  # 记录运行时设备（保留完整字符串，含 cuda:N 索引，与 trainer._resolve_runtime_device 返回的设备字符串语义对齐）。

    def _refresh_runtime_resource_meta(self) -> None:
        """刷新运行时资源元数据（参数量和内存估算）。"""
        if self.network is None:  # 网络未构建时使用默认值。
            self.runtime_resource_meta = {
                "params": _RUNTIME_RESOURCE_DEFAULT_PARAMS,  # 默认参数量。
                "ram_peak": RAM_PEAK_FLOOR_MB,  # 默认峰值内存（MB）。
                "ram_peak_mb": RAM_PEAK_FLOOR_MB,  # 默认峰值内存（MB）。
            }
        else:  # 网络已构建，计算所有模块的实际资源。
            self.runtime_resource_meta = _build_module_runtime_resource_meta(
                (
                    self.network,  # Liquid 网络。
                    self.output_backbone,  # 共享 backbone。
                    self.bias_head,  # bias 输出头。
                    self.risk_head,  # risk 输出头。
                    self.uwb_scaling_head,  # UWB 缩放输出头。
                    self.vio_scaling_head,  # VIO 缩放输出头。
                    self.risk_calibration,  # 风险校准模块。
                )
            )
        # 用 coerce_finite_scalar 校验有限性，拒绝 NaN/Inf（D5 数值安全）；[key] 访问保证键缺失时报错而非静默失败（D3）。
        self.params = coerce_finite_scalar(self.runtime_resource_meta["params"], name="runtime_resource_meta.params")  # 更新参数量。
        self.ram_peak = coerce_finite_scalar(self.runtime_resource_meta["ram_peak"], name="runtime_resource_meta.ram_peak")  # 更新峰值内存。
        self.ram_peak_mb = coerce_finite_scalar(self.runtime_resource_meta["ram_peak_mb"], name="runtime_resource_meta.ram_peak_mb")  # 更新峰值内存（MB）。

    def reset(self) -> None:
        """重置模型级状态（前提指导 §1.4 学习隐状态按轨重置）。

        本方法为 no-op 是合规而非疏漏：Liquid 网络的 hidden_state / cell_state
        不在 model 层持久化，而是在每次 ``network.forward`` 入口由 ``reset_state``
        自动重置为零状态（见 liquid/network.py 的
        ``if initial_state is None: hidden_state = self.reset_state(...)``
        分支）。fusion_runner.run_fusion 每轨开始时调 ``model_infer.reset()``
        是契约挂钩点而非实际清零位置——实际清零在 forward 入口发生，因此
        跨轨 hidden_state / cell_state 不会泄漏，§1.4 「按轨重置、轨间不泄漏」合规。

        若未来在 _LiquidModel 增设跨 forward 持久状态字段
        （例如推理缓存、KV cache、轨迹级统计量等），必须在本方法里显式清零,
        否则 §1.4 被破。
        """
        return None

    def train(self) -> None:
        """将网络、共享 backbone、所有输出头和风险校准模块设置为训练模式。"""
        self.network.train()  # 网络设为训练模式。
        self.output_backbone.train()  # 共享 backbone 设为训练模式。
        for head in self.output_heads.values():  # 逐个输出头设为训练模式。
            head.train()
        self.risk_calibration.train()  # 风险校准也必须与主模型同步进入训练模式。

    def eval(self) -> None:
        """将网络、共享 backbone、所有输出头和风险校准模块设置为评估模式。"""
        self.network.eval()  # 网络设为评估模式。
        self.output_backbone.eval()  # 共享 backbone 设为评估模式。
        for head in self.output_heads.values():  # 逐个输出头设为评估模式。
            head.eval()
        self.risk_calibration.eval()  # 风险校准也必须与主模型同步进入评估模式。

    def parameters(self) -> Iterator[nn.Parameter]:
        """迭代器：产出网络、共享 backbone、所有输出头和风险校准的可学习参数。"""
        yield from self.network.parameters()  # 网络参数。
        yield from self.output_backbone.parameters()  # 共享 backbone 参数。
        for head in self.output_heads.values():  # 逐个输出头参数。
            yield from head.parameters()
        yield from self.risk_calibration.parameters()  # 风险校准参数。

    def _normalize_window(self, window_tensor: Any) -> Any:
        """校验特征窗口的特征顺序是否与期望一致。

        参数
        ----------
        window_tensor : Any
            原始窗口数据。

        返回
        -------
        Any
            校验后的窗口字典或原始窗口。

        异常
        ------
        TypeError
            当需要校验但窗口不是映射时抛出。
        ValueError
            当特征顺序不匹配时抛出。
        """
        if not self.expected_feature_order_required:  # 不需要特征顺序校验。
            return window_tensor
        if not isinstance(window_tensor, Mapping):  # 需要校验但窗口不是映射。
            raise TypeError(
                f"{self.name} with cfg/checkpoint expected_feature_order requires a structured "
                "feature window mapping to validate expected_feature_order; "
                f"got {type(window_tensor).__name__}"
            )
        source = _coerce_structured_window(window_tensor)  # 校验并转换窗口。
        if source["feature_order"] != self.expected_feature_order:  # 特征顺序不匹配。
            raise ValueError(
                "window_tensor.feature_order must match the model expected_feature_order; "
                f"expected {self.expected_feature_order}, got {source['feature_order']}"
            )
        return source  # 返回校验后的窗口。

    def predict_intermediate_tensors(self, window_tensor: Any) -> dict[str, torch.Tensor]:
        """前向推理，返回归一化后的中间张量字典。

        v3.1 head-shared 架构：先通过共享 backbone 计算中间张量，
        再分别通过 4 个独立的 final-projection head 产出各头标量。

        参数
        ----------
        window_tensor : Any
            特征窗口数据。

        返回
        -------
        dict[str, torch.Tensor]
            归一化且按模态合同约束后的输出字典。
        """
        shared_features = self.network.extract_shared_features(self._normalize_window(window_tensor))  # 网络提取共享特征。
        backbone_output = self.output_backbone.forward_backbone(shared_features)  # 共享 backbone 前向推理。
        raw_outputs = {key: head(backbone_output) for key, head in self.output_heads.items()}  # 各输出头 final-projection 前向推理。
        # D5 数值安全：与 LSTM 路径 _coerce_output_vector L1516-1525 的 NaN/Inf 拒绝对齐，
        # 防止非有限值流入 risk_calibration / _normalize_liquid_output_tensors 造成静默污染。
        # 提前在每个头上校验，错误消息包含 head 名（与 _coerce_scalar_tensor 的 "{name} must be finite, got {value}" 格式对齐），
        # 保证下游测试 match="uwb_scaling must be finite" 等模式可命中。
        for _head_key in list(raw_outputs.keys()):
            raw_outputs[_head_key] = _coerce_scalar_tensor(raw_outputs[_head_key], name=_head_key)
        raw_outputs["risk"] = self.risk_calibration(raw_outputs["risk"])  # risk 经过校准模块。
        normalized_outputs = _normalize_liquid_output_tensors(raw_outputs)  # 归一化输出张量。
        # D3 数据合同：current_modality 是 shared_features 的必填字段（network.py L825 保证写入），
        # 直接 [key] 访问，缺失时抛 KeyError 指向根因；禁止 .get() 静默返回 None 被
        # coerce_supported_modality 误判为"值非法"而掩盖"键缺失"的合同违反。
        modality = coerce_supported_modality(
            shared_features["current_modality"],
            name="shared_features.current_modality",
        )  # 取出并校验当前模态。
        return apply_liquid_modality_output_contract(normalized_outputs, modality=modality)  # 按模态合同约束输出。

    def infer_intermediate(self, window_tensor: Any) -> ModelIntermediate:
        """无梯度推理，返回 ModelIntermediate 数据类。

        参数
        ----------
        window_tensor : Any
            特征窗口数据。

        返回
        -------
        ModelIntermediate
            推理中间结果，包含 bias、risk、uwb_scaling、vio_scaling。
        """
        with torch.no_grad():  # 禁用梯度计算。
            normalized_outputs = self.predict_intermediate_tensors(window_tensor)  # 前向推理。
            # D7/D10：lazy refresh 移至公共推理入口，predict_intermediate_tensors 保持纯函数（无 forward 路径状态突变）。
            # input_dim 在 predict_intermediate_tensors 内的 extract_shared_features 中被解析（cell 已就绪），
            # 此处可安全刷新 runtime_resource_meta；与 LSTM 路径对齐（LSTM 无 lazy dim，不需此刷新）。
            # 直接调用方（trainer _predict_single_sample）不消费 runtime_resource_meta，故不在此处刷新也无副作用泄漏。
            if self._resource_meta_dirty and self.network.cell is not None:
                self._refresh_runtime_resource_meta()
                self._resource_meta_dirty = False
            # D9：引用 MODEL_INTERMEDIATE_KEYS 单源真相，禁止硬编码键名漂移；
            # 与 _normalize_liquid_output_tensors / apply_liquid_modality_output_contract 的键名同源，
            # 并与 LSTM 路径 _LSTMModel.infer_intermediate L2115 的解包方式对齐（D7 公平性）。
            bias_key, risk_key, uwb_scaling_key, vio_scaling_key = MODEL_INTERMEDIATE_KEYS
            return ModelIntermediate(
                bias=round(float(normalized_outputs[bias_key].detach().item()), 6),  # bias 保留 6 位小数。
                risk=round(float(normalized_outputs[risk_key].detach().item()), 6),  # risk 保留 6 位小数。
                uwb_scaling=float(normalized_outputs[uwb_scaling_key].detach().item()),  # UWB 缩放因子。
                vio_scaling=float(normalized_outputs[vio_scaling_key].detach().item()),  # VIO 缩放因子。
            )

    def run_head_forward(self, shared_features: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """v3.1 head-shared 单样本前向：先通过共享 backbone，再分发到 4 个 final-projection head。

        替代 v3 中 trainer 直接调用 ``head(sf)`` 的模式。返回与 v3 兼容的
        ``{key: scalar_tensor}`` 字典。
        """
        backbone_output = self.output_backbone.forward_backbone(shared_features)
        return {key: head(backbone_output) for key, head in self.output_heads.items()}

    def run_head_forward_batch(self, shared_features_list: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        """v3.1 head-shared 批量前向：先共享 backbone.forward_backbone_batch，再分发到 4 个 head。

        返回与 v3 兼容的 ``{key: (B,) tensor}`` 字典。
        """
        backbone_output = self.output_backbone.forward_backbone_batch(shared_features_list)
        return {key: head.forward_batch(backbone_output) for key, head in self.output_heads.items()}

    def state_dict(self) -> dict[str, Any]:
        """返回模型状态字典，包含网络、共享 backbone、所有输出头和风险校准的状态。

        v3.1 head-shared 架构新增 ``output_backbone`` 键。
        输出头键名派生自 ``MODEL_INTERMEDIATE_KEYS`` 单源真相（``f"{key}_head"``），
        与 :meth:`load_state_dict` 的消费合同一致；顺序与 ``output_heads`` 字典
        插入顺序一致（即 ``MODEL_INTERMEDIATE_KEYS`` 顺序）。

        返回
        -------
        dict[str, Any]
            包含 ``network``、``output_backbone``、四个输出头（``bias_head`` / ``risk_head`` /
            ``uwb_scaling_head`` / ``vio_scaling_head``）和 ``risk_calibration``
            键的状态字典。
        """
        state: dict[str, Any] = {
            "network": self.network.state_dict(),  # Liquid 网络状态。
            "output_backbone": self.output_backbone.state_dict(),  # v3.1: 共享 backbone 状态。
        }
        # 键名走 MODEL_INTERMEDIATE_KEYS 单源真相（D9 配置表面漂移防护），
        # 与 load_state_dict 的 `state_dict.get(f"{key}_head")` 消费点对齐。
        for key, head in self.output_heads.items():  # 逐个 final-projection head 序列化。
            state[f"{key}_head"] = head.state_dict()
        state["risk_calibration"] = self.risk_calibration.state_dict()  # 风险校准状态。
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """加载模型状态字典，支持 v3 (legacy) 和 v3.1 (head-shared) 两种 checkpoint。

        处理三种情况：
        1. 网络状态：补齐旧版可选键（pooling_gate、reliability_projection）
        2. v3.1 output_backbone 状态：旧 v3 ckpt 无此键时使用初始化默认值
           （此时 v3 的 4 个独立 head 中含相同 gate 权重，但 v3.1 已经聚合到 backbone，
           无对应映射——接受初始化默认值而非试图拆分旧 head）
        3. 输出头状态：v3 head 含 14 个 gate+projection 参数；v3.1 仅含 projection
           和 residual_projection 2 个键，故 strict=False 加载，多余键被忽略，
           projection/residual_projection 维度填充逻辑保留
        4. 风险校准状态：补齐缺失键

        参数
        ----------
        state_dict : Mapping[str, Any]
            状态字典。

        异常
        ------
        TypeError
            当 state_dict 不是映射时抛出。
        ValueError
            当 network 状态、输出头状态不是映射，或缺少必要输出头键时抛出。
        """
        if not isinstance(state_dict, Mapping):  # state_dict 必须是映射。
            raise TypeError("state_dict must be a mapping")

        # ---- 1. 加载网络状态 ----
        # 取出网络状态：当前协议要求 "network" 键；旧版裸网络状态字典无该键时回退到 state_dict 本身。
        if "network" in state_dict:
            network_state = state_dict["network"]
        else:  # 旧版 checkpoint 直接存储裸网络状态。
            network_state = state_dict
        if not isinstance(network_state, Mapping):  # 网络状态必须是映射。
            raise ValueError("state_dict must contain a network state mapping")

        current_network_state = self.network.state_dict()  # 当前网络的完整状态。
        patched_network_state = dict(network_state)  # 可变副本。
        for key in _LIQUID_LEGACY_OPTIONAL_NETWORK_KEYS:  # 逐个检查旧版可选键。
            if key not in patched_network_state and key in current_network_state:  # checkpoint 缺失但当前模型有。
                patched_network_state[key] = current_network_state[key]  # 用当前模型的值补齐。
        self.network.load_state_dict(patched_network_state)  # 加载补齐后的网络状态。

        # ---- 2. 加载 v3.1 共享 backbone 状态 ----
        # v3 checkpoint 无 output_backbone 键：保持初始化默认值（gate 权重为零）
        # 这意味着 v3 ckpt 加载到 v3.1 模型时，backbone gate 需要重新训练；
        # 为保留 v3 行为，应在加载 v3 ckpt 后手工将每个 head 的 gate 状态复制到 backbone。
        backbone_state = state_dict.get("output_backbone")
        if isinstance(backbone_state, Mapping):
            current_backbone_state = self.output_backbone.state_dict()
            patched_backbone_state = dict(backbone_state)
            for state_key, state_value in current_backbone_state.items():
                if state_key not in patched_backbone_state:
                    patched_backbone_state[state_key] = state_value
            self.output_backbone.load_state_dict(patched_backbone_state, strict=False)
        elif backbone_state is None:
            # v3 legacy ckpt: 尝试从 bias_head 提取 gate 权重复制到 backbone
            # (v3 4 个 head 中 gate 都相同，取 bias_head 即可)
            bias_head_state = state_dict.get(f"{MODEL_INTERMEDIATE_KEYS[0]}_head")
            if isinstance(bias_head_state, Mapping):
                current_backbone_state = self.output_backbone.state_dict()
                patched_backbone_state = dict(current_backbone_state)
                # 复制 gate 层（temporal_context_gate, observation_context_gate, filter_context_gate,
                # branch_mix_gate, uwb_branch_mix_gate, vio_branch_mix_gate）
                for gate_name in (
                    "temporal_context_gate.weight", "temporal_context_gate.bias",
                    "observation_context_gate.weight", "observation_context_gate.bias",
                    "filter_context_gate.weight", "filter_context_gate.bias",
                    "branch_mix_gate.weight", "branch_mix_gate.bias",
                    "uwb_branch_mix_gate.weight", "uwb_branch_mix_gate.bias",
                    "vio_branch_mix_gate.weight", "vio_branch_mix_gate.bias",
                ):
                    if gate_name in bias_head_state:
                        patched_backbone_state[gate_name] = bias_head_state[gate_name]
                self.output_backbone.load_state_dict(patched_backbone_state, strict=False)

        # ---- 3. 加载输出头状态 ----
        for key, head in self.output_heads.items():  # 逐个 final-projection head 加载。
            # 输出头状态为必填键（无 legacy 回退），显式检查键存在性（D3），值合同违例用 ValueError（D1）。
            if f"{key}_head" not in state_dict:
                raise ValueError(f"state_dict must contain {key}_head")
            head_state = state_dict[f"{key}_head"]
            if not isinstance(head_state, Mapping):  # 头状态必须是映射。
                raise ValueError(f"state_dict[{key}_head] must be a mapping")
            current_head_state = head.state_dict()  # 当前 final-projection head 的完整状态（仅 4 个键）。
            patched_head_state = dict(head_state)  # 可变副本。
            for state_key, state_value in current_head_state.items():  # 逐个检查头状态键。
                if state_key not in patched_head_state:  # checkpoint 缺失该键。
                    patched_head_state[state_key] = state_value  # 用当前值补齐。
                    continue  # 跳到下一个键。
                loaded_value = patched_head_state[state_key]  # checkpoint 中的值。
                if (  # 检查是否需要维度填充。
                    torch.is_tensor(loaded_value)  # 必须是张量。
                    and torch.is_tensor(state_value)  # 当前值也是张量。
                    and tuple(loaded_value.shape) != tuple(state_value.shape)  # 形状不匹配。
                    and loaded_value.ndim == 2  # 必须是 2D。
                    and state_value.ndim == 2  # 当前值也是 2D。
                    and int(loaded_value.shape[0]) == int(state_value.shape[0])  # 行数一致。
                    and int(loaded_value.shape[1]) < int(state_value.shape[1])  # 列数更少，需要填充。
                ):
                    padded_value = state_value.detach().clone()  # 克隆当前值作为填充模板，继承当前权重设备。
                    padded_value.zero_()  # 清零。
                    source_value = loaded_value.to(padded_value.device)  # 显式迁移到当前权重设备，避免跨设备赋值。
                    padded_value[:, : int(source_value.shape[1])] = source_value  # 前部对齐填入旧权重。
                    patched_head_state[state_key] = padded_value  # 替换为填充后的权重。
            head.load_state_dict(patched_head_state, strict=False)  # 加载补齐后的头状态（允许旧版多余键，如 v3 的 gate 层）。

        # ---- 4. 加载风险校准状态，兼容旧 checkpoint 无 risk_calibration 键或缺少新增键的情况 ----
        calibration_state = state_dict.get("risk_calibration")  # 取出校准状态。
        if isinstance(calibration_state, Mapping):  # checkpoint 提供了校准状态。
            patched_calibration_state = dict(calibration_state)  # 可变副本。
            current_calibration_state = self.risk_calibration.state_dict()  # 当前校准的完整状态。
            for state_key, state_value in current_calibration_state.items():  # 逐个检查校准状态键。
                if state_key not in patched_calibration_state:  # checkpoint 缺失该键。
                    patched_calibration_state[state_key] = state_value  # 用当前值补齐。
            self.risk_calibration.load_state_dict(patched_calibration_state)  # 加载校准状态。
        # 否则保持初始化默认值。

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """从磁盘加载 checkpoint，合并配置并恢复模型状态。

        参数
        ----------
        checkpoint_path : str | Path
            checkpoint 文件路径。

        返回
        -------
        dict[str, Any]
            完整的 checkpoint payload 字典（缺失 checkpoint 时返回空字典 + warning）。

        异常
        ------
        TypeError
            当 checkpoint 的 model_cfg 不是映射时抛出。
        ValueError
            当 checkpoint 缺少 model_state 时抛出。

        说明
        ----
        阶段 12 全面审计修复 (2026-09-06): 当 checkpoint_path 解析的文件不存在时
        (训练脚本 05/06/07 仅跑 smoke 不存 ckpt)，不再 raise FileNotFoundError，
        改为 warning + 返回空 payload，模型继续以默认 cfg + 随机初始化运行。
        这允许 E9 实验在缺少预训练 ckpt 时仍能跑通（基线降级），
        并打印 warning 提醒用户先跑 05/06/07 真训练并存 ckpt 后重跑。
        """
        path = _resolve_checkpoint_path(checkpoint_path, self.cfg)  # 解析为绝对路径。
        # 阶段 12 修复: checkpoint 缺失容忍 (3 个 model 类共用此逻辑)
        if not path.is_file():
            import sys as _sys
            print(
                f"[load_checkpoint] WARNING: checkpoint file not found at {path}; "
                f"falling back to default cfg + random initialization. "
                f"Run scripts/05_train_lstm.py / 06_train_liquid.py / 07_train_transformer.py "
                f"to produce a real checkpoint first.",
                file=_sys.stderr,
                flush=True,
            )
            # 提前返回前必须构建模块，否则 self.network 停留 None，与
            # docstring「随机初始化继续运行」承诺不符，.eval()/forward 直接 AttributeError。
            self._build_modules()  # 用默认 cfg 构建网络与风险校准模块。
            self._apply_runtime_device()  # 应用运行时设备。
            self._refresh_runtime_resource_meta()  # 刷新资源元数据（与无 checkpoint 路径同口径）。
            return {}  # 返回空 payload, 模型走默认 cfg + 随机初始化
        payload = _load_checkpoint_payload(path)  # 加载 checkpoint 内容。
        model_cfg = payload.get("model_cfg")  # 取出内嵌模型配置（保留 falsy 真值交由后续类型校验）。
        if model_cfg is None:  # 键缺失或显式 None 才回退到空映射。
            model_cfg = {}
        if not isinstance(model_cfg, Mapping):  # model_cfg 必须是映射，捕获 0/False/"" 等 falsy 非映射值。
            raise TypeError("checkpoint model_cfg must be a mapping")

        merged_cfg = _merge_checkpoint_model_cfg(self.cfg, model_cfg)  # 合并基础配置和 checkpoint 配置。
        self.cfg = merged_cfg  # 更新配置。
        feature_order = merged_cfg.get("feature_order")  # 优先用合并后的 feature_order（显式空列表也保留）。
        if feature_order is None:  # 合并配置未提供才回退到 payload 顶层 feature_order。
            feature_order = payload.get("feature_order")
        if feature_order is None:  # 均未提供则视为无特征顺序要求。
            feature_order = []
        self.expected_feature_order = list(feature_order)  # 更新特征顺序。
        self.expected_feature_order_required = bool(self.expected_feature_order)  # 更新校验标志。
        self._build_modules()  # 用合并后的配置重建网络和输出头。

        state_dict = payload.get("model_state")  # 优先取 model_state（保留空字典等 falsy 真值）。
        if state_dict is None:  # model_state 缺失才回退到 legacy state_dict 键。
            state_dict = payload.get("state_dict")
        if state_dict is None:  # 必须有模型状态，缺失属于值合同违例。
            raise ValueError("checkpoint payload must contain model_state")
        self.load_state_dict(state_dict)  # 加载模型状态。

        self.checkpoint_meta = {  # 记录 checkpoint 元数据。
            "checkpoint_path": str(path),  # checkpoint 文件路径。
            "checkpoint_format": payload.get("checkpoint_format"),  # 格式版本。
            "best_epoch": payload.get("best_epoch"),  # 最佳训练轮次。
            "best_loss": payload.get("best_loss"),  # 最佳损失值。
            "train_window_count": payload.get("train_window_count"),  # 训练窗口数。
            "val_window_count": payload.get("val_window_count"),  # 验证窗口数。
        }
        self._apply_runtime_device()  # 应用运行时设备。
        self._refresh_runtime_resource_meta()  # 刷新资源元数据。
        self._resource_meta_dirty = False  # 清除脏标志。
        return dict(payload)  # 返回完整 payload。


# =============================================================================
# _TransformerModel
# =============================================================================
# §10.2 Transformer+EKF：真实现（替换此前 NotImplementedError 占位壳）。
# 与 _LSTMModel 完全相同架构，区别仅在 _build_modules 使用 TransformerNetwork 而非 LSTMNetwork。
# 同一因果输入 + 四头 + bridge + EKF 后端——满足 spec §10.2 表行2 的 strict causal 要求。


@dataclass(slots=True)
class _TransformerModel(_NeuralModelLifecycleMixin, ModelAPI):
    """Transformer-EKF 模型的工厂实例。

    封装 Transformer 网络的创建、checkpoint 加载、推理和状态管理。
    实现 ``ModelAPI`` 接口，提供统一的 ``infer_intermediate`` 方法。

    与 _LSTMModel 的区别：使用 TransformerNetwork backbone（含因果掩码自注意力）。
    所有其他方法（predict_intermediate_tensors / infer_intermediate /
    state_dict / load_state_dict / load_checkpoint）与 _LSTMModel 完全相同，
    复用相同的 normalize/coerce/risk_calibration 辅助函数。
    """

    name: str  # 模型名字。
    cfg: dict[str, Any] = field(default_factory=dict)  # 模型配置字典，默认为空。
    params: float = 0.0  # 模型参数量。
    ram_peak: float = 1.0  # 峰值内存估算（MB）。
    ram_peak_mb: float = 1.0  # 峰值内存估算（MB），与 ram_peak 相同。
    runtime_resource_meta: dict[str, float] = field(init=False, repr=False, default_factory=dict)  # 运行时资源元数据。
    network: Any = field(init=False, repr=False, default=None)  # Transformer 网络实例。
    risk_calibration: Any = field(init=False, repr=False, default=None)  # 风险校准模块。
    expected_feature_order: list[str] = field(init=False, repr=False, default_factory=list)  # 期望的特征顺序。
    checkpoint_meta: dict[str, Any] = field(init=False, repr=False, default_factory=dict)  # checkpoint 元数据。
    expected_feature_order_required: bool = field(init=False, repr=False, default=False)  # 是否要求特征顺序校验。
    runtime_device: str = field(init=False, default=DEVICE_CPU)  # 运行时设备，默认 CPU。

    def __post_init__(self) -> None:
        self._initialize_model(
            network_type="TransformerNetwork",
            log_label="_TransformerModel.__init__ 入口参数",
        )

    def _build_modules(self) -> None:
        """构建 Transformer 网络模块和风险校准模块。"""
        from liquidloc.models.transformer.network import TransformerNetwork  # 延迟导入，避免循环依赖。
        self.network = TransformerNetwork(dict(self.cfg))  # 用配置字典创建 Transformer 网络。
        self.risk_calibration = RiskCalibration()  # 创建风险校准模块。

    def _apply_runtime_device(self) -> None:
        return _NeuralModelLifecycleMixin._apply_runtime_device(self)

    def _refresh_runtime_resource_meta(self) -> None:
        return _NeuralModelLifecycleMixin._refresh_runtime_resource_meta(self)

    def reset(self) -> None:
        """重置模型级状态（Transformer 无 hidden_state，跟 LSTM/Liquid 对齐）。

        §1.4 兼容：Transformer 的注意力状态通过因果掩码在 forward 内自包含，
        跨轨不泄漏。本方法为 no-op。
        """
        return _NeuralModelLifecycleMixin.reset(self)

    def train(self) -> None:
        return _NeuralModelLifecycleMixin.train(self)

    def eval(self) -> None:
        return _NeuralModelLifecycleMixin.eval(self)

    def parameters(self):
        yield from _NeuralModelLifecycleMixin.parameters(self)

    def _normalize_window(self, window_tensor: Any) -> Any:
        """校验特征窗口的特征顺序是否与期望一致。"""
        return _NeuralModelLifecycleMixin._normalize_window(self, window_tensor)

    def predict_intermediate_tensors(self, window_tensor: Any) -> dict[str, torch.Tensor]:
        """前向推理，返回归一化后的中间张量字典（与 _LSTMModel 完全相同逻辑）。"""
        return _NeuralModelLifecycleMixin.predict_intermediate_tensors(self, window_tensor)

    def infer_intermediate(self, window_tensor: Any) -> ModelIntermediate:
        """无梯度推理，返回 ModelIntermediate 数据类（与 _LSTMModel 完全相同逻辑）。"""
        return _NeuralModelLifecycleMixin.infer_intermediate(self, window_tensor)

    def state_dict(self) -> dict[str, Any]:
        """返回模型状态字典（与 _LSTMModel 完全相同结构）。"""
        return _NeuralModelLifecycleMixin.state_dict(self)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """加载模型状态字典，支持旧版 checkpoint 的输出层权重填充（与 _LSTMModel 完全相同逻辑）。"""
        return _NeuralModelLifecycleMixin.load_state_dict(self, state_dict)

    def _prepare_network_state_dict(self, network_state: Mapping[str, Any]) -> Mapping[str, Any]:
        output_layer_weight = network_state.get("output_layer.weight")
        if not torch.is_tensor(output_layer_weight):
            return network_state
        current_weight = self.network.output_layer.weight
        if not (
            output_layer_weight.ndim == 2
            and current_weight.ndim == 2
            and int(output_layer_weight.shape[0]) == int(current_weight.shape[0])
            and int(output_layer_weight.shape[1]) < int(current_weight.shape[1])
        ):
            return network_state
        padded_weight = current_weight.detach().clone()
        padded_weight.zero_()
        source_weight = output_layer_weight.to(current_weight.device)
        padded_weight[:, : int(source_weight.shape[1])] = source_weight
        patched_network_state = dict(network_state)
        patched_network_state["output_layer.weight"] = padded_weight
        return patched_network_state

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """从磁盘加载 checkpoint，合并配置并恢复模型状态（与 _LSTMModel 完全相同逻辑）。"""
        return _NeuralModelLifecycleMixin.load_checkpoint(self, checkpoint_path)


def create_model(name: str, cfg: Mapping[str, Any] | None) -> ModelAPI:
    """模型工厂入口：根据名字和配置创建模型实例。

    参数
    ----------
    name : str
        模型名字，必须在 ``_SUPPORTED`` 集合内
        （``MODEL_NAME_LSTM`` 或 ``MODEL_NAME_LIQUID``）。
    cfg : Mapping[str, Any] | None
        模型配置字典，可以为 None。

    返回
    -------
    ModelAPI
        对应的模型实例（``_LSTMModel`` 或 ``_LiquidModel``）。

    异常
    ------
    ValueError
        当 *name* 不在支持列表中时抛出。
    """
    # §10.2 双向 TF 禁区前瞻守卫：任何 Transformer 模型必须强制 bidirectional=False 后方可进主表。
    # 位置在 _SUPPORTED 检查之前，确保无论模型名是否已知均被拦截。
    if _TRANSFORMER_BIDIRECTIONAL_FORBIDDEN and hasattr(name, "lower") and "transformer" in str(name).lower():
        # MODEL_NAME_TRANSFORMER 本身已在 _SUPPORTED 中，允许通过；
        # 但 cfg.network.bidirectional 若显式为 True 则拒绝（未来真正实现 Transformer 模型时该守卫从被动转为主动）。
        _net_cfg = (cfg or {}).get("network") if isinstance(cfg, Mapping) else None
        _bidir = _net_cfg.get("bidirectional") if isinstance(_net_cfg, Mapping) else None
        if _bidir is True:
            raise ValueError(
                f"§10.2 双向 TF 禁区：Transformer 模型 {name!r} 必须强制 bidirectional=False 后方可进主表；"
                f"cfg.network.bidirectional=True 违反 spec §10.1(c) 双向注意力看未来禁令。"
            )
        # 其他含 transformer 子串但不在 _SUPPORTED 的名字仍走下方 _SUPPORTED 守卫被拒。

    if name not in _SUPPORTED:  # 只允许支持列表里的模型。
        available = ", ".join(sorted(_SUPPORTED))  # 拼出可用模型名。
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "name": name,
        "cfg_keys": list(cfg.keys()) if isinstance(cfg, dict) else None,
        "has_checkpoint_path": "checkpoint_path" in (cfg or {}) if isinstance(cfg, dict) else False,
        "network_keys": list((cfg or {}).get("network", {}).keys()) if isinstance(cfg, dict) and isinstance((cfg or {}).get("network"), dict) else None,
        "feature_order_len": len((cfg or {}).get("feature_order", [])) if isinstance(cfg, dict) else None,
        "window": (cfg or {}).get("window") if isinstance(cfg, dict) else None,
    }, "create_model 入口参数")

    model_cfg = dict(cfg) if cfg is not None else {}  # 复制配置为字典，避免污染调用方对象；显式 None 检查防止 falsy 非 None 值误回退。
    if name == MODEL_NAME_LIQUID:  # Liquid-EKF 模型。
        return _LiquidModel(name=name, cfg=model_cfg)  # 创建 Liquid 模型实例。
    if name == MODEL_NAME_TRANSFORMER:  # Transformer-EKF 模型（§10.2 第 2 行：Transformer+EKF 须为 strict causal Transformer）。
        return _TransformerModel(name=name, cfg=model_cfg)  # 真 Transformer 模型本体（TransformerNetwork 严格因果 + 4 头 + bridge + EKF），与 _LSTMModel 同口径。
    return _LSTMModel(name=name, cfg=model_cfg)  # 默认创建 LSTM 模型实例。
