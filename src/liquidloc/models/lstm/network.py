"""LSTM 网络与结构化窗口适配层。

这个模块把“结构化窗口”变成真正能喂给 LSTM 的张量，也把 LSTM 的输出头
保持在固定协议里。它既负责配置解析，也负责输入标准化、上下文张量构造、
序列批处理和最终前向传播，所以是 `lstm` 子目录里最核心的实现文件。

上游通常来自工厂、训练脚本和推理脚本；下游则会把这里产出的网络输出接到
`inference.py`、`trainer.py`、`pipelines` 和更高层的融合逻辑。这个文件必须
保持输出头顺序、输入字段约定和上下文维度约定稳定，否则整个模型链路都会乱。
"""

from __future__ import annotations  # 允许后面用尚未定义的类型名做注解。

from collections.abc import Mapping  # 用于判断是否是"键值映射"输入。
from collections.abc import Sequence  # 用于描述可迭代序列型参数。
from liquidloc.common.constants import CONTEXT_DIM, CONTEXT_FEATURE_KEYS, MAX_MODEL_DIM, MODEL_INTERMEDIATE_KEYS, MODALITY_UWB, MODALITY_VIO, RISK_PRIOR_LOGIT  # 模态名、风险先验常量与维度 OOM 上界单源真相，避免硬编码漂移；输出头键名单源真相（D9 根因修复）。
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_integer, is_real, is_string_like  # 统一判断数值与布尔类型。
from typing import Any  # 用于承接不确定类型的配置和输入对象。

import torch  # LSTM 和张量运算都依赖 PyTorch。
from torch import nn  # 只需要神经网络层和初始化工具。


_OUTPUT_KEYS = MODEL_INTERMEDIATE_KEYS  # D9：引用单源真相常量，禁止本地重复定义；保留 _OUTPUT_KEYS 别名供本模块内部使用。
_VALID_MODALITIES = frozenset({MODALITY_UWB, MODALITY_VIO})  # LSTM 合同只接受这两种有效模态，引用单源真相避免漂移。
_CONTEXT_FEATURE_KEYS = CONTEXT_FEATURE_KEYS  # §11.3 / §12.3-C2b：从 common.constants 单源引用，禁止本地重复定义（第十三轮消除 LSTM/Liquid 独立常量副本隐患）。
_CONTEXT_DIM = CONTEXT_DIM  # 同上单源引用，避免与 Liquid 端独立推导出不同维度数。


def _coerce_supported_modality(value: Any, *, name: str) -> str:
    """把模态字段统一转成受支持的字符串，并拒绝未知模态。

    与 liquid/network.py 的同名函数保持行为一致：先校验字符串类型，
    再做 strip().lower() 归一化，确保 LSTM 与 Liquid 公平对比时
    模态归一化口径完全相同。
    """
    if not is_string_like(value):
        raise ValueError(f"{name} must be one of {sorted(_VALID_MODALITIES)}")
    modality = str(value).strip().lower()
    if modality not in _VALID_MODALITIES:
        raise ValueError(f"{name} must be one of {sorted(_VALID_MODALITIES)}")
    return modality


def _coerce_positive_int(  # 把配置值转换成正整数。
    value: Any,  # 待转换的原始值。
    *,  # 下面参数必须用关键字传入，避免位置混淆。
    name: str,  # 错误信息里显示的字段名。
    default: int,  # 该字段缺失时采用的默认值。
) -> int:  # 返回校验通过的正整数。
    """把配置值转换成正整数，失败时给出明确字段名。

    校验口径与 Liquid 侧 cell.py `_require_int` / network.py `_build_cell` 对齐：
    非整数报 TypeError，越界报 ValueError，OOM 上界引用 common.constants.MAX_MODEL_DIM，
    保证 LSTM 与 Liquid 对 input_dim/hidden_dim 的拒绝边界一致（D7 公平性）。
    """
    if value is None:  # 配置没写时，直接采用默认值。
        if not is_integer(default):  # 默认值本身也必须是整数（排除 bool）。
            raise TypeError(f"{name} default must be an integer, got {type(default).__name__}: {default!r}")  # 默认类型不对就报错。
        if default < 1:  # 默认值必须为正。
            raise ValueError(f"{name} default must be a positive integer, got {default!r}")  # 默认非法就报错。
        if default > MAX_MODEL_DIM:  # 默认值也要守 OOM 上界，与显式值同口径。
            raise ValueError(f"{name} default must be <= {MAX_MODEL_DIM} to avoid OOM, got {default}")  # 超过上限就拒绝。
        return int(default)  # 统一转成普通 int 返回。
    if not is_integer(value):  # 布尔值和非整数都拒绝。
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}: {value!r}")  # 告诉调用方字段类型错了。
    if value < 1:  # 正整数必须至少为 1。
        raise ValueError(f"{name} must be positive, got {value!r}")  # 小于 1 直接报错。
    if value > MAX_MODEL_DIM:  # 上界保护，与 Liquid 侧 cell.py _require_int / network.py _build_cell 同源（common.constants.MAX_MODEL_DIM），保证 LSTM 与 Liquid 对维度配置的拒绝口径一致（D7 公平性）。
        raise ValueError(f"{name} must be <= {MAX_MODEL_DIM} to avoid OOM, got {value}")  # 超过上限就拒绝。
    return int(value)  # 统一转成普通 int 返回。


def _coerce_probability(  # 把配置值转换成半开区间概率。
    value: Any,  # 待转换的原始值。
    *,  # 下面参数必须用关键字传入，避免位置混淆。
    name: str,  # 错误信息里显示的字段名。
    default: float,  # 该字段缺失时采用的默认值。
) -> float:  # 返回校验通过的概率值。
    """把配置值转换成半开区间 [0, 1) 的概率。"""
    if value is None:  # 没配置就用默认概率。
        validated_default = coerce_finite_scalar(default, name=f"{name}.default")  # 默认值也要走有限性校验，与 Liquid 侧 _require_probability_like 对齐。
        if not 0.0 <= validated_default < 1.0:  # 默认值同样必须在 [0, 1) 内。
            raise ValueError(f"{name}.default must be in [0.0, 1.0).")  # 越界就报错。
        return validated_default  # 返回校验后的默认值。
    if is_bool_like(value) or (not is_real(value) and not torch.is_tensor(value)):  # 布尔和非数字都不允许。
        raise TypeError(f"{name} must be numeric.")  # 提示字段必须是数值。
    value = coerce_finite_scalar(value, name=name)  # 先把输入统一成浮点数。
    if not 0.0 <= value < 1.0:  # 这里明确要求概率不能到 1。
        raise ValueError(f"{name} must be in [0.0, 1.0).")  # 越界就报错。
    return value  # 返回标准化后的概率。


def _coerce_feature_order(  # 把特征顺序转成字符串列表。
    raw_feature_order: Any,  # 原始的 feature_order 配置。
) -> list[str]:  # 返回验证通过的特征名列表。
    """把 feature_order 统一成字符串列表。"""
    if raw_feature_order is None:  # 没给 feature_order 时，返回空列表。
        return []  # 空列表表示没有显式特征顺序。
    # 字符串/字节串/字节数组本身可迭代但会被拆成字符/整数序列，必须拒绝；
    # 映射（dict）会静默用键当字段顺序，掩盖调用方传错意图，也必须拒绝。
    # 用 is_string_like 覆盖 numpy.str_（NumPy 2.x 不再是 str 子类），
    # 与 feature_builder.build_feature_vector 的拒绝口径对齐。
    if (
        is_string_like(raw_feature_order)
        or isinstance(raw_feature_order, (bytes, bytearray))
        or isinstance(raw_feature_order, Mapping)
    ):
        raise TypeError("feature_order must be an iterable of feature names.")  # 明确拒绝字符串/字节/映射输入。

    try:  # 这里尝试把任意可迭代对象转成 list。
        feature_order = list(raw_feature_order)  # 固化成列表，后面就不会被外部迭代器耗尽。
    except TypeError as exc:  # 如果根本不可迭代，就捕获原始异常。
        raise TypeError("feature_order must be an iterable of feature names.") from exc  # 再抛出更明确的错误。

    for index, feature_name in enumerate(feature_order):  # 逐个检查每个特征名。
        if not is_string_like(feature_name) or not feature_name:  # 特征名必须是非空字符串。
            raise ValueError(f"feature_order[{index}] must be a non-empty string.")  # 指出具体坏掉的位置。
    if len(set(feature_order)) != len(feature_order):
        raise ValueError("feature_order must not contain duplicate field names.")
    return feature_order  # 返回验证完成的特征顺序。


def _coerce_window_vector(  # 把一维数据转成窗口向量张量。
    raw_values: Any,  # 原始一维数据。
    *,  # 下面参数必须关键字传入，便于错误定位。
    name: str,  # 错误信息里的字段名。
    feature_dim: int,  # 期望的一维长度。
    dtype: torch.dtype,  # 目标张量类型。
) -> torch.Tensor:  # 返回校验后的向量张量。
    """把一维窗口向量转成指定 dtype 的张量，并检查长度。"""
    if raw_values is None:  # 如果字段缺失，就先造一个空向量。
        vector = torch.empty((0,), dtype=dtype)  # 空向量只是占位，后面会统一校验维度。
    else:  # 字段存在时，尝试转成张量。
        try:
            vector = torch.as_tensor(raw_values, dtype=dtype)  # 不强制拷贝，优先复用原始数据。
        except (TypeError, ValueError) as exc:  # 转换失败时，说明数据形状或类型不对。
            raise TypeError(f"{name} must be a numeric vector aligning with feature_order.") from exc  # 用字段名把问题说清楚。
    if vector.ndim != 1 or vector.numel() != feature_dim:  # 必须是一维，且长度要和特征数一致。
        raise ValueError(f"{name} must be a 1D vector of length {feature_dim}")  # 用字段名和期望长度定位问题。
    if vector.numel() > 0 and not torch.isfinite(vector).all():  # NaN/Inf 不能参与计算。
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")  # 非有限值报错。
    return vector  # 返回标准化后的向量。


def _coerce_window_matrix(  # 把二维数据转成窗口矩阵张量。
    raw_values: Any,  # 原始二维数据。
    *,  # 下面参数必须关键字传入，便于错误定位。
    name: str,  # 错误信息里的字段名。
    feature_dim: int,  # 每行应当具有的列数。
    dtype: torch.dtype,  # 目标张量类型。
) -> torch.Tensor:  # 返回校验后的矩阵张量。
    """把二维窗口矩阵转成张量，并检查行列约束。"""
    if raw_values is None:  # 矩阵字段不能缺失，因为没有矩阵就没有窗口历史。
        raise TypeError(f"{name} must be a 2D structured window")  # 直接说明必须是二维窗口。
    try:  # 尝试把输入转成张量。
        matrix = torch.as_tensor(raw_values, dtype=dtype)  # 这里不做额外变换，只做类型统一。
    except (TypeError, ValueError, RuntimeError) as exc:  # 原始数据不能转成张量时直接报错；RuntimeError 覆盖部分后端（如稀疏/不规则输入）抛出的运行时错误。
        raise TypeError(f"{name} must be a 2D structured window") from exc  # 保留原始异常链。
    if matrix.ndim != 2:  # 窗口历史必须是二维。
        raise TypeError(f"{name} must be a 2D structured window")  # 维度不对就拒绝。
    if matrix.shape[0] < 1:  # 至少要有一行历史数据。
        raise ValueError(f"{name} must be non-empty")  # 空矩阵没有意义。
    if int(matrix.shape[1]) != feature_dim:  # 每一行的列数必须和 feature_order 对齐。
        raise ValueError(f"{name} rows must align with feature_order")  # 列数不对就报错。
    if not torch.isfinite(matrix).all():  # NaN/Inf 不能参与计算。
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")  # 非有限值报错。
    return matrix  # 返回通过检查的窗口矩阵。


def resolve_lstm_network_cfg(  # 把外部配置整理成网络内部配置。
    model_cfg: Mapping[str, Any] | None,  # 原始模型配置映射。
) -> dict[str, Any]:  # 返回标准化后的配置字典。
    """把外部模型配置解析成 LSTM 网络真正需要的内部配置。"""
    if model_cfg is not None and not isinstance(model_cfg, Mapping):  # 非 None 时必须是映射，避免 dict() 把字符串/序列等 truthy 值静默拆成字符项。
        raise TypeError("model_cfg must be a mapping or None.")  # 类型不对就明确报错。
    cfg = dict(model_cfg or {})  # 先把空配置或映射配置都归一成普通字典。
    raw_network_cfg = cfg.get("network") or {}  # network 子配置缺失时，使用空字典兜底。
    if not isinstance(raw_network_cfg, Mapping):  # network 子项必须是映射。
        raise TypeError("model_cfg.network must be a mapping when provided.")  # 类型不对就明确报错。

    feature_order = _coerce_feature_order(cfg.get("feature_order"))  # 先解析特征顺序。
    feature_dim = len(feature_order)  # 特征维度就是顺序长度。
    input_dim = _coerce_positive_int(  # 解析输入维度，空值时用默认。
        raw_network_cfg.get("input_dim"),  # 从 network 子配置里读输入维度。
        name="model_cfg.network.input_dim",  # 这个名字会出现在错误信息里。
        default=max(1, feature_dim),  # 默认值至少为 1，或者跟特征维度对齐。
    )
    hidden_dim = _coerce_positive_int(  # 解析隐藏层宽度。
        raw_network_cfg.get("hidden_dim"),  # 从配置里读取隐藏维度。
        name="model_cfg.network.hidden_dim",  # 错误消息中的字段名。
        default=64,  # 没配置时默认 64。
    )
    num_layers = _coerce_positive_int(  # 解析 LSTM 层数。
        raw_network_cfg.get("num_layers"),  # 从配置里读取层数。
        name="model_cfg.network.num_layers",  # 错误消息中的字段名。
        default=1,  # 默认单层。
    )
    dropout = _coerce_probability(  # 解析 dropout 概率。
        raw_network_cfg.get("dropout"),  # 从配置里读取 dropout。
        name="model_cfg.network.dropout",  # 错误消息中的字段名。
        default=0.0,  # 默认不丢弃。
    )

    output_heads = list(raw_network_cfg.get("output_heads") or _OUTPUT_KEYS)  # 输出头若没配就用固定协议默认值。
    if output_heads != list(_OUTPUT_KEYS):  # 输出头顺序和数量都不能偏。
        raise ValueError(  # 如果不一致，直接提示契约不匹配。
            "model_cfg.network.output_heads must keep the fixed order "
            "['bias', 'risk', 'uwb_scaling', 'vio_scaling']."  # 把必须的顺序写死在错误里。
        )

    return {  # 返回内部使用的标准配置字典。
        "feature_order": feature_order,  # 规范化后的特征顺序。
        "feature_dim": feature_dim,  # 特征总数。
        "input_dim": input_dim,  # LSTM 实际输入维度。
        "hidden_dim": hidden_dim,  # 隐藏状态维度。
        "num_layers": num_layers,  # LSTM 层数。
        "dropout": dropout,  # dropout 概率。
        "output_heads": output_heads,  # 输出头名称和顺序。
    }


def normalize_structured_window(  # 把结构化窗口整理成统一格式。
    window_tensor: Any,  # 原始窗口对象。
    *,  # 后面的参数必须用关键字传入。
    expected_feature_order: list[str] | None = None,  # 可选的特征顺序强校验。
) -> dict[str, Any]:  # 返回标准化窗口字典。
    """校验并整理结构化窗口映射，保证后续输入形状一致。"""
    if not isinstance(window_tensor, Mapping):  # 入口必须是映射，而不是裸张量。
        raise TypeError("window_tensor must be a structured feature window mapping.")  # 不是映射就直接报错。

    feature_order = _coerce_feature_order(window_tensor.get("feature_order"))  # 读取并校验特征顺序。
    if not feature_order:  # 特征顺序不能为空。
        raise ValueError("feature_window.feature_order must be non-empty")  # 空顺序没有办法对齐。
    if expected_feature_order is not None and feature_order != list(expected_feature_order):  # 显式 None 才跳过，空列表也参与强校验。
        raise ValueError(  # 顺序不同会直接破坏模型输入语义。
            "window_tensor.feature_order must match the model feature_order; "
            f"expected {expected_feature_order}, got {feature_order}."  # 把期望和实际都写进错误里。
        )

    feature_dim = len(feature_order)  # 这里的特征维度直接来自顺序长度。
    current_modality = _coerce_supported_modality(
        window_tensor.get("current_modality"),
        name="feature_window.current_modality",
    )  # 当前模态必须是受支持的协议模态。

    feature_values = _coerce_window_vector(  # 读取当前步的特征值向量。
        window_tensor.get("feature_values"),  # 当前步值必须和特征顺序对齐。
        name="feature_window.feature_values",  # 给错误信息提供字段名。
        feature_dim=feature_dim,  # 长度必须和特征数一致。
        dtype=torch.float32,  # 统一成 float32 便于后续网络计算。
    )
    missing_mask = _coerce_window_vector(  # 读取当前步缺失掩码。
        window_tensor.get("missing_mask"),  # 掩码向量也必须和特征顺序对齐。
        name="feature_window.missing_mask",  # 错误字段名。
        feature_dim=feature_dim,  # 掩码长度必须一致。
        dtype=torch.float32,  # 统一成 float32，和特征值使用同一数值类型。
    )

    dt = window_tensor.get("dt")  # 读取当前窗口对应的时间间隔。
    if dt is None:  # dt 不能省略。
        raise ValueError("feature_window.dt must be provided explicitly")  # 没有 dt 就无法解释时间关系。

    feature_window = _coerce_window_matrix(  # 读取历史特征窗口矩阵。
        window_tensor.get("feature_window"),  # 历史特征必须是完整二维矩阵。
        name="feature_window.feature_window",  # 错误字段名。
        feature_dim=feature_dim,  # 每行列数必须对齐。
        dtype=torch.float32,  # 统一数值类型。
    )
    missing_mask_window = _coerce_window_matrix(  # 读取历史缺失掩码矩阵。
        window_tensor.get("missing_mask_window"),  # 掩码历史也必须是二维矩阵。
        name="feature_window.missing_mask_window",  # 错误字段名。
        feature_dim=feature_dim,  # 每行列数必须对齐。
        dtype=torch.float32,  # 统一数值类型。
    )
    # 缺失掩码只允许 0.0/1.0，与 Liquid 侧 _coerce_mask_value 的 0/1 校验对齐。
    if not ((missing_mask == 0.0) | (missing_mask == 1.0)).all():
        raise ValueError("feature_window.missing_mask must contain only 0/1 values")
    if not ((missing_mask_window == 0.0) | (missing_mask_window == 1.0)).all():
        raise ValueError("feature_window.missing_mask_window must contain only 0/1 values")
    if feature_window.shape != missing_mask_window.shape:  # 两个历史矩阵的形状必须完全一样。
        raise ValueError("feature_window.feature_window and missing_mask_window must have the same row count")  # 行数不一致就不能配对。
    if not torch.equal(feature_window[-1], feature_values):  # 最后一行必须严格等于当前步特征值，与 Liquid 侧精确比对对齐。
        raise ValueError("feature_window current-step feature_values must match the last window row")  # 当前步特征值镜像不一致就报错。
    if not torch.equal(missing_mask_window[-1], missing_mask):  # 最后一行掩码也必须严格一致，不再做 int64 截断。
        raise ValueError("feature_window current-step missing_mask must match the last window row")  # 掩码镜像不一致也报错。

    return {  # 返回统一后的结构化窗口对象。
        "current_modality": current_modality,  # 当前模态标签。
        "feature_order": feature_order,  # 已校验的特征顺序。
        "feature_values": feature_values,  # 当前步特征值。
        "missing_mask": missing_mask,  # 当前步缺失掩码。
        "dt": coerce_finite_scalar(dt, name="feature_window.dt", min_value=0.0),  # 时间间隔统一转成非负有限浮点数，与 Liquid 侧 _resolve_step_dts 的 fallback_dt 非负口径对齐（D7 公平性）。
        "feature_window": feature_window,  # 历史特征窗口矩阵。
        "missing_mask_window": missing_mask_window,  # 历史缺失掩码矩阵。
    }


def build_lstm_sequence_tensor(  # 把结构化窗口拼成 LSTM 序列。
    window_tensor: Any,  # 原始结构化窗口。
    *,  # 下面参数必须关键字传入。
    expected_feature_order: list[str] | None = None,  # 可选的特征顺序强校验。
) -> tuple[torch.Tensor, dict[str, Any]]:  # 返回序列张量和标准化窗口。
    """把结构化窗口拼成 LSTM 输入序列，并返回标准化后的窗口。"""
    # Fast path for already-normalized windows (e.g., from forward path or
    # inference path): feature_window and missing_mask_window are already
    # tensors, so skip re-validation to avoid duplicate normalize_structured_window
    # calls. This mirrors Liquid's normalize_window_tensor fast path for cached
    # samples, ensuring LSTM and Liquid have comparable forward-path overhead.
    if (
        isinstance(window_tensor, Mapping)
        and isinstance(window_tensor.get("feature_window"), torch.Tensor)
        and isinstance(window_tensor.get("missing_mask_window"), torch.Tensor)
        and "current_modality" in window_tensor
        and "feature_order" in window_tensor
        and "feature_values" in window_tensor
        and "missing_mask" in window_tensor
        and "dt" in window_tensor
    ):
        normalized = window_tensor  # 已标准化的窗口直接复用，避免重复校验。
    else:
        normalized = normalize_structured_window(  # 先把输入整理成标准窗口。
            window_tensor,  # 原始窗口作为输入。
            expected_feature_order=expected_feature_order,  # 如果上游指定了顺序，这里继续强校验。
        )
    feature_window = normalized["feature_window"]  # 取出历史特征窗口。
    missing_mask_window = normalized["missing_mask_window"]  # 取出历史掩码窗口。
    masked_features = feature_window * (1.0 - missing_mask_window)  # 缺失位置用 0 屏蔽，保留可见位置。
    sequence_tensor = torch.cat([masked_features, missing_mask_window], dim=-1).unsqueeze(0)  # 特征和掩码拼接成单批序列。
    return sequence_tensor, normalized  # 同时返回序列张量和标准化窗口，方便上层复用。


def build_lstm_context_tensor_from_normalized_window(  # 构造单样本上下文向量。
    normalized_window: Mapping[str, Any],  # 标准化后的窗口映射。
    *,  # 下面参数必须关键字传入。
    dtype: torch.dtype,  # 输出张量类型。
    device: torch.device,  # 输出张量所在设备。
) -> torch.Tensor:  # 返回上下文向量张量。
    """从标准化窗口里提取上下文向量，供输出层拼接使用。"""
    current_modality = _coerce_supported_modality(
        normalized_window.get("current_modality"),
        name="feature_window.current_modality",
    )  # 上下文构造必须消费已校验的协议模态。
    modality_vector = {  # 把模态映射成两维 one-hot 风格向量。
        MODALITY_UWB: [1.0, 0.0],  # UWB 模态对应第一个位置。
        MODALITY_VIO: [0.0, 1.0],  # VIO 模态对应第二个位置。
    }[current_modality]  # 未知模态已在上游被拒绝，这里不再静默兜底。

    feature_order = list(normalized_window.get("feature_order") or [])  # 读取特征顺序，便于按名字定位。
    feature_index_by_name = {str(name): index for index, name in enumerate(feature_order)}  # 建立特征名到位置的索引表。
    feature_values = normalized_window.get("feature_values")  # 当前步特征值向量。
    missing_mask = normalized_window.get("missing_mask")  # 当前步缺失掩码向量。
    context_values: list[float] = list(modality_vector)  # 上下文先从模态向量开始。

    for feature_name in _CONTEXT_FEATURE_KEYS:  # 逐个上下文特征名提取对应值。
        feature_index = feature_index_by_name.get(feature_name)  # 找到这个特征在窗口里的位置。
        observed = False  # 默认认为没观测到。
        scalar_value = 0.0  # 没观测到时，数值先用 0。
        if feature_index is not None and feature_values is not None and missing_mask is not None:  # 三个条件都满足才尝试取值。
            missing_value = coerce_finite_scalar(missing_mask[feature_index], name=f"missing_mask.{feature_name}")  # 先读掩码，判断这个位置是否可见。
            observed = missing_value < 0.5  # 掩码小于 0.5 视为观测到。
            if observed:  # 只有确实观测到才读取数值。
                scalar_value = coerce_finite_scalar(feature_values[feature_index], name=f"feature_values.{feature_name}")  # 把候选值转成普通浮点数。
                # coerce_finite_scalar 已保证返回有限浮点数（非有限时直接抛错），无需重复 isfinite 校验。
        context_values.append(scalar_value)  # 先放入该特征的数值。
        context_values.append(1.0 if observed else 0.0)  # 再放入是否观测到的标志位。

    return torch.tensor(context_values, dtype=dtype, device=device)  # 把 Python 列表变成张量并放到指定设备上。


def build_lstm_context_batch_from_sequence(  # 构造序列批次对应的上下文批次。
    sequence_tensor: Any,  # 原始序列输入。
    *,  # 下面参数必须关键字传入。
    feature_order: Sequence[str],  # 当前模型的特征顺序。
    modalities: Sequence[str] | None,  # 批次对应的模态标签。
    dtype: torch.dtype,  # 输出张量类型。
    device: torch.device,  # 输出张量所在设备。
) -> torch.Tensor:  # 返回上下文批次张量。
    """从一批序列张量里构造对应的上下文批次。"""
    raw_sequence = torch.as_tensor(sequence_tensor, dtype=dtype, device=device)  # 先统一转成张量。
    if raw_sequence.ndim == 2:  # 单条序列时，补一个 batch 维。
        raw_sequence = raw_sequence.unsqueeze(0)  # 变成批次形式，便于统一处理。
    if raw_sequence.ndim != 3:  # 这里最终必须是三维批次序列。
        raise ValueError("sequence_tensor must be a 2D or 3D sequence tensor")  # 维度不对就报错。

    batch_size = int(raw_sequence.shape[0])  # 批次大小等于第一维。
    feature_dim = len(list(feature_order))  # 特征维度来自 feature_order 长度。
    last_dim = int(raw_sequence.shape[-1])  # 最后一维决定是特征还是特征+掩码。
    if last_dim == feature_dim:  # 如果只有特征，没有掩码。
        feature_values = raw_sequence[:, -1, :]  # 最后一帧就是当前步特征值。
        missing_mask = torch.zeros_like(feature_values)  # 没有掩码时，用全零表示都已观测。
    elif last_dim == feature_dim * 2:  # 如果已经拼过特征和掩码。
        feature_values = raw_sequence[:, -1, :feature_dim]  # 前半部分是特征值。
        missing_mask = raw_sequence[:, -1, feature_dim:]  # 后半部分是缺失掩码。
    else:  # 既不是纯特征，也不是特征+掩码。
        if feature_dim > 0:
            raise ValueError(
                "sequence_tensor last dimension must align with either feature_dim "
                "or feature_dim * 2 (features + missing mask)."
            )
        feature_values = None  # 这里先置空，后面会给上下文补默认值。
        missing_mask = None  # 同样把掩码置空。

    if modalities is None:  # 纯序列批次里没有结构化窗口，无法可靠推断模态。
        raise ValueError(
            "modalities must be provided for sequence batches because the LSTM context contract "
            "requires per-sample current_modality."
        )  # 在入口层显式拒绝，避免更深层报出误导性的 current_modality 错误。
    if len(modalities) != batch_size:
        raise ValueError("modalities must match the batch size")
    resolved_modalities = [
        _coerce_supported_modality(modality, name="modalities")
        for modality in modalities
    ]  # 序列批次旁路也必须遵守同一套模态合同。
    if len(resolved_modalities) != batch_size:  # redundant: length checked before coercion  # 模态数量必须和 batch 一致。
        raise ValueError("modalities must match the batch size")  # 数量不对就报错。

    context_rows: list[torch.Tensor] = []  # 这里收集每个样本的上下文向量。
    for row_index in range(batch_size):  # 逐个样本构造上下文。
        normalized_window = {  # 先为当前样本拼一个最小标准窗口。
            "current_modality": resolved_modalities[row_index],  # 当前样本的模态标签。
            "feature_order": list(feature_order),  # 复制特征顺序，避免外部可变对象影响。
            "feature_values": (  # 当前步特征值。
                feature_values[row_index]  # 若已有特征值，就直接取当前样本这一行。
                if feature_values is not None  # 判断是否真的有可用特征值。
                else torch.zeros((feature_dim,), dtype=dtype, device=device)  # 否则用全零向量兜底。
            ),
            "missing_mask": (  # 当前步缺失掩码。
                missing_mask[row_index]  # 若已有掩码，就直接取当前样本这一行。
                if missing_mask is not None  # 判断是否真的有可用掩码。
                else torch.ones((feature_dim,), dtype=dtype, device=device)  # 否则用全一表示全部缺失。
            ),
        }
        context_rows.append(  # 把当前样本的上下文向量加入批次列表。
            build_lstm_context_tensor_from_normalized_window(  # 复用单样本上下文构造函数。
                normalized_window,  # 把刚拼好的标准窗口传进去。
                dtype=dtype,  # 保持数据类型一致。
                device=device,  # 保持设备一致。
            )
        )
    return torch.stack(context_rows, dim=0)  # 把每个样本的上下文堆成 batch。


class LSTMNetwork(nn.Module):  # 结构化窗口对应的 LSTM 前端模型。
    """保留四个固定中间输出头的轻量 LSTM 前端网络。"""

    def __init__(self, model_cfg: Mapping[str, Any] | None = None):
        super().__init__()  # 初始化 nn.Module 基类状态。
        resolved_cfg = resolve_lstm_network_cfg(model_cfg)  # 先把外部配置归一化。
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "feature_dim": resolved_cfg.get("feature_dim") if isinstance(resolved_cfg, dict) else None,
            "input_dim": resolved_cfg.get("input_dim") if isinstance(resolved_cfg, dict) else None,
            "hidden_dim": resolved_cfg.get("hidden_dim") if isinstance(resolved_cfg, dict) else None,
            "num_layers": resolved_cfg.get("num_layers") if isinstance(resolved_cfg, dict) else None,
            "dropout": resolved_cfg.get("dropout") if isinstance(resolved_cfg, dict) else None,
            "output_heads": resolved_cfg.get("output_heads") if isinstance(resolved_cfg, dict) else None,
        }, "LSTMNetwork.__init__ 入口参数")
        self.feature_order = list(resolved_cfg["feature_order"])  # 保存特征顺序，供后续输入校验。
        self.feature_dim = int(resolved_cfg["feature_dim"])  # 保存特征维度。
        self.input_dim = int(resolved_cfg["input_dim"])  # 保存 LSTM 输入维度。
        self.hidden_dim = int(resolved_cfg["hidden_dim"])  # 保存隐藏层维度。
        self.num_layers = int(resolved_cfg["num_layers"])  # 保存 LSTM 层数。
        self.dropout = float(resolved_cfg["dropout"])  # 保存 dropout 概率。
        self.output_heads = tuple(resolved_cfg["output_heads"])  # 保存输出头顺序，保证协议稳定。
        self.context_dim = _CONTEXT_DIM  # 上下文维度固定，和特征抽取约定绑定。
        self.combined_input_dim = self.feature_dim * 2 if self.feature_dim else self.input_dim  # 兼容特征+掩码的联合输入维度。

        self.input_projection = None  # 默认不需要投影层。
        if self.combined_input_dim != self.input_dim:  # 如果联合输入维度和 LSTM 期望输入不一致，就加一个投影层。
            self.input_projection = nn.Linear(self.combined_input_dim, self.input_dim)  # 用线性层把维度对齐。

        self.lstm = nn.LSTM(  # 真正的序列编码器。
            input_size=self.input_dim,  # 输入特征维度。
            hidden_size=self.hidden_dim,  # 隐藏状态维度。
            num_layers=self.num_layers,  # 堆叠层数。
            dropout=self.dropout if self.num_layers > 1 else 0.0,  # 单层时不启用 dropout。
            batch_first=True,  # 输入张量采用 batch-first 布局。
            bidirectional=False,  # §10.1/§10.2 强制单向因果：禁止双向注意力看未来。
        )
        self.output_layer = nn.Linear(self.hidden_dim + self.context_dim, len(self.output_heads))  # 把隐藏态和上下文拼接后映射到四个头。
        self._reset_output_layer_contract_biases()  # 初始化输出层偏置，保证默认先验稳定。

    # 这个内部方法负责把输出层偏置初始化成符合协议先验的状态。
    def _reset_output_layer_contract_biases(self) -> None:
        """初始化输出层权重和偏置，让输出契约从一开始就稳定。"""
        nn.init.xavier_uniform_(self.output_layer.weight)  # 权重用 Xavier 初始化，保持数值尺度合理。
        if self.output_layer.bias is None:  # 如果这个层没有 bias，就不用再做任何事。
            return  # 直接结束。
        with torch.no_grad():  # 初始化阶段不需要梯度记录。
            self.output_layer.bias.zero_()  # 先把所有偏置清零。
            # 从冻结协议取 risk 头位置，避免硬编码索引漂移；显式校验该位置确实是 risk 头。
            if "risk" in self.output_heads:
                risk_head_index = self.output_heads.index("risk")
                self.output_layer.bias[risk_head_index] = RISK_PRIOR_LOGIT  # 给风险头一个先验偏置。

    # 重置所有可学习参数到初始状态，与 LiquidNetwork.reset_parameters 同名同口径（D7 公平性）。
    def reset_parameters(self) -> None:
        """重置所有可学习参数到初始状态。

        重置范围：
        - input_projection（若存在）：委托给 nn.Linear.reset_parameters。
        - lstm：委托给 nn.LSTM.reset_parameters。
        - output_layer：通过 _reset_output_layer_contract_biases 重置，保留协议偏置契约。

        与 LiquidNetwork.reset_parameters 保持同名同口径（D7 公平性）。

        参数:
        这个方法不接收外部参数，只操作当前实例内部已经创建好的层。

        返回值:
        无返回值，只修改模块内部状态。
        """
        if self.input_projection is not None:  # 输入投影层存在时重置为默认初始化。
            self.input_projection.reset_parameters()
        self.lstm.reset_parameters()  # LSTM 重置为默认初始化。
        self._reset_output_layer_contract_biases()  # 输出层重置，保留协议偏置契约。

    # 这个内部方法负责把任意输入统一成三维序列批次。
    def _coerce_sequence_batch(self, window_tensor: Any) -> torch.Tensor:
        """把输入统一成三维序列批次张量。"""
        if isinstance(window_tensor, Mapping):  # 如果输入是结构化窗口。
            sequence_tensor, _ = build_lstm_sequence_tensor(  # 先把窗口转成序列张量。
                window_tensor,  # 原始结构化窗口。
                expected_feature_order=self.feature_order or None,  # 如果模型已知顺序，就强校验。
            )
            # 校验结构化窗口产出的序列最后一维与模型联合输入维度对齐，
            # 防止 feature_order 缺失时窗口维度与模型 input_dim 不匹配导致下游 LSTM 报错。
            if int(sequence_tensor.shape[-1]) != self.combined_input_dim:
                raise ValueError(
                    "window_tensor last dimension must align with the model combined_input_dim."
                )
            return sequence_tensor  # 直接返回序列张量。

        try:  # 裸张量转换可能因不规则形状或对象数组失败，统一捕获后给出契约级错误信息。
            sequence_tensor = torch.as_tensor(window_tensor, dtype=torch.float32)  # 把裸输入转成浮点张量。
        except (TypeError, ValueError, RuntimeError) as exc:  # 与 _coerce_window_matrix 的异常口径对齐。
            raise TypeError(
                "window_tensor must be a 2D/3D tensor or a structured feature window mapping."
            ) from exc
        if sequence_tensor.ndim == 2:  # 单条序列补 batch 维。
            sequence_tensor = sequence_tensor.unsqueeze(0)  # 变成三维批次张量。
        elif sequence_tensor.ndim != 3:  # 不是 2D 或 3D 就不接受。
            raise ValueError("window_tensor must be a 2D/3D tensor or a structured feature window mapping.")  # 说明输入类型不对。
        if any(int(dim) <= 0 for dim in sequence_tensor.shape):  # 任何一维非正都不允许。
            raise ValueError("window_tensor must be non-empty in every dimension.")  # 空张量没有意义。

        last_dim = int(sequence_tensor.shape[-1])  # 读取最后一维长度。
        if self.feature_dim and last_dim == self.feature_dim:  # 如果只有特征没有掩码。
            zero_mask = torch.zeros_like(sequence_tensor)  # 造一个全零掩码。
            sequence_tensor = torch.cat([sequence_tensor, zero_mask], dim=-1)  # 把特征和零掩码拼起来。
        elif last_dim != self.combined_input_dim:  # 既不是特征维（需补掩码），也不是联合输入维。
            raise ValueError(  # 说明最后一维不匹配模型预期。
                "window_tensor last dimension must align with either the model feature_dim or the "
                "combined_input_dim."
            )
        if not torch.isfinite(sequence_tensor).all():  # NaN/Inf 不能参与计算，与 _coerce_window_matrix 的有限性校验对齐。
            raise ValueError("window_tensor contains non-finite values (NaN or Inf)")  # 非有限值报错。
        return sequence_tensor  # 返回标准化后的序列批次。

    # 这个内部方法负责把输入搬到模型所在设备，并在需要时做投影。
    def _prepare_inputs(self, window_tensor: Any) -> torch.Tensor:
        """把输入搬到模型所在设备，并在需要时做投影。"""
        sequence_tensor = self._coerce_sequence_batch(window_tensor)  # 先统一输入形状。
        device = next(self.parameters()).device  # 读取模型当前所在设备。
        sequence_tensor = sequence_tensor.to(device=device, dtype=next(self.parameters()).dtype)  # 搬到同一设备并对齐 dtype，防止 Mapping 路径窗口 dtype 与模型权重不一致。
        if self.input_projection is not None:  # 如果需要先做维度投影。
            sequence_tensor = self.input_projection(sequence_tensor)  # 先把联合维压到 input_dim。
        return sequence_tensor  # 返回可直接喂给 LSTM 的输入。

    # 这个前向分支专门处理已经是序列批次的输入。
    def forward_sequence_batch(
        self,
        sequence_tensor: Any,
        *,
        modalities: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """对已经是序列批次的输入执行前向传播。"""
        # 先把输入统一成序列批次，后面才能按 batch 处理。
        raw_sequence = self._coerce_sequence_batch(sequence_tensor)  # 先把输入统一成序列批次。
        # 读取模型当前所在设备，保证输入输出设备一致。
        device = next(self.parameters()).device  # 读取模型所在设备。
        # 把数据搬到同一设备，避免张量跨设备计算报错。
        raw_sequence = raw_sequence.to(device=device, dtype=next(self.parameters()).dtype)  # 搬到同一设备并对齐 dtype，防止 Mapping 路径窗口 dtype 与模型权重不一致。
        # 这里根据序列和模态构造附加上下文，给输出层更多条件信息。
        context_batch = build_lstm_context_batch_from_sequence(  # 构造与序列批次对应的上下文批次。
            raw_sequence,  # 原始序列张量。
            feature_order=self.feature_order,  # 特征顺序来自模型配置。
            modalities=modalities,  # 上游若提供模态，这里就带上。
            dtype=raw_sequence.dtype,  # 保持 dtype 一致。
            device=device,  # 保持设备一致。
        )
        # 默认先直接用原始序列，只有维度不匹配时才做投影。
        sequence_batch = raw_sequence  # 默认直接用原始序列。
        if self.input_projection is not None:  # 如果需要先降维或对齐维度。
            sequence_batch = self.input_projection(sequence_batch)  # 先做输入投影。
        # LSTM 负责把整段序列编码成隐藏表示。
        encoded_sequence, _ = self.lstm(sequence_batch)  # LSTM 编码整个序列。
        # 只取最后一个时间步，代表整段序列的主摘要。
        last_hidden = encoded_sequence[:, -1, :]  # 只取最后一个时间步的隐藏态。
        # 把隐藏态和上下文拼起来，再送进最终输出层。
        network_output = self.output_layer(torch.cat([last_hidden, context_batch], dim=-1))  # 拼接上下文后输出四头。
        if network_output.shape[0] == 1:  # 单样本时压掉 batch 维，方便外部消费。
            return network_output.squeeze(0)  # 返回一维四头向量。
        return network_output  # 多样本时保留 batch 维。

    # 这个前向入口只接受结构化窗口映射，裸张量请走 forward_sequence_batch。
    def forward(self, window_tensor: Any) -> torch.Tensor:
        """对结构化窗口映射执行一次完整前向传播。"""
        if not isinstance(window_tensor, Mapping):  # 如果不是结构化窗口，就走批次序列路径。
            raise TypeError(
                "LSTMNetwork.forward requires a structured feature window mapping (Mapping only); "
                "use forward_sequence_batch(sequence_tensor, modalities=...) for raw sequence batches."
            )  # 裸张量入口缺少模态合同，必须显式走批次序列接口。

        # 结构化窗口需要先统一字段和顺序，再进入真正的网络计算。
        normalized_window = normalize_structured_window(  # 先把映射窗口标准化。
            window_tensor,  # 原始结构化窗口。
            expected_feature_order=self.feature_order or None,  # 若模型已有特征顺序，就强制对齐。
        )
        # 标准化后再做设备搬运和输入投影。
        sequence_tensor = self._prepare_inputs(normalized_window)  # 准备好真正喂给 LSTM 的输入。
        # 单样本上下文向量要补成 batch 维，和 LSTM 输出对齐。
        context_batch = build_lstm_context_tensor_from_normalized_window(  # 单样本上下文向量。
            normalized_window,  # 标准化后的窗口。
            dtype=sequence_tensor.dtype,  # 跟输入保持一致的 dtype。
            device=sequence_tensor.device,  # 跟输入保持一致的 device。
        ).unsqueeze(0)  # 补 batch 维，和 LSTM 输出对齐。
        # LSTM 主干先编码序列，再取最后时刻的隐藏态。
        encoded_sequence, _ = self.lstm(sequence_tensor)  # LSTM 正式编码。
        last_hidden = encoded_sequence[:, -1, :]  # 取最后时刻的隐藏态。
        # 把隐藏态和上下文拼起来，得到最终四头输出。
        network_output = self.output_layer(torch.cat([last_hidden, context_batch], dim=-1))  # 拼接上下文后映射为四头输出。
        return network_output.squeeze(0)  # 单样本窗口始终为 batch=1，去掉 batch 维返回一维四头向量。
