"""
文件：src/liquidloc/models/liquid/network.py

【文件职责】定义 Liquid network，串联 `LiquidCell` 与输出头前的共享主干。
【本文件不负责】不做训练循环，不直接输出最终轨迹，不把四个任务头硬塞进 cell。
【上游依赖】`models/liquid/cell.py`、`common/types.py`。
【下游调用者】`models/liquid/inference.py`、`models/liquid/trainer.py`、`tests/models/test_liquid_network.py`。
【输入对象定义】
- `window_tensor`
- `network_cfg`
- `initial_state`

【输出对象定义】
- `network_hidden` 或 `shared_features`

【核心变量定义】
- `window_tensor`
- `network_cfg`
- `hidden_state`
- `shared_features`

【推荐编写顺序】1. 先写 `__init__`。2. 再写 `forward`。3. 最后写 `reset_state`（如需要）。
【建议先写的函数 / 类】
- `LiquidNetwork`
  签名：`class LiquidNetwork`
  作用：维护 cell 序列展开与共享主干特征输出。
  输入：`network_cfg`、`window_tensor`
  输出：`shared_features`
  关键局部变量：
    - `window_tensor`
    - `hidden_state`
    - `shared_features`
  伪代码：
    1) `__init__` 中创建 `LiquidCell` / 主干层。
    2) `forward` 逐步消费 `window_tensor`。
    3) 输出共享特征 `shared_features`。
【最容易让 Codex 理解错的地方】
- 不要在 network 层直接接四头输出。
【最小手工测试步骤】1. 构造最小 `window_tensor`。2. 检查 `forward` 返回 `shared_features`。
【完成标准】cell/network/head 三层职责分离。
【实现要求】
- 当前阶段保持空白实现，只保留细化编写说明。
- 真正实现时先写签名和 docstring，再写前置检查，再写主体逻辑，最后补测试。
- 任何字段名和变量名优先服从 protocol 和 configs，不能临时发明。
"""

from __future__ import annotations  # 允许在类型注解里引用后面定义的类型。

import math  # 用于有限性检查和时间差计算。
from collections.abc import Mapping, Sequence  # 用于判断映射和序列。
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_integer, is_numeric, is_real, is_string_like  # 统一判断标量数值与布尔类型。
from typing import Any  # 用于接收不确定类型的输入和值。

import torch  # 这里负责张量计算。
from torch import nn  # 这里负责神经网络层定义。

from liquidloc.common.constants import (  # 单源真相常量，避免字面量漂移（D9 配置表面漂移根因修复）。
    ASYNC_GAP_FULL_SCALE_S,
    LIQUID_READOUT_CONTEXT_KEYS as _READOUT_CONTEXT_KEYS,  # 读出上下文键名单源真相，别名保留以避免改下游引用。
    MAX_MODEL_DIM,
    MODEL_INTERMEDIATE_KEYS,
    MODALITY_UWB,
    MODALITY_VIO,
    SHARED_FEATURES_KEY,
    VIO_HIGH_REPROJ_ERR_THRESHOLD,
    VIO_REPROJ_ERR_FULL_SCALE,
    VIO_TRACKED_FEATURES_FLOOR,
)
from liquidloc.models.liquid.cell import LiquidCell  # 这个网络复用 LiquidCell。
from liquidloc.common.weight_init import _fill_linear  # 等间距线性层初始化 helper 的权威定义来源（common 层），避免与 cell.py 重复定义漂移。

_VALID_MODALITIES = frozenset({MODALITY_UWB, MODALITY_VIO})  # Liquid 合同只接受这两种有效模态，引用单源真相避免漂移。


def _coerce_supported_modality(modality: Any, *, name: str) -> str:
    """把模态字段规范化为网络支持的合同值。"""
    if not is_string_like(modality):
        raise ValueError(f"{name} must be one of {sorted(_VALID_MODALITIES)}")
    modality_name = str(modality).strip().lower()
    if modality_name not in _VALID_MODALITIES:
        raise ValueError(f"{name} must be one of {sorted(_VALID_MODALITIES)}")
    return modality_name


def _coerce_numeric_vector(values: Any, *, name: str, cast) -> list[float] | list[int]:
    """把可迭代输入统一转成指定类型的数值列表。

    参数:
    `values` 是待转换的可迭代输入。
    `name` 是错误信息里显示的字段名。
    `cast` 是类型转换函数（float 或 int）。

    返回值:
    返回转换后的数值列表。

    失败条件:
    输入不可迭代或元素不是有限数值时抛出异常。
    """
    if isinstance(values, (str, bytes, Mapping)):  # 字符串/字节/映射不是合法数值向量，提前拒绝。
        raise TypeError(f"{name} must be an iterable of numeric values, not {type(values).__name__}")
    try:
        raw_list = list(values)  # 先把输入固化成列表。
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of numeric values") from exc  # 不可迭代就报错。
    normalized = []  # 存转换后的数值。
    for index, value in enumerate(raw_list):  # 逐个转换并校验。
        normalized.append(cast(coerce_finite_scalar(value, name=f"{name}[{index}]")))  # 每个元素都统一转成有限数再 cast。
    return normalized  # 返回标准化后的数值列表。


def _coerce_mask_value(value: Any, *, name: str) -> int:
    """把单个缺失掩码值规范成 0/1。"""
    if isinstance(value, torch.Tensor) and value.dtype == torch.bool:  # torch.bool 单元素张量与 LSTM 的 bool→0/1 语义对齐。
        if value.numel() != 1:
            raise TypeError(f"{name} must be a scalar bool tensor, got shape {tuple(value.shape)}")
        return int(bool(value.item()))
    if is_bool_like(value):
        return int(bool(value))
    scalar = coerce_finite_scalar(value, name=name)
    if scalar not in (0.0, 1.0):
        raise ValueError(f"{name} must be 0/1 or bool")
    return int(scalar)


def _coerce_mask_vector(values: Any, *, name: str) -> list[int]:
    """把一维缺失掩码统一成 0/1 整型列表。"""
    if is_string_like(values) or isinstance(values, (bytes, bytearray, Mapping)):  # 字符串(含 numpy.str_)/字节/字节数组/映射不是合法掩码向量，提前拒绝，防止 list() 把它们拆成字符或整数静默穿透。
        raise TypeError(f"{name} must be an iterable of 0/1 values, not {type(values).__name__}")
    try:
        raw_list = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of 0/1 values, got {type(values).__name__}") from exc
    return [_coerce_mask_value(value, name=f"{name}[{index}]") for index, value in enumerate(raw_list)]


def _coerce_float_rows(rows: Any, *, name: str, feature_dim: int) -> list[list[float]]:  # 把二维结构化窗口统一成浮点二维列表。
    """把二维结构化窗口统一成浮点二维列表。

    参数:
    `rows` 是待转换的二维结构。
    `name` 是报错时显示的字段名。
    `feature_dim` 是每行应该具有的特征宽度。

    返回值:
    返回一个二维浮点列表。

    失败条件:
    输入不是二维结构，或者每行宽度不匹配时会报错。
    """
    if is_string_like(rows) or isinstance(rows, (bytes, bytearray, Mapping)):  # 字符串(含 numpy.str_)/字节/字节数组/映射不是合法二维结构，提前拒绝，防止 list() 把它们拆成字符或整数静默穿透。
        raise TypeError(f"{name} must be a 2D structured window, not {type(rows).__name__}")
    try:
        row_list = list(rows)  # 先把外层转成列表，便于检查行数。
    except TypeError as exc:
        raise TypeError(f"{name} must be a 2D structured window") from exc  # 外层不可迭代时直接报错。
    if not row_list:  # 外层不能是空的。
        raise ValueError(f"{name} must be non-empty")

    normalized_rows: list[list[float]] = []  # 存放标准化后的每一行。
    for row_index, row in enumerate(row_list):  # 逐行标准化，保留行号用于错误定位。
        if isinstance(row, (str, bytes, Mapping)):  # 字符串/字节/映射不是合法数值行，提前拒绝，与 _coerce_mask_rows / _coerce_numeric_vector 口径对齐。
            raise TypeError(f"{name}[{row_index}] must be a row of numeric values, not {type(row).__name__}")
        try:
            values = list(row)  # 每一行都先转成列表。
        except TypeError as exc:  # 行不可迭代时直接报错，便于定位。
            raise TypeError(f"{name}[{row_index}] must be a row of numeric values") from exc
        if len(values) != feature_dim:  # 每行宽度必须和特征顺序一致。
            raise ValueError(f"{name} rows must align with feature_order")
        normalized_rows.append(
            [
                coerce_finite_scalar(value, name=f"{name}[{row_index}][{col_index}]")
                for col_index, value in enumerate(values)
            ]
        )  # 每个值都统一抽成有限 float，错误信息携带行列索引便于定位。
    return normalized_rows


def _coerce_mask_rows(rows: Any, *, name: str, feature_dim: int) -> list[list[int]]:  # 把二维掩码窗口统一成整型二维列表。
    """把二维掩码窗口统一成整型二维列表。

    参数:
    `rows` 是待转换的掩码结构。
    `name` 是报错时显示的字段名。
    `feature_dim` 是每行应该具有的特征宽度。

    返回值:
    返回一个二维整型列表。
    """
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


def normalize_window_tensor(window_tensor: Any) -> dict[str, Any]:  # 把外部窗口输入标准化成网络内部统一字典。
    """把外部窗口输入规范化成网络内部可消费的字典。

    参数:
    `window_tensor` 是外部传入的结构化窗口。

    返回值:
    返回一个标准化后的窗口字典。

    失败条件:
    缺少必需键、长度不对、时间戳不合法时会报错。
    """
    if not isinstance(window_tensor, Mapping):  # 只接受映射型窗口输入。
        raise TypeError("window_tensor must be a structured feature window mapping")

    # Fast path for trainer-cached materialized samples: they were already
    # normalized once by _materialize_samples and only had tensor device moves after.
    # 必须校验全部输出合同键在场并核对缓存值类型，避免手拼字典绕过校验静默放入 NaN/Inf 或非序列对象。
    cached_feature_order = window_tensor.get("feature_order")
    cached_feature_window = window_tensor.get("feature_window")
    cached_missing_mask_window = window_tensor.get("missing_mask_window")
    cached_step_dts = window_tensor.get("step_dts")
    if (
        isinstance(cached_feature_order, list)
        and cached_feature_order
        and isinstance(cached_feature_window, (list, torch.Tensor))
        and isinstance(cached_missing_mask_window, (list, torch.Tensor))
        and isinstance(cached_step_dts, list)
        and "current_modality" in window_tensor
        and "feature_values" in window_tensor
        and "missing_mask" in window_tensor
        and "dt" in window_tensor
        and "window_index_map" in window_tensor
        and "event_time_window" in window_tensor
        and "readout_context_by_name" in window_tensor
        and "readout_context_observed_by_name" in window_tensor
        and "context_vector" in window_tensor
        and "filter_context_vector" in window_tensor
    ):
        # 张量快路径必须强制 mask 值为 0/1，与 LSTM normalize_structured_window 的口径对齐。
        # 列表路径已由 _coerce_mask_rows 保证 0/1，张量路径直接返回会绕过校验，静默放入 NaN/负值会污染均值。
        if isinstance(cached_missing_mask_window, torch.Tensor) and not (
            (cached_missing_mask_window == 0) | (cached_missing_mask_window == 1)
        ).all():
            offending = cached_missing_mask_window[
                ~((cached_missing_mask_window == 0) | (cached_missing_mask_window == 1))
            ].unique().tolist()
            raise ValueError(
                "missing_mask_window must contain only 0/1 values, got "
                f"{offending}"
            )
        return dict(window_tensor)

    feature_order = list(window_tensor.get("feature_order") or [])  # 读取特征顺序。
    if not feature_order:  # 特征顺序不能为空。
        raise ValueError("feature_window.feature_order must be non-empty")
    if any(not is_string_like(name) or not name for name in feature_order):  # 特征名必须是非空字符串。
        raise ValueError("feature_window.feature_order must contain non-empty field names")

    current_modality = _coerce_supported_modality(
        window_tensor.get("current_modality"),
        name="feature_window.current_modality",
    )  # 读取并校验当前模态，只允许协议支持的 UWB/VIO。

    feature_dim = len(feature_order)  # 特征维度由特征顺序决定。
    raw_feature_values = window_tensor.get("feature_values")
    feature_values = [] if raw_feature_values is None else _coerce_numeric_vector(
        raw_feature_values,
        name="feature_window.feature_values",
        cast=float,
    )  # 当前步特征值。
    raw_missing_mask = window_tensor.get("missing_mask")
    missing_mask = [] if raw_missing_mask is None else _coerce_mask_vector(
        raw_missing_mask,
        name="feature_window.missing_mask",
    )  # 当前步缺失掩码。
    dt = window_tensor.get("dt")  # 当前步时间间隔。
    if len(feature_values) != feature_dim or len(missing_mask) != feature_dim:  # 当前步向量必须和特征顺序对齐。
        raise ValueError("feature_window feature_order, feature_values, and missing_mask must align")
    if dt is None:  # dt 必须显式提供。
        raise ValueError("feature_window.dt must be provided explicitly")
    normalized_dt = coerce_finite_scalar(dt, name="feature_window.dt")  # 当前步时间间隔，单次解析并复用，避免下游重复调用与口径漂移。

    feature_window = _coerce_float_rows(  # 把整段特征窗口标准化成二维浮点列表。
        window_tensor.get("feature_window"),  # 外部传入的特征窗口。
        name="feature_window.feature_window",  # 报错字段名。
        feature_dim=feature_dim,  # 每行宽度要求。
    )
    missing_mask_window = _coerce_mask_rows(  # 把整段掩码窗口标准化成二维整型列表。
        window_tensor.get("missing_mask_window"),  # 外部传入的掩码窗口。
        name="feature_window.missing_mask_window",  # 报错字段名。
        feature_dim=feature_dim,  # 每行宽度要求。
    )
    if len(feature_window) != len(missing_mask_window):  # 特征窗口和掩码窗口行数必须一致。
        raise ValueError("feature_window.feature_window and missing_mask_window must have the same row count")
    if feature_values != feature_window[-1] or missing_mask != missing_mask_window[-1]:  # 当前步镜像必须和窗口最后一行一致。
        raise ValueError("feature_window current-step mirrors must match the last window row")
    event_time_window = _coerce_optional_time_window(  # 可选的事件时间窗。
        window_tensor.get("event_time_window"),  # 外部传入的事件时间序列。
        step_count=len(feature_window),  # 时间长度必须和窗口行数一致。
        name="feature_window.event_time_window",  # 报错字段名。
    )
    raw_readout_context = window_tensor.get("readout_context_by_name")
    raw_readout_observed = window_tensor.get("readout_context_observed_by_name")
    readout_context_by_name: dict[str, float] = {}
    readout_context_observed_by_name: dict[str, bool] = {}
    for key in _READOUT_CONTEXT_KEYS:
        observed = bool(raw_readout_observed.get(key, False) if isinstance(raw_readout_observed, Mapping) else False)
        has_raw_value = isinstance(raw_readout_context, Mapping) and key in raw_readout_context
        raw_value = 0.0 if not has_raw_value else raw_readout_context.get(key, 0.0)
        try:
            scalar = coerce_finite_scalar(raw_value, name=f"readout_context_by_name.{key}")
        except (TypeError, ValueError):
            scalar = 0.0
            observed = False
        if not has_raw_value:
            observed = False
        # coerce_finite_scalar 已保证有限性；无值或非法值统一降为“未观测”。
        readout_context_by_name[key] = scalar if observed else 0.0
        readout_context_observed_by_name[key] = observed
    explicit_context_vector = window_tensor.get("context_vector")
    explicit_filter_context_vector = window_tensor.get("filter_context_vector")
    # 校验显式上下文向量：拒绝 str/bytes/Mapping（虽有 __len__ 但不是合法数值向量），与 _coerce_numeric_vector 口径对齐。
    if explicit_context_vector is not None:
        if isinstance(explicit_context_vector, (str, bytes, Mapping)) or (
            not isinstance(explicit_context_vector, (list, tuple)) and not hasattr(explicit_context_vector, "__len__")
        ):
            import warnings
            warnings.warn(f"context_vector should be a sequence, got {type(explicit_context_vector).__name__}", stacklevel=2)
            explicit_context_vector = None
    if explicit_filter_context_vector is not None:
        if isinstance(explicit_filter_context_vector, (str, bytes, Mapping)) or (
            not isinstance(explicit_filter_context_vector, (list, tuple)) and not hasattr(explicit_filter_context_vector, "__len__")
        ):
            import warnings
            warnings.warn(f"filter_context_vector should be a sequence, got {type(explicit_filter_context_vector).__name__}", stacklevel=2)
            explicit_filter_context_vector = None

    raw_window_index_map = window_tensor.get("window_index_map")  # 显式区分 None 与空列表，避免 [] 被静默替换为 range。
    window_index_map = (
        list(raw_window_index_map) if raw_window_index_map is not None
        else list(range(len(feature_window)))  # 默认按顺序给窗口索引编号。
    )
    return {  # 返回标准化后的窗口字典，供后续递推和统计直接消费。
        "current_modality": current_modality,  # 当前步模态名。
        "feature_order": feature_order,  # 特征顺序，用来对齐列含义。
        "feature_values": feature_values,  # 当前步特征值镜像。
        "missing_mask": missing_mask,  # 当前步缺失掩码镜像。
        "dt": normalized_dt,  # 当前步时间间隔，已 coerce_finite_scalar 校验为有限 float。
        "feature_window": feature_window,  # 整段特征窗口。
        "missing_mask_window": missing_mask_window,  # 整段掩码窗口。
        "step_dts": _resolve_step_dts(  # 计算每一步实际用于递推的时间间隔序列。
            feature_order,  # 传入特征顺序，供 dt 列兜底解析。
            feature_window,  # 传入特征窗口，供 dt 列兜底解析。
            normalized_dt,  # 没有显式时间窗时的兜底步长，复用已校验的 normalized_dt。
            event_time_window=event_time_window,  # 有事件时间窗时优先按事件时间窗推导。
        ),
        "window_index_map": window_index_map,  # 窗口索引序列，None 时回退为顺序编号。
        "event_time_window": event_time_window,  # 原始事件时间窗，便于下游复用或排查。
        "readout_context_by_name": readout_context_by_name,
        "readout_context_observed_by_name": readout_context_observed_by_name,
        "context_vector": explicit_context_vector,
        "filter_context_vector": explicit_filter_context_vector,
    }


# 向后兼容别名：旧代码可能仍通过 _normalize_window_tensor 引用。
_normalize_window_tensor = normalize_window_tensor


def _reduce_feature_stats(  # 统计整段窗口里每个特征的均值和是否被观测到。
    feature_order: Sequence[str],
    feature_window: Sequence[Sequence[float]],
    missing_mask_window: Sequence[Sequence[int]],
) -> tuple[dict[str, float], dict[str, bool]]:
    if isinstance(feature_window, torch.Tensor) and isinstance(missing_mask_window, torch.Tensor):
        feature_means_by_name: dict[str, float] = {}
        feature_observed_by_name: dict[str, bool] = {}
        # 与列表路径 not bool(missing_row[index]) 严格对齐：仅 0 视为已观测。张量快路径未强制 mask 为 0/1，<= 0 会把负值误判为已观测而污染均值，故用 == 0。
        observed_mask = missing_mask_window == 0
        # 张量路径也必须校验有限性：观测位的NaN/Inf会被torch.where原样保留，污染均值。与列表路径的coerce_finite_scalar口径对齐。
        if observed_mask.any() and not torch.isfinite(feature_window[observed_mask]).all():
            raise ValueError("feature_window observed values must be finite (no NaN or Inf)")
        safe_window = torch.where(observed_mask, feature_window, torch.zeros_like(feature_window))
        observed_counts = observed_mask.sum(dim=0)
        # 用float64累加与列表路径的math.fsum精度对齐，防止同输入两路径产出不同位级均值。
        observed_sums = safe_window.to(torch.float64).sum(dim=0)
        for index, feature_name in enumerate(feature_order):
            count = int(observed_counts[index].item())
            if count <= 0:
                feature_means_by_name[str(feature_name)] = 0.0
                feature_observed_by_name[str(feature_name)] = False
                continue
            feature_means_by_name[str(feature_name)] = float(observed_sums[index].item()) / count
            feature_observed_by_name[str(feature_name)] = True
        return feature_means_by_name, feature_observed_by_name

    feature_means_by_name: dict[str, float] = {}  # 每个特征名对应的窗口均值（缺失列回退0.0，不为None）。
    feature_observed_by_name: dict[str, bool] = {}  # 每个特征名对应的观测标记。
    if len(feature_window) != len(missing_mask_window):  # 两个窗口行数必须一致，避免 zip 静默截断丢失行。
        raise ValueError(
            f"feature_window/missing_mask_window length mismatch: "
            f"{len(feature_window)}/{len(missing_mask_window)}"
        )
    for index, feature_name in enumerate(feature_order):  # 逐列统计，保证和特征顺序严格对齐。
        values = [  # 收集这一列里所有未被掩码掉的值。
            coerce_finite_scalar(feature_row[index], name=f"feature_window[{index}]")  # 当前行该特征的数值。
            for feature_row, missing_row in zip(feature_window, missing_mask_window)  # 同步遍历特征窗口和掩码窗口。
            if not bool(missing_row[index])  # 只有真正观测到的值才参与统计。
        ]
        if not values:  # 整段窗口都缺失时走中性回退，由 observed flag 显式表达“未观测”。
            feature_means_by_name[str(feature_name)] = 0.0  # 数值侧回退到中性值，避免上层读出统计链断裂。
            feature_observed_by_name[str(feature_name)] = False  # 保留缺失语义，供门控和读出层区分“零值”和“未观测”。
            continue
        feature_means_by_name[str(feature_name)] = math.fsum(values) / len(values)  # 对有效观测值求平均，用 fsum 提高精度。
        feature_observed_by_name[str(feature_name)] = True  # 标记为已观测。
    return feature_means_by_name, feature_observed_by_name  # 同时返回均值字典和观测字典。


def _resolve_current_feature_state(  # 统计当前步特征值和当前步观测状态。
    feature_order: Sequence[str],
    feature_values: Sequence[float],
    missing_mask: Sequence[int],
) -> tuple[dict[str, float], dict[str, bool]]:
    if isinstance(feature_values, torch.Tensor) and isinstance(missing_mask, torch.Tensor):
        current_feature_by_name: dict[str, float] = {}
        current_observed_by_name: dict[str, bool] = {}
        # 与列表路径 not bool(missing_value) 严格对齐：仅 0 视为已观测。张量快路径未强制 mask 为 0/1，
        # <= 0 会把负值误判为已观测而写入垃圾特征值，故用 == 0（与 _reduce_feature_stats 张量路径同口径）。
        observed_mask = missing_mask == 0
        # 张量路径与标量路径同口径：逐元素用 coerce_finite_scalar 校验并转 float，
        # 既挡住观测位 NaN/Inf 静默写入 dict，又保证两条路径产出位级一致（D7 公平性）。
        # coerce_finite_scalar 内部已 detach + 有限性校验，无需再保留 bulk isfinite 守卫或显式 detach。
        for index, feature_name in enumerate(feature_order):
            is_observed = bool(observed_mask[index].item())
            current_observed_by_name[str(feature_name)] = is_observed
            current_feature_by_name[str(feature_name)] = (
                coerce_finite_scalar(feature_values[index], name=f"feature_values.{feature_name}")
                if is_observed
                else 0.0
            )
        return current_feature_by_name, current_observed_by_name

    current_feature_by_name: dict[str, float] = {}  # 保存当前步每个特征的值。
    current_observed_by_name: dict[str, bool] = {}  # 保存当前步每个特征是否被观测到。
    if not (len(feature_order) == len(feature_values) == len(missing_mask)):  # 三向量长度必须一致，避免 zip 静默截断丢失特征。
        raise ValueError(
            f"feature_order/feature_values/missing_mask length mismatch: "
            f"{len(feature_order)}/{len(feature_values)}/{len(missing_mask)}"
        )
    for feature_name, feature_value, missing_value in zip(feature_order, feature_values, missing_mask):  # 逐项对齐当前步值和掩码。
        is_observed = not bool(missing_value)  # 掩码为 0 表示该特征可用。
        current_observed_by_name[str(feature_name)] = is_observed  # 记录这个特征当前是否有效。
        current_feature_by_name[str(feature_name)] = (
            coerce_finite_scalar(feature_value, name=f"feature_values.{feature_name}") if is_observed else 0.0
        )  # 未观测时用 0.0 占位。
    return current_feature_by_name, current_observed_by_name  # 返回当前步值字典和观测字典。


def _coerce_optional_time_window(  # 将可选时间窗统一成浮点列表，缺失时返回 None。
    raw_time_window: Any,
    *,
    step_count: int,
    name: str,
) -> list[float] | None:
    if raw_time_window is None:  # 没给时间窗时直接返回空。
        return None
    # _coerce_numeric_vector 已对每个元素调用 coerce_finite_scalar 保证有限性，
    # 并对 str/bytes/Mapping/不可迭代输入抛 TypeError、对 NaN/Inf 抛 ValueError，
    # 错误消息已带 {name}[index]；此处不再二次包装，避免把 ValueError 静默降级成 TypeError
    # 并丢失元素下标。与 common/validation.py 的口径保持单一真相源（D2 分层边界）。
    time_window = _coerce_numeric_vector(raw_time_window, name=name, cast=float)  # 每个时间点都转成有限 float。
    if len(time_window) != step_count:  # 时间长度必须和窗口步数一致。
        raise ValueError(f"{name} must align with the feature window row count")  # 不一致就报错。
    for previous_time, current_time in zip(time_window, time_window[1:]):  # 显式时间窗必须保持非递减，禁止把倒序事件洗成零步长。
        if current_time < previous_time:
            raise ValueError(f"{name} must be non-decreasing")
    return time_window  # 返回标准化后的时间窗。


def _resolve_step_dts(  # 计算每个时间步用于递推更新的时间间隔。
    feature_order: Sequence[str],  # 特征顺序，用来判断是否可以从窗口列里直接读 dt。
    feature_window: Sequence[Sequence[float]],  # 特征窗口，用来在没有事件时间窗时回退取值。
    fallback_dt: float,  # 兜底步长，前两种路径都不可用时使用。
    *,  # 下面参数必须关键字传入，避免调用时顺序混淆。
    event_time_window: Sequence[float] | None = None,  # 可选事件时间窗，存在时优先按它计算步长。
) -> list[float]:  # 返回每一步最终使用的时间间隔列表。
    # fallback_dt 统一经 coerce_finite_scalar 校验有限性与非负性，与 cell.py _coerce_dt_tensor 的非负口径对齐，避免分散的手动比对漂移。
    resolved_fallback_dt = coerce_finite_scalar(
        fallback_dt, name="fallback_dt", min_value=0.0,
    )
    if event_time_window is not None:  # 有显式事件时间窗时优先使用它。
        # 事件时间窗长度必须与特征窗口行数一致，保证返回的 step_dts 长度与下游递推合同对齐（D3 数据合同）。
        if len(event_time_window) != len(feature_window):
            raise ValueError(
                "event_time_window length must match feature_window row count: "
                f"got {len(event_time_window)} vs {len(feature_window)}"
            )
        if not event_time_window:  # 空窗口直接返回空列表，避免 event_time_window[0] 抛 IndexError。
            return []
        step_dts = [resolved_fallback_dt]  # 首步使用兜底步长，避免 dt=0 导致首步输入被忽略。
        previous_time = coerce_finite_scalar(event_time_window[0], name="event_time_window[0]")  # 先记住第一个时间点。
        for current_time in event_time_window[1:]:  # 从第二个时间点开始逐段计算差值。
            current_time = coerce_finite_scalar(current_time, name="event_time_window value")  # 每个时间点都规范成有限 float。
            # coerce_finite_scalar 已保证 previous_time/current_time 有限；负时间差属于数据错误，与 _coerce_optional_time_window 的非递减校验口径一致，直接报错而非静默截断。
            step_dt = current_time - previous_time
            if step_dt < 0.0:
                raise ValueError(
                    f"event_time_window must be non-decreasing, got negative step dt={step_dt}"
                )
            step_dts.append(step_dt)
            previous_time = current_time  # 更新前一个时间点。
        return step_dts  # 显式时间窗优先时直接返回。

    try:  # 没有事件时间窗时，尝试从特征列里找 dt。
        dt_index = list(feature_order).index("dt")  # 找到 dt 列的位置。
    except ValueError:  # 没有 dt 列时就继续使用兜底值。
        dt_index = None  # 这里用 None 表示不能从特征列读取 dt。

    step_dts: list[float] = []  # 存每一步最终使用的 dt。
    for feature_row in feature_window:  # 逐行读取窗口中的每一步。
        candidate = resolved_fallback_dt  # 每一步默认先拿兜底 dt。
        if dt_index is not None:  # 只有存在 dt 列时才尝试读取。
            # 分离"读取失败"（索引越界/不可索引，回退兜底）与"值不合法"（NaN/Inf/负值，直接报错），避免把数据错误静默洗成兜底 dt。
            try:
                raw_dt = feature_row[dt_index]
            except (TypeError, IndexError):
                raw_dt = None  # 读取失败，保持兜底 dt。
            if raw_dt is not None:
                candidate = coerce_finite_scalar(
                    raw_dt, name="feature_window.dt_column", min_value=0.0,
                )  # dt 列也要求非负，与 fallback_dt 口径对齐，防止负时间间隔静默穿透到 cell 递推。
        step_dts.append(candidate)  # candidate 要么是兜底 dt，要么是经校验的 dt 列值，均为有限非负 float。
    return step_dts  # 返回整段窗口的 dt 序列。


class LiquidNetwork(nn.Module):  # 面向缺失感知窗口的 Liquid 序列网络，只输出共享主干特征。
    """面向缺失感知窗口的 Liquid 序列网络，输出共享主干特征。

    这个类负责把一段窗口序列逐步送入 `LiquidCell`，再把每步结果汇总成共享特征。
    它不负责四个任务头的最终映射，也不负责训练循环和损失函数。

    状态变化说明:
    - `__init__` 负责读配置和准备共享层。
    - `_build_cell` 会在输入维度明确后创建 cell 和池化门。
    - `extract_shared_features` 会沿时间步推进并生成共享特征字典。
    - `forward` 只是把调用包装成统一的共享特征输出。

    失败条件说明:
    - `hidden_dim`、`input_dim` 或 `pooling_logit_scale` 不合法时会报错。
    - 窗口结构、特征顺序、时间戳或状态形状不对时会报错。
    """

    def __init__(self, network_cfg: Mapping[str, Any] | None = None):  # 读取网络配置并准备共享层与可选 cell。
        super().__init__()  # 先初始化 nn.Module，保证参数注册和子模块管理走标准流程。
        if network_cfg is not None and not isinstance(network_cfg, Mapping):  # 配置必须是映射型或 None，拒绝 str/list 等静默转换。
            raise TypeError(f"network_cfg must be a mapping or None, got {type(network_cfg).__name__}")
        cfg = dict(network_cfg or {})  # 把映射型配置转成普通字典，便于逐项读取。
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "hidden_dim": cfg.get("hidden_dim") if isinstance(cfg, dict) else None,
            "input_dim": cfg.get("input_dim") if isinstance(cfg, dict) else None,
            "feature_order_len": len(cfg.get("feature_order", [])) if isinstance(cfg, dict) else None,
            "pooling_logit_scale": cfg.get("pooling_logit_scale") if isinstance(cfg, dict) else None,
        }, "LiquidNetwork.__init__ 入口参数")
        raw_hidden_dim = cfg.get("hidden_dim", 64)  # 读取隐藏维度。
        if not is_integer(raw_hidden_dim):  # hidden_dim 必须是整数（排除 bool）。
            raise TypeError(f"hidden_dim must be an integer, got {type(raw_hidden_dim).__name__}")
        if raw_hidden_dim < 1:  # hidden_dim 不能小于 1。
            raise ValueError(f"hidden_dim must be positive, got {raw_hidden_dim}")
        if raw_hidden_dim > MAX_MODEL_DIM:  # 上界保护，防止 nn.Linear 构造时 OOM；与 cell.py _require_int / _build_cell 同源（common.constants.MAX_MODEL_DIM），禁止本地重复字面量漂移（D5/D7/D9）。
            raise ValueError(f"hidden_dim must be <= {MAX_MODEL_DIM} to avoid OOM, got {raw_hidden_dim}")

        raw_input_dim = cfg.get("input_dim")  # 可选输入维度。
        if raw_input_dim is None:  # 没提供时保持未知。
            input_dim = None
        elif not is_integer(raw_input_dim):  # 提供时必须是整数（排除 bool）。
            raise TypeError(f"input_dim must be an integer when provided, got {type(raw_input_dim).__name__}")
        elif raw_input_dim < 1:  # 小于 1 的输入维度是非法值，直接报错而非静默回退 None。
            raise ValueError(f"input_dim must be positive when provided, got {raw_input_dim}")
        elif raw_input_dim > MAX_MODEL_DIM:  # 上界保护，与 cell.py _require_int / _build_cell 同源（common.constants.MAX_MODEL_DIM），禁止本地重复字面量漂移（D5/D7/D9）。
            raise ValueError(f"input_dim must be <= {MAX_MODEL_DIM} to avoid OOM, got {raw_input_dim}")
        else:
            input_dim = raw_input_dim  # 合法时直接采用。

        self.hidden_dim = raw_hidden_dim  # 保存隐藏维度。
        self.input_dim = input_dim  # 保存输入维度，如果暂时未知就留空。
        raw_output_heads = list(cfg.get("output_heads") or MODEL_INTERMEDIATE_KEYS)  # 输出头默认引用单源真相，与 LSTMNetwork.__init__ / resolve_lstm_network_cfg 口径对齐（D7/D9）。
        if raw_output_heads != list(MODEL_INTERMEDIATE_KEYS):  # 输出头顺序和数量都不能偏，与 LSTM 侧保持一致。
            raise ValueError(  # 不一致直接报错，保证协议稳定。
                "network_cfg.output_heads must keep the fixed order "
                f"{list(MODEL_INTERMEDIATE_KEYS)}."
            )
        self.output_heads = tuple(raw_output_heads)  # 保存校验后的输出头顺序。
        raw_feature_order_cfg = cfg.get("feature_order")  # 读取特征顺序原始配置。
        # 字符串/字节/映射本身可迭代但会被 list() 拆成字符或键序列，必须提前拒绝；
        # 与 LSTM 侧 _coerce_feature_order 及 cell.py 的拒绝口径对齐，保证公平性（D7/D9）。
        if raw_feature_order_cfg is not None and (
            is_string_like(raw_feature_order_cfg)
            or isinstance(raw_feature_order_cfg, (bytes, bytearray, Mapping))
        ):
            raise TypeError("feature_order must be an iterable of feature names, not str/bytes/mapping")
        raw_feature_order = list(raw_feature_order_cfg or [])  # 固化成列表，后面就不会被外部迭代器耗尽。
        if any(not is_string_like(name) or not str(name).strip() for name in raw_feature_order):  # 特征名必须是非空字符串，拒绝整数/None 等被 str() 掩盖的类型错误。
            raise ValueError("feature_order must contain non-empty string names")
        if len(set(str(name) for name in raw_feature_order)) != len(raw_feature_order):  # 重复名会导致 _feature_index_by_name 静默覆盖，与 cell.py 对齐。
            raise ValueError("feature_order must not contain duplicate field names")
        self.feature_order = tuple(str(name) for name in raw_feature_order)  # 保存特征顺序。
        raw_pooling_logit_scale = cfg.get("pooling_logit_scale", 0.35)  # 读取池化门控尺度。
        # 统一走 coerce_finite_scalar 完成类型/有限性/区间校验，防止裸 float() 在巨大整数时抛 OverflowError 或 NaN/Inf 静默穿透 [0,1] 比较；与项目根因修复规范对齐（D5）。
        self.pooling_logit_scale = coerce_finite_scalar(
            raw_pooling_logit_scale,
            name="pooling_logit_scale",
            min_value=0.0,
            max_value=1.0,
        )
        self.cell_cfg = {
            "hidden_dim": self.hidden_dim,  # 传给 cell 的隐藏维度。
            "feature_order": list(self.feature_order),  # 传给 cell 的特征顺序。
            # 更新幅度：缺省跟随 yaml 冻结面 0.70/0.30；显式 cfg 可覆盖。
            # 注意：cell.py 自身 default 是 0.10/0.80，但 network 入口必须与 liquid_ekf.yaml 对齐，
            # 避免“没写 yaml 键时 silently 换一套更新幅度”。
            "update_scale_floor": cfg.get("cell_update_scale_floor", 0.70),
            "update_scale_span": cfg.get("cell_update_scale_span", 0.30),
            # 可靠性偏置：与 liquid_ekf.yaml / cell 当前主链 1.0 对齐。
            "reliability_bias_init": cfg.get("reliability_bias_init", 1.0),
            "bad_observation_floor": cfg.get("bad_observation_floor", 0.20),  # 坏观测下限，默认值与 cell.py 对齐。
            "bad_observation_span": cfg.get("bad_observation_span", 0.80),  # 坏观测跨度，默认值与 cell.py 对齐。
            "bad_observation_async_full_scale": cfg.get(  # 异步尺度，引用单源真相常量，与 cell.py 默认值同源（D9）。
                "bad_observation_async_full_scale", float(ASYNC_GAP_FULL_SCALE_S),
            ),
            "bad_observation_residual_full_scale": cfg.get("bad_observation_residual_full_scale", 0.50),  # 残差尺度，默认值与 cell.py 对齐。
            "bad_observation_reproj_full_scale": cfg.get(  # 重投影尺度，引用单源真相常量，与 cell.py 默认值同源（D9）。
                "bad_observation_reproj_full_scale", float(VIO_REPROJ_ERR_FULL_SCALE),
            ),
            "bad_observation_tracked_features_floor": cfg.get(  # 特征数阈值，引用单源真相常量，与 cell.py 默认值同源（D9）。
                "bad_observation_tracked_features_floor", float(VIO_TRACKED_FEATURES_FLOOR),
            ),
            "bad_observation_track_drop_full_scale": cfg.get("bad_observation_track_drop_full_scale", 25.0),  # 掉特征尺度，默认值与 cell.py 对齐。
            "bad_observation_reproj_slope_full_scale": cfg.get(  # 误差斜率尺度，引用单源真相常量，与 cell.py 默认值同源（D9）。
                "bad_observation_reproj_slope_full_scale", float(VIO_HIGH_REPROJ_ERR_THRESHOLD),
            ),
            "bad_observation_geom_floor": cfg.get("bad_observation_geom_floor", 0.30),  # 几何下限，默认值与 cell.py 对齐。
            "bad_observation_interaction_coeff": cfg.get("bad_observation_interaction_coeff", 0.75),  # 交互项系数，默认值与 cell.py 对齐。
            # 2026-07-26 τ-scale / OOD 门控透传：network 层必须把这些键交给 LiquidCell，
            # 否则 yaml 配置静默失效，cell 只能吃硬编码默认值。
            # 默认值与 cell.py 同源（180/s，按 sim 真实 dt 校准），避免 yaml 缺键时漂到旧 1/s 量级。
            "time_rate_scale": cfg.get("time_rate_scale", 180.0),
            "cfA_clip_min": cfg.get("cfA_clip_min", 0.05),
            "cfA_clip_max": cfg.get("cfA_clip_max", 0.99),
            "cfB_clip_min": cfg.get("cfB_clip_min", 0.05),
            "cfB_clip_max": cfg.get("cfB_clip_max", 0.99),
            "enable_forget_root": cfg.get("enable_forget_root", True),
            "risk_short_circuit_threshold": cfg.get("risk_short_circuit_threshold", 1.0),
        }
        self.cell: LiquidCell | None = None  # 先不建 cell，等输入维度明确后再建。
        self.pooling_gate: nn.Linear | None = None  # 池化门也先占位。
        self.shared_projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim)  # 共享主干投影。
        self.shared_activation = nn.Tanh()  # 共享主干激活。
        self.reset_parameters()  # 初始化共享投影层。
        if self.input_dim is not None:  # 如果输入维度已知，就顺手建 cell。
            self._build_cell(self.input_dim)

    def reset_parameters(self) -> None:  # 重置所有可学习参数到初始状态。
        """重置所有可学习参数到初始状态。

        重置范围：
        - shared_projection：用 _fill_linear 等间距初始化（始终存在）。
        - cell（若已构建）：委托给 LiquidCell.reset_parameters。
        - pooling_gate（若已构建）：权重和偏置清零，与 _build_cell 初始化口径对齐。

        与 LiquidCell.reset_parameters、OutputHead.reset_parameters 的"重置全部参数"
        口径对齐；与 LSTMNetwork.reset_parameters 保持同名同口径（D7 公平性）。

        参数:
        这个方法不接收外部参数，只操作当前实例内部已经创建好的层。

        返回值:
        无返回值，只修改模块内部状态。
        """
        _fill_linear(self.shared_projection, start=-0.10, end=0.10)  # 初始化共享投影层。
        if self.cell is not None:  # cell 已构建时一并重置，遵循"重置全部参数"约定。
            self.cell.reset_parameters()
        if self.pooling_gate is not None:  # pooling_gate 已构建时清零，与 _build_cell 初始化口径对齐。
            with torch.no_grad():  # 重置阶段不需要梯度记录。
                self.pooling_gate.weight.zero_()
                if self.pooling_gate.bias is not None:  # 如果有偏置就清零。
                    self.pooling_gate.bias.zero_()

    def _build_cell(self, input_dim: int) -> None:  # 在输入维度明确后创建内部 cell 和池化门。
        """在输入维度明确后构建内部 cell 和池化门。

        参数:
        `input_dim` 是当前窗口的特征维度。

        返回值:
        无返回值，只更新内部结构。
        """
        if not is_integer(input_dim):  # input_dim 必须是整数（排除 bool/np.bool_），与本类 __init__ 及 cell.py _require_int 口径对齐；非整数报 TypeError 而非 ValueError，避免类型错误被误判为值域错误。
            raise TypeError(f"input_dim must be an integer, got {type(input_dim).__name__}: {input_dim!r}")
        if input_dim < 1:  # 正整数至少为 1，防止 nn.Linear 构造时 OOM。
            raise ValueError(f"input_dim must be a positive integer, got {input_dim!r}")
        if input_dim > MAX_MODEL_DIM:  # 上界保护，与 cell.py _require_int 同源（common.constants.MAX_MODEL_DIM），禁止本地重复字面量漂移。
            raise ValueError(f"input_dim must be <= {MAX_MODEL_DIM} to avoid OOM, got {input_dim}")
        input_dim = int(input_dim)  # 归一为 Python int，与 cell.py _require_int 返回 int(raw_value) 口径对齐，避免 np.integer 流入 self.input_dim 与后续 nn.Linear。
        device = next(self.shared_projection.parameters()).device  # 跟随共享层所在设备。
        self.cell = LiquidCell({"input_dim": input_dim, **self.cell_cfg}).to(device)  # 构建 cell。
        pooling_input_dim = self.hidden_dim + (input_dim * 2)  # 池化门输入宽度。
        self.pooling_gate = nn.Linear(pooling_input_dim, 1).to(device)  # 构建池化门。
        with torch.no_grad():  # 初始化阶段不需要梯度。
            self.pooling_gate.weight.zero_()  # 先让池化门从中性开始。
            if self.pooling_gate.bias is not None:  # 如果有偏置就清零。
                self.pooling_gate.bias.zero_()
        self.input_dim = input_dim  # 记录当前输入维度。

    def reset_state(  # 复位内部 cell 状态，通常用于新序列开始时。
        self,
        *,  # 强制后续参数只能用关键字传入。
        batch_size: int = 1,  # 要重置多少条状态向量。
        device: torch.device | None = None,  # 状态张量所在设备。
        dtype: torch.dtype | None = None,  # 状态张量的数据类型。
    ) -> torch.Tensor:
        if self.cell is None:  # cell 还没建好时无法重置。
            raise RuntimeError("LiquidNetwork state is unavailable before input_dim is known")
        # 边界先校验：与 __init__ hidden_dim / _build_cell input_dim 口径对齐，
        # 在网络层入口就拒绝非法 batch_size（bool/float/负数），避免错误归因到下游 cell（D5/D10）。
        # cell.reset_state 仍保留同口径校验作为内部不变量守卫，双重校验是防御性边界模式，非冗余。
        if not is_integer(batch_size):  # 布尔值和非整数（含 torch.Tensor）都拒绝，与 is_integer 语义一致。
            raise TypeError(f"batch_size must be an integer, got {type(batch_size).__name__}")
        if batch_size < 1:  # 批次大小至少为 1，防止 torch.zeros 退化成空张量污染下游 shape 推断。
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return self.cell.reset_state(batch_size=batch_size, device=device, dtype=dtype)  # 直接委托给 cell。

    def extract_shared_features(  # 抽取共享特征的主入口，会沿时间步推进并返回完整字典。
        self,
        window_tensor: Any,  # 当前窗口输入。
        initial_state: Any | None = None,  # 可选的初始状态。
    ) -> dict[str, Any]:
        normalized_window = normalize_window_tensor(window_tensor)  # 先把窗口输入规范化。
        feature_order = normalized_window["feature_order"]  # 特征顺序。
        feature_dim = len(feature_order)  # 当前特征维度。
        device = next(self.shared_projection.parameters()).device  # 共享层所在设备。
        if self.cell is None or self.input_dim != feature_dim:  # 如果 cell 还没建好或输入维度变化了，就重建。
            self._build_cell(feature_dim)

        feature_tensor = torch.as_tensor(  # 把特征窗口转成张量。
            normalized_window["feature_window"],  # 特征窗口数据。
            dtype=torch.float32,  # 使用浮点类型。
            device=device,  # 放到当前设备。
        )
        missing_tensor = torch.as_tensor(  # 把掩码窗口转成张量。
            normalized_window["missing_mask_window"],  # 掩码窗口数据。
            dtype=torch.float32,  # 使用浮点类型。
            device=device,  # 放到当前设备。
        ).clamp(0.0, 1.0)  # 非原地：torch.as_tensor 可能返回 view，原地 clamp 会改调用方张量

        # Fix 1 (D8/D10)：空窗口 guard。fast-path 缓存窗口可能 feature_window 为空，
        # 此时下方 zip 不产出任何步、torch.stack([]) 会抛模糊的 RuntimeError。
        # 显式拒绝并指向 feature_window，避免上游契约违规被掩盖成栈错误。
        if feature_tensor.shape[0] == 0:
            raise ValueError("feature_window must contain at least one row for time-step unrolling")

        # Fix 2 (D3/D10)：step_dts 长度对齐 guard。normalize_window_tensor 的 fast-path
        # （trainer 缓存样本）只校验 step_dts 是 list，不校验长度与 feature_window 行数一致。
        # 长度不一致会在循环里 IndexError 或在 compute_step_weights 里 shape 报错，根因被掩盖。
        step_dts = normalized_window["step_dts"]
        if len(step_dts) != feature_tensor.shape[0]:
            raise ValueError(
                f"step_dts length ({len(step_dts)}) must match feature_window row count "
                f"({feature_tensor.shape[0]})"
            )

        # Fix 3 (D5/D7)：前置 bulk isfinite 检查。与 LSTMNetwork._coerce_sequence_batch 的
        # `if not torch.isfinite(sequence_tensor).all()` 同口径（D7 公平性）。
        # 原实现依赖 cell.step 内部 _as_float_tensor 在循环中逐次抛错，错误信息指向
        # "input_tensor" 而非 "feature_window"，且对 NaN@masked 位也会拒绝（与 cell.step 行为一致，
        # 不改变语义，仅提前抛出并给出更准确的字段名）。
        if not torch.isfinite(feature_tensor).all():
            raise ValueError("feature_window contains non-finite values (NaN or Inf)")
        if not torch.isfinite(missing_tensor).all():
            raise ValueError("missing_mask_window contains non-finite values (NaN or Inf)")

        if initial_state is None:  # 没给初始状态时用零状态。
            hidden_state = self.reset_state(  # 自动创建初始隐藏状态。
                batch_size=1,  # 当前实现按单条窗口处理。
                device=device,  # 放到当前设备。
                dtype=feature_tensor.dtype,  # 与特征张量保持一致。
            )
        else:  # 外部给了初始状态，就把它规范成网络内部使用的张量。
            hidden_state = torch.as_tensor(initial_state, dtype=feature_tensor.dtype, device=device)  # 读取外部初始状态。
            if hidden_state.ndim == 1:  # 一维状态自动补 batch 维。
                hidden_state = hidden_state.unsqueeze(0)
            if hidden_state.shape != (1, self.hidden_dim):  # 形状必须匹配。
                raise ValueError(  # Fix 4 (D6)：错误信息补 actual shape，便于定位上游构造错误。
                    f"initial_state must have shape ({self.hidden_dim},) or (1, {self.hidden_dim}), "
                    f"got {tuple(hidden_state.shape)}"
                )

        hidden_steps = []  # 存每一步的 cell 输出。
        for index, (feature_row, missing_row) in enumerate(zip(feature_tensor, missing_tensor)):  # 逐步展开窗口。
            hidden_state, cell_output = self.cell.step(  # 每个时间步推进一次 cell。
                feature_row.unsqueeze(0),  # 当前步特征。
                hidden_state,  # 上一步隐藏状态。
                dt=step_dts[index],  # 当前步时间间隔（复用已校验长度的 step_dts 别名）。
                missing_mask=missing_row.unsqueeze(0),  # 当前步掩码。
            )
            hidden_steps.append(cell_output.squeeze(0))  # 收集当前步输出。

        hidden_sequence = torch.stack(hidden_steps, dim=0)  # 合成时间序列。
        final_hidden = hidden_state.squeeze(0)  # 最后一步隐藏状态。
        step_weights = self.compute_step_weights(  # 计算每一步的池化权重。
            hidden_sequence.unsqueeze(0),  # batch 维。
            feature_tensor.unsqueeze(0),  # batch 维。
            missing_tensor.unsqueeze(0),  # batch 维。
            torch.as_tensor(  # 把 step_dts 转成张量（复用已校验长度的别名）。
                step_dts,  # 每步 dt。
                dtype=hidden_sequence.dtype,  # 与隐藏序列同 dtype。
                device=hidden_sequence.device,  # 与隐藏序列同设备。
            ).unsqueeze(0),
        ).squeeze(0)
        pooled_hidden = (hidden_sequence * step_weights.unsqueeze(-1)).sum(dim=0)  # 按权重池化。
        shared_vector = self.shared_activation(  # 再做共享主干投影。
            self.shared_projection(torch.cat((final_hidden, pooled_hidden), dim=0))
        )
        feature_means_by_name, feature_observed_by_name = _reduce_feature_stats(  # 统计整段窗口的特征均值和观测状态。
            feature_order,  # 特征顺序。
            normalized_window["feature_window"],  # 特征窗口。
            normalized_window["missing_mask_window"],  # 掩码窗口。
        )
        current_feature_by_name, current_observed_by_name = _resolve_current_feature_state(  # 统计当前步特征状态。
            feature_order,  # 特征顺序。
            normalized_window["feature_values"],  # 当前步特征值。
            normalized_window["missing_mask"],  # 当前步掩码。
        )

        return {  # 返回整段窗口的共享特征和若干中间统计量。
            "current_modality": normalized_window["current_modality"],  # 当前模态名。
            "feature_order": list(feature_order),  # 特征顺序。
            "feature_window": normalized_window["feature_window"],  # 特征窗口原始值。
            "missing_mask_window": normalized_window["missing_mask_window"],  # 掩码窗口原始值。
            "window_index_map": normalized_window["window_index_map"],  # 窗口索引映射。
            "event_time_window": normalized_window["event_time_window"],  # 事件时间窗。
            "shared_vector": shared_vector,  # 共享主干向量。
            "final_hidden": final_hidden,  # 最后一步隐藏状态。
            "pooled_hidden": pooled_hidden,  # 池化后的隐藏状态。
            "hidden_sequence": hidden_sequence,  # 每一步的隐藏序列。
            "feature_means_by_name": feature_means_by_name,  # 按名字统计的特征均值。
            "feature_observed_by_name": feature_observed_by_name,  # 按名字统计的观测标记。
            "current_feature_by_name": current_feature_by_name,  # 当前步特征字典。
            "current_observed_by_name": current_observed_by_name,  # 当前步观测字典。
            "readout_context_by_name": dict(normalized_window["readout_context_by_name"]),
            "readout_context_observed_by_name": dict(normalized_window["readout_context_observed_by_name"]),
            "context_vector": normalized_window.get("context_vector"),
            "filter_context_vector": normalized_window.get("filter_context_vector"),
        }

    def extract_shared_features_batch(  # 批量抽取共享特征，对同一序列长度的窗口做真正的批量矩阵运算。
        self,
        window_list: list[dict[str, Any]],  # 已标准化的窗口字典列表，所有窗口必须同序列长度和特征维度。
        initial_states: list[Any | None] | None = None,  # 可选的初始状态列表，None 时自动初始化为零。
        *,
        minimal_output_head_features: bool = False,
    ) -> list[dict[str, Any]]:
        """批量提取共享特征，对同一序列长度的窗口做批量前向。

        与逐个调用 extract_shared_features 不同，此方法将多个窗口的特征
        堆叠成批次张量，在 cell 递推时做真正的批量矩阵运算，而非逐样本循环。
        所有窗口必须具有相同的序列长度和特征维度（由训练器的长度分组保证）。

        参数:
        `window_list` 是已标准化的窗口字典列表。
        `initial_states` 是可选的初始状态列表。

        返回值:
        返回共享特征字典列表，每个元素与 extract_shared_features 返回值一致。

        失败条件:
        窗口列表为空、序列长度或特征维度不一致时会抛出异常。
        """
        if not window_list:  # 空列表直接返回。
            return []
        if initial_states is not None and len(initial_states) != len(window_list):  # 初始状态数量必须和窗口数量一致。
            raise ValueError("initial_states length must match window_list length")
        if len(window_list) == 1:  # 单样本走原有路径，避免不必要的堆叠开销。
            single_result = self.extract_shared_features(
                window_list[0],
                initial_state=initial_states[0] if initial_states else None,
            )
            if minimal_output_head_features:
                return [self._to_minimal_output_head_shared_features(single_result)]
            return [single_result]

        normalized_windows = [normalize_window_tensor(window) for window in window_list]  # 批量路径与单样本路径保持同一入口合同。
        first = normalized_windows[0]  # 取第一个窗口获取公共信息。
        feature_order = first.get("feature_order", [])  # 特征顺序。
        if isinstance(feature_order, torch.Tensor):  # 张量形式转列表。
            feature_order = feature_order.tolist()
        for window in normalized_windows[1:]:  # 整个批次必须共享同一列语义，避免批量路径静默串列。
            current_order = window.get("feature_order", [])
            if isinstance(current_order, torch.Tensor):
                current_order = current_order.tolist()
            if list(current_order) != list(feature_order):
                raise ValueError("all windows in a liquid batch must share the same feature_order")
        feature_dim = len(feature_order)  # 当前特征维度。
        device = next(self.shared_projection.parameters()).device  # 共享层所在设备。

        if self.cell is None or self.input_dim != feature_dim:  # cell 还没建好或维度变了就重建。
            self._build_cell(feature_dim)

        batch_size = len(window_list)  # 批次大小。

        # 按序列长度分组，避免 torch.stack 要求等长。
        seq_lens = [
            len(w["feature_window"]) if not isinstance(w["feature_window"], torch.Tensor) else w["feature_window"].shape[0]
            for w in normalized_windows
        ]
        unique_lens = sorted(set(seq_lens))  # 去重排序。

        # 如果所有窗口等长，走原来的快速路径。
        if len(unique_lens) == 1:
            return self._extract_shared_features_batch_uniform(
                normalized_windows,
                initial_states,
                feature_order,
                feature_dim,
                device,
                minimal_output_head_features=minimal_output_head_features,
            )

        # 按长度分组处理，再按原始顺序合并。
        results_by_index: dict[int, dict[str, Any]] = {}
        for seq_len_val in unique_lens:
            group_indices = [i for i, sl in enumerate(seq_lens) if sl == seq_len_val]
            group_windows = [normalized_windows[i] for i in group_indices]
            group_initial_states = None
            if initial_states is not None:
                group_initial_states = [initial_states[i] for i in group_indices]
            group_results = self._extract_shared_features_batch_uniform(
                group_windows,
                group_initial_states,
                feature_order,
                feature_dim,
                device,
                minimal_output_head_features=minimal_output_head_features,
            )
            if len(group_results) != len(group_indices):  # D10: zip 静默截断会掩盖 helper 返回数量不一致，导致下游 KeyError 丢失根因。
                raise RuntimeError(
                    f"_extract_shared_features_batch_uniform returned {len(group_results)} results "
                    f"for {len(group_indices)} windows (seq_len={seq_len_val})"
                )
            for idx, result in zip(group_indices, group_results):
                results_by_index[idx] = result

        return [results_by_index[i] for i in range(batch_size)]

    def _extract_shared_features_batch_uniform(
        self,
        window_list: list[dict[str, Any]],
        initial_states: list[Any] | None,
        feature_order: list[str],
        feature_dim: int,
        device: torch.device,
        *,
        minimal_output_head_features: bool = False,
    ) -> list[dict[str, Any]]:
        """对等长窗口执行批量共享特征提取。"""
        batch_size = len(window_list)

        # 堆叠特征窗口 (B, T, F) 和掩码窗口 (B, T, F)。
        feature_tensors = torch.stack([  # 把每个窗口的特征窗口堆叠成批次张量。
            torch.as_tensor(w["feature_window"], dtype=torch.float32, device=device)
            for w in window_list
        ])
        missing_tensors = torch.stack([  # 把每个窗口的掩码窗口堆叠成批次张量。
            torch.as_tensor(w["missing_mask_window"], dtype=torch.float32, device=device)
            for w in window_list
        ]).clamp(0.0, 1.0)  # 非原地：与单样本路径保持一致

        # 堆叠 step_dts (B, T)。
        step_dts_tensor = torch.stack([  # 把每个窗口的时间间隔堆叠成批次张量。
            torch.as_tensor(w["step_dts"], dtype=torch.float32, device=device)
            for w in window_list
        ])

        # Fix D5/D7（批量有限性守卫）：与单样本路径 extract_shared_features 的 bulk isfinite
        # 检查（行 747-750）同口径。原实现依赖 cell.step 内部 _as_float_tensor 在循环中逐次抛错，
        # 错误信息指向 "input_tensor"/"missing_mask" 而非 "feature_window"/"missing_mask_window"，
        # 且批量路径只在第一个含 NaN 的步才暴露，掩盖了根因字段。前置 bulk 检查立即抛出并指向
        # 正确字段名，与 LSTMNetwork._coerce_sequence_batch 的 isfinite 守卫（行 561）对齐（D7 公平性）。
        if not torch.isfinite(feature_tensors).all():
            raise ValueError("feature_window contains non-finite values (NaN or Inf)")
        if not torch.isfinite(missing_tensors).all():
            raise ValueError("missing_mask_window contains non-finite values (NaN or Inf)")

        # 初始化隐藏状态 (B, hidden_dim)。
        if initial_states is not None:  # 外部提供了初始状态。
            if len(initial_states) != batch_size:  # 初始状态数量必须和窗口数量一致。
                raise ValueError("initial_states length must match window_list length")
            normalized_states: list[torch.Tensor] = []  # 先逐个规范化，保持与单样本路径一致。
            for state in initial_states:
                if state is None:
                    normalized_states.append(
                        self.reset_state(batch_size=1, device=device, dtype=torch.float32).squeeze(0)
                    )
                    continue
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                if state_tensor.ndim == 2 and state_tensor.shape == (1, self.hidden_dim):
                    state_tensor = state_tensor.squeeze(0)
                elif state_tensor.ndim != 1 or state_tensor.shape[0] != self.hidden_dim:
                    # Fix D6/D7：错误信息补 actual shape，与单样本路径行 763-766 口径对齐，
                    # 便于定位上游构造错误（原实现只报期望形状，不报实际形状）。
                    raise ValueError(
                        f"each initial state must have shape ({self.hidden_dim},) or "
                        f"(1, {self.hidden_dim}), got {tuple(state_tensor.shape)}"
                    )
                normalized_states.append(state_tensor)
            hidden_state = torch.stack(normalized_states)  # 堆叠成 (B, hidden_dim)。
        else:  # 没给初始状态时用零状态。
            hidden_state = self.reset_state(
                batch_size=batch_size, device=device, dtype=torch.float32,
            )

        # Fix D8/D10/D7（批量形状与边界守卫）：与单样本路径 extract_shared_features 的
        # 空窗口 guard（行 729-730）和 step_dts 长度对齐 guard（行 736-740）同口径。
        # 原实现缺少这些守卫：feature_tensors.ndim!=3 时 shape[1] 会取错维度（特征维被当成序列长）；
        # 空窗口会导致 torch.stack([]) 抛模糊 RuntimeError；step_dts 长度不一致会在循环里 IndexError
        # 或在 compute_step_weights 里 shape 报错，根因被掩盖。feature_dim 守卫利用此前未使用的
        # 形参（D1 死参数根因修复），防止 caller 传入不一致的 feature_dim 与实际特征维度。
        if feature_tensors.ndim != 3:
            raise ValueError(
                f"feature_window must be 2D per window, got stacked ndim={feature_tensors.ndim}"
            )
        seq_len = feature_tensors.shape[1]  # 序列长度（所有窗口相同）。
        if seq_len == 0:
            raise ValueError("feature_window must contain at least one row for time-step unrolling")
        if step_dts_tensor.shape[1] != seq_len:
            raise ValueError(
                f"step_dts length ({step_dts_tensor.shape[1]}) must match feature_window row count "
                f"({seq_len})"
            )
        if feature_tensors.shape[-1] != feature_dim:
            raise ValueError(
                f"feature_window last dimension ({feature_tensors.shape[-1]}) must match "
                f"feature_dim ({feature_dim})"
            )

        # 逐步展开窗口 — 批量处理：每步同时处理 B 个样本。
        hidden_steps: list[torch.Tensor] = []  # 存每步的 cell_output。
        for t in range(seq_len):  # 沿时间步推进。
            feature_row = feature_tensors[:, t, :]  # 当前步特征 (B, F)。
            missing_row = missing_tensors[:, t, :]  # 当前步掩码 (B, F)。
            dt_row = step_dts_tensor[:, t]  # 当前步时间间隔 (B,)。
            hidden_state, cell_output = self.cell.step(  # 批量 cell 更新。
                feature_row,
                hidden_state,
                dt=dt_row,
                missing_mask=missing_row,
            )
            hidden_steps.append(cell_output)  # 收集 (B, hidden_dim)。

        # hidden_sequence: (B, T, hidden_dim)。
        hidden_sequence = torch.stack(hidden_steps, dim=1)

        # 计算池化权重 (B, T)。
        step_weights = self.compute_step_weights(
            hidden_sequence, feature_tensors, missing_tensors, step_dts_tensor,
        )

        # 池化: (B, hidden_dim)。
        pooled_hidden = (hidden_sequence * step_weights.unsqueeze(-1)).sum(dim=1)

        # 共享向量: (B, hidden_dim)。
        shared_vector = self.shared_activation(
            self.shared_projection(torch.cat((hidden_state, pooled_hidden), dim=-1))
        )

        # 为每个样本构建共享特征字典。
        results: list[dict[str, Any]] = []
        for i in range(batch_size):
            w = window_list[i]  # 当前样本的原始窗口元数据。
            # 逐样本统计信息：特征均值和观测状态。
            fw = w["feature_window"]  # 特征窗口（可能是张量或列表）。
            mw = w["missing_mask_window"]  # 掩码窗口。
            fv = w["feature_values"]  # 当前步特征值。
            mm = w["missing_mask"]  # 当前步掩码。
            current_feature_by_name, current_observed_by_name = _resolve_current_feature_state(
                feature_order, fv, mm,
            )
            result = {
                "current_modality": w["current_modality"],
                "feature_order": list(feature_order),
                "shared_vector": shared_vector[i],  # (hidden_dim,) 张量。
                "final_hidden": hidden_state[i],  # (hidden_dim,) 张量。
                "pooled_hidden": pooled_hidden[i],  # (hidden_dim,) 张量。
                "current_feature_by_name": current_feature_by_name,
                "current_observed_by_name": current_observed_by_name,
                # Fix D3/D7：readout_context_* 改为直接访问 + dict 拷贝，与单样本路径行 820-821
                # 同口径（fail fast）。normalize_window_tensor 总是设置这两个键为 dict（行 360-361），
                # .get(... or {}) 默认值只会掩盖缺键 bug。context_vector/filter_context_vector
                # 单样本路径也用 .get()（行 822-823），此处保持一致，无需修改。
                "readout_context_by_name": dict(w["readout_context_by_name"]),
                "readout_context_observed_by_name": dict(w["readout_context_observed_by_name"]),
                "context_vector": w.get("context_vector"),
                "filter_context_vector": w.get("filter_context_vector"),
            }
            if minimal_output_head_features:
                results.append(self._to_minimal_output_head_shared_features(result))
                continue

            feature_means_by_name, feature_observed_by_name = _reduce_feature_stats(
                feature_order, fw, mw,
            )
            # Fix D3/D7：返回类型与单样本路径 extract_shared_features（行 808-811）对齐。
            # 原实现 fw.tolist() 强制把张量转成 list，与单样本返回原始 feature_window 类型不一致，
            # 导致同一公共 API extract_shared_features_batch 在 batch_size==1（走单样本路径）与
            # batch_size>1（走批量路径）时返回类型不同（tensor vs list），违反数据合同（D3）。
            # window_index_map/event_time_window 改为直接访问，与单样本路径同口径（fail fast）：
            # normalize_window_tensor 总是设置这两个键（行 340-343/359），.get() 默认值只会
            # 掩盖 normalize_window_tensor 缺键的 bug；单样本路径已是直接访问，批量路径须对齐。
            result.update({
                "feature_window": fw,
                "missing_mask_window": mw,
                "window_index_map": w["window_index_map"],
                "event_time_window": w["event_time_window"],
                "hidden_sequence": hidden_sequence[i],  # (T, hidden_dim) 张量。
                "feature_means_by_name": feature_means_by_name,
                "feature_observed_by_name": feature_observed_by_name,
            })
            results.append(result)

        return results

    @staticmethod
    def _to_minimal_output_head_shared_features(shared_features: Mapping[str, Any]) -> dict[str, Any]:
        """裁剪成输出头前向真正需要的最小 shared_features 合同。

        必需键缺失时直接抛 KeyError，避免静默兜底掩盖上游合同漂移（D4 异常路径清晰）。
        final_hidden / pooled_hidden 不再用 shared_vector 兜底，防止形状不一致的
        张量悄悄流入输出头前向（D5 数值安全）。current_modality 不再回退空串，
        空串不是合法模态、会在下游 _coerce_supported_modality 才报错，掩盖根因。
        context_vector / filter_context_vector 保留 .get() 返回 None，与
        extract_shared_features 显式允许 None 的合同一致。
        """
        return {
            "current_modality": shared_features["current_modality"],
            "feature_order": list(shared_features["feature_order"]),
            "shared_vector": shared_features["shared_vector"],
            "final_hidden": shared_features["final_hidden"],
            "pooled_hidden": shared_features["pooled_hidden"],
            "current_feature_by_name": dict(shared_features["current_feature_by_name"]),
            "current_observed_by_name": dict(shared_features["current_observed_by_name"]),
            "readout_context_by_name": dict(shared_features["readout_context_by_name"]),
            "readout_context_observed_by_name": dict(shared_features["readout_context_observed_by_name"]),
            "context_vector": shared_features.get("context_vector"),
            "filter_context_vector": shared_features.get("filter_context_vector"),
        }

    def forward_shared(  # 暴露与 extract_shared_features 完全一致的共享特征接口。
        self,
        window_tensor: Any,  # 当前窗口输入。
        initial_state: Any | None = None,  # 可选的初始状态。
    ) -> dict[str, Any]:
        """把共享特征提取接口暴露成单独方法。

        参数:
        `window_tensor` 是当前窗口输入。
        `initial_state` 是可选初始状态。

        返回值:
        返回和 `extract_shared_features` 完全一致的字典。
        """
        return self.extract_shared_features(window_tensor, initial_state=initial_state)  # 直接转发。

    def compute_step_weights(  # 计算池化时每一步的权重。
        self,
        hidden_sequence: torch.Tensor,  # 每一步的隐藏序列。
        feature_tensor: torch.Tensor,  # 每一步的特征序列。
        missing_tensor: torch.Tensor,  # 每一步的缺失掩码序列。
        step_dts: torch.Tensor,  # 每一步对应的时间间隔。
    ) -> torch.Tensor:
        """计算池化时每一步的权重。

        参数:
        `hidden_sequence` 是形状为 `(batch, steps, hidden_dim)` 的隐藏序列。
        `feature_tensor` 是形状为 `(batch, steps, feature_dim)` 的特征序列。
        `missing_tensor` 是形状为 `(batch, steps, feature_dim)` 的缺失掩码序列。
        `step_dts` 是形状为 `(batch, steps)` 的时间间隔序列。

        返回值:
        返回形状为 `(batch, steps)` 的 softmax 权重。

        失败条件:
        输入维度不对齐、池化门未构建或时间步结构不一致时会报错。
        """
        if self.pooling_gate is None:  # 池化门还没构建时不能算权重。
            raise RuntimeError("LiquidNetwork pooling gate is unavailable before input_dim is known")
        if hidden_sequence.ndim != 3:  # 隐序列必须是三维，错误消息带上实际 ndim 便于定位根因（D6）。
            raise ValueError(
                f"hidden_sequence must have shape (batch, steps, hidden_dim), got ndim={hidden_sequence.ndim}"
            )
        if feature_tensor.ndim != 3:  # 特征序列必须是三维，单独报错避免与 missing_tensor 误判混在一起（D6）。
            raise ValueError(
                f"feature_tensor must have shape (batch, steps, feature_dim), got ndim={feature_tensor.ndim}"
            )
        if missing_tensor.ndim != 3:  # 掩码序列必须是三维，与 feature_tensor 分开报错便于定位（D6）。
            raise ValueError(
                f"missing_tensor must have shape (batch, steps, feature_dim), got ndim={missing_tensor.ndim}"
            )
        if step_dts.ndim != 2:  # dt 序列必须是二维。
            raise ValueError(
                f"step_dts must have shape (batch, steps), got ndim={step_dts.ndim}"
            )
        if hidden_sequence.shape[:2] != feature_tensor.shape[:2]:  # batch 和步数都必须对齐，错误消息带上实际 shape（D6）。
            raise ValueError(
                f"hidden_sequence and feature_tensor must align on batch and steps: "
                f"hidden_sequence.shape={tuple(hidden_sequence.shape)}, "
                f"feature_tensor.shape={tuple(feature_tensor.shape)}"
            )
        if feature_tensor.shape != missing_tensor.shape:  # 特征和掩码必须完全同形，分开校验避免合并报错掩盖根因（D6）。
            raise ValueError(
                f"feature_tensor and missing_tensor must share shape: "
                f"feature_tensor.shape={tuple(feature_tensor.shape)}, "
                f"missing_tensor.shape={tuple(missing_tensor.shape)}"
            )
        if hidden_sequence.shape[:2] != step_dts.shape:  # dt 序列也必须和 batch/步数对齐。
            raise ValueError(
                f"step_dts must align with hidden_sequence batch and steps: "
                f"step_dts.shape={tuple(step_dts.shape)}, "
                f"hidden_sequence.shape[:2]={tuple(hidden_sequence.shape[:2])}"
            )

        observed_features = torch.where(  # 用 where 屏蔽缺失位，避免 NaN*0 仍传播 NaN。
            missing_tensor.bool(),
            feature_tensor.new_zeros(()),
            feature_tensor,
        )  # 先把缺失位置清零后的观测特征取出来。
        pooling_inputs = torch.cat((hidden_sequence, observed_features, missing_tensor), dim=-1)  # 把隐藏序列、观测特征和掩码拼成池化输入。
        learned_delta = self.pooling_logit_scale * torch.tanh(self.pooling_gate(pooling_inputs).squeeze(-1))  # 学得的池化偏移量。
        if not torch.isfinite(learned_delta).all():  # NaN/Inf 会污染 softmax 产生 NaN 权重，进而传播到 pooled_hidden 与下游 readout；与 coerce_finite_scalar 对 NaN/Inf 报错的口径一致，根因修复而非静默掩盖（D5）。
            raise ValueError(
                "compute_step_weights learned_delta contains non-finite values (NaN/Inf); "
                "this indicates upstream NaN/Inf in hidden_sequence or feature_tensor"
            )
        # 保持零初始化的池化门尽量中性。
        # LiquidCell 在递推更新里已经会使用每步 dt，所以这里不再额外塞入反比 dt 先验。
        # 这样能避免在学习前就过度偏向高频步或零 dt 步。
        base_logits = torch.zeros_like(step_dts)  # 先给所有步一个中性基础分数。
        return torch.softmax(base_logits + learned_delta, dim=1)  # 再做 softmax，得到池化权重。

    def forward(  # nn.Module.__call__ 入口，返回包装了 shared_features 的字典。
        self,
        window_tensor: Any,  # 当前窗口输入。
        initial_state: Any | None = None,  # 可选初始状态。
    ) -> dict[str, Any]:
        """nn.Module.__call__ 兼容入口，返回包装了 shared_features 的字典。

        注意：训练和推理主路径直接调用 `extract_shared_features` /
        `extract_shared_features_batch`，不经过本方法。本方法主要为
        PyTorch `nn.Module.__call__` 协议和 inference fallback 链保留。

        参数:
        `window_tensor` 是当前窗口输入。
        `initial_state` 是可选初始状态。

        返回值:
        返回一个只包含 `shared_features` 的字典。
        """
        return {SHARED_FEATURES_KEY: self.extract_shared_features(window_tensor, initial_state=initial_state)}  # 直接包装共享特征，键名引用单源真相常量（D9 根因修复）。
