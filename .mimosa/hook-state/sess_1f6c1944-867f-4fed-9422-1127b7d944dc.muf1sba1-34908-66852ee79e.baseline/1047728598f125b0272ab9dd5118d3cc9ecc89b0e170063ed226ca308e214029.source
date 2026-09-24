"""Data utilities submodule for Liquid trainer.

Handles sample extraction, feature ordering, window normalization,
sample weighting, and output projection.
"""
from __future__ import annotations

import torch

from pathlib import Path
from typing import Any, Mapping

from ._constants import (  # type: ignore[import]
    _ACTIVE_HEADS_BY_MODALITY,
    _OUTPUT_KEYS,
    _SCALING_NEUTRAL_FLOOR,
    _TAIL_SELECTION_OBSERVATION_COEFF,
    _TAIL_SELECTION_MAX_WEIGHT,
    _TAIL_SELECTION_TAIL_COEFF,
)
from liquidloc.common.seed_utils import set_global_seed
from liquidloc.common.types import ModelIntermediate
from liquidloc.common.validation import coerce_finite_scalar, is_string_like
from liquidloc.common.constants import BRIDGE_BIAS_MAX, DEVICE_CPU  # 设备默认值常量（D9 单源真相）
from liquidloc.models.features.normalization import neutral_floor_softplus
from liquidloc.models.liquid.network import normalize_window_tensor
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS

def _extract_window_and_target(sample: Any) -> tuple[dict[str, Any], dict[str, float]]:
    """从原始样本中分离特征窗口和目标中间结果。

    样本可以是完整的映射（含 window_tensor ?target_intermediate），
    也可以直接就是窗口映射本身（此时 target_intermediate 必须在顶层）。

    参数：
        sample: 原始训练/验证样本。

    返回值：
    二元（window_tensor, target_outputs)。
        - window_tensor: 特征窗口字典（浅拷贝；窗口标准化由调用方负责。
        ?LSTM 侧在函数内即标准化的口径不同，Liquid 推迟。
          _materialize_samples 调用 normalize_window_tensor 完成）；
        - target_outputs: 校验后的目标输出字典。

    失败条件。
        TypeError: 当样本或窗口不是映射类型时抛出。
        ValueError: 当样本缺target_intermediate 时抛出。
    """

    if not isinstance(sample, Mapping):

        raise TypeError("train/val samples must be mappings containing a structured window")



    window_tensor = sample.get("window_tensor")

    if window_tensor is None:

        window_tensor = sample

    if not isinstance(window_tensor, Mapping):  # 窗口必须是字典式映射。
        raise TypeError("window_tensor must be a structured feature window mapping")  # 类型不对就报错。


    raw_target = sample.get("target_intermediate")  # 取训练目标中间结果。
    if raw_target is None:  # 每个样本都必须带目标。
        raise ValueError("each train/val sample must contain target_intermediate")  # 缺失就报错。
    return dict(window_tensor), _coerce_target_outputs(raw_target)  # 返回窗口浅拷贝和目标字典；窗口标准化由调用方负责（与 LSTM 侧口径不同）。




def _materialize_samples(
    samples: list[Any],
    *,
    tail_selection_observation_coeff: float = _TAIL_SELECTION_OBSERVATION_COEFF,
) -> list[tuple[dict[str, Any], torch.Tensor, int, float]]:
    """将原始样本列表材料化为训练器可直接消费的格式。

    对每个样本：分离窗口和目标、标准化窗口结构、校验特征顺序一致。
    按固定顺序构造目标张量、计算、样权重。返回的四元组包。
    (元数。 目标张量, 序列长度, 选样权重)。

    参数：
        samples: 原始训练/验证样本列表。

    返回值：
        材料化样本列表，每个元素是四元组。
        (metadata_dict, target_tensor, seq_len, selection_weight)。。

    失败条件。
        ValueError: 当不同样本的 feature_order 不一致时抛出。
    """

    resolved_tail_selection_observation_coeff = coerce_finite_scalar(
        tail_selection_observation_coeff,
        name="tail_selection_observation_coeff",
        min_value=0.0,
    )

    targets: list[torch.Tensor] = []  # 存每个样本的目标张量。
    metadata: list[dict[str, Any]] = []  # 存每个样本的窗口元数据。
    selection_weights: list[float] = []  # 存每个样本的选样权重。
    feature_order: list[str] | None = None  # 先用首个样本锁定特征顺序。
    for sample in samples:  # 逐个样本整理成训练器霢、要的格式。
        window_tensor, target_outputs = _extract_window_and_target(sample)  # 分离窗口和目标。
        normalized = normalize_window_tensor(window_tensor)  # 标准化窗口结构。
        if feature_order is None:  # 第一条样本决定特征顺序。
            feature_order = list(normalized["feature_order"])  # 记录基准顺序。
        elif list(normalized["feature_order"]) != feature_order:  # 后续样本必须完全致。
            raise ValueError("all liquid training windows must share the same feature_order")  # 不一致就报错。
        targets.append(torch.tensor([target_outputs[key] for key in _OUTPUT_KEYS], dtype=torch.float32))  # 目标按固定顺序入张量。
        metadata.append(normalized)  # 保存标准化后的窗口元数据。
        selection_weights.append(
            _resolve_selection_sample_weight(
                sample,
                target_outputs,
                observation_coeff=resolved_tail_selection_observation_coeff,
            )
        )  # 计算样本权重。
    return [  # 把三路结果重新打包成材料化样本。
        (meta, target, len(meta["feature_window"]), selection_weight)  # 元数据、目标、长度、权重。
        for meta, target, selection_weight in zip(metadata, targets, selection_weights)  # 按样本、个打包。
    ]  # 材料化样本结束。




def _resolve_sample_cache_device(model: Any) -> torch.device:
    """根据模型实际参数所在的设备，决定样本缓存的目标设备。

    §13.8 OOM 根因修复: 原始代码将 cache_device 硬编码为 torch.device('cpu')，
    导致 1860 序列的全部训练样本 (~2.5M 个) 的 window_tensor / target_intermediate
    / target_trace 全部常驻 CPU 内存，峰值 RSS 超出可用内存而崩溃。
    改为与模型同一设备: 模型在 cuda 时缓存到 GPU，模型在 cpu 时缓存到 cpu。
    GPU 7.96 GB 显存足够装 1860 序列样本 (~5 GB), CPU 内存同时释放。

    参数：
        model: Liquid 模型实例，其参数决定了目标设备。

    返回值：
        torch.device: 与模型参数同设备的目标设备。
    """
    try:
        # 从任意一个参数的 device 推断（模型已在 L5162+ 移动到 runtime_device）
        p = next(model.parameters()) if hasattr(model, 'parameters') and callable(model.parameters) else None
        if p is not None:
            return p.device
        # 无参数（模型为空）时回退 cpu
        return torch.device(DEVICE_CPU)
    except (StopIteration, TypeError):
        return torch.device(DEVICE_CPU)


def _move_materialized_samples(

    materialized: list[tuple[dict[str, Any], torch.Tensor, int, float]],

    device: torch.device,

) -> list[tuple[dict[str, Any], torch.Tensor, int, float]]:
    """将材料化样本中所有张量搬到目标设备。

    包括特征窗口、缺失掩码窗口特征、缺失掩码。
    读出上下文等所有张量字段，以及目标张量本身。

    参数：
        materialized: 材料化样本列表。
        device: 目标设备（如 cuda ?cpu）。

    返回值：
        搬运后的材料化样本列表，结构不变，张量在目标设备上。
    """

    moved_samples: list[tuple[dict[str, Any], torch.Tensor, int, float]] = []  # 存搬到目标设备后的样本。
    for metadata, target_tensor, seq_len, selection_weight in materialized:  # 逐个样本搬设备。
        moved_metadata = dict(metadata)  # 先拷贝元数据，避免原地改动。
        moved_metadata["feature_window"] = torch.as_tensor(moved_metadata["feature_window"], dtype=torch.float32, device=device)  # 特征窗口搬到目标设备。
        moved_metadata["missing_mask_window"] = torch.as_tensor(moved_metadata["missing_mask_window"], dtype=torch.float32, device=device)  # 缺失掩码窗口搬设备。
        moved_metadata["feature_values"] = torch.as_tensor(moved_metadata["feature_values"], dtype=torch.float32, device=device)  # 特征值搬设备。
        moved_metadata["missing_mask"] = torch.as_tensor(moved_metadata["missing_mask"], dtype=torch.float32, device=device)  # 缺失掩码搬设备。
        if "readout_context_by_name" in moved_metadata:  # 如果有读出上下文字典。
            moved_metadata["readout_context_by_name"] = {  # 把每个、都搬到目标设备。
                str(key): torch.as_tensor(value, dtype=torch.float32, device=device).reshape(())  # 转成标量张量。
                for key, value in dict(moved_metadata["readout_context_by_name"]).items()  # 遍历每个键、对。
            }
        if "readout_context_observed_by_name" in moved_metadata:  # 如果有读出上下文观测标记字典。
            moved_metadata["readout_context_observed_by_name"] = {  # 把每个、转成布尔。
                str(key): bool(value)  # 观测标记是布尔、，不需要搬设备。
                for key, value in dict(moved_metadata["readout_context_observed_by_name"]).items()  # 遍历每个键、对。
            }
        moved_samples.append((moved_metadata, target_tensor.to(device), seq_len, selection_weight))  # 追加搬运后的样本。
    return moved_samples  # 返回搬运结果。




def _resolve_feature_order(train_windows: list[Any], trainer_state: dict[str, Any]) -> list[str]:
    """解析特征顺序，优先使用配置中的显式顺序，否则从第丢、条窗口推导。

    参数：
        train_windows: 训练窗口列表，至少包含含一个元素。
        trainer_state: 训练器状态字典，可能包含 "feature_order" 键。

    返回值：
        特征名字符串列表。

    失败条件。
        ValueError: 当既没有配置也没有内嵌特征顺序，或字段名为空/重复时抛出。
        TypeError: ?feature_order 是字符串/字节/映射或不可迭代对象时抛出。
    """

    raw_feature_order = trainer_state["feature_order"]  # 取显式配置（可能为空列表）。
    if not raw_feature_order:  # 状里没有显式顺序时，从窗口推导。
        if not train_windows:  # 既没配置也没有窗口，就无法推导。
            raise ValueError("feature_order must be provided in train_cfg or in the training windows")
        window_tensor, _ = _extract_window_and_target(train_windows[0])  # 用第丢、条训练窗口推导顺序。
        raw_feature_order = window_tensor.get("feature_order")  # 从窗口里读特征顺序。

        # ?lstm/network.py _coerce_feature_order、liquid/network.py normalize_window_tensor 对齐。
    # 字符字节字节数组本身可迭代但会被拆成字符/整数序列，必须拒绝；
    # 映射（dict）会静默用键当字段顺序，掩盖调用方传错意图，也必须拒绝。
    # is_string_like 覆盖 numpy.str_（NumPy 2.x 不再str 子类）。
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
    if not feature_order:  # 既没配置也没内嵌，就不能继续。
        raise ValueError("feature_order must be provided in train_cfg or in the training windows")
    for index, feature_name in enumerate(feature_order):  # 逐个检查特征名。
        if not is_string_like(feature_name) or not feature_name:  # 必须是非空字符串。
            raise ValueError(f"feature_order[{index}] must be a non-empty string.")
    if len(set(feature_order)) != len(feature_order):  # 重复名会导致列对齐静默错位，LiquidNetwork.__init__ 对齐。
        raise ValueError("feature_order must not contain duplicate field names.")
    return feature_order  # 返回校验后的特征顺序。


def _resolve_selection_sample_weight(
    sample: Mapping[str, Any],
    target_outputs: Mapping[str, float],
    *,
    observation_coeff: float = _TAIL_SELECTION_OBSERVATION_COEFF,
) -> float:
    """根据目标风险和跟踪信号计算样本的选样权重。

    高风险、高对齐风险、高观测风险的样本应该得到更高权重，
    以确保训练不会忽略尾部困难样本、权重从 1.0 开始，
    按尾部信号和观测信号的加成、增，最终截断到硬上限。

    参数：
        sample: 原始样本映射，可能包target_trace 子映射。
        target_outputs: 已校验的目标输出字典，至少包含含"risk" 键。

    返回值：
    截断到 [1.0, _TAIL_SELECTION_MAX_WEIGHT] 区间内的样本权重浮点数。
    """

    resolved_observation_coeff = coerce_finite_scalar(
        observation_coeff,
        name="tail_selection_observation_coeff",
        min_value=0.0,
    )

    if "risk" not in target_outputs:
        raise KeyError("target_intermediate is missing required key: 'risk'")
    risk = _clamp_unit_interval(target_outputs["risk"], name="target_intermediate.risk")

    alignment_risk = risk  # 默认先把对齐风险设成risk。
    observation_risk = risk  # 观测风险默认也先沿用risk。
    quality_risk = risk  # 质量风险默认先沿用主 risk。
    modality_signal = 0.0  # 模、信号默认没有额外加成。
    target_trace = sample.get("target_trace")  # 取目标跟踪信息，可能为空。
    if target_trace is not None and not isinstance(target_trace, Mapping):
        raise TypeError(f"target_trace must be a mapping, got {type(target_trace).__name__}")
    if isinstance(target_trace, Mapping):  # 只有字典型 trace 才展开读取。
        alignment_risk = _clamp_unit_interval(  # 读取并裁剪对齐风险。
            target_trace.get("alignment_risk", alignment_risk),  # 没有就回到默认。
            name="target_trace.alignment_risk",  # 报错字段名。
        )

        observation_risk = _clamp_unit_interval(  # 读取并裁剪观测风险。
            target_trace.get("observation_risk", observation_risk),  # 没有就回到默认。
            name="target_trace.observation_risk",  # 报错字段名。
        )

        quality_risk = _clamp_unit_interval(  # 读取并裁剪质量风险。
            target_trace.get("quality_risk", quality_risk),  # 没有就回到默认。
            name="target_trace.quality_risk",  # 报错字段名。
        )

        modality_signal = _clamp_unit_interval(  # 读取并裁剪模态信号。
            target_trace.get("modality_signal", modality_signal),  # 没有就回到默认。
            name="target_trace.modality_signal",  # 报错字段名。
        )

    tail_signal = max(risk, alignment_risk, observation_risk)  # 尾部信号取几个风险里朢、强的那个。
    observation_tail = max(observation_risk, quality_risk, modality_signal)  # 观测尾部再综合质量和模信号。
    return min( # 最终再限制到上限。
        _TAIL_SELECTION_MAX_WEIGHT,  # 尾部权重绝不能超过硬上限。
        1.0  # 基础权重 1.0 开始。
        + (_TAIL_SELECTION_TAIL_COEFF * tail_signal)  # 尾部风险越高，权重越大。
        + (resolved_observation_coeff * observation_tail),  # 观测、质量和模、信号以较弱系数补充抬权。
    )  # 返回截断后的最终样本权重。


def _resolve_current_modality(window_tensor: Mapping[str, Any]) -> str:
    """从窗口张量中提取并校验当前模态名。

    训练器只认识 _ACTIVE_HEADS_BY_MODALITY 中注册的模。
    ?uwb" ?"vio"），其他模一律报错。

    参数：
        window_tensor: 包含 "current_modality" 字段的结构化窗口映射。

    返回值：
    校验通过的模态名字符串（"uwb" ?"vio"）。

    失败条件。
        ValueError: 当模态不是字符串、或strip().lower() 归一化后不在合法集合中时抛出。
        归一化口径与 liquidloc.models.liquid.network._coerce_supported_modality 保持一致，
        确保 LSTM ?Liquid 公平对比时模态校验完全相同。
    """
    raw_modality = window_tensor.get("current_modality")
    if not is_string_like(raw_modality): # 模态必须是字符串，拒绝 None/数值、布尔/列表等静默转换。
        raise ValueError(  # 报错时把合法集合带出来，方便排查。
            "feature_window.current_modality must be one of "
            f"{sorted(_ACTIVE_HEADS_BY_MODALITY)} for trainer loss"
        )
    modality = str(raw_modality).strip().lower() # 与 network._coerce_supported_modality 保持同一归一化口径.
    if modality not in _ACTIVE_HEADS_BY_MODALITY:  # 只允许训练器认识的模态。
        raise ValueError(  # 报错时把合法集合带出来，方便排查。
            "feature_window.current_modality must be one of "
            f"{sorted(_ACTIVE_HEADS_BY_MODALITY)} for trainer loss"
        )

    return modality  # 返回校验通过的模态名。




def _coerce_positive_sample_weight(value: Any, *, name: str) -> float:
    """校验并转换样本权重，要求必须是有限正数。

    样本权重用于在损失计算中对不同样本施加不同重要、，
    零或负权重没有意义，NaN/Inf 也不合法。

    参数：
        value: 待校验的权重值，必须是数值类型（不接受bool）。
        name: 字段名，仅用于报错信息。

    返回值：
        校验通过的浮点权重。

    失败条件。
        TypeError: 当输入为 bool、非数、类型或非标量时抛出。
        ValueError: 当输入为 NaN、无穷大或非正数时抛出。
    """
    return coerce_finite_scalar(value, name=name, min_value=0.0, inclusive=False)  # 委托公共校验入口，保证与 LSTM 侧口径一致。




def _clamp_unit_interval(value: Any, *, name: str) -> float:
    """将输入、裁剪到 [0, 1] 区间，并校验有限性。

    用于对风险、信号等概率域标量做合法性检查和截断。

    参数：
        value: 待裁剪的输入值，可以是任何可float 的类型。
        name: 字段名，仅用于报错信息。

    返回值：
    裁剪到 [0, 1] 的浮点数。

    失败条件。
        ValueError: 当输入为 NaN 或无穷大时抛出。
    """
    scalar_value = coerce_finite_scalar(value, name=name)  # 统一走中心校验：排除 bool、处理单元素张量、有限、检查并带字段名报错。
    return min(1.0, max(0.0, scalar_value))  # 再裁剪到单位区间 [0, 1]。




def _build_head_mask_tensor(

    modalities: list[str],

    *,

    reference_tensor: torch.Tensor,

) -> torch.Tensor:
    """按模态列表构造输出头掩码张量。

    每种模、只濢、活自己对应的输出头（_ACTIVE_HEADS_BY_MODALITY），
    其余头位置填零，这样损失函数只对被激活的头做回归。
    掩码的dtype 和device 中reference_tensor 一致。

    参数：
        modalities: 当前批次每个样本的模态名列表。
        reference_tensor: 用于确定 dtype ?device 的参考张量。

    返回值：
        形状态(batch_size, len(_OUTPUT_KEYS)) 的掩码张量，
        濢、活位置为 1.0，其余为 0.0。

    失败条件。
        ValueError: 当模态不被支持或模态数量与参数张量 batch 维不一致。
        或整个批次没有激活头时抛出。
    """
    # ?LSTM ?_build_head_mask_tensor 对齐：模态数量必须和参数张量batch 维一致，
    # 否则后续 mask * sample_weight_tensor 会因广播错误产生静默错误结果。
    if len(modalities) != int(reference_tensor.shape[0]):
        raise ValueError(
            "modalities length must match reference_tensor batch dimension: "
            f"got {len(modalities)} modalities vs batch {int(reference_tensor.shape[0])}"
        )
    mask = torch.zeros((len(modalities), len(_OUTPUT_KEYS)), dtype=reference_tensor.dtype, device=reference_tensor.device)

    for row_index, modality in enumerate(modalities):

        if modality not in _ACTIVE_HEADS_BY_MODALITY:  # 只接受训练器认识的模态。
            raise ValueError(f"unsupported modality for trainer loss: {modality!r}")  # 不合法就直接报错。
        for head_key in _ACTIVE_HEADS_BY_MODALITY[modality]:  # 每种模、只濢、活自己对应的头。
            mask[row_index, _OUTPUT_KEYS.index(head_key)] = 1.0  # 把对应位置标成可监督。
    if float(mask.sum().detach().item()) <= 0.0:  # 如果整个批次都没有激活头，就说明输入有问题。
        raise ValueError("trainer loss head mask must activate at least one head")  # 直接失败，避免除零。
    return mask  # 返回构、好的输出头掩码。




def _build_sample_weight_tensor(

    sample_weights: list[float] | None,

    *,

    reference_tensor: torch.Tensor,

) -> torch.Tensor:
    """将样本权重列表转为张量，未提供时返回全 1。

    样本权重用于在损失计算中对不同样本施加不同重要、，
    例如高风险样本可以给更高权重。每个权重都必须是有限正数。

    参数：
        sample_weights: 样本权重列表，None 时默认等权。
        reference_tensor: 用于确定 batch_size、dtype ?device 的参考张量。

    返回值：
        形状态(batch_size,) 的样本权重张量。

    失败条件。
        ValueError: 当权重数量与 batch_size 不一致或权重不合法时抛出。
    """
    batch_size = int(reference_tensor.shape[0])  # 批大小直接从参、张量取。
    if sample_weights is None:  # 没传权重时，默认每个样本等权。
        return torch.ones((batch_size,), dtype=reference_tensor.dtype, device=reference_tensor.device)  # 直接返回全 1 张量。
    if len(sample_weights) != batch_size:  # 权重数量必须batch 大小致。
        raise ValueError("sample_weights must match the prediction batch size")  # 不一致就报错。
    return torch.tensor(  # 把校验后Python 列表转成张量。
        [_coerce_positive_sample_weight(weight, name=f"sample_weights[{index}]") for index, weight in enumerate(sample_weights)],  # 每个权重都、个校验。
        dtype=reference_tensor.dtype,  # dtype 和参考张量一致。
        device=reference_tensor.device,  # device 和参考张量一致。
    )  # 张量构结束。




def _coerce_raw_output_vector(raw_output: Any) -> torch.Tensor:
    """将模型原始输出统一转为丢、维张量，并校验输出头数量。

    支持标量张量、一维张量和 batch=1 的二维张量作为输入，
    最终压平成固定长度len(_OUTPUT_KEYS) 的一维张量。

    参数：
        raw_output: 模型原始输出，可以是张量、列表或其他可转换类型。

    返回值：
        形状态(4,) 的一float32 张量，顺序为 bias, risk, uwb_scaling, vio_scaling。

    失败条件。
    ValueError: ?batch 维度不为 1、输出头数量不等 4 或存NaN/Inf 非有限、时抛出。
    """

    # 统一转成 float32 张量；张量输入时保留原设备且仅在 dtype 不一致时拷贝（与 lstm/trainer.py _project_train_outputs 口径一致，D4 设备/dtype 对齐、D7 公平性）。
    raw_tensor = torch.as_tensor(raw_output, dtype=torch.float32)
    if raw_tensor.ndim == 2:  # 如果是二维，通常表示batch 维。
        if raw_tensor.shape[0] != 1:  # 这里只允batch=1。
            raise ValueError("model output batch dimension must be 1 for trainer loss.")  # 不满足就报错。
        raw_tensor = raw_tensor.squeeze(0)  # 去掉单元batch 维。
        raw_tensor = raw_tensor.reshape(-1) # 然后压平成一维。
    if raw_tensor.numel() != len(_OUTPUT_KEYS):  # 输出头数量必须固定。
        raise ValueError(  # 不对就直接失败。
            "model output must contain the fixed four heads ordered as "  # 先说明必须的头序。
            "bias, risk, uwb_scaling, vio_scaling."
        )
    if not torch.isfinite(raw_tensor).all():  # 非有限、（NaN/Inf）直接报错，防止torch 操作静默穿、污染损失（D5 数、安全），与 lstm/trainer.py 口径一致（D7）。
        raise ValueError("model output contains non-finite values (NaN or Inf).")

    return raw_tensor  # 返回整理后的输出向量。




def _coerce_target_outputs(raw_target: Any) -> dict[str, float]:
    """将目标中间结果统一转成标准化的浮点数字典。

    支持 ModelIntermediate 对象、映射（dict 等）和带属、的
    任意对象作为输入。对每个输出头做有限性校验和值域裁剪。
    - bias 不允许为负；
    - risk 裁剪BRIDGE_THRESHOLDS 区间。
    - scaling 不允许低于中性下界1.0。

    参数：
        raw_target: 原始目标中间结果，可以是 ModelIntermediate。
            映射或带 bias/risk/uwb_scaling/vio_scaling 属、的对象。

    返回值：
    包含 "bias"?risk"?uwb_scaling"?vio_scaling" 四个键的
        浮点数字典，所有项都经过校验和裁剪。

    失败条件。
        KeyError: 当映射输入缺少必要字段时抛出。
        TypeError: 当输入类型无法识别时抛出。
        ValueError: 当任何字段为 NaN/Inf ?scaling 低于下限时抛出。
    """

    if isinstance(raw_target, ModelIntermediate):  # 如果已经是统一中间结果对象，就直接取字段。
        source = {  # 先把中间结果转成普、映射，后面统一处理。
            "bias": raw_target.bias,  # 偏置值。
            "risk": raw_target.risk,  # 风险值。
            "uwb_scaling": raw_target.uwb_scaling,  # UWB 缩放值。
            "vio_scaling": raw_target.vio_scaling,  # VIO 缩放值。
        }  # 中间结果字段提取结束。
    elif isinstance(raw_target, Mapping):  # 如果本来就是映射，就直接读取。
        missing_keys = [key for key in _OUTPUT_KEYS if key not in raw_target]  # 先找出缺失字段。
        if missing_keys:  # 少字段就不继续往下走。
            raise KeyError(f"target_intermediate is missing required keys: {missing_keys}")  # 直接报错提醒。
        source = raw_target  # 直接复用原映射。
    else:  # 既不是中间对象，也不是映射时，尝试按属、读取。
        source = {}  # 先准备一个空字典承接属性。
        for key in _OUTPUT_KEYS:  # 逐个输出头去对象上取值。
            value = getattr(raw_target, key, None)  # 如果没有对应属、就返回 None。
            if value is None:  # 缺任何一个都不能继续。
                raise TypeError(  # 不符合输入契约就直接报错。
                    "target_intermediate must be a mapping, ModelIntermediate, or object exposing "  # 说明允许的输入形式。
                    f"{_OUTPUT_KEYS}; got {type(raw_target).__name__}"  # 报告实际类型。
                )  # 类型错误结束。
            source[key] = value  # 把属性、收进统一字典。


    bias = float(source["bias"])  # 偏置值转成浮点数。
    risk = float(source["risk"])  # 风险值转成浮点数。
    uwb_scaling = float(source["uwb_scaling"])  # UWB 缩放值转成浮点数。
    vio_scaling = float(source["vio_scaling"])  # VIO 缩放值转成浮点数。
    for name, value in (  # 逐个检查四个输出头的有限。
        ("bias", bias),  # 偏置校正量。
        ("risk", risk),  # 风险。。
        ("uwb_scaling", uwb_scaling),  # UWB 缩放校正量。
        ("vio_scaling", vio_scaling),  # VIO 缩放校正量。
    ):  # 閫愰」妫€鏌ョ粨条熴€。
        coerce_finite_scalar(value, name=f"target_intermediate.{name}")  # D2：任何一个都不能NaN 或无穷大，有限、校验统一走中央入口。
    bias = max(0.0, bias)  # 偏置不允许为负。
    risk = min(BRIDGE_THRESHOLDS["risk_max"], max(BRIDGE_THRESHOLDS["risk_min"], risk))  # 风险裁剪到训练阈值区间。
    if uwb_scaling < _SCALING_NEUTRAL_FLOOR:  # 缩放值不能低于中性下限。
        raise ValueError("target_intermediate.uwb_scaling must be >= 1.0")  # 低于下限就报错。
    if vio_scaling < _SCALING_NEUTRAL_FLOOR:  # 缩放值不能低于中性下限。
        raise ValueError("target_intermediate.vio_scaling must be >= 1.0")  # 低于下限就报错。
    return {  # 返回统一整理后的目标字典。
        "bias": bias,  # 偏置校正量。
        "risk": risk,  # 风险。。
        "uwb_scaling": uwb_scaling,  # UWB 缩放校正量。
        "vio_scaling": vio_scaling,  # VIO 缩放校正量。
    }  # 目标字典结束。




def _project_train_scaling(raw_scaling: torch.Tensor) -> torch.Tensor:
    """将原始缩放输出投影到训练区间，保证不低于协议中、下限。

    委托给规范实``neutral_floor_softplus``，并显式传入
    ``BRIDGE_THRESHOLDS["scaling_min"]`` 作为下限单源真相，与
    ``inference.py::_clamp_protocol_scaling`` 口径一致（D7 公平性/ D9 单源）。
    上界由规范实现默认从 ``BRIDGE_THRESHOLDS["scaling_max"]`` 读取。
    straight-through 梯度估计clamp 路径均封装在规范实现内，本函数不再本地实现。

    参数：
        raw_scaling: 原始缩放输出标量张量。

    返回值：
        投影后的缩放标量张量，、落。
        ``[BRIDGE_THRESHOLDS["scaling_min"], BRIDGE_THRESHOLDS["scaling_max"]]``。。
    """
    # D7/D9：下限显式取BRIDGE_THRESHOLDS["scaling_min"]，禁止本地字面量与协议单源真相漂移。
    return neutral_floor_softplus(
        raw_scaling,
        neutral_floor=float(BRIDGE_THRESHOLDS["scaling_min"]),
    )


def _normalize_liquid_train_outputs(
    raw_outputs: Mapping[str, Any],
    *,
    risk_already_normalized: bool = False,
) -> dict[str, torch.Tensor]:
    """将模型原始输出字典标准化为训练、输出字典。

    ?bias 直接转张量，risk 调用 _project_train_risk 做区间映射，
    ?scaling 调用 _project_train_scaling 做下限保护。

    参数：
        raw_outputs: 模型原始输出字典，必须包bias、risk。
            uwb_scaling、vio_scaling 四个。
        risk_already_normalized: risk 是否已经归一化到 [0, 1]。

    返回值：
        包含四个标量张量的字典，值已投影到训练区间。
    """
    # Deferred import for circular dependency with _loss_compute
    from ._loss_compute import _project_train_risk
    missing_keys = [key for key in _OUTPUT_KEYS if key not in raw_outputs]
    if missing_keys:  # 缺键是数据合同问题，应抛 ValueError 而非 KeyError，与 inference.py _normalize_intermediate_outputs 口径一致。
        raise ValueError(f"raw_outputs is missing required keys: {missing_keys}")

    bias_raw = torch.as_tensor(raw_outputs["bias"], dtype=torch.float32).reshape(())  # bias 原始值转标量张量。

    risk_raw = torch.as_tensor(raw_outputs["risk"], dtype=torch.float32).reshape(())  # risk 原始值转标量张量。

    uwb_scaling_raw = torch.as_tensor(raw_outputs["uwb_scaling"], dtype=torch.float32).reshape(())  # UWB 缩放原始值转标量张量。

    vio_scaling_raw = torch.as_tensor(raw_outputs["vio_scaling"], dtype=torch.float32).reshape(())  # VIO 缩放原始值转标量张量。

    # D5 数、安全：NaN/Inf 会经 torch.clamp/softplus 静默穿、（clamp(NaN)=NaN、F.softplus(NaN)=NaN），污染batch 梯度。
    # 训练态需保留梯度路径，故torch.isfinite 守护而非 coerce_finite_scalar（后者返回Python float ?detach 梯度）；
    # ?LSTM ?_project_train_outputs（lstm/trainer.py L222）及本文_project_train_risk（L248）口径一致（D7 公平性）。
    if not (torch.isfinite(bias_raw) and torch.isfinite(risk_raw) and torch.isfinite(uwb_scaling_raw) and torch.isfinite(vio_scaling_raw)):
        raise ValueError("raw_outputs contain non-finite values (NaN or Inf)")

    bias = torch.clamp(bias_raw, min=0.0, max=BRIDGE_BIAS_MAX)  # bias 非负且上限与推理侧对齐（L1 根因修复：训推理 clamp 口径一致，对齐 docs/loss_function.md §核心约束 3）。

    uwb_scaling = _project_train_scaling(uwb_scaling_raw)  # UWB 缩放投影到训练区间。

    vio_scaling = _project_train_scaling(vio_scaling_raw)  # VIO 缩放投影到训练区间。

    return {

        "bias": bias,  # bias 偏置校正量。

        "risk": _project_train_risk(risk_raw, already_normalized=risk_already_normalized),  # risk 投影到训练区间。

        "uwb_scaling": uwb_scaling,  # UWB 缩放校正量。

        "vio_scaling": vio_scaling,  # VIO 缩放校正量。

    }


def _project_train_risk_batch(raw_risk: torch.Tensor, *, already_normalized: bool = False) -> torch.Tensor:
    """将批量原始风险输出投影到训练区间（批量版本）。

    ?_project_train_risk 逻辑一致，但接（B,) 张量而非标量。

    参数：
        raw_risk: 原始风险输出张量，形状(B,)。
        already_normalized: 是否已经归一化到 [0, 1]。

    返回值：
        映射到训练区间的风险张量，形状(B,)。
    """
    risk_span = BRIDGE_THRESHOLDS["risk_max"] - BRIDGE_THRESHOLDS["risk_min"]
    risk = torch.as_tensor(raw_risk, dtype=torch.float32)  # 保持 (B,) 形状。
    if not torch.isfinite(risk).all():  # 与标量版一致：NaN/Inf 防护。
        raise ValueError(f"raw_risk must be finite, got {risk}")
    if not already_normalized:
        risk = torch.sigmoid(risk)
    if risk_span != 1.0 or BRIDGE_THRESHOLDS["risk_min"] != 0.0:
        risk = risk * risk_span + BRIDGE_THRESHOLDS["risk_min"]
    # 与推理侧保持一致：末尾 clamp 到协议区间。
    risk = torch.clamp(risk, BRIDGE_THRESHOLDS["risk_min"], BRIDGE_THRESHOLDS["risk_max"])
    return risk


def _project_train_scaling_batch(raw_scaling: torch.Tensor) -> torch.Tensor:
    """将批量原始缩放输出投影到训练区间（批量版本）。

    ?_project_train_scaling 逻辑一致，但接（B,) 张量而非标量。
    委托给规范实``neutral_floor_softplus``，并显式传入
    ``BRIDGE_THRESHOLDS["scaling_min"]`` 作为下限单源真相，与
    ``inference.py::_clamp_protocol_scaling`` 口径一致（D7 公平性/ D9 单源）。
    上界由规范实现默认从 ``BRIDGE_THRESHOLDS["scaling_max"]`` 读取。
    NaN/Inf 拦截由规范实现统一负责（D5），避免批量/标量口径漂移。

    参数：
        raw_scaling: 原始缩放输出张量，形状(B,)。

    返回值：
        投影后的缩放张量，形状(B,)，、落。
        ``[BRIDGE_THRESHOLDS["scaling_min"], BRIDGE_THRESHOLDS["scaling_max"]]``。。
    """
    # D7/D9：下限显式取BRIDGE_THRESHOLDS["scaling_min"]，与标量版口径一致，禁止本地字面量与协议单源真相漂移。
    return neutral_floor_softplus(
        raw_scaling,
        neutral_floor=float(BRIDGE_THRESHOLDS["scaling_min"]),
    )


def _normalize_liquid_train_outputs_batch(
    raw_batch_outputs: Mapping[str, torch.Tensor],
    *,
    risk_already_normalized: bool = False,
) -> dict[str, torch.Tensor]:
    """将批量模型原始输出字典标准化为训练、输出字典。

    批量版本_normalize_liquid_train_outputs，接（B,) 形状的张量。

    参数：
        raw_batch_outputs: 批量原始输出字典，每个、是 (B,) 张量。
        risk_already_normalized: risk 是否已经归一化到 [0, 1]。

    返回值：
        包含四个 (B,) 张量的字典，值已投影到训练区间。
    """
    missing_keys = [key for key in _OUTPUT_KEYS if key not in raw_batch_outputs] # D9：键名走 _OUTPUT_KEYS?MODEL_INTERMEDIATE_KEYS）单源真相，禁止字面量漂移，与标量版口径一致。
    if missing_keys:  # 缺键是数据合同问题，应抛 ValueError 而非 KeyError，与 inference.py _normalize_intermediate_outputs 口径一致。
        raise ValueError(f"raw_batch_outputs is missing required keys: {missing_keys}")

    bias_raw = torch.as_tensor(raw_batch_outputs["bias"], dtype=torch.float32)
    risk_raw = torch.as_tensor(raw_batch_outputs["risk"], dtype=torch.float32)
    uwb_scaling_raw = torch.as_tensor(raw_batch_outputs["uwb_scaling"], dtype=torch.float32)
    vio_scaling_raw = torch.as_tensor(raw_batch_outputs["vio_scaling"], dtype=torch.float32)

    # D5 数、安全：NaN/Inf 会经 torch.clamp/softplus 静默穿、（clamp(NaN)=NaN、F.softplus(NaN)=NaN），污染batch 梯度。
    # 训练态需保留梯度路径，故torch.isfinite 守护而非 coerce_finite_scalar（后者返回Python float ?detach 梯度）；
    # 与标量版 _normalize_liquid_train_outputs 及本文件 _project_train_risk_batch 口径一致（D7 公平性）。
    if not (torch.isfinite(bias_raw).all() and torch.isfinite(risk_raw).all() and torch.isfinite(uwb_scaling_raw).all() and torch.isfinite(vio_scaling_raw).all()):
        raise ValueError("raw_batch_outputs contain non-finite values (NaN or Inf)")

    bias = torch.clamp(bias_raw, min=0.0, max=BRIDGE_BIAS_MAX)  # bias 非负且上限与推理侧对齐（L1 根因修复：训推理 clamp 口径一致，对齐 docs/loss_function.md §核心约束 3）。
    uwb_scaling = _project_train_scaling_batch(uwb_scaling_raw)  # UWB 缩放投影到训练区间。
    vio_scaling = _project_train_scaling_batch(vio_scaling_raw)  # VIO 缩放投影到训练区间。
    return {
        "bias": bias,
        "risk": _project_train_risk_batch(risk_raw, already_normalized=risk_already_normalized),
        "uwb_scaling": uwb_scaling,
        "vio_scaling": vio_scaling,
    }





