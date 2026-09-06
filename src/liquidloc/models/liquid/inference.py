"""Liquid 推理辅助模块。

【文件职责】
把 Liquid 网络的共享特征输出映射到四个固定中间量头（bias、risk、
uwb_scaling、vio_scaling），并整理成上层协议要求的 ModelIntermediate。

【本文件绝对不负责】
不负责训练、数据准备和模型结构定义。

【上游依赖】
models/liquid/network.py、common/types.py、common/constants.py。

【下游调用者】
models/liquid/__init__.py、pipelines/inference_pipeline.py、
tests/models/test_liquid_inference.py。

【输入对象定义】
- window_tensor：结构化窗口输入。
- model_state：模型状态对象，必须暴露 network 和 output_heads。

【输出对象定义】
- ModelIntermediate：包含 bias、risk、uwb_scaling、vio_scaling 四个中间量。

【核心变量定义】
- _OUTPUT_KEYS：固定输出头顺序。
- scaling 下限使用 BRIDGE_THRESHOLDS["scaling_min"]（D9 配置表面漂移：禁止本地常量漂移）。
"""

from __future__ import annotations  # 允许在类型注解里引用尚未定义的类型名。

import numpy as np  # 用于数值安全的 sigmoid 计算（np.exp 溢出返回 inf 而非崩溃）。
import torch  # 用于推理路径 torch.no_grad()，与 lstm/inference.py 和 _LiquidModelWrapper 对齐（D4 设备/D7 公平性）。
from collections.abc import Mapping  # 用于判断映射型输入。
from liquidloc.common.validation import coerce_finite_scalar, is_string_like, require_not_none  # 统一标量校验入口（D2 分层边界，禁止本地重写 coerce_finite_scalar）、字符串类型判断和参数非空校验。
from typing import Any  # 用于承接不确定类型的输入和值。

from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS  # 读取桥接层业务阈值范围，保证输出符合公共协议。
from liquidloc.protocol.risk_projection import project_risk_to_protocol_range  # 风险映射公式协议层权威实现，禁止本地重实现（D2 分层边界）。
from liquidloc.common.constants import BRIDGE_BIAS_MAX, MODEL_INTERMEDIATE_KEYS, MODALITY_UWB, MODALITY_VIO, SHARED_FEATURES_KEY  # 模型输出偏置上界，与 ModelIntermediate 验证范围对齐；模态名单源真相，避免字面量漂移；输出头键名单源真相（D9 根因修复）；shared_features 包装键名单源真相（D9 根因修复）。
from liquidloc.common.types import ModelIntermediate  # 使用统一的中间量数据结构返回结果。
from liquidloc.models.features.normalization import neutral_floor_softplus  # 统一的缩放因子 softplus 变换。

_OUTPUT_KEYS = MODEL_INTERMEDIATE_KEYS  # D9：引用单源真相常量，禁止本地重复定义；保留 _OUTPUT_KEYS 别名供本模块内部使用。
_VALID_MODALITIES = frozenset({MODALITY_UWB, MODALITY_VIO})  # 推理路径只接受训练器认识的两种模态，引用单源真相避免漂移，与 network._VALID_MODALITIES 对齐。


def _lookup(source: Any, name: str) -> Any:
    """从映射或对象中按名字查找值。

    参数:
    `source` 是映射或普通对象。
    `name` 是要查找的键名或属性名。

    返回值:
    找到时返回对应值，否则返回 None。
    """
    if isinstance(source, Mapping):  # 如果是映射，用 get 查找。
        return source.get(name)  # 映射查找，缺失返回 None。
    return getattr(source, name, None)  # 否则用属性查找，缺失返回 None。


def _project_risk_to_protocol_range(risk: float) -> float:
    """将风险值线性重映射到协议区间 [risk_min, risk_max]（本地薄包装）。

    本函数仅负责把 BRIDGE_THRESHOLDS 的 risk_min/risk_max 绑定到协议层
    project_risk_to_protocol_range，公式实现统一在 protocol 层，禁止重实现
    （D2 分层边界）。数值安全检查（float 转换、isfinite 拒绝 NaN/Inf）与
    线性映射 + clamp 全部委托 protocol 层权威实现，本地不再持有公式副本。

    与训练侧 _project_train_risk 在公式上一致；训练侧因携带 sigmoid 预归一化
    与 torch.clamp 张量语义未直接委托，详见 trainer.py docstring。

    参数:
    `risk` 是已归一化到 [0, 1] 的风险值。

    返回值:
    映射到 [risk_min, risk_max] 区间的风险值。
    """
    return project_risk_to_protocol_range(
        risk, BRIDGE_THRESHOLDS["risk_min"], BRIDGE_THRESHOLDS["risk_max"]
    )


def _clamp_protocol_scaling(value: float) -> float:
    """把已处于协议语义的 scaling 约束回合法区间，而不是再次做 softplus。

    D2 分层边界：使用中央 coerce_finite_scalar 做有限性校验，禁止本地手写
    math.isfinite 检查（与 _coerce_scalar 删除后的统一口径一致）。
    D5 数值安全：coerce_finite_scalar 拦截 NaN/Inf，防止经 min/max 静默穿透
    （Python 中 max(1.0, nan) 返回 nan，min(max_val, nan) 也返回 nan）。
    D9 配置表面漂移：下限使用 BRIDGE_THRESHOLDS["scaling_min"]，
    禁止本地 _SCALING_NEUTRAL_FLOOR 常量与协议单源真相漂移。
    """
    scalar = coerce_finite_scalar(value, name="scaling")  # D2/D5：使用中央入口做有限性校验。
    scaling_min = float(BRIDGE_THRESHOLDS["scaling_min"])  # 缩放下限走协议单源真相，禁止本地常量漂移。
    scaling_max = float(BRIDGE_THRESHOLDS["scaling_max"])
    return min(scaling_max, max(scaling_min, scalar))


def _contains_all_output_keys(candidate: Any) -> bool:
    """检查候选对象是否包含全部四个输出头。

    参数:
    `candidate` 是待检查的对象，可以是 ModelIntermediate、映射或普通对象。

    返回值:
    包含全部四个输出头时返回 True，否则返回 False。
    """
    if isinstance(candidate, ModelIntermediate):  # ModelIntermediate 天然包含四头。
        return True  # 直接返回 True。
    if isinstance(candidate, Mapping):  # 映射型检查所有键是否存在。
        return all(key in candidate for key in _OUTPUT_KEYS)  # 逐个检查键是否存在。
    return all(hasattr(candidate, key) for key in _OUTPUT_KEYS)  # 对象型检查所有属性是否存在。


def _normalize_intermediate_outputs(raw_outputs: Any) -> dict[str, float]:
    """把各种形式的原始输出统一转成标准化的中间量字典。

    参数:
    `raw_outputs` 可以是 ModelIntermediate、映射或暴露四头属性的对象。

    返回值:
    返回包含 bias、risk、uwb_scaling、vio_scaling 四个键的字典。
    risk 经过阈值裁剪，scaling 经过 neutral_floor_softplus。

    失败条件:
    缺少必需键或值不是有限数时抛出异常。
    """
    if isinstance(raw_outputs, ModelIntermediate):  # 防御性路径：输入已过 ModelIntermediate.__post_init__ 协议校验。
        raw_bias = coerce_finite_scalar(raw_outputs.bias, name="bias")  # D2：使用中央 coerce_finite_scalar，禁止本地重写。
        raw_risk = coerce_finite_scalar(raw_outputs.risk, name="risk")  # 风险值应已处于协议语义，仅做防御性 clamp。
        raw_uwb_scaling = coerce_finite_scalar(raw_outputs.uwb_scaling, name="uwb_scaling")
        raw_vio_scaling = coerce_finite_scalar(raw_outputs.vio_scaling, name="vio_scaling")
        raw_bias = min(float(BRIDGE_BIAS_MAX), max(0.0, raw_bias))  # 偏置非负且上限与 ModelIntermediate 验证范围对齐，与 Path B/C 写法统一。
        raw_risk = min(float(BRIDGE_THRESHOLDS["risk_max"]), max(float(BRIDGE_THRESHOLDS["risk_min"]), raw_risk))
        return {
            "bias": round(raw_bias, 6),  # D7 公平性：与 lstm/inference.py L88、model_factory.py L979 对齐 round(bias, 6)。
            "risk": round(raw_risk, 6),  # D7 公平性：与 lstm/inference.py L101、model_factory.py L980 对齐 round(risk, 6)。
            "uwb_scaling": _clamp_protocol_scaling(raw_uwb_scaling),  # 已归一化 scaling 不再二次 softplus。
            "vio_scaling": _clamp_protocol_scaling(raw_vio_scaling),
        }
    elif isinstance(raw_outputs, Mapping):  # 如果本来就是映射。
        missing_keys = [key for key in _OUTPUT_KEYS if key not in raw_outputs]  # 先找出缺失字段。
        if missing_keys:  # 少字段就不继续往下走。
            raise ValueError(f"raw_outputs is missing required keys: {missing_keys}")  # 缺失键是值合同问题，应抛 ValueError 而非 KeyError。
        source = raw_outputs  # 直接复用原映射。
    elif _contains_all_output_keys(raw_outputs):  # 如果对象暴露了全部四头属性。
        source = {key: getattr(raw_outputs, key) for key in _OUTPUT_KEYS}  # 按属性名提取。
    else:  # 都不满足时直接报错。
        raise TypeError(
            "raw_outputs must be a mapping, ModelIntermediate, or object exposing "
            f"{_OUTPUT_KEYS}, got {type(raw_outputs).__name__}"
        )  # 说明允许的输入形式和实际类型。

    raw_bias = coerce_finite_scalar(source["bias"], name="bias")  # 偏置值转成有限浮点数（D2：使用中央 coerce_finite_scalar）。
    raw_bias = min(float(BRIDGE_BIAS_MAX), max(0.0, raw_bias))  # UWB 距离偏置物理上不可能为负且上限与 ModelIntermediate 验证范围对齐。
    risk = coerce_finite_scalar(source["risk"], name="risk")  # 风险值转成有限浮点数。
    uwb_scaling = coerce_finite_scalar(source["uwb_scaling"], name="uwb_scaling")  # UWB 缩放值转成有限浮点数。
    vio_scaling = coerce_finite_scalar(source["vio_scaling"], name="vio_scaling")  # VIO 缩放值转成有限浮点数。
    return {  # 返回标准化后的中间量字典。
        "bias": round(raw_bias, 6),  # D7 公平性：与 lstm/inference.py L88、model_factory.py L979 对齐 round(bias, 6)。
        "risk": round(_project_risk_to_protocol_range(risk), 6),  # D7 公平性：与 lstm/inference.py L101、model_factory.py L980 对齐 round(risk, 6)。
        "uwb_scaling": neutral_floor_softplus(uwb_scaling),  # UWB 缩放转成安全正缩放因子。
        "vio_scaling": neutral_floor_softplus(vio_scaling),  # VIO 缩放转成安全正缩放因子。
    }


def _unwrap_shared_features(network_output: Any) -> Any:
    """从网络输出中提取共享特征字典。

    参数:
    `network_output` 是网络前向输出，可能是包装了 shared_features 的字典或对象。

    返回值:
    返回共享特征字典，必须为 Mapping 且可被下游 output head 消费。

    失败条件:
    `network_output` 为 None、包装的 `shared_features` 值为 None、
    或既不暴露 `shared_features` 字段也不是合法共享特征映射时抛出 TypeError。
    """
    if network_output is None:  # 显式拒绝 None，避免静默走 fallback 掩盖上游契约违规。
        raise TypeError("network_output must not be None when extracting shared_features")
    if isinstance(network_output, Mapping) and SHARED_FEATURES_KEY in network_output:  # 优先识别 forward() 包装格式，键名引用单源真相常量（D9 根因修复）。
        shared_features = network_output[SHARED_FEATURES_KEY]  # 取出包装的共享特征。
        if shared_features is None:  # 显式 None 说明上游契约违规，不能当作特征返回。
            raise TypeError("network_output['shared_features'] is None; network must return a non-null mapping")
        return shared_features  # 返回包装的共享特征。
    if hasattr(network_output, SHARED_FEATURES_KEY):  # 用 hasattr 区分"属性不存在"和"属性为 None"，避免把显式 None 当特征返回。
        shared_features = getattr(network_output, SHARED_FEATURES_KEY, None)  # 取出属性值。
        if shared_features is None:  # 属性存在但显式为 None，说明上游契约违规。
            raise TypeError("network_output.shared_features is None; network must return a non-null mapping")
        return shared_features  # 返回属性值。
    # 既不是 forward() 包装，也不暴露 shared_features 属性，则假定输出本身就是共享特征。
    # 此时必须是 Mapping 才能被下游 output head 消费；否则会在下游以不清晰的 TypeError 失败。
    if not isinstance(network_output, Mapping):  # 校验 fallback 路径返回值类型。
        raise TypeError(
            "network_output must be a mapping with 'shared_features', an object exposing "
            f"'shared_features', or a shared_features mapping itself; got {type(network_output).__name__}"
        )
    return network_output  # 返回本身就是特征的映射。


def _resolve_modality_from_window(window_tensor: Any, *, fallback: Any | None = None) -> str:
    """从共享特征或结构化窗口中提取并校验当前模态名称。

    Liquid 训练器只接受 "uwb" / "vio" 两种模态；推理侧若静默回落到默认值，
    会把非当前模态的 scaling 合同错误地改写成中性 1.0，因此这里必须硬校验。

    归一化口径与 trainer._resolve_current_modality 和 network._coerce_supported_modality
    完全对齐：先 is_string_like 严格类型校验，再 strip().lower()，最后白名单校验，
    避免非字符串类型被 str() 静默转换掩盖数据合同问题。只读 current_modality 字段
    （窗口级真相），不回退到 modality 字段（事件级，见 PRIMARY_EVENT_KEYS），
    避免跨层语义混淆与公平性分歧。fallback 触发统一用 is None 判定，避免 or 短路
    在空串/0/False 等 falsy 值上误回退到事件级字段。
    """
    modality = _lookup(window_tensor, "current_modality")  # 复用 common 层 _lookup helper，避免重复实现 Mapping/属性双路径查找。
    if modality is None and fallback is not None:  # 仅当主源未提供时回退到次源，用 is None 而非真值判断，避免空串被误判为缺失。
        modality = _lookup(fallback, "current_modality")
    if not is_string_like(modality):  # 严格类型校验，拒绝 None/数值/布尔/列表等被 str() 静默转换，与 trainer/network 对齐。
        raise ValueError(
            "feature_window.current_modality must be one of "
            f"{sorted(_VALID_MODALITIES)} for liquid inference"
        )
    modality_name = str(modality).strip().lower()  # 与 trainer._resolve_current_modality 和 network._coerce_supported_modality 对齐：统一 strip().lower() 归一化口径。
    if modality_name not in _VALID_MODALITIES:
        raise ValueError(
            "feature_window.current_modality must be one of "
            f"{sorted(_VALID_MODALITIES)} for liquid inference"
        )
    return modality_name


def _extract_shared_features(window_tensor: Any, network: Any) -> Any:
    """从网络中提取共享特征。

    参数:
    `window_tensor` 是结构化窗口输入。
    `network` 是网络对象，可以是映射、模块或可调用对象。

    返回值:
    返回共享特征字典。

    失败条件:
    网络不暴露任何可调用的特征提取方法时抛出异常。
    """
    if isinstance(network, Mapping) and SHARED_FEATURES_KEY in network:  # 如果网络是映射且已预计算特征，键名引用单源真相常量（D9 根因修复）。
        precomputed = network[SHARED_FEATURES_KEY]  # 取出预计算特征。
        if precomputed is None:  # 预计算特征不能为 None，否则下游 _project_outputs 调用输出头时会拿到模糊 TypeError 而非根因。
            raise TypeError("precomputed shared_features must not be None")  # 显式报错，避免下游模糊错误。
        return precomputed  # 返回预计算特征。
    for method_name in ("extract_shared_features", "forward_shared", "forward"):  # 按优先级尝试调用。
        method = getattr(network, method_name, None)  # 尝试获取方法。
        if callable(method):  # 如果方法可调用。
            try:  # D10：为 method 调用添加错误上下文，便于定位是哪个方法名失败。
                return _unwrap_shared_features(method(window_tensor))  # 调用并解包共享特征。
            except (TypeError, ValueError, RuntimeError) as exc:  # 保留原异常类型，仅追加方法名上下文。
                raise type(exc)(f"{method_name}(window_tensor) on {type(network).__name__} failed: {exc}") from exc
    if callable(network):  # 如果网络本身就是可调用对象。
        try:  # D10：为 network 调用添加错误上下文。
            return _unwrap_shared_features(network(window_tensor))  # 直接调用并解包。
        except (TypeError, ValueError, RuntimeError) as exc:  # 保留原异常类型，仅追加上下文。
            raise type(exc)(f"network(window_tensor) on {type(network).__name__} failed: {exc}") from exc
    raise TypeError(  # 都不满足时报错。
        "model_state or network must expose a callable shared-feature extractor "
        f"or precomputed shared_features, got {type(network).__name__}"  # 含实际类型，便于调试。
    )  # 说明必须提供特征提取能力。


def _lookup_output_head(source: Any, key: str) -> Any:
    """从模型状态或网络中查找指定输出头。

    参数:
    `source` 是模型状态或网络对象。
    `key` 是输出头名称（如 "bias"）。

    返回值:
    找到时返回输出头对象，否则返回 None。
    """
    head = _lookup(source, f"{key}_head")  # 先尝试直接按 "key_head" 查找。
    if head is not None:  # 找到了就直接返回。
        return head
    head_collection = _lookup(source, "output_heads")  # 仅支持 output_heads 集合名，与 _LiquidModelWrapper 和文档约定一致；不接收 "heads" 等未注册别名，避免字符串路由歧义。
    if isinstance(head_collection, Mapping) and key in head_collection:  # 集合必须是映射且包含该头。
        return head_collection[key]  # 返回对应的输出头。
    return None  # 都没找到时返回 None，由 _project_outputs 收集后统一报错。


def _project_outputs(shared_features: Any, model_state: Any, network: Any) -> Any:
    """把共享特征通过各输出头投影成四头输出。

    参数:
    `shared_features` 是共享特征字典。
    `model_state` 是模型状态对象。
    `network` 是网络对象。

    返回值:
    返回包含四个输出头的映射。

    失败条件:
    共享特征不包含四头且模型不暴露输出头时抛出异常。
    """
    if _contains_all_output_keys(shared_features):  # 如果共享特征已经包含四头输出。
        return shared_features  # 直接返回，不需要再投影。
    owners = (model_state, network) if network is not model_state else (model_state,)  # 去重后确定查找来源。
    # v3.1 head-shared: 若 model_state 暴露 run_head_forward，优先使用共享 backbone + 4 个 final-projection head
    run_head_forward = getattr(model_state, "run_head_forward", None)
    if callable(run_head_forward):
        try:
            return dict(run_head_forward(shared_features))  # 一次走完 backbone + 4 head
        except Exception:  # 防御性兜底：若 backbone 路径异常，回退到逐 head 查找流程
            pass
    projected = {}  # 存投影后的输出。
    missing_heads = []  # 记录缺失的输出头。
    for key in _OUTPUT_KEYS:  # 逐个输出头查找并投影。
        head = None  # 当前输出头对象。
        for owner in owners:  # 从各个来源中查找。
            head = _lookup_output_head(owner, key)  # 尝试查找输出头。
            if head is not None:  # 找到了就停止查找。
                break
        if head is None:  # 没找到输出头。
            missing_heads.append(f"{key}_head")  # 记录缺失。
            continue  # 跳过这个头。
        if not callable(head):  # 输出头必须可调用。
            raise TypeError(f"{key}_head must be callable, got {type(head).__name__}")  # 不可调用就报错。
        projected[key] = head(shared_features)  # 调用输出头投影特征。
    if missing_heads:  # 如果有缺失的输出头。
        raise ValueError(  # 报错说明必须提供全部四头。
            "model_state or network must expose the four output heads or return "
            f"precomputed outputs with keys {_OUTPUT_KEYS}; missing {missing_heads}"
        )
    return projected  # 返回投影后的输出映射。


def _apply_risk_calibration(raw_outputs: Any, model_state: Any) -> Any:
    """对风险输出应用校准（如果有校准模块）或默认 sigmoid。

    参数:
    `raw_outputs` 是原始四头输出映射。
    `model_state` 是模型状态对象，可能包含 risk_calibration 模块。

    返回值:
    返回校准后的输出映射。
    """
    if not isinstance(raw_outputs, Mapping) or "risk" not in raw_outputs:  # 输出不是映射或没有 risk 键。
        return dict(raw_outputs) if isinstance(raw_outputs, Mapping) else raw_outputs  # D2：复制 Mapping 保持与其他分支一致的不变性，非 Mapping 原样返回。
    calibration = _lookup(model_state, "risk_calibration")  # 查找校准模块。
    if not callable(calibration):  # 没有校准模块时，用默认 sigmoid。
        raw_outputs = dict(raw_outputs)  # 先拷贝一份，避免修改原始输出。
        raw_outputs["risk"] = float(1.0 / (1.0 + np.exp(-coerce_finite_scalar(raw_outputs["risk"], name="risk"))))  # 数值安全 sigmoid（D2：使用中央 coerce_finite_scalar；np.exp 溢出返回 inf，1/(1+inf)=0.0）。
        return raw_outputs  # 返回 sigmoid 后的结果。
    raw_outputs = dict(raw_outputs)  # 有校准模块时，先拷贝一份。
    coerce_finite_scalar(raw_outputs["risk"], name="risk")  # D5：仅做有限性校验，不替换原值，保留原设备传给校准模块（与 lstm/inference.py L89 保留 raw_risk_tensor 设备对齐）；避免 RiskCalibration.forward 内部 torch.as_tensor 静默接受 NaN/inf 后产生静默 NaN sigmoid。
    with torch.no_grad():  # D4：推理路径不构建梯度图，与 lstm/inference.py L92-93 对齐；torch 已在文件头 L33 导入；grad-enabled 上下文调用 infer_intermediate 时避免校准模块 forward 累积激活内存。
        try:  # D10：为 calibration 调用添加错误上下文，便于定位校准模块失败原因。
            calibrated = calibration(raw_outputs["risk"])
        except (RuntimeError, ValueError, TypeError) as exc:  # 保留原异常类型，仅追加校准上下文。
            raise type(exc)(f"risk_calibration(risk) failed: {exc}") from exc
    if hasattr(calibrated, "detach"):  # D4/D7：切断计算图引用，与 lstm/inference.py L94 detach().cpu().item() 对齐；no_grad 已保证无 _grad_fn，detach 作为防御性冗余。
        calibrated = calibrated.detach()
    raw_outputs["risk"] = calibrated  # 校准后的值（标量或张量，由下游 coerce_finite_scalar 统一取标量）。
    return raw_outputs  # 返回校准后的结果。


def infer_intermediate(window_tensor: Any, model_state: Any = None) -> ModelIntermediate:
    """对一个结构化窗口执行一次 Liquid 推理，并整理成中间量对象。

    参数:
    `window_tensor` 是结构化窗口输入。
    `model_state` 是模型状态对象，必须暴露 network 和 output_heads。

    返回值:
    返回 `ModelIntermediate`，其中包含 `bias`、`risk`、`uwb_scaling` 和
    `vio_scaling` 四个按协议固定顺序排列的中间量。

    失败条件:
    window_tensor 或 model_state 为 None、网络不暴露特征提取方法、
    输出头缺失或输出值不是有限数时会抛出异常。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "model_state_type": type(model_state).__name__,
        "window_tensor_type": type(window_tensor).__name__,
        "has_network": _lookup(model_state, "network") is not None,
    }, "liquid.infer_intermediate")
    require_not_none(window_tensor, "window_tensor")  # 窗口输入不能为空。
    require_not_none(model_state, "model_state")  # 模型状态不能为空。
    network = _lookup(model_state, "network")  # 从模型状态中查找网络。
    if network is None:  # 如果没有 network 属性。
        network = model_state  # 假定 model_state 本身就是网络。
    with torch.no_grad():  # D4/D7：推理路径不构建计算图，与 lstm/inference.py L73 和 _LiquidModelWrapper.predict_intermediate_tensors（factories/model_factory.py L976/L1342）对齐；head 与 risk_calibration 均为 nn.Module，调用会触发 forward hook 与梯度记录。
        shared_features = _extract_shared_features(window_tensor, network)  # 提取共享特征。
        raw_outputs = _project_outputs(shared_features, model_state, network)  # 通过输出头投影四头输出。
        calibrated_outputs = _apply_risk_calibration(raw_outputs, model_state)  # 应用风险校准。
        normalized = _normalize_intermediate_outputs(calibrated_outputs)  # 标准化输出。
    # 应用模态输出合约：模态必须与网络实际消费的共享特征一致，不能回头读原始坏输入。
    modality = _resolve_modality_from_window(shared_features, fallback=window_tensor)
    if modality == MODALITY_UWB:
        normalized["vio_scaling"] = 1.0  # UWB 模态下 VIO 缩放因子强制为 1.0。
    elif modality == MODALITY_VIO:
        normalized["uwb_scaling"] = 1.0  # VIO 模态下 UWB 缩放因子强制为 1.0。
