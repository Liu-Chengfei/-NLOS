"""LSTM 中间量推理实现。

这个模块只负责把结构化窗口送进 LSTM，然后把网络的四个固定输出头整理成
上层协议要求的 `ModelIntermediate`。它不负责训练，也不负责数据准备，所以
职责边界很窄，适合被推理管线和测试直接复用。

上游输入通常来自 `normalize_structured_window` 兼容的窗口映射，或者来自
已经准备好的推理管线中间结果；下游输出会被 `fusion`、`pipelines` 和评估
代码继续消费。这个文件的关键点是保持输出顺序稳定，并且把风险值和缩放值
映射到协议允许的范围内。
"""

from __future__ import annotations  # 允许在类型标注里安全引用尚未定义的类型名。

from collections.abc import Mapping  # 用于判断映射型输入。

import numpy as np  # 这里只用 numpy 做数值变换和有限性检查。
import math  # 用于有限性检查，与 numpy 互为补充。
import torch  # 用于保持 risk_calibration 与模型输出在同一设备上执行。

from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS  # 读取桥接层业务阈值范围，保证输出符合公共协议。
from liquidloc.protocol.risk_projection import project_risk_to_protocol_range  # §12.3-C2a 第十四轮：risk 投影权威单源，禁止 LSTM 路径本地手抄公式副本（与 Liquid 路径一致委托）。
from liquidloc.common.constants import MODALITY_UWB, MODALITY_VIO  # 模态名单源真相，避免字面量漂移。
from liquidloc.common.constants import BRIDGE_BIAS_MAX  # 模型输出偏置上界，与 ModelIntermediate 验证范围对齐。
from liquidloc.common.types import ModelIntermediate  # 使用统一的中间量数据结构返回结果。
from liquidloc.models.features.normalization import neutral_floor_softplus  # 统一的缩放因子 softplus 变换。
from liquidloc.models.lstm.network import normalize_structured_window  # 复用同一套结构化窗口校验逻辑。


# 下面这个入口函数负责把模型输出整理成上层可直接消费的中间量对象。
def infer_intermediate(  # 对结构化窗口执行一次推理并整理输出。
    window_tensor,  # 结构化窗口输入。
    model,  # 提供 forward 方法的模型对象。
):  # 返回四个固定中间量。
    """对一个结构化窗口执行一次 LSTM 推理，并整理成中间量对象。

    参数:
    `window_tensor` 是结构化窗口，可以是映射，也可以是网络已经能接受的
    结构化输入对象；`model` 必须提供可调用的 `forward` 方法。

    返回值:
    返回 `ModelIntermediate`，其中包含 `bias`、`risk`、`uwb_scaling` 和
    `vio_scaling` 四个按协议固定顺序排列的中间量。

    失败条件:
    如果模型没有 `forward`，或者网络输出不是四个有限数值，就会抛出异常。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "model_type": type(model).__name__,
        "window_tensor_type": type(window_tensor).__name__,
        "has_predict_intermediate_tensors": callable(getattr(model, "predict_intermediate_tensors", None)),
    }, "lstm.infer_intermediate")
    if model is None:  # 模型对象不能为空。
        raise TypeError("model must provide a callable forward method.")  # 保持原有错误类型。

    predict_intermediate_tensors = getattr(model, "predict_intermediate_tensors", None)
    if callable(predict_intermediate_tensors):  # 优先兼容工厂返回的真实模型接口。
        with torch.no_grad():
            normalized_outputs = predict_intermediate_tensors(window_tensor)
            missing_keys = [key for key in ("bias", "risk", "uwb_scaling", "vio_scaling") if key not in normalized_outputs]
            if missing_keys:
                raise ValueError(f"predict_intermediate_tensors must return keys bias, risk, uwb_scaling, vio_scaling; missing: {missing_keys}")
            for key in ("bias", "risk", "uwb_scaling", "vio_scaling"):
                value = float(torch.as_tensor(normalized_outputs[key]).detach().reshape(()).cpu().item())
                if not math.isfinite(value):
                    raise ValueError(f"predict_intermediate_tensors must return finite values; {key}={value}")
        return ModelIntermediate(
            bias=round(float(torch.as_tensor(normalized_outputs["bias"]).detach().reshape(()).cpu().item()), 6),
            risk=round(float(torch.as_tensor(normalized_outputs["risk"]).detach().reshape(()).cpu().item()), 6),
            uwb_scaling=float(torch.as_tensor(normalized_outputs["uwb_scaling"]).detach().reshape(()).cpu().item()),
            vio_scaling=float(torch.as_tensor(normalized_outputs["vio_scaling"]).detach().reshape(()).cpu().item()),
        )

    if not callable(getattr(model, "forward", None)):  # 再兼容只有 forward 的裸网络对象。
        raise TypeError("model must provide a callable forward method.")  # 两条真实入口都没有时再报错。

    window_tensor = normalize_structured_window(window_tensor)  # 先把输入统一成网络能接受的结构化窗口。

    with torch.no_grad():  # 推理路径不需要梯度，避免构建计算图浪费内存。
        raw_output = model.forward(window_tensor)  # 调用模型拿到原始四头输出。
    if hasattr(raw_output, "detach"):  # 如果输出还挂着计算图，就先切断，避免后续误用训练图。
        raw_output = raw_output.detach()  # 切断梯度追踪，只保留数值。
    raw_output_tensor = torch.as_tensor(raw_output, dtype=torch.float32).reshape(-1)  # 保持在原设备上整理，避免校准模块设备漂移。
    if raw_output_tensor.numel() != 4:  # 协议要求必须恰好四个输出头。
        raise ValueError(  # 输出头数量不对时，直接提示上游模型契约不匹配。
            "model.forward(window_tensor) must return four outputs ordered as "
            "bias, risk, uwb_scaling, vio_scaling."
        )
    if not torch.isfinite(raw_output_tensor).all().item():  # 任何一个值是 nan 或 inf 都不能继续往下传。
        raise ValueError("model.forward(window_tensor) must return finite values.")  # 非有限数值直接拒绝。

    raw_output_cpu = raw_output_tensor.cpu().numpy()  # 仅在数值后处理前搬回 CPU，避免影响设备上的校准模块。
    bias = min(float(BRIDGE_BIAS_MAX), max(0.0, float(raw_output_cpu[0])))  # 偏置非负且上限与 ModelIntermediate 验证范围对齐。
    bias = round(bias, 6)  # round 精度与 fast path 对齐。
    raw_risk_tensor = raw_output_tensor[1].reshape(())  # 第二个输出头是原始 risk logit，保留原设备给校准模块。
    # 如果模型有 risk_calibration 模块，必须经过校准（与训练路径和工厂推理路径一致）。
    if hasattr(model, 'risk_calibration') and callable(getattr(model, 'risk_calibration', None)):
        with torch.no_grad():
            calibrated_risk = model.risk_calibration(raw_risk_tensor)
        risk = float(calibrated_risk.detach().cpu().item())
    else:
        risk = float(1.0 / (1.0 + np.exp(-float(raw_risk_tensor.detach().cpu().item()))))  # 无校准模块时用简单 sigmoid。
    # §12.3-C2a 第十四轮：risk 协议区间投影委托 protocol 单源 project_risk_to_protocol_range，
    # 不再本地手抄 risk_span = risk_max - risk_min / if risk_span != 1.0 / risk = risk*span+min / min/max clamp。
    # 委托方能保证 D5 数值安全（math.isfinite 显式拒绝 NaN/Inf，避免静默穿透）与 Liquid 路径行为等价。
    # 历史公式（与 protocol 单源字字一致，但少 D5 isfinite 守卫）已废弃：
    #   risk_span = BRIDGE_THRESHOLDS["risk_max"] - BRIDGE_THRESHOLDS["risk_min"]
    #   if risk_span != 1.0 or BRIDGE_THRESHOLDS["risk_min"] != 0.0:
    #       risk = float(risk * risk_span + BRIDGE_THRESHOLDS["risk_min"])
    #   risk = min(float(BRIDGE_THRESHOLDS["risk_max"]), max(float(BRIDGE_THRESHOLDS["risk_min"]), risk))
    risk = project_risk_to_protocol_range(
        risk,
        BRIDGE_THRESHOLDS["risk_min"],
        BRIDGE_THRESHOLDS["risk_max"],
    )
    risk = round(risk, 6)  # round 精度与 fast path 对齐。
    uwb_scaling = neutral_floor_softplus(float(raw_output_cpu[2]))  # 第三个输出转成 UWB 正缩放因子。
    vio_scaling = neutral_floor_softplus(float(raw_output_cpu[3]))  # 第四个输出转成 VIO 正缩放因子。

    # 模态输出合约：非当前模态的 scaling 强制为 1.0，与 Liquid 推理器保持一致。
    modality = window_tensor.get("current_modality", "") if isinstance(window_tensor, Mapping) else ""
    if modality == MODALITY_UWB:
        vio_scaling = 1.0
    elif modality == MODALITY_VIO:
        uwb_scaling = 1.0

    return ModelIntermediate(  # 用统一的数据对象返回，方便上层按字段名消费。
        bias=bias,  # 原始偏置值，供上层做协议解释。
        risk=risk,  # 已压到协议区间内的风险分数。
        uwb_scaling=uwb_scaling,  # UWB 通道的正缩放因子。
        vio_scaling=vio_scaling,  # VIO 通道的正缩放因子。
    )  # 这里结束中间量对象构造。
