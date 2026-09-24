"""UWB 偏置适配器——兼容层，将模型输出的 bias 解释为 UWB 距离修正量。

.. deprecated::
    本模块是兼容包装层。生产链路 ``fusion_runner.run_fusion`` 直接调用
    ``protocol.liquid_bridge_contract.build_measurement_control`` 完成控制量转换，
    不经过本模块。本模块仅保留供历史调用方和测试使用。

本模块的职责：
1. 将模型输出的 ``bias`` 值应用到 UWB 事件的距离测量上，得到修正后的距离
2. bias 截断逻辑委托协议层 ``clip_uwb_bias`` 唯一权威实现，不复制截断规则
3. 返回修正结果和诊断报告，便于上层追踪偏置来源和修正效果

核心概念：
- **偏置修正公式**：``corrected_range = max(0.0, raw_range - bias_value)``
  即从原始距离中减去偏置值，结果不低于 0
- **截断规则**：由 ``protocol.liquid_bridge_contract.clip_uwb_bias`` 统一实现，
  包含非负约束、比例截断（uwb_bias_max_ratio）、绝对值上限（uwb_bias_absolute_max）

分层边界：
- 本模块属于 **fusion 桥接层**，只做"偏置值 → 修正距离"的数值转换
- 不决定偏置值本身（由模型推理产生），也不决定修正后的距离如何被使用（由估计器消费）
- bias 截断规则由协议层唯一权威实现，本模块不复制

与其他模块的关系：
- ``protocol.liquid_bridge_contract.clip_uwb_bias``：bias 截断的唯一权威实现
- ``protocol.bridge_thresholds``：提供偏置截断阈值
- ``protocol.event_schema``：提供事件校验能力
- ``fusion_runner``：生产链路直接调用协议层，不经过本模块
"""

from __future__ import annotations  # 延迟解析注解，减少导入阶段的类型依赖问题。

import numpy as np  # 用于 ndarray 类型判断，防止数组穿透到标量计算中。

from liquidloc.common.validation import coerce_finite_scalar  # 有限浮点标量转换的统一入口。
from liquidloc.protocol.event_schema import Event, validate_event  # 导入事件对象和事件校验函数。
from liquidloc.protocol.liquid_bridge_contract import clip_uwb_bias  # bias 截断的唯一权威实现。


def _coerce_real_number(value, *, name):  # 把输入统一转成实数浮点数。
    """把输入统一转成有限实数浮点数，拒绝布尔值和不可转换类型。

    委托到 common.validation.coerce_finite_scalar，额外显式拒绝 np.ndarray
    （包括 0 维数组），防止数组穿透到标量计算中。与 predict_step.py 的
    _coerce_numeric_scalar 和 uwb_update_step.py 的 _coerce_scalar 口径一致。

    Args:
        value: 待转换的值，必须是数值标量（int、float、np.float64 等），
            不接受 bool、np.bool_ 和 np.ndarray（包括 0 维数组）。
        name: 参数名称，用于构造错误信息，方便定位问题来源。

    Returns:
        转换后的有限浮点数。

    Raises:
        TypeError: 当输入是布尔值、np.ndarray 或非数值类型时。
        ValueError: 当输入为 None、是 NaN 或 inf 时。
    """
    if isinstance(value, np.ndarray):  # 标量位置不能塞数组，防止数组穿透到标量计算中。
        raise TypeError(f"{name} must be a numeric scalar, got ndarray")  # 直接拒绝。
    return coerce_finite_scalar(value, name=name)  # 委托公共函数做有限性和数值校验。


def apply_bias(uwb_event, bias_value):  # 把偏置修正应用到单个 UWB 事件上。
    """将偏置修正应用到单个 UWB 事件的距离测量上。

    .. deprecated::
        生产链路 ``fusion_runner.run_fusion`` 直接调用
        ``build_measurement_control``，不经过本函数。本函数仅保留供历史调用方和测试使用。

    .. note::
        本函数为 **z 侧重写**（subtractive）的旧接口。生产链路自 §3.0.2 / §3.1.2
        起，所有 NN bias 写入改为 h 侧（``predict_range(..., extra_bias=...)`` /
        ``run_uwb_update(..., extra_bias=...)`` / FGO ``constraint["extra_bias"]`` /
        联合路径 ``step_joint(..., uwb_extra_biases=...)``）。本函数保留 z 侧语义
        仅为兼容历史调用方与离线测试，不得用于任何主路径（否则违反 §3.0.2 写入口边界
        与「残差定义在原始读数合同上」要求）。

    处理流程：
    1. 校验输入事件和偏置值非空
    2. 将事件统一转为字典形态
    3. 校验事件结构合法性（validate_event）
    4. 从事件中提取 UWB 负载和原始距离
    5. 委托 ``clip_uwb_bias`` 施加非负约束、比例截断和绝对值上限截断
    6. 计算修正后距离（不低于 0）
    7. 返回修正结果和诊断报告

    Args:
        uwb_event: UWB 测量事件，支持 Event 实例或字典形态；
            必须包含 `uwb_payload` 字段，且 `uwb_payload.range` 必须为数值。
        bias_value: 模型输出的偏置值，将被从原始距离中减去；
            会先经 _coerce_real_number 转换为浮点数，
            再经 ``clip_uwb_bias`` 做协议级截断防止极端值。

    Returns:
        二元组 (corrected_range, adapter_report)：
        - corrected_range: 修正后的距离（float，不低于 0.0）
        - adapter_report: 诊断报告字典，包含：
            - "raw_range": 原始距离
            - "bias_value": 截断后的偏置值
            - "corrected_range": 修正后的距离

    Raises:
        ValueError: 事件为 None、偏置为 None、或事件缺少 uwb_payload。
        TypeError: 事件类型不支持、或距离/偏置无法转为浮点数。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "bias_value": bias_value,
        "uwb_event_type": type(uwb_event).__name__,
    }, "apply_bias")
    if uwb_event is None:  # 输入事件不能为空，否则没有东西可修正。
        raise ValueError("uwb_event must not be None")  # 明确告诉调用方事件丢了。
    if bias_value is None:  # 偏置也不能为空，否则没法计算修正量。
        raise ValueError("bias_value must not be None")  # 明确告诉调用方偏置缺失。

    if isinstance(uwb_event, Event):  # 如果传进来的是标准事件对象，就先转成字典。
        event_payload = uwb_event.to_dict()  # 这样后面可以统一按键读取。
    elif isinstance(uwb_event, dict):  # 如果本来就是字典，就直接复用。
        event_payload = uwb_event  # 这里不做额外拷贝，保持原始内容。
    else:  # 既不是 Event 也不是 dict，就说明调用方式错了。
        raise TypeError(f"uwb_event must be an Event or dict, got {type(uwb_event).__name__}")  # 直接报出实际类型。

    # 先校验事件结构合法性，再提取 payload 字段，避免在非法事件上读取字段产生误导性错误。
    validate_event(uwb_event)

    uwb_payload = event_payload.get("uwb_payload")  # 从事件里取出 UWB 负载。
    if uwb_payload in (None, {}):  # 如果没有负载，就没法读距离。
        raise ValueError("uwb_event must contain a uwb_payload")  # 直接阻断，避免后面 KeyError。

    raw_range = _coerce_real_number(uwb_payload["range"], name="uwb_payload.range")  # 先把原始距离统一成 float。
    bias_value = _coerce_real_number(bias_value, name="bias_value")  # 再把偏置也统一成 float。

    # bias 截断委托协议层唯一权威实现，禁止在 fusion 层复制截断规则。
    # 截断规则：非负约束 + 比例截断（uwb_bias_max_ratio）+ 绝对值上限（uwb_bias_absolute_max）。
    bias_value = clip_uwb_bias(bias_value, raw_range)

    corrected_range = max(0.0, raw_range - bias_value)  # 修正后距离不能小于 0，所以这里做下界截断。

    adapter_report = {  # 报告字典只记录修正过程中的关键数值。
        "raw_range": raw_range,  # 原始距离，便于对比修正前后差异。
        "bias_value": bias_value,  # 模型给出的偏置值（经截断后），便于排查偏置来源。
        "corrected_range": corrected_range,  # 最终输出距离，供上层直接消费。
    }  # 报告字典在这里组装完毕。
    return corrected_range, adapter_report  # 同时返回修正结果和报告。
