"""UWB 事件提取与几何预测模块。

核心数据流：
    Event/dict → extract_uwb_measurement → {"anchor_id", "measured_range", "valid", "quality"}
    (state, anchor_pos) → predict_range_to_anchor → float

上游依赖：
    liquidloc.common.constants（DEFAULT_THRESHOLDS, MODALITY_UWB, PAYLOAD_KEYS）
    liquidloc.common.validation（coerce_finite_scalar, is_bool_like, is_integer, require_keys, require_not_none）
    liquidloc.protocol.event_schema（Event, validate_event）
    liquidloc.protocol.sensor_contract（get_required_payload_fields）

下游调用者：
    liquidloc.dataio.sim_materializer（predict_range_to_anchor）
    liquidloc.pipelines.core_pipeline（predict_range_to_anchor）
    liquidloc.models.features.feature_builder（predict_range_to_anchor）
    liquidloc.pipelines.train_pipeline（extract_uwb_measurement, predict_range_to_anchor）
    liquidloc.sensors.__init__（对外导出）

关键设计决策：
    - extract_uwb_measurement 先调 validate_event 做协议层校验，再做字段级提取。
      validate_event 已经校验过 valid 必须是 bool，但这里做了重复检查作为双重防线——
      防止 validate_event 的异常被意外吞掉时，仍然能拦截非法 valid。
    - quality 通过 coerce_finite_scalar 做严格范围校验（[0, 1]），超界直接报错，
      不走 get_event_quality 的裁剪归一化路径，与 VIO 路径的严格模式一致。
    - predict_range_to_anchor 会对 state["px"/"py"] 和 anchor_pos 坐标做有限性校验，
      拒绝 NaN/inf，避免几何 teacher 和 residual 链路继续传播非法数值。
"""

from __future__ import annotations  # 允许前向类型标注。

import math  # 用于距离计算和有限性检查。
from liquidloc.common.constants import DEFAULT_THRESHOLDS, MODALITY_UWB, PAYLOAD_KEYS  # UWB 模态、标准 payload 键和阈值表。
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_integer, require_keys, require_not_none  # 空值和字段校验工具。
from typing import Any  # 用于返回值的宽松字段类型标注。
from liquidloc.protocol.event_schema import Event, validate_event  # 统一事件类型和校验函数。
from liquidloc.protocol.sensor_contract import get_required_payload_fields  # 统一读取冻结传感器字段合同。


_UWB_FIELDS = get_required_payload_fields()[MODALITY_UWB]  # UWB payload 的冻结字段顺序。


def _coerce_nonnegative_finite_scalar(value, *, name: str) -> float:
    """把单个数值规整成非负且有限的浮点数。

    参数：
        value: 输入值，必须是数值类型（int、float、np.float64 等），不接受 bool。
        name: 字段名，用于错误消息。

    返回：
        非负有限浮点数。

    异常：
        TypeError: 输入是 bool 或非数值类型。
        ValueError: 输入是 NaN、无穷大或负数。
    """
    scalar = coerce_finite_scalar(value, name=name, min_value=0.0)  # 复用公共函数做有限性+非负校验，与 vision_model.py 的 reproj_err 校验模式一致。
    return scalar  # 返回规整后的数值。


def extract_uwb_measurement(uwb_event) -> dict[str, Any]:
    """从 UWB 事件里提取统一测量字典。

    处理流程：
        1. 非 None 检查
        2. 调用 validate_event 做协议层校验（模态、payload 完整性、quality 范围、valid 类型等）
        3. 统一转字典
        4. 模态二次确认（防止 validate_event 通过但模态实际不是 uwb）
        5. 提取字段：anchor_id 原样保留、range 经非负有限校验、valid 二次类型检查、
           quality 经 coerce_finite_scalar 严格范围校验

    双重防线说明：
        validate_event 已经校验过 valid 必须是 bool（见 event_schema._validate_uwb_payload），
        但这里对 valid 做了重复检查。这是故意的双重防线：防止 validate_event 的异常
        被意外吞掉时（如上层 try/except 过宽），仍然能拦截非法 valid。

    参数：
        uwb_event: Event 对象或字典，必须包含 ``"modality"`` 和 ``"uwb_payload"``。

    返回：
        标准化测量字典 ``{"anchor_id": str/int, "measured_range": float, "valid": bool, "quality": float}``。

    异常：
        ValueError: 输入为 None、模态不匹配或字段值非法。
        TypeError: valid 类型不是 bool，或字段值类型非法。
        KeyError: payload 缺少必需字段。
    """
    require_not_none(uwb_event, "uwb_event")  # 先排除空输入。
    validate_event(uwb_event)  # 协议层校验：模态、payload 完整性、quality 范围、valid 类型等。
    event_dict = uwb_event.to_dict() if isinstance(uwb_event, Event) else uwb_event  # 统一成字典，后续只走一种读取方式。

    modality = event_dict["modality"]  # 模态必须是 UWB。
    if modality != MODALITY_UWB:  # 不是 UWB 就不能按这个提取器处理。
        raise ValueError(f"uwb_event.modality must be {MODALITY_UWB!r}, got {modality!r}")  # 明确指出模态不匹配。

    payload_key = PAYLOAD_KEYS[MODALITY_UWB]  # 查出 UWB payload 的标准键名。
    payload = event_dict[payload_key]  # 按标准键拿 payload。
    require_keys(payload, _UWB_FIELDS, name=payload_key)  # 这组字段都必须在。

    anchor_id = payload["anchor_id"]  # anchor id 原样保留。
    if is_integer(anchor_id):  # numpy 整数转为 Python int（is_integer 已排除 bool）。
        anchor_id = int(anchor_id)
    measured_range = _coerce_nonnegative_finite_scalar(payload["range"], name=f"{payload_key}.range")  # 距离必须非负且有限。
    # quality 严格范围校验（与 VIO 路径一致，超上界报错而非静默裁剪）。
    quality = coerce_finite_scalar(
        payload["quality"], name=f"{payload_key}.quality",
        min_value=DEFAULT_THRESHOLDS["quality_min"],
        max_value=DEFAULT_THRESHOLDS["quality_max"],
    )  # coerce_finite_scalar 已返回 Python float，无需二次转换。
    valid = payload["valid"]  # 原始有效性标记保留。
    # 双重防线：validate_event 已经校验过 valid 必须是 bool，
    # 但如果 validate_event 的异常被上层意外吞掉，这里仍然能拦截非法 valid。
    if not is_bool_like(valid):  # valid 必须是布尔值。
        raise TypeError(f"{payload_key}.valid must be bool, got {type(valid).__name__}")  # 明确说明 valid 类型不对。
    valid = bool(valid)  # 确保 Python 原生 bool，防止 np.bool_ 泄漏。

    uwb_measurement = {  # 按固定键名重新组装。
        "anchor_id": anchor_id,  # 保存 anchor id。
        "measured_range": measured_range,  # 保存清洗后的测距值。
        "valid": valid,  # 保存有效性标记。
        "quality": quality,  # 保存归一化后的质量值。
    }  # 字典构建结束。
    return uwb_measurement  # 返回标准化后的 UWB 测量字典。


def predict_range_to_anchor(state, anchor_pos) -> float:
    """根据当前状态和 anchor 坐标预测几何距离。

    参数：
        state: 包含 ``"px"`` 和 ``"py"`` 键的映射（状态向量或字典）。
        anchor_pos: 二维坐标容器，长度必须为 2（元组、列表、numpy 数组等）。

    返回：
        欧氏距离（非负浮点数）。坐标输入通过有限性校验后，返回值保证有限且非负。

    异常：
        ValueError: state 为 None、anchor_pos 为 None、anchor_pos 维度不是 2。
        TypeError: anchor_pos 不是合法坐标容器。

    注意：
        训练与仿真链路依赖该几何量可直接参与 residual 和 teacher 目标构造，
        因此这里会拒绝 NaN/inf 坐标，而不是把非法值继续向下游传播。
    """
    require_not_none(state, "state")  # 状态不能是空。
    require_not_none(anchor_pos, "anchor_pos")  # anchor 坐标也不能是空。
    require_keys(state, ("px", "py"), name="state")  # 只需要平面位置就能算距离。

    if isinstance(anchor_pos, (str, bytes, dict)):  # 字符串和映射不是合法坐标容器。
        raise TypeError("anchor_pos must be a 2D coordinate record")  # 明确说明类型不对。
    try:  # 先确认 anchor_pos 至少可求长度。
        anchor_dim = len(anchor_pos)  # 这里用长度判断是不是二维坐标。
    except TypeError as exc:  # 不能取长度就说明它不是坐标容器。
        raise TypeError("anchor_pos must be a 2D coordinate record") from exc  # 保留原始异常链。
    if anchor_dim != 2:  # anchor 必须是二维平面坐标。
        raise ValueError(f"anchor_pos must contain exactly 2 coordinates, got {anchor_dim}")  # 维度不对就直接报错。

    px = coerce_finite_scalar(state["px"], name='state.px')  # 当前状态的 x 坐标。
    py = coerce_finite_scalar(state["py"], name='state.py')  # 当前状态的 y 坐标。
    anchor_x = coerce_finite_scalar(anchor_pos[0], name='anchor_pos[0]')  # anchor 的 x 坐标。
    anchor_y = coerce_finite_scalar(anchor_pos[1], name='anchor_pos[1]')  # anchor 的 y 坐标。
    predicted_range = math.hypot(anchor_x - px, anchor_y - py)  # 用欧氏距离作为几何预测值。
    return predicted_range  # 返回预测距离。
