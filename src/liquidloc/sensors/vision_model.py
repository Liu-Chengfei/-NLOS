"""VIO 事件提取模块。

核心数据流：
    Event/dict → extract_vio_measurement → {"dx", "dy", "dyaw", "quality"}

上游依赖：
    liquidloc.common.constants（DEFAULT_THRESHOLDS, MODALITY_VIO, PAYLOAD_KEYS）
    liquidloc.common.validation（coerce_finite_scalar, is_integer, require_keys）
    liquidloc.protocol.bridge_thresholds（BRIDGE_THRESHOLDS）
    liquidloc.protocol.event_schema（Event, REQUIRED_PAYLOAD_FIELDS, validate_event）

下游调用者：
    liquidloc.estimators.vision_update_step（extract_vio_measurement → build_vio_measurement）
    liquidloc.sensors.__init__（对外导出）

关键设计决策：
    - coerce_finite_scalar 用于字段级有限性与范围校验（如 dx/dy/dyaw 位移范围、
      quality ∈ [0, 1]），拒绝 bool/NaN/inf 和越界值。
    - extract_vio_measurement 先调用 validate_event 做协议层结构校验，
      再在此基础上做更严格的字段级校验（dx/dy/dyaw 有限、quality 严格范围），
      保证协议层统一校验入口不被绕过。
    - quality 通过 coerce_finite_scalar 做严格范围校验（[0, 1]），超界直接报错，
      不走裁剪归一化路径，与 uwb_model 的严格模式一致。
    - VIO 紧耦合仅输出位姿增量 (dx, dy, dyaw) 与 quality，不输出 tracked_features /
      reproj_err（铁律 3，Stage A1 下游修复，2026-07-23）。
"""

from __future__ import annotations  # 允许前向类型标注。

from typing import Any  # 用于 Event 转 dict 后的宽松字段类型标注。

from liquidloc.common.constants import DEFAULT_THRESHOLDS, MODALITY_VIO, PAYLOAD_KEYS  # VIO 模态、标准 payload 键和阈值表。
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS  # 桥接层业务阈值（vio_dyaw_min/max）。
from liquidloc.common.validation import coerce_finite_scalar, is_integer, require_keys  # 有限数值标量校验、整数判断和必需字段校验工具。
from liquidloc.protocol.event_schema import Event, REQUIRED_PAYLOAD_FIELDS, validate_event  # VIO 协议要求的字段集 + 统一事件校验入口。


def _coerce_nonnegative_int(value, *, name: str) -> int:
    """把单个字段规整成非负整数。

    参数：
        value: 输入值，必须是整数类型（int、np.int64 等），不接受 bool。
        name: 字段名，用于错误消息。

    返回：
        非负整数。

    异常：
        TypeError: 输入是 bool 或非整数类型（包括浮点数）。
        ValueError: 输入为负数。

    注意：
        bool 虽然 ``isinstance(True, Integral)`` 为 True，但这里显式排除。
        兼容 numpy 整数标量（``np.int64`` 等），因为 ``isinstance(np.int64, Integral)`` 为 True。
    """
    if not is_integer(value):  # tracked_features 必须是整数。
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")  # 明确说明类型不对。
    int_value = int(value)  # 一次性转为 Python int，避免重复转换。
    if int_value < 0:  # 特征数量不能是负数。
        raise ValueError(f"{name} must be >= 0, got {value}")  # 负值直接拒绝。
    return int_value  # 返回标准整数。


def _as_vio_event_dict(vio_event: Event | dict[str, Any]) -> dict[str, Any]:
    """把 Event 对象或 dict 统一成普通字典。

    参数：
        vio_event: Event 对象或字典。

    返回：
        字典视图。对 Event 输入返回新字典副本；对 dict 输入返回原引用（不做防御性拷贝），
        调用者不应修改返回的字典。

    异常：
        TypeError: 输入既不是 Event 也不是 dict。
    """
    if isinstance(vio_event, Event):  # Event 对象先转成字典。
        return vio_event.to_dict()  # 对象型输入转成普通映射。
    if isinstance(vio_event, dict):  # 字典输入直接放行。
        return vio_event  # 直接复用原始字典。
    raise TypeError(f"vio_event must be an Event or dict, got {type(vio_event).__name__}")  # 其他类型都不合法。


def extract_vio_measurement(vio_event) -> dict[str, float | int]:
    """从 VIO 事件里提取统一的相对位姿测量字典。

    处理流程：
        1. 统一输入为字典（_as_vio_event_dict）
        2. 调用 validate_event 做协议层结构校验（模态、payload 存在性、必需字段齐全）
        3. 模态二次确认（防止 validate_event 通过但模态实际不是 vio）
        4. 逐字段提取与类型/值域校验：
           - dx/dy/dyaw: 有限浮点数（通过 coerce_finite_scalar，含物理范围约束）
           - quality: 严格校验必须落在 [0, 1]（通过 coerce_finite_scalar，超界直接报错）

    参数：
        vio_event: Event 对象或字典，必须包含 ``"modality"`` 和 ``"vio_payload"``。

    返回：
        标准化测量字典 ``{"dx": float, "dy": float, "dyaw": float, "quality": float}``。

    异常：
        ValueError: 模态不匹配、payload 缺失或字段值非法（NaN/inf/负数等）。
        TypeError: payload 不是 dict，或字段值类型非法（如 bool）。
        KeyError: payload 缺少必需字段。

    注意：
        本函数先调用 ``validate_event`` 做协议层结构校验（模态、payload 存在性、
        必需字段齐全），再在此基础上做更严格的字段级校验（dx/dy/dyaw 有限、
        quality 严格范围）。
        quality 通过 ``coerce_finite_scalar`` 做严格范围校验，超界直接报错，
        不走裁剪归一化路径，与 ``uwb_model`` 的严格模式一致。
        VIO 紧耦合仅输出 4 字段（dx/dy/dyaw/quality），不输出 tracked_features /
        reproj_err（铁律 3，Stage A1 下游修复，2026-07-23）。
    """
    from liquidloc.common.tee_logger import print_dict  # 局部导入参数打印工具。
    print_dict({"vio_event": vio_event}, "extract_vio_measurement 入参", prefix="[配置]")
    event_dict = _as_vio_event_dict(vio_event)  # 统一输入类型。
    validate_event(event_dict)  # 协议层统一校验入口，保证模态/payload/必需字段齐全。

    modality = event_dict["modality"]  # 读取模态字段。
    if modality != MODALITY_VIO:  # 不是 VIO 就不能按这个提取器处理。
        raise ValueError(f"extract_vio_measurement expects modality={MODALITY_VIO!r}, got {modality!r}")  # 明确说明模态不匹配。

    payload_key = PAYLOAD_KEYS[MODALITY_VIO]  # 通过模态拿到标准 payload 键名。
    payload = event_dict[payload_key]  # 按标准键拿 payload。
    if payload is None:  # payload 缺失时不能继续。
        raise ValueError(f"Missing payload for modality={MODALITY_VIO}: {payload_key}")  # 明确指出缺的是哪个 payload。
    if not isinstance(payload, dict):  # 这里要求 payload 本身是字典。
        raise TypeError(f"{payload_key} must be a dict")  # 不符合协议就直接报错。

    require_keys(payload, REQUIRED_PAYLOAD_FIELDS[MODALITY_VIO], name=payload_key)  # 必需字段必须齐全。

    dx = coerce_finite_scalar(
        payload["dx"], name=f"{payload_key}.dx",
        min_value=DEFAULT_THRESHOLDS["vio_displacement_min"],
        max_value=DEFAULT_THRESHOLDS["vio_displacement_max"],
    )  # 平移 x 分量，必须在协议位移范围内。
    dy = coerce_finite_scalar(
        payload["dy"], name=f"{payload_key}.dy",
        min_value=DEFAULT_THRESHOLDS["vio_displacement_min"],
        max_value=DEFAULT_THRESHOLDS["vio_displacement_max"],
    )  # 平移 y 分量，必须在协议位移范围内。
    dyaw = coerce_finite_scalar(payload["dyaw"], name=f"{payload_key}.dyaw",
                                min_value=BRIDGE_THRESHOLDS["vio_dyaw_min"],
                                max_value=BRIDGE_THRESHOLDS["vio_dyaw_max"])  # 航向变化，含物理范围约束 [-2π, 2π]。
    quality = coerce_finite_scalar(  # 质量严格范围校验，超界直接报错，不走裁剪归一化路径。
        payload["quality"],  # 从 payload 中读取原始质量值。
        name=f"{payload_key}.quality",  # 在报错里带上完整字段路径。
        min_value=DEFAULT_THRESHOLDS["quality_min"],  # 质量值不能低于协议下界。
        max_value=DEFAULT_THRESHOLDS["quality_max"],  # 质量值不能高于协议上界。
    )
    # 铁律 3 (Stage A1 下游修复, 2026-07-23 audit Round 6): 删除 tracked_features /
    # reproj_err 提取 — VIO 紧耦合不再输出此两字段. Round 6 审计 (子代理
    # #47ca7cc9) 揭露这是 P0 致命 bug: payload["tracked_features"] KeyError
    # 会让所有 5 个模型的 VIO 更新路径全崩. 现在 vio_measurement 只含 4 字段.
    vio_measurement = {  # 统一按固定键名重组。
        "dx": dx,  # x 平移分量。
        "dy": dy,  # y 平移分量。
        "dyaw": dyaw,  # 航向变化。
        "quality": quality,  # 归一化质量。
    }  # 字典构建结束。
    return vio_measurement  # 返回标准化后的 VIO 测量字典。
