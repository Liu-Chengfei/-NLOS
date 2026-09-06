"""统一事件结构与序列校验模块。

文件职责：
  把不同来源的事件整理成同一种结构，并检查事件序列的时间顺序、
  模态字段和各类 payload 的完整性。

本文件绝对不负责：
  不修改事件数据本身，只做校验和格式整理。

上游依赖：liquidloc.common.constants、liquidloc.common.validation、liquidloc.protocol.sensor_contract
下游调用者：scenarios/async_levels.py、scenarios/nlos_levels.py、
  scenarios/visual_levels.py、pipelines/core_pipeline.py、
  fusion/、estimators/、dataio/adapters/event_builder.py 等。

输入对象定义：
  - Event   dataclass 事件对象
  - dict    字典格式事件

输出对象定义：
  - validate_event            单事件严格校验（无返回值，异常即错误）
  - validate_event_sequence   序列校验（含时间单调性、dt 一致性、scene_id/seq_id 一致性）
  - coerce_event_for_feature_extraction  宽松格式整理（补齐空 payload，供特征提取用）

核心变量定义：
  - REQUIRED_PAYLOAD_FIELDS   每种模态必须具备的 payload 字段

关键设计决策：
  - validate_event 是严格模式：非当前模态的 payload 必须为 None 或 {}。
  - coerce_event_for_feature_extraction 是宽松模式：只补齐不报错跨模态冲突。
  - 空序列被 validate_event_sequence 显式拒绝。
  - 数值型字段通过 _require_numeric 拒绝 nan/inf（内置有限性检查）；IMU 的 ax/ay/gz 仅检查有限性不限范围，VIO 的 dyaw 通过 BRIDGE_THRESHOLDS 检查范围，其余数值型字段通过 require_in_range 检查范围。
  - 可选 payload 字段的默认值统一为 None，coerce 时转换为空字典（深拷贝嵌套内容，隔离可变引用）。
"""

from __future__ import annotations  # 允许类型注解里引用尚未定义的类型名。

import copy
from dataclasses import asdict  # 将 dataclass 对象转成普通字典。
from dataclasses import dataclass  # 提供数据类装饰器。
from math import isfinite  # 用来判断浮点数是否有限（非 nan/inf）。
from typing import Any  # 用来表示任意类型。

from liquidloc.common.constants import ALLOWED_MODALITIES  # 允许出现的模态列表。
from liquidloc.common.constants import DEFAULT_THRESHOLDS  # 通用阈值配置。
from liquidloc.common.constants import META_KEYS  # meta 字段必须具备的键。
from liquidloc.common.constants import MODALITY_IMU  # IMU 模态名常量。
from liquidloc.common.constants import MODALITY_UWB  # UWB 模态名常量。
from liquidloc.common.constants import MODALITY_VIO  # VIO 模态名常量。
from liquidloc.common.constants import MODALITY_FLOW  # 光流/流量类模态名常量。
from liquidloc.common.constants import MODALITY_TOF  # ToF 测距类模态名常量。
from liquidloc.common.constants import PAYLOAD_KEYS  # 每种模态对应的 payload 键名。
from liquidloc.common.constants import PRIMARY_EVENT_KEYS  # 事件主键集合。
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS  # 桥接层阈值（VIO dyaw 范围等）。
from liquidloc.common.validation import is_bool_like, is_integer  # 检查值是否为布尔类型（含 numpy.bool_）和整数类型。
from liquidloc.common.validation import is_real  # 检查值是否为实数类型（排除 bool 和 numpy.bool_）。
from liquidloc.common.validation import is_string_like  # 检查值是否为字符串类型（含 numpy.str_）。
from liquidloc.common.validation import require_in_range  # 检查数值范围。
from liquidloc.common.validation import require_iterable  # 检查输入是否可迭代。
from liquidloc.common.validation import require_keys  # 检查字典是否包含必须键。
from liquidloc.common.validation import require_not_none  # 检查对象不为空。
from liquidloc.protocol.sensor_contract import get_required_payload_fields  # 统一读取冻结传感器字段合同。


import os

def _payload_get(payload: dict[str, Any], key: str) -> Any:
    """安全读取 payload 字段值，避免将字典键名（如 'range'）遮蔽 Python 内置。

    直接写 ``range_val = payload["range"]`` 虽然不会运行时冲突，
    但如果开发者误写 ``range = payload["range"]`` 就会遮蔽内置函数。
    统一通过此函数访问，从根源消除命名冲突风险。

    前置条件：调用者必须先通过 ``require_keys`` 确保 ``key`` 存在于 ``payload`` 中，
    否则将抛出 ``KeyError``。
    """
    if os.environ.get("LIQUIDLOC_DEBUG_TRACE") == "1":
        from liquidloc.common.tee_logger import print_dict
        print_dict({"payload": payload, "key": key}, "_payload_get 入参", prefix="[配置]")
    return payload[key]


def _require_numeric(value: Any, field_path: str) -> float:
    """校验值为有限数值型（排除 bool、complex、nan、inf），返回 float。

    统一 IMU/UWB/VIO 各模态的数值类型检查逻辑，避免各校验函数
    各自手写 isinstance 判断导致风格不一致。
    同时内置有限性检查，确保 nan/inf 不会静默通过。
    """
    if os.environ.get("LIQUIDLOC_DEBUG_TRACE") == "1":
        from liquidloc.common.tee_logger import print_dict
        print_dict({"value": value, "field_path": field_path}, "_require_numeric 入参", prefix="[配置]")
    if not is_real(value):
        raise TypeError(f"{field_path} must be numeric, got {type(value).__name__}")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field_path} must be finite, got {result}")
    return result


def _require_integer(value: Any, field_path: str) -> int:
    """校验值为整数型（排除 bool），返回 int。

    统一 tracked_features 等整数字段的类型检查逻辑。
    """
    if os.environ.get("LIQUIDLOC_DEBUG_TRACE") == "1":
        from liquidloc.common.tee_logger import print_dict
        print_dict({"value": value, "field_path": field_path}, "_require_integer 入参", prefix="[配置]")
    if not is_integer(value):
        raise TypeError(f"{field_path} must be an integer, got {type(value).__name__}")
    return int(value)


@dataclass(slots=True)  # 定义一个轻量事件对象，slots=True 节省内存。
class Event:
    """事件对象，承载单条传感器观测的完整信息。

    每个事件对应一个时间步的一种模态观测，包含时间戳、模态标识、
    元信息和对应模态的负载数据。非当前模态的 payload 必须为 None。

    .. warning::
        Event 构造函数不做任何校验。构造后**必须**经过 ``validate_event()`` 校验，
        否则非法值（如 t=nan、dt=-1）会静默传播到下游。

    属性：
        t：事件时间戳（秒）。
        dt：与上一个事件的时间间隔（秒）。
        modality：当前事件模态名称（imu/uwb/vio/flow/tof）。
        meta：事件元信息字典，必须包含 scene_id 和 seq_id。
        imu_payload：IMU 模态负载，非 IMU 事件时为 None。
        uwb_payload：UWB 模态负载，非 UWB 事件时为 None。
        vio_payload：VIO 模态负载，非 VIO 事件时为 None。
        flow_payload：光流/流量类模态负载，非 flow 事件时为 None（UTIL 数据集）。
        tof_payload：ToF 测距类模态负载，非 ToF 事件时为 None（UTIL 数据集）。
    """

    t: float  # 事件时间戳。
    dt: float  # 与上一个事件的时间间隔。
    modality: str  # 当前事件模态。
    meta: dict[str, Any]  # 事件元信息。
    imu_payload: dict[str, Any] | None = None  # IMU 模态的负载。
    uwb_payload: dict[str, Any] | None = None  # UWB 模态的负载。
    vio_payload: dict[str, Any] | None = None  # VIO 模态的负载。
    flow_payload: dict[str, Any] | None = None  # 光流/流量类模态的负载（UTIL 数据集）。
    tof_payload: dict[str, Any] | None = None  # ToF 测距类模态的负载（UTIL 数据集）。

    def to_dict(self) -> dict[str, Any]:
        """把事件对象转成普通字典。"""
        return asdict(self)  # 直接展开 dataclass。


REQUIRED_PAYLOAD_FIELDS = get_required_payload_fields()  # 每种模态必须具备的 payload 字段。
# 键集对齐断言：REQUIRED_PAYLOAD_FIELDS 的模态集合必须与 ALLOWED_MODALITIES 一致，
# 防止 sensor_contract 与 constants 注册表不同步。
if set(REQUIRED_PAYLOAD_FIELDS.keys()) != set(ALLOWED_MODALITIES):
    raise AssertionError(
        f"REQUIRED_PAYLOAD_FIELDS keys {set(REQUIRED_PAYLOAD_FIELDS.keys())} "
        f"!= ALLOWED_MODALITIES {set(ALLOWED_MODALITIES)}"
    )
# 冻结外层字典，防止运行时篡改模态字段定义。
_REQUIRED_PAYLOAD_FIELDS_DATA = dict(REQUIRED_PAYLOAD_FIELDS)  # 保留原始数据。


class _FrozenDict(dict):  # type: ignore[misc]
    """只读字典包装，禁止增删改操作。"""

    __slots__ = ()

    def __setitem__(self, key, value):  # type: ignore[override]
        raise TypeError("REQUIRED_PAYLOAD_FIELDS is read-only")

    def __delitem__(self, key):  # type: ignore[override]
        raise TypeError("REQUIRED_PAYLOAD_FIELDS is read-only")

    def pop(self, *args):  # type: ignore[override]
        raise TypeError("REQUIRED_PAYLOAD_FIELDS is read-only")

    def popitem(self):  # type: ignore[override]
        raise TypeError("REQUIRED_PAYLOAD_FIELDS is read-only")

    def clear(self):  # type: ignore[override]
        raise TypeError("REQUIRED_PAYLOAD_FIELDS is read-only")

    def update(self, *args, **kwargs):  # type: ignore[override]
        raise TypeError("REQUIRED_PAYLOAD_FIELDS is read-only")

    def setdefault(self, *args):  # type: ignore[override]
        raise TypeError("REQUIRED_PAYLOAD_FIELDS is read-only")

    def __ior__(self, other):  # type: ignore[override]
        raise TypeError("REQUIRED_PAYLOAD_FIELDS is read-only")

    def __hash__(self):  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __eq__(self, other):  # type: ignore[override]
        if isinstance(other, _FrozenDict):
            return dict(self) == dict(other)
        return dict(self) == other

    def __contains__(self, key):  # type: ignore[override]
        return dict.__contains__(self, key)


REQUIRED_PAYLOAD_FIELDS = _FrozenDict(_REQUIRED_PAYLOAD_FIELDS_DATA)


def _as_event_dict(event: Event | dict[str, Any]) -> dict[str, Any]:
    """把事件统一成字典视图，不做深拷贝。

    参数：
        event：Event 对象或字典格式的事件。

    返回：
        dict[str, Any]：事件的字典视图。Event 对象会通过 to_dict() 转换，
        字典直接复用引用（不拷贝）。

    异常：
        TypeError：当 event 既不是 Event 也不是 dict 时抛出。
    """
    if os.environ.get("LIQUIDLOC_DEBUG_TRACE") == "1":
        from liquidloc.common.tee_logger import print_dict
        print_dict({"event": event}, "_as_event_dict 入参", prefix="[配置]")
    if isinstance(event, Event):  # 如果本来就是 Event 对象。
        return event.to_dict()  # 先转成字典。
    if isinstance(event, dict):  # 如果本来就是字典。
        return event  # 直接复用，不做拷贝。
    raise TypeError(f"event must be an Event or dict, got {type(event).__name__}")  # 其他类型不接受。


def _validate_imu_payload(required_payload: dict[str, Any], required_payload_key: str) -> None:
    """校验 IMU payload 的字段值是否有限（非 nan/inf）。

    逐个检查 ax、ay、gz 三个字段，要求它们必须是数值型（排除 bool）且为有限数。

    参数：
        required_payload：IMU payload 字典，必须包含 ax、ay、gz 键。
        required_payload_key：payload 在事件中的键名，用于构造错误信息。

    异常：
        TypeError：字段值不是数值型（包括 bool）时抛出。
        ValueError：字段值为 nan 或 inf 时抛出。
    """
    if os.environ.get("LIQUIDLOC_DEBUG_TRACE") == "1":
        from liquidloc.common.tee_logger import print_dict
        print_dict({"required_payload": required_payload, "required_payload_key": required_payload_key}, "_validate_imu_payload 入参", prefix="[配置]")
    for field_name in REQUIRED_PAYLOAD_FIELDS[MODALITY_IMU]:  # 遍历 ax, ay, gz。
        _require_numeric(  # 统一数值类型+有限性检查（内置 isfinite）。
            _payload_get(required_payload, field_name),
            f"{required_payload_key}.{field_name}",
        )


def _validate_uwb_payload(required_payload: dict[str, Any], required_payload_key: str) -> None:
    """校验 UWB payload 的额外约束：anchor_id 非空字符串或整数、quality 范围、valid 类型、range 非负。

    参数：
        required_payload：UWB payload 字典，必须包含 anchor_id、quality、valid、range 键。
        required_payload_key：payload 在事件中的键名，用于构造错误信息。

    异常：
        ValueError：quality 超出 [0,1] 范围或 range 为负数时抛出。
        TypeError：valid 不是 bool 类型或 anchor_id 不是非空字符串或整数时抛出。
    """
    if os.environ.get("LIQUIDLOC_DEBUG_TRACE") == "1":
        from liquidloc.common.tee_logger import print_dict
        print_dict({"required_payload": required_payload, "required_payload_key": required_payload_key}, "_validate_uwb_payload 入参", prefix="[配置]")
    # anchor_id 必须是非空字符串或整数（含 numpy 整数），作为锚点标识符不允许为空。
    anchor_id = _payload_get(required_payload, "anchor_id")
    if is_bool_like(anchor_id) or not (is_string_like(anchor_id) or is_integer(anchor_id)):
        raise TypeError(
            f"{required_payload_key}.anchor_id must be a string or integer, "
            f"got {type(anchor_id).__name__} {anchor_id!r}"
        )
    if is_string_like(anchor_id) and not str(anchor_id).strip():
        raise ValueError(
            f"{required_payload_key}.anchor_id must be a non-empty string"
        )
    if not is_string_like(anchor_id):
        # 整数 anchor_id（含 numpy 整数）必须非负。
        anchor_id_int = int(anchor_id)
        if anchor_id_int < 0:
            raise ValueError(
                f"{required_payload_key}.anchor_id must be non-negative when integer, got {anchor_id_int}"
            )
    require_in_range(  # 检查质量范围。
        _payload_get(required_payload, "quality"),
        f"{required_payload_key}.quality",
        min_value=DEFAULT_THRESHOLDS["quality_min"],
        max_value=DEFAULT_THRESHOLDS["quality_max"],
    )
    valid_value = _payload_get(required_payload, "valid")
    if not is_bool_like(valid_value):  # valid 必须是布尔值（含 numpy.bool_）。
        raise TypeError(
            f"{required_payload_key}.valid must be a bool, got {type(valid_value).__name__}"
        )
    require_in_range(  # range 不能为负。通过 _payload_get 访问，避免 'range' 遮蔽 Python 内置。
        _payload_get(required_payload, "range"),
        f"{required_payload_key}.range",
        min_value=0.0,
    )


def _validate_vio_payload(required_payload: dict[str, Any], required_payload_key: str) -> None:
    """校验 VIO payload 的额外约束：dx/dy 范围、dyaw 有限性、quality 范围。

    参数：
        required_payload：VIO payload 字典，必须包含 dx、dy、dyaw、quality 键
        （由 REQUIRED_PAYLOAD_FIELDS[MODALITY_VIO] = (dx, dy, dyaw, quality) 保证）。
        required_payload_key：payload 在事件中的键名，用于构造错误信息。

    异常：
        ValueError：quality 超出 [0,1] 范围、dx/dy/dyaw 超出位移/角度范围时抛出。
        TypeError：字段不是数值型（如 bool）时抛出。

    注意：
        VIO 紧耦合仅校验 4 字段（dx/dy/dyaw/quality），不校验 tracked_features /
        reproj_err（铁律 3，Stage A1 下游修复，2026-07-23）。
    """
    if os.environ.get("LIQUIDLOC_DEBUG_TRACE") == "1":
        from liquidloc.common.tee_logger import print_dict
        print_dict({"required_payload": required_payload, "required_payload_key": required_payload_key}, "_validate_vio_payload 入参", prefix="[配置]")
    require_in_range(  # 检查质量范围。
        _payload_get(required_payload, "quality"),
        f"{required_payload_key}.quality",
        min_value=DEFAULT_THRESHOLDS["quality_min"],
        max_value=DEFAULT_THRESHOLDS["quality_max"],
    )
    # 校验 VIO 增量的范围（NaN/inf 不允许进入滤波器）。
    # dx/dy 为位移增量（米），范围由 DEFAULT_THRESHOLDS 统一管理。
    # dyaw 为航向增量（弧度），范围由 BRIDGE_THRESHOLDS 统一管理。
    # 场景生成器在 drift_bias_m 应用后已执行 wrap_angle_rad，
    # 出口校验时 dyaw 必然在 [-π, π) 内，[-2π, 2π] 为宽松上界。
    for field_name in ("dx", "dy"):
        require_in_range(
            _payload_get(required_payload, field_name),
            f"{required_payload_key}.{field_name}",
            min_value=DEFAULT_THRESHOLDS["vio_displacement_min"],
            max_value=DEFAULT_THRESHOLDS["vio_displacement_max"],
        )
    # dyaw 检查有限性和范围。
    # 场景生成器在 drift_bias_m 应用后已执行 wrap_angle_rad，
    # 出口校验时 dyaw 必然在 [-π, π) 内。使用 [-2π, 2π] 作为宽松上界，
    # 与 vision_model.py extract_vio_measurement 的校验策略对齐。
    require_in_range(
        _payload_get(required_payload, "dyaw"),
        f"{required_payload_key}.dyaw",
        min_value=BRIDGE_THRESHOLDS["vio_dyaw_min"],
        max_value=BRIDGE_THRESHOLDS["vio_dyaw_max"],
    )
    # 铁律 3 (Stage A1 下游修复, 2026-07-23 audit Round 6): 删除 tracked_features /
    # reproj_err 的 require_in_range 校验 — VIO 紧耦合不再输出此两字段,
    # sensors.yaml vio_fields 已缩到 4 项 (dx, dy, dyaw, quality).
    # Round 6 审计 (子代理 #47ca7cc9) 揭露这是 P0 致命 bug: 任何 VIO 事件
    # validate_event → _validate_vio_payload → _payload_get("tracked_features")
    # 必然 KeyError, 5 个模型 (ekf/robust_ekf/fgo/liquid_ekf/lstm_ekf)
    # 共享 extract_vio_measurement → validate_event 调用链全崩.


def _validate_flow_payload(required_payload: dict[str, Any], required_payload_key: str) -> None:
    """校验 flow payload 的额外约束：dx/dy 有限且在位移范围内，quality 在 [0,1]。

    参数：
        required_payload：flow payload 字典，必须包含 dx、dy、quality 键。
        required_payload_key：payload 在事件中的键名，用于构造错误信息。

    异常：
        ValueError：quality 超出 [0,1] 范围或 dx/dy 超出位移范围时抛出。
        TypeError：字段不是数值型时抛出。
    """
    if os.environ.get("LIQUIDLOC_DEBUG_TRACE") == "1":
        from liquidloc.common.tee_logger import print_dict
        print_dict({"required_payload": required_payload, "required_payload_key": required_payload_key}, "_validate_flow_payload 入参", prefix="[配置]")
    require_in_range(
        _payload_get(required_payload, "quality"),
        f"{required_payload_key}.quality",
        min_value=DEFAULT_THRESHOLDS["quality_min"],
        max_value=DEFAULT_THRESHOLDS["quality_max"],
    )
    for field_name in ("dx", "dy"):
        require_in_range(
            _payload_get(required_payload, field_name),
            f"{required_payload_key}.{field_name}",
            min_value=DEFAULT_THRESHOLDS["flow_displacement_min"],
            max_value=DEFAULT_THRESHOLDS["flow_displacement_max"],
        )


def _validate_tof_payload(required_payload: dict[str, Any], required_payload_key: str) -> None:
    """校验 ToF payload 的额外约束：range 非负，quality 在 [0,1]。

    参数：
        required_payload：ToF payload 字典，必须包含 range、quality 键。
        required_payload_key：payload 在事件中的键名，用于构造错误信息。

    异常：
        ValueError：quality 超出 [0,1] 范围或 range 为负时抛出。
        TypeError：字段不是数值型时抛出。
    """
    if os.environ.get("LIQUIDLOC_DEBUG_TRACE") == "1":
        from liquidloc.common.tee_logger import print_dict
        print_dict({"required_payload": required_payload, "required_payload_key": required_payload_key}, "_validate_tof_payload 入参", prefix="[配置]")
    require_in_range(
        _payload_get(required_payload, "quality"),
        f"{required_payload_key}.quality",
        min_value=DEFAULT_THRESHOLDS["quality_min"],
        max_value=DEFAULT_THRESHOLDS["quality_max"],
    )
    require_in_range(
        _payload_get(required_payload, "range"),
        f"{required_payload_key}.range",
        min_value=0.0,
    )


def coerce_event_for_feature_extraction(event: Event | dict[str, Any]) -> dict[str, Any]:
    """把输入事件整理成适合特征提取的宽松格式。

    与 validate_event 不同，此函数不检查跨模态 payload 冲突，
    只补齐缺失的 payload 为空字典，让后续逻辑不用反复判空。
    """
    if os.environ.get("LIQUIDLOC_DEBUG_TRACE") == "1":
        from liquidloc.common.tee_logger import print_dict
        print_dict({"event": event}, "coerce_event_for_feature_extraction 入参", prefix="[配置]")
    payload = dict(_as_event_dict(event))  # 复制一份，避免改动原对象。
    require_keys(payload, PRIMARY_EVENT_KEYS, name="event")  # 先检查主键完整性。
    require_in_range(payload["t"], "event.t")  # 检查时间戳是否合法（仅校验有限性，允许相对时间戳为负）。
    require_in_range(payload["dt"], "event.dt", min_value=0.0)  # dt 不能为负。
    modality = payload["modality"]  # 读取当前模态。
    if not is_string_like(modality):  # 模态必须是字符串（含 numpy.str_）。
        raise TypeError(f"event.modality must be a string, got {type(modality).__name__}")
    if modality not in ALLOWED_MODALITIES:  # 模态不在允许列表里就拒绝。
        raise ValueError(f"Unsupported modality: {modality}")  # 直接报错。
    require_keys(payload["meta"], META_KEYS, name="event.meta")  # meta 也要完整。
    # source_t 为可选字段，但治理规则要求"不可漂移"：存在时必须为有限数值。
    source_t = payload["meta"].get("source_t")
    if source_t is not None:
        _require_numeric(source_t, "event.meta.source_t")
    payload["meta"] = copy.deepcopy(payload["meta"])  # 深拷贝 meta，避免与原始事件共享可变引用，防止下游修改污染 source_t/anchor metadata。
    for payload_key in PAYLOAD_KEYS.values():  # 逐个补齐所有 payload 键。
        raw_payload = payload.get(payload_key)  # 读取当前 payload。
        if raw_payload is None:  # 缺失时补空字典。
            payload[payload_key] = {}  # 让后续逻辑不用反复判空。
            continue  # 跳到下一个 payload 键。
        if not isinstance(raw_payload, dict):  # 如果存在但不是字典。
            raise TypeError(f"{payload_key} must be a dict when present")  # 直接拒绝。
        payload[payload_key] = copy.deepcopy(raw_payload)  # 深拷贝，避免与原始事件共享嵌套可变引用。
    return payload  # 返回可用于特征提取的宽松事件。


def validate_event(event: Event | dict[str, Any]) -> None:
    """严格检查单个事件。

    严格模式要求：
    - 当前模态的 payload 必须存在且字段完整。
    - 非当前模态的 payload 必须为 None 或 {}。
    - IMU 的 ax/ay/gz 必须有限（非 nan/inf）。
    - UWB 的 quality 在 [0,1]，valid 为 bool，range 非负。
    - VIO 的 quality 在 [0,1]，dx/dy/dyaw 在协议范围内（铁律 3 后仅 4 字段）。
    """
    payload = _as_event_dict(event)  # 统一成字典视图。
    require_keys(payload, PRIMARY_EVENT_KEYS, name="event")  # 检查主键。
    require_in_range(payload["t"], "event.t")  # 检查 t（仅校验有限性，允许相对时间戳为负）。
    require_in_range(payload["dt"], "event.dt", min_value=0.0)  # 检查 dt。
    modality = payload["modality"]  # 读取模态。
    if not is_string_like(modality):  # 模态必须是字符串（含 numpy.str_）。
        raise TypeError(f"event.modality must be a string, got {type(modality).__name__}")
    if modality not in ALLOWED_MODALITIES:  # 模态必须受支持。
        raise ValueError(f"Unsupported modality: {modality}")  # 不支持就报错。
    require_keys(payload["meta"], META_KEYS, name="event.meta")  # 检查 meta。
    # source_t 为可选字段，但治理规则要求"不可漂移"：存在时必须为有限数值。
    source_t = payload["meta"].get("source_t")
    if source_t is not None:
        _require_numeric(source_t, "event.meta.source_t")  # 统一数值类型+有限性检查。
    required_payload_key = PAYLOAD_KEYS[modality]  # 找到当前模态对应的 payload 键。
    required_payload = payload.get(required_payload_key)  # 读取该 payload。
    require_not_none(required_payload, required_payload_key)  # 该 payload 不能缺失。
    require_keys(required_payload, REQUIRED_PAYLOAD_FIELDS[modality], name=required_payload_key)  # 检查该 payload 的专属字段。
    for payload_key in PAYLOAD_KEYS.values():  # 遍历所有 payload 键。
        if payload_key == required_payload_key:
            continue  # 跳过当前模态。
        other_payload = payload.get(payload_key)
        if other_payload is None:
            continue  # None 是合法的空值。
        if not isinstance(other_payload, dict):  # 非 dict 类型（如 []、""、0、False）一律拒绝。
            raise TypeError(
                f"{payload_key} must be None or dict when modality={modality}; "
                f"got {type(other_payload).__name__} {other_payload!r}"
            )
        if other_payload:  # 非空字典也拒绝（严格模式要求跨模态互斥）。
            raise ValueError(
                f"{payload_key} must be empty when modality={modality}; got {other_payload!r}"
            )

    # 按模态分派额外校验。
    if modality == MODALITY_IMU:  # IMU 校验：ax/ay/gz 必须有限。
        _validate_imu_payload(required_payload, required_payload_key)
    elif modality == MODALITY_UWB:  # UWB 校验：quality 范围、valid 类型、range 非负。
        _validate_uwb_payload(required_payload, required_payload_key)
    elif modality == MODALITY_VIO:  # VIO 校验：dx/dy/dyaw 范围、quality 范围（铁律 3 后仅 4 字段）。
        _validate_vio_payload(required_payload, required_payload_key)
    elif modality == MODALITY_FLOW:  # flow 校验：dx/dy 位移范围、quality 范围。
        _validate_flow_payload(required_payload, required_payload_key)
    elif modality == MODALITY_TOF:  # ToF 校验：range 非负、quality 范围。
        _validate_tof_payload(required_payload, required_payload_key)
    else:  # 防御性断言：若模态通过 ALLOWED_MODALITIES 检查但无分派分支，说明分派链与白名单不同步。
        raise AssertionError(
            f"Modality {modality!r} passed ALLOWED_MODALITIES check "
            f"but has no validation dispatch in validate_event"
        )


_UNSET = object()  # 哨兵对象，用于区分"未设置"与合法值 None。


def validate_event_sequence(events: Any) -> None:
    """检查事件序列是否自洽。

    校验内容：
    - 序列不能为空（空序列在流水线中无意义，应显式拒绝）。
    - 每个事件通过 validate_event 的严格校验。
    - 首条事件的 dt 必须接近 0（不超过 time_tolerance）。
    - 时间戳单调递增（在容差范围内）。
    - dt 与相邻时间差一致（在容差范围内）。
    - scene_id 和 seq_id 在序列内保持一致。
    """
    if os.environ.get("LIQUIDLOC_DEBUG_TRACE") == "1":
        from liquidloc.common.tee_logger import print_dict
        print_dict({"keys": list(events.keys()) if hasattr(events, "keys") else type(events).__name__}, "validate_event_sequence 校验对象 keys", prefix="[配置]")
    require_iterable(events, name="events")  # 序列本身必须可迭代。
    from collections.abc import Mapping as _Mapping
    if isinstance(events, _Mapping):
        raise TypeError("events must be a sequence of events, not a single mapping")
    event_list = list(events)  # 转为列表以便检查长度。
    if not event_list:  # 空序列在流水线中无意义，显式拒绝。
        raise ValueError("event sequence must not be empty")
    scene_id_ref: Any = _UNSET  # 记录第一条事件的 scene_id（用哨兵区分"未设置"与 None）。
    seq_id_ref: Any = _UNSET  # 记录第一条事件的 seq_id。
    prev_t = None  # 记录上一条事件时间。
    tolerance = DEFAULT_THRESHOLDS["time_tolerance"]  # 时间容差来自全局阈值。
    for index, event in enumerate(event_list):  # 逐条扫描序列。
        payload = _as_event_dict(event)  # 拿字典视图，供后续时间连续性检查使用。
        validate_event(payload)  # 检查单条事件是否合法（直接传入已转换的字典，避免重复转换）。
        current_t = float(payload["t"])  # 当前时间戳统一转成浮点数。
        meta = payload["meta"]  # 读取 meta。
        if prev_t is None and float(payload["dt"]) > tolerance:  # 首条事件没有前驱，dt 必须接近 0。
            raise ValueError(
                f"first event dt exceeds tolerance at index {index}: "
                f"got {payload['dt']}, expected <= {tolerance}"
            )
        if prev_t is not None:  # 从第二条开始才能检查时间连续性。
            if current_t < prev_t - tolerance:  # 时间戳不允许倒退（容差范围内相等可接受）。
                raise ValueError(f"events are not time-monotonic at index {index}: t={current_t} < prev_t={prev_t}")
            expected_dt = current_t - prev_t  # 根据相邻时间戳算理论 dt。
            if abs(float(payload["dt"]) - expected_dt) > tolerance:  # 如果写入的 dt 和理论 dt 不一致。
                raise ValueError(  # 抛出不一致错误。
                    f"event.dt mismatch at index {index}: got {payload['dt']}, expected {expected_dt}"
                )
        prev_t = current_t  # 更新上一条时间戳。
        scene_id = meta["scene_id"]  # 读取当前 scene_id。
        seq_id = meta["seq_id"]  # 读取当前 seq_id。
        if scene_id_ref is _UNSET:  # 第一条事件先建立基准。
            scene_id_ref = scene_id  # 记录基准 scene_id。
            seq_id_ref = seq_id  # 记录基准 seq_id。
            continue  # 第一条不用做一致性比较。
        if scene_id != scene_id_ref or seq_id != seq_id_ref:  # 跨序列边界：重置 prev_t 和基准标识。
            scene_id_ref = scene_id  # 更新基准 scene_id。
            seq_id_ref = seq_id  # 更新基准 seq_id。
            prev_t = current_t  # 重置 prev_t，新子序列内部保持单调即可。
            continue  # 跳过 dt 一致性检查（跨序列边界 dt 已由合并逻辑设置）。
