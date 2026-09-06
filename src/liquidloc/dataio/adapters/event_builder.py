"""把 IMU、UWB、VIO 原始表构造成统一的规范事件记录。

这个模块负责把每个流的表行转成协议层和管线层都能接受的事件字典。
每个构造函数只负责自己那一类载荷，公共元数据由共享封装补齐，最后
再把多个流合并、排序并校验。这样做的好处是，每一层看到的事件形状
都一致，调试时也更容易追到具体哪个流出了问题。

上游依赖:
- adapters.field_mapper 输出的统一字段名原始行
- protocol.event_schema 中的校验函数

下游调用者:
- 准备流程脚本（直接调用 build_xxx_events 和 merge_and_finalize_events）
- dataio/__init__.py（重新导出这些函数作为顶层 API）

核心变量:
- _base_event: 所有模态共用的事件外壳模板
- imu_events / uwb_events / vio_events: 各模态构造后的事件列表
- all_events: 合并排序后的最终事件流

关键设计决策:
- 缺失必需字段时抛出 ValueError，由调用方处理
- 事件合并时保留组顺序和组内顺序，时间相同时按输入顺序排列
- dt 在合并阶段重新计算，确保时间连续性
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from collections.abc import Iterable
from decimal import Decimal
from typing import Any, Optional

from liquidloc.common.constants import (
    MODALITY_FLOW,
    MODALITY_IMU,
    MODALITY_TOF,
    MODALITY_UWB,
    MODALITY_VIO,
    PAYLOAD_KEYS,
)
from liquidloc.common.validation import is_bool_like, is_integer, is_real, is_string_like
from liquidloc.protocol.event_schema import validate_event_sequence
from liquidloc.protocol.sensor_contract import get_required_payload_fields  # 从协议层获取字段名，防止硬编码漂移。

__all__ = (
    "build_imu_events",
    "build_uwb_events",
    "build_vio_events",
    "build_flow_events",
    "build_tof_events",
    "merge_and_finalize_events",
)

# 模块加载时断言：event_builder 中硬编码的字段名必须与 sensor_contract 冻结合同一致。
# 这是"受控双源"模式：event_builder 仍硬编码字段名用于构造，但在加载时校验一致性。
_REQUIRED_PAYLOAD_FIELDS = get_required_payload_fields()
if set(_REQUIRED_PAYLOAD_FIELDS.get(MODALITY_IMU, ())) != {"ax", "ay", "gz"}:
    raise AssertionError(
        f"event_builder IMU fields {{ax, ay, gz}} != sensor_contract {_REQUIRED_PAYLOAD_FIELDS.get(MODALITY_IMU)}"
    )
if set(_REQUIRED_PAYLOAD_FIELDS.get(MODALITY_UWB, ())) != {"anchor_id", "range", "valid", "quality"}:
    raise AssertionError(
        f"event_builder UWB fields {{anchor_id, range, valid, quality}} != sensor_contract {_REQUIRED_PAYLOAD_FIELDS.get(MODALITY_UWB)}"
    )
# 铁律 3 (Stage A1 下游修复, 2026-07-23): VIO 字段从 6 项缩减到 4 项 — 删 tracked_features / reproj_err.
# 单目 VIO 紧耦合只输出位姿增量 (dx, dy, dyaw) 与质量指示 (quality), 不再输出特征点跟踪数与重投影误差.
if set(_REQUIRED_PAYLOAD_FIELDS.get(MODALITY_VIO, ())) != {"dx", "dy", "dyaw", "quality"}:
    raise AssertionError(
        f"event_builder VIO fields {{dx, dy, dyaw, quality}} != sensor_contract {_REQUIRED_PAYLOAD_FIELDS.get(MODALITY_VIO)}"
    )
if set(_REQUIRED_PAYLOAD_FIELDS.get(MODALITY_FLOW, ())) != {"dx", "dy", "quality"}:
    raise AssertionError(
        f"event_builder FLOW fields {{dx, dy, quality}} != sensor_contract {_REQUIRED_PAYLOAD_FIELDS.get(MODALITY_FLOW)}"
    )
if set(_REQUIRED_PAYLOAD_FIELDS.get(MODALITY_TOF, ())) != {"range", "quality"}:
    raise AssertionError(
        f"event_builder TOF fields {{range, quality}} != sensor_contract {_REQUIRED_PAYLOAD_FIELDS.get(MODALITY_TOF)}"
    )


def _coerce_valid(raw_valid: Any) -> bool:
    """把原始 valid 字段安全转成布尔值，防御字符串输入。

    参数：
        raw_valid: 待转换的原始值，可以是布尔、数值或字符串类型。

    返回：
        bool: 转换后的布尔值。

    说明：
        bool("false") 在 Python 中返回 True，因此需要显式处理字符串。
    """
    if is_bool_like(raw_valid):
        return bool(raw_valid)
    if is_real(raw_valid):
        if not isfinite(raw_valid):
            return False
        return bool(raw_valid)
    if is_string_like(raw_valid):
        normalized = str(raw_valid).strip().lower()
        if not normalized:
            return False
        try:
            float_val = float(normalized)
            if not isfinite(float_val):
                return False
            return bool(float_val)
        except (ValueError, TypeError):
            pass
        return normalized not in ('false', 'f', 'no', 'none', 'null')
    return bool(raw_valid)


def _require_field(row: dict[str, Any], key: str, *, modality: str) -> Any:
    """从映射行中读取必需字段，缺失时抛出 ValueError。

    参数：
        row: 映射后的行字典。
        key: 字段名。
        modality: 模态名称，用于构造错误消息。

    返回：
        Any: 字段值。

    异常：
        ValueError: 字段缺失或值为 None 时抛出。

    说明：
        field_mapper 对缺失字段补 None 占位，本函数将 None 占位转为明确的
        ValueError，避免下游 float(None) 等隐式 TypeError。
    """
    value = row.get(key)
    if value is None:
        raise ValueError(f"{modality} event requires field '{key}' but it is missing")
    return value


def _require_integer_field(row: dict[str, Any], key: str, *, modality: str) -> int:
    """读取必需整数语义字段，拒绝非整数数值的静默截断。"""
    value = _require_field(row, key, modality=modality)
    if is_integer(value):
        return int(value)
    if is_string_like(value):
        normalized = str(value).strip()
        if normalized and normalized.lstrip("+-").isdigit():
            return int(normalized)
    raise ValueError(f"{modality} event field '{key}' must be an integer, got {value!r}")


def _base_event(
    *,
    t: float,
    modality: str,
    scene_id: str,
    seq_id: str,
    source_t: Optional[float] = None,
) -> dict[str, Any]:
    """创建所有模态共用的事件外壳。

    参数：
        t：事件时间戳，用于排序和后续 dt 计算。
        modality：模态标签，用来标识这个事件属于哪个流。
        scene_id：写入元数据的场景编号。
        seq_id：写入元数据的序列编号。
        source_t：可选的原始来源时间戳，用于溯源。

    返回：
        dict[str, Any]: 一个只填好了公共元数据、载荷位还未填满的事件字典。
    """
    resolved_source_t = float(t) if source_t is None else float(source_t)
    return {
        't': float(t),
        'dt': 0.0,
        'modality': modality,
        'meta': {'scene_id': scene_id, 'seq_id': seq_id, 'source_t': resolved_source_t},
        PAYLOAD_KEYS[MODALITY_IMU]: None,
        PAYLOAD_KEYS[MODALITY_UWB]: None,
        PAYLOAD_KEYS[MODALITY_VIO]: None,
        PAYLOAD_KEYS[MODALITY_FLOW]: None,
        PAYLOAD_KEYS[MODALITY_TOF]: None,
    }


def build_imu_events(
    imu_table: Optional[list[dict[str, Any]]],
    scene_id: str,
    seq_id: str,
) -> list[dict[str, Any]]:
    """根据标准化后的 IMU 表构造 IMU 事件。

    参数：
        imu_table：标准化后的 IMU 行列表，如果没有行也可以传空值。
        scene_id：写入每个输出事件的场景编号。
        seq_id：写入每个输出事件的序列编号。

    返回：
        list[dict[str, Any]]: 已经构造好的 IMU 事件列表，等待后续和其他模态合并。

    异常：
        ValueError: 缺失必需字段时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"scene_id": scene_id, "seq_id": seq_id, "imu_table_rows": len(imu_table) if imu_table else 0}, "build_imu_events 入口参数")
    imu_events = []
    for row in imu_table or []:
        event = _base_event(
            t=_require_field(row, 'timestamp', modality=MODALITY_IMU),
            modality=MODALITY_IMU,
            scene_id=scene_id,
            seq_id=seq_id,
            source_t=row.get('source_t'),
        )
        # 构造 IMU payload，缺失字段补零并生成 missing_mask。
        # missing_mask 与 sensors.yaml imu_fields [ax, ay, gz] 一一对应，
        # 1 表示缺失（补零），0 表示真实测量值。
        _imu_ax = row.get('ax')
        _imu_ay = row.get('ay')
        _imu_gz = row.get('gz')
        _imu_missing_mask = [
            1 if _imu_ax is None else 0,
            1 if _imu_ay is None else 0,
            1 if _imu_gz is None else 0,
        ]
        event[PAYLOAD_KEYS[MODALITY_IMU]] = {
            'ax': float(_imu_ax if _imu_ax is not None else 0.0),
            'ay': float(_imu_ay if _imu_ay is not None else 0.0),
            'gz': float(_imu_gz if _imu_gz is not None else 0.0),
            'missing_mask': _imu_missing_mask,
        }
        imu_events.append(event)
    return imu_events


def build_uwb_events(
    uwb_table: Optional[list[dict[str, Any]]],
    scene_id: str,
    seq_id: str,
) -> list[dict[str, Any]]:
    """根据标准化后的 UWB 表构造 UWB 事件。

    参数：
        uwb_table：标准化后的 UWB 行列表，如果没有行也可以传空值。
        scene_id：写入每个输出事件的场景编号。
        seq_id：写入每个输出事件的序列编号。

    返回：
        list[dict[str, Any]]: 已经构造好的 UWB 事件列表，等待后续和其他模态合并。

    异常：
        ValueError: 缺失必需字段时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"scene_id": scene_id, "seq_id": seq_id, "uwb_table_rows": len(uwb_table) if uwb_table else 0}, "build_uwb_events 入口参数")
    uwb_events = []
    for row in uwb_table or []:
        event = _base_event(
            t=_require_field(row, 'timestamp', modality=MODALITY_UWB),
            modality=MODALITY_UWB,
            scene_id=scene_id,
            seq_id=seq_id,
            source_t=row.get('source_t'),
        )
        event[PAYLOAD_KEYS[MODALITY_UWB]] = {
            'anchor_id': _require_field(row, 'anchor_id', modality=MODALITY_UWB),
            'range': float(_require_field(row, 'range', modality=MODALITY_UWB)),
            'valid': _coerce_valid(_require_field(row, 'valid', modality=MODALITY_UWB)),
            'quality': float(_require_field(row, 'quality', modality=MODALITY_UWB)),
        }
        uwb_events.append(event)
    return uwb_events


def build_vio_events(
    vio_table: Optional[list[dict[str, Any]]],
    scene_id: str,
    seq_id: str,
) -> list[dict[str, Any]]:
    """根据标准化后的 VIO 表构造 VIO 事件。

    参数：
        vio_table：标准化后的 VIO 行列表，如果没有行也可以传空值。
        scene_id：写入每个输出事件的场景编号。
        seq_id：写入每个输出事件的序列编号。

    返回：
        list[dict[str, Any]]: 已经构造好的 VIO 事件列表，等待后续和其他模态合并。

    异常：
        ValueError: 缺失必需字段时抛出。

    说明：
        若行携带 ``missing_mask``（如由 ``_flow_rows_as_vio_rows`` 转换的
        flow 行），则原样透传到 VIO payload，使下游能区分真实值和补零。
        真实 VIO 行所有字段必填（``_require_field`` 对 None 抛错），
        missing_mask 默认全 0，与 IMU payload 的 missing_mask 约定一致，
        遵守 sensors.yaml feature_missing_policy.require_missing_mask: true。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"scene_id": scene_id, "seq_id": seq_id, "vio_table_rows": len(vio_table) if vio_table else 0}, "build_vio_events 入口参数")
    vio_events = []
    for row in vio_table or []:
        event = _base_event(
            t=_require_field(row, 'timestamp', modality=MODALITY_VIO),
            modality=MODALITY_VIO,
            scene_id=scene_id,
            seq_id=seq_id,
            source_t=row.get('source_t'),
        )
        # 读取行级 missing_mask（flow 转换行携带）；真实 VIO 行所有字段必填，默认全 0。
        # 与 IMU payload 的 missing_mask 列表约定一致，遵守 sensors.yaml
        # feature_missing_policy.require_missing_mask: true。
        _row_missing_mask = row.get('missing_mask')
        if _row_missing_mask is None:
            _row_missing_mask = [0, 0, 0, 0]  # 真实 VIO 行所有字段都存在 (铁律 3: 4 字段)
        event[PAYLOAD_KEYS[MODALITY_VIO]] = {
            'dx': float(_require_field(row, 'dx', modality=MODALITY_VIO)),
            'dy': float(_require_field(row, 'dy', modality=MODALITY_VIO)),
            'dyaw': float(_require_field(row, 'dyaw', modality=MODALITY_VIO)),
            'quality': float(_require_field(row, 'quality', modality=MODALITY_VIO)),
            # 铁律 3 (Stage A1 下游修复, 2026-07-23): 删除 tracked_features / reproj_err.
            # sim_materializer 已不再输出此两字段, 紧耦合 VIO 只用 dx/dy/dyaw/quality.
            'missing_mask': _row_missing_mask,
        }
        vio_events.append(event)
    return vio_events


def build_flow_events(
    flow_table: Optional[list[dict[str, Any]]],
    scene_id: str,
    seq_id: str,
) -> list[dict[str, Any]]:
    """根据标准化后的 flow 表构造光流/流量类事件（UTIL 数据集）。

    flow 模态的字段结构与 VIO 类似但更精简，通常只包含 dx/dy/quality，
    缺少 dyaw、tracked_features、reproj_err。这些缺失字段在载荷中
    用合理默认值填充，下游桥接函数（_flow_rows_as_vio_rows）会
    将其转为完整 VIO 行。

    参数：
        flow_table：标准化后的 flow 行列表，如果没有行也可以传空值。
        scene_id：写入每个输出事件的场景编号。
        seq_id：写入每个输出事件的序列编号。

    返回：
        list[dict[str, Any]]: 已经构造好的 flow 事件列表，等待后续和其他模态合并或桥接。

    异常：
        ValueError: 缺失必需字段时抛出。
    """
    flow_events = []
    for row in flow_table or []:
        event = _base_event(
            t=_require_field(row, 'timestamp', modality=MODALITY_FLOW),
            modality=MODALITY_FLOW,
            scene_id=scene_id,
            seq_id=seq_id,
            source_t=row.get('source_t'),
        )
        event[PAYLOAD_KEYS[MODALITY_FLOW]] = {
            'dx': float(_require_field(row, 'dx', modality=MODALITY_FLOW)),
            'dy': float(_require_field(row, 'dy', modality=MODALITY_FLOW)),
            'quality': float(_require_field(row, 'quality', modality=MODALITY_FLOW)),
        }
        flow_events.append(event)
    return flow_events


def build_tof_events(
    tof_table: Optional[list[dict[str, Any]]],
    scene_id: str,
    seq_id: str,
) -> list[dict[str, Any]]:
    """根据标准化后的 ToF 表构造 ToF 测距类事件（UTIL 数据集）。

    ToF 模态的字段结构与 UWB 类似但更精简，只包含 range/quality，
    缺少 anchor_id 和 valid。ToF 测距无法直接桥接为 UWB 事件
    （缺少 anchor_id），当前 prepare_pipeline 会显式拒绝 ToF 数据。

    参数：
        tof_table：标准化后的 ToF 行列表，如果没有行也可以传空值。
        scene_id：写入每个输出事件的场景编号。
        seq_id：写入每个输出事件的序列编号。

    返回：
        list[dict[str, Any]]: 已经构造好的 ToF 事件列表，等待后续和其他模态合并或桥接。

    异常：
        ValueError: 缺失必需字段时抛出。
    """
    tof_events = []
    for row in tof_table or []:
        event = _base_event(
            t=_require_field(row, 'timestamp', modality=MODALITY_TOF),
            modality=MODALITY_TOF,
            scene_id=scene_id,
            seq_id=seq_id,
            source_t=row.get('source_t'),
        )
        event[PAYLOAD_KEYS[MODALITY_TOF]] = {
            'range': float(_require_field(row, 'range', modality=MODALITY_TOF)),
            'quality': float(_require_field(row, 'quality', modality=MODALITY_TOF)),
        }
        tof_events.append(event)
    return tof_events


def merge_and_finalize_events(
    event_groups: Iterable[Iterable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """合并多个模态事件组，排序并重新计算 dt。

    参数：
        event_groups：事件组的可迭代对象，每个子组对应一种模态或一个来源组。

    返回：
        list[dict[str, Any]]: 一个按时间排序且已经校验过的事件列表。

    设计说明：
        - 事件排序优先级：时间戳 > 组顺序 > 组内顺序
        - 首条事件 dt 为 0.0，后续事件 dt 为相邻时间戳差值
        - 合并后调用 validate_event_sequence 进行协议校验
    """
    from liquidloc.common.tee_logger import print_dict
    _event_groups_list = list(event_groups) if event_groups else []
    print_dict({
        "group_count": len(_event_groups_list),
        "group_sizes": [len(list(g)) for g in _event_groups_list],
    }, "merge_and_finalize_events 入口参数")
    all_events: list[dict[str, Any]] = []
    for group_order, group in enumerate(event_groups):
        for event_order, event in enumerate(group):
            normalized_event = deepcopy(event)
            normalized_event['_merge_group_order'] = group_order
            normalized_event['_merge_event_order'] = event_order
            all_events.append(normalized_event)
    all_events.sort(
        key=lambda item: (
            float(item['t']),
            int(item.get('_merge_group_order', 0)),
            int(item.get('_merge_event_order', 0)),
        )
    )

    prev_t = None
    for event in all_events:
        current_t = float(event['t'])
        event['dt'] = 0.0 if prev_t is None else current_t - prev_t
        prev_t = current_t
        event.pop('_merge_group_order', None)
        event.pop('_merge_event_order', None)

    validate_event_sequence(all_events)
    return all_events
