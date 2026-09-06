"""真值（GT）行归一化、时间对齐与锚点位置解析的共享工具。

本模块收拢了原先分散在 train_pipeline、metric_runner、sim_materializer、
ekf_core 和 feature_builder 中的重复实现，提供唯一的规范版本。

上游依赖：
    - liquidloc.common.angle_utils   — 角度归一化与角差计算
    - liquidloc.common.constants     — 默认阈值常量
    - liquidloc.common.validation    — 类型判断工具

下游调用者：
    - liquidloc.pipelines.train_pipeline
    - liquidloc.pipelines.core_pipeline
    - liquidloc.metrics.metric_runner
    - liquidloc.dataio.sim_materializer
    - liquidloc.estimators.ekf_core
    - liquidloc.models.features.feature_builder
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from liquidloc.common.angle_utils import angle_delta_rad, wrap_angle_rad
from liquidloc.common.constants import DEFAULT_THRESHOLDS
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_integer, is_string_like

import numpy as np

__all__ = (
    "normalize_gt_rows",
    "align_ground_truth",
    "resolve_anchor_position",
    "GT_TIME_TOLERANCE",
)

GT_TIME_TOLERANCE: float = float(DEFAULT_THRESHOLDS['time_tolerance'])
_GT_TIME_TOLERANCE: float = GT_TIME_TOLERANCE  # 向后兼容别名。


def normalize_gt_rows(
    gt_rows: Any,
    *,
    yaw_default: float | None = 0.0,
) -> list[dict[str, float]]:
    """归一化真值行列表，确保每行包含 timestamp/px/py/yaw 字段。

    Args:
        gt_rows: 原始真值行列表，每行应是 Mapping 类型。
        yaw_default: yaw 字段缺失时的默认值。训练链路通常传 0.0；
            指标链路可传 None 以便下游 yaw 指标过滤缺失行。

    Returns:
        归一化后的真值行列表，按 timestamp 升序排列。

    Raises:
        TypeError: 某行不是 Mapping 类型。
        KeyError: 某行缺少 timestamp/px/py 字段（错误消息包含行号）。
        ValueError: 某行数值字段为 NaN 或 Inf。
    """
    normalized_rows: list[dict[str, float]] = []
    for row_index, row in enumerate(list(gt_rows or [])):
        if not isinstance(row, Mapping):
            raise TypeError(f'gt_rows[{row_index}] must be a mapping')
        timestamp = row.get('timestamp', row.get('t'))
        if timestamp is None:
            raise KeyError(f'gt_rows[{row_index}] must contain timestamp')
        yaw_value = row.get('yaw', yaw_default)
        if yaw_value is not None:
            yaw_value = coerce_finite_scalar(yaw_value, name=f"gt_rows[{row_index}].yaw")
        normalized_rows.append({
            'timestamp': coerce_finite_scalar(timestamp, name=f"gt_rows[{row_index}].timestamp"),
            'px': coerce_finite_scalar(row['px'], name=f"gt_rows[{row_index}].px"),
            'py': coerce_finite_scalar(row['py'], name=f"gt_rows[{row_index}].py"),
            'yaw': yaw_value,
        })
    normalized_rows.sort(key=lambda item: item['timestamp'])
    return normalized_rows


def _trailing_ground_truth_tolerance(
    gt_rows: list[dict[str, float]],
    *,
    prefer_source_t: bool = False,
) -> float:
    """计算尾部真值的容差，取默认容差和最后两个真值间隔的较大值。"""
    if len(gt_rows) < 2:
        return _GT_TIME_TOLERANCE
    trailing_interval = _resolve_align_time(gt_rows[-1], prefer_source_t) - _resolve_align_time(gt_rows[-2], prefer_source_t)
    if not np.isfinite(trailing_interval):
        return _GT_TIME_TOLERANCE
    return max(_GT_TIME_TOLERANCE, trailing_interval)


def _resolve_align_time(row: Mapping[str, Any], prefer_source_t: bool) -> float:
    """选择对齐用的时间键：prefer_source_t 且行含 source_t 时用 source_t，否则用 timestamp。

    这是 align_ground_truth 时间轴选择的核心辅助函数，集中处理 source_t
    优先策略，避免在每个比较点重复实现。返回 float 形式的时间值。
    """
    if prefer_source_t:
        source_t = row.get('source_t')
        if source_t is not None:
            return float(source_t)
    return float(row['timestamp'])


def align_ground_truth(
    gt_rows: list[dict[str, float]],
    timestamp: float,
    *,
    prefer_source_t: bool = False,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    """将给定时间戳与真值行列表对齐，支持精确匹配、线性插值和尾部容差。

    对齐策略：
    1. 精确匹配：时间戳与某真值行的时间差 <= _GT_TIME_TOLERANCE
    2. 线性插值：时间戳落在两个真值行之间
    3. 尾部容差：时间戳在最后一个真值行之后，但间隔 <= 尾部容差
    4. 超出范围：返回 None

    时间轴选择：
        - prefer_source_t=False（默认）：使用每行的 ``timestamp`` 字段做对齐，
          行为与历史版本完全一致。
        - prefer_source_t=True：若行包含非 None 的 ``source_t`` 字段，则优先用
          ``source_t`` 做对齐比较；否则回退到 ``timestamp``。这用于 core_pipeline
          铺叠后 GT 行的场景，避免 timestamp 与 source_t 时间轴不匹配导致的对齐漂移。
          注意：返回的 aligned_gt_timestamp 和 support_timestamps 仍取行内的
          ``timestamp`` 字段，保持消费者合同不变；只有 time_gap 和对齐比较
          使用 source_t（若可用）。

    Args:
        gt_rows: 已排序的真值行列表。
        timestamp: 待对齐的时间戳（prefer_source_t=True 时按 source_t 语义解释）。
        prefer_source_t: 是否优先用 source_t 做对齐，默认 False 保持向后兼容。

    Returns:
        (aligned_gt, alignment_info) 元组。
        aligned_gt 为对齐后的真值字典，或None（无法对齐）。
        alignment_info 包含 mode/time_gap/support_timestamps 等审计信息。

        aligned_gt 字典包含以下键：
            - timestamp (float): 对齐后的时间戳。
            - px (float): x 坐标。
            - py (float): y 坐标。
            - yaw (float | None): 航向角（弧度），缺失时为 None。
    """
    if not gt_rows:
        return None, {'mode': 'missing_ground_truth', 'aligned_gt_timestamp': None, 'time_gap': None, 'support_timestamps': []}
    if len(gt_rows) == 1:
        only_row = gt_rows[0]
        only_align_t = _resolve_align_time(only_row, prefer_source_t)
        time_gap = abs(only_align_t - float(timestamp))
        if time_gap <= _GT_TIME_TOLERANCE:
            return (
                dict(only_row),
                {'mode': 'exact', 'aligned_gt_timestamp': float(only_row['timestamp']), 'time_gap': time_gap, 'support_timestamps': [float(only_row['timestamp'])]},
            )
        return None, {'mode': 'outside_single_ground_truth_timestamp', 'aligned_gt_timestamp': float(only_row['timestamp']), 'time_gap': time_gap, 'support_timestamps': [float(only_row['timestamp'])]}

    for row in gt_rows:
        row_align_t = _resolve_align_time(row, prefer_source_t)
        time_gap = abs(row_align_t - float(timestamp))
        if time_gap <= _GT_TIME_TOLERANCE:
            return (
                dict(row),
                {'mode': 'exact', 'aligned_gt_timestamp': float(row['timestamp']), 'time_gap': time_gap, 'support_timestamps': [float(row['timestamp'])]},
            )

    first_align_t = _resolve_align_time(gt_rows[0], prefer_source_t)
    if float(timestamp) < first_align_t:
        return None, {'mode': 'before_ground_truth_span', 'aligned_gt_timestamp': float(gt_rows[0]['timestamp']), 'time_gap': first_align_t - float(timestamp), 'support_timestamps': [float(gt_rows[0]['timestamp'])]}

    for lower_row, upper_row in zip(gt_rows, gt_rows[1:]):
        lower_t = _resolve_align_time(lower_row, prefer_source_t)
        upper_t = _resolve_align_time(upper_row, prefer_source_t)
        if lower_t <= float(timestamp) <= upper_t:
            span = upper_t - lower_t
            if span <= 0.0:
                return (
                    dict(lower_row),
                    {'mode': 'exact', 'aligned_gt_timestamp': float(lower_row['timestamp']), 'time_gap': abs(lower_t - float(timestamp)), 'support_timestamps': [float(lower_row['timestamp'])]},
                )
            alpha = (float(timestamp) - lower_t) / span
            # yaw 插值：两侧都有非None yaw 时才插值，否则设为 None。
            lower_yaw = lower_row.get('yaw')
            upper_yaw = upper_row.get('yaw')
            if lower_yaw is not None and upper_yaw is not None:
                interp_yaw = wrap_angle_rad(
                    float(lower_yaw) + angle_delta_rad(float(upper_yaw), float(lower_yaw)) * alpha
                )
            else:
                interp_yaw = None
            return (
                {
                    'timestamp': float(timestamp),
                    'px': lower_row['px'] + (upper_row['px'] - lower_row['px']) * alpha,
                    'py': lower_row['py'] + (upper_row['py'] - lower_row['py']) * alpha,
                    'yaw': interp_yaw,
                },
                {'mode': 'linear_interpolation', 'aligned_gt_timestamp': float(timestamp), 'time_gap': 0.0, 'support_timestamps': [float(lower_row['timestamp']), float(upper_row['timestamp'])]},
            )

    trailing_tolerance = _trailing_ground_truth_tolerance(gt_rows, prefer_source_t=prefer_source_t)
    last_align_t = _resolve_align_time(gt_rows[-1], prefer_source_t)
    trailing_gap = float(timestamp) - last_align_t
    if 0.0 < trailing_gap <= trailing_tolerance:
        return (
            dict(gt_rows[-1]),
            {'mode': 'trailing_nearest_within_tolerance', 'aligned_gt_timestamp': float(gt_rows[-1]['timestamp']), 'time_gap': trailing_gap, 'tolerance': trailing_tolerance, 'support_timestamps': [float(gt_rows[-1]['timestamp'])]},
        )
    return None, {'mode': 'after_ground_truth_span', 'aligned_gt_timestamp': float(gt_rows[-1]['timestamp']), 'time_gap': trailing_gap, 'tolerance': trailing_tolerance, 'support_timestamps': [float(gt_rows[-1]['timestamp'])]}


def resolve_anchor_position(
    anchor_id: Any,
    anchor_lookup: Mapping[Any, Any],
) -> tuple[float, float]:
    """根据 anchor_id 查找锚点位置，支持多种 ID 格式。

    查找策略：
    1. 直接匹配 anchor_id
    2. 整数 ID -> 尝试字符串和 Axx 格式
    3. Axx 字符串 -> 尝试整数和纯数字字符串格式
    4. 数字字符串 -> 尝试整数和 Axx 格式

    找到候选键后，还会校验坐标结构：必须是长度为 2 的可迭代对象，
    且不能是 str/bytes 类型。

    Note:
        Axx 格式（如 "A1"、"A2"）是 UWB 锚点 ID 的常见命名约定。
        本函数支持整数、字符串和 Axx 格式之间的自动转换查找。

    Args:
        anchor_id: 锚点 ID。
        anchor_lookup: 锚点查找表。

    Returns:
        (x, y) 锚点位置元组。

    Raises:
        ValueError: 找不到对应锚点位置或坐标结构不合法。
    """
    candidates = [anchor_id]
    if is_integer(anchor_id):
        candidates.append(str(anchor_id))
        candidates.append(f'A{anchor_id}')
    if is_string_like(anchor_id):
        if anchor_id.isdigit():
            int_key = int(anchor_id)
            a_key = f'A{anchor_id}'
            if int_key not in candidates:
                candidates.append(int_key)
            if a_key not in candidates:
                candidates.append(a_key)
        if anchor_id.startswith('A'):
            raw_index = anchor_id[1:]
            if raw_index.isdigit():
                int_key = int(raw_index)
                if int_key not in candidates:
                    candidates.append(int_key)
                if raw_index not in candidates:
                    candidates.append(raw_index)

    for candidate in candidates:
        if candidate in anchor_lookup:
            anchor_position = anchor_lookup[candidate]
            if isinstance(anchor_position, (str, bytes)):
                continue
            try:
                coords = list(anchor_position)
            except TypeError:
                continue
            if len(coords) != 2:
                continue  # 跳过非二维坐标。
            try:
                return float(coords[0]), float(coords[1])
            except (TypeError, ValueError):
                continue  # 坐标分量非数值时跳过，与 docstring 声明只抛 ValueError 保持一致。

    raise ValueError(f'anchor position unavailable for anchor_id={anchor_id!r}')
