"""案例选择编排模块。

把 eval_pipeline 里的最优方法选择、案例分组规则生成和空选例容器
集中到这个模块，减轻 eval_pipeline 的体积。
实际的案例抽取逻辑在 case_selector.py 里。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from liquidloc.common.constants import CASE_GROUP_BOUNDARY, CASE_GROUP_FAILURE, CASE_GROUP_MAIN  # D9 单源：案例分组名常量。
from liquidloc.common.types import MetricDirection, _METRIC_DIRECTIONS  # 指标方向类型别名及合法方向集合（单一真相源，与 metric_schema.py 共享）。
from liquidloc.common.validation import is_numeric
from liquidloc.protocol.metric_schema import get_metric_meta


def _direction_aware_sort_value(raw_value: float, *, direction: MetricDirection) -> float:
    """将指标值转换为统一按升序比较的排序值。

    lower_is_better: 原值越小越好，直接用原值排序（升序最小即最优）。
    higher_is_better: 原值越大越好，取负后排序（升序最小即原值最大即最优）。
    neutral: 不区分优劣，保持原值（不影响排序但保留确定性）。

    Args:
        raw_value: 待转换的原始指标值，必须是有限浮点数（NaN/inf 会破坏 min/sorted 比较）。
        direction: 指标优劣方向，必须是 ``MetricDirection`` 之一
            （``lower_is_better`` / ``higher_is_better`` / ``neutral``）。

    Returns:
        转换后的排序值，升序最小即最优。

    Raises:
        ValueError: ``direction`` 不在合法方向集合中，或 ``raw_value`` 非有限（NaN/inf）时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "raw_value_type": type(raw_value).__name__,
        "direction": direction,
    }, "_direction_aware_sort_value 入口参数")
    if direction not in _METRIC_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {sorted(_METRIC_DIRECTIONS)}, got {direction!r}"
        )
    if not math.isfinite(raw_value):
        raise ValueError(f"raw_value must be finite, got {raw_value}")
    if direction == 'higher_is_better':
        return -raw_value
    return raw_value


def select_best_method(
    method_summary: Mapping[str, Mapping[str, Any]],
    *,
    priority_metrics: list[str],
) -> str:
    """按优先指标选择最优方法名。

    Args:
        method_summary: 方法名到统计摘要的映射，里面至少要有 `mean_rmse`。
        priority_metrics: 用来优先比较的方法级均值指标列表。

    Returns:
        排序后最优的方法名。

    Raises:
        ValueError: method_summary 为空、priority_metrics 为空、priority_metrics
            中存在未在指标 schema 注册的指标名、或某个方法摘要缺少
            ``mean_{metric}`` 必填键时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "method_summary_size": len(method_summary) if hasattr(method_summary, "__len__") else None,
        "priority_metrics": list(priority_metrics) if isinstance(priority_metrics, (list, tuple)) else None,
    }, "select_best_method 入口参数")
    if not method_summary:
        raise ValueError('method_summary must be non-empty')
    if not priority_metrics:
        raise ValueError('priority_metrics must be non-empty')

    metric_meta = get_metric_meta()
    # 预校验优先指标均已注册，避免在排序键里静默回退到方向默认值（D6/D9 单源真相）。
    for metric_name in priority_metrics:
        if metric_name not in metric_meta:
            raise ValueError(
                f"metric '{metric_name}' is not registered in metric schema"
            )

    def _summary_sort_key(item: tuple[str, Mapping[str, Any]]) -> tuple[Any, ...]:
        method_name, summary = item
        sort_values: list[Any] = []
        for metric_name in priority_metrics:
            mean_key = f'mean_{metric_name}'
            if mean_key not in summary:  # 必填键缺失按 value contract 报 ValueError，不静默回退。
                raise ValueError(
                    f"method '{method_name}' summary is missing required key '{mean_key}'"
                )
            value = summary[mean_key]
            if is_numeric(value):
                try:  # float() 可能抛 OverflowError（如超大 int），需显式守卫（D5 数值安全）。
                    float_value = float(value)
                except (TypeError, ValueError, OverflowError):
                    sort_values.append(math.inf)
                    continue
                if not math.isfinite(float_value):  # NaN/inf 走最差值回退，避免破坏 min 比较。
                    sort_values.append(math.inf)
                    continue
                direction = metric_meta[metric_name]['direction']  # 已预校验注册，直接索引取冻结方向。
                sort_values.append(_direction_aware_sort_value(float_value, direction=direction))
            else:
                sort_values.append(math.inf)
        sort_values.append(str(method_name))
        return tuple(sort_values)

    return min(method_summary.items(), key=_summary_sort_key)[0]


def select_case_rules(case_catalog: dict[str, dict[str, Any]], *, priority_metrics: list[str]) -> dict[str, list[str]]:
    """根据优先指标把案例分成主样本、失败样本和边界样本。

    Args:
        case_catalog: case_ref 到案例记录的映射，记录必须包含 ``scene_id`` / ``seq_id``
            必填键（由 eval_pipeline._build_case_record 保证），``task_id`` / ``repeat_id``
            为可选键。
        priority_metrics: 用于排序的优先指标列表，所有指标必须在指标 schema 中注册。

    Returns:
        包含 ``main_cases`` / ``failure_cases`` / ``boundary_cases`` 三个键的字典，
        分别对应最优、最差和中位分组。

    Raises:
        ValueError: case_catalog 为空、priority_metrics 为空、或 priority_metrics 中
            存在未在指标 schema 注册的指标名时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "case_catalog_size": len(case_catalog) if hasattr(case_catalog, "__len__") else None,
        "priority_metrics": list(priority_metrics) if isinstance(priority_metrics, (list, tuple)) else None,
    }, "select_case_rules 入口参数")
    candidate_items = list(case_catalog.items())
    if not candidate_items:
        raise ValueError('case_catalog must be non-empty')
    if not priority_metrics:
        raise ValueError('priority_metrics must be non-empty')

    metric_meta = get_metric_meta()
    # 预校验优先指标均已注册，避免在排序键里静默回退到方向默认值（D6/D9 单源真相）。
    for metric_name in priority_metrics:
        if metric_name not in metric_meta:
            raise ValueError(
                f"metric '{metric_name}' is not registered in metric schema"
            )

    def _metric_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[Any, ...]:
        _, case_record = item
        sort_values: list[Any] = []
        for metric_name in priority_metrics:
            metric_value = case_record.get(metric_name)
            if is_numeric(metric_value):
                try:  # float() 可能抛 OverflowError（如超大 int），需显式守卫（D5 数值安全）。
                    float_value = float(metric_value)
                except (TypeError, ValueError, OverflowError):
                    sort_values.append(math.inf)
                    continue
                if not math.isfinite(float_value):  # NaN/inf 走最差值回退，避免破坏 sorted 比较。
                    sort_values.append(math.inf)
                    continue
                direction = metric_meta[metric_name]['direction']  # 已预校验注册，直接索引取冻结方向。
                sort_values.append(_direction_aware_sort_value(float_value, direction=direction))
            else:
                sort_values.append(math.inf)
        sort_values.extend(
            [
                # task_id / repeat_id 在 _build_case_record 中是可选字段（仅存在时写入），保留 .get 回退；
                # scene_id / seq_id 是必填字段，用 [] 直接访问以暴露缺键的合同违约，而非静默回退到空串（D3）。
                str(case_record.get('task_id', '')),
                str(case_record['scene_id']),
                str(case_record['seq_id']),
                str(case_record.get('repeat_id', '')),
            ]
        )
        return tuple(sort_values)

    items_by_metric_key: dict[tuple[Any, ...], list[str]] = {}
    for case_ref, case_record in candidate_items:
        items_by_metric_key.setdefault(_metric_sort_key((case_ref, case_record)), []).append(case_ref)

    ordered_metric_keys = sorted(items_by_metric_key)
    main_case_refs = sorted(items_by_metric_key[ordered_metric_keys[0]])
    failure_case_refs = sorted(items_by_metric_key[ordered_metric_keys[-1]])
    boundary_case_refs = sorted(items_by_metric_key[ordered_metric_keys[len(ordered_metric_keys) // 2]])
    return {
        CASE_GROUP_MAIN: list(main_case_refs),
        CASE_GROUP_FAILURE: list(failure_case_refs),
        CASE_GROUP_BOUNDARY: list(boundary_case_refs),
    }


def build_empty_selected_cases() -> dict[str, list[dict[str, Any]]]:
    """返回不含正式案例的空 selected_cases 标准容器。"""
    return {
        CASE_GROUP_MAIN: [],
        CASE_GROUP_FAILURE: [],
        CASE_GROUP_BOUNDARY: [],
    }
