"""通用统计工具函数。

职责：
    提供跨模块复用的统计计算函数，避免上层模块反向依赖 metrics 内部实现。

上游依赖：
    - 无内部模块依赖（仅依赖标准库 math）

下游调用者：
    - liquidloc.metrics.metric_runner — 复用 compute_trace_correlation
    - liquidloc.analysis.calibration_analysis — 复用 compute_trace_correlation
"""

from __future__ import annotations

import math


def compute_trace_correlation(lhs_trace, rhs_trace, *, lhs_name):
    """计算两条等长轨迹之间的 Pearson 相关系数，结果截断到 [-1, 1]。

    此函数原位于 metrics.metric_runner._compute_trace_correlation，
    因 analysis 层不应依赖 metrics 内部实现，已迁至 common 层作为公共入口。

    参数：
        lhs_trace: 左侧轨迹数值列表，可含 None 表示缺失。
        rhs_trace: 右侧轨迹数值列表（误差轨迹），可含 None 表示缺失。
        lhs_name: 左侧轨迹名称，仅用于错误信息。

    返回：
        float: 相关系数，范围 [-1, 1]；样本不足时返回 0.0。

    异常：
        ValueError: 两条轨迹长度不一致或包含非数值非 None 值时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"lhs_trace": lhs_trace, "rhs_trace": rhs_trace, "lhs_name": lhs_name}, "compute_trace_correlation 入口参数")
    if len(lhs_trace) != len(rhs_trace):
        raise ValueError(f"{lhs_name} and error_trace must have the same length")

    aligned_lhs = []
    aligned_error = []
    for lhs_value, error_value in zip(lhs_trace, rhs_trace):
        if lhs_value is None or error_value is None:
            continue
        try:
            lhs_value = float(lhs_value)
            error_value = float(error_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{lhs_name} and error_trace must contain numeric values or None") from exc
        if not (math.isfinite(lhs_value) and math.isfinite(error_value)):
            continue
        aligned_lhs.append(lhs_value)
        aligned_error.append(error_value)

    if len(aligned_lhs) < 2:
        return 0.0

    lhs_mean = math.fsum(aligned_lhs) / len(aligned_lhs)
    error_mean = math.fsum(aligned_error) / len(aligned_error)
    covariance = math.fsum(
        (lhs_value - lhs_mean) * (error_value - error_mean)
        for lhs_value, error_value in zip(aligned_lhs, aligned_error)
    )
    lhs_variance = math.fsum((lhs_value - lhs_mean) ** 2 for lhs_value in aligned_lhs)
    error_variance = math.fsum((error_value - error_mean) ** 2 for error_value in aligned_error)
    denominator = math.sqrt(lhs_variance * error_variance)
    if denominator == 0.0:
        return 0.0
    return max(-1.0, min(1.0, covariance / denominator))
