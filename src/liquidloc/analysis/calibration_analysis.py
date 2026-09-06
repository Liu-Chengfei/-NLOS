"""
校准分析工具集：对齐四条 trace 并计算 trace-to-error 相关性。

本模块负责把 bias_trace、risk_trace、scaling_trace 和 error_trace 四条校准曲线
按索引对齐、过滤无效行，再计算各 trace 与 error_trace 之间的 Pearson 相关系数，
供 summary、analysis 和 plotting 复用。

上游依赖：
- 调用方传入已计算好的四条 trace（字典或列表形式）。

下游调用者：
- summary_builder、plotting 模块以及测试脚本。

核心变量：
- aligned_bias / aligned_risk / aligned_scaling / aligned_error：对齐后的数值列表。
- bias_alignment / risk_error_corr / corr_scaling_error：三条 trace 与 error 的相关系数。
"""

from __future__ import annotations  # 让参数化容器和联合类型注解在运行期不求值，与 analysis 同模块保持一致。

import math  # 用于有限性判断和平方根等基础数值运算。
from collections.abc import Iterable, Mapping  # 用抽象基类判断 trace 是映射还是可迭代对象，与同模块其他文件一致。
from typing import Any  # trace 的键值类型在归一化前无法收窄，用 Any 接住。

from liquidloc.common.statistics_utils import compute_trace_correlation  # 公共轨迹相关性计算，消除对 metrics 内部的反向依赖。


def _as_indexed_trace(  # 把 trace 统一转成 (index, value) 对列表。
    trace_name: str,
    trace: Mapping[Any, Any] | Iterable[Any],
) -> list[tuple[Any, Any]]:
    """把 trace 规范化成有序的 (index, value) 对列表。

    支持两种输入形式：
    - 字典：按 .items() 取出键值对，键即为索引。
    - 可迭代对象：用 enumerate 自动编号作为索引。

    Args:
        trace_name: trace 的名字，仅用于报错时定位是哪条 trace 出了问题。
        trace: 待规范化的 trace 对象，可以是字典或可迭代对象。

    Returns:
        list[tuple]: 有序的 (index, value) 对列表。

    Raises:
        TypeError: trace 是字符串/字节，或者不可迭代。
        ValueError: trace 为空。
    """
    if isinstance(trace, (str, bytes)):  # 字符串和字节虽然可迭代，但不是合法的 trace。
        raise TypeError(f"{trace_name} must be a non-string iterable")  # 直接拒绝字符串输入。
    if hasattr(trace, "items"):  # 如果有 items 方法，说明是字典风格的对象（保留鸭子类型以兼容现有测试 fixture）。
        items = list(trace.items())  # 把字典的键值对转成列表，键作为索引。
    else:  # 否则按可迭代对象处理。
        try:  # 尝试用 enumerate 给元素编号。
            items = list(enumerate(trace))  # 每个元素变成 (序号, 值) 的形式。
        except TypeError as exc:  # 如果不可迭代，就说明输入类型不对。
            raise TypeError(f"{trace_name} must be iterable") from exc  # 报错时指出哪条 trace 有问题。
    if not items:  # 空 trace 没有分析意义。
        raise ValueError(f"{trace_name} must not be empty")  # 空输入直接拒绝。
    return items  # 返回规范化后的 (index, value) 列表。


def align_traces(  # 对齐四条 trace 并过滤无效行。
    bias_trace: Mapping[Any, Any] | Iterable[Any],
    risk_trace: Mapping[Any, Any] | Iterable[Any],
    scaling_trace: Mapping[Any, Any] | Iterable[Any],
    error_trace: Mapping[Any, Any] | Iterable[Any],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """按索引对齐四条 trace，过滤掉含 None 或非有限值的行。

    以 bias_trace 的索引为基准，要求其余三条 trace 的索引集合完全一致。
    对齐后只保留四条 trace 在同一索引处都是有限数值的行。

    Args:
        bias_trace: 偏置 trace，作为索引基准。可以是字典或可迭代对象。
        risk_trace: 风险 trace，索引必须与 bias_trace 一一对应。
        scaling_trace: 缩放 trace，索引必须与 bias_trace 一一对应。
        error_trace: 误差 trace，索引必须与 bias_trace 一一对应。

    Returns:
        tuple[list[float], list[float], list[float], list[float]]:
            四个对齐后的数值列表，依次为 aligned_bias、aligned_risk、
            aligned_scaling、aligned_error。

    Raises:
        TypeError: 任意 trace 是字符串或不可迭代。
        ValueError: 任意 trace 为空、索引重复、或索引集合与 bias_trace 不一致。
    """
    bias_items = _as_indexed_trace("bias_trace", bias_trace)  # 把 bias_trace 规范化成 (index, value) 列表。
    risk_items = _as_indexed_trace("risk_trace", risk_trace)  # 把 risk_trace 规范化成 (index, value) 列表。
    scaling_items = _as_indexed_trace("scaling_trace", scaling_trace)  # 把 scaling_trace 规范化成 (index, value) 列表。
    error_items = _as_indexed_trace("error_trace", error_trace)  # 把 error_trace 规范化成 (index, value) 列表。
    expected_indices = [index for index, _ in bias_items]  # 从 bias_trace 提取基准索引序列。
    if len(set(expected_indices)) != len(expected_indices):  # 检查基准索引是否有重复。
        raise ValueError("bias_trace must not contain duplicated indices")  # 重复索引会导致对齐歧义。

    aligned_item_groups = [bias_items]  # 先把 bias 的 (index, value) 对放进对齐组列表。
    for trace_name, trace_items in (  # 依次处理其余三条 trace。
        ("risk_trace", risk_items),  # 风险 trace。
        ("scaling_trace", scaling_items),  # 缩放 trace。
        ("error_trace", error_items),  # 误差 trace。
    ):  # 遍历结束。
        trace_indices = [index for index, _ in trace_items]
        if len(set(trace_indices)) != len(trace_indices):  # 直接用集合大小判断重复，比间接比较 dict 去重后更清晰。
            raise ValueError(f"{trace_name} must not contain duplicated indices")  # 重复索引不允许。
        trace_lookup = dict(trace_items)  # 把当前 trace 转成字典，便于按索引查找。
        if set(trace_lookup.keys()) != set(expected_indices):  # 索引集合必须与基准完全一致。
            raise ValueError(f"{trace_name} must have the same length and index as bias_trace")  # 索引不匹配就报错。
        aligned_item_groups.append([(index, trace_lookup[index]) for index in expected_indices])  # 按基准索引顺序重排当前 trace。

    aligned_bias = []  # 收集对齐后的 bias 数值。
    aligned_risk = []  # 收集对齐后的 risk 数值。
    aligned_scaling = []  # 收集对齐后的 scaling 数值。
    aligned_error = []  # 收集对齐后的 error 数值。
    for (_, bias_value), (_, risk_value), (_, scaling_value), (_, error_value) in zip(*aligned_item_groups):  # 逐行对齐四条 trace。
        row_values = (bias_value, risk_value, scaling_value, error_value)  # 把当前行的四个值打包。
        if any(value is None for value in row_values):  # 任意一个值为 None 就跳过该行。
            continue  # None 无法参与数值计算。
        try:  # 尝试把四个值都转成 float。
            numeric_values = [float(value) for value in row_values]  # 转成浮点数列表。
        except (TypeError, ValueError, OverflowError) as exc:  # 转换失败说明有非数值数据；OverflowError 守卫超大整数（如 float(10**1000)）。
            raise ValueError(  # 统一报错，指出四条 trace 都必须是数值或 None。
                "bias_trace, risk_trace, scaling_trace, and error_trace must contain numeric values or None"
            ) from exc  # 保留原始异常链。
        if not all(math.isfinite(value) for value in numeric_values):  # inf 和 nan 不适合做相关性分析。
            continue  # 跳过含非有限值的行。
        aligned_bias.append(numeric_values[0])  # 把 bias 值加入对齐列表。
        aligned_risk.append(numeric_values[1])  # 把 risk 值加入对齐列表。
        aligned_scaling.append(numeric_values[2])  # 把 scaling 值加入对齐列表。
        aligned_error.append(numeric_values[3])  # 把 error 值加入对齐列表。
    return aligned_bias, aligned_risk, aligned_scaling, aligned_error  # 返回四条对齐后的数值列表。


def compute_trace_correlations(  # 计算三条 trace 与 error 的相关系数。
    bias_trace: list[float],
    risk_trace: list[float],
    scaling_trace: list[float],
    error_trace: list[float],
) -> dict[str, float]:
    """从已对齐的数值 trace 计算 bias/risk/scaling 与 error 之间的 Pearson 相关系数。

    当任意一条 trace 为空时，返回全零字典，避免下游需要额外判空。

    Args:
        bias_trace: 已对齐的偏置 trace 数值列表。
        risk_trace: 已对齐的风险 trace 数值列表。
        scaling_trace: 已对齐的缩放 trace 数值列表。
        error_trace: 已对齐的误差 trace 数值列表。

    Returns:
        dict[str, float]: 包含三个相关系数的字典：
            - bias_alignment: bias 与 error 的相关系数。
            - risk_error_corr: risk 与 error 的相关系数。
            - corr_scaling_error: scaling 与 error 的相关系数。
    """
    if not bias_trace or not risk_trace or not scaling_trace or not error_trace:  # 任意一条 trace 为空，说明没有有效数据点。
        return {  # 返回全零字典，让下游不需要额外判空。
            "bias_alignment": 0.0,  # bias 与 error 相关性默认为 0。
            "risk_error_corr": 0.0,  # risk 与 error 相关性默认为 0。
            "corr_scaling_error": 0.0,  # scaling 与 error 相关性默认为 0。
        }  # 空数据字典结束。
    return {  # 正常情况下计算并返回三个相关系数。
        "bias_alignment": compute_trace_correlation(bias_trace, error_trace, lhs_name="bias"),  # 复用公共实现。
        "risk_error_corr": compute_trace_correlation(risk_trace, error_trace, lhs_name="risk"),  # 复用公共实现。
        "corr_scaling_error": compute_trace_correlation(scaling_trace, error_trace, lhs_name="scaling"),  # 复用公共实现。
    }  # 相关性字典结束。


def build_calibration_report(  # 对外入口：从四条原始 trace 构建统一校准报告。
    bias_trace: Mapping[Any, Any] | Iterable[Any],
    risk_trace: Mapping[Any, Any] | Iterable[Any],
    scaling_trace: Mapping[Any, Any] | Iterable[Any],
    error_trace: Mapping[Any, Any] | Iterable[Any],
) -> dict[str, float]:
    """从四条原始 trace 构建统一校准报告。

    先对齐四条 trace 并过滤无效行，再计算各 trace 与 error 的 Pearson 相关系数。
    这是本模块对外暴露的主入口，调用方只需传入原始 trace 即可拿到完整报告。

    Args:
        bias_trace: 偏置 trace，字典或可迭代对象。
        risk_trace: 风险 trace，字典或可迭代对象。
        scaling_trace: 缩放 trace，字典或可迭代对象。
        error_trace: 误差 trace，字典或可迭代对象。

    Returns:
        dict[str, float]: 包含三个相关系数的校准报告字典：
            - bias_alignment: bias 与 error 的相关系数。
            - risk_error_corr: risk 与 error 的相关系数。
            - corr_scaling_error: scaling 与 error 的相关系数。

    Raises:
        TypeError: 任意 trace 是字符串或不可迭代。
        ValueError: 任意 trace 为空、索引重复、或索引集合不一致。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "bias_trace_type": type(bias_trace).__name__,
            "risk_trace_type": type(risk_trace).__name__,
            "scaling_trace_type": type(scaling_trace).__name__,
            "error_trace_type": type(error_trace).__name__,
        },
        "build_calibration_report 入口参数",
        prefix="[analysis]",
    )
    aligned_bias, aligned_risk, aligned_scaling, aligned_error = align_traces(  # 先对齐四条 trace。
        bias_trace=bias_trace,  # 偏置 trace。
        risk_trace=risk_trace,  # 风险 trace。
        scaling_trace=scaling_trace,  # 缩放 trace。
        error_trace=error_trace,  # 误差 trace。
    )  # 对齐结束。
    return compute_trace_correlations(  # 再基于对齐结果计算相关系数。
        bias_trace=aligned_bias,  # 对齐后的 bias。
        risk_trace=aligned_risk,  # 对齐后的 risk。
        scaling_trace=aligned_scaling,  # 对齐后的 scaling。
        error_trace=aligned_error,  # 对齐后的 error。
    )  # 校准报告返回。
