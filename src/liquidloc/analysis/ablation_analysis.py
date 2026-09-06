"""
比较 full 基线和多个 ablation 变体的指标差异。

这个模块负责把 full 方法和各个消融方法按同一组指标对齐，
然后计算每个指标的绝对差和相对差，供 summary、analysis 和 plotting 复用。
它不重算原始指标，也不改协议字段，只做对比结果的统一整理。
"""

import math  # 用于 isfinite 检查，拒绝 inf/nan 污染 delta 和 relative_delta。

from collections.abc import Mapping  # 只用 Mapping 来判断输入是否是指标字典。
from typing import Any  # 用 Any 接住不确定的指标值，统一类型注解。

from liquidloc.common.validation import is_numeric, is_string_like  # 集中判断数值类型（排除 bool / np.bool_）和字符串类型。

# 基线接近 0 时的除零容差，与项目其他几何零阈值（_GEOMETRIC_ZERO_EPSILON 等）同量级，
# 用于判断 full 基线是否退化到 0 以避免 relative_delta 除零。
_BASELINE_ZERO_EPSILON = 1e-12


def _coerce_metric_row(row_name: str, metric_row: Mapping, compare_keys: list[str], fallback_method_name: str | None = None) -> dict:  # 先把单行指标规范化，避免后面比较时字段缺失。
    """把单行指标表规范化成可比较的字典，并补齐 method_name。"""
    if not isinstance(metric_row, Mapping):  # 先确认这一行真的是映射，不是列表或别的容器。
        raise TypeError(f"{row_name} must be a mapping, got {type(metric_row).__name__}")  # 类型不对就直接报错，避免后面误读字段。

    normalized_row = dict(metric_row)  # 拷贝一份，避免原始输入被原地改掉。
    if fallback_method_name is not None and "method_name" not in normalized_row:  # 如果缺 method_name，就用外部传入的兜底值。
        normalized_row["method_name"] = fallback_method_name  # 补齐方法名，方便后面统一比较。

    missing_keys = [metric_name for metric_name in compare_keys if metric_name not in normalized_row]  # 找出需要比较但当前行没有的指标。
    if missing_keys:  # 只要缺了任意一个，就不能安全比较。
        missing = ", ".join(missing_keys)  # 把缺失项拼成可读字符串。
        raise ValueError(f"{row_name} is missing compare_keys: {missing}")  # 缺键属于值契约违规，按项目硬约束用 ValueError 而非 KeyError。
    return normalized_row  # 返回补齐后的指标行。


def align_ablation_tables(full_metrics: Mapping[str, Any], ablation_metrics: dict[str, Mapping[str, Any]], compare_keys: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:  # 对齐 full 基线和所有消融结果。
    """对齐 full 基线和所有 ablation 变体，保证后续比较字段一致。"""
    if not isinstance(compare_keys, list):  # compare_keys 必须是列表，保证顺序稳定。
        raise TypeError(f"compare_keys must be a list, got {type(compare_keys).__name__}")  # 类型不对就报错。
    if any(not is_string_like(metric_name) or not metric_name for metric_name in compare_keys):  # 列表里每个指标名都必须是非空字符串。
        raise TypeError("compare_keys must contain non-empty strings")  # 发现非法指标名就拒绝。
    if not isinstance(ablation_metrics, dict):  # 消融结果必须是字典，才能按方法名逐个对比。
        raise TypeError(f"ablation_metrics must be a dict, got {type(ablation_metrics).__name__}")  # 类型不对就报错。

    baseline_row = _coerce_metric_row("full_metrics", full_metrics, compare_keys)  # 先把 full 基线整理成统一格式。
    aligned_ablations = []  # 这里收集每个消融方法的规范化结果。
    for ablation_name, ablation_row in ablation_metrics.items():  # 逐个方法处理，避免漏掉任何消融变体。
        aligned_ablations.append(  # 把规范化后的结果追加到列表。
            _coerce_metric_row(  # 继续复用单行规范化逻辑。
                f"ablation_metrics[{ablation_name!r}]",  # 这里的名字用于报错定位。
                ablation_row,  # 当前方法对应的那一行指标。
                compare_keys,  # 需要比较的指标集合。
                fallback_method_name=ablation_name,  # 如果没写 method_name，就用字典 key 兜底。
            )  # 规范化函数调用结束。
        )  # append 结束。
    return baseline_row, aligned_ablations  # 返回基线和所有对齐后的消融行。


def compare_one_ablation(baseline_row: Mapping[str, Any], ablation_row: Mapping[str, Any], compare_keys: list[str]) -> list[dict[str, Any]]:  # 逐指标比较某一个消融方法和 full 基线。
    """把单个 ablation 结果和 full 基线逐指标做差值比较。"""
    if "method_name" not in ablation_row:  # method_name 是必需键，缺键必须显式报错，不能用 .get() 静默返回 None。
        raise TypeError("ablation row must contain a non-empty method_name")  # 缺键就报错，避免后面静默失败。
    method_name = ablation_row["method_name"]  # 直接取方法名，后面报错和结果都要用它。
    if not is_string_like(method_name) or not method_name:  # 方法名必须是非空字符串。
        raise TypeError("ablation row must contain a non-empty method_name")  # 方法名不合法就报错。

    comparison_rows = []  # 用这个列表收集每个指标的比较结果。
    for metric_name in compare_keys:  # 按传入顺序逐个指标比较，避免顺序漂移。
        baseline_value = baseline_row[metric_name]  # 取 full 基线值。
        ablation_value = ablation_row[metric_name]  # 取消融方法对应值。

        if not is_numeric(baseline_value):  # 基线值必须是数值，bool 不算数值。
            raise TypeError(f"full_metrics[{metric_name!r}] must be numeric")  # 类型不对就报错。
        if not is_numeric(ablation_value):  # 消融值也必须是数值。
            raise TypeError(f"ablation_metrics[{method_name!r}][{metric_name!r}] must be numeric")  # 类型不对就报错。

        try:  # float() 转换需要守卫：超大整数会 OverflowError，与 coerce_finite_scalar 守卫口径对齐。
            baseline_float = float(baseline_value)  # 统一转 float，方便后面做有限性检查和除法。
            ablation_float = float(ablation_value)  # 同样转 float。
        except (OverflowError, TypeError, ValueError) as exc:  # 与项目其他 float() 守卫口径一致。
            raise ValueError(f"failed to convert metrics[{metric_name!r}] to float for method {method_name!r}") from exc  # 转换失败就报错。
        if not (math.isfinite(baseline_float) and math.isfinite(ablation_float)):  # inf/nan 会污染 delta 和 relative_delta，必须拒绝。
            raise ValueError(f"metrics[{metric_name!r}] must be finite for method {method_name!r}, got baseline={baseline_value!r}, ablation={ablation_value!r}")  # 非有限值直接报错。

        delta = ablation_float - baseline_float  # 先算绝对差值，正负表示相对基线的变化方向。
        if abs(baseline_float) < _BASELINE_ZERO_EPSILON:  # 基线接近 0 时，相对差值容易除零，用容差比较避免浮点精度问题。
            if delta != 0.0:  # 如果基线是 0 但差值又不为 0，相对变化率就没有定义。
                raise ValueError(f"relative_delta is undefined when full_metrics[{metric_name!r}] is zero and delta is non-zero")  # 直接拒绝这种输入。
            relative_delta = 0.0  # 只有完全相等时，才把相对变化率记为 0。
        else:  # 基线不为 0，就可以正常算相对变化。
            relative_delta = delta / baseline_float  # 相对变化率按基线归一化。

        comparison_rows.append(  # 把当前指标的比较结果追加进去。
            {  # 每一行都是一个标准化比较记录。
                "method_name": method_name,  # 当前消融方法名。
                "metric_name": metric_name,  # 当前比较的指标名。
                "delta": delta,  # 与 full 的绝对差。
                "relative_delta": relative_delta,  # 与 full 的相对差。
            }  # 这一行记录结束。
        )  # append 结束。
    return comparison_rows  # 返回这个方法的逐指标比较结果。


def build_ablation_table(full_metrics: Mapping, ablation_metrics: dict, compare_keys: list[str]) -> list[dict]:  # 把所有消融方法的比较结果拼成一张总表。
    """把 full 和多个 ablation 的对比结果打平成统一表。"""
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "compare_keys": list(compare_keys) if compare_keys is not None else None,
            "ablation_methods": list(ablation_metrics.keys()) if isinstance(ablation_metrics, Mapping) else None,
        },
        "build_ablation_table 入口参数",
        prefix="[analysis]",
    )
    baseline_row, aligned_ablations = align_ablation_tables(full_metrics, ablation_metrics, compare_keys)  # 先把输入对齐，保证后续比较统一。

    comparison_table = []  # 这里收集所有方法、所有指标的比较记录。
    for ablation_row in aligned_ablations:  # 逐个方法展开。
        comparison_table.extend(compare_one_ablation(baseline_row, ablation_row, compare_keys))  # 把这个方法的所有指标结果并入总表。
    return comparison_table  # 返回统一的 ablation comparison 表。
