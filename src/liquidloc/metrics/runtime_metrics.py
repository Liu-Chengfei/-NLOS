"""文件:src/liquidloc/metrics/runtime_metrics.py  # 文件头：说明这个文件是谁、做什么。

【文件职责】从 runtime_log 里提取 latency 和资源占用指标。  # 说明文件功能。
它存在的原因是把"运行开销如何统一统计"固定下来，让分析和绘图都能共享同一口径。  # 说明存在理由。

【上游依赖】  # 说明本文件依赖哪些模块。
- 无外部项目模块依赖，仅使用标准库 collections.abc、math、numbers 和 statistics。  # 只依赖标准库。

【下游调用者】  # 说明谁会用。
- metric_runner.py：通过 compute_runtime_metrics 调用，作为四类指标之一。  # 总调度器。
- 运行时分析脚本和展示性能分布的工具：直接调用本模块计算运行时指标。  # 分析面。
- tests/metrics/test_runtime_metrics.py：单元测试。  # 测试面。

【输入对象定义】  # 说明输入对象。
- runtime_log: 映射，必须包含 latency（延迟序列）、params（参数量）和 ram_peak（峰值内存）三个键。  # 运行时日志。

【输出对象定义】  # 说明输出对象。
- dict[str, float]：包含 latency_mean、latency_p50、latency_p95、params、ram_peak 五个键的指标字典。  # 运行时指标。

【核心变量定义】  # 说明核心中间变量。
- latency_values: 规范化并排序后的延迟样本列表。  # 延迟样本。
- latency_mean: 延迟均值。  # 均值。
- latency_p50: 延迟中位数。  # P50。
- latency_p95: 延迟 95 分位数。  # P95。
- params: 模型参数量。  # 参数量。
- ram_peak: 峰值内存占用。  # 峰值内存。

【推荐编写顺序】1. 先看 compute_runtime_metrics 入口。2. 再看内部 percentile 辅助函数。  # 推荐阅读顺序。
"""

from __future__ import annotations  # 允许类型注解中使用现代写法（如 dict[str, float]），与其它 metrics 模块保持一致。

from collections.abc import Iterable, Mapping  # 用于判断输入是否可迭代以及是否为映射。
from typing import Any  # 用于类型注解。

from liquidloc.common.validation import is_real  # 统一判断标量数值类型。
from math import isfinite  # 用于判断数值是否是有限值。
from statistics import fmean  # 用于计算浮点平均值。


def compute_runtime_metrics(runtime_log: Mapping[str, Any]) -> dict[str, float]:
    """从运行时日志计算延迟和资源占用指标。  # 函数总说明。

    作用：把运行时日志里的 latency、params、ram_peak 规范化后，计算均值和分位数等指标。  # 作用说明。
    参数：  # 参数说明。
    - runtime_log: 必须是映射，并且至少包含 latency、params、ram_peak 三个键。  # 参数说明。
    返回值：  # 返回值说明。
    - dict[str, float]：运行时指标字典。  # 返回值说明。
    异常/失败条件：  # 失败条件说明。
    - runtime_log 不是映射时抛 TypeError。  # 类型错误。
    - 缺少必要键时抛 KeyError。  # 键错误。
    - latency 不是合法数值序列、为空或含非有限值时抛异常。  # 数据错误。
    状态变化：  # 状态变化说明。
    - 不修改原始 runtime_log，只返回新的指标字典。  # 状态说明。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "runtime_log_keys": list(runtime_log.keys()) if isinstance(runtime_log, Mapping) else None,
        },
        "compute_runtime_metrics 入口参数",
        prefix="[metrics]",
    )
    if not isinstance(runtime_log, Mapping):  # runtime_log 必须是映射，才能按键取值。
        raise TypeError("runtime_log must be a mapping.")  # 类型不对直接拒绝。

    required_keys = ("latency", "params", "ram_peak")  # 定义运行时日志必须包含的键。
    missing_keys = [key for key in required_keys if key not in runtime_log]  # 找出缺失的必需键。
    if missing_keys:  # 只要有缺失键就不能继续。
        raise KeyError(f"runtime_log missing required keys: {missing_keys}")  # 明确告诉调用方缺了什么。

    latency = runtime_log["latency"]  # 取出延迟序列。
    if isinstance(latency, (str, bytes)) or not isinstance(latency, Iterable):  # 延迟必须是可迭代数值序列。
        raise TypeError("runtime_log['latency'] must be a numeric iterable.")  # 类型不对直接报错。

    latency_values = []  # 保存规范化后的延迟样本。
    for value in latency:  # 逐个检查 latency 里的值。
        if not is_real(value):  # 布尔值和非实数都不允许。
            raise TypeError("runtime_log['latency'] must contain only numeric values.")  # 元素类型不合格。
        try:  # 极大的 Real（如大整数或 Fraction）转 float 可能溢出。
            numeric_value = float(value)  # 统一转成浮点数，方便统计。
        except (OverflowError, TypeError, ValueError) as exc:  # 与 coerce_finite_scalar 守卫口径对齐。
            raise ValueError(f"runtime_log['latency'] contains unconvertible value: {value!r}") from exc
        if not isfinite(numeric_value):  # NaN 和无穷大不能参与统计。
            raise ValueError("runtime_log['latency'] must contain only finite values.")  # 非有限值直接报错。
        latency_values.append(numeric_value)  # 收集合法延迟值。
    if not latency_values:  # 空延迟序列没有统计意义。
        raise ValueError("runtime_log['latency'] must not be empty.")  # 直接拒绝空序列。

    params = runtime_log["params"]  # 取出参数数量。
    if not is_real(params):  # 参数数量必须是实数。
        raise TypeError("runtime_log['params'] must be numeric.")  # 类型不对直接报错。
    try:  # 极大的 Real 转 float 可能溢出。
        params = float(params)  # 统一转成浮点数。
    except (OverflowError, TypeError, ValueError) as exc:  # 与 coerce_finite_scalar 守卫口径对齐。
        raise ValueError(f"runtime_log['params'] must be a finite numeric value, got {params!r}") from exc
    if not isfinite(params):  # 参数数量也必须是有限值。
        raise ValueError("runtime_log['params'] must be finite.")  # 非有限值直接报错。

    ram_peak = runtime_log["ram_peak"]  # 取出峰值内存。
    if not is_real(ram_peak):  # 峰值内存也必须是实数。
        raise TypeError("runtime_log['ram_peak'] must be numeric.")  # 类型错误。
    try:  # 极大的 Real 转 float 可能溢出。
        ram_peak = float(ram_peak)  # 转成浮点数统一口径。
    except (OverflowError, TypeError, ValueError) as exc:  # 与 coerce_finite_scalar 守卫口径对齐。
        raise ValueError(f"runtime_log['ram_peak'] must be a finite numeric value, got {ram_peak!r}") from exc
    if not isfinite(ram_peak):  # 峰值内存也必须是有限值。
        raise ValueError("runtime_log['ram_peak'] must be finite.")  # 非有限值报错。

    latency_values.sort()  # 分位数计算需要先排序。
    latency_mean = fmean(latency_values)  # 计算延迟均值。

    def percentile(percent):
        """在当前延迟样本上做线性插值分位数估计。  # 内部函数说明。

        作用：根据百分位数位置，在已经排序的 latency_values 上计算分位数。  # 作用说明。
        参数：  # 参数说明。
        - percent: 0 到 1 之间的分位比例。  # 参数说明。
        返回值：  # 返回值说明。
        - float：分位数估计值。  # 返回值说明。
        异常/失败条件：  # 失败条件说明。
        - 这里不单独抛错，依赖外层已校验的 latency_values。  # 说明失败策略。
        状态变化：  # 状态变化说明。
        - 不修改外部状态，只读 latency_values。  # 状态说明。
        """
        if len(latency_values) == 1:  # 只有一个样本时，分位数就是它本身。
            return latency_values[0]  # 直接返回唯一样本。

        position = (len(latency_values) - 1) * percent  # 计算分位数所在的插值位置。
        lower_index = int(position)  # 下界索引。
        upper_index = min(lower_index + 1, len(latency_values) - 1)  # 上界索引，不能越界。
        lower_value = latency_values[lower_index]  # 下界值。
        upper_value = latency_values[upper_index]  # 上界值。
        weight = position - lower_index  # 插值权重。
        return lower_value + (upper_value - lower_value) * weight  # 线性插值得到分位数。

    latency_p50 = percentile(0.50)  # 计算中位数。
    latency_p95 = percentile(0.95)  # 计算 95 分位数。

    runtime_metric_dict = {  # 组装最终 runtime 指标字典。
        "latency_mean": latency_mean,  # 平均延迟。
        "latency_p50": latency_p50,  # 中位延迟。
        "latency_p95": latency_p95,  # 95 分位延迟。
        "params": params,  # 参数数量。
        "ram_peak": ram_peak,  # 峰值内存。
    }
    return runtime_metric_dict  # 返回运行时指标结果。
