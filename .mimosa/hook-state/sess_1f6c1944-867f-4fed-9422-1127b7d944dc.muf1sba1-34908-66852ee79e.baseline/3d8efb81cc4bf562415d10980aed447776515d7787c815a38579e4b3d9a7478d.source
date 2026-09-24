"""文件:src/liquidloc/metrics/tail_metrics.py  # 文件头：说明这个文件的身份。

【文件职责】从误差序列里提取尾部指标，例如 p95、p99 和 failure_rate。  # 说明文件功能。
它存在的原因是把"长尾是否严重"从轨迹指标里分离出来，便于单独看失败比例和连续失败段。  # 说明存在理由。

【上游依赖】  # 说明本文件依赖哪些模块。
- 无外部项目模块依赖，仅使用标准库 math、collections.abc 和 numbers。  # 只依赖标准库。

【下游调用者】  # 说明谁会用。
- metric_runner.py：通过 compute_tail_metrics 调用，作为四类指标之一。  # 总调度器。
- 分析脚本和关注鲁棒性的诊断流程：直接调用本模块计算尾部指标。  # 分析面。
- tests/metrics/test_tail_metrics.py：单元测试。  # 测试面。

【输入对象定义】  # 说明输入对象。
- errors: 误差值序列，必须是非字符串可迭代对象，元素为非负有限实数。  # 误差序列。
- failure_threshold: 失败阈值，默认跟随实验协议 evaluation.default_failure_threshold_m，必须为正有限实数。  # 失败阈值。

【输出对象定义】  # 说明输出对象。
- tuple[dict[str, float], dict[str, list[tuple[int, int]]]]：(尾部指标字典, 支撑报告)；尾部指标字典含 p95/p99/failure_rate，支撑报告含 long_failure_segments。  # 尾部指标。

【核心变量定义】  # 说明核心中间变量。
- normalized_errors: 规范化后的误差值列表。  # 规范化误差。
- sorted_errors: 排序后的误差列表，用于分位数计算。  # 排序误差。
- p95: 95 分位误差。  # P95。
- p99: 99 分位误差。  # P99。
- failure_rate: 超过失败阈值的比例。  # 失败率。
- long_failure_segments: 连续失败段列表，每段用 (start, end) 闭区间表示。  # 长失败段。

【推荐编写顺序】1. 先看 compute_tail_metrics 入口。2. 再看内部校验和分位数/失败段计算逻辑。  # 推荐阅读顺序。
"""

from __future__ import annotations  # 允许延迟类型注解，方便未来扩展。

import math  # 提供有限性、分位数索引和数学运算支持。
from collections.abc import Iterable, Sequence  # 用于判断误差输入可迭代与已排序序列类型。
from liquidloc.common.validation import is_real  # 统一判断标量数值类型。
from liquidloc.protocol.experiment_gates import get_default_failure_threshold_m  # 从协议读取默认失败阈值。

def compute_tail_metrics(errors: Iterable[float], failure_threshold: float | None = None) -> tuple[dict[str, float], dict[str, list[tuple[int, int]]]]:
    """根据误差序列计算尾部指标（p95、p99、failure_rate 和连续失败段）。  # 函数总说明。

    作用：根据误差序列计算 p95、p99、failure_rate 和连续失败段。  # 作用说明。
    参数：  # 参数说明。
    - errors: 误差序列，必须是非字符串可迭代对象。  # 参数说明。
    - failure_threshold: 失败阈值；若不显式传入，则默认跟随实验协议 evaluation.default_failure_threshold_m。  # 参数说明。
    返回值：  # 返回值说明。
    - tuple[dict[str, float], dict[str, list[tuple[int, int]]]]：(尾部指标字典, 支撑报告)；尾部指标字典含 p95/p99/failure_rate，支撑报告含 long_failure_segments。  # 返回值说明。
    异常/失败条件：  # 失败条件说明。
    - errors 不是合法可迭代对象时抛 TypeError。  # 类型错误。
    - errors 为空、元素非法或阈值非法时抛异常。  # 数据错误。
    状态变化：  # 状态变化说明。
    - 不修改输入，只返回新字典。  # 状态说明。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "failure_threshold": failure_threshold,
            "errors_type": type(errors).__name__,
        },
        "compute_tail_metrics 入口参数",
        prefix="[metrics]",
    )
    if failure_threshold is None:  # 默认口径统一取自实验协议，避免本地常量漂移。
        failure_threshold = get_default_failure_threshold_m()
    if isinstance(errors, (str, bytes)) or not isinstance(errors, Iterable):  # 字符串不是合法误差序列。
        raise TypeError(f"errors must be an iterable of numbers, got {type(errors).__name__}")  # 指出输入类型。
    if not is_real(failure_threshold):  # 阈值必须是实数，不能是布尔值。
        raise TypeError("failure_threshold must be a real number")  # 阈值类型不对。
    if not math.isfinite(failure_threshold):  # 阈值也必须是有限值。
        raise ValueError("failure_threshold must be finite")  # 非有限阈值拒绝。
    if failure_threshold <= 0.0:  # 阈值必须大于 0。
        raise ValueError('failure_threshold must be positive')  # 阈值非正则报错。

    raw_errors = list(errors)  # 先转成列表，方便多次遍历和排序。
    if not raw_errors:  # 空误差序列没有统计意义。
        raise ValueError("errors must be non-empty")  # 直接拒绝。

    normalized_errors = []  # 保存规范化后的误差值。
    for index, error in enumerate(raw_errors):  # 逐个检查误差值。
        if not is_real(error):  # 布尔值和非实数都不允许。
            raise TypeError(f"errors[{index}] must be numeric, got {type(error).__name__}")  # 明确指出坏元素位置。
        try:  # 极大 int（如 10**1000）通过 is_real 但 float() 会溢出，需单独守卫。
            value = float(error)  # 统一转成浮点数。
        except OverflowError:  # int 过大无法表示为 float。
            raise ValueError(f"errors[{index}] overflow when converting to float") from None  # 转换溢出报错。
        if not math.isfinite(value):  # NaN 和无穷大不能参与统计。
            raise ValueError(f"errors[{index}] must be finite")  # 非有限值报错。
        if value < 0.0:  # 误差值不能为负。
            raise ValueError(f"errors[{index}] must be non-negative, got {value}")  # 负误差直接拒绝。
        normalized_errors.append(value)  # 收集合法误差值。

    sorted_errors = sorted(normalized_errors)  # 排序后便于计算分位数。
    p95 = _percentile(sorted_errors, 0.95)  # 95 分位数（线性插值，与 runtime_metrics 一致）。
    p99 = _percentile(sorted_errors, 0.99)  # 99 分位数（线性插值，与 runtime_metrics 一致）。
    failure_rate = sum(error > failure_threshold for error in normalized_errors) / len(normalized_errors)  # 超阈值比例。

    long_failure_segments = []  # 保存连续失败段。
    segment_start = None  # 当前失败段的起点，None 表示还没进入失败段。
    for index, error in enumerate(normalized_errors):  # 逐个误差检查是否进入/退出失败段。
        if error > failure_threshold:  # 超阈值表示处于失败状态。
            if segment_start is None:  # 第一次进入失败段时记录起点。
                segment_start = index  # 记录失败段起始下标。
            continue  # 继续扫描后续点，等待失败段结束。
        if segment_start is not None and index - segment_start >= 2:  # 失败段长度至少 2 才算长失败段。
            long_failure_segments.append((segment_start, index - 1))  # 记录失败段闭区间。
        segment_start = None  # 当前点正常，失败段结束。

    if segment_start is not None and len(normalized_errors) - segment_start >= 2:  # 处理序列结尾仍在失败段中的情况。
        long_failure_segments.append((segment_start, len(normalized_errors) - 1))  # 把尾部失败段补进去。

    tail_metric_dict = {  # 组装最终尾部指标字典。
        "p95": p95,  # 95 分位误差。
        "p99": p99,  # 99 分位误差。
        "failure_rate": failure_rate,  # 失败比例。
    }
    # long_failure_segments 不在 metric_schema 中注册，通过 support_report 传递。
    support_report = {
        "long_failure_segments": long_failure_segments,  # 连续失败段列表，供诊断使用。
    }
    return tail_metric_dict, support_report  # 返回尾部指标和支撑报告。


def _percentile(sorted_values: Sequence[float], percent: float) -> float:
    """线性插值分位数计算，与 runtime_metrics.percentile 口径一致。

    采用线性插值法：当分位数位置落在两个数据点之间时，
    按位置权重在相邻两点间做线性插值，而非取最近邻。
    这保证了分位数估计的连续性，避免阶梯式跳变。

    参数:
        sorted_values: 已排序的数值列表（升序）。
        percent: 0 到 1 之间的分位比例，如 0.95 表示第 95 百分位。

    返回值:
        float: 分位数估计值。空列表返回 0.0，单元素列表返回该元素。
    """
    if not 0.0 <= percent <= 1.0:  # 分位比例必须在 [0, 1] 内，越界会导致位置越界或负索引外推。
        raise ValueError(f"percent must be in [0, 1], got {percent}")  # 越界直接拒绝。
    n = len(sorted_values)
    if n == 0:  # 空列表没有可计算的值。
        return 0.0
    if n == 1:  # 只有一个值时，任何分位数都等于它。
        return sorted_values[0]
    position = (n - 1) * percent  # 分位数在排序数组中的浮点位置。
    lower_index = int(position)  # 下界索引（取整）。
    upper_index = min(lower_index + 1, n - 1)  # 上界索引，不能越界。
    lower_value = sorted_values[lower_index]  # 下界值。
    upper_value = sorted_values[upper_index]  # 上界值。
    weight = position - lower_index  # 插值权重，0 表示取下界，1 表示取上界。
    return lower_value + (upper_value - lower_value) * weight  # 线性插值。
