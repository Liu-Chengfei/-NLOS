"""
对模型间指标差异执行统计检验和多重比较校正，输出统一 statistics_table。

这个模块把 metric_table 按 group_keys 分组后，针对指定指标做成对统计检验，
再把原始 p_value、效应量和校正后的 adjusted_p_value 组装成标准统计结果表。
它不改原始指标，也不负责绘图，只负责让 summary 和 analysis 读取同一份统计口径。
"""

from __future__ import annotations  # 允许函数签名里使用后面再出现的类型名。

import math  # 用于均值、平方根和有限性判断等基础数值运算。
import random  # 用于 bootstrap 和 Bayesian bootstrap 的重采样随机数。
from collections.abc import Mapping, Sequence  # 用于判断输入是否为映射或序列。
from itertools import combinations  # 用于枚举分组之间的两两组合。
from statistics import NormalDist  # 用于把 z 分数转成正态分布的尾部概率。
from typing import Any  # 用于类型注解。

from liquidloc.common.validation import coerce_finite_scalar, is_string_like, require_iterable, require_keys, require_not_none  # 复用通用输入校验工具。
from liquidloc.protocol.metric_schema import get_metric_meta, get_metric_order  # 复用指标协议中的元信息和顺序。

# 手册 P22 硬约束：5 方法 10 对两两比较用 Holm-Bonferroni 多重比较校正。
# BH-FDR 仍保留为 _bh_fdr_adjust_p_values 用于非 handbook P22 场景。
from liquidloc.analysis.handbook_p22_statistics import holm_bonferroni_adjust  # 手册 P22 校正函数。


def _normalize_name_list(values: Any, *, name: str) -> list[str]:  # 把名字列表规范化成去重后的非空字符串列表。
    """把输入的名称列表规范化成去重后的非空字符串列表。"""
    require_not_none(values, name)  # 先保证值本身不是 None。
    if not isinstance(values, Sequence) or is_string_like(values) or isinstance(values, (bytes, bytearray)):  # 只接受真正的序列，拒绝 str/numpy.str_/bytes/bytearray 等会被逐字符迭代的类型。
        raise TypeError(f"{name} must be a non-string sequence of names")  # 类型不对就拒绝。

    normalized_values = []  # 用这个列表保存去重后的顺序结果。
    seen_values = set()  # 用集合记录已经出现过的名称。
    for index, raw_value in enumerate(values):  # 逐个检查输入名称。
        if not is_string_like(raw_value):  # 每个元素都必须是字符串。
            raise TypeError(f"{name}[{index}] must be a string, got {type(raw_value).__name__}")  # 报错时指出具体位置。
        normalized_value = str(raw_value).strip()  # 去掉首尾空白，避免同一个名字写成不同空格版本。
        if not normalized_value:  # 清洗后为空说明这个名字无效。
            raise ValueError(f"{name}[{index}] must be a non-empty string")  # 空名字不允许。
        if normalized_value in seen_values:  # 重复名字只保留第一次出现。
            continue  # 继续下一个元素。
        seen_values.add(normalized_value)  # 记录这个名字已出现。
        normalized_values.append(normalized_value)  # 按原顺序保留非重复名字。

    if not normalized_values:  # 如果全部都被过滤掉了，就说明输入没给出有效名称。
        raise ValueError(f"{name} must be non-empty")  # 空列表不允许。
    return normalized_values  # 返回干净的名称列表。


def _normalize_metric_rows(metric_table: Any) -> list[dict[str, Any]]:  # 把 metric_table 统一成字典行列表。
    """把 metric_table 规范成字典行列表，统一后续分组入口。"""
    require_not_none(metric_table, "metric_table")  # 先保证表不是 None。

    if hasattr(metric_table, "to_dict") and hasattr(metric_table, "columns"):  # 支持类似 pandas.DataFrame 的对象。
        metric_rows = metric_table.to_dict(orient="records")  # DataFrame 风格转成记录列表。
    elif isinstance(metric_table, Mapping):  # 如果已经是映射，就检查它是不是单行或 row dict 集合。
        if metric_table and all(isinstance(row, Mapping) for row in metric_table.values()):  # 如果 value 也是映射，说明可能是 case_ref -> row。
            metric_rows = list(metric_table.values())  # 取出所有行。
        else:  # 否则把它当成单行表。
            metric_rows = [metric_table]  # 把单个映射包成一行列表。
    else:  # 再不行就把它当成普通可迭代对象。
        require_iterable(metric_table, name="metric_table")  # 先确认它真的可迭代。
        metric_rows = list(metric_table)  # 转成列表，便于重复遍历和检查。

    if not metric_rows:  # 空表没有统计意义。
        raise ValueError("metric_table must be non-empty")  # 空输入直接拒绝。

    normalized_rows = []  # 用这个列表保存规范化后的每一行。
    for index, metric_row in enumerate(metric_rows):  # 逐行检查。
        if not isinstance(metric_row, Mapping):  # 每行都必须是映射。
            raise TypeError(f"metric_table[{index}] must be a mapping, got {type(metric_row).__name__}")  # 报错指出具体行。
        normalized_rows.append(dict(metric_row))  # 复制一份，避免后面误改原对象。
    return normalized_rows  # 返回规范化后的行列表。


def _coerce_numeric_value(value: Any, *, name: str) -> float:  # 把任意值转成有限浮点数。
    """把任意数值样式的输入转成有限 float。

    布尔值被视为值合同违例：底层 coerce_finite_scalar 对布尔值抛出的
    TypeError（消息含 "must be numeric"）会被转成 ValueError 重新抛出，
    与项目硬约束"value contract 问题应抛 ValueError 而非 TypeError"对齐；
    其它 TypeError 保持原样向上传播。非有限值（inf/nan）继续走 ValueError 路径。
    """
    try:
        return coerce_finite_scalar(value, name=name)  # 统一走 coerce_finite_scalar：非数值/bool 报 TypeError，非有限报 ValueError。
    except TypeError as exc:  # 捕获布尔值/非数值类型触发的 TypeError。
        if "must be numeric" in str(exc):  # 布尔值或非数值类型属于值合同违例，转成 ValueError。
            raise ValueError(str(exc)) from exc  # 保持消息不变，仅切换异常类型。
        raise  # 其它 TypeError（如 None 输入触发的 require_not_none 报错）原样向上传播。


def _validate_metric_names(metric_names: list[str]) -> list[str]:  # 按协议顺序过滤请求的指标。
    """按协议顺序过滤并验证请求的指标名。"""
    metric_order = get_metric_order()  # 先拿到协议定义的标准顺序。
    unknown_metric_names = [metric_name for metric_name in metric_names if metric_name not in metric_order]  # 找出协议里没有的名字。
    if unknown_metric_names:  # 如果有未知名字，就说明调用方传错了。
        raise KeyError(f"metric_names contains unknown schema metrics: {unknown_metric_names}")  # 直接报出未知项。
    return [metric_name for metric_name in metric_order if metric_name in metric_names]  # 按协议顺序返回请求过的指标。


def group_metric_values(metric_table: Any, group_keys: list[str], metric_names: list[str]) -> dict[str, dict[tuple, list[float]]]:  # 把指标按分组键聚合。
    """按分组键和指标名聚合指标值，返回分组后的数值字典。

    作用：把 metric_table 中的指标值按 group_keys 分组聚合，
    支持长表（含 metric/value 列）和宽表（每列一个指标）两种格式。
    返回的字典结构为 {metric_name: {group_tuple: [values]}}，
    供后续统计检验直接消费。

    参数:
        metric_table: 指标表，支持 DataFrame、单个映射或映射序列。
        group_keys: 分组键列表，如 ["method_name"]。
        metric_names: 需要聚合的指标名列表。

    返回值:
        dict[str, dict[tuple, list[float]]]: 按指标名和分组键聚合后的数值字典。

    异常:
        TypeError: 输入类型不正确。
        ValueError: 长表和宽表混用、字段缺失或数值非法。
    """
    metric_rows = _normalize_metric_rows(metric_table)  # 先把输入统一成行列表。
    grouped_values = {metric_name: {} for metric_name in metric_names}  # 为每个指标准备一个分组字典。
    metric_name_set = set(metric_names)  # 用集合加速 membership 判断。

    long_form_rows = [("metric" in metric_row) or ("value" in metric_row) for metric_row in metric_rows]  # 判断每行是不是长表格式。
    if any(long_form_rows) and not all(long_form_rows):  # 不能一部分长表、一部分宽表混着来。
        raise ValueError("metric_table must use either long-form rows or wide-form rows consistently")  # 混用会让逻辑分叉失真。

    if all(long_form_rows):  # 如果全都是长表，就按 metric/value 读取。
        for row_index, metric_row in enumerate(metric_rows):  # 逐行处理长表。
            require_keys(metric_row, [*group_keys, "metric", "value"], name=f"metric_table[{row_index}]")  # 每行至少要有分组键、metric 和 value。
            metric_name = metric_row["metric"]  # 读取当前行属于哪个指标。
            if metric_name not in metric_name_set:  # 如果不是本次要检验的指标，就跳过。
                continue  # 不参与分组。
            group_value = tuple(metric_row[group_key] for group_key in group_keys)  # 把多个分组键拼成一个组元组。
            numeric_value = _coerce_numeric_value(  # 把 value 转成数值，供统计检验使用。
                metric_row["value"],  # 当前行的数值字段。
                name=f"metric_table[{row_index}].value",  # 报错时指出具体位置。
            )  # 数值转换结束。
            grouped_values[metric_name].setdefault(group_value, []).append(numeric_value)  # 把这个数值塞进对应组。
        return grouped_values  # 长表模式处理完直接返回。

    for row_index, metric_row in enumerate(metric_rows):  # 宽表模式下逐行处理。
        require_keys(metric_row, [*group_keys, *metric_names], name=f"metric_table[{row_index}]")  # 宽表每行要同时具备所有需要的指标。
        group_value = tuple(metric_row[group_key] for group_key in group_keys)  # 先算出这个样本属于哪个组。
        for metric_name in metric_names:  # 再逐个指标取值。
            numeric_value = _coerce_numeric_value(  # 把当前指标值转成可统计的浮点数。
                metric_row[metric_name],  # 当前指标值。
                name=f"metric_table[{row_index}].{metric_name}",  # 报错时指出具体行和列。
            )  # 数值转换结束。
            grouped_values[metric_name].setdefault(group_value, []).append(numeric_value)  # 按组把值收集起来。
    return grouped_values  # 宽表模式处理完返回分组结果。


def _average_ranks(values: Sequence[float]) -> tuple[list[float], list[int]]:  # 计算平均秩和 tie 规模。
    """为秩和检验计算平均秩，并保留每个 tie block 的大小。"""
    ranked_values = sorted(enumerate(values), key=lambda item: item[1])  # 按数值从小到大排序，并保留原位置。
    ranks = [0.0] * len(values)  # 先准备一个和输入等长的秩数组。
    tie_sizes = []  # 用这个列表记录每个平局块的大小。

    rank_start = 1  # 秩从 1 开始而不是从 0 开始。
    index = 0  # 当前遍历位置。
    while index < len(ranked_values):  # 直到遍历完所有值。
        tie_end = index + 1  # 先假设 tie 只包含一个元素。
        while tie_end < len(ranked_values) and ranked_values[tie_end][1] == ranked_values[index][1]:  # 向后扩展相等值区间。
            tie_end += 1  # 只要值相等就继续扩展 tie。

        tie_size = tie_end - index  # 当前平局块的长度。
        average_rank = (rank_start + (rank_start + tie_size - 1)) / 2.0  # 这个平局块用平均秩。
        for tied_index in range(index, tie_end):  # 平局块里的每个元素都拿同一个平均秩。
            original_index, _ = ranked_values[tied_index]  # 找回原始位置。
            ranks[original_index] = average_rank  # 把平均秩写回原位置。

        tie_sizes.append(tie_size)  # 记录这个平局块的大小。
        rank_start += tie_size  # 下一个秩块从后面继续。
        index = tie_end  # 跳到下一段。

    return ranks, tie_sizes  # 返回平均秩和 tie 信息。


def _compute_mann_whitney_p_value(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:  # 计算两组样本的 Mann-Whitney U 检验 p 值。
    """计算两组样本的 Mann-Whitney U 双侧 p 值。"""
    if not sample_a or not sample_b:  # 任何一组为空都不能做检验。
        raise ValueError("pairwise tests require non-empty samples")  # 空样本直接报错。

    n_a = len(sample_a)  # 样本 A 的大小。
    n_b = len(sample_b)  # 样本 B 的大小。
    combined_values = [*sample_a, *sample_b]  # 把两组样本合并后一起排序。
    combined_ranks, tie_sizes = _average_ranks(combined_values)  # 计算秩和 tie 信息。

    rank_sum_a = math.fsum(combined_ranks[:n_a])  # 前 n_a 个秩对应 sample_a。
    u_a = rank_sum_a - (n_a * (n_a + 1) / 2.0)  # 从秩和换算成 U 值。
    u_b = (n_a * n_b) - u_a  # 另一侧的 U 值。
    u_value = min(u_a, u_b)  # 双侧检验取更小的那个 U。
    mean_u = (n_a * n_b) / 2.0  # U 的期望值。

    total_count = n_a + n_b  # 总样本量。
    if total_count < 2:  # 样本总数太少时没法做近似正态检验。
        return 1.0  # 直接返回不显著。

    tie_term = math.fsum((tie_size ** 3) - tie_size for tie_size in tie_sizes)  # tie 校正项。
    variance = (n_a * n_b / 12.0) * ((total_count + 1) - (tie_term / (total_count * (total_count - 1))))  # 计算 U 的方差。
    if variance <= 0.0:  # 方差异常时也不能继续。
        return 1.0  # 返回保守结果。

    # 连续性校正：从偏离均值的方向减 0.5；当 |u_value - mean_u| < 0.5 时
    # 校正会把分子压到负数（U 落在均值附近，无实质偏离），此时钳到 0，
    # 让 z 为 0、p 值为 1.0，与 scipy.stats.mannwhitneyu 的 clip(p, 0, 1) 行为一致。
    z_score = max(abs(u_value - mean_u) - 0.5, 0.0) / math.sqrt(variance)  # 带连续性校正的 z 分数。
    p_value = 2.0 * (1.0 - NormalDist().cdf(z_score))  # z_score 已非负，直接取上侧尾概率做双侧 p 值。
    return min(max(p_value, 0.0), 1.0)  # 最后裁剪到合法区间。


def _compute_cliffs_delta(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:  # 计算 Cliff's delta。
    """计算 Cliff's delta 作为非参数效应量。"""
    total_pairs = len(sample_a) * len(sample_b)  # 所有成对比较的总数。
    if total_pairs == 0:  # 任何一组为空都没法算效应量。
        raise ValueError("effect size requires non-empty samples")  # 空样本直接报错。

    more_extreme_pairs = 0  # 这里累积“谁更大”的净胜分。
    for value_a in sample_a:  # 遍历 A 组每个值。
        for value_b in sample_b:  # 遍历 B 组每个值。
            if value_a > value_b:  # A 大于 B 记正分。
                more_extreme_pairs += 1  # A 胜一回合。
            elif value_a < value_b:  # A 小于 B 记负分。
                more_extreme_pairs -= 1  # B 胜一回合。
    return more_extreme_pairs / total_pairs  # 净胜分除以总配对数得到效应量。


def _bootstrap_mean_difference_ci(  # 用 bootstrap 给均值差算置信区间。
    sample_a: Sequence[float],  # 第一组样本。
    sample_b: Sequence[float],  # 第二组样本。
    *,  # 后面都是关键字参数，避免调用时看错位置。
    paired: bool,  # 说明这是配对还是非配对 bootstrap。
    confidence_level: float = 0.95,  # 默认置信水平。
    n_resamples: int = 2000,  # 默认重采样次数。
) -> dict[str, float | str]:  # 返回一个描述置信区间的字典。
    """用 bootstrap percentile 计算均值差置信区间。

    输入合同：sample_a / sample_b 的元素必须是有限 float。本函数不做 float()
    转换守卫，依赖上游 `_coerce_numeric_value`（在 group_metric_values 中调用）
    已将指标值清洗为有限浮点数；直接调用本私有函数需自行保证此合同。

    可复现性：配对与非配对路径均使用 random.Random(0) 固定随机种子，
    保证同一输入多次调用结果一致。

    默认值说明：n_resamples=2000 与 _bayesian_bootstrap_mean_difference 的
    n_draws=4000 不一致——两种方法收敛速率不同，非协议定义，可在调用方覆盖。
    confidence_level=0.95 与 Bayesian 版本保持一致。
    """
    if not 0.0 < confidence_level < 1.0:  # 置信水平必须在 0 和 1 之间。
        raise ValueError("confidence_level must be between 0 and 1")  # 越界就报错。
    if n_resamples <= 0:  # 重采样次数必须是正数。
        raise ValueError("n_resamples must be positive")  # 非正数不允许。

    if paired:  # 配对情况要逐对重采样。
        if len(sample_a) != len(sample_b):  # 配对样本长度必须一致。
            raise ValueError("paired confidence intervals require equal-length samples")  # 长度不对就报错。
        if not sample_a:  # 配对样本也不能空（长度相等意味着两者同时为空，否则 L241 会 ZeroDivisionError）。
            raise ValueError("confidence intervals require non-empty samples")  # 空样本不行。
        paired_differences = [float(b) - float(a) for a, b in zip(sample_a, sample_b)]  # 先把配对差值算出来。
        observed_statistic = math.fsum(paired_differences) / len(paired_differences)  # 观测到的均值差。
        bootstrap_statistics = []  # 收集每次重采样得到的统计量。
        rng = random.Random(0)  # 固定随机种子，保证结果可复现。
        for _ in range(n_resamples):  # 重复抽样多次。
            resampled_differences = [  # 每次从配对差值里有放回抽样。
                paired_differences[rng.randrange(len(paired_differences))]  # 随机取一个差值。
                for _ in range(len(paired_differences))  # 抽样数量和原样本一样多。
            ]  # 一次重采样的差值集合。
            bootstrap_statistics.append(math.fsum(resampled_differences) / len(resampled_differences))  # 记录这次的均值差。
    else:  # 非配对情况要分别重采样两组。
        if not sample_a or not sample_b:  # 两组都不能空。
            raise ValueError("confidence intervals require non-empty samples")  # 空样本不行。
        observed_statistic = (math.fsum(float(value) for value in sample_b) / len(sample_b)) - (  # 先算观测到的均值差。
            math.fsum(float(value) for value in sample_a) / len(sample_a)  # A 组均值。
        )  # 观测统计量计算结束。
        bootstrap_statistics = []  # 收集重采样统计量。
        rng = random.Random(0)  # 固定随机源，保持可复现。
        for _ in range(n_resamples):  # 重复重采样。
            resampled_a = [float(sample_a[rng.randrange(len(sample_a))]) for _ in range(len(sample_a))]  # 对 A 组做有放回重采样。
            resampled_b = [float(sample_b[rng.randrange(len(sample_b))]) for _ in range(len(sample_b))]  # 对 B 组做有放回重采样。
            bootstrap_statistics.append(  # 把这次的均值差放进结果列表。
                (math.fsum(resampled_b) / len(resampled_b)) - (math.fsum(resampled_a) / len(resampled_a))  # 计算这次重采样的均值差。
            )  # append 结束。

    bootstrap_statistics.sort()  # 先把 bootstrap 结果排序，便于取分位数。
    alpha = (1.0 - confidence_level) / 2.0  # 两端尾部各留多少比例。
    lower_index = max(0, min(len(bootstrap_statistics) - 1, int(math.floor(alpha * (len(bootstrap_statistics) - 1)))))  # 下界分位索引。
    upper_index = max(0, min(len(bootstrap_statistics) - 1, int(math.ceil((1.0 - alpha) * (len(bootstrap_statistics) - 1)))))  # 上界分位索引。
    return {  # 返回一个标准化的区间描述字典。
        "method": "bootstrap_percentile",  # 说明用了哪种区间方法。
        "statistic": "mean_difference",  # 说明区间对应的统计量。
        "confidence_level": confidence_level,  # 记录置信水平。
        "lower": bootstrap_statistics[lower_index],  # 区间下界。
        "upper": bootstrap_statistics[upper_index],  # 区间上界。
        "observed": observed_statistic,  # 记录观测到的差值。
    }  # 字典结束。


def _draw_dirichlet_weights(sample_size: int, *, rng: random.Random) -> list[float]:  # 生成 Bayesian bootstrap 的 Dirichlet 权重。
    """生成用于 Bayesian bootstrap 的 Dirichlet 权重。"""
    if sample_size <= 0:  # 样本数必须是正的。
        raise ValueError("sample_size must be positive")  # 非正数不允许。
    gamma_draws = [rng.expovariate(1.0) for _ in range(sample_size)]  # 用指数分布采样代替 Dirichlet 的 gamma 采样。
    total = math.fsum(gamma_draws)  # 先把所有权重原始值加起来。
    if not math.isfinite(total) or total <= 0.0:  # 数值异常（非有限或非正）就退回均匀权重。
        return [1.0 / sample_size for _ in range(sample_size)]  # 返回均匀分布。
    return [value / total for value in gamma_draws]  # 否则归一化成概率权重。


def _bayesian_bootstrap_mean_difference(  # 用 Bayesian bootstrap 给均值差估计后验区间。
    sample_a: Sequence[float],  # 第一组样本。
    sample_b: Sequence[float],  # 第二组样本。
    *,  # 后面是关键字参数。
    metric_name: str,  # 指标名，用来查协议里这个指标是“越大越好”还是“越小越好”。
    paired: bool,  # 是否使用配对后验。
    n_draws: int = 4000,  # 默认后验抽样次数。
    confidence_level: float = 0.95,  # 默认后验区间置信水平。
) -> dict[str, float | str]:  # 返回后验区间和偏好概率。
    """用 Bayesian bootstrap 估计均值差的后验区间。

    输入合同：sample_a / sample_b 的元素必须是有限 float。本函数不做 float()
    转换守卫，依赖上游 `_coerce_numeric_value`（在 group_metric_values 与
    _build_paired_samples 中调用）已将指标值清洗为有限浮点数；直接调用本私有
    函数需自行保证此合同。函数体内的 float() 调用仅为类型归一，不构成守卫。

    可复现性：配对与非配对路径共用同一个 random.Random(0) 随机源，
    保证同一输入多次调用结果一致。

    默认值说明：n_draws=4000 与 _bootstrap_mean_difference_ci 的
    n_resamples=2000 不一致——Bayesian bootstrap 后验收敛速率不同于
    频率派 bootstrap，非协议定义，可在调用方覆盖。
    confidence_level=0.95 与频率派版本保持一致。
    """
    if n_draws <= 0:  # 抽样次数必须为正。
        raise ValueError("n_draws must be positive")  # 非正数不允许。
    if not 0.0 < confidence_level < 1.0:  # 置信水平必须合法。
        raise ValueError("confidence_level must be between 0 and 1")  # 越界就报错。
    if paired and len(sample_a) != len(sample_b):  # 配对模式下两组长度必须一致。
        raise ValueError("paired Bayesian bootstrap requires equal-length samples")  # 长度不一致不允许。
    if not sample_a or not sample_b:  # 两组样本都不能空。
        raise ValueError("Bayesian bootstrap requires non-empty samples")  # 空样本不允许。

    metric_meta = get_metric_meta()  # 读取协议里的指标元数据。
    if metric_name not in metric_meta:  # 如果指标不在协议里，就不能判断方向。
        raise KeyError(f"unknown metric for Bayesian bootstrap: {metric_name}")  # 未知指标直接报错。
    direction = metric_meta[metric_name]["direction"]  # 取出这个指标的优劣方向。

    observed_mean_difference = (math.fsum(float(value) for value in sample_b) / len(sample_b)) - (  # 先算观测到的均值差。
        math.fsum(float(value) for value in sample_a) / len(sample_a)  # A 组均值。
    )  # 观测统计量结束。
    rng = random.Random(0)  # 固定随机源，保证可复现。
    posterior_differences: list[float] = []  # 收集后验均值差。
    if paired:  # 配对模式下用配对差值做后验抽样。
        paired_differences = [float(b) - float(a) for a, b in zip(sample_a, sample_b)]  # 先算每对样本的差。
        for _ in range(n_draws):  # 重复抽后验样本。
            weights = _draw_dirichlet_weights(len(paired_differences), rng=rng)  # 给每个差值分配权重。
            posterior_differences.append(  # 记录这一轮的加权均值差。
                math.fsum(weight * difference for weight, difference in zip(weights, paired_differences))  # 权重乘差值再求和。
            )  # 这一轮抽样结束。
    else:  # 非配对模式下分别对两组做 Bayesian bootstrap。
        values_a = [float(value) for value in sample_a]  # 先把 A 组转成浮点数组。
        values_b = [float(value) for value in sample_b]  # 再把 B 组转成浮点数组。
        for _ in range(n_draws):  # 每轮都重新抽权重。
            weights_a = _draw_dirichlet_weights(len(values_a), rng=rng)  # A 组权重。
            weights_b = _draw_dirichlet_weights(len(values_b), rng=rng)  # B 组权重。
            posterior_mean_a = math.fsum(weight * value for weight, value in zip(weights_a, values_a))  # A 组后验均值。
            posterior_mean_b = math.fsum(weight * value for weight, value in zip(weights_b, values_b))  # B 组后验均值。
            posterior_differences.append(posterior_mean_b - posterior_mean_a)  # 记录后验均值差。

    posterior_differences.sort()  # 对后验样本排序，便于取分位数。
    alpha = (1.0 - confidence_level) / 2.0  # 两边尾部比例。
    lower_index = max(0, min(len(posterior_differences) - 1, int(math.floor(alpha * (len(posterior_differences) - 1)))))  # 下界索引。
    upper_index = max(  # 上界索引可能要跨多行写清楚。
        0,  # 下限不能小于 0。
        min(len(posterior_differences) - 1, int(math.ceil((1.0 - alpha) * (len(posterior_differences) - 1)))),  # 上限不能超过最后一个元素。
    )  # upper_index 计算结束。
    if direction == "lower_is_better":  # 如果这个指标是越小越好，就看差值小于 0 的概率。
        prob_b_better = math.fsum(1.0 for value in posterior_differences if value < 0.0) / len(posterior_differences)  # B 更优的概率。
    else:  # 否则默认越大越好。
        prob_b_better = math.fsum(1.0 for value in posterior_differences if value > 0.0) / len(posterior_differences)  # B 更优的概率。
    prob_a_better = max(0.0, min(1.0, 1.0 - prob_b_better))  # A 更优概率作为互补值。

    return {  # 返回后验结果字典。
        "method": "bayesian_bootstrap_mean_difference",  # 标注方法名。
        "direction": str(direction),  # 记录方向约定。
        "confidence_level": confidence_level,  # 记录置信水平。
        "lower": posterior_differences[lower_index],  # 后验区间下界。
        "upper": posterior_differences[upper_index],  # 后验区间上界。
        "observed": observed_mean_difference,  # 观测到的均值差。
        "prob_b_better": prob_b_better,  # B 更优概率。
        "prob_a_better": prob_a_better,  # A 更优概率。
    }  # 后验结果字典结束。


def _compute_wilcoxon_signed_rank_p_value(differences: Sequence[float]) -> tuple[float, float]:  # 计算配对差值的 Wilcoxon p 值。
    """计算配对样本的 Wilcoxon signed-rank p 值与 rank-biserial 效应量。"""
    nonzero_differences = [diff for diff in differences if diff != 0.0]  # 先去掉差值为 0 的样本。
    if not nonzero_differences:  # 如果全是 0，就没有可检验的变化。
        return 1.0, 0.0  # 返回不显著且效应量为 0。

    abs_differences = [abs(diff) for diff in nonzero_differences]  # 取绝对差值作为排序键。
    ranks, tie_sizes = _average_ranks(abs_differences)  # 复用统一秩计算，避免重算（D2 分层边界）。

    w_plus = math.fsum(rank for rank, diff in zip(ranks, nonzero_differences) if diff > 0.0)  # 正差值对应的秩和。
    w_minus = math.fsum(rank for rank, diff in zip(ranks, nonzero_differences) if diff < 0.0)  # 负差值对应的秩和。
    w_stat = min(w_plus, w_minus)  # Wilcoxon 统计量取较小者。
    n = len(nonzero_differences)  # 有效样本数。
    mean_w = n * (n + 1) / 4.0  # 统计量均值。
    tie_term = math.fsum((tie_size ** 3) - tie_size for tie_size in tie_sizes)  # tie 修正项。
    variance_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0  # 方差公式。
    if variance_w <= 0.0:  # 方差异常时不能继续。
        return 1.0, 0.0  # 返回保守结果。

    # 连续性校正：从偏离方向减 0.5；当 |w_stat - mean_w| < 0.5 时
    # 校正会把分子压到负数（W 落在均值附近，无实质偏离），此时钳到 0，
    # 让 z 为 0、p 值为 1.0，与 scipy.stats.wilcoxon(correction=True) 行为一致。
    # w_stat = min(w_plus, w_minus) <= mean_w，故用 abs 取偏离幅度。
    z_score = max(abs(w_stat - mean_w) - 0.5, 0.0) / math.sqrt(variance_w)  # 带连续性校正的非负 z 分数。
    p_value = 2.0 * (1.0 - NormalDist().cdf(z_score))  # z_score 已非负，直接取上侧尾概率做双侧 p 值。
    rank_biserial = 0.0 if (w_plus + w_minus) == 0.0 else (w_plus - w_minus) / (w_plus + w_minus)  # rank-biserial 效应量。
    return min(max(p_value, 0.0), 1.0), rank_biserial  # 返回 p 值和效应量。


def _extract_paired_samples(  # 尝试从 metric_rows 里提取配对样本。
    metric_rows: Sequence[Mapping[str, object]],  # 原始指标行。
    *,  # 后面的参数只允许关键字调用。
    group_keys: Sequence[str],  # 分组键。
    metric_names: Sequence[str],  # 当前要检验的指标集合。
    pairing_keys: Sequence[str] | None,  # 配对键，如果有就优先用。
    metric_name: str,  # 当前正在处理的指标名。
    group_a: tuple,  # 分组 A 的坐标。
    group_b: tuple,  # 分组 B 的坐标。
) -> tuple[list[float], list[float], list[str]] | None:  # 成功时返回两组配对样本和配对字段，否则返回 None。
    """从长表或宽表里提取可配对的两组样本。"""
    long_form_rows = [("metric" in metric_row) or ("value" in metric_row) for metric_row in metric_rows]  # 判断是否是长表格式。
    if any(long_form_rows) and not all(long_form_rows):  # 长表和宽表不能混着来。
        return None  # 混用时直接放弃配对。

    if all(long_form_rows):  # 长表模式。
        if pairing_keys:  # 如果显式给了配对键，就直接使用。
            candidate_fields = list(pairing_keys)  # 复制一份候选配对字段。
        else:  # 否则从行字段里推断。
            candidate_fields = [key for key in metric_rows[0] if key not in set(group_keys) | {"metric", "value"}]  # 取剩余字段作为配对键。
        if not candidate_fields:  # 没有任何配对字段就没法配对。
            return None  # 放弃配对。
        pair_map: dict[tuple, dict[tuple, float]] = {}  # 用 pair_key -> group_value -> value 建映射。
        for row_index, row in enumerate(metric_rows):  # 逐行扫描。
            if row["metric"] != metric_name:  # 当前行不是目标指标就跳过。metric 是长表 required key，用 [] 防止静默失败。
                continue  # 继续下一个。
            group_value = tuple(row[group_key] for group_key in group_keys)  # 当前行属于哪个分组。
            if group_value not in {group_a, group_b}:  # 只关心这两个组。
                continue  # 其他组不参与。
            if any(field not in row for field in candidate_fields):  # 配对字段缺失就不能配对。
                return None  # 直接放弃。
            pair_key = tuple((field, row[field]) for field in candidate_fields)  # 用配对字段拼出稳定配对键。
            group_entry = pair_map.setdefault(pair_key, {})  # 同一个配对键下记录两个组的值。
            if group_value in group_entry:  # 同一个组重复出现同一个配对键，说明数据不干净。
                return None  # 放弃配对。
            group_entry[group_value] = _coerce_numeric_value(  # 把当前行的数值读出来。
                row["value"],  # 长表里用 value 字段。
                name=f"metric_table[{row_index}].value",  # 报错时指出具体位置。
            )  # 数值转换结束。
        paired_a = []  # 收集配对后的 A 组样本。
        paired_b = []  # 收集配对后的 B 组样本。
        for pair_key in sorted(pair_map):  # 按配对键排序，保证顺序稳定。
            group_entry = pair_map[pair_key]  # 取出这个配对键下的组值。
            if group_a not in group_entry or group_b not in group_entry:  # 只要一边缺失，就不能形成完整配对。
                return None  # 放弃配对。
            paired_a.append(group_entry[group_a])  # 加入 A 组值。
            paired_b.append(group_entry[group_b])  # 加入 B 组值。
        if not paired_a:  # 如果一个配对都没形成。
            return None  # 返回 None。
        return paired_a, paired_b, candidate_fields  # 返回配对样本和使用的字段。

    if metric_name not in metric_rows[0]:  # 宽表模式下，当前指标列必须存在。
        return None  # 没有这个列就不能配对。
    if pairing_keys:  # 如果显式给了配对键。
        candidate_fields = list(pairing_keys)  # 直接使用。
    else:  # 否则从宽表里推断配对字段。
        candidate_fields = [key for key in metric_rows[0] if key not in set(group_keys) | set(metric_names)]  # 取剩余字段。
    if not candidate_fields:  # 没字段可用就没法配对。
        return None  # 直接返回 None。
    pair_map: dict[tuple, dict[tuple, float]] = {}  # 用同样的映射结构存配对样本。
    for row_index, row in enumerate(metric_rows):  # 逐行扫描宽表。
        if metric_name not in row:  # 没有当前指标列就跳过。
            continue  # 继续下一行。
        group_value = tuple(row[group_key] for group_key in group_keys)  # 当前行属于哪个组。
        if group_value not in {group_a, group_b}:  # 只关心两个目标组。
            continue  # 其他组忽略。
        if any(field not in row for field in candidate_fields):  # 配对字段缺失就放弃。
            return None  # 不能配对。
        pair_key = tuple((field, row[field]) for field in candidate_fields)  # 生成配对键。
        group_entry = pair_map.setdefault(pair_key, {})  # 记录两个组的值。
        if group_value in group_entry:  # 同组重复出现同一个配对键，说明数据不适合配对。
            return None  # 放弃。
        group_entry[group_value] = _coerce_numeric_value(  # 读取当前指标值。
            row[metric_name],  # 宽表里用指标列直接取值。
            name=f"metric_table[{row_index}].{metric_name}",  # 报错定位。
        )  # 数值转换结束。
    paired_a = []  # 配对后的 A 组样本。
    paired_b = []  # 配对后的 B 组样本。
    for pair_key in sorted(pair_map):  # 按配对键排序，保持顺序稳定。
        group_entry = pair_map[pair_key]  # 读取这一组的两个样本。
        if group_a not in group_entry or group_b not in group_entry:  # 任一组缺失就无法形成配对。
            return None  # 返回 None 表示不能做配对检验。
        paired_a.append(group_entry[group_a])  # 加入 A 组值。
        paired_b.append(group_entry[group_b])  # 加入 B 组值。
    if not paired_a:  # 如果还是没有样本。
        return None  # 返回 None。
    return paired_a, paired_b, candidate_fields  # 返回配对样本和字段。


def run_pairwise_tests(  # 对每个指标的组间组合做成对检验。
    grouped_values: dict[str, dict[tuple, list[float]]],  # 已按指标和分组键聚合好的值。
    group_keys: list[str],  # 分组键列表。
    metric_names: list[str],  # 需要检验的指标列表。
    metric_rows: Sequence[Mapping[str, object]] | None = None,  # 原始行，必要时用来做配对检验。
    pairing_keys: Sequence[str] | None = None,  # 可选的配对键。
) -> list[dict]:  # 返回一张平展的检验结果表。
    """对每个指标的所有组间组合执行成对统计检验。

    作用：遍历每个指标的所有分组两两组合，优先尝试配对检验
    （Wilcoxon signed-rank），无法配对时退回非配对检验
    （Mann-Whitney U）。同时计算效应量、bootstrap 置信区间
    和 Bayesian bootstrap 后验区间。

    参数:
        grouped_values: 按指标名和分组键聚合好的数值字典。
        group_keys: 分组键名称列表。
        metric_names: 需要检验的指标名列表。
        metric_rows: 原始指标行，用于提取配对样本。
        pairing_keys: 可选的配对键列表。

    返回值:
        list[dict[str, Any]]: 检验结果行列表，每行包含指标名、组对、p 值、效应量等。
    """
    test_rows = []  # 收集所有指标、所有组对的检验记录。
    for metric_name in metric_names:  # 逐个指标处理。
        metric_groups = grouped_values.get(metric_name, {})  # 拿到这个指标下的分组样本。
        for (group_a, sample_a), (group_b, sample_b) in combinations(metric_groups.items(), 2):  # 对所有组两两组合。
            paired_samples = None  # 先默认没有配对样本。
            if metric_rows is not None:  # 如果提供了原始行，就尝试提取配对样本。
                paired_samples = _extract_paired_samples(  # 从原始行里找可以配对的样本。
                    metric_rows,  # 原始指标行。
                    group_keys=group_keys,  # 分组键。
                    metric_names=metric_names,  # 指标列表。
                    pairing_keys=pairing_keys,  # 配对键。
                    metric_name=metric_name,  # 当前指标名。
                    group_a=group_a,  # A 组坐标。
                    group_b=group_b,  # B 组坐标。
                )  # 配对提取结束。

            if paired_samples is not None:  # 如果成功提取到配对样本，就做配对检验。
                paired_sample_a, paired_sample_b, pairing_fields = paired_samples  # 拆出配对样本和字段。
                if len(paired_sample_a) != len(paired_sample_b):  # 配对样本长度必须一致，防止 zip 静默截断。
                    raise ValueError("paired samples must have equal length")  # 长度不一致直接报错。
                differences = [float(b) - float(a) for a, b in zip(paired_sample_a, paired_sample_b)]  # 先算每一对的差值。
                p_value, effect_size = _compute_wilcoxon_signed_rank_p_value(differences)  # 用 Wilcoxon 做配对检验。
                confidence_interval = _bootstrap_mean_difference_ci(  # 再给均值差算 bootstrap 区间。
                    paired_sample_a,  # 配对 A 组。
                    paired_sample_b,  # 配对 B 组。
                    paired=True,  # 明确这是配对模式。
                )  # bootstrap 区间结束。
                bayes_posterior = _bayesian_bootstrap_mean_difference(  # 再算 Bayesian bootstrap 后验区间。
                    paired_sample_a,  # 配对 A 组。
                    paired_sample_b,  # 配对 B 组。
                    metric_name=metric_name,  # 当前指标名，用来判断好坏方向。
                    paired=True,  # 明确这是配对模式。
                )  # Bayesian bootstrap 结束。
                test_name = "paired_wilcoxon"  # 记录这次用的是配对 Wilcoxon。
                sample_size_a = len(paired_sample_a)  # 记录 A 组样本量。
                sample_size_b = len(paired_sample_b)  # 记录 B 组样本量。
                paired_sample_size = len(paired_sample_a)  # 配对样本数就是这两组的共同长度。
            else:  # 如果提取不到配对样本，就退回到非配对检验。
                p_value = _compute_mann_whitney_p_value(sample_a, sample_b)  # 用 Mann-Whitney U 做非配对检验。
                effect_size = _compute_cliffs_delta(sample_a, sample_b)  # 非配对效应量用 Cliff's delta。
                confidence_interval = _bootstrap_mean_difference_ci(  # 计算非配对均值差区间。
                    sample_a,  # A 组样本。
                    sample_b,  # B 组样本。
                    paired=False,  # 明确这是非配对模式。
                )  # bootstrap 区间结束。
                bayes_posterior = _bayesian_bootstrap_mean_difference(  # 计算非配对 Bayesian 后验。
                    sample_a,  # A 组样本。
                    sample_b,  # B 组样本。
                    metric_name=metric_name,  # 指标名。
                    paired=False,  # 非配对模式。
                )  # Bayesian bootstrap 结束。
                test_name = "mann_whitney"  # 记录这次用的是 Mann-Whitney。
                sample_size_a = len(sample_a)  # A 组样本量。
                sample_size_b = len(sample_b)  # B 组样本量。
                paired_sample_size = None  # 非配对模式没有配对样本数。
                pairing_fields = None  # 非配对模式也没有配对字段。
            test_rows.append(  # 把这次检验结果写成一行记录。
                {  # 一行记录开始。
                    "metric_name": metric_name,  # 当前指标名。
                    "group_a": group_a,  # A 组分组键。
                    "group_b": group_b,  # B 组分组键。
                    "p_value": p_value,  # 原始 p 值。
                    "effect_size": effect_size,  # 效应量。
                    "sample_size_a": sample_size_a,  # A 组样本数。
                    "sample_size_b": sample_size_b,  # B 组样本数。
                    "test_name": test_name,  # 检验方法名。
                    "confidence_interval": confidence_interval,  # bootstrap 区间对象。
                    "ci_low": confidence_interval["lower"],  # 区间下界。
                    "ci_high": confidence_interval["upper"],  # 区间上界。
                    "ci_method": confidence_interval["method"],  # 区间计算方法名。
                    "bayes_posterior": bayes_posterior,  # Bayesian 后验对象。
                    "bayes_method": bayes_posterior["method"],  # 后验方法名。
                    "bayes_ci_low": bayes_posterior["lower"],  # Bayesian 区间下界。
                    "bayes_ci_high": bayes_posterior["upper"],  # Bayesian 区间上界。
                    "bayes_prob_b_better": bayes_posterior["prob_b_better"],  # B 更优概率。
                    "bayes_prob_a_better": bayes_posterior["prob_a_better"],  # A 更优概率。
                    "paired_sample_size": paired_sample_size,  # 配对样本数，没有则为 None。
                    "pairing_fields": pairing_fields,  # 配对字段，没有则为 None。
                }  # 一行记录结束。
            )  # append 结束。
    return test_rows  # 返回所有 pairwise test 结果。


def _bh_fdr_adjust_p_values(p_values: Sequence[float]) -> list[float]:  # 做 BH-FDR 多重比较校正。
    """做单调的 Benjamini-Hochberg FDR 多重比较校正（非 handbook P22 路径专用）。

    使用 BH-FDR 校正公式 ``p * n / rank``（rank 为 1-indexed 升序排名），
    再从大 rank 往小 rank 取累积最小值强制单调不下降，保证校正后的 p 值
    单调性。校正结果裁剪到 [0, 1] 区间。

    适用范围：本函数是 BH-FDR 的私有实现，仅用于**非 handbook P22** 的 FDR 风格
    校正场景。手册 P22 硬约束"两两比较附 Holm-Bonferroni 多重比较校正"必须用
    ``holm_bonferroni_adjust``（控制 FWER 而非 FDR），该函数已由
    ``run_significance_tests`` 直接调用。本函数保留是为了：
    1. ``handbook_p22_statistics.run_p22_full_analysis`` 需要同时报告 Holm 与 BH-FDR 两种结果。
    2. 未来非 P22 场景仍可能用到 BH-FDR 风格校正。

    输入合同：p_values 的元素必须是 [0, 1] 区间内的有限 float。非数值类型
    （含 bool/bytearray/numpy.str_）、NaN/Inf、超出 [0, 1] 区间的值均视为
    值合同违例，抛出 ValueError，与项目硬约束对齐。

    参数:
        p_values: 原始 p 值序列，每个值必须在 [0, 1] 区间内且为有限浮点数。

    返回值:
        list[float]: 校正后的 p 值列表，顺序与输入一致，每个值在 [0, 1] 区间内。

    异常:
        ValueError: p_values 包含非数值类型（含 bool）、NaN/Inf 或超出 [0, 1] 的值。
    """
    if not p_values:  # 空列表没有要校正的值。
        return []  # 直接返回空列表。

    # 入口校验：每个 p 值必须是 [0, 1] 区间内的有限 float。
    # 复用 coerce_finite_scalar 做类型/有限性/范围统一校验；bool/bytearray/
    # numpy.str_ 等非数值类型触发的 TypeError 转为 ValueError，与项目硬约束
    # "value contract 问题应抛 ValueError"及 _coerce_numeric_value 模式对齐。
    validated_p_values: list[float] = []
    for index, raw_p_value in enumerate(p_values):
        try:
            validated_p_value = coerce_finite_scalar(  # 统一类型/有限性/范围校验。
                raw_p_value,  # 原始 p 值。
                name=f"p_values[{index}]",  # 报错时指出具体位置。
                min_value=0.0,  # p 值下界。
                max_value=1.0,  # p 值上界。
            )  # 校验结束。
        except TypeError as exc:  # 非数值类型（含 bool/bytearray/numpy.str_）触发。
            if "must be numeric" in str(exc):  # 值合同违例转 ValueError。
                raise ValueError(str(exc)) from exc  # 保持消息不变，仅切换异常类型。
            raise  # 其它 TypeError 原样向上传播。
        validated_p_values.append(validated_p_value)  # 收集校验后的 float。

    ranked_p_values = sorted(enumerate(validated_p_values), key=lambda item: item[1])  # 按原始 p 值从小到大排序，同时保留原索引。
    total_tests = len(validated_p_values)  # 总共做了多少次检验。
    adjusted_ranked_p_values = [1.0] * total_tests  # 先准备一个全 1 的校正结果数组。

    for rank_index in range(total_tests - 1, -1, -1):  # 从大 p 值往小 p 值回推。
        _, raw_p_value = ranked_p_values[rank_index]  # 取出当前排序位置的原始 p 值。
        adjusted_ranked_p_values[rank_index] = min(  # 做保守校正并限制上界为 1。
            1.0,  # p 值不可能超过 1。
            raw_p_value * total_tests / (rank_index + 1),  # 这是按排序位置做的保守修正。
        )  # 当前位置校正结束。

    for rank_index in range(total_tests - 2, -1, -1):  # 再从后往前强制单调不下降。
        adjusted_ranked_p_values[rank_index] = min(  # 保证前面的值不会比后面更大。
            adjusted_ranked_p_values[rank_index],  # 当前校正值。
            adjusted_ranked_p_values[rank_index + 1],  # 后一个位置的校正值。
        )  # 单调化结束。

    adjusted_p_values = [1.0] * total_tests  # 再开一个数组放回原始顺序。
    for rank_index, (original_index, _) in enumerate(ranked_p_values):  # 把排序后的校正值映射回原位置。
        adjusted_p_values[original_index] = adjusted_ranked_p_values[rank_index]  # 回填校正后的 p 值。
    return adjusted_p_values  # 返回按原始顺序排列的校正结果。


def build_statistics_table(  # 把 pairwise 检验结果打平为标准统计表。
    test_rows: Sequence[Mapping[str, Any]],  # 原始 pairwise 检验行。
    adjusted_p_values: Sequence[float],  # 校正后的 p 值列表。
    group_keys: list[str],  # 分组键名称列表。
) -> list[dict[str, Any]]:  # 返回扁平化的 statistics_table。
    """把成对检验结果和校正后 p 值合并为扁平化的统计表。

    作用：将 run_pairwise_tests 的输出与多重比较校正后的 p 值
    一一配对，拆分分组坐标为单独列，输出标准化的统计结果表。

    参数:
        test_rows: 成对检验结果行列表。每行必须包含必需键 metric_name、p_value、
            effect_size、sample_size_a、sample_size_b、group_a、group_b；其中
            group_a 与 group_b 的长度必须等于 group_keys 的长度，否则视为值合同违例。
        adjusted_p_values: 与 test_rows 等长的校正后 p 值列表。
        group_keys: 分组键名称列表，用于拆分 group_a/group_b 为单独列。

    返回值:
        list[dict[str, Any]]: 扁平化的统计表，每行包含指标名、p 值、校正 p 值、效应量等。

    异常:
        ValueError: test_rows 和 adjusted_p_values 长度不一致、test_row 缺少必需键、
            或 group_a/group_b 长度与 group_keys 不一致时抛出。
    """
    if len(test_rows) != len(adjusted_p_values):  # 两个列表长度必须一致。
        raise ValueError("test_rows and adjusted_p_values must have the same length")  # 不一致就报错。

    required_keys = (  # 每行 test_row 必须包含的必需键，缺键属值合同违例。
        "metric_name",  # 指标名。
        "p_value",  # 原始 p 值。
        "effect_size",  # 效应量。
        "sample_size_a",  # A 组样本数。
        "sample_size_b",  # B 组样本数。
        "group_a",  # A 组分组坐标。
        "group_b",  # B 组分组坐标。
    )  # 必需键列表结束。

    statistics_table = []  # 收集最终的统计行。
    for test_row, adjusted_p_value in zip(test_rows, adjusted_p_values):  # 一行检验结果配一个校正后的 p 值。
        missing_keys = [key for key in required_keys if key not in test_row]  # 找出缺失的必需键。
        if missing_keys:  # 缺键属值合同违例，按项目硬约束 raise ValueError 而非 KeyError。
            raise ValueError(f"test_row is missing required keys: {missing_keys}")  # 明确指出缺失项。
        group_a = test_row["group_a"]  # A 组分组坐标。
        group_b = test_row["group_b"]  # B 组分组坐标。
        if len(group_a) != len(group_keys) or len(group_b) != len(group_keys):  # 分组坐标长度必须与 group_keys 一致，防止索引越界静默失败。
            raise ValueError(  # 长度不一致属值合同违例。
                f"group_a/group_b length must equal group_keys length: "
                f"got len(group_a)={len(group_a)}, len(group_b)={len(group_b)}, "
                f"len(group_keys)={len(group_keys)}"
            )  # 长度守卫结束。
        statistics_row = {  # 先构造最核心的列。
            "metric_name": test_row["metric_name"],  # 指标名。
            "p_value": test_row["p_value"],  # 原始 p 值。
            "effect_size": test_row["effect_size"],  # 效应量。
            "adjusted_p_value": adjusted_p_value,  # 校正后的 p 值。
            "sample_size_a": test_row["sample_size_a"],  # A 组样本数。
            "sample_size_b": test_row["sample_size_b"],  # B 组样本数。
        }  # 基础统计列结束。
        for optional_key in (  # 下面这些列有则保留，没有也不强求。
            "test_name",  # 检验方法名。
            "confidence_interval",  # bootstrap 区间对象。
            "ci_low",  # 区间下界。
            "ci_high",  # 区间上界。
            "ci_method",  # 区间方法名。
            "bayes_posterior",  # Bayesian 后验对象。
            "bayes_method",  # Bayesian 方法名。
            "bayes_ci_low",  # Bayesian 下界。
            "bayes_ci_high",  # Bayesian 上界。
            "bayes_prob_b_better",  # B 更优概率。
            "bayes_prob_a_better",  # A 更优概率。
            "paired_sample_size",  # 配对样本数。
            "pairing_fields",  # 配对字段。
        ):  # 可选键遍历结束。
            if optional_key in test_row and test_row[optional_key] is not None:  # 只有存在且不为空才保留。
                statistics_row[optional_key] = test_row[optional_key]  # 把可选信息写进结果。
        for group_index, group_key in enumerate(group_keys):  # 把分组坐标拆回单独列。
            statistics_row[f"{group_key}_a"] = group_a[group_index]  # 写入 A 组对应维度。
            statistics_row[f"{group_key}_b"] = group_b[group_index]  # 写入 B 组对应维度。
        statistics_table.append(statistics_row)  # 把这一行加入最终表。
    return statistics_table  # 返回扁平化的统计表。


def run_significance_tests(  # 对外入口：从原始表直接跑完整统计检验。
    metric_table: Any,  # 输入的指标表，支持 DataFrame、单个映射或映射序列。
    group_keys: list[str],  # 分组键列表。
    metric_names: list[str],  # 需要检验的指标列表。
    pairing_keys: list[str] | None = None,  # 可选的配对键列表。
) -> list[dict[str, Any]]:  # 函数签名结束，返回扁平化的统计结果表。
    """对指标表执行完整的成对统计检验和多重比较校正。

    作用：这是本模块的对外统一入口。它先规范化输入，
    再按分组聚合指标值，然后对每个指标的所有组间组合
    执行成对检验（优先配对 Wilcoxon，退回 Mann-Whitney U），
    最后做多重比较校正并输出标准统计表。

    参数:
        metric_table: 指标表，支持 DataFrame、单个映射或映射序列。
        group_keys: 分组键列表，如 ["method_name"]。
        metric_names: 需要检验的指标名列表。
        pairing_keys: 可选的配对键列表，用于配对检验。

    返回值:
        list[dict[str, Any]]: 扁平化的统计结果表，包含 p 值、校正 p 值、效应量等。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "group_keys": list(group_keys) if group_keys is not None else None,
            "metric_names": list(metric_names) if metric_names is not None else None,
            "pairing_keys": list(pairing_keys) if pairing_keys is not None else None,
            "metric_table_type": type(metric_table).__name__,
        },
        "run_significance_tests 入口参数",
        prefix="[analysis]",
    )
    normalized_group_keys = _normalize_name_list(group_keys, name="group_keys")  # 先规范化分组键。
    requested_metric_names = _normalize_name_list(metric_names, name="metric_names")  # 再规范化请求的指标名。
    ordered_metric_names = _validate_metric_names(requested_metric_names)  # 再按协议顺序过滤。
    normalized_pairing_keys = (  # 配对键如果提供，也要做同样规范化。
        _normalize_name_list(pairing_keys, name="pairing_keys")  # 对配对键做去重和空白清理。
        if pairing_keys is not None  # 只有传了才处理。
        else None  # 没传就保持 None。
    )  # 配对键规范化结束。

    # 先把 metric_table 规范化成行列表（_normalize_metric_rows 会 materialize 成 list），
    # 再把同一份行列表喂给 group_metric_values 与 run_pairwise_tests。这样对一次性迭代器
    # 只消费一次，避免第二次 _normalize_metric_rows 调用拿到已耗尽迭代器而静默抛
    # "metric_table must be non-empty"（D10 动态链路守卫）。对 list[dict] 输入幂等无副作用。
    metric_rows = _normalize_metric_rows(metric_table)  # 统一成行列表，供分组与配对逻辑共用。
    grouped_values = group_metric_values(metric_rows, normalized_group_keys, ordered_metric_names)  # 再按组把指标值聚合起来。
    test_rows = run_pairwise_tests(  # 对每个指标、每对组做检验。
        grouped_values,  # 已经分组好的值。
        normalized_group_keys,  # 规范化后的分组键。
        ordered_metric_names,  # 协议顺序的指标名。
        metric_rows,  # 原始行。
        pairing_keys=normalized_pairing_keys,  # 可选配对字段。
    )  # pairwise tests 结束。
    p_values = [test_row["p_value"] for test_row in test_rows]  # 提取所有原始 p 值。
    # 手册 P22 硬约束：两两比较用 Holm-Bonferroni 控制 FWER。
    adjusted_p_values = holm_bonferroni_adjust(p_values)  # 做多重比较校正。
    statistics_table = build_statistics_table(test_rows, adjusted_p_values, normalized_group_keys)  # 组装最终统计表。
    return statistics_table  # 返回统一的统计结果。
