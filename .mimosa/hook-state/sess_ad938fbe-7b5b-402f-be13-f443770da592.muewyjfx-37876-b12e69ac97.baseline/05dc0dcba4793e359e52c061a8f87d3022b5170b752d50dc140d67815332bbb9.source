"""文件:src/liquidloc/metrics/reliability_metrics.py  # 文件头：说明这个文件的身份。

【文件职责】把风险轨迹和误差轨迹合在一起，计算可靠性相关指标。  # 说明文件做什么。
它存在的原因是把"风险是否真的和误差相关、覆盖率够不够、支持长度够不够"  # 说明为什么存在。
这些判断集中在一个地方，避免分析面自己重复实现。  # 说明集中实现的好处。

【上游依赖】  # 说明本文件依赖哪些模块。
- 无外部项目模块依赖，仅使用标准库 math 和 collections.abc。  # 只依赖标准库。

【下游调用者】  # 说明谁会用这个文件。
- metric_runner.py：通过 compute_reliability_metrics 调用，作为四类指标之一。  # 总调度器。
- 校准分析脚本：直接调用本模块分析风险与误差的相关性。  # 分析面。
- 需要可靠性诊断的测试代码：直接调用本模块检查覆盖率与状态。  # 测试面。

【输入对象定义】  # 说明输入对象。
- risk_trace: 风险值序列，元素为数值或 None。  # 风险轨迹。
- error_trace: 与风险对齐的误差序列，元素为数值或 None。  # 误差轨迹。
- support_context: 可选支持上下文映射，包含 prediction_length/ground_truth_length/aligned_length。  # 支持上下文。

【输出对象定义】  # 说明输出对象。
- dict[str, float | int | str]：包含 coverage、risk_error_corr、valid_pair_count、reliability_status 四个键的指标字典。  # 可靠性指标。

【核心变量定义】  # 说明核心中间变量。
- aligned_risk: 过滤后的有效风险样本列表。  # 有效风险。
- aligned_error: 过滤后的有效误差样本列表。  # 有效误差。
- valid_pair_count: 有效配对样本数。  # 有效样本数。
- coverage: 有效样本覆盖率。  # 覆盖率。
- risk_error_corr: 风险与误差的皮尔逊相关系数。  # 相关系数。
- reliability_status: 可靠性状态字符串，"ok" 或 "insufficient_overlap"。  # 状态。

【推荐编写顺序】1. 先看 compute_reliability_metrics 入口。2. 再看 _normalize_support_context 辅助函数。  # 推荐阅读顺序。
"""

from __future__ import annotations  # 允许类型注解中使用现代写法（如 X | None）。

import math  # 提供有限性检查、平方根等数学操作。
from collections.abc import Iterable, Mapping  # 用于识别可迭代对象和映射。

from liquidloc.common.constants import RELIABILITY_STATUS_OK, RELIABILITY_STATUS_INSUFFICIENT  # 可靠性状态字符串单源真相，禁止本地硬编码 "ok"/"insufficient_overlap"。
from liquidloc.common.validation import is_bool_like, is_integer  # 集中判断 bool / np.bool_ 和整数类型。


def _normalize_support_context(support_context: Mapping | None, *, aligned_length: int) -> dict[str, int]:
    """把支持上下文规范成固定字段的整数映射。  # 函数作用总述。

    作用：把外部传入的 support_context 统一成固定字段，避免后续统计口径漂移。  # 作用。
    参数：  # 下面列参数。
    - support_context: 可能是 None，也可能是带长度字段的映射。  # 参数说明。
    - aligned_length: 当前误差轨迹的实际对齐长度，用来校验支持上下文是否一致。  # 参数说明。
    返回值：  # 下面列返回。
    - dict[str, int]：规范后的支持上下文字典。  # 返回值说明。
    异常/失败条件：  # 下面列失败条件。
    - support_context 类型不对时抛 TypeError。  # 类型失败。
    - 字段缺失、非整数或非正数时抛 TypeError / ValueError。  # 值失败。
    - aligned_length 与上下文不一致时抛 ValueError。  # 一致性失败。
    状态变化：  # 说明是否改状态。
    - 不修改原对象，只返回新字典。  # 状态变化说明。
    """
    # 入口校验 aligned_length：必须是非布尔正整数，防御误传 0/负数/非整数，
    # 也让 None 分支与字段校验分支的“正数”口径保持一致。
    if not is_integer(aligned_length):  # 排除 bool / numpy.bool_ 及非整数类型。
        raise TypeError("aligned_length must be an integer")  # 类型不对直接报错。
    if aligned_length <= 0:  # 长度必须是正数。
        raise ValueError("aligned_length must be positive")  # 非正数直接拒绝。
    aligned_length = int(aligned_length)  # 统一成 Python int，兑现返回类型 dict[str, int] 契约。
    if support_context is None:  # 没传时，直接用对齐长度作为默认支持长度。
        return {  # 返回一个新字典，不改原对象。
            "prediction_length": aligned_length,  # 默认预测长度等于当前对齐长度。
            "ground_truth_length": aligned_length,  # 默认真值长度也等于当前对齐长度。
            "aligned_length": aligned_length,  # 对齐长度保持原值。
        }
    if not isinstance(support_context, Mapping):  # 传入值必须是映射，才能按字段读取。
        raise TypeError("support_context must be a mapping when provided")  # 类型不对直接报错。

    normalized: dict[str, int] = {}  # 保存规范化后的支持上下文。
    for field_name in ("prediction_length", "ground_truth_length", "aligned_length"):  # 逐个校验三个必需字段。
        field_value = support_context.get(field_name)  # 从外部上下文里取出当前字段。
        if not is_integer(field_value):  # 布尔值和非整数都不接受。
            raise TypeError(f"support_context['{field_name}'] must be an integer")  # 明确指出字段错误。
        if field_value <= 0:  # 长度必须是正数。
            raise ValueError(f"support_context['{field_name}'] must be positive")  # 非正数直接拒绝。
        normalized[field_name] = int(field_value)  # 显式转 Python int，防止 numpy 标量流入返回字典破坏类型契约。

    if normalized["aligned_length"] != aligned_length:  # 对齐长度必须和实际轨迹长度一致。
        raise ValueError("support_context['aligned_length'] must match the aligned trace length")  # 一致性不满足时报错。
    if normalized["prediction_length"] < aligned_length:  # 预测长度不能短于对齐长度（每个对齐点对应一个预测点）。
        raise ValueError(  # 分行只是为了读起来清楚，不改异常语义。
            "support_context prediction_length must be greater than or equal to the aligned trace length"  # 明确要求预测长度至少覆盖对齐部分。
        )
    return normalized  # 返回规范化后的支持上下文。


def compute_reliability_metrics(
    risk_trace: Iterable[float | None],
    error_trace: Iterable[float | None],
    *,
    support_context: Mapping[str, int] | None = None,
) -> tuple[dict[str, float], dict[str, int | str]]:
    """从对齐的风险轨迹和误差轨迹计算可靠性指标。  # 入口函数说明。

    作用：把风险轨迹和误差轨迹配对后，计算覆盖率、相关性和有效样本数。  # 详细作用。
    参数：  # 参数说明。
    - risk_trace: 风险值序列。  # 参数说明。
    - error_trace: 与风险对齐的误差序列。  # 参数说明。
    - support_context: 可选支持上下文，用于覆盖率和状态判断。  # 参数说明。
    返回值：  # 返回值说明。
    - tuple[dict[str, float], dict[str, int | str]]：可靠性指标字典与支撑报告元组。  # 返回值说明。
    异常/失败条件：  # 失败条件说明。
    - 输入不是非字符串可迭代对象时抛 TypeError。  # 类型错误。
    - 输入为空、长度不等、元素不是数值或 None 时抛异常。  # 数据错误。
    状态变化：  # 状态变化说明。
    - 不修改输入，只返回新字典。  # 不改状态。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "support_context": dict(support_context) if support_context is not None else None,
            "risk_trace_type": type(risk_trace).__name__,
            "error_trace_type": type(error_trace).__name__,
        },
        "compute_reliability_metrics 入口参数",
        prefix="[metrics]",
    )
    if isinstance(risk_trace, (str, bytes)) or isinstance(error_trace, (str, bytes)):  # 字符串不是合法数值序列。
        raise TypeError("risk_trace and error_trace must be non-string iterables")  # 直接拒绝字符串。

    try:  # 尝试把输入都转成列表，方便后续多次遍历和长度比较。
        risk_trace = list(risk_trace)  # 统一成列表，避免一次性迭代器被耗尽。
        error_trace = list(error_trace)  # 统一成列表，避免一次性迭代器被耗尽。
    except TypeError as exc:  # 如果不能迭代，就说明输入类型不对。
        raise TypeError("risk_trace and error_trace must be iterable") from exc  # 保留原始异常链，方便定位。

    if not risk_trace or not error_trace:  # 任一序列为空都没有意义。
        raise ValueError("risk_trace and error_trace must not be empty")  # 空序列直接拒绝。
    if len(risk_trace) != len(error_trace):  # 风险和误差必须一一对齐。
        raise ValueError("risk_trace and error_trace must have the same length")  # 长度不同无法配对。

    normalized_support = _normalize_support_context(support_context, aligned_length=len(risk_trace))  # 统一支持上下文口径。
    aligned_risk = []  # 保存过滤后的风险样本。
    aligned_error = []  # 保存过滤后的误差样本。
    for risk_value, error_value in zip(risk_trace, error_trace):  # 一一配对遍历风险和误差。
        if risk_value is None or error_value is None:  # 任一侧缺失就跳过，不参与统计。
            continue  # 缺失样本不统计。
        if is_bool_like(risk_value) or is_bool_like(error_value):  # 布尔值不能当成数值。
            raise ValueError("risk_trace and error_trace must contain numeric values or None")  # 说明元素类型不合法。
        try:  # 尝试把元素转成浮点数，统一后续计算。
            risk_value = float(risk_value)  # 风险值转浮点数。
            error_value = float(error_value)  # 误差值转浮点数。
        except (OverflowError, TypeError, ValueError) as exc:  # 任何一侧无法转数值都不合法（含超大值溢出）。
            raise ValueError("risk_trace and error_trace must contain numeric values or None") from exc  # 保留异常链。
        if not (math.isfinite(risk_value) and math.isfinite(error_value)):  # NaN/inf 不参与统计。
            continue  # 跳过非有限值。
        aligned_risk.append(risk_value)  # 收集有效风险值。
        aligned_error.append(error_value)  # 收集有效误差值。

    valid_pair_count = len(aligned_risk)  # 有效配对样本数。
    denominator = max(  # 覆盖率分母取两边长度较大者。
        normalized_support["prediction_length"],  # 预测长度。
        normalized_support["ground_truth_length"],  # 真值长度。
    )
    coverage = valid_pair_count / denominator  # 计算覆盖率。
    reliability_status = (  # 根据支持量判断状态。
        RELIABILITY_STATUS_INSUFFICIENT  # 支持不足时的状态字符串（引用单源真相常量）。
        if normalized_support["aligned_length"] < 2 or valid_pair_count < 2  # 对齐太短或有效样本太少。
        else RELIABILITY_STATUS_OK  # 否则认为支持正常（引用单源真相常量）。
    )

    if valid_pair_count < 2:  # 少于两个有效样本时无法算相关性。
        risk_error_corr = 0.0  # 相关系数退化为 0。
    else:
        risk_mean = math.fsum(aligned_risk) / valid_pair_count  # 风险均值。
        error_mean = math.fsum(aligned_error) / valid_pair_count  # 误差均值。

        covariance = math.fsum(  # 协方差分子。
            (risk_value - risk_mean) * (error_value - error_mean)  # 每个样本对的中心化乘积。
            for risk_value, error_value in zip(aligned_risk, aligned_error)  # 逐对遍历有效样本。
        )
        risk_variance = math.fsum((risk_value - risk_mean) ** 2 for risk_value in aligned_risk)  # 风险方差。
        error_variance = math.fsum((error_value - error_mean) ** 2 for error_value in aligned_error)  # 误差方差。
        denominator = math.sqrt(risk_variance * error_variance)  # 相关系数分母。
        risk_error_corr = covariance / denominator if denominator > 0.0 else 0.0  # 分母为 0 时退化为 0。
        risk_error_corr = max(-1.0, min(1.0, risk_error_corr))  # 限制到合法相关系数范围内。

    reliability_metric_dict = {  # 组装最终可靠性指标字典（仅float字段，符合metric_schema）。
        "coverage": coverage,  # 覆盖率。
        "risk_error_corr": risk_error_corr,  # 风险与误差相关系数。
    }
    # valid_pair_count 和 reliability_status 不在 metric_schema 中注册，
    # 通过 support_report 传递，供上层诊断和覆盖率判断使用。
    support_report = {
        "valid_pair_count": valid_pair_count,  # 有效配对数量。
        "reliability_status": reliability_status,  # 可靠性状态字符串。
    }
    return reliability_metric_dict, support_report  # 返回可靠性指标和支撑报告。
