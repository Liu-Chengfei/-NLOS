"""
汇总主表、统计结果和案例索引，形成最终 summary 文件。

这个模块负责把 metric_table、statistics_table 和 selected_cases
统一包装成一个可落盘、可校验、可供下游直接消费的 summary 结构。
它不重新计算指标，也不做统计检验，只负责把已有结果整理成标准输出。
"""

from __future__ import annotations  # 允许类型注解里引用当前模块后面定义的内容。

from collections.abc import Iterable, Mapping, Sequence  # 用这些抽象基类判断输入容器类型。
from typing import Any  # 用 Any 接住不确定的输入对象，后面再做运行时校验。

from liquidloc.common.constants import CASE_GROUP_NAMES, LONG_FORM_METRIC_KEY, LONG_FORM_VALUE_KEY  # D9 单源常量：长表字段名与案例分组名。
from liquidloc.common.validation import is_string_like  # 复用统一字符串类型校验。
from liquidloc.analysis.case_selector import _extract_case_ref  # D9 单源：复用权威定义，禁止本地重复。
from liquidloc.protocol.result_schema import SummaryResult, validate_summary_result  # 复用协议里的结果结构和校验函数。


def _is_long_form_metric_row(row: Mapping[str, Any]) -> bool:  # 判断一行是不是被误传进来的长表 metric_table 记录。
    """判断一行是否长得像 eval_pipeline 导出的 long-form metric_table 行。"""
    return LONG_FORM_METRIC_KEY in row and LONG_FORM_VALUE_KEY in row  # D9：引用单源常量，禁止本地字面量。


def _validate_main_table_ref(table_ref: Any, *, name: str) -> None:  # 主表必须是聚合视图，不能退回 long-form metric_table。
    """拒绝把 long-form metric_table 行误当成 main_table 聚合视图。"""
    if isinstance(table_ref, Mapping):  # 单行主表也不能长成 metric/value 长表记录。
        if _is_long_form_metric_row(table_ref):
            raise ValueError(f"{name} must contain aggregated main_table rows, not long-form metric_table rows")
        return
    if isinstance(table_ref, Sequence) and not isinstance(table_ref, (str, bytes, bytearray)):
        for index, row in enumerate(table_ref):
            if isinstance(row, Mapping) and _is_long_form_metric_row(row):
                raise ValueError(
                    f"{name}[{index}] must be an aggregated main_table row, not a long-form metric_table row"
                )


def _normalize_table_ref(table_obj: Any, *, name: str) -> dict[str, Any] | list[dict[str, Any]]:  # 把输入表统一成可序列化的引用结构。
    """把表对象统一成可序列化的引用结构。"""
    if isinstance(table_obj, Mapping):  # 如果本来就是映射，就按字典处理。
        if not table_obj:  # 空字典没法作为有效引用。
            raise ValueError(f"{name} must be non-empty")  # 直接拒绝空输入。
        return dict(table_obj)  # 复制一份，避免外部对象被修改。

    if isinstance(table_obj, Sequence) and not (is_string_like(table_obj) or isinstance(table_obj, (bytes, bytearray))):  # 也允许列表式输入，但不能是字符串（含 numpy.str_）。
        table_rows = list(table_obj)  # 先转成 list，便于遍历和长度检查。
        if not table_rows:  # 空列表同样不允许。
            raise ValueError(f"{name} must be non-empty")  # 空输入直接报错。

        normalized_rows: list[dict[str, Any]] = []  # 用这个列表保存规范化后的每一行。
        for index, row in enumerate(table_rows):  # 每一行都要检查类型。
            if not isinstance(row, Mapping):  # 行本身必须是映射，不能是纯值或列表。
                raise TypeError(f"{name}[{index}] must be a mapping, got {type(row).__name__}")  # 报错里指出具体位置。
            normalized_rows.append(dict(row))  # 复制行内容，避免改到外部对象。
        return normalized_rows  # 返回规范化后的行列表。

    raise TypeError(f"{name} must be a mapping or a non-string sequence of mappings")  # 其他类型一律拒绝。


def _collect_case_refs(selected_cases: Any) -> list[str]:  # 把 selected_cases 里的案例引用合并成一个去重列表。
    """把 selected_cases 里的案例引用合并成一个去重后的列表。"""
    if not isinstance(selected_cases, Mapping):  # selected_cases 必须是映射，才能按组名读取。
        raise TypeError(f"selected_cases must be a mapping, got {type(selected_cases).__name__}")  # 类型不对就报错。

    # selected_cases 必须精确包含三个必需分组（与 case_selector.select_cases 输出合同对齐）：
    # 缺键属于值契约违规，按项目硬约束用 ValueError 而非 KeyError；多余键会破坏 case_refs 来源可追溯性。
    required_case_groups: tuple[str, ...] = CASE_GROUP_NAMES  # D9：引用单源常量，禁止本地字面量。
    missing_groups = [name for name in required_case_groups if name not in selected_cases]  # 缺键会让 build_summary 静默返回空 case_refs，掩盖输入错误。
    if missing_groups:  # 缺键属于值契约违规，不能静默失败。
        raise ValueError(f"selected_cases is missing required groups: {missing_groups}")  # 显式报错，防止空 dict 静默返回 []。
    unexpected_groups = [name for name in selected_cases if name not in required_case_groups]  # 多余键违反 case_selector 输出合同。
    if unexpected_groups:  # 多余键会让 case_refs 混入非预期组的案例。
        raise ValueError(f"selected_cases contains unsupported groups: {unexpected_groups}")  # 拒绝未知分组，保证协议一致。

    case_refs: list[str] = []  # 这里保存最终的引用顺序。
    seen_case_refs: set[str] = set()  # 这里记住已经出现过的引用，防止重复。

    for group_name, case_group in selected_cases.items():  # 逐个分组读取。
        if not is_string_like(group_name):  # 组名必须是字符串。
            raise TypeError(  # 不是字符串就报错。
                f"selected_cases group names must be strings, got {type(group_name).__name__}"  # 报错里写出真实类型。
            )  # raise 参数结束。
        if not str(group_name).strip():  # 空白组名没意义。
            raise ValueError("selected_cases group names must be non-empty strings")  # 空组名不允许。
        if isinstance(case_group, (str, bytes, bytearray, Mapping)) or not isinstance(case_group, Iterable):  # 每个分组应该是案例列表，而不是单个对象。
            raise TypeError(f"selected_cases.{group_name} must be an iterable of case mappings")  # 类型不对就报错。

        for index, case_obj in enumerate(case_group):  # 分组里的每个案例都要看。
            case_ref = _extract_case_ref(case_obj, location=f"selected_cases.{group_name}[{index}]")  # 提取稳定引用。
            if case_ref in seen_case_refs:  # 如果引用已经出现过，就跳过。
                continue  # 保持去重。
            seen_case_refs.add(case_ref)  # 记录这个案例已经收过。
            case_refs.append(case_ref)  # 按首次出现顺序保留引用。
    return case_refs  # 返回去重后的引用列表。


def build_summary(metric_table: Mapping[str, Any] | Sequence[Mapping[str, Any]], statistics_table: Mapping[str, Any] | Sequence[Mapping[str, Any]], selected_cases: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:  # 对外入口：把三类输入装成 summary。
    """从冻结的指标表、统计表和案例选择结果构建 summary 载荷。

    作用：把 metric_table、statistics_table 和 selected_cases 三类输入
    统一包装成可落盘、可校验、可供下游直接消费的 summary 结构。

    参数:
        metric_table: 主指标表，可以是单个映射或映射序列。
        statistics_table: 统计检验结果表，可以是单个映射或映射序列。
        selected_cases: 案例选择结果映射，包含 main_cases/failure_cases/boundary_cases 分组。

    返回值:
        dict[str, Any]: 经过协议校验的 summary 字典，包含 case_refs 和 summary_stats。

    异常:
        TypeError: 输入类型不正确。
        ValueError: 输入为空或字段缺失。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "metric_table_type": type(metric_table).__name__,
            "statistics_table_type": type(statistics_table).__name__,
            "selected_cases_groups": list(selected_cases.keys()) if isinstance(selected_cases, Mapping) else None,
        },
        "build_summary 入口参数",
        prefix="[analysis]",
    )
    main_table_ref = _normalize_table_ref(metric_table, name="metric_table")  # 先把主表规范化成可序列化引用。
    _validate_main_table_ref(main_table_ref, name="metric_table")  # 主表消费者只接受聚合 main_table 视图，拒绝 long-form 指标长表。
    statistics_ref = _normalize_table_ref(statistics_table, name="statistics_table")  # 再把统计结果规范化。
    case_refs = _collect_case_refs(selected_cases)  # 最后从 selected_cases 里收集所有案例引用。

    summary = SummaryResult(  # 用协议定义的结果结构来包装 summary。
        case_refs=case_refs,  # 案例引用列表放到专门字段里。
        summary_stats={  # summary_stats 保存主表和统计表的引用。
            "main_table_ref": main_table_ref,  # 主表引用。
            "statistics_ref": statistics_ref,  # 统计结果引用。
        },  # summary_stats 字典结束。
    ).to_dict()  # 先构造成协议对象，再转成普通字典。
    validate_summary_result(summary)  # 按协议再验证一遍，避免拼装出来的结构不合规。
    return summary  # 返回最终可落盘的 summary。
