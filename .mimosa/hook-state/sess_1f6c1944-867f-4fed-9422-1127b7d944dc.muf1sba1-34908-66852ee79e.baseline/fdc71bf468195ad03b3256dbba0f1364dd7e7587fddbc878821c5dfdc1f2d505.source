"""
按冻结规则从 metric_table 里选择主案例、失败案例和边界案例。

这个模块把案例引用、分组规则和去重逻辑集中起来，
保证 plotting 和 summary 拿到的是同一套可复现的案例集合。
它只负责选案例，不负责算指标，也不负责决定图怎么画。
"""

from __future__ import annotations  # 让类型注解里的前向引用更稳，不影响运行逻辑。

from collections.abc import Iterable, Mapping, Sequence  # 用这些抽象类型判断输入容器的形状。
from typing import Any  # 用 Any 接住不确定的案例对象，便于做运行时校验。

from liquidloc.common.constants import (  # D9 单源：案例分组名与引用字段名常量。
    CASE_GROUP_BOUNDARY,
    CASE_GROUP_FAILURE,
    CASE_GROUP_MAIN,
    CASE_REF_KEY,
    CASE_REF_SEPARATOR,
    SCENE_ID_KEY,
    SEQ_ID_KEY,
)
from liquidloc.common.validation import is_string_like  # 复用统一字符串类型校验。


def _extract_case_ref(case_obj: Any, *, location: str) -> str:  # 从单个案例对象里抽出稳定引用名。
    """从案例对象里提取可稳定引用的 case_ref。

    权威定义：analysis/summary_builder.py 和 plotting/plot_cases.py 均从此处导入，
    禁止在别处重复定义相同逻辑。
    """
    if not isinstance(case_obj, Mapping):  # 案例必须像字典一样能取键。
        raise TypeError(f"{location} must be a mapping, got {type(case_obj).__name__}")  # 不是映射就直接报错。

    if CASE_REF_KEY in case_obj:  # 如果显式给了 case_ref，就优先用它。
        case_ref = case_obj[CASE_REF_KEY]  # 读取 case_ref 原值。
        if not is_string_like(case_ref):  # case_ref 必须是字符串，不能是数字或别的对象。
            raise TypeError(f"{location}.{CASE_REF_KEY} must be a string, got {type(case_ref).__name__}")  # 类型不对就报错。
        normalized_case_ref = str(case_ref).strip()  # 去掉首尾空白，避免同一个案例被写成不同空格版本。
        if not normalized_case_ref:  # 去空白后还为空，说明这个引用没有内容。
            raise ValueError(f"{location}.{CASE_REF_KEY} must be a non-empty string")  # 空字符串不允许。
        return normalized_case_ref  # 返回清洗后的 case_ref。

    scene_id = case_obj.get(SCENE_ID_KEY)  # 没有显式 case_ref 时，优先把场景轴拼进引用里避免跨 scene 冲突。
    seq_id = case_obj.get(SEQ_ID_KEY)  # seq_id 仍然是兜底引用的核心部分。
    if is_string_like(scene_id) and str(scene_id).strip() and is_string_like(seq_id) and str(seq_id).strip():
        return f"{str(scene_id).strip()}{CASE_REF_SEPARATOR}{str(seq_id).strip()}"  # 场景+序列联合引用，权威定义供 summary_builder / plot_cases 复用。
    if not is_string_like(seq_id):  # 退化到裸 seq_id 时，类型仍必须正确。
        raise TypeError(f"{location}.{SEQ_ID_KEY} must be a string, got {type(seq_id).__name__}")  # 类型不对就报错。
    normalized_case_ref = str(seq_id).strip()  # 最后再退回裸 seq_id。
    if not normalized_case_ref:  # 兜底引用也不能为空。
        raise ValueError(f"{location}.{SEQ_ID_KEY} must be a non-empty string")  # 空 seq_id 不允许。
    return normalized_case_ref  # 返回最终稳定引用。


def _normalize_case_record(case_obj: Any, *, location: str) -> dict[str, Any]:  # 把单个案例标准化成固定字典结构。
    """把单个案例记录规范化成包含 case_ref 的字典。"""
    if not isinstance(case_obj, Mapping):  # 仍然先确认它是映射。
        raise TypeError(f"{location} must be a mapping, got {type(case_obj).__name__}")  # 类型不对就拒绝。

    normalized_case = dict(case_obj)  # 复制一份，避免修改外部原对象。
    normalized_case_ref = _extract_case_ref(case_obj, location=location)  # 从原对象里提取统一引用。
    normalized_case["case_ref"] = normalized_case_ref  # 把统一引用写回标准字段。
    return normalized_case  # 返回规范化后的案例记录。


def _normalize_metric_table(metric_table: Any) -> dict[str, Any]:  # 把 metric_table 统一成 case_ref -> case_record 的字典。
    """把 metric_table 规范化成 case_ref -> case_record 的字典。"""
    if isinstance(metric_table, Mapping):  # 如果本来就是字典，就按键值对处理。
        normalized_table = {}  # 这里收集规范化后的表。
        for case_ref, case_obj in metric_table.items():  # 逐个案例处理，避免漏掉任何一行。
            if not is_string_like(case_ref):  # 字典 key 必须是字符串，才能稳定做引用。
                raise TypeError(  # 类型不对就直接报错。
                    f"metric_table keys must be strings, got {type(case_ref).__name__}"  # 报错里写明真实类型。
                )  # raise 参数结束。
            normalized_case_ref = str(case_ref).strip()  # 去掉 key 两边的空白。
            if not normalized_case_ref:  # 清洗后为空说明 key 不合法。
                raise ValueError("metric_table keys must be non-empty strings")  # 空 key 不允许。
            if normalized_case_ref in normalized_table:  # 如果清洗后重复，说明原始输入有冲突。
                raise ValueError(  # 重复引用会让后续选择时无法区分。
                    f"metric_table contains duplicate case_ref after normalization: {normalized_case_ref}"  # 报错中说明冲突来源。
                )  # raise 结束。
            normalized_case = _normalize_case_record(  # 把单个案例再标准化一次。
                case_obj,  # 这里传入原始案例对象。
                location=f"metric_table['{case_ref}']",  # 报错定位信息要能指出原始 key。
            )  # 单个案例规范化结束。
            # Mapping key 是权威 case_ref：覆盖 record 内部提取的 case_ref，
            # 保证 available_cases 的 key 与 record.case_ref 一致，
            # 避免 _select_case_group 按 key 查找、下游消费者按 record.case_ref 显示时出现错位。
            normalized_case["case_ref"] = normalized_case_ref
            normalized_table[normalized_case_ref] = normalized_case  # 存入规范化后的案例。
        if not normalized_table:  # 经过处理后如果一条都没有，说明表本身空。
            raise ValueError("metric_table must be non-empty")  # 空表不能继续。
        return normalized_table  # 返回规范化后的字典。

    if isinstance(metric_table, Sequence) and not isinstance(metric_table, (str, bytes)):  # 也允许传顺序表，但不能是字符串。
        case_rows = list(metric_table)  # 先转成 list，方便后续多次遍历。
        if not case_rows:  # 空列表也不行。
            raise ValueError("metric_table must be non-empty")  # 空表直接拒绝。

        normalized_table = {}  # 继续用字典收集标准化结果。
        for index, case_obj in enumerate(case_rows):  # 顺序表里每一项都要处理。
            if not isinstance(case_obj, Mapping):  # 每一项都必须是映射。
                raise TypeError(  # 不是映射就报错。
                    f"metric_table[{index}] must be a mapping, got {type(case_obj).__name__}"  # 明确指出是哪一项错了。
                )  # raise 参数结束。
            case_ref = _extract_case_ref(case_obj, location=f"metric_table[{index}]")  # 提取这一项的引用名。
            if case_ref in normalized_table:  # 如果重复了，就说明案例引用冲突。
                raise ValueError(f"metric_table contains duplicate case_ref: {case_ref}")  # 重复引用不能容忍。
            normalized_table[case_ref] = _normalize_case_record(  # 把这一项也标准化。
                case_obj,  # 当前案例对象。
                location=f"metric_table[{index}]",  # 用索引定位错误来源。
            )  # 单项标准化结束。
        return normalized_table  # 返回顺序表转成的标准字典。

    raise TypeError("metric_table must be a mapping or a non-string sequence of mappings")  # 既不是字典也不是列表，就拒绝。


def _normalize_case_group(case_rules: Mapping[str, Any], *, group_name: str) -> list[str]:  # 把一组案例引用规则变成干净列表。
    """把一个案例分组规则规范化成去重后的 case_ref 列表。"""
    case_group = case_rules.get(group_name)  # 从规则字典里取出对应分组。
    if case_group is None:  # 如果压根没给这个分组，就说明规则不完整。
        raise ValueError(f"case_rules is missing '{group_name}'")  # 缺少必需分组键属于值契约违规，按项目硬约束用 ValueError 而非 KeyError。
    if isinstance(case_group, (str, bytes, Mapping)) or not isinstance(case_group, Iterable):  # 分组必须是可迭代引用列表，不是单个字符串或字典。
        raise TypeError(f"case_rules.{group_name} must be an iterable of strings")  # 类型不对就报错。

    normalized_refs = []  # 这里按顺序收集去重后的引用。
    seen_refs = set()  # 用集合记住已经见过的引用。
    for index, case_ref in enumerate(case_group):  # 每个引用都要检查。
        if not is_string_like(case_ref):  # 分组里的引用必须是字符串。
            raise TypeError(  # 不是字符串就拒绝。
                f"case_rules.{group_name}[{index}] must be a string, got {type(case_ref).__name__}"  # 说明具体是哪一项错了。
            )  # raise 参数结束。
        normalized_ref = str(case_ref).strip()  # 去掉空白，统一引用格式。
        if not normalized_ref:  # 空字符串没有意义。
            raise ValueError(f"case_rules.{group_name}[{index}] must be a non-empty string")  # 直接拒绝。
        if normalized_ref in seen_refs:  # 重复项不再重复加入。
            continue  # 跳过重复引用，保持输出去重。
        seen_refs.add(normalized_ref)  # 记录这个引用已经出现过。
        normalized_refs.append(normalized_ref)  # 按原顺序保留第一次出现的引用。
    return normalized_refs  # 返回干净的引用列表。


def _select_case_group(  # 这个辅助函数只负责“按引用名抽出案例对象”。
    available_cases: Mapping[str, Any],  # 可用案例字典，key 是 case_ref。
    case_refs: Iterable[str],  # 需要选择的引用名集合。
    *,  # 下面的参数必须用关键字传入，避免调用时看错顺序。
    group_name: str,  # 组名只用于报错定位。
) -> list[Any]:  # 输出是一个案例对象列表。
    """按 case_ref 从可用案例里抽取一个案例组。"""
    selected_group = []  # 用这个列表按顺序装入选择结果。
    for case_ref in case_refs:  # 按引用顺序逐个抽取。
        if case_ref not in available_cases:  # 如果引用不存在，说明规则引用了未知案例。
            raise KeyError(f"case_rules.{group_name} references unknown case: {case_ref}")  # 引用未知案例属于字典查找失败（available_cases 中无此 key），与 Python dict[k] 语义一致，用 KeyError。
        selected_group.append(dict(available_cases[case_ref]))  # 复制一份案例记录，避免改到原表。
    return selected_group  # 返回这个分组的案例列表。


def select_cases(metric_table: Any, case_rules: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:  # 对外入口：根据规则挑出三类案例。
    """按冻结规则从 metric_table 中选择主案例、失败案例和边界案例。

    作用：根据 case_rules 中定义的三组案例引用（main_cases、failure_cases、
    boundary_cases），从 metric_table 中抽取对应的案例记录，返回按组分类
    的标准结构。案例引用必须与 metric_table 中的行匹配。

    参数:
        metric_table: 指标表，支持字典（case_ref -> 行）或映射序列。
        case_rules: 案例规则映射，必须包含 main_cases、failure_cases、boundary_cases 三个键。

    返回值:
        dict[str, list[dict]]: 包含三个分组的案例字典，键为组名，值为案例记录列表。

    异常:
        TypeError: 输入类型不正确。
        ValueError: 缺少必需的分组键、引用名为空或重复。
        KeyError: 引用了未知案例。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "metric_table_type": type(metric_table).__name__,
            "case_rules_groups": list(case_rules.keys()) if isinstance(case_rules, Mapping) else None,
        },
        "select_cases 入口参数",
        prefix="[analysis]",
    )
    available_cases = _normalize_metric_table(metric_table)  # 先把输入表统一成 case_ref 字典。
    if not isinstance(case_rules, Mapping):  # case_rules 必须是字典，才能按组名取规则。
        raise TypeError(f"case_rules must be a mapping, got {type(case_rules).__name__}")  # 类型不对就直接报错。

    main_case_refs = _normalize_case_group(case_rules, group_name=CASE_GROUP_MAIN)  # 规范化主案例引用。
    failure_case_refs = _normalize_case_group(case_rules, group_name=CASE_GROUP_FAILURE)  # 规范化失败案例引用。
    boundary_case_refs = _normalize_case_group(case_rules, group_name=CASE_GROUP_BOUNDARY)  # 规范化边界案例引用。

    main_cases = _select_case_group(  # 把主案例从可用表里抽出来。
        available_cases,  # 传入可用案例字典。
        main_case_refs,  # 传入主案例引用列表。
        group_name=CASE_GROUP_MAIN,  # 用于报错定位。
    )  # 主案例选择结束。
    failure_cases = _select_case_group(  # 把失败案例抽出来。
        available_cases,  # 同一个可用表。
        failure_case_refs,  # 失败案例引用列表。
        group_name=CASE_GROUP_FAILURE,  # 错误定位组名。
    )  # 失败案例选择结束。
    boundary_cases = _select_case_group(  # 把边界案例抽出来。
        available_cases,  # 同一个可用表。
        boundary_case_refs,  # 边界案例引用列表。
        group_name=CASE_GROUP_BOUNDARY,  # 错误定位组名。
    )  # 边界案例选择结束。

    selected_cases = {  # 最后把三类案例统一打包。
        CASE_GROUP_MAIN: main_cases,  # 主案例放这一组。
        CASE_GROUP_FAILURE: failure_cases,  # 失败案例放这一组。
        CASE_GROUP_BOUNDARY: boundary_cases,  # 边界案例放这一组。
    }  # selected_cases 字典结束。
    return selected_cases  # 返回对外使用的标准结构。
