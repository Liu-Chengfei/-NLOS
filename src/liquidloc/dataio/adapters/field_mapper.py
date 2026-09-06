"""把数据集特有的原始字段映射成内部统一字段形状。

这个模块的作用是让 reader 可以接收某个数据集自己的原始包，然后把
列名转换成后续管线约定的稳定键名。映射逻辑统一放在这里，原始时间戳
如果存在也会保留下来，同时返回一份简明报告，方便调用方知道哪些流
已经覆盖，哪些源字段缺失。

上游依赖:
- readers 输出的按流分组的原始包（字典，键名形如 xxx_raw）
- 外部传入的字段映射定义（字典，键名是流名，值是源→目标字段映射）

下游调用者:
- adapters.event_builder（需要统一字段名才能构造事件）
- 准备流程脚本（需要映射报告判断覆盖是否完整）

核心变量:
- raw_bundle: 按流分组的原始数据包
- field_mapping: 按流给出的字段映射定义
- internal_raw_bundle: 映射后的内部原始包
- mapping_report: 映射覆盖情况报告

关键设计决策:
- 缺失字段统一填充 None，保持返回结构一致
- 类型转换失败时保留原值，由下游消费者处理
- 映射报告记录缺失字段的出现次数，便于追踪数据质量
"""

from __future__ import annotations

import copy
from math import isfinite
from typing import Any, Callable, Optional

from liquidloc.common.validation import is_bool_like, is_integer, is_real, is_string_like

__all__ = ("map_external_fields",)


def _coerce_bool(raw_value: Any) -> bool:
    """把原始值安全转成布尔值，防御字符串输入。
    
    参数：
        raw_value: 待转换的原始值，可以是布尔、数值或字符串类型。
    
    返回：
        bool: 转换后的布尔值。
    
    转换规则：
        - 布尔类型（含 numpy.bool_）直接转换
        - 数值类型先检查有限性，非有限值返回 False
        - 字符串类型先尝试解析为数值，失败则检查假值单词列表
        - 其他类型使用 Python 内置 bool() 转换
    """
    if is_bool_like(raw_value):
        return bool(raw_value)
    if is_real(raw_value):
        if not isfinite(raw_value):
            return False
        return bool(raw_value)
    if is_string_like(raw_value):
        normalized = str(raw_value).strip().lower()
        if not normalized:
            return False
        try:
            float_val = float(normalized)
            if not isfinite(float_val):
                return False
            return bool(float_val)
        except (ValueError, TypeError):
            pass
        return normalized not in ('false', 'f', 'no', 'none', 'null')
    return bool(raw_value)


def _coerce_float(raw_value: Any) -> float:
    """把原始值安全转成浮点数。
    
    参数：
        raw_value: 待转换的原始值。
    
    返回：
        float: 转换后的浮点数。
    
    异常：
        ValueError: 无法转换为浮点数时抛出。
    """
    return float(raw_value)


def _coerce_int(raw_value: Any) -> int:
    """把原始值安全转成整数。
    
    参数：
        raw_value: 待转换的原始值。
    
    返回：
        int: 转换后的整数。
    
    异常：
        ValueError: 无法转换为整数时抛出。
    """
    if is_integer(raw_value):
        return int(raw_value)
    if is_string_like(raw_value):
        normalized = str(raw_value).strip()
        if normalized and normalized.lstrip("+-").isdigit():
            return int(normalized)
        raise ValueError(f"non-integer string value: {raw_value!r}")
    if is_real(raw_value):
        numeric_value = float(raw_value)
        if isfinite(numeric_value) and numeric_value.is_integer():
            return int(numeric_value)
        raise ValueError(f"non-integer numeric value: {raw_value!r}")
    raise ValueError(f"cannot coerce non-integer value {raw_value!r} to int")


# 已知语义字段的类型强制转换表：目标字段名 -> 转换函数。
# 在映射层统一做类型契约，消费者不需要各自做类型防御。
_SEMANTIC_TYPE_COERCIONS: dict[str, Callable[[Any], Any]] = {
    'valid': _coerce_bool,
    'quality': _coerce_float,
    'timestamp': _coerce_float,
    'source_t': _coerce_float,
    'range': _coerce_float,
    'ax': _coerce_float,
    'ay': _coerce_float,
    'az': _coerce_float,
    'gx': _coerce_float,
    'gy': _coerce_float,
    'gz': _coerce_float,
    'dx': _coerce_float,
    'dy': _coerce_float,
    'dyaw': _coerce_float,
    'px': _coerce_float,
    'py': _coerce_float,
    'yaw': _coerce_float,
    # 铁律 3 (Stage A1 下游修复, 2026-07-23): 删 reproj_err / tracked_features.
    # VIO 紧耦合不再输出此两字段, event_builder 验证层已同步移除.
    'anchor_id': _coerce_int,
}


def _map_rows(rows: list[dict[str, Any]], mapping: dict[str, str], *, stream_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把一个流的原始行列表映射成统一字段名。

    参数：
        rows：某个流对应的原始记录列表，调用前应当已经按流分组。
        mapping：这个流的源字段到目标字段映射表。
        stream_name：流的逻辑名称，用于错误消息和报告。

    返回：
        一个二元组，前半部分是映射后的行，后半部分是覆盖情况报告。

    异常：
        TypeError：当 rows 不是列表，或列表元素不是字典时抛出。
        ValueError：当映射为空或不是字典时抛出。
    """
    if not isinstance(rows, list):  # 先拒绝非列表输入，这样调用方能立刻知道形状不对。
        raise TypeError(f'{stream_name} rows must be a list')
    if not isinstance(mapping, dict) or not mapping:  # 必须给出非空映射，否则适配器会静默丢掉所有字段。
        raise ValueError(f'{stream_name} mapping must be a non-empty mapping')

    mapped_rows: list[dict[str, Any]] = []  # 按输入顺序收集归一化后的行。
    missing_fields: dict[str, int] = {}  # 统计每个源字段在整个流里缺失了多少次。
    for row in rows:  # 逐行处理，避免把整批数据当成一个粗糙的模式猜测。
        if not isinstance(row, dict):  # 保持严格，后续构造器都假定记录是字典。
            raise TypeError(f'{stream_name} rows must contain dict records')
        mapped_row: dict[str, Any] = {}  # 先构造一行映射结果，方便看清楚转换过程。
        for source_key, target_key in mapping.items():  # 只复制显式声明的列，保证内部模式稳定。
            if source_key in row:  # 原始行里有这个字段就保留它的值。
                value = row[source_key]  # 取出原始值。
                coercer = _SEMANTIC_TYPE_COERCIONS.get(target_key)  # 查找目标字段的类型转换函数。
                if coercer is not None:  # 有类型契约就强制转换，消费者不需要各自做类型防御。
                    try:
                        value = coercer(value)  # 应用类型转换。
                    except (TypeError, ValueError):  # 转换失败时保留原值，由下游消费者处理。
                        pass
                mapped_row[target_key] = value  # 把值写到内部约定的字段名下。
            else:
                missing_fields[source_key] = missing_fields.get(source_key, 0) + 1
                mapped_row[target_key] = None
        mapped_rows.append(mapped_row)  # 当前行的映射完成后再加入结果列表。

    for source_row, mapped_row in zip(rows, mapped_rows):  # 把原始行和映射行一起回看，补回可保留的元数据。
        if isinstance(source_row, dict) and 'source_t' in source_row:  # 如果原始行显式带了 source_t，就继续传递下去。
            if 'source_t' not in mapped_row:  # 只在映射阶段未产出 source_t 时才补回，避免覆盖已转换的值。
                mapped_row['source_t'] = _coerce_float(source_row['source_t'])  # 补回时也走类型转换，保证契约一致。

    return mapped_rows, {  # 同时返回映射后的行和一份简明诊断报告。
        'row_count': len(rows),  # 记录处理了多少行。
        'source_fields': sorted(mapping.keys()),  # 记录期望看到的外部字段。
        'target_fields': sorted(mapping.values()),  # 记录实际产出的内部字段。
        'missing_source_fields': missing_fields,  # 把字段级缺口暴露出来，便于检查覆盖是否完整。
        'is_complete': not missing_fields,  # 只有没有缺失字段时才算完整。
        'preserved_source_t': any(  # 看看是否至少有一行保留了原始来源时间戳。
            isinstance(source_row, dict) and 'source_t' in source_row  # 只有字典行才可能带这个可选元数据。
            for source_row in rows  # 这里扫描原始行，而不是映射后的行。
        ),
    }


def map_external_fields(
    raw_bundle: dict[str, list[dict[str, Any]] | dict[str, Any]],
    field_mapping: dict[str, dict[str, str]],
) -> tuple[dict[str, list[dict[str, Any]] | dict[str, Any]], dict[str, Any]]:
    """把完整原始包映射成内部按流键名组织的原始包。

    参数：
        raw_bundle：外部原始包，键名是各个流对应的原始名称。
        field_mapping：按流给出的字段映射定义。

    返回：
        一个二元组，前半部分是归一化后的原始包，后半部分是映射报告。

    异常：
        ValueError：当任一输入为空，或者找不到任何重叠流时抛出。

    设计说明：
        - 缺失字段统一填充 None，保持返回结构一致
        - 缺失字段计数记录在 mapping_report['streams'][stream_name]['missing_source_fields'] 中
        - 整条流缺失时，missing_source_fields 中每个字段计数为 1，同时设置 missing_stream=True
        - 单行字段缺失时，missing_source_fields 中对应字段计数递增，missing_stream=False
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "raw_bundle_keys": list(raw_bundle.keys()) if isinstance(raw_bundle, dict) else None,
        "field_mapping_streams": list(field_mapping.keys()) if isinstance(field_mapping, dict) else None,
    }, "map_external_fields 入口参数")
    if not isinstance(raw_bundle, dict) or not raw_bundle:
        raise ValueError('raw_bundle must be a non-empty mapping')
    if not isinstance(field_mapping, dict) or not field_mapping:
        raise ValueError('field_mapping must be a non-empty mapping')

    internal_raw_bundle: dict[str, list[dict[str, Any]] | dict[str, Any]] = {}
    mapping_report: dict[str, Any] = {'streams': {}, 'is_complete': True}

    for stream_name, mapping in field_mapping.items():
        raw_key = f'{stream_name}_raw'
        if raw_key not in raw_bundle:
            mapping_report['streams'][stream_name] = {
                'row_count': 0,
                'source_fields': sorted(mapping.keys()),
                'target_fields': sorted(mapping.values()),
                'missing_source_fields': {source_key: 1 for source_key in mapping},
                'is_complete': False,
                'preserved_source_t': False,
                'missing_stream': True,
            }
            mapping_report['is_complete'] = False
            continue
        raw_value = raw_bundle[raw_key]
        if isinstance(raw_value, dict):
            internal_raw_bundle[raw_key] = copy.deepcopy(raw_value)
            mapping_report['streams'][stream_name] = {
                'row_count': None,
                'source_fields': sorted(raw_value.keys()),
                'target_fields': sorted(raw_value.keys()),
                'missing_source_fields': {},
                'is_complete': True,
                'preserved_source_t': False,
                'missing_stream': False,
            }
            continue
        mapped_rows, stream_report = _map_rows(raw_value, mapping, stream_name=stream_name)
        stream_report['missing_stream'] = False
        internal_raw_bundle[raw_key] = mapped_rows
        mapping_report['streams'][stream_name] = stream_report
        mapping_report['is_complete'] = mapping_report['is_complete'] and stream_report['is_complete']

    if not internal_raw_bundle:
        raise ValueError('No overlapping streams found between raw bundle and field mapping')
    return internal_raw_bundle, mapping_report
