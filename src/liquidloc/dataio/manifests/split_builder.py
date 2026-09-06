"""构建确定性数据集切分，并报告泄漏或重叠风险。

这个模块通常在数据集清单构建完成之后使用。它要么消费显式给出的切
分编号，要么按简单的确定性规则推导出切分结果，然后检查结果里是否
存在重复序列编号、共享作用域值或共享布局族。这里不追求花哨算法，
目标是稳定地产出切分清单和一份容易读懂的泄漏报告。

上游依赖:
- manifests.build_manifests 输出的数据集清单（包含序列记录列表）
- 序列目录下的 anchor_layout.json（用于解析布局族编号）

下游调用者:
- 准备流程脚本（需要切分清单来划分 train/val/test）
- 公开验证流程（需要泄漏报告来确认切分公平性）

核心变量:
- _SPLIT_SCOPE_FIELDS: 用于检查跨切分复用的作用域字段名列表
- split_manifest: 包含 train_ids / val_ids / test_ids 的切分结果
- leak_report: 包含 leak_items 和 is_clean 的泄漏报告
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Optional

from liquidloc.dataio.manifests.dataset_checks import resolve_layout_family_from_dir

_SPLIT_SCOPE_FIELDS = ('scene_id',)
_SEQ_DIR_FIELD = 'seq_dir'  # 集中定义 seq_dir 字段名，避免字面量散落
_MIN_FAMILY_COUNT_FOR_GROUPING = 3  # 族分组切分所需的最小族数
_RESERVED_FAMILY_COUNT_FOR_VAL = 2  # 分配 train 时为 val/test 预留的最小族数
_RESERVED_FAMILY_COUNT_FOR_TEST = 1  # 分配 val 时为 test 预留的最小族数


def _resolve_sequence_layout_family(record: dict[str, Any]) -> Optional[str]:
    """尽可能解析单条清单记录对应的布局族编号。

    参数：
        record: 清单记录字典，至少需要包含 seq_dir 字段。

    返回：
        布局族编号字符串；无法解析时返回 None。

    注意：
        非 dict 输入、seq_dir 缺失/为 None/为空字符串/类型无效时
        静默返回 None 并通过 warnings.warn 记录原因，便于诊断。
    """
    # 修复组 2: 类型校验替代真值检查
    if not isinstance(record, dict):
        warnings.warn('_resolve_sequence_layout_family: record 非 dict，跳过', stacklevel=2)
        return None

    # 修复组 1: 使用集中化字段名常量
    seq_dir = record.get(_SEQ_DIR_FIELD)
    # 修复组 2: 区分 None（缺失）和空字符串，拒绝非 str/Path 类型
    if seq_dir is None:
        warnings.warn('_resolve_sequence_layout_family: seq_dir 缺失，跳过', stacklevel=2)
        return None
    if seq_dir == '':
        warnings.warn('_resolve_sequence_layout_family: seq_dir 为空字符串，跳过', stacklevel=2)
        return None
    if not isinstance(seq_dir, (str, Path)):
        warnings.warn(f'_resolve_sequence_layout_family: seq_dir 类型无效 {type(seq_dir).__name__}，跳过', stacklevel=2)
        return None

    return resolve_layout_family_from_dir(seq_dir)


def _collect_record_scope_overlap_items(
    sequences: list[dict[str, Any]],
    split_manifest: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """当同一个记录作用域出现在多个切分里时，收集重叠项。

    参数：
        sequences: 数据集清单中的序列记录列表。
        split_manifest: 包含 train_ids / val_ids / test_ids 的切分结果。

    返回：
        重叠项列表，每项是一个结构化报告字典，包含 kind、scope_field、
        scope_value、seq_ids、splits 和 reason 字段。
    """
    record_by_seq_id: dict[str, dict[str, Any]] = {}
    for record in sequences:
        if isinstance(record, dict) and 'seq_id' in record:
            record_by_seq_id[str(record['seq_id'])] = record

    scope_to_split_entries: dict[str, dict[str, list[tuple[str, str, Any]]]] = {}
    # 修复组 4: split_manifest 键访问防御，使用 .get() 带默认值
    for split_name, split_ids in (
        ('train', split_manifest.get('train_ids', [])),
        ('val', split_manifest.get('val_ids', [])),
        ('test', split_manifest.get('test_ids', [])),
    ):
        # 修复组 5: split_ids 类型守卫，防止字符串逐字符迭代
        if not isinstance(split_ids, (list, tuple)):
            continue
        for seq_id in split_ids:
            record = record_by_seq_id.get(str(seq_id))
            if not isinstance(record, dict):
                # 修复组 8: 未知 seq_id 添加诊断
                warnings.warn(f'_collect_record_scope_overlap_items: seq_id {seq_id} 不在 sequences 中，跳过', stacklevel=2)
                continue

            for scope_field in _SPLIT_SCOPE_FIELDS:
                scope_value = record.get(scope_field)
                if scope_value is None:
                    continue
                # 修复组 3: 拒绝 bool 类型（str(True)="True" 不应作为 scope_value）
                if isinstance(scope_value, bool):
                    continue

                scope_text = str(scope_value).strip()
                if not scope_text:
                    continue

                # 修复组 7: 存储原始 scope_value 以便输出时保留原始值
                scope_to_split_entries.setdefault(scope_field, {}).setdefault(scope_text, []).append(
                    (split_name, str(seq_id), scope_value)
                )

    overlap_items: list[dict[str, Any]] = []
    # 修复组 14: 按 scope_field 和 scope_value 排序，确保输出顺序确定性
    for scope_field, value_map in sorted(scope_to_split_entries.items()):
        for scope_value, entries in sorted(value_map.items()):
            split_names = sorted({split_name for split_name, _, _ in entries})
            if len(split_names) < 2:
                continue

            # 修复组 7: 使用原始 scope_value 而非归一化后的 scope_text
            original_scope_value = entries[0][2] if entries else scope_value
            overlap_items.append({
                'kind': 'shared_record_scope',
                'scope_field': scope_field,
                'scope_value': original_scope_value,
                # 修复组 6: seq_ids 去重
                'seq_ids': list(dict.fromkeys(seq_id for _, seq_id, _ in sorted(entries))),
                'splits': split_names,
                # 修复组 34: reason 使用 ", ".join 格式
                'reason': f'{scope_field} {scope_value} spans multiple splits: {", ".join(split_names)}',
            })

    return overlap_items


def _collect_semantic_overlap_items(
    sequences: list[dict[str, Any]],
    split_manifest: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """当同一个布局族出现在多个切分里时，收集重叠项。

    参数：
        sequences: 数据集清单中的序列记录列表。
        split_manifest: 包含 train_ids / val_ids / test_ids 的切分结果。

    返回：
        语义重叠项列表，每项是一个结构化报告字典，包含 kind、layout_family、
        seq_ids、splits 和 reason 字段。
    """
    record_by_seq_id: dict[str, dict[str, Any]] = {}
    for record in sequences:
        if isinstance(record, dict) and 'seq_id' in record:
            record_by_seq_id[str(record['seq_id'])] = record

    family_to_split_entries: dict[str, list[tuple[str, str]]] = {}
    # 修复组 9: split_manifest 键访问防御，使用 .get() 带默认值
    for split_name, split_ids in (
        ('train', split_manifest.get('train_ids', [])),
        ('val', split_manifest.get('val_ids', [])),
        ('test', split_manifest.get('test_ids', [])),
    ):
        for seq_id in split_ids:
            # 修复组 13: 未知 seq_id 添加诊断
            record = record_by_seq_id.get(str(seq_id))
            if not isinstance(record, dict):
                warnings.warn(f'_collect_semantic_overlap_items: seq_id {seq_id} 不在 sequences 中，跳过', stacklevel=2)
                continue
            family_id = _resolve_sequence_layout_family(record)
            if family_id is None:
                continue

            family_to_split_entries.setdefault(family_id, []).append((split_name, str(seq_id)))

    overlap_items: list[dict[str, Any]] = []
    # 修复组 11: 按 family_id 排序，确保输出顺序确定性
    for family_id, entries in sorted(family_to_split_entries.items()):
        split_names = sorted({split_name for split_name, _ in entries})
        if len(split_names) < 2:
            continue

        overlap_items.append({
            'kind': 'shared_layout_family',
            'layout_family': family_id,
            # 修复组 10: seq_ids 去重
            'seq_ids': list(dict.fromkeys(seq_id for _, seq_id in sorted(entries))),
            'splits': split_names,
            # 修复组 12: reason 使用 ", ".join 格式
            'reason': f'layout family {family_id} spans multiple splits: {", ".join(split_names)}',
        })

    return overlap_items


def _resolve_default_family_grouped_split(
    sequences: list[dict[str, Any]],
    split_rules: dict[str, Any],
) -> Optional[dict[str, list[str]]]:
    """在默认切分路径下，优先按 layout family 成组切分。

    参数：
        sequences: 数据集清单中的序列记录列表。
        split_rules: 切分策略字典，可能包含 train_count 和 val_count。

    返回：
        切分结果字典，包含 train_ids / val_ids / test_ids；
        返回 None 的条件：
        1. 任何序列缺 family 元数据（无法按族分组）
        2. 族数 < 3（不足以分配 train/val/test 三组）
    """
    # 修复组 16: split_rules 校验
    if not isinstance(split_rules, dict):
        raise TypeError('split_rules must be a dict')

    # 修复组 15: int() 类型校验提前到 family 解析之前，确保输入校验先于业务逻辑
    # 注意：默认值依赖 seq_ids，需在 seq_ids 定义后应用，此处只做类型校验
    raw_train_count = split_rules.get('train_count')
    if raw_train_count is not None:
        if isinstance(raw_train_count, bool) or not isinstance(raw_train_count, int):
            raise TypeError(f'train_count must be an int, got {type(raw_train_count).__name__}')
        if raw_train_count < 0:
            raise ValueError(f'train_count must be non-negative, got {raw_train_count}')

    raw_val_count = split_rules.get('val_count')
    if raw_val_count is not None:
        if isinstance(raw_val_count, bool) or not isinstance(raw_val_count, int):
            raise TypeError(f'val_count must be an int, got {type(raw_val_count).__name__}')
        if raw_val_count < 0:
            raise ValueError(f'val_count must be non-negative, got {raw_val_count}')

    # 修复组 17: seq_ids 去重
    seq_ids = sorted({
        str(record['seq_id'])
        for record in sequences
        if isinstance(record, dict) and 'seq_id' in record
    })

    record_by_seq_id: dict[str, dict[str, Any]] = {
        str(record['seq_id']): record
        for record in sequences
        if isinstance(record, dict) and 'seq_id' in record
    }

    family_to_seq_ids: dict[str, list[str]] = {}
    ordered_family_ids: list[str] = []

    for seq_id in seq_ids:
        family_id = _resolve_sequence_layout_family(record_by_seq_id.get(seq_id))
        if family_id is None:
            return None

        if family_id not in family_to_seq_ids:
            family_to_seq_ids[family_id] = []
            ordered_family_ids.append(family_id)

        family_to_seq_ids[family_id].append(seq_id)

    # 修复组 20: 族数 < 3 常量
    if len(ordered_family_ids) < _MIN_FAMILY_COUNT_FOR_GROUPING:
        return None

    # 应用默认值（依赖 seq_ids 长度）
    requested_train_count = raw_train_count if raw_train_count is not None else max(1, len(seq_ids) - 2)
    requested_val_count = raw_val_count if raw_val_count is not None else 1

    train_target = max(1, min(requested_train_count, len(seq_ids) - 2))
    val_target = max(1, min(requested_val_count, len(seq_ids) - train_target - 1))

    train_family_ids: list[str] = []
    val_family_ids: list[str] = []
    test_family_ids: list[str] = []
    train_seq_count = 0
    val_seq_count = 0

    for family_index, family_id in enumerate(ordered_family_ids):
        remaining_families = len(ordered_family_ids) - family_index - 1
        family_seq_count = len(family_to_seq_ids[family_id])

        # 修复组 19: 使用常量替代硬编码
        if not train_family_ids or (train_seq_count < train_target and remaining_families >= _RESERVED_FAMILY_COUNT_FOR_VAL):
            train_family_ids.append(family_id)
            train_seq_count += family_seq_count
            continue

        # 修复组 19: 使用常量替代硬编码
        if not val_family_ids or (val_seq_count < val_target and remaining_families >= _RESERVED_FAMILY_COUNT_FOR_TEST):
            val_family_ids.append(family_id)
            val_seq_count += family_seq_count
            continue

        test_family_ids.append(family_id)

    # 修复组 21: 实际切分尺寸偏离目标时警告
    if train_seq_count != train_target or val_seq_count != val_target:
        warnings.warn(
            f'_resolve_default_family_grouped_split: 实际切分尺寸偏离目标 '
            f'(train: {train_seq_count}/{train_target}, val: {val_seq_count}/{val_target})',
            stacklevel=2,
        )

    return {
        'train_ids': [seq_id for family_id in train_family_ids for seq_id in family_to_seq_ids[family_id]],
        'val_ids': [seq_id for family_id in val_family_ids for seq_id in family_to_seq_ids[family_id]],
        'test_ids': [seq_id for family_id in test_family_ids for seq_id in family_to_seq_ids[family_id]],
    }


# Bug 3 修复 (2026-07-23 audit Round 1+): 按 lead_id 分组防止 var_01/var_02 间符号偏置泄漏.
# 原状: _resolve_default_family_grouped_split 按 anchor_layout family 分组, 但 sim_curve_01 /
#   sim_curve_01_var_01 / sim_curve_01_var_02 共享同一 base 轨迹, 仅噪声扰动不同 — anchor_layout
#   完全相同 (family 也同), 三者仍可被分散到 train/test, 测试集 RMSE 被低估.
# 修复: 从 seq_id 抽出 lead_id (去掉 _var_XX 尾缀), 同 lead_id 必须分到同一 split.
# 优先级: explicit_ids > lead_id_grouped > family_grouped > per_sequence.
import re as _re_split_bug3  # noqa: E402  避免在文末新增 import 块打乱原 import 顺序.

_VAR_SUFFIX_RE = _re_split_bug3.compile(r'_var_\d+$')  # 仅去除尾部 _var_XX, 保留 base lead_id.


def _resolve_lead_id_from_seq_id(seq_id: str) -> str:
    """从 seq_id 抽 lead_id, 去掉尾部 _var_XX.

    例: sim_curve_01_var_02 -> sim_curve_01; sim_long_10m_01 -> sim_long_10m_01.
    """
    return _VAR_SUFFIX_RE.sub('', str(seq_id))


def _resolve_default_lead_id_grouped_split(
    sequences: list[dict[str, Any]],
    split_rules: dict[str, Any],
) -> Optional[dict[str, list[str]]]:
    """在默认切分路径下, 按 lead_id (seq_id 去 _var_XX) 成组切分.

    同一 lead_id 的所有 variants (base + var_01 + var_02) 必须分到同一 split,
    防止"同 base 轨迹不同噪声扰动"分布在 train/test 引发的符号偏置泄漏.

    返回:
        切分结果字典; 返回 None 的条件:
        1. 全部 seq_id 都无 _var_XX 尾缀 (lead_id 等于 seq_id, 无分组意义, 走 family_grouped 兜底)
        2. lead_id 组数 < 3 (不足以分配 train/val/test 三组)
    """
    if not isinstance(split_rules, dict):
        raise TypeError('split_rules must be a dict')

    raw_train_count = split_rules.get('train_count')
    if raw_train_count is not None:
        if isinstance(raw_train_count, bool) or not isinstance(raw_train_count, int):
            raise TypeError(f'train_count must be an int, got {type(raw_train_count).__name__}')
        if raw_train_count < 0:
            raise ValueError(f'train_count must be non-negative, got {raw_train_count}')

    raw_val_count = split_rules.get('val_count')
    if raw_val_count is not None:
        if isinstance(raw_val_count, bool) or not isinstance(raw_val_count, int):
            raise TypeError(f'val_count must be an int, got {type(raw_val_count).__name__}')
        if raw_val_count < 0:
            raise ValueError(f'val_count must be non-negative, got {raw_val_count}')

    seq_ids = sorted({
        str(record['seq_id'])
        for record in sequences
        if isinstance(record, dict) and 'seq_id' in record
    })
    if not seq_ids:
        return None

    lead_to_seq_ids: dict[str, list[str]] = {}
    ordered_lead_ids: list[str] = []
    has_any_variant = False

    for seq_id in seq_ids:
        lead_id = _resolve_lead_id_from_seq_id(seq_id)
        if lead_id != seq_id:
            has_any_variant = True
        if lead_id not in lead_to_seq_ids:
            lead_to_seq_ids[lead_id] = []
            ordered_lead_ids.append(lead_id)
        lead_to_seq_ids[lead_id].append(seq_id)

    # 若无任何 _var_XX 后缀, lead_id 分组与 per_sequence 同义, 让出给 family_grouped 兜底.
    if not has_any_variant:
        return None

    if len(ordered_lead_ids) < _MIN_FAMILY_COUNT_FOR_GROUPING:
        return None

    requested_train_count = raw_train_count if raw_train_count is not None else max(1, len(ordered_lead_ids) - 2)
    requested_val_count = raw_val_count if raw_val_count is not None else 1

    train_target = max(1, min(requested_train_count, len(ordered_lead_ids) - 2))
    val_target = max(1, min(requested_val_count, len(ordered_lead_ids) - train_target - 1))

    train_lead_ids: list[str] = []
    val_lead_ids: list[str] = []
    test_lead_ids: list[str] = []
    train_seq_count = 0
    val_seq_count = 0

    for idx, lead_id in enumerate(ordered_lead_ids):
        remaining = len(ordered_lead_ids) - idx - 1
        group_size = len(lead_to_seq_ids[lead_id])

        if not train_lead_ids or (train_seq_count < train_target and remaining >= _RESERVED_FAMILY_COUNT_FOR_VAL):
            train_lead_ids.append(lead_id)
            train_seq_count += group_size
            continue

        if not val_lead_ids or (val_seq_count < val_target and remaining >= _RESERVED_FAMILY_COUNT_FOR_TEST):
            val_lead_ids.append(lead_id)
            val_seq_count += group_size
            continue

        test_lead_ids.append(lead_id)

    if train_seq_count != train_target or val_seq_count != val_target:
        warnings.warn(
            f'_resolve_default_lead_id_grouped_split: 实际切分尺寸偏离目标 '
            f'(train: {train_seq_count}/{train_target}, val: {val_seq_count}/{val_target})',
            stacklevel=2,
        )

    return {
        'train_ids': [seq_id for lead_id in train_lead_ids for seq_id in lead_to_seq_ids[lead_id]],
        'val_ids': [seq_id for lead_id in val_lead_ids for seq_id in lead_to_seq_ids[lead_id]],
        'test_ids': [seq_id for lead_id in test_lead_ids for seq_id in lead_to_seq_ids[lead_id]],
    }


def build_splits(
    dataset_manifest: dict[str, Any],
    split_rules: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """构建 train/val/test 切分编号，并输出泄漏报告。

    参数：
        dataset_manifest：包含序列列表的数据集清单。
        split_rules：切分策略，可以是显式编号，也可以是按数量默认切分。

    返回：
        一个二元组 (split_manifest, leak_report)：
        - split_manifest: 切分清单字典，包含 train_ids、val_ids 和 test_ids
        - leak_report: 泄漏报告字典，包含 leak_items、is_clean 和 split_path 字段

    异常：
        ValueError：当数据集没有序列，或者无法生成测试集时抛出。
        TypeError：当 split_rules 不是字典时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "split_rules": split_rules,
        "manifest_keys": list(dataset_manifest.keys()) if isinstance(dataset_manifest, dict) else None,
        "sequence_count": len(dataset_manifest.get("sequences", [])) if isinstance(dataset_manifest, dict) else None,
    }, "build_splits")
    # 修复组 22: dataset_manifest 校验
    if not isinstance(dataset_manifest, dict):
        raise TypeError('dataset_manifest must be a dict')
    sequences = dataset_manifest.get('sequences', [])
    if not isinstance(sequences, list):
        raise TypeError('dataset_manifest.sequences must be a list')
    if not sequences:
        raise ValueError('dataset_manifest.sequences must be non-empty')
    # 修复组 22: 校验所有序列记录非空且至少有一个有效 seq_id
    valid_seq_count = sum(1 for record in sequences if isinstance(record, dict) and 'seq_id' in record)
    if valid_seq_count == 0:
        raise ValueError('dataset_manifest.sequences contains no valid records with seq_id')

    if not isinstance(split_rules, dict):
        raise TypeError('split_rules must be a dict')

    explicit = split_rules.get('explicit_ids')
    # 修复组 28: 提前计算 seq_ids 列表，用于后续校验
    all_seq_ids = [
        str(record['seq_id'])
        for record in sequences
        if isinstance(record, dict) and 'seq_id' in record
    ]
    available_seq_ids = set(all_seq_ids)

    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]
    # 修复组 31: 提前初始化 family_grouped_split，便于路径元数据记录.
    # Bug 3: 同口径提前初始化 lead_id_grouped_split.
    family_grouped_split: Optional[dict[str, list[str]]] = None
    lead_id_grouped_split: Optional[dict[str, list[str]]] = None

    # 修复组 23: if explicit 改为 if explicit is not None
    if explicit is not None:
        # 修复组 24: explicit_ids 类型校验
        if not isinstance(explicit, dict):
            raise TypeError('explicit_ids must be a dict')
        # 修复组 24: 校验 train/val/test 是 list
        for key in ('train', 'val', 'test'):
            value = explicit.get(key, [])
            if not isinstance(value, list):
                raise TypeError(f'explicit_ids.{key} must be a list, got {type(value).__name__}')
        # 修复组 25: 校验 train 键存在（val/test 可选但 train 必须存在）
        if 'train' not in explicit:
            raise ValueError('explicit_ids must contain train key')

        train_ids = [str(seq_id) for seq_id in explicit.get('train', [])]
        val_ids = [str(seq_id) for seq_id in explicit.get('val', [])]
        test_ids = [str(seq_id) for seq_id in explicit.get('test', [])]

        referenced_seq_ids = [str(seq_id) for split_ids in (train_ids, val_ids, test_ids) for seq_id in split_ids]
        unknown_seq_ids = sorted({seq_id for seq_id in referenced_seq_ids if seq_id not in available_seq_ids})

        if unknown_seq_ids:
            raise ValueError(f'explicit split ids reference unknown sequences: {unknown_seq_ids}')

        uncovered_seq_ids = sorted(available_seq_ids - set(referenced_seq_ids))
        if uncovered_seq_ids:
            raise ValueError(f'explicit split ids must cover all manifest sequences, missing: {uncovered_seq_ids}')
    else:
        # Bug 3 (2026-07-23 audit Round 1+): lead_id 分组优先于 family 分组,
        # 防止"同 base 轨迹不同 _var_XX 噪声扰动"被分散到 train/test 引发符号偏置泄漏.
        # 优先级: explicit > lead_id_grouped > family_grouped > per_sequence.
        lead_id_grouped_split = _resolve_default_lead_id_grouped_split(sequences, split_rules)
        if lead_id_grouped_split is not None:
            train_ids = list(lead_id_grouped_split['train_ids'])
            val_ids = list(lead_id_grouped_split['val_ids'])
            test_ids = list(lead_id_grouped_split['test_ids'])
        else:
            family_grouped_split = _resolve_default_family_grouped_split(sequences, split_rules)

            if family_grouped_split is not None:
                train_ids = list(family_grouped_split['train_ids'])
                val_ids = list(family_grouped_split['val_ids'])
                test_ids = list(family_grouped_split['test_ids'])
            else:
                # 修复组 33: 逐序列路径排序统一
                seq_ids = sorted([
                    str(record['seq_id'])
                    for record in sequences
                    if isinstance(record, dict) and 'seq_id' in record
                ])

                if len(seq_ids) == 1:
                    train_ids = seq_ids[:]
                    val_ids = []
                    test_ids = []
                elif len(seq_ids) == 2:
                    train_ids = seq_ids[:1]
                    val_ids = seq_ids[1:]
                    test_ids = []
                else:
                    # 修复组 26: int() 类型校验
                    raw_train_count = split_rules.get('train_count', max(1, len(seq_ids) - 2))
                    if isinstance(raw_train_count, bool) or not isinstance(raw_train_count, int):
                        raise TypeError(f'train_count must be an int, got {type(raw_train_count).__name__}')
                    if raw_train_count < 0:
                        raise ValueError(f'train_count must be non-negative, got {raw_train_count}')
                    requested_train_count = raw_train_count
                    train_count = max(1, min(requested_train_count, len(seq_ids) - 2))
                    raw_val_count = split_rules.get('val_count', 1)
                    if isinstance(raw_val_count, bool) or not isinstance(raw_val_count, int):
                        raise TypeError(f'val_count must be an int, got {type(raw_val_count).__name__}')
                    if raw_val_count < 0:
                        raise ValueError(f'val_count must be non-negative, got {raw_val_count}')
                    requested_val_count = raw_val_count
                    max_val_count = len(seq_ids) - train_count - 1
                    val_count = max(1, min(requested_val_count, max_val_count))

                    # 修复组 27: 钳制时警告
                    if train_count != requested_train_count:
                        warnings.warn(f'build_splits: train_count 钳制 {requested_train_count} -> {train_count}', stacklevel=2)
                    if val_count != requested_val_count:
                        warnings.warn(f'build_splits: val_count 钳制 {requested_val_count} -> {val_count}', stacklevel=2)

                    train_ids = seq_ids[:train_count]
                    val_ids = seq_ids[train_count:train_count + val_count]
                    test_ids = seq_ids[train_count + val_count:]

    # 修复组 28: 用 seq_ids 列表长度（而非 available_seq_ids set 长度）检查
    if len(all_seq_ids) >= 3 and not test_ids:
        raise ValueError('split_rules must produce a non-empty test split')

    # 修复组 29/30: 重写重复检测，正确区分切分内和跨切分重复
    leak_items: list[dict[str, Any]] = []
    # 记录每个 seq_id 出现在哪些 split 中（保留重复以检测切分内重复）
    seq_id_splits: dict[str, list[str]] = {}
    for split_name, split_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        for seq_id in split_ids:
            seq_id_str = str(seq_id)
            seq_id_splits.setdefault(seq_id_str, []).append(split_name)

    for seq_id_str, splits in seq_id_splits.items():
        # 切分内重复：同一 split 出现多次
        split_counts: dict[str, int] = {}
        for split_name in splits:
            split_counts[split_name] = split_counts.get(split_name, 0) + 1

        for split_name, count in split_counts.items():
            if count > 1:
                leak_items.append({
                    'seq_id': seq_id_str,
                    'splits': [split_name],
                    'reason': f'duplicated within split {split_name}'
                })

        # 跨切分重复：出现在多个不同 split 中
        unique_splits = sorted(set(splits))
        if len(unique_splits) > 1:
            leak_items.append({
                'seq_id': seq_id_str,
                'splits': unique_splits,
                'reason': f'duplicated across splits: {", ".join(unique_splits)}'
            })

    split_manifest = {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}

    # §9 量级门检查 (前提指导 §9.2 / §9.3 / §0.2 B04-B05).
    # 在 split_manifest 构建完成后, 训练/测试规模比 ≤ 0.1 (少样本) 与 N_traj_te ≥ 20
    # 是 §9 强制量级门; 违反时按 §9 「放松则伤」记录 leak_items 但不阻断切分
    # (eval/scoring 接口读取 leak_report.is_clean=False 做降级处理), 主结论不构成全序.
    _emit_section9_leak_warnings(
        leak_items,
        n_traj_train=len(train_ids),
        n_traj_test=len(test_ids),
        n_traj_total=len(all_seq_ids),
        sequences=sequences,
        test_ids=test_ids,
        dataset_manifest=dataset_manifest,
        train_ids=train_ids,
    )

    record_scope_items = _collect_record_scope_overlap_items(sequences, split_manifest)

    # 修复组 32: 不再对 explicit 路径过滤 scene_id 重叠项，统一判定标准

    leak_items.extend(record_scope_items)
    leak_items.extend(_collect_semantic_overlap_items(sequences, split_manifest))

    # 修复组 35: 对 leak_items 去重，避免 scene_id 与 layout_family 双重计数
    seen_leak_keys: set[tuple] = set()
    deduplicated_leak_items: list[dict[str, Any]] = []
    for item in leak_items:
        # 以 (kind, tuple(seq_ids)) 为去重键
        key = (item.get('kind', ''), tuple(item.get('seq_ids', [])))
        if key not in seen_leak_keys:
            seen_leak_keys.add(key)
            deduplicated_leak_items.append(item)
    leak_items = deduplicated_leak_items

    # 修复组 31: 记录切分路径元数据.
    # Bug 3 (2026-07-23 audit Round 1+): 加 'lead_id_grouped' 路径标识, 优先级高于 family_grouped.
    if explicit is not None:
        split_path = 'explicit'
    elif lead_id_grouped_split is not None:
        split_path = 'lead_id_grouped'
    elif family_grouped_split is not None:
        split_path = 'family_grouped'
    else:
        split_path = 'per_sequence'
    leak_report = {
        'leak_items': leak_items,
        'is_clean': not leak_items,
        'split_path': split_path,
    }
    return split_manifest, leak_report


# =============================================================================
# §9 量级门检查（前提指导 §9.2 / §9.3 与 §0.2 B04-B05 操作定义槽）.
#
# 本函数在 build_splits 末尾被调用, 对当前切分结果做三类 §9 量级门审计并
# 把违规项记录进 leak_items; 不阻断切分 (返回 leak_report.is_clean=False
# 触发下游降级), 但任何违规都按 §9 「放松则伤」明确报错与建议升级.
#
# 不在加载时立即 raise 的设计动机: 切分本身无错, 是协议层判定该切分是否
# 能产生全序结论. 与 leak_items 的 scene_id / layout_family 同口径处理.
# =============================================================================
def _emit_section9_leak_warnings(
    leak_items: list[dict[str, Any]],
    *,
    n_traj_train: int,
    n_traj_test: int,
    n_traj_total: int,
    sequences: list[dict[str, Any]],
    test_ids: list[str],
    dataset_manifest: dict[str, Any] | None = None,
    train_ids: list[str] | None = None,
) -> None:
    """§9 量级门审计: train/test 比例 / N_traj_te / 测试布局族数 / 种子分列.

    把审计违规项追加到 leak_items (原地修改); 不返回值.
    五类量级门:
      0. train/test split 序列 ID 集合 disjointness (§9.2 字面守门 / §4.3.1.5),
         由协议层 normalize_train_test_disjointness 真实接入主路径执行.
      1. train/test 比例 ≤ train_test_ratio_max=0.1 (§9.2 / §0.2 B05).
      2. N_traj_te ≥ N_traj_te_min=20 (§9.3 / §0.2 B04).
      3. 测试覆盖 ≥ layout_family_min=3 种布局族 (§9.3 「禁单布局刷满 20 轨」).
      4. 种子分列: scene_seed_scope 各字段独立存在 ( §9 细节「轨迹随机种子与 NLOS/异步种子分列」).

    参数:
        leak_items: 已有 leak_items 列表, 本函数原地追加 §9 违规项.
        n_traj_train: train_ids 长度.
        n_traj_test: test_ids 长度.
        n_traj_total: 全部 seq_ids 长度.
        sequences: dataset_manifest.sequences, 用于解析测试段布局族.
        test_ids: 切分后的 test_ids 列表.
        dataset_manifest: 可选数据清单, 用于种子分列审计.
        train_ids: 切分后的 train_ids 列表, 用于 §9.2 disjointness 协议层守门.
    """
    # 延迟导入避免 split_builder 模块加载时形成循环依赖.
    try:
        from liquidloc.protocol.experiment_gates import (
            check_train_test_ratio,
            check_layout_family_count,
            normalize_train_test_disjointness,
        )
    except Exception as exc:  # pragma: no cover - 仅在协议层 import 失败时降级
        warnings.warn(
            f'_emit_section9_leak_warnings: 协议层 import 失败 ({exc!r}), §9 量级门审计跳过',
            stacklevel=2,
        )
        return

    # §9.2 字面守门 (§4.3.1.5): 训练 split 与测试 split 的序列 ID 集合必须不交.
    # 协议层 normalize_train_test_disjointness 是 raise-only, 这里捕获 ValueError
    # 转为 leak_items 遵循 §9 「放松则伤」范式; 允许 split_builder 自身 inline
    # 跨切分重复检测 (L596-618) 照常运行, 本守门为协议层显式断言的真入口.
    if train_ids is not None and test_ids is not None:
        try:
            normalize_train_test_disjointness(train_ids, test_ids)
        except ValueError as _exc:
            leak_items.append({
                'kind': 'section9_seq_id_disjointness',
                'reason': str(_exc),
            })
        except TypeError as _exc:
            warnings.warn(
                f'_emit_section9_leak_warnings: normalize_train_test_disjointness 类型异常 ({_exc!r})',
                stacklevel=2,
            )

    # §9.2 / §0.2 B05: 训练/测试规模比 ≤ 0.1 (少样本硬门).
    try:
        ratio_report = check_train_test_ratio(
            n_traj_train=n_traj_train,
            n_traj_test=n_traj_test,
        )
        if ratio_report.get('violated') or ratio_report.get('violated_traj_te_min'):
            leak_items.append({
                'kind': 'section9_train_test_ratio',
                'n_traj_train': int(n_traj_train),
                'n_traj_test': int(n_traj_test),
                'ratio': float(ratio_report['ratio']),
                'max_allowed': float(ratio_report['max_allowed']),
                'n_traj_te_min': int(ratio_report['n_traj_te_min']),
                'violated': bool(ratio_report.get('violated')),
                'violated_traj_te_min': bool(ratio_report.get('violated_traj_te_min')),
                'reason': ratio_report.get('message', ''),
            })
    except Exception as exc:  # pragma: no cover - 协议层异常不应中断切分
        warnings.warn(
            f'_emit_section9_leak_warnings: check_train_test_ratio 异常 ({exc!r})',
            stacklevel=2,
        )

    # §9.3 「测试覆盖 ≥3 种锚点-环境布局族, 禁单布局刷满 20 轨」.
    try:
        record_by_seq_id: dict[str, dict[str, Any]] = {}
        for record in sequences:
            if isinstance(record, dict) and 'seq_id' in record:
                record_by_seq_id[str(record['seq_id'])] = record

        test_family_set: set[str] = set()
        bad_seq_ids: list[str] = []
        for seq_id in test_ids:
            seq_id_str = str(seq_id)
            record = record_by_seq_id.get(seq_id_str)
            if not isinstance(record, dict):
                bad_seq_ids.append(seq_id_str)
                continue
            family_id = _resolve_sequence_layout_family(record)
            if family_id is None:
                continue
            test_family_set.add(family_id)

        if bad_seq_ids:
            warnings.warn(
                f'_emit_section9_leak_warnings: 测试段含未在 sequences 中的 seq_id '
                f'{bad_seq_ids[:5]}{"..." if len(bad_seq_ids) > 5 else ""}',
                stacklevel=2,
            )

        family_report = check_layout_family_count(len(test_family_set))
        if family_report.get('violated'):
            leak_items.append({
                'kind': 'section9_layout_family',
                'n_test_layout_families': int(family_report['n_test_layout_families']),
                'min_required': int(family_report['min_required']),
                'test_family_ids': sorted(test_family_set),
                'reason': family_report.get('message', ''),
            })
    except Exception as exc:  # pragma: no cover - 协议层异常不应中断切分
        warnings.warn(
            f'_emit_section9_leak_warnings: check_layout_family_count 异常 ({exc!r})',
            stacklevel=2,
        )

    # §9 细节「轨迹/NLOS/初值/网络初始化/划分 多源种子应分列 (§9 细节)」.
    # 协议 seed_policy.scene_seed_scope 列出要求独立存在的种子类别, 本审计校验
    # dataset_manifest.seeds (若存在) 中各 scope 字段独立存在. 缺 seeds 字段时
    # 不报错 (新生产路径可能尚未写 seeds; 仅记审计跳过, 由调用方审计).
    try:
        from liquidloc.protocol.experiment_gates import (
            get_seed_policy,
        )
        seed_policy = get_seed_policy()
        scene_seed_scope = seed_policy.get('scene_seed_scope') or []
        seeds_dict = dataset_manifest.get('seeds') if isinstance(dataset_manifest, dict) else None
        if isinstance(seeds_dict, Mapping) and scene_seed_scope:
            missing_scope_keys = [
                key for key in scene_seed_scope
                if key not in seeds_dict
            ]
            if missing_scope_keys:
                leak_items.append({
                    'kind': 'section9_scene_seed_scope',
                    'missing_scope_keys': list(missing_scope_keys),
                    'expected_scope_keys': list(scene_seed_scope),
                    'present_keys': sorted(list(seeds_dict.keys())),
                    'reason': (
                        '§9 细节: 多源种子 (scene/nlos_packet_loss/init_value/'
                        'network_init/split) 应分列; 缺失字段会污染种子独立性审计'
                    ),
                })
    except Exception as exc:  # pragma: no cover - 协议层异常不应中断切分
        warnings.warn(
            f'_emit_section9_leak_warnings: scene_seed_scope 审计异常 ({exc!r})',
            stacklevel=2,
        )
