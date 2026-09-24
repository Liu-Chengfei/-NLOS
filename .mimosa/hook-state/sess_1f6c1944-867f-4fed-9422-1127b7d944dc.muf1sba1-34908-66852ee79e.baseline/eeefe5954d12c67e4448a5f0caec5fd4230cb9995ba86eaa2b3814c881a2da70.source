"""场景模块共享的事件工具函数。

文件职责：
  提供 clone_event、get_event_value 和 stable_window_start 三个公共函数，
  供 async_levels.py / nlos_levels.py / missing_modalities.py / visual_levels.py 共用，
  避免在多个模块里重复定义相同逻辑。

本文件绝对不负责：
  不负责场景构建、事件修改或协议校验。
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

from liquidloc.common.validation import is_integer, is_string_like
from liquidloc.protocol.event_schema import Event


def clone_event(event: Event | dict[str, Any]) -> Event | dict[str, Any]:
    """复制单个事件，保证后续修改不影响原始输入。

    参数:
        event: 协议事件对象，或兼容的字典形式事件。

    返回:
        与输入形态一致的深拷贝事件。

    异常:
        TypeError: 输入既不是 Event 也不是 dict。
    """
    # 第 11 轮审查 MEDIUM-1 修复（R11-A M1）：clone_event 是事件复制入口函数，
    # 每事件调用一次（apply_async_level / nlos_levels / missing_modalities / visual_levels
    # 均经此路径）。入口 print_dict 违反工程规范
    # "Recursive functions and entry points should avoid print_dict calls"，
    # 且与 R10-A M1（apply_async_level 入口 print_dict 移除）同源。
    if isinstance(event, Event):
        return Event(
            t=event.t,
            dt=event.dt,
            modality=event.modality,
            meta=deepcopy(event.meta),
            imu_payload=deepcopy(event.imu_payload),
            uwb_payload=deepcopy(event.uwb_payload),
            vio_payload=deepcopy(event.vio_payload),
            flow_payload=deepcopy(event.flow_payload),
            tof_payload=deepcopy(event.tof_payload),
        )
    if isinstance(event, dict):
        return deepcopy(event)
    raise TypeError(f"event must be an Event or dict, got {type(event).__name__}")


def get_event_value(event: Event | dict[str, Any], key: str) -> Any:
    """从事件里读取某个字段的值，兼容 Event 对象和字典两种形态。

    参数:
        event: Event 对象或字典事件。
        key:   要读取的字段名。

    返回:
        对应字段的值。
    """
    if isinstance(event, Event):
        return getattr(event, key)
    return event[key]


def stable_window_start(window_count: int, selected_count: int, *, key: str, selection_mode: str = "window") -> int | list[int]:
    """用稳定哈希确定窗口的起始位置或散布索引列表。

    相同的 key 始终映射到相同的起始位置/散布索引, 保证实验可复现。
    本函数是 async_levels.py / nlos_levels.py / visual_levels.py 的共享实现，
    消除三处重复定义。

    参数:
        window_count:   窗口总数（可选起始位置范围 / Poisson 散布池总数）。
        selected_count: 需要选中的窗口数量。
        key:            哈希键，通常由等级名、场景 ID 等拼接而成。
        selection_mode: "window" (v1 默认, 选连续窗口 [start, start+selected_count))
                        或 "poisson" (v2 D-2 新增, 按 Poisson 过程散布 selected_count 个 index).

    返回:
        "window" 模式: 窗口起始索引 (int), 保证 [start, start + selected_count) 不越界.
        "poisson" 模式: 散布索引列表 (list[int]), 长度 = selected_count, 元素按 Poisson 过程散布于 [0, window_count).

    异常:
        TypeError:  window_count/selected_count 不是非 bool 整数, key 不是字符串,
                    selection_mode 不是 str.
        ValueError: window_count/selected_count 为负数, selection_mode 不是 "window"/"poisson".
    """
    # 运行时类型校验：bool 是 int 子类，必须显式排除；负数窗口数无意义。
    if not is_integer(window_count) or isinstance(window_count, bool):
        raise TypeError(f"window_count must be a non-bool integer, got {type(window_count).__name__}")
    if not is_integer(selected_count) or isinstance(selected_count, bool):
        raise TypeError(f"selected_count must be a non-bool integer, got {type(selected_count).__name__}")
    if not is_string_like(key):
        raise TypeError(f"key must be a string, got {type(key).__name__}")
    if not is_string_like(selection_mode):
        raise TypeError(f"selection_mode must be a string, got {type(selection_mode).__name__}")
    if selection_mode not in ("window", "poisson"):
        raise ValueError(f"selection_mode must be 'window' or 'poisson', got {selection_mode!r}")
    if window_count < 0 or selected_count < 0:
        raise ValueError(f"window_count and selected_count must be non-negative, got {window_count}/{selected_count}")
    if selected_count <= 0 or window_count <= selected_count:  # 无需选择或全选时起始为 0.
        if selection_mode == "poisson":
            return list(range(min(selected_count, window_count))) if selected_count > 0 else []
        return 0

    if selection_mode == "window":
        # v1 路径: 连续窗口, 起始位置由 SHA-256(key) 决定.
        available_starts = window_count - selected_count + 1  # 可选起始位置的数量.
        digest = hashlib.sha256(key.encode("utf-8")).digest()  # 对 key 做 SHA-256 哈希.
        return int.from_bytes(digest[:8], "big") % available_starts  # 取前 8 字节转整数, 取模得起始.

    # v2 D-2 路径: Poisson 散布.
    # 设 rate = selected_count / window_count (期望密度), 用 Knuth 算法对每个 index i ∈ [0, window_count)
    # 按 Poisson(rate·1) 决定是否选中, 直到选满 selected_count 个为止; 不足则按哈希补全到 selected_count,
    # 多余则截断到 selected_count. 这样保证: (a) 选中的 index 在 [0, window_count) 内; (b) 散布近似 Poisson 过程;
    # (c) 长度严格 = selected_count; (d) key 决定随机性, 可复现.
    return _poisson_scatter(window_count, selected_count, key=key)


def _poisson_scatter(window_count: int, selected_count: int, *, key: str) -> list[int]:
    """按 Poisson 过程散布 selected_count 个 index 到 [0, window_count) 区间.

    用稳定哈希 SHA-256(key + i) 作为第 i 个位置的 Bernoulli 试验输入, 期望密度
    rate = selected_count / window_count, 让稀疏均匀散布近似 Poisson 过程.
    若首轮采样不足 selected_count, 用哈希补全; 若超出, 按 SHA-256 排序截断前 selected_count 个.

    参数:
        window_count:   总窗口数 (Poisson 散布池).
        selected_count: 期望散布的索引数.
        key:            哈希键, 决定可复现性.

    返回:
        长度严格 = selected_count 的 index 列表, 元素散布于 [0, window_count), 不重复.
    """
    import math
    import random

    rate = selected_count / window_count  # 期望密度.
    # 第一轮: 对每个 index 做 Bernoulli(rate) 试验, 用 hash(key + i) 作种子.
    candidates: list[tuple[int, int]] = []  # (hash_rank, index) 对.
    for i in range(window_count):
        digest = hashlib.sha256(f"{key}#{i}".encode("utf-8")).digest()
        hash_int = int.from_bytes(digest[:8], "big")
        # Bernoulli 试验: hash_int / 2^64 < rate 表示选中.
        threshold = int(rate * (1 << 64))
        if hash_int < threshold:
            candidates.append((hash_int, i))

    # 第二轮: 若首轮不足, 从未选中位置按哈希补全.
    if len(candidates) < selected_count:
        selected_indices = {c[1] for c in candidates}
        # 对每个未选中位置计算哈希, 排序后取前 (selected_count - len) 个补全.
        remaining: list[tuple[int, int]] = []
        for i in range(window_count):
            if i in selected_indices:
                continue
            digest = hashlib.sha256(f"{key}#fill#{i}".encode("utf-8")).digest()
            hash_int = int.from_bytes(digest[:8], "big")
            remaining.append((hash_int, i))
        remaining.sort()
        need = selected_count - len(candidates)
        candidates.extend(remaining[:need])
    elif len(candidates) > selected_count:
        # 第三轮: 若超出, 按 hash_int 排序取前 selected_count 个 (确定性截断).
        candidates.sort()
        candidates = candidates[:selected_count]

    # 排序后输出, 让 index 单调递增 (调用方处理方便).
    sorted_indices = sorted(c[1] for c in candidates)
    return sorted_indices
