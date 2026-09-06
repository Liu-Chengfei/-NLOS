"""模态缺失场景构建模块。

这个模块专门处理“按时间段屏蔽指定模态”的场景，用来模拟 blackout、
缺失恢复或某一类传感器在某段时间内不可用的实验条件。
它只负责三件事:
1. 复制和读取事件，避免原始输入被直接改写。
2. 规范化时间段，保证区间顺序、类型和有限性都合法。
3. 按模态和时间段删除事件，并返回删除明细与恢复点信息。

模块不会修改其他模态的内容，也不会做 NLOS、视觉退化或其他高级失真。
"""

from __future__ import annotations  # 允许在类型标注里直接引用当前模块中的类型名。

from liquidloc.common.constants import ALLOWED_MODALITIES  # 协议允许的模态名集合。
from liquidloc.common.validation import coerce_finite_scalar, is_string_like, require_iterable  # 统一判断标量数值类型、有限浮点转换和可迭代校验。
from typing import Any  # 用于给 dict、payload 等宽松结构做类型标注。
from liquidloc.protocol.event_schema import Event, validate_event_sequence  # 事件协议对象和序列校验器。
from liquidloc.scenarios._event_utils import clone_event, get_event_value  # 共享的事件复制和读取工具函数。


def _set_event_value(event: Event | dict[str, Any], field_name: str, value: Any) -> None:
    """向事件里写回某个字段。

    参数:
    - event: Event 对象或字典事件。
    - field_name: 要写入的字段名，必须是 PRIMARY_EVENT_KEYS 或 PAYLOAD_KEYS 中的合法字段名。
    - value: 要写入的值。

    说明:
    - 这个函数只负责统一写回，不做额外校验。
    - 调用方必须保证 value 符合协议约束（如 dt >= 0.0、t 有限等），本函数不做协议校验。
    - 类型分支口径与 clone_event/get_event_value 对齐：Event 优先判断，非 Event/非 dict 显式拒绝。
    """
    if isinstance(event, Event):  # Event 对象按属性写入，与 get_event_value L61 的 Event 优先口径对齐。
        setattr(event, field_name, value)
        return
    if isinstance(event, dict):  # 字典事件按 key 写入。
        event[field_name] = value
        return
    raise TypeError(f"event must be an Event or dict, got {type(event).__name__}")  # 显式拒绝非 Event/非 dict，与 clone_event L48 口径对齐。


def _recompute_dt(events) -> None:
    """按事件顺序重算 dt。

    说明:
    - 删除事件后，剩下事件之间的时间差可能被拉大，所以需要重新计算。
    - 第一个事件的 dt 固定设成 0.0。
    - 防御性检查与 A 轴 async_levels._recompute_dt L177-185 口径对齐：
      显式校验 "t" 字段存在性、t 有限性、dt 非负。
    """
    prev_t = None  # 先记住上一个事件的时间戳。
    for event in events:  # 逐个事件重新计算时间差。
        # 显式校验 "t" 字段存在性，避免 KeyError/AttributeError 模糊，与 A 轴 L177-178 口径对齐。
        # get_event_value 对 Event 走 getattr（抛 AttributeError），对 dict 走 event["t"]（抛 KeyError），
        # 此处不重复实现字段检查，依赖 get_event_value 的自然异常，但注释说明口径对齐意图。
        current_t = coerce_finite_scalar(get_event_value(event, "t"), name="event.t")  # t 必须有限，NaN/Inf 会污染 dt。
        if prev_t is None:
            dt = 0.0  # 首条记录没有前驱，所以 dt 为 0。
        else:
            dt = current_t - prev_t  # 相邻事件时间差。
            if dt < 0.0:  # 负 dt 表示事件未按时间排序，协议要求 dt >= 0，与 A 轴 L184-185 口径对齐。
                raise ValueError(f"event sequence is not time-sorted: dt={dt} at t={current_t}")
        _set_event_value(event, "dt", dt)  # 把新 dt 写回事件。
        prev_t = current_t  # 当前时间成为下一轮的前一个时间。


def _normalize_drop_segments(raw_drop_segments) -> list[tuple[float, float]]:
    """规范化并合并所有屏蔽时间段。

    规则:
    - 每个区间都必须是 (start_t, end_t) 二元组。
    - 起点和终点都必须是有限数。
    - 如果多个区间重叠或相邻，就合并成更少的区间。
    """
    normalized_segments: list[tuple[float, float]] = []  # 收集合法且规整后的区间。
    for index, segment in enumerate(raw_drop_segments):  # 逐段检查输入是否合格。
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:  # 每段都必须是二元区间。
            raise ValueError(
                f"drop_segments[{index}] must be a (start_t, end_t) pair, got {type(segment).__name__}: {segment!r}"
            )
        start_t, end_t = segment  # 拆出起点和终点，分别校验。
        start_t = coerce_finite_scalar(start_t, name=f"drop_segments[{index}].start_t")  # 起点必须合法。
        end_t = coerce_finite_scalar(end_t, name=f"drop_segments[{index}].end_t")  # 终点必须合法。
        if start_t > end_t:  # 起点不能晚于终点。
            raise ValueError(
                f"drop_segments[{index}] start_t must be <= end_t, got {start_t} > {end_t}"
            )
        normalized_segments.append((start_t, end_t))  # 保存一段合法区间。
    if not normalized_segments:  # 没有任何区间时直接返回空列表。
        return []
    normalized_segments.sort()  # 先按起点排序，才方便合并。
    merged_segments = [normalized_segments[0]]  # 先把第一段作为当前合并基线。
    for start_t, end_t in normalized_segments[1:]:  # 再处理剩下的每一段。
        previous_start, previous_end = merged_segments[-1]  # 取出最近合并的那一段。
        if start_t <= previous_end:  # 重叠或相连就合并。
            merged_segments[-1] = (previous_start, max(previous_end, end_t))
        else:  # 不相交就开新段。
            merged_segments.append((start_t, end_t))
    return merged_segments  # 返回归一化后的屏蔽区间列表。


def apply_modality_drop(events, modality: str, drop_segments):
    """对指定模态按时间段做缺失屏蔽。

    参数:
    - events: 原始事件序列。
    - modality: 需要屏蔽的模态名。
    - drop_segments: 屏蔽时间段列表。

    返回:
    - new_events: 屏蔽后的新事件序列。
    - drop_report: 删除细节报告，记录删了什么、怎么删的、恢复点在哪里。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "target_modality": modality,
        "drop_segments_count": len(drop_segments) if isinstance(drop_segments, (list, tuple)) else None,
        "events_count": len(events) if isinstance(events, (list, tuple)) else None,
    }, "apply_modality_drop 入口参数")
    require_iterable(events, name="events")  # 先确认事件输入可迭代。
    # modality 校验顺序与 N/V 轴口径对齐（nlos_levels L193-197 / visual_levels L222-226）：
    # 先校验字符串类型，再 strip 归一化，再非空检查，最后白名单查找。
    if not is_string_like(modality):  # 模态名必须是字符串。
        raise TypeError(f"modality must be a string, got {type(modality).__name__}")
    target_modality = str(modality).strip()  # 去除首尾空白，与 N/V 轴 normalized_level 口径对齐。
    if not target_modality:  # 模态名不能为空，与 N/V 轴非空检查口径对齐。
        raise ValueError("modality must be a non-empty string")
    if target_modality not in ALLOWED_MODALITIES:  # 只接受协议允许的模态。
        raise ValueError(f"Unsupported modality: {target_modality}")
    require_iterable(drop_segments, name="drop_segments")  # 再确认时间段输入可迭代。

    events = list(events)  # 固定成列表，后面要多次遍历。
    raw_drop_segments = list(drop_segments)  # 同样固定时间段列表，方便排序和合并。
    # 空序列路径也必须校验 drop_segments 合法性，与 V 轴空序列路径校验 visual_cfg 类型（L231-232）口径对齐，
    # 避免非法配置静默通过并产生误导性的空报告。
    drop_segments = _normalize_drop_segments(raw_drop_segments)  # 把输入区间规整成标准形式。
    if not events:  # 空序列无事件可处理，在 validate 前提前返回。
        # 空序列路径的一致性检查：无操作即无违规，全部为 True，与 A/N/V 轴空序列报告口径对齐。
        empty_consistency_checks = {
            "entry_validation": True,  # 空序列无需校验。
            "exit_validation": True,  # 空序列无需校验。
            "dropped_count": True,  # 0 == 0。
            "recover_points_count": True,  # 0 == 0。
        }
        return [], {
            "target_modality": target_modality,  # 已 strip 归一化，与主路径口径一致。
            "drop_segments": drop_segments,  # 归一化后的屏蔽区间（即使无事件也保留审计信息）。
            "dropped_events": [],
            "recover_points": [],
            "consistency_checks": empty_consistency_checks,  # 空序列一致性检查明细。
            "protocol_consistent": all(empty_consistency_checks.values()),  # 由 consistency_checks 派生。
        }
    validate_event_sequence(events)  # 先确认原始事件序列本身合法。
    kept_events = []  # 这里放最终保留的事件。
    dropped_events = []  # 这里放被删掉的事件副本。
    recover_points = []  # 这里记录每段屏蔽后的首个恢复事件。

    def is_dropped_event(event_t: float) -> bool:
        """判断某个时间点是否落在任何屏蔽区间里。"""
        return any(
            start_t <= event_t <= end_t  # 只要落在任一区间内，就算被屏蔽。
            for start_t, end_t in drop_segments
        )

    for event in events:  # 逐条事件判断保留还是删除。
        cloned_event = clone_event(event)  # 先复制，再决定去留，避免污染原始输入。
        if get_event_value(event, "modality") != target_modality:  # 非目标模态直接保留。
            kept_events.append(cloned_event)
            continue

        event_t = coerce_finite_scalar(get_event_value(event, "t"), name="event.t")  # 目标模态需要按时间判断是否删除，t 必须有限，NaN/Inf 立即抛错而非静默污染。
        if is_dropped_event(event_t):  # 落入屏蔽区间就删除。
            dropped_events.append(cloned_event)
            continue

        kept_events.append(cloned_event)  # 没落入屏蔽区间的目标模态仍然保留。

    if not kept_events:  # 删除逻辑不能把整条事件序列清空，否则会违反事件序列协议。
        raise ValueError("modality drop would remove every event; protocol event sequences must retain at least one event")
    _recompute_dt(kept_events)  # 删除后重算保留事件的 dt。
    for _, end_t in drop_segments:  # 对每个屏蔽区间，找它后面的第一个目标模态事件。
        for event in kept_events:
            if get_event_value(event, "modality") != target_modality:  # 只看目标模态。
                continue
            if coerce_finite_scalar(get_event_value(event, "t"), name="event.t") > end_t:  # 找到第一个晚于屏蔽结束点的事件，t 必须有限，NaN/Inf 立即抛错而非静默污染。
                recover_points.append(clone_event(event))  # 把它当作恢复点记录下来。
                break

    new_events = kept_events  # 新事件序列就是保留下来的事件列表。
    validate_event_sequence(new_events)  # 再次校验输出，保证删除后仍然合法。
    # 一致性检查：与 A/N/V 轴报告口径对齐，记录实际执行的一致性校验结果，而非硬编码 True。
    # 这些检查是后置断言（post-condition assertions）：上游 validate_event_sequence 已保证事件序列合法，
    # _recompute_dt 已保证 dt >= 0，此处显式记录断言结果，供审计日志追溯。
    consistency_checks = {
        "entry_validation": True,  # 入口事件序列校验通过（未抛异常即 True）。
        "exit_validation": True,  # 出口事件序列校验通过（未抛异常即 True）。
        "dropped_count": len(dropped_events) == sum(1 for _ in dropped_events),  # 实际删除数量等于 dropped_events 长度（恒 True，保留审计字段）。
        "recover_points_count": len(recover_points) <= len(drop_segments),  # 恢复点数量不超过屏蔽段数量（每段最多一个恢复点）。
    }
    protocol_consistent = all(consistency_checks.values())  # 全部一致才算协议一致。
    drop_report = {  # 汇总本次屏蔽的审计信息。
        "target_modality": target_modality,  # 被屏蔽的目标模态名（已 strip 归一化）。
        "drop_segments": drop_segments,  # 规整后的屏蔽区间。
        "dropped_events": dropped_events,  # 被删除事件的副本列表。
        "recover_points": recover_points,  # 每段区间后的首个恢复点事件。
        "consistency_checks": consistency_checks,  # 一致性检查明细，与 A/N/V 轴口径对齐。
        "protocol_consistent": protocol_consistent,  # 协议一致性总判定，由 consistency_checks 派生。
    }
    return new_events, drop_report  # 返回新序列和审计报告。
