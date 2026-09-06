"""融合侧事件队列——提供按游标顺序消费事件的队列封装。

本模块的职责：
1. 将输入事件序列标准化为内部不可变缓存（元组形态）
2. 提供游标式访问接口（peek / pop / has_next / reset）
3. 校验事件序列的合法性（委托给 protocol.event_schema）

核心概念：
- **游标（cursor）**：指向当前待消费事件的索引，从 0 开始，
  每次调用 pop() 前进一步，peek() 不前进
- **不可变缓存**：内部事件存储为元组，防止外部通过列表操作增删事件，
  确保融合主循环遍历的事件序列在运行期间不会改变
- **深拷贝隔离**：每次 peek/pop 返回事件的深拷贝，
  防止调用方修改返回值时污染内部缓存

分层边界：
- 本模块属于 **fusion 桥接层**，只做"事件序列的存储和游标管理"
- 不决定事件的内容（由数据 reader 产生），也不决定事件如何被消费（由估计器消费）
- 事件校验委托给 `protocol.event_schema.validate_event_sequence`

与其他模块的关系：
- `protocol.event_schema`：提供事件校验能力
- `fusion_runner`：本模块的上游调度者，可选使用 EventQueue 管理事件遍历

前提指导 §1.4 「学习隐状态按轨重置、轨间不泄漏」合同挂钩：
- 本模块当前**未被 fusion_runner.run_fusion 内部使用**（fusion_runner 直接遍历 list 形态 events；
  见 fusion_runner.py:493 处仅作守卫口径引用），EventQueue 是供外部消费方可选使用的工具类。
- 若未来 fusion_runner 改为内部使用 EventQueue 跨轨复用单实例，
  必须在每次 run_fusion 入口处显式调 ``EventQueue.reset()`` 把 cursor 归零，
  否则本次消费会从上一轨遗留 cursor 位置开始、产出错轨状态——直接破 §1.4。
- 单实例 EventQueue 跨轨复用还必须重新调构造函数（或重置内部 ``_events`` 缓存），
  否则上一轨的事件列表会泄漏到本次运行，违反 §1.4 轨间不泄漏合同。
"""

from __future__ import annotations  # 延迟解析类型注解，减少导入阶段依赖。

from copy import deepcopy  # 用于对事件做深拷贝，避免外部修改污染内部缓存。
from collections.abc import Iterable, Mapping  # 用于输入事件序列和 Mapping 守卫（Python 3.9+ 推荐 abc 版本）。
from typing import Any  # Any 用于事件字典类型。

from liquidloc.protocol.event_schema import Event, validate_event_sequence  # 导入事件对象和序列校验函数。


class EventQueue:
    """维护一个已经校验过的事件序列，并提供游标式访问。

    设计原则：
    - 构造时一次性校验整个序列，后续访问不再重复校验
    - 内部缓存不可变（元组），外部无法增删事件
    - 每次访问返回深拷贝，外部修改不影响内部状态
    - 游标只前进不后退（除 reset 外），保证消费顺序的确定性
    """

    def __init__(self, events: Iterable[Event | dict[str, Any]]):
        """将输入事件序列标准化成内部不可变缓存。

        处理流程：
        1. 展开可迭代输入为列表
        2. 校验序列合法性（时序、字段完整性等）
        3. 将每个事件标准化为字典快照（Event → dict，dict → deepcopy）
        4. 将列表转为元组，锁定内部缓存

        Args:
            events: 事件序列，每个元素可以是 Event 实例或字典；
                必须通过 validate_event_sequence 校验。

        Raises:
            ValueError: 当事件序列不满足协议要求时（由 validate_event_sequence 抛出）。
            TypeError: 当 events 是 str/bytes 或单个 Mapping（dict）时。
        """
        from liquidloc.common.tee_logger import print_dict
        try:
            _event_count = len(events) if hasattr(events, "__len__") else None
        except TypeError:
            _event_count = None
        print_dict({"event_count": _event_count, "events_type": type(events).__name__}, "EventQueue.__init__")
        # 显式拒绝会被 list() 误转换的类型，与 validate_event_sequence 的守卫对齐。
        # 必须在 list() 之前检查，否则：
        # - list(None) 抛不清晰的 TypeError，而非语义化的 ValueError
        # - list("abc") 拆成字符列表，绕过事件校验，错误信息令人困惑
        # - list({...}) 返回键列表，绕过 validate_event_sequence 的 Mapping 守卫
        if events is None:
            raise ValueError("events must not be None")
        if isinstance(events, (str, bytes)):
            raise TypeError("events must be a sequence of events, not a string or bytes")
        if isinstance(events, Mapping):
            raise TypeError("events must be a sequence of events, not a single mapping")
        event_list = list(events)  # 先把可迭代输入展开成列表，后续才能稳定遍历和校验。
        validate_event_sequence(event_list)  # 先验证整个事件序列是否满足协议要求。

        self._events = tuple(  # 内部事件缓存固定成元组，避免外部再直接增删。
            event.to_dict() if isinstance(event, Event) else deepcopy(event)  # Event 先转字典，普通字典则深拷贝。
            for event in event_list  # 对输入序列逐个处理。
        )  # 元组化后内部缓存不可被外部列表操作影响。
        self.cursor = 0  # 游标从第一个事件开始。
        self.queue_length = len(self._events)  # 记录总长度，便于快速判断是否还有下一个事件。

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        """返回内部事件缓存的只读快照（包含全部事件，不受游标位置影响）。

        返回的是每个事件的深拷贝组成的新元组，调用方可以自由修改返回值
        而不会影响内部缓存。注意：本属性返回全部事件，与游标当前位置无关；
        若需要按游标顺序消费，请使用 peek()/pop()。

        Returns:
            由全部事件的深拷贝字典组成的元组；元组本身和每个字典都是新对象。
        """
        return tuple(deepcopy(event) for event in self._events)  # 对外只暴露快照，避免外部改坏内部缓存。

    @property
    def current_event(self) -> dict[str, Any] | None:
        """返回当前游标位置的事件快照，但不推进游标。

        这是 peek() 的语义别名，提供更直观的属性访问方式。

        Returns:
            当前事件的深拷贝字典；若游标已到末尾则返回 None。
        """
        return self.peek()  # current_event 只是 peek 的语义别名。

    def has_next(self) -> bool:
        """判断队列里是否还有未消费事件。

        Returns:
            True 如果游标未到末尾，否则 False。
        """
        # 同时约束下界，避免 cursor 被外部置为负数时 has_next 误判为 True，
        # 否则 peek() 会因 Python 负索引静默返回末尾事件。
        return 0 <= self.cursor < self.queue_length  # 游标在 [0, queue_length) 内才说明还有剩余事件。

    def peek(self) -> dict[str, Any] | None:
        """查看当前事件，但不消费它（游标不前进）。

        返回的是深拷贝，调用方可以自由修改返回值而不会影响内部缓存。

        Returns:
            当前事件的深拷贝字典；若游标已到末尾则返回 None。
        """
        if not self.has_next():  # 如果已经没有下一个事件，就返回空。
            return None  # 这里不抛错，方便上层写循环判断。
        return deepcopy(self._events[self.cursor])  # 返回深拷贝，避免调用方改坏内部缓存。

    def pop(self) -> dict[str, Any] | None:
        """取出当前事件，并把游标前进一格。

        返回的是深拷贝，调用方可以自由修改返回值而不会影响内部缓存。
        游标只在成功取出事件后才前进；若已到末尾则不前进。

        Returns:
            当前事件的深拷贝字典；若游标已到末尾则返回 None。
        """
        event = self.peek()  # 先拿当前事件快照。
        if event is None:  # 如果已经没有事件了，就直接返回空。
            return None  # 不推进游标，保持尾部状态不变。
        self.cursor += 1  # 消费一个事件后，游标前进一步。
        return event  # 返回刚刚取出的事件快照。

    def reset(self) -> None:
        """把游标重置回第一个事件。

        典型用途：在多次融合运行之间重置队列，复用同一批事件。
        注意：重置只影响游标位置，不影响内部缓存内容。

        前提指导 §1.4 挂钩：本方法**仅重置 cursor**，不清理 ``_events`` 缓存。
        若跨轨复用单实例 EventQueue，调用方必须保证：
        (a) 在新轨入口处显式调 ``reset()`` 把 cursor 归零；
        (b) 通过重新构造或缓存替换确保 ``_events`` 是新轨的事件列表，
        旧轨事件不留存——否则违反 §1.4 \"轨间不泄漏\" 合同。
        当前 fusion_runner.run_fusion 不使用 EventQueue（直接遍历 events list），
        故本方法暂无 §1.4 强合同触发点；但作为可选外部工具，本方法的语义
        必须支持 §1.4 兼容（即仅重置消费侧 cursor、不污染事件真相）。
        """
        self.cursor = 0  # 直接把游标归零即可。
