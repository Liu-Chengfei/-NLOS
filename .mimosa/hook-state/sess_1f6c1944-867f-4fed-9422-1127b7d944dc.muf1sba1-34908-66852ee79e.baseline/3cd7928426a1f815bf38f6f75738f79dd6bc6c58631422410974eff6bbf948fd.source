"""融合层公共导出模块。

本模块是 `liquidloc.fusion` 包的入口，把融合子目录里最常用的两个符号统一导出：
- `EventQueue`：事件队列，提供游标式顺序消费事件的能力
- `run_fusion`：融合主入口，驱动整条"事件 → 模型推理 → 控制量 → 估计器更新"链路

职责边界：
- 只做符号重导出，不承载任何额外业务逻辑
- 对外提供稳定的短路径（`from liquidloc.fusion import run_fusion`），避免调用方
  直接依赖子模块内部路径

与其他模块的关系：
- `event_queue.EventQueue`：事件队列实现，被 `fusion_runner` 间接使用
- `fusion_runner.run_fusion`：融合主循环，是整个 fusion 包的核心入口
"""

from liquidloc.fusion.event_queue import EventQueue  # 导出事件队列，供外部直接消费事件序列。
from liquidloc.fusion.fusion_runner import run_fusion  # 导出融合主入口，供外部直接跑整条融合链路。

__all__ = (  # 显式声明对外公开的名字，防止 `from liquidloc.fusion import *` 泄露内部符号。
    "EventQueue",  # 事件队列是 fusion 对外的基础工具之一。
    "run_fusion",  # run_fusion 是 fusion 对外的主入口之一。
)  # 导出列表在这里结束。
