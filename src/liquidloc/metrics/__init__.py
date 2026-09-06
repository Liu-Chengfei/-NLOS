"""文件:src/liquidloc/metrics/__init__.py  # 文件头：说明这个文件的身份。

【文件职责】metrics 包的公共导出入口，把最常用的指标计算函数统一暴露给外部，  # 说明本文件的主职责。
方便外部从一个地方直接拿到 metric runner 和各类分项指标函数，而不用记住每个内部文件名。  # 说明为什么需要这个文件。

【上游依赖】  # 说明本文件依赖哪些模块。
- metric_runner.py：提供统一指标计算入口 compute_metrics。  # 总调度器。
- trajectory_metrics.py：提供轨迹误差指标计算入口 compute_trajectory_metrics。  # 轨迹指标。
- tail_metrics.py：提供尾部指标计算入口 compute_tail_metrics。  # 尾部指标。
- reliability_metrics.py：提供可靠性指标计算入口 compute_reliability_metrics。  # 可靠性指标。
- runtime_metrics.py：提供运行时指标计算入口 compute_runtime_metrics。  # 运行时指标。

【下游调用者】  # 说明哪些模块/脚本会 import 这个包。
- analysis/*：分析脚本通过 from liquidloc.metrics import compute_metrics 等方式调用。  # 分析面。
- scripts/11_compute_metrics.py：指标计算脚本。  # 计算脚本。
- tests/metrics/*：测试代码。  # 测试面。
- 任何需要直接调用指标计算的外部模块。  # 其他调用方。

【核心变量定义】  # 说明本文件导出的核心对象。
- compute_metrics：统一指标计算入口，支持单序列和多序列。  # 总入口函数。
- compute_trajectory_metrics：轨迹误差指标（RMSE/MAE/ATE/RPE）。  # 轨迹指标函数。
- compute_tail_metrics：尾部指标（p95/p99/failure_rate/长失败段）。  # 尾部指标函数。
- compute_reliability_metrics：可靠性指标（覆盖率/相关系数/有效样本数/状态）。  # 可靠性指标函数。
- compute_runtime_metrics：运行时指标（延迟均值/分位数/参数量/峰值内存）。  # 运行时指标函数。
"""

from liquidloc.metrics.metric_runner import compute_metrics  # 导入统一指标计算入口，这是外部最常用的调用点。
from liquidloc.metrics.reliability_metrics import compute_reliability_metrics  # 导入可靠性指标计算函数，用于风险-误差相关性分析。
from liquidloc.metrics.runtime_metrics import compute_runtime_metrics  # 导入运行时指标计算函数，用于延迟和资源统计。
from liquidloc.metrics.tail_metrics import compute_tail_metrics  # 导入尾部指标计算函数，用于长尾和失败率分析。
from liquidloc.metrics.trajectory_metrics import compute_trajectory_metrics  # 导入轨迹误差指标计算函数，用于 RMSE/MAE/ATE/RPE。

__all__ = (  # 定义 from liquidloc.metrics import * 时导出的公共接口列表。
    "compute_metrics",  # 统一指标计算入口。
    "compute_reliability_metrics",  # 可靠性指标计算入口。
    "compute_runtime_metrics",  # 运行时指标计算入口。
    "compute_tail_metrics",  # 尾部指标计算入口。
    "compute_trajectory_metrics",  # 轨迹误差指标计算入口。
)
