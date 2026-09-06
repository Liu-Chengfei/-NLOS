"""工厂入口聚合模块。

职责
----
把估计器、指标和模型的创建入口集中导出，方便上层代码统一从
``liquidloc.factories`` 取对象，而不需要知道各子工厂的具体路径。

上游依赖
--------
- ``liquidloc.factories.estimator_factory``  — 估计器创建入口
- ``liquidloc.factories.metric_factory``     — 指标计算器创建入口
- ``liquidloc.factories.model_factory``      — 模型创建入口

下游调用者
----------
- ``liquidloc.pipelines.*``  — 各流水线通过本模块获取工厂函数
- ``scripts/`` 下的训练 / 评估脚本
- ``tests/factories/*``  — 工厂层单元测试

核心变量
--------
- ``create_estimator``          — 估计器工厂函数
- ``create_metric_calculator``  — 指标计算器工厂函数
- ``create_model``              — 模型工厂函数
"""

from liquidloc.factories.estimator_factory import create_estimator  # 导出估计器创建入口。
from liquidloc.factories.metric_factory import create_metric_calculator  # 导出指标计算器创建入口。
from liquidloc.factories.model_factory import create_model  # 导出模型创建入口。

__all__ = (
    "create_estimator",  # 对外公开的估计器创建函数名。
    "create_metric_calculator",  # 对外公开的指标计算器创建函数名。
    "create_model",  # 对外公开的模型创建函数名。
)
