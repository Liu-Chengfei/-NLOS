"""LiquidLoc 包入口。

本模块是 liquidloc 顶层包的 __init__.py，负责导出核心公共类型。
这些类型是跨模块交互的基础数据结构，下游 pipeline、metrics、analysis 等模块均依赖它们。

导出类型说明：
    MeasurementControl : 量测控制信号，学习前端输出给后端 EKF 的调节量。
    MetricRow          : 指标行，单次评估结果的一行指标数据。
    ModelIntermediate  : 模型中间结果，训练/推理过程中前端网络的中间输出。
    PredictionBundle   : 预测包，包含状态估计、协方差等完整预测结果。
    StageResult        : 阶段结果，单阶段（训练/评估/推理）的完整产出。
    StateEstimate      : 状态估计，单个时刻的滤波器状态和协方差。
"""

# 从 common.types 导出核心公共类型，供外部和跨模块使用。
from liquidloc.common.types import (
    MeasurementControl,  # 量测控制信号：学习前端输出给后端 EKF 的调节量。
    MetricRow,           # 指标行：单次评估结果的一行指标数据。
    ModelIntermediate,   # 模型中间结果：训练/推理过程中前端网络的中间输出。
    PredictionBundle,    # 预测包：包含状态估计、协方差等完整预测结果。
    StageResult,         # 阶段结果：单阶段（训练/评估/推理）的完整产出。
    StateEstimate,       # 状态估计：单个时刻的滤波器状态和协方差。
)

# __all__ 显式声明公开 API，防止内部符号被 from liquidloc import * 泄露。
__all__ = (
    "MeasurementControl",
    "MetricRow",
    "ModelIntermediate",
    "PredictionBundle",
    "StageResult",
    "StateEstimate",
)
