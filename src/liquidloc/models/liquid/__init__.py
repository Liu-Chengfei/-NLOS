"""Liquid 子模块统一入口。

这个文件只负责把 Liquid 网络、推理入口和训练入口集中导出。
它不写业务逻辑，只定义上层应该从哪里拿到这些符号。
"""

from liquidloc.models.liquid.inference import infer_intermediate  # 导出液体模型的中间量推理入口，供上层调用。
from liquidloc.models.liquid.network import LiquidNetwork  # 导出液体网络类，供构造和前向使用。
from liquidloc.models.liquid.output_head import LiquidOutputHead, RiskCalibration  # 导出输出头与风险校准模块。

# 延迟导入训练入口，避免 model_factory → models → trainer → model_factory 循环导入。


def __getattr__(name: str):
    """延迟导入训练函数，打破 model_factory ↔ trainer 的循环依赖。"""
    if name == "train_model":
        from liquidloc.models.liquid.trainer import train_model
        return train_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (  # 显式声明这个包允许外部直接导入的符号。
    "LiquidNetwork",  # 液体网络类。
    "LiquidOutputHead",  # 液体输出头类。
    "RiskCalibration",  # 风险校准模块。
    "infer_intermediate",  # 液体中间量推理函数。
    "train_model",  # 液体训练函数。
)
