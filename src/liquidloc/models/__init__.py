"""模型总入口。

这个文件不实现具体模型，只负责把 `liquid`、`lstm` 和 `transformer` 三条模型链路的
核心类、推理函数和训练函数统一导出，方便上层从一个稳定位置完成引用。
上游通常是工厂、训练脚本和推理脚本；下游则会继续进入各自子目录里的
具体网络、训练器和推理器实现。
"""

from liquidloc.common.types import ModelIntermediate  # 统一的中间量数据结构，供三条模型链路共享。
from liquidloc.models.liquid.inference import infer_intermediate as infer_liquid_intermediate  # Liquid 推理入口。
from liquidloc.models.liquid.network import LiquidNetwork  # Liquid 网络类。
from liquidloc.models.lstm.inference import infer_intermediate as infer_lstm_intermediate  # LSTM 推理入口。
from liquidloc.models.lstm.network import (  # LSTM 网络相关工具。
    LSTMNetwork,  # LSTM 网络类。
    build_lstm_sequence_tensor,  # LSTM 序列构造函数。
    normalize_structured_window,  # LSTM 结构化窗口标准化函数。
    resolve_lstm_network_cfg,  # LSTM 配置解析函数。
)
from liquidloc.models.transformer.inference import infer_intermediate as infer_transformer_intermediate  # Transformer 推理入口。
from liquidloc.models.transformer.network import (  # Transformer 网络相关工具。
    TransformerNetwork,  # Transformer 网络类。
    build_transformer_sequence_tensor,  # Transformer 序列构造函数。
    normalize_structured_window as normalize_transformer_window,  # Transformer 结构化窗口标准化函数（与 LSTM 同接口）。
    resolve_transformer_network_cfg,  # Transformer 配置解析函数。
)

# 延迟导入训练入口，避免 model_factory → models → trainer → model_factory 循环导入。
# 只有在首次访问 train_liquid_model / train_lstm_model / train_transformer_model 时才会触发实际导入。


def __getattr__(name: str):
    """延迟导入训练函数，打破 model_factory ↔ trainer 的循环依赖。"""
    if name == "train_liquid_model":
        from liquidloc.models.liquid.trainer import train_model
        return train_model
    if name == "train_lstm_model":
        from liquidloc.models.lstm.trainer import train_model
        return train_model
    if name == "train_transformer_model":
        from liquidloc.models.transformer.trainer import train_model
        return train_model
    raise AttributeError(f"module {__name!r} has no attribute {name!r}")


# 这里把模型总入口真正允许外部引用的符号列出来，避免调用方直接依赖内部私有实现。
__all__ = (
    "LSTMNetwork",  # LSTM 网络类。
    "LiquidNetwork",  # Liquid 网络类。
    "TransformerNetwork",  # Transformer 网络类。
    "ModelIntermediate",  # 通用中间量结构。
    "build_lstm_sequence_tensor",  # LSTM 序列构造函数。
    "build_transformer_sequence_tensor",  # Transformer 序列构造函数。
    "infer_liquid_intermediate",  # Liquid 中间量推理函数。
    "infer_lstm_intermediate",  # LSTM 中间量推理函数。
    "infer_transformer_intermediate",  # Transformer 中间量推理函数。
    "normalize_structured_window",  # LSTM 结构化窗口标准化函数。
    "normalize_transformer_window",  # Transformer 结构化窗口标准化函数。
    "resolve_lstm_network_cfg",  # LSTM 配置解析函数。
    "resolve_transformer_network_cfg",  # Transformer 配置解析函数。
    "train_liquid_model",  # Liquid 训练入口。
    "train_lstm_model",  # LSTM 训练入口。
    "train_transformer_model",  # Transformer 训练入口。
)
