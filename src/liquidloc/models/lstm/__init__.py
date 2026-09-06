"""LSTM 模块对外统一入口。

这个文件不实现模型本身，只负责把 `lstm` 子目录里真正给外部使用的
类、函数和训练入口集中导出，方便上层代码通过一个稳定路径完成调用。
上游通常是 `src/liquidloc/factories/model_factory.py`、训练脚本和推理脚本；
下游则是 `network.py`、`inference.py`、`trainer.py` 中的具体实现。

这个入口存在的目的，是把"模型结构""推理逻辑""训练流程"三类能力
放在一个明确的命名空间里，避免调用方去记每个实现文件的具体位置。
如果以后内部实现移动，只要这里保持导出契约不变，上层导入方式就不用改。
"""

from liquidloc.models.lstm.inference import infer_intermediate  # 导出 LSTM 中间量推理函数，给外部统一调用。
from liquidloc.models.lstm.network import (  # 导出网络结构相关工具，供训练、推理和配置解析共用。
    LSTMNetwork,  # 导出真正的 LSTM 网络类，供工厂、训练器和推理器实例化。
    build_lstm_sequence_tensor,  # 导出结构化窗口转序列张量的构造函数，保证输入形状一致。
    normalize_structured_window,  # 导出结构化窗口标准化函数，供前后端统一校验字段。
    resolve_lstm_network_cfg,  # 导出网络配置解析函数，供工厂和测试复用配置约定。
)

# 延迟导入训练入口，避免 model_factory → models → trainer → model_factory 循环导入。


def __getattr__(name: str):
    """延迟导入训练函数，打破 model_factory ↔ trainer 的循环依赖。"""
    if name == "train_model":
        from liquidloc.models.lstm.trainer import train_model
        return train_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (  # 定义该包对外允许显式导出的符号，避免外部误依赖内部私有实现。
    "LSTMNetwork",  # 对外暴露 LSTM 网络类。
    "build_lstm_sequence_tensor",  # 对外暴露结构化窗口转序列张量函数。
    "infer_intermediate",  # 对外暴露中间量推理函数。
    "normalize_structured_window",  # 对外暴露结构化窗口标准化函数。
    "resolve_lstm_network_cfg",  # 对外暴露 LSTM 配置解析函数。
    "train_model",  # 对外暴露训练入口函数。
)
