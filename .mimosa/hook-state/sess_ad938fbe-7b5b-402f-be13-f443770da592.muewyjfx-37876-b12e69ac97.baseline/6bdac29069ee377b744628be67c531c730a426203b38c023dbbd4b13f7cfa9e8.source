"""线性层权重初始化 helper。

职责：
    提供跨模块复用的线性层等间距初始化函数，避免 cell.py 与 network.py
    重复定义导致漂移。此处为 _fill_linear 的权威定义来源。

上游依赖：
    - 无内部模块依赖（仅依赖 torch）

下游调用者：
    - liquidloc.models.liquid.cell — LiquidCell 初始化投影层时复用 _fill_linear
    - liquidloc.models.liquid.network — Liquid 网络初始化共享投影层时复用 _fill_linear
"""

from __future__ import annotations

import torch
from torch import nn


def _fill_linear(linear: nn.Linear, *, start: float, end: float) -> None:
    """用线性等间距值初始化线性层的权重和偏置。

    参数:
    `linear` 是要初始化的线性层。
    `start` 是权重初始化的起始值。
    `end` 是权重初始化的结束值。

    返回值:
    无返回值，只修改传入层的参数。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"linear": linear, "start": start, "end": end}, "_fill_linear 入口参数")
    with torch.no_grad():  # 初始化阶段不需要梯度记录。
        linear.weight.copy_(torch.linspace(start, end, steps=linear.weight.numel(), device=linear.weight.device, dtype=linear.weight.dtype).view_as(linear.weight))  # 用等间距值填充权重，对齐 device/dtype 防止跨设备拷贝。
        if linear.bias is not None:  # 如果有偏置就也做初始化。
            linear.bias.copy_(torch.linspace(-0.05, 0.05, steps=linear.bias.numel(), device=linear.bias.device, dtype=linear.bias.dtype))  # 偏置用小范围等间距值填充，对齐 device/dtype。
