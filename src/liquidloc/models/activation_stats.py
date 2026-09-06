"""激活统计工具（准则 27）。

为神经网络 5 个 head (bias / risk / uwb_scaling / vio_scaling / imu_proxy) 提供
激活统计导出：均值、方差、稀疏度、饱和度。供调试和论文激活图使用。

使用模式：
- 通过 attach_activation_stats_hooks(model) 把钩子挂到指定 head 上，
  训练/推理后用 collect_activation_stats() 拿到所有 head 的统计。
- detach_activation_stats_hooks(model) 在结束时清除钩子，避免影响推理路径。

设计原则：
- 仅统计 forward 路径上的输出 tensor (numpy 数组)，
  不计算梯度，不修改原网络结构。
- 统计缓存采用 deque(maxlen=N)，避免长序列内存爆炸。
"""

from __future__ import annotations

import collections
from typing import Any, Mapping

import numpy as np


# 默认 head 名称注册表（与 liquidloc.common.constants.MODEL_INTERMEDIATE_KEYS 一致）
DEFAULT_HEAD_NAMES: tuple[str, ...] = (
    "bias",
    "risk",
    "uwb_scaling",
    "vio_scaling",
    "imu_proxy",
)


class ActivationStatsBuffer:
    """单 head 的激活统计缓冲。

    累积最近 N 次 forward pass 的输出 tensor，记录均值/方差/稀疏度/饱和度。
    """

    def __init__(self, head_name: str, maxlen: int = 1024) -> None:
        self.head_name = head_name
        self.maxlen = maxlen
        self._values: collections.deque[float] = collections.deque(maxlen=maxlen)

    def update(self, tensor: Any) -> None:
        """记录一次 forward 输出 tensor（numpy 数组或标量）。"""
        # 解 autograd 梯度（准则 27：仅统计输出，不污染梯度图）
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        try:
            arr = np.asarray(tensor, dtype=np.float64)
        except (TypeError, ValueError):
            return
        # 标量：直接 append；数组：展平后逐元素 append
        if arr.ndim == 0:
            self._values.append(float(arr))
        else:
            for v in arr.flatten().tolist():
                self._values.append(float(v))

    def summary(self) -> dict[str, float]:
        """返回当前累积的统计。"""
        if not self._values:
            return {"mean": 0.0, "std": 0.0, "sparsity": 0.0, "saturation": 0.0, "count": 0}
        values = np.asarray(self._values, dtype=np.float64)
        finite_mask = np.isfinite(values)
        finite_vals = values[finite_mask]
        if finite_vals.size == 0:
            return {"mean": 0.0, "std": 0.0, "sparsity": 0.0, "saturation": 0.0, "count": 0}
        mean = float(finite_vals.mean())
        std = float(finite_vals.std())
        # 稀疏度 = |x| < 1e-3 的比例
        sparsity = float((np.abs(finite_vals) < 1e-3).mean())
        # 饱和度 = 接近 max/min 边界的比例（head 输出已被 sigmoid/tanh 归一时）
        near_one = np.abs(np.abs(finite_vals) - 1.0) < 1e-3
        saturation = float(near_one.mean())
        return {
            "mean": mean,
            "std": std,
            "sparsity": sparsity,
            "saturation": saturation,
            "count": int(finite_vals.size),
        }

    def reset(self) -> None:
        self._values.clear()


class ActivationStatsCollector:
    """跨多 head 的激活统计聚合器。

    给定一组 head_name + maxlen，自动管理 buffer，并按 head 输出 dict[str, float]
    形式的合并统计。
    """

    def __init__(self, head_names: tuple[str, ...] = DEFAULT_HEAD_NAMES, maxlen: int = 1024) -> None:
        self._buffers: dict[str, ActivationStatsBuffer] = {
            name: ActivationStatsBuffer(name, maxlen=maxlen) for name in head_names
        }

    def update(self, head_outputs: Mapping[str, Any]) -> None:
        """记录一次 multi-head forward 输出。"""
        for name, tensor in head_outputs.items():
            buf = self._buffers.get(name)
            if buf is not None:
                buf.update(tensor)

    def summary(self) -> dict[str, dict[str, float]]:
        """返回 {head_name: {mean, std, sparsity, saturation, count}}。"""
        return {name: buf.summary() for name, buf in self._buffers.items()}

    def reset(self) -> None:
        for buf in self._buffers.values():
            buf.reset()


def attach_activation_stats_hooks(
    model: Any,
    collector: ActivationStatsCollector,
    head_attr_name: str = "output_heads",
) -> list[Any]:
    """给 model.output_heads (nn.ModuleDict) 中的每个 head 注册 forward hook，
    每次 forward 后把输出 tensor 写入 collector。

    返回注册的 hook handle 列表，调用方需在结束后调 detach_activation_stats_hooks
    移除以避免内存泄漏。
    """
    handles: list[Any] = []
    heads = getattr(model, head_attr_name, None)
    if heads is None or not hasattr(heads, "items"):
        return handles

    for head_name, head_module in heads.items():
        def make_hook(n: str) -> Any:
            def _hook(_module: Any, _inputs: tuple, output: Any) -> None:
                # output 可能是 tensor 或 tuple-of-tensors
                if isinstance(output, tuple) and len(output) >= 1:
                    payload = output[0]
                else:
                    payload = output
                collector.update({n: payload})
            return _hook

        handle = head_module.register_forward_hook(make_hook(head_name))
        handles.append(handle)
    return handles


def detach_activation_stats_hooks(handles: list[Any]) -> None:
    """移除所有 hook handle。"""
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


__all__ = (
    "ActivationStatsBuffer",
    "ActivationStatsCollector",
    "attach_activation_stats_hooks",
    "detach_activation_stats_hooks",
    "DEFAULT_HEAD_NAMES",
)