"""Runtime policies shared by training pipeline orchestration."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from liquidloc.common.constants import ALLOWED_TRAIN_DEVICES, DEVICE_AUTO, DEVICE_CPU, DEVICE_CUDA
from liquidloc.common.validation import is_string_like


def resolve_train_device(
    device_request: str | None,
    *,
    allow_auto_cuda: bool,
    cuda_available_fn: Callable[[], bool],
    cuda_runtime_fn: Callable[[], bool],
) -> dict[str, Any]:
    """Resolve a training device request into a stable runtime report."""
    if device_request is None:
        requested_device = DEVICE_AUTO
    elif is_string_like(device_request):
        requested_device = str(device_request).strip().lower()
    else:
        raise TypeError(f"device must be a string or None, got {type(device_request).__name__}")
    if requested_device not in ALLOWED_TRAIN_DEVICES:
        raise ValueError(f"device must be one of {sorted(ALLOWED_TRAIN_DEVICES)}, got: {device_request!r}")

    cuda_available = bool(cuda_available_fn())
    cuda_runtime_available = bool(cuda_available and cuda_runtime_fn())
    if requested_device == DEVICE_CPU:
        selected_device = DEVICE_CPU
    elif requested_device == DEVICE_CUDA and cuda_runtime_available:
        selected_device = DEVICE_CUDA
    elif requested_device == DEVICE_CUDA:
        selected_device = DEVICE_CPU
    else:
        selected_device = DEVICE_CUDA if allow_auto_cuda and cuda_runtime_available else DEVICE_CPU

    return {
        "requested_device": requested_device,
        "selected_device": selected_device,
        "cuda_available": cuda_available,
        "cuda_runtime_available": cuda_runtime_available,
    }


__all__ = ("resolve_train_device",)
