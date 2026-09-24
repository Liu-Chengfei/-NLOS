"""Model runtime concerns shared by model factory backends."""
from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from liquidloc.common.constants import (
    DEVICE_AUTO,
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_CUDA_PREFIX,
    MODEL_INTERMEDIATE_KEYS,
    RAM_PEAK_FLOOR_MB,
    RAM_PEAK_PARAMS_PER_MB,
)
from liquidloc.common.paths import get_standard_dirs
from liquidloc.common.seed_utils import cuda_runtime_usable
from liquidloc.common.validation import is_integer


def resolve_checkpoint_path(checkpoint_path: str | Path, cfg: Mapping[str, Any]) -> Path:
    path = Path(checkpoint_path).expanduser()
    if path.is_absolute():
        return path
    project_root = cfg.get("project_root")
    if project_root is None:
        raise ValueError(
            f"_resolve_checkpoint_path: checkpoint_path={checkpoint_path!r}; "
            "relative checkpoint_path requires cfg.project_root or an absolute path"
        )
    return get_standard_dirs(project_root)["project_root"] / path


def load_checkpoint_payload(checkpoint_path: str | Path) -> Mapping[str, Any]:
    path = Path(checkpoint_path)
    try:
        payload = torch.load(path, map_location=DEVICE_CPU, weights_only=True)
    except Exception as exc:
        message = str(exc).lower()
        if isinstance(exc, pickle.UnpicklingError) or "weights only load failed" in message or "unsupported global" in message:
            raise ValueError(f"unsupported serialized objects: {path}") from exc
        raise
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload must be a Mapping, got {type(payload).__name__}: {path}")
    return payload


def resolve_explicit_runtime_device(
    cfg: Mapping[str, Any] | None,
    *,
    cuda_usable_fn=None,
) -> torch.device:
    if not isinstance(cfg, Mapping):
        return torch.device(DEVICE_CPU)
    raw_request = cfg.get("inference_device")
    if raw_request is None:
        return torch.device(DEVICE_CPU)
    request = str(raw_request).strip().lower()
    if not request or request == DEVICE_CPU:
        return torch.device(DEVICE_CPU)
    if request == DEVICE_CUDA or request.startswith(DEVICE_CUDA_PREFIX):
        cuda_check = cuda_usable_fn or cuda_runtime_usable
        if torch.cuda.is_available() and cuda_check():
            return torch.device(request)
        return torch.device(DEVICE_CPU)
    if request != DEVICE_AUTO:
        raise ValueError(
            f"inference_device must be one of 'cpu', 'cuda', 'cuda:N', or 'auto'; got {raw_request!r}"
        )
    return torch.device(DEVICE_CPU)


def move_modules_to_device(modules: tuple[Any, ...], device: torch.device) -> None:
    for module in modules:
        if module is not None and hasattr(module, "to"):
            module.to(device)


def build_module_runtime_resource_meta(
    module: torch.nn.Module | tuple[torch.nn.Module | None, ...],
) -> dict[str, float]:
    """Calculate parameter and estimated peak-memory metadata for modules."""
    modules = (module,) if isinstance(module, torch.nn.Module) else tuple(item for item in module if item is not None)
    params = float(sum(int(parameter.numel()) for item in modules for parameter in item.parameters()))
    ram_peak_mb = max(RAM_PEAK_FLOOR_MB, params / RAM_PEAK_PARAMS_PER_MB)
    return {"params": params, "ram_peak": ram_peak_mb, "ram_peak_mb": ram_peak_mb}


def merge_checkpoint_model_cfg(base_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    merged_cfg = dict(base_cfg)
    for key in ("feature_order", "window"):
        if key in model_cfg:
            merged_cfg[key] = copy.deepcopy(model_cfg[key])
    if "network" in model_cfg:
        checkpoint_network_cfg = model_cfg["network"]
        if not isinstance(checkpoint_network_cfg, Mapping):
            raise TypeError(
                "checkpoint model_cfg.network must be a mapping when provided; "
                f"got {type(checkpoint_network_cfg).__name__}"
            )
        network_cfg = dict(merged_cfg.get("network") or {})
        network_cfg.update(copy.deepcopy(checkpoint_network_cfg))
        merged_cfg["network"] = network_cfg
    for key in ("hidden_dim", "input_dim", "output_heads"):
        if key in model_cfg:
            merged_cfg.setdefault("network", {})[key] = copy.deepcopy(model_cfg[key])
    return merged_cfg


def resolve_liquid_network_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw_network_cfg = cfg.get("network") or {}
    if not isinstance(raw_network_cfg, Mapping):
        raise TypeError(f"cfg.network must be a mapping when provided, got {type(raw_network_cfg).__name__}")
    resolved_cfg = dict(raw_network_cfg)
    hidden_dim = resolved_cfg.get("hidden_dim", 44)
    if not is_integer(hidden_dim):
        raise TypeError(f"cfg.network.hidden_dim must be an integer, got {type(hidden_dim).__name__}: {hidden_dim!r}")
    if hidden_dim < 1:
        raise ValueError(f"cfg.network.hidden_dim must be positive, got {hidden_dim}")
    input_dim = resolved_cfg.get("input_dim")
    if input_dim is not None:
        if not is_integer(input_dim):
            raise TypeError(f"cfg.network.input_dim must be an integer when provided, got {type(input_dim).__name__}: {input_dim!r}")
        if input_dim < 1:
            input_dim = None
    elif cfg.get("feature_order"):
        input_dim = len(list(cfg.get("feature_order") or []))
    output_heads = list(resolved_cfg.get("output_heads") or MODEL_INTERMEDIATE_KEYS)
    if output_heads != list(MODEL_INTERMEDIATE_KEYS):
        raise ValueError(
            "cfg.network.output_heads must keep the fixed order "
            f"{list(MODEL_INTERMEDIATE_KEYS)!r}, got {output_heads!r}"
        )
    resolved_cfg.update(hidden_dim=hidden_dim, input_dim=input_dim, output_heads=output_heads)
    if cfg.get("feature_order"):
        resolved_cfg["feature_order"] = list(cfg.get("feature_order") or [])
    return resolved_cfg


__all__ = (
    "build_module_runtime_resource_meta",
    "load_checkpoint_payload",
    "merge_checkpoint_model_cfg",
    "move_modules_to_device",
    "resolve_checkpoint_path",
    "resolve_explicit_runtime_device",
    "resolve_liquid_network_cfg",
)
