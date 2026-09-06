"""Standalone matrix-level feature normalization helpers.

These utilities normalize dense feature matrices or vectors with shared
mean/std statistics. They are kept as reusable helpers, but they are not the
current structured-window main path used by the paper-grade training and
inference pipeline.
"""

from __future__ import annotations

import math
import numpy as np

from liquidloc.common.validation import coerce_finite_scalar


def neutral_floor_softplus(value, *, neutral_floor: float | None = None, scaling_max: float | None = None):
    """Canonical neutral-floor softplus for scaling factors.

    Computes ``neutral_floor + softplus(x) - softplus(0)``, then clamps
    the result to ``[neutral_floor, scaling_max]``.  When *x* = 0 the
    output equals *neutral_floor* (defaults to ``1.0``), so the scaling factor is neutral
    (no scaling).  The output is always >= *neutral_floor*, guaranteeing a
    positive scaling factor.

    Supports both Python floats and PyTorch tensors.  For tensors the
    straight-through gradient estimator is applied so that gradients flow
    through the softplus path while the clamped value is used in the
    forward pass.

    Args:
        value: Raw scaling logit — a Python float, NumPy scalar, or
            PyTorch tensor.
        neutral_floor: Lower bound for the output.  When *None*, uses ``1.0``
            (non-current modality outputs no scaling).
        scaling_max: Upper bound.  When *None*, reads
            ``BRIDGE_THRESHOLDS["scaling_max"]``.

    Returns:
        A Python float (when *value* is float-like) or a PyTorch tensor
        (when *value* is a tensor), clamped to
        ``[neutral_floor, scaling_max]``.
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "value_type": type(value).__name__,
        "neutral_floor": neutral_floor,
        "scaling_max": scaling_max,
    }, "neutral_floor_softplus 入口参数")
    if neutral_floor is None or scaling_max is None:
        from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
        # v2：解耦 neutral_floor 与 BRIDGE_THRESHOLDS["scaling_min"]。
        # - BRIDGE_SCALING_MIN=1.0 (v3 回退) 是 MeasurementControl/ModelIntermediate 的下界校验。
        # - neutral_floor_softplus 的 neutral_floor 是"非当前模态静默缩放"的语义下界，
        #   必须严格 = 1.0（neutral，不参与估计 = 完全不缩放）。
        # 若两者耦合且 BRIDGE_SCALING_MIN<1.0，非当前模态 scaling 会被错降到 BRIDGE_SCALING_MIN，破坏模态输出合约。
        if neutral_floor is None:
            neutral_floor = 1.0
        if scaling_max is None:
            scaling_max = float(BRIDGE_THRESHOLDS["scaling_max"])

    softplus_zero = math.log(2.0)

    # --- PyTorch tensor path (training / model_factory) ---
    try:
        import torch
        if torch.is_tensor(value):
            if not torch.isfinite(value).all():  # D5：拦截 NaN/Inf，防止经 softplus + torch.clamp 静默穿透（torch.clamp 不替换 NaN，会原样返回 NaN 并污染训练 loss）。
                raise ValueError(f"scaling value must be finite, got {value}")
            from torch.nn import functional as F
            shifted = neutral_floor + F.softplus(value) - softplus_zero
            clamped = torch.clamp(shifted, min=neutral_floor, max=scaling_max)
            return shifted + (clamped - shifted).detach()
    except ImportError:
        pass

    # --- Float path (inference) ---
    fv = coerce_finite_scalar(value, name="scaling value")  # D5：拦截 NaN/Inf 标量，与张量路径口径一致；Python max/min 对 NaN 行为不确定，必须前置拦截。
    shifted = neutral_floor + float(np.logaddexp(0.0, fv)) - softplus_zero
    result = max(neutral_floor, shifted)
    return min(scaling_max, result)


def fit_norm_stats(feature_matrix):
    """Fit shared mean/std statistics from a 1D or 2D feature array."""
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "feature_matrix_type": type(feature_matrix).__name__,
        "feature_matrix_len": len(feature_matrix) if hasattr(feature_matrix, "__len__") else None,
    }, "fit_norm_stats 入口参数")
    if feature_matrix is None:
        raise ValueError("feature_matrix must not be None")

    feature_matrix = np.asarray(feature_matrix, dtype=float)
    if feature_matrix.ndim == 0 or feature_matrix.size == 0:
        raise ValueError("feature_matrix must be a non-empty 1D or 2D array")
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(1, -1)
    elif feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be a 1D or 2D array")
    if not np.isfinite(feature_matrix).all():
        raise ValueError("feature_matrix contains non-finite values (NaN or Inf)")

    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return {"mean": mean, "std": std}


def _prepare_norm_inputs(feature_matrix, norm_stats):
    """Validate and prepare inputs shared by normalize/denormalize.

    Returns ``(feature_matrix, mean, std)`` where *std* has zero entries
    replaced by 1.0 so that the normalize/denormalize pair stays exact
    inverses even for constant feature columns.  Centralising this logic
    here guarantees the two public functions remain perfectly symmetric
    (DRY): any validation fix applies to both paths at once.
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "feature_matrix_type": type(feature_matrix).__name__,
        "feature_matrix_len": len(feature_matrix) if hasattr(feature_matrix, "__len__") else None,
        "norm_stats_type": type(norm_stats).__name__,
    }, "_prepare_norm_inputs 入口参数")
    if feature_matrix is None:
        raise ValueError("feature_matrix must not be None")
    if not isinstance(norm_stats, dict):
        raise TypeError("norm_stats must be a dict with mean/std")
    if "mean" not in norm_stats or "std" not in norm_stats:
        raise KeyError("norm_stats must contain 'mean' and 'std'")

    feature_matrix = np.asarray(feature_matrix, dtype=float)
    if feature_matrix.ndim == 0 or feature_matrix.size == 0:
        raise ValueError("feature_matrix must be a non-empty 1D or 2D array")
    if feature_matrix.ndim not in (1, 2):
        raise ValueError("feature_matrix must be a 1D or 2D array")
    if not np.all(np.isfinite(feature_matrix)):
        raise ValueError("feature_matrix must contain only finite values (no NaN or Inf)")

    mean = np.asarray(norm_stats["mean"], dtype=float).ravel()
    std = np.asarray(norm_stats["std"], dtype=float).ravel()
    if not np.all(np.isfinite(mean)):
        raise ValueError("norm_stats mean must contain only finite values (no NaN or Inf)")
    if not np.all(np.isfinite(std)):
        raise ValueError("norm_stats std must contain only finite values (no NaN or Inf)")
    if np.any(std < 0):
        raise ValueError("std must be non-negative")
    if mean.shape != std.shape:
        raise ValueError("norm_stats mean/std must have the same shape")

    feature_width = feature_matrix.shape[-1]
    if mean.shape[0] != feature_width:
        raise ValueError("norm_stats width must match feature_matrix width")

    std = np.where(std == 0.0, 1.0, std)
    return feature_matrix, mean, std


def normalize_features(feature_matrix, norm_stats):
    """Normalize a 1D or 2D feature array with pre-fit shared stats."""
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "feature_matrix_type": type(feature_matrix).__name__,
        "feature_matrix_len": len(feature_matrix) if hasattr(feature_matrix, "__len__") else None,
        "norm_stats_type": type(norm_stats).__name__,
    }, "normalize_features 入口参数")
    feature_matrix, mean, std = _prepare_norm_inputs(feature_matrix, norm_stats)
    return (feature_matrix - mean) / std


def denormalize_features(feature_matrix, norm_stats):
    """Restore a normalized 1D or 2D feature array to the original scale."""
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "feature_matrix_type": type(feature_matrix).__name__,
        "feature_matrix_len": len(feature_matrix) if hasattr(feature_matrix, "__len__") else None,
        "norm_stats_type": type(norm_stats).__name__,
    }, "denormalize_features 入口参数")
    feature_matrix, mean, std = _prepare_norm_inputs(feature_matrix, norm_stats)
    return feature_matrix * std + mean
