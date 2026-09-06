"""协方差缩放工具。

职责：
1. 将模型输出的 ``uwb_scaling`` / ``vio_scaling`` 映射到融合更新实际使用的协方差。
2. 支持 mapping / array / scalar 等多种协方差结构，并按通道语义做缩放。
3. 递归校验协方差容器中的每个叶子值都是真实、有限的实数。
4. 返回缩放后的协方差以及诊断报告。

注意：
- ``risk`` 语义已由 ``liquid_bridge_contract._compose_noise_multiplier`` 吸收为
  ``scaling^2 * (1 + risk)``，本模块只消费最终 ``noise_multiplier`` 对应的缩放效果。
- 本模块原位于 fusion 桥接层，已下沉至 common 层以消除估计器对融合层的反向依赖。
  fusion/covariance_adapter.py 保留为重导出兼容层。
- 本模块只负责"缩放意图 -> 协方差数值"的映射，不改写估计器语义。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np

from liquidloc.common.constants import VIO_MEASUREMENT_ITEMS  # VIO 测量项常量，避免反向依赖 protocol 层。
from liquidloc.common.validation import is_bool_like, is_real, require_not_none

__all__ = ("build_effective_cov",)

_UWB_COV_KEYS = ("uwb", "uwb_cov", "R_range", "range")
_VIO_COV_KEYS = ("vio", "vio_cov", "R_vio")
_VIO_POS_YAW_KEYS = ("pos", "yaw")
_VIO_DXY_DYAW_KEYS = VIO_MEASUREMENT_ITEMS  # VIO 测量项键名来自 common 层冻结常量，而非协议层运行时调用。


def _coerce_scalar(value, *, name):
    """把单个真实数值叶子规整成 float。"""
    if is_bool_like(value):
        raise TypeError(f"{name} must be numeric")
    scalar = np.asarray(value)
    if scalar.size != 1:
        raise ValueError(f"{name} must be a scalar")
    scalar_value = scalar.reshape(-1)[0]
    if not is_real(scalar_value):
        raise TypeError(f"{name} must be numeric")
    return float(scalar_value)


def _coerce_scaling(value, *, name):
    """把缩放因子规整成有限正浮点数。"""
    if value is None:
        return None
    scaling = _coerce_scalar(value, name=name)
    if not np.isfinite(scaling) or scaling <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return scaling


def _looks_like_vio_cov_mapping(base_cov):
    """判断 mapping 是否看起来像 VIO 协方差结构。"""
    return all(key in base_cov for key in _VIO_POS_YAW_KEYS) or all(
        key in base_cov for key in _VIO_DXY_DYAW_KEYS
    )


def _scale_cov_like(cov_value, multiplier):
    """递归按同一倍数缩放协方差结构，并保持容器类型。"""
    if multiplier == 1.0:
        return copy.deepcopy(cov_value)
    if isinstance(cov_value, Mapping):
        return {key: _scale_cov_like(value, multiplier) for key, value in cov_value.items()}
    if isinstance(cov_value, np.ndarray):
        if cov_value.dtype.kind == "b":
            raise TypeError("base_cov must be numeric")
        return cov_value.astype(float, copy=True) * multiplier
    if isinstance(cov_value, list):
        return [_scale_cov_like(value, multiplier) for value in cov_value]
    if isinstance(cov_value, tuple):
        return tuple(_scale_cov_like(value, multiplier) for value in cov_value)
    if isinstance(cov_value, (str, bytes)):
        raise TypeError("base_cov must be numeric")

    cov_array = np.asarray(cov_value, dtype=float)
    if cov_array.ndim == 0:
        return float(cov_array) * multiplier
    return cov_array * multiplier


def _validate_numeric_cov(value, *, name):
    """递归检查协方差叶子值是否都是真实、有限的实数。"""
    if is_bool_like(value):
        raise TypeError(f"{name} must be numeric")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_numeric_cov(item, name=f"{name}.{key}")
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            raise TypeError(f"{name} must be numeric")
        if value.dtype.kind in {"U", "S", "c"}:
            raise TypeError(f"{name} must be numeric")
        if value.dtype.kind == "O":
            if value.ndim == 0:
                _validate_numeric_cov(value.item(), name=name)
                return
            for index, item in np.ndenumerate(value):
                _validate_numeric_cov(item, name=f"{name}[{index}]")
            return
        if not np.all(np.isfinite(value.astype(float, copy=False))):
            raise ValueError(f"{name} must contain finite values")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_numeric_cov(item, name=f"{name}[{index}]")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_numeric_cov(item, name=f"{name}[{index}]")
        return
    if not is_real(value):
        raise TypeError(f"{name} must be numeric")
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must contain finite values")


def _apply_named_scaling(effective_cov, *, uwb_scaling, vio_scaling):
    """按 UWB / VIO 通道语义应用缩放。"""
    if isinstance(effective_cov, Mapping):
        scaled_cov = dict(effective_cov)
        for key in _UWB_COV_KEYS:
            if key in scaled_cov and uwb_scaling is not None:
                scaled_cov[key] = _scale_cov_like(scaled_cov[key], uwb_scaling)
        if _looks_like_vio_cov_mapping(scaled_cov):
            if vio_scaling is not None:
                # 保存已缩放的 UWB 键值，防止被 VIO 整体缩放二次放大
                saved_uwb = {key: scaled_cov[key] for key in _UWB_COV_KEYS if key in scaled_cov}
                scaled_cov = _scale_cov_like(scaled_cov, vio_scaling)
                # 恢复 UWB 键为仅 uwb_scaling 缩放后的值
                for key, value in saved_uwb.items():
                    scaled_cov[key] = value
                return scaled_cov
            return scaled_cov
        for key in _VIO_COV_KEYS:
            if key in scaled_cov and vio_scaling is not None:
                scaled_cov[key] = _scale_cov_like(scaled_cov[key], vio_scaling)
        return scaled_cov

    if isinstance(effective_cov, (list, tuple)):
        if uwb_scaling is not None and vio_scaling is not None and uwb_scaling != vio_scaling:
            raise ValueError("sequence base_cov cannot distinguish uwb_scaling from vio_scaling")
        scaling = uwb_scaling if uwb_scaling is not None else vio_scaling
        if scaling is None:
            return copy.deepcopy(effective_cov)
        return _scale_cov_like(effective_cov, scaling)

    cov_array = np.asarray(effective_cov, dtype=float)
    if cov_array.ndim == 0:
        if uwb_scaling is not None and vio_scaling is not None and uwb_scaling != vio_scaling:
            raise ValueError("scalar base_cov cannot apply distinct uwb_scaling and vio_scaling")
        scaling = uwb_scaling if uwb_scaling is not None else vio_scaling
        if scaling is None:
            return copy.deepcopy(effective_cov)
        return _scale_cov_like(effective_cov, scaling)

    if cov_array.shape in {(3,), (3, 3)}:
        if vio_scaling is None:
            return copy.deepcopy(effective_cov)
        return _scale_cov_like(effective_cov, vio_scaling)

    if uwb_scaling is not None and vio_scaling is not None and uwb_scaling != vio_scaling:
        raise ValueError("base_cov shape cannot distinguish uwb_scaling from vio_scaling")
    scaling = uwb_scaling if uwb_scaling is not None else vio_scaling
    if scaling is None:
        return copy.deepcopy(effective_cov)
    return _scale_cov_like(effective_cov, scaling)


def _to_report_value(value):
    """深拷贝报告值，避免和调用方共享引用。"""
    return copy.deepcopy(value)


def build_effective_cov(base_cov, uwb_scaling=None, vio_scaling=None):
    """构造融合更新时真正使用的协方差。"""
    require_not_none(base_cov, "base_cov")
    _validate_numeric_cov(base_cov, name="base_cov")

    effective_cov = copy.deepcopy(base_cov)
    uwb_scaling = _coerce_scaling(uwb_scaling, name="uwb_scaling")
    vio_scaling = _coerce_scaling(vio_scaling, name="vio_scaling")

    effective_cov = _apply_named_scaling(
        effective_cov,
        uwb_scaling=uwb_scaling,
        vio_scaling=vio_scaling,
    )

    cov_report = {
        "base_cov": _to_report_value(base_cov),
        "uwb_scaling": uwb_scaling,
        "vio_scaling": vio_scaling,
        "effective_cov": _to_report_value(effective_cov),
    }
    return effective_cov, cov_report
