"""Liquid 模型输出头与风险校准模块 —— 向后兼容重导出层。

本模块原先包含 LiquidOutputHead / RiskCalibration 及相关常量与函数的实现。
为对齐 docs/liquid_architecture_current.md §6（"实现位于 model_factory.py 中的 _LiquidOutputHead"），
全部实现已迁移至 liquidloc.factories.model_factory。
本文件保留为向后兼容重导出层，确保现有 `from liquidloc.models.liquid.output_head import ...` 语句继续可用。
"""

from liquidloc.factories.model_factory import (  # noqa: F401 — 重导出供下游引用。
    LIQUID_CONTEXT_DIM,
    LIQUID_CONTEXT_FEATURE_KEYS,
    LIQUID_FILTER_CONTEXT_DIM,
    LIQUID_MODALITY_CONTEXT_INDICES,
    LIQUID_OBSERVATION_CONTEXT_FEATURE_KEYS,
    LIQUID_READOUT_CONTEXT_KEYS,
    LIQUID_TEMPORAL_CONTEXT_FEATURE_KEYS,
    LIQUID_UWB_BRANCH_CONTEXT_FEATURE_KEYS,
    LIQUID_UWB_FAST_CONTEXT_FEATURE_KEYS,
    LIQUID_UWB_FILTER_CONTEXT_KEYS,
    LIQUID_VIO_BRANCH_CONTEXT_FEATURE_KEYS,
    LIQUID_VIO_FAST_CONTEXT_FEATURE_KEYS,
    LIQUID_VIO_FILTER_CONTEXT_KEYS,
    SOFTPLUS_ONE_INVERSE,
    _SCALING_NEUTRAL_FLOOR,
    _VALID_MODALITIES,
    _apply_context_mask,
    _build_index_mask,
    _build_liquid_context_tensor,
    _build_liquid_filter_context_tensor,
    _liquid_context_indices_for_features,
    _liquid_filter_context_indices_for_keys,
    _slice_context_vector,
    apply_liquid_modality_output_contract,
    coerce_supported_modality,
)
from liquidloc.factories.model_factory import LiquidOutputHead, RiskCalibration  # noqa: F401 — 重导出类供下游引用。

__all__ = [
    "LIQUID_CONTEXT_DIM",
    "LIQUID_CONTEXT_FEATURE_KEYS",
    "LIQUID_FILTER_CONTEXT_DIM",
    "LIQUID_MODALITY_CONTEXT_INDICES",
    "LIQUID_OBSERVATION_CONTEXT_FEATURE_KEYS",
    "LIQUID_READOUT_CONTEXT_KEYS",
    "LIQUID_TEMPORAL_CONTEXT_FEATURE_KEYS",
    "LIQUID_UWB_BRANCH_CONTEXT_FEATURE_KEYS",
    "LIQUID_UWB_FAST_CONTEXT_FEATURE_KEYS",
    "LIQUID_UWB_FILTER_CONTEXT_KEYS",
    "LIQUID_VIO_BRANCH_CONTEXT_FEATURE_KEYS",
    "LIQUID_VIO_FAST_CONTEXT_FEATURE_KEYS",
    "LIQUID_VIO_FILTER_CONTEXT_KEYS",
    "SOFTPLUS_ONE_INVERSE",
    "LiquidOutputHead",
    "RiskCalibration",
    "apply_liquid_modality_output_contract",
    "coerce_supported_modality",
]


if __name__ == "__main__":  # 仅在直接运行时打印重导出常量，避免 import 侧效应。
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "LIQUID_CONTEXT_DIM": LIQUID_CONTEXT_DIM,
        "LIQUID_CONTEXT_FEATURE_KEYS": LIQUID_CONTEXT_FEATURE_KEYS,
        "LIQUID_FILTER_CONTEXT_DIM": LIQUID_FILTER_CONTEXT_DIM,
        "LIQUID_MODALITY_CONTEXT_INDICES": LIQUID_MODALITY_CONTEXT_INDICES,
        "LIQUID_OBSERVATION_CONTEXT_FEATURE_KEYS": LIQUID_OBSERVATION_CONTEXT_FEATURE_KEYS,
        "LIQUID_READOUT_CONTEXT_KEYS": LIQUID_READOUT_CONTEXT_KEYS,
        "LIQUID_TEMPORAL_CONTEXT_FEATURE_KEYS": LIQUID_TEMPORAL_CONTEXT_FEATURE_KEYS,
        "LIQUID_UWB_BRANCH_CONTEXT_FEATURE_KEYS": LIQUID_UWB_BRANCH_CONTEXT_FEATURE_KEYS,
        "LIQUID_UWB_FAST_CONTEXT_FEATURE_KEYS": LIQUID_UWB_FAST_CONTEXT_FEATURE_KEYS,
        "LIQUID_UWB_FILTER_CONTEXT_KEYS": LIQUID_UWB_FILTER_CONTEXT_KEYS,
        "LIQUID_VIO_BRANCH_CONTEXT_FEATURE_KEYS": LIQUID_VIO_BRANCH_CONTEXT_FEATURE_KEYS,
        "LIQUID_VIO_FAST_CONTEXT_FEATURE_KEYS": LIQUID_VIO_FAST_CONTEXT_FEATURE_KEYS,
        "LIQUID_VIO_FILTER_CONTEXT_KEYS": LIQUID_VIO_FILTER_CONTEXT_KEYS,
        "SOFTPLUS_ONE_INVERSE": SOFTPLUS_ONE_INVERSE,
    }, "output_head 重导出常量")
