"""估计器共享工具函数

职责：
提供被多个估计器实现共同使用的协方差控制和报告转换函数，
避免跨模块导入私有函数。

上游依赖：
- liquidloc.common.covariance_utils — 协方差缩放核心实现
- liquidloc.common.types — MeasurementControl 类型
- liquidloc.common.validation — is_bool_like 等校验工具

下游调用者：
- liquidloc.estimators.ekf_core — EKF 核心估计器
- liquidloc.estimators.fgo_core — FGO 核心估计器
- liquidloc.estimators.robust_ekf_core — 鲁棒 EKF 核心估计器
"""

from __future__ import annotations

from typing import Any

import numpy as np

from liquidloc.common.covariance_utils import build_effective_cov
from liquidloc.common.types import MeasurementControl

__all__ = ("build_controlled_measurement_cov", "control_to_dict")


def control_to_dict(control: MeasurementControl | None) -> dict[str, Any] | None:
    """把控制对象转成普通字典，便于写入报告。

    参数
    ----------
    control : MeasurementControl | None
    待转换的测量控制对象；为 None 时直接返回 None。

    返回
    -------
    dict[str, Any] | None
    包含 modality、bias_applied、scaling、risk、noise_multiplier、
    gate_action 六个字段的字典，或 None。
    """
    if control is None:
        return None
    return {
        "modality": control.modality,
        "bias_applied": float(control.bias_applied),
        "scaling": float(control.scaling),
        "risk": float(control.risk),
        "noise_multiplier": float(control.noise_multiplier),
        "gate_action": control.gate_action,
    }


def build_controlled_measurement_cov(
    base_cov: Any,
    control: MeasurementControl,
    *,
    modality: str,
    calibration_frozen: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """按 measurement_control 对基础协方差做缩放，并生成报告。

    参数
    ----------
    base_cov : Any
        基础测量噪声协方差，可以是标量、向量或矩阵。
    control : MeasurementControl
        测量控制对象，提供 noise_multiplier、scaling、risk、gate_action。
    modality : str
        量测模态，目前仅支持 ``"uwb"`` 和 ``"vio"``。
    calibration_frozen : bool
        **前提指导 §2.2 标定冻结约束**。为 ``True`` 时，标定视距后的基础协方差不得被
        LNN 的 ``noise_multiplier`` 再缩放——无论模型输出何种缩放因子，有效协方差恒等
        于 ``base_cov``。这确保 R/Q 在标定后不可被在线微调破坏。

    返回
    -------
    tuple[Any, dict[str, Any]]
        ``(effective_cov, cov_report)`` —— 缩放后的有效协方差和协方差报告字典。

    报告字典额外包含 bridge_scaling、bridge_risk、noise_multiplier、
    gate_action、bridge_semantics_authority 字段。

    异常
    ------
    ValueError
    当 noise_multiplier 非有限或非正、或 modality 不受支持时抛出。
    """
    from liquidloc.common.tee_logger import print_dict  # 局部导入参数打印工具。
    if calibration_frozen:
        # §2.2 标定冻结：R/Q 已通过标定确定，LNN 输出的 noise_multiplier 被强制忽略。
        # 无论模型输出何种缩放因子，有效协方差恒等于标定后的基础值。
        cov_report = dict(
            bridge_scaling=float(control.scaling),
            bridge_risk=float(control.risk),
            noise_multiplier=1.0,
            gate_action=control.gate_action,
            bridge_semantics_authority="calibration_freeze",
            calibration_frozen=True,
            raw_noise_multiplier=float(control.noise_multiplier),
        )
        return base_cov, cov_report

    print_dict({"base_cov": base_cov, "control": control, "modality": modality}, "build_controlled_measurement_cov 入参", prefix="[配置]")
    noise_multiplier = float(control.noise_multiplier)
    if not np.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError("measurement_control.noise_multiplier must be finite and > 0")
    if modality == "uwb":
        effective_cov, cov_report = build_effective_cov(base_cov, uwb_scaling=noise_multiplier)
    elif modality == "vio":
        effective_cov, cov_report = build_effective_cov(base_cov, vio_scaling=noise_multiplier)
    else:
        raise ValueError(f"unsupported modality for controlled covariance: {modality}")
    cov_report = dict(cov_report)
    cov_report["bridge_scaling"] = float(control.scaling)
    cov_report["bridge_risk"] = float(control.risk)
    cov_report["noise_multiplier"] = noise_multiplier
    cov_report["gate_action"] = control.gate_action
    cov_report["bridge_semantics_authority"] = "noise_multiplier"
    return effective_cov, cov_report
