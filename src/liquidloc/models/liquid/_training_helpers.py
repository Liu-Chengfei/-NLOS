"""Training helpers submodule for Liquid trainer.

Handles phase scheduling, optimizer building, trainable parameter management,
phase control state, device resolution, and the main training loop.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from ._constants import (  # type: ignore[import]
    _ACTIVE_HEADS_BY_MODALITY,
    _BATCH_DIAGNOSTIC_PROBE_LIMIT,
    _BIAS_HUBER_DELTA,
    _CROSS_TRAINER_PARITY_DECLARATION,
    _EPOCH_ARTIFACT_BATCH_LIMIT,
    _EPOCH_FIXED_PROBE_LIMIT,
    _ENABLE_SMOOTH_REGULARIZATION,
    _FULL_TUNING_AUX_WARMUP_EPOCHS,
    _FULL_TUNING_ENTRY_BRIDGE_EPOCHS,
    _FULL_TUNING_ENTRY_BRIDGE_MIN_FRACTION,
    _FULL_TUNING_ENTRY_BUFFER_MULTIPLIER,
    _FULL_TUNING_LATE_CONSOLIDATION_EPOCHS,
    _FULL_TUNING_LATE_CONSOLIDATION_MIN_FRACTION,
    _FULL_TUNING_MIN_JOINT_DRIVE_EPOCHS,
    _FULL_TUNING_REENTRY_AUX_WARMUP_EPOCHS,
    _FULL_TUNING_REENTRY_EXTRA_BUFFER_EPOCHS,
    _FULL_TUNING_REENTRY_FILTER_GATE_LR_CAP_SCALE,
    _FULL_TUNING_REENTRY_GATE_CAL_LR_CAP_SCALE,
    _FULL_TUNING_REENTRY_LR_RELEASE_EPOCHS,
    _FULL_TUNING_REENTRY_READOUT_LR_CAP_SCALE,
    _FULL_TUNING_REENTRY_REPEAT_CAP_MULTIPLIER,
    _FULL_TUNING_REENTRY_UPSTREAM_CAP_MULTIPLIER,
    _FULL_TUNING_REPEAT_ROLLBACK_THRESHOLD,
    _FULL_TUNING_UPSTREAM_FALLBACK_EXTRA_BUFFER_EPOCHS,
    _GATE_ALIGNMENT_AUX_WARMUP_EPOCHS,
    _GATE_ALIGNMENT_BACKBONE_LR_SCALE,
    _GATE_ALIGNMENT_READOUT_LR_SCALE,
    _GLOBAL_L2_WEIGHT,
    _LIQUID_CALIBRATION_WEIGHT,
    _LIQUID_GATE_L1_WEIGHT,
    _LIQUID_MONO_WEIGHT,
    _OUTPUT_KEYS,
    _PHASE_TRANSITION_OVERLAP_EPOCHS,
    _RISK_CALIBRATION_BIN_COUNT,
    _SELECTION_WEIGHTS,
    _SCALING_LOG_EPS,
    _SCALING_NEUTRAL_FLOOR,
    _SOFT_CONTROL_BUFFER_EPOCHS,
    _SOFT_CONTROL_COOLDOWN_EPOCHS,
    _SOFT_CONTROL_LR_DECAY,
    _SOFT_CONTROL_LR_FLOOR_SCALE,
    _SOFT_CONTROL_LR_RECOVERY_SCALE,
    _SOFT_CONTROL_ROLLBACK_PATIENCE,
    _SOFT_CONTROL_SHOCK_THRESHOLD,
    _SOFT_CONTROL_STREAK_LR_DECAY,
    _SOFT_CONTROL_STREAK_LENGTH,
    _TAIL_SELECTION_MAX_WEIGHT,
    _TAIL_SELECTION_OBSERVATION_COEFF,
    _TAIL_SELECTION_TAIL_COEFF,
)
from liquidloc.common.seed_utils import cuda_runtime_usable as _check_cuda_runtime_usable, set_global_seed
from liquidloc.common.constants import (
    BRIDGE_BIAS_MAX,
    MODALITY_UWB,
    MODALITY_VIO,
    RISK_LABEL_MODE_DEFAULT,
    UWB_SCALING_LABEL_MODE_DEFAULT,
    VIO_SCALING_LABEL_MODE_DEFAULT,
    DEVICE_AUTO,
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_CUDA_PREFIX,
)
from liquidloc.common.validation import coerce_finite_scalar, is_integer
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
from liquidloc.models.features.normalization import neutral_floor_softplus

# Deferred imports for cross-module dependencies (avoid circular imports)
def _get_data_utils():
    from liquidloc.models.liquid._data_utils import (
        _move_materialized_samples,
        _materialize_samples,
        _resolve_sample_cache_device,
    )
    return _move_materialized_samples, _materialize_samples, _resolve_sample_cache_device

def _get_checkpoint_utils():
    from liquidloc.models.liquid._checkpoint_utils import _safe_finite_float
    return _safe_finite_float

def _get_loss_compute():
    from liquidloc.models.liquid._loss_compute import (
        _risk_calibration_trainable,
        _loss_for_length_group,
    )
    return _risk_calibration_trainable, _loss_for_length_group

def _resolve_full_tuning_subphase(
    trainer_state: Mapping[str, Any],
    phase_state_runtime: Mapping[str, Any] | None = None,
) -> str:
    """Resolve the explicit full_tuning subphase from current phase-local runtime."""
    # 铁律 6 (2026-07-23 N2 中和): 让 train.disable_full_tuning_subphase_scheduler
    # 能短路子阶段调度器, 永远返回 "joint_drive" (LR=1.0 全程, 与 LSTM 平坦 LR 对齐).
    # N2 是 Liquid 独有的 full_tuning 三子阶段 (entry_bridge/joint_drive/late_consolidation)
    # 分组 LR 系数调度; LSTM trainer 全程平坦 LR, 这里短路以中和训练不公平.
    try:
        disable_full_tuning_subphase_scheduler = bool(
            trainer_state.get("disable_full_tuning_subphase_scheduler", False)
        )
    except Exception:
        disable_full_tuning_subphase_scheduler = False
    if disable_full_tuning_subphase_scheduler:
        return "joint_drive"
    runtime_state = dict(phase_state_runtime or {})
    phase_epoch_count = max(0, int(runtime_state.get("phase_epoch_count") or 0))
    phase_budget = dict(trainer_state.get("phase_schedule_budget") or {})
    total_full_tuning_epochs = max(0, int(phase_budget.get("full_tuning_epochs") or 0))
    if total_full_tuning_epochs <= 0:
        return "joint_drive"
    entry_bridge_epochs = min(
        _FULL_TUNING_ENTRY_BRIDGE_EPOCHS,
        max(1, int(math.ceil(total_full_tuning_epochs * _FULL_TUNING_ENTRY_BRIDGE_MIN_FRACTION))),
    )
    remaining_after_entry = max(0, total_full_tuning_epochs - entry_bridge_epochs)
    late_window = min(
        _FULL_TUNING_LATE_CONSOLIDATION_EPOCHS,
        max(1, int(math.ceil(total_full_tuning_epochs * _FULL_TUNING_LATE_CONSOLIDATION_MIN_FRACTION))),
    )
    max_late_window = max(1, remaining_after_entry - _FULL_TUNING_MIN_JOINT_DRIVE_EPOCHS)
    late_window = min(late_window, max_late_window)
    if late_window > 0 and phase_epoch_count >= max(0, total_full_tuning_epochs - late_window):
        return "late_consolidation"
    if phase_epoch_count < entry_bridge_epochs:
        return "entry_bridge"
    return "joint_drive"


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b with t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


def _compute_overfit_audit(train_losses: list[float], val_losses: list[float]) -> dict[str, Any]:
    """准则 34 静态监控：分析 train/val loss 曲线判断过拟合风险。

    关键诊断：
    - 过拟合触发条件：连续 N 轮 val_loss 上升且 train_loss 仍下降。
    - 报告最终 epoch 的 val/train 比例（gap > 1.2 视为风险）。
    - 输出"最佳 epoch" = min val_loss 对应位置（早停/选择依据）。

    Args:
        train_losses: 每 epoch 训练集损失列表。
        val_losses: 每 epoch 验证集损失列表（与 train_losses 等长）。

    Returns:
        dict：包含 best_epoch, final_gap, consecutive_val_rise_count, overfit_risk
        字段，JSON 序列化安全（inf/NaN 替换为 None）。
    """
    if not train_losses or not val_losses or len(train_losses) != len(val_losses):
        return {
            "best_epoch": -1,
            "final_gap": None,
            "consecutive_val_rise_count": 0,
            "overfit_risk": "insufficient_data",
            "early_stop_recommended": False,
            "val_train_ratio_final": None,
        }
    best_epoch = int(min(range(len(val_losses)), key=lambda i: val_losses[i]))
    final_train = float(train_losses[-1])
    final_val = float(val_losses[-1])
    val_train_ratio = final_val / max(final_train, 1e-9)
    # 连续 val_loss 上升计数（从末尾往前数）
    consecutive_rise = 0
    for i in range(len(val_losses) - 1, 0, -1):
        if val_losses[i] > val_losses[i - 1] * 1.001:  # 0.1% 容差避免浮点噪声
            consecutive_rise += 1
        else:
            break
    # 过拟合风险等级
    if val_train_ratio > 1.5 and consecutive_rise >= 3:
        risk = "high"
    elif val_train_ratio > 1.2 and consecutive_rise >= 2:
        risk = "medium"
    elif val_train_ratio > 1.05:
        risk = "low"
    else:
        risk = "minimal"
    return {
        "best_epoch": best_epoch,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "final_gap": float(final_val - final_train),
        "val_train_ratio_final": float(val_train_ratio),
        "consecutive_val_rise_count": int(consecutive_rise),
        "overfit_risk": str(risk),
        "early_stop_recommended": bool(consecutive_rise >= 5 and val_train_ratio > 1.3),
    }


def _full_tuning_subphase_group_scales(subphase_name: str, *, subphase_epoch_count: int | None = None) -> dict[str, float]:
    """Return relative LR multipliers for explicit full_tuning subphases.

    T1: phase overlap ramp. 在阶段切换的前 _PHASE_TRANSITION_OVERLAP_EPOCHS
    个 epoch 内，把 LR scale 从前一阶段的目标值线性 ramp 到当前阶段的目标值，
    避免硬切换引起的优化器重新适应震荡。

    参数:
        subphase_name: full_tuning 子阶段名 (entry_bridge / joint_drive / late_consolidation)
        subphase_epoch_count: 当前子阶段的 epoch 计数（从 0 开始）。None 时不启用 ramp。
    """
    target_scales = {
        "readout": 1.0,
        "gate_cal": 1.0,
        "filter_context_gate": 1.0,
        "backbone": 1.0,
    }
    if subphase_name == "entry_bridge":
        target_scales = {
            "readout": 0.85,
            "gate_cal": 0.55,
            "filter_context_gate": 0.45,
            "backbone": 0.35,
        }
    elif subphase_name == "late_consolidation":
        target_scales = {
            "readout": 0.80,
            "gate_cal": 0.70,
            "filter_context_gate": 0.70,
            "backbone": 0.75,
        }
        # P3 (v7 patch): late_consolidation 入 ramp, 从 joint_drive 终态 {1,1,1,1} 平滑过渡.
        # 子代 G 报告: 原 trainer.py:158-164 无 ramp, joint_drive→late_consolidation 硬跳.
        if subphase_epoch_count is not None:
            joint_drive_scales = {
                "readout": 1.0, "gate_cal": 1.0, "filter_context_gate": 1.0, "backbone": 1.0,
            }
            overlap = max(1, _PHASE_TRANSITION_OVERLAP_EPOCHS)
            t = min(1.0, subphase_epoch_count / overlap)
            return {
                group: _lerp(joint_drive_scales[group], target_scales[group], t)
                for group in target_scales
            }

    # T1: entry_bridge 入门 ramp，从 gate_alignment 终态线性过渡到 entry_bridge 目标
    # gate_alignment 终态: readout=0.25, backbone=0.35 (P1.b: 已解冻, 起点不再 0), gate_cal=1.0, filter_context_gate=1.0
    if subphase_name == "entry_bridge" and subphase_epoch_count is not None:
        prev_phase_scales = {
            "readout": _GATE_ALIGNMENT_READOUT_LR_SCALE,  # 0.25
            "gate_cal": 1.0,
            "filter_context_gate": 1.0,
            "backbone": 0.35,  # P1.b: gate_alignment 已解冻 backbone, 起点不再是 0
        }
        overlap = max(1, _PHASE_TRANSITION_OVERLAP_EPOCHS)
        t = min(1.0, subphase_epoch_count / overlap)
        return {
            group: _lerp(prev_phase_scales[group], target_scales[group], t)
            for group in target_scales
        }

    # T1: joint_drive 入门 ramp，从 entry_bridge 终态线性 ramp 到 1.0
    if subphase_name == "joint_drive" and subphase_epoch_count is not None:
        entry_bridge_scales = {
            "readout": 0.85,
            "gate_cal": 0.55,
            "filter_context_gate": 0.45,
            "backbone": 0.35,
        }
        overlap = max(1, _PHASE_TRANSITION_OVERLAP_EPOCHS)
        t = min(1.0, subphase_epoch_count / overlap)
        return {
            group: _lerp(entry_bridge_scales[group], target_scales[group], t)
            for group in target_scales
        }

    return target_scales



def _compute_full_tuning_ramp_epoch(epoch_index: int, trainer_state: Mapping[str, Any], phase_state_runtime: Mapping[str, Any] | None = None) -> int:
    """计算 full_tuning 阶段ramp epoch, 支持 rollback 后重入重置."""
    if isinstance(phase_state_runtime, Mapping):
        reentry_epoch = phase_state_runtime.get("reentry_ramp_epoch")
        if is_integer(reentry_epoch):
            return max(0, int(reentry_epoch))
    phase_schedule = trainer_state["phase_schedule"]
    warmup_epochs = int(phase_schedule["warmup_epochs"])
    gate_alignment_epochs = int(phase_schedule["gate_alignment_epochs"])
    full_tuning_start_epoch = warmup_epochs + gate_alignment_epochs + 1
    return max(0, int(epoch_index) - full_tuning_start_epoch)


def _phase_progress_scale(epoch_count: int, warmup_epochs: int) -> float:
    """Map a phase-local epoch count to a smooth [0, 1] control scale."""
    if warmup_epochs <= 0:
        return 1.0
    clamped_epoch = max(0, int(epoch_count))
    return min(1.0, float(clamped_epoch) / float(warmup_epochs))


def _build_phase_control_state(
    phase_name: str,
    trainer_state: Mapping[str, Any],
    phase_state_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build explicit phase control state for loss consumption and re-entry behavior."""
    runtime_state = dict(phase_state_runtime or {})
    phase_epoch_count = max(0, int(runtime_state.get("phase_epoch_count") or 0))
    reentry_mode = str(runtime_state.get("reentry_mode") or "normal")
    reentry_buffer_remaining = max(0, int(runtime_state.get("reentry_buffer_remaining") or 0))
    same_checkpoint_rollback_count = max(0, int(runtime_state.get("same_checkpoint_rollback_count") or 0))
    full_tuning_subphase = None
    full_tuning_subphase_entry_epoch = 0
    subphase_progress = 1.0
    group_lr_scales = {
        "readout": 1.0,
        "gate_cal": 1.0,
        "filter_context_gate": 1.0,
        "backbone": 1.0,
    }
    if phase_name == "readout_warmup":
        aux_scale = 0.0
        gate_scale = 0.0
        calibration_scale = 0.0
        regularization_scale = 1.0
    elif phase_name == "gate_alignment":
        gate_progress = _phase_progress_scale(phase_epoch_count, _GATE_ALIGNMENT_AUX_WARMUP_EPOCHS)
        aux_scale = gate_progress
        gate_scale = gate_progress
        calibration_scale = gate_progress
        regularization_scale = 1.0
    elif phase_name == "full_tuning":
        full_tuning_subphase = str(
            runtime_state.get("full_tuning_subphase")
            or runtime_state.get("full_tuning_reentry_target_subphase")
            or _resolve_full_tuning_subphase(trainer_state, runtime_state)
        )
        full_tuning_subphase_entry_epoch = max(0, int(runtime_state.get("full_tuning_subphase_entry_epoch") or 0))
        subphase_epoch_count = max(0, phase_epoch_count - full_tuning_subphase_entry_epoch)
        if full_tuning_subphase == "entry_bridge":
            subphase_progress = _phase_progress_scale(subphase_epoch_count, _FULL_TUNING_ENTRY_BRIDGE_EPOCHS)
            aux_scale = subphase_progress
            gate_scale = subphase_progress
            calibration_scale = subphase_progress
            regularization_scale = subphase_progress
        elif full_tuning_subphase == "late_consolidation":
            subphase_progress = 1.0
            # P4 (v7 patch): late_consolidation aux/gate/cal scale 从 1.0 平滑 ramp 到目标.
            # 子代 G 报告: 原 trainer.py:562-567 硬跳 {0.90, 0.85, 0.85, 0.85}, val loss spike.
            late_warmup_overlap = max(1, _PHASE_TRANSITION_OVERLAP_EPOCHS)
            t_late = min(1.0, subphase_epoch_count / late_warmup_overlap)
            aux_scale = _lerp(1.0, 0.90, t_late)
            gate_scale = _lerp(1.0, 0.85, t_late)
            calibration_scale = _lerp(1.0, 0.85, t_late)
            regularization_scale = _lerp(1.0, 0.85, t_late)
        else:
            subphase_progress = _phase_progress_scale(subphase_epoch_count, _FULL_TUNING_AUX_WARMUP_EPOCHS)
            aux_scale = subphase_progress
            gate_scale = subphase_progress
            calibration_scale = subphase_progress
            regularization_scale = subphase_progress
        if reentry_mode == "conservative":
            reentry_progress = _phase_progress_scale(phase_epoch_count, _FULL_TUNING_REENTRY_AUX_WARMUP_EPOCHS)
            aux_scale = min(aux_scale, reentry_progress)
            gate_scale = min(gate_scale, reentry_progress)
            calibration_scale = min(calibration_scale, reentry_progress)
            regularization_scale = min(regularization_scale, max(0.35, reentry_progress))
        group_lr_scales = _full_tuning_subphase_group_scales(full_tuning_subphase, subphase_epoch_count=subphase_epoch_count)
    else:
        aux_scale = 1.0
        gate_scale = 1.0
        calibration_scale = 1.0
        regularization_scale = 1.0
    # 铁律 6 (2026-07-23 N4 中和): 让 train.phase_aux_scale / phase_gate_scale /
    # phase_calibration_scale / phase_regularization_scale 覆盖键能强制固定阶段感知
    # 辅助损失缩放, 与 LSTM 平坦 aux loss 对齐. 用 sentinel (None) 区分 "未提供覆盖"
    # 与 "显式提供数值": 未提供时保留自然 ramp 行为; 一旦用户/script 提供任意数值 (含
    # 1.0) 即直接覆盖该 scale (multiplicative 0 * 1.0 = 0 无法中和 epoch 0 的 ramp
    # 起点为 0, 必须 direct cover). 用 None sentinel 在 _build_phase_control_state
    # 内做缺失检测, 这样 build_trainer_state 在用户未显式 set 时不写入 train_section
    # 顶层别名 (此处 _build_phase_control_state 仍可读到 None, 保留原 ramp 行为).
    phase_aux_scale_override = trainer_state.get("phase_aux_scale", None)
    phase_gate_scale_override = trainer_state.get("phase_gate_scale", None)
    phase_calibration_scale_override = trainer_state.get("phase_calibration_scale", None)
    phase_regularization_scale_override = trainer_state.get("phase_regularization_scale", None)
    if phase_aux_scale_override is not None:
        aux_scale = float(phase_aux_scale_override)
    if phase_gate_scale_override is not None:
        gate_scale = float(phase_gate_scale_override)
    if phase_calibration_scale_override is not None:
        calibration_scale = float(phase_calibration_scale_override)
    if phase_regularization_scale_override is not None:
        regularization_scale = float(phase_regularization_scale_override)
    reentry_release_scale = 1.0
    if phase_name == "full_tuning" and reentry_mode == "conservative":
        rollback_pressure = (
            _FULL_TUNING_REENTRY_REPEAT_CAP_MULTIPLIER
            if same_checkpoint_rollback_count >= _FULL_TUNING_REPEAT_ROLLBACK_THRESHOLD
            else 1.0
        )
        # The release cap is consumed before the current epoch starts training.
        # Use the upcoming epoch slot so conservative re-entry never starts from
        # an unusable zero-cap state.
        release_progress = _phase_progress_scale(
            phase_epoch_count + 1,
            _FULL_TUNING_REENTRY_LR_RELEASE_EPOCHS,
        )
        reentry_release_scale = min(1.0, max(0.0, release_progress * rollback_pressure))
    return {
        "phase_name": str(phase_name),
        "phase_epoch_count": phase_epoch_count,
        "reentry_mode": reentry_mode,
        "reentry_buffer_remaining": reentry_buffer_remaining,
        "same_checkpoint_rollback_count": same_checkpoint_rollback_count,
        "aux_scale": float(aux_scale),
        "gate_scale": float(gate_scale),
        "calibration_scale": float(calibration_scale),
        "regularization_scale": float(regularization_scale),
        "reentry_release_scale": float(reentry_release_scale),
        "full_tuning_subphase": full_tuning_subphase,
        "full_tuning_subphase_entry_epoch": int(full_tuning_subphase_entry_epoch),
        "subphase_progress": float(subphase_progress),
        "group_lr_scales": dict(group_lr_scales),
    }


def _apply_group_lr_cap(optimizer: Any, *, baseline_lrs_by_group: Mapping[str, float], cap_scales_by_group: Mapping[str, float]) -> list[dict[str, float | str]]:
    """按组把学习率压到相对基线的保守上限, 用于 rollback 后的保守重入."""
    adjusted: list[dict[str, float | str]] = []
    for param_group in optimizer.param_groups:
        group_name = str(param_group.get("group_name", "") or "")
        if group_name not in cap_scales_by_group:
            continue
        if group_name not in baseline_lrs_by_group:
            continue
        current_lr = coerce_finite_scalar(
            param_group.get("lr", 0.0),
            name=f"optimizer.param_group.{group_name}.lr",
            min_value=0.0,
            inclusive=False,
        )
        baseline_lr = coerce_finite_scalar(
            baseline_lrs_by_group[group_name],
            name=f"baseline_lrs_by_group.{group_name}",
            min_value=0.0,
            inclusive=False,
        )
        cap_scale = coerce_finite_scalar(
            cap_scales_by_group[group_name],
            name=f"cap_scales_by_group.{group_name}",
            min_value=0.0,
            inclusive=False,
        )
        capped_lr = min(current_lr, baseline_lr * cap_scale)
        if capped_lr < current_lr:
            param_group["lr"] = capped_lr
            adjusted.append(
                {
                    "group_name": group_name,
                    "old_lr": current_lr,
                    "new_lr": capped_lr,
                    "baseline_lr": baseline_lr,
                    "cap_scale": cap_scale,
                }
            )
    return adjusted


def _capture_control_group_lrs(optimizer: Any) -> dict[str, float]:
    """提取 soft-control 关心的参数组当前 lr."""
    baseline_lrs: dict[str, float] = {}
    for param_group in optimizer.param_groups:
        group_name = str(param_group.get("group_name", "") or "")
        if group_name not in {"readout", "gate_cal", "filter_context_gate"}:
            continue
        baseline_lrs[group_name] = coerce_finite_scalar(
            param_group.get("lr", 0.0),
            name=f"optimizer.param_group.{group_name}.lr",
            min_value=0.0,
            inclusive=False,
        )
    return baseline_lrs


def _assert_phase_epoch_count_within_budget(
    phase_name: str,
    phase_state_runtime: dict[str, Any],
    trainer_state: dict[str, Any],
) -> None:
    """§13.6.2.1 rollback 总 epoch 上限 fail-loud 守门 (第三轮精读补 + 模块级提取以便单元测试).

    校验两点:
    1. ``phase_state_runtime["phase_epoch_count"]`` 不超过 ``trainer_state["phase_schedule_budget"]``
       中该 phase 对应的 epoch 上限 (rollback chain 在 trainer.py 训练循环末只减 cooldown_epochs 个 epoch
       保留主体进度, 但反复 rollback 仍可能让单阶段总训练 epoch 超过 yaml 钉的上限).
    2. session 累计训练 epoch (含被回退过的 epoch, 通过 last_rollback_phase_epoch_count 累计)
       不超 budget * rollback_total_epoch_hard_cap_multiplier (默认 1.5x, 允许 rollback 多消耗少量 epoch
       但不能突破结构性硬上限).

    任一违规立即 raise RuntimeError, 字面消息含 `§13.6.2.1` 以便审计 / 测试 grep.

    参数:
        phase_name: 当前阶段名 ("warmup" / "gate_alignment" / "full_tuning" / "readout_warmup").
        phase_state_runtime: 单阶段运行时状态字典, 至少含 `phase_epoch_count` 字段,
            可选含 `last_rollback_phase_epoch_count` 字段.
        trainer_state: trainer 全局状态字典, 至少含 `phase_schedule_budget` 字典 (`<phase>_epochs` 子字段)
            与可选 `rollback_total_epoch_hard_cap_multiplier` (默认 1.5).

    异常:
        RuntimeError: phase_epoch_count 超过 phase_schedule_budget.<phase>_epochs, 或
            session 累计 epoch 超过 hard cap 时抛出.
    """
    _phase_budget = trainer_state.get("phase_schedule_budget") or {}
    _phase_budget_key = f"{phase_name}_epochs"
    _phase_budget_value = int(_phase_budget.get(_phase_budget_key) or 0)
    if _phase_budget_value <= 0:
        return  # 无预算约束时不校验 (yaml 未声明该 phase).

    _current_phase_epoch_count = int(phase_state_runtime.get("phase_epoch_count") or 0)
    if _current_phase_epoch_count > _phase_budget_value:
        raise RuntimeError(
            f"§13.6.2.1 phase_epoch_count over-budget fail-loud: "
            f"phase={phase_name} phase_epoch_count={_current_phase_epoch_count} "
            f"exceeds phase_schedule_budget.{_phase_budget_key}={_phase_budget_value}. "
            f"rollback chain must not let single-phase total training epochs exceeded the yaml-pinned budget."
        )
    # session 累计 epoch (含被回退过的) 不超 budget * hard_cap_multiplier (默认 1.5x).
    _rollback_total_epoch_hard_cap_multiplier = float(
        trainer_state.get("rollback_total_epoch_hard_cap_multiplier") or 1.5
    )
    _session_total_epochs = (
        int(phase_state_runtime.get("last_rollback_phase_epoch_count") or 0)
        + _current_phase_epoch_count
    )
    _session_hard_cap = int(math.ceil(_phase_budget_value * _rollback_total_epoch_hard_cap_multiplier))
    if _session_total_epochs > _session_hard_cap:
        raise RuntimeError(
            f"§13.6.2.1 session total epoch over hard-cap fail-loud: "
            f"phase={phase_name} session_total_epochs={_session_total_epochs} "
            f"(last_rollback_phase_epoch_count={int(phase_state_runtime.get('last_rollback_phase_epoch_count') or 0)} "
            f"+ current_phase_epoch_count={_current_phase_epoch_count}) "
            f"exceeds hard_cap=budget*{_rollback_total_epoch_hard_cap_multiplier}={_session_hard_cap}. "
            f"rollback chain must not exceed §13.6.2.1 total-epoch hard cap."
        )


def _optimizer_group_specs() -> list[tuple[str, list[str]]]:
    """Return the canonical optimizer group definitions."""
    return [
        ("readout", [".projection.", ".residual_projection."]),
        ("filter_context_gate", [".filter_context_gate."]),
        (
            "gate_cal",
            [
                ".temporal_context_gate.",
                ".observation_context_gate.",
                ".branch_mix_gate.",
                ".uwb_branch_mix_gate.",
                ".vio_branch_mix_gate.",
                "risk_calibration.",
            ],
        ),
        ("backbone", ["network."]),
    ]


def _collect_named_parameter_pairs(model: Any) -> list[tuple[str, torch.nn.Parameter]]:
    """Collect named parameters from the composite Liquid model structure (v3.1 head-shared)."""
    named_param_pairs: list[tuple[str, torch.nn.Parameter]] = []
    network = getattr(model, "network", None)
    if network is not None and hasattr(network, "named_parameters"):
        for name, param in network.named_parameters():
            named_param_pairs.append((f"network.{name}", param))
    # v3.1 head-shared: 共享 backbone 参数优先。
    backbone = getattr(model, "output_backbone", None)
    if backbone is not None and hasattr(backbone, "named_parameters"):
        for name, param in backbone.named_parameters():
            named_param_pairs.append((f"output_backbone.{name}", param))
    for head_key, head in getattr(model, "output_heads", {}).items():
        if hasattr(head, "named_parameters"):
            for name, param in head.named_parameters():
                named_param_pairs.append((f"output_heads.{head_key}.{name}", param))
    risk_cal = getattr(model, "risk_calibration", None)
    if risk_cal is not None and hasattr(risk_cal, "named_parameters"):
        for name, param in risk_cal.named_parameters():
            named_param_pairs.append((f"risk_calibration.{name}", param))
    return named_param_pairs


def _resolve_param_group_lr(
    *,
    group_name: str,
    base_lr: float,
    lr_layer: Mapping[str, Any],
    current_phase_name: str,
    gate_alignment_readout_lr_scale: float,
    model: Any = None,
) -> float:
    """Resolve the effective base LR for one optimizer group under the current phase."""
    if group_name == "default":
        return base_lr
    group_lr = coerce_finite_scalar(
        lr_layer.get(group_name, base_lr),
        name=f"lr_layer.{group_name}",
        min_value=0.0,
        inclusive=False,
    )
    if group_name == "readout" and current_phase_name == "gate_alignment":
        # C-A (v7 patch): readout lr 4× 跳变 (1e-3 → 2.5e-4) 加 3 ep overlap ramp.
        # 子代 G 报告: 原 trainer.py:2440 直接 ×0.25 硬跳, val loss 在 ep 21 小 spike ~5%.
        # patch 后 ramp 在 phase_epoch_count [0, _PHASE_TRANSITION_OVERLAP_EPOCHS) 内线性过渡.
        if model is not None:
            phase_state = getattr(model, "_phase_control_state", {}) or {}
            gate_phase_epoch = int(phase_state.get("phase_epoch_count") or 0)
            overlap = max(1, _PHASE_TRANSITION_OVERLAP_EPOCHS)
            t = min(1.0, gate_phase_epoch / overlap)
            scale = _lerp(1.0, gate_alignment_readout_lr_scale, t)
            group_lr *= scale
        else:
            # Fallback: 如果调用者未传 model, 直接用终态 scale (与原硬跳一致)
            group_lr *= gate_alignment_readout_lr_scale
    elif group_name == "backbone" and current_phase_name == "gate_alignment":
        # P1.c (v7 patch 2.0): gate_alignment 阶段 backbone lr scale 0.1, 防过激 (P1 patch 1.0 引发 epoch 36+ overfit).
        # 真实观察到 v7 h44_lr3e-4 + h96_lr1e-3 epoch 24-35 短暂胜 baseline (gap -0.001),
        # 但 epoch 36+ 反输 baseline 且 gap 越来越大 (epoch 45 +0.0034), 因为 backbone lr=1e-3 过激.
        # 解冻配合 lr scale=0.1, 让 backbone 慢慢追赶 readout (与 full_tuning 的 0→1 ramp 同思想)
        # 同时也保留 readout_warmup→gate_alignment 的稳定过渡.
        _GATE_ALIGNMENT_BACKBONE_LR_SCALE = 0.1
        group_lr *= _GATE_ALIGNMENT_BACKBONE_LR_SCALE
    return group_lr


def _apply_phase_group_lrs(optimizer: Any, model: Any, optimizer_cfg: Mapping[str, Any]) -> None:
    """Refresh optimizer group LRs for the current phase without rebuilding optimizer state."""
    base_lr = coerce_finite_scalar(
        optimizer_cfg["lr"],
        name="optimizer_cfg.lr",
        min_value=0.0,
        inclusive=False,
    )
    lr_layer = getattr(model, "_lr_layer", {})
    current_phase_name = str(getattr(model, "_current_phase_name", "") or "")
    readout_scale = coerce_finite_scalar(
        getattr(model, "_gate_alignment_readout_lr_scale", _GATE_ALIGNMENT_READOUT_LR_SCALE),
        name="model._gate_alignment_readout_lr_scale",
        min_value=0.0,
        inclusive=False,
    )
    phase_control_state = getattr(model, "_phase_control_state", {}) or {}
    full_tuning_group_scales = dict(phase_control_state.get("group_lr_scales") or {})
    for param_group in optimizer.param_groups:
        group_name = str(param_group.get("group_name", "default") or "default")
        target_lr = _resolve_param_group_lr(
            group_name=group_name,
            base_lr=base_lr,
            lr_layer=lr_layer,
            current_phase_name=current_phase_name,
            gate_alignment_readout_lr_scale=readout_scale,
            model=model,
        )
        if current_phase_name == "full_tuning" and group_name in full_tuning_group_scales:
            # T1: overlap ramp 让 backbone LR 从 0 开始线性 ramp，所以允许 = 0
            target_lr *= coerce_finite_scalar(
                full_tuning_group_scales[group_name],
                name=f"phase_control_state.group_lr_scales.{group_name}",
                min_value=0.0,
                inclusive=True,
            )
        param_group["lr"] = target_lr


def _build_optimizer(model: Any, optimizer_cfg: Mapping[str, Any]):
    """按配置构造优化器。

    只收集当前阶requires_grad=True 的参数，支持 Adam ?SGD。
    weight_decay ?optimizer_cfg 读取；默认YAML ?0.0（L2 正则已显式进入损失主链）。
    但若配置非零值则实际生效，避免死配置靃69。

    ?model 上挂载了 _lr_layer 字典时，按参数名扢、属层分配不同学习率：
    - readout: 输出头投影层
    - gate_cal: 门控和校准参。
    - filter_context_gate: 滤波上下文门。
    - backbone: 网络主干
    未匹配的参数使用 optimizer_cfg 中的基础 lr。

    参数：
        model: 液体模型实例。
        optimizer_cfg: 优化器配置映射，必须包含 "name" ?"lr"。

    返回值：
        构好PyTorch 优化器实例。

    失败条件。
        ValueError: 当没有可训练参数或优化器名不被支持时抛出。
    """
    optimizer_name = str(optimizer_cfg["name"]).lower()  # 优化器名统一小写。
    # D5/D7：与 lstm/trainer.py 同名函数口径对齐，统丢coerce_finite_scalar 中心入口。
    # 拒绝 NaN/Inf/bool/负零学习率，避免数值异常静默穿透到优化器。
    base_lr = coerce_finite_scalar(
        optimizer_cfg["lr"], name="optimizer_cfg.lr", min_value=0.0, inclusive=False
    )
    # D5/D7：weight_decay 走同丢、校验入口，仅校验非负有限（与 LSTM 侧一致）。
    # ?.get(..., 0.0) 仅在键缺失时补默认，不再`or 0.0` 把显falsy 值（None/False）掩盖成 0。
    weight_decay = coerce_finite_scalar(
        optimizer_cfg.get("weight_decay", 0.0), name="optimizer_cfg.weight_decay", min_value=0.0
    )

    lr_layer = getattr(model, "_lr_layer", {})
    current_phase_name = str(getattr(model, "_current_phase_name", "") or "")
    # D2/D9：与 lstm/trainer.py 口径对齐，拒绝未lr_layer 键，避免配置拼写错误静默逢、化为 base_lr。
    # backbone_unfreeze_ramp ?_apply_backbone_unfreeze_ramp 单独消费（为 dict，非标量），故纳入白名单但不参与 lr 取。
    _allowed_lr_layer_keys = {
        "readout", "gate_cal", "filter_context_gate", "backbone", "backbone_unfreeze_ramp",
    }
    _unsupported_lr_layer_keys = [key for key in lr_layer if key not in _allowed_lr_layer_keys]
    if _unsupported_lr_layer_keys:
        raise ValueError(
            f"unsupported lr_layer keys: {_unsupported_lr_layer_keys}; "
            f"allowed: {sorted(_allowed_lr_layer_keys)}"
        )

    # 按参数名扢、属层分组，未匹配的使用基硢 lr。
    group_specs = _optimizer_group_specs()

    # 收集各组的可训练参数。
    # model ?_LiquidModel（非 nn.Module），要从子模块收集命名参数。
    group_params: dict[str, list[torch.nn.Parameter]] = {name: [] for name, _ in group_specs}
    group_params["default"] = []  # 未匹配的参数归入默认组。

    named_param_pairs = _collect_named_parameter_pairs(model)

    for name, param in named_param_pairs:
        matched = False
        for group_name, patterns in group_specs:
            if any(pattern in name for pattern in patterns):
                group_params[group_name].append(param)
                matched = True
                break
        if not matched:
            group_params["default"].append(param)

    # 构建参数组列表。
    param_groups: list[dict[str, Any]] = []
    has_any_param = False
    for group_name, params in group_params.items():
        if not params:
            continue
        has_any_param = True
        group_lr = _resolve_param_group_lr(
            group_name=group_name,
            base_lr=base_lr,
            lr_layer=lr_layer,
            current_phase_name=current_phase_name,
            gate_alignment_readout_lr_scale=coerce_finite_scalar(
                getattr(model, "_gate_alignment_readout_lr_scale", _GATE_ALIGNMENT_READOUT_LR_SCALE),
                name="model._gate_alignment_readout_lr_scale",
                min_value=0.0,
                inclusive=False,
            ),
        )
        param_groups.append({"params": params, "lr": group_lr, "group_name": group_name})

    if not has_any_param:  # 没有参数就没法训练。
        raise ValueError("model does not expose trainable parameters")  # 直接报错。

        # 如果没有分层配置或所有组都用基。 lr，退化为单参数列表。
    if not lr_layer or all(pg["lr"] == base_lr for pg in param_groups):
        all_params = [p for pg in param_groups for p in pg["params"]]
        if optimizer_name == "adam":
            return torch.optim.Adam(
                all_params,
                lr=base_lr,
                weight_decay=weight_decay,
                foreach=False,
                fused=False,
            )
        if optimizer_name == "sgd":
            return torch.optim.SGD(all_params, lr=base_lr, weight_decay=weight_decay)
        raise ValueError(f"unsupported optimizer: {optimizer_name}")

    if optimizer_name == "adam":  # Adam 分支。
        return torch.optim.Adam(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
            foreach=False,
            fused=False,
        )  # 返回 Adam。
    if optimizer_name == "sgd":  # SGD 分支。
        return torch.optim.SGD(param_groups, lr=base_lr, weight_decay=weight_decay)  # 返回 SGD。
    raise ValueError(f"unsupported optimizer: {optimizer_name}")  # 其他名字都不接受。


def _apply_backbone_unfreeze_ramp(
    optimizer: Any,
    model: Any,
    epoch_index: int,
    trainer_state: Mapping[str, Any],
    *,
    ramp_epoch: int | None = None,
) -> None:
    """?full_tuning 阶段内按 backbone_unfreeze_ramp 配置渐进调整 backbone 学习率。

    YAML 配置示例。
        backbone_unfreeze_ramp:
          step1_lr: 1.0e-5
          step1_epochs: 5
          step2_lr: 5.0e-5
          step2_epochs: 5
          step3_lr: 1.0e-4

    如果配置缺失，不做任何调整（使用 lr_layer.backbone 的固定、）。
    """
    lr_layer = getattr(model, "_lr_layer", {})
    ramp_cfg = lr_layer.get("backbone_unfreeze_ramp")
    if not ramp_cfg:
        return # ?ramp 配置，使用固backbone lr。
        # D5/D2：ramp_cfg 必须Mapping，避免list/str/int ?truthy ?Mapping 值在 .get() ?AttributeError。
    if not isinstance(ramp_cfg, Mapping):
        raise TypeError(
            "lr_layer.backbone_unfreeze_ramp must be a mapping, got "
            f"{type(ramp_cfg).__name__}"
        )

    # Compute the in-phase full_tuning epoch offset.
    # Explicit ramp_epoch takes precedence so rollback reentry can restart conservatively.
    if ramp_epoch is None:
        phase_schedule = trainer_state["phase_schedule"]
        warmup_epochs = int(phase_schedule["warmup_epochs"])
        gate_alignment_epochs = int(phase_schedule["gate_alignment_epochs"])
        full_tuning_start_epoch = warmup_epochs + gate_alignment_epochs + 1
        epoch_offset = epoch_index - full_tuning_start_epoch
    else:
        epoch_offset = int(ramp_epoch)

        # ?ramp 步骤确定 backbone 学习率。
        # D5/D7：step*_lr ?coerce_finite_scalar 中心入口，拒NaN/Inf/bool/负零，
        # 口径_build_optimizer ?lr_layer.<group> 取（L1534-1539）一致。
    step1_lr = coerce_finite_scalar(
        ramp_cfg.get("step1_lr", 1e-5),
        name="lr_layer.backbone_unfreeze_ramp.step1_lr",
        min_value=0.0,
        inclusive=False,
    )
    step2_lr = coerce_finite_scalar(
        ramp_cfg.get("step2_lr", 5e-5),
        name="lr_layer.backbone_unfreeze_ramp.step2_lr",
        min_value=0.0,
        inclusive=False,
    )
    step3_lr = coerce_finite_scalar(
        ramp_cfg.get("step3_lr", 1e-4),
        name="lr_layer.backbone_unfreeze_ramp.step3_lr",
        min_value=0.0,
        inclusive=False,
    )
    # D5/D7：step*_epochs 必须是有限非负整数，口径_resolve_phase_epochs（L1788-1789）对齐。
    step1_epochs_raw = ramp_cfg.get("step1_epochs", 5)
    if not is_integer(step1_epochs_raw):
        raise TypeError(
            "lr_layer.backbone_unfreeze_ramp.step1_epochs must be an integer"
        )
    step1_epochs = int(step1_epochs_raw)
    if step1_epochs < 0:
        raise ValueError(
            "lr_layer.backbone_unfreeze_ramp.step1_epochs must be non-negative"
        )
    step2_epochs_raw = ramp_cfg.get("step2_epochs", 5)
    if not is_integer(step2_epochs_raw):
        raise TypeError(
            "lr_layer.backbone_unfreeze_ramp.step2_epochs must be an integer"
        )
    step2_epochs = int(step2_epochs_raw)
    if step2_epochs < 0:
        raise ValueError(
            "lr_layer.backbone_unfreeze_ramp.step2_epochs must be non-negative"
        )

    if epoch_offset < step1_epochs:
        target_lr = step1_lr
    elif epoch_offset < step1_epochs + step2_epochs:
        target_lr = step2_lr
    else:
        target_lr = step3_lr

        # 修改 optimizer ?backbone 参数组的学习率。
    # 收集 network 中所requires_grad 的参数id，用于匹param_group。
    network_param_ids = {id(p) for _, p in model.network.named_parameters() if p.requires_grad}
    for param_group in optimizer.param_groups:
        param_ids = {id(p) for p in param_group["params"]}
        if param_ids & network_param_ids:  # 交集非空说明backbone 。
            param_group["lr"] = target_lr
            break


def _head_phase_trainability_snapshot(head: Any) -> dict[str, bool]:
    """快照一个输出头（或 v3.1 共享 backbone）中各子模块的可训练状态。

    v3.1 head-shared 架构下，gate 层在 output_backbone 上，
    projection/residual_projection 在 output_heads 各 head 中。
    遍历 projection、residual_projection 和六种门控子模块。
    记录每个子模块是否有至少一个 requires_grad=True 的参数。
    用于训练阶段切换时的可训练、审计。

    参数：
        head: 输出头模块实例或共享 backbone（v3.1）。

    返回值：
        字典，键为子模块名，值为布尔值表示是否可训练。
    """
    result: dict[str, bool] = {}
    for attr in ("projection", "residual_projection"):
        module = getattr(head, attr, None)
        result[attr] = (
            any(parameter.requires_grad for parameter in module.parameters())
            if module is not None and callable(getattr(module, "parameters", None))
            else False
        )
    for attr in (
        "temporal_context_gate",
        "observation_context_gate",
        "filter_context_gate",
        "branch_mix_gate",
        "uwb_branch_mix_gate",
        "vio_branch_mix_gate",
    ):
        module = getattr(head, attr, None)
        result[attr] = (
            any(parameter.requires_grad for parameter in module.parameters())
            if module is not None and callable(getattr(module, "parameters", None))
            else False
        )
    return result


def _build_phase_trainability_contract(model: Any, *, phase_name: str) -> dict[str, Any]:
    """构建当前训练阶段的可训练性合约。

    合约记录network、risk_calibration 和输出头各子模块。
    可训练状态，以及各辅助损失项是否启用。用于训练日志和审计。

    参数：
        model: 液体模型实例。
        phase_name: 当前阶段名（readout_warmup / gate_alignment / full_tuning）。

    返回值：
        合约字典，包network_trainable、risk_calibration_trainable。
        head_trainability、auxiliary_terms_enabled ?phase_name 五个。

    失败条件。
        ValueError: 当模型没有输出头时抛出。
    """
    _safe_finite_float_fn = _get_checkpoint_utils()
    _risk_calibration_trainable_fn, _loss_for_length_group_fn = _get_loss_compute()
    phase_state = {  # 收集各模块的可训练状态。
        "network_trainable": any(parameter.requires_grad for parameter in model.network.parameters()),  # 共享网络是否可训练。
        "risk_calibration_trainable": _risk_calibration_trainable_fn(model), # risk_calibration 是否可训练（与辅助损失判_risk_calibration_trainable 同口径，避免漂移）。
    }
    # v3.1 head-shared: gate 在 output_backbone，projection 在 output_heads。统一从两个对象快照。
    reference_head = next(iter(getattr(model, "output_heads", {}).values()), None)  # 取第一个 final-projection head 作为代表。
    reference_backbone = getattr(model, "output_backbone", None)  # v3.1 共享 backbone。
    if reference_head is None and reference_backbone is None:  # 两个都没有就不能继续。
        raise ValueError("liquid model must expose at least one output head or output_backbone")  # 直接报错。
    # 拼合：head 提供 projection/residual_projection 可训练性；backbone 提供 6 个 gate 可训练性。
    head_snapshot = _head_phase_trainability_snapshot(reference_head) if reference_head is not None else {}
    backbone_snapshot = _head_phase_trainability_snapshot(reference_backbone) if reference_backbone is not None else {}
    # v3.1 head-shared bug fix: 旧 dict merge 用 `{**backbone_snapshot, **head_snapshot}` 让 head 的
    # gate_*=False（head 已无 gate，属性查不到回退 False）覆盖 backbone 的 gate_*=True。
    # 现只从 head 提取 projection/residual_projection，从 backbone 提取所有 *_gate，避免反向覆盖。
    head_only_keys = {"projection", "residual_projection"}
    merged_snapshot = dict(backbone_snapshot)  # 先放 backbone（含 6 个 *_gate）。
    for key in head_only_keys:  # 再用 head 的 projection/residual_projection 覆盖。
        if key in head_snapshot:
            merged_snapshot[key] = head_snapshot[key]
    phase_state["head_trainability"] = merged_snapshot
    phase_state["auxiliary_terms_enabled"] = {  # 各辅助损失项是否启用。
        "calibration": phase_state["risk_calibration_trainable"],  # calibration 只在 risk_calibration 可训练时启用。
        "mono": phase_state["risk_calibration_trainable"],  # mono 同上。
        "gate_l1": any( # 只要有任何门控可训练就启gate_l1；从 head_trainability 快照\"_gate" 后缀派生，避免与 _head_phase_trainability_snapshot 键名重复定义造成漂移。
            trainable
            for key, trainable in phase_state["head_trainability"].items()
            if key.endswith("_gate")
        ),
        "reg_l2": True,  # L2 正则始终启用。
    }
    phase_state["phase_name"] = phase_name  # 记录当前阶段名。
    phase_control_state = getattr(model, "_phase_control_state", {}) or {}
    phase_state["phase_control_consumption"] = {
        "aux_scale": _safe_finite_float_fn(phase_control_state.get("aux_scale")),
        "gate_scale": _safe_finite_float_fn(phase_control_state.get("gate_scale")),
        "calibration_scale": _safe_finite_float_fn(phase_control_state.get("calibration_scale")),
        "reentry_release_scale": _safe_finite_float_fn(phase_control_state.get("reentry_release_scale")),
    }
    phase_state["full_tuning_subphase"] = (
        str(phase_control_state.get("full_tuning_subphase") or "joint_drive")
        if phase_name == "full_tuning"
        else None
    )
    phase_state["full_tuning_group_lr_scales"] = (
        dict(phase_control_state.get("group_lr_scales") or {})
        if phase_name == "full_tuning"
        else {}
    )
    return phase_state  # 返回完整的可训练性合约。


def _count_trainable_parameters(model: Any) -> int:
    """统计模型中可训练参数的、数量。

    参数：
        model: 模型实例。

    返回值：
        可训练参数的总元素数。
    """
    return sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)  # 累加所有可训练参数的元素数。


def _optimizer_effective_state(optimizer: Any) -> dict[str, Any]:
    """提取优化器的有效状摘要。

    包括参数组数量、各组学习率、各组权重衰减。
    各组可训练参数数量和总可训练参数数量。

    参数：
        optimizer: PyTorch 优化器实例。

    返回值：
        包含 param_group_count。乸aram_group_lrs。乸aram_group_weight_decays。。
        param_group_trainable_counts ?trainable_parameter_count 的字典。
    """
    param_group_lrs = [coerce_finite_scalar(group.get("lr", 0.0), name="lr") for group in optimizer.param_groups]  # 各参数组的学习率，拒NaN/Inf 静默穿。
    param_group_weight_decays = [coerce_finite_scalar(group.get("weight_decay", 0.0), name="weight_decay") for group in optimizer.param_groups]  # 各参数组的权重衰减，拒绝 NaN/Inf 静默穿。
    param_group_trainable_counts = [  # 各参数组的可训练参数数量。
        sum(int(parameter.numel()) for parameter in group.get("params", []) if getattr(parameter, "requires_grad", False))
        for group in optimizer.param_groups
    ]
    return {
        "param_group_count": len(optimizer.param_groups),
        "param_group_lrs": param_group_lrs,
        "param_group_weight_decays": param_group_weight_decays,
        "param_group_trainable_counts": param_group_trainable_counts,
        "trainable_parameter_count": sum(param_group_trainable_counts),
    }


def _maybe_apply_phase_shock_soft_control(
    optimizer: Any,
    *,
    phase_name: str,
    phase_switch_shock: Mapping[str, float | None],
    trainer_state: Mapping[str, Any],
    prior_shock_streak: int = 0,
    cooldown_remaining: int = 0,
    phase_epoch_count: int = 0,
    reentry_buffer_remaining: int = 0,
    baseline_lrs_by_group: Mapping[str, float] | None = None,
    full_tuning_subphase: str | None = None,
) -> dict[str, Any] | None:
    """根据 phase shock 做最小恢复控制：缓冲、冷却限速与降速。"""
    if phase_name not in {"gate_alignment", "full_tuning"}:
        return None
    shock_threshold = coerce_finite_scalar(
        trainer_state.get("soft_control_shock_threshold", _SOFT_CONTROL_SHOCK_THRESHOLD),
        name="soft_control_shock_threshold",
        min_value=0.0,
        inclusive=False,
    )
    lr_decay = coerce_finite_scalar(
        trainer_state.get("soft_control_lr_decay", _SOFT_CONTROL_LR_DECAY),
        name="soft_control_lr_decay",
        min_value=0.0,
        inclusive=False,
    )
    streak_length = int(
        coerce_finite_scalar(
            trainer_state.get("soft_control_streak_length", _SOFT_CONTROL_STREAK_LENGTH),
            name="soft_control_streak_length",
            min_value=1.0,
            inclusive=True,
        )
    )
    streak_lr_decay = coerce_finite_scalar(
        trainer_state.get("soft_control_streak_lr_decay", _SOFT_CONTROL_STREAK_LR_DECAY),
        name="soft_control_streak_lr_decay",
        min_value=0.0,
        inclusive=False,
    )
    buffer_epochs = int(
        coerce_finite_scalar(
            trainer_state.get("soft_control_buffer_epochs", _SOFT_CONTROL_BUFFER_EPOCHS),
            name="soft_control_buffer_epochs",
            min_value=0.0,
        )
    )
    cooldown_epochs = int(
        coerce_finite_scalar(
            trainer_state.get("soft_control_cooldown_epochs", _SOFT_CONTROL_COOLDOWN_EPOCHS),
            name="soft_control_cooldown_epochs",
            min_value=0.0,
        )
    )
    lr_floor_scale = coerce_finite_scalar(
        trainer_state.get("soft_control_lr_floor_scale", _SOFT_CONTROL_LR_FLOOR_SCALE),
        name="soft_control_lr_floor_scale",
        min_value=0.0,
        inclusive=False,
    )
    rollback_patience = int(
        coerce_finite_scalar(
            trainer_state.get("soft_control_rollback_patience", _SOFT_CONTROL_ROLLBACK_PATIENCE),
            name="soft_control_rollback_patience",
            min_value=1.0,
            inclusive=True,
        )
    )
    shock_value = phase_switch_shock.get("supervised_loss_relative")
    trigger_metric = "supervised_loss_relative"
    if shock_value is None:
        shock_value = phase_switch_shock.get("selection_score_relative")
        trigger_metric = "selection_score_relative"
    if shock_value is None:
        return None
    # Relative deltas are allowed to be negative when validation improves.
    # Soft control only reacts to degradation, so improvements map to zero shock.
    shock_value = max(
        coerce_finite_scalar(shock_value, name="phase_switch_shock.trigger_value"),
        0.0,
    )
    if shock_value < shock_threshold:
        return None
    effective_buffer_remaining = max(
        int(reentry_buffer_remaining),
        max(0, int(buffer_epochs) - int(phase_epoch_count)),
    )
    if effective_buffer_remaining > 0:
        # buffer_only 阶段不施加 LR 衰减，但事件仍带 applied_lr_decay 字段以便消费者统一读取（避免 KeyError）。
        return {
            "mode": "buffer_only",
            "phase_name": str(phase_name),
            "full_tuning_subphase": full_tuning_subphase,
            "trigger_metric": trigger_metric,
            "trigger_value": shock_value,
            "shock_threshold": shock_threshold,
            "lr_decay": lr_decay,
            "applied_lr_decay": 1.0,  # buffer_only 不衰减，置 1.0 表示"未施加任何 LR 衰减"。
            "shock_streak": int(prior_shock_streak),
            "buffer_epochs": int(buffer_epochs),
            "effective_buffer_remaining": int(effective_buffer_remaining),
            "phase_epoch_count": int(phase_epoch_count),
            "cooldown_remaining": int(max(0, cooldown_remaining)),
            "cooldown_epochs": int(cooldown_epochs),
            "rollback_patience": int(rollback_patience),
            "lr_floor_scale": lr_floor_scale,
            "affected_groups": [],
        }
    if int(cooldown_remaining) > 1:
        # cooldown_only 阶段仍在冷却窗口内，不施加新的 LR 衰减，但保持 applied_lr_decay 字段口径一致。
        return {
            "mode": "cooldown_only",
            "phase_name": str(phase_name),
            "full_tuning_subphase": full_tuning_subphase,
            "trigger_metric": trigger_metric,
            "trigger_value": shock_value,
            "shock_threshold": shock_threshold,
            "lr_decay": lr_decay,
            "applied_lr_decay": 1.0,  # cooldown_only 不衰减，置 1.0 保持字段口径统一。
            "shock_streak": int(prior_shock_streak),
            "streak_length": streak_length,
            "streak_lr_decay": streak_lr_decay,
            "buffer_epochs": int(buffer_epochs),
            "effective_buffer_remaining": int(effective_buffer_remaining),
            "phase_epoch_count": int(phase_epoch_count),
            "cooldown_remaining": int(cooldown_remaining),
            "cooldown_epochs": int(cooldown_epochs),
            "rollback_patience": int(rollback_patience),
            "lr_floor_scale": lr_floor_scale,
            "request_rollback": False,
            "affected_groups": [],
        }
    # Cooldown 最后一轮：不继续降速，而是向 baseline_lr 做部分恢复
    # （Learning without Forgetting, Li & Hoiem TPAMI 2018 — 保真机制）
    if int(cooldown_remaining) == 1:
        lr_recovery_scale = coerce_finite_scalar(
            trainer_state.get("soft_control_lr_recovery_scale", _SOFT_CONTROL_LR_RECOVERY_SCALE),
            name="soft_control_lr_recovery_scale",
            min_value=0.0,
            inclusive=False,
        )
        affected_recovery_groups: list[dict[str, float | str]] = []
        for param_group in optimizer.param_groups:
            group_name = str(param_group.get("group_name", "") or "")
            if group_name not in {"readout", "gate_cal", "filter_context_gate"}:
                continue
            current_lr = coerce_finite_scalar(
                param_group.get("lr", 0.0),
                name=f"optimizer.param_group.{group_name}.lr",
                min_value=0.0,
                inclusive=False,
            )
            baseline_lr = (
                coerce_finite_scalar(
                    baseline_lrs_by_group[group_name],
                    name=f"baseline_lrs_by_group.{group_name}",
                    min_value=0.0,
                    inclusive=True,
                )
                if isinstance(baseline_lrs_by_group, Mapping) and group_name in baseline_lrs_by_group
                else current_lr
            )
            # 从当前降速后的 lr 向 baseline_lr 恢复 lr_recovery_scale 的比例
            recovery_target = baseline_lr * lr_recovery_scale
            if recovery_target > current_lr:
                param_group["lr"] = min(current_lr * 2.0, recovery_target)
            affected_recovery_groups.append(
                {
                    "group_name": group_name,
                    "old_lr": current_lr,
                    "new_lr": float(param_group["lr"]),
                    "baseline_lr": baseline_lr,
                    "recovery_target": recovery_target,
                }
            )
        return {
            "mode": "cooldown_recovery",
            "phase_name": str(phase_name),
            "full_tuning_subphase": full_tuning_subphase,
            "trigger_metric": trigger_metric,
            "trigger_value": shock_value,
            "shock_threshold": shock_threshold,
            "lr_decay": lr_decay,
            # cooldown_recovery 分支向 baseline 恢复 lr，不施加新的衰减；保持 applied_lr_decay=1.0 字段口径统一
            "applied_lr_decay": 1.0,
            "shock_streak": int(prior_shock_streak),
            "streak_length": streak_length,
            "streak_lr_decay": streak_lr_decay,
            "cooldown_remaining": 0,
            "cooldown_epochs": int(cooldown_epochs),
            "buffer_epochs": int(buffer_epochs),
            "effective_buffer_remaining": int(effective_buffer_remaining),
            "phase_epoch_count": int(phase_epoch_count),
            "rollback_patience": int(rollback_patience),
            "lr_floor_scale": lr_floor_scale,
            "request_rollback": False,
            "affected_groups": affected_recovery_groups,
        }
    shock_streak = int(prior_shock_streak) + 1
    applied_lr_decay = lr_decay
    control_mode = "slowdown_only"
    if shock_streak >= streak_length:
        applied_lr_decay *= streak_lr_decay
        control_mode = "slowdown_streak"
    should_request_rollback = shock_streak >= rollback_patience

    affected_groups: list[dict[str, float | str]] = []
    for param_group in optimizer.param_groups:
        group_name = str(param_group.get("group_name", "") or "")
        if group_name not in {"readout", "gate_cal", "filter_context_gate"}:
            continue
        current_lr = coerce_finite_scalar(
            param_group.get("lr", 0.0),
            name=f"optimizer.param_group.{group_name}.lr",
            min_value=0.0,
            inclusive=False,
        )
        baseline_lr = (
            coerce_finite_scalar(
                baseline_lrs_by_group[group_name],
                name=f"baseline_lrs_by_group.{group_name}",
                min_value=0.0,
                inclusive=False,
            )
            if isinstance(baseline_lrs_by_group, Mapping) and group_name in baseline_lrs_by_group
            else current_lr
        )
        lr_floor = max(baseline_lr * lr_floor_scale, 1e-12)
        new_lr = max(current_lr * applied_lr_decay, lr_floor)
        param_group["lr"] = new_lr
        affected_groups.append(
            {
                "group_name": group_name,
                "old_lr": current_lr,
                "new_lr": new_lr,
                "baseline_lr": baseline_lr,
                "lr_floor": lr_floor,
                "hit_lr_floor": bool(new_lr <= lr_floor + 1e-12),
            }
        )
    if not affected_groups:
        return None
    return {
        "mode": control_mode,
        "phase_name": str(phase_name),
        "full_tuning_subphase": full_tuning_subphase,
        "trigger_metric": trigger_metric,
        "trigger_value": shock_value,
        "shock_threshold": shock_threshold,
        "lr_decay": lr_decay,
        "applied_lr_decay": applied_lr_decay,
        "shock_streak": shock_streak,
        "streak_length": streak_length,
        "streak_lr_decay": streak_lr_decay,
        "buffer_epochs": int(buffer_epochs),
        "effective_buffer_remaining": int(effective_buffer_remaining),
        "phase_epoch_count": int(phase_epoch_count),
        "cooldown_epochs": int(cooldown_epochs),
        "cooldown_remaining": int(max(0, cooldown_remaining)),
        "rollback_patience": int(rollback_patience),
        "request_rollback": bool(should_request_rollback),
        "lr_floor_scale": lr_floor_scale,
        "affected_groups": affected_groups,
        "reentry_target_subphase": (
            "entry_bridge"
            if phase_name == "full_tuning" and str(full_tuning_subphase or "joint_drive") == "joint_drive"
            else ("joint_drive" if phase_name == "full_tuning" else None)
        ),
    }




def _resolve_runtime_device(device_request: str) -> tuple[torch.device, str]:
    """解析运行时设备请求，返回 torch.device 和设备名字符串。

    支持 "auto"?cuda"?cuda:N" ?"cpu" 四种请求格式。
    "auto" 时优先使CUDA，不可用时回逢CPU。
    显式请求 CUDA 但不可用时也回CPU。

    参数：
        device_request: 设备请求字符串。

    返回值：
    二元（torch.device, device_name_str)。
        - torch.device: 实际使用的设备对象；
        - device_name_str: 设备名字符串cuda"?cuda:N" ?"cpu"）。
    """
    request = str(device_request or DEVICE_CPU).strip().lower()  # 统一转小写，空、默认CPU。
    if request == DEVICE_AUTO:  # 自动选择模式 → 必须用 CUDA，不可用则报错 (C-4)
        if _check_cuda_runtime_usable():  # CUDA 可用。
            return torch.device(DEVICE_CUDA), DEVICE_CUDA  # 使用 CUDA。
        raise RuntimeError("CUDA unavailable: GPU required for training (C-4: 训练默认 GPU)")
    if request == DEVICE_CPU:  # 显式请求 CPU。
        return torch.device(DEVICE_CPU), DEVICE_CPU
    if request == DEVICE_CUDA:  # 显式请求 CUDA。
        if _check_cuda_runtime_usable():  # CUDA 可用。
            return torch.device(DEVICE_CUDA), DEVICE_CUDA  # 使用 CUDA。
        raise RuntimeError("CUDA unavailable: GPU required for training (C-4: 训练默认 GPU)")
    if request.startswith(DEVICE_CUDA_PREFIX):  # 显式请求指定 CUDA 设备。
        if _check_cuda_runtime_usable():  # CUDA 可用。
            return torch.device(request), request  # 使用指定 CUDA 设备。
        raise RuntimeError("CUDA unavailable: GPU required for training (C-4: 训练默认 GPU)")
    raise ValueError(  # 未知请求必须显式报错，避免静默回逢 CPU 破坏公平可复现、（lstm/trainer.py 对齐）。
        f"unsupported device request: {device_request!r}; expected 'auto', 'cpu', 'cuda' or 'cuda:N'"
    )


def _resolve_phase_epochs(
    phase_cfg: Mapping[str, Any] | None,
    *,
    total_epochs: int,
) -> tuple[int, int, int]:
    """从阶段调度配置中解析 warmup、gate_alignment ?full_tuning ?epoch 数。

    warmup_epochs ?readout_warmup 阶段持续多少epoch。
    gate_alignment_epochs ?gate_alignment 阶段持续多少epoch。
    full_tuning_epochs ?full_tuning 阶段持续多少epoch。
    三阶段预算必须与总训epoch 数严格对齐，避免 paper 预算配置静默失效。

    参数：
        phase_cfg: 阶段调度配置映射，可能包warmup_epochs。
        gate_alignment_epochs ?full_tuning_epochs 键，None 时前两、默认为 0。
            full_tuning 自动由剩余、预算推导。
        total_epochs: 总计 epoch 数量。

    返回值：
    三元（warmup_epochs, gate_alignment_epochs, full_tuning_epochs)。

    失败条件。
        TypeError: 当配置类型不正确时抛出。
        ValueError: 当配置、为负、和超过 total_epochs。
            或显full_tuning_epochs 与剩余预算不一致时抛出。
    """
    warmup_epochs = 0  # 默认不做 warmup。
    gate_alignment_epochs = 0  # 默认不做 gate alignment。
    full_tuning_epochs: int | None = None  # full_tuning 预算可显式声明，也可由、预算推导。
    if isinstance(phase_cfg, Mapping):  # 配置存在时读取。
        raw_warmup = phase_cfg.get("warmup_epochs", 0)  # 读取 warmup epoch 数量。
        raw_gate_alignment = phase_cfg.get("gate_alignment_epochs", 0)  # 读取 gate alignment epoch 数量。
        raw_full_tuning = phase_cfg.get("full_tuning_epochs")  # 读取 full_tuning epoch 数量。
        if not is_integer(raw_warmup):  # 必须是整数，不能是布尔。
            raise TypeError("train.phase_schedule.warmup_epochs must be an integer")  # 类型错误。
        if not is_integer(raw_gate_alignment):  # 同上。
            raise TypeError("train.phase_schedule.gate_alignment_epochs must be an integer")  # 类型错误。
        if raw_full_tuning is not None and not is_integer(raw_full_tuning):  # 显式提供时也必须是整数。
            raise TypeError("train.phase_schedule.full_tuning_epochs must be an integer")  # 类型错误。
        warmup_epochs = int(raw_warmup)  # 赋校验过warmup epoch 数。
        gate_alignment_epochs = int(raw_gate_alignment)  # 赋校验过gate alignment epoch 数。
        full_tuning_epochs = int(raw_full_tuning) if raw_full_tuning is not None else None  # 保存显式 full_tuning 预算。
    if warmup_epochs < 0:  # 不允许负数。
        raise ValueError("train.phase_schedule.warmup_epochs must be non-negative")  # 报错。
    if gate_alignment_epochs < 0:  # 不允许负数。
        raise ValueError("train.phase_schedule.gate_alignment_epochs must be non-negative")  # 报错。
    if full_tuning_epochs is not None and full_tuning_epochs < 0:  # 不允许负数。
        raise ValueError("train.phase_schedule.full_tuning_epochs must be non-negative")  # 报错。
    if warmup_epochs + gate_alignment_epochs > total_epochs: # 冻结阶段总时长不能超过 epoch 数。
        raise ValueError("train.phase_schedule total frozen epochs must not exceed train.epochs")  # 报错。
    derived_full_tuning_epochs = total_epochs - warmup_epochs - gate_alignment_epochs  # 剩余预算全部full_tuning。
    if full_tuning_epochs is None:
        full_tuning_epochs = derived_full_tuning_epochs  # 未显式配置时直接采用推导值。
    elif full_tuning_epochs != derived_full_tuning_epochs:
        raise ValueError(
            "train.phase_schedule.full_tuning_epochs must equal "
            "train.epochs - warmup_epochs - gate_alignment_epochs"
        )  # 防止 YAML 里的 paper 预算键被静默忽略。
    return warmup_epochs, gate_alignment_epochs, full_tuning_epochs  # 返回三个阶段epoch 数。


def _resolve_epoch_phase_name(
    epoch_index: int,
    *,
    warmup_epochs: int,
    gate_alignment_epochs: int,
) -> str:
    """根据 epoch 索引判断当前属于哪个训练阶段。

    - epoch_index <= warmup_epochs: readout_warmup（只训练输出头投影层）。
    - warmup_epochs < epoch_index <= warmup_epochs + gate_alignment_epochs: gate_alignment（训练门控，v7 patch：同时解冻 backbone）。
    - 其余: full_tuning（全部解冻训练）

    参数：
        epoch_index: 当前 epoch 索引（从 1 开始）。
        warmup_epochs: warmup 阶段的 epoch 数量。
        gate_alignment_epochs: gate alignment 阶段的 epoch 数量。

    返回值：
        阶段名字符串 readout_warmup / gate_alignment / full_tuning。
    """
    if epoch_index <= warmup_epochs:  # 在 warmup 轮次范围内。
        return "readout_warmup"  # 只训练输出头投影层。
    if epoch_index <= warmup_epochs + gate_alignment_epochs:  # 在 gate alignment 轮次范围内。
        return "gate_alignment"  # 训练门控（v7 patch：同时解冻 backbone）。
    return "full_tuning"  # 全部解冻训练。


def _set_requires_grad_for_module(module: Any, enabled: bool) -> None:
    """批量设置模块中所有参数的 requires_grad 标志。

    安全地处理 None 输入和非 nn.Module 对象。

    参数：
        module: 模块实例，可以是 None。
        enabled: True 表示开启梯度，False 表示关闭梯度。
    """
    if module is None:  # 模块None 时直接返回。
        return  # 不做任何操作。
    parameters = getattr(module, "parameters", None)  # 安全地取 parameters 方法。
    if not callable(parameters):  # 没有 parameters 方法说明不是 nn.Module。
        return  # 不做任何操作。
    for parameter in parameters():  # 遍历模块的所有参数。
        parameter.requires_grad_(enabled)  # 原地设置 requires_grad。


def _configure_training_phase(model: Any, *, phase_name: str) -> dict[str, bool]:
    """根据阶段名配置模型各模块的可训练状态。

    三个阶段的冻结策略：
    - readout_warmup: 冻结共享网络和门控，只训练输出头投影层；
    - gate_alignment: 解冻共享网络（v7 patch：从 warmup 终态起 backbone 即解冻），
      训练门控、校准层和低学习率读出层。
    - full_tuning: 解冻共享网络，输出头全部可训练。
    risk_calibration 在 readout_warmup 阶段冻结，其余阶段解冻。

    参数：
        model: 液体模型实例。
        phase_name: 阶段名，必须为 readout_warmup、gate_alignment 或 full_tuning。

    返回值：
        字典，包含 network_trainable 与 risk_calibration_trainable 两个布尔值。

    失败条件：
        ValueError: 当阶段名不被支持时抛出。
    """
    if phase_name not in {"readout_warmup", "gate_alignment", "full_tuning"}:
        raise ValueError(f"unsupported training phase: {phase_name}")
    model._current_phase_name = str(phase_name)
    if not hasattr(model, "_phase_control_state"):
        model._phase_control_state = {}
    # PATCH P1 (v7): gate_alignment 阶段也解冻 backbone, 解消前 60 ep 训练预算亏.
    # readout_warmup 仍保持冻结以保留 readout-only warmup 设计契约.
    # 子代 G 报告: trainer.py:3265 原代码让 backbone 在 ep 21-60 完全 0 梯度,
    # 等价吃 60-70 ep 训练预算亏 vs LSTM 160 ep 全程满梯度.
    _network_trainable = phase_name in {"gate_alignment", "full_tuning"}
    _set_requires_grad_for_module(model.network, _network_trainable)
    # v3.1 head-shared: 共享 backbone 中的 gate 在 readout_warmup 阶段必须冻结，
    # gate_alignment / full_tuning 阶段解冻。
    _backbone = getattr(model, "output_backbone", None)
    if _backbone is not None:
        _set_requires_grad_for_module(_backbone, phase_name in {"gate_alignment", "full_tuning"})
    for head in getattr(model, "output_heads", {}).values():
        _set_requires_grad_for_module(head, True)
        if phase_name == "readout_warmup":
            # v3.1: head 已无 *_gate 子模块，仅剩 projection/residual_projection (保持可训练)。
            head_named_children = getattr(head, "named_children", None)
            if callable(head_named_children):
                for attribute_name, child_module in head_named_children():
                    if attribute_name.endswith("_gate"):
                        _set_requires_grad_for_module(child_module, False)  # 冻结该门控。
    _set_requires_grad_for_module(getattr(model, "risk_calibration", None), phase_name != "readout_warmup")  # warmup 阶段冻结 calibration。
    phase_contract = _build_phase_trainability_contract(model, phase_name=phase_name)  # 构建可训练、合约。
    return {  # 返回规范化的可训练状态。
        "network_trainable": bool(phase_contract["network_trainable"]),  # 共享网络是否可训练。
        "risk_calibration_trainable": bool(phase_contract["risk_calibration_trainable"]),  # risk_calibration 是否可训练。
    }


def _resolve_amp_state(train_cfg: Mapping[str, Any], runtime_device: str) -> dict[str, Any]:
    """解析混合精度训练（AMP）状态。

    ?CUDA 设备上默认启AMP，CPU 上始终禁用。
    支持的精度类型：float16（默认）。bfloat16。

    参数：
    train_cfg: 训练配置映射，可能包amp_enabled ?amp_dtype 键。
    runtime_device: 运行时设备名字符串（"cuda"?cuda:N" ?"cpu"）。

    返回值：
        字典，包requested（原始请求、）、enabled（是否启用）
        ?dtype_name（精度类型名，如 "float16" ?"bfloat16"）。
    """
    amp_requested = train_cfg.get("amp_enabled", DEVICE_AUTO)  # 读取 AMP 开关，默认自动。
    amp_dtype = str(train_cfg.get("amp_dtype", DEVICE_AUTO)).strip().lower()  # 读取精度类型，默认自动。
    if runtime_device.startswith(DEVICE_CUDA):  # CUDA 设备上才考虑启用 AMP。
        if amp_requested == DEVICE_AUTO:  # 自动模式。
            enabled = True  # CUDA 上默认启用。
        elif isinstance(amp_requested, bool):  # 布尔值直接使用，拒绝静默强转。
            enabled = amp_requested
        elif is_string_like(amp_requested):  # 字符串需归一化后枚举校验，非法、直接报错。
            normalized = str(amp_requested).strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                enabled = True
            elif normalized in {"false", "0", "no", "off"}:
                enabled = False
            else:
                raise ValueError(f"amp_enabled must be 'auto' or bool-like, got {amp_requested!r}")
        else:
            raise ValueError(f"amp_enabled must be 'auto' or bool-like, got {amp_requested!r}")
        if not enabled:  # 不启用时直接返回。
            return {"requested": amp_requested, "enabled": False, "dtype_name": None}  # 禁用状。
        dtype_name = "float16"  # 默认精度类型。
        if amp_dtype in {"bfloat16", "bf16"}:  # 用户请求 bfloat16。
            dtype_name = "bfloat16"  # 使用 bfloat16。
        return {"requested": amp_requested, "enabled": True, "dtype_name": dtype_name}  # 启用状。
    return {"requested": amp_requested, "enabled": False, "dtype_name": None}  # CPU 上始终禁AMP。





def _to_cpu_recursive(obj: Any) -> Any:
    """递归将张量移到 CPU, 确保 checkpoint 的 weights_only=True 加载兼容."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return type(obj)({k: _to_cpu_recursive(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_recursive(v) for v in obj)
    return obj


def _build_checkpoint_payload(

    *,

    model: Any,

    trainer_state: dict[str, Any],

    optimizer: Any,

    seed_report: dict[str, Any],

    best_epoch: int,

    best_loss: float,

    train_window_count: int,

    val_window_count: int,

    current_epoch: int | None = None,

    lr_scheduler: Any = None,

) -> dict[str, Any]:
    """构建检查点保存的完整载荷字典。

    包含模型配置、训练配置、优化器状、种子报告。
    朢epoch/损失、窗口计数和模型参数状。

    所有张量强制移至 CPU 后再保存，确保与 model_factory 接口一致。
    ``weights_only=True`` 加载路径兼容。

    参数：
        model: 液体模型实例。
        trainer_state: 训练器状态字典。
        optimizer: 优化器实例。
        seed_report: 种子设置报告字典。
        best_epoch: 本€浼epoch 绱㈠紩。
        best_loss: 最优验证损失。
        train_window_count: 训练窗口数量。
        val_window_count: 验证窗口数量。

    返回值：
        检查点载荷字典，可直接传给 torch.save。
    """

    network_cfg = dict(trainer_state["network"])  # 鎷疯礉缃戠粶配置。。

    network_cfg["hidden_dim"] = int(model.network.hidden_dim)  # 从模型实例读取实际隐藏维度。

    network_cfg["input_dim"] = int(model.network.input_dim or len(trainer_state["feature_order"]))  # 从模型实例读取实际输入维度。

    network_cfg.setdefault("output_heads", list(_OUTPUT_KEYS))  # 确保有输出头列表。

    return {

        "checkpoint_format": "liquid_real_v1",  # 检查点格式版本标识。

        "model_name": trainer_state["model_name"],  # 模型名称。

        "model_cfg": {  # 模型配置子字典。

            "feature_order": list(trainer_state["feature_order"]),  # 版瑰緛项哄簭。。

            "window": dict(trainer_state["window"]),  # 窗口配置。

            "network": network_cfg,  # 网络配置（含实际维度）。

        },

        "train_cfg": {  # 训练配置子字典。

            "train": dict(trainer_state["train"]),  # 训练参数。

            "seed": trainer_state["seed"],  # 随机种子。

            "deterministic": trainer_state["deterministic"],  # 确定性标志。

        },

        "optimizer": dict(trainer_state["optimizer"]),  # 优化器配。

        "optimizer_state": _to_cpu_recursive(optimizer.state_dict()),  # 优化器状态字典，张量强制 CPU。
        # audit #20 M1: 持久化 lr_scheduler.state_dict() — optimizer_state 只含 param_group['lr']
        # 当前快照, 不含 CosineAnnealingLR 的 base_lrs / last_epoch / etas 等退火进度.
        # retry7 重启若不 load_state_dict 会丢退火进度 (last_epoch 复位 → lr 又从 baseline lr 开始, 退火曲线被打断).
        "lr_scheduler_state_dict": (
            _to_cpu_recursive(lr_scheduler.state_dict()) if lr_scheduler is not None else None
        ),
        "seed_report": dict(seed_report),  # 种子设置报告。
        "current_epoch": None if current_epoch is None else int(current_epoch),  # 当前训练态对应的 epoch。
        "best_epoch": int(best_epoch),  # 本€浼epoch 绱㈠紩。
        "best_loss": coerce_finite_scalar(best_loss, name="best_loss"),  # 朢、优验证损失：写入 checkpoint 前强制有限、校验，阻断 NaN/Inf 静默穿、至 save/load 边界。

        "train_window_count": int(train_window_count),  # 训练窗口数量。

        "val_window_count": int(val_window_count),  # 验证窗口数量。

        "model_state": _to_cpu_recursive(model.state_dict()),  # 模型参数状、字典，张量强制 CPU。

    }





def train_one_epoch(train_windows, val_windows, epoch_index: int, *, model=None, optimizer=None, amp_state: Mapping[str, Any] | None = None, grad_scaler: Any = None) -> float:
    """执行一个完整的训练 epoch，返回平均训练损失。

    将材料化样本按序列长度分组，同组样本可以堆叠成批。
    做前向和反向传播。每mini-batch 做一次梯度更新。

    C-3: amp_state["enabled"]=True 时前向用 torch.autocast 包裹，
    反向/更新走 GradScaler (scaler.scale → unscale_ → clip → step → update)。

    参数：
        train_windows: 训练窗口列表（未使用，仅用于校验）。
        val_windows: 验证窗口列表（未使用，仅用于校验）。
        epoch_index: 当前 epoch 索引（从 1 开始）。
        model: 液体模型实例，必须提供。
        optimizer: 优化器实例，必须提供。
        amp_state: AMP 状态字典（_resolve_amp_state 返回值），None = 禁用。
        grad_scaler: torch.amp.GradScaler 实例，amp_state 启用时必须提供。

    返回值：
    ?epoch 的平均训练损失。

    失败条件。
    TypeError: ?epoch_index 不是整数时抛出。
        ValueError: 褰epoch_index < 1。乵odel 鎴optimizer 个None。
            或训验证窗口为空时抛出。
    """
    _amp_enabled = bool(amp_state and amp_state.get("enabled"))
    if _amp_enabled and grad_scaler is None:
        raise ValueError("grad_scaler must be provided when amp_state['enabled'] is True (C-3)")

    _loss_for_length_group_fn = _get_loss_compute()[1]  # unpack deferred import
    if not is_integer(epoch_index):

        raise TypeError("epoch_index must be an integer.")

    if epoch_index < 1:

        raise ValueError("epoch_index must be at least 1.")

    if model is None:

        raise ValueError("model must be provided for real Liquid training")

    if optimizer is None:

        raise ValueError("optimizer must be provided for real Liquid training")



    train_count = len(train_windows)

    val_count = len(val_windows)

    if train_count < 1:

        raise ValueError("train_windows must contain at least one window.")

    if val_count < 1:

        raise ValueError("val_windows must contain at least one window.")



    batch_size = int(getattr(model, "_train_batch_size", 0) or 0)

    if batch_size < 1:

        batch_size = train_count

    model.train()

    cached_materialized = getattr(model, "_cached_train_windows", None)

    if not cached_materialized:

        cache_device = _resolve_sample_cache_device(model)  # §13.8: 与 model 同设备，释放 CPU 内存
        cached_materialized = _move_materialized_samples(
            _materialize_samples(
                train_windows,
                tail_selection_observation_coeff=coerce_finite_scalar(
                    getattr(model, "_tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
                    name="tail_selection_observation_coeff",
                    min_value=0.0,
                ),
            ),
            cache_device,
        )
        setattr(model, "_cached_train_windows", cached_materialized)

    loss_sum = 0.0

    loss_count = 0

    grouped: dict[int, list[tuple[dict[str, Any], torch.Tensor, int, float]]] = {}

    for sample in cached_materialized:

        grouped.setdefault(sample[2], []).append(sample)

    for sequence_group in grouped.values():

        for start_index in range(0, len(sequence_group), batch_size):

            end_index = min(len(sequence_group), start_index + batch_size)

            batch_samples = sequence_group[start_index:end_index]

            optimizer.zero_grad(set_to_none=True)

            if _amp_enabled:
                # C-3: AMP 前向用 autocast 包裹（fp16/bf16 自动混合精度）
                _dtype = torch.float16 if amp_state.get("dtype_name") == "float16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=_dtype):
                    loss_value = _loss_for_length_group_fn(model, batch_samples)
            else:
                loss_value = _loss_for_length_group_fn(model, batch_samples)

            if _amp_enabled:
                # C-3: 缩放 loss → backward → unscale → clip → scaled step → update
                grad_scaler.scale(loss_value).backward()
                gradient_clip_max_norm = coerce_finite_scalar(
                    getattr(model, "_gradient_clip_max_norm", 1.0),
                    name="model._gradient_clip_max_norm",
                    min_value=0.0,
                )
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss_value.backward()
                # 梯度裁剪：防止极端NLOS 样本和阶段切换导致梯度爆炸。
                # 阈、从 YAML train.gradient_clip_max_norm 读取，与文档 trainer.md §5.4 对齐。
                # D5：与 lstm/trainer.py 同名函数口径对齐，统丢coerce_finite_scalar 中心入口。
                # float() 会把 NaN/Inf/负、静默穿透到 clip_grad_norm_，导致NaN 梯度无声泄漏。
                gradient_clip_max_norm = coerce_finite_scalar(
                    getattr(model, "_gradient_clip_max_norm", 1.0),
                    name="model._gradient_clip_max_norm",
                    min_value=0.0,
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
                optimizer.step()

            # D7：与 lstm/trainer.py.train_one_epoch 口径对齐，统一。item() 取标量，
            # 避免 float(tensor) 在不torch 版本下的隐式行为差异。
            loss_sum += float(loss_value.detach().item())

            loss_count += 1

    return loss_sum / float(loss_count)



