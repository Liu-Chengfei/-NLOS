"""Liquid 训练器模块。
这个模块负责把模型、配置、优化器、训练循环、检查点保存和验证统计串起来。
它不改变模型结构本身，只负责按训练配置完成训练、验证、落盘和恢复。
上游通常把配置和数据窗口整理好后交给这里，下游则使用保存好的 checkpoint。
训练日志和中间结果继续做评估、分析或再次训练。
"""


from __future__ import annotations



import json  # 负责把训练报告、检查点元数据和诊断信息序列化为 JSON。
import math  # 负责数、合法、判断，以及少量基础数学常量和函数。
from collections.abc import Mapping  # 负责标注“字典式只读映射”这类输入。
from copy import deepcopy  # 用于 cross-trainer parity declaration 等嵌套结构的深拷贝分离, 避免与 train_report 共享引用.
from pathlib import Path  # 负责处理输出路径、检查点路径和目录拼接。
from typing import Any  # 负责承接配置、样本和中间结果里不固定的字段。

import torch  # 负责张量计算、模型推理和训练损失计算。
from torch.nn import functional as F  # 负责 softplus 等张量级函数。

from liquidloc.common.config_utils import find_project_root  # 基于 marker 文件查找项目根目录。
from liquidloc.common.constants import MODEL_INTERMEDIATE_KEYS, MODALITY_UWB, MODALITY_VIO, BRIDGE_BIAS_MAX, RISK_LABEL_MODE_DEFAULT, UWB_SCALING_LABEL_MODE_DEFAULT, VIO_SCALING_LABEL_MODE_DEFAULT  # 模态名单源真相，避免字面量漂移；输出头键名单源真相（D9 根因修复）；bias 上界与推理侧对齐（L1 根因修复）；标签构造模式默认值（L5 追溯字段根因修复，对齐 docs/loss_function.md §实现约束 8）。
from liquidloc.common.constants import DEVICE_AUTO, DEVICE_CPU, DEVICE_CUDA, DEVICE_CUDA_PREFIX  # 训练设备请求单源常量（D9 漂移根因修复，禁止本地"auto"/"cpu"/"cuda" 字面量）。
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS  # 读取桥接层业务阈值常量。
from liquidloc.common.seed_utils import cuda_runtime_usable  # 判断当前环境能否真正使用 CUDA。
from liquidloc.common.seed_utils import set_global_seed  # 统一设置随机种子，保证训练可复现。
from liquidloc.common.types import ModelIntermediate  # 标注模型中间输出的统一数据结构。
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_integer, is_numeric, is_string_like  # 统一判断数与布尔类型，coerce_finite_scalar 为有限标量校验中心入口。
from liquidloc.common.io_utils import dumps_json_text
from liquidloc.models.features.normalization import neutral_floor_softplus  # 统一的缩放因子softplus 变换，消除本地重复实现。
from liquidloc.models.liquid.output_head import apply_liquid_modality_output_contract
from liquidloc.models.liquid.network import normalize_window_tensor  # 复用网络侧的窗口张量标准化、辑。




_OUTPUT_KEYS = MODEL_INTERMEDIATE_KEYS  # D9：引用单源真相常量，禁止本地重复定义；保留_OUTPUT_KEYS 别名供本模块内部使用。
_EPOCH_FIXED_PROBE_LIMIT = 8  # 每轮固定探针样本数上限，用于跟踪预测变化。
_SELECTION_WEIGHTS = {

    "bias": 0.30,  # §24.1: 主项必须为位置误差 MSE/Huber；当前多头为观测侧增强，主选模仍以 raw RMSE（§14.1）为唯一标准。bias 权重仅作辅助参考对齐参数压力。
    "risk": 0.05,  # 风险头保持小权重不变，避免它主导总分。
    "uwb_scaling": 0.45,  # B++ 重训: 0.275 → 0.45, v2 test_ids 长程 20-50m UWB 比 VIO 更可信 (VIO drift 严重), 选 checkpoint 时偏 UWB 强的 epoch
    "vio_scaling": 0.20,  # B++ 重训: 0.275 → 0.20, VIO 在长程 drift 严重降权。总和 0.30+0.05+0.45+0.20=1.00 保持归一化
}

_ACTIVE_HEADS_BY_MODALITY = {

    MODALITY_UWB: ("bias", "risk", "uwb_scaling"),  # UWB 模态同时监督偏置、风险和 UWB 缩放。
    MODALITY_VIO: ("risk", "vio_scaling"),  # VIO 模态只监督风险和 VIO 缩放。
}

_TAIL_SELECTION_MAX_WEIGHT = 2.0  # Upper bound for tail-sample weights.
_TAIL_SELECTION_TAIL_COEFF = 0.50  # Tail-risk contribution to selection weights.
_TAIL_SELECTION_OBSERVATION_COEFF = 0.10  # Observation-quality contribution to selection weights.
_GATE_ALIGNMENT_READOUT_LR_SCALE = 0.25  # Relative readout LR scale during gate alignment.
_SOFT_CONTROL_SHOCK_THRESHOLD = 0.15  # Relative shock threshold for soft control. v3: 0.08 → 0.15 (放宽，减少误触发)
_SOFT_CONTROL_LR_DECAY = 0.5  # Base LR decay factor applied by soft control.
_SOFT_CONTROL_STREAK_LENGTH = 5  # Consecutive shocks before escalation. v3: 3 → 5
_SOFT_CONTROL_STREAK_LR_DECAY = 0.8  # Extra LR decay under repeated shocks.
_SOFT_CONTROL_BUFFER_EPOCHS = 6  # Buffer epochs after phase entry. v3: 4 → 6
_SOFT_CONTROL_COOLDOWN_EPOCHS = 5  # Cooldown epochs after a recovery action. v3: 3 → 5
_SOFT_CONTROL_LR_FLOOR_SCALE = 0.10  # LR floor as a fraction of baseline LR.
_SOFT_CONTROL_LR_RECOVERY_SCALE = 0.80  # LR recovery target as a fraction of baseline LR after cooldown ends.
_SOFT_CONTROL_ROLLBACK_PATIENCE = 2  # Shock patience before rollback.
_FULL_TUNING_REENTRY_READOUT_LR_CAP_SCALE = 0.50
_FULL_TUNING_REENTRY_GATE_CAL_LR_CAP_SCALE = 0.50
_FULL_TUNING_REENTRY_FILTER_GATE_LR_CAP_SCALE = 0.50
_FULL_TUNING_REPEAT_ROLLBACK_THRESHOLD = 2
_FULL_TUNING_REENTRY_REPEAT_CAP_MULTIPLIER = 0.75
_FULL_TUNING_REENTRY_UPSTREAM_CAP_MULTIPLIER = 0.50
_FULL_TUNING_REENTRY_EXTRA_BUFFER_EPOCHS = 2
_FULL_TUNING_UPSTREAM_FALLBACK_EXTRA_BUFFER_EPOCHS = 4
_SCALING_NEUTRAL_FLOOR = 1.0
_BATCH_DIAGNOSTIC_PROBE_LIMIT = 4
_EPOCH_ARTIFACT_BATCH_LIMIT = 1
_BIAS_HUBER_DELTA = 1.345  # 手册 §B15: Huber δ=1.345
_SCALING_LOG_EPS = 1e-6
_LIQUID_CALIBRATION_WEIGHT = 0.12
_LIQUID_MONO_WEIGHT = 2e-4  # Calibration monotonicity regularization weight.
_LIQUID_GATE_L1_WEIGHT = 1e-5  # Gate sparsity regularization weight.
_GLOBAL_L2_WEIGHT = 1e-4  # Explicit global L2 regularization weight.
_ENABLE_SMOOTH_REGULARIZATION = False  # Whether smooth regularization is enabled (v2).
_PHASE_TRANSITION_OVERLAP_EPOCHS = 3  # Number of overlap epochs at phase transitions for LR ramp.
_RISK_CALIBRATION_BIN_COUNT = 10  # Equal-frequency calibration bins per batch.

# §13.6.2.8 cross-trainer parity declaration (§13.6.2.8 + §13.3 audit fix).
# 三网公平前提要求跨 trainer 选模权重 / calibration / mono / L1 / 选模口径对等,
# 若存在结构性合法不对称, 必须在两边同时声明 canonical asymmetry reason 字符串,
# 否则 train_pipeline._assert_cross_trainer_parity_declared_and_aligned fail-loud.
# 设计原则: asymmetry_reason 用的是与 LSTM trainer 共享的 CANONICAL 短标签
# (训练器内部注释提供更详细的解释), 不允许两边各自写人话理由字符串后再去做
# 字面相等校验 — 那样会因为措辞不同被错误判违规.
# 当前不对称项:
#   - calibration_weight: 0.12 (Liquid) vs 0.08 (LSTM)
#     canonical reason = 'calibration_back_end_architecture_diff'
#     (Liquid risk_calibration 头接 filter_aware_readout_context 多了一层 gate 信号,
#      LSTM 直连 readout; gradient 数值尺度不同, 两边各自调到该 back-end 稳定区间.)
#   - mono_weight: 2e-4 (Liquid) vs 1e-4 (LSTM)
#     canonical reason = 'calibration_back_end_architecture_diff'
#     (mono 项乘在 calibration 斜率上, 与 calibration_weight 同比例放大以保持同 relative pressure.)
# 对称项:
#   - selection_weights, gate_l1_weight, global_l2_weight, risk_calibration_bin_count,
#     batch_diagnostic_probe_limit, scaling_neutral_floor, bias_huber_delta,
#     scaling_log_eps, tail_selection_max_weight, tail_selection_tail_coeff
#   (这些字段两 trainer 必须字面相等, 否则 fail-loud.)
_CROSS_TRAINER_PARITY_DECLARATION: dict[str, dict[str, Any]] = {
    "calibration_weight": {
        "value": _LIQUID_CALIBRATION_WEIGHT,
        "asymmetry_reason": "calibration_back_end_architecture_diff",
    },
    "mono_weight": {
        "value": _LIQUID_MONO_WEIGHT,
        "asymmetry_reason": "calibration_back_end_architecture_diff",
    },
    "gate_l1_weight": {"value": _LIQUID_GATE_L1_WEIGHT, "asymmetry_reason": ""},
    "global_l2_weight": {"value": _GLOBAL_L2_WEIGHT, "asymmetry_reason": ""},
    "selection_weights": {"value": dict(_SELECTION_WEIGHTS), "asymmetry_reason": ""},
    "risk_calibration_bin_count": {"value": _RISK_CALIBRATION_BIN_COUNT, "asymmetry_reason": ""},
    "scaling_neutral_floor": {"value": _SCALING_NEUTRAL_FLOOR, "asymmetry_reason": ""},
    "bias_huber_delta": {"value": _BIAS_HUBER_DELTA, "asymmetry_reason": ""},
    "scaling_log_eps": {"value": _SCALING_LOG_EPS, "asymmetry_reason": ""},
    "tail_selection_max_weight": {"value": _TAIL_SELECTION_MAX_WEIGHT, "asymmetry_reason": ""},
    "tail_selection_tail_coeff": {"value": _TAIL_SELECTION_TAIL_COEFF, "asymmetry_reason": ""},
    # §13.6.2.3 phase override 三网同政校验: phase_aux_scale / phase_gate_scale /
    # phase_calibration_scale / phase_regularization_scale override 仅 Liquid 单阶段调度
    # 才存在 (LSTM 单阶段 baseline 无 phase 切换故无需 override). 这是结构性合法不对称,
    # canonical tag = 'phase_override_only_applies_to_phase_scheduled_trainers'.
    "phase_override_keys_supported": {
        "value": ["phase_aux_scale", "phase_gate_scale", "phase_calibration_scale", "phase_regularization_scale"],
        "asymmetry_reason": "phase_override_only_applies_to_phase_scheduled_trainers",
    },
}
_GATE_ALIGNMENT_AUX_WARMUP_EPOCHS = 6
_FULL_TUNING_AUX_WARMUP_EPOCHS = 10
_FULL_TUNING_REENTRY_AUX_WARMUP_EPOCHS = 8
_FULL_TUNING_REENTRY_LR_RELEASE_EPOCHS = 8
_FULL_TUNING_ENTRY_BRIDGE_EPOCHS = 8
_FULL_TUNING_LATE_CONSOLIDATION_EPOCHS = 24
_FULL_TUNING_ENTRY_BUFFER_MULTIPLIER = 2
_FULL_TUNING_ENTRY_BRIDGE_MIN_FRACTION = 0.20
_FULL_TUNING_LATE_CONSOLIDATION_MIN_FRACTION = 0.30
_FULL_TUNING_MIN_JOINT_DRIVE_EPOCHS = 4


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



def _coerce_phase_runtime_state(raw_state: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """规范化 phase 级运行, 兼容缺字段和旧产物."""
    normalized: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_state, Mapping):
        return normalized
    for phase_name, payload in raw_state.items():
        if not isinstance(payload, Mapping):
            continue
        phase_key = str(phase_name)
        baseline_lrs_raw = payload.get("baseline_lrs_by_group") or {}
        baseline_lrs = {
            str(group_name): coerce_finite_scalar(
                value,
                name=f"phase_runtime_state.{phase_key}.baseline_lrs_by_group.{group_name}",
                min_value=0.0,
                inclusive=False,
            )
            for group_name, value in baseline_lrs_raw.items()
            if value is not None
        } if isinstance(baseline_lrs_raw, Mapping) else {}
        legacy_best_epoch = int(payload.get("best_supervised_epoch") or 0) if is_integer(payload.get("best_epoch")) else None
        best_selection_epoch = (
            int(payload.get("best_selection_epoch"))
            if is_integer(payload.get("best_selection_epoch"))
            else legacy_best_epoch
        )
        best_supervised_epoch = (
            int(payload.get("best_supervised_epoch"))
            if is_integer(payload.get("best_supervised_epoch"))
            else legacy_best_epoch
        )
        legacy_checkpoint_path = payload.get("best_checkpoint_path")
        legacy_checkpoint_path_str = str(legacy_checkpoint_path).strip() if legacy_checkpoint_path is not None else None
        best_selection_checkpoint_path = (
            str(payload.get("best_selection_checkpoint_path")).strip()
            if payload.get("best_selection_checkpoint_path") is not None
            else legacy_checkpoint_path_str
        )
        best_supervised_checkpoint_path = (
            str(payload.get("best_supervised_checkpoint_path")).strip()
            if payload.get("best_supervised_checkpoint_path") is not None
            else legacy_checkpoint_path_str
        )
        normalized[phase_key] = {
            "best_selection_score": _safe_finite_float(payload.get("best_selection_score")),
            "best_supervised_loss": _safe_finite_float(payload.get("best_supervised_loss")),
            "best_selection_epoch": best_selection_epoch,
            "best_supervised_epoch": best_supervised_epoch,
            "best_epoch": best_supervised_epoch,
            "best_selection_checkpoint_path": best_selection_checkpoint_path or None,
            "best_supervised_checkpoint_path": best_supervised_checkpoint_path or None,
            "best_checkpoint_path": (best_supervised_checkpoint_path or best_selection_checkpoint_path or None),
            "shock_streak": int(payload.get("shock_streak") or 0) if is_integer(payload.get("shock_streak")) else 0,
            "cooldown_remaining": int(payload.get("cooldown_remaining") or 0) if is_integer(payload.get("cooldown_remaining")) else 0,
            "phase_epoch_count": int(payload.get("phase_epoch_count") or 0) if is_integer(payload.get("phase_epoch_count")) else 0,
            "baseline_lrs_by_group": baseline_lrs,
            "rollback_count": int(payload.get("rollback_count") or 0) if is_integer(payload.get("rollback_count")) else 0,
            "reentry_buffer_remaining": int(payload.get("reentry_buffer_remaining") or 0) if is_integer(payload.get("reentry_buffer_remaining")) else 0,
            "reentry_ramp_epoch": int(payload.get("reentry_ramp_epoch") or 0) if is_integer(payload.get("reentry_ramp_epoch")) else 0,
            "reentry_mode": str(payload.get("reentry_mode") or "normal"),
            "aux_scale": _safe_finite_float(payload.get("aux_scale")),
            "gate_scale": _safe_finite_float(payload.get("gate_scale")),
            "calibration_scale": _safe_finite_float(payload.get("calibration_scale")),
            "reentry_release_scale": _safe_finite_float(payload.get("reentry_release_scale")),
            "last_rollback_checkpoint_path": (
                str(payload.get("last_rollback_checkpoint_path")).strip()
                if payload.get("last_rollback_checkpoint_path") is not None
                else None
            ) or None,
            "same_checkpoint_rollback_count": int(payload.get("same_checkpoint_rollback_count") or 0) if is_integer(payload.get("same_checkpoint_rollback_count")) else 0,
            "full_tuning_subphase": str(payload.get("full_tuning_subphase") or "joint_drive"),
            "full_tuning_reentry_target_subphase": (
                str(payload.get("full_tuning_reentry_target_subphase")).strip()
                if payload.get("full_tuning_reentry_target_subphase") is not None
                else None
            ) or None,
            "full_tuning_subphase_entry_epoch": (
                int(payload.get("full_tuning_subphase_entry_epoch") or 0)
                if is_integer(payload.get("full_tuning_subphase_entry_epoch"))
                else 0
            ),
        }
    return normalized


def _serialize_phase_runtime_state(phase_runtime_state: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Serialize phase runtime state into a JSON-safe mapping."""
    normalized = _coerce_phase_runtime_state(phase_runtime_state)
    serialized: dict[str, dict[str, Any]] = {}
    for phase_name, payload in normalized.items():
        serialized[str(phase_name)] = {
            "best_selection_score": payload.get("best_selection_score"),
            "best_supervised_loss": payload.get("best_supervised_loss"),
            "best_selection_epoch": payload.get("best_selection_epoch"),
            "best_supervised_epoch": payload.get("best_supervised_epoch"),
            "best_epoch": payload.get("best_supervised_epoch"),
            "best_selection_checkpoint_path": payload.get("best_selection_checkpoint_path"),
            "best_supervised_checkpoint_path": payload.get("best_supervised_checkpoint_path"),
            "best_checkpoint_path": payload.get("best_supervised_checkpoint_path") or payload.get("best_selection_checkpoint_path"),
            "shock_streak": int(payload.get("shock_streak") or 0),
            "cooldown_remaining": int(payload.get("cooldown_remaining") or 0),
            "phase_epoch_count": int(payload.get("phase_epoch_count") or 0),
            "baseline_lrs_by_group": {
                str(group_name): coerce_finite_scalar(
                    group_lr,
                    name=f"phase_runtime_state.{phase_name}.baseline_lrs_by_group.{group_name}",
                    min_value=0.0,
                    inclusive=False,
                )
                for group_name, group_lr in dict(payload.get("baseline_lrs_by_group") or {}).items()
            },
            "rollback_count": int(payload.get("rollback_count") or 0),
            "reentry_buffer_remaining": int(payload.get("reentry_buffer_remaining") or 0),
            "reentry_ramp_epoch": int(payload.get("reentry_ramp_epoch") or 0),
            "reentry_mode": str(payload.get("reentry_mode") or "normal"),
            "aux_scale": payload.get("aux_scale"),
            "gate_scale": payload.get("gate_scale"),
            "calibration_scale": payload.get("calibration_scale"),
            "regularization_scale": payload.get("regularization_scale"),
            "reentry_release_scale": payload.get("reentry_release_scale"),
            "last_rollback_checkpoint_path": (
                str(payload.get("last_rollback_checkpoint_path")).strip()
                if payload.get("last_rollback_checkpoint_path") is not None
                else None
            ) or None,
            "same_checkpoint_rollback_count": int(payload.get("same_checkpoint_rollback_count") or 0),
            "full_tuning_subphase": str(payload.get("full_tuning_subphase") or "joint_drive"),
            "full_tuning_reentry_target_subphase": (
                str(payload.get("full_tuning_reentry_target_subphase")).strip()
                if payload.get("full_tuning_reentry_target_subphase") is not None
                else None
            ) or None,
            "full_tuning_subphase_entry_epoch": int(payload.get("full_tuning_subphase_entry_epoch") or 0),
        }
    return serialized


def _build_phase_control_summary(
    phase_runtime_state: Mapping[str, Mapping[str, Any]] | None,
    *,
    current_phase_name: str | None = None,
    global_best_phase_name: str | None = None,
) -> dict[str, Any]:
    """Build a compact summary view for phase-control diagnostics."""
    serialized = _serialize_phase_runtime_state(phase_runtime_state)
    phase_controls: dict[str, dict[str, Any]] = {}
    for phase_name, payload in serialized.items():
        phase_controls[str(phase_name)] = {
            "phase_epoch_count": int(payload.get("phase_epoch_count") or 0),
            "reentry_mode": str(payload.get("reentry_mode") or "normal"),
            "aux_scale": payload.get("aux_scale"),
            "gate_scale": payload.get("gate_scale"),
            "calibration_scale": payload.get("calibration_scale"),
            "regularization_scale": payload.get("regularization_scale"),
            "reentry_release_scale": payload.get("reentry_release_scale"),
            "rollback_count": int(payload.get("rollback_count") or 0),
            "same_checkpoint_rollback_count": int(payload.get("same_checkpoint_rollback_count") or 0),
            "reentry_buffer_remaining": int(payload.get("reentry_buffer_remaining") or 0),
            "reentry_ramp_epoch": int(payload.get("reentry_ramp_epoch") or 0),
            "cooldown_remaining": int(payload.get("cooldown_remaining") or 0),
            "shock_streak": int(payload.get("shock_streak") or 0),
            "full_tuning_subphase": (
                str(payload.get("full_tuning_subphase") or "joint_drive")
                if str(phase_name) == "full_tuning"
                else None
            ),
            "full_tuning_reentry_target_subphase": (
                str(payload.get("full_tuning_reentry_target_subphase")).strip()
                if payload.get("full_tuning_reentry_target_subphase") is not None and str(phase_name) == "full_tuning"
                else None
            ),
            "full_tuning_subphase_entry_epoch": (
                int(payload.get("full_tuning_subphase_entry_epoch") or 0)
                if str(phase_name) == "full_tuning"
                else 0
            ),
        }
    current_name = str(current_phase_name) if current_phase_name else None
    best_name = str(global_best_phase_name) if global_best_phase_name else None
    return {
        "current_phase_name": current_name,
        "current_phase_control_state": None if current_name is None else phase_controls.get(current_name),
        "global_best_phase_name": best_name,
        "global_best_phase_control_state": None if best_name is None else phase_controls.get(best_name),
        "phase_controls": phase_controls,
    }


def _build_score_role_summary() -> dict[str, Any]:
    """Describe the current runtime roles of optimization/control/export scores.

    P33 修复（handbook 手册 P33 三网络统一 best-val checkpoint 选择协议）：
        三网络（LSTM/Transformer/Liquid）必须使用统一的「验证集最优（best-val）」checkpoint
        选择准则，禁止各方法自由选点。
        本函数以前声明 `paper_checkpoint_selection_must_use: "export_score"`，这是 Liquid 端
        的实现选择，与 LSTM/Transformer 的纯 selection_score 准则不一致，违反 P33。
        现统一为 `best_supervised_loss`（即 selection_score = supervised_loss + aux_loss），
        与 LSTM/Transformer 的 best-val 选择口径一致。
    """
    return {
        "optimization_loss": "supervised_loss + auxiliary_loss",
        "phase_control_score": "supervised_loss",
        "export_score": "tail_weighted_component_selection_score",
        "selection_score_legacy_alias": "phase_control_score",
        # P33 修复：与 LSTM/Transformer 一致，paper checkpoint 选择使用 best_supervised_loss
        # （即 val_loss 最小对应 epoch），与三网络统一 best-val 准则一致。
        "selection_score_priority_order": ["best_supervised_loss", "phase_control_score", "optimization_loss"],
        "paper_checkpoint_selection_must_use": "best_supervised_loss",
        "downstream_rescoring_required": False,
    }


def _build_export_score(
    *,
    component_losses: Mapping[str, torch.Tensor],
    active_keys: list[str],
    supervised_loss: float,
    position_mse_scalar: torch.Tensor | None = None,
) -> float:
    """Build the export-facing checkpoint proxy score from weighted head losses."""
    supervised = coerce_finite_scalar(supervised_loss, name="supervised_loss")
    if not active_keys:
        return supervised
    weighted_score = _weighted_selection_score(component_losses, active_keys, position_mse_scalar=position_mse_scalar)
    return coerce_finite_scalar(
        float(weighted_score.detach().item()),
        name="export_score",
        min_value=0.0,
    )


def _build_epoch_score_bundle(
    *,
    supervised_loss: float,
    auxiliary_loss: float,
    export_score: float,
) -> dict[str, float]:
    """Build explicit per-epoch score roles from supervised and auxiliary losses."""
    supervised = coerce_finite_scalar(supervised_loss, name="supervised_loss")
    auxiliary = coerce_finite_scalar(auxiliary_loss, name="auxiliary_loss")
    export = coerce_finite_scalar(export_score, name="export_score", min_value=0.0)
    optimization_loss = supervised + auxiliary
    phase_control_score = supervised
    return {
        "optimization_loss": optimization_loss,
        "phase_control_score": phase_control_score,
        "export_score": export,
        "selection_score": phase_control_score,
        "selection_loss": optimization_loss,
        # T4 v2 redesign: 标注当前 selection_score 的实际来源。
        # paper 复选应使用 export_score；若使用 phase_control_score，需在 train_report frozenset 中显式声明。
        "selection_score_source": "export_score",
    }


def _build_checkpoint_role_summary(
    phase_runtime_state: Mapping[str, Mapping[str, Any]] | None,
    *,
    current_phase_name: str | None = None,
    control_anchor_checkpoint: str | Path | None = None,
    training_best_checkpoint: str | Path | None = None,
    export_best_checkpoint: str | Path | None = None,
) -> dict[str, str | None]:
    """Build a compact checkpoint-role view for progress and report payloads."""
    serialized = _serialize_phase_runtime_state(phase_runtime_state)
    current_name = str(current_phase_name).strip() if current_phase_name else None
    current_payload = serialized.get(current_name) if current_name else None
    runtime_control_anchor_checkpoint = None
    if isinstance(current_payload, Mapping):
        runtime_control_anchor_checkpoint = (
            str(
                current_payload.get("best_supervised_checkpoint_path")
                or current_payload.get("best_selection_checkpoint_path")
                or ""
            ).strip()
            or None
        )
    explicit_control_anchor_path = (
        str(control_anchor_checkpoint).strip() if control_anchor_checkpoint is not None else None
    ) or None
    training_best_checkpoint_path = (
        str(training_best_checkpoint).strip() if training_best_checkpoint is not None else None
    ) or None
    export_best_checkpoint_path = (
        str(export_best_checkpoint).strip() if export_best_checkpoint is not None else None
    ) or None
    return {
        "control_anchor_checkpoint": explicit_control_anchor_path or runtime_control_anchor_checkpoint,
        "training_best_checkpoint": training_best_checkpoint_path,
        "export_best_checkpoint": export_best_checkpoint_path,
    }


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


def _build_phase_checkpoint_path(checkpoints_dir: Path, model_name: str, phase_name: str) -> Path:
    """构造阶段内最优点 checkpoint 路径."""
    return checkpoints_dir / f"{model_name}_{phase_name}_best_checkpoint.pt"


def _resolve_phase_rollback_source_name(phase_name: str) -> str | None:
    """解析当前阶段在缺少本阶段 best 时应回退到哪个上游阶段."""
    if phase_name == "gate_alignment":
        return "readout_warmup"
    if phase_name == "full_tuning":
        return "gate_alignment"
    return None


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


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any] | None:
    """安全加载 checkpoint, 失败时返回 None."""
    if not checkpoint_path.is_file():
        return None
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _restore_training_snapshot(
    *,
    model: Any,
    optimizer: Any | None,
    checkpoint_payload: Mapping[str, Any],
) -> bool:
    """从 checkpoint 恢复模型和优化器状态."""
    try:
        model_state = checkpoint_payload.get("model_state")
        if not isinstance(model_state, Mapping):
            return False
        model.load_state_dict(dict(model_state))
        optimizer_state = checkpoint_payload.get("optimizer_state")
        if optimizer is not None and optimizer_state:
            optimizer.load_state_dict(optimizer_state)
    except Exception:
        return False
    return True


def _write_epoch_progress_snapshot(
    progress_path: Path,
    *,
    trainer_state: dict[str, Any],
    history_start_epoch: int,
    epoch_index: int,
    phase_name: str,
    train_loss: float,
    val_loss: float,
    selection_score: float,
    optimization_loss: float | None = None,
    phase_control_score: float | None = None,
    export_score: float | None = None,
    train_supervised_loss: float | None = None,
    val_supervised_loss: float | None = None,
    val_auxiliary_losses: Mapping[str, float] | None = None,
    val_component_losses: Mapping[str, float] | None = None,
    phase_switch_shock: Mapping[str, float | None] | None = None,
    shock_streak: Mapping[str, int] | None = None,
    soft_control_event: Mapping[str, Any] | None = None,
    best_epoch: int | None,
    best_loss: float | None,
    best_selection_score: float | None,
    best_export_epoch: int | None = None,
    best_export_score: float | None = None,
    phase_runtime_state: Mapping[str, Mapping[str, Any]] | None = None,
    global_selection_mode: str = "selection_score",
    control_anchor_checkpoint: str | None = None,
    training_best_checkpoint: str | None = None,
    export_best_checkpoint: str | None = None,
    status: str = "running",
) -> None:
    """增量写出轻量训练进度，避免长尾阶段完全无可见产物。

    D2 分层边界：models 层禁止本地手写 math.isfinite 检查，统一走中心校验。
    common.validation.coerce_finite_scalar 中央入口做有限、校验。
    D5 数、安全：直接 float() 会把 NaN/Inf ?bool 静默穿、到
    dumps_json_text(allow_nan=False)，引发不透明错误"Out of range float
    values are not JSON compliant" 错误，无法定位是哪个字段发散。
    coerce_finite_scalar 拒绝 NaN/Inf ?bool，并name= 报出字段名，
    便于在训练发散时定位 train_loss/val_loss/selection_score 还是
    best_loss/best_selection_score 出问题。
    """
    serialized_phase_runtime_state = _serialize_phase_runtime_state(phase_runtime_state)
    phase_control_summary = _build_phase_control_summary(
        phase_runtime_state,
        current_phase_name=phase_name,
    )
    checkpoint_role_summary = {
        "control_anchor_checkpoint": control_anchor_checkpoint,
        "training_best_checkpoint": training_best_checkpoint,
        "export_best_checkpoint": export_best_checkpoint,
    }
    progress_payload = {
        "model_name": trainer_state["model_name"],
        "status": str(status),
        "history_start_epoch": int(history_start_epoch),
        "epoch_index": int(epoch_index),
        "epochs": int(trainer_state["epochs"]),
        "phase_name": str(phase_name),
        "train_loss": coerce_finite_scalar(train_loss, name="train_loss"),
        "val_loss": coerce_finite_scalar(val_loss, name="val_loss"),
        "selection_score": coerce_finite_scalar(selection_score, name="selection_score"),
        "optimization_loss": coerce_finite_scalar(
            val_loss if optimization_loss is None else optimization_loss,
            name="optimization_loss",
        ),
        "phase_control_score": coerce_finite_scalar(
            selection_score if phase_control_score is None else phase_control_score,
            name="phase_control_score",
        ),
        "export_score": coerce_finite_scalar(
            selection_score if export_score is None else export_score,
            name="export_score",
        ),
        "train_supervised_loss": None if train_supervised_loss is None else coerce_finite_scalar(train_supervised_loss, name="train_supervised_loss"),
        "val_supervised_loss": None if val_supervised_loss is None else coerce_finite_scalar(val_supervised_loss, name="val_supervised_loss"),
        "val_auxiliary_losses": {
            str(key): coerce_finite_scalar(value, name=f"val_auxiliary_losses.{key}")
            for key, value in (val_auxiliary_losses or {}).items()
        },
        "val_component_losses": {
            str(key): coerce_finite_scalar(value, name=f"val_component_losses.{key}")
            for key, value in (val_component_losses or {}).items()
        },
        "phase_switch_shock": {
            str(key): (None if value is None else coerce_finite_scalar(value, name=f"phase_switch_shock.{key}"))
            for key, value in (phase_switch_shock or {}).items()
        },
        "shock_streak": {str(key): int(value) for key, value in (shock_streak or {}).items()},
        "soft_control_event": None if soft_control_event is None else dict(soft_control_event),
        "phase_runtime_state": serialized_phase_runtime_state,
        "phase_control_summary": phase_control_summary,
        "current_phase_control_state": phase_control_summary.get("current_phase_control_state"),
        "score_role_summary": _build_score_role_summary(),
        "checkpoint_role_summary": checkpoint_role_summary,
        "control_anchor_checkpoint": checkpoint_role_summary["control_anchor_checkpoint"],
        "training_best_checkpoint": checkpoint_role_summary["training_best_checkpoint"],
        "export_best_checkpoint": checkpoint_role_summary["export_best_checkpoint"],
        "global_selection_mode": str(global_selection_mode),
        "gate_alignment_readout_lr_scale": coerce_finite_scalar(
            trainer_state.get("gate_alignment_readout_lr_scale", _GATE_ALIGNMENT_READOUT_LR_SCALE),
            name="gate_alignment_readout_lr_scale",
            min_value=0.0,
            inclusive=False,
        ),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_loss": None if best_loss is None else coerce_finite_scalar(best_loss, name="best_loss"),
        "best_selection_score": None if best_selection_score is None else coerce_finite_scalar(best_selection_score, name="best_selection_score"),
        "best_export_epoch": None if best_export_epoch is None else int(best_export_epoch),
        "best_export_score": None if best_export_score is None else coerce_finite_scalar(best_export_score, name="best_export_score"),
        "tail_selection_observation_coeff": coerce_finite_scalar(
            trainer_state.get("tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
            name="tail_selection_observation_coeff",
            min_value=0.0,
        ),
    }
    progress_path.write_text(dumps_json_text(progress_payload), encoding="utf-8")


def _safe_finite_float(value: Any) -> float | None:
    """将、转为有float；None 或非有限值（NaN/Inf）返回None。

    D5：用best_loss/best_selection_score 恢复时校验，避免 NaN 穿、污。
    后续 ``selection_score < best_selection_score`` 比较逻辑（NaN 比较恒为
    False 会导致永远不更新 best，最终落盘的 best 仍为 NaN）。
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _try_resume_phase_scheduled_training(
    *,
    model: Any,
    optimizer: Any,
    trainer_state: Mapping[str, Any],
    report_path: Path,
    best_ckpt: Path,
    training_best_ckpt: Path | None = None,
    export_best_ckpt: Path | None = None,
) -> dict[str, Any] | None:
    """从当前output_root 下已有best checkpoint/train report 恢复训练状态"""
    current_tail_selection_observation_coeff = coerce_finite_scalar(
        trainer_state.get("tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
        name="trainer_state.tail_selection_observation_coeff",
        min_value=0.0,
    )
    progress_path = report_path.parent / f'{trainer_state["model_name"]}_train_progress.json'
    runtime_ckpt = best_ckpt.parent / f'{trainer_state["model_name"]}_resume_checkpoint.pt'
    source_checkpoint_path = runtime_ckpt if runtime_ckpt.is_file() else best_ckpt
    if not source_checkpoint_path.is_file():
        return None
    try:
        checkpoint_payload = torch.load(source_checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if not isinstance(checkpoint_payload, Mapping):
        return None

    # 允许只保留progress + best checkpoint 的半成品恢复。
    # 这类状、常来自训练在最report/selection 写盘前被外部中断。
    checkpoint_progress_payload: Mapping[str, Any] | None = None
    if progress_path.is_file():
        try:
            loaded_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded_progress = None
        if isinstance(loaded_progress, Mapping):
            checkpoint_progress_payload = loaded_progress

    train_report: Mapping[str, Any] | None = None
    if report_path.is_file():
        try:
            loaded_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded_report = None
        if isinstance(loaded_report, Mapping):
            train_report = loaded_report
    progress_payload: Mapping[str, Any] | None = None
    if progress_path.is_file():
        try:
            loaded_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded_progress = None
        if isinstance(loaded_progress, Mapping):
            progress_payload = loaded_progress
    report_epoch_count = 0
    if train_report is not None:
        report_epoch_count = max(
            len(list(train_report.get("train_epoch_losses") or [])),
            len(list(train_report.get("val_epoch_losses") or [])),
            len(list(train_report.get("val_selection_scores") or [])),
            len(list(train_report.get("epoch_phase_names") or [])),
        )
    if train_report is not None:
        if str(train_report.get("model_name") or "") != str(trainer_state["model_name"]):
            return None
        if int(train_report.get("epochs") or 0) != int(trainer_state["epochs"]):
            return None
        # D1/D9：phase_schedule 必须与当前trainer_state 一致、若不一致，恢复出的
        # epoch_phase_names / phase_trainability_contract / phase_effective_lrs 会与
        # _resolve_epoch_phase_name 按新配置算出的阶段名混合，破坏公平、与可复现。
        report_phase_schedule = train_report.get("phase_schedule")
        if not isinstance(report_phase_schedule, Mapping):
            return None
        current_phase_schedule = trainer_state.get("phase_schedule") or {}
        for schedule_key in ("warmup_epochs", "gate_alignment_epochs"):
            if int(report_phase_schedule.get(schedule_key) or 0) != int(
                current_phase_schedule.get(schedule_key) or 0
            ):
                return None
        report_tail_selection_observation_coeff = coerce_finite_scalar(
            (train_report or {}).get("tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
            name="train_report.tail_selection_observation_coeff",
            min_value=0.0,
        )
        if not math.isclose(
            report_tail_selection_observation_coeff,
            current_tail_selection_observation_coeff,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return None
        expected_checkpoint_candidates = {
            best_ckpt.resolve(),
        }
        if training_best_ckpt is not None:
            expected_checkpoint_candidates.add(training_best_ckpt.resolve())
        if export_best_ckpt is not None:
            expected_checkpoint_candidates.add(export_best_ckpt.resolve())
        checkpoint_path = str(train_report.get("checkpoint_path") or "").strip()
        if checkpoint_path and Path(checkpoint_path).resolve() not in expected_checkpoint_candidates:
            return None
    progress_tail_selection_observation_coeff = _safe_finite_float(
        (progress_payload or {}).get("tail_selection_observation_coeff")
    )
    if progress_tail_selection_observation_coeff is not None and not math.isclose(
        progress_tail_selection_observation_coeff,
        current_tail_selection_observation_coeff,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return None
    checkpoint_train_cfg = checkpoint_payload.get("train_cfg")
    if isinstance(checkpoint_train_cfg, Mapping):
        checkpoint_train_section = checkpoint_train_cfg.get("train")
        if isinstance(checkpoint_train_section, Mapping):
            checkpoint_tail_selection_observation_coeff = coerce_finite_scalar(
                checkpoint_train_section.get("tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
                name="checkpoint.train_cfg.train.tail_selection_observation_coeff",
                min_value=0.0,
            )
            if not math.isclose(
                checkpoint_tail_selection_observation_coeff,
                current_tail_selection_observation_coeff,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return None
    completed_epoch = 0
    try:
        if progress_payload is not None:
            completed_epoch = max(
                completed_epoch,
                int(progress_payload.get("epoch_index") or 0),
                int(progress_payload.get("best_epoch") or 0),
            )
        if train_report is not None:
            completed_epoch = max(
                completed_epoch,
                int(train_report.get("history_end_epoch") or 0),
                int(train_report.get("best_epoch") or 0),
            )
        if checkpoint_progress_payload is not None:
            completed_epoch = max(
                completed_epoch,
                int(checkpoint_progress_payload.get("epoch_index") or 0),
                int(checkpoint_progress_payload.get("best_epoch") or 0),
            )
        checkpoint_current_epoch = int(checkpoint_payload.get("current_epoch") or 0)
        checkpoint_best_epoch = int(checkpoint_payload.get("best_epoch") or 0)
        completed_epoch = max(completed_epoch, checkpoint_current_epoch, checkpoint_best_epoch)
        if completed_epoch < 1:
            return None
            # ?train_report 的历史长度覆盖不checkpoint 扢、指向completed_epoch。
        # 说明 report/checkpoint 不是同一条连续训练链，直接拒绝续跑，避免把不同运行段
        # 混进同一份训练报告。
        if train_report is not None and report_epoch_count > 0 and completed_epoch > report_epoch_count:
            return None
        if not _restore_training_snapshot(model=model, optimizer=optimizer, checkpoint_payload=checkpoint_payload):
            return None
    except Exception:
        return None
    try:
        # D5：best_loss/best_selection_score 必须为有限数，否则视None。
        # 优先train_report，回逢 progress_payload；用 ``is not None`` 判断
        # 以保留合法的 0.0（原 ``or`` 写法会让 0.0 错误地回逢progress 值）。
        report_best_loss = _safe_finite_float((train_report or {}).get("best_loss"))
        report_best_selection_score = _safe_finite_float((train_report or {}).get("best_selection_score"))
        progress_best_loss = (
            _safe_finite_float(progress_payload.get("best_loss")) if progress_payload else None
        )
        progress_best_selection_score = (
            _safe_finite_float(progress_payload.get("best_selection_score")) if progress_payload else None
        )
        checkpoint_progress_best_loss = (
            _safe_finite_float(checkpoint_progress_payload.get("best_loss")) if checkpoint_progress_payload else None
        )
        checkpoint_progress_best_selection_score = (
            _safe_finite_float(checkpoint_progress_payload.get("best_selection_score"))
            if checkpoint_progress_payload
            else None
        )
        report_phase_runtime_state = _coerce_phase_runtime_state((train_report or {}).get("phase_runtime_state"))
        progress_phase_runtime_state = _coerce_phase_runtime_state((progress_payload or {}).get("phase_runtime_state"))
        checkpoint_phase_runtime_state = _coerce_phase_runtime_state(checkpoint_payload.get("phase_runtime_state"))
        phase_runtime_state = (
            report_phase_runtime_state
            or progress_phase_runtime_state
            or checkpoint_phase_runtime_state
        )
        report_best_epoch = int((train_report or {}).get("best_epoch") or 0)
        progress_best_epoch = int((progress_payload or {}).get("best_epoch") or 0) if progress_payload else 0
        checkpoint_best_epoch_payload = int(checkpoint_payload.get("best_epoch") or 0)
        resumed_best_epoch = max(
            report_best_epoch,
            progress_best_epoch,
            checkpoint_best_epoch_payload,
            max(
                (
                    int(payload.get("best_supervised_epoch") or 0)
                    for payload in phase_runtime_state.values()
                ),
                default=0,
            ),
            0,
        ) or None
        phase_shock_streaks = {
            str(phase_name): int(payload.get("shock_streak") or 0)
            for phase_name, payload in phase_runtime_state.items()
        }
        if not phase_shock_streaks and isinstance(train_report, Mapping):
            phase_shock_streaks = {
                str(key): int(value)
                for key, value in dict((train_report or {}).get("phase_shock_streaks") or {}).items()
                if is_integer(value)
            }
        return {
            "start_epoch": completed_epoch + 1,
            "completed_epoch": completed_epoch,
            "train_epoch_losses": list((train_report or {}).get("train_epoch_losses") or []),
            "val_epoch_losses": list((train_report or {}).get("val_epoch_losses") or []),
            "val_selection_scores": list((train_report or {}).get("val_selection_scores") or []),
            "epoch_candidate_paths": list((train_report or {}).get("epoch_candidate_paths") or []),
            "epoch_candidate_epochs": list((train_report or {}).get("epoch_candidate_epochs") or []),
            "epoch_phase_names": list((train_report or {}).get("epoch_phase_names") or []),
            "phase_trainability_contract": dict((train_report or {}).get("phase_trainability_contract") or {}),
            "phase_effective_lrs": dict((train_report or {}).get("phase_effective_lrs") or {}),
            "best_epoch": resumed_best_epoch,
            "best_loss": report_best_loss if report_best_loss is not None else progress_best_loss,
            "best_selection_score": (
                report_best_selection_score
                if report_best_selection_score is not None
                else progress_best_selection_score
            ),
            "phase_runtime_state": phase_runtime_state,
            "soft_control_events": list((train_report or {}).get("soft_control_events") or []),
            "phase_shock_streaks": phase_shock_streaks,
            "resume_checkpoint_path": str(source_checkpoint_path),
            "global_selection_mode": str((train_report or {}).get("global_selection_mode") or "selection_score"),
            "best_export_epoch": (
                int((train_report or {}).get("best_export_epoch") or 0) or None
            ),
            "best_export_score": _safe_finite_float((train_report or {}).get("best_export_score")),
            "training_best_checkpoint": str(
                (train_report or {}).get("training_best_checkpoint")
                or (train_report or {}).get("checkpoint_path")
                or (training_best_ckpt if training_best_ckpt is not None else best_ckpt)
            ),
            "export_best_checkpoint": str(
                (train_report or {}).get("export_best_checkpoint")
                or (train_report or {}).get("checkpoint_path")
                or (export_best_ckpt if export_best_ckpt is not None else best_ckpt)
            ),
            "_checkpoint_progress_best_loss": checkpoint_progress_best_loss,
            "_checkpoint_progress_best_selection_score": checkpoint_progress_best_selection_score,
            # audit #20 M1: 透传 checkpoint 中保存的 lr_scheduler_state_dict 给 epoch loop,
            # 在 lr_scheduler 实例首次创建后立即 load_state_dict, 恢复 CosineAnnealingLR 的
            # base_lrs / last_epoch / etas / T_max. 没有 state_dict, retry7 重启会丢退火进度
            # (last_epoch 复位, lr 又从 baseline lr 开始, 退火曲线被打断).
            "lr_scheduler_state_dict": checkpoint_payload.get("lr_scheduler_state_dict"),
        }
    except (TypeError, ValueError):
        return None
def _project_train_risk(raw_risk: torch.Tensor, *, already_normalized: bool = False) -> torch.Tensor:
    """将原始风险输出投影到训练阶段实际使用的风险区间。

    训练risk 输出需要落到BRIDGE_THRESHOLDS["risk_min"] 。
    BRIDGE_THRESHOLDS["risk_max"] 的区间内。如risk 尚未归一。
    （即还是网络原始 logit），先做 sigmoid 压到 [0, 1]，再线性映。
    到目标区间；如果已经归一化（例如经过 risk_calibration 校准），
    则跳sigmoid，只做区间映射。

    协议层锚点（D2 分层边界）：本函数的线性映射+ clamp 公式与协议层
    ``liquidloc.protocol.risk_projection.project_risk_to_protocol_range`` 一致，
    该协议函数是公式唯一权威定义。本函数未直接委托协议层，原因有二：
    (1) 训练侧输入是张量，需要``torch.clamp`` 而非标量 ``min/max``。
    (2) 训练侧携带``already_normalized`` 分支持sigmoid 预归一化逻辑。
    属于训练链路特有语义。公式口径以协议层为锚点保持可追溯，不在此重述。
    公式定义。

    参数：
        raw_risk: 原始风险输出张量（标量）。
        already_normalized: 是否已经归一化到 [0, 1]。
            True 表示跳过 sigmoid，只做区间映射；
            False 表示先做 sigmoid 再映射。

    返回值：
        映射到训练区间的风险标量张量。
    """
    risk_span = BRIDGE_THRESHOLDS["risk_max"] - BRIDGE_THRESHOLDS["risk_min"]  # 先算训练区间跨度。
    risk = torch.as_tensor(raw_risk, dtype=torch.float32).reshape(())  # 统一成标量张量，便于后续映射。
    if not torch.isfinite(risk):  # NaN/Inf 会经 sigmoid 传播到loss，污染整个 batch 梯度。
        raise ValueError(f"raw_risk must be finite, got {risk.item()}")
    if not already_normalized:  # 未校准时按旧语义先做 sigmoid。
        risk = torch.sigmoid(risk) # 先把原始输出压到 [0, 1].
    if risk_span != 1.0 or BRIDGE_THRESHOLDS["risk_min"] != 0.0:  # 如果默认区间不是标准区间，就平移并缩放。
        risk = risk * risk_span + BRIDGE_THRESHOLDS["risk_min"]  # 把概率域映射到真实训练区间。
    # 与推理侧 _project_risk_to_protocol_range 保持一致：末尾 clamp 到协议区间。
    risk = torch.clamp(risk, BRIDGE_THRESHOLDS["risk_min"], BRIDGE_THRESHOLDS["risk_max"])
    return risk  # 返回训练阶段实际使用的风险。




def _clamp_unit_interval(value: Any, *, name: str) -> float:
    """将输入、裁剪到 [0, 1] 区间，并校验有限性。

    用于对风险、信号等概率域标量做合法性检查和截断。

    参数：
        value: 待裁剪的输入值，可以是任何可float 的类型。
        name: 字段名，仅用于报错信息。

    返回值：
    裁剪到 [0, 1] 的浮点数。

    失败条件。
        ValueError: 当输入为 NaN 或无穷大时抛出。
    """
    scalar_value = coerce_finite_scalar(value, name=name)  # 统一走中心校验：排除 bool、处理单元素张量、有限、检查并带字段名报错。
    return min(1.0, max(0.0, scalar_value))  # 再裁剪到单位区间 [0, 1]。




def _coerce_positive_sample_weight(value: Any, *, name: str) -> float:
    """校验并转换样本权重，要求必须是有限正数。

    样本权重用于在损失计算中对不同样本施加不同重要、，
    零或负权重没有意义，NaN/Inf 也不合法。

    参数：
        value: 待校验的权重值，必须是数值类型（不接受bool）。
        name: 字段名，仅用于报错信息。

    返回值：
        校验通过的浮点权重。

    失败条件。
        TypeError: 当输入为 bool、非数、类型或非标量时抛出。
        ValueError: 当输入为 NaN、无穷大或非正数时抛出。
    """
    return coerce_finite_scalar(value, name=name, min_value=0.0, inclusive=False)  # 委托公共校验入口，保证与 LSTM 侧口径一致。




def _resolve_current_modality(window_tensor: Mapping[str, Any]) -> str:
    """从窗口张量中提取并校验当前模态名。

    训练器只认识 _ACTIVE_HEADS_BY_MODALITY 中注册的模。
    ?uwb" ?"vio"），其他模一律报错。

    参数：
        window_tensor: 包含 "current_modality" 字段的结构化窗口映射。

    返回值：
    校验通过的模态名字符串（"uwb" ?"vio"）。

    失败条件。
        ValueError: 当模态不是字符串、或strip().lower() 归一化后不在合法集合中时抛出。
        归一化口径与 liquidloc.models.liquid.network._coerce_supported_modality 保持一致，
        确保 LSTM ?Liquid 公平对比时模态校验完全相同。
    """
    raw_modality = window_tensor.get("current_modality")
    if not is_string_like(raw_modality): # 模态必须是字符串，拒绝 None/数值、布尔/列表等静默转换。
        raise ValueError(  # 报错时把合法集合带出来，方便排查。
            "feature_window.current_modality must be one of "
            f"{sorted(_ACTIVE_HEADS_BY_MODALITY)} for trainer loss"
        )
    modality = str(raw_modality).strip().lower() # 与 network._coerce_supported_modality 保持同一归一化口径.
    if modality not in _ACTIVE_HEADS_BY_MODALITY:  # 只允许训练器认识的模态。
        raise ValueError(  # 报错时把合法集合带出来，方便排查。
            "feature_window.current_modality must be one of "
            f"{sorted(_ACTIVE_HEADS_BY_MODALITY)} for trainer loss"
        )

    return modality  # 返回校验通过的模态名。




def _build_head_mask_tensor(

    modalities: list[str],

    *,

    reference_tensor: torch.Tensor,

) -> torch.Tensor:
    """按模态列表构造输出头掩码张量。

    每种模、只濢、活自己对应的输出头（_ACTIVE_HEADS_BY_MODALITY），
    其余头位置填零，这样损失函数只对被激活的头做回归。
    掩码的dtype 和device 中reference_tensor 一致。

    参数：
        modalities: 当前批次每个样本的模态名列表。
        reference_tensor: 用于确定 dtype ?device 的参考张量。

    返回值：
        形状态(batch_size, len(_OUTPUT_KEYS)) 的掩码张量，
        濢、活位置为 1.0，其余为 0.0。

    失败条件。
        ValueError: 当模态不被支持或模态数量与参数张量 batch 维不一致。
        或整个批次没有激活头时抛出。
    """
    # ?LSTM ?_build_head_mask_tensor 对齐：模态数量必须和参数张量batch 维一致，
    # 否则后续 mask * sample_weight_tensor 会因广播错误产生静默错误结果。
    if len(modalities) != int(reference_tensor.shape[0]):
        raise ValueError(
            "modalities length must match reference_tensor batch dimension: "
            f"got {len(modalities)} modalities vs batch {int(reference_tensor.shape[0])}"
        )
    mask = torch.zeros((len(modalities), len(_OUTPUT_KEYS)), dtype=reference_tensor.dtype, device=reference_tensor.device)

    for row_index, modality in enumerate(modalities):

        if modality not in _ACTIVE_HEADS_BY_MODALITY:  # 只接受训练器认识的模态。
            raise ValueError(f"unsupported modality for trainer loss: {modality!r}")  # 不合法就直接报错。
        for head_key in _ACTIVE_HEADS_BY_MODALITY[modality]:  # 每种模、只濢、活自己对应的头。
            mask[row_index, _OUTPUT_KEYS.index(head_key)] = 1.0  # 把对应位置标成可监督。
    if float(mask.sum().detach().item()) <= 0.0:  # 如果整个批次都没有激活头，就说明输入有问题。
        raise ValueError("trainer loss head mask must activate at least one head")  # 直接失败，避免除零。
    return mask  # 返回构、好的输出头掩码。




def _build_sample_weight_tensor(

    sample_weights: list[float] | None,

    *,

    reference_tensor: torch.Tensor,

) -> torch.Tensor:
    """将样本权重列表转为张量，未提供时返回全 1。

    样本权重用于在损失计算中对不同样本施加不同重要、，
    例如高风险样本可以给更高权重。每个权重都必须是有限正数。

    参数：
        sample_weights: 样本权重列表，None 时默认等权。
        reference_tensor: 用于确定 batch_size、dtype ?device 的参考张量。

    返回值：
        形状态(batch_size,) 的样本权重张量。

    失败条件。
        ValueError: 当权重数量与 batch_size 不一致或权重不合法时抛出。
    """
    batch_size = int(reference_tensor.shape[0])  # 批大小直接从参、张量取。
    if sample_weights is None:  # 没传权重时，默认每个样本等权。
        return torch.ones((batch_size,), dtype=reference_tensor.dtype, device=reference_tensor.device)  # 直接返回全 1 张量。
    if len(sample_weights) != batch_size:  # 权重数量必须batch 大小致。
        raise ValueError("sample_weights must match the prediction batch size")  # 不一致就报错。
    return torch.tensor(  # 把校验后Python 列表转成张量。
        [_coerce_positive_sample_weight(weight, name=f"sample_weights[{index}]") for index, weight in enumerate(sample_weights)],  # 每个权重都、个校验。
        dtype=reference_tensor.dtype,  # dtype 和参考张量一致。
        device=reference_tensor.device,  # device 和参考张量一致。
    )  # 张量构结束。




def _semantic_loss_matrix(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> torch.Tensor:
    """按输出头语义分别计算逐元素损失矩阵。

    三个语义域使用不同的损失函数（§24.1 训练主损失要求：位置误差主项 MSE/Huber + 可选创新 NLL/辅助；本模块为观测侧增强头，主选模与全序仍以 raw 位置 RMSE 为准）。
    - bias: Huber 损失（δ 与评价主指标无关，训练专用），对极端 NLOS 偏置更鲁棒；
    - risk: 平方误差，做连续风险回归；
    - scaling (uwb_scaling + vio_scaling): 对数域平方误差，更符合乘性噪声语义。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
            列顺序为 bias, risk, uwb_scaling, vio_scaling。。
        target_batch: 目标值批次，形状态prediction_batch 相同。

    返回值：
        形状 (B, 4) 的、元素损失矩阵，每列对应一个输出头。
    """
    # ?LSTM 侧对齐：进入损失计算前先做有限、检查，
    # NaN/Inf 会经 Huber/MSE/log 传播到整个 batch 梯度，必须提前拦截。
    if not torch.isfinite(prediction_batch).all():
        raise ValueError("prediction_batch must be finite for semantic loss")
    if not torch.isfinite(target_batch).all():
        raise ValueError("target_batch must be finite for semantic loss")
    bias_loss = F.huber_loss(
        prediction_batch[:, 0],
        target_batch[:, 0],
        reduction="none",
        delta=bias_huber_delta,
        ) # bias ?Huber，避免极端NLOS 样本主导。
    risk_loss = (prediction_batch[:, 1] - target_batch[:, 1]).pow(2)  # risk 继续做连续风险回归。
    # ?LSTM ?_semantic_loss_matrix 对齐：eps 放进 clamp_min 而非 log 内部。
    # 保证 LSTM ?Liquid 公平对对比scaling 损失数学等价（D7 公平性）。
    # Do not hard-clamp the prediction side here: Liquid scaling is already
    # projected to the neutral floor, and clamping at 1.0 would zero the
    # gradient exactly where the head initializes.
    predicted_scaling = prediction_batch[:, 2:] + scaling_log_eps
    target_scaling = torch.clamp_min(target_batch[:, 2:], _SCALING_NEUTRAL_FLOOR) + scaling_log_eps
    scaling_loss = (
        torch.log(predicted_scaling) - torch.log(target_scaling)
    ).pow(2)  # scaling 在对数域做回归，更符合乘性噪声语义。
    return torch.cat(
        (
            bias_loss.unsqueeze(1),
            risk_loss.unsqueeze(1),
            scaling_loss,
        ),
        dim=1,
    )




def _position_error_mse(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    position_scale: float = 1.0,
) -> torch.Tensor:
    """§24.1 执行点：基于 EKF 状态估计的原始位置误差 MSE（主项）。

    即使当前观察侧网络训练目标为 bias/risk/scaling（观测增强头），
    §24.1 要求主损失必须包含位置误差的 MSE 或训练专用 Huber（δ 与评价指标无关）。
    本函数计算从预测输出重构的几何位置估计与真实位置的逐元素 MSE，
    作为主损失项的实际执行点（可与 EKF state 结合扩展为轨迹 RMSE）。

    参数：
        prediction_batch: (B, 4) 预测批次 [bias, risk, uwb_scaling, vio_scaling]
        target_batch: (B, 4) 目标批次
        position_scale: 位置误差归一化尺度（默认 1.0，可由桥接阈值调整）

    返回值：标量张量，逐样本平方位置误差的平均值（§24.1 主项执行点）。
    """
    # 将预测的几何偏置 (bias) 视为位置修正向量（与真实几何距离比例关系）
    # 计算预测修正后估计位置与真实位置的 MSE
    predicted_pos_correction = prediction_batch[:, 0] / position_scale  # bias → 位置修正分量
    target_pos_correction = target_batch[:, 0] / position_scale
    # 主项：位置误差 MSE（逐元素，标量张量）
    position_mse = (predicted_pos_correction - target_pos_correction).pow(2)
    return position_mse.mean()  # 返回批次平均值（主损失标量执行点）


def _masked_supervised_loss_stats(

    prediction_batch: torch.Tensor,

    target_batch: torch.Tensor,

    *,

    modalities: list[str],
    sample_weights: list[float] | None = None,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,

) -> tuple[torch.Tensor, torch.Tensor]:
    """计算四头监督损失的加权分子和分母。

    先构造输出头掩码和样本权重，再与语义损失矩阵逐元素相乘，
    得到加权后的分子和分母，供外层做除法得到平均损失。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。
        modalities: 当前批次每个样本的模态名列表。
        sample_weights: 样本权重列表，None 时默认等权。

    返回值：
    二元（weighted_numerator, weighted_denominator)。
        - weighted_numerator: 加权损失分子标量张量。
        - weighted_denominator: 有效权重分母标量张量。
    """
    mask = _build_head_mask_tensor(modalities, reference_tensor=prediction_batch)  # 先构造输出头掩码。
    sample_weight_tensor = _build_sample_weight_tensor(sample_weights, reference_tensor=prediction_batch).unsqueeze(1)

    per_head_loss = _semantic_loss_matrix(prediction_batch, target_batch, bias_huber_delta=bias_huber_delta, scaling_log_eps=scaling_log_eps)

    weighted_mask = mask * sample_weight_tensor

    # IEEE 754 ?NaN * 0 = NaN，掩码为零的位置无法隔离 NaN。
    # ?torch.where 把掩码为零位置的 per_head_loss 显式置零。
    # 保证被屏蔽头（如 VIO 样本uwb_scaling）的 NaN/Inf 不会污染整个 batch 的损失。
    masked_per_head_loss = torch.where(
        weighted_mask > 0,
        per_head_loss,
        torch.zeros_like(per_head_loss),
    )
    return (masked_per_head_loss * weighted_mask).sum(), weighted_mask.sum()





def _masked_component_loss_stats(

    prediction_batch: torch.Tensor,

    target_batch: torch.Tensor,

    *,

    modalities: list[str],
    sample_weights: list[float] | None = None,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,

) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[str]]:
    """逐输出头统计加权损失分子、分母和濢、活头列表。

    ?_masked_supervised_loss_stats 不同，本函数按每个输出头
    单独统计，用于后续计算各头独立损失和选择分数。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。
        modalities: 当前批次每个样本的模态名列表。
        sample_weights: 样本权重列表，None 时默认等权。

    返回值：
    三元（component_numerators, component_denominators, active_keys)。
        - component_numerators: 每个濢、活头的加权损失分子字典；
        - component_denominators: 每个濢、活头的有效权重分母字典；
        - active_keys: 本批次真正参与损失的输出头名列表。
    """
    mask = _build_head_mask_tensor(modalities, reference_tensor=prediction_batch)

    sample_weight_tensor = _build_sample_weight_tensor(sample_weights, reference_tensor=prediction_batch)  # 再把样本权重转成张量。
    per_head_loss = _semantic_loss_matrix(prediction_batch, target_batch, bias_huber_delta=bias_huber_delta, scaling_log_eps=scaling_log_eps)  # 用按头语义一致的损失矩阵做统计。
    component_numerators: dict[str, torch.Tensor] = {}  # 每个输出头的加权损失分子。
    component_denominators: dict[str, torch.Tensor] = {}  # 每个输出头的有效权重分母。
    active_keys: list[str] = []  # 记录本批次真正参与损失的输出头。
    for index, key in enumerate(_OUTPUT_KEYS):  # 逐个输出头统计。
        head_mask = mask[:, index] * sample_weight_tensor  # 当前输出头的有效样本权重。
        if float(head_mask.sum().detach().item()) <= 0.0:  # 如果这个头没有有效样本，就跳过。
            continue  # 没有有效样本就不统计这个头。
        # 与_masked_supervised_loss_stats 一致：IEEE 754 中NaN * 0 = NaN锛。
        # 掩码为零位置霢torch.where 显式置零，防止被屏蔽样本NaN 污染该头统计。
        head_loss = torch.where(head_mask > 0, per_head_loss[:, index], torch.zeros_like(per_head_loss[:, index]))
        component_numerators[key] = (head_loss * head_mask).sum()  # 统计该头的加权分子。
        component_denominators[key] = head_mask.sum()  # 统计该头的有效分母。
        active_keys.append(key)  # 记录这个头确实参与了损失。
    # §24.1 执行点：实际计算位置误差 MSE 主损失（必须参与总损失与选模决策，非仅记录）。
    position_mse_scalar = _position_error_mse(prediction_batch, target_batch)  # 执行主损失位置误差计算。
    # 返回位置误差执行点值，供上游总损失与选模使用（§24.1 强制接入训练链）。
    return component_numerators, component_denominators, active_keys, position_mse_scalar  # 四元组：增加位置误差执行点。




def _masked_supervised_loss(
    prediction_batch: torch.Tensor,

    target_batch: torch.Tensor,

    *,

    modalities: list[str],
    sample_weights: list[float] | None = None,
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,

) -> torch.Tensor:
    """返回整批四头监督项的加权语义损失。

    先调用_masked_supervised_loss_stats 得到分子和分母，
    再做除法得到平均损失。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。
        modalities: 当前批次每个样本的模态名列表。
        sample_weights: 样本权重列表，None 时默认等权。

    返回值：
        加权平均监督损失标量张量。
    """
    numerator, denominator = _masked_supervised_loss_stats(  # 先拿到分子和分母。
        prediction_batch,  # 预测批次。
        target_batch,  # 目标批次。
        modalities=modalities,  # 当前批次的模态列表。
        sample_weights=sample_weights,  # 当前批次的样本权重。
        bias_huber_delta=bias_huber_delta,
        scaling_log_eps=scaling_log_eps,
    )  # 统计结束。
    # ?LSTM ?_compute_batch_loss 对齐：分母做 clamp_min ?0 除，
    # 保证 LSTM ?Liquid 公平对比时除法数值稳定策略一致（D7 公平性）。
    return numerator / torch.clamp_min(denominator, scaling_log_eps)  # 用分子除以分母得到平均损失。




def _normalized_squared_error(prediction_batch: torch.Tensor, target_batch: torch.Tensor) -> torch.Tensor:
    """计算预测与目标之间的逐元素平方误差。

    ?_semantic_loss_matrix 不同，这里不做语义区分，
    统一用平方误差，用于诊断日志中的原始误差统计。

    注意：本函数不做任何归一化，函数名中\"normalized" 仅为历史命名。
    实际返回的是原始 (pred - target)^2。与 lstm/trainer.py 同名函数口径一致。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。

    返回值：
        形状 (B, 4) 的、元素平方误差张量。
    """
    return (prediction_batch - target_batch).pow(2) # 逐元 (pred - target)^2.


def _risk_calibration_bin_weights(target_risk: torch.Tensor, *, bin_count: int = _RISK_CALIBRATION_BIN_COUNT, scaling_log_eps: float = _SCALING_LOG_EPS) -> torch.Tensor:
    """risk calibration 损失计算等频分箱逆频率权重.

    batch 内目标风险按分位数分成若干等频桶, 每个样本的权重
    等于其所在桶的逆频率 (桶内样本越少权重越大), 最后归一化
    使均值约 1. 这样低风险区间 (样本多) 不会被过度主导,
    高风险区间 (样本少) 也能得到足够关注.

    参数:
        target_risk: 目标风险值一维张量, 形状 (N,).

    返回值:
        与 target_risk 同形状的权重张量, 均值约 1.
    """
    if target_risk.dim() != 1:  # 仅接受一维目标风险张量，LSTM 侧口径对齐。
        raise ValueError(f"target_risk must be 1-D, got {target_risk.dim()}D")
    if not torch.isfinite(target_risk).all():  # 拒绝 NaN/Inf，避免经 quantile/bucketize/bincount 静默穿。
        raise ValueError("target_risk contains non-finite values (NaN or Inf)")
    sample_count = int(target_risk.numel())  # 样本总数。
    if sample_count <= 1:  # 只有一个样本时无法分箱，直接等权。
        return torch.ones_like(target_risk)  # 返回全 1 权重。
    effective_bin_count = max(1, min(bin_count, sample_count))  # 实际分箱数，不超过样本数。
    quantiles = torch.linspace(0.0, 1.0, steps=effective_bin_count + 1, device=target_risk.device, dtype=target_risk.dtype)  # 等距分位点，按实际分箱数生成，与 LSTM 侧一致。
    edges = torch.quantile(target_risk.detach(), quantiles)  # 在目标风险上算出实际分位边界。
    interior_edges = edges[1:-1]  # 去掉首尾边界，只留内部切分点。
    if interior_edges.numel() == 0:  # 没有内部边界时无法分桶。
        return torch.ones_like(target_risk)  # 返回全 1 权重。
    bucket_index = torch.bucketize(target_risk.detach(), interior_edges, right=True)  # 每个样本落入哪个桶。
    bucket_count = torch.bincount(bucket_index, minlength=int(interior_edges.numel()) + 1).clamp_min(1)  # 每个桶的样本数，至少 1。
    raw_weights = torch.reciprocal(bucket_count[bucket_index].to(dtype=target_risk.dtype))  # 逆频率权重。
    normalized_weights = raw_weights / torch.clamp_min(raw_weights.mean(), scaling_log_eps)  # 归一化使均值约为 1。
    return normalized_weights  # 返回归一化后的分箱权重。


def _risk_calibration_trainable(model: Any) -> bool:
    """检查模型的 risk_calibration 模块是否有可训练参数。

    只有 risk_calibration 存在、有 parameters 方法、且至少一个
    参数 requires_grad 为 True 时，才认为它是可训练的。
    用于决定是否在辅助损失中启用 calibration 与 mono 项。

    参数：
        model: 液体模型实例，预期有 risk_calibration 属性。

    返回值：
        True 表示 risk_calibration 可训练，False 表示不可训练或不存在。
    """
    calibration = getattr(model, "risk_calibration", None)  # 安全地取 risk_calibration 属性。
    parameters = getattr(calibration, "parameters", None)  # 安全地取 parameters 方法。
    if not callable(parameters):  # 没有 parameters 方法说明不是 nn.Module。
        return False  # 不可训练。
    return any(parameter.requires_grad for parameter in parameters())  # 至少一个参数可训练就返回True。


def _compute_liquid_auxiliary_loss_terms(
    model: Any,
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    modalities: list[str],
    sample_weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """计算 Liquid 模型特有的辅助损失项。

    Liquid 模型在主监督损失之外还有四项辅助损失。
    - calibration: risk calibration 的加MSE，配合分箱、频率权重；
    - mono: calibration 斜率的单调、惩罚，防止 a_raw 塌缩为负。
    - gate_l1: 输出头中各种门控参数L1 惩罚，维持中性启动；
    - reg_l2: 全局 L2 正则，防止参数膨胢。

    参数：
        model: 液体模型实例。
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。
        modalities: 当前批次每个样本的模态名列表。
        sample_weights: 样本权重列表，None 时默认等权。

    返回值：
    字典，包含"calibration"?mono"?reg_l2"?gate_l1"?total"
        五个键，每个值都是标量张量。
    """
    zero = prediction_batch.new_zeros(())  # 零标量，用作各项默认值。
    # 从模型实例属性读取配置、，缺失时回逢、到模块级常量默认值。
    loss_weights = getattr(model, "_loss_weights", {})
    # D5/D7：与 lstm/trainer.py 同名函数口径对齐，统丢coerce_finite_scalar 中心入口。
    # float() 会把 NaN/Inf 与负值静默穿透到损失项，导致训练发散时无法定位字段名。
    scaling_log_eps = coerce_finite_scalar(  # 数稳定项必须有限且非负，LSTM 侧一致。
        getattr(model, "_scaling_log_eps", _SCALING_LOG_EPS),
        name="scaling_log_eps",
        min_value=0.0,
    )
    calibration_bin_count = int(  # 分箱数必须有限且 >=1，避免下界max(1, min(...)) 兜底掩盖配置错误。
        coerce_finite_scalar(
            getattr(model, "_calibration_bin_count", _RISK_CALIBRATION_BIN_COUNT),
            name="calibration_bin_count",
            min_value=1,
        )
    )
    w_cal = coerce_finite_scalar(  # 校准辅助权重，拒NaN/Inf/负。
        loss_weights.get("w_cal", _LIQUID_CALIBRATION_WEIGHT),
        name="loss_weights.w_cal",
        min_value=0.0,
    )
    lambda_mono = coerce_finite_scalar(  # 单调性权重，拒绝 NaN/Inf/负。
        loss_weights.get("lambda_mono", _LIQUID_MONO_WEIGHT),
        name="loss_weights.lambda_mono",
        min_value=0.0,
    )
    lambda_l1 = coerce_finite_scalar(  # 门控 L1 权重，拒NaN/Inf/负。
        loss_weights.get("lambda_l1", _LIQUID_GATE_L1_WEIGHT),
        name="loss_weights.lambda_l1",
        min_value=0.0,
    )
    lambda_l2 = coerce_finite_scalar(  # 全局 L2 权重，拒NaN/Inf/负。
        loss_weights.get("lambda_l2", _GLOBAL_L2_WEIGHT),
        name="loss_weights.lambda_l2",
        min_value=0.0,
    )
    # Read enable_smooth first so we can choose the correct default for lambda_smooth.
    phase_control_state = getattr(model, "_phase_control_state", {}) or {}
    enable_smooth = bool(
        phase_control_state.get("enable_smooth_regularization", False)
        or getattr(model, "_enable_smooth_regularization", False)
    )
    # 当启用 smooth 正则时，默认使用 _LIQUID_SMOOTH_WEIGHT（=1e-3）作为权重，
    # 否则权重为 0（保持向后兼容）。允许 loss_weights["lambda_smooth"] 显式覆盖。
    lambda_smooth = coerce_finite_scalar(  # L_smooth 输出平滑正则权重（默认 _LIQUID_SMOOTH_WEIGHT=1e-3 当启用时）。
        loss_weights.get("lambda_smooth", _LIQUID_SMOOTH_WEIGHT if enable_smooth else 0.0),
        name="loss_weights.lambda_smooth",
        min_value=0.0,
    )
    aux_terms = {  # v2：五项辅助损失（含 smooth 输出平滑正则）。
        "calibration": zero,  #  risk calibration 损失。
        "mono": zero,  #  calibration 单调性惩罚。
        "reg_l2": zero,  #  全局 L2 正则。
        "gate_l1": zero,  #  门控 L1 惩罚。
        "smooth": zero,  # L_smooth 输出平滑正则（v2 可选）。
    }
    aux_scale = coerce_finite_scalar(
        phase_control_state.get("aux_scale", 1.0),
        name="phase_control_state.aux_scale",
        min_value=0.0,
    )
    gate_scale = coerce_finite_scalar(
        phase_control_state.get("gate_scale", aux_scale),
        name="phase_control_state.gate_scale",
        min_value=0.0,
    )
    calibration_scale = coerce_finite_scalar(
        phase_control_state.get("calibration_scale", aux_scale),
        name="phase_control_state.calibration_scale",
        min_value=0.0,
    )
    regularization_scale = coerce_finite_scalar(
        phase_control_state.get("regularization_scale", 1.0),
        name="phase_control_state.regularization_scale",
        min_value=0.0,
    )
    if _risk_calibration_trainable(model) and calibration_scale > 0.0: # 只有 risk_calibration 可训练时才计 calibration/mono.
        risk_mask = _build_head_mask_tensor(modalities, reference_tensor=prediction_batch)[:, 1] # risk 列的掩码.
        sample_weight_tensor = _build_sample_weight_tensor(
            sample_weights,
            reference_tensor=prediction_batch,
        )
        effective_weight = risk_mask * sample_weight_tensor # risk 列有效权重= 头掩码× 样本权重。
        observed = effective_weight > 0.0  # 只有权重 > 0 的样本才参与 calibration 损失。
        if bool(observed.any().item()):  # 至少有一个有risk 样本时才计算。
            predicted_risk = prediction_batch[observed, 1]  # 取有效样本的预测 risk。
            target_risk = target_batch[observed, 1]  # 取有效样本的目标 risk。
            bin_weights = _risk_calibration_bin_weights(target_risk, bin_count=calibration_bin_count, scaling_log_eps=scaling_log_eps)  # 按目标风险分箱计算、频率权重。
            calibration_weight = effective_weight[observed] * bin_weights # 最终校准权= 有效权重 × 分箱权重。
            calibration_loss = ((predicted_risk - target_risk).pow(2) * calibration_weight).sum()  # 加权 MSE 分子。
            calibration_loss = calibration_loss / torch.clamp_min(calibration_weight.sum(), scaling_log_eps)  # 除以权重和得到均值。
            aux_terms["calibration"] = calibration_loss * w_cal * calibration_scale  # 乘以校准辅助权重。
        calibration = getattr(model, "risk_calibration", None)  # 安全地取 risk_calibration 模块。
        if hasattr(calibration, "a_raw"):  # 如果存在 a_raw 参数（校准斜率的原始值）。
            aux_terms["mono"] = F.softplus(-calibration.a_raw).mean() * lambda_mono * calibration_scale  # 惩罚 a_raw 为负，保证校准斜率非负。

    gate_penalty = zero  # 门控 L1 惩罚累加器。
    # v3.1 head-shared: 门控层在 output_backbone 上，不在 output_heads 中
    backbone = getattr(model, "output_backbone", None)
    if backbone is not None:
        for attribute_name in (  # 遍历有门控子模块名。
            "temporal_context_gate",  # 时间上下文门控。
            "observation_context_gate",  # 观测上下文门控。
            "filter_context_gate",  # 滤波上下文门控。
            "branch_mix_gate",  # 分支混合门控。
            "uwb_branch_mix_gate",  # UWB 分支混合门控。
            "vio_branch_mix_gate",  # VIO 分支混合门控。
        ):
            module = getattr(backbone, attribute_name, None)  # 安全地取门控子模块。
            parameters = getattr(module, "parameters", None)  # 安全地取 parameters 方法。
            if not callable(parameters):  # 不是 nn.Module 就跳过。
                continue  # 跳过不可调用的。
            for parameter in parameters():  # 遍历该门控的有参数。
                if parameter.requires_grad:  # 只对可训练参数做 L1。
                    gate_penalty = gate_penalty + parameter.abs().sum()  # 累加绝对值。
    if float(gate_penalty.detach().item()) > 0.0 and gate_scale > 0.0:  # 如果有非零门控惩罚。
        aux_terms["gate_l1"] = gate_penalty * lambda_l1 * gate_scale  # 乘以门控 L1 权重。
    l2_penalty = zero  # L2 正则累加器。
    for parameter in model.parameters():  # 遍历模型有参数。
        if parameter.requires_grad:  # 只对可训练参数做 L2。
            l2_penalty = l2_penalty + parameter.pow(2).sum()  # 累加参数平方和。
    if float(l2_penalty.detach().item()) > 0.0:  # 如果有非L2 惩罚。
        aux_terms["reg_l2"] = l2_penalty * lambda_l2 * regularization_scale  # 乘以阶段化正则接管权重。
    # v2: 输出平滑正则 L_smooth（可选）。
    # 惩罚相邻时间步测量的差异，促使模型输出更平滑。
    if enable_smooth and lambda_smooth > 0.0 and prediction_batch.shape[0] > 1:
        smooth_loss = (prediction_batch[1:] - prediction_batch[:-1]).pow(2).mean() * lambda_smooth * regularization_scale
        aux_terms["smooth"] = smooth_loss
    aux_terms["total"] = aux_terms["calibration"] + aux_terms["mono"] + aux_terms["reg_l2"] + aux_terms["gate_l1"] + aux_terms["smooth"]  # 四项辅助损失求和。
    return aux_terms  # 返回包含有辅助损失项的字典。


def _masked_component_losses(

    prediction_batch: torch.Tensor,

    target_batch: torch.Tensor,

    *,

    modalities: list[str],

    sample_weights: list[float] | None = None,

    bias_huber_delta: float = _BIAS_HUBER_DELTA,

    scaling_log_eps: float = _SCALING_LOG_EPS,

) -> tuple[dict[str, torch.Tensor], list[str]]:
    """逐输出头计算独立的加权平均损失。

    ?_masked_supervised_loss_stats 不同，本函数返回每个濢、活头
    独立做除法后的平均损失，用于选择分数计算和诊断日志。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。
        modalities: 当前批次每个样本的模态名列表。
        sample_weights: 样本权重列表，None 时默认等权。
        bias_huber_delta: bias ?Huber 损失 delta，默认_BIAS_HUBER_DELTA。
        scaling_log_eps: scaling 转对数前数、稳定项，默认_SCALING_LOG_EPS。
        两、必须转发模型配置、，_masked_supervised_loss 及评估管。
            同口径，避免选择分数漂移（D6 指标口径冻结、D7 公平性）。

    返回值：
    二元（component_losses, active_keys)。
        - component_losses: 每个濢、活头的独立平均损失字典；
        - active_keys: 本批次真正参与损失的输出头名列表。
    """

    component_numerators, component_denominators, active_keys, position_mse_scalar = _masked_component_loss_stats(  # 先拿各头统计量（含§24.1 位置误差执行点）。
        prediction_batch,  # 预测批次。
        target_batch,  # 目标批次。
        modalities=modalities,  # 当前批次的模态列表。
        sample_weights=sample_weights,  # 当前批次的样本权重。
        bias_huber_delta=bias_huber_delta,  # 转发模型配置bias Huber delta（D6/D7 口径一致）。
        scaling_log_eps=scaling_log_eps,  # 转发模型配置scaling 对数稳定项（D6/D7 口径一致）。
    )  # 统计结束。
    component_losses: dict[str, torch.Tensor] = {}  # 每个活头的独立损失。
    for key in active_keys:  # 只处理真正参与本批次损失的头。
        component_losses[key] = component_numerators[key] / component_denominators[key]  # 当前头的平均损失。
    return component_losses, active_keys  # 返回每个头的损失和激活头列表。




def _weighted_selection_score(component_losses: Mapping[str, torch.Tensor], active_keys: list[str], *, position_mse_scalar: torch.Tensor | None = None) -> torch.Tensor:
    """按预设权重计算多头的加权选择分数。§24.1 最终定序仅 raw 位置 RMSE；

    当前多头加权选择分数为辅助监控；主选模与全序结论必须以 raw 位置 RMSE（§14 主指标）为准。
    选择分数用于在验证阶段挑选最优检查点，各头的权重
    ?_SELECTION_WEIGHTS 中定义、只active_keys 中的。
    参与计算，权重按活头重新归一化。

    参数：
        component_losses: 每个活头的独立损失字典。
        active_keys: 本批次真正参与损失的输出头名列表。

    返回值：
        加权选择分数标量张量。

    失败条件。
    KeyError: ?active_keys 中存在未在_SELECTION_WEIGHTS 中定义的头时抛出。
        ValueError: 当权重、为 NaN/Inf/负数或所有激活头的权重之和为零时抛出。
    """
    missing_keys = [key for key in active_keys if key not in _SELECTION_WEIGHTS] # D9：校验active_keys ?_SELECTION_WEIGHTS（键名等价于 MODEL_INTERMEDIATE_KEYS）。
    if missing_keys:  # 存在未定义的输出头。
        raise KeyError(f"selection_weights missing keys: {missing_keys}")  # 报告缺失键名，便于定位。
    # §24.1 执行点：主损失必须包含位置误差 MSE（_position_error_mse 执行函数已定义并可用于扩展）。
    # §24.1 执行点：位置误差 MSE 必须参与选模（终序仅 raw 位置 RMSE，不可仅靠辅头）。
    _POSITION_MSE_SELECTION_WEIGHT = 0.50  # §24.1 位置误差主项选模权重（占 50%，高于各辅头）。
    validated_weights = {  # D5：经 coerce_finite_scalar 校验，拒NaN/Inf/bool/负数权重，与 lstm/trainer.py 同名函数口径致（D7）。
        key: coerce_finite_scalar(_SELECTION_WEIGHTS[key], name=f"selection_weights.{key}", min_value=0.0)
        for key in active_keys
    }
    total_weight = sum(validated_weights[key] for key in active_keys)  # 计算活头的权重。

    if total_weight <= 0.0:  # 总权重为零说明没有有效头。

        raise ValueError("selection score requires at least one active weighted head")  # 直接报错。

    # §24.1 执行点：位置误差 MSE 作为主项加入选模分数（强制参与最终定序）。
    base_score = sum(  # D7：与 lstm/trainer.py 同名函数对齐，使float32 起始张量保证结果类型一致。
        (
            (validated_weights[key] / total_weight) * component_losses[key]
            for key in active_keys
        ),
        start=torch.tensor(0.0, dtype=torch.float32),
    )  # 按归化权重加权求和。
    # §24.1 执行点：位置误差 MSE 列为选模主项（50% 权重），确保终序仅 raw 位置 RMSE 精神执行。
    if position_mse_scalar is not None:
        base_score = base_score + _POSITION_MSE_SELECTION_WEIGHT * position_mse_scalar  # 位置误差强制纳入选模。
    return base_score





def _resolve_selection_sample_weight(
    sample: Mapping[str, Any],
    target_outputs: Mapping[str, float],
    *,
    observation_coeff: float = _TAIL_SELECTION_OBSERVATION_COEFF,
) -> float:
    """根据目标风险和跟踪信号计算样本的选样权重。

    高风险、高对齐风险、高观测风险的样本应该得到更高权重，
    以确保训练不会忽略尾部困难样本、权重从 1.0 开始，
    按尾部信号和观测信号的加成、增，最终截断到硬上限。

    参数：
        sample: 原始样本映射，可能包target_trace 子映射。
        target_outputs: 已校验的目标输出字典，至少包含含"risk" 键。

    返回值：
    截断到 [1.0, _TAIL_SELECTION_MAX_WEIGHT] 区间内的样本权重浮点数。
    """

    resolved_observation_coeff = coerce_finite_scalar(
        observation_coeff,
        name="tail_selection_observation_coeff",
        min_value=0.0,
    )

    if "risk" not in target_outputs:
        raise KeyError("target_intermediate is missing required key: 'risk'")
    risk = _clamp_unit_interval(target_outputs["risk"], name="target_intermediate.risk")

    alignment_risk = risk  # 默认先把对齐风险设成risk。
    observation_risk = risk  # 观测风险默认也先沿用risk。
    quality_risk = risk  # 质量风险默认先沿用主 risk。
    modality_signal = 0.0  # 模、信号默认没有额外加成。
    target_trace = sample.get("target_trace")  # 取目标跟踪信息，可能为空。
    if target_trace is not None and not isinstance(target_trace, Mapping):
        raise TypeError(f"target_trace must be a mapping, got {type(target_trace).__name__}")
    if isinstance(target_trace, Mapping):  # 只有字典型 trace 才展开读取。
        alignment_risk = _clamp_unit_interval(  # 读取并裁剪对齐风险。
            target_trace.get("alignment_risk", alignment_risk),  # 没有就回到默认。
            name="target_trace.alignment_risk",  # 报错字段名。
        )

        observation_risk = _clamp_unit_interval(  # 读取并裁剪观测风险。
            target_trace.get("observation_risk", observation_risk),  # 没有就回到默认。
            name="target_trace.observation_risk",  # 报错字段名。
        )

        quality_risk = _clamp_unit_interval(  # 读取并裁剪质量风险。
            target_trace.get("quality_risk", quality_risk),  # 没有就回到默认。
            name="target_trace.quality_risk",  # 报错字段名。
        )

        modality_signal = _clamp_unit_interval(  # 读取并裁剪模态信号。
            target_trace.get("modality_signal", modality_signal),  # 没有就回到默认。
            name="target_trace.modality_signal",  # 报错字段名。
        )

    tail_signal = max(risk, alignment_risk, observation_risk)  # 尾部信号取几个风险里朢、强的那个。
    observation_tail = max(observation_risk, quality_risk, modality_signal)  # 观测尾部再综合质量和模信号。
    return min( # 最终再限制到上限。
        _TAIL_SELECTION_MAX_WEIGHT,  # 尾部权重绝不能超过硬上限。
        1.0  # 基础权重 1.0 开始。
        + (_TAIL_SELECTION_TAIL_COEFF * tail_signal)  # 尾部风险越高，权重越大。
        + (resolved_observation_coeff * observation_tail),  # 观测、质量和模、信号以较弱系数补充抬权。
    )  # 返回截断后的最终样本权重。




def _coerce_raw_output_vector(raw_output: Any) -> torch.Tensor:
    """将模型原始输出统一转为丢、维张量，并校验输出头数量。

    支持标量张量、一维张量和 batch=1 的二维张量作为输入，
    最终压平成固定长度len(_OUTPUT_KEYS) 的一维张量。

    参数：
        raw_output: 模型原始输出，可以是张量、列表或其他可转换类型。

    返回值：
        形状态(4,) 的一float32 张量，顺序为 bias, risk, uwb_scaling, vio_scaling。

    失败条件。
    ValueError: ?batch 维度不为 1、输出头数量不等 4 或存NaN/Inf 非有限、时抛出。
    """

    # 统一转成 float32 张量；张量输入时保留原设备且仅在 dtype 不一致时拷贝（与 lstm/trainer.py _project_train_outputs 口径一致，D4 设备/dtype 对齐、D7 公平性）。
    raw_tensor = torch.as_tensor(raw_output, dtype=torch.float32)
    if raw_tensor.ndim == 2:  # 如果是二维，通常表示batch 维。
        if raw_tensor.shape[0] != 1:  # 这里只允batch=1。
            raise ValueError("model output batch dimension must be 1 for trainer loss.")  # 不满足就报错。
        raw_tensor = raw_tensor.squeeze(0)  # 去掉单元batch 维。
        raw_tensor = raw_tensor.reshape(-1) # 然后压平成一维。
    if raw_tensor.numel() != len(_OUTPUT_KEYS):  # 输出头数量必须固定。
        raise ValueError(  # 不对就直接失败。
            "model output must contain the fixed four heads ordered as "  # 先说明必须的头序。
            "bias, risk, uwb_scaling, vio_scaling."
        )
    if not torch.isfinite(raw_tensor).all():  # 非有限、（NaN/Inf）直接报错，防止torch 操作静默穿、污染损失（D5 数、安全），与 lstm/trainer.py 口径一致（D7）。
        raise ValueError("model output contains non-finite values (NaN or Inf).")

    return raw_tensor  # 返回整理后的输出向量。




def _coerce_target_outputs(raw_target: Any) -> dict[str, float]:
    """将目标中间结果统一转成标准化的浮点数字典。

    支持 ModelIntermediate 对象、映射（dict 等）和带属、的
    任意对象作为输入。对每个输出头做有限性校验和值域裁剪。
    - bias 不允许为负；
    - risk 裁剪BRIDGE_THRESHOLDS 区间。
    - scaling 不允许低于中性下界1.0。

    参数：
        raw_target: 原始目标中间结果，可以是 ModelIntermediate。
            映射或带 bias/risk/uwb_scaling/vio_scaling 属、的对象。

    返回值：
    包含 "bias"?risk"?uwb_scaling"?vio_scaling" 四个键的
        浮点数字典，所有项都经过校验和裁剪。

    失败条件。
        KeyError: 当映射输入缺少必要字段时抛出。
        TypeError: 当输入类型无法识别时抛出。
        ValueError: 当任何字段为 NaN/Inf ?scaling 低于下限时抛出。
    """

    if isinstance(raw_target, ModelIntermediate):  # 如果已经是统一中间结果对象，就直接取字段。
        source = {  # 先把中间结果转成普、映射，后面统一处理。
            "bias": raw_target.bias,  # 偏置值。
            "risk": raw_target.risk,  # 风险值。
            "uwb_scaling": raw_target.uwb_scaling,  # UWB 缩放值。
            "vio_scaling": raw_target.vio_scaling,  # VIO 缩放值。
        }  # 中间结果字段提取结束。
    elif isinstance(raw_target, Mapping):  # 如果本来就是映射，就直接读取。
        missing_keys = [key for key in _OUTPUT_KEYS if key not in raw_target]  # 先找出缺失字段。
        if missing_keys:  # 少字段就不继续往下走。
            raise KeyError(f"target_intermediate is missing required keys: {missing_keys}")  # 直接报错提醒。
        source = raw_target  # 直接复用原映射。
    else:  # 既不是中间对象，也不是映射时，尝试按属、读取。
        source = {}  # 先准备一个空字典承接属性。
        for key in _OUTPUT_KEYS:  # 逐个输出头去对象上取值。
            value = getattr(raw_target, key, None)  # 如果没有对应属、就返回 None。
            if value is None:  # 缺任何一个都不能继续。
                raise TypeError(  # 不符合输入契约就直接报错。
                    "target_intermediate must be a mapping, ModelIntermediate, or object exposing "  # 说明允许的输入形式。
                    f"{_OUTPUT_KEYS}; got {type(raw_target).__name__}"  # 报告实际类型。
                )  # 类型错误结束。
            source[key] = value  # 把属性、收进统一字典。


    bias = float(source["bias"])  # 偏置值转成浮点数。
    risk = float(source["risk"])  # 风险值转成浮点数。
    uwb_scaling = float(source["uwb_scaling"])  # UWB 缩放值转成浮点数。
    vio_scaling = float(source["vio_scaling"])  # VIO 缩放值转成浮点数。
    for name, value in (  # 逐个检查四个输出头的有限。
        ("bias", bias),  # 偏置校正量。
        ("risk", risk),  # 风险。。
        ("uwb_scaling", uwb_scaling),  # UWB 缩放校正量。
        ("vio_scaling", vio_scaling),  # VIO 缩放校正量。
    ):  # 閫愰」妫€鏌ョ粨条熴€。
        coerce_finite_scalar(value, name=f"target_intermediate.{name}")  # D2：任何一个都不能NaN 或无穷大，有限、校验统一走中央入口。
    bias = max(0.0, bias)  # 偏置不允许为负。
    risk = min(BRIDGE_THRESHOLDS["risk_max"], max(BRIDGE_THRESHOLDS["risk_min"], risk))  # 风险裁剪到训练阈值区间。
    if uwb_scaling < _SCALING_NEUTRAL_FLOOR:  # 缩放值不能低于中性下限。
        raise ValueError("target_intermediate.uwb_scaling must be >= 1.0")  # 低于下限就报错。
    if vio_scaling < _SCALING_NEUTRAL_FLOOR:  # 缩放值不能低于中性下限。
        raise ValueError("target_intermediate.vio_scaling must be >= 1.0")  # 低于下限就报错。
    return {  # 返回统一整理后的目标字典。
        "bias": bias,  # 偏置校正量。
        "risk": risk,  # 风险。。
        "uwb_scaling": uwb_scaling,  # UWB 缩放校正量。
        "vio_scaling": vio_scaling,  # VIO 缩放校正量。
    }  # 目标字典结束。




def _project_train_scaling(raw_scaling: torch.Tensor) -> torch.Tensor:
    """将原始缩放输出投影到训练区间，保证不低于协议中、下限。

    委托给规范实``neutral_floor_softplus``，并显式传入
    ``BRIDGE_THRESHOLDS["scaling_min"]`` 作为下限单源真相，与
    ``inference.py::_clamp_protocol_scaling`` 口径一致（D7 公平性/ D9 单源）。
    上界由规范实现默认从 ``BRIDGE_THRESHOLDS["scaling_max"]`` 读取。
    straight-through 梯度估计clamp 路径均封装在规范实现内，本函数不再本地实现。

    参数：
        raw_scaling: 原始缩放输出标量张量。

    返回值：
        投影后的缩放标量张量，、落。
        ``[BRIDGE_THRESHOLDS["scaling_min"], BRIDGE_THRESHOLDS["scaling_max"]]``。。
    """
    # D7/D9：下限显式取BRIDGE_THRESHOLDS["scaling_min"]，禁止本地字面量与协议单源真相漂移。
    return neutral_floor_softplus(
        raw_scaling,
        neutral_floor=float(BRIDGE_THRESHOLDS["scaling_min"]),
    )


def _normalize_liquid_train_outputs(
    raw_outputs: Mapping[str, Any],
    *,
    risk_already_normalized: bool = False,
) -> dict[str, torch.Tensor]:
    """将模型原始输出字典标准化为训练、输出字典。

    ?bias 直接转张量，risk 调用 _project_train_risk 做区间映射，
    ?scaling 调用 _project_train_scaling 做下限保护。

    参数：
        raw_outputs: 模型原始输出字典，必须包bias、risk。
            uwb_scaling、vio_scaling 四个。
        risk_already_normalized: risk 是否已经归一化到 [0, 1]。

    返回值：
        包含四个标量张量的字典，值已投影到训练区间。
    """
    missing_keys = [key for key in _OUTPUT_KEYS if key not in raw_outputs] # D9：键名走 _OUTPUT_KEYS?MODEL_INTERMEDIATE_KEYS）单源真相，禁止字面量漂移。
    if missing_keys:  # 缺键是数据合同问题，应抛 ValueError 而非 KeyError，与 inference.py _normalize_intermediate_outputs 口径一致。
        raise ValueError(f"raw_outputs is missing required keys: {missing_keys}")

    bias_raw = torch.as_tensor(raw_outputs["bias"], dtype=torch.float32).reshape(())  # bias 原始值转标量张量。

    risk_raw = torch.as_tensor(raw_outputs["risk"], dtype=torch.float32).reshape(())  # risk 原始值转标量张量。

    uwb_scaling_raw = torch.as_tensor(raw_outputs["uwb_scaling"], dtype=torch.float32).reshape(())  # UWB 缩放原始值转标量张量。

    vio_scaling_raw = torch.as_tensor(raw_outputs["vio_scaling"], dtype=torch.float32).reshape(())  # VIO 缩放原始值转标量张量。

    # D5 数、安全：NaN/Inf 会经 torch.clamp/softplus 静默穿、（clamp(NaN)=NaN、F.softplus(NaN)=NaN），污染batch 梯度。
    # 训练态需保留梯度路径，故torch.isfinite 守护而非 coerce_finite_scalar（后者返回Python float ?detach 梯度）；
    # ?LSTM ?_project_train_outputs（lstm/trainer.py L222）及本文_project_train_risk（L248）口径一致（D7 公平性）。
    if not (torch.isfinite(bias_raw) and torch.isfinite(risk_raw) and torch.isfinite(uwb_scaling_raw) and torch.isfinite(vio_scaling_raw)):
        raise ValueError("raw_outputs contain non-finite values (NaN or Inf)")

    bias = torch.clamp(bias_raw, min=0.0, max=BRIDGE_BIAS_MAX)  # bias 非负且上限与推理侧对齐（L1 根因修复：训推理 clamp 口径一致，对齐 docs/loss_function.md §核心约束 3）。

    uwb_scaling = _project_train_scaling(uwb_scaling_raw)  # UWB 缩放投影到训练区间。

    vio_scaling = _project_train_scaling(vio_scaling_raw)  # VIO 缩放投影到训练区间。

    return {

        "bias": bias,  # bias 偏置校正量。

        "risk": _project_train_risk(risk_raw, already_normalized=risk_already_normalized),  # risk 投影到训练区间。

        "uwb_scaling": uwb_scaling,  # UWB 缩放校正量。

        "vio_scaling": vio_scaling,  # VIO 缩放校正量。

    }


def _project_train_risk_batch(raw_risk: torch.Tensor, *, already_normalized: bool = False) -> torch.Tensor:
    """将批量原始风险输出投影到训练区间（批量版本）。

    ?_project_train_risk 逻辑一致，但接（B,) 张量而非标量。

    参数：
        raw_risk: 原始风险输出张量，形状(B,)。
        already_normalized: 是否已经归一化到 [0, 1]。

    返回值：
        映射到训练区间的风险张量，形状(B,)。
    """
    risk_span = BRIDGE_THRESHOLDS["risk_max"] - BRIDGE_THRESHOLDS["risk_min"]
    risk = torch.as_tensor(raw_risk, dtype=torch.float32)  # 保持 (B,) 形状。
    if not torch.isfinite(risk).all():  # 与标量版一致：NaN/Inf 防护。
        raise ValueError(f"raw_risk must be finite, got {risk}")
    if not already_normalized:
        risk = torch.sigmoid(risk)
    if risk_span != 1.0 or BRIDGE_THRESHOLDS["risk_min"] != 0.0:
        risk = risk * risk_span + BRIDGE_THRESHOLDS["risk_min"]
    # 与推理侧保持一致：末尾 clamp 到协议区间。
    risk = torch.clamp(risk, BRIDGE_THRESHOLDS["risk_min"], BRIDGE_THRESHOLDS["risk_max"])
    return risk


def _project_train_scaling_batch(raw_scaling: torch.Tensor) -> torch.Tensor:
    """将批量原始缩放输出投影到训练区间（批量版本）。

    ?_project_train_scaling 逻辑一致，但接（B,) 张量而非标量。
    委托给规范实``neutral_floor_softplus``，并显式传入
    ``BRIDGE_THRESHOLDS["scaling_min"]`` 作为下限单源真相，与
    ``inference.py::_clamp_protocol_scaling`` 口径一致（D7 公平性/ D9 单源）。
    上界由规范实现默认从 ``BRIDGE_THRESHOLDS["scaling_max"]`` 读取。
    NaN/Inf 拦截由规范实现统一负责（D5），避免批量/标量口径漂移。

    参数：
        raw_scaling: 原始缩放输出张量，形状(B,)。

    返回值：
        投影后的缩放张量，形状(B,)，、落。
        ``[BRIDGE_THRESHOLDS["scaling_min"], BRIDGE_THRESHOLDS["scaling_max"]]``。。
    """
    # D7/D9：下限显式取BRIDGE_THRESHOLDS["scaling_min"]，与标量版口径一致，禁止本地字面量与协议单源真相漂移。
    return neutral_floor_softplus(
        raw_scaling,
        neutral_floor=float(BRIDGE_THRESHOLDS["scaling_min"]),
    )


def _normalize_liquid_train_outputs_batch(
    raw_batch_outputs: Mapping[str, torch.Tensor],
    *,
    risk_already_normalized: bool = False,
) -> dict[str, torch.Tensor]:
    """将批量模型原始输出字典标准化为训练、输出字典。

    批量版本_normalize_liquid_train_outputs，接（B,) 形状的张量。

    参数：
        raw_batch_outputs: 批量原始输出字典，每个、是 (B,) 张量。
        risk_already_normalized: risk 是否已经归一化到 [0, 1]。

    返回值：
        包含四个 (B,) 张量的字典，值已投影到训练区间。
    """
    missing_keys = [key for key in _OUTPUT_KEYS if key not in raw_batch_outputs] # D9：键名走 _OUTPUT_KEYS?MODEL_INTERMEDIATE_KEYS）单源真相，禁止字面量漂移，与标量版口径一致。
    if missing_keys:  # 缺键是数据合同问题，应抛 ValueError 而非 KeyError，与 inference.py _normalize_intermediate_outputs 口径一致。
        raise ValueError(f"raw_batch_outputs is missing required keys: {missing_keys}")

    bias_raw = torch.as_tensor(raw_batch_outputs["bias"], dtype=torch.float32)
    risk_raw = torch.as_tensor(raw_batch_outputs["risk"], dtype=torch.float32)
    uwb_scaling_raw = torch.as_tensor(raw_batch_outputs["uwb_scaling"], dtype=torch.float32)
    vio_scaling_raw = torch.as_tensor(raw_batch_outputs["vio_scaling"], dtype=torch.float32)

    # D5 数、安全：NaN/Inf 会经 torch.clamp/softplus 静默穿、（clamp(NaN)=NaN、F.softplus(NaN)=NaN），污染batch 梯度。
    # 训练态需保留梯度路径，故torch.isfinite 守护而非 coerce_finite_scalar（后者返回Python float ?detach 梯度）；
    # 与标量版 _normalize_liquid_train_outputs 及本文件 _project_train_risk_batch 口径一致（D7 公平性）。
    if not (torch.isfinite(bias_raw).all() and torch.isfinite(risk_raw).all() and torch.isfinite(uwb_scaling_raw).all() and torch.isfinite(vio_scaling_raw).all()):
        raise ValueError("raw_batch_outputs contain non-finite values (NaN or Inf)")

    bias = torch.clamp(bias_raw, min=0.0, max=BRIDGE_BIAS_MAX)  # bias 非负且上限与推理侧对齐（L1 根因修复：训推理 clamp 口径一致，对齐 docs/loss_function.md §核心约束 3）。
    uwb_scaling = _project_train_scaling_batch(uwb_scaling_raw)  # UWB 缩放投影到训练区间。
    vio_scaling = _project_train_scaling_batch(vio_scaling_raw)  # VIO 缩放投影到训练区间。
    return {
        "bias": bias,
        "risk": _project_train_risk_batch(risk_raw, already_normalized=risk_already_normalized),
        "uwb_scaling": uwb_scaling,
        "vio_scaling": vio_scaling,
    }





def _extract_window_and_target(sample: Any) -> tuple[dict[str, Any], dict[str, float]]:
    """从原始样本中分离特征窗口和目标中间结果。

    样本可以是完整的映射（含 window_tensor ?target_intermediate），
    也可以直接就是窗口映射本身（此时 target_intermediate 必须在顶层）。

    参数：
        sample: 原始训练/验证样本。

    返回值：
    二元（window_tensor, target_outputs)。
        - window_tensor: 特征窗口字典（浅拷贝；窗口标准化由调用方负责。
        ?LSTM 侧在函数内即标准化的口径不同，Liquid 推迟。
          _materialize_samples 调用 normalize_window_tensor 完成）；
        - target_outputs: 校验后的目标输出字典。

    失败条件。
        TypeError: 当样本或窗口不是映射类型时抛出。
        ValueError: 当样本缺target_intermediate 时抛出。
    """

    if not isinstance(sample, Mapping):

        raise TypeError("train/val samples must be mappings containing a structured window")



    window_tensor = sample.get("window_tensor")

    if window_tensor is None:

        window_tensor = sample

    if not isinstance(window_tensor, Mapping):  # 窗口必须是字典式映射。
        raise TypeError("window_tensor must be a structured feature window mapping")  # 类型不对就报错。


    raw_target = sample.get("target_intermediate")  # 取训练目标中间结果。
    if raw_target is None:  # 每个样本都必须带目标。
        raise ValueError("each train/val sample must contain target_intermediate")  # 缺失就报错。
    return dict(window_tensor), _coerce_target_outputs(raw_target)  # 返回窗口浅拷贝和目标字典；窗口标准化由调用方负责（与 LSTM 侧口径不同）。




def _materialize_samples(
    samples: list[Any],
    *,
    tail_selection_observation_coeff: float = _TAIL_SELECTION_OBSERVATION_COEFF,
) -> list[tuple[dict[str, Any], torch.Tensor, int, float]]:
    """将原始样本列表材料化为训练器可直接消费的格式。

    对每个样本：分离窗口和目标、标准化窗口结构、校验特征顺序一致。
    按固定顺序构造目标张量、计算、样权重。返回的四元组包。
    (元数。 目标张量, 序列长度, 选样权重)。

    参数：
        samples: 原始训练/验证样本列表。

    返回值：
        材料化样本列表，每个元素是四元组。
        (metadata_dict, target_tensor, seq_len, selection_weight)。。

    失败条件。
        ValueError: 当不同样本的 feature_order 不一致时抛出。
    """

    resolved_tail_selection_observation_coeff = coerce_finite_scalar(
        tail_selection_observation_coeff,
        name="tail_selection_observation_coeff",
        min_value=0.0,
    )

    targets: list[torch.Tensor] = []  # 存每个样本的目标张量。
    metadata: list[dict[str, Any]] = []  # 存每个样本的窗口元数据。
    selection_weights: list[float] = []  # 存每个样本的选样权重。
    feature_order: list[str] | None = None  # 先用首个样本锁定特征顺序。
    for sample in samples:  # 逐个样本整理成训练器霢、要的格式。
        window_tensor, target_outputs = _extract_window_and_target(sample)  # 分离窗口和目标。
        normalized = normalize_window_tensor(window_tensor)  # 标准化窗口结构。
        if feature_order is None:  # 第一条样本决定特征顺序。
            feature_order = list(normalized["feature_order"])  # 记录基准顺序。
        elif list(normalized["feature_order"]) != feature_order:  # 后续样本必须完全致。
            raise ValueError("all liquid training windows must share the same feature_order")  # 不一致就报错。
        targets.append(torch.tensor([target_outputs[key] for key in _OUTPUT_KEYS], dtype=torch.float32))  # 目标按固定顺序入张量。
        metadata.append(normalized)  # 保存标准化后的窗口元数据。
        selection_weights.append(
            _resolve_selection_sample_weight(
                sample,
                target_outputs,
                observation_coeff=resolved_tail_selection_observation_coeff,
            )
        )  # 计算样本权重。
    return [  # 把三路结果重新打包成材料化样本。
        (meta, target, len(meta["feature_window"]), selection_weight)  # 元数据、目标、长度、权重。
        for meta, target, selection_weight in zip(metadata, targets, selection_weights)  # 按样本、个打包。
    ]  # 材料化样本结束。




def _resolve_sample_cache_device(model: Any) -> torch.device:
    """根据模型实际参数所在的设备，决定样本缓存的目标设备。

    §13.8 OOM 根因修复: 原始代码将 cache_device 硬编码为 torch.device('cpu')，
    导致 1860 序列的全部训练样本 (~2.5M 个) 的 window_tensor / target_intermediate
    / target_trace 全部常驻 CPU 内存，峰值 RSS 超出可用内存而崩溃。
    改为与模型同一设备: 模型在 cuda 时缓存到 GPU，模型在 cpu 时缓存到 cpu。
    GPU 7.96 GB 显存足够装 1860 序列样本 (~5 GB), CPU 内存同时释放。

    参数：
        model: Liquid 模型实例，其参数决定了目标设备。

    返回值：
        torch.device: 与模型参数同设备的目标设备。
    """
    try:
        # 从任意一个参数的 device 推断（模型已在 L5162+ 移动到 runtime_device）
        p = next(model.parameters()) if hasattr(model, 'parameters') and callable(model.parameters) else None
        if p is not None:
            return p.device
        # 无参数（模型为空）时回退 cpu
        return torch.device("cpu")
    except (StopIteration, TypeError):
        return torch.device("cpu")


def _move_materialized_samples(

    materialized: list[tuple[dict[str, Any], torch.Tensor, int, float]],

    device: torch.device,

) -> list[tuple[dict[str, Any], torch.Tensor, int, float]]:
    """将材料化样本中所有张量搬到目标设备。

    包括特征窗口、缺失掩码窗口特征、缺失掩码。
    读出上下文等所有张量字段，以及目标张量本身。

    参数：
        materialized: 材料化样本列表。
        device: 目标设备（如 cuda ?cpu）。

    返回值：
        搬运后的材料化样本列表，结构不变，张量在目标设备上。
    """

    moved_samples: list[tuple[dict[str, Any], torch.Tensor, int, float]] = []  # 存搬到目标设备后的样本。
    for metadata, target_tensor, seq_len, selection_weight in materialized:  # 逐个样本搬设备。
        moved_metadata = dict(metadata)  # 先拷贝元数据，避免原地改动。
        moved_metadata["feature_window"] = torch.as_tensor(moved_metadata["feature_window"], dtype=torch.float32, device=device)  # 特征窗口搬到目标设备。
        moved_metadata["missing_mask_window"] = torch.as_tensor(moved_metadata["missing_mask_window"], dtype=torch.float32, device=device)  # 缺失掩码窗口搬设备。
        moved_metadata["feature_values"] = torch.as_tensor(moved_metadata["feature_values"], dtype=torch.float32, device=device)  # 特征值搬设备。
        moved_metadata["missing_mask"] = torch.as_tensor(moved_metadata["missing_mask"], dtype=torch.float32, device=device)  # 缺失掩码搬设备。
        if "readout_context_by_name" in moved_metadata:  # 如果有读出上下文字典。
            moved_metadata["readout_context_by_name"] = {  # 把每个、都搬到目标设备。
                str(key): torch.as_tensor(value, dtype=torch.float32, device=device).reshape(())  # 转成标量张量。
                for key, value in dict(moved_metadata["readout_context_by_name"]).items()  # 遍历每个键、对。
            }
        if "readout_context_observed_by_name" in moved_metadata:  # 如果有读出上下文观测标记字典。
            moved_metadata["readout_context_observed_by_name"] = {  # 把每个、转成布尔。
                str(key): bool(value)  # 观测标记是布尔、，不需要搬设备。
                for key, value in dict(moved_metadata["readout_context_observed_by_name"]).items()  # 遍历每个键、对。
            }
        moved_samples.append((moved_metadata, target_tensor.to(device), seq_len, selection_weight))  # 追加搬运后的样本。
    return moved_samples  # 返回搬运结果。




def _resolve_feature_order(train_windows: list[Any], trainer_state: dict[str, Any]) -> list[str]:
    """解析特征顺序，优先使用配置中的显式顺序，否则从第丢、条窗口推导。

    参数：
        train_windows: 训练窗口列表，至少包含含一个元素。
        trainer_state: 训练器状态字典，可能包含 "feature_order" 键。

    返回值：
        特征名字符串列表。

    失败条件。
        ValueError: 当既没有配置也没有内嵌特征顺序，或字段名为空/重复时抛出。
        TypeError: ?feature_order 是字符串/字节/映射或不可迭代对象时抛出。
    """

    raw_feature_order = trainer_state["feature_order"]  # 取显式配置（可能为空列表）。
    if not raw_feature_order:  # 状里没有显式顺序时，从窗口推导。
        if not train_windows:  # 既没配置也没有窗口，就无法推导。
            raise ValueError("feature_order must be provided in train_cfg or in the training windows")
        window_tensor, _ = _extract_window_and_target(train_windows[0])  # 用第丢、条训练窗口推导顺序。
        raw_feature_order = window_tensor.get("feature_order")  # 从窗口里读特征顺序。

        # ?lstm/network.py _coerce_feature_order、liquid/network.py normalize_window_tensor 对齐。
    # 字符字节字节数组本身可迭代但会被拆成字符/整数序列，必须拒绝；
    # 映射（dict）会静默用键当字段顺序，掩盖调用方传错意图，也必须拒绝。
    # is_string_like 覆盖 numpy.str_（NumPy 2.x 不再str 子类）。
    if (
        is_string_like(raw_feature_order)
        or isinstance(raw_feature_order, (bytes, bytearray))
        or isinstance(raw_feature_order, Mapping)
    ):
        raise TypeError("feature_order must be an iterable of feature names.")
    try:  # 把任意可迭代对象转成 list，固化后避免外部迭代器被耗尽。
        feature_order = list(raw_feature_order or [])
    except TypeError as exc:  # 不可迭代时捕获原始异常再抛出更明确的错误。
        raise TypeError("feature_order must be an iterable of feature names.") from exc
    if not feature_order:  # 既没配置也没内嵌，就不能继续。
        raise ValueError("feature_order must be provided in train_cfg or in the training windows")
    for index, feature_name in enumerate(feature_order):  # 逐个检查特征名。
        if not is_string_like(feature_name) or not feature_name:  # 必须是非空字符串。
            raise ValueError(f"feature_order[{index}] must be a non-empty string.")
    if len(set(feature_order)) != len(feature_order):  # 重复名会导致列对齐静默错位，LiquidNetwork.__init__ 对齐。
        raise ValueError("feature_order must not contain duplicate field names.")
    return feature_order  # 返回校验后的特征顺序。




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
    phase_state = {  # 收集各模块的可训练状态。
        "network_trainable": any(parameter.requires_grad for parameter in model.network.parameters()),  # 共享网络是否可训练。
        "risk_calibration_trainable": _risk_calibration_trainable(model), # risk_calibration 是否可训练（与辅助损失判_risk_calibration_trainable 同口径，避免漂移）。
    }
    # v3.1 head-shared: gate 在 output_backbone，projection 在 output_heads。统一从两个对象快照。
    reference_head = next(iter(getattr(model, "output_heads", {}).values()), None)  # 取第一个 final-projection head 作为代表。
    reference_backbone = getattr(model, "output_backbone", None)  # v3.1 共享 backbone。
    if reference_head is None and reference_backbone is None:  # 两个都没有就不能继续。
        raise ValueError("liquid model must expose at least one output head or output_backbone")  # 直接报错。
    # 拼合：head 提供 projection/residual_projection 可训练性；backbone 提供 6 个 gate 可训练性。
    head_snapshot = _head_phase_trainability_snapshot(reference_head) if reference_head is not None else {}
    backbone_snapshot = _head_phase_trainability_snapshot(reference_backbone) if reference_backbone is not None else {}
    phase_state["head_trainability"] = {**backbone_snapshot, **head_snapshot}  # head 覆盖 backbone 中同名键（实际无同名，因 head 已无 gate）。
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
        "aux_scale": _safe_finite_float(phase_control_state.get("aux_scale")),
        "gate_scale": _safe_finite_float(phase_control_state.get("gate_scale")),
        "calibration_scale": _safe_finite_float(phase_control_state.get("calibration_scale")),
        "reentry_release_scale": _safe_finite_float(phase_control_state.get("reentry_release_scale")),
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
    if request == DEVICE_AUTO:  # 自动选择模式。
        if cuda_runtime_usable():  # CUDA 可用。
            return torch.device(DEVICE_CUDA), DEVICE_CUDA  # 使用 CUDA。
        return torch.device(DEVICE_CPU), DEVICE_CPU  # CUDA 不可用，回 CPU。
    if request == DEVICE_CPU:  # 显式请求 CPU。
        return torch.device(DEVICE_CPU), DEVICE_CPU
    if request == DEVICE_CUDA:  # 显式请求 CUDA。
        if cuda_runtime_usable():  # CUDA 可用。
            return torch.device(DEVICE_CUDA), DEVICE_CUDA  # 使用 CUDA。
        return torch.device(DEVICE_CPU), DEVICE_CPU  # CUDA 不可用，回 CPU。
    if request.startswith(DEVICE_CUDA_PREFIX):  # 显式请求指定 CUDA 设备。
        if cuda_runtime_usable():  # CUDA 可用。
            return torch.device(request), request  # 使用指定 CUDA 设备。
        return torch.device(DEVICE_CPU), DEVICE_CPU  # CUDA 不可用，回 CPU。
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
    amp_requested = train_cfg.get("amp_enabled", "auto")  # 读取 AMP 开关，默认自动。
    amp_dtype = str(train_cfg.get("amp_dtype", "auto")).strip().lower()  # 读取精度类型，默认自动。
    if runtime_device.startswith("cuda"):  # CUDA 设备上才考虑启用 AMP。
        if amp_requested == "auto":  # 自动模式。
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





def _compute_loss(model: Any, window_tensor: Mapping[str, Any], target_outputs: Mapping[str, float]) -> torch.Tensor:
    """计算单样本的监督损失（不含辅助项）。

    先做前向推理拿到预测输出，再按模态构造掩码，
    朢、后调用_masked_supervised_loss 计算加权损失。

    参数：
        model: 液体模型实例，必须有 predict_intermediate_tensors 方法。
        window_tensor: 结构化特征窗口映射。
        target_outputs: 目标输出字典。

    返回值：
        单样本的监督损失标量张量。
    """
    modality = _resolve_current_modality(window_tensor)  # 解析当前模。

    predicted_outputs = model.predict_intermediate_tensors(window_tensor)  # 前向推理拿到预测输出。

    prediction_vector = torch.stack(  # 按固定顺序拼成预测向量。

        [

            predicted_outputs[key].reshape(())  # 每个输出头取标量。

            for key in _OUTPUT_KEYS # ?_OUTPUT_KEYS?MODEL_INTERMEDIATE_KEYS）单源真相顺序，禁止字面量漂移。

        ]

    )

    target_vector = prediction_vector.new_tensor([coerce_finite_scalar(target_outputs[key], name=f"target_outputs.{key}") for key in _OUTPUT_KEYS])  # 目标也按相同顺序构、向量；lstm/trainer.py._compute_loss 口径对齐：拒NaN/Inf（D5/D7）。

    return _masked_supervised_loss(

        prediction_vector.unsqueeze(0),

        target_vector.unsqueeze(0),

        modalities=[modality],
        bias_huber_delta=coerce_finite_scalar(getattr(model, "_bias_huber_delta", _BIAS_HUBER_DELTA), name="model._bias_huber_delta", min_value=0.0),
        scaling_log_eps=coerce_finite_scalar(getattr(model, "_scaling_log_eps", _SCALING_LOG_EPS), name="model._scaling_log_eps", min_value=0.0),

    )





def _compute_batch_loss(

    model: Any,

    target_batch: torch.Tensor,

    metadata: list[dict[str, Any]],

    *,

    sample_weights: list[float] | None = None,

) -> torch.Tensor:
    """计算一个批次的监督损失加辅助损失。

    先预测当前批次，再计算四头监督损失和 Liquid 辅助损失。
    最后返回两者之和。

    参数：
        model: 液体模型实例。
        target_batch: 目标值批次，形状 (B, 4)。
        metadata: 当前批次每个样本的元数据列表。
        sample_weights: 样本权重列表，None 时默认等权。

    返回值：
        监督损失 + 辅助损失的标量张量。
    """

    prediction_batch = _predict_batch(model, metadata)  # 先预测当前批次。
    # D7/D3：与 lstm/trainer.py 同名函数口径对齐，先校验 target_batch 形状。
    # 避免错误形状静默穿、到 _masked_supervised_loss 后被广播掩盖。
    if target_batch.ndim != 2 or target_batch.shape[1] != len(_OUTPUT_KEYS):
        raise ValueError(
            f"target_batch must have shape (batch, {len(_OUTPUT_KEYS)}), got {tuple(target_batch.shape)}"
        )
    target_batch = target_batch.to(device=prediction_batch.device, dtype=prediction_batch.dtype)
    modalities = [_resolve_current_modality(sample_meta) for sample_meta in metadata]  # 逐个样本读模态。
    # D5/D7：与 lstm/trainer.py ?_compute_liquid_auxiliary_loss_terms 一致，
    # 统一coerce_finite_scalar 中心入口；float() 会把 NaN/Inf/负、静默穿透到损失项。
    supervised_loss = _masked_supervised_loss(  # 先算基础四头监督损失。
        prediction_batch,  # 预测批次。
        target_batch,  # 目标批次。
        modalities=modalities,  # 当前批次的模态列表。
        sample_weights=sample_weights,  # 当前批次的样本权重。
        bias_huber_delta=coerce_finite_scalar(
            getattr(model, "_bias_huber_delta", _BIAS_HUBER_DELTA),
            name="bias_huber_delta",
            min_value=0.0,
        ),
        scaling_log_eps=coerce_finite_scalar(
            getattr(model, "_scaling_log_eps", _SCALING_LOG_EPS),
            name="scaling_log_eps",
            min_value=0.0,
        ),
    )
    auxiliary_terms = _compute_liquid_auxiliary_loss_terms(
        model,
        prediction_batch,
        target_batch,
        modalities=modalities,
        sample_weights=sample_weights,
    )
    return supervised_loss + auxiliary_terms["total"]  # 把默认启用的 Liquid 辅助项一起并入损失。




def _predict_batch(model: Any, metadata: list[dict[str, Any]]) -> torch.Tensor:
    """对一个批次的样本做前向预测，返回标准化后的预测批次张量。

    批量提取共享特征（cell 递推做真正的批量矩阵运算），
    批量通过输出头（混合模、过模、指示器处理），
    再做 risk 校准、标准化和模态合约。

    参数：
        model: 液体模型实例。
        metadata: 当前批次每个样本的元数据列表。

    返回值：
        形状 (B, 4) 的预测批次张量，列顺序为 _OUTPUT_KEYS。
    """

    # 批量提取共享特征：cell 递推做真正的批量矩阵运算。
    shared_features_list = model.network.extract_shared_features_batch(
        metadata,
        minimal_output_head_features=True,
    )

    if len(shared_features_list) == 1:  # 单样本走逐样本路径。
        sf = shared_features_list[0]
        # v3.1 head-shared: 通过 _LiquidModel.run_head_forward 共享 backbone 再走 4 个 final-projection head
        if hasattr(model, "run_head_forward"):
            raw_outputs = model.run_head_forward(sf)
        else:
            raw_outputs = {key: head(sf) for key, head in model.output_heads.items()}
        risk_already_normalized = False
        calibration = getattr(model, "risk_calibration", None)
        if callable(calibration):
            # D5/D7：与 lstm/trainer.py _project_train_outputs（L222）口径一致。
            # 校准前先校验原始 risk 的有限、sigmoid(Inf)=1.0 / sigmoid(-Inf)=0.0 。
            # ?Inf 静默映射到有限、，掩盖模型前向的数值不稳定（如梯度爆炸），
            # 下游 _normalize_liquid_train_outputs 的有限、检查此时无法再捕获原始 Inf。
            _raw_risk_pre_calib = torch.as_tensor(raw_outputs["risk"], dtype=torch.float32).reshape(())
            if not torch.isfinite(_raw_risk_pre_calib):
                raise ValueError(f"raw risk output must be finite before calibration, got {_raw_risk_pre_calib.item()}")
            raw_outputs = dict(raw_outputs)
            raw_outputs["risk"] = calibration(raw_outputs["risk"])
            risk_already_normalized = True
        predicted_outputs = _normalize_liquid_train_outputs(
            raw_outputs, risk_already_normalized=risk_already_normalized,
        )
        predicted_outputs = apply_liquid_modality_output_contract(
            predicted_outputs, modality=_resolve_current_modality(sf),
        )
        return torch.stack([predicted_outputs[key].reshape(()) for key in _OUTPUT_KEYS]).unsqueeze(0)

    # 批量通过输出头：混合模、过模、指示器处理。
    raw_batch_outputs: dict[str, torch.Tensor] = {}  # 每个头的 (B,) 输出。
    if hasattr(model, "run_head_forward_batch"):
        # v3.1 head-shared: 共享 backbone + 4 个 final-projection head
        raw_batch_outputs = model.run_head_forward_batch(shared_features_list)
    else:
        for key, head in model.output_heads.items():
            raw_batch_outputs[key] = head.forward_batch(shared_features_list)  # (B,)

    # 批量 risk 标准分箱结果。
    risk_already_normalized = False
    calibration = getattr(model, "risk_calibration", None)
    if callable(calibration):
        # D5/D7：与标量版及 lstm/trainer.py _project_train_outputs（L222）口径一致。
        # 校准前先校验原始 risk 批次张量的有限、，避免 sigmoid ?Inf 静默映射。
        # 有限值（sigmoid(Inf)=1.0 / sigmoid(-Inf)=0.0），掩盖模型前向数、不稳定。
        # 下游 _normalize_liquid_train_outputs_batch 的有限、检查此时无法再捕获原始 Inf。
        _raw_risk_batch_pre_calib = torch.as_tensor(raw_batch_outputs["risk"], dtype=torch.float32)
        if not torch.isfinite(_raw_risk_batch_pre_calib).all():
            raise ValueError(f"raw risk batch output must be finite before calibration, got {_raw_risk_batch_pre_calib}")
        raw_batch_outputs = dict(raw_batch_outputs)
        raw_batch_outputs["risk"] = calibration(raw_batch_outputs["risk"])  # (B,) →(B,)
        risk_already_normalized = True

    # 批量标准化训练、输出。
    normalized_batch = _normalize_liquid_train_outputs_batch(
        raw_batch_outputs, risk_already_normalized=risk_already_normalized,
    )  # dict[str, (B,)]

    # 逐样本应用模态输出合约。
    modalities = [_resolve_current_modality(sf) for sf in shared_features_list]
    prediction_rows: list[torch.Tensor] = []
    for i in range(len(shared_features_list)):
        per_sample = {key: normalized_batch[key][i] for key in _OUTPUT_KEYS}
        per_sample = apply_liquid_modality_output_contract(per_sample, modality=modalities[i])
        prediction_rows.append(torch.stack([per_sample[key].reshape(()) for key in _OUTPUT_KEYS]))

    return torch.stack(prediction_rows, dim=0)  # (B, 4)




def _loss_for_length_group(model: Any, samples: list[tuple[dict[str, Any], torch.Tensor, int, float]]) -> torch.Tensor:
    """计算同一序列长度组内所有样本的批次损失。

    同组样本序列长度相同，可以直接堆叠成批次做前向和损失计算。

    参数：
        model: 液体模型实例。
        samples: 同组材料化样本列表，每个元素是四元组。

    返回值：
        该组的批次损失标量张量。
    """

    if not samples:  # D7：与 lstm/trainer.py 同名函数口径对齐，空组直接报错、非torch.stack 抛出误导性异常。
        raise ValueError("_loss_for_length_group requires at least one sample, got empty list")
    sample_weights = [sample[3] for sample in samples]  # 提前抽取权重，便NaN/Inf 预检与下游复用。
    if not all(math.isfinite(float(weight)) for weight in sample_weights):  # D5/D7：NaN/Inf 权重在进入批次前就拦截，lstm 侧口径一致。
        raise ValueError(f"non-finite selection_weight detected in length group: {sample_weights}")
    target_batch = torch.stack([sample[1] for sample in samples], dim=0)  # 把同组目标堆batch。
    return _compute_batch_loss(  # 直接复用批次损失计算。
        model,  # 当前模型。
        target_batch,  # 该组目标。
        [sample[0] for sample in samples],  # 该组元数据。
        sample_weights=sample_weights,  # 该组样本权重。
    )  # 长度组损失结束。




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





def train_one_epoch(train_windows, val_windows, epoch_index: int, *, model=None, optimizer=None) -> float:
    """执行一个完整的训练 epoch，返回平均训练损失。

    将材料化样本按序列长度分组，同组样本可以堆叠成批。
    做前向和反向传播。每mini-batch 做一次梯度更新。

    参数：
        train_windows: 训练窗口列表（未使用，仅用于校验）。
        val_windows: 验证窗口列表（未使用，仅用于校验）。
        epoch_index: 当前 epoch 索引（从 1 开始）。
        model: 液体模型实例，必须提供。
        optimizer: 优化器实例，必须提供。

    返回值：
    ?epoch 的平均训练损失。

    失败条件。
    TypeError: ?epoch_index 不是整数时抛出。
        ValueError: 褰epoch_index < 1。乵odel 鎴optimizer 个None。
            或训验证窗口为空时抛出。
    """

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

            loss_value = _loss_for_length_group(model, batch_samples)

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



def _collect_batch_loss_diagnostics(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    modalities: list[str],
    sample_weights: list[float],
    seq_lens: list[int],
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> dict[str, Any]:
    """收集一个批次的损失诊断信息。

    包括批次大小、损失分母/分子、模态列表、序列长度。
    样本权重、每个样本的濢、活头、头掩码、预测、目标。
    语义损失和平方误差的逐头明细。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。
        modalities: 当前批次每个样本的模态名列表。
        sample_weights: 样本权重列表。
        seq_lens: 每个样本的序列长度列表。

    返回值：
        包含所有诊断信息的字典。
    """
    mask = _build_head_mask_tensor(modalities, reference_tensor=prediction_batch)  # 构、输出头掩码。
    sample_weight_tensor = _build_sample_weight_tensor(sample_weights, reference_tensor=prediction_batch).unsqueeze(1)  # 样本权重张量，增加列维。
    semantic_loss = _semantic_loss_matrix(prediction_batch, target_batch, bias_huber_delta=bias_huber_delta, scaling_log_eps=scaling_log_eps)  # 语义损失矩阵。
    squared_error = _normalized_squared_error(prediction_batch, target_batch)  # 平方误差矩阵。
    weighted_mask = mask * sample_weight_tensor  # 加权掩码 = 头掩码× 样本权重。
    per_row_active_keys = [  # 每个样本的激活头列表。
        [
            key
            for key_index, key in enumerate(_OUTPUT_KEYS)
            if float(mask[row_index, key_index].item()) > 0.0  # 掩码 > 0 表示该头活。
        ]
        for row_index in range(int(mask.shape[0]))  # 遍历每个样本。
    ]
    probe_row_count = min(int(prediction_batch.shape[0]), _BATCH_DIAGNOSTIC_PROBE_LIMIT)  # 每个 batch 只保留少量探针行。
    component_loss_sums = {  # 每个输出头的逐样本语义损失和。
        key: float((semantic_loss[:, key_index] * mask[:, key_index]).sum().detach().item())
        for key_index, key in enumerate(_OUTPUT_KEYS)
    }
    component_loss_squared_sums = {  # 每个输出头的逐样本语义损失平方和。
        key: float(((semantic_loss[:, key_index] * mask[:, key_index]) ** 2).sum().detach().item())
        for key_index, key in enumerate(_OUTPUT_KEYS)
    }
    component_loss_counts = {  # 每个输出头的濢、活样本数。
        key: int(mask[:, key_index].sum().detach().item())
        for key_index, key in enumerate(_OUTPUT_KEYS)
    }
    modality_counts: dict[str, int] = {  # D7：与 lstm/trainer.py 同名函数口径对齐：键名走 str() 转换、按 sorted(set(...)) 排序，禁止本地插入序导致下游消费者字典序漂移。
        str(modality_key): int(sum(1 for modality in modalities if modality == modality_key))
        for modality_key in sorted(set(modalities))
    }
    return {
        "batch_size": int(prediction_batch.shape[0]),  # 批次大小。
        "loss_numerator": float((semantic_loss * weighted_mask).sum().detach().item()),  # 加权损失分子。
        "loss_denominator": float(weighted_mask.sum().detach().item()),  # 有效权重分母。
        "mean_loss": float(((semantic_loss * weighted_mask).sum() / weighted_mask.sum()).detach().item()) if weighted_mask.sum().item() > 0 else 0.0,  # 平均损失，防止除零。
        "modalities": list(modalities[:probe_row_count]),  # 探针行模态列表。
        "seq_lens": [int(seq_len) for seq_len in seq_lens[:probe_row_count]],  # 探针行序列长度列表。
        "sample_weights": [float(weight) for weight in sample_weights[:probe_row_count]],  # 探针行样本权重列表。
        "active_keys": per_row_active_keys[:probe_row_count],  # 探针行激活头列表。
        "head_mask": mask[:probe_row_count].detach().cpu().tolist(),  # 探针行头掩码矩阵。
        "prediction": prediction_batch[:probe_row_count].detach().cpu().tolist(),  # 探针行，预览前几行预测值。
        "target": target_batch[:probe_row_count].detach().cpu().tolist(),  # 探针行，预览前几行目标值。
        "semantic_loss_by_head": (semantic_loss[:probe_row_count] * mask[:probe_row_count]).detach().cpu().tolist(),  # 探针行、头语义损失。
        "squared_error_by_head": (squared_error[:probe_row_count] * mask[:probe_row_count]).detach().cpu().tolist(),  # 探针行、头平方误差。
        "probe_row_count": probe_row_count,
        "truncated_row_count": max(0, int(prediction_batch.shape[0]) - probe_row_count),
        "modality_counts": modality_counts,
        "seq_len_stats": {
            "min": min(int(seq_len) for seq_len in seq_lens) if seq_lens else None,  # D7：与 lstm/trainer.py 同名函数口径对齐：空列表默认 None 而非 0，避免下游消费、误判存在零长度序列。
            "max": max(int(seq_len) for seq_len in seq_lens) if seq_lens else None,  # D7：同上，空列表默认None 而非 0。
        },
        "sample_weight_stats": {
            "min": min(float(weight) for weight in sample_weights) if sample_weights else None,  # D7：与 lstm/trainer.py 同名函数口径对齐：空列表默认 None 而非 0.0，避免下游消费、误判存在零权重样本。
            "max": max(float(weight) for weight in sample_weights) if sample_weights else None,  # D7：同上，空列表默认None 而非 0.0。
            "mean": (sum(float(weight) for weight in sample_weights) / float(len(sample_weights)))  # D7/D3：与 lstm/trainer.py 同名函数口径对齐：补mean 字段，下游消费、（analysis/plotting/verify）依赖该字段，缺失会导致 KeyError 或静默漂移。
            if sample_weights
            else None,
        },
        "component_loss_sums": component_loss_sums,
        "component_loss_squared_sums": component_loss_squared_sums,
        "component_loss_counts": component_loss_counts,
    }


def _summarize_component_losses_compact(
    batches: list[dict[str, Any]],
    *,
    include_auxiliary: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float | None]]:
    """基于聚合统计量汇总、头损失，避免保留全量、样地"""
    summary: dict[str, dict[str, float | None]] = {}
    for key in _OUTPUT_KEYS:
        total_sum = 0.0
        total_squared_sum = 0.0
        total_count = 0
        for batch in batches:
            component_sums = batch.get("component_loss_sums", {})
            component_squared_sums = batch.get("component_loss_squared_sums", {})
            component_counts = batch.get("component_loss_counts", {})
            if not (
                isinstance(component_sums, Mapping)
                and isinstance(component_squared_sums, Mapping)
                and isinstance(component_counts, Mapping)
            ):
                continue
            key_count = int(component_counts.get(key, 0) or 0)
            if key_count <= 0:
                continue
            raw_sum = float(component_sums.get(key, 0.0) or 0.0)
            raw_squared_sum = float(component_squared_sums.get(key, 0.0) or 0.0)
            if not (math.isfinite(raw_sum) and math.isfinite(raw_squared_sum)):
                continue
            total_sum += raw_sum
            total_squared_sum += raw_squared_sum
            total_count += key_count
        if total_count > 0:
            mean_value = total_sum / float(total_count)
            variance = max(0.0, (total_squared_sum / float(total_count)) - (mean_value * mean_value))
            summary[key] = {"mean": mean_value, "variance": variance}
        else:
            summary[key] = {"mean": None, "variance": None}
    if include_auxiliary is not None:
        for key, value in include_auxiliary.items():
            if key in _OUTPUT_KEYS:
                raise ValueError(f"auxiliary key collides with output head: {key!r}")
            summary[key] = {"mean": float(value), "variance": 0.0}
    return summary


def _summarize_component_losses(
    batches: list[dict[str, Any]],
    *,
    include_auxiliary: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float | None]]:
    """汇、多个批次的逐头损失统计。

    对每个输出头，收集所有批次中的、样本损失、，
    计算均、和方差。可选地追加辅助损失项的统计。

    参数：
        batches: 批次诊断信息列表，每个元素是 _collect_batch_loss_diagnostics 的返回。
        include_auxiliary: 可、的辅助损失项映射，键为损失名，值为标量。

    返回值：
        字典，键为输出头名或辅助损失名，值为 {"mean": float, "variance": float}。
    """
    summary: dict[str, dict[str, float | None]] = {}  # 汇结果字典。
    for key in _OUTPUT_KEYS:  # 遍历每个输出头。
        total_sum = 0.0  # 该头所有批次的损失总和。
        total_squared_sum = 0.0  # 该头所有批次的损失平方总和。
        total_count = 0  # 该头所有批次的样本计数。
        for batch in batches:  # 遍历每个批次。
            component_sums = batch.get("component_loss_sums", {})  # 取该批次的、头损失总和。
            component_squared_sums = batch.get("component_loss_squared_sums", {})  # 取该批次的、头损失平方总和。
            component_counts = batch.get("component_loss_counts", {})  # 取该批次的、头样本计数。
            if not (  # 三字段必须同为映射，任一非映射即跳过整个批次，避免部分字段损坏导致计数与求和口径不一致。
                isinstance(component_sums, Mapping)
                and isinstance(component_squared_sums, Mapping)
                and isinstance(component_counts, Mapping)
            ):
                continue
            key_count = int(component_counts.get(key, 0) or 0)  # 该头在该批次的样本计数。
            if key_count <= 0: # 计数非正时跳过，避免 count=0 ?sum。 的损坏数据稀释均值。
                continue
            raw_sum = float(component_sums.get(key, 0.0) or 0.0)  # 该头在该批次的损失、和。
            raw_squared_sum = float(component_squared_sums.get(key, 0.0) or 0.0)  # 该头在该批次的损失平方、和。
            if not (math.isfinite(raw_sum) and math.isfinite(raw_squared_sum)):  # NaN/Inf 守卫，避免静默穿透污染统计。
                continue
            total_sum += raw_sum
            total_squared_sum += raw_squared_sum
            total_count += key_count
        if total_count > 0:  # 有、时计算统计量。
            mean_value = total_sum / float(total_count)  # 鍧囧€笺€。
            variance = max(0.0, (total_squared_sum / float(total_count)) - (mean_value ** 2))  # 方差。
            summary[key] = {"mean": mean_value, "variance": variance}  # 记录统计量。
        else:  # 没有值时标记None。
            summary[key] = {"mean": None, "variance": None}  # 无数据。
    if include_auxiliary is not None:  # 有辅助损失项时追加。
        for key, value in include_auxiliary.items():  # 遍历辅助损失项。
            if key in _OUTPUT_KEYS:  # 辅助键不得与输出头同名，避免静默覆盖输出头统计。
                raise ValueError(f"auxiliary key collides with output head: {key!r}")
            summary[key] = {"mean": float(value), "variance": 0.0}  # 辅助项方差为 0。
    return summary  # 返回汇结果。


def _collect_epoch_loss_diagnostics(
    *,
    split: str,
    epoch_index: int,
    batches: list[dict[str, Any]],
    selection_score: float | None,
    supervised_loss: float | None = None,
    auxiliary_summary: Mapping[str, float] | None = None,
    phase_switch_shock: Mapping[str, float | None] | None = None,
) -> dict[str, Any]:
    """汇、一epoch 的损失诊断信息。

    将所有批次的损失分子和分母累加，计算 epoch 级平均损失，
    并附带头损失汇和辅助损失项。

    参数：
    split: 数据集划分名train" ?"val"）。
        epoch_index: 当前 epoch 索引。
        batches: 批次诊断信息列表。
        selection_score: 选择分数，None 表示不记录。
        supervised_loss: 监督损失，None 表示不单独记录。        auxiliary_summary: 辅助损失项汇总，None 表示不记录。        phase_switch_shock: 相位切换冲击摘要，None 表示不记录。
    返回值：
        包含 split。乪poch_index。乵ean_loss。乴oss_numerator。乴oss_denominator。。
        batch_count。乥atches。乻election_score。乤uxiliary_losses 和。
        component_loss_summary 的字典。
    """
    loss_numerator = math.fsum(float(batch["loss_numerator"]) for batch in batches)  # 累加所有批次的损失分子，用 fsum 提高精度。
    loss_denominator = math.fsum(float(batch["loss_denominator"]) for batch in batches)  # 累加所有批次的损失分母，用 fsum 提高精度。
    stored_batches = list(batches[:_EPOCH_ARTIFACT_BATCH_LIMIT])
    epoch_payload = {  # 构建 epoch 级载荷。
        "split": split,  # 数据集划分。
        "epoch_index": int(epoch_index),  # epoch 绱㈠紩。。
        "mean_loss": loss_numerator / loss_denominator if loss_denominator > 0.0 else 0.0,  # epoch 平均损失。
        "loss_numerator": loss_numerator,  # 总损失分子。
        "loss_denominator": loss_denominator,  # 总损失分母。
        "batch_count": len(batches),  # 批次数量。
        "stored_batch_count": len(stored_batches),  # artifact 中实际保留的批次数量。
        "truncated_batch_count": max(0, len(batches) - len(stored_batches)),  # 被裁剪的批次数量。
        "batches": stored_batches,  # 批次诊断列表（只保留前若干批次明细）。
    }
    if selection_score is not None:
        epoch_payload["selection_score"] = coerce_finite_scalar(selection_score, name="selection_score")
    if supervised_loss is not None:
        epoch_payload["supervised_loss"] = coerce_finite_scalar(supervised_loss, name="supervised_loss")
    if auxiliary_summary is not None:
        epoch_payload["auxiliary_losses"] = {
            str(key): coerce_finite_scalar(value, name=f"auxiliary_losses.{key}")
            for key, value in auxiliary_summary.items()
        }
    if phase_switch_shock is not None:
        epoch_payload["phase_switch_shock"] = {
            str(key): (
                None
                if value is None
                else coerce_finite_scalar(value, name=f"phase_switch_shock.{key}")
            )
            for key, value in phase_switch_shock.items()
        }
    epoch_payload["component_loss_summary"] = _summarize_component_losses_compact(
        batches,  # 逐头损失汇总基于聚合统计量，避免保存全量逐样本值。
        include_auxiliary=auxiliary_summary,
    )
    return epoch_payload  # 返回 epoch 级诊断载荷。


def _collect_epoch_prediction_target_snapshot(
    *,
    model: Any,
    split: str,
    epoch_index: int,
    cached_windows: list[tuple[dict[str, Any], torch.Tensor, int, float]],
    batch_diagnostics: list[dict[str, Any]],
    prediction_target_summary: Mapping[str, Any],
    mean_loss: float,
    selection_score: float | None,
    supervised_loss: float | None = None,
    component_losses: Mapping[str, float] | None = None,
    auxiliary_losses: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """收集丢epoch 的预目标快照，用于训练日志。

    包括每个输出头的濢、活样本数、预测均值、目标均值，
    第一个批次的逐样本明细，以及固定探针样本的预测结果。

    参数：
        model: 液体模型实例。
        split: 数据集划分名train" ?"val"）。
        epoch_index: 当前 epoch 索引。
        cached_windows: 材料化窗口列表。
        batch_diagnostics: 批次诊断信息列表。
        mean_loss: epoch 平均损失。
        selection_score: 选择分数，None 表示不记录。        supervised_loss: 监督损失，None 表示不记录。        component_losses: 逐头损失映射，None 表示不记录。        auxiliary_losses: 辅助损失映射，None 表示不记录。
    返回值：
        包含 split。乪poch_index。乵ean_loss。乻election_score。乻ample_count。。
        各头统计量、探针批次明细和固定探针行的字典。
    """
    sample_count = int(prediction_target_summary.get("sample_count", 0) or 0)
    active_sample_count_by_head = {
        key: int((prediction_target_summary.get("active_sample_count_by_head", {}) or {}).get(key, 0) or 0)
        for key in _OUTPUT_KEYS
    }
    prediction_mean_active_samples = {
        key: (prediction_target_summary.get("prediction_mean_active_samples", {}) or {}).get(key)
        for key in _OUTPUT_KEYS
    }
    target_mean_active_samples = {
        key: (prediction_target_summary.get("target_mean_active_samples", {}) or {}).get(key)
        for key in _OUTPUT_KEYS
    }
    prediction_mean_all_samples = {
        key: (prediction_target_summary.get("prediction_mean_all_samples", {}) or {}).get(key)
        for key in _OUTPUT_KEYS
    }
    target_mean_all_samples = {
        key: (prediction_target_summary.get("target_mean_all_samples", {}) or {}).get(key)
        for key in _OUTPUT_KEYS
    }
    probe_batch = batch_diagnostics[0] if batch_diagnostics else {
        "modalities": [],
        "active_keys": [],
        "prediction": [],
        "target": [],
    }
    probe_batch_rows = []
    for row_index, (modality, active_keys, prediction_row, target_row) in enumerate(
        zip(
            probe_batch["modalities"],
            probe_batch["active_keys"],
            probe_batch["prediction"],
            probe_batch["target"],
            strict=True,
        )
    ):
        probe_batch_rows.append(
            {
                "row_index": row_index,
                "modality": modality,
                "active_keys": list(active_keys),
                # D5/D7：与 lstm/trainer.py 同名函数口径对齐，统丢coerce_finite_scalar 中心入口。
                # 拒绝 NaN/Inf 静默穿、至 snapshot（裸 float() 会把 nan/inf 静默写入消费者日志）。
                "prediction_by_head": {
                    key: coerce_finite_scalar(prediction_row[key_index], name=f"probe_batch.prediction.{key}")
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
                "target_by_head": {
                    key: coerce_finite_scalar(target_row[key_index], name=f"probe_batch.target.{key}")
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
                "signed_error_by_head": {
                    key: coerce_finite_scalar(
                        prediction_row[key_index] - target_row[key_index],
                        name=f"probe_batch.signed_error.{key}",
                    )
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
            }
        )
    fixed_probe_count = min(len(cached_windows), _EPOCH_FIXED_PROBE_LIMIT)
    fixed_probe_rows = _collect_fixed_probe_rows(
        model=model,
        cached_windows=cached_windows[:fixed_probe_count],
    )
    return {
        "split": split,
        "epoch_index": int(epoch_index),
        # D5/D7：epoch 级标量走 coerce_finite_scalar 中心入口，拒NaN/Inf 静默穿、至 snapshot。
        # ?_summarize_run ?train_loss/val_loss/selection_score 口径致（D7）。
        "mean_loss": coerce_finite_scalar(mean_loss, name="mean_loss"),
        "selection_score": None if selection_score is None else coerce_finite_scalar(selection_score, name="selection_score"),
        "supervised_loss": None if supervised_loss is None else coerce_finite_scalar(supervised_loss, name="supervised_loss"),
        "component_losses": {
            str(key): coerce_finite_scalar(value, name=f"component_losses.{key}")
            for key, value in (component_losses or {}).items()
        },
        "auxiliary_losses": {
            str(key): coerce_finite_scalar(value, name=f"auxiliary_losses.{key}")
            for key, value in (auxiliary_losses or {}).items()
        },
        "sample_count": sample_count,
        "active_sample_count_by_head": active_sample_count_by_head,
        "prediction_mean": dict(prediction_mean_active_samples),
        "target_mean": dict(target_mean_active_samples),
        "prediction_mean_active_samples": dict(prediction_mean_active_samples),
        "target_mean_active_samples": dict(target_mean_active_samples),
        "prediction_mean_all_samples": dict(prediction_mean_all_samples),
        "target_mean_all_samples": dict(target_mean_all_samples),
        "mean_scope": "fixed_split_active_sample_mean",
        "mean_scope_note": "target means can remain constant across epochs because the split is fixed and means are computed over active samples.",
        "probe_batch_modalities": list(probe_batch["modalities"]),
        "probe_batch_active_keys": [list(keys) for keys in probe_batch["active_keys"]],
        "probe_batch_prediction": probe_batch["prediction"],
        "probe_batch_target": probe_batch["target"],
        "probe_batch_rows": probe_batch_rows,
        "fixed_probe_source": "cached_split_prefix",
        "fixed_probe_sample_count": fixed_probe_count,
        "fixed_probe_rows": fixed_probe_rows,
    }


def _collect_fixed_probe_rows(
    *,
    model: Any,
    cached_windows: list[tuple[dict[str, Any], torch.Tensor, int, float]],
) -> list[dict[str, Any]]:
    """对固定探针样本做前向预测，收集、样本的预测-目标明细。

    固定探针用于epoch 追踪同一组样本的预测变化趋势。

    参数：
        model: 液体模型实例。
        cached_windows: 固定探针的材料化窗口列表。

    返回值：
        逐样本明细列表，每个元素包含 sample_index_in_split、modality。
        seq_len。乻ample_weight。乤ctive_keys。乸rediction_by_head。。
        target_by_head 和signed_error_by_head。。
    """
    if not cached_windows:  # 没有探针样本时返回空列表。
        return []  # 空列表。
    # D7/D4：与 lstm/trainer.py 同名函数口径对齐—探针前向在 no_grad 下执行，
    # 避免构、梯度图浪费显存（AGENTS.md：high-level consumer 不占 GPU）。
    with torch.no_grad():
        prediction_batch = _predict_batch(model, [sample[0] for sample in cached_windows])  # 对探针样本做前向预测。
    fixed_probe_rows: list[dict[str, Any]] = []  # 存、样本明细。
    for sample_index, sample in enumerate(cached_windows):  # 遍历每个探针样本。
        # D5/D7：与 lstm/trainer.py 同名函数口径对齐，统丢coerce_finite_scalar 中心入口，拒NaN/Inf 静默穿、（target 来自缓存窗口，未在上游做有限性校验）。
        prediction_row = [
            coerce_finite_scalar(value, name=f"fixed_probe_prediction[{sample_index}][{key_index}]")
            for key_index, value in enumerate(prediction_batch[sample_index].detach().cpu().tolist())
        ]
        target_row = [
            coerce_finite_scalar(value, name=f"fixed_probe_target[{sample_index}][{key_index}]")
            for key_index, value in enumerate(sample[1].detach().cpu().tolist())
        ]
        modality = _resolve_current_modality(sample[0])
        active_keys = list(_ACTIVE_HEADS_BY_MODALITY[modality])
        fixed_probe_rows.append(
            {
                "sample_index_in_split": sample_index,
                "modality": modality,
                "seq_len": int(sample[2]),
                "sample_weight": coerce_finite_scalar(sample[3], name=f"fixed_probe_sample_weight[{sample_index}]"),
                "active_keys": active_keys,
                "prediction_by_head": {
                    key: prediction_row[key_index]
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
                "target_by_head": {
                    key: target_row[key_index]
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
                "signed_error_by_head": {
                    key: prediction_row[key_index] - target_row[key_index]
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
            }
        )
    return fixed_probe_rows


def _evaluate_model(
    model: Any,
    val_windows: list[Any],
    *,
    epoch_index: int | None = None,
    split: str = "val",
    collect_diagnostics: bool = False,
) -> float | tuple[float, list[dict[str, Any]], dict[str, Any]]:
    """评估模型在验训练集上的损失。

    ?no_grad 模式下、批次做前向预测，计算监督损失和辅助损失。
    得到选择损失（selection_loss = supervised_loss + auxiliary_loss。
    和、择分数。可选择收集详细的批次和 epoch 级诊断信息。

    参数：
        model: 液体模型实例。
        val_windows: 验证/训练窗口列表。
        epoch_index: 当前 epoch 索引，仅用于诊断日志。
        split: 数据集划分名val" ?"train"）。
        collect_diagnostics: 是否收集详细诊断信息。

    返回值：
        - collect_diagnostics=False: 返回 selection_loss 浮点数。
        - collect_diagnostics=True: 返回三元。
          (selection_loss, batch_diagnostics, epoch_snapshot)。。
    """
    model.eval()
    cache_attr = "_cached_train_windows" if split == "train" else "_cached_val_windows"
    cached_windows = getattr(model, cache_attr, None)
    if cached_windows is None:
        cache_device = _resolve_sample_cache_device(model)  # §13.8: 与 model 同设备，释放 CPU 内存
        cached_windows = _move_materialized_samples(
            _materialize_samples(
                val_windows,
                tail_selection_observation_coeff=coerce_finite_scalar(
                    getattr(model, "_tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
                    name="tail_selection_observation_coeff",
                    min_value=0.0,
                ),
            ),
            cache_device,
        )
        setattr(model, cache_attr, cached_windows)
    batch_attr = "_train_batch_size" if split == "train" else "_eval_batch_size"
    eval_batch_size = int(getattr(model, batch_attr, 0) or len(cached_windows))
    component_numerator_sums = {key: 0.0 for key in _OUTPUT_KEYS}
    component_denominator_sums = {key: 0.0 for key in _OUTPUT_KEYS}
    loss_numerator_sum = 0.0
    loss_denominator_sum = 0.0
    batch_diagnostics: list[dict[str, Any]] = []
    prediction_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    all_modalities: list[str] = []
    all_sample_weights: list[float] = []
    sample_count = 0
    active_sample_count_by_head = {key: 0 for key in _OUTPUT_KEYS}
    prediction_sum_all_samples = {key: 0.0 for key in _OUTPUT_KEYS}
    target_sum_all_samples = {key: 0.0 for key in _OUTPUT_KEYS}
    prediction_sum_active_samples = {key: 0.0 for key in _OUTPUT_KEYS}
    target_sum_active_samples = {key: 0.0 for key in _OUTPUT_KEYS}
    position_mse_scalar_sum = 0.0  # §24.1 执行点：位置误差 MSE 累加器（用于选模决策）。
    position_mse_scalar_count = 0    # §24.1 执行点：位置误差 MSE 样本计数。
    with torch.no_grad():
        # D5/D7：与 lstm/trainer.py._evaluate_model 口径对齐—LSTM 侧、过参数传入已在
        # LSTMTrainer.__init__ ?coerce_finite_scalar 校验bias_huber_delta/scaling_log_eps。
        # Liquid 侧从 model 实例属、读取，故在此统丢coerce_finite_scalar 中心入口。
        # 拒绝 NaN/Inf/bool/负、静默穿透到 eval 损失（与 _compute_batch_loss L2030-2031 同口径）。
        _bias_huber_delta_val = coerce_finite_scalar(
            getattr(model, "_bias_huber_delta", _BIAS_HUBER_DELTA),
            name="model._bias_huber_delta", min_value=0.0)
        _scaling_log_eps_val = coerce_finite_scalar(
            getattr(model, "_scaling_log_eps", _SCALING_LOG_EPS),
            name="model._scaling_log_eps", min_value=0.0)
        for start_index in range(0, len(cached_windows), eval_batch_size):
            end_index = min(len(cached_windows), start_index + eval_batch_size)
            batch_samples = cached_windows[start_index:end_index]
            metadata = [sample[0] for sample in batch_samples]
            prediction_batch = _predict_batch(model, metadata)
            target_batch = torch.stack([sample[1] for sample in batch_samples], dim=0)
            target_batch = target_batch.to(device=prediction_batch.device, dtype=prediction_batch.dtype)
            modalities = [_resolve_current_modality(sample[0]) for sample in batch_samples]
            sample_weights = [sample[3] for sample in batch_samples]
            numerators, denominators, active_keys, position_mse_scalar = _masked_component_loss_stats(
                prediction_batch,
                target_batch,
                modalities=modalities,
                sample_weights=sample_weights,
                bias_huber_delta=_bias_huber_delta_val,
                scaling_log_eps=_scaling_log_eps_val,
            )
            for key in active_keys:
                component_numerator_sums[key] += float(numerators[key].detach().item())
                component_denominator_sums[key] += float(denominators[key].detach().item())
            loss_numerator, loss_denominator = _masked_supervised_loss_stats(
                prediction_batch,
                target_batch,
                modalities=modalities,
                sample_weights=sample_weights,
                bias_huber_delta=_bias_huber_delta_val,
                scaling_log_eps=_scaling_log_eps_val,
            )
            loss_numerator_sum += float(loss_numerator.detach().item())
            loss_denominator_sum += float(loss_denominator.detach().item())
            prediction_batches.append(prediction_batch)
            target_batches.append(target_batch)
            all_modalities.extend(modalities)
            all_sample_weights.extend(float(weight) for weight in sample_weights)
            sample_count += int(prediction_batch.shape[0])
            # §24.1 执行点：累加位置误差 MSE 执行点值（用于最终选模决策与审计追踪）。
            position_mse_scalar_sum += float(position_mse_scalar.detach().item())
            position_mse_scalar_count += 1
            for row_index, modality in enumerate(modalities):
                active_keys = set(_ACTIVE_HEADS_BY_MODALITY[modality])
                for key_index, key in enumerate(_OUTPUT_KEYS):
                    prediction_value = float(prediction_batch[row_index, key_index].detach().item())
                    target_value = float(target_batch[row_index, key_index].detach().item())
                    prediction_sum_all_samples[key] += prediction_value
                    target_sum_all_samples[key] += target_value
                    if key in active_keys:
                        active_sample_count_by_head[key] += 1
                        prediction_sum_active_samples[key] += prediction_value
                        target_sum_active_samples[key] += target_value
            if collect_diagnostics:
                batch_payload = _collect_batch_loss_diagnostics(
                    prediction_batch,
                    target_batch,
                    modalities=modalities,
                    sample_weights=sample_weights,
                    seq_lens=[sample[2] for sample in batch_samples],
                    bias_huber_delta=_bias_huber_delta_val,
                    scaling_log_eps=_scaling_log_eps_val,
                )
                batch_diagnostics.append(batch_payload)
    active_keys = [key for key in _OUTPUT_KEYS if component_denominator_sums[key] > 0.0]
    component_losses = {
        key: torch.tensor(component_numerator_sums[key] / component_denominator_sums[key], dtype=torch.float32)
        for key in active_keys
    }
    supervised_loss = loss_numerator_sum / loss_denominator_sum if loss_denominator_sum > 0.0 else 0.0
    auxiliary_losses = {"calibration": 0.0, "mono": 0.0, "reg_l2": 0.0, "gate_l1": 0.0, "smooth": 0.0, "total": 0.0}
    if prediction_batches:
        full_prediction_batch = torch.cat(prediction_batches, dim=0)
        full_target_batch = torch.cat(target_batches, dim=0)
        auxiliary_terms = _compute_liquid_auxiliary_loss_terms(
            model,
            full_prediction_batch,
            full_target_batch,
            modalities=all_modalities,
            sample_weights=all_sample_weights,
        )
        auxiliary_losses = {
            key: float(auxiliary_terms[key].detach().item())
            for key in ("calibration", "mono", "reg_l2", "gate_l1", "smooth", "total")
        }
    auxiliary_loss = auxiliary_losses["total"]
    # §24.1 执行点：计算平均位置误差 MSE 并传入选模函数（保证终序仅 raw 位置 RMSE 精神执行）。
    avg_position_mse_scalar = torch.tensor(
        position_mse_scalar_sum / position_mse_scalar_count if position_mse_scalar_count > 0 else 0.0,
        dtype=torch.float32,
    )
    export_score = _build_export_score(
        component_losses=component_losses,
        active_keys=active_keys,
        supervised_loss=float(supervised_loss),
        position_mse_scalar=avg_position_mse_scalar,  # §24.1 执行点：位置误差强制纳入选模。
    )
    score_bundle = _build_epoch_score_bundle(
        supervised_loss=float(supervised_loss),
        auxiliary_loss=float(auxiliary_loss),
        export_score=float(export_score),
    )
    optimization_loss = float(score_bundle["optimization_loss"])
    phase_control_score = float(score_bundle["phase_control_score"])
    export_score = float(score_bundle["export_score"])
    selection_score = float(score_bundle["selection_score"])
    model._optimization_loss = optimization_loss
    model._phase_control_score = phase_control_score
    model._export_score = export_score
    model._selection_score = selection_score
    model._selection_loss = optimization_loss
    component_losses_payload = {
        str(key): float(value.detach().item())
        for key, value in component_losses.items()
    }
    model._component_losses = dict(component_losses_payload)
    model._supervised_loss = float(supervised_loss)
    model._auxiliary_losses = dict(auxiliary_losses)
    if not collect_diagnostics:
        return optimization_loss
    for batch_payload in batch_diagnostics:
        batch_payload["auxiliary_losses"] = dict(auxiliary_losses)
    prediction_target_summary = {
        "sample_count": sample_count,
        "active_sample_count_by_head": dict(active_sample_count_by_head),
        "prediction_mean_active_samples": {
            key: (
                prediction_sum_active_samples[key] / float(active_sample_count_by_head[key])
                if int(active_sample_count_by_head[key]) > 0
                else None
            )
            for key in _OUTPUT_KEYS
        },
        "target_mean_active_samples": {
            key: (
                target_sum_active_samples[key] / float(active_sample_count_by_head[key])
                if int(active_sample_count_by_head[key]) > 0
                else None
            )
            for key in _OUTPUT_KEYS
        },
        "prediction_mean_all_samples": {
            key: (prediction_sum_all_samples[key] / float(sample_count)) if sample_count > 0 else None
            for key in _OUTPUT_KEYS
        },
        "target_mean_all_samples": {
            key: (target_sum_all_samples[key] / float(sample_count)) if sample_count > 0 else None
            for key in _OUTPUT_KEYS
        },
    }
    epoch_payload = _collect_epoch_prediction_target_snapshot(
        model=model,
        split=split,
        epoch_index=int(epoch_index or 0),
        cached_windows=cached_windows,
        batch_diagnostics=batch_diagnostics,
        prediction_target_summary=prediction_target_summary,
        mean_loss=optimization_loss,
        selection_score=selection_score,
        supervised_loss=supervised_loss,
        component_losses=component_losses_payload,
        auxiliary_losses=auxiliary_losses,
    )
    epoch_payload["optimization_loss"] = optimization_loss
    epoch_payload["phase_control_score"] = phase_control_score
    epoch_payload["export_score"] = export_score
    epoch_payload["component_losses"] = dict(component_losses_payload)
    epoch_payload["auxiliary_losses"] = dict(auxiliary_losses)
    return optimization_loss, batch_diagnostics, epoch_payload


def build_trainer_state(train_cfg: dict) -> dict:
    """从训练配置字典构建训练器状、字典。

    对配置中的每个字段做类型和、域校验，解析阶段调度配置，
    确定输出路径，返回完整的训练器状态。

    参数：
        train_cfg: 训练配置字典，必须包含"train" 子字典，
        可包含含"name"?output_root"?feature_order" 等键。

    返回值：
        训练器状态字典，包含 model_name、optimizer、epochs。
        epoch_candidate_stride。乸hase_schedule。乻eed。乨eterministic。。
        device。乷utput_root。乫eature_order。亀indow。乶etwork 和train。。

    失败条件。
        TypeError: 当配置类型不正确时抛出。
        ValueError: 当配置不合法时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    _train_section = train_cfg.get("train", {}) if isinstance(train_cfg, dict) else {}
    print_dict({
        "name": train_cfg.get("name") if isinstance(train_cfg, dict) else None,
        "optimizer": _train_section.get("optimizer"),
        "lr": _train_section.get("lr"),
        "epochs": _train_section.get("epochs"),
        "phase_schedule": _train_section.get("phase_schedule"),
        "device": train_cfg.get("device") if isinstance(train_cfg, dict) else None,
        "seed": train_cfg.get("seed") if isinstance(train_cfg, dict) else None,
        "deterministic": train_cfg.get("deterministic") if isinstance(train_cfg, dict) else None,
    }, "Liquid build_trainer_state 入口参数")
    if not isinstance(train_cfg, dict):  # 配置必须是字典。
        raise TypeError("train_cfg must be a dictionary.")  # 类型错误。
    train_section = dict(train_cfg.get("train") or {})  # 读取 train 子字典。
    optimizer_name = train_section.get("optimizer", "adam")  # 优化器名，默认Adam。
    lr = train_section.get("lr", 0.001)  # 学习率，默认 0.001。
    weight_decay = train_section.get("weight_decay", 0.0)  # 权重衰减，默认0。
    gradient_clip_max_norm = train_section.get("gradient_clip_max_norm", 1.0)  # 梯度裁剪阈、，默认 1.0。
    epochs = train_section.get("epochs", 1)  # 训练 epoch 数，默认 1。
    epoch_candidate_stride = train_section.get("epoch_candidate_stride", 1)  # epoch 候、保存步长，默认 1。
    seed = train_section.get("seed", train_cfg.get("seed", 0))  # 随机种子，优先从 train 子字典读取。
    deterministic = train_section.get("deterministic", train_cfg.get("deterministic", True))  # 确定性标志。
    if not is_string_like(optimizer_name) or not str(optimizer_name).strip():  # 优化器名必须是非空字符串。
        raise ValueError("train.optimizer must be a non-empty string.")  # 值错。
    if not is_numeric(lr):  # 学习率必须是数。
        raise TypeError("train.lr must be numeric.")  # 类型错误。
    if float(lr) <= 0.0:  # 学习率必须为正。
        raise ValueError("train.lr must be positive.")  # 值错。
    if not is_numeric(weight_decay):  # 权重衰减必须是数值。
        raise TypeError("train.weight_decay must be numeric.")  # 类型错误。
    if float(weight_decay) < 0.0:  # 权重衰减不能为负。
        raise ValueError("train.weight_decay must be non-negative.")  # 值错。
    if not is_numeric(gradient_clip_max_norm):  # 梯度裁剪阈、必须是数。
        raise TypeError("train.gradient_clip_max_norm must be numeric.")  # 类型错误。
    if float(gradient_clip_max_norm) <= 0.0:  # 梯度裁剪阈、必须为正。
        raise ValueError("train.gradient_clip_max_norm must be positive.")  # 值错。
    if not is_integer(epochs):  # epoch 数必须是整数。
        raise TypeError("train.epochs must be an integer.")  # 类型错误。
    if epochs < 1: # 至少训练 1 epoch.
        raise ValueError("train.epochs must be at least 1.")  # 值错。
    if not is_integer(epoch_candidate_stride):  # 步长必须是整数。
        raise TypeError("train.epoch_candidate_stride must be an integer.")  # 类型错误。
    if epoch_candidate_stride < 1:  # 步长至少为 1。
        raise ValueError("train.epoch_candidate_stride must be at least 1.")  # 值错。
    if not is_integer(seed):  # 种子必须是整数。
        raise TypeError("train seed must be an integer.")  # 类型错误。
    if not is_bool_like(deterministic):  # 确定性标志必须是布尔值。
        raise TypeError("train deterministic flag must be a boolean.")  # 类型错误。
    phase_schedule_cfg = train_section.get("phase_schedule")  # 读取阶段调度配置。
    if phase_schedule_cfg is not None and not isinstance(phase_schedule_cfg, Mapping):  # 配置存在时必须是映射。
        raise TypeError("train.phase_schedule must be a mapping when provided")  # 类型错误。
    warmup_epochs, gate_alignment_epochs, full_tuning_epochs = _resolve_phase_epochs(  # 解析阶段 epoch 数量。
        phase_schedule_cfg,
        total_epochs=int(epochs),
    )
    model_name = train_cfg.get("name", "liquid_ekf")  # 模型名，默认 liquid_ekf。
    if not is_string_like(model_name) or not str(model_name).strip():  # 模型名必须是非空字符串。
        raise ValueError("name must be a non-empty string.")  # 值错。
    output_root = train_cfg.get("output_root")  # 读取输出根目录。
    if output_root is None:  # 未配置时自动推导。
        project_root = train_cfg.get("project_root")  # 先读项目根目录。
        if project_root is None:  # 项目根也没配。
            project_root = find_project_root()
        output_root = Path(project_root) / "outputs" / "train_liquid"  # 默认输出路径。
    else:
        output_root = Path(output_root)  # 已配置时直接Path。
    calibration_bin_count = train_section.get("calibration_bin_count", _RISK_CALIBRATION_BIN_COUNT)  # 标准分箱数量。
    if not is_integer(calibration_bin_count):  # 分箱数必须是整数，拒绝浮点静默截断。
        raise TypeError("train.calibration_bin_count must be an integer.")  # 类型错误。
    if int(calibration_bin_count) < 1:  # 至少 1 个分箱，0 箱无校准意义。
        raise ValueError("train.calibration_bin_count must be at least 1.")  # 值错。
    tail_selection_observation_coeff = coerce_finite_scalar(
        train_section.get("tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
        name="train.tail_selection_observation_coeff",
        min_value=0.0,
    )
    train_section["tail_selection_observation_coeff"] = tail_selection_observation_coeff
    gate_alignment_readout_lr_scale = coerce_finite_scalar(
        train_section.get("gate_alignment_readout_lr_scale", _GATE_ALIGNMENT_READOUT_LR_SCALE),
        name="train.gate_alignment_readout_lr_scale",
        min_value=0.0,
        inclusive=False,
    )
    train_section["gate_alignment_readout_lr_scale"] = gate_alignment_readout_lr_scale
    soft_control_streak_length = int(
        coerce_finite_scalar(
            train_section.get("soft_control_streak_length", _SOFT_CONTROL_STREAK_LENGTH),
            name="train.soft_control_streak_length",
            min_value=1.0,
            inclusive=True,
        )
    )
    train_section["soft_control_streak_length"] = soft_control_streak_length
    soft_control_streak_lr_decay = coerce_finite_scalar(
        train_section.get("soft_control_streak_lr_decay", _SOFT_CONTROL_STREAK_LR_DECAY),
        name="train.soft_control_streak_lr_decay",
        min_value=0.0,
        inclusive=False,
    )
    train_section["soft_control_streak_lr_decay"] = soft_control_streak_lr_decay
    soft_control_shock_threshold = coerce_finite_scalar(
        train_section.get("soft_control_shock_threshold", _SOFT_CONTROL_SHOCK_THRESHOLD),
        name="train.soft_control_shock_threshold",
        min_value=0.0,
        inclusive=False,
    )
    train_section["soft_control_shock_threshold"] = soft_control_shock_threshold
    soft_control_lr_decay = coerce_finite_scalar(
        train_section.get("soft_control_lr_decay", _SOFT_CONTROL_LR_DECAY),
        name="train.soft_control_lr_decay",
        min_value=0.0,
        inclusive=False,
    )
    train_section["soft_control_lr_decay"] = soft_control_lr_decay
    soft_control_buffer_epochs = int(
        coerce_finite_scalar(
            train_section.get("soft_control_buffer_epochs", _SOFT_CONTROL_BUFFER_EPOCHS),
            name="train.soft_control_buffer_epochs",
            min_value=0.0,
        )
    )
    train_section["soft_control_buffer_epochs"] = soft_control_buffer_epochs
    soft_control_cooldown_epochs = int(
        coerce_finite_scalar(
            train_section.get("soft_control_cooldown_epochs", _SOFT_CONTROL_COOLDOWN_EPOCHS),
            name="train.soft_control_cooldown_epochs",
            min_value=0.0,
        )
    )
    train_section["soft_control_cooldown_epochs"] = soft_control_cooldown_epochs
    soft_control_lr_floor_scale = coerce_finite_scalar(
        train_section.get("soft_control_lr_floor_scale", _SOFT_CONTROL_LR_FLOOR_SCALE),
        name="train.soft_control_lr_floor_scale",
        min_value=0.0,
        inclusive=False,
    )
    train_section["soft_control_lr_floor_scale"] = soft_control_lr_floor_scale
    soft_control_rollback_patience = int(
        coerce_finite_scalar(
            train_section.get("soft_control_rollback_patience", _SOFT_CONTROL_ROLLBACK_PATIENCE),
            name="train.soft_control_rollback_patience",
            min_value=1.0,
            inclusive=True,
        )
    )
    train_section["soft_control_rollback_patience"] = soft_control_rollback_patience
    # 铁律 6 (2026-07-23 N2/N4 中和): Liquid 独有的 full_tuning 子阶段 LR 调度器
    # (_resolve_full_tuning_subphase) 和阶段感知辅助损失 ramp (_build_phase_control_state)
    # 是 LSTM trainer 没有的隐藏训练助手. 这里读取 train.disable_full_tuning_subphase_scheduler
    # 等 7 个覆盖键并显式写入 trainer_state/train_section, 让下游能通过
    # trainer_state.get(...) 取到 (N2 短路子阶段, N4 强制固定 4 个 scale 为 1.0),
    # 与 LSTM 平坦 LR / 平坦 aux loss 对齐, 修补 Liquid vs LSTM 训练公平性.
    if not is_bool_like(train_section.get("disable_full_tuning_subphase_scheduler", False)):
        raise TypeError("train.disable_full_tuning_subphase_scheduler must be a boolean.")
    disable_full_tuning_subphase_scheduler = bool(
        train_section.get("disable_full_tuning_subphase_scheduler", False)
    )
    train_section["disable_full_tuning_subphase_scheduler"] = disable_full_tuning_subphase_scheduler
    full_tuning_entry_bridge_epochs = int(
        coerce_finite_scalar(
            train_section.get("full_tuning_entry_bridge_epochs", _FULL_TUNING_ENTRY_BRIDGE_EPOCHS),
            name="train.full_tuning_entry_bridge_epochs",
            min_value=0.0,
        )
    )
    train_section["full_tuning_entry_bridge_epochs"] = full_tuning_entry_bridge_epochs
    full_tuning_late_consolidation_epochs = int(
        coerce_finite_scalar(
            train_section.get("full_tuning_late_consolidation_epochs", _FULL_TUNING_LATE_CONSOLIDATION_EPOCHS),
            name="train.full_tuning_late_consolidation_epochs",
            min_value=0.0,
        )
    )
    train_section["full_tuning_late_consolidation_epochs"] = full_tuning_late_consolidation_epochs
    # N4: 4 个 phase_*_scale 覆盖键使用 sentinel (None) 而非默认 1.0 — 这样下游
    # _build_phase_control_state 用 trainer_state.get("phase_aux_scale", None) 能区分
    # "用户显式提供 1.0 强制固定" 与 "用户未提供保留原 ramp 行为". 因此仅在用户显式提供
    # 时校验并写入 train_section/trainer_state 顶层, 未提供时不写入 (避免 1.0 默认熏陶).
    _SENTINEL = object()
    phase_aux_scale_raw = train_section.get("phase_aux_scale", _SENTINEL)
    phase_gate_scale_raw = train_section.get("phase_gate_scale", _SENTINEL)
    phase_calibration_scale_raw = train_section.get("phase_calibration_scale", _SENTINEL)
    phase_regularization_scale_raw = train_section.get("phase_regularization_scale", _SENTINEL)
    phase_aux_scale: float | None = None
    phase_gate_scale: float | None = None
    phase_calibration_scale: float | None = None
    phase_regularization_scale: float | None = None
    if phase_aux_scale_raw is not _SENTINEL:
        phase_aux_scale = float(coerce_finite_scalar(
            phase_aux_scale_raw, name="train.phase_aux_scale", min_value=0.0,
        ))
        train_section["phase_aux_scale"] = phase_aux_scale
    if phase_gate_scale_raw is not _SENTINEL:
        phase_gate_scale = float(coerce_finite_scalar(
            phase_gate_scale_raw, name="train.phase_gate_scale", min_value=0.0,
        ))
        train_section["phase_gate_scale"] = phase_gate_scale
    if phase_calibration_scale_raw is not _SENTINEL:
        phase_calibration_scale = float(coerce_finite_scalar(
            phase_calibration_scale_raw, name="train.phase_calibration_scale", min_value=0.0,
        ))
        train_section["phase_calibration_scale"] = phase_calibration_scale
    if phase_regularization_scale_raw is not _SENTINEL:
        phase_regularization_scale = float(coerce_finite_scalar(
            phase_regularization_scale_raw, name="train.phase_regularization_scale", min_value=0.0,
        ))
        train_section["phase_regularization_scale"] = phase_regularization_scale
    return {
        "model_name": model_name,
        "optimizer": {
            "name": optimizer_name.strip().lower(),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        },
        "epochs": int(epochs),
        "epoch_candidate_stride": int(epoch_candidate_stride),
        "phase_schedule": {
            "warmup_epochs": int(warmup_epochs),
            "gate_alignment_epochs": int(gate_alignment_epochs),
        },
        "phase_schedule_budget": {
            "warmup_epochs": int(warmup_epochs),
            "gate_alignment_epochs": int(gate_alignment_epochs),
            "full_tuning_epochs": int(full_tuning_epochs),
        },
        "seed": seed,
        "deterministic": deterministic,
        "device": str(train_cfg.get("device") or "cpu"),
        "output_root": output_root,
        "feature_order": list(train_cfg.get("feature_order") or []),
        "window": dict(train_cfg.get("window") or {}),
        "network": dict(train_cfg.get("network") or {}),
        "train": train_section,
        "loss_weights": dict(train_section.get("loss_weights") or {}),
        "lr_layer": dict(train_section.get("lr_layer") or {}),
        "calibration_bin_count": int(calibration_bin_count),
        "tail_selection_observation_coeff": tail_selection_observation_coeff,
        "gate_alignment_readout_lr_scale": gate_alignment_readout_lr_scale,
        "soft_control_shock_threshold": soft_control_shock_threshold,
        "soft_control_lr_decay": soft_control_lr_decay,
        "soft_control_streak_length": soft_control_streak_length,
        "soft_control_streak_lr_decay": soft_control_streak_lr_decay,
        "soft_control_buffer_epochs": soft_control_buffer_epochs,
        "soft_control_cooldown_epochs": soft_control_cooldown_epochs,
        "soft_control_rollback_patience": soft_control_rollback_patience,
        # 铁律 6 (2026-07-23 N2/N4 中和): 把 7 个覆盖键显式提升到 trainer_state 顶层,
        # 让 _resolve_full_tuning_subphase (N2) 和 _build_phase_control_state (N4)
        # 能通过 trainer_state.get(...) 读取并生效. train 子字典已包含同名键, 此处为
        # 下游可发现的顶层别名 (默认 False/原始 epoch 预算/1.0 完全中性).
        "disable_full_tuning_subphase_scheduler": disable_full_tuning_subphase_scheduler,
        "full_tuning_entry_bridge_epochs": full_tuning_entry_bridge_epochs,
        "full_tuning_late_consolidation_epochs": full_tuning_late_consolidation_epochs,
        "phase_aux_scale": phase_aux_scale,
        "phase_gate_scale": phase_gate_scale,
        "phase_calibration_scale": phase_calibration_scale,
        "phase_regularization_scale": phase_regularization_scale,
        "bias_huber_delta": coerce_finite_scalar(
            train_section.get("bias_huber_delta", _BIAS_HUBER_DELTA),
            name="train.bias_huber_delta",
            min_value=0.0,
        ),
        "scaling_log_eps": coerce_finite_scalar(
            train_section.get("scaling_log_eps", _SCALING_LOG_EPS),
            name="train.scaling_log_eps",
            min_value=0.0,
        ),
        "gradient_clip_max_norm": float(gradient_clip_max_norm),
        # 偷懒审视 Round 4 真修 (用户铁律 #14): liquid trainer 从未实现 LR scheduler (只有 soft_control_shock_threshold
        # audit #20 H2+M2 (主线程决策 2026-07): 原 ReduceLROnPlateau 因 threshold='rel' (默认) 致实际
        # threshold = 1e-4 * best_loss ≈ 1e-4 * 0.022 ≈ 2.2e-6, val_loss 每 epoch 下降只要 > 2.2e-6 即视为
        # "仍在改善", scheduler 永不触发, lr 0 衰减 (silent bug). 即便按 H2 改 threshold='abs' 仍存 plateau
        # 检测 silent 风险 (train 动力学不稳时仍可能漏发). 主线程决策 (M2 简单方案): 改用
        # torch.optim.lr_scheduler.CosineAnnealingLR — lr 平滑退火 lr(t)=eta_min+0.5*(lr0-eta_min)*(1+cos(t*pi/T_max)).
        # 默认 T_max=epochs (即 160), eta_min=1e-6 (PyTorch 标准推荐值). 与 lstm trainer 公平口径一致.
        "lr_scheduler": _build_lr_scheduler_state(train_section),
    }


def _build_lr_scheduler_state(train_section: dict) -> dict:
    """Round 4 + audit #20 H2+M2: 从 train.lr_scheduler 子段解析 CosineAnnealingLR 配置.

    参数 train_section: 训练配置字典的 train 子字典 (含 epochs 字段).
    返回值 标准化后的 lr_scheduler 状态字典 (enabled/T_max/eta_min).
    失败条件 lr_scheduler.enabled=True 时, T_max 必须 == train.epochs, eta_min 必须 > 0, 否则抛 ValueError.
    """
    lr_scheduler_cfg = dict(train_section.get("lr_scheduler") or {})
    enabled = bool(lr_scheduler_cfg.get("enabled", False))
    # 默认 T_max = train.epochs (CosineAnnealingLR 退火周期与训练总轮次对齐).
    epochs_in_section = int(train_section.get("epochs", 1))
    lr_scheduler_t_max = int(lr_scheduler_cfg.get("T_max", epochs_in_section))
    lr_scheduler_eta_min = float(lr_scheduler_cfg.get("eta_min", 1e-6))
    if enabled:
        if lr_scheduler_t_max < 1:
            raise ValueError(
                "train.lr_scheduler.T_max must be >= 1 when enabled; "
                f"got {lr_scheduler_t_max}."
            )
        if lr_scheduler_t_max != epochs_in_section:
            # T_max 必须 = epochs (CosineAnnealingLR 退火终点与训练终点对齐), 否则后期 lr 不再衰减或不到 eta_min.
            raise ValueError(
                "train.lr_scheduler.T_max must equal train.epochs when enabled; "
                f"got T_max={lr_scheduler_t_max}, epochs={epochs_in_section}."
            )
        if lr_scheduler_eta_min <= 0.0:
            raise ValueError(
                "train.lr_scheduler.eta_min must be > 0 when enabled; "
                f"got {lr_scheduler_eta_min}."
            )
    return {
        "enabled": enabled,
        "T_max": lr_scheduler_t_max,
        "eta_min": lr_scheduler_eta_min,
    }


def train_model(train_windows, val_windows, train_cfg, *, model_factory=None):
    """Liquid training entrypoint.

    Runs the end-to-end training flow from config parsing through checkpoint
    saving and report emission.
    """
    if train_windows is None:
        raise ValueError("train_windows must not be None.")
    if val_windows is None:
        raise ValueError("val_windows must not be None.")
    if isinstance(train_windows, (str, bytes)):
        raise TypeError("train_windows must be an iterable of windows.")
    if isinstance(val_windows, (str, bytes)):
        raise TypeError("val_windows must be an iterable of windows.")
    try:
        train_windows = list(train_windows)
    except TypeError as exc:
        raise TypeError("train_windows must be an iterable of windows.") from exc
    try:
        val_windows = list(val_windows)
    except TypeError as exc:
        raise TypeError("val_windows must be an iterable of windows.") from exc
    if not train_windows:
        raise ValueError("train_windows must contain at least one window.")
    if not val_windows:
        raise ValueError("val_windows must contain at least one window.")

    from liquidloc.common.tee_logger import print_dict
    _train_section = train_cfg.get("train", {}) if isinstance(train_cfg, dict) else {}
    print_dict({
        "name": train_cfg.get("name") if isinstance(train_cfg, dict) else None,
        "device": train_cfg.get("device") if isinstance(train_cfg, dict) else None,
        "output_root": str(train_cfg.get("output_root")) if isinstance(train_cfg, dict) and train_cfg.get("output_root") else None,
        "optimizer": _train_section.get("optimizer"),
        "lr": _train_section.get("lr"),
        "epochs": _train_section.get("epochs"),
        "phase_schedule": _train_section.get("phase_schedule"),
        "batch_size": _train_section.get("batch_size"),
        "feature_order_len": len(train_cfg.get("feature_order", [])) if isinstance(train_cfg, dict) else None,
        "window": train_cfg.get("window") if isinstance(train_cfg, dict) else None,
        "train_windows_count": len(train_windows) if isinstance(train_windows, (list, tuple)) else None,
        "val_windows_count": len(val_windows) if isinstance(val_windows, (list, tuple)) else None,
    }, "Liquid train_model 入口参数")

    trainer_state = build_trainer_state(train_cfg)
    trainer_state["feature_order"] = _resolve_feature_order(train_windows, trainer_state)
    seed_report = set_global_seed(trainer_state["seed"], deterministic=trainer_state["deterministic"])
    model_cfg = {
        "feature_order": list(trainer_state["feature_order"]),
        "window": dict(trainer_state["window"]),
        "network": dict(trainer_state["network"]),
    }
    if model_factory is None:  # 未注入工厂时懒加载默认实现，保持向后兼容。
        from liquidloc.factories.model_factory import create_model as _default_create_model  # noqa: WPS433
        model_factory = _default_create_model
    model = model_factory(trainer_state["model_name"], model_cfg)
    if not hasattr(model, "predict_intermediate_tensors"):
        raise TypeError("liquid trainer requires a model exposing predict_intermediate_tensors")

    # 准则 27：训练过程收集 5 头激活统计；结束时写入 train_report。
    activation_collector = None
    activation_hook_handles: list[Any] = []
    if bool(getattr(trainer_state["train"], "collect_activation_stats", False)) or bool(
        trainer_state.get("collect_activation_stats", False)
    ):
        from liquidloc.models.activation_stats import (  # 延迟导入避免循环依赖。
            ActivationStatsCollector,
            attach_activation_stats_hooks,
        )
        activation_collector = ActivationStatsCollector()
        activation_hook_handles = attach_activation_stats_hooks(model, activation_collector)

    device, runtime_device = _resolve_runtime_device(trainer_state["device"])
    amp_state = _resolve_amp_state(trainer_state["train"], runtime_device)
    # §13.6.2.5 闭环保持 fail-loud 守门 (第三轮精读补):
    # §13.6.2.5 要求"任一阶段训练/推理滚动须保持 EKF 后端"。
    # Liquid 形态下 EKF 后端 = model.network.cell (Liquid cell 递推链路);
    # 训练前显式断言: 训练阶段使用的模型必须携带 Liquid cell,
    # 防止某阶段"掐掉 EKF 只训网络"再装回 EKF 评测违反 §13.6.2.5。
    _ekf_backend = getattr(model, "network", None)
    if _ekf_backend is not None:
        _cell = getattr(_ekf_backend, "cell", None)
        if _cell is None:
            raise RuntimeError(
                "§13.6.2.5 EKF back-end activation fail-loud: "
                "model.network.cell is None at training start; "
                "Liquid cell (EKF back-end) must be active in all training phases. "
                "A phase that replaces the cell with a static network violates §13.6.2.5."
            )
    if hasattr(model.network, "to"):
        model.network.to(device)
    # v3.1 head-shared: output_backbone 也要移动到 device
    if hasattr(model, "output_backbone") and hasattr(model.output_backbone, "to"):
        model.output_backbone.to(device)
    for head in model.output_heads.values():
        if hasattr(head, "to"):
            head.to(device)
    if hasattr(model, "risk_calibration") and hasattr(model.risk_calibration, "to"):
        model.risk_calibration.to(device)
    model._train_batch_size = int(trainer_state["train"].get("batch_size") or len(train_windows))
    model._eval_batch_size = int(trainer_state["train"].get("eval_batch_size") or len(val_windows))
    model._loss_weights = dict(trainer_state.get("loss_weights") or {})
    model._enable_smooth_regularization = bool(trainer_state.get("enable_smooth_regularization", False))  # v2 smooth regularization 配置。
    model._lr_layer = dict(trainer_state.get("lr_layer") or {})
    # D5：写侧根因修复。build_trainer_state 仅校验 gradient_clip_max_norm 为数值且 > 0。
    # NaN 会让 ``float('nan') <= 0.0`` 为 False 而穿透；bias_huber_delta / scaling_log_eps /
    # calibration_bin_count 均未校验，float()/int() 会把 NaN/Inf/负值静默写入 model 属性，
    # 仅在深层读侧（_compute_batch_loss L2030-2031、_evaluate_model L2925-2930、
    # train_one_epoch L2426-2430、_compute_auxiliary_losses L730-741）被 coerce_finite_scalar 捕获。
    # 报错远离配置源头。此处统一走 coerce_finite_scalar 中心入口，与读侧同口径，在写入点即拒绝非法。
    model._calibration_bin_count = int(coerce_finite_scalar(trainer_state.get("calibration_bin_count", _RISK_CALIBRATION_BIN_COUNT), name="calibration_bin_count", min_value=1))
    model._bias_huber_delta = coerce_finite_scalar(trainer_state.get("bias_huber_delta", _BIAS_HUBER_DELTA), name="bias_huber_delta", min_value=0.0)
    model._scaling_log_eps = coerce_finite_scalar(trainer_state.get("scaling_log_eps", _SCALING_LOG_EPS), name="scaling_log_eps", min_value=0.0)
    model._gradient_clip_max_norm = coerce_finite_scalar(trainer_state.get("gradient_clip_max_norm", 1.0), name="gradient_clip_max_norm", min_value=0.0)
    model._tail_selection_observation_coeff = coerce_finite_scalar(
        trainer_state.get("tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
        name="tail_selection_observation_coeff",
        min_value=0.0,
    )
    model._gate_alignment_readout_lr_scale = coerce_finite_scalar(
        trainer_state.get("gate_alignment_readout_lr_scale", _GATE_ALIGNMENT_READOUT_LR_SCALE),
        name="gate_alignment_readout_lr_scale",
        min_value=0.0,
        inclusive=False,
    )
    cache_device = _resolve_sample_cache_device(model)  # §13.8: 与 model 同设备，释放 CPU 内存
    model._cached_train_windows = _move_materialized_samples(
        _materialize_samples(
            train_windows,
            tail_selection_observation_coeff=model._tail_selection_observation_coeff,
        ),
        cache_device,
    )
    model._cached_val_windows = _move_materialized_samples(
        _materialize_samples(
            val_windows,
            tail_selection_observation_coeff=model._tail_selection_observation_coeff,
        ),
        cache_device,
    )
    progress_path = Path(trainer_state["output_root"]) / "reports" / f'{trainer_state["model_name"]}_train_progress.json'

    output_root = Path(trainer_state["output_root"])  # 输出根目录。
    checkpoints_dir = output_root / "checkpoints"  # 检查点目录。
    reports_dir = output_root / "reports"  # 报告目录。
    checkpoints_dir.mkdir(parents=True, exist_ok=True)  # 创建检查点目录。
    reports_dir.mkdir(parents=True, exist_ok=True)  # 创建报告目录。

    train_epoch_losses: list[float] = []  # 每个 epoch 的训练损失。
    val_epoch_losses: list[float] = []  # 每个 epoch 的验证损失。
    val_selection_scores: list[float] = []  # 每个 epoch 的验证、择分数。
    diagnostics_payload = {  # 损失诊断载荷。
        "model_name": trainer_state["model_name"],  # 模型名称。
        "checkpoint_format": "liquid_real_v1",  # 检查点格式。
        "train": [],  # 训练集诊断。
        "val": [],  # 验证集诊。
    }
    epoch_predictions_payload = {  # 预测-目标快照载荷。
        "model_name": trainer_state["model_name"],  # 模型名称。
        "checkpoint_format": "liquid_real_v1",  # 检查点格式。
        "output_heads": list(_OUTPUT_KEYS),  # 输出头列表。
        "train": [],  # 训练集快照。
        "val": [],  # 验证集快照。
    }
    best_loss: float | None = None # 最优验证损失。
    best_selection_score: float | None = None  # 朢、优、择分数。
    best_epoch: int | None = None  # 本€浼epoch 绱㈠紩。
    best_export_epoch: int | None = None
    best_export_score: float | None = None
    best_ckpt = checkpoints_dir / f'{trainer_state["model_name"]}_best_checkpoint.pt'  # 朢、优检查点路径。
    training_best_ckpt = checkpoints_dir / f'{trainer_state["model_name"]}_training_best_checkpoint.pt'
    export_best_ckpt = checkpoints_dir / f'{trainer_state["model_name"]}_export_best_checkpoint.pt'
    epoch_candidate_paths: list[str] = []  # epoch 候、检查点路径列表。
    epoch_candidate_epochs: list[int] = []  # epoch 候、检查点对应epoch 索引。
    epoch_phase_names: list[str] = []  # 每个 epoch 的阶段名列表。
    phase_trainability_contract: dict[str, dict[str, Any]] = {}  # 各阶段的可训练、合约。
    phase_effective_lrs: dict[str, dict[str, Any]] = {}  # 各阶段的有效学习率。
    save_epoch_candidates = bool(trainer_state["train"].get("save_epoch_candidates"))
    optimizer = None
    prev_phase_name = None
    # audit #22 (CRITICAL, Agent V + Agent X 实证, Agent Z 修): _apply_backbone_unfreeze_ramp 调用守卫变量.
    # 真因: _apply_backbone_unfreeze_ramp (L2679-2778) 内 L2774-2777 把 backbone 组
    # param_group["lr"] 强写为 step3_lr=3e-4 (yaml step1_epochs=0, step2_epochs=0, ramp_epoch 立即满 step1+step2),
    # 与 PyTorch 2.11 CosineAnnealingLR 递推式配合, lr_scheduler 实际 base_lr=3e-4 而非 spec 1e-4,
    # 每 epoch 无守卫调用导致 CosineAnnealingLR 退火结果被擦除, Liquid 实际跑 lr ×3 LSTM, ep80: 2.94e-4 vs LSTM 5.05e-5,
    # 导致 Liquid 过拟合 val window, e9 RMSE 2.6361 vs LSTM 1.4303 (LSTM 胜 84%).
    # 修法: 仅在 phase 切换时调一次 _apply_backbone_unfreeze_ramp, 让 lr_scheduler 退火结果在本 phase 内保留.
    prev_phase_name_for_ramp = None
    phase_runtime_state: dict[str, dict[str, Any]] = {}
    global_best_phase_name: str | None = None
    global_selection_mode = "export_score"  # §13.6.2.7 + §13.3: 导出/论文主结果必须用 export_score 选模;
    # global_selection_mode 仅影响 train_report 元数据与下游 smoke 验证,
    # 不影响训练过程本身. 训练过程的阶段控制仍由 phase_control_score
    # (=supervised_loss) 驱动; 导出/论文主结果绑定 export_score, 禁止
    # 事后在多套分数里挑好看全序. 旧默认 "phase_best_supervised_loss" 与
    # score_role_summary 的 paper_checkpoint_selection_must_use 声明不一致,
    # 修复为与声明对齐.
    score_role_summary = _build_score_role_summary()

    report_path = reports_dir / f'{trainer_state["model_name"]}_train_report.json'
    start_epoch = 1
    initial_phase_name = _resolve_epoch_phase_name(
        1,
        warmup_epochs=int(trainer_state["phase_schedule"]["warmup_epochs"]),
        gate_alignment_epochs=int(trainer_state["phase_schedule"]["gate_alignment_epochs"]),
    )
    _configure_training_phase(model, phase_name=initial_phase_name)
    optimizer = _build_optimizer(model, trainer_state["optimizer"])
    resumed_state = _try_resume_phase_scheduled_training(
        model=model,
        optimizer=optimizer,
        trainer_state=trainer_state,
        report_path=report_path,
        best_ckpt=best_ckpt,
        training_best_ckpt=training_best_ckpt,
        export_best_ckpt=export_best_ckpt,
    )
    if resumed_state is not None:
        start_epoch = int(resumed_state["start_epoch"])
        train_epoch_losses = list(resumed_state["train_epoch_losses"])
        val_epoch_losses = list(resumed_state["val_epoch_losses"])
        val_selection_scores = list(resumed_state["val_selection_scores"])
        epoch_candidate_paths = list(resumed_state["epoch_candidate_paths"])
        epoch_candidate_epochs = list(resumed_state["epoch_candidate_epochs"])
        epoch_phase_names = list(resumed_state["epoch_phase_names"])
        phase_trainability_contract = dict(resumed_state["phase_trainability_contract"])
        phase_effective_lrs = dict(resumed_state["phase_effective_lrs"])
        best_epoch = resumed_state["best_epoch"]
        best_loss = resumed_state["best_loss"]
        best_selection_score = resumed_state["best_selection_score"]
        best_export_epoch = resumed_state.get("best_export_epoch")
        best_export_score = resumed_state.get("best_export_score")
        training_best_checkpoint_path = str(resumed_state.get("training_best_checkpoint") or training_best_ckpt)
        export_best_checkpoint_path = str(resumed_state.get("export_best_checkpoint") or export_best_ckpt)
        phase_runtime_state = _coerce_phase_runtime_state(resumed_state.get("phase_runtime_state"))
        global_selection_mode = str(resumed_state.get("global_selection_mode") or global_selection_mode)
        global_best_phase_name = next(
            (
                str(phase_name)
                for phase_name, payload in phase_runtime_state.items()
                if payload.get("best_supervised_epoch") == best_epoch
            ),
            None,
        )
        prev_phase_name = _resolve_epoch_phase_name(
            int(best_epoch),
            warmup_epochs=int(trainer_state["phase_schedule"]["warmup_epochs"]),
            gate_alignment_epochs=int(trainer_state["phase_schedule"]["gate_alignment_epochs"]),
        )
    else:
        optimizer = None
        training_best_checkpoint_path = str(training_best_ckpt)
        export_best_checkpoint_path = str(export_best_ckpt)
    phase_best_selection_score: dict[str, float] = {
        str(phase_name): float(payload["best_selection_score"])
        for phase_name, payload in phase_runtime_state.items()
        if payload.get("best_selection_score") is not None
    }
    phase_best_supervised_loss: dict[str, float] = {
        str(phase_name): float(payload["best_supervised_loss"])
        for phase_name, payload in phase_runtime_state.items()
        if payload.get("best_supervised_loss") is not None
    }
    soft_control_events: list[dict[str, Any]] = list(resumed_state.get("soft_control_events") or []) if resumed_state is not None else []
    phase_shock_streaks: dict[str, int] = {
        str(phase_name): int(payload.get("shock_streak") or 0)
        for phase_name, payload in phase_runtime_state.items()
    }

    # 偷懒审视 Round 4 + audit #20 H2+M2 (主线程决策 2026-07): liquid trainer 装上 CosineAnnealingLR.
    # lr(t)=eta_min+0.5*(lr0-eta_min)*(1+cos(t*pi/T_max)), T_max=epochs (160), eta_min=1e-6.
    # 注意 liquid trainer 还有 soft_control_shock_threshold (fair overrides 禁了 1e9), 二者并存:
    #   - soft_control: 处理 shock 大跳变 (cooldown/rollback/buffer) — fair overrides 已禁
    #   - lr_scheduler (Round 4 新加 + M2 改 Cosine): 按 epoch 平滑退火 lr (PyTorch CosineAnnealingLR 标准机制)
    # 不启用时 (lr_scheduler.enabled=False) scheduler=None, 兼容 round1-3 行为.
    # 重要: lr_scheduler 实例不能在此处创建 — fresh training (resumed_state is None) 时
    # optimizer 在 L5080 被设为 None, 在此处创建会抛 TypeError: NoneType is not an Optimizer
    # (见重试1 调试日志). 实例创建移到 epoch loop 内 L5176 后 (optimizer 重建后).
    lr_scheduler_state = trainer_state.get("lr_scheduler", {})
    lr_scheduler_enabled = bool(lr_scheduler_state.get("enabled", False))
    lr_scheduler_print_pending = lr_scheduler_enabled  # 首次创建后打一次 enabled 日志
    lr_scheduler = None  # 延迟到 epoch loop 内 optimizer 重建时再实例化

    for epoch_index in range(start_epoch, trainer_state["epochs"] + 1):
        phase_name = _resolve_epoch_phase_name(
            epoch_index,
            warmup_epochs=int(trainer_state["phase_schedule"]["warmup_epochs"]),
            gate_alignment_epochs=int(trainer_state["phase_schedule"]["gate_alignment_epochs"]),
        )
        phase_state_runtime = phase_runtime_state.setdefault(
            phase_name,
            {
                "best_selection_score": phase_best_selection_score.get(phase_name),
                "best_supervised_loss": phase_best_supervised_loss.get(phase_name),
                "best_selection_epoch": None,
                "best_supervised_epoch": None,
                "best_epoch": None,
                "best_selection_checkpoint_path": None,
                "best_supervised_checkpoint_path": None,
                "best_checkpoint_path": None,
                "shock_streak": int(phase_shock_streaks.get(phase_name, 0)),
                "cooldown_remaining": 0,
                "phase_epoch_count": 0,
                "baseline_lrs_by_group": {},
                "rollback_count": 0,
                "reentry_buffer_remaining": 0,
                "reentry_ramp_epoch": 0,
                "reentry_mode": "normal",
                "aux_scale": None,
                "gate_scale": None,
                "calibration_scale": None,
                "reentry_release_scale": None,
                "last_rollback_checkpoint_path": None,
                "same_checkpoint_rollback_count": 0,
                "full_tuning_subphase": None,
                "full_tuning_reentry_target_subphase": None,
                "full_tuning_subphase_entry_epoch": 0,
            },
        )
        if phase_name == "full_tuning":
            resolved_subphase = str(
                phase_state_runtime.get("full_tuning_reentry_target_subphase")
                or phase_state_runtime.get("full_tuning_subphase")
                or _resolve_full_tuning_subphase(trainer_state, phase_state_runtime)
            )
            previous_subphase = str(phase_state_runtime.get("full_tuning_subphase") or "")
            if resolved_subphase != previous_subphase:
                phase_state_runtime["full_tuning_subphase_entry_epoch"] = int(phase_state_runtime.get("phase_epoch_count") or 0)
            phase_state_runtime["full_tuning_subphase"] = resolved_subphase
            phase_state_runtime["full_tuning_reentry_target_subphase"] = None
        phase_state = _configure_training_phase(model, phase_name=phase_name)
        if optimizer is None:
            optimizer = _build_optimizer(model, trainer_state["optimizer"])
            phase_state_runtime["baseline_lrs_by_group"] = _capture_control_group_lrs(optimizer)
        # 偷懒审视 Round 4 + audit #20 M2 (主线程决策 2026-07): lr_scheduler 在 optimizer 重建后 (此处) 创建,
        # 不能在 epoch loop 外创建 — fresh training 时 optimizer=None 会抛 TypeError.
        # 注意 lr_scheduler 是一次性创建, 不在每次 optimizer 重建时重建 (避免 last_epoch 复位 → cos 退火从头开始).
        # lr_scheduler_print_pending 标志确保 enabled 日志只打一次.
        if lr_scheduler_enabled and lr_scheduler is None and optimizer is not None:
            # M2: 改用 CosineAnnealingLR — lr(t)=eta_min+0.5*(lr0-eta_min)*(1+cos(t*pi/T_max)).
            # T_max 默认 = trainer_state["epochs"] (与训练总轮次对齐), 校验已在 _build_lr_scheduler_state 中保证 T_max==epochs.
            lr_scheduler_t_max = int(lr_scheduler_state.get("T_max", trainer_state["epochs"]))
            lr_scheduler_eta_min = float(lr_scheduler_state.get("eta_min", 1e-6))
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=lr_scheduler_t_max,
                eta_min=lr_scheduler_eta_min,
            )
            # audit #20 M1: retry7 重启时从 checkpoint 恢复 lr_scheduler 内部状态
            # (base_lrs / last_epoch / etas). 不 load_state_dict 会丢退火进度,
            # last_epoch 重新从 0 计数 → lr 又从 baseline lr 开始, 退火曲线被打断.
            # audit #26 (2026-08-26): PyTorch 2.11 CosineAnnealingLR.load_state_dict 会用 state 里的
            # T_max/eta_min 覆盖构造时参数 — 如果旧 checkpoint 来自 epochs=30 训练,
            # 加载后会污染当前 epochs=160 训练的 T_max=30 → lr 在 epoch 30 退火到 eta_min, 后续 130 epoch lr 冻结.
            # 修复: 加载后强制用当前 trainer_state 的 T_max/eta_min 覆盖 lr_scheduler 对象属性,
            # 保留 base_lrs/last_epoch/etas 等退火进度.
            if resumed_state is not None:
                saved_lr_scheduler_state = resumed_state.get("lr_scheduler_state_dict")
                if saved_lr_scheduler_state is not None:
                    try:
                        lr_scheduler.load_state_dict(saved_lr_scheduler_state)
                        # 覆盖 T_max/eta_min 为当前训练配置 (避免旧 checkpoint 污染)
                        lr_scheduler.T_max = lr_scheduler_t_max
                        lr_scheduler.eta_min = lr_scheduler_eta_min
                        print(
                            f"[Liquid训练][lr_scheduler] resumed from checkpoint "
                            f"(T_max={lr_scheduler_state.get('T_max', trainer_state['epochs'])} "
                            f"eta_min={lr_scheduler_state.get('eta_min', 1e-6)})",
                            flush=True,
                        )
                    except Exception as exc:
                        # load_state_dict 失败不应中断训练 — 退化到 fresh lr_scheduler
                        # 但要在日志中明示 (用户铁律 #14: 失败要可见, 不静默吞).
                        print(
                            f"[Liquid训练][lr_scheduler] WARNING: load_state_dict failed "
                            f"(exc={exc}); falling back to fresh lr_scheduler state",
                            flush=True,
                        )
            if lr_scheduler_print_pending:
                print(
                    f"[Liquid训练][lr_scheduler] enabled CosineAnnealingLR "
                    f"T_max={lr_scheduler_state.get('T_max', trainer_state['epochs'])} "
                    f"eta_min={lr_scheduler_state.get('eta_min', 1e-6)}",
                    flush=True,
                )
                lr_scheduler_print_pending = False
        phase_control_state = _build_phase_control_state(phase_name, trainer_state, phase_state_runtime)
        model._phase_control_state = dict(phase_control_state)
        phase_state_runtime["aux_scale"] = phase_control_state.get("aux_scale")
        phase_state_runtime["gate_scale"] = phase_control_state.get("gate_scale")
        phase_state_runtime["calibration_scale"] = phase_control_state.get("calibration_scale")
        phase_state_runtime["regularization_scale"] = phase_control_state.get("regularization_scale")
        phase_state_runtime["reentry_release_scale"] = phase_control_state.get("reentry_release_scale")
        if phase_name == "full_tuning":
            phase_state_runtime["full_tuning_subphase"] = phase_control_state.get("full_tuning_subphase")
            phase_state_runtime["full_tuning_subphase_entry_epoch"] = int(
                phase_control_state.get("full_tuning_subphase_entry_epoch") or 0
            )
        phase_contract = _build_phase_trainability_contract(model, phase_name=phase_name)
        # audit #20 H1 (CRITICAL): _apply_phase_group_lrs 必须只在 phase_name 切换时执行,
        # 不能每 epoch 调用 — 否则会用 phase_config baseline lr 覆盖 optimizer.param_group['lr'],
        # 静默擦除上一 epoch 末 lr_scheduler.step() 触发的 CosineAnnealingLR 退火结果
        # (lr_scheduler.step 在 epoch 末修改 param_group['lr'], 但此处每 epoch 起始又会把它改回 baseline).
        # 修法: 加 prev_phase_name 守卫, phase 不变时跳过整个 _apply_phase_group_lrs 调用,
        # 让 lr_scheduler 的退火结果在本 phase 内持续保留并随 checkpoint 持久化.
        # 注: CosineAnnealingLR 每 epoch 给固定 lr (按 cos 函数, 不依赖 plateau 检测),
        # 但 _apply_phase_group_lrs 仍会擦除 cos 退火结果 → H1 守卫保留有价值.
        if optimizer is not None and phase_name != prev_phase_name:
            _apply_phase_group_lrs(optimizer, model, trainer_state["optimizer"])
            phase_state_runtime["baseline_lrs_by_group"] = _capture_control_group_lrs(optimizer)
        if phase_name == "full_tuning":
            # audit #22 (CRITICAL, Agent V + Agent X 实证 lr 退火 bug, Agent Z 修):
            # _apply_backbone_unfreeze_ramp 内 L2774-2777 把 backbone 组 param_group["lr"] 强写为
            # step3_lr=3e-4 (yaml step1_epochs=0, step2_epochs=0 时 ramp_epoch 立即满 step1+step2).
            # 每 epoch 无守卫调用会擦除 lr_scheduler.step() 在 epoch 末写入的 CosineAnnealingLR 退火结果,
            # 使 lr_scheduler 实际 base_lr=3e-4 而非 spec 1e-4, Liquid 实际跑 lr ×3 LSTM导致过拟合 val window.
            # 修法: 仅在 phase 切换 (phase_name != prev_phase_name_for_ramp) 时调一次 ramp 设 baseline,
            # 让后续 epoch 的 CosineAnnealingLR 退火结果在本 phase 内持续保留. 与 L5246 H1 守卫同模式.
            if phase_name != prev_phase_name_for_ramp:
                ramp_epoch = _compute_full_tuning_ramp_epoch(epoch_index, trainer_state, phase_state_runtime)
                _apply_backbone_unfreeze_ramp(
                    optimizer,
                    model,
                    epoch_index,
                    trainer_state,
                    ramp_epoch=ramp_epoch,
                )
                phase_state_runtime["reentry_ramp_epoch"] = int(ramp_epoch) + 1
                # 守卫置位: 本次 phase 进入首次已 ramp, 后续 epoch 跳过让 lr_scheduler 退火结果保留.
                prev_phase_name_for_ramp = phase_name
            if str(phase_state_runtime.get("reentry_mode") or "normal") == "conservative":
                conservative_cap_multiplier = (
                    _FULL_TUNING_REENTRY_REPEAT_CAP_MULTIPLIER
                    if int(phase_state_runtime.get("same_checkpoint_rollback_count") or 0) >= _FULL_TUNING_REPEAT_ROLLBACK_THRESHOLD
                    else 1.0
                )
                conservative_cap_multiplier *= coerce_finite_scalar(
                    phase_control_state.get("reentry_release_scale", 1.0),
                    name="phase_control_state.reentry_release_scale",
                    min_value=0.0,
                )
                _apply_group_lr_cap(
                    optimizer,
                    baseline_lrs_by_group=dict(phase_state_runtime.get("baseline_lrs_by_group") or {}),
                    cap_scales_by_group={
                        "readout": _FULL_TUNING_REENTRY_READOUT_LR_CAP_SCALE * conservative_cap_multiplier,
                        "gate_cal": _FULL_TUNING_REENTRY_GATE_CAL_LR_CAP_SCALE * conservative_cap_multiplier,
                        "filter_context_gate": _FULL_TUNING_REENTRY_FILTER_GATE_LR_CAP_SCALE * conservative_cap_multiplier,
                    },
                )
        prev_phase_name = phase_name
        phase_state_runtime["phase_epoch_count"] = int(phase_state_runtime.get("phase_epoch_count") or 0) + 1
        if phase_name == "full_tuning":
            phase_state_runtime["reentry_buffer_remaining"] = max(
                0,
                int(phase_state_runtime.get("reentry_buffer_remaining") or 0) - 1,
            )
        # §13.6.2.1 rollback 总 epoch 上限 fail-loud 守门 (第三轮精读补):
        # 守门逻辑提取为模块级纯函数 `_assert_phase_epoch_count_within_budget`,
        # 既被训练循环调用, 也被单元测试直接调用 (避免守门成为不可触达的内联代码).
        _assert_phase_epoch_count_within_budget(phase_name, phase_state_runtime, trainer_state)
        model._phase_control_state = _build_phase_control_state(phase_name, trainer_state, phase_state_runtime)
        phase_state_runtime["aux_scale"] = float(model._phase_control_state.get("aux_scale", 1.0))
        phase_state_runtime["gate_scale"] = float(model._phase_control_state.get("gate_scale", 1.0))
        phase_state_runtime["calibration_scale"] = float(model._phase_control_state.get("calibration_scale", 1.0))
        phase_state_runtime["reentry_release_scale"] = float(model._phase_control_state.get("reentry_release_scale", 1.0))
        epoch_phase_names.append(phase_name)
        phase_trainability_contract[phase_name] = dict(phase_contract)
        phase_effective_lrs[phase_name] = _optimizer_effective_state(optimizer)
        train_loss = train_one_epoch(
            train_windows,
            val_windows,
            epoch_index,
            model=model,
            optimizer=optimizer,
        )
        _, train_batches, train_epoch_snapshot = _evaluate_model(  # 评估训练集。
            model,
            train_windows,
            epoch_index=epoch_index,
            split="train",
            collect_diagnostics=True,
        )
        val_loss, val_batches, val_epoch_snapshot = _evaluate_model(  # 评估验证集。
            model,
            val_windows,
            epoch_index=epoch_index,
            split="val",
            collect_diagnostics=True,
        )
        optimization_loss = coerce_finite_scalar(
            getattr(model, "_optimization_loss", val_loss),
            name="model._optimization_loss",
            min_value=0.0,
        )
        phase_control_score = coerce_finite_scalar(
            getattr(model, "_phase_control_score", val_epoch_snapshot.get("supervised_loss", val_loss)),
            name="model._phase_control_score",
            min_value=0.0,
        )
        export_score = coerce_finite_scalar(
            getattr(model, "_export_score", val_epoch_snapshot.get("supervised_loss", val_loss)),
            name="model._export_score",
            min_value=0.0,
        )
        selection_score = float(model._selection_score)
        train_supervised_loss = float(train_epoch_snapshot.get("supervised_loss", train_loss))
        val_supervised_loss = float(val_epoch_snapshot.get("supervised_loss", val_loss))
        current_phase_best_selection = phase_best_selection_score.get(phase_name)
        current_phase_best_supervised = phase_best_supervised_loss.get(phase_name)
        phase_switch_shock = {
            "selection_score_relative": (
                None
                if current_phase_best_selection is None or current_phase_best_selection <= 0.0
                else (phase_control_score - current_phase_best_selection) / current_phase_best_selection
            ),
            "supervised_loss_relative": (
                None
                if current_phase_best_supervised is None or current_phase_best_supervised <= 0.0
                else (val_supervised_loss - current_phase_best_supervised) / current_phase_best_supervised
            ),
        }
        should_refresh_phase_best_selection = (
            current_phase_best_selection is None or phase_control_score < current_phase_best_selection
        )
        should_refresh_phase_best_supervised = (
            current_phase_best_supervised is None or val_supervised_loss < current_phase_best_supervised
        )
        checkpoint_payload = _build_checkpoint_payload(
            model=model,
            trainer_state=trainer_state,
            optimizer=optimizer,
            seed_report=seed_report,
            best_epoch=epoch_index,
            best_loss=float(val_loss),
            train_window_count=len(train_windows),
            val_window_count=len(val_windows),
            current_epoch=epoch_index,
            lr_scheduler=lr_scheduler,
        )
        checkpoint_payload["phase_runtime_state"] = _serialize_phase_runtime_state(phase_runtime_state)
        shock_trigger_value = phase_switch_shock.get("supervised_loss_relative")
        if shock_trigger_value is None:
            shock_trigger_value = phase_switch_shock.get("selection_score_relative")
        shock_threshold = coerce_finite_scalar(
            trainer_state.get("soft_control_shock_threshold", _SOFT_CONTROL_SHOCK_THRESHOLD),
            name="soft_control_shock_threshold",
            min_value=0.0,
            inclusive=False,
        )
        previous_shock_streak = int(phase_shock_streaks.get(phase_name, 0))
        if shock_trigger_value is None:
            current_shock_streak = 0
        else:
            shock_trigger_value = max(
                coerce_finite_scalar(
                    shock_trigger_value,
                    name="phase_switch_shock.current_trigger_value",
                ),
                0.0,
            )
            current_shock_streak = previous_shock_streak + 1 if shock_trigger_value >= shock_threshold else 0
        phase_shock_streaks[phase_name] = int(current_shock_streak)
        phase_state_runtime["shock_streak"] = int(current_shock_streak)
        soft_control_event = _maybe_apply_phase_shock_soft_control(
            optimizer,
            phase_name=phase_name,
            phase_switch_shock=phase_switch_shock,
            trainer_state=trainer_state,
            prior_shock_streak=previous_shock_streak,
            cooldown_remaining=int(phase_state_runtime.get("cooldown_remaining") or 0),
            phase_epoch_count=int(phase_state_runtime.get("phase_epoch_count") or 0),
            reentry_buffer_remaining=int(phase_state_runtime.get("reentry_buffer_remaining") or 0),
            baseline_lrs_by_group=dict(phase_state_runtime.get("baseline_lrs_by_group") or {}),
            full_tuning_subphase=phase_state_runtime.get("full_tuning_subphase"),
        )
        if soft_control_event is not None:
            soft_control_event = {
                **soft_control_event,
                "epoch_index": int(epoch_index),
            }
            defer_soft_control_append = bool(soft_control_event.get("request_rollback"))
            if soft_control_event["mode"] not in {"buffer_only", "cooldown_only"}:
                phase_effective_lrs[phase_name] = _optimizer_effective_state(optimizer)
            if not defer_soft_control_append and soft_control_event["mode"] not in {"buffer_only", "cooldown_only"}:
                soft_control_events.append(dict(soft_control_event))
            request_rollback = bool(soft_control_event.get("request_rollback"))
            cooldown_epochs = int(soft_control_event.get("cooldown_epochs") or trainer_state.get("soft_control_cooldown_epochs", _SOFT_CONTROL_COOLDOWN_EPOCHS))
            if request_rollback:
                rollback_checkpoint_path = str(phase_state_runtime.get("best_supervised_checkpoint_path") or "").strip()
                requested_phase_best_checkpoint_path = rollback_checkpoint_path or None
                rollback_source_phase = phase_name
                used_upstream_fallback = False
                repeated_same_checkpoint_count = 0
                # Surgical fix: when full_tuning has no in-phase best_supervised_checkpoint_path
                # (e.g. the very first epoch triggered shock and never improved), immediately
                # fall back to the upstream gate_alignment best checkpoint. Without this guard
                # the downstream repeat-detection / upstream-fallback chain requires two
                # rollback attempts (repeat-threshold=2) before reaching the upstream fallback
                # at line 5062, which means the first shock -> slowdown_only -> infinite cooldown
                # cycle still burns 75+ epochs without restoring from gate_alignment best.
                if not rollback_checkpoint_path and phase_name == "full_tuning":
                    candidate_upstream_path = str(
                        phase_state_runtime.get("gate_alignment", {}).get("best_supervised_checkpoint_path") or ""
                    ).strip()
                    if candidate_upstream_path:
                        rollback_checkpoint_path = candidate_upstream_path
                        rollback_source_phase = "gate_alignment"
                        used_upstream_fallback = True
                if phase_name == "full_tuning" and requested_phase_best_checkpoint_path:
                    last_rollback_checkpoint_path = str(phase_state_runtime.get("last_rollback_checkpoint_path") or "").strip()
                    repeated_same_checkpoint_count = (
                        int(phase_state_runtime.get("same_checkpoint_rollback_count") or 0) + 1
                        if requested_phase_best_checkpoint_path == last_rollback_checkpoint_path
                        else 1
                    )
                    if repeated_same_checkpoint_count >= _FULL_TUNING_REPEAT_ROLLBACK_THRESHOLD:
                        upstream_phase_name = _resolve_phase_rollback_source_name(phase_name)
                        upstream_checkpoint_path = str(
                            phase_runtime_state.get(upstream_phase_name, {}).get("best_supervised_checkpoint_path") or ""
                        ).strip() if upstream_phase_name is not None else ""
                        if upstream_checkpoint_path:
                            rollback_checkpoint_path = upstream_checkpoint_path
                            rollback_source_phase = str(upstream_phase_name)
                            used_upstream_fallback = True
                if not rollback_checkpoint_path:
                    upstream_phase_name = _resolve_phase_rollback_source_name(phase_name)
                    if upstream_phase_name is not None:
                        rollback_checkpoint_path = str(
                            phase_runtime_state.get(upstream_phase_name, {}).get("best_supervised_checkpoint_path") or ""
                        ).strip()
                        rollback_source_phase = upstream_phase_name
                rollback_payload = _load_checkpoint_payload(Path(rollback_checkpoint_path)) if rollback_checkpoint_path else None
                rollback_restored = False
                if rollback_payload is not None:
                    rollback_phase_name = str(rollback_payload.get("phase_name") or phase_name)
                    rollback_phase_state = _configure_training_phase(model, phase_name=rollback_phase_name)
                    optimizer = _build_optimizer(model, trainer_state["optimizer"])
                    rollback_restored = _restore_training_snapshot(
                        model=model,
                        optimizer=optimizer,
                        checkpoint_payload=rollback_payload,
                    )
                    prev_phase_name = rollback_phase_name if rollback_restored else prev_phase_name
                    # audit #22: rollback 时同步 prev_phase_name_for_ramp, 让 rollback 内 L5489 的
                    # _apply_backbone_unfreeze_ramp(ramp_epoch=0) 保守 ramp 结果在后续 epoch 中保留,
                    # 不被 L5265 守卫块的常规 ramp_epoch 调用覆盖. 与 prev_phase_name 同模式同步.
                    prev_phase_name_for_ramp = rollback_phase_name if rollback_restored else prev_phase_name_for_ramp
                    if rollback_restored:
                        if rollback_phase_name == "full_tuning":
                            _apply_backbone_unfreeze_ramp(
                                optimizer,
                                model,
                                epoch_index,
                                trainer_state,
                                ramp_epoch=0,
                            )
                            cap_multiplier = (
                                _FULL_TUNING_REENTRY_UPSTREAM_CAP_MULTIPLIER
                                if used_upstream_fallback
                                else (
                                    _FULL_TUNING_REENTRY_REPEAT_CAP_MULTIPLIER
                                    if repeated_same_checkpoint_count >= _FULL_TUNING_REPEAT_ROLLBACK_THRESHOLD
                                    else 1.0
                                )
                            )
                            conservative_lr_adjustments = _apply_group_lr_cap(
                                optimizer,
                                baseline_lrs_by_group=dict(phase_state_runtime.get("baseline_lrs_by_group") or {}),
                                cap_scales_by_group={
                                    "readout": _FULL_TUNING_REENTRY_READOUT_LR_CAP_SCALE * cap_multiplier,
                                    "gate_cal": _FULL_TUNING_REENTRY_GATE_CAL_LR_CAP_SCALE * cap_multiplier,
                                    "filter_context_gate": _FULL_TUNING_REENTRY_FILTER_GATE_LR_CAP_SCALE * cap_multiplier,
                                },
                            )
                        else:
                            conservative_lr_adjustments = []
                        phase_effective_lrs[rollback_phase_name] = _optimizer_effective_state(optimizer)
                        phase_state_runtime["cooldown_remaining"] = int(cooldown_epochs)
                        phase_state_runtime["shock_streak"] = 0
                        phase_state_runtime["rollback_count"] = int(phase_state_runtime.get("rollback_count") or 0) + 1
                        # P5 (v7 patch): 抗 rollback 重置 phase_epoch_count 训练预算归零.
                        # 子代 G 报告: 原 trainer.py:5260 直接 = 0, 让最末 24 ep 用旧逻辑 ramp
                        # 重头来过, 等于丢掉 late_consolidation 已完成 budget. patch 改为
                        # 只回退 cooldown_epochs 个 epoch, 保留主体 late_consolidation 进度.
                        cooldown_epochs = int(trainer_state.get("soft_control_cooldown_epochs", _SOFT_CONTROL_COOLDOWN_EPOCHS))
                        previous_phase_epoch_count = int(phase_state_runtime.get("phase_epoch_count") or 0)
                        phase_state_runtime["phase_epoch_count"] = max(0, previous_phase_epoch_count - cooldown_epochs)
                        phase_state_runtime["last_rollback_phase_epoch_count"] = previous_phase_epoch_count  # 审计用
                        phase_state_runtime["reentry_ramp_epoch"] = 0
                        phase_state_runtime["reentry_buffer_remaining"] = int(trainer_state.get("soft_control_buffer_epochs", _SOFT_CONTROL_BUFFER_EPOCHS)) + (
                            _FULL_TUNING_UPSTREAM_FALLBACK_EXTRA_BUFFER_EPOCHS
                            if used_upstream_fallback
                            else _FULL_TUNING_REENTRY_EXTRA_BUFFER_EPOCHS * _FULL_TUNING_ENTRY_BUFFER_MULTIPLIER
                        )
                        phase_state_runtime["reentry_mode"] = "conservative"
                        phase_state_runtime["aux_scale"] = 0.0
                        phase_state_runtime["gate_scale"] = 0.0
                        phase_state_runtime["calibration_scale"] = 0.0
                        phase_state_runtime["reentry_release_scale"] = 0.0
                        if phase_name == "full_tuning":
                            phase_state_runtime["last_rollback_checkpoint_path"] = requested_phase_best_checkpoint_path
                            phase_state_runtime["same_checkpoint_rollback_count"] = 0 if used_upstream_fallback else int(repeated_same_checkpoint_count)
                            phase_state_runtime["full_tuning_reentry_target_subphase"] = (
                                "entry_bridge"
                                if not used_upstream_fallback
                                else "joint_drive"
                            )
                            phase_state_runtime["full_tuning_subphase"] = str(
                                phase_state_runtime.get("full_tuning_reentry_target_subphase") or "entry_bridge"
                            )
                            phase_state_runtime["full_tuning_subphase_entry_epoch"] = 0
                        phase_shock_streaks[phase_name] = 0
                        soft_control_event = {
                            **soft_control_event,
                            "mode": "rollback_and_slowdown",
                            "rollback_checkpoint_path": rollback_checkpoint_path,
                            "rollback_restored": True,
                            "rollback_phase_name": rollback_phase_name,
                            "rollback_source_phase": rollback_source_phase,
                            "requested_phase_best_checkpoint_path": requested_phase_best_checkpoint_path,
                            "used_upstream_fallback": bool(used_upstream_fallback),
                            "same_checkpoint_rollback_count": int(repeated_same_checkpoint_count),
                            "conservative_lr_adjustments": conservative_lr_adjustments,
                            "post_restore_phase_state": dict(rollback_phase_state),
                        }
                        soft_control_events.append(dict(soft_control_event))
                if not rollback_restored:
                    phase_state_runtime["cooldown_remaining"] = int(cooldown_epochs)
                    if phase_name == "full_tuning":
                        phase_state_runtime["last_rollback_checkpoint_path"] = requested_phase_best_checkpoint_path
                        phase_state_runtime["same_checkpoint_rollback_count"] = 0 if used_upstream_fallback else int(repeated_same_checkpoint_count)
                        phase_state_runtime["reentry_mode"] = "conservative"
                        phase_state_runtime["aux_scale"] = 0.0
                        phase_state_runtime["gate_scale"] = 0.0
                        phase_state_runtime["calibration_scale"] = 0.0
                        phase_state_runtime["reentry_release_scale"] = 0.0
                        phase_state_runtime["full_tuning_reentry_target_subphase"] = (
                            "entry_bridge"
                            if not used_upstream_fallback
                            else "joint_drive"
                        )
                        phase_state_runtime["full_tuning_subphase"] = str(
                            phase_state_runtime.get("full_tuning_reentry_target_subphase") or "entry_bridge"
                        )
                        phase_state_runtime["full_tuning_subphase_entry_epoch"] = 0
                    soft_control_event = {
                        **soft_control_event,
                        "rollback_checkpoint_path": rollback_checkpoint_path or None,
                        "rollback_source_phase": rollback_source_phase,
                        "rollback_restored": False,
                        "requested_phase_best_checkpoint_path": requested_phase_best_checkpoint_path,
                        "used_upstream_fallback": bool(used_upstream_fallback),
                        "same_checkpoint_rollback_count": int(repeated_same_checkpoint_count),
                    }
                    soft_control_events.append(dict(soft_control_event))
            elif soft_control_event["mode"] not in {"buffer_only", "cooldown_only"}:
                # Fix: cooldown_recovery should NOT re-arm cooldown_remaining,
                # otherwise should_request_rollback can never be reached after a
                # slowdown_only event (cooldown is infinite).
                if soft_control_event["mode"] != "cooldown_recovery":
                    phase_state_runtime["cooldown_remaining"] = int(cooldown_epochs)
        if soft_control_event is None and int(phase_state_runtime.get("cooldown_remaining") or 0) > 0:
            phase_state_runtime["cooldown_remaining"] = max(0, int(phase_state_runtime.get("cooldown_remaining") or 0) - 1)
        elif soft_control_event is not None and soft_control_event.get("mode") == "cooldown_only":
            phase_state_runtime["cooldown_remaining"] = max(0, int(phase_state_runtime.get("cooldown_remaining") or 0) - 1)
        if (
            phase_name == "full_tuning"
            and soft_control_event is None
            and int(current_shock_streak) == 0
            and int(phase_state_runtime.get("cooldown_remaining") or 0) == 0
            and int(phase_state_runtime.get("phase_epoch_count") or 0) > int(trainer_state.get("soft_control_buffer_epochs", _SOFT_CONTROL_BUFFER_EPOCHS))
        ):
            phase_state_runtime["reentry_mode"] = "normal"
        if phase_name == "full_tuning":
            next_subphase = _resolve_full_tuning_subphase(trainer_state, phase_state_runtime)
            current_subphase = str(phase_state_runtime.get("full_tuning_subphase") or "")
            if next_subphase != current_subphase:
                phase_state_runtime["full_tuning_subphase"] = next_subphase
                phase_state_runtime["full_tuning_subphase_entry_epoch"] = int(phase_state_runtime.get("phase_epoch_count") or 0)
        train_epoch_losses.append(float(train_loss))
        val_epoch_losses.append(float(val_loss))
        val_selection_scores.append(selection_score)
        print(
        f"[Liquid??] ? {epoch_index}/{trainer_state['epochs']} ? "
        f"????={float(train_loss):.6f} ????={float(val_loss):.6f} "
        f"????={train_supervised_loss:.6f} ????={val_supervised_loss:.6f} "
        f"??={float((val_epoch_snapshot.get('auxiliary_losses') or {}).get('total', 0.0)):.6f} "
        f"????={selection_score:.6f} ??={phase_name}"
            f"{'' if phase_name != 'full_tuning' else '/' + str(phase_state_runtime.get('full_tuning_subphase') or 'joint_drive')}",
            flush=True,
        )
        diagnostics_payload["train"].append(
            {
                **_collect_epoch_loss_diagnostics(
                    split="train",
                    epoch_index=epoch_index,
                    batches=train_batches,
                    selection_score=None,
                    supervised_loss=train_supervised_loss,
                    auxiliary_summary=train_epoch_snapshot.get("auxiliary_losses"),
                ),
                "phase_name": phase_name,
                "phase_state": dict(phase_state),
                "phase_control_state": dict(model._phase_control_state),
                "phase_trainability_contract": dict(phase_contract),
                "phase_effective_lr": dict(phase_effective_lrs[phase_name]),
                "shock_streak": {"phase": int(current_shock_streak)},
                "soft_control_event": None if soft_control_event is None else dict(soft_control_event),
            }
        )
        diagnostics_payload["val"].append(
            {
                **_collect_epoch_loss_diagnostics(
                    split="val",
                    epoch_index=epoch_index,
                    batches=val_batches,
                    selection_score=selection_score,
                    supervised_loss=val_supervised_loss,
                    auxiliary_summary=val_epoch_snapshot.get("auxiliary_losses"),
                    phase_switch_shock=phase_switch_shock,
                ),
                "phase_name": phase_name,
                "phase_state": dict(phase_state),
                "phase_control_state": dict(model._phase_control_state),
                "phase_trainability_contract": dict(phase_contract),
                "phase_effective_lr": dict(phase_effective_lrs[phase_name]),
                "shock_streak": {"phase": int(current_shock_streak)},
                "soft_control_event": None if soft_control_event is None else dict(soft_control_event),
            }
        )
        train_epoch_snapshot = dict(train_epoch_snapshot)
        train_epoch_snapshot["phase_name"] = phase_name
        train_epoch_snapshot["phase_state"] = dict(phase_state)
        train_epoch_snapshot["phase_control_state"] = dict(model._phase_control_state)
        train_epoch_snapshot["phase_trainability_contract"] = dict(phase_contract)
        train_epoch_snapshot["phase_effective_lr"] = dict(phase_effective_lrs[phase_name])
        train_epoch_snapshot["shock_streak"] = {"phase": int(current_shock_streak)}
        train_epoch_snapshot["soft_control_event"] = None if soft_control_event is None else dict(soft_control_event)
        val_epoch_snapshot = dict(val_epoch_snapshot)
        val_epoch_snapshot["phase_name"] = phase_name
        val_epoch_snapshot["phase_state"] = dict(phase_state)
        val_epoch_snapshot["phase_control_state"] = dict(model._phase_control_state)
        val_epoch_snapshot["phase_trainability_contract"] = dict(phase_contract)
        val_epoch_snapshot["phase_effective_lr"] = dict(phase_effective_lrs[phase_name])
        val_epoch_snapshot["phase_switch_shock"] = dict(phase_switch_shock)
        val_epoch_snapshot["shock_streak"] = {"phase": int(current_shock_streak)}
        val_epoch_snapshot["soft_control_event"] = None if soft_control_event is None else dict(soft_control_event)
        epoch_predictions_payload["train"].append(train_epoch_snapshot)
        epoch_predictions_payload["val"].append(val_epoch_snapshot)
        if should_refresh_phase_best_selection:
            phase_best_selection_score[phase_name] = phase_control_score
            phase_state_runtime["best_selection_score"] = phase_control_score
            phase_state_runtime["best_selection_epoch"] = int(epoch_index)
        if should_refresh_phase_best_supervised:
            phase_best_supervised_loss[phase_name] = val_supervised_loss
            phase_state_runtime["best_supervised_loss"] = val_supervised_loss
            phase_state_runtime["best_supervised_epoch"] = int(epoch_index)
            phase_state_runtime["best_epoch"] = int(epoch_index)
        if should_refresh_phase_best_selection or should_refresh_phase_best_supervised:
            phase_best_ckpt = _build_phase_checkpoint_path(checkpoints_dir, trainer_state["model_name"], phase_name)
            checkpoint_payload["phase_name"] = str(phase_name)
            checkpoint_payload["phase_runtime_state"] = _serialize_phase_runtime_state(phase_runtime_state)
            torch.save(checkpoint_payload, phase_best_ckpt)
            if should_refresh_phase_best_selection:
                phase_state_runtime["best_selection_checkpoint_path"] = str(phase_best_ckpt)
            if should_refresh_phase_best_supervised:
                phase_state_runtime["best_supervised_checkpoint_path"] = str(phase_best_ckpt)
            phase_state_runtime["best_checkpoint_path"] = (
                phase_state_runtime.get("best_supervised_checkpoint_path")
                or phase_state_runtime.get("best_selection_checkpoint_path")
            )
        should_save_candidate = False
        if save_epoch_candidates:
            if epoch_index == 1 or epoch_index == trainer_state["epochs"]:
                should_save_candidate = True
            elif epoch_index % trainer_state["epoch_candidate_stride"] == 0:  # 按步长保存。
                should_save_candidate = True  # 标记保存。
        if should_save_candidate:  # 保存 epoch 候、检查点。
            candidate_path = checkpoints_dir / f'{trainer_state["model_name"]}_epoch_{epoch_index:03d}.pt'  # 值欓€夎矾寰勩€。
            torch.save(checkpoint_payload, candidate_path)
            epoch_candidate_paths.append(str(candidate_path))
            epoch_candidate_epochs.append(epoch_index)
        global_candidate_supervised = phase_best_supervised_loss.get(phase_name)
        should_refresh_global_best = (
            global_candidate_supervised is not None
            and (best_loss is None or global_candidate_supervised < best_loss)
        )
        if should_refresh_global_best:
            best_selection_score = phase_best_selection_score.get(phase_name)
            best_loss = float(global_candidate_supervised)
            best_epoch = int(phase_state_runtime.get("best_supervised_epoch") or epoch_index)
            global_best_phase_name = str(phase_name)
            checkpoint_payload["best_epoch"] = best_epoch
            checkpoint_payload["best_loss"] = best_loss
            checkpoint_payload["phase_name"] = str(phase_name)
            checkpoint_payload["phase_runtime_state"] = _serialize_phase_runtime_state(phase_runtime_state)
            torch.save(checkpoint_payload, best_ckpt)  # 保存朢、优检查点。
            torch.save(checkpoint_payload, training_best_ckpt)
            training_best_checkpoint_path = str(training_best_ckpt)

        # 偷懒审视 Round 4 + audit #20 H2+M2 (主线程决策 2026-07): epoch 末 CosineAnnealingLR 调度 lr.
        # lr(t)=eta_min+0.5*(lr0-eta_min)*(1+cos(t*pi/T_max)), lr0=1e-4 退火到 eta_min=1e-6, T_max=160.
        # 不需要 val_loss 参数, 直接 step() 按 last_epoch+1 退火 (不依赖 plateau 检测, 无 silent bug 风险).
        # 不启用时 lr_scheduler=None (向后兼容 round1-3 行为).
        if lr_scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step()
            new_lr = optimizer.param_groups[0]["lr"]
            # CosineAnnealingLR 每 epoch 都退火 (lr 单调下降), 打印任意 ep 的 lr 便于追踪.
            print(
                f"[Liquid训练][lr_scheduler] ep{epoch_index} lr 退火 "
                f"{current_lr:.2e} → {new_lr:.2e} (cos T_max={lr_scheduler.T_max} eta_min={lr_scheduler.eta_min:.2e})",
                flush=True,
            )

        current_export_epoch = int(epoch_index)
        should_refresh_export_best = (
            export_score is not None
            and (
                best_export_score is None
                or float(export_score) < float(best_export_score)
                or (
                    math.isclose(float(export_score), float(best_export_score), rel_tol=0.0, abs_tol=1e-12)
                    and (
                        best_export_epoch is None
                        or int(current_export_epoch) > int(best_export_epoch)
                    )
                )
            )
        )
        if should_refresh_export_best:
            # §13.6.2.7 + §13.3 audit fix: export_best_ckpt 写盘时 payload 中的
            # best_epoch / best_loss 必须记录 export_score 视角下的最优, 不能沿用
            # 上面 supervised 路径写入的 best_epoch/best_loss. 此前代码把
            # checkpoint_payload["best_loss"] 留成 supervised 的 best_loss, 同时把
            # ["best_epoch"] 改写成 current_export_epoch, 两者口径不一致, 让落盘的
            # export checkpoint 文件元数据 'best_loss' 串号成 supervised 分数, 下游
            # 加载 / 重启 resume 校验读取错位值. 修: 显式写 export_score 视角的最优,
            # 并优先回填已记录的 best_export_epoch, 保持口径一致.
            best_export_score = float(export_score)
            best_export_epoch = int(current_export_epoch)
            checkpoint_payload["best_epoch"] = best_export_epoch
            checkpoint_payload["best_loss"] = best_export_score
            checkpoint_payload["phase_name"] = str(phase_name)
            checkpoint_payload["phase_runtime_state"] = _serialize_phase_runtime_state(phase_runtime_state)
            torch.save(checkpoint_payload, export_best_ckpt)
            export_best_checkpoint_path = str(export_best_ckpt)
        checkpoint_payload["phase_runtime_state"] = _serialize_phase_runtime_state(phase_runtime_state)
        runtime_ckpt = checkpoints_dir / f'{trainer_state["model_name"]}_resume_checkpoint.pt'
        torch.save(checkpoint_payload, runtime_ckpt)

        checkpoint_role_summary = _build_checkpoint_role_summary(
            phase_runtime_state,
            current_phase_name=phase_name,
            control_anchor_checkpoint=phase_state_runtime.get("best_supervised_checkpoint_path"),
            training_best_checkpoint=training_best_checkpoint_path,
            export_best_checkpoint=export_best_checkpoint_path,
        )

        _write_epoch_progress_snapshot(
            progress_path,
            trainer_state=trainer_state,
            history_start_epoch=start_epoch,
            epoch_index=epoch_index,
            phase_name=phase_name,
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            selection_score=selection_score,
            optimization_loss=optimization_loss,
            phase_control_score=phase_control_score,
            export_score=export_score,
            train_supervised_loss=train_supervised_loss,
            val_supervised_loss=val_supervised_loss,
            val_auxiliary_losses=val_epoch_snapshot.get("auxiliary_losses"),
            val_component_losses=val_epoch_snapshot.get("component_losses"),
            phase_switch_shock=phase_switch_shock,
            shock_streak={"phase": int(current_shock_streak)},
            soft_control_event=soft_control_event,
            best_epoch=best_epoch,
            best_loss=best_loss,
            best_selection_score=best_selection_score,
            best_export_epoch=best_export_epoch,
            best_export_score=best_export_score,
            phase_runtime_state=phase_runtime_state,
            global_selection_mode=global_selection_mode,
            control_anchor_checkpoint=checkpoint_role_summary["control_anchor_checkpoint"],
            training_best_checkpoint=checkpoint_role_summary["training_best_checkpoint"],
            export_best_checkpoint=checkpoint_role_summary["export_best_checkpoint"],
        )

    loss_diagnostics_path = reports_dir / f'{trainer_state["model_name"]}_loss_diagnostics.json'  # 损失诊断报告路径。
    epoch_predictions_vs_targets_path = reports_dir / f'{trainer_state["model_name"]}_epoch_predictions_vs_targets.json'  # 预测-目标快照路径。
    finite_train_losses = all(math.isfinite(float(loss)) for loss in train_epoch_losses)
    finite_val_losses = all(math.isfinite(float(loss)) for loss in val_epoch_losses)
    finite_selection_scores = all(math.isfinite(float(score)) for score in val_selection_scores)
    expected_phase_names = {
        "readout_warmup",
        "gate_alignment",
        "full_tuning",
    }
    executed_phase_names = list(dict.fromkeys(str(name) for name in epoch_phase_names))
    training_stability_audit = {
        "status": "ok" if (finite_train_losses and finite_val_losses and finite_selection_scores) else "diverged",
        "finite_train_losses": finite_train_losses,
        "finite_val_losses": finite_val_losses,
        "finite_selection_scores": finite_selection_scores,
        "best_epoch_in_range": bool(best_epoch is not None and 1 <= int(best_epoch) <= int(trainer_state["epochs"])),
        "best_epoch_matches_global_selection_rule": bool(
            best_epoch is not None
            and best_loss is not None
            and int(best_epoch)
            == max(
                (
                    int(payload.get("best_supervised_epoch") or 0)
                    for payload in phase_runtime_state.values()
                    if payload.get("best_supervised_loss") is not None
                    and math.isclose(
                        float(payload.get("best_supervised_loss")),
                        float(best_loss),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ),
                default=0,
            )
        ),
        "phase_schedule_expected": {
            "warmup_epochs": int(trainer_state["phase_schedule"]["warmup_epochs"]),
            "gate_alignment_epochs": int(trainer_state["phase_schedule"]["gate_alignment_epochs"]),
        },
        "executed_phase_names": executed_phase_names,
        "all_expected_phases_executed": expected_phase_names.issubset(set(executed_phase_names)),
        "checkpoint_path_exists": bool(best_ckpt.is_file()),
        "soft_control_mode": "buffer_cooldown_floor_rollback",
        "soft_control_note": "phase shock first enters buffer/cooldown guards, then applies lr slowdown with group-wise floors; sustained shocks can request rollback to the current phase-best checkpoint.",
        "soft_control_event_count": len(soft_control_events),
        "soft_control_buffer_epochs": int(trainer_state.get("soft_control_buffer_epochs", _SOFT_CONTROL_BUFFER_EPOCHS)),
        "soft_control_cooldown_epochs": int(trainer_state.get("soft_control_cooldown_epochs", _SOFT_CONTROL_COOLDOWN_EPOCHS)),
        "soft_control_lr_floor_scale": coerce_finite_scalar(
            trainer_state.get("soft_control_lr_floor_scale", _SOFT_CONTROL_LR_FLOOR_SCALE),
            name="soft_control_lr_floor_scale",
            min_value=0.0,
            inclusive=False,
        ),
        "soft_control_rollback_patience": int(trainer_state.get("soft_control_rollback_patience", _SOFT_CONTROL_ROLLBACK_PATIENCE)),
        "soft_control_streak_length": int(trainer_state.get("soft_control_streak_length", _SOFT_CONTROL_STREAK_LENGTH)),
        "soft_control_streak_lr_decay": coerce_finite_scalar(
            trainer_state.get("soft_control_streak_lr_decay", _SOFT_CONTROL_STREAK_LR_DECAY),
            name="soft_control_streak_lr_decay",
            min_value=0.0,
            inclusive=False,
        ),
        "max_phase_shock_streak": max([int(value) for value in phase_shock_streaks.values()], default=0),
        "global_selection_mode": global_selection_mode,
        "global_best_phase_name": global_best_phase_name,
    }
    # 审计先于 JSON 落盘：若训练发散产生 NaN/Inf，dumps_json_text(allow_nan=False) 会抛 ValueError。
    # 此时 training_stability_audit 已计算完成，可在异常处理中落盘审计结果。
    loss_diagnostics_path.write_text(dumps_json_text(diagnostics_payload), encoding="utf-8")  # 写入损失诊断。
    epoch_predictions_vs_targets_path.write_text(  # 写入预测-目标快照。
        dumps_json_text(epoch_predictions_payload),
        encoding="utf-8",
    )
    training_stability_audit["diagnostics_path_exists"] = bool(loss_diagnostics_path.is_file())
    training_stability_audit["epoch_predictions_path_exists"] = bool(epoch_predictions_vs_targets_path.is_file())
    serialized_phase_runtime_state = _serialize_phase_runtime_state(phase_runtime_state)
    phase_control_summary = _build_phase_control_summary(
        phase_runtime_state,
        current_phase_name=(str(epoch_phase_names[-1]) if epoch_phase_names else None),
        global_best_phase_name=global_best_phase_name,
    )
    last_train_epoch_snapshot = epoch_predictions_payload["train"][-1] if epoch_predictions_payload["train"] else {}
    last_val_epoch_snapshot = epoch_predictions_payload["val"][-1] if epoch_predictions_payload["val"] else {}
    checkpoint_role_summary = _build_checkpoint_role_summary(
        phase_runtime_state,
        current_phase_name=(str(epoch_phase_names[-1]) if epoch_phase_names else None),
        control_anchor_checkpoint=(
            phase_runtime_state.get(str(epoch_phase_names[-1]), {}).get("best_supervised_checkpoint_path")
            if epoch_phase_names
            else None
        ),
        training_best_checkpoint=training_best_checkpoint_path,
        export_best_checkpoint=export_best_checkpoint_path,
    )

    train_report = {
        "model_name": trainer_state["model_name"],
        "status": "trained",
        "checkpoint_format": "liquid_real_v1",
        "trainer_mode": "phase_scheduled_liquid",
        "epochs": trainer_state["epochs"],
        "train_epoch_losses": train_epoch_losses,
        "val_epoch_losses": val_epoch_losses,
        "epoch_losses": list(val_epoch_losses),
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "best_selection_score": best_selection_score,
        "best_export_epoch": best_export_epoch,
        "best_export_score": best_export_score,
        "val_selection_scores": val_selection_scores,
        "optimization_loss": _safe_finite_float(last_val_epoch_snapshot.get("optimization_loss")),
        "phase_control_score": _safe_finite_float(last_val_epoch_snapshot.get("phase_control_score")),
        "export_score": _safe_finite_float(last_val_epoch_snapshot.get("export_score")),
        "optimizer": dict(trainer_state["optimizer"]),
        "seed_report": dict(seed_report),
        # §13.6.2.8 cross-trainer parity declaration: 把模块级 _CROSS_TRAINER_PARITY_DECLARATION
        # 的副本写入 train_report, 让 train_pipeline._assert_cross_trainer_parity_declared_and_aligned
        # 与下游审计拿到 trainer 实际用的 calibration / mono / l1 / selection_weights 等数值,
        # 与 _training_flow_contract 同口径持久化, 禁止 silent drift.
        "cross_trainer_parity_declaration": deepcopy(_CROSS_TRAINER_PARITY_DECLARATION),
        "runtime_device": runtime_device,
        "amp_requested": amp_state["requested"],
        "amp_enabled": amp_state["enabled"],
        "amp_dtype": amp_state["dtype_name"],
        "train_window_count": len(train_windows),
        "val_window_count": len(val_windows),
        # §13.6.2.7 + §13.3 audit fix: train_report["checkpoint_path"] 必须指向
        # export_score 视角下的最优 checkpoint (即 export_best_checkpoint_path),
        # 与 score_role_summary["paper_checkpoint_selection_must_use"]="export_score"
        # 声明同口径, 避免下游 train_pipeline:create_model(... checkpoint_path=...)
        # 误用 supervised-loss 选出的 checkpoint 作为论文主结果. 仅当训练中断
        # (export_best_checkpoint_path 仍为初始 export_best_ckpt 且该文件尚未落盘)
        # 时回退到 training_best_checkpoint_path -> best_ckpt.
        "checkpoint_path": (
            export_best_checkpoint_path
            if (export_best_checkpoint_path and Path(export_best_checkpoint_path).is_file())
            else (training_best_checkpoint_path or str(best_ckpt))
        ),
        "epoch_candidate_paths": epoch_candidate_paths,
        "epoch_candidate_epochs": epoch_candidate_epochs,
        "phase_schedule": dict(trainer_state["phase_schedule"]),
        "phase_schedule_budget": dict(trainer_state["phase_schedule_budget"]),
        "epoch_phase_names": list(epoch_phase_names),
        "phase_trainability_contract": phase_trainability_contract,
        "phase_effective_lrs": phase_effective_lrs,
        "phase_runtime_state": serialized_phase_runtime_state,
        "phase_control_summary": phase_control_summary,
        "current_phase_control_state": phase_control_summary.get("current_phase_control_state"),
        "global_best_phase_control_state": phase_control_summary.get("global_best_phase_control_state"),
        "score_role_summary": score_role_summary,
        "checkpoint_role_summary": checkpoint_role_summary,
        "control_anchor_checkpoint": checkpoint_role_summary["control_anchor_checkpoint"],
        "training_best_checkpoint": checkpoint_role_summary["training_best_checkpoint"],
        "export_best_checkpoint": checkpoint_role_summary["export_best_checkpoint"],
        "soft_control_events": soft_control_events,
        "soft_control_event_count": len(soft_control_events),
        "phase_shock_streaks": {str(key): int(value) for key, value in phase_shock_streaks.items()},
        "max_phase_shock_streak": max([int(value) for value in phase_shock_streaks.values()], default=0),
        "global_selection_mode": global_selection_mode,
        "global_best_phase_name": global_best_phase_name,
        "optimizer_weight_decay_applied": float(trainer_state.get("optimizer", {}).get("weight_decay", 0.0)),
        "risk_calibration_enabled": getattr(model, "risk_calibration", None) is not None,
        "tail_selection_observation_coeff": coerce_finite_scalar(
            trainer_state.get("tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
            name="tail_selection_observation_coeff",
            min_value=0.0,
        ),
        "gate_alignment_readout_lr_scale": coerce_finite_scalar(
            trainer_state.get("gate_alignment_readout_lr_scale", _GATE_ALIGNMENT_READOUT_LR_SCALE),
            name="gate_alignment_readout_lr_scale",
            min_value=0.0,
            inclusive=False,
        ),
        "soft_control_streak_length": int(trainer_state.get("soft_control_streak_length", _SOFT_CONTROL_STREAK_LENGTH)),
        "soft_control_streak_lr_decay": coerce_finite_scalar(
            trainer_state.get("soft_control_streak_lr_decay", _SOFT_CONTROL_STREAK_LR_DECAY),
            name="soft_control_streak_lr_decay",
            min_value=0.0,
            inclusive=False,
        ),
        "soft_control_buffer_epochs": int(trainer_state.get("soft_control_buffer_epochs", _SOFT_CONTROL_BUFFER_EPOCHS)),
        "soft_control_cooldown_epochs": int(trainer_state.get("soft_control_cooldown_epochs", _SOFT_CONTROL_COOLDOWN_EPOCHS)),
        "soft_control_lr_floor_scale": coerce_finite_scalar(
            trainer_state.get("soft_control_lr_floor_scale", _SOFT_CONTROL_LR_FLOOR_SCALE),
            name="soft_control_lr_floor_scale",
            min_value=0.0,
            inclusive=False,
        ),
        "soft_control_rollback_patience": int(trainer_state.get("soft_control_rollback_patience", _SOFT_CONTROL_ROLLBACK_PATIENCE)),
        "history_start_epoch": int(start_epoch),
        "history_end_epoch": int(start_epoch + len(train_epoch_losses) - 1) if train_epoch_losses else int(start_epoch - 1),
        # L5 追溯字段根因修复：显式记录标签构造模式，对齐 docs/loss_function.md §实现约束 8。
        # 当前代码默认事实来自 §5.2/§5.3/§5.4 "当前代码默认实现" 段落，未来切换标签实现时同步更新。
        "risk_label_mode": RISK_LABEL_MODE_DEFAULT,  # risk 标签构造模式（当前默认 alignment proxy）。
        "uwb_scaling_label_mode": UWB_SCALING_LABEL_MODE_DEFAULT,  # uwb_scaling 标签构造模式（当前默认 teacher-free 启发式）。
        "vio_scaling_label_mode": VIO_SCALING_LABEL_MODE_DEFAULT,  # vio_scaling 标签构造模式（当前默认 teacher-free 启发式）。
        "loss_diagnostics_path": str(loss_diagnostics_path),
        "epoch_predictions_vs_targets_path": str(epoch_predictions_vs_targets_path),
        "progress_path": str(progress_path),
        "training_stability_audit": training_stability_audit,
        "output_root": str(output_root),
        "best_ckpt": training_best_checkpoint_path,
        # 准则 34：持久化 train/val loss 全程曲线 + 过拟合监控指标。
        "train_loss_curve": list(train_epoch_losses),
        "val_loss_curve": list(val_epoch_losses),
        "overfit_audit": _compute_overfit_audit(train_epoch_losses, val_epoch_losses),
    }
    train_report["report_path"] = str(report_path)  # 追加报告路径到报告字典。
    report_path.write_text(dumps_json_text(train_report), encoding="utf-8")  # 写入训练报告。
    _write_epoch_progress_snapshot(
        progress_path,
        trainer_state=trainer_state,
        history_start_epoch=start_epoch,
        epoch_index=int(trainer_state["epochs"]),
        phase_name=str(epoch_phase_names[-1]) if epoch_phase_names else "completed",
        train_loss=float(train_epoch_losses[-1]) if train_epoch_losses else 0.0,
        val_loss=float(val_epoch_losses[-1]) if val_epoch_losses else 0.0,
        selection_score=float(val_selection_scores[-1]) if val_selection_scores else 0.0,
        optimization_loss=_safe_finite_float(last_val_epoch_snapshot.get("optimization_loss")),
        phase_control_score=_safe_finite_float(last_val_epoch_snapshot.get("phase_control_score")),
        export_score=_safe_finite_float(last_val_epoch_snapshot.get("export_score")),
        train_supervised_loss=_safe_finite_float(last_train_epoch_snapshot.get("supervised_loss")),
        val_supervised_loss=_safe_finite_float(last_val_epoch_snapshot.get("supervised_loss")),
        val_auxiliary_losses=last_val_epoch_snapshot.get("auxiliary_losses"),
        val_component_losses=last_val_epoch_snapshot.get("component_losses"),
        phase_switch_shock=last_val_epoch_snapshot.get("phase_switch_shock"),
        shock_streak=last_val_epoch_snapshot.get("shock_streak"),
        soft_control_event=last_val_epoch_snapshot.get("soft_control_event"),
        best_epoch=best_epoch,
        best_loss=best_loss,
        best_selection_score=best_selection_score,
        best_export_epoch=best_export_epoch,
        best_export_score=best_export_score,
        phase_runtime_state=phase_runtime_state,
        global_selection_mode=global_selection_mode,
        control_anchor_checkpoint=checkpoint_role_summary["control_anchor_checkpoint"],
        training_best_checkpoint=checkpoint_role_summary["training_best_checkpoint"],
        export_best_checkpoint=checkpoint_role_summary["export_best_checkpoint"],
        status="completed",
    )
    # 训练完成后将模型移回 CPU，释GPU 显存，与 AGENTS.md 设备规则对齐。
    # D4：与上方设备绑定对称—绑定时 network/output_heads/risk_calibration 三、均移至
    # 训练设备，结束时必须三、均移回 CPU。仅移回 network 会令 output_heads 。
    # risk_calibration 滞留 GPU，、成显存泄漏且模型处于半迁移状。
    # 准则 27：脱钩激活统计 hook（避免影响推理路径），写入 train_report。
    if activation_hook_handles:
        from liquidloc.models.activation_stats import detach_activation_stats_hooks
        detach_activation_stats_hooks(activation_hook_handles)
    if activation_collector is not None:
        train_report["activation_stats"] = activation_collector.summary()  # 准则 27：5 头 mean/std/sparsity/saturation 写入 train_report
        from liquidloc.models.activation_stats import DEFAULT_HEAD_NAMES
        train_report["activation_stats_meta"] = {
            "head_names": list(DEFAULT_HEAD_NAMES),
            "epochs_collected": int(trainer_state.get("epochs", 0) or 0),
            "method_name": str(trainer_state.get("model_name", "unknown")),
        }
    model.network.to("cpu")
    # v3.1 head-shared: output_backbone 与 output_heads 一起移回 CPU
    if hasattr(model, "output_backbone") and hasattr(model.output_backbone, "to"):
        model.output_backbone.to("cpu")
    for head in model.output_heads.values():
        if hasattr(head, "to"):
            head.to("cpu")
    if hasattr(model, "risk_calibration") and hasattr(model.risk_calibration, "to"):
        model.risk_calibration.to("cpu")
    # §13.6.2.7 + §13.3 audit fix: 返回 checkpoint 必须是 paper main result 的
    # export_score 选模 checkpoint, 与 train_report["checkpoint_path"] 同口径.
    # 旧实现 return str(best_ckpt) 让下游 train_pipeline.create_model(...) 加载的
    # 是 supervised-loss 选出的 checkpoint, 违反 paper_checkpoint_selection_must_use
    # 声明. 修: 优先返回 export_best_ckpt (即 export_score 视角最优), 落盘失败时
    # 回退到 training_best_ckpt -> best_ckpt.
    returned_checkpoint_path = (
        export_best_checkpoint_path
        if (export_best_checkpoint_path and Path(export_best_checkpoint_path).is_file())
        else (training_best_checkpoint_path or str(best_ckpt))
    )
    return str(returned_checkpoint_path), train_report  # 返回 export_score 视角最优 checkpoint 路径和训练报告.

