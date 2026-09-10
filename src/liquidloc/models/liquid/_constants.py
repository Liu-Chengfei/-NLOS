"""Liquid trainer constants — single source of truth for all magic numbers and lookup tables."""
from typing import Any
from liquidloc.common.constants import MODEL_INTERMEDIATE_KEYS, MODALITY_UWB, MODALITY_VIO
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
_GATE_ALIGNMENT_BACKBONE_LR_SCALE = 0.10  # Relative backbone LR scale during gate alignment phase.
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
