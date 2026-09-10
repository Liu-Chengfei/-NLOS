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
from liquidloc.common.constants import MODEL_INTERMEDIATE_KEYS, MODALITY_UWB, MODALITY_VIO, BRIDGE_BIAS_MAX, RISK_LABEL_MODE_DEFAULT, UWB_SCALING_LABEL_MODE_DEFAULT, VIO_SCALING_LABEL_MODE_DEFAULT, MODEL_NAME_LIQUID  # 模态名单源真相，避免字面量漂移；输出头键名单源真相（D9 根因修复）；bias 上界与推理侧对齐（L1 根因修复）；标签构造模式默认值（L5 追溯字段根因修复，对齐 docs/loss_function.md §实现约束 8）；Liquid 模型名默认值。
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

# === Constants imported from _constants.py ===
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
    _GATE_ALIGNMENT_READOUT_LR_SCALE,
    _GLOBAL_L2_WEIGHT,
    _LIQUID_CALIBRATION_WEIGHT,
    _LIQUID_GATE_L1_WEIGHT,
    _LIQUID_MONO_WEIGHT,
    _OUTPUT_KEYS,
    _PHASE_TRANSITION_OVERLAP_EPOCHS,
    _RISK_CALIBRATION_BIN_COUNT,
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

# === Re-exports from submodules (for test compatibility) ===
from ._checkpoint_utils import (  # type: ignore[import,misc]
    _build_phase_checkpoint_path,
    _resolve_phase_rollback_source_name,
    _load_checkpoint_payload,
    _restore_training_snapshot,
    _try_resume_phase_scheduled_training,
    _serialize_phase_runtime_state,
    _coerce_phase_runtime_state,
    _build_phase_control_summary,
    _build_checkpoint_role_summary,
    _atomic_save_checkpoint,
    _add_config_hash_to_payload,
    _validate_checkpoint_config_hash,
    _build_score_role_summary,
    _build_export_score,
    _build_epoch_score_bundle,
    _safe_finite_float,
    _write_epoch_progress_snapshot,
)
from ._data_utils import (  # type: ignore[import,misc]
    _extract_window_and_target,
    _materialize_samples,
    _move_materialized_samples,
    _resolve_sample_cache_device,
    _resolve_current_modality,
    _resolve_feature_order,
    _clamp_unit_interval,
    _build_head_mask_tensor,
    _build_sample_weight_tensor,
    _project_train_scaling,
    _project_train_risk_batch,
    _normalize_liquid_train_outputs,
    _normalize_liquid_train_outputs_batch,
    _resolve_selection_sample_weight,
)
from ._loss_compute import (  # type: ignore[import,misc]
    _project_train_risk,
    _semantic_loss_matrix,
    _normalized_squared_error,
    _compute_loss,
    _compute_liquid_auxiliary_loss_terms,
    _predict_batch,
    _weighted_selection_score,
    _loss_for_length_group,
    _risk_calibration_trainable,
    _masked_component_loss_stats,
    _masked_supervised_loss_stats,
)
from ._diagnostics_utils import (  # type: ignore[import,misc]
    _evaluate_model,
    _collect_batch_loss_diagnostics,
    _summarize_component_losses,
    _summarize_component_losses_compact,
    _collect_epoch_prediction_target_snapshot,
)
from ._training_helpers import (  # type: ignore[import,misc]
    _build_optimizer,
    _build_phase_trainability_contract,
    _capture_control_group_lrs,
    _optimizer_effective_state,
    _resolve_amp_state,
    _resolve_epoch_phase_name,
    _resolve_full_tuning_subphase,
    _resolve_phase_epochs,
    _resolve_runtime_device,
    _build_checkpoint_payload,
    _apply_backbone_unfreeze_ramp,
    _apply_group_lr_cap,
    _apply_phase_group_lrs,
    _compute_full_tuning_ramp_epoch,
    _compute_overfit_audit,
    _configure_training_phase,
    _maybe_apply_phase_shock_soft_control,
    _to_cpu_recursive,
    _build_phase_control_state,
    _assert_phase_epoch_count_within_budget,
    train_one_epoch,
)
from ._diagnostics_utils import (  # type: ignore[import,misc]
    _evaluate_model,
    _collect_epoch_loss_diagnostics,
    _summarize_component_losses,
    _summarize_component_losses_compact,
)


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
    model_name = train_cfg.get("name", MODEL_NAME_LIQUID)  # 模型名，默认 liquid_ekf。
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
        "device": str(train_cfg.get("device") or DEVICE_CPU),
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
    # CODE-9④: 计算 config_hash 绑定到所有 checkpoint
    from liquidloc.pipelines.core_pipeline import _resolve_config_hash
    _config_hash = _resolve_config_hash(train_cfg)

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
    grad_scaler = torch.amp.GradScaler(enabled=bool(amp_state.get("enabled"))) if amp_state.get("enabled") else None  # C-3: AMP scaler
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
        config_hash=_config_hash,  # CODE-9④：校验 checkpoint config_hash 匹配
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
            amp_state=amp_state,
            grad_scaler=grad_scaler,
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
        # CODE-9④：绑定 config_hash（加载时验证）
        checkpoint_payload = _add_config_hash_to_payload(checkpoint_payload, _config_hash)
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
                # CODE-9④ 强制校验：加载的 checkpoint 必须 config_hash 匹配当前配置
                # 不匹配 → 禁止 restore（可能是用了不同配置/旧权重的 checkpoint）
                if rollback_payload is not None and not _validate_checkpoint_config_hash(
                    Path(rollback_checkpoint_path), _config_hash
                ):
                    rollback_payload = None
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
            _atomic_save_checkpoint(phase_best_ckpt, checkpoint_payload)  # CODE-9⑦ 原子写入
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
            _atomic_save_checkpoint(candidate_path, checkpoint_payload)  # CODE-9⑦ 原子写入
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
            _atomic_save_checkpoint(best_ckpt, checkpoint_payload)  # CODE-9⑦ 原子写入
            _atomic_save_checkpoint(training_best_ckpt, checkpoint_payload)  # CODE-9⑦ 原子写入
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
            _atomic_save_checkpoint(export_best_ckpt, checkpoint_payload)  # CODE-9⑦ 原子写入
            export_best_checkpoint_path = str(export_best_ckpt)
        runtime_ckpt = checkpoints_dir / f'{trainer_state["model_name"]}_resume_checkpoint.pt'
        _atomic_save_checkpoint(runtime_ckpt, checkpoint_payload)  # CODE-9⑦ 原子写入

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
    model.network.to(DEVICE_CPU)
    # v3.1 head-shared: output_backbone 与 output_heads 一起移回 CPU
    if hasattr(model, "output_backbone") and hasattr(model.output_backbone, "to"):
        model.output_backbone.to(DEVICE_CPU)
    for head in model.output_heads.values():
        if hasattr(head, "to"):
            head.to(DEVICE_CPU)
    if hasattr(model, "risk_calibration") and hasattr(model.risk_calibration, "to"):
        model.risk_calibration.to(DEVICE_CPU)
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

