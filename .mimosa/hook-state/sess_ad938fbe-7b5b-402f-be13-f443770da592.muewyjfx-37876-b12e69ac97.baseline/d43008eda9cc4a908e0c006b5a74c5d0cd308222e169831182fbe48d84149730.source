"""Checkpoint utilities submodule for Liquid trainer.

Handles checkpoint saving, loading, rollback, and state serialization.
"""
from __future__ import annotations

import json
import math
import os

from pathlib import Path
from typing import Any, Mapping

import torch

from ._constants import (  # type: ignore[import]
    _GATE_ALIGNMENT_READOUT_LR_SCALE,
    _TAIL_SELECTION_OBSERVATION_COEFF,
)
from liquidloc.common.types import ModelIntermediate
from liquidloc.common.io_utils import dumps_json_text
from liquidloc.common.validation import coerce_finite_scalar, is_integer
from liquidloc.common.constants import DEVICE_CPU  # 设备默认值常量（D9 单源真相）
from liquidloc.common.validation import coerce_finite_scalar
from liquidloc.models.liquid._loss_compute import _weighted_selection_score  # type: ignore[import]

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


def _atomic_save_checkpoint(checkpoint_path: Path, payload: dict[str, Any]) -> Path:
    """原子写入 checkpoint：写临时文件后 os.replace 到正式路径。

    CODE-9⑦ 落盘细则：
    1. 原子写入：写 `<name>.pt.tmp.<pid>` 再 os.replace 到正式路径
    2. 禁止半文件：崩溃时仅留 tmp 文件，恢复器只认完整 checkpoint
    """
    checkpoints_dir = checkpoint_path.parent
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoints_dir / f"{checkpoint_path.name}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    return checkpoint_path


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any] | None:
    """安全加载 checkpoint, 失败时返回 None。

    CODE-9⑦ 损坏自检：
    - 文件存在 + 大小 > 0 + 可反序列化 + 关键字段齐全 → 有效
    - 任一失败 → 视为损坏，禁止修复式读取
    """
    if not checkpoint_path.is_file():
        return None
    if checkpoint_path.stat().st_size == 0:
        return None
    try:
        payload = torch.load(checkpoint_path, map_location=DEVICE_CPU, weights_only=True)
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    # 必需字段：model_state (非必需，可兼容无 model_state 的旧文件) + model_name
    # config_hash 非必需：支持旧 checkpoint 加载
    if "model_name" not in payload:
        return None
    return dict(payload)


def _add_config_hash_to_payload(payload: dict[str, Any], current_config_hash: str) -> dict[str, Any]:
    """CODE-9④：为 checkpoint 绑定 config_hash，加载时校验匹配。

    不一致 → 用错配置的 checkpoint，禁止使用。
    """
    payload = dict(payload)  # 复制避免修改原地
    payload["config_hash"] = current_config_hash
    return payload


def _validate_checkpoint_config_hash(checkpoint_path: Path, expected_hash: str) -> bool:
    """加载 checkpoint 前校验 config_hash 是否匹配。"""
    payload = _load_checkpoint_payload(checkpoint_path)
    if payload is None:
        return False
    stored_hash = payload.get("config_hash", "")
    return stored_hash == expected_hash


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
    config_hash: str,  # CODE-9④：用于校验 checkpoint 配置匹配
    training_best_ckpt: Path | None = None,
    export_best_ckpt: Path | None = None,
) -> dict[str, Any] | None:
    """从当前output_root 下已有best checkpoint/train report 恢复训练状态

    C-10: 论文级实验 (deterministic=True) 禁止 resume → 从头重训（无旧权重 load）。
    若 checkpoint 仍残留（未过 RZ-0），以 None 返回，强制从头开始。
    """
    # C-10 合规：论文级实验禁止加载旧 checkpoint
    if trainer_state.get("deterministic") is True:
        return None

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
    # CODE-9④：加载 checkpoint 前必须校验 config_hash
    if not _validate_checkpoint_config_hash(source_checkpoint_path, config_hash):
        return None
    try:
        checkpoint_payload = torch.load(source_checkpoint_path, map_location=DEVICE_CPU, weights_only=True)
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

    P33 修复（三网络统一 best-val checkpoint 选择协议）：
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


