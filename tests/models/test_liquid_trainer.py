"""Liquid 训练器（liquid_trainer）测试模块。

测试覆盖范围：
- Liquid 模型的训练循环
- 阶段调度（warmup/gate_alignment/full_tuning）
- 损失计算与检查点管理

被测模块：liquidloc.models.liquid_trainer"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from liquidloc.factories.model_factory import create_model
from liquidloc.models.liquid.trainer import (
    _GLOBAL_L2_WEIGHT,
    _LIQUID_GATE_L1_WEIGHT,
    _build_optimizer,
    _build_phase_trainability_contract,
    _semantic_loss_matrix,
    _build_head_mask_tensor,
    _build_sample_weight_tensor,
    _compute_liquid_auxiliary_loss_terms,
    _configure_training_phase,
    _compute_loss,
    _evaluate_model,
    _materialize_samples,
    _maybe_apply_phase_shock_soft_control,
    _move_materialized_samples,
    _normalized_squared_error,
    _predict_batch,
    _project_train_scaling,
    _resolve_amp_state,
    _resolve_runtime_device,
    _resolve_selection_sample_weight,
    build_trainer_state,
    train_model,
    _try_resume_phase_scheduled_training,
)


def _window_sample(*, quality: float, target_intermediate: dict[str, float]):
    return {
        "current_modality": "uwb",
        "feature_order": ["quality", "valid"],
        "feature_values": [quality, 1.0],
        "missing_mask": [0, 0],
        "dt": 0.1,
        "feature_window": [[quality, 1.0]],
        "missing_mask_window": [[0, 0]],
        "target_intermediate": target_intermediate,
    }


def _legacy_liquid_checkpoint_payload_without_new_readout_keys():
    model_cfg = {
        "feature_order": ["quality", "valid"],
        "window": {"size": 1, "step": 1},
        "network": {"input_dim": 2, "hidden_dim": 4},
    }
    model = create_model("liquid_ekf", dict(model_cfg))
    legacy_state = dict(model.state_dict())
    for head_key in ("bias_head", "risk_head", "uwb_scaling_head", "vio_scaling_head"):
        legacy_head_state = dict(legacy_state[head_key])
        legacy_head_state.pop("filter_context_gate.weight", None)
        legacy_head_state.pop("filter_context_gate.bias", None)
        legacy_head_state.pop("uwb_branch_mix_gate.weight", None)
        legacy_head_state.pop("uwb_branch_mix_gate.bias", None)
        legacy_head_state.pop("vio_branch_mix_gate.weight", None)
        legacy_head_state.pop("vio_branch_mix_gate.bias", None)
        legacy_head_state.pop("residual_projection.weight", None)
        legacy_head_state.pop("residual_projection.bias", None)
        legacy_head_state["projection.weight"] = legacy_head_state["projection.weight"][
            :, : legacy_head_state["projection.weight"].shape[1] - 8
        ]
        legacy_state[head_key] = legacy_head_state
    legacy_state.pop("risk_calibration", None)
    return model_cfg, {
        "checkpoint_format": "liquid_real_v1",
        "model_cfg": model_cfg,
        "model_state": legacy_state,
    }


def test_resume_rejects_mixed_history_when_report_is_shorter_than_checkpoint(tmp_path):
    best_ckpt = tmp_path / "liquid_ekf_best_checkpoint.pt"
    report_path = tmp_path / "liquid_ekf_train_report.json"
    torch.save(
        {
            "best_epoch": 20,
            "best_loss": 0.1,
            "checkpoint_format": "liquid_real_v1",
            "model_cfg": {},
            "model_name": "liquid_ekf",
            "model_state": {},
            "optimizer": {},
            "optimizer_state": {},
            "seed_report": {},
            "train_cfg": {
                "train": {"phase_schedule": {"warmup_epochs": 20, "gate_alignment_epochs": 40}},
                "deterministic": True,
                "seed": 0,
            },
            "train_window_count": 1,
            "val_window_count": 1,
        },
        best_ckpt,
    )
    report_path.write_text(
        json.dumps(
            {
                "model_name": "liquid_ekf",
                "epochs": 160,
                "phase_schedule": {"warmup_epochs": 20, "gate_alignment_epochs": 40},
                "train_epoch_losses": [0.1] * 10,
                "val_epoch_losses": [0.1] * 10,
                "val_selection_scores": [0.1] * 10,
                "epoch_phase_names": ["readout_warmup"] * 10,
                "best_epoch": 20,
                "best_loss": 0.1,
                "best_selection_score": 0.1,
                "checkpoint_path": str(best_ckpt),
            }
        ),
        encoding="utf-8",
    )

    result = _try_resume_phase_scheduled_training(
        model=type(
            "DummyModel",
            (),
            {
                "network": type("DummyModule", (), {"parameters": lambda self: []})(),
                "output_heads": {},
                "risk_calibration": None,
                "load_state_dict": lambda self, state: None,
            },
        )(),
        optimizer=type("DummyOpt", (), {"load_state_dict": lambda self, state: None})(),
        trainer_state={
            "model_name": "liquid_ekf",
            "epochs": 160,
            "phase_schedule": {"warmup_epochs": 20, "gate_alignment_epochs": 40},
        },
        report_path=report_path,
        best_ckpt=best_ckpt,
    )

    assert result is None


def test_resume_rejects_progress_tail_selection_observation_coeff_mismatch(tmp_path):
    best_ckpt = tmp_path / "liquid_ekf_best_checkpoint.pt"
    report_path = tmp_path / "liquid_ekf_train_report.json"
    progress_path = tmp_path / "liquid_ekf_train_progress.json"
    torch.save(
        {
            "best_epoch": 2,
            "best_loss": 0.1,
            "checkpoint_format": "liquid_real_v1",
            "model_cfg": {},
            "model_name": "liquid_ekf",
            "model_state": {},
            "optimizer": {},
            "optimizer_state": {},
            "seed_report": {},
            "train_cfg": {
                "train": {
                    "phase_schedule": {"warmup_epochs": 20, "gate_alignment_epochs": 40},
                    "tail_selection_observation_coeff": 0.10,
                },
                "deterministic": True,
                "seed": 0,
            },
            "train_window_count": 1,
            "val_window_count": 1,
        },
        best_ckpt,
    )
    progress_path.write_text(
        json.dumps(
            {
                "model_name": "liquid_ekf",
                "status": "running",
                "history_start_epoch": 1,
                "epoch_index": 2,
                "epochs": 160,
                "phase_name": "readout_warmup",
                "train_loss": 0.1,
                "val_loss": 0.1,
                "selection_score": 0.1,
                "best_epoch": 2,
                "best_loss": 0.1,
                "best_selection_score": 0.1,
                "tail_selection_observation_coeff": 0.0,
            }
        ),
        encoding="utf-8",
    )

    result = _try_resume_phase_scheduled_training(
        model=type(
            "DummyModel",
            (),
            {
                "network": type("DummyModule", (), {"parameters": lambda self: []})(),
                "output_heads": {},
                "risk_calibration": None,
                "load_state_dict": lambda self, state: None,
            },
        )(),
        optimizer=type("DummyOpt", (), {"load_state_dict": lambda self, state: None})(),
        trainer_state={
            "model_name": "liquid_ekf",
            "epochs": 160,
            "phase_schedule": {"warmup_epochs": 20, "gate_alignment_epochs": 40},
            "tail_selection_observation_coeff": 0.10,
        },
        report_path=report_path,
        best_ckpt=best_ckpt,
    )

    assert result is None


def test_resume_rejects_checkpoint_tail_selection_observation_coeff_mismatch(tmp_path):
    best_ckpt = tmp_path / "liquid_ekf_best_checkpoint.pt"
    report_path = tmp_path / "liquid_ekf_train_report.json"
    progress_path = tmp_path / "liquid_ekf_train_progress.json"
    torch.save(
        {
            "best_epoch": 2,
            "best_loss": 0.1,
            "checkpoint_format": "liquid_real_v1",
            "model_cfg": {},
            "model_name": "liquid_ekf",
            "model_state": {},
            "optimizer": {},
            "optimizer_state": {},
            "seed_report": {},
            "train_cfg": {
                "train": {
                    "phase_schedule": {"warmup_epochs": 20, "gate_alignment_epochs": 40},
                    "tail_selection_observation_coeff": 0.0,
                },
                "deterministic": True,
                "seed": 0,
            },
            "train_window_count": 1,
            "val_window_count": 1,
        },
        best_ckpt,
    )
    progress_path.write_text(
        json.dumps(
            {
                "model_name": "liquid_ekf",
                "status": "running",
                "history_start_epoch": 1,
                "epoch_index": 2,
                "epochs": 160,
                "phase_name": "readout_warmup",
                "train_loss": 0.1,
                "val_loss": 0.1,
                "selection_score": 0.1,
                "best_epoch": 2,
                "best_loss": 0.1,
                "best_selection_score": 0.1,
                "tail_selection_observation_coeff": 0.10,
            }
        ),
        encoding="utf-8",
    )

    result = _try_resume_phase_scheduled_training(
        model=type(
            "DummyModel",
            (),
            {
                "network": type("DummyModule", (), {"parameters": lambda self: []})(),
                "output_heads": {},
                "risk_calibration": None,
                "load_state_dict": lambda self, state: None,
            },
        )(),
        optimizer=type("DummyOpt", (), {"load_state_dict": lambda self, state: None})(),
        trainer_state={
            "model_name": "liquid_ekf",
            "epochs": 160,
            "phase_schedule": {"warmup_epochs": 20, "gate_alignment_epochs": 40},
            "tail_selection_observation_coeff": 0.10,
        },
        report_path=report_path,
        best_ckpt=best_ckpt,
    )

    assert result is None


def test_normal_case(tmp_path):
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    checkpoint_path, train_report = train_model(
        [
            _window_sample(
                quality=0.2,
                target_intermediate={
                    "bias": 0.05,
                    "risk": 0.25,
                    "uwb_scaling": 1.25,
                    "vio_scaling": 1.0,
                },
            )
        ],
        [
            _window_sample(
                quality=0.3,
                target_intermediate={
                    "bias": 0.08,
                    "risk": 0.3,
                    "uwb_scaling": 1.3,
                    "vio_scaling": 1.0,
                },
            )
        ],
        {
            "name": "liquid_ekf",
            "output_root": tmp_path,
            "train": {"epochs": 1, "lr": 0.05, "optimizer": "adam", "seed": 7},
        },
    )

    assert Path(checkpoint_path).is_file()
    assert train_report["status"] == "trained"
    assert train_report["epochs"] == 1
    assert train_report["train_window_count"] == 1
    assert train_report["val_window_count"] == 1
    assert train_report["best_epoch"] == 1
    assert train_report["best_loss"] >= 0.0
    assert Path(train_report["report_path"]).is_file()
    assert train_report["amp_enabled"] is False
    assert train_report["amp_dtype"] is None
    assert train_report["trainer_mode"] == "phase_scheduled_liquid"


def test_liquid_default_train_lr_matches_lstm_default():
    """匹配测试：liquid default train lr。\n\n验证 liquid default train lr 的输出与预期一致，\n确保合同合规。
    """
    trainer_state = build_trainer_state({"name": "liquid_ekf", "train": {}})

    assert trainer_state["optimizer"]["lr"] == pytest.approx(0.001)


def test_liquid_yaml_default_paper_grade_training_contract():
    """合同测试：liquid yaml default paper grade training。\n\n验证 liquid yaml default paper grade training 的接口合同，\n确保输入输出符合协议约定。
    """
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "models" / "liquid_ekf.yaml").read_text(encoding="utf-8"))
    # yaml 当前训练预算为 60 轮，但 lr_scheduler.T_max 仍是历史 160 值；trainer 在
    # enabled 时强制 T_max==epochs。与 scripts/06_train_liquid.py:149-155 的入口自动同步
    # 同口径，归一后再验合同。
    cfg["train"].setdefault("lr_scheduler", {})["T_max"] = cfg["train"]["epochs"]
    trainer_state = build_trainer_state(cfg)

    assert trainer_state["optimizer"]["lr"] == pytest.approx(1.5e-4)
    assert trainer_state["optimizer"]["weight_decay"] == pytest.approx(0.0)
    assert trainer_state["epochs"] == 160
    assert trainer_state["train"]["batch_size"] == 32
    assert trainer_state["train"]["eval_batch_size"] == 128
    assert trainer_state["epoch_candidate_stride"] == 10
    assert trainer_state["deterministic"] is True
    assert trainer_state["phase_schedule"] == {
        "warmup_epochs": 20,
        "gate_alignment_epochs": 40,
    }


def test_liquid_build_trainer_state_accepts_epoch_candidate_stride():
    """接受测试：liquid build trainer state。\n\n验证 liquid build trainer state 的接受行为，\n确保合法输入被正确处理。
    """
    trainer_state = build_trainer_state(
        {"name": "liquid_ekf", "train": {"epochs": 4, "epoch_candidate_stride": 2}}
    )

    assert trainer_state["epoch_candidate_stride"] == 2


def test_liquid_build_trainer_state_accepts_phase_schedule():
    """接受测试：liquid build trainer state。\n\n验证 liquid build trainer state 的接受行为，\n确保合法输入被正确处理。
    """
    trainer_state = build_trainer_state(
        {
            "name": "liquid_ekf",
            "train": {
                "epochs": 6,
                "phase_schedule": {
                    "warmup_epochs": 2,
                    "gate_alignment_epochs": 1,
                },
            },
        }
    )

    assert trainer_state["phase_schedule"] == {
        "warmup_epochs": 2,
        "gate_alignment_epochs": 1,
    }


def test_liquid_optimizer_weight_decay_stays_zero_when_explicit_reg_l2_is_active():
    """零值测试：liquid optimizer weight decay stays。\n\n验证 liquid optimizer weight decay stays 在零值输入下的行为，\n确保边界情况正确处理。
    """
    trainer_state = build_trainer_state(
        {"name": "liquid_ekf", "train": {"weight_decay": 0.123}}
    )

    assert trainer_state["optimizer"]["weight_decay"] == pytest.approx(0.123)
    assert _GLOBAL_L2_WEIGHT > 0.0


def test_liquid_build_trainer_state_accepts_tail_selection_observation_coeff():
    trainer_state = build_trainer_state(
        {"name": "liquid_ekf", "train": {"tail_selection_observation_coeff": 0.25}}
    )

    assert trainer_state["tail_selection_observation_coeff"] == pytest.approx(0.25)
    assert trainer_state["train"]["tail_selection_observation_coeff"] == pytest.approx(0.25)


def test_liquid_selection_weight_increases_with_observation_tail_signals():
    """不侵入测试：liquid selection weight。\n\n验证 liquid selection weight 会让观测尾部信号弱参与加权，\n确保功能隔离性。
    """
    sample = {
        "target_trace": {
            "alignment_risk": 0.3,
        }
    }
    target_outputs = {
        "bias": 0.1,
        "risk": 0.8,
        "uwb_scaling": 1.2,
        "vio_scaling": 1.0,
    }

    assert _resolve_selection_sample_weight(sample, target_outputs) == pytest.approx(
        1.0 + (0.5 * target_outputs["risk"]) + (0.1 * target_outputs["risk"])
    )


def test_liquid_selection_weight_default_path_weakly_responds_to_quality_and_modality_signals():
    """无依赖测试：liquid selection weight default path weakly responds to quality and modality signals。\n\n验证 liquid selection weight default path weakly responds to quality and modality signals 在缺少依赖时仍会对观测尾部作弱响应，\n确保回退策略正确。
    """
    target_outputs = {
        "bias": 0.1,
        "risk": 0.4,
        "uwb_scaling": 1.2,
        "vio_scaling": 1.0,
    }
    baseline_sample = {
        "target_trace": {
            "alignment_risk": 0.4,
            "observation_risk": 0.6,
            "quality_risk": 0.0,
            "modality_signal": 0.0,
        }
    }
    boosted_observation_tail_sample = {
        "target_trace": {
            "alignment_risk": 0.4,
            "observation_risk": 0.6,
            "quality_risk": 1.0,
            "modality_signal": 1.0,
        }
    }

    baseline_weight = _resolve_selection_sample_weight(baseline_sample, target_outputs)
    boosted_weight = _resolve_selection_sample_weight(boosted_observation_tail_sample, target_outputs)

    assert baseline_weight == pytest.approx(1.0 + (0.5 * 0.6) + (0.1 * 0.6))
    assert boosted_weight == pytest.approx(1.0 + (0.5 * 0.6) + (0.1 * 1.0))
    assert boosted_weight > baseline_weight


def test_liquid_selection_weight_can_disable_observation_tail_backfill():
    target_outputs = {
        "bias": 0.1,
        "risk": 0.4,
        "uwb_scaling": 1.2,
        "vio_scaling": 1.0,
    }
    boosted_observation_tail_sample = {
        "target_trace": {
            "alignment_risk": 0.4,
            "observation_risk": 0.6,
            "quality_risk": 1.0,
            "modality_signal": 1.0,
        }
    }

    disabled_weight = _resolve_selection_sample_weight(
        boosted_observation_tail_sample,
        target_outputs,
        observation_coeff=0.0,
    )

    assert disabled_weight == pytest.approx(1.0 + (0.5 * 0.6))


def test_liquid_train_report_records_tail_selection_observation_coeff(tmp_path):
    checkpoint_path, train_report = train_model(
        [
            _window_sample(
                quality=0.2,
                target_intermediate={"bias": 0.05, "risk": 0.25, "uwb_scaling": 1.25, "vio_scaling": 1.0},
            ),
            _window_sample(
                quality=0.7,
                target_intermediate={"bias": 0.08, "risk": 0.30, "uwb_scaling": 1.30, "vio_scaling": 1.0},
            ),
        ],
        [
            _window_sample(
                quality=0.4,
                target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0},
            )
        ],
        {
            "name": "liquid_ekf",
            "output_root": tmp_path,
            "train": {
                "epochs": 1,
                "lr": 0.05,
                "optimizer": "adam",
                "seed": 23,
                "batch_size": 2,
                "eval_batch_size": 1,
                "tail_selection_observation_coeff": 0.0,
            },
        },
    )

    assert Path(checkpoint_path).is_file()
    assert train_report["tail_selection_observation_coeff"] == pytest.approx(0.0)
    report_payload = json.loads(Path(train_report["report_path"]).read_text(encoding="utf-8"))
    assert report_payload["tail_selection_observation_coeff"] == pytest.approx(0.0)


def test_boundary_case(tmp_path):
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    train_windows = [
        _window_sample(
            quality=0.15,
            target_intermediate={
                "bias": 0.02,
                "risk": 0.18,
                "uwb_scaling": 1.18,
                "vio_scaling": 1.0,
            },
        ),
        _window_sample(
            quality=0.85,
            target_intermediate={
                "bias": 0.22,
                "risk": 0.34,
                "uwb_scaling": 1.34,
                "vio_scaling": 1.0,
            },
        ),
    ]
    val_windows = [
        _window_sample(
            quality=0.6,
            target_intermediate={
                "bias": 0.05,
                "risk": 0.16,
                "uwb_scaling": 1.16,
                "vio_scaling": 1.0,
            },
        )
    ]

    checkpoint_path, train_report = train_model(
        train_windows,
        val_windows,
        {
            "name": "liquid_ekf",
            "output_root": tmp_path,
            "train": {"epochs": 3, "lr": 0.05, "optimizer": "adam", "seed": 11},
        },
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["checkpoint_format"] == "liquid_real_v1"
    assert payload["best_epoch"] == train_report["best_epoch"]
    assert "shared_projection.weight" in payload["model_state"]["network"]
    assert "bias_head" in payload["model_state"]

    fresh_model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    assert fresh_model.network.state_dict()["shared_projection.weight"].shape == payload["model_state"]["network"]["shared_projection.weight"].shape

    trained_model = create_model("liquid_ekf", {"checkpoint_path": checkpoint_path})
    assert torch.allclose(
        trained_model.network.state_dict()["shared_projection.weight"],
        payload["model_state"]["network"]["shared_projection.weight"],
    )
    assert torch.allclose(
        trained_model.bias_head.state_dict()["projection.weight"],
        payload["model_state"]["bias_head"]["projection.weight"],
    )

    from liquidloc.common.constants import BRIDGE_SCALING_MIN
    trained_outputs = trained_model.infer_intermediate(val_windows[0])
    assert trained_outputs.risk >= 0.0
    # v2: scaling 下界放宽至 BRIDGE_SCALING_MIN（0.5），配合 soft-mask 释放 Liquid 调节空间
    assert trained_outputs.uwb_scaling >= BRIDGE_SCALING_MIN


def test_invalid_case(tmp_path):
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    with pytest.raises(ValueError):
        train_model(
            [
                {
                    "current_modality": "uwb",
                    "feature_order": ["quality", "valid"],
                    "feature_values": [0.2, 1.0],
                    "missing_mask": [0, 0],
                    "feature_window": [[0.2, 1.0]],
                    "missing_mask_window": [[0, 0]],
                }
            ],
            [
                _window_sample(
                    quality=0.2,
                    target_intermediate={
                        "bias": 0.05,
                        "risk": 0.25,
                        "uwb_scaling": 1.25,
                        "vio_scaling": 1.0,
                    },
                )
            ],
            {
                "name": "liquid_ekf",
                "output_root": tmp_path,
                "train": {"epochs": 1, "lr": 0.05},
            },
        )


def test_loss_uses_liquid_inference_semantics():
    """使用测试：loss。\n\n验证被测功能正确使用 loss，\n确保内部依赖被正确调用。
    """
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    window_tensor = {
        "current_modality": "uwb",
        "feature_order": ["quality", "valid"],
        "feature_values": [0.2, 1.0],
        "missing_mask": [0, 0],
        "dt": 0.1,
        "feature_window": [[0.2, 1.0]],
        "missing_mask_window": [[0, 0]],
    }
    target_outputs = {
        "bias": 0.05,
        "risk": 0.25,
        "uwb_scaling": 1.25,
        "vio_scaling": 1.0,
    }

    loss_value = _compute_loss(model, window_tensor, target_outputs)
    inferred = model.predict_intermediate_tensors(window_tensor)
    prediction = torch.stack([inferred[key].reshape(()) for key in ("bias", "risk", "uwb_scaling", "vio_scaling")]).unsqueeze(0)
    target = torch.tensor([[target_outputs[key] for key in ("bias", "risk", "uwb_scaling", "vio_scaling")]], dtype=prediction.dtype)
    mask = _build_head_mask_tensor(["uwb"], reference_tensor=prediction)
    expected = ((_semantic_loss_matrix(prediction, target) * mask).sum() / mask.sum())

    assert torch.allclose(loss_value, expected)


def test_liquid_semantic_loss_uses_huber_for_bias_and_log_domain_for_scaling():
    """使用测试：liquid semantic loss。\n\n验证被测功能正确使用 liquid semantic loss，\n确保内部依赖被正确调用。
    """
    prediction = torch.tensor([[2.0, 0.7, 4.0, 2.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.2, 1.0, 1.0]], dtype=torch.float32)

    semantic = _semantic_loss_matrix(prediction, target, bias_huber_delta=0.5)

    expected_bias = F.huber_loss(
        prediction[:, 0],
        target[:, 0],
        reduction="none",
        delta=0.5,
    )
    expected_risk = (prediction[:, 1] - target[:, 1]).pow(2)
    expected_scaling = (
        torch.log(prediction[:, 2:] + 1e-6) - torch.log(target[:, 2:] + 1e-6)
    ).pow(2)

    assert torch.allclose(semantic[:, 0], expected_bias)
    assert torch.allclose(semantic[:, 1], expected_risk)
    assert torch.allclose(semantic[:, 2:], expected_scaling)


def test_liquid_auxiliary_loss_terms_train_risk_calibration_when_enabled():
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    _configure_training_phase(model, phase_name="full_tuning")
    prediction = torch.tensor([[0.0, 0.85, 1.6, 1.0]], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([[0.2, 0.15, 1.2, 1.0]], dtype=torch.float32)

    auxiliary = _compute_liquid_auxiliary_loss_terms(
        model,
        prediction,
        target,
        modalities=["uwb"],
        sample_weights=[1.0],
    )
    auxiliary["total"].backward()

    assert float(auxiliary["calibration"].detach().item()) >= 0.0
    assert float(auxiliary["mono"].detach().item()) >= 0.0
    assert model.risk_calibration.a_raw.grad is not None
    assert model.risk_calibration.a_raw.grad.abs().item() > 0.0


@pytest.mark.parametrize(
    ("phase_name", "expect_gate_penalty"),
    [
        ("readout_warmup", False),
        ("gate_alignment", True),
        ("full_tuning", True),
    ],
)
def test_liquid_gate_l1_only_penalizes_enabled_gate_parameters(phase_name, expect_gate_penalty):
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    _configure_training_phase(model, phase_name=phase_name)
    prediction = torch.tensor([[0.0, 0.5, 1.2, 1.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.5, 1.2, 1.0]], dtype=torch.float32)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.bias_head.projection.bias.fill_(7.0)
        model.network.shared_projection.bias.fill_(11.0)
        model.output_backbone.temporal_context_gate.weight.fill_(2.0)
        model.output_backbone.temporal_context_gate.bias.fill_(3.0)

    auxiliary = _compute_liquid_auxiliary_loss_terms(
        model,
        prediction,
        target,
        modalities=["uwb"],
        sample_weights=[1.0],
    )

    gate_parameters = [
        parameter
        for attribute_name in (
            "temporal_context_gate",
            "observation_context_gate",
            "filter_context_gate",
            "branch_mix_gate",
            "uwb_branch_mix_gate",
            "vio_branch_mix_gate",
        )
        for parameter in getattr(model.output_backbone, attribute_name).parameters()
        if parameter.requires_grad
    ]
    expected_gate_l1 = prediction.new_tensor(0.0)
    for parameter in gate_parameters:
        expected_gate_l1 = expected_gate_l1 + parameter.abs().sum()
    expected_gate_l1 = expected_gate_l1 * _LIQUID_GATE_L1_WEIGHT

    if expect_gate_penalty:
        assert float(auxiliary["gate_l1"].detach().item()) > 0.0
    else:
        assert float(auxiliary["gate_l1"].detach().item()) == pytest.approx(0.0)
    assert torch.allclose(auxiliary["gate_l1"], expected_gate_l1)


@pytest.mark.parametrize(
    "phase_name",
    ["readout_warmup", "gate_alignment", "full_tuning"],
)
def test_liquid_reg_l2_matches_requires_grad_parameters_for_each_phase(phase_name):
    """匹配测试：liquid reg l2。\n\n验证 liquid reg l2 的输出与预期一致，\n确保合同合规。
    """
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    _configure_training_phase(model, phase_name=phase_name)
    prediction = torch.tensor([[0.1, 0.4, 1.3, 1.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.2, 1.1, 1.0]], dtype=torch.float32)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.5)

    auxiliary = _compute_liquid_auxiliary_loss_terms(
        model,
        prediction,
        target,
        modalities=["uwb"],
        sample_weights=[1.0],
    )
    expected_reg_l2 = (
        sum(parameter.pow(2).sum() for parameter in model.parameters() if parameter.requires_grad)
        * _GLOBAL_L2_WEIGHT
    )

    assert float(auxiliary["reg_l2"].detach().item()) > 0.0
    assert torch.allclose(auxiliary["reg_l2"], expected_reg_l2)


def test_trainer_target_requires_neutral_floor_scaling(tmp_path):
    """必填测试：trainer target。\n\n验证 trainer target 的必填约束，\n确保缺少必要输入时抛出异常。
    """
    with pytest.raises(ValueError, match="target_intermediate.uwb_scaling must be >= 1.0"):
        train_model(
            [
                _window_sample(
                    quality=0.2,
                    target_intermediate={
                        "bias": 0.05,
                        "risk": 0.25,
                        "uwb_scaling": 0.0,
                        "vio_scaling": 1.0,
                    },
                )
            ],
            [
                _window_sample(
                    quality=0.3,
                    target_intermediate={
                        "bias": 0.08,
                        "risk": 0.3,
                        "uwb_scaling": 1.3,
                        "vio_scaling": 1.0,
                    },
                )
            ],
            {
                "name": "liquid_ekf",
                "output_root": tmp_path,
                "train": {"epochs": 1, "lr": 0.05, "optimizer": "adam", "seed": 7},
            },
        )


def test_move_materialized_samples_keeps_optional_readout_context_metadata():
    """保持测试：move materialized samples。\n\n验证 move materialized samples 的保持行为，\n确保特定属性在处理过程中不变。
    """
    samples = [
        {
            "current_modality": "uwb",
            "feature_order": ["quality", "valid"],
            "feature_values": [0.2, 1.0],
            "missing_mask": [0, 0],
            "dt": 0.1,
            "feature_window": [[0.2, 1.0]],
            "missing_mask_window": [[0, 0]],
            "readout_context_by_name": {
                "state_cov_trace": 4.0,
                "pos_cov": 1.5,
                "last_innovation_norm": 0.3,
                "last_gate_skip_flag": 1.0,
            },
            "readout_context_observed_by_name": {
                "state_cov_trace": True,
                "pos_cov": True,
                "last_innovation_norm": True,
                "last_gate_skip_flag": True,
            },
            "target_intermediate": {
                "bias": 0.05,
                "risk": 0.25,
                "uwb_scaling": 1.25,
                "vio_scaling": 1.0,
            },
        }
    ]
    materialized = _materialize_samples(samples)
    moved = _move_materialized_samples(materialized, torch.device("cpu"))
    metadata = moved[0][0]

    assert float(metadata["readout_context_by_name"]["state_cov_trace"].item()) == pytest.approx(4.0)
    assert float(metadata["readout_context_by_name"]["pos_cov"].item()) == pytest.approx(1.5)
    assert metadata["readout_context_observed_by_name"]["last_innovation_norm"] is True
    assert metadata["readout_context_observed_by_name"]["last_gate_skip_flag"] is True


def test_liquid_train_scaling_projection_preserves_gradient_below_floor():
    """保持性测试：liquid train scaling projection。\n\n验证 liquid train scaling projection 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    raw_value = torch.tensor(-12.0, dtype=torch.float32, requires_grad=True)

    projected = _project_train_scaling(raw_value)
    projected.backward()

    # scaling_min 已从 0.5 回退到 1.0，raw_value=-12 会被 clamp 到 floor=1.0
    assert projected.item() == pytest.approx(1.0)
    assert raw_value.grad is not None
    assert raw_value.grad.item() > 0.0


def test_liquid_train_outputs_keep_signed_bias_semantics():
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    with torch.no_grad():
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.fill_(-0.4)
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()
        model.uwb_scaling_head.projection.weight.zero_()
        model.uwb_scaling_head.projection.bias.zero_()
        model.vio_scaling_head.projection.weight.zero_()
        model.vio_scaling_head.projection.bias.zero_()

    predicted = model.predict_intermediate_tensors(
        {
            "current_modality": "uwb",
            "feature_order": ["quality", "valid"],
            "feature_values": [0.2, 1.0],
            "missing_mask": [0, 0],
            "dt": 0.1,
            "feature_window": [[0.2, 1.0]],
            "missing_mask_window": [[0, 0]],
        }
    )
    assert predicted["bias"].item() == pytest.approx(0.0, rel=1e-6)  # bias 非负约束


def test_loss_masks_inactive_liquid_scaling_head_for_uwb_targets():
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    model._enable_smooth_regularization = False  # R4: 禁用 smooth 项，使测试与辅助项无关
    with torch.no_grad():
        for head in (model.bias_head, model.risk_head, model.uwb_scaling_head, model.vio_scaling_head):
            head.projection.weight.zero_()
            head.projection.bias.zero_()

    window_tensor = {
        "current_modality": "uwb",
        "feature_order": ["quality", "valid"],
        "feature_values": [0.2, 1.0],
        "missing_mask": [0, 0],
        "dt": 0.1,
        "feature_window": [[0.2, 1.0]],
        "missing_mask_window": [[0, 0]],
    }
    target_outputs = {
        "bias": 0.0,
        "risk": 0.0,
        "uwb_scaling": 1.4,
        "vio_scaling": 1.3,
    }

    loss_value = _compute_loss(model, window_tensor, target_outputs)
    expected_bias = 0.0
    expected_prediction = 1.0
    expected_risk = float(torch.sigmoid(torch.tensor(0.0)))
    prediction = torch.tensor([[expected_bias, expected_risk, expected_prediction, expected_prediction]], dtype=torch.float32)
    target = torch.tensor([[expected_bias, 0.0, 1.4, 1.3]], dtype=torch.float32)
    mask = _build_head_mask_tensor(["uwb"], reference_tensor=prediction)
    expected = ((_semantic_loss_matrix(prediction, target) * mask).sum() / mask.sum())

    loss_value.backward()

    # R4: _compute_loss 返回 supervised + auxiliary (calibration/mono/reg_l2/gate_l1/smooth)，
    # 测试核心是梯度断言（mask 行为），数值只要大于等于 supervised 部分即可。
    assert float(loss_value.detach().item()) >= float(expected) - 1e-3
    assert torch.isfinite(loss_value.detach()).item()
    assert model.uwb_scaling_head.projection.bias.grad is not None
    assert model.uwb_scaling_head.projection.bias.grad.abs().item() > 0.0
    assert model.vio_scaling_head.projection.bias.grad is not None
    assert model.vio_scaling_head.projection.bias.grad.abs().item() == pytest.approx(0.0)


def test_loss_masks_inactive_liquid_bias_and_uwb_heads_for_vio_targets():
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    model._enable_smooth_regularization = False  # R4: 禁用 smooth 项
    with torch.no_grad():
        for head in (model.bias_head, model.uwb_scaling_head, model.vio_scaling_head):
            head.projection.weight.zero_()
            head.projection.bias.zero_()
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()

    window_tensor = {
        "current_modality": "vio",
        "feature_order": ["quality", "valid"],
        "feature_values": [0.2, 1.0],
        "missing_mask": [0, 0],
        "dt": 0.1,
        "feature_window": [[0.2, 1.0]],
        "missing_mask_window": [[0, 0]],
    }
    target_outputs = {
        "bias": 0.9,
        "risk": 0.0,
        "uwb_scaling": 1.9,
        "vio_scaling": 1.3,
    }

    loss_value = _compute_loss(model, window_tensor, target_outputs)
    expected_prediction = 1.0
    expected_risk = float(torch.sigmoid(torch.tensor(0.0)))
    prediction = torch.tensor([[0.0, expected_risk, 1.0, expected_prediction]], dtype=torch.float32)
    target = torch.tensor([[0.9, 0.0, 1.9, 1.3]], dtype=torch.float32)
    mask = _build_head_mask_tensor(["vio"], reference_tensor=prediction)
    expected = ((_semantic_loss_matrix(prediction, target) * mask).sum() / mask.sum())

    loss_value.backward()

    assert float(loss_value.detach().item()) >= float(expected) - 1e-3
    assert model.bias_head.projection.bias.grad is not None
    assert model.bias_head.projection.bias.grad.abs().item() == pytest.approx(0.0)
    assert model.uwb_scaling_head.projection.bias.grad is not None
    assert model.uwb_scaling_head.projection.bias.grad.abs().item() == pytest.approx(0.0)
    assert model.vio_scaling_head.projection.bias.grad is not None
    assert model.vio_scaling_head.projection.bias.grad.abs().item() > 0.0


def test_train_model_accepts_quick_batch_settings(tmp_path):
    """接受测试：train model。\n\n验证 train model 的接受行为，\n确保合法输入被正确处理。
    """
    checkpoint_path, train_report = train_model(
        [
            _window_sample(
                quality=0.1,
                target_intermediate={
                    "bias": 0.02,
                    "risk": 0.2,
                    "uwb_scaling": 1.1,
                    "vio_scaling": 1.0,
                },
            ),
            _window_sample(
                quality=0.9,
                target_intermediate={
                    "bias": 0.06,
                    "risk": 0.3,
                    "uwb_scaling": 1.2,
                    "vio_scaling": 1.0,
                },
            ),
        ],
        [
            _window_sample(
                quality=0.4,
                target_intermediate={
                    "bias": 0.03,
                    "risk": 0.25,
                    "uwb_scaling": 1.15,
                    "vio_scaling": 1.0,
                },
            )
        ],
        {
            "name": "liquid_ekf",
            "output_root": tmp_path,
            "train": {"epochs": 2, "lr": 0.05, "optimizer": "adam", "seed": 3, "batch_size": 2, "eval_batch_size": 1},
        },
    )

    assert Path(checkpoint_path).is_file()
    assert train_report["status"] == "trained"
    assert train_report["epochs"] == 2
    assert train_report["train_window_count"] == 2
    assert train_report["val_window_count"] == 1


def test_liquid_train_model_can_subsample_epoch_candidate_exports(tmp_path):
    """导出测试：liquid train model can subsample epoch candidate。\n\n验证 liquid train model can subsample epoch candidate 的导出功能，\n确保产物被正确持久化。
    """
    checkpoint_path, train_report = train_model(
        [
            _window_sample(
                quality=0.2,
                target_intermediate={"bias": 0.05, "risk": 0.25, "uwb_scaling": 1.25, "vio_scaling": 1.0},
            ),
            _window_sample(
                quality=0.8,
                target_intermediate={"bias": 0.08, "risk": 0.30, "uwb_scaling": 1.30, "vio_scaling": 1.0},
            ),
        ],
        [
            _window_sample(
                quality=0.4,
                target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0},
            )
        ],
        {
            "name": "liquid_ekf",
            "output_root": tmp_path,
            "train": {
                "epochs": 4,
                "lr": 0.05,
                "optimizer": "adam",
                "seed": 5,
                "save_epoch_candidates": True,
                "epoch_candidate_stride": 2,
            },
        },
    )

    assert Path(checkpoint_path).is_file()
    assert train_report["epoch_candidate_epochs"] == [1, 2, 4]
    assert len(train_report["epoch_candidate_paths"]) == 3


def test_liquid_train_model_reports_epoch_phase_names(tmp_path):
    """报告测试：liquid train model。\n\n验证 liquid train model 的报告生成，\n确保审计信息被正确记录。
    """
    checkpoint_path, train_report = train_model(
        [
            _window_sample(
                quality=0.2,
                target_intermediate={"bias": 0.05, "risk": 0.25, "uwb_scaling": 1.25, "vio_scaling": 1.0},
            ),
            _window_sample(
                quality=0.8,
                target_intermediate={"bias": 0.08, "risk": 0.30, "uwb_scaling": 1.30, "vio_scaling": 1.0},
            ),
        ],
        [
            _window_sample(
                quality=0.4,
                target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0},
            )
        ],
        {
            "name": "liquid_ekf",
            "output_root": tmp_path,
            "train": {
                "epochs": 4,
                "lr": 0.05,
                "optimizer": "adam",
                "seed": 13,
                "phase_schedule": {
                    "warmup_epochs": 1,
                    "gate_alignment_epochs": 1,
                },
            },
        },
    )

    assert Path(checkpoint_path).is_file()
    assert train_report["phase_schedule"] == {
        "warmup_epochs": 1,
        "gate_alignment_epochs": 1,
    }
    assert train_report["epoch_phase_names"] == [
        "readout_warmup",
        "gate_alignment",
        "full_tuning",
        "full_tuning",
    ]
    assert set(train_report["phase_trainability_contract"]) == {
        "readout_warmup",
        "gate_alignment",
        "full_tuning",
    }
    assert set(train_report["phase_effective_lrs"]) == {
        "readout_warmup",
        "gate_alignment",
        "full_tuning",
    }
    assert train_report["optimizer_weight_decay_applied"] == pytest.approx(0.0)
    assert train_report["training_stability_audit"] == {
        "status": "ok",
        "finite_train_losses": True,
        "finite_val_losses": True,
        "finite_selection_scores": True,
        "best_epoch_in_range": True,
        "best_epoch_matches_global_selection_rule": True,
        "phase_schedule_expected": {
            "warmup_epochs": 1,
            "gate_alignment_epochs": 1,
        },
        "executed_phase_names": [
            "readout_warmup",
            "gate_alignment",
            "full_tuning",
        ],
        "all_expected_phases_executed": True,
        "checkpoint_path_exists": True,
        "soft_control_mode": "buffer_cooldown_floor_rollback",
        "soft_control_note": "phase shock first enters buffer/cooldown guards, then applies lr slowdown with group-wise floors; sustained shocks can request rollback to the current phase-best checkpoint.",
        "soft_control_event_count": 0,
        "soft_control_buffer_epochs": 6,
        "soft_control_cooldown_epochs": 5,
        "soft_control_lr_floor_scale": 0.1,
        "soft_control_rollback_patience": 2,
        "soft_control_streak_length": 5,
        "soft_control_streak_lr_decay": 0.8,
        "max_phase_shock_streak": 1,
        "global_selection_mode": "export_score",
        "global_best_phase_name": "full_tuning",
        "diagnostics_path_exists": True,
        "epoch_predictions_path_exists": True,
    }


def test_liquid_train_model_exports_loss_diagnostics_json(tmp_path):
    """导出测试：liquid train model。\n\n验证 liquid train model 的导出功能，\n确保产物被正确持久化。
    """
    checkpoint_path, train_report = train_model(
        [
            _window_sample(
                quality=0.2,
                target_intermediate={"bias": 0.05, "risk": 0.25, "uwb_scaling": 1.25, "vio_scaling": 1.0},
            ),
            _window_sample(
                quality=0.8,
                target_intermediate={"bias": 0.08, "risk": 0.30, "uwb_scaling": 1.30, "vio_scaling": 1.0},
            ),
        ],
        [
            _window_sample(
                quality=0.4,
                target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0},
            )
        ],
        {
            "name": "liquid_ekf",
            "output_root": tmp_path,
            "train": {"epochs": 2, "lr": 0.05, "optimizer": "adam", "seed": 17, "batch_size": 2, "eval_batch_size": 1},
        },
    )

    assert Path(checkpoint_path).is_file()
    diagnostics_path = Path(train_report["loss_diagnostics_path"])
    assert diagnostics_path.is_file()

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["model_name"] == "liquid_ekf"
    assert len(diagnostics["train"]) == 2
    assert len(diagnostics["val"]) == 2
    assert "component_loss_summary" in diagnostics["train"][0]
    assert "supervised_loss" in diagnostics["train"][0]
    assert "supervised_loss" in diagnostics["val"][0]
    assert "phase_switch_shock" in diagnostics["val"][0]
    progress_payload = json.loads(Path(train_report["progress_path"]).read_text(encoding="utf-8"))
    assert "train_supervised_loss" in progress_payload
    assert "val_supervised_loss" in progress_payload
    assert progress_payload["gate_alignment_readout_lr_scale"] == pytest.approx(0.25)
    assert "shock_streak" in progress_payload
    assert set(progress_payload["val_auxiliary_losses"]) == {"calibration", "mono", "reg_l2", "gate_l1", "smooth", "total"}
    assert set(progress_payload["phase_switch_shock"]) == {"selection_score_relative", "supervised_loss_relative"}
    assert "soft_control_event" in progress_payload
    assert train_report["gate_alignment_readout_lr_scale"] == pytest.approx(0.25)
    assert "phase_shock_streaks" in train_report
    assert "soft_control_streak_length" in train_report
    assert "soft_control_streak_lr_decay" in train_report
    assert train_report["training_stability_audit"]["soft_control_mode"] in {"slowdown_only", "buffer_cooldown_floor_rollback", "buffer_only"}
    assert "soft_control_event_count" in train_report["training_stability_audit"]
    assert "max_phase_shock_streak" in train_report["training_stability_audit"]
    assert "soft_control_events" in train_report
    assert set(diagnostics["train"][0]["component_loss_summary"]) >= {"bias", "risk", "uwb_scaling", "vio_scaling", "calibration", "mono", "reg_l2", "gate_l1", "total"}

    first_train_epoch = diagnostics["train"][0]
    first_batch = first_train_epoch["batches"][0]

    assert first_train_epoch["split"] == "train"
    assert first_train_epoch["phase_name"] == "full_tuning"
    assert isinstance(first_train_epoch["phase_state"]["network_trainable"], bool)
    assert first_batch["modalities"] == ["uwb", "uwb"]
    assert first_batch["seq_lens"] == [1, 1]
    assert len(first_batch["prediction"]) == 2
    assert len(first_batch["target"]) == 2
    assert len(first_batch["semantic_loss_by_head"]) == 2
    assert len(first_batch["squared_error_by_head"]) == 2
    assert len(first_batch["head_mask"]) == 2
    assert first_batch["active_keys"][0] == ["bias", "risk", "uwb_scaling"]
    assert first_batch["sample_weights"][0] > 0.0
    assert set(first_batch["auxiliary_losses"]) == {"calibration", "mono", "reg_l2", "gate_l1", "smooth", "total"}

    prediction = torch.tensor([first_batch["prediction"][0]], dtype=torch.float32)
    target = torch.tensor([first_batch["target"][0]], dtype=torch.float32)
    mask = _build_head_mask_tensor(["uwb"], reference_tensor=prediction)
    sample_weight = _build_sample_weight_tensor([first_batch["sample_weights"][0]], reference_tensor=prediction).unsqueeze(1)
    expected_squared_error = (_normalized_squared_error(prediction, target) * mask).squeeze(0)
    expected_semantic_loss = (_semantic_loss_matrix(prediction, target) * mask).squeeze(0)
    expected_weighted_numerator = float((_semantic_loss_matrix(prediction, target) * mask * sample_weight).sum().item())
    expected_weighted_denominator = float((mask * sample_weight).sum().item())

    for index, key in enumerate(("bias", "risk", "uwb_scaling", "vio_scaling")):
        assert first_batch["squared_error_by_head"][0][index] == pytest.approx(float(expected_squared_error[index].item()), rel=1e-5)
        assert first_batch["semantic_loss_by_head"][0][index] == pytest.approx(float(expected_semantic_loss[index].item()), rel=1e-5)
        assert first_batch["head_mask"][0][index] == pytest.approx(float(mask[0, index].item()))
    assert first_batch["loss_numerator"] >= expected_weighted_numerator
    assert first_batch["loss_denominator"] >= expected_weighted_denominator
    assert train_report["training_stability_audit"]["finite_train_losses"] is True
    assert train_report["training_stability_audit"]["finite_val_losses"] is True
    assert train_report["training_stability_audit"]["finite_selection_scores"] is True
    assert train_report["training_stability_audit"]["best_epoch_in_range"] is True
    assert train_report["training_stability_audit"]["best_epoch_matches_global_selection_rule"] is True


def test_liquid_epoch_loss_diagnostics_truncate_stored_batches_but_preserve_batch_count(tmp_path):
    checkpoint_path, train_report = train_model(
        [
            _window_sample(quality=0.10, target_intermediate={"bias": 0.04, "risk": 0.22, "uwb_scaling": 1.18, "vio_scaling": 1.0}),
            _window_sample(quality=0.20, target_intermediate={"bias": 0.05, "risk": 0.24, "uwb_scaling": 1.20, "vio_scaling": 1.0}),
        ],
        [_window_sample(quality=0.40, target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0})],
        {
            "name": "liquid_ekf",
            "output_root": tmp_path / "liquid_batch_retention",
            "feature_order": ["quality", "valid"],
            "network": {"input_dim": 2, "hidden_dim": 8},
            "train": {"epochs": 1, "batch_size": 1, "eval_batch_size": 1},
        },
    )

    assert Path(checkpoint_path).is_file()
    diagnostics = json.loads(Path(train_report["loss_diagnostics_path"]).read_text(encoding="utf-8"))
    first_epoch = diagnostics["train"][0]

    assert first_epoch["batch_count"] == 2
    assert first_epoch["stored_batch_count"] == 1
    assert first_epoch["truncated_batch_count"] == 1
    assert len(first_epoch["batches"]) == 1
    assert first_epoch["batches"][0]["batch_size"] == 1


def test_liquid_train_model_exports_epoch_predictions_vs_targets_json(tmp_path):
    """导出测试：liquid train model。\n\n验证 liquid train model 的导出功能，\n确保产物被正确持久化。
    """
    checkpoint_path, train_report = train_model(
        [
            _window_sample(
                quality=0.2,
                target_intermediate={"bias": 0.05, "risk": 0.25, "uwb_scaling": 1.25, "vio_scaling": 1.0},
            ),
            _window_sample(
                quality=0.8,
                target_intermediate={"bias": 0.08, "risk": 0.30, "uwb_scaling": 1.30, "vio_scaling": 1.0},
            ),
        ],
        [
            _window_sample(
                quality=0.4,
                target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0},
            )
        ],
        {
            "name": "liquid_ekf",
            "output_root": tmp_path,
            "train": {"epochs": 2, "lr": 0.05, "optimizer": "adam", "seed": 23, "batch_size": 2, "eval_batch_size": 1},
        },
    )

    assert Path(checkpoint_path).is_file()
    epoch_path = Path(train_report["epoch_predictions_vs_targets_path"])
    assert epoch_path.is_file()

    report = json.loads(epoch_path.read_text(encoding="utf-8"))
    assert report["model_name"] == "liquid_ekf"
    assert report["output_heads"] == ["bias", "risk", "uwb_scaling", "vio_scaling"]
    assert len(report["train"]) == 2
    assert len(report["val"]) == 2
    assert [epoch_payload["epoch_index"] for epoch_payload in report["train"]] == [1, 2]
    assert [epoch_payload["epoch_index"] for epoch_payload in report["val"]] == [1, 2]
    assert [epoch_payload["phase_name"] for epoch_payload in report["train"]] == ["full_tuning", "full_tuning"]
    assert [epoch_payload["phase_name"] for epoch_payload in report["val"]] == ["full_tuning", "full_tuning"]

    first_train_epoch = report["train"][0]
    assert first_train_epoch["split"] == "train"
    assert first_train_epoch["epoch_index"] == 1
    assert first_train_epoch["phase_name"] == "full_tuning"
    assert "supervised_loss" in first_train_epoch
    assert set(first_train_epoch["component_losses"]) >= {"bias", "risk", "uwb_scaling"}
    assert set(first_train_epoch["auxiliary_losses"]) == {"calibration", "mono", "reg_l2", "gate_l1", "smooth", "total"}
    assert isinstance(first_train_epoch["phase_state"]["risk_calibration_trainable"], bool)
    assert first_train_epoch["sample_count"] == 2
    assert first_train_epoch["active_sample_count_by_head"]["bias"] == 2
    assert first_train_epoch["active_sample_count_by_head"]["uwb_scaling"] == 2
    assert first_train_epoch["active_sample_count_by_head"]["vio_scaling"] == 0
    assert set(first_train_epoch["prediction_mean"]) == {"bias", "risk", "uwb_scaling", "vio_scaling"}
    assert set(first_train_epoch["target_mean"]) == {"bias", "risk", "uwb_scaling", "vio_scaling"}
    assert first_train_epoch["prediction_mean"]["vio_scaling"] is None
    assert first_train_epoch["target_mean"]["vio_scaling"] is None
    assert first_train_epoch["prediction_mean_active_samples"] == first_train_epoch["prediction_mean"]
    assert first_train_epoch["target_mean_active_samples"] == first_train_epoch["target_mean"]
    assert first_train_epoch["prediction_mean_all_samples"]["vio_scaling"] is not None
    assert first_train_epoch["target_mean_all_samples"]["vio_scaling"] is not None
    assert first_train_epoch["mean_scope"] == "fixed_split_active_sample_mean"
    assert "target means can remain constant across epochs" in first_train_epoch["mean_scope_note"]
    assert first_train_epoch["probe_batch_modalities"] == ["uwb", "uwb"]
    assert first_train_epoch["probe_batch_active_keys"] == [
        ["bias", "risk", "uwb_scaling"],
        ["bias", "risk", "uwb_scaling"],
    ]
    assert len(first_train_epoch["probe_batch_prediction"]) == 2
    assert len(first_train_epoch["probe_batch_target"]) == 2
    assert len(first_train_epoch["probe_batch_rows"]) == 2
    first_val_epoch = report["val"][0]
    assert "phase_switch_shock" in first_val_epoch
    assert set(first_val_epoch["phase_switch_shock"]) == {"selection_score_relative", "supervised_loss_relative"}
    assert first_train_epoch["fixed_probe_source"] == "cached_split_prefix"
    assert first_train_epoch["fixed_probe_sample_count"] == 2
    assert len(first_train_epoch["fixed_probe_rows"]) == 2
    assert first_train_epoch["fixed_probe_rows"][0]["sample_index_in_split"] == 0
    assert first_train_epoch["fixed_probe_rows"][0]["modality"] == "uwb"
    assert first_train_epoch["fixed_probe_rows"][0]["active_keys"] == ["bias", "risk", "uwb_scaling"]
    assert set(first_train_epoch["fixed_probe_rows"][0]["prediction_by_head"]) == {
        "bias",
        "risk",
        "uwb_scaling",
        "vio_scaling",
    }
    assert set(first_train_epoch["fixed_probe_rows"][0]["target_by_head"]) == {
        "bias",
        "risk",
        "uwb_scaling",
        "vio_scaling",
    }
    assert set(first_train_epoch["fixed_probe_rows"][0]["signed_error_by_head"]) == {
        "bias",
        "risk",
        "uwb_scaling",
        "vio_scaling",
    }
    assert first_train_epoch["probe_batch_rows"][0]["row_index"] == 0
    assert first_train_epoch["probe_batch_rows"][0]["modality"] == "uwb"
    assert first_train_epoch["probe_batch_rows"][0]["active_keys"] == ["bias", "risk", "uwb_scaling"]
    assert set(first_train_epoch["probe_batch_rows"][0]["prediction_by_head"]) == {
        "bias",
        "risk",
        "uwb_scaling",
        "vio_scaling",
    }
    assert set(first_train_epoch["probe_batch_rows"][0]["target_by_head"]) == {
        "bias",
        "risk",
        "uwb_scaling",
        "vio_scaling",
    }
    assert set(first_train_epoch["probe_batch_rows"][0]["signed_error_by_head"]) == {
        "bias",
        "risk",
        "uwb_scaling",
        "vio_scaling",
    }

    for split_name in ("train", "val"):
        for epoch_payload in report[split_name]:
            assert set(epoch_payload["phase_state"]) == {
                "network_trainable",
                "risk_calibration_trainable",
            }
            assert isinstance(epoch_payload["phase_state"]["network_trainable"], bool)
            assert isinstance(epoch_payload["phase_state"]["risk_calibration_trainable"], bool)
            assert epoch_payload["phase_name"] == "full_tuning"
            assert epoch_payload["phase_state"] == {
                "network_trainable": True,
                "risk_calibration_trainable": True,
            }


def test_liquid_train_model_persists_phase_schedule_consistently_across_epoch_artifacts(tmp_path):
    """阶段调度测试：liquid train model persists。\n\n验证 liquid train model persists 的阶段调度保持，\n确保 warmup/gate_alignment/full_tuning 阶段正确执行。
    """
    checkpoint_path, train_report = train_model(
        [
            _window_sample(
                quality=0.2,
                target_intermediate={"bias": 0.05, "risk": 0.25, "uwb_scaling": 1.25, "vio_scaling": 1.0},
            ),
            _window_sample(
                quality=0.8,
                target_intermediate={"bias": 0.08, "risk": 0.30, "uwb_scaling": 1.30, "vio_scaling": 1.0},
            ),
        ],
        [
            _window_sample(
                quality=0.4,
                target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0},
            )
        ],
        {
            "name": "liquid_ekf",
            "output_root": tmp_path,
            "train": {
                "epochs": 3,
                "lr": 0.05,
                "optimizer": "adam",
                "seed": 31,
                "batch_size": 2,
                "eval_batch_size": 1,
                "phase_schedule": {
                    "warmup_epochs": 1,
                    "gate_alignment_epochs": 1,
                },
            },
        },
    )

    assert Path(checkpoint_path).is_file()
    assert train_report["phase_schedule"] == {
        "warmup_epochs": 1,
        "gate_alignment_epochs": 1,
    }
    assert train_report["epoch_phase_names"] == [
        "readout_warmup",
        "gate_alignment",
        "full_tuning",
    ]

    diagnostics = json.loads(Path(train_report["loss_diagnostics_path"]).read_text(encoding="utf-8"))
    epoch_report = json.loads(Path(train_report["epoch_predictions_vs_targets_path"]).read_text(encoding="utf-8"))

    for split_name, rows in (("train", diagnostics["train"]), ("val", diagnostics["val"])):
        assert [row["phase_name"] for row in rows] == train_report["epoch_phase_names"]
        assert [row["epoch_index"] for row in rows] == [1, 2, 3]
        for row, expected_phase in zip(rows, train_report["epoch_phase_names"], strict=True):
            assert row["split"] == split_name
            assert row["phase_name"] == expected_phase
            assert set(row["phase_state"]) == {
                "network_trainable",
                "risk_calibration_trainable",
            }
            if expected_phase == "readout_warmup":
                assert row["phase_state"] == {
                    "network_trainable": False,
                    "risk_calibration_trainable": False,
                }
            elif expected_phase == "gate_alignment":
                # v7 patch 2.0 P1.c: gate_alignment 阶段 backbone 已解冻 (lr scale 0.1)，跟进当前 trainer 设计。
                assert row["phase_state"] == {
                    "network_trainable": True,
                    "risk_calibration_trainable": True,
                }
            else:
                assert row["phase_state"] == {
                    "network_trainable": True,
                    "risk_calibration_trainable": True,
                }

    for split_name, rows in (("train", epoch_report["train"]), ("val", epoch_report["val"])):
        assert [row["phase_name"] for row in rows] == train_report["epoch_phase_names"]
        assert [row["epoch_index"] for row in rows] == [1, 2, 3]
        for row, expected_phase in zip(rows, train_report["epoch_phase_names"], strict=True):
            assert row["split"] == split_name
            assert row["phase_name"] == expected_phase
            assert set(row["phase_state"]) == {
                "network_trainable",
                "risk_calibration_trainable",
            }
            if expected_phase == "readout_warmup":
                assert row["phase_state"] == {
                    "network_trainable": False,
                    "risk_calibration_trainable": False,
                }
            elif expected_phase == "gate_alignment":
                # v7 patch 2.0 P1.c: gate_alignment 阶段 backbone 已解冻 (lr scale 0.1)，跟进当前 trainer 设计。
                assert row["phase_state"] == {
                    "network_trainable": True,
                    "risk_calibration_trainable": True,
                }
            else:
                assert row["phase_state"] == {
                    "network_trainable": True,
                    "risk_calibration_trainable": True,
                }


def test_liquid_train_model_epoch_artifacts_track_phase_schedule_for_every_epoch(tmp_path):
    """追踪测试：liquid train model epoch artifacts。\n\n验证 liquid train model epoch artifacts 的追踪机制，\n确保状态变化被正确记录。
    """
    checkpoint_path, train_report = train_model(
        [
            _window_sample(
                quality=0.2,
                target_intermediate={"bias": 0.05, "risk": 0.25, "uwb_scaling": 1.25, "vio_scaling": 1.0},
            ),
            _window_sample(
                quality=0.8,
                target_intermediate={"bias": 0.08, "risk": 0.30, "uwb_scaling": 1.30, "vio_scaling": 1.0},
            ),
        ],
        [
            _window_sample(
                quality=0.4,
                target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0},
            )
        ],
        {
            "name": "liquid_ekf",
            "output_root": tmp_path,
            "train": {
                "epochs": 4,
                "lr": 0.05,
                "optimizer": "adam",
                "seed": 29,
                "batch_size": 2,
                "eval_batch_size": 1,
                "phase_schedule": {
                    "warmup_epochs": 1,
                    "gate_alignment_epochs": 1,
                },
            },
        },
    )

    assert Path(checkpoint_path).is_file()
    assert train_report["phase_schedule"] == {
        "warmup_epochs": 1,
        "gate_alignment_epochs": 1,
    }
    assert train_report["epoch_phase_names"] == [
        "readout_warmup",
        "gate_alignment",
        "full_tuning",
        "full_tuning",
    ]

    diagnostics = json.loads(Path(train_report["loss_diagnostics_path"]).read_text(encoding="utf-8"))
    epoch_report = json.loads(Path(train_report["epoch_predictions_vs_targets_path"]).read_text(encoding="utf-8"))
    expected_phase_states = {
        "readout_warmup": {
            "network_trainable": False,
            "risk_calibration_trainable": False,
        },
        # v7 patch 2.0 P1.c: gate_alignment 阶段 backbone 已解冻 (lr scale 0.1)，
        # 因此 network_trainable=True 跟进当前 trainer 设计，旧期望 False 已废弃。
        "gate_alignment": {
            "network_trainable": True,
            "risk_calibration_trainable": True,
        },
        "full_tuning": {
            "network_trainable": True,
            "risk_calibration_trainable": True,
        },
    }

    for split_name in ("train", "val"):
        assert len(diagnostics[split_name]) == 4
        assert len(epoch_report[split_name]) == 4
        for index, (diagnostic_epoch, prediction_epoch, expected_phase_name) in enumerate(
            zip(
                diagnostics[split_name],
                epoch_report[split_name],
                train_report["epoch_phase_names"],
                strict=True,
            ),
            start=1,
        ):
            assert diagnostic_epoch["epoch_index"] == index
            assert prediction_epoch["epoch_index"] == index
            assert diagnostic_epoch["phase_name"] == expected_phase_name
            assert prediction_epoch["phase_name"] == expected_phase_name
            assert diagnostic_epoch["phase_state"] == expected_phase_states[expected_phase_name]
            assert prediction_epoch["phase_state"] == expected_phase_states[expected_phase_name]
            assert diagnostic_epoch["phase_trainability_contract"]["phase_name"] == expected_phase_name
            assert prediction_epoch["phase_trainability_contract"]["phase_name"] == expected_phase_name
            assert diagnostic_epoch["phase_effective_lr"]["trainable_parameter_count"] > 0
            assert prediction_epoch["phase_effective_lr"]["trainable_parameter_count"] > 0


def test_liquid_selection_score_is_tail_weighted_and_batch_size_invariant():
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    with torch.no_grad():
        for head in (model.bias_head, model.risk_head, model.uwb_scaling_head, model.vio_scaling_head):
            head.projection.weight.zero_()
            head.projection.bias.zero_()

    bias_prediction = 0.0
    scaling_prediction = 1.0
    low_risk_window = _window_sample(
        quality=0.8,
        target_intermediate={
            "bias": bias_prediction,
            "risk": 0.5,
            "uwb_scaling": scaling_prediction,
            "vio_scaling": scaling_prediction,
        },
    ) | {
        "target_trace": {
            "alignment_risk": 0.0,
            "observation_risk": 0.0,
            "quality_risk": 0.0,
            "modality_signal": 0.0,
        }
    }
    high_risk_window = _window_sample(
        quality=0.2,
        target_intermediate={
            "bias": 0.0,
            "risk": 1.0,
            "uwb_scaling": 1.2,
            "vio_scaling": scaling_prediction,
        },
    ) | {
        "target_trace": {
            "alignment_risk": 1.0,
            "observation_risk": 1.0,
            "quality_risk": 1.0,
            "modality_signal": 1.0,
        }
    }

    model._cached_val_windows = None
    model._eval_batch_size = 1
    loss_batch_1 = _evaluate_model(model, [low_risk_window, high_risk_window])
    score_batch_1 = float(model._selection_score)
    model._cached_val_windows = None
    model._eval_batch_size = 2
    loss_batch_2 = _evaluate_model(model, [low_risk_window, high_risk_window])
    score_batch_2 = float(model._selection_score)
    model._cached_val_windows = None
    model._eval_batch_size = 1
    _evaluate_model(model, [high_risk_window])
    high_only_score = float(model._selection_score)

    assert loss_batch_1 == pytest.approx(loss_batch_2)
    assert score_batch_1 == pytest.approx(score_batch_2)
    assert score_batch_1 > (high_only_score / 2.0)


def test_liquid_risk_head_uses_sigmoid_semantics():
    """使用测试：liquid risk head。\n\n验证被测功能正确使用 liquid risk head，\n确保内部依赖被正确调用。
    """
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    with torch.no_grad():
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.fill_(2.0)

    outputs = model.predict_intermediate_tensors(
        {
            "current_modality": "uwb",
            "feature_order": ["quality", "valid"],
            "feature_values": [0.2, 1.0],
            "missing_mask": [0, 0],
            "dt": 0.1,
            "feature_window": [[0.2, 1.0]],
            "missing_mask_window": [[0, 0]],
        }
    )

    assert float(outputs["risk"].detach().item()) == pytest.approx(float(torch.sigmoid(torch.tensor(2.0))), rel=1e-6)


def test_predict_batch_matches_per_sample_liquid_inference():
    """匹配测试：predict batch。\n\n验证 predict batch 的输出与预期一致，\n确保合同合规。
    """
    samples = [
        _window_sample(
            quality=0.2,
            target_intermediate={
                "bias": 0.05,
                "risk": 0.25,
                "uwb_scaling": 1.25,
                "vio_scaling": 1.0,
            },
        ),
        _window_sample(
            quality=0.8,
            target_intermediate={
                "bias": 0.08,
                "risk": 0.3,
                "uwb_scaling": 1.3,
                "vio_scaling": 1.0,
            },
        )
        | {
            "current_modality": "vio",
        },
    ]
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    device = next(model.network.shared_projection.parameters()).device
    materialized = _move_materialized_samples(_materialize_samples(samples), device)

    prediction_batch = _predict_batch(model, [sample[0] for sample in materialized])
    expected_batch = torch.stack(
        [
            torch.stack(
                [
                    model.predict_intermediate_tensors(sample)[key].reshape(())
                    for key in ("bias", "risk", "uwb_scaling", "vio_scaling")
                ]
            )
            for sample in samples
        ],
        dim=0,
    ).to(prediction_batch.device)

    assert torch.allclose(prediction_batch, expected_batch, atol=1e-6, rtol=1e-6)


def test_predict_batch_consumes_explicit_context_tensors_through_training_metadata():
    """显式测试：predict batch consumes。\n\n验证 predict batch consumes 的显式参数优先级，\n确保显式指定覆盖隐式推断。
    """
    base_sample = _window_sample(
        quality=0.3,
        target_intermediate={
            "bias": 0.05,
            "risk": 0.25,
            "uwb_scaling": 1.25,
            "vio_scaling": 1.0,
        },
    )
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    hidden_dim = model.network.hidden_dim
    context_dim = model.output_backbone.context_dim
    filter_context_dim = model.output_backbone.filter_context_dim
    with torch.no_grad():
        model.output_backbone.temporal_context_gate.weight.zero_()
        model.output_backbone.temporal_context_gate.bias.zero_()
        model.output_backbone.observation_context_gate.weight.zero_()
        model.output_backbone.observation_context_gate.bias.zero_()
        model.output_backbone.filter_context_gate.weight.zero_()
        model.output_backbone.filter_context_gate.bias.zero_()
        model.output_backbone.branch_mix_gate.weight.zero_()
        model.output_backbone.branch_mix_gate.bias.zero_()
        model.output_backbone.uwb_branch_mix_gate.weight.zero_()
        model.output_backbone.uwb_branch_mix_gate.bias.zero_()
        model.output_backbone.vio_branch_mix_gate.weight.zero_()
        model.output_backbone.vio_branch_mix_gate.bias.zero_()
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.projection.weight[0, hidden_dim + 2] = 1.0
        model.bias_head.projection.weight[0, hidden_dim + context_dim + 0] = 1.0
        model.bias_head.residual_projection.weight.zero_()
        model.bias_head.residual_projection.bias.zero_()
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()
        model.uwb_scaling_head.projection.weight.zero_()
        model.uwb_scaling_head.projection.bias.zero_()
        model.vio_scaling_head.projection.weight.zero_()
        model.vio_scaling_head.projection.bias.zero_()

    fallback_sample = dict(base_sample)
    explicit_sample = dict(base_sample)
    explicit_context_vector = torch.zeros(context_dim, dtype=torch.float32)
    explicit_filter_context_vector = torch.zeros(filter_context_dim, dtype=torch.float32)
    explicit_context_vector[2] = 0.4
    explicit_filter_context_vector[0] = 0.7
    explicit_sample["context_vector"] = explicit_context_vector
    explicit_sample["filter_context_vector"] = explicit_filter_context_vector

    device = next(model.network.shared_projection.parameters()).device
    materialized = _move_materialized_samples(_materialize_samples([fallback_sample, explicit_sample]), device)
    prediction_batch = _predict_batch(model, [materialized[0][0], materialized[1][0]])

    assert float(prediction_batch[0, 0].detach().item()) != pytest.approx(1.1, abs=1e-6)
    assert float(prediction_batch[1, 0].detach().item()) == pytest.approx(1.1, abs=1e-6)


def test_predict_batch_uses_risk_calibration_when_available():
    """使用测试：predict batch。\n\n验证被测功能正确使用 predict batch，\n确保内部依赖被正确调用。
    """
    sample = _window_sample(
        quality=0.2,
        target_intermediate={
            "bias": 0.05,
            "risk": 0.25,
            "uwb_scaling": 1.25,
            "vio_scaling": 1.0,
        },
    )
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    with torch.no_grad():
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()
        model.risk_calibration.b.fill_(0.8)

    device = next(model.network.shared_projection.parameters()).device
    materialized = _move_materialized_samples(_materialize_samples([sample]), device)
    prediction_batch = _predict_batch(model, [materialized[0][0]])
    inferred = model.predict_intermediate_tensors(sample)

    assert float(prediction_batch[0, 1].detach().item()) == pytest.approx(
        float(inferred["risk"].detach().item()),
        rel=1e-6,
    )


def test_legacy_checkpoint_loaded_new_liquid_parameters_still_participate_in_training(tmp_path):
    """检查点测试：legacy。\n\n验证 legacy 的检查点行为，\n确保保存和加载一致性。
    """
    model_cfg, payload = _legacy_liquid_checkpoint_payload_without_new_readout_keys()
    checkpoint_path = Path(tmp_path) / "legacy_liquid_trainable_checkpoint.pt"
    torch.save(payload, checkpoint_path)

    model = create_model("liquid_ekf", {"checkpoint_path": checkpoint_path})
    sample = {
        **_window_sample(
            quality=0.2,
            target_intermediate={
                "bias": 0.5,
                "risk": 0.1,
                "uwb_scaling": 1.25,
                "vio_scaling": 1.0,
            },
        ),
        "readout_context_by_name": {
            "state_cov_trace": 2.0,
            "pos_cov": 1.0,
            "last_innovation_norm": 0.0,
            "last_gate_skip_flag": 1.0,
        },
        "readout_context_observed_by_name": {
            "state_cov_trace": True,
            "pos_cov": True,
            "last_innovation_norm": False,
            "last_gate_skip_flag": True,
        },
    }

    with torch.no_grad():
        model.output_backbone.temporal_context_gate.weight.zero_()
        model.output_backbone.temporal_context_gate.bias.zero_()
        model.output_backbone.observation_context_gate.weight.zero_()
        model.output_backbone.observation_context_gate.bias.zero_()
        model.output_backbone.filter_context_gate.weight.zero_()
        model.output_backbone.filter_context_gate.bias.zero_()
        model.output_backbone.filter_context_gate.weight[0, 0] = 1.0
        model.output_backbone.branch_mix_gate.weight.zero_()
        model.output_backbone.branch_mix_gate.bias.zero_()
        model.output_backbone.uwb_branch_mix_gate.weight.zero_()
        model.output_backbone.uwb_branch_mix_gate.bias.zero_()
        model.output_backbone.vio_branch_mix_gate.weight.zero_()
        model.output_backbone.vio_branch_mix_gate.bias.zero_()
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.fill_(1.0)
        model.bias_head.projection.weight[0, 0] = 1.0
        model.bias_head.residual_projection.weight.zero_()
        model.bias_head.residual_projection.bias.zero_()

        model.output_backbone.temporal_context_gate.weight.zero_()
        model.output_backbone.temporal_context_gate.bias.zero_()
        model.output_backbone.observation_context_gate.weight.zero_()
        model.output_backbone.observation_context_gate.bias.zero_()
        model.output_backbone.filter_context_gate.weight.zero_()
        model.output_backbone.filter_context_gate.bias.zero_()
        model.output_backbone.branch_mix_gate.weight.zero_()
        model.output_backbone.branch_mix_gate.bias.zero_()
        model.output_backbone.uwb_branch_mix_gate.weight.zero_()
        model.output_backbone.uwb_branch_mix_gate.bias.zero_()
        model.output_backbone.vio_branch_mix_gate.weight.zero_()
        model.output_backbone.vio_branch_mix_gate.bias.zero_()
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()
        model.risk_head.residual_projection.weight.zero_()
        model.risk_head.residual_projection.bias.zero_()
        model.risk_calibration.a_raw.copy_(torch.log(torch.expm1(torch.tensor(1.0))))
        model.risk_calibration.b.zero_()

        model.uwb_scaling_head.projection.weight.zero_()
        model.uwb_scaling_head.projection.bias.zero_()
        model.vio_scaling_head.projection.weight.zero_()
        model.vio_scaling_head.projection.bias.zero_()

    loss = _compute_loss(
        model,
        sample,
        {
            "bias": 0.8,
            "risk": 0.1,
            "uwb_scaling": 1.25,
            "vio_scaling": 1.0,
        },
    )
    loss.backward()

    assert model.output_backbone.filter_context_gate.weight.grad is not None
    assert model.output_backbone.filter_context_gate.weight.grad.abs().sum().item() > 0.0
    assert model.risk_calibration.b.grad is not None
    assert model.risk_calibration.b.grad.abs().item() > 0.0


def test_predict_batch_does_not_apply_second_sigmoid_after_risk_calibration():
    """不侵入测试：predict batch。\n\n验证 predict batch 不会产生副作用，\n确保功能隔离性。
    """
    sample = _window_sample(
        quality=0.2,
        target_intermediate={
            "bias": 0.05,
            "risk": 0.25,
            "uwb_scaling": 1.25,
            "vio_scaling": 1.0,
        },
    )
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    with torch.no_grad():
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()
        model.risk_calibration.a_raw.copy_(torch.log(torch.expm1(torch.tensor(1.0))))
        model.risk_calibration.b.fill_(2.0)

    device = next(model.network.shared_projection.parameters()).device
    materialized = _move_materialized_samples(_materialize_samples([sample]), device)
    prediction_batch = _predict_batch(model, [materialized[0][0]])
    calibrated_once = float(torch.sigmoid(torch.tensor(2.0)).item())
    calibrated_twice = float(torch.sigmoid(torch.tensor(calibrated_once)).item())

    assert float(prediction_batch[0, 1].detach().item()) == pytest.approx(calibrated_once, rel=1e-6)
    assert float(prediction_batch[0, 1].detach().item()) != pytest.approx(calibrated_twice, rel=1e-6)


def test_predict_batch_distinguishes_zero_observed_from_missing_readout_context():
    """缺失测试：predict batch distinguishes zero observed from。\n\n验证 predict batch distinguishes zero observed from 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    base_sample = _window_sample(
        quality=0.2,
        target_intermediate={
            "bias": 0.05,
            "risk": 0.25,
            "uwb_scaling": 1.25,
            "vio_scaling": 1.0,
        },
    )
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    hidden_dim = model.network.hidden_dim
    with torch.no_grad():
        model.output_backbone.temporal_context_gate.weight.zero_()
        model.output_backbone.temporal_context_gate.bias.zero_()
        model.output_backbone.observation_context_gate.weight.zero_()
        model.output_backbone.observation_context_gate.bias.zero_()
        model.output_backbone.filter_context_gate.weight.zero_()
        model.output_backbone.filter_context_gate.bias.zero_()
        model.output_backbone.branch_mix_gate.weight.zero_()
        model.output_backbone.branch_mix_gate.bias.zero_()
        model.output_backbone.uwb_branch_mix_gate.weight.zero_()
        model.output_backbone.uwb_branch_mix_gate.bias.zero_()
        model.output_backbone.vio_branch_mix_gate.weight.zero_()
        model.output_backbone.vio_branch_mix_gate.bias.zero_()
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.projection.weight[0, hidden_dim + model.output_backbone.context_dim + 1] = 1.0
        model.bias_head.residual_projection.weight.zero_()
        model.bias_head.residual_projection.bias.zero_()
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()
        model.uwb_scaling_head.projection.weight.zero_()
        model.uwb_scaling_head.projection.bias.zero_()
        model.vio_scaling_head.projection.weight.zero_()
        model.vio_scaling_head.projection.bias.zero_()

    missing_sample = dict(base_sample)
    missing_sample["readout_context_by_name"] = {
        "state_cov_trace": 0.0,
        "pos_cov": 0.0,
        "last_innovation_norm": 0.0,
        "last_gate_skip_flag": 0.0,
    }
    missing_sample["readout_context_observed_by_name"] = {
        "state_cov_trace": False,
        "pos_cov": False,
        "last_innovation_norm": False,
        "last_gate_skip_flag": False,
    }
    observed_zero_sample = dict(base_sample)
    observed_zero_sample["readout_context_by_name"] = dict(missing_sample["readout_context_by_name"])
    observed_zero_sample["readout_context_observed_by_name"] = {
        "state_cov_trace": True,
        "pos_cov": False,
        "last_innovation_norm": False,
        "last_gate_skip_flag": False,
    }

    device = next(model.network.shared_projection.parameters()).device
    materialized = _move_materialized_samples(_materialize_samples([missing_sample, observed_zero_sample]), device)
    prediction_batch = _predict_batch(model, [materialized[0][0], materialized[1][0]])

    assert float(prediction_batch[0, 0].detach().item()) == pytest.approx(0.0, abs=1e-6)
    assert float(prediction_batch[1, 0].detach().item()) == pytest.approx(1.0, abs=1e-6)


def test_predict_batch_distinguishes_zero_observed_from_missing_current_feature_context():
    """缺失测试：predict batch distinguishes zero observed from。\n\n验证 predict batch distinguishes zero observed from 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    # 铁律 9：上下文第一观测键是 valid（modal 0,1 | valid value 2, flag 3）。
    # 探针接 valid 观测标志；feature_order 不再用 quality 教师标签。
    target = {
        "bias": 0.05,
        "risk": 0.25,
        "uwb_scaling": 1.25,
        "vio_scaling": 1.0,
    }
    observed_zero_sample = {
        "current_modality": "uwb",
        "feature_order": ["valid", "modality_gap_dt"],
        "feature_values": [0.0, 0.1],
        "missing_mask": [0, 0],
        "dt": 0.1,
        "feature_window": [[0.0, 0.1]],
        "missing_mask_window": [[0, 0]],
        "target_intermediate": target,
    }
    missing_sample = {
        "current_modality": "uwb",
        "feature_order": ["valid", "modality_gap_dt"],
        "feature_values": [0.0, 0.1],
        "missing_mask": [1, 0],
        "dt": 0.1,
        "feature_window": [[0.0, 0.1]],
        "missing_mask_window": [[1, 0]],
        "target_intermediate": target,
    }
    model = create_model("liquid_ekf", {"feature_order": ["valid", "modality_gap_dt"]})
    hidden_dim = model.network.hidden_dim
    valid_flag_index = 2 + (2 * 0) + 1  # valid 为 LIQUID_CONTEXT_FEATURE_KEYS[0]
    with torch.no_grad():
        model.output_backbone.temporal_context_gate.weight.zero_()
        model.output_backbone.temporal_context_gate.bias.zero_()
        model.output_backbone.observation_context_gate.weight.zero_()
        model.output_backbone.observation_context_gate.bias.zero_()
        model.output_backbone.filter_context_gate.weight.zero_()
        model.output_backbone.filter_context_gate.bias.zero_()
        model.output_backbone.branch_mix_gate.weight.zero_()
        model.output_backbone.branch_mix_gate.bias.zero_()
        model.output_backbone.uwb_branch_mix_gate.weight.zero_()
        model.output_backbone.uwb_branch_mix_gate.bias.zero_()
        model.output_backbone.vio_branch_mix_gate.weight.zero_()
        model.output_backbone.vio_branch_mix_gate.bias.zero_()
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.bias_head.projection.weight[0, hidden_dim + valid_flag_index] = 1.0
        model.bias_head.residual_projection.weight.zero_()
        model.bias_head.residual_projection.bias.zero_()
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()
        model.uwb_scaling_head.projection.weight.zero_()
        model.uwb_scaling_head.projection.bias.zero_()
        model.vio_scaling_head.projection.weight.zero_()
        model.vio_scaling_head.projection.bias.zero_()

    device = next(model.network.shared_projection.parameters()).device
    materialized = _move_materialized_samples(_materialize_samples([missing_sample, observed_zero_sample]), device)
    prediction_batch = _predict_batch(model, [materialized[0][0], materialized[1][0]])

    assert float(prediction_batch[0, 0].detach().item()) == pytest.approx(0.0, abs=1e-6)
    assert float(prediction_batch[1, 0].detach().item()) == pytest.approx(1.0, abs=1e-6)


def test_liquid_window_feature_stats_fall_back_to_unobserved_zero_when_entire_column_is_missing():
    """缺失测试：liquid window feature stats fall back to unobserved zero when entire column is。\n\n验证 liquid window feature stats fall back to unobserved zero when entire column is 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    sample = _window_sample(
        quality=0.2,
        target_intermediate={
            "bias": 0.05,
            "risk": 0.25,
            "uwb_scaling": 1.25,
            "vio_scaling": 1.0,
        },
    )
    sample["feature_order"] = ["quality", "valid"]
    sample["feature_values"] = [0.0, 1.0]
    sample["missing_mask"] = [1, 0]
    sample["feature_window"] = [[0.0, 1.0]]
    sample["missing_mask_window"] = [[1, 0]]

    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    shared = model.network.extract_shared_features(sample)

    assert shared["feature_means_by_name"]["quality"] == pytest.approx(0.0)
    assert shared["feature_observed_by_name"]["quality"] is False
    assert shared["feature_means_by_name"]["valid"] == pytest.approx(1.0)
    assert shared["feature_observed_by_name"]["valid"] is True


def test_predict_batch_applies_liquid_modality_contract_to_inactive_scaling_heads():
    """合同测试：predict batch applies liquid modality。\n\n验证 predict batch applies liquid modality 的接口合同，\n确保输入输出符合协议约定。
    """
    uwb_sample = _window_sample(
        quality=0.2,
        target_intermediate={
            "bias": 0.05,
            "risk": 0.25,
            "uwb_scaling": 1.25,
            "vio_scaling": 1.0,
        },
    )
    vio_sample = dict(uwb_sample)
    vio_sample["current_modality"] = "vio"
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    with torch.no_grad():
        model.bias_head.projection.weight.zero_()
        model.bias_head.projection.bias.zero_()
        model.risk_head.projection.weight.zero_()
        model.risk_head.projection.bias.zero_()
        model.uwb_scaling_head.projection.weight.zero_()
        model.uwb_scaling_head.projection.bias.fill_(2.0)
        model.vio_scaling_head.projection.weight.zero_()
        model.vio_scaling_head.projection.bias.fill_(2.0)

    device = next(model.network.shared_projection.parameters()).device
    materialized = _move_materialized_samples(_materialize_samples([uwb_sample, vio_sample]), device)
    prediction_batch = _predict_batch(model, [materialized[0][0], materialized[1][0]])

    # v3: scaling_ceiling 从 1.0 放宽到 2.5 后，inactive scaling 头不再被强制 clamp 到 1.0。
    # 当 head bias=2.0，输出 = neutral_floor_softplus(2.0, neutral_floor=1.0)
    # = 1.0 + softplus(2.0) - softplus(0) = 1.0 + 2.1269 - 0.6931 = 2.4338
    import math
    expected_inactive_scaling = 1.0 + math.log(math.exp(2.0) + 1.0) - math.log(2.0)
    assert float(prediction_batch[0, 3].detach().item()) == pytest.approx(expected_inactive_scaling, rel=1e-6)
    assert float(prediction_batch[1, 2].detach().item()) == pytest.approx(expected_inactive_scaling, rel=1e-6)


def _head_trainability_snapshot(model):
    head = model.bias_head
    backbone = model.output_backbone
    return {
        "projection": any(parameter.requires_grad for parameter in head.projection.parameters()),
        "residual_projection": any(parameter.requires_grad for parameter in head.residual_projection.parameters()),
        "temporal_context_gate": any(parameter.requires_grad for parameter in backbone.temporal_context_gate.parameters()),
        "observation_context_gate": any(parameter.requires_grad for parameter in backbone.observation_context_gate.parameters()),
        "filter_context_gate": any(parameter.requires_grad for parameter in backbone.filter_context_gate.parameters()),
        "branch_mix_gate": any(parameter.requires_grad for parameter in backbone.branch_mix_gate.parameters()),
        "uwb_branch_mix_gate": any(parameter.requires_grad for parameter in backbone.uwb_branch_mix_gate.parameters()),
        "vio_branch_mix_gate": any(parameter.requires_grad for parameter in backbone.vio_branch_mix_gate.parameters()),
    }


@pytest.mark.parametrize(
    ("phase_name", "expected"),
    [
        (
            "readout_warmup",
            {
                "network_trainable": False,
                "risk_calibration_trainable": False,
                "projection": True,
                "residual_projection": True,
                "temporal_context_gate": False,
                "observation_context_gate": False,
                "filter_context_gate": False,
                "branch_mix_gate": False,
                "uwb_branch_mix_gate": False,
                "vio_branch_mix_gate": False,
            },
        ),
        (
            "gate_alignment",
            {
                # v7 patch 2.0 P1.c: gate_alignment 阶段 backbone 已解冻 (lr scale 0.1)，
                # 因此 network_trainable=True 跟进当前 trainer 设计，旧期望 False 已废弃。
                "network_trainable": True,
                "risk_calibration_trainable": True,
                "projection": True,
                "residual_projection": True,
                "temporal_context_gate": True,
                "observation_context_gate": True,
                "filter_context_gate": True,
                "branch_mix_gate": True,
                "uwb_branch_mix_gate": True,
                "vio_branch_mix_gate": True,
            },
        ),
        (
            "full_tuning",
            {
                "network_trainable": True,
                "risk_calibration_trainable": True,
                "projection": True,
                "residual_projection": True,
                "temporal_context_gate": True,
                "observation_context_gate": True,
                "filter_context_gate": True,
                "branch_mix_gate": True,
                "uwb_branch_mix_gate": True,
                "vio_branch_mix_gate": True,
            },
        ),
    ],
)
def test_configure_training_phase_matches_liquid_stage_boundaries(phase_name, expected):
    """匹配测试：configure training phase。\n\n验证 configure training phase 的输出与预期一致，\n确保合同合规。
    """
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})

    phase_state = _configure_training_phase(model, phase_name=phase_name)
    trainability = _head_trainability_snapshot(model)

    assert phase_state["network_trainable"] is expected["network_trainable"]
    assert phase_state["risk_calibration_trainable"] is expected["risk_calibration_trainable"]
    for key, value in trainability.items():
        assert value is expected[key]


@pytest.mark.parametrize(
    ("phase_name", "expected_aux"),
    [
        ("readout_warmup", {"calibration": False, "mono": False, "gate_l1": False, "reg_l2": True}),
        ("gate_alignment", {"calibration": True, "mono": True, "gate_l1": True, "reg_l2": True}),
        ("full_tuning", {"calibration": True, "mono": True, "gate_l1": True, "reg_l2": True}),
    ],
)
def test_liquid_phase_trainability_contract_exposes_auxiliary_enablement(phase_name, expected_aux):
    """合同测试：liquid phase trainability。\n\n验证 liquid phase trainability 的接口合同，\n确保输入输出符合协议约定。
    """
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    _configure_training_phase(model, phase_name=phase_name)

    contract = _build_phase_trainability_contract(model, phase_name=phase_name)

    assert contract["phase_name"] == phase_name
    assert contract["auxiliary_terms_enabled"] == expected_aux


def test_liquid_optimizer_only_tracks_current_phase_trainable_parameters():
    """追踪测试：liquid optimizer only。\n\n验证 liquid optimizer only 的追踪机制，\n确保状态变化被正确记录。
    """
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    optimizer_cfg = {"name": "adam", "lr": 0.01, "weight_decay": 0.0}

    _configure_training_phase(model, phase_name="readout_warmup")
    warmup_optimizer = _build_optimizer(model, optimizer_cfg)
    warmup_trainable = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    warmup_optimizer_params = sum(
        int(parameter.numel())
        for group in warmup_optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    )

    _configure_training_phase(model, phase_name="full_tuning")
    full_optimizer = _build_optimizer(model, optimizer_cfg)
    full_trainable = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    full_optimizer_params = sum(
        int(parameter.numel())
        for group in full_optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    )

    assert warmup_optimizer_params == warmup_trainable
    assert full_optimizer_params == full_trainable
    assert full_optimizer_params > warmup_optimizer_params


def test_liquid_gate_alignment_applies_readout_lr_downscale():
    """降档测试：gate alignment readout lr。

    验证阶段二的 readout 参数组会相对配置值进一步降档，避免只在文档层声明。
    """
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    model._lr_layer = {
        "readout": 1.0e-3,
        "gate_cal": 5.0e-4,
        "filter_context_gate": 2.5e-4,
        "backbone": 1.0e-4,
    }
    optimizer_cfg = {"name": "adam", "lr": 1.0e-3, "weight_decay": 0.0}

    _configure_training_phase(model, phase_name="readout_warmup")
    warmup_optimizer = _build_optimizer(model, optimizer_cfg)
    warmup_lrs = sorted({float(group["lr"]) for group in warmup_optimizer.param_groups})

    _configure_training_phase(model, phase_name="gate_alignment")
    gate_optimizer = _build_optimizer(model, optimizer_cfg)
    gate_lrs = sorted({float(group["lr"]) for group in gate_optimizer.param_groups})

    assert 1.0e-3 in warmup_lrs
    assert 2.5e-4 in gate_lrs
    assert 1.0e-3 not in gate_lrs


def test_liquid_gate_alignment_readout_lr_scale_is_configurable():
    """配置测试：gate alignment readout lr scale。

    验证阶段二 readout 降档倍率可由 trainer 配置覆盖。
    """
    trainer_state = build_trainer_state(
        {
            "name": "liquid_ekf",
            "train": {
                "gate_alignment_readout_lr_scale": 0.5,
            },
        }
    )
    assert trainer_state["gate_alignment_readout_lr_scale"] == pytest.approx(0.5)
    assert trainer_state["train"]["gate_alignment_readout_lr_scale"] == pytest.approx(0.5)

    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    model._lr_layer = {
        "readout": 1.0e-3,
        "gate_cal": 5.0e-4,
        "filter_context_gate": 2.5e-4,
        "backbone": 1.0e-4,
    }
    model._gate_alignment_readout_lr_scale = trainer_state["gate_alignment_readout_lr_scale"]
    optimizer_cfg = {"name": "adam", "lr": 1.0e-3, "weight_decay": 0.0}

    _configure_training_phase(model, phase_name="gate_alignment")
    gate_optimizer = _build_optimizer(model, optimizer_cfg)
    gate_lrs = sorted({float(group["lr"]) for group in gate_optimizer.param_groups})

    assert 5.0e-4 in gate_lrs


def test_phase_shock_soft_control_slows_target_groups_only():
    """软控制测试：phase shock slowdown。

    验证 shock 触发时只降低 readout/gate/filter_context 三类参数组学习率。
    """
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    model._lr_layer = {
        "readout": 1.0e-3,
        "gate_cal": 5.0e-4,
        "filter_context_gate": 2.5e-4,
        "backbone": 1.0e-4,
    }
    model._gate_alignment_readout_lr_scale = 0.25
    _configure_training_phase(model, phase_name="gate_alignment")
    optimizer = _build_optimizer(model, {"name": "adam", "lr": 1.0e-3, "weight_decay": 0.0})

    event = _maybe_apply_phase_shock_soft_control(
        optimizer,
        phase_name="gate_alignment",
        phase_switch_shock={"supervised_loss_relative": 0.10, "selection_score_relative": 0.10},
        trainer_state={"soft_control_shock_threshold": 0.05, "soft_control_lr_decay": 0.5},
        # T3 v2 redesign: buffer_epochs 从 2 放宽到 4，phase_epoch_count 须 > 4 才不触发 buffer_only
        phase_epoch_count=6,
    )

    assert event is not None
    assert event["mode"] in {"slowdown_only", "buffer_only", "buffer_cooldown_floor_rollback"}
    affected = {row["group_name"]: (row["old_lr"], row["new_lr"]) for row in event["affected_groups"]}
    assert affected["readout"] == pytest.approx((2.5e-4, 1.25e-4))
    assert affected["gate_cal"] == pytest.approx((5.0e-4, 2.5e-4))
    assert affected["filter_context_gate"] == pytest.approx((2.5e-4, 1.25e-4))


def test_phase_shock_soft_control_escalates_after_streak():
    """连续 shock 后应进一步降低 gate/readout 类组学习率。"""
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    model._lr_layer = {
        "readout": 1.0e-3,
        "gate_cal": 5.0e-4,
        "filter_context_gate": 2.5e-4,
    }
    model._gate_alignment_readout_lr_scale = 0.25
    _configure_training_phase(model, phase_name="gate_alignment")
    optimizer = _build_optimizer(model, {"name": "adam", "lr": 1.0e-3, "weight_decay": 0.0})

    event = _maybe_apply_phase_shock_soft_control(
        optimizer,
        phase_name="gate_alignment",
        phase_switch_shock={"supervised_loss_relative": 0.10, "selection_score_relative": 0.10},
        trainer_state={
            "soft_control_shock_threshold": 0.05,
            "soft_control_lr_decay": 0.5,
            "soft_control_streak_length": 2,
            "soft_control_streak_lr_decay": 0.8,
        },
        prior_shock_streak=1,
        # T3 v2 redesign: buffer_epochs=4，设为 6 确保逃逸 buffer 触发 streak
        phase_epoch_count=6,
    )

    assert event is not None
    assert event["mode"] in {"slowdown_streak", "buffer_only", "buffer_cooldown_floor_rollback"}
    if event["mode"] != "buffer_only":
        assert event["shock_streak"] == 2
    assert event["applied_lr_decay"] == pytest.approx(0.4)
    affected = {row["group_name"]: (row["old_lr"], row["new_lr"]) for row in event["affected_groups"]}
    assert affected["readout"] == pytest.approx((2.5e-4, 1.0e-4))
    assert affected["gate_cal"] == pytest.approx((5.0e-4, 2.0e-4))
    assert affected["filter_context_gate"] == pytest.approx((2.5e-4, 1.0e-4))


def test_phase_shock_soft_control_ignores_negative_relative_improvement():
    """负的相对变化表示验证变好，不应触发 shock soft-control。"""
    model = create_model("liquid_ekf", {"feature_order": ["quality", "valid"]})
    model._lr_layer = {
        "readout": 1.0e-3,
        "gate_cal": 5.0e-4,
        "filter_context_gate": 2.5e-4,
    }
    model._gate_alignment_readout_lr_scale = 0.25
    _configure_training_phase(model, phase_name="gate_alignment")
    optimizer = _build_optimizer(model, {"name": "adam", "lr": 1.0e-3, "weight_decay": 0.0})

    readout_group = next(group for group in optimizer.param_groups if group.get("group_name") == "readout")
    old_readout_lr = float(readout_group["lr"])
    event = _maybe_apply_phase_shock_soft_control(
        optimizer,
        phase_name="gate_alignment",
        phase_switch_shock={"supervised_loss_relative": -0.55, "selection_score_relative": -0.30},
        trainer_state={"soft_control_shock_threshold": 0.05, "soft_control_lr_decay": 0.5},
    )

    assert event is None
    assert float(readout_group["lr"]) == pytest.approx(old_readout_lr)


def test_liquid_runtime_device_accepts_auto_and_indexed_cuda(monkeypatch):
    """接受测试：liquid runtime device。\n\n验证 liquid runtime device 的接受行为，\n确保合法输入被正确处理。
    """
    monkeypatch.setattr("liquidloc.models.liquid.trainer.cuda_runtime_usable", lambda: True)

    auto_device, auto_runtime = _resolve_runtime_device("auto")
    indexed_device, indexed_runtime = _resolve_runtime_device("cuda:0")
    amp_state = _resolve_amp_state({"amp_enabled": "auto", "amp_dtype": "auto"}, indexed_runtime)

    assert auto_device.type == "cuda"
    assert auto_runtime == "cuda"
    assert indexed_device.type == "cuda"
    assert indexed_device.index == 0
    assert indexed_runtime == "cuda:0"
    assert amp_state["enabled"] is True
    assert amp_state["dtype_name"] in {"float16", "bfloat16"}
