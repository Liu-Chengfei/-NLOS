"""LSTM 训练器（lstm_trainer）测试模块。

测试覆盖范围：
- LSTM 模型的训练循环
- 损失计算与反向传播
- 检查点保存与加载
- 训练配置的传递

被测模块：liquidloc.models.lstm_trainer"""

import json
from pathlib import Path

import torch
import pytest
import torch.nn.functional as F
import yaml

from liquidloc.factories.model_factory import create_model
from liquidloc.models.lstm.trainer import build_trainer_state, train_model
from liquidloc.models.liquid.trainer import _resolve_selection_sample_weight as resolve_liquid_selection_sample_weight
from liquidloc.models.features.normalization import neutral_floor_softplus as _neutral_floor_softplus
from liquidloc.models.lstm.trainer import (
    _GLOBAL_L2_WEIGHT,
    _compute_batch_loss,
    _compute_lstm_auxiliary_loss_terms,
    _evaluate_model,
    _extract_window_and_target,
    _materialize_samples,
    _move_materialized_samples,
    _build_head_mask_tensor,
    _build_sample_weight_tensor,
    _normalized_squared_error,
    _predict_batch,
    _project_train_outputs,
    _semantic_loss_matrix,
    _masked_supervised_loss_stats,
    _resolve_amp_state,
    _resolve_runtime_device,
    _resolve_selection_sample_weight,
)


def test_lstm_runtime_device_accepts_auto_and_indexed_cuda(monkeypatch):
    """接受测试：lstm runtime device。\n\n验证 lstm runtime device 的接受行为，\n确保合法输入被正确处理。
    """
    monkeypatch.setattr("liquidloc.models.lstm.trainer.cuda_runtime_usable", lambda: True)

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


def test_lstm_tail_selection_weight_matches_liquid_for_same_tail_signal():
    """匹配测试：lstm tail selection weight。\n\n验证 lstm tail selection weight 的输出与预期一致，\n确保合同合规。
    """
    sample = {
        "target_trace": {
            "alignment_risk": 0.8,
            "observation_risk": 0.9,
            "quality_risk": 0.7,
            "modality_signal": 0.6,
            "uwb_geometry_risk": 0.85,
            "current_modality_gap_dt": 0.27,
        }
    }
    target_outputs = {
        "bias": 0.1,
        "risk": 0.75,
        "uwb_scaling": 1.2,
        "vio_scaling": 1.0,
    }

    lstm_weight = _resolve_selection_sample_weight(sample, target_outputs)
    liquid_weight = resolve_liquid_selection_sample_weight(sample, target_outputs)

    assert lstm_weight == pytest.approx(liquid_weight)


def test_lstm_selection_weight_includes_observation_tail_backfill():
    """不侵入测试：lstm selection weight。\n\n验证 lstm selection weight 会把观测尾部信号纳入加权，\n确保功能隔离性。
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
    expected_weight = 1.0 + (0.5 * target_outputs["risk"]) + (0.1 * target_outputs["risk"])

    lstm_weight = _resolve_selection_sample_weight(sample, target_outputs)
    liquid_weight = resolve_liquid_selection_sample_weight(sample, target_outputs)

    assert lstm_weight == pytest.approx(expected_weight)
    assert liquid_weight == pytest.approx(expected_weight)
    assert lstm_weight == pytest.approx(liquid_weight)


def test_lstm_selection_weight_can_disable_observation_tail_backfill():
    sample = {
        "target_trace": {
            "alignment_risk": 0.4,
            "observation_risk": 0.6,
            "quality_risk": 1.0,
            "modality_signal": 1.0,
        }
    }
    target_outputs = {
        "bias": 0.1,
        "risk": 0.4,
        "uwb_scaling": 1.2,
        "vio_scaling": 1.0,
    }

    lstm_weight = _resolve_selection_sample_weight(sample, target_outputs, observation_coeff=0.0)
    liquid_weight = resolve_liquid_selection_sample_weight(sample, target_outputs, observation_coeff=0.0)

    assert lstm_weight == pytest.approx(1.0 + (0.5 * 0.6))
    assert liquid_weight == pytest.approx(1.0 + (0.5 * 0.6))
    assert lstm_weight == pytest.approx(liquid_weight)


def test_lstm_trainer_ignores_optional_liquid_readout_context_fields():
    """读出上下文测试：lstm trainer ignores optional liquid。\n\n验证 lstm trainer ignores optional liquid 的读出上下文构建，\n确保协方差摘要和门控标志正确传递。
    """
    sample = _lstm_window_sample(
        quality=0.2,
        target_intermediate={"bias": 0.05, "risk": 0.25, "uwb_scaling": 1.25, "vio_scaling": 1.0},
    )
    sample["readout_context_by_name"] = {
        "state_cov_trace": 4.0,
        "pos_cov": 1.5,
        "last_innovation_norm": 0.3,
        "last_gate_skip_flag": 1.0,
    }
    sample["readout_context_observed_by_name"] = {
        "state_cov_trace": True,
        "pos_cov": True,
        "last_innovation_norm": True,
        "last_gate_skip_flag": True,
    }

    normalized_window, target_outputs = _extract_window_and_target(sample)
    materialized = _materialize_samples([sample])
    moved = _move_materialized_samples(materialized, torch.device("cpu"))
    moved_metadata = moved[0][0]

    assert "readout_context_by_name" not in normalized_window
    assert "readout_context_observed_by_name" not in normalized_window
    assert "readout_context_by_name" not in moved_metadata
    assert "readout_context_observed_by_name" not in moved_metadata
    assert target_outputs["bias"] == pytest.approx(0.05)
    assert target_outputs["risk"] == pytest.approx(0.25)
    assert target_outputs["uwb_scaling"] == pytest.approx(1.25)
    assert target_outputs["vio_scaling"] == pytest.approx(1.0)


def test_lstm_train_scaling_projection_preserves_gradient_below_floor():
    """保持性测试：lstm train scaling projection。\n\n验证 lstm train scaling projection 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    raw_value = torch.tensor(-12.0, dtype=torch.float32, requires_grad=True)

    projected = _neutral_floor_softplus(raw_value)
    projected.backward()

    assert projected.item() == pytest.approx(1.0)
    assert raw_value.grad is not None
    assert raw_value.grad.item() > 0.0


def test_lstm_semantic_loss_uses_huber_for_bias_and_log_domain_for_scaling():
    """使用测试：lstm semantic loss。\n\n验证被测功能正确使用 lstm semantic loss，\n确保内部依赖被正确调用。
    """
    prediction = torch.tensor([[2.0, 0.7, 4.0, 2.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.2, 1.0, 1.0]], dtype=torch.float32)

    semantic = _semantic_loss_matrix(prediction, target)
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


def test_lstm_train_and_eval_paths_use_same_semantic_supervision_chain():
    model = create_model("lstm_ekf", {"feature_order": ["quality", "valid"]})
    model._train_batch_size = 1  # type: ignore[attr-defined]
    model._eval_batch_size = 1  # type: ignore[attr-defined]

    train_sample = _lstm_window_sample(
        quality=0.3,
        target_intermediate={"bias": 0.08, "risk": 0.35, "uwb_scaling": 1.40, "vio_scaling": 1.0},
    )
    metadata, target_outputs = _extract_window_and_target(train_sample)
    target_batch = torch.tensor(
        [[target_outputs["bias"], target_outputs["risk"], target_outputs["uwb_scaling"], target_outputs["vio_scaling"]]],
        dtype=torch.float32,
    )

    raw_output_batch = torch.tensor([[0.18, 0.55, 1.80, 1.0]], dtype=torch.float32)
    # 使用 _predict_batch 构造 projected_prediction_batch，
    # 因为 _project_train_outputs 的 risk 现在保留 raw logit（由下游校准处理），
    # 而 _predict_batch 会正确应用 risk_calibration。
    original_forward_sequence_batch = model.network.forward_sequence_batch
    model.network.forward_sequence_batch = lambda sequence_batch, modalities=None: raw_output_batch.clone()  # type: ignore[assignment]
    try:
        projected_prediction_batch = _predict_batch(model, [metadata])
        train_loss = _compute_batch_loss(model, target_batch, [metadata], sample_weights=[1.0])
    finally:
        model.network.forward_sequence_batch = original_forward_sequence_batch  # type: ignore[assignment]

    expected_num, expected_den = _masked_supervised_loss_stats(
        projected_prediction_batch,
        target_batch,
        modalities=["uwb"],
        sample_weights=[1.0],
    )
    expected_supervised = expected_num / expected_den
    # LSTM 现在与 Liquid 对齐，拥有 risk_calibration，辅助损失包含 calibration、mono、gate_l1。
    expected_auxiliary = _compute_lstm_auxiliary_loss_terms(
        model,
        projected_prediction_batch,
        target_batch,
        modalities=["uwb"],
        sample_weights=[1.0],
    )
    expected_total = expected_supervised + expected_auxiliary["total"]

    assert torch.allclose(train_loss, expected_total, atol=1e-6)

    materialized = _move_materialized_samples(_materialize_samples([train_sample]), torch.device("cpu"))
    model._cached_val_windows = materialized  # type: ignore[attr-defined]
    model.network.forward_sequence_batch = lambda sequence_batch, modalities=None: raw_output_batch.clone()  # type: ignore[assignment]
    try:
        eval_loss = _evaluate_model(model, [train_sample], collect_diagnostics=False)
    finally:
        model.network.forward_sequence_batch = original_forward_sequence_batch  # type: ignore[assignment]

    assert float(getattr(model, "_selection_score")) == pytest.approx(float(expected_total.item()))
    assert float(eval_loss) == pytest.approx(float(expected_total.item()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this device placement regression test")
def test_lstm_evaluate_model_moves_target_batch_to_prediction_device(monkeypatch):
    model = create_model("lstm_ekf", {"feature_order": ["quality", "valid"]})
    model._eval_batch_size = 1  # type: ignore[attr-defined]

    sample = _lstm_window_sample(
        quality=0.3,
        target_intermediate={"bias": 0.08, "risk": 0.35, "uwb_scaling": 1.40, "vio_scaling": 1.0},
    )
    prediction_batch = torch.tensor([[0.18, 0.55, 1.80, 1.0]], device="cuda", dtype=torch.float32)
    seen: dict[str, torch.device] = {}

    def fake_predict_batch(_model, metadata):
        assert len(metadata) == 1
        return prediction_batch.clone()

    def fake_masked_component_loss_stats(prediction_batch, target_batch, **kwargs):
        seen["component_target"] = target_batch.device
        assert target_batch.device == prediction_batch.device
        zero = torch.zeros((), device=prediction_batch.device)
        numerators = {key: zero for key in ("bias", "risk", "uwb_scaling", "vio_scaling")}
        denominators = {key: torch.ones((), device=prediction_batch.device) for key in ("bias", "risk", "uwb_scaling", "vio_scaling")}
        return numerators, denominators, ["bias", "risk", "uwb_scaling", "vio_scaling"]

    def fake_masked_supervised_loss_stats(prediction_batch, target_batch, **kwargs):
        seen["supervised_target"] = target_batch.device
        assert target_batch.device == prediction_batch.device
        zero = torch.zeros((), device=prediction_batch.device)
        return zero, torch.ones((), device=prediction_batch.device)

    def fake_aux_terms(model, prediction_batch=None, target_batch=None, **kwargs):
        assert prediction_batch is not None
        assert target_batch is not None
        seen["aux_target"] = target_batch.device
        assert target_batch.device == prediction_batch.device
        zero = torch.zeros((), device=prediction_batch.device)
        return {"calibration": zero, "mono": zero, "reg_l2": zero, "gate_l1": zero, "total": zero}

    monkeypatch.setattr("liquidloc.models.lstm.trainer._predict_batch", fake_predict_batch)
    monkeypatch.setattr("liquidloc.models.lstm.trainer._masked_component_loss_stats", fake_masked_component_loss_stats)
    monkeypatch.setattr("liquidloc.models.lstm.trainer._masked_supervised_loss_stats", fake_masked_supervised_loss_stats)
    monkeypatch.setattr("liquidloc.models.lstm.trainer._compute_lstm_auxiliary_loss_terms", fake_aux_terms)

    loss = _evaluate_model(model, [sample], collect_diagnostics=False)

    assert float(loss) == pytest.approx(0.0)
    assert seen["component_target"] == prediction_batch.device
    assert seen["supervised_target"] == prediction_batch.device
    assert seen["aux_target"] == prediction_batch.device


def test_lstm_epoch_diagnostics_report_explicit_reg_l2_summary(tmp_path):
    """显式测试：lstm epoch diagnostics report。\n\n验证 lstm epoch diagnostics report 的显式参数优先级，\n确保显式指定覆盖隐式推断。
    """
    checkpoint_path, train_report = train_model(
        [
            _lstm_window_sample(
                quality=0.2,
                target_intermediate={"bias": 0.05, "risk": 0.25, "uwb_scaling": 1.25, "vio_scaling": 1.0},
            ),
            _lstm_window_sample(
                quality=0.7,
                target_intermediate={"bias": 0.08, "risk": 0.30, "uwb_scaling": 1.30, "vio_scaling": 1.0},
            ),
        ],
        [
            _lstm_window_sample(
                quality=0.4,
                target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0},
            )
        ],
        {
            "name": "lstm_ekf",
            "output_root": tmp_path,
            "train": {"epochs": 1, "lr": 0.05, "optimizer": "adam", "seed": 23, "batch_size": 2, "eval_batch_size": 1},
        },
    )

    assert Path(checkpoint_path).is_file()
    diagnostics = json.loads(Path(train_report["loss_diagnostics_path"]).read_text(encoding="utf-8"))
    train_epoch = diagnostics["train"][0]
    summary = train_epoch["component_loss_summary"]
    auxiliary = train_epoch["auxiliary_losses"]

    assert auxiliary["reg_l2"] > 0.0
    assert summary["reg_l2"]["mean"] == pytest.approx(auxiliary["reg_l2"])
    assert summary["reg_l2"]["variance"] == pytest.approx(0.0)
    assert train_report["trainer_mode"] == "single_phase_baseline"
    assert train_report["optimizer_weight_decay_applied"] == pytest.approx(0.0)
    stability = train_report["training_stability_audit"]
    assert stability["status"] == "ok"
    assert stability["finite_train_losses"] is True
    assert stability["finite_val_losses"] is True
    assert stability["finite_selection_scores"] is True
    assert stability["best_epoch_in_range"] is True
    assert stability["best_epoch_matches_best_selection_score"] is True
    assert stability["checkpoint_path_exists"] is True
    assert stability["diagnostics_path_exists"] is True
    assert stability["epoch_predictions_path_exists"] is True
    assert stability["early_stop_patience_matches_config"] is True
    assert stability["early_stopped"] is False


def test_lstm_optimizer_weight_decay_stays_zero_when_explicit_reg_l2_is_active():
    """零值测试：lstm optimizer weight decay stays。\n\n验证 lstm optimizer weight decay stays 在零值输入下的行为，\n确保边界情况正确处理。
    """
    trainer_state = build_trainer_state(
        {"name": "lstm_ekf", "train": {"weight_decay": 0.123}}
    )

    assert trainer_state["optimizer"]["weight_decay"] == pytest.approx(0.123)
    assert _GLOBAL_L2_WEIGHT > 0.0


def test_lstm_build_trainer_state_accepts_tail_selection_observation_coeff():
    trainer_state = build_trainer_state(
        {"name": "lstm_ekf", "train": {"tail_selection_observation_coeff": 0.25}}
    )

    assert trainer_state["tail_selection_observation_coeff"] == pytest.approx(0.25)
    assert trainer_state["train"]["tail_selection_observation_coeff"] == pytest.approx(0.25)


def test_lstm_train_report_records_tail_selection_observation_coeff(tmp_path):
    checkpoint_path, train_report = train_model(
        [
            _lstm_window_sample(
                quality=0.2,
                target_intermediate={"bias": 0.05, "risk": 0.25, "uwb_scaling": 1.25, "vio_scaling": 1.0},
            ),
            _lstm_window_sample(
                quality=0.7,
                target_intermediate={"bias": 0.08, "risk": 0.30, "uwb_scaling": 1.30, "vio_scaling": 1.0},
            ),
        ],
        [
            _lstm_window_sample(
                quality=0.4,
                target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0},
            )
        ],
        {
            "name": "lstm_ekf",
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


def _lstm_window_sample(*, quality: float, target_intermediate: dict[str, float]):
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


def test_lstm_build_trainer_state_accepts_epoch_candidate_stride():
    """接受测试：lstm build trainer state。\n\n验证 lstm build trainer state 的接受行为，\n确保合法输入被正确处理。
    """
    trainer_state = build_trainer_state(
        {"name": "lstm_ekf", "train": {"epochs": 4, "epoch_candidate_stride": 3}}
    )

    assert trainer_state["epoch_candidate_stride"] == 3


def test_lstm_yaml_default_paper_grade_training_contract():
    """合同测试：lstm yaml default paper grade training。\n\n验证 lstm yaml default paper grade training 的接口合同，\n确保输入输出符合协议约定。
    """
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "models" / "lstm_ekf.yaml").read_text(encoding="utf-8"))
    # yaml 当前训练预算为 60 轮（配套 outputs/lstm_full_60ep 检查点），但 patience 与
    # lr_scheduler.T_max 仍是历史 160 值；trainer 强制 patience==epochs 且 T_max==epochs。
    # 与 scripts/05_train_lstm.py:100-112 的入口自动同步同口径，归一后再验合同。
    train_section = cfg["train"]
    train_section["patience"] = train_section["epochs"]
    train_section.setdefault("lr_scheduler", {})["T_max"] = train_section["epochs"]
    trainer_state = build_trainer_state(cfg)

    assert trainer_state["optimizer"]["lr"] == pytest.approx(1.5e-4)
    assert trainer_state["optimizer"]["weight_decay"] == pytest.approx(0.0)
    assert trainer_state["epochs"] == 60
    assert trainer_state["patience"] == 60
    assert trainer_state["train"]["batch_size"] == 32
    assert trainer_state["train"]["eval_batch_size"] == 128
    assert trainer_state["train"]["amp_enabled"] == "auto"
    assert trainer_state["train"]["amp_dtype"] == "bf16"
    assert trainer_state["train"]["save_epoch_candidates"] is True
    assert trainer_state["epoch_candidate_stride"] == 10
    assert trainer_state["deterministic"] is True


def test_lstm_train_model_can_subsample_epoch_candidate_exports(tmp_path):
    """导出测试：lstm train model can subsample epoch candidate。\n\n验证 lstm train model can subsample epoch candidate 的导出功能，\n确保产物被正确持久化。
    """
    checkpoint_path, train_report = train_model(
        [
            _lstm_window_sample(
                quality=0.2,
                target_intermediate={"bias": 0.05, "risk": 0.25, "uwb_scaling": 1.25, "vio_scaling": 1.0},
            ),
            _lstm_window_sample(
                quality=0.8,
                target_intermediate={"bias": 0.08, "risk": 0.30, "uwb_scaling": 1.30, "vio_scaling": 1.0},
            ),
        ],
        [
            _lstm_window_sample(
                quality=0.4,
                target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0},
            )
        ],
        {
            "name": "lstm_ekf",
            "output_root": tmp_path,
            "train": {
                "epochs": 4,
                "lr": 0.05,
                "optimizer": "adam",
                "seed": 3,
                "save_epoch_candidates": True,
                "epoch_candidate_stride": 3,
            },
        },
    )

    assert Path(checkpoint_path).is_file()
    assert train_report["epoch_candidate_epochs"] == [1, 3, 4]
    assert len(train_report["epoch_candidate_paths"]) == 3


def test_lstm_train_model_exports_loss_diagnostics_json(tmp_path):
    """导出测试：lstm train model。\n\n验证 lstm train model 的导出功能，\n确保产物被正确持久化。
    """
    checkpoint_path, train_report = train_model(
        [
            _lstm_window_sample(
                quality=0.2,
                target_intermediate={"bias": 0.05, "risk": 0.25, "uwb_scaling": 1.25, "vio_scaling": 1.0},
            ),
            _lstm_window_sample(
                quality=0.7,
                target_intermediate={"bias": 0.08, "risk": 0.30, "uwb_scaling": 1.30, "vio_scaling": 1.0},
            ),
        ],
        [
            _lstm_window_sample(
                quality=0.4,
                target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0},
            )
        ],
        {
            "name": "lstm_ekf",
            "output_root": tmp_path,
            "train": {"epochs": 2, "lr": 0.05, "optimizer": "adam", "seed": 13, "batch_size": 2, "eval_batch_size": 1},
        },
    )

    assert Path(checkpoint_path).is_file()
    diagnostics_path = Path(train_report["loss_diagnostics_path"])
    assert diagnostics_path.is_file()

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["model_name"] == "lstm_ekf"
    assert len(diagnostics["train"]) == 2
    assert len(diagnostics["val"]) == 2
    assert "component_loss_summary" in diagnostics["train"][0]
    assert set(diagnostics["train"][0]["component_loss_summary"]) >= {"bias", "risk", "uwb_scaling", "vio_scaling", "reg_l2", "total"}

    first_train_epoch = diagnostics["train"][0]
    first_batch = first_train_epoch["batches"][0]

    assert first_train_epoch["split"] == "train"
    assert first_batch["modalities"] == ["uwb", "uwb"]
    assert first_batch["seq_lens"] == [1, 1]
    assert len(first_batch["prediction"]) == 2
    assert len(first_batch["target"]) == 2
    assert len(first_batch["semantic_loss_by_head"]) == 2
    assert len(first_batch["squared_error_by_head"]) == 2
    assert len(first_batch["head_mask"]) == 2
    assert first_batch["active_keys"][0] == ["bias", "risk", "uwb_scaling"]
    assert first_batch["sample_weights"][0] > 0.0
    assert set(first_batch["auxiliary_losses"]) >= {"reg_l2", "total"}

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


def test_lstm_loss_diagnostics_truncate_batch_probe_rows_under_large_batch(tmp_path):
    train_windows = [
        _lstm_window_sample(
            quality=0.1 + (0.01 * index),
            target_intermediate={
                "bias": 0.05 + (0.001 * index),
                "risk": 0.20 + (0.002 * index),
                "uwb_scaling": 1.10 + (0.01 * index),
                "vio_scaling": 1.0,
            },
        )
        for index in range(12)
    ]
    checkpoint_path, train_report = train_model(
        train_windows,
        [_lstm_window_sample(quality=0.4, target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0})],
        {
            "name": "lstm_ekf",
            "output_root": tmp_path,
            "train": {"epochs": 1, "lr": 0.01, "optimizer": "adam", "seed": 29, "batch_size": 12, "eval_batch_size": 12},
        },
    )

    assert Path(checkpoint_path).is_file()
    diagnostics = json.loads(Path(train_report["loss_diagnostics_path"]).read_text(encoding="utf-8"))
    first_batch = diagnostics["train"][0]["batches"][0]

    assert first_batch["batch_size"] == 12
    assert first_batch["probe_row_count"] == 4
    assert first_batch["truncated_row_count"] == 8
    assert len(first_batch["prediction"]) == 4
    assert len(first_batch["target"]) == 4
    assert len(first_batch["semantic_loss_by_head"]) == 4
    assert len(first_batch["squared_error_by_head"]) == 4
    assert len(first_batch["head_mask"]) == 4
    assert first_batch["modality_counts"] == {"uwb": 12}
    assert first_batch["seq_len_stats"] == {"min": 1, "max": 1}
    assert first_batch["component_loss_counts"]["bias"] == 12
    assert first_batch["component_loss_counts"]["risk"] == 12
    assert first_batch["component_loss_counts"]["uwb_scaling"] == 12
    assert first_batch["component_loss_counts"]["vio_scaling"] == 0


def test_lstm_epoch_loss_diagnostics_truncate_stored_batches_but_preserve_batch_count(tmp_path):
    checkpoint_path, train_report = train_model(
        [
            _lstm_window_sample(
                quality=0.10,
                target_intermediate={"bias": 0.04, "risk": 0.22, "uwb_scaling": 1.18, "vio_scaling": 1.0},
            ),
            _lstm_window_sample(
                quality=0.20,
                target_intermediate={"bias": 0.05, "risk": 0.24, "uwb_scaling": 1.20, "vio_scaling": 1.0},
            ),
        ],
        [_lstm_window_sample(quality=0.4, target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0})],
        {
            "name": "lstm_ekf",
            "output_root": tmp_path / "lstm_batch_retention",
            "train": {"epochs": 1, "lr": 0.01, "optimizer": "adam", "seed": 31, "batch_size": 1, "eval_batch_size": 1},
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


def test_lstm_train_model_exports_epoch_predictions_vs_targets_json(tmp_path):
    """导出测试：lstm train model。\n\n验证 lstm train model 的导出功能，\n确保产物被正确持久化。
    """
    checkpoint_path, train_report = train_model(
        [
            _lstm_window_sample(
                quality=0.2,
                target_intermediate={"bias": 0.05, "risk": 0.25, "uwb_scaling": 1.25, "vio_scaling": 1.0},
            ),
            _lstm_window_sample(
                quality=0.7,
                target_intermediate={"bias": 0.08, "risk": 0.30, "uwb_scaling": 1.30, "vio_scaling": 1.0},
            ),
        ],
        [
            _lstm_window_sample(
                quality=0.4,
                target_intermediate={"bias": 0.06, "risk": 0.28, "uwb_scaling": 1.22, "vio_scaling": 1.0},
            )
        ],
        {
            "name": "lstm_ekf",
            "output_root": tmp_path,
            "train": {"epochs": 2, "lr": 0.05, "optimizer": "adam", "seed": 19, "batch_size": 2, "eval_batch_size": 1},
        },
    )

    assert Path(checkpoint_path).is_file()
    epoch_path = Path(train_report["epoch_predictions_vs_targets_path"])
    assert epoch_path.is_file()

    report = json.loads(epoch_path.read_text(encoding="utf-8"))
    assert report["model_name"] == "lstm_ekf"
    assert report["output_heads"] == ["bias", "risk", "uwb_scaling", "vio_scaling"]
    assert len(report["train"]) == 2
    assert len(report["val"]) == 2

    first_train_epoch = report["train"][0]
    assert first_train_epoch["split"] == "train"
    assert first_train_epoch["epoch_index"] == 1
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
