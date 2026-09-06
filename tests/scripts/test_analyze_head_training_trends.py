from __future__ import annotations

"""头部训练趋势分析脚本（analyze_head_training_trends）测试模块。

测试覆盖范围：
- 训练趋势分析的数据处理
- 头部输出的统计与可视化输入

被测模块：scripts.analyze_head_training_trends"""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    script_path = ROOT / "scripts" / "21_analyze_head_training_trends.py"
    spec = importlib.util.spec_from_file_location("analyze_head_training_trends", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_head_error_trend_report_summarizes_active_head_metrics():
    """指标测试：build head error trend report summarizes active head。\n\n验证 build head error trend report summarizes active head 的指标计算，\n确保指标值和分组正确。
    """
    module = _load_module()
    loss_payload = {
        "model_name": "demo_model",
        "checkpoint_format": "demo_v1",
        "train": [
            {
                "split": "train",
                "epoch_index": 1,
                "mean_loss": 0.5,
                "selection_score": None,
                "batch_count": 1,
                "batches": [
                    {
                        "prediction": [
                            [0.2, 0.4, 1.3, 1.0],
                            [0.0, 0.8, 1.0, 1.5],
                        ],
                        "target": [
                            [0.1, 0.5, 1.1, 1.0],
                            [0.2, 0.6, 1.0, 1.7],
                        ],
                        "active_keys": [
                            ["bias", "risk", "uwb_scaling"],
                            ["risk", "vio_scaling"],
                        ],
                        "sample_weights": [2.0, 1.0],
                    }
                ],
            }
        ],
        "val": [],
    }
    epoch_predictions_payload = {
        "output_heads": ["bias", "risk", "uwb_scaling", "vio_scaling"],
        "train": [
            {
                "split": "train",
                "epoch_index": 1,
                "active_sample_count_by_head": {
                    "bias": 1,
                    "risk": 2,
                    "uwb_scaling": 1,
                    "vio_scaling": 1,
                },
                "prediction_mean": {
                    "bias": 0.2,
                    "risk": 0.6,
                    "uwb_scaling": 1.3,
                    "vio_scaling": 1.5,
                },
                "target_mean": {
                    "bias": 0.1,
                    "risk": 0.55,
                    "uwb_scaling": 1.1,
                    "vio_scaling": 1.7,
                },
            }
        ],
        "val": [],
    }

    report = module.build_head_error_trend_report(
        loss_diagnostics_payload=loss_payload,
        epoch_predictions_payload=epoch_predictions_payload,
        source_paths={"loss_diagnostics": "demo_loss.json"},
    )

    assert report["model_name"] == "demo_model"
    train_epoch = report["train"][0]
    assert train_epoch["snapshot_validation"] == {
        "available": True,
        "validation_mode": "active_head_snapshot_strict",
        "counts_match": True,
        "means_match": True,
    }

    bias_metrics = train_epoch["head_metrics"]["bias"]
    assert bias_metrics["active_sample_count"] == 1
    assert bias_metrics["mae"] == pytest.approx(0.1)
    assert bias_metrics["rmse"] == pytest.approx(0.1)
    assert bias_metrics["mean_signed_error"] == pytest.approx(0.1)

    risk_metrics = train_epoch["head_metrics"]["risk"]
    assert risk_metrics["active_sample_count"] == 2
    assert risk_metrics["mae"] == pytest.approx(0.15)
    assert risk_metrics["rmse"] == pytest.approx(0.15811388)
    assert risk_metrics["mean_signed_error"] == pytest.approx(0.05)
    assert risk_metrics["weighted_mae"] == pytest.approx(0.13333333)
    assert risk_metrics["weighted_rmse"] == pytest.approx(0.14142136)
    assert risk_metrics["weighted_mean_signed_error"] == pytest.approx(0.0)

    uwb_metrics = train_epoch["head_metrics"]["uwb_scaling"]
    assert uwb_metrics["mae"] == pytest.approx(0.2)
    assert uwb_metrics["rmse"] == pytest.approx(0.2)

    vio_metrics = train_epoch["head_metrics"]["vio_scaling"]
    assert vio_metrics["mean_signed_error"] == pytest.approx(-0.2)
    assert vio_metrics["mae"] == pytest.approx(0.2)


def test_build_head_error_trend_report_ignores_augmented_loss_diagnostics_fields():
    module = _load_module()
    loss_payload = {
        "model_name": "demo_model",
        "checkpoint_format": "demo_v1",
        "train": [
            {
                "split": "train",
                "epoch_index": 1,
                "mean_loss": 0.5,
                "selection_score": None,
                "batch_count": 1,
                "auxiliary_losses": {
                    "calibration": 0.1,
                    "mono": 0.01,
                    "reg_l2": 0.001,
                    "gate_l1": 0.002,
                    "total": 0.113,
                },
                "component_loss_summary": {
                    "bias": {"mean": 0.1, "min": 0.1, "max": 0.1},
                    "risk": {"mean": 0.2, "min": 0.2, "max": 0.2},
                },
                "batches": [
                    {
                        "prediction": [[0.2, 0.4, 1.3, 1.0]],
                        "target": [[0.1, 0.5, 1.1, 1.0]],
                        "active_keys": [["bias", "risk", "uwb_scaling"]],
                        "sample_weights": [1.0],
                        "semantic_loss_by_head": [[0.01, 0.01, 0.04, 0.0]],
                        "squared_error_by_head": [[0.01, 0.01, 0.04, 0.0]],
                        "auxiliary_losses": {
                            "calibration": 0.1,
                            "mono": 0.01,
                            "reg_l2": 0.001,
                            "gate_l1": 0.002,
                            "total": 0.113,
                        },
                    }
                ],
            }
        ],
        "val": [],
    }
    epoch_predictions_payload = {
        "output_heads": ["bias", "risk", "uwb_scaling", "vio_scaling"],
        "train": [
            {
                "split": "train",
                "epoch_index": 1,
                "active_sample_count_by_head": {
                    "bias": 1,
                    "risk": 1,
                    "uwb_scaling": 1,
                    "vio_scaling": 0,
                },
                "prediction_mean": {
                    "bias": 0.2,
                    "risk": 0.4,
                    "uwb_scaling": 1.3,
                    "vio_scaling": None,
                },
                "target_mean": {
                    "bias": 0.1,
                    "risk": 0.5,
                    "uwb_scaling": 1.1,
                    "vio_scaling": None,
                },
            }
        ],
        "val": [],
    }

    report = module.build_head_error_trend_report(
        loss_diagnostics_payload=loss_payload,
        epoch_predictions_payload=epoch_predictions_payload,
    )

    assert report["train"][0]["head_metrics"]["bias"]["mae"] == pytest.approx(0.1)
    assert report["train"][0]["head_metrics"]["risk"]["rmse"] == pytest.approx(0.1)


def test_resolve_input_paths_uses_train_report_references(tmp_path: Path):
    """使用测试：resolve input paths。\n\n验证被测功能正确使用 resolve input paths，\n确保内部依赖被正确调用。
    """
    module = _load_module()
    loss_path = tmp_path / "demo_loss_diagnostics.json"
    epoch_path = tmp_path / "demo_epoch_predictions_vs_targets.json"
    train_report_path = tmp_path / "demo_train_report.json"
    loss_path.write_text("{}", encoding="utf-8")
    epoch_path.write_text("{}", encoding="utf-8")
    train_report_path.write_text(
        json.dumps(
            {
                "loss_diagnostics_path": str(loss_path),
                "epoch_predictions_vs_targets_path": str(epoch_path),
            }
        ),
        encoding="utf-8",
    )

    resolved_loss_path, resolved_epoch_path = module._resolve_input_paths(
        train_report_path=train_report_path,
        loss_diagnostics_path=None,
        epoch_predictions_path=None,
    )

    assert resolved_loss_path == loss_path
    assert resolved_epoch_path == epoch_path


def test_build_head_error_trend_report_accepts_legacy_snapshot_schema():
    """接受测试：build head error trend report。\n\n验证 build head error trend report 的接受行为，\n确保合法输入被正确处理。
    """
    module = _load_module()
    loss_payload = {
        "model_name": "demo_model",
        "checkpoint_format": "demo_v1",
        "train": [
            {
                "split": "train",
                "epoch_index": 1,
                "mean_loss": 0.5,
                "selection_score": None,
                "batch_count": 1,
                "batches": [
                    {
                        "prediction": [[0.2, 0.4, 1.3, 1.0]],
                        "target": [[0.1, 0.5, 1.1, 1.0]],
                        "active_keys": [["bias", "risk", "uwb_scaling"]],
                        "sample_weights": [1.0],
                    }
                ],
            }
        ],
        "val": [],
    }
    epoch_predictions_payload = {
        "output_heads": ["bias", "risk", "uwb_scaling", "vio_scaling"],
        "train": [
            {
                "split": "train",
                "epoch_index": 1,
                "prediction_mean": {"bias": 0.2, "risk": 0.4, "uwb_scaling": 1.3, "vio_scaling": 1.0},
                "target_mean": {"bias": 0.1, "risk": 0.5, "uwb_scaling": 1.1, "vio_scaling": 1.0},
            }
        ],
        "val": [],
    }

    report = module.build_head_error_trend_report(
        loss_diagnostics_payload=loss_payload,
        epoch_predictions_payload=epoch_predictions_payload,
    )

    assert report["train"][0]["snapshot_validation"] == {
        "available": True,
        "validation_mode": "legacy_snapshot_not_strictly_comparable",
    }


def test_build_head_error_trend_report_preserves_fixed_probe_rows():
    """保持性测试：build head error trend report。\n\n验证 build head error trend report 在处理过程中保持关键属性不变，\n确保数据不被意外修改。
    """
    module = _load_module()
    loss_payload = {
        "model_name": "demo_model",
        "checkpoint_format": "demo_v1",
        "train": [
            {
                "split": "train",
                "epoch_index": 1,
                "mean_loss": 0.25,
                "selection_score": 0.2,
                "batch_count": 1,
                "batches": [
                    {
                        "prediction": [[0.2, 0.4, 1.3, 1.0]],
                        "target": [[0.1, 0.5, 1.1, 1.0]],
                        "active_keys": [["bias", "risk", "uwb_scaling"]],
                        "sample_weights": [1.0],
                    }
                ],
            }
        ],
        "val": [],
    }
    epoch_predictions_payload = {
        "output_heads": ["bias", "risk", "uwb_scaling", "vio_scaling"],
        "train": [
            {
                "split": "train",
                "epoch_index": 1,
                "active_sample_count_by_head": {
                    "bias": 1,
                    "risk": 1,
                    "uwb_scaling": 1,
                    "vio_scaling": 0,
                },
                "prediction_mean": {
                    "bias": 0.2,
                    "risk": 0.4,
                    "uwb_scaling": 1.3,
                    "vio_scaling": None,
                },
                "target_mean": {
                    "bias": 0.1,
                    "risk": 0.5,
                    "uwb_scaling": 1.1,
                    "vio_scaling": None,
                },
                "fixed_probe_source": "cached_split_prefix",
                "fixed_probe_sample_count": 1,
                "fixed_probe_rows": [
                    {
                        "sample_index_in_split": 0,
                        "modality": "uwb",
                        "seq_len": 20,
                        "sample_weight": 1.0,
                        "active_keys": ["bias", "risk", "uwb_scaling"],
                        "prediction_by_head": {
                            "bias": 0.2,
                            "risk": 0.4,
                            "uwb_scaling": 1.3,
                            "vio_scaling": 1.0,
                        },
                        "target_by_head": {
                            "bias": 0.1,
                            "risk": 0.5,
                            "uwb_scaling": 1.1,
                            "vio_scaling": 1.0,
                        },
                        "signed_error_by_head": {
                            "bias": 0.1,
                            "risk": -0.1,
                            "uwb_scaling": 0.2,
                            "vio_scaling": 0.0,
                        },
                    }
                ],
            }
        ],
        "val": [],
    }

    report = module.build_head_error_trend_report(
        loss_diagnostics_payload=loss_payload,
        epoch_predictions_payload=epoch_predictions_payload,
    )

    fixed_probe_trends = report["fixed_probe_trends"]["train"]
    assert len(fixed_probe_trends) == 1
    assert fixed_probe_trends[0]["epoch_index"] == 1
    assert fixed_probe_trends[0]["fixed_probe_source"] == "cached_split_prefix"
    assert fixed_probe_trends[0]["fixed_probe_sample_count"] == 1
    assert len(fixed_probe_trends[0]["fixed_probe_rows"]) == 1
    assert fixed_probe_trends[0]["fixed_probe_rows"][0]["sample_index_in_split"] == 0
    assert fixed_probe_trends[0]["fixed_probe_rows"][0]["modality"] == "uwb"
    assert fixed_probe_trends[0]["fixed_probe_rows"][0]["prediction_by_head"]["bias"] == pytest.approx(0.2)
    assert fixed_probe_trends[0]["fixed_probe_rows"][0]["target_by_head"]["risk"] == pytest.approx(0.5)
    assert fixed_probe_trends[0]["fixed_probe_rows"][0]["signed_error_by_head"]["uwb_scaling"] == pytest.approx(0.2)
