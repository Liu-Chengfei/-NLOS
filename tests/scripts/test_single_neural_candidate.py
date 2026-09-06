from __future__ import annotations

"""单神经候选脚本（single_neural_candidate）测试模块。

测试覆盖范围：
- 单个神经候选模型的训练与评估
- 候选配置的验证

被测模块：scripts.single_neural_candidate"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    script_path = ROOT / "scripts" / "22_run_single_neural_candidate.py"
    spec = importlib.util.spec_from_file_location("single_neural_candidate_script", script_path)
    script = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(script)
    return script


def test_resolve_candidate_overrides_for_lstm_uses_surface_defaults():
    """使用测试：resolve candidate overrides for lstm。\n\n验证被测功能正确使用 resolve candidate overrides for lstm，\n确保内部依赖被正确调用。
    """
    script = _load_script()

    surface = {
        "neural_grid": {
            "lstm_ekf": {
                "window": (20, 30),
                "hidden_dim": (64, 96),
                "lr": (1e-3, 3e-4),
                "weight_decay": (0.0, 1e-4),
            }
        }
    }

    overrides = script._resolve_candidate_overrides(
        None,
        model_name="lstm_ekf",
        surface=surface,
        window_size=None,
        hidden_dim=None,
        lr=None,
        weight_decay=None,
        liquid_profile_name=None,
    )

    assert overrides == {
        "window.size": 20,
        "network.hidden_dim": 64,
        "train.lr": 1e-3,
        "train.weight_decay": 0.0,
    }


def test_resolve_candidate_overrides_for_liquid_merges_selected_profile():
    """覆盖测试：resolve candidate。\n\n验证 resolve candidate 的覆盖行为，\n确保显式参数优先于默认值。
    """
    script = _load_script()

    surface = {
        "neural_grid": {
            "liquid_ekf": {
                "window": (20, 30),
                "hidden_dim": (47, 71),
                "lr": (1e-3, 3e-4),
            }
        },
        "liquid_robustness_profiles": [
            {
                "name": "balanced",
                "train.weight_decay": 1e-4,
                "network.pooling_logit_scale": 0.25,
            },
            {
                "name": "conservative",
                "train.weight_decay": 3e-4,
                "network.pooling_logit_scale": 0.30,
            },
        ],
    }

    overrides = script._resolve_candidate_overrides(
        None,
        model_name="liquid_ekf",
        surface=surface,
        window_size=30,
        hidden_dim=71,
        lr=3e-4,
        weight_decay=None,
        liquid_profile_name="conservative",
    )

    assert overrides == {
        "window.size": 30,
        "network.hidden_dim": 71,
        "train.lr": 3e-4,
        "train.weight_decay": 3e-4,
        "network.pooling_logit_scale": 0.30,
    }


def test_resolve_candidate_overrides_for_liquid_rejects_unknown_profile():
    """拒绝测试：resolve candidate overrides for liquid。\n\n验证被测功能对 resolve candidate overrides for liquid 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    script = _load_script()

    surface = {
        "neural_grid": {
            "liquid_ekf": {
                "window": (20,),
                "hidden_dim": (47,),
                "lr": (3e-4,),
            }
        },
        "liquid_robustness_profiles": [{"name": "balanced", "train.weight_decay": 1e-4}],
    }

    with pytest.raises(ValueError, match="unknown liquid robustness profile"):
        script._resolve_candidate_overrides(
            None,
            model_name="liquid_ekf",
            surface=surface,
            window_size=None,
            hidden_dim=None,
            lr=None,
            weight_decay=None,
            liquid_profile_name="missing",
        )


def test_resolve_budget_uses_profile_specific_epoch_key():
    """使用测试：resolve budget。\n\n验证被测功能正确使用 resolve budget，\n确保内部依赖被正确调用。
    """
    script = _load_script()

    surface = {
        "training_budget": {
            "lstm_epochs": 160,
            "liquid_epochs": 160,
            "batch_size": 64,
            "eval_batch_size": 128,
        }
    }

    assert script._resolve_budget(surface, "lstm_ekf") == (160, 64, 128)
    assert script._resolve_budget(surface, "liquid_ekf") == (160, 64, 128)


def test_resolve_runtime_surface_backfills_paper_epoch_stride_when_report_is_legacy(tmp_path):
    script = _load_script()

    paper_run_root = tmp_path / "paper_run"
    paper_run_root.mkdir(parents=True, exist_ok=True)
    (paper_run_root / "paper_run_report.json").write_text(
        """
        {
          "neural_search_surface": {
            "profile_name": "paper",
            "checkpoint_selection_mode": "downstream_val_rescore",
            "epoch_candidate_stride": 10,
            "liquid_robustness_profiles": []
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    class _Module:
        @staticmethod
        def _resolve_neural_search_surface(profile_name):
            assert profile_name == "paper"
            return {
                "profile_name": "paper",
                "checkpoint_selection_mode": "trainer_validation_best",
                "epoch_candidate_stride": None,
                "liquid_robustness_profiles": [],
            }

    surface = script._resolve_runtime_surface(_Module(), paper_run_root=paper_run_root, requested_profile="paper")

    assert surface["checkpoint_selection_mode"] == "trainer_validation_best"
    assert surface["epoch_candidate_stride"] == 10


def test_resolve_baseline_scoring_metrics_by_experiment_falls_back_to_ekf_selected_metrics():
    script = _load_script()

    metrics = script._resolve_baseline_scoring_metrics_by_experiment(
        {
            "ekf": {
                "selected_metrics": {
                    "e9_dual_degradation": {"rmse": 0.64, "p95": 0.84},
                    "e0_safe_mode": {"rmse": 0.47, "p95": 0.74},
                }
            }
        }
    )

    assert metrics == {
        "e9_dual_degradation": {"rmse": 0.64, "p95": 0.84},
        "e0_safe_mode": {"rmse": 0.47, "p95": 0.74},
    }


def test_main_passes_paper_baseline_metrics_to_neural_candidate_and_summary(tmp_path, monkeypatch):
    script = _load_script()
    paper_run_root = tmp_path / "paper_run"
    prepare_root = paper_run_root / "prepare" / "sim"
    raw_root = paper_run_root / "raw_sim"
    split_root = paper_run_root / "splits"
    classical_root = paper_run_root / "classical_search"
    output_root = tmp_path / "out"

    prepare_root.mkdir(parents=True)
    raw_root.mkdir(parents=True)
    split_root.mkdir(parents=True)
    classical_root.mkdir(parents=True)

    (paper_run_root / "audits").mkdir(parents=True, exist_ok=True)
    (paper_run_root / "audits" / "data_readiness.json").write_text(
        json.dumps({"sim": {"raw_root": str(raw_root)}}),
        encoding="utf-8",
    )
    (split_root / "split_manifest.json").write_text(
        json.dumps({"train_ids": ["train_a"], "val_ids": ["val_a"], "test_ids": ["test_a"]}),
        encoding="utf-8",
    )
    (classical_root / "search_audit.json").write_text(
        json.dumps(
            {
                "selected_estimator_cfgs": {
                    "ekf": {"name": "ekf"},
                    "robust_ekf": {"name": "robust_ekf"},
                    "fgo": {"name": "fgo"},
                },
                "ekf": {
                    "selected_metrics": {
                        "e9_dual_degradation": {"rmse": 0.64, "p95": 0.84},
                        "e0_safe_mode": {"rmse": 0.47, "p95": 0.74},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_run_neural_search_candidate(**kwargs):
        captured["baseline_scoring_metrics_by_experiment"] = kwargs["baseline_scoring_metrics_by_experiment"]
        captured["checkpoint_selection_mode"] = kwargs["checkpoint_selection_mode"]
        captured["epoch_candidate_stride"] = kwargs["epoch_candidate_stride"]
        return {
            "signature": "candidate_sig",
            "score_vector": [0.1, 0.2],
            "scoring_runs": [
                {
                    "experiment_id": "e9_dual_degradation",
                    "baseline_reused": True,
                    "metrics": {"rmse": 0.61},
                }
            ],
            "train_report": {"paper_selected_best_epoch": 90},
            "checkpoint_selection": {"selected_best_epoch": 90},
        }

    fake_module = SimpleNamespace(
        _resolve_neural_search_surface=lambda profile_name: {
            "profile_name": profile_name,
            "checkpoint_selection_mode": "downstream_val_rescore",
            "epoch_candidate_stride": 10,
            "training_budget": {
                "liquid_epochs": 160,
                "lstm_epochs": 160,
                "batch_size": 32,
                "eval_batch_size": 32,
            },
            "neural_grid": {
                "liquid_ekf": {"window": [20], "hidden_dim": [44], "lr": [1e-3]},
                "lstm_ekf": {"window": [20], "hidden_dim": [44], "lr": [1e-3], "weight_decay": [0.0]},
            },
            "liquid_robustness_profiles": [
                {"name": "v14_cellgate_tail_async_mid", "train.weight_decay": 1.5e-4}
            ],
        },
        _run_neural_search_candidate=_fake_run_neural_search_candidate,
    )
    monkeypatch.setattr(script, "_load_paper_module", lambda: fake_module)

    assert (
        script.main(
            [
                "--paper-run-root",
                str(paper_run_root),
                "--model-name",
                "liquid_ekf",
                "--profile",
                "paper",
                "--seed",
                "0",
                "--device",
                "cpu",
                "--liquid-robustness-profile",
                "v14_cellgate_tail_async_mid",
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )

    summary = json.loads((output_root / "single_candidate_summary.json").read_text(encoding="utf-8"))
    assert captured["baseline_scoring_metrics_by_experiment"] == {
        "e9_dual_degradation": {"rmse": 0.64, "p95": 0.84},
        "e0_safe_mode": {"rmse": 0.47, "p95": 0.74},
    }
    assert captured["checkpoint_selection_mode"] == "downstream_val_rescore"
    assert captured["epoch_candidate_stride"] == 10
    assert summary["baseline_scoring_metrics_by_experiment"] == {
        "e9_dual_degradation": {"rmse": 0.64, "p95": 0.84},
        "e0_safe_mode": {"rmse": 0.47, "p95": 0.74},
    }
