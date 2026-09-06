"""Unit tests for handbook_p_series_tools (P15 / P16 / P20 / P23 / P36 / P37)."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from liquidloc.analysis.handbook_p_series_tools import (
    HYPERPARAMETER_KEYS,
    SeedListReport,
    build_unified_report,
    build_warmup_windows_4x1,
    check_code_freeze_compliance,
    compare_hyperparameter_tables,
    compute_hyperparameter_hash,
    compute_robust_statistics,
    evaluate_warmup_window_4x1,
    serialize_hyperparameters,
    verify_seed_list,
)


# =============================================================================
# P15：seed list verification
# =============================================================================


class TestVerifySeedList:
    def test_perfect_match(self):
        report = verify_seed_list(
            manifest_seeds=[0, 1, 2, 3, 4],
            requested_seeds=[0, 1, 2, 3, 4],
        )
        assert report.pass_overall
        assert report.missing_seeds == []
        assert report.unexpected_seeds == []
        assert report.covers_design
        assert report.duplicates_removed == 0

    def test_missing_seeds(self):
        report = verify_seed_list(
            manifest_seeds=[0, 1],
            requested_seeds=[0, 1, 2, 3, 4],
        )
        assert not report.pass_overall
        assert report.missing_seeds == [2, 3, 4]
        assert report.unexpected_seeds == []

    def test_unexpected_seeds(self):
        report = verify_seed_list(
            manifest_seeds=[0, 1, 2, 99],
            requested_seeds=[0, 1, 2],
        )
        assert not report.pass_overall
        assert report.unexpected_seeds == [99]

    def test_duplicates_removed(self):
        report = verify_seed_list(
            manifest_seeds=[0, 0, 1, 1, 2, 2],
            requested_seeds=[0, 1, 2],
            required_min_count=3,  # explicit 3 for this small test
        )
        assert report.pass_overall
        assert report.duplicates_removed == 3
        assert report.manifest_seeds == [0, 1, 2]

    def test_contiguous_flag(self):
        r_contig = verify_seed_list(manifest_seeds=[0, 1, 2, 3, 4], requested_seeds=[0, 1, 2, 3, 4])
        assert r_contig.contiguous
        r_gap = verify_seed_list(manifest_seeds=[0, 2, 4], requested_seeds=[0, 2, 4])
        assert not r_gap.contiguous

    def test_to_dict_roundtrip(self):
        report = verify_seed_list(
            manifest_seeds=[0, 1, 2],
            requested_seeds=[0, 1, 2],
            required_min_count=3,
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["pass_overall"] is True
        assert d["manifest_seeds"] == [0, 1, 2]


# =============================================================================
# P16：hyperparameter table
# =============================================================================


class TestHyperparameterTable:
    def test_serialize_basic(self):
        cfg = {"lr": 0.001, "batch_size": 32, "epochs": 100, "optimizer": "adam"}
        out = serialize_hyperparameters(cfg)
        assert out["lr"] == 0.001
        assert out["batch_size"] == 32
        assert out["epochs"] == 100
        assert out["optimizer"] == "adam"
        assert "output_root" not in out  # excluded by default key list

    def test_serialize_skip_missing(self):
        cfg = {"lr": 0.001, "foo": "bar"}
        out = serialize_hyperparameters(cfg, include_keys=["lr", "nonexistent"])
        assert "lr" in out
        assert "nonexistent" not in out

    def test_serialize_non_finite_falls_back_to_str(self):
        cfg = {"lr": 0.001, "obj": {"a": 1}}
        out = serialize_hyperparameters(cfg, include_keys=["lr", "obj"])
        assert out["lr"] == 0.001
        # dict is not in (str, int, float, bool, None) → str() fallback
        assert out["obj"] == "{'a': 1}"

    def test_serialize_drops_unknown_keys_with_default(self):
        cfg = {"lr": 0.001, "obj": {"a": 1}}
        out = serialize_hyperparameters(cfg)  # default HYPERPARAMETER_KEYS
        assert "lr" in out
        assert "obj" not in out  # 'obj' not in default keys → skipped

    def test_hash_deterministic(self):
        cfg = {"lr": 0.001, "batch_size": 32, "epochs": 100}
        h1 = compute_hyperparameter_hash(cfg)
        h2 = compute_hyperparameter_hash(cfg)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_changes_on_value(self):
        cfg_a = {"lr": 0.001}
        cfg_b = {"lr": 0.002}
        assert compute_hyperparameter_hash(cfg_a) != compute_hyperparameter_hash(cfg_b)

    def test_compare_tables_match(self):
        t1 = {"lr": 0.001, "epochs": 100}
        t2 = {"lr": 0.001, "epochs": 100}
        result = compare_hyperparameter_tables(t1, t2)
        assert result["match"] is True
        assert result["differences"] == []

    def test_compare_tables_mismatch(self):
        t1 = {"lr": 0.001, "epochs": 100}
        t2 = {"lr": 0.002, "epochs": 100}
        result = compare_hyperparameter_tables(t1, t2)
        assert result["match"] is False
        assert result["differences"] == [{"key": "lr", "a": 0.001, "b": 0.002}]


# =============================================================================
# P20：4×1 warmup window evaluation
# =============================================================================


class TestWarmupWindow4x1:
    def test_build_returns_4_windows(self):
        windows = build_warmup_windows_4x1(total_duration_s=60.0, warmup_s=10.0)
        assert len(windows) == 4
        labels = {w.label for w in windows}
        assert labels == {"warmup_A2N2", "warmup_A2N3", "warmup_A3N2", "warmup_A3N3"}

    def test_build_duration_guard(self):
        with pytest.raises(ValueError, match="must be positive"):
            build_warmup_windows_4x1(total_duration_s=0.0)
        with pytest.raises(ValueError, match="warmup_s must be"):
            build_warmup_windows_4x1(total_duration_s=60.0, warmup_s=60.0)

    def test_evaluate_returns_rmse_p95(self):
        # 60 frames at 10 Hz = 6 s; warmup_s=2 s → frames 0–19
        trajectories = {
            "A2N2": [0.5] * 60,
            "A2N3": [1.0] * 60,
        }
        result = evaluate_warmup_window_4x1(
            trajectories=trajectories,
            warmup_s=2.0,
            total_duration_s=6.0,
        )
        assert "A2N2" in result
        assert "A2N3" in result
        assert result["A2N2"]["rmse"] == 0.5
        assert result["A2N3"]["rmse"] == 1.0
        assert result["A2N2"]["n_frames"] == 20
        assert result["A2N2"]["p95"] == 0.5  # all identical → p95 = 0.5

    def test_evaluate_empty_trajectory(self):
        # warmup_s=1.0 < total_duration_s=2.0 → passes guard; empty trajectory gives NaN
        result = evaluate_warmup_window_4x1(
            trajectories={"A2N2": []},
            warmup_s=1.0,
            total_duration_s=2.0,
        )
        assert math.isnan(result["A2N2"]["rmse"])


# =============================================================================
# P23：code freeze gate
# =============================================================================


class TestCodeFreezeGate:
    def test_no_marker_passes(self, tmp_path: Path):
        pass_freeze, info = check_code_freeze_compliance(
            repo_root=tmp_path,
            marker_path=tmp_path / "CODE_FREEZE_COMMIT",
        )
        assert pass_freeze is True
        assert info["pass"] is True
        assert info["reason"] == "no_marker"

    def test_marker_exists_but_empty(self, tmp_path: Path):
        marker = tmp_path / "CODE_FREEZE_COMMIT"
        marker.write_text("{}", encoding="utf-8")
        pass_freeze, info = check_code_freeze_compliance(
            repo_root=tmp_path,
            marker_path=marker,
        )
        assert pass_freeze is False
        assert info["pass"] is False
        assert info["reason"] == "missing_commit_info"

    def test_marker_with_commit_no_git(self, tmp_path: Path):
        marker = tmp_path / "CODE_FREEZE_COMMIT"
        marker.write_text(json.dumps({"git_commit": "abc123"}), encoding="utf-8")
        pass_freeze, info = check_code_freeze_compliance(
            repo_root=tmp_path,
            marker_path=marker,
        )
        # No real git repo → current_commit empty → mismatch
        assert pass_freeze is False
        assert info["frozen_commit"] == "abc123"


# =============================================================================
# P36：robust statistics
# =============================================================================


class TestRobustStatistics:
    def test_mean_and_std(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_robust_statistics(values)
        assert abs(result["mean"] - 3.0) < 1e-9
        # statistics.stdev uses sample std (n-1 divisor): std = sqrt(10/4) = sqrt(2.5)
        assert abs(result["std"] - math.sqrt(2.5)) < 1e-9
        assert result["n_frames"] == 5

    def test_mae(self):
        values = [1.0, 2.0, 3.0]
        result = compute_robust_statistics(values)
        # mean=2, deviations=[1,0,1], mae=2/3
        assert abs(result["mae"] - 2.0 / 3.0) < 1e-9

    def test_median(self):
        values = [1.0, 5.0, 2.0, 4.0, 3.0]
        result = compute_robust_statistics(values)
        assert result["median"] == 3.0

    def test_trimmed_mean(self):
        values = list(range(1, 21))  # 1..20
        result = compute_robust_statistics(values, trim_fraction=0.1)
        # 10% trim → drop 2 from each end → values 3..18, mean = 10.5
        assert abs(result["trimmed_mean"] - 10.5) < 1e-9

    def test_empty_values_returns_nan(self):
        result = compute_robust_statistics([])
        assert math.isnan(result["mean"])
        assert result["n_frames"] == 0

    def test_filters_non_finite(self):
        values = [1.0, float("nan"), 3.0, float("inf")]
        result = compute_robust_statistics(values)
        assert result["n_frames"] == 2
        assert result["mean"] == 2.0


# =============================================================================
# P37：unified statistical reporting
# =============================================================================


class TestUnifiedReport:
    def test_basic_report_structure(self):
        table = [{"frame_error": 1.0}, {"frame_error": 2.0}, {"frame_error": 3.0}]
        report = build_unified_report(
            method_name="liquid_ekf",
            metric_table=table,
        )
        summary = report["method_summary"]
        assert summary["method_name"] == "liquid_ekf"
        assert summary["metric"] == "rmse"
        assert summary["n_frames"] == 3
        assert summary["mean_rmse"] == 2.0
        assert "mae" in summary
        assert "trimmed_mean_rmse" in summary
        assert report["robust_statistics_declaration"]["include_mae"] is True
        assert report["robust_statistics_declaration"]["include_trimmed_mean"] is True

    def test_report_with_p95_values(self):
        table = [{"frame_error": 1.0}]
        p95_vals = [0.5, 0.6, 0.7]
        report = build_unified_report(
            method_name="lstm",
            metric_table=table,
            p95_values=p95_vals,
        )
        assert "p95_metric_summary" in report["method_summary"]
        p95_sum = report["method_summary"]["p95_metric_summary"]
        assert p95_sum["n_p95_values"] == 3

    def test_report_relative_to_baseline(self):
        table = [{"frame_error": 1.0}]
        report = build_unified_report(
            method_name="transformer",
            metric_table=table,
            baseline_method="ekf",
        )
        assert "relative_to_baseline" in report
        rb = report["relative_to_baseline"]
        assert rb["baseline_method"] == "ekf"
        assert rb["cohens_d"] is None  # baseline data not provided → None

    def test_outlier_policy_stated(self):
        report = build_unified_report(method_name="test", metric_table=[])
        policy = report["robust_statistics_declaration"]["outlier_policy"]
        assert policy == "no_removal_no_truncation_no_winsorization"


