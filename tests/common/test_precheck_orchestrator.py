"""Unit tests for precheck_orchestrator + metrics_quality modules.

覆盖 P7/P8/P9/P21/P27/P33/P34/P36/P39 实施完整性 + 全部 60+ 个 orchestrator check。
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from liquidloc.analysis.metrics_quality import (
    ActivationStatsResult,
    BestValResult,
    MetricBundle,
    NormalizationLeakResult,
    OverfitResult,
    RotationChainResult,
    Sim3AlignmentResult,
    UnitCheckResult,
    align_rigid,
    align_sim3,
    check_best_val_checkpoint,
    check_normalization_leak,
    check_overfit,
    check_units,
    compute_activation_stats,
    compute_all_metrics,
    compute_config_hash,
    compute_mae,
    compute_median_error,
    compute_p50,
    compute_p95,
    compute_rmse,
    compute_trimmed_mean,
    ensure_finite,
    evaluate_sim3_alignment,
    get_git_commit_hash,
    run_rotation_chain_test,
)
from liquidloc.common.precheck_orchestrator import (
    CheckResult,
    PrecheckReport,
    check_DQ1_difficulty_gradient,
    check_DQ2_snr,
    check_DQ3_distribution,
    check_DQ4_sample_size,
    check_E1_audit_trail,
    check_E2_gate_records,
    check_E3_dual_review,
    check_E4_spot_rerun,
    check_E5_alert_log_consistency,
    check_E6_audit_exit,
    check_G1_unit_completeness,
    check_G2_metrics_readable,
    check_G3_config_data_cross,
    check_G4_alerts_cleared,
    check_G5_output_aligned,
    check_I1_data_generator,
    check_I2_s9_script,
    check_I3_5_methods,
    check_I4_stats_script,
    check_I5_eval_pipeline,
    check_P1_composition,
    check_P10_2d_only,
    check_P11_input_isolation,
    check_P12_train_test_symmetry,
    check_P13_rq_matching,
    check_P14_shared_4_heads,
    check_P15_per_method_tuning,
    check_P16_ekf_init_protocol,
    check_P17_robust_ekf_huber,
    check_P18_baselines_with_heads,
    check_P19_seed_system,
    check_P20_window,
    check_P21_normalization_isolation,
    check_P22_stats_script,
    check_P23_config_consistency,
    check_P24_no_nan_inf,
    check_P25_async_timestamps,
    check_P26_vio_payload,
    check_P27_activation_stats,
    check_P28_dual_rate,
    check_P29_windowing,
    check_P30_split_before_window,
    check_P31_edge_cases,
    check_P32_same_seed_init,
    check_P33_best_val,
    check_P34_no_overfit,
    check_P35_head_consistency,
    check_P36_metric_declaration,
    check_P37_warmup_removed,
    check_P38_world_frame,
    check_P39_persistence,
    check_P2_missing_source,
    check_P3_m1_burst,
    check_P4_k1_gdop,
    check_P5_no_out_of_domain,
    check_P6_size_and_split,
    check_P7_rotation_chain,
    check_P8_units_and_normalization,
    check_P9_sim3_alignment,
    check_Pre1_environment_locked,
    check_Pre2_decision_log,
    check_Pre3_directory_structure,
    check_Pre4_checksums,
    check_Pre5_resource_budget,
    check_Pre6_seed_manifest_consistency,
    run_all_prechecks,
)


# ============================================================================
# metrics_quality 实施完整性测试
# ============================================================================


class TestEnsureFinite:
    def test_finite_passes(self):
        ensure_finite(np.array([1.0, 2.0, 3.0]))

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            ensure_finite(np.array([1.0, np.nan, 3.0]), name="test")

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            ensure_finite(np.array([1.0, np.inf]))


class TestMetricBundle:
    def test_rmse(self):
        err = np.array([3.0, 4.0])
        assert compute_rmse(err) == pytest.approx(5.0 / math.sqrt(2), rel=1e-6)

    def test_mae(self):
        assert compute_mae(np.array([1.0, -2.0, 3.0])) == pytest.approx(2.0)

    def test_median(self):
        assert compute_median_error(np.array([1.0, 2.0, 3.0])) == 2.0

    def test_p50(self):
        assert compute_p50(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == 3.0

    def test_p95(self):
        x = np.arange(100, dtype=float)
        assert compute_p95(x) == pytest.approx(94.05)

    def test_trimmed_mean(self):
        # 10% trim of [1, 2, ..., 100]: drop 5 from each side, mean = 50.5
        x = np.arange(1, 101, dtype=float)
        assert compute_trimmed_mean(x, trim_ratio=0.05) == pytest.approx(50.5)

    def test_full_bundle(self):
        err = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
        b = compute_all_metrics(err)
        assert b.rmse > 0
        assert b.mae > 0
        assert b.median == 1.0
        assert b.p50 == 1.0
        assert b.p95 > 2.0
        assert b.trimmed_mean > 0


class TestRotationChain:
    def test_rotation_chain_detects_turn(self):
        # 100 步，每步 2π/99 rad → cumsum[-1] = 2π/99 * 99 = 2π ≥ threshold → has_turned=True
        n = 99
        gyro_z = np.full(n, 2 * math.pi / n)  # rad/s, dt=1.0 → final cumsum = 2π
        # 闭合圆轨迹（起点终点一致）
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False) + 2 * math.pi / n / 2
        traj_x = 10 + 2 * np.cos(angles)
        traj_z = 10 + 2 * np.sin(angles)
        result = run_rotation_chain_test(gyro_z, imu_dt=1.0, trajectory_x=traj_x, trajectory_z=traj_z, start_x=traj_x[0], start_z=traj_z[0])
        # 绕完一圈 → yaw_accumulated ≥ 2π
        assert result.yaw_direction_correct, f"yaw_accum={result.imu_angle_accumulated_rad}"
        assert result.imu_angle_accumulated_rad >= 6.28
        # 起终点差异 < 0.5m（circle 闭合）
        assert result.passed, f"failed: {result.detail}"


class TestUnitCheck:
    def test_meters_and_radians(self):
        r = check_units(
            range_values=np.array([5.0, 10.0, 15.0]),
            yaw_values=np.array([0.0, 0.5, -0.5]),
            position_values=np.array([10.0, 5.0, 0.0]),
        )
        assert r.passed
        assert r.distance_unit == "m"
        assert r.yaw_unit == "rad"
        assert not r.has_x57_or_x100_artifact

    def test_detect_degree_misuse(self):
        # yaw 在 [-180, 180]（疑似 deg 当 rad）
        r = check_units(
            range_values=np.array([5.0]),
            yaw_values=np.array([0.0, 90.0, -90.0]),
            position_values=np.array([10.0, 5.0]),
        )
        assert r.yaw_unit == "deg"
        assert not r.passed

    def test_detect_x57_artifact(self):
        r = check_units(
            range_values=np.array([5.0, 57.3]),
            yaw_values=np.array([0.0]),
            position_values=np.array([10.0]),
        )
        assert r.has_x57_or_x100_artifact
        assert not r.passed


class TestSim3Alignment:
    def test_identity_alignment(self):
        n = 30
        x = np.random.rand(n, 2)
        aligned = align_sim3(x, x)
        assert np.linalg.norm(aligned - x).max() < 1e-9

    def test_rotation_recovery(self):
        n = 30
        x = np.random.rand(n, 2)
        theta = 0.5
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        y = (R @ x.T).T + np.array([1.0, 2.0])
        aligned = align_sim3(x, y)
        error = float(np.linalg.norm(aligned - y, axis=1).max())
        assert error < 1e-6

    def test_evaluation(self):
        n = 30
        x = np.random.rand(n, 3)  # 3D 预测
        y = x + np.array([1.0, 0.0, 1.0])  # 简单平移
        result = evaluate_sim3_alignment(x, y, warmup=0)
        assert result.post_align_rmse < 0.01
        assert result.pre_align_rmse > 0.5
        assert result.passed


class TestNormalizationLeak:
    def test_independent_distributions(self):
        # 固定值：train mean=5, test mean=6 (dev=20% < 30% → 通过)
        train = np.array([5.0] * 100)
        test = np.array([6.0] * 100)
        r = check_normalization_leak(train, test)
        # 有一定偏差但在正常范围内 → 通过
        assert r.passed, r.detail
        assert r.train_test_deviation < 0.30  # 20% dev

    def test_suspiciously_similar_means(self):
        # 两个数据集（看似不同 stage）但统计量完全相同 → 强烈疑似归一化泄漏
        # 强制让两个分布完全一致（无任何噪声）
        train = np.ones(100) * 5.0
        test = np.ones(100) * 5.0
        r = check_normalization_leak(train, test)
        # 完全相同 → 必然可疑 → 失败
        assert not r.passed


class TestActivationStats:
    def test_4_heads_with_stats(self, tmp_path):
        np.random.seed(42)
        result = compute_activation_stats(
            uwb_scaling=np.random.rand(100),
            uwb_bias=np.random.rand(100) * 0.5,
            vio_scaling=np.random.rand(100) * 0.8,
            risk=np.random.rand(100) * 0.3,
            save_path=tmp_path / "act.json",
        )
        assert result.passed
        assert result.n_4_heads_with_stats == 4
        assert result.activation_heatmap_saved
        assert "uwb_scaling" in result.per_head_stats


class TestBestValCheckpoint:
    def test_normal_convergence(self):
        # val 在 epoch=2 处最低，然后上升 → best_val checkpoint 在 0.8 之前
        val = [5.0, 3.5, 2.8, 3.0, 3.2, 3.5]
        train = [6.0, 4.5, 3.8, 3.2, 2.7, 2.2]
        epochs = [0, 1, 2, 3, 4, 5]
        r = check_best_val_checkpoint(train, val, epochs, use_best_val=True)
        # best_epoch=2 < 80% threshold=4 → passes
        assert r.passed, r.detail
        assert r.best_epoch == 2
        assert r.no_test_set_selection

    def test_suspicious_last_epoch_selection(self):
        # val 持续下降到最后一个 epoch
        val = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5]
        train = [6.0, 4.5, 3.5, 2.5, 1.5, 1.0]
        epochs = [0, 1, 2, 3, 4, 5]
        r = check_best_val_checkpoint(train, val, epochs, use_best_val=False)
        # use_best_val=False → 失败
        assert not r.passed


class TestOverfit:
    def test_normal_training(self):
        # train 收敛，val 不持续上升
        train = [5.0, 4.0, 3.0, 2.5, 2.0, 1.8, 1.7, 1.6, 1.55, 1.5]
        val = [5.5, 4.5, 3.5, 3.0, 2.5, 2.3, 2.2, 2.15, 2.1, 2.1]
        r = check_overfit(train, val, overfit_threshold=2.0)
        assert r.train_loss_converged
        assert r.val_loss_not_rising
        assert r.passed

    def test_overfitting(self):
        # val 持续上升，gap 很大
        train = [5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.3, 0.2, 0.1]
        val = [5.5, 4.5, 3.5, 2.8, 2.5, 2.5, 2.8, 3.5, 4.5, 6.0]
        r = check_overfit(train, val, overfit_threshold=2.0)
        assert not r.passed


class TestGitAndConfigHash:
    def test_git_commit(self):
        h = get_git_commit_hash()
        assert isinstance(h, str)
        # 我们在 git 仓库里
        if h != "no-git":
            assert len(h) >= 7

    def test_config_hash_stable(self):
        cfg = {"lr": 0.001, "epochs": 100, "batch_size": 32}
        h1 = compute_config_hash(cfg)
        h2 = compute_config_hash(cfg)
        assert h1 == h2
        assert len(h1) == 12

    def test_config_hash_order_invariant(self):
        cfg_a = {"a": 1, "b": 2}
        cfg_b = {"b": 2, "a": 1}
        assert compute_config_hash(cfg_a) == compute_config_hash(cfg_b)


# ============================================================================
# precheck_orchestrator 60+ check 函数测试
# ============================================================================


class TestDatasetCorrectnessChecks:
    """P1~P6 数据正确性"""

    def test_P1_pass(self):
        r = check_P1_composition({"composition_counts": {"A2N2": 25, "A2N3": 25, "A3N2": 25, "A3N3": 25}})
        assert r.passed

    def test_P1_fail_unbalanced(self):
        r = check_P1_composition({"composition_counts": {"A2N2": 50, "A2N3": 30, "A3N2": 10, "A3N3": 10}})
        assert not r.passed

    def test_P2_pass(self):
        r = check_P2_missing_source({"missing_source_stats": {"valid_rate": 0.95, "nlos_coverage": 0.4}})
        assert r.passed

    def test_P3_pass(self):
        r = check_P3_m1_burst({"m1_burst_stats": {"mean_gap_s": 0.5, "max_gap_s": 1.0, "missing_rate": 0.05}})
        assert r.passed

    def test_P3_fail_outside_range(self):
        r = check_P3_m1_burst({"m1_burst_stats": {"mean_gap_s": 5.0, "max_gap_s": 6.0, "missing_rate": 0.5}})
        assert not r.passed

    def test_P4_pass(self):
        r = check_P4_k1_gdop({"measured_gdop": 1.21, "expected_gdop": 1.19})
        assert r.passed

    def test_P4_fail(self):
        r = check_P4_k1_gdop({"measured_gdop": 2.5, "expected_gdop": 1.19})
        assert not r.passed

    def test_P5_pass(self):
        r = check_P5_no_out_of_domain({"boundary_violations": 0, "n_sequences": 100})
        assert r.passed

    def test_P5_fail(self):
        r = check_P5_no_out_of_domain({"boundary_violations": 3, "n_sequences": 100})
        assert not r.passed

    def test_P6_pass(self):
        r = check_P6_size_and_split({"n_train_per_seed": 6, "n_test_per_seed": 60, "train_test_overlap": 0})
        assert r.passed

    def test_P6_fail_too_few(self):
        r = check_P6_size_and_split({"n_train_per_seed": 1, "n_test_per_seed": 10, "train_test_overlap": 0})
        assert not r.passed


class TestCoordinateAndImplementationChecks:
    """P7~P13 坐标系/实现"""

    def test_P7_pass(self):
        r = check_P7_rotation_chain({
            "imu_circle_return_passed": True,
            "yaw_direction_correct": True,
            "imu_straight_vs_turn_diff_m": 0.1,
        })
        assert r.passed

    def test_P7_fail(self):
        r = check_P7_rotation_chain({
            "imu_circle_return_passed": False,
            "yaw_direction_correct": True,
            "imu_straight_vs_turn_diff_m": 5.0,
        })
        assert not r.passed

    def test_P8_pass(self):
        r = check_P8_units_and_normalization({
            "unit_check": {"distance_unit": "m", "yaw_unit": "rad", "has_x57_or_x100_artifact": False},
        })
        assert r.passed

    def test_P9_pass(self):
        r = check_P9_sim3_alignment({
            "sim3_alignment_applied": True,
            "pre_align_rmse": 10.0,
            "post_align_rmse": 0.5,
        })
        assert r.passed

    def test_P10_pass(self):
        r = check_P10_2d_only({"has_z_in_2d_metrics": False, "coordinate_dim": 2})
        assert r.passed

    def test_P11_pass(self):
        r = check_P11_input_isolation({
            "input_features": ["range", "imu", "vio", "t"],
        })
        assert r.passed

    def test_P11_fail_leaked(self):
        r = check_P11_input_isolation({
            "input_features": ["range", "imu", "vio", "t", "anchor_x", "anchor_y"],
        })
        assert not r.passed

    def test_P12_pass(self):
        r = check_P12_train_test_symmetry({
            "train_nlos_stats": {"rho": 0.30, "mu_m": 5.0},
            "test_nlos_stats": {"rho": 0.32, "mu_m": 5.1},
        })
        assert r.passed

    def test_P13_pass(self):
        r = check_P13_rq_matching({"ekf_r_std": 0.65, "ekf_q_std": 0.1, "injected_uwb_noise": 0.6})
        assert r.passed


class TestMethodFairnessChecks:
    """P14~P18 方法公平性"""

    def test_P14_pass(self):
        head_dict = {
            "uwb_scaling": 1, "bias": 1, "vio_scaling": 1, "risk": 1
        }
        r = check_P14_shared_4_heads({
            "network_4_heads": {
                "lnn": head_dict, "lstm": head_dict, "transformer": head_dict,
            }
        })
        assert r.passed

    def test_P14_fail_different(self):
        r = check_P14_shared_4_heads({
            "network_4_heads": {
                "lnn": {"uwb_scaling": 1, "bias": 1, "vio_scaling": 1, "risk": 1},
                "lstm": {"uwb_scaling": 1, "bias": 1, "vio_scaling": 1, "risk": 2},
                "transformer": {"uwb_scaling": 1, "bias": 1, "vio_scaling": 1, "risk": 1},
            }
        })
        assert not r.passed

    def test_P15_pass(self):
        r = check_P15_per_method_tuning({"tuning_budget": {"lnn": 50, "lstm": 50, "transformer": 50}})
        assert r.passed

    def test_P16_pass(self):
        r = check_P16_ekf_init_protocol({
            "ekf_trilateration_recovered": True,
            "ekf_delayed_init_works": True,
            "ekf_vio_yaw_used": True,
        })
        assert r.passed

    def test_P17_pass(self):
        r = check_P17_robust_ekf_huber({
            "robust_ekf_huber_applied": True,
            "robust_ekf_not_just_r_scaling": True,
        })
        assert r.passed

    def test_P18_pass(self):
        r = check_P18_baselines_with_heads({"lstm_n_output_heads": 4, "transformer_n_output_heads": 4})
        assert r.passed


class TestStatsScaffoldingChecks:
    """P19~P23 统计脚手架"""

    def test_P19_pass(self):
        r = check_P19_seed_system({"n_seeds": 5, "deterministic_mode": True, "tf32_explicit_set": True})
        assert r.passed

    def test_P20_pass(self):
        r = check_P20_window({
            "window_size": 128, "warmup_s": 10.0,
            "no_window_cross_seq": True, "imu_no_cross_seq_concat": True,
        })
        assert r.passed

    def test_P21_pass(self):
        r = check_P21_normalization_isolation({
            "normalization_uses_train_only": True, "no_test_leak_in_normalization": True,
        })
        assert r.passed

    def test_P22_pass(self):
        r = check_P22_stats_script({
            "has_holm_bonferroni": True, "has_wilcoxon": True,
            "has_4_combo_slicing": True, "has_2x2_anova": True, "has_c1_vs_c4_diag": True,
        })
        assert r.passed

    def test_P23_pass(self):
        r = check_P23_config_consistency({
            "yaml_frozen_axes": {"A2", "A3", "N2", "N3"},
            "manifest_axes": {"A2", "A3", "N2", "N3"},
        })
        assert r.passed


class TestStabilityChecks:
    """P24~P27 稳定性/接口"""

    def test_P24_pass(self):
        r = check_P24_no_nan_inf({"n_nan_detected": 0, "n_inf_detected": 0, "gradient_exploded": False})
        assert r.passed

    def test_P25_pass(self):
        r = check_P25_async_timestamps({
            "timestamps_monotonic": True,
            "async_injected_before_resample": True,
            "jitter_per_packet": True,
        })
        assert r.passed

    def test_P26_pass(self):
        r = check_P26_vio_payload({
            "vio_payload_keys": ["dx", "dy", "dyaw", "quality"],
            "vio_valid_and_scaling_wired": True,
        })
        assert r.passed

    def test_P27_pass(self):
        r = check_P27_activation_stats({"activation_heatmap_saved": True, "n_4_heads_with_stats": 4})
        assert r.passed


class TestDataPipelineChecks:
    """P28~P31 数据管道"""

    def test_P28_pass(self):
        r = check_P28_dual_rate({
            "imu_at_150hz_native": True,
            "imu_not_downsampled": True,
            "resample_method_is_nearest_neighbor": True,
        })
        assert r.passed

    def test_P29_pass(self):
        r = check_P29_windowing({
            "window_no_cross_seq_boundary": True,
            "warmup_excluded_from_eval": True,
            "imu_segments_no_cross_concat": True,
            "short_segment_zero_padded": True,
        })
        assert r.passed

    def test_P30_pass(self):
        r = check_P30_split_before_window({"split_before_windowing": True})
        assert r.passed

    def test_P31_pass(self):
        r = check_P31_edge_cases({"n_zero_div_events": 0, "n_nan_after_edge_case": 0})
        assert r.passed


class TestTrainingConsistencyChecks:
    """P32~P35"""

    def test_P32_pass(self):
        r = check_P32_same_seed_init({
            "lnn_init_seed": 42, "lstm_init_seed": 42, "transformer_init_seed": 42,
        })
        assert r.passed

    def test_P33_pass(self):
        r = check_P33_best_val({
            "uses_best_val_checkpoint": True,
            "no_test_set_selection": True,
        })
        assert r.passed

    def test_P34_pass(self):
        r = check_P34_no_overfit({
            "train_loss_converged": True,
            "val_loss_not_rising": True,
        })
        assert r.passed

    def test_P35_pass(self):
        lstm_layers = {"uwb_scaling": (4, 1), "bias": (4, 1)}
        lnn_layers = {"uwb_scaling": (4, 1), "bias": (4, 1)}
        r = check_P35_head_consistency({
            "lstm_head_layers": lstm_layers, "lnn_head_layers": lnn_layers,
        })
        assert r.passed


class TestEvalDisciplineChecks:
    """P36~P39"""

    def test_P36_pass(self):
        r = check_P36_metric_declaration({
            "report_rmse": True, "report_mean": True, "report_std": True,
            "report_p50": True, "report_p95": True,
            "report_mae": True, "report_trimmed_mean": True, "report_median": True,
            "no_winsorization": True,
        })
        assert r.passed

    def test_P37_pass(self):
        r = check_P37_warmup_removed({"warmup_10s_removed": True})
        assert r.passed

    def test_P38_pass(self):
        r = check_P38_world_frame({"errors_in_world_frame": True})
        assert r.passed

    def test_P39_pass(self):
        r = check_P39_persistence({
            "config_hash_persisted": True, "seed_persisted": True,
            "metrics_persisted": True, "git_commit_persisted": True,
        })
        assert r.passed


class TestPreSeriesChecks:
    """Pre-1~Pre-6"""

    def test_Pre1_pass(self):
        r = check_Pre1_environment_locked({
            "requirements_lock_path": "requirements.lock",
            "cuda_version": "12.1",
            "torch_version": "2.1.0",
        })
        assert r.passed

    def test_Pre2_pass(self):
        r = check_Pre2_decision_log({"decision_log_path": "decision.log", "decision_log_entries": 5})
        assert r.passed

    def test_Pre3_pass(self):
        r = check_Pre3_directory_structure({"run_directories": ["run-2026-01-01-42-liquid-abcdef1234"]})
        assert r.passed

    def test_Pre4_pass(self):
        r = check_Pre4_checksums({"n_files_with_sha256": 100, "n_files_total": 100})
        assert r.passed

    def test_Pre5_pass(self):
        r = check_Pre5_resource_budget({"gpu_model": "RTX 4090", "smoke_unit_seconds": 60.0})
        assert r.passed

    def test_Pre6_pass(self):
        r = check_Pre6_seed_manifest_consistency({
            "n_manifest_seeds": 5, "n_random_sources": 4, "deterministic_mode": True,
        })
        assert r.passed


class TestImplementationCompletionChecks:
    """I-1~I-5"""

    def test_I1_pass(self):
        r = check_I1_data_generator({
            "data_generator_steps_done": [
                "scene_gen", "sensor_sim", "async_inject", "nlos_inject",
                "missing_inject", "resample", "write_npz",
            ]
        })
        assert r.passed

    def test_I2_pass(self):
        r = check_I2_s9_script({
            "s9_report_keys": [
                "composition", "nlos_injection", "missing", "m_burst",
                "gdop", "boundary", "schema",
            ]
        })
        assert r.passed

    def test_I3_pass(self):
        r = check_I3_5_methods({"available_methods": ["lnn", "lstm", "transformer", "ekf", "robust_ekf"]})
        assert r.passed

    def test_I4_pass(self):
        r = check_I4_stats_script({
            "stats_script_outputs": [
                "mean", "std", "p95", "p50", "delta_vs_lnn", "wilcoxon_p", "ci_95",
                "holm_bonferroni", "slicing_4_combo", "anova_2x2", "c1_vs_c4",
            ]
        })
        assert r.passed

    def test_I5_pass(self):
        r = check_I5_eval_pipeline({
            "sim3_alignment": True, "is_2d": True,
            "warmup_removed": True, "world_frame": True,
        })
        assert r.passed


class TestDQQualityGates:
    """DQ-1~DQ-4"""

    def test_DQ1_pass(self):
        r = check_DQ1_difficulty_gradient({"c4_mean_rmse": 5.0, "c1_mean_rmse": 2.0, "target_diff_m": 2.4})
        assert r.passed

    def test_DQ2_pass(self):
        r = check_DQ2_snr({"nlos_bias_m": 5.0, "nlos_std_m": 1.5})
        # SNR = 6.5/0.714 ≈ 9.1, > 3
        assert r.passed

    def test_DQ3_pass(self):
        r = check_DQ3_distribution({
            "train_rho": 0.3, "test_rho": 0.31,
            "train_mu": 5.0, "test_mu": 5.1,
            "train_sigma": 1.5, "test_sigma": 1.55,
        })
        assert r.passed

    def test_DQ4_pass(self):
        r = check_DQ4_sample_size({"n_seeds": 5, "n_test_per_seed": 60})
        assert r.passed

    def test_DQ4_fail(self):
        r = check_DQ4_sample_size({"n_seeds": 2, "n_test_per_seed": 30})
        assert not r.passed


class TestGlobalSummaryChecks:
    """G-1~G-5"""

    def test_G1_pass(self):
        r = check_G1_unit_completeness({"n_units_complete": 25, "n_resumable_pending": 0})
        assert r.passed

    def test_G2_pass(self):
        r = check_G2_metrics_readable({"n_metric_nan": 0, "n_metric_missing_fields": 0})
        assert r.passed

    def test_G3_pass(self):
        r = check_G3_config_data_cross({"n_config_data_mismatch": 0})
        assert r.passed

    def test_G4_pass(self):
        r = check_G4_alerts_cleared({"n_unresolved_alerts": 0})
        assert r.passed

    def test_G5_pass(self):
        r = check_G5_output_aligned({"all_methods_same_test_set": True})
        assert r.passed


class TestExecAuditChecks:
    """E-1~E-6"""

    def test_E1_pass(self):
        r = check_E1_audit_trail({"n_decision_log_entries": 10, "n_git_commits_linked": 5})
        assert r.passed

    def test_E2_pass(self):
        gates = {g: True for g in [
            "Pre-1", "Pre-2", "Pre-3", "Pre-4", "Pre-5", "Pre-6",
            "I-1", "I-2", "I-3", "I-4", "I-5", "S9",
            "G-1", "G-2", "G-3", "G-4", "G-5",
        ]}
        r = check_E2_gate_records({"gate_records": gates})
        assert r.passed

    def test_E3_pass(self):
        r = check_E3_dual_review({"dual_review_completed": True})
        assert r.passed

    def test_E4_pass(self):
        r = check_E4_spot_rerun({"n_spot_rerun": 2, "n_spot_match": 2})
        assert r.passed

    def test_E5_pass(self):
        r = check_E5_alert_log_consistency({"n_alerts_logged": 5, "n_alerts_resolved": 5})
        assert r.passed

    def test_E6_pass(self):
        r = check_E6_audit_exit({
            "e1_passed": True, "e2_passed": True, "e3_passed": True,
            "e4_passed": True, "e5_passed": True,
        })
        assert r.passed


class TestRunAllPrechecks:
    """综合端到端测试"""

    def _build_clean_cfg(self) -> dict:
        return {
            # P1
            "composition_counts": {"A2N2": 25, "A2N3": 25, "A3N2": 25, "A3N3": 25},
            # P2
            "missing_source_stats": {"valid_rate": 0.95, "nlos_coverage": 0.4},
            # P3
            "m1_burst_stats": {"mean_gap_s": 0.5, "max_gap_s": 1.0, "missing_rate": 0.05},
            # P4
            "measured_gdop": 1.19, "expected_gdop": 1.19, "anchor_geometry_class": "asymmetric_4",
            # P5/P6
            "boundary_violations": 0, "n_sequences": 100,
            "n_train_per_seed": 6, "n_test_per_seed": 60, "train_test_overlap": 0,
            # P7
            "imu_circle_return_passed": True, "yaw_direction_correct": True,
            "imu_straight_vs_turn_diff_m": 0.1,
            # P8
            "unit_check": {"distance_unit": "m", "yaw_unit": "rad", "has_x57_or_x100_artifact": False},
            # P9
            "sim3_alignment_applied": True, "pre_align_rmse": 10.0, "post_align_rmse": 0.5,
            # P10
            "has_z_in_2d_metrics": False, "coordinate_dim": 2,
            # P11
            "input_features": ["range", "imu", "vio", "t"],
            # P12
            "train_nlos_stats": {"rho": 0.30, "mu_m": 5.0},
            "test_nlos_stats": {"rho": 0.32, "mu_m": 5.1},
            # P13
            "ekf_r_std": 0.65, "ekf_q_std": 0.1, "injected_uwb_noise": 0.6,
            # P14
            "network_4_heads": {
                "lnn": {"uwb_scaling": 1, "bias": 1, "vio_scaling": 1, "risk": 1},
                "lstm": {"uwb_scaling": 1, "bias": 1, "vio_scaling": 1, "risk": 1},
                "transformer": {"uwb_scaling": 1, "bias": 1, "vio_scaling": 1, "risk": 1},
            },
            # P15
            "tuning_budget": {"lnn": 50, "lstm": 50, "transformer": 50},
            # P16
            "ekf_trilateration_recovered": True, "ekf_delayed_init_works": True, "ekf_vio_yaw_used": True,
            # P17
            "robust_ekf_huber_applied": True, "robust_ekf_not_just_r_scaling": True,
            # P18
            "lstm_n_output_heads": 4, "transformer_n_output_heads": 4,
            # P19
            "n_seeds": 5, "deterministic_mode": True, "tf32_explicit_set": True,
            # P20
            "window_size": 128, "warmup_s": 10.0, "no_window_cross_seq": True, "imu_no_cross_seq_concat": True,
            # P21
            "normalization_uses_train_only": True, "no_test_leak_in_normalization": True,
            # P22
            "has_holm_bonferroni": True, "has_wilcoxon": True,
            "has_4_combo_slicing": True, "has_2x2_anova": True, "has_c1_vs_c4_diag": True,
            # P23
            "yaml_frozen_axes": {"A2", "A3", "N2", "N3"},
            "manifest_axes": {"A2", "A3", "N2", "N3"},
            # P24
            "n_nan_detected": 0, "n_inf_detected": 0, "gradient_exploded": False,
            # P25
            "timestamps_monotonic": True, "async_injected_before_resample": True, "jitter_per_packet": True,
            # P26
            "vio_payload_keys": ["dx", "dy", "dyaw", "quality"], "vio_valid_and_scaling_wired": True,
            # P27
            "activation_heatmap_saved": True, "n_4_heads_with_stats": 4,
            # P28
            "imu_at_150hz_native": True, "imu_not_downsampled": True, "resample_method_is_nearest_neighbor": True,
            # P29
            "window_no_cross_seq_boundary": True, "warmup_excluded_from_eval": True,
            "imu_segments_no_cross_concat": True, "short_segment_zero_padded": True,
            # P30
            "split_before_windowing": True,
            # P31
            "n_zero_div_events": 0, "n_nan_after_edge_case": 0,
            # P32
            "lnn_init_seed": 42, "lstm_init_seed": 42, "transformer_init_seed": 42,
            # P33
            "uses_best_val_checkpoint": True, "no_test_set_selection": True,
            # P34
            "train_loss_converged": True, "val_loss_not_rising": True,
            # P35
            "lstm_head_layers": {"uwb_scaling": (4, 1), "bias": (4, 1)},
            "lnn_head_layers": {"uwb_scaling": (4, 1), "bias": (4, 1)},
            # P36
            "report_rmse": True, "report_mean": True, "report_std": True,
            "report_p50": True, "report_p95": True,
            "report_mae": True, "report_trimmed_mean": True, "report_median": True,
            "no_winsorization": True,
            # P37/P38
            "warmup_10s_removed": True, "errors_in_world_frame": True,
            # P39
            "config_hash_persisted": True, "seed_persisted": True,
            "metrics_persisted": True, "git_commit_persisted": True,
            # Pre-1~6
            "requirements_lock_path": "r.lock", "cuda_version": "12.1", "torch_version": "2.1.0",
            "decision_log_path": "log", "decision_log_entries": 5,
            "run_directories": ["run-2026-01-01-42-liquid-abcdef1234"],
            "n_files_with_sha256": 100, "n_files_total": 100,
            "gpu_model": "RTX 4090", "smoke_unit_seconds": 60.0,
            "n_manifest_seeds": 5, "n_random_sources": 4,
            # I-1~5
            "data_generator_steps_done": [
                "scene_gen", "sensor_sim", "async_inject", "nlos_inject",
                "missing_inject", "resample", "write_npz",
            ],
            "s9_report_keys": [
                "composition", "nlos_injection", "missing", "m_burst",
                "gdop", "boundary", "schema",
            ],
            "available_methods": ["lnn", "lstm", "transformer", "ekf", "robust_ekf"],
            "stats_script_outputs": [
                "mean", "std", "p95", "p50", "delta_vs_lnn", "wilcoxon_p", "ci_95",
                "holm_bonferroni", "slicing_4_combo", "anova_2x2", "c1_vs_c4",
            ],
            "sim3_alignment": True, "is_2d": True, "warmup_removed": True, "world_frame": True,
            # DQ-1~4
            "c4_mean_rmse": 5.0, "c1_mean_rmse": 2.0, "target_diff_m": 2.4,
            "nlos_bias_m": 5.0, "nlos_std_m": 1.5,
            "train_rho": 0.3, "test_rho": 0.31, "train_mu": 5.0, "test_mu": 5.1,
            "train_sigma": 1.5, "test_sigma": 1.55,
            # G-1~5
            "n_units_complete": 25, "n_resumable_pending": 0,
            "n_metric_nan": 0, "n_metric_missing_fields": 0,
            "n_config_data_mismatch": 0, "n_unresolved_alerts": 0,
            "all_methods_same_test_set": True,
            # E-1~6
            "n_decision_log_entries": 10, "n_git_commits_linked": 5,
            "gate_records": {g: True for g in [
                "Pre-1", "Pre-2", "Pre-3", "Pre-4", "Pre-5", "Pre-6",
                "I-1", "I-2", "I-3", "I-4", "I-5", "S9",
                "G-1", "G-2", "G-3", "G-4", "G-5",
            ]},
            "dual_review_completed": True,
            "n_spot_rerun": 2, "n_spot_match": 2,
            "n_alerts_logged": 5, "n_alerts_resolved": 5,
            "e1_passed": True, "e2_passed": True, "e3_passed": True,
            "e4_passed": True, "e5_passed": True,
        }

    def test_full_clean_run(self):
        cfg = self._build_clean_cfg()
        report = run_all_prechecks(cfg)
        assert report.total == 65  # 39 P + 6 Pre + 5 I + 4 DQ + 5 G + 6 E = 65
        # 39 + 6 + 5 + 4 + 5 + 6 = 65
        assert report.total == 65
        assert report.n_passed == 65
        assert report.n_failed == 0
        assert report.overall_pass is True

    def test_dirty_run_fails(self):
        cfg = self._build_clean_cfg()
        # 故意破坏几项
        cfg["composition_counts"] = {"A2N2": 100, "A2N3": 0, "A3N2": 0, "A3N3": 0}  # P1 fail
        cfg["input_features"] = ["range", "anchor_x"]  # P11 leak
        cfg["n_seeds"] = 2  # P19 fail
        report = run_all_prechecks(cfg)
        assert report.overall_pass is False
        assert report.n_failed >= 3
        # 检查具体哪些失败
        failed_codes = {r.code for r in report.results if not r.passed}
        assert "P1" in failed_codes
        assert "P11" in failed_codes
        assert "P19" in failed_codes

    def test_soft_failure_does_not_block(self):
        cfg = self._build_clean_cfg()
        # P2 是 soft（缺数据时 severity=soft），不阻塞
        cfg.pop("missing_source_stats", None)
        report = run_all_prechecks(cfg)
        # P2 应 fail 但是 soft，n_hard_failed 应仍为 0
        assert report.overall_pass is True
        p2_result = next(r for r in report.results if r.code == "P2")
        assert not p2_result.passed
        assert p2_result.severity == "soft"

    def test_report_to_dict(self):
        cfg = self._build_clean_cfg()
        report = run_all_prechecks(cfg)
        d = report.to_dict()
        assert "total" in d
        assert "results" in d
        assert isinstance(d["results"], list)
        assert all("code" in r and "passed" in r for r in d["results"])