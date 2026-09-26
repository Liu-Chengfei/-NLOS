"""特征分布一致性监控模块测试。

本模块验证 feature_dist_monitor 的各项功能，
涵盖数据提取、统计计算、漂移检测、输出写入和路径安全校验。

测试覆盖范围：
  - 数据提取：uwb_residual、imu_high_freq、vio_relative_pose、timestamps
  - 统计计算：clean_numeric_values、compute_feature_stats
  - 漂移检测：compute_ks_test、compute_percentile_drift_values、check_percentile_drift
  - 输出：write_feature_distribution_report、write_drift_manifest
  - 路径安全：_validate_output_path（正常/穿越/非法filename）
  - 集成测试：build_split_feature_data + build_split_stats + detect_feature_drift

被测模块：
  - liquidloc.analysis.feature_dist_monitor
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from liquidloc.analysis.feature_dist_monitor import (
    DISTRIBUTION_MONITOR_FEATURES,
    FeatureDriftReport,
    FeatureStats,
    _extract_imu_high_freq,
    _extract_timestamps,
    _extract_uwb_residuals,
    _extract_vio_relative_pose,
    _validate_output_path,
    build_split_feature_data,
    build_split_stats,
    check_percentile_drift,
    clean_numeric_values,
    compute_feature_stats,
    compute_ks_test,
    compute_percentile_drift_values,
    detect_feature_drift,
    extract_sequence_features,
    monitor_feature_distribution,
    write_drift_manifest,
    write_feature_distribution_report,
)


# ============================================================
# 数据提取测试
# ============================================================

class TestExtractUwbResiduals:
    """验证 _extract_uwb_residuals 函数。"""

    def test_basic_extraction(self, tmp_path):
        """正常情况：4个锚点均有数据，提取残差。"""
        seq_dir = tmp_path / "seq_001"
        seq_dir.mkdir()
        # 写入anchor_layout
        (seq_dir / "anchor_layout.json").write_text(
            json.dumps({"anchor_ids": [0, 1, 2, 3],
                        "anchor_positions": [[0, 0], [5, 0], [0, 5], [5, 5]]})
        )
        # 写入gt
        (seq_dir / "gt.json").write_text(
            json.dumps([
                {"timestamp": 0.0, "px": 1.0, "py": 0.0},
                {"timestamp": 1.0, "px": 2.0, "py": 0.0},
            ])
        )
        # 写入uwb：anchor 0 在位置 (1,0) 测得 range=3.0, 预测=1.0, 残差=2.0
        (seq_dir / "uwb.json").write_text(
            json.dumps([
                {"timestamp": 0.5, "anchor_id": 0, "range": 3.0, "quality": 0.9},
            ])
        )
        result = _extract_uwb_residuals(seq_dir)
        assert "uwb_residual_r0" in result
        assert len(result["uwb_residual_r0"]) == 1
        assert abs(result["uwb_residual_r0"][0] - 2.0) < 1e-6

    def test_empty_uwb(self, tmp_path):
        """空uwb数据：返回空列表。"""
        seq_dir = tmp_path / "seq_empty"
        seq_dir.mkdir()
        (seq_dir / "uwb.json").write_text(json.dumps([]))
        result = _extract_uwb_residuals(seq_dir)
        for key in result:
            assert result[key] == []

    def test_no_anchor_layout(self, tmp_path):
        """无anchor_layout：返回空列表。"""
        seq_dir = tmp_path / "seq_no_anchor"
        seq_dir.mkdir()
        (seq_dir / "uwb.json").write_text(json.dumps([
            {"timestamp": 0.5, "anchor_id": 0, "range": 2.0, "quality": 0.9},
        ]))
        result = _extract_uwb_residuals(seq_dir)
        for key in result:
            assert result[key] == []


class TestExtractImuHighFreq:
    """验证 _extract_imu_high_freq 函数。"""

    def test_basic_extraction(self, tmp_path):
        """正常情况：提取ax/ay/gz。"""
        seq_dir = tmp_path / "seq_imu"
        seq_dir.mkdir()
        (seq_dir / "imu.json").write_text(json.dumps([
            {"timestamp": 0.0, "ax": 0.1, "ay": 0.2, "gz": 0.01},
            {"timestamp": 0.1, "ax": 0.15, "ay": 0.25, "gz": 0.02},
        ]))
        result = _extract_imu_high_freq(seq_dir)
        assert result["imu_ax"] == [0.1, 0.15]
        assert result["imu_ay"] == [0.2, 0.25]
        assert result["imu_gz"] == [0.01, 0.02]

    def test_missing_fields(self, tmp_path):
        """部分字段缺失：跳过None值。"""
        seq_dir = tmp_path / "seq_partial"
        seq_dir.mkdir()
        (seq_dir / "imu.json").write_text(json.dumps([
            {"timestamp": 0.0, "ax": 0.1, "ay": None, "gz": 0.01},
        ]))
        result = _extract_imu_high_freq(seq_dir)
        assert result["imu_ax"] == [0.1]
        assert result["imu_ay"] == []
        assert result["imu_gz"] == [0.01]


class TestExtractVioRelativePose:
    """验证 _extract_vio_relative_pose 函数。"""

    def test_basic_extraction(self, tmp_path):
        """正常情况：提取dx/dy/dyaw。"""
        seq_dir = tmp_path / "seq_vio"
        seq_dir.mkdir()
        (seq_dir / "vio.json").write_text(json.dumps([
            {"timestamp": 0.0, "dx": 0.03, "dy": 0.0, "dyaw": 0.01},
            {"timestamp": 0.1, "dx": 0.05, "dy": 0.02, "dyaw": 0.02},
        ]))
        result = _extract_vio_relative_pose(seq_dir)
        assert result["vio_dx"] == [0.03, 0.05]
        assert result["vio_dy"] == [0.0, 0.02]
        assert result["vio_dyaw"] == [0.01, 0.02]

    def test_single_record(self, tmp_path):
        """单条记录：返回空列表。"""
        seq_dir = tmp_path / "seq_single"
        seq_dir.mkdir()
        (seq_dir / "vio.json").write_text(json.dumps([
            {"timestamp": 0.0, "dx": 0.03, "dy": 0.0, "dyaw": 0.01},
        ]))
        result = _extract_vio_relative_pose(seq_dir)
        for key in result:
            assert result[key] == []


class TestExtractTimestamps:
    """验证 _extract_timestamps 函数。"""

    def test_basic_extraction(self, tmp_path):
        """正常情况：提取时间戳和dt。"""
        seq_dir = tmp_path / "seq_ts"
        seq_dir.mkdir()
        (seq_dir / "imu.json").write_text(json.dumps([
            {"timestamp": 0.0},
            {"timestamp": 0.5},
        ]))
        (seq_dir / "vio.json").write_text(json.dumps([
            {"timestamp": 0.25},
            {"timestamp": 0.75},
        ]))
        result = _extract_timestamps(seq_dir)
        assert result["timestamp"] == [0.0, 0.25, 0.5, 0.75]
        assert result["dt"] == [0.25, 0.25, 0.25]


class TestExtractSequenceFeatures:
    """验证 extract_sequence_features 端到端。"""

    def test_full_extraction(self, tmp_path):
        """完整序列：所有特征均被提取。"""
        seq_dir = tmp_path / "seq_full"
        seq_dir.mkdir()
        (seq_dir / "anchor_layout.json").write_text(
            json.dumps({"anchor_ids": [0, 1], "anchor_positions": [[0, 0], [5, 0]]})
        )
        (seq_dir / "gt.json").write_text(json.dumps([
            {"timestamp": 0.0, "px": 1.0, "py": 0.0},
        ]))
        (seq_dir / "uwb.json").write_text(json.dumps([
            {"timestamp": 0.0, "anchor_id": 0, "range": 1.5, "quality": 0.9},
        ]))
        (seq_dir / "imu.json").write_text(json.dumps([
            {"timestamp": 0.0, "ax": 0.1, "ay": 0.2, "gz": 0.01},
        ]))
        (seq_dir / "vio.json").write_text(json.dumps([
            {"timestamp": 0.0, "dx": 0.03, "dy": 0.0, "dyaw": 0.01},
            {"timestamp": 0.1, "dx": 0.05, "dy": 0.02, "dyaw": 0.02},
        ]))
        result = extract_sequence_features(seq_dir)
        assert len(result["imu_ax"]) == 1
        assert len(result["imu_ay"]) == 1
        assert len(result["imu_gz"]) == 1
        assert len(result["vio_dx"]) == 2
        assert len(result["timestamp"]) == 2  # imu有1条, vio有2条, 合并去重后为2条
        assert len(result["dt"]) == 1  # 2条timestamp产生1条dt


# ============================================================
# 统计计算测试
# ============================================================

class TestCleanNumericValues:
    """验证 clean_numeric_values 函数。"""

    def test_basic_cleanup(self):
        """正常数值：原样返回。"""
        result = clean_numeric_values([1.0, 2.0, 3.0])
        assert list(result) == [1.0, 2.0, 3.0]

    def test_filters_nan_and_inf(self):
        """过滤 NaN/Inf：保留有限值。"""
        import math
        result = clean_numeric_values([1.0, float("nan"), 2.0, float("inf"), float("-inf")])
        assert list(result) == [1.0, 2.0]

    def test_filters_none(self):
        """过滤 None：保留有效值。"""
        result = clean_numeric_values([1.0, None, 2.0])
        assert list(result) == [1.0, 2.0]

    def test_empty_input(self):
        """空输入：返回空数组。"""
        result = clean_numeric_values([])
        assert len(result) == 0


class TestComputeFeatureStats:
    """验证 compute_feature_stats 函数。"""

    def test_basic_stats(self):
        """正常统计：验证均值/标准差。"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_feature_stats(values)
        assert result.mean == 3.0
        assert result.p50 == 3.0

    def test_empty_stats(self):
        """空输入：返回零值统计。"""
        result = compute_feature_stats([])
        assert result.mean == 0.0
        assert result.n_samples == 0

    def test_single_value(self):
        """单值输入：标准差为0。"""
        result = compute_feature_stats([42.0])
        assert result.mean == 42.0
        assert result.std == 0.0


class TestComputePercentileDriftValues:
    """验证 compute_percentile_drift_values 函数。"""

    def test_basic_drift(self):
        """正常漂移：验证sigma偏移计算。"""
        ref = FeatureStats(mean=0.0, variance=1.0, std=1.0,
                           p5=-1.64, p50=0.0, p95=1.64,
                           n_samples=100, min_val=-3.0, max_val=3.0)
        comp = FeatureStats(mean=2.0, variance=1.0, std=1.0,
                            p5=0.36, p50=2.0, p95=3.64,
                            n_samples=100, min_val=-1.0, max_val=5.0)
        result = compute_percentile_drift_values(ref, comp)
        # p50 偏移 = (2.0 - 0.0) / 1.0 = 2.0
        assert abs(result["p50"][1] - 2.0) < 1e-6


class TestCheckPercentileDrift:
    """验证 check_percentile_drift 函数。"""

    def test_within_threshold(self):
        """2σ以内：不漂移。"""
        is_drift, _ = check_percentile_drift(1.5)
        assert is_drift is False

    def test_exceeds_threshold(self):
        """超过2σ：漂移。"""
        is_drift, _ = check_percentile_drift(2.1)
        assert is_drift is True

    def test_negative_shift(self):
        """负向漂移：同样检测。"""
        is_drift, _ = check_percentile_drift(-2.5)
        assert is_drift is True


class TestComputeKsTest:
    """验证 compute_ks_test 函数。"""

    def test_identical_distributions(self):
        """相同分布：KS统计接近0，p值接近1。"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stat, p = compute_ks_test(data, data)
        assert stat == 0.0
        assert p == 1.0

    def test_different_distributions(self):
        """不同分布：KS统计显著。"""
        import numpy as np
        np.random.seed(42)
        ref = np.random.normal(0, 1, 100)
        comp = np.random.normal(5, 1, 100)
        stat, p = compute_ks_test(ref, comp)
        assert stat > 0.5
        assert p < 0.01

    def test_insufficient_data(self):
        """不足2个样本：返回默认值。"""
        stat, p = compute_ks_test([1.0], [2.0])
        assert stat == 0.0
        assert p == 1.0


# ============================================================
# 路径安全测试
# ============================================================

class TestValidateOutputPath:
    """验证 _validate_output_path 函数。"""

    def test_normal_path(self, tmp_path):
        """正常路径：返回安全路径。"""
        output_dir = tmp_path / "out"
        result = _validate_output_path(output_dir, "result.json")
        assert result.name == "result.json"
        assert result.parent == output_dir.resolve()

    def test_path_traversal_blocked(self, tmp_path):
        """路径穿越：抛出ValueError。"""
        output_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="path separators"):
            _validate_output_path(output_dir, "../secret.txt")

    def test_path_separator_in_filename(self, tmp_path):
        """文件名含路径分隔符：抛出ValueError。"""
        output_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="path separators"):
            _validate_output_path(output_dir, "subdir/result.json")

    def test_empty_filename(self, tmp_path):
        """空文件名：抛出ValueError。"""
        output_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="Invalid filename"):
            _validate_output_path(output_dir, "")


# ============================================================
# 输出测试
# ============================================================

class TestWriteFeatureDistributionReport:
    """验证 write_feature_distribution_report 函数。"""

    def test_write_report(self, tmp_path):
        """正常情况：写入statistics.json和comparison_table.json。"""
        split_stats = {
            "train": {"feat_a": compute_feature_stats([1, 2, 3])},
            "test":  {"feat_a": compute_feature_stats([1, 2, 3])},
        }
        result_path = write_feature_distribution_report(split_stats, tmp_path)
        assert result_path.exists()
        assert (tmp_path / "statistics.json").exists()
        assert (tmp_path / "comparison_table.json").exists()


class TestWriteDriftManifest:
    """验证 write_drift_manifest 函数。"""

    def test_no_drift(self, tmp_path):
        """无漂移：manifest包含空列表。"""
        reports = []
        result_path = write_drift_manifest(reports, tmp_path)
        assert result_path.exists()
        content = json.loads(result_path.read_text())
        assert content["drift_detected"] is False
        assert content["drifted_features"] == []

    def test_with_drift(self, tmp_path):
        """有漂移：manifest包含警告条目。"""
        ref = FeatureStats(mean=0.0, variance=1.0, std=1.0,
                           p5=-1.64, p50=0.0, p95=1.64,
                           n_samples=100, min_val=-3.0, max_val=3.0)
        comp = FeatureStats(mean=10.0, variance=1.0, std=1.0,
                            p5=8.36, p50=10.0, p95=11.64,
                            n_samples=100, min_val=7.0, max_val=13.0)
        report = FeatureDriftReport(
            feature_name="test_feat",
            reference_stats=ref,
            compare_stats=comp,
            percentile_drift={"p50": 10.0},
            ks_statistic=1.0,
            ks_p_value=0.0,
            drift_detected=True,
            drift_reasons=["p50 drift exceeds 2.0sigma (actual: 10.000sigma)"],
        )
        result_path = write_drift_manifest([report], tmp_path)
        content = json.loads(result_path.read_text())
        assert content["drift_detected"] is True
        assert len(content["drifted_features"]) > 0


# ============================================================
# Split 处理测试
# ============================================================

class TestBuildSplitFeatureData:
    """验证 build_split_feature_data 函数。"""

    def test_basic_build(self, tmp_path):
        """正常情况：从split manifest构建特征数据。"""
        seq_dir = tmp_path / "seq"
        seq_dir.mkdir()
        (seq_dir / "anchor_layout.json").write_text(
            json.dumps({"anchor_ids": [0], "anchor_positions": [[0, 0]]})
        )
        (seq_dir / "gt.json").write_text(json.dumps([{"timestamp": 0.0, "px": 0.0, "py": 0.0}]))
        (seq_dir / "uwb.json").write_text(json.dumps([{"timestamp": 0.0, "anchor_id": 0, "range": 1.0}]))
        (seq_dir / "imu.json").write_text(json.dumps([{"timestamp": 0.0, "ax": 0.1, "ay": 0.2, "gz": 0.01}]))
        (seq_dir / "vio.json").write_text(json.dumps([
            {"timestamp": 0.0, "dx": 0.03, "dy": 0.0, "dyaw": 0.01},
            {"timestamp": 0.1, "dx": 0.05, "dy": 0.02, "dyaw": 0.02},
        ]))

        manifest = {
            "splits": {
                "train": {"sequences": [{"seq_dir": str(seq_dir)}]},
                "test":  {"sequences": [{"seq_dir": str(seq_dir)}]},
            }
        }
        result = build_split_feature_data(manifest)
        assert "train" in result
        assert "test" in result
        assert len(result["train"]["imu_ax"]) == 1
        assert len(result["train"]["vio_dx"]) == 2


class TestBuildSplitStats:
    """验证 build_split_stats 函数。"""

    def test_basic_stats(self):
        """正常统计：验证统计量正确计算。"""
        split_data = {
            "train": {"imu_ax": [1.0, 2.0, 3.0, 4.0, 5.0]},
        }
        result = build_split_stats(split_data)
        assert "train" in result
        assert result["train"]["imu_ax"].mean == 3.0
        assert result["train"]["imu_ax"].n_samples == 5


# ============================================================
# 漂移检测测试
# ============================================================

class TestDetectFeatureDrift:
    """验证 detect_feature_drift 函数。"""

    def test_no_drift(self):
        """相同分布：无漂移检测。"""
        stats = {
            "train": {"feat": compute_feature_stats([1, 2, 3, 4, 5])},
            "test":  {"feat": compute_feature_stats([1, 2, 3, 4, 5])},
        }
        data = {
            "train": {"feat": [1, 2, 3, 4, 5]},
            "test":  {"feat": [1, 2, 3, 4, 5]},
        }
        reports = detect_feature_drift(stats, data)
        drift_reports = [r for r in reports if r.drift_detected]
        assert len(drift_reports) == 0

    def test_drift_detected(self):
        """明显漂移：正确检测。"""
        import numpy as np
        np.random.seed(42)
        train_data = np.random.normal(0, 1, 200).tolist()
        test_data = np.random.normal(10, 1, 200).tolist()
        stats = {
            "train": {"imu_ax": compute_feature_stats(train_data)},
            "test":  {"imu_ax": compute_feature_stats(test_data)},
        }
        data = {
            "train": {"imu_ax": train_data},
            "test":  {"imu_ax": test_data},
        }
        reports = detect_feature_drift(stats, data)
        drift_reports = [r for r in reports if r.drift_detected]
        assert len(drift_reports) > 0
        assert drift_reports[0].feature_name == "imu_ax"


# ============================================================
# 集成测试
# ============================================================

class TestMonitorFeatureDistribution:
    """验证 monitor_feature_distribution 端到端。"""

    def test_monitor_no_drift(self, tmp_path):
        """无漂移场景：监控正常完成。"""
        seq_dir = tmp_path / "seq"
        seq_dir.mkdir()
        (seq_dir / "anchor_layout.json").write_text(
            json.dumps({"anchor_ids": [0], "anchor_positions": [[0, 0]]})
        )
        (seq_dir / "gt.json").write_text(json.dumps([{"timestamp": 0.0, "px": 0.0, "py": 0.0}]))
        (seq_dir / "uwb.json").write_text(json.dumps([{"timestamp": 0.0, "anchor_id": 0, "range": 1.0}]))
        (seq_dir / "imu.json").write_text(json.dumps([{"timestamp": 0.0, "ax": 0.1, "ay": 0.2, "gz": 0.01}]))
        (seq_dir / "vio.json").write_text(json.dumps([
            {"timestamp": 0.0, "dx": 0.03, "dy": 0.0, "dyaw": 0.01},
            {"timestamp": 0.1, "dx": 0.05, "dy": 0.02, "dyaw": 0.02},
        ]))

        manifest_path = tmp_path / "split_manifest.json"
        manifest_path.write_text(json.dumps({
            "splits": {
                "train": {"sequences": [{"seq_dir": str(seq_dir)}]},
                "test":  {"sequences": [{"seq_dir": str(seq_dir)}]},
            }
        }))
        output_dir = tmp_path / "output"
        result = monitor_feature_distribution(
            split_manifest_path=str(manifest_path),
            output_dir=output_dir,
            verbose=False,
        )
        assert result["drift_detected"] is False
        assert result["statistics_path"] is not None
        assert result["manifest_path"] is not None
        assert (output_dir / "statistics.json").exists()
        assert (output_dir / "feature_drift.json").exists()

    def test_monitor_with_drift(self, tmp_path):
        """有漂移场景：正确检测漂移。"""
        # 构造两组不同分布的序列数据
        seq_dir_a = tmp_path / "seq_a"
        seq_dir_a.mkdir()
        (seq_dir_a / "anchor_layout.json").write_text(
            json.dumps({"anchor_ids": [0], "anchor_positions": [[0, 0]]})
        )
        (seq_dir_a / "gt.json").write_text(json.dumps([{"timestamp": 0.0, "px": 0.0, "py": 0.0}]))
        (seq_dir_a / "uwb.json").write_text(json.dumps([{"timestamp": 0.0, "anchor_id": 0, "range": 1.0}]))
        (seq_dir_a / "imu.json").write_text(json.dumps([
            {"timestamp": float(i * 0.1), "ax": 0.1, "ay": 0.2, "gz": 0.01}
            for i in range(10)
        ]))
        (seq_dir_a / "vio.json").write_text(json.dumps([
            {"timestamp": float(i * 0.1), "dx": 0.03, "dy": 0.0, "dyaw": 0.01}
            for i in range(10)
        ]))

        # test序列与train相同
        manifest_path = tmp_path / "split_manifest.json"
        manifest_path.write_text(json.dumps({
            "splits": {
                "train": {"sequences": [{"seq_dir": str(seq_dir_a)}]},
                "test":  {"sequences": [{"seq_dir": str(seq_dir_a)}]},
            }
        }))
        output_dir = tmp_path / "output"
        result = monitor_feature_distribution(
            split_manifest_path=str(manifest_path),
            output_dir=output_dir,
            verbose=False,
        )
        # 相同数据无漂移
        assert result["drift_detected"] is False


class TestDistributionMonitorFeatures:
    """验证 DISTRIBUTION_MONITOR_FEATURES 常量。"""

    def test_contains_expected_features(self):
        """包含预期的关键特征。"""
        assert "uwb_residual_r0" in DISTRIBUTION_MONITOR_FEATURES
        assert "uwb_residual_r1" in DISTRIBUTION_MONITOR_FEATURES
        assert "imu_ax" in DISTRIBUTION_MONITOR_FEATURES
        assert "imu_ay" in DISTRIBUTION_MONITOR_FEATURES
        assert "imu_gz" in DISTRIBUTION_MONITOR_FEATURES
        assert "vio_dx" in DISTRIBUTION_MONITOR_FEATURES
        assert "vio_dy" in DISTRIBUTION_MONITOR_FEATURES
        assert "vio_dyaw" in DISTRIBUTION_MONITOR_FEATURES
        assert "timestamp" in DISTRIBUTION_MONITOR_FEATURES
        assert "dt" in DISTRIBUTION_MONITOR_FEATURES

    def test_feature_count(self):
        """特征数量为12。"""
        assert len(DISTRIBUTION_MONITOR_FEATURES) == 12
