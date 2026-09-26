"""轨迹多样性校验模块测试。

本模块验证方位角均匀性断言和速度占比一致性断言函数的正确性，
涵盖正常情况、边界情况和异常情况。

测试覆盖范围：
  - 方位角均匀性断言：均匀分布通过、单 bin 超限失败、全零失败、总和不为 1 失败
  - 速度占比一致性断言：完全匹配通过、单 bin 超限失败、训练为 0 测试也为 0 通过
  - 方位角直方图计算：8 bin 输出、proportions 求和为 1
  - 速度分布计算：5 bin 输出、全区间覆盖检查
  - 已知分布验证断言触发：构造极端偏离分布验证 assertion 正确触发
  - 端到端集成测试：合成轨迹数据验证完整 manifest 生成流程

被测模块：
  - liquidloc.analysis.traj_diversity
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from liquidloc.analysis.traj_diversity import (
    AZIMUTH_BIN_COUNT,
    AZIMUTH_MAX_DEVIATION,
    SPEED_BIN_COUNT,
    SPEED_PROPORTION_TOLERANCE,
    SPEED_RANGE,
    _assert_azimuth_uniformity,
    _assert_speed_proportion_consistency,
    _compute_azimuth_histogram,
    _compute_speed_distribution,
    _extract_gt_features,
    compute_traj_diversity_manifest,
    write_traj_diversity_manifest,
)


# ============================================================
# 方位角均匀性断言测试
# ============================================================

class TestAssertAzimuthUniformity:
    """验证 _assert_azimuth_uniformity 在各种输入下的行为。"""

    def test_uniform_distribution_passes(self):
        """均匀分布（各 bin = 1/8）应通过断言。"""
        proportions = [1.0 / AZIMUTH_BIN_COUNT] * AZIMUTH_BIN_COUNT
        result = _assert_azimuth_uniformity(proportions)
        assert result["passed"] is True
        assert result["max_deviation"] == 0.0
        assert result["mean_proportion"] == 1.0 / AZIMUTH_BIN_COUNT

    def test_single_bin_exceeds_deviation_fails(self):
        """单个 bin 偏离均值超过 20% 应失败。"""
        # 构造极端偏离：bin 0 = 0.30, 其余平分剩余 0.70
        proportions = [0.30] + [0.70 / (AZIMUTH_BIN_COUNT - 1)] * (AZIMUTH_BIN_COUNT - 1)
        result = _assert_azimuth_uniformity(proportions)
        assert result["passed"] is False
        assert result["max_deviation"] > AZIMUTH_MAX_DEVIATION
        assert len(result["violations"]) > 0
        # 确认违规 bin 是 bin 0
        assert result["violations"][0]["bin_index"] == 0
        assert result["violations"][0]["direction"] == "N"

    def test_deviation_within_tolerance_passes(self):
        """偏离在容忍范围内应通过。"""
        # 构造偏离约 4% 的分布（在 20% 容忍内）
        # bin 0,1 略高（0.13），bin 2,3 略低（0.12），其余为均值 0.125
        proportions = [0.13, 0.13, 0.12, 0.12, 0.125, 0.125, 0.125, 0.125]
        # 总和 = 1.0，无需归一化
        result = _assert_azimuth_uniformity(proportions)
        assert result["passed"] is True

    def test_all_zeros_fails(self):
        """全零比例应失败。"""
        proportions = [0.0] * AZIMUTH_BIN_COUNT
        result = _assert_azimuth_uniformity(proportions)
        assert result["passed"] is False
        # 全零时 proportions 总和为 0，不等于 1.0，提前返回 max_deviation=None
        assert result["max_deviation"] is None

    def test_proportions_not_sum_to_one_fails(self):
        """比例总和不等于 1 应失败。"""
        proportions = [0.2] * AZIMUTH_BIN_COUNT  # 总和 = 1.6
        result = _assert_azimuth_uniformity(proportions)
        assert result["passed"] is False
        assert "sum to" in result["reason"]

    def test_wrong_bin_count_fails(self):
        """bin 数量不匹配应失败。"""
        proportions = [0.25, 0.25, 0.25, 0.25]  # 4 bin 而非 8
        result = _assert_azimuth_uniformity(proportions)
        assert result["passed"] is False
        assert "bin count mismatch" in result["reason"]

    def test_exactly_at_threshold_passes(self):
        """偏离恰好等于阈值（20%）应通过（≤20%）。"""
        mean_p = 1.0 / AZIMUTH_BIN_COUNT
        # 构造偏离恰好 20% 的分布
        target_deviation = AZIMUTH_MAX_DEVIATION  # 0.20
        # bin 0 = mean * (1 + 0.20), 其余平分剩余
        bin0 = mean_p * (1.0 + target_deviation)
        remaining = 1.0 - bin0
        other_mean = remaining / (AZIMUTH_BIN_COUNT - 1)
        proportions = [bin0] + [other_mean] * (AZIMUTH_BIN_COUNT - 1)
        result = _assert_azimuth_uniformity(proportions)
        # 恰好 20% 应在容忍范围内
        assert result["passed"] is True


# ============================================================
# 速度占比一致性断言测试
# ============================================================

class TestAssertSpeedProportionConsistency:
    """验证 _assert_speed_proportion_consistency 在各种输入下的行为。"""

    def test_identical_proportions_passes(self):
        """训练与测试占比完全一致应通过。"""
        train_props = [0.20, 0.20, 0.20, 0.20, 0.20]
        test_props = [0.20, 0.20, 0.20, 0.20, 0.20]
        result = _assert_speed_proportion_consistency(train_props, test_props)
        assert result["passed"] is True
        assert result["max_deviation"] == 0.0

    def test_single_bin_exceeds_deviation_fails(self):
        """单个 bin 偏离训练占比超过 20% 应失败。"""
        train_props = [0.20, 0.20, 0.20, 0.20, 0.20]
        # bin 0 偏离 25%（>20% 容忍）
        test_props = [0.25, 0.1875, 0.1875, 0.1875, 0.1875]
        # 归一化
        total = sum(test_props)
        test_props = [p / total for p in test_props]
        result = _assert_speed_proportion_consistency(train_props, test_props)
        assert result["passed"] is False
        assert result["max_deviation"] > SPEED_PROPORTION_TOLERANCE
        assert len(result["violations"]) > 0

    def test_deviation_within_tolerance_passes(self):
        """偏离在容忍范围内应通过。"""
        train_props = [0.20, 0.20, 0.20, 0.20, 0.20]
        # bin 0 偏离约 15%（在 20% 容忍内）
        test_props = [0.23, 0.1925, 0.1925, 0.1925, 0.1925]
        total = sum(test_props)
        test_props = [p / total for p in test_props]
        result = _assert_speed_proportion_consistency(train_props, test_props)
        assert result["passed"] is True

    def test_train_zero_test_zero_passes(self):
        """训练 bin 为 0，测试 bin 也为 0 应通过。"""
        train_props = [0.30, 0.30, 0.30, 0.10, 0.00]
        test_props = [0.30, 0.30, 0.30, 0.10, 0.00]
        result = _assert_speed_proportion_consistency(train_props, test_props)
        assert result["passed"] is True

    def test_train_zero_test_nonzero_fails(self):
        """训练 bin 为 0，测试 bin 非 0 应失败。"""
        train_props = [0.25, 0.25, 0.25, 0.25, 0.00]
        test_props = [0.20, 0.20, 0.20, 0.20, 0.20]
        result = _assert_speed_proportion_consistency(train_props, test_props)
        assert result["passed"] is False

    def test_wrong_bin_count_fails(self):
        """bin 数量不匹配应失败。"""
        train_props = [0.25, 0.25, 0.25, 0.25]
        test_props = [0.20, 0.20, 0.20, 0.20, 0.20]
        result = _assert_speed_proportion_consistency(train_props, test_props)
        assert result["passed"] is False
        assert "bin count mismatch" in result["reason"]

    def test_exactly_at_threshold_passes(self):
        """偏离恰好等于阈值（20%）应通过（≤20%）。"""
        train_props = [0.20, 0.20, 0.20, 0.20, 0.20]
        # bin 0 = 0.20 * (1 + 0.20) = 0.24, 其余平分剩余
        bin0 = 0.20 * (1.0 + SPEED_PROPORTION_TOLERANCE)
        remaining = 1.0 - bin0
        other = remaining / (SPEED_BIN_COUNT - 1)
        test_props = [bin0] + [other] * (SPEED_BIN_COUNT - 1)
        result = _assert_speed_proportion_consistency(train_props, test_props)
        assert result["passed"] is True


# ============================================================
# 方位角直方图计算测试
# ============================================================

class TestComputeAzimuthHistogram:
    """验证 _compute_azimuth_histogram 的输出结构。"""

    def test_eight_bins_output(self):
        """输出应有 8 个 bin。"""
        azimuths = np.linspace(0, 2 * math.pi - 0.01, 100)
        result = _compute_azimuth_histogram(azimuths)
        assert result["bin_count"] == AZIMUTH_BIN_COUNT
        assert len(result["counts"]) == AZIMUTH_BIN_COUNT
        assert len(result["proportions"]) == AZIMUTH_BIN_COUNT
        assert len(result["direction_names"]) == AZIMUTH_BIN_COUNT

    def test_proportions_sum_to_one(self):
        """proportions 总和应约等于 1。"""
        azimuths = np.linspace(0, 2 * math.pi - 0.01, 100)
        result = _compute_azimuth_histogram(azimuths)
        total = sum(result["proportions"])
        assert abs(total - 1.0) < 1e-6

    def test_direction_names_correct(self):
        """方向名称应正确（N, NE, E, SE, S, SW, W, NW）。"""
        azimuths = np.array([0.0])
        result = _compute_azimuth_histogram(azimuths)
        expected_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        assert result["direction_names"] == expected_names

    def test_shift_applied_for_compass_alignment(self):
        """验证 π/8 偏移使方位角正确映射到标准罗盘方向。

        标准罗盘划分：N=[337.5°,22.5°), NE=[22.5°,67.5°), E=[67.5°,112.5°), ...
        偏移 π/8 后，正北（0°/360°）应落在 bin 0（N），
        西北边界（337.5°）应落在 bin 7（NW）。
        """
        # 构造覆盖所有 8 个标准罗盘方向中心点的方位角
        centers_deg = [0, 45, 90, 135, 180, 225, 270, 315]
        centers_rad = [d * math.pi / 180 for d in centers_deg]
        azimuths = np.array(centers_rad, dtype=float)
        result = _compute_azimuth_histogram(azimuths)
        # 每个方向中心点应落在对应 bin 中
        for i, expected_bin in enumerate(range(8)):
            assert result["counts"][expected_bin] == 1, (
                f"Direction {result['direction_names'][i]} center should be in bin {expected_bin}"
            )

    def test_empty_azimuths_returns_zero_proportions(self):
        """空方位角数组返回零比例。"""
        result = _compute_azimuth_histogram(np.array([], dtype=float))
        assert result["total_samples"] == 0
        assert all(p == 0.0 for p in result["proportions"])


# ============================================================
# 速度分布计算测试
# ============================================================

class TestComputeSpeedDistribution:
    """验证 _compute_speed_distribution 的输出结构。"""

    def test_five_bins_output(self):
        """输出应有 5 个 bin（[0.5, 1.5] 等分）。"""
        speeds = np.linspace(0.5, 1.5, 100)
        result = _compute_speed_distribution(speeds)
        assert result["bin_count"] == SPEED_BIN_COUNT
        assert len(result["counts"]) == SPEED_BIN_COUNT
        assert len(result["proportions"]) == SPEED_BIN_COUNT

    def test_full_coverage_when_all_bins_populated(self):
        """所有 bin 都有样本时 full_coverage 为 True。"""
        # 构造覆盖所有 5 个 bin 的速度数据
        speeds = np.array([0.51, 0.71, 0.91, 1.11, 1.31])
        result = _compute_speed_distribution(speeds)
        assert result["full_coverage"] is True

    def test_not_full_coverage_when_bin_empty(self):
        """有 bin 无样本时 full_coverage 为 False。"""
        # 所有样本集中在第一个 bin
        speeds = np.array([0.51, 0.52, 0.53, 0.54, 0.55])
        result = _compute_speed_distribution(speeds)
        assert result["full_coverage"] is False

    def test_in_range_ratio_correct(self):
        """in_range_ratio 应正确反映区间内占比。"""
        speeds = np.array([0.3, 0.51, 0.71, 1.31, 1.7])  # 3/5 在 [0.5, 1.5]
        result = _compute_speed_distribution(speeds)
        assert result["in_range_count"] == 3
        assert abs(result["in_range_ratio"] - 0.6) < 1e-6

    def test_speed_range_config_matches_requirement(self):
        """速度区间配置应与 §175 要求一致 [0.5, 1.5] m/s。"""
        assert SPEED_RANGE[0] == 0.5
        assert SPEED_RANGE[1] == 1.5


# ============================================================
# 已知分布验证断言触发测试
# ============================================================

class TestKnownDistributionAssertionTrigger:
    """用已知构造分布验证断言正确触发（§175 单测要求）。"""

    def test_azimuth_assertion_triggers_on_extreme_skew(self):
        """构造极端偏斜的方位角分布，验证断言正确触发失败。"""
        # 构造 90% 集中在 N 方向的极端偏斜分布
        proportions = [0.90] + [0.10 / (AZIMUTH_BIN_COUNT - 1)] * (AZIMUTH_BIN_COUNT - 1)
        result = _assert_azimuth_uniformity(proportions)
        assert result["passed"] is False
        assert result["violations"][0]["direction"] == "N"
        assert result["violations"][0]["proportion"] == 0.90
        # 确认 deviation 远大于 20%
        assert result["violations"][0]["deviation"] > 5.0  # (0.90 - 0.125) / 0.125 = 6.2

    def test_speed_assertion_triggers_on_large_deviation(self):
        """构造训练/测试速度占比差异超过 20% 的分布，验证断言触发失败。"""
        # 训练：均匀分布
        train_props = [0.20, 0.20, 0.20, 0.20, 0.20]
        # 测试：bin 0 占比 50%（偏离 150%，远超 20%）
        test_props = [0.50, 0.125, 0.125, 0.125, 0.125]
        result = _assert_speed_proportion_consistency(train_props, test_props)
        assert result["passed"] is False
        assert len(result["violations"]) > 0
        # bin 0 的 deviation = |0.50 - 0.20| / 0.20 = 1.5 (150%)
        assert result["violations"][0]["relative_deviation"] > 1.0

    def test_azimuth_uniform_distribution_passes_known_case(self):
        """构造已知均匀分布，验证断言正确通过。"""
        # 精确均匀：每 bin = 1/8
        proportions = [0.125] * AZIMUTH_BIN_COUNT
        result = _assert_azimuth_uniformity(proportions)
        assert result["passed"] is True
        assert result["max_deviation"] == 0.0

    def test_speed_identical_distribution_passes_known_case(self):
        """构造已知一致分布，验证断言正确通过。"""
        train_props = [0.15, 0.25, 0.25, 0.20, 0.15]
        test_props = [0.15, 0.25, 0.25, 0.20, 0.15]
        result = _assert_speed_proportion_consistency(train_props, test_props)
        assert result["passed"] is True
        assert result["max_deviation"] == 0.0


# ============================================================
# 端到端集成测试（合成数据验证完整 manifest 生成流程）
# ============================================================

def _create_synthetic_gt_json(
    tmp_dir: Path,
    seq_id: str,
    duration_s: float = 10.0,
    dt: float = 0.1,
    speed_mps: float = 1.0,
    yaw_rad: float = 0.0,
) -> Path:
    """创建合成的 gt.json 文件，模拟匀速直线运动。

    参数：
        tmp_dir: 临时目录。
        seq_id: 序列 ID（用作目录名）。
        duration_s: 运动时长（秒）。
        dt: 采样间隔（秒）。
        speed_mps: 速度（m/s）。
        yaw_rad: 航向角（弧度）。

    返回：
        gt.json 文件路径。
    """
    seq_dir = tmp_dir / seq_id
    seq_dir.mkdir(parents=True, exist_ok=True)

    n_steps = int(duration_s / dt) + 1
    rows = []
    for i in range(n_steps):
        t = i * dt
        x = speed_mps * math.cos(yaw_rad) * t
        y = speed_mps * math.sin(yaw_rad) * t
        rows.append({
            "timestamp": round(t, 6),
            "px": round(x, 6),
            "py": round(y, 6),
            "yaw": round(yaw_rad, 6),
        })

    gt_path = seq_dir / "gt.json"
    gt_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return gt_path


def _create_synthetic_manifest(
    tmp_dir: Path,
    train_seq_ids: list[str],
    val_seq_ids: list[str],
    test_seq_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """创建合成的 dataset_manifest 和 split_manifest。

    参数：
        tmp_dir: 临时目录。
        train_seq_ids: 训练序列 ID 列表。
        val_seq_ids: 验证序列 ID 列表。
        test_seq_ids: 测试序列 ID 列表。

    返回：
        (dataset_manifest, split_manifest) 元组。
    """
    all_seq_ids = train_seq_ids + val_seq_ids + test_seq_ids
    sequences = []
    for seq_id in all_seq_ids:
        seq_dir = tmp_dir / seq_id
        sequences.append({
            "seq_id": seq_id,
            "seq_dir": str(seq_dir),
            "scene_id": seq_id,
        })

    dataset_manifest = {
        "data_root": str(tmp_dir),
        "sequence_count": len(all_seq_ids),
        "sequences": sequences,
    }

    split_manifest = {
        "train_ids": train_seq_ids,
        "val_ids": val_seq_ids,
        "test_ids": test_seq_ids,
    }

    return dataset_manifest, split_manifest


class TestEndToEndTrajDiversity:
    """用合成轨迹数据验证完整 manifest 生成流程。"""

    def test_full_pipeline_with_diverse_trajectories(self):
        """验证训练/测试轨迹方向一致、速度一致时 overall_passed=True。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # 训练轨迹：均匀覆盖 8 个方向，速度 1.0 m/s
            train_seq_ids = []
            for i in range(8):
                yaw = i * math.pi / 4  # 8 个方向均匀分布
                seq_id = f"train_dir_{i}"
                _create_synthetic_gt_json(tmp_dir, seq_id, duration_s=5.0, dt=0.1,
                                          speed_mps=1.0, yaw_rad=yaw)
                train_seq_ids.append(seq_id)

            # 测试轨迹：同样均匀覆盖 8 个方向，速度 1.0 m/s
            test_seq_ids = []
            for i in range(8):
                yaw = i * math.pi / 4
                seq_id = f"test_dir_{i}"
                _create_synthetic_gt_json(tmp_dir, seq_id, duration_s=5.0, dt=0.1,
                                          speed_mps=1.0, yaw_rad=yaw)
                test_seq_ids.append(seq_id)

            # 构造 manifest
            dataset_manifest, split_manifest = _create_synthetic_manifest(
                tmp_dir, train_seq_ids, [], test_seq_ids
            )

            # 运行轨迹多样性校验
            result = compute_traj_diversity_manifest(dataset_manifest, split_manifest)

            # 验证输出结构
            assert "overall_passed" in result
            assert "azimuth_assertion" in result
            assert "speed_full_coverage" in result
            assert "speed_proportion_assertion" in result
            assert "train_stats" in result
            assert "test_stats" in result
            assert "config" in result

            # 验证配置
            assert result["config"]["azimuth_bin_count"] == 8
            assert result["config"]["azimuth_max_deviation"] == 0.20
            assert result["config"]["speed_range_mps"] == [0.5, 1.5]
            assert result["config"]["speed_bin_count"] == 5
            assert result["config"]["speed_proportion_tolerance"] == 0.20

            # 验证训练/测试各有 8 条轨迹
            assert result["train_stats"]["seq_count"] == 8
            assert result["test_stats"]["seq_count"] == 8

            # 验证方位角直方图（每个切分都有 8 bin）
            assert len(result["train_stats"]["seq_azimuth_histograms"]) == 8
            assert len(result["test_stats"]["seq_azimuth_histograms"]) == 8

            # 验证速度分布（每个切分都有分布）
            assert len(result["train_stats"]["seq_speed_distributions"]) == 8
            assert len(result["test_stats"]["seq_speed_distributions"]) == 8

    def test_full_pipeline_with_skewed_direction_fails_azimuth(self):
        """验证训练方向均匀但测试方向偏斜时方位角断言失败。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # 训练轨迹：8 个方向均匀
            train_seq_ids = []
            for i in range(8):
                yaw = i * math.pi / 4
                seq_id = f"train_dir_{i}"
                _create_synthetic_gt_json(tmp_dir, seq_id, duration_s=5.0, dt=0.1,
                                          speed_mps=1.0, yaw_rad=yaw)
                train_seq_ids.append(seq_id)

            # 测试轨迹：全部朝北（偏斜）
            test_seq_ids = []
            for i in range(8):
                seq_id = f"test_north_{i}"
                _create_synthetic_gt_json(tmp_dir, seq_id, duration_s=5.0, dt=0.1,
                                          speed_mps=1.0, yaw_rad=0.0)
                test_seq_ids.append(seq_id)

            dataset_manifest, split_manifest = _create_synthetic_manifest(
                tmp_dir, train_seq_ids, [], test_seq_ids
            )

            result = compute_traj_diversity_manifest(dataset_manifest, split_manifest)

            # 方位角断言应失败（测试方向偏斜）
            assert result["overall_passed"] is False
            assert result["azimuth_assertion"]["passed"] is False

    def test_manifest_write_and_read_roundtrip(self):
        """验证 traj_diversity manifest 写入和读取的完整流程。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            output_dir = tmp_dir / "output"
            output_dir.mkdir()

            # 构造最小 manifest
            manifest = {
                "overall_passed": True,
                "azimuth_assertion": {"passed": True, "max_deviation": 0.0},
                "speed_full_coverage": True,
                "speed_proportion_assertion": {"passed": True, "max_deviation": 0.0},
                "train_stats": {"seq_count": 8},
                "test_stats": {"seq_count": 8},
                "config": {
                    "azimuth_bin_count": 8,
                    "azimuth_max_deviation": 0.20,
                    "speed_range_mps": [0.5, 1.5],
                    "speed_bin_count": 5,
                    "speed_proportion_tolerance": 0.20,
                },
            }

            # 写入
            output_path = write_traj_diversity_manifest(manifest, output_dir)
            assert output_path.is_file()
            assert output_path.name == "traj_diversity.json"

            # 读取验证
            with open(output_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["overall_passed"] is True
            assert loaded["azimuth_assertion"]["passed"] is True
            assert loaded["speed_full_coverage"] is True
            assert loaded["config"]["azimuth_bin_count"] == 8
            assert loaded["config"]["speed_range_mps"] == [0.5, 1.5]

    def test_per_trajectory_azimuth_assertion_recorded(self):
        """验证每条轨迹的方位角断言结果被记录在 manifest 中。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # 训练轨迹：直线行驶（单方向，正常情况）
            train_seq_ids = []
            for i in range(4):
                yaw = i * math.pi / 2  # 4 个不同方向
                seq_id = f"train_{i}"
                _create_synthetic_gt_json(tmp_dir, seq_id, duration_s=5.0, dt=0.1,
                                          speed_mps=1.0, yaw_rad=yaw)
                train_seq_ids.append(seq_id)

            # 测试轨迹：直线行驶（单方向，正常情况）
            test_seq_ids = []
            for i in range(4):
                yaw = i * math.pi / 2
                seq_id = f"test_{i}"
                _create_synthetic_gt_json(tmp_dir, seq_id, duration_s=5.0, dt=0.1,
                                          speed_mps=1.0, yaw_rad=yaw)
                test_seq_ids.append(seq_id)

            dataset_manifest, split_manifest = _create_synthetic_manifest(
                tmp_dir, train_seq_ids, [], test_seq_ids
            )

            result = compute_traj_diversity_manifest(dataset_manifest, split_manifest)

            # 验证每条轨迹的方位角断言被记录
            test_seq_assertions = result["test_stats"]["seq_azimuth_assertions"]
            assert len(test_seq_assertions) == 4
            for assertion in test_seq_assertions:
                assert "seq_id" in assertion
                assert "passed" in assertion
                assert "direction" not in assertion  # 聚合级断言有 direction，单条没有

            # 验证轨迹级汇总
            assert "azimuth_trajectory_summary" in result
            traj_summary = result["azimuth_trajectory_summary"]
            assert traj_summary["total_with_data"] == 4
            assert traj_summary["all_passed"] is True

    def test_per_trajectory_speed_coverage_recorded(self):
        """验证每条轨迹的速度区间覆盖结果被记录在 manifest 中。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # 训练轨迹：速度 1.0 m/s（在 [0.5, 1.5] 区间内）
            train_seq_ids = []
            for i in range(2):
                seq_id = f"train_spd_{i}"
                _create_synthetic_gt_json(tmp_dir, seq_id, duration_s=5.0, dt=0.1,
                                          speed_mps=1.0, yaw_rad=0.0)
                train_seq_ids.append(seq_id)

            # 测试轨迹：速度 1.0 m/s
            test_seq_ids = []
            for i in range(2):
                seq_id = f"test_spd_{i}"
                _create_synthetic_gt_json(tmp_dir, seq_id, duration_s=5.0, dt=0.1,
                                          speed_mps=1.0, yaw_rad=0.0)
                test_seq_ids.append(seq_id)

            dataset_manifest, split_manifest = _create_synthetic_manifest(
                tmp_dir, train_seq_ids, [], test_seq_ids
            )

            result = compute_traj_diversity_manifest(dataset_manifest, split_manifest)

            # 验证每条轨迹的速度覆盖被记录
            test_speed_coverage = result["test_stats"]["seq_speed_coverage"]
            assert len(test_speed_coverage) == 2
            for coverage in test_speed_coverage:
                assert "seq_id" in coverage
                assert "has_in_range_sample" in coverage
                assert coverage["has_in_range_sample"] is True

            # 验证轨迹级汇总
            assert "speed_trajectory_summary" in result
            speed_summary = result["speed_trajectory_summary"]
            assert speed_summary["total_with_data"] == 2
            assert speed_summary["all_have_in_range"] is True

    def test_degenerate_trajectory_detected(self):
        """验证静止轨迹（零速度）被标记为退化。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # 训练轨迹：正常运动
            train_seq_ids = []
            for i in range(2):
                seq_id = f"train_{i}"
                _create_synthetic_gt_json(tmp_dir, seq_id, duration_s=5.0, dt=0.1,
                                          speed_mps=1.0, yaw_rad=0.0)
                train_seq_ids.append(seq_id)

            # 测试轨迹：静止（零速度）
            test_seq_ids = []
            for i in range(2):
                seq_id = f"test_stationary_{i}"
                # 创建静止轨迹：所有样本在同一位置
                seq_dir = tmp_dir / seq_id
                seq_dir.mkdir(parents=True, exist_ok=True)
                rows = []
                for j in range(51):
                    rows.append({
                        "timestamp": round(j * 0.1, 6),
                        "px": 0.0,
                        "py": 0.0,
                        "yaw": 0.0,
                    })
                gt_path = seq_dir / "gt.json"
                gt_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
                test_seq_ids.append(seq_id)

            dataset_manifest, split_manifest = _create_synthetic_manifest(
                tmp_dir, train_seq_ids, [], test_seq_ids
            )

            result = compute_traj_diversity_manifest(dataset_manifest, split_manifest)

            # 静止轨迹应被标记为退化
            test_azimuth_assertions = result["test_stats"]["seq_azimuth_assertions"]
            for assertion in test_azimuth_assertions:
                if "stationary" in assertion.get("seq_id", ""):
                    assert assertion["passed"] is False

            # overall_passed 应为 False（有退化轨迹）
            assert result["overall_passed"] is False
