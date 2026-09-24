"""显著性检验模块测试。

本模块验证 run_significance_tests 和 _bh_fdr_adjust_p_values 函数的正确性，
涵盖统计检验、p 值校正、配对检验、贝叶斯推断等核心功能。

测试覆盖范围：
  - 正常情况：两组方法的显著性检验，验证输出字段完整性
  - 边界情况：重复 metric_name 去重后仅保留一条
  - 异常情况：未知 metric_name 抛出 KeyError
  - 异常情况：布尔类型 metric 值被拒绝
  - 配对 Wilcoxon 检验与 Bootstrap 置信区间
  - 显式 pairing_keys 覆盖默认配对字段
  - BH-FDR p 值校正的累积最小值逻辑

被测模块：
  - liquidloc.analysis.significance_tests
"""

from __future__ import annotations

import pytest

from liquidloc.analysis.significance_tests import _bh_fdr_adjust_p_values, run_significance_tests


def test_normal_case():
    """验证两组方法的显著性检验正常输出。

    测试场景：传入 ekf 和 liquid 两组方法各 2 条记录，
    对 coverage 和 rmse 两个指标进行检验。
    预期行为：输出按 metric_name 排序（rmse 在前），
    包含 method_name_a/b、sample_size_a/b、adjusted_p_value 等字段，
    adjusted_p_value 在 [0, 1] 范围内。
    """
    statistics_table = run_significance_tests(
        metric_table=[
            {"method_name": "ekf", "rmse": 0.10, "coverage": 0.80},
            {"method_name": "ekf", "rmse": 0.20, "coverage": 0.82},
            {"method_name": "liquid", "rmse": 0.40, "coverage": 0.91},
            {"method_name": "liquid", "rmse": 0.50, "coverage": 0.93},
        ],
        group_keys=["method_name"],
        metric_names=["coverage", "rmse"],
    )

    # 验证输出按 metric_name 排序（rmse 在前）
    assert [row["metric_name"] for row in statistics_table] == ["rmse", "coverage"]
    # 验证方法名按字典序排列
    assert statistics_table[0]["method_name_a"] == "ekf"
    assert statistics_table[0]["method_name_b"] == "liquid"
    # 验证样本量正确
    assert statistics_table[0]["sample_size_a"] == 2
    assert statistics_table[0]["sample_size_b"] == 2
    # 验证校正后 p 值在合法范围内
    assert 0.0 <= statistics_table[0]["adjusted_p_value"] <= 1.0


def test_boundary_case():
    """验证重复 metric_name 去重后仅保留一条检验结果。

    测试场景：metric_names 中包含两个 "rmse"，应去重为一条。
    预期行为：输出仅 1 条记录，metric_name 为 rmse，
    包含 p_value 和 effect_size 字段。
    """
    statistics_table = run_significance_tests(
        metric_table=[
            {"method_name": "ekf", "metric": "rmse", "value": 0.10},
            {"method_name": "ekf", "metric": "rmse", "value": 0.20},
            {"method_name": "liquid", "metric": "rmse", "value": 0.40},
            {"method_name": "liquid", "metric": "rmse", "value": 0.50},
        ],
        group_keys=["method_name"],
        metric_names=["rmse", "rmse"],
    )

    # 重复 metric_name 去重后仅 1 条结果
    assert len(statistics_table) == 1
    assert statistics_table[0]["metric_name"] == "rmse"
    # 验证核心输出字段存在
    assert "p_value" in statistics_table[0]
    assert "effect_size" in statistics_table[0]


def test_invalid_case():
    """验证未知 metric_name 抛出 KeyError。

    测试场景：metric_names 中包含数据表中不存在的 "unknown_metric"。
    预期行为：抛出 KeyError，提示未知的 schema metrics。
    """
    with pytest.raises(KeyError, match="unknown schema metrics"):
        run_significance_tests(
            metric_table=[{"method_name": "ekf", "rmse": 0.10}],
            group_keys=["method_name"],
            metric_names=["unknown_metric"],
        )


def test_rejects_boolean_metric_values():
    """验证布尔类型的 metric 值被拒绝。

    测试场景：metric_table 中 rmse 的值为 True（布尔类型）。
    预期行为：抛出 ValueError，提示 metric 值必须为数值类型。
    """
    with pytest.raises(ValueError, match="must be numeric"):
        run_significance_tests(
            metric_table=[{"method_name": "ekf", "rmse": True}],
            group_keys=["method_name"],
            metric_names=["rmse"],
        )


def test_paired_wilcoxon_and_bootstrap_ci_are_reported_for_matched_rows():
    """验证配对 Wilcoxon 检验与 Bootstrap 置信区间的完整输出。

    测试场景：传入按 scene_id 配对的 ekf 和 liquid 数据，
    系统自动检测配对关系。
    预期行为：使用 paired_wilcoxon 检验，paired_sample_size=2，
    报告 bootstrap_percentile 置信区间和贝叶斯推断结果，
    ci_low <= ci_high，bayes_prob_b_better >= bayes_prob_a_better。
    """
    statistics_table = run_significance_tests(
        metric_table=[
            {"scene_id": "s1", "method_name": "ekf", "rmse": 0.20},
            {"scene_id": "s1", "method_name": "liquid", "rmse": 0.10},
            {"scene_id": "s2", "method_name": "ekf", "rmse": 0.30},
            {"scene_id": "s2", "method_name": "liquid", "rmse": 0.15},
        ],
        group_keys=["method_name"],
        metric_names=["rmse"],
    )

    assert len(statistics_table) == 1
    row = statistics_table[0]
    # 验证使用配对 Wilcoxon 检验
    assert row["test_name"] == "paired_wilcoxon"
    # 验证配对样本量
    assert row["paired_sample_size"] == 2
    # 验证置信区间方法为 bootstrap 百分位法
    assert row["ci_method"] == "bootstrap_percentile"
    assert row["confidence_interval"]["statistic"] == "mean_difference"
    # 验证置信区间下界 <= 上界
    assert row["ci_low"] <= row["ci_high"]
    # 验证贝叶斯推断字段
    assert row["bayes_method"] == "bayesian_bootstrap_mean_difference"
    assert row["bayes_ci_low"] <= row["bayes_ci_high"]
    # 验证贝叶斯概率在合法范围内
    assert 0.0 <= row["bayes_prob_b_better"] <= 1.0
    assert 0.0 <= row["bayes_prob_a_better"] <= 1.0
    # liquid 方法表现更好，prob_b_better 应 >= prob_a_better
    assert row["bayes_prob_b_better"] >= row["bayes_prob_a_better"]
    # 验证 p 值在合法范围内
    assert 0.0 <= row["p_value"] <= 1.0


def test_explicit_pairing_keys_override_non_pairable_case_ref_fields():
    """验证显式 pairing_keys 参数覆盖默认的 case_ref 配对字段。

    测试场景：数据中包含 case_ref 和 task_id 两个字段，
    通过 pairing_keys=["task_id"] 显式指定使用 task_id 进行配对。
    预期行为：使用 paired_wilcoxon 检验，paired_sample_size=2，
    pairing_fields 为 ["task_id"]。
    """
    statistics_table = run_significance_tests(
        metric_table=[
            {"case_ref": "scene_00::ekf", "task_id": "scene_00", "method_name": "ekf", "rmse": 0.20},
            {"case_ref": "scene_00::liquid", "task_id": "scene_00", "method_name": "liquid", "rmse": 0.10},
            {"case_ref": "scene_01::ekf", "task_id": "scene_01", "method_name": "ekf", "rmse": 0.30},
            {"case_ref": "scene_01::liquid", "task_id": "scene_01", "method_name": "liquid", "rmse": 0.15},
        ],
        group_keys=["method_name"],
        metric_names=["rmse"],
        pairing_keys=["task_id"],
    )

    assert len(statistics_table) == 1
    row = statistics_table[0]
    # 验证使用配对检验
    assert row["test_name"] == "paired_wilcoxon"
    assert row["paired_sample_size"] == 2
    # 验证配对字段为显式指定的 task_id
    assert row["pairing_fields"] == ["task_id"]


def test_adjust_p_values_uses_bh_fdr_cumulative_min():
    """验证 BH-FDR p 值校正使用累积最小值逻辑。

    测试场景：传入三个 p 值 [0.01, 0.02, 0.03]。
    预期行为：BH-FDR 校正后所有 p 值均为 0.03，
    因为校正公式为 p * n / rank，取累积最小值后
    最大的校正 p 值会向下传播。
    """
    assert _bh_fdr_adjust_p_values([0.01, 0.02, 0.03]) == [0.03, 0.03, 0.03]
