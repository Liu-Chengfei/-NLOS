"""摘要构建模块测试。

本模块验证 build_summary 函数的正确性，该函数将指标表、统计检验表
和选中案例汇总为最终摘要结构。

测试覆盖范围：
  - 正常情况：列表格式的 metric_table 和 statistics_table，提取 case_refs 和 summary_stats
  - 边界情况：映射格式的 metric_table 和 statistics_table（路径引用），空案例列表
  - 异常情况：空 metric_table 时抛出 ValueError

被测模块：
  - liquidloc.analysis.summary_builder
"""

from __future__ import annotations

import pytest

from liquidloc.analysis.summary_builder import build_summary


def test_normal_case():
    """验证列表格式输入时摘要的正确构建。

    测试场景：传入列表格式的 metric_table 和 statistics_table，
    以及包含 main_cases、failure_cases、boundary_cases 的选中案例。
    预期行为：case_refs 去重合并所有案例引用，
    summary_stats 包含 main_table_ref 和 statistics_ref。
    """
    summary = build_summary(
        metric_table=[
            {"metric_name": "rmse", "value": 0.12},
            {"metric_name": "coverage", "value": 0.91},
        ],
        statistics_table=[
            {
                "metric_name": "rmse",
                "p_value": 0.03,
                "effect_size": -1.0,
                "adjusted_p_value": 0.03,
            }
        ],
        selected_cases={
            "main_cases": [{"case_ref": "case_main"}],
            "failure_cases": [{"seq_id": "case_failure"}],
            "boundary_cases": [{"case_ref": "case_main"}, {"case_ref": "case_boundary"}],
        },
    )

    # case_refs 应去重合并所有案例引用（case_main 出现两次但去重为一次）
    assert summary["case_refs"] == ["case_main", "case_failure", "case_boundary"]
    # summary_stats 应包含 metric_table 和 statistics_table 的引用
    assert summary["summary_stats"]["main_table_ref"][0]["metric_name"] == "rmse"
    assert summary["summary_stats"]["statistics_ref"][0]["adjusted_p_value"] == 0.03


def test_boundary_case():
    """验证映射格式输入且空案例列表时的摘要构建。

    测试场景：metric_table 和 statistics_table 为映射格式
    （包含路径引用而非实际数据），所有案例列表为空。
    预期行为：case_refs 为空列表，summary_stats 直接透传映射引用。
    """
    summary = build_summary(
        metric_table={"table_path": "outputs/metric_table.csv"},
        statistics_table={"table_path": "outputs/statistics_table.json"},
        selected_cases={
            "main_cases": [],
            "failure_cases": [],
            "boundary_cases": [],
        },
    )

    # 空案例列表导致 case_refs 为空
    assert summary == {
        "case_refs": [],
        "summary_stats": {
            # 映射格式直接透传，不做展开
            "main_table_ref": {"table_path": "outputs/metric_table.csv"},
            "statistics_ref": {"table_path": "outputs/statistics_table.json"},
        },
    }


def test_invalid_case():
    """验证空 metric_table 时抛出 ValueError。

    测试场景：metric_table 为空列表，statistics_table 非空。
    预期行为：抛出 ValueError，提示 metric_table 不能为空。
    """
    with pytest.raises(ValueError, match="metric_table must be non-empty"):
        build_summary(
            metric_table=[],
            statistics_table=[{"metric_name": "rmse"}],
            selected_cases={
                "main_cases": [],
                "failure_cases": [],
                "boundary_cases": [],
            },
        )
