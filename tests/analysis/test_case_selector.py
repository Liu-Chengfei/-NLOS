"""案例选择器模块测试。

本模块验证 select_cases 函数的正确性，该函数根据规则从指标表中
选择主案例、失败案例和边界案例。

测试覆盖范围：
  - 正常情况：列表格式 metric_table，按规则正确选择三类案例
  - 边界情况：映射格式 metric_table，重复 case_ref 去重
  - 异常情况：规则引用不存在的 case_ref 时抛出 KeyError
  - 映射格式 metric_table 的键名空白字符剥离
  - 选中案例记录的归一化处理

被测模块：
  - liquidloc.analysis.case_selector
"""

from __future__ import annotations

import pytest

from liquidloc.analysis.case_selector import select_cases


def test_normal_case():
    """验证列表格式 metric_table 的案例选择正确性。

    测试场景：传入列表格式的 metric_table 和 case_rules，
    分别指定 main_cases、failure_cases、boundary_cases。
    预期行为：各类案例按 case_ref 正确匹配并返回。
    """
    selected_cases = select_cases(
        metric_table=[
            {"case_ref": "case_main", "rmse": 0.10},
            {"case_ref": "case_failure", "rmse": 0.80},
            {"case_ref": "case_boundary", "rmse": 0.35},
        ],
        case_rules={
            "main_cases": ["case_main"],
            "failure_cases": ["case_failure"],
            "boundary_cases": ["case_boundary"],
        },
    )

    # 验证各类案例按 case_ref 正确选择
    assert [case["case_ref"] for case in selected_cases["main_cases"]] == ["case_main"]
    assert [case["case_ref"] for case in selected_cases["failure_cases"]] == ["case_failure"]
    assert [case["case_ref"] for case in selected_cases["boundary_cases"]] == ["case_boundary"]


def test_boundary_case():
    """验证映射格式 metric_table 中重复 case_ref 的去重行为。

    测试场景：传入映射格式的 metric_table，
    case_rules 中 main_cases 和 boundary_cases 各包含重复的 case_ref。
    预期行为：重复的 case_ref 被去重，每类仅保留一条。
    """
    selected_cases = select_cases(
        metric_table={
            "case_main": {"case_ref": "case_main", "rmse": 0.10},
            "case_failure": {"case_ref": "case_failure", "rmse": 0.80},
            "case_boundary": {"case_ref": "case_boundary", "rmse": 0.35},
        },
        case_rules={
            "main_cases": ["case_main", "case_main"],
            "failure_cases": ["case_failure"],
            "boundary_cases": ["case_boundary", "case_boundary"],
        },
    )

    # 重复的 case_ref 应被去重
    assert [case["case_ref"] for case in selected_cases["main_cases"]] == ["case_main"]
    assert [case["case_ref"] for case in selected_cases["boundary_cases"]] == ["case_boundary"]


def test_invalid_case():
    """验证规则引用不存在的 case_ref 时抛出 KeyError。

    测试场景：case_rules 中 failure_cases 引用了 metric_table 中
    不存在的 "case_missing"。
    预期行为：抛出 KeyError，提示未知的 case。
    """
    with pytest.raises(KeyError, match="unknown case"):
        select_cases(
            metric_table=[{"case_ref": "case_main", "rmse": 0.10}],
            case_rules={
                "main_cases": ["case_main"],
                "failure_cases": ["case_missing"],
                "boundary_cases": [],
            },
        )


def test_mapping_metric_table_strips_case_refs():
    """验证映射格式 metric_table 的键名空白字符被剥离后再匹配。

    测试场景：映射的键为 " case_main "（前后有空格），
    case_rules 中引用 "case_main"（无空格）。
    预期行为：strip 后键名匹配成功，正确选择案例。
    """
    selected_cases = select_cases(
        metric_table={
            " case_main ": {"case_ref": "case_main", "rmse": 0.10},
            "case_failure": {"case_ref": "case_failure", "rmse": 0.80},
            "case_boundary": {"case_ref": "case_boundary", "rmse": 0.35},
        },
        case_rules={
            "main_cases": ["case_main"],
            "failure_cases": ["case_failure"],
            "boundary_cases": ["case_boundary"],
        },
    )

    # 映射键名 strip 后应能正确匹配
    assert [case["case_ref"] for case in selected_cases["main_cases"]] == ["case_main"]


def test_selected_case_records_are_normalized():
    """验证选中案例记录的归一化处理：case_ref 被 strip，但其他字段保持原样。

    测试场景：映射中第一条记录的键和 seq_id 都有前后空格，
    第二条记录的 case_ref 有前后空格。
    预期行为：选中记录的 case_ref 被 strip 归一化，
    但 seq_id 等其他字段保持原值不变。
    """
    selected_cases = select_cases(
        metric_table={
            " case_main ": {"seq_id": " case_main ", "rmse": 0.10},
            "case_failure": {"case_ref": " case_failure ", "rmse": 0.80},
            "case_boundary": {"case_ref": "case_boundary", "rmse": 0.35},
        },
        case_rules={
            "main_cases": ["case_main"],
            "failure_cases": ["case_failure"],
            "boundary_cases": ["case_boundary"],
        },
    )

    # case_ref 被 strip 归一化
    assert [case["case_ref"] for case in selected_cases["main_cases"]] == ["case_main"]
    # seq_id 等其他字段保持原值，不做 strip
    assert selected_cases["main_cases"][0]["seq_id"] == " case_main "
    # failure_cases 的 case_ref 被 strip 归一化
    assert selected_cases["failure_cases"][0]["case_ref"] == "case_failure"
