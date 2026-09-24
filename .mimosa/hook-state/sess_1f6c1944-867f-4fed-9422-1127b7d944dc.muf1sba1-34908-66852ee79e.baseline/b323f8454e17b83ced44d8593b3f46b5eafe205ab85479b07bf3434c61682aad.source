"""案例绘图模块测试。

本模块验证 render_case_figures 函数的正确性，该函数
根据选中的案例数据渲染各类案例图表。

测试覆盖范围：
  - 正常情况：渲染 main_cases、failure_cases、boundary_cases 图表，
    验证 case_refs 和 figure_path
  - 异常情况：缺少 boundary_cases 键时抛出 ValueError

被测模块：
  - liquidloc.plotting.plot_cases
"""

from __future__ import annotations

from pathlib import Path

import pytest

from liquidloc.plotting.plot_cases import render_case_figures


def test_normal_case(tmp_path):
    """验证三类案例图表的正确渲染。

    测试场景：传入包含 main_cases、failure_cases、boundary_cases
    的选中案例数据，指定 figure_path。
    预期行为：manifest 中各类案例的 case_refs 正确，
    各类图表文件已生成，figure_path 指向正确的文件路径。
    """
    manifest = render_case_figures(
        {
            "main_cases": [{"case_ref": "case_main", "rmse": 0.10}],
            "failure_cases": [{"seq_id": "case_failure", "rmse": 0.80}],
            "boundary_cases": [{"case_ref": "case_boundary", "rmse": 0.35}],
        },
        {"figure_path": tmp_path / "cases.png"},
    )

    # 验证各类案例的 case_refs
    assert manifest["main_cases"]["case_refs"] == ["case_main"]
    assert manifest["failure_cases"]["case_refs"] == ["case_failure"]
    assert manifest["boundary_cases"]["case_refs"] == ["case_boundary"]
    # 验证图表文件已生成
    assert Path(manifest["main_cases"]["figure_path"]).is_file()
    assert Path(manifest["failure_cases"]["figure_path"]).is_file()
    assert Path(manifest["boundary_cases"]["figure_path"]).is_file()
    # 验证 figure_path 指向正确的文件路径
    assert manifest["main_cases"]["figure_path"] == str((tmp_path / "cases_main_cases.png").resolve())
    assert manifest["failure_cases"]["figure_path"] == str((tmp_path / "cases_failure_cases.png").resolve())
    assert manifest["boundary_cases"]["figure_path"] == str((tmp_path / "cases_boundary_cases.png").resolve())


def test_invalid_case(tmp_path):
    """验证缺少 boundary_cases 键时抛出 ValueError。

    测试场景：传入的选中案例数据中缺少 boundary_cases 键。
    预期行为：抛出 ValueError，提示缺少 boundary_cases。
    """
    with pytest.raises(ValueError, match="boundary_cases"):
        render_case_figures(
            {
                "main_cases": [{"case_ref": "case_main"}],
                "failure_cases": [{"case_ref": "case_failure"}],
            },
            {"figure_path": tmp_path / "cases.png"},
        )
