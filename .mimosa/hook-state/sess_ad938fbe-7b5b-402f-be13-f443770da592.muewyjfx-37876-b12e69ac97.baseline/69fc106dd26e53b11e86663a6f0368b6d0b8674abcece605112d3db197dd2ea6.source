"""主表格绘图模块测试。

本模块验证 render_main_table_figure 和 build_main_table_figure_spec
函数的正确性，涵盖表格图表的渲染和规格构建。

测试覆盖范围：
  - 正常情况：渲染主表格图表，验证文件生成
  - 异常情况：缺少 figure_path 时抛出 ValueError
  - 列名空白字符归一化：build_main_table_figure_spec 对列名做 strip

被测模块：
  - liquidloc.plotting.plot_main_table
"""

from __future__ import annotations

from pathlib import Path

import pytest

from liquidloc.plotting.plot_main_table import build_main_table_figure_spec, render_main_table_figure


def test_normal_case(tmp_path):
    """验证主表格图表的正确渲染。

    测试场景：传入两个方法的指标数据，指定 figure_path。
    预期行为：返回的渲染路径与 figure_path 一致，
    图表文件已生成。
    """
    figure_path = tmp_path / "main_table.png"

    rendered_figure_path = render_main_table_figure(
        [
            {"method_name": "ekf", "rmse": 0.10, "coverage": 0.80},
            {"method_name": "liquid", "rmse": 0.08, "coverage": 0.92},
        ],
        {"figure_path": figure_path},
    )

    # 渲染路径应与输入的 figure_path 一致
    assert rendered_figure_path == str(figure_path)
    # 图表文件应已生成
    assert Path(rendered_figure_path).is_file()


def test_invalid_case(tmp_path):
    """验证缺少 figure_path 时抛出 ValueError。

    测试场景：传入 output_path 而非 figure_path 作为绘图选项。
    预期行为：抛出 ValueError，提示缺少 figure_path（项目硬约束：value contract 问题用 ValueError 而非 KeyError）。
    """
    with pytest.raises(ValueError, match="figure_path"):
        render_main_table_figure(
            [{"method_name": "ekf", "rmse": 0.10}],
            # 错误的键名，应为 figure_path
            {"output_path": tmp_path / "main_table.png"},
        )


def test_build_spec_normalizes_whitespace_wrapped_column_names(tmp_path):
    """验证 build_main_table_figure_spec 对列名做空白字符归一化。

    测试场景：传入的列名包含前后空格（如 " method_name "）。
    预期行为：spec 中的 columns 列表已去除列名的前后空格，
    cell_text 中的值对应归一化后的列。
    """
    spec = build_main_table_figure_spec(
        [{" method_name ": "ekf", " rmse ": 0.10}],
        {"figure_path": tmp_path / "main_table.png"},
    )

    # 列名应被 strip 归一化
    assert spec["columns"] == ["method_name", "rmse"]
    # 单元格文本对应归一化后的列值
    assert spec["cell_text"] == [["ekf", "0.1"]]
