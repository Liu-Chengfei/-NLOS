from __future__ import annotations

"""扫描绘图（plot_sweeps）测试模块。

文件职责：验证 render_sweep_figure 能正确渲染
参数扫描图表。

测试覆盖范围：
- 正常场景：渲染折线扫描图
- 异常场景：非法 y 轴值类型
- 自动推断跳过非常见首行键
- 自动推断优先选择扫描轴而非 case_ref
- 布尔 x 轴不视为数值

被测模块：liquidloc.plotting.plot_sweeps"""


import pytest

from liquidloc.plotting.plot_sweeps import render_sweep_figure


def test_normal_case(tmp_path):
    figure_path = tmp_path / "sweep.svg"

    manifest = render_sweep_figure(
        [
            {"epoch": 1, "rmse": 0.12},
            {"epoch": 2, "rmse": 0.08},
        ],
        {
            "figure_path": figure_path,
            "x_axis": "epoch",
            "y_axis": "rmse",
            "kind": "line",
            "return_manifest": True,
        },
    )

    assert figure_path.is_file()
    assert manifest["figure_path"] == str(figure_path.resolve())
    assert manifest["x_axis"] == "epoch"
    assert manifest["y_axis"] == "rmse"
    assert manifest["kind"] == "line"


def test_invalid_case(tmp_path):
    with pytest.raises(TypeError, match="rmse"):
        render_sweep_figure(
            [
                {"epoch": 1, "rmse": 0.12},
                {"epoch": 2, "rmse": "bad"},
            ],
            {
                "figure_path": tmp_path / "sweep.svg",
                "x_axis": "epoch",
                "y_axis": "rmse",
            },
        )


def test_auto_inference_skips_non_common_first_row_key(tmp_path):
    figure_path = tmp_path / "wide.svg"

    manifest = render_sweep_figure(
        [
            {"note": "first", "epoch": 1, "rmse": 0.12},
            {"epoch": 2, "rmse": 0.08},
        ],
        {
            "figure_path": figure_path,
            "return_manifest": True,
        },
    )

    assert figure_path.is_file()
    assert manifest["x_axis"] == "epoch"
    assert manifest["y_axis"] == "rmse"


def test_auto_inference_prefers_sweep_axes_over_case_ref(tmp_path):
    figure_path = tmp_path / "priority.svg"

    manifest = render_sweep_figure(
        [
            {
                "case_ref": "case-1",
                "seq_id": "mini_seq",
                "scene_id": "S(A2,N2,V2,K0,M0)",
                "method_name": "ekf",
                "A": "A2",
                "N": "N2",
                "V": "V2",
                "G": "K1",
                "K": "K0",
                "rmse": 0.12,
            },
            {
                "case_ref": "case-2",
                "seq_id": "mini_seq",
                "scene_id": "S(A3,N2,V2,K0,M0)",
                "method_name": "ekf",
                "A": "A3",
                "N": "N2",
                "V": "V2",
                "G": "K1",
                "K": "K0",
                "rmse": 0.08,
            },
        ],
        {
            "figure_path": figure_path,
            "return_manifest": True,
        },
    )

    assert figure_path.is_file()
    assert manifest["x_axis"] == "A"
    assert manifest["y_axis"] == "rmse"


def test_bool_x_axis_is_not_treated_as_numeric(tmp_path):
    figure_path = tmp_path / "bool.svg"

    manifest = render_sweep_figure(
        [
            {"flag": True, "rmse": 0.12},
            {"flag": False, "rmse": 0.08},
        ],
        {
            "figure_path": figure_path,
            "x_axis": "flag",
            "y_axis": "rmse",
            "return_manifest": True,
        },
    )

    assert figure_path.is_file()
    assert manifest["kind"] == "bar"
