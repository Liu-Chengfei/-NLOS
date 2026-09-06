from __future__ import annotations

"""训练趋势绘图（plot_training_trends）测试模块。

文件职责：验证 render_training_trend_figure 能正确渲染
训练趋势图表。

测试覆盖范围：
- 正常场景：渲染训练趋势图并返回清单
- 异常场景：缺失 split 拒绝

被测模块：liquidloc.plotting.plot_training_trends"""


import pytest

from liquidloc.plotting.plot_training_trends import render_training_trend_figure


def _training_trend_report():
    return {
        "train": [
            {
                "epoch_index": 1,
                "mean_loss": 1.2,
                "selection_score": 1.2,
                "head_metrics": {
                    "bias": {"rmse": 0.9},
                    "risk": {"rmse": 0.7},
                },
            },
            {
                "epoch_index": 2,
                "mean_loss": 0.8,
                "selection_score": 0.8,
                "head_metrics": {
                    "bias": {"rmse": 0.6},
                    "risk": {"rmse": 0.5},
                },
            },
        ],
        "val": [
            {
                "epoch_index": 1,
                "mean_loss": 1.4,
                "selection_score": 1.4,
                "head_metrics": {
                    "bias": {"rmse": 1.0},
                    "risk": {"rmse": 0.8},
                },
            },
            {
                "epoch_index": 2,
                "mean_loss": 0.95,
                "selection_score": 0.95,
                "head_metrics": {
                    "bias": {"rmse": 0.7},
                    "risk": {"rmse": 0.6},
                },
            },
        ],
        "fixed_probe_trends": {
            "train": [
                {
                    "epoch_index": 1,
                    "fixed_probe_rows": [
                        {"sample_index_in_split": 0, "signed_error_by_head": {"bias": 0.2}},
                    ],
                },
                {
                    "epoch_index": 2,
                    "fixed_probe_rows": [
                        {"sample_index_in_split": 0, "signed_error_by_head": {"bias": 0.1}},
                    ],
                },
            ],
            "val": [],
        },
    }


def test_render_training_trend_figure_returns_manifest(tmp_path):
    manifest = render_training_trend_figure(
        _training_trend_report(),
        {"figure_path": tmp_path / "training_trends.svg", "return_manifest": True},
    )

    assert (tmp_path / "training_trends.svg").is_file()
    assert manifest["metric_key"] == "selection_score"
    assert manifest["panel_keys"] == [
        "selection_score",
        "rmse:bias",
        "rmse:risk",
        "fixed_probe_signed_error:bias",
    ]


def test_render_training_trend_figure_rejects_missing_split(tmp_path):
    with pytest.raises(ValueError, match="missing split"):
        render_training_trend_figure(
            {"train": []},
            {"figure_path": tmp_path / "training_trends.svg", "splits": ["train", "val"]},
        )
