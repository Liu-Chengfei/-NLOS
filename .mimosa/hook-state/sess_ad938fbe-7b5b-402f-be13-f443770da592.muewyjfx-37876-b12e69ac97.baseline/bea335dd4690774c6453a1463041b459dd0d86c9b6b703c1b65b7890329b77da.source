from __future__ import annotations

"""校准绘图（plot_calibration）测试模块。

文件职责：验证 render_calibration_figure 能正确渲染
机制指标的校准图表。

测试覆盖范围：
- 正常场景：渲染校准图表并返回清单
- 遗留报告兼容当前指标名和旧别名
- 仅旧别名的遗留报告兼容
- 异常场景：缺少机制指标行
- 负覆盖率拒绝
- 宽表模式缺失标签轴拒绝

被测模块：liquidloc.plotting.plot_calibration"""


import pytest

from liquidloc.plotting.plot_calibration import build_calibration_figure_spec, render_calibration_figure


def _mechanism_metric_rows():
    rows = []
    for case_ref in ('case_a', 'case_b'):
        for metric_name, value, group, unit in (
            ('rmse', '0.1', 'primary', 'm'),
            ('risk_error_corr', '0.8', 'mechanism', 'corr'),
            ('coverage', '1.0', 'mechanism', 'ratio'),
            ('bias_alignment', '0.6', 'mechanism', 'corr'),
            ('corr_scaling_error', '0.7', 'mechanism', 'corr'),
            ('latency_mean', '1.5', 'runtime', 'ms'),
        ):
            rows.append({
                'case_ref': case_ref,
                'metric': metric_name,
                'value': value,
                'unit': unit,
                'direction': 'higher_is_better' if group == 'mechanism' else 'lower_is_better',
                'group': group,
            })
    return rows


def test_normal_case(tmp_path):
    manifest = render_calibration_figure(
        _mechanism_metric_rows(),
        {'figure_path': tmp_path / 'calibration.svg', 'return_manifest': True},
    )
    assert (tmp_path / 'calibration.svg').is_file()
    assert manifest['metrics'] == ['risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error']


def test_legacy_report_prefers_current_metric_names_and_accepts_old_aliases(tmp_path):
    manifest = render_calibration_figure(
        {
            'bias_alignment': 0.6,
            'corr_bias_error': 0.1,
            'risk_error_corr': -0.2,
            'corr_risk_error': 0.9,
            'corr_scaling_error': 0.3,
        },
        {'figure_path': tmp_path / 'legacy_calibration.svg', 'return_manifest': True},
    )
    assert (tmp_path / 'legacy_calibration.svg').is_file()
    assert manifest['metrics'] == ['bias_alignment', 'risk_error_corr', 'corr_scaling_error']


def test_legacy_report_alias_only_payload_is_accepted(tmp_path):
    spec = build_calibration_figure_spec(
        {
            'corr_bias_error': 0.1,
            'corr_risk_error': -0.2,
            'corr_scaling_error': 0.3,
        },
        {'figure_path': tmp_path / 'legacy_alias.svg'},
    )
    assert spec['mode'] == 'legacy_report'
    assert [item['metric_name'] for item in spec['series']] == [
        'bias_alignment',
        'risk_error_corr',
        'corr_scaling_error',
    ]


def test_invalid_case(tmp_path):
    with pytest.raises(ValueError):
        render_calibration_figure(
            [{'case_ref': 'case_a', 'metric': 'rmse', 'value': '0.0', 'group': 'primary'}],
            {'figure_path': tmp_path / 'calibration.svg'},
        )


def test_rejects_negative_coverage(tmp_path):
    with pytest.raises(ValueError, match="metric_table\\['coverage'\\] must be >= 0.0"):
        render_calibration_figure(
            [
                {'case_ref': 'case_a', 'metric': 'coverage', 'value': '-0.1', 'group': 'mechanism'},
            ],
            {'figure_path': tmp_path / 'calibration.svg'},
        )


def test_rejects_missing_label_axis_in_wide_form(tmp_path):
    with pytest.raises(ValueError, match="Unable to infer label_axis"):
        render_calibration_figure(
            [
                {'risk_error_corr': '0.8', 'coverage': '1.0', 'bias_alignment': '0.6', 'corr_scaling_error': '0.7'},
                {'risk_error_corr': '0.7', 'coverage': '0.9', 'bias_alignment': '0.5', 'corr_scaling_error': '0.6'},
            ],
            {'figure_path': tmp_path / 'calibration.svg'},
        )
