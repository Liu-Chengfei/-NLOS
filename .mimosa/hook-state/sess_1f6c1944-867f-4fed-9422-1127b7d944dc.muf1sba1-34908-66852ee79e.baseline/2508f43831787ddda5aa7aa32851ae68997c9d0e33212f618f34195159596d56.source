from __future__ import annotations

"""运行时绘图（plot_runtime）测试模块。

文件职责：验证 render_runtime_figure 能正确渲染
运行时指标的 SVG 图表。

测试覆盖范围：
- 正常场景：渲染运行时图表并返回清单
- 异常场景：缺少运行时指标行
- 重复运行时指标行拒绝
- 全负运行时值处理
- 宽表行自动推断标签轴

被测模块：liquidloc.plotting.plot_runtime"""


import pytest

from liquidloc.plotting.plot_runtime import render_runtime_figure


def _runtime_metric_rows():
    rows = []
    for case_ref in ('case_a', 'case_b'):
        for metric_name, value, group, unit in (
            ('rmse', '0.1', 'primary', 'm'),
            ('latency_mean', '1.5', 'runtime', 'ms'),
            ('latency_p50', '1.5', 'runtime', 'ms'),
            ('latency_p95', '1.95', 'runtime', 'ms'),
            ('params', '0.0', 'runtime', 'count'),
            ('ram_peak', '0.0', 'runtime', 'MB'),
            ('coverage', '1.0', 'mechanism', 'ratio'),
        ):
            rows.append({
                'case_ref': case_ref,
                'metric': metric_name,
                'value': value,
                'unit': unit,
                'direction': 'lower_is_better' if group != 'mechanism' else 'higher_is_better',
                'group': group,
            })
    return rows


def test_normal_case(tmp_path):
    manifest = render_runtime_figure(
        _runtime_metric_rows(),
        {'figure_path': tmp_path / 'runtime.svg', 'return_manifest': True},
    )
    assert (tmp_path / 'runtime.svg').is_file()
    assert manifest['metrics'] == ['latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']


def test_invalid_case(tmp_path):
    with pytest.raises(ValueError):
        render_runtime_figure(
            [{'case_ref': 'case_a', 'metric': 'rmse', 'value': '0.0', 'group': 'primary'}],
            {'figure_path': tmp_path / 'runtime.svg'},
        )


def test_rejects_duplicate_runtime_metric_rows(tmp_path):
    duplicate_rows = _runtime_metric_rows()
    duplicate_rows.append(
        {
            'case_ref': 'case_a',
            'metric': 'latency_mean',
            'value': '2.0',
            'unit': 'ms',
            'direction': 'lower_is_better',
            'group': 'runtime',
        }
    )

    with pytest.raises(ValueError, match='duplicate runtime metric'):
        render_runtime_figure(
            duplicate_rows,
            {'figure_path': tmp_path / 'runtime.svg'},
        )


def test_handles_all_negative_runtime_values(tmp_path):
    rows = []
    for case_ref, latency in (('case_a', '-2.0'), ('case_b', '-1.0')):
        rows.append({
            'case_ref': case_ref,
            'metric': 'latency_mean',
            'value': latency,
            'unit': 'ms',
            'direction': 'lower_is_better',
            'group': 'runtime',
        })

    render_runtime_figure(rows, {'figure_path': tmp_path / 'runtime.svg'})
    svg = (tmp_path / 'runtime.svg').read_text(encoding='utf-8')
    assert 'stroke="#111827"' in svg
    assert 'x1="1032.00"' in svg


def test_infers_shared_label_axis_for_wide_rows(tmp_path):
    manifest = render_runtime_figure(
        [
            {
                'case_ref': 'case_a',
                'latency_mean': '1.0',
                'latency_p50': '1.0',
                'latency_p95': '1.0',
                'params': '0.0',
                'ram_peak': '0.0',
            },
            {
                'case_ref': 'case_b',
                'latency_mean': '2.0',
                'latency_p50': '2.0',
                'latency_p95': '2.0',
                'params': '0.0',
                'ram_peak': '0.0',
            },
        ],
        {'figure_path': tmp_path / 'runtime.svg', 'return_manifest': True},
    )
    assert manifest['label_axis'] == 'case_ref'
