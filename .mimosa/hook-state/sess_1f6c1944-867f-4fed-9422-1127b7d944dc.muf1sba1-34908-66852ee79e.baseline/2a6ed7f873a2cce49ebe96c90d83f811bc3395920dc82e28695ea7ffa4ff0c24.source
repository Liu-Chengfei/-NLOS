"""校准分析模块测试。

本模块验证 build_calibration_report 函数的正确性，该函数根据
bias、risk、scaling 和 error 的时序轨迹计算校准指标。

测试覆盖范围：
  - 正常情况：完整轨迹输入，计算 bias_alignment、risk_error_corr、corr_scaling_error
  - 边界情况：轨迹中包含 None 值（缺失数据），所有指标退化为 0.0
  - 异常情况：轨迹长度不一致时抛出 ValueError
  - 异常情况：轨迹索引重复时抛出 ValueError

被测模块：
  - liquidloc.analysis.calibration_analysis
"""

from __future__ import annotations

import pytest

from liquidloc.analysis.calibration_analysis import build_calibration_report


def test_normal_case():
    """验证完整轨迹输入时校准指标的正确计算。

    测试场景：传入完美线性相关的 bias_trace、risk_trace、scaling_trace 和 error_trace。
    预期行为：三个校准指标（bias_alignment、risk_error_corr、corr_scaling_error）
    均为 1.0，表示完美校准。
    """
    report = build_calibration_report(
        bias_trace=[0.0, 1.0],
        risk_trace=[0.0, 1.0],
        scaling_trace=[1.0, 2.0],
        error_trace=[0.0, 1.0],
    )
    # bias 与 error 完美相关
    assert report['bias_alignment'] == pytest.approx(1.0)
    # risk 与 error 完美相关
    assert report['risk_error_corr'] == pytest.approx(1.0)
    # scaling 与 error 完美相关
    assert report['corr_scaling_error'] == pytest.approx(1.0)


def test_boundary_case():
    """验证轨迹中包含 None 值时，所有校准指标退化为 0.0。

    测试场景：第二个时间步的 bias、risk、scaling 均为 None，
    表示缺失数据。
    预期行为：无法计算有效相关系数，所有指标退化为 0.0。
    """
    report = build_calibration_report(
        bias_trace=[0.0, None],
        risk_trace=[0.0, None],
        scaling_trace=[1.0, None],
        error_trace=[0.0, 1.0],
    )
    # 缺失数据导致无法计算有效相关系数，指标退化为 0.0
    assert report == {
        'bias_alignment': 0.0,
        'risk_error_corr': 0.0,
        'corr_scaling_error': 0.0,
    }


def test_invalid_case():
    """验证轨迹长度不一致时抛出 ValueError。

    测试场景：risk_trace 长度（1）与其他轨迹长度（2）不一致。
    预期行为：抛出 ValueError，提示长度和索引必须一致。
    """
    with pytest.raises(ValueError, match='same length and index'):
        build_calibration_report(
            bias_trace=[0.0, 1.0],
            risk_trace=[0.0],
            scaling_trace=[1.0, 2.0],
            error_trace=[0.0, 1.0],
        )


def test_duplicate_bias_indices():
    """验证轨迹索引重复时抛出 ValueError。

    测试场景：使用自定义 DuplicateKeyTrace 类模拟索引为 0 的重复条目。
    预期行为：抛出 ValueError，提示存在重复索引。
    """
    # 自定义类模拟具有重复键的轨迹数据。
    class DuplicateKeyTrace:
        def items(self):
            return [(0, 0.0), (0, 1.0)]

    with pytest.raises(ValueError, match='duplicated indices'):
        build_calibration_report(
            bias_trace=DuplicateKeyTrace(),
            risk_trace={0: 0.0},
            scaling_trace={0: 1.0},
            error_trace={0: 0.0},
        )
