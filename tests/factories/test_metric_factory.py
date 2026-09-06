
"""指标工厂（metric_factory）测试模块。

文件职责：验证 create_metric_factory 能根据指标名称正确创建指标计算器，
并确保非法名称被拒绝。

测试覆盖范围：
- 正常场景：创建合法指标（rmse）并计算
- 边界场景：创建 coverage 指标，验证分组字段
- 异常场景：传入未知指标名称触发 KeyError

被测模块：liquidloc.factories.metric_factory"""

from liquidloc.factories.metric_factory import create_metric_calculator


def test_normal_case():
    """测试正常场景：创建 rmse 指标计算器并验证计算结果。

验证指标名称和计算值正确返回。"""
    metric = create_metric_calculator('rmse', {})  # 通过工厂方法创建 rmse 指标计算器
    row = metric.compute(0.5)  # 计算指标值
    assert row.metric == 'rmse'  # 验证返回的指标名称正确
    assert row.value == 0.5  # 验证计算值与输入一致


def test_boundary_case():
    """测试边界场景：创建 coverage 指标，验证其分组为 mechanism。

覆盖率的分组属于机制类指标，而非主指标。"""
    metric = create_metric_calculator('coverage', {})  # 创建 coverage 指标计算器
    row = metric.compute(1.0)
    assert row.group == 'mechanism'  # coverage 属于 mechanism 分组


def test_invalid_case():
    """测试异常场景：传入非法指标名称 'bad_metric'，应抛出 ValueError。

确保工厂方法不会静默接受未知指标。"""
    try:
        create_metric_calculator('bad_metric', {})  # 传入非法指标名称
    except ValueError as exc:
        assert 'Unknown metric' in str(exc)  # 错误信息应包含 'Unknown metric'
    else:
        raise AssertionError('Expected invalid metric name to raise ValueError')  # 若未抛出异常则测试失败
