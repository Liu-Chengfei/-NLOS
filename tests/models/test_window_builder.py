"""窗口构建器（window_builder）测试模块。

测试覆盖范围：
- 滑动窗口的构建与步进
- 窗口索引映射
- 事件时间窗口的正确性

被测模块：liquidloc.models.window_builder"""

import pytest

from liquidloc.models.features.window_builder import build_windows


def test_normal_case():
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    window_tensor, window_index_map = build_windows(
        [[1, 10], [2, 20], [3, 30], [4, 40]],
        window_size=2,
        step_size=2,
    )

    assert window_tensor == [[[1, 10], [2, 20]], [[3, 30], [4, 40]]]
    assert window_index_map == [[0, 1], [2, 3]]


def test_boundary_case():
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    window_tensor, window_index_map = build_windows([[1, 2]], window_size=2, step_size=1)

    assert window_tensor == []
    assert window_index_map == []


def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    with pytest.raises(ValueError, match="rectangular"):
        build_windows([[1, 2], [3]], window_size=2, step_size=1)
