from __future__ import annotations

"""通用校验工具（validation）测试模块。

文件职责：验证 require_not_none、require_in_range、
require_keys、require_shape、require_iterable 的正确性。

测试覆盖范围：
- 正常场景：所有校验函数通过合法输入
- 异常场景：None 值、超范围、缺失键、类型错误、形状不匹配

被测模块：liquidloc.common.validation"""


import pytest

from liquidloc.common.validation import (
    require_in_range,
    require_iterable,
    require_keys,
    require_not_none,
    require_shape,
)



def test_normal_case():
    require_not_none(1, "x")
    require_in_range(0.5, "ratio", min_value=0.0, max_value=1.0)
    require_keys({"a": 1, "b": 2}, ["a", "b"])
    require_shape([1, 2, 3], (3,))
    require_iterable([{"a": 1}])



def test_invalid_case():
    with pytest.raises(ValueError, match="x must not be None"):
        require_not_none(None, "x")
    with pytest.raises(ValueError, match="ratio must be <= 1.0, got 2.0"):
        require_in_range(2.0, "ratio", max_value=1.0)
    with pytest.raises(TypeError, match="ratio must be numeric, got str"):
        require_in_range("bad", "ratio")
    with pytest.raises(TypeError, match="ratio must be numeric, got bool"):
        require_in_range(True, "ratio")
    with pytest.raises(KeyError, match="mapping is missing required keys: \\['b'\\]"):
        require_keys({"a": 1}, ["a", "b"])
    with pytest.raises(TypeError, match="mapping must be a mapping, got list"):
        require_keys(["a", "b"], ["a"])
    with pytest.raises(TypeError, match="value does not expose a shape"):
        require_shape("abc", (3,))
    with pytest.raises(ValueError, match="value has shape \\(2,\\), expected \\(3,\\)"):
        require_shape([1, 2], (3,))
    with pytest.raises(TypeError, match="value must be a non-string iterable"):
        require_iterable("abc")
