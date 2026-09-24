from __future__ import annotations

"""随机种子工具（seed_utils）测试模块。

文件职责：验证 set_global_seed 能正确设置全局随机种子
并确保可复现性。

测试覆盖范围：
- 正常场景：重置 Python 随机状态确保可复现
- 异常场景：布尔种子、负数种子、超大种子、非法 deterministic 参数

被测模块：liquidloc.common.seed_utils"""


import random

import pytest

from liquidloc.common.seed_utils import set_global_seed


def test_set_global_seed_resets_python_random_state():
    report = set_global_seed(123, deterministic=True)
    first = random.random()

    set_global_seed(123, deterministic=True)
    second = random.random()

    assert first == second
    assert report["seed"] == 123
    assert report["deterministic"] is True
    assert report["python_seeded"] is True
    assert isinstance(report["numpy_seeded"], bool)
    assert isinstance(report["torch_seeded"], bool)


@pytest.mark.parametrize(
    ("seed", "deterministic", "expected_exception"),
    [
        (True, True, TypeError),
        (-1, True, ValueError),
        (2**32, True, ValueError),
        (0, "yes", TypeError),
    ],
)
def test_set_global_seed_rejects_invalid_inputs(seed, deterministic, expected_exception):
    with pytest.raises(expected_exception):
        set_global_seed(seed, deterministic=deterministic)
