"""
文件：tests/common/test_io_utils.py

【文件职责】
为 `io_utils` 提供正常、边界、异常三类测试，作为 common 层最小门禁。

【本文件绝对不负责】
不负责实现业务模块，不负责替代 protocol 或 integration 测试。

【上游依赖】
待测模块：src/liquidloc/common/io_utils.py

【下游调用者】
pytest、本地 smoke、本地回归门禁。

【输入对象定义】
- 手工构造的小样本或最小对象

【输出对象定义】
- pytest 通过或按预期抛异常

【核心变量定义】
- input_obj
- expected_output
- raised_exception

【推荐编写顺序】
1. 先写正常测试。
2. 再写边界测试。
3. 最后写异常测试。

【建议先写的函数 / 类】
- 正常测试
  签名：def test_normal_case()
  作用：验证正常路径。
  输入：小样本
  输出：pytest 通过
  关键局部变量：
    - input_obj
    - output_obj
  伪代码：
    1) 构造最小正常输入。
    2) 调用待测函数。
    3) 断言关键输出存在。

- 异常测试
  签名：def test_invalid_case()
  作用：验证异常路径。
  输入：非法样本
  输出：pytest 通过
  关键局部变量：
    - bad_input
  伪代码：
    1) 构造非法输入。
    2) 调用待测函数。
    3) 断言抛出清晰异常。

【最容易让 Codex 理解错的地方】
- 不要把 common 层测试写成业务大闭环。

【最小手工测试步骤】
1. 直接运行 pytest 指定 `tests/common/test_io_utils.py`。

【完成标准】
- common 层至少有正常和异常两类最小门禁。

【实现要求】
- 当前阶段保持空白实现，只保留细化编写说明。
- 真正实现时先写签名和 docstring，再写前置检查，再写主体逻辑，最后补测试。
- 任何字段名和变量名优先服从 protocol 和 configs，不能临时发明。
"""

from __future__ import annotations

import json

import pytest

from liquidloc.common.io_utils import read_json, write_json


def test_normal_case(tmp_path):
    input_obj = {"sensor": "uwb", "count": 1}
    output_obj = tmp_path / "nested" / "config.json"

    write_json(output_obj, input_obj)

    assert output_obj.is_file()
    assert read_json(output_obj) == input_obj


def test_invalid_case(tmp_path):
    bad_input = tmp_path / "broken.json"
    bad_input.write_text("{bad json}", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_json(bad_input)


@pytest.mark.parametrize("func,args", [(read_json, (None,)), (write_json, (None, {}))])
def test_path_must_not_be_none(func, args):
    with pytest.raises(ValueError, match="path must not be None"):
        func(*args)
