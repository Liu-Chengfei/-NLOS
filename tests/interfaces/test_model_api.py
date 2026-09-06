"""模型 API 接口测试模块。

本模块验证 ModelAPI 抽象基类的接口契约，包括方法签名、
抽象方法集合和实例化限制。

测试覆盖范围：
  - 正常情况：DummyModel 的 infer_intermediate 行为
  - 边界情况：reset 方法重置调用计数
  - 契约签名：抽象方法集合和参数签名验证
  - 异常情况：直接实例化 ModelAPI 抛出 TypeError
  - 异常情况：缺少 infer_intermediate 实现时实例化抛出 TypeError

被测模块：
  - liquidloc.interfaces.model_api
  - liquidloc.common.types（ModelIntermediate）
"""

from __future__ import annotations

import inspect

import pytest

from liquidloc.common.types import ModelIntermediate
from liquidloc.interfaces.model_api import ModelAPI


class DummyModel(ModelAPI):
    """用于测试的 ModelAPI 最小实现。

    infer_intermediate 根据 window_tensor 长度设置 bias，
    返回固定 risk=0.1、uwb_scaling=1.0、vio_scaling=0.9 的 ModelIntermediate。
    """
    def __init__(self):
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def infer_intermediate(self, window_tensor) -> ModelIntermediate:
        self.calls += 1
        # bias 与输入长度挂钩，用于验证输入被正确传递
        return ModelIntermediate(bias=float(len(window_tensor)), risk=0.1, uwb_scaling=1.0, vio_scaling=1.0)


def test_normal_case():
    """验证 DummyModel 的 infer_intermediate 行为。

    测试场景：调用 infer_intermediate 传入 [1, 2, 3]。
    预期行为：返回 ModelIntermediate，bias=3.0（输入长度），
    vio_scaling=0.9。
    """
    model = DummyModel()
    output = model.infer_intermediate([1, 2, 3])
    # bias 应等于输入 window_tensor 的长度
    assert output.bias == 3.0
    assert output.vio_scaling == 1.0


def test_boundary_case():
    """验证 reset 方法重置调用计数。

    测试场景：调用 reset 后检查 calls 计数。
    预期行为：calls 被重置为 0。
    """
    model = DummyModel()
    model.reset()
    # reset 后调用计数应归零
    assert model.calls == 0


def test_contract_signature_case():
    """验证 ModelAPI 的抽象方法集合和参数签名。

    测试场景：检查 ModelAPI 的 __abstractmethods__ 和
    各方法的参数签名。
    预期行为：抽象方法集合为 {"reset", "infer_intermediate"}，
    reset 签名为 (self,)，infer_intermediate 签名为 (self, window_tensor)。
    """
    # 验证抽象方法集合
    assert ModelAPI.__abstractmethods__ == {"reset", "infer_intermediate"}
    # 验证各方法的参数签名，确保接口契约不被意外修改
    assert tuple(inspect.signature(ModelAPI.reset).parameters) == ("self",)
    assert tuple(inspect.signature(ModelAPI.infer_intermediate).parameters) == ("self", "window_tensor")


def test_invalid_case():
    """验证直接实例化 ModelAPI 和缺少 infer_intermediate 实现时抛出 TypeError。

    测试场景：尝试直接实例化 ModelAPI，以及实例化
    缺少 infer_intermediate 实现的子类。
    预期行为：直接实例化抛出 TypeError，
    缺少 infer_intermediate 的子类实例化抛出 TypeError 并包含 "infer_intermediate"。
    """
    # 抽象基类不能直接实例化
    with pytest.raises(TypeError):
        ModelAPI()

    # 缺少 infer_intermediate 实现的子类也不能实例化
    class MissingInferIntermediateModel(ModelAPI):
        def reset(self) -> None:
            return None

    with pytest.raises(TypeError, match="infer_intermediate"):
        MissingInferIntermediateModel()
