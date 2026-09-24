"""管线 API 接口测试模块。

本模块验证 PipelineAPI 抽象基类的接口契约，包括方法签名、
抽象方法集合和实例化限制。

测试覆盖范围：
  - 正常情况：DummyPipeline 的 run 方法行为
  - 边界情况：不传参数时使用默认值
  - 契约签名：抽象方法集合和参数签名验证
  - 异常情况：直接实例化 PipelineAPI 抛出 TypeError
  - 异常情况：缺少 run 实现时实例化抛出 TypeError

被测模块：
  - liquidloc.interfaces.pipeline_api
  - liquidloc.common.types（StageResult）
"""

from __future__ import annotations

import inspect

import pytest

from liquidloc.common.types import StageResult
from liquidloc.interfaces.pipeline_api import PipelineAPI


class DummyPipeline(PipelineAPI):
    """用于测试的 PipelineAPI 最小实现。

    run 方法从 pipeline_cfg 中提取 artifact 字段，
    返回包含该 artifact 的 StageResult。
    """
    def run(self, pipeline_cfg: dict | None = None, runtime_context: dict | None = None) -> StageResult:
        cfg = pipeline_cfg or {}
        return StageResult(stage_name="dummy", artifacts=[cfg.get("artifact", "a.txt")])


def test_normal_case():
    """验证 DummyPipeline 的 run 方法行为。

    测试场景：调用 run 传入 {"artifact": "x.txt"}。
    预期行为：返回 StageResult，stage_name 为 "dummy"，
    artifacts 为 ["x.txt"]。
    """
    pipeline = DummyPipeline()
    result = pipeline.run({"artifact": "x.txt"})
    assert result.stage_name == "dummy"
    assert result.artifacts == ["x.txt"]


def test_boundary_case():
    """验证不传参数时使用默认值。

    测试场景：调用 run 不传任何参数。
    预期行为：返回 StageResult，artifacts 为 ["a.txt"]（默认值）。
    """
    pipeline = DummyPipeline()
    result = pipeline.run()
    # 不传 pipeline_cfg 时使用默认 artifact 名称
    assert result.artifacts == ["a.txt"]


def test_contract_signature_case():
    """验证 PipelineAPI 的抽象方法集合和参数签名。

    测试场景：检查 PipelineAPI 的 __abstractmethods__ 和
    run 方法的参数签名。
    预期行为：抽象方法集合为 {"run"}，
    run 签名为 (self, pipeline_cfg, runtime_context)。
    """
    # 验证抽象方法集合
    assert PipelineAPI.__abstractmethods__ == {"run"}
    # 验证 run 方法的参数签名，确保接口契约不被意外修改
    assert tuple(inspect.signature(PipelineAPI.run).parameters) == (
        "self",
        "pipeline_cfg",
        "runtime_context",
    )


def test_invalid_case():
    """验证直接实例化 PipelineAPI 和缺少 run 实现时抛出 TypeError。

    测试场景：尝试直接实例化 PipelineAPI，以及实例化
    缺少 run 实现的子类。
    预期行为：直接实例化抛出 TypeError，
    缺少 run 的子类实例化抛出 TypeError 并包含 "run"。
    """
    # 抽象基类不能直接实例化
    with pytest.raises(TypeError):
        PipelineAPI()

    # 缺少 run 实现的子类也不能实例化
    class MissingRunPipeline(PipelineAPI):
        pass

    with pytest.raises(TypeError, match="run"):
        MissingRunPipeline()
