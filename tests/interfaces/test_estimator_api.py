"""估计器 API 接口测试模块。

本模块验证 EstimatorAPI 抽象基类的接口契约，包括方法签名、
抽象方法集合和实例化限制。

测试覆盖范围：
  - 正常情况：DummyEstimator 的 step 和 get_state 行为
  - 边界情况：reset 方法重置初始状态
  - 契约签名：抽象方法集合和参数签名验证
  - 异常情况：直接实例化 EstimatorAPI 抛出 TypeError
  - 异常情况：缺少 get_state 实现时实例化抛出 TypeError

被测模块：
  - liquidloc.interfaces.estimator_api
  - liquidloc.common.types（StateEstimate）
"""

from __future__ import annotations

import inspect

import pytest

from liquidloc.common.types import StateEstimate
from liquidloc.interfaces.estimator_api import EstimatorAPI


class DummyEstimator(EstimatorAPI):
    """用于测试的 EstimatorAPI 最小实现。

    step 方法将 event["t"] 作为 px 写入状态，
    get_state 返回当前状态，reset 重置到指定或默认状态。
    """
    def __init__(self):
        self._state = StateEstimate(state={"px": 0.0}, covariance_diag=[1.0], timestamp=0.0)

    def reset(self, initial_state: dict | None = None) -> None:
        state = {"px": 0.0} if initial_state is None else initial_state
        self._state = StateEstimate(state=state, covariance_diag=[1.0], timestamp=0.0)

    def step(self, event) -> StateEstimate:
        self._state = StateEstimate(state={"px": event["t"]}, covariance_diag=[1.0], timestamp=event["t"])
        return self._state

    def get_state(self) -> StateEstimate:
        return self._state


def test_normal_case():
    """验证 DummyEstimator 的 step 和 get_state 行为。

    测试场景：调用 step 传入 {"t": 0.2}，然后调用 get_state。
    预期行为：step 返回的 StateEstimate 的 timestamp 为 0.2，
    get_state 返回的状态中 px 为 0.2。
    """
    estimator = DummyEstimator()
    state = estimator.step({"t": 0.2})
    assert state.timestamp == 0.2
    # step 后状态应持久化，get_state 返回相同状态
    assert estimator.get_state().state["px"] == 0.2


def test_boundary_case():
    """验证 reset 方法重置初始状态。

    测试场景：调用 reset({"px": 1.0})，然后调用 get_state。
    预期行为：get_state 返回的状态中 px 为 1.0。
    """
    estimator = DummyEstimator()
    estimator.reset({"px": 1.0})
    assert estimator.get_state().state["px"] == 1.0


def test_contract_signature_case():
    """验证 EstimatorAPI 的抽象方法集合和参数签名。

    测试场景：检查 EstimatorAPI 的 __abstractmethods__ 和
    各方法的参数签名。
    预期行为：抽象方法集合为 {reset, step, get_state}，
    reset 签名为 (self, initial_state)，step 签名为 (self, event)，
    get_state 签名为 (self,)。
    """
    # 验证抽象方法集合完整
    assert EstimatorAPI.__abstractmethods__ == {"reset", "step", "get_state"}
    # 验证各方法的参数签名，确保接口契约不被意外修改
    assert tuple(inspect.signature(EstimatorAPI.reset).parameters) == ("self", "initial_state")
    assert tuple(inspect.signature(EstimatorAPI.step).parameters) == ("self", "event")
    assert tuple(inspect.signature(EstimatorAPI.get_state).parameters) == ("self",)


def test_invalid_case():
    """验证直接实例化 EstimatorAPI 和缺少方法实现时抛出 TypeError。

    测试场景：尝试直接实例化 EstimatorAPI，以及实例化
    缺少 get_state 实现的子类。
    预期行为：直接实例化抛出 TypeError，
    缺少 get_state 的子类实例化抛出 TypeError 并包含 "get_state"。
    """
    # 抽象基类不能直接实例化
    with pytest.raises(TypeError):
        EstimatorAPI()

    # 缺少 get_state 实现的子类也不能实例化
    class MissingGetStateEstimator(EstimatorAPI):
        def reset(self, initial_state: dict | None = None) -> None:
            return None

        def step(self, event) -> StateEstimate:
            return StateEstimate()

    with pytest.raises(TypeError, match="get_state"):
        MissingGetStateEstimator()
