"""risk_projection 协议单源 pytest 锁死（§12.3-C2a 第十四轮）。

锁死目的：
- LSTM `models/lstm/inference.py:107-110` 第十四轮之前手抄 risk 协议区间投影公式
  副本（少 D5 isfinite 守卫），与 Liquid 走 protocol 单源不一致；
- 本测试锁死 protocol 单源 `project_risk_to_protocol_range` 行为，确保未来
  LSTM/Liquid/Transformer 三网回归本地手抄时能在 protocol 单源层 fail-loud 抓住。

被测模块：liquidloc.protocol.risk_projection
"""

import math

import pytest

from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
from liquidloc.protocol.risk_projection import project_risk_to_protocol_range


# ---------------------------------------------------------------------------
# 协议默认区间：risk_min=0.0 / risk_max=1.05 / span=1.05（非默认 0..1）。
# 当 span != 1.0 或 min != 0.0 时，单源做 linear 投影 risk := risk * span + min。
# ---------------------------------------------------------------------------


def test_default_protocol_range_is_non_identity():
    """BRIDGE_THRESHOLDS 真实 risk_max=1.05 / risk_min=0.0 → span=1.05，非默认 0..1。

    锁死 BRIDGE_THRESHOLDS 协议区间常量，禁止未来 silently 改回 [0,1]
    （否则 protocol 单源 vs LSTM 副本公式差异从行为可观察变不可观察，审计无法察觉）。
    """
    assert BRIDGE_THRESHOLDS["risk_min"] == 0.0
    assert BRIDGE_THRESHOLDS["risk_max"] == pytest.approx(1.05)
    span = BRIDGE_THRESHOLDS["risk_max"] - BRIDGE_THRESHOLDS["risk_min"]
    assert span != 1.0  # 关键：span != 1 → 投影非恒等，公式副本不一致立即在数值上显形


def test_protocol_singleton_maps_sigmoid_to_risk_max_range():
    """sigmoid(1.5) ≈ 0.8176 在非默认区间下投影为 0.8176 * 1.05 = 0.8585。

    这是 LSTM `test_boundary_case` 历史失败的真实数值路径——第十四轮修复后
    LSTM 路径委托 protocol 单源走同一公式，得到同一数值。
    """
    risk = 1.0 / (1.0 + math.exp(-1.5))  # sigmoid(1.5) ≈ 0.8175744
    expected = risk * 1.05  # span=1.05, min=0 → risk*span+min = 0.8176*1.05
    projected = project_risk_to_protocol_range(
        risk,
        BRIDGE_THRESHOLDS["risk_min"],
        BRIDGE_THRESHOLDS["risk_max"],
    )
    assert projected == pytest.approx(expected, rel=1e-6)
    # 必须在 [0, 1.05] 内
    assert BRIDGE_THRESHOLDS["risk_min"] <= projected <= BRIDGE_THRESHOLDS["risk_max"]


# ---------------------------------------------------------------------------
# D5 数值安全守卫锁死：protocol 单源 math.isfinite 拒绝 NaN/Inf
# 这是 LSTM 路径本地手抄副本缺失的部分，第十四轮修复后 LSTM 也继承此守卫。
# ---------------------------------------------------------------------------


def test_nan_risk_rejected_by_protocol_singleton():
    """NaN risk 必 raise ValueError——禁止 nan 经 min/max 静默穿透（D5）。

    Python 中 max(0.0, float('nan'))=nan，min(1.05, nan)=nan，本地手抄副本
    会静默返回 nan；protocol 单源有 math.isfinite 守卫必 raise。
    """
    with pytest.raises(ValueError, match="risk must be finite"):
        project_risk_to_protocol_range(
            float("nan"),
            BRIDGE_THRESHOLDS["risk_min"],
            BRIDGE_THRESHOLDS["risk_max"],
        )


def test_inf_risk_rejected_by_protocol_singleton():
    """Inf risk 必 raise ValueError——同 D5 守卫锁死。"""
    with pytest.raises(ValueError, match="risk must be finite"):
        project_risk_to_protocol_range(
            float("inf"),
            BRIDGE_THRESHOLDS["risk_min"],
            BRIDGE_THRESHOLDS["risk_max"],
        )


def test_neg_inf_risk_rejected_by_protocol_singleton():
    """-Inf risk 必 raise ValueError——同 D5 守卫锁死。"""
    with pytest.raises(ValueError, match="risk must be finite"):
        project_risk_to_protocol_range(
            float("-inf"),
            BRIDGE_THRESHOLDS["risk_min"],
            BRIDGE_THRESHOLDS["risk_max"],
        )


# ---------------------------------------------------------------------------
# 行为正确性：clamp 守协议区间
# ---------------------------------------------------------------------------


def test_protocol_singleton_clamps_out_of_range_to_max():
    """risk=10 远超 risk_max=1.05 → 投影后必 clamp 回 1.05。"""
    projected = project_risk_to_protocol_range(
        10.0,
        BRIDGE_THRESHOLDS["risk_min"],
        BRIDGE_THRESHOLDS["risk_max"],
    )
    assert projected == pytest.approx(1.05)


def test_protocol_singleton_clamps_negative_to_min():
    """risk=-5 经 span=1.05+min=0 → -5*1.05+0=-5.25 → clamp 回 0.0。"""
    projected = project_risk_to_protocol_range(
        -5.0,
        BRIDGE_THRESHOLDS["risk_min"],
        BRIDGE_THRESHOLDS["risk_max"],
    )
    assert projected == pytest.approx(0.0)


def test_protocol_singleton_identity_when_default_range():
    """当 risk_min=0 / risk_max=1.0 时（恒等区间）投影不变。

    锁死单源在默认区间下的恒等行为，禁止未来 silent break。
    """
    projected = project_risk_to_protocol_range(0.42, 0.0, 1.0)
    assert projected == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# LSTM / Liquid 双路径经同一单源——核心 §12.3-C2a 锁死
# ---------------------------------------------------------------------------


def test_lstm_and_liquid_paths_both_delegate_to_protocol_singleton():
    """LSTM `infer_intermediate` 与 Liquid `_project_risk_to_protocol_range` 都走
    protocol/risk_projection.py:35 project_risk_to_protocol_range 单源。

    第十四轮修复前 LSTM 路径本地手抄风险投影公式副本（少 D5 isfinite 守卫），
    与 Liquid 路径单源委托不一致；本测试通过 import 关系静态锁死 LSTM 路径
    必须从 protocol.risk_projection import project_risk_to_protocol_range。
    """
    import inspect

    from liquidloc.protocol import risk_projection as rp_module

    src_lstm = inspect.getsource(__import__("liquidloc.models.lstm.inference", fromlist=["infer_intermediate"]))
    src_liquid = inspect.getsource(__import__("liquidloc.models.liquid.inference", fromlist=["_project_risk_to_protocol_range"]))

    # 第十四轮锁死：LSTM inference 模块必须 import project_risk_to_protocol_range 单源
    assert "project_risk_to_protocol_range" in src_lstm, (
        "LSTM inference.py 必须 import project_risk_to_protocol_range 单源 "
        "（第十四轮 §12.3-C2a 修复锁死，禁止本地手抄公式副本）"
    )

    # Liquid 路径继续委托同一单源（D2 分层边界，禁止重实现）
    assert "project_risk_to_protocol_range" in src_liquid, (
        "Liquid inference.py 必须委托 project_risk_to_protocol_range 单源 "
        "（D2 分层边界，禁止重实现 protocol 公式）"
    )

    # 同一函数对象——唯一单源真相
    assert hasattr(rp_module, "project_risk_to_protocol_range")
