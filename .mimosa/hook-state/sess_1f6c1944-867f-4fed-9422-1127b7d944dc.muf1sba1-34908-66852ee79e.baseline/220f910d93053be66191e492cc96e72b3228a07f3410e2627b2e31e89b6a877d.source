from __future__ import annotations

"""事件模式（event_schema）测试模块。

测试覆盖范围：
- 事件结构的字段验证
- 模态类型与载荷匹配
- 元数据（scene_id/seq_id/source_t）的约束

被测模块：liquidloc.protocol.event_schema"""

import pytest

from liquidloc.protocol.event_schema import Event, validate_event, validate_event_sequence



def _imu_event(t: float = 0.0, dt: float = 0.0):
    return Event(
        t=t,
        dt=dt,
        modality="imu",
        meta={"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "seq_0"},
        imu_payload={"ax": 0.1, "ay": 0.0, "gz": 0.0},
    )



def test_normal_case():
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    validate_event(_imu_event())
    validate_event(
        {
            "t": 0.1,
            "dt": 0.1,
            "modality": "uwb",
            "meta": {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "seq_0"},
            "imu_payload": None,
            "uwb_payload": {"anchor_id": 1, "range": 2.2, "valid": True, "quality": 0.8},
            "vio_payload": None,
        }
    )



def test_boundary_case():
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    events = [
        _imu_event(),
        {
            "t": 0.1,
            "dt": 0.1,
            "modality": "vio",
            "meta": {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "seq_0"},
            "imu_payload": None,
            "uwb_payload": None,
            "vio_payload": {
                "dx": 0.0,
                "dy": 0.0,
                "dyaw": 0.0,
                "quality": 1.0,
            },
        },
    ]
    validate_event_sequence(events)



def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    bad = _imu_event().to_dict()
    bad["uwb_payload"] = {"anchor_id": 1, "range": 1.0, "valid": True, "quality": 0.5}
    with pytest.raises(ValueError):
        validate_event(bad)

    bad_uwb = {
        "t": 0.1,
        "dt": 0.1,
        "modality": "uwb",
        "meta": {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "seq_0"},
        "imu_payload": None,
        "uwb_payload": {"anchor_id": 1, "range": 1.0, "valid": "yes", "quality": 0.5},
        "vio_payload": None,
    }
    with pytest.raises(TypeError):
        validate_event(bad_uwb)

    bad_vio = {
        "t": 0.1,
        "dt": 0.1,
        "modality": "vio",
        "meta": {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "seq_0"},
        "imu_payload": None,
        "uwb_payload": None,
        "vio_payload": {
            "dx": 0.0,
            "dy": 0.0,
            "dyaw": 0.0,
            # quality 超过 [0,1] 上界, 必须被 reject — 仍是 4 字段协议下的合法校验.
            "quality": 2.0,
        },
    }
    with pytest.raises(ValueError):
        validate_event(bad_vio)

    # 铁律 3 (Stage A1 下游修复, 2026-07-23): tracked_features 已从 VIO 协议
    # 删除 — 旧 bad_vio_fractional_features (tracked_features=1.5) 测试不再
    # 适用, 已删除.

    bad_seq = [_imu_event(), _imu_event(t=0.05, dt=0.1)]
    with pytest.raises(ValueError):
        validate_event_sequence(bad_seq)


def test_empty_sequence_rejected():
    """空事件序列应被显式拒绝。"""
    with pytest.raises(ValueError, match="must not be empty"):
        validate_event_sequence([])


def test_imu_nan_rejected():
    """IMU payload 的 ax/ay/gz 为 nan 时应被拒绝。"""
    bad_imu = _imu_event().to_dict()
    bad_imu["imu_payload"]["ax"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_event(bad_imu)


def test_imu_inf_rejected():
    """IMU payload 的 ax/ay/gz 为 inf 时应被拒绝。"""
    bad_imu = _imu_event().to_dict()
    bad_imu["imu_payload"]["gz"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_event(bad_imu)


def test_imu_negative_values_allowed():
    """IMU 的加速度/角速度可以为负值。"""
    neg_imu = Event(
        t=0.0, dt=0.0, modality="imu",
        meta={"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "seq_0"},
        imu_payload={"ax": -9.8, "ay": -0.5, "gz": -1.0},
    )
    validate_event(neg_imu)  # 不应报错。
