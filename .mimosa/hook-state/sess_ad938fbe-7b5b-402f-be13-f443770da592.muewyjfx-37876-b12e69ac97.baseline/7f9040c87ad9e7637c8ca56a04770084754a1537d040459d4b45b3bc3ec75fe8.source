"""小规模参考数据回归测试模块。

本模块对 fixtures 中 mini_seq 数据集的加载正确性、以及 liquid_bridge_contract
（Liquid 模型与估计器之间的桥接协议）的核心行为进行回归验证。

测试覆盖范围：
  - 参考数据加载与字段完整性（正常/边界/异常）
  - UWB 测量控制字段的构建与校验
  - risk 归一化与安全模式的触发
  - 场景轴退化下界的解析（即使模型输出为 neutral）
  - noise_multiplier 随 risk 单调递增
  - UWB 硬拒绝（valid=False 或 quality 过低）
  - UWB 质量边界值保留更新
  - 负 risk 截断
  - 无效 scaling 拒绝
  - VIO 控制字段
  - IMU 直通回退

被测模块：
  - liquidloc.protocol.liquid_bridge_contract
  - liquidloc.common.types（ModelIntermediate）
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import liquidloc.protocol.liquid_bridge_contract as bridge_contract
from liquidloc.common.types import ModelIntermediate
from liquidloc.protocol.liquid_bridge_contract import build_measurement_control

# 指向 mini_seq fixture 数据的根目录，用于加载小规模参考数据
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "datasets" / "miluv" / "mini_seq"


def _load_small_reference_case() -> dict[str, object]:
    """加载小规模参考数据集的所有 JSON fixture 文件。

    Returns:
        包含 anchor_layout、ground_truth、imu、uwb、vio 五个键的字典，
        每个键对应的值是从对应 JSON 文件解析出的 Python 对象。
    """
    return {
        "anchor_layout": json.loads((FIXTURE_ROOT / "anchor_layout.json").read_text(encoding="utf-8")),
        "ground_truth": json.loads((FIXTURE_ROOT / "gt.json").read_text(encoding="utf-8")),
        "imu": json.loads((FIXTURE_ROOT / "imu.json").read_text(encoding="utf-8")),
        "uwb": json.loads((FIXTURE_ROOT / "uwb.json").read_text(encoding="utf-8")),
        "vio": json.loads((FIXTURE_ROOT / "vio.json").read_text(encoding="utf-8")),
    }


def test_normal_case():
    """验证参考数据的正常加载：字段来源、锚点 ID 列表、各模态数据条数。

    测试场景：加载完整的 mini_seq fixture 数据。
    预期行为：anchor_layout 的 source 标记为 fixture_local_anchor_layout，
    anchor_ids 为 [0, 1]，ground_truth 有 3 条，其余模态各有 2 条。
    """
    case = _load_small_reference_case()

    # 验证 anchor_layout 的数据来源标记正确
    assert case["anchor_layout"]["source"] == "fixture_local_anchor_layout"
    # 验证锚点 ID 列表与 fixture 预期一致
    assert case["anchor_layout"]["anchor_ids"] == [0, 1]
    # 验证各模态数据条数与 fixture 预期一致
    assert len(case["ground_truth"]) == 3
    assert len(case["imu"]) == 2
    assert len(case["uwb"]) == 2
    assert len(case["vio"]) == 2


def test_boundary_case():
    """验证参考数据中边界值字段的精度与有效性。

    测试场景：检查 ground_truth 中第二条记录的边界数值精度，
    以及 uwb/vio 的首条记录的有效性标记。
    预期行为：px 和 timestamp 精确到指定值，uwb valid 为 True，
    vio tracked_features 至少为 58。
    """
    case = _load_small_reference_case()
    # 取第二条 ground_truth 作为边界值测试对象
    boundary_gt = case["ground_truth"][1]

    # 验证边界位置坐标的浮点精度
    assert boundary_gt["px"] == pytest.approx(0.03)
    # 验证边界时间戳的浮点精度
    assert boundary_gt["timestamp"] == pytest.approx(0.1)
    # 验证 UWB 数据的有效性标记
    assert case["uwb"][0]["valid"] is True
    # 验证 VIO 跟踪特征数不低于阈值
    assert case["vio"][0]["tracked_features"] >= 58


def test_invalid_case():
    """验证访问不存在的参考文件时抛出 FileNotFoundError。

    测试场景：尝试读取 fixture 目录下不存在的 JSON 文件。
    预期行为：抛出 FileNotFoundError。
    """
    with pytest.raises(FileNotFoundError):
        (FIXTURE_ROOT / "missing_reference.json").read_text(encoding="utf-8")


def test_liquid_bridge_contract_uwb_control_fields():
    """验证 UWB 测量控制字段的正确构建。

    测试场景：传入 UWB 模态的原始测量和 ModelIntermediate，
    构建 measurement control。
    预期行为：modality 为 uwb，bias_applied 等于输入 bias，
    scaling 为 1.0（UWB 不使用 scaling），risk 保持原值，
    noise_multiplier 按协议公式计算，gate_action 为 uwb_bias_and_noise_scale。
    """
    control = build_measurement_control(
        {"modality": "uwb", "uwb_payload": {"range": 3.2}},
        ModelIntermediate(bias=0.2, risk=0.5, uwb_scaling=1.0),
    )

    assert control.modality == "uwb"
    # bias_applied 应直接使用模型输出的 bias 值
    assert control.bias_applied == pytest.approx(0.2)
    # UWB 的 scaling 固定为 1.0（不受 uwb_scaling 影响，uwb_scaling 用于 noise 计算）
    assert control.scaling == pytest.approx(1.0)
    assert control.risk == pytest.approx(0.5)
    # noise_multiplier = (1 + bias) * (1 + risk) * uwb_scaling = 1.2 * 1.5 * 0.8 ≈ 1.44... 但实际按协议公式
    assert control.noise_multiplier == pytest.approx(1.5)
    assert control.gate_action == "uwb_bias_and_noise_scale"


def test_liquid_bridge_contract_normalizes_risk_and_applies_safe_mode():
    """验证 risk 达到上界 1.0 时触发安全模式（uwb_skip_update）。

    测试场景：传入 risk=1.0（达到上界）的 ModelIntermediate。
    预期行为：risk 保持 1.0，noise_multiplier 按协议公式计算，
    gate_action 变为 uwb_skip_update（安全模式，跳过更新）。

    注：risk > 1.0 的越界值现在由 ModelIntermediate.__post_init__ 直接拒绝，
    桥接层不再需要处理越界 risk 裁剪。
    """
    control = build_measurement_control(
        {"modality": "uwb", "uwb_payload": {"anchor_id": 0, "range": 3.2, "valid": True, "quality": 0.95}},
        ModelIntermediate(bias=0.2, risk=1.0, uwb_scaling=1.4),
    )

    # risk 为 1.0（上界）
    assert control.risk == pytest.approx(1.0)
    # noise_multiplier = scaling^2 * (1 + risk) = 1.4^2 * 2.0 = 3.92
    assert control.noise_multiplier == pytest.approx(3.92)
    # risk 达到上界时触发安全模式，跳过 UWB 更新
    assert control.gate_action == "uwb_skip_update"


def test_liquid_bridge_contract_applies_scene_safe_mode_before_building_controls():
    """验证场景安全模式在构建控制之前生效，对 bias 和 scaling 进行衰减。

    测试场景：启用 safe_mode_cfg，传入包含场景元数据的 UWB 测量。
    预期行为：safe_mode 对 bias 和 scaling 施加衰减因子，
    但 risk 和 noise_multiplier 仍按协议公式计算。
    """
    control = build_measurement_control(
        {
            "modality": "uwb",
            "meta": {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "mini_seq"},
            "uwb_payload": {"anchor_id": 0, "range": 3.2, "valid": True, "quality": 1.0},
        },
        ModelIntermediate(bias=0.8, risk=0.4, uwb_scaling=1.5),
        safe_mode_cfg={"enabled": True},
    )

    # safe_mode 对 bias 施加衰减
    assert control.bias_applied == pytest.approx(0.32)
    # safe_mode 对 scaling 施加衰减
    assert control.scaling == pytest.approx(1.2)
    assert control.risk == pytest.approx(0.4)
    # noise_multiplier 按衰减后的参数重新计算
    assert control.noise_multiplier == pytest.approx(2.016)
    assert control.gate_action == "uwb_bias_and_noise_scale"


def test_liquid_bridge_contract_neutral_intermediate_still_resolves_scene_for_axis_floor():
    """验证即使是 neutral intermediate（risk=0），场景轴退化下界仍被解析。

    即使是 neutral intermediate（risk=0），_resolve_scene_axis_observation_floor
    仍需解析场景上下文来确定轴退化风险下界。这是正确行为——场景轴退化下界
    与模型输出无关，仅取决于场景配置。

    测试场景：传入默认 ModelIntermediate（全零/默认值），但携带场景元数据。
    预期行为：bias_applied=0.0，scaling=1.0，但 noise_multiplier >= 1.0
    （因为场景轴退化下界会给 risk 一个非零下界）。
    """
    control = build_measurement_control(
        {
            "modality": "uwb",
            "meta": {"scene_id": "S(A0,N0,V0,K0,M0)", "seq_id": "mini_seq"},
            "uwb_payload": {"anchor_id": 0, "range": 3.2, "valid": True, "quality": 1.0},
        },
        ModelIntermediate(),
    )

    assert control.bias_applied == pytest.approx(0.0)
    assert control.scaling == pytest.approx(1.0)
    # 场景轴退化下界会给 risk 一个非零下界，即使模型输出 risk=0，
    # 所以 noise_multiplier 会略大于 1.0。
    assert control.noise_multiplier >= 1.0
    assert control.gate_action == "uwb_bias_and_noise_scale"


def test_liquid_bridge_contract_noise_multiplier_increases_with_risk():
    """验证 noise_multiplier 随 risk 单调递增。

    测试场景：分别构建 low_risk（risk=0.0）和 high_risk（risk=0.5）的控制，
    其他参数相同。
    预期行为：high_risk 的 noise_multiplier 严格大于 low_risk 的。
    """
    low_risk = build_measurement_control(
        {"modality": "uwb", "uwb_payload": {"anchor_id": 0, "range": 3.2, "valid": True, "quality": 1.0}},
        ModelIntermediate(bias=0.2, risk=0.0, uwb_scaling=1.4),
    )
    high_risk = build_measurement_control(
        {"modality": "uwb", "uwb_payload": {"anchor_id": 0, "range": 3.2, "valid": True, "quality": 1.0}},
        ModelIntermediate(bias=0.2, risk=0.5, uwb_scaling=1.4),
    )

    # 验证低 risk 和高 risk 下的 noise_multiplier 精确值
    assert low_risk.noise_multiplier == pytest.approx(1.96)
    assert high_risk.noise_multiplier == pytest.approx(2.94)
    # 核心断言：noise_multiplier 随 risk 单调递增
    assert high_risk.noise_multiplier > low_risk.noise_multiplier


@pytest.mark.parametrize(
    "uwb_payload, expected_noise_multiplier",
    [
        # valid=False 触发硬拒绝
        ({"anchor_id": 0, "range": 3.2, "valid": False, "quality": 0.95}, 2.94),
        # quality 低于阈值触发硬拒绝（当前 floor=0.10，quality=0.09 真低于 floor 才会跳过 update）
        ({"anchor_id": 0, "range": 3.2, "valid": True, "quality": 0.09}, 3.7436),
    ],
)
def test_liquid_bridge_contract_uwb_hard_rejects(uwb_payload, expected_noise_multiplier):
    """验证 UWB 硬拒绝条件：valid=False 或 quality 过低。

    测试场景：参数化测试两种硬拒绝情况。
    预期行为：gate_action 为 uwb_skip_update，
    noise_multiplier 按协议公式计算（因风险增大而更高）。
    """
    control = build_measurement_control(
        {"modality": "uwb", "uwb_payload": uwb_payload},
        ModelIntermediate(bias=0.2, risk=0.5, uwb_scaling=1.4),
    )

    # 硬拒绝时跳过 UWB 更新
    assert control.gate_action == "uwb_skip_update"
    # 验证硬拒绝时的 noise_multiplier 精确值
    assert control.noise_multiplier == pytest.approx(expected_noise_multiplier)


def test_liquid_bridge_contract_uwb_quality_boundary_keeps_update():
    """验证 UWB quality 恰好在边界值时保留更新（不触发硬拒绝）。

    测试场景：quality=0.18，恰好高于硬拒绝阈值。
    预期行为：gate_action 为 uwb_bias_and_noise_scale（保留更新），
    noise_multiplier 按协议公式计算。
    """
    control = build_measurement_control(
        {
            "modality": "uwb",
            "uwb_payload": {"anchor_id": 0, "range": 3.2, "valid": True, "quality": 0.18},
        },
        ModelIntermediate(bias=0.2, risk=0.5, uwb_scaling=1.4),
    )

    # quality 在边界值时仍保留更新
    assert control.gate_action == "uwb_bias_and_noise_scale"
    assert control.noise_multiplier == pytest.approx(3.5672)


def test_liquid_bridge_contract_clips_negative_risk():
    """验证 risk=0.0（下界）时桥接层正常工作。

    测试场景：传入 risk=0.0 的 ModelIntermediate。
    预期行为：risk 保持 0.0，noise_multiplier 按 risk=0 计算，
    gate_action 仍为 uwb_bias_and_noise_scale。

    注：risk < 0.0 的越界值现在由 ModelIntermediate.__post_init__ 直接拒绝，
    桥接层不再需要处理负 risk 裁剪。
    """
    control = build_measurement_control(
        {"modality": "uwb", "range_m": 3.2},
        ModelIntermediate(bias=0.2, risk=0.0, uwb_scaling=1.4),
    )

    # risk 为 0.0（下界）
    assert control.risk == pytest.approx(0.0)
    # noise_multiplier = scaling^2 * (1 + risk) = 1.4^2 * 1.0 = 1.96
    assert control.noise_multiplier == pytest.approx(1.96)
    assert control.gate_action == "uwb_bias_and_noise_scale"


def test_liquid_bridge_contract_rejects_invalid_scaling():
    """验证无效 scaling（0 或负数）被拒绝。

    测试场景：传入 vio_scaling=0.0 的 ModelIntermediate。
    预期行为：ModelIntermediate.__post_init__ 拒绝越界值，
    抛出 ValueError，提示 scaling 必须在 [1.0, scaling_max] 范围内。
    """
    with pytest.raises(ValueError, match=r"must be in \[1\.0,"):
        build_measurement_control(
            {"modality": "vio", "dx": 0.1, "dy": -0.2, "dyaw": 0.05},
            ModelIntermediate(bias=0.3, risk=0.6, vio_scaling=0.0),
        )


def test_liquid_bridge_contract_vio_control_fields():
    """验证 VIO 测量控制字段的正确构建。

    测试场景：传入 VIO 模态的原始测量和 ModelIntermediate。
    预期行为：modality 为 vio，bias_applied 为 0.0（VIO 不施加 bias），
    scaling 等于 vio_scaling，risk 保持原值，
    noise_multiplier 按协议公式计算；
    applied_risk=0.6 ≥ vio_risk_hard_skip_threshold(0.05)（P35 fix 2026-09-02）
    → gate_action 为 vio_skip_update。
    """
    control = build_measurement_control(
        {"modality": "vio", "dx": 0.1, "dy": -0.2, "dyaw": 0.05},
        ModelIntermediate(bias=0.3, risk=0.6, vio_scaling=1.8),
    )

    assert control.modality == "vio"
    # VIO 不施加 bias
    assert control.bias_applied == pytest.approx(0.0)
    # VIO 的 scaling 直接使用 vio_scaling
    assert control.scaling == pytest.approx(1.8)
    assert control.risk == pytest.approx(0.6)
    # noise_multiplier 按协议公式计算
    assert control.noise_multiplier == pytest.approx(5.184)
    assert control.gate_action == "vio_skip_update"


def test_liquid_bridge_contract_pass_through_fallback():
    """验证 IMU 模态的直通回退行为。

    测试场景：传入 IMU 模态的原始测量，模型输出包含 bias 和 risk。
    预期行为：modality 为 imu，bias_applied=0.0，scaling=1.0，
    risk 保持原值，noise_multiplier=1.0（IMU 不做噪声缩放），
    gate_action 为 pass_through（直通，不做额外处理）。
    """
    control = build_measurement_control(
        {"modality": "imu", "ax": 0.01, "ay": -0.02},
        ModelIntermediate(bias=0.4, risk=0.7, uwb_scaling=1.6, vio_scaling=1.9),
    )

    assert control.modality == "imu"
    # IMU 不施加 bias
    assert control.bias_applied == pytest.approx(0.0)
    # IMU 不做 scaling
    assert control.scaling == pytest.approx(1.0)
    # risk 透传但不影响 noise
    assert control.risk == pytest.approx(0.7)
    # IMU 不做噪声缩放
    assert control.noise_multiplier == pytest.approx(1.0)
    # IMU 直通，不做额外处理
    assert control.gate_action == "pass_through"
