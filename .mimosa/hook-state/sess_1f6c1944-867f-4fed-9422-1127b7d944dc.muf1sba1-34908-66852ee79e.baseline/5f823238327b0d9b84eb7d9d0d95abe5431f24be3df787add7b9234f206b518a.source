"""Liquid 单元（liquid_cell）测试模块。

测试覆盖范围：
- Liquid 时间常数的动态行为
- 单元状态更新与前向传播
- 输入门/遗忘门的计算

被测模块：liquidloc.models.liquid_cell"""

import pytest
import torch

from liquidloc.models.features.feature_builder import build_feature_vector
from liquidloc.models.liquid.cell import LiquidCell


def test_normal_case():
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    cell = LiquidCell({"input_dim": 3, "hidden_dim": 4})
    new_state, output = cell.step([[1.0, 2.0, 3.0]], torch.zeros(1, 4), dt=0.1)

    assert new_state.shape == (1, 4)
    assert output.shape == (1, 4)
    assert not torch.allclose(new_state, torch.zeros_like(new_state))
    assert torch.isfinite(output).all()


def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    cell = LiquidCell({"input_dim": 3, "hidden_dim": 4})

    with pytest.raises(ValueError, match="last dimension must be 3"):
        cell.step([[1.0, 2.0]], torch.zeros(1, 4), dt=0.1)


def test_requires_explicit_dt():
    """显式测试：requires。\n\n验证 requires 的显式参数优先级，\n确保显式指定覆盖隐式推断。
    """
    cell = LiquidCell({"input_dim": 3, "hidden_dim": 4})

    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'dt'"):
        cell.step([[1.0, 2.0, 3.0]], torch.zeros(1, 4))


def test_zero_dt_keeps_state_unchanged():
    """保持测试：zero dt。\n\n验证 zero dt 的保持行为，\n确保特定属性在处理过程中不变。
    """
    cell = LiquidCell({"input_dim": 3, "hidden_dim": 4})
    state = torch.randn(1, 4)

    new_state, output = cell.step([[1.0, 2.0, 3.0]], state, dt=0.0)

    assert torch.allclose(new_state, state)
    assert torch.allclose(output, cell.output_norm(state))


def test_batch_dt_matches_independent_rollout():
    """匹配测试：batch dt。\n\n验证 batch dt 的输出与预期一致，\n确保合同合规。
    """
    cell = LiquidCell({"input_dim": 3, "hidden_dim": 4})
    input_batch = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [0.5, 1.5, 2.5],
        ],
        dtype=torch.float32,
    )
    state_batch = torch.randn(2, 4)
    dt_batch = torch.tensor([0.1, 0.25], dtype=torch.float32)
    missing_batch = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    batch_state, batch_output = cell.step(
        input_batch,
        state_batch,
        dt=dt_batch,
        missing_mask=missing_batch,
    )
    single_results = [
        cell.step(
            input_batch[index].unsqueeze(0),
            state_batch[index].unsqueeze(0),
            dt=float(dt_batch[index].item()),
            missing_mask=missing_batch[index].unsqueeze(0),
        )
        for index in range(2)
    ]
    expected_state = torch.cat([result[0] for result in single_results], dim=0)
    expected_output = torch.cat([result[1] for result in single_results], dim=0)

    assert torch.allclose(batch_state, expected_state, atol=1e-6, rtol=1e-6)
    assert torch.allclose(batch_output, expected_output, atol=1e-6, rtol=1e-6)


def test_reliability_projection_can_reduce_update_magnitude():
    """投影测试：reliability。\n\n验证 reliability 的坐标投影，\n确保 3D→2D 投影正确。
    """
    cell = LiquidCell({"input_dim": 3, "hidden_dim": 4})
    state = torch.randn(1, 4)
    input_tensor = torch.tensor([[1.0, 0.2, -0.4]], dtype=torch.float32)

    with torch.no_grad():
        cell.reliability_projection.weight.zero_()
        cell.reliability_projection.bias.fill_(-6.0)
    low_reliability_state, _ = cell.step(input_tensor, state, dt=0.2)

    with torch.no_grad():
        cell.reliability_projection.weight.zero_()
        cell.reliability_projection.bias.fill_(6.0)
    high_reliability_state, _ = cell.step(input_tensor, state, dt=0.2)

    low_delta = torch.linalg.vector_norm(low_reliability_state - state)
    high_delta = torch.linalg.vector_norm(high_reliability_state - state)

    assert low_delta < high_delta


def test_custom_reliability_gate_settings_are_applied():
    cell = LiquidCell(
        {
            "input_dim": 3,
            "hidden_dim": 4,
            "update_scale_floor": 0.45,
            "update_scale_span": 0.40,
            "reliability_bias_init": 1.25,
            "bad_observation_floor": 0.18,
            "bad_observation_span": 0.82,
            "bad_observation_interaction_coeff": 0.9,
        }
    )

    assert cell.update_scale_floor == pytest.approx(0.45)
    assert cell.update_scale_span == pytest.approx(0.40)
    assert cell.reliability_projection.bias.detach().cpu().item() == pytest.approx(1.25)
    assert cell.bad_observation_floor == pytest.approx(0.18)
    assert cell.bad_observation_span == pytest.approx(0.82)
    assert cell.bad_observation_interaction_coeff == pytest.approx(0.9)


def test_rejects_invalid_update_scale_budget():
    with pytest.raises(ValueError, match="update_scale_floor \\+ update_scale_span must be <= 1"):
        LiquidCell(
            {
                "input_dim": 3,
                "hidden_dim": 4,
                "update_scale_floor": 0.70,
                "update_scale_span": 0.40,
            }
        )


def test_rejects_invalid_bad_observation_scale_budget():
    """观测测试：rejects invalid bad。\n\n验证 rejects invalid bad 的观测风险计算，\n确保观测质量正确影响缩放。
    """
    with pytest.raises(ValueError, match="bad_observation_floor \\+ bad_observation_span must be <= 1"):
        LiquidCell(
            {
                "input_dim": 3,
                "hidden_dim": 4,
                "bad_observation_floor": 0.40,
                "bad_observation_span": 0.70,
            }
        )


def test_bad_observation_gate_reduces_update_for_degraded_steps():
    """观测测试：bad。\n\n验证 bad 的观测风险计算，\n确保观测质量正确影响缩放。
    """
    cell = LiquidCell(
        {
            "input_dim": 8,
            "hidden_dim": 4,
            "feature_order": [
                "valid",
                "quality",
                "modality_gap_dt",
                "uwb_range_residual",
                "geom_score",
                "tracked_features",
                "reproj_err",
                "tracked_features_drop",
            ],
        }
    )
    state = torch.randn(1, 4)
    clean_input = torch.tensor([[1.0, 1.0, 0.01, 0.0, 1.0, 120.0, 0.1, 0.0]], dtype=torch.float32)
    degraded_input = torch.tensor([[0.0, 0.05, 0.30, 1.2, 0.1, 6.0, 4.0, 30.0]], dtype=torch.float32)

    with torch.no_grad():
        # v6 Patch 1 已把 time_a/time_b lowercase 改为大写 time_A/time_B（闭式时间插值），
        # 测试同步引用大写名称以匹配 cell.py 当前属性。
        for linear in (
            cell.backbone_projection,
            cell.ff1_projection,
            cell.ff2_projection,
            cell.time_A_projection,
            cell.time_B_projection,
        ):
            linear.weight.zero_()
            linear.bias.fill_(0.25)
        cell.reliability_projection.weight.zero_()
        cell.reliability_projection.bias.fill_(6.0)

    clean_state, _ = cell.step(clean_input, state, dt=0.2)
    degraded_state, _ = cell.step(degraded_input, state, dt=0.2)

    clean_delta = torch.linalg.vector_norm(clean_state - state)
    degraded_delta = torch.linalg.vector_norm(degraded_state - state)

    assert degraded_delta < clean_delta


def test_build_feature_vector_marks_two_anchor_geom_score_missing():
    """缺失测试：build feature vector marks two anchor geom score。\n\n验证 build feature vector marks two anchor geom score 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    event = {
        "t": 0.1,
        "dt": 0.1,
        "modality": "uwb",
        "meta": {"scene_id": "scene", "seq_id": "seq"},
        "uwb_payload": {"anchor_id": 0, "range": 1.0, "valid": True, "quality": 0.9},
    }
    state_ctx = {
        "px": 0.0,
        "py": 0.0,
        "anchor_lookup": {
            0: (0.0, 0.0),
            1: (10.0, 0.0),
        },
    }

    feature = build_feature_vector(event, state_ctx, ["geom_score"])

    assert feature["feature_values"] == [0.0]
    assert feature["missing_mask"] == [1]


def test_time_rate_scale_aligns_t_interp_with_millisecond_dt():
    """τ 尺度修复：毫秒级 dt 下 t_interp 必须可学习，不能锁死在 ~0.006。

    旧实现 B=sigmoid(...)≤1 (1/s) → dt=10ms 时 t_interp≈A*B*dt≈0.006。
    新实现 B_eff=time_rate_scale*cfB，默认 scale=180 → 同 dt 下 t_interp 应显著更大。
    本测试不依赖绝对阈值（dt 量级可能因数据集变化），用相对比值 ≥3× 校验尺度生效。
    """
    cell = LiquidCell(
        {
            "input_dim": 3,
            "hidden_dim": 4,
            "time_rate_scale": 180.0,
            "enable_forget_root": False,  # 隔离 forget_root，只测 t_interp 路径
            "update_scale_floor": 1.0,
            "update_scale_span": 0.0,
            "bad_observation_floor": 0.0,
            "bad_observation_span": 0.0,
        }
    )
    state = torch.zeros(1, 4)
    with torch.no_grad():
        # 固定候选与时间投影，便于解析对照
        for linear in (
            cell.backbone_projection,
            cell.ff1_projection,
            cell.ff2_projection,
        ):
            linear.weight.zero_()
            linear.bias.fill_(0.5)  # tanh(0.5) 给出非零候选
        cell.time_A_projection.weight.zero_()
        cell.time_A_projection.bias.fill_(2.5)  # 锁到本轮重校准目标值（cell.py 默认亦为 2.5）
        cell.time_B_projection.weight.zero_()
        cell.time_B_projection.bias.fill_(1.0)  # cfB≈0.731
        cell.reliability_projection.weight.zero_()
        cell.reliability_projection.bias.fill_(6.0)  # reliability≈1
        cell.forget_projection.weight.zero_()
        cell.forget_projection.bias.fill_(6.0)  # forget≈1

    dt = 0.001  # 真实 sim 数据 p50 ≈ 1ms
    new_state, _ = cell.step([[0.1, 0.2, 0.3]], state, dt=dt)
    delta = torch.linalg.vector_norm(new_state - state).item()

    # scale=180 应让 dt=1ms 时 t_interp≈0.116, state-mix 权重 ≈0.116
    # 用相对阈值（>= 0.05），避免依赖具体 dt 量级的绝对数
    assert delta > 0.05, f"hidden barely moved under ms-scale dt: delta={delta} (scale=180)"

    # 对照：把 time_rate_scale 压到 1.0（旧 1/s 量级）应显著更小
    cell_old = LiquidCell(
        {
            "input_dim": 3,
            "hidden_dim": 4,
            "time_rate_scale": 1.0,
            "enable_forget_root": False,
            "update_scale_floor": 1.0,
            "update_scale_span": 0.0,
            "bad_observation_floor": 0.0,
            "bad_observation_span": 0.0,
        }
    )
    with torch.no_grad():
        for src, dst in (
            (cell.backbone_projection, cell_old.backbone_projection),
            (cell.ff1_projection, cell_old.ff1_projection),
            (cell.ff2_projection, cell_old.ff2_projection),
            (cell.time_A_projection, cell_old.time_A_projection),
            (cell.time_B_projection, cell_old.time_B_projection),
            (cell.reliability_projection, cell_old.reliability_projection),
            (cell.forget_projection, cell_old.forget_projection),
        ):
            dst.weight.copy_(src.weight)
            if src.bias is not None and dst.bias is not None:
                dst.bias.copy_(src.bias)
    old_state, _ = cell_old.step([[0.1, 0.2, 0.3]], state, dt=dt)
    old_delta = torch.linalg.vector_norm(old_state - state).item()
    assert delta > 3.0 * old_delta, (
        f"time_rate_scale=180 should dominate scale=1: new={delta}, old={old_delta}"
    )


def test_tau_init_matches_real_sim_dt_scale():
    """τ 初始化绑定真实 sim dt 尺度：未训练 cell 在 sim p50/p90/max dt 下
    t_interp 必须落在与 ODE 时间常数匹配的绝对区间，不能只是"远大于旧值"。

    data/raw/sim 实测合并事件 dt 直方图: p50=1.02ms, p90=5.45ms, max=174ms。
    init A=sigmoid(2.5)=0.924, cfB=sigmoid(1.0)=0.731, scale=180
      → τ_eff = 131.6/s, 时间常数 7.6ms
      → t_interp(p50)=0.116, t_interp(p90)=0.473, t_interp(max)→A=0.924

    用绝对区间断言：cell 在毫秒级真实 dt 上"开箱即用"达到 LSTM 步级下限。
    """
    # 用 Linear 投影权重清零 + 已知 bias，把 t_interp 解析地隔离开
    cell = LiquidCell(
        {
            "input_dim": 3,
            "hidden_dim": 4,
            "time_rate_scale": 180.0,  # 默认
            "enable_forget_root": False,
            "update_scale_floor": 1.0,  # 让 effective_update = t_interp
            "update_scale_span": 0.0,
            "bad_observation_floor": 0.0,
            "bad_observation_span": 0.0,
        }
    )
    with torch.no_grad():
        # candidate 维度无关：让 candidate 与零状态充分不同，state mix 才能反映 t_interp
        for linear in (cell.backbone_projection, cell.ff1_projection, cell.ff2_projection):
            linear.weight.zero_()
            linear.bias.fill_(0.5)
        # 显式把 τ/forget bias 锁到本轮重校准的目标值（cell.py 默认也是这些值，见
        # test_cell_default_init_biases_are_locked_to_recalibrated_values 的直接断言）。
        # 这里再次显式覆盖是为了让本测试在 weight=0 + 已知 bias 的纯解析条件下运行，
        # 与 reset_parameters 中 _fill_linear 给 weight 添加的随机扰动解耦，便于比值校验。
        cell.time_A_projection.weight.zero_()
        cell.time_A_projection.bias.fill_(2.5)  # A=sigmoid(2.5)≈0.924
        cell.time_B_projection.weight.zero_()
        cell.time_B_projection.bias.fill_(1.0)  # cfB=sigmoid(1.0)≈0.731
        cell.reliability_projection.weight.zero_()
        cell.reliability_projection.bias.fill_(6.0)
        cell.forget_projection.weight.zero_()
        cell.forget_projection.bias.fill_(6.0)

    state = torch.zeros(1, 4)
    feat = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)

    def _delta_at(dt: float) -> float:
        s, _ = cell.step(feat, state, dt=dt)
        return torch.linalg.vector_norm(s - state).item()

    # 因为 update_scale_floor=1, span=0, risk=0, forget=1, f_root disabled,
    # effective_update = t_interp, 且 candidate-state 差的范数 = ||candidate - 0||.
    # candidate=tanh(ff2(tanh(ff1(tanh(backbone))))) 仍是固定非零向量，所以
    # delta ∝ t_interp * ||candidate||。我们用 *之间的比值* 校验 t_interp 的形状：
    #   t_interp(dt) / t_interp(p50_dt) ≈ (1-exp(-τ_eff·dt)) / (1-exp(-τ_eff·p50))
    # 不依赖 candidate 的具体范数。
    d_p50 = _delta_at(0.00102)   # sim p50 dt
    d_p90 = _delta_at(0.00545)   # sim p90 dt
    d_p95 = _delta_at(0.00788)   # sim p95 dt
    d_max = _delta_at(0.174)     # sim max dt → 应触达渐近 A

    # 绝对：p50 dt 下 state 步级位移 >= 0.05（LSTM 步级下限 ~0.10 乘 candidate 范数 ~0.7）
    assert d_p50 > 0.05, f"τ init too cold at p50 dt: ||Δstate||={d_p50}"

    # 相对形状：t_interp 应随 dt 单调上升，且 p90/p50 比值 ≈ 4.07（解析 0.473/0.116）
    ratio_p90_p50 = d_p90 / max(d_p50, 1e-8)
    assert 3.5 < ratio_p90_p50 < 4.6, (
        f"t_interp shape broken: p90/p50 ratio={ratio_p90_p50} (expected ≈4.07)"
    )

    # p95 比 p90 更大但增量更小（指数饱和曲线）
    ratio_p95_p90 = d_p95 / max(d_p90, 1e-8)
    assert 1.1 < ratio_p95_p90 < 1.4, (
        f"t_interp saturating too fast/slow: p95/p90 ratio={ratio_p95_p90} (expected ≈1.26)"
    )

    # max dt 处应触达渐近 A=0.924：d_max/d_p95 应约 1.55（0.924/0.597）
    ratio_max_p95 = d_max / max(d_p95, 1e-8)
    assert ratio_max_p95 > 1.4, (
        f"t_interp not reaching asymptote A at max sim dt: max/p95 ratio={ratio_max_p95}"
    )


def test_missing_observation_still_allows_dt_driven_state_evolution():
    """异步自由演化：全缺失掩码时状态仍应随 dt 演化（不能当“重复填充最近观测”）。

    全 missing 时 observed_input=0，但 t_interp 仍依赖 dt；若实现错误地在
    missing 时跳过积分，则不同 dt 下状态变化会相同或为零。
    """
    cell = LiquidCell(
        {
            "input_dim": 3,
            "hidden_dim": 4,
            "time_rate_scale": 180.0,
            "enable_forget_root": False,
        }
    )
    state = torch.randn(1, 4)
    missing = torch.ones(1, 3)  # 全部缺失
    input_tensor = torch.tensor([[9.0, 9.0, 9.0]], dtype=torch.float32)

    state_small_dt, _ = cell.step(input_tensor, state, dt=0.001, missing_mask=missing)
    state_large_dt, _ = cell.step(input_tensor, state, dt=0.05, missing_mask=missing)

    delta_small = torch.linalg.vector_norm(state_small_dt - state).item()
    delta_large = torch.linalg.vector_norm(state_large_dt - state).item()

    # 更大 dt 应产生更大（或至少不更小）的状态位移；全缺失时仍允许 ODE 自由演化
    assert delta_large >= delta_small - 1e-6
    # 且至少有一侧非零，证明 missing 并未把递推完全短路
    assert delta_large > 0.0 or delta_small > 0.0


def test_real_sim_dt_window_does_not_freeze_hidden_state():
    """整窗口 20 步（真实 sim per-step dt 直方图）驱动 LiquidCell：
    hidden 状态必须显著离开零点，不允许在毫秒级 dt 下冻结。

    data/raw/sim 实测合并事件 dt: p50=1.02ms, p90=5.45ms, p95=7.88ms。
    用 20 步窗口（trainer 默认 window_size=20），第一步用 fallback dt=0。
    在默认 time_rate_scale=180 下，hidden delta 应远大于 1e-3（旧 scale=1 时会 < 1e-4）。
    """
    real_dt_window = [
        0.005,  # window[0] fallback dt（trainer 首步用 fallback_dt，避免 dt=0 输入被丢）
        0.001, 0.001, 0.001, 0.001,  # 5× p50 ≈ 1ms
        0.002, 0.003, 0.001, 0.005, 0.001,  # 混合 1-5ms
        0.005, 0.001, 0.001, 0.008, 0.001,  # 接近 p95
        0.001, 0.001, 0.005, 0.001, 0.001,  # 收尾
    ]
    assert len(real_dt_window) == 20

    cell = LiquidCell({"input_dim": 3, "hidden_dim": 8})
    state = torch.zeros(1, 8)
    feat = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)

    for dt in real_dt_window:
        state, _ = cell.step(feat, state, dt=dt)

    final_delta = torch.linalg.vector_norm(state).item()
    # 默认 scale=180 下，20 步 1-8ms dt 累积 hidden 应远大于 1e-3
    # （实测 ~0.3-0.5；旧 scale=1 时仅 ~5e-4）
    assert final_delta > 1e-3, (
        f"hidden state froze under real sim dt window: ||state||={final_delta}"
    )

    # 对照：scale=1（旧 scale）应显著更小，证明 scale 起作用
    cell_old = LiquidCell({"input_dim": 3, "hidden_dim": 8, "time_rate_scale": 1.0})
    state_old = torch.zeros(1, 8)
    for dt in real_dt_window:
        state_old, _ = cell_old.step(feat, state_old, dt=dt)
    old_delta = torch.linalg.vector_norm(state_old).item()
    assert final_delta > 10.0 * old_delta, (
        f"scale=180 should dominate scale=1 over a real window: "
        f"new={final_delta}, old={old_delta}"
    )


def test_network_forwards_time_rate_scale_into_cell():
    """network.cell_cfg 必须透传 time_rate_scale，避免 yaml 配置静默失效。"""
    from liquidloc.models.liquid.network import LiquidNetwork

    network = LiquidNetwork(
        {
            "input_dim": 3,
            "hidden_dim": 4,
            "time_rate_scale": 180.0,
            "cell_update_scale_floor": 0.15,
            "cell_update_scale_span": 0.70,
            "reliability_bias_init": 0.5,
        }
    )
    assert network.cell is not None
    assert network.cell.time_rate_scale == pytest.approx(180.0)
    assert network.cell.update_scale_floor == pytest.approx(0.15)
    assert network.cell.update_scale_span == pytest.approx(0.70)
    assert network.cell.reliability_projection.bias.detach().cpu().item() == pytest.approx(0.5)


def test_gate_init_matches_lstm_step_magnitude_under_real_sim_dt():
    """门控初始化匹配 LSTM 步级量级：未训练 cell 在 sim p50 dt 下，
    20 步窗口后 hidden 必须显著离开 0（达到 candidate 范数的合理比例），
    不允许因 forget × forget_root 叠乘把 eff_per_step 压到 1e-3 量级。

    修复前 (scale=1, A_bias=1.5, forget.bias=-0.5, forget_root.bias=+0.5):
      - p50 dt(1.02ms) t_interp≈6e-4, gate product=0.378×0.378×0.92≈0.131
      - eff_per_step≈7.5e-5, 20 步后 ||state||/||candidate|| ≈ 0
    修复后 (scale=180, A_bias=2.5, forget.bias=+1.0, forget_root.bias=0.0):
      - p50 dt t_interp≈0.116, gate product=0.731×0.5×0.92≈0.336
      - eff_per_step≈0.039, N50≈17 步
      - 实测 20 步后 ||state||/||candidate||≈0.375（理论 N50=17 给 ~56%，
        因 candidate 方向随 state 漂移而低于理论上限，仍合理）
    """
    cell = LiquidCell({"input_dim": 3, "hidden_dim": 8})  # 全默认
    state = torch.zeros(1, 8)
    feat = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    sim_p50_dt = 0.00102  # 真实 sim p50

    for _ in range(20):
        state, _ = cell.step(feat, state, dt=sim_p50_dt)

    # 计算 candidate 自身范数作归一化基准
    with torch.no_grad():
        observed_input = feat
        cell_input = torch.cat([observed_input, torch.zeros_like(feat)], dim=-1)
        state_and_input = torch.cat([cell_input, torch.zeros(1, 8)], dim=-1)
        backbone = torch.tanh(cell.backbone_projection(state_and_input))
        ff_hidden = torch.tanh(cell.ff1_projection(backbone))
        candidate = torch.tanh(cell.ff2_projection(ff_hidden))
    candidate_norm = float(torch.linalg.vector_norm(candidate))
    state_norm = float(torch.linalg.vector_norm(state))
    ratio = state_norm / max(candidate_norm, 1e-8)

    # 20 步 p50 dt 后 hidden 应达 candidate 的 25% 以上
    # (旧 init 此 ratio ≈ 0，新 init ≈ 0.375，下界 0.25 留余量)
    assert ratio > 0.25, (
        f"gate init too conservative: ||state(20)||/||candidate|| = {ratio:.3f} "
        f"(state_norm={state_norm:.4f}, candidate_norm={candidate_norm:.4f})"
    )


def test_gate_init_less_conservative_than_old_init_at_p50_dt():
    """对照测试：完整老 init (scale=1, A_bias=1.5, forget.bias=-0.5, forget_root.bias=+0.5)
    vs 完整新 init (scale=180, A_bias=2.5, forget.bias=+1.0, forget_root.bias=0.0)
    在 sim p50 dt 下的 hidden 演化对比。

    上一版这个测试只回退 bias 而漏回退 time_rate_scale，导致 cell_old 仍用新 scale=180，
    比值只有 2.5×，与"真实老 init"的几乎冻结（ratio→0）行为不符，且阈值被 sneaking down 到 2×。
    本版把 cell_old 完整回退到老 init，比值应 ≥50×（实测约 400×）。
    """
    cell_new = LiquidCell({"input_dim": 3, "hidden_dim": 8})  # 全新默认

    cell_old = LiquidCell({"input_dim": 3, "hidden_dim": 8, "time_rate_scale": 1.0})  # 老 scale
    with torch.no_grad():
        cell_old.time_A_projection.bias.fill_(1.5)      # 老 A bias
        cell_old.forget_projection.bias.fill_(-0.5)      # 老 forget bias
        cell_old.forget_root_projection.bias.fill_(0.5)  # 老 forget_root bias

    feat = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    sim_p50_dt = 0.00102

    s_new = torch.zeros(1, 8)
    s_old = torch.zeros(1, 8)
    for _ in range(20):
        s_new, _ = cell_new.step(feat, s_new, dt=sim_p50_dt)
        s_old, _ = cell_old.step(feat, s_old, dt=sim_p50_dt)

    new_norm = float(torch.linalg.vector_norm(s_new))
    old_norm = float(torch.linalg.vector_norm(s_old))
    # 完整老 init 在 p50 dt 下几乎不动（理论 ratio≈0.0016，实测 ||state||~1e-4），
    # 新 init ||state||≈0.037，比值实测约 250-400×。设阈值 50× 给 5× 余量。
    assert new_norm > 50.0 * old_norm, (
        f"complete new init should dominate complete old init: "
        f"new={new_norm}, old={old_norm}, ratio={new_norm/max(old_norm,1e-9):.2f}x"
    )


def test_cell_default_init_biases_are_locked_to_recalibrated_values():
    """直接断言 cell.py 默认 bias / scale 已落到本轮重校准的值。
    防止手抖或回退让 default 漂回旧 init，但其他测试因显式覆盖 bias 而 silent pass。
    """
    cell = LiquidCell({"input_dim": 3, "hidden_dim": 8})  # 全默认
    biases = {
        "time_A_projection": (float(cell.time_A_projection.bias.detach().cpu()[0]), 2.5),
        "time_B_projection": (float(cell.time_B_projection.bias.detach().cpu()[0]), 1.0),
        "forget_projection": (float(cell.forget_projection.bias.detach().cpu()[0]), 1.0),
        "forget_root_projection": (float(cell.forget_root_projection.bias.detach().cpu()[0]), 0.0),
    }
    for name, (actual, expected) in biases.items():
        assert actual == pytest.approx(expected, abs=1e-6), (
            f"{name}.bias default drifted: got {actual}, expected {expected}"
        )
    assert cell.time_rate_scale == pytest.approx(180.0, abs=1e-6), (
        f"time_rate_scale default drifted: got {cell.time_rate_scale}, expected 180.0"
    )


def test_bad_observation_risk_skips_missing_geom_score_but_penalizes_observed_low_geom_score():
    """缺失测试：bad observation risk skips。\n\n验证 bad observation risk skips 在数据缺失时的处理，\n确保缺失数据不影响整体流程。
    """
    cell = LiquidCell(
        {
            "input_dim": 1,
            "hidden_dim": 4,
            "feature_order": ["geom_score"],
        }
    )
    observed_low_geom = torch.tensor([[0.0]], dtype=torch.float32)
    missing_geom = torch.tensor([[0.0]], dtype=torch.float32)
    observed_mask = torch.tensor([[0.0]], dtype=torch.float32)
    missing_mask = torch.tensor([[1.0]], dtype=torch.float32)

    observed_risk = cell.estimate_bad_observation_risk(observed_low_geom, observed_mask)
    missing_risk = cell.estimate_bad_observation_risk(missing_geom, missing_mask)

    assert observed_risk.item() == pytest.approx(1.0)
    assert missing_risk.item() == pytest.approx(cell.bad_observation_floor)


def test_step_keeps_missing_two_anchor_geom_from_triggering_geom_penalty():
    """保持测试：step。\n\n验证 step 的保持行为，\n确保特定属性在处理过程中不变。
    """
    cell = LiquidCell(
        {
            "input_dim": 1,
            "hidden_dim": 4,
            "feature_order": ["geom_score"],
        }
    )
    state = torch.randn(1, 4)
    input_tensor = torch.tensor([[0.0]], dtype=torch.float32)
    missing_mask = torch.tensor([[1.0]], dtype=torch.float32)

    with torch.no_grad():
        # v6 Patch 1 已把 time_a/time_b lowercase 改为大写 time_A/time_B（闭式时间插值），
        # 测试同步引用大写名称以匹配 cell.py 当前属性。
        for linear in (
            cell.backbone_projection,
            cell.ff1_projection,
            cell.ff2_projection,
            cell.time_A_projection,
            cell.time_B_projection,
        ):
            linear.weight.zero_()
            linear.bias.fill_(0.25)
        cell.reliability_projection.weight.zero_()
        cell.reliability_projection.bias.fill_(6.0)

    no_penalty_state, _ = cell.step(input_tensor, state, dt=0.2, missing_mask=missing_mask)
    penalty_state, _ = cell.step(input_tensor, state, dt=0.2, missing_mask=torch.zeros_like(missing_mask))

    no_penalty_delta = torch.linalg.vector_norm(no_penalty_state - state)
    penalty_delta = torch.linalg.vector_norm(penalty_state - state)

    assert penalty_delta < no_penalty_delta
