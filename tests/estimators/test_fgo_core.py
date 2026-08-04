from __future__ import annotations

"""因子图优化（FGO）核心测试模块。

测试覆盖范围：
- FGO 滑动窗口优化
- IMU/UWB/VIO 因子的构建与残差计算
- Gauss-Newton 优化器
- 鲁棒权重与门控

被测模块：liquidloc.estimators.fgo_core"""

import math

import numpy as np
import pytest

from liquidloc.common.types import MeasurementControl, ModelIntermediate
from liquidloc.estimators.fgo_core import FGOCore
from liquidloc.estimators.vision_update_step import build_vio_measurement, compute_vio_residual
from liquidloc.protocol.liquid_bridge_contract import build_measurement_control


def _cfg():
    return {
        "measurement_noise": {"uwb": 0.25, "vio": {"pos": 0.08, "yaw": 0.03}},
        "anchor_layout": {"anchor_ids": [0, 1], "anchor_positions": [(2.0, 0.0), (2.0, 2.0)]},
        # §1.1 主表 8 维 + §2.3 紧耦合扩维（uwb_clock_bias / vio_scale），10 维
        "init_cov": [1.0] * 10,
        "window_size": 3,
        "optimizer": {"name": "gauss_newton", "max_iters": 5},
        "factor_weights": {"imu": 1.0, "uwb": 1.0, "vio": 1.0},
        # §3.4 + §10.3 / B10–B11 窗长比断言：本模块用 window_size ∈ {2,3,4} 步
        # / dt=0.1s = 10Hz 的极短窗进行 FGO 数学正确性单测，不是主表声称；按"测试
        # 协议"档把 tau_filt_s 显式写死成与窗口同量级（0.2s），使 2/3/4 步都落在
        # [0.5, 3]×τ_filt = [0.1, 0.6]s = [1, 6] 步 @10Hz 内。这与 §10.3 "禁测试后
        # 调 τ_filt" 兼容——每次开跑前都已写死，不存在博弈。主表档请见
        # configs/models/fgo.yaml（window_size=500 / τ=5s）。
        "nominal_event_rate_hz": 10.0,  # 测试 dt=0.1s 即 10Hz，与 _imu_event 的 dt 同口径。
        "tau_filt_s": 0.2,  # 测试协议 τ_filt，与测试窗口同量级。
        # §10.3 诚实边界守卫（B2 v6）：tau_filt_assumed_t_eff_min_s 显性声明 τ_filt=0.2
        # 的协议层 T_eff 域假设，与 fgo.yaml 同口径（min(5, 0.2·25)=5 ≥ 0.2 一致）。
        "tau_filt_assumed_t_eff_min_s": 25.0,
        "window_length_ratio_min": 0.5,
        "window_length_ratio_max": 3.0,
    }


def _imu_event(*, t=0.1, dt=0.1, ax=0.2, ay=0.0, gz=0.01):
    return {
        "t": t,
        "dt": dt,
        "modality": "imu",
        "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
        "imu_payload": {"ax": ax, "ay": ay, "gz": gz},
        "uwb_payload": None,
        "vio_payload": None,
    }


def _vio_event(*, t=0.2, dt=0.1, dx=0.4, dy=0.0, dyaw=0.05):
    return {
        "t": t,
        "dt": dt,
        "modality": "vio",
        "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
        "imu_payload": None,
        "uwb_payload": None,
        "vio_payload": {
            "dx": dx,
            "dy": dy,
            "dyaw": dyaw,
            "quality": 0.9,
            "tracked_features": 120,
            "reproj_err": 0.3,
        },
    }


def _uwb_event(*, t=0.2, dt=0.1, rng=0.9, quality=0.95):
    return {
        "t": t,
        "dt": dt,
        "modality": "uwb",
        "meta": {"scene_id": "S(A0,N0,V0,G0,K6)", "seq_id": "mini_seq"},
        "imu_payload": None,
        "uwb_payload": {"anchor_id": 0, "range": rng, "valid": True, "quality": quality},
        "vio_payload": None,
    }


def test_normal_case():
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    estimator = FGOCore(_cfg())
    estimator.step(_imu_event())
    estimator.step(_vio_event())  # 首帧 VIO 初始化参考位姿（被跳过）
    estimator.step(_imu_event(t=0.2, ax=0.0, ay=0.1))
    state = estimator.step(_vio_event(t=0.3, dx=0.1, dy=0.0, dyaw=0.02))
    report = estimator.last_update_report
    assert state.timestamp == 0.3
    assert report["modality"] == "vio"
    assert report["update_applied"] is True
    assert report["solver_report"]["constraint_count"] >= 1
    assert report["solver_report"]["window_length"] == 3
    assert len(estimator.window_states) == 3


def test_uwb_constraint_report_isolated_from_internal_state():
    estimator = FGOCore(_cfg())
    estimator.step(_imu_event())
    estimator.step(_uwb_event())
    report = estimator.last_update_report

    report["constraint"]["noise"] = 123.0
    report["constraint"]["anchor_pos"] = (456.0, 0.0)

    assert estimator.constraints[-1]["noise"] != 123.0
    assert estimator.constraints[-1]["anchor_pos"][0] != 456.0


def test_bridge_risk_inflates_fgo_measurement_covariance_via_noise_multiplier():
    """膨胀测试：bridge risk。\n\n验证 bridge risk 的膨胀效应，\n确保恶化观测条件导致协方差增大。
    """
    low_risk_control = build_measurement_control(
        _uwb_event(rng=0.9, quality=1.0),
        intermediate=ModelIntermediate(bias=0.0, risk=0.0, uwb_scaling=1.4, vio_scaling=1.0),
    )
    high_risk_control = build_measurement_control(
        _uwb_event(rng=0.9, quality=1.0),
        intermediate=ModelIntermediate(bias=0.0, risk=0.5, uwb_scaling=1.4, vio_scaling=1.0),
    )

    low_risk_estimator = FGOCore(_cfg())
    low_risk_estimator.step(_imu_event())
    low_risk_estimator.set_measurement_control(low_risk_control)
    low_risk_estimator.step(_uwb_event(rng=0.9, quality=1.0))

    high_risk_estimator = FGOCore(_cfg())
    high_risk_estimator.step(_imu_event())
    high_risk_estimator.set_measurement_control(high_risk_control)
    high_risk_estimator.step(_uwb_event(rng=0.9, quality=1.0))

    low_report = low_risk_estimator.last_update_report["covariance_report"]
    high_report = high_risk_estimator.last_update_report["covariance_report"]
    assert low_report["noise_multiplier"] == pytest.approx(1.4**2 * (1 + 1 / 60))
    assert high_report["noise_multiplier"] == pytest.approx(1.4**2 * 1.5)
    assert low_report["effective_cov"] == pytest.approx(0.0625 * 1.4**2 * (1 + 1 / 60))
    assert high_report["effective_cov"] == pytest.approx(0.0625 * 1.4**2 * 1.5)
    assert high_report["effective_cov"] > low_report["effective_cov"]


def test_uwb_skip_update_is_reported_and_does_not_add_constraint():
    """不侵入测试：uwb skip update is reported and。\n\n验证 uwb skip update is reported and 不会产生副作用，\n确保功能隔离性。
    """
    estimator = FGOCore(_cfg())
    estimator.step(_imu_event())
    before_constraints = len(estimator.constraints)
    estimator.set_measurement_control(
        MeasurementControl(
            modality="uwb",
            bias_applied=0.2,
            scaling=1.0,
            risk=0.0,
            noise_multiplier=1.0,
            gate_action="uwb_skip_update",
        )
    )

    state = estimator.step(_uwb_event())
    report = estimator.last_update_report
    assert state.timestamp == pytest.approx(0.2)
    assert report["update_applied"] is False
    assert report["reason"] == "uwb_skip_update"
    assert len(estimator.constraints) == before_constraints


def test_vio_skip_update_is_reported_and_does_not_add_constraint():
    """不侵入测试：vio skip update is reported and。\n\n验证 vio skip update is reported and 不会产生副作用，\n确保功能隔离性。
    """
    estimator = FGOCore(_cfg())
    estimator.step(_imu_event())
    before_constraints = len(estimator.constraints)
    estimator.set_measurement_control(
        MeasurementControl(
            modality="vio",
            bias_applied=0.0,
            scaling=1.8,
            risk=0.9,
            noise_multiplier=2.34,
            gate_action="vio_skip_update",
        )
    )

    state = estimator.step(_vio_event())
    report = estimator.last_update_report
    assert state.timestamp == pytest.approx(0.2)
    assert report["update_applied"] is False
    assert report["reason"] == "vio_skip_update"
    assert len(estimator.constraints) == before_constraints


def test_boundary_case():
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    estimator = FGOCore(_cfg() | {"window_size": 2})
    estimator.step(_imu_event(t=0.1))
    estimator.step(_imu_event(t=0.2, ax=0.0, ay=0.1))
    estimator.step(_imu_event(t=0.3, ax=0.0, ay=0.0))
    assert len(estimator.window_states) == 2
    assert estimator.last_update_report["window_length"] == 2


def test_skipped_uwb_between_imus_does_not_change_fgo_imu_motion():
    """不侵入测试：skipped uwb between imus。\n\n验证 skipped uwb between imus 不会产生副作用，\n确保功能隔离性。
    """
    plain = FGOCore(_cfg())
    plain.step(_imu_event(t=0.1, dt=0.1, ax=1.0, gz=0.0))
    plain.step(_imu_event(t=0.2, dt=0.1, ax=0.0, gz=0.0))

    interleaved = FGOCore(_cfg())
    interleaved.step(_imu_event(t=0.1, dt=0.1, ax=1.0, gz=0.0))
    interleaved.set_measurement_control(MeasurementControl(modality="uwb", gate_action="uwb_skip_update"))
    interleaved.step(_uwb_event(t=0.15, dt=0.05, rng=0.9))
    interleaved.step(_imu_event(t=0.2, dt=0.05, ax=0.0, gz=0.0))

    assert interleaved.window_states[-1]["px"] == pytest.approx(plain.window_states[-1]["px"])
    assert interleaved.window_states[-1]["vx"] == pytest.approx(plain.window_states[-1]["vx"])
    assert interleaved._window_entries[-1]["motion_from_prev"] == pytest.approx(
        plain._window_entries[-1]["motion_from_prev"]
    )


def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    with pytest.raises(ValueError, match="window_size must be positive"):
        FGOCore(_cfg() | {"window_size": 0})


@pytest.mark.parametrize(
    "override, match",
    [
        ({"window_size": 2.5}, "window_size must be a positive integer"),
        ({"optimizer": {"max_iters": 1.5}}, "optimizer.max_iters must be a positive integer"),
        ({"factor_weights": {"imu": 0.0}}, "factor_weights.imu must be > 0.0, got 0.0"),
        ({"measurement_noise": {"vio": {"pos": 0.0, "yaw": 0.03}}}, "measurement_noise.vio.pos must be positive"),
    ],
)
def test_invalid_cfg_case(override, match):
    with pytest.raises((TypeError, ValueError), match=match):
        FGOCore(_cfg() | override)


@pytest.mark.parametrize(
    "missing_key",
    [
        "window_size",  # §10.3 第二轮偷懒审视固化：与 τ/ratio 同口径显式写死（fgo.yaml L14），禁代码默认值兜底。
        "optimizer",  # §10.3 第三轮偷懒审视固化：max_iters 与 window_size / τ 同口径（spec 第1663行 K_FGO 上限须固定），子键缺一即 KeyError。
        "nominal_event_rate_hz",
        "tau_filt_s",
        "tau_filt_assumed_t_eff_min_s",  # §10.3 诚实边界守卫：τ_filt=5.0 隐含 T_eff≥25s 协议假设，须显式写死（协议填槽表 L103 C8 修复）。
        "window_length_ratio_min",
        "window_length_ratio_max",
    ],
)
def test_missing_section10_3_key_raises_keyerror(missing_key):
    """§10.3 偷懒审视固化：六键缺一即 KeyError，禁代码默认值兜底。

    spec 第 1663 行要求 window_size / τ_filt / α / β / nominal_event_rate_hz /
    max_iters(K_FGO) 在协议层显式落字（fgo.yaml），不可由代码默认值静默回退——否则 sweep 缺键
    会用未在协议里写死过的默认值凑出合规档位，把「必须可复述」洗成
    「代码凑出了」。
    """
    cfg = _cfg()
    cfg.pop(missing_key, None)  # 真正删键，触发 cfg.get(key, _MISSING) == _MISSING 分支
    with pytest.raises(KeyError, match=missing_key):
        FGOCore(cfg)


def test_missing_section10_3_optimizer_max_iters_raises_keyerror():
    """§10.3 偷懒审视固化第三轮：optimizer dict 存在但 max_iters 子键缺失 → KeyError。

    spec 第 1663 行 K_FGO 上限须固定（§3.4、§0.2），与 τ/window_size 同口径显式
    写死；若仅删 max_iters（保留 optimizer dict）也必须 raise——否则 sweep
    缺子键会用代码默认值凑出合规档位，把「必须可复述」洗成「代码凑出了」。
    """
    cfg = _cfg()
    cfg["optimizer"] = {"name": "gauss_newton"}  # 保留 optimizer dict 仅删 max_iters 子键。
    with pytest.raises(KeyError, match="optimizer.max_iters"):
        FGOCore(cfg)


def test_uwb_noise_must_be_scalar():
    estimator = FGOCore(_cfg() | {"measurement_noise": {"uwb": [0.25], "vio": {"pos": 0.08, "yaw": 0.03}}})
    estimator.step(_imu_event())

    with pytest.raises(ValueError, match="effective UWB noise must be scalar"):
        estimator.step(_uwb_event())


def test_constraint_copy_is_isolated():
    estimator = FGOCore(_cfg())
    estimator.step(_imu_event())
    estimator.step(_vio_event())  # 首帧 VIO 初始化参考位姿（被跳过）
    estimator.step(_imu_event(t=0.2, ax=0.0, ay=0.1))
    estimator.step(_vio_event(t=0.3, dx=0.1, dy=0.0, dyaw=0.02))
    copied = list(estimator.last_update_report["constraint"]["reference_pose"])
    estimator.last_update_report["constraint"]["reference_pose"][0] = 999.0
    assert estimator.constraints[-1]["reference_pose"] == copied


def test_constraint_count_includes_measurements_on_window_head():
    estimator = FGOCore(_cfg() | {"window_size": 2})
    estimator.step(_imu_event(t=0.1))
    estimator.step(_vio_event(t=0.2))  # 首帧 VIO 初始化参考位姿（被跳过）
    estimator.step(_imu_event(t=0.3, ax=0.0, ay=0.1))
    report = estimator.step(_vio_event(t=0.4, dx=0.1, dy=0.2, dyaw=0.03))

    assert report.timestamp == 0.4
    assert estimator.last_update_report["solver_report"]["constraint_count"] == 2


def test_window_head_pose_prior_rebinds_after_trim():
    estimator = FGOCore(_cfg() | {"window_size": 2})
    estimator.step(_imu_event(t=0.1))
    estimator.step(_vio_event(t=0.2, dx=0.35, dy=0.0, dyaw=0.0))
    estimator.step(_imu_event(t=0.3, ax=0.0, ay=0.1))
    trimmed_head = estimator._window_entries[0]

    assert trimmed_head["motion_from_prev"] is None
    assert trimmed_head["pose_prior"].tolist() == estimator._pose_from_state_vector(trimmed_head["full_state"]).tolist()


def test_vio_constraint_keeps_reference_pose_semantics_after_trim():
    """保持测试：vio constraint。\n\n验证 vio constraint 的保持行为，\n确保特定属性在处理过程中不变。
    """
    estimator = FGOCore(_cfg() | {"window_size": 2})
    estimator.step(_imu_event(t=0.1))
    estimator.step(_vio_event(t=0.2, dx=0.35, dy=0.0, dyaw=0.0))
    # 紧耦合扩维后位姿字典含 5 个键 (px/yaw/uwb_clock_bias/vio_scale),
    # VIO 参考只取 px/py/yaw 前三维 (紧耦合项仅用于键集校验, 不参与参考位姿).
    expected_reference_pose = list(estimator._last_vio_reference_pose.values())[:3]
    estimator.step(_imu_event(t=0.3, ax=0.0, ay=0.1))
    estimator.step(_vio_event(t=0.4, dx=0.35, dy=0.0, dyaw=0.0))

    latest_vio = estimator._window_entries[-1]["vio_factors"][-1]

    assert latest_vio["reference_pose"] == expected_reference_pose
    assert estimator._window_entries[0]["motion_from_prev"] is None


def test_second_vio_constraint_links_to_previous_window_state_when_available():
    estimator = FGOCore(_cfg() | {"window_size": 4})
    estimator.step(_imu_event(t=0.1))
    estimator.step(_vio_event(t=0.2, dx=0.35, dy=0.0, dyaw=0.0))
    estimator.step(_imu_event(t=0.3, ax=0.0, ay=0.1))
    estimator.step(_vio_event(t=0.4, dx=0.10, dy=0.05, dyaw=0.02))

    latest_vio = estimator._window_entries[-1]["vio_factors"][-1]

    assert latest_vio["reference_index"] == 1
    assert latest_vio["reference_pose"] == pytest.approx(
        [
            estimator.window_states[1]["px"],
            estimator.window_states[1]["py"],
            estimator.window_states[1]["yaw"],
        ],
        abs=1e-5,
    )


def test_failed_solve_rolls_back_latest_constraint():
    estimator = FGOCore(_cfg())
    estimator.step(_imu_event())
    estimator.step(_vio_event())  # 首帧 VIO 初始化参考位姿（被跳过）
    estimator.step(_imu_event(t=0.2, ax=0.0, ay=0.1))
    estimator.step(_vio_event(t=0.3, dx=0.1, dy=0.0, dyaw=0.02))
    baseline_constraints = len(estimator.constraints)
    baseline_vio_factors = len(estimator._window_entries[-1]["vio_factors"])

    estimator._window_entries[-1]["vio_factors"][-1]["noise"][0][0] = float("nan")

    with pytest.raises(ValueError, match="solve system contains non-finite values"):
        estimator.step(_vio_event(t=0.4, dx=0.2, dy=0.1, dyaw=0.02))

    assert len(estimator.constraints) == baseline_constraints
    assert len(estimator._window_entries[-1]["vio_factors"]) == baseline_vio_factors


def test_solve_updates_pose_covariance_after_window_optimization():
    estimator = FGOCore(_cfg())
    baseline_pose_covariance = estimator._covariance[np.ix_([0, 1, 4], [0, 1, 4])].copy()

    estimator.step(_imu_event())
    estimator.step(_vio_event())  # 首帧 VIO 初始化参考位姿（被跳过）
    estimator.step(_imu_event(t=0.2, ax=0.0, ay=0.1))
    estimator.step(_vio_event(t=0.3, dx=0.1, dy=0.0, dyaw=0.02))

    updated_pose_covariance = estimator._covariance[np.ix_([0, 1, 4], [0, 1, 4])]

    assert not np.allclose(updated_pose_covariance, baseline_pose_covariance)
    assert np.all(np.isfinite(updated_pose_covariance))
    assert estimator.last_update_report["solver_report"]["latest_pose_covariance_diag"] is not None


def test_write_latest_pose_covariance_clamps_negative_diagonal(monkeypatch):
    estimator = FGOCore(_cfg())
    estimator._window_entries = [
        {
            "timestamp": 0.0,
            "full_state": np.zeros(8, dtype=float),
            "pose_prior": np.zeros(3, dtype=float),
            "motion_from_prev": None,
            "uwb_factors": [],
            "vio_factors": [],
        }
    ]
    estimator._covariance = np.eye(8, dtype=float)

    fake_inverse = np.array(
        [
            [-1e-6, 0.0, 0.0],
            [0.0, 2e-6, 0.0],
            [0.0, 0.0, -3e-6],
        ],
        dtype=float,
    )
    monkeypatch.setattr("liquidloc.estimators.fgo_core.np.linalg.inv", lambda _: fake_inverse)

    diag = estimator._write_latest_pose_covariance(np.eye(3, dtype=float))

    assert diag == [0.0, 2e-6, 0.0]
    assert np.all(np.diag(estimator._covariance)[[0, 1, 4]] >= 0.0)


def test_fgo_solve_respects_reference_local_frame_for_linked_vio_factor():
    """尊重测试：fgo solve。\n\n验证被测功能尊重 fgo solve 的规则，\n确保协议约束被正确执行。
    """
    estimator = FGOCore(_cfg())
    reference_state = np.zeros(11, dtype=float)
    reference_state[4] = math.pi / 2.0
    current_state = np.zeros(11, dtype=float)
    current_state[1] = 1.0
    current_state[4] = math.pi / 2.0
    estimator._window_entries = [
        {
            "timestamp": 0.0,
            "full_state": reference_state.copy(),
            "pose_prior": np.asarray([0.0, 0.0, math.pi / 2.0], dtype=float),
            "motion_from_prev": None,
            "uwb_factors": [],
            "vio_factors": [],
        },
        {
            "timestamp": 0.1,
            "full_state": current_state.copy(),
            "pose_prior": np.asarray([0.0, 1.0, math.pi / 2.0], dtype=float),
            "motion_from_prev": None,
            "uwb_factors": [],
            "vio_factors": [
                {
                    "type": "vio",
                    "z_vio": [1.0, 0.0, 0.0],
                    "noise": np.diag([0.08, 0.08, 0.03]).tolist(),
                    "reference_pose": [0.0, 0.0, math.pi / 2.0],
                    "reference_index": 0,
                }
            ],
        },
    ]

    report = estimator.solve()

    assert report["constraint_count"] == 1
    assert estimator.window_states[0]["px"] == pytest.approx(0.0)
    assert estimator.window_states[0]["py"] == pytest.approx(0.0)
    assert estimator.window_states[0]["yaw"] == pytest.approx(math.pi / 2.0)
    assert estimator.window_states[1]["px"] == pytest.approx(0.0, abs=1e-9)
    assert estimator.window_states[1]["py"] == pytest.approx(1.0, abs=1e-9)
    assert estimator.window_states[1]["yaw"] == pytest.approx(math.pi / 2.0, abs=1e-9)


def test_vio_gate_uses_updated_pose_covariance_on_following_step():
    """使用测试：vio gate。\n\n验证被测功能正确使用 vio gate，\n确保内部依赖被正确调用。
    """
    estimator = FGOCore(_cfg())
    estimator.step(_imu_event())
    # 首个 VIO 事件初始化参考位姿（被跳过，不做 solve）
    estimator.step(_vio_event(dx=0.35, dy=0.0, dyaw=0.0))
    # 第二个 VIO 事件做真正的更新和 solve
    estimator.step(_imu_event(t=0.2, ax=0.0, ay=0.1))
    estimator.step(_vio_event(t=0.3, dx=0.10, dy=0.05, dyaw=0.02))

    covariance_before_third_vio = estimator._covariance.copy()
    next_event = _vio_event(t=0.4, dx=0.05, dy=0.02, dyaw=0.01)
    z_vio = build_vio_measurement(next_event)
    reference_pose = estimator._last_vio_reference_pose or estimator._current_pose_reference()
    _, residual, H, _ = compute_vio_residual(estimator._state_vector(), z_vio, reference_pose=reference_pose)
    expected_S = H @ covariance_before_third_vio @ H.T + np.diag([0.08**2, 0.08**2, 0.03**2])
    expected_nis = float(residual.T @ np.linalg.solve(expected_S, residual))

    estimator.step(next_event)

    assert estimator.last_update_report["gate"]["nis"] == pytest.approx(expected_nis)


def test_vio_gate_uses_optimized_reference_pose_after_intermediate_uwb_solve():
    """使用测试：vio gate。\n\n验证被测功能正确使用 vio gate，\n确保内部依赖被正确调用。
    """
    estimator = FGOCore(_cfg() | {"window_size": 4})
    estimator.step(_imu_event(t=0.1))
    estimator.step(_vio_event(t=0.2, dx=0.35, dy=0.0, dyaw=0.0))
    stale_reference_pose = list(estimator._last_vio_reference_pose.values())

    estimator.step(_uwb_event(t=0.25, rng=0.2))
    estimator.step(_imu_event(t=0.3, ax=0.0, ay=0.1))

    reference_index = estimator._last_vio_reference_index
    assert reference_index is not None
    expected_reference_pose = [
        estimator.window_states[reference_index]["px"],
        estimator.window_states[reference_index]["py"],
        estimator.window_states[reference_index]["yaw"],
    ]
    assert expected_reference_pose != pytest.approx(stale_reference_pose)

    estimator.step(_vio_event(t=0.4, dx=0.10, dy=0.05, dyaw=0.02))

    assert estimator.last_update_report["reference_pose"] == pytest.approx(expected_reference_pose)


def test_fgo_wraps_vio_yaw_residual_across_pi_boundary():
    """包装测试：fgo。\n\n验证 fgo 的包装/归一化行为，\n确保角度等值被正确归一化到合法范围。
    """
    estimator = FGOCore(_cfg())
    estimator._state["yaw"] = -3.12
    estimator._window_entries[-1]["full_state"][4] = -3.12
    # 紧耦合扩维后位姿字典需要 5 个键 (含 uwb_clock_bias/vio_scale 占位),
    # 与 fgo_core._pose_dict_from_vector 输出保持一致, 否则
    # _resolve_state_items_from_mapping 会因键集不全抛 KeyError.
    estimator._last_vio_reference_pose = {
        "px": 0.0,
        "py": 0.0,
        "yaw": 3.12,
        "uwb_clock_bias": 0.0,
        "vio_scale": 0.0,
    }
    estimator._last_vio_reference_index = None

    state = estimator.step(_vio_event(dyaw=(2.0 * math.pi) - 6.24))

    assert state.timestamp == pytest.approx(0.2)
    assert estimator.last_update_report["update_applied"] is True
    assert estimator.last_update_report["residual"][2] == pytest.approx(0.0, abs=1e-9)


def test_vio_reference_pose_stale_skips_update():
    """VIO 参考位姿超过 0.5 秒未更新时跳过本次更新并重置参考位姿。"""
    estimator = FGOCore(_cfg())
    estimator.step(_imu_event(t=0.1))
    estimator.step(_uwb_event(t=0.2))
    # 首个 VIO 事件初始化参考位姿（被跳过）
    estimator.step(_vio_event(t=0.3))
    # 第二个 VIO 事件做真正的更新
    estimator.step(_imu_event(t=0.35, ax=0.0, ay=0.1))
    estimator.step(_vio_event(t=0.4))
    assert estimator.last_update_report["update_applied"] is True
    # 模拟参考位姿过时：设置参考位姿时间戳为很久以前
    estimator._last_vio_reference_pose_timestamp = 0.0
    # 下一个 VIO 事件应跳过更新
    state_before = estimator.get_state()
    estimator.step(_vio_event(t=0.6))
    report = estimator.last_update_report
    assert report["update_applied"] is False
    assert report["reason"] == "vio_reference_pose_stale"
    # 参考位姿应被重置
    assert estimator._last_vio_reference_pose is not None


def test_tau_filt_assumed_t_eff_min_s_consistency_guard_rejects_inconsistent_tau():
    """§10.3 诚实边界守卫：tau_filt_s 超过 spec §10.3 选项 1 上限 min(5, 0.2·T_eff_assumed) 时 ValueError。

    协议填槽表 L103 此前记录 C8 半过原因之一是 τ_filt 隐含假设未在 yaml 显式落地；
    本测试验证 fgo_core.py sentinel 强制 tau_filt ≤ min(5, 0.2 × assumed_T_eff) 一致性——
    若 tau_filt_s=6.0 而 assumed_T_eff=25.0（min(5,5)=5），6 > 5 应 fail-loud。
    """
    with pytest.raises(ValueError, match="§10.3 τ_filt 诚实边界违例"):
        FGOCore(_cfg() | {"tau_filt_s": 6.0, "tau_filt_assumed_t_eff_min_s": 25.0})


def test_tau_filt_assumed_t_eff_min_s_consistency_guard_passes_when_consistent():
    """tau_filt_s 与 tau_filt_assumed_t_eff_min_s 一致时通过（边界值）。

    tau_filt_s=5.0, assumed_T_eff=25.0 → min(5, 0.2×25)=5.0 → 5.0 ≤ 5.0 合规。
    同步用 window_size=50, nominal_event_rate_hz=10 使 ℓ_win=5s, ratio=1.0 ∈ [0.5, 3] 合规。
    """
    estimator = FGOCore(_cfg() | {
        "tau_filt_s": 5.0, "tau_filt_assumed_t_eff_min_s": 25.0,
        "window_size": 50, "nominal_event_rate_hz": 10.0,
    })
    assert estimator.tau_filt_s == 5.0
    assert estimator.tau_filt_assumed_t_eff_min_s == 25.0


def test_tau_filt_assumed_t_eff_min_s_consistency_guard_passes_when_tau_below_bound():
    """tau_filt_s 低于 min(5, 0.2·T_eff_assumed) 上限时通过（更保守的 τ 假设）。

    tau_filt_s=4.0, assumed_T_eff=25.0 → min(5, 5)=5 → 4.0 ≤ 5 合规。
    同步用 window_size=40, nominal_event_rate_hz=10 使 ℓ_win=4s, ratio=1.0 ∈ [0.5, 3] 合规。
    """
    estimator = FGOCore(_cfg() | {
        "tau_filt_s": 4.0, "tau_filt_assumed_t_eff_min_s": 25.0,
        "window_size": 40, "nominal_event_rate_hz": 10.0,
    })
    assert estimator.tau_filt_s == 4.0


# ════════════════════════════════════════════════════════════
#  §11.5 UWB 路径标量 S jitter fallback（v8 修复锁死）
#  修复位置：src/liquidloc/estimators/fgo_core.py:1883-1907（UWB S<=0 守门）
#  v6 audit 发现 v5 仅修了 VIO 路径与 uwb_update_step 内部 stacked-H 路径的
#  jitter fallback，但漏审 FGO UWB 路径的标量 S<=0 守门。
#
#  设计诚实声明：
#  对 P 正定 + scalar_noise ≥ 0，S = H·P·Hᵀ + scalar_noise ≥ 0 恒成立（数学期望）。
#  本测试用正交投影构造 P 使 H·P·Hᵀ ≈ 1e-12（正定但极小），再通过 monkeypatch
#  coerce_finite_scalar 把 scalar_noise 设为负值，精确控制 S 进入 v8 jitter 分支。
#  锁死：(a) 可恢复 S（|S| < eps）走 jitter 救活，update_applied=True；
#       (b) 不可恢复 S（S < -eps）仍 fail-loud 拒绝。
# ════════════════════════════════════════════════════════════

class TestUwbScalarSJitterFallbackFGO:
    """§11.5 UWB 路径标量 S jitter fallback 锁死（FGO 单元）。"""

    @staticmethod
    def _orthogonal_pd_P(ekf):
        """构造 P 正定但 H 行空间投影 ≈ 0（使 H·P·Hᵀ ≈ 1e-12）。

        P = I - (1 - 1e-12) · vvᵀ / ||v||²，v = H[0] 行向量。
        """
        from liquidloc.estimators.uwb_update_step import build_uwb_jacobian
        P = np.eye(ekf._covariance.shape[0])
        x_prev = ekf._state_vector()
        H = build_uwb_jacobian(x_prev, ekf._resolve_anchor_position(0))
        v = H[0]
        v_norm_sq = float(v @ v)
        P -= (1 - 1e-12) * np.outer(v, v) / v_norm_sq
        return P

    def test_uwb_recoverable_S_triggers_jitter_fallback(self, monkeypatch):
        """§11.5 可恢复病态 S（|S| < cov_jitter_eps）走 jitter 救活。"""
        from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
        from liquidloc.estimators.uwb_update_step import build_uwb_jacobian
        import liquidloc.estimators.fgo_core as fgo_mod
        eps = float(BRIDGE_THRESHOLDS["cov_jitter_eps"])
        estimator = FGOCore(_cfg())
        estimator._covariance = self._orthogonal_pd_P(estimator)
        x_prev = estimator._state_vector()
        anchor_pos = estimator._resolve_anchor_position(0)
        H = build_uwb_jacobian(x_prev, anchor_pos)
        HPtH = float((H @ estimator._covariance @ H.T)[0, 0])

        target_noise = -5e-10 - HPtH  # S = -5e-10 ∈ (-eps, 0]
        # 注意：FGO _coerce_uwb_noise_scalar 内部有 min_value=0.0, inclusive=False 守门，
        # 直接 monkeypatch coerce_finite_scalar 会被那个守门拦截。
        # 改 monkeypatch _coerce_uwb_noise_scalar 直接返回负值，绕过 0 守门。
        monkeypatch.setattr(fgo_mod, "_coerce_uwb_noise_scalar", lambda *a, **kw: target_noise)

        control = MeasurementControl(modality="uwb", gate_action="pass_through")
        result = estimator._handle_uwb(_uwb_event(), x_prev, control)

        assert result["update_applied"] is True, \
            f"§11.5 可恢复病态 S 应被 jitter 救活；reason={result.get('reason')}"
        assert result.get("reason") != "nonpositive_innovation_covariance"

    def test_uwb_unrecoverable_S_still_fails_loud(self, monkeypatch):
        """§11.5 不可恢复病态 S（S < -eps）仍 fail-loud 拒绝。"""
        from liquidloc.estimators.uwb_update_step import build_uwb_jacobian
        import liquidloc.estimators.fgo_core as fgo_mod
        estimator = FGOCore(_cfg())
        estimator._covariance = self._orthogonal_pd_P(estimator)
        x_prev = estimator._state_vector()
        anchor_pos = estimator._resolve_anchor_position(0)
        H = build_uwb_jacobian(x_prev, anchor_pos)
        HPtH = float((H @ estimator._covariance @ H.T)[0, 0])

        target_noise = -0.0625 - HPtH  # S ≈ -0.0625 << -eps=1e-9
        monkeypatch.setattr(fgo_mod, "_coerce_uwb_noise_scalar", lambda *a, **kw: target_noise)

        control = MeasurementControl(modality="uwb", gate_action="pass_through")
        result = estimator._handle_uwb(_uwb_event(), x_prev, control)

        assert result["update_applied"] is False, \
            "§11.5 不可恢复病态 S 必须被拒绝，jitter fallback 不应掩盖真病态"
        assert result.get("reason") == "nonpositive_innovation_covariance"
