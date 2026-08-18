from __future__ import annotations

"""融合运行器（fusion_runner）测试模块。

文件职责：验证 run_fusion 能正确执行传感器融合流程，
包括 EKF 状态估计、模型输出桥接、诊断记录等。

测试覆盖范围：
- EKF 纯估计路径
- Liquid EKF 模型桥接路径
- 模型输出缺失键拒绝
- 模型输出多余键拒绝
- 可迭代事件输入
- 时间戳精确保留
- 诊断记录 applied 控制量
- 基准场景 applied 等于 raw
- 安全模式配置
- 负偏差裁剪
- 滤波感知 readout 上下文注入
- readout 上下文无交叉干扰
- Mapping 类型 last_update_report 兼容
- readout 辅助函数
- 非有限协方差处理
- 同时间戳事件保持输入顺序
- 顺序不被静默归一化
- ModelIntermediate 实例输入分支
- bias/risk/scaling 值域裁剪
- bool/NaN/inf/非 dict 模型输出拒绝
- run_fusion 输入守卫（None/str/Mapping/空序列/estimator=None）

被测模块：liquidloc.pipelines.core_pipeline（run_fusion）"""


from collections import UserDict
import math

import pytest

from liquidloc.common.constants import STATE_ITEMS as _STATE_ITEMS
from liquidloc.common.types import ModelIntermediate
from liquidloc.factories.estimator_factory import create_estimator
from liquidloc.factories.model_factory import create_model
from liquidloc.fusion.fusion_runner import _build_readout_context
from liquidloc.fusion.fusion_runner import _coerce_intermediate
from liquidloc.fusion.fusion_runner import _coerce_residual_norm
from liquidloc.fusion.fusion_runner import _init_readout_context_cache
from liquidloc.fusion.fusion_runner import _update_readout_context_cache
from liquidloc.fusion.fusion_runner import run_fusion
from liquidloc.protocol.liquid_bridge_contract import build_measurement_control


# 紧耦合扩维后 8→10 维 (px, py, vx, vy, yaw, bax, bay, bg, uwb_clock_bias, vio_scale).
# Bug 2 (2026-07-23 audit Round 1+): uwb_anchor_bias 已删, 消解雅可比秩亏.
_STATE_DIM = len(_STATE_ITEMS)


def _events():
    return [
        {'t': 0.0, 'dt': 0.0, 'modality': 'imu', 'meta': {'scene_id': 'S(A2,N2,V2,G1,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.1, 'ay': 0.0, 'gz': 0.01}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.05, 'dt': 0.05, 'modality': 'uwb', 'meta': {'scene_id': 'S(A2,N2,V2,G1,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 0.10}, 'vio_payload': None},
        {'t': 0.08, 'dt': 0.03, 'modality': 'vio', 'meta': {'scene_id': 'S(A2,N2,V2,G1,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.03, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.85, 'tracked_features': 20, 'reproj_err': 1.4}},
    ]


def _estimator_cfg():
    return {
        'process_noise': {
            'pos': 0.05, 'vel': 0.10, 'yaw': 0.02, 'accel_bias': 0.001, 'gyro_bias': 0.001,
            # 紧耦合扩维新增过程噪声键 (8→10, Bug 2 删 uwb_anchor_bias), 与 src/ 状态项一致.
            'uwb_clock_bias': 0.001, 'vio_scale': 0.001,
        },
        'measurement_noise': {
            'uwb': 0.25,
            'vio': {'pos': 0.08, 'yaw': 0.03},
        },
        'anchor_layout': {
            'anchor_ids': [0, 1],
            'anchor_positions': [(0.0, 0.0), (1.0, 0.0)],
        },
        'init_state': {
            'px': 0.0,
            'py': 0.0,
            'vx': 0.0,
            'vy': 0.0,
            'yaw': 0.0,
            'bax': 0.0,
            'bay': 0.0,
            'bg': 0.0,
            # 紧耦合扩维新增状态项 (8→10, Bug 2 删 uwb_anchor_bias), 与 configs/base/task.yaml state_items 一致,
            # vio_scale=1.0 (恒等初值, 因 VIO 量测本身是 1:1).
            'uwb_clock_bias': 0.0,
            'vio_scale': 1.0,
        },
        'init_cov': [1.0] * _STATE_DIM,
    }


def _feature_builder(history, event, cfg):
    if event['modality'] == 'uwb':
        return {
            'current_modality': 'uwb',
            'feature_order': ['quality', 'valid', 'tracked_features', 'reproj_err'],
            'feature_values': [event['uwb_payload']['quality'], 0.0, 0.0, 0.0],
            'missing_mask': [0, 1, 1, 1],
            'feature_window': [[event['uwb_payload']['quality'], 0.0, 0.0, 0.0]],
            'missing_mask_window': [[0, 1, 1, 1]],
        }
    return {
        'current_modality': 'vio',
        'feature_order': ['quality', 'valid', 'tracked_features', 'reproj_err'],
        'feature_values': [event['vio_payload']['quality'], 0.0, 0.0, event['vio_payload']['reproj_err']],
        'missing_mask': [0, 1, 1, 0],
        'feature_window': [[event['vio_payload']['quality'], 0.0, 0.0, event['vio_payload']['reproj_err']]],
        'missing_mask_window': [[0, 1, 1, 0]],
    }


def _expected_control(event, *, bias: float, risk: float, uwb_scaling: float, vio_scaling: float, safe_mode_cfg=None):
    return build_measurement_control(
        event,
        ModelIntermediate(
            bias=float(bias),
            risk=float(risk),
            uwb_scaling=float(uwb_scaling),
            vio_scaling=float(vio_scaling),
        ),
        safe_mode_cfg=safe_mode_cfg,
    )


def test_normal_case():
    estimator = create_estimator('ekf', _estimator_cfg())
    model = create_model('liquid_ekf', {})
    bundle = run_fusion(
        _events(),
        estimator,
        model,
        feature_builder=_feature_builder,
        cfg={'method_name': 'liquid_ekf'},
    )
    assert bundle['method_name'] == 'liquid_ekf'
    assert len(bundle['states']) == 3
    assert len(bundle['diagnostics']['risk_trace']) == 3
    assert 0.0 <= bundle['diagnostics']['risk_trace'][1] <= 1.0
    assert 0.0 <= bundle['diagnostics']['risk_trace'][2] <= 1.0
    # v2 放宽 uwb_hard_skip_quality_floor=0.10 后, quality=0.10 不再触发 uwb_skip_update
    # (q+epsilon 不严格小于 floor, 边界 release 出去); 改为走 uwb_bias_and_noise_scale 默认动作.
    assert bundle['diagnostics']['gate_action_trace'][1] == 'uwb_bias_and_noise_scale'


def test_boundary_case():
    estimator = create_estimator('ekf', _estimator_cfg())
    bundle = run_fusion(_events(), estimator, cfg={'method_name': 'ekf'})
    assert bundle['diagnostics']['bias_trace'] == [0.0, 0.0, 0.0]
    assert bundle['mechanism_contract']['bridge_enabled'] is False


def test_invalid_case():
    with pytest.raises(ValueError):
        run_fusion(_events(), create_estimator('ekf', _estimator_cfg()), create_model('liquid_ekf', {}))


def test_invalid_model_output_rejected():
    estimator = create_estimator('ekf', _estimator_cfg())

    class _BadModel:
        def infer_intermediate(self, window_tensor):
            return {'risk': 0.2}

    with pytest.raises(ValueError, match='missing required keys'):
        run_fusion(_events(), estimator, _BadModel(), feature_builder=_feature_builder, cfg={'method_name': 'liquid_ekf'})


def test_model_output_with_extra_keys_is_rejected_to_preserve_measurement_level_bridge():
    estimator = create_estimator('ekf', _estimator_cfg())

    class _BadModel:
        def infer_intermediate(self, window_tensor):
            return {
                'bias': 0.1,
                'risk': 0.2,
                'uwb_scaling': 1.1,
                'vio_scaling': 1.2,
                'state_delta': [1.0, 2.0],
            }

    with pytest.raises(ValueError, match='unsupported keys'):
        run_fusion(_events(), estimator, _BadModel(), feature_builder=_feature_builder, cfg={'method_name': 'liquid_ekf'})


def test_iterable_events_case():
    estimator = create_estimator('ekf', _estimator_cfg())
    bundle = run_fusion(iter(_events()), estimator, cfg={'method_name': 'ekf'})
    assert len(bundle['states']) == 3
    assert bundle['timestamps'] == [0.0, 0.05, 0.08]


def test_timestamps_preserve_event_times_exactly():
    estimator = create_estimator('ekf', _estimator_cfg())
    bundle = run_fusion(_events(), estimator, cfg={'method_name': 'ekf'})
    assert bundle['timestamps'] == [0.0, 0.05, 0.08]


def test_diagnostics_record_applied_controls_not_only_raw_model_outputs():
    estimator = create_estimator('ekf', _estimator_cfg())

    class _DeterministicModel:
        def infer_intermediate(self, window_tensor):
            return {
                'bias': 0.7,
                'risk': 0.2,
                'uwb_scaling': 1.3,
                'vio_scaling': 1.1,
            }

    bundle = run_fusion(
        _events(),
        estimator,
        _DeterministicModel(),
        feature_builder=_feature_builder,
        cfg={'method_name': 'liquid_ekf'},
    )
    uwb_control = _expected_control(_events()[1], bias=0.7, risk=0.2, uwb_scaling=1.3, vio_scaling=1.1)
    vio_control = _expected_control(_events()[2], bias=0.7, risk=0.2, uwb_scaling=1.3, vio_scaling=1.1)

    assert bundle['diagnostics']['bias_trace'][1] == pytest.approx(0.7)
    assert bundle['diagnostics']['risk_trace'][1] == pytest.approx(0.2)
    assert bundle['diagnostics']['applied_bias_trace'][1] == pytest.approx(uwb_control.bias_applied)
    assert bundle['diagnostics']['applied_bias_trace'][2] == pytest.approx(vio_control.bias_applied)
    assert bundle['diagnostics']['applied_risk_trace'][1] == pytest.approx(uwb_control.risk)
    assert bundle['diagnostics']['applied_risk_trace'][2] == pytest.approx(vio_control.risk)
    assert bundle['diagnostics']['applied_uwb_scaling_trace'][1] == pytest.approx(uwb_control.scaling)
    assert bundle['diagnostics']['applied_vio_scaling_trace'][2] == pytest.approx(vio_control.scaling)
    assert bundle['diagnostics']['noise_multiplier_trace'][1] == pytest.approx(uwb_control.noise_multiplier)
    assert bundle['diagnostics']['noise_multiplier_trace'][2] == pytest.approx(vio_control.noise_multiplier)


def test_scene_default_path_keeps_applied_controls_equal_to_raw_controls():
    events = [
        {'t': 0.0, 'dt': 0.0, 'modality': 'imu', 'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.1, 'ay': 0.0, 'gz': 0.01}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.05, 'dt': 0.05, 'modality': 'uwb', 'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 1.0}, 'vio_payload': None},
        {'t': 0.08, 'dt': 0.03, 'modality': 'vio', 'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.03, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.95, 'tracked_features': 150, 'reproj_err': 0.1}},
    ]
    estimator = create_estimator('ekf', _estimator_cfg())

    class _DeterministicModel:
        def infer_intermediate(self, window_tensor):
            return {
                'bias': 0.8,
                'risk': 0.4,
                'uwb_scaling': 1.5,
                'vio_scaling': 1.6,
            }

    bundle = run_fusion(
        events,
        estimator,
        _DeterministicModel(),
        feature_builder=_feature_builder,
        cfg={'method_name': 'liquid_ekf'},
    )
    uwb_control = _expected_control(events[1], bias=0.8, risk=0.4, uwb_scaling=1.5, vio_scaling=1.6)
    vio_control = _expected_control(events[2], bias=0.8, risk=0.4, uwb_scaling=1.5, vio_scaling=1.6)

    assert bundle['diagnostics']['bias_trace'][1] == pytest.approx(0.8)
    assert bundle['diagnostics']['uwb_scaling_trace'][1] == pytest.approx(1.5)
    assert bundle['diagnostics']['vio_scaling_trace'][2] == pytest.approx(1.6)
    assert bundle['diagnostics']['applied_bias_trace'][1] == pytest.approx(uwb_control.bias_applied)
    assert bundle['diagnostics']['applied_bias_trace'][2] == pytest.approx(vio_control.bias_applied)
    assert bundle['diagnostics']['applied_uwb_scaling_trace'][1] == pytest.approx(uwb_control.scaling)
    assert bundle['diagnostics']['applied_vio_scaling_trace'][2] == pytest.approx(vio_control.scaling)
    assert bundle['diagnostics']['noise_multiplier_trace'][1] == pytest.approx(uwb_control.noise_multiplier)
    assert bundle['diagnostics']['noise_multiplier_trace'][2] == pytest.approx(vio_control.noise_multiplier)
    assert bundle['diagnostics']['gate_action_trace'][1] == 'uwb_bias_and_noise_scale'


def test_run_fusion_safe_mode_cfg_can_enable_nominal_scene_convergence():
    events = [
        {'t': 0.0, 'dt': 0.0, 'modality': 'imu', 'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.1, 'ay': 0.0, 'gz': 0.01}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.05, 'dt': 0.05, 'modality': 'uwb', 'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 1.0}, 'vio_payload': None},
    ]
    estimator = create_estimator('ekf', _estimator_cfg())

    class _DeterministicModel:
        def infer_intermediate(self, window_tensor):
            return {
                'bias': 0.8,
                'risk': 0.4,
                'uwb_scaling': 1.5,
                'vio_scaling': 1.6,
            }

    bundle = run_fusion(
        events,
        estimator,
        _DeterministicModel(),
        feature_builder=_feature_builder,
        cfg={
            'method_name': 'liquid_ekf',
            'safe_mode': {'enabled': True, 'risk_threshold': 0.5},
        },
    )

    assert bundle['diagnostics']['bias_trace'][1] == pytest.approx(0.8)
    assert bundle['diagnostics']['applied_bias_trace'][1] == pytest.approx(0.32)
    assert bundle['diagnostics']['applied_uwb_scaling_trace'][1] == pytest.approx(1.2)


def test_negative_bias_is_clamped_before_applied_uwb_control():
    estimator = create_estimator('ekf', _estimator_cfg())

    class _DeterministicModel:
        def infer_intermediate(self, window_tensor):
            return {
                'bias': -0.5,
                'risk': 0.4,
                'uwb_scaling': 1.2,
                'vio_scaling': 1.0,
            }

    bundle = run_fusion(
        _events(),
        estimator,
        _DeterministicModel(),
        feature_builder=_feature_builder,
        cfg={'method_name': 'liquid_ekf'},
    )

    # bias_trace 记录经过 _coerce_intermediate 处理后的值，非负约束将 -0.5 裁剪为 0.0
    assert bundle['diagnostics']['bias_trace'][1] == pytest.approx(0.0)
    assert bundle['diagnostics']['applied_bias_trace'][1] == pytest.approx(0.0)


def test_run_fusion_injects_filter_aware_readout_context_into_feature_window():
    estimator = create_estimator('ekf', _estimator_cfg())
    captured_windows = []

    class _CapturingModel:
        def infer_intermediate(self, window_tensor):
            captured_windows.append(dict(window_tensor))
            return {
                'bias': 0.0,
                'risk': 0.0,
                'uwb_scaling': 1.0,
                'vio_scaling': 1.0,
            }

    run_fusion(
        _events(),
        estimator,
        _CapturingModel(),
        feature_builder=_feature_builder,
        cfg={'method_name': 'liquid_ekf'},
    )

    assert len(captured_windows) == 2
    uwb_window, vio_window = captured_windows
    assert "readout_context_by_name" in uwb_window
    assert "readout_context_observed_by_name" in uwb_window
    assert uwb_window["readout_context_observed_by_name"]["state_cov_trace"] is True
    assert uwb_window["readout_context_observed_by_name"]["pos_cov"] is True
    assert uwb_window["readout_context_observed_by_name"]["last_innovation_norm"] is False
    assert uwb_window["readout_context_observed_by_name"]["last_gate_skip_flag"] is False
    assert vio_window["readout_context_observed_by_name"]["last_innovation_norm"] is False
    assert vio_window["readout_context_observed_by_name"]["last_gate_skip_flag"] is False


def test_run_fusion_updates_readout_context_cache_per_modality_without_cross_talk():
    estimator = create_estimator('ekf', _estimator_cfg())
    captured_windows = []
    events = [
        {'t': 0.0, 'dt': 0.0, 'modality': 'imu', 'meta': {'scene_id': 'S(A2,N2,V2,G1,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.1, 'ay': 0.0, 'gz': 0.01}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.05, 'dt': 0.05, 'modality': 'uwb', 'meta': {'scene_id': 'S(A2,N2,V2,G1,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 1.0}, 'vio_payload': None},
        # v3 铁律 7: 同 ±5ms 窗内 UWB/VIO 批量 step, 但 UWB 2 在 feature_window 构建后才
        # 到 flush, 之前的 UWB innovation 不可见. 在两者之间加 IMU 事件触发 IMU 路径先
        # flush pending_buffer (L759-772), UWB 1 在 UWB 2 窗口构造前 step 完成把
        # innovation_observed=True 写入 cache.
        {'t': 0.075, 'dt': 0.025, 'modality': 'imu', 'meta': {'scene_id': 'S(A2,N2,V2,G1,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.10, 'dt': 0.025, 'modality': 'uwb', 'meta': {'scene_id': 'S(A2,N2,V2,G1,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 1.8, 'valid': True, 'quality': 1.0}, 'vio_payload': None},
        {'t': 0.20, 'dt': 0.10, 'modality': 'vio', 'meta': {'scene_id': 'S(A2,N2,V2,G1,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.03, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.85, 'tracked_features': 80, 'reproj_err': 0.2}},
        {'t': 0.27, 'dt': 0.07, 'modality': 'vio', 'meta': {'scene_id': 'S(A2,N2,V2,G1,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.04, 'dy': 0.0, 'dyaw': 0.0, 'quality': 0.88, 'tracked_features': 90, 'reproj_err': 0.15}},
    ]

    class _CapturingModel:
        def infer_intermediate(self, window_tensor):
            captured_windows.append(dict(window_tensor))
            return {
                'bias': 0.0,
                'risk': 0.0,
                'uwb_scaling': 1.0,
                'vio_scaling': 1.0,
            }

    run_fusion(
        events,
        estimator,
        _CapturingModel(),
        feature_builder=_feature_builder,
        cfg={'method_name': 'liquid_ekf', 'safe_mode': {'enabled': False}},
    )

    # captured: [uwb1, uwb2(=second), vio1, vio2(=second)]
    second_uwb_window = captured_windows[1]
    first_vio_window = captured_windows[2]
    second_vio_window = captured_windows[3]
    assert second_uwb_window["readout_context_observed_by_name"]["last_innovation_norm"] is True
    assert second_uwb_window["readout_context_by_name"]["last_innovation_norm"] >= 0.0
    assert second_uwb_window["readout_context_observed_by_name"]["last_gate_skip_flag"] is True
    assert second_uwb_window["readout_context_by_name"]["last_gate_skip_flag"] == pytest.approx(0.0)
    assert first_vio_window["readout_context_observed_by_name"]["last_innovation_norm"] is False
    assert first_vio_window["readout_context_observed_by_name"]["last_gate_skip_flag"] is False
    # 首帧VIO事件跳过更新（初始化参考位姿），不产生VIO innovation。
    # 第二个VIO事件正常更新，readout_context 中的 last_innovation_norm
    # 来自前一次成功更新（可能是UWB），VIO模态的innovation_norm可能仍为未观测。
    # 这是因为首帧VIO跳过后，VIO模态的innovation尚未产生。
    assert second_vio_window["readout_context_observed_by_name"]["last_innovation_norm"] is False
    assert second_vio_window["readout_context_by_name"]["last_innovation_norm"] >= 0.0


def test_run_fusion_accepts_mapping_like_last_update_report_for_readout_context_cache():
    estimator = create_estimator('ekf', _estimator_cfg())
    captured_windows = []

    class _MappingReportEstimator:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.last_update_report = None

        def reset(self, initial_state=None):
            if hasattr(self._wrapped, 'reset'):
                self._wrapped.reset(initial_state)
            self.last_update_report = None

        def get_state(self):
            return self._wrapped.get_state()

        def step(self, event):
            state_estimate = self._wrapped.step(event)
            report = getattr(self._wrapped, 'last_update_report', None)
            self.last_update_report = UserDict(dict(report)) if isinstance(report, dict) else report
            return state_estimate

    class _CapturingModel:
        def infer_intermediate(self, window_tensor):
            captured_windows.append(dict(window_tensor))
            return {
                'bias': 0.0,
                'risk': 0.0,
                'uwb_scaling': 1.0,
                'vio_scaling': 1.0,
            }

    wrapped_estimator = _MappingReportEstimator(estimator)
    run_fusion(
        [
            {'t': 0.0, 'dt': 0.0, 'modality': 'imu', 'meta': {'scene_id': 'S(A2,N2,V2,G1,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.1, 'ay': 0.0, 'gz': 0.01}, 'uwb_payload': None, 'vio_payload': None},
            {'t': 0.05, 'dt': 0.05, 'modality': 'uwb', 'meta': {'scene_id': 'S(A2,N2,V2,G1,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 1.0}, 'vio_payload': None},
            # 加 IMU 事件 flush pending_buffer, 让第一个 UWB 在第二个窗口前先 step,
            # 把 innovation_observed=True 写入 cache (同 without_cross_talk 测试模式).
            {'t': 0.075, 'dt': 0.025, 'modality': 'imu', 'meta': {'scene_id': 'S(A2,N2,V2,G1,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0}, 'uwb_payload': None, 'vio_payload': None},
            {'t': 0.10, 'dt': 0.025, 'modality': 'uwb', 'meta': {'scene_id': 'S(A2,N2,V2,G1,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 1.8, 'valid': True, 'quality': 1.0}, 'vio_payload': None},
        ],
        wrapped_estimator,
        _CapturingModel(),
        feature_builder=_feature_builder,
        cfg={'method_name': 'liquid_ekf', 'safe_mode': {'enabled': False}},
    )

    assert captured_windows[1]["readout_context_observed_by_name"]["last_innovation_norm"] is True
    assert captured_windows[1]["readout_context_observed_by_name"]["last_gate_skip_flag"] is True


def test_fusion_readout_context_helpers_capture_vio_residual_norm_and_covariance_summary():
    estimator = create_estimator('ekf', _estimator_cfg())
    cache = _init_readout_context_cache()
    report = {
        'modality': 'vio',
        'update_applied': True,
        'reason': 'vio_update',
        'residual': [3.0, 4.0],
    }

    assert _coerce_residual_norm(report['residual'], modality='vio') == pytest.approx(5.0)
    _update_readout_context_cache(cache, report)
    values, observed = _build_readout_context(estimator, cache, modality='vio')

    assert observed['state_cov_trace'] is True
    assert observed['pos_cov'] is True
    assert observed['last_innovation_norm'] is True
    assert observed['last_gate_skip_flag'] is True
    assert values['state_cov_trace'] > 0.0
    assert values['pos_cov'] > 0.0
    assert values['last_innovation_norm'] == pytest.approx(5.0)
    assert values['last_gate_skip_flag'] == pytest.approx(0.0)


def test_build_readout_context_keeps_pos_cov_when_tail_covariance_entries_are_non_finite():
    class _EstimatorState:
        covariance_diag = [4.0, 5.0, float("nan")]

    class _Estimator:
        def get_state(self):
            return _EstimatorState()

    values, observed = _build_readout_context(
        _Estimator(),
        _init_readout_context_cache(),
        modality="uwb",
    )

    assert observed["state_cov_trace"] is False
    assert observed["pos_cov"] is True
    assert values["state_cov_trace"] == pytest.approx(0.0)
    assert values["pos_cov"] == pytest.approx(math.sqrt(9.0))


def test_same_timestamp_events_are_consumed_in_input_order():
    events = [
        {'t': 0.0, 'dt': 0.0, 'modality': 'imu', 'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.1, 'dt': 0.1, 'modality': 'vio', 'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.3, 'dy': 0.0, 'dyaw': 0.0, 'quality': 1.0, 'tracked_features': 120, 'reproj_err': 0.1}},
        {'t': 0.1, 'dt': 0.0, 'modality': 'uwb', 'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 1.0, 'valid': True, 'quality': 1.0}, 'vio_payload': None},
    ]

    estimator = create_estimator('ekf', _estimator_cfg())
    bundle = run_fusion(events, estimator, cfg={'method_name': 'ekf'})

    assert bundle['diagnostics']['modalities'] == ['imu', 'vio', 'uwb']
    assert bundle['timestamps'] == [0.0, 0.1, 0.1]


def test_same_timestamp_order_is_not_silently_normalized_to_fixed_modality_priority():
    uwb_then_vio = [
        {'t': 0.0, 'dt': 0.0, 'modality': 'imu', 'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': {'ax': 0.0, 'ay': 0.0, 'gz': 0.0}, 'uwb_payload': None, 'vio_payload': None},
        {'t': 0.1, 'dt': 0.1, 'modality': 'uwb', 'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 1.0, 'valid': True, 'quality': 1.0}, 'vio_payload': None},
        {'t': 0.1, 'dt': 0.0, 'modality': 'vio', 'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.3, 'dy': 0.0, 'dyaw': 0.0, 'quality': 1.0, 'tracked_features': 120, 'reproj_err': 0.1}},
    ]
    vio_then_uwb = [
        uwb_then_vio[0],
        {'t': 0.1, 'dt': 0.1, 'modality': 'vio', 'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': None, 'vio_payload': {'dx': 0.3, 'dy': 0.0, 'dyaw': 0.0, 'quality': 1.0, 'tracked_features': 120, 'reproj_err': 0.1}},
        {'t': 0.1, 'dt': 0.0, 'modality': 'uwb', 'meta': {'scene_id': 'S(A0,N0,V0,G0,K6)', 'seq_id': 'mini_seq'}, 'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 1.0, 'valid': True, 'quality': 1.0}, 'vio_payload': None},
    ]

    uwb_bundle = run_fusion(uwb_then_vio, create_estimator('ekf', _estimator_cfg()), cfg={'method_name': 'ekf'})
    vio_bundle = run_fusion(vio_then_uwb, create_estimator('ekf', _estimator_cfg()), cfg={'method_name': 'ekf'})

    assert uwb_bundle['diagnostics']['modalities'] == ['imu', 'uwb', 'vio']
    assert vio_bundle['diagnostics']['modalities'] == ['imu', 'vio', 'uwb']
    # 首帧VIO事件跳过更新（初始化参考位姿），所以两种顺序下VIO都不更新，
    # 最终状态由UWB更新决定，两种顺序结果相同。这是正确行为。
    # 如果需要测试顺序敏感性，需要至少两个VIO事件（首帧初始化+次帧更新）。
    assert uwb_bundle['states'][-1]['px'] == pytest.approx(vio_bundle['states'][-1]['px'])


def test_coerce_intermediate_accepts_model_intermediate_instance():
    """_coerce_intermediate 接受 ModelIntermediate 实例（而非 dict）并正确提取字段。"""
    intermediate = ModelIntermediate(bias=0.5, risk=0.3, uwb_scaling=1.5, vio_scaling=1.2)
    result = _coerce_intermediate(intermediate)
    assert isinstance(result, ModelIntermediate)
    assert result.bias == pytest.approx(0.5)
    assert result.risk == pytest.approx(0.3)
    assert result.uwb_scaling == pytest.approx(1.5)
    assert result.vio_scaling == pytest.approx(1.2)


def test_coerce_intermediate_clamps_bias_to_bridge_bias_max():
    """bias 超过 BRIDGE_BIAS_MAX(=10.0) 时被裁剪到上界。"""
    result = _coerce_intermediate({
        'bias': 15.0,
        'risk': 0.2,
        'uwb_scaling': 1.0,
        'vio_scaling': 1.0,
    })
    assert result.bias == pytest.approx(10.0)


def test_coerce_intermediate_clamps_risk_to_bridge_range():
    """risk 超过 risk_max(=1.0) 或低于 risk_min(=0.0) 时被裁剪到边界。"""
    result_high = _coerce_intermediate({
        'bias': 0.0, 'risk': 1.5, 'uwb_scaling': 1.0, 'vio_scaling': 1.0,
    })
    assert result_high.risk == pytest.approx(1.0)
    result_low = _coerce_intermediate({
        'bias': 0.0, 'risk': -0.3, 'uwb_scaling': 1.0, 'vio_scaling': 1.0,
    })
    assert result_low.risk == pytest.approx(0.0)


def test_coerce_intermediate_clamps_scaling_to_bridge_range():
    """scaling 超出 [scaling_min=1.0, scaling_max=50.0] 时被裁剪到边界。"""
    result_high = _coerce_intermediate({
        'bias': 0.0, 'risk': 0.0, 'uwb_scaling': 100.0, 'vio_scaling': 1.0,
    })
    assert result_high.uwb_scaling == pytest.approx(50.0)
    result_low = _coerce_intermediate({
        'bias': 0.0, 'risk': 0.0, 'uwb_scaling': 1.0, 'vio_scaling': 0.5,
    })
    # v3 回退 scaling_min=1.0, vio_scaling=0.5 被裁剪到下限 1.0.
    assert result_low.vio_scaling == pytest.approx(1.0)


def test_coerce_intermediate_rejects_bool_model_output():
    """dict 中 bias/risk/scaling 为 True/False 时抛 TypeError。"""
    with pytest.raises(TypeError, match="must be numeric"):
        _coerce_intermediate({
            'bias': True, 'risk': 0.2, 'uwb_scaling': 1.0, 'vio_scaling': 1.0,
        })
    with pytest.raises(TypeError, match="must be numeric"):
        _coerce_intermediate({
            'bias': 0.0, 'risk': False, 'uwb_scaling': 1.0, 'vio_scaling': 1.0,
        })


def test_coerce_intermediate_rejects_non_finite_model_output():
    """dict 中值为 NaN 或 inf 时抛 ValueError。"""
    with pytest.raises(ValueError, match="must be finite"):
        _coerce_intermediate({
            'bias': float('nan'), 'risk': 0.2, 'uwb_scaling': 1.0, 'vio_scaling': 1.0,
        })
    with pytest.raises(ValueError, match="must be finite"):
        _coerce_intermediate({
            'bias': 0.0, 'risk': float('inf'), 'uwb_scaling': 1.0, 'vio_scaling': 1.0,
        })


def test_coerce_intermediate_rejects_unsupported_output_type():
    """非 dict/非 ModelIntermediate 模型输出（list、None、int）抛 TypeError。"""
    for bad_output in [[1, 2, 3], None, 42]:
        with pytest.raises(TypeError, match="must be a ModelIntermediate or dict"):
            _coerce_intermediate(bad_output)


def test_run_fusion_rejects_none_events():
    with pytest.raises(ValueError, match="events must not be None"):
        run_fusion(None, create_estimator('ekf', _estimator_cfg()))


def test_run_fusion_rejects_string_events():
    with pytest.raises(TypeError, match="not a string or bytes"):
        run_fusion("abc", create_estimator('ekf', _estimator_cfg()))


def test_run_fusion_rejects_mapping_events():
    with pytest.raises(TypeError, match="not a single mapping"):
        run_fusion({'t': 0.0}, create_estimator('ekf', _estimator_cfg()))


def test_run_fusion_rejects_empty_events():
    with pytest.raises(ValueError, match="non-empty"):
        run_fusion([], create_estimator('ekf', _estimator_cfg()))


def test_run_fusion_rejects_none_estimator():
    with pytest.raises(ValueError, match="estimator must not be None"):
        run_fusion(_events(), None)


def test_mechanism_level_ablations_neutralise_target_intermediate_field():
    """机制级消融变体在 fusion_runner 内对 intermediate 对应字段的真结构级中性化验证。

    用 stub model 强制输出非零 bias=0.7 / risk=0.6 / vio_scaling=1.4，
    分别以 four method_name 跑 run_fusion：
      - liquid_ekf_full             : 全字段保留模型原值
      - liquid_ekf_wo_bias_memory   : bias_trace 中性化为 0，risk / vio_scaling 仍保留
      - liquid_ekf_wo_risk_gate     : risk_trace 中性化为 0，bias / vio_scaling 仍保留
      - liquid_ekf_wo_vio_confidence: 仅 VIO 事件 vio_scaling_trace 中性化为 1.0，bias / risk 仍保留

    这条测试是真正的反证：之前用真实（小）训练后 liquid 模型在 mini_seq 上 bias/vio_scaling
    本来就输出 0/1.0，无法区分是"模型未学到"还是"消融中性化"。stub model 强制非平凡值后，
    消融逻辑有义务把对应字段无条件压成中性默认。
    """

    class _StubAblationModel:
        """强制输出非零 bias/risk/vio_scaling，让任何"消融中性化"都能被立刻识别。"""

        def infer_intermediate(self, window_tensor):
            return {
                'bias': 0.7,
                'risk': 0.6,
                'uwb_scaling': 1.2,
                'vio_scaling': 1.4,
            }

    def _run(method_name: str):
        bundle = run_fusion(
            _events(),
            create_estimator('ekf', _estimator_cfg()),
            _StubAblationModel(),
            feature_builder=_feature_builder,
            cfg={'method_name': method_name, 'resolved_method_name': method_name},
        )
        return bundle['diagnostics']

    full_d = _run('liquid_ekf_full')
    wo_bias_d = _run('liquid_ekf_wo_bias_memory')
    wo_risk_d = _run('liquid_ekf_wo_risk_gate')
    wo_vio_d = _run('liquid_ekf_wo_vio_confidence')

    # --- 共同基线 liquid_ekf_full：模型非零输出应原样出现在 trace 上 ---
    # 注意 _events()[0] 是 imu 事件、走 estimator.step 不调 build_measurement_control，
    # 在 IMU 路径 _record_trace_entry 用 ablation_intermediate；但 IMU 路径模型未运行，
    # intermediate 默认 ModelIntermediate(0,0,1,1) → bias_trace[0]==0, risk_trace[0]==0。
    # 索引 1 是 uwb、索引 2 是 vio —— 这两条走紧耦合 buffer flush，trace 用 ablation_intermediate。
    assert full_d['bias_trace'] == [0.0, pytest.approx(0.7), pytest.approx(0.7)]
    assert full_d['risk_trace'] == [0.0, pytest.approx(0.6), pytest.approx(0.6)]
    assert full_d['vio_scaling_trace'] == [1.0, pytest.approx(1.4), pytest.approx(1.4)]

    # --- liquid_ekf_wo_bias_memory：bias 字段被结构级压为 0，risk / vio_scaling 不动 ---
    assert wo_bias_d['bias_trace'] == [0.0, 0.0, 0.0]
    # UWB 事件的 applied_bias 也必须为 0（被消融的 bias 直接进桥接合约）
    assert wo_bias_d['applied_bias_trace'][1] == pytest.approx(0.0)
    # risk 应保留模型原值（不能被误伤）
    assert wo_bias_d['risk_trace'] == [0.0, pytest.approx(0.6), pytest.approx(0.6)]
    # vio_scaling 不应被该消融影响
    assert wo_bias_d['vio_scaling_trace'] == [1.0, pytest.approx(1.4), pytest.approx(1.4)]

    # --- liquid_ekf_wo_risk_gate：risk 字段被结构级压为 0，bias / vio_scaling 不动 ---
    assert wo_risk_d['risk_trace'] == [0.0, 0.0, 0.0]
    # bias 不应被误伤
    assert wo_risk_d['bias_trace'] == [0.0, pytest.approx(0.7), pytest.approx(0.7)]
    # vio_scaling 不应被误伤
    assert wo_risk_d['vio_scaling_trace'] == [1.0, pytest.approx(1.4), pytest.approx(1.4)]

    # --- liquid_ekf_wo_vio_confidence：仅 VIO 事件上 vio_scaling 被结构级压为 1.0 ---
    # 索引 0 (imu) = 1.0 (默认)，索引 1 (uwb) = 1.4 (未触及，模型原值), 索引 2 (vio) = 1.0 (消融)
    assert wo_vio_d['vio_scaling_trace'] == [1.0, pytest.approx(1.4), 1.0]
    # applied_vio_scaling 在 VIO 事件 (索引 2) 应为 1.0；其他模态事件填中性 1.0
    assert wo_vio_d['applied_vio_scaling_trace'] == [1.0, 1.0, 1.0]
    # bias / risk 不应被该消融影响
    assert wo_vio_d['bias_trace'] == [0.0, pytest.approx(0.7), pytest.approx(0.7)]
    assert wo_vio_d['risk_trace'] == [0.0, pytest.approx(0.6), pytest.approx(0.6)]
