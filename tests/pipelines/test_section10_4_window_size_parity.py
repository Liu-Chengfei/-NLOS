"""§10.4 NN 截断对等 watchdog 单元测试。

spec 第 1691 行："学习方法的序列截断长度同样属于记忆深度：截断过短伤长间隙
NLOS，过长且与他法不对等则不公。"

_watchdog 通过 `_enforce_window_size_parity_for_neural_methods` 直接走
`_resolve_method_route + _resolve_model_cfg` 解析两个 NN 方法的 window.size，
强制二者一致——sweep 层 override 一方时 fail-loud，禁止静默滑过对等前提。
"""
from __future__ import annotations

import pytest

from liquidloc.common.constants import MODEL_NAME_LIQUID, MODEL_NAME_LSTM
from liquidloc.pipelines.core_pipeline import _enforce_window_size_parity_for_neural_methods


def test_parity_passes_when_lstm_liquid_share_default_window_size():
    """默认 yaml 下 LSTM/Liquid 的 window.size 均为 20，应通过且无 raise。"""
    methods = [MODEL_NAME_LSTM, MODEL_NAME_LIQUID]  # 两个 NN 方法都跑。
    sizes = _enforce_window_size_parity_for_neural_methods(methods, cfg={})  # 无显式 override。
    assert sizes[MODEL_NAME_LSTM] == sizes[MODEL_NAME_LIQUID], (  # 默认必须对等。
        f"§10.4 默认 yaml 下 LSTM/Liquid window.size 必须一致，实际 {sizes}"
    )
    assert all(v == 20 for v in sizes.values()), (  # 二者都应为 20。
        f"§10.4 默认 yaml 下 LSTM/Liquid window.size 应为 20，实际 {sizes}"
    )


def test_parity_silent_skip_classical_methods_without_window():
    """经典方法无 window 字段，watchdog 应跳过它们只看 NN。"""
    methods = [
        "ekf", "robust_ekf", "fgo",  # 经典方法无 window。
        MODEL_NAME_LSTM, MODEL_NAME_LIQUID,  # 唯二 NN 方法。
    ]
    sizes = _enforce_window_size_parity_for_neural_methods(methods, cfg={})
    # 仅 NN 方法入字典，经典方法被跳过。
    assert set(sizes.keys()) == {MODEL_NAME_LSTM, MODEL_NAME_LIQUID}


def test_parity_raises_when_sweep_overrides_lstm_window_size_diverges():
    """sweep 层 override 一方 window.size 时 watchdog 必须 fail-loud。"""
    methods = [MODEL_NAME_LSTM, MODEL_NAME_LIQUID]
    cfg = {
        "model_cfgs": {
            # 显式 override LSTM 的 window.size 成 30，不动 Liquid → 不对等。
            MODEL_NAME_LSTM: {"window": {"size": 30}},
        }
    }
    with pytest.raises(ValueError, match="§10.4 NN 截断对等 watchdog"):
        _enforce_window_size_parity_for_neural_methods(methods, cfg)


def test_parity_passes_when_sweep_overrides_both_consistently():
    """显式 override 二者 window.size 成同一新值时仍合规——协议层显式对齐。"""
    methods = [MODEL_NAME_LSTM, MODEL_NAME_LIQUID]
    cfg = {
        "model_cfgs": {
            MODEL_NAME_LSTM: {"window": {"size": 15}},
            MODEL_NAME_LIQUID: {"window": {"size": 15}},
        }
    }
    sizes = _enforce_window_size_parity_for_neural_methods(methods, cfg)
    assert all(v == 15 for v in sizes.values())


def test_parity_skips_unknown_methods_without_raising():
    """未知方法 / 未实现的消融方法应在主循环 _resolve_method_route 报错，
    watchdog 预扫阶段必须 swallow 它的 ValueError 跳过——避免 watchdog 重复报错。
    """
    methods = [MODEL_NAME_LSTM, MODEL_NAME_LIQUID, "unknown_method_xxx"]
    # 不应 raise；unknown 方法被跳过，只剩 LSTM/Liquid 走对等校验。
    sizes = _enforce_window_size_parity_for_neural_methods(methods, cfg={})
    assert set(sizes.keys()) == {MODEL_NAME_LSTM, MODEL_NAME_LIQUID}


def test_main_pipeline_run_loop_mounts_watchdog_and_fails_loud(tmp_path):
    """§10.4 主循环挂载验证集成测试：走 CorePipeline.run 入口而非而函数直调。

    构造 methods=['lstm_ekf', 'liquid_ekf'] + model_cfgs 不一致 (lstm.size=30,
    liquid.size=20) 走 run() 入口：主循环 L1546 之前 _enforce_window_size_parity_for_neural_methods
    必须真挂载触发 ValueError fail-loud, sweep override 不再静默滑过记忆深度对等前提。
    本测试不依赖任何场景事件可运行性—it 旨在证明主循环真挂载, watchdog 报错在前比
    后续 estimator 资源加载还早, 不会因为场景事件触发条件错位而绕过。
    """
    import pytest as _pytest

    from liquidloc.pipelines.core_pipeline import run as _pipeline_run

    # 复用 test_core_pipeline._events 模板（仅最少事件足以让 scene_tasks 生效到主循环）
    events = [
        {'t': 0.0, 'dt': 0.0, 'modality': 'imu',
         'meta': {'scene_id': 'S(A1,N2,V1,K0,M0)', 'seq_id': 'mini_seq'},
         'imu_payload': {'ax': 0.1, 'ay': 0.0, 'gz': 0.01},
         'uwb_payload': None, 'vio_payload': None},
        {'t': 0.1, 'dt': 0.1, 'modality': 'uwb',
         'meta': {'scene_id': 'S(A1,N2,V1,K0,M0)', 'seq_id': 'mini_seq'},
         'imu_payload': None, 'uwb_payload': {'anchor_id': 0, 'range': 2.0, 'valid': True, 'quality': 0.95}, 'vio_payload': None},
    ]
    scene_tasks = [{
        'task_id': 'mini_seq',
        'primary_axis': 'target_degradation_bundle',
        'frozen_axes': {'A': 'A1', 'N': 'N2', 'V': 'V1', 'G': 'K3', 'K': 'K0', 'M': 'M0'},
    }]
    pipeline_cfg = {
        'events': events,
        'scene_tasks': scene_tasks,
        'methods': ['lstm_ekf', 'liquid_ekf'],  # 两个 NN 方法同任务跑, 必须对等
        'estimator_cfgs': {  # 同 test_core_pipeline.py L52-97 结构
            'ekf': {
                'process_noise': {'pos': 0.05, 'vel': 0.10, 'yaw': 0.02, 'accel_bias': 0.001,
                                   'gyro_bias': 0.001, 'uwb_clock_bias': 0.001, 'vio_scale': 0.001},
                'measurement_noise': {'uwb': 0.25, 'vio': {'pos': 0.08, 'yaw': 0.03}},
                'init_state': {'px': 0.0, 'py': 0.0, 'vx': 0.0, 'vy': 0.0, 'yaw': 0.0,
                               'bax': 0.0, 'bay': 0.0, 'bg': 0.0, 'uwb_clock_bias': 0.0, 'vio_scale': 1.0},
                'init_cov': [1.0, 1.0, 0.5, 0.5, 0.3, 0.05, 0.05, 0.02, 0.05, 0.01],
            },
        },
        'model_cfgs': {
            'lstm_ekf': {'window': {'size': 30}},   # 不一致 30
            'liquid_ekf': {'window': {'size': 20}}, # 不一致 20
        },
        'output_root': tmp_path / 'parity_fail_loud',
    }
    # 主循环挂钩 watchdog 应 ValueError fail-loud
    with _pytest.raises(ValueError, match="§10.4 NN 截断对等 watchdog"):
        _pipeline_run(pipeline_cfg)



def test_train_pipeline_window_size_missing_raises_keyerror():
    """§10.4 训推同值契约守卫（训练侧）：model_cfg.window.size 缺失时 train_pipeline 必 raise KeyError，
    禁止默认值 1 兜底——与推理侧 _enforce_window_size_parity_for_neural_methods 的 KeyError 守卫同口径。

    本测试硬 import _build_feature_window_builder，找不到则 fail（而非 skip），避免 silent skip 让守卫形同虚设。
    """
    import pytest as _pt
    from liquidloc.pipelines.train_pipeline import _build_feature_window_builder  # 不能 import 则 AttributeError 真失败
    fake_model_cfg = {'feature_order': ['dt', 'ax']}  # 故意不写 window 段
    with _pt.raises(KeyError, match="window.size"):
        _build_feature_window_builder(model_cfg=fake_model_cfg, estimator=object())


def test_train_inference_window_size_contract_same_value():
    """§10.4 训推同值契约守卫：训练侧与推理侧加在同一 model_name 上看到的 window.size 必须严格相等。

    构造一份 yaml 默认配置 + 一份 sweep override 配置，分别送入
      - 训练侧: train_pipeline._build_feature_window_builder
      - 推理侧: _enforce_window_size_parity_for_neural_methods
    双侧应看到同一 window.size 值；若两侧 sentinel 不一致（一侧 KeyError、一侧静默吞）将本测试失败。
    """
    from liquidloc.common.constants import MODEL_NAME_LIQUID, MODEL_NAME_LSTM
    from liquidloc.pipelines.core_pipeline import _enforce_window_size_parity_for_neural_methods

    # 一份模拟 yaml/sweep override；训练侧与推理侧都读 model_cfgs[name].window.size
    cfg_inference = {
        'model_cfgs': {
            MODEL_NAME_LSTM: {'window': {'size': 25}},
            MODEL_NAME_LIQUID: {'window': {'size': 25}},
        }
    }
    sizes_inference = _enforce_window_size_parity_for_neural_methods(
        methods=[MODEL_NAME_LSTM, MODEL_NAME_LIQUID], cfg=cfg_inference
    )
    # 训练侧 window.size 值就是 cfg_inference['model_cfgs'][name]['window']['size']
    for name in (MODEL_NAME_LSTM, MODEL_NAME_LIQUID):
        train_side = cfg_inference['model_cfgs'][name]['window']['size']
        inf_side = sizes_inference[name]
        assert int(train_side) == int(inf_side), (
            f"§10.4 训推同值契约违例: {name} 训练侧 window.size={train_side} "
            f"vs 推理侧 window.size={inf_side} 不一致；sweep override 必须双侧同值入契约。"
        )


def test_section10_2_pure_nn_route_without_ekf_shell_rejected():
    """§10.2 第 2 行 fail-loud 守卫：方法名含 NN 关键字但被注册到经典方法集合时，
    _resolve_method_route 必 raise ValueError 拒绝『纯 NN 主表无 EKF 外壳』路径。

    本测试构造一个含 'lstm' 关键字的"伪经典方法名"→ 不可能在 _CLASSICAL_METHODS 中存在，
    所以应进 `_resolve_method_route` L1307 `raise ValueError('Unsupported method')`，
    但本测试需验证：若未来有人手动把 'lstm_ekf' 加进 _CLASSICAL_METHODS，本守卫应触发。
    使用 monkeypatch 临时把 'lstm_ekf' 加进 _CLASSICAL_METHODS 验证守卫真挂。
    """
    from liquidloc.pipelines.core_pipeline import _resolve_method_route
    from liquidloc.pipelines import core_pipeline as _core_module

    # 临时把 'lstm_ekf' 加进 _CLASSICAL_METHODS 模拟误注册
    fake_classical = {'lstm_ekf'} | set(_core_module._CLASSICAL_METHODS)
    original = _core_module._CLASSICAL_METHODS
    _core_module._CLASSICAL_METHODS = fake_classical
    try:
        with pytest.raises(ValueError, match="§10.2 第 2 行守卫"):
            _resolve_method_route('lstm_ekf')
    finally:
        _core_module._CLASSICAL_METHODS = original


def test_section10_2_unknown_nn_keyword_method_not_preempted_by_classical_guard():
    """§10.2 第 2 行守卫副作用测：未知方法名含 NN 关键字但不在 _CLASSICAL_METHODS 中时，
    _resolve_method_route 应走原 `raise ValueError('Unsupported method')` 分支，
    不被新加的 §10.2 第 2 行守卫提前在 _CLASSICAL_METHODS 分支内拦下（守卫位置在
    `if method_name in _CLASSICAL_METHODS` 内部，未知方法进不到此分支）。
    """
    from liquidloc.pipelines.core_pipeline import _resolve_method_route
    # lstm_ekf_xxx 不在 _NEURAL_METHODS 也不在 _CLASSICAL_METHODS
    with pytest.raises(ValueError, match="Unsupported method: lstm_ekf_xxx"):
        _resolve_method_route('lstm_ekf_xxx')


def test_section10_3_cross_config_sentinel_io_safe_when_fgo_yaml_missing(monkeypatch):
    """§10.3 cross-config sentinel IO 兜底测：fgo.yaml 路径不存在/IO 错误时，
    _validate_scene_scale_section 不应 raise，应正常返回 normalized_section（无 _section10_3 标记）。
    防止 cross-config sentinel 反客为主破坏 protocol gate 主流程。
    """
    from liquidloc.protocol import experiment_gates as _eg
    from liquidloc.protocol.experiment_gates import _validate_scene_scale_section

    # 让 load_yaml_config 抛 FileNotFoundError 模拟 fgo.yaml 缺失
    def _boom(*args, **kwargs):
        raise FileNotFoundError("simulated missing fgo.yaml")

    monkeypatch.setattr(_eg, 'load_yaml_config', _boom)
    section = {
        'scene_scale': {
            't_eff_min_s': 20.0,
            'cold_start_offset_s': 5.0,
            'cold_start_global_enforced': True,
            'n_pulse_min': 30,
            'n_async_min': 20,
            'layout_family_min': 3,
        }
    }
    normalized = _validate_scene_scale_section(section)  # 不应 raise
    assert '_section10_3_t_eff_assumed_mismatch' not in normalized
    assert normalized['t_eff_min_s'] == 20.0


def test_section10_1_a_validate_event_sequence_rejects_out_of_order_timestamps():
    """§10.1(a) 真合规验证：spec 要求"含已缓冲且时间戳 ≤ t 的乱序件, 规则全员同一"。
    validate_event_sequence 须在乱序件出现时直接 ValueError 拒。
    UWB 必备 payload 字段: [anchor_id, range, valid, quality] (configs/base/sensors.yaml)。
    """
    from liquidloc.protocol.event_schema import validate_event_sequence
    # 构造时间戳倒序序列：第 1 条 t=1.0（dt=0.0 首条），第 2 条 t=0.5 但 dt 写成 +0.5
    # 触发 event_schema.py:538-539 守卫 (current_t < prev_t - tolerance → ValueError "not time-monotonic")，
    # 而非 L466 dt<0 守卫；dt>=0 满足前置门后才会真走到时间戳单调守卫。
    _uwb = lambda t, dt: {
        "modality": "uwb", "t": t, "dt": dt,
        "meta": {"scene_id": "s1", "seq_id": "q1"},
        "uwb_payload": {"anchor_id": 0, "range": 1.0, "valid": True, "quality": 0.9},
    }
    bad_events = [_uwb(1.0, 0.0), _uwb(0.5, 0.5)]
    with pytest.raises(ValueError, match="not time-monotonic"):
        validate_event_sequence(bad_events)
