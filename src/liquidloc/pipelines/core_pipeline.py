"""核心实验编排流水线。

这个模块负责把场景任务、方法路由、模型与估计器、场景协议和各种场景修饰层串起来，
生成最终的预测、统计、审计和导出产物。它是整个实验链里最核心的调度层。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence  # 用来判断 bundle、context、配置、事件序列是否是映射或可迭代对象；主代理二次审查补 Iterable：L270 _clone_events(events: Iterable[Any]) 注解此前缺导入，与 Round 1 StateEstimate、Round 3 MODALITY_UWB 同款"子代理声称加 import 未落地"问题，from __future__ import annotations 让注解在运行期延迟为字符串故未报 NameError，但 typing.get_type_hints 解析时会失败，违反一致性。Sequence 用于 _build_section8_cfg_payload 的 list 判断，与 Mapping 同源。
from copy import deepcopy  # 用于复制嵌套结构，避免共享引用。
import hashlib  # 用于确定性哈希，保证锚点重映射跨进程可复现。
import time  # 用于统计模型和估计器运行时延。
from pathlib import Path  # 用于解析输出目录和配置路径。
from typing import Any  # 用于给复杂字典和动态对象做宽松标注。

import torch  # 用于 no_grad 上下文管理器，推理阶段禁用梯度计算。
from liquidloc.common.config_utils import find_project_root, load_yaml_config  # 读取 YAML 模型配置。
from liquidloc.common.constants import (  # D9 单源常量：神经方法名、经典估计器名、模态名，禁止本地重复字面量。
    ESTIMATOR_NAME_EKF,
    ESTIMATOR_NAME_FGO,
    ESTIMATOR_NAME_ROBUST_EKF,
    MODALITY_UWB,
    MODEL_NAME_LIQUID,
    MODEL_NAME_LSTM,
    MODEL_NAME_TRANSFORMER,
)
from liquidloc.common.paths import resolve_output_root  # 解析项目标准目录和统一输出根目录。
from liquidloc.common.types import ModelIntermediate  # 模型中间输出结构。
from liquidloc.common.types import StateEstimate  # 估计器状态估计摘要，用于 _TimedEstimatorProxy.step 返回类型注解。
from liquidloc.common.types import StageResult  # 流水线阶段结果结构。
from liquidloc.common.validation import coerce_finite_scalar, is_integer, is_numeric, is_real, is_string_like, validate_path_component  # 统一判断数值类型和路径校验。
from liquidloc.factories.estimator_factory import create_estimator  # 构建估计器实例。
from liquidloc.factories.model_factory import create_model  # 构建模型实例。
from liquidloc.fusion.fusion_runner import run_fusion  # 执行融合和估计更新。
from liquidloc.interfaces.pipeline_api import PipelineAPI, normalize_pipeline_cfg  # 流水线接口基类与统一配置规整 helper。
from liquidloc.protocol.scene_axis_protocol import attach_scene_parameters  # 把轴值补成场景参数。
from liquidloc.protocol.scene_axis_protocol import get_nominal_levels  # 获取各轴正常等级名，避免硬编码默认场景。
from liquidloc.protocol.scene_axis_protocol import load_scene_axis_protocol  # 读取场景轴协议。
from liquidloc.protocol.scene_axis_protocol import SceneParameters  # 场景参数数据结构，用于 isinstance 检查。
from liquidloc.protocol.scene_axis_protocol import SCENE_AXES  # 冻结的场景轴名元组，避免本地硬编码漂移。
from liquidloc.protocol.scene_schema import SceneSpec, encode_scene  # 场景规格对象与编码，用于构造默认场景编码。
# §8.3 单轨片段覆盖门禁：实验级断言全部 7 项片段类型在 sweep 中被覆盖。
# 局部导入避免顶层循环依赖；按需在 sample_scenes 之后调用。
def _assert_experiment_segment_coverage(scene_tasks, experiment_cfg):
    from liquidloc.protocol.experiment_gates import check_experiment_segment_coverage
    noise_spec = (experiment_cfg or {}).get('noise_spec') or {}
    return check_experiment_segment_coverage(scene_tasks, noise_spec, raise_on_violation=True)

def _run_section8_deep_audit(scene_tasks, cfg):
    """Section 8 deep audit gate set (opt-in; default ON since §8 hardening).

    Triggered by experiment_cfg['enable_section8_deep_audit']=True (default True
    after §8 hardening; can be disabled by setting it explicitly to False).

    Coverage (15 gates, all wired in this function):
      1. §8.4 cold-start × K3 underdetermined geometry conjunction.
         (assert_cold_start_x_underdetermined_geometry; fired on scene_tasks)
      2. §8.1 H + 细节 R-D-C 锚点源一致性 + 移动锚点真值差分:
         (assert_moving_anchor_truth_equality; fired on cfg['method_moving_anchor_truth'])
         ensures moving anchor truth trajectories (per-sample positions) are identical
         across methods; static layouts (no moving anchors) auto-pass.
      3. §8.1 G 锚点切换规则可知:
         (assert_anchor_switch_ruleknown; fired on cfg['anchor_layout'].anchor_switch)
         requires anchor-switch events carry explicit switch_times + reason fields.
      3a. §8.1 H + 细节 R-D-C anchor uniform source:
         (assert_anchor_uniform_source; fired on cfg['method_anchor_layouts'])
         ensures all methods use the same anchor_layout AND Na ∈ {3,4,5}
         (no Na≥8 high-redundancy GDOP masquerading as underdetermined pressure).
      4. §8.1 细节 R-D-F anchor switch anti strong-continuity smoothing:
         (assert_anchor_switch_anti_smoothing; fired on cfg['anchor_switch_events'])
         checks that raw residual pulses exist near anchor-switch events so that
         front-end strong-continuity smoothing is not hiding switch transients.
      5. 细节 R-D-D cross-method plane/Z constraint equality:
         (assert_cross_method_plane_z_equality; fired on cfg['method_meta'])
         ensures plane_constraint / z_constraint / motion_constraint values are
         uniform across all methods (no single-method advantage via extra factors).
      6. §8.2 R8.2-D IMU bias observability:
         (assert_imu_bias_observability; fired on cfg['trajectory_envelope'])
         requires accel excitation (a95 ≥ 0.5 m/s²) + stop-go (≥ 3% near-zero speed)
         + yaw excitation (turn_total ≥ 2π or ≥ 3 significant turns) so IMU bias
         is distinguishable from position/velocity (§5.3).
      7. §8.1 L1364 病态观测占比 ≥ 10%:
         (assert_underdetermined_observation_ratio; fired on cfg['weak_geometry_mask'])
         ensures weak-geometry (病态 GDOP) periods cover ≥ 10% of total samples.
      8. §8.3 单轨片段覆盖 7 类:
         (check_experiment_segment_coverage; fired on scene_tasks + cfg['noise_spec'])
         ensures all 7 segment types (los_decent_geometry, pulsed_nlos, all_anchor_nlos,
         significant_async, vio_degraded_or_interrupted, short_term_imu_dominated,
         cold_start_transition) are covered by at least one task in the sweep.
      9. §8.2.0 政策5 seed_required:
         (assert_seed_required; fired on cfg['trajectory_seeds'])
         ensures every sequence has a finite numeric seed (no 'manual single-track'
         or 'non-reproducible teleop' as sole sweep evidence per spec L1388).
     10. §8.2.0 政策6 seed_decoupling:
         (assert_seed_decoupling; fired on cfg['trajectory_seed','nlos_seed','async_seed'])
         ensures trajectory/NLOS/async seeds are NOT fully coupled (all equal),
         so ranking contributions can be attributed to motion vs NLOS vs async per L1389.
     11. §8.2 强周期主导门禁:
         (periodicity_max_ratio check inside compute_trajectory_envelope)
         ensures dominant_periodicity_ratio < 0.7 (no strong-period single-track).
     12. §8.1 L1361 三维测距另声明:
         (assert_anchor_3d_declaration; fired on cfg['method_anchor_layouts'] first layout)
         ensures 3D layouts declare min_anchor_count_3d (≥4) and vertical_distribution.
     13. §8.2.0 政策1 多样本:
         (assert_trajectory_collection_multi_sample; fired on scene_tasks)
         ensures sweep contains ≥2 tasks (no single-track dead script defining total order).
     14. §8.2.0 政策2 随机源可种子化:
         (assert_trajectory_generator_pol_2; fired on cfg['generator_metadata'])
         ensures trajectory generator covers ≥1 of path_shape/turn_phase/stop_time/speed_envelope
         OR uses a sufficiently large fixed trajectory library (≥30).
     15. §8.2.0 政策4 允许生成族:
         (assert_trajectory_generator_pol_4; fired on cfg['generator_metadata'])
         ensures trajectory generator family ∈ {seeded_random_walk, parameterized_spline,
         trajectory_lib, combined, ...} per spec L1384-1387.

    Failure mode: raise by default; cfg['section8_deep_audit_soft']=True for log-only.
    Required cfg keys (any missing → that gate silently skips):
      'weak_geometry_mask'        → list[int] of length n_samples
      'method_moving_anchor_truth'→ dict[method_name -> list[{'t','anchor_id','px','py','pz?'}, ...]]
      'method_meta'               → dict[method_name -> {'plane_constraint','z_constraint','motion_constraint'}]
      'anchor_switch_events'      → list[{'t','anchor_id_dropped','anchor_id_added'}, ...]
      'raw_residuals'             → list[{'t','anchor_id','r'}, ...]
      'trajectory_envelope'       → output of compute_trajectory_envelope
    """
    from liquidloc.protocol.experiment_gates import (
        assert_cold_start_x_underdetermined_geometry,
        assert_cross_method_plane_z_equality,
        assert_anchor_switch_anti_smoothing,
        assert_anchor_switch_ruleknown,
        assert_anchor_uniform_source,
        assert_imu_bias_observability,
        assert_underdetermined_observation_ratio,
        assert_moving_anchor_truth_equality,
        assert_seed_required,
        assert_seed_decoupling,
        assert_anchor_3d_declaration,
        assert_trajectory_collection_multi_sample,
        assert_trajectory_generator_pol_2,
        assert_trajectory_generator_pol_4,
    )
    soft_mode = bool((cfg or {}).get('section8_deep_audit_soft', False))
    raise_kw = not soft_mode

    if scene_tasks:
        csr = assert_cold_start_x_underdetermined_geometry(
            list(scene_tasks), raise_on_violation=raise_kw)
        _log_section8_audit('cold_start_x_underdetermined_geometry', csr, soft_mode)

    method_meta = (cfg or {}).get('method_meta') or {}
    if method_meta and len(method_meta) >= 2:
        czr = assert_cross_method_plane_z_equality(method_meta, raise_on_violation=raise_kw)
        _log_section8_audit('cross_method_plane_z_equality', czr, soft_mode)

    switch_events = (cfg or {}).get('anchor_switch_events') or []
    raw_residuals = (cfg or {}).get('raw_residuals') or []
    if switch_events:
        ar = assert_anchor_switch_anti_smoothing(
            switch_events, raw_residuals, raise_on_violation=raise_kw)
        _log_section8_audit('anchor_switch_anti_smoothing', ar, soft_mode)

    envelope = (cfg or {}).get('trajectory_envelope') or {}
    if envelope:
        # §8.2-A 周期主导硬门禁：从 envelope.dominant_periodicity_ratio 显式触发，
        # 防止 compute_trajectory_envelope 的内部 raise 被某条 try/except 吞掉而 silent skip。
        # 详见 §8.2 原文 "功率以中频宽带为主；禁止强周期主导（周期核与宽网捡分 → 伤 5、6）"。
        per_max = cfg.get('periodicity_max_ratio', 0.7)
        per_ratio = envelope.get('dominant_periodicity_ratio')
        if isinstance(per_ratio, (int, float)) and per_ratio > per_max:
            per_report = {
                'periodicity_not_dominant': False,
                'dominant_periodicity_ratio': float(per_ratio),
                'periodicity_max_ratio': float(per_max),
            }
            _log_section8_audit('periodicity_not_dominant', per_report, soft_mode)
            if raise_kw:
                raise ValueError(
                    f'§8.2-A periodicity_not_dominant violation: ratio={per_ratio:.3f} > '
                    f'max={per_max}'
                )
        ior = assert_imu_bias_observability(envelope, raise_on_violation=raise_kw)
        _log_section8_audit('imu_bias_observability', ior, soft_mode)

    # §8.1 L1364 病态观测占比门禁（差 GDOP 时段占比 ≥ 10%）。
    weak_geometry_mask = (cfg or {}).get('weak_geometry_mask') or []
    if weak_geometry_mask:
        uor = assert_underdetermined_observation_ratio(
            weak_geometry_mask, min_ratio=0.10, raise_on_violation=raise_kw)
        _log_section8_audit('underdetermined_observation_ratio', uor, soft_mode)

    # §8.3 单轨片段覆盖：7 类片段（视距、脉冲 NLOS、全锚 NLOS、异步、VIO 退化、
    # 短时 IMU 主导、差冷启动过渡）须在 sweep 中至少被一条轨迹覆盖。
    # 需 cfg['noise_spec'] 含 vio_outage_segments / cold_start_offset_s 才能完整审计。
    _noise_spec = (cfg or {}).get('noise_spec') or {}
    if scene_tasks:
        from liquidloc.protocol.experiment_gates import check_experiment_segment_coverage
        secr = check_experiment_segment_coverage(list(scene_tasks), _noise_spec, raise_on_violation=raise_kw)
        _log_section8_audit('experiment_segment_coverage', secr, soft_mode)

    # 细节 R-D-C 移动锚点真值差分门禁（移动锚点存在时轨迹全员同一）。
    method_moving_anchor_truth = (cfg or {}).get('method_moving_anchor_truth') or {}
    if method_moving_anchor_truth and len(method_moving_anchor_truth) >= 2:
        mar = assert_moving_anchor_truth_equality(
            method_moving_anchor_truth, raise_on_violation=raise_kw)
        _log_section8_audit('moving_anchor_truth_equality', mar, soft_mode)

    # §8.1 H + 细节 R-D-C 锚点源一致性：所有方法必须使用同一 anchor_layout，
    # 且 Na ∈ {3,4,5}（禁 Na≥8 高冗余优 GDOP）。
    # 需 cfg['method_anchor_layouts'] 非空；由 _build_section8_cfg_payload 构造。
    method_anchor_layouts = (cfg or {}).get('method_anchor_layouts') or {}
    if method_anchor_layouts:
        aur = assert_anchor_uniform_source(method_anchor_layouts, raise_on_violation=raise_kw)
        _log_section8_audit('anchor_uniform_source', aur, soft_mode)

    # §8.1 L1361 三维测距另声明门禁：3D 布局须显式声明 min_anchor_count_3d (≥4)
    # 与 vertical_distribution；2D 布局自动通过。
    if method_anchor_layouts:
        ad = assert_anchor_3d_declaration(
            next(iter(method_anchor_layouts.values())),
            raise_on_violation=raise_kw)
        _log_section8_audit('anchor_3d_declaration', ad, soft_mode)

    # §8.2.0 政策1 多样本门禁：禁单条死脚本定全序；sweep 须 ≥2 条任务。
    # 由 scene_tasks 直接驱动，无需额外 payload。
    mss = assert_trajectory_collection_multi_sample(scene_tasks, raise_on_violation=raise_kw)
    _log_section8_audit('policy1_multi_sample', mss, soft_mode)

    # §8.2.0 政策2 随机源可种子化门禁：轨迹生成器须对路径/转向/停走/速度包络之一做种子化变异。
    generator_metadata = (cfg or {}).get('generator_metadata')
    if generator_metadata is not None:
        p2r = assert_trajectory_generator_pol_2(generator_metadata, raise_on_violation=raise_kw)
        _log_section8_audit('policy2_seedable', p2r, soft_mode)

    # §8.2.0 政策4 允许生成族门禁：generator_family 须落入允许族列表。
    if generator_metadata is not None:
        p4r = assert_trajectory_generator_pol_4(generator_metadata, raise_on_violation=raise_kw)
        _log_section8_audit('policy4_generator_family', p4r, soft_mode)

    # §8.1 G 锚点切换规则可知：若存在锚点切换，切换时刻必须规则可知。
    # 需 cfg['anchor_layout'] 含显式 anchor_switch 字段；由 _build_section8_cfg_payload 透传。
    anchor_layout = (cfg or {}).get('anchor_layout') or {}
    if anchor_layout:
        ask = assert_anchor_switch_ruleknown(anchor_layout, raise_on_violation=raise_kw)
        _log_section8_audit('anchor_switch_ruleknown', ask, soft_mode)

    # §8.2.0 政策5 展位门禁：禁止"无种子人工单轨"或"不可复现遥操作未落盘"。
    # 需 cfg['trajectory_seeds'] 为 dict[seq_id -> seed] 或 list[seed]；
    # 显式 None/空表示用户声明"无种子" → 仍触发门禁让 raise 表态，符合规范 L1388。
    if 'trajectory_seeds' in cfg:
        srr = assert_seed_required(cfg.get('trajectory_seeds'), raise_on_violation=raise_kw)
        _log_section8_audit('seed_required', srr, soft_mode)

    # §8.2.0 政策6 展位门禁：轨迹/NLOS/异步种子宜解耦（便于诊断名次来源）。
    # 需 cfg 提供三个独立种子；若任一缺失则该门禁静默跳过（部分审计）。
    t_seed = (cfg or {}).get('trajectory_seed')
    n_seed = (cfg or {}).get('nlos_seed')
    a_seed = (cfg or {}).get('async_seed')
    if t_seed is not None and n_seed is not None and a_seed is not None:
        sdr = assert_seed_decoupling(t_seed, n_seed, a_seed, raise_on_violation=raise_kw)
        _log_section8_audit('seed_decoupling', sdr, soft_mode)


def _log_section8_audit(gate_name, report, soft_mode):
    """Unified audit log: raise in hard mode; logger.warning in soft mode.

    §8 偷懒修补（soft mode 误判）：原 ok 判定优先取 uniform/sufficient/covered
    等单一字段，导致 gate 报告了 reasons（违规）但 uniform=True 时仍判 ok=True，
    soft mode 下显示 PASS 掩盖违规。本修复以 report['passed'] 或 'violated'
    字段为权威判据；若 gate 报告含非空 reasons 列表也视为未通过。
    """
    import logging
    log = logging.getLogger('liquidloc.section8_audit')
    # 权威判据：优先用 gate 报告的 passed/violated 字段；
    # 其次用 reasons 非空判定违规；最后 fallback 到原逻辑。
    passed = report.get('passed')
    violated = report.get('violated')
    reasons = report.get('reasons')
    if passed is not None:
        ok = bool(passed)
    elif violated is not None:
        ok = not bool(violated)
    elif isinstance(reasons, list) and reasons:
        ok = False
    else:
        ok = bool(report.get('observable',
                report.get('anti_smoothed',
                report.get('uniform',
                report.get('sufficient',
                report.get('covered', True))))))
    if soft_mode and not ok:
        log.warning('[section8_audit] %s VIOLATION (soft mode): %s', gate_name, report)
    else:
        log.info('[section8_audit] %s %s: %s',
                 gate_name, 'PASS' if ok else 'FAIL', report)


def _build_section8_cfg_payload(scene_tasks, cfg, experiment_cfg):
    """Construct the synthetic cfg fields the deep audit needs from real pipeline data.

    Without this, _run_section8_deep_audit would silently skip its 6 cfg-keyed gates
    (the keys are never populated by the pipeline). This helper builds them from:
      - trajectory envelope: computed from the FIRST task's gt_rows (representative
        trajectory; envelope metrics are seed-stable across tasks of one sweep).
      - weak_geometry_mask: built by classifying each sample's GDOP heuristic from
        G-axis level (K3 = all weak; K0/K1 = weak ratio 0); conservative rule that
        catches §8.1 L1364 requirement at protocol-described granularity.
      - method_meta: read per-method from cfg['method_meta'] if user provided it, else
        defaults to all-methods-uniform "plane_constraint=enforced, z_constraint=fixed,
        motion_constraint=none" (the protocol-blessed baseline).
      - method_moving_anchor_truth: built from each method's anchor_layout if present in
        cfg['method_anchor_layouts']; static layout (a None/empty or static layout)
        yields empty per-method list, which the gate auto-passes.
      - anchor_switch_events: read from scene_protocol_cfg['anchor_switch_events'] if
        explicitly provided; default = no switches.

    Returns a fresh dict (does NOT mutate cfg) so callers can choose to merge.
    """
    payload: dict[str, Any] = {}

    # Compute trajectory envelope from the first task's resolved ground truth.
    target_task = scene_tasks[0] if scene_tasks else None
    gt_rows = None
    if target_task is not None:
        # Try direct injection first (used in tests and synthetic cfg), then fall back to resolver.
        gt_rows = target_task.get('gt_rows') or _resolve_ground_truth_rows_for_task(target_task, cfg)
    # §8.2 envelope silent-skip 偷懒保护：原实现 `if gt_rows:` 仅在能解析到 GT 时
    # 调用 compute_trajectory_envelope。当 scene_tasks 非空但 _resolve_ground_truth_rows_for_task
    # 全部回退路径都 None（cfg 既无 ground_truth_by_seq_id 也无 by_scene_id 也无 ground_truth_root
    # 或对应文件不存在）时，§8.2 envelope 全套硬门禁（周期主导 / a95 / v95 / near_zero /
    # turn / path_length / l_xy / t_eff）整段静默跳过——主表 audit 路径下这是真偷懒。
    # 修复：soft mode 保留原 silent-skip（log-only mode 不应中断流程）；
    # hard mode（soft_mode=False）下若 target_task 存在但 gt_rows 解析失败 → 显式 raise，
    # 让"声称主表 audit 却没提供 GT"的口径显形，而非 silent pass。
    soft_mode_env = bool((cfg or {}).get('section8_deep_audit_soft', False))
    if (
        gt_rows is None
        and target_task is not None
        and not soft_mode_env
    ):
        raise ValueError(
            "§8.2 envelope hard gate cannot run: target_task has no resolvable gt_rows "
            f"(seq_id={target_task.get('seq_id')!r}, scene_id={target_task.get('scene_id')!r}). "
            "无法计算 §8.2 envelope 全套硬门禁（周期主导 / a95 / v95 / near_zero / turn / "
            "path_length / l_xy / t_eff）。主表 audit 需 cfg 提供 ground_truth_by_seq_id / "
            "ground_truth_by_scene_id / ground_truth_root 之一且能解析到 GT 行；或在 task"
            "['gt_rows'] 直接注入。若需 §8.2 envelope 该序列跳过，请显式设 "
            "cfg['section8_deep_audit_soft']=True 进入 soft 模式。"
        )
    if gt_rows:
        # compute_trajectory_envelope itself enforces ALL §8.2 envelope hard gates
        # (incl. periodicity_not_dominant) by raising on `checks["passed"] == False`.
        # We do NOT swallow the raise: a sweep whose first task's trajectory violates
        # §8.2 periodicity / a95 / v95 / near_zero / turn / path_length must abort the
        # audit BEFORE _run_section8_deep_audit runs (otherwise the periodicity gate
        # would silently pass via "no envelope stored → skipped" path).
        try:
            from liquidloc.scenarios.geometry_motion_envelope import compute_trajectory_envelope
            payload['trajectory_envelope'] = compute_trajectory_envelope(gt_rows)
            # Sanity: verify the computed envelope reports passed=True (or no checks key).
            env_report = payload['trajectory_envelope']
            if isinstance(env_report, Mapping) and 'passed' in env_report and not bool(env_report['passed']):
                # Reconstruct the failure detail — this should not happen because
                # compute_trajectory_envelope itself raises on failure, but defensive.
                failed_checks = {k: v for k, v in env_report.get('checks', {}).items() if not v}
                raise ValueError(
                    '§8.2 envelope hard gate failed (silent-skip guard): '
                    f'failed_checks={failed_checks}'
                )
        except ValueError:
            raise  # propagate envelope violations
        except Exception:
            # 非 ValueError 异常属计算 bug；原实现吞掉并把 trajectory_envelope 设 None，
            # 导致 dispatcher `if envelope:` 静默 skip——掩盖 envelope 计算真 bug。
            # 现改：重新 raise（附带上下文），让真 bug 显形而非 silent-pass。
            import logging as _logging
            _logging.getLogger(__name__).exception(
                "compute_trajectory_envelope raised non-ValueError; "
                "cannot silently mask — propagating to caller."
            )
            raise

    # weak_geometry_mask: §8.1 L1364 病态观测占比门禁数据源。
    # 早期实现使用 K 轴 proxy 派生（K3→100%、K1→5%、K0→0%），随机分布；
    # 这种代理无法区分 K 轴内部 GDOP 时序差异，与"病态观测达到足够占比"严格语义不 1:1 对应。
    # 现改造为：使用每任务 geometry_report.gdop_above_floor_ratio（0 or 1）按样本量加权聚合；
    # 这是 build_anchor_layout 在 §4.2.2 V1 修复中显式产出的协议级 0/1 标签，
    # 直接反映每个 K 轴级是否落入"差 GDOP"族（gdop_value ≥ floor → 1）。
    # 注意：scene_tasks 在 audit 时不含 geometry_report（该字段在 _apply_scene_task 阶段才写入
    # scenario_reports，不回写到原始 task dict），因此 fallback 至 K 轴级映射（五轴档位协议 G 已并入 K）：
    # K3 → 1.0（差 GDOP 族），K1 → 0.5（临界），K0 → 0.0（优几何）。
    weak_mask: list[int] = []
    if gt_rows and scene_tasks:
        n_total = len(gt_rows)
        per_task_samples = max(1, n_total // max(1, len(scene_tasks)))
        for task in scene_tasks:
            task_axes = (task or {}).get('axes') or {}
            k_level = str(task_axes.get('K', 'K0'))  # 五轴档位协议：G 已并入 K
            task_geom_report = task.get('geometry_report') or {}
            if isinstance(task_geom_report, Mapping) and 'gdop_above_floor_ratio' in task_geom_report:
                gdop_above_floor = float(task_geom_report.get('gdop_above_floor_ratio', 0.0))
            else:
                # Fallback: K 轴级映射（K3→1.0 差 GDOP，K1→0.5 临界，K0→0.0 优几何；G 已并入 K）。
                gdop_above_floor = 1.0 if k_level == 'K3' else (0.5 if k_level == 'K1' else 0.0)
            for _ in range(per_task_samples):
                weak_mask.append(1 if gdop_above_floor >= 1.0 else 0)
        while len(weak_mask) < n_total:
            weak_mask.append(weak_mask[-1] if weak_mask else 0)
        weak_mask = weak_mask[:n_total]
        payload['weak_geometry_mask'] = weak_mask

    # method_meta: prefer cfg-provided; else default all methods to protocol baseline.
    methods = (experiment_cfg or {}).get('methods') or cfg.get('methods') or []
    if isinstance(methods, str):
        methods = [m.strip() for m in methods.split(',') if m.strip()]
    user_method_meta = (cfg or {}).get('method_meta') or {}
    if user_method_meta:
        payload['method_meta'] = user_method_meta
    elif methods:
        # Protocol-blessed baseline: all methods use enforced plane + fixed z + no motion constraint.
        # (§8.1 R-D-D requires uniformity; using the same default for all is the protocol baseline.)
        baseline_meta = {
            'plane_constraint': 'enforced',
            'z_constraint': 'fixed',
            'motion_constraint': 'none',
        }
        payload['method_meta'] = {m: dict(baseline_meta) for m in methods}

    # method_moving_anchor_truth: build per-method from cfg['method_anchor_layouts'] if given.
    user_method_layouts = (cfg or {}).get('method_anchor_layouts') or {}
    if user_method_layouts and methods:
        moving_truth: dict[str, list[dict[str, Any]]] = {}
        for m in methods:
            layout = user_method_layouts.get(m)
            if not layout:
                moving_truth[m] = []
                continue
            anchors = layout.get('anchors') if isinstance(layout, Mapping) else None
            if not anchors:
                moving_truth[m] = []
                continue
            # Detect moving anchors: any anchor with non-empty 'trajectory' field.
            traj_entries: list[dict[str, Any]] = []
            for anc in anchors:
                if not isinstance(anc, Mapping):
                    continue
                anc_traj = anc.get('trajectory') or []
                if not anc_traj:
                    continue
                # Convert per-sample {t, px, py, pz?} into the gate's expected schema.
                for entry in anc_traj:
                    if not isinstance(entry, Mapping):
                        continue
                    traj_entries.append({
                        't': float(entry.get('t', 0.0)),
                        'anchor_id': str(anc.get('anchor_id', anc.get('id', ''))),
                        'px': float(entry.get('px', 0.0)),
                        'py': float(entry.get('py', 0.0)),
                        'pz': float(entry.get('pz', 0.0) or 0.0),
                    })
            moving_truth[m] = traj_entries
        payload['method_moving_anchor_truth'] = moving_truth

    # method_anchor_layouts: derive from task G/K axes using build_anchor_layout.
    # All methods share the same layout (protocol baseline); gate checks equality
    # (uniformity) as well as Na ∈ {3,4,5} (no high-redundancy Na≥8).
    # anchor_layout (singular) is also set here for assert_anchor_switch_ruleknown.
    _anchor_layout = None
    _anchor_layouts: dict[str, dict[str, Any]] = {}
    if target_task is not None and experiment_cfg is not None:
        ax = target_task.get('axes') or {}
        # 五轴档位协议 K 轴锚数全档固定 4（与 K 档位无关，K0/K1/K3 均为 4 锚）。
        anchor_count = 4  # 协议硬约束：锚数全档固定 4
        k_level = str(ax.get('K', 'K3'))  # K 档位仅用于选几何条件（geom_condition）
        # Build layout matching the pipeline's own _materialize_scene_task path.
        try:
            from liquidloc.scenarios.geometry_levels import build_anchor_layout
            from liquidloc.protocol.scene_axis_protocol import load_scene_axis_protocol
            _protocol_cfg = load_scene_axis_protocol()
            _k_cfg = _protocol_cfg.get('axes', {}).get('K', {})
            _anchor_layout, _ = build_anchor_layout(anchor_count, k_level, _k_cfg)
        except ValueError:
            # build_anchor_layout 用 ValueError 表达"G/K 级不存在或参数越界"
            # ——这是实验 cfg 不含 §8 完整 G/K 坐标的合法信号（如非 §8 风格的 pipeline test），
            # silent-skip 该门禁可接受：§8 审计是 best-effort，不强制全部 cfg 必须提供 §8 维度。
            _anchor_layout = None
        except Exception:
            # 非 ValueError 属 build_anchor_layout 内部 bug；原实现吞掉并把
            # _anchor_layout 设 None，导致 dispatcher `if method_anchor_layouts:` 与
            # `if anchor_layout:` 静默 skip——掩盖锚点布局真 bug。现改重新 raise，
            # 让真 bug 显形而非 silent-pass；§8.1 偷懒路径若存在则应在更高层显形。
            import logging as _logging
            _logging.getLogger(__name__).exception(
                "build_anchor_layout raised non-ValueError; "
                "cannot silently mask — propagating to caller."
            )
            raise
    if _anchor_layout is not None and methods:
        # Static layout (no moving anchors in this default build).
        _layout_payload: dict[str, Any] = dict(_anchor_layout) if isinstance(_anchor_layout, Mapping) else {}
        for m in methods:
            _anchor_layouts[m] = _layout_payload
        payload['method_anchor_layouts'] = _anchor_layouts
        payload['anchor_layout'] = _layout_payload

    # anchor_switch_events: read from experiment_cfg / cfg if explicitly provided.
    switches = (experiment_cfg or {}).get('anchor_switch_events') or (cfg or {}).get('anchor_switch_events') or []
    if isinstance(switches, Sequence):
        payload['anchor_switch_events'] = list(switches)

    # raw_residuals: only available at runtime; if cfg already provides aggregated residuals
    # (e.g., from a previous run's audit bundle), use them; otherwise, gate will skip.
    user_residuals = (cfg or {}).get('raw_residuals') or []
    if isinstance(user_residuals, Sequence):
        payload['raw_residuals'] = list(user_residuals)

    # noise_spec: aggregate from tasks (cold_start_offset_s, vio_outage_segments) for
    # §8.3 experiment_segment_coverage (sweep-level 7-segment coverage check).
    cold_offsets = [float((t.get('noise_spec') or {}).get('cold_start_offset_s', 0.0) or 0.0)
                    for t in scene_tasks if isinstance(t, Mapping)]
    has_vio_outage = any(
        bool((t.get('noise_spec') or {}).get('vio_outage_segments'))
        for t in scene_tasks if isinstance(t, Mapping)
    )
    payload['noise_spec'] = {
        'cold_start_offset_s': max(cold_offsets) if cold_offsets else 0.0,
        'vio_outage_segments': [(0.0, 1.0)] if has_vio_outage else [],
    }

    # trajectory_seeds: per-task seed for §8.2.0 政策5 assert_seed_required.
    # 从 task['seed'] / experiment_cfg['seed'] 收集每任务的种子（缺失记为 None）。
    user_traj_seeds = (experiment_cfg or {}).get('trajectory_seeds') or (cfg or {}).get('trajectory_seeds')
    if user_traj_seeds is not None:
        payload['trajectory_seeds'] = user_traj_seeds
    else:
        seeds: dict[str, Any] = {}
        for ti, task in enumerate(scene_tasks):
            if not isinstance(task, Mapping):
                continue
            seed_val = task.get('seed')
            if seed_val is None:
                # Fallback: 派生于 task_id + seq_id，保证可复现但每任务不同。
                seed_val = int(hash((task.get('task_id', ti), task.get('seq_id', f'<task_{ti}>'))) & 0xFFFFFFFF)
            seeds[str(task.get('seq_id', f'<task_{ti}>'))] = seed_val
        if seeds:
            payload['trajectory_seeds'] = seeds

    # trajectory_seed/nlos_seed/async_seed: §8.2.0 政策6 assert_seed_decoupling.
    # 优先取 experiment_cfg/cfg 显式声明的解耦种子；缺失时派生 3 个独立哈希种子，
    # 保证三者数值不同（满足解耦门禁的最低要求）。
    t_seed = (experiment_cfg or {}).get('trajectory_seed') or (cfg or {}).get('trajectory_seed')
    n_seed = (experiment_cfg or {}).get('nlos_seed') or (cfg or {}).get('nlos_seed')
    a_seed = (experiment_cfg or {}).get('async_seed') or (cfg or {}).get('async_seed')
    if t_seed is not None and n_seed is not None and a_seed is not None:
        payload['trajectory_seed'] = t_seed
        payload['nlos_seed'] = n_seed
        payload['async_seed'] = a_seed
    else:
        # 派生 3 个独立种子：基于固定命名空间 hash，保证三者数值互不相同。
        _ns = 'liquidloc.section8.seed_decoupling'
        payload['trajectory_seed'] = int(hashlib.sha256(f'{_ns}.trajectory'.encode()).hexdigest()[:8], 16)
        payload['nlos_seed'] = int(hashlib.sha256(f'{_ns}.nlos'.encode()).hexdigest()[:8], 16)
        payload['async_seed'] = int(hashlib.sha256(f'{_ns}.async'.encode()).hexdigest()[:8], 16)

    # generator_metadata: §8.2.0 政策2/4 audit 数据源。
    # 描述当前 sweep 使用的 trajectory generator 的族类与种子化维度。
    # 默认值反映 protocol_trajectory.py 的实现（种子化随机参数族，覆盖全部 5 个变异维度）。
    user_gen_meta = (experiment_cfg or {}).get('generator_metadata') or (cfg or {}).get('generator_metadata')
    if user_gen_meta is not None:
        payload['generator_metadata'] = user_gen_meta
    else:
        payload['generator_metadata'] = {
            'generator_family': 'parameterized_spline',  # protocol_trajectory.py 是种子化参数族
            'seeded_dimensions': [
                'path_shape',       # jag (L98)
                'turn_phase',       # aspect/phase (L67/L70)
                'stop_time',        # stop_c/stop_w/ramp_dur (L129/L133/L146)
                'speed_envelope',   # spd (L230)
            ],
            'seed_param_present': True,
            'is_seedable': True,
            'is_trajectory_lib': False,
            'lib_size': None,
        }

    return payload
from liquidloc.metrics.runtime_metrics import compute_runtime_metrics  # 统一运行时指标口径。
from liquidloc.models.features.feature_builder import build_feature_state_history  # 构建特征状态历史。
from liquidloc.models.features.feature_builder import build_feature_vector  # 构建特征向量。
from liquidloc.scenarios.async_levels import apply_async_level  # 应用异步层。
from liquidloc.scenarios.geometry_levels import build_anchor_layout  # 构建锚点布局。
from liquidloc.scenarios.geometry_levels import project_anchor_layout_to_reference  # 把锚点布局投影到参考坐标系。
from liquidloc.scenarios.nlos_levels import apply_nlos_level  # 应用 NLOS 层。
from liquidloc.scenarios.scene_sampler import sample_scenes  # 展开场景任务。
from liquidloc.scenarios.visual_levels import apply_visual_level  # 应用视觉层。
from liquidloc.sensors.anchor_model import build_anchor_lookup  # 构建锚点查找表。
from liquidloc.sensors.uwb_model import predict_range_to_anchor  # 预测到锚点的 UWB 距离。
from liquidloc.common.gt_utils import align_ground_truth, normalize_gt_rows, resolve_anchor_position  # 真值归一化、对齐与锚点解析的规范实现
from liquidloc.common.io_utils import dumps_json_text, read_json  # D10：严格标准 JSON 读写，拒绝 NaN/Infinity；dumps_json_text 提升到模块顶层，避免在 run 主循环内反复延迟 import（原函数内 import 经多次 task/method 迭代重复执行，且空 scene_tasks 路径下落盘 L938/L941 会 NameError）。


_NEURAL_METHODS = {MODEL_NAME_LSTM, MODEL_NAME_LIQUID, MODEL_NAME_TRANSFORMER}  # 需要模型参与的神经方法。
_CLASSICAL_METHODS = {ESTIMATOR_NAME_EKF, ESTIMATOR_NAME_ROBUST_EKF, ESTIMATOR_NAME_FGO}  # 纯估计器或经典方法。
_SCENE_AXES = SCENE_AXES  # 场景协议里六个主轴的固定顺序，来源为冻结协议。
_PARAM_ATTR_CANDIDATES = ('params', 'param_count', 'parameter_count', 'num_params')  # 参数数量字段的候选名字。
_RAM_ATTR_CANDIDATES = ('ram_peak', 'ram_peak_mb', 'peak_ram_mb', 'memory_peak_mb')  # 峰值内存字段的候选名字。
_MODEL_CFG_ROOT = find_project_root() / 'configs' / 'models'  # 模型配置目录根路径。

# 默认场景编码的惰性缓存，避免模块导入时加载协议文件。
_DEFAULT_SCENE_CODE: str | None = None


def _is_neural_method(method_name: str) -> bool:
    """检测方法名是否为神经方法（直接名或 `<neural>_ekf` 组合名）。

    严格白名单匹配：只承认 `{lstm, liquid, transformer}` 及其 EKF 外壳组合名
    `{lstm_ekf, liquid_ekf, transformer_ekf}`。任意含关键字的未知变体
    （如 `lstm_ekf_xxx`）不属于已注册神经面，必须返回 False，交由
    `_resolve_method_route` 抛 Unsupported method，禁止宽匹配静默放行。
    """
    if method_name in _NEURAL_METHODS:
        return True
    return method_name in {f"{neural}_ekf" for neural in _NEURAL_METHODS}


def _enforce_window_size_parity_for_neural_methods(
    methods: list[str], cfg: dict[str, Any]
) -> dict[str, int]:
    """§10.4 NN 截断对等 watchdog：解析所有神经方法的 window.size，
    保证二者一致，不一致时 fail-loud。

    - 经典方法 (ekf/robust_ekf/fgo) 和未知方法被静默跳过。
    - 返回 {method_name: window_size}，仅含检测到的神经方法。
    """
    model_cfgs = cfg.get('model_cfgs') or {}
    sizes: dict[str, int] = {}

    for method in methods:
        if method in _CLASSICAL_METHODS:
            continue  # 经典方法无 window 字段，跳过
        if not _is_neural_method(method):
            continue  # 未知方法，跳过（主循环会 later 报错，这里不重复）
        # 解析 window.size：优先显式覆盖，其次默认 YAML
        if method in model_cfgs:
            sizes[method] = model_cfgs[method]['window']['size']
        else:
            default_cfg = _resolve_model_cfg(cfg, method)
            sizes[method] = default_cfg['window']['size']

    # 对等校验：所有神经方法必须 window.size 一致
    unique = set(sizes.values())
    if len(unique) > 1:
        raise ValueError(
            f'§10.4 NN 截断对等 watchdog 检测到不一致: '
            f'{dict(zip(sizes.keys(), [sizes[k] for k in sizes.keys()]))}'
        )
    return sizes


def _resolve_git_commit() -> str | None:  # 获取当前 git HEAD commit hash。
    """返回当前源码 HEAD 的短 hash（7位）；git 不可用时返回 None，不抛出。"""
    import subprocess as _subprocess
    try:
        return _subprocess.check_output(  # 调用 git rev-parse --short=7 HEAD。
            ['git', 'rev-parse', '--short=7', 'HEAD'],
            cwd=find_project_root(),  # 用项目根目录而非 cwd，避免 checkout 后的边缘情况。
            stderr=_subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:  # 捕获 git 不存在、非仓库、git 进程超时等所有异常。
        return None


def _resolve_config_hash(cfg: Mapping[str, Any]) -> str:  # 计算 experiment config 的稳定 hash。
    """对实验配置计算 SHA256 hexdigest；排除 output_root/dataset_name 等运行时字段，保证不同实验间可比。"""
    import hashlib as _hashlib
    import json as _json
    stable_cfg = {  # 只保留与实验结果强相关的字段，排除运行时推断字段。
        k: v for k, v in cfg.items()
        if k not in (
            'output_root', 'project_root', 'registry_path', 'experiment_protocol_path',
            'raw_root', 'field_mapping', 'seq_ids', 'dataset_name',
        )
    }
    stable_str = _json.dumps(stable_cfg, sort_keys=True, ensure_ascii=True)
    return _hashlib.sha256(stable_str.encode()).hexdigest()


def _get_default_scene_code() -> str:
    """惰性获取协议默认场景编码，从 get_nominal_levels 动态构造。"""
    global _DEFAULT_SCENE_CODE
    if _DEFAULT_SCENE_CODE is None:
        _nominal = get_nominal_levels()
        _DEFAULT_SCENE_CODE = encode_scene(SceneSpec(
            A_level=_nominal["A"], N_level=_nominal["N"], V_level=_nominal["V"],
            K_value=_nominal["K"],
        ))
    return _DEFAULT_SCENE_CODE


def _resolve_output_root(cfg: Mapping[str, Any], default_name: str) -> Path:  # 解析输出根目录。
    """解析输出根目录，委托给 common.paths.resolve_output_root 统一处理。"""
    return resolve_output_root(dict(cfg), default_name)  # shallow copy: resolve_output_root only reads, never mutates


class _RuntimeTraceCollector:  # 收集运行时延时轨迹。
    """收集运行时延轨迹。"""

    def __init__(self) -> None:  # 初始化运行时延时收集器。
        self.latency_trace_ms: list[float] = []  # 每一步的总时延轨迹。
        self._pending_model_latency_ms = 0.0  # 尚未并入 step 的模型耗时。

    def observe_model_latency(self, latency_ms: float) -> None:  # 记录模型前向耗时。
        """记录模型前向耗时。"""
        safe_latency_ms = coerce_finite_scalar(latency_ms, name="latency_ms")  # 先拒绝 NaN/Inf，防止下游统计被污染。
        self._pending_model_latency_ms += max(safe_latency_ms, 0.0)  # 累计非负模型耗时。

    def observe_step_latency(self, latency_ms: float) -> None:  # 记录一次 step 的总耗时。
        """记录一步融合或估计的总耗时。"""
        safe_step_ms = coerce_finite_scalar(latency_ms, name="latency_ms")  # 先拒绝 NaN/Inf，防止 NaN 经 max 泄漏到 latency_trace_ms。
        total_latency_ms = max(self._pending_model_latency_ms + safe_step_ms, 0.0)  # 合并模型和 step 耗时。
        self.latency_trace_ms.append(total_latency_ms)  # 写入轨迹。
        self._pending_model_latency_ms = 0.0  # 清空待结转耗时。


class _TimedEstimatorProxy:  # 给估计器包一层耗时统计代理。
    """给估计器包一层耗时统计代理。"""

    def __init__(self, estimator: Any, collector: _RuntimeTraceCollector) -> None:  # 保存原始估计器和收集器。
        self._estimator = estimator  # 原始估计器对象。
        self._collector = collector  # 耗时收集器。

    def __getattr__(self, name: str) -> Any:  # 其余属性直接透传。
        """把未显式包装的属性继续转给原始估计器。"""
        return getattr(self._estimator, name)  # 未显式包装的属性继续透传给原始估计器，保证接口不丢。

    def step(self, event: Any) -> StateEstimate:  # 对 step 调用做耗时采样。
        """对 step 调用做耗时采样。"""
        start = time.perf_counter()  # 记录开始时间。
        result = self._estimator.step(event)  # 调用真实 step。
        elapsed_ms = (time.perf_counter() - start) * 1000.0  # 换算成毫秒。
        self._collector.observe_step_latency(elapsed_ms)  # 记录到收集器。
        return result  # 把原始结果原样返回。


class _TimedModelProxy:  # 给模型包一层耗时统计代理。
    """给模型前向推理包一层耗时统计代理。"""

    def __init__(self, model: Any, collector: _RuntimeTraceCollector) -> None:  # 保存原始模型和收集器。
        self._model = model  # 原始模型对象。
        self._collector = collector  # 耗时收集器。

    def __getattr__(self, name: str) -> Any:  # 其余属性直接透传。
        """把未显式包装的属性继续转给原始模型。"""
        return getattr(self._model, name)  # 未显式包装的属性继续透传给原始模型，避免破坏模型接口。

    def infer_intermediate(self, window_tensor: Any) -> ModelIntermediate:  # 对模型中间推理做耗时采样；签名对齐 ModelAPI.infer_intermediate，与 _TimedEstimatorProxy.step 同口径补全类型注解。
        """对模型中间推理做耗时采样。"""
        start = time.perf_counter()  # 记录开始时间。
        result = self._model.infer_intermediate(window_tensor)  # 调用真实推理。
        elapsed_ms = (time.perf_counter() - start) * 1000.0  # 换算成毫秒。
        self._collector.observe_model_latency(elapsed_ms)  # 记录模型耗时。
        return result  # 返回原始中间结果。


def _resolve_events_for_task(task: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:  # 按任务优先级从不同配置入口找对应事件序列。
    """根据任务优先级从不同配置入口找对应事件序列。

    规则是先按 seq_id/task_id/scene_variant_id/scene_id 分派，再回退到
    全局显式 `events`。这样可以兼容直接注入事件和按任务分组提供事件
    两类调用方式。如果同时传入按 seq_id 分派和全局合并事件流，优先
    使用按 seq_id 分派的结果，避免在多任务批次里把全部合并流灌到每个
    任务中重复运行。
    """
    by_task_id = cfg.get('events_by_task_id') or {}  # 按任务编号分派的事件表。
    task_id = task.get('task_id')  # 取任务编号，避免 .get() 与 [] 混用导致 KeyError，与 _resolve_scene_context_for_task 一致。
    if isinstance(by_task_id, Mapping) and task_id in by_task_id:  # 命中 task_id 就直接取对应事件。
        return list(by_task_id[task_id])  # 命中 task_id 就直接取对应事件。
    by_scene_variant_id = cfg.get('events_by_scene_variant_id') or {}  # 按场景变体编号分派。
    scene_variant_id = task.get('scene_variant_id')  # 变体编号，作为第二优先级键。
    if isinstance(by_scene_variant_id, Mapping) and scene_variant_id in by_scene_variant_id:  # 变体级别匹配优先于更粗粒度。
        return list(by_scene_variant_id[scene_variant_id])  # 变体级别匹配优先于更粗粒度。
    by_seq_id = cfg.get('events_by_seq_id') or {}  # 按序列编号分派。
    seq_id = task.get('seq_id')  # 序列编号，用于更粗粒度回退。
    if isinstance(by_seq_id, Mapping) and seq_id in by_seq_id:  # 序列级匹配继续回退。
        return list(by_seq_id[seq_id])  # 序列级匹配继续回退。
    by_scene_id = cfg.get('events_by_scene_id') or {}  # 最后按场景编号分派。
    scene_id = task.get('scene_id')  # 最后回退到场景编号。
    if isinstance(by_scene_id, Mapping) and scene_id in by_scene_id:  # 最后按场景编号分派。
        return list(by_scene_id[scene_id])  # 仍然找不到前面更细的键时再用场景级事件。
    if isinstance(cfg.get('events'), list):  # 最后回退到全局显式事件流。
        return list(cfg['events'])  # 全局显式事件优先级最低，用于单任务调用。
    raise ValueError(f"No events available for task {task.get('task_id', '<unknown>')}")  # 没有任何可用事件时直接报错，说明任务输入不完整。


def _resolve_scene_context_for_task(task: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:  # 解析快速场景上下文，找不到就返回空字典。
    """解析快速场景上下文，找不到就返回空字典。

    这里的上下文通常包含预处理过的场景信息、几何结果或报告。
    解析顺序和事件解析保持一致，先细后粗，方便局部覆盖全局默认。
    """
    quick_scene_context_by_task_id = cfg.get('quick_scene_context_by_task_id') or {}  # task 级快速上下文表。
    task_id = task.get('task_id')  # 取当前任务编号，后面反复使用。
    if isinstance(quick_scene_context_by_task_id, Mapping) and task_id in quick_scene_context_by_task_id:  # 命中 task 级别直接返回拷贝。
        # 深拷贝而非浅拷贝：scene_context 含 anchor_layout / geometry_report / scenario_reports
        # 等嵌套结构，浅拷贝会与外部共享引用，破坏 D3 anchor metadata 不可漂移合同。
        return deepcopy(quick_scene_context_by_task_id[task_id] or {})
    quick_scene_context_by_scene_variant_id = cfg.get('quick_scene_context_by_scene_variant_id') or {}  # 变体级上下文表。
    scene_variant_id = task.get('scene_variant_id')  # 变体编号，作为第二优先级键。
    if isinstance(quick_scene_context_by_scene_variant_id, Mapping) and scene_variant_id in quick_scene_context_by_scene_variant_id:  # 变体命中时就不再往下找。
        # 深拷贝：同上，保护 anchor_layout 嵌套结构不与配置共享引用，防止跨任务漂移。
        return deepcopy(quick_scene_context_by_scene_variant_id[scene_variant_id] or {})
    quick_scene_context_by_seq_id = cfg.get('quick_scene_context_by_seq_id') or {}  # 序列级上下文表。
    seq_id = task.get('seq_id')  # 序列编号，用于更粗粒度回退。
    if isinstance(quick_scene_context_by_seq_id, Mapping) and seq_id in quick_scene_context_by_seq_id:  # 序列命中后返回对应上下文。
        # 深拷贝：同上，保护 anchor_layout 嵌套结构不与配置共享引用，防止跨任务漂移。
        return deepcopy(quick_scene_context_by_seq_id[seq_id] or {})
    quick_scene_context_by_scene_id = cfg.get('quick_scene_context_by_scene_id') or {}  # 场景级上下文表。
    scene_id = task.get('scene_id')  # 最后回退到场景编号。
    if isinstance(quick_scene_context_by_scene_id, Mapping) and scene_id in quick_scene_context_by_scene_id:  # 场景级命中后返回。
        # 深拷贝：同上，保护 anchor_layout 嵌套结构不与配置共享引用，防止跨任务漂移。
        return deepcopy(quick_scene_context_by_scene_id[scene_id] or {})
    return {}  # 所有分层都没有命中时，说明当前任务没有可复用的快速上下文。


def _resolve_source_report_for_task(task: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:  # 解析任务的来源报告，优先用任务自带值，其次查配置回退表。
    """解析任务的来源报告，优先用任务自带值，其次查配置回退表。"""
    source_report = task.get('source_report')  # 任务内直接携带的来源报告优先级最高。
    if isinstance(source_report, Mapping):  # 任务内直接携带的来源报告优先级最高。
        # 深拷贝而非浅拷贝：source_report 含 anchor_layout / anchor_layout_metadata / scene_parameters
        # 等嵌套结构，浅拷贝会与外部共享引用，破坏 D3 anchor metadata 不可漂移合同。
        return deepcopy(source_report)

    source_report_by_seq_id = cfg.get('source_report_by_seq_id') or {}  # 序列级来源报告表。
    seq_id = task.get('seq_id')  # 当前任务所属序列编号。
    if isinstance(source_report_by_seq_id, Mapping) and seq_id in source_report_by_seq_id:  # 序列级命中后返回拷贝。
        # 深拷贝：source_report_by_seq_id[seq_id] 在同一 seq_id 的多个任务间共享，
        # anchor_layout 为冻结几何合同，必须隔离以防止跨任务漂移。
        return deepcopy(source_report_by_seq_id[seq_id] or {})

    source_report_by_scene_id = cfg.get('source_report_by_scene_id') or {}  # 场景级来源报告表。
    scene_id = task.get('scene_id')  # 当前任务所属场景编号。
    if isinstance(source_report_by_scene_id, Mapping) and scene_id in source_report_by_scene_id:  # 场景级继续回退。
        # 深拷贝：同上，保护 anchor_layout 嵌套结构不与配置共享引用。
        return deepcopy(source_report_by_scene_id[scene_id] or {})
    return {}  # 没有来源报告时返回空字典，调用方按“无来源信息”处理。


def _resolve_ground_truth_rows_for_task(task: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, float]] | None:  # 解析任务对应的真值轨迹行，找不到就返回 None。
    """解析任务对应的真值轨迹行，找不到就返回 `None`。

    解析顺序优先查配置中的显式映射，再查磁盘上的 `gt.json`。
    这样可以同时支持内存注入和按序列目录读取两种数据接入方式。
    """
    ground_truth_by_seq_id = cfg.get('ground_truth_by_seq_id') or {}  # 序列级真值表。
    seq_id = task.get('seq_id')  # 当前任务所属序列编号。
    if isinstance(ground_truth_by_seq_id, Mapping) and seq_id in ground_truth_by_seq_id:  # 显式映射命中后直接规范化。
        return normalize_gt_rows(ground_truth_by_seq_id[seq_id])  # 显式映射命中后直接规范化。

    ground_truth_by_scene_id = cfg.get('ground_truth_by_scene_id') or {}  # 场景级真值表。
    scene_id = task.get('scene_id')  # 当前任务所属场景编号。
    if isinstance(ground_truth_by_scene_id, Mapping) and scene_id in ground_truth_by_scene_id:  # 场景级命中后规范化返回。
        return normalize_gt_rows(ground_truth_by_scene_id[scene_id])  # 场景级命中后规范化返回。

    ground_truth_root = cfg.get('ground_truth_root')  # 磁盘真值目录根路径。
    if ground_truth_root is not None and is_string_like(seq_id) and str(seq_id).strip():  # 磁盘真值目录存在且序列编号有效时才尝试读文件。
        seq_id_str = str(seq_id)  # 统一转为 Python str，兼容 numpy.str_，避免 validate_path_component 的 isinstance 检查失败。
        validate_path_component(seq_id_str, name='seq_id')  # 校验 seq_id 不含路径穿越字符。
        gt_path = Path(ground_truth_root) / seq_id_str / 'gt.json'  # 按序列约定去找真值文件。
        if gt_path.is_file():  # 只有真值文件真的存在才读。
            return normalize_gt_rows(read_json(gt_path))  # 从磁盘读后再规范化。
    return None  # 所有回退路径都没有真值时显式返回 None。


def _clone_events(events: Iterable[Any]) -> list[dict[str, Any]]:  # 把事件序列复制成可安全改写的字典列表。
    """把事件序列复制成可安全改写的字典列表。"""
    cloned = []  # 新建结果列表，避免原始事件被后续场景变换污染。
    for event in list(events):  # 逐个复制事件，避免共享引用。
        if hasattr(event, 'to_dict') and callable(event.to_dict):  # 支持对象型事件的显式序列化接口。
            cloned.append(deepcopy(event.to_dict()))  # 先 to_dict 再深拷贝，保证与 Mapping 路径一致的全独立副本，不依赖 to_dict 实现是否深拷贝。
        elif isinstance(event, Mapping):  # 映射型事件先转 dict 再深拷贝。
            cloned.append(deepcopy(dict(event)))  # 映射型事件先转 dict，再深拷贝一层嵌套结构。
        else:  # 其他类型事件直接报错，避免继续扩散错误输入。
            raise TypeError(f'event must be a Mapping or have a to_dict() method, got {type(event).__name__}')  # 事件必须是映射型或带 to_dict 接口，后面要按键读字段。
    return cloned  # 返回完全独立的事件副本链。


def _resolve_scene_parameters(task: dict[str, Any], protocol_cfg: dict[str, Any]) -> dict[str, Any]:  # 把任务里的场景参数整理成统一的 axes/flat 结构。
    """把任务里的场景参数整理成统一的 axes/flat 结构。"""
    scene_parameters = task.get('scene_parameters')  # 任务可能已经带了完整结构。
    axes = dict(task.get('axes') or {})  # 任务里的轴级简写仍然是场景退化事实来源。

    def _normalize_partial_axis_levels(axis_levels: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for axis, level in axis_levels.items():
            if axis not in _SCENE_AXES:
                continue
            if not is_string_like(level):
                raise TypeError(f"scene axis level must be a string, got {type(level).__name__} for axis {axis}")
            stripped_level = str(level).strip()
            if not stripped_level:
                raise ValueError(f"scene axis level must not be blank for axis {axis}")
            payload: dict[str, Any] = {'level': stripped_level}
            # F2-H1（阶段 8 修复 HIGH-9）: K level 名（如 K3）只用于选几何条件，anchor_count 不从 level 名后缀解析（K0/K1/K3 后缀是 0/1/3 与协议锚数 4 完全错位）。
            # 从 protocol_cfg['axes']['K'][level].get('anchor_count') 读取协议事实，缺省为 4。
            if axis == 'K':
                k_payload = (((protocol_cfg.get('axes') or {}).get('K') or {}).get(stripped_level) or {})
                payload['anchor_count'] = k_payload.get('anchor_count', 4)
            normalized[axis] = payload
        return normalized

    has_scene_axes = any(axis in axes for axis in _SCENE_AXES)  # 只要显式声明了任一场景轴，就可能需要协议展开。
    has_complete_scene_axes = all(axis in axes for axis in _SCENE_AXES)  # 只有六轴齐全时才能展开完整协议事实。
    if isinstance(scene_parameters, Mapping):  # 任务里已经带完整结构时直接规整。
        resolved = {  # 先构造标准化参数字段，后面统一复制、补齐和回填。
            'axes': dict(scene_parameters.get('axes') or {}),  # 轴级参数保留到 axes 区。
            'flat': dict(scene_parameters.get('flat') or {}),  # 扁平化参数单独放 flat 区。
            'axis_metadata': dict(scene_parameters.get('axis_metadata') or {}),  # 轴元数据保留到 axis_metadata 区。
        }
        if has_complete_scene_axes:  # 只有六轴齐全时，才能把显式 axes 还原成完整协议事实并合并回 scene_parameters。
            protocol_resolved = attach_scene_parameters(axes, protocol_cfg)  # 先按协议展开完整六轴事实。
            protocol_resolved = protocol_resolved.to_dict() if isinstance(protocol_resolved, SceneParameters) else protocol_resolved  # SceneParameters → 纯 dict。
            merged_axes: dict[str, Any] = {}  # 轴级事实以协议展开结果为准，只接纳非冲突补充字段。
            for axis, payload in dict(protocol_resolved.get('axes') or {}).items():
                merged_payload = dict(payload or {})  # 先复制协议真相，避免后续覆盖 anchor_count / geom_condition 等冻结字段。
                explicit_payload = resolved['axes'].get(axis)
                if isinstance(explicit_payload, Mapping):
                    for key, value in explicit_payload.items():
                        if key not in merged_payload:
                            merged_payload[key] = value  # 只保留协议里没有的补充注释字段。
                merged_axes[axis] = merged_payload
            for axis, payload in resolved['axes'].items():
                if axis in _SCENE_AXES and axis not in merged_axes:
                    merged_axes[axis] = dict(payload) if isinstance(payload, Mapping) else payload
            merged_flat = dict(protocol_resolved.get('flat') or {})  # 扁平字段同样以协议展开结果为准。
            for key, value in resolved['flat'].items():
                merged_flat.setdefault(key, value)  # 显式 scene_parameters 只能补充非冲突展示字段。
            merged_axis_metadata = dict(protocol_resolved.get('axis_metadata') or {})  # 轴元数据同样以协议展开结果为准。
            for key, value in resolved['axis_metadata'].items():
                merged_axis_metadata.setdefault(key, value)  # 显式 scene_parameters 只能补充非冲突轴元数据字段。
            resolved = {
                'axes': merged_axes,
                'flat': merged_flat,
                'axis_metadata': merged_axis_metadata,
            }
        task['scene_parameters'] = deepcopy(resolved)  # 回写一个安全副本，避免下游误改原引用。
        return resolved  # 任务本身已有参数时直接规范化返回。

    if has_complete_scene_axes:  # 只有六轴齐全时，才能按冻结协议展开完整事实。
        resolved = attach_scene_parameters(axes, protocol_cfg)  # 由协议把轴级简写展开为完整参数。
        resolved = resolved.to_dict() if isinstance(resolved, SceneParameters) else resolved  # SceneParameters → 纯 dict。
        task['scene_parameters'] = deepcopy(resolved)  # 回写一个安全副本，避免下游误改原引用。
        return resolved

    if has_scene_axes:  # 部分轴只保留显式声明事实，不伪造缺失轴也不触发完整协议展开。
        resolved = {
            'axes': _normalize_partial_axis_levels(axes),
            'flat': {},
            'axis_metadata': {},
        }
        task['scene_parameters'] = deepcopy(resolved)  # 回写一个安全副本，避免下游误改原引用。
        return resolved

    resolved = {'axes': {}, 'flat': {}, 'axis_metadata': {}}  # 没有任何参数时返回空骨架，保证字段结构稳定。
    task['scene_parameters'] = deepcopy(resolved)  # 回写一个安全副本，避免下游误改原引用。
    return resolved


def _apply_scene_task(events: Iterable[Any], task: dict[str, Any], protocol_cfg: dict[str, Any], gt_rows: list[dict[str, float]] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:  # 按任务的 A/N/V/G/K/M 轴对事件做场景化改写。第 13 轮审查 MEDIUM-1 修复（R13-C M1）：补全 events 类型注解，与 _clone_events L269 / _materialize_scene_task L612 口径一致。R12-C L1 声称修复但 L363 遗漏，本轮补全。
    """按任务的 A/N/V/G/K/M 轴对事件做场景化改写。

    公开 benchmark 数据集（miluv/ntu_viral）禁止施加模拟场景轴扰动，
    这些数据应使用真实退化而非模拟退化。

    .. note::
        协议一致性张力说明（第 2 轮审查 HIGH-5）：
        `configs/base/scene_axis_protocol.yaml` 声明
        `perturbation_injection_point: sim_materializer`，用户审查标准 #5 也要求
        "扰动一次定稿 — 只在 sim_materializer 注入，reader/prepare 不再扰动"。
        但当前实现中，A/N/V/M 轴的场景扰动（async/NLOS/visual/modality_drop）
        实际由本函数在 core_pipeline 层注入，而非 sim_materializer。
        sim_materializer 现默认协议轨迹 + axes_override G/K；A/N/V/M 场景轴仍在
        core_pipeline 层（见 _remap_uwb_ranges_for_geometry 的 F3-H1 张力说明）。

        这是已知架构决策，核心约束"reader/prepare 不再扰动"仍然成立
        （reader 保持原始语义保真，prepare 不注入场景扰动），但与协议声明
        "只在 sim_materializer 注入"存在张力。彻底修复需要：
        (a) 把 A/N/V/M/G 场景扰动整体迁移到 sim_materializer（为每个场景变体
            预生成独立 raw 数据），或
        (b) 更新协议声明为两阶段架构（sim_materializer 基线 + core_pipeline
            场景轴），并同步更新用户审查标准 #5 的表述。
        本轮审查不修改注入逻辑，避免引入新的科学语义偏移；此处仅补充文档，
        确保张力对下游消费者可见。TODO(C1) 跟踪此架构决策的最终落地方案。
    """
    # 显式守门：公开 benchmark 数据集禁止场景轴扰动。
    from liquidloc.protocol.experiment_gates import get_public_benchmark_allowed_datasets, normalize_public_dataset_name
    raw_dataset_name = task.get('dataset_name', '')
    # 归一化 dataset_name（小写、去空白），避免 'MILUV'/' MiLuV ' 等大小写或空白变体绕过公开基准守门。
    # 空值/None 跳过归一化，dataset_name 保持空字符串（不在公开基准集合中，守门自然通过）。
    dataset_name = normalize_public_dataset_name(raw_dataset_name) if raw_dataset_name else ''
    _public_datasets = frozenset(get_public_benchmark_allowed_datasets())
    if dataset_name in _public_datasets:
        # 第 3 轮审查 HIGH-3 修复：守门必须同时检查 task['axes'] 与
        # task['scene_parameters']['axes'] 两条路径，否则任务可以通过
        # scene_parameters 路径注入模拟场景扰动而绕过守门。
        task_axes = task.get('axes') or {}
        scene_param_axes = (task.get('scene_parameters') or {}).get('axes') or {}
        scene_axes_in_task = [
            ax for ax in _SCENE_AXES
            if ax in task_axes or ax in scene_param_axes
        ]
        if scene_axes_in_task:
            raise ValueError(
                f'Public benchmark dataset {dataset_name!r} must not have simulated scene-axis '
                f'perturbations, but task has axes: {scene_axes_in_task} '
                f'(checked task[\'axes\'] and task[\'scene_parameters\'][\'axes\']). '
                f'Use real degradation from the dataset instead.'
            )
    working_events = _clone_events(events)  # 先复制一份事件，后面的场景变换都在副本上做。
    scene_parameters = _resolve_scene_parameters(task, protocol_cfg)  # 先把任务参数展开成统一结构。
    axes = dict(task.get('axes') or {})  # 任务里原始轴级选择。
    axis_params = dict(scene_parameters.get('axes') or {})  # 展开后的轴级参数，包含各轴详细设置。

    scenario_reports: dict[str, Any] = {}  # 收集各轴变换的报告，供后续评估和审计使用。
    if 'A' in axis_params:  # A 轴代表异步扰动。
        async_level = str(axes.get('A') or axis_params['A'].get('level'))  # 选取异步等级，优先用任务显式值。
        working_events, async_report = apply_async_level(working_events, async_level, protocol_cfg['axes']['A'])  # 写回异步变换结果和报告。
        scenario_reports['A'] = async_report
        # B08 修复: 把 async_report 中的 misalignment_event_count 提升到顶层 scenario_reports 字段，
        # 供 precheck 的总累计 ≥ 20 校验使用。
        scenario_reports['A_misalignment_event_count'] = int(async_report.get('misalignment_event_count', 0))
        # apply_async_level 内部已排序并重算 dt（含 validate_event_sequence 校验），
        # 此处不再冗余排序和重算，避免与 _recompute_dt 的口径差异。

    if 'N' in axis_params:  # N 轴代表 NLOS 扰动。
        nlos_level = str(axes.get('N') or axis_params['N'].get('level'))  # 选取 NLOS 等级。
        working_events, scenario_reports['N'] = apply_nlos_level(working_events, nlos_level, protocol_cfg['axes']['N'], gt_rows=gt_rows)  # 把 NLOS 失真写入事件流。

    if 'V' in axis_params:  # V 轴代表视觉侧扰动。
        visual_level = str(axes.get('V') or axis_params['V'].get('level'))  # 选取视觉等级。
        working_events, scenario_reports['V'] = apply_visual_level(working_events, visual_level, protocol_cfg['axes']['V'])  # 应用视觉侧场景扰动。

    anchor_layout = None  # 默认没有几何布局改写。
    geometry_report = None  # 默认没有几何报告。
    # 五轴档位协议：G 轴已并入 K 轴。几何条件 + 锚数都由 K 轴决定。
    if 'K' in axis_params:
        k_payload = axis_params['K']
        if not isinstance(k_payload, Mapping):
            raise TypeError(f"K axis payload must be a mapping, got {type(k_payload).__name__}")
        k_level = str(axes.get('K') or k_payload.get('level') or 'K3').strip()
        anchor_count = k_payload.get('anchor_count', 4)  # 五轴档位协议：K 轴锚数全档固定 4
        if not isinstance(anchor_count, int) or anchor_count < 3:
            anchor_count = 4  # 兜底：协议硬约束 = 4
        # 直接传原值，build_anchor_layout 内部 is_integer 校验。
        anchor_layout, geometry_report = build_anchor_layout(
            anchor_count, k_level, protocol_cfg['axes']['K'],
        )
        scenario_reports['K'] = geometry_report  # 几何报告放入 K 轴场景报告。

    # M 轴代表模态缺失（按时间段屏蔽指定模态），在 A/N/V/K 之后处理。
    # 协议层 M 轴参数为 modality_drop_prob（单帧丢失概率）和 affected_modalities（受影响模态列表）。
    # 此处将 prob 转换为连续缺失时间段（长度 = 总时长 × prob），调用 apply_modality_drop。
    if 'M' in axis_params:
        from liquidloc.scenarios.missing_modalities import apply_modality_drop
        modality_cfg = axis_params['M']
        # 五轴档位协议 M 轴 modality_drop_prob 是区间 [low, high] 形式。
        # 消费者取中点作为该序列的注入代表值（与 K 轴 geom_condition 区间处理一致）。
        modality_drop_prob_raw = modality_cfg.get('modality_drop_prob', 0.0)
        if isinstance(modality_drop_prob_raw, (list, tuple)) and len(modality_drop_prob_raw) >= 1:
            modality_drop_prob = float(sum(modality_drop_prob_raw) / len(modality_drop_prob_raw))
        else:
            modality_drop_prob = coerce_finite_scalar(
                modality_drop_prob_raw, name='modality_drop_prob'
            )
        # 第 3 轮审查 HIGH-1 修复：modality_drop_prob 是概率，必须在 [0, 1] 区间。
        # 与 async_levels.py L393-394 的 burst_missing_prob 校验口径保持一致，
        # 否则协议 YAML 注释（"单帧丢失概率"）会被违反且无错误信号。
        if not 0.0 <= modality_drop_prob <= 1.0:
            raise ValueError(
                f'modality_drop_prob must be within [0, 1], got {modality_drop_prob}'
            )
        # 第 8 阶段修复 HIGH-12: IMU 缺失率有专属字段 imu_drop_prob（如 M3 imu_drop_prob=[0.01,0.05]），
        # 不能用 modality_drop_prob=0.30 给 IMU（违反协议层 "IMU ≤5%" 硬约束）。
        # IMU 是时间基准，缺失过高导致估计器状态发散。
        imu_drop_prob_raw = modality_cfg.get('imu_drop_prob', 0.0)
        if isinstance(imu_drop_prob_raw, (list, tuple)) and len(imu_drop_prob_raw) >= 1:
            imu_drop_prob = float(sum(imu_drop_prob_raw) / len(imu_drop_prob_raw))
        else:
            imu_drop_prob = coerce_finite_scalar(
                imu_drop_prob_raw, name='imu_drop_prob', min_value=0.0, max_value=1.0
            )
        if not 0.0 <= imu_drop_prob <= 1.0:
            raise ValueError(
                f'imu_drop_prob must be within [0, 1], got {imu_drop_prob}'
            )
        # 第 3 轮审查 HIGH-2 修复：affected_modalities 必须是列表/元组，
        # 字符串虽是 Sequence 但会被 list() 拆成字符（list("uwb") → ['u','w','b']），
        # 导致下游 apply_modality_drop 收到非法模态名。此处显式拒绝字符串。
        raw_affected_modalities = modality_cfg.get('affected_modalities') or []
        if isinstance(raw_affected_modalities, str):
            raise TypeError(
                f'affected_modalities must be a list or tuple of modality names, '
                f'got a string {raw_affected_modalities!r} which would be split into '
                f'individual characters by list().'
            )
        affected_modalities = list(raw_affected_modalities)
        m_reports: list[dict[str, Any]] = []
        if modality_drop_prob > 0.0 and affected_modalities and working_events:
            # 计算当前事件序列的时间范围，用于构造缺失时间段。
            event_times = [coerce_finite_scalar(e.get('t', 0.0), name='event.t') for e in working_events]
            t_min = min(event_times)
            t_max = max(event_times)
            total_duration = t_max - t_min
            if total_duration > 0.0:
                scene_id_str = str(task.get('scene_id', ''))
                for modality_name in affected_modalities:
                    # 语义说明（第 2 轮审查 MEDIUM）：协议 YAML 注释将 modality_drop_prob 描述为
                    # "单帧某模态数据丢失的概率"（per-frame Bernoulli 模型），但当前实现将其解释为
                    # "总时长 × prob = 连续缺失时间段长度"（contiguous block 模型）。两者统计特性不同：
                    # Bernoulli 模型产生散点缺失（每帧独立丢弃，易于插值），
                    # contiguous 模型产生连续块缺失（整段丢失，难于插值）。
                    # 当前采用 contiguous 模型的原因：(1) 确定性可复现（无需随机数生成器）；
                    # (2) 对下游估计器构成更强压力（连续缺失比散点缺失更难处理）；
                    # (3) 与 A 轴 burst_missing 的 contiguous 窗口语义一致。
                    # 若需切换到 Bernoulli 模型，需同步更新协议 YAML 注释和此实现。
                    # 第 8 阶段修复 HIGH-12: IMU 走 imu_drop_prob（协议层 M3 imu_drop_prob=[0.01,0.05]），
                    # 其他模态（UWB/VIO）走 modality_drop_prob。
                    modality_drop_prob_for_modality = imu_drop_prob if str(modality_name).strip().lower() == 'imu' else modality_drop_prob
                    drop_duration = total_duration * modality_drop_prob_for_modality
                    # 用稳定哈希选择时间段起始位置，保证可复现（不引入 random 依赖）。
                    m_seed_str = f"modality_drop:{modality_name}:{scene_id_str}"
                    m_seed = int(hashlib.sha256(m_seed_str.encode('utf-8')).hexdigest()[:8], 16)
                    # 起始位置在 [t_min, t_max - drop_duration] 范围内确定性选取。
                    start_range = max(0.0, total_duration - drop_duration)
                    # 用哈希值低 16 位生成 [0, 1) 确定性浮点数，替代 random.Random。
                    random_fraction = (m_seed & 0xFFFF) / 65536.0
                    drop_start = t_min + random_fraction * start_range
                    drop_end = drop_start + drop_duration
                    working_events, m_report = apply_modality_drop(
                        working_events, str(modality_name).strip().lower(), [(drop_start, drop_end)]
                    )
                    m_reports.append(m_report)
        scenario_reports['M'] = m_reports

    return working_events, {  # 返回重写后的事件序列和场景报告，供后续流程继续处理。
        'scene_parameters': scene_parameters,  # 返回统一格式的场景参数。
        'scenario_reports': scenario_reports,  # 返回每个轴的处理报告。
        'anchor_layout': anchor_layout,  # 返回几何布局，供后续估计器使用。
        'geometry_report': geometry_report,  # 返回几何构造细节，方便审计。
    }


def _remap_uwb_ranges_for_geometry(  # 当几何布局变化时，按残差逻辑把 UWB 测距迁移到新布局。
    events: list[dict[str, Any]],  # 当前场景事件序列，后面会复制后再改写。
    gt_rows: list[dict[str, float]] | None,  # 与事件时间对齐的真值行，缺了就没法算残差。
    old_anchor_layout: dict[str, Any] | None,  # 原始锚点布局，用来解释旧测距对应的几何位置。
    new_anchor_layout: dict[str, Any] | None,  # 新锚点布局，用来决定重映射后的目标位置。
) -> list[dict[str, Any]]:  # 返回几何重映射后的事件序列。
    """在几何布局变化后，把 UWB 距离按同一残差逻辑重映射到新锚点布局。

    这个函数只处理几何轴变化后的观测对齐，不负责改造事件生成逻辑本身。
    它依赖三类输入同时到位:  # 下面三项缺一不可，否则几何重映射会失去可靠基准。
    1. `gt_rows` 用来提供时间对齐后的真值位置，保证重映射不是盲改。
    2. `old_anchor_layout` 用来解释原始 UWB 观测对应的是哪一个旧锚点。
    3. `new_anchor_layout` 用来决定改写后应该落到哪个新锚点上。

    核心做法是保留“观测值相对几何理论值的残差”，再把这个残差迁移到新布局。
    这样能尽量维持误差结构一致，而不是直接把原测距原样拷贝到新几何里。

    .. note::
        F3-H1 张力说明：G 轴几何扰动在此注入（通过 `_apply_scene_task` 调用
        `build_anchor_layout` 生成新布局后，本函数对 UWB 测距做残差重映射），
        与用户审查标准 #5 “扰动一次定稿 — 只在 sim_materializer 注入”存在张力。
        sim_materializer 写协议轨迹与 G/K；A/N/V/M 场景轴仍可在 core_pipeline 注入。
        C1 冲突的彻底修复（把 G 轴几何扰动迁移到 sim_materializer）留待后续迭代，
        本任务不修改实际注入逻辑，避免引入新的科学语义偏移。
    """
    # TODO(C1): G 轴几何扰动应迁移到 sim_materializer，与用户审查标准 #5 对齐。
    # 当前实现：G/K 可在物化层按 axes_override 落盘；A/N/V/M 仍可在 core_pipeline 注入，
    # 彻底修复需要把 build_anchor_layout + 残差重映射整体迁移到物化层，
    # 并由 C1 冲突修复组统一处理场景扰动注入点迁移。
    if not gt_rows or not old_anchor_layout or not new_anchor_layout:  # 任意前置条件缺失时都不能做可靠重映射。
        return events  # 三个前置条件缺任意一个都不能算出可靠残差，所以直接原样返回。

    old_lookup = build_anchor_lookup(old_anchor_layout)  # 把旧布局转成可按锚点编号查几何位置的索引表。
    new_lookup = build_anchor_lookup(new_anchor_layout)  # 把新布局也转成同样结构，保证后面能做一一对照。
    old_anchor_ids = list(old_anchor_layout.get('anchor_ids') or [])  # 旧布局里声明过的锚点编号顺序，后面用于判断是否可以保留同名锚点。
    new_anchor_ids = list(new_anchor_layout.get('anchor_ids') or [])  # 新布局里声明过的锚点编号顺序，后面用于轮转分配新锚点。
    remapped: list[dict[str, Any]] = []  # 这里放的是改写后的完整事件流，返回时不会再额外压缩。
    for event in events:  # 逐条扫描事件，只有 UWB 事件才会被改写。
        updated = dict(event)  # 先对单条事件做浅拷贝，确保下面改写不会污染原始输入列表。
        payload = updated.get('uwb_payload')  # 只读取 UWB 观测载荷，其他模态的事件完全不参与这一步。
        if updated.get('modality') == MODALITY_UWB and isinstance(payload, dict):  # 只有 UWB 事件且载荷是字典时才做测距重映射。D9：模态名引用单源常量，禁止字面量漂移。
            # G7-H3: sim_materializer 已铺叠 GT 行 timestamp 到新时间轴，
            # 统一用事件时间 t 对齐，避免 source_t 与铺叠 GT 时间轴不匹配。
            # 旧策略优先用 meta.source_t 对齐，但 sim_materializer 铺叠后 GT 行 timestamp
            # 已偏移到新时间轴，source_t（旧时间轴）与 GT timestamp（新时间轴）不再对齐，
            # 会导致 align_ground_truth 找不到匹配行或对齐到错误的 GT 行。
            # 现 strategy：直接用事件 t（新时间轴）+ prefer_source_t=False（用 GT timestamp）。
            geometry_t = coerce_finite_scalar(updated['t'], name='geometry_t')  # D5：用 coerce_finite_scalar 替代 float()，拒绝 NaN/inf 时间戳污染下游对齐。
            gt_row, _ = align_ground_truth(gt_rows, geometry_t, prefer_source_t=False)  # 按事件时间找最近的真值行，后续残差就基于这行计算；prefer_source_t=False 与 sim_materializer 铺叠后的 GT timestamp 时间轴对齐。
            if gt_row is not None:  # 只有能对齐到真值时才有残差可迁移。
                source_anchor_id = payload.get('anchor_id')  # 先取原始锚点编号，看看新布局里能不能继续沿用这个身份。
                target_anchor_id = source_anchor_id  # 默认不改锚点编号，只有布局不兼容时才切到新的锚点。
                if new_anchor_ids:  # 新布局确实包含锚点编号时，才考虑按身份或轮转重映射。
                    preserve_identity = False  # 默认先不保留，只有新布局里确实还存在同名锚点时才沿用原身份。
                    try:  # 先试着在新布局里找同名锚点。
                        resolve_anchor_position(source_anchor_id, new_lookup)  # 只要新布局里还能解析这个同名锚点，就继续保留它。
                    except ValueError:  # 找不到同名锚点就说明不能保留身份。
                        preserve_identity = False
                    else:
                        preserve_identity = True
                    if not preserve_identity:  # 不能保留同名身份时，用确定性哈希分配新锚点。
                        # 用原始 anchor_id 的确定性哈希值做稳定分配，而非事件序号，
                        # 避免异步扰动删除部分 UWB 事件后导致 A-G 轴耦合。
                        # 注意：必须用 hashlib 而非内置 hash()，因为 Python 3.3+ 默认
                        # 启用 PYTHONHASHSEED 随机盐，内置 hash() 跨进程不确定。
                        anchor_hash = int(hashlib.md5(str(source_anchor_id).encode('utf-8')).hexdigest(), 16) if source_anchor_id is not None else 0
                        target_anchor_id = new_anchor_ids[anchor_hash % len(new_anchor_ids)]
                try:  # 同时查旧布局和新布局里的锚点位置。
                    old_anchor_position = resolve_anchor_position(source_anchor_id, old_lookup)  # 旧锚点的几何坐标，用来算原始理论距离。
                    new_anchor_position = resolve_anchor_position(target_anchor_id, new_lookup)  # 新锚点的几何坐标，用来算重映射后的理论距离。
                except ValueError:  # 任一布局查不到都把几何位置置空。
                    old_anchor_position = None  # 旧布局或新布局里任意一个锚点查不到时，就把几何坐标置空。
                    new_anchor_position = None  # 这样后面会自然跳过本次重映射，避免伪造距离。
                if old_anchor_position is not None and new_anchor_position is not None:  # 两边位置都拿到后才能计算残差迁移。
                    old_geometric_range = predict_range_to_anchor(gt_row, old_anchor_position)  # 先算旧布局下的理论距离，作为残差基线。
                    residual = coerce_finite_scalar(payload['range'], name='uwb_payload.range') - float(old_geometric_range)  # D5：观测距离用 coerce_finite_scalar 严格校验有限性，拒绝 NaN/inf 污染残差；old_geometric_range 来自 predict_range_to_anchor 已保证有限。
                    new_geometric_range = predict_range_to_anchor(gt_row, new_anchor_position)  # 再算新布局下的理论距离，用来重建观测。
                    new_measured_range = max(0.0, float(new_geometric_range) + residual)  # 把同一个残差叠回新理论距离，且下限钳成非负。
                    payload = dict(payload)  # 复制载荷后再改值，避免原始 dict 被多个事件共享引用。
                    payload['anchor_id'] = target_anchor_id  # 把观测目标切换成新锚点编号。
                    payload['range'] = new_measured_range  # 写入重映射后的测距值，这就是这一步唯一真正改写的数值。
                    updated['uwb_payload'] = payload  # 改完再塞回事件，保持事件整体结构不变。
        remapped.append(updated)  # 不管有没有发生重映射，事件都要写回结果序列，保证长度和顺序完全一致。
    return remapped  # 返回完整事件流；非 UWB 原样保留，UWB 则在几何约束下被重写。


def _materialize_scene_task(  # 把一个场景任务真正展开成可运行的事件和场景上下文。
    events: Iterable[Any],  # 已经过上游准备的事件序列,后面会在副本上做场景化处理。第 12 轮审查 LOW-1 修复补全类型注解。
    task: dict[str, Any],  # 当前任务本身,里面带着 scene_id、seq_id、axes 等控制信息。
    protocol_cfg: dict[str, Any],  # 场景轴协议配置,用来把任务里的简写参数展开成完整规则。
    scene_context: Mapping[str, Any] | None = None,  # 上游可选传入的快速场景上下文,存在时可以跳过部分重算。
    gt_rows: Mapping[str, Any] | list[dict[str, float]] | None = None,  # 上游可选传入 GT 行列表,用于 NLOS 状态依赖选择。第 8 阶段修复 HIGH-8。
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把一个场景任务真正展开成可运行的事件和场景上下文。"""
    scene_context = dict(scene_context or {})  # 先把上下文转成可修改的普通字典。
    if scene_context.get('quick_preprocessed_axes'):  # 快速路径已经预处理过轴时，直接复用现成结果。
        working_events = _clone_events(events)  # 快速路径只复制事件，不再重复做重型场景变换。
        resolved_scene_parameters = _resolve_scene_parameters(task, protocol_cfg)  # 仍然要把参数整理成统一结构。
        # 深拷贝而非浅拷贝：anchor_layout / geometry_report / scenario_reports 属于冻结几何合同与
        # 审计报告，浅拷贝会与上游 scene_context 共享引用，破坏 D3 anchor metadata 不可漂移合同，
        # 与 _resolve_scene_context_for_task 保持同口径隔离策略。
        return working_events, {  # 快速预处理分支只复制事件并返回已处理的上下文。
            'scene_parameters': resolved_scene_parameters,  # 返回标准化后的场景参数。
            'scenario_reports': deepcopy(scene_context.get('scenario_reports') or {}),  # 深拷贝已预处理报告，避免内部结构共享引用。
            'anchor_layout': deepcopy(scene_context.get('anchor_layout') or task.get('anchor_layout')),  # 有现成几何布局就深拷贝复用。
            'geometry_report': deepcopy(scene_context.get('geometry_report') or task.get('geometry_report')),  # 几何报告同样深拷贝沿用。
            'quick_preprocessed_axes': list(scene_context.get('quick_preprocessed_axes') or []),  # 记录哪些轴已经预处理过（轴名为字符串不可变，浅拷贝列表即可）。
        }
    return _apply_scene_task(events, task, protocol_cfg, gt_rows=gt_rows)  # 第 8 阶段修复 HIGH-8: 把 gt_rows 传给 _apply_scene_task，让 N 轴 _apply_nlos_level 能用状态依赖 NLOS 注入。


def _read_numeric_attribute(obj: Any, candidate_names: tuple[str, ...]) -> float | None:  # 从对象属性里按候选名找第一个数值字段。
    """从对象属性里按候选名找第一个数值字段。"""
    for candidate_name in candidate_names:  # 挨个候选名尝试，找到第一个可用数值就返回。
        value = getattr(obj, candidate_name, None)  # 每次只读一个候选字段。
        if is_numeric(value):  # 只有真数值才算资源字段。
            return coerce_finite_scalar(value, name=candidate_name)  # D5：用 coerce_finite_scalar 替代 float()，拒绝 NaN/Inf 污染下游 params/ram_peak 冻结口径；与 _resolve_runtime_override 同口径。
    return None  # 没有任何候选字段可用时返回空值。


def _read_runtime_resource_value(obj: Any, candidate_names: tuple[str, ...]) -> float | None:  # 优先从 runtime_resource_meta 里读资源数值，不行再回退到对象属性。
    """优先从 runtime_resource_meta 里读资源数值，不行再回退到对象属性。"""
    runtime_resource_meta = getattr(obj, 'runtime_resource_meta', None)  # 运行时资源元信息可能比对象属性更精确。
    if isinstance(runtime_resource_meta, Mapping):  # 元信息存在时优先读元信息。
        for candidate_name in candidate_names:  # 在元信息里逐个候选名查找。
            value = runtime_resource_meta.get(candidate_name)  # 先看元信息里有没有对应键。
            if is_numeric(value):  # 元信息里命中后直接返回。
                return coerce_finite_scalar(value, name=candidate_name)  # D5：与 _read_numeric_attribute / _resolve_runtime_override 同口径，拒绝 NaN/Inf 污染下游 params/ram_peak 冻结口径。
    return _read_numeric_attribute(obj, candidate_names)  # 元信息没有时再读对象属性。


def _resolve_runtime_override(cfg: dict[str, Any], task: dict[str, Any], method_name: str) -> dict[str, float]:  # 汇总配置和任务里所有运行时覆盖项，生成统一的 override 表。
    """汇总配置和任务里所有运行时覆盖项，生成统一的 override 表。"""
    overrides: dict[str, float] = {}  # 最终覆盖结果，按后写覆盖前写合并。
    for candidate in (cfg.get('runtime_override'), task.get('runtime_override')):  # 先合并单点覆盖。
        if isinstance(candidate, Mapping):  # 单点覆盖只保留映射型输入。
            overrides.update(  # 把显式 runtime_override 合并进覆盖表。
                {  # 只保留数值型覆盖项，避免把字符串或布尔误当配置值。
                    key: coerce_finite_scalar(value, name=f"runtime_override.{key}")  # D5：用 coerce_finite_scalar 替代 float()，拒绝 NaN/Inf 污染下游 params/ram_peak 冻结口径。
                    for key, value in candidate.items()  # 遍历每个覆盖项。
                    if is_real(value)  # D6：与分组覆盖同口径，统一用 is_real 排除 bool 和 complex，避免单点窄集合、分组宽集合的非对称漂移。
                }  # 数值覆盖字典结束。
            )  # 覆盖表合并完成。

    runtime_overrides = cfg.get('runtime_overrides')  # 按任务或方法分组的覆盖表。
    if isinstance(runtime_overrides, Mapping):  # 再合并按任务、场景或方法分组的覆盖。
        for lookup_key in (task.get('task_id'), task.get('scene_id'), task.get('seq_id'), method_name):  # 按任务、场景、序列和方法名回退。
            if lookup_key in runtime_overrides and isinstance(runtime_overrides[lookup_key], Mapping):  # 命中分组覆盖表时再合并。
                overrides.update(  # 再合并按任务、场景或方法命中的细粒度覆盖。
                    {  # 仍然只保留数值型项。
                        key: coerce_finite_scalar(value, name=f"runtime_overrides.{lookup_key}.{key}")  # D5：用 coerce_finite_scalar 替代 float()，拒绝 NaN/Inf 污染下游 params/ram_peak 冻结口径。
                        for key, value in runtime_overrides[lookup_key].items()  # 遍历命中的覆盖子表。
                        if is_real(value)  # D6：与单点覆盖同口径，排除 bool 和 complex，保证两层覆盖合并行为一致。
                    }  # 细粒度覆盖字典结束。
                )  # 细粒度覆盖合并完成。
    return overrides  # 返回合并后的覆盖表，供 runtime log 计算使用。


def _resolve_runtime_log(  # 汇总 latency、params、ram_peak 等运行时日志字段。
    runtime_collector: _RuntimeTraceCollector,  # 当前任务对应的耗时收集器。
    estimator: Any,  # 当前任务实际使用的估计器对象。
    model: Any,  # 当前任务实际使用的模型对象，可能为 None。
    cfg: dict[str, Any],  # 流水线配置，里面可能带 runtime 覆盖项。
    task: dict[str, Any],  # 当前场景任务，用来查覆盖表。
    method_name: str,  # 当前实际执行的方法名。
) -> dict[str, Any]:  # 返回 latency / params / ram_peak 的 runtime 日志。
    """汇总 runtime 相关的 latency、params、ram_peak。"""
    overrides = _resolve_runtime_override(cfg, task, method_name)  # 先拿到所有手工覆盖。
    latency_trace_ms = list(runtime_collector.latency_trace_ms)  # 复制一份延时轨迹，避免后续误改。
    if not latency_trace_ms:  # 没有延时轨迹就说明没有有效运行数据。
        raise ValueError('runtime latency trace must not be empty')  # 没有延时轨迹就说明没有有效运行数据。

    param_values = [_read_runtime_resource_value(obj, _PARAM_ATTR_CANDIDATES) for obj in (estimator, model) if obj is not None]  # 依次从 estimator/model 读参数量。
    params = overrides.get('params')  # 允许配置显式覆盖参数量。
    if params is None:  # 没有显式覆盖时从对象上合计参数量。
        params = sum(value for value in param_values if value is not None) if any(value is not None for value in param_values) else 0.0  # 没有覆盖时就从对象上合计。

    ram_values = [_read_runtime_resource_value(obj, _RAM_ATTR_CANDIDATES) for obj in (estimator, model) if obj is not None]  # 依次从 estimator/model 读峰值内存。
    ram_peak = overrides.get('ram_peak')  # 允许配置显式覆盖内存峰值。
    if ram_peak is None:  # 没有显式覆盖时取估计器和模型中较大的峰值内存。
        ram_peak = max((value for value in ram_values if value is not None), default=0.0)  # 没有覆盖时取两者中的最大值。

    return {  # 返回统一格式的运行时日志，供后续统计和审计。
        'latency': latency_trace_ms,  # 延时轨迹原样输出。
        'params': coerce_finite_scalar(params, name='runtime_log.params'),  # D5：用 coerce_finite_scalar 替代 float()，拒绝 NaN/Inf 经 override 累加或 sum 漂移后静默泄漏到冻结的 params 口径；与 _resolve_runtime_override / _read_runtime_resource_value 同口径。
        'ram_peak': coerce_finite_scalar(ram_peak, name='runtime_log.ram_peak'),  # D5：用 coerce_finite_scalar 替代 float()，拒绝 NaN/Inf 经 override 或 max 漂移后静默泄漏到冻结的 ram_peak 口径；与 _resolve_runtime_override / _read_runtime_resource_value 同口径。
    }


def _deep_merge_model_cfg(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:  # 递归深合并模型配置，override 覆盖 base 的同名键。
    """递归深合并两个模型配置字典，override 中的值覆盖 base 中的同名键。

    合并语义：
    - 双方同名键均为 Mapping 时递归合并
    - 否则 override 的值深拷贝覆盖 base 的值
    - 返回的字典与输入完全独立（深拷贝），修改返回值不会影响原对象
    """
    merged: dict[str, Any] = {}  # 从空字典开始构建，避免与 base 共享嵌套可变引用。
    for key, base_value in base.items():
        if key not in override:
            merged[key] = deepcopy(base_value)  # base 中独有的字段深拷贝，避免与 base 共享可变引用。
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):  # 双方同名键都是映射时递归合并。
            merged[key] = _deep_merge_model_cfg(base[key], value)  # 递归合并嵌套字典，递归内部从空 dict 构建保证独立。
        else:
            merged[key] = deepcopy(value)  # 标量或列表直接深拷贝覆盖。
    return merged


def _resolve_model_cfg(cfg: dict[str, Any], method_name: str) -> dict[str, Any]:  # 按方法名解析模型配置，先用内存覆盖与默认 YAML 深合并，找不到再去默认 YAML。
    """按方法名解析模型配置，显式覆盖与默认 YAML 深合并，无覆盖时回退默认 YAML。

    显式 model_cfgs[method_name] 中的字段会深合并覆盖到默认模型 YAML 之上，
    允许实验级配置只覆盖部分子字段（如 safe_mode）而不丢失默认配置的
    feature_order、window、network 等必填项。
    """
    default_cfg_path = _MODEL_CFG_ROOT / f'{method_name}.yaml'  # 约定的默认模型 YAML 路径。
    base_cfg = load_yaml_config(default_cfg_path)  # 先加载默认配置作为基础。
    model_cfgs = cfg.get('model_cfgs') or {}  # 用户可能已经把模型配置放在 pipeline_cfg 里。
    if isinstance(model_cfgs, Mapping) and isinstance(model_cfgs.get(method_name), Mapping):  # 显式配置命中时深合并。
        override = deepcopy(model_cfgs[method_name])  # 命中时全隔离深拷贝，避免与 cfg['model_cfgs'][method_name] 共享 safe_mode、window、network、feature_order 等嵌套可变引用，防止跨 task/method 循环调用累积污染；与 _resolve_estimator_cfg L647、_resolve_method_route L629 同口径。
        return _deep_merge_model_cfg(base_cfg, override)  # 深合并覆盖到默认配置上。
    return base_cfg  # 无显式覆盖时返回默认配置。


def _resolve_method_route(method_name: str) -> dict[str, Any]:  # 把方法名映射成方法路由信息，告诉流水线该用哪个 estimator/model。
    """把方法名映射成方法路由信息，告诉流水线该用哪个 estimator/model。"""
    if method_name in _CLASSICAL_METHODS:  # 经典方法只需要 estimator。
        # §10.2 第 2 行守卫：检测神经关键字被误注册进经典方法集合。
        for neural in _NEURAL_METHODS:
            if neural in method_name:
                raise ValueError(
                    f'§10.2 第 2 行守卫：方法名 "{method_name}" 含神经关键字 "{neural}" '
                    f'但被注册在 _CLASSICAL_METHODS 集合中；纯 NN 主表无 EKF 外壳被拒绝。'
                )
        return {'method_name': method_name, 'estimator_name': method_name, 'model_name': None}  # 经典方法只需要 estimator。
    if _is_neural_method(method_name):  # 神经方法复用 EKF 外壳（E9 配置用 lstm_ekf/liquid_ekf/transformer_ekf 全名，_is_neural_method 后缀匹配）。
        return {'method_name': method_name, 'estimator_name': ESTIMATOR_NAME_EKF, 'model_name': method_name}  # 神经方法复用 EKF 外壳。# D9：引用单源常量 ESTIMATOR_NAME_EKF，禁止本地 'ekf' 字面量漂移。
    raise ValueError(f'Unsupported method: {method_name}')  # 未知方法属于值合同违例，与 model_factory/estimator_factory 同口径用 ValueError，避免静默跑偏。


def _resolve_estimator_cfg(  # 按估计器名解析配置，优先使用显式覆盖，再回退默认文件。
    cfg: dict[str, Any],  # 当前流水线配置，可能包含显式覆盖项。
    estimator_name: str,  # 估计器名称决定优先读取哪个配置文件。
    *,
    anchor_layout: dict[str, Any] | None = None,  # 可选锚点布局，会影响几何相关初始化。
) -> dict[str, Any]:  # 返回解析后的估计器配置。
    """解析估计器配置，并在需要时把锚点布局注入进去。"""
    estimator_cfgs = cfg.get('estimator_cfgs') or {}  # 先查内存里的估计器配置覆盖表。
    if isinstance(estimator_cfgs, Mapping) and isinstance(estimator_cfgs.get(estimator_name), Mapping):  # 显式配置命中时优先返回。
        estimator_cfg = deepcopy(estimator_cfgs[estimator_name])  # 命中时全隔离深拷贝，避免与 cfg['estimator_cfgs'] 共享 cov 矩阵、process_noise、anchor_layout_metadata 等嵌套可变引用，防止跨 task/method 循环调用累积污染。
    else:  # 显式配置没命中时回退默认 YAML。
        default_cfg_path = _MODEL_CFG_ROOT / f'{estimator_name}.yaml'  # 默认估计器 YAML 路径。
        estimator_cfg = load_yaml_config(default_cfg_path)  # 回退到磁盘默认估计器配置。
    if anchor_layout is not None:  # 只在有布局时注入几何上下文。
        estimator_cfg['anchor_layout'] = deepcopy(anchor_layout)  # 注入锚点布局副本，避免后续被误改。
    return estimator_cfg  # 返回最终估计器配置。


def _build_feature_window_builder(model_cfg: Mapping[str, Any], estimator: Any):  # 构造神经方法用的窗口特征构建器。
    """构造神经方法用的窗口特征构建器。

    Args:  # 参数说明开始。
        model_cfg: 模型配置，里面至少要有 `feature_order` 和窗口配置。
        estimator: 当前任务用的估计器对象，特征构建时可能要读它的锚点查找表。

    Returns:  # 返回值说明开始。
        一个闭包构建器，供融合主链路按历史事件生成特征窗口。
    """
    feature_order = list(model_cfg.get('feature_order') or [])  # 特征顺序决定特征向量的列排列。
    if not feature_order:  # 没有特征顺序就根本无法构窗。
        raise ValueError('model_cfg.feature_order must be non-empty for neural methods')  # 没有特征顺序就无法构窗。

    window_cfg = dict(model_cfg.get('window') or {})  # 读取滑窗配置。
    if 'size' not in window_cfg:  # §10.4 训推对等守卫续补：缺 size 时 raise 注入守卫，禁止默认值兜底静默破坏对等。
        raise KeyError(
            f"§10.4 NN 截断对等 watchdog: model_cfg.window.size 必须显式声明, "
            f"禁止默认值兜底（spec §10.4 记忆深度对等前提, 训推同侧）."
        )
    _window_size_raw = window_cfg['size']
    if not is_integer(_window_size_raw):
        raise TypeError(f"window.size must be an integer, got {type(_window_size_raw).__name__}")
    if int(_window_size_raw) <= 0:
        raise ValueError(f"window.size must be a positive integer, got {_window_size_raw!r}")
    window_size = int(_window_size_raw)  # 滑窗长度。D7：移除未使用的 step_size（window_cfg.get('step') 全程未被消费，属死代码；当前 trailing window 实现不依赖步长，仅取尾部 window_size 条事件）。
    anchor_lookup = dict(getattr(estimator, '_anchor_lookup', {}) or {})  # 优先用估计器自己已有的锚点查找表。
    if not anchor_lookup:  # 优先用估计器自带的查找表，没有才尝试从配置构造。
        anchor_layout = getattr(estimator, 'cfg', {}).get('anchor_layout') if hasattr(estimator, 'cfg') else None  # 若估计器配置里带布局，就临时构造查找表。
        if anchor_layout is not None:  # 只有真的有布局才构造查找表。
            anchor_lookup = build_anchor_lookup(anchor_layout)  # 从布局生成查找表。

    def _builder(history, event, cfg):  # 这个闭包真正负责把历史事件转成当前窗口。
        """根据历史事件和状态构造当前模型输入窗口。"""
        state_history = list((cfg or {}).get('state_history') or [])  # 可选的状态历史输入。
        if state_history and len(state_history) != len(history):  # 显式状态历史必须和事件历史一一对齐。
            raise ValueError('state_history must align with history length')  # 长度不一致说明输入已经错位。
        if not state_history:  # 没有显式状态历史时，就用估计器当前状态填一份默认上下文。
            state_ctx = {}  # 没有显式状态历史时，用当前估计状态补一份默认上下文。
            if hasattr(estimator, 'get_state'):  # 估计器如果提供状态接口，就优先读它。
                state_estimate = estimator.get_state()  # 取估计器当前状态。
                _state_payload = getattr(state_estimate, 'state', None)  # 取 state 载荷。
                if isinstance(_state_payload, Mapping):  # 仅当 state 是映射类型时才转字典，避免 list/tuple/str 触发 dict() 异常。
                    for _k, _v in dict(_state_payload).items():  # D5：逐字段用 coerce_finite_scalar 一步到位完成数值转换+有限性校验+tensor/.item() 处理，替代 float()+isfinite 两步式写法；过滤非数值/非有限值，防止 NaN/inf 或 tensor 泄漏。
                        try:
                            _sv = coerce_finite_scalar(_v, name=f"state.{_k}")
                        except (TypeError, ValueError):
                            continue
                        state_ctx[str(_k)] = _sv
            state_history = [state_ctx] * len(history)  # 用同一状态填满历史长度，保证下游能对齐。
        row_count = len(history)  # 当前已经积累的历史长度。
        window_start = max(0, row_count - window_size)  # 当前样本只消费尾部 trailing window。
        window_index_map = list(range(window_start, row_count))  # 保证当前事件总在窗口最后一行。
        window_history = [history[index] for index in window_index_map]  # 只保留当前窗口对应的事件子序列。
        window_state_history = [state_history[index] for index in window_index_map]  # 只保留当前窗口对应的状态子序列。
        feature_state_history = build_feature_state_history(  # 把窗口内事件和状态拼成特征所需的中间上下文。
            window_history,
            window_state_history,
            anchor_lookup=anchor_lookup or None,
        )

        feature_rows = [  # 每条历史事件都会变成一行特征，后面再统一抽成矩阵。
            build_feature_vector(hist_event, hist_state_ctx, feature_order)  # 用当前事件、对齐后的状态上下文和固定特征顺序生成一行特征。
            for hist_event, hist_state_ctx in zip(window_history, feature_state_history, strict=True)  # 严格一一对应遍历窗口内事件和状态历史，避免错位。
        ]  # 列表推导结束，得到按时间顺序排列的特征行集合。
        feature_matrix = [list(row['feature_values']) for row in feature_rows]  # D3：逐行 list() 拷贝隔离 feature_rows 内部列表，避免返回的 feature_window 与 build_feature_vector 返回值共享可变引用，防止下游消费者修改窗口行时反向污染 current_row['feature_values']。
        missing_matrix = [list(row['missing_mask']) for row in feature_rows]  # D3：同上，缺失掩码窗口也逐行隔离。
        current_row = feature_rows[-1]  # 当前时刻对应的是窗口最后一行特征。
        feature_window = feature_matrix  # 当前 builder 只返回当前事件实际消费的 trailing window。
        missing_window = missing_matrix  # 缺失掩码窗口与特征窗口逐行对齐。
        event_time_window = [coerce_finite_scalar(window_history[index]['t'], name='event.t') for index in range(len(window_history))]  # D5：用 coerce_finite_scalar 严格校验时间戳有限性，禁止 NaN/inf 泄漏到窗口时间轴和后续 dt 推导。
        if len(event_time_window) >= 2:  # 只要窗口里已经有相邻事件，就优先用窗口首步真实时间差。
            window_dt = max(0.0, event_time_window[1] - event_time_window[0])  # 显式 dt 应与 event_time_window 推导出的首步递推间隔一致。
        elif window_history:  # 单步窗口没有相邻时间差时，才回退到当前唯一事件自带的 dt。
            window_dt = coerce_finite_scalar(window_history[0]['dt'], name='event.dt')  # D5：单步窗口仍沿用事件自带的显式步长，并严格校验有限性，禁止 NaN/inf 进入 Liquid step_dts 首步 fallback。
        else:
            window_dt = coerce_finite_scalar(event['dt'], name='event.dt')  # D5：理论兜底——空窗口时退回当前事件步长，并严格校验有限性，与上游 validate_event 的 dt 非负+有限约束对齐。

        return {  # 返回当前模型输入窗口所需的全部结构化字段。
            'current_modality': event['modality'],  # 当前事件模态，用于模型判断输入分支。
            'dt': window_dt,  # 显式 dt 代表窗口首步步长，供 Liquid step_dts 的首步 fallback 使用。
            'feature_order': list(feature_order),  # 特征列顺序，供模型对齐使用。
            'feature_values': list(current_row['feature_values']),  # 当前时刻的单步特征向量。
            'missing_mask': list(current_row['missing_mask']),  # 当前时刻的缺失掩码。
            'feature_window': feature_window,  # 滑窗内的特征矩阵。
            'missing_mask_window': missing_window,  # 滑窗内的缺失掩码矩阵。
            'window_index_map': window_index_map,  # 窗口对应原始历史索引。
            'event_time_window': event_time_window,  # 窗口中的原始时间戳列表。
        }  # 结束窗口输入字典。

    return _builder  # 返回闭包，供神经方法在推理阶段动态构窗。


class CorePipeline(PipelineAPI):  # 把场景生成、估计器/模型推理、融合、运行日志和产物落盘串成一条完整链路。
    """把场景生成、估计器/模型推理、融合、运行日志和产物落盘串成一条完整链路。"""

    def run(self, pipeline_cfg: dict | None = None, runtime_context: dict | None = None) -> StageResult:  # 执行核心实验流水线，并返回包含产物路径和元数据的阶段结果。
        """执行核心实验流水线，并返回包含产物路径和元数据的阶段结果。"""
        cfg = normalize_pipeline_cfg(pipeline_cfg)  # 先把外部配置转成普通字典，并拒绝非映射输入。
        # 第 10 轮审查 MEDIUM-1 修复：CorePipeline.run 是核心流水线入口函数，
        # 每场景任务调用一次。入口 print_dict 违反工程规范
        # "Recursive functions and entry points should avoid print_dict calls"，
        # StageResult 返回值与运行日志已包含完整入口参数明细。
        experiment_cfg = dict(cfg.get('experiment_cfg') or {})  # 单独提取实验配置，方便回退读取。
        # 与 methods 回退逻辑一致：顶层显式配置优先，experiment_cfg 回退补充。
        if not cfg.get('model_cfgs') and experiment_cfg.get('model_cfgs'):  # model_cfgs 回退：顶层缺失时从 experiment_cfg 补充。
            cfg['model_cfgs'] = experiment_cfg['model_cfgs']
        if not cfg.get('estimator_cfgs') and experiment_cfg.get('estimator_cfgs'):  # estimator_cfgs 回退：顶层缺失时从 experiment_cfg 补充。
            cfg['estimator_cfgs'] = experiment_cfg['estimator_cfgs']
        if not cfg.get('scene_axis_protocol_path') and experiment_cfg.get('scene_axis_protocol_path'):  # scene_axis_protocol_path 回退：顶层缺失时从 experiment_cfg 补充。
            cfg['scene_axis_protocol_path'] = experiment_cfg['scene_axis_protocol_path']
        if not cfg.get('sweep_cfgs') and experiment_cfg.get('sweep_cfgs'):  # sweep_cfgs 回退：顶层缺失时从 experiment_cfg 补充。
            cfg['sweep_cfgs'] = experiment_cfg['sweep_cfgs']
        scene_protocol_cfg = load_scene_axis_protocol(cfg.get('scene_axis_protocol_path'))  # 读取场景轴协议。
        methods = list(cfg.get('methods') or experiment_cfg.get('methods') or [])  # 方法列表优先取显式配置。
        if not methods:  # 没有方法就无法跑任何实验分支。
            raise ValueError('methods must be a non-empty list')  # 没有方法就无法跑任何实验分支。
        # §10.4 NN 截断对等 watchdog：在任务循环开始前对所有神经方法的 window.size 做对等校验，
        # 不一致时 fail-loud（守卫必须在任何估计器步进之前挂载，防止先炸在无关路径上）。
        _enforce_window_size_parity_for_neural_methods(methods, cfg)

        scene_tasks = list(cfg.get('scene_tasks') or [])  # 先看是否已经有现成任务清单。
        if not scene_tasks:  # 没有显式任务时尝试从实验配置或事件中回退生成。
            if experiment_cfg:  # 有实验配置时优先按配置采样场景。
                scene_tasks = sample_scenes(experiment_cfg, cfg.get('sweep_cfgs'))  # 没有任务时按实验配置采样。
            elif isinstance(cfg.get('events'), list):  # 单事件输入时从首条事件推导最小任务。
                first = cfg['events'][0]  # 退回到单事件输入时，从第一条事件里推导最小任务。
                first_meta = first.get('meta', {}) if isinstance(first, dict) else {}  # 防御性取 meta。
                scene_tasks = [{  # 单事件输入时构造一个最小任务，确保下游仍能按任务链路执行。
                    'task_id': 'scene_00',  # 单事件输入默认构造一个基础任务编号。
                    'scene_id': first_meta.get('scene_id', _get_default_scene_code()),  # 场景编号从事件元信息提取，缺失时从协议动态获取默认值。
                    'seq_id': first_meta.get('seq_id', 'unknown'),  # 序列编号也从元信息提取，缺失时用默认值。
                    'axes': cfg.get('axes', {}),  # 轴配置直接沿用顶层输入。
                }]
            else:  # 三种入口都没有时直接报错。
                raise ValueError('Either scene_tasks, experiment_cfg, or events must be provided')  # 三种入口至少要有一种。
        scene_tasks = deepcopy(scene_tasks)  # 复制任务表，避免后续写回污染原始配置。

        # §8.3 单轨片段覆盖门禁：实验 sweep 必须覆盖全部 7 项必需片段类型。
        # 仅在通过 experiment_cfg 采样场景（即 sweep 路径）时启用；单事件或显式 scene_tasks
        # 输入路径不参与该门禁（测试/快速裁剪场景可能仅含部分片段，不应阻断运行）。
        if experiment_cfg and cfg.get('scene_tasks') is None and not isinstance(cfg.get('events'), list):
            _assert_experiment_segment_coverage(scene_tasks, experiment_cfg)
            if bool(experiment_cfg.get('enable_section8_deep_audit', True)):
                # Build synthetic cfg keys from real pipeline data so the 6 cfg-keyed
                # gates inside _run_section8_deep_audit actually fire (otherwise they
                # would be silent-skipped because cfg never contains those keys).
                # See _build_section8_cfg_payload docstring for the field-by-field plan.
                audit_payload = _build_section8_cfg_payload(scene_tasks, cfg, experiment_cfg)
                merged_cfg = dict(cfg or {})
                merged_cfg.update(audit_payload)
                _run_section8_deep_audit(scene_tasks, merged_cfg)

        output_root = _resolve_output_root(cfg, 'core_pipeline')  # 统一解析输出根目录。
        predictions_dir = output_root / 'predictions'  # 预测 bundle 的落盘目录。
        logs_dir = output_root / 'logs'  # 日志目录。
        audits_dir = output_root / 'audits'  # 审计索引目录。
        plotting_dir = output_root / 'plotting_inputs'  # 提供给绘图/统计的中间输入目录。
        predictions_dir.mkdir(parents=True, exist_ok=True)  # 逐个目录确保存在。
        logs_dir.mkdir(parents=True, exist_ok=True)  # 日志目录同样保证可写。
        audits_dir.mkdir(parents=True, exist_ok=True)  # 审计目录保证可写。
        plotting_dir.mkdir(parents=True, exist_ok=True)  # 绘图输入目录保证可写。

        artifacts: list[str] = []  # 汇总所有产物路径，StageResult 里要返回。
        prediction_index: list[dict[str, Any]] = []  # 预测索引表，供下游快速定位文件。
        retain_prediction_bundles = bool(cfg.get('retain_prediction_bundles', True))  # 默认保留，必要时可关闭以降低内存占用。
        in_memory_bundles: list[dict[str, Any]] = [] if retain_prediction_bundles else []  # 仅在需要时保留内存 bundle。
        runtime_rows: list[dict[str, Any]] = []  # 运行时统计表，供绘图和对比分析。
        for task in scene_tasks:  # 逐个场景任务跑完整条估计/融合链。
            events = _resolve_events_for_task(task, cfg)  # 解析当前任务对应的事件序列。
            scene_context = _resolve_scene_context_for_task(task, cfg)  # 解析快速场景上下文。
            source_report = _resolve_source_report_for_task(task, cfg)  # 解析来源报告。
            ground_truth_rows = _resolve_ground_truth_rows_for_task(task, cfg)  # 解析对应真值。
            scenario_events, scenario_context = _materialize_scene_task(events, task, scene_protocol_cfg, scene_context, gt_rows=ground_truth_rows)  # 第 8 阶段修复 HIGH-8: 把 ground_truth_rows 传给 materialize_scene_task, 让 N 轴 _apply_nlos_level 能用状态依赖 NLOS 注入。
            if not scene_context.get('quick_preprocessed_axes'):  # 只有非快速预处理场景才需要做几何投影和重映射。
                source_anchor_layout = source_report.get('anchor_layout')  # 若来源里有旧锚点布局，后面用于几何对齐。
                if scenario_context.get('anchor_layout') is not None and isinstance(source_anchor_layout, Mapping):  # 当前布局和来源布局都存在时才投影。
                    scenario_context = dict(scenario_context)  # 复制一份上下文，准备就地替换布局。
                    scenario_context['anchor_layout'] = project_anchor_layout_to_reference(  # 先把当前布局投影到来源参考系，保证后面的几何对齐仍然和旧报告一致。
                        scenario_context['anchor_layout'],  # 当前场景里的布局是待投影对象，投影后会改写成参考系中的坐标表达。
                        source_anchor_layout,  # 来源报告里的旧布局是参考系目标，用来保证两个链路里的锚点语义一致。
                    )  # 投影完成后再写回上下文，避免后面的重映射拿错参考系。
                if scenario_context.get('anchor_layout') is not None and ground_truth_rows is not None:  # 有新布局且有真值时才可能重映射 UWB。
                    if isinstance(source_anchor_layout, Mapping):  # 来源布局也必须是映射，才能找锚点几何位置。
                        scenario_events = _remap_uwb_ranges_for_geometry(  # 当几何参考系变化时，把旧测距的残差逻辑同步迁移到新布局。
                            scenario_events,  # 当前场景事件序列是重映射的对象，返回后会用新测距整体替换旧测距。
                            ground_truth_rows,  # 与事件时间对齐后的真值行是残差基准，没有它就无法稳定重算。
                            source_anchor_layout,  # 原始锚点布局用于解释旧测距对应的几何关系。
                            scenario_context.get('anchor_layout'),  # 当前场景的新锚点布局是重映射后的目标参考系。
                        )  # 重映射结束后得到新的事件序列，再继续往后跑。
                for event in scenario_events:  # 把修正后的场景事件逐个写回 meta。
                    meta = event.get('meta')  # 每个事件的 meta 都要尽量和当前任务统一。
                    if isinstance(meta, Mapping):  # 只有映射型 meta 才能安全写字段。
                        meta = dict(meta)  # 复制 meta，避免原事件对象受影响。
                        if task.get('scene_id') is not None:  # 当前任务有 scene_id 时就写回。
                            meta['scene_id'] = task['scene_id']  # 场景编号写回当前任务编号。
                        if task.get('seq_id') is not None:  # 当前任务有 seq_id 时同步写回。
                            meta['seq_id'] = task['seq_id']  # 序列编号同步写回。
                        if scenario_context.get('anchor_layout') is not None:  # 当前布局存在时一并写入。
                            meta['anchor_layout'] = deepcopy(scenario_context['anchor_layout'])  # 当前锚点布局也写入事件元信息。第 15 轮审查 LOW-2 修复（R15-C LOW-2）：原实现浅引用共享 scenario_context['anchor_layout']，违反 D3 anchor metadata 不可漂移合同。同文件 L201/L206/L211/L216/L226/L233/L239 均对 anchor_layout 做 deepcopy 隔离，此处遗漏。for event in scenario_events 循环中每个事件的 meta['anchor_layout'] 共享同一引用，下游 run_fusion 若修改会跨事件污染。改为 deepcopy 与全文件口径对齐。
                        event['meta'] = meta  # 把修正后的 meta 放回事件。
            for method_name in methods:  # 同一任务下对所有方法逐个跑。
                route = _resolve_method_route(method_name)  # 把方法名展开成具体路由。
                resolved_method_name = route['method_name']  # 实际执行的方法名。
                estimator_name = route['estimator_name']  # 对应估计器名。
                model_name = route['model_name']  # 对应模型名，经典方法这里是 None。
                # D11: 提前跳过已存在的预测，避免重复推理（断点续评估）。
                _early_out_path = predictions_dir / f"{validate_path_component(task['task_id'], name='task_id')}__{validate_path_component(method_name, name='method_name')}.json"
                if _early_out_path.exists():
                    continue
                if model_name is not None:  # 神经方法需要模型和 EKF 外壳一起工作。
                    estimator = create_estimator(  # 神经方法先创建 EKF 外壳估计器，让神经模型只负责输出中间策略量。
                        ESTIMATOR_NAME_EKF,  # D9：神经方法统一复用 EKF 外壳，引用单源常量禁止本地 'ekf' 字面量漂移，与 L633 同口径。
                        _resolve_estimator_cfg(  # 先把 EKF 外壳的配置解析出来，后面创建估计器要直接吃这个结果。
                            cfg,  # 当前流水线配置里可能包含显式覆盖项。
                            ESTIMATOR_NAME_EKF,  # D9：解析 EKF 外壳配置时引用单源常量，禁止本地 'ekf' 字面量漂移，与 L633 同口径。
                            anchor_layout=task.get('anchor_layout') or scenario_context.get('anchor_layout'),  # 当前任务可用的锚点布局会影响几何初始化。
                        ),  # 结束 EKF 配置解析，结果直接喂给 create_estimator。
                    )  # EKF 外壳估计器创建完成后，后面再叠加神经模型。
                    model_cfg = _resolve_model_cfg(cfg, model_name)  # 读取模型配置。
                    model = create_model(model_name, model_cfg)  # 按模型名创建具体网络。
                    if hasattr(model, 'eval'):  # 如果模型支持 eval，就切到推理态。
                        model.eval()  # 推理阶段切到 eval 模式。
                    feature_builder = _build_feature_window_builder(model_cfg, estimator)  # 先准备窗口特征构建器，让神经方法在推理时按同一套模型配置构窗。
                elif estimator_name in _CLASSICAL_METHODS:  # 经典方法只需要 estimator。
                    estimator = create_estimator(  # 经典方法只建 estimator，不额外加载神经模型。
                        estimator_name,  # 经典方法直接使用自己的名称创建估计器，避免再做别名映射。
                        _resolve_estimator_cfg(  # 先解析该估计器对应的专用配置。
                            cfg,  # 当前流水线配置可能覆盖默认文件。
                            estimator_name,  # 经典方法的估计器名就是它自己。
                            anchor_layout=task.get('anchor_layout') or scenario_context.get('anchor_layout'),  # 当前任务可用的锚点布局会影响初始化几何。
                        ),  # 结束估计器配置解析，结果直接传给 create_estimator。
                    )  # 经典估计器创建完成后，不再补模型对象。
                    model = None  # 经典方法没有独立神经模型。
                    feature_builder = None  # 所以也不需要窗口特征构建器。
                else:  # 其余分支仍按估计器处理。
                    estimator = create_estimator(  # 其余分支仍按估计器处理，保持路由语义不变。
                        estimator_name,  # 直接按路由给出的估计器名创建，避免把别的分支硬塞进来。
                        _resolve_estimator_cfg(  # 继续走同一套配置解析流程。
                            cfg,  # 当前流水线配置决定最终覆盖关系。
                            estimator_name,  # 具体估计器名由路由给出。
                            anchor_layout=task.get('anchor_layout') or scenario_context.get('anchor_layout'),  # 当前任务可用的锚点布局仍然是几何初始化关键输入。
                        ),  # 结束估计器配置解析，返回给 create_estimator。
                    )  # 兜底估计器分支完成。
                    model = None  # 没有模型对象。
                    feature_builder = None  # 也没有特征构建器。

                runtime_collector = _RuntimeTraceCollector()  # 这个任务方法单独一个耗时收集器。
                timed_estimator = _TimedEstimatorProxy(estimator, runtime_collector)  # 给 estimator 套耗时代理。
                timed_model = _TimedModelProxy(model, runtime_collector) if model is not None else None  # 有模型时才套耗时代理。
                if feature_builder is not None:  # 构窗时尽量用计时后的 estimator。
                    feature_builder = _build_feature_window_builder(model_cfg, timed_estimator)  # 构窗时尽量用计时后的 estimator，保证耗时统计覆盖真实推理链路。

                with torch.no_grad():  # 推理阶段禁用梯度计算，防止显存泄漏。
                    # 三角定位必须使用与 UWB 范围同坐标系的 anchor_layout：
                    # UWB 范围已被 _remap_uwb_ranges_for_geometry 重映射到 scenario_context 的投影坐标系，
                    # 因此三角定位应使用 scenario_context 的 anchor_layout（投影后），而非 source_report 的原始布局。
                    triangulation_anchor = scenario_context.get('anchor_layout')
                    bundle = run_fusion(  # 把场景事件、估计器、模型和特征构建器一起送进融合主链路。
                        scenario_events,  # 场景事件流是融合主链路的输入，决定本次推理的时间顺序。
                        timed_estimator,  # 套了计时代理的 estimator，用来同步收集耗时。
                        timed_model,  # 套了计时代理的 model，可能为 None，经典方法不走这条。
                        feature_builder=feature_builder,  # 可选的窗口特征构建器，神经方法才会真正用到。
                        cfg={  # 传给融合链路的最小方法标识和桥接配置，只保留它必须知道的信息。
                            'method_name': method_name,
                            'resolved_method_name': resolved_method_name,
                            'safe_mode': deepcopy(model_cfg.get('safe_mode') or {}) if model_name is not None else None,  # 第 15 轮审查 LOW-8 修复（R15-C LOW-8）：原实现 dict() 浅拷贝，嵌套可变结构会跨 task 共享污染 model_cfg['safe_mode']，违反 D3 不可漂移合同。model_cfg 来自 _resolve_model_cfg 跨 task 共享，run_fusion 内部修改嵌套字段会反向污染。改为 deepcopy 与 _resolve_estimator_cfg/_resolve_model_cfg 口径对齐。
                            'anchor_layout': deepcopy(triangulation_anchor) if triangulation_anchor is not None else None,  # 三角定位用 scenario_context 投影后的 anchor_layout（与重映射后的 UWB 范围同坐标系），而非 source_report 的原始布局。
                        },
                    )  # 调用融合主链路后得到最终 bundle。
                bundle['scene_id'] = task['scene_id']  # 强制写回当前任务的场景编号。
                if task.get('seq_id') is not None:  # 有序列编号时同步写回。
                    bundle['seq_id'] = task['seq_id']  # 有序列编号时同步写回。
                if task.get('repeat_id') is not None:  # 有重复编号时也保留。
                    bundle['repeat_id'] = task['repeat_id']  # 重复编号也保留。
                if task.get('scene_variant_id') is not None:  # 变体编号同样保留。
                    bundle['scene_variant_id'] = task['scene_variant_id']  # 变体编号也写回。
                resolved_seq_id = task.get('seq_id')  # 先以任务里的序列号为准。
                if resolved_seq_id is None:  # 先从 bundle 自身回退。
                    resolved_seq_id = bundle.get('seq_id')  # 任务没有时再看 bundle 里有没有。
                if resolved_seq_id is None and scenario_events:  # 还没有时再从首事件元信息里兜底。
                    first_event = scenario_events[0]  # 仍然没有时从首事件元信息里兜底。
                    if isinstance(first_event, Mapping):  # 首事件必须是映射型才能读 meta。
                        meta = first_event.get('meta')  # 取首事件元信息。
                        if isinstance(meta, Mapping):  # meta 也必须是映射型才能继续读 seq_id。
                            resolved_seq_id = meta.get('seq_id')  # 从元信息里取序列号兜底。
                bundle['seq_id'] = resolved_seq_id  # 写回最终解析出的序列号。
                bundle['task_id'] = task['task_id']  # 任务编号必写。
                bundle['axes'] = deepcopy(task.get('axes') or {})  # D3：轴配置深拷贝写入 bundle，避免嵌套结构（如 axes 内的 list/dict）与 task 共享引用被后续迭代或 bundle 序列化前的写回污染；与 L903 scenario_context deepcopy 同口径。
                bundle['scenario_context'] = deepcopy(scenario_context)  # 场景上下文写副本，避免后续共享修改。
                bundle['runtime_log'] = _resolve_runtime_log(runtime_collector, estimator, model, cfg, task, method_name)  # 生成运行日志。
                # 输出本次任务-方法的定位结果（取最终状态作为终点位姿）
                _states = bundle.get('states') or []
                if _states:
                    _final_state = _states[-1]
                    _px = _final_state.get('px')
                    _py = _final_state.get('py')
                    _yaw = _final_state.get('yaw')
                    _n_steps = len(_states)
                    if _px is not None and _py is not None:
                        _pose_str = f"px={float(_px):.4f} py={float(_py):.4f}"
                        if _yaw is not None:
                            _pose_str += f" yaw={float(_yaw):.4f}"
                        try:
                            # 终端/重定向句柄异常不能打断 checkpoint selection 的后续评估链路。
                            print(
                                f"[定位结果] 任务={task['task_id']} 场景={task['scene_id']} "
                                f"方法={method_name} 步数={_n_steps} 终点位姿: {_pose_str}",
                                flush=True,
                            )
                        except OSError:
                            pass
                if retain_prediction_bundles:
                    in_memory_bundles.append(bundle)  # 仅在显式要求时缓存到内存中，避免把大对象挂到返回值上。
                rt_metrics = compute_runtime_metrics(bundle['runtime_log'])  # 委托给 runtime_metrics 模块统一口径计算。
                runtime_rows.append(  # 把当前方法这次任务的运行时指标补成一行。
                    {  # 这一行是运行时统计表的单条记录。
                        'task_id': task['task_id'],  # 统计行里记录任务编号，方便回查。
                        'scene_id': task['scene_id'],  # 记录场景编号，便于按场景汇总性能。
                        'seq_id': resolved_seq_id,  # 记录序列编号，方便和外部数据对齐。
                        'method_name': method_name,  # 记录方法名，后面统计才知道是哪条链路。
                        'latency_mean': rt_metrics['latency_mean'],  # 延迟均值，统一口径。
                        'latency_p50': rt_metrics['latency_p50'],  # 延时中位数反映典型响应时间。
                        'latency_p95': rt_metrics['latency_p95'],  # 延时 95 分位反映尾延迟压力。
                        'params': rt_metrics['params'],  # 参数量，统一口径。
                        'ram_peak': rt_metrics['ram_peak'],  # 峰值内存，统一口径。
                    }  # 运行时统计行结构结束。
                )  # 这次 append 结束，形成一条完整统计行。

                # 第 12 轮审查 MEDIUM-1 修复（R12-C M1）：task_id 与 method_name 同为用户可控路径组件，
                # 但原实现仅校验 method_name，构成路径穿越防御不对称。task_id 来源包括用户通过
                # cfg['scene_tasks'] 直接注入，若含 ../ 可写出 predictions_dir 之外。此处对 task_id
                # 同样调用 validate_path_component 校验，与 L262 seq_id / method_name 口径对齐。
                out_path = predictions_dir / f"{validate_path_component(task['task_id'], name='task_id')}__{validate_path_component(method_name, name='method_name')}.json"  # 每个任务方法一份预测文件。
                out_path.parent.mkdir(parents=True, exist_ok=True)  # 运行中若目录被并发重建或清理，这里再次兜底。
                if out_path.exists():  # D11: 跳过已存在的预测，支持断点续评估。
                    continue  # 预测文件已存在，跳过写入。
                out_path.write_text(dumps_json_text(bundle), encoding='utf-8')  # 落盘 bundle；dumps_json_text 已在模块顶层导入（D10）。
                artifacts.append(str(out_path))  # 记录产物路径。
                prediction_index.append({  # 把当前 bundle 的落盘位置写进审计索引。
                    'task_id': task['task_id'],  # 索引中记录任务编号，方便按任务检索。
                    'scene_id': task['scene_id'],  # 索引中记录场景编号，便于按场景查看。
                    'seq_id': resolved_seq_id,  # 索引中记录序列编号，便于和真值表对应。
                    'method_name': method_name,  # 索引中记录方法名，便于按算法过滤。
                    'repeat_id': task.get('repeat_id'),  # 索引中保留重复编号，方便复现实验。
                    'scene_variant_id': task.get('scene_variant_id'),  # 索引中保留场景变体编号，方便审计场景扰动。
                    'prediction_path': str(out_path),  # 索引中写入文件路径，便于下游直接打开文件。
                })  # 预测索引这一条记录完成。

        index_path = audits_dir / 'prediction_index.json'  # 审计用的预测索引文件。
        index_path.parent.mkdir(parents=True, exist_ok=True)  # 防御性确保目录存在（bak 搜索中对 output_root 的 mkdir 可能遗漏）。
        index_path.write_text(dumps_json_text(prediction_index), encoding='utf-8')  # 写出索引 JSON。
        artifacts.append(str(index_path))  # 把索引文件也计入产物。
        runtime_table_path = plotting_dir / 'runtime_table.json'  # 给绘图和统计消费的运行时表。
        runtime_table_path.write_text(dumps_json_text(runtime_rows), encoding='utf-8')  # 写出运行时表。
        artifacts.append(str(runtime_table_path))  # 把运行时表加入产物列表。
        log_path = logs_dir / 'core_pipeline.log'  # 运行日志文件。
        log_path.write_text(  # 把本次运行规模写成一行日志，便于快速核对任务量。
            f"scene_tasks={len(scene_tasks)} methods={len(methods)} bundles={len(prediction_index)}\n",  # 日志正文按实际落盘 bundle 数统计。
            encoding='utf-8',  # 统一用 UTF-8，避免中文说明乱码。
        )  # 日志文件写出完成。
        artifacts.append(str(log_path))  # 日志文件同样加入产物列表。

        return StageResult(  # 把 core_pipeline 的阶段结果统一包装成标准 StageResult。
            stage_name='core_pipeline',  # 阶段名固定为 core_pipeline，供上层路由识别。
            artifacts=artifacts,  # 产物路径列表返回给上层，作为外部可读结果。
            metadata={  # metadata 放运行过程中的结构化上下文，方便审计和二次消费。
                'scene_tasks': scene_tasks,  # 保留实际执行的任务清单，便于复盘。
                'methods': methods,  # 保留实际执行的方法列表，便于回查分支。
                'prediction_index': prediction_index,  # 保留预测索引元数据，便于查文件。
                'prediction_bundles': in_memory_bundles if retain_prediction_bundles else [],  # 默认不把大对象挂到 metadata，必要时才保留。
                'runtime_table': runtime_rows,  # 保留运行时统计表，便于性能分析。
            },  # metadata 结构结束。
        )  # StageResult 构造完成，作为整个阶段最终输出。



def run(pipeline_cfg: dict | None = None) -> StageResult:  # 保留旧式函数入口，直接转发给 CorePipeline.run。
    """保留旧式函数入口，直接转发给 `CorePipeline.run`。"""
    return CorePipeline().run(pipeline_cfg)  # 对外兼容老调用方式。
