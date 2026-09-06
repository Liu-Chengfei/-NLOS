"""定义并应用 V0–V3 视觉退化等级，只对 VIO 事件注入偏差和质量变化。

文件职责：
  把协议里定义的 V0–V3 视觉退化施加到 VIO 事件上。

本文件绝对不负责：
  不修改 IMU/UWB，不改几何布局。

上游依赖：configs/base/scene_axis_protocol.yaml、protocol/event_schema.py
下游调用者：pipelines/core_pipeline.py、tests/scenarios/test_visual_levels.py

输入对象定义：
  - events       合法事件序列
  - visual_level 视觉退化等级名（如 "V0"、"V2"）
  - visual_cfg   包含该等级参数的配置映射

输出对象定义：
  - filtered_events  经过视觉退化后的事件序列（blackout 事件被移除）
  - visual_report    可审计的退化报告

核心变量定义：
  - candidate_vio_events  所有 VIO 事件及其索引
  - blackout_indices      被 blackout 的 VIO 事件索引集合
  - feature_drop_plan     每个被选中事件的 tracked_features 变更记录
  - reproj_err_plan       每个被选中事件的 reproj_err 变更记录
  - drift_transform_plan  每个被选中事件的 dx/dy/dyaw 变更记录

关键设计决策：
  - 视觉退化只作用于 VIO，不动 IMU/UWB，不改 anchor 布局。
  - blackout 事件从输出中移除（不是标记），其余 VIO 事件的 payload 被原地修改。
  - drift_bias_m 允许为 0.0（无漂移），约束为非负。
  - 漂移为 Wiener 随机游走过程：bias_x/y 按时间累积 N(0, drift_bias_m·√dt)，
    第一个 VIO 事件无前驱不累积；dx_after = dx_before + bias_x·sign(dx_before)。
    与 Scaramuzza VIO 教程一致（VO 长期漂移是随机积累误差，非固定 scale）。
  - blackout_ratio 是 blackout_prob 的规范别名，两者取一即可。
"""

from __future__ import annotations  # 允许在类型注解中使用前向引用。

from collections.abc import Mapping, Sequence  # 用于 isinstance 检查映射和序列类型。
from copy import deepcopy  # 深拷贝，保证输入事件不被修改。
import hashlib  # 用于 VIO drift 随机游走的确定性种子。
import math  # 数学工具，用于 isfinite、sqrt 等。
import random  # 用于 VIO drift 随机游走的高斯采样。
from liquidloc.common.angle_utils import wrap_angle_rad  # 角度归一化到 [-π, π)。
from liquidloc.common.constants import DEFAULT_THRESHOLDS, DRIFT_PENALTY_SATURATION_M, MODALITY_VIO  # VIO 模态标识常量、默认阈值和漂移惩罚饱和阈值。
from liquidloc.common.validation import coerce_finite_scalar, is_real, is_string_like, sample_axis_interval  # 统一判断标量数值类型和有限浮点转换、区间采样。
from liquidloc.protocol.event_schema import validate_event_sequence  # 事件序列合法性校验。
from liquidloc.scenarios._event_utils import stable_window_start as _stable_window_start  # 共享的稳定哈希窗口起始函数。


# 漂移方向判定的浮点零容差：当 dx/dy/dyaw 绝对值小于此值时认为无方向性位移，不叠加漂移。
# 主循环（L441-455）和一致性检查（L540-544）必须共用同一容差，否则会导致 protocol_consistent 误报。
_DRIFT_ZERO_TOL = 1e-12

# 质量值保留的小数位数，避免浮点噪声。与 N 轴 nlos_levels._QUALITY_DECIMALS 口径对齐。
_QUALITY_DECIMALS = 6


def _resolve_visual_level_params(visual_level: str, visual_cfg, rng: random.Random | None = None) -> dict:
    """从配置映射中解析指定等级的参数字典，支持区间字段均匀采样。

    2026-09-03 区间化：
      - blackout_duration_range_s / blackout_fraction / keyframe_interruptions_per_seq /
        keyframe_burst_duration_range_s / keyframe_burst_fraction / keyframe_recovery_jump_m
        支持标量或 [low, high] 区间。
      - rng 不为 None 时在区间内均匀采样；rng=None 时回退为确定性 hash 派生 RNG（保持可复现）。
    """
    if not is_string_like(visual_level) or not str(visual_level).strip():  # 等级名必须是非空字符串。
        raise ValueError('visual_level must be a non-empty string')
    if not isinstance(visual_cfg, Mapping):  # 配置必须是映射类型。
        raise TypeError('visual_cfg must be a mapping')

    # 尝试三层嵌套查找：visual_cfg[level] → visual_cfg['levels'][level] → visual_cfg 本身。
    if visual_level in visual_cfg and isinstance(visual_cfg[visual_level], Mapping):
        params = dict(visual_cfg[visual_level])  # 第一层：配置以等级名为 key。
    else:
        raw_levels = visual_cfg.get('levels')  # 第二层：配置嵌套在 'levels' 下。
        if isinstance(raw_levels, Mapping) and visual_level in raw_levels and isinstance(raw_levels[visual_level], Mapping):
            params = dict(raw_levels[visual_level])
        else:
            params = dict(visual_cfg)  # 第三层：配置本身就是参数（无嵌套）。
            if params.get('level') not in (None, visual_level):  # 如果有 level 字段但不匹配，报错。
                raise ValueError(f'visual_cfg does not match requested level {visual_level}')

    # blackout_ratio 是 blackout_prob 的规范别名，两者取一即可；
    # 旧名 prob 保留兼容，但实际用作确定性比例而非随机概率。
    # 无论两者是否同时存在，都删除别名字段，避免下游消费方看到两个语义重复的键，与 A 轴 L103-105 对齐。
    if 'blackout_ratio' in params:
        if 'blackout_prob' not in params:
            params['blackout_prob'] = params['blackout_ratio']  # 仅当规范名缺失时才映射。
        del params['blackout_ratio']  # 始终删除别名字段，避免"两者同时存在"时残留非白名单键。

    # 协议字段双向 backfill (D-5 v2 渐进式改名):
    # scene_axis_protocol.yaml V 轴漂移系数已协议冻结为 drift_bias_sigma_mps (Wiener σ, m/√s).
    # 但 visual_levels.py 内部消费方仍以 drift_bias_m 命名 (line 270/285/382/389 等 ~10 处),
    # 一次性改全部消费方会触发多处 KeyError 风险. v2 采用双向 backfill 渐进改名:
    #   (1) 上游传新名 drift_bias_sigma_mps → backfill 到旧名 drift_bias_m, 内部消费方继续用旧名.
    #   (2) 上游传旧名 drift_bias_m (历史代码) → backfill 到新名 drift_bias_sigma_mps, 协议层一致.
    # 双向 backfill 让 v1 → v2 平滑过渡, 后续可在 separate PR 中完全改完所有消费方.
    # 历史审计: v1 单向 backfill 自 2026-04-15 仅有新→旧方向, v2 (2026-04-28) 改双向.
    if 'drift_bias_sigma_mps' in params and 'drift_bias_m' not in params:
        params['drift_bias_m'] = params['drift_bias_sigma_mps']
    if 'drift_bias_m' in params and 'drift_bias_sigma_mps' not in params:
        params['drift_bias_sigma_mps'] = params['drift_bias_m']

    required_keys = ('tracked_features_range', 'reproj_err_max', 'blackout_prob', 'drift_bias_m')  # 必需字段 (内部消费用旧名).
    missing = tuple(k for k in required_keys if k not in params)  # 检查必需字段是否齐全.
    if missing:  # 缺字段就不能继续，后面每一步都依赖它们。
        raise KeyError(f'visual_cfg is missing required authority fields: {missing}')
    # 区间采样辅助：rng 为 None 时用 hash 派生确定性 RNG（保持可复现）
    if rng is None:
        _fallback_seed = int(hashlib.sha256(
            f"visual_interval:{visual_level}".encode("utf-8")
        ).hexdigest()[:8], 16)
        rng = random.Random(_fallback_seed)

    # 区间化字段：支持 [low, high] 区间均匀采样，标量直接返回。
    interval_params = [
        ("blackout_duration_range_s", "blackout_duration_range_s"),
        ("blackout_fraction", "blackout_fraction"),
        ("keyframe_interruptions_per_seq", "keyframe_interruptions_per_seq"),
        ("keyframe_burst_duration_range_s", "keyframe_burst_duration_range_s"),
        ("keyframe_burst_fraction", "keyframe_burst_fraction"),
        ("keyframe_recovery_jump_m", "keyframe_recovery_jump_m"),
        # 2026-09-03 补区间化（V0/V1/V2 档从标量升级为 [min, max]）：
        ("reproj_err_max", "reproj_err_max"),
        ("drift_bias_sigma_mps", "drift_bias_sigma_mps"),
        ("drift_bias_m", "drift_bias_m"),  # 与 drift_bias_sigma_mps 同步区间化，backfill 后两者皆为标量。
        ("increment_noise_std_mps", "increment_noise_std_mps"),
        # 2026-10 补区间化（V1 blackout_prob 从标量升级为 [min, max]）：
        ("blackout_prob", "blackout_prob"),
    ]
    for cfg_key, out_key in interval_params:
        if cfg_key in params:
            v = params[cfg_key]
            if isinstance(v, (list, tuple)) and len(v) == 2:
                sampled = sample_axis_interval(v, rng, name=f"visual_cfg['{cfg_key}']")
                params[cfg_key] = sampled  # 替换为标量采样值

    return params  # 返回规整后的参数字典。


def _normalize_feature_range(feature_range) -> tuple[float, float]:
    """将 tracked_features_range 归一化为 (lower, upper) 浮点元组。"""
    if not isinstance(feature_range, Sequence) or isinstance(feature_range, (str, bytes, bytearray, memoryview)) or len(feature_range) != 2:
        raise TypeError("visual_cfg['tracked_features_range'] must be a two-value sequence")  # 必须是二元序列，排除 str/bytes/bytearray/memoryview 等二进制容器，与 A 轴 _normalize_offset_range 对齐。
    lower = coerce_finite_scalar(feature_range[0], name="visual_cfg['tracked_features_range'][0]")  # 强制转有限浮点。
    upper = coerce_finite_scalar(feature_range[1], name="visual_cfg['tracked_features_range'][1]")  # 强制转有限浮点。
    if lower < 0.0 or upper < 0.0:  # 特征数不能为负。
        raise ValueError("visual_cfg['tracked_features_range'] must be non-negative")
    if lower > upper:  # 下限不能超过上限。
        raise ValueError("visual_cfg['tracked_features_range'] lower bound must be <= upper bound")
    return lower, upper  # 返回 (下限, 上限) 元组。


def _is_close(lhs: float, rhs: float, *, tolerance: float = 1e-9) -> bool:
    """浮点近似比较，容差默认 1e-9（仅绝对容差，无相对容差）。

    委托给标准库 math.isclose(rel_tol=0.0, abs_tol=tolerance)，
    获得 inf==inf → True 的正确语义。本函数显式禁用相对容差（rel_tol=0.0），
    因为一致性检查需要纯绝对容差口径（比较退化函数输出与记录值的绝对差），
    而仓内其余模块（plotting/dataio/geometry_levels）使用 math.isclose 默认相对容差，
    口径不同——这是有意为之，不是"统一口径"。
    调用方需保证 lhs/rhs 为有限值；本函数对 inf==inf 返回 True（遵循 math.isclose 语义）。
    """
    return math.isclose(float(lhs), float(rhs), rel_tol=0.0, abs_tol=tolerance)


def _degrade_tracked_features(*, tracked_before: int, feature_lower: float, feature_upper: float) -> int:
    """将 tracked_features clamp 到 [feature_lower, feature_upper] 后取整。

    先 clamp 再取整；对 clamp 后的结果使用 round-half-down（math.ceil(x-0.5)）取整，
    避免 Python round() 的银行家舍入。取整本身可能在 0.5 处向下偏离浮点下界，
    因此下一步用整数域 clamp 兜底，保证中间结果不低于 feature_lower 的整数上界、
    不超过 feature_upper 的整数下界（当 ceil(lower) <= floor(upper) 时）。

    最后一步 `min(tracked_before, result)` 施加"退化不增大"约束：当 tracked_before
    低于 feature_lower 时，最终结果取 tracked_before，可能低于 feature_lower 的整数上界。
    这与 sampling_envelope 语义一致——退化不应人为抬高特征数。

    边界情形：当 feature_lower == feature_upper 且为非整数（如 2.5）时，
    ceil(lower) > floor(upper)，整数区间为空，此时回退到 floor(feature_upper)
    避免结果超过 feature_upper。
    """
    clamped = min(float(feature_upper), max(float(feature_lower), float(tracked_before)))  # clamp 到区间。
    result = int(math.ceil(clamped - 0.5))  # round-half-down 取整；下界保证由下一步整数域 clamp 兜底。
    # 整数域二次 clamp，确保取整后不违反浮点声明的上下界。
    int_lower = int(math.ceil(feature_lower))
    int_upper = int(math.floor(feature_upper))
    if int_lower <= int_upper:  # 正常区间：整数域 clamp 有效。
        result = max(int_lower, min(int_upper, result))
    else:  # 退化区间（非整数等值范围如 [2.5, 2.5]）：回退到 floor(upper) 避免越界。
        result = min(int_upper, result)
    # 退化函数不应增大 tracked_features：当 tracked_before 低于 feature_lower 时，
    # 整数域 clamp 可能把结果抬到 tracked_before 之上，必须截断。
    result = min(tracked_before, result)
    return result


def _degrade_reproj_err(*, reproj_err_before: float, reproj_err_max: float, drift_bias_m: float) -> float:
    """将重投影误差叠加累积漂移幅度并 clamp 到 [0, reproj_err_max]。

    drift_bias_m 语义：累积漂移幅度（米），非 Wiener 系数 σ（m/√s）。
    调用方应传 math.hypot(drift_bias_x, drift_bias_y)，第一个 VIO 事件无前驱时为 0。
    工程代理：用米量级的漂移近似像素量级的重投影误差增大，单位不严格匹配但量级相关。

    入口对三个参数做有限性守卫，避免 NaN/Inf 在 max/min 中产生反直觉行为
    （如 NaN 被 max(0.0, NaN) 静默替换为 0.0，掩盖数据损坏）。
    正常调用链中上游 validate_event_sequence 已校验有限性，此守卫是防御性兜底。
    """
    reproj_err_before_f = coerce_finite_scalar(reproj_err_before, name="reproj_err_before")
    reproj_err_max_f = coerce_finite_scalar(reproj_err_max, name="reproj_err_max", min_value=0.0)  # 函数级防御：负上限会穿透 clamp 返回负值。
    drift_bias_m_f = coerce_finite_scalar(drift_bias_m, name="drift_bias_m")
    amplified_error = reproj_err_before_f + drift_bias_m_f  # 加法漂移叠加。
    return min(reproj_err_max_f, max(0.0, amplified_error))  # clamp 到 [0, max]。


def _degrade_quality(
    *,
    quality_before: float,
    tracked_before: int,
    tracked_after: int,
    reproj_err_before: float,
    reproj_err_after: float,
    drift_bias_m: float,
) -> float:
    """Map visual degradation severity into a monotonic VIO quality drop.

    drift_bias_m 语义：累积漂移幅度（米），非 Wiener 系数 σ（m/√s）。
    调用方应传 math.hypot(drift_bias_x, drift_bias_y)，第一个 VIO 事件无前驱时为 0。
    drift_penalty = min(1.0, drift_bias_m / DRIFT_PENALTY_SATURATION_M) 为无量纲比值。

    入口对 quality_before 做有限性守卫，避免 NaN 在 min/max 中被静默映射为
    quality_max（Python 中 min(1.0, NaN) 返回 1.0），掩盖数据损坏。
    正常调用链中上游 validate_event_sequence 已校验有限性，此守卫是防御性兜底。
    """
    quality_before_f = coerce_finite_scalar(quality_before, name="quality_before")
    quality_min = DEFAULT_THRESHOLDS["quality_min"]
    quality_max = DEFAULT_THRESHOLDS["quality_max"]
    tracked_loss_ratio = 0.0
    if tracked_before > 0:
        tracked_loss_ratio = max(0.0, float(tracked_before - tracked_after) / float(tracked_before))
    # 分母下界 1.0 是工程选择：避免小 reproj_err_before 时 penalty 过度放大。
    # 当 reproj_err_before < 1.0 时，penalty 会被压缩（相对增长被低估），
    # 这是有意为之——小误差场景下退化应更温和。
    reproj_denominator = max(float(reproj_err_before), 1.0)
    reproj_penalty = max(0.0, float(reproj_err_after - reproj_err_before) / reproj_denominator)
    reproj_penalty = min(1.0, reproj_penalty)
    # 加法漂移惩罚：drift_bias_m 越大惩罚越重，用 sigmoid 式映射到 [0, 1]。
    # DRIFT_PENALTY_SATURATION_M 参考尺度来源：典型室内 VIO 系统（如 ORB-SLAM）在 10m 轨迹上的累积漂移约 0.1-1.0m，
    # 1.0m 是"严重漂移"的工程阈值，超过此值认为 VIO 已不可信。
    # 此值与 protocol.scene_axis_protocol.py L252-253 的 drift_bias_m 上限校验共享（单源真相）。
    drift_penalty = min(1.0, float(drift_bias_m) / DRIFT_PENALTY_SATURATION_M)  # 超过饱和阈值则满惩罚。
    # 质量退化权重：tracked_loss (0.55) > reproj_penalty (0.30) > drift_penalty (0.25)，
    # 权重来源：特征点丢失对 VIO 质量影响最大（直接决定位姿估计能力），
    # 重投影误差次之（反映内参/外参精度），漂移最末（缓慢累积，短期可容忍）。
    # 权重总和 1.10 > 1.0 是有意为之：确保单一维度满惩罚时加权和也能接近 0.95 截断，
    # min(0.95, ...) 确保质量最多退化 95%（保留最低 5% 信任度）。
    quality_penalty = min(
        0.95,
        (0.55 * tracked_loss_ratio) + (0.30 * reproj_penalty) + (0.25 * drift_penalty),
    )
    degraded_quality = quality_before_f * max(0.0, 1.0 - quality_penalty)
    degraded_quality = max(quality_min, min(quality_max, degraded_quality))
    return round(float(degraded_quality), _QUALITY_DECIMALS)


def apply_visual_level(events, visual_level: str, visual_cfg, rng: random.Random | None = None):
    """将协议定义的视觉退化等级应用到 VIO 事件上，支持区间字段均匀采样。

    Args:
        events: 事件序列。
        visual_level: 视觉退化等级名（V0/V1/V2/V3）。
        visual_cfg: 该等级参数配置。
        rng: 可选序列级确定性 RNG，用于区间字段均匀采样。None 时回退 hash 派生（可复现）。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "visual_level": visual_level,
        # 日志记录原始输入参数；scenario_context（在函数末尾构建）使用采样后的 params。
        "blackout_prob_raw": (visual_cfg or {}).get("blackout_prob"),
        "drift_bias_m_raw": (visual_cfg or {}).get("drift_bias_m"),
        "reproj_err_max_raw": (visual_cfg or {}).get("reproj_err_max"),
        "tracked_features_range": (visual_cfg or {}).get("tracked_features_range"),
        "events_count": len(events) if isinstance(events, (list, tuple)) else None,
    }, "apply_visual_level 入口参数")
    # 入口校验并归一化 visual_level，与 N 轴 apply_nlos_level L193-197 口径对齐，
    # 确保空序列路径和主路径报告中的 visual_level 一致（已 strip，无前后空白）。
    if not is_string_like(visual_level):
        raise TypeError(f"visual_level must be a string, got {type(visual_level).__name__}")
    normalized_visual_level = str(visual_level).strip()
    if not normalized_visual_level:
        raise ValueError("visual_level must be a non-empty string")

    event_list = list(events)  # 确保可多次遍历。
    # 空序列路径也必须校验 visual_cfg 类型，与 N 轴 apply_nlos_level L198-199 口径对齐，
    # 避免非法配置静默通过并产生误导性的空报告。
    if not isinstance(visual_cfg, Mapping):
        raise TypeError(f"visual_cfg must be a mapping, got {type(visual_cfg).__name__}")
    if not event_list:  # 空序列无事件可退化，在 validate 前提前返回。
        return [], {'visual_level': normalized_visual_level, 'candidate_vio_events': 0, 'feature_drop_plan': [], 'reproj_err_plan': [], 'quality_plan': [], 'drift_transform_plan': [], 'blackout_segments': [], 'reproj_err_max': 0.0, 'drift_bias_m': 0.0, 'blackout_prob': 0.0, 'blackout_count_expected': 0, 'blackout_count': 0, 'blackout_selection_start': 0, 'blackout_strategy': 'stable_hashed_contiguous_vio_window', 'tracked_features_range_declared': [0.0, 0.0], 'tracked_features_range_after': None, 'reproj_err_max_after': None, 'consistency_checks': {'tracked_features_range': True, 'reproj_err_max': True, 'blackout': True, 'drift_bias_m': True}, 'protocol_consistent': True}
    validate_event_sequence(event_list)  # 入口校验事件序列合法性。

    # 校验 V 轴 range_semantics 语义约束：tracked_features_range 是采样包络，不是分类边界。
    # 场景等级是先给定的（V0/V1/V2/V3），而非由观测到的特征点数量反推判定。
    # 如果 range_semantics 存在但不是 sampling_envelope，说明协议语义已被篡改，必须阻止。
    if isinstance(visual_cfg, Mapping):
        range_semantics = visual_cfg.get('range_semantics')
        if range_semantics is not None and range_semantics != 'sampling_envelope':
            raise ValueError(
                f"visual_cfg.range_semantics must be 'sampling_envelope' (sampling envelope, not classification boundary), "
                f"got {range_semantics!r}; scene levels are predetermined, not inferred from feature counts"
            )

    params = _resolve_visual_level_params(normalized_visual_level, visual_cfg, rng=rng)  # 解析等级参数（用归一化后的等级名）。
    feature_lower, feature_upper = _normalize_feature_range(params['tracked_features_range'])  # 归一化特征范围。
    reproj_err_max = coerce_finite_scalar(params['reproj_err_max'], name="visual_cfg['reproj_err_max']")  # 重投影误差上限。
    blackout_prob = coerce_finite_scalar(params['blackout_prob'], name="visual_cfg['blackout_prob']")  # 确定性缺失比例。
    drift_bias_m = coerce_finite_scalar(params['drift_bias_m'], name="visual_cfg['drift_bias_m']")  # 漂移系数（随机游走 σ）。
    # 提取序列身份，用于 VIO drift 随机游走的确定性种子（与 A 轴口径对齐）。
    sequence_scene_id, sequence_seq_id = "", ""
    for _ev in event_list:
        _raw_meta = _ev.get("meta")
        if isinstance(_raw_meta, Mapping):
            _sid = _raw_meta.get("scene_id", "")
            _sqid = _raw_meta.get("seq_id", "")
            sequence_scene_id = "" if _sid is None else str(_sid)
            sequence_seq_id = "" if _sqid is None else str(_sqid)
            break
    if reproj_err_max <= 0.0:  # 重投影误差上限必须严格正（与协议层 scene_axis_protocol.py L244-245 对齐，不允许 0.0）。
        raise ValueError("visual_cfg['reproj_err_max'] must be positive")
    if not 0.0 <= blackout_prob <= 1.0:  # blackout 比例必须在 [0, 1] 区间。
        raise ValueError("visual_cfg['blackout_prob'] must be within [0, 1]")
    if drift_bias_m < 0.0:  # drift_bias_m 允许 0.0（完全消除 VIO 运动），但不能为负。
        raise ValueError("visual_cfg['drift_bias_m'] must be non-negative")

    # 将所有事件转为可修改的 dict 副本，保证输入不被修改。
    new_events = []
    for event in event_list:
        if hasattr(event, "to_dict") and callable(event.to_dict):  # Event dataclass 优先走 to_dict。
            new_events.append(event.to_dict())
        elif isinstance(event, Mapping):  # 普通字典走 deepcopy。
            new_events.append(deepcopy(dict(event)))
        else:
            raise TypeError(f"event must be mapping-like, got {type(event).__name__}")  # 其它类型拒绝。

    # 筛出所有 VIO 事件及其索引，后续只对这些事件做退化。
    candidate_vio_events = [
        (index, event)  # (序列索引, 事件字典)
        for index, event in enumerate(new_events)
        if event["modality"] == MODALITY_VIO  # 只选 VIO 模态。
    ]

    # 按 blackout_prob 确定性比例计算要删掉多少 VIO 事件。
    blackout_target_count = min(
        len(candidate_vio_events),  # 不超过 VIO 事件总数。
        max(0, int(math.floor((len(candidate_vio_events) * blackout_prob) + 0.5))),  # 显式 half-up 取整，避免 Python round() 的银行家舍入把 0.5 压成 0。
    )
    blackout_segments = []  # 记录 blackout 时间段 [(t_start, t_end), ...]。
    blackout_indices = set()  # 记录被 blackout 事件的序列索引。
    blackout_selection_start = 0  # 默认窗口起始位置。
    if blackout_target_count:  # 有事件需要 blackout 时才执行。
        representative_meta = {}
        raw_meta = candidate_vio_events[0][1].get("meta")
        if isinstance(raw_meta, Mapping):
            representative_meta = dict(raw_meta)
        blackout_key = "|".join(
            [
                "visual_blackout",
                normalized_visual_level,  # 已 strip 归一化，避免带空白的等级名污染哈希 key。
                # 仅将 None 归一化为空字符串，保留其他 falsy 值（如 0/False）的字符串形式，
                # 避免 `or ""` 模式把 0 和 None 映射为同一 key 导致哈希碰撞。与 A 轴 L218 口径对齐。
                "" if representative_meta.get("scene_id") is None else str(representative_meta.get("scene_id")),
                "" if representative_meta.get("seq_id") is None else str(representative_meta.get("seq_id")),
                str(len(candidate_vio_events)),
                str(blackout_target_count),
            ]
        )
        blackout_selection_start = _stable_window_start(
            len(candidate_vio_events),
            blackout_target_count,
            key=blackout_key,
        )
        blackout_slice = candidate_vio_events[
            blackout_selection_start : blackout_selection_start + blackout_target_count
        ]  # 取稳定哈希决定的连续段。
        blackout_indices = {event_index for event_index, _ in blackout_slice}  # 收集索引。
        blackout_segments = [
            (float(blackout_slice[0][1]["t"]), float(blackout_slice[-1][1]["t"]))  # 记录时间跨度。
        ]

    filtered_events = []  # 最终输出事件列表（blackout 事件被移除）。
    feature_drop_plan = []  # 每个被选中事件的 tracked_features 变更审计。
    reproj_err_plan = []  # 每个被选中事件的 reproj_err 变更审计。
    quality_plan = []  # 每个被选中事件的 quality 变更审计。
    drift_transform_plan = []  # 每个被选中事件的 dx/dy/dyaw 变更审计。
    # VIO drift 随机过程状态：bias_x/bias_y 累积 Wiener 过程，last_vio_t 记录上一个 VIO 事件时间。
    drift_bias_x = 0.0  # x 方向累积 bias（米）。
    drift_bias_y = 0.0  # y 方向累积 bias（米）。
    last_vio_t: float | None = None  # 上一个 VIO 事件时间戳，用于计算 dt。
    for index, event in enumerate(new_events):  # 遍历所有事件。
        if event["modality"] != MODALITY_VIO:  # 非 VIO 事件直接透传。
            filtered_events.append(event)
            continue

        vio_payload = deepcopy(event["vio_payload"])  # 深拷贝 payload，避免修改原始数据。
        # 铁律 3 (Stage A1 下游修复, 2026-07-23): 新 VIO schema 不再包含
        # tracked_features / reproj_err (build_vio_events 已删除这两键). 视觉退化
        # 仍然需要这两个字段, 这里用 nominal baseline 值兜底: tracked = 120 (中段),
        # reproj_err = 0.0 (无误差); 物理对应 baseline scene_parameters V0 默认.
        tracked_before = int(vio_payload.get("tracked_features", 120))  # 原始 tracked_features, 缺失回落 baseline。
        tracked_after = _degrade_tracked_features(  # 退化后的 tracked_features。
            tracked_before=tracked_before,
            feature_lower=feature_lower,
            feature_upper=feature_upper,
        )
        feature_drop_plan.append(  # 记录变更审计。
            {
                "event_index": index,  # 事件在序列中的位置。
                "tracked_features_before": tracked_before,  # 变更前值。
                "tracked_features_after": tracked_after,  # 变更后值。
                "dropped_by_blackout": index in blackout_indices,  # 是否被 blackout。
            }
        )

        if index in blackout_indices:  # blackout 事件不进入输出，但已记录在 plan 中。
            continue

        dx_before = float(vio_payload["dx"])  # 原始 dx。
        dy_before = float(vio_payload["dy"])  # 原始 dy。
        dyaw_before = float(vio_payload["dyaw"])  # 原始 dyaw。
        reproj_err_before = float(vio_payload.get("reproj_err", 0.0))  # 原始重投影误差, 缺失回落 0.0 (baseline)。
        quality_before = float(vio_payload["quality"])  # 原始质量值。
        # drift Wiener 累积必须在 reproj_err/quality 退化之前完成，使退化函数能感知累积漂移量级。
        # drift_bias_m 解释为随机游走系数 σ（米·√s⁻¹），bias 方差随时间线性增长：
        #   bias_x(t+dt) = bias_x(t) + N(0, σ · √dt)。
        # 参考 Scaramuzza & Fraundorfer (2011) Visual Odometry Tutorial：
        # 单目 VO 漂移是累积随机误差，而非每事件固定偏差。
        # §6.2 #8 偏置型增量误差子机制对应代码层落点：
        # 协议层 drift_bias_sigma_mps (yaml V 轴 V0/V1/V2/V3 = 0/0.05/0.15/0.30 m/√s) 经
        # _resolve_visual_level_params 双向 backfill 映射到内部 drift_bias_m，本处 L399-411
        # 按 Wiener 累积生成 bias_x/bias_y，再于 L444-449 加性叠加到 dx/dy：
        #   vio_payload["dx"] = dx_before + drift_bias_x * sign(dx_before)
        # 与 §6.2 #7 尺度漂移 (sim_materializer.vio_scale_drift_rate 乘性 scale_factor) 在
        # 物化路径分离且全员同一叙事（§6.2 表格全方法同实现）。
        current_t = float(event["t"])
        if last_vio_t is not None:
            dt = max(0.0, current_t - last_vio_t)
            if dt > 0.0 and drift_bias_m > 0.0:
                sqrt_dt = math.sqrt(dt)
                # bias_x/bias_y 各自独立累积 Wiener 过程。
                drift_x_seed_str = f"vio_drift_x:{sequence_scene_id}:{sequence_seq_id}:{index}"
                drift_x_seed = int(hashlib.sha256(drift_x_seed_str.encode("utf-8")).hexdigest()[:8], 16)
                drift_bias_x += random.Random(drift_x_seed).gauss(0.0, drift_bias_m * sqrt_dt)
                drift_y_seed_str = f"vio_drift_y:{sequence_scene_id}:{sequence_seq_id}:{index}"
                drift_y_seed = int(hashlib.sha256(drift_y_seed_str.encode("utf-8")).hexdigest()[:8], 16)
                drift_bias_y += random.Random(drift_y_seed).gauss(0.0, drift_bias_m * sqrt_dt)
        last_vio_t = current_t
        # 累积漂移幅度（米）：用 hypot(bias_x, bias_y) 表示总漂移量级。
        # 单位与 DRIFT_PENALTY_SATURATION_M（米）一致，保证 quality penalty 是无量纲比值。
        # 第一个 VIO 事件无前驱，magnitude=0，reproj_err/quality 不受 drift 影响。
        drift_bias_magnitude = math.hypot(drift_bias_x, drift_bias_y)
        vio_payload["tracked_features"] = tracked_after  # 写入退化后的 tracked_features。
        vio_payload["reproj_err"] = _degrade_reproj_err(  # 写入退化后的 reproj_err。
            reproj_err_before=reproj_err_before,
            reproj_err_max=reproj_err_max,
            drift_bias_m=drift_bias_magnitude,  # 传累积幅度（米），非原始 σ（m/√s）
        )
        reproj_err_plan.append(  # 记录 reproj_err 变更审计。
            {
                "event_index": index,  # 事件在序列中的位置。
                "reproj_err_before": reproj_err_before,  # 变更前值。
                "reproj_err_after": float(vio_payload["reproj_err"]),  # 变更后值。
                "drift_bias_magnitude": drift_bias_magnitude,  # 累积漂移幅度（米），供一致性检查复现。
            }
        )
        vio_payload["quality"] = _degrade_quality(
            quality_before=quality_before,
            tracked_before=tracked_before,
            tracked_after=tracked_after,
            reproj_err_before=reproj_err_before,
            reproj_err_after=float(vio_payload["reproj_err"]),
            drift_bias_m=drift_bias_magnitude,  # 传累积幅度（米），非原始 σ（m/√s）
        )  # 写入退化后的 quality。
        quality_plan.append(
            {
                "event_index": index,  # 事件在序列中的位置。
                "quality_before": quality_before,  # 变更前值。
                "quality_after": float(vio_payload["quality"]),  # 变更后值。
            }
        )
        _dx_sign = (1.0 if dx_before > 0 else -1.0) if abs(dx_before) > _DRIFT_ZERO_TOL else 0.0
        _dy_sign = (1.0 if dy_before > 0 else -1.0) if abs(dy_before) > _DRIFT_ZERO_TOL else 0.0
        # 实际施加的漂移 = bias_x · sign(dx)，按方向叠加。
        drift_dx = drift_bias_x * _dx_sign
        drift_dy = drift_bias_y * _dy_sign
        vio_dx = dx_before + drift_dx  # dx 叠加时变漂移偏差。
        vio_dy = dy_before + drift_dy  # dy 叠加时变漂移偏差。
        vio_payload["dx"] = vio_dx
        vio_payload["dy"] = vio_dy
        # dyaw 漂移：叠加与原始方向一致的角漂移。
        # 映射比例 0.01 的来源：1m 位置漂移对应约 0.01rad（≈0.57°）航向偏差，
        # 这是典型室内 VIO 系统（如 VINS-Mono）在 1m 位移上的航向漂移量级，
        # 来源为 Scaramuzza & Fraundorfer, "Visual Odometry" Tutorial (2011) 中
        # 报告的单目 VO 航向漂移率约 1%/m ≈ 0.01 rad/m。
        _dyaw_sign = (1.0 if dyaw_before > 0 else -1.0) if abs(dyaw_before) > _DRIFT_ZERO_TOL else 0.0
        # dyaw 漂移也按 bias_x 的尺度（位置漂移 → 航向漂移），方向跟随 dyaw。
        dyaw_bias = drift_bias_x * 0.01 * _dyaw_sign
        vio_payload["dyaw"] = wrap_angle_rad(dyaw_before + dyaw_bias)  # dyaw 叠加漂移后归一化到 [-π, π)。
        drift_transform_plan.append(  # 记录 dx/dy/dyaw 变更审计。
            {
                "event_index": index,  # 事件在序列中的位置。
                "dx_before": dx_before,  # 变更前 dx。
                "dx_after": float(vio_payload["dx"]),  # 变更后 dx。
                "dy_before": dy_before,  # 变更前 dy。
                "dy_after": float(vio_payload["dy"]),  # 变更后 dy。
                "dyaw_before": dyaw_before,  # 变更前 dyaw。
                "dyaw_after": float(vio_payload["dyaw"]),  # 变更后 dyaw。
                "dyaw_wrapped_from_raw": wrap_angle_rad(dyaw_before + dyaw_bias),  # 用于一致性校验的原始 wrap 值。
                "drift_bias_x": drift_bias_x,  # 当前 x 方向累积 bias（米）。
                "drift_bias_y": drift_bias_y,  # 当前 y 方向累积 bias（米）。
                "drift_dx": drift_dx,  # 实际施加的 dx 漂移量（米）。
                "drift_dy": drift_dy,  # 实际施加的 dy 漂移量（米）。
            }
        )
        event["vio_payload"] = vio_payload  # 写回修改后的 payload。
        filtered_events.append(event)  # 非 blackout 的 VIO 事件加入输出。

    if not filtered_events:  # 场景轴函数必须返回合法事件序列，不能把整条序列删空后再交给协议层兜底。
        raise ValueError("visual blackout would remove every event; protocol event sequences must retain at least one event")
    # 重新计算 dt，因为 blackout 可能移除了事件导致相邻事件变化。
    for index, event in enumerate(filtered_events):
        if index == 0:  # 首个事件 dt 固定为 0。
            event["dt"] = 0.0
            continue
        event["dt"] = float(event["t"]) - float(filtered_events[index - 1]["t"])  # 相邻事件时间差。

    validate_event_sequence(filtered_events)  # 出口校验事件序列合法性。

    # 收集存活 VIO 事件的 payload，用于统计和一致性校验。
    surviving_vio_payloads = [
        event["vio_payload"]
        for event in filtered_events
        if event["modality"] == MODALITY_VIO  # 只看存活 VIO。
    ]
    tracked_after_values = [int(payload["tracked_features"]) for payload in surviving_vio_payloads]  # 所有存活 VIO 的 tracked_features。
    reproj_after_values = [float(payload["reproj_err"]) for payload in surviving_vio_payloads]  # 所有存活 VIO 的 reproj_err。
    tracked_range_after = None  # 退化后 tracked_features 的 [min, max]。
    reproj_err_max_after = None  # 退化后 reproj_err 的最大值。
    tracked_features_consistent = True  # tracked_features 是否满足退化不变量。
    if tracked_after_values:  # 有存活 VIO 事件时才统计。
        tracked_range_after = [min(tracked_after_values), max(tracked_after_values)]  # 实际 [min, max]。
        # V 轴里的 tracked_features_range 是采样包络，不是“必须把所有输入硬抬进区间”的分类边界。
        # 当原始 tracked_features 已经低于下界时，退化函数应保持“不增大”原则；一致性检查应验证
        # 结果是否等于规范退化函数的输出，而不是简单要求 after 必须落进声明区间。
        tracked_features_consistent = all(
            feature_step["tracked_features_after"]
            == _degrade_tracked_features(
                tracked_before=feature_step["tracked_features_before"],
                feature_lower=feature_lower,
                feature_upper=feature_upper,
            )
            and feature_step["tracked_features_after"] <= feature_step["tracked_features_before"]
            for feature_step in feature_drop_plan
        )
    reproj_err_consistent = True  # reproj_err 是否在声明上限内。
    if reproj_after_values:  # 有存活 VIO 事件时才统计。
        reproj_err_max_after = max(reproj_after_values)  # 实际最大值。
        reproj_err_consistent = all(  # 检查每个 reproj_err_after 是否可由 _degrade_reproj_err 复现且不超过上限。
            _is_close(
                reproj_step["reproj_err_after"],
                _degrade_reproj_err(  # 用相同参数重算，应与记录值一致。
                    reproj_err_before=reproj_step["reproj_err_before"],
                    reproj_err_max=reproj_err_max,
                    drift_bias_m=reproj_step["drift_bias_magnitude"],  # 用记录的累积幅度，非原始 σ
                ),
            )
            and reproj_step["reproj_err_after"] <= reproj_err_max  # 不超过声明上限。
            for reproj_step in reproj_err_plan
        )

    # blackout 一致性：实际 blackout 数量 = 预期数量，且 plan 中标记数量 = blackout 索引数量。
    blackout_consistent = len(blackout_indices) == blackout_target_count
    blackout_consistent = blackout_consistent and (
        sum(1 for feature_step in feature_drop_plan if feature_step["dropped_by_blackout"]) == len(blackout_indices)
    )
    # drift_bias_m 一致性：dx/dy_after 可由 before + drift_bias_x/y * sign(before) 复现，
    # dyaw_after 与 wrap 值一致。随机过程模型下每个事件的 drift_bias_x/y 不同，
    # 但仍可用 drift_step 中记录的 bias 值精确复现。
    # 当 before ≈ 0 时 sign=0，不叠加漂移（复用模块级 _DRIFT_ZERO_TOL，与主循环同口径）。
    drift_bias_m_consistent = all(
        _is_close(drift_step["dx_after"], drift_step["dx_before"] + drift_step["drift_dx"])
        and _is_close(drift_step["dy_after"], drift_step["dy_before"] + drift_step["drift_dy"])
        and _is_close(drift_step["dyaw_after"], drift_step["dyaw_wrapped_from_raw"])
        for drift_step in drift_transform_plan
    )
    consistency_checks = {  # 汇总四项一致性检查结果。
        "tracked_features_range": tracked_features_consistent,  # tracked_features 是否在声明范围。
        "reproj_err_max": reproj_err_consistent,  # reproj_err 是否不超过声明上限。
        "blackout": blackout_consistent,  # blackout 数量是否与预期一致。
        "drift_bias_m": drift_bias_m_consistent,  # dx/dy/dyaw 漂移是否与 drift_bias_m 一致。
    }
    protocol_consistent = all(consistency_checks.values())  # 全部一致才算协议一致。

    visual_report = {  # 完整审计报告。
        "visual_level": normalized_visual_level,  # 等级名（已 strip 归一化，与空序列路径口径一致）。
        "candidate_vio_events": len(candidate_vio_events),  # 候选 VIO 事件总数。
        "feature_drop_plan": feature_drop_plan,  # tracked_features 变更审计。
        "reproj_err_plan": reproj_err_plan,  # reproj_err 变更审计。
        "quality_plan": quality_plan,  # quality 变更审计。
        "drift_transform_plan": drift_transform_plan,  # dx/dy/dyaw 变更审计。
        "blackout_segments": blackout_segments,  # blackout 时间段。
        "reproj_err_max": reproj_err_max,  # 声明的重投影误差上限。
        "drift_bias_m": drift_bias_m,  # 声明的漂移偏差（米/事件）。
        "blackout_prob": blackout_prob,  # 确定性缺失比例。
        "blackout_count_expected": blackout_target_count,  # 预期 blackout 数量。
        "blackout_count": len(blackout_indices),  # 实际 blackout 数量。
        "blackout_selection_start": blackout_selection_start,  # blackout 连续窗口起始位置。
        "blackout_strategy": "stable_hashed_contiguous_vio_window",  # blackout 选择策略标识。
        "tracked_features_range_declared": [feature_lower, feature_upper],  # 声明的特征范围。
        "tracked_features_range_after": tracked_range_after,  # 退化后实际特征范围。
        "reproj_err_max_after": reproj_err_max_after,  # 退化后实际最大重投影误差。
        "consistency_checks": consistency_checks,  # 四项一致性检查。
        "protocol_consistent": protocol_consistent,  # 协议一致性总判定。
    }
    return filtered_events, visual_report
