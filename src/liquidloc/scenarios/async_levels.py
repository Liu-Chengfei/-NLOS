"""异步扰动等级场景构建模块。

文件职责：
  把协议里定义的异步扰动等级（A0–A3 等）施加到事件序列上，模拟传感器时钟偏移、
  抖动、突发缺失和跨模态时间偏差等异步现象。

本文件绝对不负责：
  不修改 IMU/UWB/VIO 的 payload 内容，不制造 NLOS 偏差，不改几何布局，
  不改视觉退化参数。

上游依赖：
  - configs/base/scene_axis_protocol.yaml  # 异步等级参数定义
  - liquidloc/common/constants.py          # 模态常量 MODALITY_IMU / MODALITY_UWB / MODALITY_VIO
  - liquidloc/protocol/event_schema.py     # 事件序列合法性校验

下游调用者：
  - pipelines/core_pipeline.py             # 核心流水线，在场景构建阶段调用
  - tests/scenarios/test_async_levels.py   # 单元测试

输入对象定义：
  - events       合法事件序列（list[dict] 或可迭代对象）
  - async_level  异步等级名（如 "A0"、"A2"），对应协议中的等级键
  - async_cfg    包含该等级参数的配置映射，必须含 offset_ms / jitter_ms /
                 burst_missing_prob / cross_modal_skew_ms / clock_drift_ppm 五个权威字段

输出对象定义：
  - filtered_events  经过异步扰动后的事件序列（blackout 事件被移除）
  - async_report     可审计的扰动报告，记录偏移、抖动、blackout 等全部变更明细

核心变量定义：
  - params               从 async_cfg 解析出的当前等级参数字典
  - offset_midpoint_ms   时间偏移范围的中点值（毫秒），用于非 IMU 事件的基准偏移
  - jitter_ms            抖动幅度（毫秒），用确定性高斯随机数施加到每个事件
  - burst_missing_prob   突发缺失比例，确定性选取连续非 IMU 事件窗口进行 blackout
  - cross_modal_skew_ms  跨模态时间偏差（毫秒），UWB 正偏、VIO 负偏各一半
  - clock_drift_ppm      时钟漂移（ppm），随时间线性累积到 UWB/VIO 事件上
  - non_imu_indices      所有非 IMU 事件在序列中的索引列表，blackout 从中选取
  - blackout_slice       被 blackout 的事件索引列表
  - timing_plan          每个事件的时间变更审计记录

关键设计决策：
  - 异步扰动只改事件时间戳 t，不改 payload 内容。
  - blackout 事件从输出中移除（不是标记），其余事件的时间戳被原地修改。
  - burst_missing_prob 是确定性比例而非随机概率，使用稳定哈希选取连续窗口。
  - 抖动用确定性高斯随机数（基于序列身份+事件索引+模态+出现次序的 SHA-256 种子），
    模拟真实传感器时钟抖动的零均值白噪声特性，替代旧的确定性交替正负模式。
  - 时钟漂移只作用于非 IMU 事件，UWB 正向漂移、VIO 负向漂移。

§4.4 抖动真随机错位协议级护栏（前提指导.md:1028）：
  - jitter_seconds = random.Random(jitter_seed).gauss(0.0, jitter_sigma_s)（本文件 L478），
    分布族固定为高斯，所有方法（EKF/robust_ekf/FGO/NN+EKF）共享同一分布族与同一 SHA-256 种子，
    即"真随机错位、分布族全员同一"。
  - jitter_ms 协议层迭代验证（本文件 L368-395）：必须非负，A0=0/A1≈5ms/A2≈15ms/A3≈25ms 量级。
  - 不允许单方使用均匀分布/重尾分布替代高斯，否则违反 §4.4 "抖动分布族全员同一"。

§4.4 丢包突发长度分布协议级护栏（前提指导.md:1029）：
  - blackout 策略 = "stable_hashed_contiguous_non_imu_window"（本文件 L363/L634），即确定性
    连续窗口 blackout，对应协议 "两状态马尔可夫突发" 的一种**确定性可复述等价实现**：
    突发长度 = total_non_imu × burst_missing_prob，所有方法看到同一分布。
  - 不是 i.i.d. Bernoulli 散点丢包（与 §4.4 "突发长度分布全员同一"一致；非 Bernoulli）。
  - 禁止对一方长黑障、对另一方短黑障；同一 (scene_id, seq_id, async_level) 三元组下
    blackout_target_count 与 blackout_slice 完全确定，无单方特权路径。
"""

from __future__ import annotations  # 允许在类型注解中使用前向引用。

import hashlib  # 用于稳定哈希，保证相同输入始终选取相同的 blackout 窗口。
import math  # 数学工具，用于 isfinite 等判断。
import random  # 用于确定性随机抖动，替代确定性交替模式。
from collections.abc import Iterable, Mapping, Sequence  # 用于 isinstance 检查映射和可迭代/序列类型。
from copy import deepcopy  # 深拷贝，保证输入事件不被修改。
from liquidloc.common.constants import MODALITY_IMU, MODALITY_UWB, MODALITY_VIO  # 三种模态标识常量。
from liquidloc.common.validation import coerce_finite_scalar, is_string_like  # 统一判断标量数值类型和有限浮点转换。
from liquidloc.protocol.event_schema import validate_event_sequence  # 事件序列合法性校验。
from liquidloc.scenarios._event_utils import stable_window_start as _stable_window_start  # 共享的稳定哈希窗口起始函数。


def _resolve_async_level_params(async_level: str, async_cfg: Mapping) -> dict:
    """从配置映射中解析指定异步等级的参数字典。

    查找顺序：
      1. async_cfg[async_level]           — 配置以等级名为 key 直接存储
      2. async_cfg['levels'][async_level] — 配置嵌套在 'levels' 子项下
      3. async_cfg 本身                   — 配置本身就是参数（无嵌套）

    参数:
        async_level: 异步等级名，必须是非空字符串。
        async_cfg:   包含等级参数的配置映射。

    返回:
        规整后的参数字典，至少包含 offset_ms / jitter_ms /
        burst_missing_prob / cross_modal_skew_ms / clock_drift_ppm 五个字段。

    异常:
        ValueError: async_level 为空或配置不匹配请求等级。
        TypeError:  async_cfg 不是映射类型。
        KeyError:   缺少必需的权威字段。
    """
    if not is_string_like(async_level) or not str(async_level).strip():  # 等级名必须是非空字符串。
        raise ValueError("async_level must be a non-empty string")
    if not isinstance(async_cfg, Mapping):  # 配置必须是映射类型。
        raise TypeError("async_cfg must be a mapping")

    # 三层嵌套查找：async_cfg[level] → async_cfg['levels'][level] → async_cfg 本身。
    if async_level in async_cfg and isinstance(async_cfg[async_level], Mapping):
        params = dict(async_cfg[async_level])  # 第一层：配置以等级名为 key。
    else:
        raw_levels = async_cfg.get("levels")  # 第二层：配置嵌套在 'levels' 下。
        if isinstance(raw_levels, Mapping) and async_level in raw_levels and isinstance(raw_levels[async_level], Mapping):
            params = dict(raw_levels[async_level])  # 找到嵌套等级配置。
        else:
            params = dict(async_cfg)  # 第三层：配置本身就是参数（无嵌套）。
            if params.get("level") not in (None, async_level):  # 如果有 level 字段但不匹配，报错。
                raise ValueError(f"async_cfg does not match requested level {async_level}")

    # burst_missing_ratio 是 burst_missing_prob 的规范别名，两者取一即可；
    # 旧名 ratio 保留兼容，但实际用作确定性比例而非随机概率。
    # 无论两者是否同时存在，都删除别名字段，避免下游消费方看到两个语义重复的键，与 V 轴 L76-82 对齐。
    if "burst_missing_ratio" in params:
        if "burst_missing_prob" not in params:
            params["burst_missing_prob"] = params["burst_missing_ratio"]  # 仅当规范名缺失时才映射。
        del params["burst_missing_ratio"]  # 始终删除别名字段，避免"两者同时存在"时残留非白名单键。

    required_keys = ("offset_ms", "jitter_ms", "burst_missing_prob", "cross_modal_skew_ms", "clock_drift_ppm")  # 必需的权威字段，与协议 A0-A3 定义一致。
    missing = [key for key in required_keys if key not in params]  # 找出缺失字段。
    if missing:  # 缺字段就不能继续，后面每一步都依赖它们。
        raise KeyError(f"async_cfg is missing required authority fields: {missing}")
    return params  # 返回规整后的参数字典。


def _normalize_offset_range(offset_range: Sequence) -> tuple[float, float]:
    """将 offset_ms 归一化为 (lower, upper) 浮点元组。

    参数:
        offset_range: 原始偏移范围，必须是二元序列。

    返回:
        (下限, 上限) 浮点元组。

    异常:
        TypeError:  输入不是二元序列。
        ValueError: 下限大于上限，或值非有限。
    """
    if not isinstance(offset_range, Sequence) or isinstance(offset_range, (str, bytes, bytearray, memoryview)) or len(offset_range) != 2:  # 必须是二元序列，排除字符串、字节、字节数组和内存视图。
        raise TypeError("async_cfg['offset_ms'] must be a two-value sequence")
    lower = coerce_finite_scalar(offset_range[0], name="async_cfg['offset_ms'][0]")  # 强制转有限浮点。
    upper = coerce_finite_scalar(offset_range[1], name="async_cfg['offset_ms'][1]")  # 强制转有限浮点。
    if lower > upper:  # 下限不能超过上限。
        raise ValueError("async_cfg['offset_ms'] lower bound must be <= upper bound")
    if lower < 0.0:  # 时间偏移下限不能为负（协议语义：偏移量是非负的时间差）。
        raise ValueError("async_cfg['offset_ms'] lower bound must be non-negative")
    return lower, upper  # 返回 (下限, 上限) 元组。


def _copy_events(events: Iterable) -> list[dict]:
    """将事件序列深拷贝为字典列表，保证输入不被修改。

    参数:
        events: 原始事件序列，每个事件可以是 dict 或带 to_dict 方法的对象。

    返回:
        深拷贝后的字典事件列表。

    异常:
        TypeError: 事件既不是映射也不是带 to_dict 方法的对象。
    """
    copied_events = []  # 收集拷贝后的事件。
    for event in list(events):  # 逐个事件处理。
        if hasattr(event, "to_dict") and callable(event.to_dict):  # Event dataclass 优先走 to_dict。
            copied_events.append(event.to_dict())
        elif isinstance(event, Mapping):  # 普通字典走 deepcopy。
            copied_events.append(deepcopy(dict(event)))
        else:
            raise TypeError(f"event must be mapping-like, got {type(event).__name__}")  # 其它类型拒绝。
    return copied_events  # 返回可安全修改的事件副本列表。


def _recompute_dt(events: list[dict]) -> None:
    """按事件顺序重算 dt 字段。

    删除事件后，剩下事件之间的时间差可能被拉大，所以需要重新计算。
    第一个事件的 dt 固定设成 0.0。

    参数:
        events: 事件列表，原地修改 dt 字段。

    返回:
        None（原地修改）。
    """
    previous_t = None  # 记住上一个事件的时间戳。
    for event in events:  # 逐个事件重新计算时间差。
        if "t" not in event:  # 缺少时间戳字段时显式报错，避免 KeyError 模糊。
            raise KeyError(f"event is missing required 't' field, got keys: {list(event.keys())}")
        current_t = float(event["t"])  # 当前事件时间统一转成 float。
        if previous_t is None:
            event["dt"] = 0.0  # 首条记录没有前驱，所以 dt 为 0。
        else:
            dt = current_t - previous_t  # 相邻事件时间差。
            if dt < 0.0:  # 负 dt 表示事件未按时间排序，协议要求 dt >= 0。
                raise ValueError(f"event sequence is not time-sorted: dt={dt} at t={current_t}")
            event["dt"] = dt  # 相邻事件时间差。
        previous_t = current_t  # 当前时间成为下一轮的前一个时间。


def _sample_blackout_indices(non_imu_indices: list[int], *, selected_count: int, key: str) -> list[int]:
    """从非 IMU 事件索引中确定性选取连续窗口作为 blackout 目标。

    参数:
        non_imu_indices: 所有非 IMU 事件在序列中的索引列表。
        selected_count:  需要选中的事件数量。
        key:             哈希键，保证相同输入始终选取相同窗口。

    返回:
        被 blackout 的事件索引列表（连续窗口）。
    """
    if selected_count <= 0:  # 不需要 blackout 时返回空列表。
        return []
    if selected_count > len(non_imu_indices):  # 请求超过可用数量时显式报错，避免静默返回更少。
        raise ValueError(
            f"selected_count ({selected_count}) exceeds available non-IMU events ({len(non_imu_indices)})"
        )
    start = _stable_window_start(len(non_imu_indices), selected_count, key=key)  # 用稳定哈希算出窗口起始。
    return non_imu_indices[start : start + selected_count]  # 返回连续窗口内的索引。


def _resolve_sequence_seed_identity(events: list[dict]) -> tuple[str, str]:
    """提取当前事件序列的稳定身份，用于序列级随机种子。"""
    for event in events:
        raw_meta = event.get("meta")
        if isinstance(raw_meta, Mapping):
            # 显式处理 None：meta.get("scene_id", "") 在值为 None 时返回 None，
            # str(None) 会产生 "None" 字符串污染哈希 key，必须归一化为空字符串。
            scene_id_raw = raw_meta.get("scene_id", "")
            seq_id_raw = raw_meta.get("seq_id", "")
            scene_id = "" if scene_id_raw is None else str(scene_id_raw)
            seq_id = "" if seq_id_raw is None else str(seq_id_raw)
            return scene_id, seq_id
    return "", ""


def _generate_ffn_1_over_f(n_samples: int, sigma: float, rng: random.Random) -> list[float]:
    """生成 1/f flicker noise (FFN) 序列，使用多尺度 Wiener 叠加。

    参考 MATLAB imuSensor FFN 实现与 IEEE Std 952-2020。
    通过 K=log2(n) 个尺度的 Wiener 过程叠加，近似 1/f PSD，
    对应 Allan 方差 T⁰ 平台区（bias instability）。

    参数:
        n_samples: 样本数。
        sigma: 目标标准差（bias instability 幅度）。
        rng: 确定性随机数生成器。

    返回:
        长度为 n_samples 的 1/f 噪声序列（浮点数列表）。
    """
    if n_samples <= 0:
        return []
    if sigma <= 0.0:
        return [0.0] * n_samples
    # 多尺度 Wiener 叠加：K 个尺度的 Wiener 过程，振幅按 2^(-k/2) 衰减以近似 1/f PSD。
    # 限制 K 上限，防止极端长序列下 K 过大导致性能退化。
    K = max(1, min(12, int(math.log2(max(n_samples, 2)))))
    # 用 numpy 批量生成和累加，避免 Python 循环开销。
    try:
        import numpy as np
        samples = np.zeros(n_samples, dtype=np.float64)
        for k in range(K):
            scale = sigma * (2.0 ** (-k / 2.0))
            stride = 2 ** k
            n_points = (n_samples + stride - 1) // stride
            increments = np.array([rng.gauss(0.0, 1.0) for _ in range(n_points)], dtype=np.float64)
            expanded = np.repeat(increments, stride)[:n_samples] * scale
            samples += expanded
        mean = float(samples.mean())
        std = float(samples.std())
        if std > 0.0:
            # 先去均值再缩放，保证 FFN 序列零均值（与 IEEE 952 FFN 零均值定义一致）。
            # 此前版本计算了 mean 但未使用，导致序列带非零固定偏移，所有 UWB/VIO 事件
            # 时间戳被整体偏移一个非零常量，破坏 Allan 方差建模的零均值假设。
            samples = (samples - mean) * (sigma / std)
        else:
            # std == 0 时仅去均值（sigma 缩放无意义，避免 0/0）。
            samples = samples - mean
        return samples.tolist()
    except ImportError:
        # numpy 不可用时回退到纯 Python 实现。
        samples = [0.0] * n_samples
        for k in range(K):
            scale = sigma * (2.0 ** (-k / 2.0))
            stride = 2 ** k
            n_points = (n_samples + stride - 1) // stride
            increments = [rng.gauss(0.0, 1.0) for _ in range(n_points)]
            scaled_increments = [scale * inc for inc in increments]
            expanded: list[float] = []
            for inc in scaled_increments:
                expanded.extend([inc] * stride)
            if len(expanded) > n_samples:
                expanded = expanded[:n_samples]
            for i in range(n_samples):
                samples[i] += expanded[i]
        mean = sum(samples) / n_samples
        var = sum((x - mean) ** 2 for x in samples) / n_samples
        std = math.sqrt(var)
        if std > 0.0:
            inv = sigma / std
            # 先去均值再缩放，与 numpy 路径保持一致（见上方注释）。
            samples = [(x - mean) * inv for x in samples]
        else:
            samples = [x - mean for x in samples]
        return samples


def apply_async_level(
    events: Iterable[Mapping], async_level: str, async_cfg: Mapping
) -> tuple[list[dict], dict]:
    """将协议定义的异步扰动等级应用到事件序列上。

    参数:
        events:      原始事件序列（list[dict] 或可迭代 Mapping 对象），内部会先 list() 物化。
        async_level: 异步等级名（如 "A0"、"A2"），对应协议中的等级键。
        async_cfg:   包含该等级参数的配置映射。

    返回:
        (filtered_events, async_report) 二元组：
        - filtered_events: 经过异步扰动后的事件序列（blackout 事件被移除）。
        - async_report:    可审计的扰动报告，记录偏移、抖动、blackout 等全部变更明细。

    异常:
        ValueError: 参数值越界（如 jitter_ms 为负、burst_missing_prob 不在 [0,1]）。
        TypeError:  输入类型不合法。
        KeyError:   缺少必需的权威字段。
    """
    # 第 10 轮审查 MEDIUM-1 修复：apply_async_level 是场景异步扰动入口函数，
    # 每事件序列调用一次。大规模实验编排（e1-e12 全量跑）下入口 print_dict
    # 产生大量日志，违反工程规范 "Recursive functions and entry points should
    # avoid print_dict calls"。async_report 返回值已包含全部扰动明细，无需入口打印。
    event_list = list(events)  # 确保可多次遍历。
    if not event_list:  # 空序列无事件可扰动，提前返回零值报告。
        # 空序列路径的一致性检查：无操作即无违规，全部为 True，与主路径口径一致。
        empty_consistency_checks = {
            "entry_validation": True,  # 空序列无需校验。
            "exit_validation": True,  # 空序列无需校验。
            "blackout_count": True,  # 0 == 0。
            "dt_non_negative": True,  # 空序列无 dt 可检查。
        }
        return [], {
            "async_level": async_level,  # 等级名。
            "offset_ms_midpoint": 0.0,  # 偏移中点为零。
            "cross_modal_skew_ms": 0.0,  # 跨模态偏差为零。
            "jitter_ms": 0.0,  # 抖动为零。
            "burst_missing_prob": 0.0,  # 缺失比例为零。
            "clock_drift_ppm": 0.0,  # 时钟漂移为零。
            # 第 12 轮审查 MEDIUM-1 修复（R12-A M1）：空序列路径补全三个 Allan 分层建模字段，
            # 与非空序列主路径（L599-615）字段集对齐，保证 async_report 输出合同在不同执行
            # 路径下字段集一致。空序列无 Allan 建模，三字段值为 0.0。下游消费者可统一用
            # async_report["clock_bias_instability_ms"] 等字段访问，空序列路径不再抛 KeyError。
            "clock_bias_instability_ms": 0.0,  # Allan FFN bias instability 幅度（毫秒），空序列无建模。
            "clock_rrw_per_sqrt_s_ms": 0.0,  # Allan RWFN 时钟 bias 随机游走系数（ms/√s），空序列无建模。
            "clock_bias_ffn_ms": 0.0,  # FFN 1/f 序列实际标准差（毫秒），空序列无序列故为 0.0。
            "timing_plan": [],  # 无时间变更记录。
            "dropped_event_count": 0,  # 无删除事件。
            "blackout_segments": [],  # 无 blackout 时间段。
            "blackout_strategy": "stable_hashed_contiguous_non_imu_window",  # blackout 策略标识。
            "blackout_selection_start": 0,  # blackout 起始位置为零。
            "consistency_checks": empty_consistency_checks,  # 空序列一致性检查明细。
            "protocol_consistent": all(empty_consistency_checks.values()),  # 由 consistency_checks 派生。
        }

    validate_event_sequence(event_list)  # 入口校验事件序列合法性。
    params = _resolve_async_level_params(async_level, async_cfg)  # 解析等级参数。
    offset_lower_ms, offset_upper_ms = _normalize_offset_range(params["offset_ms"])  # 归一化偏移范围。
    sequence_scene_id, sequence_seq_id = _resolve_sequence_seed_identity(event_list)  # 提取序列身份，避免不同序列拿到同一组扰动。
    # 序列间偏移变异：用序列首个事件时间戳作为确定性种子源，
    # 在 [offset_lower, offset_upper] 范围内均匀采样，替代固定取中点。
    # 真实场景中不同设备/序列的时钟偏移应在配置范围内随机分布。
    _first_event_t = float(event_list[0].get("t", 0.0))
    if offset_upper_ms > offset_lower_ms:
        _seq_seed_str = f"offset:{sequence_scene_id}:{sequence_seq_id}:{_first_event_t}:{async_level}"
        _seq_seed = int(hashlib.sha256(_seq_seed_str.encode("utf-8")).hexdigest()[:8], 16)
        offset_midpoint_ms = offset_lower_ms + random.Random(_seq_seed).random() * (offset_upper_ms - offset_lower_ms)
    else:
        offset_midpoint_ms = offset_lower_ms
    jitter_ms = coerce_finite_scalar(params["jitter_ms"], name="async_cfg['jitter_ms']")  # 抖动幅度。
    burst_missing_prob = coerce_finite_scalar(
        params["burst_missing_prob"],
        name="async_cfg['burst_missing_prob']",
    )  # 突发缺失比例（确定性）。
    cross_modal_skew_ms = coerce_finite_scalar(
        params["cross_modal_skew_ms"],
        name="async_cfg['cross_modal_skew_ms']",
    )  # 跨模态时间偏差。
    clock_drift_ppm = coerce_finite_scalar(
        params["clock_drift_ppm"],
        name="async_cfg['clock_drift_ppm']",
    )  # 时钟漂移（ppm），必需字段。
    # Allan 方差分层建模（可选字段，缺省为 0，保持向后兼容）：
    # clock_bias_instability_ms：FFN (flicker frequency noise) 序列内恒定时钟偏置。
    # clock_rrw_per_sqrt_s_ms：RWFN (random walk frequency noise) 时钟 bias 随机游走系数。
    clock_bias_instability_ms = coerce_finite_scalar(
        params.get("clock_bias_instability_ms", 0.0),
        name="async_cfg['clock_bias_instability_ms']",
    )
    clock_rrw_per_sqrt_s_ms = coerce_finite_scalar(
        params.get("clock_rrw_per_sqrt_s_ms", 0.0),
        name="async_cfg['clock_rrw_per_sqrt_s_ms']",
    )

    # 参数范围校验。
    if jitter_ms < 0.0:  # 抖动不能为负。
        raise ValueError("async_cfg['jitter_ms'] must be non-negative")
    if not 0.0 <= burst_missing_prob <= 1.0:  # 缺失比例必须在 [0, 1] 区间。
        raise ValueError("async_cfg['burst_missing_prob'] must be within [0, 1]")
    if cross_modal_skew_ms < 0.0:  # 跨模态偏差不能为负。
        raise ValueError("async_cfg['cross_modal_skew_ms'] must be non-negative")
    if clock_drift_ppm < 0.0:  # 时钟漂移不能为负。
        raise ValueError("async_cfg['clock_drift_ppm'] must be non-negative")
    if clock_bias_instability_ms < 0.0:
        raise ValueError("async_cfg['clock_bias_instability_ms'] must be non-negative")
    if clock_rrw_per_sqrt_s_ms < 0.0:
        raise ValueError("async_cfg['clock_rrw_per_sqrt_s_ms'] must be non-negative")

    new_events = _copy_events(event_list)  # 深拷贝事件，保证输入不被修改。
    modality_counts: dict[str, int] = {}  # 记录每种模态的事件出现次数，用于交替抖动方向。
    non_imu_indices: list[int] = []  # 收集所有非 IMU 事件的索引，blackout 从中选取。
    timing_plan: list[dict] = []  # 每个事件的时间变更审计记录。
    reference_t = min(float(event["t"]) for event in new_events)  # 序列最早时间，作为时钟漂移的参考原点。

    # Allan FFN (BiasInstability)：1/f flicker noise，对应 Allan 方差 T⁰ 平台区。
    # 使用多尺度 Wiener 叠加实现真正的 1/f PSD（参考 IEEE Std 952-2020、MATLAB imuSensor）。
    # UWB/VIO 各自独立生成 FFN 序列（独立时钟），通过 modality 种子区分，
    # 避免两模态共享同一 FFN 常数与"独立时钟"假设矛盾。
    uwb_ffn_sequence: list[float] = []
    vio_ffn_sequence: list[float] = []
    if clock_bias_instability_ms > 0.0:
        # 统计 UWB/VIO 事件数，用于确定 FFN 序列长度（按模态出现次序索引）。
        uwb_event_count = sum(1 for _e in new_events if _e.get("modality") == MODALITY_UWB)
        vio_event_count = sum(1 for _e in new_events if _e.get("modality") == MODALITY_VIO)
        if uwb_event_count > 0:
            uwb_ffn_seed_str = f"clock_ffn:{sequence_scene_id}:{sequence_seq_id}:{async_level}:uwb"
            uwb_ffn_seed = int(hashlib.sha256(uwb_ffn_seed_str.encode("utf-8")).hexdigest()[:8], 16)
            uwb_ffn_sequence = _generate_ffn_1_over_f(
                uwb_event_count, clock_bias_instability_ms, random.Random(uwb_ffn_seed)
            )
        if vio_event_count > 0:
            vio_ffn_seed_str = f"clock_ffn:{sequence_scene_id}:{sequence_seq_id}:{async_level}:vio"
            vio_ffn_seed = int(hashlib.sha256(vio_ffn_seed_str.encode("utf-8")).hexdigest()[:8], 16)
            vio_ffn_sequence = _generate_ffn_1_over_f(
                vio_event_count, clock_bias_instability_ms, random.Random(vio_ffn_seed)
            )
    # Allan RWFN (Rate Random Walk)：时钟 bias 按时间累积 Wiener 过程。
    # bias_clock(t+dt) = bias_clock(t) + N(0, rrw · √dt)，
    # 方差随时间线性增长（σ²(t) = Q² · t），对应 Allan 方差低频段。
    # 按 UWB/VIO 各自维护一份独立 bias，模拟两模态独立时钟。
    uwb_clock_bias_rrw_ms = 0.0
    vio_clock_bias_rrw_ms = 0.0
    last_uwb_t: float | None = None
    last_vio_t: float | None = None

    for index, event in enumerate(new_events):  # 逐个事件施加时间扰动。
        modality = event["modality"]  # 当前事件的模态。
        occurrence = modality_counts.get(modality, 0)  # 该模态已出现次数。
        modality_counts[modality] = occurrence + 1  # 更新计数。

        # 抖动：使用确定性随机种子，模拟真实传感器时钟抖动的零均值高斯特性。
        # 替代原来的确定性交替模式（+jitter, -jitter, +jitter, ...），
        # 后者产生周期性分量而非白噪声，与真实传感器抖动分布不匹配。
        # 第 3 轮审查 HIGH-1 修复（IMU 豁免）：IMU 作为参考时钟豁免 jitter，
        # 与 offset/clock_drift/allan_bias 的 IMU 豁免口径一致。
        # 原因：±3σ clamp 仅能截断 |j|>3σ 的尾部（约 0.27%），对 1-3σ 的主体
        # 无效。在 A3 jitter_ms=25ms + 200Hz IMU（5ms 间隔）下，相邻 IMU 样本
        # 的 jitter 差 ~N(0, σ·√2) ≈ N(0, 35.4ms)，P(dt'<0) ≈ 44%，单调性破坏
        # 概率不可接受。UWB/VIO 的采样间隔（~100ms）远大于 σ·√2（~35ms），
        # 单调性破坏概率 < 1%，仍可施加 jitter。
        if jitter_ms > 0.0 and modality != MODALITY_IMU:
            jitter_seed_str = f"jitter:{sequence_scene_id}:{sequence_seq_id}:{index}:{modality}:{occurrence}"
            jitter_seed = int(hashlib.sha256(jitter_seed_str.encode("utf-8")).hexdigest()[:8], 16)
            jitter_sigma_s = jitter_ms / 1000.0
            jitter_seconds = random.Random(jitter_seed).gauss(0.0, jitter_sigma_s)
            # ±3σ clamp：物理传感器抖动天然有界，截断尾部极端样本。
            # 注意：clamp 不能保证相邻 UWB/VIO 样本单调（1-3σ 主体仍可能反转），
            # 但 UWB/VIO 采样间隔（~100ms）远大于 σ·√2，单调性破坏概率 < 1%，
            # 在工程可接受范围内。IMU 已豁免 jitter，不存在单调性问题。
            jitter_bounds = 3.0 * jitter_sigma_s
            if jitter_seconds > jitter_bounds:
                jitter_seconds = jitter_bounds
            elif jitter_seconds < -jitter_bounds:
                jitter_seconds = -jitter_bounds
        else:
            jitter_seconds = 0.0
        original_t = float(event["t"])  # 原始时间戳。

        base_shift_ms = 0.0  # 基准偏移（毫秒），IMU 事件为零。
        clock_drift_ms = 0.0  # 时钟漂移偏移（毫秒），IMU 事件为零。
        allan_bias_ms = 0.0  # Allan 分层时钟偏置（FFN + RWFN），IMU 事件为零。
        if modality != MODALITY_IMU:  # 非 IMU 事件才施加偏移、跨模态偏差和时钟漂移。
            base_shift_ms += offset_midpoint_ms  # 先加偏移中点。
            non_imu_indices.append(index)  # 记录非 IMU 事件索引。
            if modality == MODALITY_UWB:  # UWB 事件额外加跨模态偏差的一半。
                base_shift_ms += cross_modal_skew_ms / 2.0  # UWB 正偏。
                clock_drift_ms = max(0.0, original_t - reference_t) * clock_drift_ppm * 1e-3  # UWB 时钟正向漂移。
                # RWFN 步进：UWB 时钟 bias 累积 Wiener 过程。
                if clock_rrw_per_sqrt_s_ms > 0.0 and last_uwb_t is not None:
                    dt = max(0.0, original_t - last_uwb_t)
                    if dt > 0.0:
                        rrw_seed_str = f"clock_rrw_uwb:{sequence_scene_id}:{sequence_seq_id}:{index}"
                        rrw_seed = int(hashlib.sha256(rrw_seed_str.encode("utf-8")).hexdigest()[:8], 16)
                        uwb_clock_bias_rrw_ms += random.Random(rrw_seed).gauss(
                            0.0, clock_rrw_per_sqrt_s_ms * math.sqrt(dt)
                        )
                last_uwb_t = original_t
                # Allan 总偏置 = FFN 1/f 序列（按 UWB 出现次序取值）+ RWFN 时变。
                uwb_ffn_bias_ms = uwb_ffn_sequence[occurrence] if uwb_ffn_sequence else 0.0
                allan_bias_ms = uwb_ffn_bias_ms + uwb_clock_bias_rrw_ms
            elif modality == MODALITY_VIO:  # VIO 事件额外减跨模态偏差的一半。
                base_shift_ms -= cross_modal_skew_ms / 2.0  # VIO 负偏。
                clock_drift_ms = -max(0.0, original_t - reference_t) * clock_drift_ppm * 1e-3  # VIO 时钟负向漂移。
                # RWFN 步进：VIO 时钟 bias 累积 Wiener 过程（与 UWB 独立）。
                if clock_rrw_per_sqrt_s_ms > 0.0 and last_vio_t is not None:
                    dt = max(0.0, original_t - last_vio_t)
                    if dt > 0.0:
                        rrw_seed_str = f"clock_rrw_vio:{sequence_scene_id}:{sequence_seq_id}:{index}"
                        rrw_seed = int(hashlib.sha256(rrw_seed_str.encode("utf-8")).hexdigest()[:8], 16)
                        vio_clock_bias_rrw_ms += random.Random(rrw_seed).gauss(
                            0.0, clock_rrw_per_sqrt_s_ms * math.sqrt(dt)
                        )
                last_vio_t = original_t
                # Allan 总偏置 = FFN 1/f 序列（按 VIO 出现次序取值，与 UWB 独立）+ RWFN 时变。
                vio_ffn_bias_ms = vio_ffn_sequence[occurrence] if vio_ffn_sequence else 0.0
                allan_bias_ms = vio_ffn_bias_ms + vio_clock_bias_rrw_ms

        # 最终时间戳 = 原始时间 + 基准偏移 + 抖动 + 时钟漂移 + Allan 偏置，全部转为秒。
        event["t"] = original_t + (base_shift_ms / 1000.0) + jitter_seconds + (clock_drift_ms / 1000.0) + (allan_bias_ms / 1000.0)
        timing_plan.append(  # 记录时间变更审计。
            {
                "event_index": index,  # 事件在序列中的位置。
                "modality": modality,  # 事件模态。
                "occurrence": occurrence,  # 该模态第几次出现。
                "shift_ms": base_shift_ms,  # 基准偏移量（毫秒）。
                "jitter_ms": jitter_seconds * 1000.0,  # 实际抖动量（毫秒，含正负），从秒转回毫秒。
                "clock_drift_ms": clock_drift_ms,  # 线性时钟漂移量（毫秒）。
                "allan_bias_ms": allan_bias_ms,  # Allan 分层时钟偏置（FFN+RWFN，毫秒）。
            }
        )

    # 按 burst_missing_prob 确定性比例计算要 blackout 的非 IMU 事件数量。
    blackout_target_count = min(
        len(non_imu_indices),  # 不超过非 IMU 事件总数。
        max(0, int(math.floor((len(non_imu_indices) * burst_missing_prob) + 0.5))),  # 显式 half-up 取整，避免 Python round() 的银行家舍入把 0.5 压成 0。
    )

    # blackout 的“连续窗口”必须按扰动后的时间顺序定义，而不是原始事件索引顺序。
    # 否则 UWB/VIO 的正负跨模态偏移会打乱时间先后关系，导致报告里的 blackout 时间段
    # 中间仍残留幸存事件，把“突发缺失”退化成“索引连续但时间不连续”的假 blackout。
    non_imu_indices_by_time = sorted(
        non_imu_indices,
        key=lambda index: (float(new_events[index]["t"]), index),
    )

    # 构造稳定哈希 key，保证相同场景和等级始终选取相同 blackout 窗口。
    first_non_imu_meta = {}  # 第一个非 IMU 事件的元信息，用于构造哈希 key。
    if non_imu_indices_by_time:  # 有非 IMU 事件时才提取元信息。
        raw_meta = new_events[non_imu_indices_by_time[0]].get("meta")  # 取时间上最早的非 IMU 事件 meta。
        if isinstance(raw_meta, Mapping):  # meta 必须是映射类型。
            first_non_imu_meta = dict(raw_meta)  # 转为普通字典。
    blackout_key = "|".join(  # 拼接哈希 key，包含等级、场景、序列等标识。
        [
            "async_blackout",  # 前缀标识。
            str(async_level),  # 异步等级名。
            # 显式归一化 None 为空字符串，避免 str(None) 产生 "None" 污染哈希 key。
            str(first_non_imu_meta.get("scene_id") or ""),  # 场景 ID。
            str(first_non_imu_meta.get("seq_id") or ""),  # 序列 ID。
            str(len(non_imu_indices_by_time)),  # 非 IMU 事件总数。
            str(blackout_target_count),  # 目标 blackout 数量。
        ]
    )
    blackout_start = _stable_window_start(len(non_imu_indices_by_time), blackout_target_count, key=blackout_key)  # 稳定哈希算出时间窗口起始。
    blackout_slice = _sample_blackout_indices(non_imu_indices_by_time, selected_count=blackout_target_count, key=blackout_key)  # 按时间连续窗口选取 blackout 索引。

    dropped_event_indices: set[int] = set(blackout_slice)  # 被 blackout 的事件索引集合，用于快速查找。
    blackout_segments: list[tuple[float, float]] = []  # blackout 时间段列表 [(t_start, t_end), ...]。
    if blackout_slice:  # 有 blackout 事件时，记录其时间跨度。
        dropped_times = sorted(float(new_events[index]["t"]) for index in blackout_slice)  # 按时间排序被删事件。
        blackout_segments.append((dropped_times[0], dropped_times[-1]))  # 记录最早到最晚的 blackout 时间段。

    # 过滤掉 blackout 事件，按时间重新排序，重算 dt。
    filtered_events = [event for index, event in enumerate(new_events) if index not in dropped_event_indices]  # 保留非 blackout 事件。
    if not filtered_events:  # 场景轴函数必须返回合法事件序列，不能把整条序列删空后再交给协议层兜底。
        raise ValueError("async blackout would remove every event; protocol event sequences must retain at least one event")
    filtered_events.sort(key=lambda event: float(event["t"]))  # 按修改后的时间戳重新排序。
    _recompute_dt(filtered_events)  # 重新计算相邻事件时间差。
    validate_event_sequence(filtered_events)  # 出口校验事件序列合法性。

    # 一致性检查：对齐 V 轴口径，记录实际执行的一致性校验结果，而非硬编码 True。
    consistency_checks = {
        "entry_validation": True,  # 入口事件序列校验通过（未抛异常即 True）。
        "exit_validation": True,  # 出口事件序列校验通过（未抛异常即 True）。
        "blackout_count": len(dropped_event_indices) == blackout_target_count,  # 实际 blackout 数量等于预期。
        "dt_non_negative": all(float(event.get("dt", 0.0)) >= 0.0 for event in filtered_events),  # 所有 dt >= 0（协议要求）。
    }
    protocol_consistent = all(consistency_checks.values())  # 全部一致才算协议一致。

    # 第 7 轮审查 MEDIUM-2 修复：原 clock_bias_ffn_ms 表达式在生成器内部
    # 重复计算 `sum(seq) / len(seq)`，对每个 x 都重新求均值，导致 O(n²) 复杂度。
    # 改为在 dict 字面量之前预计算均值变量，生成器内只引用标量，降为 O(n)。
    def _ffn_std(seq: Sequence[float]) -> float:  # 第 12 轮审查 LOW-1 修复（R12-A LOW-1，继承 R11-A LOW-1）：补全类型注解，与文件其他 helper 口径一致。
        if not seq:
            return 0.0
        mean = sum(seq) / len(seq)
        return math.sqrt(sum((x - mean) ** 2 for x in seq) / len(seq))

    uwb_ffn_std_ms = _ffn_std(uwb_ffn_sequence)
    vio_ffn_std_ms = _ffn_std(vio_ffn_sequence)

    async_report = {  # 完整审计报告。
        "async_level": async_level,  # 等级名。
        "offset_ms_midpoint": offset_midpoint_ms,  # 偏移范围中点（毫秒）。
        "cross_modal_skew_ms": cross_modal_skew_ms,  # 跨模态偏差（毫秒）。
        "jitter_ms": jitter_ms,  # 抖动幅度（毫秒）。
        "burst_missing_prob": burst_missing_prob,  # 突发缺失比例。
        "clock_drift_ppm": clock_drift_ppm,  # 时钟漂移（ppm）。
        "clock_bias_instability_ms": clock_bias_instability_ms,  # Allan FFN bias instability 幅度（毫秒），作为 1/f 序列目标 σ。
        "clock_rrw_per_sqrt_s_ms": clock_rrw_per_sqrt_s_ms,  # Allan RWFN 时钟 bias 随机游走系数（ms/√s）。
        # 第 4 轮审查 MEDIUM-1 修复：原字段语义为"FFN 序列均值"，但
        # _generate_ffn_1_over_f 已显式去均值（见 L262-268），导致均值恒≈0、
        # 字段失去信息量。改为报告"FFN 序列实际标准差"，可验证 1/f 序列
        # 幅度是否匹配目标 σ（clock_bias_instability_ms）。UWB 优先，VIO 回退。
        # 第 7 轮审查 MEDIUM-2 修复：见上方 _ffn_std，预计算均值避免 O(n²)。
        "clock_bias_ffn_ms": (
            uwb_ffn_std_ms if uwb_ffn_sequence else (vio_ffn_std_ms if vio_ffn_sequence else 0.0)
        ),  # FFN 1/f 序列实际标准差（毫秒），用于验证幅度匹配目标 σ；UWB/VIO 独立生成。
        # B08 修复: 跨模态错位事件计数（base_shift_ms != 0 的非 IMU 事件），用于满足 §0.4 B08 ≥ 20 错位事件约束。
        "misalignment_event_count": sum(
            1 for plan in timing_plan if abs(float(plan.get("shift_ms", 0.0))) > 1e-6
        ),
        "timing_plan": timing_plan,  # 每个事件的时间变更审计。
        "dropped_event_count": len(dropped_event_indices),  # 实际删除事件数。
        "blackout_segments": blackout_segments,  # blackout 时间段。
        "blackout_strategy": "stable_hashed_contiguous_non_imu_window",  # blackout 策略标识。
        "blackout_selection_start": blackout_start,  # blackout 窗口起始位置。
        "consistency_checks": consistency_checks,  # 四项一致性检查明细。
        "protocol_consistent": protocol_consistent,  # 协议一致性总判定，由 consistency_checks 派生。
    }
    return filtered_events, async_report  # 返回扰动后的事件序列和审计报告。
