"""融合主运行器——驱动整条"事件 → 模型推理 → 控制量 → 估计器更新"链路。

本模块是 `liquidloc.fusion` 包的核心，负责：
1. 逐事件驱动估计器（EKF）的预测-更新循环
2. 在每个测量事件到来时，调用模型推理得到中间控制量（bias / risk / scaling）
3. 将中间控制量经桥接合约（liquid_bridge_contract）转换为 MeasurementControl，
   再注入估计器，完成"模型调控几何后端"的闭环
4. 收集诊断 trace（risk_trace、bias_trace、scaling_trace 等），供后续分析和绘图消费

核心概念：
- **中间控制量（ModelIntermediate）**：模型输出的四维控制信号（bias, risk, uwb_scaling, vio_scaling），
  是"模型意图"的数值表达
- **测量控制（MeasurementControl）**：经桥接合约和安全模式收缩后，实际注入估计器的控制指令，
  包含 bias_applied、risk、scaling、noise_multiplier、gate_action 等字段
- **读出上下文（readout_context）**：从估计器状态和更新报告中提取的实时诊断信息，
  作为模型特征的一部分反馈给下一轮推理

分层边界：
- 本模块属于 **fusion 桥接层**，只做"调度"和"数据搬运"，不实现几何算法或模型推理
- 估计器（estimator）是经典几何后端，本模块不修改其内部状态
- 模型特例不得通过本模块侵入 estimator 的几何逻辑

与其他模块的关系：
- `protocol.event_schema`：提供事件校验
- `protocol.liquid_bridge_contract`：提供中间控制量 → MeasurementControl 的转换
- `common.types`：定义 ModelIntermediate、MeasurementControl 等数据类型
- `common.constants`：提供默认阈值
- `fusion.bias_adapter` / `covariance_adapter` / `risk_adapter` / `safe_mode`：
  本模块是它们的上游调度者，但当前主循环直接使用桥接合约完成控制量转换
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

from liquidloc.common.constants import MODALITY_IMU, MODALITY_UWB, MODALITY_VIO, BRIDGE_BIAS_MAX
from liquidloc.common.constants import LIQUID_READOUT_CONTEXT_KEYS as _READOUT_CONTEXT_KEYS  # 读出上下文键名单源真相，别名保留以避免改下游引用。
from liquidloc.common.constants import LIQUID_READOUT_CONTEXT_CACHE_DEFAULTS as _READOUT_CONTEXT_CACHE_DEFAULTS  # 读出上下文模态缓存键名及默认值单源真相（D9 单源，训练/推理同口径）。
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
from liquidloc.common.types import MeasurementControl, ModelIntermediate
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_real
from liquidloc.protocol.event_schema import Event, validate_event_sequence
from liquidloc.protocol.liquid_bridge_contract import build_measurement_control, clip_uwb_bias

# Bug 1 (2026-07-23 audit Round 1+): fusion_runner 静默 fallback 可观测性.
# step_joint 抛异常时原状是完全静默退化逐事件 fallback, 审稿无法知道 fallback 比例.
# 此处加 logger + 计数 + 异常类型记录, 不改数学路径, 让审稿可看 fallback ratio.
_LOGGER = logging.getLogger("liquidloc.fusion.fusion_runner")

# 模型输出字典中必须包含的四个键，用于构造 ModelIntermediate。
# 任何多余或缺失的键都会在 _coerce_intermediate 中被拒绝，
# 防止模型偷偷输出未定义的控制信号。
_INTERMEDIATE_KEYS = ("bias", "risk", "uwb_scaling", "vio_scaling")




# 读出上下文按模态（uwb / vio）分别维护，因为两种传感器的更新节奏和门控行为不同。
_READOUT_CONTEXT_MODALITIES = (MODALITY_UWB, MODALITY_VIO)


def _compute_initial_position_from_anchor_geometry(
    events: Sequence[Mapping[str, Any]],
    anchor_layout: Mapping[str, Any] | None,
) -> dict[str, float] | None:
    """用 anchor 几何 + 首个 UWB 测距做 least-squares 三角定位，计算 EKF 初始位置。

    背景：EKF 默认从 (px=0, py=0) 冷启动，但真实轨迹的起始位置在 anchor 几何
    范围内（例如 (-11.6, -8.3)），模型从未见过非零起始位置，因此预测的位移
    始终围绕 (0,0) 小幅漂移，无法跟踪目标的实际运动。

    本函数从第一个 UWB 事件提取 anchor_id 和 range，结合 anchor_layout 中的
    anchor_positions，用 3 锚点 least-squares 三角定位求解初始 (px, py)，
    作为 EKF 的 initial_state 传入，使滤波器从真实起始位置附近启动。

    参数
    ----
    events : sequence of mapping
        事件序列，至少包含一个 UWB 事件。
    anchor_layout : mapping or None
        anchor 布局，需含 ``anchor_ids`` 和 ``anchor_positions``。

    返回
    ----
    dict or None
        包含 ``px``/``py`` 的初始状态字典，或定位失败时返回 None。
    """
    if anchor_layout is None:
        return None

    anchor_ids = anchor_layout.get("anchor_ids")
    anchor_positions = anchor_layout.get("anchor_positions")
    if not anchor_ids or not anchor_positions:
        return None

    # 构建 anchor_id -> (x, y) 查找表。
    anchor_map: dict[str, tuple[float, float]] = {}
    for aid, pos in zip(anchor_ids, anchor_positions):
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            anchor_map[str(aid)] = (float(pos[0]), float(pos[1]))

    if len(anchor_map) < 3:
        return None

    # 找到第一个 UWB 事件。
    first_uwb = None
    for event in events:
        if event.get("modality") == MODALITY_UWB:
            first_uwb = event
            break

    if first_uwb is None:
        return None

    uwb_payload = first_uwb.get("uwb_payload")
    if not uwb_payload:
        return None

    anchor_id = str(uwb_payload.get("anchor_id", ""))
    range_val = uwb_payload.get("range")
    if anchor_id not in anchor_map or range_val is None:
        return None

    # 用前 3 个 UWB 事件做 least-squares 三角定位。
    uwb_events = [e for e in events if e.get("modality") == MODALITY_UWB]
    if len(uwb_events) < 3:
        return None

    # 取前 3 个不同 anchor 的测距。
    anchors_used: list[tuple[str, float]] = []
    seen: set[str] = set()
    for e in uwb_events:
        p = e.get("uwb_payload")
        if not p:
            continue
        aid = str(p.get("anchor_id", ""))
        r = p.get("range")
        if aid in anchor_map and aid not in seen and r is not None:
            anchors_used.append((aid, float(r)))
            seen.add(aid)
        if len(anchors_used) == 3:
            break

    if len(anchors_used) < 3:
        return None

    # 精确三角定位：用 3 个 anchor 的测距做 least-squares 求解。
    # 方程：||x - a_i|| = r_i, i=1,2,3。
    # 用第 1 个 anchor 作为参考点，从 ||x-a_i||^2 = r_i^2 中消去二次项，
    # 得到关于 x 的线性方程组 A·x = b，用 lstsq 求解（3 方程 2 未知数）。
    import numpy as np

    a0 = np.array(anchor_map[anchors_used[0][0]], dtype=float)
    a_coords = np.array([anchor_map[aid] for aid, _ in anchors_used], dtype=float)
    r_obs = np.array([r for _, r in anchors_used], dtype=float)

    d01 = a_coords[1] - a0
    d02 = a_coords[2] - a0
    b = np.array([
        r_obs[0] ** 2 - r_obs[1] ** 2 + np.dot(a_coords[1], a_coords[1]) - np.dot(a0, a0),
        r_obs[0] ** 2 - r_obs[2] ** 2 + np.dot(a_coords[2], a_coords[2]) - np.dot(a0, a0),
    ])
    A = np.array([2.0 * d01, 2.0 * d02])

    try:
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    # 验证残差。
    final_residuals = np.empty(len(anchors_used))
    for i, (aid, _) in enumerate(anchors_used):
        ax, ay = a_coords[i]
        dist = math.hypot(x[0] - ax, x[1] - ay)
        final_residuals[i] = dist - r_obs[i]

    rmse = math.sqrt(float(np.mean(final_residuals ** 2)))
    # 当 anchor 共线（如 sim_curve_02 的 4 锚点全在 x 轴上）时，2D 三角定位退化为 1D，
    # 线性方程组系数矩阵 A 的两行线性相关，lstsq 给出的 y 坐标被约束为 0（投影到
    # anchor 线上），残差 RMSE 会偏大（>1.0m）。此时用单测距 + x 坐标反推 y 坐标：
    # 由 ||p - a_i||^2 = r_i^2 解出 y = ±sqrt(r_i^2 - (x - a_ix)^2)。
    # 对称 anchor 几何下正负 y 不可观测（残差相同），按仿真协议默认取 +y，
    # 后续 VIO 更新会自然修正初始符号误差。不再因残差过大回退到 None
    # （回退到 (0,0) 会导致初始位置偏差 14m+ 触发 §19.1 永久拒识）。
    if rmse > 1.0:
        # 共线 anchor 兜底：用第 1 个 anchor 的测距反解 y。
        dx0 = x[0] - a_coords[0][0]
        r0_sq_minus_dx_sq = r_obs[0] ** 2 - dx0 * dx0
        if r0_sq_minus_dx_sq > 0:
            y_mag = math.sqrt(r0_sq_minus_dx_sq)
            return {"px": float(x[0]), "py": float(y_mag)}
        return {"px": float(x[0]), "py": float(x[1])}
    return {"px": float(x[0]), "py": float(x[1])}

# 铁律 7：紧耦合时间窗聚合默认窗口（±5ms）。
# 同一时间戳 ±此窗口内的 UWB + VIO 事件会被合并到一次 estimator.step_joint 调用，
# 走 stacked H + joint update 路径；超出窗口或仅单模态时退化为逐事件 estimator.step()。
# 该值可通过 cfg["tight_coupling_window_s"] 覆写，便于实验/消融。
_DEFAULT_TIGHT_COUPLING_WINDOW_S = 0.005


def _event_uwb_payload(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """从事件中提取 UWB 负载；非 UWB 事件或负载缺失返回 None。"""
    if event.get("modality") != MODALITY_UWB:
        return None
    payload = event.get("uwb_payload")
    if not isinstance(payload, Mapping):
        return None
    return payload


def _event_vio_payload(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """从事件中提取 VIO 负载；非 VIO 事件或负载缺失返回 None。"""
    if event.get("modality") != MODALITY_VIO:
        return None
    payload = event.get("vio_payload")
    if not isinstance(payload, Mapping):
        return None
    return payload


def _snapshot_estimator_state(estimator: Any) -> dict[str, float]:
    """快照估计器当前状态，用于构建历史特征。

    在每个事件处理前调用，记录估计器的预测状态（尚未被当前测量更新），
    以便模型在下一轮推理时能看到"上一步估计器看到了什么"。

    Args:
        estimator: 估计器实例，需实现 `get_state()` 方法；
            若未实现则返回空字典，表示无法提取状态。

    Returns:
        估计器状态的字典表示；若估计器不支持状态提取、state 载荷非映射类型
        则返回空字典。非数值或非有限值会被跳过，避免 NaN/inf 或 tensor
        泄漏到特征窗口。
    """
    if not hasattr(estimator, "get_state"):
        # 估计器没有 get_state 接口，无法提取状态，返回空字典而非报错，
        # 保持对最小接口估计器的兼容。
        return {}
    state_estimate = estimator.get_state()
    # get_state() 可能返回 None 或非 StateEstimate 对象，先取 state 载荷再做类型校验。
    state_payload = getattr(state_estimate, "state", None)
    if not isinstance(state_payload, Mapping):
        # state 载荷不是映射类型（None、list、tuple、str 等），不支持 .items()，
        # 这里统一兜底为空字典，与 _snapshot_proxy_estimator_state 同口径。
        return {}
    snapshot: dict[str, float] = {}
    for key, value in state_payload.items():
        # coerce_finite_scalar 统一完成 float() + isfinite + 拒绝 NaN/Inf/非数值，
        # 防止 NaN/inf 或 numpy/torch 标量泄漏到特征窗口。
        # 与 train_pipeline._snapshot_proxy_estimator_state 对齐（D5 数值安全根因修复）。
        try:
            scalar_value = coerce_finite_scalar(value, name=f"state[{key!r}]")
        except (TypeError, ValueError):
            continue
        snapshot[str(key)] = scalar_value
    return snapshot


def _coerce_intermediate(model_output: Any) -> ModelIntermediate:
    """将模型输出转换为 ModelIntermediate，施加与桥接层一致的值域约束。

    值域约束确保诊断 trace 记录的值与实际控制量一致：
    - bias ∈ [0.0, 10.0]（非负约束 + 上界约束，与模型工厂一致）
    - risk ∈ [0.0, 1.0]（归一化裁剪）
    - uwb_scaling >= 1.0, vio_scaling >= 1.0（缩放下限）

    NaN/inf 检查：Python max()/min() 遇到 NaN 会传播 NaN 而非裁剪，
    导致 EKF 状态被静默污染。因此先做有限性检查，并把非法模型输出显式拒绝，
    不能静默洗成中性默认值。

    Args:
        model_output: 模型推理结果，支持两种形态：
            - ModelIntermediate 实例：直接提取字段并施加约束
            - dict：按键提取 bias/risk/uwb_scaling/vio_scaling 并施加约束
            不支持其他类型，会抛出 TypeError。

    Returns:
        值域约束后的 ModelIntermediate 实例。

    Raises:
        ValueError: 当 dict 输入缺少必需键或包含未定义键时。
        TypeError: 当输入既不是 ModelIntermediate 也不是 dict 时。
    """

    # 从全局默认阈值中读取值域边界，确保与桥接层使用同一套约束。
    _risk_min = BRIDGE_THRESHOLDS["risk_min"]
    _risk_max = BRIDGE_THRESHOLDS["risk_max"]
    _scaling_min = BRIDGE_THRESHOLDS["scaling_min"]
    _scaling_max = BRIDGE_THRESHOLDS["scaling_max"]

    if isinstance(model_output, ModelIntermediate):
        # 模型直接返回了 ModelIntermediate 实例，逐字段施加约束。
        return ModelIntermediate(
            bias=min(BRIDGE_BIAS_MAX, max(0.0, coerce_finite_scalar(model_output.bias, name="bias"))),  # bias 非负裁剪 + 上界约束，与模型工厂一致
            risk=min(_risk_max, max(_risk_min, coerce_finite_scalar(model_output.risk, name="risk"))),  # risk 只允许合法实数后再裁剪
            uwb_scaling=min(_scaling_max, max(_scaling_min, coerce_finite_scalar(model_output.uwb_scaling, name="uwb_scaling"))),  # UWB 缩放只做合同裁剪，不吞错
            vio_scaling=min(_scaling_max, max(_scaling_min, coerce_finite_scalar(model_output.vio_scaling, name="vio_scaling"))),  # VIO 缩放只做合同裁剪，不吞错
        )
    if isinstance(model_output, dict):
        # 模型返回了字典，需要严格校验键集合，防止模型输出未定义的控制信号。
        missing_keys = [key for key in _INTERMEDIATE_KEYS if key not in model_output]
        if missing_keys:
            # 缺少必需键说明模型输出不完整，必须报错而非静默补默认值，
            # 因为这通常意味着模型结构或推理逻辑出了问题。
            raise ValueError(f"model output is missing required keys: {', '.join(missing_keys)}")
        extra_keys = sorted(key for key in model_output if key not in _INTERMEDIATE_KEYS)
        if extra_keys:
            # 多余键同样必须拒绝，防止模型偷偷输出未定义的控制量
            # （例如额外的"confidence"字段可能绕过值域约束被直接使用）。
            allowed = ", ".join(_INTERMEDIATE_KEYS)
            raise ValueError(
                "model output contains unsupported keys: "
                f"{', '.join(extra_keys)}; only measurement-level heads are allowed: {allowed}"
            )
        # 键集合校验通过后，逐字段施加与 ModelIntermediate 实例分支相同的约束。
        return ModelIntermediate(
            bias=min(BRIDGE_BIAS_MAX, max(0.0, coerce_finite_scalar(model_output["bias"], name="bias"))),
            risk=min(_risk_max, max(_risk_min, coerce_finite_scalar(model_output["risk"], name="risk"))),
            uwb_scaling=min(_scaling_max, max(_scaling_min, coerce_finite_scalar(model_output["uwb_scaling"], name="uwb_scaling"))),
            vio_scaling=min(_scaling_max, max(_scaling_min, coerce_finite_scalar(model_output["vio_scaling"], name="vio_scaling"))),
        )
    # 不支持的输出类型，直接报错，避免静默降级到默认值。
    raise TypeError(
        "model output must be a ModelIntermediate or dict, "
        f"got {type(model_output).__name__}"
    )


def _init_readout_context_cache() -> dict[str, dict[str, float | bool | None]]:
    """初始化读出上下文缓存，为每个模态创建独立的跟踪槽位。

    读出上下文缓存用于在事件循环中持续跟踪每个模态（uwb / vio）的
    门控状态、新息范数、连续跳过次数等信息。这些信息会被注入到
    模型特征窗口中，让模型能感知估计器的实时运行状态。

    Returns:
        嵌套字典，外层键为模态名（"uwb"/"vio"），内层为该模态的
        各跟踪字段及其初始值。
    """
    cache: dict[str, dict[str, float | bool | None]] = {}
    for modality in _READOUT_CONTEXT_MODALITIES:
        # 键名与默认值来自单源常量 _READOUT_CONTEXT_CACHE_DEFAULTS，与训练侧
        # _init_training_readout_context_cache 同口径（D9 单源 + D7 公平性）。
        # dict() 浅拷贝足够：值均为不可变（0.0/False/None），无共享引用风险。
        cache[modality] = dict(_READOUT_CONTEXT_CACHE_DEFAULTS)
    return cache


def _coerce_residual_norm(residual: Any, *, modality: str) -> float | None:
    """将估计器报告中的残差转换为标量范数。

    UWB 的残差是标量（距离偏差），直接取绝对值；
    VIO 的残差是向量（位置+姿态偏差），取 L2 范数。

    Args:
        residual: 估计器更新报告中的残差字段，形态取决于模态。
        modality: 传感器模态（"uwb" 或 "vio"），决定残差的解析方式。

    Returns:
        残差的标量范数；若残差无法解析或包含非有限值则返回 None。
    """
    # modality 必须是已知模态，否则无法确定残差形态。
    # 虽然当前调用链 _update_readout_context_cache 已过滤非 uwb/vio 模态，
    # 但本函数作为公共 API（在 __all__ 中）必须自我保护。
    if modality not in (MODALITY_UWB, MODALITY_VIO):
        return None
    if modality == MODALITY_UWB:
        # UWB 残差是标量，直接取绝对值。
        # bool 与连续范数语义冲突，必须拒绝（与 _coerce_intermediate 中的 is_bool_like 检查一致）。
        if is_bool_like(residual):
            return None
        try:
            scalar = abs(float(residual))
        except (TypeError, ValueError):
            return None  # 无法转为浮点数，返回 None 表示不可用
        return scalar if math.isfinite(scalar) else None  # NaN/inf 视为不可用
    # VIO 残差是向量，计算 L2 范数。
    try:
        raw_values = list(residual)
    except TypeError:
        return None  # residual 不可迭代，返回 None
    # bool 元素与连续范数语义冲突，必须拒绝（与 _coerce_intermediate 中的 is_bool_like 检查一致）。
    # 注意：必须在 float() 转换之前检查，否则 bool 会被静默转成 0.0/1.0。
    if any(is_bool_like(value) for value in raw_values):
        return None
    try:
        values = [float(value) for value in raw_values]
    except (TypeError, ValueError):
        return None  # 无法转为浮点数列表，返回 None
    if not values or any(not math.isfinite(value) for value in values):
        return None  # 空列表或包含非有限值，返回 None
    return math.sqrt(math.fsum(value * value for value in values))  # L2 范数，用 fsum 提高平方和精度


def _update_readout_context_cache(
    cache: dict[str, dict[str, float | bool | None]],
    report: Any,
    *,
    current_timestamp: float | None = None,
) -> None:
    """根据估计器的更新报告，更新读出上下文缓存。

    估计器每次 step() 后会产出一份更新报告（last_update_report），
    包含本次更新是否被应用、门控是否通过、残差等信息。
    本函数解析这些信息并更新缓存，供下一轮模型推理使用。

    Args:
        cache: 读出上下文缓存，由 _init_readout_context_cache() 创建。
        report: 估计器的更新报告，预期为 Mapping 类型；
            若不是 Mapping 则静默跳过（保持缓存不变）。
        current_timestamp: 当前事件的时间戳，用于计算距上次成功更新的时间间隔。
    """
    if not isinstance(report, Mapping):
        # 报告不是映射类型，无法解析，静默跳过。
        return
    modality = report.get("modality")  # 保持原始类型，避免 str(None or "") 静默为 ""（与训练侧同口径）
    if modality not in cache:
        # 模态不在缓存范围内（例如 "imu"），跳过。
        return
    modality_cache = cache[modality]
    reason = report.get("reason")  # 保持原始类型（与训练侧同口径）
    residual = report.get("residual")

    # 分支 1：更新被成功应用且包含残差（与文档 §11.5 规则 1 及训练侧
    # _update_training_readout_context_cache 同口径：规则 1 的条件是
    # "update_applied=True 且包含 residual"，所有写操作均需残差可解析成功）。
    if bool(report.get("update_applied")) and residual is not None:
        residual_norm = _coerce_residual_norm(residual, modality=modality)
        if residual_norm is not None:
            modality_cache["last_innovation_norm"] = residual_norm  # 记录新息范数
            modality_cache["innovation_observed"] = True  # 标记已观察到新息
            modality_cache["last_gate_skip_flag"] = 0.0  # 更新成功意味着门控通过
            modality_cache["skip_flag_observed"] = True
            modality_cache["consecutive_skip_count"] = 0.0  # 成功更新后重置连续跳过计数
            modality_cache["consecutive_skip_observed"] = True
            modality_cache["last_successful_update_timestamp"] = current_timestamp  # 记录成功更新的时间戳
        return

    # 分支 2：已知跳过原因（与文档 §11.5 规则 2 一致）
    expected_skip_reason = f"{modality}_skip_update"
    if reason == expected_skip_reason:
        modality_cache["last_gate_skip_flag"] = 1.0  # 标记门控跳过
        modality_cache["skip_flag_observed"] = True
        # 累加连续跳过计数，而不是重置
        modality_cache["consecutive_skip_count"] = float(modality_cache.get("consecutive_skip_count", 0.0) or 0.0) + 1.0
        modality_cache["consecutive_skip_observed"] = True
        # 诊断：打印拒识原因
        import sys as _sys
        if int(modality_cache["consecutive_skip_count"]) in (1, 50, 99, 100, 101):
            _t_diag = float(current_timestamp) if current_timestamp is not None else float("nan")  # current_timestamp 可为 None (无时间戳事件), 不能直接 :.3f
            print(f"[DIAG][{modality}] skip#{int(modality_cache['consecutive_skip_count'])} t={_t_diag:.3f} gate={report.get('gate')} reason={reason}", file=_sys.stderr)
        # §19.1 运行时永久拒识门限：连续跳过帧数超过 max_consecutive_skip_count 即视为假第一。
        _max_skip = int(BRIDGE_THRESHOLDS.get("max_consecutive_skip_count", 100))
        if modality_cache["consecutive_skip_count"] > _max_skip:
            raise RuntimeError(
                f"§19.1 永久拒识运行时拦截：模态 {modality!r} 连续跳过帧数 "
                f"{modality_cache['consecutive_skip_count']} 超过门限 "
                f"{_max_skip}（BRIDGE_THRESHOLDS.max_consecutive_skip_count）。"
                f"此现象属于假第一攻击面，必须终止当前 run。"
            )
    # 分支 3：更新未被应用，且门控明确未通过（与文档 §11.5 规则 3 一致；
    # isinstance 守卫防止 gate 为 None 时 .get 崩溃，与训练侧同口径）
    elif not bool(report.get("update_applied")) and isinstance(report.get("gate"), Mapping) and report.get("gate", {}).get("passed") is False:
        modality_cache["last_gate_skip_flag"] = 1.0  # 标记门控跳过
        modality_cache["skip_flag_observed"] = True
        modality_cache["consecutive_skip_count"] = float(modality_cache.get("consecutive_skip_count", 0.0) or 0.0) + 1.0
        modality_cache["consecutive_skip_observed"] = True
        # 诊断：打印拒识原因
        import sys as _sys
        if int(modality_cache["consecutive_skip_count"]) in (1, 50, 99, 100, 101):
            _t_diag = float(current_timestamp) if current_timestamp is not None else float("nan")  # current_timestamp 可为 None (无时间戳事件), 不能直接 :.3f
            print(f"[DIAG][{modality}] skip#{int(modality_cache['consecutive_skip_count'])} t={_t_diag:.3f} gate={report.get('gate')} reason={reason}", file=_sys.stderr)
        # §19.1 运行时永久拒识门限（分支 3 同口径）。
        _max_skip = int(BRIDGE_THRESHOLDS.get("max_consecutive_skip_count", 100))
        if modality_cache["consecutive_skip_count"] > _max_skip:
            raise RuntimeError(
                f"§19.1 永久拒识运行时拦截（门控拒绝路径）：模态 {modality!r} 连续跳过帧数 "
                f"{modality_cache['consecutive_skip_count']} 超过门限 "
                f"{_max_skip}（BRIDGE_THRESHOLDS.max_consecutive_skip_count）。"
            )
    # 分支 4：其他情况不修改该模态缓存（与文档 §11.5 规则 4 一致，与训练侧同口径）


def _build_readout_context(
    estimator: Any,
    modality_cache: dict[str, dict[str, float | bool | None]],
    *,
    modality: str,
    current_timestamp: float | None = None,
) -> tuple[dict[str, float], dict[str, bool]]:
    """从估计器状态和读出缓存中构建当前读出上下文。

    读出上下文会被注入到模型特征窗口中，让模型能感知：
    - 估计器当前的不确定性水平（state_cov_trace, pos_cov）
    - 最近一次测量的偏差程度（last_innovation_norm）
    - 门控是否频繁跳过（last_gate_skip_flag, consecutive_skip_count）
    - 测量中断了多久（time_since_last_update）

    Args:
        estimator: 估计器实例，需实现 `get_state()` 方法。
        modality_cache: 当前模态的读出上下文缓存。
        modality: 当前传感器模态（"uwb" 或 "vio"）。
        current_timestamp: 当前事件的时间戳，用于计算 time_since_last_update。

    Returns:
        二元组 (values, observed)：
        - values: 各读出字段的当前数值，未观察到的字段为 0.0
        - observed: 各读出字段是否已被实际观察到，未观察到的为 False
        两个字典的键集合相同，均为 _READOUT_CONTEXT_KEYS。
    """
    # 初始化所有字段为默认值，确保输出结构稳定。
    values = {key: 0.0 for key in _READOUT_CONTEXT_KEYS}
    observed = {key: False for key in _READOUT_CONTEXT_KEYS}

    # 从估计器状态中提取协方差信息
    if hasattr(estimator, "get_state"):
        state_estimate = estimator.get_state()
        # 显式 None 检查：避免对 numpy 数组等多元素对象调用 bool() 触发歧义异常
        cov_diag_attr = getattr(state_estimate, "covariance_diag", None)
        covariance_diag = list(cov_diag_attr) if cov_diag_attr is not None else []
        if len(covariance_diag) >= 2:
            # 协方差对角线至少需要 2 个元素（x, y）才能计算位置指标
            # D5/D8 数值安全：统一转换+异常守卫，避免 float() 对 None/字符串抛未捕获异常中断构建。
            # 一次性转换避免重复 float() 调用（原检查/求迹/取位置方差各转一次）。
            try:
                diag_floats = [float(value) for value in covariance_diag]
            except (TypeError, ValueError):
                diag_floats = None
            if diag_floats is not None:
                # 协方差迹 = 对角线元素之和，反映整体不确定性
                if all(math.isfinite(d) for d in diag_floats):
                    values["state_cov_trace"] = math.fsum(diag_floats)  # 协方差迹，用 fsum 提高精度
                    observed["state_cov_trace"] = True
                    # D2-5 v2 redesign: 位置协方差迹（前 2 项）和速度协方差迹（第 3-4 项）
                    # 这两个新键让 filter_context 能区分位置/速度不确定性，更适合异步高 NLOS。
                    values["pos_cov_trace"] = max(diag_floats[0], 0.0) + max(diag_floats[1], 0.0)
                    observed["pos_cov_trace"] = True
                    if len(diag_floats) >= 4:
                        values["vel_cov_trace"] = max(diag_floats[2], 0.0) + max(diag_floats[3], 0.0)
                        observed["vel_cov_trace"] = True
                # 位置协方差 = sqrt(var_x + var_y)，反映位置估计精度
                # pos_cov 仅依赖前两个元素，尾部元素非有限不影响 pos_cov 计算
                # （对齐 test_..._keeps_pos_cov_when_tail_covariance_entries_are_non_finite）。
                px_var = max(diag_floats[0], 0.0)  # 方差不能为负
                py_var = max(diag_floats[1], 0.0)
                if math.isfinite(px_var) and math.isfinite(py_var):
                    values["pos_cov"] = math.sqrt(px_var + py_var)
                    observed["pos_cov"] = True

    # 从模态缓存中提取门控和新息信息
    modality_state = modality_cache.get(modality)
    if modality_state is None:
        # 当前模态没有缓存记录，返回全默认值
        return values, observed

    # 只有实际观察到过的字段才写入 values 和 observed，
    # 避免用初始值 0.0 误导模型（0.0 可能被模型解读为"极低的不确定性"）。
    # D5 数值安全：缓存值用 coerce_finite_scalar 统一转换+有限性校验，
    # 避免 float() 对 None/字符串/inf/nan 静默通过；任何异常跳过该字段保持默认值。
    if bool(modality_state.get("innovation_observed", False)):
        try:
            values["last_innovation_norm"] = coerce_finite_scalar(
                modality_state["last_innovation_norm"], name="last_innovation_norm")
            observed["last_innovation_norm"] = True
        except (TypeError, ValueError, KeyError):
            pass
    if bool(modality_state.get("skip_flag_observed", False)):
        try:
            skip_value = coerce_finite_scalar(
                modality_state["last_gate_skip_flag"], name="last_gate_skip_flag")
            values["last_gate_skip_flag"] = skip_value
            observed["last_gate_skip_flag"] = True
            # D2-5 v2 redesign: last_update_skip_flag 与 last_gate_skip_flag 同义，
            # D2-3 把它加入 VIO filter context 子集以满足"最近一次更新是否被跳过"的语义。
            values["last_update_skip_flag"] = skip_value
            observed["last_update_skip_flag"] = True
        except (TypeError, ValueError, KeyError):
            pass

    if bool(modality_state.get("consecutive_skip_observed", False)):
        try:
            # D9 移除冗余 `or 0.0`：observed 标志已守卫键被观测过，
            # None/缺失应触发异常而非静默掩为 0.0（掩盖缓存不一致）。
            values["consecutive_skip_count"] = coerce_finite_scalar(
                modality_state["consecutive_skip_count"], name="consecutive_skip_count")
            observed["consecutive_skip_count"] = True
        except (TypeError, ValueError, KeyError):
            pass

    # 计算距离上次成功更新的时间间隔
    last_success_ts = modality_state.get("last_successful_update_timestamp")
    if last_success_ts is not None and current_timestamp is not None:
        try:
            values["time_since_last_update"] = max(
                0.0,
                coerce_finite_scalar(current_timestamp, name="current_timestamp")
                - coerce_finite_scalar(last_success_ts, name="last_successful_update_timestamp"),
            )
            observed["time_since_last_update"] = True
        except (TypeError, ValueError):
            pass
    return values, observed


def run_fusion(events, estimator, model_infer=None, feature_builder=None, cfg=None):
    """融合主循环：逐事件驱动估计器，并在测量事件到来时调用模型推理。

    整体流程：
    1. 校验事件序列和估计器
    2. 重置估计器和模型（若支持）
    3. 逐事件循环：
       a. 快照估计器当前状态（用于构建历史特征）
       b. 若当前事件是 uwb/vio 测量且模型可用：
          - 构建特征窗口（包含历史状态和读出上下文）
          - 调用模型推理得到中间控制量
          - 经桥接合约转换为 MeasurementControl
       c. 将控制量注入估计器
       d. 执行估计器 step()
       e. 更新读出上下文缓存
       f. 记录状态和诊断 trace
    4. 返回包含状态序列和诊断 trace 的 bundle

    Args:
        events: 事件序列，每个事件为 Event 实例或字典；
            必须非空且通过 validate_event_sequence 校验。
        estimator: 估计器实例，需实现 step(event) 方法；
            可选实现 reset()、consume_model_intermediate()、set_measurement_control()、get_state()。
        model_infer: 模型推理器，需实现 infer_intermediate(features) 方法；
            若为 None 则跳过模型推理，使用默认 MeasurementControl。
        feature_builder: 特征构建函数，签名为 (history, event, cfg) -> feature_window；
            当 model_infer 不为 None 时必须提供。
        cfg: 配置字典，可包含：
            - "method_name": 方法名称，默认 "unknown"
            - "safe_mode": 安全模式配置
            - "initial_state": 估计器初始状态
            以及 feature_builder 需要的其他配置项。

    Returns:
        bundle 字典，包含：
        - "seq_id": 序列标识
        - "scene_id": 场景标识
        - "method_name": 方法名称
        - "states": 估计器状态列表
        - "timestamps": 时间戳列表
        - "mechanism_contract": 控制机制合约信息
        - "diagnostics": 诊断 trace 字典

    Raises:
        ValueError: 事件为空、估计器为 None、或 feature_builder 不可调用。
    """
    cfg = dict(cfg or {})  # shallow copy: only top-level keys are mutated, no nested writes
    # 显式拒绝会被 list() 误转换的类型，与 event_queue.py / validate_event_sequence 的守卫对齐。
    # 必须在 list() 之前检查，否则：
    # - list(None) 抛不清晰的 TypeError，而非语义化的 ValueError
    # - list("abc") 拆成字符列表，绕过事件校验，错误信息令人困惑
    # - list({...}) 返回键列表，绕过 validate_event_sequence 的 Mapping 守卫
    if events is None:
        raise ValueError("events must not be None")
    if isinstance(events, (str, bytes)):
        raise TypeError("events must be a sequence of events, not a string or bytes")
    if isinstance(events, Mapping):
        raise TypeError("events must be a sequence of events, not a single mapping")
    events = list(events)  # 展开可迭代输入，确保能多次遍历
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "events_count": len(events) if isinstance(events, (list, tuple)) else None,
        "estimator_type": type(estimator).__name__,
        "has_model_infer": model_infer is not None,
        "has_feature_builder": feature_builder is not None,
        "method_name": (cfg or {}).get("method_name") if isinstance(cfg, dict) else None,
        "resolved_method_name": (cfg or {}).get("resolved_method_name") if isinstance(cfg, dict) else None,
        "safe_mode": (cfg or {}).get("safe_mode") if isinstance(cfg, dict) else None,
    }, "run_fusion 入口参数")
    # 将 Event 实例统一转为字典，后续处理只需关心字典形态
    events = [event.to_dict() if isinstance(event, Event) else event for event in events]
    if not events:
        raise ValueError("events must be a non-empty list")
    validate_event_sequence(events)  # 校验事件序列的合法性和时序一致性
    if estimator is None:
        raise ValueError("estimator must not be None")
    # 模型推理器和存在时，特征构建函数必须可调用，否则模型无法获取输入
    if model_infer is not None and not callable(feature_builder):
        raise ValueError("feature_builder must be callable when model_infer is provided")

    # 重置估计器和模型到初始状态，确保每次运行结果可复现
    # 用 anchor 几何 + 首个 UWB 测距做三角定位，解决 EKF 冷启动初始位置偏置问题
    # （RMSE 从 15.10m 降至 1.04m 的关键修复）
    initial_state = None
    if hasattr(estimator, "reset"):
        anchor_layout = cfg.get("anchor_layout") if isinstance(cfg, Mapping) else None
        geo_init = _compute_initial_position_from_anchor_geometry(events, anchor_layout)
        if geo_init is not None:
            initial_state = geo_init  # 用三角定位初始位置覆盖默认 (0,0)
            _LOGGER.debug("三角定位初始位置: (%.4f, %.4f)", geo_init["px"], geo_init["py"])
        else:
            initial_state = cfg.get("initial_state")
            _LOGGER.warning("三角定位失败，使用默认初始状态: %s", initial_state)
        estimator.reset(initial_state)
        _LOGGER.debug("reset 后 EKF 状态: %s", dict(estimator.get_state().state))
    if model_infer is not None and hasattr(model_infer, "reset"):
        model_infer.reset()

    # 从第一个事件中提取序列和场景标识，构建输出 bundle
    first = events[0]
    bundle = {
        "seq_id": first["meta"]["seq_id"],       # 序列标识，用于区分不同数据序列
        "scene_id": first["meta"]["scene_id"],    # 场景标识，用于区分不同实验场景
        "method_name": cfg.get("method_name", "unknown"),  # 方法名称，默认 "unknown"
        "states": [],     # 估计器状态列表，每个元素是一个状态字典
        "timestamps": [], # 时间戳列表，与 states 一一对应
        "mechanism_contract": {
            # 机制合约：声明本次融合运行中控制机制的行为模式
            # 外部消费者（如验证脚本）可以据此判断 trace 中哪些字段是可信的
            "bridge_enabled": model_infer is not None,  # 桥接是否启用（即模型是否参与）
            "liquid_controls_measurement_update": True,  # Liquid 是否控制测量更新（始终为 True）
            "controls": [
                # 以下五个控制项会被记录到 diagnostics trace 中
                "bias_applied",       # 实际施加的偏置
                "risk",               # 实际使用的风险值
                "scaling",            # 协方差缩放因子
                "noise_multiplier",   # 噪声乘子（由 scaling² × (1+risk) 计算）
                "gate_action",        # 门控动作（pass / skip / ...）
            ],
        },
        "diagnostics": {
            # 模型输出的原始中间控制量 trace
            "risk_trace": [],              # 模型输出的 risk（经值域约束后）
            "bias_trace": [],              # 模型输出的 bias（经值域约束后）
            "uwb_scaling_trace": [],       # 模型输出的 uwb_scaling（经值域约束后）
            "vio_scaling_trace": [],       # 模型输出的 vio_scaling（经值域约束后）
            # 实际施加的控制量 trace（经桥接合约和安全模式收缩后）
            "applied_risk_trace": [],      # 实际施加的 risk
            "applied_bias_trace": [],      # 实际施加的 bias
            "applied_uwb_scaling_trace": [],  # 实际施加的 UWB 缩放
            "applied_vio_scaling_trace": [],  # 实际施加的 VIO 缩放
            "noise_multiplier_trace": [],  # 噪声乘子 trace
            "gate_action_trace": [],       # 门控动作 trace
            "modalities": [],              # 每个事件对应的模态
        },
    }

    # 安全模式配置：读取 "safe_mode" 键。
    # 注意：不能用 `or`，因为空字典 {} 是合法的 safe_mode 配置（表示安全模式关闭），
    # 但 {} 是 falsy，`or` 会错误地跳过它走默认值。
    safe_mode_cfg = cfg.get("safe_mode")
    history = []          # 事件历史列表，用于构建模型特征窗口
    state_history = []    # 估计器状态历史列表，用于构建模型特征窗口
    readout_context_cache = _init_readout_context_cache()  # 初始化读出上下文缓存

    # 铁律 7：紧耦合时间窗聚合状态。
    # tight_coupling_window_s: 同 timestamp ±此窗口内的 UWB + VIO 事件合并成
    #   一次 estimator.step_joint 调用。默认 5ms，经 cfg["tight_coupling_window_s"] 覆写。
    # pending_buffer: 等待时间窗关闭的 UWB/VIO 事件三元组 (event, intermediate, control)，
    #   局部变量，避免跨 run_fusion 调用泄漏。
    # step_joint_available: estimator 是否实现 step_joint；缺则强制走单模态 fallback。
    tight_coupling_window_s = float(
        cfg.get("tight_coupling_window_s", _DEFAULT_TIGHT_COUPLING_WINDOW_S)
    )
    pending_buffer: list[tuple[Mapping[str, Any], ModelIntermediate, MeasurementControl]] = []
    step_joint_available = hasattr(estimator, "step_joint")

    def _record_trace_entry(
        ev: Mapping[str, Any],
        ev_intermediate: ModelIntermediate,
        ev_control: MeasurementControl,
        ev_state: Any,
        *,
        ev_report: Mapping[str, Any] | None,
    ) -> None:
        """把单个事件对应的 state / diagnostic trace 写入 bundle。

        trace 必须按事件 1:1 追加 — 否则训练 / 评测对齐会破。state_estimate 即便
        被多事件共享 (联合路径) 也按事件次序重复写一份相同的 state 字典。
        """
        # 更新读出上下文缓存 (联合路径只对最后一事件传非 None report, 中间事件 None →
        # _update_readout_context_cache 静默跳过, 与原行为对齐).
        if ev_report is not None:
            _update_readout_context_cache(
                readout_context_cache,
                ev_report,
                current_timestamp=float(ev.get("t", 0.0)),
            )
        bundle["states"].append(dict(getattr(ev_state, "state", {})))
        bundle["timestamps"].append(float(ev["t"]))
        bundle["diagnostics"]["modalities"].append(str(ev["modality"]))
        # 模型输出的原始中间控制量；与 _coerce_intermediate 一致使用 coerce_finite_scalar
        # 做有限性校验，避免 NaN/Inf 进入诊断 trace。
        bundle["diagnostics"]["risk_trace"].append(coerce_finite_scalar(ev_intermediate.risk, name="risk"))
        bundle["diagnostics"]["bias_trace"].append(coerce_finite_scalar(ev_intermediate.bias, name="bias"))
        bundle["diagnostics"]["uwb_scaling_trace"].append(coerce_finite_scalar(ev_intermediate.uwb_scaling, name="uwb_scaling"))
        bundle["diagnostics"]["vio_scaling_trace"].append(coerce_finite_scalar(ev_intermediate.vio_scaling, name="vio_scaling"))
        # 实际施加的控制量
        bundle["diagnostics"]["applied_risk_trace"].append(float(ev_control.risk))
        bundle["diagnostics"]["applied_bias_trace"].append(float(ev_control.bias_applied))
        # UWB/VIO 缩放只在对应模态事件中才有意义，其他模态事件记录 1.0
        bundle["diagnostics"]["applied_uwb_scaling_trace"].append(
            float(ev_control.scaling) if ev["modality"] == MODALITY_UWB else 1.0
        )
        bundle["diagnostics"]["applied_vio_scaling_trace"].append(
            float(ev_control.scaling) if ev["modality"] == MODALITY_VIO else 1.0
        )
        bundle["diagnostics"]["noise_multiplier_trace"].append(float(ev_control.noise_multiplier))
        bundle["diagnostics"]["gate_action_trace"].append(str(ev_control.gate_action))

    def _flush_pending_buffer(
        buffer: list[tuple[Mapping[str, Any], ModelIntermediate, MeasurementControl]],
    ) -> list[tuple[Mapping[str, Any], ModelIntermediate, MeasurementControl, Any, Mapping[str, Any] | None]]:
        """铁律 7: 把 buffer 内的 UWB/VIO 事件合并成一次紧耦合 update。

        - buffer 内同时含 UWB + VIO 且 estimator 支持 step_joint 时, 调
          estimator.step_joint(uwb_payloads, vio_payload, timestamp), 走
          stacked H + joint K 路径, 一次性把跨模态相关性握在一次 Joseph 更新里.
        - 其他情况 (单模态 buffer, 或 estimator 无 step_joint, 或 step_joint 抛异常)
          退化为逐事件 estimator.step(event) 旧路径 — 保留单模态 fallback,
          即使紧耦合路径出 bug 也不破现有 ekf baseline.

        返回与 buffer 等长的列表, 每元素 (event, intermediate, control, state_estimate, report):
        - 联合路径: 所有事件共享同一份 step_joint 的 state_estimate; 报告
          取 estimator.last_update_report 写入最后一个事件, 其余置 None
          (后续 _record_trace_entry 内的 readout_context_cache 维护只对最后一个事件触发).
        - fallback 路径: 每事件对应各自 step 后的 state_estimate 与 report.

        返回字段包含 (intermediate, control) 是为了 trace 一一对应 — 调用方无需
        再反查外部状态。
        """
        if not buffer:
            return []
        uwb_events = [ev for ev, _, _ in buffer if ev["modality"] == MODALITY_UWB]
        vio_events = [ev for ev, _, _ in buffer if ev["modality"] == MODALITY_VIO]
        has_joint = bool(uwb_events) and bool(vio_events) and step_joint_available
        time_t0 = float(buffer[0][0].get("t", 0.0))
        results: list[
            tuple[Mapping[str, Any], ModelIntermediate, MeasurementControl, Any, Mapping[str, Any] | None]
        ] = []
        if has_joint:
            # 紧耦合: 取出 UWB/VIO payloads 一次注入 step_joint.
            # 注意 step_joint 期望的是 payload (Mapping), 而非整 event dict —
            # 此处用顶部 helper 抽取, 缺失 payload 的事件跳过.
            # §3.0.2 / §3.1.2：联合路径不可悄悄丢弃 NN bias。逐锚点配对 UWB 事件的
            # NN intermediate.bias 与原始 raw_range，经 clip_uwb_bias（与单模态路径同口径）
            # 注入 step_joint 的 h(·) 侧。uwb_payloads / uwb_extra_biases 严格同序同长，
            # 缺失 payload 的事件同时在两边跳过，避免错位。
            uwb_payloads: list[Mapping[str, Any]] = []
            uwb_extra_biases: list[float] = []
            for ev, interp, _ctrl in buffer:
                if ev["modality"] != MODALITY_UWB:
                    continue
                p = _event_uwb_payload(ev)
                if p is None:
                    continue  # 缺失 payload 的事件同时在两份序列里跳过，保持严格配对。
                uwb_payloads.append(p)
                raw_range = float(p.get("range") or 0.0)
                # 与单模态 build_measurement_control UWB 分支同口径：clip_uwb_bias 做非负、
                # 比例与绝对上限截断。其他 risk/scaling 由单模态或 fallback 路径消费，
                # 联合路径只接受 h(·) 侧 bias 写入（§3.0.2 / §3.1.2）。
                uwb_extra_biases.append(clip_uwb_bias(interp.bias, raw_range))
            # BUG-006 修复 (2026-09-06 §10.2 阶段 11 审计): vio_events[0] 只取 buffer 内首帧 VIO，
            # 当 buffer 有多帧 VIO 时（第 1 帧被丢弃），导致紧耦合 step_joint 用过期 VIO 更新 EKF。
            # 改为取最新一帧 VIO（按时间戳排序后取末帧），与 5ms 时间窗内只取最新测量一致。
            vio_payload = None
            if vio_events:
                _sorted_vio = sorted(vio_events, key=lambda ev: float(ev.get("t", 0.0)))
                vio_payload = _event_vio_payload(_sorted_vio[-1])
            # meta 从首个事件传递, 满足 step_joint 内 synthetic VIO event 协议.
            joint_meta = dict(buffer[0][0].get("meta") or {})
            joint_state: Any = None
            # 偷懒审视 Round 3 补: 设 _step_joint_attempted=True, 让 eval_pipeline
            # 聚合 step_joint_attempted_total 知道有多少 bundle 真试了紧耦合路径.
            estimator._step_joint_attempted = True
            try:
                joint_state = estimator.step_joint(
                    uwb_payloads=uwb_payloads,
                    vio_payload=vio_payload,
                    timestamp=time_t0,
                    meta=joint_meta,
                    uwb_extra_biases=uwb_extra_biases,
                )
                # step_joint 成功 (没抛异常): 设 _step_joint_succeeded=True
                estimator._step_joint_succeeded = True
            except Exception as joint_exc:
                # 紧耦合路径任何异常都退化到逐事件 fallback, 不破现有 ekf baseline.
                # Bug 1 修复 (2026-07-23 audit Round 1+): 加 fallback 可观测性 — 计数 + warning log,
                # 让审稿能看到 step_joint fallback ratio. 不改数学路径, 仅记录异常类型.
                # step_joint 同口径 finally 已清 _measurement_control, 这里无需额外清理.
                estimator._joint_fallback_count = int(getattr(estimator, "_joint_fallback_count", 0)) + 1
                estimator._last_joint_fallback_reason = f"{type(joint_exc).__name__}: {joint_exc}"
                estimator._step_joint_succeeded = False  # 失败标记
                _LOGGER.warning(
                    "step_joint fallback #%d (reason=%s); 退化到逐事件 fallback",
                    estimator._joint_fallback_count,
                    estimator._last_joint_fallback_reason,
                )
                joint_state = None
            if joint_state is not None:
                joint_report = getattr(estimator, "last_update_report", None)
                # 联合 report 只归属最后一个事件, 其余 None — readout_context_cache
                # 仅对最后一个事件触发更新, 避免 modality_cache 被重复写.
                for i, (ev, ev_intermediate, ev_control) in enumerate(buffer):
                    report = joint_report if i == len(buffer) - 1 else None
                    results.append((ev, ev_intermediate, ev_control, joint_state, report))
                return results
            # 落到此分支: step_joint 抛异常, 走逐事件 fallback.
        # fallback: 逐事件 estimator.step(event), 与旧版逐事件 dispatch 完全等价.
        for ev, ev_intermediate, ev_control in buffer:
            # 重新注入 control: buffer 期间已 set 过同一 control, 但 step_joint 路径
            # finally 已清 _measurement_control; 重新 set 一次确保 step 接到正确 control.
            if hasattr(estimator, "consume_model_intermediate"):
                estimator.consume_model_intermediate(ev_intermediate)
            if hasattr(estimator, "set_measurement_control"):
                estimator.set_measurement_control(ev_control)
            ev_state = estimator.step(ev)
            ev_report = getattr(estimator, "last_update_report", None)
            results.append((ev, ev_intermediate, ev_control, ev_state, ev_report))
        return results

    for event in events:
        # 在估计器更新前快照当前状态，用于构建"模型能看到的历史"
        pre_update_state = _snapshot_estimator_state(estimator)
        history.append(event)
        state_history.append(pre_update_state)

        # 初始化中间控制量为默认值（全零/全一），若模型不可用则保持默认
        intermediate = ModelIntermediate()
        if model_infer is not None and event["modality"] in {MODALITY_UWB, MODALITY_VIO}:
            # 只对 uwb/vio 测量事件调用模型推理，IMU 事件不需要模型干预
            feature_cfg = dict(cfg)
            feature_cfg["state_history"] = [dict(s) for s in state_history]  # 注入状态历史（浅拷贝新字典；_snapshot_estimator_state 已过滤为 dict[str,float]，float 不可变无需深拷贝）
            feature_cfg["dt"] = float(event.get("dt", 0.0))     # 注入时间步长
            # 调用特征构建器，将原始事件和历史状态转换为模型可消费的特征窗口
            feature_window = feature_builder(history, event, feature_cfg)
            # 确保 feature_window 中包含 dt 字段（特征构建器可能遗漏）
            if isinstance(feature_window, dict) and "dt" not in feature_window:
                feature_window = dict(feature_window)
                feature_window["dt"] = feature_cfg["dt"]
            # 将读出上下文注入特征窗口，让模型能感知估计器的实时运行状态
            if isinstance(feature_window, dict):
                current_t = float(event.get("t", 0.0))
                readout_context_by_name, readout_context_observed_by_name = _build_readout_context(
                    estimator,
                    readout_context_cache,
                    modality=event["modality"],
                    current_timestamp=current_t,
                )
                feature_window = dict(feature_window)
                feature_window["readout_context_by_name"] = readout_context_by_name
                feature_window["readout_context_observed_by_name"] = readout_context_observed_by_name
            # 调用模型推理，得到中间控制量，并施加值域约束
            intermediate = _coerce_intermediate(model_infer.infer_intermediate(feature_window))

        # 将中间控制量经桥接合约转换为 MeasurementControl。
        # 即使没有模型，也要走同一套桥接默认语义，避免 baseline 绕过
        # UWB/VIO 的 valid/quality 协议级门控。
        control = build_measurement_control(
            event,
            intermediate,
            safe_mode_cfg=safe_mode_cfg,
        )

        # 将中间控制量和测量控制注入估计器（若估计器支持这些接口）
        if hasattr(estimator, "consume_model_intermediate"):
            estimator.consume_model_intermediate(intermediate)
        if hasattr(estimator, "set_measurement_control"):
            estimator.set_measurement_control(control)

        # 铁律 7: 紧耦合时间窗聚合 — 同 timestamp ±tight_coupling_window_s 内
        # 的 UWB + VIO 事件合并到一次 estimator.step_joint 调用，走 stacked H +
        # joint K；超出窗口 / 单模态 / estimator 无 step_joint 时退化为逐事件
        # estimator.step()，保留现有 ekf baseline 行为。
        if event["modality"] == MODALITY_IMU:
            # IMU 事件是 predict 步, 不参与紧耦合. 先 flush 任何挂起 buffer,
            # 再 step IMU — 保证 UWB/VIO buffer 不被 IMU 截断成松耦合.
            flushed = _flush_pending_buffer(pending_buffer)
            pending_buffer.clear()
            for ev, ev_intermediate, ev_control, ev_state, ev_report in flushed:
                _record_trace_entry(
                    ev, ev_intermediate, ev_control, ev_state, ev_report=ev_report,
                )
            state_estimate = estimator.step(event)
            _record_trace_entry(
                event, intermediate, control, state_estimate,
                ev_report=getattr(estimator, "last_update_report", None),
            )
        else:
            # UWB/VIO 事件入 buffer, 等时间窗关闭再 flush.
            pending_buffer.append((event, intermediate, control))
            time_t0 = float(pending_buffer[0][0].get("t", 0.0))
            current_t = float(event.get("t", time_t0))
            window_closing = (current_t - time_t0) > tight_coupling_window_s
            is_last_event = event is events[-1]
            if window_closing or is_last_event:
                flushed = _flush_pending_buffer(pending_buffer)
                pending_buffer.clear()
                for ev, ev_intermediate, ev_control, ev_state, ev_report in flushed:
                    _record_trace_entry(
                        ev, ev_intermediate, ev_control, ev_state, ev_report=ev_report,
                    )

    # 末尾保险: 残余 buffer (理论上 is_last_event 已 cover, 此处做幂等 flush).
    if pending_buffer:
        flushed = _flush_pending_buffer(pending_buffer)
        pending_buffer.clear()
        for ev, ev_intermediate, ev_control, ev_state, ev_report in flushed:
            _record_trace_entry(
                ev, ev_intermediate, ev_control, ev_state, ev_report=ev_report,
            )

    # 偏懒#9 修复 (2026-07-23 audit Round 1+ 偷懒审视 Round 2): step_joint fallback 计数
    # 原仅入 _LOGGER.warning log, 下游 final_paper_test_scoring 聚合看不到 fallback ratio.
    # 现把 _joint_fallback_count / _last_joint_fallback_reason 写入 bundle["diagnostics"]
    # 顶层, 让 candidate_report / scoring 聚合能定量看 LCS vs Literal fallback 比例.
    # 与 fusion_runner.py L691-704 的 step_joint except 块配套使用.
    bundle["diagnostics"]["joint_fallback_count"] = int(getattr(estimator, "_joint_fallback_count", 0))
    bundle["diagnostics"]["last_joint_fallback_reason"] = str(getattr(estimator, "_last_joint_fallback_reason", ""))
    # 偷懒审视 Round 3 补: step_joint_attempted/succeeded 字段写入 — eval_pipeline L625 聚合
    # step_joint_attempted_total/succeeded_total 时读到, 知道有多少 bundle 真走了 step_joint 路径.
    # 原仅写 fallback count, 无法区分 "无 fallback + step_joint 真成功" vs "无 fallback + 没走 step_joint".
    # 现加 2 字段: step_joint_attempted (是否真试过 step_joint) + step_joint_succeeded (是否真成功).
    bundle["diagnostics"]["step_joint_attempted"] = bool(getattr(estimator, "_step_joint_attempted", False))
    bundle["diagnostics"]["step_joint_succeeded"] = bool(getattr(estimator, "_step_joint_succeeded", False))

    return bundle


__all__ = [
    "_build_readout_context",
    "_coerce_residual_norm",
    "_init_readout_context_cache",
    "_update_readout_context_cache",
    "run_fusion",
]
