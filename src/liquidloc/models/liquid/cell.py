"""Liquid Cell 模块：实现时间感知、缺失感知的液体状态更新单元。

【文件职责】
定义 LiquidCell，一个基于连续时间递推的循环单元，支持缺失掩码、
时间间隔感知更新和坏观测风险评估。它是 Liquid 网络的核心计算单元。

【本文件绝对不负责】
不负责网络层串联、输出头映射、训练循环和推理。

【上游依赖】
configs/models/liquid_ekf.yaml、common/constants.py。

【下游调用者】
models/liquid/network.py、tests/models/test_liquid_cell.py。

【输入对象定义】
- input_tensor：当前步特征向量。
- cell_state：上一步隐藏状态。
- dt：当前步时间间隔。
- missing_mask：当前步缺失掩码。

【输出对象定义】
- new_cell_state：更新后的隐藏状态。
- cell_output：经过 LayerNorm 后的输出。

【核心变量定义】
- input_dim：输入特征维度。
- hidden_dim：隐藏状态维度。
- update_scale_floor / update_scale_span：状态更新幅度的下限和跨度。
- bad_observation_floor / bad_observation_span：坏观测风险的下限和跨度。
- backbone_projection：主干投影层。
- ff1_projection / ff2_projection：前馈子层.
- time_A_projection / time_B_projection：闭式时间插值 A∈(0,1] / B>0 (v6 Patch 1).
- reliability_projection：可靠性投影.
- output_norm：输出归一化层。
"""

from __future__ import annotations  # 允许在类型注解里引用尚未定义的类型名。

import math  # 用于有限性检查。
from collections.abc import Mapping, Sequence  # Mapping 用于判断字典式输入，Sequence 用于序列型参数。
from typing import Any  # 用于承接不确定类型的配置和输入对象。

import torch  # LiquidCell 的张量计算和参数管理都依赖 PyTorch。
from torch import nn  # 只需要神经网络层和初始化工具。

from liquidloc.common.constants import (  # 从常量源导入坏观测尺度参数和维度 OOM 上界，避免重复定义。
    ASYNC_GAP_FULL_SCALE_S,
    MAX_MODEL_DIM,
    VIO_HIGH_REPROJ_ERR_THRESHOLD,
    VIO_REPROJ_ERR_FULL_SCALE,
    VIO_TRACKED_FEATURES_FLOOR,
)
from liquidloc.common.validation import is_bool_like, is_integer, is_numeric  # 集中判断 bool / np.bool_、整数和数值类型。
from liquidloc.common.weight_init import _fill_linear  # 等间距线性层初始化 helper 的权威定义来源（已从本文件提取至 common 层，避免与 network.py 重复定义漂移）。


def _require_int(cfg: Mapping[str, Any], name: str, *, default: int | None = None) -> int:
    """从配置映射中读取并校验一个正整数字段。

    参数:
    `cfg` 是配置字典。
    `name` 是要读取的字段名。
    `default` 是字段缺失时的默认值，None 表示必须提供。

    返回值:
    返回校验通过的正整数。

    失败条件:
    字段不是整数或小于 1 时抛出异常。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "name": name,
        "default": default,
        "cfg_type": type(cfg).__name__,
    }, "_require_int 入口参数")
    raw_value = cfg.get(name, default)  # 从配置里读取原始值，缺失时用默认值。
    if raw_value is None:  # 字段缺失且无默认值，必须显式报"缺失"而非"类型不对"。
        raise TypeError(f"{name} is required and must be provided")  # 缺失就报缺失。
    if not is_integer(raw_value):  # 布尔不算整数，必须排除。
        raise TypeError(f"{name} must be an integer, got {type(raw_value).__name__}")  # 类型不对就报错并带上实际类型。
    if raw_value < 1:  # 正整数至少为 1。
        raise ValueError(f"{name} must be positive, got {raw_value}")  # 小于 1 直接拒绝并带值。
    if raw_value > MAX_MODEL_DIM:  # 上限保护，防止 input_dim/hidden_dim 配置过大导致 OOM；上界单源真相 common.constants.MAX_MODEL_DIM，禁止本地重复字面量。
        raise ValueError(f"{name} must be <= {MAX_MODEL_DIM} to avoid OOM, got {raw_value}")  # 超过上限就拒绝。
    return int(raw_value)  # 显式转成 Python int，保证返回值与注解 int 一致（np.integer 也归一为 int）。


def _require_probability_like(cfg: Mapping[str, Any], name: str, *, default: float) -> float:
    """从配置映射中读取并校验一个 [0, 1] 区间的概率值字段。

    参数:
    `cfg` 是配置字典。
    `name` 是要读取的字段名。
    `default` 是字段缺失时的默认值。

    返回值:
    返回校验通过的概率值。

    失败条件:
    字段不是数值或不在 [0, 1] 区间时抛出异常。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "name": name,
        "default": default,
        "cfg_type": type(cfg).__name__,
    }, "_require_probability_like 入口参数")
    raw_value = cfg.get(name, default)  # 从配置里读取原始值，缺失时用默认值。
    if not is_numeric(raw_value):  # 布尔不算数值，必须排除。
        raise TypeError(f"{name} must be numeric, got {type(raw_value).__name__}")  # 类型不对就报错并带实际类型。
    value = float(raw_value)  # 统一转成浮点数。
    if not math.isfinite(value):  # 显式有限性检查，防止 NaN/Inf 静默通过 [0,1] 比较（NaN 比较返回 False 会被误判为越界）。
        raise ValueError(f"{name} must be finite, got {value}")  # 非有限就报错并带值。
    if not 0.0 <= value <= 1.0:  # 概率值必须在 [0, 1] 闭区间内。
        raise ValueError(f"{name} must be within [0, 1], got {value}")  # 越界就报错并带值。
    return value  # 返回校验通过的概率值。


def _require_positive_float(cfg: Mapping[str, Any], name: str, *, default: float) -> float:
    """从配置映射中读取并校验一个正浮点数字段。

    参数:
    `cfg` 是配置字典。
    `name` 是要读取的字段名。
    `default` 是字段缺失时的默认值。

    返回值:
    返回校验通过的正浮点数。

    失败条件:
    字段不是数值或不是正数时抛出异常。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "name": name,
        "default": default,
        "cfg_type": type(cfg).__name__,
    }, "_require_positive_float 入口参数")
    raw_value = cfg.get(name, default)  # 从配置里读取原始值，缺失时用默认值。
    if not is_numeric(raw_value):  # 布尔不算数值，必须排除。
        raise TypeError(f"{name} must be numeric, got {type(raw_value).__name__}")  # 类型不对就报错并带实际类型。
    try:  # 巨大整数转 float 可能抛 OverflowError，必须捕获。
        value = float(raw_value)  # 统一转成浮点数。
    except (OverflowError, ValueError) as exc:  # 捕获溢出和无效转换。
        raise ValueError(f"{name} must be a finite numeric value, got {raw_value!r}") from exc  # 转换失败就报错。
    if not math.isfinite(value):  # 必须是有限数，防止 NaN/Inf 静默传播到除法运算导致风险计算崩溃。
        raise ValueError(f"{name} must be finite, got {value}")  # 非有限就报错并带值。
    if value <= 0.0:  # 正浮点数必须严格大于 0。
        raise ValueError(f"{name} must be positive, got {value}")  # 非正就报错并带值。
    return value  # 返回校验通过的正浮点数。


def _as_float_tensor(value: Any, *, name: str, last_dim: int) -> torch.Tensor:
    """把输入统一转成二维 float32 张量，并校验最后一维宽度。

    参数:
    `value` 是待转换的输入，可以是标量、一维或二维序列。
    `name` 是错误信息里显示的字段名。
    `last_dim` 是期望的最后一维宽度。

    返回值:
    返回形状为 (batch, last_dim) 的 float32 张量。

    失败条件:
    输入维度超过二维或最后一维宽度不匹配时抛出异常。
    """
    tensor = torch.as_tensor(value, dtype=torch.float32)  # 先统一转成 float32 张量。
    if tensor.ndim == 0:  # 标量自动扩展成 (1, 1)。
        tensor = tensor.reshape(1, 1)  # 标量变成单元素二维张量。
    elif tensor.ndim == 1:  # 一维向量自动补 batch 维。
        tensor = tensor.unsqueeze(0)  # 在第 0 维插入，变成 (1, last_dim)。
    if tensor.ndim != 2:  # 最终必须是二维。
        raise ValueError(f"{name} must be a 1D or 2D tensor-like object")  # 超过二维就拒绝。
    if tensor.shape[-1] != last_dim:  # 最后一维宽度必须和期望一致。
        raise ValueError(f"{name} last dimension must be {last_dim}, got {tensor.shape[-1]}")  # 不一致就报错。
    if not torch.isfinite(tensor).all():  # 有限性检查，防止 NaN/Inf 静默传播到后续状态更新和风险计算。
        raise ValueError(f"{name} must contain only finite values")  # 非有限值就拒绝。
    return tensor  # 返回标准化后的二维张量。


def _coerce_dt_tensor(dt: Any, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """把时间间隔输入统一转成一维张量，并校验批次维度和非负性。

    参数:
    `dt` 是时间间隔输入，可以是标量或一维序列。
    `batch_size` 是期望的批次大小。
    `device` 是目标设备。
    `dtype` 是目标数据类型。

    返回值:
    返回形状为 (batch_size,) 的时间间隔张量。

    失败条件:
    批次维度不匹配或存在负值时抛出异常。
    """
    dt_tensor = torch.as_tensor(dt, dtype=dtype, device=device)  # 先统一转成指定类型和设备的张量。
    if dt_tensor.ndim == 0:  # 标量自动扩展到整个批次。
        dt_tensor = dt_tensor.expand(batch_size)  # 把标量复制成 batch_size 个。
    else:  # 非标量时压成一维。
        dt_tensor = dt_tensor.reshape(-1)  # 确保是一维张量。
    if dt_tensor.shape[0] != batch_size:  # 批次维度必须和输入一致。
        raise ValueError(f"dt batch dimension ({dt_tensor.shape[0]}) must match input_tensor batch ({batch_size})")  # 不一致就报错并带维度。
    if not torch.isfinite(dt_tensor).all():  # 有限性检查必须在非负检查之前：NaN < 0.0 返回 False 会绕过非负检查，NaN 随后通过 torch.exp(-dt*...) 永久污染 cell state。
        raise ValueError("dt must be finite (no NaN or Inf)")  # 非有限值直接拒绝。
    if torch.any(dt_tensor < 0.0):  # 时间间隔不能为负。
        raise ValueError(f"dt must be non-negative, got min={float(dt_tensor.min())}")  # 出现负值就拒绝并带最小值。
    return dt_tensor  # 返回校验通过的时间间隔张量。


class LiquidCell(nn.Module):
    """时间感知、缺失感知的液体状态更新单元。

    这个 Cell 实现了基于连续时间递推的状态更新机制，核心特点包括：
    - 时间间隔感知：状态更新幅度随 dt 变化，dt=0 时状态不变。
    - 缺失掩码支持：缺失特征用 0 填充，掩码信息传递给更新逻辑。
    - 坏观测风险评估：根据多个信号（质量、残差、重投影误差等）评估当前观测的可靠性。
    - 可靠性门控：基于输入和状态计算可靠性，控制状态更新幅度。

    参数:
    `cell_cfg` 是单元配置映射，包含 input_dim、hidden_dim 和各种超参数。

    返回值:
    step/forward 方法返回 (new_cell_state, cell_output) 元组。

    失败条件:
    配置参数不合法、输入维度不匹配或时间间隔为负时会抛出异常。
    """

    def __init__(self, cell_cfg: Mapping[str, Any] | None = None):
        super().__init__()  # 初始化 nn.Module 基类，注册参数管理。
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "has_cell_cfg": cell_cfg is not None,
            "cfg_keys": list(cell_cfg.keys()) if hasattr(cell_cfg, "keys") else None,
        }, "LiquidCell.__init__ 入口参数")
        cfg = dict(cell_cfg or {})  # 把空配置或映射配置归一成普通字典。
        self.input_dim = _require_int(cfg, "input_dim")  # 输入维度是必填项，必须为正整数。
        self.hidden_dim = _require_int(cfg, "hidden_dim", default=64)  # 隐藏维度默认 64。
        # 特征顺序：缺失/空列表/None → 空元组，后续 estimate_bad_observation_risk 直接返回默认风险值。
        # 提供重复名会直接拒绝。非空时长度必须与 input_dim 对齐，否则 _observed_feature 索引会越界。
        raw_feature_order = cfg.get("feature_order", None)  # 返回 None 表示未提供，与空列表区别开来仅用于日志/调试。
        self.feature_order = tuple(str(name) for name in list(raw_feature_order or []))  # 空/None → ()；冻结为元组防止外部修改。
        if len(set(self.feature_order)) != len(self.feature_order):  # 重复名会导致 _feature_index_by_name 静默覆盖，与 LSTM 侧 network.py 对齐。
            raise ValueError("feature_order must not contain duplicate field names")  # 重复名直接拒绝。
        if self.feature_order and len(self.feature_order) != self.input_dim:  # feature_order 长度必须与 input_dim 对齐，否则 _observed_feature 索引会越界。
            raise ValueError(  # 长度不一致直接拒绝，把运行期 IndexError 提前到构造期。
                f"feature_order length ({len(self.feature_order)}) must match input_dim ({self.input_dim})"
            )

        # 更新幅度控制参数：v3 大改放宽 scale 控制。
        # 原 design: floor=0.70, span=0.30 → update_scale ∈ [0.70, 1.00]
        # v3 改造: floor=0.10, span=0.80 → update_scale ∈ [0.10, 0.90]
        # 这让 LiquidCell 拥有比 LSTM 4-gate 更大的更新幅度可调空间。
        # reliability=0 时保守更新 (scale=0.10)，reliability=1 时激进更新 (scale=0.90)。
        self.update_scale_floor = _require_probability_like(cfg, "update_scale_floor", default=0.10)
        self.update_scale_span = _require_probability_like(cfg, "update_scale_span", default=0.80)
        if self.update_scale_floor + self.update_scale_span > 1.0:
            raise ValueError("update_scale_floor + update_scale_span must be <= 1")

        # B2 路径 (2026-07-26 e9 OOD 根因 v2): cfA/cfB clip 限幅, 防止 OOD 大残差下
        # time_A_projection / time_B_projection 的 sigmoid 输出漂移到极值 0/1 让 t_interp 失稳.
        # 2026-07-26 τ-scale 修复: cfB 是 *归一化* 时间速率因子 ∈ (0,1]，真实速率为
        #   B_eff = time_rate_scale * cfB  (单位 1/s)
        # 因此 cfB 的 clip 只约束归一化因子，不再用 [0.5,0.95] 强行抬高“伪时间常数”。
        # 默认 cfA_clip ∈ [0.05, 0.99]（保留可学的渐近幅度范围）;
        # 默认 cfB_clip ∈ [0.05, 0.99]（归一化速率，真正尺度由 time_rate_scale 提供）。
        self.cfA_clip_min = _require_probability_like(cfg, "cfA_clip_min", default=0.05)
        self.cfA_clip_max = _require_probability_like(cfg, "cfA_clip_max", default=0.99)
        self.cfB_clip_min = _require_probability_like(cfg, "cfB_clip_min", default=0.05)
        self.cfB_clip_max = _require_probability_like(cfg, "cfB_clip_max", default=0.99)
        if not (0.0 <= self.cfA_clip_min < self.cfA_clip_max <= 1.0):
            raise ValueError(f"cfA_clip range invalid: [{self.cfA_clip_min}, {self.cfA_clip_max}]")
        if not (0.0 <= self.cfB_clip_min < self.cfB_clip_max <= 1.0):
            raise ValueError(f"cfB_clip range invalid: [{self.cfB_clip_min}, {self.cfB_clip_max}]")

        # 时间速率尺度（1/s）：cfB 是归一化因子，B_eff = time_rate_scale * cfB (1/s)。
        # 默认值按 sim 真实 per-step dt 校准（data/raw/sim 事件流，2026-07-26 实测）：
        #   - 合并事件 dt 中位数 ≈ 1.02ms，p90 ≈ 5.45ms，p95 ≈ 7.88ms
        #   - 在 init A=sigmoid(2.5)=0.924, cfB=sigmoid(1.0)=0.731 下，scale=180 给出：
        #       dt=1.02ms (p50) → t_interp ≈ 0.116  （达到 LSTM 步级更新下限 ~0.10）
        #       dt=5.45ms (p90) → t_interp ≈ 0.473
        #       dt=7.88ms (p95) → t_interp ≈ 0.597
        #       dt=174ms  (max) → t_interp → A=0.924（不进 clamp, rate_dt=22.9 < 40）
        #   - τ_eff init = 131.6/s, 时间常数 7.6ms，与 sim p90-p95 dt 对齐
        # 旧实现把 cfB 直接当 1/s 用（B_eff≤1/s），p50 dt 时 t_interp≈6e-4，hidden 几乎冻结
        # → LNN 学不动、系统性弱于 LSTM（LSTM 每步无 dt 加权，固定更新）。
        # 可通过 cfg time_rate_scale 覆盖；切换数据集时务必按真实 dt 直方图重校。
        self.time_rate_scale = _require_positive_float(cfg, "time_rate_scale", default=180.0)
        if self.time_rate_scale > 1.0e4:
            raise ValueError(
                f"time_rate_scale must be <= 1e4 (1/s) to avoid stiff exp overflow, got {self.time_rate_scale}"
            )

        # B2 v3 (2026-07-26 e9 OOD inference-time 硬短路): risk_short_circuit_threshold.
        # 见 step 方法相应注释. 默认 1.0 (禁用短路) — 实证 (2026-07-26 inference-only 跑):
        #   thr=1.0 → rmse=11.0993
        #   thr=0.7 → rmse=11.0528 (-0.4%)
        #   thr=0.3 → rmse=11.0528 (-0.4%)   # 与 thr=0.7 完全一致
        # 解释: bad_observation_risk 在 e9 OOD N3 大残差下全部低于 0.3 (cell 内部 valid/quality/async
        # 信号未异常), 短路机制根本不触发. B2 v3 在 e9 上无效, 默认禁用 (1.0) 避免训练 loss 受影响.
        # 但保留 cfg 字段, 未来若 cell 改 risk 信号源让 OOD 时 risk 真升高, 可再启用.
        raw_threshold = cfg.get("risk_short_circuit_threshold", 1.0)
        if raw_threshold is None:
            raw_threshold = 1.0  # 禁用
        try:
            self.risk_short_circuit_threshold = float(raw_threshold)
        except (TypeError, ValueError):
            raise TypeError("risk_short_circuit_threshold must be numeric or None")
        if not (0.0 <= self.risk_short_circuit_threshold <= 1.0):
            raise ValueError(f"risk_short_circuit_threshold must be in [0,1], got {self.risk_short_circuit_threshold}")

        raw_reliability_bias = cfg.get("reliability_bias_init", 1.0)  # B+ 重训: 默认由 2.0 降至 1.0, sigmoid(1.0)≈0.73, 让 reliability 学得到非饱和 (cell.py t_interp 修复后配合)
        if not is_numeric(raw_reliability_bias):  # 布尔不算数值，必须排除。
            raise TypeError("reliability_bias_init must be numeric")  # 类型不对就报错。
        self.reliability_bias_init = float(raw_reliability_bias)  # 统一转成浮点数。
        if not math.isfinite(self.reliability_bias_init):  # 必须有限，防止 NaN/Inf 传播到 sigmoid 导致可靠性门控崩溃。
            raise ValueError("reliability_bias_init must be finite")  # 非有限就报错。
        if abs(self.reliability_bias_init) > 10.0:  # 上界保护，防止 sigmoid 饱和导致可靠性门控失效。
            raise ValueError(f"reliability_bias_init must be within [-10.0, 10.0], got {self.reliability_bias_init}")  # 超出范围就报错。

        # 坏观测风险评估参数：floor 是风险下限，span 是可调范围。
        self.bad_observation_floor = _require_probability_like(cfg, "bad_observation_floor", default=0.20)  # 坏观测风险下限，默认 0.20。
        self.bad_observation_span = _require_probability_like(cfg, "bad_observation_span", default=0.80)  # 坏观测风险跨度，默认 0.80。
        if self.bad_observation_floor + self.bad_observation_span > 1.0:  # 下限加跨度不能超过 1。
            raise ValueError("bad_observation_floor + bad_observation_span must be <= 1")  # 超过就报错。

        # 以下是一组坏观测风险评估的尺度参数，用于把原始信号归一化到 [0, 1] 风险区间。
        self.bad_observation_async_full_scale = _require_positive_float(
            cfg,
            "bad_observation_async_full_scale",
            default=float(ASYNC_GAP_FULL_SCALE_S),
        )  # 异步时间间隔的全尺度，超过此值风险视为 1。
        self.bad_observation_residual_full_scale = _require_positive_float(
            cfg,
            "bad_observation_residual_full_scale",
            default=0.50,
        )  # UWB 距离残差的全尺度。
        self.bad_observation_reproj_full_scale = _require_positive_float(
            cfg,
            "bad_observation_reproj_full_scale",
            default=float(VIO_REPROJ_ERR_FULL_SCALE),
        )  # VIO 重投影误差的全尺度。
        self.bad_observation_tracked_features_floor = _require_positive_float(
            cfg,
            "bad_observation_tracked_features_floor",
            default=float(VIO_TRACKED_FEATURES_FLOOR),
        )  # 跟踪特征数的阈值，低于此值风险上升。
        self.bad_observation_track_drop_full_scale = _require_positive_float(
            cfg,
            "bad_observation_track_drop_full_scale",
            default=25.0,
        )  # 特征数下降幅度的全尺度。
        self.bad_observation_reproj_slope_full_scale = _require_positive_float(
            cfg,
            "bad_observation_reproj_slope_full_scale",
            default=float(VIO_HIGH_REPROJ_ERR_THRESHOLD),
        )  # 重投影误差斜率的全尺度。
        self.bad_observation_geom_floor = _require_probability_like(
            cfg,
            "bad_observation_geom_floor",
            default=0.30,
        )  # 几何质量评分阈值，低于此值风险上升。
        self.bad_observation_interaction_coeff = _require_positive_float(
            cfg,
            "bad_observation_interaction_coeff",
            default=0.75,
        )  # 多信号交互项的系数，控制风险信号的协同放大。
        if self.bad_observation_interaction_coeff > 10.0:  # 上界保护，防止交互项导致风险爆炸到 inf。
            raise ValueError(f"bad_observation_interaction_coeff must be <= 10.0, got {self.bad_observation_interaction_coeff}")  # 超出上界就拒绝。

        # 构建内部线性层。
        augmented_input_dim = self.input_dim * 2  # 增广输入维度 = 原始特征 + 缺失掩码，各占 input_dim。
        self.backbone_projection = nn.Linear(augmented_input_dim + self.hidden_dim, self.hidden_dim)  # 主干投影：把增广输入和状态拼起来映射到隐藏空间。
        self.ff1_projection = nn.Linear(self.hidden_dim, self.hidden_dim)  # 前馈第一层：在主干输出上做非线性变换。
        self.ff2_projection = nn.Linear(self.hidden_dim, self.hidden_dim)  # 前馈第二层：生成候选状态。
        # removed lowercase time_a_projection / time_b_projection (dead v6 leftover;
        # closed-form t_interp uses uppercase time_A_projection / time_B_projection below).
        self.reliability_projection = nn.Linear(augmented_input_dim + self.hidden_dim, 1)  # 可靠性投影：输出一个标量表示输入的可靠程度。
        # v3 大改：新增 forget gate，类似 LSTM 4-gate 风格，控制状态更新力度。
        # 与 reliability 共同调制 eff_update：forget 控制"是否重置"，reliability 控制"重置幅度"。
        # v6 Patch 2 (GLA Yang et al. 2024 arXiv:2312.06635): per-dimension data-dependent
        # gating → 每个隐藏维度独立 retention rate，提升长序列泛化。
        self.forget_projection = nn.Linear(augmented_input_dim + self.hidden_dim, self.hidden_dim)  # forget gate 每个隐藏维度独立门控。

        # v6 Patch 1 (CfC Hasani et al. 2022 NeurIPS arXiv:2211.01768): 闭式时间插值 t_interp。
        # 旧 dt_factor = 1 - exp(-dt * (a + b))。
        # 新 closed-form: t_interp = A * (1 - exp(-B * dt))，A∈(0,1], B>0 可学习，dt=0 → t_interp=0，dt→∞ → A。
        # A 控制渐近更新幅度，B 控制时间常数 → 更稳定的长时常数、更精确的不规则时间插值。
        self.time_A_projection = nn.Linear(self.hidden_dim, self.hidden_dim)  # 渐近幅度 A ∈ (0,1]。
        self.time_B_projection = nn.Linear(self.hidden_dim, self.hidden_dim)  # 时间常数 B > 0。

        # B2 v4 (2026-07-26 e9 OOD 根因 v4): cell-level state forget gate (LSTM 4-gate 等价).
        # 当前 forget_projection 作用在 gated_update 上调 magnitude (即 update_weight 0..1),
        # LSTM 的 forget gate 是 cell state 上乘法式 forget (c_t = f*c_{t-1} + i*g),
        # OOD 大残差时 f→0 让 c_t = c_{t-1}*0 + ... 完全更新, 或 f→1 让 c_t = c_{t-1} 完全保留.
        # B2 v4 加独立 forget_root_projection 让 cell state 也有乘法 forget, 不依赖 bad_observation_risk
        # 信号源 (B2 v3 实证: bad_observation_risk 在 e9 OOD 下未超 0.3, 短路无效).
        # B2 v5 修正 (2026-07-26 e9 OOD 根因 v5): forget_root_projection 输入从 prev_state 改为
        # cat(prev_state, backbone) 即 2*hidden 维, 让 f_root 看到当前 step 的 input 信号 (backbone 已编码 input).
        # input-conditioned forget gate, 与 LSTM forget gate f_t=sigmoid(W_f·[h_{t-1}, x_t]+b_f) 等价.
        # B2 v4 实证: 仅基于 prev_state 的 forget_root 训练后 bias=0.425 全场均匀偏低, 没在 OOD N3
        # 时饱和到 1, e9 RMSE 11.13 ≈ default 11.10 (无改善) - 因 forget_root 看不到 input, 学不到
        # "看到大残差 input 时 forget=1" 的策略. v5 cat backbone (hidden_dim) 后 f_root 能感知 input 信号.
        # new_state = f_root * prev_state + (1 - f_root) * new_state_v1
        #   - f_root→0: 完全 apply v1 update (LIKE-LSTM f=0 i=g)
        #   - f_root→1: 完全保留 prev_state (LIKE-LSTM f=1 i=0)
        # 训练时 cell 学到 f_root 在 OOD 大残差时 (backbone 异常) 饱和到 1. 重训后 expectation:
        #   e9 OOD N3 大残差 state 不漂移, 同分布 N0/N1 时正常更新.
        # B2 v5: input_dim 从 hidden 改为 2*hidden (cat(prev_state, backbone) 维度).
        self.forget_root_projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim)  # B2 v5: input-conditioned forget gate
        # 配置: 是否启用 B2 v4/v5 forget_root (默认 True). 设 False 回 B2 v1/v3 行为 (用于 ablation).
        self.enable_forget_root = bool(cfg.get("enable_forget_root", True))

        self.output_norm = nn.LayerNorm(self.hidden_dim)  # 输出归一化：对更新后的状态做 LayerNorm，稳定数值。
        self._feature_index_by_name = {name: index for index, name in enumerate(self.feature_order)}  # 特征名到索引的映射表，供坏观测评估按名字定位特征。
        self.reset_parameters()  # 初始化所有可学习参数。

    def reset_parameters(self) -> None:
        """重置所有可学习参数到初始状态。

        参数:
        无外部参数。

        返回值:
        无返回值，只修改内部参数。
        """
        _fill_linear(self.backbone_projection, start=-0.12, end=0.12)  # 主干投影用窄范围等间距初始化。
        _fill_linear(self.ff1_projection, start=-0.10, end=0.10)  # 前馈第一层用稍窄范围初始化。
        _fill_linear(self.ff2_projection, start=-0.08, end=0.08)  # 前馈第二层用更窄范围初始化。
        # removed _fill_linear for dead time_a/time_b projections (v6 leftover, layers above also removed).
        _fill_linear(self.forget_projection, start=-0.04, end=0.04)  # v3: forget gate 窄范围初始化。
        _fill_linear(self.time_A_projection, start=-0.04, end=0.04)  # time_A 渐近幅度 A ∈ (0,1] 经 sigmoid，窄范围初始化。
        _fill_linear(self.time_B_projection, start=-0.06, end=0.06)  # time_B 归一化速率因子经 sigmoid；真实速率 = time_rate_scale * cfB。
        # B2 v4 (2026-07-26): forget_root_projection 窄初始化 (sigmoid 输出), bias=0.0 中性 →
        # f_root=sigmoid(0)=0.5，初始 50/50 between 保留 prev_state 与 apply v1 update。
        # 旧值 +0.5 → f_root=0.622 → 1-f_root=0.378，叠加 forget_gate=0.378 后乘 0.143，
        # 在 p50 dt(1ms) t_interp=0.116 时 eff_per_step 仅 0.015（LSTM 步级 ~0.5，差距 30×）。
        # 把 forget_root.bias 从 +0.5 改为 0.0 让 init 不偏向"死保 prev_state"，交给训练去学。
        _fill_linear(self.forget_root_projection, start=-0.04, end=0.04)
        # 时间常数初始化 (2026-07-26 τ-scale 修复 + τ init 重校准):
        #   t_interp = A * (1 - exp(-time_rate_scale * cfB * dt))
        #   - time_A bias=+2.5  → A=sigmoid(2.5)≈0.924
        #     渐近近完全更新但仍 < cfA_clip_max=0.99，留学习空间
        #     （旧 bias=1.5 → A=0.817 把"渐近更新上限"压到 0.817，浪费可学幅度）
        #   - time_B bias=+1.0  → cfB=sigmoid(1.0)≈0.731
        #     配合默认 time_rate_scale=180 → B_eff≈131.6/s
        #     对应时间常数 τ ≈ 7.6ms，与 sim 真实合并事件 dt 直方图
        #     p90=5.45ms / p95=7.88ms 对齐（cell 在该尺度上区分能力最强）
        #   - 校验 (sim 真实 dt, init A/cfB):
        #       dt=1.02ms (p50)  → t_interp≈0.116 （LSTM 步级更新下限 ~0.10 ✓）
        #       dt=5.45ms (p90)  → t_interp≈0.473
        #       dt=7.88ms (p95)  → t_interp≈0.597
        #       dt=174ms  (max)  → t_interp≈A=0.924（rate_dt=22.9<40 不进 clamp ✓）
        #   - 对照: v6 原始实现把 sigmoid(time_B)≤1 直接当 1/s 用（等价 scale=1, A=0.817, cfB=0.731）
        #     p50 dt → t_interp≈6e-4，本轮 scale=180 后提升约 200×
        with torch.no_grad():
            self.time_A_projection.bias.fill_(2.5)
            self.time_B_projection.bias.fill_(1.0)
            # B2 v4 forget_root bias=0.0 中性 → f_root=sigmoid(0)=0.5
            # 旧 +0.5 → f_root=0.622 → 1-f_root=0.378，叠 forget=0.378 后乘 0.143，
            # p50 dt t_interp=0.116 被 eff_per_step 压到 0.015（LSTM 步级 ~0.5，差 30×）。
            self.forget_root_projection.bias.fill_(0.0)
        with torch.no_grad():  # 以下初始化不需要梯度记录。
            self.reliability_projection.weight.zero_()  # 可靠性投影权重清零，从中性状态开始。
            self.reliability_projection.bias.fill_(self.reliability_bias_init)  # 可靠性偏置设为正值，初始偏向"可靠"。
            # v3 init: forget bias 从 -0.5 改为 +1.0（LSTM 习惯: forget_gate=sigmoid(1.0)≈0.731）
            # 旧 bias=-0.5 → forget=0.378，与 forget_root 叠乘 0.143 后 t_interp 0.116 被压到
            # eff_per_step 0.015，比 LSTM 步级 ~0.5 小 30×。改 +1.0 后 eff≈0.039，N50 从 45 步
            # 降到 23 步，仍保守但与 LSTM 同量级。OOD 大残差下训练可学 forget→0 实现短路。
            self.forget_projection.weight.zero_()  # forget 投影权重清零，从中性状态开始（输出仅由 bias 决定）。
            self.forget_projection.bias.fill_(1.0)  # LSTM 习惯的 forget bias，初始偏向"应用更新"。
            self.output_norm.weight.fill_(1.0)  # LayerNorm 权重初始化为 1。
            self.output_norm.bias.zero_()  # LayerNorm 偏置清零。

    def reset_state(
        self,
        *,
        batch_size: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """创建全零初始隐藏状态。

        参数:
        `batch_size` 是批次大小，默认 1。
        `device` 是目标设备，默认 None 表示 CPU。
        `dtype` 是数据类型，默认 float32。

        返回值:
        返回形状为 (batch_size, hidden_dim) 的全零张量。

        失败条件:
        batch_size 不是正整数时抛出异常。
        """
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "batch_size": batch_size,
            "device": str(device) if device is not None else None,
            "dtype": str(dtype) if dtype is not None else None,
        }, "LiquidCell.reset_state 入口参数")
        if not is_integer(batch_size):  # 布尔不算整数。
            raise TypeError("batch_size must be an integer")  # 类型不对就报错。
        if batch_size < 1:  # 批次大小至少为 1。
            raise ValueError("batch_size must be positive")  # 小于 1 就拒绝。
        return torch.zeros(
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype or torch.float32,
        )  # 返回全零初始状态张量。

    def _observed_feature(self, input_tensor: torch.Tensor, missing_mask: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """按特征名从输入张量中提取观测值和观测标记。

        参数:
        `input_tensor` 是当前步特征张量，形状 (batch, input_dim)。
        `missing_mask` 是当前步缺失掩码张量，形状 (batch, input_dim)。
        `name` 是要提取的特征名。

        返回值:
        返回 (value, observed) 元组, value 是特征值(缺失时为 0), observed 是是否观测到(0 或 1).
        """
        # removed debug print_dict (was flooding logs, ~6 lines per (batch, feature) call).
        index = self._feature_index_by_name.get(name)  # 从映射表里查找特征位置。
        if index is None:  # 特征名不在映射表中，返回零值。
            zeros = input_tensor.new_zeros((input_tensor.shape[0],))  # 返回全零值和全零观测标记。
            return zeros, zeros  # 缺失特征默认值 0、观测标记 0。
        if index >= input_tensor.shape[1]:  # 特征名在映射表中但索引超出维度范围，说明配置错误。
            raise IndexError(  # 抛出异常而非静默返回零，避免掩盖特征维度不匹配。
                f"Feature '{name}' has index {index} but input_tensor only has "
                f"{input_tensor.shape[1]} columns. Check feature_order and input dimension."
            )
        observed = 1.0 - missing_mask[:, index]  # 掩码取反：1 表示观测到，0 表示缺失。
        value = torch.where(observed.bool(), input_tensor[:, index], input_tensor.new_zeros(input_tensor.shape[0]))  # 缺失时清零（用 where 避免 NaN*0=NaN），显式对齐形状避免标量广播。
        return value, observed  # 返回特征值和观测标记。

    def estimate_bad_observation_risk(
        self,
        input_tensor: Any,
        missing_mask: Any | None = None,
    ) -> torch.Tensor:
        """根据多个信号评估当前观测的坏观测风险。

        参数:
        `input_tensor` 是当前步特征张量。
        `missing_mask` 是当前步缺失掩码，None 表示全部观测到。

        返回值:
        返回形状为 (batch,) 的风险值张量，范围 [0, 1]。

        关键局部变量:
        `risk` 累加各项风险贡献。
        `valid_value/quality_value/async_value` 等是各信号的观测值。
        `normalized_terms` 收集所有归一化后的风险项，用于计算交互项。
        """
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "input_tensor_type": type(input_tensor).__name__,
            "has_missing_mask": missing_mask is not None,
            "input_dim": self.input_dim,
        }, "LiquidCell.estimate_bad_observation_risk 入口参数")
        input_tensor = _as_float_tensor(input_tensor, name="input_tensor", last_dim=self.input_dim)  # 统一转成二维张量。
        if missing_mask is None:  # 没给掩码时默认全部观测到。
            missing_mask_tensor = torch.zeros_like(input_tensor)  # 全零掩码表示无缺失。
        else:
            missing_mask_tensor = _as_float_tensor(
                missing_mask,
                name="missing_mask",
                last_dim=self.input_dim,
            ).to(device=input_tensor.device, dtype=input_tensor.dtype).clamp(0.0, 1.0)  # 统一类型和设备，并裁剪到 [0, 1]；用非 in-place 版本避免修改调用方原始张量。

        # 当 feature_order 为空（缺失或显式空列表）时，所有命名特征都查不到索引，
        # _observed_feature 始终返回零值与未观测标记，risk_terms 收集到的是空列表，
        # 最终 risk 保持 0，最终公式 clamp(floor + span * 0) 自洽回退到 floor。
        # 这里直接短路返回，跳过 _observed_feature 的 N 次空查表，同时把"无命名特征 → 默认风险下限"
        # 这个契约显式化，便于读者一眼看出降级行为，而不是要从 9 个 if-any 分支里反推。
        if not self.feature_order:
            return input_tensor.new_full(
                (input_tensor.shape[0],),
                float(self.bad_observation_floor),
            ).clamp(0.0, 1.0)

        risk = input_tensor.new_zeros((input_tensor.shape[0],))  # 初始化风险累加器为 0。

        # 以下逐项评估各信号对坏观测风险的贡献。
        # 使用加权平均而非直接累加，避免多信号同时恶化时风险饱和到1.0失去区分能力。
        risk_terms: list[tuple[torch.Tensor, torch.Tensor]] = []  # (weighted_value, observed) 对。

        # 1. valid 标志：无效观测直接贡献风险。
        valid_value, valid_observed = self._observed_feature(input_tensor, missing_mask_tensor, "valid")  # 读取 valid 特征。
        if valid_observed.any():  # 只在有观测时才计算。
            risk_terms.append((valid_observed * (1.0 - valid_value.clamp(0.0, 1.0)), valid_observed))  # valid=0 时风险最高。

        # 2. quality 分数：质量越低风险越高。
        quality_value, quality_observed = self._observed_feature(input_tensor, missing_mask_tensor, "quality")  # 读取 quality 特征。
        if quality_observed.any():  # 只在有观测时才计算。
            risk_terms.append((quality_observed * (1.0 - quality_value.clamp(0.0, 1.0)), quality_observed))  # quality=0 时风险最高。

        # 3. 异步时间间隔：间隔越大风险越高。
        async_value, async_observed = self._observed_feature(input_tensor, missing_mask_tensor, "modality_gap_dt")  # 读取异步间隔。
        if async_observed.any():  # 只在有观测时才计算。
            async_risk = (async_value.abs() / self.bad_observation_async_full_scale).clamp(0.0, 1.0)  # 归一化到 [0, 1]。
            risk_terms.append((async_observed * async_risk, async_observed))  # 异步风险。

        # 4. UWB 距离残差：残差越大风险越高。
        residual_value, residual_observed = self._observed_feature(input_tensor, missing_mask_tensor, "uwb_range_residual")  # 读取残差。
        if residual_observed.any():  # 只在有观测时才计算。
            residual_risk = (residual_value.abs() / self.bad_observation_residual_full_scale).clamp(0.0, 1.0)  # 归一化到 [0, 1]。
            risk_terms.append((residual_observed * residual_risk, residual_observed))  # 残差风险。

        # 铁律 3 (audit #18-补3, 2026-07-23): 删除 reproj_err / vio_reproj_err_slope /
        # tracked_features / tracked_features_drop 4 处 _observed_feature 死调用 + 4 个 if-any
        # 风险块。这 4 个死键已从 model_factory.LIQUID_CONTEXT_FEATURE_KEYS 删除 (VIO 紧耦合
        # 不再输出), _feature_index_by_name.get(name) 永远返回 None, _observed_feature 直接
        # 返回全零 value + 全零 observed, 下游 if xxx_observed.any() 永远 False 短路, 这 4
        # 段 risk_terms.append 不可能进入, 维护成本浪费且掩盖"特征存不存在"的真实假设。
        # risk_terms：valid/async/residual/geom 为主；quality 仅当 feature 中存在时计入（主表 feature_order 不含 quality）。

        # 9. 几何质量评分：评分低于阈值时风险上升。
        geom_value, geom_observed = self._observed_feature(input_tensor, missing_mask_tensor, "geom_score")  # 读取几何评分。
        if geom_observed.any():  # 只在有观测时才计算。
            geom_risk = torch.clamp((self.bad_observation_geom_floor - geom_value) / max(self.bad_observation_geom_floor, 1e-6), min=0.0, max=1.0)  # 低于阈值时风险为正。
            risk_terms.append((geom_observed * geom_risk, geom_observed))  # 几何风险。

        # 计算加权平均风险：取已观测信号的均值，而非直接累加。
        # 直接累加会导致多信号同时恶化时风险饱和到1.0，失去细粒度区分能力。
        if risk_terms:
            risk_sum = sum(term[0] for term in risk_terms)
            obs_sum = sum(term[1] for term in risk_terms)
            # 已观测信号数（避免除零），每个 observed 是 0 或 1 的张量。
            num_observed = torch.clamp(obs_sum, min=1.0)
            risk = risk_sum / num_observed  # 加权平均：总风险 / 已观测信号数。
        else:
            risk = torch.zeros_like(input_tensor[:, 0:1]).squeeze(-1)

        # 10. 交互项：多个风险信号同时高时额外放大。
        # 交互项基于加权平均后的 risk 进行放大：当均值风险高且最大单项风险也高时，
        # 说明多个信号同时恶化，应额外放大风险。
        if risk_terms:
            # 计算已观测信号中最大单项风险。
            max_term = torch.stack([term[0] for term in risk_terms], dim=1).max(dim=1).values
            interaction = self.bad_observation_interaction_coeff * risk * torch.clamp(max_term, min=0.0, max=1.0)
            risk = risk + interaction  # 叠加交互风险。

        # 最终风险值映射到 [floor, floor+span] 区间。
        # 不对 risk 做内层 clamp(0,1)：交互项叠加后 risk 可能超过 1.0，
        # 内层 clamp 会截断交互项的区分能力（多信号同时恶化时 risk=1.34 和 1.54
        # 都被截断到 1.0，映射后均为 1.0，无法区分"中度恶化"和"严重恶化"）。
        # floor + span 映射本身会将超1.0的risk映射到 floor+span 以上，
        # 外层 clamp(0, 1) 仍保底不超过1.0，但保留了交互项的区分度。
        return torch.clamp(self.bad_observation_floor + (self.bad_observation_span * risk), 0.0, 1.0)

    def step(
        self,
        input_tensor: Any,
        cell_state: Any | None = None,
        *,
        dt: Any,
        missing_mask: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行一步 Liquid Cell 更新。

        参数:
        `input_tensor` 是当前步特征输入。
        `cell_state` 是上一步隐藏状态，None 时自动初始化为零。
        `dt` 是当前步时间间隔。
        `missing_mask` 是当前步缺失掩码，None 表示全部观测到。

        返回值:
        返回 (new_cell_state, cell_output) 元组。
        `new_cell_state` 是更新后的隐藏状态。
        `cell_output` 是经过 LayerNorm 后的输出。

        关键局部变量:
        `observed_input` 是缺失位置清零后的特征。
        `cell_input` 是增广输入（观测特征 + 掩码）。
        `backbone` 是主干投影输出。
        `candidate` 是候选新状态。
        `time_scale_a/b` 是时间衰减率。
        `dt_factor` 是时间间隔因子，dt=0 时为 0。
        `reliability` 是可靠性标量。
        `update_scale` 是最终更新幅度。
        `bad_observation_risk` 是坏观测风险。
        `effective_update` 是有效更新量。
        """
        input_tensor = _as_float_tensor(input_tensor, name="input_tensor", last_dim=self.input_dim)  # 统一输入为二维张量。
        if missing_mask is None:  # 没给掩码时默认全部观测到。
            missing_mask_tensor = torch.zeros_like(input_tensor)  # 全零掩码。
        else:
            missing_mask_tensor = _as_float_tensor(
                missing_mask,
                name="missing_mask",
                last_dim=self.input_dim,
            ).to(device=input_tensor.device, dtype=input_tensor.dtype).clamp(0.0, 1.0)  # 统一类型、设备和范围；用非 in-place 版本避免修改调用方原始张量。

        if cell_state is None:  # 没给状态时自动初始化。
            state_tensor = self.reset_state(
                batch_size=input_tensor.shape[0],
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )  # 创建全零初始状态。
        else:
            state_tensor = _as_float_tensor(
                cell_state,
                name="cell_state",
                last_dim=self.hidden_dim,
            ).to(device=input_tensor.device, dtype=input_tensor.dtype)  # 统一状态张量的类型和设备。
            if state_tensor.shape[0] != input_tensor.shape[0]:  # 批次维度必须一致。
                raise ValueError("cell_state batch dimension must match input_tensor")  # 不一致就报错。

        dt_tensor = _coerce_dt_tensor(
            dt,
            batch_size=input_tensor.shape[0],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )  # 统一时间间隔张量。
        observed_input = torch.where(missing_mask_tensor.bool(), input_tensor.new_zeros(input_tensor.shape), input_tensor)  # 缺失位置清零（用 where 避免 NaN*0=NaN 传播），显式对齐形状避免标量广播。
        cell_input = torch.cat((observed_input, missing_mask_tensor), dim=-1)  # 增广输入 = 观测特征 + 掩码。
        state_and_input = torch.cat((cell_input, state_tensor), dim=-1)  # 拼接增广输入和当前状态。

        # 主干网络：从拼接向量生成隐藏表示。
        backbone = torch.tanh(self.backbone_projection(state_and_input))  # 主干投影 + tanh 激活。
        ff_hidden = torch.tanh(self.ff1_projection(backbone))  # 前馈第一层 + tanh 激活。
        candidate = torch.tanh(self.ff2_projection(ff_hidden))  # 前馈第二层生成候选状态。

        # v6 Patch 1 + τ-scale (CfC Hasani et al. 2022; 2026-07-26 对齐真实 Δt + τ init 重校):
        # t_interp = A * (1 - exp(-time_rate_scale * cfB * dt))
        #   A = sigmoid(time_A) ∈ (0,1]     渐近更新幅度（init bias=2.5 → A≈0.924）
        #   cfB = sigmoid(time_B) ∈ (0,1]   归一化速率因子（init bias=1.0 → cfB≈0.731）
        #   B_eff = time_rate_scale * cfB     真实时间速率 (1/s)，default 180 → τ_eff=131.6/s
        # dt=0 → t_interp=0；dt→∞ → t_interp→A。
        # 禁止把 cfB 直接当 1/s：cfB≤1 时对 dt≈1ms 的 t_interp 会锁死在 ~6e-4，隐状态几乎不更新。
        # CfC 闭式解本身就是 ODE dx/dt=B(c−x) 的精确解，不存在数值求解器/步长不匹配问题；
        # 真正要匹配的是 τ init 与真实 dt 直方图的尺度（已在 reset_parameters 完成）。
        cfA = torch.sigmoid(self.time_A_projection(state_tensor))  # 渐近幅度 A，每个隐藏维度独立。
        cfB = torch.sigmoid(self.time_B_projection(candidate))  # 归一化速率因子，每个隐藏维度独立。
        cfA = cfA.clamp(min=self.cfA_clip_min, max=self.cfA_clip_max)
        cfB = cfB.clamp(min=self.cfB_clip_min, max=self.cfB_clip_max)
        # D5: τ range check [0.1s, 10s] per handbook §D5.
        # τ_eff = 1.0 / (time_rate_scale * cfB) (seconds), cfB already sigmoid ∈ (0,1).
        # Use clamp(min=1e-3) to avoid division by zero for cfB near 0.
        tau_eff_s = 1.0 / (self.time_rate_scale * cfB.clamp(min=1e-3))
        if not ((tau_eff_s >= 0.1) & (tau_eff_s <= 10.0)).all():
            import logging as _d5_log
            _d5_log.getLogger(__name__).warning(
                "D5: effective τ (seconds) outside documented [0.1s, 10s] range "
                "(time_rate_scale=%.1f, cfB mean=%.4f, τ_eff mean=%.4fs). "
                "Architecture math is correct (CfC closed-form), but τ range guidance from §D5 is not satisfied.",
                self.time_rate_scale, float(cfB.mean().item()), float(tau_eff_s.mean().item()),
            )
        # 限幅指数自变量，避免极大 dt * rate 在 float32 下溢出（stiff 区间数值保护）。
        rate_dt = (self.time_rate_scale * cfB * dt_tensor.unsqueeze(-1)).clamp(max=40.0)
        t_interp = cfA * (1.0 - torch.exp(-rate_dt))  # CfC 闭式时间插值，dt=0 时为 0。
        # zero_dt_mask 仍保留作为硬冻保险，避免 dt=0 时浮点漂移。

        # 可靠性门控：基于输入和状态计算更新幅度。
        reliability = torch.sigmoid(self.reliability_projection(state_and_input))  # 可靠性标量，0 到 1 之间。
        update_scale = self.update_scale_floor + (self.update_scale_span * reliability)  # 更新幅度 = 下限 + 跨度 × 可靠性。
        bad_observation_risk = self.estimate_bad_observation_risk(input_tensor, missing_mask_tensor).unsqueeze(-1)  # 坏观测风险，增加一维便于广播。
        effective_update = update_scale * (1.0 - bad_observation_risk) * t_interp  # 有效更新 = 幅度 × (1 - 坏风险) × CfC 时间插值。

        # v3 大改：forget gate 控制状态是否重置。
        # 与 LSTM 的 forget gate 类似，但 wrapping 的是 eff_update，不直接乘 candidate。
        # forget≈0 时 new_state ≈ prev_state（保守），forget≈1 时完全应用 eff_update。
        forget_gate = torch.sigmoid(self.forget_projection(state_and_input))  # forget gate，0 到 1 之间。
        gated_update = forget_gate * effective_update  # gate 调制后的有效更新幅度。

        # B2 v3 (2026-07-26 e9 OOD 根因 v3): risk_short_circuit 硬短路大残差观测.
        # LSTM 在 OOD 大残差下 sigmoid forget gate 饱和到 1 + input gate 饱和到 0 + cell state 全保留 →
        # 隐式 short-circuit 大残差观测, 鲁棒性极佳. Liquid 的 forget_gate 由训练权重决定, OOD 时
        # 未必饱和到 0; bad_observation_risk 已学但还是可能不到 1, 让 eff_update 仍非零, state 仍漂移.
        # B2 v3 显式硬短路: 当 bad_observation_risk >= risk_short_circuit_threshold 时强制 gated_update=0,
        # 完全保留 prev_state, 不论训练权重如何. 这是 inference-time OOD 鲁棒保护层, 与训练可学可调.
        # 默认 threshold=1.0 (禁用短路, 见 __init__ 相应注释与 e9 实证): risk>=thr → 该步必短路; risk<thr 仍由 forget_gate 学.
        # 通过 cfg risk_short_circuit_threshold 覆盖 (None / 1.0 即禁用短路回到 B2 v1 行为).
        if self.risk_short_circuit_threshold < 1.0:
            risk_mask = (bad_observation_risk.squeeze(-1) >= self.risk_short_circuit_threshold).unsqueeze(-1)
            gated_update = torch.where(risk_mask, torch.zeros_like(gated_update), gated_update)

        # 状态更新：在旧状态和候选状态之间做插值，幅度由 forget*eff_update 决定。
        new_cell_state_v1 = state_tensor + gated_update * (candidate - state_tensor)  # 插值更新：gated_update 越大越偏向候选。

        # B2 v5 (2026-07-26 e9 OOD 根因 v5): cell-level state forget gate 应用.
        # new = f_root * prev_state + (1 - f_root) * new_v1, 其中 f_root 来自 forget_root_projection(cat(prev_state, backbone)).
        # f_root→1: 完全保留 prev_state (LSTM-like OOD 短路); f_root→0: 应用 v1 update (LIGHT cfA/cfB control).
        # 训练时 cell 学到 f_root 在 OOD 大残差时 (input 信号异常) 饱和到 1, 同分布 N0/N1 时学合适值.
        # 不依赖 bad_observation_risk 信号源 (B2 v3 实证在 e9 协议下该信号未异常).
        # B2 v5 修正: forget_root_projection 输入改为 cat(prev_state, backbone) 让 f_root 看到 input 信号 (LSTM 等价).
        #   - B2 v4 实证: 仅看 prev_state 时 forget_root bias 学到 ~0.425 全场均匀, 没在 OOD N3 饱和到 1.
        #   - v5 cat backbone (已编码 input 信息, hidden_dim) 后 f_root 能感知 "看到大残差 input 时 forget=1" 的策略.
        #   - 用 backbone (而非 raw cell_input) 因 backbone 维度 = hidden_dim 与 state 一致, cat 后 dim=2*hidden 与 init 一致.
        if self.enable_forget_root:
            # cat(state, backbone): [B, hidden] + [B, hidden] → [B, 2*hidden] 与 forget_root_projection input_dim 一致
            f_root_input = torch.cat([state_tensor, backbone], dim=-1)  # [B, 2*hidden]
            f_root = torch.sigmoid(self.forget_root_projection(f_root_input))  # ∈ (0, 1) per-dim.
            # 用 clamp 让 f_root 不漂到极值 0/1 (训练稳定性), clamp [0.05, 0.95].
            # 注: clamp 范围比 cfA/cfB 更宽, 让 f_root 仍有空间饱和到 LSTM-like 区间.
            f_root = f_root.clamp(min=0.05, max=0.95)
            new_cell_state = f_root * state_tensor + (1.0 - f_root) * new_cell_state_v1
        else:
            new_cell_state = new_cell_state_v1

        # dt=0 时硬冻状态：dt_factor=0 导致 eff_update=0，新状态必然等于旧状态，
        # 但仍显式做 where 以避免极小的浮点漂移（exp(-0)≈1 但 1-exp(-0) 可能是 1e-16 而非 0）。
        zero_dt_mask = (dt_tensor == 0).unsqueeze(-1)  # dt=0 的批次维度掩码。
        new_cell_state = torch.where(zero_dt_mask, state_tensor, new_cell_state)  # dt=0 时强制保持旧状态。
        cell_output = self.output_norm(new_cell_state)  # 输出经过 LayerNorm 稳定数值。
        return new_cell_state, cell_output  # 返回新状态和归一化输出。

    def forward(
        self,
        input_tensor: Any,
        cell_state: Any | None = None,
        *,
        dt: Any,
        missing_mask: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播入口，直接委托给 step 方法。

        参数:
        `input_tensor` 是当前步特征输入。
        `cell_state` 是上一步隐藏状态，None 时自动初始化。
        `dt` 是当前步时间间隔。
        `missing_mask` 是当前步缺失掩码，None 表示全部观测到。

        返回值:
        返回 (new_cell_state, cell_output) 元组，与 step 方法一致。
        """
        return self.step(
            input_tensor,
            cell_state,
            dt=dt,
            missing_mask=missing_mask,
        )  # 直接转发到 step 方法。
