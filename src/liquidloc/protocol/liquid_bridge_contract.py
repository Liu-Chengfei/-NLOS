"""液体桥接层控制契约模块。

文件职责：
  把模型中间量转换成测量控制信息。
  核心目标是把"模型内部值"变成"下游可以直接消费的控制指令"。
  只负责规范化、裁剪和安全门控，不训练模型，不改原始观测。

本文件绝对不负责：
  不训练模型。
  不修改原始观测数据。
  不直接执行融合或定位更新。

核心数据流：
    ModelIntermediate → [安全模式收敛] → build_measurement_control → MeasurementControl

安全模式语义：
    安全模式默认关闭，只有显式启用时才允许对中间量执行"收敛"。
    启用后，当风险低于阈值时，触发"收敛"——
    即把偏置和缩放按风险比例压缩，让中间量更保守。
    高风险时不收敛：非正常场景下对 bias 做部分衰减（缩放保持原值），
    正常场景下保持原值，由下游门控（apply_safe_mode）决定是否跳过更新。
    正常场景不覆盖风险阈值判断——即使场景正常，高风险也不收敛。

上游依赖：
  liquidloc.common.constants（DEFAULT_THRESHOLDS 全局阈值）、
  liquidloc.common.types（MeasurementControl、ModelIntermediate 类型）、
  liquidloc.common.validation（require_in_range、require_not_none 校验工具）、
  liquidloc.protocol.scene_schema（SceneSpec、decode_scene 场景编解码）、
  numpy（数值有限性判断和裁剪）

下游调用者：
  fusion/（融合层消费 MeasurementControl）、
  estimators/（估计器使用门控动作决定是否更新）、
  pipelines/（流水线中调用 build_measurement_control）

输入对象定义：
  - event          事件字典，包含 modality、payload 和 meta
  - intermediate   ModelIntermediate 对象，包含 bias、risk、uwb_scaling、vio_scaling
  - safe_mode_cfg  安全模式配置字典（可选）

输出对象定义：
  - LiquidBridgeDecision              桥接层决策记录对象
  - normalize_risk                    风险值归一化到 [0,1]
  - apply_safe_mode                   安全模式动作判断
  - build_measurement_control         合成测量控制对象
  - adjust_intermediate_for_safe_mode 安全模式中间量修正

核心变量定义：
  - _UWB_HARD_SKIP_QUALITY_FLOOR      UWB 质量硬跳过门槛（0.10）
  - _VIO_HARD_SKIP_QUALITY_FLOOR      VIO 质量硬跳过门槛（0.12）
  - _QUALITY_FLOOR_EPSILON            质量比较浮点容差（1e-9）
  - _UWB_NOISE_MULTIPLIER_CEILING     UWB 噪声倍数硬上限（5000.0）；不等同于 teacher-free 标签已默认裁剪 s_{u,max}=20
  - _VIO_NOISE_MULTIPLIER_CEILING     VIO 噪声倍数硬上限（5000.0）；不等同于 teacher-free 标签已默认裁剪 s_{v,max}=20
  - _SCENE_AXIS_FIELDS                场景轴字段顺序元组

关键设计决策：
  - UWB 在无效/低质/高风险时直接跳过更新，不做任何修正。
  - VIO 在低质/高风险时直接跳过更新。
  - 缩放值 < BRIDGE_THRESHOLDS["scaling_min"]（v3：1.0）会被裁剪到 scaling_min 并发出警告，v3 已回退到 1.0，不再依赖 apply_liquid_modality_output_contract 的 soft-mask。
  - 安全模式收敛只影响 bias 和 scaling，不改变 risk 本身。
  - 场景上下文支持多种输入格式（SceneSpec、场景编码字符串、字典），统一解析后再判断。
"""

from __future__ import annotations  # 允许在类型注解中引用尚未定义的类名。

from dataclasses import asdict  # 将 dataclass 实例转成普通字典。
from dataclasses import dataclass  # 提供数据类装饰器，简化只存数据的对象定义。
from copy import deepcopy  # 深拷贝，用于隔离嵌套结构。
from functools import lru_cache  # 给场景编码解析结果加缓存，减少重复解析开销。
from typing import Any  # 允许类型注解里表示"任意类型"。
import math  # 用于 isfinite 检查。

import numpy as np  # 使用 numpy 判断数值是否有限，并做简单的数值裁剪。

from liquidloc.common.constants import ASYNC_GAP_FULL_SCALE_S  # 异步满量程共享常量
from liquidloc.common.constants import MODALITY_UWB, MODALITY_VIO  # 模态名常量（单源真相，禁止字面量漂移）
from liquidloc.common.constants import DEFAULT_THRESHOLDS  # 读取全局阈值，统一质量等纯算法/协议级范围。
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS  # 读取桥接层业务阈值，统一风险和缩放的合法范围。
from liquidloc.common.constants import QUALITY_FLOOR_EPSILON  # 质量门槛比较浮点容差共享常量。
from liquidloc.common.constants import (  # 推理侧风险信号共享常量。
    UWB_INVALID_SIGNAL_FLOOR,
    VIO_LOW_FEATURES_SIGNAL_FLOOR,
    VIO_HIGH_REPROJ_ERR_SIGNAL_FLOOR,
    VIO_HIGH_REPROJ_ERR_THRESHOLD,
    VIO_TRACKED_FEATURES_FLOOR,
    VIO_TRACKED_FEATURES_SAFE_FLOOR,
    VIO_REPROJ_ERR_NORM_FLOOR,
    VIO_REPROJ_ERR_FULL_SCALE,
    RISK_PARTIAL_DAMPING_COEFF,
)
from liquidloc.common.types import MeasurementControl  # 桥接层最终要输出的控制对象类型。
from liquidloc.common.types import ModelIntermediate  # 模型内部的中间量结构，用来承载 bias、risk 等字段。
from liquidloc.common.validation import is_bool_like  # 检查值是否为布尔类型（含 numpy.bool_）。
from liquidloc.common.validation import is_numeric  # 检查值是否为数值类型（排除 bool 和 numpy.bool_）。
from liquidloc.common.validation import is_real  # 检查值是否为实数类型（排除 bool 和 numpy.bool_）。
from liquidloc.common.validation import is_string_like  # 检查值是否为字符串类型（含 numpy.str_）。
from liquidloc.common.validation import quality_below_floor  # 判断质量是否低于门槛（共享入口）。
from liquidloc.common.validation import require_in_range  # 检查数值是否在允许范围内。
from liquidloc.common.validation import require_not_none  # 检查关键输入不能为空。
from liquidloc.protocol.task_contract import get_vio_update_contract  # 读取冻结 VIO 控制合同，防止桥接消费语义漂移。
from liquidloc.protocol.scene_schema import SceneSpec  # 读取场景规格对象，便于按场景做安全判断。
from liquidloc.protocol.scene_schema import axis_levels  # 读取当前协议允许的场景轴层级（公共接口）。
from liquidloc.protocol.scene_schema import coerce_axis_value  # 校验+返回单个场景轴值是否合法（公共接口）。
from liquidloc.protocol.scene_schema import decode_scene  # 把场景编码字符串解码成 SceneSpec。
from liquidloc.protocol.scene_axis_protocol import load_scene_axis_protocol, get_nominal_levels  # 读取冻结场景轴协议，用于轴级退化风险下界；获取正常等级名，避免硬编码。

_UWB_HARD_SKIP_QUALITY_FLOOR = float(BRIDGE_THRESHOLDS["uwb_hard_skip_quality_floor"])  # UWB 质量低于这个值时，直接跳过更新。
_VIO_HARD_SKIP_QUALITY_FLOOR = float(BRIDGE_THRESHOLDS["vio_hard_skip_quality_floor"])  # VIO 质量低于这个值时，直接跳过更新。
_QUALITY_FLOOR_EPSILON = QUALITY_FLOOR_EPSILON  # 质量比较时使用的浮点容差，来自 constants。
_UWB_NOISE_MULTIPLIER_CEILING = float(BRIDGE_THRESHOLDS["uwb_noise_multiplier_ceiling"])  # UWB 噪声倍数硬上限；不等同于 teacher-free 标签已默认裁剪 s_{u,max}=20。D11-R2 修复后 ceiling=5000.0 = scaling_max² × (1+risk_max) = 50²×2，scaling_max=50 在 risk>0 时不再被截断为死代码；s_max=20 对应 20²×2=800 远低于 ceiling，正常训练不会触及上限。
_VIO_NOISE_MULTIPLIER_CEILING = float(BRIDGE_THRESHOLDS["vio_noise_multiplier_ceiling"])  # VIO 噪声倍数硬上限；同 UWB，ceiling=5000.0 与 scaling_max=50 的关系一致。
_RISK_HARD_SKIP_THRESHOLD = float(BRIDGE_THRESHOLDS["risk_hard_skip_threshold"])  # 风险高于此值时直接跳过更新，不做任何修正。
_SCALING_MAX = float(BRIDGE_THRESHOLDS["scaling_max"])  # 缩放值上限，与 inference.py 一致。

UWB_BIAS_MAX_RATIO = float(BRIDGE_THRESHOLDS["uwb_bias_max_ratio"])  # UWB 偏置修正量占原始测距的最大比例，防止过度修正。
_UWB_BIAS_MAX_RATIO = UWB_BIAS_MAX_RATIO  # 向后兼容别名。
_UWB_BIAS_ABSOLUTE_MAX = float(BRIDGE_THRESHOLDS["uwb_bias_absolute_max"])  # UWB 偏置绝对值上限（米），与文档 b_max=5.0m 对齐（v2 放宽至 5.0m）。
_SCENE_AXIS_FIELDS = tuple(SceneSpec.__dataclass_fields__.keys())  # 从 SceneSpec 动态提取场景轴字段顺序。
_VIO_LEARNED_CONTROL_ENTRY = str(get_vio_update_contract()["learned_control_entry"])  # 冻结合同规定的 VIO 学习控制落点。
if _VIO_LEARNED_CONTROL_ENTRY != "noise_multiplier":  # 当前 bridge 真实消费语义固定为最终噪声倍数。
    raise RuntimeError(
        "task contract drift: vio_update_contract.learned_control_entry must be 'noise_multiplier' "
        f"for the current bridge semantics, got {_VIO_LEARNED_CONTROL_ENTRY!r}"
    )

# 从冻结场景轴协议读取基准 K 值，避免硬编码。
# K 轴的正常等级（按五轴档位协议为 K0）即为基准档位，锚数全档固定 4。
# 使用惰性初始化，避免模块导入时重复加载协议（_load_scene_axis_protocol_cached 已有缓存）。
_DEFAULT_K_VALUE: str | None = None


def _get_default_k_value() -> str:
    """惰性获取协议基准 K 值，直接从 get_nominal_levels 获取。"""
    global _DEFAULT_K_VALUE
    if _DEFAULT_K_VALUE is None:
        _DEFAULT_K_VALUE = get_nominal_levels()["K"]
    return _DEFAULT_K_VALUE


@dataclass(slots=True)  # 让这个类只做数据承载，并启用 slots 降低属性开销。
class LiquidBridgeDecision:
    """桥接层一次决策的完整记录。"""

    modality: str  # 当前决策对应的模态名称。
    bias_applied: float = 0.0  # 实际应用到该模态上的 bias 修正值。
    scaling: float = 1.0  # 实际采用的缩放值。
    risk: float = 0.0  # 经过归一化后的风险值。
    noise_multiplier: float = 1.0  # 最终噪声倍数。
    gate_action: str = "pass_through"  # 最终门控动作名。

    def __post_init__(self) -> None:
        """校验所有字段的类型和值域。"""
        if not isinstance(self.modality, str):
            raise TypeError(f"modality must be str, got {type(self.modality).__name__}")
        if is_bool_like(self.bias_applied):
            raise TypeError(f"bias_applied must be numeric, got bool")
        if not is_numeric(self.bias_applied):
            raise TypeError(f"bias_applied must be numeric, got {type(self.bias_applied).__name__}")
        self.bias_applied = float(self.bias_applied)
        if is_bool_like(self.scaling):
            raise TypeError(f"scaling must be numeric, got bool")
        if not is_numeric(self.scaling):
            raise TypeError(f"scaling must be numeric, got {type(self.scaling).__name__}")
        self.scaling = float(self.scaling)
        if is_bool_like(self.risk):
            raise TypeError(f"risk must be numeric, got bool")
        if not is_numeric(self.risk):
            raise TypeError(f"risk must be numeric, got {type(self.risk).__name__}")
        self.risk = float(self.risk)
        if not math.isfinite(self.risk):
            raise ValueError("risk must be finite")
        if is_bool_like(self.noise_multiplier):
            raise TypeError(f"noise_multiplier must be numeric, got bool")
        if not is_numeric(self.noise_multiplier):
            raise TypeError(f"noise_multiplier must be numeric, got {type(self.noise_multiplier).__name__}")
        self.noise_multiplier = float(self.noise_multiplier)
        if not isinstance(self.gate_action, str):
            raise TypeError(f"gate_action must be str, got {type(self.gate_action).__name__}")

    def to_dict(self) -> dict[str, Any]:
        """把对象展开成普通字典。"""

        return asdict(self)  # 直接展开 dataclass，便于日志和序列化。

    def complete(self, required_fields: tuple[str, ...] | None = None) -> list[str]:
        """检查此决策是否包含所有必需字段，返回缺失字段列表。

        Stage 2 (桥接轴合同) 定义：LiquidBridgeDecision 必须包含
        modality / gate_action / noise_multiplier / risk 四个核心字段，
        以及 bias_applied / scaling 两个可由默认值填充的字段。

        调用方使用方式:
        - decision.complete() → 返回缺失字段列表，空列表表示完整。
        - decision.complete(('modality', 'gate_action')) → 只检查指定字段。

        返回:
        list[str]: 缺失的必需字段名。空列表表示决策完整（所有必需字段均已设置）。
        """
        if required_fields is not None:
            missing: list[str] = []
            for field in required_fields:
                if not hasattr(self, field):
                    missing.append(field)
                    continue
                value = getattr(self, field)
                if value is None:
                    missing.append(field)
                elif isinstance(value, str) and not value.strip():
                    missing.append(field)
            return missing

        # 核心合同字段（Stage 2 定义）。
        core_required = ('modality', 'gate_action', 'noise_multiplier', 'risk')
        default_ok = ('bias_applied', 'scaling')  # 有默认值的字段，0.0/1.0 即为有效。
        missing: list[str] = []
        for field in core_required:
            value = getattr(self, field)
            if value is None:
                missing.append(field)
            elif isinstance(value, str) and not value.strip():
                missing.append(field)
        for field in default_ok:
            value = getattr(self, field)
            if value is None:
                missing.append(field)
        return missing



def normalize_risk(value: float) -> float:
    """把风险值裁剪到 ``[0, 1]``。

    参数：
        value: 原始风险值。

    返回：
        裁剪后的风险值，保证在 ``[0, 1]`` 闭区间内。

    异常：
        TypeError: 当输入为布尔值时抛出。
        ValueError: 当输入为 NaN 或 inf 时抛出。
    """

    if is_bool_like(value):  # bool 不应隐式转为 0/1 后通过风险检查。
        raise TypeError("risk must be numeric, got bool")
    risk = float(value)  # 先把输入转成浮点数，统一后续比较口径。
    if not math.isfinite(risk):  # 只允许有限数，NaN 和 inf 都要拒绝。
        raise ValueError("risk must be finite")  # 异常值直接报错，避免污染门控逻辑。
    return min(1.0, max(0.0, risk))  # 把风险限制在闭区间 [0, 1]。


def _quality_below_floor(quality: float, floor: float) -> bool:
    """判断质量是否低于门槛。委托到 common.validation.quality_below_floor 共享入口。"""

    return quality_below_floor(quality, floor, epsilon=_QUALITY_FLOOR_EPSILON)


def _resolve_uwb_valid_flag(value: Any) -> bool:
    """Only bool-like False marks UWB as invalid."""

    if is_bool_like(value):
        return bool(value)
    return True


def apply_safe_mode(*, modality: str, valid: bool, quality: float, risk: float) -> str:
    """根据模态、质量和风险决定安全动作。

    参数：
        modality: 模态名称，支持 "uwb"、"vio" 及其他。
        valid: 测量是否有效。对 UWB 特别关键，无效时直接跳过更新。
        quality: 测量质量，范围 ``[0, 1]``。
        risk: 风险评分，范围 ``[0, 1]``。
    """

    if modality == MODALITY_UWB and (not valid or _quality_below_floor(quality, _UWB_HARD_SKIP_QUALITY_FLOOR) or risk > _RISK_HARD_SKIP_THRESHOLD):  # UWB 在无效、低质或高风险时直接跳过。
        return "uwb_skip_update"  # 告诉上层不要更新 UWB 分支。
    if modality == MODALITY_VIO and (_quality_below_floor(quality, _VIO_HARD_SKIP_QUALITY_FLOOR) or risk > _RISK_HARD_SKIP_THRESHOLD):  # VIO 在低质或高风险时直接跳过。
        return "vio_skip_update"  # 告诉上层不要更新 VIO 分支。
    default_actions = {  # 其余模态走默认动作映射表。
        "uwb": "uwb_bias_and_noise_scale",  # UWB 默认同时处理偏置和噪声缩放。
        "vio": "vio_confidence_scale",  # VIO 默认按置信度缩放处理。
    }  # 字典结束。
    return default_actions.get(modality, "pass_through")  # 未知模态就原样透传。


def _coerce_positive_scaling(value: float, *, name: str) -> float:
    """把缩放值规范化到 `[BRIDGE_THRESHOLDS["scaling_min"], scaling_max]`。

    只拒绝非法值（非有限数、零、负数），不静默裁剪合法缩放值。
    注意：此函数保证输出 >= BRIDGE_THRESHOLDS["scaling_min"]（v3 已回退到 1.0，
    不再依赖 apply_liquid_modality_output_contract 的 soft-mask 释放调节空间；
    v2 的 0.5 soft-mask 在 e9 场景下让 Liquid 不当降权好测量，不利于高 NLOS + 异步场景）。
    上游如果需要进一步表达"降低权重"，应通过 bias 或 gate_action 实现，
    而不是绕过 scaling_min 的下界。
    同时保证输出 <= BRIDGE_THRESHOLDS["scaling_max"]，与 inference.py
    的值域约束一致。
    """

    if is_bool_like(value):  # bool 不应隐式转为 0/1 后通过缩放检查。
        raise TypeError(f"{name} must be numeric, got bool")
    scaling = float(value)  # 统一转成浮点数。
    if not math.isfinite(scaling) or scaling <= 0.0:  # 只接受有限正数。
        raise ValueError(f"{name} must be finite and > 0")  # 非法值直接拒绝。
    scaling_min_floor = float(BRIDGE_THRESHOLDS["scaling_min"])  # 单源下界（v3：1.0）。
    if scaling < scaling_min_floor:  # 缩放低于 scaling_min 时裁剪并发出警告（不是静默吞掉）。
        import warnings  # 延迟导入，避免在正常路径增加开销。
        warnings.warn(  # 发出 UserWarning，提醒上游检查输入。
            f"{name}={scaling} < scaling_min={scaling_min_floor} was clipped to scaling_min; "
            "use bias or gate_action to reduce weight instead",
            stacklevel=3,  # 让警告指向调用 _coerce_positive_scaling 的上层。
        )  # 警告结束。
        return scaling_min_floor  # 裁剪到 scaling_min，保持保护效果不被进一步削弱。
    if scaling > _SCALING_MAX:  # 超过上限时裁剪并发出警告，与低值处理对称。
        import warnings
        warnings.warn(
            f"{name}={scaling} > scaling_max={_SCALING_MAX} was clipped to scaling_max",
            stacklevel=3,
        )
        return _SCALING_MAX
    return scaling  # 合法值原样返回。


def _resolve_quality(value: Any) -> float:
    """把输入质量值折算到 `[0, 1]`。"""

    if is_bool_like(value):  # bool 语义与连续质量分数不兼容，不能静默当成 0/1。
        raise TypeError("quality must be numeric, got bool")
    quality = float(value)  # 统一转成浮点数，减少上游类型分歧。
    if not math.isfinite(quality):  # 只接受有限值。
        raise ValueError("quality must be finite")  # 无效质量直接报错。
    return min(1.0, max(0.0, quality))  # 质量同样裁剪到 [0, 1]。


def _compose_noise_multiplier(*, scaling: float, risk: float, ceiling: float) -> float:
    """合成噪声倍数。

    scaling 是测量噪声标准差的倍数（>= 1.0），协方差（方差）需要乘以 scaling^2。
    risk 是归一化风险值 [0, 1]，对协方差做额外膨胀 (1 + risk)。
    最终 noise_multiplier = scaling^2 * (1 + risk)，受 ceiling 封顶。
    """
    if not math.isfinite(ceiling) or ceiling <= 0.0:
        raise ValueError(f"ceiling must be finite and > 0, got {ceiling}")
    _scaling = float(scaling)
    if not math.isfinite(_scaling) or _scaling <= 0.0:
        raise ValueError(f"scaling must be finite and > 0, got {scaling}")
    variance_multiplier = _scaling ** 2  # 标准差倍数平方为方差倍数。scaling 应已由上游保证 >= 1.0。
    risk_multiplier = 1.0 + normalize_risk(risk)  # 风险越高，噪声倍数越大。
    return min(float(ceiling), variance_multiplier * risk_multiplier)  # 最后按上限封顶。


def _resolve_modality_signal(event: dict[str, Any]) -> float:
    """从事件内容里推断额外风险信号。

    前置条件：调用方须保证事件已通过 ``validate_event`` 校验，
    否则缺失/非法 payload 可能导致静默错误。
    """

    modality = event.get("modality")  # 安全读取模态名称。
    if modality is None:
        raise KeyError("event must contain 'modality' key")
    if not isinstance(modality, str):
        raise TypeError(f"event['modality'] must be str, got {type(modality).__name__}")
    if modality == MODALITY_UWB:  # UWB 单独走一条风险分支。
        uwb_payload = event.get("uwb_payload") or {}  # 没有 payload 时当作空字典处理。
        # 与 build_measurement_control L350 保持一致的 valid 语义：
        # 仅当 valid 严格为 False 时才视为无效，None/0/"false" 均当有效处理。
        uwb_valid = uwb_payload.get("valid", True)  # 读取 valid，默认为 True。
        if not _resolve_uwb_valid_flag(uwb_valid):  # 无效 UWB 直接抬高风险。
            return UWB_INVALID_SIGNAL_FLOOR  # 返回固定附加风险。
        return 0.0  # 有效 UWB 不额外加风险。
    if modality == MODALITY_VIO:  # VIO 也单独评估。
        vio_payload = event.get("vio_payload") or {}  # 同样容忍缺失 payload。
        signal = 0.0  # 从零开始累积风险信号。
        tracked_features = vio_payload.get("tracked_features")  # 读取跟踪特征数量。
        # 与训练侧 _resolve_modality_observation_signal 保持一致：仅当 tracked_features
        # 严格低于 30 时才抬高额外风险；30 本身属于边界安全值，不应在推理侧单独加罚。
        if tracked_features is not None and float(tracked_features) < VIO_TRACKED_FEATURES_FLOOR:  # 特征太少说明跟踪不稳。
            signal = max(signal, VIO_LOW_FEATURES_SIGNAL_FLOOR)  # 把风险抬到至少 VIO_LOW_FEATURES_SIGNAL_FLOOR。
        reproj_err = vio_payload.get("reproj_err")  # 读取重投影误差。
        if reproj_err is not None and float(reproj_err) >= VIO_HIGH_REPROJ_ERR_THRESHOLD:  # 误差过大说明视觉质量差。
            signal = max(signal, VIO_HIGH_REPROJ_ERR_SIGNAL_FLOOR)  # 再补一个风险下限。
        return signal  # 返回累计后的风险值。
    return 0.0  # 其他模态默认不额外加风险。


def _resolve_effective_risk(event: dict[str, Any], *, base_risk: float, quality: float) -> float:
    """综合基础风险、质量风险和模态风险。

    参数 quality 应为已 resolved 的 [0, 1] 值（由调用方通过 _resolve_quality 预处理）。
    """

    if not math.isfinite(base_risk):
        raise ValueError(f"base_risk must be finite, got {base_risk}")
    quality_risk = 1.0 - max(0.0, min(1.0, float(quality)))  # 质量越低，折算出来的风险越高。quality 应已由上游裁剪到 [0,1]。
    modality_signal = _resolve_modality_signal(event)  # 从事件本身再提一个附加风险。
    axis_floor = _resolve_scene_axis_observation_floor(event)  # 再叠加来自场景轴的退化下界。
    return min(1.0, max(float(base_risk), quality_risk, modality_signal, axis_floor))  # 取最保守的结果。


def _scene_spec_from_axis_fields(scene_context: dict[str, Any]) -> SceneSpec:
    """从轴字段字典拼出 `SceneSpec`。

    警告：此函数直接构造 SceneSpec，不经 coerce_axis_value 校验。
    调用方必须通过 _normalize_scene_spec 重新校验后方可使用。
    """

    missing_axis_fields = [field for field in _SCENE_AXIS_FIELDS if field not in scene_context]  # 找出缺失的轴字段。
    if missing_axis_fields:  # 只要缺字段就不能继续拼。
        raise KeyError(  # 直接抛出缺项错误。
            f"scene_context has partial axis fields; "  # 错误信息第一段：说明是部分轴字段缺失。
            f"provide all axis fields or use scene_id/scene_code instead; "  # 错误信息第二段：给出两种修复路径。
            f"missing {missing_axis_fields}"  # 错误信息第三段，列出缺失字段。
        )  # KeyError 结束。
    for field in _SCENE_AXIS_FIELDS:  # 校验每个轴字段的值类型。
        val = scene_context[field]
        if not is_string_like(val):
            raise TypeError(f"scene_context[{field!r}] must be str, got {type(val).__name__}")
    # 审查修复：显式拒绝非 SceneSpec 字段，避免静默丢失审计信息。
    # 已知场景编码字段不参与 SceneSpec 构造，但不应导致轴字段路径失败（fallback 兼容）。
    _KNOWN_SCENE_CODE_FIELDS = {"scene_id", "scene_code", "seq_id", "seed", "meta"}
    extra_fields = [key for key in scene_context if key not in _SCENE_AXIS_FIELDS and key not in _KNOWN_SCENE_CODE_FIELDS]
    if extra_fields:
        raise KeyError(
            f"scene_context contains non-SceneSpec fields: {extra_fields}; "
            f"SceneSpec only accepts {sorted(_SCENE_AXIS_FIELDS)}; "
            f"remove extra fields or use scene_id/scene_code path."
        )
    return SceneSpec(**{field: scene_context[field] for field in _SCENE_AXIS_FIELDS})  # 动态构造 SceneSpec，与 _SCENE_AXIS_FIELDS 自动对齐。


def _normalize_scene_spec(scene_spec: SceneSpec) -> SceneSpec:
    """按当前协议重新校验场景规格。"""

    _levels = axis_levels()  # 先读取当前协议允许的层级集合。
    return SceneSpec(  # 返回一个重新校验后的场景对象。
        A_level=coerce_axis_value("A", scene_spec.A_level, _levels),  # 校验 A 轴。
        N_level=coerce_axis_value("N", scene_spec.N_level, _levels),  # 校验 N 轴。
        V_level=coerce_axis_value("V", scene_spec.V_level, _levels),  # 校验 V 轴。
        M_level=coerce_axis_value("M", scene_spec.M_level, _levels),  # 校验 M 轴（2026-08-31：G 轴并入 K，M 补位）。
        K_value=coerce_axis_value("K", scene_spec.K_value, _levels),  # 校验 K 值。
    )  # SceneSpec 返回结束。


@lru_cache(maxsize=256)  # 给常见场景编码结果做缓存，避免反复解析。
def _decode_scene_cached(scene_code: str) -> SceneSpec:
    """缓存场景编码解析结果。"""

    return decode_scene(scene_code)  # 直接复用解码器结果。


@lru_cache(maxsize=1)
def _load_scene_axis_protocol_cached() -> dict[str, Any]:
    """缓存场景轴协议，避免桥接层每次构造控制量都重复读配置。

    返回冻结快照的深拷贝，防止调用者篡改 lru_cache 缓存的内部对象。
    load_scene_axis_protocol() 默认路径已返回深拷贝，但 lru_cache 会缓存
    其返回值，因此仍需 deepcopy 隔离缓存对象与调用者拿到的对象。
    """

    return deepcopy(load_scene_axis_protocol())


def _try_decode_scene_code(scene_code: Any) -> SceneSpec | None:
    """尝试把场景编码解析成 `SceneSpec`。"""

    if not is_string_like(scene_code):  # 不是字符串就不要继续尝试。
        return None  # 返回空，交给上层别的来源继续试。
    try:  # 走缓存版解码，避免重复计算。
        return _decode_scene_cached(scene_code)  # 成功就直接返回。
    except (ValueError, TypeError):  # 编码非法时吞掉异常。
        return None  # 失败用 None 表示未解析到。


def _resolve_safe_mode_scene(scene_context: SceneSpec | str | dict[str, Any]) -> SceneSpec:
    """把多种场景表示统一成 `SceneSpec`。"""

    if isinstance(scene_context, SceneSpec):  # 已经是对象时直接校验。
        return _normalize_scene_spec(scene_context)  # 返回校验后的场景对象。
    if is_string_like(scene_context):  # 字符串时按场景编码处理。
        return _normalize_scene_spec(_decode_scene_cached(str(scene_context)))  # 解码后也走校验，保证与 SceneSpec 路径一致。
    if not isinstance(scene_context, dict):  # 其他类型一律不接受。
        raise TypeError(  # 抛出类型错误。
            f"scene_context must be a SceneSpec, scene code str, or dict, got {type(scene_context).__name__}"  # 说明实际类型。
        )  # 异常参数结束。

    def _decode_first_available_scene(payload: dict[str, Any]) -> SceneSpec | None:
        for field_name in ("scene_id", "scene_code"):
            if field_name not in payload:
                continue
            parsed = _try_decode_scene_code(payload[field_name])
            if parsed is not None:
                return _normalize_scene_spec(parsed)
        return None

    axis_field_scene = None  # 先假设没有轴字段来源。
    if any(field in scene_context for field in _SCENE_AXIS_FIELDS):  # 如果存在任一轴字段。
        try:  # 尝试从轴字段拼接，但可能只有部分字段。
            axis_field_scene = _normalize_scene_spec(_scene_spec_from_axis_fields(scene_context))  # 就从轴字段拼接并校验。
        except KeyError:  # 轴字段不全，无法拼接 SceneSpec。
            axis_field_scene = None  # 置空，让后续 scene_code 等路径有机会兜底。
    parsed_scene = _decode_first_available_scene(scene_context)
    if parsed_scene is None and "meta" in scene_context and isinstance(scene_context["meta"], dict):  # 再看 meta 内的场景编码来源。
        meta = scene_context["meta"]  # 只读取一次，避免重复索引。
        parsed_scene = _decode_first_available_scene(meta)
    if parsed_scene is None:  # 都没有时退回轴字段来源。
        parsed_scene = axis_field_scene  # 采用轴字段结果。
    if parsed_scene is None:  # 三条路都失败。
        raise KeyError("scene_context must provide scene_id, scene_code, meta.scene_id, or axis fields")  # 直接报缺失。
    if axis_field_scene is not None and parsed_scene != axis_field_scene:  # 两种来源同时存在且不一致。
        raise ValueError("scene_context scene schema inputs disagree between scene_id/scene_code and axis fields")  # 拒绝冲突输入。
    return parsed_scene  # 返回最终统一场景。


def _resolve_safe_mode_context_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """从事件里提取可用于安全模式的场景上下文。"""

    meta = event.get("meta")  # 先看事件里的 meta。
    if isinstance(meta, dict) and (  # meta 必须是字典，且里面要有可用场景信息。
        any(field in meta for field in _SCENE_AXIS_FIELDS)  # meta 里有轴字段。
        or _try_decode_scene_code(meta.get("scene_id")) is not None  # meta.scene_id 能解析。
        or _try_decode_scene_code(meta.get("scene_code")) is not None  # meta.scene_code 能解析。
    ):  # 条件结束。
        return meta  # 优先把 meta 当上下文返回。
    if (  # 否则看事件本体。
        any(field in event for field in _SCENE_AXIS_FIELDS)  # 事件本体有轴字段。
        or _try_decode_scene_code(event.get("scene_id")) is not None  # 顶层 scene_id 能解析。
        or _try_decode_scene_code(event.get("scene_code")) is not None  # 顶层 scene_code 能解析。
    ):  # 条件结束。
        return event  # 事件本体也能作为上下文。
    return None  # 两处都没有就返回空。


def _resolve_scene_axis_observation_floor(event: dict[str, Any]) -> float:
    """根据事件所属的 A/N/V 轴级场景，给默认噪声链提供最小退化下界。"""

    scene_context = _resolve_safe_mode_context_from_event(event)
    if scene_context is None:
        return 0.0
    try:
        scene_spec = _resolve_safe_mode_scene(scene_context)
    except (KeyError, TypeError, ValueError) as exc:
        import warnings
        warnings.warn(f"_resolve_scene_axis_observation_floor: failed to resolve scene: {exc}", stacklevel=2)
        return 0.0

    protocol_cfg = _load_scene_axis_protocol_cached()
    axes_cfg = dict(protocol_cfg.get("axes") or {})
    async_cfg = dict((axes_cfg.get("A") or {}).get(scene_spec.A_level) or {})
    nlos_cfg = dict((axes_cfg.get("N") or {}).get(scene_spec.N_level) or {})
    visual_cfg = dict((axes_cfg.get("V") or {}).get(scene_spec.V_level) or {})

    async_axis_risk = 0.0
    cross_modal_skew_ms = async_cfg.get("cross_modal_skew_ms")
    if cross_modal_skew_ms is not None:
        _skew_val = float(cross_modal_skew_ms)
        if not math.isfinite(_skew_val):
            raise ValueError(f"cross_modal_skew_ms must be finite, got {cross_modal_skew_ms}")
        async_axis_risk = min(1.0, max(0.0, _skew_val / (ASYNC_GAP_FULL_SCALE_S * 1000.0)))

    modality = event.get("modality")
    if modality == MODALITY_UWB:
        nlos_ratio = nlos_cfg.get("nlos_ratio")
        if nlos_ratio is None:
            nlos_axis_risk = 0.0
        else:
            # 区间化 2026-08-31：nlos_ratio 支持 [low, high] 区间。
            # 风险下界取区间上界（worst-case 保守）以保证安全模式门槛不被低估。
            if isinstance(nlos_ratio, (list, tuple)) and len(nlos_ratio) == 2:
                _nlos_val = float(nlos_ratio[1])
            else:
                _nlos_val = float(nlos_ratio)
            if not math.isfinite(_nlos_val):
                raise ValueError(f"nlos_ratio must be finite, got {nlos_ratio}")
            nlos_axis_risk = min(1.0, max(0.0, _nlos_val))
        return max(async_axis_risk, nlos_axis_risk)
    if modality == MODALITY_VIO:
        tracked_features_range = visual_cfg.get("tracked_features_range")
        reproj_err_max = visual_cfg.get("reproj_err_max")
        # interval 处理：reproj_err_max ∈ [low, high] 时取上界（worst-case reproj err）。
        # tracked_features_range ∈ [low, high] 时取下界（worst-case feature count，已于 L585 raw 取 [0]）。
        if isinstance(reproj_err_max, (list, tuple)) and len(reproj_err_max) == 2:
            reproj_err_max = float(reproj_err_max[1])
        feature_floor_risk = 0.0
        reproj_floor_risk = 0.0
        if isinstance(tracked_features_range, (list, tuple)) and len(tracked_features_range) == 2:
            # 用下界（tracked_features_range[0]）计算特征风险下界：
            # 下界越低说明该等级下特征数可能越少，风险下界越高。
            # 以 VIO_TRACKED_FEATURES_SAFE_FLOOR（V0 下界 100）为零风险参考点，
            # 特征数达到此值时 VIO 工作正常，风险为 0；低于此值风险线性增加。
            feature_lower = max(0.0, float(tracked_features_range[0]))
            feature_floor_risk = min(1.0, max(0.0, (VIO_TRACKED_FEATURES_SAFE_FLOOR - min(VIO_TRACKED_FEATURES_SAFE_FLOOR, feature_lower)) / VIO_TRACKED_FEATURES_SAFE_FLOOR))
        if reproj_err_max is not None:
            _reproj_val = float(reproj_err_max)
            if not math.isfinite(_reproj_val):
                raise ValueError(f"reproj_err_max must be finite, got {reproj_err_max}")
            _denom = VIO_REPROJ_ERR_FULL_SCALE - VIO_REPROJ_ERR_NORM_FLOOR
            if _denom <= 0.0:
                raise ValueError(
                    f"VIO_REPROJ_ERR_FULL_SCALE({VIO_REPROJ_ERR_FULL_SCALE}) must be > "
                    f"VIO_REPROJ_ERR_NORM_FLOOR({VIO_REPROJ_ERR_NORM_FLOOR})"
                )
            reproj_floor_risk = min(1.0, max(0.0, (_reproj_val - VIO_REPROJ_ERR_NORM_FLOOR) / _denom))
        return max(async_axis_risk, feature_floor_risk, reproj_floor_risk)
    return 0.0


def _coerce_safe_mode_enabled_flag(value: Any) -> bool:
    """严格解析 safe_mode.enabled，拒绝 truthiness 漂移。"""

    if value is None:
        return False
    if is_bool_like(value):
        return bool(value)  # 显式转为 Python bool，避免 numpy.bool_ 泄漏。
    raise TypeError("safe_mode_cfg.enabled must be a boolean")


_DEFAULT_VALUE_TOLERANCE = 1e-10  # 中性判断容差，与 QUALITY_FLOOR_EPSILON 同量级但用途不同。


def _is_neutral_intermediate(intermediate: ModelIntermediate) -> bool:
    """判断中间量是否还是默认中性状态。

    动态检查 ModelIntermediate 的所有带默认值的数值字段，
    当所有字段都回到默认值时才算中性。这样当 ModelIntermediate
    新增字段时，此函数无需手动同步更新。
    """

    import dataclasses  # 延迟导入，避免在正常路径增加开销。
    for f in dataclasses.fields(intermediate):
        default_val = f.default
        if default_val is dataclasses.MISSING:
            if f.default_factory is not dataclasses.MISSING:  # 有 default_factory 的字段也要检查。
                default_val = f.default_factory()
            else:
                continue  # 无默认值的字段跳过。
        current_val = getattr(intermediate, f.name)
        if is_real(default_val) and is_real(current_val):
            if not math.isfinite(float(current_val)) or not math.isfinite(float(default_val)):
                return False
            if abs(float(current_val) - float(default_val)) >= _DEFAULT_VALUE_TOLERANCE:
                return False
        elif current_val != default_val:
            return False
    return True


def adjust_intermediate_for_safe_mode(
    intermediate_outputs: ModelIntermediate | dict[str, Any],  # 输入的模型中间量，可以是对象也可以是字典。
    scene_context: SceneSpec | str | dict[str, Any],  # 用来判断是否进入安全模式的场景上下文。
    safe_mode_cfg: dict[str, Any] | None = None,  # 安全模式的可选配置，没传就用默认值。
) -> tuple[ModelIntermediate, bool]:  # 返回修正后的中间量，以及是否触发收敛。
    """按安全模式规则修正中间量。

    收敛条件（converge_flag 为 True）：
        - enabled=True 且 risk <= risk_threshold
        - 收敛时：bias *= risk, scaling 向 1.0 按 risk 比例收敛
        - 不收敛时：非正常场景对 bias 做部分衰减，正常场景保持原值

    参数：
        intermediate_outputs: 模型中间量，支持 ``ModelIntermediate`` 对象或字典。
        scene_context: 场景上下文，支持 ``SceneSpec``、场景编码字符串或字典。
        safe_mode_cfg: 安全模式配置，默认 ``{"enabled": False, "risk_threshold": 0.5}``。

    返回：
        (修正后的中间量, 是否触发了收敛) 的二元组。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "safe_mode_cfg": safe_mode_cfg,
        "scene_context_type": type(scene_context).__name__,
    }, "adjust_intermediate_for_safe_mode 入口参数")

    require_not_none(intermediate_outputs, "intermediate_outputs")  # 中间量不能为空，空了后面没法算。
    require_not_none(scene_context, "scene_context")  # 场景上下文不能为空，因为要靠它决定收敛逻辑。
    if safe_mode_cfg is None:  # 如果用户没有传配置。
        safe_mode_cfg = {}  # 就用空字典兜底，后续统一按字典读。
    if not isinstance(safe_mode_cfg, dict):  # 配置必须是字典。
        raise TypeError(f"safe_mode_cfg must be a dict, got {type(safe_mode_cfg).__name__}")  # 类型不对就直接拒绝。
    if isinstance(intermediate_outputs, ModelIntermediate):  # 如果传进来的是对象形式。
        bias_value = float(intermediate_outputs.bias)  # 读取 bias，后面可能会缩放它。
        risk_value = float(intermediate_outputs.risk)  # 读取 risk，它本身只做约束不重算。
        uwb_scaling = float(intermediate_outputs.uwb_scaling)  # 读取 UWB 缩放，必要时会收敛。
        vio_scaling = float(intermediate_outputs.vio_scaling)  # 读取 VIO 缩放，必要时会收敛。
        output_kind = "model_intermediate"  # 记录原始输出形态，便于最后回填。
    elif isinstance(intermediate_outputs, dict):  # 如果传进来的是字典形式。
        required_keys = ("bias", "risk", "uwb_scaling", "vio_scaling")  # 定义必须存在的四个键。
        missing_keys = [key for key in required_keys if key not in intermediate_outputs]  # 找出缺失项。
        if missing_keys:  # 只要缺了任意一个键就不能继续。
            raise KeyError(f"intermediate_outputs is missing required keys: {missing_keys}")  # 直接报错，把缺项列出来。
        bias_value = float(intermediate_outputs["bias"])  # 读取 bias 数值。
        risk_value = float(intermediate_outputs["risk"])  # 读取 risk 数值。
        uwb_scaling = float(intermediate_outputs["uwb_scaling"])  # 读取 UWB 缩放数值。
        vio_scaling = float(intermediate_outputs["vio_scaling"])  # 读取 VIO 缩放数值。
        # dict 路径额外校验：排除 bool 隐式转 0/1 后通过数值检查。
        for _k, _v in [("bias", bias_value), ("risk", risk_value), ("uwb_scaling", uwb_scaling), ("vio_scaling", vio_scaling)]:
            if is_bool_like(intermediate_outputs[_k]):
                raise TypeError(f"intermediate_outputs[{_k!r}] must be numeric, got bool")
        output_kind = "dict"  # 记录原始输出是字典。
    else:  # 其他类型都不接受。
        raise TypeError(  # 抛出类型错误。
            "intermediate_outputs must be a ModelIntermediate or dict, "  # 第一段说明允许的类型。
            f"got {type(intermediate_outputs).__name__}"  # 第二段说明实际类型。
        )  # TypeError 参数结束。
    require_in_range(  # 校验 risk 是否在允许区间。
        risk_value,  # 待校验的 risk。
        "risk_value",  # 字段名。
        min_value=BRIDGE_THRESHOLDS["risk_min"],  # 风险下限。
        max_value=BRIDGE_THRESHOLDS["risk_max"],  # 风险上限。
    )  # risk 校验结束。
    require_in_range(uwb_scaling, "uwb_scaling", min_value=BRIDGE_THRESHOLDS["scaling_min"])  # 校验 UWB 缩放下限。
    require_in_range(vio_scaling, "vio_scaling", min_value=BRIDGE_THRESHOLDS["scaling_min"])  # 校验 VIO 缩放下限。
    parsed_scene = _resolve_safe_mode_scene(scene_context)  # 把场景上下文统一成 SceneSpec。
    enabled = _coerce_safe_mode_enabled_flag(safe_mode_cfg.get("enabled", False))  # 读取开关，拒绝 truthiness 漂移。
    risk_threshold = float(safe_mode_cfg.get("risk_threshold", 0.5))  # 读取风险阈值，默认 0.5（收敛阈值，与 risk_hard_skip_threshold 语义不同）。
    require_in_range(  # 校验阈值本身是否合法。
        risk_threshold,  # 待校验的阈值。
        "safe_mode_cfg.risk_threshold",  # 字段名。
        min_value=BRIDGE_THRESHOLDS["risk_min"],  # 下限。
        max_value=BRIDGE_THRESHOLDS["risk_max"],  # 上限。
    )  # 阈值校验结束。
    _nominal = get_nominal_levels()  # 从协议动态获取正常等级名，避免硬编码。
    is_normal_scene = (  # 判断是否属于默认正常场景。
        parsed_scene.A_level == _nominal["A"]  # A 轴是否正常。
        and parsed_scene.N_level == _nominal["N"]  # N 轴是否正常。
        and parsed_scene.V_level == _nominal["V"]  # V 轴是否正常。
        and parsed_scene.M_level == _nominal["M"]  # M 轴是否正常。
        and parsed_scene.K_value == _get_default_k_value()  # K 值是否为协议基准值。
    )  # 正常场景判断结束。
    converge_flag = enabled and (risk_value <= risk_threshold)  # 是否触发安全模式收敛：仅低风险时收敛，正常场景不覆盖风险阈值判断。
    if converge_flag:  # 进入收敛时，按风险比例压缩偏置和缩放。
        adjusted_bias = bias_value * risk_value  # bias 按风险缩小。
        adjusted_uwb_scaling = 1.0 + max(0.0, uwb_scaling - 1.0) * risk_value  # UWB 缩放按风险收敛。
        adjusted_vio_scaling = 1.0 + max(0.0, vio_scaling - 1.0) * risk_value  # VIO 缩放按风险收敛。
    elif enabled and not is_normal_scene:  # 高NLOS等非正常场景：安全模式已启用但不收敛，对 bias 做部分衰减防止极端值通过。
        # 非正常场景下模型更容易输出极端 bias，按 (1 - risk) 做部分衰减，
        # 风险越高衰减越多，但不像完全收敛那样按 risk 缩小，保留模型的方向性判断。
        partial_damping = 1.0 - RISK_PARTIAL_DAMPING_COEFF * risk_value  # 风险 0.5 时衰减 25%，风险 1.0 时衰减 50%。
        if partial_damping < 0.0:  # 防御性校验：COEFF 过大时 partial_damping 不应为负。
            partial_damping = 0.0
        adjusted_bias = bias_value * partial_damping  # bias 部分衰减。
        adjusted_uwb_scaling = uwb_scaling  # 缩放保持原值，由噪声倍数上限兜底。
        adjusted_vio_scaling = vio_scaling  # 缩放保持原值，由噪声倍数上限兜底。
    else:  # 不收敛时就保持原值。
        adjusted_bias = bias_value  # bias 原样保留。
        adjusted_uwb_scaling = uwb_scaling  # UWB 缩放原样保留。
        adjusted_vio_scaling = vio_scaling  # VIO 缩放原样保留。
    if output_kind == "model_intermediate":  # 如果原来是对象，就返回对象。
        adjusted_outputs = ModelIntermediate(  # 构造新的中间量对象。
            bias=adjusted_bias,  # 写入修正后的 bias。
            risk=risk_value,  # risk 不改，保持原值。
            uwb_scaling=adjusted_uwb_scaling,  # 写入修正后的 UWB 缩放。
            vio_scaling=adjusted_vio_scaling,  # 写入修正后的 VIO 缩放。
        )  # 对象构造结束。
    else:  # 原来是字典，也返回 ModelIntermediate 保持返回类型一致。
        adjusted_outputs = ModelIntermediate(  # 构造新的中间量对象。
            bias=adjusted_bias,  # 写入修正后的 bias。
            risk=risk_value,  # risk 不改，保持原值。
            uwb_scaling=adjusted_uwb_scaling,  # 写入修正后的 UWB 缩放。
            vio_scaling=adjusted_vio_scaling,  # 写入修正后的 VIO 缩放。
        )  # 对象构造结束。
    return adjusted_outputs, converge_flag  # 返回修正结果和是否触发收敛。


def clip_uwb_bias(bias_value: float, raw_range: float) -> float:
    """对 UWB 偏置修正量施加协议级截断，返回实际可施加的 bias。

    截断规则（与文档 b_max=5.0m 对齐，v2 放宽至 5.0m）：
    1. bias 强制非负（UWB 只允许施加非负 NLOS bias 修正）
    2. bias 不超过原始测距的指定比例（uwb_bias_max_ratio），防止过度修正
    3. 缺少参考距离（raw_range <= 0）时 bias 归零
    4. bias 不超过绝对值上限（uwb_bias_absolute_max）

    本函数是 bias 截断逻辑的唯一权威实现，fusion 层 apply_bias 和
    build_measurement_control 都应调用本函数，禁止复制截断逻辑。

    Args:
        bias_value: 模型输出的原始 bias 值。
        raw_range: UWB 原始测距值，作为比例截断的参考。

    Returns:
        截断后的 bias 值（非负 float）。
    """
    raw_bias = max(0.0, float(bias_value))  # UWB 只允许施加非负 NLOS bias 修正。
    if raw_range > 0.0 and raw_bias > raw_range * _UWB_BIAS_MAX_RATIO:  # 偏置超过比例上限时裁剪。
        raw_bias = raw_range * _UWB_BIAS_MAX_RATIO  # 裁剪到比例上限，保证 corrected_range > 0。
    elif raw_range <= 0.0 and raw_bias > 0.0:  # 缺少参考距离时无法安全施加偏置修正。
        raw_bias = 0.0  # 无参考距离时将 bias 归零，防止 corrected_range 变负。
    if raw_bias > _UWB_BIAS_ABSOLUTE_MAX:  # 绝对值截断，与文档 b_max=5.0m 对齐（v2 放宽至 5.0m）。
        raw_bias = _UWB_BIAS_ABSOLUTE_MAX
    return raw_bias


def build_measurement_control(
    event: dict[str, Any],
    intermediate: ModelIntermediate,
    safe_mode_cfg: dict[str, Any] | None = None,
) -> MeasurementControl:
    """把事件和中间量合成为测量控制对象。

    处理流程：
        1. 对 uwb/vio 模态且非中性中间量，先走安全模式收敛
        2. 对 risk 做归一化裁剪
        3. 按模态分支计算 scaling、bias_applied、quality、applied_risk
        4. 合成噪声倍数和门控动作
        5. 组装 ``MeasurementControl`` 返回

    参数：
        event: 事件字典，必须包含 ``"modality"`` 键。
            - UWB 事件应包含 ``"uwb_payload"`` 字典（含 ``"valid"`` 和 ``"quality"``）
            - VIO 事件应包含 ``"vio_payload"`` 字典（含 ``"quality"``）
            - 安全模式收敛需要事件包含 ``"meta"`` 字典（含 ``"scene_id"``）
        intermediate: 模型中间量对象。

    返回：
        ``MeasurementControl`` 实例，包含 bias_applied、scaling、risk、
        noise_multiplier 和 gate_action。

    前置条件：调用方须保证事件已通过 ``validate_event`` 校验。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "modality": event.get("modality") if isinstance(event, dict) else None,
        "safe_mode_cfg": safe_mode_cfg,
    }, "build_measurement_control 入口参数")

    modality = event["modality"]  # 先读取当前模态。
    if modality in {MODALITY_UWB, MODALITY_VIO} and not _is_neutral_intermediate(intermediate):  # 只有有意义的中间量才走安全模式修正。
        scene_context = _resolve_safe_mode_context_from_event(event)  # 从事件中找可用场景上下文。
        if scene_context is not None:  # 只要找到了上下文就修正中间量。
            intermediate, _ = adjust_intermediate_for_safe_mode(
                intermediate,
                scene_context,
                safe_mode_cfg=safe_mode_cfg,
            )  # 忽略标志，只拿修正后的中间量。
    risk = normalize_risk(intermediate.risk)  # 先把风险统一裁剪到合法区间。
    if modality == MODALITY_UWB:  # UWB 走 UWB 专属分支。
        uwb_payload = event.get("uwb_payload") or {}  # 没有 payload 时当作空字典处理。
        scaling = _coerce_positive_scaling(intermediate.uwb_scaling, name="uwb_scaling")  # 校验并规范化缩放。
        raw_range = float((uwb_payload.get("range")) or 0.0)  # 从 payload 读取原始测距。
        if not np.isfinite(raw_range):  # NaN/Inf 测距无法作为偏置修正的参考，按缺失处理。
            raw_range = 0.0
        bias_applied = clip_uwb_bias(intermediate.bias, raw_range)  # 委托唯一权威实现做 bias 截断。
        quality = _resolve_quality(uwb_payload.get("quality", 1.0))  # 从 payload 里读取质量，缺省为 1。
        applied_risk = _resolve_effective_risk(event, base_risk=risk, quality=quality)  # 计算最终风险。
        noise_multiplier = _compose_noise_multiplier(  # 计算噪声倍数。
            scaling=scaling,  # 传入缩放。
            risk=applied_risk,  # 传入最终风险。
            ceiling=_UWB_NOISE_MULTIPLIER_CEILING,  # 使用 UWB 上限。
        )  # 噪声倍数结束。
        action = apply_safe_mode(  # 计算安全模式动作。
            modality="uwb",  # 当前模态是 UWB。
            valid=_resolve_uwb_valid_flag(uwb_payload.get("valid", True)),  # 统一 valid 语义。
            quality=quality,  # 传入质量。
            risk=applied_risk,  # 传入最终风险。
        )  # 动作计算结束。
        import sys
        print(f"[DBG-UWB] risk={applied_risk:.3f} >= {_RISK_HARD_SKIP_THRESHOLD} → action={action}", file=sys.stderr)
    elif modality == MODALITY_VIO:  # VIO 走 VIO 专属分支。
        vio_payload = event.get("vio_payload") or {}  # 没有 payload 时当作空字典处理。
        scaling = _coerce_positive_scaling(intermediate.vio_scaling, name="vio_scaling")  # 校验并规范化缩放。
        bias_applied = 0.0  # VIO 不直接应用 bias。
        quality = _resolve_quality(vio_payload.get("quality", 1.0))  # 读取 VIO 质量。
        applied_risk = _resolve_effective_risk(event, base_risk=risk, quality=quality)  # 计算最终风险。
        noise_multiplier = _compose_noise_multiplier(  # 计算噪声倍数。
            scaling=scaling,  # 传入缩放。
            risk=applied_risk,  # 传入最终风险。
            ceiling=_VIO_NOISE_MULTIPLIER_CEILING,  # 使用 VIO 上限。
        )  # 噪声倍数结束。
        action = apply_safe_mode(  # 计算安全动作。
            modality="vio",  # 当前模态是 VIO。
            valid=True,  # VIO 这里固定认为有效。
            quality=quality,  # 传入质量。
            risk=applied_risk,  # 传入最终风险。
        )  # 动作计算结束。
    else:  # 其他模态不做特殊处理。
        scaling = 1.0  # 缩放回到默认值。
        bias_applied = 0.0  # bias 设为零。
        noise_multiplier = 1.0  # 噪声倍数回到默认值。
        action = "pass_through"  # 门控动作直接透传。
        applied_risk = risk  # 风险保持当前值。
    return MeasurementControl(  # 组装最终控制对象。
        modality=modality,  # 写入模态。
        bias_applied=bias_applied,  # 写入 bias。
        scaling=scaling,  # 写入缩放。
        risk=applied_risk,  # 写入风险。
        noise_multiplier=noise_multiplier,  # 写入噪声倍数。
        gate_action=action,  # 写入动作名。
    )  # 对象构造结束。
