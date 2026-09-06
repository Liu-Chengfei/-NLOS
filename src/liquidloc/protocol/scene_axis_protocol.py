"""冻结场景轴协议加载与展开辅助模块。

文件职责：
  加载并校验冻结场景轴协议（configs/base/scene_axis_protocol.yaml），
  提供轴层级解析和场景参数展开功能。
  核心目标是确保场景轴定义在运行时不会漂移，所有场景参数都来自冻结协议。

本文件绝对不负责：
  不定义协议内容本身（轴名、层级名、参数表由 YAML 定义）。
  不修改协议文件。
  不执行任何场景生成或数据模拟逻辑。

核心数据流：
    scene_axis_protocol.yaml → load_scene_axis_protocol → 校验通过 → resolve_axis_level / attach_scene_parameters

上游依赖：
  liquidloc.common.config_utils（load_yaml_config 加载 YAML 配置）

下游调用者：
  protocol/scene_schema.py（使用 get_protocol_axes 和 load_scene_axis_protocol）、
  protocol/liquid_bridge_contract.py（使用 load_scene_axis_protocol 获取轴级退化风险下界）、
  scenarios/（使用 resolve_axis_level 和 attach_scene_parameters）

输入对象定义：
  - protocol_path  可选的协议文件路径，默认使用 configs/base/scene_axis_protocol.yaml
  - axis           轴名（A/N/V/G/K/M）
  - level          层级名（如 A0、A1 等）
  - axes           轴名到层级名的映射字典

输出对象定义：
  - load_scene_axis_protocol        加载并校验冻结场景轴协议
  - resolve_axis_level              解析单个轴层级的参数
  - attach_scene_parameters         展开所有轴的参数并扁平化

核心变量定义：
  - AXES                            冻结的轴名元组 (A, N, V, K, M)，2026-08-31 删除 G 轴并入 K
  - _PROTOCOL_VERSION               协议版本号（2）
  - _DEFAULT_PATH                   冻结场景轴协议 YAML 的默认路径

关键设计决策：
  - 场景轴协议在模块加载时即冻结，后续调用只读。
  - 校验时与仓库冻结快照做精确比较，任何漂移都会导致加载失败。
  - 只允许 A/N/V/K/M 五个轴，不允许动态扩展（G 轴已删除）。
  - 每个轴的每个层级必须是映射类型，承载该层级的参数表。
"""

from __future__ import annotations  # 允许类型注解中引用尚未定义的类型。

import copy
from collections.abc import Mapping  # 用于类型检查映射类型。
from dataclasses import dataclass, field  # 提供数据类装饰器和字段工厂。
from functools import lru_cache  # 给协议快照加缓存，避免重复加载。
from pathlib import Path  # 用于处理协议文件路径。
from typing import Any  # 允许类型注解里表示"任意类型"。

from types import MappingProxyType  # 只读字典代理，防止外部修改冻结协议数据。

from liquidloc.common.config_utils import find_project_root, load_yaml_config  # 加载 YAML 配置文件和项目根查找器。
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_integer, is_real, is_string_like  # 标量强转、数值类型校验工具。第 8 阶段修复 HIGH-5: _interval_midpoint 需要 coerce_finite_scalar。
from liquidloc.protocol.version import PROTOCOL_VERSION as _PROTOCOL_VERSION  # 从集中版本模块导入协议版本号。

AXES = ("A", "N", "V", "K", "M")  # 冻结的轴名元组，不允许动态扩展。2026-08-31 删 G 轴（并入 K）。


def _interval_midpoint(value: Any) -> float:
    """把标量或区间 [a,b] 映射为中点标量, 用于范围校验。"""
    if isinstance(value, (list, tuple)):
        lo = coerce_finite_scalar(value[0], name='interval_low')
        hi = coerce_finite_scalar(value[1] if len(value) >= 2 else value[0], name='interval_high')
        return (lo + hi) / 2.0
    return coerce_finite_scalar(value, name='scalar')
_AXES = AXES  # 向后兼容别名。
SCENE_AXES = AXES  # 公共导出别名，供下游模块引用，避免重复定义导致漂移。
AXIS_METADATA_KEYS = {
    "range_semantics", "backward_compatibility_alias",
    # H26 真改 §4.3.1 L933「主表须事先写死生成族」：N 轴顶层声明字段（不是层级定义）。
    # 写在 N 轴 yaml 顶层与 N0/N1/N2/N3 同级，让协议 yaml 显式声明所选 NLOS 生成族。
    # 这些键被 L561-563/L572 跳过 level payload 校验，由 _validate_axis_param_semantic_range
    # 在 N 轴分支单独枚举校验（_ALLOWED_NLOS_GENERATION_FAMILIES 三选一）。
    "nlos_generation_family", "nlos_generation_family_doc",
}  # 轴级别的元数据键，不是层级定义。
_AXIS_METADATA_KEYS = AXIS_METADATA_KEYS  # 向后兼容别名。
# 每个轴每个层级允许的参数键白名单，防止 YAML 中意外注入未声明参数。
_AXIS_LEVEL_PARAM_KEYS = {
    "A": {"label", "offset_ms", "jitter_ms", "burst_missing_prob", "cross_modal_skew_ms", "clock_drift_ppm",
          # Allan 方差分层建模（可选字段，缺省为 0，保持向后兼容）。
          "clock_bias_instability_ms", "clock_rrw_per_sqrt_s_ms"},
    "N": {"label", "nlos_ratio", "bias_strength_m", "nlos_noise_std_m",
          # H22c-spillover 真改 §4.3 L927：簇发最短持续时段（秒），>0 时 apply_nlos_level
          # 调用 _enforce_min_cluster_duration 强制至少一段连续簇发时长 ≥ min_cluster_duration_s。
          # 缺省 0.0 表示"不强制连续簇发时段"，向后兼容旧 yaml/旧测试。
          "min_cluster_duration_s",
          # H24d 真改 §4.3.1 L1002：遮挡物类型枚举，apply_nlos_level selection_key_parts
          # 注入 occluder_type 让 NLOS 候选事件选择按遮挡物类型分桶（不同 occluder_type
          # 产生不同 selection_key → 不同 RNG 桶）。枚举由 _NLOS_OCCLUDER_TYPES 校验。
          # 缺省 machine_body 与旧 yaml/旧测试向后兼容。
          "occluder_type",
          # H26 真改 §4.3.1 L933「主表须事先写死生成族」+ L935「允许的生成族须声明」：
          # 协议层显式声明 N 轴所选 NLOS 生成族（A 几何视线/B 状态条件统计混合/C=A+B 混合）。
          # 白名单内允许 yaml 顶层 N 轴写 nlos_generation_family + nlos_generation_family_doc 两个字段；
          # 真值校验在 _validate_axis_param_semantic_range 中执行（必须取值
          # C_AB_hybrid / A_geometric_los / B_state_conditional_statistical 三选一）。
          "nlos_generation_family", "nlos_generation_family_doc"},
    "V": {"label", "tracked_features_range", "reproj_err_max", "blackout_prob", "drift_bias_sigma_mps",
          # 2026-08-31 目标表对齐改造：新增 3 字段承载 V2 (σ 噪声放大) 和 V3 (关键帧中断 + 跳变)：
          #   increment_noise_std_mps       ← 增量噪声标准差 σ (m/步)，V2 模拟 σ×5 放大 (0.05-0.15m)
          #   keyframe_interruptions_per_seq ← 关键帧中断次数/序列，V3 模拟 2-4 次/序列
          #   keyframe_recovery_jump_m      ← 关键帧中断恢复跳变 (m)，V3 模拟 0.3-1m 跳变
          "increment_noise_std_mps", "keyframe_interruptions_per_seq", "keyframe_recovery_jump_m",
          # 2026-09-01 补齐：V 轴黑屏/关键帧突发的持续时间区间(s) + 占轨迹比例。
          #   blackout_duration_range_s      ← V1/V2/V3 黑屏单次持续时间区间(s)
          #   blackout_fraction              ← 黑屏占轨迹比例（协议表：V1 5-10%, V2 20-40%, V3 40-60%）
          #   keyframe_burst_duration_range_s ← V2/V3 关键帧中断 burst 持续时间区间(s)
          #   keyframe_burst_fraction         ← 关键帧 burst 占轨迹比例（V3 10-20%）
          "blackout_duration_range_s", "blackout_fraction",
          "keyframe_burst_duration_range_s", "keyframe_burst_fraction"},
    # 2026-08-31：G 轴并入 K 轴（原 geom_condition 现属 K 轴）。
    "K": {"label", "anchor_count", "geom_condition"},  # 两字段同存（geom_condition=1/2/10，K0/K1/K3 全档固定 4 锚；文档「五轴档位协议定义」硬约束「锚数全档固定 4」）。
    "M": {"label", "modality_drop_prob", "imu_drop_prob", "affected_modalities", "cluster_duration_range_s"},  # M 轴：数据缺失（含NLOS不可解帧），drop_prob ∈ [0,1]，imu_drop_prob ≤0.05 (IMU上限)，affected_modalities 是 ["uwb","vio","imu"] 的子集，cluster_duration_range_s ∈ [0,5]。
}
_DEFAULT_PATH = (find_project_root() / "configs" / "base" / "scene_axis_protocol.yaml").resolve()  # 默认协议路径，resolve() 在模块加载时完成。

# H24d 真改 §4.3.1 L1002「机身/人体自遮挡、墙体材料（混凝土/金属/玻璃等）随位姿
# 与锚点视线变化」遮挡物类型枚举：本常量声明 N 轴 occluder_type 字段允许的 5 类枚举值，
# _validate_axis_param_semantic_range 校验 occluder_type 必须属于此集合。任何新增类型
# （如 wood_wall / plaster_wall）须同时改本 frozenset + scene_axis_protocol.yaml +
# 单测，与 _AXIS_LEVEL_PARAM_KEYS 单轴白名单同口径协议级护栏。
_NLOS_OCCLUDER_TYPES: frozenset[str] = frozenset({
    "machine_body",    # 机身自遮挡（N1 默认值，对应轻度 NLOS 偶发反射/衍射瞬时主导）。
    "human_body",       # 人体动态遮挡（行人/手持/小型地面机器人等动态体遮挡）。
    "concrete_wall",    # 混凝土墙体遮挡（N2 默认值，对应中度 NLOS 多径散射）。
    "metal_wall",       # 金属墙体遮挡（N3 默认值，对应重度 NLOS 强反射大脉冲）。
    "glass_wall",       # 玻璃墙体遮挡（半透射+反射混合，单测覆盖未入 yaml 默认）。
})


# H26 真改 §4.3.1 L933「主表须事先写死生成族」+ L935「允许的生成族（协议二选一或组合，须声明）」：
# 协议层 N 轴 nlos_generation_family 字段允许的取值枚举：
#   - A_geometric_los:                A 几何视线/遮挡族（墙体/转角/动态遮挡体，视线决定是否 NLOS）
#   - B_state_conditional_statistical: B 状态条件统计混合族（按位置/朝向/锚点分区，LOS/NLOS 抽样脉冲）
#   - C_AB_hybrid:                     C = A+B 混合族（几何决定是否可能 NLOS，统计决定幅度/簇发）
# 三选一，与 §4.3.1 L937-941 表对应。本仓库默认 C_AB_hybrid（见 configs/base/scene_axis_protocol.yaml）。
_ALLOWED_NLOS_GENERATION_FAMILIES: frozenset[str] = frozenset({
    "A_geometric_los",
    "B_state_conditional_statistical",
    "C_AB_hybrid",
})


@dataclass(frozen=True)
class SceneParameters(Mapping):
    """场景参数的结构化返回类型，替代裸 dict。

    继承 collections.abc.Mapping，支持 dict 风格访问（isinstance 检查、
    keys/values/items 迭代等）。嵌套字典在构造后通过 MappingProxyType
    包装为一层只读代理，阻止最常见的外部修改模式（如 params.axes["A"] = {}）。
    深层嵌套对象仍可被修改，但 to_dict() 返回深拷贝可安全使用。

    属性：
        axes: 轴名到参数字典的映射，保留层级结构。
            每个轴的参数字典包含该轴的所有参数字段，
            值类型为标量（float/str）或列表（如 A_offset_ms: [0, 10]）。
        flat: 扁平化参数字典，键格式为 {轴名}_{参数名}。
            值类型与 axes 中对应字段一致——标量或列表混存，
            下游消费者需根据协议定义区分。
        axis_metadata: 轴名到轴级元数据的映射，保留不属于具体层级的轴级声明。
            例如 V 轴的 range_semantics 声明，下游消费者可据此校验语义约束。
    """
    axes: dict[str, dict[str, Any]]  # 轴名 → 参数字典（保留层级结构）。
    flat: dict[str, Any]  # 扁平化参数字典（标量与列表混存）。
    axis_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)  # 轴名 → 轴级元数据。

    def __post_init__(self):
        """将嵌套字典替换为只读代理，防止外部修改冻结协议数据。"""
        object.__setattr__(self, 'axes', MappingProxyType(self.axes))
        object.__setattr__(self, 'flat', MappingProxyType(self.flat))
        object.__setattr__(self, 'axis_metadata', MappingProxyType(self.axis_metadata))

    def __getitem__(self, key: str) -> Any:
        """支持 dict 风格访问（result["axes"]、result["flat"]、result["axis_metadata"]），保持向后兼容。"""
        if key == "axes":
            return self.axes
        if key == "flat":
            return self.flat
        if key == "axis_metadata":
            return self.axis_metadata
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """支持 dict 风格 .get() 访问，保持向后兼容。"""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """支持 dict 风格 .keys()，保持向后兼容。"""
        return ("axes", "flat", "axis_metadata")

    def values(self):
        """支持 dict 风格 .values()，保持向后兼容。"""
        return (self.axes, self.flat, self.axis_metadata)

    def items(self):
        """支持 dict 风格 .items()，保持向后兼容。"""
        return (("axes", self.axes), ("flat", self.flat), ("axis_metadata", self.axis_metadata))

    def __iter__(self):
        """支持迭代，按 ('axes', 'flat', 'axis_metadata') 顺序产出键名。"""
        return iter(("axes", "flat", "axis_metadata"))

    def __len__(self) -> int:
        """返回字段数量，始终为 3。"""
        return 3

    def __contains__(self, key: object) -> bool:
        """支持 `in` 运算符，保持向后兼容。"""
        return key in ("axes", "flat", "axis_metadata")

    def to_dict(self) -> dict[str, Any]:
        """转换为纯 dict，用于 JSON 序列化等场景。返回深拷贝，修改不影响原对象。"""
        return copy.deepcopy({"axes": dict(self.axes), "flat": dict(self.flat), "axis_metadata": dict(self.axis_metadata)})


def _validate_protocol_version(cfg: Mapping[str, Any], *, required: bool) -> None:
    """校验协议版本号是否与冻结值一致。

    参数：
        cfg: 协议配置映射。
        required: 是否要求版本号必须存在。加载冻结快照时为 True，外部传入时为 False。

    异常：
        ValueError: 版本号缺失（required=True 时）或不匹配时抛出。
    """
    if "protocol_version" not in cfg:  # 版本号字段缺失。
        if required:  # 加载冻结快照时必须存在。
            raise ValueError(f"scene axis protocol version is missing, expected {_PROTOCOL_VERSION}")
        return  # 外部传入时缺失不报错，由后续校验决定。
    version = cfg.get("protocol_version")
    if not is_integer(version):
        raise TypeError(f"scene axis protocol_version must be an integer, got {type(version).__name__}")
    if version != _PROTOCOL_VERSION:  # 版本号不匹配。
        raise ValueError(f"scene axis protocol version must be {_PROTOCOL_VERSION}")


def _validate_axis_param_semantic_range(axis: str, level: str, payload: Mapping[str, Any]) -> None:
    """校验轴层级参数的语义范围约束。

    本函数只校验 _AXIS_LEVEL_PARAM_KEYS 白名单内各轴字段的物理量范围。

    各轴参数的物理含义决定了其合法范围：
    - A 轴：offset_ms 下界 >= 0，jitter_ms >= 0，burst_missing_prob ∈ [0,1]，
      cross_modal_skew_ms >= 0，clock_drift_ppm >= 0
    - N 轴：nlos_ratio ∈ [0,1]，bias_strength_m >= 0，bias_drift_mps >= 0，nlos_noise_std_m >= 0
    - V 轴：tracked_features_range 下界 >= 0，reproj_err_max > 0，blackout_prob ∈ [0,1]，drift_bias_sigma_mps ∈ [0,1]，
      increment_noise_std_mps >= 0，keyframe_interruptions_per_seq >= 0，keyframe_recovery_jump_m >= 0，
      blackout_duration_range_s >= 0，blackout_fraction ∈ [0,1]，
      keyframe_burst_duration_range_s >= 0，keyframe_burst_fraction ∈ [0,1]
    - G 轴：geom_condition > 0
    - K 轴：anchor_count 为正整数且 >= 3
    - M 轴：modality_drop_prob ∈ [0,1]，affected_modalities 是 ["uwb","vio","imu"] 的子集，cluster_duration_range_s ∈ [0,5]

    §4.4 协议纪律（前提指导.md L1081-1091）：以下 11 项 UWB 衍生观测量字段名由
    _AXIS_LEVEL_PARAM_KEYS 白名单（L68-79）拒绝，写入 yaml 即在 _validate_axis_protocol_payload
    L463-474 的 unknown_keys 校验里抛 ValueError，本函数不重复校验：
      differential_range / range_diff / doppler / radial_velocity /
      carrier_phase / integer_ambiguity / antenna_selection / antenna_diversity /
      reference_transponder / golden_node / aoa / pdoa / tdoa /
      cir_corrected / peak_corrected / sliding_median / rolling_mean / exponential_smooth /
      outlier_clip / soft_clip / rf_switch / antenna_switch_event。

    参数：
        axis: 轴名（A/N/V/G/K/M）。
        level: 层级名（如 A0、V3 等）。
        payload: 该层级的参数映射。

    异常：
        ValueError: 参数值超出语义范围时抛出。
    """
    prefix = f"scene axis protocol {axis}.{level}"

    if axis == "A":
        offset_ms = payload.get("offset_ms")
        if isinstance(offset_ms, (list, tuple)) and len(offset_ms) >= 2:
            if float(offset_ms[0]) < 0.0:
                raise ValueError(f"{prefix}.offset_ms lower bound must be non-negative, got {offset_ms[0]}")
            if len(offset_ms) >= 2 and float(offset_ms[0]) > float(offset_ms[1]):
                raise ValueError(f"{prefix}.offset_ms lower bound must be <= upper bound, got {offset_ms}")
            if float(offset_ms[1]) > 1000.0:
                raise ValueError(f"{prefix}.offset_ms upper bound must be <= 1000.0ms, got {offset_ms[1]}")
        jitter_ms = payload.get("jitter_ms")
        if is_real(jitter_ms) and float(jitter_ms) < 0.0:
            raise ValueError(f"{prefix}.jitter_ms must be non-negative, got {jitter_ms}")
        burst_missing_prob = payload.get("burst_missing_prob")
        if is_real(burst_missing_prob) and not (0.0 <= float(burst_missing_prob) <= 1.0):
            raise ValueError(f"{prefix}.burst_missing_prob must be in [0, 1], got {burst_missing_prob}")
        cross_modal_skew_ms = payload.get("cross_modal_skew_ms")
        if is_real(cross_modal_skew_ms) and float(cross_modal_skew_ms) < 0.0:
            raise ValueError(f"{prefix}.cross_modal_skew_ms must be non-negative, got {cross_modal_skew_ms}")
        clock_drift_ppm = payload.get("clock_drift_ppm")
        if is_real(clock_drift_ppm) and float(clock_drift_ppm) < 0.0:
            raise ValueError(f"{prefix}.clock_drift_ppm must be non-negative, got {clock_drift_ppm}")

    elif axis == "N":
        # N 轴：nlos_ratio / bias_strength_m / nlos_noise_std_m 支持两种形式：
        #   1) 标量单值（兼容旧版 yaml）
        #   2) 区间列表 [low, high]（v2 区间化，生成时在区间内均匀采样）
        # 区间语义：low <= high；is_real() 兼容 list/tuple，但为与 0.0<=x<=1.0 范围校验
        # 互不矛盾（区间下界和上界都需在合法范围内），下面分别处理标量和区间。
        for n_param, max_or_low in (
            ("nlos_ratio", 1.0),
            ("nlos_noise_std_m", None),  # 仅校验 >= 0
        ):
            v = payload.get(n_param)
            if v is None:
                continue
            if isinstance(v, (list, tuple)) and len(v) == 2:
                lo, hi = float(v[0]), float(v[1])
                if lo < 0.0:
                    raise ValueError(f"{prefix}.{n_param} interval lower bound must be >= 0, got {lo}")
                if lo > hi:
                    raise ValueError(f"{prefix}.{n_param} interval lower bound must be <= upper bound, got {v}")
                if n_param == "nlos_ratio" and hi > 1.0:
                    raise ValueError(f"{prefix}.{n_param} interval upper bound must be <= 1.0, got {hi}")
            elif is_real(v):
                if float(v) < 0.0:
                    raise ValueError(f"{prefix}.{n_param} must be non-negative, got {v}")
                if n_param == "nlos_ratio" and float(v) > 1.0:
                    raise ValueError(f"{prefix}.{n_param} must be in [0, 1], got {v}")
            else:
                raise TypeError(
                    f"{prefix}.{n_param} must be a real number or a 2-value interval [low, high], got {type(v).__name__}"
                )
        # bias_strength_m 单独处理：有 50.0m 绝对上限（铁律 4 大脉冲模型）
        bias_strength_m = payload.get("bias_strength_m")
        if bias_strength_m is not None:
            if isinstance(bias_strength_m, (list, tuple)) and len(bias_strength_m) == 2:
                lo, hi = float(bias_strength_m[0]), float(bias_strength_m[1])
                if lo < 0.0:
                    raise ValueError(f"{prefix}.bias_strength_m interval lower bound must be >= 0, got {lo}")
                if lo > hi:
                    raise ValueError(f"{prefix}.bias_strength_m interval lower bound must be <= upper bound, got {bias_strength_m}")
                if hi > 50.0:
                    raise ValueError(f"{prefix}.bias_strength_m interval upper bound must be <= 50.0 (large-pulse model upper bound), got {hi}")
            elif is_real(bias_strength_m):
                if float(bias_strength_m) < 0.0:
                    raise ValueError(f"{prefix}.bias_strength_m must be non-negative, got {bias_strength_m}")
                if float(bias_strength_m) > 50.0:
                    raise ValueError(f"{prefix}.bias_strength_m must be <= 50.0 (large-pulse model upper bound), got {bias_strength_m}")
            else:
                raise TypeError(
                    f"{prefix}.bias_strength_m must be a real number or a 2-value interval [low, high], got {type(bias_strength_m).__name__}"
                )
        # H22c-spillover 真改 §4.3 L927：min_cluster_duration_s 非负有限校验，
        # apply_nlos_level 在 >0 时调用 _enforce_min_cluster_duration 强制至少一段
        # 连续簇发时长 ≥ min_cluster_duration_s。缺字段不校验（向后兼容旧 yaml）。
        min_cluster_duration_s = payload.get("min_cluster_duration_s")
        if is_real(min_cluster_duration_s) and float(min_cluster_duration_s) < 0.0:
            raise ValueError(
                f"{prefix}.min_cluster_duration_s must be non-negative, got {min_cluster_duration_s}"
            )
        # H24d 真改 §4.3.1 L1002：occluder_type 校验，必须属于 _NLOS_OCCLUDER_TYPES 枚举。
        # 缺字段不校验（向后兼容旧 yaml，apply_nlos_level 缺省回退 machine_body）。
        occluder_type = payload.get("occluder_type")
        if occluder_type is not None and not isinstance(occluder_type, str):
            raise ValueError(
                f"{prefix}.occluder_type must be a string, got {type(occluder_type).__name__}"
            )
        if isinstance(occluder_type, str) and occluder_type not in _NLOS_OCCLUDER_TYPES:
            raise ValueError(
                f"{prefix}.occluder_type must be one of "
                f"{sorted(_NLOS_OCCLUDER_TYPES)}, got {occluder_type!r}"
            )
        # H26 真改 §4.3.1 L933「主表须事先写死生成族」+ L935「允许的生成族须声明」：
        # nlos_generation_family 必须属于 _ALLOWED_NLOS_GENERATION_FAMILIES 三选一枚举。
        # 缺字段不校验（向后兼容旧 yaml），但写新 yaml 必须显式声明所选生成族（A/B/C）。
        # 本字段写在 N 轴顶层（与 N0/N1/N2/N3 同级），非每个档位独立字段——
        # 因为 §4.3.1 L933 要求「主表」（即主 yaml）层面事先写死生成族，而非每档分别写。
        nlos_generation_family = payload.get("nlos_generation_family")
        if nlos_generation_family is not None and not isinstance(nlos_generation_family, str):
            raise ValueError(
                f"{prefix}.nlos_generation_family must be a string, got "
                f"{type(nlos_generation_family).__name__}"
            )
        if (
            isinstance(nlos_generation_family, str)
            and nlos_generation_family not in _ALLOWED_NLOS_GENERATION_FAMILIES
        ):
            raise ValueError(
                f"{prefix}.nlos_generation_family must be one of "
                f"{sorted(_ALLOWED_NLOS_GENERATION_FAMILIES)}, got {nlos_generation_family!r}"
            )
        # nlos_generation_family_doc 仅字符串校验，非空字符串即可（docstring 不参与枚举校验）。
        nlos_generation_family_doc = payload.get("nlos_generation_family_doc")
        if nlos_generation_family_doc is not None and not isinstance(nlos_generation_family_doc, str):
            raise ValueError(
                f"{prefix}.nlos_generation_family_doc must be a string, got "
                f"{type(nlos_generation_family_doc).__name__}"
            )

    elif axis == "V":
        tracked_features_range = payload.get("tracked_features_range")
        if isinstance(tracked_features_range, (list, tuple)) and len(tracked_features_range) >= 2:
            if float(tracked_features_range[0]) < 0.0:
                raise ValueError(f"{prefix}.tracked_features_range lower bound must be non-negative, got {tracked_features_range[0]}")
            if float(tracked_features_range[0]) > float(tracked_features_range[1]):
                raise ValueError(f"{prefix}.tracked_features_range lower bound must be <= upper bound, got {tracked_features_range}")
        if isinstance(tracked_features_range, (list, tuple)) and len(tracked_features_range) >= 2:
            if float(tracked_features_range[1]) > 1000.0:
                raise ValueError(f"{prefix}.tracked_features_range upper bound must be <= 1000.0, got {tracked_features_range[1]}")
        reproj_err_max = payload.get("reproj_err_max")
        if _interval_midpoint(reproj_err_max) <= 0.0:  # HIGH-5 修复: 兼容 list/tuple 区间值, 取中点校验。
            raise ValueError(f"{prefix}.reproj_err_max must be positive, got {reproj_err_max}")
        blackout_prob = payload.get("blackout_prob")
        if not (0.0 <= _interval_midpoint(blackout_prob) <= 1.0):  # HIGH-5 修复: 兼容 list/tuple 区间值, 取中点校验。
            raise ValueError(f"{prefix}.blackout_prob must be in [0, 1], got {blackout_prob}")
        drift_bias_sigma_mps = payload.get("drift_bias_sigma_mps")
        drift_bias_mid = _interval_midpoint(drift_bias_sigma_mps)  # HIGH-5 修复: 兼容 list/tuple 区间值, 取中点校验。
        if drift_bias_mid < 0.0:
            raise ValueError(f"{prefix}.drift_bias_sigma_mps must be non-negative, got {drift_bias_sigma_mps}")
        if drift_bias_mid > 1.0:
            # drift_bias_sigma_mps: Wiener drift 标准差 σ（m/√s），累积幅度 = σ·√T
            raise ValueError(f"{prefix}.drift_bias_sigma_mps must be <= 1.0 (degrade_quality saturation limit), got {drift_bias_sigma_mps}")
        # 2026-08-31 新增字段校验（V2 σ 噪声放大 + V3 关键帧中断+跳变）
        increment_noise_std_mps = payload.get("increment_noise_std_mps")
        inc_noise_mid = _interval_midpoint(increment_noise_std_mps)  # HIGH-5 修复: 兼容 list/tuple 区间值, 取中点校验。
        if inc_noise_mid < 0.0:
            raise ValueError(
                f"{prefix}.increment_noise_std_mps must be non-negative (V0 baseline 0; V2 σ×5 放大至 0.05-0.15m), "
                f"got {increment_noise_std_mps}"
            )
        if inc_noise_mid > 1.0:  # HIGH-5 修复: 用已算中点, 兼容 list/tuple 区间值。
            raise ValueError(
                f"{prefix}.increment_noise_std_mps must be <= 1.0 m/步, got {increment_noise_std_mps}"
            )
        keyframe_interruptions_per_seq = payload.get("keyframe_interruptions_per_seq")
        if _interval_midpoint(keyframe_interruptions_per_seq) < 0.0:  # HIGH-5 修复: 兼容 list/tuple 区间值。
            raise ValueError(
                f"{prefix}.keyframe_interruptions_per_seq must be non-negative (V0 baseline 0; V3 模拟 2-4 次/序列), "
                f"got {keyframe_interruptions_per_seq}"
            )
        keyframe_recovery_jump_m = payload.get("keyframe_recovery_jump_m")
        if _interval_midpoint(keyframe_recovery_jump_m) < 0.0:  # HIGH-5 修复: 兼容 list/tuple 区间值。
            raise ValueError(
                f"{prefix}.keyframe_recovery_jump_m must be non-negative (V0 baseline 0; V3 模拟 0.3-1m 跳变), "
                f"got {keyframe_recovery_jump_m}"
            )
        # 2026-09-01 补齐字段校验（V 轴黑屏/关键帧突发持续时间 + 占轨迹比例）
        blackout_duration_range_s = payload.get("blackout_duration_range_s")
        if isinstance(blackout_duration_range_s, (list, tuple)) and len(blackout_duration_range_s) >= 2:
            if float(blackout_duration_range_s[0]) < 0.0:
                raise ValueError(f"{prefix}.blackout_duration_range_s lower bound must be non-negative, got {blackout_duration_range_s[0]}")
        blackout_fraction = payload.get("blackout_fraction")
        if isinstance(blackout_fraction, (list, tuple)) and len(blackout_fraction) >= 2:
            if not (0.0 <= float(blackout_fraction[0]) <= 1.0 and 0.0 <= float(blackout_fraction[1]) <= 1.0):
                raise ValueError(f"{prefix}.blackout_fraction must be in [0, 1], got {blackout_fraction}")
        keyframe_burst_duration_range_s = payload.get("keyframe_burst_duration_range_s")
        if isinstance(keyframe_burst_duration_range_s, (list, tuple)) and len(keyframe_burst_duration_range_s) >= 2:
            if float(keyframe_burst_duration_range_s[0]) < 0.0:
                raise ValueError(f"{prefix}.keyframe_burst_duration_range_s lower bound must be non-negative, got {keyframe_burst_duration_range_s[0]}")
        keyframe_burst_fraction = payload.get("keyframe_burst_fraction")
        if isinstance(keyframe_burst_fraction, (list, tuple)) and len(keyframe_burst_fraction) >= 2:
            if not (0.0 <= float(keyframe_burst_fraction[0]) <= 1.0 and 0.0 <= float(keyframe_burst_fraction[1]) <= 1.0):
                raise ValueError(f"{prefix}.keyframe_burst_fraction must be in [0, 1], got {keyframe_burst_fraction}")

    elif axis == "K":
        # 2026-08-31：G 轴并入 K 轴，K 轴同时承载 anchor_count 和 geom_condition。
        # K0/K1 固定 4 锚对称/非对称；K3 固定 4 锚近共线退化（3 锚共线 + 1 锚孤立）。
        # 与「五轴档位协议定义」文档一致：文档规定「锚数全档固定 4」。
        anchor_count = payload.get("anchor_count")
        if is_real(anchor_count):
            if not (is_integer(anchor_count)):
                raise ValueError(f"{prefix}.anchor_count must be an integer, got {anchor_count}")
            anchor_count_int = int(anchor_count)
            if anchor_count_int < 3:
                raise ValueError(
                    f"{prefix}.anchor_count must be >= 3 (at least 3 anchors for 2D localization), got {anchor_count_int}"
                )
        # geom_condition 区间/标量校验（1.0/2.0/10.0 + 区间化）
        geom_condition = payload.get("geom_condition")
        if geom_condition is not None:
            if isinstance(geom_condition, (list, tuple)) and len(geom_condition) == 2:
                lo, hi = float(geom_condition[0]), float(geom_condition[1])
                if lo <= 0.0:
                    raise ValueError(f"{prefix}.geom_condition interval lower bound must be positive, got {lo}")
                if lo > hi:
                    raise ValueError(f"{prefix}.geom_condition interval lower bound must be <= upper bound, got {geom_condition}")
            elif is_real(geom_condition):
                if float(geom_condition) <= 0.0:
                    raise ValueError(f"{prefix}.geom_condition must be positive, got {geom_condition}")
            else:
                raise TypeError(
                    f"{prefix}.geom_condition must be a real number or a 2-value interval [low, high], "
                    f"got {type(geom_condition).__name__}"
                )
        # §4.2.e 协议纪律（前提指导.md:897）：锚点坐标误差模型全员同一。
        # 本协议对 K 轴默认 anchor 坐标误差模型sigma_anchor_xy = 0.0m（即视为精确已知），
        # 所有估计器（EKF / robust_ekf / FGO / NN+EKF）共享同一锚点几何，无单方私有误差模型。
        # 若需引入 anchor 坐标不确定性，必须在协议层登记（如新增 sigma_anchor_xy_m 字段）
        # 且由 predict_range / _coerce_anchor_xy 全员同一应用，禁止任一方法单方更"真"的锚。
        # 历史代码路径 grep `anchor.*sigma|anchor.*error|anchor.*noise` 在 src/ 下 0 命中。

    elif axis == "M":
        # modality_drop_prob 支持标量或区间 [low, high]（v2 区间化）
        modality_drop_prob = payload.get("modality_drop_prob")
        if modality_drop_prob is not None:
            if isinstance(modality_drop_prob, (list, tuple)) and len(modality_drop_prob) == 2:
                lo, hi = float(modality_drop_prob[0]), float(modality_drop_prob[1])
                if lo < 0.0 or lo > 1.0:
                    raise ValueError(f"{prefix}.modality_drop_prob interval lower bound must be in [0, 1], got {lo}")
                if hi < 0.0 or hi > 1.0:
                    raise ValueError(f"{prefix}.modality_drop_prob interval upper bound must be in [0, 1], got {hi}")
                if lo > hi:
                    raise ValueError(f"{prefix}.modality_drop_prob interval lower bound must be <= upper bound, got {modality_drop_prob}")
            elif is_real(modality_drop_prob):
                if not (0.0 <= float(modality_drop_prob) <= 1.0):
                    raise ValueError(f"{prefix}.modality_drop_prob must be in [0, 1], got {modality_drop_prob}")
            else:
                raise TypeError(
                    f"{prefix}.modality_drop_prob must be a real number or a 2-value interval [low, high], got {type(modality_drop_prob).__name__}"
                )
        affected_modalities = payload.get("affected_modalities")
        if affected_modalities is not None:
            if not isinstance(affected_modalities, (list, tuple)):
                raise TypeError(f"{prefix}.affected_modalities must be a list, got {type(affected_modalities).__name__}")
            _allowed_modalities = {"uwb", "vio", "imu"}
            _invalid = [m for m in affected_modalities if m not in _allowed_modalities]
            if _invalid:
                raise ValueError(
                    f"{prefix}.affected_modalities must be a subset of ['uwb','vio','imu'], "
                    f"got invalid entries: {_invalid}"
                )
        # 2026-08-31 协议强化：imu_drop_prob 独立字段（IMU 缺失率上限 5%）。
        # 当 affected_modalities 含 imu 时，imu_drop_prob 区间上界必须 ≤ 0.05。
        # IMU 是时间基准，缺失过高导致状态发散，协议层硬约束 ≤ 5%。
        if affected_modalities and "imu" in affected_modalities:
            imu_drop_prob = payload.get("imu_drop_prob")
            if imu_drop_prob is not None:
                if isinstance(imu_drop_prob, (list, tuple)) and len(imu_drop_prob) == 2:
                    lo, hi = float(imu_drop_prob[0]), float(imu_drop_prob[1])
                    if lo < 0.0 or lo > 0.05:
                        raise ValueError(
                            f"{prefix}.imu_drop_prob interval lower bound must be in [0, 0.05] "
                            f"(IMU is time reference, max 5% loss), got {lo}"
                        )
                    if hi > 0.05:
                        raise ValueError(
                            f"{prefix}.imu_drop_prob interval upper bound must be <= 0.05 "
                            f"(IMU is time reference, max 5% loss), got {hi}"
                        )
                elif is_real(imu_drop_prob):
                    if float(imu_drop_prob) > 0.05:
                        raise ValueError(
                            f"{prefix}.imu_drop_prob must be <= 0.05 "
                            f"(IMU is time reference, max 5% loss), got {imu_drop_prob}"
                        )
        # 2026-09-01 补齐：M 轴成簇缺失持续时间区间(s)。
        # M0=0s baseline，M1/M2=0.3-2s，M3=1-3s。
        cluster_duration_range_s = payload.get("cluster_duration_range_s")
        if isinstance(cluster_duration_range_s, (list, tuple)) and len(cluster_duration_range_s) >= 2:
            lo, hi = float(cluster_duration_range_s[0]), float(cluster_duration_range_s[1])
            if lo < 0.0:
                raise ValueError(f"{prefix}.cluster_duration_range_s lower bound must be non-negative, got {lo}")
            if hi > 5.0:
                raise ValueError(f"{prefix}.cluster_duration_range_s upper bound must be <= 5.0s, got {hi}")
            if lo > hi:
                raise ValueError(f"{prefix}.cluster_duration_range_s lower bound must be <= upper bound, got {cluster_duration_range_s}")


def _validate_axis_monotonicity(axis: str, levels: Mapping[str, Mapping[str, Any]]) -> None:
    """校验轴层级参数的单调递增约束。

    对于 A/N/V 轴，各档位的退化参数应随档位递增而递增（退化程度加深）。
    对于 G 轴，geom_condition 应随档位递增而递增（几何质量递减）。
    对于 K 轴，anchor_count >= 3（K0/K1=4 锚好几何，K3=3 锚欠定退化）。
    对于 M 轴，modality_drop_prob 应随档位递增而递增（缺失程度加深）。

    参数：
        axis: 轴名（A/N/V/G/K/M）。
        levels: 该轴所有层级的参数映射（不含元数据键）。

    异常：
        ValueError: 参数不满足单调递增约束时抛出。
    """
    level_names = [k for k in levels if k not in _AXIS_METADATA_KEYS]  # 保持 YAML 原有顺序（语义序，未按字典序排序）
    if len(level_names) < 2:
        return  # 少于2个层级无法比较单调性。

    if axis == "A":
        # offset_ms 上界单调递增校验（offset_ms 是 [下界, 上界] 列表）。
        offset_upper_values = []
        for ln in level_names:
            v = levels[ln].get("offset_ms")
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                offset_upper_values.append(float(v[1]))
        if len(offset_upper_values) == len(level_names) and offset_upper_values != sorted(offset_upper_values):
            raise ValueError(
                f"scene axis protocol axis {axis} parameter offset_ms upper bound must be monotonically "
                f"increasing across levels, got {offset_upper_values}"
            )
        # A轴：offset_ms上界、jitter_ms、burst_missing_prob、cross_modal_skew_ms、clock_drift_ppm 应递增
        for param in ("jitter_ms", "burst_missing_prob", "cross_modal_skew_ms", "clock_drift_ppm"):
            values = []
            for ln in level_names:
                v = levels[ln].get(param)
                if is_real(v):
                    values.append(float(v))
            if len(values) == len(level_names) and values != sorted(values):
                raise ValueError(
                    f"scene axis protocol axis {axis} parameter {param} must be monotonically "
                    f"increasing across levels, got {values}"
                )
    elif axis == "N":
        # N 轴：nlos_ratio / bias_strength_m / nlos_noise_std_m 支持区间 [low, high]。
        # 区间化时：下界 lo 和上界 hi 均需分别单调递增。
        for param in ("nlos_ratio", "bias_strength_m", "nlos_noise_std_m"):
            upper_values = []
            lower_values = []
            for ln in level_names:
                v = levels[ln].get(param)
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    lower_values.append(float(v[0]))
                    upper_values.append(float(v[1]))
                elif is_real(v):
                    lower_values.append(float(v))
                    upper_values.append(float(v))
            if len(lower_values) == len(level_names):
                if lower_values != sorted(lower_values):
                    raise ValueError(
                        f"scene axis protocol axis {axis} parameter {param} lower bound must be monotonically "
                        f"increasing across levels, got lower={lower_values}"
                    )
                if upper_values != sorted(upper_values):
                    raise ValueError(
                        f"scene axis protocol axis {axis} parameter {param} upper bound must be monotonically "
                        f"increasing across levels, got upper={upper_values}"
                    )
    elif axis == "V":
        # tracked_features_range 下界单调递减校验（特征数越少退化越严重）。
        tf_lower_values = []
        for ln in level_names:
            v = levels[ln].get("tracked_features_range")
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                tf_lower_values.append(float(v[0]))
        if len(tf_lower_values) == len(level_names) and tf_lower_values != sorted(tf_lower_values, reverse=True):
            raise ValueError(
                f"scene axis protocol axis {axis} parameter tracked_features_range lower bound must be monotonically "
                f"decreasing across levels, got {tf_lower_values}"
            )
        for param in ("reproj_err_max", "blackout_prob", "drift_bias_sigma_mps"):
            values = []
            for ln in level_names:
                v = levels[ln].get(param)
                try:
                    values.append(_interval_midpoint(v))  # HIGH-5 修复: 兼容 list/tuple 区间值, 取中点参与单调性比较。
                except (TypeError, ValueError, KeyError):
                    pass  # 缺字段或格式错误跳过, 由单档校验兜底。
            if len(values) == len(level_names) and values != sorted(values):
                raise ValueError(
                    f"scene axis protocol axis {axis} parameter {param} must be monotonically "
                    f"increasing across levels, got {values}"
                )
    elif axis == "K":
        # 2026-08-31：K 轴承载 anchor_count（K0/K1/K3 全档固定 4 锚，文档「五轴档位协议定义」硬约束「锚数全档固定 4」）
        # 和 geom_condition（单调）。
        # anchor_count = 4（4锚布局：K0 对称/K1 非对称/K3 近共线退化），不校验单调性；
        # geom_condition 单调递增。
        geom_values = []
        for ln in level_names:
            v = levels[ln].get("geom_condition")
            if isinstance(v, (list, tuple)) and len(v) == 2:
                geom_values.append(float(v[1]))  # 区间上界参与单调性
            elif is_real(v):
                geom_values.append(float(v))
        if len(geom_values) == len(level_names) and geom_values != sorted(geom_values):
            raise ValueError(
                f"scene axis protocol axis {axis} parameter geom_condition must be monotonically "
                f"increasing across levels, got {geom_values}"
            )
    elif axis == "M":
        # M 轴：modality_drop_prob 应随档位递增而递增（缺失程度加深）。
        # 支持区间 [low, high]，下界和上界都需单调递增。
        upper_values = []
        lower_values = []
        for ln in level_names:
            v = levels[ln].get("modality_drop_prob")
            if isinstance(v, (list, tuple)) and len(v) == 2:
                lower_values.append(float(v[0]))
                upper_values.append(float(v[1]))
            elif is_real(v):
                lower_values.append(float(v))
                upper_values.append(float(v))
        if len(lower_values) == len(level_names):
            if lower_values != sorted(lower_values):
                raise ValueError(
                    f"scene axis protocol axis {axis} parameter modality_drop_prob lower bound must be "
                    f"monotonically increasing across levels, got lower={lower_values}"
                )
            if upper_values != sorted(upper_values):
                raise ValueError(
                    f"scene axis protocol axis {axis} parameter modality_drop_prob upper bound must be "
                    f"monotonically increasing across levels, got upper={upper_values}"
                )


def get_protocol_axes(cfg: Any) -> Mapping[str, Any]:
    """从协议配置中提取并校验轴定义。

    校验内容包括：配置必须是映射、axes 必须是映射、不允许未知轴、
    不允许缺失轴、每个轴的负载必须是映射、每个层级的负载也必须是映射。

    §4.4 协议纪律（前提指导.md L1081-1091）：本函数 L463-474 通过 _AXIS_LEVEL_PARAM_KEYS
    白名单（L68-79）拒绝写入 §4.4 默认关闭的 11 项 UWB 衍生观测量字段名
    （differential_range / range_diff / doppler / radial_velocity / carrier_phase /
    integer_ambiguity / antenna_selection / antenna_diversity / reference_transponder /
    golden_node / aoa / pdoa / tdoa / cir_corrected / peak_corrected / sliding_median /
    rolling_mean / exponential_smooth / outlier_clip / soft_clip / rf_switch /
    antenna_switch_event），任一写入 yaml 即抛 ValueError，无需协议层外单独审核。

    参数：
        cfg: 协议配置。

    返回：
        Mapping[str, Any]: 校验通过的轴定义映射。

    异常：
        TypeError: 配置或轴负载类型不正确时抛出。
        ValueError: 存在未知轴、缺失轴或写入 §4.4 默认关闭字段名时抛出。
    """
    if not isinstance(cfg, Mapping):  # 协议配置必须是映射。
        raise TypeError("scene axis protocol must be a mapping")
    _validate_protocol_version(cfg, required=False)  # 校验版本号（不强制存在）。
    raw_axes = {} if "axes" not in cfg else cfg.get("axes")  # 读取 axes 段，缺失时为空字典。
    if not isinstance(raw_axes, Mapping):  # axes 必须是映射。
        raise TypeError("scene axis protocol axes must be a mapping")
    unknown = [axis for axis in raw_axes if axis not in _AXES]  # 找出未知轴。
    if unknown:  # 不允许未知轴。
        raise ValueError(f"scene axis protocol contains unknown axes: {unknown}")
    axes = copy.deepcopy(raw_axes)  # 深拷贝，隔离嵌套可变引用，防止返回值与输入共享引用。
    missing = [axis for axis in _AXES if axis not in axes]  # 找出缺失轴。
    if missing:  # 不允许缺失轴。
        raise ValueError(f"scene axis protocol missing axes: {missing}")
    non_mapping_axes = [  # 找出负载不是映射的轴。
        axis for axis, axis_payload in axes.items() if not isinstance(axis_payload, Mapping)
    ]
    if non_mapping_axes:  # 轴负载必须是映射。
        raise TypeError(
            f"scene axis protocol axis payload must be a mapping: {non_mapping_axes[0]}"
        )
    for axis, axis_payload in axes.items():  # 逐轴检查层级负载。
        non_mapping_levels = [  # 找出负载不是映射的层级（排除元数据键）。
            level for level, level_payload in axis_payload.items()
            if level not in _AXIS_METADATA_KEYS and not isinstance(level_payload, Mapping)
        ]
        if non_mapping_levels:  # 层级负载必须是映射。
            raise TypeError(
                f"scene axis protocol level payload must be a mapping: {axis}.{non_mapping_levels[0]}"
            )
        # 逐层级检查未知参数键，防止 YAML 中意外注入未声明参数。
        allowed_keys = _AXIS_LEVEL_PARAM_KEYS.get(axis, set())  # 获取该轴允许的参数键白名单。
        for level, level_payload in axis_payload.items():  # 逐层级检查。
            if level in _AXIS_METADATA_KEYS:  # 跳过元数据键。
                continue
            if not isinstance(level_payload, Mapping):  # 非映射类型已在上面校验过。
                continue
            unknown_keys = [k for k in level_payload if k not in allowed_keys]  # 找出未知参数键。
            if unknown_keys:  # 不允许未知参数键。
                raise ValueError(
                    f"scene axis protocol axis {axis} level {level} contains unknown keys: "
                    f"{unknown_keys}; expected only {sorted(allowed_keys)}"
                )
            # 数值有限性校验：拒绝 NaN 穿透协议层。
            # H14c-2 真改：原 L499/L506 含死代码 `not is_real(param_value)`（在 is_real(param_value)
            # 已为 True 的分支里永远 False），且 Inf != Inf 为 False 导致 Inf 漏检。
            # Inf 实际由 _validate_axis_param_semantic_range 兜底拒绝
            # （如 bias_strength_m <= 50.0 上界 L227-230），故本段只负责 NaN 拒绝。
            for param_key, param_value in level_payload.items():
                if param_key == "label":  # label 是字符串，跳过数值校验。
                    continue
                if is_real(param_value):  # 标量数值。
                    if param_value != param_value:  # NaN != NaN 是唯一成立的 True 条件，Inf != Inf 为 False 不触发。
                        raise ValueError(
                            f"scene axis protocol {axis}.{level}.{param_key} must be finite, "
                            f"got {param_value!r}"
                        )
                elif isinstance(param_value, (list, tuple)):  # 列表/元组数值。
                    for i, item in enumerate(param_value):
                        if is_real(item) and item != item:  # is_real 排除 bool/None，item != item 仅 NaN 触发。
                            raise ValueError(
                                f"scene axis protocol {axis}.{level}.{param_key}[{i}] must be finite, "
                                f"got {item!r}"
                            )
            # 语义范围校验：各轴参数的物理约束。
            _validate_axis_param_semantic_range(axis, level, level_payload)
    # 单调递增校验：确保各轴参数随档位递增而递增。
    for axis, axis_payload in axes.items():
        _validate_axis_monotonicity(axis, axis_payload)
    return axes  # 返回校验通过的轴定义。


@lru_cache(maxsize=1)  # 只缓存一份，因为冻结合同不会变。
def _load_frozen_scene_axis_protocol_snapshot() -> dict[str, Any]:
    """加载仓库拥有的冻结场景轴协议快照（只加载一次）。

    从默认路径加载 scene_axis_protocol.yaml，校验其结构后缓存。
    后续所有校验都与此快照做精确比较。

    注意：lru_cache 缓存不感知文件系统变更，修改 YAML 后需重启进程才能生效。

    返回：
        dict[str, Any]: 冻结场景轴协议快照。
    """
    cfg = load_yaml_config(_DEFAULT_PATH)  # 加载 YAML 配置。
    if not isinstance(cfg, Mapping):  # 配置必须是映射。
        raise TypeError("scene axis protocol must be a mapping")
    normalized_cfg = copy.deepcopy(cfg)  # 深拷贝，隔离 YAML 加载器的可变引用，防止 lru_cache 缓存被篡改。
    _validate_protocol_version(normalized_cfg, required=True)  # 加载快照时版本号必须存在。
    get_protocol_axes(normalized_cfg)  # 校验轴定义（返回值用于内部校验，此处只需校验效果）。
    return normalized_cfg  # 返回冻结快照。


def _deep_equal_with_nan_check(a: Any, b: Any) -> bool:
    """递归比较两个对象是否相等，正确处理 NaN（NaN == NaN 视为相等）。

    用于冻结协议快照比较，避免 Python 默认的 NaN != NaN 语义导致误判。

    H16a 真改：原 L557 `isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))`
    将 list-tuple 混合视为同类型比较（实际场景 yaml 不会产生 tuple，但理论缺陷
    应消除）。新逻辑要求 type(a) == type(b)，list-tuple 混合走 L561 type 不一致
    分支被 L565 is_real 兜底拒绝（list/tuple 不注册 numbers.Real）。
    """
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:  # 两个都是 NaN。
            return True
        return a == b
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_equal_with_nan_check(a[k], b[k]) for k in a)
    # H16a: 严格类型一致才走 list/tuple 递归分支，list-tuple 混合走 L561 type 不一致。
    if type(a) is type(b) and isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_deep_equal_with_nan_check(ai, bi) for ai, bi in zip(a, b))
    if type(a) is not type(b):
        # 允许 int 子类型互比（如 np.int64 vs int），但拒绝 int vs float / bool vs int
        if is_bool_like(a) or is_bool_like(b):
            return False  # bool 不应与 int/float 等价
        if is_real(a) and is_real(b):
            return a == b  # 数值类型允许跨精度比较（int vs float vs np.float64 等）
        return False
    return a == b


def _find_first_diff_path(a: Any, b: Any, path: str = "") -> str:
    """递归查找两个对象首次出现差异的路径，用于诊断冻结快照比较失败。"""
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return ""  # 双 NaN 视为相等
        if a == b:
            return ""
        return path or "root"
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        all_keys = set(a.keys()) | set(b.keys())
        for k in sorted(all_keys):
            if k not in a:
                return f"{path}.{k}" if path else f".{k}"
            if k not in b:
                return f"{path}.{k}" if path else f".{k}"
            sub = _find_first_diff_path(a[k], b[k], f"{path}.{k}" if path else f".{k}")
            if sub:
                return sub
        return ""
    # H16b 真改：同 _deep_equal_with_nan_check L557-560 严格类型一致，list-tuple 混合走 L598 type 不一致。
    if type(a) is type(b) and isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}[len:{len(a)}vs{len(b)}]"
        for i, (ai, bi) in enumerate(zip(a, b)):
            sub = _find_first_diff_path(ai, bi, f"{path}[{i}]")
            if sub:
                return sub
        return ""
    if type(a) is not type(b):
        if is_bool_like(a) or is_bool_like(b):
            return f"{path}[type:{type(a).__name__}vs{type(b).__name__}]"
        if is_real(a) and is_real(b):
            if a == b:
                return ""
            return f"{path}[value:{a}vs{b}]"
        return f"{path}[type:{type(a).__name__}vs{type(b).__name__}]"
    if a != b:
        return f"{path}[value:{a}vs{b}]"
    return ""


def _validate_frozen_scene_axis_protocol_cfg(cfg: Any) -> dict[str, Any]:
    """校验场景轴协议是否与仓库冻结快照完全一致。

    先对输入配置做结构校验，然后与冻结快照做精确比较。
    任何漂移都会导致校验失败。

    参数：
        cfg: 待校验的场景轴协议配置。

    返回：
        dict[str, Any]: 校验通过的配置。

    异常：
        TypeError: 配置类型不正确时抛出。
        ValueError: 配置与冻结快照不一致时抛出。
    """
    if not isinstance(cfg, Mapping):  # 配置必须是映射。
        raise TypeError("scene axis protocol must be a mapping")
    normalized_cfg = copy.deepcopy(cfg)  # 深拷贝，隔离嵌套可变引用，防止外部修改影响内部状态。
    _validate_protocol_version(normalized_cfg, required=True)  # 校验版本号。
    get_protocol_axes(normalized_cfg)  # 校验轴定义（返回值用于内部校验，此处只需校验效果）。
    frozen_snapshot = _load_frozen_scene_axis_protocol_snapshot()  # 加载冻结快照。
    # 逐字段递归比较，避免 NaN 陷阱（NaN != NaN 导致误判）。
    if not _deep_equal_with_nan_check(normalized_cfg, frozen_snapshot):
        diff_path = _find_first_diff_path(normalized_cfg, frozen_snapshot)
        raise ValueError(
            f"scene axis protocol must match the frozen repository snapshot exactly; "
            f"first diff at: {diff_path or 'unknown'}"
        )
    return normalized_cfg  # 一致则返回。


def _resolve_protocol_cfg(protocol_cfg: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """从调用者输入或冻结默认路径解析并校验场景轴协议。

    如果调用者未提供协议配置，则从默认路径加载冻结协议；
    如果提供了，则直接校验其是否与冻结合同一致。

    参数：
        protocol_cfg: 可选的外部协议配置字典。

    返回：
        tuple[dict[str, Any], dict[str, Any]]: (校验通过的协议配置, 校验通过的轴定义字典)。
    """
    if protocol_cfg is None:  # 未提供时从默认路径加载。
        cfg = load_scene_axis_protocol()
    else:
        cfg = _validate_frozen_scene_axis_protocol_cfg(protocol_cfg)  # 提供时校验一致性。
    axes = get_protocol_axes(cfg)  # 提取轴定义（校验已通过，此处提取数据）。
    return cfg, axes


def load_scene_axis_protocol(protocol_path: str | Path | None = None) -> dict[str, Any]:
    """加载冻结场景轴协议并校验其与冻结合同一致。

    参数：
        protocol_path: 协议文件路径，默认使用 configs/base/scene_axis_protocol.yaml。

    返回：
        dict[str, Any]: 校验通过的场景轴协议配置。

    异常：
        ValueError: 路径为空白字符串或协议校验失败时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"protocol_path": str(protocol_path) if protocol_path else None}, "load_scene_axis_protocol 入口参数")
    if protocol_path is None:  # 未指定路径时使用默认路径，返回冻结快照的深拷贝以防止调用者篡改缓存。
        return copy.deepcopy(_load_frozen_scene_axis_protocol_snapshot())
    else:
        if not str(protocol_path).strip():  # 空白路径不允许（兼容字符串和 Path 对象）。
            raise ValueError("scene axis protocol path must not be blank")
        path = Path(protocol_path).resolve()  # 转为 Path 对象并规范化，消除 .. 等路径穿越。
        if path.suffix.lower() not in ('.yaml', '.yml'):  # 只允许 YAML 文件。
            raise ValueError(f"scene axis protocol path must be a YAML file, got {path.suffix!r}")
        project_root = find_project_root().resolve()  # 项目根目录的绝对路径。
        if not path.is_relative_to(project_root):  # 路径必须在项目根目录下（is_relative_to 防止兄弟目录前缀绕过）。
            raise ValueError(
                f"scene axis protocol path must be within project root {project_root}, got {path}"
            )
    cfg = load_yaml_config(path)  # 加载 YAML 配置。
    return _validate_frozen_scene_axis_protocol_cfg(cfg)  # 校验并返回。


def resolve_axis_level(
    axis: str,
    level: str,
    protocol_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """解析单个轴层级的参数。

    从协议配置中读取指定轴和层级的参数表，并在返回值中附加
    axis 和 level 字段以便下游识别来源。

    参数：
        axis: 轴名（A/N/V/G/K/M）。
        level: 层级名（如 A0、A1 等）。
        protocol_cfg: 可选的外部协议配置字典。

    返回：
        dict[str, Any]: 该轴层级的参数字典，额外包含 axis 和 level 字段。

    异常：
        TypeError: axis 或 level 不是字符串时抛出。
        ValueError: axis 或 level 为空白时抛出。
        KeyError: 轴名或层级名不存在时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"axis": axis, "level": level}, "resolve_axis_level 入口参数")
    _, axes = _resolve_protocol_cfg(protocol_cfg)  # 解析并校验协议，获取轴定义（cfg 未使用）。
    if not is_string_like(axis):  # 轴名必须是字符串（含 numpy.str_）。
        raise TypeError(f"axis must be a string, got {type(axis).__name__}")
    if not is_string_like(level):  # 层级名必须是字符串（含 numpy.str_）。
        raise TypeError(f"level must be a string, got {type(level).__name__}")
    normalized_axis = axis.strip()  # 去除首尾空白。
    normalized_level = level.strip()  # 去除首尾空白。
    if not normalized_axis:  # 轴名不能为空白。
        raise ValueError("axis must not be blank")
    if not normalized_level:  # 层级名不能为空白。
        raise ValueError("level must not be blank")
    if normalized_axis not in axes:  # 轴名不存在。
        raise KeyError(f"Unknown axis: {normalized_axis}")
    if normalized_level in _AXIS_METADATA_KEYS:  # 误传元数据键作为层级名时给出清晰错误。
        raise ValueError(f"{normalized_level} is a metadata key, not a valid level for axis {normalized_axis}")
    if normalized_level not in axes[normalized_axis]:  # 层级名不存在。
        raise KeyError(f"Unknown level for axis {normalized_axis}: {normalized_level}")
    raw_payload = axes[normalized_axis][normalized_level]  # 读取原始参数表。
    if not isinstance(raw_payload, Mapping):  # 参数表必须是映射。
        raise TypeError(
            f"scene axis protocol level payload must be a mapping: {normalized_axis}.{normalized_level}"
        )
    payload = copy.deepcopy(dict(raw_payload))  # 深拷贝，隔离嵌套可变引用（如列表参数）。
    payload["axis"] = normalized_axis  # 附加轴名，便于下游识别来源。
    payload["level"] = normalized_level  # 附加层级名，便于下游识别来源。
    return payload  # 返回参数字典。


def attach_scene_parameters(
    axes: dict[str, Any],
    protocol_cfg: dict[str, Any] | None = None,
) -> SceneParameters:
    """展开所有轴的参数并扁平化。

    对每个轴调用 resolve_axis_level 获取参数，然后生成两种格式：
    - axes: 轴名到参数字典的映射（保留层级结构）
    - flat: 扁平化的参数字典（键格式为 {轴名}_{参数名}）

    返回 SceneParameters 结构化对象，明确 axes 和 flat 的类型，
    下游可通过 .axes 和 .flat 访问，不再依赖裸 dict 的键名约定。

    参数：
        axes: 轴名到层级名的映射字典，必须包含所有六个轴。
        protocol_cfg: 可选的外部协议配置字典。

    返回：
        SceneParameters: 包含 axes 和 flat 的结构化对象。

    异常：
        KeyError: 存在未知轴时抛出。
        ValueError: 缺失轴时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"axes": axes}, "attach_scene_parameters 入口参数")
    cfg, axes_data = _resolve_protocol_cfg(protocol_cfg)  # 解析并校验协议，获取 cfg 和轴定义。
    unknown_axes = [axis for axis in axes if axis not in _AXES]  # 找出未知轴。
    if unknown_axes:  # 不允许未知轴。
        raise KeyError(f"Unknown axis: {unknown_axes[0]}")
    missing_axes = [axis for axis in _AXES if axis not in axes]  # 找出缺失轴。
    if missing_axes:  # 不允许缺失轴。
        raise ValueError(f"missing scene axes: {missing_axes}")

    resolved: dict[str, Any] = {}  # 存放每个轴的参数字典。
    for axis in _AXES:  # 按固定轴顺序遍历。
        resolved[axis] = resolve_axis_level(axis, axes[axis], cfg)  # 解析每个轴的参数。
    flat = {  # 生成扁平化参数字典。
        f"{axis}_{key}": value  # 键格式为 {轴名}_{参数名}。
        for axis, payload in resolved.items()  # 遍历每个轴的参数。
        for key, value in payload.items()  # 遍历参数中的每个键值对。
        if key not in {"axis", "level"}  # 排除附加的 axis 和 level 字段。
    }
    # 提取轴级元数据（如 V 轴的 range_semantics），保留不属于具体层级的轴级声明。
    axis_metadata: dict[str, dict[str, Any]] = {}
    for axis in _AXES:
        axis_payload = axes_data.get(axis, {})
        if isinstance(axis_payload, Mapping):
            metadata = {k: v for k, v in axis_payload.items() if k in _AXIS_METADATA_KEYS}
            if metadata:
                axis_metadata[axis] = dict(metadata)
    return SceneParameters(axes=resolved, flat=flat, axis_metadata=axis_metadata)


@lru_cache(maxsize=1)
def get_nominal_levels() -> dict[str, str]:
    """获取每个轴的正常等级名。

    A/N/V/M 轴的正常等级为第一个非元数据键的等级（A0/N0/V0/M0）。
    K 轴（2026-08-31 重制定）优先 K0（好几何基线），其次 K1（中等），最后回退到中间值。

    用于判断场景是否为"正常场景"，避免下游硬编码等级名。
    如果协议中正常等级名变更，此函数自动跟随。

    返回：
        dict[str, str]: 轴名到正常等级名的映射，如 {"A": "A0", "N": "N0", ..., "K": "K0"}。
    """
    cfg = load_scene_axis_protocol()
    axes = get_protocol_axes(cfg)
    nominal: dict[str, str] = {}
    for axis in _AXES:
        axis_payload = axes.get(axis, {})
        level_names = sorted([k for k in axis_payload if k not in _AXIS_METADATA_KEYS])  # 按字典序排列，与 _validate_axis_monotonicity 一致
        if not level_names:
            continue
        if axis == "K":
            # 2026-08-31：K 轴承载几何分布，优先 K0（4 锚对称矩形四角，好几何基线），
            # 其次 K1（中等），最后回退到中间值；禁止默认严重退化的 K3。
            if "K0" in level_names:
                nominal[axis] = "K0"
            elif "K1" in level_names:
                nominal[axis] = "K1"
            else:
                nominal[axis] = level_names[len(level_names) // 2]
        else:  # A/N/V/M 轴取第一个等级作为正常等级。
            nominal[axis] = level_names[0]
    # 确保五轴齐全——如果某轴没有非元数据层级，说明协议定义异常。
    if set(nominal.keys()) != set(_AXES):
        missing = set(_AXES) - set(nominal.keys())
        raise ValueError(f"scene axis protocol axes {missing} have no valid levels")
    return nominal


# ---------------------------------------------------------------------------
# 向后兼容别名：保留旧名供已有代码使用。
# ---------------------------------------------------------------------------
_get_protocol_axes = get_protocol_axes
