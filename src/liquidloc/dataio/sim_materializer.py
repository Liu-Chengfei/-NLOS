"""仿真数据物化器：从 fixture 种子生成可复现的仿真原始数据集。

本模块负责把 tests/fixtures/datasets/miluv 下的短序列 fixture（种子数据）
通过平移、旋转、镜像、弯曲等几何变换，铺叠（tile）成多条长周期仿真序列，
并为每条序列重新生成 IMU、UWB、VIO 和真值（GT）流文件。物化过程完全
确定性——只要 fixture 和 SimSequenceSpec 不变，输出就严格可复现。

核心概念：
- **SimSequenceSpec**：描述一条仿真序列的变换参数（平移、旋转、镜像、
  弯曲强度、周期数、周期间隔），是物化的输入契约。
- **SimNoiseSpec**：描述基础传感器高斯噪声参数（UWB 测距标准差、IMU
  加速度/角速度标准差、VIO 位移/航向标准差）。注意：场景轴退化（A/N/V）
  不在此层实现，由 runtime 的 core_pipeline 施加。
- **铺叠（tiling）**：把一条短 fixture 序列按 cycle_count 次重复铺开，
  每个 cycle 之间插入 cycle_gap_s 秒间隔，GT 位置按终端速度外推，
  保证轨迹在 cycle 边界处连续。
- **残差保留**：物化时先从 fixture 原始数据中提取传感器观测与几何真值
  之间的残差（residual），再在变换后的新序列上叠加这些残差，使仿真数据
  保留真实传感器的系统偏差特征。

上游依赖：
- tests/fixtures/datasets/miluv/ 下的 fixture JSON 文件（imu.json, uwb.json,
  vio.json, gt.json, anchor_layout.json）
- liquidloc.common.angle_utils（角度差和角度归一化）
- liquidloc.sensors.anchor_model / uwb_model（锚点查找和测距预测）

下游调用者：
- 准备流程脚本（通过 materialize_sim_raw 生成仿真原始数据目录）
- 烟雾测试脚本（通过 can_materialize_sim_raw 检查目标目录是否可写入）

核心变量：
- DEFAULT_SIM_SEQUENCE_SPECS：默认的仿真序列规格元组
- DEFAULT_SIM_NOISE_SPEC：默认的传感器噪声参数
- ZERO_SIM_NOISE_SPEC：零噪声参数（用于无噪声对照实验）
"""

from __future__ import annotations

from bisect import bisect_left
from copy import deepcopy
from dataclasses import asdict, dataclass, field  # §5.3: 重新引入 field，因 SimNoiseSpec.uwb_outage_segments/vio_outage_segments 用 field(default_factory=tuple) 表示空 tuple 默认值。R11-B LOW-1 移除时这两字段尚未存在。
import hashlib
import logging
import math
from pathlib import Path
import random
import shutil
from typing import Any
import warnings

from liquidloc.common.angle_utils import angle_delta_rad, wrap_angle_rad
from liquidloc.common.gt_utils import normalize_gt_rows  # 真值行归一化的规范实现
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, validate_path_component  # 第 11 轮审查 LOW-2 修复（R11-B LOW-2）：移除未使用的 is_real, is_string_like 导入（已 Grep 全文件无调用；sim_materializer 内数值/字符串类型判断走 coerce_finite_scalar + validate_path_component，不需要 is_real/is_string_like）。
from liquidloc.common.config_utils import find_project_root, load_yaml_config
from liquidloc.common.io_utils import read_json, write_json
from liquidloc.protocol.scene_axis_protocol import load_scene_axis_protocol, get_nominal_levels
from liquidloc.scenarios.geometry_levels import build_anchor_layout, project_anchor_layout_to_reference
from liquidloc.scenarios.geometry_motion_envelope import assert_geometry_motion_envelope
from liquidloc.scenarios.protocol_trajectory import generate_protocol_gt_rows
from liquidloc.sensors.anchor_model import build_anchor_lookup
from liquidloc.sensors.uwb_model import predict_range_to_anchor


# D-7 patch: SimNoiseSpec 缺失字段显式 logger.warning 审计.
# 给 fixture 字段被静默跳过的位置加 logger 审计线索, 保持向后兼容 (只警告不抛错).
# _logger 为模块级 logger, 调用方未配置 handler 时默认 WARN 级别会冒到 root logger,
# 不引入新的 logging.basicConfig 以避免污染其他模块的日志配置.
_logger = logging.getLogger(__name__)


# fixture 种子序列编号白名单，只有这些编号对应的 fixture 目录才会被加载
_FIXTURE_BASE_SEQ_IDS = ("mini_seq", "mini_seq_02", "mini_seq_03")
# 每条物化序列必须产出的契约文件名
_RAW_CONTRACT_FILES = ("imu.json", "uwb.json", "vio.json", "gt.json", "anchor_layout.json")
# 默认铺叠周期数
_DEFAULT_CYCLE_COUNT = 128
# 默认周期间隔（秒）
_DEFAULT_CYCLE_GAP_S = 0.10
# 浮点输出精度（小数位数），用于 round(..., 6) 的统一收口
_ROUND_PRECISION = 6
# 安全差分除法的最小非零 dt 阈值（秒），低于此值视为退化
_SAFE_DIFF_EPSILON = 1e-6
# 曲线弯曲/角度重算的数值零阈值
_GEOMETRIC_ZERO_EPSILON = 1e-12
# FFN (1/f flicker noise) 多尺度 Wiener 叠加的最大尺度数上限，防止极端长序列下 K 过大。
_FFN_MAX_SCALES = 12


def _normalize_outage_segments(
    segments: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """§5.3: 排序 + 合并重叠 outage 段，返回闭区间规范形式 ((start, end), ...)。

    输入必须已经是 __post_init__ 校验过的合法段（端点非负有限、end > start）。
    本函数：1) 按 start 升序；2) 相邻或重叠段合并（取 max(end_prev, end_curr)）；
    3) 返回 tuple of tuple，与 dataclass field 声明兼容。
    """
    if not segments:
        return ()
    sorted_segs = sorted(segments, key=lambda s: (s[0], s[1]))
    merged: list[list[float]] = []
    for start_s, end_s in sorted_segs:
        if merged and start_s <= merged[-1][1]:
            # 重叠或相邻：合并到当前段（end 取较大值）
            merged[-1][1] = max(merged[-1][1], end_s)
        else:
            merged.append([start_s, end_s])
    return tuple((float(s), float(e)) for s, e in merged)


def _is_timestamp_in_outage(
    timestamp: float,
    segments: tuple[tuple[float, float], ...],
) -> bool:
    """§5.3: 二分查找判定 timestamp 是否落在 outage 段内（含端点）。"""
    if not segments:
        return False
    # 段已由 _normalize_outage_segments 排序保证 start 单调非降
    lo, hi = 0, len(segments) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start_s, end_s = segments[mid]
        if timestamp < start_s:
            hi = mid - 1
        elif timestamp > end_s:
            lo = mid + 1
        else:
            return True
    return False


# sim_noise_spec.yaml 中的字段名集合（与 SimNoiseSpec 字段一一对齐）
_SIM_NOISE_SPEC_FIELD_NAMES: tuple[str, ...] = (
    "uwb_range_std_m",
    "imu_ax_std",
    "imu_ay_std",
    "imu_gz_std",
    "imu_accel_bias_instability",
    "imu_gyro_bias_instability",
    "imu_accel_rrw_mps2_per_sqrt_s",
    "imu_gyro_rrw_rads_per_sqrt_s",
    "vio_dx_std",
    "vio_dy_std",
    "vio_dyaw_std",
    "vio_scale_drift_rate",
    "base_seed",
    # §5.3 (R*-D-?): 短时 IMU 主导段——UWB 全锚 NLOS 持续段 + VIO 持续中断段.
    # 与统计级 N3 (nlos_ratio) / V3 (blackout_prob) 不同，本字段注入的是时序段而非
    # 逐观测脉冲. 默认空 tuple, 即"无 outage"; 配置时为 [(start_s, end_s), ...].
    # 时戳语义与 transformed_gt_rows 一致 (序列内累计秒), 边界闭合 [start, end].
    "uwb_outage_segments",
    "vio_outage_segments",
)
# 论文级 sim raw 基线合同：G 轴和 K 轴使用协议定义的正常等级名。


def _resolve_sim_noise_yaml_path() -> Path:
    """返回 configs/base/sim_noise_spec.yaml 的绝对路径。

    使用独立 yaml 文件而非 sensors.yaml，因为 sensors.yaml 的顶级键由
    protocol 层 sensor_contract.py 严格校验（仅允许 *_fields 和
    feature_missing_policy），sim_noise_spec 不属于传感器字段合同。
    """
    return find_project_root() / "configs" / "base" / "sim_noise_spec.yaml"


def _load_sim_noise_defaults(yaml_path: str | Path | None = None) -> dict[str, float | int | tuple[tuple[float, float], ...]]:  # §5.3: 增 outage_segments 字段,返回类型扩展.
    """从 configs/base/sim_noise_spec.yaml 的 sim_noise_spec 段读取协议层冻结默认值。

    若 sim_noise_spec 段缺失或某字段未指定，回退到 SimNoiseSpec dataclass 的硬编码
    默认值（保证 dataclass 在 yaml 不可用时仍可实例化）。

    参数：
        yaml_path: 可选的显式 yaml 路径；为 None 时使用项目默认路径。

    返回：
        字段名 → 默认值的字典，键集合与 _SIM_NOISE_SPEC_FIELD_NAMES 一致。
        数值字段为 float/int；outage_segments 字段为 tuple[tuple[float, float], ...].
    """
    path = Path(yaml_path) if yaml_path is not None else _resolve_sim_noise_yaml_path()
    defaults: dict[str, float | int | tuple[tuple[float, float], ...]] = {}  # §5.3: 类型扩展含 outage_segments
    if not path.is_file():
        return defaults  # yaml 不存在则回退到 dataclass 硬编码默认
    try:
        payload = load_yaml_config(path)
    except (FileNotFoundError, TypeError, ValueError):
        return defaults  # 解析失败则回退，不阻断 sim 物化
    spec_section = payload.get("sim_noise_spec")
    if not isinstance(spec_section, dict):
        return defaults
    for key in _SIM_NOISE_SPEC_FIELD_NAMES:
        raw_value = spec_section.get(key)
        if raw_value is None:
            continue
        # §5.3: outage_segments 走 list-of-pair → tuple-of-tuple 解析路径
        if key in ("uwb_outage_segments", "vio_outage_segments"):
            if not isinstance(raw_value, (list, tuple)):
                _logger.warning(
                    "SimNoiseSpec field %s expected list of (start, end) pairs, got %s; ignoring",
                    key, type(raw_value).__name__,
                )
                continue
            parsed_segments: list[tuple[float, float]] = []
            for seg in raw_value:
                if not isinstance(seg, (list, tuple)) or len(seg) != 2:
                    _logger.warning(
                        "SimNoiseSpec field %s entry must be a 2-tuple (start, end), got %r; skipping entry",
                        key, seg,
                    )
                    continue
                try:
                    start_s = float(seg[0])
                    end_s = float(seg[1])
                except (TypeError, ValueError):
                    _logger.warning(
                        "SimNoiseSpec field %s entry endpoints must be numeric, got %r; skipping entry",
                        key, seg,
                    )
                    continue
                if not (math.isfinite(start_s) and math.isfinite(end_s)):
                    _logger.warning(
                        "SimNoiseSpec field %s entry endpoints must be finite, got %r; skipping entry",
                        key, seg,
                    )
                    continue
                if end_s <= start_s:
                    _logger.warning(
                        "SimNoiseSpec field %s entry must have end > start, got %r; skipping entry",
                        key, seg,
                    )
                    continue
                parsed_segments.append((_normalize_zero(start_s), _normalize_zero(end_s)))
            defaults[key] = tuple(parsed_segments)
            continue
        try:
            if key == "base_seed":
                # base_seed 必须是 int，yaml 解析可能返回 float 或 str
                defaults[key] = int(raw_value)
            else:
                defaults[key] = float(raw_value)
        except (TypeError, ValueError):
            continue  # 跳过不可解析的字段
    return defaults


# 模块加载时一次性读取协议层冻结的 sim_noise_spec 默认值。
# 若 configs/base/sim_noise_spec.yaml 不存在或段缺失，回退到 dataclass 硬编码默认。
_SIM_NOISE_DEFAULTS: dict[str, float | int] = _load_sim_noise_defaults()  # 第 13 轮审查 LOW-1 修复（R13-B L1）：base_seed 为 int，类型注解精确化。R12-B L2 + R13-B L1 两次 Edit 均未持久化，本轮第三次重做并 Read 验证。
_nominal_levels = get_nominal_levels()
_PAPER_SIM_GEOMETRY_LEVEL = _nominal_levels["G"]
_PAPER_SIM_K_LEVEL = _nominal_levels["K"]


def _require_non_empty(items: list, *, name: str) -> None:
    """校验列表非空，否则抛 ValueError。"""
    if not items:
        raise ValueError(f"{name} must be non-empty")


def _require_index_in_bounds(index: int, items: list, *, name: str) -> int:
    """校验索引在列表范围内，否则抛 IndexError 带上下文。"""
    idx = int(index)
    if idx < 0 or idx >= len(items):
        raise IndexError(f"{name} index {idx} out of range for list of length {len(items)}")
    return idx


def _normalize_zero(value: float) -> float:
    """将 -0.0 归一化为 0.0，避免 JSON 序列化产生 "-0.0"。"""
    return 0.0 if value == 0.0 else value


def _round_normalize(value: float, *, precision: int = _ROUND_PRECISION) -> float:
    """round 后归一化 -0.0，统一浮点输出处理。"""
    return _normalize_zero(round(float(value), precision))


@dataclass(frozen=True, slots=True)
class SimSequenceSpec:
    """一条仿真序列的变换参数规格。

    属性：
        seq_id: 序列唯一编号，也用作输出子目录名。
        base_seq_id: 依赖的 fixture 种子序列编号。
        translation_xy: 二维平移偏移量 (x, y)，单位米。
        rotation_rad: 绕原点的旋转角度，单位弧度。
        mirror_x: 是否沿 X 轴做镜像（Y 取反）。
        curve_strength: 弯曲强度参数，正值向上弯、负值向下弯。
        cycle_count: 铺叠周期数，决定序列总时长。
        cycle_gap_s: 周期间隔秒数，影响 cycle 边界处的时间跳跃。
        axes_override: (v2 D-10/D-11/D-12) 场景轴档位覆盖, dict 形如
            {"A":"A1", "N":"N2", "V":"V1", "G":"G1", "K":"K4"}. 默认 None 表示
            缺省时由 _resolve_spec_gk_levels 回落 G1/K4（主表欠定）；axes_override 可覆盖。
        dt_imu_override: (v2 D-13) IMU 采样周期覆盖 (秒). 默认 None 表示用 baseline dt_imu=0.01s (100Hz).
            典型值: 0.005s (200Hz), 0.01s (100Hz), 0.02s (50Hz).
        dt_uwb_override: (v2 D-13) UWB 采样周期覆盖 (秒). 默认 None 表示用 baseline dt_uwb=0.05s (20Hz).
            典型值: 0.02s (50Hz), 0.05s (20Hz), 0.1s (10Hz).
        dt_vio_override: (v2 D-13) VIO 采样周期覆盖 (秒). 默认 None 表示用 baseline dt_vio=0.0333s (30Hz).
            典型值: 0.0167s (60Hz), 0.0333s (30Hz).

    异常：
        TypeError: cycle_count 不是 int、mirror_x 不是 bool、translation_xy 不是二元组时抛出。
        ValueError: cycle_count < 1、cycle_gap_s 非正、rotation_rad/curve_strength
            含 NaN/Inf、seq_id 含路径穿越字符、base_seq_id 不在白名单内时抛出。
    """
    seq_id: str
    base_seq_id: str
    translation_xy: tuple[float, float]
    rotation_rad: float = 0.0
    mirror_x: bool = False
    curve_strength: float = 0.0
    cycle_count: int = _DEFAULT_CYCLE_COUNT
    cycle_gap_s: float = _DEFAULT_CYCLE_GAP_S
    # v2 D-10/D-11/D-12: 场景轴档位覆盖, 让 30 seqs 跨多场景轴组合.
    axes_override: tuple[tuple[str, str], ...] = ()
    # v2 D-13: 传感器频率覆盖, 让 IMU/UWB/VIO 取不同 nyquist 频率.
    dt_imu_override: float | None = None
    dt_uwb_override: float | None = None
    dt_vio_override: float | None = None
    # §8.2 协议轨迹：True 时用 generate_protocol_gt_rows，不再 mini-fixture 铺叠。
    use_protocol_trajectory: bool = True
    duration_s: float = 45.0
    workspace_span_m: float = 20.0
    envelope_profile: str = "main_table"  # main_table | smoke
    allow_high_anchor_count: bool = False

    def __post_init__(self) -> None:
        # seq_id 作为输出子目录名，必须不含路径穿越字符（防止 ../ 逃逸 output_root）
        validate_path_component(self.seq_id, name="seq_id")
        # base_seq_id 必须在 fixture 种子白名单内，禁止任意路径注入
        if self.base_seq_id not in _FIXTURE_BASE_SEQ_IDS:
            raise ValueError(
                f"base_seq_id must be one of {_FIXTURE_BASE_SEQ_IDS}, got {self.base_seq_id!r}"
            )
        # translation_xy 必须是二元组，禁止静默接受错误维度
        if not isinstance(self.translation_xy, tuple) or len(self.translation_xy) != 2:
            raise TypeError(
                f"translation_xy must be a 2-tuple, got {type(self.translation_xy).__name__} "
                f"of length {len(self.translation_xy) if hasattr(self.translation_xy, '__len__') else 'N/A'}"
            )
        # 数值字段必须有限，禁止 NaN/Inf 进入几何变换链
        coerce_finite_scalar(self.translation_xy[0], name="translation_xy[0]")
        coerce_finite_scalar(self.translation_xy[1], name="translation_xy[1]")
        coerce_finite_scalar(self.rotation_rad, name="rotation_rad")
        coerce_finite_scalar(self.curve_strength, name="curve_strength")
        # mirror_x 必须是 bool（接受 numpy bool_），禁止 int/str 隐式真值
        if not is_bool_like(self.mirror_x):
            raise TypeError(
                f"mirror_x must be bool, got {type(self.mirror_x).__name__}: {self.mirror_x!r}"
            )
        object.__setattr__(self, "mirror_x", bool(self.mirror_x))
        # 组7: 校验 cycle_count 为正整数，禁止 int() 静默截断浮点
        if not isinstance(self.cycle_count, int) or isinstance(self.cycle_count, bool):
            raise TypeError(f"cycle_count must be int, got {type(self.cycle_count).__name__}: {self.cycle_count!r}")
        if self.cycle_count < 1:
            raise ValueError(f"cycle_count must be >= 1, got {self.cycle_count}")
        # cycle_gap_s 必须为正有限值（周期间隔 ≤0 会导致时间戳折叠）
        gap = coerce_finite_scalar(self.cycle_gap_s, name="cycle_gap_s")
        if gap <= 0.0:
            raise ValueError(f"cycle_gap_s must be > 0, got {gap}")
        object.__setattr__(self, "cycle_gap_s", float(gap))
        # v2 D-10/D-11/D-12: axes_override 校验 (tuple of (axis_name, level_name)).
        if not isinstance(self.axes_override, tuple):
            raise TypeError(
                f"axes_override must be a tuple of (axis, level) pairs, got {type(self.axes_override).__name__}"
            )
        for pair in self.axes_override:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(f"axes_override entry must be a 2-tuple, got {pair!r}")
            axis_name, level_name = pair
            if not isinstance(axis_name, str) or not isinstance(level_name, str):
                raise TypeError(f"axes_override entry components must be strings, got {pair!r}")
            if axis_name not in ("A", "N", "V", "G", "K", "M"):
                raise ValueError(
                    f"axes_override axis_name must be one of A/N/V/G/K/M, got {axis_name!r}"
                )
            if not level_name:
                raise ValueError(f"axes_override level_name must be non-empty, got {level_name!r}")
        if not is_bool_like(self.use_protocol_trajectory):
            raise TypeError(
                f"use_protocol_trajectory must be bool, got {type(self.use_protocol_trajectory).__name__}"
            )
        object.__setattr__(self, "use_protocol_trajectory", bool(self.use_protocol_trajectory))
        object.__setattr__(
            self,
            "duration_s",
            float(coerce_finite_scalar(self.duration_s, name="duration_s", min_value=1e-3, inclusive=False)),
        )
        object.__setattr__(
            self,
            "workspace_span_m",
            float(
                coerce_finite_scalar(
                    self.workspace_span_m, name="workspace_span_m", min_value=1e-3, inclusive=False
                )
            ),
        )
        if self.envelope_profile not in ("main_table", "smoke"):
            raise ValueError(
                f"envelope_profile must be 'main_table' or 'smoke', got {self.envelope_profile!r}"
            )
        if not is_bool_like(self.allow_high_anchor_count):
            raise TypeError(
                f"allow_high_anchor_count must be bool, got {type(self.allow_high_anchor_count).__name__}"
            )
        object.__setattr__(self, "allow_high_anchor_count", bool(self.allow_high_anchor_count))
        # v2 D-13: dt_imu_override / dt_uwb_override / dt_vio_override 校验.
        for field_name, field_value in (
            ("dt_imu_override", self.dt_imu_override),
            ("dt_uwb_override", self.dt_uwb_override),
            ("dt_vio_override", self.dt_vio_override),
        ):
            if field_value is not None:
                dt_val = coerce_finite_scalar(field_value, name=field_name)
                if dt_val <= 0.0:
                    raise ValueError(f"{field_name} must be > 0, got {dt_val}")
                if dt_val > 1.0:
                    raise ValueError(f"{field_name} must be <= 1.0s (≥1Hz), got {dt_val}")
                object.__setattr__(self, field_name, float(dt_val))


@dataclass(frozen=True, slots=True)
class SimNoiseSpec:
    """仿真基础传感器噪声参数。

    注意：这些参数只控制基础高斯噪声（传感器精度级别），
    不包含场景轴退化（A/N/V 轴的 async/NLOS/visual 退化）。
    场景轴退化由 runtime 的 core_pipeline 施加，不在此层实现。

    协议层冻结：13 项噪声/协方差参数的默认值由 configs/base/sim_noise_spec.yaml
    的 sim_noise_spec 段冻结。当 yaml 缺失或字段未指定时，回退到本
    dataclass 的硬编码默认值。调用方应使用 SimNoiseSpec.from_yaml()
    获取协议层冻结的默认规格，或使用 DEFAULT_SIM_NOISE_SPEC（已通过
    from_yaml 构造）。直接 SimNoiseSpec() 调用仅使用硬编码回退默认，
    适用于 yaml 不可用的极端场景。

    属性：
        uwb_range_std_m: UWB 测距噪声标准差（米）。
        imu_ax_std: IMU x 轴加速度噪声标准差（m/s²）。
        imu_ay_std: IMU y 轴加速度噪声标准差（m/s²）。
        imu_gz_std: IMU z 轴角速度噪声标准差（rad/s）。
        imu_accel_bias_instability: 加速度计 bias 不稳定性（m/s²），
            对应 MATLAB imuSensor 的 Accelerometer.BiasInstability 参数。
            模拟 Allan 方差中频段（FFN, flicker frequency noise），
            通过多尺度 Wiener 叠加生成 1/f 噪声序列，逐样本叠加到 IMU bias。
        imu_gyro_bias_instability: 陀螺仪 bias 不稳定性（rad/s），
            对应 MATLAB imuSensor 的 Gyroscope.BiasInstability 参数。
            模拟 Allan 方差中频段（FFN, flicker frequency noise），
            通过多尺度 Wiener 叠加生成 1/f 噪声序列，逐样本叠加到 IMU bias。
        imu_accel_rrw_mps2_per_sqrt_s: 加速度计 Rate Random Walk 系数
            （m/s² · √s⁻¹），对应 Allan 方差低频段（RWFN, random walk
            frequency noise）。bias 在序列内按 Wiener 过程累积：
            bias(t+dt) = bias(t) + N(0, rrw · √dt)。典型 MEMS 约 1e-4。
        imu_gyro_rrw_rads_per_sqrt_s: 陀螺仪 Rate Random Walk 系数
            （rad/s · √s⁻¹），对应 Allan 方差低频段。典型 MEMS 约 1e-5。
        vio_dx_std: VIO x 方向位移增量噪声标准差（米）。
        vio_dy_std: VIO y 方向位移增量噪声标准差（米）。
        vio_dyaw_std: VIO 航向角增量噪声标准差（弧度）。
        vio_scale_drift_rate: VIO 尺度漂移速率（1/s），
            模拟纯视觉里程计的累积尺度漂移，每秒的尺度因子变化率。
            典型值约 1e-4（即 100s 后尺度漂移 1%）。
        base_seed: 高斯噪声的确定性种子基，保证可复现。

    异常：
        TypeError: base_seed 不是 int 时抛出。
        ValueError: 任一 std/bias/rrw/drift 字段为 NaN/Inf 或为负数时抛出。
    """
    uwb_range_std_m: float = 0.05
    imu_ax_std: float = 0.03
    imu_ay_std: float = 0.03
    imu_gz_std: float = 0.01
    imu_accel_bias_instability: float = 0.015  # 典型 MEMS 加速度计 bias 不稳定性约 0.015 m/s²
    imu_gyro_bias_instability: float = 0.0005  # 典型 MEMS 陀螺仪 bias 不稳定性约 0.0005 rad
    imu_accel_rrw_mps2_per_sqrt_s: float = 1.0e-4  # 加速度计 RRW 系数 (m/s²·√s⁻¹)，典型 MEMS 约 1e-4
    imu_gyro_rrw_rads_per_sqrt_s: float = 1.0e-5  # 陀螺仪 RRW 系数 (rad/s·√s⁻¹)，典型 MEMS 约 1e-5
    vio_dx_std: float = 0.01
    vio_dy_std: float = 0.01
    vio_dyaw_std: float = 0.005
    vio_scale_drift_rate: float = 1e-4  # 尺度漂移速率，典型纯 VO 约 1%/100s
    base_seed: int = 20260605
    # §5.3 短时 IMU 主导段：UWB 全锚 NLOS 持续段 + VIO 持续中断段。
    # 与统计级 N3/V3 不同——本字段注入的是时序段，段内 UWB 行 valid=False
    # (全锚 NLOS) / VIO 行 quality=0 且 dx/dy/dyaw=0 (VIO 黑屏)。
    # 默认空 tuple 即无 outage；配置时为 ((start_s, end_s), ...)，
    # 时戳语义同序列内累计秒；段边界闭合 [start, end]；end > start；
    # 段间不要求排序（校验时自动排序去重）。
    uwb_outage_segments: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    vio_outage_segments: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # 13 项噪声/协方差参数必须有限且非负（std/bias/rrw/drift 物理上不可为负）
        _finite_nonnegative_fields = (
            "uwb_range_std_m",
            "imu_ax_std",
            "imu_ay_std",
            "imu_gz_std",
            "imu_accel_bias_instability",
            "imu_gyro_bias_instability",
            "imu_accel_rrw_mps2_per_sqrt_s",
            "imu_gyro_rrw_rads_per_sqrt_s",
            "vio_dx_std",
            "vio_dy_std",
            "vio_dyaw_std",
            "vio_scale_drift_rate",
        )
        for fname in _finite_nonnegative_fields:
            value = float(getattr(self, fname))
            if not math.isfinite(value):
                raise ValueError(f"{fname} must be finite, got {value!r}")
            if value < 0.0:
                raise ValueError(f"{fname} must be >= 0, got {value!r}")
        # base_seed 必须是 int（排除 bool），保证可哈希且可作为种子
        if not isinstance(self.base_seed, int) or isinstance(self.base_seed, bool):
            raise TypeError(
                f"base_seed must be int, got {type(self.base_seed).__name__}: {self.base_seed!r}"
            )
        # §5.3: 校验 uwb_outage_segments / vio_outage_segments（每段二元组、端点有限非负、end>start）
        # 段间允许重叠/乱序——校验通过后由 _normalize_outage_segments 排序去重 + 合并重叠。
        for _seg_field in ("uwb_outage_segments", "vio_outage_segments"):
            _segments = getattr(self, _seg_field)
            if not isinstance(_segments, tuple):
                raise TypeError(
                    f"{_seg_field} must be a tuple of (start, end) pairs, "
                    f"got {type(_segments).__name__}"
                )
            for _idx, _seg in enumerate(_segments):
                if (not isinstance(_seg, tuple)) or len(_seg) != 2:
                    raise TypeError(
                        f"{_seg_field}[{_idx}] must be a 2-tuple (start, end), got {_seg!r}"
                    )
                try:
                    _start_s = float(_seg[0])
                    _end_s = float(_seg[1])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{_seg_field}[{_idx}] endpoints must be numeric, got {_seg!r}"
                    ) from exc
                if not (math.isfinite(_start_s) and math.isfinite(_end_s)):
                    raise ValueError(
                        f"{_seg_field}[{_idx}] endpoints must be finite, got {_seg!r}"
                    )
                if _start_s < 0.0 or _end_s < 0.0:
                    raise ValueError(
                        f"{_seg_field}[{_idx}] endpoints must be non-negative, got {_seg!r}"
                    )
                if _end_s <= _start_s:
                    raise ValueError(
                        f"{_seg_field}[{_idx}] must have end > start, got {_seg!r}"
                    )
            # frozen dataclass 用 object.__setattr__ 写回排序去重后的段
            object.__setattr__(self, _seg_field, _normalize_outage_segments(_segments))

    @classmethod
    def from_yaml(cls, yaml_path: str | Path | None = None) -> "SimNoiseSpec":
        """从 configs/base/sim_noise_spec.yaml 的 sim_noise_spec 段加载协议层冻结默认值。

        参数：
            yaml_path: 可选的显式 yaml 路径；为 None 时使用项目默认路径
                configs/base/sim_noise_spec.yaml（与 sensors.yaml 分离，
                因为 sensors.yaml 顶级键由传感器字段合同独占）。

        返回：
            SimNoiseSpec 实例，字段值优先取自 yaml 的 sim_noise_spec 段；
            yaml 未指定的字段回退到 dataclass 硬编码默认。
        """
        defaults = _load_sim_noise_defaults(yaml_path)
        if not defaults:
            return cls()  # yaml 不可用时回退到硬编码默认
        return cls(**defaults)

    # D-7 patch: SimNoiseSpec 缺失字段显式 logger.warning 审计.
    @staticmethod
    def warn_missing_fixture_fields(
        missing_fields: list[str],
        *,
        sensor_kind: str,
        seq_id: str,
        fixture_path_hint: str = "",
    ) -> None:
        """记录 fixture 字段缺失审计日志 (不抛错, 保持向后兼容).

        D-7 patch 引入: 仿照 "fixture 字段缺失被静默跳过应留审计线索" 的需求,
        当 _materialize_imu_rows / _materialize_uwb_rows / _materialize_vio_rows
        发现对应 fixture 流的关键观测字段缺失 (被 row.get(..., default) 静默
        回退或被 _derive_base_*_residuals 等同绕开) 时, 由本 helper 统一记一条
        WARNING 级别日志, 给审计一条可见线索.

        参数：
            missing_fields: 缺失的 fixture 字段名列表 (如 ["ax", "ay"]).
            sensor_kind: 传感器种类, 取值 "imu" / "uwb" / "vio", 用于日志归类.
            seq_id: 当前物化序列编号, 用于定位哪条序列出问题.
            fixture_path_hint: 可选的 fixture 路径或 base_seq_id 提示, 便于回查.

        注意：
            - 本方法只记日志, 不抛错, 数据流保持原行为 (向后兼容).
            - 每次物化调用一次, 不在行级热路径触发, 避免日志噪声.
            - 字段分组映射 (由调用方在 _materialize_*_rows 中维护):
              IMU noise group (imu_ax_std 等) → fixture keys ax/ay/gz/timestamp;
              UWB noise group (uwb_range_std_m) → fixture keys
                  anchor_id/range/valid/quality/timestamp;
              VIO noise group (vio_dx_std 等) → fixture keys
                  dx/dy/dyaw/quality/timestamp
                  (铁律 3: 物化端不再输出 tracked_features/reproj_err;
                   但 fixture 残差回退仍按 fixture 自身字段读取).
        """
        if not missing_fields:
            return
        # 用 _logger 而非 warnings.warn, 因为这是审计线索而非用户应处理的退化.
        _logger.warning(
            "SimNoiseSpec fixture field audit (D-7): sensor_kind=%s seq_id=%s "
            "missing_fixture_fields=%s fixture_path_hint=%r "
            "(fields will be silently skipped or use default fallback values)",
            sensor_kind,
            seq_id,
            missing_fields,
            fixture_path_hint,
        )


# 默认噪声规格，用于大多数仿真场景。
# 优先从 configs/base/sim_noise_spec.yaml 的 sim_noise_spec 段加载协议层冻结默认值；
# yaml 不可用时回退到 SimNoiseSpec 硬编码默认。
DEFAULT_SIM_NOISE_SPEC = SimNoiseSpec.from_yaml()
# 零噪声规格，用于无噪声对照实验或几何验证
ZERO_SIM_NOISE_SPEC = SimNoiseSpec(
    uwb_range_std_m=0.0,
    imu_ax_std=0.0,
    imu_ay_std=0.0,
    imu_gz_std=0.0,
    imu_accel_bias_instability=0.0,
    imu_gyro_bias_instability=0.0,
    imu_accel_rrw_mps2_per_sqrt_s=0.0,
    imu_gyro_rrw_rads_per_sqrt_s=0.0,
    vio_dx_std=0.0,
    vio_dy_std=0.0,
    vio_dyaw_std=0.0,
    vio_scale_drift_rate=0.0,
)


def _coerce_valid_flag(raw_valid: Any) -> bool:
    """Normalize UWB valid flag, aligned with protocol _resolve_uwb_valid_flag.

    Protocol (protocol/liquid_bridge_contract.py _resolve_uwb_valid_flag):
    only bool-like False marks UWB as invalid; all other values (0, None,
    'false') are treated as valid. This avoids the semantic conflict where
    the old implementation converted 0/None/'false' to False, contradicting
    the protocol's conservative "only explicit False is invalid" rule.
    """
    if is_bool_like(raw_valid):
        return bool(raw_valid)
    return True


def _build_default_sim_sequence_specs() -> tuple[SimSequenceSpec, ...]:
    """构建默认主表仿真序列规格（协议轨迹 + 欠定几何）。

    10 lead × 2 var = 30 条。默认：
    - use_protocol_trajectory=True（协议轨迹路径，GT 直接满足 B20–B23 包络；
      sample count 由 duration/dt 决定，不再依赖 fixture×cycle）
    - G/K 主表以 G1/K4 为主，穿插 G2 差几何与 K3 临界；K6/K8 仅少量压力条且 allow_high_anchor_count
    - duration ≥ 45s，workspace 约 20m
    """
    # 主表 G/K 池：欠定为主，禁止默认全 K6 优几何。
    gk_pool: list[tuple[tuple[str, str], ...]] = [
        (("G", "G1"), ("K", "K4")),
        (("G", "G2"), ("K", "K3")),
        (("G", "G1"), ("K", "K3")),
        (("G", "G2"), ("K", "K4")),
        (("G", "G1"), ("K", "K4")),
        (("G", "G2"), ("K", "K3")),
        (("G", "G1"), ("K", "K4")),
        (("G", "G2"), ("K", "K4")),
        (("G", "G1"), ("K", "K3")),
        (("G", "G2"), ("K", "K3")),
    ]
    lead_meta = [
        ("sim_line_01", "mini_seq", 45.0, 20.0),
        ("sim_line_02", "mini_seq_02", 48.0, 18.0),
        ("sim_curve_01", "mini_seq", 50.0, 22.0),
        ("sim_curve_02", "mini_seq", 46.0, 20.0),
        ("sim_mirror_01", "mini_seq_02", 47.0, 19.0),
        ("sim_rotate_01", "mini_seq_03", 52.0, 21.0),
        ("sim_shift_01", "mini_seq_03", 45.0, 20.0),
        ("sim_shift_02", "mini_seq_03", 49.0, 23.0),
        ("sim_turn_01", "mini_seq_03", 55.0, 20.0),
        ("sim_turn_02", "mini_seq_03", 60.0, 25.0),
    ]
    # 协议轨迹路径不依赖 cycle_count（GT 由 duration/dt 直接生成），
    # 仍保留 cycle_count 异质化以满足 SimSequenceSpec 字段约束（>=1）。
    lead_specs: list[SimSequenceSpec] = []
    for idx, (seq_id, base_id, dur, span) in enumerate(lead_meta):
        g_level = dict(gk_pool[idx])["G"]
        k_level = dict(gk_pool[idx])["K"]
        high_k = k_level in ("K6", "K8")
        lead_specs.append(
            SimSequenceSpec(
                seq_id,
                base_id,
                translation_xy=(0.0, 0.0),
                cycle_count=128 + idx * 2,  # 协议轨迹路径不消费，保留字段约束
                cycle_gap_s=0.05,
                axes_override=gk_pool[idx],
                use_protocol_trajectory=True,  # 协议轨迹路径，满足 B20–B23 包络
                duration_s=float(dur),
                workspace_span_m=float(span),
                envelope_profile="main_table",
                allow_high_anchor_count=high_k,
                dt_imu_override=0.01,
                dt_uwb_override=0.1,
                dt_vio_override=0.05,
            )
        )
    sequence_specs = list(lead_specs)
    for lead_spec in lead_specs:
        for variant_index, dur_delta in enumerate((3.0, -2.0), start=1):
            sequence_specs.append(
                SimSequenceSpec(
                    seq_id=f"{lead_spec.seq_id}_var_{variant_index:02d}",
                    base_seq_id=lead_spec.base_seq_id,
                    translation_xy=lead_spec.translation_xy,
                    rotation_rad=lead_spec.rotation_rad,
                    mirror_x=lead_spec.mirror_x,
                    curve_strength=lead_spec.curve_strength,
                    cycle_count=lead_spec.cycle_count + variant_index,
                    cycle_gap_s=lead_spec.cycle_gap_s,
                    axes_override=lead_spec.axes_override,
                    use_protocol_trajectory=True,
                    duration_s=max(30.0, float(lead_spec.duration_s) + dur_delta),
                    workspace_span_m=float(lead_spec.workspace_span_m),
                    envelope_profile="main_table",
                    allow_high_anchor_count=bool(lead_spec.allow_high_anchor_count),
                    dt_imu_override=lead_spec.dt_imu_override,
                    dt_uwb_override=lead_spec.dt_uwb_override,
                    dt_vio_override=lead_spec.dt_vio_override,
                )
            )
    return tuple(sequence_specs)


DEFAULT_SIM_SEQUENCE_SPECS = _build_default_sim_sequence_specs()


def _build_sim_e9_only_compact_sequence_specs() -> tuple[SimSequenceSpec, ...]:
    """e9 双退化主表序列：协议轨迹 + 欠定几何 + A/N 压力。

    相对旧 compact：
    - 不再 short 2–4s / L_xy<1m 的 fixture 铺叠
    - 默认 G1/K4（欠定），穿插 G2/K3；K6/K8 仅压力条且 allow_high_anchor_count
    - duration ≥ 30s，workspace 15–25m，满足 B20–B23
    - 保留 A/N/V 多样性与 dt 覆盖，供异步/NLOS 命题
    """
    axes_pool: list[tuple[tuple[str, str], ...]] = [
        (("A", "A2"), ("N", "N2"), ("V", "V0"), ("G", "G1"), ("K", "K4")),
        (("A", "A2"), ("N", "N3"), ("V", "V0"), ("G", "G2"), ("K", "K3")),
        (("A", "A3"), ("N", "N2"), ("V", "V0"), ("G", "G1"), ("K", "K3")),
        (("A", "A3"), ("N", "N3"), ("V", "V0"), ("G", "G2"), ("K", "K4")),
        (("A", "A2"), ("N", "N2"), ("V", "V0"), ("G", "G1"), ("K", "K4")),
        (("A", "A2"), ("N", "N3"), ("V", "V0"), ("G", "G2"), ("K", "K3")),
        (("A", "A3"), ("N", "N2"), ("V", "V0"), ("G", "G1"), ("K", "K4")),
        (("A", "A3"), ("N", "N3"), ("V", "V0"), ("G", "G2"), ("K", "K3")),
        (("A", "A2"), ("N", "N2"), ("V", "V0"), ("G", "G1"), ("K", "K4")),
        (("A", "A3"), ("N", "N3"), ("V", "V0"), ("G", "G2"), ("K", "K4")),
        # long / 压力：允许少量 K6/K8
        (("A", "A2"), ("N", "N2"), ("V", "V0"), ("G", "G1"), ("K", "K4")),
        (("A", "A3"), ("N", "N3"), ("V", "V0"), ("G", "G2"), ("K", "K3")),
        (("A", "A2"), ("N", "N3"), ("V", "V0"), ("G", "G1"), ("K", "K4")),
        (("A", "A3"), ("N", "N2"), ("V", "V0"), ("G", "G2"), ("K", "K4")),
        (("A", "A2"), ("N", "N2"), ("V", "V0"), ("G", "G1"), ("K", "K4")),
        (("A", "A0"), ("N", "N2"), ("V", "V0"), ("G", "G1"), ("K", "K4")),
        (("A", "A1"), ("N", "N3"), ("V", "V0"), ("G", "G2"), ("K", "K3")),
        (("A", "A2"), ("N", "N2"), ("V", "V0"), ("G", "G1"), ("K", "K4")),
        (("A", "A3"), ("N", "N3"), ("V", "V0"), ("G", "G2"), ("K", "K3")),
        (("A", "A2"), ("N", "N3"), ("V", "V0"), ("G", "G1"), ("K", "K4")),
    ]
    dt_pool: list[tuple[float | None, float | None, float | None]] = [
        (0.01, 0.1, 0.05),
        (0.01, 0.05, 0.05),
        (0.02, 0.1, 0.05),
        (0.01, 0.1, 0.0333),
        (0.005, 0.05, 0.05),
    ]
    lead_names = [
        ("sim_line_01", "mini_seq", 40.0, 18.0),
        ("sim_line_02", "mini_seq_02", 42.0, 20.0),
        ("sim_curve_01", "mini_seq", 45.0, 20.0),
        ("sim_curve_02", "mini_seq", 45.0, 22.0),
        ("sim_mirror_01", "mini_seq_02", 48.0, 19.0),
        ("sim_rotate_01", "mini_seq_03", 50.0, 20.0),
        ("sim_shift_01", "mini_seq_03", 45.0, 21.0),
        ("sim_shift_02", "mini_seq_03", 47.0, 20.0),
        ("sim_turn_01", "mini_seq_03", 55.0, 23.0),
        ("sim_turn_02", "mini_seq_03", 55.0, 20.0),
        ("sim_long_10m_01", "mini_seq", 35.0, 15.0),
        ("sim_long_20m_01", "mini_seq_02", 40.0, 20.0),
        ("sim_long_30m_01", "mini_seq_03", 50.0, 25.0),
        ("sim_long_40m_01", "mini_seq", 55.0, 28.0),
        ("sim_long_50m_01", "mini_seq_02", 60.0, 30.0),
        ("sim_long_10m_02", "mini_seq_03", 35.0, 16.0),
        ("sim_long_20m_02", "mini_seq_03", 42.0, 20.0),
        ("sim_long_30m_02", "mini_seq_02", 50.0, 24.0),
        ("sim_long_40m_02", "mini_seq_02", 55.0, 27.0),
        ("sim_long_50m_02", "mini_seq", 60.0, 30.0),
    ]
    lead_specs: list[SimSequenceSpec] = []
    for idx, (seq_id, base_id, dur, span) in enumerate(lead_names):
        axes = axes_pool[idx % len(axes_pool)]
        k_level = dict(axes).get("K", "K4")
        high_k = k_level in ("K6", "K8")
        dt = dt_pool[idx % len(dt_pool)]
        lead_specs.append(
            SimSequenceSpec(
                seq_id,
                base_id,
                translation_xy=(0.0, 0.0),
                cycle_count=max(2, int(dur)),
                cycle_gap_s=0.05,
                axes_override=axes,
                dt_imu_override=dt[0],
                dt_uwb_override=dt[1],
                dt_vio_override=dt[2],
                use_protocol_trajectory=True,
                duration_s=float(dur),
                workspace_span_m=float(span),
                envelope_profile="main_table",
                allow_high_anchor_count=high_k,
            )
        )
    sequence_specs = list(lead_specs)
    for lead_spec in lead_specs:
        for variant_index, dur_delta in enumerate((2.0, -2.0), start=1):
            sequence_specs.append(
                SimSequenceSpec(
                    seq_id=f"{lead_spec.seq_id}_var_{variant_index:02d}",
                    base_seq_id=lead_spec.base_seq_id,
                    translation_xy=lead_spec.translation_xy,
                    rotation_rad=lead_spec.rotation_rad,
                    mirror_x=lead_spec.mirror_x,
                    curve_strength=lead_spec.curve_strength,
                    cycle_count=lead_spec.cycle_count + variant_index,
                    cycle_gap_s=lead_spec.cycle_gap_s,
                    axes_override=lead_spec.axes_override,
                    dt_imu_override=lead_spec.dt_imu_override,
                    dt_uwb_override=lead_spec.dt_uwb_override,
                    dt_vio_override=lead_spec.dt_vio_override,
                    use_protocol_trajectory=True,
                    duration_s=max(30.0, float(lead_spec.duration_s) + dur_delta),
                    workspace_span_m=float(lead_spec.workspace_span_m),
                    envelope_profile="main_table",
                    allow_high_anchor_count=bool(lead_spec.allow_high_anchor_count),
                )
            )
    return tuple(sequence_specs)


SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS = _build_sim_e9_only_compact_sequence_specs()


def _read_json_records(path: Path, *, required_mapping: bool = False) -> Any:
    """读取 JSON 文件并校验其顶层结构。

    参数：
        path: JSON 文件路径，必须存在。
        required_mapping: 为 True 时要求顶层是字典，否则要求顶层是字典列表。

    返回：
        解析后的 Python 对象（字典或字典列表）。

    异常：
        FileNotFoundError: 文件不存在时抛出。
        TypeError: 文件内容结构不符合要求时抛出。
    """
    if not path.is_file():
        raise FileNotFoundError(f"required fixture file not found: {path}")
    payload = read_json(path)
    if required_mapping:
        # 要求顶层是字典（如 anchor_layout.json）
        if not isinstance(payload, dict):
            raise TypeError(f"expected dict payload in {path}")
        return payload
    # 要求顶层是字典列表（如 imu.json, uwb.json 等流文件）
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise TypeError(f"expected list[dict] payload in {path}")
    return payload


def _fixture_root(default_root: str | Path | None = None) -> Path:
    """解析 fixture 根目录路径。

    参数：
        default_root: 可选的显式根目录路径；为 None 时使用项目默认路径。

    返回：
        fixture 根目录的绝对路径。
    """
    if default_root is None:
        # 默认路径：项目根/tests/fixtures/datasets/miluv
        return find_project_root() / "tests" / "fixtures" / "datasets" / "miluv"
    return Path(default_root).resolve()


def _load_fixture_bundle(base_seq_id: str, fixture_root: str | Path | None = None) -> dict[str, Any]:
    """加载一条 fixture 种子序列的全部流文件。

    参数：
        base_seq_id: fixture 种子序列编号，必须在 _FIXTURE_BASE_SEQ_IDS 白名单中。
        fixture_root: 可选的 fixture 根目录路径。

    返回：
        包含 base_seq_id、seq_root 和各流数据的字典。

    异常：
        ValueError: base_seq_id 不在白名单中时抛出。
    """
    if base_seq_id not in _FIXTURE_BASE_SEQ_IDS:
        raise ValueError(f"unsupported base fixture seq_id: {base_seq_id}")
    seq_root = _fixture_root(fixture_root) / base_seq_id
    return {
        "base_seq_id": base_seq_id,
        "seq_root": str(seq_root),
        "imu": _read_json_records(seq_root / "imu.json"),
        "uwb": _read_json_records(seq_root / "uwb.json"),
        "vio": _read_json_records(seq_root / "vio.json"),
        "gt": _read_json_records(seq_root / "gt.json"),
        # anchor_layout 顶层是字典，不是列表
        "anchor_layout": _read_json_records(seq_root / "anchor_layout.json", required_mapping=True),
    }


# 模块内部别名，保持 _前缀 名称的向后兼容
_normalize_gt_rows = normalize_gt_rows


def _build_gt_timestamp_index(gt_rows: list[dict[str, float]]) -> list[float]:
    """从已排序的 GT 行列表中提取时间戳索引，供 bisect 加速插值查找。

    参数：
        gt_rows: 已按 timestamp 升序排序的 GT 行列表。

    返回：
        与 gt_rows 等长的时间戳列表（float）。
    """
    return [float(row["timestamp"]) for row in gt_rows]


def _interpolate_pose(
    gt_rows: list[dict[str, float]],
    timestamp: float,
    *,
    timestamp_index: list[float] | None = None,
) -> dict[str, float]:
    """在 GT 行列表中对指定时间戳做线性插值，返回位姿。

    对于 yaw 角使用 angle_delta_rad 做角度空间插值，避免 0/2π 边界跳变。

    参数：
        gt_rows: 已按时间戳排序的 GT 行列表，不能为空。
        timestamp: 需要插值的目标时间戳。
        timestamp_index: 可选的预构建时间戳索引（与 gt_rows 等长）。
            传入时跳过内部 bisect 用的列表构建，用于热路径加速。
            若为 None 则内部从 gt_rows 现场提取时间戳列表。

    返回：
        插值后的位姿字典，包含 timestamp、px、py、yaw。

    异常：
        ValueError: gt_rows 为空或 timestamp 含 NaN/Inf 时抛出。
    """
    if not gt_rows:
        raise ValueError("gt_rows must be non-empty")
    target = float(timestamp)
    if not math.isfinite(target):
        raise ValueError(f"interpolation target timestamp must be finite, got {timestamp!r}")
    # 低于最早时间戳时，钳位到首帧
    first_ts = float(gt_rows[0]["timestamp"])
    if target <= first_ts:
        return dict(gt_rows[0])
    # 高于最晚时间戳时，钳位到末帧
    last_ts = float(gt_rows[-1]["timestamp"])
    if target >= last_ts:
        return dict(gt_rows[-1])
    # 用 bisect 在 O(log N) 内定位包含 target 的区间 [lower, upper)。
    # timestamp_index 可由调用方预构建并复用，避免每次插值都重新扫描。
    if timestamp_index is None:
        timestamp_index = [float(row["timestamp"]) for row in gt_rows]
    # bisect_left 返回第一个 >= target 的索引；插值区间是 [idx-1, idx]。
    idx = bisect_left(timestamp_index, target)
    if idx <= 0:
        return dict(gt_rows[0])
    if idx >= len(gt_rows):
        return dict(gt_rows[-1])
    lower = gt_rows[idx - 1]
    upper = gt_rows[idx]
    lt = float(lower["timestamp"])
    ut = float(upper["timestamp"])
    # 精确命中下界
    if target == lt:
        return dict(lower)
    # 精确命中上界
    if target == ut:
        return dict(upper)
    # lt < target < ut（bisect_left 保证 target > timestamp_index[idx-1]，
    # 且 target < timestamp_index[idx] 因为前面已检查 target < last_ts）
    alpha = (target - lt) / (ut - lt)  # 线性插值系数
    lower_px = float(lower["px"])
    lower_py = float(lower["py"])
    lower_yaw = float(lower["yaw"])
    upper_px = float(upper["px"])
    upper_py = float(upper["py"])
    upper_yaw = float(upper["yaw"])
    return {
        "timestamp": target,
        "px": (1.0 - alpha) * lower_px + alpha * upper_px,
        "py": (1.0 - alpha) * lower_py + alpha * upper_py,
        # yaw 在角度空间插值，用 angle_delta_rad 处理环绕
        "yaw": wrap_angle_rad(lower_yaw + alpha * angle_delta_rad(upper_yaw, lower_yaw)),
    }


def _body_frame_delta(prev_pose: dict[str, float], curr_pose: dict[str, float]) -> tuple[float, float, float]:
    """计算两个位姿之间在上一帧体坐标系下的位移增量和航向增量。

    参数：
        prev_pose: 前一帧位姿，包含 px、py、yaw。
        curr_pose: 当前帧位姿，包含 px、py、yaw。

    返回：
        三元组 (dx_local, dy_local, dyaw)，分别为体坐标系 x/y 位移和航向角增量。
    """
    dx_world = coerce_finite_scalar(curr_pose["px"], name="curr_pose.px") - coerce_finite_scalar(prev_pose["px"], name="prev_pose.px")
    dy_world = coerce_finite_scalar(curr_pose["py"], name="curr_pose.py") - coerce_finite_scalar(prev_pose["py"], name="prev_pose.py")
    prev_yaw = coerce_finite_scalar(prev_pose["yaw"], name="prev_pose.yaw")
    cos_value = math.cos(prev_yaw)
    sin_value = math.sin(prev_yaw)
    # 世界坐标系位移旋转到体坐标系
    dx_local = (cos_value * dx_world) + (sin_value * dy_world)
    dy_local = (-sin_value * dx_world) + (cos_value * dy_world)
    dyaw = angle_delta_rad(coerce_finite_scalar(curr_pose["yaw"], name="curr_pose.yaw"), prev_yaw)
    return dx_local, dy_local, dyaw


def _mirror_xy(x: float, y: float) -> tuple[float, float]:
    """沿 X 轴做镜像：x 不变，y 取反。

    参数：
        x: X 坐标。
        y: Y 坐标。

    返回：
        镜像后的 (x, -y)。
    """
    return float(x), float(-y) if y != 0.0 else 0.0


def _mirror_yaw(yaw: float) -> float:
    """镜像后的航向角：取反并归一化到 [-π, π)。

    参数：
        yaw: 原始航向角（弧度）。

    返回：
        镜像后的归一化航向角。
    """
    return wrap_angle_rad(-float(yaw))


def _rotate_xy(x: float, y: float, rotation_rad: float) -> tuple[float, float]:
    """将二维坐标绕原点旋转指定角度。

    参数：
        x: X 坐标。
        y: Y 坐标。
        rotation_rad: 旋转角度（弧度），正值为逆时针。

    返回：
        旋转后的 (x', y')。
    """
    cos_value = math.cos(rotation_rad)
    sin_value = math.sin(rotation_rad)
    return (cos_value * float(x) - sin_value * float(y), sin_value * float(x) + cos_value * float(y))


def _warp_curve_position(x: float, y: float, curve_strength: float) -> tuple[float, float]:
    """对位置施加抛物线弯曲变形：y += curve_strength * x²。

    当 curve_strength 接近 0 时退化为恒等变换。

    参数：
        x: X 坐标。
        y: Y 坐标。
        curve_strength: 弯曲强度，正值向上弯，负值向下弯。

    返回：
        弯曲后的 (x, y')。

    异常：
        ValueError: 输入含 NaN/Inf 时抛出。
    """
    x = coerce_finite_scalar(x, name="x")
    y = coerce_finite_scalar(y, name="y")
    curve_strength = coerce_finite_scalar(curve_strength, name="curve_strength")
    if math.isclose(curve_strength, 0.0, abs_tol=_GEOMETRIC_ZERO_EPSILON):
        return float(x), float(y)
    return float(x), float(y) + float(curve_strength) * float(x) * float(x)


def _transform_position(x: float, y: float, spec: SimSequenceSpec) -> tuple[float, float]:
    """按规格对位置施加完整的几何变换链：镜像 → 旋转 → 弯曲 → 平移。

    参数：
        x: 原始 X 坐标。
        y: 原始 Y 坐标。
        spec: 序列规格，包含变换参数。

    返回：
        变换后的 (x', y')。
    """
    tx, ty = float(x), float(y)
    if spec.mirror_x:
        tx, ty = _mirror_xy(tx, ty)
    tx, ty = _rotate_xy(tx, ty, float(spec.rotation_rad))
    tx, ty = _warp_curve_position(tx, ty, float(spec.curve_strength))
    # 最后叠加平移偏移
    return tx + float(spec.translation_xy[0]), ty + float(spec.translation_xy[1])


def _transform_yaw(yaw: float, spec: SimSequenceSpec) -> float:
    """按规格对航向角施加变换：镜像 → 旋转叠加。

    参数：
        yaw: 原始航向角（弧度）。
        spec: 序列规格，包含变换参数。

    返回：
        变换后归一化的航向角。
    """
    value = float(yaw)
    if spec.mirror_x:
        value = _mirror_yaw(value)
    return wrap_angle_rad(value + float(spec.rotation_rad))


def _round_gt_rows(gt_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """对 GT 行的所有数值字段做 6 位小数四舍五入，保证 JSON 输出稳定。

    对 yaw 字段先 wrap 再 round 再 wrap，确保 round 后仍落在 [-π, π)，
    避免 round(π, 6) = 3.141593 > π 的边界溢出。所有字段经 _round_normalize
    归一化 -0.0，避免 JSON 序列化产生 "-0.0"。

    参数：
        gt_rows: GT 行列表。

    返回：
        四舍五入后的 GT 行列表。
    """
    return [
        {
            "timestamp": _round_normalize(coerce_finite_scalar(row["timestamp"], name="row.timestamp")),
            "px": _round_normalize(coerce_finite_scalar(row["px"], name="row.px")),
            "py": _round_normalize(coerce_finite_scalar(row["py"], name="row.py")),
            # yaw: 先 wrap 再 round 再 wrap，确保 round 后仍落在 [-π, π)
            "yaw": wrap_angle_rad(_round_normalize(wrap_angle_rad(coerce_finite_scalar(row["yaw"], name="row.yaw")))),
        }
        for row in gt_rows
    ]


def _recompute_yaw_from_positions(gt_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """根据相邻位置重新计算每帧的航向角，覆盖原始 yaw 值。

    几何变换后原始 yaw 可能不再与轨迹方向一致，因此需要从位置差
    重新推导航向角。首帧用自身与下一帧的方向，末帧用上一帧与自身
    的方向，中间帧用前后两帧的中心差分方向。

    参数：
        gt_rows: GT 行列表，至少包含 px 和 py 字段。

    返回：
        更新了 yaw 字段的 GT 行列表。
    """
    if len(gt_rows) <= 1:
        return [dict(row) for row in gt_rows]
    rows = [dict(row) for row in gt_rows]
    has_cycle_boundary = all("_cycle_index" in row for row in rows)
    if has_cycle_boundary:
        segments: list[list[dict[str, float]]] = []
        current_segment: list[dict[str, float]] = []
        current_cycle_index: int | None = None
        for row in rows:
            cycle_index = int(row["_cycle_index"])
            if current_segment and cycle_index != current_cycle_index:
                segments.append(current_segment)
                current_segment = []
            current_segment.append(row)
            current_cycle_index = cycle_index
        if current_segment:
            segments.append(current_segment)
    else:
        segments = [rows]

    for segment in segments:
        if len(segment) <= 1:
            continue
        for index, row in enumerate(segment):
            if index == 0:
                # 首帧：用本段首帧和次帧的方向，避免跨 cycle 污染
                ref_a, ref_b = segment[0], segment[1]
            elif index == len(segment) - 1:
                # 末帧：用本段倒数第二帧和末帧的方向
                ref_a, ref_b = segment[-2], segment[-1]
            else:
                # 中间帧：用本段前后两帧的中心差分方向
                ref_a, ref_b = segment[index - 1], segment[index + 1]
            dx = coerce_finite_scalar(ref_b["px"], name="ref_b.px") - coerce_finite_scalar(ref_a["px"], name="ref_a.px")
            dy = coerce_finite_scalar(ref_b["py"], name="ref_b.py") - coerce_finite_scalar(ref_a["py"], name="ref_a.py")
            # 只有位移足够大时才更新 yaw，避免静止时 atan2 不稳定
            if math.hypot(dx, dy) > 1e-12:
                row["yaw"] = wrap_angle_rad(math.atan2(dy, dx))
    return rows


def _compute_segment_delta(base_gt_rows: list[dict[str, float]]) -> tuple[float, float]:
    """计算 GT 序列首尾之间的位置差。

    参数：
        base_gt_rows: GT 行列表，至少 1 行。

    返回：
        二元组 (dx, dy)，首帧到末帧的位置差。

    异常：
        ValueError: base_gt_rows 为空或含 NaN/Inf 时抛出。
    """
    _require_non_empty(base_gt_rows, name="base_gt_rows")
    first_px = coerce_finite_scalar(base_gt_rows[0]["px"], name="first_px")
    first_py = coerce_finite_scalar(base_gt_rows[0]["py"], name="first_py")
    last_px = coerce_finite_scalar(base_gt_rows[-1]["px"], name="last_px")
    last_py = coerce_finite_scalar(base_gt_rows[-1]["py"], name="last_py")
    return (last_px - first_px, last_py - first_py)


def _safe_diff_divide(delta: float, dt: float) -> float:
    """安全差分除法：dt 过小时返回 0.0，防止速度/加速度爆炸。

    当 dt <= 0 时抛出 ValueError（时间戳必须单调递增）；
    当 0 < dt < _SAFE_DIFF_EPSILON 时返回 0.0（退化，差分结果不可靠）。

    异常：
        ValueError: delta/dt 为 NaN/Inf 或 dt <= 0 时抛出。
    """
    delta = coerce_finite_scalar(delta, name="delta")
    dt = coerce_finite_scalar(dt, name="dt")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt!r}")
    if dt < _SAFE_DIFF_EPSILON:
        return 0.0
    return delta / dt


def _compute_terminal_velocity(base_gt_rows: list[dict[str, float]]) -> tuple[float, float]:
    """计算 GT 序列末端的瞬时速度（用最后两帧的差分估计）。

    参数：
        base_gt_rows: GT 行列表，至少 2 行才能算差分。

    返回：
        二元组 (vx, vy)，末端速度。

    异常：
        ValueError: 末两帧时间戳/位置含 NaN/Inf 时抛出。
    """
    if len(base_gt_rows) < 2:
        return 0.0, 0.0
    tail = base_gt_rows[-1]
    prev = base_gt_rows[-2]
    dt = coerce_finite_scalar(tail["timestamp"], name="tail_timestamp") - coerce_finite_scalar(prev["timestamp"], name="prev_timestamp")
    return (
        _safe_diff_divide(coerce_finite_scalar(tail["px"], name="tail_px") - coerce_finite_scalar(prev["px"], name="prev_px"), dt),
        _safe_diff_divide(coerce_finite_scalar(tail["py"], name="tail_py") - coerce_finite_scalar(prev["py"], name="prev_py"), dt),
    )


def _compute_cycle_pose_increment(base_gt_rows: list[dict[str, float]], *, cycle_gap_s: float) -> tuple[float, float]:
    """计算一个 cycle 的位姿增量，用于铺叠时保持轨迹连续。

    增量 = 首尾位置差 + 末端速度 × 周期间隔，保证 cycle 边界处
    GT 位置按终端速度外推，使铺叠后的轨迹在 cycle 间平滑衔接。

    参数：
        base_gt_rows: 单个 cycle 的 GT 行列表。
        cycle_gap_s: 周期间隔秒数。

    返回：
        二元组 (increment_x, increment_y)，一个 cycle 的位姿增量。

    异常：
        ValueError: cycle_gap_s 为 NaN/Inf 时抛出。
    """
    cycle_gap_s = coerce_finite_scalar(cycle_gap_s, name="cycle_gap_s")
    seg_dx, seg_dy = _compute_segment_delta(base_gt_rows)
    vel_x, vel_y = _compute_terminal_velocity(base_gt_rows)
    return seg_dx + vel_x * cycle_gap_s, seg_dy + vel_y * cycle_gap_s


def _tile_base_gt_rows(
    base_gt_rows: list[dict[str, float]],
    *,
    cycle_count: int,
    cycle_gap_s: float,
    base_seed: int = 0,
    seq_id: str = "inline_seq",
    jitter_sigma_s: float = 0.0,
    randomize_cycle_count: bool = False,
) -> list[dict[str, float]]:
    """将单 cycle 的 GT 行铺叠成多 cycle 的长序列。

    每个 cycle 的 GT 位置按 cycle_pose_increment 偏移，时间戳按
    cycle_stride 偏移，保证轨迹在 cycle 边界处连续。铺叠后保留
    _cycle_index 和 _base_row_index 辅助字段，供后续物化使用。

    铁律 5: GT timestamp 异步 jitter. jitter_sigma_s > 0 时每行加
      N(0, sigma) 时戳噪声 (确定性 _seeded_gaussian(seed, "gt", seq_id,
      cycle_idx, base_row_idx, "timestamp_jitter")). GT jitter 默认 0
      (把 GT 作为多传感器同步基准, jitter 留给 IMU/UWB/VIO 各自), 接口保留.

    铁律 6: 轨迹无规律随机. 把固定 cycle_pose_increment 改为 per-cycle 加扰:
      σ_cycle_delta = 0.05 * cycle_distance (per-cycle 增量幅度)
      cycle_dx_new = rotated(cycle_dx, ±10°) + N(0, σ_cycle_delta)
      并加 ±10° 主方向偏离包络: 对每个 cycle 用旋转矩阵 R(±10°) 作用增量.
      randomize_cycle_count=True 时启用 per-cycle 随机化 (铁律 6 主开关). 默认
      False 保持强周期 (debug 兼容), materialize_sim_raw 在调用点用全局开关启用.

    参数：
        base_gt_rows: 单 cycle 的 GT 行列表（已标准化和排序）。
        cycle_count: 铺叠周期数。
        cycle_gap_s: 周期间隔秒数。
        base_seed: 确定性 noise seed (用于 jitter 与 cycle 随机化).
        seq_id: 序列 ID (jitter 区分用).
        jitter_sigma_s: GT timestamp jitter sigma (秒).
        randomize_cycle_count: 是否启用 per-cycle 增量随机化 (铁律 6 主开关).

    返回：
        铺叠后的 GT 行列表，包含辅助字段 _cycle_index 和 _base_row_index。

    异常：
        ValueError: base_gt_rows 为空时抛出。
    """
    base_rows = _normalize_gt_rows(base_gt_rows)
    if not base_rows:
        raise ValueError("base_gt_rows must be non-empty after normalization")
    base_start = coerce_finite_scalar(base_rows[0]["timestamp"], name="base_rows[0].timestamp")
    base_end = coerce_finite_scalar(base_rows[-1]["timestamp"], name="base_rows[-1].timestamp")
    # 一个 cycle 的时间跨度 = 序列时长 + 间隔
    cycle_stride = (base_end - base_start) + float(cycle_gap_s)
    cycle_dx_nominal, cycle_dy_nominal = _compute_cycle_pose_increment(base_rows, cycle_gap_s=cycle_gap_s)
    # 铁律 6: per-cycle 轨迹随机参数预计算
    # σ_cycle_delta = 0.05 * cycle 单 cycle 位移幅度
    cycle_distance = math.hypot(cycle_dx_nominal, cycle_dy_nominal)
    sigma_cycle_delta = 0.05 * cycle_distance
    # 锚点：首帧位置，铺叠时减去以消除 fixture 原点偏移
    anchor_px = coerce_finite_scalar(base_rows[0]["px"], name="base_rows[0].px")
    anchor_py = coerce_finite_scalar(base_rows[0]["py"], name="base_rows[0].py")
    # 铁律 5: GT timestamp jitter (默认 sigma=0 不生成). 用 _seeded_gaussian 与 IMU 等路径一致.
    use_gt_jitter = jitter_sigma_s > 0.0
    # 铁律 6: 主开关. 仅当当传 randomize_cycle_count=True 时启用 per-cycle
    # 增量随机化, 默认 False 保持向后兼容 (debug 与既有单测期望强周期).
    use_cycle_random = bool(randomize_cycle_count)
    tiled_rows: list[dict[str, float]] = []
    effective_cycle_count = int(cycle_count)
    # 铁律 6: 累积偏移递推 (per-cycle 个性化增量), 而非 nominal cycle_idx 倍乘.
    cum_offset_x = 0.0
    cum_offset_y = 0.0
    for cycle_index in range(effective_cycle_count):
        offset_t = cycle_index * cycle_stride  # 时间偏移仍按固定 stride
        # 铁律 6: 每 cycle 增量加 per-cycle 随机偏移
        if use_cycle_random:
            delta_x_noise = _seeded_gaussian(
                int(base_seed),
                sigma_cycle_delta,
                "gt", seq_id, cycle_index, "delta_x",
            )
            delta_y_noise = _seeded_gaussian(
                int(base_seed),
                sigma_cycle_delta,
                "gt", seq_id, cycle_index, "delta_y",
            )
            heading_dev_rad = _seeded_uniform(
                int(base_seed),
                math.radians(10.0),
                "gt", seq_id, cycle_index, "heading_dev",
            )
            cos_h = math.cos(heading_dev_rad)
            sin_h = math.sin(heading_dev_rad)
            rotated_dx = cycle_dx_nominal * cos_h - cycle_dy_nominal * sin_h
            rotated_dy = cycle_dx_nominal * sin_h + cycle_dy_nominal * cos_h
            this_cycle_dx = rotated_dx + delta_x_noise
            this_cycle_dy = rotated_dy + delta_y_noise
        else:
            this_cycle_dx = cycle_dx_nominal
            this_cycle_dy = cycle_dy_nominal
        offset_x = cum_offset_x
        offset_y = cum_offset_y
        cum_offset_x += this_cycle_dx
        cum_offset_y += this_cycle_dy
        for base_row_index, row in enumerate(base_rows):
            ts = (coerce_finite_scalar(row["timestamp"], name="row.timestamp") - base_start) + offset_t
            if use_gt_jitter:
                ts += _seeded_gaussian(int(base_seed), jitter_sigma_s, "gt", seq_id, cycle_index, base_row_index, "timestamp_jitter")
            tiled_rows.append(
                {
                    "timestamp": _round_normalize(ts),
                    "px": coerce_finite_scalar(row["px"], name="row.px") - anchor_px + offset_x,
                    "py": coerce_finite_scalar(row["py"], name="row.py") - anchor_py + offset_y,
                    "yaw": coerce_finite_scalar(row["yaw"], name="row.yaw"),
                    "_cycle_index": cycle_index,
                    "_base_row_index": base_row_index,
                }
            )
    return tiled_rows


def _tile_base_stream_rows(
    base_rows: list[dict[str, Any]],
    *,
    cycle_count: int,
    cycle_gap_s: float,
    stride_s: float,
    gt_base_start: float,
    dt_override: float | None = None,
    jitter_sigma_s: float = 0.0,
    base_seed: int = 0,
    seq_id: str = "inline_seq",
    modality: str = "stream",
) -> list[dict[str, Any]]:
    """将单 cycle 的传感器流行（IMU/UWB/VIO）铺叠成多 cycle 的长序列。

    与 _tile_base_gt_rows 不同，传感器流不做位置偏移，只做时间戳偏移。
    每行深拷贝后更新时间戳，并附加 _cycle_index 和 _base_row_index 辅助字段。

    时间戳偏移以 GT 的 base_start 为参考原点，保证传感器流与 GT 之间的
    相对时间对齐不变（例如 UWB 首观测在 GT 起始后 0.05s，铺叠后仍为 0.05s）。

    铁律 5: timestamp 异步 jitter + dt_override 重采样.
    - jitter_sigma_s > 0 时，每行 timestamp += N(0, jitter_sigma_s) (确定性 seed),
      模拟各 modality 内部时钟相对 GT 时钟的异步抖动 (IMU/UWB/VIO 各自独立).
    - dt_override 不为 None 且 > 0 时，跳过 fixture 原生 timestamp，按
      base_row_index * dt_override 重新构造单 cycle 内时戳网格 (等距重采样).
      cycle_stride 用 max(stride_s, N*dt_override) 仍按 stride_s 时间位移，
      保证 cycle 间间隔语义不变 (stride_s 仍以 GT 单 cycle 时长为锚).

    参数：
        base_rows: 单 cycle 的传感器流行列表。
        cycle_count: 铺叠周期数。
        cycle_gap_s: 周期间隔秒数。
        stride_s: 单个 cycle 的时间跨度（不含间隔）。
        gt_base_start: GT 序列的首帧时间戳，作为时间偏移的参考原点。
        dt_override: 铁律 5b 新增. 若为正浮点, 用 base_row_index * dt_override 重采样每行 timestamp.
        jitter_sigma_s: 铁律 5 新增. 每行 timestamp 加 N(0, sigma) 时钟 jitter.
        base_seed: 用于 jitter 确定性 RNG 的种子基.
        seq_id: 用于 jitter 确定性 RNG 区分不同序列.
        modality: "imu"/"uwb"/"vio" 用于 jitter 确定性 RNG 区分不同传感器流.

    返回：
        铺叠后的传感器流行列表，包含辅助字段。

    异常：
        ValueError: base_rows 为空时抛出。
    """
    if not base_rows:
        raise ValueError("base_rows must be non-empty")
    cycle_stride = float(stride_s) + float(cycle_gap_s)
    use_dt_override = (
        dt_override is not None
        and coerce_finite_scalar(float(dt_override), name="dt_override") > 0.0
    )
    if use_dt_override:
        dt_step = float(dt_override)
    # 铁律 5: per-modality jitter RNG, 仅当 sigma > 0 时启用.
    if jitter_sigma_s > 0.0:
        rng_jitter = _make_deterministic_rng(int(base_seed), seq_id, modality, "timestamp_jitter")
    else:
        rng_jitter = None
    tiled_rows: list[dict[str, Any]] = []
    for cycle_index in range(int(cycle_count)):
        offset_t = cycle_index * cycle_stride
        for base_row_index, row in enumerate(base_rows):
            cloned = deepcopy(row)  # 深拷贝避免修改原始 fixture 数据
            if use_dt_override:
                # 铁律 5b: 用 dt_override 等距重采样每行 timestamp (丢弃 fixture 采样间距).
                cycle_local_t = float(base_row_index) * dt_step
            else:
                cycle_local_t = coerce_finite_scalar(row["timestamp"], name="row.timestamp") - float(gt_base_start)
            stamped = cycle_local_t + offset_t
            # 铁律 5: per-modality 异步 jitter (IMU 0.1ms / UWB 0.5ms / VIO 0.2ms)
            if rng_jitter is not None:
                stamped += rng_jitter.gauss(0.0, float(jitter_sigma_s))
            cloned["timestamp"] = _round_normalize(stamped)
            cloned["_cycle_index"] = cycle_index
            cloned["_base_row_index"] = base_row_index
            tiled_rows.append(cloned)
    return tiled_rows


def _transform_gt_rows(tiled_gt_rows: list[dict[str, float]], spec: SimSequenceSpec) -> list[dict[str, float]]:
    """对铺叠后的 GT 行施加几何变换并重新计算航向角。

    参数：
        tiled_gt_rows: 铺叠后的 GT 行列表（含辅助字段）。
        spec: 序列规格，包含变换参数。

    返回：
        变换后的 GT 行列表，航向角已从位置重新推导，数值已四舍五入。
    """
    transformed = []
    for row in tiled_gt_rows:
        px, py = _transform_position(coerce_finite_scalar(row["px"], name="row.px"), coerce_finite_scalar(row["py"], name="row.py"), spec)
        transformed.append(
            {
                "timestamp": coerce_finite_scalar(row["timestamp"], name="row.timestamp"),
                "px": px,
                "py": py,
                "yaw": _transform_yaw(coerce_finite_scalar(row["yaw"], name="row.yaw"), spec),
                "_cycle_index": int(row["_cycle_index"]),
                "_base_row_index": int(row["_base_row_index"]),
            }
        )
    # 几何变换后 yaw 可能与轨迹方向不一致，需要从位置重新推导
    transformed = _recompute_yaw_from_positions(transformed)
    return _round_gt_rows(transformed)


def _resolve_spec_gk_levels(spec: SimSequenceSpec) -> tuple[str, str]:
    """从 axes_override 解析 G/K 档位；缺省回落协议主表默认 G1/K4（欠定，非优几何 K6）。"""
    g_level = "G1"
    k_level = "K4"
    for axis_name, level_name in spec.axes_override:
        if axis_name == "G":
            g_level = str(level_name).strip()
        elif axis_name == "K":
            k_level = str(level_name).strip()
    return g_level, k_level


def _transform_anchor_layout(anchor_layout: dict[str, Any], spec: SimSequenceSpec) -> dict[str, Any]:
    """按 axes_override 的 G/K 生成锚点，并缩放到 workspace_span_m（§8.1/§8.2.1）。

    不再写死 G0/K6 paper 基线。主表默认 G1/K4（欠定压力）；若 override 给 K6/K8
    则允许高锚数消融，但须在 envelope 中显式 allow_high_anchor_count。
    """
    protocol_cfg = load_scene_axis_protocol()
    g_level, k_level = _resolve_spec_gk_levels(spec)
    if k_level not in protocol_cfg["axes"]["K"]:
        raise ValueError(f"unknown K level {k_level!r} for seq={spec.seq_id}")
    if g_level not in protocol_cfg["axes"]["G"]:
        raise ValueError(f"unknown G level {g_level!r} for seq={spec.seq_id}")
    anchor_count = int(protocol_cfg["axes"]["K"][k_level]["anchor_count"])
    paper_anchor_layout, geometry_report = build_anchor_layout(
        anchor_count,
        g_level,
        protocol_cfg["axes"]["G"],
        workspace_span_m=float(spec.workspace_span_m),
    )
    # 协议轨迹模式：锚点已在世界系米制尺度，直接使用，不再投影到迷你 fixture 锚。
    if spec.use_protocol_trajectory:
        projected_layout = dict(paper_anchor_layout)
    else:
        reference_positions = []
        reference_spec = SimSequenceSpec(
            seq_id=spec.seq_id,
            base_seq_id=spec.base_seq_id,
            translation_xy=spec.translation_xy,
            rotation_rad=spec.rotation_rad,
            mirror_x=spec.mirror_x,
            curve_strength=0.0,  # 锚点基础设施不跟随轨迹弯曲
            cycle_count=spec.cycle_count,
            cycle_gap_s=spec.cycle_gap_s,
            use_protocol_trajectory=False,
            envelope_profile="smoke",
        )
        for position in anchor_layout["anchor_positions"]:
            px, py = _transform_position(
                coerce_finite_scalar(position[0], name="position[0]"),
                coerce_finite_scalar(position[1], name="position[1]"),
                reference_spec,
            )
            reference_positions.append([_round_normalize(px), _round_normalize(py)])
        projected_layout = project_anchor_layout_to_reference(
            paper_anchor_layout,
            {
                "anchor_ids": deepcopy(anchor_layout["anchor_ids"]),
                "anchor_positions": reference_positions,
                "layout_id": anchor_layout.get("layout_id"),
            },
        )

    family_id = spec.seq_id.split("_var_")[0]
    projected_layout["fixture_layout_id"] = anchor_layout.get("layout_id")
    projected_layout["base_layout_id"] = family_id
    projected_layout["layout_id"] = spec.seq_id
    projected_layout["source"] = "protocol_geometry" if spec.use_protocol_trajectory else anchor_layout.get("source", "sim_materialized")
    projected_layout["protocol_geometry_level"] = g_level
    projected_layout["protocol_k_level"] = k_level
    projected_layout["geometry_report"] = geometry_report
    projected_layout["workspace_span_m"] = float(spec.workspace_span_m)
    return projected_layout


def _derive_base_uwb_residuals(base_uwb_rows: list[dict[str, Any]], base_gt_rows: list[dict[str, Any]], anchor_layout: dict[str, Any]) -> list[float]:
    """从 fixture 原始数据中提取 UWB 测距残差（观测值 - 几何预测值）。

    残差保留了真实 UWB 传感器的系统偏差特征，物化时叠加到新序列上。

    参数：
        base_uwb_rows: fixture 的 UWB 行列表。
        base_gt_rows: fixture 的 GT 行列表。
        anchor_layout: 锚点布局字典。

    返回：
        残差列表，每个元素是对应 UWB 观测的测距残差（米）。

    异常：
        ValueError: 输入为空时抛出。
    """
    _require_non_empty(base_uwb_rows, name="base_uwb_rows")  # 组4
    gt_rows = _normalize_gt_rows(base_gt_rows)
    if not gt_rows:
        raise ValueError("base_gt_rows must be non-empty after normalization")
    # 组17: build_anchor_lookup 内部已浅拷贝，无需 deepcopy
    anchor_lookup = build_anchor_lookup(anchor_layout)
    residuals = []
    for row in base_uwb_rows:
        anchor_id = row["anchor_id"]
        if anchor_id not in anchor_lookup:
            raise ValueError(f"UWB row references anchor_id '{anchor_id}' not found in anchor_layout")
        pose = _interpolate_pose(gt_rows, coerce_finite_scalar(row["timestamp"], name="row.timestamp"))
        predicted = coerce_finite_scalar(predict_range_to_anchor(pose, anchor_lookup[anchor_id]), name="predicted_range")
        residuals.append(coerce_finite_scalar(row["range"], name="row.range") - predicted)  # 残差 = 观测 - 预测
    return residuals


def _derive_geometric_vio_deltas(gt_rows: list[dict[str, float]], timestamps: list[float]) -> list[tuple[float, float, float]]:
    """从 GT 行中推导指定时间戳序列的几何 VIO 增量。

    参数：
        gt_rows: 已标准化的 GT 行列表，不能为空。
        timestamps: 需要计算增量的时间戳列表，每个值必须有限。

    返回：
        增量列表，每个元素是 (dx, dy, dyaw) 体坐标系增量。首帧增量为 (0,0,0)。

    异常：
        ValueError: gt_rows 为空或 timestamp 含 NaN/Inf 时抛出。
    """
    if not gt_rows:
        raise ValueError("gt_rows must be non-empty")
    deltas = []
    previous_timestamp = None
    for timestamp in timestamps:
        ts = coerce_finite_scalar(timestamp, name="vio delta timestamp")
        if previous_timestamp is None:
            # 首帧增量为零
            deltas.append((0.0, 0.0, 0.0))
        else:
            prev_pose = _interpolate_pose(gt_rows, previous_timestamp)
            curr_pose = _interpolate_pose(gt_rows, ts)
            deltas.append(_body_frame_delta(prev_pose, curr_pose))
        previous_timestamp = ts
    return deltas


def _derive_base_vio_residuals(base_vio_rows: list[dict[str, Any]], base_gt_rows: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    """从 fixture 原始数据中提取 VIO 增量残差（观测增量 - 几何增量）。

    参数：
        base_vio_rows: fixture 的 VIO 行列表。
        base_gt_rows: fixture 的 GT 行列表。

    返回：
        残差列表，每个元素是 (residual_dx, residual_dy, residual_dyaw)。

    异常：
        ValueError: 输入为空或含 NaN/Inf 时抛出。
    """
    _require_non_empty(base_vio_rows, name="base_vio_rows")
    _require_non_empty(base_gt_rows, name="base_gt_rows")
    gt_rows = _normalize_gt_rows(base_gt_rows)
    timestamps = [coerce_finite_scalar(row["timestamp"], name="row.timestamp") for row in base_vio_rows]
    for ts in timestamps:
        coerce_finite_scalar(ts, name="vio timestamp")
    geometric = _derive_geometric_vio_deltas(gt_rows, timestamps)
    residuals = []
    for row, (dx, dy, dyaw) in zip(base_vio_rows, geometric, strict=True):
        residuals.append(
            (
                coerce_finite_scalar(row["dx"], name="dx") - float(dx),
                coerce_finite_scalar(row["dy"], name="dy") - float(dy),
                angle_delta_rad(coerce_finite_scalar(row["dyaw"], name="row.dyaw"), float(dyaw)),
            )
        )
    return residuals


def _derive_imu_rows_from_gt(gt_rows: list[dict[str, Any]], timestamps: list[float]) -> list[dict[str, float]]:
    """从 GT 行中推导指定时间戳序列的 IMU 观测值（加速度和角速度）。

    推导过程：
    1. 对 GT 位置做插值，得到各时间戳的位姿。
    2. 用数值差分（首尾帧前向/后向差分，中间帧中心差分）计算世界坐标系速度。
    3. 对速度再做差分得到世界坐标系加速度。
    4. 将加速度旋转到体坐标系，角速度从 yaw 差分得到。

    参数：
        gt_rows: GT 行列表（未标准化也可以，函数内部会标准化）。
        timestamps: 需要推导 IMU 观测的时间戳列表。

    返回：
        推导出的 IMU 行列表，每行包含 timestamp、ax、ay、gz。

    异常：
        ValueError: timestamps 含 NaN/Inf 时抛出。

    注意：
        本函数返回全精度值（未 round），由调用方负责 round 收口，
        避免双重 round 导致精度损失。
    """
    norm_gt = _normalize_gt_rows(gt_rows)
    for ts in timestamps:
        coerce_finite_scalar(ts, name="imu derived timestamp")
    # 预构建 GT 时间戳索引，让 _interpolate_pose 用 bisect 做 O(log N) 查找。
    # _materialize_imu_rows 每个 cycle 都会调用本函数，未优化时是 O(N²) 热路径。
    gt_timestamp_index = _build_gt_timestamp_index(norm_gt)
    return _derive_imu_rows_from_normalized_gt(
        norm_gt, timestamps, gt_timestamp_index=gt_timestamp_index,
    )


def _derive_imu_rows_from_normalized_gt(
    norm_gt: list[dict[str, float]],
    timestamps: list[float],
    *,
    gt_timestamp_index: list[float] | None = None,
) -> list[dict[str, float]]:
    """从已标准化的 GT 行推导 IMU 观测值（_derive_imu_rows_from_gt 的内部快路径）。

    与 _derive_imu_rows_from_gt 的区别：本函数假设 gt_rows 已通过 _normalize_gt_rows
    标准化（所有字段为 float、按 timestamp 升序排序），跳过重复标准化开销。
    _materialize_imu_rows 每个 cycle 都会调用本函数，预标准化一次可避免 132 次
    重复 O(N log N) 排序（N≈50K 时单次标准化 ~2ms，132 次累计 ~260ms）。

    参数：
        norm_gt: 已标准化的 GT 行列表（字段为 float、按 timestamp 升序）。
        timestamps: 需要推导 IMU 观测的时间戳列表。
        gt_timestamp_index: 可选的预构建时间戳索引，避免每次调用重建。

    返回：
        推导出的 IMU 行列表，每行包含 timestamp、ax、ay、gz。

    【前提指导 §5.1 — 重力处理（减去/保留在模型中）全员同一】
    本函数是仿真层"五方法同一 IMU 物理"的源头（实际实现：ekf / robust_ekf / fgo /
    lstm_ekf / liquid_ekf，详见 sensors/imu_model.py §5.1 注释），约定如下：

    - 2D 平面 ENU 水平运动模型：GT 的 px/py 是水平面上位置，yaw 是绕垂直轴的航向；
      GT 不含垂直分量，加速度也在水平面内。
    - 世界系水平加速度 ``ax_world``/``ay_world`` 来自 GT 位置二阶差分，
      **是水平面内的运动学加速度**（non-gravitational，已隐式不含重力投影，
      因为重力沿垂直方向作用，与水平面解耦）。
    - 机体系 ``ax``/``ay`` 通过当前 yaw 把世界系水平加速度旋转回机体前/侧向，
      旋转仅作用于水平分量，重力项始终不出现在 ax/ay 上。
    - ``gz`` 由 yaw 的中心差分得到，单位 rad/s，垂直轴角速度，与 §5.1 一致。

    这一约定与 predict_step.run_predict_step 配合：predict 端把 ax/ay 当作机体系水平比力，
    直接做体→世界旋转后积分到速度/位置，**不做 any 形式的重力补偿或投影**，
    也不把重力塞进状态向量某个分量再减掉。所有方法经由 run_predict_step 共享这一约定，
    实现 §5.1「重力处理全员同一」的前提。
    """
    for ts in timestamps:
        coerce_finite_scalar(ts, name="imu derived timestamp")
    if gt_timestamp_index is None:
        gt_timestamp_index = _build_gt_timestamp_index(norm_gt)
    poses = [_interpolate_pose(norm_gt, float(ts), timestamp_index=gt_timestamp_index) for ts in timestamps]
    # 第一步：计算各帧的世界坐标系速度
    velocities: list[tuple[float, float]] = []
    for index, pose in enumerate(poses):
        if len(poses) == 1:
            velocities.append((0.0, 0.0))
        elif index == 0:
            # 首帧用前向差分
            next_pose = poses[1]
            dt = coerce_finite_scalar(next_pose["timestamp"], name="next_pose.timestamp") - coerce_finite_scalar(pose["timestamp"], name="pose.timestamp")
            velocities.append((_safe_diff_divide(coerce_finite_scalar(next_pose["px"], name="next_pose.px") - coerce_finite_scalar(pose["px"], name="pose.px"), dt), _safe_diff_divide(coerce_finite_scalar(next_pose["py"], name="next_pose.py") - coerce_finite_scalar(pose["py"], name="pose.py"), dt)))
        elif index == len(poses) - 1:
            # 末帧用后向差分
            prev_pose = poses[index - 1]
            dt = coerce_finite_scalar(pose["timestamp"], name="pose.timestamp") - coerce_finite_scalar(prev_pose["timestamp"], name="prev_pose.timestamp")
            velocities.append((_safe_diff_divide(coerce_finite_scalar(pose["px"], name="pose.px") - coerce_finite_scalar(prev_pose["px"], name="prev_pose.px"), dt), _safe_diff_divide(coerce_finite_scalar(pose["py"], name="pose.py") - coerce_finite_scalar(prev_pose["py"], name="prev_pose.py"), dt)))
        else:
            # 中间帧用中心差分（更精确）
            prev_pose = poses[index - 1]
            next_pose = poses[index + 1]
            dt = coerce_finite_scalar(next_pose["timestamp"], name="next_pose.timestamp") - coerce_finite_scalar(prev_pose["timestamp"], name="prev_pose.timestamp")
            velocities.append((_safe_diff_divide(coerce_finite_scalar(next_pose["px"], name="next_pose.px") - coerce_finite_scalar(prev_pose["px"], name="prev_pose.px"), dt), _safe_diff_divide(coerce_finite_scalar(next_pose["py"], name="next_pose.py") - coerce_finite_scalar(prev_pose["py"], name="prev_pose.py"), dt)))
    # 第二步：从速度差分推导加速度，并旋转到体坐标系
    imu_rows: list[dict[str, float]] = []
    for index, pose in enumerate(poses):
        yaw = coerce_finite_scalar(pose["yaw"], name="pose.yaw")
        cos_value = math.cos(yaw)
        sin_value = math.sin(yaw)
        if len(poses) == 1:
            ax_world = 0.0
            ay_world = 0.0
            gz = 0.0
        elif index == 0:
            # 首帧：前向差分计算加速度
            next_pose = poses[1]
            next_vel = velocities[1]
            curr_vel = velocities[0]
            dt = coerce_finite_scalar(next_pose["timestamp"], name="next_pose.timestamp") - coerce_finite_scalar(pose["timestamp"], name="pose.timestamp")
            ax_world = _safe_diff_divide(next_vel[0] - curr_vel[0], dt)
            ay_world = _safe_diff_divide(next_vel[1] - curr_vel[1], dt)
            gz = _safe_diff_divide(angle_delta_rad(coerce_finite_scalar(next_pose["yaw"], name="next_pose.yaw"), yaw), dt)
        elif index == len(poses) - 1:
            # 末帧用后向差分
            prev_pose = poses[index - 1]
            curr_vel = velocities[index]
            prev_vel = velocities[index - 1]
            dt = coerce_finite_scalar(pose["timestamp"], name="pose.timestamp") - coerce_finite_scalar(prev_pose["timestamp"], name="prev_pose.timestamp")
            ax_world = _safe_diff_divide(curr_vel[0] - prev_vel[0], dt)
            ay_world = _safe_diff_divide(curr_vel[1] - prev_vel[1], dt)
            gz = _safe_diff_divide(angle_delta_rad(yaw, coerce_finite_scalar(prev_pose["yaw"], name="prev_pose.yaw")), dt)
        else:
            # 中间帧用中心差分（更精确，与速度差分策略一致）
            prev_pose = poses[index - 1]
            next_pose = poses[index + 1]
            dt = coerce_finite_scalar(next_pose["timestamp"], name="next_pose.timestamp") - coerce_finite_scalar(prev_pose["timestamp"], name="prev_pose.timestamp")
            next_vel = velocities[index + 1]
            prev_vel = velocities[index - 1]
            ax_world = _safe_diff_divide(next_vel[0] - prev_vel[0], dt)
            ay_world = _safe_diff_divide(next_vel[1] - prev_vel[1], dt)
            gz = _safe_diff_divide(angle_delta_rad(coerce_finite_scalar(next_pose["yaw"], name="next_pose.yaw"), coerce_finite_scalar(prev_pose["yaw"], name="prev_pose.yaw")), dt)
        # 世界坐标系加速度旋转到体坐标系（返回全精度，由调用方 round）
        imu_rows.append(
            {
                "timestamp": coerce_finite_scalar(pose["timestamp"], name="pose.timestamp"),
                "ax": cos_value * ax_world + sin_value * ay_world,
                "ay": -sin_value * ax_world + cos_value * ay_world,
                "gz": gz,
            }
        )
    return imu_rows


def _derive_base_imu_residuals(base_imu_rows: list[dict[str, Any]], base_gt_rows: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    """从 fixture 原始数据中提取 IMU 观测残差（观测值 - 几何推导值）。

    参数：
        base_imu_rows: fixture 的 IMU 行列表。
        base_gt_rows: fixture 的 GT 行列表。

    返回：
        残差列表，每个元素是 (residual_ax, residual_ay, residual_gz)。

    异常：
        ValueError: 输入为空或含 NaN/Inf 时抛出。
    """
    _require_non_empty(base_imu_rows, name="base_imu_rows")
    _require_non_empty(base_gt_rows, name="base_gt_rows")
    timestamps = [coerce_finite_scalar(row["timestamp"], name="row.timestamp") for row in base_imu_rows]
    for ts in timestamps:
        coerce_finite_scalar(ts, name="imu timestamp")
    derived_rows = _derive_imu_rows_from_gt(base_gt_rows, timestamps)
    residuals = []
    for raw_row, derived_row in zip(base_imu_rows, derived_rows, strict=True):
        residuals.append(
            (
                coerce_finite_scalar(raw_row["ax"], name="ax") - float(derived_row["ax"]),
                coerce_finite_scalar(raw_row["ay"], name="ay") - float(derived_row["ay"]),
                coerce_finite_scalar(raw_row["gz"], name="gz") - float(derived_row["gz"]),
            )
        )
    return residuals


def _normalize_noise_spec(noise_spec: SimNoiseSpec | dict[str, Any] | None) -> SimNoiseSpec:
    """将各种形式的噪声规格统一转换为 SimNoiseSpec 实例。

    参数：
        noise_spec: 可以是 SimNoiseSpec 实例、字典或 None。

    返回：
        标准化后的 SimNoiseSpec 实例。

    异常：
        TypeError: 输入类型不支持或字段类型不匹配时抛出。
        ValueError: std 字段为 NaN/Inf/负数或含未知键时抛出。
    """
    if noise_spec is None:
        return DEFAULT_SIM_NOISE_SPEC
    if isinstance(noise_spec, SimNoiseSpec):
        return noise_spec
    if isinstance(noise_spec, dict):
        # 组15: 对未知键给出友好错误信息
        valid_keys = {
            "uwb_range_std_m", "imu_ax_std", "imu_ay_std", "imu_gz_std",
            "imu_accel_bias_instability", "imu_gyro_bias_instability",
            "imu_accel_rrw_mps2_per_sqrt_s", "imu_gyro_rrw_rads_per_sqrt_s",
            "vio_dx_std", "vio_dy_std", "vio_dyaw_std",
            "vio_scale_drift_rate", "base_seed",
        }
        unknown = set(noise_spec.keys()) - valid_keys
        if unknown:
            raise ValueError(f"unknown noise spec keys: {sorted(unknown)}; valid keys: {sorted(valid_keys)}")
        try:
            spec = SimNoiseSpec(**noise_spec)
        except TypeError as exc:
            raise TypeError(f"invalid noise spec dict: {exc}") from exc
        # 组15: 校验所有 std/bias/drift 字段 >= 0 且有限
        for field_name in ("uwb_range_std_m", "imu_ax_std", "imu_ay_std", "imu_gz_std",
                           "imu_accel_bias_instability", "imu_gyro_bias_instability",
                           "imu_accel_rrw_mps2_per_sqrt_s", "imu_gyro_rrw_rads_per_sqrt_s",
                           "vio_dx_std", "vio_dy_std", "vio_dyaw_std", "vio_scale_drift_rate"):
            value = getattr(spec, field_name)
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite, got {value!r}")
            if float(value) < 0.0:
                raise ValueError(f"{field_name} must be >= 0, got {value!r}")
        # 组15: base_seed 必须是 int（排除 bool）
        if not isinstance(spec.base_seed, int) or isinstance(spec.base_seed, bool):
            raise TypeError(f"base_seed must be int, got {type(spec.base_seed).__name__}: {spec.base_seed!r}")
        return spec
    raise TypeError(f"unsupported noise spec: {type(noise_spec)!r}")


def _seeded_gaussian(base_seed: int, sigma: float, *parts: object) -> float:
    """基于确定性种子生成一个高斯随机数。

    使用 SHA-256 将种子和附加标识符混合，再用标准库 Random 的高斯
    采样，保证相同输入始终产生相同输出。

    参数：
        base_seed: 种子基数值。
        sigma: 高斯噪声标准差；≤0 时返回 0。
        *parts: 附加标识符，用于区分不同流/字段/行。

    返回：
        采样得到的高斯随机数。

    异常：
        ValueError: sigma 为 NaN/Inf 时抛出。
    """
    sigma = coerce_finite_scalar(sigma, name="sigma")
    base_seed = int(base_seed)
    if sigma <= 0.0:
        return 0.0
    # 将所有标识符拼成字符串后做 SHA-256，取前 8 字节作为 Random 种子
    digest = hashlib.sha256("|".join([str(base_seed), *[str(part) for part in parts]]).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big", signed=False)).gauss(0.0, float(sigma))


def _seeded_uniform(base_seed: int, amplitude: float, *parts: object) -> float:
    """基于确定性种子生成一个均匀分布随机数，范围 [-amplitude, +amplitude]。

    用于序列级 bias 不稳定性模拟：在序列内生成一个恒定偏置，
    不同序列间确定性变化。参考 MATLAB imuSensor 的 BiasInstability 实现。

    参数：
        base_seed: 种子基数值。
        amplitude: 偏置幅度；≤0 时返回 0。
        *parts: 附加标识符，用于区分不同流/字段/序列。

    返回：
        采样得到的均匀随机数，范围 [-amplitude, +amplitude]。

    异常：
        ValueError: amplitude 为 NaN/Inf 时抛出。
    """
    amplitude = coerce_finite_scalar(amplitude, name="amplitude")
    base_seed = int(base_seed)
    if amplitude <= 0.0:
        return 0.0
    digest = hashlib.sha256("|".join([str(base_seed), *[str(part) for part in parts]]).encode("utf-8")).digest()
    # uniform(-amplitude, +amplitude)
    return random.Random(int.from_bytes(digest[:8], "big", signed=False)).uniform(-float(amplitude), float(amplitude))


def _materialize_uwb_rows(
    tiled_uwb_rows: list[dict[str, Any]],
    *,
    transformed_gt_rows: list[dict[str, float]],
    transformed_anchor_layout: dict[str, Any],
    base_uwb_residuals: list[float],
    noise_spec: SimNoiseSpec | dict[str, Any] | None = None,
    seq_id: str = "inline_seq",
) -> list[dict[str, Any]]:
    """物化铺叠后的 UWB 行：几何预测 + fixture 残差 + 高斯噪声。

    参数：
        tiled_uwb_rows: 铺叠后的 UWB 行列表（含辅助字段）。
        transformed_gt_rows: 变换后的 GT 行列表。
        transformed_anchor_layout: 变换后的锚点布局。
        base_uwb_residuals: fixture 中提取的 UWB 残差列表。
        noise_spec: 噪声规格。
        seq_id: 序列编号，用于噪声种子区分。

    返回：
        物化后的 UWB 行列表。每行包含 8 个核心字段：
        timestamp / anchor_id / range / valid / quality（向后兼容旧消费者），
        以及铁律 1 新增的 4 个 TWR 原始时戳 + tof：
        tof / twr_poll_tx / twr_poll_rx / twr_resp_tx / twr_resp_rx。

    异常：
        ValueError: base_uwb_residuals 为空或测距合计非有限时抛出。
        IndexError: _base_row_index 越界时抛出。
    """
    noise = _normalize_noise_spec(noise_spec)
    # 第 4 轮审查 MEDIUM-2 修复：空输入校验对称性。与 _materialize_vio_rows
    # 和 _materialize_imu_rows 保持一致——若 tiled_uwb_rows 为空，直接返回
    # 空列表（不抛错，避免无意义的 residuals 校验）；非空时校验 residuals 非空。
    if not tiled_uwb_rows:
        return []
    # build_anchor_lookup 内部已浅拷贝，无需 deepcopy（组17）
    anchor_lookup = build_anchor_lookup(transformed_anchor_layout)
    target_anchor_ids = list(transformed_anchor_layout.get("anchor_ids") or [])
    _require_non_empty(base_uwb_residuals, name="base_uwb_residuals")  # 组4
    base_row_count = len(base_uwb_residuals)  # 组4: 移除 max(1, ...) 静默兜底
    # D-7 patch: SimNoiseSpec 缺失字段显式 logger.warning 审计.
    # UWB noise group (uwb_range_std_m) 对应 fixture 观测字段
    # timestamp/anchor_id/range/valid/quality, 其中 valid/quality 用 row.get 静默回退
    # (line 1649-1650), 缺失即被静默跳过——此处给审计一条可见线索.
    # 铁律 1: 显式 不在 审计集合里加 twr_*/tof 字段, 因为这些是物化端构造的, fixture 不会携带.
    _uwb_fixture_audit_fields = ("timestamp", "anchor_id", "range", "valid", "quality")
    _first_uwb_row = tiled_uwb_rows[0]
    _uwb_missing = [k for k in _uwb_fixture_audit_fields if k not in _first_uwb_row]
    if _uwb_missing:
        SimNoiseSpec.warn_missing_fixture_fields(
            _uwb_missing,
            sensor_kind="uwb",
            seq_id=seq_id,
        )
    # 预构建 GT 时间戳索引，让 _interpolate_pose 用 bisect 做 O(log N) 查找
    # （热路径优化：避免每行 UWB 都对 N≈50000 的 GT 列表做线性扫描）。
    gt_timestamp_index = _build_gt_timestamp_index(transformed_gt_rows)
    # 预生成确定性 RNG 流，替代逐行 _seeded_gaussian 的 SHA-256 开销。
    # 仅当 noise.uwb_range_std_m > 0 时启用，保持与 IMU 路径一致的优化模式。
    if noise.uwb_range_std_m > 0.0:
        rng_uwb_range = _make_deterministic_rng(noise.base_seed, seq_id, "uwb", "range_stream")
    else:
        rng_uwb_range = None
    # 铁律 1: UWB 必须 输出原始 TWR 4 时戳 + tof + range.
    # TWR (Two-Way Ranging) 协议物理模型:
    #   tof = (resp_rx - poll_tx - T_reply) / 2, range = c * tof
    # Decawave 标准 TWR 协议 4 时戳:
    #   twr_poll_tx  : tag 发 Poll 时刻
    #   twr_poll_rx  : anchor 收 Poll 时刻 (poll_tx + tof)
    #   twr_resp_tx  : anchor 发 Resp 时刻 (poll_rx + T_reply)
    #   twr_resp_rx  : tag 收 Resp 时刻 (resp_tx + tof + clock_jitter)
    # T_reply = 100μs (Decawave 推荐). clock_jitter ~ N(0, 1ns) 模拟 tag 时钟噪声.
    # 时戳噪声量级 ~ ns 量级 (3e-9 s), 远小于 range 噪声引起的 tof 偏差 (~30cm/3e8 = 1ns).
    _TWR_C = 299_702_547.0  # 光速 m/s
    _TWR_REPLY_S = 1e-4  # Decawave 标准 T_reply = 100 μs
    _TWR_CLOCK_JITTER_SIGMA_S = 1e-9  # tag 时钟噪声 ~1 ns
    rng_uwb_clock_jitter = _make_deterministic_rng(noise.base_seed, seq_id, "uwb", "twr_clock_jitter_stream")
    # 锚点回退告警去重集合：每个 (seq_id, anchor_id) 组合只告警一次，
    # 避免在长序列下产生数千行重复 warning（性能与日志噪声双重问题）。
    warned_fallback_pairs: set[tuple[str, str]] = set()
    rows = []
    for row in tiled_uwb_rows:
        row_timestamp = coerce_finite_scalar(row["timestamp"], name="row.timestamp")
        anchor_id = row["anchor_id"]
        # §5.3: 短时 IMU 主导段——UWB 全锚 NLOS 持续段。段内所有观测（无论
        # 来自哪个 anchor）均被标记为 NLOS：valid=False、quality=0。
        # range/tof 仍按几何+噪声物化（保持下游字段完整），但 valid=False
        # 让 fusion 端跳过该观测，IMU 主导该时段的姿态推进。
        if _is_timestamp_in_outage(row_timestamp, noise.uwb_outage_segments):
            # 几何预测仍计算（保持 range 字段非零、便于审计），但 valid=False
            pose = _interpolate_pose(
                transformed_gt_rows,
                row_timestamp,
                timestamp_index=gt_timestamp_index,
            )
            if target_anchor_ids and anchor_id not in anchor_lookup:
                # 与非 outage 路径一致的锚点回退处理
                anchor_cursor = int(row["_cycle_index"]) * base_row_count + int(row["_base_row_index"])
                anchor_id = target_anchor_ids[anchor_cursor % len(target_anchor_ids)]
            if anchor_id in anchor_lookup:
                predicted_range = coerce_finite_scalar(
                    predict_range_to_anchor(pose, anchor_lookup[anchor_id]),
                    name="predicted_range_outage",
                )
            else:
                predicted_range = 0.0
            measured_range = max(0.0, predicted_range + (
                rng_uwb_range.gauss(0.0, noise.uwb_range_std_m) if rng_uwb_range is not None else 0.0
            ))
            ideal_tof = measured_range / _TWR_C
            twr_poll_tx = row_timestamp - 1e-5
            twr_poll_rx = twr_poll_tx + ideal_tof
            twr_resp_tx = twr_poll_rx + _TWR_REPLY_S
            twr_resp_rx = twr_resp_tx + ideal_tof + (
                rng_uwb_clock_jitter.gauss(0.0, _TWR_CLOCK_JITTER_SIGMA_S)
            )
            rows.append(
                {
                    "timestamp": _round_normalize(row_timestamp),
                    "anchor_id": anchor_id,
                    "range": _round_normalize(measured_range),
                    "valid": False,  # §5.3 全锚 NLOS 段
                    "quality": 0.0,  # §5.3 段内观测质量置 0
                    "tof": _round_normalize(ideal_tof, precision=9),
                    "twr_poll_tx": _round_normalize(twr_poll_tx, precision=9),
                    "twr_poll_rx": _round_normalize(twr_poll_rx, precision=9),
                    "twr_resp_tx": _round_normalize(twr_resp_tx, precision=9),
                    "twr_resp_rx": _round_normalize(twr_resp_rx, precision=9),
                }
            )
            continue
        pose = _interpolate_pose(
            transformed_gt_rows,
            row_timestamp,
            timestamp_index=gt_timestamp_index,
        )
        fell_back = False  # 组16: 标记是否发生锚点回退
        if target_anchor_ids and anchor_id not in anchor_lookup:
            # 旧 fixture 锚点 ID 可能不在当前协议布局中；回退到当前布局锚点轮询（主表 K3/K4）。
            warn_key = (seq_id, str(anchor_id))
            if warn_key not in warned_fallback_pairs:
                warned_fallback_pairs.add(warn_key)
                warnings.warn(
                    f"UWB anchor_id '{anchor_id}' not in layout; falling back to target anchor for seq={seq_id}",
                    stacklevel=2,
                )
            anchor_cursor = int(row["_cycle_index"]) * base_row_count + int(row["_base_row_index"])
            anchor_id = target_anchor_ids[anchor_cursor % len(target_anchor_ids)]
            fell_back = True
        if anchor_id not in anchor_lookup:
            raise ValueError(f"UWB anchor_id '{anchor_id}' not found in transformed anchor layout")
        predicted = coerce_finite_scalar(predict_range_to_anchor(pose, anchor_lookup[anchor_id]), name="predicted_range")
        # 从 fixture 残差中按 _base_row_index 取对应行的残差
        if fell_back:
            # 组16: 回退锚点与原 anchor 不匹配，残差置 0 避免注入错误偏差
            residual = 0.0
        else:
            residual = float(base_uwb_residuals[_require_index_in_bounds(row["_base_row_index"], base_uwb_residuals, name="base_uwb_residuals")])  # 组5
        # 叠加确定性高斯噪声：用预生成的 RNG 流采样，与逐行 _seeded_gaussian
        # 在统计分布上等价但不保证逐样本相同（仅影响噪声的具体取值，不影响
        # 均值/方差/确定性可复现性——同一 seq_id 的 RNG 流始终产生相同序列）。
        if rng_uwb_range is not None:
            extra_noise = rng_uwb_range.gauss(0.0, noise.uwb_range_std_m)
        else:
            extra_noise = 0.0
        # 组2/组8: 先校验合计有限，再 max(0.0, ...)，避免 max 静默吞 NaN
        total = predicted + residual + extra_noise
        if not math.isfinite(total):
            raise ValueError(f"uwb range total must be finite, got {total!r} (predicted={predicted!r}, residual={residual!r}, noise={extra_noise!r})")
        measured_range = max(0.0, total)
        # 铁律 1: 构造 TWR 4 时戳. ideal_tof 用 measured range 反推 (含 NLOS jitter 后真实 ToF).
        # poll_tx 略早于 row timestamp (Poll 在 timestamp 前 10μs 发出), 简化与 anchor 同 timestamp 同步语义.
        # 紧耦合 scoring 端按 row timestamp ±5ms 同步窗聚合 (Stage B), 不依赖 twr_poll_tx 作为同步锚.
        ideal_tof = measured_range / _TWR_C
        twr_poll_tx = float(row["timestamp"]) - 1e-5
        twr_poll_rx = twr_poll_tx + ideal_tof
        twr_resp_tx = twr_poll_rx + _TWR_REPLY_S
        # resp_rx 叠加 ns 量级时钟噪声, 模拟 tag Decawave 时钟 jitter (决定 TWR 精度上限 ~1cm).
        twr_clock_jitter = rng_uwb_clock_jitter.gauss(0.0, _TWR_CLOCK_JITTER_SIGMA_S)
        twr_resp_rx = twr_resp_tx + ideal_tof + twr_clock_jitter
        rows.append(
            {
                "timestamp": _round_normalize(coerce_finite_scalar(row["timestamp"], name="row.timestamp")),
                "anchor_id": anchor_id,
                # 测距值 = 几何预测 + fixture 残差 + 高斯噪声，下限 0
                "range": _round_normalize(max(0.0, total)),
                "valid": _coerce_valid_flag(row.get("valid", True)),
                "quality": coerce_finite_scalar(row.get("quality", 1.0), name="row.quality"),
                # 铁律 1: 原始 TWR 4 时戳 + tof, fusion 端可重建多 anchor 同步关系.
                # TWR 时戳基于最终 range 反推 ideal_tof, 然后叠加 ns 量级时钟噪声.
                # 紧耦合 scoring 端 (Stage B fusion_runner) 用同 timestamp ±5ms 窗聚合多 anchor.
                "tof": _round_normalize(ideal_tof, precision=9),
                "twr_poll_tx": _round_normalize(twr_poll_tx, precision=9),
                "twr_poll_rx": _round_normalize(twr_poll_rx, precision=9),
                "twr_resp_tx": _round_normalize(twr_resp_tx, precision=9),
                "twr_resp_rx": _round_normalize(twr_resp_rx, precision=9),
            }
        )
    return rows


def _materialize_vio_rows(
    tiled_vio_rows: list[dict[str, Any]],
    *,
    transformed_gt_rows: list[dict[str, float]],
    base_vio_residuals: list[tuple[float, float, float]],
    noise_spec: SimNoiseSpec | dict[str, Any] | None = None,
    seq_id: str = "inline_seq",
) -> list[dict[str, Any]]:
    """物化铺叠后的 VIO 行：几何增量 + 尺度漂移 + fixture 残差 + 高斯噪声。

    在 cycle 边界处（每个 cycle 的第一帧），VIO 增量设为 (0,0,0)，
    不叠加残差，quality 设为 0，与 IMU 按 cycle 分组处理的语义一致。

    尺度漂移模拟纯视觉里程计的累积尺度误差：
    scale_factor = 1.0 + drift_rate * elapsed_t
    dx/dy 增量乘以尺度因子，模拟 VIO 估计的距离随时间系统性偏离真实距离。
    参考 Scaramuzza & Fraundorfer "Visual Odometry" Tutorial (2011) 中
    报告的单目 VO 尺度漂移率约 1%/m。

    §6.2 #4 高置信增量 (V0 nominal_vio) 子机制对应代码层落点说明：
    V0 时所有退化触发分支都不激活——无 cycle 切换、无 vio_outage_segments 命中、
    vio_scale_drift_rate 标称、drift_bias_sigma_mps=0、vio_dx_std/dy_std/dyaw_std 低噪声、
    visual_levels.py 退化函数也都不触发——物化输出 dx/dy/dyaw = 几何增量 + 标称噪声。
    三估计器 _handle_vio 成功路径 EKF L1188/RobustEKF L696/FGO L2224 同口径接受高置信更新。
    与 §6.2 #5 退化段/#6 短时中断/#7 尺度漂移/#8 偏置型/#9 突发关联/#10 重定位跳变是
    "全部退化不触发"基线合取，故无 V0 专属代码块；标称物化路径即 #4 落点。

    参数：
        tiled_vio_rows: 铺叠后的 VIO 行列表（含辅助字段）。
        transformed_gt_rows: 变换后的 GT 行列表。
        base_vio_residuals: fixture 中提取的 VIO 残差列表。
        noise_spec: 噪声规格。
        seq_id: 序列编号，用于噪声种子区分。

    返回：
        物化后的 VIO 行列表。

    异常：
        ValueError: base_vio_residuals 为空或增量合计非有限时抛出。
        IndexError: _base_row_index 越界时抛出。
    """
    noise = _normalize_noise_spec(noise_spec)
    if tiled_vio_rows:
        _require_non_empty(base_vio_residuals, name="base_vio_residuals")  # 组4
    # D-7 patch: SimNoiseSpec 缺失字段显式 logger.warning 审计.
    # VIO noise group (vio_dx_std/vio_dy_std/vio_dyaw_std/vio_scale_drift_rate)
    # 对应 fixture 观测字段 timestamp/dx/dy/dyaw/quality. 铁律 3 删除
    # tracked_features/reproj_err: 这些字段不再由 sim 输出, 下游若依赖需重写.
    # TODO(STAGE-B/D): 下游 event_builder.py / field_mapper.py / ntu_viral_reader.py
    # / feature_builder.py / model_factory.py 仍依赖 tracked_features/reproj_err
    # (grep "tracked_features|reproj_err" src/), 由 Stage B 子代理统一去除依赖.
    if tiled_vio_rows:
        _vio_fixture_audit_fields = (
            "timestamp", "dx", "dy", "dyaw",
            "quality",  # 铁律 3: 删 tracked_features / reproj_err, 仅保留 5 字段
        )
        _first_vio_row = tiled_vio_rows[0]
        _vio_missing = [k for k in _vio_fixture_audit_fields if k not in _first_vio_row]
        if _vio_missing:
            SimNoiseSpec.warn_missing_fixture_fields(
                _vio_missing,
                sensor_kind="vio",
                seq_id=seq_id,
            )
    # 序列级尺度漂移方向：不同序列有不同的漂移方向（正或负），
    # 用确定性种子采样，保证可复现。
    scale_drift_direction = 1.0
    if noise.vio_scale_drift_rate > 0.0:
        scale_drift_direction = _seeded_uniform(noise.base_seed, 1.0, seq_id, "vio", "scale_drift_dir")
    # 序列起始时间，用于计算累积漂移
    seq_start_t = coerce_finite_scalar(tiled_vio_rows[0]["timestamp"], name="tiled_vio_rows[0].timestamp") if tiled_vio_rows else 0.0
    # 预构建 GT 时间戳索引，让 _interpolate_pose 用 bisect 做 O(log N) 查找。
    # VIO 每行调用 2 次 _interpolate_pose（prev/curr），是热路径。
    gt_timestamp_index = _build_gt_timestamp_index(transformed_gt_rows)
    # 预生成确定性 RNG 流，替代逐行 3 次 _seeded_gaussian 的 SHA-256 开销。
    # 与 IMU 路径一致的优化模式：同 seq_id 的 RNG 流始终产生相同序列，
    # 保证可复现性；统计分布与原 _seeded_gaussian 等价。
    if noise.vio_dx_std > 0.0:
        rng_vio_dx = _make_deterministic_rng(noise.base_seed, seq_id, "vio", "dx_stream")
    else:
        rng_vio_dx = None
    if noise.vio_dy_std > 0.0:
        rng_vio_dy = _make_deterministic_rng(noise.base_seed, seq_id, "vio", "dy_stream")
    else:
        rng_vio_dy = None
    if noise.vio_dyaw_std > 0.0:
        rng_vio_dyaw = _make_deterministic_rng(noise.base_seed, seq_id, "vio", "dyaw_stream")
    else:
        rng_vio_dyaw = None
    rows = []
    previous_timestamp = None
    previous_cycle_index = None
    for row in tiled_vio_rows:
        cycle_index = int(row.get("_cycle_index", 0))
        # cycle 边界处：VIO 增量设为 (0,0,0)，不叠加 residual。
        # 这与 IMU 按 cycle 分组处理的语义一致：每个 cycle 的第一帧是参考帧。
        if previous_timestamp is None or cycle_index != previous_cycle_index:
            # §6.2 #10 重定位跳变子机制对应代码层落点：
            # 关键帧重置（cycle 切换）导致 VIO 不连续，dx/dy/dyaw=0+quality=0，
            # 与 #6 短时中断（vio_outage_segments 段内连续零增量）是两个独立代码路径。
            # 全员同一：三估计器 _handle_vio vio_quality_zero 路径同口径跳过更新并重置参考位姿。
            # cycle 边界帧：零增量帧应标记为无效质量，避免语义矛盾。
            rows.append(
                {
                    "timestamp": _round_normalize(coerce_finite_scalar(row["timestamp"], name="row.timestamp")),
                    "dx": 0.0,
                    "dy": 0.0,
                    "dyaw": 0.0,
                    "quality": 0.0,  # 零增量帧质量标记为 0
                    # 铁律 3: 删除 tracked_features / reproj_err (VIO 不再输出此两字段).
                }
            )
        else:
            prev_pose = _interpolate_pose(transformed_gt_rows, previous_timestamp, timestamp_index=gt_timestamp_index)
            curr_pose = _interpolate_pose(
                transformed_gt_rows,
                coerce_finite_scalar(row["timestamp"], name="row.timestamp"),
                timestamp_index=gt_timestamp_index,
            )
            # §5.3: 短时 IMU 主导段——VIO 持续中断段。段内 VIO 行按"设备失能"语义
            # 输出 dx/dy/dyaw=0、quality=0（与 cycle 边界帧同口径，但持续段更长）。
            # pose 插值仍执行以保持后续帧 prev_pose 取值连续，但本帧几何增量被覆盖。
            # §6.2 #6 短时中断子机制对应代码层落点：
            # 跟踪丢失若干间隔（vio_outage_segments），无增量输出，逼测「无视觉更新」全员一致。
            # 三估计器 _handle_vio vio_quality_zero 路径同口径跳过更新，与 §6.4 #20「不伪造增量」合取：
            # §6.4 #20 关键帧失效不伪造增量+全员同一「跳过视觉更新」硬子要求对应代码层落点——
            # EKF ekf_core.py vio_quality_zero 拒绝路径 / RobustEKF robust_ekf_core.py vio_quality_zero
            # 拒绝路径 / FGO fgo_core.py vio_quality_zero 拒绝路径 同口径跳过更新并重置参考位姿，无补增量。
            if _is_timestamp_in_outage(
                coerce_finite_scalar(row["timestamp"], name="row.timestamp"),
                noise.vio_outage_segments,
            ):
                rows.append(
                    {
                        "timestamp": _round_normalize(coerce_finite_scalar(row["timestamp"], name="row.timestamp")),
                        "dx": 0.0,
                        "dy": 0.0,
                        "dyaw": 0.0,
                        "quality": 0.0,
                    }
                )
                previous_timestamp = coerce_finite_scalar(row["timestamp"], name="row.timestamp")
                previous_cycle_index = cycle_index
                continue
            dx, dy, dyaw = _body_frame_delta(prev_pose, curr_pose)
            residual_dx, residual_dy, residual_dyaw = base_vio_residuals[_require_index_in_bounds(row["_base_row_index"], base_vio_residuals, name="base_vio_residuals")]  # 组5
            # 尺度漂移因子：scale = 1 + drift_rate * direction * elapsed_t
            # drift_rate * direction 确定漂移的方向和速率，elapsed_t 是从序列开始累积的时间
            # §6.2 #7 尺度漂移子机制对应代码层落点：
            # 单目或尺度不稳导致增量尺度慢漂，由 vio_scale_drift_rate (1/s) 乘性作用于
            # dx_scaled/dy_scaled 模拟纯 VO 累积尺度误差。与 §6.2 #8 偏置型增量误差
            # （drift_bias_sigma_mps 按时间加性累积 N(0, σ·√dt)）实现分离：
            # #7 在 sim_materializer 本处乘性 scale，#8 在 visual_levels.apply_visual_level
            # 加性 Wiener 累积，二者通过 vio_payload 传导且全员同一叙事（§6.2 表格全员同实现）。
            current_t = coerce_finite_scalar(row["timestamp"], name="row.timestamp")
            elapsed_t = max(0.0, current_t - seq_start_t)
            scale_factor = 1.0 + noise.vio_scale_drift_rate * scale_drift_direction * elapsed_t
            # 尺度因子作用于几何增量（dx/dy），模拟 VIO 对距离的估计偏差
            dx_scaled = dx * scale_factor
            dy_scaled = dy * scale_factor
            # 组2: 先计算合计并校验有限，再 round，避免 NaN/Inf 进入输出
            # §6.2 #5 退化段子机制对应代码层落点：
            # V2/V3 yaml 档 reproj_err_max=2/4 + tracked_features_range=[30,80]/[0,40] 联动
            # visual_levels.py _degrade_tracked_features/_degrade_reproj_err/_degrade_quality
            # 退化函数与本处 vio_dx_std/dy_std/dyaw_std 高斯噪声叠加，共同体现"弱纹理、运动
            # 模糊、低特征数噪声大"的退化语义。三估计器 _handle_vio quality_floor/mahalanobis_sq
            # 拒绝路径同口径处理低质量 VIO。
            # §6.2 #9 突发错误关联子机制对应代码层落点：
            # V2/V3 reproj_err_max=2/4 + blackout_prob=0.08/0.20 偶发触发 fixture 残差（L2320）
            # 大跳变 + visual_levels blackout 整事件移除，测三估计器门控与信度。EKF/RobustEKF/FGO
            # 三估计器拒绝路径均跳过更新并重置参考位姿（§6.4 #20 不伪造增量合取）。
            dx_noise = rng_vio_dx.gauss(0.0, noise.vio_dx_std) if rng_vio_dx is not None else 0.0
            dy_noise = rng_vio_dy.gauss(0.0, noise.vio_dy_std) if rng_vio_dy is not None else 0.0
            dyaw_noise = rng_vio_dyaw.gauss(0.0, noise.vio_dyaw_std) if rng_vio_dyaw is not None else 0.0
            dx_total = dx_scaled + residual_dx + dx_noise
            dy_total = dy_scaled + residual_dy + dy_noise
            dyaw_total = dyaw + residual_dyaw + dyaw_noise
            if not (math.isfinite(dx_total) and math.isfinite(dy_total) and math.isfinite(dyaw_total)):
                raise ValueError(f"vio delta totals must be finite, got dx={dx_total!r}, dy={dy_total!r}, dyaw={dyaw_total!r}")
            rows.append(
                {
                    "timestamp": _round_normalize(coerce_finite_scalar(row["timestamp"], name="row.timestamp")),
                    # VIO 增量 = 几何增量 × 尺度因子 + fixture 残差 + 高斯噪声
                    "dx": _round_normalize(dx_total),
                    "dy": _round_normalize(dy_total),
                    # dyaw 先 wrap 再 round 再 wrap，避免 round 后落到 π 边界外
                    "dyaw": wrap_angle_rad(_round_normalize(wrap_angle_rad(dyaw_total))),
                    "quality": coerce_finite_scalar(row.get("quality", 1.0), name="row.quality"),
                    # 铁律 3: 删除 tracked_features / reproj_err 字段
                    # VIO 仅输出 dx/dy/dyaw/quality/valid (此模块无 valid 字段)
                    # TODO(STAGE-B/D): 下游 feature_builder / event_builder /
                    # field_mapper / model_factory / ntu_viral_reader 仍依赖这两字段
                    # (grep "tracked_features|reproj_err" src/), 由其他子代理统一去除.
                }
            )
        previous_timestamp = coerce_finite_scalar(row["timestamp"], name="row.timestamp")
        previous_cycle_index = cycle_index
    return rows


def _make_deterministic_rng(base_seed: int, *parts: object) -> random.Random:
    """基于 SHA-256 混合 base_seed 与附加标识符构造确定性 random.Random。

    用于 FFN 多尺度 Wiener 叠加需要多个确定性高斯样本的场景，
    相比 _seeded_gaussian 单次采样，本函数返回的 RNG 可连续采样。

    参数：
        base_seed: 种子基数值。
        *parts: 附加标识符，用于区分不同流/字段/序列。

    返回：
        已播种的 random.Random 实例。
    """
    digest = hashlib.sha256(
        "|".join([str(int(base_seed)), *[str(part) for part in parts]]).encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big", signed=False))


def _generate_ffn_1_over_f(n_samples: int, sigma: float, rng: random.Random) -> list[float]:
    """生成 1/f flicker noise (FFN) 序列，使用多尺度 Wiener 叠加。

    参考 MATLAB imuSensor FFN 实现与 IEEE Std 952-2020 Allan 方差模型。
    通过 K=log2(n) 个尺度的 Wiener 过程叠加，近似 1/f PSD，
    对应 Allan 方差 T⁰ 平台区（bias instability）。

    与 scenarios/async_levels.py 中的 _generate_ffn_1_over_f 实现等价，
    本地副本避免 dataio 层依赖 scenarios 层。

    参数：
        n_samples: 样本数。
        sigma: 目标标准差（bias instability 幅度）。
        rng: 确定性随机数生成器。

    返回：
        长度为 n_samples 的 1/f 噪声序列（浮点数列表）。
    """
    if n_samples <= 0:
        return []
    if sigma <= 0.0:
        return [0.0] * n_samples
    # 多尺度 Wiener 叠加：K 个尺度的 Wiener 过程，振幅按 2^(-k/2) 衰减以近似 1/f PSD。
    K = max(1, min(_FFN_MAX_SCALES, int(math.log2(max(n_samples, 2)))))
    # 用 numpy 批量生成和累加，避免 Python 循环开销。
    try:
        import numpy as np
        samples = np.zeros(n_samples, dtype=np.float64)
        for k in range(K):
            scale = sigma * (2.0 ** (-k / 2.0))
            stride = 2 ** k
            n_points = (n_samples + stride - 1) // stride
            # 用 Python random 生成增量保证确定性（与原实现种子一致），再转 numpy。
            increments = np.array([rng.gauss(0.0, 1.0) for _ in range(n_points)], dtype=np.float64)
            # 上采样：每个增量重复 stride 次。
            expanded = np.repeat(increments, stride)[:n_samples] * scale
            samples += expanded
        # 归一化到目标 sigma。
        mean = float(samples.mean())
        std = float(samples.std())
        if std > 0.0:
            # 先去均值再缩放，保证 FFN 序列零均值（与 IEEE 952 FFN 零均值定义一致）。
            # 此前版本计算了 mean 但未使用，导致序列带非零固定偏移。
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


def _materialize_imu_rows(
    tiled_imu_rows: list[dict[str, Any]],
    *,
    transformed_gt_rows: list[dict[str, float]],
    base_imu_residuals: list[tuple[float, float, float]],
    noise_spec: SimNoiseSpec | dict[str, Any] | None = None,
    seq_id: str = "inline_seq",
) -> list[dict[str, Any]]:
    """物化铺叠后的 IMU 行：几何推导值 + fixture 残差 + bias(FFN+RRW) + 高斯噪声。

    IMU 按 cycle 分组处理：每个 cycle 内独立从 GT 推导加速度和角速度，
    再叠加 fixture 残差、bias 项（FFN 时变偏置 + RRW 时变随机游走）和高斯测量噪声。

    bias 模型（参考 MATLAB imuSensor + Allan 方差标准 IEEE Std 952-2020）：
        - FFN (flicker frequency noise, BiasInstability)：1/f 噪声序列，
          通过多尺度 Wiener 叠加生成，对应 Allan 方差 T⁰ 平台区。
          序列内逐样本变化（非恒定偏置），覆盖 Allan 方差中频段。
        - RRW (random walk frequency noise, Rate Random Walk)：bias 按时间累积
          Wiener 过程：bias(t+dt) = bias(t) + N(0, rrw · √dt)，
          方差随时间线性增长（σ²(t) = Q² · t），覆盖 Allan 方差低频段。

    参数：
        tiled_imu_rows: 铺叠后的 IMU 行列表（含辅助字段）。
        transformed_gt_rows: 变换后的 GT 行列表。
        base_imu_residuals: fixture 中提取的 IMU 残差列表。
        noise_spec: 噪声规格。
        seq_id: 序列编号，用于噪声种子区分。

    返回：
        物化后的 IMU 行列表，包含 timestamp、ax、ay、gz。

    异常：
        ValueError: 输入为空或观测合计非有限时抛出。
        IndexError: _base_row_index 越界时抛出。
    """
    noise = _normalize_noise_spec(noise_spec)
    # 第 4 轮审查 MEDIUM-2 修复：空输入校验对称性。与 _materialize_vio_rows
    # 和 _materialize_uwb_rows 保持一致——若 tiled_imu_rows 为空，直接返回
    # 空列表（不抛错）；非空时校验 base_imu_residuals 非空。
    if not tiled_imu_rows:
        return []
    _require_non_empty(base_imu_residuals, name="base_imu_residuals")  # 组4
    # D-7 patch: SimNoiseSpec 缺失字段显式 logger.warning 审计.
    # 铺叠后的 tiled_imu_rows 保留原始 fixture row 的 key 集合,
    # 用第一行作为 fixture 字段清单的采样代表 (铺叠只复制不增删 fixture key).
    # IMU noise group (imu_ax_std/imu_ay_std/imu_gz_std/imu_*_bias_*/imu_*_rrw_*)
    # 对应 fixture 观测字段 ax/ay/gz/timestamp, 缺失则改行被静默跳过或残差取 0.
    _imu_fixture_audit_fields = ("timestamp", "ax", "ay", "gz")
    _first_imu_row = tiled_imu_rows[0]
    _imu_missing = [k for k in _imu_fixture_audit_fields if k not in _first_imu_row]
    if _imu_missing:
        SimNoiseSpec.warn_missing_fixture_fields(
            _imu_missing,
            sensor_kind="imu",
            seq_id=seq_id,
        )
    # FFN (BiasInstability)：1/f flicker noise 序列，对应 Allan 方差 T⁰ 平台区。
    # 通过多尺度 Wiener 叠加生成，逐样本变化（非恒定偏置）。
    # FFN 序列长度等于铺叠后 IMU 总样本数，按 (cycle_index, base_row_index) 顺序索引。
    n_total = len(tiled_imu_rows)
    if noise.imu_accel_bias_instability > 0.0:
        rng_ffn_ax = _make_deterministic_rng(noise.base_seed, seq_id, "imu", "ffn_ax")
        rng_ffn_ay = _make_deterministic_rng(noise.base_seed, seq_id, "imu", "ffn_ay")
        ffn_accel_ax = _generate_ffn_1_over_f(n_total, noise.imu_accel_bias_instability, rng_ffn_ax)
        ffn_accel_ay = _generate_ffn_1_over_f(n_total, noise.imu_accel_bias_instability, rng_ffn_ay)
    else:
        ffn_accel_ax = [0.0] * n_total
        ffn_accel_ay = [0.0] * n_total
    if noise.imu_gyro_bias_instability > 0.0:
        rng_ffn_gz = _make_deterministic_rng(noise.base_seed, seq_id, "imu", "ffn_gz")
        ffn_gyro_gz = _generate_ffn_1_over_f(n_total, noise.imu_gyro_bias_instability, rng_ffn_gz)
    else:
        ffn_gyro_gz = [0.0] * n_total
    # 按 cycle 分组，每组独立推导 IMU 观测
    rows_by_cycle: dict[int, list[dict[str, Any]]] = {}
    for row in tiled_imu_rows:
        rows_by_cycle.setdefault(int(row["_cycle_index"]), []).append(row)
    materialized = []
    # RRW (Rate Random Walk)：bias 按时间累积 Wiener 过程。
    # bias(t+dt) = bias(t) + N(0, rrw · √dt)，rrw=0 时退化为零。
    rrw_accel_ax = 0.0
    rrw_accel_ay = 0.0
    rrw_gyro_gz = 0.0
    last_t: float | None = None  # 上一个 IMU 时间戳，用于计算 dt
    ffn_index = 0  # FFN 序列全局索引，按 cycle_index 升序、base_row_index 升序推进
    # 预生成确定性 RNG 用于 RRW 和高斯测量噪声，避免逐样本 SHA-256 开销。
    rng_rrw_ax = _make_deterministic_rng(noise.base_seed, seq_id, "imu", "rrw_ax_stream")
    rng_rrw_ay = _make_deterministic_rng(noise.base_seed, seq_id, "imu", "rrw_ay_stream")
    rng_rrw_gz = _make_deterministic_rng(noise.base_seed, seq_id, "imu", "rrw_gz_stream")
    rng_noise_ax = _make_deterministic_rng(noise.base_seed, seq_id, "imu", "noise_ax_stream")
    rng_noise_ay = _make_deterministic_rng(noise.base_seed, seq_id, "imu", "noise_ay_stream")
    rng_noise_gz = _make_deterministic_rng(noise.base_seed, seq_id, "imu", "noise_gz_stream")
    # 预标准化 GT 行并构建时间戳索引，避免每个 cycle 重复 O(N log N) 标准化。
    # transformed_gt_rows 已是 _transform_gt_rows 的输出（字段为 float、按 timestamp 升序），
    # 但 _derive_imu_rows_from_gt 内部会再次调用 _normalize_gt_rows，此处预标准化一次
    # 并走 _derive_imu_rows_from_normalized_gt 快路径，省去 132 次重复排序。
    norm_gt_rows = _normalize_gt_rows(transformed_gt_rows)
    norm_gt_timestamp_index = _build_gt_timestamp_index(norm_gt_rows)
    for cycle_index in sorted(rows_by_cycle):
        cycle_rows = rows_by_cycle[cycle_index]
        # 从变换后的 GT 推导当前 cycle 各时间戳的 IMU 观测
        derived_rows = _derive_imu_rows_from_normalized_gt(
            norm_gt_rows,
            [coerce_finite_scalar(row["timestamp"], name="row.timestamp") for row in cycle_rows],
            gt_timestamp_index=norm_gt_timestamp_index,
        )
        for tiled_row, derived_row in zip(cycle_rows, derived_rows, strict=True):
            # 从 fixture 残差中按 _base_row_index 取对应行的残差
            residual_ax, residual_ay, residual_gz = base_imu_residuals[_require_index_in_bounds(tiled_row["_base_row_index"], base_imu_residuals, name="base_imu_residuals")]  # 组5
            current_t = coerce_finite_scalar(tiled_row["timestamp"], name="tiled_row.timestamp")
            # RRW 步进：rrw_bias += N(0, rrw · √dt)，dt 为相对上一个 IMU 时刻的时间差。
            # 第一个样本无 dt，不更新（RRW 初始为 0）。
            # 第 11 轮审查 MEDIUM-1 修复（R11-B M1）：原 dt = max(0.0, current_t - last_t)
            # 对负 dt 静默归零，与同文件 _safe_diff_divide（L854-869 dt<=0 抛 ValueError）
            # 策略不一致，可能掩盖时间戳退化。改为：负 dt 抛 ValueError（与 _safe_diff_divide
            # 对齐），零 dt 物理正确跳过（无时间 elapsed，无 bias 更新）。
            if last_t is not None:
                dt = current_t - last_t
                if dt < 0.0:
                    raise ValueError(
                        f"IMU timestamp went backwards: current_t={current_t!r}, last_t={last_t!r}"
                    )
                if dt > 0.0:
                    sqrt_dt = math.sqrt(dt)
                    if noise.imu_accel_rrw_mps2_per_sqrt_s > 0.0:
                        rrw_accel_ax += rng_rrw_ax.gauss(0.0, noise.imu_accel_rrw_mps2_per_sqrt_s * sqrt_dt)
                        rrw_accel_ay += rng_rrw_ay.gauss(0.0, noise.imu_accel_rrw_mps2_per_sqrt_s * sqrt_dt)
                    if noise.imu_gyro_rrw_rads_per_sqrt_s > 0.0:
                        rrw_gyro_gz += rng_rrw_gz.gauss(0.0, noise.imu_gyro_rrw_rads_per_sqrt_s * sqrt_dt)
            last_t = current_t
            # 组2: 先计算合计并校验有限，再 round，避免 NaN/Inf 进入输出
            # IMU 观测 = 几何推导值 + fixture 残差 + FFN(1/f) + RRW(Wiener) + 高斯测量噪声
            accel_bias_ax = ffn_accel_ax[ffn_index] + rrw_accel_ax
            accel_bias_ay = ffn_accel_ay[ffn_index] + rrw_accel_ay
            gyro_bias_gz = ffn_gyro_gz[ffn_index] + rrw_gyro_gz
            ffn_index += 1
            ax_total = float(derived_row["ax"]) + residual_ax + accel_bias_ax + rng_noise_ax.gauss(0.0, noise.imu_ax_std)
            ay_total = float(derived_row["ay"]) + residual_ay + accel_bias_ay + rng_noise_ay.gauss(0.0, noise.imu_ay_std)
            gz_total = float(derived_row["gz"]) + residual_gz + gyro_bias_gz + rng_noise_gz.gauss(0.0, noise.imu_gz_std)
            if not (math.isfinite(ax_total) and math.isfinite(ay_total) and math.isfinite(gz_total)):
                raise ValueError(f"imu observation totals must be finite, got ax={ax_total!r}, ay={ay_total!r}, gz={gz_total!r}")
            materialized.append(
                {
                    "timestamp": _round_normalize(current_t),
                    # IMU 观测 = 几何推导值 + fixture 残差 + FFN(1/f) + RRW(Wiener) + 高斯测量噪声
                    "ax": _round_normalize(ax_total),
                    "ay": _round_normalize(ay_total),
                    "gz": _round_normalize(gz_total),
                }
            )
    return materialized


def _sequence_dir_entries(seq_root: Path) -> tuple[str, ...]:
    """列出序列目录下的所有文件名（排序后）。

    参数：
        seq_root: 序列目录路径。

    返回：
        排序后的文件名元组。
    """
    return tuple(sorted(entry.name for entry in seq_root.iterdir() if entry.is_file()))


def _write_json(path: Path, payload: Any) -> None:
    """将 Python 对象序列化为 JSON 并写入文件。

    注意：本函数为非原子写入（直接覆盖目标文件，无临时文件 + rename）。
    调用方（如 materialize_sim_raw）需自行负责失败回滚清理。

    参数：
        path: 输出文件路径。
        payload: 要序列化的 Python 对象。
    """
    write_json(path, payload)


def can_materialize_sim_raw(output_root: str | Path) -> bool:
    """检查目标目录是否可以用于物化仿真原始数据。

    只有当目录不存在或为空（不含可见文件）时才允许物化，避免覆盖已有数据。

    注意：本函数与 materialize_sim_raw 之间存在 TOCTOU 窗口（检查与使用
    非原子），非线程安全。并发调用方需自行加锁。

    参数：
        output_root: 目标输出根目录路径。

    返回：
        True 表示可以物化，False 表示目录已存在且非空。
    """
    # 第 10 轮审查 MEDIUM-1 修复：can_materialize_sim_raw 是物化前置检查入口，
    # 物化流程会对其逐序列调用。入口 print_dict 违反工程规范
    # "Recursive functions and entry points should avoid print_dict calls"，
    # 且仅打印 output_root 信息价值低，物化报告已包含完整路径明细。
    path = Path(output_root)
    if not path.exists():
        return True  # 目录不存在，可以创建
    if not path.is_dir():
        return False  # 路径存在但不是目录，不能写入
    visible_entries = [entry for entry in path.iterdir() if not entry.name.startswith(".")]
    return not visible_entries  # 只有空目录才允许


def materialize_sim_raw(
    output_root: str | Path,
    *,
    fixture_root: str | Path | None = None,
    sequence_specs: tuple[SimSequenceSpec, ...] = DEFAULT_SIM_SEQUENCE_SPECS,
    noise_spec: SimNoiseSpec | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """物化仿真原始数据集：从 fixture 种子生成完整的仿真数据目录。

    对每条序列规格，执行以下步骤：
    1. 加载 fixture 种子数据
    2. 铺叠 GT 和传感器流
    3. 对 GT 施加几何变换
    4. 从 fixture 提取传感器残差
    5. 物化 IMU/UWB/VIO 流（几何推导 + 残差 + 噪声）
    6. 写入输出目录

    参数：
        output_root: 输出根目录，必须为空或不存在。
        fixture_root: 可选的 fixture 根目录路径。
        sequence_specs: 序列规格元组，默认使用 DEFAULT_SIM_SEQUENCE_SPECS。
        noise_spec: 噪声规格，默认使用 DEFAULT_SIM_NOISE_SPEC。

    返回：
        物化报告字典，包含每条序列的状态、计数和变换参数。

    异常：
        RuntimeError: 输出目录非空或已被并发创建时抛出。
        ValueError: sequence_specs 为空或 seq_id 重复时抛出。
    """
    # 第 10 轮审查 MEDIUM-1 修复：materialize_sim_raw 是物化流程入口函数，
    # 每实验调用一次。入口 print_dict 违反工程规范
    # "Recursive functions and entry points should avoid print_dict calls"，
    # 物化报告字典（返回值）已包含全部序列规格、噪声规格和路径明细。
    _require_non_empty(sequence_specs, name="sequence_specs")  # 组4
    # 组4: seq_id 唯一性校验，避免覆盖
    seen_seq_ids: set[str] = set()
    for spec in sequence_specs:
        # P0-7: seq_id 用于路径拼装（output_root_path / spec.seq_id），必须校验
        # 不含路径穿越字符（../、绝对路径、null 字节），防止逃逸 output_root。
        # 注意：SimSequenceSpec.__post_init__ 已对 seq_id 做过 validate_path_component，
        # 此处为防御性二次校验，覆盖未来可能新增的 sequence_specs 来源。
        validate_path_component(spec.seq_id, name="spec.seq_id")
        if spec.seq_id in seen_seq_ids:
            raise ValueError(f"duplicate seq_id in sequence_specs: {spec.seq_id!r}")
        seen_seq_ids.add(spec.seq_id)
    output_root_path = Path(output_root).resolve()
    fixture_root_path = _fixture_root(fixture_root)
    if not can_materialize_sim_raw(output_root_path):
        raise RuntimeError("sim raw target must be empty before materialization; paper-run should only auto-generate into an empty root")
    # 组9: can_materialize_sim_raw 已确认目录不存在或为空（不含可见文件）。
    # exist_ok=True 容忍"目录已存在但为空"的合法情况（与 docstring "必须为空或不存在" 一致），
    # 同时保留对 TOCTOU 窗口的检测——若并发写入了可见文件，后续写文件操作会自然暴露冲突。
    output_root_path.mkdir(parents=True, exist_ok=True)
    noise = _normalize_noise_spec(noise_spec)
    sequence_reports: dict[str, Any] = {}
    sequence_ids: list[str] = []
    for spec in sequence_specs:
        # 加载 fixture 种子数据
        bundle = _load_fixture_bundle(spec.base_seq_id, fixture_root_path)
        base_gt_rows = _normalize_gt_rows(bundle["gt"])
        base_start = coerce_finite_scalar(base_gt_rows[0]["timestamp"], name="base_gt_rows[0].timestamp")
        base_end = coerce_finite_scalar(base_gt_rows[-1]["timestamp"], name="base_gt_rows[-1].timestamp")
        stride_s = base_end - base_start
        # P0-8: stride_s 必须为正，否则 cycle_stride = stride_s + gap_s 会退化为仅 gap_s，
        # 导致多 cycle 铺叠时间戳折叠（fixture 单 cycle 时长为 0 是退化输入）。
        if stride_s <= 0.0:
            raise ValueError(
                f"fixture stride_s must be > 0 (base_end - base_start), got {stride_s}; "
                f"seq_id={spec.seq_id!r}, base_seq_id={spec.base_seq_id!r}"
            )

        if spec.use_protocol_trajectory:
            # §8.2 协议轨迹：不再依赖 mini-fixture 铺叠。GT 直接满足 B20–B23 包络。
            transformed_gt_rows = generate_protocol_gt_rows(
                seq_id=spec.seq_id,
                seed=int(noise.base_seed),
                duration_s=float(spec.duration_s),
                workspace_span_m=float(spec.workspace_span_m),
                dt_s=float(spec.dt_imu_override or 0.05),
            )
            transformed_anchor_layout = _transform_anchor_layout(bundle["anchor_layout"], spec)
            dt_imu = float(spec.dt_imu_override or 0.01)
            dt_uwb = float(spec.dt_uwb_override or 0.1)
            dt_vio = float(spec.dt_vio_override or 0.05)
            t0 = float(transformed_gt_rows[0]["timestamp"])
            t1 = float(transformed_gt_rows[-1]["timestamp"])
            # 合成传感器时间网格（无 fixture residual；几何由 GT 推导）。
            def _grid(dt: float) -> list[dict[str, Any]]:
                n = max(2, int(math.floor((t1 - t0) / dt)) + 1)
                rows_local: list[dict[str, Any]] = []
                for i in range(n):
                    rows_local.append(
                        {
                            "timestamp": t0 + i * dt,
                            "_cycle_index": 0,
                            "_base_row_index": i,
                            "anchor_id": "A0",
                            "range": 0.0,
                            "valid": True,
                            "quality": 1.0,
                            "dx": 0.0,
                            "dy": 0.0,
                            "dyaw": 0.0,
                            "ax": 0.0,
                            "ay": 0.0,
                            "gz": 0.0,
                        }
                    )
                return rows_local

            tiled_imu_rows = _grid(dt_imu)
            tiled_uwb_rows = _grid(dt_uwb)
            # UWB 按锚点轮询：同一时间网格上循环 anchor_id
            anchor_ids = list(transformed_anchor_layout.get("anchor_ids") or ["A0"])
            for i, row in enumerate(tiled_uwb_rows):
                row["anchor_id"] = anchor_ids[i % len(anchor_ids)]
            tiled_vio_rows = _grid(dt_vio)
            zero_imu_residuals = [(0.0, 0.0, 0.0)] * max(1, len(tiled_imu_rows))
            zero_uwb_residuals = [0.0] * max(1, len(tiled_uwb_rows))
            zero_vio_residuals = [(0.0, 0.0, 0.0)] * max(1, len(tiled_vio_rows))
            imu_rows = _materialize_imu_rows(
                tiled_imu_rows,
                transformed_gt_rows=transformed_gt_rows,
                base_imu_residuals=zero_imu_residuals,
                noise_spec=noise,
                seq_id=spec.seq_id,
            )
            uwb_rows = _materialize_uwb_rows(
                tiled_uwb_rows,
                transformed_gt_rows=transformed_gt_rows,
                transformed_anchor_layout=transformed_anchor_layout,
                base_uwb_residuals=zero_uwb_residuals,
                noise_spec=noise,
                seq_id=spec.seq_id,
            )
            vio_rows = _materialize_vio_rows(
                tiled_vio_rows,
                transformed_gt_rows=transformed_gt_rows,
                base_vio_residuals=zero_vio_residuals,
                noise_spec=noise,
                seq_id=spec.seq_id,
            )
            # 主表硬门禁：B20–B23 / B30
            envelope_report = assert_geometry_motion_envelope(
                transformed_gt_rows,
                transformed_anchor_layout,
                profile=str(spec.envelope_profile),
                allow_high_anchor_count=bool(spec.allow_high_anchor_count),
            )
        else:
            # 兼容旧 fixture 铺叠路径（smoke / 回归）。
            tiled_gt_rows = _tile_base_gt_rows(
                base_gt_rows,
                cycle_count=spec.cycle_count,
                cycle_gap_s=spec.cycle_gap_s,
                base_seed=int(noise.base_seed),
                seq_id=spec.seq_id,
                jitter_sigma_s=0.0,
                randomize_cycle_count=True,
            )
            transformed_gt_rows = _transform_gt_rows(tiled_gt_rows, spec)
            transformed_anchor_layout = _transform_anchor_layout(bundle["anchor_layout"], spec)
            tiled_imu_rows = _tile_base_stream_rows(
                bundle["imu"],
                cycle_count=spec.cycle_count,
                cycle_gap_s=spec.cycle_gap_s,
                stride_s=stride_s,
                gt_base_start=base_start,
                dt_override=spec.dt_imu_override,
                jitter_sigma_s=0.001,
                base_seed=int(noise.base_seed),
                seq_id=spec.seq_id,
                modality="imu",
            )
            tiled_uwb_rows = _tile_base_stream_rows(
                bundle["uwb"],
                cycle_count=spec.cycle_count,
                cycle_gap_s=spec.cycle_gap_s,
                stride_s=stride_s,
                gt_base_start=base_start,
                dt_override=spec.dt_uwb_override,
                jitter_sigma_s=0.005,
                base_seed=int(noise.base_seed),
                seq_id=spec.seq_id,
                modality="uwb",
            )
            tiled_vio_rows = _tile_base_stream_rows(
                bundle["vio"],
                cycle_count=spec.cycle_count,
                cycle_gap_s=spec.cycle_gap_s,
                stride_s=stride_s,
                gt_base_start=base_start,
                dt_override=spec.dt_vio_override,
                jitter_sigma_s=0.002,
                base_seed=int(noise.base_seed),
                seq_id=spec.seq_id,
                modality="vio",
            )
            base_imu_residuals = _derive_base_imu_residuals(bundle["imu"], base_gt_rows)
            if spec.mirror_x:
                base_imu_residuals = [(ax, -ay, -gz) for ax, ay, gz in base_imu_residuals]
            imu_rows = _materialize_imu_rows(
                tiled_imu_rows,
                transformed_gt_rows=transformed_gt_rows,
                base_imu_residuals=base_imu_residuals,
                noise_spec=noise,
                seq_id=spec.seq_id,
            )
            uwb_rows = _materialize_uwb_rows(
                tiled_uwb_rows,
                transformed_gt_rows=transformed_gt_rows,
                transformed_anchor_layout=transformed_anchor_layout,
                base_uwb_residuals=_derive_base_uwb_residuals(bundle["uwb"], base_gt_rows, bundle["anchor_layout"]),
                noise_spec=noise,
                seq_id=spec.seq_id,
            )
            base_vio_residuals = _derive_base_vio_residuals(bundle["vio"], base_gt_rows)
            if spec.mirror_x:
                base_vio_residuals = [(dx, -dy, -dyaw) for dx, dy, dyaw in base_vio_residuals]
            vio_rows = _materialize_vio_rows(
                tiled_vio_rows,
                transformed_gt_rows=transformed_gt_rows,
                base_vio_residuals=base_vio_residuals,
                noise_spec=noise,
                seq_id=spec.seq_id,
            )
            # §8.2.1 硬门禁：原实现写 fake report={passed: None, "skips §8.2.1 hard gate"}
            # 把整段硬门 silent-skip——一旦 SimSequenceSpec 设 use_protocol_trajectory=False
            # 但 envelope_profile="main_table"，§8.2.1 全部硬门（B20–B23 / B30 / z_extent /
            # periodicity）会被吞掉，违背 spec L1372-L1410 主表硬门禁主张。
            # 修复：legacy path 也调用 assert_geometry_motion_envelope，按 spec.envelope_profile
            # 走对应硬门（main_table 全门 / smoke 仅 3 项极弱门且 fail 也 raise）。
            # legacy fixture 路径虽非默认主表生产路径，但 spec 已允许该组合存在，
            # 门禁必须对其仍然生效，不能写成不带任何检查的 fake dict。
            envelope_report = assert_geometry_motion_envelope(
                transformed_gt_rows,
                transformed_anchor_layout,
                profile=str(spec.envelope_profile),
                allow_high_anchor_count=bool(spec.allow_high_anchor_count),
            )
            envelope_report = dict(envelope_report)  # 不污染原报告
            envelope_report["trajectory_path"] = "legacy_fixture_tile"
            if spec.envelope_profile == "smoke":
                envelope_report["note"] = (
                    "legacy path with smoke profile; §8.2.1 main_table hard gates intentionally not enforced"
                )

        # 写入输出目录
        seq_root = output_root_path / spec.seq_id
        seq_root.mkdir(parents=True, exist_ok=False)  # 不允许覆盖已有序列目录
        # v2 D-10/D-11/D-12/D-13: 写 sim_meta.json 标注场景轴档位 + 传感器频率覆盖.
        # 下游 prepare_manifest 阶段读 sim_meta.json 后, 把 axes_override 注入 manifest 的
        # scene_parameters.flat.{A,N,V,G,K}_level 字段, 让 30 seqs 跨多场景轴档位.
        # 铁律 5b: dt_imu_override / dt_uwb_override / dt_vio_override 现已基础物化路径消费
        # (在 _tile_base_stream_rows 内 base_row_index * dt 重采样), sim_meta 仍标注实际生效值.
        sim_meta = {
            "seq_id": spec.seq_id,
            "base_seq_id": spec.base_seq_id,
            "axes_override": dict(spec.axes_override) if spec.axes_override else {},
            "dt_imu_override_s": spec.dt_imu_override,
            "dt_uwb_override_s": spec.dt_uwb_override,
            "dt_vio_override_s": spec.dt_vio_override,
            "v2_protocol_version": 2,
            "v2_documentation": "13 维扩展 (D-10 场景轴多样性 + D-11 G 轴 + D-12 K 轴 + D-13 传感器频率)",
        }
        # 组10: 非原子写入回滚——若任一 _write_json 失败，清理已创建的 seq_root
        try:
            _write_json(seq_root / "imu.json", imu_rows)
            _write_json(seq_root / "uwb.json", uwb_rows)
            _write_json(seq_root / "vio.json", vio_rows)
            _write_json(seq_root / "gt.json", transformed_gt_rows)
            _write_json(seq_root / "anchor_layout.json", transformed_anchor_layout)
            _write_json(seq_root / "sim_meta.json", sim_meta)
        except Exception:
            # 写入失败时清理半成品目录，避免留下不完整契约
            shutil.rmtree(seq_root, ignore_errors=True)
            raise
        sequence_ids.append(spec.seq_id)
        sequence_reports[spec.seq_id] = {
            "seq_id": spec.seq_id,
            "base_seq_id": spec.base_seq_id,
            "seq_root": str(seq_root),
            "status": "materialized",
            "cycle_count": int(spec.cycle_count),
            "use_protocol_trajectory": bool(spec.use_protocol_trajectory),
            "duration_s": float(spec.duration_s),
            "workspace_span_m": float(spec.workspace_span_m),
            "counts": {
                "imu": len(imu_rows),
                "uwb": len(uwb_rows),
                "vio": len(vio_rows),
                "gt": len(transformed_gt_rows),
            },
            "transform": {
                "translation_xy": [float(spec.translation_xy[0]), float(spec.translation_xy[1])],
                "rotation_rad": float(spec.rotation_rad),
                "mirror_x": bool(spec.mirror_x),
                "curve_strength": float(spec.curve_strength),
            },
            "v2_axes_override": dict(spec.axes_override) if spec.axes_override else {},
            "v2_dt_overrides": {
                "imu_s": spec.dt_imu_override,
                "uwb_s": spec.dt_uwb_override,
                "vio_s": spec.dt_vio_override,
            },
            "geometry_motion_envelope": envelope_report,
            "contract_files": list(_RAW_CONTRACT_FILES) + ["sim_meta.json"],
        }
    return {
        "status": "ok",
        "output_root": str(output_root_path),
        "fixture_root": str(fixture_root_path),
        "sequence_ids": sequence_ids,
        "required_files": list(_RAW_CONTRACT_FILES),
        "sequences": sequence_reports,
        "noise": asdict(noise),
    }


__all__ = [
    "DEFAULT_SIM_NOISE_SPEC",
    "DEFAULT_SIM_SEQUENCE_SPECS",
    "SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS",
    "SimNoiseSpec",
    "SimSequenceSpec",
    "ZERO_SIM_NOISE_SPEC",
    "can_materialize_sim_raw",
    "materialize_sim_raw",
]
