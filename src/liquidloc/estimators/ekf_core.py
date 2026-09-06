"""纯几何 EKF 基线。

这个模块把 IMU 预测、UWB 更新和 VIO 更新串成一个最小可用的状态估计核心。
它维护当前状态、协方差、锚点映射和最近一次更新报告，供上层 pipeline 反复调用。

模块内容
--------
- :func:`_count_measurement_noise_terms` —— 统计噪声标量总数
- :func:`_build_runtime_resource_meta` —— 估算资源占用
- :func:`_require_positive_measurement_noise_entry` —— 递归校验噪声正值
- :func:`_reject_boolean_like_measurement_noise_entry` —— 拒绝布尔噪声
- :class:`EKFCore` —— EKF 核心估计器
"""

from __future__ import annotations  # 允许类型标注使用前向引用，避免定义顺序限制。

import math  # 用来做平方根、有限性检查和 Huber 规则。

from collections.abc import Mapping, Sequence  # 用来判断配置是不是映射、序列。
from typing import Any  # 给报告和缓存字典保留灵活类型。

import copy  # 深拷贝，防止嵌套配置被外部修改污染内部状态。

import numpy as np  # 用来做矩阵和向量运算。

from liquidloc.common.covariance_utils import build_effective_cov  # 复用协方差缩放适配器。
from liquidloc.common.gt_utils import resolve_anchor_position  # 锚点位置解析的规范实现
from liquidloc.common.types import MeasurementControl, ModelIntermediate, StateEstimate  # 统一的测量控制和状态容器类型。
from liquidloc.common.constants import RAM_PEAK_FLOOR_MB, RAM_PEAK_PARAMS_PER_MB, VIO_REF_POSE_STALE_SECONDS  # VIO 参考位姿过时阈值；ram_peak 换算因子与下限（D9 漂移根因修复，与 model_factory 共用单源真相）。
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_integer, is_real, quality_below_floor  # 集中判断 bool / np.bool_ / 实数类型，以及有限浮点转换。
from liquidloc.estimators.predict_step import run_predict_step  # 复用 IMU 预测步骤。
from liquidloc.estimators.shared import build_controlled_measurement_cov, control_to_dict  # 共享的协方差控制和报告转换函数。
from liquidloc.estimators.state_definition import state_items  # 复用冻结的状态顺序。
from liquidloc.estimators.uwb_update_step import run_uwb_update, predict_range, build_uwb_jacobian  # 复用 UWB 更新步骤。
from liquidloc.estimators.vision_update_step import (  # 复用 VIO 更新链路。
    _ensure_positive_definite_vio_innovation_covariance,  # 创新协方差正定性校验。
    _normalize_vio_covariance,  # 视觉噪声规整成 3x3 协方差矩阵。
    apply_vision_update,  # 把视觉量测真正写回状态和协方差。
    build_vio_measurement,  # 从负载里提取标准化 VIO 量测向量。
    compute_vio_residual,  # 计算 VIO 预测量测、残差和雅可比。
)
from liquidloc.estimators.estimator_api import EstimatorAPI  # 估计器抽象基类，权威定义位于 estimators 层。
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS  # 桥接层硬阈值（quality_floor 回退默认值）。
from liquidloc.protocol.event_schema import validate_event  # 每个输入事件都要先过协议校验。
from liquidloc.protocol.task_contract import get_sensor_roles  # 冻结任务合同定义了当前估计器允许消费的模态。
from liquidloc.sensors.anchor_model import build_anchor_lookup  # 锚点布局要转成可查询映射。


def _count_measurement_noise_terms(measurement_noise: Any) -> int:
    """统计 measurement_noise 里一共有多少个噪声标量。

    参数
    ----------
    measurement_noise : Any
        待统计的噪声配置，支持 Mapping、ndarray、Number、list/tuple 等类型。

    返回
    -------
    int
        噪声标量的总数；布尔值不参与计数。

    注意
    -----
    - ``complex`` 是 ``Number`` 的子类但不是实数噪声项，遇到时按 0 计，
      不计入有效噪声数量。
    - 布尔值和布尔数组元素跳过不计。
    """
    if isinstance(measurement_noise, Mapping):  # 映射型配置要递归统计每个子项。
        return sum(_count_measurement_noise_terms(value) for value in measurement_noise.values())  # 递归求和。
    if is_bool_like(measurement_noise):  # 布尔值不算噪声项。
        return 0  # 布尔值不参与计数。
    if isinstance(measurement_noise, np.ndarray):  # 数组噪声按元素数量统计。
        kind = measurement_noise.dtype.kind  # 取 dtype 种类，统一判断。
        if kind == "b":  # 布尔数组不算。
            return 0  # 布尔数组元素不参与计数。
        if kind == "c":  # 复数数组不算实数噪声。
            return 0  # 复数数组不计入。
        if kind == "O":  # object 数组按元素递归，与 list/tuple 行为一致。
            if measurement_noise.ndim == 0:  # 0-d object 数组取 item 后递归。
                return _count_measurement_noise_terms(measurement_noise.item())  # 递归统计内部对象。
            return sum(_count_measurement_noise_terms(value) for value in measurement_noise.flat)  # 递归求和。
        if kind not in ("i", "u", "f"):  # 非数值类型（str/bytes/void/datetime 等）不计入。
            return 0  # 非数值数组元素不参与计数。
        if measurement_noise.ndim == 0:  # 标量数组算 1 个噪声项。
            return 1  # 0 维数组只有 1 个元素。
        return int(measurement_noise.size)  # 多维数组按元素总数统计。
    if isinstance(measurement_noise, np.complexfloating):  # numpy 复数标量不是实数噪声。
        return 0  # 复数标量不计入。
    if isinstance(measurement_noise, np.generic):  # numpy 标量按 1 个噪声项统计。
        return 1  # numpy 标量只算 1 个。
    if isinstance(measurement_noise, complex):  # Python 原生复数不是实数噪声。
        return 0  # complex 不计入。
    if is_real(measurement_noise):  # 普通数值标量。
        return 1  # 数值标量只算 1 个。
    if isinstance(measurement_noise, (list, tuple)):  # 列表和元组继续递归展开。
        return sum(_count_measurement_noise_terms(value) for value in measurement_noise)  # 递归求和。
    return 0  # 其他类型不参与噪声项计数。


def _build_runtime_resource_meta(cfg: dict[str, Any]) -> dict[str, float]:
    """根据配置粗略估算资源占用，供上层汇总展示。

    参数
    ----------
    cfg : dict[str, Any]
        初始化配置字典，可包含 ``measurement_noise`` 和 ``init_cov`` 等字段。

    返回
    -------
    dict[str, float]
        包含 ``params``（参数规模指标）、``ram_peak`` 和 ``ram_peak_mb``
        （峰值内存 MB）的资源摘要字典。
    """
    measurement_noise = cfg.get("measurement_noise")  # 取出测量噪声配置。
    if measurement_noise is None:  # 没配测量噪声时就按空配置处理。
        measurement_noise = {}  # 空映射兜底。
    measurement_noise_term_count = _count_measurement_noise_terms(measurement_noise)  # 统计噪声标量总数。
    init_cov = cfg.get("init_cov")  # 取出初始协方差配置。
    init_cov_length = len(init_cov) if isinstance(init_cov, (list, tuple)) and init_cov else len(state_items)  # 按列表/元组长度或状态维度估算。
    params = float(len(state_items) + measurement_noise_term_count + init_cov_length)  # 粗略参数规模指标。
    ram_peak_mb = max(RAM_PEAK_FLOOR_MB, params / RAM_PEAK_PARAMS_PER_MB)  # 用冻结换算因子估算峰值内存 MB，与 model_factory 保持同口径。
    return {
        "params": params,  # 用一个可读指标表示估计规模。
        "ram_peak": ram_peak_mb,  # 保留旧字段名，和上层兼容。
        "ram_peak_mb": ram_peak_mb,  # 统一按 MB 汇报。
    }


def _require_positive_measurement_noise_entry(value: Any, *, path: str) -> None:
    """递归校验 measurement_noise 的每个分量都为正。

    参数
    ----------
    value : Any
        待校验的噪声项，支持 Mapping、标量、一维向量或方阵。
    path : str
        错误信息中的配置路径，便于定位问题。

    异常
    ------
    TypeError
        当输入类型无法解析为合法噪声结构时抛出；
        ``complex`` 数值也被视为非法类型。
    ValueError
        当噪声值非正或非有限、向量含非正或非有限元素、
        或对角线含非正或非有限元素时抛出。
    """
    _reject_boolean_like_measurement_noise_entry(value, path=path)  # 先排除布尔型伪值。
    if isinstance(value, complex):  # Python 原生复数不能当实数噪声。
        raise TypeError(f"{path} must be real numeric, got complex")  # 直接拒绝复数。
    if isinstance(value, Mapping):  # 映射型要递归检查每个子项。
        if not value:  # 空映射说明没有噪声配置。
            raise ValueError(f"{path} must provide positive measurement noise values")  # 空映射不允许。
        for key, nested_value in value.items():  # 逐项递归检查。
            _require_positive_measurement_noise_entry(nested_value, path=f"{path}.{key}")  # 递归检查子项。
        return  # 映射检查完毕。

    value_array = np.asarray(value, dtype=float)  # 统一转成浮点数组。
    if value_array.ndim == 0:  # 标量噪声。
        scalar = float(value_array.reshape(-1)[0])  # 取出标量值。
        if not np.isfinite(scalar) or scalar <= 0.0:  # NaN/inf 会破坏滤波器，必须与非正数一起拒绝。
            raise ValueError(f"{path} must be positive")  # 非有限或非正直接报错。
        return  # 标量检查完毕。
    if value_array.ndim == 1:  # 一维向量噪声。
        if value_array.size == 0 or not np.all(np.isfinite(value_array)) or np.any(value_array <= 0.0):  # 空向量、含非有限或非正元素都不允许。
            raise ValueError(f"{path} must contain positive diagonal entries")  # 直接报错。
        return  # 向量检查完毕。
    if value_array.ndim == 2:  # 二维矩阵噪声。
        if value_array.shape[0] != value_array.shape[1] or value_array.shape[0] == 0:  # 必须是方阵且非空。
            raise ValueError(f"{path} must be a square covariance matrix")  # 非方阵直接报错。
        diag = np.diag(value_array)  # 取出对角线，避免重复调用。
        if not np.all(np.isfinite(diag)) or np.any(diag <= 0.0):  # 对角线必须有限且全为正。
            raise ValueError(f"{path} must contain positive diagonal entries")  # 非有限或非正对角线直接报错。
        return  # 矩阵检查完毕。
    raise TypeError(f"{path} must be a scalar, vector, or square covariance matrix")  # 其他维度不支持。


def _square_std_to_var(noise_spec: Any) -> Any:
    """将测量噪声配置从标准差转换为方差。

    配置文件中 measurement_noise 的语义是标准差（与 process_noise 一致），
    但底层更新函数（run_uwb_update / run_vio_update）期望接收方差。
    此函数递归地对所有数值节点做平方，保持字典/列表结构不变。

    参数
    ----------
    noise_spec : Any
        标准差形式的噪声配置，可以是标量、字典或列表。

    返回
    -------
    Any
        方差形式的噪声配置，结构不变。
    """
    if isinstance(noise_spec, Mapping):  # 字典递归处理每个值。
        return {k: _square_std_to_var(v) for k, v in noise_spec.items()}
    if isinstance(noise_spec, (list, tuple)):  # 列表/元组递归处理每个元素。
        return type(noise_spec)(_square_std_to_var(v) for v in noise_spec)
    if is_bool_like(noise_spec):  # 布尔值不是合法噪声。
        raise TypeError(f"noise_spec must be numeric, got bool: {noise_spec}")
    scalar = coerce_finite_scalar(noise_spec, name="noise std", min_value=0.0)  # 标准差必须非负有限。
    variance = scalar * scalar  # σ² → 方差。
    variance = coerce_finite_scalar(variance, name="noise variance")  # 标准差过大时平方可能溢出为 inf。
    return variance


def _reject_boolean_like_measurement_noise_entry(value: Any, *, path: str) -> None:
    """递归拒绝 measurement_noise 里所有布尔类输入。

    参数
    ----------
    value : Any
        待检查的噪声项，支持 Mapping、ndarray、list/tuple 等。
    path : str
        错误信息中的配置路径，便于定位问题。

    异常
    ------
    TypeError
        当发现布尔值（bool、np.bool_）或布尔数组时抛出。
    """
    if is_bool_like(value):  # 纯布尔值不能被当成噪声。
        raise TypeError(f"{path} must be numeric, got bool")  # 直接拒绝。
    if isinstance(value, Mapping):  # 映射型继续向下递归。
        for key, nested_value in value.items():  # 每个子项都要单独检查。
            _reject_boolean_like_measurement_noise_entry(nested_value, path=f"{path}.{key}")  # 继续深入路径。
        return  # 映射已经处理完毕。
    if isinstance(value, np.ndarray):  # 数组要检查 dtype 和对象元素。
        if value.dtype.kind == "b":  # 布尔数组不允许。
            raise TypeError(f"{path} must be numeric, got bool")  # 直接拒绝。
        if value.dtype == object:  # 对象数组要逐个元素拆开检查。
            for index, nested_value in np.ndenumerate(value):  # 逐个位置递归。
                index_suffix = "".join(f"[{item}]" for item in index)  # 把数组索引写进路径。
                _reject_boolean_like_measurement_noise_entry(nested_value, path=f"{path}{index_suffix}")  # 继续检查。
        return  # 数组已经处理完毕。
    if isinstance(value, (list, tuple)):  # 列表和元组也要逐项检查。
        for index, nested_value in enumerate(value):  # 每个元素都要递归。
            _reject_boolean_like_measurement_noise_entry(nested_value, path=f"{path}[{index}]")  # 带上元素索引。


def _validate_state_override_mapping(state_override: Any, *, path: str) -> dict[str, float]:
    """校验并规整初始状态覆盖项，防止拼写或类型错误被静默吞掉。"""
    if not isinstance(state_override, Mapping):  # 初始状态覆盖必须是映射。
        raise TypeError(f"{path} must be a mapping")
    normalized_override: dict[str, float] = {}
    for key, value in state_override.items():  # 逐项检查每个覆盖键和值。
        if key not in state_items:  # 未知状态键会让实验以错误初值运行。
            raise KeyError(f"{path} contains unknown state key: {key!r}")
        if is_bool_like(value) or not is_real(value):  # bool/复数等都不能作为状态值。
            raise TypeError(f"{path}.{key} must be numeric, got {type(value).__name__}")
        scalar_value = coerce_finite_scalar(value, name=f"{path}.{key}")  # NaN/Inf 会直接污染状态。
        normalized_override[key] = scalar_value
    return normalized_override


def _build_initial_covariance(init_cov: Any) -> np.ndarray:
    """构造初始对角协方差，禁止错误配置静默退回默认单位阵。"""
    if init_cov is None:  # 没配时允许使用默认单位阵。
        return np.eye(len(state_items), dtype=float)
    if isinstance(init_cov, (str, bytes, Mapping)):  # 字符串和映射不是合法的协方差序列，必须显式拒绝。
        raise TypeError(f"init_cov must be a sequence, got {type(init_cov).__name__}")
    raw_init_cov = np.asarray(init_cov, dtype=object)  # 先保留原始元素类型，逐项校验。
    expected_shape = (len(state_items),)
    if raw_init_cov.shape != expected_shape:  # 必须严格对齐冻结状态维度。
        raise ValueError(f"init_cov must have shape {expected_shape}, got {raw_init_cov.shape}")
    diagonal_values: list[float] = []
    for index, value in enumerate(raw_init_cov.tolist()):  # 逐个对角元素检查。
        if is_bool_like(value) or not is_real(value):  # bool/复数等都不允许。
            raise TypeError(f"init_cov[{index}] must be numeric")
        scalar_value = coerce_finite_scalar(value, name=f"init_cov[{index}]", min_value=0.0)  # NaN/Inf 不允许进入协方差，方差不能为负。
        diagonal_values.append(scalar_value)
    return np.diag(diagonal_values)


class EKFCore(EstimatorAPI):
    """把预测、UWB 更新和 VIO 更新串成一个最小 EKF 核心。

    本类实现了 :class:`EstimatorAPI` 接口，维护当前状态、协方差、
    锚点映射和最近一次更新报告，供上层 pipeline 反复调用。

    属性
    ----------
    cfg : dict
        初始化配置字典。
    name : str
        估计器名称标识。
    runtime_resource_meta : dict[str, float]
        资源占用估算。
    params : float
        参数规模指标。
    ram_peak : float
        峰值内存（兼容字段）。
    ram_peak_mb : float
        峰值内存 MB。
    last_update_report : dict[str, Any] | None
        最近一次更新的摘要报告。
    """

    def __init__(self, init_cfg: dict | None = None) -> None:
        """初始化 EKF 核心并准备默认状态、协方差和锚点映射。

        参数
        ----------
        init_cfg : dict | None
            初始化配置字典，可包含 ``name``、``init_state``、``init_cov``、
            ``anchor_layout``、``measurement_noise``、``process_noise`` 等字段。
            为 None 时使用空配置。

        注意
        -----
        构造完成后会自动调用 :meth:`reset` 回到标准初态。
        """
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "estimator": "EKFCore",
            "name": (init_cfg or {}).get("name", "ekf"),
            "cfg_keys": list(init_cfg.keys()) if isinstance(init_cfg, dict) else None,
            "measurement_noise": (init_cfg or {}).get("measurement_noise"),
            "process_noise": (init_cfg or {}).get("process_noise"),
        }, "EKFCore.__init__")
        self.cfg = copy.deepcopy(init_cfg or {})  # 深拷贝配置，防止外部修改嵌套字典污染内部状态。
        self.name = str(self.cfg.get("name") or "ekf")  # 如果没配名字，就使用默认标识。
        self.runtime_resource_meta = _build_runtime_resource_meta(self.cfg)  # 估算资源占用。
        self.params = float(self.runtime_resource_meta["params"])  # 对外展示的参数规模指标。
        self.ram_peak = float(self.runtime_resource_meta["ram_peak"])  # 保留兼容字段。
        self.ram_peak_mb = float(self.runtime_resource_meta["ram_peak_mb"])  # 统一按 MB 汇报。
        self.last_update_report: dict[str, Any] | None = None  # 最近一次更新摘要。
        self._state: dict[str, float] = {}  # 内部状态字典，按 state_items 保存。
        self._covariance = np.eye(len(state_items), dtype=float)  # 默认协方差先设成单位阵。
        self._anchor_lookup: dict[Any, tuple[float, float]] = {}  # 锚点位置映射。
        self._timestamp: float | None = None  # 最近一次事件时间戳。
        self._last_imu_timestamp: float | None = None  # 最近一次 IMU 事件时间戳。
        self._measurement_control: MeasurementControl | None = None  # 下一次 step 要用的控制参数。
        self._last_intermediate: ModelIntermediate | None = None  # 最近一次模型中间结果。
        self._last_vio_reference_pose: dict[str, float] | None = None  # 最近一次 VIO 参考位姿。
        self._last_vio_reference_pose_timestamp: float | None = None  # 最近一次 VIO 参考位姿的时间戳。
        # §2.2 真意：单位/时间零点/消偏规则全员同一，**不**禁止 R 在线缩放；
        # §5 期望默认 NN control(uwb_scaling/vio_scaling/bridge_scaling) 能生效，
        # 因此默认解冻，由配置/调用方显式 opt-in 标定冻结。
        self._calibration_frozen: bool = False  # 默认解冻；标定冻结需配置显式开启。
        EKFCore.reset(self)  # 初始化完成后，立刻回到标准初态。

    # ──────────────────────────────────────────────
    # 门控与鲁棒权重公共 helper（从 RobustEKFCore 提升）
    # ──────────────────────────────────────────────

    def _gate_cfg(self) -> dict[str, Any]:
        """取出 gate 子配置。"""
        gate = self.cfg.get("gate")
        if gate is None:
            return {}
        if not isinstance(gate, Mapping):
            raise TypeError("gate must be a mapping")
        return copy.deepcopy(gate)

    def _robust_cfg(self) -> dict[str, Any]:
        """取出 robust_weight 子配置。"""
        robust_weight = self.cfg.get("robust_weight")
        if robust_weight is None:
            return {}
        if not isinstance(robust_weight, Mapping):
            raise TypeError("robust_weight must be a mapping")
        return copy.deepcopy(robust_weight)

    def _quality_floor(self, modality: str) -> float:
        """按模态读取质量门槛。"""
        gate = self._gate_cfg()
        quality_floor_cfg = gate.get("quality_floor")
        if quality_floor_cfg is None:
            value = float(BRIDGE_THRESHOLDS[f"{modality}_hard_skip_quality_floor"])
        elif isinstance(quality_floor_cfg, Mapping):
            if modality not in quality_floor_cfg:
                raise KeyError(f"gate.quality_floor is missing modality key: {modality!r}")
            if is_bool_like(quality_floor_cfg[modality]):
                raise TypeError("gate.quality_floor mapping values must not be boolean-like")
            value = float(quality_floor_cfg[modality])
        else:
            if is_bool_like(quality_floor_cfg):
                raise TypeError("gate.quality_floor must not be boolean-like")
            value = float(quality_floor_cfg)
        value = coerce_finite_scalar(value, name="gate.quality_floor", min_value=0.0)
        return value

    def _nis_threshold(self, modality: str | None = None) -> float:
        """读取门控配置里的马氏距离阈值。

        支持两种配置形态：
        - 标量：``mahalanobis_sq: 9.21``，所有模态共享同一阈值。
        - 按模态映射：``mahalanobis_sq: {uwb: 6.635, vio: 11.345}``，需传入 ``modality`` 选值。

        参数
        ------
        modality : str | None
            量测模态（``"uwb"`` / ``"vio"``）。当配置为按模态映射时必传；
            配置为标量时可省略，直接返回标量值。
        """
        gate = self._gate_cfg()
        # §11.1 silent-skip 守卫：当 cfg 显式声明了 `gate` 块（说明用户有意启用卡方门控）
        # 但 `mahalanobis_sq` 键缺失时，旧实现会静默回退 `float("inf")` → NIS 永不超阈 →
        # 共享卡方门禁被无声关闭。这违反 §11.1「卡方置信水平全员同一、参数同源」要求，
        # 与 B14 audit 修正前的 `robust_ekf.yaml` 0.99 静默泄漏同源——属偷懒路径。
        # 现改为 fail-loud：cfg 显式给了 `gate` 但缺 `mahalanobis_sq` 即抛 KeyError，逼显式落配置。
        # 仅当 cfg 没有 `gate` 块时才允许 inf 兜底（与铁律 10 FGO 裸跑向后兼容）。
        if "gate" in self.cfg and self.cfg["gate"] is not None and "mahalanobis_sq" not in gate:
            raise KeyError(
                "gate.mahalanobis_sq is missing while `gate` block is explicitly declared; "
                "§11.1 共享卡方门禁要求显式配置卡方阈值，"
                "缺失会被静默回退为 inf（NIS 永不超阈），属 §11.1 silent-skip 偷懒路径，必须显式落值"
            )
        raw = gate.get("mahalanobis_sq", float("inf"))
        if isinstance(raw, Mapping):
            if modality is None:
                raise ValueError(
                    "gate.mahalanobis_sq is a per-modality mapping; "
                    "modality must be provided"
                )
            if modality not in raw:
                raise KeyError(
                    f"gate.mahalanobis_sq is missing modality key: {modality!r}"
                )
            entry = raw[modality]
            if is_bool_like(entry):
                raise TypeError("gate.mahalanobis_sq mapping values must not be boolean-like")
            value = float(entry)
        else:
            if is_bool_like(raw):
                raise TypeError("gate.mahalanobis_sq must be numeric, got bool")
            value = float(raw)
        if math.isinf(value):
            if value < 0.0:
                raise ValueError("gate.mahalanobis_sq must not be negative infinity")
            return value
        value = coerce_finite_scalar(value, name="gate.mahalanobis_sq", min_value=0.0)
        return value

    def _huber_weight(self, whitened_residual_norm: float) -> float:
        """按 Huber 规则计算鲁棒权重。

        身份纪律（cmp2）：
        - **无** ``robust_weight`` 配置（标准 EKF 身份）→ 强制返回 1.0，不做 M 估计降权。
        - **有** ``robust_weight``（Robust-EKF 身份）→ 按 type/delta 计算 Huber 权重。
        禁止「标准 EKF yaml 写 Huber」却仍声称与 Robust 同档比较 2。
        """
        if is_bool_like(whitened_residual_norm):
            raise TypeError("whitened_residual_norm must be numeric, got bool")
        norm_value = coerce_finite_scalar(whitened_residual_norm, name="whitened_residual_norm", min_value=0.0)
        robust_cfg = self._robust_cfg()
        # 标准 EKF：无 robust_weight 块 → 平方损失（权重恒 1），与 Robust-EKF 身份分离。
        if not robust_cfg:
            return 1.0
        robust_type = str(robust_cfg.get("type", "huber")).lower()
        if robust_type != "huber":
            raise ValueError(f"Unsupported robust weight type: {robust_type}")
        delta_raw = robust_cfg.get("delta", 1.0)
        if is_bool_like(delta_raw):
            raise TypeError("robust_weight.delta must be numeric, got bool")
        delta = coerce_finite_scalar(delta_raw, name="robust_weight.delta", min_value=0.0, inclusive=False)
        if norm_value <= delta:
            return 1.0
        return max(delta / norm_value, 1e-6)

    def _quality_value(self, payload_section: Mapping[str, Any] | None) -> float:
        """从负载里提取 quality。

        §6.4 #21 质量/特征数/重投影不得变相成为 NLOS 真值标签硬子要求对应代码层落点：
        本函数把 quality 严格规范化为 [0, 1] 区间连续标量，仅作为「共享无效门 / 已声明通道」
        使用——下游 _handle_vio 里 quality_floor 校验（vio_quality_floor / quality_below_floor）
        与 vio_quality_zero 协议级短路（quality ≤ 0 视为无效观测跳过更新）是「门控」语义，不是
        NLOS 二分类标签。tracked_features / reproj_err 字段已由铁律 3 Stage A1 删除（VIO schema
        根本不再有这两字段，sim_materializer L2283-2295 注释交叉验证），故 #21 中"特征数/重投影"
        候选标签在 schema 层已被排除，仅"质量"一项走本函数规范化为门控标量。
        全员同一：EKF 本处 L450 + RobustEKF 继承 + FGO fgo_core.py:755 重写（引用 DEFAULT_THRESHOLDS
        单源常量更严格），4 项校验（非 Mapping → 1.0 / bool → TypeError / NaN → ValueError /
        范围 → ValueError）逻辑等价，三者行为同口径。
        """
        if not isinstance(payload_section, Mapping):
            return 1.0
        quality = payload_section.get("quality", 1.0)
        if is_bool_like(quality):
            raise TypeError("quality must be numeric, got bool")
        value = coerce_finite_scalar(quality, name="quality")
        quality_min = 0.0
        quality_max = 1.0
        if value < quality_min:
            raise ValueError(f"quality must be non-negative, got {value}")
        if value > quality_max:
            raise ValueError(f"quality must not exceed {quality_max}, got {value}")
        return value

    def reset(self, initial_state: dict | None = None) -> None:
        """把内部状态恢复到初始值。

        参数
        ----------
        initial_state : dict | None
            外部显式指定的初始状态，优先级高于配置中的 ``init_state``。

        注意
        -----
        优先级顺序：``initial_state`` 参数 > 配置中的 ``init_state`` > 全零默认。
        """
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "estimator": "EKFCore",
            "has_initial_state": initial_state is not None,
            "initial_state_keys": list(initial_state.keys()) if isinstance(initial_state, dict) else None,
        }, "EKFCore.reset")
        base_state = {state_key: 0.0 for state_key in state_items}  # 先给每个状态项置零。
        cfg_init_state = self.cfg.get("init_state")  # 读取配置中的初始状态覆盖。
        if cfg_init_state is not None:  # 显式给了就必须合法，不能静默忽略。
            base_state.update(_validate_state_override_mapping(cfg_init_state, path="init_state"))
        if initial_state is not None:  # 外部显式传入的初始状态优先级更高。
            base_state.update(_validate_state_override_mapping(initial_state, path="initial_state"))
        self._state = base_state  # 保存初始化后的状态字典。
        self._covariance = _build_initial_covariance(self.cfg.get("init_cov"))  # 初始协方差也必须显式合法。
        anchor_layout = self.cfg.get("anchor_layout")  # 取锚点布局配置。
        self._anchor_lookup = build_anchor_lookup(anchor_layout) if anchor_layout is not None else {}  # 生成锚点查询表。
        self._timestamp = None  # 重置时间戳缓存。
        self._last_imu_timestamp = None  # 重置 IMU 专属时间戳缓存。
        self._measurement_control = None  # 清空待用控制参数。
        self._last_intermediate = None  # 清空上游中间结果。
        self._last_vio_reference_pose = None  # 清空 VIO 参考位姿。
        self._last_vio_reference_pose_timestamp = None  # 清空 VIO 参考位姿时间戳。
        self.last_update_report = None  # 清空上一次更新报告。
        # 与 __init__ 同口径：默认解冻，保留 NN control 全套缩放字段（§5 期望）。
        self._calibration_frozen = False  # reset 后仍解冻；标定冻结需配置显式开启。

    def apply_p16_init_from_first_frame(
        self,
        uwb_events: Sequence[Any],
        vio_events: Sequence[Any],
        *,
        z_anchor_m: float = 2.5,
        z_tag_m: float = 1.2,
    ) -> dict[str, Any]:
        """手册 P16 EKF 初始化协议的统一入口（异步高NLOS实验全流程保障手册）。

        调用 ``ekf_init_protocol_p16`` 拿到 ``initial_state`` 与 ``init_cov_diagonal``，
        再以这两个值调用 ``self.reset(initial_state=...)`` 并重写 cov 为 P16 计算的对角向量。
        返回初始化报告（含 mode / n_anchors_used / trilaterated_position_xy）供上游审计与日志。

        参数:
            uwb_events: 首批 UWB 事件（建议取序列前若干帧，包含所有 4 锚的首次有效测量）。
            vio_events: 首批 VIO 事件（建议取序列前若干帧，包含首帧 vio_yaw）。
            z_anchor_m / z_tag_m: 锚/tag z 高度（手册 S2 锚高 2.5m，tag 高 1.1-1.3m）。

        返回:
            dict（与 ``ekf_init_protocol_p16`` 同结构）。
        """
        from liquidloc.estimators.ekf_init_protocol import ekf_init_protocol_p16  # 局部导入避免循环依赖

        report = ekf_init_protocol_p16(
            uwb_events=uwb_events,
            vio_events=vio_events,
            anchor_lookup=self._anchor_lookup,
        )
        self.reset(initial_state=report["initial_state"])
        # _build_initial_covariance 期望 init_cov 是与 state_items 同长的对角向量
        self._covariance = _build_initial_covariance(report["init_cov_diagonal"])
        return report

    def consume_model_intermediate(self, intermediate: ModelIntermediate) -> None:
        """保存上游模型中间结果，供需要时转发或调试。

        参数
        ----------
        intermediate : ModelIntermediate
            上游模型产生的中间结果对象，只缓存不修改。
        """
        if not isinstance(intermediate, ModelIntermediate):
            raise TypeError(
                f"intermediate must be a ModelIntermediate, got {type(intermediate).__name__}"
            )
        self._last_intermediate = intermediate  # 只缓存，不修改内容。

    def set_measurement_control(self, control: MeasurementControl) -> None:
        """设置下一次 step 要使用的测量控制参数。

        参数
        ----------
        control : MeasurementControl
            待使用的测量控制对象，会在下一次 :meth:`step` 时生效，
            用完后自动清空。

        注意
        -----
        控制参数只用一次，:meth:`step` 的 ``finally`` 块会清空此缓存。
        """
        if not isinstance(control, MeasurementControl):
            raise TypeError(
                f"control must be a MeasurementControl, got {type(control).__name__}"
            )
        self._measurement_control = control  # 这会在下一次事件处理时生效。

    def _state_vector(self) -> np.ndarray:
        """把内部字典状态转成固定顺序的向量。

        返回
        -------
        np.ndarray
            按 :data:`state_items` 冻结顺序排列的浮点状态向量。
        """
        return np.asarray([float(self._state[state_key]) for state_key in state_items], dtype=float)  # 顺序严格按契约，缺键直接抛 KeyError 避免静默漂移。

    def _update_from_vector(self, x_vector: np.ndarray | Sequence[float], covariance: np.ndarray | Sequence[Sequence[float]]) -> None:
        """把向量状态和协方差写回内部缓存。

        参数
        ----------
        x_vector : np.ndarray | Sequence[float]
            状态向量，长度必须与 :data:`state_items` 一致。
        covariance : np.ndarray | Sequence[Sequence[float]]
            协方差矩阵，维度为 N×N（N 为状态维度）。

        注意
        -----
        写入时按 :data:`state_items` 冻结顺序重建内部字典。
        """
        x_vector = np.asarray(x_vector, dtype=float)  # 先规整成浮点数组。
        covariance = np.asarray(covariance, dtype=float)  # 协方差也先规整成浮点数组，供后续校验和写回使用。
        expected_dim = len(state_items)  # 冻结状态维度，作为形状校验基准。
        if x_vector.shape != (expected_dim,):  # 状态向量必须是一维且长度与 state_items 一致，否则索引会错位或越界。
            raise ValueError(
                f"x_vector must have shape ({expected_dim},), got {x_vector.shape}"
            )
        if covariance.shape != (expected_dim, expected_dim):  # 协方差必须是 N×N 方阵，否则后续矩阵运算会出错。
            raise ValueError(
                f"covariance must have shape ({expected_dim}, {expected_dim}), got {covariance.shape}"
            )
        if not np.all(np.isfinite(x_vector)):  # NaN/Inf 会污染状态，必须拒绝，与 _validate_state_override_mapping 对齐。
            raise ValueError("x_vector contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(covariance)):  # NaN/Inf 会破坏协方差矩阵，必须拒绝，与 _build_initial_covariance 对齐。
            raise ValueError("covariance contains non-finite values (NaN or Inf)")
        self._state = {  # 按冻结顺序重建状态字典。
            state_key: float(x_vector[index])  # 每个键都写回对应分量。
            for index, state_key in enumerate(state_items)
        }
        self._covariance = covariance.copy()  # 协方差也同步写回，强制拷贝防止视图别名污染。

    def _measurement_noise_cfg(self) -> dict[str, Any]:
        """取出 measurement_noise 子配置并强制转成普通字典。

        返回
        -------
        dict[str, Any]
            measurement_noise 配置的普通字典副本。

        异常
        ------
    TypeError
        当 ``measurement_noise`` 存在但不是 Mapping 类型时抛出。
        """
        measurement_noise = self.cfg.get("measurement_noise")  # 取出原始噪声配置。
        if measurement_noise is None:  # 没配时用空字典兜底。
            measurement_noise = {}  # 空配置不会影响后续读取。
        if not isinstance(measurement_noise, Mapping):  # 结构必须是映射。
            raise TypeError("measurement_noise must be a mapping")  # 不对就报错。
        return copy.deepcopy(dict(measurement_noise))  # 深拷贝，防止嵌套dict被外部修改污染cfg。

    def _required_measurement_noise(self, modality: str) -> Any:
        """取出指定模态的基础测量噪声。

        参数
        ----------
        modality : str
            量测模态名称，如 ``"uwb"`` 或 ``"vio"``。

        返回
        -------
        Any
            该模态的基础噪声结构（标量、向量、矩阵或映射）。

        异常
        ------
        KeyError
            当指定模态未在 ``measurement_noise`` 中配置时抛出。
        TypeError
            当噪声值为布尔类或 ``complex`` 时抛出。
        ValueError
            当噪声值非正时抛出。
        """
        measurement_noise = self._measurement_noise_cfg()  # 读取完整噪声配置（U14 提供深拷贝隔离）。
        if modality not in measurement_noise:  # 指定模态必须显式配置。
            raise KeyError(f"measurement_noise.{modality} is required for {modality} updates")  # 缺项直接报错。
        base_noise = measurement_noise[modality]  # 取出基础噪声。
        _require_positive_measurement_noise_entry(base_noise, path=f"measurement_noise.{modality}")  # 再检查值合法。
        return base_noise  # 返回原始噪声结构。

    def _current_pose_reference(self) -> dict[str, float]:
        """构造当前位姿参考字典。

        返回
        -------
        dict[str, float]
            包含 VIO 更新合同 ``updated_state_items`` 所要求的全部位姿键
            (px、py、yaw 以及紧耦合扩维后的 uwb_clock_bias、
            vio_scale). 紧耦合项不参与 VIO 帧间位姿参考值, 写入 0.0,
            仅为通过 ``_resolve_state_items_from_mapping`` 的键集校验.
        """
        return {  # 这里只保留 VIO 参考需要的位姿量, 紧耦合项填零占位.
            "px": float(self._state["px"]),  # 当前 x，缺键直接抛 KeyError 避免静默漂移。
            "py": float(self._state["py"]),  # 当前 y，缺键直接抛 KeyError 避免静默漂移。
            "yaw": float(self._state["yaw"]),  # 当前航向，缺键直接抛 KeyError 避免静默漂移。
            "uwb_clock_bias": float(self._state.get("uwb_clock_bias", 0.0)),  # 紧耦合项填零占位.
            "vio_scale": float(self._state.get("vio_scale", 0.0)),  # 紧耦合项填零占位.
        }

    def _resolve_anchor_position(self, anchor_id: Any) -> tuple[float, float]:
        """把锚点编号解析成实际坐标，委托给 gt_utils.resolve_anchor_position。

        参数
        ----------
        anchor_id : Any
            锚点编号，通常为整数或字符串（如 ``0``、``"A0"``）。
            查找时会自动尝试整数、纯数字字符串和 ``Axx`` 格式之间的互转。

        返回
        -------
        tuple[float, float]
            锚点的 ``(x, y)`` 二维坐标，坐标由 ``build_anchor_lookup`` 保证为有限浮点数。

        异常
        ------
        ValueError
            当 ``anchor_id`` 在 ``self._anchor_lookup`` 中找不到任何匹配候选，
            或对应坐标结构不合法时抛出。调用方（如 :meth:`_handle_uwb`）应确保
            事件已通过 ``validate_event`` 协议校验，但锚点是否存在于当前布局
            需在此处显式失败。
        """
        return resolve_anchor_position(anchor_id, self._anchor_lookup)

    def _default_control(self, modality: str) -> MeasurementControl:
        """构造默认测量控制对象。

        参数
        ----------
        modality : str
            量测模态名称。

        返回
        -------
        MeasurementControl
            仅绑定模态的默认控制对象，所有缩放参数为默认值。
        """
        return MeasurementControl(modality=modality)  # 默认控制只绑定模态。

    def _effective_imu_dt(self, payload: dict[str, Any]) -> float:
        """返回本次 IMU 预测应使用的有效时间步长。

        如果之前没有记录过 IMU 时间戳，就直接使用负载里的 dt 字段；
        否则用当前时间减去上次 IMU 时间戳，得到真实经过的时间间隔。

        参数
        ----------
        payload : dict[str, Any]
            IMU 事件负载，必须包含 ``t``（当前时间戳）和 ``dt``（声明步长）字段。

        返回
        -------
        float
            有效时间步长。可能为 0 或负值（首帧 dt 可为 0，或相邻 IMU
            时间戳在容差范围内相等/倒序），由调用方 ``_handle_imu`` 在
            ``<= 0`` 时跳过预测步。
        """
        if self._last_imu_timestamp is None:  # 首次 IMU 事件没有前驱，只能用负载自带的 dt。
            return float(payload["dt"])  # 直接返回声明步长。
        return float(payload["t"]) - float(self._last_imu_timestamp)  # 用相邻 IMU 事件时间差作为真实步长。

    def _handle_imu(self, payload: dict[str, Any], x_prev: np.ndarray, control: MeasurementControl) -> dict[str, Any]:
        """处理 IMU 事件，执行预测步。

        参数
        ----------
        payload : dict[str, Any]
            IMU 事件负载，必须包含 ``dt``、``ax``、``ay``、``gz`` 字段。
        x_prev : np.ndarray
            预测前的状态向量。
        control : MeasurementControl
            本次测量控制参数。

        返回
        -------
        dict[str, Any]
            更新报告，包含 modality、update_applied、reason、measurement_control。
        """
        current_t = float(payload["t"])  # 取出当前 IMU 事件的时间戳。
        imu_dt = self._effective_imu_dt(payload)  # 计算本次预测应使用的有效时间步长。
        # 检查 IMU 数据缺失掩码：如果 ax/ay/gz 中有缺失字段，
        # 增大过程噪声以反映预测不确定性，避免基于补零值产生错误预测。
        imu_missing_mask = payload.get("imu_payload", {}).get("missing_mask")
        imu_missing_inflation = 1.0  # 默认无膨胀。
        # 与 predict_step.py 对齐：missing_mask 必须是 list/tuple 且长度 >= 3
        # （对应 imu_fields [ax, ay, gz]），否则视为无掩码，避免部分掩码语义漂移。
        if isinstance(imu_missing_mask, (list, tuple)) and len(imu_missing_mask) >= 3:
            # imu_fields 顺序为 [ax, ay, gz]，missing_mask 与之一一对应。
            imu_field_missing = [bool(m) for m in imu_missing_mask[:3]]
            if any(imu_field_missing):
                # 任一 IMU 字段缺失时，膨胀过程噪声 10 倍以反映预测不确定性。
                # 这比直接跳过预测步更安全——跳过会导致状态完全不变，
                # 而膨胀噪声至少让协方差增长，反映真实的不确定性。
                imu_missing_inflation = float(BRIDGE_THRESHOLDS["imu_missing_inflation"])  # §11.2 Q 固定：膨胀系数由协议 bridge_thresholds 单源真相写死，非 estimator 私调。
        if imu_dt <= 0.0:  # 非正时间步长说明事件顺序异常或重复，不能推进。
            self._last_imu_timestamp = current_t  # 仍然要更新时间戳缓存，避免后续事件也卡住。
            return {
                "modality": "imu",  # 模态标签。
                "update_applied": False,  # 没有真正预测。
                "reason": "nonpositive_dt",  # dt 非正。
                # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "gate": {"passed": False, "rejected_by": "nonpositive_dt"},
            }

        # 如果 IMU 数据有缺失，膨胀过程噪声以反映预测不确定性。
        predict_cfg = self.cfg  # 默认使用原始配置。
        if imu_missing_inflation > 1.0:
            predict_cfg = copy.deepcopy(self.cfg)
            # 与 predict_step.py 对齐：process_noise 为 None 时按空映射处理，
            # 避免 dict(None) 抛出 TypeError。
            process_noise = dict(predict_cfg.get("process_noise") or {})
            for noise_key in ("pos", "vel", "yaw", "accel_bias", "gyro_bias"):
                if noise_key in process_noise:
                    process_noise[noise_key] = float(process_noise[noise_key]) * imu_missing_inflation
            predict_cfg["process_noise"] = process_noise

        x_pred, P_pred = run_predict_step(  # 调用纯 IMU 预测步骤，推进状态和协方差。
            x_prev,  # 上一时刻状态向量。
            self._covariance,  # 上一时刻协方差矩阵。
            payload,  # IMU 事件负载，包含加速度和角速度。
            imu_dt,  # 有效时间步长。
            predict_cfg,  # 预测配置，含可能膨胀的过程噪声。
        )
        self._update_from_vector(x_pred, P_pred)  # 把预测结果写回内部状态和协方差缓存。
        self._last_imu_timestamp = current_t  # 记录本次 IMU 时间戳，供下次计算真实 dt。
        return {
            "modality": "imu",  # 模态标签。
            "update_applied": True,  # 已成功执行预测。
            "reason": "predict_step",  # 说明这是预测步。
            "measurement_control": control_to_dict(control),  # 当前控制参数。
        }

    def _handle_uwb(self, payload: dict[str, Any], x_prev: np.ndarray, control: MeasurementControl) -> dict[str, Any]:
        """处理 UWB 事件，执行范围更新。

        参数
        ----------
        payload : dict[str, Any]
            UWB 事件负载，必须包含 ``uwb_payload`` 子字典
            （含 anchor_id、range、valid、quality 字段）。
        x_prev : np.ndarray
            更新前的状态向量。
        control : MeasurementControl
            本次测量控制参数。

        返回
        -------
        dict[str, Any]
            更新报告，包含 modality、update_applied、measurement_control、
            covariance_report 及 UWB 更新详情。

        注意
        -----
        - 当 ``gate_action == "uwb_skip_update"`` 时跳过更新。
        - 当 ``uwb_payload`` 缺失时跳过更新。
        - 当 ``valid == False`` 时跳过更新（协议级数据有效性检查，
          与 ``_handle_vio`` 的 ``quality <= 0.0`` 检查对齐）。
        - 偏置修正后距离小于 0 时会被钳位到 0。
        """
        if control.gate_action == "uwb_skip_update":  # 控制层显式要求跳过本次 UWB 更新。
            return {
                "modality": "uwb",  # 模态标签。
                "update_applied": False,  # 被门控跳过。
                "reason": "uwb_skip_update",  # 跳过原因。
                # §3.0.2 audit: UWB skip also declares h_side.
                "bias_writeport": "h_side",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "gate": {"passed": False, "rejected_by": "uwb_skip_update"},
            }
        uwb_payload = payload.get("uwb_payload")  # 提取 UWB 负载。
        if uwb_payload is None:  # 负载缺失时无法执行更新，直接跳过（与 RobustEKFCore/FGOCore 对齐）。
            return {
                "modality": "uwb",
                "update_applied": False,
                "reason": "missing_uwb_payload",
# §3.0.2 审计：UWB 缺失 payload 也声明 h_side。
            "bias_writeport": "h_side",
            "measurement_control": control_to_dict(control),
                "gate": {"passed": False, "rejected_by": "missing_uwb_payload"},
            }
        # valid=False 的 UWB 事件语义上是无效观测，跳过更新。
        # 与 _handle_vio 的 quality<=0.0 检查对齐：协议级数据有效性检查，
        # 不依赖控制层 gate_action（默认 pass_through 时仍需生效）。
        uwb_valid = uwb_payload.get("valid", True)
        if is_bool_like(uwb_valid) and not bool(uwb_valid):
            return {
                "modality": "uwb",
                "update_applied": False,
                "reason": "uwb_invalid_measurement",
                # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "h_side",
                "measurement_control": control_to_dict(control),
                "gate": {"passed": False, "rejected_by": "uwb_invalid_measurement"},
            }
        # ── 质量门控（gate.quality_floor 配置存在时启用） ──
        if self.cfg.get("gate") is not None:
            quality_gate = self._quality_value(uwb_payload)
            if quality_below_floor(quality_gate, self._quality_floor("uwb")):
                return {
                    "modality": "uwb",
                    "update_applied": False,
                    "reason": "quality_floor",
                    # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "h_side",
                    "measurement_control": control_to_dict(control),
                    "covariance_report": None,
                    "robust_covariance_report": None,
                    "gate": {"passed": False, "quality": quality_gate,
                             "quality_floor": self._quality_floor("uwb"), "nis": None,
                             "mahalanobis_sq_threshold": self._nis_threshold("uwb"),
                             "rejected_by": "quality_floor"},
                    "robust": None,
                }
        else:
            quality_gate = None

        base_uwb_noise = self._required_measurement_noise("uwb")  # 取基础 UWB 噪声（标准差）。
        base_uwb_noise = _square_std_to_var(base_uwb_noise)  # 标准差平方为方差，与过程噪声语义一致。
        effective_uwb_noise, cov_report = build_controlled_measurement_cov(
            base_uwb_noise,  # 方差。
            control,  # 当前控制对象。
            modality="uwb",  # 本次处理的是 UWB。
            calibration_frozen=self._calibration_frozen,
        )
        # UWB 更新只接受标量噪声（一维测距），非标量配置需在此显式拒绝，
        # 与 robust_ekf_core 和 fgo_core 对齐，避免错误深入 run_uwb_update 后才暴露。
        try:
            noise_arr = np.asarray(effective_uwb_noise, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                f"effective UWB noise must be scalar, got {type(effective_uwb_noise).__name__}"
            ) from None
        if noise_arr.ndim != 0:
            raise ValueError(f"effective UWB noise must be scalar, got shape {noise_arr.shape}")
        scalar_noise = coerce_finite_scalar(noise_arr.reshape(-1)[0], name="effective UWB noise")
        anchor_pos = self._resolve_anchor_position(uwb_payload["anchor_id"])  # 解析锚点坐标。
        # §3.0.2 / §3.1.2：NN+EKF 的 bias 作为 h(·) 侧有界偏置进入观测模型，
        # 残差仍在原始读数合同上定义（raw_range - (z_pred_geom + uwb_clock_bias + bias_applied)）。
        # 不再对 raw 距离做 subtractive 改写为 corrected_range；raw_range 直接作为 z 进 update。
        raw_range = float(uwb_payload["range"])
        bias_applied_h = float(control.bias_applied)  # 已由 clip_uwb_bias 截断为非负有界。

        # ── NIS 门控 + Huber 降权（gate 配置存在时启用） ──
        z_pred = predict_range(x_prev, anchor_pos, extra_bias=bias_applied_h)  # h 侧含 bias 的预测测距。
        residual = raw_range - z_pred  # 残差定义在原始 z 上（§3.0.2）。
        H = build_uwb_jacobian(x_prev, anchor_pos)
        S = coerce_finite_scalar((H @ self._covariance @ H.T)[0, 0] + scalar_noise, name="UWB innovation covariance")
        # §11.5 抖动注入：UWB 路径标量 S 与 VIO 路径同口径，缺 jitter fallback 是 v5 漏审。
        # 三方法（EKF / Robust-EKF / FGO）UWB 路径 S<=0 拒绝前先尝试 jitter 修补；
        # 二次仍失败 → fail-loud（与 VIO 路径 _ensure_positive_definite_vio_innovation_covariance 同政策）。
        if S <= 0.0:
            from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
            cov_jitter_eps = float(BRIDGE_THRESHOLDS["cov_jitter_eps"])
            jittered_S = S + cov_jitter_eps
            if jittered_S > 0.0:
                S = jittered_S  # 接受 jittered 版本作为该次创新协方差。
            else:
                return {
                    "modality": "uwb", "update_applied": False, "reason": "nonpositive_innovation_covariance",
                    # §3.0.2 审计：UWB reject 路径声明 h_side（与成功路径同口径）。
                    "bias_writeport": "h_side",
                    "measurement_control": control_to_dict(control),
                    "covariance_report": cov_report, "robust_covariance_report": None,
                    "gate": {"passed": False, "rejected_by": "nonpositive_innovation_covariance",
                             "quality": quality_gate if quality_gate is not None else uwb_payload.get("quality", 1.0),
                             "nis": None, "mahalanobis_sq_threshold": self._nis_threshold("uwb")},
                    "robust": None,
                }

        robust_weight_applied = 1.0  # 默认不降权（无 gate 配置时）
        if self.cfg.get("gate") is not None:
            nis = float((residual * residual) / S)
            if not math.isfinite(nis):
                return {
            "modality": "uwb", "update_applied": False, "reason": "nonfinite_nis",
            # §3.0.2 审计：UWB reject 声明 h_side。
            "bias_writeport": "h_side",
            "measurement_control": control_to_dict(control),
            "covariance_report": cov_report, "robust_covariance_report": None,
                    "gate": {"passed": False, "rejected_by": "nonfinite_nis",
                             "quality": quality_gate, "nis": nis,
                             "mahalanobis_sq_threshold": self._nis_threshold("uwb")},
                    "robust": None,
                }
            if nis > self._nis_threshold("uwb"):
                return {
            "modality": "uwb", "update_applied": False, "reason": "mahalanobis_sq",
            # §3.0.2 审计：UWB reject 声明 h_side。
            "bias_writeport": "h_side",
            "measurement_control": control_to_dict(control),
            "covariance_report": cov_report, "robust_covariance_report": None,
                    "gate": {"passed": False, "rejected_by": "mahalanobis_sq",
                             "quality": quality_gate, "nis": nis,
                             "mahalanobis_sq_threshold": self._nis_threshold("uwb")},
                    "robust": None,
                }
            whitened = math.sqrt(max(nis, 0.0))
            robust_weight_applied = self._huber_weight(whitened)
            covariance_scale = 1.0 / max(robust_weight_applied, 1e-6)
            effective_uwb_noise, robust_cov_report = build_effective_cov(
                effective_uwb_noise, uwb_scaling=covariance_scale
            )
        else:
            robust_cov_report = None
        x_upd, P_upd, update_info = run_uwb_update(
            x_prev,  # 更新前状态。
            self._covariance,  # 更新前协方差。
            anchor_pos,  # 锚点坐标。
            raw_range,  # 原始测距（残差在原始 z 合同上定义，bias 已进 h(·)）。
            effective_uwb_noise,  # 有效测量噪声。
            extra_bias=bias_applied_h,  # h 侧有界偏置修正（§3.0.2 / §3.1.2）。
        )
        self._update_from_vector(x_upd, P_upd)  # 把更新结果写回内部缓存。
        # 从 update_info 中排除 gate/modality/update_applied/reason，
        # 因为 ekf_core 有自己的报告结构，这些字段由上层构造。
        _update_detail = {k: v for k, v in update_info.items()
                          if k not in ("gate", "modality", "update_applied", "reason")}
        return {
            "modality": "uwb",  # 模态标签。
            "update_applied": True,  # 已成功更新。
            "reason": "uwb_update_success",  # 更新原因。
            "measurement_control": control_to_dict(control),  # 当前控制参数。
            "covariance_report": cov_report,  # 协方差报告。
            # §3.0.2 审计：显式标记本次 bias 写入 h(·) 侧，残差定义在原始 z 上。
            "bias_writeport": "h_side",
            "gate": {"passed": True, "rejected_by": None},  # 门控信息。
            **_update_detail,  # 展开 UWB 更新的详细信息（不含 gate 等已构造字段）。
        }

    def _handle_vio(self, payload: dict[str, Any], x_prev: np.ndarray, control: MeasurementControl) -> dict[str, Any]:
        """处理 VIO 事件，执行相对位姿更新。

        参数
        ----------
        payload : dict[str, Any]
            VIO 事件负载，必须包含 ``vio_payload`` 子字典
            （含 dx、dy、dyaw 等字段）。
        x_prev : np.ndarray
            更新前的状态向量。
        control : MeasurementControl
            本次测量控制参数。

        返回
        -------
        dict[str, Any]
            更新报告，包含 modality、update_applied、measurement_control、
            covariance_report 及视觉更新详情。

        注意
        -----
        - 当 ``gate_action == "vio_skip_update"`` 时跳过更新。
        - 优先使用 ``_last_vio_reference_pose``，未设置时跳过更新并初始化参考位姿为当前位姿。
        - 更新成功后会刷新 ``_last_vio_reference_pose``。
        """
        if control.gate_action == "vio_skip_update":  # 控制层显式要求跳过本次 VIO 更新。
            # 跳过更新时重置参考位姿为当前估计位姿，避免下一帧 VIO 更新时
            # 使用过时参考位姿导致残差语义不匹配（仿真 VIO 数据提供相邻帧增量，
            # 而非参考位姿到当前的增量）。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {
                "modality": "vio",  # 模态标签。
                "update_applied": False,  # 被门控跳过。
                "reason": "vio_skip_update",  # 跳过原因。
                # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "gate": {"passed": False, "rejected_by": "vio_skip_update"},
            }
        # quality=0.0 的 VIO 事件语义上是无效观测（如仿真 cycle 边界帧），跳过更新。
        vio_payload = payload.get("vio_payload") or {}
        # §11.3-d 共享「VIO 无效」串项同源 v11 audit 修复：
        # v10 之前本处走捷径 `float(vio_payload.get("quality", 1.0))`，绕过
        # `self._quality_value()` 规范化（不校验 is_bool_like / coerce_finite_scalar /
        # [0,1] 范围），与 Robust-EKF `robust_ekf_core.py:536` + FGO `fgo_core.py:2098`
        # 不同源。后果：(a) np.bool_(False) 在 EKF 触发跳过而 Robust/FGO 抛 TypeError；
        # (b) NaN 在 EKF 不跳过继续，Robust/FGO 抛 ValueError；(c) 大于 1.0 的 quality
        # 在 EKF 接受进入后续，Robust/FGO 抛 ValueError。违 §11.3-d 「串联顺序全员固定」
        # 与 `ekf_core.py:473-474` 自家注释「4 项校验逻辑等价」自相矛盾。
        # v11 改为走 `_quality_value` 与 Robust/FGO 同口径同源规范化。
        try:
            vio_quality = self._quality_value(vio_payload)
        except (TypeError, ValueError):
            # 规范化失败时（bool / NaN / 超界）仿 Robust/FGO raise 路径，
            # 但 VIO 路径必须保护参考位姿 - 先重置再 raise。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            raise
        if vio_quality is not None and float(vio_quality) <= 0.0:
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {
                "modality": "vio",
                "update_applied": False,
                "reason": "vio_quality_zero",
                # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),
                "gate": {"passed": False, "quality": float(vio_quality), "rejected_by": "vio_quality_zero"},
            }
        # ── VIO 质量门控（gate 配置存在时启用） ──
        if self.cfg.get("gate") is not None:
            vio_quality_gate = self._quality_value(vio_payload)
            if quality_below_floor(vio_quality_gate, self._quality_floor("vio")):
                self._last_vio_reference_pose = self._current_pose_reference()
                self._last_vio_reference_pose_timestamp = self._timestamp
                return {
                    "modality": "vio",
                    "update_applied": False,
                    "reason": "quality_floor",
                    # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "none",
                    "measurement_control": control_to_dict(control),
                    "covariance_report": None,
                    "robust_covariance_report": None,
                    "gate": {"passed": False, "quality": vio_quality_gate,
                             "quality_floor": self._quality_floor("vio"), "nis": None,
                             "mahalanobis_sq_threshold": self._nis_threshold("vio"),
                             "rejected_by": "quality_floor"},
                    "robust": None,
                }

        base_vio_noise = self._required_measurement_noise("vio")  # 取基础 VIO 噪声（标准差）。
        base_vio_noise = _square_std_to_var(base_vio_noise)  # 标准差平方为方差，与过程噪声语义一致。
        effective_vio_cov, cov_report = build_controlled_measurement_cov(
            base_vio_noise,  # 方差。
            control,  # 当前控制对象。
            modality="vio",  # 本次处理的是 VIO。
            calibration_frozen=self._calibration_frozen,
        )
        # 首个 VIO 事件参考位姿检查：如果 _last_vio_reference_pose 为 None，
        # 参考位姿回退到当前估计位姿，导致 z_hat=0 而 z_vio 非零（仿真 VIO 提供
        # 相邻帧增量），残差异常放大，等效于双重计数运动量。此时应跳过更新，
        # 将参考位姿设为当前位姿供下一帧使用。
        if self._last_vio_reference_pose is None:
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {
                "modality": "vio",
                "update_applied": False,
                "reason": "vio_first_frame_reference_init",
                # §3.0.2 审计要求：VIO 不接受 NN bias（_handle_vio 全程 bias_applied=0.0），
                # 拒绝路径同口径标记 "none"，便于顶层审计区分「update_applied=False 也声明 write-port」。
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),
                "covariance_report": cov_report,
                "gate": {"passed": False, "rejected_by": "vio_first_frame_reference_init"},
            }
        # 参考位姿时效性检查：如果参考位姿超过阈值未更新，说明中间可能有 VIO 事件
        # 被 blackout 移除或门控跳过。仿真 VIO 数据提供相邻帧增量，参考位姿过时时
        # 增量语义不匹配（相邻帧增量 ≠ 参考帧增量），此时应重置参考位姿并跳过更新，
        # 避免残差方向错误导致定位精度退化。
        ref_pose_stale = (
            self._last_vio_reference_pose is not None
            and self._last_vio_reference_pose_timestamp is not None
            and self._timestamp is not None
            and (self._timestamp - self._last_vio_reference_pose_timestamp) > VIO_REF_POSE_STALE_SECONDS
        )
        if ref_pose_stale:
            # 参考位姿过时：重置为当前位姿，跳过本次更新。
            # 下一次 VIO 事件的参考位姿将是当前位姿，增量语义恢复一致。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {
                "modality": "vio",
                "update_applied": False,
                "reason": "vio_reference_pose_stale",
                # §3.0.2 审计要求：VIO 不接受 NN bias（_handle_vio 全程 bias_applied=0.0），
                # 标记 "none" 与成功路径同口径，便于顶层审计区分「update_applied=False 也声明 write-port」。
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),
                "covariance_report": cov_report,
                "gate": {"passed": False, "rejected_by": "vio_reference_pose_stale"},
            }
        # ── VIO NIS 门控 + Huber 降权（gate 配置存在时启用） ──
        if self.cfg.get("gate") is not None:
            z_vio = build_vio_measurement(payload)
            z_hat, residual, H, reference_pose_array = compute_vio_residual(
                x_prev, z_vio, reference_pose=self._last_vio_reference_pose,
            )
            R_vio = _normalize_vio_covariance(effective_vio_cov)
            S = H @ self._covariance @ H.T + R_vio
            if not np.all(np.isfinite(S)):
                raise ValueError("VIO innovation covariance must be finite")
            try:
                S = _ensure_positive_definite_vio_innovation_covariance(S, name="VIO innovation covariance")
            except ValueError:
                self._last_vio_reference_pose = self._current_pose_reference()
                self._last_vio_reference_pose_timestamp = self._timestamp
                return {
                    "modality": "vio",
                    "update_applied": False,
                    "reason": "nonpositive_innovation_covariance",
                    "measurement_control": control_to_dict(control),
                    "covariance_report": cov_report,
                    "robust_covariance_report": None,
                    # §3.0.2 审计：VIO 不接受 NN bias，标记 "none"。
                    "bias_writeport": "none",
                    "gate": {"passed": False, "rejected_by": "nonpositive_innovation_covariance",
                             "nis": None, "mahalanobis_sq_threshold": self._nis_threshold("vio")},
                    "robust": None,
                }
            try:
                nis = float(residual.T @ np.linalg.solve(S, residual))
            except np.linalg.LinAlgError:
                self._last_vio_reference_pose = self._current_pose_reference()
                self._last_vio_reference_pose_timestamp = self._timestamp
                return {
                    "modality": "vio",
                    "update_applied": False,
                    "reason": "nonpositive_innovation_covariance",
                    "measurement_control": control_to_dict(control),
                    "covariance_report": cov_report,
                    "robust_covariance_report": None,
                    # §3.0.2 审计：VIO 不接受 NN bias。
                    "bias_writeport": "none",
                    "gate": {"passed": False, "rejected_by": "nonpositive_innovation_covariance",
                             "nis": None, "mahalanobis_sq_threshold": self._nis_threshold("vio")},
                    "robust": None,
                }
            if not math.isfinite(nis):
                self._last_vio_reference_pose = self._current_pose_reference()
                self._last_vio_reference_pose_timestamp = self._timestamp
                return {
                    "modality": "vio",
                    "update_applied": False,
                    "reason": "nonfinite_nis",
                    "measurement_control": control_to_dict(control),
                    "covariance_report": cov_report,
                    "robust_covariance_report": None,
                    # §3.0.2 审计：VIO 不接受 NN bias。
                    "bias_writeport": "none",
                    "gate": {"passed": False, "rejected_by": "nonfinite_nis",
                             "nis": nis, "mahalanobis_sq_threshold": self._nis_threshold("vio")},
                    "robust": None,
                }
            if nis > self._nis_threshold("vio"):
                self._last_vio_reference_pose = self._current_pose_reference()
                self._last_vio_reference_pose_timestamp = self._timestamp
                return {
                    "modality": "vio",
                    "update_applied": False,
                    "reason": "mahalanobis_sq",
                    "measurement_control": control_to_dict(control),
                    "covariance_report": cov_report,
                    "robust_covariance_report": None,
                    # §3.0.2 审计：VIO 不接受 NN bias。
                    "bias_writeport": "none",
                    "gate": {"passed": False, "rejected_by": "mahalanobis_sq",
                             "nis": nis, "mahalanobis_sq_threshold": self._nis_threshold("vio")},
                    "robust": None,
                }
            whitened = math.sqrt(max(nis, 0.0))
            robust_weight = self._huber_weight(whitened)
            covariance_scale = 1.0 / max(robust_weight, 1e-6)
            effective_vio_cov, robust_cov_report = build_effective_cov(
                effective_vio_cov, vio_scaling=covariance_scale,
            )
        else:
            robust_cov_report = None

        # §6.4 #18 紧耦合角色硬子要求对应代码层落点：
        # VIO 因子/更新挂增量残差（reference_pose=上一帧参考位姿 self._last_vio_reference_pose）
        # 与 UWB 距离残差并列（_handle_uwb 同帧调 apply_uwb_update 同口径），而非先视觉里程计
        # 积分成轨迹再松耦合（§6.4 #19）。三估计器同口径：EKF 本处 L1193 / RobustEKF
        # robust_ekf_core.py:694 / FGO fgo_core.py:2224。残差锚定上一帧参考位姿，
        # 通过 _update_from_vector 同步写回状态/协方差，并在 L1196 重置参考位姿为当前位姿。
        x_upd, P_upd, update_report = apply_vision_update(
            x_prev,  # 更新前状态。
            self._covariance,  # 更新前协方差。
            payload,  # 视觉事件负载。
            effective_vio_cov,  # 有效视觉协方差（可能已被 Huber 膨胀）。
            reference_pose=self._last_vio_reference_pose,  # 参考位姿。
        )
        self._update_from_vector(x_upd, P_upd)  # 把更新结果写回内部缓存。
        self._last_vio_reference_pose = self._current_pose_reference()  # 更新最近参考位姿。
        self._last_vio_reference_pose_timestamp = self._timestamp  # 记录参考位姿更新时间戳。
        _vio_update_detail = {k: v for k, v in update_report.items()
                              if k not in ("gate", "modality", "update_applied", "reason")}
        return {
            "modality": "vio",
            "update_applied": True,
            "reason": "vio_update_success",
            "measurement_control": control_to_dict(control),
            "covariance_report": cov_report,
            "robust_covariance_report": robust_cov_report,
            # §3.0.2 审计要求：VIO 不接受 NN bias（build_measurement_control VIO 分支
            # 强制 bias_applied=0.0，残差定义在原始增量 z 上）。显式标记 "none"，
            # 与 EKF UWB 路径 / Robust VIO / FGO VIO 同口径，便于顶层聚合审计。
            "bias_writeport": "none",
            "gate": {"passed": True, "rejected_by": None},
            **_vio_update_detail,
        }

    def step(self, event: Any) -> StateEstimate:
        """消费单个事件并返回当前状态估计。

        参数
        ----------
        event : Any
            输入事件，必须通过 :func:`validate_event` 协议校验。
            支持字典或带 ``to_dict()`` 方法的对象。

        返回
        -------
        StateEstimate
            当前状态估计对象，包含 state、covariance_diag、timestamp。

        异常
        ------
        各种异常
            事件校验失败、处理逻辑错误等异常会原样抛出；
            异常时 ``_timestamp``、``_state``、``_covariance``、
            ``_last_imu_timestamp``、``_last_vio_reference_pose`` 和
            ``_last_vio_reference_pose_timestamp`` 会回滚到上次值，
            ``_measurement_control`` 仍会被清空。

        注意
        -----
        - 控制参数（如有）只用一次，无论成功或异常都会在 ``finally`` 中清空。
        - 控制参数模态必须与当前事件模态匹配，不匹配时回退到默认控制，
          防止跨模态控制参数（如 UWB 的 bias_applied 用到 VIO 更新）污染。
        - 未识别的模态或缺少 payload 的事件会返回
          ``update_applied=False, reason="missing_payload"`` 的报告。
        """
        self.last_update_report = None  # 每次 step 前先清空旧报告。
        previous_timestamp = self._timestamp  # 保存旧时间戳，出错时可回滚。
        previous_state = dict(self._state)  # 保存旧状态快照，出错时可回滚。
        previous_covariance = self._covariance.copy()  # 保存旧协方差快照，出错时可回滚。
        previous_last_imu_timestamp = self._last_imu_timestamp  # 保存旧 IMU 时间戳，出错时可回滚。
        previous_last_vio_reference_pose = self._last_vio_reference_pose  # 保存旧 VIO 参考位姿，出错时可回滚。
        previous_last_vio_reference_pose_timestamp = self._last_vio_reference_pose_timestamp  # 保存旧 VIO 参考位姿时间戳，出错时可回滚。
        try:  # 整个事件处理放在 try 里，出错时回滚时间戳和状态。
            validate_event(event)  # 先做协议校验，确保事件结构合法。
            payload = event if isinstance(event, dict) else event.to_dict()  # 统一成字典负载，兼容对象和字典两种输入。
            self._timestamp = float(payload["t"])  # 写入当前事件时间戳。
            if payload["modality"] not in get_sensor_roles():  # 当前任务只允许 task contract 冻结的三种模态进入估计器主链。
                raise ValueError(
                    f"Unsupported estimator modality for current task contract: {payload['modality']!r}; "
                    f"expected one of {sorted(get_sensor_roles())}"
                )
            # 控制参数必须与当前事件模态匹配；不匹配时回退到默认控制，
            # 防止跨模态控制参数（如 UWB 的 bias_applied/noise_multiplier 用到 VIO 更新）污染估计器。
            if self._measurement_control is not None and self._measurement_control.modality == payload["modality"]:
                control = self._measurement_control  # 模态匹配，使用注入的控制参数。
            else:
                control = self._default_control(payload["modality"])  # 模态不匹配或未注入，回退到默认控制。
            x_prev = self._state_vector()  # 读取当前内部状态向量。

            if payload["modality"] == "imu" and payload["imu_payload"]:  # IMU 事件且有负载时走预测分支。
                report = self._handle_imu(payload, x_prev, control)  # 执行 IMU 预测并拿到报告。
            elif payload["modality"] == "uwb" and payload["uwb_payload"]:  # UWB 事件且有负载时走测距更新分支。
                report = self._handle_uwb(payload, x_prev, control)  # 执行 UWB 更新并拿到报告。
            elif payload["modality"] == "vio" and payload["vio_payload"]:  # VIO 事件且有负载时走视觉更新分支。
                report = self._handle_vio(payload, x_prev, control)  # 执行 VIO 更新并拿到报告。
            else:  # 未识别的模态或缺少负载，不做任何更新。
                report = {
                    "modality": payload["modality"],
                    "update_applied": False,
                    "reason": "missing_payload",
                    # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "none",
                    "gate": {"passed": False, "rejected_by": "missing_payload"},
                    "measurement_control": control_to_dict(control),
                }

            self.last_update_report = report  # 保存本次更新摘要。
            return self.get_state()  # 返回当前状态估计对象。
        except Exception:  # 事件处理过程中任何异常都要回滚时间戳和状态。
            self._timestamp = previous_timestamp  # 出错时回滚时间戳，保持状态一致性。
            self._state = previous_state  # 出错时回滚状态，防止部分修改残留。
            self._covariance = previous_covariance  # 出错时回滚协方差，防止部分修改残留。
            self._last_imu_timestamp = previous_last_imu_timestamp  # 出错时回滚 IMU 时间戳，防止部分修改残留。
            self._last_vio_reference_pose = previous_last_vio_reference_pose  # 出错时回滚 VIO 参考位姿，防止部分修改残留。
            self._last_vio_reference_pose_timestamp = previous_last_vio_reference_pose_timestamp  # 出错时回滚 VIO 参考位姿时间戳，防止部分修改残留。
            raise  # 把原始异常继续向上抛出，由调用方处理。
        finally:  # 无论成功还是异常，控制参数只用一次。
            self._measurement_control = None  # 控制参数只用一次。

    def step_joint(
        self,
        *,
        uwb_payloads: Sequence[Mapping[str, Any]],
        vio_payload: Mapping[str, Any] | None,
        timestamp: float,
        meta: Mapping[str, Any] | None = None,
        uwb_extra_biases: Sequence[float] | None = None,
    ) -> StateEstimate:
        """对同一时间戳的多个 UWB 锚点 + 单个 VIO 帧做一次紧耦合联合 EKF 更新。

    本方法是 ``step`` 的并行入口，专门处理 MATLAB 官方紧耦合 EKF 文献
    中的典型场景：在同一时刻，多个 UWB 锚点的测距 + 单次 VIO 帧间相对
    量测同时到达。此时把两路观测堆叠到一次更新中比依次分别调用单模态
    更新更准确，能正确刻画跨模态的耦合相关性。

    实现要点：
        1. 不走 ``step()`` 主 dispatch，直接调用联合更新函数
           :func:`run_joint_uwb_vio_update`，避免与已冻结的 IMU/UWB/VIO
           单事件路径相互污染。
        2. 接收 UWB 锚点列表 + VIO payload + 显式 timestamp + （可选）
           h 侧有界偏置修正 ``uwb_extra_biases``，做一次 Joseph 形式协方差
           更新后写回内部缓存。
        3. 不修改 EKF 状态维度，与现有 8 维状态完全兼容（前提指导 §1.1+§2.3
           全体同增 10 维硬合同由工厂/yaml/state_definition 层强制保障真实
           estimator 路径必传 10 维），对 protocol/task_contract、predict_step、
           单锚点 UWB / 单帧 VIO 测试都无任何破坏。

    参数
    ----------
    uwb_payloads : Sequence[Mapping[str, Any]]
        同一时刻到达的 N 个 UWB 事件负载列表，每个负载需含
        ``anchor_id``、``range``、``valid``（默认 True）、``quality``
        （默认 1.0）。空时退化为纯 VIO 更新。
    vio_payload : Mapping[str, Any] | None
        VIO 事件负载；None 退化为多锚点 UWB 更新。需含 ``dx``、``dy``、
        ``dyaw``，与 apply_vision_update 同口径。
    timestamp : float
        该联合观测对应的统一时间戳，会写入内部 ``_timestamp``。
    uwb_extra_biases : Sequence[float] | None
        各 UWB 锚点对应的 h 侧有界偏置修正（§3.0.2 / §3.1.2）。
        **由调用方（fusion_runner）从 NN intermediate.bias 经 clip_uwb_bias
        截断后逐锚点注入**，与单模态 ``_handle_uwb`` 路径同口径；
        长度必须与 ``uwb_payloads`` 一致。None 时按全 0 处理（保留与
        历史 ekf baseline 同口径），但 NN+EKF 主路径必须传入，否则
        NN bias 在联合路径上被悄悄丢弃，违反 §3.0 item5 / §3.0.2 写入口边界。

        返回
        -------
        StateEstimate
            当前状态估计对象，含更新后 state、covariance_diag、timestamp。

        异常
        ------
        ValueError
            - 联合创新协方差 S 非正定
            - VIO 测量或噪声不合规（透传自 joint 路径）
        TypeError
            与上下文一致。
        KeyError
            某个 UWB 负载 missing anchor_id；锚点解析失败等。

        注意
        -----
        - 不走门控配置（``gate_action`` / NIS / Huber），任何在 joint 路径
          上的门控应由调用方在外部按需过滤后再传入。
        - NN bias 通过 ``uwb_extra_biases`` 显式注入到 h(·) 侧（§3.0.2 / §3.1.2，
          与单模态 ``_handle_uwb`` 同口径）；联合路径不再静默丢弃 NN bias。
          UWB/VIO 的噪声取默认噪声并做 ``uwb_scaling``/``vio_scaling`` 缩放
          （与单模态路径同口径，但 NN bias 不通过 ``measurement_control`` 注入）。
        - 标记 ``last_update_report['modality'] == 'joint_uwb_vio'``，
          方便审计层区分联合更新与单模态更新。
        - 出错时回滚内部状态/协方差/时间戳/VIO 参考位姿，与 step() 同口径。
        """
        # 保存快照以便异常时回滚，与 step() 同口径。
        previous_timestamp = self._timestamp
        previous_state = dict(self._state)
        previous_covariance = self._covariance.copy()
        previous_last_vio_reference_pose = self._last_vio_reference_pose
        previous_last_vio_reference_pose_timestamp = self._last_vio_reference_pose_timestamp
        self.last_update_report = None  # 清空旧报告，准备新报告。
        try:
            # ── 1. 规整 UWB 锚点列表为 (anchor_pos, z, R) 列表。 ──
            uwb_anchors_resolved: list[tuple[float, float]] = []
            z_uwb_list: list[float] = []
            r_uwb_list: list[float] = []
            # 调用方提供的逐锚点 h 侧有界偏置修正（§3.0.2 / §3.1.2）。
            # fusion_runner 从 NN intermediate.bias 经 clip_uwb_bias 截断后逐锚点注入；
            # 与单模态 _handle_uwb 同口径，禁止联合路径悄悄丢弃 NN bias。
            uwb_extra_biases_resolved: list[float] = []
            if uwb_extra_biases is not None and len(uwb_extra_biases) != len(uwb_payloads or ()):
                raise ValueError(
                    "uwb_extra_biases length must match uwb_payloads, "
                    f"got {len(uwb_extra_biases)} vs {len(uwb_payloads or ())}"
                )

            base_uwb_noise = self._required_measurement_noise("uwb")  # 标准差
            base_uwb_noise = _square_std_to_var(base_uwb_noise)  # 转方差
            # UWB 噪声现在是标量（由 _handle_uwb 强约束标量化），对每个锚点
            # 用同一标量方差；若上游 measurement_noise.uwb 是矩阵/向量，按
            # 单锚点路径同口径拒绝。
            try:
                raw_uwb_noise = np.asarray(base_uwb_noise, dtype=float)
            except (TypeError, ValueError):
                raise ValueError(  # noqa: TRY004
                    f"effective UWB noise must be scalar, got {type(base_uwb_noise).__name__}"
                ) from None
            if raw_uwb_noise.ndim != 0:
                raise ValueError(f"effective UWB noise must be scalar, got shape {raw_uwb_noise.shape}")
            scalar_uwb_var = coerce_finite_scalar(
                raw_uwb_noise.reshape(-1)[0],
                name="effective UWB noise (joint path)",
            )

            for i, uwb_payload in enumerate(uwb_payloads or ()):
                if not isinstance(uwb_payload, Mapping):
                    raise TypeError(f"uwb_payloads[{i}] must be a mapping, got {type(uwb_payload).__name__}")
                # §11.3-d 同方法内一致：与 _handle_uwb L856-869 同口径，兼容 numpy.bool_(False)。
                # v10 audit 发现：旧 `is False` 漏掉 numpy.bool_(False)，与 _handle_uwb L860 不同口径；
                # 改用 is_bool_like + not bool 同口径覆盖 Python False 与 numpy.bool_(False)。
                uwb_valid_i = uwb_payload.get("valid", True)
                if is_bool_like(uwb_valid_i) and not bool(uwb_valid_i):
                    # valid=False 跳过此锚点，与 _handle_uwb 同口径。
                    continue
                # §11.1+§11.3-d v14 修复：v9-v13 漏审 step_joint 联合路径完全不走门控族。
                # 紧耦合路径的卡方/Huber 属 §12 紧耦合专筹（联合自由度与降权需重写 stacked
                # 残差路径），但 quality_floor / VIO quality<=0 / S<=0 jitter 是单模态路径
                # 的同口径协议级检查，必须在 joint 路径同源同口径，违 §11.3-d「方法内部
                # 串联顺序全员固定」+§11.5「禁止只救一方静默重置」。
                try:
                    uwb_quality_i = self._quality_value(uwb_payload)
                except (TypeError, ValueError):
                    # 规范化失败：bool/NaN/超界均由 _quality_value raise（与 _handle_vio
                    # v11 修复同口径）。joint 路径无单 VIO 参考位姿可重置，直接 raise
                    # 让 fusion_runner fallback 逐事件路径处理（与 _handle_vio raise 路径
                    # 经 fusion_runner L712 fallback 等价）。
                    raise
                if quality_below_floor(uwb_quality_i, self._quality_floor("uwb")):
                    # quality<uwb_floor 跳过此锚点，与 _handle_uwb L871-888 同口径。
                    # 不重置 VIO 参考位姿（联合路径 VIO 自有 reference_pose 检查）。
                    continue
                anchor_pos = self._resolve_anchor_position(uwb_payload["anchor_id"])
                # §3.0.2 / §3.1.2 / §12.2：联合路径不再对 raw 距离做 subtractive 改写
                # 或 max(0,·) 等单方截断；raw_range 直接作为 z（与单模态 _handle_uwb 同口径：
                # ekf_core.py:915 / robust_ekf_core.py:358 / fgo_core.py:1872），
                # bias 进 h(·)。负值/非有限值由下游 run_uwb_update 的 coerce_finite_scalar
                # 协议级数据校验拒绝（raise），不在联合路径单方 clamp 改写 raw 距离。
                raw_range_i = float(uwb_payload["range"])
                bias_h_i = float(uwb_extra_biases[i]) if uwb_extra_biases is not None else 0.0
                ax, ay = anchor_pos

                # 身份纪律（前提指导 §3.2 / §3 事件流细节）：禁止仅 step_joint 私有
                # nlos_sanity_gate 硬砍。大脉冲拒识必须走全员同一的 gate 卡方/质量门
                # （configs/models/ekf.yaml 与 robust_ekf.yaml 的 mahalanobis_sq），
                # 不得在 joint 路径再开单方法几何阈值通道。历史 nlos_sanity_gate 块已作废。

                uwb_anchors_resolved.append((ax, ay))
                z_uwb_list.append(raw_range_i)
                r_uwb_list.append(scalar_uwb_var)
                uwb_extra_biases_resolved.append(bias_h_i)

            # ── 2. 规整 VIO 路径：从 measurement_noise.vio 取默认噪声。 ──
            z_vio_arr = None
            R_vio_arr = None  # 由 vision_update_step._normalize_vio_covariance 来规整
            if vio_payload is not None:
                if not isinstance(vio_payload, Mapping):
                    raise TypeError(f"vio_payload must be a mapping, got {type(vio_payload).__name__}")
                # §11.1+§11.3-d v14 修复：v9-v13 漏审 step_joint 跳过 VIO quality 规范化。
                # 与 _handle_vio L1057-1075 同口径：quality<=0 是协议级（仿真 cycle 边界
                # 帧）检查，不依赖 quality_floor 配置即必须生效；规范化走 _quality_value
                # 同源（防止 bool/NaN/超界绕过）。
                try:
                    vio_quality = self._quality_value(vio_payload)
                except (TypeError, ValueError):
                    # 规范化失败：与 _handle_vio v11 修复同源 raise，但联合路径必须保护
                    # VIO 参考位姿。先重置再 raise，让 fusion_runner 走 fallback 单模态。
                    self._last_vio_reference_pose = self._current_pose_reference()
                    self._last_vio_reference_pose_timestamp = self._timestamp
                    raise
                if vio_quality is not None and float(vio_quality) <= 0.0:
                    # quality<=0：跳过 VIO 此次联合更新，重置参考位姿，与 _handle_vio
                    # L1064-1075 同口径。仅 UWB 路径仍可继续联合更新（保留 NLOS 帧时
                    # UWB 多锚点测距仍能贡献定位信息）。
                    self._last_vio_reference_pose = self._current_pose_reference()
                    self._last_vio_reference_pose_timestamp = self._timestamp
                    vio_payload = None  # 标记跳过 VIO，仅走纯 UWB 联合更新
                elif quality_below_floor(vio_quality, self._quality_floor("vio")):
                    # quality<vio_floor：同 _handle_vio L1079-1096 同口径拒绝 VIO 部分。
                    # 联合路径不重置参考位姿（与 _handle_vio L1080 仅在 quality_floor
                    # 配置存在时重置一致；此处仅 UWB 联合路径走，VIO 参考位姿不动
                    # 让下次 VIO 单事件路径自己处理）。
                    self._last_vio_reference_pose = self._current_pose_reference()
                    self._last_vio_reference_pose_timestamp = self._timestamp
                    vio_payload = None
                # 提取 VIO 标准差并转方差，与 _handle_vio 同口径。
                base_vio_noise = self._required_measurement_noise("vio")
                base_vio_noise = _square_std_to_var(base_vio_noise)
                # build_vio_measurement 期望一个完整 VIO 事件（含 modality + vio_payload + dt + meta 含 scene_id/seq_id），
                # 此处把仅 payload 包装成 minimal synthetic event，与 _handle_vio 同口径。
                # meta 由调用方传入；缺省时使用占位 meta 满足 validate_event 协议字段。
                # v14 §11.3-d 修复：上方可能把 vio_payload 改写为 None（quality<=0
                # 或 quality<floor），此 if 守门避免对 None 调用 build_vio_measurement。
                if vio_payload is not None:
                    meta_for_event = dict(meta) if meta is not None else {
                        "scene_id": "S(J,U,V,0,K3)",  # 五轴档位协议 K 轴仅 K0/K1/K3
                        "seq_id": "joint_step",
                    }
                    synthetic_vio_event = {
                        "modality": "vio",
                        "t": float(timestamp),
                        "dt": 0.0,  # 联合路径已显式给出 timestamp；dt 仅满足 validate_event 协议字段。
                        "meta": meta_for_event,
                        "vio_payload": dict(vio_payload),
                    }
                    z_vio_arr = build_vio_measurement(synthetic_vio_event)
                    R_vio_arr = base_vio_noise  # vision_update_step._normalize_vio_covariance 支持标量/3 元组/3x3 矩阵等多种承载。


            # ── 3. 调用联合更新函数（uwb_update_step.run_joint_uwb_vio_update）。 ──
            # 空观测短路：UWB 锚点空且 VIO None 时，与 step() 单模态 reject 路径同口径
            # 返回 no-op 报告。不抛 ValueError、不推进时间戳、不动 state/covariance，
            # 让调用方拿到 update_applied=False 的明确语义。§5.1 同一预测模型护栏：
            # 此处不引入单方特权（仍对外宣告 modality=joint_uwb_vio），仅按合同短路。
            if not uwb_anchors_resolved and vio_payload is None:
                self.last_update_report = {
                    "modality": "joint_uwb_vio",
                    "update_applied": False,
                    "reason": "no_joint_observation",
                    "gate": {"passed": False, "rejected_by": "no_joint_observation"},
                    "uwb_anchor_count": 0,
                    "has_vio": False,
                    "covariance_report": None,
                    "robust_covariance_report": None,
                    "measurement_control": None,
                }
                return self.get_state()
            from liquidloc.estimators.uwb_update_step import run_joint_uwb_vio_update  # 局部导入，避免顶层循环依赖。
            x_prev = self._state_vector()  # 当前状态向量。
            x_upd, P_upd, joint_info = run_joint_uwb_vio_update(
                x_prev,
                self._covariance,
                uwb_anchors=uwb_anchors_resolved,
                z_uwb=z_uwb_list,
                R_uwb=r_uwb_list,
                z_vio=z_vio_arr if z_vio_arr is not None else None,
                R_vio=R_vio_arr if R_vio_arr is not None else None,
                vio_reference_pose=self._last_vio_reference_pose,
                uwb_extra_biases=uwb_extra_biases_resolved,
            )
            self._update_from_vector(x_upd, P_upd)  # 写回内部缓存。
            self._timestamp = float(timestamp)  # 推进时间戳。
            if vio_payload is not None:
                # VIO 参考位姿在每次成功联合更新后刷新，与 _handle_vio 同口径。
                self._last_vio_reference_pose = self._current_pose_reference()
                self._last_vio_reference_pose_timestamp = self._timestamp

            # 把 joint_info 整理为对外报告（与单模态报告字段对齐）。
            self.last_update_report = {
                "modality": "joint_uwb_vio",
                "update_applied": True,
                "reason": "joint_uwb_vio_update_success",
                "gate": {"passed": True, "rejected_by": None},
                "uwb_anchor_count": len(uwb_anchors_resolved),
                "vio_included": z_vio_arr is not None,
                # §3.0.2 审计：联合路径与单模态写入口同口径区分 h 侧 vs z 侧；
                # bias 进 h(·)，残差定义在原始 z 上。逐锚点记录便于审计回溯。
                "uwb_extra_biases": list(uwb_extra_biases_resolved),
                "bias_writeport": "h_side",  # §3.0.2 显式标记 bias 进 h(·)。
                **joint_info,
            }
            return self.get_state()
        except Exception:
            # 异常时回滚，与 step() 同口径。
            self._timestamp = previous_timestamp
            self._state = previous_state
            self._covariance = previous_covariance
            self._last_vio_reference_pose = previous_last_vio_reference_pose
            self._last_vio_reference_pose_timestamp = previous_last_vio_reference_pose_timestamp
            raise
        finally:
            # 与 step() 同口径：控制参数在联合路径上一次性用完；这里 joint
            # 路径已通过 uwb_extra_biases 显式消费 NN bias（§3.0.2），与
            # 单模态路径同口径。仍清空 _measurement_control 保留对外语义一致。
            self._measurement_control = None

    def get_state(self) -> StateEstimate:
        """把内部缓存打包成对外统一状态对象。

        返回
        -------
        StateEstimate
            包含 state（状态字典副本）、covariance_diag（协方差对角线）、
            timestamp（最近时间戳）的状态估计对象。
        """
        covariance_diag = [max(float(value), 0.0) for value in np.diag(self._covariance)]  # 摘要层只输出非负方差，避免数值毛刺污染消费者。
        return StateEstimate(
            state=dict(self._state),  # 当前状态字典副本。
            covariance_diag=covariance_diag,  # 协方差对角线摘要。
            timestamp=self._timestamp,  # 最近时间戳。
        )
