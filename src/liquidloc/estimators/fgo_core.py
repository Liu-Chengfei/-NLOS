"""最小滑窗 FGO（Factor Graph Optimization）基线。

这个模块把 EKF 预测、UWB 因子和 VIO 因子拼成一个基于 NumPy 的高斯牛顿求解器。
它的职责是维护滑窗、累积因子、构造正规方程，并把最新优化结果写回状态缓存。

核心概念
--------
- **滑窗（Sliding Window）**：只保留最近 N 个位姿节点，超出时裁剪最老节点。
  每个节点存储完整状态、位姿先验、运动增量和挂载的 UWB/VIO 因子。
- **因子（Factor）**：UWB 测距因子和 VIO 相对位姿因子分别挂载到对应时间节点上，
  每个因子携带量测值、噪声协方差和参考位姿信息。
- **高斯牛顿求解**：将所有因子线性化后累加成正规方程 (Hδ = g)，
  迭代求解位姿增量直到收敛或达到最大迭代轮数。
- **鲁棒权重 (铁律 10 已禁用)**: 历史上 FGO 通过 Huber 规则对大残差因子降权，
  但铁律 10 要求 FGO 必须裸跑 (无鲁棒核)，因此 ``_huber_weight`` 强制返回 1.0。
  本字段保留是为了向后兼容旧 YAML 配置和文档语义，不再实际生效。
- **门控 (铁律 10 已禁用)**: 历史上质量门控和 NIS (归一化创新平方) 门控
  在因子加入窗口前过滤低质量和异常残差，铁律 10 要求 FGO 裸跑，
  因此 ``_quality_floor`` 强制返回 0.0、``_nis_threshold`` 强制返回 ``inf``，
  任何量测都进入因子图。本字段保留是为了向后兼容旧 YAML 配置和报告语义。

与其他模块的关系
----------------
- 继承 :class:`EKFCore`，复用其状态管理、协方差维护和锚点解析逻辑，
  但覆写了 ``_handle_imu``、``_handle_uwb``、``_handle_vio`` 三个核心方法。
- 调用 :mod:`predict_step` 执行 IMU 预测步，与 EKF 共享同一预测逻辑。
- 调用 :mod:`uwb_update_step` 和 :mod:`vision_update_step` 中的几何模型
  （雅可比、预测距离、VIO 量测构造），但不直接调用它们的 EKF 更新函数。
- 调用 :mod:`fusion.covariance_adapter` 做噪声缩放，与 EKF/RobustEKF 共享
  同一套协方差适配逻辑。

模块内容
--------
- :func:`_reject_boolean_like` —— 递归拒绝布尔型噪声配置
- :func:`_coerce_positive_int` —— 规整正整数参数
- :func:`_coerce_positive_scalar` —— 规整有限正标量
- :func:`_square_std_to_var` —— 标准差转方差
- :func:`_normalize_vio_covariance` —— VIO 噪声规整成 3×3 协方差矩阵
- :func:`_coerce_uwb_noise_scalar` —— UWB 噪声规整成标量方差
- :func:`_quality_below_floor` —— 质量门槛判断
- :func:`_rotation_world_to_reference` —— 世界系到参考系旋转矩阵
- :func:`_linearize_vio_factor` —— 线性化单个 VIO 因子
- :func:`_noise_cache_key` —— 协方差缓存键生成
- :class:`FGOCore` —— 滑窗 FGO 核心估计器
"""

from __future__ import annotations  # 允许后面的类型注解延迟求值，避免前向引用报错。

import copy  # 复制因子和窗口条目时要保留原对象的独立副本。
import logging  # 求解失败时记录非预期异常日志。
import math  # 用来做三角函数、平方根和有限性检查。
from collections.abc import Mapping, Sequence  # 用来判断配置和噪声结构是不是映射，以及列表序列。
from typing import Any  # 给窗口条目和约束字典保留灵活类型。

import numpy as np  # 用来做向量、矩阵和线性代数。

_logger = logging.getLogger(__name__)  # 模块级日志器，用于记录求解失败等非预期异常。

from liquidloc.common.angle_utils import angle_delta_rad, wrap_angle_rad  # 处理角度差和角度归一化。
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS  # 桥接层业务阈值常量
from liquidloc.common.constants import DEFAULT_THRESHOLDS  # 全局默认阈值表（质量上下界单源真相）。
from liquidloc.common.constants import QUALITY_FLOOR_EPSILON  # 质量门槛比较浮点容差
from liquidloc.common.constants import VIO_MEASUREMENT_ITEMS  # VIO 测量项常量（单源真相）。
from liquidloc.common.constants import VIO_REF_POSE_STALE_SECONDS  # VIO 参考位姿过时阈值。
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_integer, quality_below_floor  # 集中判断 bool / np.bool_ / 整数类型，以及有限浮点转换；质量门槛共享入口。
from liquidloc.estimators.ekf_core import EKFCore  # 复用基础 EKF 状态管理。
from liquidloc.estimators.shared import build_controlled_measurement_cov, control_to_dict  # 共享的协方差控制和报告转换函数。
from liquidloc.estimators.predict_step import run_predict_step  # 复用 IMU 预测步。
from liquidloc.estimators.state_definition import state_items  # 复用冻结的状态顺序。
# §2 细节 23 五方法同模型——FGO 节点 3 维但 self._state 含 uwb_clock_bias / vio_scale
# （继承 EKFCore 10 维状态向量）。用旁路 EKF 让这两项在 UWB / VIO 更新末尾被辨识，
# 与 EKF 族（ekf_core / robust_ekf_core 经 run_uwb_update / run_vio_update 路径）
# 保持「同在线辨识」对等。索引与 uwb_update_step.py / vision_update_step.py 同口径。
_FGO_IDX = {name: i for i, name in enumerate(state_items)}  # 状态名→索引映射。
_FGO_IDX_UWB_CLOCK_BIAS = _FGO_IDX["uwb_clock_bias"]  # uwb_clock_bias 索引。
_FGO_IDX_VIO_SCALE = _FGO_IDX["vio_scale"]  # vio_scale 索引。
# §10.3 偷懒审视固化：spec 第 1663-1666 行强制 τ_filt / α / β / nominal_event_rate_hz
# 在协议层显式落字、不得依赖代码默认值静默回退。这里用一个独立哨兵区分「配置缺键」与
# 「合法 None」，与同模块 _coerce_positive_scalar 的 default 路径解耦，避免默认值兜底
# 把 spec 的「必须可复述」要求洗成「代码凑出了」——属被禁偷懒式。
_MISSING = object()  # 配置缺键哨兵，仅供本文件 §10.3 强制显式写死纪律使用。
from liquidloc.estimators.uwb_update_step import build_uwb_jacobian, predict_range  # 复用 UWB 几何模型。
from liquidloc.estimators.vision_update_step import (  # 复用 VIO 量测、残差和创新协方差校验。
    _ensure_positive_definite_vio_innovation_covariance,
    build_vio_measurement,
    compute_vio_residual,
)
from liquidloc.common.covariance_utils import build_effective_cov  # 复用噪声缩放适配器。


def _reject_boolean_like(value, *, path: str) -> None:
    """递归拒绝布尔型噪声或配置项。

    布尔值常常是误写成开关标志（如 ``True``/``False``）而非数值噪声，
    如果混入噪声配置会导致协方差计算静默出错，因此必须在入口处拦截。

    参数
    ----------
    value : Any
        待检查的噪声配置项，支持 Mapping、ndarray、list/tuple 等。
    path : str
        错误信息中的配置路径，便于定位问题。

    异常
    ------
    TypeError
        当发现布尔值（bool、np.bool_）或布尔数组时抛出。
    """
    if is_bool_like(value):  # 叶子节点如果是布尔值就直接拒绝。
        raise TypeError(f"{path} must not be boolean-like")  # 布尔型不能冒充数值噪声。
    if isinstance(value, Mapping):  # 映射类型要逐层继续检查。
        for key, nested_value in value.items():  # 每个键值都要继续向下递归。
            _reject_boolean_like(nested_value, path=f"{path}.{key}")  # 递归检查更深层嵌套值，避免布尔伪值混进配置树。
        return  # 映射已经检查完毕。
    if isinstance(value, np.ndarray):  # 数组需要按 dtype 再分支检查。
        if value.dtype.kind == "b":  # 布尔数组也不能混进来。
            raise TypeError(f"{path} must not be boolean-like")  # 直接拒绝布尔数组。
        if value.dtype == object:  # object 数组要继续展开元素。
            for index, nested_value in np.ndenumerate(value):  # 对每个元素递归检查。
                index_suffix = "".join(f"[{item}]" for item in index)  # 把多维数组索引拼成路径后缀。
                _reject_boolean_like(nested_value, path=f"{path}{index_suffix}")  # 对 object 数组元素继续递归检查。
        return  # 数组已经检查完毕。
    if isinstance(value, (list, tuple)):  # 列表和元组也要逐项检查。
        for index, nested_value in enumerate(value):  # 每个位置都继续递归。
            _reject_boolean_like(nested_value, path=f"{path}[{index}]")  # 逐项检查列表和元组里的每个元素。
        return  # 列表和元组已经检查完毕，与 Mapping/ndarray 分支保持一致。


def _coerce_positive_int(value, *, path: str) -> int:
    """把输入规整成正整数。

    用于校验滑窗大小、最大迭代轮数等必须为正整数的配置项。

    参数
    ----------
    value : Any
        待规整的输入值，必须是整数类型且严格大于 0。
    path : str
        错误信息中的配置路径，便于定位问题。

    返回
    -------
    int
        规整后的正整数。

    异常
    ------
    TypeError
        当输入为布尔型或非整数类型时抛出。
    ValueError
        当输入为 0 或负数时抛出。
    """
    if not is_integer(value):  # 非整数或布尔值都不行。
        raise TypeError(f"{path} must be a positive integer, got {type(value).__name__}: {value!r}")  # 这里必须是正整数。
    int_value = int(value)  # 先转成 Python int。
    if int_value <= 0:  # 0 和负数都不接受。
        raise ValueError(f"{path} must be positive, got {int_value}")  # 参数必须严格大于 0。
    return int_value  # 返回规整后的正整数。


def _coerce_positive_scalar(value: Any, *, path: str) -> float:  # 把输入规整成有限正标量。
    """把输入规整成有限正标量。

    参数
    ----------
    value : Any
        待规整的输入值，只接受有限正实数标量。
    path : str
        错误信息中使用的配置路径，便于定位问题。

    返回
    -------
    float
        规整后的正浮点数。

    异常
    ------
    TypeError
        当输入为布尔型、复数型或非数值标量时抛出。
    ValueError
        当输入为非正数或非有限值时抛出。

    注意
    -----
    复数类型（包括 ``complex`` 和 ``np.complexfloating``）被显式拒绝，
    因为噪声方差不能是复数。
    """
    if isinstance(value, np.ndarray):  # 显式处理 ndarray，仅接受 0 维标量，防止数组穿透到标量计算中。
        if value.shape != ():  # 非 0 维数组不是标量。
            raise TypeError(f"{path} must be a scalar numeric value")  # 这里不能是向量或矩阵。
        value = value.item()  # 取出 Python 标量。
    # 委托中心入口：bool/complex/有限性校验 + 严格正值（> 0），与 uwb_update_step._coerce_scalar 口径对齐。
    return coerce_finite_scalar(value, name=path, min_value=0.0, inclusive=False)


def _square_std_to_var(noise_spec: Any) -> Any:
    """将测量噪声配置从标准差转换为方差。

    配置文件中 measurement_noise 的语义是标准差（与 process_noise 一致），
    但底层更新函数期望接收方差。此函数递归地对所有数值节点做平方。

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


def _normalize_vio_covariance(R_vio: Any) -> np.ndarray:  # 把 VIO 噪声配置规整成 3x3 协方差矩阵。
    """把 VIO 噪声配置规整成 3x3 协方差矩阵。

    参数
    ----------
    R_vio : Any
        VIO 噪声配置，支持以下形式：

        - ``dict`` — 必须包含 ``pos/yaw`` 或 ``dx/dy/dyaw`` 键；
        - 标量 — 三维同方差的正实数；
        - 长度 3 的一维数组 — 对角线方差；
        - 3x3 矩阵 — 完整协方差矩阵。

    返回
    -------
    np.ndarray
        形状为 ``(3, 3)`` 的协方差矩阵。

    异常
    ------
    TypeError
        当输入为布尔型或复数型时抛出。
    KeyError
        当字典键不满足 ``pos/yaw`` 或 ``dx/dy/dyaw`` 组合时抛出。
    ValueError
        当输入值非正、非有限或形状不正确时抛出。

    注意
    -----
    复数类型（包括 ``complex`` 和 ``np.complexfloating``）被显式拒绝，
    因为协方差矩阵必须是实对称正定的。
    """
    _reject_boolean_like(R_vio, path="R_vio")  # 先拒绝布尔型配置。
    if isinstance(R_vio, complex):  # Python 原生复数不能冒充实数噪声。
        raise TypeError("R_vio must not be complex")  # 复数不是合法的实数噪声。
    if isinstance(R_vio, Mapping):  # 字典配置支持不同命名风格。
        has_pos_yaw = all(key in R_vio for key in ("pos", "yaw"))
        has_dx_dy_dyaw = all(key in R_vio for key in VIO_MEASUREMENT_ITEMS)
        if has_pos_yaw and has_dx_dy_dyaw:  # 两种键同时存在，语义歧义，必须明确指定一种。
            raise ValueError("R_vio mapping must not mix pos/yaw and dx/dy/dyaw keys; choose one naming convention")  # 不允许混用。
        if has_pos_yaw:  # pos/yaw 代表位置与航向噪声。
            diag_values = [  # 按 x/x/yaw 的顺序组装对角线值。
                _coerce_positive_scalar(R_vio["pos"], path='R_vio["pos"]'),  # x 方向位置噪声。
                _coerce_positive_scalar(R_vio["pos"], path='R_vio["pos"]'),  # y 方向位置噪声，和 x 共用同一量级。
                _coerce_positive_scalar(R_vio["yaw"], path='R_vio["yaw"]'),  # 航向噪声。
            ]  # pos/yaw 形式的对角线到这里结束。
        elif has_dx_dy_dyaw:  # dx/dy/dyaw 代表三个分量的独立噪声，复用已计算标志位，避免与 VIO_MEASUREMENT_ITEMS 漂移。
            diag_values = [  # 按 dx/dy/dyaw 的顺序组装对角线值。
                _coerce_positive_scalar(R_vio["dx"], path='R_vio["dx"]'),  # x 增量噪声。
                _coerce_positive_scalar(R_vio["dy"], path='R_vio["dy"]'),  # y 增量噪声。
                _coerce_positive_scalar(R_vio["dyaw"], path='R_vio["dyaw"]'),  # 航向增量噪声。
            ]  # dx/dy/dyaw 形式的对角线到这里结束。
        else:  # 其他键组合都不接受。
            raise KeyError("R_vio mapping must provide either pos/yaw or dx/dy/dyaw noise entries")  # 必须提供完整噪声项。
        return np.diag(diag_values)  # 把对角线噪声转成矩阵。

    cov = np.asarray(R_vio, dtype=float)  # 把剩余输入统一转成数组。
    if cov.ndim == 0:  # 标量时表示三维各自同方差。
        scalar = coerce_finite_scalar(float(cov), name="R_vio scalar", min_value=0.0, inclusive=False)  # 标量必须有限且为正。
        return np.eye(3, dtype=float) * scalar  # 扩成 3x3 对角矩阵。
    if cov.shape == (3,):  # 长度 3 的向量表示对角线。
        if not np.all(np.isfinite(cov)) or np.any(cov <= 0.0):  # 每个分量都要有限且为正。
            raise ValueError("R_vio diagonal entries must be positive")  # 对角线不能有非正值。
        return np.diag(cov)  # 转成对角协方差矩阵。
    if cov.shape != (3, 3):  # 其他形状不接受。
        raise ValueError(f"R_vio must be shape (3, 3), got {cov.shape}")  # 形状必须严格匹配。
    if not np.all(np.isfinite(cov)):  # 整个矩阵都必须是有限数。
        raise ValueError("R_vio must contain finite values")  # 不能有 inf 或 nan。
    if np.any(np.diag(cov) <= 0.0):  # 对角线不能非正。
        raise ValueError("R_vio diagonal entries must be positive")  # 对角线噪声必须为正。
    return cov  # 已经是合法协方差矩阵，直接返回。


def _coerce_uwb_noise_scalar(effective_noise, *, path: str = "effective UWB noise") -> float:
    """把 UWB 有效噪声规整成标量方差。

    FGO 的 UWB 因子只接受标量噪声（一维测距），因此需要把经过控制缩放
    和鲁棒缩放后的有效噪声强制规整为标量。

    参数
    ----------
    effective_noise : Any
        经过缩放后的 UWB 有效噪声，必须是标量。
    path : str
        错误信息中使用的配置路径，便于定位问题，默认 ``effective_uwb_noise``。

    返回
    -------
    float
        规整后的标量方差，严格为正。

    异常
    ------
    TypeError
        当输入为布尔型、复数型或非数值标量时抛出。
    ValueError
        当输入非标量、非有限或非正时抛出。

    注意
    -----
    复数类型（包括 ``complex`` 和 ``np.complexfloating``）被显式拒绝，
    因为噪声方差不能是复数；若直接 ``np.asarray(..., dtype=float)`` 会
    静默丢弃虚部（仅产生 ``ComplexWarning``），必须先在不带 dtype 的
    数组视图上检查 ``complexfloating`` 再转 float。
    """
    _reject_boolean_like(effective_noise, path=path)  # 先拒绝布尔型伪值。
    if isinstance(effective_noise, complex):  # Python 原生复数不能冒充实数噪声。
        raise TypeError(f"{path} must not be complex, got {effective_noise!r}")  # 复数不是合法的实数噪声。
    raw_array = np.asarray(effective_noise)  # 先不带 dtype 转数组，便于检查复数和 dtype。
    if isinstance(raw_array, np.complexfloating):  # NumPy 复数标量同样拒绝，避免静默丢虚部。
        raise TypeError(f"{path} must not be complex, got {effective_noise!r}")  # np.complex128 等也不合法。
    if raw_array.ndim != 0:  # 这里只允许标量。
        raise ValueError(f"{path} must be scalar, got shape {raw_array.shape}")  # 向量或矩阵都不行。
    scalar_value = raw_array.item()  # 取出 Python 标量，便于做类型检查。
    if not isinstance(scalar_value, (int, float, np.integer, np.floating)):  # 只接受数值类型，str/None 等在此拦截。
        raise TypeError(f"{path} must be numeric, got {type(scalar_value).__name__}: {scalar_value!r}")  # 不是数值就报错。
    # 委托 coerce_finite_scalar：float 转换、OverflowError→ValueError、有限性、严格正值（> 0）一次性完成。
    return coerce_finite_scalar(scalar_value, name=path, min_value=0.0, inclusive=False)


def _quality_below_floor(quality: float, floor: float) -> bool:
    """判断量测质量是否低于门槛。委托到 common.validation.quality_below_floor 共享入口。"""
    return quality_below_floor(quality, floor, epsilon=QUALITY_FLOOR_EPSILON)


def _rotation_world_to_reference(reference_yaw: float) -> np.ndarray:
    """构造从世界坐标系到参考航向局部坐标系的 2×2 旋转矩阵。

    该旋转将世界坐标系下的平移向量转换到参考航向定义的局部坐标系中，
    即 R(-yaw_ref) @ [dx_world, dy_world]^T = [dx_local, dy_local]^T
    （等价于 R(yaw_ref)^T @ ...，其中 R(θ)=[[cosθ,-sinθ],[sinθ,cosθ]]
    为标准逆时针旋转矩阵；世界系到局部系需旋转 -yaw_ref）。
    这个旋转矩阵在 VIO 因子线性化时被反复使用。

    参数
    ----------
    reference_yaw : float
        参考航向角（弧度）。

    返回
    -------
    np.ndarray
        形状 ``(2, 2)`` 的旋转矩阵。
    """
    cos_yaw = math.cos(reference_yaw)  # 预先算余弦。
    sin_yaw = math.sin(reference_yaw)  # 预先算正弦。
    return np.asarray(  # 返回 2x2 旋转矩阵。
        [  # 2x2 旋转矩阵的两行。
            [cos_yaw, sin_yaw],  # 第一行把世界系 x/y 投到参考系。
            [-sin_yaw, cos_yaw],  # 第二行是正交补方向。
        ],  # 旋转矩阵行列表结束。
        dtype=float,  # 明确使用浮点数。
    )  # 旋转矩阵构造结束。


def _linearize_vio_factor(  # 线性化单个 VIO 因子并返回量测与雅可比。
    current_pose: np.ndarray,  # 当前位姿。
    reference_pose: np.ndarray,  # 参考位姿。
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对单个 VIO 因子做线性化，返回预测量测和两个雅可比。

    VIO 因子的量测模型为：将当前位姿相对参考位姿的增量从世界系
    旋转到参考航向定义的局部坐标系，得到 ``[dx_local, dy_local, dyaw]``。
    本函数同时计算量测残差对当前位姿和参考位姿的雅可比，供正规方程累加使用。

    参数
    ----------
    current_pose : np.ndarray
        当前位姿向量 ``[px, py, yaw]``。
    reference_pose : np.ndarray
        参考位姿向量 ``[px, py, yaw]``。

    返回
    -------
    local_measurement : np.ndarray
        预测的局部 VIO 量测 ``[dx_local, dy_local, dyaw]``。
    current_jacobian : np.ndarray
        残差对当前位姿的雅可比，形状 ``(3, 3)``。
    reference_jacobian : np.ndarray
        残差对参考位姿的雅可比，形状 ``(3, 3)``。
    """
    delta_world = np.asarray(  # 先算世界系下的位置差。
        [  # 世界系下的平移差。
            float(current_pose[0]) - float(reference_pose[0]),  # x 方向位移差。
            float(current_pose[1]) - float(reference_pose[1]),  # y 方向位移差。
        ],  # 平移差向量结束。
        dtype=float,  # 使用浮点数组。
    )  # 世界位移差构造结束。
    rotation_world_to_reference = _rotation_world_to_reference(float(reference_pose[2]))  # 参考航向决定旋转方向。
    local_translation = rotation_world_to_reference @ delta_world  # 把世界差转到参考系局部坐标。
    local_measurement = np.asarray(  # VIO 量测包含平移和航向差。
        [  # 局部 VIO 量测的三个分量。
            float(local_translation[0]),  # 局部系下的 x 平移。
            float(local_translation[1]),  # 局部系下的 y 平移。
            angle_delta_rad(float(current_pose[2]), float(reference_pose[2])),  # 当前航向相对参考航向的角差。
        ],  # 局部量测向量结束。
        dtype=float,  # 使用浮点数组。
    )  # 局部量测构造结束。
    delta_x = float(delta_world[0])  # x 差值。
    delta_y = float(delta_world[1])  # y 差值。
    reference_yaw = float(reference_pose[2])  # 参考航向。
    cos_yaw = math.cos(reference_yaw)  # 预先算余弦。
    sin_yaw = math.sin(reference_yaw)  # 预先算正弦。
    d_local_d_reference_yaw = np.asarray(  # 残差位置行对参考航向的偏导。
        [  # 残差位置行对参考航向的偏导。
            (sin_yaw * delta_x) - (cos_yaw * delta_y),  # 残差局部 x 行对参考航向的偏导。
            (cos_yaw * delta_x) + (sin_yaw * delta_y),  # 残差局部 y 行对参考航向的偏导。
        ],  # 导数向量结束。
        dtype=float,  # 使用浮点数组。
    )  # 导数构造结束。
    current_jacobian = np.asarray(  # 残差对当前位姿的雅可比。
        [  # 残差对当前位姿的雅可比。
            [-rotation_world_to_reference[0, 0], -rotation_world_to_reference[0, 1], 0.0],  # 残差位置行对当前 x/y 的偏导。
            [-rotation_world_to_reference[1, 0], -rotation_world_to_reference[1, 1], 0.0],  # 残差位置行对当前 x/y 的偏导。
            [0.0, 0.0, -1.0],  # 残差航向行对当前 yaw 的偏导。
        ],  # 当前位姿雅可比结束。
        dtype=float,  # 使用浮点数组。
    )  # 当前位姿雅可比构造结束。
    reference_jacobian = np.asarray(  # 残差对参考位姿的雅可比。
        [  # 残差对参考位姿的雅可比。
            [rotation_world_to_reference[0, 0], rotation_world_to_reference[0, 1], d_local_d_reference_yaw[0]],  # 残差位置行对参考 x/y/yaw 的偏导。
            [rotation_world_to_reference[1, 0], rotation_world_to_reference[1, 1], d_local_d_reference_yaw[1]],  # 残差位置行对参考 x/y/yaw 的偏导。
            [0.0, 0.0, 1.0],  # 残差航向行对参考 yaw 的偏导。
        ],  # 参考位姿雅可比结束。
        dtype=float,  # 使用浮点数组。
    )  # 参考位姿雅可比构造结束。
    return local_measurement, current_jacobian, reference_jacobian  # 返回局部量测和两个雅可比。


def _noise_cache_key(covariance: Any) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """把协方差规整成可哈希的缓存键。

    因子的平方信息矩阵只依赖噪声协方差和权重，同一噪声配置下
    多次调用时可以复用缓存结果，避免重复做 Cholesky 分解。

    参数
    ----------
    covariance : Any
        噪声协方差，可以是标量、向量或矩阵。

    返回
    -------
    tuple[tuple[int, ...], tuple[float, ...]]
        由形状和扁平值组成的可哈希键。
    """
    covariance_array = np.asarray(covariance, dtype=float)  # 统一成数组。
    return covariance_array.shape, tuple(float(value) for value in covariance_array.reshape(-1))  # 形状加扁平值组成缓存键。


class FGOCore(EKFCore):
    """基于滑动窗口的因子图优化（FGO）核心估计器。

    本类继承 :class:`EKFCore`，复用其状态管理、协方差维护和锚点解析逻辑，
    但用滑窗高斯牛顿求解替代了逐帧 EKF 更新。核心流程为：

    1. IMU 事件触发预测步，将运动增量追加到滑窗末尾；
    2. UWB/VIO 事件先过门控（质量 + NIS），通过后构造因子并追加到当前节点；
    3. 每追加一个因子后立刻对整个滑窗做一轮高斯牛顿优化；
    4. 优化完成后将最新节点的位姿和协方差写回内部缓存。

    与 EKF 的关键区别：
    - EKF 逐帧更新，FGO 在滑窗内联合优化所有位姿；
    - FGO 的 VIO 因子可以引用窗口内任意历史节点作为参考位姿；
    - FGO 通过 Huber 鲁棒权重在因子层面降权，而非在 EKF 增益层面。

    配置字段
    ----------
    window_size : int
        滑窗长度，默认 20。
    optimizer : dict
        优化器配置，包含 ``max_iters``（最大迭代轮数，默认 10）。
    factor_weights : dict
        因子权重配置，包含 ``imu``、``uwb``、``vio`` 三个正标量权重。
    gate : dict
        门控配置，包含 ``quality_floor`` 和 ``mahalanobis_sq``。
    robust_weight : dict
        鲁棒权重配置，包含 ``type``（目前仅支持 ``"huber"``）和 ``delta``。
    """
    _POSE_KEYS = ("px", "py", "yaw")  # 滑窗里保留的位姿状态顺序。

    def __init__(self, init_cfg: dict | None = None) -> None:
        """初始化滑窗大小、优化器参数和权重配置。

        参数
        ----------
        init_cfg : dict | None
            初始化配置字典，除标准 EKF 配置外还支持
            ``window_size``、``optimizer``、``factor_weights``、
            ``gate`` 和 ``robust_weight`` 子配置。

        注意
        -----
        构造完成后会自动调用 :meth:`reset` 建立空滑窗和初始锚点状态。
        """
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "estimator": "FGOCore",
            "name": (init_cfg or {}).get("name", "fgo"),
            "cfg_keys": list(init_cfg.keys()) if isinstance(init_cfg, dict) else None,
            "window_size": (init_cfg or {}).get("window_size", _MISSING),
            "optimizer": (init_cfg or {}).get("optimizer"),
            "factor_weights": (init_cfg or {}).get("factor_weights"),
            "gate": (init_cfg or {}).get("gate"),
            "robust_weight": (init_cfg or {}).get("robust_weight"),
        }, "FGOCore.__init__")
        super().__init__(init_cfg)  # 先建立基础状态、协方差和默认配置。
        self.name = str(self.cfg.get("name") or "fgo")  # 给上层日志用的名字。
        self._pose_indices_cache = tuple(state_items.index(key) for key in self._POSE_KEYS)  # 缓存位姿在完整状态里的索引。
        # §10.3 偷懒审视固化第二轮：window_size 与 τ/ratio 同属 spec 第1663行 "可复述且不可博弈"
        # 五量之一（fgo.yaml L14-24 注释明示同口径），同样强制显式写死，禁止代码默认值
        # 兜底——否则 sweep 缺键时代码会拼凑合规档位，把 spec「必须可复述」洗成「代码凑出了」。
        _MISSING_KEY_TEMPLATE = (
            "FGO §10.3 memory-depth \"write-and-freeze\" discipline: key '{key}' "
            "must be provided explicitly by configs/models/fgo.yaml (or sweep "
            "override), not inferred from a code-side default. Rewrite the "
            "config to make the value reproducible per §10.3 / §27.3 / B11."
        )
        _window_size_raw = self.cfg.get("window_size", _MISSING)  # §10.3 与 τ_filt_s 同口径：先取哨兵，禁默认值兜底。
        if _window_size_raw is _MISSING:  # 哨兵未被替换说明 cfg 缺键。
            raise KeyError(_MISSING_KEY_TEMPLATE.format(key="window_size"))
        self.window_size = _coerce_positive_int(_window_size_raw, path="window_size")  # 滑窗长度。
        # §3.4 + §10.3 / B10–B11：窗长与滤波记忆同量级档；配置以「步」计窗须用
        # 标称事件率换算秒后再验比 ℓ_win/τ_filt ∈ [0.5, 3]。下面 nominal_event_rate_hz /
        # tau_filt_s / ratio_min/max 都从协议层写死的标量继承，越界直接 raise——
        # 这是 spec 硬约束，避免运行时静默漂到 [0.5, 3] 之外的「窗长特权/无意义短窗」。
        nominal_event_rate_hz_raw = self.cfg.get("nominal_event_rate_hz", _MISSING)
        if nominal_event_rate_hz_raw is _MISSING:
            raise KeyError(_MISSING_KEY_TEMPLATE.format(key="nominal_event_rate_hz"))
        nominal_event_rate_hz = _coerce_positive_scalar(
            nominal_event_rate_hz_raw, path="nominal_event_rate_hz"
        )  # 标称事件率（Hz）；sim 默认 IMU 100Hz，但必须由 fgo.yaml 显式落字。
        tau_filt_s_raw = self.cfg.get("tau_filt_s", _MISSING)
        if tau_filt_s_raw is _MISSING:
            raise KeyError(_MISSING_KEY_TEMPLATE.format(key="tau_filt_s"))
        # §10.3 二选一代理 τ 路径封堵：spec 允许二选一代理（标准 EKF + 视距标定段
        # 位置误差自相关降至 1/e 的时延），但必须"一次冻结"——即 τ 必须是协议层
        # 显式写死的标量常量，绝不允许运行时根据 FGO/学习方法的实际误差相关时间
        # 事后调 τ 再改 ℓ_win（spec 第 1666 行）。这里拒绝任何非标量的 τ 代理
        # （callable / dict / tensor），防止 sweep 层把 τ 换成动态推断值。
        if callable(tau_filt_s_raw):
            raise TypeError(
                "tau_filt_s must be a concrete scalar in fgo.yaml, not a callable. "
                "§10.3 二选一代理 τ 路径（autocorrelation on 视距标定段）须在协议版本"
                "中一次冻结后以标量写入 fgo.yaml；禁止运行时根据某次失败轨/测试高压段"
                "的误差相关时间事后调 τ 再改 ℓ_win。"
            )
        if isinstance(tau_filt_s_raw, Mapping):
            raise TypeError(
                "tau_filt_s must be a concrete scalar in fgo.yaml, not a mapping. "
                "§10.3 二选一代理 τ 路径须以标量写死；dict 形式的动态代理违反"
                "§10.3 '禁止用 FGO/学习方法的实际误差相关时间事后调 τ'。"
            )
        tau_filt_s = _coerce_positive_scalar(
            tau_filt_s_raw, path="tau_filt_s"
        )  # 滤波记忆尺度（秒）；§10.3 默认 min(5s, 0.2·T_eff)，须显式写死。
        # §10.3 诚实边界守卫（B2 v6 / 协议填槽表 L103 C8 修复）：
        # tau_filt_assumed_t_eff_min_s 显性声明 tau_filt=5.0 的协议层 T_eff 域假设
        # （spec §10.3 选项 1：τ_filt = min(5s, 0.2·T_eff)，T_eff≥25s 时 min(5,5)=5）。
        # 协议填槽表 L103 此前记录"C8 半过"原因之一是 τ_filt 隐含假设未在 yaml 显式落地；
        # 本字段补齐后 sentinel 强制校验 tau_filt ≤ min(5, 0.2 × assumed_T_eff) 一致性。
        _assumed_t_eff_raw = self.cfg.get("tau_filt_assumed_t_eff_min_s", _MISSING)
        if _assumed_t_eff_raw is _MISSING:
            raise KeyError(_MISSING_KEY_TEMPLATE.format(key="tau_filt_assumed_t_eff_min_s"))
        _assumed_t_eff = _coerce_positive_scalar(
            _assumed_t_eff_raw, path="tau_filt_assumed_t_eff_min_s"
        )  # spec §10.3 选项 1 协议层 T_eff 显式假设（秒）。
        _tau_filt_upper_bound = min(5.0, 0.2 * float(_assumed_t_eff))  # spec §10.3 选项 1 公式。
        if float(tau_filt_s) > _tau_filt_upper_bound + 1e-9:  # 容差吸收浮点误差。
            raise ValueError(
                f"§10.3 τ_filt 诚实边界违例：tau_filt_s={tau_filt_s}s 超过 "
                f"spec §10.3 选项 1 上限 min(5s, 0.2·T_eff)={_tau_filt_upper_bound}s "
                f"（tau_filt_assumed_t_eff_min_s={_assumed_t_eff}s）。须把 "
                f"tau_filt_s 降到 ≤ {_tau_filt_upper_bound}s，或提高 "
                f"tau_filt_assumed_t_eff_min_s 与协议 scene_scale.t_eff_min_s 一致后"
                f"再跑主表。spec 第 1666 行明确禁止事后调 τ。"
            )
        ratio_min_raw = self.cfg.get("window_length_ratio_min", _MISSING)
        if ratio_min_raw is _MISSING:
            raise KeyError(_MISSING_KEY_TEMPLATE.format(key="window_length_ratio_min"))
        ratio_min = _coerce_positive_scalar(
            ratio_min_raw, path="window_length_ratio_min"
        )  # α = 0.5（§10.3 / B10）；spec 要求协议显式落字。
        ratio_max_raw = self.cfg.get("window_length_ratio_max", _MISSING)
        if ratio_max_raw is _MISSING:
            raise KeyError(_MISSING_KEY_TEMPLATE.format(key="window_length_ratio_max"))
        ratio_max = _coerce_positive_scalar(
            ratio_max_raw, path="window_length_ratio_max"
        )  # β = 3.0（§10.3 / B10）；spec 要求协议显式落字。
        # 检查上下界本身合理，防止协议被反向写错。
        if not (ratio_min < ratio_max):
            raise ValueError(
                f"window_length_ratio_min ({ratio_min}) must be < "
                f"window_length_ratio_max ({ratio_max})"
            )  # 区间反向会让断言永不触发或永远触发，必须挡下。
        # 诚实边界（§10.3(e)）：ℓ/τ 是信息集公平代理，不是「滤波记忆 = 有限窗」的物理定理；
        # EKF 协方差记忆连续衰减，FGO 是硬截断窗——比值只防止数量级作弊，不宣称两者信息算子等价。
        # 步→秒换算：ℓ_win_seconds = window_size / nominal_event_rate_hz。
        # 这是 §3.4 字面"配置若以「步」计窗，须用标称事件率换算秒后再验比"的代码落地。
        ell_win_seconds = float(self.window_size) / float(nominal_event_rate_hz)
        ratio = ell_win_seconds / float(tau_filt_s)
        # 区间端点采用闭区间是与 §10.3 写法 "[α, β]" 一致；端点处仍合规。
        if not (ratio_min <= ratio <= ratio_max):
            raise ValueError(
                "FGO window length violates §3.4 / §10.3 / B10–B11 fairness band: "
                f"window_size={self.window_size} steps × nominal_event_rate_hz="
                f"{nominal_event_rate_hz} Hz → ℓ_win={ell_win_seconds:.4f}s; "
                f"τ_filt={tau_filt_s}s → ratio={ratio:.4f}; "
                f"required ratio ∈ [{ratio_min}, {ratio_max}]. "
                "Fix configs/models/fgo.yaml: window_size, nominal_event_rate_hz, "
                "or tau_filt_s before running the main table (spec forbids "
                "test-time tuning of τ_filt)."
            )  # spec 铁律：越界退出"无核 FGO"身份；不合规即停表。
        # 把换算结果挂到实例字段，便于上层审计 / log 解释 FGO 当前合规档位。
        self.nominal_event_rate_hz = float(nominal_event_rate_hz)
        self.tau_filt_s = float(tau_filt_s)
        self.tau_filt_assumed_t_eff_min_s = float(_assumed_t_eff)  # §10.3 诚实边界守卫：协议层 T_eff 假设。
        self.window_length_ratio_min = float(ratio_min)
        self.window_length_ratio_max = float(ratio_max)
        self.window_length_seconds = ell_win_seconds  # 当前窗口换算后的秒数。
        self.window_length_ratio = float(ratio)  # 当前 ℓ_win / τ_filt 比值。
        optimizer_cfg = self.cfg.get("optimizer") or {}  # 读取优化器子配置。
        if not isinstance(optimizer_cfg, Mapping):  # 必须是映射。
            raise TypeError("optimizer must be a mapping")  # 否则无法读取 max_iters 等字段。
        self.solver_cfg = dict(optimizer_cfg)  # 保存一份可修改的求解器配置。
        optimizer_name = str(self.solver_cfg.get("name", "gauss_newton"))
        if optimizer_name != "gauss_newton":  # 当前仅支持高斯牛顿优化器。
            raise ValueError(f"optimizer.name only supports 'gauss_newton', got '{optimizer_name}'")
        _max_iters_raw = self.solver_cfg.get("max_iters", _MISSING)  # §10.3 与 τ/window_size 同口径：先取哨兵，禁默认值兜底。
        if _max_iters_raw is _MISSING:  # 哨兵未被替换说明 optimizer.max_iters 缺键（含 optimizer dict 整键缺失）。
            raise KeyError(_MISSING_KEY_TEMPLATE.format(key="optimizer.max_iters"))
        self.max_iters = _coerce_positive_int(_max_iters_raw, path="optimizer.max_iters")  # 最大高斯牛顿迭代轮数。
        factor_weights = self.cfg.get("factor_weights") or {}  # 读取因子权重子配置。
        if not isinstance(factor_weights, Mapping):  # 必须是映射。
            raise TypeError("factor_weights must be a mapping")  # 否则没法取 imu/uwb/vio 权重。
        self.factor_weights = {  # 把三个模态的权重集中保存。
            "imu": _coerce_positive_scalar(factor_weights.get("imu", 1.0), path="factor_weights.imu"),  # IMU 因子权重。
            "uwb": _coerce_positive_scalar(factor_weights.get("uwb", 1.0), path="factor_weights.uwb"),  # UWB 因子权重。
            "vio": _coerce_positive_scalar(factor_weights.get("vio", 1.0), path="factor_weights.vio"),  # VIO 因子权重。
        }  # 因子权重字典构造结束。
        measurement_noise_cfg = self.cfg.get("measurement_noise")  # 读取量测噪声配置。
        if isinstance(measurement_noise_cfg, Mapping):  # 如果用户显式给了噪声配置，就检查相关模态。
            for modality in ("uwb", "vio"):  # 这里只对外部会直接用到的模态做预检查。
                if modality in measurement_noise_cfg:  # 只检查配置里实际存在的项。
                    self._required_measurement_noise(modality)  # 预热并校验该模态噪声。
        self.runtime_resource_meta["params"] += float(self.window_size * len(self._POSE_KEYS))  # 估算额外参数规模。
        self.params = float(self.runtime_resource_meta["params"])  # 把估算后的参数规模同步到实例字段。
        self.runtime_resource_meta["ram_peak"] = max(  # 更新峰值内存估算。
            float(self.runtime_resource_meta["ram_peak"]),  # 保留旧估算和新估算里更大的那个。
            float(self.params / 192.0),  # 这里用参数规模粗略换算内存。
        )  # 峰值内存估算更新完毕。
        self.runtime_resource_meta["ram_peak_mb"] = self.runtime_resource_meta["ram_peak"]  # 对外提供 MB 口径。
        self.ram_peak = float(self.runtime_resource_meta["ram_peak"])  # 缓存峰值内存。
        self.ram_peak_mb = float(self.runtime_resource_meta["ram_peak_mb"])  # 缓存 MB 口径的峰值内存。
        self._window_entries: list[dict[str, Any]] = []  # 存放滑窗里每个时间点的条目。
        self.constraints: list[dict[str, Any]] = []  # 存放所有追加过的因子约束。
        self.window_states: list[dict[str, float]] = []  # 存放当前窗口的可读状态快照。
        self._last_vio_reference_index: int | None = None  # 最近一次 VIO 参考位姿在窗口中的索引。
        self.reset()  # 初始化完配置后，立即建立空滑窗和初始锚点状态。

    def _gate_cfg(self) -> dict[str, Any]:
        """取出 gate 子配置。

        返回
        -------
        dict[str, Any]
            gate 配置的普通字典副本，包含 ``quality_floor`` 和 ``mahalanobis_sq``。

        异常
        ------
        TypeError
            当 ``gate`` 存在但不是 Mapping 类型时抛出。
        """
        gate = self.cfg.get("gate")  # 取出 gate 子配置。
        if gate is None:  # 缺省时返回空字典。
            return {}
        if not isinstance(gate, Mapping):  # gate 必须是映射，方便按键读取门控阈值。
            raise TypeError("gate must be a mapping")  # 不是映射就无法解释里面的字段。
        return copy.deepcopy(gate)  # 返回深拷贝，避免外部改动原配置对象（含嵌套映射）。

    def _robust_cfg(self) -> dict[str, Any]:
        """取出 robust_weight 子配置。

        返回
        -------
        dict[str, Any]
            robust_weight 配置的普通字典副本，包含 ``type`` 和 ``delta``。

        异常
        ------
        TypeError
            当 ``robust_weight`` 存在但不是 Mapping 类型时抛出。
        """
        robust_weight = self.cfg.get("robust_weight")  # 取出鲁棒权重子配置。
        if robust_weight is None:  # 缺省时返回空字典。
            return {}
        if not isinstance(robust_weight, Mapping):  # 鲁棒配置必须是映射。
            raise TypeError("robust_weight must be a mapping")  # 否则不能读 type/delta。
        return copy.deepcopy(robust_weight)  # 返回深拷贝，避免外部改动原配置对象（含嵌套映射）。

    def _quality_floor(self, modality: str) -> float:
        """按模态读取质量门槛。

        铁律 10 (FGO 裸跑): 强制返回 0.0，不做质量门控。
        任何量测都不因质量过低被拒绝，与旧配置向后兼容。

        返回
        -------
        float
            始终返回 0.0，即不拒绝任何量测。
        """
        # 铁律 10: 裸跑 FGO，不做质量门控。保留旧逻辑作为注释供回溯。
        # gate = self._gate_cfg()
        # quality_floor_cfg = gate.get("quality_floor")
        # ...
        return 0.0

    def _nis_threshold(self, modality: str | None = None) -> float:
        """读取马氏距离平方阈值。

        铁律 10 (FGO 裸跑): 强制返回 ``inf``，不做 NIS 门控。
        任何残差都通过门控进入因子图，与旧配置向后兼容。

        返回
        -------
        float
            始终返回 ``float("inf")``，即不做 NIS 门控。
        """
        # 铁律 10: 裸跑 FGO，不做 NIS 门控。保留旧逻辑作为注释供回溯。
        return float("inf")

    def _huber_weight(self, whitened_residual_norm: float) -> float:
        """按 Huber 规则计算鲁棒权重。

        铁律 10 (FGO 裸跑): 强制返回 1.0，不做 Huber 降权。
        ``robust_type`` 配置字段无论为何值均不报错，保留向后兼容性。

        参数
        ----------
        whitened_residual_norm : float
            白化残差范数（NIS 的平方根），此处仅校验类型向后兼容。

        返回
        -------
        float
            始终返回 1.0，即不做任何鲁棒降权。
        """
        # 铁律 10: 裸跑 FGO，不做 Huber 降权。保留旧逻辑作为注释供回溯。
        return 1.0

    def _robust_report(
        self,
        whitened_residual_norm: float,
        robust_weight: float,
        covariance_scale: float,
    ) -> dict[str, Any]:
        """构造无核身份下的鲁棒子报告。

        协议 cmp3 / 铁律 10 身份是平方损失裸奔 (``type=l2``)。
        即便 YAML 历史残留 ``type:huber``，运行时权重恒为 1.0，报告仍声明 ``l2``，
        避免把配置文案误写成“有核 FGO”。
        ``delta`` 记为 ``inf`` 表示无截断阈值（非 Huber）。

        参数
        ----------
        whitened_residual_norm : float
            白化残差范数。
        robust_weight : float
            实际应用的权重（铁律 10 下恒为 1.0）。
        covariance_scale : float
            协方差缩放（铁律 10 下恒为 1.0）。

        返回
        -------
        dict[str, Any]
            无核语义的 robust 子报告。
        """
        return {
            "type": "l2",  # 无核平方损失身份，不是 huber。
            "delta": float("inf"),  # 无截断阈值。
            "whitened_residual_norm": whitened_residual_norm,  # 白化残差范数。
            "weight": robust_weight,  # 实际权重（应为 1.0）。
            "covariance_scale": covariance_scale,  # 协方差缩放（应为 1.0）。
            "huber_forced_off": True,  # cmp3 封印：运行时强制无核。
            "robust_weight_bypass": True,  # 配置/旁路均不启用有效鲁棒降权。
        }

    def _quality_value(self, payload_section: Mapping[str, Any] | None) -> float:
        """从量测载荷里提取 quality。

        参数
        ----------
        payload_section : Mapping[str, Any] | None
            量测负载子字典，应包含 ``"quality"`` 键；
            为 None 或非 Mapping 时默认返回 1.0。

        返回
        -------
        float
            规范化后的质量值。

        异常
        ------
        ValueError
            当质量值非有限或超出 [0, 1] 区间时抛出。
        TypeError
            当质量值为 bool 类型时抛出。
        """
        if not isinstance(payload_section, Mapping):  # 没有载荷就按满质量处理。
            return 1.0  # 缺省认为质量满分。
        quality = payload_section.get("quality", 1.0)  # quality 没给时默认 1。
        if is_bool_like(quality):  # bool 质量值与协议层和质量模型的连续分数语义冲突。
            raise TypeError("quality must be numeric, got bool")
        value = coerce_finite_scalar(quality, name="quality")  # 质量必须是有限值。
        quality_min = DEFAULT_THRESHOLDS["quality_min"]  # 质量下界（单源真相，与 validate_event 对齐）。
        quality_max = DEFAULT_THRESHOLDS["quality_max"]  # 质量上界（单源真相，与 validate_event 对齐）。
        if value < quality_min:  # 质量值不能低于下界。
            raise ValueError(f"quality must be non-negative, got {value}")  # 与协议层 validate_event 严格模式对齐。
        if value > quality_max:  # 质量值不能超过上界。
            raise ValueError(f"quality must not exceed {quality_max}, got {value}")  # 与协议层 validate_event 严格模式对齐，下界与上界对称严格。
        return value  # 返回校验后的质量值。

    def reset(self, initial_state: dict | None = None) -> None:
        """清空窗口和约束，并重新放入初始状态。

        参数
        ----------
        initial_state : dict | None
            外部显式指定的初始状态，优先级高于配置中的 ``init_state``。

        注意
        -----
        调用父类 :meth:`EKFCore.reset` 恢复基础状态和协方差后，
        再清空滑窗条目、因子约束和 VIO 参考位姿，最后追加初始窗口条目。
        """
        super().reset(initial_state)  # 先让父类恢复基础状态和协方差。
        self._window_entries = []  # 清掉滑窗条目。
        self.constraints = []  # 清掉已挂载的因子。
        self.window_states = []  # 清掉可读快照。
        self._last_vio_reference_pose = None  # 清掉最近 VIO 参考位姿。
        self._last_vio_reference_index = None  # 清掉最近 VIO 参考索引。
        self._last_vio_reference_pose_timestamp = None  # 清掉参考位姿时间戳。
        self._append_window_entry(self._state_vector(), timestamp=None, motion_from_prev=None)  # 重新塞入初始窗口条目。

    def _pose_indices(self) -> list[int]:
        """返回位姿在完整状态里的索引列表。

        返回
        -------
        list[int]
            位姿分量（px, py, yaw）在完整状态向量中的索引。
        """
        return list(self._pose_indices_cache)  # 返回缓存副本。

    def _apply_uwb_clock_bias_kalman_update(self, residual: float, scalar_noise: float) -> None:
        """§2 细节 23 五方法同模型——FGO 旁路 EKF 让 uwb_clock_bias 在线辨识。

        背景：FGOCore 继承 EKFCore 的 10 维状态向量与协方差，但因子图节点
        只保留 px/py/yaw 三维（``_POSE_KEYS``），uwb_clock_bias 不进因子图节点。
        这与 EKF 族（ekf_core / robust_ekf_core 经 ``run_uwb_update`` 用 1×10 H
        做 K 更新）「不同模型」，违反 §2 细节 23「五方法全体同增同模型」。

        修法：在 ``_handle_uwb`` 末尾、solve() 之后做一次单标量 Kalman 旁路更新。
        pose 节点仍由因子图优化，uwb_clock_bias 由旁路 EKF 用同一份 self._covariance
        的 [_IDX_UWB_CLOCK_BIAS, _IDX_UWB_CLOCK_BIAS] 自方差 + 测量噪声做标量 K 更新。
        不动 covariance 的 pose 子块（与因子图 pose 优化结果保持一致），不破 VIO NIS
        测试（VIO 雅可比第 _IDX_UWB_CLOCK_BIAS 列为 0，covariance[8,8] 改不影响 VIO S）。

        参数
        ----------
        residual : float
            UWB 测距残差（``raw_range - z_pred_h``），由 ``_handle_uwb`` 算出（§3.0.2 / §3.1.2：bias 进 h(·)，残差定义在原始 z 上）。
            ``predict_range(..., extra_bias=bias_applied_h)`` 已含 uwb_clock_bias 与 h 侧 bias 加项，
            故残差=「扣掉当前钟差估计与 h 侧有界偏置后的测距偏差」，正是钟差辨识所需信息量。
        scalar_noise : float
            UWB 测距测量噪声方差（标量），与 ``_handle_uwb`` 同口径。
        """
        if not math.isfinite(residual):  # 非有限残差不做旁路更新，与 NIS 门控口径一致。
            return
        P_bb = float(self._covariance[_FGO_IDX_UWB_CLOCK_BIAS, _FGO_IDX_UWB_CLOCK_BIAS])  # 钟差自方差。
        if P_bb <= 0.0:  # 钟差自方差非正时跳过，避免负方差污染。
            return
        S_bias = P_bb + max(0.0, float(scalar_noise))  # 1 维创新协方差（钟差自方差 + 测量噪声）。
        if S_bias <= 0.0:  # 创新协方差非正跳过。
            return
        K_b = P_bb / S_bias  # 标量 Kalman 增益。
        bias_step = K_b * float(residual)  # 钟差修正步长。
        # 仅更新 self._state['uwb_clock_bias'] 标量，不动 pose/vx/vy/bax/bay/bg 项。
        self._state["uwb_clock_bias"] = float(self._state.get("uwb_clock_bias", 0.0)) + bias_step
        # Joseph 形式更新钟差自方差（保 PSD）：P' = (1-K) P (1-K) + K R K，对 1 维即 (1-K)^2 P + K^2 R。
        new_P_bb = (1.0 - K_b) ** 2 * P_bb + K_b ** 2 * max(0.0, float(scalar_noise))
        self._covariance[_FGO_IDX_UWB_CLOCK_BIAS, _FGO_IDX_UWB_CLOCK_BIAS] = max(0.0, new_P_bb)
        # C5: 开放钟差与其他状态的交叉协方差传播（保留 pose 子块不变，仅更新 bias 行/列）。
        # 先更新 P[8,:] 和 P[:,8] 的交叉项：P_cross = K_b * P_before @ H[:,~idx_b]
        # H[:,~idx_b] 是状态转移中非 bias 维度的 Jacobian 列，此处用简化近似：
        #         对所有状态维度 j≠bias_idx 传播交叉协方差，for 循环跳过 bias_idx 自身。
        # 注意：与 _apply_vio_scale_kalman_update 同口径，使用 self._covariance.shape[0]
        # 取状态维数（10），避免引用未定义的 state_dim 局部变量（前提指导 §5.1 全员同一模型）。
        for j in range(self._covariance.shape[0]):
            if j == _FGO_IDX_UWB_CLOCK_BIAS:
                continue
            P_bj = float(self._covariance[_FGO_IDX_UWB_CLOCK_BIAS, j])
            # 交叉协方差更新：P[8,j] = (1-K) * P[8,j]，与 scalar K 一致
            self._covariance[_FGO_IDX_UWB_CLOCK_BIAS, j] = (1.0 - K_b) * P_bj
            self._covariance[j, _FGO_IDX_UWB_CLOCK_BIAS] = (1.0 - K_b) * P_bj

    def _apply_vio_scale_kalman_update(self, vio_residual: np.ndarray, R_vio: np.ndarray) -> None:
        """§2 细节 23 五方法同模型——FGO 旁路 EKF 让 vio_scale 在线辨识。

        与 ``_apply_uwb_clock_bias_kalman_update`` 同模式：在 ``_handle_vio`` 末尾、
        solve() 之后做单标量 Kalman 旁路更新 vio_scale。pose 节点仍由因子图优化，
        vio_scale 由旁路 EKF 用 self._covariance 的 [_IDX_VIO_SCALE, _IDX_VIO_SCALE]
        自方差 + VIO 测量噪声方差做标量 K 更新。

        参数
        ----------
        vio_residual : np.ndarray
            VIO 三维残差向量 [dx, dy, dyaw]，由 ``_handle_vio`` 末尾算出。
        R_vio : np.ndarray
            VIO 三维测量噪声协方差矩阵（3x3）。
        """
        if vio_residual is None or not np.all(np.isfinite(vio_residual)):  # 非有限残差跳过。
            return
        P_ss = float(self._covariance[_FGO_IDX_VIO_SCALE, _FGO_IDX_VIO_SCALE])  # vio_scale 自方差。
        if P_ss <= 0.0:  # 非正自方差跳过。
            return
        # VIO 测量模型 z_vio = R(-yaw_ref) @ (vio_scale * [px; py])，vio_scale 仅对 dx/dy 量测有偏导，
        # 对 dyaw 偏导为 0（见 vision_update_step.py:324 注释）。残差对 vio_scale 的等效标量
        # 信息量取 ||[dx_res, dy_res]|| 当作观测残差，R 取 R_vio 的 [0:2, 0:2] 子块平均对角。
        dx_dy_residual = float(np.hypot(vio_residual[0], vio_residual[1]))  # 平面残差范数。
        R_pos = max(0.0, 0.5 * (float(R_vio[0, 0]) + float(R_vio[1, 1])))  # 平面位置噪声方差均值。
        S_s = P_ss + R_pos  # 标量创新协方差。
        if S_s <= 0.0:
            return
        K_s = P_ss / S_s  # 标量 Kalman 增益。
        scale_step = K_s * dx_dy_residual  # vio_scale 修正步长（仅平面位置残差驱动）。
        self._state["vio_scale"] = float(self._state.get("vio_scale", 0.0)) + scale_step
        new_P_ss = (1.0 - K_s) ** 2 * P_ss + K_s ** 2 * R_pos  # Joseph 形式 vio_scale 自方差。
        self._covariance[_FGO_IDX_VIO_SCALE, _FGO_IDX_VIO_SCALE] = max(0.0, new_P_ss)
        # C5 交叉协方差传播：与 _apply_uwb_clock_bias_kalman_update 同模式，
        # 对 j != vio_scale_idx 传播 P[vio_scale_idx, j] 和 P[j, vio_scale_idx]。
        for j in range(self._covariance.shape[0]):
            if j == _FGO_IDX_VIO_SCALE:
                continue
            P_sj = float(self._covariance[_FGO_IDX_VIO_SCALE, j])
            self._covariance[_FGO_IDX_VIO_SCALE, j] = (1.0 - K_s) * P_sj
            self._covariance[j, _FGO_IDX_VIO_SCALE] = (1.0 - K_s) * P_sj

    def _pose_from_state_vector(self, x_vector: np.ndarray) -> np.ndarray:
        """从完整状态向量里抽出位姿部分。

        参数
        ----------
        x_vector : np.ndarray
            完整状态向量，长度与 :data:`state_items` 一致。

        返回
        -------
        np.ndarray
            位姿向量 ``[px, py, yaw]``。
        """
        return np.asarray([float(x_vector[index]) for index in self._pose_indices()], dtype=float)  # 按固定顺序抽取。

    def _write_pose_into_state(self, x_vector: np.ndarray, pose_vector: np.ndarray) -> np.ndarray:
        """把优化后的位姿写回完整状态向量。

        参数
        ----------
        x_vector : np.ndarray
            原始完整状态向量。
        pose_vector : np.ndarray
            优化后的位姿向量 ``[px, py, yaw]``。

        返回
        -------
        np.ndarray
            更新后的完整状态向量副本，仅位姿分量被替换。
        """
        updated = np.asarray(x_vector, dtype=float).copy()  # 先拷贝一份，避免污染输入。
        for pose_index, state_index in enumerate(self._pose_indices()):  # 逐个把位姿分量塞回状态向量。
            updated[state_index] = float(pose_vector[pose_index])  # 每个位置对应一个位姿分量。
        return updated  # 返回写回后的完整状态。

    def _pose_dict_from_vector(self, pose_vector: np.ndarray) -> dict[str, float]:
        """把位姿向量整理成字典，方便窗口引用。

        参数
        ----------
        pose_vector : np.ndarray
            位姿向量 ``[px, py, yaw]``。

        返回
        -------
        dict[str, float]
            包含 VIO 更新合同 ``updated_state_items`` 所要求的全部位姿键
            (px、py、yaw 以及紧耦合扩维后的 uwb_clock_bias、
            vio_scale). 紧耦合项不参与 VIO 帧间位姿参考值, 写入 0.0,
            仅为通过 ``_resolve_state_items_from_mapping`` 的键集校验.
            yaw 会做角度归一化。
        """
        return {  # 返回统一字段名的位姿字典, 紧耦合项填零占位.
            "px": float(pose_vector[0]),  # x 坐标。
            "py": float(pose_vector[1]),  # y 坐标。
            "yaw": wrap_angle_rad(float(pose_vector[2])),  # 航向角，先做归一化。
            "uwb_clock_bias": 0.0,  # 紧耦合项填零占位.
            "vio_scale": 0.0,  # 紧耦合项填零占位.
        }  # 位姿字典构造完成。

    def _freeze_trimmed_vio_references(self, trimmed_pose: np.ndarray) -> None:
        """窗口裁剪后，重排所有 VIO 参考索引。

        当最老节点被裁出窗口时，所有 VIO 因子的 ``reference_index`` 需要
        整体左移 1 位。如果参考索引变为负数（即参考节点已被裁掉），
        则将参考位姿冻结为被裁掉节点的位姿副本，索引清空。

        参数
        ----------
        trimmed_pose : np.ndarray
            被裁剪出去的最老节点的位姿向量 ``[px, py, yaw]``。
        """
        frozen_reference_pose = [  # 先把被裁剪出去的位姿冻成一个稳定副本。
            float(trimmed_pose[0]),  # x。
            float(trimmed_pose[1]),  # y。
            float(trimmed_pose[2]),  # yaw。
        ]  # 冻结的参考位姿数组结束。
        for entry in self._window_entries:  # 遍历当前窗口里所有条目。
            for factor in entry["vio_factors"]:  # 对每个 VIO 因子都重排索引。
                reference_index = factor.get("reference_index")  # 读取原参考索引。
                if reference_index is None:  # 没索引的因子直接跳过。
                    continue  # 不需要重排。
                rebased_index = int(reference_index) - 1  # 窗口左移一格，所以索引减 1。
                if rebased_index < 0:  # 如果参考点已经被裁掉了。
                    factor["reference_index"] = None  # 索引清空。
                    factor["reference_pose"] = list(frozen_reference_pose)  # 参考位姿固定成旧值副本。
                else:  # 还在窗口里的参考点只需要整体左移。
                    factor["reference_index"] = rebased_index  # 保存重排后的索引。
        if self._last_vio_reference_index is None:  # 如果本来就没有最近参考点。
            return  # 这里就不用继续了。
        rebased_last_index = int(self._last_vio_reference_index) - 1  # 最近参考索引也要左移。
        if rebased_last_index < 0:  # 如果最近参考也被裁掉了。
            self._last_vio_reference_index = None  # 清空索引。
            self._last_vio_reference_pose = self._pose_dict_from_vector(trimmed_pose)  # 保留裁掉那一帧的位姿副本。
        else:  # 否则只调整索引。
            self._last_vio_reference_index = rebased_last_index  # 更新最近参考索引。

    def _refresh_last_vio_reference_pose_from_window(self) -> None:
        """根据当前窗口刷新最近一次 VIO 参考位姿。

        当窗口内的位姿经过优化后发生了变化，需要从窗口中重新读取
        最近参考索引对应的位姿，保证后续 VIO 因子使用的参考位姿
        与窗口内实际值一致。
        """
        if self._last_vio_reference_index is None:  # 没有参考索引就不用刷新。
            return  # 直接结束。
        reference_index = int(self._last_vio_reference_index)  # 先转成整数。
        if reference_index < 0 or reference_index >= len(self._window_entries):  # 索引越界说明参考已经失效。
            self._last_vio_reference_index = None  # 清空索引。
            self._last_vio_reference_pose = None  # 同步清空位姿，避免消费者读到与窗口脱节的陈旧参考。
            return  # 不再刷新位姿。
        reference_pose = self._pose_from_state_vector(self._window_entries[reference_index]["full_state"])  # 从窗口里重新读位姿。
        self._last_vio_reference_pose = self._pose_dict_from_vector(reference_pose)  # 保存成字典形式。

    def _resolve_latest_vio_reference(self) -> tuple[dict[str, float], int | None]:  # 返回当前可用的 VIO 参考位姿和窗口索引。
        """返回当前可用的 VIO 参考位姿和它的窗口索引。

        优先使用最近一次 VIO 更新时缓存的参考位姿；如果参考位姿过时
        （超过 ``VIO_REF_POSE_STALE_SECONDS`` 未更新），则回退到当前位姿。
        如果没有任何缓存参考，也回退到当前位姿。

        返回
        -------
        tuple[dict[str, float], int | None]
            ``(reference_pose, reference_index)``，参考位姿字典和对应窗口索引。
            回退到当前位姿时索引为 None。
        """
        self._refresh_last_vio_reference_pose_from_window()  # 先尝试用窗口状态刷新最近参考。
        if self._last_vio_reference_pose is not None:  # 如果已经有可用参考。
            # 参考位姿时效性检查：如果参考位姿超过阈值未更新，说明中间可能有 VIO 事件
            # 被 blackout 移除或门控跳过。仿真 VIO 数据提供相邻帧增量，参考位姿过时时
            # 增量语义不匹配，此时应回退到当前位姿。与 EKF 行为一致。
            if (
                self._last_vio_reference_pose_timestamp is not None
                and self._timestamp is not None
                and (self._timestamp - self._last_vio_reference_pose_timestamp) > VIO_REF_POSE_STALE_SECONDS
            ):
                return self._current_pose_reference(), None  # 回退到当前位姿。
            return dict(self._last_vio_reference_pose), self._last_vio_reference_index  # 返回参考字典和对应索引。
        return self._current_pose_reference(), None  # 否则退回当前位姿作为参考。

    def _append_window_entry(  # 向滑窗末尾追加一条状态条目。
        self,  # 当前实例本身。
        x_vector: np.ndarray,  # 要写入窗口的完整状态向量。
        *,  # 后面的参数必须显式写名，避免调用方混淆。
        timestamp: float | None,  # 该条目的时间戳。
        motion_from_prev: np.ndarray | None,  # 该条目相对前一条目的运动增量。
    ) -> None:
        """向滑窗末尾追加一条状态条目。

        新条目包含完整状态、位姿先验、运动增量和空的因子列表。
        如果追加后窗口长度超过 ``window_size``，自动裁剪最老条目
        并重排所有 VIO 参考索引。最后重建可读快照列表。

        参数
        ----------
        x_vector : np.ndarray
            要写入窗口的完整状态向量。
        timestamp : float | None
            该条目的时间戳。
        motion_from_prev : np.ndarray | None
            该条目相对前一条目的位姿运动增量 ``[dpx, dpy, dyaw]``。
        """
        pose_vector = self._pose_from_state_vector(x_vector)  # 先抽出当前位姿。
        self._window_entries.append(  # 把新条目写入窗口末尾。
            {  # 单个窗口条目的内部状态。
                "timestamp": timestamp,  # 这个窗口条目的时间戳。
                "full_state": np.asarray(x_vector, dtype=float).copy(),  # 完整状态副本。
                "pose_prior": pose_vector.copy(),  # 当前条目的位姿先验。
                "motion_from_prev": None if motion_from_prev is None else np.asarray(motion_from_prev, dtype=float).copy(),  # 相邻条目的运动增量。
                "uwb_factors": [],  # 这个时间点挂上的 UWB 因子列表。
                "vio_factors": [],  # 这个时间点挂上的 VIO 因子列表。
            }  # 单个窗口条目结束。
        )  # 新条目结构构造完毕。
        while len(self._window_entries) > self.window_size:  # 超出窗口长度就裁剪最老条目。
            trimmed_entry = self._window_entries.pop(0)  # 超出窗口后，最旧条目先弹出。
            trimmed_pose = self._pose_from_state_vector(trimmed_entry["full_state"])  # 弹出条目的位姿。
            self._freeze_trimmed_vio_references(trimmed_pose)  # 裁剪后重排 VIO 参考索引。
            if self._window_entries:  # 如果裁剪后窗口里还有条目。
                self._window_entries[0]["motion_from_prev"] = None  # 新窗口头部不再依赖被裁剪的前驱。
                self._window_entries[0]["pose_prior"] = self._pose_from_state_vector(  # 重新构造新窗口头部的位姿先验。
                    self._window_entries[0]["full_state"]  # 使用裁剪后的新窗口头部状态。
                )  # 重新用当前状态作为先验。
        self._rebuild_constraints_cache()  # 窗口裁剪后同步丢弃已出窗的历史因子。
        self.window_states = [  # 重新生成可读快照列表。
            {  # 每个窗口条目都转成一份按状态名索引的字典。
                state_key: float(entry["full_state"][index])  # 把完整状态转成可读字典。
                for index, state_key in enumerate(state_items)  # 逐个状态项展开成键值对。
            }  # 单个窗口条目的快照结束。
            for entry in self._window_entries  # 遍历整个窗口，重建可读视图。
        ]  # 所有快照重新生成完毕。
        self._refresh_last_vio_reference_pose_from_window()  # 重新刷新最近一次 VIO 参考位姿。

    def _ensure_latest_window_entry(self) -> dict[str, Any]:
        """确保窗口里至少有一条最新条目可写。

        如果窗口为空，就用当前状态补一条初始条目。

        返回
        -------
        dict[str, Any]
            窗口末尾的最新条目。
        """
        if not self._window_entries:  # 如果窗口为空，就先补一条当前状态。
            self._append_window_entry(self._state_vector(), timestamp=self._timestamp, motion_from_prev=None)  # 用当前状态开头。
        return self._window_entries[-1]  # 返回最后一条，也就是最新条目。

    def _rebuild_constraints_cache(self) -> None:
        """从当前滑窗条目重建活跃因子缓存。"""
        rebuilt_constraints: list[dict[str, Any]] = []  # 只保留当前窗口仍然挂载着的活跃约束。
        for entry in self._window_entries:  # 按窗口顺序重建缓存，保持与求解真实输入一致。
            rebuilt_constraints.extend(copy.deepcopy(factor) for factor in entry["uwb_factors"])
            rebuilt_constraints.extend(copy.deepcopy(factor) for factor in entry["vio_factors"])
        self.constraints = rebuilt_constraints  # 完全覆盖旧缓存，避免出窗因子残留。

    def _append_constraint(self, constraint: dict[str, Any]) -> None:  # 把一个因子约束挂到当前最新窗口条目上。
        """把一个因子约束挂到当前最新窗口条目上。

        根据因子类型（``"uwb"`` 或 ``"vio"``）追加到对应因子列表，
        同时在总约束表中保留一份深拷贝。

        参数
        ----------
        constraint : dict[str, Any]
            因子约束字典，必须包含 ``"type"`` 键。

        异常
        ------
        ValueError
            当因子类型不是 ``"uwb"`` 或 ``"vio"`` 时抛出。
        """
        latest_entry = self._ensure_latest_window_entry()  # 先拿到最新条目。
        constraint_copy = copy.deepcopy(constraint)  # 保留一份独立副本。
        constraint_type = constraint["type"]  # 因子类型决定挂到哪个列表。
        if constraint_type == "uwb":  # UWB 因子进 UWB 列表。
            latest_entry["uwb_factors"].append(constraint_copy)  # 追加到 UWB 因子数组。
        elif constraint_type == "vio":  # VIO 因子进 VIO 列表。
            latest_entry["vio_factors"].append(constraint_copy)  # 追加到 VIO 因子数组。
        else:  # 其他类型说明调用方传错了。
            raise ValueError(f"Unsupported constraint type: {constraint_type}")  # 直接报错。
        self.constraints.append(copy.deepcopy(constraint))  # 总约束表也保留一份。

    def _rollback_latest_constraint(self, *, constraint_type: str, entry_factor_count: int, total_constraint_count: int) -> None:
        """回滚刚刚追加但求解失败的约束。

        当高斯牛顿求解抛出异常时，需要把本次追加的因子从窗口条目
        和总约束表中删除，恢复到追加前的状态。

        参数
        ----------
        constraint_type : str
            因子类型，``"uwb"`` 或 ``"vio"``。
        entry_factor_count : int
            追加前该条目的因子数量，回滚时删除此索引之后的所有因子。
        total_constraint_count : int
            追加前总约束表的数量，回滚时删除此索引之后的所有约束。
        """
        if constraint_type not in ("uwb", "vio"):  # 只允许 UWB 或 VIO，与 _append_constraint 保持一致。
            raise ValueError(f"Unsupported constraint type: {constraint_type}")  # 直接报错。
        if not self._window_entries:  # 空窗口说明状态已不一致，不能静默新建条目。
            raise RuntimeError("cannot rollback constraint on empty window")  # 显式报错。
        latest_entry = self._window_entries[-1]  # 直接取最新条目，不触发新建副作用。
        factor_key = f"{constraint_type}_factors"  # 根据因子类型定位列表键。
        del latest_entry[factor_key][entry_factor_count:]  # 删除本次追加之后的新因子。
        del self.constraints[total_constraint_count:]  # 总表也裁掉多加的部分。

    def add_constraint(self, constraint: dict[str, Any]) -> None:
        """对外暴露的约束追加入口。

        参数
        ----------
        constraint : dict[str, Any]
            因子约束字典，必须包含 ``"type"`` 键（``"uwb"`` 或 ``"vio"``）。
        """
        self._append_constraint(constraint)  # 直接复用内部追加逻辑。

    def _sqrt_information(self, covariance, factor_weight: float) -> np.ndarray:
        """把协方差转换成平方信息矩阵（Cholesky 分解的逆）。

        平方信息矩阵 Σ^{-1/2} = √w · L^{-1}，其中 L 是协方差的
        Cholesky 分解，w 是因子权重。用于将最小二乘问题写成
        标准形式 ‖Σ^{-1/2}(z - h(x))‖²。

        参数
        ----------
        covariance : Any
            噪声协方差，标量或矩阵。
        factor_weight : float
            因子权重，用于缩放信息矩阵。

        返回
        -------
        np.ndarray
            平方信息矩阵。
        """
        covariance_array = np.asarray(covariance, dtype=float)  # 先把输入转成数组。
        if covariance_array.ndim == 0:  # 标量协方差直接变成 1x1 信息块。
            scalar_cov = max(float(covariance_array), 1e-8)  # 给协方差下一个极小下限。
            return np.asarray([[math.sqrt(max(factor_weight, 1e-8) / scalar_cov)]], dtype=float)  # 返回 1x1 平方信息。
        stabilized = covariance_array + np.eye(covariance_array.shape[0], dtype=float) * 1e-8  # 先加一点数值稳定项。
        stabilized = 0.5 * (stabilized + stabilized.T)  # 强制对称化，防止浮点累积误差导致 Cholesky 失败。
        chol_cov = np.linalg.cholesky(stabilized)  # 对稳定且对称化后的协方差做 Cholesky 分解。
        identity = np.eye(chol_cov.shape[0], dtype=float)  # 构造单位阵。
        return math.sqrt(max(factor_weight, 1e-8)) * np.linalg.solve(chol_cov, identity)  # 返回平方信息矩阵。

    def _factor_sqrt_information(self, factor: dict[str, Any], factor_weight: float) -> np.ndarray:
        """按缓存键获取或重建因子的平方信息矩阵。

        如果因子的噪声协方差没有变化（缓存键命中），直接返回上次计算结果；
        否则重新计算并更新缓存。

        参数
        ----------
        factor : dict[str, Any]
            因子字典，必须包含 ``"noise"`` 键。
        factor_weight : float
            因子权重。

        返回
        -------
        np.ndarray
            平方信息矩阵。
        """
        cache_key = _noise_cache_key(factor["noise"])  # 用噪声内容生成缓存键。
        cached_sqrt_info = factor.get("_sqrt_info")  # 读取上次缓存的平方信息。
        if factor.get("_sqrt_info_cache_key") == cache_key and isinstance(cached_sqrt_info, np.ndarray):  # 缓存命中就直接返回。
            return cached_sqrt_info  # 不重复计算。
        sqrt_info = self._sqrt_information(factor["noise"], factor_weight)  # 缓存失效时重建。
        factor["_sqrt_info_cache_key"] = cache_key  # 更新缓存键。
        factor["_sqrt_info"] = sqrt_info  # 保存新的平方信息。
        return sqrt_info  # 返回重建后的结果。

    def _factor_information(self, factor: dict[str, Any], factor_weight: float) -> np.ndarray:
        """按缓存键获取或重建因子的完整信息矩阵。

        信息矩阵 = 平方信息矩阵的转置乘积（Σ^{-1} = (Σ^{-1/2})^T · Σ^{-1/2}）。
        如果缓存命中则直接返回，否则从平方信息矩阵重建。

        参数
        ----------
        factor : dict[str, Any]
            因子字典，必须包含 ``"noise"`` 键。
        factor_weight : float
            因子权重。

        返回
        -------
        np.ndarray
            完整信息矩阵。
        """
        cache_key = _noise_cache_key(factor["noise"])  # 用噪声内容生成缓存键。
        cached_information = factor.get("_information")  # 读取上次缓存的完整信息矩阵。
        if factor.get("_information_cache_key") == cache_key and isinstance(cached_information, np.ndarray):  # 命中缓存就直接返回。
            return cached_information  # 不重复计算。
        sqrt_info = self._factor_sqrt_information(factor, factor_weight)  # 先拿平方信息矩阵。
        information = sqrt_info.T @ sqrt_info  # 再从平方信息重建完整信息矩阵。
        factor["_information_cache_key"] = cache_key  # 更新缓存键。
        factor["_information"] = information  # 保存完整信息矩阵缓存。
        return information  # 返回完整信息矩阵。

    def _linearize_system(self, pose_vector: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        """把当前窗口线性化成最小二乘系统的稀疏块。

        遍历窗口中所有节点的先验约束、IMU 运动约束、UWB 因子和 VIO 因子，
        对每个因子计算加权雅可比和残差，组装成稀疏行块 (rows) 和右侧向量 (rhs)。
        这是高斯牛顿法中构造 J^T J δ = -J^T r 的稀疏形式。

        参数
        ----------
        pose_vector : np.ndarray
            当前所有节点位姿拼成的长向量，长度为 ``pose_dim × state_count``。

        返回
        -------
        rows : list[np.ndarray]
            加权雅可比行块列表，每块形状为 ``(factor_dim, pose_dim × state_count)``。
        rhs : list[np.ndarray]
            加权残差右侧向量列表，每块形状为 ``(factor_dim,)``。
        constraint_count : int
            当前窗口中的约束总数。
        """
        state_count = len(self._window_entries)  # 当前窗口里有多少个位姿节点。
        pose_dim = len(self._POSE_KEYS)  # 每个节点只保留 px/py/yaw 三个自由度。
        rows: list[np.ndarray] = []  # 存放所有线性化块的左侧矩阵。
        rhs: list[np.ndarray] = []  # 存放所有线性化块对应的右侧向量。
        constraint_count = sum(  # 先统计相邻节点之间的 IMU 约束。
            0 if entry["motion_from_prev"] is None else 1 for entry in self._window_entries[1:]  # 每个非空 motion 都算一条。
        )  # IMU 约束初步统计完成。
        constraint_count += sum(  # 再统计每个节点挂载的 UWB 和 VIO 因子。
            len(entry["uwb_factors"]) + len(entry["vio_factors"]) for entry in self._window_entries  # 每个条目的因子数相加。
        )  # 总约束数统计完成。

        prior_row = np.zeros((pose_dim, pose_dim * state_count), dtype=float)  # 初始先验的块矩阵。
        prior_row[:, 0:pose_dim] = np.eye(pose_dim, dtype=float)  # 先验只约束第一个位姿。
        prior_residual = pose_vector[0:pose_dim] - self._window_entries[0]["pose_prior"]  # 先验残差。
        prior_residual[2] = angle_delta_rad(  # 航向残差按角差包裹，避免跨越 ±π 边界时产生 2π 跳变。
            float(pose_vector[pose_dim - 1]), float(self._window_entries[0]["pose_prior"][2])
        )
        prior_sqrt_info = math.sqrt(max(self.factor_weights["imu"], 1e-8)) * np.eye(pose_dim, dtype=float)  # 先验权重。
        rows.append(prior_sqrt_info @ prior_row)  # 把先验约束写入线性化块。
        rhs.append(-prior_sqrt_info @ prior_residual)  # 先验残差写入右侧。

        for state_index in range(1, state_count):  # 从第二个节点开始，因为第一个节点只有先验没有前驱。
            motion_from_prev = self._window_entries[state_index]["motion_from_prev"]  # 当前节点相对前一节点的运动。
            if motion_from_prev is None:  # 没有运动增量就不加这一条约束。
                continue  # 跳过空的 IMU 边。
            row = np.zeros((pose_dim, pose_dim * state_count), dtype=float)  # 相邻位姿运动因子的块矩阵。
            prev_slice = slice((state_index - 1) * pose_dim, state_index * pose_dim)  # 前一节点在大向量里的切片。
            curr_slice = slice(state_index * pose_dim, (state_index + 1) * pose_dim)  # 当前节点在大向量里的切片。
            row[:, prev_slice] = -np.eye(pose_dim, dtype=float)  # 前一节点对应负单位块。
            row[:, curr_slice] = np.eye(pose_dim, dtype=float)  # 当前节点对应正单位块。
            residual = pose_vector[curr_slice] - pose_vector[prev_slice] - motion_from_prev  # 运动残差。
            residual[2] = angle_delta_rad(  # 航向残差要按角差而不是普通减法来算。
                float(pose_vector[curr_slice][2]) - float(pose_vector[prev_slice][2]),  # 当前 yaw 减去前一 yaw。
                float(motion_from_prev[2]),  # 预测的 yaw 增量。
            )  # 航向残差计算完成。
            sqrt_info = math.sqrt(max(self.factor_weights["imu"], 1e-8)) * np.eye(pose_dim, dtype=float)  # 运动因子权重。
            rows.append(sqrt_info @ row)  # 把 IMU 运动因子写入系统。
            rhs.append(-sqrt_info @ residual)  # 右侧对应运动残差。

        for state_index, entry in enumerate(self._window_entries):  # 遍历每个窗口节点，收集挂载在该节点上的因子。
            pose_slice = slice(state_index * pose_dim, (state_index + 1) * pose_dim)  # 当前节点对应的大向量切片。
            pose_state = pose_vector[pose_slice]  # 取出当前节点的位姿。

            for factor in entry["uwb_factors"]:  # 处理当前节点上的每一个 UWB 因子。
                anchor_x, anchor_y = factor["anchor_pos"]  # 读取当前 UWB 因子对应的锚点坐标。
                dx = float(pose_state[0]) - float(anchor_x)  # 当前位姿到锚点的 x 差。
                dy = float(pose_state[1]) - float(anchor_y)  # 当前位姿到锚点的 y 差。
                z_pred = math.hypot(dx, dy)  # 几何距离预测。
                # §3.0.2 / §3.1.2：bias 进 h(·)（z_pred + extra_bias），残差定义在原始 z 上。
                extra_bias_ = float(factor.get("extra_bias", 0.0))
                z_pred_with_bias = z_pred + extra_bias_
                if z_pred > 0.0:  # 非零距离时才计算有效导数。
                    jacobian = np.asarray([[-dx / z_pred, -dy / z_pred, 0.0]], dtype=float)  # 距离对位姿的雅可比。
                else:  # 距离为零时直接用零导数，避免除零。
                    jacobian = np.zeros((1, pose_dim), dtype=float)  # 距离为零时不给导数，避免除零。
                residual = np.asarray([float(factor["z_range"]) - z_pred_with_bias], dtype=float)  # UWB 残差（原始 z - h(·)）。
                row = np.zeros((1, pose_dim * state_count), dtype=float)  # UWB 因子的块矩阵。
                row[:, pose_slice] = jacobian  # 只写当前位姿对应的列块。
                sqrt_info = self._factor_sqrt_information(factor, self.factor_weights["uwb"])  # UWB 权重。
                rows.append(sqrt_info @ row)  # 加入加权后的线性化块。
                rhs.append(-sqrt_info @ residual)  # 加入加权后的残差。

            for factor in entry["vio_factors"]:  # 再处理当前节点上的所有 VIO 因子。
                reference_pose = np.asarray(factor["reference_pose"], dtype=float)  # 读取参考位姿。
                reference_index = factor.get("reference_index")  # 参考位姿在窗口中的索引。
                z_vio = np.asarray(factor["z_vio"], dtype=float)  # 读取视觉量测。
                row = np.zeros((pose_dim, pose_dim * state_count), dtype=float)  # VIO 因子的块矩阵。
                if reference_index is not None:  # 如果这个 VIO 因子还指向窗口内部节点。
                    reference_index = int(reference_index)  # 把参考索引显式转成整数。
                    if 0 <= reference_index < state_count and reference_index != state_index:  # 参考节点确实还在窗口里。
                        reference_slice = slice(reference_index * pose_dim, (reference_index + 1) * pose_dim)  # 参考节点切片。
                        reference_pose = pose_vector[reference_slice]  # 若参考节点仍在窗口内，就直接取窗口里的位姿。
                z_hat, current_jacobian, reference_jacobian = _linearize_vio_factor(pose_state, reference_pose)  # 线性化 VIO 因子。
                row[:, pose_slice] = current_jacobian  # 当前节点对应的雅可比块。
                if reference_index is not None and 0 <= reference_index < state_count and reference_index != state_index:  # 只有引用窗口内别的节点时才写入交叉块。
                    row[:, reference_slice] = reference_jacobian  # 参考节点对应的雅可比块。
                residual = z_vio - z_hat  # VIO 的三维残差。
                residual[2] = angle_delta_rad(float(z_vio[2]), float(z_hat[2]))  # 航向残差按角差算。
                sqrt_info = self._factor_sqrt_information(factor, self.factor_weights["vio"])  # VIO 权重。
                rows.append(sqrt_info @ row)  # 加入加权后的 VIO 线性块。
                rhs.append(-sqrt_info @ residual)  # 加入加权后的 VIO 残差。

        return rows, rhs, constraint_count  # 返回线性化块、右侧向量和总约束数。

    def _accumulate_normal_system(self, pose_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """把线性化块累加成正规方程矩阵和梯度。

        直接累加 H^T W H 和 H^T W r，比 :meth:`_linearize_system` 的稀疏块形式
        更适合小规模滑窗的密集求解。先验和 IMU 约束用简化公式直接写入对角块
        和耦合块，UWB/VIO 因子通过信息矩阵累加局部 Hessian 和梯度。

        参数
        ----------
        pose_vector : np.ndarray
            当前所有节点位姿拼成的长向量。

        返回
        -------
        normal_matrix : np.ndarray
            正规方程矩阵 H^T W H，形状 ``(total_dim, total_dim)``。
        gradient : np.ndarray
            正规方程梯度 -H^T W r，形状 ``(total_dim,)``。
        constraint_count : int
            当前窗口中的约束总数。
        """
        state_count = len(self._window_entries)  # 当前窗口节点数。
        pose_dim = len(self._POSE_KEYS)  # 每个节点三维。
        total_dim = pose_dim * state_count  # 展开后的总自由度。
        normal_matrix = np.zeros((total_dim, total_dim), dtype=float)  # 正规方程矩阵。
        gradient = np.zeros(total_dim, dtype=float)  # 正规方程梯度。
        constraint_count = sum(  # 先统计相邻节点的 IMU 约束。
            0 if entry["motion_from_prev"] is None else 1 for entry in self._window_entries[1:]  # 每个有效 motion 算一条。
        )  # IMU 约束计数完成。
        constraint_count += sum(  # 再统计每个节点上的 UWB/VIO 因子。
            len(entry["uwb_factors"]) + len(entry["vio_factors"]) for entry in self._window_entries  # 每个条目的因子数相加。
        )  # 总约束计数完成。

        imu_weight = max(self.factor_weights["imu"], 1e-8)  # IMU 因子权重。
        prior_slice = slice(0, pose_dim)  # 第一个位姿的切片。
        prior_residual = pose_vector[prior_slice] - self._window_entries[0]["pose_prior"]  # 先验残差。
        prior_residual[2] = angle_delta_rad(  # 航向残差按角差包裹，避免跨越 ±π 边界时产生 2π 跳变。
            float(pose_vector[prior_slice][2]), float(self._window_entries[0]["pose_prior"][2])
        )
        normal_matrix[prior_slice, prior_slice] += imu_weight * np.eye(pose_dim, dtype=float)  # 先验对角块。
        gradient[prior_slice] += -imu_weight * prior_residual  # 先验梯度。

        for state_index in range(1, state_count):  # 从第二个节点开始处理相邻 IMU 约束。
            motion_from_prev = self._window_entries[state_index]["motion_from_prev"]  # 读取当前节点对应的运动增量。
            if motion_from_prev is None:  # 没有运动信息就不写这一条边。
                continue  # 直接跳过。
            prev_slice = slice((state_index - 1) * pose_dim, state_index * pose_dim)  # 前一位姿切片。
            curr_slice = slice(state_index * pose_dim, (state_index + 1) * pose_dim)  # 当前位姿切片。
            residual = pose_vector[curr_slice] - pose_vector[prev_slice] - motion_from_prev  # IMU 运动残差。
            residual[2] = angle_delta_rad(  # 航向残差要按角差算。
                float(pose_vector[curr_slice][2]) - float(pose_vector[prev_slice][2]),  # 当前 yaw 减去前一 yaw。
                float(motion_from_prev[2]),  # 预测的 yaw 增量。
            )  # 航向残差计算完成。
            block = imu_weight * np.eye(pose_dim, dtype=float)  # IMU 因子的局部块。
            normal_matrix[prev_slice, prev_slice] += block  # 前一位姿对角块。
            normal_matrix[curr_slice, curr_slice] += block  # 当前位姿对角块。
            normal_matrix[prev_slice, curr_slice] -= block  # 非对角耦合块。
            normal_matrix[curr_slice, prev_slice] -= block  # 非对角耦合块。
            gradient[prev_slice] += imu_weight * residual  # 前一位姿梯度。
            gradient[curr_slice] += -imu_weight * residual  # 当前位姿梯度。

        for state_index, entry in enumerate(self._window_entries):  # 遍历所有窗口节点上的因子。
            pose_slice = slice(state_index * pose_dim, (state_index + 1) * pose_dim)  # 当前节点切片。
            pose_state = pose_vector[pose_slice]  # 当前节点位姿。

            for factor in entry["uwb_factors"]:  # 逐个处理当前节点上的 UWB 因子。
                anchor_x, anchor_y = factor["anchor_pos"]  # 当前 UWB 因子对应的锚点。
                dx = float(pose_state[0]) - float(anchor_x)  # x 差值。
                dy = float(pose_state[1]) - float(anchor_y)  # y 差值。
                z_pred = math.hypot(dx, dy)  # 几何预测距离。
                # §3.0.2 / §3.1.2：bias 进 h(·)（z_pred + extra_bias），残差定义在原始 z 上。
                extra_bias_ = float(factor.get("extra_bias", 0.0))
                z_pred_with_bias = z_pred + extra_bias_
                if z_pred > 0.0:  # 非零距离才构造有效雅可比。
                    jacobian = np.asarray([[-dx / z_pred, -dy / z_pred, 0.0]], dtype=float)  # 距离对位姿的雅可比。
                else:  # 零距离用零矩阵，避免数值异常。
                    jacobian = np.zeros((1, pose_dim), dtype=float)  # 距离为零时不给导数，避免除零。
                residual = np.asarray([float(factor["z_range"]) - z_pred_with_bias], dtype=float)  # UWB 残差（原始 z - h(·)）。
                information = self._factor_information(factor, self.factor_weights["uwb"])  # UWB 信息矩阵。
                local_hessian = jacobian.T @ information @ jacobian  # 局部 Hessian。
                local_gradient = jacobian.T @ (information @ residual)  # 局部梯度。
                normal_matrix[pose_slice, pose_slice] += local_hessian  # 累加到正规方程矩阵。
                gradient[pose_slice] += -local_gradient.reshape(-1)  # 累加到梯度。

            for factor in entry["vio_factors"]:  # 再把 VIO 因子累加进来。
                reference_pose = np.asarray(factor["reference_pose"], dtype=float)  # 读取参考位姿。
                reference_index = factor.get("reference_index")  # 读取参考节点索引。
                reference_slice: slice | None = None  # 默认没有参考切片。
                if reference_index is not None:  # 如果参考节点还指向窗口内。
                    reference_index = int(reference_index)  # 把参考索引显式转成整数。
                    if 0 <= reference_index < state_count and reference_index != state_index:  # 参考节点仍然有效且不是自己。
                        reference_slice = slice(reference_index * pose_dim, (reference_index + 1) * pose_dim)  # 参考节点切片。
                        reference_pose = pose_vector[reference_slice]  # 若参考节点仍在窗口里，就直接用窗口值。
                z_vio = np.asarray(factor["z_vio"], dtype=float)  # 读取视觉量测。
                z_hat, current_jacobian, reference_jacobian = _linearize_vio_factor(pose_state, reference_pose)  # 线性化 VIO 因子。
                residual = z_vio - z_hat  # VIO 残差。
                residual[2] = angle_delta_rad(float(z_vio[2]), float(z_hat[2]))  # 航向残差按角差计算。
                information = self._factor_information(factor, self.factor_weights["vio"])  # VIO 信息矩阵。
                current_hessian = current_jacobian.T @ information @ current_jacobian  # 当前节点 Hessian。
                current_gradient = current_jacobian.T @ (information @ residual)  # 当前节点梯度。
                normal_matrix[pose_slice, pose_slice] += current_hessian  # 累加当前节点 Hessian。
                gradient[pose_slice] += -current_gradient.reshape(-1)  # 累加当前节点梯度。
                if reference_slice is not None:  # 只有参考节点在窗口内时才有耦合块。
                    reference_hessian = reference_jacobian.T @ information @ reference_jacobian  # 参考节点 Hessian。
                    cross_hessian = current_jacobian.T @ information @ reference_jacobian  # 节点间耦合块。
                    reference_gradient = reference_jacobian.T @ (information @ residual)  # 参考节点梯度。
                    normal_matrix[reference_slice, reference_slice] += reference_hessian  # 参考节点对角块。
                    normal_matrix[pose_slice, reference_slice] += cross_hessian  # 当前到参考的耦合块。
                    normal_matrix[reference_slice, pose_slice] += cross_hessian.T  # 参考到当前的耦合块。
                    gradient[reference_slice] += -reference_gradient.reshape(-1)  # 参考节点梯度。

        return normal_matrix, gradient, constraint_count  # 返回正规方程和约束数量。

    def _write_latest_pose_covariance(self, information_matrix: np.ndarray) -> list[float]:
        """把最新位姿的协方差从信息矩阵里反解出来。

        对正规方程矩阵（即信息矩阵）加微小正则项后求逆得到协方差，
        然后只截取最新节点对应的 3×3 子块，写回完整状态协方差矩阵
        的位姿位置。最后做数值对称化。

        参数
        ----------
        information_matrix : np.ndarray
            最终一轮优化后的正规方程矩阵（信息矩阵）。

        返回
        -------
        list[float]
            最新位姿协方差的对角线元素列表 ``[var_px, var_py, var_yaw]``。

        异常
        ------
        ValueError
            当反解出的协方差含非有限值时抛出。
        """
        def _stabilize_psd(matrix: np.ndarray, *, name: str) -> np.ndarray:
            """把对称协方差块修到半正定，避免出口出现负方差。"""
            stabilized = np.asarray(matrix, dtype=float)
            stabilized = 0.5 * (stabilized + stabilized.T)
            eigvals, eigvecs = np.linalg.eigh(stabilized)
            min_eig = float(np.min(eigvals))
            if min_eig < 0.0:
                eigvals = np.maximum(eigvals, 0.0)
                stabilized = eigvecs @ np.diag(eigvals) @ eigvecs.T
                stabilized = 0.5 * (stabilized + stabilized.T)
                _logger.warning("%s had negative eigenvalues; clipped to PSD", name)
            return stabilized

        pose_dim = len(self._POSE_KEYS)  # 单个节点的自由度。
        state_count = len(self._window_entries)  # 当前窗口节点数。
        stabilized_information = information_matrix + np.eye(information_matrix.shape[0], dtype=float) * 1e-8  # 防止奇异。
        full_pose_covariance = np.linalg.inv(stabilized_information)  # 从信息矩阵反解协方差。
        latest_pose_slice = slice((state_count - 1) * pose_dim, state_count * pose_dim)  # 最新节点的切片。
        latest_pose_covariance = np.asarray(  # 截取最新位姿对应的协方差块。
            full_pose_covariance[latest_pose_slice, latest_pose_slice],  # 取最新节点的协方差子块。
            dtype=float,  # 用浮点数组保存。
        )  # 协方差子块截取完成。
        latest_pose_covariance = _stabilize_psd(latest_pose_covariance, name="latest_pose_covariance")  # 数值对称化并修正为半正定。
        if not np.all(np.isfinite(latest_pose_covariance)):  # 只要有非有限值就不能继续。
            raise ValueError("optimized pose covariance must be finite")  # 最新位姿协方差必须正常。
        updated_covariance = np.asarray(self._covariance, dtype=float).copy()  # 从当前协方差拷贝一份。
        pose_indices = self._pose_indices()  # 找到位姿对应的状态索引。
        updated_covariance[np.ix_(pose_indices, pose_indices)] = latest_pose_covariance  # 只回写位姿块。
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)  # 再整体对称化。
        if not np.all(np.isfinite(updated_covariance)):  # 整体状态协方差也必须正常。
            raise ValueError("optimized state covariance must be finite")  # 不能留下非法值。
        self._covariance = updated_covariance  # 校验通过后再写回，避免异常时半更新状态。
        return [max(float(value), 0.0) for value in np.diag(latest_pose_covariance)]  # 只返回最新位姿对角线，且不允许负方差。

    def solve(self) -> dict[str, Any]:
        """对当前滑窗做一次高斯牛顿优化并写回状态。

        迭代流程：
        1. 构造正规方程 H^T W H δ = -H^T W r
        2. 求解增量 δ
        3. 更新位姿向量 x ← x + δ
        4. 归一化每个节点的航向角
        5. 检查收敛（‖δ‖ ≤ 1e-6 则停止）
        6. 最后一轮反解最新位姿协方差

        返回
        -------
        dict[str, Any]
            求解结果，包含：
            - ``iterations`` : 实际迭代轮数
            - ``constraint_count`` : 当前窗口约束总数
            - ``window_length`` : 当前窗口长度
            - ``latest_pose_covariance_diag`` : 最新位姿协方差对角线
        """
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "estimator": "FGOCore",
            "window_size": getattr(self, "window_size", None),
            "max_iters": getattr(self, "max_iters", None),
            "window_entries": len(self._window_entries) if hasattr(self, "_window_entries") else None,
        }, "FGOCore.solve")
        if not self._window_entries:  # 空窗口直接返回。
            return {  # 空窗口直接返回空结果，字段与非空返回保持一致。
                "iterations": 0,  # 没有迭代。
                "constraint_count": 0,  # 没有约束。
                "window_length": 0,  # 窗口长度为零。
                "latest_pose_covariance_diag": None,  # 空窗口没有协方差可反解。
            }  # 空窗口返回块结束。

        state_count = len(self._window_entries)  # 当前窗口节点数。
        pose_dim = len(self._POSE_KEYS)  # 每个节点的位姿自由度。
        pose_vector = np.concatenate(  # 把窗口里每个节点的位姿拼成一个长向量。
            [self._pose_from_state_vector(entry["full_state"]) for entry in self._window_entries]  # 每个窗口条目只取位姿。
        )  # 长位姿向量构造完成。
        constraint_count = 0  # 先初始化约束计数。
        latest_pose_covariance_diag: list[float] | None = None  # 先不确定是否能算出最新协方差。
        executed_iterations = 0  # 只统计真正完成线性求解的迭代轮数。

        for iteration in range(self.max_iters):  # 按最大迭代轮数做高斯牛顿。
            normal_matrix, gradient, constraint_count = self._accumulate_normal_system(pose_vector)  # 构造正规方程。
            if constraint_count == 0 and state_count <= 1:  # 没有约束且只有一个节点时就没必要迭代。
                break  # 没有约束就直接停。
            if not np.all(np.isfinite(normal_matrix)) or not np.all(np.isfinite(gradient)):  # 正规方程必须全是有限值。
                raise ValueError("solve system contains non-finite values")  # 发现非法值就直接报错。
            stabilized_normal_matrix = normal_matrix + np.eye(normal_matrix.shape[0], dtype=float) * 1e-8  # 防止奇异。
            try:  # 先尝试直接解线性系统。
                delta = np.linalg.solve(stabilized_normal_matrix, gradient)  # 优先直接求解。
            except np.linalg.LinAlgError:  # 如果直接求解失败就退回最小二乘。
                delta, *_ = np.linalg.lstsq(stabilized_normal_matrix, gradient, rcond=None)  # 退回最小二乘。
            if not np.all(np.isfinite(delta)):  # 增量必须是有限值。
                raise ValueError("solve produced non-finite delta")  # 不接受 nan 或 inf。
            executed_iterations = iteration + 1  # 只有真正求解出增量才记作执行了一轮。
            pose_vector = pose_vector + delta  # 把增量加到当前位姿向量上。
            for state_index in range(state_count):  # 每个节点的 yaw 都要重新归一化。
                yaw_index = (state_index * pose_dim) + 2  # 找到 yaw 所在位置。
                pose_vector[yaw_index] = wrap_angle_rad(float(pose_vector[yaw_index]))  # 防止角度漂移。
            if np.linalg.norm(delta) <= 1e-6:  # 增量足够小时认为收敛。
                break  # 收敛就停止。

        final_normal_matrix, _, _ = self._accumulate_normal_system(pose_vector)  # 最后一轮再算一次正规方程。
        if final_normal_matrix.size:  # 如果最后确实形成了正规方程，就反解协方差。
            if not np.all(np.isfinite(final_normal_matrix)):  # 最后一轮正规矩阵也必须正常。
                raise ValueError("solve system contains non-finite values")  # 不能留下非法数值。
            latest_pose_covariance_diag = self._write_latest_pose_covariance(final_normal_matrix)  # 反解最新协方差。

        for state_index, entry in enumerate(self._window_entries):  # 把优化后的位姿写回每个窗口条目。
            pose_slice = slice(state_index * pose_dim, (state_index + 1) * pose_dim)  # 当前节点切片。
            entry["full_state"] = self._write_pose_into_state(entry["full_state"], pose_vector[pose_slice])  # 回写完整状态。

        latest_state = self._window_entries[-1]["full_state"]  # 最新节点的完整状态。
        self._state = {  # 把最新窗口节点同步成当前状态。
            state_key: float(latest_state[index])  # 每个状态项都取最新节点对应分量。
            for index, state_key in enumerate(state_items)  # 逐个状态项建立字典。
        }  # 当前状态缓存更新完毕。
        self.window_states = [  # 重新生成可读快照。
            {  # 单个窗口条目转成状态字典。
                state_key: float(entry["full_state"][index])  # 取出每个状态分量。
                for index, state_key in enumerate(state_items)  # 逐项展开成字典。
            }  # 单个条目的快照结束。
            for entry in self._window_entries  # 遍历窗口中的每个条目。
        ]  # 全部快照重新生成完毕。
        self._refresh_last_vio_reference_pose_from_window()  # 同步最近 VIO 参考。
        return {  # 返回本轮求解结果。
            "iterations": executed_iterations,  # 只统计真正执行了线性求解的轮数。
            "constraint_count": constraint_count,  # 当前窗口里一共有多少个约束。
            "window_length": len(self._window_entries),  # 当前窗口长度。
            "latest_pose_covariance_diag": latest_pose_covariance_diag,  # 最新位姿协方差对角线。
        }  # 求解结果结构到此结束。

    def _handle_imu(self, payload: dict[str, Any], x_prev: np.ndarray, control) -> dict[str, Any]:
        """处理 IMU 事件，先预测再把运动增量塞进滑窗。

        与 EKF 的区别：预测后不仅更新内部状态和协方差，
        还要计算位姿增量并追加到滑窗末尾。

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
            更新报告，包含 modality、update_applied、window_length 等。
        """
        current_t = float(payload["t"])  # 当前 IMU 时间戳。
        imu_dt = self._effective_imu_dt(payload)  # 只按相邻 IMU 事件计算推进时间。
        # 检查 IMU 数据缺失掩码：与父类 EKFCore._handle_imu 对齐，
        # 如果 ax/ay/gz 中有缺失字段，增大过程噪声以反映预测不确定性，
        # 避免基于补零值产生错误预测。
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
        if imu_dt <= 0.0:  # 非正 dt 说明这次不应该做预测推进。
            latest_entry = self._ensure_latest_window_entry()  # 仍然要刷新最新条目的时间戳。
            latest_entry["timestamp"] = self._timestamp  # 非正 dt 时至少要刷新当前时间戳。
            self._last_imu_timestamp = current_t  # 记录已经消费过这帧 IMU。
            return {  # 直接说明这次没有做预测。
                "modality": "imu",  # 模态标签。
                "update_applied": False,  # 没有真正加入预测。
                "reason": "nonpositive_dt",  # 不接受非正时间步长。
                # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "window_length": len(self._window_entries),  # 当前窗口长度。
                "gate": {"passed": False, "rejected_by": "nonpositive_dt"},
            }  # IMU 非正时间步长的返回块结束。

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

        x_pred, P_pred = run_predict_step(  # 先复用 IMU 预测步。
            x_prev,  # 预测前的状态。
            self._covariance,  # 预测前的协方差。
            payload,  # 当前 IMU 载荷。
            imu_dt,  # IMU 有效时间步长。
            predict_cfg,  # 预测配置，含可能膨胀的过程噪声。
        )  # 预测步执行结束。
        self._update_from_vector(x_pred, P_pred)  # 把预测结果写回内部状态。
        self._last_imu_timestamp = current_t  # 更新最近一次 IMU 时间戳。
        pose_pred = self._pose_from_state_vector(x_pred)  # 缓存预测后位姿。
        pose_prev = self._pose_from_state_vector(x_prev)  # 缓存预测前位姿。
        motion_from_prev = pose_pred - pose_prev  # 计算位姿增量。
        motion_from_prev[2] = angle_delta_rad(  # 航向增量单独按角差计算。
            float(pose_pred[2]),  # 预测后 yaw。
            float(pose_prev[2]),  # 预测前 yaw。
        )  # 航向增量计算结束。
        self._append_window_entry(x_pred, timestamp=self._timestamp, motion_from_prev=motion_from_prev)  # 把预测条目追加进窗口。
        return {  # 返回预测步的窗口写入结果。
            "modality": "imu",  # 模态标签。
            "update_applied": True,  # 预测已成功写入窗口。
            "reason": "predict_step_window_append",  # 说明是预测步并追加了窗口。
            "measurement_control": control_to_dict(control),  # 当前控制参数。
            "window_length": len(self._window_entries),  # 当前窗口长度。
        }  # IMU 预测写入报告结束。

    def _handle_uwb(self, payload: dict[str, Any], x_prev: np.ndarray, control) -> dict[str, Any]:
        """处理 UWB 事件，按质量门槛和鲁棒权重决定是否加入滑窗并求解。

        处理流程：
        1. 控制层跳过检查（``uwb_skip_update``）
        2. 质量门控（``quality < quality_floor`` → 拒绝）
        3. 协方差控制缩放
        4. NIS 门控（``nis > mahalanobis_sq`` → 拒绝）
        5. Huber 鲁棒权重计算 + 协方差膨胀
        6. 构造 UWB 因子并追加到窗口
        7. 立刻求解；求解失败则回滚因子

        参数
        ----------
        payload : dict[str, Any]
            UWB 事件负载。
        x_prev : np.ndarray
            更新前的状态向量。
        control : MeasurementControl
            当前控制对象。

        返回
        -------
        dict[str, Any]
            更新报告，包含 gate、robust、covariance_report、solver_report 等。
        """
        if control.gate_action == "uwb_skip_update":  # 控制层要求跳过时直接返回。
            return {  # 直接说明这次被控制逻辑跳过。
                "modality": "uwb",  # 模态标签。
                "update_applied": False,  # 这次没有真正更新。
                "reason": "uwb_skip_update",  # 门控要求跳过 UWB。
                # §3.0.2 审计：UWB 走 h-side bias 通道（跳过路径同口径声明）。
                "bias_writeport": "h_side",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "window_length": len(self._window_entries),  # 当前窗口长度。
                "gate": {"passed": False, "rejected_by": "uwb_skip_update"},
            }  # UWB 跳过报告结束。
        uwb_payload = payload.get("uwb_payload")  # 提取 UWB 负载。
        if uwb_payload is None:  # 负载缺失时无法执行更新，直接跳过。
            return {  # 返回跳过报告。
                "modality": "uwb",
                "update_applied": False,
                "reason": "missing_uwb_payload",
                # §3.0.2 审计：UWB 走 h-side bias 通道
                "bias_writeport": "h_side",
                "measurement_control": control_to_dict(control),
                "window_length": len(self._window_entries),
                "gate": {"passed": False, "rejected_by": "missing_uwb_payload"},
            }
        quality = self._quality_value(uwb_payload)  # 读取质量值。
        if _quality_below_floor(quality, self._quality_floor("uwb")):  # 质量低于门槛就直接拒绝。
            return {  # 先返回门控失败结果。
                "modality": "uwb",  # 模态标签。
                "update_applied": False,  # 没有通过质量门。
                "reason": "quality_floor",  # 拒绝原因是质量太低。
                # §3.0.2 审计：UWB 走 h-side bias 通道
                "bias_writeport": "h_side",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "gate": {  # 门控子报告。
                    "passed": False,  # 门控未通过。
                    "quality": quality,  # 当前质量值。
                    "quality_floor": self._quality_floor("uwb"),  # 当前模态的质量门槛。
                    "nis": None,  # 这里还没算出 NIS。
                    "mahalanobis_sq_threshold": self._nis_threshold("uwb"),  # 马氏距离阈值。
                    "rejected_by": "quality_floor",  # 拒绝原因。
                },  # gate 子报告结束。
                "window_length": len(self._window_entries),  # 当前窗口长度。
            }  # UWB 质量门失败报告结束。
        base_uwb_noise = self._required_measurement_noise("uwb")  # 读取基础 UWB 噪声（标准差）。
        base_uwb_noise = _square_std_to_var(base_uwb_noise)  # 标准差平方为方差，与过程噪声语义一致。
        effective_noise, cov_report = build_controlled_measurement_cov(  # 先把方差按控制策略缩放。
            base_uwb_noise,  # 方差。
            control,  # 当前控制对象。
            modality="uwb",  # 这是 UWB 处理分支。
            calibration_frozen=self._calibration_frozen,
        )  # 协方差控制报告返回完成。
        anchor_pos = self._resolve_anchor_position(uwb_payload["anchor_id"])  # 解析锚点坐标。
        # §3.0.2 / §3.1.2：bias 在 h(·) 侧注入到 z_pred；残差定义在原始 z 上。
        # 不再对 raw 距离做 subtractive 改写为 corrected_range；raw_range 直接做 z，
        # bias 作为 h 侧常量进入因子残差（residual = z_range - (z_pred + bias)）。
        # §4.4 L1056 协议级护栏（前提指导.md:1049-1063 前置滤波与「干净距离」偷换）：
        # raw_range 不做 max(0,·)/clip 等单方截断；与 ekf_core.py:891 + build_measurement_control
        # L829 同口径 `float(uwb_payload["range"])`，避免 fgo 单方"软削波只给一方"违 §4.4。
        raw_range = float(uwb_payload["range"])
        bias_applied_h = float(control.bias_applied)  # 已由 clip_uwb_bias 截断为非负有界。
        z_pred = predict_range(x_prev, anchor_pos, extra_bias=bias_applied_h)  # h 侧含 bias 的预测测距。
        residual = raw_range - z_pred  # 残差定义在原始 z 上（§3.0.2）。
        H = build_uwb_jacobian(x_prev, anchor_pos)  # 构造雅可比。
        scalar_noise = _coerce_uwb_noise_scalar(effective_noise)  # 取出标量噪声。
        S = coerce_finite_scalar(float((H @ self._covariance @ H.T)[0, 0] + scalar_noise), name="UWB innovation covariance")  # 创新协方差必须是有限数。
        # §11.5 抖动注入：UWB 路径标量 S 与 VIO 路径同口径，缺 jitter fallback 是 v5 漏审。
        # 三方法（EKF / Robust-EKF / FGO）UWB 路径 S<=0 拒绝前先尝试 jitter 修补；
        # 二次仍失败 → fail-loud（与 VIO 路径 _ensure_positive_definite_vio_innovation_covariance 同政策）。
        if S <= 0.0:  # 非正创新协方差先尝试 jitter 修补。
            from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
            cov_jitter_eps = float(BRIDGE_THRESHOLDS["cov_jitter_eps"])
            jittered_S = S + cov_jitter_eps
            if jittered_S > 0.0:
                S = jittered_S  # 接受 jittered 版本作为该次创新协方差。
            else:
                return {  # 直接返回创新协方差失效的门控结果。
                    "modality": "uwb",  # 模态标签。
                    "update_applied": False,  # 本次不允许入图。
                    "reason": "nonpositive_innovation_covariance",  # 拒绝原因。
                    # §3.0.2 审计：UWB 走 h-side bias 通道
                    "bias_writeport": "h_side",
                    "measurement_control": control_to_dict(control),  # 当前控制参数。
                    "covariance_report": cov_report,  # 基础协方差报告。
                    "gate": {  # 门控子报告。
                        "passed": False,  # 门控未通过。
                        "quality": quality,  # 当前质量值。
                        "quality_floor": self._quality_floor("uwb"),  # 当前模态的质量门槛。
                        "nis": None,  # 非正协方差下不再定义 NIS。
                        "mahalanobis_sq_threshold": self._nis_threshold("uwb"),  # 当前门限。
                        "rejected_by": "nonpositive_innovation_covariance",  # 拒绝原因。
                    },  # gate 子报告结束。
                    "window_length": len(self._window_entries),  # 当前窗口长度。
                }  # UWB 非正创新协方差拒绝报告结束。
        nis = float((residual * residual) / S)  # 计算 NIS。
        if not math.isfinite(nis):  # NaN/inf NIS 会绕过门控（NaN > 阈值为 False），必须显式拒绝。
            return {  # 与 RobustEKFCore 对齐：非有限 NIS 直接拒绝，不让 nan/inf 进入 Huber 权重链路。
                "modality": "uwb",  # 模态标签。
                "update_applied": False,  # 本次不允许入图。
                "reason": "nonfinite_nis",  # 拒绝原因。
                # §3.0.2 审计：UWB 走 h-side bias 通道
                "bias_writeport": "h_side",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "covariance_report": cov_report,  # 基础协方差报告。
                "gate": {  # 门控子报告。
                    "passed": False,  # 门控未通过。
                    "quality": quality,  # 当前质量值。
                    "quality_floor": self._quality_floor("uwb"),  # 当前模态的质量门槛。
                    "nis": nis,  # 当前 NIS（nan 或 inf）。
                    "mahalanobis_sq_threshold": self._nis_threshold("uwb"),  # 当前门限。
                    "rejected_by": "nonfinite_nis",  # 拒绝原因。
                },  # gate 子报告结束。
                "window_length": len(self._window_entries),  # 当前窗口长度。
            }  # UWB 非有限 NIS 拒绝报告结束。
        if nis > self._nis_threshold("uwb"):  # NIS 超门限说明该量测太离谱。
            return {  # 先返回马氏距离门失败结果。
                "modality": "uwb",  # 模态标签。
                "update_applied": False,  # 没有通过马氏距离门。
                "reason": "mahalanobis_sq",  # 拒绝原因是 NIS 太大。
                # §3.0.2 审计：UWB 走 h-side bias 通道
                "bias_writeport": "h_side",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "covariance_report": cov_report,  # 基础协方差报告。
                "gate": {  # 门控子报告。
                    "passed": False,  # 门控未通过。
                    "quality": quality,  # 当前质量值。
                    "quality_floor": self._quality_floor("uwb"),  # 当前模态的质量门槛。
                    "nis": nis,  # 当前 NIS。
                    "mahalanobis_sq_threshold": self._nis_threshold("uwb"),  # 阈值。
                    "rejected_by": "mahalanobis_sq",  # 拒绝原因。
                },  # gate 子报告结束。
                "window_length": len(self._window_entries),  # 当前窗口长度。
            }  # UWB 马氏距离门失败报告结束。
        whitened = math.sqrt(max(nis, 0.0))  # 白化残差范数；Inf→sqrt(Inf)=Inf→Huber最大降权。
        robust_weight = self._huber_weight(whitened)  # 计算鲁棒权重。
        covariance_scale = 1.0 / max(robust_weight, 1e-6)  # 转成协方差缩放因子。
        robust_noise, robust_cov_report = build_effective_cov(  # 再根据鲁棒权重缩放噪声。
            effective_noise,  # 先用有效噪声作为基底。
            uwb_scaling=covariance_scale,  # 再乘鲁棒缩放。
        )  # 鲁棒协方差报告返回完成。
        noise_value = _coerce_uwb_noise_scalar(robust_noise, path="robust_uwb_noise")  # 鲁棒后的噪声仍需是标量。
        constraint = {  # 当前 UWB 因子的内部表示。
            "type": "uwb",  # 因子类型。
            "anchor_pos": anchor_pos,  # 锚点坐标。
            "z_range": raw_range,  # 原始测距；残差在原始 z 合同上定义（§3.0.2）。
            "extra_bias": bias_applied_h,  # h 侧有界偏置修正（§3.0.2 / §3.1.2）；进 h(·)，不重写 z。
            "noise": noise_value,  # 最终标量噪声。
        }  # UWB 因子约束字典结束。
        latest_entry = self._ensure_latest_window_entry()  # 取当前最新窗口条目。
        entry_factor_count = len(latest_entry["uwb_factors"])  # 记录追加前的因子数量。
        total_constraint_count = len(self.constraints)  # 记录追加前的总约束数。
        self._append_constraint(constraint)  # 把当前 UWB 因子挂进去。
        try:  # 挂进去之后立刻求解，失败就回滚刚加的约束。
            solve_report = self.solve()  # 立刻求解整个窗口。
        except (ValueError, np.linalg.LinAlgError) as exc:  # 求解失败说明本次约束不可接受。
            self._rollback_latest_constraint(  # 求解失败时回滚本次 UWB 因子。
                constraint_type="uwb",  # 要回滚的约束类型是 UWB。
                entry_factor_count=entry_factor_count,  # 回滚到追加前的 UWB 因子数量。
                total_constraint_count=total_constraint_count,  # 回滚到追加前的总约束数量。
            )  # 回滚调用结束。
            raise  # 继续把原始异常抛给上层。
        except Exception:  # 非预期异常：回滚后记录日志再抛出。
            self._rollback_latest_constraint(
                constraint_type="uwb",
                entry_factor_count=entry_factor_count,
                total_constraint_count=total_constraint_count,
            )
            _logger.exception("UWB update: unexpected exception during solve, rolling back")
            raise
        self._refresh_last_vio_reference_pose_from_window()  # 求解后同步 VIO 参考。
        # §2 细节 23 五方法同模型——FGO 节点 3 维不含 uwb_clock_bias，用旁路 EKF 让
        # self._state['uwb_clock_bias'] 在线辨识，与 EKF 族保持「同在线辨识」对等。
        # 在 solve 后调用，让因子图先更新 pose 与 pose 协方差，旁路仅更新 uwb_clock_bias
        # 标量与 covariance[8,8]，避免与因子图 pose 优化结果冲突。
        self._apply_uwb_clock_bias_kalman_update(residual, scalar_noise)
        return {  # 返回 UWB 成功更新结果。
            "modality": "uwb",  # 模态标签。
            "update_applied": True,  # 这次 UWB 被成功加入窗口并求解。
            "measurement_control": control_to_dict(control),  # 当前控制参数。
            "covariance_report": cov_report,  # 基础协方差报告。
            "robust_covariance_report": robust_cov_report,  # 鲁棒缩放后的协方差报告。
            # §3.0.2 审计：显式标记本次 bias 写入 h(·) 侧，残差定义在原始 z 上。
            "bias_writeport": "h_side",
            "gate": {  # 门控子报告。
                "passed": True,  # 门控通过。
                "quality": quality,  # 当前质量值。
                "quality_floor": self._quality_floor("uwb"),  # 当前模态的质量门槛。
                "nis": nis,  # 当前 NIS。
                "mahalanobis_sq_threshold": self._nis_threshold("uwb"),  # 阈值。
                "rejected_by": None,  # 没有拒绝。
            },  # 门控子报告结束。
            "robust": self._robust_report(whitened, robust_weight, covariance_scale),  # 无核身份报告。
            "residual": float(residual),  # UWB 观测残差，供 fusion_runner 更新 last_innovation_norm。
            "constraint": copy.deepcopy(constraint),  # 本次真正加入的约束副本。
            "solver_report": solve_report,  # 优化器求解结果。
        }  # UWB 成功更新报告结束。

    def _handle_vio(self, payload: dict[str, Any], x_prev: np.ndarray, control) -> dict[str, Any]:
        """处理 VIO 事件，按参考位姿构造相对因子并加入滑窗。

        处理流程与 UWB 一致：
        1. 控制层跳过检查（``vio_skip_update``）
        2. quality<=0 协议级检查（跳过并重置参考位姿）
        3. 质量门控
        4. 首帧参考位姿初始化检查
        5. 参考位姿时效性检查
        6. 协方差控制缩放
        7. NIS 门控
        8. Huber 鲁棒权重计算 + 协方差膨胀
        9. 构造 VIO 因子并追加到窗口
        10. 立刻求解；求解失败则回滚因子

        参数
        ----------
        payload : dict[str, Any]
            VIO 事件负载。
        x_prev : np.ndarray
            更新前的状态向量。
        control : MeasurementControl
            当前控制对象。

        返回
        -------
        dict[str, Any]
            更新报告，包含 gate、robust、covariance_report、solver_report 等。

        注意
        -----
        更新成功后会刷新 ``_last_vio_reference_pose`` 和 ``_last_vio_reference_index``。
        """
        if control.gate_action == "vio_skip_update":  # 控制层要求跳过时直接返回。
            # 跳过时重置参考位姿为当前估计位姿，与 EKF 行为一致。
            # 仿真 VIO 数据提供相邻帧增量，跳过后下一帧 VIO 仍是相邻帧增量，
            # 参考位姿必须重置为当前位姿，否则残差语义不匹配。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_index = len(self._window_entries) - 1 if self._window_entries else None
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {  # 直接说明这次被控制逻辑跳过。
                "modality": "vio",  # 模态标签。
                "update_applied": False,  # 这次没有加入 VIO 因子。
                "reason": "vio_skip_update",  # 门控要求跳过 VIO。
                # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "window_length": len(self._window_entries),  # 当前窗口长度。
                "gate": {"passed": False, "rejected_by": "vio_skip_update"},
            }  # VIO 跳过报告结束。
        quality = self._quality_value(payload.get("vio_payload"))  # 先从 VIO 负载里取质量值。
        # quality<=0 的 VIO 事件语义上是无效观测（如仿真 cycle 边界帧），跳过更新并重置参考位姿。
        # 与 ekf_core 行为一致：quality<=0 是协议级检查，不受 quality_floor 配置影响。
        if quality <= 0.0:
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_index = len(self._window_entries) - 1 if self._window_entries else None
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {
                "modality": "vio",
                "update_applied": False,
                "reason": "vio_quality_zero",
                # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),
                "gate": {"passed": False, "quality": 0.0, "rejected_by": "vio_quality_zero"},
                "window_length": len(self._window_entries),
            }
        if _quality_below_floor(quality, self._quality_floor("vio")):  # 质量低于门槛就先拒绝。
            # VIO 量测被拒绝时重置参考位姿为当前估计位姿，与 EKF 行为一致。
            # 仿真 VIO 数据提供相邻帧增量，拒绝后下一帧 VIO 仍是相邻帧增量，
            # 参考位姿必须重置为当前位姿，否则残差语义不匹配。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_index = len(self._window_entries) - 1 if self._window_entries else None
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {  # 先返回门控失败结果。
                "modality": "vio",  # 模态标签。
                "update_applied": False,  # 没有通过质量门。
                "reason": "quality_floor",  # 拒绝原因是质量太低。
                # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "gate": {  # 门控子报告。
                    "passed": False,  # 门控未通过。
                    "quality": quality,  # 当前质量值。
                    "quality_floor": self._quality_floor("vio"),  # 当前模态的质量门槛。
                    "nis": None,  # 这里还没算出 NIS。
                    "mahalanobis_sq_threshold": self._nis_threshold("vio"),  # 阈值。
                    "rejected_by": "quality_floor",  # 拒绝原因。
                },  # VIO 质量门的 gate 子报告结束。
                "window_length": len(self._window_entries),  # 当前窗口长度。
            }  # VIO 质量门失败报告结束。
        base_vio_noise = self._required_measurement_noise("vio")  # 读取基础 VIO 噪声（标准差）。
        base_vio_noise = _square_std_to_var(base_vio_noise)  # 标准差平方为方差，与过程噪声语义一致。
        effective_cov, cov_report = build_controlled_measurement_cov(  # 先把方差按控制策略缩放。
            base_vio_noise,  # 方差。
            control,  # 当前控制对象。
            modality="vio",  # 这是 VIO 处理分支。
            calibration_frozen=self._calibration_frozen,
        )  # 协方差控制报告返回完成。
        # 首个 VIO 事件参考位姿检查：如果 _last_vio_reference_pose 为 None，
        # _resolve_latest_vio_reference 会回退到当前估计位姿，导致 z_hat=0 而
        # z_vio 非零，残差异常放大。此时跳过更新，与 EKF/RobustEKF 行为一致。
        if self._last_vio_reference_pose is None:
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_index = len(self._window_entries) - 1 if self._window_entries else None
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {
                "modality": "vio",
                "update_applied": False,
                "reason": "vio_first_frame_reference_init",
                # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),
                "gate": {"passed": False, "rejected_by": "vio_first_frame_reference_init"},
                "window_length": len(self._window_entries),
            }
        # 参考位姿时效性检查：如果参考位姿超过阈值未更新，说明中间可能有 VIO 事件
        # 被 blackout 移除或门控跳过。仿真 VIO 数据提供相邻帧增量，参考位姿过时时
        # 增量语义不匹配（相邻帧增量 ≠ 参考帧增量），此时应重置参考位姿并跳过更新，
        # 避免残差方向错误导致定位精度退化。与 EKF 行为一致。
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
            self._last_vio_reference_index = len(self._window_entries) - 1 if self._window_entries else None
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {
                "modality": "vio",
                "update_applied": False,
                "reason": "vio_reference_pose_stale",
                # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),
                "gate": {"passed": False, "rejected_by": "vio_reference_pose_stale"},
                "window_length": len(self._window_entries),
            }
        reference_pose, reference_index = self._resolve_latest_vio_reference()  # 取最近参考位姿。
        z_vio = build_vio_measurement(payload)  # 从事件里提取视觉量测。
        z_hat, residual, H, reference_pose_array = compute_vio_residual(  # 计算量测残差及雅可比。
            x_prev,  # 当前状态。
            z_vio,  # 视觉量测。
            reference_pose=reference_pose,  # 参考位姿。
        )  # VIO 线性化结果返回完成。
        R_vio = _normalize_vio_covariance(effective_cov)  # 把噪声规整成矩阵。
        S = H @ self._covariance @ H.T + R_vio  # 计算创新协方差。
        if not np.all(np.isfinite(S)):  # 创新协方差必须全是有限数。
            raise ValueError("VIO innovation covariance must be finite")  # 不能把 nan 或 inf 带进更新。
        try:
            S = _ensure_positive_definite_vio_innovation_covariance(
                S,
                name="VIO innovation covariance",
            )
        except ValueError:  # 非正定时，VIO NIS 和鲁棒缩放都失去物理语义。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_index = len(self._window_entries) - 1 if self._window_entries else None
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {
                "modality": "vio",
                "update_applied": False,
                "reason": "nonpositive_innovation_covariance",
                # §3.0.2 audit: reject path declares write-port.
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),
                "covariance_report": cov_report,
                "gate": {
                    "passed": False,
                    "quality": quality,
                    "quality_floor": self._quality_floor("vio"),
                    "nis": None,
                    "mahalanobis_sq_threshold": self._nis_threshold("vio"),
                    "rejected_by": "nonpositive_innovation_covariance",
                },
                "z_vio": z_vio.tolist(),
                "z_hat": z_hat.tolist(),
                "reference_pose": reference_pose_array.tolist(),
                "residual": residual.tolist(),
                "window_length": len(self._window_entries),
            }
        try:
            nis = float(residual.T @ np.linalg.solve(S, residual))  # 计算 NIS。
        except np.linalg.LinAlgError:  # S 不正定时回退。
            # VIO 量测被拒绝时重置参考位姿为当前估计位姿，与 EKF 行为一致。
            # 仿真 VIO 数据提供相邻帧增量，拒绝后下一帧 VIO 仍是相邻帧增量，
            # 参考位姿必须重置为当前位姿，否则残差语义不匹配。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_index = len(self._window_entries) - 1 if self._window_entries else None
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {  # 返回创新协方差不可解的拒绝报告。
                "modality": "vio",  # 模态标签。
                "update_applied": False,  # 没有通过。
                "reason": "nonpositive_innovation_covariance",  # 拒绝原因是 S 不可解。
                # §3.0.2 audit: VIO NIS unsolvable declares "none".
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "covariance_report": cov_report,  # 基础协方差报告。
                "gate": {  # 门控子报告。
                    "passed": False,  # 门控未通过。
                    "quality": quality,  # 当前质量值。
                    "quality_floor": self._quality_floor("vio"),  # 当前模态的质量门槛。
                    "nis": None,  # NIS 无法计算。
                    "mahalanobis_sq_threshold": self._nis_threshold("vio"),  # 阈值。
                    "rejected_by": "nonpositive_innovation_covariance",  # 拒绝原因。
                },  # VIO 创新协方差不可解的 gate 子报告结束。
                "z_vio": z_vio.tolist(),  # 原始视觉量测。
                "z_hat": z_hat.tolist(),  # 预测视觉量测。
                "reference_pose": reference_pose_array.tolist(),  # 参考位姿。
                "residual": residual.tolist(),  # 残差。
                "window_length": len(self._window_entries),  # 当前窗口长度。
            }  # VIO 创新协方差不可解的拒绝报告结束。
        if not math.isfinite(nis):  # NaN/inf NIS 会绕过门控（NaN > 阈值为 False），必须显式拒绝。与 RobustEKFCore 行为一致。
            # VIO 量测被拒绝时重置参考位姿为当前估计位姿，与 EKF 行为一致。
            # 仿真 VIO 数据提供相邻帧增量，拒绝后下一帧 VIO 仍是相邻帧增量，
            # 参考位姿必须重置为当前位姿，否则残差语义不匹配。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_index = len(self._window_entries) - 1 if self._window_entries else None
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {  # 返回非有限 NIS 的拒绝报告。
                "modality": "vio",  # 模态标签。
                "update_applied": False,  # 没有通过。
                "reason": "nonfinite_nis",  # 拒绝原因是 NIS 非有限。
                # §3.0.2 审计：VIO 不接受 NN bias。
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "covariance_report": cov_report,  # 基础协方差报告。
                "gate": {  # 门控子报告。
                    "passed": False,  # 门控未通过。
                    "quality": quality,  # 当前质量值。
                    "quality_floor": self._quality_floor("vio"),  # 当前模态的质量门槛。
                    "nis": nis,  # 当前 NIS（NaN 或 inf）。
                    "mahalanobis_sq_threshold": self._nis_threshold("vio"),  # 阈值。
                    "rejected_by": "nonfinite_nis",  # 拒绝原因。
                },  # VIO 非有限 NIS 的 gate 子报告结束。
                "z_vio": z_vio.tolist(),  # 原始视觉量测。
                "z_hat": z_hat.tolist(),  # 预测视觉量测。
                "reference_pose": reference_pose_array.tolist(),  # 参考位姿。
                "residual": residual.tolist(),  # 残差。
                "window_length": len(self._window_entries),  # 当前窗口长度。
            }  # VIO 非有限 NIS 的拒绝报告结束。
        if nis > self._nis_threshold("vio"):  # NIS 超门限说明该量测太离谱。
            # VIO 量测被 NIS 拒绝时重置参考位姿为当前估计位姿，与 EKF 行为一致。
            # 仿真 VIO 数据提供相邻帧增量，拒绝后下一帧 VIO 仍是相邻帧增量，
            # 参考位姿必须重置为当前位姿，否则残差语义不匹配。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_index = len(self._window_entries) - 1 if self._window_entries else None
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {  # 先返回马氏距离门失败结果。
                "modality": "vio",  # 模态标签。
                "update_applied": False,  # 没有通过马氏距离门。
                "reason": "mahalanobis_sq",  # 拒绝原因是 NIS 太大。
                # §3.0.2 审计：VIO 不接受 NN bias。
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "covariance_report": cov_report,  # 基础协方差报告。
                "gate": {  # 门控子报告。
                    "passed": False,  # 门控未通过。
                    "quality": quality,  # 当前质量值。
                    "quality_floor": self._quality_floor("vio"),  # 当前模态的质量门槛。
                    "nis": nis,  # 当前 NIS。
                    "mahalanobis_sq_threshold": self._nis_threshold("vio"),  # 阈值。
                    "rejected_by": "mahalanobis_sq",  # 拒绝原因。
                },  # VIO 马氏距离门的 gate 子报告结束。
                "z_vio": z_vio.tolist(),  # 原始视觉量测。
                "z_hat": z_hat.tolist(),  # 预测视觉量测。
                "reference_pose": reference_pose_array.tolist(),  # 参考位姿。
                "residual": residual.tolist(),  # 残差。
                "window_length": len(self._window_entries),  # 当前窗口长度。
            }  # VIO 马氏距离门失败报告结束。
        whitened = math.sqrt(max(nis, 0.0))  # 白化残差范数。
        robust_weight = self._huber_weight(whitened)  # 计算鲁棒权重。
        covariance_scale = 1.0 / max(robust_weight, 1e-6)  # 转成协方差缩放因子。
        robust_vio_cov, robust_cov_report = build_effective_cov(  # 再根据鲁棒权重缩放协方差。
            effective_cov,  # 先用有效协方差。
            vio_scaling=covariance_scale,  # 再应用鲁棒缩放。
        )  # 鲁棒协方差报告返回完成。
        if reference_index is not None and reference_index >= len(self._window_entries):  # 若参考索引超界，说明它被裁掉了。
            reference_index = None  # 参考索引如果已经超界，就清空。
        # §6.4 #18 紧耦合角色硬子要求对应代码层落点：
        # VIO 因子挂增量残差 (z_vio - z_hat(参考帧 x_ref)) 与 UWB 距离残差并列挂滑窗
        # (_handle_uwb L1882 _append_constraint 同口径挂 uwb 因子)，而非先视觉里程计积分
        # 成轨迹再松耦合 (§6.4 #19)。三估计器同口径：EKF ekf_core.py:1193 单帧 Kalman 更新 /
        # RobustEKF robust_ekf_core.py:694 单帧 Kalman 更新 / FGO 本处 L2210-2224 滑窗因子挂载。
        # 参考帧通过 reference_index + reference_pose 双锚定，残差语义为"上一帧→当前帧"增量，
        # 不泄漏未来关键帧 (§6 边缘化合同)，与 EKF 族单帧增量叙事兼容 (§6 细节预积分交叉)。
        constraint = {  # 当前 VIO 因子的内部表示。
            "type": "vio",  # 因子类型。
            "z_vio": z_vio.tolist(),  # 视觉量测。
            "noise": _normalize_vio_covariance(robust_vio_cov).tolist(),  # 最终噪声矩阵。
            "reference_pose": [  # 把参考位姿向量拆成三个标量，方便序列化。
                float(reference_pose_array[0]),  # 参考 x 坐标。
                float(reference_pose_array[1]),  # 参考 y 坐标。
                float(reference_pose_array[2]),  # 参考 yaw。
            ],  # 参考位姿数组到此结束。
            "reference_index": reference_index,  # 参考位姿索引，若已经被裁剪则为空。
        }  # VIO 因子约束字典结束。
        latest_entry = self._ensure_latest_window_entry()  # 取当前最新窗口条目。
        entry_factor_count = len(latest_entry["vio_factors"])  # 记录追加前因子数量。
        total_constraint_count = len(self.constraints)  # 记录追加前总约束数。
        self._append_constraint(constraint)  # 把当前 VIO 因子挂进去。
        try:  # 挂进去之后立刻求解，失败就回滚刚加的约束。
            solve_report = self.solve()  # 立刻优化整个窗口。
        except (ValueError, np.linalg.LinAlgError) as exc:  # 求解失败说明本次约束不可接受。
            self._rollback_latest_constraint(  # 求解失败时回滚本次 VIO 因子。
                constraint_type="vio",  # 要回滚的约束类型是 VIO。
                entry_factor_count=entry_factor_count,  # 回滚到追加前的 VIO 因子数量。
                total_constraint_count=total_constraint_count,  # 回滚到追加前的总约束数量。
            )  # 回滚调用结束。
            raise  # 继续把原始异常抛给上层。
        except Exception:  # 非预期异常：回滚后记录日志再抛出。
            self._rollback_latest_constraint(
                constraint_type="vio",
                entry_factor_count=entry_factor_count,
                total_constraint_count=total_constraint_count,
            )
            _logger.exception("VIO update: unexpected exception during solve, rolling back")
            raise
        self._last_vio_reference_pose = self._current_pose_reference()  # 更新最近参考位姿。
        self._last_vio_reference_index = len(self._window_entries) - 1 if self._window_entries else None  # 更新参考索引。
        self._last_vio_reference_pose_timestamp = self._timestamp  # 更新参考位姿时间戳。
        # §2 细节 23 五方法同模型——FGO 节点 3 维不含 vio_scale，用旁路 EKF 让
        # self._state['vio_scale'] 在线辨识，与 EKF 族保持「同在线辨识」对等。
        # 在 solve 后调用，让因子图先更新 pose 与 pose 协方差，旁路仅更新 vio_scale
        # 标量与 covariance[9,9]，避免与因子图 pose 优化结果冲突。
        self._apply_vio_scale_kalman_update(residual, R_vio)
        return {  # 返回 VIO 成功更新结果。
            "modality": "vio",  # 模态标签。
            "update_applied": True,  # 这次 VIO 被成功加入窗口并求解。
            "measurement_control": control_to_dict(control),  # 当前控制参数。
            "covariance_report": cov_report,  # 基础协方差报告。
            "robust_covariance_report": robust_cov_report,  # 鲁棒缩放后的协方差报告。
            "gate": {  # 门控子报告。
                "passed": True,  # 门控通过。
                "quality": quality,  # 当前质量值。
                "quality_floor": self._quality_floor("vio"),  # 当前模态的质量门槛。
                "nis": nis,  # 当前 NIS。
                "mahalanobis_sq_threshold": self._nis_threshold("vio"),  # 阈值。
                "rejected_by": None,  # 没有拒绝。
            },  # 门控子报告结束。
            "robust": self._robust_report(whitened, robust_weight, covariance_scale),  # 无核身份报告。
            # §3.0.2 审计要求：VIO 路径不接受 NN bias（build_measurement_control 在
            # VIO 分支强制 bias_applied=0.0，残差定义在原始增量 z 上）。标记 "none"
            # 显式说明本次 VIO 更新未触发 bias 写入口；与 EKF/Robust VIO 路口径一致。
            "bias_writeport": "none",
            "z_vio": z_vio.tolist(),  # 原始视觉量测。
            "z_hat": z_hat.tolist(),  # 预测视觉量测。
            "reference_pose": reference_pose_array.tolist(),  # 参考位姿。
            "residual": residual.tolist(),  # 残差。
            "constraint": copy.deepcopy(constraint),  # 本次加入的约束副本。
            "solver_report": solve_report,  # 优化器求解结果。
        }  # VIO 成功更新报告结束。
