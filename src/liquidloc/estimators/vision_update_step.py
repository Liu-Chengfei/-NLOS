"""
文件：src/liquidloc/estimators/vision_update_step.py

【文件职责】
这个文件实现纯 VIO 相对位姿更新。
它负责把视觉事件里的相对位移和相对航向，作为一次标准 EKF 观测来更新状态。

【本文件绝对不负责】
不做视觉质量门控，不做学习型缩放，不直接管理整条 EKF 生命周期。

【上游依赖】
estimators/state_definition.py、sensors/vision_model.py、common/validation.py。

【下游调用者】
estimators/ekf_core.py、fusion 相关流程、测试文件。
"""

from __future__ import annotations  # 允许后面的函数相互引用类型。

from collections.abc import Mapping, Sequence  # 用来判断映射型和序列型输入。
import copy  # 用来深拷贝映射型状态。
import math  # 用来做三角函数和有限性检查。
import numpy as np  # 用来做向量和矩阵运算。
from typing import Any  # 用来标注任意类型。

from liquidloc.common.angle_utils import angle_delta_rad, wrap_angle_rad  # 用来处理角度差和角度归一化。
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_real, require_not_none, require_shape  # 集中判断 bool / np.bool_ / 实数类型，以及基础校验。
from liquidloc.common.constants import VIO_MEASUREMENT_ITEMS  # VIO 测量项常量（单源真相）。
from liquidloc.protocol.task_contract import get_state_items, get_vio_update_contract
from liquidloc.sensors.vision_model import extract_vio_measurement  # 用来从事件中提取 VIO 测量。

# The full EKF state order is frozen by configs/base/task.yaml.
_FULL_STATE_ITEMS = get_state_items()  # 完整状态顺序。
_POSE_ONLY_STATE_ITEMS = tuple(get_vio_update_contract()["updated_state_items"])  # 只包含位姿的简化顺序。


def _resolve_state_items_from_mapping(x_pred: Mapping[str, Any]) -> tuple[str, ...]:
    """从映射型状态里判断它是完整状态还是仅位姿状态。

    参数
    ----
    x_pred : Mapping[str, Any]
        映射型预测状态。

    返回
    ----
    tuple[str, ...]
        状态键名元组，为 ``_FULL_STATE_ITEMS`` 或 ``_POSE_ONLY_STATE_ITEMS``。

    异常
    ----
    KeyError
        既不包含完整状态键也不包含位姿键。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"x_pred": x_pred}, "_resolve_state_items_from_mapping 入参", prefix="[配置]")
    if all(key in x_pred for key in _FULL_STATE_ITEMS):  # 如果完整状态键都在，就按完整状态处理。
        return _FULL_STATE_ITEMS  # 返回完整顺序。
    if all(key in x_pred for key in _POSE_ONLY_STATE_ITEMS):  # 如果只有位姿键，也允许处理。
        return _POSE_ONLY_STATE_ITEMS  # 返回简化顺序。
    raise KeyError("x_pred must expose either full state keys or pose-only keys: px, py, yaw")  # 否则直接报错。


def _resolve_state_items_from_vector(x_pred_array: np.ndarray) -> tuple[str, ...]:
    """从向量长度判断它对应完整状态还是仅位姿状态。

    参数
    ----
    x_pred_array : np.ndarray
        一维状态向量。

    返回
    ----
    tuple[str, ...]
        状态键名元组，为 ``_FULL_STATE_ITEMS`` 或 ``_POSE_ONLY_STATE_ITEMS``。

    异常
    ----
    ValueError
        向量非一维或长度不匹配任何已知状态。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"x_pred_array": x_pred_array}, "_resolve_state_items_from_vector 入参", prefix="[配置]")
    if x_pred_array.ndim != 1:  # 这里只接受一维状态向量。
        raise ValueError(f"x_pred must be a 1D state vector, got shape {x_pred_array.shape}")  # 结构不对就报错。
    if x_pred_array.shape[0] == len(_FULL_STATE_ITEMS):  # 长度匹配完整状态。
        return _FULL_STATE_ITEMS  # 返回完整状态顺序。
    if x_pred_array.shape[0] == len(_POSE_ONLY_STATE_ITEMS):  # 长度匹配仅位姿状态。
        return _POSE_ONLY_STATE_ITEMS  # 返回简化状态顺序。
    raise ValueError(  # 其他长度都不匹配约定。
        "x_pred vector length must match either the frozen full EKF state or the minimal pose-only state"
    )


def _coerce_numeric_scalar(value: Any, *, name: str) -> float:
    """把输入规整成有限数值标量。

    参数
    ----
    value : Any
        待规整的值。
    name : str
        参数名称，用于错误信息。

    返回
    ----
    float
        规整后的有限浮点数。

    异常
    ----
    TypeError
        值为 ``bool``、非标量或非数值类型（含 ``complex``）。
    ValueError
        值为 NaN 或 Inf。

    注意
    ----
    - 显式拒绝 ``bool`` 和 0-dim ndarray 包装的布尔值。
    - 拒绝 ``complex``（虽是 ``Number`` 子类但不可转 float）。
    - 接受 ``np.float64`` 等 numpy 标量类型。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"value": value, "name": name}, "_coerce_numeric_scalar 入参", prefix="[配置]")
    if isinstance(value, np.ndarray):  # 显式处理 ndarray，仅接受 0 维标量，防止数组穿透到标量计算中。
        if value.shape != ():  # 非 0 维数组不是标量。
            raise TypeError(f"{name} must be a scalar numeric value")  # 不是标量就报错。
        value = value.item()  # 取出原始标量值。
    # 委托中心入口：bool/complex/有限性校验，与 uwb_update_step._coerce_scalar 口径对齐。
    return coerce_finite_scalar(value, name=name)


def _coerce_state_vector(
    x_pred: np.ndarray | Sequence[float] | Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """把输入状态规整成 ndarray，并返回它对应的状态顺序。

    参数
    ----
    x_pred : np.ndarray | Sequence[float] | Mapping[str, Any]
        预测状态。映射型必须包含完整状态键或位姿键，
        数组型长度必须为 8（完整）或 3（位姿）。

    返回
    ----
    x_pred_array : np.ndarray
        一维浮点状态向量（副本）。
    state_items : tuple[str, ...]
        对应的状态键名元组。

    异常
    ----
    ValueError
        ``x_pred`` 为 ``None`` 或维度不匹配。
    TypeError
        映射中某值为非法类型。
    KeyError
        映射缺少必要键。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"x_pred": x_pred}, "_coerce_state_vector 入参", prefix="[配置]")
    require_not_none(x_pred, "x_pred")  # 预测状态不能为空。

    if isinstance(x_pred, Mapping):  # 映射型输入按键名读取。
        state_items = _resolve_state_items_from_mapping(x_pred)  # 先确定状态顺序。
        x_pred_array = np.asarray([_coerce_numeric_scalar(x_pred[key], name=f"x_pred[{key!r}]") for key in state_items], dtype=float)  # 按顺序拼数组。
        return x_pred_array, state_items  # 返回数组和对应顺序。

    x_pred_array = np.asarray(x_pred, dtype=float)  # 其余情况按数组处理。
    state_items = _resolve_state_items_from_vector(x_pred_array)  # 再判断属于哪种状态顺序。
    return x_pred_array.copy(), state_items  # 返回副本，避免污染输入。


def _restore_state_like_input(
    x_pred: np.ndarray | Mapping[str, Any],
    x_upd_array: np.ndarray,
    state_items: tuple[str, ...],
) -> np.ndarray | dict[str, Any]:
    """把更新后的状态恢复成和输入一致的承载形式。

    参数
    ----
    x_pred : np.ndarray | Mapping[str, Any]
        原始预测状态（用于判断承载类型）。
    x_upd_array : np.ndarray
        更新后的一维状态向量。
    state_items : tuple[str, ...]
        状态键名元组。

    返回
    ----
    np.ndarray | dict[str, Any]
        与 ``x_pred`` 形式一致的状态载体。

    注意
    ----
    - 映射型输入返回深拷贝字典，避免内嵌可变对象共享引用。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"x_pred": x_pred, "x_upd_array": x_upd_array, "state_items": state_items}, "_restore_state_like_input 入参", prefix="[配置]")
    if isinstance(x_pred, Mapping):  # 如果输入是映射，就回写到字典里。
        x_upd = copy.deepcopy(x_pred)  # 深拷贝原映射，避免内嵌可变对象共享引用。
        for index, key in enumerate(state_items):  # 按状态顺序逐项回写。
            x_upd[key] = float(x_upd_array[index])  # 写回更新后的值。
        return x_upd  # 返回字典形式。
    return x_upd_array  # 否则直接返回数组。


def _extract_pose_vector(x_array: np.ndarray, state_items: tuple[str, ...]) -> np.ndarray:
    """从完整状态或位姿状态里抽出 px、py、yaw。

    参数
    ----
    x_array : np.ndarray
        一维状态向量。
    state_items : tuple[str, ...]
        状态键名元组，必须包含 ``"px"``、``"py"``、``"yaw"``。

    返回
    ----
    np.ndarray
        长度 3 的浮点向量 ``[px, py, yaw]``。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"x_array": x_array, "state_items": state_items}, "_extract_pose_vector 入参", prefix="[配置]")
    return np.asarray(
        [
            x_array[state_items.index("px")],  # 位置 x。
            x_array[state_items.index("py")],  # 位置 y。
            x_array[state_items.index("yaw")],  # 航向角。
        ],
        dtype=float,
    )


def _rotation_world_to_reference(reference_yaw: float) -> np.ndarray:
    """构造从世界坐标系到参考航向局部坐标系的旋转矩阵。

    该旋转将世界坐标系下的平移向量转换到参考航向定义的局部坐标系中，
    即 R(-yaw_ref) @ [dx_world, dy_world]^T = [dx_local, dy_local]^T
    （等价于 R(yaw_ref)^T @ ...，其中 R(θ)=[[cosθ,-sinθ],[sinθ,cosθ]]
    为标准逆时针旋转矩阵；世界系到局部系需旋转 -yaw_ref）。

    参数
    ----
    reference_yaw : float
        参考航向角（弧度）。

    返回
    ----
    np.ndarray
        形状 ``(2, 2)`` 的旋转矩阵。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"reference_yaw": reference_yaw}, "_rotation_world_to_reference 入参", prefix="[配置]")
    cos_yaw = math.cos(reference_yaw)  # 余弦项。
    sin_yaw = math.sin(reference_yaw)  # 正弦项。
    return np.asarray(
        [
            [cos_yaw, sin_yaw],  # 第一行表示 x 轴投影。
            [-sin_yaw, cos_yaw],  # 第二行表示 y 轴投影。
        ],
        dtype=float,
    )


def _build_vio_jacobian(state_items: tuple[str, ...], *, reference_yaw: float, x_array: "np.ndarray | None" = None) -> np.ndarray:
    """构造 VIO 量测对状态的雅可比矩阵。

    3 维视觉量测为 ``[dx_local, dy_local, dyaw]``，对状态
    ``[px, py, vx, vy, yaw, ...]`` 的偏导。

    铁律 7 紧耦合扩维后 (IMP-A 修复, 2026-07-23 audit Round 5):
    单目 VIO 帧间位姿测量受尺度因子 ``vio_scale`` 影响, 测量模型为
    ``z_vio = R(-yaw_ref) @ (vio_scale * [px; py])``, 因此
    ``H[:, vio_scale]`` 列应为 ``R(-yaw_ref) @ [px; py]`` 即与
    位置项相同的旋转矩阵乘以当前 px/py. 这让 EKF 能从 VIO 残差中推断
    vio_scale, 实现真正的紧耦合.

    参数
    ----
    state_items : tuple[str, ...]
        状态键名元组。
    reference_yaw : float
        参考航向角（弧度），用于构造旋转矩阵。
    x_array : np.ndarray | None
        当前状态向量；estimator 路径需含 px/py/vio_scale（10 维，§1.1+§2.3
        全体同增；由工厂/yaml/state_definition 层强制）；
        几何单元测试路径可传 None 或仅含 px/py 的短状态，vio_scale H 列填 0
        （松耦合旧 8 维语义对几何单测的兼容）。

    返回
    ----
    np.ndarray
        形状 ``(3, n)`` 的雅可比矩阵，n 为状态维度。

    注意
    ----
    - 第 0 行和第 1 行仅对 ``px`` 和 ``py`` 非零，由旋转矩阵确定。
    - 第 2 行仅对 ``yaw`` 为 1.0，其余为 0。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"state_items": state_items, "reference_yaw": reference_yaw}, "_build_vio_jacobian 入参", prefix="[配置]")
    H = np.zeros((3, len(state_items)), dtype=float)  # 3 维视觉量测，状态维度按输入决定。
    rotation_world_to_reference = _rotation_world_to_reference(reference_yaw)  # 先构造旋转矩阵。
    H[0, state_items.index("px")] = float(rotation_world_to_reference[0, 0])  # dx 对 px 的偏导。
    H[0, state_items.index("py")] = float(rotation_world_to_reference[0, 1])  # dx 对 py 的偏导。
    H[1, state_items.index("px")] = float(rotation_world_to_reference[1, 0])  # dy 对 px 的偏导。
    H[1, state_items.index("py")] = float(rotation_world_to_reference[1, 1])  # dy 对 py 的偏导。
    H[2, state_items.index("yaw")] = 1.0  # dyaw 对 yaw 的偏导。
    # IMP-A 修复 (2026-07-23 audit Round 5): 紧耦合 vio_scale 雅可比列.
    # 测量模型 z_vio = R(-yaw_ref) @ (vio_scale * [px; py]),
    # 因此 dz_vio/d(vio_scale) = R(-yaw_ref) @ [px; py]. 需要 x_array 中
    # 的 px/py 当前值.
    # 前提指导 §1.1+§2.3 全体同增 10 维：本函数在 estimator 调用路径下必传 10 维状态，
    # vio_scale 必在状态中、x_array 必非 None、H 列必填。本函数同时支持纯几何(任意 ≥5 维
    # 含 px/py/yaw)的单元测试调用路径——若 state 不含 vio_scale 或 x_array 为 None 时跳过
    # H 列填充（vio_scale 列为 0，等于松耦合旧 8 维语义对单元测试的兼容）。
    # 此回退**仅面向几何单元测试**，§1.1+§2.3 全体同增硬合同由工厂/yaml/state_definition
    # 层强制保障，不会让真实 estimator 路径漂移到 8 维。
    if (
        "vio_scale" in state_items
        and x_array is not None
        and state_items.index("vio_scale") < H.shape[1]
        and state_items.index("px") < x_array.size
    ):
        _idx_vio_scale = state_items.index("vio_scale")
        _px_now = float(x_array[state_items.index("px")])
        _py_now = float(x_array[state_items.index("py")])
        H[0, _idx_vio_scale] = float(rotation_world_to_reference[0, 0]) * _px_now + float(rotation_world_to_reference[0, 1]) * _py_now
        H[1, _idx_vio_scale] = float(rotation_world_to_reference[1, 0]) * _px_now + float(rotation_world_to_reference[1, 1]) * _py_now
        # dyaw 对 vio_scale 偏导为 0 (尺度因子不影响航向测量)
    return H  # 返回雅可比矩阵。


def _coerce_vio_measurement_vector(z_vio: Any) -> np.ndarray:
    """把 VIO 量测规整成长度为 3 的浮点向量。

    参数
    ----
    z_vio : Any
        VIO 量测，必须可转为一维 3 元素浮点数组，每个元素必须为有限数值。

    返回
    ----
    np.ndarray
        形状 ``(3,)`` 的浮点量测向量 ``[dx, dy, dyaw]``。

    异常
    ----
    TypeError
        量测含布尔值、非标量或非数值类型。
    ValueError
        量测长度不为 3。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"z_vio": z_vio}, "_coerce_vio_measurement_vector 入参", prefix="[配置]")
    require_not_none(z_vio, "z_vio")  # 量测不能为空。

    if is_bool_like(z_vio):  # bool 不允许。
        raise TypeError("z_vio must not be boolean-like")  # 直接拒绝。

    raw_z_vio = np.asarray(z_vio, dtype=object)  # 先用 object 保留每个元素原貌，方便逐项检查。
    if raw_z_vio.ndim != 1:  # 量测必须是一维。
        raise TypeError(f"z_vio must be a 1D measurement vector, got shape {raw_z_vio.shape}")  # 结构不对就报错。
    if raw_z_vio.shape != (3,):  # 量测必须恰好三个分量。
        raise ValueError(f"z_vio has shape {raw_z_vio.shape}, expected (3,)")  # 长度不对就报错。

    coerced = []  # 这里收集规整后的三个数值。
    for index, item in enumerate(raw_z_vio):  # 逐项检查每个分量。
        if is_bool_like(item):  # 任何分量都不能是 bool。
            raise TypeError(f"z_vio[{index}] must not be boolean-like")  # 直接拒绝。
        scalar = np.asarray(item)  # 进一步拆成标量视图。
        if scalar.shape != ():  # 每一项都必须是标量。
            raise TypeError(f"z_vio[{index}] must be a scalar numeric value")  # 不是标量就报错。
        scalar_value = scalar.item()  # 取出原始值。
        if is_bool_like(scalar_value):  # 再防一次布尔值。
            raise TypeError(f"z_vio[{index}] must not be boolean-like")  # 直接拒绝。
        if not is_real(scalar_value):  # 必须是实数值类型（排除 complex）。
            raise TypeError(f"z_vio[{index}] must be numeric, got {type(scalar_value).__name__}")  # 直接拒绝。
        fv = coerce_finite_scalar(scalar_value, name=f"z_vio[{index}]")  # NaN/Inf 不允许穿透到残差计算。
        coerced.append(fv)  # 追加规整后的浮点数。
    return np.asarray(coerced, dtype=float)  # 返回三维浮点量测向量。


def _normalize_vio_covariance(R_vio: Any) -> np.ndarray:
    """把视觉噪声规整成 3x3 协方差矩阵。

    参数
    ----
    R_vio : Any
        视觉测量噪声，支持多种输入形式：
        - 标量：各向同性噪声，扩展为 ``scalar * I₃``。
        - 长度 3 的序列：对角项 ``[σ²_dx, σ²_dy, σ²_dyaw]``。
        - 3×3 矩阵：直接使用。
        - 映射型 ``(pos, yaw)``：pos 用于 dx/dy，yaw 用于 dyaw，
          **若同时含多组键会静默取 ``(pos, yaw)`` 优先**。
        - 映射型 ``(dx, dy, dyaw)``：分别指定三个分量的方差。

    返回
    ----
    np.ndarray
        形状 ``(3, 3)`` 的协方差矩阵。

    异常
    ----
    TypeError
        含布尔值或非法类型。
    ValueError
        含非有限值或对角项为负。
    KeyError
        映射型键名不匹配任何已知格式。

    注意
    ----
    - 仅检查对角线非负，不保证正定性。
    - 3×3 输入不做对称性校验。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"R_vio": R_vio}, "_normalize_vio_covariance 入参", prefix="[配置]")
    require_not_none(R_vio, "R_vio")  # 协方差不能为空。

    if is_bool_like(R_vio):  # bool 不允许。
        raise TypeError("R_vio must not be boolean-like")  # 直接拒绝。

    if isinstance(R_vio, Mapping):  # 映射型可以用两种常见键集。
        has_shared_scheme = all(key in R_vio for key in ("pos", "yaw"))
        has_per_axis_scheme = all(key in R_vio for key in VIO_MEASUREMENT_ITEMS)
        if has_shared_scheme and has_per_axis_scheme:
            raise ValueError("R_vio mapping must not mix pos/yaw with dx/dy/dyaw noise entries")
        if has_shared_scheme:  # 位置和航向共用位置噪声。
            diag_values = [
                _coerce_numeric_scalar(R_vio["pos"], name='R_vio["pos"]'),  # dx 噪声。
                _coerce_numeric_scalar(R_vio["pos"], name='R_vio["pos"]'),  # dy 噪声。
                _coerce_numeric_scalar(R_vio["yaw"], name='R_vio["yaw"]'),  # dyaw 噪声。
            ]
            unexpected_keys = sorted(set(R_vio.keys()) - {"pos", "yaw"})
            if unexpected_keys:
                raise ValueError(f"R_vio pos/yaw scheme must not include extra keys: {unexpected_keys}")
        elif has_per_axis_scheme:  # 或者分别给三个分量。
            diag_values = [
                _coerce_numeric_scalar(R_vio["dx"], name='R_vio["dx"]'),  # dx 噪声。
                _coerce_numeric_scalar(R_vio["dy"], name='R_vio["dy"]'),  # dy 噪声。
                _coerce_numeric_scalar(R_vio["dyaw"], name='R_vio["dyaw"]'),  # dyaw 噪声。
            ]
            unexpected_keys = sorted(set(R_vio.keys()) - set(VIO_MEASUREMENT_ITEMS))
            if unexpected_keys:
                raise ValueError(f"R_vio dx/dy/dyaw scheme must not include extra keys: {unexpected_keys}")
        else:
            raise KeyError("R_vio mapping must provide either pos/yaw or dx/dy/dyaw noise entries")  # 键不对就报错。
        if any(value <= 0.0 for value in diag_values):  # 协方差对角项必须严格为正。
            raise ValueError("R_vio diagonal entries must be positive")  # 零或负数都不允许。
        return np.diag(diag_values)  # 组装成对角矩阵。

    raw_R_vio = np.asarray(R_vio, dtype=object)  # 先保留原始结构，逐项排查 bool。
    if any(
        is_bool_like(item)  # 任何布尔值都不允许。
        or (isinstance(item, np.ndarray) and np.asarray(item).shape == () and np.asarray(item).dtype.kind == "b")  # 也不允许布尔标量包装。
        for item in raw_R_vio.ravel()  # 扫描扁平后的每个元素。
    ):
        raise TypeError("R_vio must not contain boolean-like entries")  # 直接拒绝。

    R_vio_array = np.asarray(R_vio, dtype=float)  # 再转成浮点数组。
    if R_vio_array.ndim == 0:  # 标量噪声可以视为各向同性。
        scalar_value = coerce_finite_scalar(float(R_vio_array), name="R_vio scalar", min_value=0.0, inclusive=False)  # 标量必须有限且严格为正。
        return np.eye(3, dtype=float) * scalar_value  # 扩成 3x3 对角矩阵。
    if R_vio_array.shape == (3,):  # 一维 3 元素可以视作对角项。
        if not np.all(np.isfinite(R_vio_array)):  # 每个分量都要有限。
            raise ValueError("R_vio diagonal entries must be finite")  # 直接报错。
        if np.any(R_vio_array <= 0.0):  # 不能有零或负对角项。
            raise ValueError("R_vio diagonal entries must be positive")  # 协方差方差项必须严格为正。
        return np.diag(R_vio_array)  # 转成对角矩阵。
    require_shape(R_vio_array, (3, 3), name="R_vio")  # 其他情况必须是 3x3。
    if not np.all(np.isfinite(R_vio_array)):  # 矩阵里不能有非有限值。
        raise ValueError("R_vio must contain finite values")  # 直接报错。
    if np.any(np.diag(R_vio_array) <= 0.0):  # 对角项必须严格为正。
        raise ValueError("R_vio diagonal entries must be positive")  # 协方差方差项不能为零或负。
    return R_vio_array  # 返回规整后的协方差。


def _ensure_positive_definite_vio_innovation_covariance(
    innovation_covariance: Any,
    *,
    name: str = "Innovation covariance S",
) -> np.ndarray:
    """校验 VIO 创新协方差是有限、对称且正定的 3×3 矩阵。

    §11.5 SPD 抖动注入（spec L1745「协方差对称正定保护、抖动、发散判定全员同一规则」）：
    当对称化 + 对角线阳性通过但 Cholesky 仍失败（病态条件数）时，
    先尝试 ``S += cov_jitter_eps * I`` 再重试 Cholesky；
    二次仍失败才 raise ValueError（fail-loud 路径保留）。
    三个 estimator（EKF / Robust-EKF / FGO）同调本函数 → 全员同一 jitter 政策。
    """
    from liquidloc.common.tee_logger import print_dict
    from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
    print_dict({"innovation_covariance": innovation_covariance, "name": name}, "_ensure_positive_definite_vio_innovation_covariance 入参", prefix="[配置]")
    innovation_covariance = np.asarray(innovation_covariance, dtype=float)
    require_shape(innovation_covariance, (3, 3), name=name)
    if not np.all(np.isfinite(innovation_covariance)):
        raise ValueError(f"{name} must be finite")
    innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
    if np.any(np.diag(innovation_covariance) <= 0.0):
        raise ValueError(f"{name} must be positive definite")
    try:
        np.linalg.cholesky(innovation_covariance)
    except np.linalg.LinAlgError:
        # §11.5 抖动注入：第一次 Cholesky 失败 → S += cov_jitter_eps * I 再重试。
        cov_jitter_eps = float(BRIDGE_THRESHOLDS["cov_jitter_eps"])
        jittered = innovation_covariance + cov_jitter_eps * np.eye(innovation_covariance.shape[0])
        try:
            np.linalg.cholesky(jittered)
            innovation_covariance = jittered  # 接受 jittered 版本作为该次创新协方差。
        except np.linalg.LinAlgError as exc:
            # 二次仍失败 → fail-loud；不允许任何"只救一方"的静默重置。
            raise ValueError(f"{name} must be positive definite (jitter fallback exhausted at eps={cov_jitter_eps}") from exc
    return innovation_covariance


def build_vio_measurement(vio_event: Any) -> np.ndarray:
    """从单个 VIO 事件里提取 [dx, dy, dyaw] 量测向量。

    参数
    ----
    vio_event : Any
        VIO 事件对象，需包含 ``vio_payload`` 字段，
        由 :func:`extract_vio_measurement` 解析。

    返回
    ----
    np.ndarray
        形状 ``(3,)`` 的浮点量测向量 ``[dx, dy, dyaw]``。

    异常
    ----
    ValueError
        ``vio_event`` 为 ``None`` 或字段值非有限。
    TypeError
        字段值类型非法。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"vio_event": vio_event}, "build_vio_measurement 入参", prefix="[配置]")
    require_not_none(vio_event, "vio_event")  # 事件不能为空。

    vio_measurement = extract_vio_measurement(vio_event)  # 从事件里提取 VIO 字段。
    dx = _coerce_numeric_scalar(vio_measurement["dx"], name='vio_measurement["dx"]')  # 提取 dx。
    dy = _coerce_numeric_scalar(vio_measurement["dy"], name='vio_measurement["dy"]')  # 提取 dy。
    dyaw = _coerce_numeric_scalar(vio_measurement["dyaw"], name='vio_measurement["dyaw"]')  # 提取 dyaw。
    z_vio = np.asarray([dx, dy, dyaw], dtype=float)  # 按固定顺序拼成量测向量。
    require_shape(z_vio, (3,), name="z_vio")  # 再确认一次维度。
    z_vio.flags.writeable = False  # 冻结为只读，从架构上保证 forbid_rewrite_measurement_fields_inside_update。
    return z_vio  # 返回视觉量测。


def _resolve_reference_pose(
    reference_pose: np.ndarray | Mapping[str, Any] | None,
    current_pose: np.ndarray,
) -> np.ndarray:
    """确定视觉更新使用的参考位姿。

    参数
    ----
    reference_pose : np.ndarray | Mapping[str, Any] | None
        显式指定的参考位姿。若为 ``None`` 则使用当前位姿。
    current_pose : np.ndarray
        当前位姿向量 ``[px, py, yaw]``。

    返回
    ----
    np.ndarray
        参考位姿向量 ``[px, py, yaw]``。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"reference_pose": reference_pose, "current_pose": current_pose}, "_resolve_reference_pose 入参", prefix="[配置]")
    if reference_pose is None:  # 如果没有显式参考位姿，就用当前位姿兜底。
        return current_pose.copy()  # 返回当前位姿副本。

    reference_array, reference_items = _coerce_state_vector(reference_pose)  # 先规整参考输入。
    return _extract_pose_vector(reference_array, reference_items)  # 再只抽出位姿部分。


def _predict_local_vio_measurement(current_pose: np.ndarray, reference_pose: np.ndarray) -> np.ndarray:
    """从当前位姿和参考位姿计算本地坐标系下的预测视觉量测。

    参数
    ----
    current_pose : np.ndarray
        当前位姿 ``[px, py, yaw]``。
    reference_pose : np.ndarray
        参考位姿 ``[px, py, yaw]``。

    返回
    ----
    np.ndarray
        预测量测 ``[dx_local, dy_local, dyaw]``。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"current_pose": current_pose, "reference_pose": reference_pose}, "_predict_local_vio_measurement 入参", prefix="[配置]")
    delta_world = np.asarray(
        [
            float(current_pose[0]) - float(reference_pose[0]),  # 世界系 x 差值。
            float(current_pose[1]) - float(reference_pose[1]),  # 世界系 y 差值。
        ],
        dtype=float,
    )
    rotation_world_to_reference = _rotation_world_to_reference(float(reference_pose[2]))  # 用参考航向构造旋转。
    local_translation = rotation_world_to_reference @ delta_world  # 把位移转到参考局部坐标系。
    return np.asarray(
        [
            float(local_translation[0]),  # 局部 dx。
            float(local_translation[1]),  # 局部 dy。
            angle_delta_rad(float(current_pose[2]), float(reference_pose[2])),  # 局部 dyaw。
        ],
        dtype=float,
    )


def compute_vio_residual(
    x_pred: np.ndarray | Sequence[float] | Mapping[str, Any],
    z_vio: Any,
    reference_pose: np.ndarray | Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算 VIO 量测和预测量测之间的残差。

    参数
    ----
    x_pred : np.ndarray | Sequence[float] | Mapping[str, Any]
        预测状态。
    z_vio : Any
        VIO 量测向量 ``[dx, dy, dyaw]``。
    reference_pose : np.ndarray | Mapping[str, Any] | None
        可选参考位姿，默认使用当前位姿。

    返回
    ----
    z_hat : np.ndarray
        预测量测 ``[dx_local, dy_local, dyaw]``。
    residual : np.ndarray
        残差向量 ``[dx_diff, dy_diff, dyaw_diff]``，航向按环形差处理。
    H : np.ndarray
        观测雅可比 ``(3, n)``。
    reference_pose_array : np.ndarray
        实际使用的参考位姿 ``[px, py, yaw]``。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"x_pred": x_pred, "z_vio": z_vio, "reference_pose": reference_pose}, "compute_vio_residual 入参", prefix="[配置]")
    z_vio_array = _coerce_vio_measurement_vector(z_vio)  # 先规整量测。
    z_vio_array[2] = wrap_angle_rad(z_vio_array[2])  # 航向分量先归一化。

    x_pred_array, state_items = _coerce_state_vector(x_pred)  # 再规整状态。
    current_pose = _extract_pose_vector(x_pred_array, state_items)  # 抽出当前位姿。
    current_pose[2] = wrap_angle_rad(current_pose[2])  # 当前航向先归一化。
    reference_pose_array = _resolve_reference_pose(reference_pose, current_pose)  # 确定参考位姿。
    reference_pose_array[2] = wrap_angle_rad(reference_pose_array[2])  # 参考航向也归一化。
    H = _build_vio_jacobian(state_items, reference_yaw=float(reference_pose_array[2]), x_array=x_pred_array)  # 构造雅可比 (传规整后 x_pred_array 让 vio_scale 列可计算).
    # 当 reference_pose=None 时退化到 current_pose，此时 z_hat 恒为零，
    # 但 H 的位置行（旋转矩阵行）仍正确反映量测对状态的偏导，
    # 不应置零，否则卡尔曼增益中位置信息会丢失。
    z_hat = _predict_local_vio_measurement(current_pose, reference_pose_array)  # 计算预测量测。
    residual = z_vio_array - z_hat  # 计算残差。
    residual[2] = angle_delta_rad(z_vio_array[2], z_hat[2])  # 航向残差必须按环形角差处理。
    return z_hat, residual, H, reference_pose_array  # 返回预测量、残差、雅可比和参考位姿。


def apply_vision_update(
    x_pred: np.ndarray | Sequence[float] | Mapping[str, Any],
    P_pred: np.ndarray | Sequence[Sequence[float]],
    vio_event: Any,
    R_vio: Any,
    *,
    reference_pose: np.ndarray | Mapping[str, Any] | None = None,
) -> tuple[np.ndarray | dict[str, Any], np.ndarray, dict[str, Any]]:
    """执行一次标准 VIO EKF 更新，不带门控。

    参数
    ----
    x_pred : np.ndarray | Sequence[float] | Mapping[str, Any]
        预测状态。
    P_pred : np.ndarray | Sequence[Sequence[float]]
        预测协方差矩阵，形状 ``(n, n)``。
    vio_event : Any
        VIO 事件对象，需含 ``vio_payload`` 字段。
    R_vio : Any
        视觉测量噪声（标量/序列/矩阵/映射）。
    reference_pose : np.ndarray | Mapping[str, Any] | None
        可选参考位姿，默认使用当前位姿。

    返回
    ----
    x_upd : np.ndarray | dict[str, Any]
        更新后状态，形式与 ``x_pred`` 一致。
    P_upd_array : np.ndarray
        更新后协方差矩阵。
    update_report : dict[str, Any]
        更新摘要，含 ``z_vio``、``z_hat``、``reference_pose``、``residual``、``gate``。

    异常
    ----
    ValueError
        输入为 None、维度不匹配、数值非有限、噪声对角项为负。
    TypeError
        输入类型非法（如 bool）。
    KeyError
        映射型状态/噪声缺少必要键。

    注意
    ----
    - 协方差更新采用 Joseph 形式 + 对称化，保证数值稳定。
    - 更新后航向自动归一化到 ``[-π, π)``。
    - 不含门控逻辑，所有量测均参与更新。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"x_pred": x_pred, "P_pred": P_pred, "vio_event": vio_event, "R_vio": R_vio, "reference_pose": reference_pose}, "apply_vision_update 入参", prefix="[配置]")
    require_not_none(P_pred, "P_pred")  # 协方差不能为空。

    z_vio = build_vio_measurement(vio_event)  # 先提取视觉量测（只读，架构上保证 forbid_rewrite）。
    # 运行时断言：确认量测向量不可写，执行 VIO 更新合同中的 forbid_rewrite 约束。
    if z_vio.flags.writeable:
        raise AssertionError("VIO measurement vector must be read-only (forbid_rewrite contract)")
    x_pred_array, state_items = _coerce_state_vector(x_pred)  # 再规整状态。

    P_pred_array = np.asarray(P_pred, dtype=float)  # 协方差转浮点数组。
    require_shape(P_pred_array, (x_pred_array.shape[0], x_pred_array.shape[0]), name="P_pred")  # 维度必须匹配状态。

    R_vio_array = _normalize_vio_covariance(R_vio)  # 规整视觉噪声。
    # 直接计算残差，不再通过 compute_vio_residual（避免重复 coerce 状态和量测）。
    # _coerce_vio_measurement_vector 创建计算用副本（新可写数组），
    # 原始只读 z_vio 不被修改，不违反 forbid_rewrite 合同。
    # wrap_angle_rad 是角度归一化（数学归一化），不是测量篡改。
    z_vio_array = _coerce_vio_measurement_vector(z_vio)  # 规整量测。
    z_vio_array[2] = wrap_angle_rad(z_vio_array[2])  # 航向分量归一化。
    current_pose = _extract_pose_vector(x_pred_array, state_items)  # 抽出当前位姿。
    current_pose[2] = wrap_angle_rad(current_pose[2])  # 当前航向归一化。
    reference_pose_array = _resolve_reference_pose(reference_pose, current_pose)  # 确定参考位姿。
    reference_pose_array[2] = wrap_angle_rad(reference_pose_array[2])  # 参考航向归一化。
    H = _build_vio_jacobian(state_items, reference_yaw=float(reference_pose_array[2]), x_array=x_pred_array)  # 构造雅可比 (传规整后 x_pred_array 让 vio_scale 列可计算).
    z_hat = _predict_local_vio_measurement(current_pose, reference_pose_array)  # 预测量测。
    residual = z_vio_array - z_hat  # 残差。
    residual[2] = angle_delta_rad(z_vio_array[2], z_hat[2])  # 航向残差按环形差处理。
    S = H @ P_pred_array @ H.T + R_vio_array  # 创新协方差。
    S = _ensure_positive_definite_vio_innovation_covariance(S)  # 非正定创新协方差会让更新失去统计语义。
    PHt = P_pred_array @ H.T  # 先算协方差和雅可比的乘积。
    try:
        K_gain = np.linalg.solve(S, PHt.T).T  # 解线性方程得到卡尔曼增益。
    except np.linalg.LinAlgError as exc:  # 正定矩阵理论上必须可解，这里保留成显式失败。
        raise ValueError("Innovation covariance S must be positive definite") from exc

    x_upd_array = x_pred_array + (K_gain @ residual)  # 更新状态向量。
    yaw_index = state_items.index("yaw")  # 找到 yaw 在状态里的位置。
    x_upd_array[yaw_index] = wrap_angle_rad(x_upd_array[yaw_index])  # 更新后航向重新归一化。
    identity = np.eye(x_pred_array.shape[0], dtype=float)  # 构造单位阵。
    state_transition = identity - (K_gain @ H)  # 标准协方差更新中间项。
    P_upd_array = state_transition @ P_pred_array @ state_transition.T  # Joseph 形式第一步。
    P_upd_array += K_gain @ R_vio_array @ K_gain.T  # 加上量测噪声项。
    P_upd_array = 0.5 * (P_upd_array + P_upd_array.T)  # 对称化。

    x_upd = _restore_state_like_input(x_pred, x_upd_array, state_items)  # 恢复输入承载形式。
    update_report = {  # 组织给上游看的更新摘要。
        "z_vio": z_vio.tolist(),  # 原始视觉量测。
        "z_hat": z_hat.tolist(),  # 预测视觉量测。
        "reference_pose": reference_pose_array.tolist(),  # 实际使用的参考位姿。
        "residual": residual.tolist(),  # 最终残差。
        "gate": {"enabled": False},  # 这里明确说明没有门控。
        "modality": "vio",  # 当前更新模态。
        "update_applied": True,  # VIO 更新已应用。
        "reason": "vio_update_success",  # 更新原因。
        # §3.0.2 审计要求：主路径报告须能区分 h 侧修正 与 改 z。
        # VIO 不接受 NN bias（build_measurement_control 在 VIO 分支强制 bias_applied=0.0，
        # 残差永远定义在原始增量 z 上），标记 "none" 显式说明本次更新未触发 bias 写入口。
        "bias_writeport": "none",
    }
    return x_upd, P_upd_array, update_report  # 返回更新结果。
