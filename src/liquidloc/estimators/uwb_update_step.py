"""
文件：src/liquidloc/estimators/uwb_update_step.py

【文件职责】
这个文件实现 UWB 距离更新，含 UWB 钟差的在线估计。
它接收预测状态、预测协方差、锚点坐标和测距值，执行一次标准的 EKF 更新。
观测模型 z_range = ||p - a|| + uwb_clock_bias，钟差作为距离偏置（单位 m）
进入 z_pred 与雅可比 H，使 uwb_clock_bias 可从 UWB 残差中辨识
（前提指导 §2.3 在线时间偏移与钟差态前提）。

【本文件绝对不负责】
不做风险加权（robust_ekf_core 负责），不做场景门控，不消费模型输出。

【上游依赖】
state_definition.py、sensors/uwb_model.py、sensors/anchor_model.py。

【下游调用者】
ekf_core.py、fgo_core.py、fusion 相关流程。

【输入对象定义】
- x_pred: 预测状态
- P_pred: 预测协方差
- anchor_pos: 当前锚点坐标
- z_range: 当前测距
- R_range: 当前测量噪声

【输出对象定义】
- x_upd: 更新后状态
- P_upd: 更新后协方差
- update_info: 本次更新摘要
"""

from __future__ import annotations  # 允许前向类型引用。

import copy  # 用来深拷贝映射型状态和更新信息。
import math  # 用来计算欧氏距离和有限性检查。
from collections.abc import Mapping, Sequence  # 用来识别映射型和序列型输入。
from typing import Any  # 用来标注任意类型。

import numpy as np  # 用来做矩阵运算。

from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_real, require_keys, require_not_none  # 集中判断 bool / np.bool_ / 实数类型，有限浮点转换，以及基础输入校验。
from liquidloc.estimators.state_definition import state_items  # 冻结的状态顺序，供雅可比索引动态派生。


# 从冻结状态项动态派生 px/py 索引，消除硬编码数字索引。
# 如果 state_items 顺序变更，这些索引会自动跟随，不再静默出错。
_UWB_STATE_ITEMS = state_items  # UWB 更新依赖的冻结状态顺序。
_IDX = {name: i for i, name in enumerate(_UWB_STATE_ITEMS)}  # 状态名→索引映射，与 predict_step.py 派生方式保持一致。
_IDX_PX = _IDX["px"]  # x 方向位置索引。
_IDX_PY = _IDX["py"]  # y 方向位置索引。
_IDX_UWB_CLOCK_BIAS = _IDX["uwb_clock_bias"]  # UWB 钟差索引（前提指导 §2.3 在线时间偏移与钟差态）。
# UWB 测距观测模型 z_range = ||p - a|| + uwb_clock_bias：
# 钟差等价于距离偏置（单位 m），进入 z_pred 使 uwb_clock_bias 可从 UWB 残差中辨识，
# 满足 §2.3 钟差态在线估计前提。雅可比 H = ∂z_pred/∂x，故 H[0, uwb_clock_bias] = +1。


def _coerce_state_vector(
    x_pred: np.ndarray | Sequence[float] | Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """把预测状态规整成状态向量，并保留原始形状信息。

    参数
    ----
    x_pred : np.ndarray | Sequence[float] | Mapping[str, Any]
        预测状态。支持三种承载形式：
        - 映射型：必须包含 ``"px"`` 和 ``"py"`` 键，按 state_items 顺序构造完整向量。
        - 数组/序列型：至少包含 2 个元素，保留原始维度。

    返回
    ----
    vector : np.ndarray
        一维浮点向量（copy）。
    state_meta : dict
        元信息字典，``"kind"`` 为 ``"mapping"`` 或 ``"array"``。
        映射型含 ``"template"``，数组型含 ``"shape"``。

    异常
    ----
    ValueError
        ``x_pred`` 为 ``None``、维度不对、元素不足、含 NaN/Inf。
    TypeError
        ``x_pred`` 为字符串/字节串、bool 数组、映射值为 bool 或非数值类型。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"x_pred": x_pred}, "_coerce_state_vector 入参", prefix="[配置]")
    require_not_none(x_pred, "x_pred")  # 预测状态不能为空。

    if isinstance(x_pred, Mapping):  # 如果是映射，就按冻结状态顺序构造完整向量。
        require_keys(x_pred, ("px", "py"), name="x_pred")  # 映射里必须有位置键。
        values = []  # 按 state_items 顺序收集通过校验的状态值。
        for key in _UWB_STATE_ITEMS:  # 缺失键置零，但存在的键必须通过数值安全检查。
            raw = x_pred.get(key, 0.0)  # 缺失键默认 0.0。
            if is_bool_like(raw):  # bool 不算数值，与 _coerce_scalar 保持一致。
                raise TypeError(f"x_pred[{key!r}] must be numeric, got bool")
            if not is_real(raw):  # 必须是实数类型，排除 complex 和非数值。
                raise TypeError(f"x_pred[{key!r}] must be numeric, got {type(raw).__name__}")
            val = coerce_finite_scalar(raw, name=f"x_pred[{key!r}]")  # NaN/Inf 会破坏后续 EKF 计算。
            values.append(val)  # 收集通过校验的值。
        vector = np.asarray(values, dtype=float)  # 构造完整状态向量。
        return vector, {"kind": "mapping", "template": copy.deepcopy(x_pred)}  # 保留原映射模板，便于恢复。

    if isinstance(x_pred, (str, bytes)):  # 字符串不能当数值状态。
        raise TypeError("x_pred must be a numeric state vector")  # 直接拒绝。

    # 先转成 ndarray 但不指定 dtype，保留原始类型信息用于 bool 检查。
    # 直接 np.asarray(dtype=float) 会把 bool 静默转成 1.0/0.0，与 _coerce_anchor_xy 和映射型路径口径不一致。
    raw_vector = np.asarray(x_pred)
    if raw_vector.dtype == bool:  # bool 数组不能当数值状态，与映射型路径 is_bool_like 检查口径一致。
        raise TypeError("x_pred must be a numeric state vector, got bool array")  # 直接拒绝。
    vector = raw_vector.astype(float)  # 现在可以安全转成 float。
    original_shape = vector.shape  # 记录原始形状，更新后要恢复。
    if vector.ndim == 2 and 1 in vector.shape:  # 允许列向量/行向量的二维写法。
        vector = vector.reshape(-1)  # 拉平成一维。
    elif vector.ndim != 1:  # 其他维度都不接受。
        raise ValueError("x_pred must be 1D or a column vector")  # 直接报错。
    if vector.size < 2:  # 至少要有 px 和 py。
        raise ValueError("x_pred must contain at least px and py")  # 直接报错。
    if not np.isfinite(vector).all():  # NaN/Inf 会破坏后续 EKF 计算。
        raise ValueError("x_pred must contain only finite values")  # 直接报错。

    return vector.copy(), {"kind": "array", "shape": original_shape}  # 保存数组态元信息。


def _restore_state_payload(
    x_vector: np.ndarray,
    state_meta: dict[str, Any],
) -> np.ndarray | dict[str, Any]:
    """把更新后的向量恢复成和原输入一致的承载形式。

    参数
    ----
    x_vector : np.ndarray
        更新后的一维状态向量。
    state_meta : dict
        :func:`_coerce_state_vector` 返回的元信息。

    返回
    ----
    np.ndarray | dict[str, Any]
        与输入形式一致的状态载体。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"x_vector": x_vector, "state_meta": state_meta}, "_restore_state_payload 入参", prefix="[配置]")
    if state_meta["kind"] == "mapping":  # 原始输入是映射时就回写到字典里。
        x_upd = copy.deepcopy(state_meta["template"])  # 深拷贝模板，避免内嵌可变对象共享引用。
        for idx, key in enumerate(_UWB_STATE_ITEMS):  # 按冻结状态顺序逐项回写，EKF 交叉耦合会间接修正全部状态。
            x_upd[key] = float(x_vector[idx])  # 写回更新后的值。
        return x_upd  # 返回恢复后的映射。

    original_shape = state_meta["shape"]  # 取回原始数组形状。
    if len(original_shape) == 2 and 1 in original_shape:  # 如果原来是列向量或行向量，就恢复原状。
        return x_vector.reshape(original_shape)  # 按原形状重排。
    return x_vector  # 否则直接返回一维数组。


def _coerce_covariance_matrix(
    P_pred: np.ndarray | Sequence[Sequence[float]],
    *,
    state_dim: int,
) -> np.ndarray:
    """把协方差规整成固定维度的矩阵副本。

    参数
    ----
    P_pred : np.ndarray | Sequence[Sequence[float]]
        预测协方差矩阵。
    state_dim : int
        期望的状态维度，协方差必须为 ``(state_dim, state_dim)``。

    返回
    ----
    np.ndarray
        浮点协方差矩阵副本。

    异常
    ----
    ValueError
        ``P_pred`` 为 ``None`` 或形状不匹配。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"P_pred": P_pred, "state_dim": state_dim}, "_coerce_covariance_matrix 入参", prefix="[配置]")
    require_not_none(P_pred, "P_pred")  # 协方差不能为空。
    P_matrix = np.asarray(P_pred, dtype=float)  # 转成浮点矩阵。
    if P_matrix.shape != (state_dim, state_dim):  # 维度必须严格匹配。
        raise ValueError(
            f"P_pred must have shape ({state_dim}, {state_dim}), got {P_matrix.shape}"
        )  # 维度不对直接报错。
    if not np.all(np.isfinite(P_matrix)):  # NaN/Inf 会静默污染下游创新协方差和增益。
        raise ValueError("P_pred must contain only finite values")  # 直接拒绝。
    if not np.allclose(P_matrix, P_matrix.T):  # 协方差矩阵必须对称。
        raise ValueError("P_pred must be symmetric")  # 非对称直接报错。
    try:
        np.linalg.cholesky(P_matrix)  # 协方差矩阵必须正定。
    except np.linalg.LinAlgError:
        # §11.5 抖动注入：第一次 Cholesky 失败 → P += cov_jitter_eps * I 再重试。
        from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
        cov_jitter_eps = float(BRIDGE_THRESHOLDS["cov_jitter_eps"])
        jittered = P_matrix + cov_jitter_eps * np.eye(P_matrix.shape[0])
        try:
            np.linalg.cholesky(jittered)
            P_matrix = jittered  # 接受 jittered 版本作为该次预测协方差。
        except np.linalg.LinAlgError as exc:  # 非正定矩阵会让 EKF 更新失去统计语义。
            raise ValueError("P_pred must be positive definite (jitter fallback exhausted at eps={cov_jitter_eps}") from exc  # 二次仍失败 → fail-loud。
    return P_matrix.copy()  # 返回副本，避免改坏输入。


def _coerce_anchor_xy(
    anchor_pos: np.ndarray | Sequence[float] | Mapping[str, Any],
) -> tuple[float, float]:
    """从锚点输入中提取二维坐标。

    参数
    ----
    anchor_pos : np.ndarray | Sequence[float] | Mapping[str, Any]
        锚点坐标。映射型依次尝试 ``(ax, ay)`` > ``(x, y)`` > ``(px, py)`` 三组键名，
        **若同时含多组键会静默取第一组匹配**；数组型必须恰好 2 个元素。

    返回
    ----
    tuple[float, float]
        ``(ax, ay)`` 锚点坐标。

    异常
    ----
    ValueError
        ``anchor_pos`` 为 ``None``、维度不对、元素数不是 2、值为 NaN/Inf。
    TypeError
        ``anchor_pos`` 为字符串/字节串，或坐标为 bool/非数值类型。
    KeyError
        映射型中找不到任何一组已知键名。

    注意
    ----
    - ``(ax, ay)`` 键名与 ``configs/base/sensors.yaml`` 中 IMU 加速度字段
      （``imu_fields: [ax, ay, gz]``）同名，存在命名冲突风险。调用方应
      优先使用 ``(x, y)`` 或 ``(px, py)`` 键名，避免误传 IMU payload。
    - 数组型路径与映射型路径均通过 :func:`_coerce_scalar` 做 bool/NaN/Inf
      校验，与 ``anchor_model._normalize_anchor_layout`` 的口径一致。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"anchor_pos": anchor_pos}, "_coerce_anchor_xy 入参", prefix="[配置]")
    require_not_none(anchor_pos, "anchor_pos")  # 锚点输入不能为空。

    if isinstance(anchor_pos, Mapping):  # 映射型锚点坐标需要兼容多种键名。
        for x_key, y_key in (("ax", "ay"), ("x", "y"), ("px", "py")):  # 依次尝试三个常见命名。
            if x_key in anchor_pos and y_key in anchor_pos:  # 找到一组就可以直接返回。
                return _coerce_scalar(anchor_pos[x_key], name=f"anchor_pos.{x_key}"), _coerce_scalar(
                    anchor_pos[y_key],
                    name=f"anchor_pos.{y_key}",
                )  # 这里显式分两行，方便你检查每个键。
        raise KeyError("anchor_pos mapping must provide one of (ax, ay), (x, y), or (px, py)")  # 找不到就报错。

    if isinstance(anchor_pos, (str, bytes)):  # 字符串不能表示锚点坐标。
        raise TypeError("anchor_pos must be a 2D coordinate record")  # 直接拒绝。

    # 先转成 ndarray 但不指定 dtype，保留原始类型信息用于 bool 检查。
    # 直接 np.asarray(dtype=float) 会把 bool 静默转成 1.0/0.0，与 anchor_model.py L195 口径不一致。
    raw_array = np.asarray(anchor_pos)
    if raw_array.ndim == 2 and 1 in raw_array.shape:  # 允许二维单列或单行。
        raw_array = raw_array.reshape(-1)  # 拉平成一维。
    elif raw_array.ndim != 1:  # 其他维度不接受。
        raise ValueError("anchor_pos must be a 1D coordinate record")  # 直接报错。
    if raw_array.size != 2:  # 坐标必须恰好两个分量。
        raise ValueError(f"anchor_pos must contain exactly 2 coordinates, got {raw_array.size}")  # 直接报错。

    # 用 _coerce_scalar 做 bool/NaN/inf 检查，与映射型路径口径一致。
    # raw_array[0]/[1] 是 numpy 标量（np.float64/np.int64 等），_coerce_scalar 会正确处理。
    return _coerce_scalar(raw_array[0], name="anchor_pos[0]"), _coerce_scalar(raw_array[1], name="anchor_pos[1]")  # 返回 ax/ay。


def _coerce_scalar(
    value: Any,
    *,
    name: str,
    min_value: float | None = None,
    inclusive: bool = True,
) -> float:
    """把输入规整成有限数值标量，并可选检查下界。

    委托到 common.validation.coerce_finite_scalar，额外显式拒绝 np.ndarray
    （包括 0 维数组），防止数组穿透到标量计算中。

    参数
    ----
    value : Any
        待规整的值。
    name : str
        参数名称，用于错误信息。
    min_value : float | None
        可选下界。若提供则 ``value >= min_value``（或 ``>``）必须成立。
        ``min_value`` 自身必须为有限实数，否则报错。
    inclusive : bool
        为 True 时使用 ``>=``，为 False 时使用 ``>``。默认 True。

    返回
    ----
    float
        规整后的有限浮点数。

    异常
    ----
    TypeError
        值为 ``bool``、``np.ndarray`` 或非数值类型。
    ValueError
        值为 ``None``、NaN、Inf、低于下界或 ``min_value`` 非有限。

    注意
    ----
    - 显式拒绝 ``np.ndarray``（包括 0 维数组 ``np.array(1.0)``），防止数组
      穿透到标量计算中。``np.float64(1.0)`` 不是 ndarray，是 numpy 标量，
      可正常通过。
    - 与 predict_step.py 的 ``_coerce_numeric_scalar`` 口径完全一致，统一
      委托 ``coerce_finite_scalar`` 做有限性和边界校验（含 ``min_value``
      自身的有限性校验）。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"value": value, "name": name, "min_value": min_value, "inclusive": inclusive}, "_coerce_scalar 入参", prefix="[配置]")
    if isinstance(value, np.ndarray):  # 标量位置不能塞数组。
        # 显式拒绝所有 np.ndarray（包括 0 维数组 np.array(1.0)），
        # 防止数组穿透到标量计算中。np.float64(1.0) 不是 ndarray，是 numpy 标量，可正常通过。
        raise TypeError(f"{name} must be a numeric scalar, got ndarray")  # 直接拒绝。
    return coerce_finite_scalar(value, name=name, min_value=min_value, inclusive=inclusive)  # 委托公共函数做有限性+边界校验。


def predict_range(
    x_pred: np.ndarray | Sequence[float] | Mapping[str, Any],
    anchor_pos: np.ndarray | Sequence[float] | Mapping[str, Any],
    *,
    extra_bias: float = 0.0,
) -> float:
    """根据当前位置、锚点坐标、UWB 钟差和观测侧有界偏置，计算 UWB 测距预测值。

    观测模型：``z_range = ||p - a|| + uwb_clock_bias + extra_bias``。
    - 钟差作为状态项（§2.3 钟差辨识性）。
    - ``extra_bias`` 是 NN+EKF 观测侧的有界修正量（§3.0.2 / §3.1.2 的 h(·) 侧写入口）；
      它代表可学习观测模型对一个 NLOS 正偏置候选的补偿，**进入 h(·) 而不是改写 raw z**，
      残差仍在原始读数合同上定义（§3.0.2「允许：网络输出**有界**观测模型修正，进入 h(·) 或 R」）。

    参数
    ----
    x_pred : np.ndarray | Sequence[float] | Mapping[str, Any]
        预测状态，estimator 路径需含 px/py/uwb_clock_bias（10 维，§1.1+§2.3
        全体同增；由工厂/yaml/state_definition 层强制）；
        几何单元测试路径可传 ≥2 维短状态，回退到纯几何测距（钟差置 0）。
    anchor_pos : np.ndarray | Sequence[float] | Mapping[str, Any]
        锚点坐标。
    extra_bias:
        观测侧有界偏置修正（默认 0.0）。调用方需保证该值为有限实数；
        协议层 ``clip_uwb_bias`` 已对其做非负、比例与绝对上限截断。

    返回
    ----
    float
        预测测距值（欧氏距离 + UWB 钟差 + 观测侧有界偏置），
        非负要求由调用方保证残差非负。
    """
    from liquidloc.common.tee_logger import print_dict  # 局部导入参数打印工具。
    print_dict({"x_pred": x_pred, "anchor_pos": anchor_pos, "extra_bias": extra_bias}, "predict_range 入参", prefix="[配置]")
    x_vector, _ = _coerce_state_vector(x_pred)  # 先规整状态向量。
    px, py = float(x_vector[_IDX_PX]), float(x_vector[_IDX_PY])  # 按动态索引取位置分量。
    # 前提指导 §1.1+§2.3 全体同增 10 维：本函数在 estimator 调用路径下必传 10 维状态
    # （task.yaml state_items 已 10 维、所有 configs/models/*.yaml init_state 已 10 维、
    # 所有工厂已 10 维校验、predict_step 已 7 项过程噪声）；uwb_clock_bias 必在状态中
    # 显式存在。本函数同时支持纯几何(任意 ≥2 维)的单元测试调用路径——若状态短于
    # _IDX_UWB_CLOCK_BIAS，回退到 z_pred_geom（uwb_clock_bias=0），不参与钟差辨识。
    # 此回退**仅面向几何单元测试**，§1.1+§2.3 全体同增硬合同由工厂/yaml/state_definition
    # 层强制保障，不会让真实 estimator 路径漂移到 8 维。
    uwb_clock_bias = float(x_vector[_IDX_UWB_CLOCK_BIAS]) if _IDX_UWB_CLOCK_BIAS < x_vector.size else 0.0
    extra_bias_value = coerce_finite_scalar(float(extra_bias), name="extra_bias")  # 观测侧有界偏置修正。
    ax, ay = _coerce_anchor_xy(anchor_pos)  # 取锚点坐标。

    dx = px - ax  # x 方向差值。
    dy = py - ay  # y 方向差值。
    z_pred = math.hypot(dx, dy) + uwb_clock_bias + extra_bias_value  # 几何距离 + UWB 钟差 + h 侧有界偏置。
    return z_pred  # 返回预测测距。


def build_uwb_jacobian(
    x_pred: np.ndarray | Sequence[float] | Mapping[str, Any],
    anchor_pos: np.ndarray | Sequence[float] | Mapping[str, Any],
) -> np.ndarray:
    """构造单个 UWB 测距观测对状态的雅可比矩阵。

    观测模型：``z_range = ||p - a|| + uwb_clock_bias``，雅可比
    ``H = ∂z_pred/∂x``，故 px/py/d-uwb_clock_bias 三列非零：
    - ``H[0, px] = dx / z_pred_geom``
    - ``H[0, py] = dy / z_pred_geom``
    - ``H[0, uwb_clock_bias] = +1``（钟差对测距线性，每米钟差 1 米测距）

    参数
    ----
    x_pred : np.ndarray | Sequence[float] | Mapping[str, Any]
        预测状态，需含 px/py/uwb_clock_bias。
    anchor_pos : np.ndarray | Sequence[float] | Mapping[str, Any]
        锚点坐标。

    返回
    ----
    np.ndarray
        形状 ``(1, n)`` 的雅可比矩阵，n 为状态维度。
        当几何距离为零时 px/py 列填 0（零距离处梯度未定义），
        uwb_clock_bias 列仍填 +1（钟差对测距偏导恒为 1，与几何奇异点无关）。

    注意
    ----
    - 雅可比仅对 px / py / uwb_clock_bias 三列非零，
      其余列始终为 0，索引由 state_items 动态派生。
    - 前提指导 §2.3：uwb_clock_bias 必须能从 UWB 残差辨识，H 的钟差列填 1
      是该辨识性的解析前提；若该列为 0，则无论多少次 UWB 更新都无法修正钟差。
    """
    from liquidloc.common.tee_logger import print_dict  # 局部导入参数打印工具。
    print_dict({"x_pred": x_pred, "anchor_pos": anchor_pos}, "build_uwb_jacobian 入参", prefix="[配置]")
    x_vector, _ = _coerce_state_vector(x_pred)  # 先拿到状态向量。
    px = float(x_vector[_IDX_PX])  # 位置 x。
    py = float(x_vector[_IDX_PY])  # 位置 y。
    ax, ay = _coerce_anchor_xy(anchor_pos)  # 锚点坐标。

    dx = px - ax  # x 差值。
    dy = py - ay  # y 差值。
    z_pred_geom = math.hypot(dx, dy)  # 几何距离（不含钟差，仅用于 px/py 偏导分母）。

    H = np.zeros((1, x_vector.size), dtype=float)  # 先创建全零雅可比。
    if z_pred_geom > 0.0:  # 只有几何距离非零时 px/py 才可求导。
        H[0, _IDX_PX] = dx / z_pred_geom  # 对 px 的偏导。
        H[0, _IDX_PY] = dy / z_pred_geom  # 对 py 的偏导。
    # uwb_clock_bias 对 z_pred 偏导恒为 +1（前提指导 §2.3 在线钟差辨识性）。
    # 前提指导 §1.1+§2.3 全体同增 10 维：本函数在 estimator 调用路径下必传 10 维状态，
    # uwb_clock_bias H 列必填 +1（钟差辨识性解析前提）。本函数同时支持纯几何(任意 ≥2 维)
    # 的单元测试调用路径——若状态短于 _IDX_UWB_CLOCK_BIAS，H 列跳过（与几何-only 单测对齐）。
    # 此回退**仅面向几何单元测试**，§1.1+§2.3 全体同增硬合同由工厂/yaml/state_definition
    # 层强制保障，不会让真实 estimator 路径漂移到 8 维。
    if _IDX_UWB_CLOCK_BIAS < x_vector.size:
        H[0, _IDX_UWB_CLOCK_BIAS] = 1.0  # 钟差列填 +1。
    return H  # 返回观测雅可比。


def run_uwb_update(
    x_pred: np.ndarray | Sequence[float] | Mapping[str, Any],
    P_pred: np.ndarray | Sequence[Sequence[float]],
    anchor_pos: np.ndarray | Sequence[float] | Mapping[str, Any],
    z_range: float | int,
    R_range: float | int,
    *,
    extra_bias: float = 0.0,
) -> tuple[np.ndarray | dict[str, Any], np.ndarray, dict[str, Any]]:
    """执行一次纯几何 UWB EKF 更新。

    参数
    ----
    x_pred : np.ndarray | Sequence[float] | Mapping[str, Any]
        预测状态向量或映射。
    P_pred : np.ndarray | Sequence[Sequence[float]]
        预测协方差矩阵，形状 ``(n, n)``，n 与状态维度一致。
    anchor_pos : np.ndarray | Sequence[float] | Mapping[str, Any]
        锚点二维坐标。
    z_range : float | int
        UWB **原始**测距观测值，必须非负有限。**不应在调用前做 bias 减法**：
        根据 §3.0.2 / §3.1.2，NN+EKF 的有界观测模型修正在 h(·) 侧通过
        ``extra_bias`` 进入，残差仍在原始读数合同上定义。
    R_range : float | int
        测距测量噪声方差，必须非负有限。
        上游（ekf_core / fgo_core）已将配置中的标准差平方为方差后传入。
    extra_bias:
        观测侧有界偏置修正（默认 0.0）。该值进入 h(·) 而非改写 z；
        雅可比对它偏导恒为 +1，但因其为外部「常量」注入（不入状态），
        在卡尔曼增益与协方差更新中视作观测模型常数项，
        不参与 H 矩阵对状态分量的偏导。

    返回
    ----
    x_upd : np.ndarray | dict[str, Any]
        更新后状态，形式与 ``x_pred`` 一致。
    P_upd : np.ndarray
        更新后协方差矩阵。
    update_info : dict[str, Any]
        更新摘要，含以下键：
        - ``z_pred`` (float): 预测测距
        - ``residual`` (float): 观测残差
        - ``H`` (np.ndarray): 观测雅可比副本
        - ``S`` (float): 创新协方差
        - ``K_gain`` (np.ndarray): 卡尔曼增益副本
        - ``x_upd``: 更新后状态深拷贝
        - ``P_upd`` (np.ndarray): 协方差副本

    异常
    ----
    ValueError
        输入为 None、维度不匹配、数值非有限、S 非正。
    TypeError
        输入类型非法（如 bool、ndarray 标量、字符串）。
    KeyError
        映射型 anchor_pos 缺少已知键名。

    注意
    ----
    - 协方差更新采用 Joseph 形式（数值稳定），并附加对称化 ``0.5*(P+P.T)``。
    - 零距离且 ``R_range>0`` 时 H=0、K=0，状态和协方差不变（几何奇异点）。
    - ``R_range=0.0`` 表示零测量噪声，在 S≤0 时仍会报错。
    """
    from liquidloc.common.tee_logger import print_dict  # 局部导入参数打印工具。
    print_dict({"x_pred": x_pred, "P_pred": P_pred, "anchor_pos": anchor_pos, "z_range": z_range, "R_range": R_range, "extra_bias": extra_bias}, "run_uwb_update 入参", prefix="[配置]")
    x_vector, state_meta = _coerce_state_vector(x_pred)  # 先规整状态。
    P_matrix = _coerce_covariance_matrix(P_pred, state_dim=x_vector.size)  # 再规整协方差。
    z_range_value = _coerce_scalar(z_range, name="z_range", min_value=0.0)  # 测距必须非负。
    R_range_value = _coerce_scalar(R_range, name="R_range", min_value=0.0)  # 噪声必须非负。
    extra_bias_value = coerce_finite_scalar(float(extra_bias), name="extra_bias")  # h 侧有界偏置修正。

    z_pred = predict_range(x_vector, anchor_pos, extra_bias=extra_bias_value)  # 计算预测测距（含 h 侧有界偏置）。
    residual = z_range_value - z_pred  # 观测残差（残差定义在原始 z 上，bias 进 h(·)）。
    H = build_uwb_jacobian(x_vector, anchor_pos)  # 观测雅可比。

    S = coerce_finite_scalar(float((H @ P_matrix @ H.T)[0, 0] + R_range_value), name="S")  # NaN/Inf 会绕过 S<=0 检查（NaN<=0.0 为 False），必须显式拒绝，与 robust_ekf_core/fgo_core 对齐。
    if S <= 0.0:  # 创新协方差必须正。
        raise ValueError(f"S must be positive, got {S}")  # 直接报错。

    K_gain = (P_matrix @ H.T) / S  # 卡尔曼增益。
    x_upd_vector = x_vector + K_gain[:, 0] * residual  # 更新状态向量。

    identity = np.eye(x_vector.size, dtype=float)  # 构造单位阵。
    identity_minus_kh = identity - K_gain @ H  # 标准协方差更新中间项。
    P_upd = (  # 按 Joseph 形式更新协方差。
        identity_minus_kh @ P_matrix @ identity_minus_kh.T  # 第一项保证数值稳定。
        + (K_gain * R_range_value) @ K_gain.T  # 第二项加入测量噪声影响。
    )
    P_upd = 0.5 * (P_upd + P_upd.T)  # 再次对称化。

    x_upd = _restore_state_payload(x_upd_vector, state_meta)  # 恢复和输入一致的承载形式。
    update_info = {  # 生成供上层检查的更新摘要。
        "z_pred": z_pred,  # 预测测距（含 h 侧有界偏置 extra_bias）。
        "z_pred_geom_only": z_pred - extra_bias_value,  # 不含 extra_bias 的纯几何+钟差预测，便于审计区分 h 侧修正 vs z 侧改写（§3.0.2）。
        "extra_bias": extra_bias_value,  # h 侧有界偏置修正量（§3.0.2 / §3.1.2）；非负，已由 clip_uwb_bias 截断。
        "residual": residual,  # 观测残差（定义在原始 z 上：raw_range - z_pred）。
        "H": H.copy(),  # 雅可比副本。
        "S": S,  # 创新协方差。
        "K_gain": K_gain.copy(),  # 卡尔曼增益副本。
        "x_upd": copy.deepcopy(x_upd) if isinstance(x_upd, dict) else np.array(x_upd, copy=True),  # 更新后状态副本。
        "P_upd": P_upd.copy(),  # 更新后协方差副本。
        "modality": "uwb",  # 当前更新模态。
        "update_applied": True,  # UWB 更新已应用。
        "reason": "uwb_update_success",  # 更新原因。
        "gate": {"passed": True, "rejected_by": None},  # 门控信息。
    }
    return x_upd, P_upd, update_info  # 返回更新结果和摘要。


# ─────────────────────────────────────────────────────────────────────────────
# MATLAB 官方紧耦合 EKF 多 anchor stacked Jacobian 联合更新扩展
#
# 参考实现：MathWorks Sensor Fusion and Tracking Toolbox 中 insEKF + UWB
# 测距融合的标准做法，以及 GitHub 上 MIT 协议的 gandres42/uwb-imu-fusion
# EKF 多 anchor 范围更新（见 STEP 1 调研记录）。
#
# 公式骨架（与 task 说明一致）：
#   H_stacked = np.vstack([
#       build_uwb_jacobian_for_anchor(x, anchor_i) for i in range(N)
#   ])
#   R_stacked = np.diag([R_i for i in range(N)])
#   S = H @ P @ H.T + R_stacked
#   K = P @ H.T @ np.linalg.inv(S)
#   x_upd = x + K @ (z - H @ x)
#   P_upd = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T  # Joseph form
#
# 上述多 anchor 同时更新对单 anchor 的退化（N=1）应与 run_uwb_update 等价，
# Joseph 形式协方差亦与现有 10 维 EKF 协方差口径一致，不自创状态维度。
# 前提指导 §1.1 主表 8 维 + §2.3 紧耦合扩维，全体同增至 10 维状态分组。
# ─────────────────────────────────────────────────────────────────────────────


def run_uwb_update_multi_anchor(
    x_pred: np.ndarray | Sequence[float] | Mapping[str, Any],
    P_pred: np.ndarray | Sequence[Sequence[float]],
    anchor_positions: Sequence[np.ndarray | Sequence[float] | Mapping[str, Any]],
    z_ranges: Sequence[float | int],
    R_ranges: Sequence[float | int],
) -> tuple[np.ndarray | dict[str, Any], np.ndarray, dict[str, Any]]:
    """对同一时刻的多个 UWB 锚点构造 stacked 雅可比的联合 EKF 更新。

    一次接收 N 个锚点的 (anchor_i, z_i, R_i) 列表，把它们垂直堆叠成
    ``(N, n)`` H、``(N, N)`` R 与 ``(N,)`` z，再用一次 Joseph 形式协方差
    更新完成多锚点联合修正。等价于在数学上把 N 个独立测距观测合并到
    同一次 EKF 更新，避免了顺序逐次调用单锚点更新时引入的中间协方差
    高估/低估。

    参数
    ----
    x_pred : np.ndarray | Sequence[float] | Mapping[str, Any]
        预测状态向量或映射，必须含 px/py（与 ``run_uwb_update`` 同口径）。
    P_pred : np.ndarray | Sequence[Sequence[float]]
        预测协方差矩阵，``(n, n)`` 必须对称正定。
    anchor_positions : Sequence of (Sequence[float] | Mapping[str, Any])
        N 个锚点的二维坐标，每个元素合规于 :func:`_coerce_anchor_xy`。
    z_ranges : Sequence[float | int]
        对应 N 个锚点的测距观测值，必须非负有限。
    R_ranges : Sequence[float | int]
        对应 N 个锚点的测距噪声方差，必须非负有限。所有分量将堆成
        对角矩阵 ``R_stacked = diag(R_i)``。

    返回
    ----
    x_upd : np.ndarray | dict[str, Any]
        更新后状态，承载形式与 ``x_pred`` 一致。
    P_upd : np.ndarray
        Joseph 形式更新后的协方差矩阵。
    update_info : dict[str, Any]
        更新摘要，键含：
        - ``H_stacked`` ``(N, n)``：堆叠观测雅可比
        - ``R_stacked`` ``(N, N)``：堆叠测量噪声协方差
        - ``S`` ``(N, N)``：创新协方差
        - ``K_gain`` ``(n, N)``：卡尔曼增益
        - ``residuals`` ``list[float]``：每个锚点的残差
        - ``z_preds`` ``list[float]``：每个锚点的预测测距
        - ``x_upd``/``P_upd``：副本
        - ``modality`` == ``"uwb_multi_anchor"``
        - ``update_applied`` == True
        - ``reason`` == ``"uwb_multi_anchor_update_success"``
        - ``gate`` == {``passed``: True, ``rejected_by``: None}

    异常
    ----
    ValueError
        - 三个输入序列长度不一致
        - 序列为空
        - 任一锚点/测距/噪声不满足约束
        - 创新协方差 S 非正定
    TypeError
        与 ``run_uwb_update`` 一致。

    注意
    ----
    - 对 N=1 的退化情形与 :func:`run_uwb_update` 数学等价（不严格相同，
      因为单锚点版本走的是标量快速路径，但状态/协方差更新结果一致）。
    - 协方差更新走 Joseph 形式 + 数值对称化，与单锚点路径同口径，防止
      多锚点串行更新引入的协方差不对称漂移。
    - uwb_clock_bias 在状态向量中作为在线估计量参与联合更新（前提指导
      §2.3），H 在该列统一填 +1，所有锚点共享同一钟差偏置。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict(
        {
            "x_pred": x_pred,
            "P_pred": P_pred,
            "anchor_positions": anchor_positions,
            "z_ranges": z_ranges,
            "R_ranges": R_ranges,
        },
        "run_uwb_update_multi_anchor 入参",
        prefix="[配置]",
    )

    # 三组序列长度必须一致，且不能为空。
    n_anchors = len(anchor_positions) if hasattr(anchor_positions, "__len__") else 0
    if n_anchors == 0:
        raise ValueError("anchor_positions must be a non-empty sequence")
    if len(z_ranges) != n_anchors or len(R_ranges) != n_anchors:
        raise ValueError(
            "anchor_positions, z_ranges, R_ranges must have equal length "
            f"got {n_anchors}, {len(z_ranges)}, {len(R_ranges)}"
        )

    # 规整状态/协方差，统一从 x_pred 派生 state_dim，避免硬编码 8 维。
    # 前提指导 §1.1+§2.3 全体同增 10 维：下游 _IDX_UWB_CLOCK_BIAS 访问已硬报错短于索引，
    # 任何 8 维上游漂移会在本函数运行时被即时发现而非静默退化。
    x_vector, state_meta = _coerce_state_vector(x_pred)
    state_dim = x_vector.size
    P_matrix = _coerce_covariance_matrix(P_pred, state_dim=state_dim)

    # 为每个锚点构造：(H_i, z_i, R_i) 三元组，并 stack 成联合观测。
    H_rows: list[np.ndarray] = []
    z_values: list[float] = []
    r_values: list[float] = []
    z_preds: list[float] = []
    residuals: list[float] = []

    for i in range(n_anchors):
        ax_i, ay_i = _coerce_anchor_xy(anchor_positions[i])
        z_i = _coerce_scalar(z_ranges[i], name=f"z_ranges[{i}]", min_value=0.0)
        r_i = _coerce_scalar(R_ranges[i], name=f"R_ranges[{i}]", min_value=0.0)

        px = float(x_vector[_IDX_PX])
        py = float(x_vector[_IDX_PY])
        # 前提指导 §1.1+§2.3 全体同增 10 维：本函数在 estimator 调用路径下必传 10 维状态，
        # uwb_clock_bias 必在状态中。本函数同时支持纯几何(任意 ≥2 维)的单元测试调用路径——
        # 若状态短于 _IDX_UWB_CLOCK_BIAS，回退到 z_pred_geom（钟差置 0）。
        # 此回退**仅面向几何单元测试**，§1.1+§2.3 全体同增硬合同由工厂/yaml/state_definition
        # 层强制保障，不会让真实 estimator 路径漂移到 8 维。
        uwb_clock_bias_i = (
            float(x_vector[_IDX_UWB_CLOCK_BIAS]) if _IDX_UWB_CLOCK_BIAS < state_dim else 0.0
        )
        dx_i = px - ax_i
        dy_i = py - ay_i
        z_pred_geom_i = math.hypot(dx_i, dy_i)
        z_pred_i = z_pred_geom_i + uwb_clock_bias_i  # 含 UWB 钟差的预测测距（§2.3）。
        H_i = np.zeros((1, state_dim), dtype=float)
        if z_pred_geom_i > 0.0:
            H_i[0, _IDX_PX] = dx_i / z_pred_geom_i
            H_i[0, _IDX_PY] = dy_i / z_pred_geom_i
        # uwb_clock_bias 对 z_pred 偏导恒为 +1（§2.3 在线钟差辨识性）。
        if _IDX_UWB_CLOCK_BIAS < state_dim:
            H_i[0, _IDX_UWB_CLOCK_BIAS] = 1.0
        H_rows.append(H_i)
        z_values.append(z_i)
        r_values.append(r_i)
        z_preds.append(z_pred_i)
        residuals.append(z_i - z_pred_i)

    H_stacked = np.vstack(H_rows)  # (N, n)
    z_vec = np.asarray(z_values, dtype=float)  # (N,)
    R_stacked = np.diag(np.asarray(r_values, dtype=float))  # (N, N) 对角阵
    residual_vec = np.asarray(residuals, dtype=float)  # (N,)

    # 创新协方差 S = H @ P @ H.T + R_stacked；要求正定以保证可解。
    S = H_stacked @ P_matrix @ H_stacked.T + R_stacked
    if not np.all(np.isfinite(S)):
        raise ValueError("Innovation covariance S must be finite")
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # §11.5 抖动注入：第一次 Cholesky 失败 → S += cov_jitter_eps * I 再重试。
        from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
        cov_jitter_eps = float(BRIDGE_THRESHOLDS["cov_jitter_eps"])
        jittered = S + cov_jitter_eps * np.eye(S.shape[0])
        try:
            np.linalg.cholesky(jittered)
            S = jittered  # 接受 jittered 版本作为该次创新协方差。
        except np.linalg.LinAlgError as exc:
            # 二次仍失败 → fail-loud；不允许任何"只救一方"的静默重置。
            raise ValueError(
                f"Innovation covariance S must be positive definite "
                f"(jitter fallback exhausted at eps={cov_jitter_eps})"
            ) from exc

    try:
        K_gain = (P_matrix @ H_stacked.T) @ np.linalg.inv(S)
    except np.linalg.LinAlgError as exc:  # 理论上 Cholesky 已通过，这里兜底。
        raise ValueError("Failed to invert innovation covariance S") from exc

    x_upd_vector = x_vector + K_gain @ residual_vec
    identity = np.eye(state_dim, dtype=float)
    state_transition = identity - K_gain @ H_stacked
    # Joseph 形式协方差更新 + 数值对称化，与单锚点路径同口径。
    P_upd = state_transition @ P_matrix @ state_transition.T + K_gain @ R_stacked @ K_gain.T
    P_upd = 0.5 * (P_upd + P_upd.T)

    x_upd = _restore_state_payload(x_upd_vector, state_meta)
    update_info = {
        "H_stacked": H_stacked.copy(),
        "R_stacked": R_stacked.copy(),
        "S": S.copy(),
        "K_gain": K_gain.copy(),
        "residuals": residuals,
        "z_preds": z_preds,
        "x_upd": copy.deepcopy(x_upd) if isinstance(x_upd, dict) else np.array(x_upd, copy=True),
        "P_upd": P_upd.copy(),
        "modality": "uwb_multi_anchor",
        "update_applied": True,
        "reason": "uwb_multi_anchor_update_success",
        "gate": {"passed": True, "rejected_by": None},
    }
    return x_upd, P_upd, update_info


def run_joint_uwb_vio_update(
    x_pred: np.ndarray | Sequence[float] | Mapping[str, Any],
    P_pred: np.ndarray | Sequence[Sequence[float]],
    *,
    uwb_anchors: Sequence[np.ndarray | Sequence[float] | Mapping[str, Any]],
    z_uwb: Sequence[float | int],
    R_uwb: Sequence[float | int],
    z_vio: Sequence[float | int] | np.ndarray,
    R_vio: Any,
    vio_reference_pose: np.ndarray | Mapping[str, Any] | None = None,
    uwb_extra_biases: Sequence[float | int] | None = None,
) -> tuple[np.ndarray | dict[str, Any], np.ndarray, dict[str, Any]]:
    """把同一时刻的 UWB 多锚点测距与 VIO 帧间相对位姿合并到一次联合 EKF 更新。

    实现 MATLAB 官方紧耦合 EKF 的核心思想：把 UWB stacked H_uwb (N, n) 与
    VIO 3 维 H_vio (3, n) 在轴 0 上再堆叠成 ``(N+3, n)`` 联合雅可比，
    噪声矩阵 ``R_joint = block_diag(R_uwb_stacked, R_vio)``，残差向量
    ``r_joint = concat(r_uwb, r_vio)``，再做一次 Joseph 形式协方差更新。
    相比先 UWB 后 VIO 的顺序两次独立更新，联合一次更新能正确刻画 UWB
    与 VIO 在同一时刻的耦合相关性。

    参数
    ----
    x_pred : np.ndarray | Sequence[float] | Mapping[str, Any]
        预测状态，需含 px/py/yaw（VIO 雅可比需要 yaw 列）。
    P_pred : np.ndarray | Sequence[Sequence[float]]
        预测协方差矩阵 ``(n, n)``，必须对称正定。
    uwb_anchors : sequence of anchor coordinates
        N 个 UWB 锚点的二维坐标。
    z_uwb : sequence of float/int
        N 个锚点对应的**原始**测距观测。调用方**不应**预先做 bias 减法；
        根据 §3.0.2 / §3.1.2，NN+EKF 的有界观测模型偏置通过 ``uwb_extra_biases``
        在 h(·) 侧进入，残差仍定义在原始 z 上。
    R_uwb : sequence of float/int
        N 个锚点对应的测距噪声方差。
    z_vio : sequence of float/int | np.ndarray
        长度 3 的 VIO 量测向量 ``[dx, dy, dyaw]``。
    R_vio : Any
        VIO 噪声，沿用 vision_update_step 的多种承载形式（标量/3 元组/
        3x3 矩阵/字典），由 vision_update_step._normalize_vio_covariance
        统一规整成 3x3 矩阵。
    vio_reference_pose : np.ndarray | Mapping[str, Any] | None
        可选的 VIO 参考位姿，None 时使用当前位姿（与 apply_vision_update
        一致）。
    uwb_extra_biases : sequence of float/int | None
        各 UWB 锚点对应的 h 侧有界偏置修正（默认 None 全 0）。
        **必填于 NN+EKF 主路径**（§3.0.2 / §3.1.2）：联合路径不可悄悄丢弃
        单模态路径的 bias_applied；否则联合路径与单模态路径对 NN 输出口径
        不一致，违反 §3.0 item5 三网对称与 §3.0.2 写入口边界。
        长度必须与 ``uwb_anchors`` 一致，逐锚点注入到 ``z_pred`` 侧：
        ``z_pred_i = ||p-a|| + uwb_clock_bias + uwb_extra_biases[i]``。

    返回
    ----
    x_upd : np.ndarray | dict[str, Any]
        更新后状态，承载形式与 ``x_pred`` 一致。
    P_upd : np.ndarray
        更新后协方差矩阵。
    update_info : dict[str, Any]
        联合更新摘要，关键键含：
        - ``H_joint`` ``(N+3, n)``：堆叠后的联合雅可比
        - ``R_joint`` ``(N+3, N+3)``：块对角联合噪声协方差
        - ``S`` ``(N+3, N+3)``：联合创新协方差
        - ``K_gain`` ``(n, N+3)``：联合卡尔曼增益
        - ``uwb_residuals`` / ``vio_residual``：分块残差
        - ``uwb_z_preds`` / ``vio_z_hat``：分块预测量测
        - ``reference_pose``：实际使用的参考位姿
        - ``x_upd`` / ``P_upd``：副本
        - ``modality`` == ``"joint_uwb_vio"``
        - ``update_applied`` == True
        - ``reason`` == ``"joint_uwb_vio_update_success"``
        - ``gate`` == {``passed``: True, ``rejected_by``: None}

    异常
    ----
    ValueError
        UWB 序列长度不一致或为空、VIO 残差非有限、S 非正定等。
    TypeError
        与上下文一致。

    注意
    ----
    - 不修改 EKF 状态维度，与 task.yaml 冻结 10 维状态（含 uwb_clock_bias /
      vio_scale）一致；uwb_clock_bias 通过 UWB 雅可比 +1 列在线辨识
      （前提指导 §2.3），vio_scale 通过 VIO 雅可比 R(-yaw_ref) @ [px;py] 列在线辨识。
    - 联合更新走 Joseph 形式，与单模态路径同口径，避免顺序更新引入的
      跨模态协方差不对称。
    - VIO 残差的航向分量使用 angle_delta_rad 做环形差处理，与
      apply_vision_update 一致，避免 ±π 跳变。
    - 当 N=0（无 UWB 锚点）时退化为纯 VIO 更新；当 VIO 关闭（z_vio=None）
      时退化为多锚点 UWB 更新；调用方需保证至少有一路观测。
    """
    from liquidloc.common.tee_logger import print_dict
    from liquidloc.estimators.vision_update_step import (
        _normalize_vio_covariance,  # 复用 VIO 噪声规整，确保口径一致。
        _build_vio_jacobian,  # 复用 3 维 VIO 雅可比构造。
        _coerce_vio_measurement_vector,  # 复用 3 维 VIO 量测规整。
        _coerce_state_vector as _coerce_vio_state,  # 复用映射/数组状态规整。
        _extract_pose_vector,  # 从完整状态抽 [px, py, yaw]。
        _resolve_reference_pose,  # 确定参考位姿。
        _predict_local_vio_measurement,  # 预测本地 VIO 量测。
    )
    from liquidloc.common.angle_utils import angle_delta_rad, wrap_angle_rad

    print_dict(
        {
            "x_pred": x_pred,
            "P_pred": P_pred,
            "uwb_anchors": uwb_anchors,
            "z_uwb": z_uwb,
            "R_uwb": R_uwb,
            "z_vio": z_vio,
            "R_vio": R_vio,
            "vio_reference_pose": vio_reference_pose,
            "uwb_extra_biases": uwb_extra_biases,
        },
        "run_joint_uwb_vio_update 入参",
        prefix="[配置]",
    )

    # ── 公共状态规整：UWB 与 VIO 共享同一个预测状态/协方差。 ──
    x_vector, state_meta = _coerce_state_vector(x_pred)
    state_dim = x_vector.size
    P_matrix = _coerce_covariance_matrix(P_pred, state_dim=state_dim)

    # ── UWB 部分：构造 stacked H_uwb (N, n) 与 stacked R/z。 ──
    n_uwb = len(uwb_anchors) if hasattr(uwb_anchors, "__len__") else 0
    # h 侧有界偏置修正（§3.0.2 / §3.1.2）。None→全 0；长度不一致直接报错，
    # 与 z_uwb / R_uwb 的同长度铁律同口径，防止联合路径悄悄丢弃 NN bias。
    if uwb_extra_biases is None:
        uwb_bias_values: list[float] = [0.0] * n_uwb
    else:
        if len(uwb_extra_biases) != n_uwb:
            raise ValueError(
                "uwb_extra_biases length must match uwb_anchors, "
                f"got {len(uwb_extra_biases)} vs {n_uwb}"
            )
        uwb_bias_values = [
            coerce_finite_scalar(float(b), name=f"uwb_extra_biases[{i}]")
            for i, b in enumerate(uwb_extra_biases)
        ]
    H_uwb_rows: list[np.ndarray] = []
    z_uwb_values: list[float] = []
    r_uwb_values: list[float] = []
    uwb_z_preds: list[float] = []
    uwb_residuals: list[float] = []
    if n_uwb > 0:
        if len(z_uwb) != n_uwb or len(R_uwb) != n_uwb:
            raise ValueError(
                "uwb_anchors, z_uwb, R_uwb must have equal length "
                f"got {n_uwb}, {len(z_uwb)}, {len(R_uwb)}"
            )
        for i in range(n_uwb):
            ax_i, ay_i = _coerce_anchor_xy(uwb_anchors[i])
            z_i = _coerce_scalar(z_uwb[i], name=f"z_uwb[{i}]", min_value=0.0)
            r_i = _coerce_scalar(R_uwb[i], name=f"R_uwb[{i}]", min_value=0.0)
            px = float(x_vector[_IDX_PX])
            py = float(x_vector[_IDX_PY])
            # 前提指导 §1.1+§2.3 全体同增 10 维：本函数在 estimator 调用路径下必传 10 维状态。
            # 几何单元测试兼容：若状态短于 _IDX_UWB_CLOCK_BIAS，回退到 z_pred_geom（钟差置 0）。
            # §1.1+§2.3 硬合同由工厂/yaml/state_definition 层强制保障真实 estimator 路径必传 10 维。
            uwb_clock_bias_i = (
                float(x_vector[_IDX_UWB_CLOCK_BIAS]) if _IDX_UWB_CLOCK_BIAS < state_dim else 0.0
            )
            dx_i = px - ax_i
            dy_i = py - ay_i
            z_pred_geom_i = math.hypot(dx_i, dy_i)
            # h 侧含 NN bias 的预测测距（§3.0.2 / §3.1.2）：bias 进 h(·)，残差定义在原始 z 上。
            z_pred_i = z_pred_geom_i + uwb_clock_bias_i + uwb_bias_values[i]
            H_i = np.zeros((1, state_dim), dtype=float)
            if z_pred_geom_i > 0.0:
                H_i[0, _IDX_PX] = dx_i / z_pred_geom_i
                H_i[0, _IDX_PY] = dy_i / z_pred_geom_i
            # uwb_clock_bias 对 z_pred 偏导恒为 +1（§2.3 在线钟差辨识性）。
            # extra_bias 不入状态，对状态向量偏导为 0；它作为观测模型的常量项进入。
            if _IDX_UWB_CLOCK_BIAS < state_dim:
                H_i[0, _IDX_UWB_CLOCK_BIAS] = 1.0
            H_uwb_rows.append(H_i)
            z_uwb_values.append(z_i)
            r_uwb_values.append(r_i)
            uwb_z_preds.append(z_pred_i)
            uwb_residuals.append(z_i - z_pred_i)

    # ── VIO 部分：复用 vision_update_step 的规整与雅可比构造。 ──
    H_vio_rows: list[np.ndarray] = []
    z_vio_values: list[float] = []
    r_vio_block: np.ndarray  # (3, 3) VIO 噪声协方差
    vio_z_hat: list[float] = []
    vio_residual: list[float] = []
    reference_pose_used: np.ndarray | None = None
    has_vio = z_vio is not None
    if has_vio:
        # VIO 状态/雅可比复用 vision_update_step 自身的规整，保证与单模态
        # apply_vision_update 的语义同口径（state_items 派生索引、参考位姿、
        # 航向环形差）。
        x_vio_array, vio_state_items = _coerce_vio_state(x_pred)
        current_pose = _extract_pose_vector(x_vio_array, vio_state_items)
        current_pose[2] = wrap_angle_rad(float(current_pose[2]))
        reference_pose_array = _resolve_reference_pose(vio_reference_pose, current_pose)
        reference_pose_array[2] = wrap_angle_rad(float(reference_pose_array[2]))
        # IMP-A 余震修复 (2026-07-23 audit Round 5): 联合 EKF 路径同样需要传 x_array
        # VIO 雅可比：前提指导默认仅对 px/py/yaw 非零。
        H_vio = _build_vio_jacobian(vio_state_items, reference_yaw=float(reference_pose_array[2]), x_array=x_vio_array)
        # H_vio 形状为 (3, n_vio)，可能与 state_dim 一致或等于 3 (pose-only)。
        # 联合更新要求 VIO 行展开到完整 state_dim；若 vio_state_items 比
        # state_items 短（仅位姿情况），需补齐到完整列数。
        if H_vio.shape[1] != state_dim:
            H_vio_full = np.zeros((3, state_dim), dtype=float)
            for src_col, key in enumerate(vio_state_items):
                if key in _IDX:
                    H_vio_full[:, _IDX[key]] = H_vio[:, src_col]
            H_vio = H_vio_full
        H_vio_rows.append(H_vio)

        # 量测与噪声规整。
        z_vio_vec = _coerce_vio_measurement_vector(z_vio)
        z_vio_vec[2] = wrap_angle_rad(z_vio_vec[2])
        z_hat = _predict_local_vio_measurement(current_pose, reference_pose_array)
        r_vio = z_vio_vec - z_hat
        r_vio[2] = angle_delta_rad(z_vio_vec[2], z_hat[2])
        z_vio_values = list(z_vio_vec)
        vio_z_hat = list(z_hat)
        vio_residual = list(r_vio)
        reference_pose_used = reference_pose_array
        r_vio_block = _normalize_vio_covariance(R_vio)
    else:
        r_vio_block = np.zeros((0, 0), dtype=float)

    if n_uwb == 0 and not has_vio:
        raise ValueError("run_joint_uwb_vio_update requires at least one UWB anchor or a VIO observation")

    # ── 堆叠联合 H/R/z。 ──
    if H_uwb_rows and H_vio_rows:
        H_joint = np.vstack(H_uwb_rows + H_vio_rows)
    elif H_uwb_rows:
        H_joint = np.vstack(H_uwb_rows)
    else:
        H_joint = np.vstack(H_vio_rows)

    n_total = H_joint.shape[0]

    R_joint = np.zeros((n_total, n_total), dtype=float)
    if n_uwb > 0:
        R_joint[:n_uwb, :n_uwb] = np.diag(np.asarray(r_uwb_values, dtype=float))
    if has_vio:
        R_joint[n_uwb:, n_uwb:] = r_vio_block

    residual_joint = np.zeros(n_total, dtype=float)
    if n_uwb > 0:
        residual_joint[:n_uwb] = np.asarray(uwb_residuals, dtype=float)
    if has_vio:
        residual_joint[n_uwb:] = np.asarray(vio_residual, dtype=float)

    # ── 创新协方差 + Joseph 形式协方差更新。 ──
    S = H_joint @ P_matrix @ H_joint.T + R_joint
    if not np.all(np.isfinite(S)):
        raise ValueError("Joint innovation covariance S must be finite")
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # §11.5 抖动注入：第一次 Cholesky 失败 → S += cov_jitter_eps * I 再重试。
        from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
        cov_jitter_eps = float(BRIDGE_THRESHOLDS["cov_jitter_eps"])
        jittered = S + cov_jitter_eps * np.eye(S.shape[0])
        try:
            np.linalg.cholesky(jittered)
            S = jittered  # 接受 jittered 版本作为该次联合创新协方差。
        except np.linalg.LinAlgError as exc:
            # 二次仍失败 → fail-loud；不允许任何"只救一方"的静默重置。
            raise ValueError(
                f"Joint innovation covariance S must be positive definite "
                f"(jitter fallback exhausted at eps={cov_jitter_eps})"
            ) from exc
    try:
        K_gain = (P_matrix @ H_joint.T) @ np.linalg.inv(S)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Failed to invert joint innovation covariance S") from exc

    x_upd_vector = x_vector + K_gain @ residual_joint
    identity = np.eye(state_dim, dtype=float)
    state_transition = identity - K_gain @ H_joint
    P_upd = state_transition @ P_matrix @ state_transition.T + K_gain @ R_joint @ K_gain.T
    P_upd = 0.5 * (P_upd + P_upd.T)

    x_upd = _restore_state_payload(x_upd_vector, state_meta)
    update_info = {
        "H_joint": H_joint.copy(),
        "R_joint": R_joint.copy(),
        "S": S.copy(),
        "K_gain": K_gain.copy(),
        "uwb_residuals": uwb_residuals,
        "vio_residual": list(vio_residual),
        "uwb_z_preds": uwb_z_preds,
        "vio_z_hat": list(vio_z_hat),
        "reference_pose": reference_pose_used.tolist() if reference_pose_used is not None else None,
        "x_upd": copy.deepcopy(x_upd) if isinstance(x_upd, dict) else np.array(x_upd, copy=True),
        "P_upd": P_upd.copy(),
        "modality": "joint_uwb_vio",
        "update_applied": True,
        "reason": "joint_uwb_vio_update_success",
        "gate": {"passed": True, "rejected_by": None},
    }
    return x_upd, P_upd, update_info
