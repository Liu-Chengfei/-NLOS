"""
文件：src/liquidloc/estimators/predict_step.py

【文件职责】
这个文件实现纯 IMU 驱动的预测步骤。
它负责把上一时刻状态和惯性输入推进到下一时刻，并同步更新协方差。

【本文件绝对不负责】
不做 UWB 更新，不做视觉更新，不做场景门控，不消费模型输出。

【上游依赖】
state_definition.py、sensors/imu_model.py、common/validation.py。

【下游调用者】
ekf_core.py、fgo_core.py。

【输入对象定义】
- x_prev: 上一时刻状态
- P_prev: 上一时刻协方差
- imu_input: IMU 事件或等价结构
- dt: 时间间隔
- predict_cfg: 预测相关配置

【输出对象定义】
- x_pred: 预测后状态
- P_pred: 预测后协方差

【实现要求】
- 状态顺序必须和 configs/base/task.yaml 对齐。
- 只做几何预测和噪声传播，不引入别的业务语义。

【前提指导 §5.1 — IMU 联合约束】
本文件是五种实际实现的方法（ekf / robust_ekf / fgo / lstm_ekf / liquid_ekf）共享的预测核心。
其中 ekf / robust_ekf / fgo 由 estimator_factory.create_estimator 直接构造（_SUPPORTED = {EKF, ROBUST_EKF, FGO}），
lstm_ekf / liquid_ekf 复用 EKF 估计器外壳并在其上叠加测量侧 NN 控制（见 core_pipeline._NEURAL_METHODS）。
为了让它们在 IMU 模态上做"同一物理叙事的不同推理形式"，必须满足：

1. 真实时间间隔 ``dt`` —— 严格采用 IMU 事件时间戳的差（无定频假设、无标称 dt 兜底），
   偏置随机游走与加速度/陀螺过程噪声也都按这个 ``dt`` 缩放。
2. 偏置作为在线估计状态的一部分 —— ``bax``/``bay``/``bg`` 三维与 ``vx``/``vy``/``px``/``py``
   等动力学状态同列在状态向量里，预测步直接做 ``ax_body = ax - bax``/``ay_body = ay - bay``/
   ``yaw += (gz - bg) * dt`` 的偏置补偿，没有任何方法跳过这一步。
3. 偏置随机游走强度 —— 由 ``predict_cfg.process_noise.accel_bias`` 与
   ``predict_cfg.process_noise.gyro_bias`` 提供（PSD 平方根，单位 state_unit/√s），
   协方差离散化项是 ``σ_b² * dt``。`configs/models/{ekf,robust_ekf,fgo}.yaml` 三者值严格一致
   （accel_bias=0.001，gyro_bias=0.001），五种方法共用同一噪声构造。
4. 重力处理（前提指导 §5.1：全员同一）—— 本文件采用 2D 平面运动模型，
   ``ax``/``ay`` 已经在 sim_materializer.py（1691-1692 行的世界→机体旋转）中
   被定义为水平面（ENU 平面）上的比力分量，**不含重力投影**。
   因此本文件的预测步直接使用 ``ax_body``/``ay_body``（机体系比力）做积分，
   不再做 ``+ g * sin(yaw)`` / ``- g * cos(yaw)`` 之类的重力补偿，也不在世界系
   重力向量上做任何旋转投影。所有方法经由 ``run_predict_step`` 都看到这一致的
   2D 水平比力约定，gravity-explicit / gravity-implicit 在方法之间不会出现差异。
5. 缺测字段（``missing_mask``）—— ``ax``/``ay``/``gz`` 三个分量独立标记是否缺失，
   缺失分量在预测里直接置零（保留偏置补偿痕迹但不更新相关维度协方差的"新息驱动"），
   并跳过对应的雅可比列与过程噪声输入，避免虚假积分或状态被静默拉偏。
"""

from __future__ import annotations  # 允许在类型标注里引用后面定义的名字。

import math  # 用来算三角函数和有限性检查。
from collections.abc import Mapping, Sequence  # 用来判断配置和输入是不是映射。
from typing import Any  # 用于类型标注。
import numpy as np  # 用来做向量和矩阵运算。

from liquidloc.common.angle_utils import wrap_angle_rad  # 用来把 yaw 归一化到统一范围。
from liquidloc.common.validation import coerce_finite_scalar, require_shape  # 有限浮点转换和形状检查。
from liquidloc.estimators.state_definition import state_items
from liquidloc.sensors.imu_model import extract_imu_measurement  # 用来从事件里提取 IMU 三轴量测。


_STATE_ITEMS = state_items  # IMU 预测依赖的冻结状态顺序。
_STATE_DIM = len(_STATE_ITEMS)  # 状态维度。

# 从冻结状态项动态派生索引，消除硬编码数字索引。
# 如果 state_items 顺序变更，这些索引会自动跟随，不再静默出错。
_IDX = {name: i for i, name in enumerate(_STATE_ITEMS)}  # 状态名→索引映射。
_IDX_PX = _IDX["px"]  # x 方向位置索引。
_IDX_PY = _IDX["py"]  # y 方向位置索引。
_IDX_VX = _IDX["vx"]  # x 方向速度索引。
_IDX_VY = _IDX["vy"]  # y 方向速度索引。
_IDX_YAW = _IDX["yaw"]  # 航向角索引。
_IDX_BAX = _IDX["bax"]  # x 方向加速度偏差索引。
_IDX_BAY = _IDX["bay"]  # y 方向加速度偏差索引。
_IDX_BG = _IDX["bg"]  # 陀螺仪偏差索引。
_IDX_UWB_CLOCK_BIAS = _IDX["uwb_clock_bias"]  # UWB 钟差索引（前提指导 §2.3 在线时间偏移与钟差态）。
_IDX_VIO_SCALE = _IDX["vio_scale"]  # VIO 尺度因子索引（前提指导 §2.3 在线外参/尺度因子）。

# 过程噪声分组：前提指导 §1.1/§1.3 默认 8 维状态 + §2.3 紧耦合扩维项（uwb_clock_bias / vio_scale）。
# §2.3 要求"在线估……钟差态"必须"全体同增同模型"——状态向量已扩出 uwb_clock_bias / vio_scale
# （见 task.yaml state_items），predict 步必须给这两维加入过程噪声，使协方差能随时间增长，
# 否则即使更新步雅可比列非零，卡尔曼增益也会因 P 不增长而接近 0，钟差/尺度永远卡在初值，
# 实质等价于"未在线估"，违反 §2.3 的"钟差态与 NLOS 可分性"前提。
_PROCESS_NOISE_KEYS = {
    "pos": (_IDX_PX, _IDX_PY),  # 位置噪声会影响 px/py。
    "vel": (_IDX_VX, _IDX_VY),  # 速度噪声会影响 vx/vy。
    "yaw": (_IDX_YAW,),  # 航向噪声只影响 yaw。
    "accel_bias": (_IDX_BAX, _IDX_BAY),  # 加速度偏置噪声会影响 bax/bay。
    "gyro_bias": (_IDX_BG,),  # 陀螺偏置噪声会影响 bg。
    "uwb_clock_bias": (_IDX_UWB_CLOCK_BIAS,),  # UWB 钟差随机游走噪声（单位: m/√s）。
    "vio_scale": (_IDX_VIO_SCALE,),  # VIO 尺度因子随机游走噪声（无量纲/√s）。
}


def _coerce_numeric_scalar(value: Any, *, name: str,
                           min_value: float | None = None, inclusive: bool = True) -> float:
    """把输入规整成有限的数值标量，并做范围检查。

    委托到 common.validation.coerce_finite_scalar，额外显式拒绝 np.ndarray
    （包括 0 维数组），防止数组穿透到标量计算中。

    参数：
        value: 输入值，必须是数值标量（int、float、np.float64 等），
            不接受 bool、np.bool_ 和 np.ndarray（包括 0 维数组）。
        name: 参数名，用于错误消息。
        min_value: 可选下界，如果提供则校验 scalar >= min_value（或 >）。
        inclusive: 为 True 时使用 >=，为 False 时使用 >。默认 True。

    返回：
        有限浮点数。如果指定了 min_value，则返回值满足下界约束。

    异常：
        TypeError: 输入是 bool、np.ndarray 或非数值类型。
        ValueError: 输入为 None，是 NaN 或无穷大，或低于 min_value。

    注意：
        - 显式拒绝 np.ndarray（包括 0 维数组如 np.array(1.0)），
          防止数组穿透到标量计算中。如果需要接受 0 维数组，
          调用方应先做 float() 转换。
        - np.float64(1.0) 不是 ndarray，是 numpy 标量，可通过 isinstance(Real) 检查，
          因此是合法输入。这与 np.array(1.0)（0 维 ndarray）不同。
    """
    from liquidloc.common.tee_logger import print_dict  # 局部导入参数打印工具。
    print_dict({"value": value, "name": name, "min_value": min_value, "inclusive": inclusive}, "_coerce_numeric_scalar 入参", prefix="[配置]")
    if isinstance(value, np.ndarray):  # 标量位置不能塞数组。
        # 显式拒绝所有 np.ndarray（包括 0 维数组 np.array(1.0)），
        # 防止数组穿透到标量计算中。np.float64(1.0) 不是 ndarray，是 numpy 标量，可正常通过。
        raise TypeError(f"{name} must be a numeric scalar, got ndarray")  # 直接拒绝。
    return coerce_finite_scalar(value, name=name, min_value=min_value, inclusive=inclusive)


def run_predict_step(x_prev: np.ndarray | Sequence[float],
                     P_prev: np.ndarray | Sequence[Sequence[float]],
                     imu_input: Mapping | object,
                     dt: float | int,
                     predict_cfg: Mapping | None) -> tuple[np.ndarray, np.ndarray]:
    """执行一次 IMU 预测，并返回预测后的状态和协方差。

    处理流程：
        1. 校验输入形状（x_prev: (N,), P_prev: (N,N)，N=len(state_items)=10）和 dt > 0
        2. 校验 predict_cfg 是 Mapping（None 转为 {}）
        3. 从 imu_input 提取 ax, ay, gz 并校验为有限数值
        4. 计算车体系→世界系加速度（偏置补偿 + 旋转变换）
        5. 推进状态（带偏置 IMU 积分 + yaw 归一化 + 偏置/钟差/尺度随机游走保持）
        6. 计算状态转移雅可比 F
        7. 构建过程噪声矩阵 Q（noise_value² × dt 对角矩阵，含 uwb_clock_bias/vio_scale）
        8. 协方差传播 P_pred = F @ P_prev @ F.T + Q，数值对称化

    参数：
        x_prev: 上一时刻状态向量，shape (N,)，N=len(state_items)=10，顺序为
            [px, py, vx, vy, yaw, bax, bay, bg, uwb_clock_bias, vio_scale]。
        P_prev: 上一时刻协方差矩阵，shape (N, N)，与 x_prev 维度一致。
        imu_input: IMU 事件（dict 或对象），必须包含 ax, ay, gz。
        dt: 时间间隔，必须严格大于 0。
        predict_cfg: 预测配置，可包含 ``"process_noise"`` 子映射。
            - process_noise: 映射，键为 ``"pos"/"vel"/"yaw"/"accel_bias"/"gyro_bias"/
              ``"uwb_clock_bias"/"vio_scale"``，值为该类噪声的强度参数
              （PSD 平方根，单位: state_unit/√s），不是标准差。
              离散化公式为 Q_i = σ² × dt，将连续时间噪声强度转换为离散时间方差。
              uwb_clock_bias / vio_scale 必须显式配非零值，否则 §2.3 在线时间偏移与
              钟差态前提不成立（详见 _PROCESS_NOISE_KEYS 注释）。

    返回：
        (x_pred, P_pred)
        - x_pred: 预测后状态向量，shape (N,)。
        - P_pred: 预测后协方差矩阵，shape (N, N)，保证对称。

    异常：
        ValueError: dt <= 0 或输入包含 NaN/inf。
        TypeError: dt/imu 字段为 bool 或非数值类型，predict_cfg 不是 Mapping。
        其他: 来自 extract_imu_measurement 和 require_shape 的异常。

    注意：
        - 偏置项在预测步骤中保持不变（随机游走模型）。
        - yaw 在推进后会通过 wrap_angle_rad 归一化到 [-π, π)。
        - 协方差传播后做数值对称化 ``0.5 * (P + P.T)``，避免浮点漂移。
        - 过程噪声输入语义是 PSD 平方根（噪声强度，单位: state_unit/√s），
          离散化为 σ² × dt 得到离散时间方差。这与 measurement_noise 的语义不同：
          measurement_noise 是标准差，直接平方为方差（无 dt 因子）。
          配置文件中已明确标注此差异。
    """
    x_prev = np.asarray(x_prev, dtype=float)  # 先把状态转成浮点数组。
    P_prev = np.asarray(P_prev, dtype=float)  # 先把协方差转成浮点数组。

    require_shape(x_prev, (_STATE_DIM,), name="x_prev")  # 状态必须是固定长度向量。
    require_shape(P_prev, (_STATE_DIM, _STATE_DIM), name="P_prev")  # 协方差必须是方阵。
    dt = _coerce_numeric_scalar(dt, name="dt", min_value=0.0, inclusive=False)  # dt 必须严格大于 0。

    if predict_cfg is None:  # 没给配置时就当空配置。
        predict_cfg = {}  # 用空映射兜底。
    if not isinstance(predict_cfg, Mapping):  # 配置必须是映射。
        raise TypeError(f"predict_cfg must be a mapping, got {type(predict_cfg).__name__}")  # 直接拒绝。

    # 统一 imu_input 为字典视图，兼容 Event 对象（Event 没有 .get() 方法）。
    if hasattr(imu_input, "to_dict") and not isinstance(imu_input, Mapping):
        imu_input = imu_input.to_dict()  # Event 对象转成字典。

    imu_measurement = extract_imu_measurement(imu_input)  # 从输入事件中取出 IMU 数值。
    ax = _coerce_numeric_scalar(imu_measurement["ax"], name="imu_input.imu_payload.ax")  # x 轴加速度。
    ay = _coerce_numeric_scalar(imu_measurement["ay"], name="imu_input.imu_payload.ay")  # y 轴加速度。
    gz = _coerce_numeric_scalar(imu_measurement["gz"], name="imu_input.imu_payload.gz")  # z 轴角速度。

    # 检查 IMU 缺失掩码：缺失字段的偏置补偿应跳过，避免方向错误。
    # 缺失字段已补零（event_builder），若不减偏差会得到 0-bax（方向错误），
    # 应改为直接用 0 作为体坐标系加速度（匀速假设）。
    _imu_missing_mask = imu_input.get("imu_payload", {}).get("missing_mask")
    _ax_missing = False
    _ay_missing = False
    _gz_missing = False
    if isinstance(_imu_missing_mask, (list, tuple)) and len(_imu_missing_mask) >= 3:
        _ax_missing = bool(_imu_missing_mask[0])
        _ay_missing = bool(_imu_missing_mask[1])
        _gz_missing = bool(_imu_missing_mask[2])

    # 前提指导 §1.1 默认 8 维：动态索引解包。
    px = float(x_prev[_IDX_PX])  # x 位置
    py = float(x_prev[_IDX_PY])  # y 位置
    vx = float(x_prev[_IDX_VX])  # x 速度
    vy = float(x_prev[_IDX_VY])  # y 速度
    yaw = float(x_prev[_IDX_YAW])  # 航向
    bax = float(x_prev[_IDX_BAX])  # 加速度 x 偏置
    bay = float(x_prev[_IDX_BAY])  # 加速度 y 偏置
    bg = float(x_prev[_IDX_BG])  # 陀螺偏置

    yaw_rate = (0.0 if _gz_missing else (gz - bg))  # 缺失时角速度置零（匀速假设）。
    ax_body = (0.0 if _ax_missing else (ax - bax))  # 缺失时体加速度置零，避免 0-bax 方向错误。
    ay_body = (0.0 if _ay_missing else (ay - bay))  # 缺失时体加速度置零，避免 0-bay 方向错误。

    cos_yaw = math.cos(yaw)  # 当前朝向的余弦。
    sin_yaw = math.sin(yaw)  # 当前朝向的正弦。
    ax_world = (cos_yaw * ax_body) - (sin_yaw * ay_body)  # 把车体加速度转到世界坐标系。
    ay_world = (sin_yaw * ax_body) + (cos_yaw * ay_body)  # 把车体加速度转到世界坐标系。

    dt_sq = float(dt) * float(dt)  # dt 的平方。
    half_dt_sq = 0.5 * dt_sq  # 二分之一 dt 平方，给匀加速位移项使用。

    x_pred = x_prev.copy()  # 先复制一份，避免原地改坏输入。
    x_pred[_IDX_PX] = px + (vx * dt) + (half_dt_sq * ax_world)  # 位置 x 由速度和加速度推进。
    x_pred[_IDX_PY] = py + (vy * dt) + (half_dt_sq * ay_world)  # 位置 y 由速度和加速度推进。
    x_pred[_IDX_VX] = vx + (dt * ax_world)  # x 方向速度更新。
    x_pred[_IDX_VY] = vy + (dt * ay_world)  # y 方向速度更新。
    x_pred[_IDX_YAW] = wrap_angle_rad(yaw + (dt * yaw_rate))  # 航向角推进后再归一化。
    x_pred[_IDX_BAX] = bax  # 加速度偏置默认保持不变。
    x_pred[_IDX_BAY] = bay  # 加速度偏置默认保持不变。
    x_pred[_IDX_BG] = bg  # 陀螺偏置默认保持不变。

    da_world_dyaw_x = (-sin_yaw * ax_body) - (cos_yaw * ay_body)  # 世界系加速度对 yaw 的偏导 x 项。
    da_world_dyaw_y = (cos_yaw * ax_body) - (sin_yaw * ay_body)  # 世界系加速度对 yaw 的偏导 y 项。

    F = np.eye(_STATE_DIM, dtype=float)  # 状态转移雅可比先从单位阵开始。
    F[_IDX_PX, _IDX_VX] = dt  # px 对 vx 的偏导。
    F[_IDX_PY, _IDX_VY] = dt  # py 对 vy 的偏导。
    F[_IDX_PX, _IDX_YAW] = half_dt_sq * da_world_dyaw_x  # px 对 yaw 的偏导。
    F[_IDX_PY, _IDX_YAW] = half_dt_sq * da_world_dyaw_y  # py 对 yaw 的偏导。
    F[_IDX_VX, _IDX_YAW] = dt * da_world_dyaw_x  # vx 对 yaw 的偏导。
    F[_IDX_VY, _IDX_YAW] = dt * da_world_dyaw_y  # vy 对 yaw 的偏导。
    # bax 偏导：仅当 ax 不缺失时才非零。缺失时 ax_body 被强制置 0（匀速假设），
    # ∂ax_body/∂bax=0，因此所有对 bax 的偏导应为 0，与状态传播保持一致。
    if not _ax_missing:
        F[_IDX_PX, _IDX_BAX] = -half_dt_sq * cos_yaw  # px 对 bax 的偏导。
        F[_IDX_PY, _IDX_BAX] = -half_dt_sq * sin_yaw  # py 对 bax 的偏导。
        F[_IDX_VX, _IDX_BAX] = -dt * cos_yaw  # vx 对 bax 的偏导。
        F[_IDX_VY, _IDX_BAX] = -dt * sin_yaw  # vy 对 bax 的偏导。
    # bay 偏导：仅当 ay 不缺失时才非零。缺失时 ay_body 被强制置 0（匀速假设），
    # ∂ay_body/∂bay=0，因此所有对 bay 的偏导应为 0，与状态传播保持一致。
    if not _ay_missing:
        F[_IDX_PX, _IDX_BAY] = half_dt_sq * sin_yaw  # px 对 bay 的偏导。
        F[_IDX_PY, _IDX_BAY] = -half_dt_sq * cos_yaw  # py 对 bay 的偏导。
        F[_IDX_VX, _IDX_BAY] = dt * sin_yaw  # vx 对 bay 的偏导。
        F[_IDX_VY, _IDX_BAY] = -dt * cos_yaw  # vy 对 bay 的偏导。
    # bg 偏导：仅当 gz 不缺失时才非零。缺失时 yaw_rate 被强制置 0（匀速假设），
    # ∂yaw_rate/∂bg=0，因此 yaw 对 bg 的偏导应为 0，与状态传播保持一致。
    if not _gz_missing:
        F[_IDX_YAW, _IDX_BG] = -dt  # yaw 对 bg 的偏导。

    process_noise = predict_cfg.get("process_noise", {})  # 取过程噪声配置。
    if process_noise is None:  # 允许显式传空。
        process_noise = {}  # 用空映射兜底。
    elif not isinstance(process_noise, Mapping):  # 过程噪声必须是映射。
        raise TypeError(
            f"predict_cfg.process_noise must be a mapping, got {type(process_noise).__name__}"
        )  # 结构不对直接报错。

    q_diag = np.zeros(_STATE_DIM, dtype=float)  # 先准备对角过程噪声。
    for noise_key, state_indices in _PROCESS_NOISE_KEYS.items():  # 逐类把噪声映射到状态分量。
        noise_value = process_noise.get(noise_key, 0.0)  # 没配时默认 0。
        noise_value = _coerce_numeric_scalar(  # 先把单项过程噪声规整成有限非负标量。
            noise_value,  # 当前这一类噪声的原始配置值。
            name=f"predict_cfg.process_noise.{noise_key}",  # 报错时指明是哪一类噪声项。
            min_value=0.0,  # 噪声方差参数不能是负数。
        )  # 噪声必须非负且有限。
        for state_index in state_indices:  # 同一噪声可能作用到多个状态维度。
            # 噪声离散化：Q_i = σ² × dt。
            # 输入 noise_value 是噪声强度（PSD平方根，单位: state_unit/sqrt(s)），
            # 不是标准差。离散化为 σ²*dt 得到离散时间方差。
            # 这与 measurement_noise 的语义不同：measurement_noise 是标准差，
            # 直接平方为方差（无 dt 因子）。配置文件中已明确标注此差异。
            q_diag[state_index] = noise_value**2 * dt  # 按时间尺度传播成方差。

    Q = np.diag(q_diag)  # 把对角项展开成过程噪声矩阵。
    P_pred = (F @ P_prev @ F.T) + Q  # 标准协方差传播公式。
    P_pred = 0.5 * (P_pred + P_pred.T)  # 数值对称化，避免浮点误差。
    return x_pred, P_pred  # 返回预测后的状态和协方差。
