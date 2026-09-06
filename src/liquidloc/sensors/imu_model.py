"""IMU 事件提取与校验模块。

核心数据流：
    Event/dict → validate_imu_event → [校验通过] → extract_imu_measurement → {"ax", "ay", "gz"}

上游依赖：
    liquidloc.common.constants（DEFAULT_THRESHOLDS, MODALITY_IMU, PAYLOAD_KEYS）
    liquidloc.common.validation（coerce_finite_scalar）
    liquidloc.protocol.event_schema（Event, validate_event）

下游调用者：
    liquidloc.estimators.predict_step（extract_imu_measurement）
    liquidloc.sensors.__init__（对外导出）

关键常量：
    _IMU_PAYLOAD_KEY: IMU 载荷键名（"imu_payload"）

设计决策：
    - coerce_finite_scalar 只接受有限浮点数，拒绝 bool、非数值和 NaN/inf。
    - 兼容 numpy 标量（np.float64 等），因为 isinstance(np.float64(1.0), Number) 为 True。
    - _as_imu_event_dict 对 dict 输入返回原引用（不做防御性拷贝），调用者不应修改返回值。

【前提指导 §5.1 — 五方法同一 IMU 模型】
本模块是所有融合方法（ekf / robust_ekf / fgo / lstm_ekf / liquid_ekf）拿到 IMU 测量的唯一入口，
因此 ``ax``/``ay``/``gz`` 的语义、量纲、坐标系约定必须被全员共享，否则五方法做的是不同物理叙事。
其中 ekf / robust_ekf / fgo 由 estimator_factory 直接构造；lstm_ekf / liquid_ekf 复用 EKF 估计器外壳
并在其上叠加测量侧 NN 控制路径（不修改 IMU 比力/角速度叙事）。此入口对应 estimator_factory._SUPPORTED
与 core_pipeline._NEURAL_METHODS 的并集，未列出的 lnn_ekf / tf_ekf / transformer_ekf 在当前实现中
不存在（任何旧文档里出现这些名字均为遗留表述，以本节为准）。

约定：
    - 2D 平面 / ENU 水平面运动模型（与 sim_materializer._derive_imu_rows_from_normalized_gt
      的世界→机体旋转一致，与 predict_step.run_predict_step 的体→世界旋转一致）。
    - ``ax``/``ay`` 是机体系水平比力分量（单位 m/s²），**不含重力投影**：sim_materializer
      对 GT 位置做二阶差分得到世界系水平加速度，再由当前 yaw 旋转回机体系，过程中
      重力方向（垂直向下）垂直于水平面，不进入 ax/ay。
    - ``gz`` 是绕垂直轴的角速度（单位 rad/s），与 yaw 一致指向天向（ENU 模型下即 ``z`` 轴）。
    - ``missing_mask``（与 ax/ay/gz 一一对应的 0/1 列表）由 event_builder.build_imu_events
      写入 imu_payload，predict_step.run_predict_step 在拿到 ``imu_input`` 时会直接读取
      ``imu_payload.missing_mask`` 来跳过缺测分量的偏置补偿与过程噪声，调用者绕过
      ``extract_imu_measurement`` 直接取 ``missing_mask``，保持单入口约束。

这一约定使所有五方法在重力处理（§5.1：减去/保留在模型中全员同一）上达成一致——
**没有方法做重力补偿，也没有方法把重力塞进 ax/ay**。
"""

from __future__ import annotations  # 允许在类型标注里直接写还没定义到的类型名。

from liquidloc.common.constants import DEFAULT_THRESHOLDS, MODALITY_IMU, PAYLOAD_KEYS  # IMU 模态、标准 payload 键和阈值表。
from liquidloc.common.validation import coerce_finite_scalar  # 有限数值标量校验工具。
from typing import Any  # 用于 Event 转 dict 后的宽松字段类型标注。
from liquidloc.protocol.event_schema import Event, validate_event  # 统一事件对象类型和冻结事件合同校验。


_IMU_PAYLOAD_KEY = PAYLOAD_KEYS[MODALITY_IMU]  # IMU payload 的协议标准键名。
# 坐标系约定：ax/ay 为 2D 平面加速度（m/s²），gz 为绕垂直轴的角速度（rad/s）。
# 本项目采用 2D 平面约定（ENU 水平面），ax=东向加速度，ay=北向加速度，gz=天向角速度。
# 与 MATLAB Sensor Fusion Toolbox 的 imuSensor（默认 NED 坐标系）的对应关系：
#   Python ax ↔ MATLAB accel(2)（北向→东向需旋转），Python ay ↔ MATLAB accel(1)，
#   Python gz ↔ MATLAB gyro(3)（天向→地向需取反）。
# 上游数据合同（sensors.yaml）保证输入已转换到本约定。


def _as_imu_event_dict(imu_event: Event | dict[str, Any]) -> dict[str, Any]:
    """把 Event 对象或 dict 统一成普通字典。

    参数：
        imu_event: Event 对象或字典。

    返回：
        字典视图。对 Event 输入返回新字典副本；对 dict 输入返回原引用（不做防御性拷贝），
        调用者不应修改返回的字典。

    异常：
        TypeError: 输入既不是 Event 也不是 dict。
    """
    if isinstance(imu_event, Event):  # Event 对象先转成字典。
        return imu_event.to_dict()  # 对象型输入转成普通映射。
    if isinstance(imu_event, dict):  # 字典输入直接放行。
        return imu_event  # 直接复用原始字典。
    raise TypeError(f"imu_event must be an Event or dict, got {type(imu_event).__name__}")  # 其他类型都不合法。


def validate_imu_event(imu_event) -> None:
    """校验 IMU 事件是否满足协议要求。

    检查内容：
        - 模态必须为 ``"imu"``
        - ``imu_payload`` 不能为 None
        - ``imu_payload`` 必须包含 ``ax``、``ay``、``gz`` 三个键
        - 三个字段的值必须是有限数值（非 NaN/inf，非 bool）

    参数：
        imu_event: Event 对象或字典，必须包含 ``"modality"`` 和 ``"imu_payload"``。

    异常：
        ValueError: 模态不匹配、payload 缺失或字段值非法。
        KeyError: payload 缺少必需字段。
        TypeError: 字段值类型非法（如 bool）。
    """
    event_payload = _as_imu_event_dict(imu_event)  # 先把输入统一成字典。
    validate_event(event_payload)  # 先按冻结事件合同校验 t/dt/meta/跨模态 payload 等通用约束。
    modality = event_payload.get("modality")  # 读取事件模态。
    if modality != MODALITY_IMU:  # 不是 IMU 就不能按这个模块处理。
        raise ValueError(f"imu_event.modality must be {MODALITY_IMU!r}, got {modality!r}")  # 明确指出模态不匹配。

    imu_payload = event_payload.get(_IMU_PAYLOAD_KEY)  # 从标准 payload 键里取 IMU 测量内容（validate_event 已确保非 None 且字段齐全）。
    coerce_finite_scalar(imu_payload["ax"], name=f"{_IMU_PAYLOAD_KEY}.ax",
                         min_value=DEFAULT_THRESHOLDS["imu_accel_min"],
                         max_value=DEFAULT_THRESHOLDS["imu_accel_max"])  # 校验 x 轴加速度，含物理范围约束。
    coerce_finite_scalar(imu_payload["ay"], name=f"{_IMU_PAYLOAD_KEY}.ay",
                         min_value=DEFAULT_THRESHOLDS["imu_accel_min"],
                         max_value=DEFAULT_THRESHOLDS["imu_accel_max"])  # 校验 y 轴加速度，含物理范围约束。
    coerce_finite_scalar(imu_payload["gz"], name=f"{_IMU_PAYLOAD_KEY}.gz",
                         min_value=DEFAULT_THRESHOLDS["imu_gyro_min"],
                         max_value=DEFAULT_THRESHOLDS["imu_gyro_max"])  # 校验 z 轴角速度，含物理范围约束。


def extract_imu_measurement(imu_event) -> dict[str, float]:
    """从 IMU 事件里提取标准化测量字典。

    处理流程：
        1. 统一输入为字典
        2. 调用 validate_imu_event 做完整性校验（含 coerce_finite_scalar 范围校验）
        3. 将已校验的 ax/ay/gz 转为 Python float 组装输出

    参数：
        imu_event: Event 对象或字典，必须包含 ``"modality"`` 和 ``"imu_payload"``。

    返回：
        标准化测量字典 ``{"ax": float, "ay": float, "gz": float}``。

    异常：
        与 validate_imu_event 相同。
    """
    event_payload = _as_imu_event_dict(imu_event)  # 统一输入类型。
    validate_imu_event(event_payload)  # 先校验完整性（内部已对 ax/ay/gz 调用 coerce_finite_scalar）。

    imu_payload = event_payload[_IMU_PAYLOAD_KEY]  # 这里已经被校验过，所以可以直接取值。
    # validate_imu_event 内部已通过 coerce_finite_scalar 对 ax/ay/gz 做了有限性、
    # 类型（拒绝 bool）和物理范围校验。此处只需把已校验的值转为 Python float，
    # 避免重复调用 coerce_finite_scalar 带来的性能开销和口径漂移风险。
    # 若 validate_imu_event 的校验逻辑变更，必须同步审视此处的 float() 转换是否仍安全。
    imu_measurement = {  # 按固定键名重新组装。
        "ax": float(imu_payload["ax"]),  # x 轴加速度值。
        "ay": float(imu_payload["ay"]),  # y 轴加速度值。
        "gz": float(imu_payload["gz"]),  # z 轴角速度值。
    }  # 字典组装结束。
    return imu_measurement  # 返回标准化后的 IMU 测量字典。
