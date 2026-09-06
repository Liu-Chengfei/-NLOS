"""估计器工厂模块。

职责
----
把外部传入的估计器名字和配置，转换成真正可用的估计器实例。
作为上层 pipeline、脚本和测试的统一入口，避免调用方直接关心
EKF、鲁棒 EKF 和 FGO 的具体类名。

上游依赖
--------
- ``liquidloc.estimators.ekf_core.EKFCore``          — 标准 EKF 实现
- ``liquidloc.estimators.robust_ekf_core.RobustEKFCore`` — 鲁棒 EKF 实现
- ``liquidloc.estimators.fgo_core.FGOCore``           — 滑窗 FGO 实现
- ``liquidloc.estimators.state_definition.state_items`` — 状态顺序定义

下游调用者
----------
- ``liquidloc.pipelines.*``  — 各流水线通过 ``create_estimator`` 获取估计器实例
- ``scripts/`` 下的训练 / 评估脚本
- ``tests/factories/*``  — 工厂层单元测试

核心变量
--------
- ``_SUPPORTED``           — 当前支持的估计器名字集合
- ``_PROCESS_NOISE_KEYS``  — 过程噪声必须包含的键
- ``_COMMON_REQUIRED_CFG_KEYS`` — 所有估计器通用的必填配置键
"""

from __future__ import annotations  # 允许后面使用延迟求值的类型注解。

import math  # §5.1 全员同一护栏：跨方法数值等值断言用到 math.isclose（容差浮点比较）。

from collections.abc import Mapping  # 用来判断配置是不是映射。
from typing import Any  # 用于标注任意类型参数。

from liquidloc.estimators.ekf_core import EKFCore  # 标准 EKF 的具体实现。
from liquidloc.estimators.fgo_core import FGOCore  # 滑窗 FGO 的具体实现。
from liquidloc.estimators.robust_ekf_core import RobustEKFCore  # 鲁棒 EKF 的具体实现。
from liquidloc.estimators.state_definition import state_items  # 状态顺序定义，供初始状态校验使用。
from liquidloc.common.constants import (  # D9 单源常量：估计器名、模态名、VIO 噪声键方案与方案名，禁止本地重复字面量。
    ESTIMATOR_NAME_EKF,
    ESTIMATOR_NAME_ROBUST_EKF,
    ESTIMATOR_NAME_SGPR,
    MODALITY_VIO,
    VIO_NOISE_SCHEME_PER_AXIS,
    VIO_NOISE_SCHEME_SHARED,
    VIO_SHARED_NOISE_KEYS,
)
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_integer, is_real  # 集中判断 bool / np.bool_ / 整数 / 实数类型，以及有限浮点转换。
from liquidloc.interfaces.estimator_api import EstimatorAPI  # 估计器接口协议，作为 create_estimator 的返回类型注解。
from liquidloc.protocol.task_contract import get_sensor_roles, get_vio_update_contract  # 从协议层读取传感器角色和 VIO 更新合同，用于校验量测噪声配置。


_SUPPORTED = {ESTIMATOR_NAME_EKF, ESTIMATOR_NAME_ROBUST_EKF, ESTIMATOR_NAME_SGPR}  # §16.1 封闭对手集：仅承认标准EKF、Robust-EKF、纯SGPR三种估计器身份；纯SGPR当前尚未实现主体，但须在封闭集内占位防止游离方法名混入主表。
_ROBUST_EKF_REQUIRED_KEYS = ("robust_weight", "gate")  # 鲁棒 EKF 专属必填配置键，与 configs/models/robust_ekf.yaml 对齐。
# 过程噪声必须包含的键（前提指导 §1.1 主表 8 维 + §2.3 紧耦合扩维，全体同增至 10 维状态分组）。
# 与 predict_step._PROCESS_NOISE_KEYS 严格对齐：uwb_clock_bias / vio_scale 必须显式提供，
# 否则协方差预测时这两维不增长，更新步卡尔曼增益接近 0，等价于"未在线估"，违反 §2.3 钟差态前提。
_PROCESS_NOISE_KEYS = (
    "pos", "vel", "yaw", "accel_bias", "gyro_bias",
    "uwb_clock_bias", "vio_scale",
)
# §5.1 偏置随机游走强度全员同一护栏（五方法实现：ekf / robust_ekf / fgo / lstm_ekf / liquid_ekf）：
# 任何 stagger 的 process_noise 值会摧毁"同一预测模型"前提，静默让 NN+EKF/Robust/FGO
# 的偏置/钟差/尺度辨识变成"不同物理叙事"。这里钉死与 configs/models/{ekf,robust_ekf,fgo}.yaml
# 完全一致的值，_validate_process_noise 用它做数值等值断言；任何 YAML 漂移立即抛 ValueError，
# 避免依赖外部跨-YAML 比对工具或人工复查。
_CANONICAL_PROCESS_NOISE = {
    "pos": 0.05,
    "vel": 0.10,
    "yaw": 0.02,
    "accel_bias": 0.001,
    "gyro_bias": 0.001,
    "uwb_clock_bias": 0.001,
    "vio_scale": 0.001,
}
_COMMON_REQUIRED_CFG_KEYS = ("process_noise", "measurement_noise", "init_state", "init_cov")  # 所有估计器都要有的通用配置键，缺一不可。
_TASK_SENSOR_ROLES = get_sensor_roles()  # 从协议层获取当前任务的传感器角色映射，决定哪些模态是量测更新模态。
_VIO_UPDATE_CONTRACT = get_vio_update_contract()  # 从协议层获取 VIO 更新合同，包含量测项列表和噪声键方案定义。
_HIGH_RATE_PROPAGATION_ROLE = "high_rate_propagation"  # 协议定义的传播模态角色名，传播模态不走量测更新。
_RANGE_MEASUREMENT_ROLE = "absolute_range_constraint"  # 协议冻结的测距模态角色名，噪声为标量。
_RELATIVE_POSE_MEASUREMENT_ROLE = "relative_pose_constraint"  # 协议冻结的相对位姿模态角色名，噪声为映射。
_REQUIRED_MEASUREMENT_MODALITIES = tuple(  # 需要在量测噪声配置中出现的模态列表，从协议角色映射动态推导。
    modality
    for modality, role in _TASK_SENSOR_ROLES.items()
    if role != _HIGH_RATE_PROPAGATION_ROLE  # 排除仅用于传播的高速率模态（如 IMU），它们不走量测更新。通过协议层角色名判断而非硬编码模态名。
)
_SCALAR_NOISE_MODALITIES = tuple(  # 噪声为标量的模态列表（如 UWB 测距噪声），从协议角色映射动态推导。
    modality
    for modality, role in _TASK_SENSOR_ROLES.items()
    if role == _RANGE_MEASUREMENT_ROLE
)
_MAPPING_NOISE_MODALITIES = tuple(  # 噪声为映射的模态列表（如 VIO 位姿噪声），从协议角色映射动态推导。
    modality
    for modality, role in _TASK_SENSOR_ROLES.items()
    if role == _RELATIVE_POSE_MEASUREMENT_ROLE
)
_REQUIRED_VIO_MEASUREMENT_KEYS = tuple(_VIO_UPDATE_CONTRACT["measurement_items"])  # VIO 量测项键名元组，来自协议合同，用于 per_axis 噪声方案校验。
_VIO_SHARED_NOISE_KEYS = VIO_SHARED_NOISE_KEYS  # VIO 共享噪声方案的键名，引用 common 层单源常量（D9 漂移根因修复），与 estimators 层共用同一真相源。


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:  # 把输入规整为字典，并拒绝非映射类型。
    """把输入规整为字典，并拒绝非映射类型。

    参数
    ----------
    value : Any
        待检查的输入值。
    name : str
        参数名称，用于错误消息。

    返回
    -------
    dict
        从映射复制而来的字典。

    异常
    ------
    TypeError
        当 *value* 不是 ``collections.abc.Mapping`` 实例时抛出。
    """
    if not isinstance(value, Mapping):  # 非映射说明调用方传错结构。
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)  # 复制一份，避免外部继续修改原对象。


def _require_numeric_scalar(value: Any, *, name: str, allow_negative: bool = True, inclusive: bool = True) -> float:  # 把输入规整为浮点数，并拒绝布尔值、非数字和复数。
    """把输入规整为浮点数，并拒绝布尔值、非数字和复数。

    参数
    ----------
    value : Any
        待检查的输入值。
    name : str
        参数名称，用于错误消息。
    allow_negative : bool
        是否允许负数，默认允许。协方差和噪声值应传 ``False`` 以禁止负数。
    inclusive : bool
        当 *allow_negative* 为 ``False`` 时下界 0.0 的开闭性。默认 ``True``
        表示允许 0（闭区间）；VIO 噪声等要求严格正值的场景应传 ``False``。
        当 *allow_negative* 为 ``True`` 时此参数无效（无下界）。

    返回
    -------
    float
        转换后的浮点数。

    异常
    ------
    TypeError
        当 *value* 为布尔值、非数字或复数时抛出。
    ValueError
        当 *value* 为非有限值，或 *allow_negative* 为 ``False`` 且 *value*
        不满足 *inclusive* 约束的下界时抛出。

    注意
    -----
    Python 原生 ``complex`` 和 ``numbers.Complex`` 的子类都会被拒绝，
    因为噪声、状态和协方差分量不能是复数。
    """
    if is_bool_like(value):  # Python 原生布尔值和 numpy.bool_ 都不接受。
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if not is_real(value):  # 必须是实数值类型（排除 bool 和 complex）。
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    # 委托 coerce_finite_scalar：float 转换、有限性、下界（当 allow_negative=False）及开闭性一次性完成。
    min_value = 0.0 if not allow_negative else None
    return coerce_finite_scalar(value, name=name, min_value=min_value, inclusive=inclusive)


def _resolve_vio_noise_key_scheme(vio_noise: Mapping[str, Any], *, name: str) -> str:
    """要求 VIO 噪声映射必须使用且仅使用一种明确的键方案。

    VIO 噪声配置支持两种互斥的键方案：
    - "shared" 方案：使用 ``VIO_SHARED_NOISE_KEYS``（common 层单源常量，当前为 ("pos", "yaw")）
      两个键，分别表示位置和航向的共享噪声。
    - "per_axis" 方案：使用协议合同中定义的完整量测项键列表，每个量测维度独立设噪声。

    两种方案不能混用，也不能包含多余键，否则会引发歧义。

    参数
    ----------
    vio_noise : Mapping[str, Any]
        VIO 噪声配置映射。
    name : str
        参数名称，用于错误消息。

    返回
    -------
    str
        键方案名称：``VIO_NOISE_SCHEME_SHARED`` 或 ``VIO_NOISE_SCHEME_PER_AXIS``。

    异常
    ------
    ValueError
        当两种方案同时存在、某方案下包含多余键、或两种方案的键都不完整时抛出
        （键结构不构成合法方案属值合同违例，非键访问失败）。
    """
    vio_noise_keys = set(vio_noise.keys())  # 取出 VIO 噪声配置的所有键。
    shared_keys = set(_VIO_SHARED_NOISE_KEYS)  # shared 方案所需的键集合：{"pos", "yaw"}。
    per_axis_keys = set(_REQUIRED_VIO_MEASUREMENT_KEYS)  # per_axis 方案所需的键集合，来自协议合同。
    has_shared_scheme = shared_keys.issubset(vio_noise_keys)  # 检查是否包含 shared 方案的全部键。
    has_per_axis_scheme = per_axis_keys.issubset(vio_noise_keys)  # 检查是否包含 per_axis 方案的全部键。
    if has_shared_scheme and has_per_axis_scheme:  # 两种方案同时存在，产生歧义，不允许。
        raise ValueError(
            f"{name} must use exactly one VIO noise key scheme: "
            f"{list(_VIO_SHARED_NOISE_KEYS)!r} or {list(_REQUIRED_VIO_MEASUREMENT_KEYS)!r}; "
            f"received keys={sorted(vio_noise_keys)}"
        )
    if has_shared_scheme:  # 使用 shared 方案。
        unexpected_keys = sorted(vio_noise_keys - shared_keys)  # 检查是否有多余键。
        if unexpected_keys:  # shared 方案下不允许包含额外键，避免隐式混用。
            raise ValueError(
                f"{name} using {list(_VIO_SHARED_NOISE_KEYS)!r} must not include extra keys: {unexpected_keys}; "
                f"received keys={sorted(vio_noise_keys)}"
            )
        return VIO_NOISE_SCHEME_SHARED  # 返回 shared 方案标识。
    if has_per_axis_scheme:  # 使用 per_axis 方案。
        unexpected_keys = sorted(vio_noise_keys - per_axis_keys)  # 检查是否有多余键。
        if unexpected_keys:  # per_axis 方案下不允许包含额外键，避免隐式混用。
            raise ValueError(
                f"{name} using {list(_REQUIRED_VIO_MEASUREMENT_KEYS)!r} must not include extra keys: {unexpected_keys}; "
                f"received keys={sorted(vio_noise_keys)}"
            )
        return VIO_NOISE_SCHEME_PER_AXIS  # 返回 per_axis 方案标识。
    missing_shared_keys = [key for key in _VIO_SHARED_NOISE_KEYS if key not in vio_noise]  # shared 方案缺失的键。
    missing_per_axis_keys = [key for key in _REQUIRED_VIO_MEASUREMENT_KEYS if key not in vio_noise]  # per_axis 方案缺失的键。
    raise ValueError(  # 两种方案都不完整，无法确定使用哪种，报错提示缺失键。值合同违例用 ValueError，非键访问失败。
        f"{name} must provide exactly one complete VIO noise scheme; "
        f"missing shared keys={missing_shared_keys}, missing per-axis keys={missing_per_axis_keys}; "
        f"received keys={sorted(vio_noise_keys)}"
    )


def _validate_process_noise(process_noise_cfg: Any) -> None:  # 校验过程噪声配置是否包含所有必须字段。
    """校验过程噪声配置的字段完整性、值类型，以及五方法全员同一的协议基线等值。

    参数
    ----------
    process_noise_cfg : Any
        过程噪声配置，必须是映射且包含全部过程噪声键。

    异常
    ------
    TypeError
        当 *process_noise_cfg* 不是映射，或某个噪声值是布尔型、非数字或复数时抛出。
    ValueError
        当缺少必要的过程噪声键、某个噪声值为非有限值/负数，
        或某个键值与 ``_CANONICAL_PROCESS_NOISE`` 不等值时抛出。
        噪声值必须是非负实数（PSD 平方根，离散化为 σ²×dt），与协方差分量同口径。

    备注
    -----
    §5.1「偏置随机游走强度全员同一」护栏：除了存在性/非负校验外，本函数还把
    *process_noise_cfg* 的全部七键值与 ``_CANONICAL_PROCESS_NOISE`` 做严格等值断言
    （用 ``math.isclose`` 容差 1e-12，避免浮点字面量尾部差异误报）。
    一旦任何方法的 YAML 漂移出协议基线值（例如 lstm_ekf.yaml 单方把 accel_bias
    改成 0.002），本函数即刻抛 ValueError，把"数值漂移"从静默退化转为显式失败，
    保护五方法在 IMU 模态上做"同一物理叙事的不同推理形式"的前提。
    """
    process_noise = _require_mapping(process_noise_cfg, name="estimator_cfg.process_noise")  # 先确保能按键访问。
    missing_keys = [noise_key for noise_key in _PROCESS_NOISE_KEYS if noise_key not in process_noise]  # 找出缺失的过程噪声键。
    if missing_keys:  # 只要缺一个就不允许继续。值合同违例用 ValueError，与 model_factory.py 口径一致。
        raise ValueError(
            f"estimator_cfg.process_noise is missing required keys: {missing_keys}; "
            f"expected keys: {list(_PROCESS_NOISE_KEYS)}"
        )
    for noise_key in _PROCESS_NOISE_KEYS:  # 逐个检查每个过程噪声分量。
        normalized_value = _require_numeric_scalar(
            process_noise[noise_key],
            name=f"estimator_cfg.process_noise.{noise_key}",
            allow_negative=False,
        )  # 每个值都必须是非负数。
        # §5.1 全员同一护栏：与协议基线值做严格等值断言。
        canonical_value = _CANONICAL_PROCESS_NOISE[noise_key]
        if not math.isclose(normalized_value, canonical_value, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"estimator_cfg.process_noise.{noise_key}={normalized_value!r} violates §5.1 "
                f"全员同一协议基线（canonical={canonical_value!r}）；"
                f"五方法（ekf / robust_ekf / fgo / lstm_ekf / liquid_ekf）的 process_noise "
                f"必须严格一致，详见 _CANONICAL_PROCESS_NOISE 注释。"
            )


def _validate_measurement_noise(measurement_noise_cfg: Any) -> None:  # 校验量测噪声配置是否包含所有必需模态。
    """校验量测噪声配置是否包含所有必需模态。

    根据协议角色映射动态判断每个模态的噪声类型：
    - 测距模态（uwb，角色 absolute_range_constraint）：噪声为单个非负标量
    - 相对位姿模态（vio，角色 relative_pose_constraint）：噪声为映射，支持 shared 或 per_axis 两种键方案

    参数
    ----------
    measurement_noise_cfg : Any
        量测噪声配置，必须是映射且包含所有量测更新模态。

    异常
    ------
    TypeError
        当 *measurement_noise_cfg* 不是映射，或噪声值类型不正确时抛出。
    ValueError
        当缺少必要的模态键、或 VIO 噪声键方案冲突/含多余键/两种方案都不完整时抛出
        （由 ``_resolve_vio_noise_key_scheme`` 抛出，值合同违例）。
    """
    measurement_noise = _require_mapping(measurement_noise_cfg, name="estimator_cfg.measurement_noise")  # 先确保是映射。
    missing_modalities = [
        modality for modality in _REQUIRED_MEASUREMENT_MODALITIES if modality not in measurement_noise
    ]  # 找出缺失的模态噪声。
    if missing_modalities:  # 只要缺一个模态就报错。
        raise ValueError(
            "estimator_cfg.measurement_noise is missing required modalities: "
            f"{missing_modalities}"
        )
    # 校验标量噪声模态（如 UWB 测距噪声）。
    for modality in _SCALAR_NOISE_MODALITIES:
        if modality in measurement_noise:
            _require_numeric_scalar(
                measurement_noise[modality],
                name=f"estimator_cfg.measurement_noise.{modality}",
                allow_negative=False,
            )
    # 校验映射噪声模态（如 VIO 位姿噪声）。
    for modality in _MAPPING_NOISE_MODALITIES:
        if modality in measurement_noise:
            modality_noise = _require_mapping(
                measurement_noise[modality],
                name=f"estimator_cfg.measurement_noise.{modality}",
            )
            # VIO 模态支持 shared/per_axis 两种键方案；其他映射噪声模态暂按 VIO 合同校验。
            if modality == MODALITY_VIO:  # D9：模态比较必须引用 MODALITY_VIO 单源常量，禁止硬编码字符串漂移。
                scheme = _resolve_vio_noise_key_scheme(
                    modality_noise,
                    name=f"estimator_cfg.measurement_noise.{modality}",
                )
                if scheme == VIO_NOISE_SCHEME_SHARED:
                    _require_numeric_scalar(modality_noise["pos"], name=f'estimator_cfg.measurement_noise.{modality}["pos"]', allow_negative=False, inclusive=False)
                    _require_numeric_scalar(modality_noise["yaw"], name=f'estimator_cfg.measurement_noise.{modality}["yaw"]', allow_negative=False, inclusive=False)
                else:  # per_axis 方案
                    for key in _REQUIRED_VIO_MEASUREMENT_KEYS:
                        _require_numeric_scalar(modality_noise[key], name=f'estimator_cfg.measurement_noise.{modality}["{key}"]', allow_negative=False, inclusive=False)
            else:
                # 非VIO映射噪声模态：逐键校验为非负标量。
                for key, value in modality_noise.items():
                    _require_numeric_scalar(value, name=f'estimator_cfg.measurement_noise.{modality}["{key}"]', allow_negative=False)


def _validate_init_state(init_state_cfg: Any) -> None:  # 校验初始状态是否严格覆盖状态向量每个分量且无多余键。
    """校验初始状态是否严格覆盖状态向量每个分量且无多余键。

    初始状态必须严格覆盖 :data:`state_items` 中的全部状态分量键
    （px/py/vx/vy/yaw/bax/bay/bg，冻结顺序），且不得包含多余键，
    防止拼写错误或未协议化的状态名被静默吞掉。

    参数
    ----------
    init_state_cfg : Any
        初始状态配置，必须是映射且键集合与 state_items 完全一致。

    异常
    ------
    TypeError
        当 *init_state_cfg* 不是映射，或某个状态值是布尔型、非数字或复数时抛出。
    ValueError
        当 *init_state_cfg* 缺少必要的状态键、包含多余状态键，
        或某个状态值为非有限值时抛出（值合同违例，与
        :func:`_validate_process_noise`、:func:`_validate_measurement_noise` 同口径）。
    """
    init_state = _require_mapping(init_state_cfg, name="estimator_cfg.init_state")  # 先确保是映射。
    missing_keys = [state_key for state_key in state_items if state_key not in init_state]  # 找出漏掉的状态键。
    if missing_keys:  # 只要缺一个状态键就不行。值合同违例用 ValueError，与 _validate_process_noise 同口径。
        raise ValueError(
            f"estimator_cfg.init_state is missing required keys: {missing_keys}; "
            f"expected keys: {list(state_items)}"
        )
    unexpected_keys = [key for key in init_state if key not in state_items]  # 检查是否有多余键，与 _resolve_vio_noise_key_scheme 同口径。
    if unexpected_keys:  # 多余键会让配置层与协议层静默漂移，必须显式拒绝。
        raise ValueError(
            f"estimator_cfg.init_state contains unexpected keys: {unexpected_keys}; "
            f"allowed keys: {list(state_items)}"
        )
    for state_key in state_items:  # 逐个检查状态分量。
        _require_numeric_scalar(init_state[state_key], name=f"estimator_cfg.init_state.{state_key}")  # 每个状态值都必须是数字。


def _validate_init_cov(init_cov_cfg: Any) -> None:  # 校验初始协方差是否能和状态维度一一对应。
    """校验初始协方差是否能和状态维度一一对应。

    参数
    ----------
    init_cov_cfg : Any
        初始协方差配置，必须是序列且长度与状态维度一致。

    异常
    ------
    TypeError
        当 *init_cov_cfg* 不是序列（如字符串）时抛出。
    ValueError
        当 *init_cov_cfg* 长度与状态维度不匹配时抛出。
    """
    if not isinstance(init_cov_cfg, (list, tuple)):  # 仅接受 list/tuple，统一拒绝 str/bytes/bytearray/numpy.str_/ndarray/Mapping 等其他类型，错误消息携带实际类型以便定位。
        raise TypeError(f"estimator_cfg.init_cov must be a sequence, got {type(init_cov_cfg).__name__}")
    if len(init_cov_cfg) != len(state_items):  # 长度必须和状态维度对齐。
        raise ValueError(
            "estimator_cfg.init_cov must align with state dimension "
            f"{len(state_items)}, got {len(init_cov_cfg)}"
        )
    for index, value in enumerate(init_cov_cfg):  # 逐个检查每个初始协方差分量。
        _require_numeric_scalar(value, name=f"estimator_cfg.init_cov[{index}]", allow_negative=False)  # 协方差分量必须是非负数。


def _validate_required_keys(cfg: dict[str, Any], *, estimator_name: str, required_keys: tuple[str, ...]) -> None:  # 检查某个配置字典是否包含指定的必填键。
    """检查某个配置字典是否包含指定的必填键。

    参数
    ----------
    cfg : dict[str, Any]
        待检查的配置字典。
    estimator_name : str
        估计器名字，用于错误消息。
    required_keys : tuple[str, ...]
        必须存在的键名元组。

    异常
    ------
    ValueError
        当 *cfg* 缺少任何必填键时抛出（值合同违例，与 model_factory.py 及本文件 _validate_process_noise 口径一致）。
    """
    missing_keys = [key for key in required_keys if key not in cfg]  # 收集缺失的键。
    if missing_keys:  # 缺键时直接报错。值合同违例用 ValueError，与 _validate_process_noise Round 1 修复一致。
        raise ValueError(
            f"estimator_cfg for {estimator_name} is missing required keys: {missing_keys}; "
            f"expected keys: {list(required_keys)}"
        )


def _assert_fixed_robust_kernel(robust_weight_cfg: Mapping[str, Any], estimator_name: str) -> None:  # §16.2 强制 Robust 核为固定 Huber，禁止可学习核。
    """验证 Robust 核类型为固定 Huber，禁止可学习/自适应核。

    §16.2「Robust 可学习核」退出名次语言：核类型必须为固定 Huber，
    禁止 learnable/adaptive 核。

    参数
    ----------
    robust_weight_cfg : Mapping[str, Any]
        鲁棒权重配置映射。
    estimator_name : str
        估计器名字，用于错误消息。

    异常
    ------
    ValueError
        当核类型不是固定 Huber 时抛出。
    """
    kernel_type = robust_weight_cfg.get("type")
    if kernel_type is not None and str(kernel_type).lower() != "huber":
        raise ValueError(
            f"§16.2 Robust 可学习核退出名次语言：{estimator_name!r} 的 "
            f"robust_weight.type={kernel_type!r} 不是固定 Huber；"
            f"可学习/自适应核违反 §16.2「标准EKF 自适应 R；Robust 可学习核」。"
        )


def _assert_fixed_measurement_noise(cfg: dict[str, Any], estimator_name: str) -> None:  # §16.2 强制标准EKF R为固定值，禁止自适应R。
    """验证标准 EKF 的量测噪声 R 为固定值，禁止自适应/可学习 R。

    §16.2「标准EKF 自适应 R」退出名次语言：R 必须是固定值，
    禁止 learnable/adaptive R。

    参数
    ----------
    cfg : dict[str, Any]
        完整的估计器配置字典。
    estimator_name : str
        估计器名字，用于错误消息。

    异常
    ------
    ValueError
        当 R 配置包含自适应/可学习字段时抛出。
    """
    measurement_noise = cfg.get("measurement_noise")
    if isinstance(measurement_noise, Mapping):
        for key, value in measurement_noise.items():
            if isinstance(value, Mapping) and ("learnable" in value or "adaptive" in value or "trainable" in value):
                raise ValueError(
                    f"§16.2 标准EKF 自适应 R 退出名次语言：{estimator_name!r} 的 "
                    f"measurement_noise.{key} 包含 learnable/adaptive/trainable 字段；"
                    f"自适应 R 违反 §16.2「标准EKF 自适应 R」。"
                )


def _validate_estimator_cfg(estimator_name: str, cfg: dict[str, Any]) -> None:  # 按不同估计器类型校验其专属配置。
    """按不同估计器类型校验其专属配置。

    参数
    ----------
    estimator_name : str
        估计器名字，决定要执行哪些额外校验。
    cfg : dict[str, Any]
        完整的估计器配置字典。

    异常
    ------
    ValueError
        当缺少通用必填配置键（由 ``_validate_required_keys`` 抛出）、缺少过程噪声/量测噪声模态键、
        ``init_state`` 缺少或多余状态键（由 ``_validate_init_state`` 抛出）、
        或 ``fgo`` 的 ``window_size`` 不是正整数时抛出（值合同违例）。
    TypeError
        当配置值类型不正确时抛出。
    """
    _validate_required_keys(cfg, estimator_name=estimator_name, required_keys=_COMMON_REQUIRED_CFG_KEYS)  # 先校验通用必填项。
    _validate_process_noise(cfg["process_noise"])  # 校验过程噪声配置的完整性和类型。
    _validate_measurement_noise(cfg["measurement_noise"])  # 校验量测噪声配置的完整性和类型。
    _validate_init_state(cfg["init_state"])  # 校验初始状态配置是否覆盖所有状态分量。
    _validate_init_cov(cfg["init_cov"])  # 校验初始协方差配置的维度和类型。

    if estimator_name == ESTIMATOR_NAME_EKF:  # 标准 EKF 量测噪声 R 必须为固定值。
        _assert_fixed_measurement_noise(cfg, estimator_name)  # §16.2 禁止自适应 R。
    elif estimator_name == ESTIMATOR_NAME_ROBUST_EKF:  # 鲁棒 EKF 需要额外的鲁棒和门控配置。D9：引用单源常量，禁止硬编码字符串。
        _validate_required_keys(cfg, estimator_name=estimator_name, required_keys=_ROBUST_EKF_REQUIRED_KEYS)  # 检查额外必填项。
        _require_mapping(cfg["robust_weight"], name="estimator_cfg.robust_weight")  # 鲁棒权重必须是映射，用于 M 估计的权重函数配置。
        _require_mapping(cfg["gate"], name="estimator_cfg.gate")  # gate 也必须是映射，用于新息门控阈值配置。
        # §16.2「Robust 可学习核」退出名次语言：核类型必须为固定 Huber，禁止 learnable/adaptive 核。
        _assert_fixed_robust_kernel(cfg["robust_weight"], estimator_name)


def create_estimator(estimator_name: str, estimator_cfg: Mapping[str, Any] | None) -> EstimatorAPI:  # 根据估计器名字和配置创建对应实例。
    """根据估计器名字和配置创建对应实例。

    参数
    ----------
    estimator_name : str
        估计器名字，必须在 ``_SUPPORTED`` 集合内
        （``ESTIMATOR_NAME_EKF`` / ``ESTIMATOR_NAME_ROBUST_EKF``）。
    estimator_cfg : Mapping[str, Any] | None
        估计器配置映射，若为 ``None`` 则使用空字典。

    返回
    -------
    EstimatorAPI
        对应估计器的实例（``EKFCore`` / ``RobustEKFCore``，均为 ``EstimatorAPI`` 子类）。

    异常
    ------
    ValueError
        当 *estimator_name* 不在支持列表中时抛出（值合同违例，与 ``model_factory.create_model`` 口径一致）。
    TypeError
        当 *estimator_cfg* 不是映射，或配置值类型不正确时抛出。
    """
    if estimator_name not in _SUPPORTED:  # 只允许支持列表里的估计器。值合同违例用 ValueError，与 model_factory.create_model 一致。
        available = ", ".join(sorted(_SUPPORTED))
        raise ValueError(f"Unknown estimator: {estimator_name}. Available: {available}")

    if estimator_cfg is None:  # 空配置时先用空字典兜底。显式 None 检查防止 falsy 非 None 值误回退，与 model_factory.create_model 口径一致。
        cfg = {}
    elif not isinstance(estimator_cfg, Mapping):  # 非映射说明调用方传错结构。
        raise TypeError(f"estimator_cfg for {estimator_name} must be a mapping")
    else:
        cfg = dict(estimator_cfg)  # 复制一份，避免污染调用方对象。
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "estimator_name": estimator_name,
        "cfg_keys": list(cfg.keys()) if isinstance(cfg, dict) else None,
        "has_anchor_layout": "anchor_layout" in (cfg or {}) if isinstance(cfg, dict) else False,
        "has_gate": "gate" in (cfg or {}) if isinstance(cfg, dict) else False,
    }, "create_estimator 入口参数")
    cfg.setdefault("name", estimator_name)  # 没显式给名时，就用估计器名，方便下游日志和调试。
    # §16.1 纯 SGPR 占位：SGPR 尚未实现主体，先于校验前拦截，避免因缺配置键报 misleading 错误。
    if estimator_name == ESTIMATOR_NAME_SGPR:
        raise NotImplementedError(
            f"§16.1 封闭对手集：纯 SGPR 估计器尚未实现，ESTIMATOR_NAME_SGPR={estimator_name!r} 当前无真实 SGPR 估计器本体。"
            f"spec §16.1 要求纯 SGPR 作为七种身份之一占位，但须先实现 SGPR 估计器本体后才能在主表中使用。"
        )
    _validate_estimator_cfg(estimator_name, cfg)  # 先校验再实例化，确保配置完整且类型正确。
    if estimator_name == ESTIMATOR_NAME_EKF:  # 这里返回标准 EKF 实现。D9：引用单源常量，禁止硬编码字符串。
        return EKFCore(cfg)
    elif estimator_name == ESTIMATOR_NAME_ROBUST_EKF:  # 这里返回鲁棒 EKF 实现，支持 M 估计和新息门控。D9：引用单源常量。
        return RobustEKFCore(cfg)
    raise ValueError(f"Unreachable: estimator {estimator_name!r} not dispatched")  # 防御性兜底：逻辑上不可达，防止 _SUPPORTED 集合被绕过后静默返回 None。
