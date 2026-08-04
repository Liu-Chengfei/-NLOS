"""鲁棒 EKF 基线。

这个模块在标准 EKF 的基础上增加质量门控、NIS 门控和 Huber 风格的协方差膨胀。
它主要服务于对异常量测更敏感的公开 benchmark 路径。

模块内容
--------
- :func:`_normalize_vio_covariance` —— 视觉噪声规整成 3×3 协方差矩阵
- :func:`_quality_below_floor` —— 判断质量值是否低于门槛
- :class:`RobustEKFCore` —— 鲁棒 EKF 核心估计器
"""

from __future__ import annotations  # 允许后面的类型注解延迟求值，避免前向引用报错。

import copy  # 深拷贝，防止嵌套配置被外部修改污染内部状态。
import math  # 用来做平方根、有限性检查和 Huber 规则。
from collections.abc import Mapping  # 用来识别配置映射。
from typing import Any  # 用来给报告和配置字典保留灵活类型。

import numpy as np  # 用来做矩阵和向量运算。

from liquidloc.estimators.ekf_core import EKFCore  # 复用基础 EKF 状态管理。
from liquidloc.estimators.shared import build_controlled_measurement_cov, control_to_dict  # 共享的协方差控制和报告转换函数。
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS  # 桥接层业务阈值常量
from liquidloc.common.constants import DEFAULT_THRESHOLDS  # 全局默认阈值表（质量上下界单源真相）。
from liquidloc.common.constants import QUALITY_FLOOR_EPSILON  # 质量门槛比较浮点容差
from liquidloc.common.constants import VIO_MEASUREMENT_ITEMS  # VIO 测量项常量（单源真相）。
from liquidloc.common.constants import VIO_REF_POSE_STALE_SECONDS  # VIO 参考位姿过时阈值。
from liquidloc.common.types import MeasurementControl  # 单个测量通道的控制结果。
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like  # 集中判断 bool / np.bool_ / 实数类型，以及有限浮点转换。
from liquidloc.common.validation import quality_below_floor  # 判断质量是否低于门槛（共享入口）。
from liquidloc.estimators.uwb_update_step import build_uwb_jacobian, predict_range, run_uwb_update  # 复用 UWB 几何、预测和更新步骤。
from liquidloc.estimators.vision_update_step import (  # 复用 VIO 的量测、残差和更新。
    _ensure_positive_definite_vio_innovation_covariance,  # 统一校验 VIO 创新协方差必须正定。
    apply_vision_update,  # 把视觉量测真正写回状态和协方差。
    build_vio_measurement,  # 从负载里提取标准化 VIO 量测向量。
    compute_vio_residual,  # 计算 VIO 预测量测、残差和雅可比。
)  # 这些函数一起构成 VIO 更新链路。
from liquidloc.common.covariance_utils import build_effective_cov  # 复用协方差缩放适配器。


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


def _normalize_vio_covariance(R_vio: Any) -> np.ndarray:  # 把视觉噪声规整成 3x3 协方差矩阵。
    """把视觉噪声规整成 3×3 协方差矩阵。

    参数
    ----------
    R_vio : Any
        视觉噪声配置，支持以下格式：

        - ``None`` —— 不允许，抛出 TypeError
        - ``bool`` / ``np.bool_`` —— 不允许，抛出 TypeError
        - ``complex`` —— 不允许，抛出 TypeError
        - Mapping —— 支持两种键组合：

          - ``{"pos": float, "yaw": float}`` —— 位置噪声共享给 dx/dy
          - ``{"dx": float, "dy": float, "dyaw": float}`` —— 三个独立分量

        - 标量 —— 扩成 3×3 对角阵
        - shape (3,) 向量 —— 转成对角协方差矩阵
        - shape (3, 3) 矩阵 —— 直接使用

    返回
    -------
    np.ndarray
        shape (3, 3) 的协方差矩阵。

    异常
    ------
    TypeError
        R_vio 为 None、布尔类或 complex 时抛出。
    KeyError
        Mapping 键组合不满足 pos/yaw 或 dx/dy/dyaw 时抛出。
    ValueError
        噪声值非有限、负数，或矩阵形状不合法时抛出。

    注意
    -----
    键优先级：``(pos, yaw)`` 优先于 ``(dx, dy, dyaw)``。
    """
    if R_vio is None:  # 空值说明配置缺失，不能继续推导协方差。
        raise TypeError("R_vio must not be None")  # VIO 噪声必须显式给出。
    if is_bool_like(R_vio):  # 布尔值常常是误写成开关，不是噪声。
        raise TypeError("R_vio must not be boolean-like")  # 这里拒绝把开关当数值。
    if isinstance(R_vio, complex):  # complex 不是实数噪声。
        raise TypeError("R_vio must be real numeric, got complex")  # 复数不能当实数噪声。
    if isinstance(R_vio, Mapping):  # 字典配置支持不同命名风格。
        has_pos_yaw = all(key in R_vio for key in ("pos", "yaw"))
        has_dx_dy_dyaw = all(key in R_vio for key in VIO_MEASUREMENT_ITEMS)
        if has_pos_yaw and has_dx_dy_dyaw:  # 两种键同时存在，语义歧义，必须明确指定一种。
            raise ValueError("R_vio mapping must not mix pos/yaw and dx/dy/dyaw keys; choose one naming convention")  # 不允许混用。
        if has_pos_yaw:  # pos/yaw 代表位置与航向噪声。
            if is_bool_like(R_vio["pos"]) or is_bool_like(R_vio["yaw"]):  # 布尔值不是合法噪声。
                raise TypeError("R_vio mapping values must not be boolean-like")  # 拒绝布尔伪值。
            pos = coerce_finite_scalar(R_vio["pos"], name="R_vio pos", min_value=0.0, inclusive=False)  # 位置噪声量级，必须有限且为正。
            yaw = coerce_finite_scalar(R_vio["yaw"], name="R_vio yaw", min_value=0.0, inclusive=False)  # 航向噪声量级，必须有限且为正。
            diag_values = [  # 对角线的三个分量按 x/y/yaw 顺序排列。
                pos,  # x 方向位置噪声。
                pos,  # y 方向位置噪声，和 x 共享同一个量级。
                yaw,  # 航向噪声。
            ]  # 对角线三个分量已经按 x/y/yaw 顺序排好。
        elif has_dx_dy_dyaw:  # dx/dy/dyaw 代表三个分量的独立噪声。
            if is_bool_like(R_vio["dx"]) or is_bool_like(R_vio["dy"]) or is_bool_like(R_vio["dyaw"]):  # 布尔值不是合法噪声。
                raise TypeError("R_vio mapping values must not be boolean-like")  # 拒绝布尔伪值。
            dx = coerce_finite_scalar(R_vio["dx"], name="R_vio dx", min_value=0.0, inclusive=False)  # x 增量噪声，必须有限且为正。
            dy = coerce_finite_scalar(R_vio["dy"], name="R_vio dy", min_value=0.0, inclusive=False)  # y 增量噪声，必须有限且为正。
            dyaw = coerce_finite_scalar(R_vio["dyaw"], name="R_vio dyaw", min_value=0.0, inclusive=False)  # 航向增量噪声，必须有限且为正。
            diag_values = [  # 对角线的三个分量按 dx/dy/dyaw 顺序排列。
                dx,  # x 增量噪声。
                dy,  # y 增量噪声。
                dyaw,  # 航向增量噪声。
            ]  # 对角线三个分量已经按 dx/dy/dyaw 顺序排好。
        else:  # 其他键组合都不接受。
            raise KeyError("R_vio mapping must provide either pos/yaw or dx/dy/dyaw noise entries")  # 必须提供完整噪声项。
        return np.diag(diag_values)  # 把对角线噪声转成矩阵。

    cov = np.asarray(R_vio, dtype=float)  # 把输入统一转成数值数组。
    if cov.ndim == 0:  # 标量噪声直接扩成对角阵。
        scalar = coerce_finite_scalar(cov, name="R_vio scalar", min_value=0.0, inclusive=False)  # 标量必须有限且为正。
        return np.eye(3, dtype=float) * scalar  # 标量噪声扩成 3x3 对角阵。
    if cov.shape == (3,):  # 允许直接传 3 维对角线。
        if not np.all(np.isfinite(cov)):  # 每个对角项都必须有限。
            raise ValueError("R_vio diagonal entries must be finite")  # 不能有 nan 或 inf。
        if np.any(cov <= 0.0):  # 对角线不能出现零或负值。
            raise ValueError("R_vio diagonal entries must be positive")  # 协方差方差项必须严格为正。
        return np.diag(cov)  # 向量直接转成对角协方差矩阵。
    if cov.shape != (3, 3):  # 其他形状都不接受。
        raise ValueError(f"R_vio must be shape (3, 3), got {cov.shape}")  # 非 3x3 的矩阵不接受。
    if not np.all(np.isfinite(cov)):  # 任何非有限值都会破坏更新。
        raise ValueError("R_vio matrix entries must be finite")  # 矩阵中的每个元素都要有效。
    if np.any(np.diag(cov) <= 0.0):  # 对角线必须严格为正。
        raise ValueError("R_vio diagonal entries must be positive")  # 协方差方差项不能为零或负。
    return cov  # 已经是合法矩阵就直接返回。


def _quality_below_floor(quality: float, floor: float) -> bool:  # 判断质量值是否低于门槛。委托到共享入口。
    """判断质量值是否低于门槛。委托到 common.validation.quality_below_floor 共享入口。"""
    return quality_below_floor(quality, floor, epsilon=QUALITY_FLOOR_EPSILON)


class RobustEKFCore(EKFCore):  # 在标准 EKF 基础上增加质量门控和鲁棒协方差缩放。
    """在标准 EKF 基础上增加质量门控和鲁棒协方差缩放。

    相比 :class:`EKFCore`，本类在处理 UWB 和 VIO 更新前增加了两层门控：

    1. **质量门控** —— 量测 quality 低于 ``gate.quality_floor`` 时拒绝更新
    2. **NIS 门控** —— 归一化创新平方(NIS)超过 ``gate.mahalanobis_sq`` 时拒绝更新

    通过门控后，使用 Huber 规则对残差进行降权，等价于膨胀测量协方差，
    降低异常量测对状态估计的影响。

    配置字段
    ----------
    gate : dict
        门控参数，包含 ``quality_floor`` 和 ``mahalanobis_sq``。
    robust_weight : dict
        鲁棒权重参数，包含 ``type``（目前仅支持 ``"huber"``）和 ``delta``。
    """

    def __init__(self, init_cfg: dict | None = None) -> None:  # 初始化父类并覆写模块名称。
        """初始化鲁棒 EKF，并把名字改成 robust_ekf。

        参数
        ----------
        init_cfg : dict | None
            初始化配置字典，除标准 EKF 配置外还支持
            ``gate`` 和 ``robust_weight`` 子配置。
        """
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "estimator": "RobustEKFCore",
            "name": (init_cfg or {}).get("name", "robust_ekf"),
            "cfg_keys": list(init_cfg.keys()) if isinstance(init_cfg, dict) else None,
            "gate": (init_cfg or {}).get("gate"),
            "robust_weight": (init_cfg or {}).get("robust_weight"),
        }, "RobustEKFCore.__init__")
        super().__init__(init_cfg)  # 先让父类建立基础 EKF 状态与默认配置。
        # 默认名字和标准 EKF 不同，方便上层日志区分。
        self.name = str(self.cfg.get("name") or "robust_ekf")  # 上层日志里使用的实例名。

    def _reject_report(  # 统一构造拒绝类报告。
        self,  # 当前实例本身。
        *,  # 后面的参数要求全部按关键字传入。
        modality: str,  # 当前事件模态名称。
        control: MeasurementControl,  # 当前控制对象。
        quality: float,  # 当前量测质量值。
        nis: float | None,  # 当前 NIS，可能为空。
        rejected_by: str,  # 拒绝原因。
        covariance_report: dict[str, Any] | None = None,  # 基础协方差报告。
        robust_covariance_report: dict[str, Any] | None = None,  # 鲁棒协方差报告。
        extra: dict[str, Any] | None = None,  # 额外补充字段。
    ) -> dict[str, Any]:  # 返回统一格式的拒绝报告。
        """统一构造拒绝类报告。

        参数
        ----------
        modality : str
            事件模态名称。
        control : MeasurementControl
            当前控制对象。
        quality : float
            当前量测质量值。
        nis : float | None
            当前 NIS 值，质量门拒绝时为 None。
        rejected_by : str
            拒绝原因，如 ``"quality_floor"`` 或 ``"mahalanobis_sq"``。
        covariance_report : dict[str, Any] | None
            基础控制协方差报告，可能为 None。
        robust_covariance_report : dict[str, Any] | None
            鲁棒协方差报告，拒绝时为 None。
        extra : dict[str, Any] | None
            额外补充字段，合并到报告顶层。

        返回
        -------
        dict[str, Any]
            统一格式的拒绝报告，包含 modality、update_applied=False、
            gate 子报告和 robust=None。
        """
        report = {  # 先把拒绝类报告的公共字段拼成统一结构。
            "modality": modality,  # 事件模态。
            "update_applied": False,  # 这里明确标记没有真正更新。
            "reason": rejected_by,  # 与其他估计器和消费方合同一致，顶层显式带出拒绝原因。
            # §3.0.2 审计要求：拒绝路径虽未修改状态，但报告接口须统一声明
            # write-port 类型（与成功路径同口径）。UWB 模态走 h-side bias 通道，
            # VIO 模态标记 "none"（无 bias 写入口），方便顶层聚合审计区分。
            "bias_writeport": "h_side" if modality.startswith("uwb") else "none",
            "measurement_control": control_to_dict(control),  # 当前控制信息。
            "covariance_report": covariance_report,  # 基础协方差报告。
            "robust_covariance_report": robust_covariance_report,  # 鲁棒协方差报告。
            "gate": {  # 门控子报告单独分层，方便上层直接读拒绝原因。
                "passed": False,  # 明确说明门控没通过。
                "quality": quality,  # 当前质量值。
                "quality_floor": self._quality_floor(modality),  # 当前模态的质量门槛。
                "nis": nis,  # 当前 NIS 值，可能为空。
                "mahalanobis_sq_threshold": self._nis_threshold(modality),  # NIS 阈值（按模态取，与父类 EKFCore 同口径）。
                "rejected_by": rejected_by,  # 记录拒绝原因。
            },  # 门控子报告结束。
            "robust": None,  # 被拒绝时没有鲁棒缩放结果。
        }  # 报告主体先构造完毕。
        if extra:  # 如果调用方补充了额外细节，就合并进去。
            # 额外字段只做增补，禁止覆盖主体语义字段
            for _extra_key, _extra_value in extra.items():
                if _extra_key not in report:  # 只增补新字段，不覆盖主体语义。
                    report[_extra_key] = _extra_value
        return report  # 返回统一的拒绝报告结构。

    def _handle_uwb(self, payload: dict[str, Any], x_prev: np.ndarray, control: MeasurementControl) -> dict[str, Any]:  # 处理 UWB 量测，包含质量门控和鲁棒权重。
        """处理 UWB 量测，包含质量门控和鲁棒权重。

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
            更新报告，包含 gate、robust、covariance_report、
            robust_covariance_report 等子报告。

        注意
        -----
        处理流程：

        1. 控制层跳过检查（``uwb_skip_update``）
        2. 质量门控（``quality < quality_floor`` → 拒绝）
        3. 协方差控制缩放
        4. NIS 门控（``nis > mahalanobis_sq`` → 拒绝）
        5. Huber 鲁棒权重计算 + 协方差膨胀
        6. 标准 UWB 更新

        偏置修正后距离小于 0 时钳位到 0。
        """
        if control.gate_action == "uwb_skip_update":  # 控制层显式要求跳过。
            return {  # 控制层要求跳过时，直接返回跳过报告。
                "modality": "uwb",  # 模态标签。
                "update_applied": False,  # 这次没有执行更新。
                "reason": "uwb_skip_update",  # 跳过原因。
                "bias_writeport": "h_side",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "covariance_report": None,  # 没有做协方差缩放时为空。
                "robust_covariance_report": None,  # 鲁棒协方差报告也为空。
                "gate": {"passed": False, "rejected_by": "uwb_skip_update"},
                "robust": None,  # 跳过时不返回鲁棒细节。
            }  # 跳过报告结束。
        uwb_payload = payload.get("uwb_payload")  # 提取 UWB 负载。
        if uwb_payload is None:  # 负载缺失时无法执行更新，直接跳过。
            return {  # 返回跳过报告。
                "modality": "uwb",
                "update_applied": False,
                "reason": "missing_uwb_payload",
                "bias_writeport": "h_side",  # §3.0.2 审计
                "measurement_control": control_to_dict(control),
                "covariance_report": None,
                "robust_covariance_report": None,
                "gate": {"passed": False, "rejected_by": "missing_uwb_payload"},
                "robust": None,
            }
        # §11.3-d 共享「传感器无效」硬标志串项同源：与 ekf_core.py:856-869 同口径，
        # valid=False 的 UWB 事件语义上是无效观测，跳过更新，不依赖控制层 gate_action。
        # v9 audit 发现：v6 漏审此串项不对称——EKF 做了 valid=False 检查但 Robust-EKF 缺，
        # 违 §11.3-d「无效标志 → 方法内部更新的串联顺序全员固定」。
        uwb_valid = uwb_payload.get("valid", True)
        if is_bool_like(uwb_valid) and not bool(uwb_valid):
            return self._reject_report(
                modality="uwb",
                control=control,
                quality=None,
                nis=None,
                rejected_by="uwb_invalid_measurement",
                covariance_report=None,
                robust_covariance_report=None,
            )
        quality = self._quality_value(uwb_payload)  # 从 UWB 负载里取质量值。
        if _quality_below_floor(quality, self._quality_floor("uwb")):  # 质量太低直接拒绝。
            return self._reject_report(  # 直接返回质量门失败报告。
                modality="uwb",  # 当前事件模态是 UWB。
                control=control,  # 控制对象原样带回。
                quality=quality,  # 当前质量值。
                nis=None,  # 这里还没有计算 NIS。
                rejected_by="quality_floor",  # 拒绝原因是质量门。
                covariance_report=None,  # 还没做协方差缩放。
                robust_covariance_report=None,  # 也没有鲁棒缩放报告。
            )  # 质量门失败报告结束。

        base_uwb_noise = self._required_measurement_noise("uwb")  # 基础 UWB 噪声（标准差）。
        base_uwb_noise = _square_std_to_var(base_uwb_noise)  # 标准差平方为方差，与过程噪声语义一致。
        effective_uwb_noise, cov_report = build_controlled_measurement_cov(  # 先按控制策略把方差缩放成有效噪声。
            base_uwb_noise,  # 方差。
            control,  # 当前控制对象。
            modality="uwb",  # UWB 分支。
            calibration_frozen=self._calibration_frozen,  # §2.2 标定冻结：R 标定后不被 LNN 再缩放。
        )  # 这里同时返回有效噪声和缩放报告。
        anchor_pos = self._resolve_anchor_position(uwb_payload["anchor_id"])  # 锚点坐标。
        # §3.0.2 / §3.1.2：bias 在 h(·) 侧注入到 z_pred，残差定义在原始 z 上。
        # 不再对 raw 距离做 subtractive 改写；raw_range 直接作为 z 进 update。
        # §4.4 L1056 协议级护栏（前提指导.md:1049-1063 前置滤波与「干净距离」偷换）：
        # raw_range 不做 max(0,·)/clip 等单方截断；与 ekf_core.py:891 + build_measurement_control
        # L829 同口径 `float(uwb_payload["range"])`，避免 robust_ekf 单方"软削波只给一方"违 §4.4。
        raw_range = float(uwb_payload["range"])
        bias_applied_h = float(control.bias_applied)  # 已由 clip_uwb_bias 截断为非负有界。
        z_pred = predict_range(x_prev, anchor_pos, extra_bias=bias_applied_h)  # h 侧含 bias 的预测测距。
        residual = raw_range - z_pred  # 残差定义在原始 z 上（§3.0.2）。
        H = build_uwb_jacobian(x_prev, anchor_pos)  # 雅可比。
        noise_arr = np.asarray(effective_uwb_noise, dtype=float)
        if noise_arr.ndim != 0:
            raise ValueError("effective UWB noise must be scalar, got shape {}".format(noise_arr.shape))
        scalar_noise = coerce_finite_scalar(noise_arr.reshape(-1)[0], name="effective UWB noise")  # 标量噪声必须有限。
        S = coerce_finite_scalar((H @ self._covariance @ H.T)[0, 0] + scalar_noise, name="UWB innovation covariance")  # 创新协方差必须有限。
        # §11.5 抖动注入：UWB 路径标量 S 与 VIO 路径同口径，缺 jitter fallback 是 v5 漏审。
        # 三方法（EKF / Robust-EKF / FGO）UWB 路径 S<=0 拒绝前先尝试 jitter 修补；
        # 二次仍失败 → fail-loud（与 VIO 路径 _ensure_positive_definite_vio_innovation_covariance 同政策）。
        if S <= 0.0:  # 非正创新协方差说明状态退化，先尝试 jitter 修补。
            from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS
            cov_jitter_eps = float(BRIDGE_THRESHOLDS["cov_jitter_eps"])
            jittered_S = S + cov_jitter_eps
            if jittered_S > 0.0:
                S = jittered_S  # 接受 jittered 版本作为该次创新协方差。
            else:
                return self._reject_report(
                    modality="uwb",
                    control=control,
                    quality=quality,
                    nis=None,
                    rejected_by="nonpositive_innovation_covariance",
                    covariance_report=cov_report,
                    robust_covariance_report=None,
                    extra={
                        "z_pred": z_pred,
                        "residual": residual,
                        "H": H,
                        "S": S,
                    },
                )
        nis = float((residual * residual) / S)  # 计算 NIS。
        if not math.isfinite(nis):  # NaN/inf NIS 会绕过门控（NaN > 阈值为 False），必须显式拒绝。
            return self._reject_report(
                modality="uwb",
                control=control,
                quality=quality,
                nis=nis,
                rejected_by="nonfinite_nis",
                covariance_report=cov_report,
                robust_covariance_report=None,
                extra={
                    "z_pred": z_pred,
                    "residual": residual,
                    "H": H,
                    "S": S,
                },
            )
        if nis > self._nis_threshold("uwb"):  # NIS 超阈值也直接拒绝（modality="uwb" 与父类 EKFCore 同口径）。
            return self._reject_report(  # 这里把拒绝原因、门控值和中间量一起汇总返回。
                modality="uwb",  # 当前事件类型是 UWB。
                control=control,  # 原样带回控制对象，方便上层审计。
                quality=quality,  # 当前量测质量值。
                nis=nis,  # 当前 NIS 数值。
                rejected_by="mahalanobis_sq",  # 拒绝原因是马氏距离超阈值。
                covariance_report=cov_report,  # 基础控制协方差的报告。
                robust_covariance_report=None,  # 这里还没有进入鲁棒缩放更新。
                extra={  # 额外把中间量也带回去，方便排查门控为什么失败。
                    "z_pred": z_pred,  # 几何预测距离。
                    "residual": residual,  # 当前测距残差。
                    "H": H,  # UWB 雅可比矩阵。
                    "S": S,  # 创新协方差标量。
                },  # 额外字段到这里结束。
            )  # 拒绝报告构造完成。

        whitened = math.sqrt(max(nis, 0.0))  # 白化残差范数，S>0 已保证 nis 有限。
        robust_weight = self._huber_weight(whitened)  # 根据白化残差算出鲁棒权重。
        covariance_scale = 1.0 / max(robust_weight, 1e-6)  # 权重越小，协方差放大得越多。
        robust_noise, robust_cov_report = build_effective_cov(  # 再把有效噪声按鲁棒规则放大。
            effective_uwb_noise,  # 原始有效噪声。
            uwb_scaling=covariance_scale,  # 鲁棒缩放系数。
        )  # 鲁棒缩放报告也一起返回。
        x_upd, P_upd, update_info = run_uwb_update(  # 真正执行一次 UWB 更新。
            x_prev,  # 当前状态。
            self._covariance,  # 当前协方差。
            anchor_pos,  # 锚点坐标。
            raw_range,  # 原始测距（残差在原始 z 合同上定义，bias 已进 h(·)）。
            robust_noise,  # 鲁棒后噪声。
            extra_bias=bias_applied_h,  # h 侧有界偏置修正（§3.0.2 / §3.1.2）。
        )  # 更新函数同时返回新状态、新协方差和细节报告。
        self._update_from_vector(x_upd, P_upd)  # 写回内部缓存。
        return {  # 返回一次成功的 UWB 更新报告。
            "modality": "uwb",  # 模态标签。
            "update_applied": True,  # 这次成功更新。
            "measurement_control": control_to_dict(control),  # 当前控制参数。
            "covariance_report": cov_report,  # 基础协方差报告。
            "robust_covariance_report": robust_cov_report,  # 鲁棒协方差报告。
            # §3.0.2 审计：显式标记本次 bias 写入 h(·) 侧，残差定义在原始 z 上。
            "bias_writeport": "h_side",
            "gate": {  # 门控子报告。
                "passed": True,  # 门控通过。
                "quality": quality,  # 当前质量值。
                "quality_floor": self._quality_floor("uwb"),  # 当前模态的质量门槛。
                "nis": nis,  # 当前 NIS。
                "mahalanobis_sq_threshold": self._nis_threshold("uwb"),  # 马氏距离阈值（modality="uwb" 与父类 EKFCore 同口径）。
                "rejected_by": None,  # 没有拒绝。
            },  # 门控子报告结束。
            "robust": {  # 鲁棒子报告。
                "type": str(self._robust_cfg().get("type", "huber")).lower(),  # 鲁棒类型。
                "delta": float(self._robust_cfg().get("delta", 1.0)),  # Huber 阈值。
                "whitened_residual_norm": whitened,  # 白化残差范数。
                "weight": robust_weight,  # 鲁棒权重。
                "covariance_scale": covariance_scale,  # 协方差缩放因子。
            },  # 鲁棒子报告结束。
            **{k: v for k, v in update_info.items()
               if k not in ("gate", "modality", "update_applied", "reason",
                            "covariance_report", "robust_covariance_report",
                            "measurement_control", "robust")},  # 展开 UWB 更新细节（不含已构造字段）。
        }  # UWB 成功更新报告结束。

    def _handle_vio(self, payload: dict[str, Any], x_prev: np.ndarray, control: MeasurementControl) -> dict[str, Any]:  # 处理 VIO 量测，包含质量门控和鲁棒权重。
        """处理 VIO 量测，包含质量门控和鲁棒权重。

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
            更新报告，包含 gate、robust、covariance_report、
            robust_covariance_report 等子报告。

        注意
        -----
        处理流程与 UWB 一致：

        1. 控制层跳过检查（``vio_skip_update``）
        2. 质量门控
        3. 协方差控制缩放
        4. NIS 门控
        5. Huber 鲁棒权重计算 + 协方差膨胀
        6. 标准 VIO 更新

        更新成功后刷新 ``_last_vio_reference_pose``。
        底层返回的 ``gate`` 字段会被移除，由本类统一管理门控报告。
        """
        if control.gate_action == "vio_skip_update":  # 控制层显式要求跳过。
            # 跳过更新时重置参考位姿为当前估计位姿，避免下一帧 VIO 更新时
            # 使用过时参考位姿导致残差语义不匹配（仿真 VIO 数据提供相邻帧增量，
            # 而非参考位姿到当前的增量）。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            return {  # 控制层要求跳过时，直接返回跳过报告。
                "modality": "vio",  # 模态标签。
                "update_applied": False,  # 这次没有执行更新。
                "reason": "vio_skip_update",  # 跳过原因。
                # §3.0.2 审计：VIO 不接受 NN bias，标记 "none"。
                "bias_writeport": "none",
                "measurement_control": control_to_dict(control),  # 当前控制参数。
                "covariance_report": None,  # 没有做协方差缩放时为空。
                "robust_covariance_report": None,  # 鲁棒协方差报告也为空。
                "gate": {"passed": False, "rejected_by": "vio_skip_update"},
                "robust": None,  # 跳过时不返回鲁棒细节。
            }  # 跳过报告结束。
        quality = self._quality_value(payload.get("vio_payload"))  # 先从 VIO 负载里取质量值。
        # quality<=0 的 VIO 事件语义上是无效观测（如仿真 cycle 边界帧），跳过更新并重置参考位姿。
        # 与 ekf_core 行为一致：quality<=0 是协议级检查，不受 quality_floor 配置影响。
        if quality <= 0.0:
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            return self._reject_report(
                modality="vio",
                control=control,
                quality=quality,
                nis=None,
                rejected_by="vio_quality_zero",
                covariance_report=None,
                robust_covariance_report=None,
            )
        if _quality_below_floor(quality, self._quality_floor("vio")):  # 质量太低直接拒绝。
            # 仿真 VIO 数据提供相邻帧增量，拒绝后下一帧 VIO 仍是相邻帧增量，
            # 参考位姿必须重置为当前位姿，否则残差语义不匹配。与 EKF 行为一致。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            return self._reject_report(  # 直接返回质量门失败报告。
                modality="vio",  # 当前事件模态是 VIO。
                control=control,  # 控制对象原样带回。
                quality=quality,  # 当前质量值。
                nis=None,  # 这里还没有计算 NIS。
                rejected_by="quality_floor",  # 拒绝原因是质量门。
                covariance_report=None,  # 还没做协方差缩放。
                robust_covariance_report=None,  # 也没有鲁棒缩放报告。
            )  # 质量门失败报告结束。

        base_vio_noise = self._required_measurement_noise("vio")  # 基础 VIO 噪声（标准差）。
        base_vio_noise = _square_std_to_var(base_vio_noise)  # 标准差平方为方差，与过程噪声语义一致。
        effective_vio_cov, cov_report = build_controlled_measurement_cov(  # 先按控制策略缩放 VIO 方差。
            base_vio_noise,  # 方差。
            control,  # 当前控制对象。
            modality="vio",  # VIO 分支。
            calibration_frozen=self._calibration_frozen,  # §2.2 标定冻结：R 标定后不被 LNN 再缩放。
        )  # 这里同时返回有效噪声和缩放报告。
        reference_pose = self._last_vio_reference_pose or self._current_pose_reference()  # 参考位姿。
        # 首个 VIO 事件参考位姿检查：如果 _last_vio_reference_pose 为 None，
        # 参考位姿回退到当前估计位姿，导致 z_hat=0 而 z_vio 非零，残差异常放大。
        # 此时跳过更新，将参考位姿设为当前位姿供下一帧使用。与 EKF 行为一致。
        if self._last_vio_reference_pose is None:
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            return self._reject_report(
                modality="vio",
                control=control,
                quality=quality,
                nis=None,
                rejected_by="vio_first_frame_reference_init",
                covariance_report=cov_report,
                robust_covariance_report=None,
            )
        # 参考位姿时效性检查：如果参考位姿超过阈值未更新，说明中间可能有 VIO 事件
        # 被 blackout 移除或门控跳过。仿真 VIO 数据提供相邻帧增量，参考位姿过时时
        # 增量语义不匹配，此时应重置参考位姿并跳过更新，与 EKF 行为一致。
        ref_pose_stale = (
            self._last_vio_reference_pose is not None
            and self._last_vio_reference_pose_timestamp is not None
            and self._timestamp is not None
            and (self._timestamp - self._last_vio_reference_pose_timestamp) > VIO_REF_POSE_STALE_SECONDS
        )
        if ref_pose_stale:
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            return self._reject_report(
                modality="vio",
                control=control,
                quality=quality,
                nis=None,
                rejected_by="vio_reference_pose_stale",
                covariance_report=cov_report,
                robust_covariance_report=None,
            )
        z_vio = build_vio_measurement(payload)  # VIO 量测。
        z_hat, residual, H, reference_pose_array = compute_vio_residual(  # 计算 VIO 预测量测、残差和雅可比。
            x_prev,  # 当前状态。
            z_vio,  # 视觉量测。
            reference_pose=reference_pose,  # 参考位姿。
        )  # 线性化结果到此结束。
        R_vio = _normalize_vio_covariance(effective_vio_cov)  # 规整成协方差矩阵。
        S = H @ self._covariance @ H.T + R_vio  # 创新协方差。
        if not np.all(np.isfinite(S)):  # 创新协方差必须全是有限数。
            raise ValueError("VIO innovation covariance must be finite")  # 不能把 nan 或 inf 带进更新。
        try:
            S = _ensure_positive_definite_vio_innovation_covariance(
                S,
                name="VIO innovation covariance",
            )
        except ValueError:
            # 仿真 VIO 数据提供相邻帧增量，拒绝后下一帧 VIO 仍是相邻帧增量，
            # 参考位姿必须重置为当前位姿，否则残差语义不匹配。与 LinAlgError 路径一致。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            return self._reject_report(
                modality="vio",
                control=control,
                quality=quality,
                nis=None,
                rejected_by="nonpositive_innovation_covariance",
                covariance_report=cov_report,
                robust_covariance_report=None,
                extra={
                    "z_vio": z_vio.tolist(),
                    "z_hat": z_hat.tolist(),
                    "reference_pose": reference_pose_array.tolist(),
                    "residual": residual.tolist(),
                    "covariance_report": cov_report,
                },
            )
        try:
            nis = float(residual.T @ np.linalg.solve(S, residual))  # 计算 NIS。
        except np.linalg.LinAlgError:
            # 仿真 VIO 数据提供相邻帧增量，拒绝后下一帧 VIO 仍是相邻帧增量，
            # 参考位姿必须重置为当前位姿，否则残差语义不匹配。与 EKF/FGO 行为一致。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            return self._reject_report(
                modality="vio",
                control=control,
                quality=quality,
                nis=None,
                rejected_by="nonpositive_innovation_covariance",
                covariance_report=cov_report,
                robust_covariance_report=None,
                extra={
                    "z_vio": z_vio.tolist(),
                    "z_hat": z_hat.tolist(),
                    "reference_pose": reference_pose_array.tolist(),
                    "residual": residual.tolist(),
                    "covariance_report": cov_report,
                },
            )
        if not math.isfinite(nis):  # NIS 必须有限，NaN/inf 说明状态或残差被污染。
            # 仿真 VIO 数据提供相邻帧增量，拒绝后下一帧 VIO 仍是相邻帧增量，
            # 参考位姿必须重置为当前位姿，否则残差语义不匹配。与 EKF/FGO 行为一致。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            return self._reject_report(
                modality="vio",
                control=control,
                quality=quality,
                nis=nis,
                rejected_by="nonfinite_nis",
                covariance_report=cov_report,
                robust_covariance_report=None,
                extra={
                    "z_vio": z_vio.tolist(),
                    "z_hat": z_hat.tolist(),
                    "reference_pose": reference_pose_array.tolist(),
                    "residual": residual.tolist(),
                    "covariance_report": cov_report,
                },
            )
        if nis > self._nis_threshold("vio"):  # NIS 超阈值也直接拒绝（modality="vio" 与父类 EKFCore 同口径）。
            # 仿真 VIO 数据提供相邻帧增量，拒绝后下一帧 VIO 仍是相邻帧增量，
            # 参考位姿必须重置为当前位姿，否则残差语义不匹配。与 EKF/FGO 行为一致。
            self._last_vio_reference_pose = self._current_pose_reference()
            self._last_vio_reference_pose_timestamp = self._timestamp
            return self._reject_report(  # 把拒绝原因和中间量一并返回。
                modality="vio",  # 当前事件模态是 VIO。
                control=control,  # 控制对象原样带回。
                quality=quality,  # 当前质量值。
                nis=nis,  # 当前 NIS 数值。
                rejected_by="mahalanobis_sq",  # 拒绝原因是马氏距离超阈值。
                covariance_report=cov_report,  # 基础协方差报告。
                robust_covariance_report=None,  # 这里还没有进入鲁棒更新。
                extra={  # 额外把中间量也带回去，方便排查。
                    "z_vio": z_vio.tolist(),  # 原始视觉量测。
                    "z_hat": z_hat.tolist(),  # 预测视觉量测。
                    "reference_pose": reference_pose_array.tolist(),  # 参考位姿。
                    "residual": residual.tolist(),  # 残差向量。
                },  # 额外字段结束。
            )  # 拒绝报告构造完成。

        whitened = math.sqrt(max(nis, 0.0))  # 白化残差范数，用来驱动鲁棒缩放。
        robust_weight = self._huber_weight(whitened)  # 计算 VIO 的鲁棒权重。
        covariance_scale = 1.0 / max(robust_weight, 1e-6)  # 把权重换成协方差缩放因子。
        robust_vio_cov, robust_cov_report = build_effective_cov(  # 基于鲁棒权重放大 VIO 协方差。
            effective_vio_cov,  # 有效视觉协方差。
            vio_scaling=covariance_scale,  # 鲁棒缩放。
        )  # 这里会同时返回鲁棒后的协方差和适配报告。
        # §6.4 #18 紧耦合角色硬子要求对应代码层落点：
        # VIO 因子/更新挂增量残差（reference_pose=上一帧参考位姿）与 UWB 距离残差并列，
        # 而非先视觉里程计积分成轨迹再松耦合（§6.4 #19）。
        # 三估计器同口径：EKF ekf_core.py:1188 / RobustEKF 本处 L694 / FGO fgo_core.py:2224。
        # 残差锚定上一帧参考位姿，立即写回并通过 _update_from_vector 同步状态/协方差。
        x_upd, P_upd, update_report = apply_vision_update(  # 真正执行一次 VIO 更新。
            x_prev,  # 当前状态。
            self._covariance,  # 当前协方差。
            payload,  # 视觉事件。
            robust_vio_cov,  # 鲁棒后协方差。
            reference_pose=reference_pose,  # 参考位姿。
        )  # 更新函数同时返回新状态、新协方差和更新细节。
        update_report = dict(update_report)  # 复制一份，避免修改底层函数返回对象。
        self._update_from_vector(x_upd, P_upd)  # 写回状态。
        self._last_vio_reference_pose = self._current_pose_reference()  # 更新参考位姿。
        self._last_vio_reference_pose_timestamp = self._timestamp  # 记录参考位姿更新时间戳。
        return {  # 返回一次成功的 VIO 更新报告。
            "modality": "vio",  # 模态标签。
            "update_applied": True,  # 这次成功更新。
            "measurement_control": control_to_dict(control),  # 当前控制参数。
            "covariance_report": cov_report,  # 基础协方差报告。
            "robust_covariance_report": robust_cov_report,  # 鲁棒协方差报告。
            # §3.0.2 审计要求：VIO 不接受 NN bias（build_measurement_control VIO 分支
            # 强制 bias_applied=0.0，残差定义在原始增量 z 上）。显式标记 "none"，
            # 与 EKF VIO / FGO VIO 同口径，便于顶层聚合审计。
            "bias_writeport": "none",
            "gate": {  # 门控子报告。
                "passed": True,  # 门控通过。
                "quality": quality,  # 当前质量值。
                "quality_floor": self._quality_floor("vio"),  # 当前模态的质量门槛。
                "nis": nis,  # 当前 NIS。
                "mahalanobis_sq_threshold": self._nis_threshold("vio"),  # 马氏距离阈值（modality="vio" 与父类 EKFCore 同口径）。
                "rejected_by": None,  # 没有拒绝。
            },  # 门控子报告结束。
            "robust": {  # 鲁棒子报告。
                "type": str(self._robust_cfg().get("type", "huber")).lower(),  # 鲁棒类型。
                "delta": float(self._robust_cfg().get("delta", 1.0)),  # Huber 阈值。
                "whitened_residual_norm": whitened,  # 白化残差范数。
                "weight": robust_weight,  # 鲁棒权重。
                "covariance_scale": covariance_scale,  # 协方差缩放因子。
            },  # 鲁棒子报告结束。
            **{k: v for k, v in update_report.items()
               if k not in ("gate", "modality", "update_applied", "reason")},  # 展开 VIO 更新细节（不含已构造字段）。
        }  # VIO 成功更新报告结束。
