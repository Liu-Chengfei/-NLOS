"""冻结传感器字段与特征缺失策略合同辅助模块。

文件职责：
  作为 configs/base/sensors.yaml 的唯一代码入口，校验该文件的形状
  并暴露只读辅助函数，确保协议、传感器和特征代码不会偏离声明的合同。
  核心目标是防止传感器字段定义和缺失处理策略在运行时漂移。

本文件绝对不负责：
  不定义传感器字段内容本身（内容由 sensors.yaml 定义）。
  不修改传感器配置文件。
  不执行任何数据读取或特征提取逻辑。

核心数据流：
    sensors.yaml → load_sensor_contract → 校验通过 → get_required_payload_fields / get_feature_missing_policy

上游依赖：
  liquidloc.common.config_utils（load_yaml_config 加载 YAML 配置）、
  liquidloc.common.constants（MODALITY_IMU、MODALITY_UWB、MODALITY_VIO 模态常量）

下游调用者：
  protocol/event_schema.py（使用 get_required_payload_fields 校验事件 payload）、
  dataio/（使用 get_required_payload_fields 和 get_feature_missing_policy）、
  features/（使用 get_feature_missing_policy 处理缺失值）

输入对象定义：
  - config_path  可选的传感器配置文件路径，默认使用 configs/base/sensors.yaml

输出对象定义：
  - load_sensor_contract           加载并校验冻结传感器合同
  - get_required_payload_fields    获取每种模态必需的 payload 字段元组
  - get_feature_missing_policy     获取特征缺失策略配置

核心变量定义：
  - _CONFIG_ROOT                   配置文件根目录
  - _SENSORS_CONFIG_PATH           传感器配置文件路径
  - _MODALITY_FIELD_KEYS           模态名到 YAML 字段键的映射
  - _ALLOWED_MISSING_SOURCES       允许的缺失来源类型
  - _ALLOWED_MISSING_CARRIERS      允许的缺失语义载体
  - _SENSOR_CONTRACT               模块加载时冻结的传感器合同快照

关键设计决策：
  - 传感器合同在模块加载时即冻结，后续调用只读，防止运行时篡改。
  - 校验时与仓库冻结快照做精确比较，任何漂移都会导致加载失败。
  - 特征缺失策略的 source_precedence 只允许 event_root、modality_payload、state_ctx 三种来源。
  - missing_semantics_carrier 只允许 missing_mask 一种载体。
  - 所有字段定义必须完整，涵盖 IMU、UWB、VIO、FLOW、TOF 五种模态。
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

from liquidloc.common.config_utils import find_project_root, load_yaml_config
from liquidloc.common.constants import (
    MODALITY_FLOW,
    MODALITY_IMU,
    MODALITY_TOF,
    MODALITY_UWB,
    MODALITY_VIO,
)
from liquidloc.common.validation import is_bool_like, is_real, is_string_like

_CONFIG_ROOT = (find_project_root() / "configs" / "base").resolve()  # 配置文件根目录，resolve() 在模块加载时完成。
_SENSORS_CONFIG_PATH = _CONFIG_ROOT / "sensors.yaml"  # 传感器配置文件路径。

_MODALITY_FIELD_KEYS = {  # 模态名到 YAML 中对应字段键名的映射。
    MODALITY_IMU: "imu_fields",  # IMU 模态对应的字段键。
    MODALITY_UWB: "uwb_fields",  # UWB 模态对应的字段键。
    MODALITY_VIO: "vio_fields",  # VIO 模态对应的字段键。
    MODALITY_FLOW: "flow_fields",  # 光流/流量类模态对应的字段键（UTIL 数据集）。
    MODALITY_TOF: "tof_fields",  # ToF 测距类模态对应的字段键（UTIL 数据集）。
}
_ALLOWED_MISSING_SOURCES = ("event_root", "modality_payload", "state_ctx")  # 允许的缺失来源类型，只有这三种。
_ALLOWED_MISSING_CARRIERS = ("missing_mask",)  # 允许的缺失语义载体，只有 missing_mask 一种。


def _validate_field_list(value: Any, *, name: str) -> tuple[str, ...]:
    """校验字段列表是否为非空字符串列表且无重复项。

    参数：
        value: 待校验的字段列表值。
        name: 字段名，用于构造错误信息。

    返回：
        tuple[str, ...]: 校验通过的字段名元组。

    异常：
        TypeError: 值不是非空列表或包含非字符串/空字符串项时抛出。
        ValueError: 列表包含重复字段名时抛出。
    """
    if not isinstance(value, list) or not value:  # 必须是非空列表。
        raise TypeError(f"{name} must be a non-empty list")
    fields: list[str] = []  # 存放校验后的字段名。
    for index, field_name in enumerate(value):  # 逐项检查。
        if not is_string_like(field_name) or not field_name:  # 每项必须是非空字符串。
            raise TypeError(f"{name}[{index}] must be a non-empty string")
        fields.append(field_name)
    if len(set(fields)) != len(fields):  # 不允许重复字段名。
        raise ValueError(f"{name} must not contain duplicate fields")
    return tuple(fields)  # 返回不可变元组。


def _validate_source_precedence(value: Any) -> tuple[str, ...]:
    """校验特征缺失策略的来源优先级列表。

    只允许 _ALLOWED_MISSING_SOURCES 中的来源类型，且不允许重复。

    参数：
        value: 待校验的来源优先级列表。

    返回：
        tuple[str, ...]: 校验通过的来源优先级元组。

    异常：
        TypeError: 值不是非空列表时抛出。
        ValueError: 包含不支持的来源或重复来源时抛出。
    """
    if not isinstance(value, list) or not value:  # 必须是非空列表。
        raise TypeError("feature_missing_policy.source_precedence must be a non-empty list")
    precedence: list[str] = []  # 存放校验后的来源名。
    for index, source_name in enumerate(value):  # 逐项检查。
        if source_name not in _ALLOWED_MISSING_SOURCES:  # 来源必须在允许列表中。
            raise ValueError(
                "feature_missing_policy.source_precedence "
                f"contains unsupported source at index {index}: {source_name!r}"
            )
        precedence.append(source_name)
    if len(set(precedence)) != len(precedence):  # 不允许重复来源。
        raise ValueError("feature_missing_policy.source_precedence must not contain duplicates")
    return tuple(precedence)  # 返回不可变元组。


def _normalize_sensor_contract_cfg(cfg: Any) -> dict[str, Any]:
    """将传感器合同配置规范化并校验为冻结运行时形状。

    校验每种模态的必需 payload 字段列表和特征缺失策略的各个子项，
    确保配置结构与协议合同一致。

    参数：
        cfg: 待规范化的传感器合同配置。

    返回：
        dict[str, Any]: 规范化后的传感器合同配置。

    异常：
        TypeError: 配置不是映射或字段类型不正确时抛出。
        ValueError: 字段值不合法时抛出。
    """
    if not isinstance(cfg, dict):  # 配置必须是映射。
        raise TypeError("sensor contract config must be a mapping")
    required_payload_fields = {  # 校验每种模态的必需字段列表。
        modality: _validate_field_list(cfg.get(field_key), name=field_key)  # 逐模态校验。
        for modality, field_key in _MODALITY_FIELD_KEYS.items()  # 遍历所有模态。
    }
    feature_missing_policy = cfg.get("feature_missing_policy")  # 读取特征缺失策略。
    if not isinstance(feature_missing_policy, dict):  # 策略必须是映射。
        raise TypeError("feature_missing_policy must be a mapping")
    source_precedence = _validate_source_precedence(feature_missing_policy.get("source_precedence"))  # 校验来源优先级。
    numeric_fill_value = feature_missing_policy.get("numeric_fill_value")  # 读取数值填充值。
    if not is_real(numeric_fill_value):  # 必须是数值型，排除 bool 和 complex。
        raise TypeError("feature_missing_policy.numeric_fill_value must be numeric")
    numeric_fill_value = float(numeric_fill_value)  # 统一转为浮点数。
    if not math.isfinite(numeric_fill_value):  # 必须是有限数。
        raise ValueError("feature_missing_policy.numeric_fill_value must be finite")
    require_missing_mask = feature_missing_policy.get("require_missing_mask")  # 读取是否要求缺失掩码。
    if not is_bool_like(require_missing_mask):  # 必须是布尔值（含 numpy.bool_）。
        raise TypeError("feature_missing_policy.require_missing_mask must be a bool")
    require_missing_mask = bool(require_missing_mask)  # 显式转为 Python bool，避免 numpy.bool_ 泄漏到合同内部。
    missing_semantics_carrier = feature_missing_policy.get("missing_semantics_carrier")  # 读取缺失语义载体。
    if missing_semantics_carrier not in _ALLOWED_MISSING_CARRIERS:  # 载体必须在允许列表中。
        raise ValueError(
            "feature_missing_policy.missing_semantics_carrier must be "
            f"one of {_ALLOWED_MISSING_CARRIERS}, got {missing_semantics_carrier!r}"
        )

    # ---- feature_missing_policy 子映射额外字段检测 ----
    _KNOWN_FEATURE_MISSING_POLICY_KEYS = {
        'source_precedence', 'numeric_fill_value',
        'require_missing_mask', 'missing_semantics_carrier',
    }
    unknown_fmp_keys = [k for k in feature_missing_policy if k not in _KNOWN_FEATURE_MISSING_POLICY_KEYS]
    if unknown_fmp_keys:
        raise ValueError(
            f'feature_missing_policy contains unknown keys: {unknown_fmp_keys}; '
            f'expected only {sorted(_KNOWN_FEATURE_MISSING_POLICY_KEYS)}'
        )

    # ---- 未知顶层字段检测（双向校验的 YAML→schema 方向） ----
    _KNOWN_TOP_LEVEL_KEYS = set(_MODALITY_FIELD_KEYS.values()) | {"feature_missing_policy"}
    unknown_keys = [k for k in cfg if k not in _KNOWN_TOP_LEVEL_KEYS]
    if unknown_keys:
        raise ValueError(
            f'sensor contract contains unknown top-level keys: {unknown_keys}; '
            f'expected only {sorted(_KNOWN_TOP_LEVEL_KEYS)}'
        )

    return {  # 返回规范化后的配置。
        "required_payload_fields": required_payload_fields,  # 各模态必需字段。
        "feature_missing_policy": {  # 特征缺失策略。
            "source_precedence": source_precedence,  # 来源优先级。
            "numeric_fill_value": numeric_fill_value,  # 数值填充值。
            "require_missing_mask": require_missing_mask,  # 是否要求缺失掩码。
            "missing_semantics_carrier": missing_semantics_carrier,  # 缺失语义载体。
        },
    }


@lru_cache(maxsize=1)  # 只缓存一份，因为冻结合同不会变。
def _load_frozen_sensor_contract_snapshot() -> dict[str, Any]:
    """加载仓库拥有的冻结传感器合同快照（只加载一次）。

    从默认路径加载 sensors.yaml，校验其结构后缓存。
    后续所有校验都与此快照做精确比较。

    返回：
        dict[str, Any]: 冻结传感器合同快照。
    """
    return _normalize_sensor_contract_cfg(load_yaml_config(_SENSORS_CONFIG_PATH))


def _validate_frozen_sensor_contract_cfg(cfg: Any) -> dict[str, Any]:
    """校验传感器合同是否与仓库冻结快照完全一致。

    先对输入配置做规范化，然后与冻结快照做精确比较。
    任何漂移都会导致校验失败。

    参数：
        cfg: 待校验的传感器合同配置。

    返回：
        dict[str, Any]: 校验通过的配置。

    异常：
        ValueError: 配置与冻结快照不一致时抛出。
    """
    normalized_cfg = _normalize_sensor_contract_cfg(cfg)  # 先规范化。
    from liquidloc.protocol.scene_axis_protocol import _deep_equal_with_nan_check
    if not _deep_equal_with_nan_check(normalized_cfg, _load_frozen_sensor_contract_snapshot()):  # 与冻结快照精确比较（NaN 安全）。
        raise ValueError("sensor contract must match the frozen repository snapshot exactly")
    return normalized_cfg  # 一致则返回。


def load_sensor_contract(config_path: str | Path | None = None) -> dict[str, Any]:
    """加载传感器合同配置并校验其与冻结快照一致。

    参数：
        config_path: 可选的配置文件路径，默认使用 configs/base/sensors.yaml。

    返回：
        dict[str, Any]: 校验通过的传感器合同配置。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"config_path": str(config_path) if config_path else None}, "load_sensor_contract 入口参数")
    if config_path is not None:
        if not str(config_path).strip():  # 空白路径不允许（同时覆盖 str 和 Path 对象）。
            raise ValueError('sensor contract path must not be blank')
        path = Path(config_path).resolve()  # 规范化路径，消除 .. 等穿越风险。
        if path.suffix.lower() not in ('.yaml', '.yml'):  # 只允许 YAML 文件。
            raise ValueError(f"sensor contract path must be a YAML file, got {path.suffix!r}")
        project_root = find_project_root().resolve()  # 项目根目录的绝对路径。
        if not path.is_relative_to(project_root):  # 路径必须在项目根目录下（is_relative_to 防止兄弟目录前缀绕过）。
            raise ValueError(
                f"sensor contract path must be within project root {project_root}, got {path}"
            )
    else:
        path = _SENSORS_CONFIG_PATH  # 使用默认路径。
    cfg = load_yaml_config(path)  # 加载 YAML 配置。
    return _validate_frozen_sensor_contract_cfg(cfg)  # 校验并返回。


_SENSOR_CONTRACT = load_sensor_contract()  # 模块加载时即冻结传感器合同，后续只读。


def get_required_payload_fields() -> dict[str, tuple[str, ...]]:
    """获取每种模态必需的 payload 字段名元组。

    返回的字典键为模态名（imu/uwb/vio/flow/tof），值为该模态必需的字段名元组。
    每次调用都返回新的元组副本，防止外部修改冻结合同。

    返回：
        dict[str, tuple[str, ...]]: 模态名到必需字段名元组的映射。
    """
    fields = _SENSOR_CONTRACT["required_payload_fields"]  # 从冻结合同中读取。
    return {modality: tuple(field_names) for modality, field_names in fields.items()}  # 返回副本。


def get_feature_missing_policy() -> dict[str, Any]:
    """获取特征缺失策略配置。

    返回的字典包含来源优先级、数值填充值、是否要求缺失掩码和缺失语义载体。
    每次调用都返回新的值副本，防止外部修改冻结合同。

    返回：
        dict[str, Any]: 特征缺失策略配置。
    """
    policy = _SENSOR_CONTRACT["feature_missing_policy"]  # 从冻结合同中读取。
    return {
        "source_precedence": tuple(policy["source_precedence"]),  # 来源优先级元组。
        "numeric_fill_value": float(policy["numeric_fill_value"]),  # 数值填充值。
        "require_missing_mask": bool(policy["require_missing_mask"]),  # 是否要求缺失掩码。
        "missing_semantics_carrier": str(policy["missing_semantics_carrier"]),  # 缺失语义载体。
    }


__all__ = (  # 对外导出列表。
    "get_feature_missing_policy",  # 获取特征缺失策略。
    "get_required_payload_fields",  # 获取必需 payload 字段。
    "load_sensor_contract",  # 加载传感器合同。
)
