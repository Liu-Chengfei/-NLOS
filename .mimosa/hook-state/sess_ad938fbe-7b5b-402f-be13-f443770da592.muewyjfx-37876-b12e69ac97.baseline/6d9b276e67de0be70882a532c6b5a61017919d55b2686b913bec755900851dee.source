"""冻结任务合同辅助模块。

文件职责：
  作为 configs/base/task.yaml 的唯一代码入口，校验运行时任务语义
  是否仍与冻结的论文级合同一致，并暴露只读辅助函数供下游代码使用。
  核心目标是防止任务定义（状态向量、传感器角色、VIO 更新合同）
  在运行时漂移。

本文件绝对不负责：
  不定义任务内容本身（任务内容由 task.yaml 定义）。
  不修改任务配置文件。
  不执行任何定位或滤波逻辑。

核心数据流：
    task.yaml → load_task_contract → 校验通过 → get_task_name / get_state_definition / ...

上游依赖：
  liquidloc.common.config_utils（load_yaml_config 加载 YAML 配置）

下游调用者：
  estimators/（使用 get_state_items 和 get_sensor_roles）、
  fusion/（使用 get_vio_update_contract）、
  models/（使用 get_state_items 构建网络输入/输出）、
  pipelines/（使用 get_task_name 标识实验任务）

输入对象定义：
  - config_path  可选的任务配置文件路径，默认使用 configs/base/task.yaml

输出对象定义：
  - load_task_contract          加载并校验冻结任务合同
  - get_task_name               获取任务名称
  - get_state_definition        获取状态定义
  - get_state_items             获取状态项元组
  - get_sensor_roles            获取传感器角色映射
  - get_vio_update_contract     获取 VIO 更新合同

核心变量定义：
  - _CONFIG_ROOT                配置文件根目录
  - _TASK_CONFIG_PATH           任务配置文件路径
  - _EXPECTED_TASK_NAME         冻结的任务名称
  - _EXPECTED_STATE_FRAME       冻结的状态坐标系
  - _EXPECTED_STATE_ITEMS       冻结的状态项元组
  - _EXPECTED_SENSOR_ROLES      冻结的传感器角色映射
  - _EXPECTED_VIO_UPDATE_CONTRACT  冻结的 VIO 更新合同
  - _TASK_CONTRACT              模块加载时冻结的任务合同快照

关键设计决策：
  - 任务合同在模块加载时即冻结，后续调用只读。
  - 所有字段都做精确匹配校验，不允许静默漂移。
  - VIO 更新合同中 forbid_rewrite_measurement_fields_inside_update 必须为 True，
    防止模型在更新步骤中篡改测量字段。
"""

from __future__ import annotations  # 允许类型注解中引用尚未定义的类型。

from pathlib import Path  # 用于处理配置文件路径。
from typing import Any  # 允许类型注解里表示"任意类型"。

from liquidloc.common.config_utils import find_project_root, load_yaml_config  # 加载 YAML 配置文件和项目根查找器。
from liquidloc.common.constants import VIO_MEASUREMENT_ITEMS  # VIO 测量项冻结常量，common 层为唯一真相源。
from liquidloc.common.constants import STATE_ITEMS  # 状态项冻结常量，common 层为唯一真相源。
from liquidloc.common.validation import is_bool_like, is_string_like  # 统一布尔类型校验函数、统一字符串类型校验函数。

_CONFIG_ROOT = (find_project_root() / "configs" / "base").resolve()  # 配置文件根目录，resolve() 在模块加载时完成。
_TASK_CONFIG_PATH = _CONFIG_ROOT / "task.yaml"  # 任务配置文件路径。

_EXPECTED_TASK_NAME = "uwb_imu_vio_localization"  # 冻结的任务名称：UWB+IMU+VIO 融合定位。
_EXPECTED_STATE_FRAME = "planar_xy_yaw"  # 冻结的状态坐标系：平面 XY+偏航角。
_EXPECTED_STATE_ITEMS = STATE_ITEMS  # 冻结的状态项：位置、速度、偏航角、加速度偏置、陀螺仪偏置。来自 common 层唯一真相源。
_EXPECTED_SENSOR_ROLES = {  # 冻结的传感器角色映射。
    "uwb": "absolute_range_constraint",  # UWB 提供绝对距离约束。
    "imu": "high_rate_propagation",  # IMU 提供高频传播。
    "vio": "relative_pose_constraint",  # VIO 提供相对位姿约束。
}
_EXPECTED_VIO_UPDATE_CONTRACT = {  # 冻结的 VIO 更新合同。
    "measurement_items": VIO_MEASUREMENT_ITEMS,  # VIO 测量项：相对位移和偏航变化，来自 common 层冻结常量。
    # 紧耦合扩维：VIO 直接更新位姿分量 px/py/yaw + uwb_clock_bias/vio_scale。
    "updated_state_items": ("px", "py", "yaw", "uwb_clock_bias", "vio_scale"),
    "learned_control_entry": "noise_multiplier",  # 学习控制入口：bridge 收口后的噪声倍数。
    "forbid_rewrite_measurement_fields_inside_update": True,  # 禁止在更新步骤中篡改测量字段。
}


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """要求值必须是映射类型并返回其字典副本。

    参数：
        value: 待校验的值。
        name: 字段名，用于构造错误信息。

    返回：
        dict[str, Any]: 值的字典副本。

    异常：
        TypeError: 值不是映射类型时抛出。
    """
    if not isinstance(value, dict):  # 必须是字典。
        raise TypeError(f"{name} must be a mapping")
    return dict(value)  # 返回副本。


def _require_exact_string(value: Any, *, name: str, expected: str) -> str:
    """要求值必须精确匹配预期字符串。

    参数：
        value: 待校验的值。
        name: 字段名，用于构造错误信息。
        expected: 预期字符串。

    返回：
        str: 校验通过的字符串。

    异常：
        TypeError: 值不是非空字符串时抛出。
        ValueError: 值与预期不匹配时抛出。
    """
    if not is_string_like(value) or not value:  # 必须是非空字符串。
        raise TypeError(f"{name} must be a non-empty string")
    if value != expected:  # 必须精确匹配。
        raise ValueError(f"{name} must be {expected!r}, got {value!r}")
    return value


def _require_exact_string_list(
    value: Any,
    *,
    name: str,
    expected: tuple[str, ...],
) -> tuple[str, ...]:
    """要求值必须精确匹配预期字符串列表。

    逐项校验，不允许重复，最终与预期列表做精确比较。

    参数：
        value: 待校验的值。
        name: 字段名，用于构造错误信息。
        expected: 预期字符串元组。

    返回：
        tuple[str, ...]: 校验通过的字符串元组。

    异常：
        TypeError: 值不是非空列表或包含非字符串项时抛出。
        ValueError: 列表包含重复项或与预期不匹配时抛出。
    """
    if not isinstance(value, list) or not value:  # 必须是非空列表。
        raise TypeError(f"{name} must be a non-empty list")
    items: list[str] = []  # 存放校验后的项。
    for index, item in enumerate(value):  # 逐项检查。
        if not is_string_like(item) or not item:  # 每项必须是非空字符串。
            raise TypeError(f"{name}[{index}] must be a non-empty string")
        items.append(item)
    parsed = tuple(items)  # 转为元组。
    if len(set(parsed)) != len(parsed):  # 不允许重复项。
        raise ValueError(f"{name} must not contain duplicate entries")
    if parsed != expected:  # 必须与预期精确匹配。
        raise ValueError(f"{name} must be {list(expected)!r}, got {list(parsed)!r}")
    return parsed


def load_task_contract(config_path: str | Path | None = None) -> dict[str, Any]:
    """加载冻结任务合同并校验其与冻结合同一致。

    逐字段校验任务名称、状态定义、传感器角色和 VIO 更新合同，
    确保运行时配置与冻结协议完全一致。

    参数：
        config_path: 可选的配置文件路径，默认使用 configs/base/task.yaml。

    返回：
        dict[str, Any]: 校验通过的任务合同配置。

    异常：
        TypeError: 字段类型不正确时抛出。
        ValueError: 字段值与冻结合同不一致时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"config_path": str(config_path) if config_path else None}, "load_task_contract 入口参数")
    if config_path is not None:
        if not str(config_path).strip():  # 空白路径不允许（同时覆盖 str 和 Path 对象）。
            raise ValueError('task contract path must not be blank')
        path = Path(config_path).resolve()  # 规范化路径，消除 .. 等穿越风险。
        if path.suffix.lower() not in ('.yaml', '.yml'):  # 只允许 YAML 文件。
            raise ValueError(f"task contract path must be a YAML file, got {path.suffix!r}")
        project_root = find_project_root().resolve()  # 项目根目录的绝对路径。
        if not path.is_relative_to(project_root):  # 路径必须在项目根目录下（is_relative_to 防止兄弟目录前缀绕过）。
            raise ValueError(
                f"task contract path must be within project root {project_root}, got {path}"
            )
    else:
        path = _TASK_CONFIG_PATH  # 使用默认路径。
    cfg = load_yaml_config(path)  # 加载 YAML 配置。

    task_name = _require_exact_string(  # 校验任务名称。
        cfg.get("task_name"),
        name="task_name",
        expected=_EXPECTED_TASK_NAME,
    )

    state_definition = _require_mapping(cfg.get("state_definition"), name="state_definition")  # 校验状态定义段。
    frame = _require_exact_string(  # 校验状态坐标系。
        state_definition.get("frame"),
        name="state_definition.frame",
        expected=_EXPECTED_STATE_FRAME,
    )
    state_items = _require_exact_string_list(  # 校验状态项列表。
        state_definition.get("state_items"),
        name="state_definition.state_items",
        expected=_EXPECTED_STATE_ITEMS,
    )

    sensor_roles = _require_mapping(cfg.get("sensor_roles"), name="sensor_roles")  # 校验传感器角色段。
    normalized_sensor_roles = {  # 逐个校验每个传感器的角色。
        modality: _require_exact_string(
            sensor_roles.get(modality),
            name=f"sensor_roles.{modality}",
            expected=expected_role,
        )
        for modality, expected_role in _EXPECTED_SENSOR_ROLES.items()  # 遍历所有预期角色。
    }

    vio_update_contract = _require_mapping(  # 校验 VIO 更新合同段。
        cfg.get("vio_update_contract"),
        name="vio_update_contract",
    )
    measurement_items = _require_exact_string_list(  # 校验 VIO 测量项。
        vio_update_contract.get("measurement_items"),
        name="vio_update_contract.measurement_items",
        expected=_EXPECTED_VIO_UPDATE_CONTRACT["measurement_items"],
    )
    updated_state_items = _require_exact_string_list(  # 校验 VIO 更新状态项。
        vio_update_contract.get("updated_state_items"),
        name="vio_update_contract.updated_state_items",
        expected=_EXPECTED_VIO_UPDATE_CONTRACT["updated_state_items"],
    )
    learned_control_entry = _require_exact_string(  # 校验学习控制入口。
        vio_update_contract.get("learned_control_entry"),
        name="vio_update_contract.learned_control_entry",
        expected=_EXPECTED_VIO_UPDATE_CONTRACT["learned_control_entry"],
    )
    forbid_rewrite = vio_update_contract.get("forbid_rewrite_measurement_fields_inside_update")  # 读取禁止重写标志。
    if not is_bool_like(forbid_rewrite):  # 必须是布尔值（含 numpy.bool_）。
        raise TypeError("vio_update_contract.forbid_rewrite_measurement_fields_inside_update must be a bool")
    forbid_rewrite = bool(forbid_rewrite)  # 显式转为 Python bool，避免 numpy.bool_ 在后续 is 比较中误判。
    if forbid_rewrite is not _EXPECTED_VIO_UPDATE_CONTRACT["forbid_rewrite_measurement_fields_inside_update"]:  # 必须为 True，使用 is 而非 == 确保精确匹配。
        raise ValueError(
            "vio_update_contract.forbid_rewrite_measurement_fields_inside_update "
            f"must be {_EXPECTED_VIO_UPDATE_CONTRACT['forbid_rewrite_measurement_fields_inside_update']!r}"
        )

    # ---- 子映射额外字段检测（防止 YAML 中注入未声明的子键） ----
    _KNOWN_STATE_DEFINITION_KEYS = {'frame', 'state_items'}
    unknown_sd_keys = [k for k in state_definition if k not in _KNOWN_STATE_DEFINITION_KEYS]
    if unknown_sd_keys:
        raise ValueError(
            f'state_definition contains unknown keys: {unknown_sd_keys}; '
            f'expected only {sorted(_KNOWN_STATE_DEFINITION_KEYS)}'
        )

    _KNOWN_SENSOR_ROLES_KEYS = set(_EXPECTED_SENSOR_ROLES.keys())
    unknown_sr_keys = [k for k in sensor_roles if k not in _KNOWN_SENSOR_ROLES_KEYS]
    if unknown_sr_keys:
        raise ValueError(
            f'sensor_roles contains unknown keys: {unknown_sr_keys}; '
            f'expected only {sorted(_KNOWN_SENSOR_ROLES_KEYS)}'
        )

    _KNOWN_VIO_UPDATE_CONTRACT_KEYS = {
        'measurement_items', 'updated_state_items',
        'learned_control_entry', 'forbid_rewrite_measurement_fields_inside_update',
    }
    unknown_vuc_keys = [k for k in vio_update_contract if k not in _KNOWN_VIO_UPDATE_CONTRACT_KEYS]
    if unknown_vuc_keys:
        raise ValueError(
            f'vio_update_contract contains unknown keys: {unknown_vuc_keys}; '
            f'expected only {sorted(_KNOWN_VIO_UPDATE_CONTRACT_KEYS)}'
        )

    # ---- 未知顶层字段检测（双向校验的 YAML→schema 方向） ----
    _KNOWN_TOP_LEVEL_KEYS = {
        'task_name', 'state_definition', 'sensor_roles', 'vio_update_contract',
    }
    unknown_keys = [k for k in cfg if k not in _KNOWN_TOP_LEVEL_KEYS]
    if unknown_keys:
        raise ValueError(
            f'task contract contains unknown top-level keys: {unknown_keys}; '
            f'expected only {sorted(_KNOWN_TOP_LEVEL_KEYS)}'
        )

    return {  # 返回校验通过的任务合同。
        "task_name": task_name,  # 任务名称。
        "state_definition": {  # 状态定义。
            "frame": frame,  # 坐标系。
            "state_items": state_items,  # 状态项。
        },
        "sensor_roles": normalized_sensor_roles,  # 传感器角色。
        "vio_update_contract": {  # VIO 更新合同。
            "measurement_items": measurement_items,  # 测量项。
            "updated_state_items": updated_state_items,  # 更新状态项。
            "learned_control_entry": learned_control_entry,  # 学习控制入口。
            "forbid_rewrite_measurement_fields_inside_update": forbid_rewrite,  # 禁止重写标志。
        },
    }


_TASK_CONTRACT = load_task_contract()  # 模块加载时即冻结任务合同，后续只读。


def get_task_name() -> str:
    """获取冻结的任务名称。

    返回：
        str: 任务名称字符串。
    """
    return str(_TASK_CONTRACT["task_name"])


def get_state_definition() -> dict[str, Any]:
    """获取冻结的状态定义。

    返回：
        dict[str, Any]: 包含 frame 和 state_items 的状态定义字典。
    """
    state_definition = _TASK_CONTRACT["state_definition"]
    return {
        "frame": str(state_definition["frame"]),  # 坐标系名称。
        "state_items": tuple(state_definition["state_items"]),  # 状态项元组。
    }


def get_state_items() -> tuple[str, ...]:
    """获取冻结的状态项元组。

    返回：
        tuple[str, ...]: 状态项名称元组。
    """
    return tuple(_TASK_CONTRACT["state_definition"]["state_items"])


def get_sensor_roles() -> dict[str, str]:
    """获取冻结的传感器角色映射。

    返回：
        dict[str, str]: 传感器名到角色名的映射。
    """
    sensor_roles = _TASK_CONTRACT["sensor_roles"]
    return {modality: str(role_name) for modality, role_name in sensor_roles.items()}


def get_vio_update_contract() -> dict[str, Any]:
    """获取冻结的 VIO 更新合同。

    返回：
        dict[str, Any]: VIO 更新合同字典。
    """
    vio_update_contract = _TASK_CONTRACT["vio_update_contract"]
    return {
        "measurement_items": tuple(vio_update_contract["measurement_items"]),  # 测量项元组。
        "updated_state_items": tuple(vio_update_contract["updated_state_items"]),  # 更新状态项元组。
        "learned_control_entry": str(vio_update_contract["learned_control_entry"]),  # 学习控制入口。
        "forbid_rewrite_measurement_fields_inside_update": bool(  # 禁止重写标志。
            vio_update_contract["forbid_rewrite_measurement_fields_inside_update"]
        ),
    }


__all__ = (  # 对外导出列表。
    "get_sensor_roles",  # 获取传感器角色。
    "get_state_definition",  # 获取状态定义。
    "get_state_items",  # 获取状态项。
    "get_task_name",  # 获取任务名称。
    "get_vio_update_contract",  # 获取 VIO 更新合同。
    "load_task_contract",  # 加载任务合同。
)
