"""实验协议闸门模块——训练、评测与公开基准的冻结校验入口。

文件职责：
  加载并校验冻结实验协议（configs/base/experiment_protocol.yaml），
  为训练请求、评测请求和公开基准请求提供闸门校验。
  所有闸门函数都只做"协议一致性检查"，不执行任何训练或评测逻辑。
  核心目标是防止运行时配置漂移，确保实验可复现。

本文件绝对不负责：
  不执行训练或评测。
  不修改协议文件。
  不定义协议内容本身（协议内容由 YAML 文件定义）。

核心数据流：
    experiment_protocol.yaml → load_experiment_protocol → 校验通过 → 各 validate_*_request 闸门

上游依赖：
  liquidloc.common.config_utils（load_yaml_config 加载 YAML 配置）、
  liquidloc.common.validation（类型校验工具）

下游调用者：
  pipelines/（流水线在启动训练/评测前调用闸门校验）、
  scenarios/（场景脚本校验请求合法性）、
  smoke check 脚本和审计脚本

输入对象定义：
  - cfg           请求配置字典，由上层调用者构造
  - protocol_cfg  可选的外部协议配置，默认从冻结 YAML 加载

输出对象定义：
  - load_experiment_protocol                   加载并校验冻结实验协议
  - normalize_train_request                     规范化训练请求
  - normalize_eval_request                      规范化评测请求
  - normalize_public_benchmark_request          规范化公开基准请求
  - normalize_run_mode                         规范化运行模式
  - get_default_failure_threshold_m            获取默认失效阈值
  - get_public_benchmark_allowed_datasets      获取公开基准允许的数据集

核心变量定义：
  - _DEFAULT_PATH                              冻结实验协议 YAML 的默认路径
  - _FAILURE_THRESHOLD_MAX_M                   失效阈值上限（10.0 米）
  - _EXPERIMENT_PROTOCOL_VERSION               实验协议版本号（2）
  - _QUICK_FULL_RULE                           quick/full 语义规则标识
  - _FAILURE_SAMPLE_POLICY                     失败样本保留策略
  - _AGGREGATION_ORDER                         聚合顺序
  - _CONCLUSION_PRIORITY                       结论优先级
  - _PUBLIC_BENCHMARK_ALLOWED_DATASETS         公开基准允许的数据集
  - _PUBLIC_BENCHMARK_FROZEN_EVAL_SPLIT        公开基准冻结评测分割名
  - _TRAINING_ALLOWED_SPLIT_ROLES              训练允许的数据分割角色
  - _TRAINING_FORBIDDEN_SPLIT_ROLES            训练禁止的数据分割角色
  - _DEFAULT_FAILURE_THRESHOLD_M               默认失效阈值（1.0 米）
  - _TUNING_ENABLED_STRINGS                    表示"启用调参"的字符串集合
  - _TUNING_DISABLED_STRINGS                   表示"禁用调参"的字符串集合

关键设计决策：
  - quick/full 只允许作为运行规模标签，不得写成新的科学语义。
  - 公开基准必须使用冻结的评测分割，禁止调参。
  - 训练只能使用 train/val 分割，禁止使用 test/external/frozen_public_eval。
  - 所有协议字段都做精确匹配校验，不允许静默漂移。
  - 布尔标志只接受显式 True/False，不做 truthiness 强转。
"""

from __future__ import annotations  # 允许类型注解中引用尚未定义的类型。

import math  # 用于 isfinite 检查数值有限性。
from copy import deepcopy
from functools import lru_cache
from collections.abc import Mapping  # 用于类型检查映射类型。
from collections.abc import Sequence  # 用于类型检查序列类型（与 Mapping 同源）。
from pathlib import Path  # 用于处理协议文件路径。
from typing import Any  # 允许类型注解里表示"任意类型"。

from liquidloc.common.validation import is_bool_like  # 检查值是否为布尔类型（含 numpy.bool_）。
from liquidloc.common.validation import is_integer  # 检查值是否为整数类型（排除 bool 和 numpy.bool_）。
from liquidloc.common.validation import is_real  # 检查值是否为实数类型（排除 bool 和 numpy.bool_）。
from liquidloc.common.validation import is_string_like  # 检查值是否为字符串类型（含 numpy.str_）。
from liquidloc.common.config_utils import find_project_root, load_yaml_config  # 加载 YAML 配置文件和项目根查找器。
from liquidloc.common.constants import normalize_run_mode as _normalize_run_mode_common  # 纯字符串级运行模式归一化。
from liquidloc.protocol.version import PROTOCOL_VERSION as _EXPERIMENT_PROTOCOL_VERSION  # 从集中版本模块导入协议版本号。

# 冻结实验协议 YAML 的默认路径，位于 configs/base/experiment_protocol.yaml。
_DEFAULT_PATH = (find_project_root() / 'configs' / 'base' / 'experiment_protocol.yaml').resolve()  # resolve() 在模块加载时完成，与 metric_schema 等同层模块风格一致。
_FAILURE_THRESHOLD_MAX_M: float = 10.0  # 失效阈值上限（米），室内定位场景合理上限。
_QUICK_FULL_RULE: str = 'quick_smoke_scale__full_real_execution_required'  # quick/full 语义规则：quick 是冒烟规模，full 是真实执行。
_FAILURE_SAMPLE_POLICY: str = 'retain_and_audit'  # 失败样本策略：保留并审计，不允许静默丢弃。
_AGGREGATION_ORDER: tuple[str, ...] = (  # 实验结果的聚合顺序，从内到外逐层汇总。
    'single_run',  # 单次运行。
    'repeat_summary',  # 重复运行汇总。
    'scene_summary',  # 场景汇总。
    'experiment_conclusion',  # 实验结论。
)
_CONCLUSION_PRIORITY: tuple[str, ...] = ('p95', 'failure_rate', 'rmse', 'mae')  # 结论优先级：先看 p95 和失败率，再看 RMSE 和 MAE。
_PUBLIC_BENCHMARK_ALLOWED_DATASETS: tuple[str, ...] = ('miluv', 'ntu_viral')  # 公开基准只允许这两个数据集，防止随意扩展。
_PUBLIC_BENCHMARK_FROZEN_EVAL_SPLIT: str = 'frozen_public_eval'  # 公开基准的冻结评测分割名，所有公开基准必须使用此分割。
_TRAINING_ALLOWED_SPLIT_ROLES: tuple[str, ...] = ('train', 'val')  # 训练只允许使用 train 和 val 分割。
_TRAINING_FORBIDDEN_SPLIT_ROLES: tuple[str, ...] = ('test', 'external', 'frozen_public_eval')  # 训练禁止使用 test、external 和冻结公开评测分割。
_DEFAULT_FAILURE_THRESHOLD_M: float = 1.0  # 默认失效阈值（米），定位误差超过此值视为失效。
_TUNING_ENABLED_STRINGS: frozenset[str] = frozenset({'1', 'on', 'true', 'yes'})  # 表示"启用调参"的字符串集合。
_TUNING_DISABLED_STRINGS: frozenset[str] = frozenset({'', '0', 'false', 'no', 'off'})  # 表示"禁用调参"的字符串集合。


def normalize_public_dataset_name(dataset_name: str) -> str:
    """将公开数据集名称标准化为注册表键名形式（小写、去空白）。

    此函数原位于 dataio.registry.public_dataset_registry，因 protocol 层
    不应反向依赖 dataio 层，已迁入本模块作为协议级名称规范化入口。

    参数：
        dataset_name: 原始数据集名称字符串。

    返回：
        标准化后的数据集名称（小写、去首尾空白）。

    异常：
        ValueError: 输入为空或仅含空白时抛出。
        TypeError: 输入为 None 或非字符串类型时抛出。
    """
    if dataset_name is None:
        raise TypeError("dataset_name must be a string")

    if not is_string_like(dataset_name):
        raise TypeError("dataset_name must be a string")

    normalized_name = str(dataset_name).strip().lower()
    if not normalized_name:
        raise ValueError("dataset_name must be a non-empty string")

    return normalized_name


def _validate_failure_threshold_value(value: Any, *, field_name: str) -> float:
    """校验失效阈值标量是否符合冻结协议合同。

    要求阈值必须是有限正数，且不超过上限 _FAILURE_THRESHOLD_MAX_M。

    参数：
        value: 待校验的阈值，可以是任意类型，内部会尝试转为 float。
        field_name: 字段名，用于构造错误信息。

    返回：
        float: 校验通过的阈值。

    异常：
        TypeError: 值不是数值型（包括 bool）时抛出。
        ValueError: 值非有限、非正或超过上限时抛出。
    """
    if is_bool_like(value):  # bool 虽然是 int 的子类，但这里不允许。
        raise TypeError(f'{field_name} must be a real number')
    if not is_real(value):  # 只接受数值型输入，拒绝字符串等可转 float 的非数值类型。
        raise TypeError(f'{field_name} must be a real number')
    threshold = float(value)  # 已确认是数值型，安全转为浮点数。
    if not math.isfinite(threshold):  # 拒绝 NaN 和 inf。
        raise ValueError(f'{field_name} must be finite')
    if threshold <= 0:  # 阈值必须为正数。
        raise ValueError(f'{field_name} must be positive')
    if threshold > _FAILURE_THRESHOLD_MAX_M:  # 超过上限说明配置不合理。
        raise ValueError(
            f'{field_name} must not exceed {_FAILURE_THRESHOLD_MAX_M}m, got {threshold}'
        )
    return threshold  # 返回校验通过的阈值。


def _is_enabled_tuning_value(value: Any) -> bool:
    """保守地将调参标志解释为启用/禁用。

    对于字符串，只认 _TUNING_ENABLED_STRINGS 和 _TUNING_DISABLED_STRINGS
    中的值；其他字符串保守视为禁用。对于数值，零为禁用，非零为启用。
    NaN/Inf 保守视为禁用。对于布尔值，直接返回。

    参数：
        value: 调参标志的原始值。

    返回：
        bool: True 表示启用调参，False 表示禁用。
    """
    if is_bool_like(value):  # 布尔值直接返回。
        return bool(value)
    if is_string_like(value):  # 字符串走保守解释（含 numpy.str_）。
        normalized = str(value).strip().lower()  # 统一小写并去空白。
        if normalized in _TUNING_DISABLED_STRINGS:  # 明确禁用词。
            return False
        if normalized in _TUNING_ENABLED_STRINGS:  # 明确启用词。
            return True
        return False  # 其他字符串保守视为禁用，避免意外启用调参。
    if is_real(value):  # 实数值类型（已排除 bool 和 complex）。
        if not math.isfinite(value):  # NaN/Inf 保守视为禁用，防止非有限值穿透为启用。
            return False
        return value != 0  # 零为禁用，非零为启用。
    return False  # 其他类型保守视为禁用，避免 truthiness 漂移意外启用调参。


def _coerce_non_string_list_field(value: Any, *, field_name: str) -> list[Any]:
    """将列表类协议字段规范化为列表，同时拒绝意外的字符串输入。

    字符串也是可迭代的，但协议字段期望的是稳定顺序的列表/元组，
    而不是字符串、映射、set 或 generator 这类顺序不稳定/一次性消费对象。

    参数：
        value: 待规范化的字段值。
        field_name: 字段名，用于构造错误信息。

    返回：
        list[Any]: 规范化后的列表。None 输入返回空列表。

    异常：
        TypeError: 值不是列表/元组时抛出。
    """
    if value is None:  # None 视为空列表。
        return []
    if is_string_like(value):  # 字符串（含 numpy.str_）单独拒绝，避免被逐字符拆分。
        raise TypeError(f'{field_name} must be a list or tuple, not a string')
    if not isinstance(value, (list, tuple)):  # 顶层必须是稳定顺序的序列。
        raise TypeError(f'{field_name} must be a list or tuple')
    return list(value)  # 只对稳定顺序序列做浅拷贝。


def _coerce_boolean_flag(value: Any, *, field_name: str) -> bool:
    """只接受显式布尔值作为协议拥有的标志。

    不做 truthiness 强转，避免 0/1/"" 等被误解释为布尔值。

    参数：
        value: 待校验的标志值。
        field_name: 字段名，用于构造错误信息。

    返回：
        bool: 校验通过的布尔值。None 返回 False。

    异常：
        TypeError: 值不是布尔类型时抛出。
    """
    if value is None:  # None 默认为 False。
        return False
    if is_bool_like(value):  # 只接受显式布尔值（含 numpy.bool_）。
        return bool(value)  # 显式转为 Python bool，避免 numpy.bool_ 泄漏。
    raise TypeError(f'{field_name} must be a boolean, got {type(value).__name__}')  # 其他类型一律拒绝。


def _read_protocol_boolean_flag(
    section: Mapping[str, Any],
    *,
    key: str,
    default: bool,
    field_name: str,
) -> bool:
    """从冻结协议配置中读取布尔字段，不做 truthiness 强转。

    与 _coerce_boolean_flag 配合使用，确保协议布尔字段不会被
    truthiness 规则意外解释。

    参数：
        section: 协议配置的某个子段。
        key: 要读取的键名。
        default: 键不存在时的默认值。
        field_name: 字段名，用于构造错误信息。

    返回：
        bool: 读取并校验后的布尔值。
    """
    if key not in section:  # 键不存在时使用默认值。
        return default
    raw_value = section[key]  # 键存在时读取原始值（可能为 None）。
    if raw_value is None:  # 显式 None 视同键不存在，回退到 default。
        return default
    return _coerce_boolean_flag(raw_value, field_name=field_name)  # 严格校验布尔类型。


def _normalize_non_empty_string_items(value: Any, *, field_name: str) -> list[str]:
    """规范化列表类字符串字段，拒绝空白项并去重。

    参数：
        value: 待规范化的字段值。
        field_name: 字段名，用于构造错误信息。

    返回：
        list[str]: 规范化后的字符串列表，无空白项、无重复。

    异常：
        TypeError: 值不是列表类或包含非字符串项时抛出。
        ValueError: 包含空白字符串项时抛出。
    """
    items = _coerce_non_string_list_field(value, field_name=field_name)  # 先确保是列表。
    normalized_items: list[str] = []  # 存放规范化后的项。
    seen: set[str] = set()  # 用于去重。
    for item in items:  # 逐项检查。
        if not is_string_like(item):  # 每项必须是字符串（含 numpy.str_）。
            raise TypeError(f'{field_name} items must be strings, got {type(item).__name__}')
        normalized_item = str(item).strip()  # 去除首尾空白，兼容 numpy.str_。
        if not normalized_item:  # 空白字符串不允许。
            raise ValueError(f'{field_name} items must not be blank')
        if normalized_item in seen:  # 重复项跳过。
            continue
        seen.add(normalized_item)  # 记录已见项。
        normalized_items.append(normalized_item)  # 加入结果列表。
    return normalized_items


def _normalize_optional_path_string(value: Any, *, field_name: str) -> str | None:
    """规范化可选的路径类字符串，拒绝空白字符串。

    参数：
        value: 待规范化的值，可以是 None、字符串或其他可转字符串的类型。
        field_name: 字段名，用于构造错误信息。

    返回：
        str | None: 规范化后的路径字符串，或 None。

    异常：
        ValueError: 字符串为空白时抛出。
    """
    if value is None:  # None 直接返回。
        return None
    if is_string_like(value):  # 字符串类（含 numpy.str_）走规范化路径。
        normalized_value = str(value).strip()  # 显式转 Python str 再去首尾空白。
        if not normalized_value:  # 空白字符串不允许。
            raise ValueError(f'{field_name} must not be blank')
        return normalized_value
    converted = str(value).strip()  # 非字符串类转为字符串并去首尾空白，与字符串类分支行为一致。
    if not converted:  # 空白字符串同样不允许，防止 __str__ 返回空白绕过合同。
        raise ValueError(f'{field_name} must not be blank')
    return converted


def _normalize_optional_non_empty_string_items(value: Any, *, field_name: str) -> list[str] | None:
    """规范化可选的列表类字符串字段，保留显式缺失合同。

    None 输入返回 None（表示"未提供"），而非空列表（表示"提供了空列表"），
    这两种语义在协议校验中有不同含义。

    参数：
        value: 待规范化的值。
        field_name: 字段名，用于构造错误信息。

    返回：
        list[str] | None: 规范化后的字符串列表，或 None。
    """
    if value is None:  # None 保留为 None，表示"未提供"。
        return None
    return _normalize_non_empty_string_items(value, field_name=field_name)  # 非空时走常规规范化。


def _validate_frozen_experiment_protocol_cfg(cfg: Any) -> dict[str, Any]:
    """校验冻结实验协议的每一个字段是否与仓库合同一致。

    逐字段精确匹配，任何漂移都会导致校验失败。这是防止运行时
    配置与冻结协议不一致的核心防线。

    参数：
        cfg: 待校验的协议配置，必须是映射类型。

    返回：
        dict[str, Any]: 校验通过的协议配置字典。

    异常：
        TypeError: 配置不是映射类型时抛出。
        ValueError: 任何字段与冻结合同不一致时抛出。
    """
    if not isinstance(cfg, Mapping):  # 协议配置必须是映射。
        raise TypeError('experiment protocol config must be a mapping')
    normalized_cfg = deepcopy(cfg)  # 深拷贝，防止返回值与调用者输入共享嵌套引用。

    # ---- 顶层冻结字段校验 ----
    protocol_version = normalized_cfg.get('protocol_version')  # 读取协议版本。
    if not is_integer(protocol_version):  # 协议版本必须是整数（排除 bool 和浮点数）。
        raise TypeError(f'experiment protocol_version must be an integer, got {type(protocol_version).__name__}')
    if protocol_version != _EXPERIMENT_PROTOCOL_VERSION:  # 协议版本必须匹配。
        raise ValueError(f'experiment protocol version must be {_EXPERIMENT_PROTOCOL_VERSION}')
    quick_full_rule = normalized_cfg.get('quick_full_rule')
    if not is_string_like(quick_full_rule):
        raise TypeError(f'experiment quick_full_rule must be a string, got {type(quick_full_rule).__name__}')
    if quick_full_rule != _QUICK_FULL_RULE:  # quick/full 规则必须匹配。
        raise ValueError(f'experiment quick_full_rule must be {_QUICK_FULL_RULE}')
    failure_sample_policy = normalized_cfg.get('failure_sample_policy')
    if not is_string_like(failure_sample_policy):
        raise TypeError(f'experiment failure_sample_policy must be a string, got {type(failure_sample_policy).__name__}')
    if failure_sample_policy != _FAILURE_SAMPLE_POLICY:  # 失败样本策略必须匹配。
        raise ValueError(f'experiment failure_sample_policy must be {_FAILURE_SAMPLE_POLICY}')
    aggregation_order = normalized_cfg.get('aggregation_order')
    if aggregation_order is not None and is_string_like(aggregation_order):
        raise TypeError('experiment aggregation_order must be a list or tuple, not a string')
    if tuple(aggregation_order or ()) != _AGGREGATION_ORDER:  # 聚合顺序必须匹配。
        raise ValueError(f'experiment aggregation_order must be {list(_AGGREGATION_ORDER)}')
    conclusion_priority = normalized_cfg.get('conclusion_priority')
    if conclusion_priority is not None and is_string_like(conclusion_priority):
        raise TypeError('experiment conclusion_priority must be a list or tuple, not a string')
    if tuple(conclusion_priority or ()) != _CONCLUSION_PRIORITY:  # 结论优先级必须匹配。
        raise ValueError(f'experiment conclusion_priority must be {list(_CONCLUSION_PRIORITY)}')

    # ---- 公开基准段校验 ----
    public = dict(normalized_cfg.get('public_benchmark') or {})  # 读取公开基准段。
    allowed_datasets = tuple(  # 校验允许的数据集列表。
        _normalize_non_empty_string_items(
            public.get('allowed_datasets'),
            field_name='public_benchmark.allowed_datasets',
        )
    )
    if allowed_datasets != _PUBLIC_BENCHMARK_ALLOWED_DATASETS:  # 必须精确匹配冻结列表。
        raise ValueError(
            f'experiment public_benchmark.allowed_datasets must be {list(_PUBLIC_BENCHMARK_ALLOWED_DATASETS)}'
        )
    frozen_eval_split = _normalize_optional_path_string(  # 校验冻结评测分割名。
        public.get('frozen_eval_split'),
        field_name='public_benchmark.frozen_eval_split',
    )
    if frozen_eval_split != _PUBLIC_BENCHMARK_FROZEN_EVAL_SPLIT:  # 必须精确匹配。
        raise ValueError(
            f'experiment public_benchmark.frozen_eval_split must be {_PUBLIC_BENCHMARK_FROZEN_EVAL_SPLIT}'
        )
    if _read_protocol_boolean_flag(  # tuning_forbidden 必须为 True。
        public,
        key='tuning_forbidden',
        default=False,
        field_name='public_benchmark.tuning_forbidden',
    ) is not True:
        raise ValueError('experiment public_benchmark.tuning_forbidden must be true')
    if _read_protocol_boolean_flag(  # shared_split_required 必须为 True。
        public,
        key='shared_split_required',
        default=False,
        field_name='public_benchmark.shared_split_required',
    ) is not True:
        raise ValueError('experiment public_benchmark.shared_split_required must be true')

    # ---- 训练段校验 ----
    training = dict(normalized_cfg.get('training') or {})  # 读取训练段。
    allowed_split_roles = tuple(  # 校验允许的分割角色。
        _normalize_non_empty_string_items(
            training.get('allowed_split_roles'),
            field_name='training.allowed_split_roles',
        )
    )
    if allowed_split_roles != _TRAINING_ALLOWED_SPLIT_ROLES:  # 必须精确匹配。
        raise ValueError(
            f'experiment training.allowed_split_roles must be {list(_TRAINING_ALLOWED_SPLIT_ROLES)}'
        )
    forbidden_split_roles = tuple(  # 校验禁止的分割角色。
        _normalize_non_empty_string_items(
            training.get('forbidden_split_roles'),
            field_name='training.forbidden_split_roles',
        )
    )
    if forbidden_split_roles != _TRAINING_FORBIDDEN_SPLIT_ROLES:  # 必须精确匹配。
        raise ValueError(
            f'experiment training.forbidden_split_roles must be {list(_TRAINING_FORBIDDEN_SPLIT_ROLES)}'
        )

    # ---- 评测段校验 ----
    evaluation = dict(normalized_cfg.get('evaluation') or {})  # 读取评测段。
    if _read_protocol_boolean_flag(  # require_prediction_bundles 必须为 True。
        evaluation,
        key='require_prediction_bundles',
        default=False,
        field_name='evaluation.require_prediction_bundles',
    ) is not True:
        raise ValueError('experiment evaluation.require_prediction_bundles must be true')
    default_failure_threshold = _validate_failure_threshold_value(  # 校验默认失效阈值。
        evaluation.get('default_failure_threshold_m', _DEFAULT_FAILURE_THRESHOLD_M),
        field_name='evaluation.default_failure_threshold_m',
    )
    if default_failure_threshold != _DEFAULT_FAILURE_THRESHOLD_M:  # 阈值必须精确匹配，不允许任何偏差。
        raise ValueError(
            f'experiment evaluation.default_failure_threshold_m must be {_DEFAULT_FAILURE_THRESHOLD_M}'
        )
    failure_threshold_max_m = _validate_failure_threshold_value(  # 校验失效阈值上限。
        evaluation.get('failure_threshold_max_m', _FAILURE_THRESHOLD_MAX_M),
        field_name='evaluation.failure_threshold_max_m',
    )
    if failure_threshold_max_m != _FAILURE_THRESHOLD_MAX_M:  # 上限必须精确匹配。
        raise ValueError(
            f'experiment evaluation.failure_threshold_max_m must be {_FAILURE_THRESHOLD_MAX_M}'
        )
    if _read_protocol_boolean_flag(  # require_ground_truth_unless_smoke 必须为 True。
        evaluation,
        key='require_ground_truth_unless_smoke',
        default=False,
        field_name='evaluation.require_ground_truth_unless_smoke',
    ) is not True:
        raise ValueError('experiment evaluation.require_ground_truth_unless_smoke must be true')

    # ---- 未知顶层字段检测（双向校验的 YAML→schema 方向） ----
    _KNOWN_TOP_LEVEL_KEYS = frozenset({
        'protocol_version', 'quick_full_rule', 'failure_sample_policy',
        'aggregation_order', 'conclusion_priority', 'public_benchmark',
        'training', 'evaluation', 'scene_scale', 'seed_policy',
    })
    unknown_keys = [k for k in normalized_cfg if k not in _KNOWN_TOP_LEVEL_KEYS]
    if unknown_keys:
        raise ValueError(
            f'experiment protocol contains unknown top-level keys: {unknown_keys}; '
            f'expected only {sorted(_KNOWN_TOP_LEVEL_KEYS)}'
        )

    # ---- 子映射未知键检测 ----
    _KNOWN_PUBLIC_BENCHMARK_KEYS = frozenset({
        'allowed_datasets', 'frozen_eval_split', 'tuning_forbidden', 'shared_split_required',
    })
    _KNOWN_TRAINING_KEYS = frozenset({
        'allowed_split_roles', 'forbidden_split_roles',
    })
    _KNOWN_EVALUATION_KEYS = frozenset({
        'require_prediction_bundles', 'default_failure_threshold_m', 'failure_threshold_max_m', 'require_ground_truth_unless_smoke',
    })
    for sub_name, known_keys in [
        ('public_benchmark', _KNOWN_PUBLIC_BENCHMARK_KEYS),
        ('training', _KNOWN_TRAINING_KEYS),
        ('evaluation', _KNOWN_EVALUATION_KEYS),
    ]:
        sub_cfg = normalized_cfg.get(sub_name)
        if isinstance(sub_cfg, Mapping):
            unknown_sub_keys = [k for k in sub_cfg if k not in known_keys]
            if unknown_sub_keys:
                raise ValueError(
                    f'experiment protocol {sub_name} contains unknown keys: {unknown_sub_keys}; '
                    f'expected only {sorted(known_keys)}'
                )

    # ---- conclusion_priority 与 metrics.yaml 交叉校验 ----
    # 根因修复：两个冻结配置独立定义了同一组指标的不同排序，
    # 下游消费者可能混淆"展示优先级"和"决策优先级"。
    # 交叉校验确保 conclusion_priority 中的每个指标名都在 metrics.yaml 中有定义，
    # 且 primary_metrics 中的指标全部出现在 conclusion_priority 中。
    conclusion_priority = [str(m) for m in (normalized_cfg.get('conclusion_priority') or [])]  # str() 归一化，兼容 numpy.str_
    from liquidloc.protocol.metric_schema import get_metric_meta  # 延迟导入避免循环依赖。
    metric_meta = get_metric_meta()  # 获取 metrics.yaml 的完整指标定义。
    all_defined_metrics = set(metric_meta.keys())  # 所有已定义的指标名集合。
    # 校验 conclusion_priority 中的每个指标名必须存在于 metrics.yaml。
    unknown_conclusion_metrics = [m for m in conclusion_priority if m not in all_defined_metrics]
    if unknown_conclusion_metrics:
        raise ValueError(
            f'experiment conclusion_priority references undefined metrics: {unknown_conclusion_metrics}; '
            f'all metrics must be defined in metrics.yaml'
        )
    # 校验 primary_metrics 中的指标必须全部出现在 conclusion_priority 中，
    # 确保决策优先级覆盖所有核心指标。
    from liquidloc.protocol.metric_schema import get_primary_metrics  # 延迟导入。
    primary_metrics = get_primary_metrics()
    missing_in_conclusion = [m for m in primary_metrics if m not in conclusion_priority]
    if missing_in_conclusion:
        raise ValueError(
            f'experiment conclusion_priority is missing primary metrics: {missing_in_conclusion}; '
            f'all primary_metrics must appear in conclusion_priority (order may differ)'
        )

    return normalized_cfg  # 校验全部通过，返回规范化后的配置。


@lru_cache(maxsize=1)  # 冻结协议在进程生命周期内不变，只加载和校验一次。
def _load_default_experiment_protocol_cached() -> dict[str, Any]:
    """缓存默认路径的冻结实验协议，避免每次调用都重新加载和校验。"""
    return load_experiment_protocol()


def _resolve_protocol_cfg(protocol_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """从调用者输入或冻结默认路径解析并校验协议配置。

    如果调用者未提供协议配置，则从默认路径加载冻结协议；
    如果提供了，则直接校验其是否与冻结合同一致。

    注意：当使用缓存路径时，返回深拷贝以防止调用者修改污染 lru_cache。

    参数：
        protocol_cfg: 可选的外部协议配置字典。

    返回：
        dict[str, Any]: 校验通过的协议配置。
    """
    if protocol_cfg is None:  # 未提供时从默认路径加载。
        return deepcopy(_load_default_experiment_protocol_cached())  # 深拷贝，防止调用者修改污染 lru_cache。
    return _validate_frozen_experiment_protocol_cfg(protocol_cfg)  # 提供时校验一致性（内部已 deepcopy）。


def load_experiment_protocol(protocol_path: str | Path | None = None) -> dict[str, Any]:
    """加载冻结实验协议并在入口处拒绝合同漂移。

    从指定路径或默认路径加载 YAML 配置，然后校验其是否与
    冻结合同完全一致。任何字段漂移都会导致加载失败。

    参数：
        protocol_path: 协议文件路径，默认使用 configs/base/experiment_protocol.yaml。

    返回：
        dict[str, Any]: 校验通过的冻结实验协议配置。

    异常：
        FileNotFoundError: 协议文件不存在时抛出。
        TypeError: 协议文件顶层不是映射或字段类型不匹配时抛出。
        ValueError: 路径为空白字符串、YAML 语法损坏或协议校验失败时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"protocol_path": str(protocol_path) if protocol_path else None}, "load_experiment_protocol 入口参数")
    if protocol_path is None:  # 未指定路径时使用默认路径。
        path = _DEFAULT_PATH.resolve()  # resolve() 消除符号链接和 .. 组件，与显式路径处理保持一致。
    else:
        if not str(protocol_path).strip():  # 空白路径不允许（同时覆盖 str 和 Path 对象，与 scene_axis_protocol 一致）。
            raise ValueError('experiment protocol path must not be blank')
        path = Path(protocol_path).resolve()  # 规范化路径，消除 .. 等穿越风险。
        if path.suffix.lower() not in ('.yaml', '.yml'):  # 只允许 YAML 文件。
            raise ValueError(f"experiment protocol path must be a YAML file, got {path.suffix!r}")
        project_root = find_project_root().resolve()  # 项目根目录的绝对路径。
        if not path.is_relative_to(project_root):  # 路径必须在项目根目录下（is_relative_to 防止兄弟目录前缀绕过）。
            raise ValueError(
                f"experiment protocol path must be within project root {project_root}, got {path}"
            )
    cfg = load_yaml_config(path)  # 加载 YAML 配置。
    return _validate_frozen_experiment_protocol_cfg(cfg)  # 校验并返回。


def get_default_failure_threshold_m(protocol_cfg: dict[str, Any] | None = None) -> float:
    """返回协议拥有的默认失效阈值（米）。

    参数：
        protocol_cfg: 可选的外部协议配置字典。

    返回：
        float: 默认失效阈值（米）。
    """
    proto = _resolve_protocol_cfg(protocol_cfg)  # 解析并校验协议。
    evaluation = dict(proto.get('evaluation') or {})  # 读取评测段。
    return _validate_failure_threshold_value(  # 校验并返回阈值。
        evaluation.get('default_failure_threshold_m', _DEFAULT_FAILURE_THRESHOLD_M),
        field_name='evaluation.default_failure_threshold_m',
    )


def get_public_benchmark_allowed_datasets(
    protocol_cfg: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """从协议配置中返回规范的公开基准数据集列表。

    对数据集名称做规范化（去重、统一大小写），确保返回的列表
    与冻结协议一致。

    参数：
        protocol_cfg: 可选的外部协议配置字典。

    返回：
        tuple[str, ...]: 规范化后的公开基准数据集名称元组。

    异常：
        ValueError: allowed_datasets 不是非空列表或元组时抛出。
    """
    proto = _resolve_protocol_cfg(protocol_cfg)  # 解析并校验协议。
    public = dict(proto.get('public_benchmark') or {})  # 读取公开基准段。
    allowed = public.get('allowed_datasets')  # 读取允许的数据集列表。
    if not isinstance(allowed, (list, tuple)) or not allowed:  # 必须是非空列表或元组，与 _coerce_non_string_list_field 一致。
        raise ValueError('public_benchmark.allowed_datasets must be a non-empty list or tuple')

    normalized: list[str] = []  # 存放规范化后的数据集名称。
    for dataset_name in allowed:  # 逐个规范化。
        canonical_name = normalize_public_dataset_name(dataset_name)  # 统一名称格式。
        if canonical_name not in normalized:  # 去重。
            normalized.append(canonical_name)
    return tuple(normalized)  # 返回不可变元组。


def get_public_benchmark_frozen_eval_split(
    protocol_cfg: dict[str, Any] | None = None,
) -> str:
    """从协议配置中返回公开基准冻结评测分割名。

    公开基准必须使用此冻结分割，禁止在 scene_sampler 等下游模块中硬编码。

    参数：
        protocol_cfg: 可选的外部协议配置字典。

    返回：
        str: 冻结评测分割名（如 'frozen_public_eval'）。

    异常：
        ValueError: frozen_eval_split 缺失或为空白时抛出。
    """
    proto = _resolve_protocol_cfg(protocol_cfg)  # 解析并校验协议。
    public = dict(proto.get('public_benchmark') or {})  # 读取公开基准段。
    frozen_eval_split = str(public.get('frozen_eval_split', _PUBLIC_BENCHMARK_FROZEN_EVAL_SPLIT)).strip()
    if not frozen_eval_split:  # 冻结分割名不能为空白。
        raise ValueError('public_benchmark.frozen_eval_split must not be blank')
    # C4: §2.5 运行期强制校验——冻结评测分割不得在运行时被篡改。
        if frozen_eval_split != _PUBLIC_BENCHMARK_FROZEN_EVAL_SPLIT:
            raise ValueError(
                f"runtime frozen_eval_split violation: expected "
                f"'{_PUBLIC_BENCHMARK_FROZEN_EVAL_SPLIT}', got '{frozen_eval_split}'"
            )
        return frozen_eval_split


def normalize_run_mode(
    mode: Any,
    *,
    default_mode: str,
    protocol_cfg: dict[str, Any] | None = None,
) -> str:
    """规范化运行模式并强制执行冻结的 quick/full 合同。

    在 common 层纯字符串归一化（liquidloc.common.constants.normalize_run_mode）
    基础上，叠加冻结协议校验：检查 quick_full_rule 未被篡改。

    quick 只允许作为冒烟规模标签，full 只允许作为真实执行标签，
    不允许引入新的科学语义。

    参数：
        mode: 用户指定的运行模式，可以是任意类型，内部会转为字符串。
        default_mode: 默认运行模式，必须是 'quick' 或 'full'。
        protocol_cfg: 可选的外部协议配置字典。

    返回：
        str: 规范化后的运行模式（'quick' 或 'full'）。

    异常：
        TypeError: quick_full_rule 类型不是字符串时抛出。
        ValueError: 模式不在允许集合中或 quick_full_rule 被篡改时抛出。
    """
    proto = _resolve_protocol_cfg(protocol_cfg)  # 解析并校验协议。
    quick_full_rule = proto.get('quick_full_rule', _QUICK_FULL_RULE)  # 读取 quick/full 规则。
    if not is_string_like(quick_full_rule):  # 防御性类型守卫：_resolve_protocol_cfg 已校验，但显式检查防止绕过。
        raise TypeError(f'experiment quick_full_rule must be a string, got {type(quick_full_rule).__name__}')
    if quick_full_rule != _QUICK_FULL_RULE:  # 规则被篡改则拒绝。
        raise ValueError(f'experiment quick_full_rule must be {_QUICK_FULL_RULE!r}, got {quick_full_rule!r}')
    return _normalize_run_mode_common(mode, default_mode=default_mode)


def normalize_train_request(
    cfg: dict[str, Any],
    protocol_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """根据冻结实验协议校验训练请求。

    校验内容包括：split_ids 非空、split_role 在允许列表中且不在禁止列表中、
    显式 train/val 切分不会与禁止进入训练流的序列边界重叠、运行模式合法。

    参数：
        cfg: 训练请求配置字典，必须包含 split_ids，可选 split_role、mode 等。
        protocol_cfg: 可选的外部协议配置字典。

    返回：
        dict[str, Any]: 校验通过的规范化训练请求摘要。

    异常：
        TypeError: cfg 不是映射类型或 split_role 不是字符串类型时抛出。
        ValueError: 任何字段不符合协议要求时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "split_ids": cfg.get("split_ids"),
        "split_role": cfg.get("split_role"),
        "mode": cfg.get("mode"),
        "train_split_ids": cfg.get("train_split_ids"),
        "val_split_ids": cfg.get("val_split_ids"),
    }, "normalize_train_request 入口参数")
    if not isinstance(cfg, Mapping):  # 训练请求配置必须是映射类型。
        raise TypeError('cfg must be a mapping')
    proto = _resolve_protocol_cfg(protocol_cfg)  # 解析并校验协议。
    training = dict(proto.get('training') or {})  # 读取训练段。
    split_ids = _normalize_non_empty_string_items(cfg.get('split_ids'), field_name='split_ids')  # 校验 split_ids。
    if not split_ids:  # 训练必须指定至少一个数据分割。
        raise ValueError('split_ids must be non-empty under training protocol')
    raw_split_role = cfg.get('split_role', 'train')  # 读取分割角色，默认为 train。
    if not is_string_like(raw_split_role):  # split_role 必须是字符串类型（含 numpy.str_）。
        raise TypeError(f'training split_role must be a string, got {type(raw_split_role).__name__}')
    split_role = str(raw_split_role).strip()  # 显式转 Python str 再去首尾空白。
    if not split_role:  # 角色不能为空白。
        raise ValueError('training split_role must not be blank')
    allowed_split_roles = set(training.get('allowed_split_roles') or [])  # 协议允许的角色集合。
    if split_role not in allowed_split_roles:  # 角色不在允许列表中则拒绝。
        raise ValueError(f'training split_role not allowed by protocol: {split_role}')
    forbidden_split_roles = set(training.get('forbidden_split_roles') or [])  # 协议禁止的角色集合。
    if split_role in forbidden_split_roles:  # 角色在禁止列表中则拒绝。
        raise ValueError(f'training split_role is explicitly forbidden by protocol: {split_role}')
    train_split_ids = _normalize_non_empty_string_items(  # 校验训练分割 ID。
        cfg.get('train_split_ids'),
        field_name='train_split_ids',
    )
    val_split_ids = _normalize_non_empty_string_items(  # 校验验证分割 ID。
        cfg.get('val_split_ids'),
        field_name='val_split_ids',
    )
    forbidden_training_split_ids = _normalize_optional_non_empty_string_items(
        cfg.get('forbidden_training_split_ids'),
        field_name='forbidden_training_split_ids',
    )
    split_id_set = set(split_ids)  # 提前构造，forbidden 校验和子集校验共用。
    if forbidden_training_split_ids is not None:
        unknown_forbidden_ids = [seq_id for seq_id in forbidden_training_split_ids if seq_id not in split_id_set]
        if unknown_forbidden_ids:
            raise ValueError(
                'forbidden_training_split_ids contains sequence ids outside split_ids: '
                f'{unknown_forbidden_ids}'
            )
        overlapping_train_ids = sorted(set(train_split_ids) & set(forbidden_training_split_ids))
        if overlapping_train_ids:
            raise ValueError(
                'train_split_ids must not overlap forbidden_training_split_ids under training protocol: '
                f'{overlapping_train_ids}'
            )
        overlapping_val_ids = sorted(set(val_split_ids) & set(forbidden_training_split_ids))
        if overlapping_val_ids:
            raise ValueError(
                'val_split_ids must not overlap forbidden_training_split_ids under training protocol: '
                f'{overlapping_val_ids}'
            )
    # 校验 train_split_ids 和 val_split_ids 必须是 split_ids 的子集，
    # 防止请求中包含不在实验范围内的序列 ID。
    unknown_train_ids = [seq_id for seq_id in train_split_ids if seq_id not in split_id_set]
    if unknown_train_ids:
        raise ValueError(
            'train_split_ids contains sequence ids outside split_ids: '
            f'{unknown_train_ids}'
        )
    unknown_val_ids = [seq_id for seq_id in val_split_ids if seq_id not in split_id_set]
    if unknown_val_ids:
        raise ValueError(
            'val_split_ids contains sequence ids outside split_ids: '
            f'{unknown_val_ids}'
        )
    # 校验 train_split_ids 和 val_split_ids 不得重叠，防止数据泄漏。
    overlapping_train_val_ids = sorted(set(train_split_ids) & set(val_split_ids))
    if overlapping_train_val_ids:
        raise ValueError(
            'train_split_ids must not overlap val_split_ids under training protocol: '
            f'{overlapping_train_val_ids}'
        )
    normalized_mode = normalize_run_mode(cfg.get('mode'), default_mode='full', protocol_cfg=proto)  # 规范化运行模式。
    return {  # 返回规范化后的训练请求摘要。
        'mode': normalized_mode,  # 运行模式。
        'split_role': split_role,  # 分割角色。
        'num_split_ids': len(split_ids),  # 分割 ID 数量。
        'split_ids': list(split_ids),  # 分割 ID 列表。
        'train_split_ids': list(train_split_ids),  # 训练分割 ID 列表。
        'val_split_ids': list(val_split_ids),  # 验证分割 ID 列表。
        'forbidden_training_split_ids': list(forbidden_training_split_ids or []),  # 禁止进入训练流的序列 ID。
        'failure_sample_policy': proto.get('failure_sample_policy'),  # 失败样本策略。
        'quick_full_rule': proto.get('quick_full_rule'),  # quick/full 规则。
    }


def validate_train_request(
    cfg: dict[str, Any],
    protocol_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """兼容旧公开接口名，语义保持为训练请求规范化校验。"""
    return normalize_train_request(cfg, protocol_cfg)


def normalize_train_test_disjointness(
    train_split_ids: Any,
    test_split_ids: Any,
    *,
    field_name_train: str = 'train_split_ids',
    field_name_test: str = 'test_split_ids',
) -> tuple[set[str], set[str]]:
    """§4.3.1.5 训练-测试布局不可重复显式守门：校验训练 split 与测试 split 的序列 ID 三元组差集。

    §4.3.1 第 5 条 + §9.2 要求"测试含训练未逐条见过的组合（布局 × 模式切换 × 异步突发）"，
    `前提指导.md:949` 字面守门字句为"训练—测试：遮挡布局或混合分区规则不得让测试成为
    训练的近重复脚本（§9.2）"。本函数把该协议字句落到代码层显式断言：训练 split 与测试 split
    的序列 ID 集合不得有任何交集（`(scene_id, N_level, A_level)` 三元组差集的序列层投影）。

    参数：
        train_split_ids: 训练分割的序列 ID 列表/集合/元组。
        test_split_ids: 测试分割的序列 ID 列表/集合/元组。
        field_name_train: 错误消息中训练侧字段名。
        field_name_test: 错误消息中测试侧字段名。

    返回：
        tuple[set[str], set[str]]: 校验通过后的 (train_id_set, test_id_set) 二元组。

    异常：
        TypeError: 输入不是列表/集合/元组/迭代器时抛出。
        ValueError: 训练与测试 split 存在任何序列 ID 交集时抛出。
    """
    if train_split_ids is None or test_split_ids is None:
        raise TypeError(f'{field_name_train} and {field_name_test} must not be None for disjointness check')
    if isinstance(train_split_ids, (str, bytes)) or isinstance(test_split_ids, (str, bytes)):
        raise TypeError(f'{field_name_train} and {field_name_test} must be iterables of ids, not strings')
    try:
        train_set = {str(sid).strip() for sid in train_split_ids if str(sid).strip()}
    except TypeError as exc:
        raise TypeError(f'{field_name_train} must be an iterable of stringifiable ids') from exc
    try:
        test_set = {str(sid).strip() for sid in test_split_ids if str(sid).strip()}
    except TypeError as exc:
        raise TypeError(f'{field_name_test} must be an iterable of stringifiable ids') from exc
    overlap = sorted(train_set & test_set)
    if overlap:
        raise ValueError(
            f'{field_name_train} and {field_name_test} must be disjoint under §4.3.1.5 / §9.2 '
            f'("测试含训练未逐条见过的组合"); overlapping sequence ids: {overlap}'
        )
    return train_set, test_set


def validate_train_test_disjointness(
    train_split_ids: Any,
    test_split_ids: Any,
) -> tuple[set[str], set[str]]:
    """兼容旧公开接口名，语义保持为训练-测试三元组差集规范化校验。"""
    return normalize_train_test_disjointness(train_split_ids, test_split_ids)



def normalize_eval_request(
    cfg: dict[str, Any],
    protocol_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """根据冻结实验协议校验评测请求。

    校验分为两层：
    1. **参数形状校验**：检查字段类型、取值范围、协议一致性等，
       不依赖文件系统状态，可在配置构建阶段随时调用。
    2. **运行前就绪校验**：检查 ground_truth_root 目录是否存在等
       运行时前置条件，确保评测启动时环境已就绪。

    将存在性校验放在此处而非延迟到实际评测时，是为了 fail-fast：
    错误越早暴露，可审计性越高。如果需要只做参数形状校验而不检查
    文件系统，请使用 ``normalize_eval_request_shape``。

    参数：
        cfg: 评测请求配置字典，必须包含 prediction_bundles，可选 ground_truth_root、mode 等。
        protocol_cfg: 可选的外部协议配置字典。

    返回：
        dict[str, Any]: 校验通过的规范化评测请求摘要。

    异常：
        ValueError: 任何字段不符合协议要求时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "prediction_bundles_count": len(cfg.get("prediction_bundles") or []),
        "ground_truth_root": str(cfg.get("ground_truth_root")) if cfg.get("ground_truth_root") else None,
        "mode": cfg.get("mode"),
    }, "normalize_eval_request 入口参数")
    result = normalize_eval_request_shape(cfg, protocol_cfg)  # 先做参数形状校验。
    _validate_eval_readiness(result)  # 再做运行前就绪校验。
    return result


def validate_eval_request(
    cfg: dict[str, Any],
    protocol_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """兼容旧公开接口名，语义保持为评测请求规范化校验。"""
    return normalize_eval_request(cfg, protocol_cfg)


def normalize_eval_request_shape(
    cfg: dict[str, Any],
    protocol_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """校验评测请求的参数形状，不依赖文件系统状态。

    只检查字段类型、取值范围、协议一致性等静态属性。
    适用于配置构建阶段或需要延迟文件系统检查的场景。

    参数：
        cfg: 评测请求配置字典。
        protocol_cfg: 可选的外部协议配置字典。

    返回：
        dict[str, Any]: 校验通过的规范化评测请求摘要（ground_truth_root 未做存在性检查）。
    """
    if not isinstance(cfg, Mapping):  # 评测请求配置必须是映射类型，与 normalize_train_request 防御级别一致。
        raise TypeError('cfg must be a mapping')
    proto = _resolve_protocol_cfg(protocol_cfg)  # 解析并校验协议。
    evaluation = dict(proto.get('evaluation') or {})  # 读取评测段。
    normalized_mode = normalize_run_mode(cfg.get('mode'), default_mode='full', protocol_cfg=proto)  # 规范化运行模式。
    bundles = _coerce_non_string_list_field(  # 校验 prediction_bundles。
        cfg.get('prediction_bundles'),
        field_name='prediction_bundles',
    )
    require_prediction_bundles = _read_protocol_boolean_flag(  # 读取协议中的 bundles 要求。
        evaluation,
        key='require_prediction_bundles',
        default=True,
        field_name='evaluation.require_prediction_bundles',
    )
    if require_prediction_bundles and not bundles:  # 协议要求 bundles 非空但实际为空。
        raise ValueError('prediction_bundles must be non-empty under evaluation protocol')

    ground_truth_root = _normalize_optional_path_string(  # 校验真值根目录。
        cfg.get('ground_truth_root'),
        field_name='ground_truth_root',
    )
    smoke_mode = _coerce_boolean_flag(cfg.get('smoke_mode'), field_name='smoke_mode')  # 校验冒烟模式标志。
    require_ground_truth_unless_smoke = _read_protocol_boolean_flag(  # 读取协议中的真值要求。
        evaluation,
        key='require_ground_truth_unless_smoke',
        default=True,
        field_name='evaluation.require_ground_truth_unless_smoke',
    )
    if require_ground_truth_unless_smoke:  # 协议要求提供真值（除非冒烟模式）。
        if not ground_truth_root and not smoke_mode:  # 既没有真值目录也不是冒烟模式。
            raise ValueError('ground_truth_root is required unless smoke_mode=True')

    threshold = _validate_failure_threshold_value(  # 校验失效阈值。
        cfg.get('failure_threshold', get_default_failure_threshold_m(proto)),
        field_name='failure_threshold',
    )
    return {  # 返回规范化后的评测请求摘要。
        'mode': normalized_mode,  # 运行模式。
        'smoke_mode': smoke_mode,  # 冒烟模式标志。
        'ground_truth_root': ground_truth_root,  # 真值根目录。
        'num_prediction_bundles': len(bundles),  # 预测包数量。
        'failure_threshold': threshold,  # 失效阈值。
        'aggregation_order': list(proto.get('aggregation_order') or []),  # 聚合顺序。
        'conclusion_priority': list(proto.get('conclusion_priority') or []),  # 结论优先级。
    }


def _validate_eval_readiness(eval_summary: dict[str, Any]) -> None:
    """运行前就绪校验：检查文件系统等运行时前置条件。

    此函数与 ``normalize_eval_request_shape`` 分离，使得参数形状校验
    可以独立于文件系统状态运行。存在性校验放在此处而非延迟到实际
    评测时，是为了 fail-fast：错误越早暴露，可审计性越高。

    参数：
        eval_summary: normalize_eval_request_shape 返回的规范化评测请求摘要。

    异常：
        ValueError: 运行时前置条件不满足时抛出。
    """
    ground_truth_root = eval_summary.get('ground_truth_root')  # 读取真值根目录。
    if ground_truth_root:  # 提供了真值目录时，检查目录是否存在。
        gt_root = Path(ground_truth_root).resolve()  # resolve 消除符号链接和 .. 穿越歧义，与 load_experiment_protocol 保持一致。
        if not gt_root.exists() or not gt_root.is_dir():  # 目录不存在或不是目录。
            raise ValueError(
                f'ground_truth_root must point to an existing directory, got: {ground_truth_root}'
            )


def normalize_public_benchmark_request(
    cfg: dict[str, Any],
    dataset_entry: dict[str, Any] | None = None,
    protocol_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """根据冻结实验协议校验公开基准请求。

    校验内容包括：数据集名称在允许列表中、分割名与冻结评测分割一致、
    禁止调参、seq_ids 非空、共享分割合同一致性。

    参数：
        cfg: 公开基准请求配置字典，必须包含 dataset_name 和 seq_ids。
        dataset_entry: 可选的数据集条目，包含 frozen_public_eval_seq_ids。
        protocol_cfg: 可选的外部协议配置字典。

    返回：
        dict[str, Any]: 校验通过的规范化公开基准请求摘要。

    异常：
        ValueError: 任何字段不符合协议要求时抛出。
        TypeError: dataset_entry 不是映射类型时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "dataset_name": cfg.get("dataset_name"),
        "seq_ids": cfg.get("seq_ids"),
        "split": cfg.get("split"),
        "mode": cfg.get("mode"),
    }, "normalize_public_benchmark_request 入口参数")
    if not isinstance(cfg, Mapping):  # 公开基准请求配置必须是映射类型，与 normalize_train_request 防御级别一致。
        raise TypeError('cfg must be a mapping')
    proto = _resolve_protocol_cfg(protocol_cfg)  # 解析并校验协议。
    public = dict(proto.get('public_benchmark') or {})  # 读取公开基准段。
    normalized_mode = normalize_run_mode(cfg.get('mode'), default_mode='full', protocol_cfg=proto)  # 规范化运行模式。
    allowed_datasets = frozenset(get_public_benchmark_allowed_datasets(proto))  # 获取允许的数据集集合。

    raw_dataset_name = cfg.get('dataset_name')
    if not raw_dataset_name:  # dataset_name 缺失或为空时直接报错，避免 normalize 后产生 "none" 字符串。
        raise ValueError("public benchmark requires a non-empty 'dataset_name' in config")
    if not is_string_like(raw_dataset_name):  # dataset_name 必须是字符串（含 numpy.str_），防止非字符串类型（如 int）通过 truthiness 检查后被 normalize 误用。
        raise TypeError(f"public benchmark 'dataset_name' must be a string, got {type(raw_dataset_name).__name__}")
    dataset_name = normalize_public_dataset_name(raw_dataset_name)  # 规范化数据集名称。
    if dataset_name not in allowed_datasets:  # 数据集不在允许列表中则拒绝。
        raise ValueError(
            f'dataset_name not allowed in public benchmark: {dataset_name!r}; '
            f'allowed: {sorted(allowed_datasets)}'
        )

    split = str(cfg.get('split', public.get('frozen_eval_split', 'frozen_public_eval'))).strip()  # 读取分割名。
    if not split:  # 分割名不能为空白。
        raise ValueError('public benchmark split must not be blank')
    frozen_eval_split = str(public.get('frozen_eval_split', 'frozen_public_eval')).strip()  # 协议冻结的分割名。
    if split != frozen_eval_split:  # 分割名必须与冻结值一致。
        raise ValueError(f'public benchmark split must be {frozen_eval_split}, got {split}')

    tuning_forbidden = _read_protocol_boolean_flag(  # 读取协议中的调参禁令。
        public,
        key='tuning_forbidden',
        default=True,
        field_name='public_benchmark.tuning_forbidden',
    )
    if tuning_forbidden:  # 协议禁止调参时，检查请求中是否试图调参。
        tuning_val = cfg.get('tuning_mode')  # 读取调参标志。
        tuning_enabled = _is_enabled_tuning_value(tuning_val)  # 保守解释调参标志。
        if tuning_enabled:  # 试图在公开基准中调参则拒绝。
            raise ValueError('tuning_mode is forbidden in public benchmark protocol')

    seq_ids = _normalize_non_empty_string_items(cfg.get('seq_ids'), field_name='seq_ids')  # 校验序列 ID 列表。
    if not seq_ids:  # 公开基准必须指定至少一个序列。
        raise ValueError('seq_ids must be non-empty in public benchmark protocol')

    shared_split_required = _read_protocol_boolean_flag(  # 读取协议中的共享分割要求。
        public,
        key='shared_split_required',
        default=True,
        field_name='public_benchmark.shared_split_required',
    )
    frozen_public_eval_seq_ids: list[str] | None = None  # 冻结的公开评测序列 ID 列表。
    if dataset_entry is None:  # 未提供数据集条目时。
        entry_keys: list[str] | None = None  # 条目键列表为空。
    else:
        if not isinstance(dataset_entry, Mapping):  # 数据集条目必须是映射。
            raise TypeError('dataset_entry must be a mapping when provided')
        entry_keys = sorted(dataset_entry.keys())  # 记录条目的键，用于审计。
        frozen_public_eval_seq_ids = _normalize_optional_non_empty_string_items(  # 读取冻结序列 ID。
            dataset_entry.get('frozen_public_eval_seq_ids'),
            field_name='dataset_entry.frozen_public_eval_seq_ids',
        )
    if shared_split_required:  # 协议要求共享分割时，必须提供冻结序列 ID。
        if frozen_public_eval_seq_ids is None:  # 未提供冻结序列 ID 则拒绝。
            raise ValueError(
                'public benchmark shared split requires dataset_entry.frozen_public_eval_seq_ids '
                'to declare the frozen public-eval sequence contract'
            )
        if normalized_mode == 'full':  # full 模式必须使用完整的冻结序列 ID。
            if seq_ids != frozen_public_eval_seq_ids:  # 序列 ID 不一致则拒绝。
                raise ValueError(
                    'public benchmark full mode must use dataset_entry.frozen_public_eval_seq_ids exactly '
                    f'under shared split contract: requested={seq_ids}, '
                    f'expected={frozen_public_eval_seq_ids}'
                )
            seq_ids = list(frozen_public_eval_seq_ids)  # 使用冻结序列 ID。
        else:  # quick 模式必须使用冻结序列 ID 的稳定前缀。
            requested_count = len(seq_ids)  # 请求的序列数量。
            frozen_quick_prefix = frozen_public_eval_seq_ids[:requested_count]  # 取冻结列表的前 N 个。
            if seq_ids != frozen_quick_prefix:  # 前缀不一致则拒绝。
                raise ValueError(
                    'public benchmark quick mode must use a stable prefix of '
                    'dataset_entry.frozen_public_eval_seq_ids under shared split contract: '
                    f'requested={seq_ids}, expected_prefix={frozen_quick_prefix}'
                )
    return {  # 返回规范化后的公开基准请求摘要。
        'mode': normalized_mode,  # 运行模式。
        'dataset_name': dataset_name,  # 数据集名称。
        'split': split,  # 分割名。
        'seq_ids': list(seq_ids),  # 序列 ID 列表。
        'shared_split_required': shared_split_required,  # 是否要求共享分割。
        'dataset_entry_keys': entry_keys,  # 数据集条目的键列表。
    }


def validate_public_benchmark_request(
    cfg: dict[str, Any],
    dataset_entry: dict[str, Any] | None = None,
    protocol_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """兼容旧公开接口名，语义保持为公开基准请求规范化校验。"""
    return normalize_public_benchmark_request(cfg, dataset_entry, protocol_cfg)


# =============================================================================
# §9 场景尺度与分布模块协议闸门（前提指导 §9.1 / §9.2 / §9.3 与 §0.2 操作定义槽）.
#
# 本节函数读取并校验冻结协议 experiment_protocol.yaml 中 scene_scale 与
# seed_policy 段, 为下游 split_builder / metric_runner / 12_run_statistics
# 提供 §9 量级门函数; 任何 run 不满足下界时按 §9 「放松则伤」报告对应比较
# 不可主张全序结论.
#
# 阈值来源 (与协议 default 数值同口径, 见 configs/base/experiment_protocol.yaml):
#   - N_seed_min            = 10  (§9.3 量级)
#   - N_seed_recommended     = 30  (§9.3 主结论建议)
#   - N_traj_te_min         = 20  (§9.3 量级)
#   - train_test_ratio_max  = 0.1 (§0.2 B05 / §9.2 少样本硬门)
#   - t_eff_min_s           = 20  (§9.3 量级)
#   - cold_start_offset_s   = 5   (§0.2 B06 / §9.1 冷启动统一划除)
#   - n_pulse_min           = 30  (§0.2 B07 / §9.3)
#   - n_async_min           = 20  (§0.2 B08 / §9.3)
#   - layout_family_min     = 3   (§9.3 「≥3 种布局族, 禁单布局刷满」)
# =============================================================================

_SCENE_SCALE_DEFAULT: dict[str, Any] = {
    "t_eff_min_s": 20.0,
    "cold_start_offset_s": 5.0,
    "cold_start_global_enforced": True,
    "n_pulse_min": 30,
    "n_async_min": 20,
    "layout_family_min": 3,
    "takeoff_landing_policy": "exclude_unified",
    "floor_transition_policy": "exclude_unified",
}

_SEED_POLICY_DEFAULT: dict[str, Any] = {
    "n_seed_min": 10,
    "n_seed_recommended": 30,
    "single_seed_no_conclusion": True,
    "train_test_ratio_max": 0.1,
    "n_traj_te_min": 20,
    "scene_seed_scope": ["scene", "nlos_packet_loss", "init_value", "network_init", "split"],
}

_ALLOWED_TAKEOFF_LANDING_POLICY = frozenset({"exclude_unified", "include_unified", "manual_guard"})
_ALLOWED_FLOOR_TRANSITION_POLICY = frozenset({"exclude_unified", "include_unified", "manual_guard"})


def _validate_scene_scale_section(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """校验冻结协议 scene_scale 段是否满足 §9 量级门字段集与数值约束.

    返回 normalized 后的 scene_scale 字典, 含所有 §9 必备字段;
    任一必备字段缺失/类型错/触发量级门突破则抛 ValueError.

    参数:
        cfg: 已校验过的协议配置映射.

    返回:
        dict[str, Any]: normalized 后的 scene_scale 字典.
    """
    raw = cfg.get('scene_scale')
    if raw is None:  # 老协议未声明 → 退回 §9 默认门, 但需明确记入 normalized_cfg 供下游使用.
        normalized_section: dict[str, Any] = dict(_SCENE_SCALE_DEFAULT)
    else:
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"experiment protocol scene_scale must be a mapping, got {type(raw).__name__}"
            )
        normalized_section = dict(_SCENE_SCALE_DEFAULT)  # 先填默认, 再覆盖所显式赋值.
        for raw_key, raw_value in raw.items():
            if raw_value is not None and raw_key in _SCENE_SCALE_DEFAULT:
                normalized_section[raw_key] = raw_value

    # 数值与类型校验: §9 量级门.
    t_eff_min_s = normalized_section.get('t_eff_min_s')
    if not is_real(t_eff_min_s) or math.isfinite(float(t_eff_min_s)) is False or float(t_eff_min_s) <= 0:
        raise ValueError(
            f"scene_scale.t_eff_min_s must be a positive finite real, got {t_eff_min_s!r}"
        )
    normalized_section['t_eff_min_s'] = float(t_eff_min_s)

    cold_start_offset_s = normalized_section.get('cold_start_offset_s')
    if (
        not is_real(cold_start_offset_s)
        or math.isfinite(float(cold_start_offset_s)) is False
        or float(cold_start_offset_s) < 0
    ):
        raise ValueError(
            f"scene_scale.cold_start_offset_s must be a non-negative finite real, "
            f"got {cold_start_offset_s!r}"
        )
    normalized_section['cold_start_offset_s'] = float(cold_start_offset_s)

    normalized_section['cold_start_global_enforced'] = _coerce_boolean_flag(
        normalized_section.get('cold_start_global_enforced'),
        field_name='scene_scale.cold_start_global_enforced',
    )

    n_pulse_min = normalized_section.get('n_pulse_min')
    if not is_integer(n_pulse_min) or int(n_pulse_min) < 0:
        raise ValueError(
            f"scene_scale.n_pulse_min must be a non-negative integer, got {n_pulse_min!r}"
        )
    normalized_section['n_pulse_min'] = int(n_pulse_min)

    n_async_min = normalized_section.get('n_async_min')
    if not is_integer(n_async_min) or int(n_async_min) < 0:
        raise ValueError(
            f"scene_scale.n_async_min must be a non-negative integer, got {n_async_min!r}"
        )
    normalized_section['n_async_min'] = int(n_async_min)

    layout_family_min = normalized_section.get('layout_family_min')
    if not is_integer(layout_family_min) or int(layout_family_min) < 1:
        raise ValueError(
            f"scene_scale.layout_family_min must be a positive integer, got {layout_family_min!r}"
        )
    normalized_section['layout_family_min'] = int(layout_family_min)

    takeoff_policy = normalized_section.get('takeoff_landing_policy')
    if takeoff_policy not in _ALLOWED_TAKEOFF_LANDING_POLICY:
        raise ValueError(
            f"scene_scale.takeoff_landing_policy must be one of "
            f"{sorted(_ALLOWED_TAKEOFF_LANDING_POLICY)}, got {takeoff_policy!r}"
        )

    floor_policy = normalized_section.get('floor_transition_policy')
    if floor_policy not in _ALLOWED_FLOOR_TRANSITION_POLICY:
        raise ValueError(
            f"scene_scale.floor_transition_policy must be one of "
            f"{sorted(_ALLOWED_FLOOR_TRANSITION_POLICY)}, got {floor_policy!r}"
        )

    # §9.1 「冷启动是否计入事先固定 + 全员同一」: cold_start_offset_s > 0 但
    # cold_start_global_enforced 关闭需明确显式声明, 不允许静默吞掉.
    if (
        float(normalized_section['cold_start_offset_s']) > 0
        and not normalized_section['cold_start_global_enforced']
    ):
        # 不强制 raise, 但记录在 normalized_section 额外字段供审计.
        normalized_section['_cold_start_global_enforced_explicit_off'] = True

    return normalized_section


def _validate_seed_policy_section(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """校验冻结协议 seed_policy 段是否满足 §9.3 量级门字段集与数值约束.

    返回 normalized 后的 seed_policy 字典. 任一必备字段缺失/类型错则抛 ValueError.

    参数:
        cfg: 已校验过的协议配置映射.

    返回:
        dict[str, Any]: normalized 后的 seed_policy 字典.
    """
    raw = cfg.get('seed_policy')
    if raw is None:
        normalized_section = dict(_SEED_POLICY_DEFAULT)
    else:
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"experiment protocol seed_policy must be a mapping, got {type(raw).__name__}"
            )
        normalized_section = dict(_SEED_POLICY_DEFAULT)
        for raw_key, raw_value in raw.items():
            if raw_value is not None and raw_key in _SEED_POLICY_DEFAULT:
                normalized_section[raw_key] = raw_value

    n_seed_min = normalized_section.get('n_seed_min')
    if not is_integer(n_seed_min) or int(n_seed_min) < 1:
        raise ValueError(
            f"seed_policy.n_seed_min must be a positive integer, got {n_seed_min!r}"
        )
    normalized_section['n_seed_min'] = int(n_seed_min)

    n_seed_recommended = normalized_section.get('n_seed_recommended')
    if not is_integer(n_seed_recommended) or int(n_seed_recommended) < int(n_seed_min):
        raise ValueError(
            f"seed_policy.n_seed_recommended must be ≥ n_seed_min ({int(n_seed_min)}), "
            f"got {n_seed_recommended!r}"
        )
    normalized_section['n_seed_recommended'] = int(n_seed_recommended)

    normalized_section['single_seed_no_conclusion'] = _coerce_boolean_flag(
        normalized_section.get('single_seed_no_conclusion'),
        field_name='seed_policy.single_seed_no_conclusion',
    )

    train_test_ratio_max = normalized_section.get('train_test_ratio_max')
    if (
        not is_real(train_test_ratio_max)
        or math.isfinite(float(train_test_ratio_max)) is False
        or float(train_test_ratio_max) <= 0
    ):
        raise ValueError(
            f"seed_policy.train_test_ratio_max must be a positive finite real, "
            f"got {train_test_ratio_max!r}"
        )
    normalized_section['train_test_ratio_max'] = float(train_test_ratio_max)

    n_traj_te_min = normalized_section.get('n_traj_te_min')
    if not is_integer(n_traj_te_min) or int(n_traj_te_min) < 1:
        raise ValueError(
            f"seed_policy.n_traj_te_min must be a positive integer, got {n_traj_te_min!r}"
        )
    normalized_section['n_traj_te_min'] = int(n_traj_te_min)

    scene_seed_scope = normalized_section.get('scene_seed_scope')
    if not isinstance(scene_seed_scope, (list, tuple)):
        raise TypeError(
            f"seed_policy.scene_seed_scope must be a list, got {type(scene_seed_scope).__name__}"
        )
    normalized_section['scene_seed_scope'] = list(scene_seed_scope)

    return normalized_section


def get_scene_scale(protocol_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """从冻结协议读取并返回 §9 scene_scale 段, 含全部量级门字段.

    返回的字典字段含 t_eff_min_s / cold_start_offset_s / cold_start_global_enforced /
    n_pulse_min / n_async_min / layout_family_min / takeoff_landing_policy /
    floor_transition_policy. 任一字段缺失会回退到 §9 默认门并校验.

    参数:
        protocol_cfg: 可选的外部协议配置字典.

    返回:
        dict[str, Any]: §9 scene_scale 段 normalized 字典.
    """
    proto = _resolve_protocol_cfg(protocol_cfg)
    return _validate_scene_scale_section(proto)


def get_seed_policy(protocol_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """从冻结协议读取并返回 §9 seed_policy 段, 含全部 §9.3 量级门字段.

    返回的字典字段含 n_seed_min / n_seed_recommended / single_seed_no_conclusion /
    train_test_ratio_max / n_traj_te_min / scene_seed_scope.

    参数:
        protocol_cfg: 可选的外部协议配置字典.

    返回:
        dict[str, Any]: §9 seed_policy 段 normalized 字典.
    """
    proto = _resolve_protocol_cfg(protocol_cfg)
    return _validate_seed_policy_section(proto)


def check_seed_count(
    n_seed: int,
    *,
    protocol_cfg: dict[str, Any] | None = None,
    raise_on_violation: bool = False,
) -> dict[str, Any]:
    """§9.3 N_seed 量级门检查: ≥ N_seed_min=10 量级.

    按 §9.3 「单一种子定全序不构成本排序成立」, 当 n_seed < n_seed_min 时
    返回 violated=True 并附降级声明; 与 seed_policy.single_seed_no_conclusion
    联动, 形成给 12_run_statistics / 15_build_six_cmp_aggregate 的明示信号.

    参数:
        n_seed: 当前 run 实际跨种子数 (例如 1 表示 round7 单种子).
        protocol_cfg: 可选外部协议配置; None 时从冻结 YAML 加载.
        raise_on_violation: True 时违反量级门直接 raise; 默认 False 仅返回警告.

    异常:
        TypeError: n_seed 不是整数.
        ValueError: raise_on_violation=True 且 n_seed < n_seed_min 时抛出.

    返回:
        dict 含 n_seed / n_seed_min / n_seed_recommended / violated /
        violated_recommended / single_seed_no_conclusion_allowed / message.
    """
    if not isinstance(n_seed, (int,)) or isinstance(n_seed, bool):
        raise TypeError(f"n_seed must be an int, got {type(n_seed).__name__}")
    if n_seed < 0:
        raise ValueError(f"n_seed must be non-negative, got {n_seed}")
    policy = get_seed_policy(protocol_cfg)
    n_seed_min = int(policy['n_seed_min'])
    n_seed_recommended = int(policy['n_seed_recommended'])
    single_seed_no = bool(policy['single_seed_no_conclusion'])
    violated = n_seed < n_seed_min
    violated_recommended = n_seed < n_seed_recommended
    message: str
    if violated:
        message = (
            f"§9.3 N_seed violation: n_seed={n_seed} < n_seed_min={n_seed_min}; "
            f"single_seed_no_conclusion={single_seed_no}; "
            f"按 §9.3 「单一种子定全序不构成本排序成立」不产生协议级全序结论, "
            f"仅可标『协议级候选』. 推荐提升到 n_seed≥{n_seed_recommended}."
        )
        if raise_on_violation:
            raise ValueError(message)
    elif violated_recommended:
        message = (
            f"§9.3 N_seed 达硬门但低于推荐: n_seed={n_seed} ≥ n_seed_min={n_seed_min}, "
            f"但 < n_seed_recommended={n_seed_recommended}; 主结论方差大时建议提升."
        )
    else:
        message = (
            f"§9.3 N_seed pass: n_seed={n_seed} ≥ n_seed_recommended={n_seed_recommended}."
        )
    return {
        "n_seed": int(n_seed),
        "n_seed_min": n_seed_min,
        "n_seed_recommended": n_seed_recommended,
        "violated": bool(violated),
        "violated_recommended": bool(violated_recommended),
        "single_seed_no_conclusion_allowed": single_seed_no,
        "message": message,
    }


def check_train_test_ratio(
    *,
    n_traj_train: int,
    n_traj_test: int,
    protocol_cfg: dict[str, Any] | None = None,
    raise_on_violation: bool = False,
) -> dict[str, Any]:
    """§9.2 / §0.2 B05 少样本硬门: 训练/测试规模比 ≤ train_test_ratio_max=0.1.

    按 §9.2 「少样本: 训练规模显著小于测试; 否则宽网比较 5、6 上翻」,
    违反即破 5、6 (比较 1 退化). ratio = n_traj_train / max(n_traj_test, 1).

    参数:
        n_traj_train: 训练轨数 (如 lead_id_grouped train 段长度).
        n_traj_test: 测试轨数.
        protocol_cfg: 可选外部协议配置.
        raise_on_violation: True 时直接 raise.

    异常:
        TypeError: 输入不是整数.
        ValueError: raise_on_violation=True 且 ratio > train_test_ratio_max 时抛出.

    返回:
        dict 含 n_traj_train / n_traj_test / ratio / max_allowed / violated / message.
    """
    for label, value in (("n_traj_train", n_traj_train), ("n_traj_test", n_traj_test)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{label} must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{label} must be non-negative, got {value}")
    policy = get_seed_policy(protocol_cfg)
    max_ratio = float(policy['train_test_ratio_max'])
    n_traj_te_min = int(policy['n_traj_te_min'])
    if n_traj_test == 0:
        ratio = float('inf')
    else:
        ratio = float(n_traj_train) / float(n_traj_test)
    violated = ratio > max_ratio
    violated_traj = n_traj_test < n_traj_te_min
    msg_parts: list[str] = []
    if violated:
        msg_parts.append(
            f"§9.2 train/test ratio violation: {n_traj_train}/{n_traj_test}={ratio:.4f} > "
            f"{max_ratio}; 宽网比较 5、6 上翻, 比较 1 退化."
        )
    if violated_traj:
        msg_parts.append(
            f"§9.3 N_traj_te violation: n_traj_test={n_traj_test} < n_traj_te_min={n_traj_te_min}; "
            f"统计功效不足."
        )
    if not msg_parts:
        msg_parts.append(
            f"§9.2 / §9.3 pass: train/test ratio={ratio:.4f} ≤ {max_ratio}, "
            f"N_traj_te={n_traj_test} ≥ {n_traj_te_min}."
        )
    message = " ".join(msg_parts)
    if (violated or violated_traj) and raise_on_violation:
        raise ValueError(message)
    return {
        "n_traj_train": int(n_traj_train),
        "n_traj_test": int(n_traj_test),
        "ratio": ratio,
        "max_allowed": max_ratio,
        "n_traj_te_min": n_traj_te_min,
        "violated": bool(violated),
        "violated_traj_te_min": bool(violated_traj),
        "message": message,
    }


def check_layout_family_count(
    n_test_layout_families: int,
    *,
    protocol_cfg: dict[str, Any] | None = None,
    raise_on_violation: bool = False,
) -> dict[str, Any]:
    """§9.3 测试布局族数下界: layout_family_min=3.

    按 §9.3 「测试覆盖 ≥3 种锚点-环境布局族, 禁单布局刷满 20 轨」,
    违反即破坏 §9 布局泛化主张, 不构成协议级全序结论.

    参数:
        n_test_layout_families: 测试集覆盖的不同 layout family 数 (≥1).
        protocol_cfg: 可选外部协议配置.
        raise_on_violation: True 时直接 raise.

    异常:
        TypeError: 输入不是整数.
        ValueError: raise_on_violation=True 且违反下界时抛出.

    返回:
        dict 含 n_test_layout_families / min_required / violated / message.
    """
    if not isinstance(n_test_layout_families, int) or isinstance(n_test_layout_families, bool):
        raise TypeError(
            f"n_test_layout_families must be an int, "
            f"got {type(n_test_layout_families).__name__}"
        )
    if n_test_layout_families < 0:
        raise ValueError(
            f"n_test_layout_families must be non-negative, got {n_test_layout_families}"
        )
    scene_scale = get_scene_scale(protocol_cfg)
    min_required = int(scene_scale['layout_family_min'])
    violated = n_test_layout_families < min_required
    if violated:
        message = (
            f"§9.3 layout family violation: n_test_layout_families="
            f"{n_test_layout_families} < layout_family_min={min_required}; "
            f"按 §9.3 「禁单布局刷满 20 轨」不构成协议级全序结论."
        )
        if raise_on_violation:
            raise ValueError(message)
    else:
        message = (
            f"§9.3 layout family pass: n_test_layout_families="
            f"{n_test_layout_families} ≥ layout_family_min={min_required}."
        )
    return {
        "n_test_layout_families": int(n_test_layout_families),
        "min_required": min_required,
        "violated": bool(violated),
        "message": message,
    }


def check_per_trajectory_pulse_async(
    *,
    n_pulse: int,
    n_async: int,
    protocol_cfg: dict[str, Any] | None = None,
    raise_on_violation: bool = False,
) -> dict[str, Any]:
    """§9.3 单轨脉冲/异步量级门: n_pulse≥30 / n_async≥20.

    按 §9.3 「每轨 n_pulse≥30 / n_async≥20 量级, 否则比较 1、5 无统计功效」.
    低于下界时降级该轨前 1/5 比较功效主张; 不直接 raise (除非显式要求).

    参数:
        n_pulse: 单轨有效段 UWB 脉冲测量次数.
        n_async: 单轨有效段显著异步错位事件数.
        protocol_cfg: 可选外部协议配置.
        raise_on_violation: True 时直接 raise.

    异常:
        TypeError: 输入不是整数.
        ValueError: raise_on_violation=True 且任一违反下界.

    返回:
        dict 含 n_pulse / n_pulse_min / n_async / n_async_min /
        pulse_violated / async_violated / message / cmp1_cmp5_at_risk.
    """
    for label, value in (("n_pulse", n_pulse), ("n_async", n_async)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{label} must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{label} must be non-negative, got {value}")
    scene_scale = get_scene_scale(protocol_cfg)
    n_pulse_min = int(scene_scale['n_pulse_min'])
    n_async_min = int(scene_scale['n_async_min'])
    pulse_violated = n_pulse < n_pulse_min
    async_violated = n_async < n_async_min
    parts: list[str] = []
    if pulse_violated:
        parts.append(
            f"§9.3 n_pulse violation: n_pulse={n_pulse} < n_pulse_min={n_pulse_min}"
        )
    if async_violated:
        parts.append(
            f"§9.3 n_async violation: n_async={n_async} < n_async_min={n_async_min}"
        )
    if not parts:
        parts.append(
            f"§9.3 per-trajectory pass: n_pulse={n_pulse}≥{n_pulse_min}, "
            f"n_async={n_async}≥{n_async_min}."
        )
    message = "; ".join(parts) + (
        " 按 §9.3 比较 1、5 无统计功效" if (pulse_violated or async_violated) else ""
    )
    if (pulse_violated or async_violated) and raise_on_violation:
        raise ValueError(message)
    return {
        "n_pulse": int(n_pulse),
        "n_pulse_min": n_pulse_min,
        "n_async": int(n_async),
        "n_async_min": n_async_min,
        "pulse_violated": bool(pulse_violated),
        "async_violated": bool(async_violated),
        "cmp1_cmp5_at_risk": bool(pulse_violated or async_violated),
        "message": message,
    }


# §8.3 单轨片段覆盖门禁：实验级（非单轨迹）覆盖全部 7 项片段类型。
# 单条轨迹按其轴配置覆盖一种或少数几种；整个实验 sweep 必须覆盖全部 7 项。
_REQUIRED_SEGMENT_TYPES = (
    "los_decent_geometry",        # LOS + 几何尚可 (N0 + G0/G1)
    "pulsed_nlos",                # 脉冲 NLOS (N1/N2/N3)
    "all_anchor_nlos",            # 全锚 NLOS (N3)
    "significant_async",          # 明显异步错位 (A1/A2/A3)
    "vio_degraded_or_interrupted",  # VIO 退化或中断 (V2/V3)
    "short_term_imu_dominated",   # 短时 IMU 主导 (V2/V3 + vio_outage_segments)
    "cold_start_transition",      # 差冷启动过渡到稳态 (cold_start_offset_s > 0)
)


def check_experiment_segment_coverage(
    scene_tasks: list[dict[str, Any]],
    noise_spec: Mapping[str, Any] | None = None,
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """§8.3 实验级单轨片段覆盖门禁：所有 7 项片段类型至少被一条轨迹覆盖。

    §8.3 要求"单轨应覆盖（时间上交错，非只出现一种）"——意指整个 sweep 实验需覆盖
    全部 7 项片段类型，不同轨迹按其轴配置各负担一种或少数几种。

    参数:
        scene_tasks: sample_scenes 返回的场景任务列表（每个含 'axes' 字典）。
        noise_spec: 噪声规格（含 vio_outage_segments / cold_start_offset_s）。
        raise_on_violation: True 时若任一片段类型无覆盖则 raise。

    返回:
        dict 含 covered (covered 类型集合)、missing (未覆盖)、passed。
    """
    covered: set[str] = set()
    noise_spec = noise_spec or {}
    has_vio_outage = bool(noise_spec.get("vio_outage_segments", []))
    cold_start_offset = float(noise_spec.get("cold_start_offset_s", 5.0))

    for task in scene_tasks:
        axes = task.get("axes", {})
        n_axis = axes.get("N")
        g_axis = axes.get("G")
        a_axis = axes.get("A")
        v_axis = axes.get("V")

        # 1. 视距 + 几何尚可：N0 + G0 或 N0 + G1（注：G0/G1 均为"几何尚可"，§8.3 行 1427）。
        if n_axis == "N0" and g_axis in ("G0", "G1"):
            covered.add("los_decent_geometry")
        # 2. 脉冲 NLOS：N1/N2/N3。
        if n_axis in ("N1", "N2", "N3"):
            covered.add("pulsed_nlos")
        # 3. 全锚 NLOS：N3。
        if n_axis == "N3":
            covered.add("all_anchor_nlos")
        # 4. 明显异步错位：A1/A2/A3。
        if a_axis in ("A1", "A2", "A3"):
            covered.add("significant_async")
        # 5. VIO 退化或中断：V2/V3。
        if v_axis in ("V2", "V3"):
            covered.add("vio_degraded_or_interrupted")
        # 6. 短时 IMU 主导：V2/V3 + vio_outage_segments。
        if v_axis in ("V2", "V3") and has_vio_outage:
            covered.add("short_term_imu_dominated")

    # 7. 差冷启动过渡到稳态：cold_start_offset_s > 0（协议级开关）。
    if cold_start_offset > 0.0:
        covered.add("cold_start_transition")

    required = set(_REQUIRED_SEGMENT_TYPES)
    missing = required - covered
    passed = not missing

    report = {
        "covered": sorted(covered),
        "missing": sorted(missing),
        "passed": passed,
        "n_tasks": len(scene_tasks),
    }
    if not passed and raise_on_violation:
        raise ValueError(
            "§8.3 单轨片段覆盖违规——实验未覆盖全部必需片段类型: "
            f"missing={sorted(missing)}; covered={sorted(covered)}; "
            f"n_tasks={len(scene_tasks)}"
        )
    return report


def assert_anchor_uniform_source(
    method_anchor_layouts: Mapping[str, Mapping[str, Any]],
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """§8.1 / 细节 R-D-C 锚点源一致性门禁：所有方法必须使用同一 anchor_layout，
    且锚点数 Na ∈ {3, 4, 5}（§8.1 L1360：推荐默认 Na∈{3,4,5}；禁止 Na≥8 高冗余优 GDOP）。

    §8.1 L1360（spec）："锚点数与布局保持欠定或临界压力（平面测距不宜长期高冗余优 GDOP）。
    推荐默认（平面题）：同时可用锚 Na∈{3,4,5}量级，布局以矩形/不规则四边形为主，基线与 L_xy 同量级；
    禁止默认 Na≥8 高冗余优 GDOP 却仍声称欠定压力。"

    §8.1 H（spec L1368）"禁止只给一方真值锚、另一方带偏锚"，
    细节 R-D-C（spec L1448）"移动锚点若存在: 轨迹与误差模型全员同一; 禁止单方已知移动锚真值"。

    参数:
        method_anchor_layouts: 方法名到 anchor_layout 的映射。
            锚点位置列表按字典序比对（与锚点 ID 顺序无关，仅比对位置集合）。
        raise_on_violation: True 时若任一方法锚点布局与其他方法不一致
            或 Na≥8 高冗余则 raise。

    返回:
        dict 含 uniform (bool)、na_in_range (bool)、unique_layouts_count、
        method_count、mismatches (list[str])、na_count (int)。
    """
    if not isinstance(method_anchor_layouts, Mapping):
        raise TypeError(f"method_anchor_layouts must be a Mapping, got {type(method_anchor_layouts).__name__}")
    if len(method_anchor_layouts) < 1:
        raise ValueError("method_anchor_layouts must contain at least one method")

    def _layout_fingerprint(layout: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
        positions = list(layout.get("anchor_positions") or [])
        pts = []
        for p in positions:
            coords = list(p)
            if len(coords) != 2:
                continue
            pts.append((round(float(coords[0]), 6), round(float(coords[1]), 6)))
        return tuple(sorted(pts))

    def _switch_fingerprint(layout: Mapping[str, Any]) -> tuple:
        """锚点切换规则指纹（spec L1460 "锚点布局切换/增减锚: 规则可知且全员同一"）。

        原 fingerprint 仅比对 anchor_positions，不比对 anchor_switch 字段——
        若两 method 位置同但 switch 规则不同（一个含 switch_times，一个无切换），
        silent pass 通过 uniform_source 门禁，违背 spec L1460"切换规则全员同一"。
        现补 anchor_switch 字段指纹：序列化为 (has_switch, switch_times, reason)
        三元组的可哈希表示，纳入 uniform 比对。
        """
        sw = layout.get("anchor_switch") if isinstance(layout, Mapping) else None
        if sw is None:
            return (False, (), None)
        if not isinstance(sw, Mapping):
            # 不规则 anchor_switch 由 assert_anchor_switch_ruleknown 单独审计；
            # 此处用 sentinel 保留信号以触发 uniform 比对差异。
            return (True, (), "<non-mapping>")
        switch_times = sw.get("switch_times")
        reason = sw.get("reason")
        try:
            times_tuple = tuple(round(float(t), 6) for t in (switch_times or []))
        except (TypeError, ValueError):
            times_tuple = ()  # 非数值由 ruleknown 门单独审计；此处只做统一性比对
        return (True, times_tuple, str(reason) if reason is not None else None)

    fingerprints: dict[str, tuple[tuple[float, float], ...]] = {}
    switch_fingerprints: dict[str, tuple] = {}
    na_counts: dict[str, int] = {}
    for method_name, layout in method_anchor_layouts.items():
        if not isinstance(layout, Mapping):
            raise TypeError(f"anchor_layout for method '{method_name}' must be a Mapping")
        fingerprints[method_name] = _layout_fingerprint(layout)
        switch_fingerprints[method_name] = _switch_fingerprint(layout)
        na_counts[method_name] = len(fingerprints[method_name])

    # §8.1 L1360 + 细节 R-D-F "锚点布局切换/增减锚: 规则已知且全员同一":
    # 布局位置一致 AND 切换规则一致才算 uniform；仅位置同但 switch 规则不同
    # 仍视为不一致（违反 L1460"切换规则全员同一"）。
    layout_uniform = len(set(fingerprints.values())) == 1
    switch_uniform = len(set(switch_fingerprints.values())) == 1
    uniform = layout_uniform and switch_uniform
    mismatches: list[str] = []
    if not layout_uniform:
        reference_method = next(iter(fingerprints))
        reference_fp = fingerprints[reference_method]
        for m, fp in fingerprints.items():
            if fp != reference_fp:
                mismatches.append(f"{m}: anchor_positions mismatch")
    if not switch_uniform:
        reference_method = next(iter(switch_fingerprints))
        reference_sw = switch_fingerprints[reference_method]
        for m, sw in switch_fingerprints.items():
            if sw != reference_sw:
                mismatches.append(f"{m}: anchor_switch mismatch")

    # §8.1 L1360 锚点数 Na ∈ {3, 4, 5} 校验（禁 Na≥8 高冗余）。
    # 所有方法的锚点数应一致（否则 uniform 已 flag），取任一方法的 na 值即可。
    first_method = next(iter(method_anchor_layouts))
    na_count = na_counts[first_method]
    na_in_range = bool(3 <= na_count <= 5)

    report = {
        "uniform": uniform,
        "layout_uniform": layout_uniform,
        "switch_uniform": switch_uniform,
        "na_in_range": na_in_range,
        "na_count": na_count,
        "unique_layouts_count": len(set(fingerprints.values())),
        "unique_switches_count": len(set(switch_fingerprints.values())),
        "method_count": len(method_anchor_layouts),
        "mismatches": mismatches,
    }
    reasons: list[str] = []
    if not uniform:
        reasons.append(
            f"anchor_uniform: methods using different anchor layouts/switch rules; "
            f"mismatches={mismatches}; "
            f"unique_layouts_count={len(set(fingerprints.values()))}; "
            f"unique_switches_count={len(set(switch_fingerprints.values()))}"
        )
    if not na_in_range:
        reasons.append(
            f"§8.1 L1360 anchor_count violation: Na={na_count} not in {{3,4,5}}; "
            "Na≥8 高冗余优 GDOP 禁止；Na<3 欠定压力不足"
        )
    report["reasons"] = reasons
    report["passed"] = (uniform and na_in_range)
    if reasons and raise_on_violation:
        raise ValueError(
            "§8.1 anchor_uniform_source violation: " + "; ".join(reasons)
        )
    return report


def assert_anchor_switch_ruleknown(
    anchor_layout: Mapping[str, Any],
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """§8.1 G / 细节 R-D-F 锚点切换规则可知门禁：若存在锚点切换，切换时刻必须规则可知。

    §8.1 G（spec L1366）"锚点切换若存在: 切换时刻规则可知; 禁止切换处强连续因子抹平脉冲"，
    细节 R-D-F（spec L1460）"锚点布局切换、增减锚: 规则已知且全员同一; 切换处禁止强连续抹平"。

    当前代码库未实现锚点切换（grep anchor_switch|layout_switch|k_alternate 均无命中），
    故本门禁作为「存在性审计」：当 anchor_layout 含显式 anchor_switch 字段时，
    必须 switch_times 列表 + reason 字符串都齐全；否则视为无切换通过门禁。

    参数:
        anchor_layout: anchor_layout 配置；可能含 "anchor_switch" 字段触发本门禁。
        raise_on_violation: True 时若 anchor_switch 字段存在但不规则则 raise。

    返回:
        dict 含 has_switch (bool)、rule_known (bool)、switch_count (int)、reason (str | None)。
    """
    if not isinstance(anchor_layout, Mapping):
        raise TypeError(f"anchor_layout must be a Mapping, got {type(anchor_layout).__name__}")
    switch_spec = anchor_layout.get("anchor_switch")
    has_switch = switch_spec is not None
    rule_known = True
    switch_count = 0
    reason: str | None = None
    if has_switch:
        if not isinstance(switch_spec, Mapping):
            rule_known = False
        else:
            switch_times = switch_spec.get("switch_times")
            reason = switch_spec.get("reason")
            if not isinstance(switch_times, list) or len(switch_times) == 0:
                rule_known = False
            else:
                switch_count = len(switch_times)
                for t in switch_times:
                    try:
                        float(t)
                    except (TypeError, ValueError):
                        rule_known = False
                        break
            if not isinstance(reason, str) or not reason:
                rule_known = False
    report = {
        "has_switch": has_switch,
        "rule_known": rule_known,
        "switch_count": switch_count,
        "reason": reason,
    }
    if not rule_known and raise_on_violation and has_switch:
        raise ValueError(
            "§8.1 G / R-D-F anchor_switch_ruleknown violation: anchor_switch field present but rule not "
            f"knowable; has_switch={has_switch}, switch_count={switch_count}, reason={reason!r}"
        )
    return report


def assert_imu_bias_observability(
    envelope: Mapping[str, Any],
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """§8.2 R8.2-D IMU 偏置可区分性门禁：激励使 IMU 偏置与位置/速度可区分（§5.3）。

    §8.2 行 1375："激励使 IMU 偏置与位置速度可区分（§5.3）"。
    §8.2 行 1373："含停车再走、急转、加减速，使简单恒速/过平滑运动因子失效"。

    IMU 偏置与位置/速度可区分性需要：
    - 有显著多段加减速（acceleration excitation）：a95 ≥ 0.5 m/s²
    - 有停车再走（near-zero-speed 比例 > 0.03，停走产生速度重置）
    - 有显著转向（turn_total ≥ 2π 或 significant_turn_count ≥ 3，姿态激励）

    满足此三合取，IMU 偏置项才可从位置/速度变化中区分——否则偏置被位置/速度吸收。

    参数:
        envelope: compute_trajectory_envelope 返回的运动包络字典。
        raise_on_violation: True 时若任一子项不满足则 raise。

    返回:
        dict 含 observable (bool)、accel_excited (bool)、has_stop_go (bool)、
        yaw_excited (bool)、sub_thresholds (dict)、envelope_subset (dict)。
    """
    if not isinstance(envelope, Mapping):
        raise TypeError(f"envelope must be a Mapping, got {type(envelope).__name__}")

    accel_excited = bool(envelope.get("a95_mps2", 0.0) >= 0.5)
    near_zero_speed = float(envelope.get("near_zero_speed_ratio", 0.0))
    has_stop_go = bool(near_zero_speed >= 0.03)
    turn_total = float(envelope.get("turn_total_rad", 0.0))
    sig_turns = float(envelope.get("significant_turn_count", 0.0))
    yaw_excited = bool(turn_total >= 2.0 * math.pi or sig_turns >= 3.0)
    observable = accel_excited and has_stop_go and yaw_excited

    sub_thresholds = {
        "a95_min_mps2": 0.5,
        "near_zero_speed_min_ratio": 0.03,
        "turn_total_min_rad": 2.0 * math.pi,
        "significant_turn_min_count": 3,
    }
    envelope_subset = {
        "a95_mps2": float(envelope.get("a95_mps2", 0.0)),
        "near_zero_speed_ratio": near_zero_speed,
        "turn_total_rad": turn_total,
        "significant_turn_count": sig_turns,
    }
    report = {
        "observable": observable,
        "accel_excited": accel_excited,
        "has_stop_go": has_stop_go,
        "yaw_excited": yaw_excited,
        "sub_thresholds": sub_thresholds,
        "envelope_subset": envelope_subset,
    }
    if not observable and raise_on_violation:
        failed = {
            "accel_excited": accel_excited,
            "has_stop_go": has_stop_go,
            "yaw_excited": yaw_excited,
        }
        failed = {k: v for k, v in failed.items() if not v}
        raise ValueError(
            "§8.2 R8.2-D IMU bias observability violation: trajectory lacks sufficient excitation to "
            f"distinguish IMU bias from position/velocity; failed={failed}; envelope={envelope_subset}"
        )
    return report


def assert_cold_start_x_underdetermined_geometry(
    scene_tasks: Sequence[Mapping[str, Any]],
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """§8.4 差冷启动 × 欠定几何合取审计门：单轨/实验应同时含 cold-start 与 G 病态布局压迫。

    §8.4 行 1437："差冷启动（§1.4）与欠定几何同时存在时，无核 FGO 吸引域问题最尖"。
    §8.4 行 1440 放松则伤："优几何 → 3；强周期 → 5、6；好初值 → 3"。

    本门禁要求实验 sweep 至少包含一项"差冷启动（cold_start_offset_s > 0）+ 欠定/病态几何
    （G2 或 K3/K4）"的合取任务。否则 §8.4 主张的"差冷启动 × 欠定几何"压力未被协议覆盖。

    参数:
        scene_tasks: 场景任务列表；每项含 'axes' 子字典与 'noise_spec'（可选）
            含 cold_start_offset_s（可选，缺省 0 = 无差冷启动）。
        raise_on_violation: True 时若无任何 task 同时含差冷启动 + 欠定几何则 raise。

    返回:
        dict 含 covered (bool)、tasks_with_conj (list[int])、tasks_with_cold_start (int)、
        tasks_with_underdet_geom (int)、total_tasks (int)。
    """
    if not isinstance(scene_tasks, Sequence):
        raise TypeError(f"scene_tasks must be a Sequence, got {type(scene_tasks).__name__}")
    if not scene_tasks:
        raise ValueError("scene_tasks must be non-empty")

    tasks_with_conj: list[int] = []
    tasks_with_cold_start = 0
    tasks_with_underdet_geom = 0
    for i, task in enumerate(scene_tasks):
        if not isinstance(task, Mapping):
            continue
        axes = task.get("axes") or {}
        g_axis = axes.get("G") if isinstance(axes, Mapping) else None
        k_axis = axes.get("K") if isinstance(axes, Mapping) else None
        # 欠定/病态几何：G2 或 K3/K4（高 K 也欠定，但根据 §8.1 此处采用 K3/K4）。
        underdet = g_axis == "G2" or k_axis in ("K3", "K4")
        # 差冷启动：noise_spec.cold_start_offset_s > 0
        noise_spec = task.get("noise_spec") or {}
        cold_start_offset = 0.0
        if isinstance(noise_spec, Mapping):
            cold_start_offset = float(noise_spec.get("cold_start_offset_s", 0.0) or 0.0)
        has_cold_start = cold_start_offset > 0.0
        if has_cold_start:
            tasks_with_cold_start += 1
        if underdet:
            tasks_with_underdet_geom += 1
        if has_cold_start and underdet:
            tasks_with_conj.append(i)
    covered = len(tasks_with_conj) >= 1
    report = {
        "covered": covered,
        "tasks_with_conj": tasks_with_conj,
        "tasks_with_cold_start": tasks_with_cold_start,
        "tasks_with_underdet_geom": tasks_with_underdet_geom,
        "total_tasks": len(scene_tasks),
    }
    if not covered and raise_on_violation:
        raise ValueError(
            "§8.4 cold_start_x_underdetermined_geometry violation: no task combines a bad cold-start "
            f"(cold_start_offset_s > 0) with an underdetermined geometry (G2 or K3/K4); "
            f"cold_start_count={tasks_with_cold_start}, underdet_count={tasks_with_underdet_geom}, "
            f"total={len(scene_tasks)}"
        )
    return report


def assert_cross_method_plane_z_equality(
    method_constraints: Mapping[str, Mapping[str, Any]],
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """细节 R-D-D 平面/高度统一约束门禁：所有方法的平面约束 / Z 固定设置须全员同一。

    细节 R-D-D（spec L1450-L1456）："平面约束、高度已知/Z 固定，须全员同一；单方加平面
    约束会显著改可观性与 RMSE"... "禁止只给 FGO/某一 NN+EKF 加运动约束因子抬名次"。

    参数:
        method_constraints: 方法名到约束配置的映射；每项含字段：
            - plane_constraint ('none' | 'enforced')
            - z_constraint ('none' | 'fixed' | 'modeled')
            - motion_constraint ('none' | 'nonholonomic' | 'zupt' | 'wheel_speed' |
                                'magnetometer' | 'barometer')
        raise_on_violation: True 时若任一字段在不同方法间不一致则 raise。

    返回:
        dict 含 uniform (bool)、plane_uniform (bool)、z_uniform (bool)、
        motion_uniform (bool)、mismatches (dict[str, list[str]])。
    """
    if not isinstance(method_constraints, Mapping):
        raise TypeError(f"method_constraints must be a Mapping, got {type(method_constraints).__name__}")
    if len(method_constraints) < 1:
        raise ValueError("method_constraints must contain at least one method")

    def _get(c: Mapping[str, Any], field: str, default: str) -> str:
        v = c.get(field, default)
        return str(v) if v is not None else default

    methods = list(method_constraints.keys())
    plane_vals = {m: _get(method_constraints[m], "plane_constraint", "none") for m in methods}
    z_vals = {m: _get(method_constraints[m], "z_constraint", "none") for m in methods}
    motion_vals = {m: _get(method_constraints[m], "motion_constraint", "none") for m in methods}

    plane_uniform = len(set(plane_vals.values())) == 1
    z_uniform = len(set(z_vals.values())) == 1
    motion_uniform = len(set(motion_vals.values())) == 1
    uniform = plane_uniform and z_uniform and motion_uniform

    mismatches: dict[str, list[str]] = {}
    if not plane_uniform:
        ref = plane_vals[methods[0]]
        mismatches["plane_constraint"] = [m for m in methods if plane_vals[m] != ref]
    if not z_uniform:
        ref = z_vals[methods[0]]
        mismatches["z_constraint"] = [m for m in methods if z_vals[m] != ref]
    if not motion_uniform:
        ref = motion_vals[methods[0]]
        mismatches["motion_constraint"] = [m for m in methods if motion_vals[m] != ref]

    report = {
        "uniform": uniform,
        "plane_uniform": plane_uniform,
        "z_uniform": z_uniform,
        "motion_uniform": motion_uniform,
        "mismatches": mismatches,
    }
    if not uniform and raise_on_violation:
        raise ValueError(
            "§8 细节 R-D-D cross_method_plane_z_equality violation: methods use different "
            f"plane/z/motion constraints; mismatches={mismatches}"
        )
    return report


def assert_anchor_switch_anti_smoothing(
    switch_events: Sequence[Mapping[str, Any]],
    raw_residuals: Sequence[Mapping[str, Any]],
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """细节 R-D-F 锚点切换禁强连续抹平 enforcement 门禁。

    细节 R-D-F（spec L1460）："锚点布局切换、增减锚：规则可知且全员同一；切换处禁止强
    连续抹平（§3.4）"。

    门禁逻辑：对每个 anchor switch event，检测 raw residual 在 ±1.0 s 窗口内是否含突跳；
    若支持切换时刻残差信号含突跳脉冲则视为"未强连续抹平"。否则（无突跳或无残差）视为
    可能存在前级强连续抹平。

    参数:
        switch_events: 切换事件列表，每项含 't' (秒)、'anchor_id_dropped'、'anchor_id_added'。
        raw_residuals: 原始残差序列，每项含 't' (秒)、'anchor_id'、'r' (残差米)。
        raise_on_violation: True 时若存在切换事件且无对应残差突跳则 raise。

    返回:
        dict 含 anti_smoothed (bool)、switch_count (int)、switches_with_pulse (int)、
        details (list[dict])。
    """
    if not isinstance(switch_events, Sequence):
        raise TypeError(f"switch_events must be a Sequence, got {type(switch_events).__name__}")
    if not isinstance(raw_residuals, Sequence):
        raise TypeError(f"raw_residuals must be a Sequence, got {type(raw_residuals).__name__}")

    # 按 anchor_id 分桶残差。
    residuals_by_anchor: dict[str, list[tuple[float, float]]] = {}
    for r in raw_residuals:
        if not isinstance(r, Mapping):
            continue
        aid = r.get("anchor_id")
        t = r.get("t")
        rr = r.get("r")
        if aid is None or t is None or rr is None:
            continue
        try:
            t_f = float(t)
            r_f = float(rr)
        except (TypeError, ValueError):
            continue
        residuals_by_anchor.setdefault(str(aid), []).append((t_f, r_f))

    switch_count = len(switch_events)
    switches_with_pulse = 0
    details: list[dict[str, Any]] = []
    for ev in switch_events:
        if not isinstance(ev, Mapping):
            continue
        ev_t = ev.get("t")
        aid_added = ev.get("anchor_id_added")
        if ev_t is None or aid_added is None:
            continue
        try:
            ev_t_f = float(ev_t)
        except (TypeError, ValueError):
            continue
        aid_key = str(aid_added)
        bucket = residuals_by_anchor.get(aid_key, [])
        if not bucket:
            details.append({"t": ev_t_f, "anchor_id": aid_key, "has_pulse": False, "reason": "no_residuals"})
            continue
        # 在切换时刻 ≤ 1.0 s 窗口内查找最大残差突跳。
        spikes = [abs(rr) for (tt, rr) in bucket if abs(tt - ev_t_f) <= 1.0]
        max_spike = max(spikes) if spikes else 0.0
        # §3.4 残差突跳最小阈值 0.1 m（基于 raw measurement noise floor 0.05 m × 2）。
        has_pulse = max_spike >= 0.1
        if has_pulse:
            switches_with_pulse += 1
        details.append({
            "t": ev_t_f,
            "anchor_id": aid_key,
            "has_pulse": has_pulse,
            "max_spike_m": max_spike,
            "n_residuals_in_window": len(spikes),
        })

    anti_smoothed = switch_count == 0 or switches_with_pulse == switch_count
    report = {
        "anti_smoothed": anti_smoothed,
        "switch_count": switch_count,
        "switches_with_pulse": switches_with_pulse,
        "details": details,
    }
    if not anti_smoothed and raise_on_violation and switch_count > 0:
        no_pulse = [d for d in details if not d["has_pulse"]]
        raise ValueError(
            "§8.1 细节 R-D-F anchor_switch_anti_smoothing violation: switch events without raw residual "
            f"pulse suggests strong-continuity smoothing; no_pulse={no_pulse}"
        )
    return report


def assert_underdetermined_observation_ratio(
    weak_geometry_mask: Sequence[int],
    *,
    min_ratio: float = 0.10,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """§8.1 L1364 病态观测占比门禁：差 GDOP 时段达足够占比。

    §8.1 行 1364："测试含差 GDOP 时段；病态观测（雅可比近奇异、弱几何）达到足够占比（§34）"。

    参数:
        weak_geometry_mask: 每个时间样本的"病态观测"二值标记序列（0=健康、1=病态/弱几何）。
        min_ratio: 病态占比下限（默认 0.10 = 10% 全程病态时段）；§34 推荐 ≥ 10%。
        raise_on_violation: True 时若占比 < min_ratio 则 raise。

    返回:
        dict 含 sufficient (bool)、weak_ratio (float)、n_samples (int)、n_weak (int)。
    """
    if not isinstance(weak_geometry_mask, Sequence):
        raise TypeError(f"weak_geometry_mask must be a Sequence, got {type(weak_geometry_mask).__name__}")
    n = len(weak_geometry_mask)
    if n == 0:
        raise ValueError("weak_geometry_mask must not be empty")
    n_weak = 0
    for v in weak_geometry_mask:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv != 0:
            n_weak += 1
    weak_ratio = n_weak / n
    sufficient = bool(weak_ratio >= min_ratio)
    report = {
        "sufficient": sufficient,
        "weak_ratio": float(weak_ratio),
        "n_samples": n,
        "n_weak": n_weak,
        "min_ratio_required": float(min_ratio),
    }
    if not sufficient and raise_on_violation:
        raise ValueError(
            "§8.1 L1364 underdetermined_observation_ratio violation: weak-geometry ratio "
            f"({weak_ratio:.3f}) < required ({min_ratio}); n_weak={n_weak}/{n}"
        )
    return report


def assert_moving_anchor_truth_equality(
    method_moving_anchor_truth: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """细节 R-D-C 移动锚点真值差分门禁：移动锚点真值轨迹对所有方法必须同一。

    §8.1 细节 R-D-C L1448："移动锚点若存在：轨迹与误差模型全员同一；禁止单方已知移动锚真值"。

    `assert_anchor_uniform_source` 仅检查静态锚点位置；本门禁扩展到移动锚点真值轨迹
    (per-sample position)。

    参数:
        method_moving_anchor_truth: 方法名到"移动锚点真值轨迹"列表；
            每项含 't'、'anchor_id'、'px'、'py'（数值，可选 'pz'）。
        raise_on_violation: True 时若移动锚点真值序列跨方法不一致则 raise。

    返回:
        dict 含 uniform (bool)、has_moving (bool)、method_count (int)、mismatches (list[str])。
    """
    if not isinstance(method_moving_anchor_truth, Mapping):
        raise TypeError(f"method_moving_anchor_truth must be a Mapping, got {type(method_moving_anchor_truth).__name__}")
    if len(method_moving_anchor_truth) < 1:
        raise ValueError("method_moving_anchor_truth must contain at least one method")

    def _fingerprint(traj: Sequence[Mapping[str, Any]]) -> tuple[tuple[float, float, float, float], ...]:
        items: list[tuple[float, float, float, float]] = []
        for entry in traj:
            if not isinstance(entry, Mapping):
                continue
            try:
                t = float(entry.get("t", 0.0))
                aid = str(entry.get("anchor_id", ""))
                px = round(float(entry.get("px", 0.0)), 6)
                py = round(float(entry.get("py", 0.0)), 6)
                pz = round(float(entry.get("pz", 0.0) or 0.0), 6)
            except (TypeError, ValueError):
                continue
            items.append((round(t, 6), aid, px, py, pz))  # type: ignore[arg-type]
        return tuple(sorted(items))

    fingerprints: dict[str, tuple[Any, ...]] = {}
    has_moving: list[str] = []
    for m, traj in method_moving_anchor_truth.items():
        fingerprints[m] = _fingerprint(traj)
        if len(fingerprints[m]) > 0:
            has_moving.append(m)

    has_any_moving = len(has_moving) > 0
    # 若所有方法都没有移动锚点（静态布局），自动通过。
    if not has_any_moving:
        return {
            "uniform": True,
            "has_moving": False,
            "method_count": len(method_moving_anchor_truth),
            "mismatches": [],
        }
    # 若有些方法有移动锚点真值，有些没有 → 不公平门禁违反。
    no_moving = [m for m in method_moving_anchor_truth if m not in has_moving]
    if no_moving:
        report = {
            "uniform": False,
            "has_moving": True,
            "method_count": len(method_moving_anchor_truth),
            "mismatches": no_moving,
        }
        if raise_on_violation:
            raise ValueError(
                "§8.1 R-D-C moving_anchor_truth_equality violation: some methods have moving "
                f"anchor truth, others do not; missing={no_moving}"
            )
        return report
    # 所有方法都有移动锚点真值：检查指纹是否一致。
    unique_fingerprints = set(fingerprints.values())
    uniform = len(unique_fingerprints) == 1
    mismatches: list[str] = []
    if not uniform:
        reference_method = next(iter(fingerprints))
        reference_fp = fingerprints[reference_method]
        for m, fp in fingerprints.items():
            if fp != reference_fp:
                mismatches.append(m)
    report = {
        "uniform": uniform,
        "has_moving": True,
        "method_count": len(method_moving_anchor_truth),
        "mismatches": mismatches,
    }
    if not uniform and raise_on_violation:
        raise ValueError(
            "§8.1 R-D-C moving_anchor_truth_equality violation: methods have different moving anchor "
            f"truth trajectories; mismatches={mismatches}"
        )
    return report


def assert_seed_required(
    trajectory_seeds: Mapping[str, Any] | Sequence[Any] | None,
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """§8.2.0 政策5 展位门禁：禁止"无种子人工单轨"或"不可复现遥操作未落盘"作为主表唯一证据。

    §8.2.0 L1388："**不允许**把「单次人工画的一条轨」或「不可复现的真人遥操作未落盘」当主表唯一证据。"
    §8.2.0 L1382 政策2："随机源须存在且可种子化"。

    本门禁断言每个场景/序列的轨迹种子 (trajectory_seed) 必须存在且为有限数值，
    杜绝无 seed 单轨（"人工画的一条轨"）或不可复现遥操作（无 seed 的真人数据）通过 sweep path。

    参数:
        trajectory_seeds: dict[seq_id -> seed] 或 list[seed] 形式；
            每个元素必须是有限数值或可数值化的种子；None / 空 / 缺失视为违规候选。
        raise_on_violation: True 时若任一序列无种子或种子非有限数值则 raise。

    返回:
        dict 含 all_seeded (bool)、n_sequences (int)、n_unseeded (int)、
              unseeded_seq_ids (list)、seed_types (dict[seq_id -> type_name])。
    """
    if trajectory_seeds is None:
        report = {
            "all_seeded": False,
            "n_sequences": 0,
            "n_unseeded": 0,
            "unseeded_seq_ids": ["<ROOT>"],
            "seed_types": {},
        }
        if raise_on_violation:
            raise ValueError(
                "§8.2.0 R8.2.0-P5 seed_required violation: trajectory_seeds is None; "
                "no seedable randomness source present (matches '单次人工画的一条轨' forbidden case)"
            )
        return report

    # 统一规整为 dict[seq_id -> seed]
    if isinstance(trajectory_seeds, Mapping):
        seed_dict = dict(trajectory_seeds)
    elif isinstance(trajectory_seeds, Sequence) and not isinstance(trajectory_seeds, (str, bytes)):
        seed_dict = {f"<seq[{i}]>": s for i, s in enumerate(trajectory_seeds)}
    else:
        # 单个种子对象（不是 list/dict）：视为单序列 ROOT
        seed_dict = {"<ROOT>": trajectory_seeds}

    if len(seed_dict) == 0:
        report = {
            "all_seeded": False,
            "n_sequences": 0,
            "n_unseeded": 0,
            "unseeded_seq_ids": ["<ROOT>"],
            "seed_types": {},
        }
        if raise_on_violation:
            raise ValueError(
                "§8.2.0 R8.2.0-P5 seed_required violation: trajectory_seeds is empty; "
                "no seedable randomness source present"
            )
        return report

    unseeded_seq_ids: list[str] = []
    seed_types: dict[str, str] = {}
    for seq_id, seed in seed_dict.items():
        seed_types[seq_id] = type(seed).__name__
        # 接受 int/float/可数值化字符串（如 "42"）；拒绝 None / 空字符串 / NaN/Inf。
        is_finite_numeric = False
        if isinstance(seed, bool):
            # bool 是 int 的子类，但单独 flag 不算"种子值"
            is_finite_numeric = False
        elif isinstance(seed, (int, float)):
            is_finite_numeric = math.isfinite(float(seed))
        elif isinstance(seed, str):
            s = seed.strip()
            if s:
                try:
                    v = float(s)
                    is_finite_numeric = math.isfinite(v)
                except ValueError:
                    is_finite_numeric = False
        if not is_finite_numeric:
            unseeded_seq_ids.append(seq_id)

    all_seeded = len(unseeded_seq_ids) == 0
    report = {
        "all_seeded": all_seeded,
        "n_sequences": len(seed_dict),
        "n_unseeded": len(unseeded_seq_ids),
        "unseeded_seq_ids": unseeded_seq_ids,
        "seed_types": seed_types,
    }
    if not all_seeded and raise_on_violation:
        raise ValueError(
            "§8.2.0 R8.2.0-P5 seed_required violation: trajectory lacks seedable randomness for some "
            f"sequences; unseeded={unseeded_seq_ids}; n_sequences={len(seed_dict)}"
        )
    return report


def assert_seed_decoupling(
    trajectory_seed: Any,
    nlos_seed: Any,
    async_seed: Any,
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """§8.2.0 政策6 展位门禁：轨迹随机与 NLOS/异步随机宜解耦种子（可复述）。

    §8.2.0 L1389："轨迹随机与 NLOS/异步随机**宜解耦种子**（可复述）：
    便于诊断名次来自运动还是来自 NLOS/时间。"

    解耦定义（修订自原 fully-coupled-only 标准）：
    - 完全耦合：三者都不为 None 且数值完全相同；
    - 部分耦合：任意两者都不为 None 且数值相同（如 trajectory==nlos 且 async 不同）。
    §8.2.0 L1389 明确要求"便于诊断名次来自运动还是来自 NLOS/时间"——
    任一两两耦合（含 trajectory==nlos 这种部分耦合）都会污染对应诊断轴，
    因此门禁须对**任一两两碰撞**触发 raise，而非仅三者完全相同。

    参数:
        trajectory_seed: 轨迹种子（数值或可数值化字符串）。
        nlos_seed: NLOS 扰动种子。
        async_seed: 异步扰动种子。
        raise_on_violation: True 时若有任一两两 collision 则 raise。

    返回:
        dict 含 decoupled (bool)、n_unique_seeds (int)、
              seeds (dict[field -> float])、collisions (list[str])。
    """

    def _coerce_seed(s: Any) -> float | None:
        if isinstance(s, bool):
            return None
        if isinstance(s, (int, float)):
            v = float(s)
            return v if math.isfinite(v) else None
        if isinstance(s, str):
            try:
                v = float(s.strip())
                return v if math.isfinite(v) else None
            except ValueError:
                return None
        return None

    t_seed = _coerce_seed(trajectory_seed)
    n_seed = _coerce_seed(nlos_seed)
    a_seed = _coerce_seed(async_seed)

    seeds = {"trajectory": t_seed, "nlos": n_seed, "async": a_seed}
    # 缺失种子视为 None；若三者均 None 则视为解耦（无种子无法耦合）。
    present = [v for v in seeds.values() if v is not None]
    unique_present = set(present)
    n_unique_seeds = len(unique_present)

    # 耦合定义：三者都不为 None 且数值完全相同。
    fully_coupled = (
        t_seed is not None and n_seed is not None and a_seed is not None
        and t_seed == n_seed == a_seed
    )
    # 部分耦合：任意两者都不为 None 且数值相同。
    pairs = [("trajectory", "nlos", t_seed, n_seed),
             ("trajectory", "async", t_seed, a_seed),
             ("nlos", "async", n_seed, a_seed)]
    collisions: list[str] = []
    for name_a, name_b, va, vb in pairs:
        if va is not None and vb is not None and va == vb:
            collisions.append(f"{name_a}=={name_b}({va})")
    # 解耦条件：无任何两两碰撞。
    # §8.2.0 L1389 要求"诊断名次来自运动还是 NLOS/时间"——
    # 任一 collision 都使对应诊断轴失效，故门禁对任一 collision raise。
    decoupled = (len(collisions) == 0)

    report = {
        "decoupled": decoupled,
        "n_unique_seeds": n_unique_seeds,
        "seeds": seeds,
        "collisions": collisions,
        "fully_coupled": fully_coupled,
    }
    if not decoupled and raise_on_violation:
        # 报告 collisions（含部分/完全耦合特征），便于诊断定位。
        raise ValueError(
            "§8.2.0 R8.2.0-P6 seed_decoupling violation: trajectory/NLOS/async seeds have "
            f"collision(s) {collisions}; cannot independently attribute ranking "
            "contributions to motion vs NLOS vs async per spec L1389 "
            "(partial coupling also contaminates the corresponding diagnostic axis)"
        )
    return report


def assert_anchor_3d_declaration(
    anchor_layout: Mapping[str, Any],
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """§8.1 L1361 三维测距另声明门禁：3D 锚点布局必须显式声明最小锚数与垂直分布。

    §8.1 L1361（spec）："...三维测距须另声明最小锚数与垂直分布。"

    平面题（默认 2D）锚点坐标仅 (x, y)，本门禁自动通过。
    若任一锚点含 z 坐标（len(coords) == 3），视为 3D 布局，须同时满足：
      1. anchor_layout 含 'min_anchor_count_3d' 字段且 ≥ 4（3D 测距至少 4 锚）。
      2. anchor_layout 含 'vertical_distribution' 字段（dict 或 list）描述 z 方向分布
         （如 {"z_min": ..., "z_max": ..., "z_span": ...}），禁止空白/缺失。

    参数:
        anchor_layout: 锚点布局字典，含 'anchor_positions'（list of (x[,y[,z]])）。
        raise_on_violation: True 时若 3D 但缺字段则 raise。

    返回:
        dict 含 is_3d (bool)、declared (bool)、min_anchor_count (int|None)、
              vertical_distribution_present (bool)、n_anchors_with_z (int)、z_extent (tuple|None)、
              reasons (list[str])、passed (bool)。
    """
    if not isinstance(anchor_layout, Mapping):
        raise TypeError(f"anchor_layout must be a Mapping, got {type(anchor_layout).__name__}")

    positions = list(anchor_layout.get("anchor_positions") or [])
    n_with_z = 0
    n_with_z_unparseable = 0  # §8.2.1 L1403 Lz≤5m 偷懒修复：含 z 但 z 非数值的 anchor 计数
    z_values: list[float] = []
    for p in positions:
        if not isinstance(p, (list, tuple)):
            continue
        if len(p) >= 3:
            n_with_z += 1
            try:
                z_values.append(float(p[2]))
            except (TypeError, ValueError):
                # §8.2.1 L1403 修复（先前 silent-pass 偷懒）：3D 锚点含 z 但 z 非数值
                # 是数据完整性破坏——is_3d=True 但 z_values 不增长，导致下方
                # `if is_3d and z_values:` 不触发，Lz≤5m 门禁 silent-pass。
                # 修复：计数 + reasons 标记，使下方 declared=False 路径触发 raise。
                n_with_z_unparseable += 1

    is_3d = n_with_z > 0
    declared = True
    min_anchor_count: int | None = None
    vertical_distribution_present = False
    reasons: list[str] = []

    # §8.2.1 L1403 修复（偷懒补丁）：3D 锚点含 z 但 z 非数值时，Lz≤5m 门禁不能 silent-pass。
    # 数据完整性破坏须显形，不应静默跳过 Lz 检查后默认 pass。
    if n_with_z_unparseable > 0:
        declared = False
        reasons.append(
            f"anchor_3d_declaration §8.2.1 L1403: {n_with_z_unparseable}/{n_with_z} 个 anchor "
            f"含 z 字段但 z 坐标非数值（无法转 float）；Lz≤5m 门禁无法验证，"
            "且视为数据完整性破坏（3D 布局须提供有效数值 z）→ declared=False，不应 silent-pass"
        )

    if is_3d:
        # 1. min_anchor_count_3d 字段
        mac_raw = anchor_layout.get("min_anchor_count_3d")
        if mac_raw is None:
            declared = False
            reasons.append(
                "anchor_3d_declaration: 3D 布局缺失 'min_anchor_count_3d' 字段"
            )
        else:
            try:
                mac = int(mac_raw)
                min_anchor_count = mac
                if mac < 4:
                    declared = False
                    reasons.append(
                        f"anchor_3d_declaration: min_anchor_count_3d={mac} < 4 "
                        "(3D 测距至少需要 4 锚)"
                    )
            except (TypeError, ValueError):
                declared = False
                reasons.append(
                    f"anchor_3d_declaration: min_anchor_count_3d 非整数: {mac_raw!r}"
                )

        # 2. vertical_distribution 字段
        vd = anchor_layout.get("vertical_distribution")
        if vd is None:
            declared = False
            reasons.append(
                "anchor_3d_declaration: 3D 布局缺失 'vertical_distribution' 字段"
            )
        elif isinstance(vd, Mapping):
            if len(vd) == 0:
                declared = False
                reasons.append(
                    "anchor_3d_declaration: 'vertical_distribution' 为空字典"
                )
            else:
                vertical_distribution_present = True
        elif isinstance(vd, (list, tuple)):
            if len(vd) == 0:
                declared = False
                reasons.append(
                    "anchor_3d_declaration: 'vertical_distribution' 为空列表"
                )
            else:
                vertical_distribution_present = True
        else:
            declared = False
            reasons.append(
                f"anchor_3d_declaration: 'vertical_distribution' 类型不支持: {type(vd).__name__}"
            )

    z_extent: tuple[float, float] | None = None
    if z_values:
        z_extent = (min(z_values), max(z_values))

    # §8.2.1 L1405 (table row 2): 3D 题竖直跨度 L_z ≤ 5m（同层为主）。
    # 超过 5m 视为多层大高差 → 出域；与 L1361 vertical_distribution 合并审计。
    L_Z_MAX_3D = 5.0
    l_z_in_range = True
    if is_3d and z_values:
        l_z_span = max(z_values) - min(z_values)
        l_z_in_range = l_z_span <= L_Z_MAX_3D
        if not l_z_in_range:
            reasons.append(
                f"anchor_3d_declaration §8.2.1 L1405: anchor L_z={l_z_span:.3f}m > "
                f"{L_Z_MAX_3D}m; 3D 题仅支持同层小高差，多层未建模 → 出域"
            )

    passed = (not is_3d or declared) and l_z_in_range
    report = {
        "is_3d": is_3d,
        "declared": declared,
        "min_anchor_count": min_anchor_count,
        "vertical_distribution_present": vertical_distribution_present,
        "n_anchors_with_z": n_with_z,
        "z_extent": z_extent,
        "l_z_in_range": l_z_in_range,
        "reasons": reasons,
        "passed": passed,
    }
    if not passed and raise_on_violation:
        raise ValueError(
            "§8.1 L1361 / §8.2.1 L1405 anchor_3d_declaration violation: 3D 测距须显式声明 "
            "min_anchor_count_3d (≥4)、vertical_distribution，且 L_z ≤ 5m; "
            f"reasons={reasons}; is_3d={is_3d}; n_with_z={n_with_z}"
        )
    return report


def assert_trajectory_collection_multi_sample(
    scene_tasks: Sequence[Mapping[str, Any]] | None,
    *,
    min_samples: int = 2,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """§8.2.0 政策1 多样本门禁：禁单条死脚本定全序；sweep 须含 ≥ min_samples 条轨迹。

    §8.2.0 L1381 政策1："**多样本、可复现的轨迹集合**，而不是单条死脚本定全序。"

    §8.2.0 L1394 放松则伤："单轨死脚本 → 可检验性与 5、6"。

    参数:
        scene_tasks: 任务序列（每个任务代表一条轨迹/一个场景）。
        min_samples: 最少任务数；默认 2（≥2 才算"多样本"）。
        raise_on_violation: True 时若 sweep 任务数 < min_samples 则 raise。

    返回:
        dict 含 multi_sample (bool)、n_tasks (int)、min_samples (int)。
    """
    if not isinstance(scene_tasks, Sequence) or isinstance(scene_tasks, (str, bytes)):
        raise TypeError(f"scene_tasks must be a Sequence, got {type(scene_tasks).__name__}")
    n_tasks = len(scene_tasks) if scene_tasks is not None else 0
    multi_sample = bool(n_tasks >= min_samples)
    report = {
        "multi_sample": multi_sample,
        "n_tasks": n_tasks,
        "min_samples": int(min_samples),
    }
    if not multi_sample and raise_on_violation:
        raise ValueError(
            f"§8.2.0 政策1 multi_sample violation: sweep 含 {n_tasks} 个任务 < "
            f"min_samples={min_samples}; 单条死脚本定全序禁用（spec L1381）"
        )
    return report


def assert_trajectory_generator_pol_2(
    generator_metadata: Mapping[str, Any] | None,
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """§8.2.0 政策2 随机源可种子化门禁：轨迹生成器须对路径/转向/停走/速度包络之一做种子化变异。

    §8.2.0 L1382 政策2："**随机源须存在且可种子化**（§27.1）：至少对路径形状、转向相位、
    停走时刻、速度包络之一做种子化变异；或等价地使用足够大的固定轨迹库 + 固定划分。"

    generator_metadata 形式：
        {
            "seeded_dimensions": ["path_shape", "turn_phase", "stop_time", "speed_envelope"],
            "is_trajectory_lib": False,
            "lib_size": None,
            "seed_param_present": True,
        }

    通过条件 (满足任一)：
        A. seed_param_present=True AND seeded_dimensions 与
           {"path_shape","turn_phase","stop_time","speed_envelope"} 的交集非空
        B. is_trajectory_lib=True AND lib_size >= min_lib_size (默认 30)

    参数:
        generator_metadata: 描述轨迹生成器种子化维度的元数据字典。
        raise_on_violation: True 时若既不满足 A 也不满足 B 则 raise。

    返回:
        dict 含 pol_2_satisfied (bool)、seeded_dims_present (bool)、
              n_seeded_dims (int)、is_lib_pool (bool)、lib_size (int|None)、reasons (list[str])。
    """
    if not isinstance(generator_metadata, Mapping):
        if generator_metadata is None:
            report = {
                "pol_2_satisfied": False,
                "seeded_dims_present": False,
                "n_seeded_dims": 0,
                "is_lib_pool": False,
                "lib_size": None,
                "reasons": ["generator_metadata 为 None；无法判定"],
            }
            if raise_on_violation:
                raise ValueError(
                    "§8.2.0 政策2 seed_pol_2 violation: generator_metadata 缺失"
                )
            return report
        raise TypeError(
            f"generator_metadata must be a Mapping, got {type(generator_metadata).__name__}"
        )

    required_dims = {"path_shape", "turn_phase", "stop_time", "speed_envelope"}
    seeded_dims = set(generator_metadata.get("seeded_dimensions") or [])
    n_seeded = len(seeded_dims & required_dims)
    seeded_dims_present = n_seeded >= 1
    seed_param_present = bool(generator_metadata.get("seed_param_present", False))

    is_lib_pool = bool(generator_metadata.get("is_trajectory_lib", False))
    lib_size_raw = generator_metadata.get("lib_size")
    lib_size = int(lib_size_raw) if isinstance(lib_size_raw, (int, float)) else None
    min_lib_size = 30  # "足够大的固定轨迹库"
    lib_size_ok = is_lib_pool and lib_size is not None and lib_size >= min_lib_size

    pol_2_satisfied = (
        (seed_param_present and seeded_dims_present) or lib_size_ok
    )
    reasons: list[str] = []
    if not pol_2_satisfied:
        if not seed_param_present:
            reasons.append("generator_metadata.seed_param_present=False")
        if not seeded_dims_present:
            reasons.append(
                f"未对路径形状/转向相位/停走时刻/速度包络任何一项做种子化变异 "
                f"(seeded_dimensions={list(seeded_dims)}; required≥1 of {sorted(required_dims)})"
            )
        if not lib_size_ok:
            reasons.append(
                f"非固定轨迹库路径或库大小不足 (is_trajectory_lib={is_lib_pool}, "
                f"lib_size={lib_size}, min={min_lib_size})"
            )

    report = {
        "pol_2_satisfied": pol_2_satisfied,
        "seeded_dims_present": seeded_dims_present,
        "n_seeded_dims": n_seeded,
        "seeded_dims": sorted(seeded_dims & required_dims),
        "is_lib_pool": is_lib_pool,
        "lib_size": lib_size,
        "reasons": reasons,
    }
    if not pol_2_satisfied and raise_on_violation:
        raise ValueError(
            "§8.2.0 政策2 seed_pol_2 violation: 轨迹生成器随机源不可种子化或未覆盖任一变异维度: "
            f"reasons={reasons}"
        )
    return report


def assert_trajectory_generator_pol_4(
    generator_metadata: Mapping[str, Any] | None,
    *,
    raise_on_violation: bool = True,
) -> dict[str, Any]:
    """§8.2.0 政策4 允许的生成族门禁：trajectory_generator 须落入政策4 列举的允许族之一。

    §8.2.0 L1384-1387 政策4："**允许**的生成族（协议写死一种或组合）：
       - 种子化随机游走 / 随机路点 + 光滑连接；
       - 参数化样条族，**系数或控制点由种子抽样**；
       - 有限轨迹库随机抽取（训练/测试划分防泄漏，§9.2）。"

    generator_metadata 形式：
        {
            "generator_family": "seeded_random_walk" | "parameterized_spline" | "trajectory_lib" | ...,
            "is_seedable": True,
            "trajectory_lib_split_protocol": "train_test_holdout",
        }

    参数:
        generator_metadata: 描述轨迹生成器族类的元数据字典。
        raise_on_violation: True 时若 generator_family 不在允许族列表则 raise。

    返回:
        dict 含 pol_4_satisfied (bool)、generator_family (str|None)、
              allowed_families (list[str])、reasons (list[str])。
    """
    if not isinstance(generator_metadata, Mapping):
        if generator_metadata is None:
            report = {
                "pol_4_satisfied": False,
                "generator_family": None,
                "allowed_families": [],
                "reasons": ["generator_metadata 为 None；无法判定"],
            }
            if raise_on_violation:
                raise ValueError(
                    "§8.2.0 政策4 generator_family violation: generator_metadata 缺失"
                )
            return report
        raise TypeError(
            f"generator_metadata must be a Mapping, got {type(generator_metadata).__name__}"
        )

    allowed_families = [
        "seeded_random_walk",
        "random_waypoint_smoothed",
        "parameterized_spline",
        "trajectory_lib",
        "combined",
    ]
    fam = generator_metadata.get("generator_family")
    fam_str = str(fam) if fam is not None else None
    pol_4_satisfied = fam_str in allowed_families
    is_seedable = bool(generator_metadata.get("is_seedable", True))
    reasons: list[str] = []
    if not pol_4_satisfied:
        reasons.append(
            f"generator_family={fam_str!r} 不在 §8.2.0 政策4 允许族列表 "
            f"{allowed_families}"
        )
    if not is_seedable:
        reasons.append("generator_family claims 不可种子化 (is_seedable=False)")

    report = {
        "pol_4_satisfied": pol_4_satisfied and is_seedable,
        "generator_family": fam_str,
        "allowed_families": list(allowed_families),
        "is_seedable": is_seedable,
        "reasons": reasons,
    }
    if not (pol_4_satisfied and is_seedable) and raise_on_violation:
        raise ValueError(
            "§8.2.0 政策4 generator_family violation: "
            f"trajectory generator 不属于允许的生成族: reasons={reasons}"
        )
    return report
