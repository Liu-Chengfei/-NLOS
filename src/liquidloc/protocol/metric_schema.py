"""冻结指标元数据与排序合同模块。

文件职责：
  定义所有指标的元数据（单位、方向、分组）和排序，
  并校验 configs/base/metrics.yaml 是否与冻结协议模式精确匹配。
  核心目标是防止指标定义和排序在运行时漂移。

本文件绝对不负责：
  不定义指标口径（指标口径由 configs/base/metrics.yaml 定义）。
  不计算指标值。
  不修改指标配置文件。

核心数据流：
    metrics.yaml → _validate_metrics_config → 校验通过 → get_metric_meta / get_metric_order

上游依赖：
  liquidloc.common.config_utils（load_yaml_config 加载 YAML 配置）、
  liquidloc.common.types（MetricDirection 指标方向类型）

下游调用者：
  protocol/result_schema.py（使用 get_metric_order 校验结果载荷）、
  metrics/（使用 get_metric_meta 获取指标元数据）、
  analysis/（使用 get_metric_order 排序指标）、
  plotting/（使用 get_metric_meta 标注图表）

输入对象定义：
  无外部输入，所有数据来自模块内部常量和 metrics.yaml

输出对象定义：
  - get_metric_meta    获取所有指标的元数据字典
  - get_metric_order   获取冻结的指标顺序列表

核心变量定义：
  - _METRIC_META              指标元数据注册表
  - _CONFIG_ROOT              配置文件根目录
  - _METRICS_CONFIG_PATH      指标配置文件路径
  - _METRIC_GROUP_FIELD_MAP   指标分组到 YAML 字段名的映射
  - _METRIC_ORDER             冻结的指标顺序列表

关键设计决策：
  - 指标元数据在模块内部硬编码，与 metrics.yaml 做双向校验。
  - metrics.yaml 的指标顺序必须与 _METRIC_META 的键顺序精确匹配。
  - 每个指标的分组（primary/secondary/mechanism/runtime）必须与
    metrics.yaml 中的分组字段精确匹配。
  - 指标口径（单位、方向）冻结，禁止为图表方便改定义。
"""

from __future__ import annotations  # 允许类型注解中引用尚未定义的类型。

from typing import TypedDict, cast  # 结构化字典类型和类型转换。

from liquidloc.common.config_utils import find_project_root, load_yaml_config  # 加载 YAML 配置文件和项目根查找器。
from liquidloc.common.types import MetricDirection, _METRIC_DIRECTIONS as _VALID_METRIC_DIRECTIONS  # 指标方向类型别名及合法方向集合（单一真相源）。


class MetricMeta(TypedDict):
    """单个指标的元数据结构，替代 dict[str, str | MetricDirection]。

    使用 TypedDict 从类型层面约束每个指标必须包含 unit、direction、group 三个字段，
    且 direction 的值必须属于 MetricDirection 允许的三个值。
    """
    unit: str  # 指标单位，如 "m"、"ratio"、"corr"、"ms"、"count"、"MB"。
    direction: MetricDirection  # 指标方向：lower_is_better / higher_is_better / neutral。
    group: str  # 指标分组：primary / secondary / mechanism / runtime。

# 指标元数据注册表：每个指标包含单位、方向和分组。
# 分组说明：
#   primary    - 主要指标（RMSE、p95、failure_rate），用于实验结论判定
#   secondary  - 次要指标（p99、MAE、ATE、RPE、yaw_rmse、yaw_p95），提供补充信息
#   mechanism  - 机制指标（risk_error_corr、coverage、bias_alignment、corr_scaling_error），
#                衡量模型内部机制是否有效
#   runtime    - 运行时指标（latency、params、RAM），衡量部署可行性
_METRIC_META: dict[str, MetricMeta] = {
    "rmse": {"unit": "m", "direction": "lower_is_better", "group": "primary"},  # 均方根误差，主要指标。
    "mean_rmse": {"unit": "m", "direction": "lower_is_better", "group": "primary"},  # 跨序列 RMSE 均值（准则 36 帧级 mean 跨序列聚合）.
    "std_rmse": {"unit": "m", "direction": "neutral", "group": "primary"},  # 跨序列 RMSE 样本标准差（准则 36 帧级 std 跨序列聚合，Bessel 校正）.
    "p95": {"unit": "m", "direction": "lower_is_better", "group": "primary"},  # 95 百分位误差，主要指标。
    "failure_rate": {"unit": "ratio", "direction": "lower_is_better", "group": "primary"},  # 失效率，主要指标。
    "p99": {"unit": "m", "direction": "lower_is_better", "group": "secondary"},  # 99 百分位误差，次要指标。
    "mae": {"unit": "m", "direction": "lower_is_better", "group": "secondary"},  # 平均绝对误差，次要指标。
    "ate": {"unit": "m", "direction": "lower_is_better", "group": "secondary"},  # 绝对轨迹误差，次要指标。
    "rpe": {"unit": "m", "direction": "lower_is_better", "group": "secondary"},  # 相对位姿误差，次要指标。
    "yaw_rmse": {"unit": "rad", "direction": "lower_is_better", "group": "secondary"},  # 航向角均方根误差，次要指标。
    "yaw_p95": {"unit": "rad", "direction": "lower_is_better", "group": "secondary"},  # 航向角95百分位误差，次要指标。
    "risk_error_corr": {"unit": "corr", "direction": "higher_is_better", "group": "mechanism"},  # 风险-误差相关性，机制指标，越高越好。
    "coverage": {"unit": "ratio", "direction": "higher_is_better", "group": "mechanism"},  # 覆盖率，机制指标，越高越好。
    "bias_alignment": {"unit": "corr", "direction": "higher_is_better", "group": "mechanism"},  # 偏置对齐度，机制指标，越高越好。
    "corr_scaling_error": {  # 缩放-误差相关性，机制指标，越高越好。
        "unit": "corr",
        "direction": "higher_is_better",
        "group": "mechanism",
    },
    "latency_mean": {"unit": "ms", "direction": "lower_is_better", "group": "runtime"},  # 平均延迟，运行时指标。
    "latency_p50": {"unit": "ms", "direction": "lower_is_better", "group": "runtime"},  # 50 百分位延迟，运行时指标。
    "latency_p95": {"unit": "ms", "direction": "lower_is_better", "group": "runtime"},  # 95 百分位延迟，运行时指标。
    "params": {"unit": "count", "direction": "lower_is_better", "group": "runtime"},  # 参数量，运行时指标。
    "ram_peak": {"unit": "MB", "direction": "lower_is_better", "group": "runtime"},  # 峰值内存，运行时指标。
}

# 运行时校验：确保每个指标的 direction 值都在合法集合内。
for _metric_name, _metric_meta in _METRIC_META.items():
    _direction = _metric_meta["direction"]  # TypedDict 必填字段，缺失时 KeyError 比默认值更精确。
    if _direction not in _VALID_METRIC_DIRECTIONS:
        raise ValueError(
            f"metric '{_metric_name}' has invalid direction '{_direction}'; "
            f"must be one of {sorted(_VALID_METRIC_DIRECTIONS)}"
        )

_CONFIG_ROOT = (find_project_root() / "configs" / "base").resolve()  # 配置文件根目录（resolve 防止符号链接/..穿越）。
_METRICS_CONFIG_PATH = _CONFIG_ROOT / "metrics.yaml"  # 指标配置文件路径。
_METRIC_GROUP_FIELD_MAP = {  # 指标分组到 metrics.yaml 中对应字段名的映射。
    "primary": "primary_metrics",  # 主要指标字段名。
    "secondary": "secondary_metrics",  # 次要指标字段名。
    "mechanism": "mechanism_metrics",  # 机制指标字段名。
    "runtime": "runtime_metrics",  # 运行时指标字段名。
}


def _validate_metrics_config() -> list[str]:
    """校验 metrics.yaml 是否与冻结协议模式精确匹配。

    逐组校验：每个分组字段必须是非空列表，列表中的每个指标
    必须在 _METRIC_META 中注册且分组一致，最终顺序必须与
    _METRIC_META 的键顺序精确匹配。

    返回：
        list[str]: 校验通过的指标顺序列表。

    异常：
        ValueError: 分组字段不是非空列表、指标未注册、分组不一致或顺序不匹配时抛出。
    """
    cfg = load_yaml_config(_METRICS_CONFIG_PATH)  # 加载指标配置。
    # 检测 YAML 中是否存在未知分组字段（双向校验的 YAML→schema 方向）。
    known_field_names = set(_METRIC_GROUP_FIELD_MAP.values())
    unknown_fields = [key for key in cfg if key not in known_field_names]
    if unknown_fields:
        raise ValueError(
            f"metrics config contains unknown metric group fields: {unknown_fields}; "
            f"expected only {sorted(known_field_names)}"
        )
    configured_order: list[str] = []  # 存放配置文件中的指标顺序。
    for group_name, field_name in _METRIC_GROUP_FIELD_MAP.items():  # 逐组校验。
        configured_metrics = cfg.get(field_name)  # 读取该分组的指标列表。
        if not isinstance(configured_metrics, list) or not configured_metrics:  # 必须是非空列表。
            raise ValueError(f"metrics config field {field_name} must be a non-empty list")
        for metric_name in configured_metrics:  # 逐个校验指标。
            if metric_name not in _METRIC_META:  # 指标必须在注册表中。
                raise ValueError(
                    f"metrics config references unknown metric '{metric_name}' in {field_name}"
                )
            metric_group = _METRIC_META[metric_name]["group"]  # 读取注册表中的分组。
            if metric_group != group_name:  # 分组必须与字段名一致。
                raise ValueError(
                    f"metrics config places '{metric_name}' in {field_name}, "
                    f"but schema group is '{metric_group}'"
                )
        configured_order.extend(configured_metrics)  # 追加到顺序列表。
    schema_order = list(_METRIC_META.keys())  # 注册表中的键顺序即为冻结顺序。
    if configured_order != schema_order:  # 顺序必须精确匹配。
        raise ValueError(
            "configs/base/metrics.yaml order must match protocol metric schema exactly: "
            f"{schema_order}"
        )
    return configured_order  # 返回校验通过的指标顺序。


_METRIC_ORDER = _validate_metrics_config()  # 模块加载时即校验并冻结指标顺序。
_PRIMARY_GROUP_NAME: str = "primary"  # 显式定义 primary 分组名，避免依赖 dict 插入顺序。
_PRIMARY_METRICS: list[str] = [name for name in _METRIC_ORDER if _METRIC_META[name]["group"] == _PRIMARY_GROUP_NAME]


def get_metric_meta() -> dict[str, MetricMeta]:
    """获取所有指标的元数据字典。

    返回的字典键为指标名，值为包含 unit、direction、group 的 MetricMeta。
    每次调用都返回新的字典副本，防止外部修改冻结元数据。

    返回：
        dict[str, MetricMeta]: 指标名到元数据的映射。
    """
    return {name: cast(MetricMeta, dict(meta)) for name, meta in _METRIC_META.items()}  # 返回副本，cast 保证类型标注与 MetricMeta 一致。


def get_primary_metrics() -> list[str]:
    """获取 primary_metrics 列表，与 metrics.yaml 中的定义精确匹配。

    返回：
        list[str]: 主指标名称列表。
    """
    return list(_PRIMARY_METRICS)


def get_metric_order() -> list[str]:
    """获取冻结的指标顺序列表。

    返回的列表与 metrics.yaml 中的顺序精确匹配。
    每次调用都返回新的列表副本，防止外部修改冻结顺序。

    返回：
        list[str]: 指标名称列表，按冻结顺序排列。
    """
    return list(_METRIC_ORDER)  # 返回副本。


from liquidloc.common.tee_logger import print_dict  # 局部导入参数打印工具。

print_dict(  # 打印本模块全部冻结常量，供审计核对。
    {
        "_METRIC_META": _METRIC_META,
        "_METRIC_ORDER": _METRIC_ORDER,
        "_PRIMARY_METRICS": _PRIMARY_METRICS,
        "_METRIC_GROUP_FIELD_MAP": _METRIC_GROUP_FIELD_MAP,
        "_METRICS_CONFIG_PATH": str(_METRICS_CONFIG_PATH),
    },
    "metric_schema.py 常量",
    prefix="[配置]",
)
