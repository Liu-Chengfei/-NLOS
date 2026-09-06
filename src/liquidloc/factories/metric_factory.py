"""指标工厂模块。

职责
----
把指标名转换成一个可调用的指标计算器对象。
从协议元数据里读取单位、方向和分组信息，避免每个调用点自己拼字段。

上游依赖
--------
- ``liquidloc.common.types.MetricRow``         — 指标行数据类
- ``liquidloc.protocol.metric_schema.get_metric_meta``  — 指标元数据协议源

下游调用者
----------
- ``liquidloc.pipelines.*``  — 流水线在评估阶段通过本模块创建指标计算器
- ``scripts/`` 下的评估脚本
- ``tests/factories/*``  — 工厂层单元测试

核心变量
--------
- ``_ScaffoldMetricCalculator``  — 基于元数据的轻量指标计算器数据类
- ``create_metric_calculator``    — 对外暴露的工厂函数
"""

from __future__ import annotations  # 允许后面的类型注解延迟求值，避免前向引用问题。

from collections.abc import Mapping  # 限制 cfg 只接受映射，避免静默吞掉错误结构。
from dataclasses import dataclass, field  # dataclass 用来定义轻量数据类；field 用来声明带默认值的字段。
from typing import Any  # 允许配置字段保持灵活类型，不强制具体类型。

from liquidloc.common.types import MetricRow  # 指标行数据类，工厂最终返回的行对象。
from liquidloc.common.validation import coerce_finite_scalar  # 统一数值校验+转换入口，拒绝 NaN/Inf/bool/str，与 model_factory.py / estimator_factory.py 已修复模式对齐。
from liquidloc.protocol.metric_schema import get_metric_meta  # 从协议层读取指标元数据（单位、方向、分组等）。


@dataclass(slots=True)  # slots=True 减少实例内存开销，禁止动态添加属性。
class _ScaffoldMetricCalculator:
    """基于元数据返回单个指标行的轻量计算器。

    该计算器不做复杂运算，只负责把一个原始数值包装成带协议元数据的
    ``MetricRow``，包括单位、方向和分组信息。

    属性
    ----
    name : str
        指标名字，必须在 ``get_metric_meta()`` 返回的键集合内。
    cfg : dict[str, Any]
        给指标计算器保留的配置副本，当前未使用但预留扩展空间。
    """

    name: str  # 指标名字，用于从协议元数据中查找对应条目。
    cfg: dict[str, Any] = field(default_factory=dict)  # 给指标计算器保留的配置副本，默认为空字典。

    def compute(self, value: float) -> MetricRow:
        """把一个数值包装成带元数据的 MetricRow。

        参数
        ----------
        value : float
            原始指标数值，必须是有限数值或可转为 float 的数值类型，
            不接受 NaN/Inf/bool/str/numpy.str_，与 ``MetricRow.__post_init__``
            的数值安全口径一致。

        返回
        -------
        MetricRow
            包含指标名、数值、单位、方向和分组的完整指标行。

        异常
        ------
        ValueError
            当 ``self.name`` 不在协议元数据中（构造路径被绕过），或 ``value``
            为 NaN/Inf 时抛出（值合同违例，与 estimator_factory.py 已修复模式对齐）。
        TypeError
            当 ``value`` 为非数值类型（bool/str/张量等）时抛出。
        """
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "name": self.name,
            "value_type": type(value).__name__,
        }, "_ScaffoldMetricCalculator.compute 入口参数")
        metric_meta = get_metric_meta()  # 读取全部可用指标的协议元数据字典。
        if self.name not in metric_meta:  # 名字不在元数据里说明构造路径被绕过，值合同违例用 ValueError 而非 KeyError。
            available = ', '.join(sorted(metric_meta))  # 拼出所有可用指标名，方便排错。
            raise ValueError(f'Unknown metric: {self.name}. Available: {available}')  # 值合同违例，与 estimator_factory.py / model_factory.py 已修复模式对齐。
        meta = metric_meta[self.name]  # 从协议元数据中取出该指标的完整元信息。
        coerced_value = coerce_finite_scalar(value, name=f'metric_factory.compute.{self.name}.value')  # 统一数值校验+转换，拒绝 NaN/Inf/bool/str，与 MetricRow.__post_init__ 口径一致。
        return MetricRow(
            metric=self.name,  # 指标名称。
            value=coerced_value,  # 已校验为有限 float，无需在工厂层裸 float()。
            unit=meta['unit'],  # 指标单位（如 "m"、"deg"），来自协议定义。
            direction=meta['direction'],  # 优化方向（如 'lower_is_better'、'higher_is_better'），来自协议定义。
            group=meta['group'],  # 指标分组（如 "primary"、"secondary"），来自协议定义。
        )  # 返回完整的指标行对象。


def create_metric_calculator(name: str, cfg: Mapping[str, Any] | None) -> _ScaffoldMetricCalculator:
    """根据指标名创建对应的指标计算器。

    参数
    ----------
    name : str
        指标名字，必须在 ``get_metric_meta()`` 返回的键集合内。
    cfg : Mapping[str, Any] | None
        指标配置，可以是映射或 ``None``，会被复制为字典保存。

    返回
    -------
    _ScaffoldMetricCalculator
        可调用的指标计算器实例，调用其 ``compute`` 方法即可得到 ``MetricRow``。

    异常
    ------
    ValueError
        当 *name* 不在协议元数据中时抛出（值合同违例，与
        :func:`model_factory.create_model`、:func:`estimator_factory.create_estimator` 同口径）。
    TypeError
        当 *cfg* 既不是映射也不是 ``None`` 时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "name": name,
        "cfg_type": type(cfg).__name__,
        "cfg_keys": list(cfg.keys()) if hasattr(cfg, "keys") else None,
    }, "create_metric_calculator 入口参数")
    metric_meta = get_metric_meta()  # 读取全部可用指标的协议元数据字典。
    if name not in metric_meta:  # 名字不在元数据里说明调用方传错了。值合同违例用 ValueError，与 create_model/create_estimator 同口径。
        available = ', '.join(sorted(metric_meta))  # 拼出所有可用指标名，方便排错。
        raise ValueError(f'Unknown metric: {name}. Available: {available}')  # 抛出 ValueError 提示可用列表。
    if cfg is None:  # 允许空配置，用空字典表示。
        normalized_cfg: dict[str, Any] = {}
    elif not isinstance(cfg, Mapping):  # 非映射配置必须显式拒绝，不能静默洗成空字典或任意 dict。
        raise TypeError(f"metric cfg must be a mapping or None, got {type(cfg).__name__}")
    else:
        normalized_cfg = dict(cfg)  # shallow copy: cfg is only read by compute(), never mutated
    return _ScaffoldMetricCalculator(name=name, cfg=normalized_cfg)
