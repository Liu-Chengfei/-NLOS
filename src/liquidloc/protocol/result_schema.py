"""结果载荷模式校验模块。

文件职责：
  定义实验结果的数据结构（SequenceResult、ExperimentResult、SummaryResult），
  并提供校验函数确保结果载荷符合冻结协议模式。
  核心目标是防止结果载荷的结构和指标值在运行时漂移。

本文件绝对不负责：
  不计算指标值。
  不修改结果数据。
  不定义指标口径（指标口径由 metric_schema 定义）。

核心数据流：
    实验结果 → validate_experiment_result → 校验通过 → 下游消费

上游依赖：
  liquidloc.common.types（SceneCode、SeqId 类型别名）、
  liquidloc.common.validation（require_keys 键完整性检查）、
  liquidloc.protocol.metric_schema（get_metric_order 指标顺序）

下游调用者：
  公共 API 导出，供下游消费者按需调用（pipelines/analysis/verify 等）。

输入对象定义：
  - SequenceResult    单序列结果 dataclass
  - ExperimentResult  实验结果 dataclass
  - SummaryResult     汇总结果 dataclass
  - dict              字典格式的结果

输出对象定义：
  - validate_sequence_result    校验单序列结果
  - validate_experiment_result  校验实验结果
  - validate_summary_result     校验汇总结果

核心变量定义：
  - _SEQUENCE_RESULT_KEYS    单序列结果必需键
  - _EXPERIMENT_RESULT_KEYS  实验结果必需键
  - _SUMMARY_RESULT_KEYS     汇总结果必需键
  - _METRIC_KEYS             冻结指标键元组

关键设计决策：
  - 结果载荷只允许包含协议定义的键，不允许额外字段。
  - 指标值必须是有限数值，不允许 NaN/inf。
  - 指标字典必须精确匹配冻结指标键集合，不多不少。
  - 汇总统计值支持嵌套字典和列表，但递归校验所有数值的有限性。
"""

from __future__ import annotations  # 允许类型注解中引用尚未定义的类型。

from copy import deepcopy  # 深拷贝，防止嵌套可变结构引用泄漏。

import math  # 用于 isfinite 检查数值有限性。
from collections.abc import Mapping  # 映射抽象基类，用于类型注解。
from dataclasses import dataclass, field  # 数据类工具。
from types import MappingProxyType  # 不可变字典视图，防止运行时篡改。
from typing import Any  # 允许类型注解里表示"任意类型"。

from liquidloc.common.types import SceneCode, SeqId  # 场景编码和序列 ID 类型别名。
from liquidloc.common.validation import is_integer, is_real, is_string_like, require_keys  # 整数/实数/字符串类型检查工具和键完整性检查。
from liquidloc.protocol.metric_schema import get_metric_order  # 获取冻结指标顺序。
from liquidloc.protocol.version import SNAPSHOT_VERSION  # 快照版本号，写入实验结果以标识快照格式兼容性。

_SEQUENCE_RESULT_KEYS = ("seq_id", "scene_code", "model_name", "metric_dict")  # 单序列结果必需键。
_EXPERIMENT_RESULT_KEYS = ("experiment_id", "model_results", "aggregate_metrics")  # 实验结果必需键。
_EXPERIMENT_RESULT_OPTIONAL_KEYS = ("snapshot_version", "non_main_table")  # 实验结果可选键（由 to_dict() 自动写入）。non_main_table=True 表示该实验属于消融/副表，不得进入主表比较。
_SUMMARY_RESULT_KEYS = ("case_refs", "summary_stats")  # 汇总结果必需键。
_METRIC_KEYS = tuple(get_metric_order())  # 冻结指标键元组，从 metric_schema 获取。


@dataclass(frozen=True, slots=True)
class SequenceResult:
    """单序列结果对象，承载一个序列在一种模型下的所有指标。

    metric_dict 在构造后用 MappingProxyType 包装，防止运行时篡改。
    """
    seq_id: SeqId
    scene_code: SceneCode
    model_name: str
    metric_dict: Mapping[str, int | float]  # 值接受 Python 原生 int/float 及 numpy 标量（校验通过即合法）。

    def __post_init__(self) -> None:
        """构造后校验所有字段，确保结果载荷符合冻结协议模式，并将可变字段转为不可变容器。"""
        _require_non_empty_string(self.seq_id, "sequence_result.seq_id")
        _require_non_empty_string(self.scene_code, "sequence_result.scene_code")
        _require_non_empty_string(self.model_name, "sequence_result.model_name")
        _validate_metric_dict(self.metric_dict, name="sequence_result.metric_dict")
        object.__setattr__(self, "metric_dict", MappingProxyType(self.metric_dict))

    def to_dict(self) -> dict[str, Any]:
        """把单序列结果转成普通字典，返回深拷贝。"""
        return {
            "seq_id": self.seq_id,
            "scene_code": self.scene_code,
            "model_name": self.model_name,
            "metric_dict": dict(self.metric_dict),
        }


@dataclass(frozen=True, slots=True)  # 不可变数据类，防止结果载荷被意外修改。
class ExperimentResult:
    """实验结果对象，承载一次实验的所有序列结果和聚合指标。

    model_results 在构造后转为 tuple，aggregate_metrics 用 MappingProxyType 包装，
    防止运行时篡改。

    属性：
        experiment_id: 实验标识符。
        model_results: 模型结果元组，每项为 SequenceResult 或字典。
        aggregate_metrics: 聚合指标映射，键为指标名，值为聚合值。
    """
    experiment_id: str  # 实验标识符。
    model_results: tuple[SequenceResult | dict[str, Any], ...]  # 模型结果元组。
    aggregate_metrics: Mapping[str, int | float]  # 聚合指标映射。值接受 Python 原生 int/float 及 numpy 标量。
    non_main_table: bool = False  # §19.3：消融/副表实验标记，True 时不得进入主表比较。

    def __post_init__(self) -> None:
        """构造后校验所有字段，确保结果载荷符合冻结协议模式，并将可变字段转为不可变容器。"""
        _require_non_empty_string(self.experiment_id, "experiment_result.experiment_id")
        _validate_model_results(self.model_results)
        _validate_metric_dict(self.aggregate_metrics, name="experiment_result.aggregate_metrics")
        object.__setattr__(self, "model_results", tuple(self.model_results))
        object.__setattr__(self, "aggregate_metrics", MappingProxyType(self.aggregate_metrics))

    def to_dict(self) -> dict[str, Any]:
        """把实验结果转成普通字典，递归处理嵌套的 SequenceResult，返回深拷贝。"""
        return {
            "snapshot_version": SNAPSHOT_VERSION,  # 快照版本号，标识快照格式兼容性。
            "experiment_id": self.experiment_id,  # 实验标识符。
            "non_main_table": bool(self.non_main_table),  # §19.3：非主表标识（消融/副表实验不得进入主表比较）。
            "model_results": [  # 逐个转换模型结果。
                result.to_dict() if isinstance(result, SequenceResult) else deepcopy(result)  # SequenceResult 转 dict，字典做深拷贝。
                for result in self.model_results
            ],
            "aggregate_metrics": dict(self.aggregate_metrics),  # 从 MappingProxyType 转回普通字典。
        }


@dataclass(frozen=True, slots=True)  # 不可变数据类，防止结果载荷被意外修改。
class SummaryResult:
    """汇总结果对象，承载多个实验案例的引用和汇总统计。

    case_refs 在构造后转为 tuple，summary_stats 用 MappingProxyType 包装，
    防止运行时篡改。

    属性：
        case_refs: 案例引用元组，每个元素是一个案例的标识字符串。
        summary_stats: 汇总统计映射，支持嵌套结构。
    """
    case_refs: tuple[str, ...] = field(default_factory=tuple)  # 案例引用元组，默认为空。
    summary_stats: Mapping[str, Any] = field(default_factory=dict)  # 汇总统计映射，默认为空。

    def __post_init__(self) -> None:
        """构造后校验所有字段，确保结果载荷符合冻结协议模式，并将可变字段转为不可变容器。"""
        _validate_case_refs(self.case_refs)
        _validate_summary_stats(self.summary_stats)
        object.__setattr__(self, "case_refs", tuple(self.case_refs))
        object.__setattr__(self, "summary_stats", MappingProxyType(self.summary_stats))

    def to_dict(self) -> dict[str, Any]:
        """把汇总结果转成普通字典，返回深拷贝。"""
        return {
            "case_refs": list(self.case_refs),  # 从 tuple 转回列表。
            "summary_stats": deepcopy(dict(self.summary_stats)),
        }


def _as_sequence_result_dict(sequence_result: SequenceResult | dict[str, Any]) -> dict[str, Any]:
    """把单序列结果统一成字典视图。

    参数：
        sequence_result: SequenceResult 对象或字典格式的结果。

    返回：
        dict[str, Any]: 结果的字典视图。

    异常：
        TypeError: 输入既不是 SequenceResult 也不是字典时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"sequence_result": sequence_result}, "_as_sequence_result_dict 入参", prefix="[配置]")
    if isinstance(sequence_result, SequenceResult):  # SequenceResult 对象。
        return sequence_result.to_dict()  # 先转成字典。
    if isinstance(sequence_result, dict):  # 字典格式。
        return deepcopy(sequence_result)  # 深拷贝，防止与原始共享引用。
    raise TypeError(  # 其他类型不接受。
        f"sequence_result must be a SequenceResult or dict, got {type(sequence_result).__name__}"
    )


def _as_experiment_result_dict(
    experiment_result: ExperimentResult | dict[str, Any],
) -> dict[str, Any]:
    """把实验结果统一成字典视图。

    参数：
        experiment_result: ExperimentResult 对象或字典格式的结果。

    返回：
        dict[str, Any]: 结果的字典视图。

    异常：
        TypeError: 输入既不是 ExperimentResult 也不是字典时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"experiment_result": experiment_result}, "_as_experiment_result_dict 入参", prefix="[配置]")
    if isinstance(experiment_result, ExperimentResult):  # ExperimentResult 对象。
        return experiment_result.to_dict()  # 先转成字典。
    if isinstance(experiment_result, dict):  # 字典格式。
        return deepcopy(experiment_result)  # 深拷贝，防止与原始共享引用。
    raise TypeError(  # 其他类型不接受。
        f"experiment_result must be an ExperimentResult or dict, got {type(experiment_result).__name__}"
    )


def _as_summary_result_dict(summary_result: SummaryResult | dict[str, Any]) -> dict[str, Any]:
    """把汇总结果统一成字典视图。

    参数：
        summary_result: SummaryResult 对象或字典格式的结果。

    返回：
        dict[str, Any]: 结果的字典视图。

    异常：
        TypeError: 输入既不是 SummaryResult 也不是字典时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"summary_result": summary_result}, "_as_summary_result_dict 入参", prefix="[配置]")
    if isinstance(summary_result, SummaryResult):  # SummaryResult 对象。
        return summary_result.to_dict()  # 先转成字典。
    if isinstance(summary_result, dict):  # 字典格式。
        return deepcopy(summary_result)  # 深拷贝，防止与原始共享引用。
    raise TypeError(  # 其他类型不接受。
        f"summary_result must be a SummaryResult or dict, got {type(summary_result).__name__}"
    )


def _require_non_empty_string(value: Any, name: str) -> None:
    """要求值必须是非空字符串。

    参数：
        value: 待校验的值。
        name: 字段名，用于构造错误信息。

    异常：
        TypeError: 值不是字符串时抛出。
        ValueError: 值是空白字符串时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"value": value, "name": name}, "_require_non_empty_string 入参", prefix="[配置]")
    if not is_string_like(value):  # 必须是字符串（含 numpy.str_）。
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():  # 不允许空白字符串。
        raise ValueError(f"{name} must be a non-empty string")


def _validate_metric_dict(metric_dict: Any, *, name: str) -> None:
    """校验指标字典是否与冻结指标键集合精确匹配，且所有值都是有限数值。

    本函数只做校验，不修改传入的 metric_dict。

    参数：
        metric_dict: 待校验的指标字典。
        name: 字段名，用于构造错误信息。

    异常：
        TypeError: 指标字典不是字典或包含非数值值时抛出。
        ValueError: 指标字典包含不支持指标或非有限值时抛出。
        KeyError: 指标字典缺少必需指标键时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"metric_dict": metric_dict, "name": name}, "_validate_metric_dict 入参", prefix="[配置]")
    if not isinstance(metric_dict, dict):  # 必须是字典。
        raise TypeError(f"{name} must be a dict, got {type(metric_dict).__name__}")
    require_keys(metric_dict, _METRIC_KEYS, name=name)  # 必须包含所有冻结指标键。
    unexpected = sorted(key for key in metric_dict if key not in _METRIC_KEYS)  # 找出多余键。
    if unexpected:  # 不允许额外指标。
        raise ValueError(f"{name} contains unsupported metrics: {unexpected}")
    for metric_name in _METRIC_KEYS:  # 逐个校验指标值。
        value = metric_dict[metric_name]  # 读取指标值。
        if not is_real(value):  # 必须是实数值型，排除 bool（含 numpy.bool_）和 complex。
            raise TypeError(f"{name}.{metric_name} must be numeric, got {type(value).__name__}")
        try:
            fv = float(value)
        except (OverflowError, ValueError, TypeError):
            raise ValueError(f"{name}.{metric_name} must be finite") from None
        if not math.isfinite(fv):
            raise ValueError(f"{name}.{metric_name} must be finite")


def _validate_model_results(model_results: Any) -> None:
    """校验模型结果列表中的每个元素是否通过单序列结果校验。

    参数：
        model_results: 模型结果列表或元组。

    异常：
        TypeError: 不是稳定顺序的列表/元组，或元素类型不合法时抛出。
        ValueError: 元素值不合法时抛出。
        KeyError: 元素必需键缺失时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"model_results": model_results}, "_validate_model_results 入参", prefix="[配置]")
    if not isinstance(model_results, (list, tuple)):  # 顶层必须是稳定顺序、可直接 JSON 落盘的序列。
        raise TypeError("experiment_result.model_results must be a list or tuple of sequence results")
    for index, model_result in enumerate(model_results):  # 逐个校验。
        try:
            validate_sequence_result(model_result)  # 校验单条结果。
        except ValueError as exc:
            raise ValueError(f"experiment_result.model_results[{index}]: {exc}") from exc
        except TypeError as exc:
            raise TypeError(f"experiment_result.model_results[{index}]: {exc}") from exc
        except KeyError as exc:
            raise KeyError(f"experiment_result.model_results[{index}]: {exc}") from exc


def _validate_case_refs(case_refs: Any) -> None:
    """校验案例引用列表中的每个元素是否为非空字符串。

    参数：
        case_refs: 案例引用列表。

    异常：
        TypeError: 不是稳定顺序的列表/元组或元素不是字符串时抛出。
        ValueError: 元素为空白字符串时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"case_refs": case_refs}, "_validate_case_refs 入参", prefix="[配置]")
    if not isinstance(case_refs, (list, tuple)):  # 顶层必须是稳定顺序、可直接 JSON 落盘的序列。
        raise TypeError("summary_result.case_refs must be a list or tuple of strings")
    for index, case_ref in enumerate(case_refs):  # 逐个校验。
        _require_non_empty_string(case_ref, f"summary_result.case_refs[{index}]")  # 每项必须是非空字符串。


def _validate_summary_stats(summary_stats: Any) -> None:
    """校验汇总统计字典的键和值。

    键必须是非空字符串，值递归校验：数值必须有限，字典递归，
    字符串/None允许，列表/元组逐项递归。

    参数：
        summary_stats: 汇总统计字典。

    异常：
        TypeError: 键不是字符串或值类型不正确时抛出。
        ValueError: 键为空白或数值非有限时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"summary_stats": summary_stats}, "_validate_summary_stats 入参", prefix="[配置]")
    if not isinstance(summary_stats, Mapping):  # 必须是映射类型（dict、MappingProxyType 等）。
        raise TypeError(
            f"summary_result.summary_stats must be a mapping, got {type(summary_stats).__name__}"
        )
    for key in summary_stats:  # 逐个校验键。
        if not is_string_like(key):  # 键必须是字符串（含 numpy.str_）。
            raise TypeError(
                f"summary_result.summary_stats[{key!r}] key must be a string, got {type(key).__name__}"
            )
        if not key.strip():  # 键不能为空白。
            raise ValueError(f"summary_result.summary_stats[{key!r}] key must be a non-empty string")
        _validate_summary_stat_value(  # 递归校验值。
            summary_stats[key],
            path=f"summary_result.summary_stats[{key!r}]",
        )


def _validate_summary_stat_value(value: Any, *, path: str, _depth: int = 0, _max_depth: int = 20) -> None:
    """递归校验汇总统计值的合法性。

    支持的值类型：有限数值、布尔、嵌套字典、字符串、None、列表、元组。
    布尔值在统计表中是合法字段（如 ``same_tier`` / ``strict_better``），
    必须被接受。这里故意不接受任意 Iterable，避免 set/generator/bytes 这类
    非稳定或不可直接 JSON 落盘的对象混入 summary_stats。

    参数：
        value: 待校验的值。
        path: 当前值的路径，用于构造错误信息。
        _depth: 内部参数，当前递归深度。
        _max_depth: 内部参数，最大允许递归深度（默认 20）。

    异常：
        TypeError: 值类型不受支持或子键不是字符串时抛出。
        ValueError: 数值非有限或子键为空白时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"value": value, "path": path}, "_validate_summary_stat_value 入参", prefix="[配置]")
    if _depth > _max_depth:
        raise ValueError(f"{path}: nesting depth exceeds {_max_depth}")
    if is_real(value):  # 实数值类型（已排除 bool 和 complex）。
        try:
            fv = float(value)
        except (OverflowError, ValueError, TypeError):
            raise ValueError(f"{path} must be finite") from None
        if not math.isfinite(fv):
            raise ValueError(f"{path} must be finite")
        return
    if isinstance(value, bool):  # 布尔值（统计表 same_tier / strict_better 等合法字段）。
        return
    if isinstance(value, dict):  # 嵌套字典，递归校验。
        for child_key, child_value in value.items():
            if not is_string_like(child_key):  # 子键必须是字符串（含 numpy.str_）。
                raise TypeError(f"{path} child keys must be strings, got {type(child_key).__name__}")
            if not child_key.strip():  # 子键不能为空白。
                raise ValueError(f"{path} child keys must be non-empty strings")
            _validate_summary_stat_value(child_value, path=f"{path}.{child_key}", _depth=_depth + 1, _max_depth=_max_depth)  # 递归校验子值。
        return
    if is_string_like(value) or value is None:  # 字符串（含 numpy.str_）和 None 允许。
        return
    if isinstance(value, (list, tuple)):  # 只允许顺序稳定、可直接落盘的序列类型。
        for item_index, item in enumerate(value):
            _validate_summary_stat_value(item, path=f"{path}[{item_index}]", _depth=_depth + 1, _max_depth=_max_depth)
        return
    raise TypeError(
        f"{path} must be a finite number, bool, string, None, dict, list, or tuple; "
        f"got {type(value).__name__}"
    )


def validate_sequence_result(sequence_result: SequenceResult | dict[str, Any]) -> None:
    """校验单序列结果是否符合冻结协议模式。

    SequenceResult 对象在构造时已通过 ``__post_init__`` 校验，
    且 frozen dataclass 构造后不可变，因此直接返回，避免冗余校验和深拷贝。
    仅对字典格式的输入做完整校验。

    校验内容包括：必需键完整、无额外键、seq_id/scene_code/model_name
    为非空字符串、metric_dict 与冻结指标键集合精确匹配。

    参数：
        sequence_result: SequenceResult 对象或字典格式的结果。

    异常：
        TypeError: 字段类型不正确时抛出。
        ValueError: 字段值不合法时抛出。
        KeyError: 必需键缺失时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"keys": list(sequence_result.keys()) if hasattr(sequence_result, "keys") else type(sequence_result).__name__}, "validate_sequence_result 校验对象 keys", prefix="[配置]")
    if isinstance(sequence_result, SequenceResult):  # frozen dataclass 已在构造时校验，无需重复。
        return
    payload = _as_sequence_result_dict(sequence_result)  # 统一成字典视图。
    require_keys(payload, _SEQUENCE_RESULT_KEYS, name="sequence_result")  # 检查必需键。
    unexpected = sorted(key for key in payload if key not in _SEQUENCE_RESULT_KEYS)  # 找出多余键。
    if unexpected:  # 不允许额外字段。
        raise ValueError(f"sequence_result contains unsupported fields: {unexpected}")
    _require_non_empty_string(payload["seq_id"], "sequence_result.seq_id")  # seq_id 必须非空。
    _require_non_empty_string(payload["scene_code"], "sequence_result.scene_code")  # scene_code 必须非空。
    _require_non_empty_string(payload["model_name"], "sequence_result.model_name")  # model_name 必须非空。
    _validate_metric_dict(payload["metric_dict"], name="sequence_result.metric_dict")  # 校验指标字典。


def validate_experiment_result(experiment_result: ExperimentResult | dict[str, Any]) -> None:
    """校验实验结果是否符合冻结协议模式。

    ExperimentResult 对象在构造时已通过 ``__post_init__`` 校验，
    且 frozen dataclass 构造后不可变，因此直接返回，避免冗余校验和深拷贝。
    仅对字典格式的输入做完整校验。

    .. note::
        当前无生产调用者。保留此函数是因为它作为公共 API 的一部分
        在 ``liquidloc.protocol.__init__`` 中导出，供下游消费者按需调用。

    校验内容包括：必需键完整、无额外键、experiment_id 为非空字符串、
    model_results 中每个元素通过单序列结果校验、aggregate_metrics
    与冻结指标键集合精确匹配。

    参数：
        experiment_result: ExperimentResult 对象或字典格式的结果。

    异常：
        TypeError: 字段类型不正确时抛出。
        ValueError: 字段值不合法时抛出。
        KeyError: 必需键缺失时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"keys": list(experiment_result.keys()) if hasattr(experiment_result, "keys") else type(experiment_result).__name__}, "validate_experiment_result 校验对象 keys", prefix="[配置]")
    if isinstance(experiment_result, ExperimentResult):  # frozen dataclass 已在构造时校验，无需重复。
        return
    payload = _as_experiment_result_dict(experiment_result)  # 统一成字典视图。
    require_keys(payload, _EXPERIMENT_RESULT_KEYS, name="experiment_result")  # 检查必需键。
    unexpected = sorted(key for key in payload if key not in _EXPERIMENT_RESULT_KEYS and key not in _EXPERIMENT_RESULT_OPTIONAL_KEYS)  # 找出多余键（允许可选键）。
    if unexpected:  # 不允许额外字段。
        raise ValueError(f"experiment_result contains unsupported fields: {unexpected}")
    _require_non_empty_string(payload["experiment_id"], "experiment_result.experiment_id")  # experiment_id 必须非空。
    sv = payload.get("snapshot_version")  # 快照版本号校验。
    if sv is not None and (not is_integer(sv) or sv < 1):  # 版本号必须为正整数（如果提供）。
        raise ValueError(f"experiment_result.snapshot_version must be a positive int, got {sv!r}")
    _validate_model_results(payload["model_results"])  # 校验模型结果列表。
    _validate_metric_dict(payload["aggregate_metrics"], name="experiment_result.aggregate_metrics")  # 校验聚合指标。


def validate_summary_result(summary_result: SummaryResult | dict[str, Any]) -> None:
    """校验汇总结果是否符合冻结协议模式。

    SummaryResult 对象在构造时已通过 ``__post_init__`` 校验，
    且 frozen dataclass 构造后不可变，因此直接返回，避免冗余校验和深拷贝。
    仅对字典格式的输入做完整校验。

    校验内容包括：必需键完整、无额外键、case_refs 中每个元素
    为非空字符串、summary_stats 的键和值递归校验。

    参数：
        summary_result: SummaryResult 对象或字典格式的结果。

    异常：
        TypeError: 字段类型不正确时抛出。
        ValueError: 字段值不合法时抛出。
        KeyError: 必需键缺失时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"keys": list(summary_result.keys()) if hasattr(summary_result, "keys") else type(summary_result).__name__}, "validate_summary_result 校验对象 keys", prefix="[配置]")
    if isinstance(summary_result, SummaryResult):  # frozen dataclass 已在构造时校验，无需重复。
        return
    payload = _as_summary_result_dict(summary_result)  # 统一成字典视图。
    require_keys(payload, _SUMMARY_RESULT_KEYS, name="summary_result")  # 检查必需键。
    unexpected = sorted(key for key in payload if key not in _SUMMARY_RESULT_KEYS)  # 找出多余键。
    if unexpected:  # 不允许额外字段。
        raise ValueError(f"summary_result contains unsupported fields: {unexpected}")
    _validate_case_refs(payload["case_refs"])  # 校验案例引用。
    _validate_summary_stats(payload["summary_stats"])  # 校验汇总统计。
