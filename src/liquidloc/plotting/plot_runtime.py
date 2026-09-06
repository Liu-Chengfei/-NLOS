"""
文件：`src/liquidloc/plotting/plot_runtime.py`

这个模块负责把已经整理好的 runtime 表渲染成 SVG 图。它不计算 runtime
指标本身，只负责把外部传进来的表格数据规范化，分解成 label、metric、
panel 和输出路径，然后写出 SVG 文件或者返回一个清单对象。

上游通常是 metrics 或 statistics 阶段输出的 runtime_table。
下游通常是 `scripts/13_generate_figures.py`、测试，以及需要把运行时指标
放进报告里的调用者。

最容易误解的地方：
1. 这里支持宽表和长表两种输入，但不能混用。
2. 如果输入是长表，这里会先按 case 维度折叠成宽表再画图。
3. 输出默认是 SVG，别的后缀会直接报错。
"""

from __future__ import annotations  # 这里保持前向类型注解支持，避免循环引用时出问题。

import math  # 用于数值比较、有限性判断和线性缩放。
from collections.abc import Mapping, Sequence  # 统一兼容 dict、list、pandas 风格对象。
from html import escape  # 生成 SVG 时需要转义文本，避免破坏标签结构。
from pathlib import Path  # 输出路径统一处理成 Path。
from typing import Any  # 用于 _resolve_metric_names 的 rows/cfg 参数类型注解。

from liquidloc.common.constants import FIGURE_PATH_KEY, LONG_FORM_GROUP_KEY, LONG_FORM_METRIC_KEY, LONG_FORM_METRIC_KEYS, LONG_FORM_VALUE_KEY, METRIC_GROUP_RUNTIME, PREFERRED_LABEL_KEYS  # D9 单源：长表字段名集合与标签轴优先字段名，禁止本地重复定义。
from liquidloc.common.validation import coerce_finite_scalar  # 统一数值校验入口，避免本模块重复定义数值守卫。
from liquidloc.common.paths import build_output_path  # 统一使用项目内的输出目录规则。
from liquidloc.protocol.metric_schema import get_metric_meta, get_metric_order  # 从协议层拿 runtime 指标顺序和元信息。


class _EmptyRuntimeTable(ValueError):
    """空 runtime 表信号，表示没有可画数据，应优雅跳过而非报错。"""
    pass

_METRIC_META = get_metric_meta()  # 缓存指标元信息，避免重复查询。
_RUNTIME_METRIC_NAMES = tuple(  # 只保留 group=runtime 的指标名，作为这个模块的白名单。
    metric_name for metric_name in get_metric_order() if _METRIC_META.get(metric_name, {}).get(LONG_FORM_GROUP_KEY) == METRIC_GROUP_RUNTIME  # 只筛 runtime 组。
)  # runtime 指标白名单构造完成。
_INTERNAL_LABEL_KEY = "_runtime_label"  # 长表折叠时临时塞入的内部标签键，不对外暴露。
_LONG_FORM_KEYS = LONG_FORM_METRIC_KEYS  # D9 单源：引用 common/constants.py 冻结集合，禁止本地重复定义。长表模式里会被特殊处理的字段。
_PREFERRED_LABEL_KEYS = PREFERRED_LABEL_KEYS  # D9 单源：引用 common/constants.py 标签轴优先字段名，禁止本地重复定义。
# runtime 图画布尺寸默认值与下限，避免魔数散落在 build_runtime_figure_spec 里
# （与 plot_calibration._METRIC_TABLE_* 当前同值但语义独立，runtime 模式专属；
# 不跨文件复用以避免两个绘图模块互相耦合，各自可独立演进——遵循 plot_calibration 已建立的模块顶层常量模式）。
_RUNTIME_DEFAULT_WIDTH: int = 1080  # runtime 图默认画布宽度。
_RUNTIME_MIN_WIDTH: int = 720  # runtime 图画布宽度下限，避免太窄。
_RUNTIME_DEFAULT_PANEL_HEIGHT: int = 220  # runtime 图默认单面板高度。
_RUNTIME_MIN_PANEL_HEIGHT: int = 180  # runtime 图单面板高度下限。


# 输入归一：把 dataframe、mapping 或 mapping 序列列统一成普通行列表。
def _coerce_runtime_rows(runtime_table: Any) -> list[dict]:  # 这里把各种输入统一成“行列表”。
    # runtime_table 可能来自 dataframe、单行 mapping 或多行 mapping 序列。
    # rows 最终一定是普通 dict 列表，后续所有逻辑都只认这个形状。
    # 显式拒绝 str/bytes/bytearray：它们可迭代但语义上不是“映射序列”，
    # 落到 else 分支后 dict(char) 会抛 ValueError（单字符）或静默生成错误字典
    # （双字符如 dict('hi') -> {'h': 'i'}），均绕过 except TypeError 守卫。
    if isinstance(runtime_table, (str, bytes, bytearray)):  # bytearray 同样按硬约束拒绝。
        raise TypeError(
            "runtime_table must be a mapping, dataframe-like object, or iterable of mappings."
        )
    if hasattr(runtime_table, "to_dict") and hasattr(runtime_table, "columns"):  # 兼容 dataframe-like 对象。
        rows = runtime_table.to_dict(orient="records")  # 直接转成记录列表。
    elif isinstance(runtime_table, Mapping):  # mapping 输入可能是单行，也可能是按标签组织的多行。
        if runtime_table and all(isinstance(value, Mapping) for value in runtime_table.values()):  # 多行嵌套 mapping。
            rows = []  # 逐个标签展开成普通行。
            for label, metric_dict in runtime_table.items():  # 这里把外层 key 当作标签。
                row = dict(metric_dict)  # 先复制原始行，避免改动输入对象。
                row[_INTERNAL_LABEL_KEY] = str(label)  # 把外层标签塞进内部列，供后面识别。
                rows.append(row)  # 收集展开后的行。
        else:  # 既不是嵌套 mapping，就按单个 mapping 当一行处理。
            rows = [dict(runtime_table)]  # 单个 mapping 直接当作一行。
    else:  # 其他对象按 iterable of mappings 处理。
        try:  # 逐项转字典，失败就说明输入形状不对。
            rows = [dict(row) for row in runtime_table]  # 其余情况按 iterable of mappings 处理。
        except TypeError as exc:  # 不能转成 mapping 序列就直接报错。
            raise TypeError(
                "runtime_table must be a mapping, dataframe-like object, or iterable of mappings."
            ) from exc

    if not rows:  # 空表没有可画内容；用 _EmptyRuntimeTable 信号让 render_runtime_figure 优雅跳过。
        raise _EmptyRuntimeTable("runtime_table is empty; no runtime data to plot.")
    if not all(isinstance(row, dict) for row in rows):  # 每一行都必须是 dict。
        raise TypeError("runtime_table rows must be mappings.")

    long_form_rows = [(LONG_FORM_METRIC_KEY in row) and (LONG_FORM_VALUE_KEY in row) for row in rows]  # 判断每行是不是长表：必须同时有 metric 和 value。
    if any(long_form_rows) and not all(long_form_rows):  # 长表和宽表不能混着用。
        raise ValueError("runtime_table must use either long-form or wide-form rows consistently.")
    if all(long_form_rows):  # 如果全部都是长表，就先折叠成宽表。
        pivoted_rows = {}  # 使用 case_key 作为折叠后的分组键。
        for row_index, row in enumerate(rows):  # 逐行处理长表记录。
            metric_name = row.get(LONG_FORM_METRIC_KEY)  # 这一行对应的指标名。
            if row.get(LONG_FORM_GROUP_KEY) not in (None, METRIC_GROUP_RUNTIME):  # 只接收 runtime 组，别的组跳过。
                continue
            if metric_name not in _RUNTIME_METRIC_NAMES:  # 非 runtime 指标不画。
                continue
            base_row = {key: value for key, value in row.items() if key not in _LONG_FORM_KEYS}  # 剥离长表控制列。
            case_key = tuple(sorted((str(key), str(value)) for key, value in base_row.items()))  # 用剩余字段做分组键。
            if case_key not in pivoted_rows:  # 新 case 就先建一份基础行。
                pivoted_rows[case_key] = dict(base_row)
            if metric_name in pivoted_rows[case_key]:  # 同一 case 下同一指标只能出现一次。
                raise ValueError(
                    f"runtime_table contains duplicate runtime metric '{metric_name}' for row {row_index}."
                )
            pivoted_rows[case_key][metric_name] = row[LONG_FORM_VALUE_KEY]  # 把长表 value 写回宽表列，缺少 value 时直接报错而非静默写入 None。
        rows = list(pivoted_rows.values())  # 折叠后的结果重新变成普通行列表。
        if not rows:  # 折叠完如果没有任何 runtime 指标，仍然不能画。
            raise ValueError("runtime_table must contain at least one runtime metric row.")
    return rows  # 返回统一后的行列表。


# 配置归一：把可选配置复制成普通字典，避免后面修改外部对象。
def _coerce_figure_cfg(figure_cfg: Mapping | None) -> dict[str, Any]:
    """把可选配置复制成普通字典，避免后续修改外部对象。

    Args:
        figure_cfg: 可选的配置映射，可以为 None。

    Returns:
        复制后的普通字典；如果输入为 None 则返回空字典。

    Raises:
        TypeError: 如果 figure_cfg 不是 mapping 类型。
    """
    if figure_cfg is None:  # 没传配置就用空配置。
        return {}
    if not isinstance(figure_cfg, Mapping):  # 配置必须可按键访问。
        raise TypeError("figure_cfg must be a mapping when provided.")
    return dict(figure_cfg)  # 复制一份，避免修改外部对象。


# 标签轴选择器：决定每一行在图上显示哪一列作为标签。
def _resolve_label_axis(rows: list[dict[str, Any]], cfg: Mapping[str, Any]) -> str:  # 决定横向标签列用哪一列。
    # 这一步是为了找到一个每行都有、而且更适合展示的公共列。
    # cfg 里如果显式给了 label_axis，就直接优先用它。
    if not rows:
        raise ValueError("runtime_table rows must not be empty when resolving label_axis.")
    configured_name = cfg.get("label_axis")  # 先看调用者有没有显式指定。
    common_keys = set(rows[0].keys())  # 从第一行开始找公共列。
    for row in rows[1:]:  # 再与后续每一行求交集，保证这列每行都有。
        common_keys &= set(row.keys())  # 继续求交集，保证候选标签列每行都有。
    if configured_name is not None:  # 如果用户手工指定，就优先尊重。
        if configured_name not in common_keys:  # 但这列必须真的存在。
            raise ValueError(f"label_axis='{configured_name}' is not present in runtime_table.")
        return configured_name  # 显式标签列直接返回。

    if _INTERNAL_LABEL_KEY in common_keys:  # 长表折叠时生成的内部标签优先级最高。
        return _INTERNAL_LABEL_KEY  # 长表折叠专用列直接返回。

    for preferred_key in _PREFERRED_LABEL_KEYS:  # 按常见度顺序尝试（D9 单源常量）。
        if preferred_key in common_keys:
            return preferred_key  # 命中最常见的人类可读列就直接返回。

    for name in rows[0]:  # 再从第一行里找一个既公共又非指标的列。
        if name not in common_keys:
            continue
        if name in _RUNTIME_METRIC_NAMES or name in _LONG_FORM_KEYS:  # 指标列和长表控制列都不能当标签。
            continue
        try:
            for row in rows:
                value = float(row[name])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite value in column '{name}'")
        except (TypeError, ValueError, OverflowError):
            return name  # 这列不是纯数值列，适合拿来做标签。
    raise ValueError("Unable to infer label_axis from runtime_table. Provide figure_cfg['label_axis'].")


# 指标白名单选择器：决定 runtime 图里要画哪些指标面板。
def _resolve_metric_names(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[str]:
    """决定 runtime 图里要画哪些 runtime 指标面板。

    优先使用配置中显式指定的 metrics 列表；其次从数据中自动检测
    实际出现的 runtime 指标。

    Args:
        rows: 已归一化的字典行列表。
        cfg: 已归一化的配置字典。

    Returns:
        要绘制的 runtime 指标名列表。

    Raises:
        TypeError: 如果配置中的 metrics 不是序列类型，或元素不是纯 str
            （拒绝 numpy.str_ 子类与 bytearray 等 int 序列）。
        ValueError: 如果没有可用的 runtime 指标，或请求了不支持的指标。
    """
    # 这里会优先使用配置，其次用协议白名单里在表中实际出现的指标。
    # metric_names 最终决定 panels 里有多少个面板。
    configured_metrics = cfg.get("metrics")  # 如果用户显式指定，就按这个来。
    if configured_metrics is None:  # 没显式指定时，就从数据里实际出现的 runtime 指标里挑。
        metric_names = [metric_name for metric_name in _RUNTIME_METRIC_NAMES if any(metric_name in row for row in rows)]  # 只保留出现过的 runtime 指标，避免生成空面板。
    else:  # 显式指定时，就按配置的顺序来。
        if not isinstance(configured_metrics, Sequence) or isinstance(configured_metrics, (str, bytes, bytearray)):  # 配置必须是非字符串/字节序列；bytearray 是 int 序列，必须显式拒绝。
            raise TypeError("figure_cfg['metrics'] must be a sequence when provided.")
        metric_names = list(configured_metrics)  # 转成普通 list，后面方便遍历。
        for metric_name in metric_names:  # 逐个校验元素类型。
            # 用 type() is str 精确匹配，拒绝 numpy.str_ 等子类（与 plot_calibration._resolve_metric_names 一致）。
            if type(metric_name) is not str:
                raise TypeError(
                    f"figure_cfg['metrics'] entries must be str, got {type(metric_name).__name__}: {metric_name!r}"
                )

    if not metric_names:  # 一个都没有就无法生成面板。
        raise ValueError("runtime_table must contain at least one runtime metric.")

    invalid_metric_names = [metric_name for metric_name in metric_names if metric_name not in _RUNTIME_METRIC_NAMES]  # 白名单检查。
    if invalid_metric_names:  # 出现不支持的指标名就报错。
        raise ValueError(f"Unsupported runtime metrics requested: {invalid_metric_names}")
    return metric_names  # 返回最终决定要画的 runtime 指标列表。


# 数值强转工具：runtime 指标值专用的有限浮点数守卫（无范围约束）。
def _coerce_numeric(value: Any, *, name: str) -> float:
    """把输入值强制转成有限浮点数（runtime 指标值专用，无范围约束）。

    本函数是 ``common.validation.coerce_finite_scalar`` 的薄包装。有限性检查、
    bool / bytearray / numpy.str_ 等非数值类型的拒绝全部委托给公共工具，
    避免本模块重复定义数值守卫（D5 数值安全 / D9 单源）。

    仅对纯 Python ``str`` 做一次 ``float()`` 预转换以保持向后兼容（调用方
    常传入字符串形式的数值）；``numpy.str_`` 即便在 NumPy<2.0 继承 ``str``
    也由 ``type() is str`` 精确匹配排除，交由公共工具拒绝。``float()`` 失败
    时按 value contract 抛 ``ValueError``，守卫覆盖 ``OverflowError``。

    Args:
        value: 待转换的值。
        name: 变量名，用于报错定位。

    Returns:
        转换后的有限浮点数。

    Raises:
        TypeError: 如果值是布尔型或无法转成浮点数。
        ValueError: 如果值不是有限数。
    """
    if type(value) is str:  # 纯 Python str 预转换；numpy.str_ 的 type 不是 str，交由公共工具拒绝。
        try:
            value = float(value)  # 守卫覆盖 OverflowError，符合硬约束。
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite numeric value") from exc
    return coerce_finite_scalar(value, name=name)  # 委托有限性、bool 与类型拒绝。


# 序列整理器：把每行 runtime 数据拆成标签序列和指标值序列。
def _normalize_runtime_series(rows: list[dict[str, Any]], label_axis: str, metric_names: Sequence[str]) -> tuple[list[str], dict[str, list[float]]]:  # 这里把每行整理成绘图数组。
    # labels 存横轴标签，values_by_metric 存每个指标的一列数值。
    # label_axis 是展示列名，metric_names 是需要画图的 runtime 指标列表。
    labels: list[str] = []  # 横轴标签按行顺序保存。
    values_by_metric = {metric_name: [] for metric_name in metric_names}  # 每个指标单独保存一组数值。

    for row in rows:  # 逐行整理，保证标签和每个指标一一对应。
        if label_axis not in row:  # 每行都必须有标签列。
            raise ValueError(f"Each runtime_table row must contain '{label_axis}'.")
        label_value = row[label_axis]  # 取出当前行标签。
        labels.append(str(label_value))  # 标签统一转成字符串，方便 SVG 显示。

        for metric_name in metric_names:  # 每个指标都要逐个检查。
            if metric_name not in row:  # 缺列就不能画。
                raise ValueError(f"Each runtime_table row must contain '{metric_name}'.")
            metric_value = row[metric_name]  # 当前指标值。
            # D5 数值安全：委托 _coerce_numeric 统一处理 bool 拒绝、OverflowError 守卫、
            # numpy.str_ 拒绝与 NaN/Inf 检查，避免裸 float() 漏掉 OverflowError（D5 根因）。
            metric_number = _coerce_numeric(metric_value, name=f"runtime_table['{metric_name}']")
            values_by_metric[metric_name].append(metric_number)  # 追加到对应指标序列。

    return labels, values_by_metric  # 返回标签列表和每个指标的值序列。


# 输出路径解析器：统一决定 runtime SVG 写到哪里。
def _resolve_figure_path(cfg: Mapping[str, object]) -> Path:
    """从配置中解析输出文件路径，确保是 SVG 格式且目录存在。

    Args:
        cfg: 已归一化的配置字典，可能包含 figure_path。

    Returns:
        解析后的绝对 Path 对象。

    Raises:
        TypeError: 如果 figure_path 不是 str 或 pathlib.Path 类型
            （numpy.str_、bytearray、bool、int、bytes 等一律拒绝，
            避免 Path() 静默强转导致路径语义漂移）。
        ValueError: 如果路径为空白字符串或后缀不是 .svg。
    """
    requested_path = cfg.get(FIGURE_PATH_KEY)  # 读取 figure_path 键（可选，缺省走默认路径，D9 单源引用常量）。
    if requested_path is None:  # 没指定就用默认输出路径。
        requested_path = build_output_path("figures", "runtime.svg")  # 默认落到 figures/runtime.svg。
    # 类型守卫：只接受原生 str 和 pathlib.Path。这里用 type() is str 而非 isinstance(str)，
    # 是为了排除 numpy.str_——NumPy 1.x 中它是 str 的子类，isinstance 会漏过，而 Path() 会
    # 把它静默强转成路径，掩盖调用方传错类型的事实。bytearray/bool/int/bytes 同理落入 else。
    if isinstance(requested_path, Path):  # Path 直接接受，无需再转。
        figure_path = requested_path
    elif type(requested_path) is str:  # 纯 str 接受；numpy.str_ 等子类落到 else 分支拒绝。
        if not requested_path.strip():  # 空白字符串没有输出意义（YAML 里 figure_path: "" 会被解析成空串）。
            raise ValueError("figure_cfg['figure_path'] must not be blank.")
        figure_path = Path(requested_path)  # 统一转成 Path。
    else:  # 其他类型一律拒绝，避免 Path() 把 bool/int 静默强转成路径。
        raise TypeError(
            f"figure_cfg['figure_path'] must be str or Path, got {type(requested_path).__name__}: {requested_path!r}"
        )
    if figure_path.suffix.lower() != ".svg":  # 这里只支持 SVG。
        raise ValueError("render_runtime_figure currently supports SVG output only.")
    figure_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在。
    return figure_path.resolve()  # 返回绝对路径，避免后续歧义。


# 线性映射工具：把数据域数值换算成图上的像素位置。
def _scale_linear(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:  # 线性映射工具。
    """把数值从数据域线性映射到像素区间。

    当 domain_min 和 domain_max 相等时（区间没有跨度），返回目标区间中点。
    所有输入先经 coerce_finite_scalar 校验为有限浮点数，避免 NaN/Inf 污染 SVG 坐标，
    同时统一拒绝 bool 与非数值类型。

    Args:
        value: 要映射的数值。
        domain_min: 数据域最小值。
        domain_max: 数据域最大值。
        range_min: 目标区间最小值（像素）。
        range_max: 目标区间最大值（像素）。

    Returns:
        映射后的像素坐标值。

    Raises:
        TypeError: 任一输入为非数值类型（含 bool）。
        ValueError: 任一输入为 NaN 或 inf。
    """
    value = coerce_finite_scalar(value, name="value")  # 校验为有限浮点数，守卫 float() 转换异常。
    domain_min = coerce_finite_scalar(domain_min, name="domain_min")  # 数据域下限校验。
    domain_max = coerce_finite_scalar(domain_max, name="domain_max")  # 数据域上限校验。
    range_min = coerce_finite_scalar(range_min, name="range_min")  # 目标区间下限校验。
    range_max = coerce_finite_scalar(range_max, name="range_max")  # 目标区间上限校验。
    if math.isclose(domain_min, domain_max):  # 如果区间没有跨度，就返回中点。
        return (range_min + range_max) / 2.0
    ratio = (value - domain_min) / (domain_max - domain_min)  # 先算归一化比例。
    return range_min + (range_max - range_min) * ratio  # 再映射到目标区间。


# 规范构建函数：把 runtime 表组装成面板化的绘图规格。
def build_runtime_figure_spec(runtime_table: Any, figure_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:  # 组装 runtime 图的完整规范对象。
    """把 runtime 表整理成可直接渲染的规范对象。

    作用：先统一输入格式和配置，再决定横轴标签列和要绘制的 runtime 指标，
    整理成绘图序列后组装面板列表，返回供 SVG 渲染直接消费的规范对象。

    参数:
        runtime_table: 运行时指标表，支持 DataFrame、单个映射或映射序列，
                       也支持长表格式（含 metric/value 列）。
        figure_cfg: 可选的配置映射，控制标题、尺寸、输出路径和指标选择。

    返回值:
        dict: 包含 figure_path、title、label_axis、labels、panels、width、panel_height 的规范对象。

    异常:
        TypeError: 输入类型不正确。
        ValueError: 输入为空、长表宽表混用、或缺少 runtime 指标。
    """

    # 这里返回的 dict 会被 _build_svg 直接消费，所以键名必须稳定。
    # runtime_table 是原始输入，figure_cfg 主要控制标题、尺寸和清单输出行为。
    rows = _coerce_runtime_rows(runtime_table)  # 先统一行格式；空表会抛 ValueError。
    cfg = _coerce_figure_cfg(figure_cfg)  # 再统一配置格式。
    label_axis = _resolve_label_axis(rows, cfg)  # 决定横轴标签列。
    metric_names = _resolve_metric_names(rows, cfg)  # 决定要画哪些 runtime 指标。
    labels, values_by_metric = _normalize_runtime_series(rows, label_axis=label_axis, metric_names=metric_names)  # 整理成绘图序列。

    panels = []  # 每个指标对应一个面板。
    for metric_name in metric_names:  # 逐个指标生成面板描述。
        metric_meta = _METRIC_META.get(metric_name, {})  # 查这个指标的元数据。
        unit = metric_meta.get("unit")  # 取单位，方便标题显示。
        panel_title = metric_name if not unit else f"{metric_name} ({unit})"  # 有单位就把单位写进标题。
        panels.append(  # 这里每个面板都只描述一个指标。
            {
                "metric_name": metric_name,  # 面板对应的指标名。
                "title": panel_title,  # 面板标题。
                "values": values_by_metric[metric_name],  # 这个指标在所有标签上的值序列。
            }
        )  # 单个面板对象追加完成。

    # 画布宽度/面板高度：默认值与下限常量化（D9），并用 coerce_finite_scalar 守卫
    # 拒绝 bool/NaN/Inf/非数值输入（D5），与 plot_calibration._build_metric_table_spec 对齐。
    width = max(
        int(coerce_finite_scalar(cfg.get("width", _RUNTIME_DEFAULT_WIDTH), name="figure_cfg['width']")),
        _RUNTIME_MIN_WIDTH,
    )  # 画布宽度有下限，避免太窄。
    panel_height = max(
        int(coerce_finite_scalar(cfg.get("panel_height", _RUNTIME_DEFAULT_PANEL_HEIGHT), name="figure_cfg['panel_height']")),
        _RUNTIME_MIN_PANEL_HEIGHT,
    )  # 面板高度也有下限。

    return {  # 返回后续 SVG 渲染直接使用的规范对象。
        FIGURE_PATH_KEY: _resolve_figure_path(cfg),  # 最终输出文件路径（D9 单源引用常量）。
        "title": cfg.get("title") or "Runtime metrics",  # 默认标题。
        "label_axis": None if label_axis == _INTERNAL_LABEL_KEY else label_axis,  # 内部标签不对外暴露。
        "labels": labels,  # 横轴标签。
        "panels": panels,  # 每个 runtime 指标一个面板。
        "width": width,  # 画布宽度。
        "panel_height": panel_height,  # 面板高度。
    }  # 规范对象收口。


# runtime SVG 布局常量，避免魔数散落在 _build_svg 里（D5 数值安全 / D9 单源）。
_RUNTIME_SVG_TITLE_HEIGHT: int = 56  # 顶部标题区高度。
_RUNTIME_SVG_FOOTER_HEIGHT: int = 24  # 底部留白高度。
_RUNTIME_SVG_PANEL_GAP: int = 20  # 面板之间的纵向间距。
_RUNTIME_SVG_PANEL_TITLE_HEIGHT: int = 24  # 面板标题所占空间。
_RUNTIME_SVG_PLOT_PADDING_TOP: int = 10  # 绘图区上边距。
_RUNTIME_SVG_PLOT_PADDING_BOTTOM: int = 28  # 绘图区下边距。
_RUNTIME_SVG_MARGIN_LEFT: int = 190  # 左侧给标签留的空间。
_RUNTIME_SVG_MARGIN_RIGHT: int = 48  # 右侧留白。
_RUNTIME_SVG_TICK_COUNT: int = 4  # 每个面板的刻度分段数（实际刻度数 = 分段数 + 1）。
_RUNTIME_SVG_TITLE_Y: int = 32  # 顶部标题的 y 坐标。
_RUNTIME_SVG_TITLE_FONT_SIZE: int = 22  # 顶部标题字号。
_RUNTIME_SVG_PANEL_TITLE_Y_OFFSET: int = 18  # 面板标题相对面板顶部的 y 偏移。
_RUNTIME_SVG_PANEL_TITLE_FONT_SIZE: int = 16  # 面板标题字号。
_RUNTIME_SVG_GRIDLINE_WIDTH: int = 1  # 刻度网格线宽度。
_RUNTIME_SVG_AXIS_WIDTH: float = 1.5  # 零轴线宽。
_RUNTIME_SVG_TICK_LABEL_Y_OFFSET: int = 18  # 刻度文本相对绘图区底部的 y 偏移。
_RUNTIME_SVG_TICK_LABEL_FONT_SIZE: int = 11  # 刻度文本字号。
_RUNTIME_SVG_BAR_HEIGHT_MAX: float = 28.0  # 条形高度上限。
_RUNTIME_SVG_BAR_HEIGHT_RATIO: float = 0.62  # 条形高度占纵向步长的比例。
_RUNTIME_SVG_BAR_MIN_WIDTH: float = 1.5  # 条形最小宽度，保证可见性。
_RUNTIME_SVG_BAR_CORNER_RADIUS: int = 4  # 条形圆角半径。
_RUNTIME_SVG_TEXT_X_OFFSET: int = 6  # 数值文本相对条形端点的 x 偏移。
_RUNTIME_SVG_LABEL_X_OFFSET: int = 12  # 标签文本相对左边距的 x 间隔。
_RUNTIME_SVG_TEXT_Y_OFFSET: int = 4  # 文本相对条形中心 y 的偏移。
_RUNTIME_SVG_LABEL_FONT_SIZE: int = 12  # 标签与数值文本字号。
_RUNTIME_SVG_DOMAIN_FALLBACK_SPAN: float = 1.0  # 值域退化时强行拉开的最小跨度。
_RUNTIME_SVG_BACKGROUND_COLOR: str = "white"  # 画布背景色。
# runtime SVG 配色常量（D9 单源，避免颜色字面量散落在 _build_svg 里）。
_RUNTIME_SVG_PANEL_COLORS: tuple[str, ...] = ("#2563eb", "#0f766e", "#9333ea", "#ea580c", "#dc2626")  # 面板循环色。
_RUNTIME_SVG_GRIDLINE_COLOR: str = "#e5e7eb"  # 刻度网格线颜色。
_RUNTIME_SVG_AXIS_COLOR: str = "#111827"  # 零轴与默认文本颜色。
_RUNTIME_SVG_TICK_LABEL_COLOR: str = "#4b5563"  # 刻度文本颜色。
# runtime SVG 规范对象的必需键，缺键时按 value contract raise ValueError 而非 KeyError（D3 数据合同）。
_RUNTIME_SVG_REQUIRED_SPEC_KEYS: tuple[str, ...] = (
    "width",  # 画布宽度。
    "panel_height",  # 单个面板高度。
    "panels",  # 面板列表。
    "title",  # 顶部标题。
    "labels",  # 横轴标签。
)
# runtime SVG 面板对象的必需键，缺键时按 value contract raise ValueError 而非 KeyError（D3 数据合同）。
_RUNTIME_SVG_REQUIRED_PANEL_KEYS: tuple[str, ...] = (
    "title",  # 面板标题。
    "values",  # 该指标的数值序列。
)


# SVG 组装器：把 runtime 图规范对象拼成完整 SVG 文本。
def _build_svg(spec: dict[str, Any]) -> str:
    """把 runtime 图规范对象拼成完整 SVG 字符串。

    spec 里已经包含标题、标签、面板和尺寸，这里只负责拼接 SVG 片段。
    spec["panels"] 里的每个 panel 都会被画成一个独立子图。

    Args:
        spec: runtime 图的绘图规格字典，包含 width、panel_height、
              title、panels、labels 等。

    Returns:
        完整的 SVG 字符串。

    Raises:
        ValueError: 如果 spec 缺少必需键，或面板缺少必需键，
            或面板 values 与 labels 长度不一致（value contract 缺键/不一致）。
    """
    # value contract 缺键必须抛 ValueError 而非 KeyError（D3 数据合同）。
    for required_key in _RUNTIME_SVG_REQUIRED_SPEC_KEYS:
        if required_key not in spec:
            raise ValueError(f"runtime spec is missing required key '{required_key}'.")
    width = spec["width"]  # 画布宽度。
    panel_height = spec["panel_height"]  # 单个面板高度。
    title_height = _RUNTIME_SVG_TITLE_HEIGHT  # 标题区高度。
    footer_height = _RUNTIME_SVG_FOOTER_HEIGHT  # 底部留白。
    panel_gap = _RUNTIME_SVG_PANEL_GAP  # 面板之间的间距。
    panel_count = len(spec["panels"])  # 面板数量决定总高度。
    height = title_height + footer_height + panel_count * panel_height + max(panel_count - 1, 0) * panel_gap  # 总高度计算。

    margin_left = _RUNTIME_SVG_MARGIN_LEFT  # 左边给标签留的空间。
    margin_right = _RUNTIME_SVG_MARGIN_RIGHT  # 右边留白。
    panel_title_height = _RUNTIME_SVG_PANEL_TITLE_HEIGHT  # 面板标题所占空间。
    plot_padding_top = _RUNTIME_SVG_PLOT_PADDING_TOP  # 绘图区上边距。
    plot_padding_bottom = _RUNTIME_SVG_PLOT_PADDING_BOTTOM  # 绘图区下边距。
    plot_width = width - margin_left - margin_right  # 真正用于画条形的宽度。

    svg_parts = [  # SVG 片段列表，最后统一拼接。
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{_RUNTIME_SVG_BACKGROUND_COLOR}"/>',
        f'<text x="{width / 2:.1f}" y="{_RUNTIME_SVG_TITLE_Y}" text-anchor="middle" font-size="{_RUNTIME_SVG_TITLE_FONT_SIZE}" font-family="Arial">{escape(spec["title"])}</text>',
    ]

    colors = _RUNTIME_SVG_PANEL_COLORS  # 每个面板循环使用不同颜色。
    labels = spec["labels"]  # 横轴标签。

    for panel_index, panel in enumerate(spec["panels"]):  # 逐个面板绘制。
        # 面板对象缺键必须抛 ValueError 而非 KeyError（D3 数据合同）。
        for required_key in _RUNTIME_SVG_REQUIRED_PANEL_KEYS:
            if required_key not in panel:
                raise ValueError(f"runtime panel is missing required key '{required_key}'.")
        panel_top = title_height + panel_index * (panel_height + panel_gap)  # 这个面板的顶部坐标。
        plot_top = panel_top + panel_title_height + plot_padding_top  # 实际绘图区域顶部。
        plot_height = panel_height - panel_title_height - plot_padding_top - plot_padding_bottom  # 绘图区高度。
        # 这一段先画面板标题和背景网格，再画零轴和每一条指标条形。
        svg_parts.append(  # 追加面板标题文本。
            f'<text x="{margin_left}" y="{panel_top + _RUNTIME_SVG_PANEL_TITLE_Y_OFFSET:.1f}" text-anchor="start" font-size="{_RUNTIME_SVG_PANEL_TITLE_FONT_SIZE}" font-family="Arial">{escape(panel["title"])}</text>'
        )  # 面板标题文本结束。

        values = panel["values"]  # 当前面板对应的数值序列。
        if not values:  # 空数值序列跳过该面板，避免 min/max 崩溃。
            continue
        # 显式长度守卫，防止 labels 与 values 长度不一致时 zip 静默截断（D10 动态链路安全）。
        if len(values) != len(labels):
            raise ValueError(
                f"runtime panel '{panel['title']}' has {len(values)} values "
                f"but {len(labels)} labels; lengths must match to avoid silent truncation."
            )
        domain_min = min(0.0, min(values))  # 让零点进入坐标域，方便看正负变化。
        domain_max = max(0.0, max(values))  # 同时把最大值纳入坐标域。
        if math.isclose(domain_min, domain_max):  # 如果所有值一样，就强行拉开一点。
            domain_max = domain_min + _RUNTIME_SVG_DOMAIN_FALLBACK_SPAN
        zero_x = _scale_linear(0.0, domain_min, domain_max, margin_left, margin_left + plot_width)  # 计算零轴位置。

        tick_count = _RUNTIME_SVG_TICK_COUNT  # 每个面板放 5 个刻度。
        for tick_index in range(tick_count + 1):  # 逐个刻度生成网格线和文字。
            tick_value = domain_min + (domain_max - domain_min) * (tick_index / tick_count)  # 刻度值按线性插值算。
            tick_x = _scale_linear(tick_value, domain_min, domain_max, margin_left, margin_left + plot_width)  # 映射到像素坐标。
            svg_parts.append(  # 追加网格线。
                f'<line x1="{tick_x:.2f}" y1="{plot_top:.2f}" x2="{tick_x:.2f}" y2="{plot_top + plot_height:.2f}" stroke="{_RUNTIME_SVG_GRIDLINE_COLOR}" stroke-width="{_RUNTIME_SVG_GRIDLINE_WIDTH}"/>'
            )  # 网格线结束。
            svg_parts.append(  # 追加刻度文本。
                f'<text x="{tick_x:.2f}" y="{plot_top + plot_height + _RUNTIME_SVG_TICK_LABEL_Y_OFFSET:.2f}" text-anchor="middle" font-size="{_RUNTIME_SVG_TICK_LABEL_FONT_SIZE}" font-family="Arial" fill="{_RUNTIME_SVG_TICK_LABEL_COLOR}">{tick_value:.3g}</text>'
            )  # 刻度文本结束。

        svg_parts.append(  # 追加零轴。
            f'<line x1="{zero_x:.2f}" y1="{plot_top:.2f}" x2="{zero_x:.2f}" y2="{plot_top + plot_height:.2f}" stroke="{_RUNTIME_SVG_AXIS_COLOR}" stroke-width="{_RUNTIME_SVG_AXIS_WIDTH}"/>'
        )  # 零轴结束。

        row_count = len(labels)  # 当前面板的条目数。
        row_step = plot_height / max(row_count, 1)  # 每个条目占的纵向步长。
        bar_height = min(_RUNTIME_SVG_BAR_HEIGHT_MAX, row_step * _RUNTIME_SVG_BAR_HEIGHT_RATIO)  # 条形高度上限。
        fill = colors[panel_index % len(colors)]  # 循环选色，避免所有面板都一样。

        for row_index, (label, value) in enumerate(zip(labels, values)):  # 逐个条目画条形。
            # 每一条先写标签，再画条形，最后把数值贴在条形旁边。
            center_y = plot_top + row_step * (row_index + 0.5)  # 当前条目的中心 y。
            value_x = _scale_linear(value, domain_min, domain_max, margin_left, margin_left + plot_width)  # 数值映射到 x。
            bar_x = min(zero_x, value_x)  # 条形左边界。
            bar_width = max(abs(value_x - zero_x), _RUNTIME_SVG_BAR_MIN_WIDTH)  # 条形宽度至少留一点可见性。
            bar_y = center_y - bar_height / 2  # 条形垂直居中。
            text_anchor = "start" if value_x >= zero_x else "end"  # 数值标签朝外放置。
            text_x = value_x + _RUNTIME_SVG_TEXT_X_OFFSET if value_x >= zero_x else value_x - _RUNTIME_SVG_TEXT_X_OFFSET  # 标签与条形保持一点距离。

            svg_parts.append(  # 追加标签文本。
                f'<text x="{margin_left - _RUNTIME_SVG_LABEL_X_OFFSET:.2f}" y="{center_y + _RUNTIME_SVG_TEXT_Y_OFFSET:.2f}" text-anchor="end" font-size="{_RUNTIME_SVG_LABEL_FONT_SIZE}" font-family="Arial" fill="{_RUNTIME_SVG_AXIS_COLOR}">{escape(label)}</text>'
            )  # 标签文本结束。
            svg_parts.append(  # 追加条形矩形。
                f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{fill}" rx="{_RUNTIME_SVG_BAR_CORNER_RADIUS}" ry="{_RUNTIME_SVG_BAR_CORNER_RADIUS}"/>'
            )  # 条形矩形结束。
            svg_parts.append(  # 追加数值文本。
                f'<text x="{text_x:.2f}" y="{center_y + _RUNTIME_SVG_TEXT_Y_OFFSET:.2f}" text-anchor="{text_anchor}" font-size="{_RUNTIME_SVG_LABEL_FONT_SIZE}" font-family="Arial" fill="{_RUNTIME_SVG_AXIS_COLOR}">{value:.3g}</text>'
            )  # 数值文本结束。

    svg_parts.append("</svg>")  # SVG 结束标签。
    return "".join(svg_parts)  # 拼成完整字符串并返回。


# 真实渲染函数：写出 runtime SVG，并按配置选择返回路径或清单。
def render_runtime_figure(runtime_table: Any, figure_cfg: Mapping[str, Any]) -> str | dict[str, Any]:  # 这个函数把 runtime 图真正写到磁盘。
    """把 runtime 表渲染成 SVG 文件，并返回路径或清单。

    作用：模块对外的主要渲染入口。先统一配置，再生成规范对象，
    然后拼出 SVG 文本写入磁盘，最后根据配置决定返回路径字符串或清单对象。

    Args:
        runtime_table: 运行时指标表，支持 DataFrame、单个映射或映射序列。
        figure_cfg: 配置映射，控制标题、尺寸、输出路径和清单输出行为。

    Returns:
        str | dict: 默认返回输出路径字符串。如果 figure_cfg 中 return_manifest 为真，
                    则返回包含 figure_path、label_axis 和 metrics 的字典；若
                    runtime 表为空则返回 ``{"skipped": True, "reason": "no_runtime_data"}``。

    Raises:
        TypeError: 输入类型不正确。
        ValueError: 输入为空或缺少 runtime 指标，或 figure_spec 缺少必需键。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "figure_cfg_keys": list(figure_cfg.keys()) if isinstance(figure_cfg, Mapping) else None,
            "runtime_table_type": type(runtime_table).__name__,
        },
        "render_runtime_figure 入口参数",
        prefix="[plotting]",
    )

    # return_manifest 决定是返回字符串路径，还是返回更完整的描述对象。
    # figure_table 经过规范化后会进入 build_runtime_figure_spec，再进入 _build_svg。
    cfg = _coerce_figure_cfg(figure_cfg)  # 先统一配置。
    try:
        figure_spec = build_runtime_figure_spec(runtime_table, cfg)  # 再生成规范对象。
    except _EmptyRuntimeTable:  # 空表时优雅跳过，返回结构化 skipped 结果。
        return {"skipped": True, "reason": "no_runtime_data"}
    if FIGURE_PATH_KEY not in figure_spec:  # 规范对象必须包含输出路径（D3 value contract，D9 单源引用常量）。
        raise ValueError(f"figure_spec is missing '{FIGURE_PATH_KEY}'.")  # 缺键属于值合同违规，按项目硬约束用 ValueError 而非 KeyError（与 plot_cases.render_case_figures L371-372 口径一致）。
    figure_path = figure_spec[FIGURE_PATH_KEY]  # 输出路径。
    figure_path.write_text(_build_svg(figure_spec), encoding="utf-8")  # 写出 SVG 内容。

    if cfg.get("return_manifest"):  # 如果调用者要求清单，就返回描述对象。可选键用 .get() 合理。
        if "label_axis" not in figure_spec:  # 规范对象必须包含标签列（D3 value contract）。
            raise ValueError("figure_spec is missing 'label_axis'.")  # 缺键属于值合同违规，按项目硬约束用 ValueError 而非 KeyError。
        if "panels" not in figure_spec:  # 规范对象必须包含面板列表（D3 value contract）。
            raise ValueError("figure_spec is missing 'panels'.")  # 缺键属于值合同违规，按项目硬约束用 ValueError 而非 KeyError。
        return {  # 返回更完整的清单对象。
            FIGURE_PATH_KEY: str(figure_path),  # 输出路径字符串（D9 单源引用常量）。
            "label_axis": figure_spec["label_axis"],  # 使用的标签列。
            "metrics": [panel["metric_name"] for panel in figure_spec["panels"]],  # 本次绘制的指标列表。
        }  # 清单对象结束。
    return str(figure_path)  # 默认返回路径字符串，方便外部直接使用。
