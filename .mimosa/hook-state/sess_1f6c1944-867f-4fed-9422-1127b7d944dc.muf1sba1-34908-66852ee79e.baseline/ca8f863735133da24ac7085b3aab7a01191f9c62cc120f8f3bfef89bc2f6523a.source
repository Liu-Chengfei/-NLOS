"""
文件：`src/liquidloc/plotting/plot_calibration.py`

这个模块负责把校准指标渲染成 SVG 图。它支持两种输入模式：
1. 旧版校准报告（legacy report）：只包含 bias_alignment、risk_error_corr、
   corr_scaling_error 三个标量值，渲染成单条水平条形图。
2. 指标表模式（metric table）：包含多行多列的 mechanism 组指标数据，
   渲染成多面板水平条形图，每个指标一个面板。

它不负责重新计算校准指标，也不负责决定校准指标的含义；它只接受外部
已经准备好的校准数据，然后把行、列、数值整理成统一结构，再拼出 SVG。

上游依赖：
- liquidloc.common.paths.build_output_path：统一输出目录策略。
- liquidloc.protocol.metric_schema.get_metric_meta / get_metric_order：
  从协议层读取指标元信息和顺序。

下游调用者：
- scripts/13_generate_figures.py 等脚本通过 build_calibration_figure_spec /
  render_calibration_figure 渲染校准图。
- 测试文件通过同一入口验证输出。

最容易看错的地方：
1. 旧版报告和指标表两种模式用不同的字段别名做兼容。
2. 指标表支持长表和宽表两种格式，但不能混用。
3. 输出只支持 SVG，其他后缀会直接报错。
4. 这里只做绘图，不做校准指标的计算或重算。
"""

from __future__ import annotations  # 支持前向类型注解，避免循环引用问题。

import math  # 用于数值比较、有限性判断和线性缩放。
from collections.abc import Mapping, Sequence  # 兼容字典式和序列式输入。
from html import escape  # 生成 SVG 时转义文本，防止破坏标签结构。
from pathlib import Path  # 输出路径统一用 Path 处理。
from typing import Any  # 用于 _coerce_numeric 的 value 参数类型注解。

from liquidloc.common.constants import (  # D9 单源：长表字段名、分组名与标签轴优先字段名，禁止本地重复定义。
    FIGURE_PATH_KEY,
    LONG_FORM_GROUP_KEY,
    LONG_FORM_METRIC_KEY,
    LONG_FORM_METRIC_KEYS,
    LONG_FORM_VALUE_KEY,
    METRIC_GROUP_MECHANISM,
    PREFERRED_LABEL_KEYS,
)
from liquidloc.common.validation import coerce_finite_scalar  # 统一数值校验入口，避免本模块重复定义 _coerce_numeric。
from liquidloc.common.paths import build_output_path  # 统一使用项目内的输出目录规则。
from liquidloc.protocol.metric_schema import get_metric_meta, get_metric_order  # 从协议层拿指标元信息和顺序。

# 缓存指标元信息字典，避免每次调用都重新查询。
_METRIC_META = get_metric_meta()
# 只保留 group=mechanism 的指标名，作为校准图的白名单。
_MECHANISM_METRIC_NAMES = tuple(
    metric_name  # 遍历协议顺序中的每个指标名。
    for metric_name in get_metric_order()  # 按协议定义的顺序遍历。
    if _METRIC_META.get(metric_name, {}).get(LONG_FORM_GROUP_KEY) == METRIC_GROUP_MECHANISM  # 只筛 mechanism 组。
)  # mechanism 指标白名单构造完成。

# 旧版校准报告的字段别名映射：每个标准字段名对应一组可能的旧名。
_LEGACY_CALIBRATION_FIELD_ALIASES = {
    "bias_alignment": ("bias_alignment", "corr_bias_error"),  # bias_alignment 的旧名是 corr_bias_error。
    "risk_error_corr": ("risk_error_corr", "corr_risk_error"),  # risk_error_corr 的旧名是 corr_risk_error。
    "corr_scaling_error": ("corr_scaling_error",),  # corr_scaling_error 没有旧名。
}  # 别名映射结束。

# 旧版校准报告的标准字段名列表，顺序决定旧版图里的行顺序。
_LEGACY_CALIBRATION_FIELDS = (
    "bias_alignment",  # 偏置对齐度。
    "risk_error_corr",  # 风险误差相关性。
    "corr_scaling_error",  # 缩放误差相关性。
)  # 旧版字段列表结束。

# 校准图渲染模式常量：作为 figure_spec["mode"] 的唯一合法取值与路由判据，
# 调用处必须引用本常量，禁止散写 "legacy_report" / "metric_table" 字面量（D8/D9）。
# 风险：若任一处漂移，会导致模式路由静默失败，旧版报告被误渲染成指标表或反之。
_MODE_LEGACY_REPORT: str = "legacy_report"  # 旧版校准报告模式。
_MODE_METRIC_TABLE: str = "metric_table"  # 指标表模式。

# 长表模式里会被特殊处理的控制字段集合，这些字段不当作普通指标列。
_LONG_FORM_KEYS = LONG_FORM_METRIC_KEYS  # D9 单源：引用 common/constants.py 冻结集合，禁止本地重复定义。
# 长表折叠时临时塞入的内部标签键，不对外暴露。
_INTERNAL_LABEL_KEY = "_calibration_label"
_PREFERRED_LABEL_KEYS = PREFERRED_LABEL_KEYS  # D9 单源：引用 common/constants.py 标签轴优先字段名，禁止本地重复定义。


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
        requested_path = build_output_path("figures", "calibration.svg")  # 默认落到 figures/calibration.svg。
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
        raise ValueError("render_calibration_figure currently supports SVG output only.")
    figure_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在。
    return figure_path.resolve()  # 返回绝对路径，避免后续歧义。


def _scale_linear(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
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
    if math.isclose(domain_min, domain_max):  # 区间没有跨度时返回中点。
        return (range_min + range_max) / 2.0
    ratio = (value - domain_min) / (domain_max - domain_min)  # 先算归一化比例。
    return range_min + (range_max - range_min) * ratio  # 再映射到目标区间。


def _coerce_numeric(value: Any, *, name: str, minimum: float | None = None, maximum: float | None = None) -> float:
    """把输入值强制转成有限浮点数，并检查范围。

    本函数是 ``common.validation.coerce_finite_scalar`` 的薄包装，保留
    ``minimum`` / ``maximum`` 参数名以维持本模块调用处的兼容性。有限性检查、
    边界值校验以及 bool / bytearray / numpy.str_ 等非数值类型的拒绝全部
    委托给公共工具，避免重复定义。

    仅对纯 Python ``str`` 做一次 ``float()`` 预转换以保持向后兼容（调用方
    常传入字符串形式的数值）；``numpy.str_`` 即便在 NumPy<2.0 继承 ``str``
    也由精确类型检查排除，交由公共工具拒绝。``float()`` 失败时按 value
    contract 抛 ``ValueError``。

    Args:
        value: 待转换的值。
        name: 变量名，用于报错定位。
        minimum: 可选下限，如果提供则值必须 >= minimum。
        maximum: 可选上限，如果提供则值必须 <= maximum。

    Returns:
        转换后的有限浮点数。

    Raises:
        TypeError: 如果值是布尔型或无法转成浮点数。
        ValueError: 如果值不是有限数、边界值非法或超出范围。
    """
    if type(value) is str:  # 纯 Python str 预转换；numpy.str_ 的 type 不是 str，交由公共工具拒绝。
        try:
            value = float(value)  # 守卫覆盖 OverflowError，符合硬约束。
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite numeric value") from exc
    return coerce_finite_scalar(  # 委托有限性、范围、边界值与类型拒绝。
        value,
        name=name,
        min_value=minimum,
        max_value=maximum,
        inclusive=True,
    )


# 校准值的合法区间上下限，旧版报告字段值与指标表 mechanism 指标值都必须落在此区间内。
_CALIBRATION_VALUE_MINIMUM: float = -1.0  # 校准值下限（相关性最小值）。
_CALIBRATION_VALUE_MAXIMUM: float = 1.0  # 校准值上限（相关性最大值）。
# coverage 指标的下限是 0（覆盖率非负），其他校准值下限为 _CALIBRATION_VALUE_MINIMUM。
_COVERAGE_METRIC_NAME: str = "coverage"  # coverage 指标名，来自 protocol/metric_schema.py 的冻结定义。
_COVERAGE_VALUE_MINIMUM: float = 0.0  # coverage 下限（覆盖率非负）。
# 指标表模式画布尺寸默认值与下限，避免魔数散落在 _build_metric_table_spec 里。
_METRIC_TABLE_DEFAULT_WIDTH: int = 1080  # 指标表模式默认画布宽度。
_METRIC_TABLE_MIN_WIDTH: int = 720  # 指标表模式画布宽度下限，避免太窄。
_METRIC_TABLE_DEFAULT_PANEL_HEIGHT: int = 220  # 指标表模式默认单面板高度。
_METRIC_TABLE_MIN_PANEL_HEIGHT: int = 180  # 指标表模式单面板高度下限。
# 旧版报告模式画布尺寸默认值与下限，避免魔数散落在 build_calibration_figure_spec 里。
_LEGACY_DEFAULT_WIDTH: int = 960  # 旧版报告模式默认画布宽度。
_LEGACY_MIN_WIDTH: int = 480  # 旧版报告模式画布宽度下限，避免太窄。
_LEGACY_DEFAULT_HEIGHT: int = 420  # 旧版报告模式默认画布高度。
_LEGACY_MIN_HEIGHT: int = 280  # 旧版报告模式画布高度下限，避免太矮。
# 指标表 SVG 布局常量，避免魔数散落在 _build_metric_table_svg 里。
_METRIC_TABLE_TITLE_HEIGHT: int = 56  # 顶部标题区高度。
_METRIC_TABLE_FOOTER_HEIGHT: int = 24  # 底部留白高度。
_METRIC_TABLE_PANEL_GAP: int = 20  # 面板之间的纵向间距。
_METRIC_TABLE_PANEL_TITLE_HEIGHT: int = 24  # 面板标题所占空间。
_METRIC_TABLE_PLOT_PADDING_TOP: int = 10  # 绘图区上边距。
_METRIC_TABLE_PLOT_PADDING_BOTTOM: int = 28  # 绘图区下边距。
_SVG_MARGIN_LEFT: int = 210  # SVG 左侧给标签留的空间。
_SVG_MARGIN_RIGHT: int = 48  # SVG 右侧留白。
# SVG 颜色常量，避免色值字面量散落在拼字符串代码里导致配置表面漂移。
_SVG_PANEL_COLORS: tuple[str, ...] = ("#2563eb", "#0f766e", "#9333ea")  # 面板循环色。
_SVG_GRIDLINE_COLOR: str = "#e5e7eb"  # 刻度网格线颜色。
_SVG_AXIS_COLOR: str = "#111827"  # 零轴与默认文本颜色。
_SVG_TICK_LABEL_COLOR: str = "#4b5563"  # 刻度文本颜色。
# 指标表规范对象的必需键，缺键时按 value contract raise ValueError 而非 KeyError（D3 数据合同）。
_METRIC_TABLE_REQUIRED_SPEC_KEYS: tuple[str, ...] = (
    "width",  # 画布宽度。
    "panel_height",  # 单个面板高度。
    "panels",  # 面板列表。
    "title",  # 顶部标题。
    "labels",  # 横轴标签。
)
# 指标表面板对象的必需键，缺键时按 value contract raise ValueError 而非 KeyError（D3 数据合同）。
_METRIC_TABLE_REQUIRED_PANEL_KEYS: tuple[str, ...] = (
    "title",  # 面板标题。
    "metric_name",  # 指标名，用于决定值域。
    "values",  # 该指标的数值序列。
)
# 哨兵对象，用于区分"别名未找到"和"字段值为 None"两种情况。
_MISSING_LEGACY_FIELD_SENTINEL: Any = object()

# 旧版 SVG 渲染所需的画布边距与条形几何常量（D5/D9 单源，避免魔数散落在 _build_legacy_svg 里）。
# margin_left / margin_right / 零轴色 / 刻度文本色 与指标表模式共用，直接引用 _SVG_* 单源常量消除重复定义（D9 跨函数根因）。
_LEGACY_SVG_MARGIN_TOP: int = 64  # 顶部标题空间（仅旧版模式需要）。
_LEGACY_SVG_MARGIN_BOTTOM: int = 48  # 底部刻度空间（仅旧版模式需要）。
_LEGACY_SVG_BAR_HEIGHT_MAX: float = 36.0  # 条形高度上限，避免在行数少时条形过高。
# 旧版 SVG 横轴 5 个刻度值，端点复用校准值区间常量，避免魔数漂移（D5 数值安全）。
_LEGACY_SVG_TICK_VALUES: tuple[float, ...] = (
    _CALIBRATION_VALUE_MINIMUM,  # -1.0，校准值下限。
    _CALIBRATION_VALUE_MINIMUM / 2.0,  # -0.5，中点下界。
    0.0,  # 零点。
    _CALIBRATION_VALUE_MAXIMUM / 2.0,  # 0.5，中点上界。
    _CALIBRATION_VALUE_MAXIMUM,  # 1.0，校准值上限。
)
# 旧版 SVG 配色常量（D9 单源，避免颜色字面量散落在 _build_legacy_svg 里）。
# 零轴/标签深色与刻度文本灰色与指标表模式共用，直接引用 _SVG_AXIS_COLOR / _SVG_TICK_LABEL_COLOR 单源常量。
_LEGACY_SVG_COLOR_POSITIVE: str = "#2563eb"  # 正值条形颜色（蓝色）。
_LEGACY_SVG_COLOR_NEGATIVE: str = "#dc2626"  # 负值条形颜色（红色）。
_LEGACY_SVG_COLOR_GRID: str = "#d1d5db"  # 刻度网格线浅灰（旧版专用，与指标表 _SVG_GRIDLINE_COLOR 不同色）。
# 旧版 SVG 规范对象的必需键，缺键时按 value contract raise ValueError 而非 KeyError（D3 数据合同）。
_LEGACY_SVG_REQUIRED_KEYS: tuple[str, ...] = ("width", "height", "series", "title")


def _normalize_legacy_calibration_report(calibration_report: Mapping[str, Any]) -> dict[str, float]:
    """把旧版校准报告归一化成标准字段名到浮点值的映射。

    旧版报告可能使用不同的字段别名，这里按别名映射表逐一查找，
    并把值强制转成 [-1.0, 1.0] 范围内的浮点数。

    Args:
        calibration_report: 旧版校准报告的 mapping 对象。

    Returns:
        以标准字段名为键、浮点值为值的字典。

    Raises:
        TypeError: 如果 calibration_report 不是 mapping。
        ValueError: 如果某个标准字段的所有别名都不存在，或值不是有限数值。
        TypeError/ValueError: 由 _coerce_numeric 传播的数值校验错误。
    """
    if not isinstance(calibration_report, Mapping):  # 报告必须是 mapping。
        raise TypeError("calibration_report must be a mapping.")
    normalized_report: dict[str, float] = {}  # 保存归一化后的字段值。
    for field_name in _LEGACY_CALIBRATION_FIELDS:  # 逐个标准字段处理。
        field_value: Any = _MISSING_LEGACY_FIELD_SENTINEL  # 用哨兵初始化，区分"未找到"与"值为 None"。
        for alias in _LEGACY_CALIBRATION_FIELD_ALIASES[field_name]:  # 按别名优先级查找。
            if alias in calibration_report:  # 如果这个别名存在。
                field_value = calibration_report[alias]  # 取出对应的值。
                break  # 找到第一个就停止。
        if field_value is _MISSING_LEGACY_FIELD_SENTINEL:  # 所有别名都没找到（值为 None 不会命中此分支）。
            alias_text = "', '".join(_LEGACY_CALIBRATION_FIELD_ALIASES[field_name])  # 拼出所有别名。
            raise ValueError(f"calibration_report is missing one of '{alias_text}'.")
        normalized_report[field_name] = _coerce_numeric(  # 强制转成有限浮点数并检查范围。
            field_value,  # 原始值。
            name=f"calibration_report['{field_name}']",  # 报错时标明哪个字段。
            minimum=_CALIBRATION_VALUE_MINIMUM,  # 校准值下限。
            maximum=_CALIBRATION_VALUE_MAXIMUM,  # 校准值上限。
        )
    return normalized_report  # 返回归一化后的报告。


def _coerce_metric_rows(metric_table: Any) -> list[dict]:
    """把指标表输入统一成普通行列表。

    支持三种输入：dataframe-like 对象、单个 mapping、或 mapping 序列。

    Args:
        metric_table: 原始指标表，可以是 dataframe、mapping 或映射序列。

    Returns:
        统一后的字典行列表。

    Raises:
        TypeError: 如果输入格式无法识别或行不是 mapping。
        ValueError: 如果行为空。
    """
    # 显式拒绝 str/bytes/bytearray：它们可迭代但语义上不是"映射序列"，
    # 落到 else 分支后 dict(char) 会抛 ValueError（单字符）或静默生成错误字典
    # （双字符如 dict('hi') -> {'h': 'i'}），均绕过 except TypeError 守卫。
    if isinstance(metric_table, (str, bytes, bytearray)):  # bytearray 同样按硬约束拒绝。
        raise TypeError(
            "metric_table must be a mapping, dataframe-like object, or iterable of mappings."
        )
    if hasattr(metric_table, "to_dict") and hasattr(metric_table, "columns"):  # 兼容 dataframe-like 对象。
        rows = metric_table.to_dict(orient="records")  # 直接转成记录列表。
    elif isinstance(metric_table, Mapping):  # 单个 mapping 当成一行处理。
        rows = [dict(metric_table)]  # 复制成普通 dict 列表。
    else:  # 其他对象按 iterable of mappings 处理。
        try:
            rows = [dict(row) for row in metric_table]  # 逐项转字典。
        except TypeError as exc:  # 不能转成 mapping 序列就直接报错。
            raise TypeError(
                "metric_table must be a mapping, dataframe-like object, or iterable of mappings."
            ) from exc
    if not rows:  # 空表没有可画内容。
        raise ValueError("metric_table must contain at least one row.")
    if not all(isinstance(row, dict) for row in rows):  # 每一行都必须是 dict。
        raise TypeError("metric_table rows must be mappings.")
    return rows  # 返回统一后的行列表。


def _pivot_mechanism_rows(rows: list[dict]) -> list[dict]:
    """把长表行折叠成宽表行，只保留 mechanism 组指标。

    如果输入是宽表（不含 metric/value 列），直接过滤出包含 mechanism 指标的行。
    如果输入是长表（含 metric/value 列），先按 case 维度折叠成宽表再过滤。

    Args:
        rows: 已归一化的字典行列表。

    Returns:
        只包含 mechanism 组指标的宽表行列表。

    Raises:
        ValueError: 如果长表和宽表混用、或没有 mechanism 指标行。
    """
    long_form_rows = [(LONG_FORM_METRIC_KEY in row) and (LONG_FORM_VALUE_KEY in row) for row in rows]  # 判断每行是不是长表格式：必须同时有 metric 和 value。
    if any(long_form_rows) and not all(long_form_rows):  # 长表和宽表不能混着用。
        raise ValueError("metric_table must use either long-form or wide-form rows consistently.")
    if not all(long_form_rows):  # 宽表模式：直接过滤出包含 mechanism 指标的行。
        mechanism_rows = [row for row in rows if any(metric_name in row for metric_name in _MECHANISM_METRIC_NAMES)]  # 只保留有 mechanism 指标的行。
        if not mechanism_rows:  # 没有 mechanism 指标行就无法画图。
            raise ValueError("metric_table must contain mechanism metrics.")
        return mechanism_rows  # 返回过滤后的宽表行。

    # 长表模式：按 case 维度折叠成宽表。
    pivoted_rows = {}  # 使用 case_key 作为折叠后的分组键。
    for row in rows:  # 逐行处理长表记录。
        metric_name = row.get(LONG_FORM_METRIC_KEY)  # 这一行对应的指标名。
        if row.get(LONG_FORM_GROUP_KEY) not in (None, METRIC_GROUP_MECHANISM):  # 只接收 mechanism 组，别的组跳过。
            continue
        if metric_name not in _MECHANISM_METRIC_NAMES:  # 非 mechanism 指标不画。
            continue
        base_row = {key: value for key, value in row.items() if key not in _LONG_FORM_KEYS}  # 剥离长表控制列，只保留 case 维度字段。
        case_key = tuple(sorted((str(key), str(value)) for key, value in base_row.items()))  # 用剩余字段做分组键，保证同一 case 的行能合并。
        if case_key not in pivoted_rows:  # 新 case 就先建一份基础行。
            pivoted_rows[case_key] = dict(base_row)
        pivoted_rows[case_key][metric_name] = row[LONG_FORM_VALUE_KEY]  # 把长表 value 写回宽表列，缺少 value 时直接报错而非静默写入 None。
    mechanism_rows = list(pivoted_rows.values())  # 折叠后的结果重新变成普通行列表。
    if not mechanism_rows:  # 折叠完如果没有任何 mechanism 指标，仍然不能画。
        raise ValueError("metric_table must contain mechanism metrics.")
    return mechanism_rows  # 返回折叠后的宽表行。


def _resolve_label_axis(rows: list[dict[str, Any]], cfg: Mapping[str, Any]) -> str:
    """决定每一行在图上显示哪一列作为标签。

    优先使用配置中显式指定的 label_axis；其次使用长表折叠时生成的内部标签；
    再次按常见标签字段优先级（case_ref > task_id > scene_id > seq_id > method_name）
    自动推断；最后尝试找一个非数值的公共列。

    Args:
        rows: 已归一化的字典行列表。
        cfg: 已归一化的配置字典。

    Returns:
        标签列名。

    Raises:
        ValueError: 如果 rows 为空、显式指定的列不存在，或无法自动推断标签列。
    """
    if not rows:  # 空表无法推断标签列，按值合同用 ValueError 而非 IndexError。
        raise ValueError("metric_table rows must not be empty when resolving label_axis.")
    configured_name = cfg.get("label_axis")  # 先看调用者有没有显式指定。
    common_keys = set(rows[0].keys())  # 从第一行开始找公共列。
    for row in rows[1:]:  # 再与后续每一行求交集，保证这列每行都有。
        common_keys &= set(row.keys())  # 继续求交集，保证候选标签列每行都有。
    if configured_name is not None:  # 如果用户手工指定，就优先尊重。
        if configured_name not in common_keys:  # 但这列必须真的存在。
            raise ValueError(f"label_axis='{configured_name}' is not present in metric_table.")
        return configured_name  # 显式标签列直接返回。

    if _INTERNAL_LABEL_KEY in common_keys:  # 长表折叠时生成的内部标签优先级最高。
        return _INTERNAL_LABEL_KEY  # 长表折叠专用列直接返回。

    for preferred_key in _PREFERRED_LABEL_KEYS:  # 按常见度顺序尝试（D9 单源常量）。
        if preferred_key in common_keys:  # 命中最常见的人类可读列就直接返回。
            return preferred_key

    for name in rows[0]:  # 再从第一行里找一个既公共又非指标的列。
        if name not in common_keys:  # 不是公共列就跳过。
            continue
        if name in _MECHANISM_METRIC_NAMES or name in _LONG_FORM_KEYS:  # 指标列和长表控制列都不能当标签。
            continue
        try:
            for row in rows:  # 逐行尝试数值强转，任意一行失败即视为非数值列。
                _coerce_numeric(row[name], name=name)
        except (TypeError, ValueError):
            return name  # 这列不是纯数值列，适合拿来做标签。
    raise ValueError("Unable to infer label_axis from metric_table. Provide figure_cfg['label_axis'].")  # 所有方法都失败，要求调用者显式指定。


def _resolve_metric_names(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[str]:
    """决定校准图里要画哪些 mechanism 指标面板。

    优先使用配置中显式指定的 metrics 列表；其次从数据中自动检测
    实际出现的 mechanism 指标。

    Args:
        rows: 已归一化的字典行列表。
        cfg: 已归一化的配置字典。

    Returns:
        要绘制的 mechanism 指标名列表。

    Raises:
        TypeError: 如果配置中的 metrics 不是序列类型。
        ValueError: 如果没有可用的 mechanism 指标，或请求了不支持的指标。
    """
    configured_metrics = cfg.get("metrics")  # 如果用户显式指定，就按这个来。
    if configured_metrics is None:  # 没显式指定时，就从数据里实际出现的 mechanism 指标里挑。
        metric_names = [metric_name for metric_name in _MECHANISM_METRIC_NAMES if any(metric_name in row for row in rows)]  # 只保留出现过的 mechanism 指标，避免生成空面板。
    else:  # 显式指定时，就按配置的顺序来。
        if not isinstance(configured_metrics, Sequence) or isinstance(configured_metrics, (str, bytes, bytearray)):  # 配置必须是非字符串/字节序列；bytearray 是 int 序列，必须显式拒绝。
            raise TypeError("figure_cfg['metrics'] must be a sequence when provided.")
        metric_names = list(configured_metrics)  # 转成普通 list，后面方便遍历。
        for metric_name in metric_names:  # 逐个校验元素类型。
            # 用 type() is str 精确匹配，拒绝 numpy.str_ 等子类（与 _resolve_figure_path / _coerce_numeric 一致）。
            if type(metric_name) is not str:
                raise TypeError(
                    f"figure_cfg['metrics'] entries must be str, got {type(metric_name).__name__}: {metric_name!r}"
                )
    if not metric_names:  # 一个都没有就无法生成面板。
        raise ValueError("metric_table must contain at least one mechanism metric.")
    invalid_metric_names = [metric_name for metric_name in metric_names if metric_name not in _MECHANISM_METRIC_NAMES]  # 白名单检查。
    if invalid_metric_names:  # 出现不支持的指标名就报错。
        raise ValueError(f"Unsupported mechanism metrics requested: {invalid_metric_names}")
    return metric_names  # 返回最终决定要画的 mechanism 指标列表。


def _build_metric_table_spec(metric_table: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    """把指标表组装成面板化的绘图规格。

    这个函数处理指标表模式的输入，把每行数据拆成标签序列和指标值序列，
    每个指标对应一个面板。

    Args:
        metric_table: 原始指标表（可以是 dataframe、mapping 或映射序列）。
        cfg: 已归一化的配置字典。

    Returns:
        包含 mode、figure_path、title、label_axis、labels、panels、
        width、panel_height 的绘图规格字典。

    Raises:
        ValueError: 如果行缺少标签列或指标列。
        TypeError/ValueError: 由 _coerce_numeric 传播的数值校验错误。
    """
    rows = _pivot_mechanism_rows(_coerce_metric_rows(metric_table))  # 先统一行格式，再折叠长表并过滤 mechanism 行。
    label_axis = _resolve_label_axis(rows, cfg)  # 决定横轴标签列。
    metric_names = _resolve_metric_names(rows, cfg)  # 决定要画哪些 mechanism 指标。
    labels: list[str] = []  # 横轴标签按行顺序保存。
    panels = []  # 每个指标对应一个面板。
    values_by_metric = {metric_name: [] for metric_name in metric_names}  # 每个指标单独保存一组数值。
    for row in rows:  # 逐行整理，保证标签和每个指标一一对应。
        if label_axis not in row:  # 每行都必须有标签列。
            raise ValueError(f"Each metric_table row must contain '{label_axis}'.")
        labels.append(str(row[label_axis]))  # 标签统一转成字符串，方便 SVG 显示。
        for metric_name in metric_names:  # 每个指标都要逐个检查。
            if metric_name not in row:  # 缺列就不能画。
                raise ValueError(f"Each metric_table row must contain '{metric_name}'.")
            minimum = _COVERAGE_VALUE_MINIMUM if metric_name == _COVERAGE_METRIC_NAME else _CALIBRATION_VALUE_MINIMUM  # coverage 下限是 0，其他是 -1。
            values_by_metric[metric_name].append(  # 追加到对应指标序列。
                _coerce_numeric(  # 强制转成有限浮点数并检查范围。
                    row[metric_name],  # 当前指标值。
                    name=f"metric_table['{metric_name}']",  # 报错时标明哪个指标。
                    minimum=minimum,  # 指标下限。
                    maximum=_CALIBRATION_VALUE_MAXIMUM,  # 指标上限统一为 1.0。
                )
            )
    for metric_name in metric_names:  # 逐个指标生成面板描述。
        metric_meta = _METRIC_META.get(metric_name, {})  # 查这个指标的元数据。
        unit = metric_meta.get("unit")  # 取单位，方便标题显示。
        title = metric_name if not unit else f"{metric_name} ({unit})"  # 有单位就把单位写进标题。
        panels.append({"metric_name": metric_name, "title": title, "values": values_by_metric[metric_name]})  # 面板对象追加完成。
    width = max(  # 画布宽度有下限，避免太窄。
        int(coerce_finite_scalar(cfg.get("width", _METRIC_TABLE_DEFAULT_WIDTH), name="figure_cfg['width']")),
        _METRIC_TABLE_MIN_WIDTH,
    )
    panel_height = max(  # 面板高度也有下限。
        int(coerce_finite_scalar(cfg.get("panel_height", _METRIC_TABLE_DEFAULT_PANEL_HEIGHT), name="figure_cfg['panel_height']")),
        _METRIC_TABLE_MIN_PANEL_HEIGHT,
    )
    return {  # 返回后续 SVG 渲染直接使用的规范对象。
        "mode": _MODE_METRIC_TABLE,  # 标记为指标表模式（引用 L76 定义的常量）。
        FIGURE_PATH_KEY: _resolve_figure_path(cfg),  # 最终输出文件路径（D9 单源引用常量）。
        "title": cfg.get("title") or "Calibration metrics",  # 默认标题。
        "label_axis": None if label_axis == _INTERNAL_LABEL_KEY else label_axis,  # 内部标签不对外暴露。
        "labels": labels,  # 横轴标签。
        "panels": panels,  # 每个 mechanism 指标一个面板。
        "width": width,  # 画布宽度。
        "panel_height": panel_height,  # 面板高度。
    }  # 规范对象收口。


def build_calibration_figure_spec(calibration_report: Any, figure_cfg: Mapping | None = None) -> dict[str, Any]:
    """把旧版校准报告或指标表整理成可渲染的规范对象。

    这个函数会自动检测输入是旧版报告还是指标表：
    - 如果输入包含所有旧版字段别名，就按旧版模式处理。
    - 否则按指标表模式处理。

    Args:
        calibration_report: 旧版校准报告 mapping 或指标表。
        figure_cfg: 可选的配置映射，控制标题、尺寸、输出路径等。

    Returns:
        包含 mode、figure_path、title 等字段的绘图规格字典。
        旧版模式额外包含 series、width、height。
        指标表模式额外包含 label_axis、labels、panels、width、panel_height。

    Raises:
        TypeError: 如果输入类型不正确。
        ValueError: 如果缺少必需字段（按 value contract 抛 ValueError 而非 KeyError）或数值校验失败。
    """
    cfg = _coerce_figure_cfg(figure_cfg)  # 先统一配置格式。
    if isinstance(calibration_report, Mapping) and all(  # 检查是否包含所有旧版字段。
        any(alias in calibration_report for alias in _LEGACY_CALIBRATION_FIELD_ALIASES[field_name])  # 每个标准字段至少有一个别名存在。
        for field_name in _LEGACY_CALIBRATION_FIELDS  # 遍历所有旧版标准字段。
    ):
        normalized_report = _normalize_legacy_calibration_report(calibration_report)  # 归一化旧版报告。
        series = [  # 把每个字段转成序列条目。
            {"metric_name": field_name, "value": normalized_report[field_name]}  # 字段名和归一化后的值。
            for field_name in _LEGACY_CALIBRATION_FIELDS  # 按旧版字段顺序。
        ]
        return {  # 返回旧版模式的规范对象。
            "mode": _MODE_LEGACY_REPORT,  # 标记为旧版模式。
            FIGURE_PATH_KEY: _resolve_figure_path(cfg),  # 最终输出文件路径（D9 单源引用常量）。
            "title": cfg.get("title") or "Calibration correlations",  # 默认标题。
            "series": series,  # 旧版模式的序列数据。
            "width": max(  # 画布宽度有下限，避免太窄。
                int(coerce_finite_scalar(cfg.get("width", _LEGACY_DEFAULT_WIDTH), name="figure_cfg['width']")),
                _LEGACY_MIN_WIDTH,
            ),
            "height": max(  # 画布高度有下限，避免太矮。
                int(coerce_finite_scalar(cfg.get("height", _LEGACY_DEFAULT_HEIGHT), name="figure_cfg['height']")),
                _LEGACY_MIN_HEIGHT,
            ),
        }
    return _build_metric_table_spec(calibration_report, cfg)  # 不是旧版就按指标表模式处理。


def _build_legacy_svg(spec: dict[str, Any]) -> str:
    """根据旧版模式规范对象拼出 SVG 字符串。

    旧版模式渲染成单张水平条形图，每行一个校准指标，
    正值用蓝色，负值用红色，零轴为黑色竖线。

    Args:
        spec: 旧版模式的绘图规格字典，包含 width、height、title、series。

    Returns:
        完整的 SVG 字符串。

    Raises:
        ValueError: 如果 spec 缺少 _LEGACY_SVG_REQUIRED_KEYS 中的任一必需键。
    """
    # D3 数据合同：required keys 用 if not in: raise ValueError 而非 KeyError。
    missing_keys = [key for key in _LEGACY_SVG_REQUIRED_KEYS if key not in spec]
    if missing_keys:  # 缺键时按 value contract 报 ValueError，避免 KeyError 泄漏到调用方。
        raise ValueError(f"legacy spec is missing required keys: {missing_keys}")
    width = spec["width"]  # 画布宽度。
    height = spec["height"]  # 画布高度。
    margin_left = _SVG_MARGIN_LEFT  # 左侧给标签留空间（D5/D9 常量化，与指标表模式共用）。
    margin_right = _SVG_MARGIN_RIGHT  # 右侧留白（D5/D9 常量化，与指标表模式共用）。
    margin_top = _LEGACY_SVG_MARGIN_TOP  # 顶部标题空间（D5/D9 常量化）。
    margin_bottom = _LEGACY_SVG_MARGIN_BOTTOM  # 底部刻度空间（D5/D9 常量化）。
    plot_width = width - margin_left - margin_right  # 真正绘图区宽度。
    plot_height = height - margin_top - margin_bottom  # 真正绘图区高度。
    # D5 数值安全：domain 端点复用 _CALIBRATION_VALUE_MINIMUM/_MAXIMUM，避免魔数漂移。
    baseline_x = _scale_linear(
        0.0, _CALIBRATION_VALUE_MINIMUM, _CALIBRATION_VALUE_MAXIMUM, margin_left, margin_left + plot_width
    )  # 零轴的 x 像素位置。
    row_step = plot_height / max(len(spec["series"]), 1)  # 每行占的纵向步长。
    bar_height = min(_LEGACY_SVG_BAR_HEIGHT_MAX, row_step * 0.6)  # 条形高度上限（D5/D9 常量化）。
    svg_parts = [  # SVG 片段列表，最后统一拼接。
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',  # SVG 根标签。
        '<rect width="100%" height="100%" fill="white"/>',  # 白色背景。
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="22" font-family="Arial">{escape(spec["title"])}</text>',  # 顶部标题。
    ]
    for tick_value in _LEGACY_SVG_TICK_VALUES:  # 5 个刻度值（D5/D9 常量化）。
        # D5 数值安全：domain 端点复用校准值区间常量。
        tick_x = _scale_linear(
            tick_value, _CALIBRATION_VALUE_MINIMUM, _CALIBRATION_VALUE_MAXIMUM, margin_left, margin_left + plot_width
        )  # 刻度映射到像素。
        svg_parts.append(  # 追加刻度网格线。
            f'<line x1="{tick_x:.2f}" y1="{margin_top:.2f}" x2="{tick_x:.2f}" y2="{margin_top + plot_height:.2f}" stroke="{_LEGACY_SVG_COLOR_GRID}" stroke-width="1"/>'
        )
        svg_parts.append(  # 追加刻度文本。
            f'<text x="{tick_x:.2f}" y="{margin_top + plot_height + 22:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="{_SVG_TICK_LABEL_COLOR}">{tick_value:.1f}</text>'
        )
    svg_parts.append(  # 追加零轴竖线。
        f'<line x1="{baseline_x:.2f}" y1="{margin_top:.2f}" x2="{baseline_x:.2f}" y2="{margin_top + plot_height:.2f}" stroke="{_SVG_AXIS_COLOR}" stroke-width="1.5"/>'
    )
    for row_index, item in enumerate(spec["series"]):  # 逐行画条形。
        center_y = margin_top + row_step * (row_index + 0.5)  # 当前行中心 y。
        # D5 数值安全：domain 端点复用校准值区间常量。
        value_x = _scale_linear(
            item["value"], _CALIBRATION_VALUE_MINIMUM, _CALIBRATION_VALUE_MAXIMUM, margin_left, margin_left + plot_width
        )  # 数值映射到 x。
        bar_x = min(baseline_x, value_x)  # 条形左边界取零轴和数值中较小的。
        bar_width = max(abs(value_x - baseline_x), 1.5)  # 条形宽度至少留一点可见性。
        bar_y = center_y - bar_height / 2  # 条形垂直居中。
        fill = _LEGACY_SVG_COLOR_POSITIVE if item["value"] >= 0.0 else _LEGACY_SVG_COLOR_NEGATIVE  # 正值蓝色，负值红色（D9 常量化）。
        text_anchor = "start" if item["value"] >= 0.0 else "end"  # 数值标签朝外放置。
        value_label_x = value_x + 8 if item["value"] >= 0.0 else value_x - 8  # 标签与条形保持一点距离。
        svg_parts.append(  # 追加标签文本。
            f'<text x="{margin_left - 14:.2f}" y="{center_y + 4:.2f}" text-anchor="end" font-size="13" font-family="Arial" fill="{_SVG_AXIS_COLOR}">{escape(item["metric_name"])}</text>'
        )
        svg_parts.append(  # 追加条形矩形。
            f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{fill}" rx="4" ry="4"/>'
        )
        svg_parts.append(  # 追加数值文本。
            f'<text x="{value_label_x:.2f}" y="{center_y + 4:.2f}" text-anchor="{text_anchor}" font-size="12" font-family="Arial" fill="{_SVG_AXIS_COLOR}">{item["value"]:.3f}</text>'
        )
    svg_parts.append("</svg>")  # SVG 结束标签。
    return "".join(svg_parts)  # 拼成完整字符串并返回。


def _build_metric_table_svg(spec: dict[str, Any]) -> str:
    """根据指标表模式规范对象拼出 SVG 字符串。

    指标表模式渲染成多面板水平条形图，每个指标一个面板，
    面板内每行一个标签对应的条形。

    Args:
        spec: 指标表模式的绘图规格字典，包含 width、panel_height、
              title、panels、labels 等。

    Returns:
        完整的 SVG 字符串。

    Raises:
        ValueError: 如果 spec 缺少必需键，或面板缺少必需键，
            或面板 values 与 labels 长度不一致（value contract 缺键/不一致）。
    """
    # value contract 缺键必须抛 ValueError 而非 KeyError（D3 数据合同）。
    for required_key in _METRIC_TABLE_REQUIRED_SPEC_KEYS:
        if required_key not in spec:
            raise ValueError(f"metric table spec is missing required key '{required_key}'.")
    width = spec["width"]  # 画布宽度。
    panel_height = spec["panel_height"]  # 单个面板高度。
    title_height = _METRIC_TABLE_TITLE_HEIGHT  # 标题区高度。
    footer_height = _METRIC_TABLE_FOOTER_HEIGHT  # 底部留白。
    panel_gap = _METRIC_TABLE_PANEL_GAP  # 面板之间的间距。
    panel_count = len(spec["panels"])  # 面板数量。
    height = title_height + footer_height + panel_count * panel_height + max(panel_count - 1, 0) * panel_gap  # 总高度计算。
    margin_left = _SVG_MARGIN_LEFT  # 左边给标签留的空间。
    margin_right = _SVG_MARGIN_RIGHT  # 右边留白。
    panel_title_height = _METRIC_TABLE_PANEL_TITLE_HEIGHT  # 面板标题所占空间。
    plot_padding_top = _METRIC_TABLE_PLOT_PADDING_TOP  # 绘图区上边距。
    plot_padding_bottom = _METRIC_TABLE_PLOT_PADDING_BOTTOM  # 绘图区下边距。
    plot_width = width - margin_left - margin_right  # 真正用于画条形的宽度。
    svg_parts = [  # SVG 片段列表，最后统一拼接。
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',  # SVG 根标签。
        '<rect width="100%" height="100%" fill="white"/>',  # 白色背景。
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" font-size="22" font-family="Arial">{escape(spec["title"])}</text>',  # 顶部标题。
    ]
    colors = _SVG_PANEL_COLORS  # 每个面板循环使用不同颜色。
    labels = spec["labels"]  # 横轴标签。
    for panel_index, panel in enumerate(spec["panels"]):  # 逐个面板绘制。
        # 面板对象缺键必须抛 ValueError 而非 KeyError（D3 数据合同）。
        for required_key in _METRIC_TABLE_REQUIRED_PANEL_KEYS:
            if required_key not in panel:
                raise ValueError(f"metric table panel is missing required key '{required_key}'.")
        panel_top = title_height + panel_index * (panel_height + panel_gap)  # 这个面板的顶部坐标。
        plot_top = panel_top + panel_title_height + plot_padding_top  # 实际绘图区域顶部。
        plot_height = panel_height - panel_title_height - plot_padding_top - plot_padding_bottom  # 绘图区高度。
        svg_parts.append(  # 追加面板标题文本。
            f'<text x="{margin_left}" y="{panel_top + 18:.1f}" text-anchor="start" font-size="16" font-family="Arial">{escape(panel["title"])}</text>'
        )
        # 值域复用协议冻结常量，避免魔数漂移（D5 数值安全 / D6 指标口径冻结）。
        domain_min = _COVERAGE_VALUE_MINIMUM if panel["metric_name"] == _COVERAGE_METRIC_NAME else _CALIBRATION_VALUE_MINIMUM  # coverage 下限是 0，其他是 -1。
        domain_max = _CALIBRATION_VALUE_MAXIMUM  # 上限统一为 1.0。
        baseline_x = _scale_linear(0.0, domain_min, domain_max, margin_left, margin_left + plot_width)  # 零轴的 x 像素位置。
        for tick_value in (domain_min, (domain_min + domain_max) / 2.0, domain_max):  # 3 个刻度值。
            tick_x = _scale_linear(tick_value, domain_min, domain_max, margin_left, margin_left + plot_width)  # 刻度映射到像素。
            svg_parts.append(  # 追加刻度网格线。
                f'<line x1="{tick_x:.2f}" y1="{plot_top:.2f}" x2="{tick_x:.2f}" y2="{plot_top + plot_height:.2f}" stroke="{_SVG_GRIDLINE_COLOR}" stroke-width="1"/>'
            )
            svg_parts.append(  # 追加刻度文本。
                f'<text x="{tick_x:.2f}" y="{plot_top + plot_height + 18:.2f}" text-anchor="middle" font-size="11" font-family="Arial" fill="{_SVG_TICK_LABEL_COLOR}">{tick_value:.3g}</text>'
            )
        svg_parts.append(  # 追加零轴竖线。
            f'<line x1="{baseline_x:.2f}" y1="{plot_top:.2f}" x2="{baseline_x:.2f}" y2="{plot_top + plot_height:.2f}" stroke="{_SVG_AXIS_COLOR}" stroke-width="1.5"/>'
        )
        row_step = plot_height / max(len(labels), 1)  # 每个条目占的纵向步长。
        bar_height = min(28.0, row_step * 0.62)  # 条形高度上限。
        fill = colors[panel_index % len(colors)]  # 循环选色，避免所有面板都一样。
        panel_values = panel["values"]  # 该指标的数值序列。
        # 显式长度守卫，防止 labels 与 values 长度不一致时 zip 静默截断（D10 动态链路安全）。
        if len(panel_values) != len(labels):
            raise ValueError(
                f"metric table panel '{panel['metric_name']}' has {len(panel_values)} values "
                f"but {len(labels)} labels; lengths must match to avoid silent truncation."
            )
        for row_index, (label, value) in enumerate(zip(labels, panel_values)):  # 逐个条目画条形。
            center_y = plot_top + row_step * (row_index + 0.5)  # 当前条目的中心 y。
            value_x = _scale_linear(value, domain_min, domain_max, margin_left, margin_left + plot_width)  # 数值映射到 x。
            bar_x = min(baseline_x, value_x)  # 条形左边界取零轴和数值中较小的。
            bar_width = max(abs(value_x - baseline_x), 1.5)  # 条形宽度至少留一点可见性。
            bar_y = center_y - bar_height / 2  # 条形垂直居中。
            text_anchor = "start" if value_x >= baseline_x else "end"  # 数值标签朝外放置。
            text_x = value_x + 6 if value_x >= baseline_x else value_x - 6  # 标签与条形保持一点距离。
            svg_parts.append(  # 追加标签文本。
                f'<text x="{margin_left - 12:.2f}" y="{center_y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="{_SVG_AXIS_COLOR}">{escape(label)}</text>'
            )
            svg_parts.append(  # 追加条形矩形。
                f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{fill}" rx="4" ry="4"/>'
            )
            svg_parts.append(  # 追加数值文本。
                f'<text x="{text_x:.2f}" y="{center_y + 4:.2f}" text-anchor="{text_anchor}" font-size="12" font-family="Arial" fill="{_SVG_AXIS_COLOR}">{value:.3g}</text>'
            )
    svg_parts.append("</svg>")  # SVG 结束标签。
    return "".join(svg_parts)  # 拼成完整字符串并返回。


def _build_svg(spec: dict[str, Any]) -> str:
    """根据规范对象的 mode 字段选择对应的 SVG 构建器。

    Args:
        spec: 绘图规格字典，必须包含 mode 字段。

    Returns:
        完整的 SVG 字符串。

    Raises:
        ValueError: 如果 spec 缺少 mode 字段（value contract 缺键）。
    """
    if "mode" not in spec:  # value contract 缺键必须抛 ValueError 而非 KeyError。
        raise ValueError("spec is missing required key 'mode'")  # 明确指出缺哪个键。
    if spec["mode"] == _MODE_LEGACY_REPORT:  # 旧版模式走旧版 SVG 构建器。
        return _build_legacy_svg(spec)
    if spec["mode"] == _MODE_METRIC_TABLE:  # 指标表模式走指标表 SVG 构建器。
        return _build_metric_table_svg(spec)
    raise ValueError(f"Unsupported figure spec mode: {spec['mode']!r}")  # 未知模式显式拒绝，避免静默走错分支。


def render_calibration_figure(calibration_report: Any, figure_cfg: Mapping | None) -> str | dict[str, Any]:
    """把校准数据渲染成 SVG 文件，并返回路径或清单。

    这个函数是模块对外的主要渲染入口：它先把输入规范化，
    再生成规范对象，然后拼出 SVG 文本，最后写入磁盘。

    Args:
        calibration_report: 旧版校准报告 mapping 或指标表。
        figure_cfg: 配置映射，控制标题、尺寸、输出路径和清单输出行为。

    Returns:
        默认返回输出路径字符串。如果 figure_cfg 中 return_manifest 为真，
        则返回包含 figure_path 和 metrics 的字典。

    Raises:
        TypeError: 如果输入类型不正确。
        ValueError: 如果数值校验失败、输出格式不支持或必需字段缺失。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "figure_cfg_keys": list(figure_cfg.keys()) if isinstance(figure_cfg, Mapping) else None,
            "calibration_report_type": type(calibration_report).__name__,
        },
        "render_calibration_figure 入口参数",
        prefix="[plotting]",
    )
    cfg = _coerce_figure_cfg(figure_cfg)  # 先统一配置格式。
    figure_spec = build_calibration_figure_spec(calibration_report, cfg)  # 再生成规范对象。
    figure_path = figure_spec[FIGURE_PATH_KEY]  # 输出路径（D9 单源引用常量）。
    figure_path.write_text(_build_svg(figure_spec), encoding="utf-8")  # 写出 SVG 内容。
    if cfg.get("return_manifest"):  # 如果调用者要求清单，就返回描述对象。
        if figure_spec["mode"] == _MODE_LEGACY_REPORT:  # 旧版模式从 series 取指标名。
            metrics = [item["metric_name"] for item in figure_spec["series"]]  # 提取旧版指标名列表。
        else:  # 指标表模式从 panels 取指标名。
            metrics = [panel["metric_name"] for panel in figure_spec["panels"]]  # 提取面板指标名列表。
        return {FIGURE_PATH_KEY: str(figure_path), "metrics": metrics}  # 返回清单对象（D9 单源引用常量）。
    return str(figure_path)  # 默认返回路径字符串，方便外部直接使用。
