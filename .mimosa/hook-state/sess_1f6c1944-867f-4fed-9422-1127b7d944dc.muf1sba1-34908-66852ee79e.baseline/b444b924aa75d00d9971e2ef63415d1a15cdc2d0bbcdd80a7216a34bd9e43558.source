"""
文件：`src/liquidloc/plotting/plot_sweeps.py`

这个模块负责把已经准备好的 sweep 表渲染成 SVG 图。它不负责重新跑 sweep，
也不负责生成实验结果，只负责把已有表格转成规范对象，再把规范对象写成图。

上游一般是 sweep 汇总表、统计表或者实验脚本。下游一般是 figures 生成脚本、
测试，或者直接消费输出路径的报告流程。

最容易看错的地方：
1. x_axis 和 y_axis 都可能自动推断，也可以手工指定。
2. 输入可以是单行 mapping、DataFrame 风格对象，或者映射序列。
3. 如果 x 轴是纯数值，就画线图；否则默认画柱图。
"""

from __future__ import annotations  # 支持前向注解。

import math  # 用于数值判断和线性缩放。
from collections.abc import Mapping  # 兼容字典式输入。
from html import escape  # SVG 文本转义。
from pathlib import Path  # 统一处理输出路径。
from typing import Any  # 用于 _coerce_sweep_rows 的 sweep_table 参数类型注解。
from liquidloc.common.constants import FIGURE_PATH_KEY  # D9 单源：绘图层 figure_path 键名常量。
from liquidloc.common.paths import build_output_path  # 统一输出目录策略。
from liquidloc.common.validation import coerce_finite_scalar  # 统一数值校验入口，避免本模块重复定义数值守卫。
from liquidloc.common.validation import is_real  # 统一判断标量数值类型。
from liquidloc.protocol.scene_axis_protocol import SCENE_AXES as _SCENE_AXES  # 从协议层导入冻结轴名，避免硬编码漂移。
from liquidloc.protocol.metric_schema import get_metric_meta  # 读取指标元信息。
from liquidloc.protocol.metric_schema import get_metric_order  # 读取指标顺序。


_SWEEP_AXIS_PRIORITY = _SCENE_AXES  # 常见 sweep 维度，优先当 x 轴候选。
# sweep 图画布尺寸默认值与下限，避免魔数散落在 build_sweep_figure_spec 里
# （与 plot_runtime._RUNTIME_*、plot_calibration._LEGACY_* 当前同模式但语义独立，
# sweep 模式专属；不跨文件复用以避免两个绘图模块互相耦合，各自可独立演进
# ——遵循 plot_calibration/plot_runtime 已建立的模块顶层常量模式）。
_SWEEP_DEFAULT_WIDTH: int = 960  # sweep 图默认画布宽度。
_SWEEP_MIN_WIDTH: int = 320  # sweep 图画布宽度下限，避免太窄（与 _build_svg 的下限对齐）。
_SWEEP_DEFAULT_HEIGHT: int = 540  # sweep 图默认画布高度。
_SWEEP_MIN_HEIGHT: int = 240  # sweep 图画布高度下限，避免太矮（与 _build_svg 的下限对齐）。


def _is_real_number(value: Any) -> bool:
    """判断一个值是不是可用于坐标缩放的普通数值。

    字符串形式的数字（如 "4"、"3.14"）也会被尝试转换为数值判断，
    避免纯数字字符串被误判为分类变量。
    """
    if is_real(value):  # bool 虽然是 Number，但这里不能当数值。
        return True
    if type(value) is str:  # 仅接受纯 Python str，拒绝 numpy.str_（项目硬约束）。
        try:
            float(value)
            return True
        except (TypeError, ValueError, OverflowError):
            pass
    return False


def _ordered_common_keys(rows: list[dict[str, Any]]) -> list[str]:
    """按首次出现顺序收集所有行共同拥有的键。"""
    if not rows:  # 空表无法取首行键集合，按价值合同抛 ValueError 而非 IndexError。
        raise ValueError("sweep_table rows must not be empty when collecting common keys.")
    common_keys = set(rows[0].keys())  # 先拿第一行键集合。
    for row in rows[1:]:  # 再逐行求交集。
        current_keys = set(row.keys())  # 当前行的键集合。
        common_keys &= current_keys  # 只保留所有行都拥有的键。

    ordered_keys: list[str] = []  # 按出现顺序输出的公共键。
    seen: set[str] = set()  # 去重集合。
    for row in rows:  # 按原始行序扫描。
        for key in row.keys():  # 按原始列序扫描。
            if key in common_keys and key not in seen:  # 只保留公共键的第一次出现。
                ordered_keys.append(key)  # 记录这个键。
                seen.add(key)  # 标记为已见。
    return ordered_keys  # 返回有序公共键列表。


def _coerce_sweep_rows(sweep_table: Any) -> list[dict]:
    """把 sweep 输入统一成普通行列表。"""
    # 显式拒绝 str/bytes/bytearray：它们可迭代但语义上不是“映射序列”，
    # 落到 else 分支后 dict(char) 会抛 ValueError（单字符）或静默生成错误字典
    # （双字符如 dict('hi') -> {'h': 'i'}），均绕过 except TypeError 守卫。
    if isinstance(sweep_table, (str, bytes, bytearray)):  # bytearray 同样按硬约束拒绝。
        raise TypeError(
            "sweep_table must be a mapping, dataframe-like object, or iterable of mappings."
        )
    if hasattr(sweep_table, "to_dict") and hasattr(sweep_table, "columns"):  # 兼容 dataframe 风格对象。
        rows = sweep_table.to_dict(orient="records")  # 取成记录列表。
    elif isinstance(sweep_table, Mapping):  # 单行 mapping 直接当一行。
        rows = [dict(sweep_table)]  # 复制成普通 dict。
    else:
        try:
            rows = [dict(row) for row in sweep_table]  # 其余情况按映射序列处理。
        except TypeError as exc:
            raise TypeError("sweep_table must be a mapping, dataframe-like object, or iterable of mappings.") from exc

    if not rows:  # 空表不能画。
        raise ValueError("sweep_table must contain at least one row.")
    if not all(isinstance(row, dict) for row in rows):  # 每行都必须是 dict。
        raise TypeError("sweep_table rows must be mappings.")
    return rows  # 返回统一后的行列表。


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


def _resolve_axis_name(rows: list[dict[str, Any]], axis_name: str, configured_name: str | None, excluded: set[str] | None = None) -> str:
    """按规则推断 x 或 y 轴列名。"""
    if not rows:  # 空行无法推断轴名，必须显式报 ValueError 而非让下游 rows[0] 抛 IndexError。
        raise ValueError("rows must contain at least one row to infer axis name.")
    excluded = excluded or set()  # 默认不排除任何列。
    row_keys = _ordered_common_keys(rows)  # 先找公共列。
    metric_meta = get_metric_meta()  # 用于区分指标列和普通列。
    if configured_name is not None:  # 如果调用者手工指定，就优先用。
        if configured_name not in row_keys:  # 但必须确实存在。
            raise ValueError(f"{axis_name}='{configured_name}' is not present in sweep_table.")
        return configured_name  # 直接返回用户指定列。

    if axis_name == "y_axis":  # y 轴优先找指标列。
        ordered_metric_keys = [name for name in get_metric_order() if name in row_keys and name not in excluded]  # 协议顺序里存在的指标列。
        if ordered_metric_keys:  # 优先使用协议顺序里的第一个指标列。
            return ordered_metric_keys[0]  # 直接返回第一个候选。
        extra_metric_keys = [name for name in row_keys if name in metric_meta and name not in excluded]  # 其它已知指标列。
        if extra_metric_keys:  # 再退一步找其它已知指标列。
            return extra_metric_keys[0]  # 直接返回第一个候选。
        numeric_keys = [  # 再退一步找数值列。
            name
            for name in row_keys
            if name not in excluded and all(_is_real_number(row[name]) for row in rows)
        ]
        if numeric_keys:  # 如果找到了纯数值列，就拿第一个。
            return numeric_keys[0]
        raise ValueError(f"Unable to infer {axis_name} from sweep_table.")

    non_metric_keys = [name for name in row_keys if name not in metric_meta and name not in excluded]  # x 轴尽量避开指标列。
    prioritized_sweep_axes = [name for name in _SWEEP_AXIS_PRIORITY if name in non_metric_keys]  # 优先用常见 sweep 维度。
    if prioritized_sweep_axes:  # 如果常见维度存在，就拿第一个。
        return prioritized_sweep_axes[0]  # 返回最优先的维度。
    if non_metric_keys:  # 再退一步用普通非指标列。
        return non_metric_keys[0]  # 返回第一个普通列。
    fallback_keys = [name for name in row_keys if name not in excluded]  # 最后兜底到任意可用列。
    if fallback_keys:  # 只要还有列就可以返回。
        return fallback_keys[0]  # 返回最后兜底列。
    raise ValueError(f"Unable to infer {axis_name} from sweep_table.")


def _normalize_points(rows: list[dict], x_axis: str, y_axis: str) -> list[dict]:
    """把 sweep 行整理成绘图点列表。"""
    points: list[dict] = []  # 结果点集。
    for row in rows:  # 逐行转换。
        if x_axis not in row or y_axis not in row:  # 每行都必须有 x 和 y。
            raise ValueError(f"Each sweep_table row must contain '{x_axis}' and '{y_axis}'.")
        x_value = row[x_axis]  # 当前行的 x 值。
        y_value = row[y_axis]  # 当前行的 y 值。
        if not _is_real_number(y_value):  # y 必须是数值。
            raise TypeError(f"Column '{y_axis}' must be numeric.")
        y_number = float(y_value)  # 转成浮点数方便缩放。
        if not math.isfinite(y_number):  # 只接受有限值。
            raise ValueError(f"Column '{y_axis}' must contain finite numeric values.")

        point = {  # 组装单个绘图点。
            "x": x_value,  # 原始 x 值保留。
            "y": y_number,  # 归一化后的 y 值。
            "label": str(x_value),  # 默认标签直接显示 x。
        }
        points.append(point)  # 追加到点列表。

    if all(_is_real_number(point["x"]) for point in points):  # 如果 x 全是数值，就按数值排序。
        points.sort(key=lambda point: float(point["x"]))  # 按数值大小排序。
    return points  # 返回点列表。


def build_sweep_figure_spec(sweep_table: Any, figure_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """把 sweep 表整理成可渲染的图形规范对象。

    作用：先统一输入格式和配置，再自动推断或读取 x/y 轴列名，
    整理成绘图点列表，最后判断图形类型（折线或柱图）并组装规范对象。

    参数:
        sweep_table: sweep 汇总表，支持 DataFrame、单个映射或映射序列。
        figure_cfg: 可选的配置映射，控制标题、尺寸、轴名和图形类型。

    返回值:
        dict: 包含 kind、title、x_axis、y_axis、y_label、points、width、height 的规范对象。

    异常:
        TypeError: 输入类型不正确。
        ValueError: 无法推断轴名或图形类型不合法。
    """
    rows = _coerce_sweep_rows(sweep_table)  # 先统一输入格式。
    cfg = _coerce_figure_cfg(figure_cfg)  # 再统一配置格式。
    x_axis = _resolve_axis_name(rows, "x_axis", cfg.get("x_axis"))  # 推断或读取 x 轴列名。
    y_axis = _resolve_axis_name(rows, "y_axis", cfg.get("y_axis"), excluded={x_axis})  # 推断或读取 y 轴列名。
    points = _normalize_points(rows, x_axis=x_axis, y_axis=y_axis)  # 整理成点列表。

    kind = cfg.get("kind")  # 允许调用者指定图形类型。
    if kind is None:  # 没指定时自动判断。
        if all(_is_real_number(point["x"]) for point in points):  # x 全是数值就用折线。
            kind = "line"  # 选择折线图。
        else:  # 否则用柱图。
            kind = "bar"  # 选择柱图。
    if kind not in {"line", "bar"}:  # 只接受两种图。
        raise ValueError("figure_cfg.kind must be either 'line' or 'bar'.")

    metric_meta = get_metric_meta().get(y_axis, {})  # 看 y 轴是不是指标列。
    unit = metric_meta.get("unit")  # 如果有单位，就写进标签。
    if unit:  # 有单位就把单位显示出来。
        y_label = f"{y_axis} ({unit})"
    else:  # 没单位就直接用列名。
        y_label = y_axis
    title = cfg.get("title") or f"{y_axis} vs {x_axis}"  # 默认标题直接反映轴关系。

    # 画布宽度/高度：默认值与下限常量化（D9），并用 coerce_finite_scalar 守卫
    # 拒绝 bool/NaN/Inf/非数值输入（D5），与 plot_runtime/plot_calibration 对齐。
    width = max(  # 画布宽度有下限，避免太窄。
        int(coerce_finite_scalar(cfg.get("width", _SWEEP_DEFAULT_WIDTH), name="figure_cfg['width']")),
        _SWEEP_MIN_WIDTH,
    )  # 画布宽度。
    height = max(  # 画布高度有下限，避免太矮。
        int(coerce_finite_scalar(cfg.get("height", _SWEEP_DEFAULT_HEIGHT), name="figure_cfg['height']")),
        _SWEEP_MIN_HEIGHT,
    )  # 画布高度。

    spec = {  # 组装给后续绘图用的规范对象。
        "kind": kind,  # 图类型。
        "title": title,  # 标题。
        "x_axis": x_axis,  # x 轴列名。
        "y_axis": y_axis,  # y 轴列名。
        "y_label": y_label,  # y 轴显示文本。
        "points": points,  # 绘图点。
        "width": width,  # 画布宽度。
        "height": height,  # 画布高度。
    }
    return spec  # 返回规范对象。


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
        requested_path = build_output_path("figures", "sweep.svg")  # 默认落到 figures/sweep.svg。
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
        raise ValueError("render_sweep_figure currently supports SVG output only.")
    figure_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在。
    return figure_path.resolve()  # 返回绝对路径，避免后续歧义。


def _scale_linear(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    """把数值线性映射到像素区间。"""
    if math.isclose(domain_min, domain_max):  # 区间没有跨度时返回中点。
        return (range_min + range_max) / 2.0
    ratio = (value - domain_min) / (domain_max - domain_min)  # 先归一化比例。
    scaled_value = range_min + (range_max - range_min) * ratio  # 再映射到目标区间。
    return scaled_value  # 返回映射结果。


def _build_svg(spec: dict) -> str:
    """根据规范对象拼出 SVG 字符串。"""
    width = max(spec["width"], 320)  # 给一个最小宽度。
    height = max(spec["height"], 240)  # 给一个最小高度。
    margin_left = 80  # 左侧给 y 轴标签留空间。
    margin_right = 40  # 右侧留白。
    margin_top = 60  # 顶部标题空间。
    margin_bottom = 80  # 底部 x 轴标签空间。
    plot_width = width - margin_left - margin_right  # 真正绘图区宽度。
    plot_height = height - margin_top - margin_bottom  # 真正绘图区高度。
    points = spec["points"]  # 取出点列表。

    y_values = [point["y"] for point in points]  # 收集所有 y。
    y_min = min(0.0, min(y_values))  # 底部至少从 0 开始。
    y_max = max(y_values)  # 取最大值。
    if math.isclose(y_min, y_max):  # 没有跨度就人为拉开一点。
        y_max = y_min + 1.0

    x_numeric = all(_is_real_number(point["x"]) for point in points)  # 判断 x 是否全为数值。
    if x_numeric:  # 数值 x 轴画线图。
        x_values = [float(point["x"]) for point in points]  # 先把 x 转成浮点数。
        x_min = min(x_values)  # 最小 x。
        x_max = max(x_values)  # 最大 x。
        x_positions = []  # 逐个数值 x 换算像素坐标。
        for value in x_values:  # 逐个数值换算。
            x_position = _scale_linear(value, x_min, x_max, margin_left, margin_left + plot_width)  # 当前 x 的像素位置。
            x_positions.append(x_position)  # 追加到位置列表。
    else:  # 分类 x 轴画柱图并均匀分布。
        x_positions = []  # 分类轴的像素位置列表。
        count = max(len(points), 1)  # 至少按 1 个点处理。
        step = plot_width / count  # 计算每个分类占用的步长。
        for index in range(count):  # 逐个分类计算位置。
            x_position = margin_left + step * (index + 0.5)  # 让柱子居中落在格子里。
            x_positions.append(x_position)  # 追加到位置列表。

    svg_parts = []  # SVG 片段列表，后面按顺序追加。
    svg_open_tag = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'  # SVG 根标签。
    svg_background = '<rect width="100%" height="100%" fill="white"/>'  # 白色背景，避免透明底。
    svg_title = f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="20" font-family="Arial">{escape(spec["title"])}</text>'  # 顶部标题。
    svg_x_axis_line = f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="2"/>'  # x 轴基线。
    svg_y_axis_line = f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="2"/>'  # y 轴基线。
    svg_x_label = f'<text x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="14" font-family="Arial">{escape(spec["x_axis"])}</text>'  # 底部 x 轴标签。
    svg_y_label = f'<text x="24" y="{height / 2:.1f}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90, 24, {height / 2:.1f})">{escape(spec["y_label"])}</text>'  # 左侧 y 轴标签。
    svg_parts.append(svg_open_tag)  # 先写根标签。
    svg_parts.append(svg_background)  # 再写背景。
    svg_parts.append(svg_title)  # 再写标题。
    svg_parts.append(svg_x_axis_line)  # 再写 x 轴线。
    svg_parts.append(svg_y_axis_line)  # 再写 y 轴线。
    svg_parts.append(svg_x_label)  # 再写 x 轴标签。
    svg_parts.append(svg_y_label)  # 再写 y 轴标签。

    baseline_y = margin_top + plot_height  # x 轴基线对应的 y 坐标。
    if spec["kind"] == "line":  # 数值 x 轴走折线。
        polyline_points = []  # 收集折线点串。
        for point, x_position in zip(points, x_positions):  # 逐点换算像素。
            y_position = _scale_linear(point["y"], y_min, y_max, baseline_y, margin_top)  # 计算 y 像素位置。
            point_text = f"{x_position:.2f},{y_position:.2f}"  # 把点写成 SVG 坐标串。
            polyline_points.append(point_text)  # 追加到折线点串。
        polyline_text = " ".join(polyline_points)  # 把所有点串拼起来。
        polyline_svg = f'<polyline fill="none" stroke="#2563eb" stroke-width="3" points="{polyline_text}"/>'  # 折线元素。
        svg_parts.append(polyline_svg)  # 追加折线。
        for point, x_position in zip(points, x_positions):  # 每个点再单独画圆点和标签。
            y_position = _scale_linear(point["y"], y_min, y_max, baseline_y, margin_top)  # 再算一次当前点位置。
            circle_x = f"{x_position:.2f}"  # 圆点的 x 像素坐标。
            circle_y = f"{y_position:.2f}"  # 圆点的 y 像素坐标。
            circle_svg = f'<circle cx="{circle_x}" cy="{circle_y}" r="4" fill="#2563eb"/>'  # 圆点元素。
            svg_parts.append(circle_svg)  # 追加圆点。
            label_y = f"{baseline_y + 22:.2f}"  # 标签放在基线下方的位置。
            label_text = escape(point["label"])  # 标签文本需要转义。
            label_svg = f'<text x="{circle_x}" y="{label_y}" text-anchor="middle" font-size="12" font-family="Arial">{label_text}</text>'  # 标签元素。
            svg_parts.append(label_svg)  # 追加标签。
    else:  # 非数值 x 轴走柱图。
        count = max(len(points), 1)  # 至少按 1 个点处理。
        bar_width = min(72.0, plot_width / max(count * 1.6, 1.0))  # 控制柱宽不要过宽。
        for point, x_position in zip(points, x_positions):  # 逐点画柱。
            y_position = _scale_linear(point["y"], y_min, y_max, baseline_y, margin_top)  # 计算柱顶位置。
            bar_height = baseline_y - y_position  # 柱子的高度。
            bar_x = f"{x_position - bar_width / 2:.2f}"  # 柱子左边界。
            bar_y = f"{y_position:.2f}"  # 柱子顶边。
            bar_w = f"{bar_width:.2f}"  # 柱子宽度。
            bar_h = f"{bar_height:.2f}"  # 柱子高度。
            rect_svg = f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" fill="#2563eb"/>'  # 柱子元素。
            svg_parts.append(rect_svg)  # 追加柱子。
            label_x = f"{x_position:.2f}"  # 柱子标签的水平位置。
            label_y = f"{baseline_y + 22:.2f}"  # 柱子标签的垂直位置。
            label_text = escape(point["label"])  # 标签文本需要转义。
            label_svg = f'<text x="{label_x}" y="{label_y}" text-anchor="middle" font-size="12" font-family="Arial">{label_text}</text>'  # 标签元素。
            svg_parts.append(label_svg)  # 追加标签。

    tick_count = 5  # y 轴刻度数量。
    for tick_index in range(tick_count + 1):  # 逐个刻度画网格线和数值。
        tick_ratio = tick_index / tick_count  # 把当前刻度换成 0 到 1 的比例。
        tick_value = y_min + (y_max - y_min) * tick_ratio  # 再把比例映射回实际数值。
        tick_y = _scale_linear(tick_value, y_min, y_max, baseline_y, margin_top)  # 最后映射到像素坐标。
        tick_line_x1 = margin_left - 6  # 网格线左端点。
        tick_line_x2 = margin_left + plot_width  # 网格线右端点。
        tick_line_y = f"{tick_y:.2f}"  # 网格线的 y 坐标。
        line_svg = f'<line x1="{tick_line_x1}" y1="{tick_line_y}" x2="{tick_line_x2}" y2="{tick_line_y}" stroke="#d1d5db" stroke-width="1"/>'  # 网格线元素。
        svg_parts.append(line_svg)  # 追加网格线。
        tick_text_x = margin_left - 10  # 刻度文本的水平位置。
        tick_text_y = f"{tick_y + 4:.2f}"  # 刻度文本的垂直位置。
        tick_text_value = f"{tick_value:.3g}"  # 刻度显示值。
        tick_text = f'<text x="{tick_text_x}" y="{tick_text_y}" text-anchor="end" font-size="12" font-family="Arial">{tick_text_value}</text>'  # 刻度文本元素。
        svg_parts.append(tick_text)  # 追加刻度文本。

    svg_close_tag = "</svg>"  # SVG 结束标签。
    svg_parts.append(svg_close_tag)  # 追加结束标签。
    svg_text = "".join(svg_parts)  # 把所有片段拼成完整字符串。
    return svg_text  # 返回最终 SVG 文本。


def render_sweep_figure(sweep_table: Any, figure_cfg: Mapping | None) -> str | dict[str, Any]:
    """渲染 sweep 图，并返回输出路径字符串或结构化清单。

    作用：模块对外最常用的渲染入口。它先把输入规范化，再生成 SVG，
    然后写入磁盘。若调用者显式要求清单，就返回带字段的 dict，便于外层
    再核对 x 轴、y 轴、图类型和输出路径是否符合预期。

    参数:
        sweep_table: sweep 汇总表，支持 DataFrame、单个映射或映射序列。
        figure_cfg: 配置映射，控制标题、尺寸、轴名、图形类型和输出路径。

    返回值:
        str | dict[str, Any]: 默认返回输出路径字符串。如果 figure_cfg 中 return_manifest 为真，
                    则返回包含 figure_path、x_axis、y_axis、kind 的字典。

    异常:
        TypeError: 输入类型不正确。
        ValueError: 无法推断轴名、图形类型不合法、输出格式不支持或 figure_spec 缺少必需键。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "figure_cfg_keys": list(figure_cfg.keys()) if isinstance(figure_cfg, Mapping) else None,
            "sweep_table_type": type(sweep_table).__name__,
        },
        "render_sweep_figure 入口参数",
        prefix="[plotting]",
    )
    cfg = _coerce_figure_cfg(figure_cfg)  # 先统一配置格式，避免后续重复判空。
    figure_spec = build_sweep_figure_spec(sweep_table, cfg)  # 再生成可渲染规范对象。
    figure_path = _resolve_figure_path(cfg)  # 单独算出输出路径，避免和绘图逻辑混在一起。
    svg_text = _build_svg(figure_spec)  # 先把 SVG 文本生成出来，写盘前不改业务对象。
    figure_path.write_text(svg_text, encoding="utf-8")  # 再写入磁盘文件，确保编码稳定。
    if cfg.get("return_manifest"):  # 如果调用者想要清单，就返回结构化结果。可选键用 .get() 合理。
        # D3 value contract：figure_spec 的 manifest 必需键必须显式守卫，缺键属于值合同违规，
        # 按项目硬约束用 ValueError 而非 KeyError（与 plot_runtime.render_runtime_figure L608-609 口径一致）。
        for required_key in ("x_axis", "y_axis", "kind"):  # manifest 需要的三个 spec 键。
            if required_key not in figure_spec:  # 缺键守卫。
                raise ValueError(f"figure_spec is missing '{required_key}'.")  # 值合同违规抛 ValueError。
        manifest = {  # 先组装清单对象，便于外层检查字段。
            FIGURE_PATH_KEY: str(figure_path),  # 输出路径字符串，D9 单源引用常量，方便序列化和日志打印。
            "x_axis": figure_spec["x_axis"],  # x 轴列名，供上游核对。
            "y_axis": figure_spec["y_axis"],  # y 轴列名，供上游核对。
            "kind": figure_spec["kind"],  # 图类型，供上游判断是折线还是柱图。
        }
        return manifest  # 返回清单。
    return str(figure_path)  # 默认只返回路径字符串，保持最常见用法最简单。
