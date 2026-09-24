"""
文件：`src/liquidloc/plotting/plot_main_table.py`

这个模块只负责把“主表”数据渲染成一张可视化表格图片，或者先整理成
可渲染的规范化描述对象。它不负责重新计算指标，也不负责决定主表里
应该有哪些列；它只接受外部已经准备好的主表数据，然后把行、列、单元格
整理成统一结构，再用 Pillow 画成图片。

上游通常会把已经冻结好的 `main_table` 和 `figure_cfg` 传进来。
下游通常是 `scripts/13_generate_figures.py` 或相关测试，它们只关心输出的
文件路径，或者在开启清单模式时关心返回的清单对象。

这个文件最容易看错的地方有三个：
1. 主表既可以是单个 mapping，也可以是 mapping 序列。
2. 列名会做去空格和去重，但不会改变列的语义。
3. 这里的“渲染”只负责画表格，不会改指标口径，不会重排指标顺序。
"""

from __future__ import annotations  # 保持前向注解支持，配合 TYPE_CHECKING 守卫 PIL 延迟导入（D2 分层边界）。

from collections.abc import Mapping, Sequence  # 兼容字典式和序列式输入。
from liquidloc.common.constants import FIGURE_PATH_KEY, LONG_FORM_METRIC_KEY, LONG_FORM_VALUE_KEY  # D9 单源常量：长表字段名与 figure_path 键名，禁止本地重复定义字面量。
from liquidloc.common.validation import is_bool_like, is_real, is_string_like  # 统一判断布尔、标量数值和字符串类型。
from pathlib import Path  # 统一处理输出路径。
from typing import Any, TYPE_CHECKING  # 类型注解里会用到通用对象类型；TYPE_CHECKING 仅类型检查期求值，保护 PIL 延迟导入。

if TYPE_CHECKING:  # 仅类型检查期导入 PIL 类型，运行期不触发 PIL 导入，保留 render_main_table_figure 内的延迟导入契约（D2 分层边界）。
    from PIL import ImageDraw, ImageFont


# 主表图渲染布局常量（D5/D9 单源：禁止在 render_main_table_figure 函数体内重复定义魔数，
# 统一常量化以便后续调参、审计和跨文件检索）。
# 这些值仅在 plot_main_table.render_main_table_figure 内消费，不跨模块共享，故放在本模块顶层。
_TABLE_LINE_GAP: int = 4  # 单元格内多行文本间距（像素）。
_TABLE_CELL_PADDING_X: int = 12  # 单元格左右内边距（像素）。
_TABLE_CELL_PADDING_Y: int = 10  # 单元格上下内边距（像素）。
_TABLE_BORDER_WIDTH: int = 1  # 表格边框粗细（像素）。
# 主表图颜色常量（D9 单源：避免色值字面量散落在 render_main_table_figure 拼字符串代码里导致配置表面漂移）。
_TABLE_HEADER_FILL: str = "#EAEAEA"  # 表头行底色（浅灰）。
_TABLE_BODY_FILL: str = "white"  # 正文单元格底色。
_TABLE_BORDER_COLOR: str = "black"  # 单元格边框颜色。
_TABLE_TEXT_FILL: str = "black"  # 单元格文本颜色。


def _is_long_form_metric_row(row: Mapping[str, Any]) -> bool:
    """判断一行是否像 eval_pipeline 导出的 long-form metric_table 记录。"""
    # D9：引用 common/constants.py 单源常量，禁止本地 "metric"/"value" 字面量漂移。
    return isinstance(row, Mapping) and LONG_FORM_METRIC_KEY in row and LONG_FORM_VALUE_KEY in row


def _normalize_row(row: Mapping[str, Any], *, row_label: str) -> dict[str, Any]:
    """把单行主表整理成规范字典。"""
    if _is_long_form_metric_row(row):  # 主表图只能消费聚合主表，不能把 long-form 指标长表直接拿来画主表。
        raise ValueError(f"{row_label} must be an aggregated main_table row, not a long-form metric_table row.")
    normalized_row = {}  # 保存清洗后的单行数据。
    for column_name, cell_value in row.items():  # 逐个读取原始列名和值。
        # 项目硬约束：用 type() is str 精确匹配纯 str，拒绝 numpy.str_（NumPy 1.x 中 numpy.str_ 仍是 str 子类，
        # is_string_like 会漏过），与 _normalize_columns / _stringify_cell 口径一致。
        if type(column_name) is not str:  # 列名必须是纯 str，拒绝 numpy.str_。
            raise TypeError(f"{row_label} column names must be strings, got {type(column_name).__name__}.")
        normalized_column_name = column_name.strip()  # 去掉列名前后空白。
        if not normalized_column_name:  # 空列名没有意义。
            raise ValueError(f"{row_label} contains an empty column name.")
        if normalized_column_name in normalized_row:  # 同一行里不能出现重复列名。
            raise ValueError(f"{row_label} contains duplicate column name '{normalized_column_name}'.")
        normalized_row[normalized_column_name] = cell_value  # 只规范列名，不改单元格内容。
    if not normalized_row:  # 空行无法渲染。
        raise ValueError(f"{row_label} must be non-empty.")
    return normalized_row  # 返回清洗后的单行数据。


def _normalize_main_table(main_table: Mapping[Any, Any] | Sequence[Mapping[Any, Any]]) -> list[dict[str, Any]]:
    """把单个 mapping 或多行序列统一成行列表。"""
    if isinstance(main_table, Mapping):  # 单个 mapping 当成一行处理。
        if not main_table:  # 空 mapping 不能画表。
            raise ValueError("main_table must be non-empty.")
        return [_normalize_row(main_table, row_label="main_table")]  # 单行表也走统一清洗路径。
    if isinstance(main_table, Sequence) and not isinstance(main_table, (str, bytes, bytearray)):
        table_rows = list(main_table)  # 先固定成列表，避免后续迭代被外部改动。
        if not table_rows:  # 空序列没有行可画。
            raise ValueError("main_table must be non-empty.")
        normalized_rows = []  # 收集每一行的清洗结果。
        for index, row in enumerate(table_rows):  # 逐行检查类型和内容。
            if not isinstance(row, Mapping):  # 每一行都必须是 mapping。
                raise TypeError(f"main_table[{index}] must be a mapping.")
            normalized_rows.append(_normalize_row(row, row_label=f"main_table[{index}]"))  # 追加清洗后的行。
        return normalized_rows  # 返回多行表。
    raise TypeError("main_table must be a mapping or a sequence of mappings (str/bytes/bytearray are not accepted).")  # 其他类型直接拒绝。


def _normalize_columns(table_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """按首次出现顺序收集最终表头。"""
    columns: list[str] = []  # 最终列顺序。
    seen_columns: set[str] = set()  # 去重集合。
    for row_index, row in enumerate(table_rows):  # 扫描所有行。
        for column_name in row:  # 逐列收集。
            # 项目硬约束：用 type() is str 精确匹配纯 str，拒绝 numpy.str_（NumPy 1.x 中 numpy.str_ 仍是 str 子类，
            # is_string_like 会漏过），与 plot_cases._stringify_case_value / plot_main_table._stringify_cell 口径一致。
            if type(column_name) is not str:  # 列名必须是纯 str，拒绝 numpy.str_。
                raise TypeError(
                    f"main_table[{row_index}] column names must be strings, got {type(column_name).__name__}."
                )
            normalized_column_name = column_name.strip()  # 再次去空白，保证对齐。
            if not normalized_column_name:  # 空列名不进入输出。
                raise ValueError(f"main_table[{row_index}] contains an empty column name.")
            if normalized_column_name in seen_columns:  # 已见过就跳过，保留第一次出现。
                continue
            seen_columns.add(normalized_column_name)  # 记录这个列名。
            columns.append(normalized_column_name)  # 追加到输出顺序。
    if not columns:  # 整张表没有列就无法渲染。
        raise ValueError("main_table must contain at least one column.")
    return columns  # 返回列顺序。


def _stringify_cell(cell_value: Any) -> str:
    """把单元格值转成可画到图上的字符串。"""
    if cell_value is None:  # 空值显示为空字符串。
        return ""
    # 项目硬约束：bytes/bytearray 是二进制容器，str() 会产生 "b'...'" 无意义展示，必须显式拒绝。
    if isinstance(cell_value, (bytes, bytearray)):  # 二进制容器拒绝。
        raise TypeError(f"cell_value must not be bytes/bytearray, got {type(cell_value).__name__}")
    # 项目硬约束：numpy.str_ 必须拒绝，防止上游 numpy 类型静默穿透展示层。用 type() is str
    # 精确匹配纯 str（NumPy 1.x 中 numpy.str_ 仍是 str 子类，isinstance 会漏过）。
    if is_string_like(cell_value) and type(cell_value) is not str:  # numpy.str_ 拒绝，纯 str 允许。
        raise TypeError(f"cell_value must not be numpy.str_, got {type(cell_value).__name__}")
    if is_bool_like(cell_value):  # 布尔值不当作数值。
        return str(cell_value)
    if is_real(cell_value):  # 所有实数（含 numpy 浮点）用紧凑格式。
        # D5 数值安全：守卫 float() 转换异常，按 value contract 抛 ValueError（与 coerce_finite_scalar 口径一致）。
        try:  # 覆盖 TypeError/ValueError/OverflowError 三种 float() 失败路径。
            return f"{float(cell_value):g}"
        except (TypeError, ValueError, OverflowError) as exc:  # 自定义 Real 子类可能无法转 float。
            raise ValueError(f"cell_value cannot be converted to float, got {type(cell_value).__name__}: {cell_value!r}") from exc
    return str(cell_value)  # 其他类型直接转字符串。


def _split_cell_lines(cell_text: str) -> list[str]:
    """把一个单元格文本拆成多行。"""
    # 项目硬约束：cell_text 必须是纯 str（来自 _stringify_cell），拒绝 bytes/bytearray/numpy.str_。
    # 用 type() is str 精确匹配，与 _stringify_cell / _normalize_row / _normalize_columns 口径一致
    # （NumPy 1.x 中 numpy.str_ 仍是 str 子类，isinstance 会漏过；str() 会静默掩盖违规）。
    if type(cell_text) is not str:
        raise TypeError(f"cell_text must be str, got {type(cell_text).__name__}")
    lines = cell_text.splitlines()  # 按换行拆分。
    if cell_text.endswith(("\n", "\r")):  # 保留末尾空行语义。
        lines.append("")
    return lines or [""]  # 空文本也至少返回一行。


def _measure_cell(draw: ImageDraw.ImageDraw, cell_text: str, font: ImageFont.ImageFont, *, line_gap: int) -> tuple[int, int]:
    """测量单元格文本所需的宽高。"""
    widths: list[int] = []  # 每行宽度。
    heights: list[int] = []  # 每行高度。
    for line in _split_cell_lines(cell_text):  # 逐行测量。
        bbox = draw.textbbox((0, 0), line, font=font)  # 取文本边界框。
        widths.append(bbox[2] - bbox[0])  # 记录宽度。
        heights.append(bbox[3] - bbox[1])  # 记录高度。
    text_width = max(widths, default=0)  # 取最宽的一行。
    text_height = sum(heights)  # 高度按行累加。
    if heights:  # 多行之间要留间距。
        text_height += line_gap * (len(heights) - 1)
    return text_width, text_height  # 返回整段文本尺寸。


def _draw_cell_text(
    draw: ImageDraw.ImageDraw,
    cell_box: tuple[int, int, int, int],
    cell_text: str,
    font: ImageFont.ImageFont,
    *,
    line_gap: int,
    fill: str,
) -> None:
    """把单元格文本居中写进指定边界框。"""
    left, top, right, bottom = cell_box  # 拆出单元格四边。
    _, text_height = _measure_cell(draw, cell_text, font, line_gap=line_gap)  # 先测量整段文本。
    current_y = top + (bottom - top - text_height) / 2  # 垂直居中起点。
    for line in _split_cell_lines(cell_text):  # 再逐行画出。
        bbox = draw.textbbox((0, 0), line, font=font)  # 当前行的边界框。
        line_width = bbox[2] - bbox[0]  # 当前行宽度。
        line_height = bbox[3] - bbox[1]  # 当前行高度。
        current_x = left + (right - left - line_width) / 2  # 每行水平居中。
        draw.text((current_x, current_y), line, font=font, fill=fill)  # 真正落笔。
        current_y += line_height + line_gap  # 下一行往下移。


def build_main_table_figure_spec(
    main_table: Mapping[Any, Any] | Sequence[Mapping[Any, Any]],
    figure_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """把主表整理成可渲染的规范对象。

    这个函数只做输入校验、行列整理和文本规范化，不做绘图。
    返回值里会保留输出路径、列顺序、单元格文本和行数，供渲染函数直接使用。

    参数:
        main_table: 主指标表，可以是单个映射或映射序列。
        figure_cfg: 渲染配置映射，必须包含 figure_path 键。

    返回值:
        dict: 包含 figure_path、columns、cell_text、row_count 的规范对象。

    异常:
        TypeError: 输入类型不正确。
        ValueError: 缺少 figure_path、输入为空或列名重复。
    """
    if not isinstance(figure_cfg, Mapping):  # 配置必须可按键访问。
        raise TypeError("figure_cfg must be a mapping.")
    if FIGURE_PATH_KEY not in figure_cfg:  # 输出路径是必需项（D9 单源引用常量）。
        # 缺键属于值合同违规，按项目硬约束用 ValueError 而非 KeyError（与 plot_cases.render_case_figures L371-372 口径一致）。
        raise ValueError(f"figure_cfg is missing '{FIGURE_PATH_KEY}'.")

    table_rows = _normalize_main_table(main_table)  # 先统一成行列表。
    columns = _normalize_columns(table_rows)  # 再收集表头顺序。
    cell_text = [[_stringify_cell(row.get(column_name, "")) for column_name in columns] for row in table_rows]  # 逐行逐列转字符串。
    return {  # 返回给渲染函数的中间对象。
        FIGURE_PATH_KEY: Path(figure_cfg[FIGURE_PATH_KEY]),  # 输出路径保持为 Path（D9 单源引用常量）。
        "columns": columns,  # 表头顺序。
        "cell_text": cell_text,  # 每个单元格的最终显示文本。
        "row_count": len(table_rows),  # 行数供外层估算尺寸。
    }


def render_main_table_figure(
    main_table: Mapping[Any, Any] | Sequence[Mapping[Any, Any]],
    figure_cfg: Mapping[str, Any],
) -> str:
    """把主表渲染成图片并返回路径字符串。

    调用者传入表格数据和渲染配置后，这里会完成规范化、测量尺寸、
    创建画布、绘制表头和单元格、最后保存图片并返回路径字符串。

    参数:
        main_table: 主指标表，可以是单个映射或映射序列。
        figure_cfg: 渲染配置映射，必须包含 figure_path 键。

    返回值:
        str: 输出图片的路径字符串。

    异常:
        RuntimeError: Pillow 未安装时抛出。
        TypeError/ValueError: 输入校验失败时由 build_main_table_figure_spec 传播；
            figure_cfg 不是 mapping、缺少 figure_path、main_table 为空或列名重复等
            值合同违规均按项目硬约束抛 ValueError 而非 KeyError。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "figure_cfg_keys": list(figure_cfg.keys()) if isinstance(figure_cfg, Mapping) else None,
            "main_table_type": type(main_table).__name__,
            "main_table_len": len(main_table) if hasattr(main_table, "__len__") else None,
        },
        "render_main_table_figure 入口参数",
        prefix="[plotting]",
    )
    figure_spec = build_main_table_figure_spec(main_table, figure_cfg)  # 先拿到规范对象。
    try:  # Pillow 是实际绘图依赖。
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # 没装 Pillow 就直接报错。
        raise RuntimeError("Pillow is required to render main-table figures.") from exc

    figure_path = figure_spec[FIGURE_PATH_KEY]  # 最终输出文件路径（D9 单源引用常量）。
    figure_path.parent.mkdir(parents=True, exist_ok=True)  # 先确保目录存在。
    font = ImageFont.load_default()  # 使用默认字体。
    line_gap = _TABLE_LINE_GAP  # 多行文本间距（D5/D9 单源常量）。
    cell_padding_x = _TABLE_CELL_PADDING_X  # 左右内边距（D5/D9 单源常量）。
    cell_padding_y = _TABLE_CELL_PADDING_Y  # 上下内边距（D5/D9 单源常量）。
    border_width = _TABLE_BORDER_WIDTH  # 表格边框宽度（D5/D9 单源常量）。

    dummy_image = Image.new("RGB", (1, 1), "white")  # 创建测量用占位图。
    dummy_draw = ImageDraw.Draw(dummy_image)  # 在占位图上测字宽高。
    table_rows = [figure_spec["columns"], *figure_spec["cell_text"]]  # 表头加正文组成完整表格。
    column_widths = [0] * len(figure_spec["columns"])  # 每列先从 0 开始累计最大宽度。
    row_heights = []  # 每行高度单独计算。
    for row in table_rows:  # 逐行计算所需空间。
        row_height = 0  # 当前行的最大高度。
        for column_index, cell_text in enumerate(row):  # 逐列测量单元格。
            text_width, text_height = _measure_cell(dummy_draw, cell_text, font, line_gap=line_gap)  # 计算文本尺寸。
            column_widths[column_index] = max(column_widths[column_index], text_width + cell_padding_x * 2)  # 更新列宽。
            row_height = max(row_height, text_height + cell_padding_y * 2)  # 更新行高。
        row_heights.append(row_height)  # 记录这一行的最终高度。

    image_width = sum(column_widths) + border_width * (len(column_widths) + 1)  # 整张图宽度。
    image_height = sum(row_heights) + border_width * (len(row_heights) + 1)  # 整张图高度。
    image = Image.new("RGB", (image_width, image_height), "white")  # 创建最终画布。
    draw = ImageDraw.Draw(image)  # 创建绘图对象。

    current_y = border_width  # 从顶部边框下面开始。
    for row_index, row in enumerate(table_rows):  # 逐行绘制。
        current_x = border_width  # 每行都从最左边开始。
        for column_index, cell_text in enumerate(row):  # 逐列绘制。
            cell_width = column_widths[column_index]  # 当前列宽。
            cell_height = row_heights[row_index]  # 当前行高。
            cell_box = (  # 当前单元格边界。
                current_x,  # 左边界 x，表示这一格从哪里开始。
                current_y,  # 上边界 y，表示这一格的顶部位置。
                current_x + cell_width,  # 右边界 x，表示这一格的结束位置。
                current_y + cell_height,  # 下边界 y，表示这一格的底部位置。
            )
            is_header_row = row_index == 0  # 这一行是不是表头。
            if is_header_row:  # 表头单独用浅灰底。
                cell_fill = _TABLE_HEADER_FILL  # D5/D9 单源常量。
            else:  # 正文保持白底。
                cell_fill = _TABLE_BODY_FILL  # D5/D9 单源常量。
            cell_outline = _TABLE_BORDER_COLOR  # 边框颜色（D5/D9 单源常量）。
            cell_border_width = border_width  # 边框宽度单独保存，便于检查。
            draw.rectangle(  # 先画这个单元格的底色和边框。
                cell_box,  # 这个单元格的四个边界坐标。
                fill=cell_fill,  # 这个单元格的底色。
                outline=cell_outline,  # 这个单元格的边框颜色。
                width=cell_border_width,  # 这个单元格的边框宽度。
            )
            _draw_cell_text(  # 再把文字居中写入。
                draw,  # 传入当前画布上的绘图对象。
                cell_box,  # 传入当前单元格边界。
                cell_text,  # 传入当前单元格文本。
                font,  # 传入当前字体。
                line_gap=line_gap,  # 传入行间距。
                fill=_TABLE_TEXT_FILL,  # 文字颜色（D5/D9 单源常量）。
            )
            current_x += cell_width + border_width  # 移到下一列。
        current_y += row_heights[row_index] + border_width  # 移到下一行。

    image.save(figure_path, dpi=(300, 300))  # J-8: ≥300 dpi per handbook
    return str(figure_path)  # 返回字符串路径。
