"""
文件：`src/liquidloc/plotting/plot_cases.py`

这个模块负责把已经选择好的案例组渲染成图片，并把生成结果整理成清单。
它不负责挑案例，也不负责重新排序案例；它只把外部已经冻结好的 case group
按组画成“文本优先”的图片。

上游通常会传入 main_cases、failure_cases、boundary_cases 三组数据。
下游通常是 figures 生成脚本、测试，或者需要把案例图路径写进 manifest 的流程。

最容易看错的地方：
1. 每个案例组都必须有 case_ref，或者至少有 seq_id。
2. 这里画的是文本信息图，不是流程图，也不是表格图。
3. figure_manifest 只是把组名、路径和 case_refs 收集起来，不做额外改写。
"""

from __future__ import annotations  # 保持前向注解支持。

from collections.abc import Iterable, Mapping  # 兼容案例组输入和字典式对象。
from liquidloc.common.constants import CASE_GROUP_NAMES, CASE_REF_KEY, DEFAULT_CASE_FIGURE_SUFFIX, FIGURE_PATH_KEY, SEQ_ID_KEY  # D9 单源：案例分组名、身份字段名、案例图默认后缀与 figure_path 键名常量。
from liquidloc.common.validation import is_bool_like, is_real, is_string_like  # 统一判断布尔、标量数值和字符串类型。
from liquidloc.analysis.case_selector import _extract_case_ref  # D9 单源：复用权威定义，禁止本地重复。
from pathlib import Path  # 输出路径统一处理。
from typing import Any, TYPE_CHECKING  # 类型注解里会用到通用对象类型；TYPE_CHECKING 仅类型检查期求值，保护 PIL 延迟导入。

if TYPE_CHECKING:  # 仅类型检查期导入 PIL 类型，运行期不触发 PIL 导入，保留 render_case_group 内的延迟导入契约（D2 分层边界）。
    from PIL import ImageDraw, ImageFont


_CASE_GROUP_NAMES = CASE_GROUP_NAMES  # D9 单源：引用 common/constants.py 常量，禁止本地重复定义。固定三组案例名称，顺序不能随便改。

# 渲染布局常量（D5/D9 单源：禁止在 render_case_group 函数体内重复定义魔数，
# 统一常量化以便后续调参、审计和跨文件检索）。
# 这些值仅在 plot_cases.render_case_group 内消费，不跨模块共享，故放在本模块顶层。
_DEFAULT_FONT_SIZE: int = 14  # ImageFont.load_default 默认字号（Pillow >= 10.1.0 支持 size 参数）。
_LINE_GAP: int = 4  # 文本块内行间距（像素）。
_OUTER_PADDING: int = 16  # 图片四周留白（像素）。
_CASE_PADDING: int = 10  # 案例框内留白（像素）。
_SECTION_GAP: int = 12  # 案例块之间的间距（像素）。
_HEADER_GAP: int = 10  # 标题与首个案例块之间的间距（像素）。
_BORDER_WIDTH: int = 1  # 案例框边框粗细（像素）。
_MIN_IMAGE_WIDTH: int = 320  # 图片最小宽度（像素），避免极小文本导致画布过窄。
_MIN_IMAGE_HEIGHT: int = 160  # 图片最小高度（像素），避免空组导致画布过矮。


# 案例组归一器：把任意可迭代的案例组整理成标准字典列表。
def _normalize_case_group(case_group: Any, *, group_name: str) -> list[dict[str, Any]]:  # 把一组案例转成规范列表。
    # 这里的 group_name 用于报错定位和后面 manifest 的分组 key。
    # case_group 是一整组原始案例，后面会被逐条复制和补齐 case_ref。
    if is_string_like(case_group) or isinstance(case_group, (bytes, bytearray, Mapping)) or not isinstance(case_group, Iterable):  # 排除错误容器；is_string_like 同时拒绝 str 与 numpy.str_（项目硬约束）。
        raise TypeError(f"selected_cases.{group_name} must be an iterable of case mappings")

    normalized_group: list[dict[str, Any]] = []  # 保存规范化后的案例。
    for index, case_obj in enumerate(case_group):  # 逐个案例处理。
        case_ref = _extract_case_ref(case_obj, location=f"selected_cases.{group_name}[{index}]")  # 取出稳定标识。
        normalized_case = dict(case_obj)  # 复制一份，避免改动输入对象。
        normalized_case[CASE_REF_KEY] = case_ref  # 统一写回 case_ref，方便后续显示。
        normalized_group.append(normalized_case)  # 加入当前组。
    return normalized_group  # 返回规范化后的组。


# 组迭代器：固定按 main/failure/boundary 顺序返回三组案例。
def iterate_case_groups(selected_cases: Mapping[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:  # 把三组案例整理成固定顺序列表。
    """规范化并返回主案例、失败案例和边界案例三组。

    作用：按固定顺序（main_cases、failure_cases、boundary_cases）
    遍历 selected_cases，逐组规范化后返回有序列表。
    每组都必须存在于 selected_cases 中，否则抛出 ValueError。

    参数:
        selected_cases: 案例选择结果映射，必须包含三组固定名称。

    返回值:
        list[tuple[str, list[dict]]]: 按固定顺序排列的 (组名, 案例列表) 元组列表。

    异常:
        TypeError: selected_cases 不是映射。
        ValueError: 缺少某个必需的分组。
    """

    # selected_cases 是最外层的顶层容器，必须包含三组固定名字。
    if not isinstance(selected_cases, Mapping):  # 顶层必须是 mapping。
        raise TypeError(f"selected_cases must be a mapping, got {type(selected_cases).__name__}")

    normalized_groups: list[tuple[str, list[dict[str, Any]]]] = []  # 保存按固定顺序整理好的组。
    for group_name in _CASE_GROUP_NAMES:  # 固定按三组顺序处理。
        if group_name not in selected_cases:  # 每组都必须存在。
            raise ValueError(f"selected_cases is missing '{group_name}'")  # 缺少必需分组属于值契约违规，按项目硬约束用 ValueError 而非 KeyError。
        normalized_groups.append(  # 把当前组打包进有序结果里。
            (  # 二元组开始：组名 + 组内标准化案例列表。
                group_name,  # 这一组的固定名字，后面会作为 manifest key。
                _normalize_case_group(selected_cases[group_name], group_name=group_name),  # 把这一组案例先统一成标准字典列表。
            )  # 二元组结束。
        )  # append 结束。
    return normalized_groups  # 返回三组案例的有序列表。


# 值转文本器：把案例字段值转成适合图上显示的字符串。
def _stringify_case_value(case_value: Any) -> str:  # 把案例字段值转成展示文本。
    # 项目硬约束：bytes/bytearray 是二进制容器，str() 会产生 "b'...'" 无意义展示，
    # 必须显式拒绝（与 plot_cases._normalize_case_group L50 及 model_factory 口径一致）。
    if isinstance(case_value, (bytes, bytearray)):  # 二进制容器拒绝。
        raise TypeError(f"case_value must not be bytes/bytearray, got {type(case_value).__name__}")
    # 项目硬约束：numpy.str_ 必须拒绝，防止上游 numpy 类型静默穿透展示层。用 type() is str
    # 精确匹配纯 str（NumPy 1.x 中 numpy.str_ 仍是 str 子类，isinstance 会漏过），
    # 与 plot_calibration._resolve_figure_path L129 / _coerce_numeric L201 口径一致。
    if is_string_like(case_value) and type(case_value) is not str:  # numpy.str_ 拒绝，纯 str 允许。
        raise TypeError(f"case_value must not be numpy.str_, got {type(case_value).__name__}")
    if is_bool_like(case_value):  # 布尔值不当作数值。
        return str(case_value)
    if is_real(case_value):  # 所有实数（含 numpy 浮点）用紧凑格式。
        # D5 数值安全：is_real 用 numbers.Real ABC 判断，自定义 Real 子类可能 __float__ 失败。
        # 守卫 float() 转换异常，按 value contract 抛 ValueError（与 coerce_finite_scalar 口径一致）。
        try:  # 覆盖 TypeError/ValueError/OverflowError 三种 float() 失败路径。
            return f"{float(case_value):g}"
        except (TypeError, ValueError, OverflowError) as exc:  # 自定义 Real 子类可能无法转 float。
            raise ValueError(f"case_value cannot be converted to float, got {type(case_value).__name__}: {case_value!r}") from exc
    return str(case_value)  # 其他类型直接字符串化。


# 案例文本构造器：把一个案例的字段展开成多行可读文本。
def _build_case_text(case_obj: Mapping[str, Any]) -> str:  # 把一个案例组织成多行文本。
    # 这里把身份字段放在第一行，其余字段按原字典顺序依次展开。
    identity_field = CASE_REF_KEY if CASE_REF_KEY in case_obj else SEQ_ID_KEY  # D9 单源：引用常量决定身份字段。
    if identity_field not in case_obj:  # D3 value contract：既无 case_ref 也无 seq_id 属于值合同违例，按项目硬约束抛 ValueError 而非 KeyError。
        raise ValueError(f"case_obj must contain '{CASE_REF_KEY}' or '{SEQ_ID_KEY}'")
    text_lines = [f"{identity_field}: {_stringify_case_value(case_obj[identity_field])}"]  # 第一行先写身份。
    for field_name, field_value in case_obj.items():  # 其余字段逐个写出来。
        if field_name == identity_field:  # 身份字段已经写过了，跳过。
            continue
        text_lines.append(f"{field_name}: {_stringify_case_value(field_value)}")  # 每个字段都单独成行。
    return "\n".join(text_lines)  # 用换行拼成展示文本。


# 文本块测量器：先量尺寸，避免后续框体过窄或过矮。
def _measure_block(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, *, line_gap: int) -> tuple[int, int]:  # 估算文本块尺寸。
    # draw 是临时画笔，text 是待测文本，font 和 line_gap 一起决定最终框体大小。
    widths: list[int] = []  # 逐行宽度。
    heights: list[int] = []  # 逐行高度。
    for line in text.splitlines() or [""]:  # 空文本也按一行处理，避免测量结果为零。
        bbox = draw.textbbox((0, 0), line, font=font)  # 测量当前行尺寸。
        widths.append(bbox[2] - bbox[0])  # 记录宽度。
        heights.append(bbox[3] - bbox[1])  # 记录高度。

    total_height = sum(heights)  # 总高度先加起来。
    if heights:  # 多行之间要加行间距。
        total_height += line_gap * (len(heights) - 1)  # 这里只算相邻行之间的空隙。
    return max(widths, default=0), total_height  # 返回总宽和总高。


# 文本块绘制器：按测量结果逐行写入文本框内部。
def _draw_block(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, *, origin_x: int, origin_y: int, line_gap: int) -> int:  # 把文本块逐行画到画布上。
    current_y = origin_y  # 当前行 y 坐标从起点开始。
    lines = text.splitlines() or [""]  # 空文本也至少画一行。
    for index, line in enumerate(lines):  # 逐行绘制，需区分首行以正确施加行间间距。
        if index > 0:  # 行间间距只在相邻行之间加，最后一行后不加，与 _measure_block L146-147 口径一致（D3 value contract）。
            current_y += line_gap  # 先下移行间距，再画当前行。
        draw.text((origin_x, current_y), line, font=font, fill="black")  # 直接绘制当前行。
        bbox = draw.textbbox((origin_x, current_y), line, font=font)  # 再测一次当前行高度，方便换到下一行。
        current_y += bbox[3] - bbox[1]  # 下移到当前行底部，不在最后一行后追加 line_gap。
    return current_y  # 返回绘制结束后的 y 坐标（= origin_y + 总高，与 _measure_block 总高严格一致，无尾部 line_gap）。


# 组路径解析器：给每个案例组派生一个独立的输出文件名。
def _resolve_group_figure_path(figure_path: Path, *, group_name: str) -> Path:  # 为每个组生成独立输出路径。
    if figure_path.suffix:  # 如果原路径已经有后缀，就在文件名后面插入组名。
        return figure_path.with_name(f"{figure_path.stem}_{group_name}{figure_path.suffix}")
    return figure_path.with_name(f"{figure_path.name}_{group_name}{DEFAULT_CASE_FIGURE_SUFFIX}")  # 没后缀就走 D9 单源默认后缀。


# 单组渲染函数：把一个案例组画成文本优先的图片。
def render_case_group(  # 把单个案例组渲染成图片。
    case_group: Iterable[Mapping[str, Any]],
    figure_path: str | Path,
    *,
    group_name: str,
) -> str:
    """把一个案例组渲染成文本优先的图片，并返回输出路径。

    作用：先规范化案例组，再逐个案例构造文本块，测量尺寸后创建画布，
    最后在画布上画标题、边框和文本，保存为 PNG 图片。

    参数:
        case_group: 当前案例组的原始数据，可以是映射的迭代。
        figure_path: 输出图片的基础路径，最终文件名会插入组名后缀。
        group_name: 案例组名称（如 main_cases、failure_cases、boundary_cases），
                    用于标题显示和文件名派生。

    返回值:
        str: 渲染后图片的绝对路径字符串。

    异常:
        TypeError: group_name 不是字符串，或 case_group 中存在非映射对象。
        ValueError: group_name 为空字符串，或 case_ref/seq_id 为空。
        RuntimeError: Pillow 未安装。
    """

    # figure_path 是这组案例图的基础路径，group_name 用来派生最终文件名。
    # case_group 则是这一组原始案例数据，后面会被规范化成可绘制文本块。
    if type(group_name) is not str:  # 组名必须是纯 str，拒绝 numpy.str_（硬约束），与 build_case_figure_manifest 口径一致。
        raise TypeError(f"group_name must be a string, got {type(group_name).__name__}")
    if not group_name.strip():  # 组名不能是空白。
        raise ValueError("group_name must be a non-empty string")

    normalized_group = _normalize_case_group(case_group, group_name=group_name)  # 先把组内案例规范化。
    group_figure_path = Path(figure_path).resolve()  # 输出路径统一转成绝对路径。

    try:  # Pillow 是绘图依赖，缺了就报错。
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is required to render case figures.") from exc

    try:  # Pillow >= 10.1.0 支持 size 参数，优先使用。
        font = ImageFont.load_default(size=_DEFAULT_FONT_SIZE)  # 新版 API，可指定字号。
    except TypeError:  # 旧版 Pillow 不支持 size 参数，回退到无参数版本。
        font = ImageFont.load_default()  # 使用默认字体，降低环境依赖。
    line_gap = _LINE_GAP  # 行间距。
    outer_padding = _OUTER_PADDING  # 图片四周留白。
    case_padding = _CASE_PADDING  # 框内留白。
    section_gap = _SECTION_GAP  # 案例块之间的间距。
    header_gap = _HEADER_GAP  # 标题和内容之间的间距。
    border_width = _BORDER_WIDTH  # 框线粗细。

    dummy_image = Image.new("RGB", (1, 1), "white")  # 用于测量文字的临时画布。
    dummy_draw = ImageDraw.Draw(dummy_image)  # 临时画笔。

    title_text = f"{group_name} ({len(normalized_group)})"  # 标题里写组名和数量。
    title_width, title_height = _measure_block(dummy_draw, title_text, font, line_gap=line_gap)  # 先测标题尺寸。

    case_blocks = []  # 保存每个案例块的文本和尺寸。
    max_block_width = title_width  # 初始最大宽度先用标题宽度。
    total_block_height = title_height + header_gap  # 总高度先加标题和间距。
    for case_obj in normalized_group:  # 逐个案例构造文本块。
        block_text = _build_case_text(case_obj)  # 先把案例字段拼成多行文本。
        block_width, block_height = _measure_block(dummy_draw, block_text, font, line_gap=line_gap)  # 再测尺寸。
        case_blocks.append((block_text, block_width, block_height))  # 保存文本和尺寸。
        max_block_width = max(max_block_width, block_width)  # 更新最大宽度。
        total_block_height += block_height + case_padding * 2 + border_width * 2 + section_gap  # 累加总高度。

    if not normalized_group:  # 如果没有案例，就画一个空占位文本，保证仍然有可见输出。
        empty_text = "no cases"  # 空状态文本。
        empty_width, empty_height = _measure_block(dummy_draw, empty_text, font, line_gap=line_gap)  # 测空文本尺寸。
        case_blocks.append((empty_text, empty_width, empty_height))  # 把空文本也作为一个块。
        max_block_width = max(max_block_width, empty_width)  # 更新最大宽度。
        total_block_height += empty_height + case_padding * 2 + border_width * 2  # 累加空块高度。
    else:  # 非空组时，最后一个块后面不再额外留 section_gap。
        total_block_height -= section_gap  # 最后一个块后面不需要间距。

    image_width = max(_MIN_IMAGE_WIDTH, max_block_width + outer_padding * 2 + case_padding * 2 + border_width * 2)  # 最终宽度。
    image_height = max(_MIN_IMAGE_HEIGHT, total_block_height + outer_padding * 2)  # 最终高度。

    group_figure_path.parent.mkdir(parents=True, exist_ok=True)  # 先建输出目录。
    image = Image.new("RGB", (image_width, image_height), "white")  # 创建最终画布。
    draw = ImageDraw.Draw(image)  # 创建正式画笔。

    current_y = outer_padding  # 从顶部留白开始画。
    draw.text((outer_padding, current_y), title_text, font=font, fill="black")  # 先画标题。
    current_y += title_height + header_gap  # 标题后下移。

    for block_index, (block_text, block_width, block_height) in enumerate(case_blocks):  # 逐个案例块画框和文本。
        box_left = outer_padding  # 框左边界。
        box_top = current_y  # 框顶边界。
        box_right = image_width - outer_padding  # 框右边界。
        box_bottom = box_top + block_height + case_padding * 2 + border_width * 2  # 框底边界。
        draw.rectangle((box_left, box_top, box_right, box_bottom), outline="black", width=border_width)  # 画边框。
        _draw_block(  # 把当前块文本画进对应框内。
            draw,
            block_text,  # 当前块的多行文本。
            font,  # 绘制字体。
            origin_x=box_left + case_padding + border_width,  # 内容起点 x。
            origin_y=box_top + case_padding + border_width,  # 内容起点 y。
            line_gap=line_gap,  # 块内行间距。
        )  # 当前块绘制结束。
        current_y = box_bottom + section_gap  # 移到下一块位置。
        if block_index == len(case_blocks) - 1:  # 最后一块后面不留间距。
            current_y = box_bottom

    image.save(group_figure_path)  # 保存图片。
    return str(group_figure_path)  # 返回路径字符串。


# 清单登记函数：把一组案例图和 case_refs 收进 manifest。
def build_case_figure_manifest(  # 把某组案例图登记进清单。
    figure_manifest: Mapping[str, Any],
    *,
    group_name: str,
    case_group: Any,
    figure_path: str,
) -> dict[str, dict[str, Any]]:
    """把一个案例组的渲染结果收集到 figure manifest 里。

    作用：把当前组的输出路径和 case_ref 列表以组名为键写入 manifest，
    方便上层汇总所有案例组的渲染结果。

    参数:
        figure_manifest: 已有的清单映射，会被复制后追加当前组信息。
        group_name: 案例组名称，作为 manifest 中的键。
        case_group: 当前案例组的原始数据，用于提取 case_ref 列表。
        figure_path: 当前组渲染后的图片路径字符串。

    返回值:
        dict: 更新后的 manifest 字典，新增以 group_name 为键的条目，
              包含 figure_path 和 case_refs 两个字段。

    异常:
        TypeError: figure_manifest 不是映射，group_name 不是字符串，
                   或 figure_path 不是字符串。
        ValueError: group_name 或 figure_path 为空字符串。
    """

    # 这里把当前组的文件路径和 case_refs 一起写进 manifest，方便上层汇总。
    # case_group 这里会被重新规范化一遍，是为了从中稳定提取 case_ref 列表。
    if not isinstance(figure_manifest, Mapping):  # 清单必须是 mapping。
        raise TypeError(f"figure_manifest must be a mapping, got {type(figure_manifest).__name__}")
    # 项目硬约束：用 type() is str 精确匹配纯 str，拒绝 numpy.str_（NumPy 1.x 中 numpy.str_ 仍是 str 子类，
    # is_string_like 会漏过），与 plot_calibration._resolve_figure_path L129 / _stringify_case_value L107 口径一致。
    # bytearray 落入 else 分支一并拒绝（bytearray 既不是 str 也不是 Mapping）。
    if type(group_name) is not str:  # 组名必须是纯字符串（拒绝 numpy.str_）。
        raise TypeError(f"group_name must be a str, got {type(group_name).__name__}")
    if not group_name.strip():  # 组名不能为空。
        raise ValueError("group_name must be a non-empty string")
    if type(figure_path) is not str:  # 输出路径必须是纯字符串（拒绝 numpy.str_）。
        raise TypeError(f"figure_path must be a str, got {type(figure_path).__name__}")
    if not figure_path.strip():  # 路径不能是空字符串。
        raise ValueError("figure_path must be a non-empty string")

    normalized_group = _normalize_case_group(case_group, group_name=group_name)  # 先规范化案例组，保证 case_ref 一致。
    manifest = dict(figure_manifest)  # 复制一份，避免改到外部对象。
    manifest[group_name] = {  # 以组名作为清单 key。
        FIGURE_PATH_KEY: figure_path,  # 记录这组图的路径（D9 单源引用常量）。
        "case_refs": [case_obj[CASE_REF_KEY] for case_obj in normalized_group],  # 记录这组里有哪些 case。
    }
    return manifest  # 返回更新后的清单。


# 总渲染函数：一次处理三组案例并汇总 figure manifest。
def render_case_figures(selected_cases: Mapping[str, Any], figure_cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:  # 把三组案例全部渲染成图，并汇总成清单。
    """从冻结好的 selected_cases 生成分组案例图和 figure manifest。

    作用：按固定顺序（main_cases、failure_cases、boundary_cases）遍历三组案例，
    逐组渲染成图片，并把每组的输出路径和 case_ref 列表汇总到 manifest 中。

    参数:
        selected_cases: 案例选择结果映射，必须包含 main_cases、
                        failure_cases、boundary_cases 三组。
        figure_cfg: 渲染配置映射，必须包含 figure_path 键。

    返回值:
        dict: 以组名为键的 manifest 字典，每个值包含 figure_path 和 case_refs。

    异常:
        TypeError: selected_cases 或 figure_cfg 不是映射。
        ValueError: selected_cases 缺少某个分组，或 figure_cfg 缺少 figure_path。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "figure_cfg_keys": list(figure_cfg.keys()) if isinstance(figure_cfg, Mapping) else None,
            "selected_cases_groups": list(selected_cases.keys()) if isinstance(selected_cases, Mapping) else None,
        },
        "render_case_figures 入口参数",
        prefix="[plotting]",
    )

    # figure_cfg 里至少要有根 figure_path，selected_cases 则是三组案例的顶层容器。
    # base_figure_path 是每组派生文件名的基础，case_groups 是固定顺序的三组数据。
    if not isinstance(figure_cfg, Mapping):  # 配置必须是 mapping。
        raise TypeError(f"figure_cfg must be a mapping, got {type(figure_cfg).__name__}")
    if FIGURE_PATH_KEY not in figure_cfg:  # 输出根路径是必需项（D9 单源引用常量）。
        raise ValueError(f"figure_cfg is missing '{FIGURE_PATH_KEY}'.")  # 缺键属于值合同违规，按项目硬约束用 ValueError 而非 KeyError（与 iterate_case_groups L88 口径一致）。

    base_figure_path = Path(figure_cfg[FIGURE_PATH_KEY])  # 把根路径转成 Path（D9 单源引用常量）。
    case_groups = iterate_case_groups(selected_cases)  # 先把三组案例规范化并固定顺序。

    figure_manifest: dict[str, dict[str, Any]] = {}  # 最终清单容器。
    for group_name, case_group in case_groups:  # 逐组渲染。
        group_figure_path = _resolve_group_figure_path(base_figure_path, group_name=group_name)  # 为每组生成独立文件名。
        # 这一段先画出当前组自己的图，再把这组的输出路径写回 manifest。
        rendered_figure_path = render_case_group(  # 先把当前组渲染出来。
            case_group,  # 当前组的案例列表。
            group_figure_path,  # 当前组最终输出的图片路径。
            group_name=group_name,  # 当前组名称，用于文件名和标题。
        )  # 当前组渲染完成。
        # 这里把组名、渲染后路径和 case_refs 一起收集起来，方便上层直接消费。
        figure_manifest = build_case_figure_manifest(  # 再把输出路径和 case_refs 收进清单。
            figure_manifest,  # 这里传入当前累计的清单，后续会在此基础上继续追加。
            group_name=group_name,  # 当前案例组的组名，作为 manifest 的 key。
            case_group=case_group,  # 当前案例组本体，用来提取 case_ref 列表。
            figure_path=rendered_figure_path,  # 当前组渲染后得到的图路径。
        )  # 当前组清单登记完成。

    return figure_manifest  # 返回汇总后的清单。
