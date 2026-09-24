"""配置读取、递归合并和占位扫描的共用工具。

职责：
    1. 读取 `configs/` 下的 YAML 文件。
    2. 把多个配置片段按覆盖顺序递归合并成最终配置。
    3. 扫描配置里还写着 `TBD` 的位置，方便在真正运行前发现未完成项。

上游依赖：
    - PyYAML（可选）                — 优先使用成熟 YAML 解析器
    - Python 标准库 re / pathlib    — 回退解析器的基础设施

下游调用者：
    - liquidloc.protocol.*           — 协议层用 load_yaml_config 读取协议配置
    - liquidloc.pipelines.*          — 流水线层用 merge_configs 合并多级配置
    - scripts / smoke tests          — 用 collect_tbd_paths 检查配置完整性

核心变量：
    - _FLOAT_PATTERN                 — 匹配浮点数/科学计数法的正则对象
    - _yaml                          — PyYAML 模块引用或 None（回退标志）

实现说明：
    优先使用 PyYAML；如果环境里没有安装 PyYAML，就退回到一个只覆盖
    本项目当前配置写法的简化解析器。这个回退不是为了支持完整 YAML，而是
    为了让最小运行环境也能稳定读到本仓库的配置文件。
"""

from __future__ import annotations  # 让函数签名里的前向引用类型保持可用。

import re  # 用于识别数字格式和简单文本模式。
from copy import deepcopy  # 用于深拷贝配置值，避免合并结果与输入共享可变引用。
from pathlib import Path  # 用 Path 统一处理文件路径。
from typing import Any  # 这里只用于类型注解，不参与运行逻辑。

try:  # 优先使用成熟的 PyYAML 实现。
    import yaml as _yaml  # 如果成功导入，就直接用它做 YAML 解析。
except ModuleNotFoundError:  # 环境里没有 PyYAML 时走回退逻辑。
    _yaml = None  # 用 None 表示当前没有 PyYAML。

__all__ = (
    "find_project_root",
    "load_yaml_config",
    "load_dataset_config",
    "merge_configs",
    "collect_tbd_paths",
)


class _DuplicateKeyChecker:
    """为 PyYAML 的 SafeLoader 注入重复键检测。

    PyYAML 默认允许重复键（后者覆盖前者），这在协议配置文件中是
    危险的——意外重复键会导致前值被静默丢弃。此 checker 在构造
    mapping 时记录已见键，遇到重复键时立即抛出 ValueError。

    注意：patch 会修改 SafeLoader.construct_mapping 的全局状态，
    这是有意为之——确保整个进程生命周期内所有 YAML 加载都受保护。
    提供 unpatch() 方法用于测试场景中恢复原始行为。
    """

    _patched: bool = False  # 确保只 patch 一次。
    _original_construct_mapping = None  # 保存原始方法以便 unpatch。

    @classmethod
    def patch(cls) -> None:
        """向 SafeLoader 注入重复键检测，仅执行一次。"""
        if cls._patched or _yaml is None:
            cls._patched = True
            return
        cls._original_construct_mapping = _yaml.SafeLoader.construct_mapping

        def _construct_mapping_with_dup_check(loader, node, deep=False):
            loader.flatten_mapping(node)
            seen: set[str] = set()
            for key_node, _ in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in seen:
                    raise ValueError(
                        f"YAML duplicate key detected: {key!r}"
                    )
                seen.add(key)
            return cls._original_construct_mapping(loader, node, deep=deep)

        _yaml.SafeLoader.construct_mapping = _construct_mapping_with_dup_check
        cls._patched = True

    @classmethod
    def unpatch(cls) -> None:
        """恢复 SafeLoader.construct_mapping 为原始实现。"""
        if not cls._patched:
            return
        if _yaml is not None and cls._original_construct_mapping is not None:
            _yaml.SafeLoader.construct_mapping = cls._original_construct_mapping
            cls._original_construct_mapping = None
        cls._patched = False


_DuplicateKeyChecker.patch()  # 模块加载时即注入重复键检测。


def _strip_yaml_comment(line: str) -> str:  # 去掉一行 YAML 里的外部注释。
    """移除一行 YAML 文本里位于引号外部的注释尾巴。

    逐字符扫描，跟踪单引号和双引号的开关状态，只有引号外的 `#`
    才被视为注释起始符。YAML 单引号字符串内 ``''`` 表示字面单引号，
    扫描时跳过转义对，避免误判闭合。

    如果扫描结束后引号仍处于未闭合状态，说明该行 YAML 格式有误，
    此时保留整行内容不做截断，避免将引号内的注释标记误判为注释起始。

    Args:
        line (str): 一行原始 YAML 文本。

    Returns:
        str: 去掉引号外注释尾巴并去除右侧空白后的文本。
    """
    in_single = False  # 记录当前是否在单引号字符串里。
    in_double = False  # 记录当前是否在双引号字符串里。
    index = 0  # 手动游标，支持跳过 YAML 单引号转义对 ''。
    while index < len(line):  # 逐字符扫描，找真正的注释起点。
        char = line[index]
        if char == "'" and not in_double:  # 单引号只在不处于双引号时切换状态。
            if in_single and index + 1 < len(line) and line[index + 1] == "'":  # YAML 单引号转义：'' 表示字面单引号，跳过这两个字符。
                index += 2  # 跳过转义对，不翻转 in_single。
                continue
            in_single = not in_single  # 翻转单引号状态。
        elif char == '\\' and in_double and not in_single:  # YAML 双引号转义：\" 表示字面双引号，跳过反斜杠和下一个字符，避免误判闭合。
            if index + 1 < len(line):  # 确保下一个字符存在。
                index += 2  # 跳过转义对，不翻转 in_double。
            else:  # 末尾孤反斜杠在双引号内，保留反斜杠并跳过。
                index += 1
            continue
        elif char == '"' and not in_single:  # 双引号只在不处于单引号时切换状态。
            in_double = not in_double  # 翻转双引号状态。
        elif char == '#' and not in_single and not in_double:  # 引号外的 # 才是注释开始。
            return line[:index].rstrip()  # 直接截断注释尾巴并去掉右侧空白。
        index += 1  # 游标前进。
    # 未闭合引号时保留整行，避免截断引号内的内容。
    return line.rstrip()  # 如果没有外部注释，就只去掉右侧空白。


def _split_inline_items(payload: str) -> list[str]:  # 拆分 YAML 内联列表或映射中的顶层条目。
    """按顶层逗号拆分 YAML 内联列表或映射中的条目。

    跟踪引号状态和容器嵌套深度，只在顶层且不在引号内的逗号处拆分。
    不匹配的闭合括号（depth 下溢）会立即抛出 ValueError。

    Args:
        payload (str): 去掉外层方括号或花括号后的内联内容文本。

    Returns:
        list[str]: 拆分后的条目字符串列表，每项已去除两侧空白。

    Raises:
        ValueError: 当出现不匹配的闭合括号（depth 下溢）时抛出。
    """
    items: list[str] = []  # 收集拆分后的条目。
    current: list[str] = []  # 当前条目的字符缓冲区。
    depth = 0  # 内联容器的嵌套深度。
    in_single = False  # 当前是否处于单引号内部。
    in_double = False  # 当前是否处于双引号内部。
    index = 0  # 手动游标，支持跳过 YAML 单引号转义对 ''。
    while index < len(payload):  # 逐字符扫描 payload。
        char = payload[index]
        if char == "'" and not in_double:  # 单引号只有在不处于双引号时才切换。
            if in_single and index + 1 < len(payload) and payload[index + 1] == "'":  # YAML 单引号转义：'' 表示字面单引号，跳过这两个字符。
                current.append("''")  # 保留转义对。
                index += 2  # 跳过转义对，不翻转 in_single。
                continue
            in_single = not in_single  # 翻转单引号状态。
        elif char == '\\' and in_double and not in_single:  # YAML 双引号转义：\" 表示字面双引号，跳过反斜杠和下一个字符，避免误判闭合。
            current.append(char)  # 保留反斜杠。
            if index + 1 < len(payload):  # 确保下一个字符存在。
                current.append(payload[index + 1])  # 保留被转义的字符。
                index += 2  # 跳过转义对，不翻转 in_double。
            else:  # 末尾孤反斜杠在双引号内，保留反斜杠并跳过。
                index += 1
            continue
        elif char == '"' and not in_single:  # 双引号只有在不处于单引号时才切换。
            in_double = not in_double  # 翻转双引号状态。
        elif not in_single and not in_double:  # 只有不在任意引号内部时，结构字符才有意义。
            if char in '[{':  # 进入更深一层内联容器。
                depth += 1  # 深度加一。
            elif char in ']}':  # 退出一层内联容器。
                depth -= 1  # 深度减一。
                if depth < 0:  # 不匹配的闭合括号，立即报错。
                    raise ValueError(f"Unmatched closing bracket {char!r} in inline YAML")
            elif char == ',' and depth == 0:  # 顶层逗号意味着一个条目结束。
                items.append(''.join(current).strip())  # 先收进当前条目。
                current = []  # 再清空缓冲，准备下一项。
                index += 1  # 游标前进。
                continue  # 继续处理后续字符。
        current.append(char)  # 其他情况下当前字符都归入当前条目。
        index += 1  # 游标前进。
    tail = ''.join(current).strip()  # 循环结束后，把最后一段拼出来。
    if tail:  # 尾段非空时才追加。
        items.append(tail)  # 最后一项不能丢。
    return items  # 返回拆分好的条目列表。


_FLOAT_PATTERN = re.compile(
    r'^[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?$'
    r'|^[-+]?\d+[eE][-+]?\d+$'
    r'|^[-+]?\.(?:inf|Inf|INF)$'  # YAML 特殊浮点值：.inf / .Inf / .INF
    r'|^\.(?:nan|NaN|NAN)$'  # YAML 特殊浮点值：.nan / .NaN / .NAN
)  # 匹配浮点数、科学计数法和 YAML 特殊浮点值，不单独匹配纯整数。


def _parse_yaml_scalar(text: str) -> Any:  # 把简单 YAML 标量转成 Python 值。
    """把一段简单 YAML 标量文本转换成 Python 对象。

    支持的类型包括：None、布尔、内联列表、内联映射、引号字符串、
    整数（含八进制 0o 和十六进制 0x 前缀）、浮点数/科学计数法
    （含 YAML 特殊值 .inf/.nan），其余保持为字符串。

    回退解析器与 PyYAML SafeLoader 的已知行为差异：
    - 双引号字符串内的转义序列（如 ``\\n``）不做反转义，直接保留原样；
      PyYAML SafeLoader 会处理这些转义序列。
    - 八进制/十六进制整数仅支持 Python 字面量前缀（0o/0x），
      不支持 YAML 1.1 的 0 前缀八进制写法。
    - ``key: ""`` 在回退解析器中解析为空字符串（与 PyYAML 一致）。

    Args:
        text (str): YAML 标量文本。

    Returns:
        Any: 解析后的 Python 对象，类型取决于文本内容。
    """
    text = text.strip()  # 先去掉两边空白，避免格式化影响解析。
    if not text:  # 空文本通常表示空值。
        return None  # 用 None 代表空值。
    if text in {'null', 'Null', 'NULL', '~'}:  # YAML 显式空值写法。
        return None  # 统一转成 Python 的 None。
    if text in {'true', 'True', 'TRUE'}:  # YAML 布尔真。
        return True  # 转成 Python True。
    if text in {'false', 'False', 'FALSE'}:  # YAML 布尔假。
        return False  # 转成 Python False。
    if text.startswith('[') and text.endswith(']'):  # 内联列表要先去掉外壳。
        inner = text[1:-1].strip()  # 去掉方括号后读取内部内容。
        if not inner:  # 空内联列表。
            return []  # 直接返回空列表。
        return [_parse_yaml_scalar(item) for item in _split_inline_items(inner)]  # 逐项递归解析。
    if text.startswith('{') and text.endswith('}'):  # 内联映射也先去掉外壳。
        inner = text[1:-1].strip()  # 去掉花括号。
        if not inner:  # 空内联映射。
            return {}  # 直接返回空字典。
        mapping: dict[str, Any] = {}  # 用字典收集解析结果。
        for item in _split_inline_items(inner):  # 逐项拆分每个键值对。
            if ':' not in item:  # 内联映射条目必须包含冒号。
                raise ValueError(f"Inline mapping item missing colon: {item!r}")  # 明确报错。
            key, value = item.split(':', 1)  # 只切第一个冒号，避免值里的冒号被误拆。
            stripped_key = key.strip()
            if stripped_key in mapping:  # 内联映射同样不允许重复键。
                raise ValueError(f"YAML duplicate key detected: {stripped_key!r}")
            mapping[stripped_key] = _parse_yaml_scalar(value)  # 键和值都继续走标量解析。
        return mapping  # 返回内联映射结果。
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):  # 被引号包裹的字符串不再做数值化。
        inner = text[1:-1]  # 去掉外壳引号后返回纯字符串。
        return inner  # 空引号字符串 "" 返回空字符串，与 PyYAML 一致。
    if text.startswith('0o') and len(text) > 2:  # 八进制整数（0o 前缀）。
        try:
            return int(text, 8)  # 尝试按八进制解析。
        except ValueError:
            pass  # 解析失败则继续走后续分支。
    if text.startswith('0x') and len(text) > 2:  # 十六进制整数（0x 前缀）。
        try:
            return int(text, 16)  # 尝试按十六进制解析。
        except ValueError:
            pass  # 解析失败则继续走后续分支。
    if re.fullmatch(r'[-+]?\d+', text):  # 先尝试整数。
        return int(text)  # 匹配成功就转 int。
    if _FLOAT_PATTERN.fullmatch(text):  # 再尝试浮点数或科学计数法。
        lower = text.lower()
        if lower.endswith('.nan') or lower.endswith('.inf'):
            return float(text)  # .nan → nan, .inf → inf, -.inf → -inf
        return float(text)  # 匹配成功就转 float。
    return text  # 其他内容都保持为字符串，避免误伤复杂 YAML。


def _parse_yaml_line(line: str) -> tuple[str, str | None]:  # 把一行简单键值对拆成键和值。
    """把一行简单的 `key: value` 文本拆成键和值。

    按第一个冒号切分，键名去除两侧空白，值去除两侧空白。
    空值（冒号后无内容或仅空白）返回 None。
    空引号字符串（如 ``key: ""``）由调用方 _parse_yaml_scalar
    解析为空字符串而非 None，与 PyYAML 行为一致。

    Args:
        line (str): 一行 YAML 键值对文本。

    Returns:
        tuple[str, str | None]: (键名, 值文本)，值可能为 None。

    Raises:
        ValueError: 行中不含冒号或键名为空时抛出。
    """
    if ':' not in line:  # 没有冒号就不是我们支持的键值行。
        raise ValueError(f'Unsupported YAML line: {line!r}')  # 明确报出不支持的行。
    key, value = line.split(':', 1)  # 只按第一个冒号切分。
    key = key.strip()  # 键名去掉空白。
    if not key:  # 键名不能为空。
        raise ValueError(f'Unsupported YAML key in line: {line!r}')  # 空键名直接报错。
    value = value.strip()  # 值也要去掉两边空白。
    return key, value or None  # 空字符串统一转成 None。


def _parse_yaml_block(lines: list[tuple[int, str]], start_index: int, indent: int, _depth: int = 0):  # 递归解析一个缩进块。
    """递归解析缩进块，返回解析后的容器和消费到的行号。

    根据首行判断当前块是列表块还是映射块，然后逐行扫描当前缩进层，
    遇到子缩进时递归调用自身解析子块。

    Args:
        lines (list[tuple[int, str]]): 预处理后的行列表，每项为 (缩进深度, 去缩进内容)。
        start_index (int): 当前块在 lines 中的起始位置。
        indent (int): 当前块期望的缩进深度。
        _depth (int): 当前递归深度，用于防止无限递归。

    Returns:
        tuple[list | dict, int]: (解析结果容器, 下一个未消费的行号)。
            列表块返回 list，映射块返回 dict。

    Raises:
        ValueError: 当缩进不一致、遇到不支持的行结构或递归深度超限时抛出。
    """
    _MAX_DEPTH = 50  # 递归深度上限，防止恶意或损坏的 YAML 导致栈溢出。
    if _depth >= _MAX_DEPTH:
        raise ValueError(f"YAML nesting depth exceeds {_MAX_DEPTH}, possible malformed input")
    if start_index >= len(lines):  # 游标越界说明递归到底。
        return {}, start_index  # 空块返回空字典和当前位置。

    is_list = lines[start_index][1].startswith('- ')  # 通过第一行判断当前块是列表还是映射。
    container: list[Any] | dict[str, Any]  # 结果容器可能是 list，也可能是 dict。
    container = [] if is_list else {}  # 根据块类型选择容器。
    index = start_index  # 用 index 记录当前层消费到哪一行。

    while index < len(lines):  # 按行扫描当前缩进层。
        current_indent, content = lines[index]  # 读取当前行缩进和内容。
        if current_indent < indent:  # 缩进回退表示当前块结束。
            break  # 交回上一层处理。
        if current_indent != indent:  # 当前层里缩进不一致说明结构不合法。
            raise ValueError(f'Unexpected indentation in YAML near: {content!r}')  # 直接报错。

        if is_list:  # 列表块单独处理。
            if not content.startswith('- '):  # 列表块遇到非列表项说明当前块结束。
                break  # 交回上一层。
            item_text = content[2:].strip()  # 去掉 `- ` 后才是条目内容。
            if not item_text:  # 空条目后面通常跟一个子块。
                item_value, index = _parse_yaml_block(lines, index + 1, indent + 2, _depth + 1)  # 递归读子块。
                container.append(item_value)  # 子块结果直接作为列表项。
                continue  # 继续处理下一项。

            if ':' in item_text:  # 列表项本身也可能是一个小映射。
                item_mapping: dict[str, Any] = {}  # 用字典承载这个列表项内部字段。
                item_key, item_value = _parse_yaml_line(item_text)  # 先拆出这个列表项的首个键值。
                if item_value is None:  # 如果首个键没有值，后面缩进块就是它的值。
                    nested_value, index = _parse_yaml_block(lines, index + 1, indent + 4, _depth + 1)  # 递归读子块（首个键在 indent+2，其值在 indent+4）。
                    item_mapping[item_key] = nested_value  # 把子块结果挂到这个键上。
                else:  # 如果已经在这一行写完值，就直接解析。
                    item_mapping[item_key] = _parse_yaml_scalar(item_value)  # 把值转成 Python 标量。
                    index += 1  # 当前行已消费完，游标前进。

                while index < len(lines):  # 再继续扫描这个列表项的子字段。
                    next_indent, next_content = lines[index]  # 读取下一行。
                    if next_indent < indent + 2 or next_content.startswith('- '):  # 缩进回退或遇到下一个列表项都表示当前项结束。
                        break  # 退出当前列表项。
                    if next_indent != indent + 2:  # 子字段必须正好缩进一层。
                        raise ValueError(f'Unexpected indentation in YAML near: {next_content!r}')  # 缩进不对就报错。
                    next_key, next_value = _parse_yaml_line(next_content)  # 拆子字段键值。
                    if next_key in item_mapping:  # 列表项子字段重复键检测。
                        raise ValueError(f"YAML duplicate key detected: {next_key!r}")
                    if next_value is None:  # 子字段没有值时，后面还有更深一层。
                        nested_value, index = _parse_yaml_block(lines, index + 1, indent + 4, _depth + 1)  # 继续递归。
                        item_mapping[next_key] = nested_value  # 把更深层结果挂到这个键上。
                    else:  # 子字段已经写完。
                        item_mapping[next_key] = _parse_yaml_scalar(next_value)  # 直接解析标量。
                        index += 1  # 游标前进。
                container.append(item_mapping)  # 列表项最终以映射形式加入结果。
                continue  # 继续处理下一项。

            container.append(_parse_yaml_scalar(item_text))  # 普通列表项直接作为标量加入。
            index += 1  # 消费当前行后前进。
            continue  # 回到 while 顶部。

        key, value = _parse_yaml_line(content)  # 映射块里每一行都是键值对。
        if key in container:  # 重复键检测：回退解析器同样不允许重复键。
            raise ValueError(f"YAML duplicate key detected: {key!r}")
        if value is None:  # 如果键后面没有值，后续缩进块就是它的值。
            nested_value, index = _parse_yaml_block(lines, index + 1, indent + 2, _depth + 1)  # 递归读取子块。
            container[key] = nested_value  # 把子块结果挂到这个键上。
        else:  # 单行值直接解析。
            container[key] = _parse_yaml_scalar(value)  # 把标量值写入映射。
            index += 1  # 当前行已消费，游标前进。

    return container, index  # 返回当前层解析结果和消费到的位置。


def _safe_load_yaml(text: str) -> Any:  # 先尝试 PyYAML，不行就用回退解析器。
    """优先调用 PyYAML，失败时退回到本项目需要的最小解析器。

    回退解析器与 PyYAML SafeLoader 的已知行为差异：
    - 双引号转义序列不做反转义（如 ``\\n`` 保留原样）；
    - 不支持 YAML 块标量（``|`` / ``>``）；
    - 不支持 YAML 锚点和别名（``&`` / ``*``）；
    - 不支持 YAML 1.1 的 ``0`` 前缀八进制写法；
    - ``#`` 前不强制要求空白字符（YAML 规范要求空白，回退解析器更宽松）。
    """
    if _yaml is not None:  # 如果环境里有 PyYAML，就直接用成熟实现。
        return _yaml.safe_load(text)  # 交给 PyYAML 解析文本。

    raw_lines = []  # 回退解析器先收集清洗后的行。
    for raw_line in text.splitlines():  # 逐行扫描文本。
        stripped_line = _strip_yaml_comment(raw_line)  # 先去掉注释尾巴。
        if not stripped_line.strip():  # 空行或纯注释行直接跳过。
            continue  # 不进入解析。
        indent = len(stripped_line) - len(stripped_line.lstrip(' '))  # 计算原始缩进深度。
        raw_lines.append((indent, stripped_line.lstrip(' ')))  # 保存缩进和去左侧空白后的内容。

    if not raw_lines:  # 如果文件里什么有效内容都没有。
        return {}  # 统一视作空文档。
    payload, next_index = _parse_yaml_block(raw_lines, 0, raw_lines[0][0])  # 从第一行开始递归解析。
    if next_index != len(raw_lines):  # 如果还有没消费完的行。
        raise ValueError(f'YAML fallback parser did not consume the full document, {len(raw_lines) - next_index} line(s) remaining starting at: {raw_lines[next_index][1]!r}')  # 说明结构里有解析器不支持的内容。
    return payload  # 返回解析结果。


def find_project_root() -> Path:
    """向上逐级查找项目根目录，基于 marker 文件而非硬编码目录层级。

    从当前文件位置向上搜索，找到包含 pyproject.toml 或 .git 目录的
    最近祖先目录作为项目根。比 ``Path(__file__).resolve().parents[N]``
    更健壮——包结构变化时不需要手动调整 N。

    限制：当本包通过 pip install 安装到 site-packages 后，
    ``__file__`` 指向 site-packages 路径，可能找不到项目根目录
    （除非 site-packages 内存在 pyproject.toml 或 .git）。
    此函数设计用于开发模式运行，生产部署应使用配置文件显式指定路径。

    返回：
        Path: 项目根目录。

    异常：
        RuntimeError: 找不到 marker 文件时抛出。
    """
    _MARKERS = ("pyproject.toml", ".git")
    current = Path(__file__).resolve()
    for parent in current.parents:
        if any((parent / marker).exists() for marker in _MARKERS):
            return parent
    raise RuntimeError(
        f"Cannot find project root (looked for {_MARKERS}) "
        f"starting from {current}"
    )


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:  # 读取单个 YAML 文件并保证返回字典。
    """读取一个 YAML 配置文件，并保证返回值是字典。"""
    path = Path(config_path)  # 统一转成 Path，便于后续操作。
    if not path.exists():  # 文件不存在就直接报错。
        raise FileNotFoundError(f"Config file not found: {path}")  # 明确指出找不到哪个文件。
    raw_text = path.read_text(encoding="utf-8")  # 只读一次文件，避免双重 I/O 和 TOCTOU 竞态。
    has_effective_yaml_content = any(
        _strip_yaml_comment(line).strip()
        for line in raw_text.splitlines()
    )
    try:
        payload = _safe_load_yaml(raw_text)  # 直接传文本字符串，不再二次打开文件。
    except ValueError as exc:
        raise ValueError(f"{exc} (file: {path})") from exc
    if payload is None and not has_effective_yaml_content:  # 只有真正的空文档才允许回落成空配置。
        payload = {}
    if not isinstance(payload, dict):  # 顶层必须是映射。
        raise TypeError(f"Top-level YAML payload must be a mapping, got {type(payload).__name__}: {path}")  # 不是映射就报错。
    return payload  # 返回解析后的配置字典。


def load_dataset_config(config_path: str | Path, *, required_keys: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:  # 读取数据集配置并校验必需键。
    """读取数据集 YAML 配置，并校验必需键是否存在。

    根因修复：集中校验数据集配置的 schema，避免每个调用点各自做 .get() + None 检查。
    当配置文件缺少必需键时，立即抛出 ValueError 并指出具体缺了哪个键和文件路径，
    而不是让下游在运行时才遇到 KeyError 或静默使用空值。

    参数：
        config_path: 数据集配置文件路径。
        required_keys: 必需键名的列表或元组；为 None 时不做额外校验。

    返回：
        解析后的配置字典。

    异常：
        FileNotFoundError: 配置文件不存在。
        ValueError: 配置文件缺少必需键。
    """
    payload = load_yaml_config(config_path)  # 先用通用函数加载。
    if required_keys is not None:  # 有必需键列表时逐个校验。
        path = Path(config_path)  # 用于错误消息。
        missing_keys = [key for key in required_keys if key not in payload]  # 找出缺失的键。
        if missing_keys:  # 有缺失就立即报错，不让问题流到下游。
            raise ValueError(
                f"Dataset config {path.name} is missing required keys: {missing_keys}"
            )
    return payload  # 返回校验通过的配置字典。


def merge_configs(config_dicts: list[dict[str, Any]]) -> dict[str, Any]:  # 递归合并多个配置字典。
    """按顺序递归合并多个配置字典，后者覆盖前者。

    合并规则：
    1. 对于同名键，如果两边都是字典，则递归合并子字典；
    2. 否则，后者的值覆盖前者；
    3. 返回的字典与输入完全独立（深拷贝），修改返回值不会影响原对象。

    注意：使用 isinstance(x, dict) 而非 isinstance(x, Mapping) 检查字典类型，
    因为配置合并需要可变 dict，不接受 MappingProxyType 等不可变映射。
    传入非 dict 的 Mapping 会导致 TypeError。

    Args:
        config_dicts (list[dict[str, Any]]): 要合并的配置字典列表，按优先级从低到高排列。

    Returns:
        dict[str, Any]: 合并后的完整配置字典。

    Raises:
        TypeError: 当 config_dicts 中的元素不是字典时抛出。

    Example:
        >>> base = {"a": 1, "b": {"c": 2}}
        >>> override = {"b": {"d": 3}, "e": 4}
        >>> merge_configs([base, override])
        {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': 4}
    """
    def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        """递归合并两个字典，子字典继续深合并。

        Args:
            base (dict[str, Any]): 基础字典，作为合并起点。
            update (dict[str, Any]): 更新字典，同名键覆盖 base 中的值。

        Returns:
            dict[str, Any]: 合并后的新字典，与 base 和 update 完全独立。
        """
        merged: dict[str, Any] = {}
        for key, base_value in base.items():
            if key not in update:
                merged[key] = deepcopy(base_value)
        for key, update_value in update.items():
            if isinstance(update_value, dict) and isinstance(base.get(key), dict):
                merged[key] = _merge(base[key], update_value)
            else:
                merged[key] = deepcopy(update_value)
        return merged

    merged: dict[str, Any] = {}
    for config_dict in config_dicts:
        if not isinstance(config_dict, dict):
            raise TypeError(f"merge_configs expects a list of dict objects, got {type(config_dict).__name__}")
        merged = _merge(merged, config_dict)
    return merged


def collect_tbd_paths(config: dict[str, Any], *, prefix: str = "") -> list[str]:  # 递归找出所有 TBD 占位路径。
    """递归收集配置中所有值为 `TBD` 的路径名。

    扫描配置字典和列表的每一层，当发现值为字符串 "TBD" 时，把从根到该值
    的路径记录下来，方便在运行前发现未完成的配置项。

    注意：使用 isinstance(x, dict) 而非 isinstance(x, Mapping) 检查字典类型，
    因为配置来自 load_yaml_config，保证返回原生 dict。非 dict 的 Mapping
    类型（如 MappingProxyType）不会被递归扫描。

    Args:
        config (dict[str, Any]): 要扫描的配置字典。
        prefix (str): 当前路径前缀，用于递归拼接，默认为空字符串。

    Returns:
        list[str]: 所有值为 "TBD" 的路径名列表。
    """
    def _collect(value: Any, path: str) -> list[str]:
        if isinstance(value, dict):  # 字典节点按键继续递归。
            missing: list[str] = []
            for key, nested in value.items():
                current = f"{path}.{key}" if path else str(key)
                missing.extend(_collect(nested, current))
            return missing
        if isinstance(value, list):  # 列表节点也要继续递归，避免漏掉列表里的占位。
            missing: list[str] = []
            for index, nested in enumerate(value):
                current = f"{path}[{index}]" if path else f"[{index}]"
                missing.extend(_collect(nested, current))
            return missing
        return [path] if value == "TBD" else []

    return _collect(config, prefix)  # 返回所有占位路径列表。
