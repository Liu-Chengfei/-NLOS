"""严格 JSON IO 工具函数。

职责：
    提供统一的 JSON 读写接口，使用严格标准 JSON 语义（拒绝 NaN/Infinity），
    确保生成的 artifacts 符合 JSON 规范。

上游依赖：
    - Python 标准库 json / pathlib / typing

下游调用者：
    - liquidloc.common.prepared_inputs — 读取准备阶段产物
    - liquidloc.pipelines.*            — 流水线层读写配置和结果
    - liquidloc.analysis.*             — 分析层读写指标数据

核心变量：
    - 无模块级变量，全部通过函数实现

编码规范：
    所有文件操作统一使用 UTF-8 编码，确保跨平台兼容性和中文支持。
    路径统一使用 pathlib.Path 处理，避免字符串拼接带来的平台差异。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

__all__ = (
    "loads_json_text",
    "dumps_json_text",
    "read_json",
    "write_json",
    "read_json_records",
    "read_optional_json_object",
    "read_optional_json_records",
)


def _reject_nonfinite_json_constant(token: str) -> NoReturn:
    """拒绝 NaN/Infinity 伪常量，确保生成严格标准 JSON。

    此函数作为 ``json.loads`` / ``json.load`` 的 ``parse_constant``
    回调使用。Python 的 json 解析器在遇到 NaN/Infinity/-Infinity
    时会调用此函数，而非静默接受。

    已知限制：Python json 解析器在解析超大数字（如 ``1e999``）时
    会先将其转换为 float('inf')，此时 ``parse_constant`` 回调
    不会被触发——因为解析器认为这是合法数字而非常量。这意味着
    ``1e999`` 会绕过此检查，导致 inf 值渗入结果。当前项目数据
    不包含此类极端数字，因此不构成实际风险。

    Args:
        token (str): JSON 解析过程中遇到的非有限常量。

    Raises:
        ValueError: 始终抛出，拒绝非有限值。
    """
    raise ValueError(f"non-finite JSON constant is not allowed: {token}")


def loads_json_text(text: str) -> Any:
    """使用严格标准 JSON 语义解析 JSON 文本。

    拒绝 NaN、Infinity、-Infinity 等非标准 JSON 常量。

    Args:
        text (str): 要解析的 JSON 文本。

    Returns:
        Any: 解析后的 Python 对象。

    Raises:
        json.JSONDecodeError: 当文本不是有效 JSON 时抛出。
        ValueError: 当遇到非有限常量时抛出。
    """
    return json.loads(text, parse_constant=_reject_nonfinite_json_constant)


def dumps_json_text(obj: Any, *, indent: int | None = 2, ensure_ascii: bool = False, sort_keys: bool = False) -> str:
    """序列化为严格标准 JSON 文本。

    拒绝 NaN、Infinity、-Infinity，确保输出符合 JSON 规范。

    注意：调用方需确保传入的对象仅包含 JSON 可序列化类型
   （dict、list、str、int、float、bool、None）。numpy 标量、
    torch tensor、MappingProxyType 等非标准类型需在调用前
    转换为原生 Python 类型，否则将抛出 TypeError。

    Args:
        obj: 要序列化的 Python 对象。
        indent (int | None): 缩进空格数，None 表示紧凑输出，默认为 2。
        ensure_ascii (bool): 是否确保所有非 ASCII 字符转义，默认为 False（保留原样）。
        sort_keys (bool): 是否按键名字典序输出，默认为 False。启用后可提高
            JSON 输出的可复现性和 diff 稳定性。

    Returns:
        str: 序列化后的 JSON 文本。

    Raises:
        TypeError: 当对象包含不可序列化类型时抛出。
        ValueError: 当对象包含 NaN 或 Infinity 时抛出。
    """
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, allow_nan=False, sort_keys=sort_keys)


def read_json(path: str | Path, *, encoding: str = "utf-8") -> Any:
    """从磁盘读取 JSON 文件，使用严格标准-JSON 语义。

    Args:
        path (str | Path): JSON 文件路径，支持字符串或 Path 对象。
        encoding (str): 文件编码，默认为 "utf-8"。

    Returns:
        Any: 解析后的 Python 对象。

    Raises:
        ValueError: 当 path 为 None 时抛出。
        FileNotFoundError: 当文件不存在时抛出。
        json.JSONDecodeError: 当文件内容不是有效 JSON 时抛出。
        ValueError: 当遇到非有限常量时抛出。
    """
    if path is None:
        raise ValueError("read_json: path must not be None")
    path = Path(path).resolve()
    with path.open("r", encoding=encoding) as file_obj:
        loaded_obj = json.load(file_obj, parse_constant=_reject_nonfinite_json_constant)
    return loaded_obj


def write_json(
    path: str | Path,
    obj: Any,
    *,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: int | None = 2,
    sort_keys: bool = False,
) -> None:
    """将对象写入 JSON 文件，使用严格标准-JSON 语义。

    自动创建父目录，确保写入成功。

    注意：写入为非原子操作——如果写入过程中发生异常（如磁盘满），
    可能留下部分写入的损坏文件。对于关键数据，调用方应先写入临时
    文件再 rename。

    注意：调用方需确保传入的对象仅包含 JSON 可序列化类型
   （dict、list、str、int、float、bool、None）。numpy 标量、
    torch tensor 等非标准类型需在调用前转换为原生 Python 类型。

    Args:
        path (str | Path): 输出文件路径，支持字符串或 Path 对象。
        obj: 要序列化的 Python 对象。
        encoding (str): 文件编码，默认为 "utf-8"。
        ensure_ascii (bool): 是否确保所有非 ASCII 字符转义，默认为 False。
        indent (int | None): 缩进空格数，None 表示紧凑输出，默认为 2。
        sort_keys (bool): 是否按键名字典序输出，默认为 False。启用后可提高
            JSON 输出的可复现性和 diff 稳定性。

    Raises:
        ValueError: 当 path 为 None 时抛出。
        TypeError: 当对象包含不可序列化类型时抛出。
        ValueError: 当对象包含 NaN 或 Infinity 时抛出。
        OSError: 当无法写入文件时抛出。
    """
    if path is None:
        raise ValueError("write_json: path must not be None")
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding) as file_obj:
        json.dump(obj, file_obj, ensure_ascii=ensure_ascii, indent=indent, allow_nan=False, sort_keys=sort_keys)


def read_json_records(path: Path) -> list[dict[str, Any]]:
    """读取必须存在的 JSON 记录文件，并要求内容是非空字典列表。"""
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path.name}")
    payload = read_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a list[dict] payload in {path}, got {type(payload).__name__}")
    bad_indices = [i for i, row in enumerate(payload) if not isinstance(row, dict)]
    if bad_indices:
        raise TypeError(
            f"Rows at indices {bad_indices[:5]} are not dict in {path}"
        )
    if not payload:
        raise ValueError(f"Expected non-empty list[dict] payload in {path.name}, got empty list")
    return payload


def read_optional_json_object(path: Path) -> dict[str, Any] | None:
    """读取可选 JSON 对象文件；不存在或内容无效就返回 None。

    仅捕获 ValueError（含 json.JSONDecodeError）；OSError/PermissionError 需上浮，
    因为权限问题不应被静默忽略。
    """
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
