"""终端输出双向记录工具（Tee）。

职责：
    包装 sys.stdout / sys.stderr，让所有 print 输出同时写入终端和日志文件，
    实现"所有在终端输出都记录在一个文件里"的需求。

上游依赖：
    - Python 标准库 sys / time / io / pathlib

下游调用者：
    - scripts/20_run_paper_experiments.py  — 实验入口处安装一次 tee
    - 其他需要全量终端日志的脚本入口

核心变量：
    - _ACTIVE_LOG_PATH: 当前已安装的日志文件路径，避免重复安装
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import IO, Any, Optional

__all__ = ("install_tee", "get_active_log_path", "uninstall_tee", "print_args", "print_dict")

_ACTIVE_LOG_PATH: Optional[Path] = None
_ORIGINAL_STDOUT: Optional[Any] = None
_ORIGINAL_STDERR: Optional[Any] = None


class _TeeStream:
    """把写入操作同时转发到终端流和文件流。"""

    def __init__(self, terminal_stream: Any, file_stream: IO[str], *, stream_name: str) -> None:
        self._terminal = terminal_stream  # 原始终端流（sys.stdout 或 sys.stderr）
        self._file = file_stream  # 日志文件流
        self._stream_name = stream_name  # 流名称，用于诊断

    # 标准文件对象接口转发
    def write(self, data: str) -> int:
        if not isinstance(data, str):  # 防御性类型保护
            data = str(data)
        written = 0
        try:
            written = self._terminal.write(data)  # 先写终端，保证即时可见
            self._terminal.flush()  # 强制刷新终端缓冲
        except Exception:
            pass  # 终端写入失败不影响文件记录
        try:
            self._file.write(data)  # 再写文件
            self._file.flush()  # 强制刷新文件缓冲
        except Exception:
            pass  # 文件写入失败不影响终端显示
        return written

    def flush(self) -> None:
        try:
            self._terminal.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        # 转发 isatty，部分库会检查是否连接到终端
        try:
            return self._terminal.isatty()
        except Exception:
            return False

    def fileno(self) -> int:
        # 转发 fileno，部分库会要求文件描述符
        return self._terminal.fileno()

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        try:
            return self._terminal.encoding or "utf-8"
        except Exception:
            return "utf-8"

    @property
    def errors(self) -> str:
        try:
            return self._terminal.errors or "strict"
        except Exception:
            return "strict"

    @property
    def mode(self) -> str:
        return "w"

    @property
    def name(self) -> str:
        return f"<tee:{self._stream_name}>"


def install_tee(log_dir: str | Path, *, run_name: str = "run") -> Path:
    """安装 tee，把 sys.stdout / sys.stderr 同时写入终端和日志文件。

    Args:
        log_dir: 日志文件存放目录，不存在会自动创建。
        run_name: 运行名称，用于日志文件命名（默认 "run"）。

    Returns:
        日志文件的绝对路径。

    Raises:
        ValueError: 当 log_dir 为空字符串或 run_name 为空时。
    """
    global _ACTIVE_LOG_PATH, _ORIGINAL_STDOUT, _ORIGINAL_STDERR

    if not str(log_dir).strip():
        raise ValueError(f"log_dir must be a non-empty path, got {log_dir!r}")
    if not str(run_name).strip():
        raise ValueError(f"run_name must be a non-empty string, got {run_name!r}")

    # 如果已经安装过，先卸载再重装，避免重复包装
    if _ACTIVE_LOG_PATH is not None:
        uninstall_tee()

    log_dir_path = Path(log_dir).resolve()
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # 用时间戳命名日志文件，避免覆盖
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = log_dir_path / f"{run_name}_{timestamp}.log"

    # 保存原始流
    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr

    # 以 utf-8 编码打开文件，避免中文乱码
    file_stream = open(log_file, "w", encoding="utf-8", buffering=1, newline="\n")

    # 写入日志头
    header = (
        f"{'=' * 80}\n"
        f"# 实验运行日志\n"
        f"# 创建时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
        f"# 日志文件: {log_file}\n"
        f"# 运行名称: {run_name}\n"
        f"{'=' * 80}\n"
    )
    file_stream.write(header)
    file_stream.flush()

    # 安装 tee
    sys.stdout = _TeeStream(_ORIGINAL_STDOUT, file_stream, stream_name="stdout")
    sys.stderr = _TeeStream(_ORIGINAL_STDERR, file_stream, stream_name="stderr")

    _ACTIVE_LOG_PATH = log_file

    # Keep stdout clean for entrypoints that emit machine-readable JSON.
    print(f"[tee] 日志记录已启动 | 文件: {log_file}", file=sys.stderr, flush=True)

    return log_file


def uninstall_tee() -> None:
    """卸载 tee，恢复原始 sys.stdout / sys.stderr。"""
    global _ACTIVE_LOG_PATH, _ORIGINAL_STDOUT, _ORIGINAL_STDERR

    # 先拿到当前 tee 的文件流引用，便于关闭
    file_stream_to_close: Optional[IO[str]] = None
    if isinstance(sys.stdout, _TeeStream):
        file_stream_to_close = sys.stdout._file

    # 恢复原始流
    if _ORIGINAL_STDOUT is not None:
        sys.stdout = _ORIGINAL_STDOUT
    if _ORIGINAL_STDERR is not None:
        sys.stderr = _ORIGINAL_STDERR

    # 关闭文件流并写入结尾标记
    if file_stream_to_close is not None:
        try:
            file_stream_to_close.write(
                f"\n{'=' * 80}\n# 日志结束: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n{'=' * 80}\n"
            )
            file_stream_to_close.flush()
            file_stream_to_close.close()
        except Exception:
            pass

    _ACTIVE_LOG_PATH = None
    _ORIGINAL_STDOUT = None
    _ORIGINAL_STDERR = None


def get_active_log_path() -> Optional[Path]:
    """返回当前活动的日志文件路径，未安装则返回 None。"""
    return _ACTIVE_LOG_PATH


def print_args(args: Any, script_name: str = "", *, prefix: str = "[参数]") -> None:
    """打印 argparse.Namespace 或对象的全部属性值，便于核对实验配置。

    Args:
        args: argparse.Namespace 或任意对象，会遍历其 __dict__。
        script_name: 脚本名称，用于输出标题（可选）。
        prefix: 每行输出前缀，默认 "[参数]"。
    """
    try:
        if script_name:
            print(f"{prefix} ===== {script_name} 参数清单 =====", flush=True)
        else:
            print(f"{prefix} ===== 参数清单 =====", flush=True)
        if args is None:
            print(f"{prefix} args = None", flush=True)
            return
        # argparse.Namespace 用 vars()；其他对象用 __dict__
        try:
            items = vars(args).items()
        except TypeError:
            items = getattr(args, "__dict__", {}).items()
        sorted_items = sorted(items, key=lambda kv: str(kv[0]))
        for key, value in sorted_items:
            try:
                repr_value = repr(value)
            except Exception as repr_err:
                repr_value = f"<repr 失败: {repr_err}>"
            print(f"{prefix} {key} = {repr_value}", flush=True)
        print(f"{prefix} ===== 参数清单结束 =====", flush=True)
    except Exception as print_err:
        try:
            print(f"{prefix} print_args 失败: {print_err}", flush=True)
        except Exception:
            pass


def print_dict(data: Any, title: str = "", *, prefix: str = "[配置]") -> None:
    """递归打印字典（或类似映射）的内容，最多 2 层深度。

    Args:
        data: 字典或映射对象。
        title: 标题（可选）。
        prefix: 每行输出前缀，默认 "[配置]"。
    """
    import os as _os
    if _os.environ.get("LIQUIDLOC_SUPPRESS_PRINT_DICT", "") == "1":
        return
    try:
        if title:
            print(f"{prefix} ===== {title} =====", flush=True)
        if data is None:
            print(f"{prefix} (None)", flush=True)
            return
        if not hasattr(data, "items"):
            print(f"{prefix} {data!r}", flush=True)
            return
        try:
            sorted_items = sorted(data.items(), key=lambda kv: str(kv[0]))
        except Exception:
            sorted_items = list(data.items())
        for key, value in sorted_items:
            if isinstance(value, dict):
                try:
                    inner_sorted = sorted(value.items(), key=lambda kv: str(kv[0]))
                except Exception:
                    inner_sorted = list(value.items())
                print(f"{prefix} {key}:", flush=True)
                for inner_key, inner_value in inner_sorted:
                    try:
                        repr_value = repr(inner_value)
                    except Exception:
                        repr_value = "<repr 失败>"
                    print(f"{prefix}   {inner_key} = {repr_value}", flush=True)
            else:
                try:
                    repr_value = repr(value)
                except Exception:
                    repr_value = "<repr 失败>"
                print(f"{prefix} {key} = {repr_value}", flush=True)
        if title:
            print(f"{prefix} ===== {title} 结束 =====", flush=True)
    except Exception as print_err:
        try:
            print(f"{prefix} print_dict 失败: {print_err}", flush=True)
        except Exception:
            pass
