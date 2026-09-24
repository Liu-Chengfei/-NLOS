"""从冻结后的指标、统计和选例中生成总结 payload。

这个脚本把已经落盘的指标表、统计表和选例文件读进来，交给 summary 构建器，
再把结果写成 `summary.json`。它主要用于脚本级冒烟验证，检查 summary 消费链
能否继续接住冻结输入。
"""

from __future__ import annotations  # 允许使用前向类型注解。

import argparse  # 解析命令行参数。
import os  # 处理 `os.PathLike` 类型。
import sys  # 管理导入路径和进程参数。
from collections.abc import Iterable, Mapping, Sequence  # 判断映射和序列类型。
from pathlib import Path  # 处理文件路径。
from typing import Any  # 标注任意 JSON 风格结构。

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录。
SRC = ROOT / "src"  # 源码目录。
if str(SRC) not in sys.path:  # 确保脚本能直接导入项目代码。
    sys.path.insert(0, str(SRC))  # 把源码目录放到导入路径最前面。

from liquidloc.analysis.summary_builder import build_summary  # 真正生成 summary 的函数。
from liquidloc.common.io_utils import dumps_json_text, read_json, write_json  # 严格 JSON helper。


def _resolve_non_empty_path(raw_value: str | None, *, flag_name: str, default: Path | None = None) -> Path:
    """把命令行路径参数规范成可直接使用的绝对路径。"""
    if raw_value is None:  # 没给值时走默认路径分支。
        if default is None:  # 没默认值时不能继续猜。
            raise ValueError(f"{flag_name} must be a non-empty path")  # 直接报错。
        return default.resolve()  # 返回默认路径的绝对形式。
    value = str(raw_value).strip()  # 去掉前后空白。
    if not value:  # 空字符串也算无效输入。
        raise ValueError(f"{flag_name} must be a non-empty path")  # 明确报错。
    path = Path(value)  # 转成路径对象。
    if not path.is_absolute():  # 相对路径统一按仓库根目录解释。
        path = (ROOT / path).resolve()  # 补成绝对路径。
    return path.resolve()  # 再规整一次。


def _read_json(path: Path) -> Any:
    """读取 JSON 文件并返回反序列化结果。"""
    return read_json(path, encoding="utf-8-sig")  # 兼容带 BOM 的 UTF-8 文件。


def _default_output_path() -> Path:
    """返回 summary 脚本的默认输出路径。"""
    return ROOT / "outputs" / "script_smoke" / "summary.json"  # 冒烟输出统一落这里。


def _unwrap_required_metric_table_payload(payload: Any) -> Any:
    """从输入里拆出必须存在的主表载荷，只接受 `main_table` 聚合视图。"""
    if isinstance(payload, Mapping):  # 外层如果是对象，就看有没有包装字段。
        if "main_table" not in payload:  # 不再允许把 long-form metric_table 伪装成主表输入。
            raise ValueError("payload must contain 'main_table' aggregated rows")  # 明确告诉调用方当前脚本只接受主表聚合视图。
        current: Any = payload["main_table"]  # 取出真正的主表内容。
        if isinstance(current, Mapping):
            if "main_table" in current and current["main_table"]:
                current = current["main_table"]  # 优先继续拆 main_table 包装。
        return current  # 返回拆包后的结果。
    return payload  # 非映射输入就原样返回。


def _unwrap_optional_named_payload(payload: Any, key: str) -> Any:
    """从包装对象里拆出可选字段对应的值。"""
    current = payload  # 先拿原始值。
    if isinstance(current, Mapping) and key in current:  # 如果外层对象直接带这个字段。
        current = current[key]  # 先拆一层。
    if isinstance(current, Mapping) and key in current:  # 再防一层同名包装。
        current = current[key]  # 再拆一层。
    return current  # 返回拆完后的值。


def _normalize_json_value(value: Any) -> Any:
    """把路径和嵌套结构统一转成 JSON 友好的形式。"""
    if isinstance(value, os.PathLike):  # 路径对象转字符串。
        return str(Path(value))  # 路径对象转成字符串。
    if isinstance(value, Mapping):  # 映射递归处理每个键值对。
        return {str(key): _normalize_json_value(item) for key, item in value.items()}  # 映射递归转 JSON 友好结构。
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):  # 序列递归处理每个元素。
        return [_normalize_json_value(item) for item in value]  # 序列递归转 JSON 友好结构。
    return value  # 其他类型原样返回。


def _normalize_summary_payload(summary_payload: Mapping[str, Any]) -> dict[str, Any]:
    """把 summary payload 规范成可落盘的普通字典。"""
    normalized = dict(_normalize_json_value(dict(summary_payload)))  # 先整体转成 JSON 友好结构。
    case_refs = normalized.get("case_refs")  # 取出案例引用列表。
    if isinstance(case_refs, str):  # 如果被压成单个字符串，就包回列表。
        normalized["case_refs"] = [case_refs]  # 单个字符串补成一元素列表。
    elif isinstance(case_refs, Sequence) and not isinstance(case_refs, (bytes, bytearray)):
        normalized["case_refs"] = list(case_refs)
    elif isinstance(case_refs, Iterable):
        normalized["case_refs"] = list(case_refs)
    return normalized  # 返回规范化后的 summary。


def _count_case_refs(summary_payload: Mapping[str, Any]) -> int:
    """统计 summary 里案例引用的数量。"""
    case_refs = summary_payload.get("case_refs")  # 先取案例引用字段。
    if case_refs is None:  # 没有就算 0。
        return 0  # 没有案例引用就算 0。
    if isinstance(case_refs, str):  # 单个字符串算 1。
        return 1  # 单个字符串算 1。
    if isinstance(case_refs, Sequence) and not isinstance(case_refs, (bytes, bytearray)):  # 序列直接计数。
        return len(case_refs)  # 序列直接按长度计数。
    return 0  # 其他异常形态按 0 处理。


def _print_summary(payload: dict[str, Any]) -> int:
    """打印结构化摘要并返回退出码。"""
    print(dumps_json_text(payload, indent=None))  # 打印结构化 JSON，方便上游脚本消费。
    return int(payload["exit_code"])  # 返回约定好的退出码。


def _safe_write_summary(output_path: Path, normalized_summary: Mapping[str, Any]) -> tuple[bool, str | None, str | None]:
    """尽力写出 summary 文件，失败时返回错误类型和详情。"""
    try:
        write_json(output_path, {"summary": normalized_summary})  # 写出 summary 文件。
    except OSError as exc:
        return False, type(exc).__name__, str(exc)
    return True, None, None


def main(argv: list[str] | None = None) -> int:
    """脚本主入口，读取冻结输入并生成 summary。"""
    print("[14_summary] 开始 | 从冻结实验产物构建汇总", flush=True)
    parser = argparse.ArgumentParser(description="Build a summary from frozen experiment artifacts")  # 创建参数解析器。
    parser.add_argument("--metric-table", required=True)  # 指标表路径。
    parser.add_argument("--statistics-table", required=True)  # 统计表路径。
    parser.add_argument("--selected-cases", required=True)  # 选例文件路径。
    parser.add_argument("--output-path", default=None)  # 输出路径，可选。
    args = parser.parse_args(argv)  # 解析命令行参数。

    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "14_build_summary")

    try:
        print("[14_summary] 解析路径并读取输入", flush=True)
        metric_table_path = _resolve_non_empty_path(args.metric_table, flag_name="--metric-table")  # 规范指标表路径。
        statistics_table_path = _resolve_non_empty_path(args.statistics_table, flag_name="--statistics-table")  # 规范统计表路径。
        selected_cases_path = _resolve_non_empty_path(args.selected_cases, flag_name="--selected-cases")  # 规范选例路径。
        output_path = _resolve_non_empty_path(  # 规范输出路径。
            args.output_path,  # 命令行显式传入的输出路径。
            flag_name="--output-path",  # 输出参数名，用于报错提示。
            default=_default_output_path(),  # 没传时使用的默认路径。
        )  # 输出路径解析结束。

        metric_table = _unwrap_required_metric_table_payload(_read_json(metric_table_path))  # 读并拆指标表。
        statistics_table = _unwrap_optional_named_payload(_read_json(statistics_table_path), "statistics_table")  # 读并拆统计表。
        selected_cases = _unwrap_optional_named_payload(_read_json(selected_cases_path), "selected_cases")  # 读并拆选例。
        print_dict({"metric_table_path": str(metric_table_path), "statistics_table_path": str(statistics_table_path), "selected_cases_path": str(selected_cases_path), "output_path": str(output_path)}, "路径配置")

        print("[14_summary] 构建汇总", flush=True)
        summary_payload = build_summary(metric_table, statistics_table, selected_cases)  # 真正构建 summary。
        if not isinstance(summary_payload, Mapping):  # 返回值必须是对象。
            raise ValueError("summary payload must be a mapping")  # 不对就直接报错。

        normalized_summary = _normalize_summary_payload(summary_payload)  # 把 summary 变成可落盘结构。
        output_path.parent.mkdir(parents=True, exist_ok=True)  # 先确保输出目录存在。
    except Exception as exc:
        _rc = _print_summary(
            {
                "status": "failed",
                "stage": "build_summary",
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "exit_code": 1,
            }
        )
        print(f"[14_summary] 完成 | 返回码={_rc}", flush=True)
        return _rc

    print("[14_summary] 写入汇总", flush=True)
    write_ok, error_type, error_detail = _safe_write_summary(output_path, normalized_summary)  # 尽力写出 summary。
    if not write_ok:
        _rc = _print_summary(
            {
                "status": "failed",
                "stage": "build_summary",
                "output_path": str(output_path.resolve()),
                "error_type": error_type,
                "detail": error_detail,
                "exit_code": 1,
            }
        )
        print(f"[14_summary] 完成 | 返回码={_rc}", flush=True)
        return _rc
    _rc = _print_summary(
        {
            "status": "ok",
            "stage": "build_summary",
            "output_path": str(output_path.resolve()),
            "case_count": _count_case_refs(normalized_summary),
            "exit_code": 0,
        }
    )
    print(f"[14_summary] 完成 | 返回码={_rc}", flush=True)
    return _rc


if __name__ == "__main__":  # 直接执行脚本时才走这里。
    raise SystemExit(main(sys.argv[1:]))  # 用 main 的返回码退出进程。
