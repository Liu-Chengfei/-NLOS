"""运行指标表的显著性检验脚本。

这个脚本把已经落盘的指标表读进来，按给定分组键和指标名跑显著性检验，
再把结果写成 `statistics_table.json`。它主要用于脚本级冒烟验证，检查统计
消费方和指标表结构是否仍然兼容。
"""

from __future__ import annotations  # 允许使用前向类型注解。

import argparse  # 解析命令行参数。
import json  # 读取 section9_pulse_async_audit.json 落盘内容.
import sys  # 操作 `sys.path` 和进程参数。
from collections.abc import Mapping  # 判断映射类型。
from pathlib import Path  # 处理文件路径。
from typing import Any  # 标注任意 JSON 风格结构。

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录。
SRC = ROOT / "src"  # 源码目录。
if str(SRC) not in sys.path:  # 确保脚本能直接导入项目代码。
    sys.path.insert(0, str(SRC))  # 把源码目录放到导入路径最前面。

from liquidloc.common.io_utils import dumps_json_text, read_json, write_json  # 严格 JSON 读写。
from liquidloc.analysis.significance_tests import run_significance_tests  # 真正执行显著性检验的函数。


def _resolve_non_empty_path(raw_value: str | None, *, flag_name: str, default: Path | None = None) -> Path:
    """把命令行传入的路径整理成可直接使用的绝对路径。

    如果命令行没有传值，就回退到默认路径；如果传了空字符串，就直接报错，
    这样可以避免把“看起来像有值、实际上是空”的参数继续传下去。
    """
    if raw_value is None:  # 没给值时走默认路径分支。
        if default is None:  # 没默认值时不能继续猜。
            raise ValueError(f"{flag_name} must be a non-empty path")  # 明确告诉调用方哪里缺值。
        return default.resolve()  # 直接返回默认路径的绝对形式。
    value = str(raw_value).strip()  # 先去掉前后空白，避免空格伪装成有效路径。
    if not value:  # 空字符串也算无效输入。
        raise ValueError(f"{flag_name} must be a non-empty path")  # 直接报错，防止后面读错位置。
    path = Path(value)  # 把字符串转成路径对象，方便后续判断。
    if not path.is_absolute():  # 相对路径统一按仓库根目录解释。
        path = (ROOT / path).resolve()  # 补成绝对路径，避免工作目录影响结果。
    return path.resolve()  # 再规整一次，保证输出稳定。


def _read_json(path: Path) -> Any:
    """读取 JSON 文件并返回反序列化结果。"""
    return read_json(path, encoding="utf-8-sig")  # 支持带 BOM 的 UTF-8 文件。


def _unwrap_metric_table_payload(payload: Any) -> list[dict[str, Any]]:
    """把可能被包装过的 metric_table 统一拆成列表。

    有些上游会直接给列表，有些会上包一层 `metric_table` 字段；这里都接受，
    但最终必须收敛成“列表里每项都是对象”的结构。
    """
    current = payload  # 先把原始输入拿出来，后面可能会逐层拆包。
    if isinstance(current, Mapping) and "metric_table" in current:  # 如果外层有包装字段。
        current = current["metric_table"]  # 先拆掉第一层包装。
    if isinstance(current, Mapping) and "metric_table" in current:  # 再防一层重复包装。
        current = current["metric_table"]  # 继续拆到真正的列表。
    if not isinstance(current, list):  # 最终必须是列表。
        raise TypeError("metric table payload must resolve to a list")  # 类型不对就报错。

    rows: list[dict[str, Any]] = []  # 准备标准化后的行列表。
    for index, row in enumerate(current):  # 逐行检查，避免混入非对象项。
        if not isinstance(row, Mapping):  # 每一行都必须是映射。
            raise TypeError(f"metric_table row {index} must be an object")  # 明确指出哪一行出错。
        rows.append(dict(row))  # 转成普通字典，避免后续消费方依赖特殊映射类型。
    return rows  # 返回标准化后的行列表。


def _default_output_path() -> Path:
    """返回统计脚本的默认输出路径。"""
    return ROOT / "outputs" / "script_smoke" / "statistics_table.json"  # 冒烟输出统一落这里。


def _print_summary(payload: dict[str, Any]) -> int:
    """打印结构化摘要并返回退出码。"""
    print(dumps_json_text(payload, indent=None))  # 统一输出结构化 JSON，方便上游脚本消费。
    return int(payload["exit_code"])  # 返回约定好的退出码。


def _safe_write_statistics_table(output_path: Path, payload: Mapping[str, Any]) -> tuple[bool, str | None, str | None]:
    """尽力写出统计表文件，失败时返回错误类型和详情。"""
    try:
        write_json(output_path, payload)  # 将结构化统计结果落盘成 JSON。
    except OSError as exc:
        return False, type(exc).__name__, str(exc)
    return True, None, None


def main(argv: list[str] | None = None) -> int:
    """脚本主入口，读取指标表、执行统计检验并写出结果。"""
    print("[12_statistics] 开始 | 对指标表运行显著性检验", flush=True)
    parser = argparse.ArgumentParser(description="Run significance tests on a metric table")  # 创建命令行解析器。
    parser.add_argument("--metric-table", required=True)  # 输入指标表路径。
    parser.add_argument("--group-keys", nargs="+", required=True)  # 分组键列表。
    parser.add_argument("--metric-names", nargs="+", required=True)  # 要检验的指标名列表。
    parser.add_argument("--output-path", default=None)  # 输出路径，可不传。
    parser.add_argument("--audit-dir", default=None, help="审计目录路径，含 section9_pulse_async_audit.json")  # §9.3 pulse/async 量级门违规审计目录，12 读取该文件将违规摘要写入 statistics_table.json.
    args = parser.parse_args(argv)  # 解析命令行参数。

    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "12_run_statistics")

    try:
        print("[12_statistics] 解析路径并读取指标表", flush=True)
        metric_table_path = _resolve_non_empty_path(args.metric_table, flag_name="--metric-table")  # 规范输入路径。
        output_path = _resolve_non_empty_path(  # 规范输出路径。
            args.output_path,  # 命令行显式传入的输出路径。
            flag_name="--output-path",  # 输出参数名，用于报错。
            default=_default_output_path(),  # 没传时使用的默认输出路径。
        )  # 输出路径解析结束。

        metric_table = _unwrap_metric_table_payload(_read_json(metric_table_path))  # 先读入并标准化指标表。
        print_dict({"metric_table_path": str(metric_table_path), "output_path": str(output_path), "group_keys": args.group_keys, "metric_names": args.metric_names, "metric_table_rows": len(metric_table) if isinstance(metric_table, list) else "N/A"}, "路径与统计配置")
        print("[12_statistics] 运行显著性检验", flush=True)
        statistics_table = run_significance_tests(  # 把指标表交给真正的统计检验实现。
            metric_table,  # 标准化后的指标表。
            group_keys=list(args.group_keys),  # 分组键列表。
            metric_names=list(args.metric_names),  # 需要检验的指标名列表。
        )  # 统计检验结束。

        payload = {"statistics_table": statistics_table}  # 把统计结果包成最终落盘结构。

        # §9.3 N_seed 不达标不产生结论警示 (前提指导 §9.3 「单一种子定全序不构成本排序成立」).
        # 此处用 metric_table 中独立 seed 数量作为代理估计, 供调用方/审计明确看到不达标警示.
        try:
            from liquidloc.protocol.experiment_gates import check_seed_count
            seed_field_candidates = (
                args.group_keys if args.group_keys else []
            ) + ['seed_id', 'repeat_id', 'scene_id', 'seq_id', 'case_ref']
            observed_seeds: set[str] = set()
            for _row in metric_table:
                if not isinstance(_row, Mapping):
                    continue
                for _field in seed_field_candidates:
                    _v = _row.get(_field)
                    if _v is None or _v == '':
                        continue
                    observed_seeds.add(str(_v))
                    break
            if observed_seeds:
                seed_report = check_seed_count(len(observed_seeds))
                payload["section9_n_seed_check"] = {
                    "n_seed_observed": int(seed_report['n_seed']),
                    "n_seed_min": int(seed_report['n_seed_min']),
                    "n_seed_recommended": int(seed_report['n_seed_recommended']),
                    "violated": bool(seed_report['violated']),
                    "violated_recommended": bool(seed_report['violated_recommended']),
                    "single_seed_no_conclusion_allowed": bool(seed_report['single_seed_no_conclusion_allowed']),
                    "message": str(seed_report['message']),
                }
                if seed_report['violated'] or seed_report['violated_recommended']:
                    import warnings
                    warnings.warn(
                        f"§9.3 {seed_report['message']}",
                        stacklevel=2,
                    )
        except Exception as _exc:  # pragma: no cover - 协议层 import 失败时降级, 不阻断主路径
            payload["section9_n_seed_check"] = {"error": f"{type(_exc).__name__}: {_exc}"}

        # §9.3 pulse/async 量级门跨 bundle 违规审计 (2026-07-23 §9 穷举审视 Round 4 真修复):
        # eval_pipeline 在 audits_dir/section9_pulse_async_audit.json 落盘每 bundle 的 §9 pulse/async
        # 违规聚合; 12_run_statistics 读取该文件, 将违规摘要写入 statistics_table.json 的
        # payload["section9_pulse_async_audit"] 中, 供 15_build_six_cmp_aggregate 等下游脚本
        # 与人工审计直接消费, 避免 §9 pulse_async 违规在统计链路中静默丢弃.
        audit_dir = getattr(args, 'audit_dir', None)
        if audit_dir:
            audit_dir_path = _resolve_non_empty_path(audit_dir, flag_name="--audit-dir")
            section9_audit_file = audit_dir_path / "section9_pulse_async_audit.json"
            if section9_audit_file.is_file():
                try:
                    payload["section9_pulse_async_audit"] = json.loads(
                        section9_audit_file.read_text(encoding="utf-8")
                    )
                except Exception as _exc:  # pragma: no cover - 文件损坏时降级
                    payload["section9_pulse_async_audit"] = {
                        "error": f"{type(_exc).__name__}: {_exc}"
                    }
            else:
                payload["section9_pulse_async_audit"] = {"note": "section9_pulse_async_audit.json not found in audit dir"}

        output_path.parent.mkdir(parents=True, exist_ok=True)  # 先确保输出目录存在。
    except Exception as exc:
        _rc = _print_summary(
            {
                "status": "failed",
                "stage": "run_statistics",
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "exit_code": 1,
            }
        )
        print(f"[12_statistics] 完成 | 返回码={_rc}", flush=True)
        return _rc

    print("[12_statistics] 写入统计表", flush=True)
    write_ok, error_type, error_detail = _safe_write_statistics_table(output_path, payload)  # 尽力写出统计表 JSON。
    if not write_ok:
        _rc = _print_summary(
            {
                "status": "failed",
                "stage": "run_statistics",
                "output_path": str(output_path.resolve()),
                "error_type": error_type,
                "detail": error_detail,
                "exit_code": 1,
            }
        )
        print(f"[12_statistics] 完成 | 返回码={_rc}", flush=True)
        return _rc
    _rc = _print_summary(
        {
            "status": "ok",
            "stage": "run_statistics",
            "output_path": str(output_path.resolve()),
            "row_count": len(statistics_table),
            "exit_code": 0,
        }
    )
    print(f"[12_statistics] 完成 | 返回码={_rc}", flush=True)
    return _rc


if __name__ == "__main__":  # 只有直接执行脚本时才走这里。
    raise SystemExit(main(sys.argv[1:]))  # 用 main 的返回码退出进程。
