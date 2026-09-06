"""审计输出目录是否满足输出合同。

这个脚本会读取一个输出根目录，然后交给输出合同校验器做完整性检查，
最后把审计结果写成 JSON。它主要是给脚本级产物做收口检查，确保目录结构
和关键文件没有漂移。
"""

from __future__ import annotations  # 允许使用前向类型注解。

import argparse  # 解析命令行参数。
import sys  # 处理导入路径和退出码。
from pathlib import Path  # 处理文件路径。

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录。
SRC = ROOT / "src"  # 源码目录。
if str(SRC) not in sys.path:  # 确保本脚本能直接导入项目包。
    sys.path.insert(0, str(SRC))  # 把源码目录放到最前面。

from liquidloc.common.io_utils import dumps_json_text, write_json
from liquidloc.protocol.output_contract_schema import check_output_contract as validate_output_contract  # 输出合同校验函数。


def _resolve_non_empty_path(raw_value: str | None, flag_name: str, default: Path) -> Path:
    """把路径参数整理成可用的绝对路径。"""
    if raw_value is None:  # 没传就直接用默认值。
        return default.resolve()  # 返回默认路径的绝对形式。
    value = str(raw_value).strip()  # 去掉空白，避免空字符串伪装成路径。
    if not value:  # 空字符串不合法。
        raise ValueError(f"{flag_name} must be a non-empty path")  # 直接报错。
    return Path(value).resolve()  # 返回规范后的绝对路径。


def _resolve_path_against(raw_value: str | None, *, flag_name: str, base_root: Path, default: Path | None = None) -> Path:
    """把路径参数按指定基准目录解释成绝对路径。"""
    if raw_value is None:  # 没传就回退到默认值。
        if default is None:  # 没默认值时不能继续猜。
            raise ValueError(f"{flag_name} must be a non-empty path")  # 直接报错。
        return default.resolve()  # 返回默认路径。
    value = str(raw_value).strip()  # 去掉空白。
    if not value:  # 空字符串无效。
        raise ValueError(f"{flag_name} must be a non-empty path")  # 直接报错。
    path = Path(value)  # 转成路径对象。
    if not path.is_absolute():  # 相对路径按基准目录解释。
        path = (base_root / path).resolve()  # 补成绝对路径。
    return path.resolve()  # 返回最终路径。


def _print_summary(payload: dict[str, object]) -> int:
    """打印结构化摘要并返回退出码。"""
    print(dumps_json_text(payload, indent=None))
    return int(payload["exit_code"])


def _safe_write_report(report_path: Path, payload: dict[str, object]) -> tuple[bool, str | None, str | None]:
    """尽力写出审计报告，失败时返回错误类型与详情。"""
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)  # 先确保报告目录存在。
        write_json(report_path, payload)  # 写出审计报告文件。
    except OSError as exc:
        return False, type(exc).__name__, str(exc)
    return True, None, None


def main(argv: list[str] | None = None) -> int:
    """脚本主入口，检查输出目录是否满足合同并写出审计报告。"""
    print("[15_audit] 开始 | 按输出合同审计产物目录", flush=True)
    parser = argparse.ArgumentParser(description="Audit an outputs tree against the output contract")  # 创建命令行解析器。
    parser.add_argument("--project-root", default=None)  # 项目根目录，可选。
    parser.add_argument("--output-root", default=None)  # 要审计的输出目录，可选。
    parser.add_argument("--report-path", default=None)  # 审计报告输出路径，可选。
    parser.add_argument(
        "--layout",
        default="default",
        choices=("default", "public_benchmark"),
        help="输出合同布局名；public_benchmark 适用于 PublicBenchmarkPipeline 嵌套布局（E7 公开 benchmark 验证）。",
    )  # 合同布局选择，可选。
    args = parser.parse_args(argv)  # 解析命令行参数。

    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "15_audit_outputs")

    print("[15_audit] 解析路径", flush=True)
    project_root = _resolve_non_empty_path(args.project_root, "--project-root", ROOT)  # 确定项目根目录。
    if args.output_root is None:  # 未显式传输出目录时使用默认位置。
        output_root = (project_root / "outputs" / "mini_smoke").resolve()  # 默认审计 mini_smoke 输出。
    else:  # 显式传了输出目录时就按项目根目录解释。
        output_root = _resolve_path_against(
            args.output_root,  # 命令行给出的输出目录。
            flag_name="--output-root",  # 参数名，用于报错提示。
            base_root=project_root,  # 相对路径的解释基准。
        )  # 规范输出目录。

    if args.report_path is None:  # 未显式传报告路径时使用默认位置。
        report_path = (output_root / "audits" / "output_contract_audit.json").resolve()  # 默认审计报告路径。
    else:  # 显式传了报告路径时按项目根目录解释。
        report_path = _resolve_path_against(
            args.report_path,  # 命令行给出的报告路径。
            flag_name="--report-path",  # 参数名，用于报错提示。
            base_root=project_root,  # 相对路径的解释基准。
        )  # 规范报告路径。

    print("[15_audit] 校验输出合同", flush=True)
    contract_report = validate_output_contract(output_root, layout=args.layout)  # 执行真正的输出合同校验，layout 决定合同模板。
    status = "ok" if contract_report["is_complete"] else "failed"  # 根据完整性决定状态。
    print_dict({"project_root": str(project_root), "output_root": str(output_root), "report_path": str(report_path), "status": status}, "路径与审计状态")
    if isinstance(contract_report, dict):
        print_dict(contract_report, "合同审计报告 (contract_report)")
    payload = {  # 组装要落盘的审计结果。
        "status": status,  # 审计状态。
        "output_root": str(output_root.resolve()),  # 被审计的输出目录。
        "report_path": str(report_path.resolve()),  # 报告文件路径。
        "contract_report": contract_report,  # 合同校验器返回的详细结果。
        "exit_code": 0 if contract_report["is_complete"] else 1,  # 默认退出码跟完整性保持一致。
    }

    print("[15_audit] 写入审计报告", flush=True)
    write_ok, error_type, error_detail = _safe_write_report(report_path, payload)  # 尽力把完整报告落盘。
    if not write_ok:  # 报告目录或文件不可写时，仍然返回结构化失败结果。
        payload["status"] = "failed"
        payload["exit_code"] = 1
        payload["report_write_error_type"] = error_type
        payload["report_write_error"] = error_detail
    _rc = _print_summary(payload)  # 命令行直接打印摘要，方便快速检查。
    print(f"[15_audit] 完成 | 返回码={_rc}", flush=True)
    return _rc


if __name__ == "__main__":  # 直接运行脚本时才进入这里。
    raise SystemExit(main(sys.argv[1:]))  # 用 main 的退出码结束进程。
