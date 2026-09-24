"""Run the minimal contract smoke pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from liquidloc.common.io_utils import dumps_json_text
from liquidloc.pipelines.contract_smoke_pipeline import ContractSmokePipeline


def _resolve_non_empty_path(
    raw_value: str | None,
    *,
    flag_name: str,
    default: Path | None = None,
) -> Path:
    """Resolve a CLI path against the repo root and reject blank values."""
    if raw_value is None:
        if default is None:
            raise ValueError(f"{flag_name} must be a non-empty path")
        return default.resolve()
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"{flag_name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path.resolve()


def _print_summary(payload: dict[str, object]) -> int:
    """Emit a structured JSON summary and return the exit code."""
    print(dumps_json_text(payload, indent=None))
    return int(payload["exit_code"])


def main(argv: list[str] | None = None) -> int:
    """Build the smoke config and run the contract smoke pipeline."""
    parser = argparse.ArgumentParser(description="Run a minimal scaffold smoke pipeline.")
    parser.add_argument("--output-root", default=None, help="Optional custom output directory")
    parser.add_argument("--project-root", default=None, help="Optional project root override")
    args = parser.parse_args(argv)

    # 诊断信息走 stderr，stdout 仅输出结构化 JSON summary，避免污染子进程 JSON 解析合同
    # print_args / print_dict 仅输出到 stdout，此处改用 sys.stderr 直接输出诊断行
    print(f"[99_smoke] ===== 99_mini_smoke 参数清单 =====", file=sys.stderr, flush=True)
    print(f"[99_smoke] output_root = {args.output_root!r}", file=sys.stderr, flush=True)
    print(f"[99_smoke] project_root = {args.project_root!r}", file=sys.stderr, flush=True)
    print(f"[99_smoke] ===== 参数清单结束 =====", file=sys.stderr, flush=True)

    print("[99_smoke] 开始 | output_root=" + str(args.output_root), file=sys.stderr, flush=True)

    try:
        project_root = _resolve_non_empty_path(
            args.project_root,
            flag_name="--project-root",
            default=ROOT,
        )
        cfg: dict[str, str] = {"project_root": str(project_root)}
        if args.output_root is not None:
            cfg["output_root"] = str(
                _resolve_non_empty_path(
                    args.output_root,
                    flag_name="--output-root",
                )
            )
        print(f"[99_smoke] cfg = {cfg!r}", file=sys.stderr, flush=True)
        print("[99_smoke] 正在运行契约冒烟流水线", file=sys.stderr, flush=True)
        result = ContractSmokePipeline().run(cfg)
    except Exception as exc:
        print("[99_smoke] 完成 | 返回码=1 (exception)", file=sys.stderr, flush=True)
        return _print_summary(
            {
                "status": "failed",
                "stage": "mini_smoke",
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "exit_code": 1,
            }
        )

    print("[99_smoke] 完成 | 返回码=0", file=sys.stderr, flush=True)
    return _print_summary(
        {
            "status": "ok",
            "stage": "mini_smoke",
            "pipeline_stage": result.stage_name,
            "artifact_count": len(result.artifacts),
            "artifacts": result.artifacts,
            "metadata": result.metadata,
            "exit_code": 0,
        }
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
