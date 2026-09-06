"""Materialize deterministic simulation raw data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from liquidloc.common.io_utils import dumps_json_text
from liquidloc.dataio.sim_materializer import (
    SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS,
    materialize_sim_raw,
)


def _resolve_path(raw_value: str | None, *, flag_name: str, default: Path) -> Path:
    """Normalize one path argument into an absolute path."""
    if raw_value is None:
        return default.resolve()
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"{flag_name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path.resolve()


def main(argv: list[str] | None = None) -> int:
    """Materialize simulation raw data and emit a structured report."""
    print("[02_sim_raw] 开始 | 生成仿真原始序列", flush=True)
    parser = argparse.ArgumentParser(description="Materialize deterministic sim raw sequences")
    parser.add_argument("--output-root", default=str(ROOT / "data" / "raw" / "sim_e9_main"))
    parser.add_argument("--fixture-root", default=str(ROOT / "tests" / "fixtures" / "datasets" / "miluv"))
    parser.add_argument(
        "--sequence-profile",
        choices=("sim_e9_only_compact",),
        default="sim_e9_only_compact",
    )
    args = parser.parse_args(argv)
    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "02_generate_sim_raw")

    output_root = _resolve_path(args.output_root, flag_name="--output-root", default=ROOT / "data" / "raw" / "sim")
    fixture_root = _resolve_path(
        args.fixture_root,
        flag_name="--fixture-root",
        default=ROOT / "tests" / "fixtures" / "datasets" / "miluv",
    )
    sequence_specs = SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS
    print_dict(
        {
            "output_root": str(output_root),
            "fixture_root": str(fixture_root),
            "sequence_profile": args.sequence_profile,
            "sequence_specs_count": len(sequence_specs),
        },
        "仿真序列配置",
    )
    print_dict({"sequence_specs": sequence_specs}, "序列规格详情")

    print("[02_sim_raw] 生成序列", flush=True)
    try:
        report = materialize_sim_raw(output_root, fixture_root=fixture_root, sequence_specs=sequence_specs)
    except Exception as exc:
        report = {
            "status": "failed",
            "output_root": str(output_root),
            "fixture_root": str(fixture_root),
            "sequence_profile": args.sequence_profile,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(dumps_json_text(report))
        print("[02_sim_raw] 完成 | 返回码=1", flush=True)
        return 1

    print(dumps_json_text(report))
    print("[02_sim_raw] 完成 | 返回码=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
