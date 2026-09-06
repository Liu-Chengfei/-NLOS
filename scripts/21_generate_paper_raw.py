"""Paper 数据集 (paper_main) 物化入口。

按 configs/datasets/paper_dataset.yaml 的 paper_spec 生成 5 seed × 400 序列
= 2000 条原始仿真数据，输出到 --output-root/seed{N}/<seq_id>/{imu,uwb,vio,gt}.json
+ sim_meta.json + anchor_layout.json。

本脚本不修改 sim_e9 任何数据；与 sim_materializer 的 sim 路径完全隔离。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 让脚本可以直接 python scripts/21_generate_paper_raw.py 运行
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from liquidloc.dataio.paper_materializer import (  # noqa: E402
    load_paper_spec,
    materialize_paper_raw,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper_main raw dataset")
    parser.add_argument(
        "--paper-config",
        type=str,
        default="configs/datasets/paper_dataset.yaml",
        help="paper dataset config yaml",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/raw/paper_main_v2",
        help="output root (each seed as seed{N}/ subdir)",
    )
    parser.add_argument(
        "--n-seed",
        type=int,
        default=None,
        help="override paper_spec.scale.n_seed (for quick test)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paper_cfg = load_paper_spec(args.paper_config)
    n_seed = args.n_seed or paper_cfg["paper_spec"]["scale"]["n_seed"]
    print(f"[21_generate_paper_raw] paper_config={args.paper_config}")
    print(f"[21_generate_paper_raw] output_root={args.output_root}")
    print(f"[21_generate_paper_raw] n_seed={n_seed}")
    t0 = time.time()
    report = materialize_paper_raw(
        paper_config_path=args.paper_config,
        output_root=args.output_root,
        n_seed=n_seed,
    )
    elapsed = time.time() - t0
    total_seqs = sum(s["seq_count"] for s in report["seeds"])
    print(f"[21_generate_paper_raw] DONE: {len(report['seeds'])} seeds, {total_seqs} sequences in {elapsed:.1f}s")
    for s in report["seeds"]:
        print(f"  seed{s['seed_id']}: {s['seq_count']} sequences")
    print(f"[21_generate_paper_raw] report: {args.output_root}/materialize_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
