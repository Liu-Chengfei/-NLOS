"""Paper 数据集最小化推理脚本：直接 CorePipeline.run 跑 3 个神经网络。

不依赖 09._run_supplementary_public_route（其 symlink/raw_root 解析在 paper 上有 anchor_id
解析问题）；直接调 CorePipeline.run，传入：
  - events_by_seq_id: 从 outputs/prepare_paper_main_v1_quick 读
  - ground_truth_by_seq_id: 从 data/raw/paper_main_v1_quick/seed0 读
  - source_report_by_seq_id: 从 prepare_manifest 读 anchor_layout
  - scene_tasks: 简单构造，每条 seq × 每个 method
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import gzip
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(os.getcwd())  # forward slash

# 防 escape：所有 path 走 forward slash
def _safe_path(rel: str) -> Path:
    """所有 path 走 forward slash 拼接，再 to WindowsPath。避免 Win32 把 \Q \4 等当 escape。"""
    base = PROJECT_ROOT.as_posix().rstrip("/")
    posix = f"{base}/{rel.lstrip('/')}"
    return Path(posix)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-root", type=str, default="outputs/prepare_paper_main_v1")
    parser.add_argument("--raw-root", type=str, default="data/raw/paper_main_v1_quick")
    parser.add_argument("--train-root", type=str, default="outputs/train_paper_main_v1_quick")
    parser.add_argument("--output-root", type=str, default="outputs/e20_paper_main_v1_quick")
    parser.add_argument("--methods", type=str, nargs="+", default=["lstm_ekf", "liquid_ekf", "transformer_ekf"])
    parser.add_argument("--max-sequences", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    # 1. 加载 events
    prepare_root = _safe_path(args.events_root)
    raw_root = _safe_path(args.raw_root)
    output_root = _safe_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1.1 改 yaml checkpoint_path
    import subprocess
    fix_cmd = [sys.executable, "scripts/25_fix_yaml.py", "--train-root", args.train_root, "--seed", "0"]
    print(f"[24_min] running: {' '.join(fix_cmd)}")
    subprocess.call(fix_cmd, cwd=str(PROJECT_ROOT))

    try:
        # 2. 收集 seq_ids
        seq_ids = sorted([p.stem.replace("_events.pkl", "").replace("_events", "") for p in prepare_root.glob("paper_*_events.pkl.gz")])
        seq_ids = seq_ids[: args.max_sequences]
        print(f"[24_min] {len(seq_ids)} seq_ids from {prepare_root}")

        # 3. 加载 prepared events
        events_by_seq = {}
        for sid in seq_ids:
            with gzip.open(prepare_root / f"{sid}_events.pkl.gz", "rb") as f:
                events_by_seq[sid] = pickle.load(f)
        print(f"[24_min] loaded events total: {sum(len(v) for v in events_by_seq.values())}")

        # 4. 加载 ground_truth
        gt_by_seq = {}
        for sid in seq_ids:
            with open(raw_root / "seed0" / sid / "gt.json", encoding="utf-8") as f:
                gt_by_seq[sid] = json.load(f)
        print(f"[24_min] loaded GT: {len(gt_by_seq)}")

        # 5. 加载 anchor_layout（直接读 paper 的 2D 锚点）
        anchor_by_seq = {}
        for sid in seq_ids:
            with open(raw_root / "seed0" / sid / "anchor_layout.json", encoding="utf-8") as f:
                anchor_by_seq[sid] = json.load(f)
        print(f"[24_min] loaded anchor_layout: {len(anchor_by_seq)} (first keys: {list(anchor_by_seq.values())[0]['anchor_ids']})")

        # 6. 构造 scene_tasks（每条 seq × 每个 method）
        scene_tasks = []
        for sid in seq_ids:
            ax_meta_path = raw_root / "seed0" / sid / "sim_meta.json"
            ax_meta = json.loads(ax_meta_path.read_text(encoding="utf-8")) if ax_meta_path.exists() else {}
            axes = ax_meta.get("axes_override", {})
            for method in args.methods:
                scene_tasks.append({
                    "task_id": f"paper_{sid}_{method}",  # 单下划线分隔，避免 :: 在 Windows 非法
                    "seq_id": sid,
                    "scene_id": f"S({axes.get('A','A2')},{axes.get('N','N2')},{axes.get('V','V0')},K0,M0)",  # 五轴协议：G 已并入 K
                    "axes": axes,
                    "method": method,
                    "dataset_name": "paper_main",
                    "anchor_layout": anchor_by_seq[sid],
                })
        print(f"[24_min] {len(scene_tasks)} scene_tasks")

        # 7. 跑 CorePipeline
        from liquidloc.pipelines.core_pipeline import CorePipeline
        cp = CorePipeline()
        core_output = output_root / "core"
        core_output.mkdir(parents=True, exist_ok=True)
        core_result = cp.run({
            "scene_tasks": scene_tasks,
            "events_by_seq_id": events_by_seq,
            "ground_truth_by_seq_id": gt_by_seq,
            "source_report_by_seq_id": {},
            "methods": args.methods,
            "output_root": str(core_output),
        })
        # CorePipeline 把 bundle 写到 core_output/predictions/<task_id>__<method>.json
        # 不一定返回 in-memory list，直接从盘上读回。
        pred_dir = core_output / "predictions"
        bundles = []
        if pred_dir.exists():
            for f in pred_dir.glob("*.json"):
                with open(f, encoding="utf-8") as fp:
                    bundles.append(json.load(fp))
        print(f"[24_min] loaded {len(bundles)} prediction bundles from {pred_dir}")

        # 8. 计算指标（bundle 本身就是 prediction_bundle，含 diagnostics）
        from liquidloc.metrics.metric_runner import compute_metrics
        from liquidloc.common.config_utils import load_yaml_config
        protocol_cfg = load_yaml_config(PROJECT_ROOT / "configs" / "base" / "experiment_protocol.yaml")
        all_rows = []
        for bundle in bundles:
            sid = bundle.get("seq_id")
            method = bundle.get("method_name", "unknown")
            gt_seq = gt_by_seq.get(sid, [])
            if not gt_seq:
                continue
            try:
                metrics = compute_metrics(bundle, gt_seq, 1.0, return_support=False, protocol_cfg=protocol_cfg)
                all_rows.append({
                    "case_ref": f"paper::{sid}::{method}",
                    "seq_id": sid, "method_name": method, **metrics
                })
                print(f"[24_min] {sid}/{method}: rmse={metrics.get('rmse', 'NA'):.3f}, p95={metrics.get('p95', 'NA'):.3f}, fail={metrics.get('failure_rate', 'NA'):.3f}")
            except Exception as e:
                print(f"[24_min] metrics failed for {sid}/{method}: {e}")
        if all_rows:
            metrics_path = output_root / "metric_table.csv"
            metric_keys = sorted({k for r in all_rows for k in r if k not in ("case_ref", "seq_id", "method_name")})
            with open(metrics_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["case_ref", "seq_id", "method_name"] + metric_keys)
                w.writeheader()
                for r in all_rows:
                    w.writerow(r)
            print(f"[24_min] wrote {metrics_path} ({len(all_rows)} rows)")
            agg = defaultdict(list)
            for r in all_rows:
                for k in ("rmse", "p95", "failure_rate"):
                    if k in r:
                        agg[(r["method_name"], k)].append(float(r[k]))
            print(f"\n[24_min] === aggregate by method (over {len(seq_ids)} seq_ids) ===")
            print(f"{'method':<20} {'rmse_mean':>10} {'p95_mean':>10} {'fail_mean':>10}")
            for method in args.methods:
                if (method, "rmse") in agg:
                    rmse = sum(agg[(method, "rmse")]) / len(agg[(method, "rmse")])
                    p95 = sum(agg[(method, "p95")]) / len(agg[(method, "p95")]) if agg[(method, "p95")] else 0
                    fail = sum(agg[(method, "failure_rate")]) / len(agg[(method, "failure_rate")]) if agg[(method, "failure_rate")] else 0
                    print(f"{method:<20} {rmse:>10.3f} {p95:>10.3f} {fail:>10.3f}")
        return 0
    finally:
        # 还原 yaml
        subprocess.call([sys.executable, "scripts/25_fix_yaml.py", "--restore"], cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    sys.exit(main())
