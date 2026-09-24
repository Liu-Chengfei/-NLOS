"""Paper 2000-seq 推理（无 subprocess 依赖）：直接 CorePipeline.run。

策略：
  1. 手动把 configs/models/{lstm,liquid,transformer}_ekf.yaml 的 checkpoint_path
     指向 outputs/train_paper_main/seed0/<model>/checkpoints/<best.pt>
  2. 直接调 CorePipeline.run（跳过 25_fix_yaml subprocess）
  3. 写完 metric_table.csv
  4. 恢复 yaml checkpoint_path 为空（finally）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
import gzip
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(os.getcwd())  # forward slash


def _safe_path(rel: str) -> Path:
    posix = (PROJECT_ROOT.as_posix().rstrip("/") + "/" + rel.lstrip("/")).replace("\\", "/")
    return Path(posix)


CHECKPOINT_FILE = {
    "lstm_ekf": "lstm_ekf_best_checkpoint.pt",
    "liquid_ekf": "liquid_ekf_resume_checkpoint.pt",
    "transformer_ekf": "transformer_ekf_best_checkpoint.pt",
}
MODEL_YAML = {
    "lstm_ekf": "configs/models/lstm_ekf.yaml",
    "liquid_ekf": "configs/models/liquid_ekf.yaml",
    "transformer_ekf": "configs/models/transformer_ekf.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-root", type=str, default="outputs/prepare_paper_main_v1")
    parser.add_argument("--raw-root", type=str, default="data/raw/paper_main_v1")
    parser.add_argument("--train-root", type=str, default="outputs/train_paper_main")
    parser.add_argument("--output-root", type=str, default="outputs/e20_paper_main_v1_full")
    parser.add_argument("--methods", type=str, nargs="+", default=["lstm_ekf", "liquid_ekf", "transformer_ekf"])
    parser.add_argument("--max-sequences", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _set_yaml_checkpoint(yaml_path: Path, ckpt_path: Path) -> None:
    """直接修改 yaml 写 checkpoint_path 字段。"""
    import yaml as _yaml
    cfg = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cfg["checkpoint_path"] = str(ckpt_path).replace("\\", "/")
    yaml_path.write_text(
        _yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _clear_yaml_checkpoint(yaml_path: Path) -> None:
    import yaml as _yaml
    cfg = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cfg["checkpoint_path"] = ""
    yaml_path.write_text(
        _yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    prepare_root = _safe_path(args.events_root)
    raw_root = _safe_path(args.raw_root)
    train_root = _safe_path(args.train_root)
    output_root = _safe_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. 改 yaml checkpoint_path 直接指向 full 训练产物
    yaml_backups = {}
    for method, yaml_rel in MODEL_YAML.items():
        if method not in args.methods:
            continue
        yaml_path = PROJECT_ROOT / yaml_rel
        yaml_backups[yaml_path] = yaml_path.read_text(encoding="utf-8")
        ckpt_name = CHECKPOINT_FILE[method]
        # 优先 seed0 best/，fallback epoch_
        ckpt_path = train_root / f"seed{args.seed}" / method / "checkpoints" / ckpt_name
        if not ckpt_path.exists():
            # fallback: latest epoch_*
            epoch_ckpts = sorted((train_root / f"seed{args.seed}" / method / "checkpoints").glob(f"{method}_epoch_*.pt"))
            if epoch_ckpts:
                ckpt_path = epoch_ckpts[-1]
            else:
                print(f"[24_direct] WARN: no checkpoint for {method} (train still running)")
                continue
        _set_yaml_checkpoint(yaml_path, ckpt_path)
        print(f"[24_direct] set {yaml_rel} -> {ckpt_path.name}")

    try:
        # 2. 收集 seq_ids
        seq_ids = sorted([p.stem.replace("_events.pkl", "").replace("_events", "") for p in prepare_root.glob("paper_*_events.pkl.gz")])
        seq_ids = seq_ids[: args.max_sequences]
        print(f"[24_direct] {len(seq_ids)} seq_ids from {prepare_root}")

        # 3. 加载 prepared events (批量)
        t0 = time.time()
        events_by_seq = {}
        for i, sid in enumerate(seq_ids):
            with gzip.open(prepare_root / f"{sid}_events.pkl.gz", "rb") as f:
                events_by_seq[sid] = pickle.load(f)
            if i % 200 == 0:
                print(f"  loaded {i+1}/{len(seq_ids)} events in {time.time()-t0:.1f}s")
        print(f"[24_direct] loaded events total: {sum(len(v) for v in events_by_seq.values())} in {time.time()-t0:.1f}s")

        # 4. 加载 ground_truth
        gt_by_seq = {}
        for sid in seq_ids:
            with open(raw_root / f"seed{args.seed}" / sid / "gt.json", encoding="utf-8") as f:
                gt_by_seq[sid] = json.load(f)
        print(f"[24_direct] loaded GT: {len(gt_by_seq)}")

        # 5. 加载 anchor_layout
        anchor_by_seq = {}
        for sid in seq_ids:
            with open(raw_root / f"seed{args.seed}" / sid / "anchor_layout.json", encoding="utf-8") as f:
                anchor_by_seq[sid] = json.load(f)
        print(f"[24_direct] loaded anchor_layout: {len(anchor_by_seq)}")

        # 6. 构造 scene_tasks
        scene_tasks = []
        for sid in seq_ids:
            ax_meta_path = raw_root / f"seed{args.seed}" / sid / "sim_meta.json"
            ax_meta = json.loads(ax_meta_path.read_text(encoding="utf-8")) if ax_meta_path.exists() else {}
            axes = ax_meta.get("axes_override", {})
            for method in args.methods:
                scene_tasks.append({
                    "task_id": f"paper_{sid}_{method}",
                    "seq_id": sid,
                    "scene_id": f"S({axes.get('A','A2')},{axes.get('N','N2')},{axes.get('V','V0')},K0,M0)",  # 五轴协议：G 已并入 K
                    "axes": axes,
                    "method": method,
                    "dataset_name": "paper_main",
                    "anchor_layout": anchor_by_seq[sid],
                })
        print(f"[24_direct] {len(scene_tasks)} scene_tasks")

        # 7. 跑 CorePipeline
        from liquidloc.pipelines.core_pipeline import CorePipeline
        cp = CorePipeline()
        core_output = output_root / "core"
        core_output.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        core_result = cp.run({
            "scene_tasks": scene_tasks,
            "events_by_seq_id": events_by_seq,
            "ground_truth_by_seq_id": gt_by_seq,
            "source_report_by_seq_id": {},
            "methods": args.methods,
            "output_root": str(core_output),
        })
        print(f"[24_direct] CorePipeline run done in {time.time()-t0:.1f}s")

        # 8. 从盘读回 predictions
        pred_dir = core_output / "predictions"
        bundles = []
        if pred_dir.exists():
            for f in pred_dir.glob("*.json"):
                with open(f, encoding="utf-8") as fp:
                    bundles.append(json.load(fp))
        print(f"[24_direct] loaded {len(bundles)} prediction bundles from {pred_dir}")

        # 9. 计算指标（每个 (seq_id, method) 只取一次）
        from liquidloc.metrics.metric_runner import compute_metrics
        from liquidloc.common.config_utils import load_yaml_config
        protocol_cfg = load_yaml_config(PROJECT_ROOT / "configs" / "base" / "experiment_protocol.yaml")

        results = []
        seen = set()
        for bundle in bundles:
            sid = bundle.get("seq_id")
            method = bundle.get("method_name", "unknown")
            key = (sid, method)
            if key in seen:
                continue
            seen.add(key)
            gt_seq = gt_by_seq.get(sid, [])
            if not gt_seq:
                continue
            try:
                m = compute_metrics(bundle, gt_seq, 1.0, return_support=False, protocol_cfg=protocol_cfg)
                results.append({"seq_id": sid, "method": method, **m})
            except Exception as e:
                print(f"[24_direct] metrics failed for {sid}/{method}: {e}")

        # 10. 写 csv
        if results:
            metrics_path = output_root / "metric_table.csv"
            keys = sorted({k for r in results for k in r if k not in ("seq_id", "method")})
            with open(metrics_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["seq_id", "method"] + keys)
                w.writeheader()
                for r in results:
                    w.writerow(r)
            print(f"[24_direct] wrote {metrics_path} ({len(results)} rows)")

            # 11. 汇总
            agg = defaultdict(list)
            for r in results:
                for k in ("rmse", "p95", "failure_rate"):
                    if k in r:
                        agg[(r["method"], k)].append(float(r[k]))
            print(f"\n[24_direct] === aggregate by method (over {len(seq_ids)} seq_ids) ===")
            print(f"{'method':<20} {'count':>8} {'rmse_mean':>10} {'p95_mean':>10} {'fail_mean':>10}")
            for method in args.methods:
                if (method, "rmse") in agg:
                    n = len(agg[(method, "rmse")])
                    rmse = sum(agg[(method, "rmse")]) / n
                    p95 = sum(agg[(method, "p95")]) / n
                    fail = sum(agg[(method, "failure_rate")]) / n
                    print(f"{method:<20} {n:>8d} {rmse:>10.3f} {p95:>10.3f} {fail:>10.3f}")
        return 0
    finally:
        # 恢复 yaml
        for yaml_path, content in yaml_backups.items():
            yaml_path.write_text(content, encoding="utf-8")
            print(f"[24_direct] restore {yaml_path.name}")


if __name__ == "__main__":
    sys.exit(main())
