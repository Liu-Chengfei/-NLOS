"""Paper 数据集推理入口：3 神经网络 (lstm_ekf / liquid_ekf / transformer_ekf)。

直接调 CorePipeline.run，对每个 (paper_main, seq_id, method) 三元组：
  1. 加载 prepared events + ground_truth
  2. create_model(name, cfg) with cfg['checkpoint_path'] = paper-trained .pt
  3. run_fusion() 得到 prediction bundle
  4. compute_metrics() 得到 RMSE / P95 / failure_rate
  5. 写入 metric_table.csv

本脚本依赖 scripts/25_fix_yaml.py 来修改 model yaml 的 checkpoint_path（不动 yaml 之外）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(os.getcwd())  # 直接 cwd（用 forward slash 形式），不 resolve
# 防 escape：所有 path 构造走 as_posix + 字符串拼接 + 最后 Path()
PROJECT_ROOT_POSIX = PROJECT_ROOT.as_posix()  # 'E:/Q4 - 副本'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-config", type=str, default="configs/datasets/paper_dataset.yaml")
    parser.add_argument("--events-root", type=str, default="outputs/prepare_paper_main_v1")
    parser.add_argument("--raw-root", type=str, default="data/raw/paper_main_v1")
    parser.add_argument("--train-root", type=str, default="outputs/train_paper_main")
    parser.add_argument("--output-root", type=str, default="outputs/e20_paper_main")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "full"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", type=str, nargs="+", default=["lstm_ekf", "liquid_ekf", "transformer_ekf"])
    parser.add_argument("--max-sequences", type=int, default=None, help="cap on number of seq_ids to run (quick 默认 40)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[24] os.getcwd(): {os.getcwd()}")
    print(f"[24] __file__: {__file__}")
    print(f"[24] PROJECT_ROOT: {PROJECT_ROOT}")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    # 1. 改 yaml checkpoint_path
    import subprocess
    fix_cmd = [sys.executable, "scripts/25_fix_yaml.py", "--train-root", args.train_root, "--seed", str(args.seed)]
    print(f"[24] running: {' '.join(fix_cmd)}")
    rc = subprocess.call(fix_cmd, cwd=str(PROJECT_ROOT))
    if rc != 0:
        print(f"[24] 25_fix_yaml failed: rc={rc}")
        return rc

    try:
        # 1. inline 改 yaml checkpoint_path（避免 subprocess 路径 + 中文 path bug）
        import yaml as _yaml
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
        yaml_backups = {}
        for method, yaml_rel in MODEL_YAML.items():
            if method not in args.methods:
                continue
            yaml_path = Path(PROJECT_ROOT) / yaml_rel
            yaml_backups[yaml_path] = yaml_path.read_text(encoding="utf-8")
            ckpt_name = CHECKPOINT_FILE[method]
            ckpt = Path(PROJECT_ROOT) / args.train_root / f"seed{args.seed}" / method / "checkpoints" / ckpt_name
            if not ckpt.exists():
                # fallback: 任一可用的 best/resume ckpt
                alts = sorted((Path(PROJECT_ROOT) / args.train_root / f"seed{args.seed}" / method / "checkpoints").glob("*.pt"))
                if alts:
                    ckpt = alts[-1]
            cfg = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            cfg["checkpoint_path"] = str(ckpt).replace("\\", "/")
            yaml_path.write_text(
                _yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            print(f"[24] set {yaml_rel} -> {ckpt.name}")

        # 防 escape：所有 path 走 forward slash 拼接
        def _safe_path(rel: str) -> Path:
            base = PROJECT_ROOT.as_posix().rstrip("/")
            posix = f"{base}/{rel.lstrip('/')}"
            return Path(posix)
        prepare_root = _safe_path(args.events_root)
        raw_root = _safe_path(args.raw_root)
        output_root = _safe_path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        manifest_path = prepare_root / "prepare_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        seq_ids = sorted(manifest.get("sequences", {}).keys())
        if args.max_sequences is not None:
            seq_ids = seq_ids[: args.max_sequences]
        print(f"[24] {len(seq_ids)} seq_ids from {manifest_path}")
        print(f"[24] raw_root={raw_root}, exists={raw_root.exists()}")
        print(f"[24] first seq anchor_layout exists: {(raw_root / seq_ids[0] / 'anchor_layout.json').exists()}")

        # 3. 加载 prepared events
        from liquidloc.common.prepared_inputs import (
            load_prepared_events_by_seq_id,
            load_ground_truth_by_seq_id,
            load_source_report_by_seq_id,
        )
        events_by_seq = load_prepared_events_by_seq_id(prepare_root, seq_ids)
        gt_by_seq = load_ground_truth_by_seq_id(raw_root, seq_ids)
        src_by_seq = load_source_report_by_seq_id(raw_root, manifest, seq_ids, default_source="paper_main_supplementary")
        print(f"[24] loaded events total: {sum(len(v) for v in events_by_seq.values())}")

        # 4. 构造 scene_tasks（注入 anchor_layout）
        axes_by_seq = {}
        scene_id_by_seq = {}
        anchor_layout_by_seq = {}
        for sid in seq_ids:
            meta_path = raw_root / sid / "sim_meta.json"
            layout_path = raw_root / sid / "anchor_layout.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                ax = meta.get("axes_override", {})
                axes_by_seq[sid] = {k: ax.get(k, "A2" if k in ("A", "N") else "V0" if k == "V" else "K0" if k == "K" else "M0") for k in ("A", "N", "V", "K", "M")}
                scene_id_by_seq[sid] = f"S({axes_by_seq[sid]['A']},{axes_by_seq[sid]['N']},{axes_by_seq[sid]['V']},{axes_by_seq[sid]['K']},{axes_by_seq[sid]['M']})"
            else:
                axes_by_seq[sid] = {"A": "A2", "N": "N2", "V": "V0", "K": "K0", "M": "M0"}
                scene_id_by_seq[sid] = "S(A2,N2,V0,K0,M0)"
            # 优先用 manifest 里存好的 anchor_layout（prepare 时已从 raw_root 复制）
            # raw_root/sid/anchor_layout.json 可能因 symlink 断连而丢失
            manifest_entry = manifest.get("sequences", {}).get(sid, {})
            manifest_al = manifest_entry.get("anchor_layout")
            layout_path = raw_root / sid / "anchor_layout.json"
            if manifest_al:
                anchor_layout_by_seq[sid] = manifest_al
            elif layout_path.exists():
                anchor_layout_by_seq[sid] = json.loads(layout_path.read_text(encoding="utf-8"))
            else:
                anchor_layout_by_seq[sid] = None

        scene_tasks = []
        for sid in seq_ids:
            for method in args.methods:
                # task_id 单下划线（Windows 文件名禁止 `::`）
                scene_tasks.append({
                    "task_id": f"paper_{sid}_{method}",
                    "seq_id": sid,
                    "scene_id": scene_id_by_seq[sid],
                    "axes": axes_by_seq[sid],
                    "method": method,
                    "dataset_name": "paper_main",
                    "anchor_layout": anchor_layout_by_seq[sid],
                })
        # 调试：打印前 3 个 task 的 anchor_layout 状态
        for t in scene_tasks[:3]:
            al = t.get('anchor_layout') or {}
            print(f"[24] seq={t['seq_id']} method={t['method']} anchor_layout ids={al.get('anchor_ids')}, count={al.get('anchor_count')}")

        # 5. 跑 CorePipeline：把全部 scene_tasks 一次性传入，避免每 task 重建 model
        from liquidloc.pipelines.core_pipeline import CorePipeline
        cp = CorePipeline()
        core_output = output_root / "core"
        core_output.mkdir(parents=True, exist_ok=True)
        all_bundles = []
        task_failures = 0
        # 分批：每批 batch_size 个 tasks（避免一次传入太多 events 占内存）
        BATCH_SIZE = 20
        for bi in range(0, len(scene_tasks), BATCH_SIZE):
            batch = scene_tasks[bi:bi + BATCH_SIZE]
            try:
                core_result = cp.run({
                    "scene_tasks": batch,
                    "events_by_seq_id": events_by_seq,
                    "ground_truth_by_seq_id": gt_by_seq,
                    "source_report_by_seq_id": src_by_seq,
                    "methods": args.methods,
                    "output_root": str(core_output),
                })
                bundles = core_result.metadata.get("prediction_bundles") or []
                all_bundles.extend(bundles)
                if (bi // BATCH_SIZE + 1) % 5 == 0:
                    print(f"[24]   progress: batch {bi//BATCH_SIZE+1}, {len(all_bundles)} bundles", flush=True)
            except Exception as e:
                task_failures += len(batch)
                if task_failures <= 5:
                    print(f"[24] batch starting at task {bi} failed: {type(e).__name__}: {e}", flush=True)
                continue
        bundles = all_bundles
        print(f"[24] CorePipeline: {len(bundles)} bundles, {task_failures} task failures (over {len(scene_tasks)} total)")

        # 6. 计算指标并写出 metric_table.csv
        from liquidloc.metrics.metric_runner import compute_metrics
        from liquidloc.common.config_utils import load_yaml_config
        protocol_cfg = load_yaml_config(PROJECT_ROOT / "configs" / "base" / "experiment_protocol.yaml")
        # paper_main_v1 是 20m×20m 物理尺寸场景，5m failure 阈值与 sim_e9 协议的 "可解率 65-74%"
        # 区间一致（1m 过严，10m 过松）。每个 sim_e9 评估也用 5m threshold。
        failure_threshold = 5.0
        all_rows = []
        for bundle in bundles:
            sid = bundle.get("seq_id")
            method = bundle.get("method_name", "unknown")
            preds = bundle.get("predictions") or []
            gt_seq = gt_by_seq.get(sid, [])
            if not gt_seq:
                continue
            try:
                metrics = compute_metrics(preds, gt_seq, failure_threshold, return_support=False, protocol_cfg=protocol_cfg)
                row = {"case_ref": f"paper::{sid}::{method}", "seq_id": sid, "method_name": method, **metrics}
                all_rows.append(row)
            except Exception as e:
                print(f"[24] metrics failed for {sid}/{method}: {e}")
                continue
        if all_rows:
            metrics_path = output_root / "metric_table.csv"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metric_keys = sorted({k for r in all_rows for k in r if k not in ("case_ref", "seq_id", "method_name")})
            with open(metrics_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["case_ref", "seq_id", "method_name"] + metric_keys)
                w.writeheader()
                for r in all_rows:
                    w.writerow(r)
            print(f"[24] wrote {metrics_path} ({len(all_rows)} rows)")
            # 输出每个方法的 RMSE / P95 / failure_rate 平均
            agg = defaultdict(list)
            for r in all_rows:
                for k in ("rmse", "p95", "failure_rate"):
                    if k in r:
                        agg[(r["method_name"], k)].append(float(r[k]))
            print(f"\n[24] === aggregate by method (over {len(seq_ids)} seq_ids) ===")
            print(f"{'method':<20} {'rmse_mean':>10} {'p95_mean':>10} {'fail_mean':>10}")
            for method in args.methods:
                if (method, "rmse") in agg:
                    rmse = sum(agg[(method, "rmse")]) / len(agg[(method, "rmse")])
                    p95 = sum(agg[(method, "p95")]) / len(agg[(method, "p95")]) if agg[(method, "p95")] else 0
                    fail = sum(agg[(method, "failure_rate")]) / len(agg[(method, "failure_rate")]) if agg[(method, "failure_rate")] else 0
                    print(f"{method:<20} {rmse:>10.3f} {p95:>10.3f} {fail:>10.3f}")
        return 0
    finally:
        # inline 恢复 yaml（避免 subprocess 路径 bug）
        for yaml_path, original_text in yaml_backups.items():
            yaml_path.write_text(original_text, encoding="utf-8")
        print("[24] yaml restored")


if __name__ == "__main__":
    sys.exit(main())
