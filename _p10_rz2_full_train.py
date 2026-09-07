"""P10 RZ-2 全量训练调度：10 seed × 5 method = 50 单元，GPU 10-20h 跨会话。

按 manual Part 4 §Step 4:
- 训练默认 GPU (Pre-5), 顺序 1 method × 10 seed per pass
- batch=16, epochs<=160 (best-val early stop P33), TF32+AMP+deterministic
- paper-rz0-freeze tag 已落 (commit f75586dc, 7a755095, 2b6d3a1d)
- 数据: data/processed/sim_e9_paper/seed{0..9}/ (RZ-1 重生成, 140 seq/seed)
- 切分: train_ids.txt / test_ids.txt (80/20 split, 112/28 per seed)

用 _train_all_methods.py (已存在) 但需要指定  --epochs=160 --batch_size=16 --device=cuda --tf32 --amp。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("E:/异步高NLOS")
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("LIQUIDLOC_SUPPRESS_PRINT_DICT", "1")
os.environ.setdefault("LIQUIDLOC_ALIGNMENT_POSE_FULL_SCALE_M", "25.0")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

METHODS = ["lstm", "transformer", "liquid", "ekf", "robust_ekf"]
# 5 methods × 10 seeds = 50 units
# LSTM/Transformer/Liquid: 02_train_lstm/07_train_transformer/06_train_liquid
# EKF/Robust-EKF: 07_run_baselines (no training, quick)
TRAIN_SCRIPT = {
    "lstm": "05_train_lstm.py",
    "transformer": "07_train_transformer.py",
    "liquid": "06_train_liquid.py",
    "ekf": "07_run_baselines.py",  # 0 epoch, 真实推理
    "robust_ekf": "07_run_baselines.py",  # 同上
}
N_SEEDS = 10
EPOCHS = 160  # upper budget; best-val early stop
BATCH = 16
SEQ_BATCH = 4  # streaming sub-batch (avoid OOM)
DEVICE = "cuda"
TF32 = True
AMP = True

DATA_ROOT = ROOT / "data" / "processed" / "sim_e9_paper"
RAW_ROOT = ROOT / "data" / "raw" / "sim_e9_paper"
OUTPUT_ROOT = ROOT / "checkpoints" / "paper_full_50unit"
LOG_ROOT = ROOT / "logs" / "paper_full_50unit"
LOG_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Plan
plan = []
for m in METHODS:
    for s in range(N_SEEDS):
        plan.append((m, s))
print(f"Plan: {len(plan)} units = {len(METHODS)} methods × {N_SEEDS} seeds", flush=True)


def run_unit(method: str, seed: int) -> dict:
    """Train or evaluate one (method, seed) unit."""
    output_dir = OUTPUT_ROOT / f"{method}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{method}_seed{seed}.log"
    ckpt_path = output_dir / "checkpoints" / f"{method}_ekf_best_checkpoint.pt"

    train_ids = DATA_ROOT / f"seed{seed}" / "train_ids.txt"
    test_ids = DATA_ROOT / f"seed{seed}" / "test_ids.txt"
    if not train_ids.exists():
        return {"method": method, "seed": seed, "status": "SKIP", "reason": "no train_ids.txt"}
    if not test_ids.exists():
        return {"method": method, "seed": seed, "status": "SKIP", "reason": "no test_ids.txt"}

    cmd = ["python", str(ROOT / "scripts" / TRAIN_SCRIPT[method])]
    if method in ("lstm", "transformer", "liquid"):
        cmd += [
            "--dataset-name=sim",
            f"--events-root={DATA_ROOT / f'seed{seed}'}",
            f"--raw-root={RAW_ROOT / f'seed{seed}'}",
            f"--split-ids-file={train_ids}",
            f"--output-root={output_dir}",
            f"--epochs={EPOCHS}",
            f"--seq-batch-size={SEQ_BATCH}",
            f"--seed={seed}",
        ]
    else:
        # ekf/robust_ekf baselines: no training, just inference
        cmd += [
            "--methods=ekf" if method == "ekf" else "--methods=robust_ekf",
            f"--processed-root={DATA_ROOT / f'seed{seed}'}",
            f"--raw-root={RAW_ROOT / f'seed{seed}'}",
            f"--output={output_dir / 'results.json'}",
        ]

    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(f"=== {method} seed{seed} started {time.ctime()} ===\n")
        f.write(f"cmd: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    return {
        "method": method, "seed": seed, "status": status,
        "log": str(log_path), "elapsed_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="Comma-separated methods to run (default: all 5)")
    ap.add_argument("--seeds", default="0-9",
                    help="Seed range like 0-9 or comma list 0,1,2")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan only, don't execute")
    ap.add_argument("--resume", action="store_true",
                    help="Skip units with existing output checkpoint/results")
    args = ap.parse_args()

    selected_methods = args.methods.split(",")
    # Parse seeds
    if "-" in args.seeds and "," not in args.seeds:
        lo, hi = args.seeds.split("-")
        selected_seeds = list(range(int(lo), int(hi) + 1))
    else:
        selected_seeds = [int(s) for s in args.seeds.split(",")]

    plan = [(m, s) for m in selected_methods for s in selected_seeds]

    # Resume support
    if args.resume:
        filtered = []
        for m, s in plan:
            if m in ("lstm", "transformer", "liquid"):
                ck = OUTPUT_ROOT / f"{m}_seed{s}" / "checkpoints" / f"{m}_ekf_best_checkpoint.pt"
            else:
                res = OUTPUT_ROOT / f"{m}_seed{s}" / "results.json"
                ck = res
            if ck.exists():
                print(f"SKIP {m} seed{s} (exists: {ck.name})")
                continue
            filtered.append((m, s))
        plan = filtered

    print(f"\n=== P10 RZ-2 全量训练计划 ===")
    print(f"Methods: {selected_methods}")
    print(f"Seeds: {selected_seeds}")
    print(f"Total units: {len(plan)}")
    print(f"Data: {DATA_ROOT} (10 seeds x 140 seqs x 120-132s)")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Logs: {LOG_ROOT}")
    print(f"Per-unit estimate: ~10-30min (LSTM) / ~15-45min (Transformer) / ~30-60min (Liquid)")
    print(f"Total estimate: {len(plan) * 30}-{len(plan) * 60}min = {len(plan) * 0.5:.1f}-{len(plan) * 1.0:.1f}h")
    print(f"Code freeze: paper-rz0-freeze tag")
    print(f"Config: epochs={EPOCHS} batch={BATCH} seq_batch={SEQ_BATCH} device={DEVICE} tf32={TF32} amp={AMP}")
    print()

    if args.dry_run:
        for m, s in plan:
            print(f"  {m:12s} seed{s}")
        return

    results = []
    for i, (m, s) in enumerate(plan):
        print(f"\n[{i+1}/{len(plan)}] Running {m} seed{s} ...")
        r = run_unit(m, s)
        results.append(r)
        print(f"  -> {r['status']} ({r.get('elapsed_s', 0):.0f}s) log={r.get('log', '?')}")

    print(f"\n=== P10 RZ-2 done ===")
    for r in results:
        print(f"  {r['method']:12s} seed{r['seed']}: {r['status']} ({r.get('elapsed_s',0):.0f}s)")

    summary = ROOT / "logs" / "paper_full_50unit" / "summary.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()