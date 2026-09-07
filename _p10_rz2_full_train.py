"""P10 RZ-2 全量训练调度：50 单元 (3 NN 方法 x 10 seed) + 20 基线单元 (2 EKF 方法 x 10 seed)，GPU 10-20h 跨会话。

按手册 Part 4 §Step 4 / RZ-2 重训 gate (指令 44):
- batch=16 (C-1/C-2 硬约束), epochs<=160 (P33 best-val 早停决定实际停止点)
- TF32 + AMP + deterministic (P19)
- paper-rz0-freeze tag 已落 (commit f75586dc, 7a755095, 2b6d3a1d)
- 数据: data/processed/sim_e9_paper/seed{0..9}/ (RZ-1 重生成, 140 seq/seed)
- 切分: train_ids.txt (111 seq) / test_ids.txt (27 seq) per seed
- 显式 train_split + val_split 以支持 P33 best-val 早停
- 基线 (EKF/Robust-EKF) 不训练, 仅推理 (07_run_baselines.py)

用法:
    python _p10_rz2_full_train.py --methods=lstm,liquid,transformer --seeds=0-9 --epochs=160 --batch=16 --resume
    python _p10_rz2_full_train.py --dry-run --methods=lstm,liquid,transformer --seeds=0-9
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

# 按手册 RZ-2: 5 methods x 10 seeds = 50 units, 基线不训练
METHODS = ["lstm", "transformer", "liquid", "ekf", "robust_ekf"]
NN_METHODS = {"lstm", "transformer", "liquid"}  # 需要训练的 NN 方法
TRAIN_SCRIPT = {
    "lstm": "05_train_lstm.py",
    "transformer": "07_train_transformer.py",
    "liquid": "06_train_liquid.py",
}
N_SEEDS = 10
EPOCHS = 160  # upper budget; P33 best-val 早停决定实际停止点
BATCH = 16    # 手册 C-1/C-2 硬约束
SEQ_BATCH = 4  # streaming sub-batch (防 OOM)

DATA_ROOT = ROOT / "data" / "processed" / "sim_e9_paper"
RAW_ROOT = ROOT / "data" / "raw" / "sim_e9_paper"
OUTPUT_ROOT = ROOT / "checkpoints" / "paper_full_50unit"
LOG_ROOT = ROOT / "logs" / "paper_full_50unit"
LOG_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def run_unit(method: str, seed: int) -> dict:
    """Train or evaluate one (method, seed) unit."""
    output_dir = OUTPUT_ROOT / f"{method}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{method}_seed{seed}.log"

    train_ids = DATA_ROOT / f"seed{seed}" / "train_ids.txt"
    test_ids = DATA_ROOT / f"seed{seed}" / "test_ids.txt"
    if not train_ids.exists():
        return {"method": method, "seed": seed, "status": "SKIP", "reason": "no train_ids.txt"}
    if not test_ids.exists():
        return {"method": method, "seed": seed, "status": "SKIP", "reason": "no test_ids.txt"}

    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(f"=== {method} seed{seed} started {time.ctime()} ===\n")
        f.write(f"data: {DATA_ROOT / f'seed{seed}'}\n")
        f.flush()

        if method in NN_METHODS:
            cmd = [
                "python", str(ROOT / "scripts" / TRAIN_SCRIPT[method]),
                "--dataset-name=sim",
                f"--events-root={DATA_ROOT / f'seed{seed}'}",
                f"--raw-root={RAW_ROOT / f'seed{seed}'}",
                f"--split-ids-file={train_ids}",
                f"--train-split-ids-file={train_ids}",
                f"--val-split-ids-file={test_ids}",
                f"--output-root={output_dir}",
                f"--epochs={EPOCHS}",
                f"--seq-batch-size={SEQ_BATCH}",
                f"--seed={seed}",
            ]
            if method in ("transformer", "liquid"):
                cmd.append(f"--batch-size={BATCH}")
            # LSTM: batch_size 由 configs/models/lstm_ekf.yaml 控制 (已改为 16)
            f.write(f"cmd: {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        else:
            # EKF / Robust-EKF: 仅推理, 不训练
            cmd = [
                "python", str(ROOT / "scripts" / "07_run_baselines.py"),
                f"--methods={method}",
                f"--processed-root={DATA_ROOT / f'seed{seed}'}",
                f"--raw-root={RAW_ROOT / f'seed{seed}'}",
                f"--output={output_dir / 'results.json'}",
            ]
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
    ap = argparse.ArgumentParser(description="P10 RZ-2 全量训练调度")
    ap.add_argument("--methods", default="lstm,liquid,transformer",
                    help="Comma-separated methods to train (default: lstm,liquid,transformer)")
    ap.add_argument("--seeds", default="0-9",
                    help="Seed range like 0-9 or comma list 0,1,2")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help=f"Training epochs upper budget (default: {EPOCHS})")
    ap.add_argument("--batch", type=int, default=BATCH,
                    help=f"Training batch size (default: {BATCH}, manual C-1/C-2 requires 16)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan only, don't execute")
    ap.add_argument("--resume", action="store_true",
                    help="Skip units with existing checkpoint/results")
    args = ap.parse_args()

    selected_methods = [m.strip() for m in args.methods.split(",")]
    if "-" in args.seeds and "," not in args.seeds:
        lo, hi = args.seeds.split("-")
        selected_seeds = list(range(int(lo), int(hi) + 1))
    else:
        selected_seeds = [int(s) for s in args.seeds.split(",")]

    plan = [(m, s) for m in selected_methods for s in selected_seeds]
    print(f"Plan: {len(plan)} units = {len(selected_methods)} methods x {len(selected_seeds)} seeds", flush=True)

    # Resume support
    if args.resume:
        filtered = []
        for m, s in plan:
            if m in NN_METHODS:
                ck = OUTPUT_ROOT / f"{m}_seed{s}" / "checkpoints" / f"{m}_ekf_best_checkpoint.pt"
            else:
                ck = OUTPUT_ROOT / f"{m}_seed{s}" / "results.json"
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
    print(f"Config: epochs={args.epochs} batch={args.batch} seq_batch={SEQ_BATCH}")
    print(f"Code freeze: paper-rz0-freeze tag")
    print()

    if args.dry_run:
        for m, s in plan:
            print(f"  {m:15s} seed{s}")
        return

    results = []
    for i, (m, s) in enumerate(plan):
        print(f"\n[{i+1}/{len(plan)}] Running {m} seed{s} ...", flush=True)
        r = run_unit(m, s)
        results.append(r)
        print(f"  -> {r['status']} ({r.get('elapsed_s', 0):.0f}s) log={r.get('log', '?')}", flush=True)

    print(f"\n=== P10 RZ-2 done ===")
    for r in results:
        print(f"  {r['method']:15s} seed{r['seed']}: {r['status']} ({r.get('elapsed_s', 0):.0f}s)")

    summary = LOG_ROOT / "summary.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
