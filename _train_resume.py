#!/usr/bin/env python
"""Resumable 30-unit training orchestrator (LSTM/Transformer/Liquid × 10 seeds).
- 跳过已存在 best ckpt 的单元 (断点续跑)
- 每个方法逐 seed 串行 (避免 GPU 抢占)
- 每个 method 完成后 print summary
"""
import os, sys, io, contextlib, json, time, gc, shutil, subprocess, argparse
from pathlib import Path

ROOT = Path("E:/异步高NLOS")
sys.path.insert(0, str(ROOT / "src"))
os.environ["LIQUIDLOC_SUPPRESS_PRINT_DICT"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

METHODS = ["lstm", "transformer", "liquid"]
SCRIPT = {
    "lstm": "05_train_lstm.py",
    "transformer": "07_train_transformer.py",
    "liquid": "06_train_liquid.py",
}
SEQ_BATCH = {"lstm": 4, "transformer": 4, "liquid": 1}


def train_one(method: str, seed: int, epochs: int, processed_root: Path) -> tuple[str, float]:
    """Return (status, time_seconds). 跳过已存在 best ckpt 的单元."""
    script = SCRIPT[method]
    split_file = processed_root / f"seed{seed}" / "train_ids.txt"
    if not split_file.exists():
        return ("skip: no train_ids", 0.0)
    ckpt = ROOT / "checkpoints" / f"{method}_seed{seed}" / "checkpoints" / f"{method}_ekf_best_checkpoint.pt"
    if ckpt.exists() and ckpt.stat().st_size > 1000:
        return ("skip: ckpt exists", 0.0)
    out_dir = ROOT / "checkpoints" / f"{method}_seed{seed}"
    cmd = [
        sys.executable, str(ROOT / "scripts" / script),
        "--dataset-name", "sim",
        "--events-root", str(processed_root / f"seed{seed}"),
        "--raw-root", str(ROOT / "data" / "raw" / "sim_e9_main" / f"seed{seed}"),
        "--split-ids-file", str(split_file),
        "--train-split-ids-file", str(split_file),
        "--val-split-ids-file", str(split_file.parent / "val_ids.txt"),
        "--epochs", str(epochs),
        "--seq-batch-size", str(SEQ_BATCH[method]),
        "--output-root", str(out_dir),
        "--disable-scene-leak-check",
    ]
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = subprocess.run(cmd, env={**os.environ, "PYTHONHASHSEED": "0"})
    return (("ok" if result.returncode == 0 else f"fail:{result.returncode}"), time.time() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--processed-root", default="E:/异步高NLOS/data/processed/sim_e9_main")
    args = parser.parse_args()
    processed_root = Path(args.processed_root)

    print(f"[train_resume] n_seeds={args.n_seeds}, epochs={args.epochs}, methods={METHODS}", flush=True)
    t0 = time.time()
    summary = {m: {"ok": 0, "skip": 0, "fail": 0, "time": 0.0} for m in METHODS}
    total = args.n_seeds * len(METHODS)
    cur = 0
    for method in METHODS:
        for seed in range(args.n_seeds):
            cur += 1
            status, t = train_one(method, seed, args.epochs, processed_root)
            if status == "ok":
                summary[method]["ok"] += 1
                summary[method]["time"] += t
            elif "skip" in status:
                summary[method]["skip"] += 1
            else:
                summary[method]["fail"] += 1
            print(f"  [{cur}/{total}] {method} seed{seed}: {status} ({t:.0f}s)", flush=True)
            gc.collect()
    print(f"\n[train_resume] DONE: {time.time()-t0:.0f}s", flush=True)
    for m, s in summary.items():
        print(f"  {m}: ok={s['ok']} skip={s['skip']} fail={s['fail']} time={s['time']:.0f}s")


if __name__ == "__main__":
    main()
