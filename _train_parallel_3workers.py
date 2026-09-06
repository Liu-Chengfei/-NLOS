#!/usr/bin/env python
"""并发训练 30 unit (3 NN × 10 seeds).
3 worker 并行, 每 worker 顺序跑自己 method 的 10 seeds.
- 跳过已存在 best ckpt (断点续跑)
- 每 worker 跑完打印 summary
- 全部完成打印 final summary
"""
import os, sys, io, contextlib, time, gc, subprocess, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path("E:/异步高NLOS")
sys.path.insert(0, str(ROOT / "src"))
os.environ["LIQUIDLOC_SUPPRESS_PRINT_DICT"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

METHODS = ["lstm", "transformer", "liquid"]  # 3 workers 并行
SCRIPT = {
    "lstm": "05_train_lstm.py",
    "transformer": "07_train_transformer.py",
    "liquid": "06_train_liquid.py",
}
SEQ_BATCH = {"lstm": 4, "transformer": 4, "liquid": 1}


def train_one(method: str, seed: int, epochs: int, processed_root: str) -> tuple[str, int, float]:
    """Train one method×seed. Returns (status, seed, time_seconds)."""
    import logging
    script = SCRIPT[method]
    pr = Path(processed_root)
    split_file = pr / f"seed{seed}" / "train_ids.txt"
    if not split_file.exists():
        return ("skip: no train_ids", seed, 0.0)
    ckpt = Path(f"E:/异步高NLOS/checkpoints/{method}_seed{seed}/checkpoints/{method}_ekf_best_checkpoint.pt")
    if ckpt.exists() and ckpt.stat().st_size > 1000:
        return ("skip: ckpt exists", seed, 0.0)
    out_dir = Path(f"E:/异步高NLOS/checkpoints/{method}_seed{seed}")
    pr = Path(processed_root)
    cmd = [
        sys.executable, str(ROOT / "scripts" / script),
        "--dataset-name", "sim",
        "--events-root", str(pr / f"seed{seed}"),
        "--raw-root", f"E:/异步高NLOS/data/raw/sim_e9_main/seed{seed}",
        "--split-ids-file", str(split_file),
        "--train-split-ids-file", str(split_file),
        "--val-split-ids-file", str(split_file.parent / "val_ids.txt"),
        "--epochs", str(epochs),
        "--seq-batch-size", str(SEQ_BATCH[method]),
        "--output-root", str(out_dir),
        "--disable-scene-leak-check",
    ]
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    return ((f"ok" if result.returncode == 0 else f"fail:{result.returncode}"), seed, dt)


def worker_train_method(method: str, seeds: list[int], epochs: int, processed_root: str) -> dict:
    """One worker: sequentially train method across all seeds."""
    print(f"[worker:{method}] start | {len(seeds)} seeds | epochs={epochs}", flush=True)
    results = {"method": method, "ok": 0, "skip": 0, "fail": 0, "time": 0.0}
    for seed in seeds:
        status, s, t = train_one(method, seed, epochs, processed_root)
        if status == "ok":
            results["ok"] += 1
            results["time"] += t
        elif "skip" in status:
            results["skip"] += 1
        else:
            results["fail"] += 1
        print(f"  [{method}] seed{s}: {status} ({t:.0f}s)", flush=True)
        gc.collect()
    print(f"[worker:{method}] DONE | ok={results['ok']} skip={results['skip']} fail={results['fail']} time={results['time']:.0f}s", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--processed-root", default="E:/异步高NLOS/data/processed/sim_e9_main")
    parser.add_argument("--max-workers", type=int, default=3, help="concurrent workers (one per method)")
    args = parser.parse_args()
    processed_root = args.processed_root
    seeds = list(range(args.n_seeds))

    print(f"[train_parallel] methods={METHODS} | {len(seeds)} seeds | epochs={args.epochs} | workers={args.max_workers}", flush=True)
    t0 = time.time()
    all_results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(worker_train_method, m, seeds, args.epochs, processed_root): m for m in METHODS}
        for f in as_completed(futures):
            method = futures[f]
            try:
                r = f.result()
                all_results.append(r)
            except Exception as e:
                print(f"[{method}] FAIL: {e}", flush=True)
    total = time.time() - t0
    print(f"\n[train_parallel] ALL DONE | {total:.0f}s", flush=True)
    for r in all_results:
        print(f"  {r['method']}: ok={r['ok']} skip={r['skip']} fail={r['fail']} time={r['time']:.0f}s")
    # total ok
    total_ok = sum(r["ok"] for r in all_results)
    total_skip = sum(r["skip"] for r in all_results)
    total_fail = sum(r["fail"] for r in all_results)
    print(f"\nSummary: ok={total_ok} skip={total_skip} fail={total_fail}")


if __name__ == "__main__":
    main()
