#!/usr/bin/env python
"""训练 5 method × 10 seed = 50 个 unit, 默认 epochs=80 (手册推荐 160, 但 8GB GPU 资源紧张, 先 80 出 paper grade)。
"""
import os, sys, io, contextlib, json, time, gc, shutil, argparse
from pathlib import Path

ROOT = Path("E:/异步高NLOS")
sys.path.insert(0, str(ROOT / "src"))
os.environ["LIQUIDLOC_SUPPRESS_PRINT_DICT"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

METHODS = ["lstm", "transformer", "liquid"]  # 3 NN; EKF + Robust-EKF 无需训练
SCRIPT_MAP = {
    "lstm": "05_train_lstm.py",
    "transformer": "07_train_transformer.py",
    "liquid": "06_train_liquid.py",
}
SEQ_BATCH = {"lstm": 4, "transformer": 4, "liquid": 1}  # liquid 8GB 紧张


def train_one(method: str, seed: int, epochs: int, processed_root: Path) -> tuple[bool, str]:
    script = SCRIPT_MAP[method]
    split_file = processed_root / f"seed{seed}" / "splits" / "train_ids.txt"
    if not split_file.exists():
        # 退到不带 splits 子目录的版本（直接 train_ids.txt）
        split_file = processed_root / f"seed{seed}" / "train_ids.txt"
    if not split_file.exists():
        return False, f"no train_ids.txt at {split_file}"
    out_dir = ROOT / "checkpoints" / f"{method}_seed{seed}"
    if out_dir.exists() and any(out_dir.glob("checkpoints/*_best*.pt")):
        return True, "skip (ckpt exists)"
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
    import subprocess
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = subprocess.run(cmd, env={**os.environ, "PYTHONHASHSEED": "0"})
    if result.returncode != 0:
        return False, f"train failed (exit {result.returncode}, {time.time()-t0:.0f}s)"
    return True, f"ok ({time.time()-t0:.0f}s)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--processed-root", default="E:/异步高NLOS/data/processed/sim_e9_main")
    parser.add_argument("--methods", default="lstm,transformer,liquid")
    args = parser.parse_args()
    processed_root = Path(args.processed_root)
    methods = args.methods.split(",")

    print(f"[train_all] n_seeds={args.n_seeds}, epochs={args.epochs}, methods={methods}", flush=True)
    t0 = time.time()
    ok = 0
    total = args.n_seeds * len(methods)
    cur = 0
    for method in methods:
        for seed in range(args.n_seeds):
            cur += 1
            ok_flag, msg = train_one(method, seed, args.epochs, processed_root)
            if ok_flag:
                ok += 1
            print(f"  [{cur}/{total}] {method} seed{seed}: {msg}", flush=True)
            gc.collect()
    print(f"[train_all] DONE: {ok}/{total} ok, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
