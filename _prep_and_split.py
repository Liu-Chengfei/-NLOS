#!/usr/bin/env python
"""对 N seeds × 60 specs 数据跑 02_prepare_sim_data.py + 04_build_splits.py。

执行:
  python _prep_and_split.py --n-seeds 10 --raw-root data/raw/sim_e9_main --processed-root data/processed/sim_e9_main
"""
import os, sys, io, contextlib, json, time, gc, shutil
from pathlib import Path

ROOT = Path("E:/异步高NLOS")
sys.path.insert(0, str(ROOT / "src"))
os.environ["LIQUIDLOC_SUPPRESS_PRINT_DICT"] = "1"
os.environ.setdefault("STAR_A95_MAX", "50.0")
os.environ["PYTHONHASHSEED"] = "0"


def prep_one_seed(seed_idx: int, raw_root: Path, processed_root: Path) -> tuple[bool, str]:
    raw_seed = raw_root / f"seed{seed_idx}"
    proc_seed = processed_root / f"seed{seed_idx}"
    if proc_seed.exists() and any(proc_seed.glob("*_events.pkl.gz")):
        return True, "skip (exists)"
    proc_seed.mkdir(parents=True, exist_ok=True)
    out_dir = proc_seed.parent / f"_tmp_seed{seed_idx}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "02_prepare_sim_data.py"),
        "--raw-root", str(raw_seed),
        "--output-root", str(out_dir),
        "--n-seed", "1",
    ]
    import subprocess
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = subprocess.run(cmd, env={**os.environ, "PYTHONHASHSEED": "0"})
    if result.returncode != 0:
        return False, f"prepare failed (exit {result.returncode})"
    # 移子目录 — 02_prepare_sim_data.py 直接写到 output_root (无嵌套子目录)
    for child in out_dir.iterdir():
        target = proc_seed / child.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(child), str(target))
    shutil.rmtree(out_dir, ignore_errors=True)
    return True, "ok"


def split_one_seed(seed_idx: int, processed_root: Path) -> tuple[bool, str]:
    proc_seed = processed_root / f"seed{seed_idx}"
    if not proc_seed.exists():
        return False, "no processed data"
    split_file = proc_seed / "split_manifest.json"
    if split_file.exists():
        return True, "skip (exists)"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "04_build_splits.py"),
        "--manifests-root", str(proc_seed),
        "--output-root", str(proc_seed),
    ]
    import subprocess
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = subprocess.run(cmd, env={**os.environ, "PYTHONHASHSEED": "0"})
    if result.returncode not in (0, 2):  # rc=2 = is_clean False but write OK
        return False, f"split failed (exit {result.returncode})"
    # 生成 train/val/test_ids.txt from split_manifest.json
    import json
    with open(proc_seed / "split_manifest.json") as f:
        sm = json.load(f)
    def _write_ids(path, ids):
        with open(path, "w") as f:
            f.write(",".join(ids))
    _write_ids(proc_seed / "train_ids.txt", sm["train_ids"])
    _write_ids(proc_seed / "val_ids.txt", sm["val_ids"])
    _write_ids(proc_seed / "test_ids.txt", sm["test_ids"])
    return True, "ok"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--raw-root", default="E:/异步高NLOS/data/raw/sim_e9_main")
    parser.add_argument("--processed-root", default="E:/异步高NLOS/data/processed/sim_e9_main")
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)

    print(f"[prep_split] n_seeds={args.n_seeds}", flush=True)
    print(f"[prep_split] raw={raw_root}, processed={processed_root}", flush=True)

    t0 = time.time()
    ok_count = 0
    for seed in range(args.n_seeds):
        raw_seed = raw_root / f"seed{seed}"
        if not raw_seed.exists():
            print(f"  seed{seed}: SKIP (no raw)", flush=True)
            continue
        ok1, msg1 = prep_one_seed(seed, raw_root, processed_root)
        if not ok1:
            print(f"  seed{seed}: PREP FAIL {msg1}", flush=True)
            continue
        ok2, msg2 = split_one_seed(seed, processed_root)
        if not ok2:
            print(f"  seed{seed}: SPLIT FAIL {msg2}", flush=True)
            continue
        ok_count += 1
        print(f"  seed{seed}: {msg1} | {msg2}", flush=True)
        gc.collect()
    print(f"[prep_split] DONE: {ok_count}/{args.n_seeds} seeds, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
