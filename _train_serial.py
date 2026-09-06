#!/usr/bin/env python
"""严格串行训练剩余 11 unit: 每次只跑 1 个, nohup 保护 subprocess."""
import subprocess, sys, time, os
from pathlib import Path

# 剩余 11 unit
MISSING = [
    ("transformer", 3),
    ("liquid",      6),
    ("lstm",        7),
    ("transformer", 7),
    ("liquid",      7),
    ("lstm",        8),
    ("transformer", 8),
    ("liquid",      8),
    ("lstm",        9),
    ("transformer", 9),
    ("liquid",      9),
]

SCRIPT = {"lstm": "05_train_lstm.py", "transformer": "07_train_transformer.py", "liquid": "06_train_liquid.py"}
SEQ_BATCH = {"lstm": 4, "transformer": 4, "liquid": 1}
PROCESSED_ROOT = "E:/异步高NLOS/data/processed/sim_e9_main"
ROOT = "E:/异步高NLOS"
EPOCHS = 80

def is_done(method, seed):
    p = Path(f"E:/异步高NLOS/checkpoints/{method}_seed{seed}/checkpoints/{method}_ekf_best_checkpoint.pt")
    return p.exists() and p.stat().st_size > 1000

def get_last_epoch(method, seed):
    log = Path(f"E:/异步高NLOS/logs/{method}_seed{seed}.log")
    if not log.exists():
        return None
    lines = log.read_text(errors="ignore").splitlines()
    for l in reversed(lines):
        if "] 第 " in l and "/80" in l:
            try:
                return int(l.split("] 第 ")[1].split("/")[0].strip())
            except:
                pass
    return None

def wait_for_ckpt(method, seed, poll_interval=120, max_wait=3600):
    """轮询 ckpt 是否出现 (训练进程已通过 nohup 启动)."""
    t0 = time.time()
    last_report = 0
    while time.time() - t0 < max_wait:
        if is_done(method, seed):
            return True
        if time.time() - last_report > 60:
            ep = get_last_epoch(method, seed)
            status = f"ep {ep}" if ep else "?"
            print(f"  [{method}_s{seed}] waiting... {status}", flush=True)
            last_report = time.time()
        time.sleep(poll_interval)
    return is_done(method, seed)

def launch_trainer(method, seed):
    """启动单个训练器 (不等待完成)."""
    pr = Path(PROCESSED_ROOT)
    out_dir = Path(f"E:/异步高NLOS/checkpoints/{method}_seed{seed}")
    log_file = Path(f"E:/异步高NLOS/logs/{method}_seed{seed}.log")
    if log_file.exists():
        log_file.unlink()
    cmd = [
        str(Path(ROOT) / ".venv-gpu/Scripts/python.exe"),
        "-u",
        str(Path(ROOT) / f"scripts/{SCRIPT[method]}"),
        "--dataset-name", "sim",
        "--events-root", str(pr / f"seed{seed}"),
        "--raw-root", f"{ROOT}/data/raw/sim_e9_main/seed{seed}",
        "--split-ids-file", str(pr / f"seed{seed}/train_ids.txt"),
        "--train-split-ids-file", str(pr / f"seed{seed}/train_ids.txt"),
        "--val-split-ids-file", str(pr / f"seed{seed}/val_ids.txt"),
        "--epochs", str(EPOCHS),
        "--seq-batch-size", str(SEQ_BATCH[method]),
        "--output-root", str(out_dir),
        "--disable-scene-leak-check",
    ]
    env = {**os.environ, "PYTHONHASHSEED": "0", "STAR_A95_MAX": "50.0"}
    with open(log_file, "w") as lf:
        pass
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=ROOT,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
    )
    return proc.pid

def main():
    total = len(MISSING)
    print(f"[serial] {total} missing units to train (strictly serial)", flush=True)
    for i, (method, seed) in enumerate(MISSING):
        if is_done(method, seed):
            print(f"[{i+1}/{total}] {method}_s{seed} already done, skip", flush=True)
            continue
        print(f"\n[{i+1}/{total}] Launching {method}_seed{seed}...", flush=True)
        pid = launch_trainer(method, seed)
        print(f"  PID={pid}", flush=True)
        # Wait for ckpt to appear (each seed ~40 min for LSTM, ~60 min for NN)
        max_wait = 4500  # 75 min per unit
        ok = wait_for_ckpt(method, seed, poll_interval=120, max_wait=max_wait)
        if ok:
            print(f"  DONE: {method}_seed{seed}", flush=True)
        else:
            ep = get_last_epoch(method, seed)
            print(f"  TIMEOUT: {method}_seed{seed} (last ep={ep}, is_done={is_done(method, seed)})", flush=True)
            # Print last log lines
            log = Path(f"E:/异步高NLOS/logs/{method}_seed{seed}.log")
            if log.exists():
                lines = log.read_text(errors="ignore").splitlines()
                print(f"  Last 5 log lines: {lines[-5:]}", flush=True)
    print(f"\n[serial] ALL DONE", flush=True)

if __name__ == "__main__":
    main()
