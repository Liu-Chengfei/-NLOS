#!/usr/bin/env python
"""逐个跑 22 个缺失 unit, 每步用后台 Bash 启动 subprocess 并轮询 ckpt."""
import subprocess, sys, time, os
from pathlib import Path

METHODS = ["lstm", "transformer", "liquid"]
SCRIPT = {"lstm": "05_train_lstm.py", "transformer": "07_train_transformer.py", "liquid": "06_train_liquid.py"}
SEQ_BATCH = {"lstm": 4, "transformer": 4, "liquid": 1}
PROCESSED_ROOT = "E:/异步高NLOS/data/processed/sim_e9_main"
ROOT = "E:/异步高NLOS"

def get_missing():
    missing = []
    for s in range(10):
        for m in METHODS:
            p = Path(f"E:/异步高NLOS/checkpoints/{m}_seed{s}/checkpoints/{m}_ekf_best_checkpoint.pt")
            if not (p.exists() and p.stat().st_size > 1000):
                missing.append((m, s))
    return missing

def is_seed_done(method, seed):
    p = Path(f"E:/异步高NLOS/checkpoints/{method}_seed{seed}/checkpoints/{method}_ekf_best_checkpoint.pt")
    return p.exists() and p.stat().st_size > 1000

def get_last_epoch(method, seed):
    log = Path(f"E:/异步高NLOS/logs/{method}_seed{seed}.log")
    if not log.exists():
        return None
    txt = log.read_text(errors="ignore")
    for l in reversed(txt.splitlines()):
        if "] 第 " in l and "/80" in l:
            try:
                return int(l.split("] 第 ")[1].split("/")[0].strip())
            except:
                pass
    return None

def poll_until_done(method, seed, timeout=1800):
    """Poll for ckpt or process death, up to timeout seconds."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_seed_done(method, seed):
            return True
        # Check if training log is growing (process alive)
        log = Path(f"E:/异步高NLOS/logs/{method}_seed{seed}.log")
        if log.exists():
            txt = log.read_text(errors="ignore")
            lines = txt.splitlines()
            if lines:
                last = lines[-1][:80]
                print(f"  [{method} s{seed}] ep progress: {last}", flush=True)
        time.sleep(60)
    return is_seed_done(method, seed)

def launch_one(method, seed, epochs):
    pr = Path(PROCESSED_ROOT)
    split_file = pr / f"seed{seed}" / "train_ids.txt"
    out_dir = Path(f"E:/异步高NLOS/checkpoints/{method}_seed{seed}")
    log_file = Path(f"E:/异步高NLOS/logs/{method}_seed{seed}.log")
    cmd = [
        sys.executable, "-u",
        f"{ROOT}/scripts/{SCRIPT[method]}",
        "--dataset-name", "sim",
        "--events-root", str(pr / f"seed{seed}"),
        "--raw-root", f"{ROOT}/data/raw/sim_e9_main/seed{seed}",
        "--split-ids-file", str(split_file),
        "--train-split-ids-file", str(split_file),
        "--val-split-ids-file", str(split_file.parent / "val_ids.txt"),
        "--epochs", str(epochs),
        "--seq-batch-size", str(SEQ_BATCH[method]),
        "--output-root", str(out_dir),
        "--disable-scene-leak-check",
    ]
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    # Clean old log
    if log_file.exists():
        log_file.unlink()
    with open(log_file, "w") as lf:
        pass
    # Run with nohup in bash (Git Bash supports nohup properly)
    bash_cmd = f"cd E:/异步高NLOS && nohup .venv-gpu/Scripts/python.exe -u {cmd[1]} {cmd[2]} " + " ".join(f'{k} {v}' for k, v in [
        ("--dataset-name", "sim"),
        ("--events-root", str(pr / f"seed{seed}")),
        ("--raw-root", f"{ROOT}/data/raw/sim_e9_main/seed{seed}"),
        ("--split-ids-file", str(split_file)),
        ("--train-split-ids-file", str(split_file)),
        ("--val-split-ids-file", str(split_file.parent / "val_ids.txt")),
        ("--epochs", str(epochs)),
        ("--seq-batch-size", str(SEQ_BATCH[method])),
        ("--output-root", str(out_dir)),
        ("--disable-scene-leak-check"),
    ]) + f" > logs/{method}_seed{seed}.log 2>&1 &"
    return bash_cmd

def main():
    epochs = 80
    missing = get_missing()
    total = len(missing)
    print(f"[batch_train] {total} missing units | epochs={epochs}", flush=True)
    done = 0
    for method, seed in missing:
        print(f"\n[{done}/{total}] Launching {method} seed{seed}...", flush=True)
        # Check if already done (race condition)
        if is_seed_done(method, seed):
            print(f"  already done, skip", flush=True)
            done += 1
            continue
        # Launch with nohup in a new bash subprocess
        pr = Path(PROCESSED_ROOT)
        split_file = pr / f"seed{seed}" / "train_ids.txt"
        out_dir = Path(f"E:/异步高NLOS/checkpoints/{method}_seed{seed}")
        log_file = Path(f"E:/异步高NLOS/logs/{method}_seed{seed}.log")
        if log_file.exists():
            log_file.unlink()
        cmd_list = [
            sys.executable, "-u",
            f"E:/异步高NLOS/scripts/{SCRIPT[method]}",
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
        # Run with subprocess.Popen (background)
        proc = subprocess.Popen(
            cmd_list,
            env=env,
            cwd="E:/异步高NLOS",
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
        )
        print(f"  PID={proc.pid}", flush=True)
        # Poll until done (30 min timeout per unit)
        finished = poll_until_done(method, seed, timeout=1800)
        if finished:
            print(f"  DONE: {method} seed{seed}", flush=True)
            done += 1
        else:
            print(f"  TIMEOUT: {method} seed{seed} ({is_seed_done(method, seed)})", flush=True)
            # Check log for errors
            if log_file.exists():
                txt = log_file.read_text(errors="ignore")
                if txt:
                    lines = txt.splitlines()
                    print(f"  Last log lines: {lines[-3:]}", flush=True)
    print(f"\n[batch_train] ALL DONE: {done}/{total}", flush=True)

if __name__ == "__main__":
    main()
