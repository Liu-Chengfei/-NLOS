"""Run 4 working methods (EKF, robust_ekf, LSTM, Transformer) on all 10 seeds sequentially.

Liquid_ekf is skipped because 1-epoch LNN training is insufficient.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Disable print_dict to reduce log noise
import os
os.environ.setdefault("LIQUIDLOC_SUPPRESS_PRINT_DICT", "1")

# Reuse the functions from _run_25unit
sys.path.insert(0, str(ROOT / "scripts"))
from _run_25unit import _real_predict_sequence, _real_nn_predict_sequence, _load_gt

DATA_ROOT = ROOT / "data" / "raw" / "sim_e9_10seed_50unit"
OUT_ROOT = ROOT / "outputs" / "real_4method_10seed"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

METHODS = [("ekf", "ekf"), ("robust_ekf", "ekf"),
           ("lstm_ekf", "nn"), ("transformer_ekf", "nn")]
SEEDS = list(range(10))
USE_NN = {"ekf": False, "robust_ekf": False, "lstm_ekf": True, "transformer_ekf": True}


def main():
    all_results: dict[str, dict[int, float]] = {m: {} for m, _ in METHODS}
    per_method_total = {m: 0.0 for m, _ in METHODS}

    for method, backend in METHODS:
        print(f"\n=== {method} ===", flush=True)
        for seed in SEEDS:
            seed_dir = DATA_ROOT / f"seed{seed}"
            seq_dirs = sorted(d for d in seed_dir.iterdir() if d.is_dir())
            t0 = time.time()
            rmses = []
            for seq_dir in seq_dirs:
                if USE_NN[method]:
                    r = _real_nn_predict_sequence(seq_dir, method)
                else:
                    r = _real_predict_sequence(seq_dir, method)
                rmse = r.get("rmse")
                if rmse is not None and not math.isnan(rmse):
                    rmses.append(rmse)
            elapsed = time.time() - t0
            if rmses:
                mean_rmse = sum(rmses) / len(rmses)
                all_results[method][seed] = mean_rmse
                per_method_total[method] += mean_rmse
                print(f"  seed={seed} mean_rmse={mean_rmse:.4f}m "
                      f"n={len(rmses)}/20 ({elapsed:.1f}s)", flush=True)
            else:
                print(f"  seed={seed} no valid rmse ({elapsed:.1f}s)", flush=True)

    print("\n=== Summary (10 seeds × 4 methods) ===", flush=True)
    for m, _ in METHODS:
        vals = list(all_results[m].values())
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {m:20s} avg_rmse={avg:.4f}m n_seeds={len(vals)}/10")

    # Save report
    out = {
        "n_seeds": len(SEEDS),
        "methods": [m for m, _ in METHODS],
        "rmse_by_method_seed": {m: {str(s): r for s, r in all_results[m].items()} for m, _ in METHODS},
        "n_seeds_completed": {m: len(all_results[m]) for m, _ in METHODS},
    }
    out_path = OUT_ROOT / "4method_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
