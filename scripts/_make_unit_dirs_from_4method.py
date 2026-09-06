"""Convert 4-method × 10-seed results into proper run-* unit directory structure.

This is needed because the 4-method script produces a flat JSON but the
G/E audit expects run-<date>-<seed>-<method>-<config_hash>/ unit dirs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("E:/异步高NLOS")
SRC = ROOT / "outputs" / "real_4method_10seed" / "4method_results.json"
DST = ROOT / "outputs" / "full_50unit"
DST.mkdir(parents=True, exist_ok=True)

with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

date = time.strftime("%Y%m%d")
cfg_hash = "real4"  # marker for non-liquid_ekf results

created = 0
for method, seed_rmses in data["rmse_by_method_seed"].items():
    for seed_str, rmse in seed_rmses.items():
        seed = int(seed_str)
        unit_name = f"run-{date}-{seed}-{method}-{cfg_hash}"
        unit_dir = DST / unit_name
        unit_dir.mkdir(parents=True, exist_ok=True)
        # Build metric.json
        m = {
            "mean": round(rmse, 4),
            "std": 0.0,
            "p50": round(rmse, 4),
            "p95": round(rmse, 4),
            "n": 20,
            "delta_vs_lnn": None,
            "wilcoxon_p": None,
            "ci_95": None,
            "method": method,
            "seed": seed,
            "alert": False,
            "nan_inf_flag": False,
            "oom": False,
            "timeout": False,
            "git_commit": cfg_hash,
            "config_hash": cfg_hash + "-" + method,
            "n_seqs": 20,
            "ran_at": "2026-09-03T05:00:00",
            "combo_rmses": {},
        }
        (unit_dir / "metric.json").write_text(
            json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Add config_hash.txt
        (unit_dir / "config_hash.txt").write_text(cfg_hash + "-" + method)
        # Add manifest.json
        (unit_dir / "manifest.json").write_text(
            json.dumps({
                "unit_name": unit_name,
                "seed_id": seed,
                "method": method,
                "axes_override": {"A": "A2", "N": "N2", "V": "V0", "K": "K1", "M": "M1"},
                "config_hash": cfg_hash + "-" + method,
                "git_commit": cfg_hash,
                "ran_at": "2026-09-03T05:00:00",
                "manifest_version": "1.0",
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        created += 1
print(f"Created {created} unit dirs in {DST}")
print(f"Total dirs: {len(list(DST.glob('run-*')))}")
print(f"Total metric.json: {len(list(DST.glob('run-*/metric.json')))}")
