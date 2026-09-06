"""10-seed sim_e9_main 数据重新生成（v2 阶段 RZ-B）。

每个 seed 一次 materialize_sim_raw 调用（60 specs × 4 combo 各 25% × 150Hz IMU）。
输出到 data/raw/sim_e9_main/seed{N}/。
"""
from __future__ import annotations
import dataclasses
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("E:/异步高NLOS")
sys.path.insert(0, str(ROOT / "src"))

from liquidloc.dataio.sim_materializer import (
    SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS,
    SimNoiseSpec,
    materialize_sim_raw,
)

DATA_ROOT = ROOT / "data" / "raw" / "sim_e9_main"
N_SEEDS = 10
BASE_SEED = 20260605  # 区别 v2 协议基准种子


def main():
    if DATA_ROOT.is_dir() and any(DATA_ROOT.iterdir()):
        print(f"[regen] removing old {DATA_ROOT}")
        shutil.rmtree(DATA_ROOT)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for seed_idx in range(N_SEEDS):
        noise = SimNoiseSpec(base_seed=BASE_SEED + seed_idx * 7919)
        out = DATA_ROOT / f"seed{seed_idx}"
        # 跑前清理任何残留
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        report = materialize_sim_raw(
            out,
            sequence_specs=SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS,
            noise_spec=noise,
        )
        status = report.get("status", "unknown")
        # 序列状态字段可能是 "materialized"/"ok"/"failed"，宽容匹配
        n_ok = sum(1 for s in report.get("sequences", {}).values()
                    if isinstance(s, dict) and s.get("status") in ("ok", "materialized"))
        print(f"  seed{seed_idx}: status={status}, n_ok={n_ok}/{len(SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS)} ({time.time()-t1:.1f}s)")
    print(f"\n[regen] done {N_SEEDS} seeds in {time.time()-t0:.0f}s")
    print(f"  data: {DATA_ROOT}")


if __name__ == "__main__":
    main()
