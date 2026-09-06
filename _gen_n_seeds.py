#!/usr/bin/env python
"""生成 N seeds × 60 specs 的 sim 数据, 每个 seed 输出到独立目录。

采用 SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS 60 specs × N seeds。
每个 seed:
  1. dataclass.replace 每条 spec.seq_id 加 _seedN 后缀 → 让同 axes 不同 scene_id (防 §0.2 leak)
  2. materialize_sim_raw 传 noise_spec with different base_seed → 不同 noise realization
  3. 用空 tmp 目录绕过 "target must be empty" 检查后移到 seedN 子目录

执行:
  python _gen_n_seeds.py --n-seeds 10 --output-root data/raw/sim_e9_main
"""
import os, sys, io, contextlib, json, gc, time, dataclasses, tempfile, shutil
from pathlib import Path

ROOT = Path("E:/异步高NLOS")
sys.path.insert(0, str(ROOT / "src"))
os.environ["LIQUIDLOC_SUPPRESS_PRINT_DICT"] = "1"
os.environ.setdefault("STAR_A95_MAX", "30.0")  # 放宽 a95 上限 (因 5 点中心差分 + yaw_rate=9.5 合成峰值)

from liquidloc.dataio.sim_materializer import (
    SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS,
    materialize_sim_raw,
    SimNoiseSpec,
)

_BASE_SEED = 20260605


def gen_one_seed(seed_idx: int, output_root: Path) -> tuple[int, str]:
    """为单个 seed 复制 60 specs 并改名，生成到 seedN/。"""
    noise_seed = _BASE_SEED + seed_idx * 7919
    noise_spec = SimNoiseSpec(base_seed=noise_seed)
    specs_for_this_seed = []
    for s in SIM_E9_ONLY_COMPACT_SEQUENCE_SPECS:
        new_seq_id = f"{s.seq_id}_seed{seed_idx}"
        # 保留 base_seq_id (fixture 限制只能 mini_seq/mini_seq_02/mini_seq_03)
        s_new = dataclasses.replace(s, seq_id=new_seq_id)
        specs_for_this_seed.append(s_new)

    out_dir = output_root / f"seed{seed_idx}"
    if out_dir.exists() and len(list(out_dir.iterdir())) >= len(specs_for_this_seed):
        return len(specs_for_this_seed), "skip (exists)"

    tmp = Path(tempfile.mkdtemp(prefix=f"sim_seed{seed_idx}_"))
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            reports = materialize_sim_raw(
                tmp,
                sequence_specs=tuple(specs_for_this_seed),
                noise_spec=noise_spec,
            )
        for child in tmp.iterdir():
            target = out_dir / child.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(child), str(target))
        return len(specs_for_this_seed), "ok"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def new_base_seq_id_for(orig: str, seed_idx: int) -> str:  # noqa: ARG001 保留兼容
    """保留兼容 (不再使用, base_seq_id 受 fixture 限制, 不能改)."""
    return f"{orig}_s{seed_idx}"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--output-root", default="E:/异步高NLOS/data/raw/sim_e9_main")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[gen_n_seeds] n_seeds={args.n_seeds}, output={output_root}", flush=True)

    t0 = time.time()
    total_seqs = 0
    for seed in range(args.n_seeds):
        n, status = gen_one_seed(seed, output_root)
        total_seqs += n
        print(f"  seed{seed}: {n} specs {status}  base_seed={_BASE_SEED + seed * 7919}", flush=True)
        gc.collect()
    elapsed = time.time() - t0
    print(f"[gen_n_seeds] DONE: {total_seqs} specs in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
