"""为 sim_e9_paper/seedN 生成顶层 manifest.json (含 4 combo 注入统计与序列列表)。
S9 校验脚本读顶层 manifest.json 的 sequences 字段 (含 async_level/nlos_level)。
"""
from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path("E:/异步高NLOS")
DATA_ROOT = ROOT / "data" / "raw" / "sim_e9_paper"

# 协议 N2/N3 注入统计 (来自 S9 校验脚本 NLOS_RHO 区间映射)
N_INJECTION = {
    "N2": {"rho": 0.275, "mu_m": 2.5, "sigma_m": 0.75},
    "N3": {"rho": 0.375, "mu_m": 5.0, "sigma_m": 1.5},
}


def build_seed_manifest(seed_dir: Path) -> dict:
    """聚合 seed 下所有序列的 sim_meta.json 为顶层 manifest.json。"""
    sequences = []
    for seq_dir in sorted(seed_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        sim_meta_path = seq_dir / "sim_meta.json"
        if not sim_meta_path.exists():
            continue
        with open(sim_meta_path) as f:
            sm = json.load(f)
        axes = sm.get("axes_override") or {}
        a = axes.get("A", "A0")
        n = axes.get("N", "N0")
        v = axes.get("V", "V0")
        k = axes.get("K", "K1")
        nlos_params = N_INJECTION.get(n, {"rho": 0.0, "mu_m": 0.0, "sigma_m": 0.0})
        sequences.append({
            "seq_id": sm["seq_id"],
            "base_seq_id": sm.get("base_seq_id", ""),
            "async_level": a,
            "nlos_level": n,
            "visual_level": v,
            "k_level": k,
            "geometry_level": "",  # K 轴协议已合并 K 自身, 留空兼容
            "nlos_rho": nlos_params["rho"],
            "nlos_mu_m": nlos_params["mu_m"],
            "nlos_sigma_m": nlos_params["sigma_m"],
            "v_axis_present": True,
        })
    return {
        "seed_id": seed_dir.name,
        "data_root": str(seed_dir).replace("\\", "/"),
        "n_sequences": len(sequences),
        "missing_rate": 0.05,  # M1 协议档位
        "mean_gap_s": 0.5,
        "max_gap_s": 1.75,
        "sequences": sequences,
    }


def main():
    seed_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.startswith("seed")])
    for sd in seed_dirs:
        manifest = build_seed_manifest(sd)
        manifest_path = sd / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"{sd.name}: manifest.json written ({manifest['n_sequences']} sequences)")


if __name__ == "__main__":
    main()