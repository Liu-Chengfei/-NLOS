"""Per-seed manifest.json 生成器（v2 阶段 RZ-A A2 修复）。

从 data/raw/sim_e9_main/seedN/ 下每条序列的 sim_meta.json 汇聚：
  - axes_override.A → async_level (A2/A3)
  - axes_override.N → nlos_level (N2/N3)
  - nlos_rho / nlos_mu_m / nlos_sigma_m
  - sequences list
并补顶层 missing_rate / mean_gap_s / max_gap_s 统计（M1 协议）。

输出：seedN/manifest.json （S9 校验第一个读它）。
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

DATA_ROOT = Path("E:/异步高NLOS/data/raw/sim_e9_main")


def _gather_seq_meta(seq_dir: Path) -> dict[str, Any] | None:
    """从 sim_meta.json + gt.json 读取一条序列的元信息。"""
    meta_path = seq_dir / "sim_meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    axes = meta.get("axes_override") or {}
    # 兼容 axes_override 是 list-of-tuples 或 dict
    if isinstance(axes, list):
        axes = {k: v for k, v in axes}
    return {
        "seq_id": seq_dir.name,
        "async_level": axes.get("A", ""),
        "nlos_level": axes.get("N", ""),
        "visual_level": axes.get("V", ""),
        "geometry_level": axes.get("G", ""),
        "k_level": axes.get("K", ""),
        "nlos_rho": float(meta.get("nlos_rho", 0.0)),
        "nlos_mu_m": float(meta.get("nlos_mu_m", 0.0)),
        "nlos_sigma_m": float(meta.get("nlos_sigma_m", 0.0)),
        "v_axis_present": "V" in axes,
    }


def _gather_missing_stats(seed_dir: Path) -> dict[str, float]:
    """统计 UWB missing rate（遍历每条序列的 uwb.json）。"""
    n_total = 0
    n_missing = 0
    gaps_s: list[float] = []
    for seq_dir in sorted(seed_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        uwb_path = seq_dir / "uwb.json"
        if not uwb_path.is_file():
            continue
        try:
            rows = json.loads(uwb_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not rows:
            continue
        ts = [r.get("timestamp", r.get("t", 0.0)) for r in rows]
        for i, r in enumerate(rows):
            n_total += 1
            if not r.get("valid", r.get("uwb_valid", True)):
                n_missing += 1
        for i in range(1, len(ts)):
            dt = ts[i] - ts[i - 1]
            if dt > 0.5:  # 缺测 > 0.5s
                gaps_s.append(dt)
    if n_total == 0:
        return {"missing_rate": 0.0, "mean_gap_s": 0.0, "max_gap_s": 0.0}
    return {
        "missing_rate": n_missing / n_total,
        "mean_gap_s": float(sum(gaps_s) / len(gaps_s)) if gaps_s else 0.0,
        "max_gap_s": float(max(gaps_s)) if gaps_s else 0.0,
    }


def build_manifest_for_seed(seed_dir: Path) -> dict[str, Any]:
    seqs: list[dict[str, Any]] = []
    for seq_dir in sorted(seed_dir.iterdir()):
        if not seq_dir.is_dir() or seq_dir.name.startswith("."):
            continue
        m = _gather_seq_meta(seq_dir)
        if m is not None:
            seqs.append(m)
    miss = _gather_missing_stats(seed_dir)
    return {
        "seed_id": seed_dir.name,
        "data_root": str(seed_dir.resolve()),
        "n_sequences": len(seqs),
        "missing_rate": miss["missing_rate"],
        "mean_gap_s": miss["mean_gap_s"],
        "max_gap_s": miss["max_gap_s"],
        "sequences": seqs,
    }


def main():
    if not DATA_ROOT.is_dir():
        print(f"[ERR] data_root not found: {DATA_ROOT}")
        return 1
    written = 0
    for seed_dir in sorted(DATA_ROOT.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        manifest = build_manifest_for_seed(seed_dir)
        out = seed_dir / "manifest.json"
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        n_seqs = manifest["n_sequences"]
        comp = {"A2N2": 0, "A2N3": 0, "A3N2": 0, "A3N3": 0}
        for s in manifest["sequences"]:
            k = f"{s.get('async_level', '')}{s.get('nlos_level', '')}"
            if k in comp:
                comp[k] += 1
        print(f"  {seed_dir.name}: {n_seqs} seqs, comp={comp}, miss_rate={manifest['missing_rate']:.3f}")
        written += 1
    print(f"\n[OK] wrote {written} manifest.json files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
