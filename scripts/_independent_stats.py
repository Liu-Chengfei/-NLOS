"""独立复算 stats 表（E-3 dual review 用的 stat table）。

按手册 Part 3 D26/D27/D28 要求：
- p 值：5 seed × 4 non-LNN 方法 vs LNN 配对 Wilcoxon
- 效应量：Cohen's d
- 95%CI：成对差 ±1.96·SE

独立于 _run_25unit.py 内嵌的 stats 路径（直接读 unit metric.json 重算）。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
UNIT_ROOT = ROOT / "outputs" / "full_25unit"
OUT = ROOT / "outputs" / "audit" / "independent_stats.json"


def _paired_wilcoxon(diffs: list[float]) -> tuple[float, float]:
    """配对 Wilcoxon 精确检验（无 scipy 时 fallback normal approx）。"""
    if not diffs or len(diffs) < 2:
        return (0.0, 1.0)
    try:
        from scipy.stats import wilcoxon
        _, p = wilcoxon(diffs, zero_method="wilcox", correction=False, alternative="two-sided")
        z = 0.0
    except Exception:
        # normal approx
        nz = [d for d in diffs if d != 0]
        if not nz:
            return (0.0, 1.0)
        mean_d = sum(nz) / len(nz)
        std_d = math.sqrt(sum((d - mean_d) ** 2 for d in nz) / max(1, len(nz) - 1))
        from math import erf, sqrt
        z = mean_d / max(std_d, 1e-6) / sqrt(len(nz))
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return (z, float(p))


def _cohens_d(diffs: list[float]) -> float:
    if not diffs or len(diffs) < 2:
        return 0.0
    mean_d = sum(diffs) / len(diffs)
    std_d = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / (len(diffs) - 1))
    return mean_d / std_d if std_d > 1e-6 else 0.0


def _ci95(diffs: list[float]) -> tuple[float, float]:
    if not diffs:
        return (0.0, 0.0)
    mean_d = sum(diffs) / len(diffs)
    std_d = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / max(1, len(diffs) - 1))
    se = std_d / math.sqrt(len(diffs))
    return (mean_d - 1.96 * se, mean_d + 1.96 * se)


def main() -> int:
    units = sorted(UNIT_ROOT.glob("run-*"))
    by_method: dict[str, list[dict]] = {m: [] for m in ["ekf", "robust_ekf", "lstm_ekf", "transformer_ekf", "liquid_ekf"]}
    for d in units:
        mfile = d / "metric.json"
        if not mfile.is_file():
            continue
        try:
            m = json.loads(mfile.read_text(encoding="utf-8"))
        except Exception:
            continue
        parts = d.name.split("-")
        if len(parts) < 5:
            continue
        try:
            seed_id = int(parts[2])
        except ValueError:
            continue
        method = parts[3]
        if method not in by_method:
            continue
        by_method[method].append({
            "seed": seed_id,
            "unit": d.name,
            "mean": m.get("mean"),
            "p95": m.get("p95"),
            "combo_rmses": m.get("combo_rmses"),  # may be None
        })

    # Build paired (seed → method) data, preserving combo_rmses
    paired: dict[str, dict[int, dict]] = {}
    for m, entries in by_method.items():
        paired[m] = {e["seed"]: e for e in entries}
    # 5 seed paired table
    seeds = sorted({e["seed"] for v in by_method.values() for e in v}) if by_method["ekf"] else []

    lnn_rmse = [paired["liquid_ekf"][s]["mean"] for s in seeds if s in paired["liquid_ekf"]]
    table: dict[str, Any] = {
        "n_seeds": len(seeds),
        "methods": list(by_method.keys()),
        "per_method_mean": {
            m: (sum(by_method[m][k]["mean"] for k in range(len(by_method[m]))) / max(1, len(by_method[m])))
            for m in by_method
        },
        "paired_wilcoxon_vs_lnn": {},
        "cohens_d_vs_lnn": {},
        "ci95_difference_vs_lnn": {},
        "ranking": sorted(
            by_method.keys(),
            key=lambda m: (
                sum(by_method[m][k]["mean"] for k in range(len(by_method[m])))
                / max(1, len(by_method[m]))
            ),
        ),
    }
    for other in ["ekf", "robust_ekf", "lstm_ekf", "transformer_ekf"]:
        if other == "liquid_ekf":
            continue
        if not (set(paired[other].keys()) >= set(seeds) and set(paired["liquid_ekf"].keys()) >= set(seeds)):
            table["paired_wilcoxon_vs_lnn"][other] = {"error": "missing seeds"}
            continue
        # Seed-level (n=5) for Cohen's d and CI95
        diffs_seed = [paired[other][s]["mean"] - paired["liquid_ekf"][s]["mean"] for s in seeds]
        # Per-sequence (n=100) Wilcoxon: pair per-combo seq rmses
        seq_diffs: list[float] = []
        for s in seeds:
            other_cr = paired[other].get(s, {}).get("combo_rmses") or {}
            lnn_cr = paired["liquid_ekf"].get(s, {}).get("combo_rmses") or {}
            for ck in other_cr:
                other_seq = other_cr[ck].get("rmses", [])
                lnn_seq = lnn_cr.get(ck, {}).get("rmses", [])
                if other_seq and lnn_seq and len(other_seq) == len(lnn_seq):
                    for o_r, l_r in zip(other_seq, lnn_seq):
                        if not (math.isnan(o_r) or math.isnan(l_r)):
                            seq_diffs.append(o_r - l_r)
        # Prefer per-sequence (n=100); fall back to seed-level (n=5)
        diffs = seq_diffs if len(seq_diffs) >= 5 else diffs_seed
        _, p = _paired_wilcoxon(diffs)
        d = _cohens_d(diffs_seed)  # Cohen's d uses seed-level means
        ci = _ci95(diffs_seed)      # 95% CI uses seed-level means
        table["paired_wilcoxon_vs_lnn"][other] = {
            "n_pairs": len(diffs),
            "n_pairs_seq": len(seq_diffs),
            "mean_diff": round(sum(diffs) / len(diffs), 4),
            "wilcoxon_p": round(p, 6),
            "cohens_d": round(d, 4),
            "ci_95": [round(ci[0], 4), round(ci[1], 4)],
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[independent_stats] {OUT} (n_seeds={len(seeds)})")
    for other, v in table["paired_wilcoxon_vs_lnn"].items():
        print(f"  LNN vs {other}: p={v.get('wilcoxon_p', 'N/A')} d={v.get('cohens_d', 'N/A')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
