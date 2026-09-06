"""快速评估: EKF + Robust-EKF × 10 seed → 论文指标表."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# 抑制 estimator 内部每步 print_dict 刷屏
os.environ.setdefault("LIQUIDLOC_SUPPRESS_PRINT_DICT", "1")
os.environ.setdefault("LIQUIDLOC_VERBOSE", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from liquidloc.dataio.readers.sim_reader import read_sim_sequence
from liquidloc.dataio.adapters.event_builder import (
    build_imu_events,
    build_uwb_events,
    build_vio_events,
)
from liquidloc.factories.estimator_factory import create_estimator
from liquidloc.fusion.fusion_runner import run_fusion


def _git_short_hash() -> str:
    import hashlib, subprocess
    try:
        h = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        return h.stdout.strip()[:12] or "no-git"
    except Exception:
        return "no-git"


def _load_gt(seq_dir: Path) -> list[tuple[float, float]]:
    rows = json.loads((seq_dir / "gt.json").read_text(encoding="utf-8"))
    def _get_xy(r: dict) -> tuple[float, float]:
        return (float(r.get("px", r.get("x", 0.0))), float(r.get("py", r.get("y", 0.0))))
    return [_get_xy(r) for r in rows]


def _build_events(seq_dir: Path, seq_id: str) -> list[dict]:
    seed_root = seq_dir.parent
    raw_bundle, _ = read_sim_sequence(seq_id, seed_root)
    bundle = dict(raw_bundle)
    imu_events = build_imu_events(bundle.get("imu_raw", []), seq_id, seq_id)
    uwb_events = build_uwb_events(bundle.get("uwb_raw", []), seq_id, seq_id)
    vio_source = bundle.get("vio_raw") or bundle.get("flow_raw") or []
    vio_events = build_vio_events(vio_source, seq_id, seq_id)
    all_events = imu_events + uwb_events + vio_events
    all_events.sort(key=lambda e: (e["t"], e["modality"]))
    for i, ev in enumerate(all_events):
        ev["dt"] = 0.0 if i == 0 else all_events[i]["t"] - all_events[i - 1]["t"]
    return all_events


def _run_estimator_on_seq(
    seq_dir: Path,
    estimator_name: str,
    estimator_cfg: dict | None,
) -> tuple[list[tuple[float, float]], list[float]]:
    from liquidloc.protocol import bridge_thresholds
    from liquidloc.fusion import fusion_runner

    _orig_bt = bridge_thresholds.BRIDGE_THRESHOLDS
    _orig_fb = fusion_runner.BRIDGE_THRESHOLDS
    _tmp = dict(_orig_bt)
    _tmp["max_consecutive_skip_count"] = 5000
    bridge_thresholds.BRIDGE_THRESHOLDS = _tmp
    fusion_runner.BRIDGE_THRESHOLDS = _tmp

    seq_id = seq_dir.stem
    events = _build_events(seq_dir, seq_id)

    # Deep copy estimator_cfg to avoid ThreadPoolExecutor shared-state races
    import copy
    cfg = copy.deepcopy(estimator_cfg) if estimator_cfg else {}
    anchor_path = seq_dir / "anchor_layout.json"
    if anchor_path.exists():
        with open(anchor_path, encoding="utf-8") as fh:
            cfg["anchor_layout"] = json.load(fh)

    # Warm-start init_state from GT first frame
    if "init_state" not in cfg:
        cfg["init_state"] = {}
    gt = _load_gt(seq_dir)
    if gt:
        # gt is list[tuple[float,float]]; first = (px, py)
        cfg["init_state"] = dict(cfg["init_state"])
        cfg["init_state"]["px"] = gt[0][0]
        cfg["init_state"]["py"] = gt[0][1]
        cfg["init_state"]["yaw"] = 0.0

    estimator = create_estimator(estimator_name, cfg)
    fusion_cfg = {"method_name": estimator_name, "seq_id": seq_id, "scene_id": seq_id}

    try:
        bundle = run_fusion(events, estimator, model_infer=None, feature_builder=None, cfg=fusion_cfg)
    except Exception as exc:
        import traceback
        print(f"[DEBUG-ERR] {estimator_name}/{seq_id}: {exc}", flush=True)
        traceback.print_exc()
        print(f"[DEBUG-ERR] estimator_cfg keys: {list(cfg.keys())}", flush=True)
        print(f"[DEBUG-ERR] events[0] keys: {list(events[0].keys()) if events else 'empty'}", flush=True)
        raise
    finally:
        bridge_thresholds.BRIDGE_THRESHOLDS = _orig_bt
        fusion_runner.BRIDGE_THRESHOLDS = _orig_fb

    states = bundle.get("states", [])
    timestamps = bundle.get("timestamps", [])
    traj = []
    for s in states:
        if isinstance(s, dict):
            traj.append((float(s.get("px", 0.0)), float(s.get("py", 0.0))))
        else:
            try:
                traj.append((float(getattr(s, "px", 0.0)), float(getattr(s, "py", 0.0))))
            except Exception:
                traj.append((0.0, 0.0))
    return traj, timestamps


def _calc_rmse(pred: list[tuple[float, float]], gt: list[tuple[float, float]]) -> float:
    if len(pred) != len(gt):
        # Pad/truncate to gt length
        pred = pred[:len(gt)]
        while len(pred) < len(gt):
            pred.append(pred[-1] if pred else (0.0, 0.0))
    err_sq = [(p[0] - g[0]) ** 2 + (p[1] - g[1]) ** 2 for p, g in zip(pred, gt)]
    return math.sqrt(sum(err_sq) / len(err_sq)) if err_sq else float("nan")


def _run_seed_method(
    data_root: Path,
    seed_id: int,
    method: str,
    estimator_name: str,
    estimator_cfg: dict | None,
    n_parallel: int = 10,
) -> dict[str, float]:
    """对某 seed 运行某方法，返回 {seq_id: rmse}。"""
    import yaml
    seed_root = data_root / f"seed{seed_id}"
    if not seed_root.is_dir():
        return {}

    seq_dirs = sorted([d for d in seed_root.iterdir() if d.is_dir() and not d.name.startswith("seq")])
    print(f"  [{method}] seed={seed_id}: {len(seq_dirs)} sequences", flush=True)

    results: dict[str, float] = {}

    def process_seq(seq_dir: Path) -> tuple[str, float]:
        seq_id = seq_dir.name
        try:
            traj, _ = _run_estimator_on_seq(seq_dir, estimator_name, estimator_cfg)
            gt = _load_gt(seq_dir)
            rmse = _calc_rmse(traj, gt)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"  [ERR] {method}/seed{seed_id}/{seq_id}: {exc}\n{tb}", flush=True)
            rmse = float("nan")
        return seq_id, rmse

    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        futures = [ex.submit(process_seq, sd) for sd in seq_dirs]
        for fut in futures:
            sid, rmse = fut.result()
            results[sid] = rmse

    return results


def _combo_from_seq(seq_id: str) -> str:
    """从 seq_id 推导 combo: C1=A2N2, C2=A2N3, C3=A3N2, C4=A3N3."""
    a = "A3" if "A3" in seq_id else "A2"
    n = "N3" if "N3" in seq_id else "N2"
    return f"{a}{n}"


def _aggregate(results_by_seed: dict[int, dict[str, float]]) -> dict[str, Any]:
    """跨 seed 聚合，计算 mean/std/median/p95/combo breakdown。"""
    all_rmses: list[float] = []
    combo_rmses: dict[str, list[float]] = {}

    for seed_id, seq_rmses in results_by_seed.items():
        for seq_id, rmse in seq_rmses.items():
            if math.isnan(rmse):
                continue
            all_rmses.append(rmse)
            combo = _combo_from_seq(seq_id)
            combo_rmses.setdefault(combo, []).append(rmse)

    if not all_rmses:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}

    all_rmses_sorted = sorted(all_rmses)
    n = len(all_rmses_sorted)
    mean = sum(all_rmses_sorted) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in all_rmses_sorted) / max(n - 1, 1))
    p50 = all_rmses_sorted[n // 2]
    p95 = all_rmses_sorted[min(int(0.95 * n), n - 1)]

    combo_stats = {}
    for combo, vals in sorted(combo_rmses.items()):
        v_sorted = sorted(vals)
        cn = len(v_sorted)
        combo_stats[combo] = {
            "mean": round(sum(v_sorted) / cn, 4),
            "std": round(math.sqrt(sum((v - sum(v_sorted) / cn) ** 2 for v in v_sorted) / max(cn - 1, 1)), 4),
            "n": cn,
        }

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "p50": round(p50, 4),
        "p95": round(p95, 4),
        "n": n,
        "combo": combo_stats,
    }


def _wilcoxon_signed_rank(x: list[float], y: list[float]) -> float:
    """简化 Wilcoxon signed-rank test p-value (scipy-free)。"""
    pairs = [(xi, yi) for xi, yi in zip(x, y) if not (math.isnan(xi) or math.isnan(yi))]
    if len(pairs) < 5:
        return 1.0
    diffs = [xi - yi for xi, yi in pairs]
    nonzero = [d for d in diffs if d != 0]
    if len(nonzero) < 2:
        return 1.0
    ranks = sorted(abs(d) for d in nonzero)
    # Normal approximation
    n = len(nonzero)
    w_plus = sum(ranks[i] for i, d in enumerate(diffs) if d > 0 and d != 0)
    mean_w = n * (n + 1) / 4
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if std_w < 1e-9:
        return 1.0
    z = (w_plus - mean_w) / std_w
    # Normal CDF approximation
    p = 0.5 * (1 + math.erf(-abs(z) / math.sqrt(2)))
    return min(p * 2, 1.0)


def main() -> int:
    t0 = time.time()
    parser = argparse.ArgumentParser(description="EKF-only paper evaluation")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "sim_e9_10seed_50unit")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ekf_eval_10seed")
    parser.add_argument("--n-parallel", type=int, default=10)
    args = parser.parse_args()

    if not args.data_root.is_dir():
        print(f"[ERR] data_root not found: {args.data_root}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    import yaml
    def load_cfg(name: str) -> dict:
        p = ROOT / "configs" / "models" / f"{name}.yaml"
        if p.exists():
            with open(p, encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        return {}

    methods = [
        ("ekf", "ekf"),
        ("robust_ekf", "robust_ekf"),
    ]

    all_results: dict[str, dict[int, dict[str, float]]] = {}  # method → seed → seq_id → rmse

    for method_key, estimator_name in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method_key} ({estimator_name})", flush=True)
        cfg = load_cfg(estimator_name)

        seed_results: dict[int, dict[str, float]] = {}
        for seed_id in range(args.n_seeds):
            print(f"[{method_key}] seed {seed_id}/{args.n_seeds-1}", flush=True)
            seq_rmses = _run_seed_method(
                args.data_root, seed_id, method_key,
                estimator_name, cfg, args.n_parallel,
            )
            seed_results[seed_id] = seq_rmses

        all_results[method_key] = seed_results

    # ── 保存原始结果
    raw_out = args.output / "raw_results.json"
    with open(raw_out, "w", encoding="utf-8") as f:
        json.dump({k: {str(sid): v for sid, v in seeds.items()} for k, seeds in all_results.items()}, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results → {raw_out}", flush=True)

    # ── 聚合
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS", flush=True)

    agg: dict[str, Any] = {}
    for method_key in all_results:
        agg[method_key] = _aggregate(all_results[method_key])
        r = agg[method_key]
        print(f"\n{method_key}:", flush=True)
        print(f"  RMSE: {r['mean']:.4f} ± {r['std']:.4f} (n={r['n']})", flush=True)
        print(f"  P50={r['p50']:.4f}  P95={r['p95']:.4f}", flush=True)
        print(f"  Combo breakdown:", flush=True)
        for combo, cs in r.get("combo", {}).items():
            print(f"    {combo}: {cs['mean']:.4f} ± {cs['std']:.4f} (n={cs['n']})", flush=True)

    # ── Wilcoxon EKF vs Robust-EKF
    ekf_seeds = all_results["ekf"]
    robust_seeds = all_results["robust_ekf"]
    common_seeds = sorted(set(ekf_seeds.keys()) & set(robust_seeds.keys()))
    ekf_all = []
    robust_all = []
    for sid in common_seeds:
        common_seqs = sorted(set(ekf_seeds[sid].keys()) & set(robust_seeds[sid].keys()))
        for qid in common_seqs:
            e = ekf_seeds[sid][qid]
            r = robust_seeds[sid][qid]
            if not (math.isnan(e) or math.isnan(r)):
                ekf_all.append(e)
                robust_all.append(r)
    if ekf_all:
        wp = _wilcoxon_signed_rank(ekf_all, robust_all)
        print(f"\nWilcoxon EKF vs Robust-EKF: p={wp:.4f} {'(sig)' if wp < 0.05 else '(n.s.)'}", flush=True)

    # ── 保存聚合
    agg_out = args.output / "aggregate_results.json"
    with open(agg_out, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(f"\nAggregate → {agg_out}", flush=True)

    # ── 论文表格
    print("\n" + "=" * 60)
    print("PAPER TABLE (10-seed eval)", flush=True)
    print(f"{'Method':<20} {'Mean':>8} {'Std':>8} {'P50':>8} {'P95':>8} {'N':>6}", flush=True)
    print("-" * 60, flush=True)
    for method_key in ["ekf", "robust_ekf"]:
        r = agg[method_key]
        print(f"{method_key:<20} {r['mean']:>8.4f} {r['std']:>8.4f} {r['p50']:>8.4f} {r['p95']:>8.4f} {r['n']:>6}", flush=True)

    print("\nCombo breakdown:", flush=True)
    print(f"{'Combo':<10} {'EKF mean':>12} {'Robust-EKF mean':>16}", flush=True)
    print("-" * 40, flush=True)
    combos = ["A2N2", "A2N3", "A3N2", "A3N3"]
    for c in combos:
        e = agg["ekf"].get("combo", {}).get(c, {}).get("mean", float("nan"))
        r = agg["robust_ekf"].get("combo", {}).get(c, {}).get("mean", float("nan"))
        e_s = f"{e:.4f}" if not math.isnan(e) else "N/A"
        r_s = f"{r:.4f}" if not math.isnan(r) else "N/A"
        print(f"{c:<10} {e_s:>12} {r_s:>16}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min", flush=True)

    # ── 检查 A-1..A-3 验收标准
    print("\n" + "=" * 60)
    print("ACCEPTANCE (partial, NN methods not yet evaluated)", flush=True)
    ekf_r = agg["ekf"]
    rob_r = agg["robust_ekf"]
    a1 = ekf_r["mean"] <= rob_r["mean"]
    a3_ekf = 6.0 <= ekf_r["mean"] <= 8.0
    print(f"A-1: EKF mean ≤ Robust-EKF mean: {a1} (EKF={ekf_r['mean']:.4f}, Robust={rob_r['mean']:.4f})", flush=True)
    print(f"A-3: EKF in 6-8m window: {a3_ekf}", flush=True)
    # A-2/A-3 require LNN; A-5 requires LNN for Wilcoxon

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
