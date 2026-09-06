#!/usr/bin/env python
"""RZ-3 论文级评估: 5 method × 10 seed × 12 test seqs (=120/seed × 10=1200 trajectory)
- raw RMSE (D1: 主指标, 不对齐)
- Sim(3) Procrustes (报告为参考)
- Wilcoxon signed-rank p + Holm-Bonferroni 修正
- Per-method/per-seed/per-combo 切片统计

执行: python _eval_paper.py
"""
import os, sys, io, contextlib, json, time, math, gc, gzip, pickle, argparse
from pathlib import Path

ROOT = Path("E:/异步高NLOS")
sys.path.insert(0, str(ROOT / "src"))
os.environ["LIQUIDLOC_SUPPRESS_PRINT_DICT"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import yaml, torch
import liquidloc.fusion.fusion_runner as _fr
_fr._suppress_debug = True
import liquidloc.protocol.bridge_thresholds as _bt
_bt._suppress_debug = True
from liquidloc.factories.model_factory import create_model
from liquidloc.factories.estimator_factory import create_estimator
from liquidloc.pipelines.core_pipeline import _build_feature_window_builder

METHODS = ["ekf", "robust_ekf", "lstm", "liquid", "transformer"]
N_SEEDS = 10
N_SEQS_PER_COMBO = 1  # 4 combos × 1 = 4 per seed; 40 total


def procrustes_3d(pred, gt):
    """Sim(3) Procrustes (scale + rotation + translation) for 2D points."""
    n = min(len(pred), len(gt))
    if n < 3: return float("nan")
    p, g = np.array(pred[:n], dtype=np.float64), np.array(gt[:n], dtype=np.float64)
    pc, gc = p.mean(0), g.mean(0)
    pu, gu = p - pc, g - gc
    H = pu.T @ gu
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    scale = np.trace(np.diag(S) @ np.diag([1, 1, d])) / max(np.sum(pu**2), 1e-12)
    p_aligned = scale * (pu @ R.T) + gc
    return float(np.sqrt(np.mean(np.sum((p_aligned - g)**2, axis=1))))


def procrustes_2d_xyscale(pred_x, pred_y, gt_x, gt_y):
    """Sim(2) Procrustes (scale + rotation + translation) on 2D points (sim_e9 is 2D).
    Returns aligned RMSE in meters.
    """
    n = min(len(pred_x), len(gt_x))
    if n < 3: return float("nan")
    p = np.column_stack([pred_x[:n], pred_y[:n]]).astype(np.float64)
    g = np.column_stack([gt_x[:n], gt_y[:n]]).astype(np.float64)
    pc, gc = p.mean(0), g.mean(0)
    pu, gu = p - pc, g - gc
    H = pu.T @ gu
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    scale = (np.sum(S) * d) / max(np.sum(pu**2), 1e-12)
    p_aligned = scale * (pu @ R.T) + gc
    return float(np.sqrt(np.mean(np.sum((p_aligned - g)**2, axis=1))))


def raw_rmse_2d(pred_x, pred_y, gt_x, gt_y):
    """原始 RMSE (无对齐, D1 主指标) — sim_e9 是 2D 平面."""
    n = min(len(pred_x), len(gt_x))
    if n < 3: return float("nan")
    p = np.column_stack([pred_x[:n], pred_y[:n]])
    g = np.column_stack([gt_x[:n], gt_y[:n]])
    err = np.sqrt(np.sum((p - g)**2, axis=1))
    return float(np.sqrt(np.mean(err**2)))


def load_seq(seq_id, processed_root, raw_root, seed):
    """Load events + GT + anchor_layout for one sequence."""
    import json, gzip, pickle
    ev_path = processed_root / f"seed{seed}" / f"{seq_id}_events.pkl.gz"
    if not ev_path.exists():
        return None, None, None, None
    with gzip.open(ev_path, "rb") as f:
        events = pickle.load(f)
    raw_seq = raw_root / f"seed{seed}" / seq_id
    gt_rows = json.loads((raw_seq / "gt.json").read_text())
    al = json.loads((raw_seq / "anchor_layout.json").read_text())
    return events, gt_rows, al, raw_seq


def build_ekf_cfg(al, gt_rows, est_cfg_path):
    with open(est_cfg_path) as f:
        cfg = yaml.safe_load(f)
    anchor_pos = al["anchor_positions"]
    cfg["anchor_layout"] = {
        "anchor_positions": [[float(p[0]), float(p[1])] for p in anchor_pos],
        "anchor_ids": al["anchor_ids"],
        "anchor_count": al["anchor_count"],
        "protocol_k_level": al.get("protocol_k_level", "K1"),
        "protocol_geometry_level": al.get("protocol_geometry_level", "K1"),
    }
    cfg["init_state"] = {
        "px": gt_rows[0].get("px", gt_rows[0].get("x", 0)),
        "py": gt_rows[0].get("py", gt_rows[0].get("y", 0)),
        "vx": 0, "vy": 0, "yaw": 0,
        "bax": 0, "bay": 0, "bg": 0, "uwb_clock_bias": 0, "vio_scale": 1
    }
    return cfg


def run_silent(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return _fr.run_fusion(*args, **kwargs)


def eval_method(method, seed, processed_root, raw_root, model_cfgs, model_ckpts, existing=None):
    """Eval one method on one seed's test split, return per-seq results.
    existing: set of (seq_id) already done, skip them.
    """
    test_ids_path = processed_root / f"seed{seed}" / "test_ids.txt"
    if not test_ids_path.exists():
        return []
    test_ids = [s.strip() for s in test_ids_path.read_text().split(",") if s.strip()]
    results = []
    if existing is None:
        existing = set()

    for seq_id in test_ids:
        if seq_id in existing:
            continue
        events, gt_rows, al, raw_seq = load_seq(seq_id, processed_root, raw_root, seed)
        if events is None or not gt_rows:
            continue
        # GT 字段兼容: 优先 px/py (sim_e9), 退到 x/y
        gt_px = [r.get("px", r.get("x", 0)) for r in gt_rows]
        gt_py = [r.get("py", r.get("y", 0)) for r in gt_rows]

        try:
            # 所有方法都用 ekf.yaml 作为 estimator 配置基础
            # (模型推理用单独的 model cfg，estimator 永远是 EKF)
            cfg_path = ROOT / "configs/models/ekf.yaml"
            cfg = build_ekf_cfg(al, gt_rows, cfg_path)

            model = None
            fb = None
            if method in ("lstm", "liquid", "transformer"):
                ckpt_path = model_ckpts.get(method, {}).get(seed)
                if not ckpt_path or not Path(ckpt_path).exists():
                    continue
                mcfg_path = ROOT / f"configs/models/{method}_ekf.yaml"
                with open(mcfg_path) as f:
                    mcfg = yaml.safe_load(f)
                mcfg["checkpoint_path"] = str(ckpt_path)
                mcfg["project_root"] = str(ROOT)
                mcfg.setdefault("train", {})["deterministic"] = True
                model = create_model(f"{method}_ekf", mcfg)
                estimator = create_estimator("ekf", cfg)
                fb = _build_feature_window_builder(mcfg, estimator)
            else:
                estimator = create_estimator("ekf", cfg)

            traj = run_silent(events, estimator, model_infer=model, feature_builder=fb, cfg=cfg)
            states = traj["states"]
            px = [s.get("px", 0) for s in states]
            py = [s.get("py", 0) for s in states]
            raw = raw_rmse_2d(px, py, gt_px, gt_py)
            proc = procrustes_2d_xyscale(px, py, gt_px, gt_py)
            results.append({"seed": seed, "seq": seq_id, "raw": raw, "procrustes": proc})
        except Exception as e:
            print(f"  ERR {method} seed{seed} {seq_id}: {e}", flush=True)
    return results


def load_model_ckpts():
    """Load checkpoint paths for all trained NN methods."""
    ckpts = {"lstm": {}, "transformer": {}, "liquid": {}}
    for method in ckpts:
        for seed in range(N_SEEDS):
            p = ROOT / "checkpoints" / f"{method}_seed{seed}" / "checkpoints" / f"{method}_ekf_best_checkpoint.pt"
            if p.exists():
                ckpts[method][seed] = str(p)
    return ckpts


def statistical_tests(all_results, metric="procrustes"):
    """Wilcoxon signed-rank + Holm-Bonferroni 修正.
    Compares all NN methods vs EKF baseline using the specified metric.
    metric: 'raw' or 'procrustes' (Sim(3) Procrustes RMSE)
    """
    from scipy import stats
    print(f"\n=== Statistical Tests ({metric}, Wilcoxon + Holm-Bonferroni) ===", flush=True)
    by_method = {}
    for r in all_results:
        by_method.setdefault(r["method"], []).append(r[metric])
    ekf_vals = by_method.get("ekf", [])
    print(f"  EKF: n={len(ekf_vals)}")
    if not ekf_vals:
        return
    pvals = []
    method_names = []
    for m in ["lstm", "liquid", "transformer", "robust_ekf"]:
        vals = by_method.get(m, [])
        if not vals or len(vals) != len(ekf_vals):
            continue
        ekf_dict = {(r["seed"], r["seq"]): r[metric] for r in all_results if r["method"] == "ekf"}
        m_dict = {(r["seed"], r["seq"]): r[metric] for r in all_results if r["method"] == m}
        pairs = [(ekf_dict[k], m_dict[k]) for k in ekf_dict if k in m_dict]
        if len(pairs) < 10:
            continue
        e, mm = zip(*pairs)
        try:
            stat, p = stats.wilcoxon(e, mm, zero_method="zsplit", alternative="two-sided")
            delta = np.mean(mm) - np.mean(e)
            print(f"  {m} vs EKF: n_pairs={len(pairs)}, Δ={delta:.4f}m, p={p:.4e}")
            pvals.append(p)
            method_names.append(m)
        except Exception as e:
            print(f"  {m} vs EKF: ERR {e}")
    if pvals:
        m_tests = len(pvals)
        sorted_idx = np.argsort(pvals)
        sorted_pvals = np.array(pvals)[sorted_idx]
        holm = np.minimum(sorted_pvals * np.arange(m_tests, 0, -1), 1.0)
        for i in range(m_tests - 2, -1, -1):
            holm[i] = min(holm[i], holm[i+1])
        print(f"\n  Holm-Bonferroni 修正后 (k={m_tests}):")
        for idx, i in enumerate(sorted_idx):
            sig = "***" if holm[idx] < 0.001 else "**" if holm[idx] < 0.01 else "*" if holm[idx] < 0.05 else "ns"
            print(f"    {method_names[i]:>13s}: p_adjusted={holm[idx]:.4e}  {sig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="E:/异步高NLOS/data/processed/sim_e9_main")
    parser.add_argument("--raw-root", default="E:/异步高NLOS/data/raw/sim_e9_main")
    parser.add_argument("--output", default="E:/异步高NLOS/outputs/rz3_paper_evaluation.json")
    parser.add_argument("--methods", default="ekf,robust_ekf,lstm,liquid,transformer")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (method,seed) pairs already in the output JSON (default: off; first run only)")
    parser.add_argument("--force-methods", default="",
                        help="Comma-separated methods to FORCE re-evaluate even if resume says skip (used to add new metrics like procrustes)")
    args = parser.parse_args()
    methods = args.methods.split(",")
    force_methods = set(m.strip() for m in args.force_methods.split(",") if m.strip())
    processed_root = Path(args.processed_root)
    raw_root = Path(args.raw_root)

    print(f"[eval_paper] methods={methods}, n_seeds={N_SEEDS}, resume={args.resume}, force_methods={force_methods}", flush=True)
    print(f"[eval_paper] processed={processed_root}, raw={raw_root}", flush=True)

    model_ckpts = load_model_ckpts()
    print(f"[eval_paper] ckpts loaded: " + ", ".join(
        f"{m}={sum(1 for v in vs.values())}" for m, vs in model_ckpts.items()), flush=True)

    # Resume support: load existing results.
    # If --force-methods is set, drop those methods from all_results so they re-evaluate cleanly.
    all_results = []
    completed_pairs = set()
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        try:
            existing_data = json.loads(output_path.read_text())
            for r in existing_data:
                if r.get("method") in force_methods:
                    # Drop the old result so the new run is fresh
                    continue
                all_results.append(r)
                completed_pairs.add((r["method"], r["seed"]))
            print(f"[eval_paper] resume: loaded {len(existing_data)} existing records, "
                  f"{len(completed_pairs)} (method,seed) pairs will be skipped "
                  f"(force_methods={force_methods} re-evaluated)", flush=True)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[eval_paper] resume: failed to load existing JSON ({e}), starting fresh", flush=True)
    elif not args.resume and output_path.exists():
        # Without --resume, start from scratch (overwrite)
        print(f"[eval_paper] --resume not set, will overwrite existing {output_path}", flush=True)

    t0 = time.time()
    for method in methods:
        for seed in range(N_SEEDS):
            if (method, seed) in completed_pairs:
                print(f"  {method} seed{seed}: skip (already in JSON)", flush=True)
                continue
            t_method = time.time()
            completed_seqs = {r["seq_id"] for r in all_results if r["method"] == method and r["seed"] == seed}
            runs = eval_method(method, seed, processed_root, raw_root, None, model_ckpts,
                               existing=completed_seqs)
            for r in runs:
                r["method"] = method
            all_results.extend(runs)
            # Save incrementally after each (method,seed) so resume can pick up later
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(all_results, indent=2))
            print(f"  {method} seed{seed}: n={len(runs)} ({time.time()-t_method:.0f}s, "
                  f"saved {len(all_results)} total)", flush=True)
            gc.collect()
    print(f"\n[eval_paper] DONE: {len(all_results)} trajectories in {time.time()-t0:.0f}s", flush=True)
    print(f"[eval_paper] saved: {output_path}", flush=True)

    # Per-method stats
    print("\n=== Per-Method RMSE (m) ===", flush=True)
    for metric, label in [("raw", "Raw (no align)"), ("procrustes", "Sim(3) Procrustes")]:
        by_method = {}
        n_skipped_missing = 0
        for r in all_results:
            if metric not in r:
                n_skipped_missing += 1
                continue
            by_method.setdefault(r["method"], []).append(r[metric])
        print(f"  --- {label} --- (skipped {n_skipped_missing} records missing '{metric}' field)")
        for m, vals in by_method.items():
            vals = [v for v in vals if math.isfinite(v)]
            if vals:
                print(f"  {m:>13s}: n={len(vals):3d}  mean={np.mean(vals):.4f}m  std={np.std(vals):.4f}m  "
                      f"median={np.median(vals):.4f}m  [min={min(vals):.2f}, max={max(vals):.2f}]")

    # Per-seed stats
    print("\n=== Per-Seed RMSE by Method ===", flush=True)
    by_seed_method = {}
    for r in all_results:
        by_seed_method.setdefault((r["method"], r["seed"]), []).append(r["raw"])
    methods_list = sorted(set(r["method"] for r in all_results))
    header = "  " + "method".ljust(13) + " | " + " | ".join(f"s{s}".rjust(8) for s in range(N_SEEDS)) + " | " + "mean".rjust(8)
    print(header)
    for m in methods_list:
        row = f"  {m:>13s} | " + " | ".join(
            f"{np.mean(by_seed_method.get((m, s), [float('nan')])):8.4f}" if (m, s) in by_seed_method else f"{'--':>8}"
            for s in range(N_SEEDS)
        )
        all_m = [v for k, v in by_seed_method.items() if k[0] == m for v in v]
        row += f" | {np.mean(all_m):8.4f}" if all_m else " | --"
        print(row)

    # Statistical tests for metrics present in the results
    for metric in ["raw", "procrustes"]:
        has_metric = any(metric in r for r in all_results)
        if has_metric:
            statistical_tests(all_results, metric=metric)
        else:
            print(f"\n=== Statistical Tests ({metric}) ===", flush=True)
            print(f"  (skipped - no '{metric}' field in results)", flush=True)


if __name__ == "__main__":
    main()
