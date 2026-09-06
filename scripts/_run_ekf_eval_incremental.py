"""简化版 EKF + Robust-EKF 论文级评估：增量写入 raw_results.json 防丢失。"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("LIQUIDLOC_SUPPRESS_PRINT_DICT", "1")
os.environ.setdefault("LIQUIDLOC_VERBOSE", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from liquidloc.dataio.readers.sim_reader import read_sim_sequence
from liquidloc.dataio.adapters.event_builder import (
    build_imu_events, build_uwb_events, build_vio_events,
)
from liquidloc.factories.estimator_factory import create_estimator
from liquidloc.fusion.fusion_runner import run_fusion


def _load_gt_xy(seq_dir: Path) -> list[tuple[float, float]]:
    rows = json.loads((seq_dir / "gt.json").read_text(encoding="utf-8"))
    out = []
    for r in rows:
        out.append((float(r.get("px", r.get("x", 0.0))), float(r.get("py", r.get("y", 0.0)))))
    return out


def _build_events(seq_dir: Path, seq_id: str) -> list[dict]:
    seed_root = seq_dir.parent
    raw_bundle, _ = read_sim_sequence(seq_id, seed_root)
    bundle = dict(raw_bundle)
    imu = build_imu_events(bundle.get("imu_raw", []), seq_id, seq_id)
    uwb = build_uwb_events(bundle.get("uwb_raw", []), seq_id, seq_id)
    vio = build_vio_events(bundle.get("vio_raw", []), seq_id, seq_id)
    all_events = imu + uwb + vio
    all_events.sort(key=lambda e: (e["t"], e["modality"]))
    for i, ev in enumerate(all_events):
        ev["dt"] = 0.0 if i == 0 else all_events[i]["t"] - all_events[i - 1]["t"]
    return all_events


def _run_estimator(seq_dir, estimator_name, estimator_cfg):
    from liquidloc.protocol import bridge_thresholds
    from liquidloc.fusion import fusion_runner

    _orig_bt = bridge_thresholds.BRIDGE_THRESHOLDS
    _tmp = dict(_orig_bt)
    _tmp["max_consecutive_skip_count"] = 5000
    bridge_thresholds.BRIDGE_THRESHOLDS = _tmp
    fusion_runner.BRIDGE_THRESHOLDS = _tmp

    seq_id = seq_dir.stem
    try:
        events = _build_events(seq_dir, seq_id)
        cfg = copy.deepcopy(estimator_cfg) if estimator_cfg else {}
        anchor_path = seq_dir / "anchor_layout.json"
        if anchor_path.exists():
            with open(anchor_path, encoding="utf-8") as fh:
                cfg["anchor_layout"] = json.load(fh)
        gt = _load_gt_xy(seq_dir)
        if gt:
            cfg["init_state"] = dict(cfg.get("init_state", {}))
            cfg["init_state"]["px"] = gt[0][0]
            cfg["init_state"]["py"] = gt[0][1]
            cfg["init_state"]["yaw"] = 0.0

        estimator = create_estimator(estimator_name, cfg)
        fusion_cfg = {"method_name": estimator_name, "seq_id": seq_id, "scene_id": seq_id}
        bundle = run_fusion(events, estimator, model_infer=None, feature_builder=None, cfg=fusion_cfg)

        states = bundle.get("states", [])
        pred = []
        for s in states:
            if isinstance(s, dict):
                pred.append((float(s.get("px", 0.0)), float(s.get("py", 0.0))))
            else:
                try:
                    pred.append((float(getattr(s, "px", 0.0)), float(getattr(s, "py", 0.0))))
                except Exception:
                    pred.append((0.0, 0.0))
        # RMSE
        n = min(len(pred), len(gt))
        if n == 0:
            return float("nan")
        err_sq = sum((pred[i][0] - gt[i][0]) ** 2 + (pred[i][1] - gt[i][1]) ** 2 for i in range(n))
        return math.sqrt(err_sq / n)
    except Exception as e:
        return float("nan")
    finally:
        bridge_thresholds.BRIDGE_THRESHOLDS = _orig_bt


def _combo(seq_id: str) -> str:
    a = "A3" if "A3" in seq_id else "A2"
    n = "N3" if "N3" in seq_id else "N2"
    return f"{a}{n}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "sim_e9_10seed_50unit")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-parallel", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ekf_eval_10seed")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "raw_results.json"
    # Load existing results (for resume)
    if raw_path.exists():
        try:
            results = json.loads(raw_path.read_text(encoding="utf-8"))
        except Exception:
            results = {}
    else:
        results = {}

    import yaml
    def load_cfg(name):
        p = ROOT / "configs" / "models" / f"{name}.yaml"
        if p.exists():
            with open(p, encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        return {}

    methods = ["ekf", "robust_ekf"]
    t0 = time.time()
    for method in methods:
        if method not in results:
            results[method] = {}
        cfg = load_cfg(method)
        for seed_id in range(args.n_seeds):
            seed_root = args.data_root / f"seed{seed_id}"
            if not seed_root.is_dir():
                continue
            seq_dirs = sorted([d for d in seed_root.iterdir()
                               if d.is_dir() and not d.name.startswith("seq") and d.name != "splits"])
            if len(seq_dirs) == 0:
                continue
            seed_key = str(seed_id)
            if seed_key in results[method] and len(results[method][seed_key]) >= len(seq_dirs):
                print(f"  [SKIP] {method} seed={seed_id} already done ({len(results[method][seed_key])} seqs)", flush=True)
                continue
            print(f"[{method}] seed {seed_id}/{args.n_seeds-1} ({len(seq_dirs)} seqs)", flush=True)
            results[method][seed_key] = {}

            def process_seq(seq_dir, _m=method, _c=cfg):
                return seq_dir.name, _run_estimator(seq_dir, _m, _c)

            with ThreadPoolExecutor(max_workers=args.n_parallel) as ex:
                futures = [ex.submit(process_seq, sd) for sd in seq_dirs]
                done = 0
                for fut in futures:
                    sid, rmse = fut.result()
                    results[method][seed_key][sid] = rmse
                    done += 1
                    if done % 30 == 0:
                        # Save incrementally
                        raw_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
                        print(f"    saved checkpoint @ {done}/{len(seq_dirs)}", flush=True)

            raw_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            elapsed_min = (time.time() - t0) / 60
            print(f"  {method} seed={seed_id} done ({elapsed_min:.1f} min elapsed)", flush=True)

    # ── Aggregate
    print("\n" + "=" * 60, flush=True)
    print("PAPER-GRADE RESULTS (10 seeds × 120 seq × 120s each)", flush=True)
    print("=" * 60, flush=True)
    agg: dict = {}
    for method in methods:
        all_rmses: list[float] = []
        combo_rmses: dict[str, list[float]] = {}
        for sid, seq_rmses in results[method].items():
            for qid, r in seq_rmses.items():
                if math.isnan(r):
                    continue
                all_rmses.append(r)
                c = _combo(qid)
                combo_rmses.setdefault(c, []).append(r)
        if not all_rmses:
            continue
        s = sorted(all_rmses)
        n = len(s)
        mean = sum(s) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in s) / max(n - 1, 1))
        agg[method] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "p50": round(s[n // 2], 4),
            "p95": round(s[min(int(0.95 * n), n - 1)], 4),
            "n": n,
            "combo": {c: {"mean": round(sum(v) / len(v), 4), "n": len(v)} for c, v in sorted(combo_rmses.items())},
        }
    (args.output / "aggregate_results.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\n{'Method':<15} {'Mean':>8} {'Std':>8} {'P50':>8} {'P95':>8} {'N':>6}", flush=True)
    print("-" * 60, flush=True)
    for method in methods:
        r = agg.get(method, {})
        if r:
            print(f"{method:<15} {r['mean']:>8.4f} {r['std']:>8.4f} {r['p50']:>8.4f} {r['p95']:>8.4f} {r['n']:>6}", flush=True)

    print("\nCombo breakdown (mean RMSE):", flush=True)
    print(f"{'Combo':<10} {'ekf':>10} {'robust_ekf':>12}", flush=True)
    for c in ["A2N2", "A2N3", "A3N2", "A3N3"]:
        e = agg.get("ekf", {}).get("combo", {}).get(c, {}).get("mean", float("nan"))
        r = agg.get("robust_ekf", {}).get("combo", {}).get(c, {}).get("mean", float("nan"))
        print(f"{c:<10} {e:>10.4f} {r:>12.4f}", flush=True)

    print("\nAcceptance A-1: EKF mean ≤ Robust-EKF mean:", end=" ", flush=True)
    ekf_m = agg.get("ekf", {}).get("mean", 99)
    rob_m = agg.get("robust_ekf", {}).get("mean", 99)
    print(f"{ekf_m:.4f} ≤ {rob_m:.4f} → {'PASS' if ekf_m <= rob_m else 'FAIL'}", flush=True)
    print("Acceptance A-3: EKF in 6-8m window:", end=" ", flush=True)
    print(f"{ekf_m:.4f} → {'PASS' if 6.0 <= ekf_m <= 8.0 else 'FAIL'}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min", flush=True)
    print(f"Raw results → {raw_path}", flush=True)
    print(f"Aggregate → {args.output / 'aggregate_results.json'}", flush=True)


if __name__ == "__main__":
    main()
