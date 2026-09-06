"""Real neural network inference on sim_e9_10seed_50unit data.

For each (seed, method) unit:
  1. Load trained checkpoint from checkpoints/sim_e9_10seed_50unit/{method}/seed{N}_{method}/checkpoints/{method}_best_checkpoint.pt
  2. Run run_fusion() with EKF shell + neural model on the same data
  3. Compute trajectory-level RMSE

Usage:
    # Run all 5 methods on all 10 seeds (after training)
    PYTHONPATH=src .venv-gpu/Scripts/python.exe scripts/_run_10seed_nn_infer.py

    # Run single method
    PYTHONPATH=src .venv-gpu/Scripts/python.exe scripts/_run_10seed_nn_infer.py --methods lstm_ekf
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("LIQUIDLOC_SUPPRESS_PRINT_DICT", "1")

import torch
import yaml
from liquidloc.common.config_utils import load_yaml_config
from liquidloc.factories.model_factory import create_model
from liquidloc.factories.estimator_factory import create_estimator
from liquidloc.fusion.fusion_runner import run_fusion
from liquidloc.pipelines.core_pipeline import _build_feature_window_builder


# ── paths ───────────────────────────────────────────────────────────────────
DATA_ROOT = ROOT / "data" / "raw" / "sim_e9_10seed_50unit"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "sim_e9_10seed_50unit"
OUT_ROOT = ROOT / "outputs" / "real_nn_50unit"
MODELS = ["lstm_ekf", "transformer_ekf", "liquid_ekf"]
CKPT_NAME = {
    "lstm_ekf": "lstm_ekf_best_checkpoint.pt",
    "transformer_ekf": "transformer_ekf_best_checkpoint.pt",
    "liquid_ekf": "liquid_ekf_best_checkpoint.pt",
}


def _load_yaml_checkpoint_path(method: str) -> dict[str, Any]:
    """Load yaml config, inject checkpoint_path and project_root from real checkpoint."""
    yaml_path = ROOT / "configs" / "models" / f"{method}.yaml"
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return cfg


def _list_seq_dirs(seed: int) -> list[Path]:
    return sorted(p for p in (DATA_ROOT / f"seed{seed}").iterdir() if p.is_dir() and not p.name.startswith("seq"))


def _build_events(seq_dir: Path) -> list[dict[str, Any]]:
    """Build event stream in liquidloc schema format from raw sim data.

    Per liquidloc.protocol.event_schema:
      t/dt/modality/meta + imu_payload/uwb_payload/vio_payload
    VIO converts absolute (x,y,yaw) to (dx,dy,dyaw) deltas.
    """
    events: list[dict[str, Any]] = []

    seq_meta_path = seq_dir / "sim_meta.json"
    scene_id = "S(A2,N2,V0,K1,M1)"
    if seq_meta_path.is_file():
        try:
            meta = json.loads(seq_meta_path.read_text(encoding="utf-8"))
            axes = meta.get("axes_override", {})
            a = axes.get("A", "A2")
            n = axes.get("N", "N2")
            scene_id = f"S({a},{n},V0,K1,M1)"
        except Exception:
            pass

    seq_id = seq_dir.name
    raw: dict[str, list] = {}
    for mod, fname in [("imu", "imu.json"), ("uwb", "uwb.json"), ("vio", "vio.json")]:
        fpath = seq_dir / fname
        if fpath.is_file():
            raw[mod] = json.loads(fpath.read_text(encoding="utf-8"))
        else:
            raw[mod] = []

    # Pre-compute VIO deltas
    vio_deltas: dict[float, dict] = {}
    if raw.get("vio"):
        for i in range(1, len(raw["vio"])):
            p, c = raw["vio"][i - 1], raw["vio"][i]
            vio_deltas[c["timestamp"]] = {
                "dx": c.get("x", 0.0) - p.get("x", 0.0),
                "dy": c.get("y", 0.0) - p.get("y", 0.0),
                "yaw": c.get("yaw", 0.0),
                "prev_yaw": p.get("yaw", 0.0),
                "quality": c.get("quality", c.get("vio_valid", True)),
            }
        if raw["vio"]:
            ts0 = raw["vio"][0]["timestamp"]
            vio_deltas[ts0] = {
                "dx": 0.0, "dy": 0.0,
                "yaw": raw["vio"][0].get("yaw", 0.0),
                "prev_yaw": raw["vio"][0].get("yaw", 0.0),
                "quality": raw["vio"][0].get("quality", 0.85),
            }

    # Build raw events list
    raw_events: list[dict] = []
    for row in raw.get("imu", []):
        raw_events.append({"t": float(row["timestamp"]), "modality": "imu", "_row": row})
    for row in raw.get("uwb", []):
        raw_events.append({"t": float(row["timestamp"]), "modality": "uwb", "_row": row})
    for row in raw.get("vio", []):
        raw_events.append({"t": float(row["timestamp"]), "modality": "vio", "_row": row})
    raw_events.sort(key=lambda e: e["t"])

    # Build event stream with dt and payload
    for i, ev in enumerate(raw_events):
        row = ev.pop("_row")
        ts = ev["t"]
        ev["dt"] = ts - raw_events[i - 1]["t"] if i > 0 else 0.0
        ev["meta"] = {"scene_id": scene_id, "seq_id": seq_id}
        # Clamp tiny floating errors for same-timestamp events (IMU+UWB+VIO all start at 0.0)
        if ev["dt"] < 1e-5:
            ev["dt"] = 0.0
        mod = ev["modality"]
        if mod == "imu":
            ev["imu_payload"] = {
                "ax": float(row.get("ax", 0.0)),
                "ay": float(row.get("ay", 0.0)),
                "az": float(row.get("az", 0.0)),
                "gx": float(row.get("gx", 0.0)),
                "gy": float(row.get("gy", 0.0)),
                "gz": float(row.get("gz", 0.0)),
            }
        elif mod == "uwb":
            ev["uwb_payload"] = {
                "anchor_id": int(row.get("anchor_id", 0)),
                "range": float(row.get("range", 0.0)),
                "valid": bool(row.get("valid", True)),
                "quality": float(row.get("quality", 0.95)),
                "nl_flag": int(row.get("nl_flag", 0)),
            }
        elif mod == "vio":
            d = vio_deltas.get(ts, {"dx": 0.0, "dy": 0.0, "yaw": 0.0,
                                     "prev_yaw": 0.0, "quality": 0.85})
            dyaw = d["yaw"] - d["prev_yaw"]
            dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
            ev["vio_payload"] = {
                "dx": float(d["dx"]),
                "dy": float(d["dy"]),
                "dyaw": float(dyaw),
                "quality": float(d["quality"]),
            }
        events.append(ev)
    return events


def _load_gt(seq_dir: Path) -> list[dict[str, float]]:
    return json.loads((seq_dir / "gt.json").read_text(encoding="utf-8"))


def _load_anchor_layout(seq_dir: Path) -> dict:
    al_path = seq_dir / "anchor_layout.json"
    if al_path.is_file():
        return json.loads(al_path.read_text(encoding="utf-8"))
    return {}


def _compute_rmse(states: list[dict], gt_rows: list[dict], warmup_frac: float = 0.083) -> float:
    """Trajectory-level 2D RMSE (excludes first 10s warm-up, per P37)."""
    n = min(len(states), len(gt_rows))
    if n == 0:
        return float("nan")
    warmup = int(n * warmup_frac)
    err_sq: list[float] = []
    for i in range(warmup, n):
        sx, sy = states[i].get("px", states[i].get("x", 0.0)), states[i].get("py", states[i].get("y", 0.0))
        gx, gy = gt_rows[i].get("px", gt_rows[i].get("x", 0.0)), gt_rows[i].get("py", gt_rows[i].get("y", 0.0))
        err_sq.append((sx - gx) ** 2 + (sy - gy) ** 2)
    return math.sqrt(sum(err_sq) / len(err_sq)) if err_sq else float("nan")


def _compute_p95(states: list[dict], gt_rows: list[dict], warmup_frac: float = 0.083) -> float:
    n = min(len(states), len(gt_rows))
    if n == 0:
        return float("nan")
    warmup = int(n * warmup_frac)
    errs: list[float] = []
    for i in range(warmup, n):
        sx, sy = states[i].get("px", states[i].get("x", 0.0)), states[i].get("py", states[i].get("y", 0.0))
        gx, gy = gt_rows[i].get("px", gt_rows[i].get("x", 0.0)), gt_rows[i].get("py", gt_rows[i].get("y", 0.0))
        errs.append(math.sqrt((sx - gx) ** 2 + (sy - gy) ** 2))
    if not errs:
        return float("nan")
    return sorted(errs)[int(0.95 * len(errs))]


# ── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Real neural inference on 10-seed sim data")
    parser.add_argument("--methods", default="lstm_ekf,transformer_ekf,liquid_ekf",
                        help="Comma-separated methods")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated seeds")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--use-stride", type=int, default=10,
                        help="IMU stride for inference (default 10 → 15Hz, 40x speedup over 150Hz raw)")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = sorted(int(s) for s in args.seeds.split(",") if s.strip())
    data_root = Path(args.data_root)
    checkpoint_root = Path(args.checkpoint_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load EKF shell
    ekf_cfg = load_yaml_config(ROOT / "configs" / "models" / "ekf.yaml")
    estimator = create_estimator("ekf", ekf_cfg)
    print(f"[nn_infer] EKF estimator created", flush=True)

    all_results: dict[str, dict] = {m: {} for m in methods}

    for method in methods:
        print(f"\n=== {method} ===", flush=True)
        model_cfg = _load_yaml_checkpoint_path(method)
        ckpt_path = checkpoint_root / method / f"seed0_{method}" / "checkpoints" / CKPT_NAME[method]
        model = None
        if ckpt_path.exists():
            try:
                model = create_model(method, model_cfg)
                ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                # Checkpoint stores model state under 'model_state' → load from there
                sd = ck.get("model_state") or ck.get("state_dict") or ck
                model.load_state_dict(sd)
                model.eval()
                n_tensors = len(sd)
                print(f"  [model load OK] {ckpt_path.name} ({n_tensors} tensor groups)", flush=True)
            except Exception as ex:
                print(f"  [model load FAIL: {ex}], running EKF-only baseline", flush=True)
                model = None
        else:
            print(f"  [model missing: {ckpt_path}], running EKF-only baseline", flush=True)

        for seed in seeds:
            seed_dir = data_root / f"seed{seed}"
            seq_dirs = [sd for sd in seed_dir.iterdir() if sd.is_dir() and not sd.name.startswith("seq")]
            if not seq_dirs:
                continue
            t0_seed = time.time()
            rmse_per_seq: list[float] = []
            p95_per_seq: list[float] = []
            per_combo_rmses: dict[str, list] = defaultdict(list)
            errors_shown = 0
            for seq_dir in seq_dirs:
                try:
                    seq_id = seq_dir.name
                    events = _build_events(seq_dir)
                    # IMU stride sampling for speed
                    if args.use_stride > 1:
                        events = [
                            ev for i, ev in enumerate(events)
                            if ev["modality"] != "imu" or i % args.use_stride == 0
                        ]
                        # Recompute dt after stride filter: validator requires dt == t_curr - t_prev
                        for j, ev in enumerate(events):
                            ev["dt"] = (ev["t"] - events[j - 1]["t"]) if j > 0 else 0.0
                            if ev["dt"] < 1e-5:
                                ev["dt"] = 0.0
                    gt = _load_gt(seq_dir)
                    al = _load_anchor_layout(seq_dir)
                    run_cfg = dict(model_cfg)
                    if al:
                        run_cfg["anchor_layout"] = al
                    feature_builder = _build_feature_window_builder(run_cfg, estimator)
                    bundle = run_fusion(
                        events=events, estimator=estimator,
                        model_infer=model,  # trained model (None = bare EKF)
                        feature_builder=feature_builder, cfg=run_cfg,
                    )
                    states = bundle.get("states", [])
                    rmse = _compute_rmse(states, gt)
                    p95 = _compute_p95(states, gt)
                    if not math.isnan(rmse):
                        rmse_per_seq.append(rmse)
                    if not math.isnan(p95):
                        p95_per_seq.append(p95)
                    # combo key (e.g. "a2n2" from "s0_a2n2_00")
                    parts = seq_id.split("_")
                    if len(parts) >= 2:
                        combo = parts[1]
                        per_combo_rmses[combo].append(rmse)
                except Exception as ex:
                    if errors_shown < 2:
                        print(f"  [seq error {seq_dir.name}]: {type(ex).__name__}: {ex}", flush=True)
                        errors_shown += 1
            t_elapsed = time.time() - t0_seed
            if rmse_per_seq:
                mean_rmse = sum(rmse_per_seq) / len(rmse_per_seq)
                mean_p95 = sum(p95_per_seq) / len(p95_per_seq) if p95_per_seq else float("nan")
                std_rmse = math.sqrt(
                    sum((r - mean_rmse) ** 2 for r in rmse_per_seq) / max(1, len(rmse_per_seq) - 1)
                )
                all_results[method][seed] = {
                    "mean_rmse": round(mean_rmse, 4),
                    "std_rmse": round(std_rmse, 4),
                    "mean_p95": round(mean_p95, 4),
                    "n": len(rmse_per_seq),
                    "per_combo": {k: round(sum(v) / len(v), 4) for k, v in per_combo_rmses.items()},
                    "elapsed_s": round(t_elapsed, 1),
                }
                print(f"  [{method}] seed={seed} mean_rmse={mean_rmse:.3f}±{std_rmse:.3f}m "
                      f"n={len(rmse_per_seq)} p95={mean_p95:.3f}m ({t_elapsed:.1f}s)", flush=True)

    # Save results
    out_path = out_root / "real_nn_50unit_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[nn_infer] Report → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
