"""Real neural training on sim_e9_10seed_50unit data.

Trains LSTM, Transformer, and Liquid networks on the 10-seed × 200-seq sim dataset
using TrainPipeline().run() with direct event injection (no prepare step needed).

Usage:
    # Train all 3 methods on all 10 seeds
    PYTHONPATH=src .venv-gpu/Scripts/python.exe scripts/_train_neural_methods.py

    # Train single method / single seed
    PYTHONPATH=src .venv-gpu/Scripts/python.exe scripts/_train_neural_methods.py \
        --methods lstm_ekf --seeds 0 --epochs 10
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("LIQUIDLOC_SUPPRESS_PRINT_DICT", "1")
os.environ.setdefault("LIQUIDLOC_ALIGNMENT_POSE_FULL_SCALE_M", "25.0")

# ── helpers ──────────────────────────────────────────────────────────────────


def _load_seq_ids(data_root: Path) -> list[tuple[int, str]]:
    """Return [(seed, seq_id), ...] for all sequences in the 10-seed data tree."""
    result = []
    for seed_dir in sorted(data_root.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        seed = int(seed_dir.name.replace("seed", ""))
        for seq_dir in sorted(seed_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            seq_id = seq_dir.name  # e.g. "s0_a2n2_00"
            result.append((seed, seq_id))
    return result


def _build_events(seq_dir: Path, seq_id: str) -> list[dict[str, Any]]:
    """Merge imu + uwb + vio JSON rows into a single time-sorted event stream.

    Schema (per liquidloc.protocol.event_schema):
      - Primary keys: t, dt, modality, meta
      - modality-specific payload key (PAYLOAD_KEYS[modality])
      - required payload fields (REQUIRED_PAYLOAD_FIELDS[modality])
      - meta: scene_id, seq_id

    VIO stores absolute (x, y, yaw) per frame. Compute (dx, dy, dyaw) as
    consecutive-frame deltas for the pipeline.
    """
    events = []

    # Scene ID from sim_meta
    seq_meta_path = seq_dir / "sim_meta.json"
    scene_id = seq_id
    if seq_meta_path.is_file():
        try:
            seq_meta = json.loads(seq_meta_path.read_text(encoding="utf-8"))
            axes = seq_meta.get("axes_override", {})
            a = axes.get("A", "A2")
            n = axes.get("N", "N2")
            scene_id = f"S({a},{n},V0,K1,M1)"
        except Exception:
            pass

    # Load raw modality lists
    raw: dict[str, list] = {}
    for mod, fname in [("imu", "imu.json"), ("uwb", "uwb.json"), ("vio", "vio.json")]:
        fpath = seq_dir / fname
        if fpath.is_file():
            raw[mod] = json.loads(fpath.read_text(encoding="utf-8"))
        else:
            raw[mod] = []

    # Pre-compute VIO deltas (indexed by timestamp for O(1) lookup)
    vio_deltas: dict[float, dict] = {}
    if raw.get("vio"):
        vio_rows = raw["vio"]
        for i in range(1, len(vio_rows)):
            prev, curr = vio_rows[i - 1], vio_rows[i]
            ts = curr["timestamp"]
            vio_deltas[ts] = {
                "dx": curr.get("x", 0.0) - prev.get("x", 0.0),
                "dy": curr.get("y", 0.0) - prev.get("y", 0.0),
                "yaw": curr.get("yaw", 0.0),
                "prev_yaw": prev.get("yaw", 0.0),
                "quality": curr.get("quality", curr.get("vio_valid", True)),
            }
        # First frame: zero delta
        if vio_rows:
            first_ts = vio_rows[0]["timestamp"]
            vio_deltas[first_ts] = {
                "dx": 0.0, "dy": 0.0,
                "yaw": vio_rows[0].get("yaw", 0.0),
                "prev_yaw": vio_rows[0].get("yaw", 0.0),
                "quality": vio_rows[0].get("quality", 0.85),
            }

    # Build events from raw lists, all merged into one time-sorted stream
    raw_events: list[dict] = []
    for row in raw.get("imu", []):
        raw_events.append({"t": float(row["timestamp"]), "modality": "imu", "_row": row})
    for row in raw.get("uwb", []):
        raw_events.append({"t": float(row["timestamp"]), "modality": "uwb", "_row": row})
    for row in raw.get("vio", []):
        raw_events.append({"t": float(row["timestamp"]), "modality": "vio", "_row": row})
    raw_events.sort(key=lambda e: e["t"])

    # Compute dt as diff to previous event (any modality)
    for i, ev in enumerate(raw_events):
        row = ev.pop("_row")
        ts = ev["t"]
        ev["dt"] = raw_events[i]["t"] - raw_events[i - 1]["t"] if i > 0 else 0.0
        ev["meta"] = {"scene_id": scene_id, "seq_id": seq_id}

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
            # Wrap dyaw to [-pi, pi]
            import math
            dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
            ev["vio_payload"] = {
                "dx": float(d["dx"]),
                "dy": float(d["dy"]),
                "dyaw": float(dyaw),
                "quality": float(d["quality"]),
            }
        events.append(ev)

    return events


def _build_gt(seq_dir: Path) -> list[dict[str, float]]:
    """Load gt.json as list of {timestamp, x, y}."""
    gt_path = seq_dir / "gt.json"
    if not gt_path.is_file():
        return []
    return json.loads(gt_path.read_text(encoding="utf-8"))


# ── main trainer ─────────────────────────────────────────────────────────────


def train_one(seed: int, method: str, seq_ids: list[str], events_by_seq: dict, gt_by_seq: dict,
              epochs: int, output_root: Path) -> dict[str, Any]:
    """Call TrainPipeline().run() for one (seed, method)."""
    from liquidloc.common.config_utils import load_yaml_config
    from liquidloc.pipelines.train_pipeline import TrainPipeline

    # Load model config
    model_name_map = {
        "lstm_ekf": "lstm_ekf.yaml",
        "transformer_ekf": "transformer_ekf.yaml",
        "liquid_ekf": "liquid_ekf.yaml",
    }
    model_cfg_path = ROOT / "configs" / "models" / model_name_map[method]
    model_cfg = load_yaml_config(model_cfg_path)

    # Override epochs + sync related fields (matching 05_train_lstm.py L82-112)
    _epochs = int(epochs)
    _train_cfg = model_cfg.setdefault("train", {})
    _train_cfg["seed"] = seed
    _train_cfg["epochs"] = _epochs
    _train_cfg["patience"] = _epochs  # no early stopping

    # GPU memory optimization: 7.96GB GPU (RTX 5060) -> batch_size=4 for liquid
    # 8-step grad accum gives effective batch=32 (same as original cfg)
    if method == "liquid_ekf":
        _train_cfg["batch_size"] = 4
        _train_cfg["eval_batch_size"] = 8
    else:
        _train_cfg["batch_size"] = _train_cfg.get("batch_size", 16)

    _lr_sched = _train_cfg.get("lr_scheduler", {})
    if isinstance(_lr_sched, dict) and _lr_sched.get("enabled"):
        _lr_sched["T_max"] = _epochs
    _train_cfg["lr_scheduler"] = _lr_sched

    _phase = _train_cfg.get("phase_schedule", {})
    if isinstance(_phase, dict):
        _warmup = int(_phase.get("warmup_epochs", 0))
        _gate = int(_phase.get("gate_alignment_epochs", 0))
        if _warmup + _gate > _epochs or _phase.get("full_tuning_epochs", _epochs) != _epochs:
            _train_cfg["phase_schedule"] = {
                "warmup_epochs": 0,
                "gate_alignment_epochs": 0,
                "full_tuning_epochs": _epochs,
            }

    payload = {
        "model_name": method,
        "model_cfg": model_cfg,
        "mode": "full",          # full: train to completion
        "events_by_seq_id": events_by_seq,
        "ground_truth_by_seq_id": gt_by_seq,
        "split_ids": seq_ids,
        "disable_scene_leak_check": True,
        "output_root": str(output_root / f"seed{seed}_{method}"),
    }

    print(f"[train] seed={seed} method={method} epochs={epochs} seqs={len(seq_ids)}", flush=True)
    result = TrainPipeline().run(payload)
    report = result.metadata.get("train_report") or {}
    ckpt = result.metadata.get("checkpoint_path") or ""
    print(f"[train] seed={seed} method={method} done | best_loss={report.get('best_loss')} ckpt={ckpt}", flush=True)
    return {"seed": seed, "method": method, "best_loss": report.get("best_loss"),
            "checkpoint_path": ckpt, "epochs": _epochs}


# ── entry point ──────────────────────────────────────────────────────────────


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Real neural training on sim_e9_10seed_50unit")
    parser.add_argument("--methods", default="lstm_ekf,transformer_ekf,liquid_ekf",
                        help="Comma-separated methods: lstm_ekf, transformer_ekf, liquid_ekf")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated seeds (default: 0-9)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "data" / "raw" / "sim_e9_10seed_50unit",
                        help="10-seed data root")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "checkpoints" / "sim_e9_10seed_50unit",
                        help="Training output root")
    parser.add_argument("--val-seed", type=int, default=None,
                        help="Hold out this seed as validation (default: last seed)")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seed_list = sorted(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    # Validation seed: only reserve a val seed if we have >1 seeds; otherwise train on all seeds
    val_seed = args.val_seed
    if val_seed is None and len(seed_list) > 1:
        val_seed = seed_list[-1]
    train_seeds = [s for s in seed_list if s != val_seed] if val_seed is not None else list(seed_list)
    epochs = args.epochs
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[train_neural] data_root={data_root} methods={methods} train_seeds={train_seeds} "
          f"val_seed={val_seed} epochs={epochs}", flush=True)

    # Collect all (seed, seq_id) pairs
    all_pairs = _load_seq_ids(data_root)
    if not all_pairs:
        print(f"[ERROR] No sequences found under {data_root}", flush=True)
        return 1
    print(f"[train_neural] {len(all_pairs)} total seqs across {len(seed_list)} seeds", flush=True)

    # Group by seed
    by_seed: dict[int, list[str]] = {}
    for seed, seq_id in all_pairs:
        by_seed.setdefault(seed, []).append(seq_id)

    results = []
    checkpoint_paths: dict[str, dict[int, str]] = {m: {} for m in methods}

    for method in methods:
        print(f"\n{'='*60}\n[train_neural] === Training {method} ===\n{'='*60}", flush=True)
        for seed in train_seeds:
            t0 = time.time()
            seq_ids = by_seed.get(seed, [])

            # Build events & GT for this seed
            events_by_seq: dict[str, list] = {}
            gt_by_seq: dict[str, list] = {}
            for seq_id in seq_ids:
                seq_dir = data_root / f"seed{seed}" / seq_id
                if not seq_dir.is_dir():
                    continue
                events_by_seq[seq_id] = _build_events(seq_dir, seq_id)
                gt_by_seq[seq_id] = _build_gt(seq_dir)

            if not events_by_seq:
                print(f"[WARN] seed={seed}: no sequences loaded, skipping", flush=True)
                continue

            ckpt_dir = output_root / method
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            r = train_one(
                seed=seed,
                method=method,
                seq_ids=seq_ids,
                events_by_seq=events_by_seq,
                gt_by_seq=gt_by_seq,
                epochs=epochs,
                output_root=ckpt_dir,
            )
            checkpoint_paths[method][seed] = r["checkpoint_path"]
            results.append(r)
            elapsed = time.time() - t0
            print(f"[train_neural] seed={seed} {method} done in {elapsed:.1f}s | best_loss={r['best_loss']}", flush=True)

    # Save checkpoint manifest
    manifest = {
        "methods": methods,
        "train_seeds": train_seeds,
        "val_seed": val_seed,
        "epochs": epochs,
        "data_root": str(data_root),
        "checkpoint_paths": checkpoint_paths,
        "results": results,
    }
    manifest_path = output_root / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[train_neural] All done. Manifest → {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
