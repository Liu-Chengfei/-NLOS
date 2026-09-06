"""Neural inference: run LSTM/Transformer/Liquid checkpoints on sim_e9 events.

Uses pre-trained checkpoints from outputs/{lstm,liquid,transformer}_train/
and runs inference on all 15 sequences (5 seeds × 3 variants) in
outputs/prepare_sim/. Computes per-(seed, method) RMSE.

Each model takes a windowed tensor of (dt, ax, ay, gz, range, dx, dy, dyaw)
features per the checkpoint's model_cfg.feature_order. We build the
windowed input from the prepared event stream, call
model.predict_intermediate_tensors(window_tensor), and extract (px, py).
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liquidloc.factories.model_factory import create_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVENTS_DIR = ROOT / "outputs" / "prepare_sim"
RAW_ROOT = ROOT / "data" / "raw" / "sim_e9_main"

CHECKPOINTS = {
    "lstm_ekf": ROOT / "outputs" / "lstm_train" / "checkpoints" / "lstm_ekf_best_checkpoint.pt",
    "transformer_ekf": ROOT / "outputs" / "transformer_train" / "checkpoints" / "transformer_ekf_best_checkpoint.pt",
    "liquid_ekf": ROOT / "outputs" / "liquid_train" / "checkpoints" / "liquid_ekf_training_best_checkpoint.pt",
}
WINDOW = 32  # 32 events per inference window (covers UWB + IMU + VIO)


def _load_gt_traj(seq_id: str, n_pts: int) -> list[tuple[float, float]]:
    seq_dir = RAW_ROOT / seq_id
    gt_path = seq_dir / "gt.json"
    if not gt_path.is_file():
        return []
    rows = json.loads(gt_path.read_text(encoding="utf-8"))
    pts = [(r["px"], r["py"]) for r in rows[100:]]
    if len(pts) < n_pts:
        return []
    return pts[:n_pts]


def _sim3_align_2d(preds: list[tuple[float, float]], gts: list[tuple[float, float]]) -> float:
    if len(preds) != len(gts) or len(preds) < 3:
        return float("nan")
    n = len(preds)
    pm = (sum(p[0] for p in preds) / n, sum(p[1] for p in preds) / n)
    gm = (sum(g[0] for g in gts) / n, sum(g[1] for g in gts) / n)
    pc = [(p[0] - pm[0], p[1] - pm[1]) for p in preds]
    gc = [(g[0] - gm[0], g[1] - gm[1]) for g in gts]
    Hxx = sum(p[0] * g[0] for p, g in zip(pc, gc))
    Hyy = sum(p[1] * g[1] for p, g in zip(pc, gc))
    denom = Hxx + Hyy
    if abs(denom) < 1e-9:
        return math.sqrt(sum((p[0] - g[0]) ** 2 + (p[1] - g[1]) ** 2 for p, g in zip(preds, gts)) / n)
    Hxy = sum(p[0] * g[1] for p, g in zip(pc, gc))
    Hyx = sum(p[1] * g[0] for p, g in zip(pc, gc))
    theta = 0.5 * math.atan2(Hxy - Hyx, denom)
    c, s = math.cos(theta), math.sin(theta)
    rot_pc = [(c * p[0] - s * p[1], s * p[0] + c * p[1]) for p in pc]
    sse = sum((p[0] - g[0]) ** 2 + (p[1] - g[1]) ** 2 for p, g in zip(rot_pc, gc))
    return math.sqrt(sse / n)


def _load_checkpoint(model, path: Path) -> bool:
    """Load checkpoint into ModelAPI wrapper."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return False
    # Use full model_state which has {network, risk_calibration, ...}
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "network" in ckpt:
        state = ckpt
    else:
        state = ckpt
    try:
        model.load_state_dict(state)
        return True
    except Exception as e:
        print(f"  load error: {e}")
        return False


def _events_to_tensor(events: list[dict], feature_order: list[str], window: int) -> torch.Tensor:
    """Build a (T, len(feature_order)) tensor of features from event dicts."""
    rows = []
    for ev in events:
        row = []
        for f in feature_order:
            # Try modalities: imu_payload, uwb_payload, vio_payload
            payload = None
            for mp in ("imu_payload", "uwb_payload", "vio_payload"):
                if ev.get(mp):
                    payload = ev[mp]
                    break
            if payload is None:
                row.append(0.0)
            elif f in payload:
                row.append(float(payload[f]))
            elif f == "dt":
                row.append(float(ev.get("dt", 0.0)))
            elif f == "range":
                row.append(float((payload or {}).get("range", 0.0)))
            else:
                row.append(0.0)
        rows.append(row)
    # Take last `window` events
    rows = rows[-window:]
    if len(rows) < window:
        # pad with zeros
        rows = [[0.0] * len(feature_order)] * (window - len(rows)) + rows
    return torch.tensor(rows, dtype=torch.float32)


def main():
    """Run inference for all 3 neural methods on all 15 sequences."""
    summary: dict[str, dict[int, list[float]]] = {}  # method → seed → rmse list
    sequences = sorted(EVENTS_DIR.glob("*_events.pkl.gz"))
    print(f"Found {len(sequences)} sequence events")

    for method_name, ckpt_path in CHECKPOINTS.items():
        if not ckpt_path.exists():
            print(f"SKIP {method_name}: no checkpoint at {ckpt_path}")
            continue
        print(f"\n=== {method_name} ===")
        # Get full cfg (including network) from checkpoint to match trained dimensions
        ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_cfg = ckpt_meta.get("model_cfg", {})
        network_cfg = model_cfg.get("network", {})
        feature_order = model_cfg.get("feature_order", ["dt", "ax", "ay", "gz", "range", "dx", "dy", "dyaw"])
        window_cfg = model_cfg.get("window", {})
        window_size = window_cfg.get("size", WINDOW) if isinstance(window_cfg, dict) else int(window_cfg)
        cfg_for_factory = {"network": network_cfg, "window": model_cfg.get("window", {})}
        # Pass feature_order at top level (not under network)
        if feature_order:
            cfg_for_factory["feature_order"] = feature_order
        model = create_model(method_name, cfg_for_factory)
        if not _load_checkpoint(model, ckpt_path):
            print(f"  load failed for {method_name}")
            continue
        per_seq_rmse: list[float] = []
        seq_seeds: dict[int, list[float]] = {}
        model.eval()
        with torch.no_grad():
            for ev_path in sequences:
                try:
                    import gzip
                    with gzip.open(ev_path, "rb") as f:
                        events = pickle.load(f)
                except Exception as e:
                    print(f"  load fail {ev_path.name}: {e}")
                    continue
                seq_id = ev_path.stem.replace("_events", "")
                # Parse seed from seq_id (e.g., seed0_seed0_seed0 → seed 0)
                parts = seq_id.split("_")
                try:
                    seed = int(parts[0].replace("seed", ""))
                except ValueError:
                    seed = 0
                # Inference: feed events in non-overlapping windows.
                # The model factory's predict_intermediate_tensors expects a
                # dict with: feature_order, current_modality, feature_values (1D),
                # missing_mask (1D), dt, feature_window (2D), missing_mask_window (2D).
                preds: list[tuple[float, float]] = []
                # Build per-event feature matrix
                n_events = len(events)
                all_feats: list[list[float]] = []  # (T, D)
                for ev in events:
                    row: list[float] = []
                    for f in feature_order:
                        v = 0.0
                        if f == "dt":
                            v = float(ev.get("dt", 0.0))
                        elif f == "range":
                            p = ev.get("uwb_payload") or ev.get("vio_payload") or {}
                            v = float(p.get("range", 0.0))
                        elif f == "ax":
                            p = ev.get("imu_payload") or {}
                            v = float(p.get("ax", 0.0))
                        elif f == "ay":
                            p = ev.get("imu_payload") or {}
                            v = float(p.get("ay", 0.0))
                        elif f == "gz":
                            p = ev.get("imu_payload") or {}
                            v = float(p.get("gz", 0.0))
                        elif f == "dx":
                            p = ev.get("vio_payload") or {}
                            v = float(p.get("dx", 0.0))
                        elif f == "dy":
                            p = ev.get("vio_payload") or {}
                            v = float(p.get("dy", 0.0))
                        elif f == "dyaw":
                            p = ev.get("vio_payload") or {}
                            v = float(p.get("dyaw", 0.0))
                        row.append(v)
                    all_feats.append(row)
                n_windows = 0
                for start in range(0, n_events, window_size):
                    window_evts = events[start:start + window_size]
                    if len(window_evts) < 2:
                        continue
                    win_feats = all_feats[start:start + window_size]
                    # Pad to window_size if needed
                    if len(win_feats) < window_size:
                        win_feats = [[0.0] * len(feature_order)] * (window_size - len(win_feats)) + win_feats
                    n_windows += 1
                    last_evt = window_evts[-1]
                    current_modality = last_evt.get("modality", "uwb")
                    if current_modality not in ("uwb", "vio"):
                        current_modality = "uwb"
                    # Build structured window_tensor per factory contract
                    last_row = win_feats[-1]
                    window_tensor = {
                        "feature_order": feature_order,
                        "current_modality": current_modality,
                        "feature_values": last_row,  # 1D list of last row
                        "missing_mask": [0.0] * len(feature_order),  # all observed
                        "dt": float(last_evt.get("dt", 0.0)),
                        "feature_window": [list(r) for r in win_feats],  # 2D nested list
                        "missing_mask_window": [[0] * len(feature_order)] * len(win_feats),
                    }
                    try:
                        out = model.predict_intermediate_tensors(window_tensor)
                    except Exception as e:
                        if n_windows <= 2:
                            print(f"  predict error w={n_windows}: {e}")
                        continue
                    if isinstance(out, dict):
                        px = float(out.get("px", 0.0))
                        py = float(out.get("py", 0.0))
                    elif hasattr(out, "px"):
                        px, py = out.px, out.py
                    else:
                        px, py = 0.0, 0.0
                    preds.append((px, py))
                # GT trajectory at same stride (one GT per window)
                n_pred = len(preds)
                if n_pred == 0:
                    print(f"  {seq_id}: 0 preds from {n_windows} windows (n_events={len(events)})")
                gt = _load_gt_traj(seq_id, n_pred)
                if not gt or len(gt) != n_pred:
                    # Try downsampling
                    continue
                rmse = _sim3_align_2d(preds, gt)
                per_seq_rmse.append(rmse)
                seq_seeds.setdefault(seed, []).append(rmse)
                print(f"  {seq_id}: rmse={rmse:.3f}m (n_pred={n_pred}, n_gt={len(gt)})")
                # Debug: print first prediction
                if len(preds) > 0:
                    print(f"    first pred: {preds[0]}, first_gt: {gt[0]}")
        if per_seq_rmse:
            overall_mean = sum(per_seq_rmse) / len(per_seq_rmse)
            print(f"  overall mean: {overall_mean:.3f}m ({len(per_seq_rmse)} seqs)")
            for seed in sorted(seq_seeds):
                seed_mean = sum(seq_seeds[seed]) / len(seq_seeds[seed])
                print(f"    seed {seed}: {seed_mean:.3f}m ({len(seq_seeds[seed])} seqs)")
        summary[method_name] = seq_seeds

    OUT_FILE = ROOT / "outputs" / "neural_pipeline_25unit_report.json"
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Flatten for compatibility
    flat = {m: [r for srs in seeds.values() for r in srs] for m, seeds in summary.items()}
    OUT_FILE.write_text(json.dumps({"per_method_per_seed": summary, "flat": flat},
                                  indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nNeural pipeline report → {OUT_FILE}")


if __name__ == "__main__":
    main()
