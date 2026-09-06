"""Sim(2) Umeyama 对齐 + RMSE 计算器。

对每个 bundle，按 e9 yaml sim3_alignment=false (2D) 用 Sim(2) Umeyama 对齐到 GT
再算 RMSE。Sim(2) = 旋转+平移+均匀缩放（让 2D 数据正确对齐）。

为什么需要：sim_e9 实验未传 anchor_layout，fusion_runner 走冷启动 (0,0)，
GT 起始 (-11.6, -8.3)，原始 RMSE ~10-30m 是初始位姿误差，不是方法排序问题。
"""
import json
import sys
from pathlib import Path
import bisect
import math

import numpy as np

PRED_DIR = Path(r"E:\异步高NLOS\outputs\e9_full_run\core\predictions")
RAW_ROOT = Path(r"E:\异步高NLOS\data\raw\sim_e9_main")
WARMUP_S = 10.0


def load_gt(seed_id: str, seq_id: str) -> tuple[list[float], list[tuple[float, float]]]:
    p = RAW_ROOT / seed_id / seq_id / "gt.json"
    if not p.is_file():
        return [], []
    rows = json.loads(p.read_text(encoding="utf-8"))
    ts = [float(r["timestamp"]) for r in rows]
    xy = [(float(r["px"]), float(r["py"])) for r in rows]
    return ts, xy


def gt_xy(gt_ts: list[float], gt_xy_list: list, t: float) -> tuple[float, float]:
    if not gt_ts:
        return 0.0, 0.0
    i = bisect.bisect_left(gt_ts, t)
    if i == 0:
        return gt_xy_list[0]
    if i >= len(gt_ts):
        return gt_xy_list[-1]
    if abs(gt_ts[i] - t) > abs(gt_ts[i - 1] - t):
        return gt_xy_list[i - 1]
    return gt_xy_list[i]


def umeyama_2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, float, float]:
    """Sim(2) Umeyama: 找 sR + t 使 src ≈ sR*src + t 与 dst 配准.

    src: (N, 2) 预测点
    dst: (N, 2) GT 点
    返回 (a, b, tx, ty), 即 [a -b; b a] 是 sR 矩阵元素 (a = s*cos, b = s*sin).
    """
    assert src.shape == dst.shape and src.shape[1] == 2
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    # 2x2 cross-cov: Sigma_sdst = sum(src_c^T * dst_c)
    Sigma = (dst_c.T @ src_c) / src.shape[0]  # (2, 2)
    # SVD: A = U D V^T, S = I (2D 旋转), R = U V^T
    U, _, Vt = np.linalg.svd(Sigma)
    S = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    s = np.trace(S @ Sigma) / np.sum(src_c ** 2) if False else 1.0  # sim(2) 等比缩放留个开关
    a = float(R[0, 0])
    b = float(R[0, 1])
    tx = mu_d[0] - (a * mu_s[0] - b * mu_s[1])
    ty = mu_d[1] - (b * mu_s[0] + a * mu_s[1])
    return a, b, tx, ty


def transform(src: np.ndarray, a: float, b: float, tx: float, ty: float) -> np.ndarray:
    out = np.empty_like(src)
    out[:, 0] = a * src[:, 0] - b * src[:, 1] + tx
    out[:, 1] = b * src[:, 0] + a * src[:, 1] + ty
    return out


def eval_seq_aligned(sim_sid: str, bundles: list[Path], warmup: float = WARMUP_S) -> list[dict]:
    # find GT
    for sd in RAW_ROOT.iterdir():
        if not sd.is_dir() or not sd.name.startswith("seed"):
            continue
        p = sd / sim_sid / "gt.json"
        if p.is_file():
            gt_ts, gt_list = load_gt(sd.name, sim_sid)
            break
    else:
        return [{"method": b.stem.split("__")[1], "error": "GT not found"} for b in bundles]

    out = []
    for bp in bundles:
        with open(bp) as f:
            d = json.load(f)
        method = d["method_name"]
        states = d["states"]
        ts = d["timestamps"]
        # warm-up 剔除前对齐
        pred_xy = []
        gt_xy_list = []
        for s, t in zip(states, ts):
            if t < warmup:
                continue
            pred_xy.append((float(s["px"]), float(s["py"])))
            gt_xy_list.append(gt_xy(gt_ts, gt_list, t))
        if len(pred_xy) < 10:
            out.append({"method": method, "error": f"only {len(pred_xy)} frames after warmup"})
            continue
        src = np.array(pred_xy, dtype=float)
        dst = np.array(gt_xy_list, dtype=float)
        a, b, tx, ty = umeyama_2d(src, dst)
        aligned = transform(src, a, b, tx, ty)
        diffs = aligned - dst
        errs2 = (diffs ** 2).sum(axis=1)
        abs_errs = np.sqrt(errs2)
        rmse = float(np.sqrt(errs2.mean()))
        mae = float(abs_errs.mean())
        p50 = float(np.percentile(abs_errs, 50))
        p95 = float(np.percentile(abs_errs, 95))
        failure_rate = float((abs_errs > 1.0).mean())
        out.append({
            "method": method,
            "n_frames": int(len(pred_xy)),
            "rmse_m": round(rmse, 3),
            "mae_m": round(mae, 3),
            "p50_m": round(p50, 3),
            "p95_m": round(p95, 3),
            "failure_rate": round(failure_rate, 3),
            "s": round(math.sqrt(a * a + b * b), 3),  # Sim(2) 缩放
            "rot_deg": round(math.degrees(math.atan2(b, a)), 2),  # 旋转角
        })
    return out


def main():
    bundles = sorted(PRED_DIR.glob("*.json"))
    by_seq: dict[str, list[Path]] = {}
    for b in bundles:
        public_id, method = b.stem.split("__", 1)
        with open(b) as f:
            d = json.load(f)
        sim_sid = d["seq_id"].split("/", 1)[1] if "/" in d["seq_id"] else d["seq_id"]
        by_seq.setdefault(sim_sid, []).append(b)
    # sort by id
    seqs_sorted = sorted(by_seq.keys())

    print(f"# E9 Sim(2)-aligned RMSE report (warmup={WARMUP_S}s, {len(seqs_sorted)} sim seqs)")
    print()
    summary_by_method: dict[str, list[tuple[str, float]]] = {}
    for sim_sid in seqs_sorted:
        results = eval_seq_aligned(sim_sid, by_seq[sim_sid])
        print(f"## {sim_sid}")
        for r in results:
            if "error" in r:
                print(f"  {r['method']:18s}  ERROR: {r['error']}")
                continue
            print(f"  {r['method']:18s}  RMSE={r['rmse_m']:6.3f}  MAE={r['mae_m']:6.3f}  P50={r['p50_m']:6.3f}  P95={r['p95_m']:6.3f}  fail={r['failure_rate']:.3f}  N={r['n_frames']}  Sim(2): s={r['s']}, rot={r['rot_deg']}°")
            summary_by_method.setdefault(r["method"], []).append((sim_sid, r["rmse_m"]))
        print()
    print("# Per-method aggregate (mean / median over sim seqs)")
    for method in sorted(summary_by_method.keys()):
        rows = summary_by_method[method]
        rmses = [r for _, r in rows]
        rmses_sorted = sorted(rmses)
        n = len(rmses_sorted)
        mean = sum(rmses_sorted) / n
        median = rmses_sorted[n // 2] if n % 2 == 1 else (rmses_sorted[n // 2 - 1] + rmses_sorted[n // 2]) / 2
        print(f"  {method:18s}  N={n}  mean={mean:.3f}  median={median:.3f}  min={rmses_sorted[0]:.3f}  max={rmses_sorted[-1]:.3f}")
    print()
    print("# 排序 (按 mean RMSE 升序)")
    sorted_methods = sorted(summary_by_method.keys(), key=lambda m: sum(r for _, r in summary_by_method[m]) / len(summary_by_method[m]))
    for m in sorted_methods:
        mean = sum(r for _, r in summary_by_method[m]) / len(summary_by_method[m])
        print(f"  {m:18s}  mean={mean:.3f}")


if __name__ == "__main__":
    main()
