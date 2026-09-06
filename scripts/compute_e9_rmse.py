"""最小化 e9 实跑 RMSE 计算器。

读 outputs/e9_full_run/core/predictions/*.json 预测包 + data/raw/sim_e9_main 的 GT，
按时间戳最近邻匹配，按 5 方法 × seq 输出 RMSE / mae / P95 / failure_rate（>1m）。
"""
import json
import sys
from pathlib import Path
import bisect
import math

PRED_DIR = Path(r"E:\异步高NLOS\outputs\e9_full_run\core\predictions")
RAW_ROOT = Path(r"E:\异步高NLOS\data\raw\sim_e9_main")
WARMUP_S = 10.0  # warm-up 剔除（手册 §0.4 Part 0: 前 10s 不进 RMSE）


def load_gt(seed_id: str, seq_id: str) -> tuple[list[float], list[tuple[float, float]]]:
    """从 raw 读 GT，ts, (px, py) 数组。"""
    p = RAW_ROOT / seed_id / seq_id / "gt.json"
    if not p.is_file():
        return [], []
    rows = json.loads(p.read_text(encoding="utf-8"))
    ts = [float(r["timestamp"]) for r in rows]
    xy = [(float(r["px"]), float(r["py"])) for r in rows]
    return ts, xy


def gt_at(gt_ts: list[float], gt_xy: list, t: float) -> tuple[float, float]:
    """二分查找最近 GT（闭包写法，单调 ts 假设已通过 build 验证）。"""
    if not gt_ts:
        return 0.0, 0.0
    i = bisect.bisect_left(gt_ts, t)
    if i == 0:
        return gt_xy[0]
    if i >= len(gt_ts):
        return gt_xy[-1]
    # 选更近的
    if abs(gt_ts[i] - t) > abs(gt_ts[i - 1] - t):
        return gt_xy[i - 1]
    return gt_xy[i]


def eval_seq(seq_id: str, bundles: list[Path]) -> list[dict]:
    """对 1 个 seq 多个 method bundle 计算 RMSE / mae / P95 / failure_rate。"""
    # 从 bundle 路径解析 method 和 seq_id
    # 格式: public_NN__method.json
    out = []
    # 找 GT
    # seq_id like sim_curve_01 → 找 seed0
    candidates = list(RAW_ROOT.iterdir())
    gt = None
    seed_id = None
    for sd in candidates:
        if not sd.is_dir() or not sd.name.startswith("seed"):
            continue
        p = sd / seq_id / "gt.json"
        if p.is_file():
            gt_ts, gt_xy = load_gt(sd.name, seq_id)
            gt = (gt_ts, gt_xy)
            seed_id = sd.name
            break
    if gt is None:
        return [{"method": b.stem.split("__")[1], "error": f"GT not found for {seq_id}"} for b in bundles]
    gt_ts, gt_xy = gt

    for bp in bundles:
        with open(bp) as f:
            d = json.load(f)
        method = d["method_name"]
        states = d["states"]
        ts = d["timestamps"]
        # warm-up 剔除
        errs2 = []
        for s, t in zip(states, ts):
            if t < WARMUP_S:
                continue
            gx, gy = gt_at(gt_ts, gt_xy, t)
            ex = float(s["px"]) - gx
            ey = float(s["py"]) - gy
            errs2.append(ex * ex + ey * ey)
        if not errs2:
            out.append({"method": method, "error": "no valid frames after warmup"})
            continue
        rmse = math.sqrt(sum(errs2) / len(errs2))
        abs_errs = [math.sqrt(e) for e in errs2]
        mae = sum(abs_errs) / len(abs_errs)
        abs_errs.sort()
        p50 = abs_errs[len(abs_errs) // 2]
        p95 = abs_errs[int(len(abs_errs) * 0.95)]
        failure_rate = sum(1 for e in abs_errs if e > 1.0) / len(abs_errs)
        out.append({
            "method": method,
            "n_frames": len(errs2),
            "rmse_m": round(rmse, 3),
            "mae_m": round(mae, 3),
            "p50_m": round(p50, 3),
            "p95_m": round(p95, 3),
            "failure_rate": round(failure_rate, 3),
        })
    return out


def main():
    # 列出所有 18 跑通的 seq_id
    bundles = sorted(PRED_DIR.glob("*.json"))
    by_seq = {}
    for b in bundles:
        stem = b.stem
        if "__" in stem:
            public_id, method = stem.split("__", 1)
            by_seq.setdefault(public_id, []).append((method, b))

    # seq_id mapping: 跑通 18 个 public_NN 对应哪些 sim seq
    # 从 bundle 里读 seq_id 字段
    seq_to_bundles = {}
    for public_id in sorted(by_seq.keys()):
        for method, bp in by_seq[public_id]:
            with open(bp) as f:
                d = json.load(f)
            sid = d["seq_id"]  # 'seed0/sim_curve_01'
            # strip 'seed0/' prefix
            sim_sid = sid.split("/", 1)[1] if "/" in sid else sid
            seq_to_bundles.setdefault(sim_sid, []).append((public_id, method, bp))

    print(f"# E9 minimal RMSE report (warmup={WARMUP_S}s, sim seqs={len(seq_to_bundles)})")
    print()

    # 按 seq 输出
    summary_by_method = {}  # method → list[rmse_m]
    for sim_sid in sorted(seq_to_bundles.keys()):
        bundles = [t[2] for t in seq_to_bundles[sim_sid]]
        results = eval_seq(sim_sid, bundles)
        print(f"## {sim_sid}")
        for r in results:
            if "error" in r:
                print(f"  {r['method']:18s}  ERROR: {r['error']}")
                continue
            print(f"  {r['method']:18s}  RMSE={r['rmse_m']:6.3f}  MAE={r['mae_m']:6.3f}  P50={r['p50_m']:6.3f}  P95={r['p95_m']:6.3f}  fail={r['failure_rate']:.3f}  N={r['n_frames']}")
            summary_by_method.setdefault(r["method"], []).append((sim_sid, r["rmse_m"]))
        print()

    print("# Per-method aggregate (mean / median RMSE over sim seqs)")
    for method in sorted(summary_by_method.keys()):
        rows = summary_by_method[method]
        rmses = [r for _, r in rows]
        rmses.sort()
        n = len(rmses)
        mean = sum(rmses) / n
        median = rmses[n // 2] if n % 2 == 1 else (rmses[n // 2 - 1] + rmses[n // 2]) / 2
        print(f"  {method:18s}  N={n}  mean={mean:.3f}  median={median:.3f}  min={rmses[0]:.3f}  max={rmses[-1]:.3f}")

    # 排序
    print()
    print("# 排序 (按 mean RMSE 升序)")
    sorted_methods = sorted(summary_by_method.keys(), key=lambda m: sum(r for _, r in summary_by_method[m]) / len(summary_by_method[m]))
    for m in sorted_methods:
        mean = sum(r for _, r in summary_by_method[m]) / len(summary_by_method[m])
        print(f"  {m:18s}  mean={mean:.3f}")


if __name__ == "__main__":
    main()
