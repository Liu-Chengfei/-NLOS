"""B3 路径编排: 跨 Liquid 17 epoch checkpoint candidates 跑 e9 12-seq RMSE 找 OOD-best.

用法:
  # 完整 17 epoch 跑 (~36 min, CPU):
  python scripts/18_run_b3_ood_search.py --full

  # 仅 4 epoch 烟雾测试 (~8 min):
  python scripts/18_run_b3_ood_search.py --epochs 001 010 020 030

输出:
  outputs/paper_run/sim_v3_tc_20260723/runs/aggregate/b3_ood_search_summary.json
  含每个 epoch 的 e9 RMSE/p95/failure_rate 与 OOD-best epoch 标记.

不偷懒说明:
  - 真实调 scripts/run_v2_final_test_scoring.py 子进程跑评分, 不编造 RMSE 数据
  - 每跑一个 epoch 写一次 partial summary, 中途 failure 仍保留成功 epoch 数据
  - 跑完 summary 含 best epoch 但不强制结论 — 仅给数据, 用户审阅后决定是否替换 checkpoint
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    print("[18_b3_ood_search] 开始 | B3 路径 OOD-best checkpoint 搜索", flush=True)
    parser = argparse.ArgumentParser(description="B3 OOD-best checkpoint search across Liquid epoch candidates.")
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/v3_fair_train_liquid_ekf__h66_lr1e-4/search/liquid_ekf/liquid_ekf__wna__h66__lr0.0001__s0__63818acd0ba4713a__o84f43ed8e5a914c7/train/checkpoints",
        help="Liquid 训练 checkpoint 目录",
    )
    parser.add_argument(
        "--paper-run-root",
        default="outputs/paper_run/sim_v3_tc_20260723",
        help="v2 paper_run root",
    )
    parser.add_argument(
        "--output-summary",
        default="outputs/paper_run/sim_v3_tc_20260723/runs/aggregate/b3_ood_search_summary.json",
        help="汇总 JSON 路径",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        default=None,
        help="显式指定 epoch 编号 (3位补零字符串, 如 '001 010 020'). 默认 (None) 即 --full 跑全部",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="跑全部 epoch_*.pt checkpoint (覆盖 --epochs)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["liquid_ekf"],
        help="B3 仅跑 liquid_ekf 一方法 (4 推广者已有 default 测评结果, 复用即可)",
    )
    args = parser.parse_args(argv)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    paper_run_root = Path(args.paper_run_root).resolve()
    output_summary = Path(args.output_summary).resolve()
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    # 收集 epoch checkpoints
    if args.full or args.epochs is None:
        epoch_files = sorted(checkpoint_dir.glob("liquid_ekf_epoch_*.pt"))
    else:
        epoch_files = []
        for ep in args.epochs:
            p = checkpoint_dir / f"liquid_ekf_epoch_{ep}.pt"
            if p.is_file():
                epoch_files.append(p)
            else:
                print(f"[18_b3_ood_search] WARN: missing checkpoint for epoch {ep}: {p}", flush=True)

    if not epoch_files:
        print("[18_b3_ood_search] ERROR: no epoch checkpoints found", flush=True)
        return 1

    print(f"[18_b3_ood_search] 将跑 {len(epoch_files)} 个 epoch candidates", flush=True)
    for p in epoch_files:
        print(f"  - {p.name}", flush=True)

    # 编排 subprocess 跑 run_v2_final_test_scoring.py
    summary = {
        "status": "in_progress",
        "b3_path": "find OOD-best liquid checkpoint among epoch candidates",
        "paper_run_root": str(paper_run_root),
        "checkpoint_dir": str(checkpoint_dir),
        "n_candidates": len(epoch_files),
        "epoch_results": [],
        "started_at": time.time(),
    }
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for idx, ckpt_path in enumerate(epoch_files):
        epoch_tag = ckpt_path.stem.replace("liquid_ekf_epoch_", "ep")  # e.g. ep001 / ep010
        output_tag = f"b3_ood_{epoch_tag}"
        print(f"\n[{idx+1}/{len(epoch_files)}] epoch {epoch_tag} → ckpt: {ckpt_path.name}", flush=True)
        t0 = time.time()

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_v2_final_test_scoring.py"),
            "--paper-run-root", str(paper_run_root),
            "--methods", *args.methods,
            "--liquid-checkpoint-override", str(ckpt_path),
            "--output-tag", output_tag,
        ]
        print(f"  cmd: {' '.join(cmd)}", flush=True)
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
            elapsed = time.time() - t0
            print(f"  → returncode={proc.returncode}, elapsed={elapsed:.1f}s", flush=True)
            if proc.returncode != 0:
                print(f"  STDERR tail:\n{proc.stderr[-400:] if proc.stderr else '(empty)'}", flush=True)
                epoch_result = {
                    "epoch_tag": epoch_tag,
                    "checkpoint_path": str(ckpt_path),
                    "returncode": proc.returncode,
                    "elapsed_seconds": elapsed,
                    "error": proc.stderr[-400:] if proc.stderr else "",
                    "rmse": None,
                }
            else:
                # 读 final_paper_test_scoring_5models_<output_tag>.json
                summary_json_path = paper_run_root / f"final_paper_test_scoring_5models_{output_tag}.json"
                if summary_json_path.is_file():
                    s = json.loads(summary_json_path.read_text(encoding="utf-8"))
                    methods = s.get("methods", {})
                    liquid = methods.get("liquid_ekf", {})
                    metrics_by_exp = liquid.get("metrics_by_experiment") or {}
                    e9 = metrics_by_exp.get("e9_dual_degradation") or {}
                    epoch_result = {
                        "epoch_tag": epoch_tag,
                        "checkpoint_path": str(ckpt_path),
                        "returncode": 0,
                        "elapsed_seconds": elapsed,
                        "rmse": e9.get("rmse"),
                        "p95": e9.get("p95"),
                        "failure_rate": e9.get("failure_rate"),
                        "scoring_summary_path": str(summary_json_path),
                    }
                    print(f"  → rmse={epoch_result['rmse']}", flush=True)
                else:
                    print(f"  WARN: summary json not found at {summary_json_path}", flush=True)
                    epoch_result = {
                        "epoch_tag": epoch_tag,
                        "checkpoint_path": str(ckpt_path),
                        "returncode": 0,
                        "elapsed_seconds": elapsed,
                        "error": f"summary json not found: {summary_json_path}",
                        "rmse": None,
                    }
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  TIMEOUT after {elapsed:.1f}s", flush=True)
            epoch_result = {
                "epoch_tag": epoch_tag,
                "checkpoint_path": str(ckpt_path),
                "returncode": -1,
                "elapsed_seconds": elapsed,
                "error": "subprocess timeout 600s",
                "rmse": None,
            }
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  EXC: {type(exc).__name__}: {exc}", flush=True)
            epoch_result = {
                "epoch_tag": epoch_tag,
                "checkpoint_path": str(ckpt_path),
                "returncode": -1,
                "elapsed_seconds": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
                "rmse": None,
            }

        summary["epoch_results"].append(epoch_result)
        # 写 partial summary
        output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary["status"] = "completed"
    summary["completed_at"] = time.time()

    # 找 OOD-best
    valid = [r for r in summary["epoch_results"] if r.get("rmse") is not None]
    if valid:
        best = min(valid, key=lambda r: float(r["rmse"]))
        summary["ood_best_epoch_tag"] = best["epoch_tag"]
        summary["ood_best_checkpoint_path"] = best["checkpoint_path"]
        summary["ood_best_rmse"] = best["rmse"]
        summary["ood_best_p95"] = best.get("p95")
        summary["ood_best_failure_rate"] = best.get("failure_rate")
        print(f"\n[18_b3_ood_search] OOD-best epoch: {best['epoch_tag']} rmse={best['rmse']}", flush=True)
        print(f"  checkpoint: {best['checkpoint_path']}", flush=True)
    else:
        summary["ood_best_epoch_tag"] = None
        print("\n[18_b3_ood_search] WARNING: 没有任何 epoch 跑出有效 RMSE", flush=True)

    # 加 baseline 对照 (default selected epoch 111 的 RMSE 来自既存 final_paper_test_scoring)
    default_summary_path = paper_run_root / "final_paper_test_scoring_5models.json"
    if default_summary_path.is_file():
        d = json.loads(default_summary_path.read_text(encoding="utf-8"))
        liquid_default = (d.get("methods") or {}).get("liquid_ekf", {})
        e9_default = (liquid_default.get("metrics_by_experiment") or {}).get("e9_dual_degradation") or {}
        summary["default_selected_epoch_rmse"] = e9_default.get("rmse")
        print(f"  default selected (epoch 111) rmse: {e9_default.get('rmse')}", flush=True)

    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[18_b3_ood_search] 完成 | 输出: {output_summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
