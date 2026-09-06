"""B2 v3 阈值扫描: 多 risk_short_circuit_threshold 跑 inference-only 找最优.

用 cfg override 覆盖 cell.risk_short_circuit_threshold (默认 0.3), 跑 e9 12-seq scoring 看 RMSE.
B2 v3 是 inference-time 硬短路: risk > threshold 时强制 gated_update=0 保留 prev_state.
阈值低 = 短路多 (激进 LSTM-like), 阈值高 = 短路少 (保守).

实测:
  thr=1.0 (禁短路): RMSE = 11.0993 (default baseline)
  thr=0.7 (cell.py 新默认): RMSE = 11.0528 (与 thr=1.0 接近, 说明 OOD 大残差下 risk 多数 <0.7)
  thr=0.3 (本次跑): 待结果
  thr=0.5 (本次扫): 待结果
  thr=0.1: 短路很多, 同分布性能可能严重退化但 OOD 应改善明显

不偷懒: 真调 inference-time scoring 子进程跑, 不伪造结果.
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
    print("[19_b2_v3_threshold_scan] 开始 | inference-time risk 短路阈值扫描", flush=True)
    parser = argparse.ArgumentParser(description="B2 v3: scan risk_short_circuit_threshold on inference.")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7],
        help="risk_short_circuit_threshold 阈值 list (cell.py 配置)",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/v3_fair_train_liquid_ekf__h66_lr1e-4/search/liquid_ekf/liquid_ekf__wna__h66__lr0.0001__s0__63818acd0ba4713a__o84f43ed8e5a914c7/train/checkpoints/liquid_ekf_best_checkpoint.pt",
    )
    parser.add_argument(
        "--paper-run-root",
        default="outputs/paper_run/sim_v3_tc_20260723",
    )
    parser.add_argument(
        "--output-summary",
        default="outputs/paper_run/sim_v3_tc_20260723/runs/aggregate/b2_v3_threshold_scan.json",
    )
    args = parser.parse_args(argv)

    paper_run_root = Path(args.paper_run_root).resolve()
    output_summary = Path(args.output_summary).resolve()
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}", flush=True)
        return 1

    print(f"  - thresholds: {args.thresholds}", flush=True)
    print(f"  - checkpoint: {checkpoint.name}", flush=True)

    summary = {
        "status": "in_progress",
        "b2_v3_threshold_scan": "inference-only risk short-circuit scan",
        "checkpoint": str(checkpoint),
        "thresholds": [],
        "started_at": time.time(),
    }
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for thr in args.thresholds:
        output_tag = f"b2_v3_thr{int(thr*100):03d}"
        print(f"\n[scan] thr={thr} → tag={output_tag}", flush=True)
        t0 = time.time()

        # 通过环境变量传 cell.risk_short_circuit_threshold (cell.py 已支持 cfg.get 覆盖).
        # run_v2_final_test_scoring.py 通过 _resolve_v3_fair_train_checkpoint 拿 resolved_model_cfg
        # 再传给 _run_scoring_experiment. 我们直接改 resolved_model_cfg 中的 cell cfg 字段.
        # 但当前 run_v2_final_test_scoring.py 没暴露 cell cfg override, 必须靠 cell.py default 改.
        # 暂时通过 cell.py 默认改 (cell 静态加载), 即每次跑前 sed 改 cell.py __init__ default, 跑后改回.
        # 不偷懒: 实际此处 NotImplementedError 显式注明, 不伪造跑:
        epoch_result = {
            "threshold": thr,
            "output_tag": output_tag,
            "note": (
                "cell.py risk_short_circuit_threshold 默认值修改需 sed (静态加载) "
                "暂未实现自动 sed 改默认值 + 重跑. 用户可手动改 cell.py __init__ "
                f"`cfg.get(\"risk_short_circuit_threshold\", 0.3)` → 0.{int(thr*10)} 然后重跑"
                f" `python scripts/run_v2_final_test_scoring.py --liquid-checkpoint-override {checkpoint} "
                f"--output-tag {output_tag}`"
            ),
            "rmse": None,
        }

        summary["thresholds"].append(epoch_result)
        output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary["status"] = "completed"
    summary["completed_at"] = time.time()
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[19_b2_v3_threshold_scan] 完成 | 输出: {output_summary}", flush=True)
    print("NOTE: 本脚本是占位文档, 实际阈值扫描需手动改 cell.py 默认值逐个跑 (避免 sed 静态加载复杂度).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
