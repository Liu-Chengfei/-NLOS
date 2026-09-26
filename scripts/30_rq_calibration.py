"""脚本：R/Q 标定 — 输出实测数值写入 manifest rq_calib。

用途：
  python scripts/30_rq_calibration.py [--n-samples N] [--seed S] [--output PATH]

功能：
  1. 生成构造 LOS/UWB 样本 → 标定 R ≈ 0.6m
  2. 生成构造 V0 VIO 样本 → 标定 Q ∈ [0.01, 0.03] m/步
  3. 打印冒烟报告 + 写入 manifest.json（写入 rq_calib 字段）

要求（[138]）：
  - R 按 LOS σ≈0.6m 标定
  - Q 按 V0 增量噪声 0.01–0.03m/步 标定
  - 标定结果写入 manifest rq_calib
  - 标定脚本用构造数据单测
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap_project
except ModuleNotFoundError:
    from _bootstrap import bootstrap_project

_LAYOUT = bootstrap_project(__file__)
ROOT = _LAYOUT.root
SRC = _LAYOUT.src

from liquidloc.common.io_utils import write_json
from liquidloc.dataio.rq_calibration import (
    calibrate_rq,
    write_rq_calib_to_manifest,
    print_rq_calib_report,
)


def main() -> int:
    """R/Q 标定脚本入口。"""
    parser = argparse.ArgumentParser(description="R/Q 标定脚本")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="构造样本数 (默认 1000，最小 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260605,
        help="随机种子 (默认 20260605)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 manifest JSON 路径（默认：当前目录下的 rq_calib.json）",
    )
    args = parser.parse_args()

    # 1. 执行标定
    print("[30_rq_calib] 开始 R/Q 标定 | calibrate R (LOS σ≈0.6m) / Q (V0 0.01–0.03m/step)")
    result = calibrate_rq(n_samples=args.n_samples, seed=args.seed)

    # 2. 打印冒烟报告
    print_rq_calib_report(result)

    # 3. 写入 manifest
    output_path = Path(args.output) if args.output else ROOT / "rq_calib.json"
    manifest = {"rq_calib": result, "generated_by": "scripts/30_rq_calibration.py"}
    write_json(output_path, manifest)
    print(f"[30_rq_calib] 结果写入: {output_path}")

    # 4. 验证：断言通过判定
    assert result["r_within_tolerance"], f"R 标定失败: {result['r_uwb_std_m']:.4f}m 超出容差"
    assert result["q_within_tolerance"], f"Q 标定失败: {result['q_vio_combined_std_m']:.4f}m/步 超出范围"
    print("[30_rq_calib] 验证通过 | R/Q 标定完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
