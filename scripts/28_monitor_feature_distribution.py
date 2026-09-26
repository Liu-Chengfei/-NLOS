"""脚本 28：特征分布一致性自动监控

职责：
    读取 split manifest，对每个 split 提取关键输入特征的分布统计（均值/方差/P5/P50/P95），
    输出对比表到 results/feature_dist，检测特征漂移并写入 manifest feature_drift。

要求：
    - 关键特征：UWB range 残差、IMU 高频段、VIO 相对位姿、时间戳特征
    - 漂移阈值：任一分位数偏移 > 2σ 或 KS 检验 p < 0.01
    - 超阈值输出警告并写入 manifest feature_drift

使用方法：
    python scripts/28_monitor_feature_distribution.py \\
        --split-manifest outputs/xxx/split_manifest.json \\
        [--output-dir results/feature_dist]
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap_project
except ModuleNotFoundError:
    from _bootstrap import bootstrap_project

_LAYOUT = bootstrap_project(__file__)
ROOT = _LAYOUT.root
SRC = _LAYOUT.src

from liquidloc.analysis.feature_dist_monitor import (
    DISTRIBUTION_MONITOR_FEATURES,
    monitor_feature_distribution,
)


def _resolve_cli_path(raw: str | None, *, arg_name: str) -> Path | None:
    if raw is None:
        return None
    normalized = raw.strip()
    if not normalized:
        raise ValueError(f"{arg_name} must not be empty")
    return Path(normalized).resolve()


def main(argv: list[str] | None = None) -> int:
    """入口函数"""
    print("[28_feature_dist] 开始 | 特征分布一致性监控", flush=True)

    parser = argparse.ArgumentParser(
        description="监控特征分布一致性，输出对比表到 results/feature_dist"
    )
    parser.add_argument(
        "--split-manifest",
        required=True,
        help="split manifest.json 路径",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认 results/feature_dist",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细日志",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="仅输出警告，抑制常规日志",
    )
    args = parser.parse_args(argv)

    split_manifest_path = _resolve_cli_path(args.split_manifest, arg_name="--split-manifest")
    output_dir = (
        _resolve_cli_path(args.output_dir, arg_name="--output-dir")
        or ROOT / "results" / "feature_dist"
    )

    verbose = not args.quiet

    if verbose:
        print(f"[28_feature_dist] split_manifest: {split_manifest_path}", flush=True)
        print(f"[28_feature_dist] output_dir:     {output_dir}", flush=True)
        print(f"[28_feature_dist] 监控特征: {DISTRIBUTION_MONITOR_FEATURES}", flush=True)

    try:
        result = monitor_feature_distribution(
            split_manifest_path=split_manifest_path,
            output_dir=output_dir,
            verbose=verbose,
        )
    except Exception as exc:
        print(f"[28_feature_dist] 错误: {exc}", flush=True)
        raise

    if verbose:
        print(f"\n[28_feature_dist] 统计报告: {result['statistics_path']}", flush=True)
        print(f"[28_feature_dist] 漂移 manifest: {result['manifest_path']}", flush=True)

    if result["drift_detected"]:
        print(f"\n[28_feature_dist] ⚠ 检测到特征漂移，警告数: {len(result['drift_warnings'])}", flush=True)
        for w in result["drift_warnings"]:
            print(f"[28_feature_dist]   {w}", flush=True)
        return 2  # 非零返回码，供 CI 拦截
    else:
        print("[28_feature_dist] ✓ 无漂移检测到", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())