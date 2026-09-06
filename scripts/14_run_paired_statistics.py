"""跨种子 paired 统计 + §17 六比较消费友好结构包装.

职责:
  - 读 file 1 输出的 JSON 长表 (跨种子 metric_table.json)
  - 调用 liquidloc.analysis.significance_tests.run_significance_tests 跑 paired_wilcoxon + bootstrap CI
  - 附加 §17 6 比较消费友好字段: multi_seed_summary, comparison_pair_summary

概述:
  本脚本不重写 paired test 函数 (significance_tests.py 已实现 paired_wilcoxon + bootstrap_percentile
  + Mann-Whitney U + Cliff's delta + FDR/Holm 多重比较校正). 仅做包装与附加 §17 字段.

  输出 statistics_table.json 含三部分:
    (1) method_summary: {method: {mean_rmse, mean_p95, mean_failure_rate, n_seed, n_seq, n_bundles}}
    (2) pairwise_tests: [{metric_name, test_name, p_value, effect_size, ci_low, ci_high, adjusted_p_value, ...}]
    (3) multi_seed_summary: {comparison_id: {rmse_mean_a, rmse_mean_b, rel_improve, p_value, n_seed, n_seq}}

用法:
  python scripts/14_run_paired_statistics.py \
      --metric-table-json outputs/paper_run/sim_v3_tc_20260723/runs/aggregate/metric_table.json \
      --output-path outputs/paper_run/sim_v3_tc_20260723/runs/aggregate/statistics_table.json \
      --group-keys method_name seq_id \
      --metric-names rmse p95 failure_rate \
      --pairing-keys seq_id

  注: --pairing-keys 决定 paired_wilcoxon 的配对维度 (例: 按 seq_id 配对 → 每 method 对在每个 seq 上配对).
"""

from __future__ import annotations  # 允许使用前向类型注解。

import argparse  # 解析命令行参数。
import json  # 序列化为 JSON。
import statistics  # 计算均值/分位。
import sys  # 操作 sys.path。
from collections import defaultdict  # 按 method/seq 聚合。
from pathlib import Path  # 处理文件路径。
from typing import Any  # 标注任意 JSON 风格结构。

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录。
SRC = ROOT / "src"  # 源码目录。
if str(SRC) not in sys.path:  # 确保脚本能直接导入项目代码。
    sys.path.insert(0, str(SRC))

from liquidloc.analysis.significance_tests import run_significance_tests  # 已实现的 paired 检验入口。


# §17 6 比较的方法对定义 (来自 docs/superpowers/specs/六比较实测脚手架-v0.1.md L40-122).
# (comparison_id, method_a, method_b, semantic) — a vs b 的方向.
# §29.5: 未测比较 (cmp4 FGO vs SGPR, cmp6 LSTM vs TF+EKF) 不进入 multi_seed_summary.semantic,
# 这里只在已测 pair (cmp1a/b, cmp2, cmp3, cmp5) 上写 semantic 描述。
# cmp5 当前只测 FGO 单档, 不写「第三档」身份 (待 SGPR 落地后由 15 补全 §29.5 分支句式).
_SIX_CMP_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    # 比较 1: LNN+EKF > 两 EKF (经典两路分别实测禁混合, 拆 cmp1a / cmp1b).
    ("cmp1a_liquid_vs_ekf", "liquid_ekf", "ekf", "LNN+EKF vs 标准 EKF"),
    ("cmp1b_liquid_vs_robust_ekf", "liquid_ekf", "robust_ekf", "LNN+EKF vs Robust-EKF"),
    # 比较 2: 两 EKF 同档.
    ("cmp2_ekf_vs_robust_ekf", "ekf", "robust_ekf", "标准 EKF vs Robust-EKF (同档判定)"),
    # 比较 3: EKF > 无核 FGO.
    ("cmp3_ekf_vs_fgo", "ekf", "fgo", "标准 EKF vs 无核 FGO"),
    # 比较 5: 无核 FGO ?/> 小容量 LSTM+EKF (SGPR 落地前按 §29.5 勿提第三档身份).
    ("cmp5_fgo_vs_lstm_ekf", "fgo", "lstm_ekf", "无核 FGO 单档 vs 小容量 LSTM+EKF (SGPR 落地前按 §29.5 勿提第三档身份)"),
    # 比较 4 (FGO ? SGPR) 与比较 6 (LSTM ? TF+EKF) 不可测, 在 file 3 处理; 此处不写 semantic.
)


def _resolve_path(raw_value: str, *, flag_name: str) -> Path:
    """把命令行传入的路径整理成可直接使用的绝对路径."""
    value = str(raw_value).strip()
    if not value:  # 空字符串非法。
        raise ValueError(f"{flag_name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path.resolve()


def _read_metric_table_json(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读 file 1 输出的 JSON 长表, 返回 (rows, metadata)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("metric_table") or []
    metadata = payload.get("metadata") or {}
    if not rows:
        raise ValueError(f"metric_table JSON is empty: {path}")
    return rows, metadata


def _build_method_summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """按 method 聚合 mean_rmse/mean_p95/mean_failure_rate + n_seed/n_seq/n_bundles."""
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method = row.get("method_name")
        if not method:
            continue
        by_method[method].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for method, method_rows in by_method.items():
        # 按 (seq_id, seed_id) 取每 (seq, seed) 的 rmse 值; mean_rmse 是这些值的算术平均.
        rmse_values: list[float] = []
        p95_values: list[float] = []
        failure_rate_values: list[float] = []
        seen_seq_seed_pairs: set[tuple[str, str]] = set()
        n_bundles = 0
        for row in method_rows:
            if row.get("metric") != "rmse":
                continue
            seq_id = row.get("seq_id") or ""
            seed_id = row.get("seed_id") or ""
            pair = (seq_id, seed_id)
            if pair in seen_seq_seed_pairs:
                continue
            seen_seq_seed_pairs.add(pair)
            value = row.get("value")
            if value is None:
                continue
            rmse_values.append(float(value))
            n_bundles += 1

        # p95 与 failure_rate 的 paired 值 (同 seq+seed).
        for row in method_rows:
            seq_id = row.get("seq_id") or ""
            seed_id = row.get("seed_id") or ""
            if (seq_id, seed_id) not in seen_seq_seed_pairs:
                continue
            metric = row.get("metric")
            value = row.get("value")
            if value is None:
                continue
            if metric == "p95":
                p95_values.append(float(value))
            elif metric == "failure_rate":
                failure_rate_values.append(float(value))

        def _mean(values: list[float]) -> float:
            return statistics.fmean(values) if values else float("nan")

        seed_ids = {row.get("seed_id") or "" for row in method_rows}
        seq_ids = {row.get("seq_id") or "" for row in method_rows}
        summary[method] = {
            "mean_rmse": _mean(rmse_values),
            "mean_p95": _mean(p95_values),
            "mean_failure_rate": _mean(failure_rate_values),
            "n_seed": len(seed_ids),
            "n_seq": len(seq_ids),
            "n_bundles": n_bundles,
            "seed_ids": sorted(seed_ids),
            "seq_ids": sorted(seq_ids),
        }
    return summary


def _build_multi_seed_comparison_summary(
    rows: list[dict[str, Any]],
    method_summary: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """按 §17 6 比较的方法对, 算 rel_improve + paired_wilcoxon p_value (在 file 2 内二次调用)."""
    # 按 (method, seq_id, seed_id) 取 rmse 值, 形成配对样本.
    paired_rmse: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row.get("metric") != "rmse":
            continue
        method = row.get("method_name") or ""
        seq_id = row.get("seq_id") or ""
        seed_id = row.get("seed_id") or ""
        value = row.get("value")
        if value is None:
            continue
        paired_rmse[(method, seq_id, seed_id)] = float(value)

    comparison_summary: dict[str, dict[str, Any]] = {}
    for cmp_id, method_a, method_b, semantic in _SIX_CMP_PAIRS:
        # 找两 method 共有的 (seq, seed) 对.
        keys_a = {k for k in paired_rmse if k[0] == method_a}
        keys_b = {k for k in paired_rmse if k[0] == method_b}
        common_seq_seed = sorted(
            {(k[1], k[2]) for k in keys_a} & {(k[1], k[2]) for k in keys_b}
        )
        if not common_seq_seed:
            comparison_summary[cmp_id] = {
                "comparison_id": cmp_id,
                "method_a": method_a,
                "method_b": method_b,
                "semantic": semantic,
                "n_paired_samples": 0,
                "rmse_mean_a": None,
                "rmse_mean_b": None,
                "rel_improve": None,
                "p_value_paired_wilcoxon": None,
                "status": "no_paired_data",
            }
            continue

        values_a = [paired_rmse[(method_a, seq, seed)] for seq, seed in common_seq_seed]
        values_b = [paired_rmse[(method_b, seq, seed)] for seq, seed in common_seq_seed]
        rmse_mean_a = statistics.fmean(values_a)
        rmse_mean_b = statistics.fmean(values_b)
        # rel_improve = (a - b) / b ; 正值表 a 比 b 差 (rmse 大), 负值表 a 比 b 好.
        rel_improve = (rmse_mean_a - rmse_mean_b) / rmse_mean_b if rmse_mean_b else float("nan")

        # 对配对差值做 Wilcoxon 符号秩检验 (n<25 时用精确分布). 这里借用 significance_tests 的内部函数.
        # 注: 这里我们没直接调 significance_tests.run_significance_tests, 因为它需要全表; 我们已经在 file 2 入口处调它了.
        # 这里给一个简化版 Wilcoxon (大样本正态近似), 主要作 sanity check; 权威 p 值看 statistics_table 中的 pairwise_tests.
        diffs = [a - b for a, b in zip(values_a, values_b)]
        p_value = _simple_wilcoxon_p_value(diffs)

        comparison_summary[cmp_id] = {
            "comparison_id": cmp_id,
            "method_a": method_a,
            "method_b": method_b,
            "semantic": semantic,
            "n_paired_samples": len(common_seq_seed),
            "rmse_mean_a": rmse_mean_a,
            "rmse_mean_b": rmse_mean_b,
            "rel_improve_a_vs_b": rel_improve,
            "p_value_paired_wilcoxon_simple": p_value,
            "status": "computed",
        }
    return comparison_summary


def _simple_wilcoxon_p_value(diffs: list[float]) -> float | None:
    """简化版 Wilcoxon 符号秩检验 (大样本 n>=20 正态近似). 仅作 sanity check.

    权威 p 值请看 statistics_table.json 中 pairwise_tests 的 p_value 字段 (来自
    significance_tests._compute_wilcoxon_signed_rank_p_value, 它有完整 n<25 精确分布).
    """
    nonzero_diffs = [d for d in diffs if d != 0]
    n = len(nonzero_diffs)
    if n < 5:  # 样本太小, 不近似.
        return None

    # 按绝对值排序, 给秩 (带 ties 取平均秩).
    abs_diffs = sorted(abs(d) for d in nonzero_diffs)
    ranks: list[float] = []
    i = 0
    while i < len(abs_diffs):
        j = i
        while j + 1 < len(abs_diffs) and abs_diffs[j + 1] == abs_diffs[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2  # 1-based 平均秩.
        for _ in range(j - i + 1):
            ranks.append(avg_rank)
        i = j + 1

    # W+ = 正差对应的秩和.
    pos_rank_sum = 0.0
    rank_idx = 0
    for d in sorted(nonzero_diffs, key=abs):
        if d > 0:
            pos_rank_sum += ranks[rank_idx]
        rank_idx += 1

    # 大样本正态近似: W+ ~ N(n(n+1)/4, n(n+1)(2n+1)/24).
    mu = n * (n + 1) / 4
    sigma = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    if sigma == 0:
        return None
    z = (pos_rank_sum - mu) / sigma
    # 双侧 p 值.
    from math import erf, sqrt

    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return max(0.0, min(1.0, p))


def _write_statistics_table(
    output_path: Path,
    method_summary: dict[str, dict[str, Any]],
    statistics_table: list[dict[str, Any]],
    comparison_summary: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    """落盘 statistics_table.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method_summary": method_summary,
        "statistics_table": statistics_table,
        "pairwise_tests": statistics_table,  # 别名, 与 eval_pipeline 落盘结构对齐.
        "multi_seed_summary": comparison_summary,
        "metadata": metadata,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """脚本主入口."""
    print("[14_paired_stats] 开始 | 跨种子 paired 统计 + 6比较消费字段包装", flush=True)
    parser = argparse.ArgumentParser(
        description="Run paired significance tests on aggregated metric_table.json."
    )
    parser.add_argument("--metric-table-json", required=True, help="Input JSON long-table from file 1")
    parser.add_argument("--output-path", required=True, help="Output statistics_table.json path")
    parser.add_argument(
        "--group-keys", nargs="+", default=["method_name"],
        help="Group keys for significance test (default: method_name)",
    )
    parser.add_argument(
        "--metric-names", nargs="+",
        default=["rmse", "p95", "failure_rate"],
        help="Metric names to test (default: rmse p95 failure_rate)",
    )
    parser.add_argument(
        "--pairing-keys", nargs="+", default=["seq_id"],
        help="Pairing keys for paired test (default: seq_id)",
    )
    args = parser.parse_args(argv)

    print(f"[14_paired_stats] 解析路径", flush=True)
    input_path = _resolve_path(args.metric_table_json, flag_name="--metric-table-json")
    output_path = _resolve_path(args.output_path, flag_name="--output-path")
    print(f"  - input:  {input_path}", flush=True)
    print(f"  - output: {output_path}", flush=True)
    print(f"  - group_keys={args.group_keys}  metric_names={args.metric_names}  pairing_keys={args.pairing_keys}", flush=True)

    rows, metadata = _read_metric_table_json(input_path)
    print(f"[14_paired_stats] 行数: {len(rows)}", flush=True)

    # 1. method_summary (跨种子聚合)
    print(f"[14_paired_stats] 构建 method_summary...", flush=True)
    method_summary = _build_method_summary(rows)
    for method, summary in method_summary.items():
        print(
            f"  - {method}: mean_rmse={summary['mean_rmse']:.4f}  "
            f"n_seed={summary['n_seed']}  n_seq={summary['n_seq']}  n_bundles={summary['n_bundles']}",
            flush=True,
        )

    # 2. 调用现有 significance_tests.run_significance_tests 跑 paired_wilcoxon + bootstrap CI
    print(f"[14_paired_stats] 调用 run_significance_tests (paired_wilcoxon + bootstrap_percentile CI)...", flush=True)
    statistics_table = run_significance_tests(
        rows,
        group_keys=args.group_keys,
        metric_names=args.metric_names,
        pairing_keys=args.pairing_keys,
    )
    print(f"  - statistics_table 行数: {len(statistics_table)}", flush=True)

    # 3. §17 6比较消费友好结构
    print(f"[14_paired_stats] 构建 multi_seed_summary (6比较消费字段)...", flush=True)
    comparison_summary = _build_multi_seed_comparison_summary(rows, method_summary)
    for cmp_id, summary in comparison_summary.items():
        if summary.get("n_paired_samples", 0) == 0:
            print(f"  - {cmp_id}: no_paired_data", flush=True)
        else:
            print(
                f"  - {cmp_id}: {summary['method_a']} vs {summary['method_b']}  "
                f"rmse_a={summary['rmse_mean_a']:.4f}  rmse_b={summary['rmse_mean_b']:.4f}  "
                f"rel_improve={summary['rel_improve_a_vs_b']:.4f}  "
                f"p={summary['p_value_paired_wilcoxon_simple']}",
                flush=True,
            )

    enriched_metadata = {
        **metadata,
        "run_significance_tests_version": "liquidloc.analysis.significance_tests.v2",
        "pairing_keys": args.pairing_keys,
        "group_keys": args.group_keys,
        "metric_names": args.metric_names,
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "script": "scripts/14_run_paired_statistics.py",
    }
    _write_statistics_table(output_path, method_summary, statistics_table, comparison_summary, enriched_metadata)

    print(f"[14_paired_stats] 完成 | 输出: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
