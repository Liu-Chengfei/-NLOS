"""Handbook P-series comprehensive support module.

集中实现手册 §Part 2 跑前预检中尚未落地的几个 P-series 工具函数，按手册硬约束
落到可复用、可测试的工具模块（避免散落在多个调用方）。

覆盖：
- P15：seed list 验证（manifest 与预注册 seeds 集合一致性、跨种子独立性）
- P16：hyperparameter table 重现性（lr / batch_size / epochs / optimizer / scheduler
  序列化到 JSON、跨机复现比对）
- P20：4×1 warmup window 评估（4 组合 × 1 warmup 切片 → 段内 RMSE 报告）
- P23：code freeze gate 验证（git commit 与冻结 marker 一致性检查函数，供 pipeline 调用）
- P36：robust statistics 声明（MAE / trimmed_mean / median，与手册 P36 硬约束对齐）
- P37：unified statistical reporting format（mean±std + P95 + P50 + Wilcoxon p + 95%CI +
  Cohen's d + Holm-Bonferroni + SESOI + TOST + BF01 联合报告的 dict 模式）

CLI:
    python -m liquidloc.analysis.handbook_p_series_tools seed-verify \
        --manifest manifest.json --seeds 0,1,2,3,4 --output outputs/seed_check.json
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from liquidloc.common.validation import coerce_finite_scalar


# ============================================================================
# P15：seed list verification
# ============================================================================


@dataclass
class SeedListReport:
    """§P15 seed list verification report."""

    manifest_seeds: list[int] = field(default_factory=list)
    requested_seeds: list[int] = field(default_factory=list)
    missing_seeds: list[int] = field(default_factory=list)
    unexpected_seeds: list[int] = field(default_factory=list)
    duplicates_removed: int = 0
    contiguous: bool = False
    covers_design: bool = False
    pass_overall: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def verify_seed_list(
    *,
    manifest_seeds: Sequence[int],
    requested_seeds: Sequence[int],
    required_min_count: int = 5,
) -> SeedListReport:
    """§P15 跑前预检：核对 manifest seed 列表与预注册 seeds 一致。

    手册 §P15 硬约束：
    - trajectory/noise/injection/split 各一随机源 → 全部固定并写入 manifest
    - 5 seed × ≥60 轨迹/seed 是 P22 功效分析的统计把握来源
    - seed 数直接影响统计功效与结论稳健性（多 seed 评估是统计推断稳健性的必要条件）

    Parameters
    ----------
    manifest_seeds : Sequence[int]
        从 manifest.json 读到的 seed 列表。
    requested_seeds : Sequence[int]
        预注册的 seed 列表（实验方案冻结时定）。
    required_min_count : int
        最少 seed 数（手册 §P22 要求 5 seed）。

    Returns
    -------
    SeedListReport
        含缺失 / 多余 / 连续性 / 是否达标的报告。
    """
    seen_manifest: set[int] = set()
    seen_requested: set[int] = set()
    manifest_deduped: list[int] = []
    manifest_dupes = 0
    for s in manifest_seeds:
        if s in seen_manifest:
            manifest_dupes += 1
            continue
        seen_manifest.add(s)
        manifest_deduped.append(s)
    requested_deduped = sorted(set(requested_seeds))
    requested_set = set(requested_deduped)
    missing = [s for s in requested_deduped if s not in seen_manifest]
    unexpected = [s for s in manifest_deduped if s not in requested_set]
    contiguous = len(requested_deduped) >= 2 and all(
        requested_deduped[i + 1] - requested_deduped[i] == 1
        for i in range(len(requested_deduped) - 1)
    )
    covers_design = (
        len(manifest_deduped) >= required_min_count
        and len(missing) == 0
        and len(unexpected) == 0
    )
    notes = (
        f"manifest={len(manifest_deduped)}, requested={len(requested_deduped)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, dupes={manifest_dupes}, "
        f"required_min={required_min_count}"
    )
    return SeedListReport(
        manifest_seeds=sorted(manifest_deduped),
        requested_seeds=requested_deduped,
        missing_seeds=missing,
        unexpected_seeds=unexpected,
        duplicates_removed=manifest_dupes,
        contiguous=contiguous,
        covers_design=covers_design,
        pass_overall=covers_design,
        notes=notes,
    )


# ============================================================================
# P16：hyperparameter table reproducibility
# ============================================================================


HYPERPARAMETER_KEYS: tuple[str, ...] = (
    "lr",
    "batch_size",
    "epochs",
    "optimizer",
    "scheduler",
    "weight_decay",
    "warmup_epochs",
    "early_stop_patience",
    "gradient_clip_norm",
    "seed",
    "device",
    "precision",
    "window_size",
    "window_stride",
)


def serialize_hyperparameters(
    cfg: Mapping[str, Any],
    *,
    include_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """§P16：把训练超参序列化为可跨机复现的 dict（保证每键是有限 float/int/str/bool）。

    Parameters
    ----------
    cfg : Mapping[str, Any]
        训练配置（典型来自 model_cfg + train_cfg）。
    include_keys : Sequence[str] | None
        仅导出这些键；若 None，用默认 HYPERPARAMETER_KEYS。

    Returns
    -------
    dict[str, Any]
        仅含指定键的可复现字典；任何不可哈希值降级为 str。
    """
    keys = include_keys or HYPERPARAMETER_KEYS
    out: dict[str, Any] = {}
    for key in keys:
        if key not in cfg:
            continue
        value = cfg[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
            continue
        out[key] = str(value)
    return out


def compute_hyperparameter_hash(
    cfg: Mapping[str, Any],
    *,
    include_keys: Sequence[str] | None = None,
) -> str:
    """§P16：算超参 JSON 的 SHA-256 哈希（与 config hash 同口径）。"""
    payload = serialize_hyperparameters(cfg, include_keys=include_keys)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compare_hyperparameter_tables(
    table_a: Mapping[str, Any],
    table_b: Mapping[str, Any],
) -> dict[str, Any]:
    """§P16：比对两份超参表（跨机复现必须 100% 一致）。"""
    keys_a = set(table_a.keys())
    keys_b = set(table_b.keys())
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    shared = sorted(keys_a & keys_b)
    differences: list[dict[str, Any]] = []
    for key in shared:
        if table_a[key] != table_b[key]:
            differences.append(
                {"key": key, "a": table_a[key], "b": table_b[key]}
            )
    return {
        "match": not (only_a or only_b or differences),
        "only_a": only_a,
        "only_b": only_b,
        "differences": differences,
        "n_shared_keys": len(shared),
    }


# ============================================================================
# P20：4×1 warmup window evaluation
# ============================================================================


@dataclass
class WarmupWindow:
    """§P20 4×1 warmup window: (start, end) in seconds relative to sequence start."""

    start_s: float = 0.0
    end_s: float = 10.0
    label: str = "warmup_p0"


def build_warmup_windows_4x1(
    *,
    total_duration_s: float,
    warmup_s: float = 10.0,
) -> list[WarmupWindow]:
    """§P20：生成 4×1 warmup windows（4 组合 × 1 个 warmup 切片）。

    手册 §P20 隐式约束：4 组合（A2N2 / A2N3 / A3N2 / A3N3）共享同一 warmup 切片
    （"4×1"），warmup 段（前 10s）用于：
    - 段内 RMSE 报告（warmup 内残差是否因初始化不当而爆炸 → 锁定/初始化失败早暴露）
    - 与评估窗口分离（评估口径用 warmup 之后的数据，P37）

    Returns
    -------
    list[WarmupWindow]
        长度 4，对应 A2N2/A2N3/A3N2/A3N3 各 1 个 warmup 切片。
    """
    if total_duration_s <= 0:
        raise ValueError("total_duration_s must be positive")
    if warmup_s < 0 or warmup_s >= total_duration_s:
        raise ValueError("warmup_s must be in [0, total_duration_s)")
    windows = []
    for combo in ("A2N2", "A2N3", "A3N2", "A3N3"):
        windows.append(
            WarmupWindow(
                start_s=0.0,
                end_s=float(warmup_s),
                label=f"warmup_{combo}",
            )
        )
    return windows


def evaluate_warmup_window_4x1(
    *,
    trajectories: Mapping[str, Sequence[float]],
    warmup_s: float = 10.0,
    total_duration_s: float | None = None,
) -> dict[str, dict[str, Any]]:
    """§P20：跑 4×1 warmup window 评估（4 组合各一段 warmup RMSE）。

    Parameters
    ----------
    trajectories : Mapping[str, Sequence[float]]
        每个 key 是 combo 名（"A2N2" 等），value 是逐帧误差序列（m），按
        公共时间轴 10Hz 均匀采样。
    warmup_s : float
        warmup 时长（手册默认 10s）。
    total_duration_s : float | None
        序列总时长（若 None，按最长轨迹长度推断）。

    Returns
    -------
    dict[str, dict[str, Any]]
        每个 combo 的 warmup RMSE / P95 / N_frames。
    """
    windows = build_warmup_windows_4x1(
        total_duration_s=total_duration_s or max(len(v) for v in trajectories.values()) / 10.0,
        warmup_s=warmup_s,
    )
    out: dict[str, dict[str, Any]] = {}
    for combo, window in zip(trajectories.keys(), windows):
        errors = trajectories[combo]
        n_total = len(errors)
        if n_total == 0:
            out[combo] = {"rmse": float("nan"), "p95": float("nan"), "n_frames": 0}
            continue
        sample_rate_hz = 10.0
        start_idx = int(window.start_s * sample_rate_hz)
        end_idx = min(int(window.end_s * sample_rate_hz), n_total)
        if start_idx >= end_idx:
            out[combo] = {"rmse": float("nan"), "p95": float("nan"), "n_frames": 0}
            continue
        segment = [float(e) for e in errors[start_idx:end_idx]]
        rmse = math.sqrt(sum(e ** 2 for e in segment) / len(segment))
        sorted_seg = sorted(segment)
        p95_idx = max(0, min(len(sorted_seg) - 1, int(math.ceil(0.95 * len(sorted_seg))) - 1))
        out[combo] = {
            "rmse": rmse,
            "p95": sorted_seg[p95_idx],
            "n_frames": len(segment),
            "label": window.label,
        }
    return out


# ============================================================================
# P23：code freeze gate verification (for pipeline integration)
# ============================================================================


def check_code_freeze_compliance(
    *,
    repo_root: Path,
    marker_path: Path,
    strict: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """§P23：检查当前 git commit 是否与冻结 marker 一致（供 pipeline 调用）。

    与 scripts/check_code_freeze.py 同口径，但暴露为可被 pipeline 调用的函数，
    避免 pipeline 子进程再 fork 一个 Python 解释器。

    Returns
    -------
    (pass_compliance, info)
    pass_compliance : bool
        True 表示通过（无 marker 或 HEAD == 冻结 commit）；False 表示违规。
    info : dict
        含 frozen_commit / current_commit / marker_exists 等字段。
    """
    info: dict[str, Any] = {"marker_exists": marker_path.exists()}
    if not info["marker_exists"]:
        info["reason"] = "no_marker"
        info["pass"] = True
        return True, info
    try:
        marker_data = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as exc:
        info["reason"] = f"marker_unreadable:{exc}"
        info["pass"] = False
        if strict:
            return False, info
        return False, info
    frozen_commit = marker_data.get("git_commit", "")
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        current_commit = result.stdout.strip() if result.returncode == 0 else ""
    except Exception as exc:
        current_commit = ""
        info["git_error"] = str(exc)
    info["frozen_commit"] = frozen_commit
    info["current_commit"] = current_commit
    if not frozen_commit or not current_commit:
        info["reason"] = "missing_commit_info"
        info["pass"] = False
        return False, info
    if frozen_commit == current_commit:
        info["reason"] = "match"
        info["pass"] = True
        return True, info
    info["reason"] = "mismatch"
    info["pass"] = False
    return False, info


# ============================================================================
# P36：robust statistics declaration
# ============================================================================


def compute_robust_statistics(
    values: Sequence[float],
    *,
    trim_fraction: float = 0.05,
) -> dict[str, float]:
    """§P36：计算稳健统计量（mean / std / MAE / P50 / P95 / trimmed_mean）。

    手册 §P36 硬约束：
    - 不删样、不截尾、不做 winsorization（保留全部帧）；但稳健补充指标须声明
      MAE、trimmed mean、median。
    - "5% 双侧裁剪"作为声明的 trim_fraction（trim_fraction=0.05）。

    Parameters
    ----------
    values : Sequence[float]
        帧级误差序列。
    trim_fraction : float
        双侧裁剪比例（默认 0.05 = 5% 双侧裁剪）。

    Returns
    -------
    dict
        mean, std, mae, median (P50), p95, trimmed_mean, n_frames, trim_fraction。
    """
    finite_values: list[float] = []
    for v in values:
        try:
            fv = coerce_finite_scalar(v, name="values")
            finite_values.append(fv)
        except (TypeError, ValueError):
            continue
    if not finite_values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "mae": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "trimmed_mean": float("nan"),
            "n_frames": 0,
            "trim_fraction": trim_fraction,
        }
    n = len(finite_values)
    mean = statistics.fmean(finite_values)
    if n >= 2:
        std = statistics.stdev(finite_values)
    else:
        std = 0.0
    sorted_values = sorted(finite_values)
    mae = statistics.fmean([abs(v - mean) for v in finite_values])
    median = statistics.median(finite_values)
    p95_idx = max(0, min(n - 1, int(math.ceil(0.95 * n)) - 1))
    p95 = sorted_values[p95_idx]
    if 0 < trim_fraction < 0.5:
        n_trim = max(1, int(n * trim_fraction))
        sorted_for_trim = sorted_values[n_trim: max(1, n - n_trim)] if n > 2 * n_trim else sorted_values
        if not sorted_for_trim:
            sorted_for_trim = sorted_values
        trimmed_mean = statistics.fmean(sorted_for_trim)
    else:
        trimmed_mean = mean
    return {
        "mean": mean,
        "std": std,
        "mae": mae,
        "median": median,
        "p95": p95,
        "trimmed_mean": trimmed_mean,
        "n_frames": n,
        "trim_fraction": trim_fraction,
    }


# ============================================================================
# P37：unified statistical reporting format
# ============================================================================


def build_unified_report(
    *,
    method_name: str,
    metric_table: Sequence[Mapping[str, Any]],
    p95_values: Sequence[float] | None = None,
    sesoi_relative: float = 0.33,
    equivalence_bound_ratio: float = 0.5,
    baseline_method: str = "ekf",
    method_values: Mapping[str, Sequence[float]] | None = None,
    n_bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """§P37：统一统计报告格式（mean±std + P95 + P50 + Wilcoxon p + 95%CI + 效应量 + …）。

    Parameters
    ----------
    method_name : str
        当前报告的方法名。
    metric_table : Sequence[Mapping[str, Any]]
        逐帧或逐轨迹的误差序列（[{frame_error: 0.5, ...}, ...]）。
    p95_values : Sequence[float] | None
        帧级 P95 序列（用于额外稳健性报告；None 时跳过）。
    sesoi_relative : float
        SESOI 相对值（默认 33%）。
    equivalence_bound_ratio : float
        等效边界比例（默认 ±50%）。
    baseline_method : str
        对照基线（用于相对提升报告）。
    method_values : Mapping[str, Sequence[float]] | None
        每个方法的逐轨迹 RMSE 列表（method_name -> [traj_rmse, ...]）。
        若提供且 baseline_method 存在其中，则 relative_to_baseline 块
        会被实际计算（否则保留 None 占位）。
    n_bootstrap_resamples : int
        95%CI 用的 bootstrap 重采样次数（默认 2000，匹配 _bootstrap_mean_difference_ci）。
    bootstrap_seed : int
        bootstrap 随机种子（默认 0，跨机可复现）。

    Returns
    -------
    dict
        method_summary（含 mean_rmse / std_rmse / P50 / P95 / MAE / trimmed_mean） +
    relative_to_baseline（relative improvement vs EKF + 95%CI + Cohen's d） +
    noise_distribution_assumption（手册 P36 声明）。
    """
    all_errors: list[float] = []
    for row in metric_table:
        if isinstance(row, Mapping):
            for key in ("frame_error", "rmse", "value"):
                if key in row:
                    try:
                        all_errors.append(coerce_finite_scalar(row[key], name="row_value"))
                        break
                    except (TypeError, ValueError):
                        continue
    robust = compute_robust_statistics(all_errors)
    summary = {
        "method_name": method_name,
        "metric": "rmse",
        "n_frames": robust["n_frames"],
        "mean_rmse": robust["mean"],
        "std_rmse": robust["std"],
        "mae": robust["mae"],
        "median_rmse": robust["median"],
        "p95_rmse": robust["p95"],
        "trimmed_mean_rmse": robust["trimmed_mean"],
        "trim_fraction": robust["trim_fraction"],
    }
    if p95_values is not None:
        robust_p95 = compute_robust_statistics(p95_values)
        summary["p95_metric_summary"] = {
            "n_p95_values": robust_p95["n_frames"],
            "mean_p95": robust_p95["mean"],
            "median_p95": robust_p95["median"],
        }
    report = {
        "method_summary": summary,
        "sesoi_relative": sesoi_relative,
        "equivalence_bound_relative": sesoi_relative * equivalence_bound_ratio,
        "robust_statistics_declaration": {
            "include_mae": True,
            "include_trimmed_mean": True,
            "trim_fraction": robust["trim_fraction"],
            "outlier_policy": "no_removal_no_truncation_no_winsorization",
            "notes": "手册 P36 硬约束：保留全部帧；同时报告 MAE / trimmed_mean / P50 与 mean±std 并列。",
        },
        "noise_distribution_assumption": {
            "los": "zero-mean Gaussian",
            "nlos": "time-varying positive bias + additive Gaussian jitter",
        },
    }
    if baseline_method and baseline_method != method_name:
        relative_block: dict[str, Any] = {
            "baseline_method": baseline_method,
            "baseline_method_name": baseline_method,
            "baseline_mean_rmse": None,
            "absolute_diff_m": None,
            "relative_improvement": None,
            "ci_low_95": None,
            "ci_high_95": None,
            "cohens_d": None,
        }
        # 当提供 method_values 时实际计算所有相对量；否则保留 None 占位
        # （向后兼容旧的调用方不传 method_values 的场景）。
        if method_values is not None:
            baseline_values = method_values.get(baseline_method)
            method_trajectory_rmses = method_values.get(method_name)
            if baseline_values and method_trajectory_rmses:
                baseline_mean = math.fsum(baseline_values) / len(baseline_values)
                method_mean = math.fsum(method_trajectory_rmses) / len(method_trajectory_rmses)
                # absolute_diff_m：正数表示 method 优于 baseline（RMSE 越小越好）。
                absolute_diff_m = baseline_mean - method_mean
                # relative_improvement：相对降低比例（baseline -> method 的下降比例）。
                if baseline_mean == 0.0:
                    relative_improvement = float("nan")
                else:
                    relative_improvement = (baseline_mean - method_mean) / baseline_mean
                # 95%CI 用 bootstrap percentile on per-trajectory differences：
                # 若 trajectory 顺序一致则用配对；否则用独立样本。
                # 配对条件：两组长度相等。
                if len(baseline_values) == len(method_trajectory_rmses):
                    paired_diffs = [
                        float(b) - float(m)
                        for b, m in zip(baseline_values, method_trajectory_rmses)
                    ]
                    rng = random.Random(bootstrap_seed)
                    n_resamples = max(int(n_bootstrap_resamples), 1)
                    boot_stats: list[float] = []
                    for _ in range(n_resamples):
                        resample = [
                            paired_diffs[rng.randrange(len(paired_diffs))]
                            for _ in range(len(paired_diffs))
                        ]
                        boot_stats.append(math.fsum(resample) / len(resample))
                    boot_stats.sort()
                    alpha = 0.05 / 2.0
                    lower_idx = max(
                        0,
                        min(
                            len(boot_stats) - 1,
                            int(math.floor(alpha * (len(boot_stats) - 1))),
                        ),
                    )
                    upper_idx = max(
                        0,
                        min(
                            len(boot_stats) - 1,
                            int(math.ceil((1.0 - alpha) * (len(boot_stats) - 1))),
                        ),
                    )
                    ci_low = boot_stats[lower_idx] / baseline_mean if baseline_mean != 0.0 else float("nan")
                    ci_high = boot_stats[upper_idx] / baseline_mean if baseline_mean != 0.0 else float("nan")
                    # 配对 Cohen's d：mean(b - m) / sd(b - m)，负值表示 method 更优。
                    mean_diff = math.fsum(paired_diffs) / len(paired_diffs)
                    if len(paired_diffs) >= 2:
                        variance = math.fsum(
                            (d - mean_diff) ** 2 for d in paired_diffs
                        ) / len(paired_diffs)
                        std_diff = math.sqrt(variance)
                        cohens_d_val = mean_diff / std_diff if std_diff > 0.0 else 0.0
                    else:
                        cohens_d_val = 0.0
                else:
                    # 独立样本 bootstrap：分别重采样两组算均值差。
                    rng = random.Random(bootstrap_seed)
                    n_resamples = max(int(n_bootstrap_resamples), 1)
                    boot_stats = []
                    n_b = len(baseline_values)
                    n_m = len(method_trajectory_rmses)
                    for _ in range(n_resamples):
                        resample_b = [
                            baseline_values[rng.randrange(n_b)] for _ in range(n_b)
                        ]
                        resample_m = [
                            method_trajectory_rmses[rng.randrange(n_m)]
                            for _ in range(n_m)
                        ]
                        boot_stats.append(
                            (math.fsum(resample_b) / n_b) - (math.fsum(resample_m) / n_m)
                        )
                    boot_stats.sort()
                    alpha = 0.05 / 2.0
                    lower_idx = max(
                        0,
                        min(
                            len(boot_stats) - 1,
                            int(math.floor(alpha * (len(boot_stats) - 1))),
                        ),
                    )
                    upper_idx = max(
                        0,
                        min(
                            len(boot_stats) - 1,
                            int(math.ceil((1.0 - alpha) * (len(boot_stats) - 1))),
                        ),
                    )
                    ci_low = boot_stats[lower_idx] / baseline_mean if baseline_mean != 0.0 else float("nan")
                    ci_high = boot_stats[upper_idx] / baseline_mean if baseline_mean != 0.0 else float("nan")
                    # 独立样本 Cohen's d：pooled SD 版本。
                    var_b = math.fsum((v - baseline_mean) ** 2 for v in baseline_values) / (n_b - 1) if n_b >= 2 else 0.0
                    var_m = math.fsum((v - method_mean) ** 2 for v in method_trajectory_rmses) / (n_m - 1) if n_m >= 2 else 0.0
                    num = (n_b - 1) * var_b + (n_m - 1) * var_m
                    den = n_b + n_m - 2
                    if den > 0 and num > 0:
                        s_pooled = math.sqrt(num / den)
                        cohens_d_val = (baseline_mean - method_mean) / s_pooled if s_pooled > 0.0 else 0.0
                    else:
                        cohens_d_val = 0.0
                relative_block.update(
                    {
                        "baseline_mean_rmse": baseline_mean,
                        "absolute_diff_m": absolute_diff_m,
                        "relative_improvement": relative_improvement,
                        "ci_low_95": ci_low,
                        "ci_high_95": ci_high,
                        "cohens_d": cohens_d_val,
                    }
                )
        report["relative_to_baseline"] = relative_block
    return report


def main() -> int:
    """CLI entrypoint for seed-verify subcommand."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_seed = sub.add_parser("seed-verify", help="验证 seed list (P15)")
    p_seed.add_argument("--manifest", type=Path, required=True)
    p_seed.add_argument("--seeds", type=str, required=True, help="comma-separated seed list")
    p_seed.add_argument("--required-min", type=int, default=5)
    p_seed.add_argument("--output", type=Path, required=True)

    p_hp = sub.add_parser("hyperparameter-hash", help="算超参 hash (P16)")
    p_hp.add_argument("--config", type=Path, required=True)

    args = parser.parse_args()

    if args.cmd == "seed-verify":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest_seeds = manifest.get("seeds", [])
        requested = [int(s) for s in args.seeds.split(",") if s.strip()]
        report = verify_seed_list(
            manifest_seeds=manifest_seeds,
            requested_seeds=requested,
            required_min_count=args.required_min,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[seed-verify] pass={report.pass_overall} "
            f"missing={report.missing_seeds} unexpected={report.unexpected_seeds} "
            f"dupes={report.duplicates_removed} → {args.output}",
            file=__import__("sys").stderr,
        )
        return 0 if report.pass_overall else 2

    if args.cmd == "hyperparameter-hash":
        import yaml

        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        flat: dict[str, Any] = {}
        for section in ("model_cfg", "train_cfg", "model", "train"):
            if isinstance(cfg, dict) and section in cfg and isinstance(cfg[section], dict):
                flat.update(cfg[section])
        if isinstance(cfg, dict):
            flat.update({k: v for k, v in cfg.items() if k not in ("model_cfg", "train_cfg", "model", "train")})
        h = compute_hyperparameter_hash(flat)
        print(f"[hyperparameter-hash] {h}")
        return 0

    parser.error(f"unknown cmd: {args.cmd}")
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())