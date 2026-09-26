"""特征分布一致性自动监控模块

职责：
    监控训练集/测试集关键输入特征（UWB range 残差、IMU 高频段、VIO 相对位姿、时间戳特征）的
    分布一致性，输出统计报告到 results/feature_dist。

要求（铁律）：
    - 输出各 split 的均值/方差/分位数（P5/P50/P95）对比表
    - 漂移超阈值判定：任一分位数偏移 > 2σ 或 KS 检验 p < 0.01
    - 超阈值输出警告并写入 manifest feature_drift
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

from liquidloc.common.io_utils import read_json, write_json


# ===========================================================================
# 配置常量
# ===========================================================================

DISTRIBUTION_MONITOR_FEATURES = (
    "uwb_residual_r0", "uwb_residual_r1", "uwb_residual_r2", "uwb_residual_r3",
    "imu_ax", "imu_ay", "imu_gz",
    "vio_dx", "vio_dy", "vio_dyaw",
    "timestamp", "dt",
)

PERCENTILE_DRIFT_SIGMA_THRESHOLD = 2.0
KS_P_THRESHOLD = 0.01
REFERENCE_SPLIT = "train"
MAX_ANCHORS = 4


# ===========================================================================
# 路径安全工具
# ===========================================================================

def _validate_output_path(output_dir: Path, filename: str) -> Path:
    """校验并返回安全的输出文件路径（禁止路径穿越）"""
    if not filename or not isinstance(filename, str):
        raise ValueError(f"Invalid filename: {filename!r}")
    if "\\" in filename or "/" in filename:
        raise ValueError(f"Filename must not contain path separators: {filename!r}")
    output_dir_resolved = output_dir.resolve()
    target_path = (output_dir_resolved / filename).resolve()
    try:
        target_path.relative_to(output_dir_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Path escape detected: {target_path} outside {output_dir_resolved}"
        ) from exc
    return target_path


# ===========================================================================
# 数据结构
# ===========================================================================

@dataclass
class FeatureStats:
    mean: float
    variance: float
    std: float
    p5: float
    p50: float
    p95: float
    n_samples: int
    min_val: float
    max_val: float


@dataclass
class FeatureDriftReport:
    feature_name: str
    reference_stats: FeatureStats
    compare_stats: FeatureStats
    percentile_drift: dict[str, float]
    ks_statistic: float
    ks_p_value: float
    drift_detected: bool
    drift_reasons: list[str] = field(default_factory=list)


# ===========================================================================
# 统计计算
# ===========================================================================

def clean_numeric_values(values):
    cleaned = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
            if math.isfinite(fv):
                cleaned.append(fv)
        except (TypeError, ValueError, OverflowError):
            continue
    return np.array(cleaned, dtype=np.float64) if cleaned else np.array([], dtype=np.float64)


def compute_feature_stats(values):
    arr = clean_numeric_values(list(values))
    if len(arr) == 0:
        return FeatureStats(mean=0.0, variance=0.0, std=0.0, p5=0.0, p50=0.0, p95=0.0, n_samples=0, min_val=0.0, max_val=0.0)
    return FeatureStats(
        mean=float(np.mean(arr)),
        variance=float(np.var(arr)),
        std=float(np.std(arr)),
        p5=float(np.percentile(arr, 5)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        n_samples=len(arr),
        min_val=float(np.min(arr)),
        max_val=float(np.max(arr)),
    )


def compute_percentile_drift_values(ref_stats, comp_stats):
    std_ref = ref_stats.std if ref_stats.std >= 1e-10 else 1e-10
    results = {}
    for pct_name in ["p5", "p50", "p95"]:
        ref_val = getattr(ref_stats, pct_name)
        comp_val = getattr(comp_stats, pct_name)
        delta = comp_val - ref_val
        sigma_offset = delta / std_ref
        results[pct_name] = (delta, sigma_offset)
    return results


def check_percentile_drift(sigma_offset):
    return abs(sigma_offset) > PERCENTILE_DRIFT_SIGMA_THRESHOLD, sigma_offset


def compute_ks_test(ref_values, comp_values):
    ref_arr = clean_numeric_values(ref_values)
    comp_arr = clean_numeric_values(comp_values)
    if len(ref_arr) < 2 or len(comp_arr) < 2:
        return 0.0, 1.0
    statistic, p_value = stats.ks_2samp(ref_arr, comp_arr)
    return float(statistic), float(p_value)


# ===========================================================================
# 数据提取（从原始流文件）
# ===========================================================================

def _sorted_dict_by_timestamp(records):
    return sorted(records, key=lambda r: float(r.get("timestamp", 0)))


def _nearest_gt_for_timestamp(gt_records, timestamp):
    if not gt_records:
        return None
    timestamps = [r["timestamp"] for r in gt_records]
    idx = bisect_left(timestamps, timestamp)
    if idx == 0:
        return gt_records[0]
    if idx >= len(gt_records):
        return gt_records[-1]
    prev, curr = gt_records[idx - 1], gt_records[idx]
    return prev if abs(timestamp - prev["timestamp"]) <= abs(timestamp - curr["timestamp"]) else curr


def _extract_uwb_residuals(seq_dir, max_anchors=MAX_ANCHORS):
    uwb_path = seq_dir / "uwb.json"
    gt_path = seq_dir / "gt.json"
    anchor_path = seq_dir / "anchor_layout.json"
    if not uwb_path.exists():
        return {f"uwb_residual_r{i}": [] for i in range(max_anchors)}
    uwb_records = read_json(uwb_path)
    gt_records = _sorted_dict_by_timestamp(read_json(gt_path)) if gt_path.exists() else []
    anchor_positions = {}
    if anchor_path.exists():
        anchor_data = read_json(anchor_path)
        for aid, pos in zip(anchor_data.get("anchor_ids", []), anchor_data.get("anchor_positions", [])):
            anchor_positions[int(aid)] = (float(pos[0]), float(pos[1]))
    residuals_by_anchor = {i: [] for i in range(max_anchors)}
    for uwb_rec in uwb_records:
        anchor_id = int(uwb_rec.get("anchor_id", 0))
        if anchor_id >= max_anchors:
            continue
        measured_range = uwb_rec.get("range")
        timestamp = float(uwb_rec.get("timestamp", 0))
        if measured_range is None:
            continue
        gt_rec = _nearest_gt_for_timestamp(gt_records, timestamp)
        if gt_rec is None:
            continue
        gt_x = float(gt_rec.get("px", 0))
        gt_y = float(gt_rec.get("py", 0))
        anchor_pos = anchor_positions.get(anchor_id)
        if anchor_pos is None:
            continue
        ax, ay = anchor_pos
        predicted_range = math.sqrt((gt_x - ax) ** 2 + (gt_y - ay) ** 2)
        residuals_by_anchor[anchor_id].append(float(measured_range) - predicted_range)
    return {f"uwb_residual_r{i}": residuals_by_anchor[i] for i in range(max_anchors)}


def _extract_imu_high_freq(seq_dir):
    imu_path = seq_dir / "imu.json"
    if not imu_path.exists():
        return {"imu_ax": [], "imu_ay": [], "imu_gz": []}
    records = read_json(imu_path)
    return {
        "imu_ax": [float(r["ax"]) for r in records if r.get("ax") is not None],
        "imu_ay": [float(r["ay"]) for r in records if r.get("ay") is not None],
        "imu_gz": [float(r["gz"]) for r in records if r.get("gz") is not None],
    }


def _extract_vio_relative_pose(seq_dir):
    vio_path = seq_dir / "vio.json"
    if not vio_path.exists():
        return {"vio_dx": [], "vio_dy": [], "vio_dyaw": []}
    records = _sorted_dict_by_timestamp(read_json(vio_path))
    if len(records) < 2:
        return {"vio_dx": [], "vio_dy": [], "vio_dyaw": []}
    return {
        "vio_dx": [float(r["dx"]) for r in records if r.get("dx") is not None],
        "vio_dy": [float(r["dy"]) for r in records if r.get("dy") is not None],
        "vio_dyaw": [float(r["dyaw"]) for r in records if r.get("dyaw") is not None],
    }


def _extract_timestamps(seq_dir):
    imu_path = seq_dir / "imu.json"
    vio_path = seq_dir / "vio.json"
    timestamps = []
    if imu_path.exists():
        timestamps.extend(float(r["timestamp"]) for r in read_json(imu_path) if r.get("timestamp") is not None)
    if vio_path.exists():
        timestamps.extend(float(r["timestamp"]) for r in read_json(vio_path) if r.get("timestamp") is not None)
    timestamps = sorted(set(timestamps))
    dts = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    return {"timestamp": timestamps, "dt": dts}


def extract_sequence_features(seq_dir):
    features = {}
    features.update(_extract_uwb_residuals(seq_dir))
    features.update(_extract_imu_high_freq(seq_dir))
    features.update(_extract_vio_relative_pose(seq_dir))
    features.update(_extract_timestamps(seq_dir))
    return features


# ===========================================================================
# Split 处理
# ===========================================================================

def build_split_feature_data(split_manifest, feature_keys=None):
    if feature_keys is None:
        feature_keys = DISTRIBUTION_MONITOR_FEATURES
    splits_data = {}
    splits = split_manifest.get("splits", {})
    for split_name, split_info in splits.items():
        sequences = split_info.get("sequences", [])
        splits_data[split_name] = {feat: [] for feat in feature_keys}
        for seq_info in sequences:
            seq_dir_str = seq_info.get("seq_dir")
            if not seq_dir_str:
                continue
            seq_dir = Path(seq_dir_str)
            if not seq_dir.exists():
                continue
            seq_features = extract_sequence_features(seq_dir)
            for feat in feature_keys:
                if feat in seq_features:
                    splits_data[split_name][feat].extend(seq_features[feat])
    return splits_data


def build_split_stats(split_data, feature_keys=None):
    if feature_keys is None:
        feature_keys = DISTRIBUTION_MONITOR_FEATURES
    result = {}
    for split_name, features_data in split_data.items():
        result[split_name] = {}
        for feat in feature_keys:
            result[split_name][feat] = compute_feature_stats(features_data.get(feat, []))
    return result


# ===========================================================================
# 漂移检测
# ===========================================================================

def detect_feature_drift(split_stats, split_data, feature_keys=None):
    if feature_keys is None:
        feature_keys = DISTRIBUTION_MONITOR_FEATURES
    ref_split = REFERENCE_SPLIT
    if ref_split not in split_stats:
        available = sorted(split_stats.keys())
        ref_split = available[0] if available else None
    if ref_split is None:
        return []
    ref_stats = split_stats[ref_split]
    ref_data = split_data.get(ref_split, {})
    reports = []
    for split_name, split_stat in split_stats.items():
        if split_name == ref_split:
            continue
        split_data_map = split_data.get(split_name, {})
        for feat in feature_keys:
            ref_feat_stats = ref_stats.get(feat)
            comp_feat_stats = split_stat.get(feat)
            if ref_feat_stats is None or comp_feat_stats is None:
                continue
            ref_values = ref_data.get(feat, [])
            comp_values = split_data_map.get(feat, [])
            percentile_drift = compute_percentile_drift_values(ref_feat_stats, comp_feat_stats)
            ks_stat, ks_p = compute_ks_test(ref_values, comp_values)
            drift_detected = False
            drift_reasons = []
            for pct_name, (_, sigma_offset) in percentile_drift.items():
                is_drift, sigma_val = check_percentile_drift(sigma_offset)
                if is_drift:
                    drift_detected = True
                    drift_reasons.append(f"{pct_name} drift exceeds {PERCENTILE_DRIFT_SIGMA_THRESHOLD}sigma (actual: {sigma_val:.3f}sigma)")
            if ks_p < KS_P_THRESHOLD:
                drift_detected = True
                drift_reasons.append(f"KS test p-value < {KS_P_THRESHOLD} (actual: {ks_p:.6f})")
            reports.append(FeatureDriftReport(
                feature_name=feat,
                reference_stats=ref_feat_stats,
                compare_stats=comp_feat_stats,
                percentile_drift={k: v[1] for k, v in percentile_drift.items()},
                ks_statistic=ks_stat,
                ks_p_value=ks_p,
                drift_detected=drift_detected,
                drift_reasons=drift_reasons,
            ))
    return reports


# ===========================================================================
# 输出
# ===========================================================================

def _stats_to_dict(stats):
    return {
        "mean": stats.mean, "variance": stats.variance, "std": stats.std,
        "p5": stats.p5, "p50": stats.p50, "p95": stats.p95,
        "n_samples": stats.n_samples, "min": stats.min_val, "max": stats.max_val,
    }


def generate_comparison_table(split_stats, feature_keys):
    splits = sorted(split_stats.keys())
    rows = []
    for feat in feature_keys:
        row = {"feature": feat}
        for metric in ["mean", "std", "p5", "p50", "p95", "n_samples"]:
            by_split = {}
            for sn in splits:
                s = split_stats[sn].get(feat)
                by_split[sn] = getattr(s, metric) if s else None
            row[f"{metric}_by_split"] = by_split
        rows.append(row)
    return rows


def write_feature_distribution_report(split_stats, output_dir, feature_keys=None):
    if feature_keys is None:
        feature_keys = DISTRIBUTION_MONITOR_FEATURES
    output_dir_resolved = output_dir.resolve()
    output_dir_resolved.mkdir(parents=True, exist_ok=True)
    stats_dict = {}
    for sn, feats in split_stats.items():
        stats_dict[sn] = {f: _stats_to_dict(s) for f, s in feats.items()}
    stats_path = _validate_output_path(output_dir_resolved, "statistics.json")
    write_json(str(stats_path), stats_dict)
    table_path = _validate_output_path(output_dir_resolved, "comparison_table.json")
    write_json(str(table_path), generate_comparison_table(split_stats, feature_keys))
    return stats_path


def write_drift_manifest(drift_reports, output_dir):
    output_dir_resolved = output_dir.resolve()
    output_dir_resolved.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    for report in drift_reports:
        if report.drift_detected:
            for reason in report.drift_reasons:
                manifest_entries.append({
                    "split": "val_or_test_vs_train",
                    "feature": report.feature_name,
                    "reason": reason,
                    "ks_p_value": report.ks_p_value,
                    "ks_statistic": report.ks_statistic,
                })
    manifest_content = {
        "version": "1.0",
        "drift_threshold": {"percentile_sigma": PERCENTILE_DRIFT_SIGMA_THRESHOLD, "ks_p_threshold": KS_P_THRESHOLD},
        "drifted_features": manifest_entries,
        "total_features_monitored": len(drift_reports),
        "drift_detected": len(manifest_entries) > 0,
    }
    manifest_path = _validate_output_path(output_dir_resolved, "feature_drift.json")
    write_json(str(manifest_path), manifest_content)
    return manifest_path


def monitor_feature_distribution(split_manifest_path, output_dir=None, feature_keys=None, verbose=True):
    if feature_keys is None:
        feature_keys = DISTRIBUTION_MONITOR_FEATURES
    if output_dir is None:
        output_dir = Path("results") / "feature_dist"
    else:
        output_dir = Path(output_dir)
    if verbose:
        print(f"[feature_dist] 读取 split manifest: {split_manifest_path}")
    split_manifest = read_json(split_manifest_path)
    splits_data = build_split_feature_data(split_manifest, feature_keys)
    split_stats = build_split_stats(splits_data, feature_keys)
    stats_path = write_feature_distribution_report(split_stats, output_dir, feature_keys)
    drift_reports = detect_feature_drift(split_stats, splits_data, feature_keys)
    drift_warnings = []
    for report in drift_reports:
        if report.drift_detected:
            msg = f"[WARN] {report.feature_name} 在 {REFERENCE_SPLIT} vs 其他 split: {', '.join(report.drift_reasons)}"
            drift_warnings.append(msg)
            if verbose:
                print(msg)
    manifest_path = write_drift_manifest(drift_reports, output_dir)
    return {
        "output_dir": str(output_dir),
        "statistics_path": str(stats_path),
        "manifest_path": str(manifest_path),
        "drift_warnings": drift_warnings,
        "drift_detected": len(drift_warnings) > 0,
    }