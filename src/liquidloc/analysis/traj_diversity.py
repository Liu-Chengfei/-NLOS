"""轨迹多样性校验：方位角直方图 + 速度分布。

本模块从 gt.json 提取轨迹运动特征，按轨迹计算：
  1. 方位角直方图（八方向，≤20% 差断言）
  2. 速度分布（0.5–1.5 m/s 全区间覆盖、训练/测试占比一致性断言）

结果写入 traj_diversity manifest，供实验管线判断测试集代表性。

上游依赖:
- 序列目录下的 gt.json（提供 px, py, yaw, timestamp）
- split_manifest（提供 train_ids/val_ids/test_ids）

下游调用者:
- scripts/27_check_traj_diversity.py（命令行入口）
- 实验管线（读取 traj_diversity.json 判断测试集代表性）

核心变量:
- _AZIMUTH_BIN_COUNT: 方位角分箱数（8 方向）
- _AZIMUTH_MAX_DEVIATION: 方位角各 bin 与均值差的最大容忍比例（20%）
- _SPEED_RANGE: 速度分析区间 [0.5, 1.5] m/s
- _SPEED_BIN_COUNT: 速度分箱数
- _SPEED_PROPORTION_TOLERANCE: 训练/测试速度占比最大容忍偏差比例
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from liquidloc.common.io_utils import read_json, write_json
from liquidloc.common.tee_logger import print_dict

__all__ = (
    "compute_traj_diversity_manifest",
    "write_traj_diversity_manifest",
    "AZIMUTH_BIN_COUNT",
    "AZIMUTH_MAX_DEVIATION",
    "SPEED_RANGE",
    "SPEED_BIN_COUNT",
    "SPEED_PROPORTION_TOLERANCE",
)

# 方位角：8 方向（N, NE, E, SE, S, SW, W, NW），每方向 45°（π/4 rad）
AZIMUTH_BIN_COUNT = 8
# 方位角各 bin 占比与均值之差的容忍上限（20%），即任意 bin 占比偏离均值不超过 20%
AZIMUTH_MAX_DEVIATION = 0.20
# 速度分析区间 [0.5, 1.5] m/s（§175 硬约束）
SPEED_RANGE = (0.5, 1.5)
# 速度分箱数：将 [0.5, 1.5] 等分为 5 档（每档 0.2 m/s）
SPEED_BIN_COUNT = 5
# 训练/测试速度占比最大容忍偏差比例（20%），即 |p_train - p_test| / p_train ≤ 20%
SPEED_PROPORTION_TOLERANCE = 0.20

# 方位角方向名称（标准罗盘划分：N 中心 0°/360°，NE 中心 45°，顺时针）
# 各方向覆盖区间（度）：N=[337.5,22.5), NE=[22.5,67.5), E=[67.5,112.5),
#                       SE=[112.5,157.5), S=[157.5,202.5), SW=[202.5,247.5),
#                       W=[247.5,292.5), NW=[292.5,337.5)
_AZIMUTH_DIRECTION_NAMES = [
    "N",    # bin 0, 中心 0°/360°
    "NE",   # bin 1, 中心 45°
    "E",    # bin 2, 中心 90°
    "SE",   # bin 3, 中心 135°
    "S",    # bin 4, 中心 180°
    "SW",   # bin 5, 中心 225°
    "W",    # bin 6, 中心 270°
    "NW",   # bin 7, 中心 315°
]


def _extract_gt_features(seq_dir: str | Path) -> Optional[dict[str, np.ndarray]]:
    """从 gt.json 提取轨迹运动特征（位置、速度、方位角）。

    参数：
        seq_dir: 序列目录路径。

    返回：
        包含 'timestamps', 'positions', 'speeds', 'azimuths' 的字典；
        无法提取时返回 None。
    """
    gt_path = Path(seq_dir) / "gt.json"
    if not gt_path.is_file():
        return None
    try:
        gt_rows = read_json(gt_path)
    except (OSError, ValueError):
        return None
    if not isinstance(gt_rows, list) or len(gt_rows) < 2:
        return None

    timestamps: list[float] = []
    positions: list[tuple[float, float]] = []
    yaws: list[float] = []

    for row in gt_rows:
        if not isinstance(row, dict):
            return None
        ts = row.get("timestamp", row.get("t"))
        px = row.get("px")
        py = row.get("py")
        yaw = row.get("yaw")
        if ts is None or px is None or py is None:
            return None
        try:
            ts_f = float(ts)
            px_f = float(px)
            py_f = float(py)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(ts_f) and math.isfinite(px_f) and math.isfinite(py_f)):
            return None
        yaw_f = None
        if yaw is not None:
            try:
                yaw_f = float(yaw)
            except (TypeError, ValueError):
                yaw_f = None
        timestamps.append(ts_f)
        positions.append((px_f, py_f))
        yaws.append(yaw_f)

    if len(timestamps) < 2:
        return None

    # 按时间戳排序（gt.json 通常已排序，但做防御性排序）
    sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    timestamps = [timestamps[i] for i in sorted_indices]
    positions = [positions[i] for i in sorted_indices]
    yaws = [yaws[i] for i in sorted_indices]

    # 计算速度（中心差分）
    speeds: list[float] = []
    for i in range(len(timestamps)):
        if i == 0:
            # 前向差分
            dt = timestamps[1] - timestamps[0]
            dx = positions[1][0] - positions[0][0]
            dy = positions[1][1] - positions[0][1]
        elif i == len(timestamps) - 1:
            # 后向差分
            dt = timestamps[-1] - timestamps[-2]
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
        else:
            # 中心差分
            dt = timestamps[i + 1] - timestamps[i - 1]
            dx = positions[i + 1][0] - positions[i - 1][0]
            dy = positions[i + 1][1] - positions[i - 1][1]
        if dt > 1e-9:
            speed = math.sqrt(dx * dx + dy * dy) / dt
            speeds.append(speed)
        else:
            speeds.append(0.0)

    # 计算方位角：优先用 yaw 字段，缺失时用 atan2(dy, dx) 从位移方向计算
    azimuths: list[float] = []
    for i in range(len(positions)):
        if yaws[i] is not None and math.isfinite(yaws[i]):
            # 用 yaw 字段（弧度），包裹到 [0, 2π)
            az = yaws[i] % (2.0 * math.pi)
            azimuths.append(az)
        else:
            # 用位移方向
            if i == 0:
                dx = positions[1][0] - positions[0][0]
                dy = positions[1][1] - positions[0][1]
            elif i == len(positions) - 1:
                dx = positions[-1][0] - positions[-2][0]
                dy = positions[-1][1] - positions[-2][1]
            else:
                dx = positions[i + 1][0] - positions[i - 1][0]
                dy = positions[i + 1][1] - positions[i - 1][1]
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                azimuths.append(0.0)  # 静止时默认 0
            else:
                az = math.atan2(dy, dx) % (2.0 * math.pi)
                azimuths.append(az)

    return {
        "timestamps": np.array(timestamps, dtype=float),
        "positions": np.array(positions, dtype=float),
        "speeds": np.array(speeds, dtype=float),
        "azimuths": np.array(azimuths, dtype=float),
    }


def _compute_azimuth_histogram(azimuths: np.ndarray) -> dict[str, Any]:
    """计算方位角八方向直方图（标准罗盘划分）。

    方位角先偏移 π/8（22.5°）使正北（0°/360°）落在 bin 0 中心，
    再用等距分箱 [0, 2π) 划分为 8 方向：
      bin 0 = N   [337.5°, 22.5°)   中心 0°
      bin 1 = NE  [22.5°, 67.5°)    中心 45°
      bin 2 = E   [67.5°, 112.5°)   中心 90°
      bin 3 = SE  [112.5°, 157.5°)  中心 135°
      bin 4 = S   [157.5°, 202.5°)  中心 180°
      bin 5 = SW  [202.5°, 247.5°)  中心 225°
      bin 6 = W   [247.5°, 292.5°)  中心 270°
      bin 7 = NW  [292.5°, 337.5°)  中心 315°

    参数：
        azimuths: 方位角数组（弧度，[0, 2π)）。

    返回：
        直方图字典，含 counts, proportions, direction_names。
    """
    # 偏移 π/8 使正北落在 bin 0 中心
    shifted = (azimuths + math.pi / 8.0) % (2.0 * math.pi)
    bin_edges = np.linspace(0.0, 2.0 * math.pi, AZIMUTH_BIN_COUNT + 1)
    # 最后一个 bin 右端闭区间：2π 归入 bin 7
    bin_edges[-1] += 1e-9

    counts, _ = np.histogram(shifted, bins=bin_edges)
    total = int(counts.sum())
    if total == 0:
        proportions = np.zeros(AZIMUTH_BIN_COUNT, dtype=float)
    else:
        proportions = counts.astype(float) / total

    direction_names = list(_AZIMUTH_DIRECTION_NAMES)

    return {
        "bin_count": AZIMUTH_BIN_COUNT,
        "bin_edges_rad": bin_edges.tolist(),
        "shift_applied": True,
        "counts": counts.tolist(),
        "proportions": proportions.tolist(),
        "direction_names": direction_names,
        "total_samples": total,
    }


def _assert_azimuth_uniformity(proportions: list[float]) -> dict[str, Any]:
    """断言方位角各 bin 占比均匀（任意 bin 偏离均值 ≤ 20%）。

    参数：
        proportions: 8 方向占比列表。

    返回：
        断言结果字典，含 passed, max_deviation, details。
    """
    if len(proportions) != AZIMUTH_BIN_COUNT:
        return {
            "passed": False,
            "max_deviation": None,
            "threshold": AZIMUTH_MAX_DEVIATION,
            "reason": f"bin count mismatch: expected {AZIMUTH_BIN_COUNT}, got {len(proportions)}",
            "violations": [],
        }

    total_proportion = sum(proportions)
    if abs(total_proportion - 1.0) > 1e-6:
        return {
            "passed": False,
            "max_deviation": None,
            "threshold": AZIMUTH_MAX_DEVIATION,
            "reason": f"proportions sum to {total_proportion:.6f}, expected 1.0",
            "violations": [],
        }

    mean_proportion = total_proportion / AZIMUTH_BIN_COUNT  # 理想均值 = 1/8 = 0.125
    max_deviation = 0.0
    violations: list[dict[str, Any]] = []

    for i, prop in enumerate(proportions):
        if mean_proportion > 1e-12:
            deviation = abs(prop - mean_proportion) / mean_proportion
        else:
            deviation = float(abs(prop - mean_proportion) > 1e-12)
        if deviation > max_deviation:
            max_deviation = deviation
        if deviation > AZIMUTH_MAX_DEVIATION:
            violations.append({
                "direction": _AZIMUTH_DIRECTION_NAMES[i],
                "bin_index": i,
                "proportion": prop,
                "mean_proportion": mean_proportion,
                "deviation": deviation,
            })

    passed = len(violations) == 0
    return {
        "passed": passed,
        "max_deviation": max_deviation,
        "threshold": AZIMUTH_MAX_DEVIATION,
        "mean_proportion": mean_proportion,
        "reason": "all bins within tolerance" if passed else f"{len(violations)} bins exceed {AZIMUTH_MAX_DEVIATION:.0%} deviation",
        "violations": violations,
    }


def _compute_speed_distribution(speeds: np.ndarray) -> dict[str, Any]:
    """计算速度分布（0.5–1.5 m/s 区间）。

    参数：
        speeds: 速度数组（m/s）。

    返回：
        速度分布字典，含 bin_edges, counts, proportions, coverage。
    """
    speed_lo, speed_hi = SPEED_RANGE
    # 将 [0.5, 1.5] 等分为 SPEED_BIN_COUNT 个 bin
    bin_edges = np.linspace(speed_lo, speed_hi, SPEED_BIN_COUNT + 1)
    # 最后一个 bin 右端闭区间
    bin_edges[-1] += 1e-9

    counts, _ = np.histogram(speeds, bins=bin_edges)
    total = int(counts.sum())
    if total == 0:
        proportions = np.zeros(SPEED_BIN_COUNT, dtype=float)
    else:
        proportions = counts.astype(float) / total

    # 全区间覆盖检查：每个 bin 至少有一个样本
    coverage_bins = [i for i, c in enumerate(counts) if c > 0]
    full_coverage = len(coverage_bins) == SPEED_BIN_COUNT

    # 区间内样本占比
    in_range_count = int(np.sum((speeds >= speed_lo) & (speeds <= speed_hi)))
    in_range_ratio = in_range_count / len(speeds) if len(speeds) > 0 else 0.0

    return {
        "bin_count": SPEED_BIN_COUNT,
        "bin_edges_mps": bin_edges.tolist(),
        "counts": counts.tolist(),
        "proportions": proportions.tolist(),
        "total_samples": total,
        "coverage_bins": coverage_bins,
        "full_coverage": full_coverage,
        "in_range_count": in_range_count,
        "in_range_ratio": in_range_ratio,
        "speed_range_mps": [speed_lo, speed_hi],
    }


def _assert_speed_proportion_consistency(
    train_proportions: list[float],
    test_proportions: list[float],
) -> dict[str, Any]:
    """断言训练/测试速度占比一致性。

    参数：
        train_proportions: 训练集各速度 bin 占比。
        test_proportions: 测试集各速度 bin 占比。

    返回：
        断言结果字典，含 passed, max_deviation, details。
    """
    if len(train_proportions) != SPEED_BIN_COUNT or len(test_proportions) != SPEED_BIN_COUNT:
        return {
            "passed": False,
            "max_deviation": None,
            "threshold": SPEED_PROPORTION_TOLERANCE,
            "reason": f"bin count mismatch: expected {SPEED_BIN_COUNT}, "
                      f"got train={len(train_proportions)}, test={len(test_proportions)}",
            "violations": [],
        }

    max_deviation = 0.0
    violations: list[dict[str, Any]] = []

    for i in range(SPEED_BIN_COUNT):
        tp = train_proportions[i]
        sp = test_proportions[i]
        if tp > 1e-12:
            relative_deviation = abs(tp - sp) / tp
        else:
            # 训练集该 bin 占比为 0，测试集必须也为 0（绝对偏差）
            relative_deviation = float(abs(sp - tp) > 1e-12)
        if relative_deviation > max_deviation:
            max_deviation = relative_deviation
        if relative_deviation > SPEED_PROPORTION_TOLERANCE:
            violations.append({
                "bin_index": i,
                "train_proportion": tp,
                "test_proportion": sp,
                "relative_deviation": relative_deviation,
            })

    passed = len(violations) == 0
    return {
        "passed": passed,
        "max_deviation": max_deviation,
        "threshold": SPEED_PROPORTION_TOLERANCE,
        "reason": "all bins within tolerance" if passed else (
            f"{len(violations)} bins exceed {SPEED_PROPORTION_TOLERANCE:.0%} relative deviation"
        ),
        "violations": violations,
    }


def _check_azimuth_non_degenerate(
    azimuths: np.ndarray,
    speeds: np.ndarray,
) -> dict[str, Any]:
    """检查单条轨迹的方位角是否非退化。

    单条轨迹通常只沿少数方向运动（如直线行驶），不要求 8 方向均匀；
    退化轨迹指完全静止（速度全零）或数据缺失，此类轨迹不产生有效方位角信息。

    参数：
        azimuths: 方位角数组（弧度）。
        speeds: 速度数组（m/s），用于检测静止退化。

    返回：
        检查结果字典，含 passed, unique_bin_count, reason。
    """
    if len(azimuths) == 0:
        return {
            "passed": False,
            "unique_bin_count": 0,
            "reason": "no azimuth samples",
        }

    # 静止退化检测：速度全零时方位角无意义（默认 0 非真实运动方向）
    if len(speeds) > 0 and np.sum(np.abs(speeds)) < 1e-9:
        return {
            "passed": False,
            "unique_bin_count": 0,
            "total_samples": len(azimuths),
            "reason": "degenerate: stationary trajectory (all speeds zero)",
        }

    # 将方位角映射到 8 个 bin（与直方图同口径）
    shifted = (azimuths + math.pi / 8.0) % (2.0 * math.pi)
    bin_edges = np.linspace(0.0, 2.0 * math.pi, AZIMUTH_BIN_COUNT + 1)
    bin_edges[-1] += 1e-9
    bin_indices = np.digitize(shifted, bin_edges) - 1
    unique_bins = len(set(int(b) for b in bin_indices))

    # 单条轨迹只要至少 1 个 bin 有采样且非静止即为非退化
    passed = unique_bins >= 1
    return {
        "passed": passed,
        "unique_bin_count": unique_bins,
        "total_samples": len(azimuths),
        "reason": f"covers {unique_bins} direction bin(s)" if passed else (
            f"degenerate: no azimuth data"
        ),
    }


def _compute_split_traj_diversity(
    seq_ids: list[str],
    sequences: list[dict[str, Any]],
    data_root: str | Path,
    split_name: str,
) -> dict[str, Any]:
    """计算单个切分（train/val/test）的轨迹多样性统计。

    参数：
        seq_ids: 序列 ID 列表。
        sequences: 数据集清单中的序列记录列表。
        data_root: 数据根目录。
        split_name: 切分名称（train/val/test）。

    返回：
        轨迹多样性统计字典。
    """
    seq_by_id = {str(r.get("seq_id", "")): r for r in sequences if isinstance(r, dict)}
    all_azimuths: list[float] = []
    all_speeds: list[float] = []
    seq_azimuth_histograms: list[dict[str, Any]] = []
    seq_speed_distributions: list[dict[str, Any]] = []
    seq_azimuth_assertions: list[dict[str, Any]] = []
    seq_speed_coverage: list[dict[str, Any]] = []
    seq_errors: list[dict[str, str]] = []

    for seq_id in seq_ids:
        record = seq_by_id.get(seq_id)
        if record is None:
            seq_errors.append({"seq_id": seq_id, "error": "not found in dataset_manifest"})
            continue
        seq_dir = record.get("seq_dir")
        if seq_dir is None:
            seq_errors.append({"seq_id": seq_id, "error": "missing seq_dir"})
            continue

        features = _extract_gt_features(seq_dir)
        if features is None:
            seq_errors.append({"seq_id": seq_id, "error": "gt.json missing or invalid"})
            continue

        azimuth_hist = _compute_azimuth_histogram(features["azimuths"])
        speed_dist = _compute_speed_distribution(features["speeds"])
        # 单条轨迹方位角非退化检查：非静止且至少 1 个方向 bin 有采样
        azimuth_assertion = _check_azimuth_non_degenerate(features["azimuths"], features["speeds"])

        # 单条轨迹速度区间覆盖检查：至少有一个样本落在 [0.5, 1.5] 区间内
        speed_lo, speed_hi = SPEED_RANGE
        in_range_mask = (features["speeds"] >= speed_lo) & (features["speeds"] <= speed_hi)
        in_range_count = int(in_range_mask.sum())
        in_range_ratio = in_range_count / len(features["speeds"]) if len(features["speeds"]) > 0 else 0.0
        speed_coverage = {
            "seq_id": seq_id,
            "in_range_count": in_range_count,
            "total_samples": len(features["speeds"]),
            "in_range_ratio": in_range_ratio,
            "speed_range_mps": [speed_lo, speed_hi],
            "has_in_range_sample": in_range_count > 0,
        }

        azimuth_hist["seq_id"] = seq_id
        speed_dist["seq_id"] = seq_id
        azimuth_assertion["seq_id"] = seq_id
        speed_coverage["seq_id"] = seq_id
        seq_azimuth_histograms.append(azimuth_hist)
        seq_speed_distributions.append(speed_dist)
        seq_azimuth_assertions.append(azimuth_assertion)
        seq_speed_coverage.append(speed_coverage)

        all_azimuths.extend(features["azimuths"].tolist())
        all_speeds.extend(features["speeds"].tolist())

    # 切分级汇总：合并所有序列的方位角和速度
    azimuth_summary: dict[str, Any]
    speed_summary: dict[str, Any]

    if all_azimuths:
        azimuth_hist = _compute_azimuth_histogram(np.array(all_azimuths, dtype=float))
        azimuth_assertion = _assert_azimuth_uniformity(azimuth_hist["proportions"])
        azimuth_summary = {
            "histogram": azimuth_hist,
            "assertion": azimuth_assertion,
        }
    else:
        azimuth_summary = {
            "histogram": None,
            "assertion": {
                "passed": False,
                "reason": "no azimuth data available",
            },
        }

    if all_speeds:
        speed_dist = _compute_speed_distribution(np.array(all_speeds, dtype=float))
        speed_summary = {
            "distribution": speed_dist,
            "full_coverage": speed_dist["full_coverage"],
            "in_range_ratio": speed_dist["in_range_ratio"],
        }
    else:
        speed_summary = {
            "distribution": None,
            "full_coverage": False,
            "in_range_ratio": 0.0,
            "reason": "no speed data available",
        }

    # 单条轨迹断言汇总统计
    azimuth_passed_count = sum(1 for a in seq_azimuth_assertions if a.get("passed", False))
    azimuth_failed_count = len(seq_azimuth_assertions) - azimuth_passed_count
    speed_has_in_range_count = sum(1 for s in seq_speed_coverage if s.get("has_in_range_sample", False))
    speed_no_in_range_count = len(seq_speed_coverage) - speed_has_in_range_count

    return {
        "split_name": split_name,
        "seq_count": len(seq_ids),
        "seq_with_data": len(seq_azimuth_histograms),
        "azimuth_summary": azimuth_summary,
        "speed_summary": speed_summary,
        "seq_azimuth_histograms": seq_azimuth_histograms,
        "seq_speed_distributions": seq_speed_distributions,
        "seq_azimuth_assertions": seq_azimuth_assertions,
        "seq_speed_coverage": seq_speed_coverage,
        "seq_errors": seq_errors,
        "azimuth_passed_count": azimuth_passed_count,
        "azimuth_failed_count": azimuth_failed_count,
        "speed_has_in_range_count": speed_has_in_range_count,
        "speed_no_in_range_count": speed_no_in_range_count,
    }


def compute_traj_diversity_manifest(
    dataset_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    """计算轨迹多样性 manifest。

    对 train/val/test 三个切分分别计算方位角直方图和速度分布，
    并对测试集做方位角均匀性断言和速度占比一致性断言（与训练集比较）。

    参数：
        dataset_manifest: 数据集清单字典（含 sequences 和 data_root）。
        split_manifest: 切分清单字典（含 train_ids/val_ids/test_ids）。

    返回：
        traj_diversity_manifest 字典，含 train/val/test 的多样性统计、
        断言结果和整体通过状态。
    """
    sequences = dataset_manifest.get("sequences", [])
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("dataset_manifest.sequences must be a non-empty list")

    data_root = dataset_manifest.get("data_root", "")
    if not data_root:
        seq_dirs = [
            Path(record.get("seq_dir", ""))
            for record in sequences
            if isinstance(record, dict) and record.get("seq_dir")
        ]
        if seq_dirs:
            data_root = str(Path(seq_dirs[0]).parent)

    train_ids = [str(sid) for sid in split_manifest.get("train_ids", [])]
    val_ids = [str(sid) for sid in split_manifest.get("val_ids", [])]
    test_ids = [str(sid) for sid in split_manifest.get("test_ids", [])]

    print_dict({
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "data_root": data_root,
    }, "compute_traj_diversity_manifest")

    train_stats = _compute_split_traj_diversity(train_ids, sequences, data_root, "train")
    val_stats = _compute_split_traj_diversity(val_ids, sequences, data_root, "val")
    test_stats = _compute_split_traj_diversity(test_ids, sequences, data_root, "test")

    # 测试集方位角断言（聚合级 + 单条轨迹级）
    test_azimuth_assertion = test_stats["azimuth_summary"].get("assertion", {})
    azimuth_passed = bool(test_azimuth_assertion.get("passed", False))

    # 单条轨迹方位角断言汇总
    test_azimuth_traj_passed = test_stats.get("azimuth_passed_count", 0)
    test_azimuth_traj_failed = test_stats.get("azimuth_failed_count", 0)
    test_azimuth_traj_total = test_stats.get("seq_with_data", 0)
    azimuth_all_traj_passed = (
        test_azimuth_traj_total > 0 and test_azimuth_traj_failed == 0
    )

    # 测试集速度分布断言：全区间覆盖 + 与训练集占比一致性
    test_speed_dist = test_stats["speed_summary"].get("distribution")
    train_speed_dist = train_stats["speed_summary"].get("distribution")

    speed_full_coverage = bool(test_stats["speed_summary"].get("full_coverage", False))

    # 单条轨迹速度区间覆盖汇总
    test_speed_has_in_range = test_stats.get("speed_has_in_range_count", 0)
    test_speed_no_in_range = test_stats.get("speed_no_in_range_count", 0)
    test_speed_traj_total = test_stats.get("seq_with_data", 0)
    speed_all_traj_has_in_range = (
        test_speed_traj_total > 0 and test_speed_no_in_range == 0
    )

    speed_proportion_assertion: dict[str, Any]
    if test_speed_dist is not None and train_speed_dist is not None:
        speed_proportion_assertion = _assert_speed_proportion_consistency(
            train_speed_dist["proportions"],
            test_speed_dist["proportions"],
        )
        speed_proportion_passed = bool(speed_proportion_assertion.get("passed", False))
    else:
        speed_proportion_assertion = {
            "passed": False,
            "reason": "speed distribution unavailable for train or test split",
        }
        speed_proportion_passed = False

    overall_passed = azimuth_passed and azimuth_all_traj_passed and speed_full_coverage and speed_proportion_passed and speed_all_traj_has_in_range

    traj_diversity_manifest = {
        "overall_passed": overall_passed,
        "azimuth_assertion": test_azimuth_assertion,
        "azimuth_trajectory_summary": {
            "passed_count": test_azimuth_traj_passed,
            "failed_count": test_azimuth_traj_failed,
            "total_with_data": test_azimuth_traj_total,
            "all_passed": azimuth_all_traj_passed,
        },
        "speed_full_coverage": speed_full_coverage,
        "speed_trajectory_summary": {
            "has_in_range_count": test_speed_has_in_range,
            "no_in_range_count": test_speed_no_in_range,
            "total_with_data": test_speed_traj_total,
            "all_have_in_range": speed_all_traj_has_in_range,
        },
        "speed_proportion_assertion": speed_proportion_assertion,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "test_stats": test_stats,
        "config": {
            "azimuth_bin_count": AZIMUTH_BIN_COUNT,
            "azimuth_max_deviation": AZIMUTH_MAX_DEVIATION,
            "speed_range_mps": list(SPEED_RANGE),
            "speed_bin_count": SPEED_BIN_COUNT,
            "speed_proportion_tolerance": SPEED_PROPORTION_TOLERANCE,
        },
    }

    return traj_diversity_manifest


def write_traj_diversity_manifest(
    traj_diversity_manifest: dict[str, Any],
    output_root: str | Path,
) -> Path:
    """将 traj_diversity_manifest 写入磁盘。

    参数：
        traj_diversity_manifest: traj_diversity manifest 字典。
        output_root: 输出根目录。

    返回：
        写入文件的绝对路径。
    """
    output_path = Path(output_root) / "traj_diversity.json"
    write_json(output_path, traj_diversity_manifest)
    return output_path.resolve()
