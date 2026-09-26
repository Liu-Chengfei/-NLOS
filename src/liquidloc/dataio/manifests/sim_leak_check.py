"""DTW/形状特征距离泄漏检查。

这个模块计算训练/验证/测试序列两两之间的 DTW 距离（基于 (px,py) 轨迹坐标
作为形状特征），若归一化 DTW 距离 < 阈值（默认 0.1）的两条序列分属训练与测试，
则记录为泄漏项并写入 manifest sim_leak_check。

上游依赖:
- 物化序列目录下的 gt.json（提供 (px,py) 轨迹坐标）
- split_manifest（提供 train_ids/val_ids/test_ids）

下游调用者:
- scripts/26_sim_leak_check.py（命令行入口）
- 实验管线（读取 sim_leak_check manifest 判断是否允许全序结论）

核心变量:
- _SHAPE_FEATURE_KEYS: 从 gt.json 提取的形状特征字段名
- DTW_THRESHOLD: 归一化 DTW 距离泄漏阈值，默认 0.1
- sim_leak_manifest: 包含泄漏项的清单字典
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from liquidloc.common.io_utils import read_json, write_json
from liquidloc.common.tee_logger import print_dict

_SHAPE_FEATURE_KEYS = ("px", "py")
_DTW_THRESHOLD = 0.1


def _extract_shape_features(seq_dir: str | Path) -> Optional[np.ndarray]:
    """从 gt.json 提取 (px,py) 轨迹坐标作为形状特征向量。

    参数：
        seq_dir: 序列目录路径。

    返回：
        形状为 (T, 2) 的 NumPy 数组，每行为 (px, py)；无法提取时返回 None。
    """
    gt_path = Path(seq_dir) / "gt.json"
    if not gt_path.is_file():
        return None
    try:
        gt_rows = read_json(gt_path)
    except (OSError, ValueError):
        return None
    if not isinstance(gt_rows, list) or not gt_rows:
        return None
    coords = []
    for row in gt_rows:
        if not isinstance(row, dict):
            return None
        px = row.get("px")
        py = row.get("py")
        if px is None or py is None:
            return None
        try:
            px_f = float(px)
            py_f = float(py)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(px_f) and math.isfinite(py_f)):
            return None
        coords.append([px_f, py_f])
    if len(coords) < 2:
        return None
    return np.asarray(coords, dtype=float)


def _compute_dtw(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    *,
    threshold: float = _DTW_THRESHOLD,
) -> Optional[float]:
    """计算两条形状特征序列之间的归一化 DTW 距离。

    使用动态时间规整算法计算累积距离，并按较长序列的长度归一化，
    得到每时间步的平均 DTW 距离。若累积距离超过阈值 * 长度，提前终止。

    参数：
        seq_a: 形状为 (T_a, 2) 的 NumPy 数组。
        seq_b: 形状为 (T_b, 2) 的 NumPy 数组。
        threshold: 提前终止阈值，归一化 DTW 距离超过此值时停止计算。

    返回：
        归一化 DTW 距离（浮点数）；计算失败时返回 None。
    """
    t_a, dim = seq_a.shape
    t_b, _ = seq_b.shape
    if t_a < 2 or t_b < 2:
        return None
    if dim != 2:
        return None

    max_len = max(t_a, t_b)
    # 提前终止：若累积距离超过阈值 * 长度，无需继续计算
    early_stop_threshold = threshold * max_len

    # DTW 累积距离矩阵（仅保留两列以节省内存）
    prev = np.full(t_b, np.inf, dtype=float)
    curr = np.full(t_b, np.inf, dtype=float)
    # 修复 DTW 初始化：prev[0] = D(0,0)，prev[j] = D(0,j)（水平累积）
    # 原实现只初始化了 prev[0]，prev[j>0] 保持 inf 导致垂直步永远不可达，
    # 使 DTW 距离偏大，可能漏报跨长度相似序列的泄漏。
    prev[0] = float(np.sum((seq_a[0] - seq_b[0]) ** 2))
    for j in range(1, t_b):
        prev[j] = prev[j - 1] + float(np.sum((seq_a[0] - seq_b[j]) ** 2))

    for i in range(1, t_a):
        curr[0] = prev[0] + float(np.sum((seq_a[i] - seq_b[0]) ** 2))
        for j in range(1, t_b):
            cost = float(np.sum((seq_a[i] - seq_b[j]) ** 2))
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        # 检查当前行最小值是否已超过提前终止阈值
        if curr.min() > early_stop_threshold:
            return float(early_stop_threshold) + 1.0
        prev, curr = curr, prev

    dtw_cumulative = float(prev[t_b - 1])
    normalized = dtw_cumulative / max_len
    return normalized


def _load_split_ids(split_manifest: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """从 split_manifest 提取 train/val/test 序列 ID 列表。

    参数：
        split_manifest: 切分清单字典。

    返回：
        (train_ids, val_ids, test_ids) 三元组。
    """
    train_ids = [str(sid) for sid in split_manifest.get("train_ids", [])]
    val_ids = [str(sid) for sid in split_manifest.get("val_ids", [])]
    test_ids = [str(sid) for sid in split_manifest.get("test_ids", [])]
    return train_ids, val_ids, test_ids


def _build_seq_feature_map(
    sequences: list[dict[str, Any]],
    data_root: str | Path,
) -> dict[str, Optional[np.ndarray]]:
    """为每条序列提取形状特征向量。

    参数：
        sequences: 数据集清单中的序列记录列表。
        data_root: 数据根目录。

    返回：
        {seq_id: shape_features_array} 映射；无法提取的 seq_id 对应值
        为 None。
    """
    seq_feature_map: dict[str, Optional[np.ndarray]] = {}
    for record in sequences:
        if not isinstance(record, dict) or "seq_id" not in record:
            continue
        seq_id = str(record["seq_id"])
        seq_dir = record.get("seq_dir")
        if seq_dir is None:
            seq_feature_map[seq_id] = None
            continue
        seq_feature_map[seq_id] = _extract_shape_features(seq_dir)
    return seq_feature_map


def _collect_sim_leak_items(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    seq_feature_map: dict[str, Optional[np.ndarray]],
    *,
    threshold: float = _DTW_THRESHOLD,
) -> list[dict[str, Any]]:
    """检查所有不同切分组之间的形状特征泄漏。

    对所有 train/val/test 序列对（跨切分）计算归一化 DTW 距离；
    若距离 < threshold，则记录为泄漏项。

    参数：
        train_ids: 训练序列 ID 列表。
        val_ids: 验证序列 ID 列表。
        test_ids: 测试序列 ID 列表。
        seq_feature_map: {seq_id: shape_features} 映射。
        threshold: 归一化 DTW 距离泄漏阈值。

    返回：
        泄漏项列表，每项包含 kind、split_a、seq_id_a、split_b、seq_id_b、
        normalized_dtw_distance 和 reason。
    """
    leak_items: list[dict[str, Any]] = []

    # 构建 (split_name, seq_ids) 对列表，仅包含可提取特征的序列
    split_groups = []
    if train_ids:
        split_groups.append(('train', [sid for sid in train_ids if seq_feature_map.get(sid) is not None]))
    if val_ids:
        split_groups.append(('val', [sid for sid in val_ids if seq_feature_map.get(sid) is not None]))
    if test_ids:
        split_groups.append(('test', [sid for sid in test_ids if seq_feature_map.get(sid) is not None]))

    for idx_a in range(len(split_groups)):
        split_a_name, seq_a_list = split_groups[idx_a]
        for idx_b in range(idx_a + 1, len(split_groups)):
            split_b_name, seq_b_list = split_groups[idx_b]
            for seq_id_a in seq_a_list:
                seq_a = seq_feature_map[seq_id_a]
                for seq_id_b in seq_b_list:
                    seq_b = seq_feature_map[seq_id_b]
                    if seq_a is None or seq_b is None:
                        continue
                    dtw_dist = _compute_dtw(seq_a, seq_b, threshold=threshold)
                    if dtw_dist is None:
                        continue
                    if dtw_dist < threshold:
                        leak_items.append({
                            "kind": "sim_leak_dtw",
                            "split_a": split_a_name,
                            "seq_id_a": seq_id_a,
                            "split_b": split_b_name,
                            "seq_id_b": seq_id_b,
                            "normalized_dtw_distance": float(dtw_dist),
                            "threshold": float(threshold),
                            "reason": (
                                f"{split_a_name} sequence '{seq_id_a}' and "
                                f"{split_b_name} sequence '{seq_id_b}' "
                                f"have normalized DTW distance {dtw_dist:.4f} "
                                f"< threshold {threshold:.1f}"
                            ),
                        })
    return leak_items


def compute_sim_leak_manifest(
    dataset_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    *,
    threshold: float = _DTW_THRESHOLD,
) -> dict[str, Any]:
    """计算 sim_leak_check manifest，检测训练-测试形状特征泄漏。

    对所有 train/test 序列对计算归一化 DTW 距离，若距离低于阈值则
    判定为泄漏（高相似度序列不应分属训练与测试）。

    参数：
        dataset_manifest: 数据集清单字典（含 sequences 和 data_root）。
        split_manifest: 切分清单字典（含 train_ids/val_ids/test_ids）。
        threshold: 归一化 DTW 距离泄漏阈值，默认 0.1。

    返回：
        sim_leak_manifest 字典，包含 leak_items、is_clean 和 threshold。
    """
    sequences = dataset_manifest.get("sequences", [])
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("dataset_manifest.sequences must be a non-empty list")

    data_root = dataset_manifest.get("data_root", "")
    if not data_root:
        # 尝试从 seq_dir 推断数据根目录
        seq_dirs = [
            Path(record.get("seq_dir", ""))
            for record in sequences
            if isinstance(record, dict) and record.get("seq_dir")
        ]
        if seq_dirs:
            # 取所有 seq_dir 的最长公共父目录
            data_root = str(Path(seq_dirs[0]).parent)
            if len(seq_dirs) > 1:
                # 简化处理：使用第一个 seq_dir 的父目录
                data_root = str(Path(seq_dirs[0]).parent)

    train_ids, val_ids, test_ids = _load_split_ids(split_manifest)

    print_dict({
        "data_root": data_root,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "threshold": threshold,
    }, "compute_sim_leak_manifest")

    seq_feature_map = _build_seq_feature_map(sequences, data_root)
    n_with_features = sum(1 for v in seq_feature_map.values() if v is not None)
    print(f"[sim_leak_check] 可提取形状特征的序列数: {n_with_features}/{len(seq_feature_map)}")

    leak_items = _collect_sim_leak_items(
        train_ids, val_ids, test_ids, seq_feature_map, threshold=threshold
    )

    sim_leak_manifest = {
        "leak_items": leak_items,
        "is_clean": not leak_items,
        "threshold": float(threshold),
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "data_root": data_root,
    }
    return sim_leak_manifest


def write_sim_leak_manifest(
    sim_leak_manifest: dict[str, Any],
    output_root: str | Path,
) -> Path:
    """将 sim_leak_manifest 写入磁盘。

    参数：
        sim_leak_manifest: sim_leak_check manifest 字典。
        output_root: 输出根目录。

    返回：
        写入文件的绝对路径。
    """
    output_path = Path(output_root) / "sim_leak_check.json"
    write_json(output_path, sim_leak_manifest)
    return output_path.resolve()
