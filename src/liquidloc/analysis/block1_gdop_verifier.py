"""BLOCK-1 锚点坐标 × GDOP 验证器（异步高NLOS实验全流程保障手册 Part 0 / S2）。

本模块对照手册 Part 0 / S2 锚点坐标表和 GDOP≈1.19 的约束，
验证实际 sim_e9 数据中 anchor_layout.json 的锚点坐标是否与手册一致。
不一致时将偏差量化并写入 .audit/anchor_gdop_audit.json，供实验门控（I-1）使用。

手册 Part 0 / S2 锚点坐标：
    A1 锚: (2,  2) m
    A2 锚: (18, 3) m
    A3 锚: (6, 17) m
    A4 锚: (16,18) m
    实测 GDOP ≈ 1.19（凸包内缩 1.5m 轨迹带），几何特征为非对称、无共线。
    协议裁决项：实测几何特征与协议 K1 档"分布不均、存在共线倾向"不完全一致；
                GDOP≈1.19 低于协议 K 轴全部档位区间，需作协议裁决登记（见 S2）。

本模块不对 sim_e9 数据做任何修改，只做只读验证和审计报告。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np  # 只用于 GDOP 计算。

from liquidloc.common.validation import coerce_finite_scalar  # D9 value contract.


__all__ = [
    "HANDBOOK_S2_ANCHOR_POSITIONS",
    "HANDBOOK_S2_ANCHOR_IDS",
    "compute_gdop",
    "compute_gdop_for_trajectory",
    "verify_anchor_layout_against_handbook",
    "run_block1_audit",
]


# 手册 Part 0 / S2 锚点坐标（A1-A4，共 4 个，对应好几何基准）
HANDBOOK_S2_ANCHOR_IDS = ("A1", "A2", "A3", "A4")
HANDBOOK_S2_ANCHOR_POSITIONS: dict[str, tuple[float, float]] = {
    "A1": (2.0, 2.0),
    "A2": (18.0, 3.0),
    "A3": (6.0, 17.0),
    "A4": (16.0, 18.0),
}
HANDBOOK_S2_GDOP = 1.19  # 手册 Part 0 声明的 GDOP 基线
HANDBOOK_S2_GDOP_TOL = 0.10  # ±10% 容差（BLOCK-1 / P4 约束）


def _gdop_from_matrix(H: np.ndarray) -> float:
    """从雅可比矩阵 H 计算 GDOP。

    GDOP = sqrt(trace((H.T H)^{-1}))，
    其中 H 的每一行对应一个锚点对(tag,anchor)的观测量对 (px,py) 的偏导。
    公式推导见手册 Part 0 / S2。
    """
    HT_H = H.T @ H
    try:
        inv = np.linalg.inv(HT_H)
    except np.linalg.LinAlgError:
        return float("inf")
    trace_inv = float(np.trace(inv))
    if trace_inv <= 0.0:
        return float("inf")
    return float(math.sqrt(trace_inv))


def compute_gdop(
    anchor_positions: dict[str, tuple[float, float]],
    tag_position: tuple[float, float] | None = None,
) -> float:
    """计算给定 tag 位置的 GDOP。

    若 tag_position 为 None，使用 4 个锚点的质心（典型 tag 工作区中心）。
    返回 GDOP 标量（≥1.0），单位无量纲。
    """
    if tag_position is None:
        xs = [p[0] for p in anchor_positions.values()]
        ys = [p[1] for p in anchor_positions.values()]
        px = sum(xs) / len(xs)
        py = sum(ys) / len(ys)
    else:
        px, py = float(tag_position[0]), float(tag_position[1])

    rows = []
    for (ax, ay) in anchor_positions.values():
        r = math.sqrt((ax - px) ** 2 + (ax - px) ** 2 + (ay - py) ** 2)
        if r < 1e-9:
            continue  # tag 在锚点上，跳过（此时 H 矩阵退化）
        dx = (px - ax) / r
        dy = (py - ay) / r
        rows.append([dx, dy])
    if not rows:
        return float("inf")
    H = np.array(rows, dtype=np.float64)
    return _gdop_from_matrix(H)


def compute_gdop_for_trajectory(
    anchor_positions: dict[str, tuple[float, float]],
    trajectory_xy: list[tuple[float, float]],
) -> dict[str, Any]:
    """沿轨迹计算逐帧 GDOP 和统计摘要。

    用于 BLOCK-1 / P4 的时变 GDOP 报告。如果 anchor 坐标正确，
    轨迹全程 GDOP 均值应 ≈ 1.19（±10%）。
    """
    gdop_values = []
    for (px, py) in trajectory_xy:
        g = compute_gdop(anchor_positions, tag_position=(px, py))
        gdop_values.append(g)
    finite = [g for g in gdop_values if math.isfinite(g)]
    if not finite:
        return {
            "gdop_mean": float("nan"),
            "gdop_min": float("nan"),
            "gdop_max": float("nan"),
            "gdop_std": float("nan"),
            "n_finite": 0,
            "n_total": len(gdop_values),
            "notes": "all GDOP values are infinite (tag at anchor or degenerate layout)",
        }
    return {
        "gdop_mean": float(np.mean(finite)),
        "gdop_min": float(np.min(finite)),
        "gdop_max": float(np.max(finite)),
        "gdop_std": float(np.std(finite)),
        "gdop_above_1_5_count": sum(1 for g in finite if g > 1.5),
        "gdop_above_2_0_count": sum(1 for g in finite if g > 2.0),
        "n_finite": len(finite),
        "n_total": len(gdop_values),
        "notes": "ok",
    }


def verify_anchor_layout_against_handbook(
    anchor_layout: dict[str, Any],
) -> dict[str, Any]:
    """将一个 anchor_layout dict 对照手册 S2 锚点坐标进行验证。

    参数:
        anchor_layout: 必须含 anchor_ids / anchor_positions / layout_id 字段。

    返回:
        dict，含 match（bool）、max_anchor_position_error_m、per_anchor_errors_m、
        actual_gdop（质心处）、handbook_gdop、gdop_error_rel、gdop_pass（bool）字段。

    异常:
        ValueError: anchor_layout 缺字段。
    """
    required_keys = {"anchor_ids", "anchor_positions"}
    for k in required_keys:
        if k not in anchor_layout:
            raise ValueError(f"anchor_layout missing required key '{k}'.")

    anchor_ids = anchor_layout["anchor_ids"]
    anchor_positions = anchor_layout["anchor_positions"]
    if not isinstance(anchor_ids, list):
        raise TypeError("anchor_ids must be a list.")
    if not isinstance(anchor_positions, list):
        raise TypeError("anchor_positions must be a list.")

    actual_dict: dict[str, tuple[float, float]] = {}
    per_anchor_errors: dict[str, dict[str, float]] = {}
    for i, aid in enumerate(anchor_ids):
        if i >= len(anchor_positions):
            break
        row = anchor_positions[i]
        # 兼容两种 layout 格式：
        # (a) list-of-list: [x, y]  (sim_e9_5seed_25unit stub)
        # (b) dict-of-objects: {"px": x, "py": y}  (legacy)
        if isinstance(row, dict):
            ax = coerce_finite_scalar(row.get("px", row.get("x", 0.0)),
                                      name=f"anchor_positions[{i}].px")
            ay = coerce_finite_scalar(row.get("py", row.get("y", 0.0)),
                                      name=f"anchor_positions[{i}].py")
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            ax = coerce_finite_scalar(row[0], name=f"anchor_positions[{i}][0]")
            ay = coerce_finite_scalar(row[1], name=f"anchor_positions[{i}][1]")
        else:
            raise TypeError(
                f"anchor_positions[{i}] must be dict or [x, y] list, got {type(row)}")
        actual_dict[aid] = (ax, ay)

    per_anchor_dists: dict[str, float] = {}
    max_error = 0.0
    # 兼容 anchor_ids='0','1','2','3' → 'A1','A2','A3','A4' 别名
    def _normalize_aid(aid: str) -> str:
        if aid in ("A1", "A2", "A3", "A4"):
            return aid
        if aid in ("0", "1", "2", "3"):
            return "A" + str(int(aid) + 1)
        return aid

    norm_actual_dict: dict[str, tuple[float, float]] = {
        _normalize_aid(aid): pos for aid, pos in actual_dict.items()
    }
    for aid, (hx, hy) in HANDBOOK_S2_ANCHOR_POSITIONS.items():
        if aid not in norm_actual_dict:
            per_anchor_dists[aid] = float("nan")
            max_error = float("inf")
            continue
        ax, ay = norm_actual_dict[aid]
        dist = math.sqrt((ax - hx) ** 2 + (ay - hy) ** 2)
        per_anchor_dists[aid] = dist
        if dist > max_error:
            max_error = dist

    # 锚点 ID 集合是否一致（已 _normalize_aid 映射到 A1-A4）
    id_set_match = set(HANDBOOK_S2_ANCHOR_IDS) == set(norm_actual_dict.keys())

    # GDOP 比较
    handbook_gdop = compute_gdop(HANDBOOK_S2_ANCHOR_POSITIONS)
    actual_gdop = compute_gdop(norm_actual_dict)
    if math.isfinite(actual_gdop) and math.isfinite(handbook_gdop) and handbook_gdop > 0:
        gdop_error_rel = abs(actual_gdop - handbook_gdop) / handbook_gdop
    else:
        gdop_error_rel = float("nan")
    gdop_pass = (
        math.isfinite(gdop_error_rel) and gdop_error_rel <= HANDBOOK_S2_GDOP_TOL
    )

    match = bool(id_set_match and max_error < 1e-9 and gdop_pass)

    return {
        "match": match,
        "handbook_gdop": handbook_gdop,
        "actual_gdop": actual_gdop,
        "gdop_error_rel": gdop_error_rel,
        "gdop_pass": gdop_pass,
        "max_anchor_position_error_m": max_error,
        "per_anchor_position_error_m": per_anchor_dists,
        "handbook_anchor_ids": list(HANDBOOK_S2_ANCHOR_IDS),
        "actual_anchor_ids": list(actual_dict.keys()),
        "anchor_ids_match": id_set_match,
        "layout_id": anchor_layout.get("layout_id"),
        "notes": "MATCH" if match else "MISMATCH — see per_anchor_position_error_m",
    }


def run_block1_audit(
    data_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """在指定 data_root 下扫描所有序列的 anchor_layout.json 并验证。

    读取所有 sim_e9 序列的 anchor_layout.json，对照手册 S2 坐标表进行验证，
    汇总报告写入 output_path（默认 .audit/anchor_gdop_audit.json）。

    参数:
        data_root: sim_e9 数据根目录（其下应有各 seq_id 子目录）。
        output_path: 审计报告输出路径，默认 .audit/anchor_gdop_audit.json。

    返回:
        汇总结果 dict，含 pass（bool）、total_seqs、mismatch_seqs、details。
    """
    import datetime

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if output_path is None:
        output_path = Path(".audit/anchor_gdop_audit.json")
    else:
        output_path = Path(output_path)

    # 兼容 2 种目录结构：
    # (a) data_root/seq_id/anchor_layout.json  (sim_e9 单 seed)
    # (b) data_root/seed_N/seq_id/anchor_layout.json  (sim_e9_5seed_25unit 多 seed)
    seq_dirs: list[Path] = []
    for d in root.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        # case (a): root/<seq_id>/anchor_layout.json
        if (d / "anchor_layout.json").is_file():
            seq_dirs.append(d)
        # case (b): root/<seed>/<seq_id>/anchor_layout.json
        for sub in d.iterdir():
            if sub.is_dir() and (sub / "anchor_layout.json").is_file():
                seq_dirs.append(sub)
    seq_dirs = sorted(set(seq_dirs), key=lambda d: d.name)

    details: list[dict[str, Any]] = []
    n_pass = 0
    n_mismatch = 0
    for seq_dir in seq_dirs:
        anchor_path = seq_dir / "anchor_layout.json"
        if not anchor_path.is_file():
            details.append({"seq_id": seq_dir.name, "status": "no_anchor_layout", "layout_path": str(anchor_path)})
            continue
        try:
            layout = json.loads(anchor_path.read_text(encoding="utf-8"))
            result = verify_anchor_layout_against_handbook(layout)
        except Exception as exc:
            result = {"error": str(exc)}
        result["seq_id"] = seq_dir.name
        result["layout_path"] = str(anchor_path)
        details.append(result)
        if result.get("match", False):
            n_pass += 1
        else:
            n_mismatch += 1

    summary = {
        "audit_timestamp": datetime.datetime.now().isoformat(),
        "data_root": str(root.resolve()),
        "total_seqs": len(seq_dirs),
        "pass_count": n_pass,
        "mismatch_count": n_mismatch,
        "handbook_gdop": HANDBOOK_S2_GDOP,
        "gdop_tolerance_rel": HANDBOOK_S2_GDOP_TOL,
        "max_allowed_anchor_error_m": 1e-9,
        "pass": n_mismatch == 0,
        "details": details,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = json.dumps(summary, indent=2, ensure_ascii=False)
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(output_text, encoding="utf-8")
    tmp.replace(output_path)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BLOCK-1 锚点坐标 × GDOP 验证器")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw/sim_e9",
        help="sim_e9 数据根目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".audit/anchor_gdop_audit.json",
        help="审计报告输出路径",
    )
    args = parser.parse_args()
    result = run_block1_audit(args.data_root, args.output)
    print(f"BLOCK-1 audit: pass={result['pass']}, "
          f"pass_count={result['pass_count']}, "
          f"mismatch_count={result['mismatch_count']}")
    for d in result["details"]:
        if not d.get("match", True):
            print(f"  MISMATCH seq={d['seq_id']}: "
                  f"per_anchor={d.get('per_anchor_position_error_m', {})}, "
                  f"gdop={d.get('actual_gdop')}")
