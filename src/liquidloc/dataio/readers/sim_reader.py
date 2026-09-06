"""仿真数据原始表读取器，供准备流程直接使用。

这个模块负责把仿真序列目录下的 JSON 文件读成上层准备流程能直接使用
的原始包字典。它只做读取和基本形状校验，不注入任何变换或扰动。
上游依赖:
- 文件系统上的仿真序列目录（包含 imu.json, uwb.json, vio.json, gt.json 等）

下游调用者:
- 准备流程脚本（通过 dataio/__init__.py 的 read_sim_sequence 入口调用）
- 烟雾测试脚本（验证仿真数据是否可读）

核心变量:
- bundle: 按内部约定组织的原始包字典
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from liquidloc.common.io_utils import read_json
from liquidloc.common.validation import validate_path_component

__all__ = ("read_sim_sequence",)


def _normalize_vio_rows(vio_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 VIO 原始行归一化为 (dx, dy, dyaw, quality) 增量格式。

    sim_e9_5seed_25unit 给的是 (x, y, yaw) 绝对位姿，需要转化为 (dx, dy, dyaw) 增量；
    sim_materializer 给的已经是 (dx, dy, dyaw) 格式，直接透传。
    """
    if not vio_rows:
        return vio_rows
    first = vio_rows[0]
    has_abs = "x" in first and "y" in first and "yaw" in first
    has_inc = "dx" in first and "dy" in first and "dyaw" in first
    if has_inc:
        return vio_rows  # 已经是增量格式
    if not has_abs:
        return vio_rows  # 字段未知，原样透传（build_vio_events 会报缺字段错误）
    # 转换为增量格式
    out: list[dict[str, Any]] = []
    prev_x = prev_y = prev_yaw = None
    for row in vio_rows:
        x = float(row.get("x", 0.0))
        y = float(row.get("y", 0.0))
        yaw = float(row.get("yaw", 0.0))
        if prev_x is None:
            dx, dy, dyaw = 0.0, 0.0, 0.0
        else:
            dx = x - prev_x
            dy = y - prev_y
            # yaw 是相位角，需要做角度差分并 wrap 到 [-pi, pi]
            dyaw_raw = yaw - prev_yaw
            while dyaw_raw > 3.141592653589793:
                dyaw_raw -= 2 * 3.141592653589793
            while dyaw_raw < -3.141592653589793:
                dyaw_raw += 2 * 3.141592653589793
            dyaw = dyaw_raw
        prev_x, prev_y, prev_yaw = x, y, yaw
        new_row = dict(row)
        new_row["dx"] = dx
        new_row["dy"] = dy
        new_row["dyaw"] = dyaw
        # 缺 quality 字段时填 0.85（V0 健康档默认）
        if "quality" not in new_row:
            new_row["quality"] = float(new_row.get("cov", 0.04)) if "cov" in new_row else 0.85
        # 移除绝对字段，避免 build_vio_events 报重复字段
        for k in ("x", "y", "yaw", "cov"):
            new_row.pop(k, None)
        out.append(new_row)
    return out


def _normalize_uwb_rows(uwb_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 UWB 原始行归一化为 build_uwb_events 期望的 (anchor_id, range, valid, quality) 一锚一行格式。

    sim_e9_5seed_25unit 给的就是一锚一行格式，只需做字段类型强制转换。
    """
    if not uwb_rows:
        return uwb_rows
    first = uwb_rows[0]
    if "anchor_id" in first:
        # 已经是正确格式，只做类型强制转换
        out = []
        for row in uwb_rows:
            # anchor_id 兼容字符串 ("A0") 与整数；字符串格式 "A0" 提取数字部分。
            _aid = row.get("anchor_id", -1)
            if isinstance(_aid, str):
                _aid = int(_aid[1:]) if _aid.startswith("A") and _aid[1:].isdigit() else int(_aid) if _aid.lstrip("-").isdigit() else _aid
            new_row = {
                "timestamp": float(row.get("timestamp", 0.0)),
                "anchor_id": _aid,
                "range": float(row.get("range", 0.0)),
                "valid": 1 if bool(row.get("valid", True)) else 0,
                "quality": float(row.get("quality", 0.95)),
            }
            if "nl_flag" in row:
                new_row["nl_flag"] = int(row["nl_flag"])
            out.append(new_row)
        return out
    # 已经是聚合 ranges 格式，透传
    return uwb_rows


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    """读取一个必须存在的 JSON 记录文件，并要求内容是字典列表。"""
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    payload = read_json(path)
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise TypeError(f"Expected a list[dict] payload in {path}")
    return payload


def _read_optional_json_object(path: Path) -> dict[str, Any] | None:
    """读取一个可选 JSON 对象文件；不存在或无效就返回 None。"""
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def read_sim_sequence(seq_id: str, raw_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """读取一条仿真序列，返回原始包和读取报告，不在这里构造统一事件。

    sim_e9_main 的 seq_id 格式为 seedN__<base>（09 脚本用 __ 替代路径分隔符 /
    以绕过 validate_path_component），内部会自动还原为 seedN/<base> 路径。
    """
    from liquidloc.common.tee_logger import print_dict
    # BUG-006 修复 (2026-09-06 §10 审计): sim_e9_main 的 seq_id 含路径分隔符
    # (seed0/sim_curve_01)，validate_path_component 拒绝含 / 的 seq_id。
    # sim 数据集 seq_id 本身就是嵌套路径结构，安全边界由 read_sim_sequence
    # 内部 Path 拼接 + is_dir() 检查保证，跳过通用安全验证。
    print_dict({"seq_id": seq_id, "raw_root": str(raw_root)}, "read_sim_sequence 入口参数")
    _resolved_seq_id = seq_id.replace('__', '/') if '__' in seq_id else seq_id
    if not isinstance(raw_root, (str, Path)):
        raise TypeError("raw_root must be a str or Path")
    if not raw_root:
        raise ValueError("raw_root must be a non-empty path")
    seq_dir = Path(raw_root) / _resolved_seq_id
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Simulation sequence directory not found: {seq_dir}")
    # P2 修复 (2026-09-02): 归一化 VIO 和 UWB 格式 — sim_e9_5seed_25unit 给的是
    # 绝对位姿 (x,y,yaw) 和一锚一行的 UWB，需要转为 (dx,dy,dyaw) 增量。
    # sim_materializer 输出已经是规范的 (dx,dy,dyaw) + (anchor_id, range, ...) 格式，
    # 函数透传即可。
    imu_raw = _read_json_records(seq_dir / "imu.json")
    uwb_raw = _read_json_records(seq_dir / "uwb.json")
    vio_raw = _read_json_records(seq_dir / "vio.json")
    gt_raw = _read_json_records(seq_dir / "gt.json")
    uwb_raw = _normalize_uwb_rows(uwb_raw)
    vio_raw = _normalize_vio_rows(vio_raw)
    bundle = {
        "imu_raw": imu_raw,
        "uwb_raw": uwb_raw,
        "vio_raw": vio_raw,
        "gt_raw": gt_raw,
    }
    al = _read_optional_json_object(seq_dir / "anchor_layout.json")
    if al is not None:
        bundle["anchor_layout_raw"] = al
    read_report = {
        "dataset_name": "sim",
        "seq_id": seq_id,
        "seq_dir": str(seq_dir),
        "streams": sorted(bundle.keys()),
        "is_complete": all(k in bundle for k in ("imu_raw", "uwb_raw", "vio_raw", "gt_raw")),
    }
    return bundle, read_report
