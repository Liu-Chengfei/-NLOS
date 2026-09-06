"""UTIL 原始表读取器，供公开基准烟雾流程使用。

这个模块负责把 UTIL 数据集序列目录下的 JSON 文件读成上层烟雾流程
能消费的原始包字典。UTIL 数据集与标准数据集的区别在于用 flow 和
tof 替代了 vio，所以原始包的键名和契约也相应不同。

上游依赖:
- 文件系统上的 UTIL 序列目录（包含 imu.json, uwb.json, flow.json, gt.json；
  tof.json 为可选，存在时读入但当前 pipeline 不消费）

下游调用者:
- 准备流程脚本（通过 dataio/__init__.py 的 read_util_sequence 入口调用）
- 公开基准烟雾测试脚本

核心变量:
- bundle: 按内部约定组织的 UTIL 原始包字典
- read_report: 读取摘要报告

关键设计决策:
- tof.json 为可选文件，存在时读入 bundle，不存在时跳过
- 当前 prepare_pipeline 不消费 ToF 数据（缺少 anchor_id 无法桥接到 UWB）
- 读取报告包含 tof_available 字段，便于后续处理逻辑判断
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from liquidloc.common.constants import MODALITY_FLOW, MODALITY_IMU, MODALITY_UWB
from liquidloc.common.io_utils import read_json
from liquidloc.common.validation import validate_path_component
from liquidloc.protocol.sensor_contract import get_required_payload_fields as _get_sensor_fields


_REQUIRED_STREAM_FILES = {
    "imu": "imu.json",
    "uwb": "uwb.json",
    "flow": "flow.json",
    "gt": "gt.json",
}

_SENSOR_PAYLOAD_FIELDS = _get_sensor_fields()
_REQUIRED_STREAM_FIELDS = {
    "imu": ("timestamp",) + _SENSOR_PAYLOAD_FIELDS[MODALITY_IMU],
    "uwb": ("timestamp",) + _SENSOR_PAYLOAD_FIELDS[MODALITY_UWB],
    "flow": ("timestamp",) + _SENSOR_PAYLOAD_FIELDS[MODALITY_FLOW],
    "gt": ("timestamp", "px", "py", "yaw"),
}

__all__ = ("read_util_sequence", "inspect_util_raw_readiness")


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    """读取一个必须存在的 JSON 记录文件，并要求内容是字典列表。"""
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    payload = read_json(path)
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise TypeError(f"Expected a list[dict] payload in {path}")
    return payload


def _read_optional_json_records(path: Path) -> list[dict[str, Any]] | None:
    """读取可选的 JSON 记录文件；不存在或内容无效就返回 None。"""
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except ValueError:
        return None
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        return None
    return payload


def read_util_sequence(seq_id: str, raw_root):
    """读取一条 UTIL 序列，返回原始包和读取报告。

    tof.json 为可选文件：存在时读入 bundle，不存在时跳过。
    当前 prepare_pipeline 不消费 ToF 数据（缺少 anchor_id 无法桥接到 UWB），
    但 _reject_unsupported_util_tof_bridge 会在 ToF 非空时显式报错，
    防止静默丢弃测量数据。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({"seq_id": seq_id, "raw_root": str(raw_root)}, "read_util_sequence 入口参数")
    if not seq_id:
        raise ValueError("seq_id must be a non-empty string")

    seq_dir = Path(raw_root) / seq_id
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"UTIL sequence directory not found: {seq_dir}")

    bundle: dict[str, Any] = {
        "imu_raw": _read_json_records(seq_dir / "imu.json"),
        "uwb_raw": _read_json_records(seq_dir / "uwb.json"),
        "flow_raw": _read_json_records(seq_dir / "flow.json"),
        "gt_raw": _read_json_records(seq_dir / "gt.json"),
    }
    tof_records = _read_optional_json_records(seq_dir / "tof.json")
    if tof_records is not None:
        bundle["tof_raw"] = tof_records
    read_report = {
        "dataset_name": "util",
        "seq_id": seq_id,
        "seq_dir": str(seq_dir),
        "streams": {key: len(value) for key, value in bundle.items() if isinstance(value, list)},
        "is_complete": all(bundle[key] for key in ("imu_raw", "uwb_raw", "flow_raw", "gt_raw")),
        "tof_available": tof_records is not None,
    }
    return bundle, read_report


def _list_sequence_dirs(raw_root: Path) -> list[Path]:
    """列出可处理的序列目录，排除 config 和隐藏目录。"""
    from liquidloc.dataio.readers import list_sequence_dirs

    return list_sequence_dirs(raw_root)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """去重但保留原始顺序。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _probe_required_streams(seq_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """检查一条序列里必需流的存在性、行数和基础字段。"""
    stream_report: dict[str, Any] = {}
    missing_or_invalid: list[str] = []
    for stream_name, filename in _REQUIRED_STREAM_FILES.items():
        stream_path = seq_dir / filename
        stream_state = {
            "path": str(stream_path),
            "present": stream_path.is_file(),
            "row_count": None,
            "non_empty": False,
        }
        if not stream_state["present"]:
            missing_or_invalid.append(stream_name)
            stream_report[stream_name] = stream_state
            continue
        try:
            rows = _read_json_records(stream_path)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            stream_state["error"] = f"{type(exc).__name__}: {exc}"
            missing_or_invalid.append(stream_name)
        else:
            stream_state["row_count"] = len(rows)
            stream_state["non_empty"] = bool(rows)
            if not rows:
                stream_state["error"] = "empty_stream"
                missing_or_invalid.append(stream_name)
            else:
                required_fields = _REQUIRED_STREAM_FIELDS[stream_name]
                first_row = rows[0]
                missing_fields = [field_name for field_name in required_fields if field_name not in first_row]
                stream_state["required_fields"] = list(required_fields)
                stream_state["missing_fields"] = missing_fields
                if missing_fields:
                    stream_state["error"] = f"missing_fields:{','.join(missing_fields)}"
                    missing_or_invalid.append(stream_name)
        stream_report[stream_name] = stream_state
    return stream_report, missing_or_invalid


def _inspect_util_sequence_readiness(seq_id: str, raw_root: Path) -> dict[str, Any]:
    """检查一条 UTIL 序列是否满足公开基准烟雾要求。"""
    seq_dir = raw_root / seq_id
    stream_report, missing_streams = _probe_required_streams(seq_dir)
    reasons: list[str] = []
    if missing_streams:
        reasons.append("missing_required_streams")
    ready = not reasons
    return {
        "seq_id": seq_id,
        "seq_dir": str(seq_dir),
        "status": "ready" if ready else "not_ready",
        "ready_for_public_benchmark_smoke": ready,
        "reasons": _dedupe_preserve_order(reasons),
        "missing_streams": missing_streams,
        "required_streams": stream_report,
    }


def inspect_util_raw_readiness(raw_root: str | Path) -> dict[str, Any]:
    """检查 UTIL 原始根目录是否满足公开基准烟雾要求。"""
    raw_root_path = Path(raw_root)
    sequence_dirs = _list_sequence_dirs(raw_root_path)
    sequence_reports = {
        seq_dir.name: _inspect_util_sequence_readiness(seq_dir.name, raw_root_path)
        for seq_dir in sequence_dirs
    }
    reasons: list[str] = []
    if not sequence_dirs:
        reasons.append("missing_raw_sequence")
    for seq_report in sequence_reports.values():
        if seq_report["status"] != "ready":
            reasons.extend(seq_report["reasons"])
    ready_sequence_count = sum(
        1
        for seq_report in sequence_reports.values()
        if seq_report["ready_for_public_benchmark_smoke"]
    )
    overall_ready = bool(sequence_dirs) and ready_sequence_count == len(sequence_reports)
    report = {
        "dataset_name": "util",
        "raw_root": str(raw_root_path),
        "raw_root_exists": raw_root_path.is_dir(),
        "status": "ready" if overall_ready else "not_ready",
        "gate_action": "pass" if overall_ready else "skipped",
        "ready_for_public_benchmark_smoke": overall_ready,
        "reasons": _dedupe_preserve_order(reasons),
        "sequence_count": len(sequence_reports),
        "ready_sequence_count": ready_sequence_count,
        "required_streams": list(_REQUIRED_STREAM_FILES.keys()),
        "sequence_ids": list(sequence_reports.keys()),
        "sequences": sequence_reports,
    }
    if not raw_root_path.is_dir():
        report["raw_root_error"] = f"raw_root_not_found: {raw_root_path}"
    return report
