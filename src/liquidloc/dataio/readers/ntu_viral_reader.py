"""NTU VIRAL 数据读取器。

职责：
    读取 NTU VIRAL 数据集的 JSON 原始数据包，提供序列读取和就绪检查功能。

依赖：
    - 标准 JSON 数据格式：imu.json, uwb.json, vio.json, gt.json
    - 目录结构：单序列独立目录，包含所有原始数据文件

导出函数：
    - read_ntu_viral_sequence: 读取单条序列的原始数据包
    - inspect_ntu_viral_raw_readiness: 检查数据目录就绪状态
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_ntu_viral_sequence(
    seq_name: str,
    base_dir: Path | str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """读取 NTU VIRAL 序列的原始 JSON 数据包。

    参数：
        seq_name: 序列目录名（相对于 base_dir）。
        base_dir: 包含序列子目录的根目录。

    返回：
        bundle: 包含 imu_raw, uwb_raw, vio_raw, gt_raw 列表的字典。
        report: 读取报告，包含 dataset_name, streams, is_complete 等字段。
    """
    base_dir = Path(base_dir)
    seq_dir = base_dir / seq_name

    bundle: dict[str, list[dict[str, Any]]] = {
        "imu_raw": [],
        "uwb_raw": [],
        "vio_raw": [],
        "gt_raw": [],
    }

    stream_files = {
        "imu_raw": "imu.json",
        "uwb_raw": "uwb.json",
        "vio_raw": "vio.json",
        "gt_raw": "gt.json",
    }

    for stream_key, filename in stream_files.items():
        file_path = seq_dir / filename
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                bundle[stream_key] = json.load(f)

    report = {
        "dataset_name": "ntu_viral",
        "streams": {
            "imu_raw": 1 if bundle["imu_raw"] else 0,
            "uwb_raw": 1 if bundle["uwb_raw"] else 0,
            "vio_raw": 1 if bundle["vio_raw"] else 0,
            "gt_raw": 1 if bundle["gt_raw"] else 0,
        },
        "is_complete": all(bundle.values()),
    }

    return bundle, report


def inspect_ntu_viral_raw_readiness(
    base_dir: Path | str,
) -> dict[str, Any]:
    """检查 NTU VIRAL 原始数据目录的就绪状态。

    参数：
        base_dir: 包含序列子目录的根目录。

    返回：
        report: 包含 seq_viral 检查结果的报告。
    """
    base_dir = Path(base_dir)
    report = {
        "status": "ready",
        "gate_action": "proceed",
        "sequences": {},
    }

    if not base_dir.exists():
        report["status"] = "not_ready"
        report["gate_action"] = "skipped"
        return report

    required_streams = {
        "imu": ("imu.json", ("timestamp", "ax", "ay", "gz")),
        "uwb": ("uwb.json", ("timestamp", "anchor_id", "range", "valid", "quality")),
        "vio": ("vio.json", ("timestamp", "dx", "dy", "dyaw", "quality", "tracked_features", "reproj_err")),
        "gt": ("gt.json", ("timestamp", "px", "py", "yaw")),
    }

    for seq_dir in sorted(base_dir.iterdir()):
        if not seq_dir.is_dir():
            continue

        seq_name = seq_dir.name
        missing = []
        missing_fields = {}

        for stream_name, (stream_file, required_fields) in required_streams.items():
            file_path = seq_dir / stream_file

            if not file_path.exists():
                missing.append(stream_name)
                missing_fields[stream_name] = list(required_fields)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not data:
                    missing.append(stream_name)
                    missing_fields[stream_name] = list(required_fields)
                else:
                    absent_fields = [field for field in required_fields if field not in data[0]]
                    if absent_fields:
                        missing.append(stream_name)
                        missing_fields[stream_name] = absent_fields

        seq_report = {
            "status": "ready" if not missing else "not_ready",
            "missing_streams": missing,
            "required_streams": {
                stream_name: {"missing_fields": missing_fields.get(stream_name, [])}
                for stream_name in required_streams
            },
        }

        if missing:
            seq_report["reasons"] = ["missing_required_streams"]
            if report["status"] == "ready":
                report["status"] = "not_ready"
                report["gate_action"] = "skipped"

        report["sequences"][seq_name] = seq_report

    return report
