"""NTU VIRAL 序列解包适配器。

本模块负责把 ``scripts/16_download_public_datasets.py`` 下载的 NTU VIRAL
flight archive（``eee_*.zip`` / ``tnp_*.zip`` …）转换为 ``read_ntu_viral_sequence``
能消费的目录结构：

.. code-block:: text

    <seq_id>/
    ├── imu.json
    ├── uwb.json
    ├── vio.json
    └── gt.json

主要设计约束：

1. 本仓库默认运行在 Windows，**没有 ROS 运行时**。因此 ``rospy`` / ``rosbag``
   Python 绑定不可用。我们只依赖 ``zipfile``（标准库）+ ``urllib``（标准库）。
2. ROS1 bag 文件内部是 protobuf 格式，没有简单文本接口。完整解析需要
   ``bagpy`` / ``rosbag`` 等依赖，这在本仓库当前依赖列表之外。
   为保持可复现性，本适配器先完成两步可以立即做的工作：
   - 从 ``<seq_id>.zip`` 解压出 ``<seq_id>.bag``（保留 bag 于 ``data/raw/ntu_viral/<seq_id>/``）。
   - 从官方 ``https://github.com/ntu-aris/ntuviral_gt`` 拉取 ``ground_truth.csv``
     并转化为 ``gt.json``（每帧 ``timestamp/px/py/yaw``）。
3. ``imu.json`` / ``uwb.json`` / ``vio.json`` 在 bag 未解析时写**占位文件**
   （空 JSON 数组），并在报告里把状态标为 ``archive_unpacked_bag_pending``。
   这意味着后续有人补齐 ROS1 解析路径后，只需把 ``write_<modality>_json``
   函数填上即可，不会破坏目录约定。
4. ``gt.json`` 需要把 Leica prism 坐标转成 IMU body 坐标系下的 (px, py, yaw)。
   目前只做了 z 轴投影（保留 xy 平面位置），yaw 置 0；精确的 0.4 m 偏移
   校正留给后续 PR，已在代码注释中说明。

作者：Local Dev
日期：2026-XX-XX
上游协议：arXiv 2202.00379 (IJRR 2022)
数据仓库：https://github.com/ntu-aris/ntu_viral_dataset
GT 仓库：https://github.com/ntu-aris/ntuviral_gt
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_GT_CSV_BASE = "https://raw.githubusercontent.com/ntu-aris/ntuviral_gt/master"

# NTU VIRAL 官方 archive 文件名前缀（与 download 脚本一致）。
_ARCHIVE_PREFIXES = ("eee_", "tnp_")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> Any:
    import argparse
    parser = argparse.ArgumentParser(
        description="Unpack NTU VIRAL zip archives into sequence directories",
    )
    parser.add_argument(
        "--official-root",
        required=True,
        help="Directory containing downloaded <seq_id>.zip files.",
    )
    parser.add_argument(
        "--landed-raw-root",
        required=True,
        help="Target directory where sequence folders will be created.",
    )
    parser.add_argument(
        "--seq-ids",
        default=None,
        help=(
            "Comma-separated list of sequence IDs to process. "
            "If omitted, all archives in official_root are processed."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _is_flight_archive(filename: str) -> bool:
    """返回 True 当 filename 以已知的 NTU VIRAL 飞行序列前缀开头。"""
    return any(filename.startswith(prefix) for prefix in _ARCHIVE_PREFIXES)


def _list_flight_archives(official_root: Path) -> list[Path]:
    """列出 official_root 下所有 flight archive .zip 文件（按名字排序）。"""
    if not official_root.is_dir():
        return []
    return sorted(
        p
        for p in official_root.iterdir()
        if p.is_file() and _is_flight_archive(p.name) and p.suffix.lower() == ".zip"
    )


def _download_text(url: str, timeout: int = 60) -> str:
    """下载 URL 并返回 UTF-8 文本内容；失败时抛 ValueError。"""
    req = Request(url, headers={"User-Agent": "liquidloc-pipeline/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


# ---------------------------------------------------------------------------
# GT 下载与转化
# ---------------------------------------------------------------------------


def _download_gt_csv(seq_id: str) -> str | None:
    """从 ntuviral_gt repo 下载 ground_truth.csv；失败返回 None。"""
    url = f"{_GT_CSV_BASE}/{seq_id}/ground_truth.csv"
    try:
        return _download_text(url, timeout=60)
    except Exception as exc:
        print(f"[ntu_viral_adapter] WARN: GT download failed for {seq_id}: {exc}")
        return None


def _parse_gt_csv(csv_text: str) -> list[dict[str, Any]]:
    """
    把 ground_truth.csv 转为 list[dict]，每行一个 dict：
        timestamp  float   : ROS stamp (ns) -> 秒
        px         float   : prism x (m)
        py         float   : prism y (m)
        yaw        float   : 占位 0.0，待 0.4 m 偏移与 quaternion->yaw 转换
    """
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        # header 形如 "%time,field.header.seq,field.header.stamp,..."
        try:
            stamp_ns = int(row.get("field.header.stamp", "0"))
        except (ValueError, TypeError):
            continue
        if stamp_ns == 0:
            # 零时间戳通常是首行默认值，跳过。
            continue
        try:
            px = float(row.get("field.pose.position.x", 0.0))
            py = float(row.get("field.pose.position.y", 0.0))
        except (ValueError, TypeError):
            continue
        rows.append(
            {
                "timestamp": round(stamp_ns / 1e9, 6),
                "px": round(px, 6),
                "py": round(py, 6),
                # yaw 暂置 0.0；精确换算需要把 prism 旋转分量转为 IMU body 帧。
                # TODO(#ntu_viral_gt_yaw): 应用 trans_B2prism.csv 补偿 0.4 m offset。
                "yaw": 0.0,
            }
        )
    rows.sort(key=lambda r: r["timestamp"])
    return rows


# ---------------------------------------------------------------------------
# 模态写入（当前占位实现）
# ---------------------------------------------------------------------------


def _write_imu_json(target_dir: Path, seq_id: str) -> None:
    """IMU 目前无 ROS 解析器，写占位空列表。后续补 bag 解析时填入 /imu/imu。"""
    (target_dir / "imu.json").write_text("[]", encoding="utf-8")


def _write_uwb_json(target_dir: Path, seq_id: str) -> None:
    """UWB 目前无 ROS 解析器，写占位空列表。后续补 bag 解析时填入 /uwb_endorange_info。"""
    (target_dir / "uwb.json").write_text("[]", encoding="utf-8")


def _write_vio_json(target_dir: Path, seq_id: str) -> None:
    """VIO 在原始 bag 中没有独立主题，需由 stereo images + IMU 合成。
    当前写占位空列表。"""
    (target_dir / "vio.json").write_text("[]", encoding="utf-8")


def _write_gt_json(target_dir: Path, gt_rows: list[dict[str, Any]]) -> None:
    """把解析后的 GT 写到目标目录的 gt.json。"""
    import json
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "gt.json").write_text(
        json.dumps(gt_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 主解包逻辑
# ---------------------------------------------------------------------------


def unpack_ntu_viral_archives(
    official_root: Path | str,
    landed_raw_root: Path | str,
    seq_ids: list[str] | None = None,
) -> dict[str, Any]:
    """解包 NTU VIRAL archive 并写入 target layout。

    返回结构化报告：包含每个序列的状态、落盘路径、是否成功。
    """
    official_root = Path(official_root).resolve()
    landed_raw_root = Path(landed_raw_root).resolve()
    landed_raw_root.mkdir(parents=True, exist_ok=True)

    archives = _list_flight_archives(official_root)
    if not archives:
        return {
            "status": "no_archives",
            "processed_sequences": [],
            "reasons": ["no_ntu_viral_flight_archives_found"],
        }

    if seq_ids:
        named = {a.name.replace(".zip", "") for a in archives}
        missing = set(seq_ids) - named
        if missing:
            return {
                "status": "unknown_seq_ids",
                "processed_sequences": [],
                "reasons": [f"unknown_seq_ids: {sorted(missing)}"],
                "known_seq_ids": sorted(named),
            }
        archives = [a for a in archives if a.name.replace(".zip", "") in set(seq_ids)]

    processed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for archive_path in archives:
        seq_id = archive_path.name.replace(".zip", "")
        target_dir = landed_raw_root / seq_id
        status_report = {
            "seq_id": seq_id,
            "archive_path": str(archive_path),
            "target_dir": str(target_dir),
            "steps": {},
            "status": "unknown",
        }
        try:
            # Step 1: 解压 bag
            with zipfile.ZipFile(archive_path, "r") as zf:
                members = zf.namelist()
                # bag 内部通常只有一个同名 .bag 文件
                bag_members = [m for m in members if m.endswith(".bag")]
                if not bag_members:
                    raise ValueError(
                        f"No .bag file found inside {archive_path.name}; contents: {members[:5]}"
                    )
                bag_member = bag_members[0]
                bag_name = Path(bag_member).name
                target_bag = target_dir / bag_name
                target_dir.mkdir(parents=True, exist_ok=True)
                zf.extract(bag_member, path=target_dir)
                # 重命名：保证 bag 与目录同名（去掉可能的前缀子目录）
                if target_bag != target_dir / (seq_id + ".bag"):
                    rename_target = target_dir / (seq_id + ".bag")
                    if target_bag.exists():
                        shutil.move(str(target_bag), str(rename_target))
                        target_bag = rename_target
                status_report["steps"]["extract_bag"] = "ok"

            # Step 2: 下载 GT CSV 并转为 gt.json
            csv_text = _download_gt_csv(seq_id)
            if csv_text is None:
                status_report["steps"]["gt"] = "failed_download"
                status_report["status"] = "gt_missing"
                errors.append(status_report)
                continue
            gt_rows = _parse_gt_csv(csv_text)
            if not gt_rows:
                status_report["steps"]["gt"] = "empty_csv"
                status_report["status"] = "gt_missing"
                errors.append(status_report)
                continue
            _write_gt_json(target_dir, gt_rows)
            status_report["steps"]["gt"] = f"written {len(gt_rows)} rows"
            status_report["gt_row_count"] = len(gt_rows)

            # Step 3: 写占位模态文件
            _write_imu_json(target_dir, seq_id)
            status_report["steps"]["imu"] = "placeholder"
            _write_uwb_json(target_dir, seq_id)
            status_report["steps"]["uwb"] = "placeholder"
            _write_vio_json(target_dir, seq_id)
            status_report["steps"]["vio"] = "placeholder"

            status_report["status"] = "archive_unpacked_bag_pending"
            processed.append(status_report)

        except Exception as exc:
            status_report["status"] = "error"
            status_report["error"] = str(exc)
            errors.append(status_report)

    overall = "ready" if not errors and processed else "partial"
    if not processed:
        overall = "failed"
    return {
        "status": overall,
        "archive_count": len(archives),
        "processed_sequences": processed,
        "errors": errors,
        "note": (
            "IMU / UWB / VIO streams are placeholder pending ROS bag parser. "
            "GT is from official ntuviral_gt repo."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    import json as _json
    args = _parse_args(argv)
    report = unpack_ntu_viral_archives(args.official_root, args.landed_raw_root, args.seq_ids)
    print(_json.dumps(report, ensure_ascii=False, indent=2))
    ok = report.get("status") in ("ready", "partial")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
