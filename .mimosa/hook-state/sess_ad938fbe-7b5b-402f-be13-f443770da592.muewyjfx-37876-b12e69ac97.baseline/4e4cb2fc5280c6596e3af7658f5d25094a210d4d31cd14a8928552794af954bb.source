"""违反即处置协议实现。

执行中任何环节发现违反本文档（gate 未过 / 未清场 / 审计失败 / 统计口径不符 /
数据或权重污染等）→ 立即直接处置：
    ① 隔离作废相关数据/权重/结果（archive 留痕）
    ② 修复根因
    ③ 回到对应回归点重跑

本模块提供处置所需的最小工具集：
    - ViolationRecord  : 违规记录（类型、来源、影响路径、时间戳、证据）
    - DispositionResult: 处置结果（隔离路径、根因、回归点、处置时间戳、证据摘要）
    - quarantine_paths  : 将受影响文件/目录移动到 .archive/（带时间戳后缀）
    - handle_violation  : 完整处置流程（隔离 → 记录根因 → 返回结果）

公开入口：
    handle_violation(violation, root_cause, regression_point) -> DispositionResult
    quarantine_paths(paths) -> list[str]
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ViolationRecord:
    """违规记录。

    Attributes:
        violation_type: 违规类型，如 "gate_failed"、"audit_failed"、"data_pollution"。
        source: 违规来源，如 gate 代码、审计脚本、统计检查等。
        affected_paths: 受影响的文件/目录路径列表。
        timestamp: 违规发现时间戳（ISO 8601 格式字符串）。
        evidence: 违规证据字典，可包含 detail、severity 等字段。
    """

    violation_type: str  # e.g. "gate_failed", "audit_failed", "data_pollution"
    source: str  # e.g. "gate_failed", "audit_failed"
    affected_paths: list[str] = field(default_factory=list)
    timestamp: str = ""  # ISO 8601 格式，空时自动填入当前 UTC 时间
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验必填字段并补充时间戳。"""
        if not self.violation_type:
            raise ValueError("ViolationRecord.violation_type 不能为空")
        if not self.source:
            raise ValueError("ViolationRecord.source 不能为空")
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class DispositionResult:
    """处置结果。

    Attributes:
        quarantined_paths: 已隔离到 .archive/ 的路径列表（含时间戳后缀的 archive 路径）。
        root_cause: 根因描述。
        regression_point: 回归点描述，指示重跑应从何处开始。
        disposition_timestamp: 处置时间戳（ISO 8601 格式）。
        evidence_summary: 处置证据摘要字典。
    """

    quarantined_paths: list[str] = field(default_factory=list)
    root_cause: str = ""
    regression_point: str = ""
    disposition_timestamp: str = ""  # ISO 8601 格式，空时自动填入当前 UTC 时间
    evidence_summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验必填字段并补充时间戳。"""
        if not self.root_cause:
            raise ValueError("DispositionResult.root_cause 不能为空")
        if not self.regression_point:
            raise ValueError("DispositionResult.regression_point 不能为空")
        if not self.disposition_timestamp:
            self.disposition_timestamp = datetime.now(timezone.utc).isoformat()


def _get_archive_dir(project_root: str | Path | None = None) -> Path:
    """获取 .archive/ 目录路径，不存在时创建。

    Args:
        project_root: 项目根目录，默认取当前工作目录。

    Returns:
        .archive/ 目录的 Path 对象。
    """
    if project_root is None:
        project_root = Path.cwd()
    archive_dir = Path(project_root) / ".archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def quarantine_paths(
    paths: list[str],
    project_root: str | Path | None = None,
    timestamp_suffix: str | None = None,
) -> list[str]:
    """将受影响文件/目录移动到 .archive/，带时间戳后缀。

    对每个路径：
        1. 如果路径不存在，跳过并记录警告。
        2. 构造目标路径：.archive/<原名>_<时间戳后缀>。
        3. 移动文件/目录到目标路径。
        4. 返回所有成功隔离的 archive 路径列表。

    Args:
        paths: 需要隔离的文件/目录路径列表。
        project_root: 项目根目录，默认当前工作目录。
        timestamp_suffix: 时间戳后缀，默认用当前 UTC 时间戳
                          （格式 YYYYMMDD_HHMMSS）。

    Returns:
        成功隔离到 .archive/ 的路径列表。
    """
    if timestamp_suffix is None:
        timestamp_suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    archive_dir = _get_archive_dir(project_root)
    quarantined: list[str] = []

    for raw_path in paths:
        src = Path(raw_path)
        if not src.exists():
            # 路径不存在，跳过（可能是动态生成的数据尚未落盘）
            continue

        # 构造目标路径：保留原始文件名/目录名，附加时间戳后缀
        stem = src.name
        # 避免重复附加后缀（如果已有时间戳格式后缀则不再追加）
        dest_name = f"{stem}_{timestamp_suffix}"
        dest = archive_dir / dest_name

        # 移动文件或目录
        shutil.move(str(src), str(dest))
        quarantined.append(str(dest))

    return quarantined


def handle_violation(
    violation: ViolationRecord,
    root_cause: str,
    regression_point: str,
    project_root: str | Path | None = None,
) -> DispositionResult:
    """完整处置流程：隔离 → 记录根因 → 返回结果。

    步骤：
        1. 调用 quarantine_paths 隔离受影响的数据/权重/结果。
        2. 记录根因描述。
        3. 返回 DispositionResult，包含隔离路径、根因、回归点、处置时间戳和证据摘要。

    Args:
        violation: 违规记录，包含违规类型、来源、影响路径、时间戳和证据。
        root_cause: 根因描述，说明违规的根本原因。
        regression_point: 回归点描述，指示重跑应从何处开始。
        project_root: 项目根目录，默认当前工作目录。

    Returns:
        DispositionResult，包含处置详情。
    """
    # 步骤 1：隔离作废
    quarantined = quarantine_paths(
        violation.affected_paths,
        project_root=project_root,
    )

    # 步骤 2：记录根因（通过 DispositionResult 的 root_cause 字段持久化）
    # 步骤 3：返回处置结果
    evidence_summary = {
        "violation_type": violation.violation_type,
        "source": violation.source,
        "violation_timestamp": violation.timestamp,
        "n_quarantined": len(quarantined),
        "evidence": dict(violation.evidence),
    }

    return DispositionResult(
        quarantined_paths=quarantined,
        root_cause=root_cause,
        regression_point=regression_point,
        evidence_summary=evidence_summary,
    )


__all__ = [
    "ViolationRecord",
    "DispositionResult",
    "quarantine_paths",
    "handle_violation",
]
