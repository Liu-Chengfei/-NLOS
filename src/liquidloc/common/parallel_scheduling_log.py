"""并行调度记录写入 decision log（条款 [15] 落实模块）。

手册条款 [15]：「并行只改变调度顺序，不改变任何论文级口径
（每次运行前清场、数据与权重独立生成、统计按独立单元合并等规则不变
训练默认 GPU、CPU 仅限非训练负载，见 Pre-5），并行调度记录随 decision log 留痕」

本模块提供:
  - ParallelSchedulingRecord: dataclass，单次并行调度的完整记录
  - write_scheduling_record: 向 .audit/decision_log.json 写入调度条目
  - record_stage_scheduling: 便捷函数，一次性记录某个阶段的并行调度汇总
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "ParallelSchedulingRecord",
    "write_scheduling_record",
    "record_stage_scheduling",
    "load_decision_log",
    "save_decision_log",
]

_DECISION_LOG_DEFAULT = ".audit/decision_log.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_decision_log(path: str | Path = _DECISION_LOG_DEFAULT) -> dict[str, Any]:
    p = Path(path)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    # 初始化骨架
    return {
        "_schema_version": "1.0",
        "_generated_at": _utc_now_iso(),
        "_note": "Decision log: parallel scheduling records (clause [15]) + R-series entries.",
        "parallel_scheduling": [],
        "experiments": {},
    }


def save_decision_log(log: dict[str, Any], path: str | Path = _DECISION_LOG_DEFAULT) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    log["_generated_at"] = _utc_now_iso()
    p.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


@dataclass
class ParallelSchedulingRecord:
    """单次并行调度的完整记录（条款 [15] 要求）。"""
    record_id: str                # 如 "SCHED-20260915-001"
    timestamp: str                 # ISO-8601 时间戳
    stage_name: str                # 阶段名称，如 "stage1_neural" 或 "stage2_remaining"
    parallel_degree: int           # 实际使用的并行度（max_workers）
    gpu_parallel_limit: int        # GPU 限制
    ram_parallel_limit: int        # RAM 限制
    pending_tasks: int             # 待执行任务数
    final_parallel: int            # min(gpu, ram, pending) 最终值
    ram_limit_gb: float            # RAM 硬上限（GB）
    unit_ids: list[str]            # 本阶段调度的单元 ID 列表
    worker_assignments: dict[str, int]  # unit_id -> worker_index（thread index）
    decision: str                  # "proceed" 或 "reduced"（因资源不足降度）
    notes: str = ""                # 额外说明（如降度原因、GPU 型号等）


def write_scheduling_record(
    record: ParallelSchedulingRecord,
    log_path: str | Path = _DECISION_LOG_DEFAULT,
) -> Path:
    """向 .audit/decision_log.json 写入一条并行调度记录。"""
    log = load_decision_log(log_path)
    if "parallel_scheduling" not in log:
        log["parallel_scheduling"] = []
    log["parallel_scheduling"].append(asdict(record))
    return save_decision_log(log, log_path)


def record_stage_scheduling(
    stage_name: str,
    parallel_degree: int,
    gpu_parallel_limit: int,
    ram_parallel_limit: int,
    pending_tasks: int,
    final_parallel: int,
    ram_limit_gb: float,
    unit_ids: list[str],
    worker_assignments: dict[str, int],
    decision: str = "proceed",
    notes: str = "",
    log_path: str | Path = _DECISION_LOG_DEFAULT,
    record_counter: list[int] | None = None,
) -> ParallelSchedulingRecord:
    """便捷函数：根据资源预算计算结果，一次性记录某个阶段的并行调度。"""
    if record_counter is None:
        # 自动生成 record_id
        today = datetime.now().strftime("%Y%m%d")
        existing = load_decision_log(log_path).get("parallel_scheduling", [])
        seq = len(existing) + 1
        record_id = f"SCHED-{today}-{seq:03d}"
    else:
        record_counter[0] += 1
        today = datetime.now().strftime("%Y%m%d")
        record_id = f"SCHED-{today}-{record_counter[0]:03d}"

    record = ParallelSchedulingRecord(
        record_id=record_id,
        timestamp=_utc_now_iso(),
        stage_name=stage_name,
        parallel_degree=parallel_degree,
        gpu_parallel_limit=gpu_parallel_limit,
        ram_parallel_limit=ram_parallel_limit,
        pending_tasks=pending_tasks,
        final_parallel=final_parallel,
        ram_limit_gb=ram_limit_gb,
        unit_ids=unit_ids,
        worker_assignments=worker_assignments,
        decision=decision,
        notes=notes,
    )
    return write_scheduling_record(record, log_path)
