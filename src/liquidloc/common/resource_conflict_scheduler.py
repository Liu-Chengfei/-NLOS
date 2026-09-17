"""
Resource conflict scheduler for clause [16].

When resources conflict, schedule with stagger/queue execution:
- Do NOT expand resources
- Do NOT downgrade paper-level metrics
- Stagger tasks or queue them when conflicts detected
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ResourceConflictScheduler:
    """Schedules tasks with stagger/queue when resource conflicts occur.

    Clause [16]: 资源冲突时按错峰/排队执行，不扩资源、不降级口径

    When resources (GPU, RAM) conflict:
    1. Stagger: delay task start times to avoid simultaneous peak usage
    2. Queue: serialize tasks that cannot run in parallel
    3. Never expand resources (no more GPU/RAM than available)
    4. Never downgrade paper-level metrics (same data/config/weights)
    """

    def __init__(
        self,
        log_dir: str = ".decision_logs",
        ram_limit_gb: float = 12.0,
        default_stagger_seconds: float = 5.0,
    ) -> None:
        self.log_dir = log_dir
        self.ram_limit_gb = ram_limit_gb
        self.default_stagger_seconds = default_stagger_seconds
        self._scheduling_records: List[Dict[str, Any]] = []
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

    def check_resource_conflict(
        self,
        pending_tasks: List[Dict[str, Any]],
        available_gpu: int,
        available_ram_gb: float,
        unit_peak_ram_gb: float,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Check if pending tasks have resource conflicts.

        Args:
            pending_tasks: List of pending task dicts with 'unit_id', 'gpu_required', 'ram_gb'
            available_gpu: Available GPU count
            available_ram_gb: Available RAM in GB
            unit_peak_ram_gb: Peak RAM per unit in GB

        Returns:
            Tuple of (has_conflict, conflict_details)
        """
        total_gpu_needed = sum(t.get("gpu_required", 1) for t in pending_tasks)
        total_ram_needed = sum(t.get("ram_gb", unit_peak_ram_gb) for t in pending_tasks)

        conflicts = []

        if total_gpu_needed > available_gpu:
            conflicts.append({
                "type": "gpu_conflict",
                "needed": total_gpu_needed,
                "available": available_gpu,
                "detail": f"GPU conflict: {total_gpu_needed} tasks need GPU, only {available_gpu} available",
            })

        if total_ram_needed > available_ram_gb:
            conflicts.append({
                "type": "ram_conflict",
                "needed_gb": total_ram_needed,
                "available_gb": available_ram_gb,
                "detail": f"RAM conflict: {total_ram_needed:.2f} GB needed, only {available_ram_gb:.2f} GB available",
            })

        return len(conflicts) > 0, conflicts

    def stagger_and_queue(
        self,
        tasks: List[Dict[str, Any]],
        available_gpu: int,
        available_ram_gb: float,
        unit_peak_ram_gb: float,
    ) -> Dict[str, Any]:
        """Schedule tasks with stagger/queue when resources conflict.

        Args:
            tasks: List of task dicts with 'unit_id', 'gpu_required', 'ram_gb'
            available_gpu: Available GPU count
            available_ram_gb: Available RAM in GB
            unit_peak_ram_gb: Peak RAM per unit in GB

        Returns:
            Scheduling result with staggered/queued execution plan
        """
        has_conflict, conflicts = self.check_resource_conflict(
            tasks, available_gpu, available_ram_gb, unit_peak_ram_gb
        )

        # Calculate parallel capacity
        gpu_capacity = available_gpu if available_gpu > 0 else 0
        ram_capacity = int(available_ram_gb / unit_peak_ram_gb) if unit_peak_ram_gb > 0 else 0
        parallel_capacity = min(gpu_capacity, ram_capacity, len(tasks))

        if not has_conflict and parallel_capacity >= len(tasks):
            # No conflict, all tasks can run in parallel
            schedule = {
                "mode": "parallel",
                "tasks": [
                    {**t, "scheduled_start": 0.0, "scheduled_end": None, "stagger_delay": 0.0}
                    for t in tasks
                ],
                "total_parallel": len(tasks),
            }
        elif parallel_capacity > 0:
            # Stagger: run tasks in waves with delay between waves
            waves = []
            wave_size = parallel_capacity
            stagger_delay = self.default_stagger_seconds

            for i in range(0, len(tasks), wave_size):
                wave = tasks[i:i + wave_size]
                wave_start = i * stagger_delay if i > 0 else 0.0
                waves.append({
                    "wave_index": i // wave_size,
                    "tasks": [
                        {**t, "scheduled_start": wave_start, "scheduled_end": None, "stagger_delay": stagger_delay}
                        for t in wave
                    ],
                    "start_offset": wave_start,
                })

            schedule = {
                "mode": "staggered",
                "waves": waves,
                "total_parallel": parallel_capacity,
                "stagger_delay_seconds": stagger_delay,
                "conflicts": conflicts,
            }
        else:
            # Queue: serialize all tasks (no parallel capacity)
            schedule = {
                "mode": "queued",
                "tasks": [
                    {**t, "scheduled_start": i * self.default_stagger_seconds, "scheduled_end": None, "stagger_delay": self.default_stagger_seconds}
                    for i, t in enumerate(tasks)
                ],
                "total_parallel": 0,
                "conflicts": conflicts,
            }

        # Record scheduling decision
        record = {
            "timestamp": datetime.now().isoformat(),
            "clause": "[16]",
            "mode": schedule["mode"],
            "available_gpu": available_gpu,
            "available_ram_gb": available_ram_gb,
            "parallel_capacity": parallel_capacity,
            "task_count": len(tasks),
            "conflicts": conflicts,
            "schedule": schedule,
        }
        self._scheduling_records.append(record)

        return record

    def write_scheduling_log(self, filename: Optional[str] = None) -> str:
        """Write scheduling log to JSON file.

        Args:
            filename: Optional custom filename

        Returns:
            Path to written log file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resource_conflict_scheduling_{timestamp}.json"

        log_path = os.path.join(self.log_dir, filename)

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "clause": "[16]",
            "principle": "资源冲突时按错峰/排队执行，不扩资源、不降级口径",
            "ram_limit_gb": self.ram_limit_gb,
            "scheduling_records": self._scheduling_records,
            "summary": self.get_summary(),
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        return log_path

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all scheduling decisions."""
        parallel_count = sum(1 for r in self._scheduling_records if r["mode"] == "parallel")
        stagger_count = sum(1 for r in self._scheduling_records if r["mode"] == "staggered")
        queue_count = sum(1 for r in self._scheduling_records if r["mode"] == "queued")
        conflict_count = sum(1 for r in self._scheduling_records if r.get("conflicts"))

        return {
            "total_records": len(self._scheduling_records),
            "parallel_mode": parallel_count,
            "staggered_mode": stagger_count,
            "queued_mode": queue_count,
            "conflict_count": conflict_count,
            "ram_limit_gb": self.ram_limit_gb,
        }


def write_scheduling_record(
    stage_name: str,
    parallel_degree: int,
    gpu_parallel_limit: int,
    ram_parallel_limit: int,
    pending_tasks: int,
    final_parallel: int,
    ram_limit_gb: float,
    unit_ids: List[str],
    worker_assignments: Dict[str, int],
    decision: str,
    log_path: str,
) -> str:
    """Write a scheduling record to a decision log file.

    Args:
        stage_name: Name of the scheduling stage
        parallel_degree: Requested parallel degree
        gpu_parallel_limit: GPU-limited parallel degree
        ram_parallel_limit: RAM-limited parallel degree
        pending_tasks: Number of pending tasks
        final_parallel: Final parallel degree (min of all limits)
        ram_limit_gb: RAM limit in GB
        unit_ids: List of unit IDs being scheduled
        worker_assignments: Worker assignments per unit
        decision: Scheduling decision (proceed/queue/stagger)
        log_path: Path to decision log file

    Returns:
        Path to the updated log file
    """
    log_path_obj = Path(log_path)

    # Read existing log if it exists
    if log_path_obj.is_file():
        try:
            log_data = json.loads(log_path_obj.read_text(encoding="utf-8"))
        except Exception:
            log_data = {}
    else:
        log_data = {}

    # Ensure required fields
    if "parallel_scheduling" not in log_data:
        log_data["parallel_scheduling"] = []

    # Add new scheduling record
    record = {
        "stage_name": stage_name,
        "parallel_degree": parallel_degree,
        "gpu_parallel_limit": gpu_parallel_limit,
        "ram_parallel_limit": ram_parallel_limit,
        "pending_tasks": pending_tasks,
        "final_parallel": final_parallel,
        "ram_limit_gb": ram_limit_gb,
        "unit_ids": unit_ids,
        "worker_assignments": worker_assignments,
        "decision": decision,
        "timestamp": datetime.now().isoformat(),
    }
    log_data["parallel_scheduling"].append(record)

    # Write back
    log_path_obj.parent.mkdir(parents=True, exist_ok=True)
    log_path_obj.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return str(log_path_obj)
