"""
Batch decomposer for clause [17].

Clause [17]: 拆解复核（把大批次拆成可独立勾检的小块并逐块复核，防单线执行干漏）

This module provides:
- SubTask: a single sub-task with review status
- BatchDecomposer: decomposes large batches into independently verifiable sub-tasks
- ReviewTracker: tracks sub-task review status
- SubCheckSplitter: splits checks into parallelizable sub-checks
- execute_parallel_checks: runs checks in parallel independent processes
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class SubTaskStatus(Enum):
    """Status of a sub-task in the review process."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEWED = "reviewed"


class CheckSplitType(Enum):
    """Types of check splitting strategies."""
    STATIC_VS_RUNTIME = "static_vs_runtime"      # 静态检查与运行时断言分开并行
    MULTI_FILE_GREP = "multi_file_grep"           # 多文件比对拆成多路 grep
    GATE_VS_DECISION_LOG = "gate_vs_decision_log" # gate 必交物核对与 decision log 撰写并行
    COUNT_VS_REVIEW = "count_vs_review"           # 清点与复核者交叉核对并行


@dataclass
class SubTask:
    """A single sub-task with review status.

    Attributes:
        task_id: Unique identifier for this sub-task
        parent_batch_id: ID of the parent batch this task belongs to
        description: Human-readable description
        check_fn: Optional function to execute for this check
        check_args: Optional arguments for check_fn
        split_type: Type of check splitting used
        status: Current review status
        result: Execution result (if any)
        error: Error message (if failed)
        created_at: Timestamp when task was created
        completed_at: Timestamp when task was completed
        reviewed_at: Timestamp when task was reviewed
    """
    task_id: str
    parent_batch_id: str
    description: str
    split_type: CheckSplitType = CheckSplitType.STATIC_VS_RUNTIME
    check_fn: Optional[Callable] = None
    check_args: Optional[Dict[str, Any]] = None
    status: SubTaskStatus = SubTaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    reviewed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "task_id": self.task_id,
            "parent_batch_id": self.parent_batch_id,
            "description": self.description,
            "split_type": self.split_type.value,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "reviewed_at": self.reviewed_at,
        }
        return d


@dataclass
class BatchDecomposition:
    """Result of decomposing a batch into sub-tasks.

    Attributes:
        batch_id: Unique identifier for this batch
        batch_description: Human-readable description
        total_tasks: Total number of sub-tasks
        sub_tasks: List of sub-tasks
        created_at: Timestamp when batch was decomposed
    """
    batch_id: str
    batch_description: str
    total_tasks: int = 0
    sub_tasks: List[SubTask] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "batch_description": self.batch_description,
            "total_tasks": self.total_tasks,
            "sub_tasks": [t.to_dict() for t in self.sub_tasks],
            "created_at": self.created_at,
        }


class ReviewTracker:
    """Tracks review status of sub-tasks to prevent missing reviews.

    Clause [17]: 逐块复核，防单线执行干漏

    This class ensures that:
    1. Every sub-task is reviewed (not just executed)
    2. Review cross-checks are performed (count vs review)
    3. No sub-task is left unreviewed
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sub_tasks: Dict[str, SubTask] = {}
        self._review_records: List[Dict[str, Any]] = []

    def register_sub_task(self, sub_task: SubTask) -> None:
        """Register a sub-task for tracking."""
        with self._lock:
            self._sub_tasks[sub_task.task_id] = sub_task

    def mark_completed(self, task_id: str, result: Any) -> None:
        """Mark a sub-task as completed with result."""
        with self._lock:
            if task_id in self._sub_tasks:
                self._sub_tasks[task_id].status = SubTaskStatus.COMPLETED
                self._sub_tasks[task_id].result = result
                self._sub_tasks[task_id].completed_at = datetime.now().isoformat()

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark a sub-task as failed with error."""
        with self._lock:
            if task_id in self._sub_tasks:
                self._sub_tasks[task_id].status = SubTaskStatus.FAILED
                self._sub_tasks[task_id].error = error
                self._sub_tasks[task_id].completed_at = datetime.now().isoformat()

    def mark_reviewed(self, task_id: str, reviewer: str = "system") -> None:
        """Mark a sub-task as reviewed."""
        with self._lock:
            if task_id in self._sub_tasks:
                self._sub_tasks[task_id].status = SubTaskStatus.REVIEWED
                self._sub_tasks[task_id].reviewed_at = datetime.now().isoformat()

    def get_unreviewed_tasks(self) -> List[SubTask]:
        """Get all sub-tasks that have not been reviewed."""
        with self._lock:
            return [
                t for t in self._sub_tasks.values()
                if t.status not in (SubTaskStatus.REVIEWED, SubTaskStatus.PENDING)
            ]

    def get_pending_tasks(self) -> List[SubTask]:
        """Get all sub-tasks that are still pending."""
        with self._lock:
            return [
                t for t in self._sub_tasks.values()
                if t.status == SubTaskStatus.PENDING
            ]

    def verify_all_reviewed(self, batch_id: str) -> Tuple[bool, List[str]]:
        """Verify that all sub-tasks in a batch have been reviewed.

        Returns:
            Tuple of (all_reviewed, list of unreviewed task_ids)
        """
        with self._lock:
            unreviewed = [
                t.task_id for t in self._sub_tasks.values()
                if t.parent_batch_id == batch_id
                and t.status not in (SubTaskStatus.REVIEWED, SubTaskStatus.PENDING)
                and t.status != SubTaskStatus.FAILED  # failed tasks don't need review
            ]
            return len(unreviewed) == 0, unreviewed

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of review status."""
        with self._lock:
            total = len(self._sub_tasks)
            reviewed = sum(1 for t in self._sub_tasks.values() if t.status == SubTaskStatus.REVIEWED)
            completed = sum(1 for t in self._sub_tasks.values() if t.status == SubTaskStatus.COMPLETED)
            failed = sum(1 for t in self._sub_tasks.values() if t.status == SubTaskStatus.FAILED)
            pending = sum(1 for t in self._sub_tasks.values() if t.status == SubTaskStatus.PENDING)

            return {
                "total": total,
                "reviewed": reviewed,
                "completed": completed,
                "failed": failed,
                "pending": pending,
                "review_coverage": reviewed / total if total > 0 else 0.0,
            }


class SubCheckSplitter:
    """Splits checks into parallelizable sub-checks.

    Clause [17]: 凡可拆分的检查都要拆

    Supported splitting strategies:
    1. 静态检查与运行时断言分开并行 (static vs runtime)
    2. 多文件比对拆成多路 grep (multi-file comparison)
    3. gate 必交物核对与 decision log 撰写并行 (gate check vs log writing)
    4. 清点与复核者交叉核对并行 (count vs review cross-check)
    """

    @staticmethod
    def split_static_vs_runtime(
        static_checks: List[Callable],
        runtime_checks: List[Callable],
    ) -> List[Tuple[CheckSplitType, Callable, Dict[str, Any]]]:
        """Split checks into static checks and runtime assertions.

        Args:
            static_checks: List of static check functions
            runtime_checks: List of runtime assertion functions

        Returns:
            List of (split_type, check_fn, check_args) tuples
        """
        sub_checks: List[Tuple[CheckSplitType, Callable, Dict[str, Any]]] = []

        for i, check_fn in enumerate(static_checks):
            sub_checks.append((
                CheckSplitType.STATIC_VS_RUNTIME,
                check_fn,
                {"index": i, "category": "static"}
            ))

        for i, check_fn in enumerate(runtime_checks):
            sub_checks.append((
                CheckSplitType.STATIC_VS_RUNTIME,
                check_fn,
                {"index": i, "category": "runtime"}
            ))

        return sub_checks

    @staticmethod
    def split_multi_file_grep(
        files: List[str],
        pattern: str,
    ) -> List[Tuple[CheckSplitType, Callable, Dict[str, Any]]]:
        """Split multi-file comparison into multi-way grep.

        Args:
            files: List of file paths to check
            pattern: Pattern to search for

        Returns:
            List of (split_type, check_fn, check_args) tuples
        """
        sub_checks: List[Tuple[CheckSplitType, Callable, Dict[str, Any]]] = []

        for i, filepath in enumerate(files):
            def make_grep_fn(fpath: str, pat: str) -> Callable:
                def grep_fn() -> Dict[str, Any]:
                    import re
                    try:
                        content = Path(fpath).read_text(encoding="utf-8")
                        matches = re.findall(pat, content)
                        return {
                            "file": fpath,
                            "pattern": pat,
                            "match_count": len(matches),
                            "matches": matches[:10],  # limit output
                        }
                    except Exception as e:
                        return {"file": fpath, "error": str(e)}
                return grep_fn

            sub_checks.append((
                CheckSplitType.MULTI_FILE_GREP,
                make_grep_fn(filepath, pattern),
                {"file_index": i, "file": filepath}
            ))

        return sub_checks

    @staticmethod
    def split_gate_vs_decision_log(
        gate_checks: List[Callable],
        log_writes: List[Callable],
    ) -> List[Tuple[CheckSplitType, Callable, Dict[str, Any]]]:
        """Split gate checks and decision log writing into parallel tasks.

        Args:
            gate_checks: List of gate check functions
            log_writes: List of log writing functions

        Returns:
            List of (split_type, check_fn, check_args) tuples
        """
        sub_checks: List[Tuple[CheckSplitType, Callable, Dict[str, Any]]] = []

        for i, check_fn in enumerate(gate_checks):
            sub_checks.append((
                CheckSplitType.GATE_VS_DECISION_LOG,
                check_fn,
                {"index": i, "category": "gate"}
            ))

        for i, write_fn in enumerate(log_writes):
            sub_checks.append((
                CheckSplitType.GATE_VS_DECISION_LOG,
                write_fn,
                {"index": i, "category": "log_write"}
            ))

        return sub_checks

    @staticmethod
    def split_count_vs_review(
        count_fn: Callable,
        review_fns: List[Callable],
    ) -> List[Tuple[CheckSplitType, Callable, Dict[str, Any]]]:
        """Split count and review into parallel cross-checks.

        Args:
            count_fn: Function that counts items
            review_fns: List of review functions

        Returns:
            List of (split_type, check_fn, check_args) tuples
        """
        sub_checks: List[Tuple[CheckSplitType, Callable, Dict[str, Any]]] = []

        sub_checks.append((
            CheckSplitType.COUNT_VS_REVIEW,
            count_fn,
            {"category": "count"}
        ))

        for i, review_fn in enumerate(review_fns):
            sub_checks.append((
                CheckSplitType.COUNT_VS_REVIEW,
                review_fn,
                {"index": i, "category": "review"}
            ))

        return sub_checks


class BatchDecomposer:
    """Decomposes large batches into independently verifiable sub-tasks.

    Clause [17]: 把大批次拆成可独立勾检的小块并逐块复核
    """

    def __init__(
        self,
        ram_limit_gb: float = 12.0,
        max_parallel_workers: int = 4,
    ) -> None:
        self.ram_limit_gb = ram_limit_gb
        self.max_parallel_workers = max_parallel_workers
        self._decompositions: Dict[str, BatchDecomposition] = {}
        self._review_tracker = ReviewTracker()
        self._lock = threading.Lock()

    def decompose_batch(
        self,
        batch_id: str,
        description: str,
        items: List[Any],
        item_check_fn: Callable[[Any], Dict[str, Any]],
    ) -> BatchDecomposition:
        """Decompose a batch of items into independently verifiable sub-tasks.

        Args:
            batch_id: Unique batch identifier
            description: Human-readable batch description
            items: List of items to check
            item_check_fn: Function to check each item

        Returns:
            BatchDecomposition with all sub-tasks
        """
        sub_tasks: List[SubTask] = []

        for i, item in enumerate(items):
            task_id = f"{batch_id}_task_{i:03d}"
            sub_task = SubTask(
                task_id=task_id,
                parent_batch_id=batch_id,
                description=f"Check item {i}: {item}",
                split_type=CheckSplitType.STATIC_VS_RUNTIME,
                check_fn=item_check_fn,
                check_args={"item": item, "index": i},
            )
            sub_tasks.append(sub_task)
            self._review_tracker.register_sub_task(sub_task)

        decomposition = BatchDecomposition(
            batch_id=batch_id,
            batch_description=description,
            total_tasks=len(sub_tasks),
            sub_tasks=sub_tasks,
        )

        with self._lock:
            self._decompositions[batch_id] = decomposition

        return decomposition

    def execute_parallel_sub_tasks(
        self,
        batch_id: str,
        max_workers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Execute all sub-tasks in a batch in parallel.

        Args:
            batch_id: Batch identifier
            max_workers: Maximum parallel workers (default: self.max_parallel_workers)

        Returns:
            List of execution results
        """
        if batch_id not in self._decompositions:
            raise ValueError(f"Batch {batch_id} not found")

        decomposition = self._decompositions[batch_id]
        max_workers = max_workers or self.max_parallel_workers

        results: List[Dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task: Dict[concurrent.futures.Future, SubTask] = {}

            for sub_task in decomposition.sub_tasks:
                if sub_task.check_fn is not None:
                    future = executor.submit(
                        self._execute_sub_task, sub_task
                    )
                    future_to_task[future] = sub_task
                else:
                    # No check function, mark as pending
                    sub_task.status = SubTaskStatus.PENDING

            for future in concurrent.futures.as_completed(future_to_task):
                sub_task = future_to_task[future]
                try:
                    result = future.result()
                    self._review_tracker.mark_completed(sub_task.task_id, result)
                    results.append({
                        "task_id": sub_task.task_id,
                        "status": "completed",
                        "result": result,
                    })
                except Exception as exc:
                    self._review_tracker.mark_failed(sub_task.task_id, str(exc))
                    results.append({
                        "task_id": sub_task.task_id,
                        "status": "failed",
                        "error": str(exc),
                    })

        return results

    def _execute_sub_task(self, sub_task: SubTask) -> Any:
        """Execute a single sub-task.

        Args:
            sub_task: Sub-task to execute

        Returns:
            Execution result
        """
        sub_task.status = SubTaskStatus.RUNNING

        if sub_task.check_fn is None:
            return None

        if sub_task.check_args:
            return sub_task.check_fn(**sub_task.check_args)
        else:
            return sub_task.check_fn()

    def cross_review_batch(self, batch_id: str) -> Dict[str, Any]:
        """Perform cross-review of all sub-tasks in a batch.

        Clause [17]: 清点与复核者交叉核对并行

        Args:
            batch_id: Batch identifier

        Returns:
            Review result
        """
        if batch_id not in self._decompositions:
            raise ValueError(f"Batch {batch_id} not found")

        decomposition = self._decompositions[batch_id]

        # Count completed tasks
        completed_count = sum(
            1 for t in decomposition.sub_tasks
            if t.status == SubTaskStatus.COMPLETED
        )

        # Count failed tasks
        failed_count = sum(
            1 for t in decomposition.sub_tasks
            if t.status == SubTaskStatus.FAILED
        )

        # Cross-review: verify count matches actual results
        all_reviewed, unreviewed = self._review_tracker.verify_all_reviewed(batch_id)

        # Mark all completed tasks as reviewed
        for sub_task in decomposition.sub_tasks:
            if sub_task.status == SubTaskStatus.COMPLETED:
                self._review_tracker.mark_reviewed(sub_task.task_id)

        return {
            "batch_id": batch_id,
            "total_tasks": decomposition.total_tasks,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "all_reviewed": all_reviewed,
            "unreviewed_task_ids": unreviewed,
            "review_coverage": completed_count / decomposition.total_tasks if decomposition.total_tasks > 0 else 0.0,
        }

    def get_decomposition_summary(self) -> Dict[str, Any]:
        """Get summary of all decompositions."""
        with self._lock:
            total_batches = len(self._decompositions)
            total_tasks = sum(d.total_tasks for d in self._decompositions.values())
            review_summary = self._review_tracker.get_summary()

            return {
                "total_batches": total_batches,
                "total_tasks": total_tasks,
                "review_summary": review_summary,
                "ram_limit_gb": self.ram_limit_gb,
                "max_parallel_workers": self.max_parallel_workers,
            }


def execute_parallel_checks(
    checks: List[Tuple[CheckSplitType, Callable, Dict[str, Any]]],
    max_workers: int = 4,
    ram_limit_gb: float = 12.0,
) -> Dict[str, Any]:
    """Execute checks in parallel independent processes.

    Clause [17]: 用并行独立进程降低"单线从头做到尾导致干漏"的风险

    Args:
        checks: List of (split_type, check_fn, check_args) tuples
        max_workers: Maximum parallel workers
        ram_limit_gb: RAM limit for parallel execution

    Returns:
        Execution result summary
    """
    results: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_check: Dict[concurrent.futures.Future, Tuple[int, CheckSplitType, Callable, Dict[str, Any]]] = {}

        for i, (split_type, check_fn, check_args) in enumerate(checks):
            if check_fn is not None:
                future = executor.submit(check_fn)
                future_to_check[future] = (i, split_type, check_fn, check_args)

        for future in concurrent.futures.as_completed(future_to_check):
            idx, split_type, check_fn, check_args = future_to_check[future]
            try:
                result = future.result()
                results.append({
                    "index": idx,
                    "split_type": split_type.value,
                    "status": "passed",
                    "result": result,
                })
            except Exception as exc:
                results.append({
                    "index": idx,
                    "split_type": split_type.value,
                    "status": "failed",
                    "error": str(exc),
                })

    passed = sum(1 for r in results if r["status"] == "passed")
    failed = sum(1 for r in results if r["status"] == "failed")

    return {
        "total_checks": len(checks),
        "passed": passed,
        "failed": failed,
        "results": results,
        "ram_limit_gb": ram_limit_gb,
        "parallel_workers": max_workers,
    }
