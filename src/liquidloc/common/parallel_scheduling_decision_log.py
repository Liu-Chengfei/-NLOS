"""
Parallel scheduling decision log for clause [15].

Records parallel scheduling decisions alongside decision logs,
ensuring traceability of parallel execution without changing paper-level
metrics (data/config/weights/statistics unchanged from serial execution).
"""

import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


class ParallelSchedulingDecisionLog:
    """Records parallel scheduling decisions for clause [15] compliance.

    Parallel scheduling only changes execution order, not paper-level metrics:
    - Data and weights are independently generated per unit
    - Statistics are computed per unit then merged (same as serial)
    - Training defaults to GPU (CPU only for non-training workloads)
    - Each run starts with a clean workspace
    """

    def __init__(self, log_dir: str = ".decision_logs") -> None:
        self.log_dir = log_dir
        self._units: Dict[str, Dict[str, Any]] = {}
        self._start_times: Dict[str, float] = {}
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

    def _clear_workspace(self, unit_id: str) -> None:
        """Clear temporary files from previous run for this unit."""
        for fname in os.listdir(self.log_dir):
            if fname.startswith(f"temp_{unit_id}_") or fname.startswith(f"cache_{unit_id}_"):
                fpath = os.path.join(self.log_dir, fname)
                if os.path.isfile(fpath):
                    os.remove(fpath)

    def start_unit(
        self,
        unit_id: str,
        data_seed: int,
        weight_seed: int,
        device: str = "cuda",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start tracking a parallel execution unit.

        Args:
            unit_id: Unique identifier for this execution unit
            data_seed: Seed for data generation (independent per unit)
            weight_seed: Seed for weight initialization (independent per unit)
            device: Device to use ('cuda' for training, 'cpu' for non-training)
            config: Optional configuration snapshot for this unit
        """
        if device != "cuda":
            raise RuntimeError(
                f"Clause [15] violation: training must use GPU (device='cuda'), "
                f"got device='{device}'. CPU is only for non-training workloads."
            )

        self._clear_workspace(unit_id)

        self._units[unit_id] = {
            "unit_id": unit_id,
            "status": "running",
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "data_seed": data_seed,
            "weight_seed": weight_seed,
            "device": device,
            "error": None,
            "unit_metrics": {},
            "config_snapshot": config or {},
        }
        self._start_times[unit_id] = time.time()

    def complete_unit(
        self,
        unit_id: str,
        unit_metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark a parallel execution unit as completed.

        Args:
            unit_id: Identifier of the completed unit
            unit_metrics: Metrics computed for this unit (rmse, mean_rmse, etc.)
            error: Optional error message if unit failed
        """
        if unit_id not in self._units:
            raise KeyError(f"Unit '{unit_id}' not found in scheduling log")

        end_time = time.time()
        self._units[unit_id].update({
            "status": "completed" if error is None else "failed",
            "end_time": end_time,
            "duration": end_time - self._start_times.get(unit_id, end_time),
            "error": error,
            "unit_metrics": unit_metrics or {},
        })

    def write_decision_log(self, filename: Optional[str] = None) -> str:
        """Write the scheduling decision log to a JSON file.

        Args:
            filename: Optional custom filename (default: auto-generated with timestamp)

        Returns:
            Path to the written log file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scheduling_decision_log_{timestamp}.json"

        log_path = os.path.join(self.log_dir, filename)

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "clause": "[15]",
            "principle": "Parallel scheduling only changes execution order, "
                         "not paper-level metrics (data/config/weights/statistics unchanged)",
            "units": list(self._units.values()),
            "summary": self.get_summary(),
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        return log_path

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracked units.

        Returns:
            Summary dict with unit count, completion status, and timing info
        """
        completed = sum(1 for u in self._units.values() if u["status"] == "completed")
        failed = sum(1 for u in self._units.values() if u["status"] == "failed")
        running = sum(1 for u in self._units.values() if u["status"] == "running")

        total_duration = sum(
            u["duration"] for u in self._units.values() if u["duration"] is not None
        )

        return {
            "total_units": len(self._units),
            "completed": completed,
            "failed": failed,
            "running": running,
            "total_duration_seconds": total_duration,
            "units": list(self._units.values()),
        }
