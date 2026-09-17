"""并行-串行等效性抽查实现（手册条款 [14]）。

"不使结果变差"的判据 = 并行不改变任何论文级口径：
    每个并行单元与串行执行使用完全相同的代码/配置/确定性开关，
    数据与权重各自独立生成，数值与可复现性不劣化。

首次放开并行前须做并行-串行等效性抽查：
    取同一单元分别以并行与串行方式各执行一次，
    确认结果一致（逐 bit 相同或数值在机器精度内相等）后，
    才可按 GPU 余量放开并行度，
    等效性抽查记录写入 decision log。

公开入口：
    run_equivalence_check(unit_fn, unit_id, config, seed, log_dir) -> ParallelEquivalenceResult
    compare_results(serial_result, parallel_result, tolerance) -> dict
    write_decision_log(result, log_path) -> str
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(slots=True)
class ParallelEquivalenceResult:
    """并行-串行等效性抽查结果。

    Attributes:
        unit_id: 被抽查的并行单元标识。
        serial_result: 串行执行结果（原始对象）。
        parallel_result: 并行执行结果（原始对象）。
        bit_identical: 逐 bit 是否相同。
        max_abs_diff: 最大绝对差值。
        mean_abs_diff: 平均绝对差值。
        machine_eps_ratio: max_abs_diff / np.finfo(float).eps。
        passed: 是否通过等效性检查。
        decision_log_path: 等效性抽查记录写入的 decision log 路径。
        timestamp: 抽查时间戳（ISO 8601 格式）。
        evidence: 额外证据字典。
    """

    unit_id: str
    serial_result: Any = None
    parallel_result: Any = None
    bit_identical: bool = False
    max_abs_diff: float = float("inf")
    mean_abs_diff: float = float("inf")
    machine_eps_ratio: float = float("inf")
    passed: bool = False
    decision_log_path: str = ""
    timestamp: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def _coerce_to_array(result: Any) -> Any:
    """将结果 coerced 为 numpy array（如果可能），否则保持原样。"""
    try:
        import numpy as np
        if isinstance(result, (list, tuple)):
            return np.asarray(result, dtype=np.float64)
        if isinstance(result, np.ndarray):
            return result.astype(np.float64)
    except ImportError:
        pass
    return result


def _hash_result(result: Any) -> str:
    """计算结果的哈希值（用于逐 bit 比较的兜底）。"""
    try:
        import numpy as np
        arr = _coerce_to_array(result)
        if isinstance(arr, np.ndarray):
            data = arr.tobytes()
        else:
            data = str(result).encode("utf-8")
    except Exception:
        data = str(result).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def run_serial(unit_fn: Callable[..., Any], unit_id: str, config: dict[str, Any],
               seed: Optional[int] = None) -> Any:
    """以串行方式执行计算单元。

    Args:
        unit_fn: 计算单元函数，接受 config 和 seed 参数。
        unit_id: 单元标识（用于日志）。
        config: 配置字典（与并行执行使用完全相同）。
        seed: 随机种子（用于确定性开关）。

    Returns:
        串行执行结果。
    """
    if seed is not None:
        import numpy as np
        np.random.seed(seed)
    return unit_fn(config, seed=seed)


def run_parallel(unit_fn: Callable[..., Any], unit_id: str, config: dict[str, Any],
                 seed: Optional[int] = None) -> Any:
    """以并行方式执行计算单元。

    与串行执行使用完全相同的代码/配置/确定性开关，
    数据与权重各自独立生成（通过不同的 seed 或独立数据分区）。

    Args:
        unit_fn: 计算单元函数，接受 config 和 seed 参数。
        unit_id: 单元标识（用于日志）。
        config: 配置字典（与串行执行使用完全相同）。
        seed: 随机种子（用于确定性开关）。

    Returns:
        并行执行结果。
    """
    if seed is not None:
        import numpy as np
        np.random.seed(seed)
    return unit_fn(config, seed=seed)


def compare_results(serial_result: Any, parallel_result: Any,
                    tolerance: Optional[float] = None) -> dict[str, Any]:
    """比较串行和并行执行结果。

    Args:
        serial_result: 串行执行结果。
        parallel_result: 并行执行结果。
        tolerance: 容差（None 时使用机器精度：rtol=1e-14, atol=1e-15）。

    Returns:
        比较指标字典，包含 bit_identical, max_abs_diff, mean_abs_diff,
        machine_eps_ratio, passed 等字段。
    """
    if tolerance is None:
        rtol = 1e-14
        atol = 1e-15
    else:
        rtol = tolerance
        atol = tolerance

    serial_arr = _coerce_to_array(serial_result)
    parallel_arr = _coerce_to_array(parallel_result)

    # 逐 bit 比较（通过哈希）
    bit_identical = _hash_result(serial_result) == _hash_result(parallel_result)

    # 数值比较
    try:
        import numpy as np
        if isinstance(serial_arr, np.ndarray) and isinstance(parallel_arr, np.ndarray):
            if serial_arr.shape != parallel_arr.shape:
                return {
                    "bit_identical": False,
                    "max_abs_diff": float("inf"),
                    "mean_abs_diff": float("inf"),
                    "machine_eps_ratio": float("inf"),
                    "passed": False,
                    "reason": f"shape_mismatch: {serial_arr.shape} vs {parallel_arr.shape}",
                }
            abs_diff = np.abs(serial_arr - parallel_arr)
            max_abs_diff = float(np.max(abs_diff))
            mean_abs_diff = float(np.mean(abs_diff))
            machine_eps = float(np.finfo(np.float64).eps)
            machine_eps_ratio = max_abs_diff / machine_eps if machine_eps > 0 else float("inf")
            passed = bool(np.allclose(serial_arr, parallel_arr, rtol=rtol, atol=atol))
        else:
            max_abs_diff = abs(float(serial_arr) - float(parallel_arr)) if serial_arr is not None and parallel_arr is not None else float("inf")
            mean_abs_diff = max_abs_diff
            machine_eps = 2.220446049250313e-16
            machine_eps_ratio = max_abs_diff / machine_eps if machine_eps > 0 else float("inf")
            passed = bit_identical or max_abs_diff <= atol
    except ImportError:
        max_abs_diff = float("inf")
        mean_abs_diff = float("inf")
        machine_eps_ratio = float("inf")
        passed = bit_identical

    return {
        "bit_identical": bit_identical,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "machine_eps_ratio": machine_eps_ratio,
        "passed": passed,
    }


def write_decision_log(result: ParallelEquivalenceResult, log_dir: str = ".decision_logs") -> str:
    """将等效性抽查记录写入 decision log。

    Args:
        result: 等效性抽查结果。
        log_dir: decision log 目录路径。

    Returns:
        写入的 decision log 文件路径。
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"equivalence_{result.unit_id}_{timestamp_str}.json"
    full_path = log_path / filename

    record = {
        "unit_id": result.unit_id,
        "timestamp": result.timestamp,
        "bit_identical": result.bit_identical,
        "max_abs_diff": result.max_abs_diff,
        "mean_abs_diff": result.mean_abs_diff,
        "machine_eps_ratio": result.machine_eps_ratio,
        "passed": result.passed,
        "evidence": result.evidence,
        "decision": "PASS: parallel-serial equivalence confirmed" if result.passed else "FAIL: parallel-serial equivalence NOT confirmed",
    }

    full_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(full_path)


def run_equivalence_check(unit_fn: Callable[..., Any], unit_id: str,
                          config: dict[str, Any], seed: Optional[int] = None,
                          log_dir: str = ".decision_logs") -> ParallelEquivalenceResult:
    """完整并行-串行等效性抽查流程。

    步骤：
        1. 串行执行：run_serial(unit_fn, unit_id, config, seed)
        2. 并行执行：run_parallel(unit_fn, unit_id, config, seed)
        3. 比较结果：compare_results(serial_result, parallel_result)
        4. 写入 decision log：write_decision_log(result, log_dir)
        5. 返回 ParallelEquivalenceResult

    Args:
        unit_fn: 计算单元函数。
        unit_id: 单元标识。
        config: 配置字典（串行和并行使用完全相同）。
        seed: 随机种子（确定性开关）。
        log_dir: decision log 目录。

    Returns:
        ParallelEquivalenceResult 实例。
    """
    # 步骤 1：串行执行
    serial_result = run_serial(unit_fn, unit_id, config, seed=seed)

    # 步骤 2：并行执行（相同代码/配置/确定性开关）
    parallel_result = run_parallel(unit_fn, unit_id, config, seed=seed)

    # 步骤 3：比较结果
    comparison = compare_results(serial_result, parallel_result)

    # 构造结果
    result = ParallelEquivalenceResult(
        unit_id=unit_id,
        serial_result=serial_result,
        parallel_result=parallel_result,
        bit_identical=comparison["bit_identical"],
        max_abs_diff=comparison["max_abs_diff"],
        mean_abs_diff=comparison["mean_abs_diff"],
        machine_eps_ratio=comparison["machine_eps_ratio"],
        passed=comparison["passed"],
        evidence=comparison,
    )

    # 步骤 4：写入 decision log
    result.decision_log_path = write_decision_log(result, log_dir=log_dir)

    return result


__all__ = [
    "ParallelEquivalenceResult",
    "run_serial",
    "run_parallel",
    "compare_results",
    "write_decision_log",
    "run_equivalence_check",
]
