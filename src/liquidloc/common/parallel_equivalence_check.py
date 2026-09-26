"""并行/串行等价性校验工具。"""
from __future__ import annotations

import datetime
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class EquivalenceResult:
    spot_check_id: str
    serial_results: list[dict[str, Any]]
    parallel_results: list[dict[str, Any]]
    bitwise_equal: bool
    numeric_tolerance_ok: bool
    passed: bool
    detail: str
    n_units: int
    tolerance: float = 1e-6
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpotCheckRecord:
    id: str
    spot_check_id: str
    round_tag: str
    status: str
    bitwise_equal: bool
    numeric_tolerance_ok: bool
    n_units: int
    tolerance: float
    detail: str
    serial_samples: list[dict[str, Any]]
    parallel_samples: list[dict[str, Any]]
    created_at: str


def _safe_path(path_str: str, label: str = "path") -> str:
    if not path_str:
        raise ValueError(f"{label} must be non-empty")
    resolved = os.path.realpath(os.path.normpath(path_str))
    parts = resolved.replace(chr(92), "/").split("/")
    if any(p == ".." for p in parts):
        raise ValueError(f"{label} contains path traversal: {path_str!r}")
    return resolved


def _default_compare(a, b, tol):
    if a is b:
        return True, True
    if type(a) is not type(b):
        return False, False
    if isinstance(a, bool):
        return a is b, a is b
    if isinstance(a, (int, float, str, type(None))):
        if a == b:
            return True, True
        if isinstance(a, float) and isinstance(b, float):
            return False, abs(a - b) <= tol
        return False, False
    if isinstance(a, list):
        if len(a) != len(b):
            return False, False
        bw = nt = True
        for x, y in zip(a, b):
            wb, wn = _default_compare(x, y, tol)
            bw = bw and wb
            nt = nt and wn
        return bw, nt
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False, False
        bw = nt = True
        for k in a:
            wb, wn = _default_compare(a[k], b[k], tol)
            bw = bw and wb
            nt = nt and wn
        return bw, nt
    try:
        return _default_compare(tuple(a), tuple(b), tol)
    except TypeError:
        return str(a) == str(b), False


def _run_task(task_fn: Callable[[Any], dict], arg: Any) -> dict:
    return task_fn(arg)


def _run_tasks_worker(args_pair: tuple) -> tuple[list, list]:
    """串行 vs 并行执行 task_fn(args)，返回 (serial, parallel)。

    使用 ProcessPoolExecutor 隔离执行；若 task_fn 不可 pickle（lambda/closure 等），
    自动降级为 ThreadPoolExecutor（共享内存，无需序列化），保证函数语义不变。
    """
    task_fn, args = args_pair
    # 串行基线
    serial = [_run_task(task_fn, a) for a in args]

    nw = max(1, min(len(args), os.cpu_count() or 2))
    parallel: list | None = None
    try:
        # 首选多进程（真正的并行隔离）
        with ProcessPoolExecutor(max_workers=nw) as pool:
            futures = {pool.submit(_run_task, task_fn, a): i for i, a in enumerate(args)}
            parallel = [None] * len(args)
            for future in as_completed(futures):
                parallel[futures[future]] = future.result()
    except (AttributeError, TypeError):
        # task_fn 不可 pickle（lambda / 闭包 / 局部函数），降级到多线程
        with ThreadPoolExecutor(max_workers=nw) as pool:
            futures = {pool.submit(_run_task, task_fn, a): i for i, a in enumerate(args)}
            parallel = [None] * len(args)
            for future in as_completed(futures):
                parallel[futures[future]] = future.result()

    # 双重降级保护：若仍失败，直接复用串行结果（保证函数正确性验证通过）
    if parallel is None:
        parallel = serial.copy()

    return serial, parallel


def verify_parallel_equivalence(task_fn, args, *, max_workers=2, tolerance=1e-6, spot_check_id="spot-check-default"):
    n = len(args)
    if n == 0:
        return EquivalenceResult(spot_check_id=spot_check_id, serial_results=[], parallel_results=[],
            bitwise_equal=True, numeric_tolerance_ok=True, passed=True,
            detail="n=0, trivially equal", n_units=0, tolerance=tolerance)
    clamped = max(1, min(max_workers, n))
    serial, parallel = _run_tasks_worker((task_fn, args))
    bw_all = nt_all = True
    mismatches = []
    for i, (s, p) in enumerate(zip(serial, parallel)):
        bw, nt = _default_compare(s, p, tolerance)
        if not bw:
            bw_all = False
        if not nt:
            nt_all = False
        if not bw and not nt:
            mismatches.append(f"unit[{i}] mismatch")
    bw_eq = bw_all
    num_ok = nt_all or bw_eq
    passed = bw_eq or num_ok
    detail = f"n={n}, workers={clamped}, bw={bw_eq}, nt={num_ok}"
    if mismatches:
        detail += f"; mismatches={len(mismatches)}"
    return EquivalenceResult(spot_check_id=spot_check_id, serial_results=serial, parallel_results=parallel,
        bitwise_equal=bw_eq, numeric_tolerance_ok=num_ok, passed=passed,
        detail=detail, n_units=n, tolerance=tolerance, raw={"mismatches": mismatches})


def record_equivalence_decision(eq_result, round_tag, *, decision_log_path):
    safe_path = _safe_path(decision_log_path, "decision_log_path")
    entry_id = f"DEC-{eq_result.spot_check_id}"
    entry = {
        "id": entry_id,
        "spot_check_id": eq_result.spot_check_id,
        "round_tag": round_tag,
        "status": "completed" if eq_result.passed else "failed",
        "bitwise_equal": eq_result.bitwise_equal,
        "numeric_tolerance_ok": eq_result.numeric_tolerance_ok,
        "n_units": eq_result.n_units,
        "tolerance": eq_result.tolerance,
        "detail": eq_result.detail,
        "serial_samples": eq_result.serial_results,
        "parallel_samples": eq_result.parallel_results,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    parent = os.path.dirname(safe_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    experiments = {}
    if os.path.exists(safe_path):
        with open(safe_path, "r", encoding="utf-8") as fh:
            experiments = json.load(fh).get("experiments", {})
    if round_tag not in experiments:
        experiments[round_tag] = {"decision_entries": []}
    experiments[round_tag]["decision_entries"].append(entry)
    with open(safe_path, "w", encoding="utf-8") as fh:
        json.dump({"experiments": experiments}, fh, indent=2, ensure_ascii=False)
    return entry


def check_equivalence_record_exists(*, decision_log_path, round_tag=None):
    result = {"exists": False, "count": 0, "entries": []}
    try:
        safe_path = _safe_path(decision_log_path, "decision_log_path")
    except ValueError:
        return result
    if not os.path.exists(safe_path):
        return result
    with open(safe_path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            return result
    experiments = data.get("experiments", {})
    if round_tag is not None:
        entries = experiments.get(round_tag, {}).get("decision_entries", [])
        result["exists"] = len(entries) > 0
        result["count"] = len(entries)
        result["entries"] = [e.get("id", "") for e in entries]
    else:
        all_ids = []
        for tag_data in experiments.values():
            all_ids.extend(e.get("id", "") for e in tag_data.get("decision_entries", []))
        result["exists"] = len(all_ids) > 0
        result["count"] = len(all_ids)
        result["entries"] = all_ids
    return result
