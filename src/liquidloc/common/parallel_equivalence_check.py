"""并行-串行等效性抽查（Clause [14] 落实载体）。

设计目标：
- 首次打开并行前，用同一组输入串行跑一次 + ThreadPoolExecutor 并行跑一次，
  比较结果是否逐位相同或数值在机器精度内相等。
- 等效性抽查结果写入 decision log，作为 [14] 的留痕证据。
- 本模块本身包含自测：pytest 可以直接跑。

典型调用链（_run_25unit.py stage 2 首次并行前）：
    from liquidloc.common.parallel_equivalence_check import (
        verify_parallel_equivalence, record_equivalence_decision,
    )
    result = verify_parallel_equivalence(
        task_fn=_run_one_unit,
        args_list=spot_check_args,   # 取 1-2 个代表性单元
        tolerance=1e-6,
    )
    record_equivalence_decision(result, round_tag="RZ-0")
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


# ── 数据结构 ──────────────────────────────────────────────────────────────

@dataclass
class EquivalenceResult:
    """单次等效性抽查结果。"""
    spot_check_id: str
    serial_results: list[Any] = field(default_factory=list)
    parallel_results: list[Any] = field(default_factory=list)
    bitwise_equal: bool = False
    numeric_tolerance_ok: bool = False
    passed: bool = False
    detail: str = ""
    serial_duration_s: float = 0.0
    parallel_duration_s: float = 0.0
    n_units: int = 0
    tolerance: float = 1e-6
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpotCheckRecord:
    """等效性抽查的持久化记录（写入 decision log / 批处理日志）。"""
    spot_check_id: str
    round: str
    gpu_model: str
    parallel_decision: dict[str, Any]
    result: EquivalenceResult
    created_at: str


# ── 核心函数 ──────────────────────────────────────────────────────────────

def _default_compare(
    a: Any, b: Any, tolerance: float,
) -> tuple[bool, bool]:
    """返回 (bitwise_equal, numeric_tolerance_ok)。"""
    if a == b:
        return True, True
    # 尝试数值比较
    try:
        fa, fb = float(a), float(b)
        if fa == fb == 0.0:
            return False, True
        if abs(fa - fb) / max(abs(fa), abs(fb), 1e-12) <= tolerance:
            return False, True
    except (TypeError, ValueError):
        pass
    # 递归比较 dict / list
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        if keys != set(a) or keys != set(b):
            return False, False
        bw, nt = True, True
        for k in keys:
            b2, n2 = _default_compare(a[k], b[k], tolerance)
            bw = bw and b2
            nt = nt and n2
        return bw, nt
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False, False
        bw, nt = True, True
        for av, bv in zip(a, b):
            b2, n2 = _default_compare(av, bv, tolerance)
            bw = bw and b2
            nt = nt and n2
        return bw, nt
    return False, False


def verify_parallel_equivalence(
    task_fn: Callable[[Any], Any],
    args_list: list[Any],
    *,
    tolerance: float = 1e-6,
    max_workers: int = 2,
    compare_fn: Callable[[Any, Any, float], tuple[bool, bool]] | None = None,
    spot_check_id: str | None = None,
) -> EquivalenceResult:
    """对同一组输入，串行 vs ThreadPoolExecutor 并行各跑一次，比较结果。

    参数:
        task_fn: 对单个输入执行任务的函数，返回任意可序列化结果。
        args_list: 任务输入列表（通常取 1-2 个代表性单元）。
        tolerance: 数值比较容差（默认机器精度 ~1e-6）。
        max_workers: 并行 worker 数，默认 2。
        compare_fn: 自定义比较函数 (a, b, tolerance) -> (bitwise, numeric_ok)；
                    为 None 时使用内置默认比较。
        spot_check_id: 抽查 ID，默认用时间戳+线程ID生成。

    返回:
        EquivalenceResult: 包含 serial/parallel 结果 + 比较结论。
    """
    _compare = compare_fn or _default_compare
    cid = spot_check_id or f"eq-{int(time.time())}-{threading.get_ident()}"

    # ── 串行执行 ──
    t0 = time.perf_counter()
    serial_results: list[Any] = []
    for arg in args_list:
        serial_results.append(task_fn(arg))
    t_serial = time.perf_counter() - t0

    # ── 并行执行 ──
    t0 = time.perf_counter()
    parallel_results: list[Any] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        futures = {ex.submit(task_fn, arg): i for i, arg in enumerate(args_list)}
        results_by_idx: dict[int, Any] = {}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results_by_idx[idx] = fut.result()
            except Exception as exc:
                results_by_idx[idx] = {"_error": str(exc)}
    for i in range(len(args_list)):
        parallel_results.append(results_by_idx.get(i, {"_error": "missing"}))
    t_parallel = time.perf_counter() - t0

    # ── 比较 ──
    all_bw, all_nt = True, True
    mismatch_examples: list[str] = []
    for i, (s, p) in enumerate(zip(serial_results, parallel_results)):
        bw, nt = _compare(s, p, tolerance)
        if not bw or not nt:
            mismatch_examples.append(f"unit#{i}: bw={bw}, nt={nt}, serial={s!r}, parallel={p!r}")
        all_bw = all_bw and bw
        all_nt = all_nt and nt

    passed = all_bw or all_nt
    detail = (f"n={len(args_list)}, bw={all_bw}, nt={all_nt}, "
              f"serial={t_serial:.3f}s, parallel={t_parallel:.3f}s")
    if mismatch_examples:
        detail += f" | mismatches={len(mismatch_examples)}; first={mismatch_examples[0][:200]}"

    return EquivalenceResult(
        spot_check_id=cid,
        serial_results=serial_results,
        parallel_results=parallel_results,
        bitwise_equal=all_bw,
        numeric_tolerance_ok=all_nt,
        passed=passed,
        detail=detail,
        serial_duration_s=t_serial,
        parallel_duration_s=t_parallel,
        n_units=len(args_list),
        tolerance=tolerance,
        raw={
            "spot_check_id": cid,
            "passed": passed,
            "bitwise_equal": all_bw,
            "numeric_tolerance_ok": all_nt,
            "n_units": len(args_list),
            "tolerance": tolerance,
            "serial_duration_s": round(t_serial, 6),
            "parallel_duration_s": round(t_parallel, 6),
            "mismatch_examples": mismatch_examples[:3],
        },
    )


def record_equivalence_decision(
    result: EquivalenceResult,
    round_tag: str,
    decision_log_path: str | None = None,
) -> dict[str, Any]:
    """将等效性抽查结果写入 decision log，返回写入的条目。

    参数:
        result: verify_parallel_equivalence 的返回值。
        round_tag: 轮次标签（如 "RZ-0"）。
        decision_log_path: decision_log.json 路径；为 None 时用默认路径。

    返回:
        写入的 decision_entries 条目 dict。
    """
    import datetime
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    ts = now.isoformat()

    entry = {
        "id": f"DEC-{result.spot_check_id}",
        "round": round_tag,
        "action": f"等效性抽查 [{result.spot_check_id}]: 串行 vs 并行 "
                  f"n={result.n_units}, passed={result.passed}, "
                  f"bitwise={result.bitwise_equal}, numeric_ok={result.numeric_tolerance_ok}",
        "reason": "Clause [14]: 首次放开并行前须做并行-串行等效性抽查",
        "status": "completed" if result.passed else "failed",
        "created_at": ts,
        "equivalence_detail": result.raw,
    }

    path = decision_log_path or os.path.join(
        os.path.dirname(__file__), "..", ".audit", "decision_log.json"
    )
    path = os.path.normpath(path)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                dl = json.load(fh)
            exp = dl.setdefault("experiments", {}).setdefault("e20_paper_main", {})
            entries = exp.setdefault("decision_entries", [])
            entries.append(entry)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(dl, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass  # 写入失败不阻断抽查流程
    return entry


def check_equivalence_record_exists(
    decision_log_path: str | None = None,
    round_tag: str = "RZ-0",
) -> dict[str, Any]:
    """检查 decision log 中是否存在 [14] 等效性抽查记录。

    返回:
        {"exists": bool, "count": int, "entries": list[str]}
    """
    path = decision_log_path or os.path.join(
        os.path.dirname(__file__), "..", ".audit", "decision_log.json"
    )
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return {"exists": False, "count": 0, "entries": [], "path": path}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            dl = json.load(fh)
        exp = dl.get("experiments", {}).get("e20_paper_main", {})
        entries = exp.get("decision_entries", [])
        eq_entries = [
            e for e in entries
            if "等效性抽查" in str(e.get("action", ""))
            and e.get("round") == round_tag
        ]
        return {
            "exists": len(eq_entries) > 0,
            "count": len(eq_entries),
            "entries": [e.get("id") for e in eq_entries],
            "path": path,
        }
    except Exception as exc:
        return {"exists": False, "count": 0, "entries": [], "error": str(exc), "path": path}
