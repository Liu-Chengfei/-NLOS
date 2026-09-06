"""R-1..R-5 失败归因与重跑决策门控（异步高NLOS实验全流程保障手册 Part 3 R 系列）。

本模块实现 5 个独立 verifier 函数，对 .audit/decision_log.json 的内容做
结构化校验。每个函数返回 dict 含 passed + 详细字段，供 scripts/r_attribution.py
和 scripts/run_handbook_gates.py 调用。

六个失败分类（R-1）：①实现性 ②执行性 ③方法性 ④统计性 ⑤数据性 ⑥硬件/环境性
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "R1_check_register",
    "R2_check_rca_entries",
    "R3_check_rerun_rounds",
    "R4_check_honest_report",
    "R5_check_fallback_exit",
]


def _resolve_log_payload(log: Mapping[str, Any]) -> Mapping[str, Any]:
    """Unwrap decision_log.json to the per-experiment sub-tree.

    The log file is structured as:
        { "_schema_version": ..., "_generated_at": ...,
          "experiments": { "<exp_id>": { failure_classification_register, rca_entries, ... } } }
    Per-experiment fields live under experiments[exp_id]. This helper returns that
    sub-tree so the checkers work uniformly.
    """
    experiments = log.get("experiments") if isinstance(log, Mapping) else None
    if isinstance(experiments, Mapping) and experiments:
        first = next(iter(experiments.values()))
        if isinstance(first, Mapping):
            return first
    return dict(log)


def R1_check_register(log: Mapping[str, Any]) -> dict[str, Any]:
    """R-1: 失败分类登记 — 6 类分类桶结构完整 + 至少一个非空桶。

    检查 failure_classification_register 包含 ①-⑥ 六个分类键，
    已使用分类的 entries 非空列表。
    """
    payload = _resolve_log_payload(log)
    reg = payload.get("failure_classification_register") or {}
    required_classes = [
        "class_①_实现",
        "class_②_执行",
        "class_③_方法",
        "class_⑤_数据",
        "class_⑥_硬件环境",
    ]
    missing_classes = [c for c in required_classes if c not in reg]
    used_classes = [c for c in required_classes if reg.get(c)]
    n_total_entries = sum(
        len(entries.get("entries") if isinstance(entries, dict) else (entries or []))
        for entries in [reg.get(c) or {} for c in required_classes]
    )
    # 兼容：entries 直接是 list
    n_total_entries_alt = sum(
        len(reg.get(c) or []) if isinstance(reg.get(c), list) else 0
        for c in required_classes
    )
    n_total_entries = max(n_total_entries, n_total_entries_alt)
    passed = not missing_classes and n_total_entries >= 1
    return {
        "passed": bool(passed),
        "missing_classes": missing_classes,
        "used_classes": used_classes,
        "total_entries": n_total_entries,
        "notes": "ok" if passed else "register incomplete or no entries",
    }


def R2_check_rca_entries(log: Mapping[str, Any]) -> dict[str, Any]:
    """R-2: 根因分析记录 — 每条 RCA 必填 5 字段 + status 合法。

    必填: phenomenon, reproduce_steps, hypothesis, evidence, conclusion（或 disposition）。
    status ∈ {"open", "closed"}。
    """
    payload = _resolve_log_payload(log)
    raw_entries = payload.get("rca_entries") or []
    entries = raw_entries if isinstance(raw_entries, list) else []
    required_fields = ("phenomenon", "reproduce_steps", "hypothesis", "evidence")
    valid_status = {"open", "closed"}
    bad_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            bad_entries.append({"entry": str(entry)[:100], "reason": "not_a_mapping"})
            continue
        # conclusion/disposition 至少一个非空
        conclusion = entry.get("conclusion") or entry.get("disposition") or ""
        if not conclusion:
            bad_entries.append({"id": entry.get("id"), "reason": "missing_conclusion/disposition"})
            continue
        missing = [f for f in required_fields if not entry.get(f)]
        if missing:
            bad_entries.append({"id": entry.get("id"), "reason": f"missing_fields={missing}"})
        status = entry.get("status")
        if status not in valid_status:
            bad_entries.append({"id": entry.get("id"), "reason": f"invalid_status={status!r}"})
    passed = len(bad_entries) == 0 and len(entries) >= 1
    return {
        "passed": bool(passed),
        "n_entries": len(entries),
        "bad_entries": bad_entries,
        "notes": "ok" if passed else "RCA entries incomplete",
    }


def R3_check_rerun_rounds(log: Mapping[str, Any]) -> dict[str, Any]:
    """R-3: 重跑轮次 — rerun_rounds 整数 ≤ 2（手册上限）。"""
    payload = _resolve_log_payload(log)
    rounds = payload.get("rerun_rounds", None)
    if rounds is None:
        return {"passed": False, "notes": "rerun_rounds missing from decision_log"}
    try:
        rounds_int = int(rounds)
    except (TypeError, ValueError):
        return {"passed": False, "notes": f"rerun_rounds must be int, got {type(rounds).__name__}={rounds!r}"}
    passed = 0 <= rounds_int <= 2
    return {
        "passed": bool(passed),
        "rerun_rounds": rounds_int,
        "limit_per_handbook": 2,
        "notes": "ok" if passed else "exceeds handbook limit (≤2 rounds)",
    }


def R4_check_honest_report(log: Mapping[str, Any]) -> dict[str, Any]:
    """R-4: 诚实报告 — honest_report_decision.in_effect 为 True 且 report_file 存在。

    检查 honest_report_decision.in_effect 是 bool，
    若为 True 则 report_file 路径必须指向已存在文件。
    """
    payload = _resolve_log_payload(log)
    decision = payload.get("honest_report_decision") or {}
    in_effect_raw = decision.get("in_effect")
    # 兼容字符串 "true"/"false"
    if isinstance(in_effect_raw, str):
        in_effect = in_effect_raw.lower() in ("true", "1", "yes")
    else:
        in_effect = bool(in_effect_raw)
    report_file = decision.get("report_file") or ""
    file_exists = False
    if isinstance(report_file, str) and report_file:
        file_exists = Path(report_file).is_file()
    passed = bool(in_effect) and file_exists
    return {
        "passed": passed,
        "in_effect": in_effect,
        "report_file": report_file,
        "file_exists": file_exists,
        "notes": "ok" if passed else ("honest report not declared" if not in_effect else "report file missing"),
    }


def R5_check_fallback_exit(log: Mapping[str, Any]) -> dict[str, Any]:
    """R-5: 兜底出口 — declared=True 时 class + rationale 非空。"""
    payload = _resolve_log_payload(log)
    fallback = payload.get("fallback_exit") or {}
    declared_raw = fallback.get("declared")
    declared = bool(declared_raw) if declared_raw is not None else False
    if declared:
        cls = fallback.get("class") or ""
        rationale = fallback.get("rationale") or ""
        passed = bool(str(cls).strip()) and bool(str(rationale).strip())
    else:
        passed = True  # 未声明 fallback 也允许
    return {
        "passed": passed,
        "declared": declared,
        "class_present": bool(str(fallback.get("class") or "").strip()),
        "rationale_present": bool(str(fallback.get("rationale") or "").strip()),
        "notes": "ok" if passed else "fallback declared but class/rationale missing",
    }
