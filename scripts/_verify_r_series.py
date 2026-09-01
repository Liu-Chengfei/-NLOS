"""R-series schema verifier (异步高NLOS实验全流程保障手册 Part 3 R-1..R-5).

Asserts `.audit/decision_log.json` satisfies Pre-2 / R-1..R-5 schema
completeness:

    R-1: failure_classification_register has at least one entry per class USED
        (not all 6 required, but used ones must be present)
    R-2: rca_entries all have required fields
        (phenomenon, reproduce_steps, hypothesis, evidence, conclusion, status)
    R-3: rerun_rounds is integer <= 2 (per handbook "同一失败最多重跑 2 轮")
    R-4: honest_report_decision.in_effect is boolean
    R-5: fallback_exit has class + rationale when declared=true

Output: prints R-1..R-5 status, returns 0 if all PASS else non-zero.

Usage:
    python scripts/_verify_r_series.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISION_LOG_PATH = ROOT / ".audit" / "decision_log.json"

CURRENT_EXPERIMENT_ID = "async_high_nlos_4combo"

RCA_REQUIRED_FIELDS = (
    "phenomenon",
    "reproduce_steps",
    "hypothesis",
    "evidence",
    "conclusion",
    "status",
)


def _load_log() -> dict | None:
    if not DECISION_LOG_PATH.exists():
        print(f"[verify_r] FAIL: decision_log.json not found at {DECISION_LOG_PATH}")
        return None
    try:
        return json.loads(DECISION_LOG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[verify_r] FAIL: decision_log.json is malformed: {e}")
        return None


def _experiment(log: dict) -> dict | None:
    experiments = log.get("experiments", {})
    if CURRENT_EXPERIMENT_ID not in experiments:
        print(f"[verify_r] FAIL: experiment_id {CURRENT_EXPERIMENT_ID!r} not in experiments")
        return None
    return experiments[CURRENT_EXPERIMENT_ID]


# ============================================================================
# Per-R checks (each returns (PASS, evidence_str))
# ============================================================================
def check_r1(exp: dict) -> tuple[bool, str]:
    """R-1: failure_classification_register structure — used classes present."""
    register = exp.get("failure_classification_register")
    if not isinstance(register, dict):
        return False, "failure_classification_register missing or not a dict"
    used_classes = []  # classes with at least one entry
    total_entries = 0
    for key, bucket in register.items():
        if not isinstance(bucket, dict):
            return False, f"bucket {key!r} is not a dict"
        entries = bucket.get("entries", [])
        if not isinstance(entries, list):
            return False, f"bucket {key!r}.entries is not a list"
        if len(entries) > 0:
            used_classes.append(key)
            total_entries += len(entries)
    # Validate each used bucket's entries have minimum fields
    for key in used_classes:
        for i, e in enumerate(register[key]["entries"]):
            if not isinstance(e, dict):
                return False, f"{key}.entries[{i}] not a dict"
            if "phenomenon" not in e or not str(e.get("phenomenon", "")).strip():
                return False, f"{key}.entries[{i}] missing non-empty 'phenomenon'"
    evidence = (
        f"register has {len(register)} class buckets, "
        f"{len(used_classes)} used ({used_classes}), "
        f"{total_entries} total entries"
    )
    return True, evidence


def check_r2(exp: dict) -> tuple[bool, str]:
    """R-2: rca_entries have all required fields."""
    rcas = exp.get("rca_entries")
    if not isinstance(rcas, list):
        return False, "rca_entries missing or not a list"
    if len(rcas) == 0:
        return False, "rca_entries is empty (R-2 requires at least one RCA record)"
    bad = []
    for i, r in enumerate(rcas):
        if not isinstance(r, dict):
            bad.append(f"[{i}] not a dict")
            continue
        rid = r.get("id", f"[{i}]")
        missing = [f for f in RCA_REQUIRED_FIELDS if f not in r]
        if missing:
            bad.append(f"{rid}: missing fields {missing}")
            continue
        # Empty string check (phenomenon/hypothesis/evidence/conclusion)
        for f in ("phenomenon", "hypothesis", "evidence", "conclusion"):
            if not str(r.get(f, "")).strip():
                bad.append(f"{rid}.{f} is empty")
        if r.get("status") not in ("open", "closed"):
            bad.append(f"{rid}.status not in (open, closed): {r.get('status')!r}")
    if bad:
        return False, f"{len(bad)} RCA record(s) invalid: " + "; ".join(bad[:3])
    return True, f"{len(rcas)} RCA record(s) all valid"


def check_r3(exp: dict) -> tuple[bool, str]:
    """R-3: rerun_rounds is integer ≤ 2."""
    rr = exp.get("rerun_rounds")
    if not isinstance(rr, int) or isinstance(rr, bool):
        return False, f"rerun_rounds must be integer, got {type(rr).__name__}: {rr!r}"
    if rr < 0:
        return False, f"rerun_rounds must be ≥ 0, got {rr}"
    if rr > 2:
        return False, (
            f"rerun_rounds={rr} exceeds R-3 cap of 2 "
            f"(handbook '同一失败最多重跑 2 轮'); further reruns require R-5 fallback"
        )
    return True, f"rerun_rounds = {rr} (≤ 2 per R-3)"


def check_r4(exp: dict) -> tuple[bool, str]:
    """R-4: honest_report_decision.in_effect is boolean."""
    h = exp.get("honest_report_decision")
    if not isinstance(h, dict):
        return False, "honest_report_decision missing or not a dict"
    in_effect = h.get("in_effect")
    if not isinstance(in_effect, bool):
        return False, f"honest_report_decision.in_effect must be boolean, got {type(in_effect).__name__}"
    return True, f"in_effect = {in_effect}"


def check_r5(exp: dict) -> tuple[bool, str]:
    """R-5: fallback_exit has class + rationale when declared=true."""
    f = exp.get("fallback_exit")
    if not isinstance(f, dict):
        return False, "fallback_exit missing or not a dict"
    declared = f.get("declared", False)
    if not isinstance(declared, bool):
        return False, f"fallback_exit.declared must be boolean, got {type(declared).__name__}"
    if not declared:
        return True, "fallback_exit.declared = false (R-5 not active)"
    # declared=true → require class + rationale
    cls = f.get("class")
    rationale = f.get("rationale")
    issues = []
    if not cls or not str(cls).strip():
        issues.append("class missing/empty")
    if not rationale or not str(rationale).strip():
        issues.append("rationale missing/empty")
    if issues:
        return False, f"fallback_exit.declared=true but: {issues}"
    return True, (
        f"declared=true, class={cls!r}, "
        f"rationale={str(rationale)[:60]}{'...' if len(str(rationale)) > 60 else ''}"
    )


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    print("=" * 70)
    print("R-series Decision Log Schema Verifier (Part 3 R-1..R-5)")
    print("=" * 70)
    print(f"  log_path       = {DECISION_LOG_PATH}")
    print(f"  experiment_id  = {CURRENT_EXPERIMENT_ID}")

    log = _load_log()
    if log is None:
        return 1
    exp = _experiment(log)
    if exp is None:
        return 1

    checks = [
        ("R-1", check_r1),
        ("R-2", check_r2),
        ("R-3", check_r3),
        ("R-4", check_r4),
        ("R-5", check_r5),
    ]
    results = []
    print()
    all_pass = True
    for name, fn in checks:
        try:
            ok, evidence = fn(exp)
        except Exception as e:
            ok = False
            evidence = f"check raised {type(e).__name__}: {e}"
        results.append((name, ok, evidence))
        marker = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{marker}] {name}: {evidence}")

    print()
    print("=" * 70)
    summary = "PASS" if all_pass else "FAIL"
    print(f"R-series verifier: {summary}")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())