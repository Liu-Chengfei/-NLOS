"""R-series attribution CLI (异步高NLOS实验全流程保障手册 Part 3 R-1..R-5).

Manages `.audit/decision_log.json` per the Pre-2 decision log requirement
(Part 2 §实验准备 Pre-2 + Part 3 §失败归因与重跑决策 R-1..R-5):

    R-1: failure classification register (6 classes)
    R-2: RCA entries (phenomenon/reproduce_steps/hypothesis/evidence/conclusion/status)
    R-3: rerun_rounds (integer counter, ≤ 2 per handbook "同一失败最多重跑 2 轮")
    R-4: honest_report_decision (boolean in_effect)
    R-5: fallback_exit (declared/class/rationale)

Usage:
    python scripts/r_attribution.py status
    python scripts/r_attribution.py classify --class ③ --phenomenon "..." --evidence "..."
    python scripts/r_attribution.py rca --id RCA-004 --hypothesis "..." --evidence "..."
    python scripts/r_attribution.py rerun --note "..."
    python scripts/r_attribution.py fallback --class 方法性失败 --rationale "..."

The script:
- Loads `.audit/decision_log.json` if it exists, else starts a new template.
- Validates inputs (class is one of ①②③④⑤⑥, rationale non-empty, etc.).
- Writes atomically: write to `.tmp` then rename.
- Updates header git_commit on each save via `git rev-parse HEAD`.
- Adds ISO8601 timestamp to every entry.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DECISION_LOG_PATH = ROOT / ".audit" / "decision_log.json"

VALID_CLASSES = {"①", "②", "③", "④", "⑤", "⑥"}
CLASS_LABELS = {
    "①": "①实现性失败",
    "②": "②执行性失败",
    "③": "③方法性失败",
    "④": "④统计性失败",
    "⑤": "⑤数据性失败",
    "⑥": "⑥硬件/环境性失败",
}

CURRENT_EXPERIMENT_ID = "async_high_nlos_4combo"


# ============================================================================
# Atomic I/O
# ============================================================================
def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _get_git_commit() -> str:
    return _run_git(["rev-parse", "HEAD"])


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write payload to path atomically: write to .tmp, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


# ============================================================================
# Decision log template
# ============================================================================
def _new_template() -> dict[str, Any]:
    """Return a fresh decision_log.json template skeleton."""
    now = _now_iso()
    return {
        "_schema_version": "1.0",
        "_generated_at": now,
        "_git_commit": _get_git_commit(),
        "_frozen_at": now,
        "experiments": {
            CURRENT_EXPERIMENT_ID: {
                "pre_registered_sesoi": {
                    "relative_improvement_min": 0.33,
                    "equivalence_bound_ratio": 0.5,
                    "equivalence_bound_pct": 0.165,
                    "frozen": False,
                    "frozen_at": None,
                },
                "failure_classification_register": {
                    "class_①_实现": {"label": CLASS_LABELS["①"], "entries": []},
                    "class_②_执行": {"label": CLASS_LABELS["②"], "entries": []},
                    "class_③_方法": {"label": CLASS_LABELS["③"], "entries": []},
                    "class_④_统计": {"label": CLASS_LABELS["④"], "entries": []},
                    "class_⑤_数据": {"label": CLASS_LABELS["⑤"], "entries": []},
                    "class_⑥_硬件环境": {"label": CLASS_LABELS["⑥"], "entries": []},
                },
                "rca_entries": [],
                "rerun_rounds": 0,
                "honest_report_decision": {"in_effect": False},
                "fallback_exit": {"declared": False},
            },
        },
    }


def load_log() -> dict[str, Any]:
    """Load `.audit/decision_log.json` or return a fresh template."""
    if DECISION_LOG_PATH.exists():
        try:
            return json.loads(DECISION_LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[r_attribution] WARNING: existing log is malformed ({e}); starting fresh template",
                  file=sys.stderr)
            return _new_template()
    print(f"[r_attribution] no log at {DECISION_LOG_PATH}; starting fresh template",
          file=sys.stderr)
    return _new_template()


def save_log(log: dict[str, Any]) -> None:
    """Update header fields and atomically write log."""
    log["_git_commit"] = _get_git_commit()
    _atomic_write_json(DECISION_LOG_PATH, log)


def get_experiment(log: dict[str, Any], experiment_id: str = CURRENT_EXPERIMENT_ID) -> dict[str, Any]:
    exp = log.setdefault("experiments", {}).setdefault(experiment_id, _new_template()["experiments"][CURRENT_EXPERIMENT_ID])
    return exp


# ============================================================================
# Subcommands
# ============================================================================
def cmd_classify(args: argparse.Namespace) -> int:
    cls = args.cls_arg.strip()
    if cls not in VALID_CLASSES:
        print(f"[r_attribution] --class must be one of {sorted(VALID_CLASSES)}; got {cls!r}",
              file=sys.stderr)
        return 2
    if not args.phenomenon or not args.phenomenon.strip():
        print("[r_attribution] --phenomenon is required and must be non-empty",
              file=sys.stderr)
        return 2

    log = load_log()
    exp = get_experiment(log)
    register = exp.setdefault(
        "failure_classification_register",
        {k: {"label": v, "entries": []} for k, v in CLASS_LABELS.items()},
    )
    bucket_key = {
        "①": "class_①_实现",
        "②": "class_②_执行",
        "③": "class_③_方法",
        "④": "class_④_统计",
        "⑤": "class_⑤_数据",
        "⑥": "class_⑥_硬件环境",
    }[cls]
    bucket = register.setdefault(bucket_key, {"label": CLASS_LABELS[cls], "entries": []})
    bucket.setdefault("entries", []).append({
        "phenomenon": args.phenomenon.strip(),
        "evidence": (args.evidence or "").strip(),
        "status": "open",
        "timestamp": _now_iso(),
    })
    save_log(log)
    print(f"[r_attribution] classified into {cls} ({CLASS_LABELS[cls]}): {args.phenomenon[:60]}...",
          file=sys.stderr)
    return 0


def cmd_rca(args: argparse.Namespace) -> int:
    rca_id = (args.id or "").strip()
    if not rca_id:
        print("[r_attribution] --id is required (e.g., RCA-004)", file=sys.stderr)
        return 2
    for field in ("hypothesis", "evidence"):
        val = getattr(args, field, "") or ""
        if not val.strip():
            print(f"[r_attribution] --{field} is required and must be non-empty",
                  file=sys.stderr)
            return 2

    log = load_log()
    exp = get_experiment(log)
    entries = exp.setdefault("rca_entries", [])
    # If id already exists, update; else append
    existing = next((e for e in entries if e.get("id") == rca_id), None)
    record = {
        "id": rca_id,
        "phenomenon": (args.phenomenon or "").strip() or existing.get("phenomenon", "") if existing else (args.phenomenon or "").strip(),
        "reproduce_steps": (args.reproduce_steps or "").strip() or existing.get("reproduce_steps", "") if existing else (args.reproduce_steps or "").strip(),
        "hypothesis": args.hypothesis.strip(),
        "evidence": args.evidence.strip(),
        "conclusion": (args.conclusion or "").strip() or existing.get("conclusion", "") if existing else (args.conclusion or "").strip(),
        "status": args.status if (args.status in ("open", "closed")) else "open",
        "timestamp": _now_iso(),
    }
    if existing:
        # Preserve prior fields if user didn't pass them
        for k, v in existing.items():
            if k != "timestamp" and not record.get(k):
                record[k] = v
        existing.update(record)
        action = "updated"
    else:
        entries.append(record)
        action = "added"
    save_log(log)
    print(f"[r_attribution] RCA {rca_id} {action}: {args.hypothesis[:60]}...",
          file=sys.stderr)
    return 0


def cmd_rerun(args: argparse.Namespace) -> int:
    log = load_log()
    exp = get_experiment(log)
    current = int(exp.get("rerun_rounds", 0))
    if current >= 2:
        print(
            f"[r_attribution] rerun_rounds already at {current} (max 2 per R-3). "
            f"Further reruns require R-5 fallback exit. Aborting.",
            file=sys.stderr,
        )
        return 3
    exp["rerun_rounds"] = current + 1
    history = exp.setdefault("rerun_history", [])
    history.append({
        "round": exp["rerun_rounds"],
        "note": (args.note or "").strip(),
        "timestamp": _now_iso(),
    })
    save_log(log)
    print(f"[r_attribution] rerun_rounds: {current} -> {exp['rerun_rounds']} (note: {args.note!r})",
          file=sys.stderr)
    return 0


def cmd_fallback(args: argparse.Namespace) -> int:
    if not args.rationale or not args.rationale.strip():
        print("[r_attribution] --rationale is required and must be non-empty",
              file=sys.stderr)
        return 2
    log = load_log()
    exp = get_experiment(log)
    exp["fallback_exit"] = {
        "declared": True,
        "class": (args.fb_class or "未指定").strip(),
        "rationale": args.rationale.strip(),
        "declared_at": _now_iso(),
    }
    save_log(log)
    print(
        f"[r_attribution] R-5 fallback_exit declared: class={args.fb_class!r} "
        f"rationale={args.rationale[:60]}...",
        file=sys.stderr,
    )
    return 0


def cmd_honest(args: argparse.Namespace) -> int:
    """Toggle R-4 honest_report_decision.in_effect (optional helper)."""
    log = load_log()
    exp = get_experiment(log)
    in_effect = not exp.get("honest_report_decision", {}).get("in_effect", False)
    exp["honest_report_decision"] = {
        "in_effect": in_effect,
        "note": (args.note or "").strip(),
        "timestamp": _now_iso(),
    }
    save_log(log)
    print(f"[r_attribution] R-4 honest_report_decision.in_effect = {in_effect}",
          file=sys.stderr)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    log = load_log()
    exp = get_experiment(log)
    print("=" * 70)
    print(f"R-series Status — experiment_id = {CURRENT_EXPERIMENT_ID}")
    print(f"  log_path       = {DECISION_LOG_PATH}")
    print(f"  git_commit     = {log.get('_git_commit', '?')}")
    print(f"  schema_version = {log.get('_schema_version', '?')}")
    print("=" * 70)

    # R-1
    print("\n[R-1] failure_classification_register")
    register = exp.get("failure_classification_register", {})
    for sym in sorted(VALID_CLASSES):
        key = {
            "①": "class_①_实现", "②": "class_②_执行", "③": "class_③_方法",
            "④": "class_④_统计", "⑤": "class_⑤_数据", "⑥": "class_⑥_硬件环境",
        }[sym]
        bucket = register.get(key, {})
        n = len(bucket.get("entries", []))
        print(f"  {CLASS_LABELS[sym]}: {n} entry(s)")

    # R-2
    print("\n[R-2] rca_entries")
    rcas = exp.get("rca_entries", [])
    if not rcas:
        print("  (none)")
    for r in rcas:
        rid = r.get("id", "?")
        st = r.get("status", "?")
        ph = (r.get("phenomenon") or r.get("hypothesis") or "")[:60]
        print(f"  {rid} [{st}]: {ph}")

    # R-3
    print("\n[R-3] rerun_rounds")
    print(f"  count = {exp.get('rerun_rounds', 0)} (max 2 per handbook '同一失败最多重跑 2 轮')")

    # R-4
    print("\n[R-4] honest_report_decision")
    h = exp.get("honest_report_decision", {})
    print(f"  in_effect = {h.get('in_effect', False)}")
    if h.get("report_file"):
        print(f"  report_file = {h['report_file']}")

    # R-5
    print("\n[R-5] fallback_exit")
    f = exp.get("fallback_exit", {})
    print(f"  declared = {f.get('declared', False)}")
    if f.get("class"):
        print(f"  class    = {f.get('class')}")
    if f.get("rationale"):
        print(f"  rationale = {f.get('rationale')[:80]}...")
    print()
    return 0


# ============================================================================
# CLI entrypoint
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DECISION_LOG_PATH,
                        help="决策日志路径（默认 .audit/decision_log.json）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # classify
    p_cls = sub.add_parser("classify", help="R-1: 添加失败归类")
    p_cls.add_argument("--class", dest="cls_arg", required=True,
                       choices=sorted(VALID_CLASSES),
                       help="失败类别 ①②③④⑤⑥")
    p_cls.add_argument("--phenomenon", required=True, help="现象描述")
    p_cls.add_argument("--evidence", default="", help="证据")

    # rca
    p_rca = sub.add_parser("rca", help="R-2: 添加/更新 RCA 记录")
    p_rca.add_argument("--id", required=True, help="RCA ID (如 RCA-004)")
    p_rca.add_argument("--phenomenon", default="", help="现象")
    p_rca.add_argument("--reproduce-steps", default="", help="复现步骤")
    p_rca.add_argument("--hypothesis", default="", help="假设")
    p_rca.add_argument("--evidence", default="", help="证据")
    p_rca.add_argument("--conclusion", default="", help="结论")
    p_rca.add_argument("--status", choices=["open", "closed"], default="open",
                       help="RCA 状态")

    # rerun
    p_rerun = sub.add_parser("rerun", help="R-3: 增加重跑轮次（≤2）")
    p_rerun.add_argument("--note", default="", help="重跑原因备注")

    # fallback
    p_fb = sub.add_parser("fallback", help="R-5: 声明降级交付（兜底出口）")
    p_fb.add_argument("--class", dest="fb_class", default="", help="失败分类（如 方法性失败）")
    p_fb.add_argument("--rationale", required=True, help="降级交付理由（必填）")

    # honest
    p_h = sub.add_parser("honest", help="R-4: 切换诚实报告状态")
    p_h.add_argument("--note", default="", help="备注")

    # status
    sub.add_parser("status", help="打印当前 R-1..R-5 状态")

    args = parser.parse_args()
    if args.cmd == "classify":
        return cmd_classify(args)
    if args.cmd == "rca":
        return cmd_rca(args)
    if args.cmd == "rerun":
        return cmd_rerun(args)
    if args.cmd == "fallback":
        return cmd_fallback(args)
    if args.cmd == "honest":
        return cmd_honest(args)
    if args.cmd == "status":
        return cmd_status(args)
    parser.error(f"unknown cmd: {args.cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main())