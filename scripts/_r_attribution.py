"""R-1..R-5 失败归因与重跑决策系统。

R-1 失败归因分类: 6 类失败 (①实现性 ②执行性 ③方法性 ④统计性 ⑤数据性 ⑥硬件/环境)
R-2 RCA 记录: 现象→复现→假设→证据→结论 写入 decision log
R-3 重跑决策: ≤2 轮, 5 个流程回归点
R-4 诚实报告: 负结果报告口径, 不 p-hack
R-5 兜底出口: 降级交付需用户验收裁决

P1..P10 fail-safe 补救动作（P 系列 1-10）:
P1: decision log 空窗/记录缺失补救
P2: 时间戳倒挂补救
P3: 统计未双重复核补救
P4: 抽查复跑不一致补救
P5: 告警被静默清除补救
P6: gate 走过场补救
P7: code freeze 后改码补救
P8: 旧权重热启动补救
P9: 数据混用补救
P10: 统计口径漂移补救
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DECISION_LOG_PATH = Path("E:/异步高NLOS/.audit/decision_log.json")
DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# R-1: 6 类失败分类（handbook R-1 line 765）
CLASSIFICATION_BUCKETS = [
    "class_①_实现",  # 代码/数据/口径 bug
    "class_②_执行",  # gate 未过/审计未过
    "class_③_方法",  # 机制/优势不存在
    "class_④_统计",  # 功效不足
    "class_⑤_数据",  # 数据集不合适
    "class_⑥_硬件环境",  # 硬件/环境性失败
]


@dataclass
class FailureRecord:
    """R-1: 失败归因记录（写入 decision_log.json 的 failure_classification_register）"""
    bucket: str               # one of CLASSIFICATION_BUCKETS
    phenomenon: str           # 现象
    status: str = "open"      # open | closed
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "phenomenon": self.phenomenon,
            "status": self.status,
            "timestamp": self.timestamp,
        }


@dataclass
class RCAEntry:
    """R-2: RCA 记录（现象→复现→假设→证据→结论）"""
    phenomenon: str
    reproduce_steps: str = ""
    hypothesis: str = ""
    evidence: str = ""
    conclusion: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "phenomenon": self.phenomenon,
            "reproduce_steps": self.reproduce_steps,
            "hypothesis": self.hypothesis,
            "evidence": self.evidence,
            "conclusion": self.conclusion,
            "timestamp": self.timestamp,
        }


@dataclass
class RemediationAction:
    """P1..P10: 失败补救动作（10 个补救点）"""
    code: str           # "P1".."P10"
    audit_finding: str
    action: str
    evidence_path: str = ""
    completed: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "audit_finding": self.audit_finding,
            "action": self.action,
            "evidence_path": self.evidence_path,
            "completed": self.completed,
            "timestamp": self.timestamp,
        }


# P1..P10 补救动作规范（handbook lines 744-753）
P_REMEDIATION_SPECS = [
    {
        "code": "P1",
        "audit_finding": "decision log 空窗/记录缺失 (E-1/E-2)",
        "action": "从 git 提交历史 + 训练日志时间戳链反推该区间真实事件，逐条补记（补记须标注'审计后补录'+ 依据来源）",
        "re_audit": "E-2: gate 通过记录可追溯",
    },
    {
        "code": "P2",
        "audit_finding": "时间戳倒挂/日志与 commit 对不上 (E-1)",
        "action": "定位倒挂区间（decision log vs git vs 训练日志三方交叉），隔离到 archive",
        "re_audit": "E-1 三方时间戳链无倒挂",
    },
    {
        "code": "P3",
        "audit_finding": "统计未双重复核 (E-3)",
        "action": "由第二人/独立脚本对五方法均值/排序/显著性（A-1/A-2/A-5）复算",
        "re_audit": "E-3 复算一致记录在案",
    },
    {
        "code": "P4",
        "audit_finding": "抽查复跑不一致 (E-4)",
        "action": "按 E-4 五档出口处置: 浮点末位差接受 / 同机差异查确定性模式 / 跨机记录来源 / 单元级作废重跑 / 整批核查",
        "re_audit": "E-4 抽查通过 + G-1 复核",
    },
    {
        "code": "P5",
        "audit_finding": "告警被静默清除/吞异常 (E-5)",
        "action": "从日志恢复被清除的告警记录，对每一条告警补处理记录（原因/处置/影响单元）",
        "re_audit": "E-5 告警清零有据可查 (G-4)",
    },
    {
        "code": "P6",
        "audit_finding": "gate 走过场 (Pre/I/S9/DQ/冒烟任一)",
        "action": "该 gate 按对应清单完整重做（不做局部修补），重做留痕（时间戳 + 操作人 + 结果）",
        "re_audit": "E-2: gate 通过证据完整",
    },
    {
        "code": "P7",
        "audit_finding": "code freeze 后改码未重新冒烟",
        "action": "解冻 → 记录改动（decision log + git commit）→ 重新冒烟 → 重新冻结",
        "re_audit": "冒烟 gate ①/② + RZ-0/RZ-1 留痕",
    },
    {
        "code": "P8",
        "audit_finding": "旧权重热启动/续训/load 错路径 (RZ-2 违规)",
        "action": "代码层排查所有 load_state_dict/checkpoint 加载路径，确认不指向 archive/旧权重",
        "re_audit": "RZ-2 重训留痕 + G-1 单元齐全",
    },
    {
        "code": "P9",
        "audit_finding": "数据混用 (冒烟数据并入全量 / 部分 seed 新旧混用)",
        "action": "按 RZ-1 禁令核查全量数据来源（manifest + 校验和 Pre-4），混用命中 → RZ-1 全量重生成",
        "re_audit": "RZ-1 留痕 + S9/DQ 报告 + G-3 配置-数据交叉",
    },
    {
        "code": "P10",
        "audit_finding": "统计口径漂移 (换检验/换分析单位/换判据未登记)",
        "action": "按 Pre-2 三级分类定性（协议变更/分析变更/执行偏差），写入 decision log",
        "re_audit": "Pre-2 变更登记 + 统计口径全局唯一声明复核",
    },
]


def load_decision_log() -> dict[str, Any]:
    if not DECISION_LOG_PATH.is_file():
        return _empty_log()
    try:
        return json.loads(DECISION_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _empty_log()


def save_decision_log(log: dict[str, Any]) -> None:
    DECISION_LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def _empty_log() -> dict[str, Any]:
    return {
        "failure_classification_register": {b: [] for b in CLASSIFICATION_BUCKETS},
        "rca_entries": [],
        "remediation_actions": [],
        "rerun_rounds": 0,
        "honest_report_decision": {"in_effect": False, "report_file": ""},
        "fallback_exit": {"declared": False, "class": "", "rationale": ""},
        "protocol_arbitration_register": {},
    }


def r1_classify_failure(log: dict[str, Any], record: FailureRecord) -> dict[str, Any]:
    """R-1: 失败归因分类 — 6 类失败之一."""
    if record.bucket not in CLASSIFICATION_BUCKETS:
        return {"ok": False, "error": f"invalid bucket: {record.bucket}"}
    log["failure_classification_register"].setdefault(record.bucket, []).append(record.to_dict())
    save_decision_log(log)
    return {
        "ok": True,
        "bucket": record.bucket,
        "classification_registered": record.to_dict(),
        "total_in_bucket": len(log["failure_classification_register"][record.bucket]),
    }


def r2_record_rca(log: dict[str, Any], entry: RCAEntry) -> dict[str, Any]:
    """R-2: RCA 记录 — 现象→复现→假设→证据→结论."""
    if not all([entry.phenomenon, entry.reproduce_steps, entry.hypothesis, entry.evidence, entry.conclusion]):
        return {"ok": False, "error": "RCAEntry requires phenomenon/reproduce_steps/hypothesis/evidence/conclusion"}
    log["rca_entries"].append(entry.to_dict())
    save_decision_log(log)
    return {"ok": True, "rca_index": len(log["rca_entries"]) - 1, "entry": entry.to_dict()}


def r3_check_rerun(log: dict[str, Any], attempted: int) -> dict[str, Any]:
    """R-3: 重跑决策 — ≤2 轮, 5 流程回归点."""
    over_limit = attempted > 2
    log["rerun_rounds"] = attempted
    save_decision_log(log)
    return {
        "ok": not over_limit,
        "rerun_rounds": attempted,
        "max_rounds": 2,
        "flow_regression_points": [
            "① 改码/改配置 → 重新冒烟",
            "② 换数据 → S9 校验 + DQ 适配性复核 + G-3 配置-数据交叉",
            "③ 统计口径变更 → 统计脚本冒烟验证",
            "④ 任何重跑 → G-1..G-5 完整性 + E-1..E-6 审计",
            "⑤ 重跑后指标变化 >5% → 记入 decision log 并说明原因",
        ],
    }


def r4_declare_honest_report(log: dict[str, Any], in_effect: bool, report_file: str = "") -> dict[str, Any]:
    """R-4: 诚实报告规范 — 负结果报告口径, 不 p-hack."""
    log["honest_report_decision"] = {
        "in_effect": bool(in_effect),
        "report_file": str(report_file),
    }
    save_decision_log(log)
    return {"ok": True, "in_effect": bool(in_effect), "report_file": str(report_file)}


def r5_declare_fallback(
    log: dict[str, Any],
    declared: bool,
    cls: str = "",
    rationale: str = "",
) -> dict[str, Any]:
    """R-5: 兜底出口 — 降级交付需用户验收裁决.

    前置条件: E-1..E-6 + G-1..G-5 + 八层诊断 + 重跑纪律遵守（≤2 轮） + R-4 诚实报告声明。
    """
    if declared and (not cls or not rationale):
        return {"ok": False, "error": "fallback declared requires class + rationale"}
    log["fallback_exit"] = {
        "declared": bool(declared),
        "class": str(cls),
        "rationale": str(rationale),
    }
    save_decision_log(log)
    return {
        "ok": True,
        "declared": bool(declared),
        "class": str(cls),
        "rationale": str(rationale),
        "prerequisites": [
            "E-1..E-6 audit pass",
            "G-1..G-5 integrity pass",
            "D-1..D-31 8-layer diagnostics complete",
            "R-3 rerun rounds ≤2",
            "R-4 honest_report_decision.in_effect == True",
        ],
    }


def record_remediation(log: dict[str, Any], action: RemediationAction) -> dict[str, Any]:
    """P1..P10: 记录补救动作（写入 decision_log.json 的 remediation_actions）."""
    valid_codes = {s["code"] for s in P_REMEDIATION_SPECS}
    if action.code not in valid_codes:
        return {"ok": False, "error": f"invalid code: {action.code}; expected one of {sorted(valid_codes)}"}
    log["remediation_actions"].append(action.to_dict())
    save_decision_log(log)
    return {
        "ok": True,
        "action_index": len(log["remediation_actions"]) - 1,
        "action": action.to_dict(),
    }


def list_remediation_specs() -> list[dict[str, str]]:
    """返回 P1..P10 补救动作规范（handbook lines 744-753）."""
    return P_REMEDIATION_SPECS


def main():
    print("[R-1..R-5] Failure attribution + remediation system ready.")
    print(f"  decision log: {DECISION_LOG_PATH}")
    print()
    print("  R-1: classify_failure(bucket, phenomenon)")
    print("  R-2: record_rca(phenomenon, reproduce_steps, hypothesis, evidence, conclusion)")
    print("  R-3: check_rerun(attempted) — fails if attempted > 2")
    print("  R-4: declare_honest_report(in_effect, report_file)")
    print("  R-5: declare_fallback(declared, class, rationale)")
    print(f"  P1..P10: record_remediation(code, audit_finding, action)")
    print()
    print("  Available classifications:")
    for b in CLASSIFICATION_BUCKETS:
        print(f"    {b}")
    sys.exit(0)


if __name__ == "__main__":
    main()
