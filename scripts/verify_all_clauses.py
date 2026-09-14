#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手册全量条款 → 代码实现 确定性核查（不用模糊启发式）。

匹配规则（权威）：
  - P/Pre/I/DQ/G/E 族：precheck_orchestrator.py 的 def check_<ID>_<slug>
  - C 族 + P15/P19/P22/P23/P28 子断言：clause_gates.py 的 def c<N>_* / p<N>_*
  - R 族：r_series.py 的 def R<N>_*
  - S 族：配置 + 物化 + block1 校验脚本（非单一 check 函数，按场景标注）

输出 outputs/clause_implementation_map.json，并打印 MISSING 清单。
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HANDBOOK = ROOT / "docs" / "zcode_异步手册_v696_20260914_最新.md"

# ---------- 1) 从手册提取真实条款标识 ----------
CLAUSE_RE = re.compile(
    r"(?<![A-Za-z0-9])(Pre-\d+|DQ-\d+|I-\d+|G-\d+|E-\d+|R-\d+|C-\d+|S\d+|P\d+[a-z]?)(?![0-9])"
)
# 噪声白名单：手册里 P50/P95 等是百分位/单元数数值，非条款号
NON_CLAUSES = {"P50", "P95", "P99"}

def extract_clauses() -> dict[str, list[int]]:
    txt = HANDBOOK.read_text(encoding="utf-8")
    inv: dict[str, list[int]] = {}
    for i, l in enumerate(txt.splitlines(), 1):
        for m in CLAUSE_RE.finditer(l):
            cid = m.group(1)
            if cid in NON_CLAUSES:
                continue
            inv.setdefault(cid, []).append(i)
    return inv

# ---------- 2) 源码函数索引 ----------
def index_funcs(*rels: str) -> dict[str, tuple[str, int]]:
    idx: dict[str, tuple[str, int]] = {}
    for rel in rels:
        p = ROOT / rel
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", txt, re.M):
            ln = txt.count("\n", 0, m.start()) + 1
            idx.setdefault(m.group(1), (rel, ln))
    return idx

PO_F = index_funcs(
    "src/liquidloc/common/precheck_orchestrator.py",
    "src/liquidloc/analysis/clause_gates.py",
    "src/liquidloc/analysis/r_series.py",
    "src/liquidloc/dataio/handbook_gates.py",
    "scripts/_run_25unit.py",
    "scripts/_g_e_audit.py",
)

# ---------- 3) 确定性匹配（前缀严格，不用子串）----------
def prefix_of(fn: str) -> str:
    """check_P1_x→p1, c6_ram→c6, R1_check→r1, _pre5→pre5, p15_pilot→p15, dq1→dq1。"""
    m = re.match(r"^_?(?:check_)?([prdceg])_?(\d+[a-z]?)", fn.lower())
    if m:
        return m.group(1) + m.group(2)
    m = re.match(r"^_?(pre)(\d+)", fn.lower())  # check_Pre1 / _pre1 已在上面覆盖 pre5
    if m:
        return m.group(1) + m.group(2)
    m = re.match(r"^_?(dq)(\d+)", fn.lower())
    if m:
        return m.group(1) + m.group(2)
    return ""

def norm(cid: str) -> str:
    return re.sub(r"[^a-z0-9]", "", cid.lower())

# 函数前缀 → 归一化条款号 的显式别名（处理 I/G/E 在 _g_e_audit 的 _g1/_e1 前缀）
ALIASES = {}

def match_clause(cid: str) -> tuple[str | None, str | None, int | None]:
    n = norm(cid)
    # pre-check 用 check_<cid> 精确前缀（如 check_P1_、check_Pre1_、check_I1_、check_DQ1_、check_G1_、check_E1_）
    for fn, (rel, ln) in PO_F.items():
        low = fn.lower()
        if low.startswith("check_"):
            core = norm(fn[len("check_"):])
            if core == n or core.startswith(n):
                return fn, rel, ln
    # clause_gates c<N>/p<N>/... 与 _pre<N>/R<N>/_g<N>/_e<N>
    for fn, (rel, ln) in PO_F.items():
        if prefix_of(fn) == n:
            return fn, rel, ln
    return None, None, None

# ---------- 4) S 族与场景层（配置 + 校验脚本，非单一函数）----------
S_IMPL = {
    "S1": ("configs/base/scene_axis_protocol.yaml + data/raw/*/anchor_layout.json", "场景范围/锚区/凸包"),
    "S2": ("scene_axis_protocol.yaml[arbitration] + .audit/decision_log.json + P4/P5", "K1 GDOP 协议裁决"),
    "S3": ("configs/models/*.yaml + src/liquidloc/fusion/ + state_definition.py", "状态方程/初始化(10维)"),
    "S4": ("configs/base/sim_noise_spec.yaml + sim_materializer.py", "噪声/协方差 IMU/UWB"),
    "S5": ("src/liquidloc/scenarios/async_levels.py + 物化层", "异步时间戳 A2/A3 offset/jitter/skew"),
    "S6": ("src/liquidloc/scenarios/nlos_levels.py + paper_nlos_injection.py", "NLOS 注入 bias+多径+噪声"),
    "S7": ("configs/base/sim_noise_spec.yaml M1", "成簇丢包 5%"),
    "S8": ("src/liquidloc/scenarios/visual_levels.py", "V0 健康档 VIO payload"),
    "S9": ("src/liquidloc/dataio/handbook_gates.py S9ValidationReport + src/liquidloc/analysis/block1_gdop_verifier.py", "协议一致性四重实测 + schema校验"),
}

def main():
    inv = extract_clauses()
    report = []
    for cid in sorted(inv, key=lambda x: (norm(x)[:2], norm(x))):
        fn, rel, ln = match_clause(cid)
        if fn:
            report.append({"clause": cid, "handbook_lines": inv[cid][:5],
                           "func": fn, "file": rel, "line": ln, "kind": "function"})
        elif cid in S_IMPL:
            anchor, desc = S_IMPL[cid]
            report.append({"clause": cid, "handbook_lines": inv[cid][:5],
                           "func": anchor, "file": "config+script", "line": None, "kind": "config"})
        else:
            report.append({"clause": cid, "handbook_lines": inv[cid][:5],
                           "func": None, "file": None, "line": None, "kind": "MISSING"})
    out = ROOT / "outputs" / "clause_implementation_map.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    miss = [r["clause"] for r in report if r["kind"] == "MISSING"]
    print(f"手册真实条款数: {len(report)}")
    print(f"函数承载: {sum(1 for r in report if r['kind']=='function')}")
    print(f"配置承载(S族): {sum(1 for r in report if r['kind']=='config')}")
    print(f"MISSING: {miss if miss else '无'}")
    print(f"→ {out}")
    return 1 if miss else 0

if __name__ == "__main__":
    sys.exit(main())