# -*- coding: utf-8 -*-
"""Parse handbook sub-clauses (P1..P39, Pre/DQ/I/R families) + map each to code line."""
from pathlib import Path
import re
import json

ROOT = Path(__file__).resolve().parents[1]
HANDBOOK = ROOT / "docs" / "异步高NLOS实验全流程保障手册.md"
PO = ROOT / "src" / "liquidloc" / "common" / "precheck_orchestrator.py"
RS = ROOT / "src" / "liquidloc" / "analysis" / "r_series.py"


def extract_subclauses(hb_text: str) -> list[dict]:
    """从手册 Markdown 提取所有带编号的条款段落 (P1/P2/P3①/P3②...)。"""
    out = []
    lines = hb_text.splitlines()
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        # P1/P2/P3①/P4②: heading pattern
        m = re.match(r"^(P\d+|Pre-\d|I-\d|DQ-\d|G-\d|E-\d|R-\d)([①②③④⑤⑥⑦⑧⑨⑩]|.*?\([一二三四五六七八九十]+\)\)|\d+\.\d+\.\d+\.?\d*)[::]?\s+(.*)", l, re.I)
        if m:
            fam = m.group(1).upper()
            sub = m.group(2) or ""
            desc = m.group(3)
            # gather description body (indented or next lines until blank/new heading)
            body = []
            j = i + 1
            while j < len(lines) and (lines[j].startswith("  ") or lines[j].startswith("\t") or not lines[j].strip()):
                if not lines[j].strip():
                    if body: break
                else:
                    body.append(lines[j].strip())
                    j += 1
            out.append({"id": f"{fam}{sub}", "family": fam, "text": desc + " " + " ".join(body)})
            i = j
        else:
            i += 1
    return out


def find_code_location(clause_id: str, code_files: list[Path]) -> list[tuple[str, int]]:
    """在代码中搜索与该条款 ID 对应的函数定义位置。
    
    Returns [(file_path, line_no), ...]
    """
    fname = clause_id.split("①")[0]
    hits = []
    for f in code_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in re.finditer(rf"(?:def\s+|check_)\s*{re.escape(fname)}(?:_[a-z0-9]+)?\b", text, re.I):
            start = max(0, m.start() - 200)
            end = min(len(text), m.end() + 500)
            line_no = text.count("\n", 0, m.start()) + 1
            hits.append((str(f.relative_to(ROOT)), line_no))
    return hits


def main():
    hb = HANDBOOK.read_text(encoding="utf-8")
    subs = extract_subclauses(hb)
    print(f"Extracted {len(subs)} sub-clauses from handbook")
    po_txt = PO.read_text(encoding="utf-8")
    rs_txt = RS.read_text(encoding="utf-8")
    
    results = []
    for s in subs:
        cid = s["id"]
        hits = find_code_location(cid, [PO, RS])
        if not hits:
            # try grep function name directly
            func_name = f"check_{'_'.join(s['family'].lower().split('-'))}{s['id'].split(s['family'][-1])[-1]}"
            if re.search(rf"def\s+{func_name}", po_txt):
                hits = [("src/liquidloc/common/precheck_orchestrator.py", po_txt.index(f"def {func_name}") // 200)]
        results.append({
            "id": cid,
            "family": s["family"],
            "desc_preview": s["text"][:120],
            "code_hits": hits[:3],  # top 3 matches
        })
    
    Path("outputs/subclause_map.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    no_hit = [r for r in results if not r["code_hits"]]
    print(f"Mapping complete: {len(no_hit)} clauses lack clear code location")


if __name__ == "__main__":
    main()
