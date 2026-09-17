#!/usr/bin/env python3
"""Extract clause data from the Stage 1 report and generate JSON for verification."""
import re, json, sys
from pathlib import Path

REPORT = Path(r"E:\异步高NLOS\docs\异步实验_执行手册_终版v15_总纲条款版-落实报告.md")

def extract_clauses(text):
    """Extract clause entries from report text."""
    entries = []
    # Match lines like: [6] ✅ | 条款: xxx | 代码: file.py:line | 落实: xxx | 审查: xxx | 验证: xxx | 证据: file.py:line
    pattern = re.compile(
        r'\[(\d+)\]\s+(✅|❌)\s+\|\s+条款:\s*([^|]+)\s+\|\s+代码:\s*([^|]+)\s+\|\s+落实:\s*([^|]+)\s+\|\s+审查:\s*([^|]+)\s+\|\s+验证:\s*([^|]+)\s+\|\s+证据:\s*([^\n]+)'
    )
    for m in pattern.finditer(text):
        clause_num = int(m.group(1))
        verdict = "PASS" if m.group(2) == "✅" else "FAIL"
        clause_name = m.group(3).strip()
        anchor = m.group(4).strip()
        impl = m.group(5).strip()
        review = m.group(6).strip()
        verify = m.group(7).strip()
        evidence = m.group(8).strip()
        entries.append({
            "clause": clause_num,
            "verdict": verdict,
            "anchor": anchor,
            "value": impl[:100] if impl else "",
            "note": evidence
        })
    return entries

def main():
    text = REPORT.read_text(encoding="utf-8")
    entries = extract_clauses(text)
    print(f"Found {len(entries)} clauses")
    # Output as JSON array
    json.dump(entries, sys.stdout, ensure_ascii=False, indent=2)
    print()

if __name__ == "__main__":
    main()
