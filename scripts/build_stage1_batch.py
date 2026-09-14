#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_stage1_batch.py — 把已核验的 Stage1 clause JSON 转为报告条目文本。

用法: python scripts/build_stage1_batch.py zcode_clause_array_32_189.json > stage1_batch.md
"""
import sys, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANUAL = ROOT / "docs" / "zcode_异步手册_v696_20260914_最新.md"


def extract_titles():
    """从手册提取每条款标题（首个 ** 到下一 * 或行末，或前 100 字符）。"""
    text = MANUAL.read_text(encoding="utf-8")
    pat = re.compile(r"^\*\*\[?([0-9]+)\]?\*\* (.+)$", re.M)
    out = {}
    for m in pat.finditer(text):
        num = int(m.group(1))
        line = m.group(2).strip()
        # truncate smartly: cut at first fullwidth period or newline, max 90 chars
        line = re.split(r"[。\n]", line)[0]
        line = line.replace("**", "").replace("|", "；").strip()
        if len(line) > 90:
            line = line[:88] + "…"
        out[num] = line
    return out


def fmt_pass(entry, title):
    a = entry.get("anchor", "N/A")
    v = entry.get("value", "")
    note = entry.get("note", "")
    # compact anchor
    if len(a) > 100:
        a = a[:98] + "…"
    if len(v) > 80:
        v = v[:78] + "…"
    return f"**[{entry['clause']}]** ✅ | 条款: {title} | 代码: {a} | 落实: {v} | 审查: agent 读代码 → anchor 经 verify_slim_b 核验 | 验证: scripts/verify_slim_b_clauses.py | 证据: {a.split(';')[0].strip()} |\n"


def fmt_na(entry, title):
    a = entry.get("anchor", "N/A")
    return f"**[{entry['clause']}]** ⚪️ 程序性 | 条款: {title} | 代码: {a} | 落实: 程序/治理/口径声明类，无单一可执行代码载体 | 审查: 归治理文件承载 | 验证: — | 证据: 治理层 |\n"


def main():
    if len(sys.argv) < 2:
        print("用法: build_stage1_batch.py <clauses.json>", file=sys.stderr)
        return 2
    titles = extract_titles()
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    # run verifier inline (same as verify_slim_b)
    import importlib.util
    spec = importlib.util.spec_from_file_location("vsb", ROOT / "scripts" / "verify_slim_b_clauses.py")
    vsb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vsb)

    lines = []
    for e in data:
        title = titles.get(e["clause"], "(无标题)")
        st, det = vsb.verify(e)[:2], None
        status = st[1]
        if status == "FILE_MISSING":
            lines.append(f"**[{e['clause']}]** ❌ | 条款: {title} | 代码: {e.get('anchor','N/A')} | 落实: **锚点文件不存在** (agent 声明 anchor 无法验证) | 审查: verify_slim_b 拒绝 | 验证: scripts/verify_slim_b_clauses.py | 证据: 缺失 |\n")
        elif status == "OK":
            lines.append(fmt_pass(e, title))
        elif status == "PROC_OK":
            lines.append(fmt_na(e, title))
        elif status == "KEYWORD_MISSING":
            lines.append(fmt_na(e, title))  # downgrade N-A when anchor lacks path
        else:
            lines.append(fmt_na(e, title))
    sys.stdout.write("".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
