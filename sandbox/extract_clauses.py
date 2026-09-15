# -*- coding: utf-8 -*-
"""手册条款逐字抽取器（第 1 阶段工具链 B2-1，对应计划书 sandbox/extract_clauses.py）。

职责:
  从 docs/异步实验_执行手册_终版v15_总纲条款版.md 按全局编号区间逐字抽取条款正文，
  供第 1 阶段落实报告的「条款:」字段注入（禁止手抄/摘要/改写——判定铁律：字段3 未
  逐字全文 = 未落实）。

抽取规则（与计划书 B4 裁决一致）:
  - 条款起始行 = 行首 `**[n]**`（正则 ^\\*\\*\\[\\d+\\]\\*\\*）。
  - 条款体 = 起始行起，到下一条条款起始行或 `## ` 节标题行（或 EOF）前一行；
    去掉体尾部空行；跨行体（表格/引用块/代码块）**保留原换行逐字输出**，不缝合不改写。
  - 「条款:」字段全文含起始编号标记 `**[n]** ...`（与第 0 阶段报告同构）。

断言:
  - 每条起始行 startswith `**[n]**`；
  - 请求区间内编号必须连续存在（缺号 = 抽取失败，防跳号铁律被工具层拦截）；
  - 抽取文本与手册切片逐字相等（防上游文件中途被改的时漂——重读磁盘核对）。

用法:
  python sandbox/extract_clauses.py 6 18          # 打印区间内各条（jsonl: {"n":..,"text":..}）
  from extract_clauses import extract_clauses     # 供装配脚本 import 使用
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HANDBOOK = REPO / "docs" / "异步实验_执行手册_终版v15_总纲条款版.md"
CLAUSE_START = re.compile(r"^\*\*\[(\d+)\]\*\*")


def _load_lines() -> list[str]:
    """每次从磁盘重读手册（状态只认磁盘，治理条款 [2]）。"""
    return HANDBOOK.read_text(encoding="utf-8").splitlines()


def clause_index(lines: list[str]) -> dict[int, int]:
    """返回 条款编号 -> 起始行下标（0-based）。行首 `**[n]**` 才算条款行。"""
    index: dict[int, int] = {}
    for i, line in enumerate(lines):
        m = CLAUSE_START.match(line)
        if m:
            n = int(m.group(1))
            if n in index:
                raise AssertionError(f"条款编号重复: [{n}] 出现在行 {index[n]+1} 与 {i+1}")
            index[n] = i
    return index


def _section_bounds(lines: list[str]) -> list[int]:
    """所有 `## ` 节标题行下标，用于截断最后一个条款体。"""
    return [i for i, line in enumerate(lines) if line.startswith("## ")]


def extract_clause(lines: list[str], n: int, index: dict[int, int]) -> str:
    """抽取第 n 条全文（含 `**[n]**` 标记），逐字。"""
    if n not in index:
        raise AssertionError(f"手册中不存在条款 [{n}]（禁止跳号书写报告）")
    start = index[n]
    header = lines[start]
    assert header.startswith(f"**[{n}]**"), f"[{n}] 起始行异常: {header[:40]}"
    # 体结束 = 下一条条款行 或 其后最近的节标题行 的前一行
    ends = [index[m] for m in index if index[m] > start]
    ends += [s for s in _section_bounds(lines) if s > start]
    end = min(ends) if ends else len(lines)
    body = lines[start + 1 : end]
    while body and not body[-1].strip():  # 仅去尾部空行（空行是分隔符不属正文）
        body.pop()
    return "\n".join([header, *body])


def extract_clauses(a: int, b: int) -> dict[int, str]:
    """抽取 [a..b] 闭区间全部条款；编号必须连续。"""
    lines = _load_lines()
    index = clause_index(lines)
    out: dict[int, str] = {}
    for n in range(a, b + 1):
        out[n] = extract_clause(lines, n, index)
        # 逐字回验：重读磁盘切片必须与返回文本一致（防拼装走样）
        assert out[n].startswith(f"**[{n}]**"), n
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = int(sys.argv[1]), int(sys.argv[2])
    data = extract_clauses(a, b)
    for n in sorted(data):
        rec = {"n": n, "text": data[n]}
        print(json.dumps(rec, ensure_ascii=False))
    print(f"# 共 {len(data)} 条，编号 [{a}]..[{b}] 连续", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
