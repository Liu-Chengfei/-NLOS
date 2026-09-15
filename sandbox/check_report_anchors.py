# -*- coding: utf-8 -*-
"""报告锚点真实性断言器（第 1 阶段工具链 B2-3，计划书 sandbox/check_report_anchors.py）。

背景: 第 0 阶段人工审计曾抓到臆造引用（run_clear_stage0.py 不存在、行号漂移），
本脚本把「落实证据必须引用文件:行号 + 行号与文件实际一致」（底线 A9 末条）固化为
机械断言。批报告装配后必须跑通本器才允许提交。

校验规则:
  1. `路径::符号:起[-止]`      —— 文件存在、行号在总行数内、起<=止；
     符号为 ASCII 标识符时按单词边界在文件全文检索；含 CJK 时按子串检索
     （治理类章节锚如 `AGENTS.md::五铁律:12-17` 命中 `## 五铁律` 标题行即可）。
  2. `路径.ext:起[-止]`        —— 同上（不含符号），路径须带扩展名。
  3. `手册第N行(条款原文唯一真源)` —— 手册第 N 行必须以该记录条款号 `**[n]**` 起始。
  4. `手册第N[-M]行`（其它语境）   —— N/M 在手册总行数范围内。
  5. 记录内出现的 `logs/...`、`data/...` 等相对路径证据（证据字段）——文件存在。

校验规则（按字段分治，避免把"验证命令里的假想目标/叙述性旧文件名"误判）:
  - 只扫描 代码:/落实:/证据: 三个字段的锚点；条款:/审查:/验证: 字段只核 `手册第N行`。
  1. `路径::符号:起[-止]`      —— 文件存在、行号在总行数内、起<=止；
     符号为 ASCII 标识符时按单词边界在文件全文检索；含 CJK 时按子串检索（W: 软判据）。
  2. `路径.ext:起[-止]`        —— 同上（不含符号），路径须带扩展名。
  3. 裸文件名（如 `SKILL.md`）—— 按 basename 在仓库内唯一命中解析（排除 .git/archive/
     缓存目录），无行号则只核存在性。
  4. `手册第N行(条款原文唯一真源)` —— 手册第 N 行必须以该记录条款号 `**[n]**` 起始；
     其它 `手册第N[-M]行` 引用只核行号范围。
  5. 证据字段中的 `xxx/yyy.ext` 相对路径（分号分隔逐项）——文件存在；
     含"已删除/已并入/不另存/旧"字样的叙述项跳过。

用法: python sandbox/check_report_anchors.py <报告文件>
  默认: FAIL 项 exit 1；WARN 项（CJK 符号未命中等软判据）打印但不拦截。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HANDBOOK = REPO / "docs" / "异步实验_执行手册_终版v15_总纲条款版.md"
CB = r"\u4e00-\u9fff"  # CJK 基本区
PATH_CHARS = rf"A-Za-z0-9_\-./\\{CB}（）()"
EXTS = r"(?:py|yaml|yml|md|sh|txt|json|toml|log|cfg|ini|lock|csv|jsonl|png|pdf)"

RE_SYM = re.compile(rf"(?P<path>[{PATH_CHARS}]+?)::(?P<sym>[^:;\s|]+?):(?P<lo>\d+)(?:\s*[-–]\s*(?P<hi>\d+))?")
RE_PLAIN = re.compile(rf"(?P<path>[{PATH_CHARS}]*?\.{EXTS}):(?P<lo>\d+)(?:\s*[-–]\s*(?P<hi>\d+))?(?!\d)")
RE_HB_SRC = re.compile(r"手册第(?P<lo>\d+)行\(条款原文唯一真源\)")
RE_HB_ANY = re.compile(r"手册第(?P<lo>\d+)(?:-(?P<hi>\d+))?行")
RE_RECORD = re.compile(r"^\[(\d+)\]\s*(✅|❌)\s*\|")
RE_EVID_PATH = re.compile(rf"(?P<path>[{PATH_CHARS}]*\.{EXTS})")
RE_NARRATIVE = re.compile(r"已删除|已并入|不另存|旧 |从仓库根删除")
SKIP_FILES = {".git", "archive", "__pycache__", ".venv-gpu", ".venv", "node_modules", ".venv312", "sandbox"}

_line_cache: dict[Path, list[str] | None] = {}
_basename_cache: dict[str, list[Path]] = {}


def _lines(path: Path) -> list[str] | None:
    """安全读行: 文件不存在返回 None。"""
    if path in _line_cache:
        return _line_cache[path]
    try:
        data = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        data = None
    _line_cache[path] = data
    return data


def _find_bare(name: str) -> Path | None:
    """裸文件名按 basename 全仓唯一命中解析；歧义/无命中返回 None。"""
    if name not in _basename_cache:
        hits: list[Path] = []
        for p in REPO.rglob(name):
            if set(p.relative_to(REPO).parts) & SKIP_FILES:
                continue
            hits.append(p)
        _basename_cache[name] = hits
    hits = _basename_cache[name]
    return hits[0] if len(hits) == 1 else None


def _resolve(raw: str) -> Path | None:
    """把锚点里的相对路径解析到仓库根（兼容 / 与 \\ 与绝对路径与裸文件名回退）。"""
    p = Path(raw.replace("\\", "/"))
    cand = p if p.is_absolute() else REPO / p
    if cand.exists():
        return cand
    if "/" not in raw and "\\" not in raw:  # 裸文件名 → 全仓唯一命中回退
        return _find_bare(raw)
    return None


def split_fields(text: str) -> dict[str, str]:
    """按 ' | 字段: ' 边界切八字段（条款字段内部的竖线不参与切分——先定位已知字段标签）。"""
    out: dict[str, str] = {}
    labels = ["条款:", "代码:", "落实:", "审查:", "验证:", "证据:"]
    pos: list[tuple[str, int]] = []
    for lab in labels:
        i = text.find(" | " + lab)
        if i == -1 and text.lstrip().split("|", 1)[-1].lstrip().startswith(lab):
            i = text.find(lab)
        pos.append((lab, i))
    ordered = [(lab, i) for lab, i in pos if i >= 0]
    for k, (lab, i) in enumerate(ordered):
        j = ordered[k + 1][1] if k + 1 < len(ordered) else len(text)
        out[lab.rstrip(":")] = text[i + 3 : j]
    return out


def check_anchors(n: int, text: str, hb_lines: list[str]) -> list[str]:
    """核对一条记录（按字段分治），返回问题列表（'W:' 前缀 = 软判据）。"""
    problems: list[str] = []
    fields = split_fields(text)
    hard = "\n".join(fields.get(k, "") for k in ("代码", "落实", "证据"))
    soft = "\n".join(fields.get(k, "") for k in ("条款", "审查", "验证"))
    seen_spans: set[tuple[str, str]] = set()
    for m in RE_SYM.finditer(hard):
        raw, sym, lo = m.group("path"), m.group("sym"), int(m.group("lo"))
        hi = int(m.group("hi")) if m.group("hi") else lo
        seen_spans.add((raw, m.group("lo")))
        path = _resolve(raw)
        if path is None:
            problems.append(f"[{n}] 锚点文件不存在: {raw}")
            continue
        lines = _lines(path) or []
        if not (1 <= lo <= hi <= len(lines)):
            problems.append(f"[{n}] 行号越界 {raw}:{lo}-{hi}（文件共 {len(lines)} 行）")
        token = sym.split("(")[0].split("（")[0].strip()
        if token:
            joined = "\n".join(lines)
            if re.search(rf"[{CB}]", token):
                if token not in joined and token.replace("条款[", "[").replace("]", "") not in joined:
                    problems.append(f"W:[{n}] 治理锚点 '{token}' 未在 {raw} 命中")
            elif not re.search(rf"(?<![0-9A-Za-z_]){re.escape(token)}(?![0-9A-Za-z_])", joined):
                problems.append(f"[{n}] 符号 {token!r} 不存在于 {raw}")
    for m in RE_PLAIN.finditer(hard):
        raw, lo = m.group("path"), int(m.group("lo"))
        hi = int(m.group("hi")) if m.group("hi") else lo
        if (raw, m.group("lo")) in seen_spans:
            continue
        path = _resolve(raw)
        if path is None:
            problems.append(f"[{n}] 证据文件不存在: {raw}")
            continue
        lines = _lines(path) or []
        if not (1 <= lo <= hi <= len(lines)):
            problems.append(f"[{n}] 证据行号越界 {raw}:{lo}-{hi}（共 {len(lines)} 行）")
    # 证据字段: 无行号的相对路径逐项存在性（叙述项跳过）
    for item in re.split(r"[;；]", fields.get("证据", "")):
        if RE_NARRATIVE.search(item):
            continue
        for m in RE_EVID_PATH.finditer(item):
            raw = m.group("path")
            if "://" in raw or raw.startswith(("http",)):
                continue
            if _resolve(raw) is None:
                problems.append(f"[{n}] 证据路径不存在: {raw}")
    # 手册行号引用: 硬字段 + 软字段都核（条款字段逐字含手册行号引用属合法）
    for src in (hard, soft):
        for m in RE_HB_SRC.finditer(src):
            lo = int(m.group("lo"))
            if not (1 <= lo <= len(hb_lines)):
                problems.append(f"[{n}] 手册行号越界: 第{lo}行（手册共 {len(hb_lines)} 行）")
            elif not hb_lines[lo - 1].startswith(f"**[{n}]**"):
                problems.append(f"[{n}] 手册第{lo}行并非本条款起始行: {hb_lines[lo-1][:30]!r}")
        for m in RE_HB_ANY.finditer(src):
            lo = int(m.group("lo"))
            hi = int(m.group("hi") or m.group("lo"))
            if not (1 <= lo <= hi <= len(hb_lines)):
                problems.append(f"[{n}] 手册区间越界: 第{lo}-{hi}行（手册共 {len(hb_lines)} 行）")
    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    report = Path(sys.argv[1])
    if not report.exists():
        print(f"报告不存在: {report}")
        return 2
    hb_lines = _lines(HANDBOOK) or []
    records: list[tuple[int, list[str]]] = []
    cur: list[str] | None = None
    cur_n = 0
    for line in report.read_text(encoding="utf-8").splitlines():
        m = RE_RECORD.match(line)
        if m:
            if cur is not None:
                records.append((cur_n, cur))
            cur_n = int(m.group(1))
            cur = [line]
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        records.append((cur_n, cur))
    if not records:
        print("未发现任何 `[n] ✅/❌ |` 记录行")
        return 1
    fails = warns = 0
    for n, block in records:
        text = "\n".join(block)
        for lab in ("条款:", "代码:", "落实:", "审查:", "验证:", "证据:"):
            if lab not in text:
                print(f"[{n}] 缺字段 {lab}")
                fails += 1
        for p in check_anchors(n, text, hb_lines):
            if p.startswith(f"[{n}] W:") or "W:" in p[:6]:
                print(p)
                warns += 1
            else:
                print(p)
                fails += 1
    print(f"记录 {len(records)} 条 | FAIL {fails} | WARN {warns}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
