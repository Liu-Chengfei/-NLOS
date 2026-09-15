#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MCP stdio server for 异步高NLOS 手册条款检索（离线自测，不依赖外部服务）。

协议（stdio）：
  - stdin: 逐行或拼接 JSON-RPC 2.0 请求（一行一个 JSON）。
  - stdout: 逐行 JSON-RPC 2.0 响应。
  - stderr: 诊断输出。

工具列表：
  search_manual(query: str) -> {
    matches: [{line_no, text}, ...],
    context: {before, after},
    total_hits
  }
  查询方式：条款号（如 "1"、"17"、"71"）或关键词（如 "清场"、"五铁律"、"钩子"）。
"""
import json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HAND = ROOT / "docs" / "异步实验_执行手册_终版v15_总纲条款版.md"
hand_lines = HAND.read_text(encoding="utf-8").splitlines()

# 轻量索引：条款号 -> [行号, ...]（条款行格式 = 行首 "**[N]**"）
_index: dict[str, list[int]] = {}
for i, ln in enumerate(hand_lines, 1):
    m = re.match(r"^\*\*\[(\d+)\]\*\*", ln)
    if m:
        _index.setdefault(m.group(1), []).append(i)


def search_manual(query: str, context: int = 2) -> dict:
    q = query.strip()
    hits = []
    if re.fullmatch(r"\d+", q):
        hits = _index.get(q, [])
    else:
        kw = q.lower()
        for i, ln in enumerate(hand_lines, 1):
            if kw in ln.lower():
                hits.append(i)
        # 去重 + 限制
        seen, uniq = set(), []
        for h in hits:
            if h not in seen:
                seen.add(h); uniq.append(h)
        hits = uniq[:50]

    results = []
    for ln in hits:
        lo = max(0, ln - 1 - context)
        hi = min(len(hand_lines), ln - 1 + context + 1)
        results.append({
            "line_no": ln,
            "text": hand_lines[ln - 1],
            "context_before": hand_lines[lo:ln - 1],
            "context_after": hand_lines[ln:hi],
        })
    return {"query": q, "total_hits": len(results), "matches": results}


TOOLS = [
    {
        "name": "search_manual",
        "description": "按条款号或关键词检索执行手册（764 条）原文，返回行号+前后文。",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string", "description": "条款号或关键词"}}, "required": ["query"]},
    }
]


def handle(msg: dict) -> dict | None:
    mid = msg.get("id")
    m = msg.get("method")
    if m == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "async-nlos-manual", "version": "0.1.0"}}}
    if m == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if m == "tools/call":
        args = msg.get("params", {}).get("arguments", {})
        name = msg.get("params", {}).get("name")
        if name != "search_manual":
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown tool: {name}"}}
        q = args.get("query", "")
        try:
            res = search_manual(q)
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False)}]}}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)}}
    # notifications / unknown
    return None


def main():
    sys.stderr.write("[mcp-server] async-nlos-manual starting, handbook=" + str(HAND) + "\n")
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except Exception as e:
            sys.stderr.write(f"[mcp-server] bad JSON: {e}\n")
            continue
        res = handle(msg)
        if res is not None:
            sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
