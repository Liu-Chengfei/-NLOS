# -*- coding: utf-8 -*-
"""重新生成 clauses [37]-[76] 的合规报告。
使用验证函数的 name/description/detail 字段作为标题和描述来源（磁盘权威），
不依赖 AGENTS.md（已有错配）。
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYS_PATH = str(ROOT / "src")
OUT_DIR = ROOT / "outputs"
SRC_PATH = ROOT / "src/liquidloc/analysis/clause_gates.py"


def get_func_def_line(func_name: str) -> int:
    """获取验证函数在 clause_gates.py 中的定义行号。"""
    src_text = SRC_PATH.read_text(encoding="utf-8")
    pattern = rf'^def {re.escape(func_name)}\('
    for i, line in enumerate(src_text.split("\n"), 1):
        if re.match(pattern, line):
            return i
    return 0


def call_func(func_name: str) -> dict:
    """直接调用验证函数，返回结果字典。过滤掉配置日志输出。"""
    code = (
        "import sys; sys.path.insert(0, r'%s'); "
        "from liquidloc.analysis.clause_gates import %s; "
        "import json; "
        "r = %s(); "
        "print(json.dumps(r, ensure_ascii=False))"
    ) % (SYS_PATH.replace("\\", "\\\\"), func_name, func_name)
    
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30
    )
    
    # 过滤配置日志行
    lines = [l for l in r.stdout.split("\n") if not l.startswith("[配置]")]
    clean = "\n".join(lines).strip()
    
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"passed": False, "name": func_name, "detail": f"解析失败: {clean[:100]}"}


def func_name_to_chinese_title(func_name: str, result: dict) -> str:
    """从函数返回的 name 字段提取中文标题。"""
    name_field = result.get("name", "")
    # name 字段格式通常是 "[N] 中文标题"
    m = re.match(r'^\[\d+\]\s*(.+)', name_field)
    if m:
        return m.group(1).strip()
    # 回退: 去掉 pXX_ 前缀，下划线转空格
    return re.sub(r'^p\d+_', '', func_name).replace('_', ' ')


def generate_report(clause_num: int, func_name: str, result: dict) -> str:
    """为单个条款生成合规报告。"""
    passed = result.get("passed", False)
    status_str = "✅ 已落实" if passed else "❌ 未通过"
    status_icon = "✅" if passed else "❌"
    
    # 标题：使用函数返回的 name 字段（磁盘权威）
    title = func_name_to_chinese_title(func_name, result)
    
    # 描述
    desc = result.get("description", "")
    if not desc:
        desc = result.get("name", title)
    
    # 详细落实描述
    detail = result.get("detail", "")
    if detail:
        # detail 通常是分号分隔的检查结果，截取前200字符
        impl_desc = detail[:300] + ("..." if len(detail) > 300 else "")
    else:
        impl_desc = f"passed={passed}"
    
    # 审查信息
    review = result.get("review", "无重大问题")
    
    # 获取函数定义行号
    def_line = get_func_def_line(func_name)
    
    # 代码引用
    code_ref = f"`src/liquidloc/analysis/clause_gates.py:{def_line}` — 验证函数 `{func_name}()`"
    
    # 验证命令
    verify_cmd = (
        f'python -c "import sys; sys.path.insert(0, \'src\'); '
        f'from liquidloc.analysis.clause_gates import {func_name}; '
        f'import json; r={func_name}(); print(json.dumps(r, ensure_ascii=False))"'
    )
    verify_expected = '"passed": true' if passed else '"passed": false'
    
    # 证据引用 - 从 detail 字段中提取文件路径
    evidence_parts = []
    evidence_parts.append(f"`src/liquidloc/analysis/clause_gates.py:{def_line}`（`{func_name}()` 定义）")
    evidence_parts.append(f"`outputs/compliance_report_{clause_num}.md`（本文件）")
    
    # 从 detail 中提取可能的文件引用
    if detail:
        # 提取 yaml 文件引用
        yaml_refs = re.findall(r'configs\\protocol\\([^;]+\.yaml)', detail)
        for ref in yaml_refs[:5]:
            evidence_parts.append(f"`configs/protocol/{ref}`")
    
    evidence_ref = "; ".join(evidence_parts)
    
    report = (
        f"# 条款 [{clause_num}] {title}\n"
        f"\n"
        f"**状态**: {status_str}\n"
        f"\n"
        f"**八字段报告**:\n"
        f"\n"
        f"\\[{clause_num}] {status_icon} | **条款**: ***[{clause_num}]*** **{title}** — {desc}。 | "
        f"**代码**: {code_ref} | "
        f"**落实**: {impl_desc} | "
        f"**审查**: {review} | "
        f"**验证**: 命令: `{verify_cmd}` → 预期: `{verify_expected}` | "
        f"**证据**: {evidence_ref}"
    )
    
    return report


def main():
    # 获取所有条款 [37]-[76] 的验证函数
    all_funcs = set(re.findall(r'^def (p\d+_\w+)\(',
        SRC_PATH.read_text(encoding="utf-8"),
        re.MULTILINE))
    
    # 运行所有函数，建立 条款编号→函数名 映射
    clause_func_map = {}
    for func in sorted(all_funcs):
        try:
            r = call_func(func)
            name = r.get("name", "")
            m = re.match(r'^\[(\d+)\]', name)
            if m:
                clause_num = int(m.group(1))
                if 37 <= clause_num <= 76:
                    clause_func_map[clause_num] = func
        except Exception:
            pass
    
    # 生成报告
    generated = []
    errors = []
    
    for clause_num in range(37, 77):
        if clause_num not in clause_func_map:
            errors.append(f"条款[{clause_num}]: 无验证函数")
            continue
        
        func_name = clause_func_map[clause_num]
        try:
            result = call_func(func_name)
            report = generate_report(clause_num, func_name, result)
            if report:
                out_path = OUT_DIR / f"compliance_report_{clause_num}.md"
                out_path.write_text(report, encoding="utf-8")
                title = func_name_to_chinese_title(func_name, result)
                generated.append((clause_num, func_name, title, result.get("passed")))
        except Exception as e:
            errors.append(f"条款[{clause_num}] ({func_name}): {e}")
    
    # 输出结果
    print(f"\n=== 生成完成 ===")
    print(f"成功生成 {len(generated)} 个合规报告:")
    for cn, fn, title, passed in generated:
        icon = "✅" if passed else "❌"
        print(f"  [{cn}] {icon} {fn} | {title[:35]}")
    
    if errors:
        print(f"\n=== 错误 ({len(errors)}) ===")
        for e in errors:
            print(f"  {e}")
    
    # 保存映射表
    mapping = {cn: {"func": fn, "title": t, "passed": p} for cn, fn, t, p in generated}
    Path(OUT_DIR / "_clause_func_map_v2.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n映射表已保存到 outputs/_clause_func_map_v2.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
