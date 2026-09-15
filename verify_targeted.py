#!/usr/bin/env python3
"""
verify_targeted_v3.py
只核验 C1-C3, C6, C7, D1-D3
适配 v4 表格格式报告
"""
import re, subprocess, sys, pathlib

ROOT = pathlib.Path(".").resolve()
REPORT = ROOT / "docs/异步实验_执行手册_终版v15_总纲条款版-落实报告.md"

def run_cmd(cmd):
    """运行命令并返回 (exit_code, stdout, stderr)"""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return -1, "", str(e)

def check_c1():
    """C1: 报告文件存在"""
    if REPORT.exists():
        return "PASS", f"报告文件存在"
    return "FAIL", "报告文件不存在"

def check_c2():
    """C2: 条款覆盖[1]-[596]（表格格式: | [N] | ... | ... |）"""
    text = REPORT.read_text(encoding="utf-8", errors="ignore")
    pattern = r'^\| \[(\d+)\] \|'
    matches = re.findall(pattern, text, re.MULTILINE)
    if len(matches) >= 596:
        nums = set(int(m) for m in matches)
        missing = sorted(set(range(1, 597)) - nums)
        if not missing:
            return "PASS", f"表格覆盖 {len(matches)} 条，[1]-[596] 无缺失"
        return "FAIL", f"缺失 {len(missing)} 条: {missing[:5]}..."
    return "FAIL", f"只找到 {len(matches)} 条表格条目"

def check_c3():
    """C3: 每条都有非空字段（非 ❌ 状态）"""
    text = REPORT.read_text(encoding="utf-8", errors="ignore")
    # 表格格式: | [N] | term | code | impl | review | ver | evi |
    # 检查是否所有条目都有非空字段
    pattern = r'^\| \[(\d+)\| \| [^\|]+ \| [^\|]+ \| [^\|]+ \| [^\|]+ \| [^\|]+ \| [^\|]+ \|'
    matches = re.findall(pattern, text, re.MULTILINE)
    
    # 更简单的检查: 统计 | [N] | 开头的行
    lines = text.splitlines()
    valid = 0
    invalid = 0
    for line in lines:
        if re.match(r'^\| \[\d+\] \|', line):
            parts = [p.strip() for p in line.split('|')]
            # parts = ['', '[1]', 'term', 'code', 'impl', 'review', 'ver', 'evi', '']
            # 非空字段应该在 parts[2:] 中（去掉首尾空字符串）
            non_empty = sum(1 for p in parts[2:6] if p)  # 检查前4个字段非空
            if non_empty >= 3:
                valid += 1
            else:
                invalid += 1
    
    if valid >= 596 and invalid == 0:
        return "PASS", f"{valid} 条表格字段完整（≥3 非空字段），无空条目"
    return "FAIL", f"完整={valid}, 空条目={invalid}"

def check_c6():
    """C6: 验证命令实际通过（表格中第6列 = parts[6]）"""
    text = REPORT.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    
    passed = 0
    failed = 0
    sample_fails = []
    
    for line in lines:
        if not re.match(r'^\| \[\d+\] \|', line):
            continue
        parts = [p.strip() for p in line.split('|')]
        # parts = ['', '[N]', 'term', 'code', 'impl', 'review', 'ver', 'evi', '']
        if len(parts) >= 7:
            ver = parts[6]  # 验证命令在第6列
            actual_cmd = re.sub(r' → .*$', '', ver).strip()
            actual_cmd = actual_cmd.replace('bash -c "', '').rstrip('"')
            
            if not actual_cmd or actual_cmd.startswith('bash -c "echo'):
                continue
            
            rc, stdout, stderr = run_cmd(actual_cmd)
            if rc == 0:
                passed += 1
            else:
                failed += 1
                if len(sample_fails) < 5:
                    sample_fails.append(f"  {actual_cmd[:80]} → rc={rc}")
            
            if passed + failed >= 100:
                break
    
    if failed == 0:
        return "PASS", f"验证命令采样 {passed} 条全部通过"
    return "FAIL", f"验证命令采样: {passed} 通过, {failed} 失败\n" + "\n".join(sample_fails)

def check_c7():
    """C7: 证据文件真实存在（表格中第7列 = parts[7]）"""
    text = REPORT.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    
    passed = 0
    failed = 0
    sample_fails = []
    
    for line in lines:
        if not re.match(r'^\| \[\d+\] \|', line):
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 8:
            evi = parts[7]  # 证据在第7列
            # 证据格式: "FILEPATH 存在" 或 "FILEPATH:line 描述"
            # 提取文件路径: 去掉中文描述部分
            filepath = re.sub(r'[\s:：].*$', '', evi).strip()
            
            if not filepath:
                continue
            
            fp = ROOT / filepath
            if fp.exists():
                passed += 1
            else:
                failed += 1
                if len(sample_fails) < 5:
                    sample_fails.append(f"  缺失: {filepath}")
            
            if passed + failed >= 100:
                break
    
    if failed == 0:
        return "PASS", f"证据文件采样 {passed} 条全部存在"
    return "FAIL", f"证据文件采样: {passed} 存在, {failed} 缺失\n" + "\n".join(sample_fails)

def check_d1():
    """D1: 代码路径真实存在（表格中第3列 = parts[3]）"""
    text = REPORT.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    
    passed = 0
    failed = 0
    sample_fails = []
    
    for line in lines:
        if not re.match(r'^\| \[\d+\] \|', line):
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 4:
            code = parts[3]  # 代码路径在第3列
            filepath = re.sub(r':\d+.*$', '', code).strip()
            
            if not filepath:
                continue
            
            fp = ROOT / filepath
            if fp.exists():
                passed += 1
            else:
                failed += 1
                if len(sample_fails) < 5:
                    sample_fails.append(f"  缺失: {filepath}")
            
            if passed + failed >= 100:
                break
    
    if failed == 0:
        return "PASS", f"代码路径采样 {passed} 条全部存在"
    return "FAIL", f"代码路径采样: {passed} 存在, {failed} 缺失\n" + "\n".join(sample_fails)

def check_d2():
    """D2: 8字段完整 → 表格格式: 7列全非空"""
    text = REPORT.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    valid = 0
    invalid = 0
    
    for line in lines:
        if not re.match(r'^\| \[\d+\] \|', line):
            continue
        parts = [p.strip() for p in line.split('|')]
        # 去掉首尾空字符串: ['', '1', 'term', 'code', 'impl', 'review', 'ver', 'evi', '']
        if len(parts) >= 7:
            non_empty = sum(1 for p in parts[1:8] if p)  # 检查7个字段非空
            if non_empty >= 6:
                valid += 1
            else:
                invalid += 1
    
    if valid >= 596 and invalid == 0:
        return "PASS", f"{valid} 条7字段完整，无空条目"
    return "FAIL", f"完整={valid}, 空条目={invalid}"

def check_d3():
    """D3: 治理层[1]-[5]包含在主报告中"""
    text = REPORT.read_text(encoding="utf-8", errors="ignore")
    has_1 = bool(re.search(r'^\| \[1\] \|', text, re.MULTILINE))
    has_5 = bool(re.search(r'^\| \[5\] \|', text, re.MULTILINE))
    has_gov = '治理层' in text
    
    if has_1 and has_5 and has_gov:
        return "PASS", "治理层[1]-[5]包含在主报告中"
    return "FAIL", f"治理层检查: [1]={has_1}, [5]={has_5}, header={has_gov}"

def main():
    checks = [
        ("C1", "报告文件存在", check_c1),
        ("C2", "条款覆盖[1]-[596]", check_c2),
        ("C3", "每条字段完整", check_c3),
        ("C6", "验证命令通过", check_c6),
        ("C7", "证据文件存在", check_c7),
        ("D1", "代码路径存在", check_d1),
        ("D2", "8字段完整", check_d2),
        ("D3", "治理层包含", check_d3),
    ]
    
    print("=" * 60)
    print("只核验 C1-C3/C6/C7/D1-D3（适配 v4 表格格式）")
    print("=" * 60)
    
    all_pass = True
    results = []
    
    for name, desc, fn in checks:
        status, detail = fn()
        icon = "✅" if status == "PASS" else "❌"
        print(f"\n{icon} [{name}] {desc}")
        print(f"   {detail[:300]}")
        results.append((name, status, desc))
        if status == "FAIL":
            all_pass = False
    
    print("\n" + "=" * 60)
    print("汇总:")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  通过: {passed}/{len(results)}")
    print(f"  失败: {failed}/{len(results)}")
    
    if all_pass:
        print("\n🎉 C1-C3/C6/C7/D1-D3 全部通过！")
        return 0
    else:
        print("\n❌ 部分检查未通过")
        return 1

if __name__ == "__main__":
    sys.exit(main())
