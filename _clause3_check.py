from pathlib import Path
import subprocess, json, time, re

base = Path("E:/异步高NLOS")
results = []
results.append("="*70)
results.append("条款[3]「状态只认磁盘」磁盘锚点逐项核验")
results.append("="*70)
results.append("")

# 锚点1: 清场验证.txt
results.append("【锚点1】清场验证.txt")
p1 = base / "清场验证.txt"
if p1.exists():
    text1 = p1.read_text(encoding="utf-8")
    lines1 = text1.split("\n")
    results.append(f"  ✅ 存在: 行数={len(lines1)}, 字节={p1.stat().st_size}")
    results.append(f"  最近修改时间: {time.ctime(p1.stat().st_mtime)}")
    has_content = any("清场" in l or "clear" in l.lower() or "✅" in l for l in lines1)
    results.append(f"  含验证标记: {'✅' if has_content else '❌'}")
    for l in lines1[-3:]:
        results.append(f"  ...{l}")
else:
    results.append("  ❌ 不存在")
results.append("")

# 锚点2: manifests/
results.append("【锚点2】manifests/ 目录")
p2 = base / "manifests"
if p2.exists() and p2.is_dir():
    files2 = sorted(p2.glob("*"))
    results.append(f"  ✅ 存在: 文件/目录数={len(files2)}")
    for f in files2[:10]:
        results.append(f"    {f.name}")
    if len(files2) > 10:
        results.append(f"    ...（共{len(files2)}项）")
else:
    results.append("  ❌ 不存在或非目录")
results.append("")

# 锚点3: decision_log
results.append("【锚点3】decision_log 文件")
dl_json = base / "decision_log.json"
dl_stage = sorted(base.glob("decision_log_*.md"))
dl_hidden = sorted(base.glob(".decision_logs/*.json"))

if dl_json.exists():
    results.append(f"  ✅ decision_log.json 存在")
else:
    results.append(f"  ❌ decision_log.json 不存在")
for p in dl_stage[:5]:
    results.append(f"  ✅ decision_log_{p.name} 存在")
if len(dl_stage) > 5:
    results.append(f"  ...（共{len(dl_stage)}个stage文件）")
for p in dl_hidden[:5]:
    results.append(f"  ✅ .decision_logs/{p.name} 存在")
if len(dl_hidden) > 5:
    results.append(f"  ...（共{len(dl_hidden)}个hidden文件）")
results.append("")

# 锚点4: compliance_report
results.append("【锚点4】outputs/compliance_report_*.md")
p4 = base / "outputs"
if p4.exists():
    reports = sorted(p4.glob("compliance_report_*.md"))
    results.append(f"  ✅ 存在: {len(reports)} 个报告")
    nums = [int(re.match(r"compliance_report_(\d+)", r.name).group(1)) for r in reports if re.match(r"compliance_report_(\d+)", r.name)]
    if nums:
        results.append(f"  覆盖条款编号: {min(nums)}~{max(nums)}")
        results.append(f"  [37]-[76]范围报告数: {sum(1 for n in nums if 37 <= n <= 76)} 个")
else:
    results.append("  ❌ 不存在")
results.append("")

# 锚点5: .git/HEAD
results.append("【锚点5】.git/HEAD → 有效 commit")
head_file = base / ".git" / "HEAD"
if head_file.exists():
    head_content = head_file.read_text(encoding="utf-8").strip()
    results.append(f"  HEAD内容: {head_content}")
    if head_content.startswith("ref: "):
        ref_path = base / ".git" / head_content[5:].strip()
        if ref_path.exists():
            commit_hash = ref_path.read_text(encoding="utf-8").strip()
            results.append(f"  ref指向: {ref_path.relative_to(base)}")
            results.append(f"  commit hash: {commit_hash[:12]}")
            try:
                result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(base))
                dirty = result.stdout.strip()
                if dirty:
                    results.append(f"  ⚠️ git工作区不干净:")
                    for line in dirty.split("\n")[:10]:
                        results.append(f"    {line}")
                else:
                    results.append(f"  ✅ git工作区干净")
            except Exception as e:
                results.append(f"  ⚠️ git status失败: {e}")
        else:
            results.append(f"  ❌ ref文件不存在: {ref_path}")
    else:
        results.append(f"  ⚠️ HEAD格式非标准ref")
else:
    results.append(f"  ❌ .git/HEAD不存在")
results.append("")

# 规则文件核验
results.append("【附加核验】rules/状态只认磁盘.md 内容与条款[3]原文一致性")
rule_content = (base / "rules/状态只认磁盘.md").read_text(encoding="utf-8")
checks = [
    ("manifest提及", "manifest" in rule_content.lower()),
    ("decision log提及", "decision log" in rule_content.lower()),
    ("产物提及", "产物" in rule_content),
    ("git commit提及", "git commit" in rule_content.lower() or "commit" in rule_content.lower()),
    ("铁律我记得跑过", "我记得跑过" in rule_content),
    ("跨批交接纪律", "跨批交接" in rule_content),
]
for name, ok in checks:
    results.append(f"  {'✅' if ok else '❌'} {name}")
results.append("")

# 合规报告[3]
results.append("【合规报告】compliance_report_3.md")
r3 = base / "outputs" / "compliance_report_3.md"
if r3.exists():
    content = r3.read_text(encoding="utf-8")
    results.append(f"  ✅ 存在: 大小={r3.stat().st_size}字节, 行数={len(content.split(chr(10)))}")
    for field in ["**状态**:", "**条款**:", "**代码**:", "**落实**:", "**审查**:", "**验证**:", "**证据"]:
        results.append(f"  {'✅' if field in content else '❌'} {field}")
    # 验证函数
    m = re.search(r"(p3_[a-z_][a-z0-9_]*)", content)
    if m:
        func_name = m.group(1)
        cg = (base / "src/liquidloc/analysis/clause_gates.py").read_text(encoding="utf-8")
        results.append(f"  验证函数 {func_name}: {'✅ 存在' if func_name in cg else '❌ 不存在'}")
    results.append("")
    results.append("  报告内容:")
    results.append(content)
else:
    results.append("  ❌ 不存在")

with open("E:/异步高NLOS/_clause3_check.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))
print("DONE")
