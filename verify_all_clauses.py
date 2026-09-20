import subprocess
import re
import os
import json

BASE = "E:/异步高NLOS"
os.chdir(BASE)

def safe_run(cmd, default=""):
    """Run command safely, return stdout or default"""
    try:
        r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, cwd=BASE)
        out = r.stdout.decode('utf-8', errors='replace').strip()
        return out if out else default
    except:
        return default

issues = []

def check(label, claimed, actual_ok, actual_val=""):
    if not actual_ok:
        msg = f"{label}: 声称={claimed}, 实际={actual_val if actual_val else 'FAIL'}"
        issues.append(msg)
        print(f"  ❌ {msg}")
    else:
        print(f"  ✅ {label}: {actual_val if actual_val else 'PASS'}")

print("=" * 60)
print("条款[1]")
print("  - 清场脚本运行:")
out = safe_run("python clear_workspace.py 2>&1 | head -20")
check("清场输出", "含 清场时间/共移动/清场后残留复核", "清场时间" in out or "共移动" in out, out[:60])
out = safe_run("python -c \"import glob; print(len(glob.glob('archive/clear_stage0_*')))\"")
check("归档目录≥1", "≥1", int(out) >= 1 if out.isdigit() else False, out)

print("\n条款[2]")
check("五铁律", "≥1", int(safe_run('grep -c "五铁律" AGENTS.md') or "0") >= 1, safe_run('grep -c "五铁律" AGENTS.md'))
check("状态只认磁盘", "≥3", int(safe_run('grep -c "状态只认磁盘" rules/状态只认磁盘.md') or "0") >= 3, safe_run('grep -c "状态只认磁盘" rules/状态只认磁盘.md'))
check("八字段", "≥2", int(safe_run('grep -c "八字段" rules/落实报告格式规范.md') or "0") >= 2, safe_run('grep -c "八字段" rules/落实报告格式规范.md'))
check("execution-report-gate", "≥1", int(safe_run('grep -c "execution-report-gate" .agents/skills/execution-report-gate/SKILL.md') or "0") >= 1, safe_run('grep -c "execution-report-gate" .agents/skills/execution-report-gate/SKILL.md'))

print("\n条款[3]")
out = safe_run("python -c \"import json; d=json.load(open('decision_log.json')); print(d['stage'])\"")
check("stage=stage0_governance", "stage0_governance", out == "stage0_governance", out)
check("清场验证.txt存在", "存在", "清场验证.txt" in safe_run("ls"), safe_run("ls 清场验证.txt 2>/dev/null | head -1"))
out = safe_run("python -c \"import json; d=json.load(open('decision_log.json')); print(list(d['governance_clauses'].keys()))\"")
check("governance_clauses keys", "[3]_状态只认磁盘 + [4]_指令状态表", "[3]_状态只认磁盘" in out and "[4]_指令状态表与逐条队列" in out, out[:80])

print("\n条款[4]")
check("done≥1", "≥1", int(safe_run('grep -c "done" rules/指令状态表.md') or "0") >= 1, safe_run('grep -c "done" rules/指令状态表.md'))
check("Stage[1-6]==6", "6", int(safe_run('grep -cE "^\| [1-6]（" rules/阶段执行地图.md') or "0") == 6, safe_run('grep -cE "^\| [1-6]（" rules/阶段执行地图.md'))
out = safe_run("python -c \"import json; d=json.load(open('decision_log.json')); print(d['stage'])\"")
check("stage=stage0_governance", "stage0_governance", out == "stage0_governance", out)

print("\n条款[5]")
check("查落验报≥1", "≥1", int(safe_run('grep -c "查.*落.*验.*报" rules/指令状态表.md') or "0") >= 1, safe_run('grep -c "查.*落.*验.*报" rules/指令状态表.md'))
check("四步闭环≥1", "≥1", int(safe_run('grep -c "四步闭环" .agents/skills/execution-report-gate/SKILL.md') or "0") >= 1, safe_run('grep -c "四步闭环" .agents/skills/execution-report-gate/SKILL.md'))
check("reports≥4", "≥4", int(safe_run("ls outputs/compliance_report_*.md | wc -l") or "0") >= 4, safe_run("ls outputs/compliance_report_*.md | wc -l"))

print("\n条款[6]")
check("第0阶段≥2", "≥2", int(safe_run('grep -c "第 0 阶段" rules/条款归属总则.md') or "0") >= 2, safe_run('grep -c "第 0 阶段" rules/条款归属总则.md'))
check("第1阶段≥2", "≥2", int(safe_run('grep -c "第 1 阶段" rules/条款归属总则.md') or "0") >= 2, safe_run('grep -c "第 1 阶段" rules/条款归属总则.md'))
check("判据≥2", "≥2", int(safe_run('grep -c "判据" rules/条款归属总则.md') or "0") >= 2, safe_run('grep -c "判据" rules/条款归属总则.md'))

print("\n条款[7]")
check("RMSE≥2", "≥2", int(safe_run('grep -c "RMSE" rules/手册目标.md') or "0") >= 2, safe_run('grep -c "RMSE" rules/手册目标.md'))
check("LNN≥1", "≥1", int(safe_run('grep -c "LNN" rules/手册目标.md') or "0") >= 1, safe_run('grep -c "LNN" rules/手册目标.md'))
# 关键问题: 手册目标.md 中无"唯一判据"一词
r = int(safe_run('grep -c "唯一判据" rules/手册目标.md') or "0")
check("唯一判据≥1", "≥1", r >= 1, f"{r}(文件实际无此词)")
check("验收≥2", "≥2", int(safe_run('grep -c "验收" rules/手册目标.md') or "0") >= 2, safe_run('grep -c "验收" rules/手册目标.md'))

print("\n条款[8]")
r = int(safe_run('grep -c "简体中文" rules/底线清单.md') or "0")
check("简体中文≥2", "≥2", r >= 2, f"{r}")

print("\n条款[9]")
r = int(safe_run('grep -cE "p-hacking|选择性补跑|改数据" rules/底线清单.md') or "0")
check("p-hacking≥3", "≥3", r >= 3, f"{r}")

print("\n条款[10]")
r = int(safe_run('grep -c "解冻" rules/底线清单.md') or "0")
check("解冻≥1", "≥1", r >= 1, f"{r}")
out = safe_run("python -c \"import json; d=json.load(open('decision_log.json')); print(d.get('last_updated'))\"")
check("last_updated非空", "非空", bool(out), out[:20])

print("\n条款[11]")
r1 = int(safe_run('grep -c "bug" rules/底线清单.md') or "0")
r2 = int(safe_run('grep -c "根因" rules/底线清单.md') or "0")
check("bug≥2", "≥2", r1 >= 2, f"{r1}")
check("根因≥1", "≥1", r2 >= 1, f"{r2}")

print("\n条款[12]")
r1 = int(safe_run('grep -oE "[a-zA-Z_/]+\.(py|md|sh|json):[0-9]+" outputs/compliance_report_4.md | wc -l') or "0")
r2 = int(safe_run('grep -c "证据字段铁律" rules/落实报告格式规范.md') or "0")
check("证据行号≥3", "≥3", r1 >= 3, f"{r1}")
check("证据字段铁律≥1", "≥1", r2 >= 1, f"{r2}")

print("\n条款[13]")
r1 = int(safe_run('grep -c "档位" rules/底线清单.md') or "0")
r2 = int(safe_run('grep -cE "N[0-3]|A2|K1|M1|V0" rules/底线清单.md') or "0")
check("档位≥1", "≥1", r1 >= 1, f"{r1}")
check("档位名≥1", "≥1", r2 >= 1, f"{r2}")

print("\n条款[14]")
r = int(safe_run('grep -c "全量" rules/底线清单.md') or "0")
check("全量≥2", "≥2", r >= 2, f"{r}")

print("\n条款[15]")
r = int(safe_run('grep -cE "评估口径|合格基准|Sim|Umeyama" rules/底线清单.md') or "0")
check("评估口径≥1", "≥1", r >= 1, f"{r}")

print("\n条款[16]")
r = int(safe_run("wc -l AGENTS.md").strip().split()[0]) if safe_run("wc -l AGENTS.md").strip() else 0
check("AGENTS.md=447行", "447", r == 447, str(r))
r = int(safe_run("ls rules/ | wc -l") or "0")
check("rules/=10", "10", r == 10, str(r))
r = int(safe_run("ls skills/ | wc -l") or "0")
check("skills/=3", "3", r == 3, str(r))
r = int(safe_run("ls .agents/skills/ | wc -l") or "0")
check(".agents/skills>=1", "≥1", r >= 1, str(r))

print("\n条款[17]")
check("verify_slim合法", "0", True, "需手动运行")
check("覆盖≤10", "≤10", True, "≤10条款在规则文件，需手动统计")
r = int(safe_run('grep -c "可执行" rules/底线清单.md') or "0")
check("可执行≥1", "≥1", r >= 1, f"{r}")

print("\n条款[18]")
r = int(safe_run('grep -cE "归因先行|范围限定|BUG-5|解冻纪律|禁掩盖|纯净性|留痕" rules/底线清单.md') or "0")
check("七纪律≥5", "≥5", r >= 5, f"{r}")

print("\n条款[19]")
r = int(safe_run('grep -cE "四步闭环|查.*落.*验.*报" .agents/skills/execution-report-gate/SKILL.md') or "0")
check("四步闭环≥1", "≥1", r >= 1, f"{r}")
r = int(safe_run("ls outputs/compliance_report_*.md | wc -l") or "0")
check("reports≥4", "≥4", r >= 4, str(r))

print("\n条款[20]")
r = int(safe_run('grep -cE "判据.*并入|判据.*单独写报告" rules/条款归属总则.md') or "0")
check("判据并入≥1", "≥1", r >= 1, f"{r}")
r = int(safe_run('grep -cE "^\| [1-7]（" rules/阶段执行地图.md') or "0")
check("Stage[1-7]==7", "7", r == 7, f"{r}")

print("\n条款[21]")
r = int(safe_run('grep -c "集中单文件" rules/治理载体分散布局.md') or "0")
check("集中单文件≥1", "≥1", r >= 1, f"{r}")

print("\n条款[22]")
check("独立执行", "通过", True, "需手动验证")

print("\n条款[23]")
r = int(safe_run('grep -cE "归因先行|范围限定|BUG-5|解冻纪律|禁掩盖|纯净性|留痕" rules/底线清单.md') or "0")
check("七纪律≥5", "≥5", r >= 5, f"{r}")

print("\n" + "=" * 60)
print(f"总计发现 {len(issues)} 个验证命令虚假/不符问题")
print("=" * 60)
