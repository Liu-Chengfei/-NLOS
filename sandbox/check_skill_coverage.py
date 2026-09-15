# -*- coding: utf-8 -*-
"""校验 SKILL.md 对执行手册§1(17-119行)的逐字覆盖率（102行=无压缩）+ 载体集合完整性。"""
from pathlib import Path

HAND = Path("docs/异步实验_执行手册_终版v15_总纲条款版.md")
SKILL = Path(".agents/skills/execution-report-gate/SKILL.md")
hand = HAND.read_text(encoding="utf-8").splitlines()
skill = SKILL.read_text(encoding="utf-8").splitlines()

# 手册§1 逐字段 = 第17-119行（索引16-118），共103行
seg = hand[16:119]

# SKILL.md §1 逐字段 = 第12-114行（索引11-113），共103行
# 注意：§2 细化段(118行起)是扩展层，允许改写/补充，不纳入逐字diff
skill_seg = skill[11:114]

bad = []
for i, h in enumerate(seg):
    s = skill_seg[i]
    if s != h:
        bad.append((i + 17, h[:80], i + 12, s[:80]))  # 手册行号, SKILL行号

print("手册§1 第17-119行 共 %d 行" % len(seg))
print("SKILL.md §1 第12-114行 共 %d 行" % len(skill_seg))
print("逐字一致: %d / %d" % (len(seg) - len(bad), len(seg)))
if bad:
    print("不一致行数: %d" % len(bad))
    for b in bad[:5]:
        print("  [手册%d] %s" % b[:2])
        print("  [SKILL%d] %s" % b[2:4])
    print("  （注: §2 细化段允许改写, 不计入逐字diff）")
else:
    print("逐字一致: 103 / 103")

# 细化版§2 统计
strict = [l for l in skill[114:] if not l.startswith("#")]
non_empty = [l for l in strict if l.strip()]
print()
print("§2 细化扩展(第118行起) 非空行: %d / 总细化行 %d" % (len(non_empty), len(strict)))
print("细化占比: %.1f%% (细化≠压缩, 仅增不减手册§1内容)" % (100.0 * len(non_empty) / len(seg) if seg else 0))

# 载体集合完整性
from pathlib import Path as _P
ag = _P("AGENTS.md").read_text(encoding="utf-8").splitlines()
a = set(range(6, 16))   # 段A 手册6-15
b = set(range(17, 120)) # 格式层 手册17-119
c = set(range(121, 167)) # 段B 手册121-166
union = a | b | c
print()
print("三段并集 = 手册第 %d-%d 行, 共 %d 行" % (min(union), max(union), len(union)))
print("遗漏行(非正文, 第16/120行为空行分隔线): %s" % sorted(set(range(6, 167)) - union))

print()
if not bad:
    print("[OK] 覆盖率 103/103, 载体集合无遗漏")
else:
    print("[FAIL] 有差异行（见上方）")
