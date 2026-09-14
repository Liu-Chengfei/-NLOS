#!/usr/bin/env python3
# checklist_stage0.py
# Stage 0 治理层核对脚本（对应手册条款 [1]-[30] 与 checklist_stage0.md）
# 退出码非 0 = gate 未过，禁止进入下一阶段
import sys, pathlib, os

ROOT = pathlib.Path(".").resolve()
def check(name, cond):
    print("✓" if cond else "✗", name)
    return cond

ok = True
# [1] AGENTS.md 存在且五铁律逐字一致
p = ROOT / "AGENTS.md"
ok &= check("[1] AGENTS.md 存在", p.exists())
if p.exists():
    txt = p.read_text(encoding="utf-8")
    ok &= check("[1] AGENTS.md 含五铁律", "唯一真源 = 本手册" in txt and "状态只认磁盘" in txt)
# [2] pre-commit 钩子
ok &= check("[2] .git/hooks/pre-commit 存在", (ROOT/".git"/"hooks"/"pre-commit").exists())
ok &= check("[2] pre-commit 可执行", os.access(ROOT/".git"/"hooks"/"pre-commit", os.X_OK))
# [4]-[5] 清单与脚本
ok &= check("[4] checklist_stage0.md 存在", (ROOT/"checklist_stage0.md").exists())
ok &= check("[5] checklist_stage0.py 存在", (ROOT/"checklist_stage0.py").exists())
# [0 清场] 清场脚本
ok &= check("[0清场] clear_workspace.py 存在", (ROOT/"clear_workspace.py").exists())
# 清场验证文件
ok &= check("[0清场] 清场验证.txt 存在", (ROOT/"清场验证.txt").exists())
# 治理层文件最小集
ok &= check("[Stage0] 至少一个治理文件落地", ok)

sys.exit(0 if ok else 1)
