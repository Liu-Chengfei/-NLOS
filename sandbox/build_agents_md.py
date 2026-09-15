# -*- coding: utf-8 -*-
"""AGENTS.md 生成器（规则层，条款 [1] 治理载体之一）。

治理载体分布（消除 [3] 多拷贝问题：手册总纲每一行仅保留一份逐字拷贝）：
- 本文件（规则层）：五铁律 + 语言规则 + 手册第 6-15 行 + 第 121-166 行逐字；
- .agents/skills/execution-report-gate/SKILL.md（工序层）：手册第 17-119 行逐字（§1）+ 细化扩展；
- .zcode/config.json 钩子（注入/护栏层）；.mcp.json（查询层）；docs/decision_log.md（交接层）。
"""
from pathlib import Path

HAND = Path("docs/异步实验_执行手册_终版v15_总纲条款版.md")
lines = HAND.read_text(encoding="utf-8").splitlines()
seg_a = "\n".join(lines[5:15])    # 手册第 6-15 行：## 0 标题、三阶段制、0 清场
seg_b = "\n".join(lines[120:166])  # 手册第 121-166 行：归属总则/阶段执行地图/手册目标/底线/治理层条款[1]-[5]

content = """> 本 AGENTS.md 为「异步高NLOS实验」专属执行规则入口（条款 [1] 治理载体，规则层，精简版）。
> 手册路径（唯一真源）：`docs/异步实验_执行手册_终版v15_总纲条款版.md`（764 条，7 阶段）。

## 治理载体分布（总纲全部内容由载体集合逐字承载，每行仅一份生效拷贝，与条款 [3] 一致）
- 规则层（本文件）：五铁律 + 语言规则 + 手册第 6–15 行（三阶段制、0 清场）+ 手册第 121–166 行（归属总则、阶段执行地图、手册目标、底线、治理层条款 [1]–[5]）逐字；
- 工序层 `.agents/skills/execution-report-gate/SKILL.md`：写/审/核落实报告时自动触发（触发词：落实报告、八字段、核验、验收、第 N 阶段）；§1 逐字收录手册第 17–119 行（落实报告格式规范），§2 细化扩展 + gate 检查项 + 常见错误 + 核查要点；唯一逐字载体。
- 注入层 `.zcode/config.json` SessionStart 钩：会话启动/清空/压缩时自动注入五铁律 + 当前阶段磁盘状态；
- 护栏层 `.zcode/config.json` PreToolUse 钩：第 0/1 阶段验收前拦截实验代码运行类命令；
- 查询层 `.mcp.json` async-nlos-manual 服务器：按需检索手册 764 条原文（落实/复核条款时替代通读全文）；
- 交接层 `docs/decision_log.md`：跨批交接 decision log 唯一载体。

## 五铁律（本实验唯一规则入口，逐字遵循；条文取自手册治理类落实示例）
1. 唯一真源 = 本实验《执行手册》（764 条），其他提示词不覆盖手册任何条款；
2. 状态只认磁盘：进度/结果/是否已做只认磁盘上的 manifest、decision log、产物、时间戳、git commit；
3. 原文执行：每一条条款逐字照做，不摘要、不改写、不缩写、不遗漏；
4. 不做解释性改编：不替手册条款做「合理外推」，不按记忆判断进度；
5. 一步一验失败即停：任一步骤核验不通过或不确定 → 停下等待用户批准，不进入下一步。

## 语言规则
所有回答、解释、汇报与书写一律**简体中文**（专有名词、变量名、代码标识符等除外）。

""" + seg_a + """

### 落实报告格式（指针：内容不在本文件，逐字载体唯一在工序层 skill）
落实报告八字段逐字段规范、一句话模板、批量提交前自检、完整示例、判定铁律、"落实:"字段要素说明、逐条落实工序 = 手册第 17–119 行，已逐字收录于 `.agents/skills/execution-report-gate/SKILL.md` §1（约 116 行）。触发时机：写任何落实报告（第 0/1 阶段）前必读 SKILL.md §1；§2 为细化扩展，含 gate 检查项、常见错误、核查要点。

""" + seg_b + """

> 注：本文件内两处逐字段（「## 0 · 执行总纲」起至「0 清场」末 = 手册第 6–15 行；「### 条款-三阶段归属总则」起至治理层条款 [5] 末 = 手册第 121–166 行）与 SKILL.md 的 §1（手册第 17–119 行）合并 = 手册总纲第 6–166 行全量，每行仅一份生效拷贝；完整性由 sandbox/build_agents_md.py 内置逐行 diff 断言验证。
"""

Path("AGENTS.md").write_text(content, encoding="utf-8")
ag = content.splitlines()


def verify(seg_lines, start1, end1, label):
    want = lines[start1 - 1:end1]
    idx = next(i for i, l in enumerate(ag) if l == want[0])
    got = ag[idx:idx + len(want)]
    bad = [k for k in range(len(want)) if got[k] != want[k]]
    print(f"{label}: 手册第{start1}-{end1}行 → AGENTS.md 第{idx+1}-{idx+len(want)}行, 不一致 {len(bad)}")
    assert not bad, label


verify(None, 6, 15, "段A")
verify(None, 121, 166, "段B")
print("AGENTS.md 总行数:", len(ag))
