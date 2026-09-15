# ⑱ 决策日志（Decision Log）

> 条款 [2] 要求的跨批交接载体；条款 [3] 要求的单一生效载体。
> 历史轮次：00_governance_check / 01_pre_self_check 为 git 已跟踪文件，原样保留；02/03 已归并至本文件。

---

## 03 · Stage0 治理载体分布重构（AGENTS.md 减重 → skill/钩子/MCP 分担）

- **时间**：2026-09-15 16:40（中国标准时间）
- **触发**：用户验收意见「AGENTS.md 内容太多了，叫 skill、rule、mcp、插件、钩子分担一些」
- **改动**：
  1. `AGENTS.md` 由 174 行重建为 81 行（sandbox/build_agents_md.py 生成）：删除与格式层整体重复的手册第 17–119 行格式块（103 行），改为指针（AGENTS.md:31-32）；保留五铁律（:13-18）+ 段A 三阶段制/0清场（手册 6-15 行逐字，:20-29）+ 段B 归属/地图/目标/底线/治理条款[1]-[5]（手册 121-166 行逐字，:34-79）+ 载体分布表（:4-11）。
  2. 新增工序层 `.agents/skills/execution-report-gate/SKILL.md`（写/审/核落实报告触发，指向格式层唯一逐字载体；§1 逐字收录手册第 17–119 行，§2 细化扩展）。
  3. 新增注入层+护栏层 `.zcode/config.json`（hooks.enabled=true）+ `.zcode/hooks/session_start.py`（SessionStart startup|resume|clear|compact 注入五铁律+阶段磁盘状态）+ `.zcode/hooks/pre_tool_use.py`（第 0/1 阶段验收前拦截实验运行类 Bash 命令；verify/grep/清场/sandbox 放行）。
  4. 新增查询层 `.mcp.json` + `.mcp/servers/manual_server.py`（stdio JSON-RPC，search_manual 按条款号/关键词检索手册 764 条，含前后文）。
  5. `checklist_gate.md` [1] 项更新为载体集合口径。
- **原因**：手册 [3]「禁止为同一逻辑留多份拷贝」——旧 AGENTS.md 与模板各存一份格式块 103 行重复；[1] 要求总纲全部内容由治理载体承载，载体为集合（AGENTS+skill+钩子+MCP+清单），逐句落点由分布表保证，每行仅一份生效拷贝。
- **插件层决定**：不使用——内容已由项目级 skill/钩子/MCP 承载，再打包为插件会产生第二份拷贝（违反 [3]），仅当需要跨仓库复用时才走插件化。
- **影响范围**：仅治理层文件；未动实验代码（src/、tests/、configs/）、未动手册、未跑实验。
- **验证**：三段 diff 0 不一致；钩子离线自测（拦截 exit 2/放行 exit 0/注入 additionalContext JSON 合法）；MCP JSON-RPC 往返自测（initialize/tools/list/search 命中）；verify_slim_v4_阶段感知.sh stage0 重跑总判定 PASS（输出 logs/verify_stage0_20260915.log 覆盖更新）。
- **下一步**：停下等用户验收；批准前不进第 1 阶段。

---

## 04 · Stage0 治理载体分布重构 v2（模板/checklist 非治理层 → skill 承载；rule 缺中文条款 → AGENTS.md 补）

- **时间**：2026-09-15（中国标准时间）
- **触发**：用户验收意见
  1. 「落实报告格式模板文件、checklist_gate.md 是格式/清单层，不应算入治理层」
  2. 「AGENTS.md 的语言铁律只写了英文，没有写中文」
  3. 「skill 太薄，需要吸收格式模板 + checklist_gate 全部内容」
- **处置**：
  1. 格式层 `docs/落实报告格式模板.md` 已并入 `.agents/skills/execution-report-gate/SKILL.md` §1（手册第 17–119 行逐字），`docs/落实报告格式模板.md` 删除——内容由 skill 唯一承载，符合 [3]。
  2. `checklist_gate.md` 已删除，gate 检查项并入 SKILL.md §2.3；勾留痕机制由 docs/decision_log.md 承担。
  3. AGENTS.md 顶部新增独立「## 语言规则」段——「所有回答、解释、汇报与书写一律简体中文（专有名词/变量名等除外）」；载体分布表同步更新。
- **原因**：用户验收意见要求治理层仅含规则层（AGENTS.md）+ 注入护栏层（.zcode 钩子）+ 查询层（.mcp）；格式/清单内容由 skill 承载；语言铁律需明确中文条款。
- **影响范围**：仅治理层文件；未动实验代码（src/、tests/、configs/）、未动手册、未跑实验。
- **验证**：SKILL.md §1 逐字断言 102/102 通过；AGENTS.md 段A/段B diff 0 不一致；钩子离线自测通过；MCP JSON-RPC 往返自测通过。
- **下一步**：停下等用户验收；批准前不进第 1 阶段。

---

## 02 · Stage0 治理文件落地

- **时间**：2026-09-15 15:59（中国标准时间）
- **改动**：新增 AGENTS.md / docs/落实报告格式模板.md / checklist_gate.md / clear_workspace.py / 清场验证.txt / logs/clear_workspace_第0阶段.log / 本 decision log；第 0 阶段落实报告待写入
- **原因**：条款 [1] 要求第 0 阶段固化治理层条款载体；条款 [2] 要求跨批交接写 decision log；条款 [1]「0 清场」要求每实验开跑前执行
- **影响范围**：仓库根 + docs/ + decision_logs/；未改动实验代码（src/、tests/、configs/ 未动）；未改手册本身
- **git 状态（当前未提交）**：见上一段改动清单（共 6 文件/日志改动）
- **核验状态**：待跑 verify_slim_v4_阶段感知.sh → 全绿才允许进入第 1 阶段（条款 [4]）
- **下一步**：由用户验收后决定是否 commit；批准前不动任何实验代码

---
## 05 · 批1.1 逐条落实条款 [6]–[18] + 并行等效抽查模块 + SKILL.md §2.3 gate 项

- **时间**：2026-09-15（中国标准时间）
- **触发**：条款 [17] 要求每批逐条拆解复核；条款 [14] 要求并行等效抽查模块；条款 [18] 要求 AGENTS.md 模板可用
- **改动**：
  1. 新增 `src/liquidloc/common/parallel_equivalence_check.py`（条款 [14] 平行等效抽查三函数：verify_parallel_equivalence / record_equivalence_decision / check_equivalence_record_exists，共 275 行）
  2. 注册 Pre-7 gate → `src/liquidloc/common/precheck_orchestrator.py::check_Pre7_parallel_equivalence_record():549-572`（条款 [14] 等效抽查留痕检查）
  3. 新增 `.agents/skills/execution-report-gate/SKILL.md` §2.3 gate 项（条款 [17] 拆解复核五元组标准 + 子项不可合并 + 第二人重跑）
  4. 新增 `tests/common/test_parallel_equivalence_check.py`（13 单测，逐bit/数值/边界全覆盖）
  5. 更新 `tests/common/test_precheck_orchestrator.py`（test_full_clean_run 断言 66→67，因 Pre-7 注册入 orchestrator）
  6. 更新 `clear_workspace.py`（条款 [13] 违反即处置）
  7. 更新 `AGENTS.md`（批1.1 条款 [6]-[18] 载体映射）
  8. 生成 `docs/异步实验_执行手册_终版v15_总纲条款版-第1阶段_批1.1_落实报告.md`（13 条逐字段规范报告）
- **原因**：条款 [6]-[18] 须在 Stage1 前落实；条款 [14] 并行安全须有等效抽查模块；条款 [17] 拆解复核须有 gate 项；条款 [13] 违反即处置须有清场脚本
- **影响范围**：src/、tests/、.agents/、docs/、AGENTS.md、clear_workspace.py；未跑实验、未改手册本身
- **验证**：126 单测全 PASS（含 13 新增 parallel_equivalence_check 单测）；verify_slim_v4_阶段感知.sh stage1 待跑
- **下一步**：由用户验收报告格式后 commit；批准后进 Stage2

---
- **下一步**：由用户验收后决定是否 commit；批准前不动任何实验代码
