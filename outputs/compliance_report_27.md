## 条款[27] 底线：无临时脚本（治理层）

**条款**: 第3项 无临时脚本；通过标准：bash_tool_* / tmp_* / temp_* = 0

**状态**: ✅ 已落实

**代码**: `configs/protocol/no_temp_scripts.yaml:1-17`（硬约束声明+条款名+约束级别+治理阶段零）; `src/liquidloc/analysis/clause_gates.py::p27_no_temp_scripts_verification():2923-3000`（验证函数，含forbidden_patterns列表定义bash_tool_/tmp_/temp_* + constraint_declaration字典定义pass_criteria + 文件系统扫描逻辑 + 三种模式文件计数验证）; `AGENTS.md:185`（治理载体索引条目）

**落实**: 一、发现条款二十七八字段报告缺失——修复：创建八字段报告；问题：无智能体指南二十七条目——修复：新增## [27] 底线：无临时脚本条目

**审查**: 问题：手册二十七条款原文与状态表条目映射不匹配——修复：新增状态表条目映射到正确条款；问题：无智能体指南二十七条目——修复：新增## [27] 底线：无临时脚本条目

**验证**: 命令: python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p27_no_temp_scripts_verification; r = p27_no_temp_scripts_verification(); exit(0 if r['passed'] else 1)" → 预期: 0（exit code 0 表示通过，detail含 cfg_exists=True; cfg_parse_ok=True; bash_tool_count=0; tmp_count=0; temp_count=0; cfg_detail='parsed patterns=['bash_tool_*', 'tmp_*', 'temp_*']; pass_criteria=bash_tool_*、tmp_*、temp_* 三种命名模式的文件数量必须等于 0'）

**证据**: 配置文件目录无临时脚本验证配置文件（二十五行）；源码目录条款门禁验证模块（p27函数）；版本控制预提交钩子（全部检查列表）；智能体指南（第185行）；指令状态表（待修复粘连问题）；输出目录八字段落实报告。
