## 条款[35] 无临时脚本

**条款**: 无临时脚本；通过标准：bash_tool_*/tmp_*/temp_* = 0

**状态**: ✅ 已落实

**代码**: src/liquidloc/analysis/clause_gates.py::p35_no_temp_scripts_verification():18051-18120（验证函数，扫描工作区所有bash_tool_*、tmp_*、temp_*前缀文件，通过标准为数量=0）; AGENTS.md:264（治理载体索引条目）; rules/指令状态表.md:55（指令状态表条目）; .git/hooks/pre-commit（预提交钩子调用点）; outputs/compliance_report_35.md（八字段报告）

**落实**: 一、发现无临时脚本条款八字段报告缺失——修复：创建八字段报告；二、发现验证函数缺失——修复：创建p35_no_temp_scripts_verification函数，扫描工作区三类临时脚本前缀文件；三、发现指令状态表无[35]条目——修复：新增[35]条目

**审查**: 问题：无临时脚本条款八字段报告缺失——修复：创建八字段报告；问题：验证函数缺失——修复：创建验证函数；问题：指令状态表无[35]条目——修复：新增[35]条目

**验证**: 命令: python -c "import sys; sys.path.insert(0, \"src\"); from liquidloc.analysis.clause_gates import p35_no_temp_scripts_verification; r = p35_no_temp_scripts_verification(); exit(0 if r[\"passed\"] else 1)" → 预期: 0（exit code 0 表示通过，detail含 total_temp_files=0; bash_tool_count=0; tmp_count=0; temp_count=0）

**证据**: 源码目录条款门禁验证模块（p35函数，扫描工作区临时脚本）；版本控制预提交钩子；智能体指南（第264行）；指令状态表（第55行）；输出目录八字段落实报告。

---

**附件**: 完整八字段报告见 outputs/compliance_report_35.md
