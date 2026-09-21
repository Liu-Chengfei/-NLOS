## 条款[28] 治理载体索引模板（分散落点，禁止集中单文件）

**条款**: 治理载体索引模板——执行者第一次开工、或发现治理文件缺失/过期时，按本模板建出/更新

**状态**: ✅ 已落实

**代码**: （治理载体索引条目）; （治理载体条目）; （验证函数，含AGENTS.md存在性检查 + 五铁律+索引条目验证 + 分散分布检查）

**落实**: 一、发现治理载体索引模板条目缺失——修复：在AGENTS.md第186行添加[28]治理载体索引模板条目；二、发现指令状态表无[28]条目——修复：新增状态表条目

**审查**: 问题：治理载体索引模板条目缺失——修复：在AGENTS.md第186行添加[28]治理载体索引模板条目

**验证**: 命令: python -c "import sys; sys.path.insert(0, "src"); from liquidloc.analysis.clause_gates import p28_governance_carrier_index_verification; r = p28_governance_carrier_index_verification(); exit(0 if r["passed"] else 1)" → 预期: 0（exit code 0 表示通过，detail含 agents_exists=True; has_five_rules=True; has_index_entries=True; rules_exist=True; skills_exist=True; hooks_exist=True; zcode_exist=True; checklist_count=N）

**证据**: 智能体指南（第186行）；指令状态表（第48行）；规则文件目录；技能文件目录；hooks目录；.zcode目录；八字段落实报告。

---

**附件**: 完整八字段报告见 [outputs/compliance_report_28.md](/E:/异步高NLOS/outputs/compliance_report_28.md)
