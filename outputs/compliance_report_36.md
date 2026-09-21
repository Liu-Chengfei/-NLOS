## 条款[36] git 是最后防线

**条款**: git 是最后防线 —— git 仓库与工作区实际内容必须一致（禁止"改了文件没 commit 就进下一 gate"），任何版本回退/文件损坏/误改恢复一律走 git（checkout/revert），不复制 _backup/_old 副本（与冗余清理备份禁令一致）

**状态**: ✅ 已落实

**代码**: configs/protocol/git_is_last_resort.yaml:1-80（git最后防线配置文件）; src/liquidloc/analysis/clause_gates.py::p36_git_is_last_resort():813-926（验证函数，含约束配置校验+pre-commit钩子检查+.gitignore防护规则）; AGENTS.md:265（治理载体索引条目）; rules/指令状态表.md:56（指令状态表条目）; .git/hooks/pre-commit（预提交钩子调用点）

**落实**: 一、发现条款八字段报告格式需更新——修复：重写八字段报告；二、发现指令状态表无[36]条目——修复：新增[36]条目

**审查**: 问题：指令状态表无[36]条目——修复：新增[36]条目

**验证**: 命令: python -c "import sys; sys.path.insert(0, \"src\"); from liquidloc.analysis.clause_gates import p36_git_is_last_resort; r = p36_git_is_last_resort(); exit(0 if r[\"passed\"] else 1)" → 预期: 0（exit code 0 表示通过，detail含 backup_prohibition=OK; git_recovery_mandatory=OK; working_tree_synchronization=OK; pre_commit_hook=OK）

**证据**: 配置文件目录git最后防线配置文件（八十字）；源码目录条款门禁验证模块（p36函数）；版本控制预提交钩子；智能体指南（第265行）；指令状态表（第56行）；输出目录八字段落实报告。

---

**附件**: 完整八字段报告见 outputs/compliance_report_36.md
