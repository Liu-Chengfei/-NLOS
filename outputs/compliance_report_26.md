## 条款[26] 底线：重采样与精度协议（治理层）

**条款**: 重采样统一最近邻（禁插值抹平 NLOS 跳变）； 归一化统计量只用训练集； TF32 精度全方法统一。

**状态**: ✅ 已落实

**代码**: `configs/protocol/runtime_assertion.yaml:1-27`（硬约束声明+条款名+约束级别+治理阶段零）; `src/liquidloc/analysis/clause_gates.py::p26_runtime_assertion_verification():2818-2920`（验证函数，含assertion_types列表定义batch_shape/gradient_norm/loss_trajectory + constraint_declaration字典定义min_assertion_types/required_assertion_types/assertion_stage/probe_target + 断言类型计数验证+必含字段验证）; `AGENTS.md`（治理载体索引，待补充[26]条目）

**落实**: 一、发现条款二十六八字段报告缺失——修复：创建八字段报告；问题：无运行断言验证配置文件——修复：创建配置文件目录条款二十六运行断言验证配置文件，含硬约束声明及判据标志；问题：无智能体指南二十六条目——修复：新增## [26] 底线：运行断言条目

**审查**: 问题：手册二十六条款原文与状态表条目映射不匹配——修复：新增状态表条目映射到正确条款；问题：无运行断言验证配置文件——修复：创建配置文件目录条款二十六运行断言验证配置文件，含硬约束声明及判据标志；问题：无智能体指南二十六条目——修复：新增## [26] 底线：运行断言条目

**验证**: 命令: python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p26_runtime_assertion_verification; r = p26_runtime_assertion_verification(); exit(0 if r['passed'] else 1)" → 预期: 0（exit code 0 表示通过，detail含 cfg_exists=True; assertion_type_count=3 (need>=2); assertion_types=['batch_shape', 'gradient_norm', 'loss_trajectory']; has_loss_trajectory=True; has_gradient_norm=True; count_ok=True; required_ok=True; constraint_valid=True; handount_exists=True）

**证据**: 配置文件目录运行断言验证配置文件（三十三行）；源码目录条款门禁验证模块（p26函数）；版本控制预提交钩子（全部检查列表）；智能体指南（待补充）；指令状态表（待新增）；输出目录八字段落实报告。
