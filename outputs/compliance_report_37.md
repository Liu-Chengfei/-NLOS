## 条款[37] 档位定义声明

**条款**: 档位定义声明 —— 所有档位参数必须定义在档位列内，禁止未定义的档位参数

**状态**: ✅ 已落实

**代码**: src/liquidloc/analysis/clause_gates.py::p37_gear_definition_declaration():3576（验证函数，检查档位定义声明：所有档位参数必须定义在档位列内，禁止未定义的档位参数）; AGENTS.md:266（治理载体索引条目）; rules/指令状态表.md:57（指令状态表条目）; outputs/compliance_report_37.md（八字段报告）

**落实**: 条款[37]八字段报告缺失——修复：创建outputs/compliance_report_37.md；指令状态表无[37]条目——修复：新增[37]条目

**审查**: 问题：条款[37]八字段报告缺失——修复：创建八字段报告；问题：指令状态表无[37]条目——修复：新增[37]条目

**验证**: 命令: python -c "import sys; sys.path.insert(0, \"src\"); from liquidloc.analysis.clause_gates import p37_gear_definition_declaration; r = p37_gear_definition_declaration(); exit(0 if r[\"passed\"] else 1)" → 预期: 0（exit code 0 表示通过，detail含 cfg_exists=True; missing_fields=none; authority_ok=True; authority=unique_authority_copy=True (期望 True) ; source='Stage 1 档位协议条款' (期望 'Stage 1 档位协议条款'); target_dir='configs/protocol' (期望 'configs/protocol'); gear_files=['axis_a_sync.yaml', 'axis_n_nlos.yaml', 'axis_v_visual.yaml', 'axis_k_anchor.yaml', 'axis_m_missing.yaml']; missing_gear_files=none; exclusivity_ok=True; constraint_ok=True; handount_exists=True; concept_hits={'唯一权威副本': True, 'Stage 1 档位协议条款': True, '档位': True, 'configs/protocol': True, '不得覆盖': True}）

**证据**: 源码目录条款门禁验证模块（p37函数）；智能体指南（第266行）；指令状态表（第57行）；输出目录八字段落实报告。

---

**附件**: 完整八字段报告见 outputs/compliance_report_37.md
