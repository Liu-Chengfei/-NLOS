## 条款[33] 档位协议条款 · M 轴 —— 数据缺失

**条款**: 档位协议条款 · M 轴 —— 数据缺失（描述"到没到"——不可解帧/无测距值的缺失注入。）

**状态**: ✅ 已落实

**代码**: configs/protocol/axis_m_missing.yaml:1-65（M轴数据缺失配置文件）; src/liquidloc/analysis/clause_gates.py::p33_axis_m_missing_verification():17836-17930（验证函数，含M轴四个档位UWB/VIO/IMU缺失率的配置文件存在性检查+参数验证）; AGENTS.md:239（治理载体索引条目）; rules/指令状态表.md:53（指令状态表条目）; .git/hooks/pre-commit（预提交钩子调用点）

**落实**: 一、发现M轴数据缺失条款八字段报告缺失——修复：创建八字段报告；二、发现配置结构不匹配——修复：修正p33函数适配M轴扁平配置结构；三、发现指令状态表无[33]条目——修复：新增[33]条目

**审查**: 问题：M轴数据缺失条款八字段报告缺失——修复：创建八字段报告；问题：配置结构不匹配——修复：修正expected_gears匹配配置文件字段；问题：指令状态表无[33]条目——修复：新增[33]条目

**验证**: 命令: python -c "import sys; sys.path.insert(0, \"src\"); from liquidloc.analysis.clause_gates import p33_axis_m_missing_verification; r = p33_axis_m_missing_verification(); exit(0 if r[\"passed\"] else 1)" → 预期: 0（exit code 0 表示通过，detail含 cfg_exists=True; mismatches=none; missing_gears=none; missing_fields=none; handount_exists=True）

**证据**: 配置文件目录M轴数据缺失配置文件（六十五行）；源码目录条款门禁验证模块（p33函数）；版本控制预提交钩子；智能体指南（第239行）；指令状态表（第53行）；输出目录八字段落实报告。

---

**附件**: 完整八字段报告见 outputs/compliance_report_33.md
