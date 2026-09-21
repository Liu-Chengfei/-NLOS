## 条款[30] 档位协议条款 · N 轴 —— NLOS 偏差

**条款**: 档位协议条款 · N 轴 —— NLOS 偏差（纯偏差定义。全档固定 4 锚、可解率 100%。不可解帧归 M 轴。）

**状态**: ✅ 已落实

**代码**: configs/protocol/axis_n_nlos.yaml:1-60（N轴NLOS偏差配置文件）; src/liquidloc/analysis/clause_gates.py::p30_axis_n_nlos_verification():17432-17570（验证函数，含N轴四个档位NLOS占比/偏差/噪声的配置文件存在性检查+参数验证）; AGENTS.md:188（治理载体索引条目）; rules/指令状态表.md:50（指令状态表条目）; .git/hooks/pre-commit（预提交钩子调用点）

**落实**: 一、发现N轴NLOS偏差条款八字段报告缺失——修复：创建八字段报告；二、发现配置文件缺失——修复：创建configs/protocol/axis_n_nlos.yaml；三、发现验证函数缺失——修复：创建p30_axis_n_nlos_verification函数；四、发现AGENTS.md无[30]条目——修复：新增[30]治理载体条目；五、发现指令状态表无[30]条目——修复：新增[30]条目

**审查**: 问题：N轴NLOS偏差条款八字段报告缺失——修复：创建八字段报告；问题：配置文件缺失——修复：创建配置文件；问题：验证函数缺失——修复：创建验证函数

**验证**: 命令: python -c "import sys; sys.path.insert(0, "src"); from liquidloc.analysis.clause_gates import p30_axis_n_nlos_verification; r = p30_axis_n_nlos_verification(); exit(0 if r["passed"] else 1)" → 预期: 0（exit code 0 表示通过，detail含 cfg_exists=True; mismatches=none; missing_gears=none; missing_fields=none; handount_exists=True）

**证据**: 配置文件目录N轴NLOS偏差配置文件（六十行）；源码目录条款门禁验证模块（p30函数）；版本控制预提交钩子；智能体指南（第188行）；指令状态表（第50行）；输出目录八字段落实报告。

---

**附件**: 完整八字段报告见 outputs/compliance_report_30.md
