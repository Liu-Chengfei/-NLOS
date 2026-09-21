## 条款[32] 档位协议条款 · K 轴 —— 锚点配置

**条款**: 档位协议条款 · K 轴 —— 锚点配置（原 G 轴并入 K 轴。锚数全档固定 4。模型输入不含锚点坐标。）

**状态**: ✅ 已落实

**代码**: configs/protocol/axis_k_anchor.yaml:1-40（K轴锚点配置配置文件）; src/liquidloc/analysis/clause_gates.py::p32_axis_k_anchor_verification():17701-17820（验证函数，含K轴三个档位锚点布局/GDOP/几何特征的配置文件存在性检查+参数验证）; AGENTS.md:229（治理载体索引条目）; rules/指令状态表.md:52（指令状态表条目）; .git/hooks/pre-commit（预提交钩子调用点）

**落实**: 一、发现K轴锚点配置条款八字段报告缺失——修复：创建八字段报告；二、发现验证函数缺失——修复：创建p32_axis_k_anchor_verification函数；三、发现指令状态表无[32]条目——修复：新增[32]条目

**审查**: 问题：K轴锚点配置条款八字段报告缺失——修复：创建八字段报告；问题：验证函数缺失——修复：创建验证函数；问题：指令状态表无[32]条目——修复：新增[32]条目

**验证**: 命令: python -c "import sys; sys.path.insert(0, "src"); from liquidloc.analysis.clause_gates import p32_axis_k_anchor_verification; r = p32_axis_k_anchor_verification(); exit(0 if r["passed"] else 1)" → 预期: 0（exit code 0 表示通过，detail含 cfg_exists=True; mismatches=none; missing_gears=none; missing_fields=none; handount_exists=True）

**证据**: 配置文件目录K轴锚点配置配置文件（四十行）；源码目录条款门禁验证模块（p32函数）；版本控制预提交钩子；智能体指南（第229行）；指令状态表（第52行）；输出目录八字段落实报告。

---

**附件**: 完整八字段报告见 outputs/compliance_report_32.md
