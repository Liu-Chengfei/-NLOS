## 条款[47] 三层证据结构总表

**条款**: **[47]** ***三层证据结构总表（论文级数据架构，与 论文级基线 轨迹规格衔接）***： 层场景/数据规模职责统计推断与主场景共享**主场景（受控）**主场景固定布局（20m×20m、4锚、4遮挡区）10seed×5方法=50单元

**状态**: ✅ 已落实

**代码**: configs/protocol/three_layer_evidence_structure.yaml:1-30; src/liquidloc/analysis/clause_gates.py::p47_three_layer_evidence():5316; AGENTS.md:275; rules/指令状态表.md:66; outputs/compliance_report_47_three_layer_evidence.md

**落实**:
1. 发现条款[47]八字段报告缺失 → 修复：创建 outputs/compliance_report_47_three_layer_evidence.md
2. 发现指令状态表无[47]条目 → 修复：新增[47]条目
3. 配置文件 configs/protocol/three_layer_evidence_structure.yaml 已存在，含 clause_id=47
4. 验证函数 p47_three_layer_evidence() 已实现，校验三层结构定义

**审查**: 问题：条款[47]八字段报告缺失 → 修复：创建完整八字段报告

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p47_three_layer_evidence; r = p47_three_layer_evidence(); print(f'passed={r['passed'], name={r['name']}'); exit(0 if r['passed'] else 1)"` → 预期: passed=True；name="[47] 三层证据结构总表（论文级数据架构，与 S3 轨迹规格衔接）：层场景/数据规模职责统计推断与主场景共享 主场景（受控）主场景固定布局（20m×20m、4锚、4遮挡区）10seed×5方法=50单元"；退出码0

**证据**: configs/protocol/three_layer_evidence_structure.yaml; src/liquidloc/analysis/clause_gates.py::p47_three_layer_evidence(); AGENTS.md:275; rules/指令状态表.md:66; outputs/compliance_report_47_three_layer_evidence.md
