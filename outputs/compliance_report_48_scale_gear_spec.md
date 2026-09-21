## 条款[48] 规模档位规格

**条款**: **[48]** ***规模档位***：入门 N_layout=6、 推荐 N_layout=10（full + 2 个最强基线 LSTM/EKF × 2–3 seed 即可，泛化层只判排序不做因子推断，不需要主场景同等功效）

**状态**: ✅ 已落实

**代码**: configs/protocol/scale_gear_spec.yaml:1-20; src/liquidloc/analysis/clause_gates.py::p48_scale_gear_spec():5544; AGENTS.md:276; rules/指令状态表.md:67; outputs/compliance_report_48_scale_gear_spec.md

**落实**:
1. 发现条款[48]八字段报告缺失 → 修复：创建 outputs/compliance_report_48_scale_gear_spec.md
2. 发现指令状态表无[48]条目 → 修复：新增[48]条目
3. 配置文件 configs/protocol/scale_gear_spec.yaml 已存在，含 clause_id=48
4. 验证函数 p48_scale_gear_spec() 已实现，校验规模档位配置

**审查**: 问题：条款[48]八字段报告缺失 → 修复：创建完整八字段报告

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p48_scale_gear_spec; r = p48_scale_gear_spec(); print(f'passed={r['passed'], name={r['name']}'); exit(0 if r['passed'] else 1)"` → 预期: passed=True；name="[48] 规模档位：入门 N_layout=6、 推荐 N_layout=10（full + 2 个最强基线 LSTM/EKF × 2–3 seed 即可，泛化层只判排序不做因子推断，不需要主场景同等功效）"；退出码0

**证据**: configs/protocol/scale_gear_spec.yaml; src/liquidloc/analysis/clause_gates.py::p48_scale_gear_spec(); AGENTS.md:276; rules/指令状态表.md:67; outputs/compliance_report_48_scale_gear_spec.md
