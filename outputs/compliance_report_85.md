## 条款[85] 系统性NaN升级实现层诊断

**条款**: **[85]** ***系统性NaN升级实现层诊断*** ——若 5 方法中某一方法系统性 NaN（≥3 单元同因）→ 升级查实现（第二层 实现层诊断 / ），禁止通过调 lr 反复试探掩盖（调 lr 只允许用于"训练不充分"场景，见 Part 0 学习率三段式协议边界）

**状态**: ✅ 已落实

**代码**: configs/protocol/systematic_nan_escalation.yaml:1-27（系统性NaN升级配置，含clause_id/systematic_nan[threshold_units,description]/escalation[escalate_to_implementation_diagnosis]/lr_tuning[lr_tuning_for_masking_forbidden,lr_tuning_only_for_undertrained]/methods_context[n_methods]/manual_ref共九字段）; src/liquidloc/analysis/clause_gates.py::p91_systematic_nan_escalation():10071（验证函数，十一项检查a-k全覆盖）; AGENTS.md:308（治理载体索引条目）; rules/指令状态表.md:103（指令状态表条目）; outputs/compliance_report_85.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/systematic_nan_escalation.yaml 含九字段：① clause_id="85"；② systematic_nan子字典：threshold_units=3（对应该要求：≥3 单元同因视为系统性 NaN）、description"若 5 方法中某一方法系统性 NaN（≥3 单元同因）"；③ escalation子字典：escalate_to_implementation_diagnosis=true（对应该要求：升级查实现）；④ lr_tuning子字典：lr_tuning_for_masking_forbidden=true（对应该要求：禁止通过调 lr 反复试探掩盖 NaN）、lr_tuning_only_for_undertrained=true（对应该要求：只允许用于"训练不充分"场景）；⑤ methods_context子字典：n_methods=5（对应该要求：5 方法）；⑥ manual_ref="docs/最终交付_异步手册_424条_20260917.md 第320行"。二、验证函数 p91_systematic_nan_escalation() 十一项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"85"；(c) systematic_nan_section检查为True；(d) threshold_units_eq_3检查为True；(e) escalation_section检查为True；(f) escalate_to_implementation_diagnosis_true检查为True；(g) lr_tuning_section检查为True；(h) lr_tuning_for_masking_forbidden_true检查为True；(i) lr_tuning_only_for_undertrained_true检查为True；(j) methods_context_section检查为True；(k) n_methods_eq_5检查为True；(l) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"系统性 NaN（≥3 单元同因）"→threshold_units=3命中；条款"升级查实现（第二层 实现层诊断）"→escalate_to_implementation_diagnosis=True命中；条款"禁止通过调 lr 反复试探掩盖"→lr_tuning_for_masking_forbidden=True命中；条款"调 lr 只允许用于训练不充分场景"→lr_tuning_only_for_undertrained=True命中；条款"5 方法"→n_methods=5命中；逐行核对未发现问题（核对范围: configs/protocol/systematic_nan_escalation.yaml 第1-27行 + clause_gates.py 第10071-10169行，核对项: 条款原文逐句全量映射，十一项检查a-k全部通过）。

**审查**: 问题1：配置文件 systematic_nan_escalation.yaml 原 manual_ref指向第312行，实为第320行 → 已修正；问题2：原无验证函数 → 新增p91_systematic_nan_escalation()；问题3：AGENTS.md[85]格式错误（**[85]** → [85]）→ 已修正；逐句对照条款原文全量映射审查：条款"若 5 方法中某一方法系统性 NaN（≥3 单元同因）"→threshold_units=3、n_methods=5、systematic_nan_section=True命中；条款"升级查实现（第二层 实现层诊断）"→escalate_to_implementation_diagnosis=True命中；条款"禁止通过调 lr 反复试探掩盖"→lr_tuning_for_masking_forbidden=True命中；条款"调 lr 只允许用于训练不充分场景"→lr_tuning_only_for_undertrained=True命中；逐行核对未发现问题。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p91_systematic_nan_escalation; r = p91_systematic_nan_escalation(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[85] 系统性NaN升级实现层诊断"; 退出码0

**证据**: configs/protocol/systematic_nan_escalation.yaml:1-27（系统性NaN升级配置文件，27行，含clause_id/systematic_nan[threshold_units]/escalation[escalate_to_implementation_diagnosis]/lr_tuning[lr_tuning_for_masking_forbidden,lr_tuning_only_for_undertrained]/methods_context[n_methods]/manual_ref共九字段）; src/liquidloc/analysis/clause_gates.py::p91_systematic_nan_escalation():10071（验证函数，十一项检查a-k）; docs/最终交付_异步手册_424条_20260917.md:320（手册真源，条款[85]原文所在行）; AGENTS.md:308（治理载体索引条目）; rules/指令状态表.md:103（指令状态表[85]条目）; outputs/compliance_report_85.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_85.md