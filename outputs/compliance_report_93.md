## 条款[93] 基线一致

**条款**: **[93]** ***基线一致*** ——消融各变体与主实验同数据/同 seed/同超参/同评估口径。

**状态**: ✅ 已落实

**代码**: configs/protocol/clause93_baseline_consistency.yaml:1-26（基线一致配置，含clause_id/baseline_consistency_requirements[same_data/same_seed/same_hyperparams/same_eval_standard/ablation_variable_only]/checks[4项]共八大块字段）; src/liquidloc/analysis/clause_gates.py::p98_baseline_consistency():10985（验证函数，十项检查a-j全覆盖）; AGENTS.md:316（治理载体索引条目）; rules/指令状态表.md:111（指令状态表条目）; outputs/compliance_report_93.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/clause93_baseline_consistency.yaml 含八大块字段：① clause_id="93"; ② baseline_consistency_requirements子字典含same_data=true, same_seed=true, same_hyperparams=true, same_eval_standard=true, ablation_variable_only=true（对应该要求：同数据/同seed/同超参/同评估口径，唯一差异为被消融组件）; ③ checks数组含4项：data_manifest_consistency, seed_initialization_match, hyperparam_table_identical, eval_metric_definition_same（对应该要求：四个验证检查项）; ④ manual_line=328（修正原第320行错误）。二、验证函数 p98_baseline_consistency() 十项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"93"；(c) same_data_true检查为True；(d) same_seed_true检查为True；(e) same_hyperparams_true检查为True；(f) same_eval_standard_true检查为True；(g) ablation_variable_only_true检查为True；(h) checks_4_required检查4项检查项全部存在；(i) manual_line_points_to_328检查为True；(j) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"消融各变体与主实验同数据"→baseline_consistency_requirements.same_data=true命中；条款"同seed"→same_seed=true命中；条款"同超参"→same_hyperparams=true命中；条款"同评估口径"→same_eval_standard=true命中；条款"唯一差异应为被消融的组件"→ablation_variable_only=true命中；逐行核对未发现问题（核对范围: configs/protocol/clause93_baseline_consistency.yaml 第1-26行 + clause_gates.py 第10985-11102行，核对项: 条款原文逐句全量映射，十项检查a-j全部通过）。

**审查**: 问题1：配置manual_line指向第320行，实为第328行（[93]实际行号）→ 已修正；问题2：原无验证函数 → 新增p98_baseline_consistency(); 问题3：AGENTS.md[93]格式错误（**[93]** → [93]）→ 已修正；逐句对照条款原文全量映射审查：所有子条款均匹配配置字段，无遗漏。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p98_baseline_consistency; r = p98_baseline_consistency(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[93] 基线一致"; 退出码0

**证据**: configs/protocol/clause93_baseline_consistency.yaml:1-26（基线一致配置文件，26行，含clause_id/baseline_consistency_requirements/5个一致性约束/checks[4项验证]/manual_line）; src/liquidloc/analysis/clause_gates.py::p98_baseline_consistency():10985（验证函数，十项检查a-j）; docs/最终交付_异步手册_424条_20260917.md:328（手册真源，条款[93]原文所在行）; AGENTS.md:316（治理载体索引条目）; rules/指令状态表.md:111（指令状态表[93]条目）; outputs/compliance_report_93.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_93.md
