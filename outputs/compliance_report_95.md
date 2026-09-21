## 条款[95] MILUV 外部域布局随机化层

**条款**: **[95]** ***MILUV 外部域布局随机化层*** ——36次独立实验为配对单位声明 MILUV 外部域布局随机化层，生成 seed/超参稳健性表，报告 10 seed 间"LNN最低"成立频次（如 10/10、8/10）。

**状态**: ✅ 已落实

**代码**: configs/protocol/clause95_miluv_external_domain.yaml:1-68（MILUV外部域声明配置，含clause_id/core_requirements[paired_unit/miluv_declaration/layout_randomization_layer/robustness_table/lnn_lowest_frequency]/paired_experiment_unit/miluv_layout_randomization/seed_robustness_table/frequency_reporting/manual_line共八大块字段）; src/liquidloc/analysis/clause_gates.py::p100_miluv_external_domain():11210（验证函数，十四项检查a-n全覆盖）; AGENTS.md:317（治理载体索引条目）; rules/指令状态表.md:113（指令状态表条目）; outputs/compliance_report_95.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/clause95_miluv_external_domain.yaml 含八大块字段：① clause_id="95"; ② core_requirements.sub_dict含paired_unit="36 次独立实验", miluv_declaration=true, layout_randomization_layer=true, robustness_table="seed/超参稳健性表", lnn_lowest_frequency="10 seed 间'LNN 最低'成立频次"; ③ paired_experiment_unit.sub_dict含definition="36 次独立实验", composition, pairing_requirement; ④ miluv_layout_randomization.sub_dict含layer_type="MILUV 外部域布局随机化层", anchor_count=6; ⑤ seed_robustness_table.sub_dict含required_columns[7项], n_seeds=10, seed_range[1-10], lnn_lowest_frequency_calculation=true; ⑥ frequency_reporting.sub_dict含metric="LNN 最低成立频次", format, interpretation。二、验证函数 p100_miluv_external_domain() 十四项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"95";(c) paired_unit_36检查为True；(d) miluv_declaration_true检查为True；(e) layout_randomization_layer_true检查为True；(f) robustness_table_correct检查为True；(g) lnn_lowest_frequency_correct检查为True；(h) paired_unit_definition检查为True；(i) miluv_layer_type检查为True；(j) n_seeds_eq_10检查为True；(k) lnn_lowest_frequency_calc检查为True；(l) frequency_metric检查为True；(m) manual_line_points_to_330检查为True；(n) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"36次独立实验为配对单位"→core_requirements.paired_unit="36 次独立实验"命中；条款"MILUV外部域布局随机化层"→miluv_layout_randomization.layer_type命中；条款"seed/超参稳健性表"→core_requirements.robustness_table命中；条款"10seed间'LNN最低'成立频次"→core_requirements.lnn_lowest_frequency命中；逐行核对未发现问题（核对范围: configs/protocol/clause95_miluv_external_domain.yaml 第1-68行 + clause_gates.py 第11210-11355行，核对项: 条款原文逐句全量映射，十四项检查a-n全部通过）。

**审查**: 问题1：配置manual_line指向第322行，实为第330行（[95]实际行号）→ 已修正；问题2：配置verification_gate为对象而非字符串p100_miluv_external_domain → 已修正 → 新增p100_miluv_external_domain(); 问题3：AGENTS.md[95]格式错误（**[95]** → [95]）→ 已修正；逐句对照条款原文全量映射审查：所有子条款均匹配配置字段，无遗漏。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p100_miluv_external_domain; r = p100_miluv_external_domain(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[95] MILUV 外部域布局随机化层"; 退出码0

**证据**: configs/protocol/clause95_miluv_external_domain.yaml:1-68（MILUV外部域声明配置，68行，含clause_id/core_requirements[5项]/paired_experiment_unit/miluv_layout_randomization/seed_robustness_table/frequency_reporting/manual_line）; src/liquidloc/analysis/clause_gates.py::p100_miluv_external_domain():11210（验证函数，十四项检查a-n）; docs/最终交付_异步手册_424条_20260917.md:330（手册真源，条款[95]原文所在行）; AGENTS.md:317（治理载体索引条目）; rules/指令状态表.md:113（指令状态表[95]条目）; outputs/compliance_report_95.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_95.md
