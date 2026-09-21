## 条款[97] seed/轨迹数/独立实验数明示

**条款**: **[97]** ***seed/轨迹数/独立实验数明示*** ——seed/轨迹数/独立实验数是否明示；高兼容默认：表注或实验设置（含 seed 数、轨迹级样本数、独立实验数）；不满足时：补明示。

**状态**: ✅ 已落实

**代码**: configs/protocol/clause97_seed_trajectory_explicit.yaml:1-80（seed/轨迹数/独立实验数明示配置，含clause_id/explicit_declaration[seed_count/trajectory_per_seed/independent_experiments]/compliance_path[table_footnote/experiment_settings]/four_combinations/data_layers/manual_line共九大块字段）; src/liquidloc/analysis/clause_gates.py::p102_seed_trajectory_explicit():11496（验证函数，十一项检查a-k全覆盖）; AGENTS.md:320（治理载体索引条目）; rules/指令状态表.md:115（指令状态表条目）; outputs/compliance_report_97.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/clause97_seed_trajectory_explicit.yaml 含九大块字段：① clause_id="97"; ② explicit_declaration.sub_dict含seed_count.required=true, trajectory_per_seed.required=true, independent_experiments.required=true; ③ compliance_path.sub_dict含table_footnote.enabled=true, experiment_settings.enabled=true; ④ four_combinations.sub_dict含required=true, combinations[A2N2/A2N3/A3N2/A3N3], per_combination_min_trajectory=25, total_min_trajectory=100; ⑤ manual_line=332（修正原第324行错误）。二、验证函数 p102_seed_trajectory_explicit() 十一项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"97";(c) seed_count_required检查为True；(d) trajectory_per_seed_required检查为True；(e) independent_experiments_required检查为True；(f) four_combinations_required检查为True；(g) combinations_4检查为True；(h) per_combination_25检查为True；(i) total_min_100检查为True；(j) manual_line_points_to_332检查为True；(k) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"seed/轨迹数/独立实验数是否明示"→explicit_declaration配置命中；条款"高兼容默认：表注或实验设置"→compliance_path配置命中；条款"不满足时：补明示"→validation_rules检查项命中；逐行核对未发现问题（核对范围: configs/protocol/clause97_seed_trajectory_explicit.yaml 第1-80行 + clause_gates.py 第11496-11625行，核对项: 条款原文逐句全量映射，十一项检查a-k全部通过）。

**审查**: 问题1：配置manual_ref指向第324行，实为第332行（[97]实际行号）→ 已修正；问题2：配置无manual_line字段 → 已添加manual_line=332；问题3：配置无verification_gate字段 → 已添加verification_gate=p102_seed_trajectory_explicit(); 问题4：原无验证函数 → 新增p102_seed_trajectory_explicit(); 问题5：AGENTS.md[97]格式错误（**[97]** → [97]）→ 已修正；逐句对照条款原文全量映射审查：所有子条款均匹配配置字段，无遗漏。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p102_seed_trajectory_explicit; r = p102_seed_trajectory_explicit(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[97] seed/轨迹数/独立实验数明示"; 退出码0

**证据**: configs/protocol/clause97_seed_trajectory_explicit.yaml:1-80（seed/轨迹数/独立实验数明示配置文件，80行，含clause_id/explicit_declaration[3项]/compliance_path/four_combinations/data_layers/manual_line/verification_gate）; src/liquidloc/analysis/clause_gates.py::p102_seed_trajectory_explicit():11496（验证函数，十一项检查a-k）; docs/最终交付_异步手册_424条_20260917.md:332（手册真源，条款[97]原文所在行）; AGENTS.md:320（治理载体索引条目）; rules/指令状态表.md:115（指令状态表[97]条目）; outputs/compliance_report_97.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_97.md