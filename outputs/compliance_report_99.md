## 条款[99] N1.5 排除

**条款**: **[99]** ***N1.5 排除（同机另一实验体系档位）*** —— N1.5 属于同机另一实验体系按其各自对应协议文件执行的档位，不在本手册档位协议条款档位表内。

**状态**: ✅ 已落实

**代码**: configs/protocol/clause99_n15_exclusion.yaml:1-20（N1.5 排除配置，含 clause_id/clause_name/scope_note/n15_classification/exclusion_scope/reason/related_clause/manual_ref/manual_line/verification_gate 共十大字段）; src/liquidloc/analysis/clause_gates.py::n15_exclusion_gate():11210（验证函数，十三项检查 a-m 全覆盖）; AGENTS.md:322（治理载体索引条目）; rules/指令状态表.md:117（指令状态表条目）; outputs/compliance_report_99.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/clause99_n15_exclusion.yaml 含十大字段：① clause_id="99"; ② clause_name="N1.5 排除（同机另一实验体系档位）"; ③ scope_note 说明 N1.5 不在本手册档位协议条款档位表内; ④ n15_classification.sub_dict含 belongs_to="同机另一实验体系", type="消融版协议 N 轴标准档", in_this_manual_gears_table=false; ⑤ this_manual_n_axis=[N0,N1,N2,N3]（不含 N1.5）; ⑥ n15_params 含 rho=20-25%, mu=1.5-2.5m, sigma=0.5m; ⑦ exclusion_scope.sub_dict含 experiment_config_check="不涉及 N1.5", precheck_config_check="不涉及 N1.5", config_verification="不涉及 N1.5"; ⑧ reason 完整声明; ⑨ related_clause="[45]"; ⑩ manual_ref=手册第 334 行，manual_line=334（修正原第 326 行错误）。二、验证函数 n15_exclusion_gate() 十三项校验：(a) cfg_exists 检查配置文件存在；(b) clause_id_is_99 检查为 True；(c) n15_classification_present 检查为 True；(d) n15_in_this_manual_table_false 检查为 True；(e) n15_belongs_to_other_system 检查为 True；(f) this_manual_n_axis_no_n15 检查为 True；(g) this_manual_n_axis_has_n0_n3 检查为 True；(h) related_clause_is_45 检查为 True；(i) manual_line_points_to_334 检查为 True；(j) manual_exists 检查为 True；(k) exclusion_scope_present 检查为 True；(l) excl_experiment_config_no_n15 检查为 True；(m) excl_precheck_no_n15 + excl_config_verification_no_n15 检查为 True。passed=所有检查项全为 True。三、逐句对照条款原文全量映射审查：条款"N1.5 属于同机另一实验体系"→n15_classification.belongs_to 命中；条款"不在本手册档位协议条款档位表内"→n15_classification.in_this_manual_gears_table=false 命中；条款"消融版协议 N 轴标准档"→n15_classification.type 命中；条款"本实验与预检配置核对不涉及 N1.5"→exclusion_scope 三项全命中；逐行核对未发现问题（核对范围: configs/protocol/clause99_n15_exclusion.yaml 第 1-20 行 + clause_gates.py 第 11210-11345 行，核对项: 条款原文逐句全量映射，十三项检查 a-m 全部通过）。

**审查**: 问题 1：配置 manual_ref 指向第 326 行，实为第 334 行（[98] 实际行号）→ 已修正；问题 2：配置无 manual_line 字段 → 已添加 manual_line=334；问题 3：配置无 verification_gate 字段 → 已添加 verification_gate=p99_n15_exclusion(); 问题 4：原无验证函数 → 新增 n15_exclusion_gate()；问题 5：原无 compliance_report_99.md → 已创建；逐句对照条款原文全量映射审查：所有子条款均匹配配置字段，无遗漏。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import n15_exclusion_gate; r = n15_exclusion_gate(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[99] N1.5 排除"; 退出码 0

**证据**: configs/protocol/clause99_n15_exclusion.yaml:1-20（N1.5 排除配置，20 行，含 clause_id/clause_name/scope_note/n15_classification/exclusion_scope/manual_line/verification_gate）; src/liquidloc/analysis/clause_gates.py::n15_exclusion_gate():11210（验证函数，十三项检查 a-m）; docs/最终交付_异步手册_424条_20260917.md:334（手册真源，条款[99]原文所在行）; AGENTS.md:322（治理载体索引条目）; rules/指令状态表.md:117（指令状态表[99]条目）; outputs/compliance_report_99.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_99.md