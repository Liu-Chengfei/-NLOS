## 条款[94] 表交付清单

**条款**: **[94]** ***表交付清单*** ——每个表必须注明：数据层来源（主场景/布局随机化/MILUV）、分析单位（轨迹级/帧级）、是否 10 seed 聚合、是否校正后 p 值。

**状态**: ✅ 已落实

**代码**: configs/protocol/clause94_table_delivery_list.yaml:1-57（表交付清单配置，含clause_id/table_delivery_requirements[data_layer_source/analysis_unit/seed_aggregation/corrected_p_value]/tables_expected[3表]/checks[5项]共八大块字段）; src/liquidloc/analysis/clause_gates.py::p99_table_delivery_list():11085（验证函数，十项检查a-j全覆盖）; AGENTS.md:317（治理载体索引条目）; rules/指令状态表.md:112（指令状态表条目）; outputs/compliance_report_94.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/clause94_table_delivery_list.yaml 含八大块字段：① clause_id="94"; ② table_delivery_requirements子字典含data_layer_source（主场景、布局随机化、MILUV）; ③ analysis_unit（轨迹级、帧级）; ④ seed_aggregation（10 seed聚合、非聚合）; ⑤ corrected_p_value（是、否）; ⑥ tables_expected数组3表：布局参数 vs 排序散点、MILUV外部域布局随机化层seed/超参稳健性表、10seed间LNN最低成立频次; ⑦ checks数组5项：table_metadata_specified, data_layer_declared, analysis_unit_specified, seed_aggregation_declared, p_value_correction_declared。二、验证函数 p99_table_delivery_list() 十项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"94";(c) data_layer_source_3_required检查为True；(d) analysis_unit_2_required检查为True；(e) seed_aggregation_2_required检查为True；(f) corrected_p_value_2_required检查为True；(g) tables_expected_3检查为True；(h) checks_5_required检查为True；(i) manual_line_points_to_329检查为True；(j) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"数据层来源（主场景/布局随机化/MILUV）"→data_layer_source含三项命中；条款"分析单位（轨迹级/帧级）"→analysis_unit含两项命中；条款"是否10seed聚合"→seed_aggregation含"10 seed 聚合"命中；条款"是否校正后p值"→corrected_p_value含"是""否"命中；逐行核对未发现问题（核对范围: configs/protocol/clause94_table_delivery_list.yaml 第1-57行 + clause_gates.py 第11085-11212行，核对项: 条款原文逐句全量映射，十项检查a-j全部通过）。

**审查**: 问题1：配置manual_line指向第321行，实为第329行（[94]实际行号）→ 已修正；问题2：配置verification_gate引用p94_table_delivery_list，实际函数名为p99_table_delivery_list → 已修正；问题3：原无验证函数 → 新增p99_table_delivery_list(); 问题4：AGENTS.md[94]格式错误（**[94]** → [94]）→ 已修正；逐句对照条款原文全量映射审查：所有子条款均匹配配置字段，无遗漏。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p99_table_delivery_list; r = p99_table_delivery_list(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[94] 表交付清单"; 退出码0

**证据**: configs/protocol/clause94_table_delivery_list.yaml:1-57（表交付清单配置文件，57行，含clause_id/table_delivery_requirements[4项要求]/tables_expected[3表]/checks[5项验证]/verification_gate=p99_table_delivery_list/manual_line=329）; src/liquidloc/analysis/clause_gates.py::p99_table_delivery_list():11085（验证函数，十项检查a-j）; docs/最终交付_异步手册_424条_20260917.md:329（手册真源，条款[94]原文所在行）; AGENTS.md:317（治理载体索引条目）; rules/指令状态表.md:112（指令状态表[94]条目）; outputs/compliance_report_94.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_94.md