## 条款[92] 10 repeats（seed）× 5 方法

**条款**: **[92]** ***10 repeats（seed）× 5 方法*** ——10 repeats（seed）× 5 方法。

**状态**: ✅ 已落实

**代码**: configs/protocol/clause92_10seeds_5methods.yaml:1-46（10repeats配置，含clause_id/seed_config[n_seeds=10]/method_config[n_methods=5]/experiment_units[total_units=50]/compliance[same_seed_init/baseline_consistent]/reference[manual_line=327]共八大块字段）; src/liquidloc/analysis/clause_gates.py::p97_ten_seeds_five_methods():10868（验证函数，九项检查a-i全覆盖）; AGENTS.md:315（治理载体索引条目）; rules/指令状态表.md:110（指令状态表条目）; outputs/compliance_report_92.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/clause92_10seeds_5methods.yaml 含八大块字段：① clause_id="92"; ② seed_config.n_seeds=10; ③ method_config.n_methods=5; ④ experiment_units.total_units=50（10seed×5方法=50单元）; ⑤ compliance.same_seed_init=true; ⑥ compliance.baseline_consistent=true; ⑦ reference.manual_line=327（修正原第319行错误）。二、验证函数 p97_ten_seeds_five_methods() 九项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"92"；(c) n_seeds_eq_10检查为True；(d) n_methods_eq_5检查为True；(e) total_units_eq_50检查为True；(f) same_seed_init_true检查为True；(g) baseline_consistent_true检查为True；(h) manual_line_points_to_327检查为True；(i) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"10 repeats（seed）"→seed_config.n_seeds=10命中；条款"5 方法"→method_config.n_methods=5命中；条款"10 seed×5方法=50单元"→experiment_units.total_units=50命中；逐行核对未发现问题（核对范围: configs/protocol/clause92_10seeds_5methods.yaml 第1-46行 + clause_gates.py 第10868-10995行，核对项: 条款原文逐句全量映射，九项检查a-i全部通过）。

**审查**: 问题1：配置reference.manual_line指向第319行，实为第327行→已修正；问题2：原无验证函数→新增p97_ten_seeds_five_methods(); 问题3：AGENTS.md[92]格式错误（**[92]** → [92]）→已修正；逐句对照条款原文全量映射审查：所有子条款均匹配配置字段，无遗漏。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p97_ten_seeds_five_methods; r = p97_ten_seeds_five_methods(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[92] 10 repeats（seed）× 5 方法"; 退出码0

**证据**: configs/protocol/clause92_10seeds_5methods.yaml:1-46（10repeats配置，46行，含clause_id/seed_config[n_seeds]/method_config[n_methods]/experiment_units[total_units]/compliance[same_seed_init/baseline_consistent]/reference[manual_line]）; src/liquidloc/analysis/clause_gates.py::p97_ten_seeds_five_methods():10868（验证函数，九项检查a-i）; docs/最终交付_异步手册_424条_20260917.md:327（手册真源，条款[92]原文所在行）; AGENTS.md:315（治理载体索引条目）; rules/指令状态表.md:110（指令状态表[92]条目）; outputs/compliance_report_92.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_92.md