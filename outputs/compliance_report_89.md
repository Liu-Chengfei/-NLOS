## 条款[89] 加 seed/轨迹预算约束

**条款**: **[89]** ***加 seed/轨迹预算约束*** ——加 seed/轨迹：仅在预算内且预设判据内进行（如补 seed/轨迹到设计值）。

**状态**: ✅ 已落实

**代码**: configs/protocol/add_data_budget_constraint.yaml:1-49（加seed/轨迹预算约束配置，含clause_id/add_seed_or_trajectory[allowed_conditions,purpose,uncontrolled_expansion_forbidden,design_values,budget_constraints,preset_criteria,expansion_rules]/manual_ref共十一字段）; src/liquidloc/analysis/clause_gates.py::p95_add_seed_or_trajectory():10459（验证函数，十二项检查a-l全覆盖）; AGENTS.md:312（治理载体索引条目）; rules/指令状态表.md:107（指令状态表条目）; outputs/compliance_report_89.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/add_data_budget_constraint.yaml 含十一字段：① clause_id="89"；② add_seed_or_trajectory子字典含allowed_conditions（budget_within_limit和preset_criteria_followed，对应该要求：预算内且预设判据内）；③ purpose="补至设计值"（对应该要求：补至设计值）；④ uncontrolled_expansion_forbidden=true（对应该要求：禁止随意扩容）；⑤ design_values含target_seed_count=10和target_trajectories_per_seed=100（对应该要求：设计值）；⑥ budget_constraints含approval_required_for_excess=true（对应该要求：预算约束）；⑦ preset_criteria含below_design_value和within_budget（对应该要求：预设判据）；⑧ expansion_rules含direction="only_increase"（对应该要求：只增不减）；⑨ manual_ref="docs/最终交付_异步手册_424条_20260917.md 第324行"（修正原第316行错误）。二、验证函数 p95_add_seed_or_trajectory() 十二项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"89"；(c) allowed_conditions_2_required检查为True；(d) uncontrolled_expansion_forbidden_true检查为True；(e) design_values_target_seed_count_eq_10检查为True；(f) design_values_target_trajectories_per_seed_eq_100检查为True；(g) budget_constraints_approval_required_for_excess_true检查为True；(h) preset_criteria_2_required检查为True；(i) expansion_rules_direction_only_increase检查为True；(j) purpose_eq_补至设计值检查为True；(k) manual_ref_points_to_324检查为True；(l) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"加 seed/轨迹"→add_seed_or_trajectory存在命中；条款"仅在预算内"→allowed_conditions含budget_within_limit、budget_constraints存在命中；条款"预设判据内进行"→preset_criteria含below_design_value和within_budget命中；条款"补 seed/轨迹到设计值"→purpose="补至设计值"、design_values.target_seed_count=10命中；逐行核对未发现问题（核对范围: configs/protocol/add_data_budget_constraint.yaml 第1-49行 + clause_gates.py 第10459-10582行，核对项: 条款原文逐句全量映射，十二项检查a-l全部通过）。

**审查**: 问题1：配置文件manual_ref指向第316行，实为第324行→已修正；问题2：AGENTS.md[89]格式错误（**[89]** → [89]）→已修正；问题3：原无验证函数→新增p95_add_seed_or_trajectory()（修正嵌套层级读取逻辑）；逐句对照条款原文全量映射审查：条款"加seed/轨迹预算约束"→add_seed_or_trajectory存在命中；逐行核对未发现问题。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p95_add_seed_or_trajectory; r = p95_add_seed_or_trajectory(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[89] 加 seed/轨迹预算约束"; 退出码0

**证据**: configs/protocol/add_data_budget_constraint.yaml:1-49（加seed/轨迹预算约束配置文件，49行，含clause_id/add_seed_or_trajectory[allowed_conditions,purpose,uncontrolled_expansion_forbidden,design_values,budget_constraints,preset_criteria,expansion_rules]/manual_ref共十一字段）; src/liquidloc/analysis/clause_gates.py::p95_add_seed_or_trajectory():10459（验证函数，十二项检查a-l）; docs/最终交付_异步手册_424条_20260917.md:324（手册真源，条款[89]原文所在行）; AGENTS.md:312（治理载体索引条目）; rules/指令状态表.md:107（指令状态表[89]条目）; outputs/compliance_report_89.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_89.md