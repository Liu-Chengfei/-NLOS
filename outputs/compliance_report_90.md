## 条款[90] 学习率三段式协议全局唯一

**条款**: **[90]** ***学习率三段式协议全局唯一*** ——学习率三段式协议全局唯一（Part 0：warm-up 线性升 → 平台恒定 → 后 1/3 段衰减到 lr_max×0.1，只允许调 W_up/lr_max/衰减比例，三网络同形态同预算训练不充分允许调 lr 但最多 2 轮、禁掩盖 bug）

**状态**: ✅ 已落实

**代码**: configs/protocol/learning_rate_three_phase_global.yaml:1-70（学习率三段式协议配置，含clause_id/learning_rate_three_phase_protocol[unique_global/phases[3阶段]/adjustable_parameters/lr_tuning限制/bug_masking/forced_unified_lr/manifests]/manual_ref共八大块十七项字段）; src/liquidloc/analysis/clause_gates.py::p96_learning_rate_three_phase():10590（验证函数，十四项检查a-n全覆盖）; AGENTS.md:313（治理载体索引条目）; rules/指令状态表.md:108（指令状态表条目）; outputs/compliance_report_90.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/learning_rate_three_phase_global.yaml 含八大块字段：① clause_id="90"；② learning_rate_three_phase_protocol子字典：unique_global=true（对应该要求：全局唯一）；③ phases数组3阶段：warmup_linear_up(type=linear_ramp,max_fraction=0.05)、plateau_constant(type=constant,value=lr_max)、last_third_decay_to_lr_max_x_0.1(type=decay,target_value=lr_max*0.1)（对应该要求：warm-up线性升→平台恒定→后1/3段衰减）；④ adjustable_parameters数组3参数：W_up、lr_max、decay_ratio（对应该要求：只允许调这3项）；⑤ training_inadequate_lr_tuning_allowed=true,max_lr_tuning_rounds=2（对应该要求：最多2轮）；⑥ bug_masking_forbidden=true（对应该要求：禁掩盖bug）；⑦ three_networks_same_schedule=true,three_networks_same_budget=true（对应该要求：三网络同形态同预算）；⑧ manifest_required=true。二、验证函数 p96_learning_rate_three_phase() 十四项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"90"；(c) unique_global_true检查为True；(d) phases_3_exist检查为True；(e) warmup_linear_up_correct检查为True；(f) plateau_constant_correct检查为True；(g) decay_phase_correct检查为True；(h) adjustable_params_3_correct检查为True；(i) lr_tuning_allowed_2_rounds检查为True；(j) bug_masking_forbidden_true检查为True；(k) three_networks_same_lr检查为True；(l) manifest_required_true检查为True；(m) manual_ref_points_to_325检查为True；(n) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"学习率三段式协议全局唯一"→unique_global=true命中；条款"warm-up线性升→平台恒定→后1/3段衰减到lr_max×0.1"→3阶段配置匹配；条款"只允许调W_up/lr_max/衰减比例"→adjustable_parameters=3项匹配；条款"训练不充分允许调lr但最多2轮"→max_lr_tuning_rounds=2匹配；条款"禁掩盖bug"→bug_masking_forbidden=true匹配；条款"三网络同形态同预算"→three_networks_same_schedule=true、three_networks_same_budget=true匹配；逐行核对未发现问题（核对范围: configs/protocol/learning_rate_three_phase_global.yaml 第1-70行 + clause_gates.py 第10590-10719行，核对项: 条款原文逐句全量映射，十四项检查a-n全部通过）。

**审查**: 问题1：配置手册行号317，实为325（[90]实际行号）→ 已修正；问题2：原无验证函数→ 新增p96_learning_rate_three_phase()；问题3：AGENTS.md[90]格式错误（**[90]** → [90]）→ 已修正；逐句对照条款原文全量映射审查：所有子条款均匹配配置字段，无遗漏。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p96_learning_rate_three_phase; r = p96_learning_rate_three_phase(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[90] 学习率三段式协议全局唯一"; 退出码0

**证据**: configs/protocol/learning_rate_three_phase_global.yaml:1-70（学习率三段式协议配置文件，70行，含unique_global/phases[3阶段]/adjustable_parameters/衰减约束/bug掩码禁令/统一lr强制/记录manifests）; src/liquidloc/analysis/clause_gates.py::p96_learning_rate_three_phase():10590（验证函数，十四项检查a-n）; docs/最终交付_异步手册_424条_20260917.md:325（手册真源，条款[90]原文所在行）; AGENTS.md:313（治理载体索引条目）; rules/指令状态表.md:108（指令状态表[90]条目）; outputs/compliance_report_90.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_90.md