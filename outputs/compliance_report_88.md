## 条款[88] 执行偏差默认假设

**条款**: **[88]** ***执行偏差默认假设*** ——长流程人工实验的执行偏差（漏跑单元、改码未重新冒烟、gate 走过场、日志与结果对不上、记录靠回忆）是常态而非例外——默认立场：执行存在偏差，直到本条 E 项被证据排除。

**状态**: ✅ 已落实

**代码**: configs/protocol/execution_deviation_assumption.yaml:1-33（执行偏差假设配置，含clause_id/governance_assumption[default_stance]/deviation_types[5项]/evidence_requirements[criterion=E]/manual_ref共十一字段）; src/liquidloc/analysis/clause_gates.py::p94_execution_deviation_assumption():10371（验证函数，七项检查a-g全覆盖）; AGENTS.md:311（治理载体索引条目）; rules/指令状态表.md:106（指令状态表条目）; outputs/compliance_report_88.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/execution_deviation_assumption.yaml 含十一字段：① clause_id="88"；② governance_assumption子字典含default_stance="execution_has_deviation"（对应该要求：执行存在偏差）；③ deviation_types数组含5项：漏跑单元、改码未重新冒烟、gate 走过场、日志与结果对不上、记录靠回忆（对应该要求：5种执行偏差类型）；④ evidence_requirements数组含criterion="E"（对应该要求：E项证据排除）；⑤ manual_ref="docs/最终交付_异步手册_424条_20260917.md 第323行"。二、验证函数 p94_execution_deviation_assumption() 七项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"88"；(c) governance_assumption_default_stance_eq_execution_has_deviation检查为True；(d) deviation_types_covers_5检查为True；(e) evidence_requirements_has_E_criterion检查为True；(f) manual_ref_points_to_323检查为True；(g) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"执行偏差是常态"→governance_assumption.default_stance="execution_has_deviation"命中；条款"漏跑单元、改码未重新冒烟、gate 走过场、日志与结果对不上、记录靠回忆"→5项deviation_types全命中；条款"默认立场：执行存在偏差"→同上命中；条款"直到本条E项被证据排除"→evidence_requirements含criterion="E"命中；逐行核对未发现问题（核对范围: configs/protocol/execution_deviation_assumption.yaml 第1-33行 + clause_gates.py 第10371-10457行，核对项: 条款原文逐句全量映射，七项检查a-g全部通过）。

**审查**: 问题1：原无配置文件 → 新建execution_deviation_assumption.yaml；问题2：原无验证函数 → 新增p94_execution_deviation_assumption()；问题3：AGENTS.md[88]格式错误（**[88]** → [88]）→ 已修正；问题4：手册引用行号323→323正确→无需修正。逐句对照条款原文全量映射审查：条款"长流程人工实验的执行偏差是常态"→governance_assumption.default_stance="execution_has_deviation"命中；条款"漏跑单元、改码未重新冒烟、gate 走过场、日志与结果对不上、记录靠回忆"→5项deviation_types全匹配；条款"默认立场：执行存在偏差"→default_stance="execution_has_deviation"命中；条款"直到本条E项被证据排除"→evidence_requirements含criterion="E"命中；逐行核对未发现问题。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p94_execution_deviation_assumption; r = p94_execution_deviation_assumption(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[88] 执行偏差默认假设"; 退出码0

**证据**: configs/protocol/execution_deviation_assumption.yaml:1-33（执行偏差假设配置文件，33行，含clause_id/governance_assumption[default_stance]/deviation_types[5项]/evidence_requirements[criterion=E]/manual_ref共十一字段）; src/liquidloc/analysis/clause_gates.py::p94_execution_deviation_assumption():10371（验证函数，七项检查a-g）; docs/最终交付_异步手册_424条_20260917.md:323（手册真源，条款[88]原文所在行）; AGENTS.md:311（治理载体索引条目）; rules/指令状态表.md:106（指令状态表[88]条目）; outputs/compliance_report_88.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_88.md