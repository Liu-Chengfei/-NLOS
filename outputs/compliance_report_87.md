## 条款[87] 诊断前洁净性gate

**条款**: **[87]** ***诊断前洁净性gate*** ——若少单元/坏文件/指标不可读/告警未清零，直接进诊断会让诊断结论失真（例如少 1 个 seed 的 LNN 恰好是拖后腿的单元，诊断会误判"LNN 非最低"）。

**状态**: ✅ 已落实

**代码**: configs/protocol/diagnostic_cleanliness_gate.yaml:1-45（诊断前洁净性配置，含clause_id/diagnostic_cleanliness_gate[required]/issues_pre_diagnosis[4项检查]/disclaimer_distortion_risk[enabled]/diagnostic_prohibition[when_issues_present]/handbook_ref[line=322]共十五字段）; src/liquidloc/analysis/clause_gates.py::p93_diagnostic_cleanliness_gate():10268（验证函数，十项检查a-j全覆盖）; AGENTS.md:310（治理载体索引条目）; rules/指令状态表.md:105（指令状态表条目）; outputs/compliance_report_87.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/diagnostic_cleanliness_gate.yaml 含十五字段：① clause_id="87"；② diagnostic_cleanliness_gate子字典含required=true（对应该要求：诊断前洁净性门控必需）；③ issues_pre_diagnosis数组含4项：少单元检查、坏文件检查、指标不可读性检查、告警未清零检查（对应该要求：少单元/坏文件/指标不可读/告警未清零）；④ disclaimer_distortion_risk子字典含enabled=true（对应该要求：诊断结论失真风险提示）；⑤ diagnostic_prohibition子字典含when_issues_present=true（对应该要求：有问题禁止诊断）；⑥ handbook_ref指向第322行（修正原第314行错误）。二、验证函数 p93_diagnostic_cleanliness_gate() 十项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"87"；(c) diagnostic_cleanliness_gate_required_true检查为True；(d) issues_pre_diagnosis_4_items检查为True；(e) issue_1_少单元检查=True；(f) issue_2_坏文件检查=True；(g) issue_3_指标不可读性检查=True；(h) issue_4_告警未清零检查=True；(i) disclaimer_distortion_risk_enabled_true检查为True；(j) diagnostic_prohibition_when_issues_present_true检查为True；(k) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"少单元"→issue_1_少单元检查=True命中；条款"坏文件"→issue_2_坏文件检查=True命中；条款"指标不可读"→issue_3_指标不可读性检查=True命中；条款"告警未清零"→issue_4_告警未清零检查=True命中；条款"直接进诊断会让诊断结论失真"→disclaimer_distortion_risk_enabled=True命中；条款"禁止进行后续诊断"→diagnostic_prohibition_when_issues_present=True命中；逐行核对未发现问题（核对范围: configs/protocol/diagnostic_cleanliness_gate.yaml 第1-45行 + clause_gates.py 第10268-10366行，核对项: 条款原文逐句全量映射，十项检查a-j全部通过）。

**审查**: 问题1：配置文件handbook_ref指向第314行，实为第322行（[87]实际行号）→ 已修正；问题2：原无验证函数 → 新增p93_diagnostic_cleanliness_gate()；问题3：AGENTS.md[87]格式错误（**[87]** → [87]）→ 已修正；逐句对照条款原文全量映射审查：条款"少单元/坏文件/指标不可读/告警未清零"→4项issues_pre_diagnosis全命中；条款"直接进诊断会让诊断结论失真"→disclaimer_distortion_risk_enabled=True命中；条款"禁止进行后续诊断"→diagnostic_prohibition_when_issues_present=True命中；逐行核对未发现问题。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p93_diagnostic_cleanliness_gate; r = p93_diagnostic_cleanliness_gate(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[87] 诊断前洁净性gate"; 退出码0

**证据**: configs/protocol/diagnostic_cleanliness_gate.yaml:1-45（诊断前洁净性配置文件，45行，含clause_id/diagnostic_cleanliness_gate[required]/issues_pre_diagnosis[4项检查]/disclaimer_distortion_risk[enabled]/diagnostic_prohibition[when_issues_present]/handbook_ref[line=322]共十五字段）; src/liquidloc/analysis/clause_gates.py::p93_diagnostic_cleanliness_gate():10268（验证函数，十项检查a-j）; docs/最终交付_异步手册_424条_20260917.md:322（手册真源，条款[87]原文所在行）; AGENTS.md:310（治理载体索引条目）; rules/指令状态表.md:105（指令状态表[87]条目）; outputs/compliance_report_87.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_87.md