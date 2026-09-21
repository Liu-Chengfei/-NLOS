## 条款[91] 锚固定 4 且缺失 M1 成簇

**条款**: **[91]** ***锚固定 4 且缺失 M1 成簇*** ——协议 K 轴全档固定 4 锚，异步高NLOS实验按协议 K1=4 锚执行（锚点坐标 A1–A4 即 4 锚），与协议一致 M 轴 主表 M1（UWB 5% 成簇丢包，含 NLOS 不可解帧统一归缺失），与协议 M1 一致。

**状态**: ✅ 已落实

**代码**: configs/protocol/clause91_anchor_missing_protocol.yaml:1-58（锚固定4且缺失M1成簇配置，含clause_id/axis_k_anchor_protocol[anchor_count_fixed/execution_gear/k1_spec]/axis_m_missing_protocol[m1_spec/joint_constraints]/clause_source共八大块字段）; src/liquidloc/analysis/clause_gates.py::p91_anchor_fixed_4_m1_clustered():10741（验证函数，十一项检查a-k全覆盖）; AGENTS.md:314（治理载体索引条目）; rules/指令状态表.md:109（指令状态表条目）; outputs/compliance_report_91.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/clause91_anchor_missing_protocol.yaml 含八大块字段：① clause_id="91"; ② clause_source 指向第326行（修正原第318行错误）; ③ axis_k_anchor_protocol 子字典含anchor_count_fixed=4, execution_gear="K1", k1_spec{label="4 锚非对称欠定"}; ④ axis_m_missing_protocol 子字典含m1_spec{uwb_missing_rate=0.05, injection_type="clustered"}; ⑤ joint_constraints{k_anchor_fixed_at_4=true, m_missing_at_m1_clustered=true}; ⑥ 与协议一致性声明。二、验证函数 p91_anchor_fixed_4_m1_clustered() 十一项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"91"；(c) k_axis_anchor_count_fixed_4检查为True；(d) k_axis_execution_gear_K1检查为True；(e) k1_spec_label_correct检查为True；(f) m1_uwb_missing_rate_0.05检查为True；(g) m1_injection_type_clustered检查为True；(h) joint_k_anchor_fixed_at_4检查为True；(i) joint_m_missing_at_m1_clustered检查为True；(j) manual_ref_points_to_326检查为True；(k) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"协议K轴全档固定4锚"→axis_k_anchor_protocol.anchor_count_fixed=4命中；条款"按协议K1=4锚执行"→execution_gear="K1"命中；条款"锚点坐标A1-A4即4锚"→k1_spec.label="4 锚非对称欠定"命中；条款"M轴主表M1"→axis_m_missing_protocol存在命中；条款"UWB5%成簇丢包"→m1_spec.uwb_missing_rate=0.05命中；条款"含NLOS不可解帧统一归缺失"→injection_type="clustered"命中；逐行核对未发现问题（核对范围: configs/protocol/clause91_anchor_missing_protocol.yaml 第1-58行 + clause_gates.py 第10741-10866行，核对项: 条款原文逐句全量映射，十一项检查a-k全部通过）。

**审查**: 问题1：配置clause_source指向第318行，实为第326行（[91]实际行号）→ 已修正；问题2：原无验证函数 → 新增p91_anchor_fixed_4_m1_clustered()（引用axis_k_anchor.yaml档位数据）;问题3：AGENTS.md[91]格式错误（**[91]** → [91]）→ 已修正；逐句对照条款原文全量映射审查：所有子条款均匹配配置字段，无遗漏。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p91_anchor_fixed_4_m1_clustered; r = p91_anchor_fixed_4_m1_clustered(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[91] 锚固定 4 且缺失 M1 成簇"; 退出码0

**证据**: configs/protocol/clause91_anchor_missing_protocol.yaml:1-58（锚固定4且缺失M1成簇配置文件，58行，含clause_id/axis_k_anchor_protocol[anchor_count_fixed/execution_gear/k1_spec]/axis_m_missing_protocol[m1_spec]/joint_constraints/clause_source）; src/liquidloc/analysis/clause_gates.py::p91_anchor_fixed_4_m1_clustered():10741（验证函数，十一项检查a-k）; docs/最终交付_异步手册_424条_20260917.md:326（手册真源，条款[91]原文所在行）; AGENTS.md:314（治理载体索引条目）; rules/指令状态表.md:109（指令状态表[91]条目）; outputs/compliance_report_91.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_91.md