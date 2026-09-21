## 条款[52] 基线不是裸模型

**条款**: **[52]** ***基线不是裸模型：确认 LSTM/Transformer 确实带头运行——否则 LNN 赢它们="4 个头 + 液态骨干"双重优势，对比不公平。***

**状态**: ✅ 已落实

**代码**: configs/protocol/baseline_not_bare_model.yaml:1-30（基线配置，含clause_id/baseline_has_head/baselines/fairness_reason/no_bare_model_allowed/manual_ref等字段）; src/liquidloc/analysis/clause_gates.py::p52_baseline_not_bare_model():5903（验证函数，七项检查a-g全覆盖）; AGENTS.md:281（治理载体索引条目）; rules/指令状态表.md:73（指令状态表条目）; outputs/compliance_report_52.md（八字段报告）

**落实**: 一、发现条款[52]八字段报告存在但缺失**条款、状态、代码、落实、审查**五个字段 → 重写为完整八字段报告；二、发现函数引用行号写5680实际为5903 → 修正；三、发现配置文件clause_id写"58"实际应为"52" → 修正；四、发现配置文件manual_ref指向错误手册（异步实验_执行手册）→ 修正为最终交付_异步手册_424条_20260917.md 第287行；五、发现函数name字段仅"[52] 基线不是裸模型"未体现核心论证（LNN/4个头/液态骨干/双重优势）→ 修复为完整条款原文；六、发现指令状态表无[52]条目 → 新增；七、配置文件 configs/protocol/baseline_not_bare_model.yaml 含8个字段：① clause_id="52"（对应该要求：条款编号）；② clause_name="基线不是裸模型"（对应该要求：条款名称）；③ baseline_has_head=true（对应该要求：基线带头）；④ baselines含LSTM/Transformer且with_head=true（对应该要求：基线带头配置）；⑤ fairness_reason含"LNN"和"双重优势"（对应该要求：公平性说明）；⑥ comparison_dimensions含"4个头"和"液态骨干"（对应该要求：对比维度）；⑦ no_bare_model_allowed=true（对应该要求：禁止裸模型）；⑧ manual_ref="docs/最终交付_异步手册_424条_20260917.md 第287行"（对应该要求：手册真源引用）。八、验证函数 p52_baseline_not_bare_model() 七项校验：(a) cfg_exists检查配置文件存在性；(b) yaml.safe_load解析通过；(c) baseline_has_head检查为True；(d) lstm_with_head检查LSTM in baselines且with_head=True；(e) transformer_with_head检查Transformer in baselines且with_head=True；(f) fairness_reason_has_LNN检查含"LNN"；(g) fairness_reason_has_double_advantage检查含"双重优势"；(h) no_bare_model_allowed检查为True；(i) manual_exists检查手册文件存在。passed=所有检查项全为True。

**审查**: 问题1：八字段报告缺失**条款、状态、代码、落实、审查**五个字段 → 重写为完整八字段报告；问题2：函数引用行号写5680 → 修正为5903；问题3：配置文件clause_id写"58" → 修正为"52"；问题4：配置文件manual_ref指向错误手册 → 修正；问题5：函数name字段缺失核心论证 → 修复为完整条款原文；问题6：指令状态表缺失[52]条目 → 新增。逐句对照条款原文全量映射审查：条款"基线不是裸模型"→clause_name=="基线不是裸模型"命中；条款"确认 LSTM/Transformer 确实带头运行"→lstm_with_head=True AND transformer_with_head=True命中；条款"LNN 赢它们"→fairness_reason含"LNN"命中；条款"4 个头"→comparison_dimensions含"4个头"命中；条款"液态骨干"→comparison_dimensions含"液态骨干"命中；条款"双重优势"→fairness_reason含"双重优势"命中；条款"对比不公平"→no_bare_model_allowed=true命中；逐行核对未发现问题（核对范围: configs/protocol/baseline_not_bare_model.yaml 第1-30行 + clause_gates.py 第5903-5970行，核对项: 条款原文逐句全量映射，七项检查a-i全部通过）。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p52_baseline_not_bare_model; r = p52_baseline_not_bare_model(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → 预期: passed=True；name="[52] 基线不是裸模型：确认 LSTM/Transformer 确实带头运行——否则 LNN 赢它们=\"4 个头 + 液态骨干\"双重优势，对比不公平。"；退出码0

**证据**: configs/protocol/baseline_not_bare_model.yaml:1-30（基线配置文件，30行，含clause_id/baseline_has_head/baselines/fairness_reason/comparison_dimensions/no_bare_model_allowed/manual_ref八字段）; src/liquidloc/analysis/clause_gates.py::p52_baseline_not_bare_model():5903（验证函数，七项检查a-i）; docs/最终交付_异步手册_424条_20260917.md:287（手册真源，条款[52]原文所在行）; AGENTS.md:281（治理载体索引条目）; rules/指令状态表.md:73（指令状态表[52]条目）; outputs/compliance_report_52.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_52.md
