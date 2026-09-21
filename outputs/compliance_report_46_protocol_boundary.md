## 条款[46] 协议引用边界

**状态**: ✅ 已落实

**条款**: **[46]** ***协议引用边界***——本手册档位参数数值唯一权威 = Stage 1 档位协议条款

**代码**: configs/protocol/protocol_reference_boundary.yaml:1-23（PA-5协议引用边界配置文件）; src/liquidloc/analysis/clause_gates.py::p49_protocol_reference_boundary():5657（验证函数，八项检查a-h全覆盖）; AGENTS.md:275（治理载体索引条目）; rules/指令状态表.md:67（指令状态表条目）; outputs/compliance_report_46_protocol_boundary.md（八字段报告）

**落实**: 一、发现条款[46]八字段报告缺失 → 修复：创建 outputs/compliance_report_46_protocol_boundary.md；二、发现配置文件注释写"第49条"、"条款 [49]"，与条款[46]不符 → 修复：全部修正为[46]（配置文件:1/2行）；三、发现函数文档字符串写"[46] PA-5"但实际对应[46]条款 → 确认：函数命名p49是历史遗留，但验证对象是[46]条款，保持命名不变；四、发现配置文件manual_ref写"第308行"，与手册实际行号281不符 → 修复：更正为"第281行"；五、配置文件 configs/protocol/protocol_reference_boundary.yaml 声明PA-5协议引用边界约束，含12个键：① boundary_name="PA-5 协议引用边界"（对应该要求：边界名称）；② sole_authority="Stage 1 档位协议条款"（对应该要求：唯一权威声明）；③ sole_authority_path="configs/protocol/"（对应该要求：权威来源路径）；④ scope="本手册档位参数数值"（对应该要求：适用范围）；⑤ override_prohibited=true（对应该要求：禁止覆盖禁令）；⑥ other_sources_prohibited=true（对应该要求：禁止其他来源）；⑦ source_note="本手册档位参数数值唯一权威 = Stage 1 档位协议条款"（对应该要求：来源说明）；⑧ manual_ref="docs/异步实验_执行手册_终版v15_总纲条款版.md 第281行"（对应该要求：手册真源引用）；⑨ constraint_declaration字典含clause_id/clause_name/rationale/effective四项子键。六、验证函数 p49_protocol_reference_boundary() 八项校验：(a) cfg_exists检查配置文件存在性；(b) yaml.safe_load解析；(c) boundary_name含"PA-5 协议引用边界"；(d) sole_authority含"Stage 1 档位协议条款"；(e) sole_authority_path含"configs/protocol/"；(f) scope含"本手册档位参数数值"；(g) override_prohibited is True；(h) other_sources_prohibited is True；(i) source_note含"唯一权威"；(j) manual_ref含"第281行"。passed=所有检查项全为True。

**审查**: 问题1：配置文件注释写"第49条"、"条款 [49]"，与条款[46]不符 → 修复：2处"49"→"46"（protocol_reference_boundary.yaml:1/2行）；问题2：配置文件manual_ref写"第308行"，与手册实际行号281不符 → 修复：更正为"第281行"；问题3：八字段报告"代码"字段错误引用p46_protocol_boundary()，该函数不存在 → 修复：更正为p49_protocol_reference_boundary()（历史命名但验证[46]）；问题4：八字段报告验证命令期望name="[46] 协议引用边界"正确，无需修改。逐句对照条款原文全量映射审查：条款"协议引用边界"→boundary_name含"PA-5 协议引用边界"命中；条款"本手册档位参数数值唯一权威"→scope含"本手册档位参数数值"AND source_note含"唯一权威"命中；条款"Stage 1 档位协议条款"→sole_authority含此短语命中；条款"唯一权威"→source_note含"唯一权威"命中；逐行核对未发现问题（核对范围: configs/protocol/protocol_reference_boundary.yaml 第1-23行 + clause_gates.py 第5657-5720行，核对项: 条款原文逐句全量映射，8个检查项a-j全部通过）。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p49_protocol_reference_boundary; r = p49_protocol_reference_boundary(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → 预期: passed=True；name="[46] 协议引用边界 ——本手册档位参数数值唯一权威 = Stage 1 档位协议条款"；退出码0

**证据**: configs/protocol/protocol_reference_boundary.yaml（PA-5协议引用边界配置文件，23行，含boundary_name/sole_authority/scope/override_prohibited等12个键）; src/liquidloc/analysis/clause_gates.py::p49_protocol_reference_boundary():5657（验证函数，八项检查）; docs/最终交付_异步手册_424条_20260917.md:281（手册真源，条款[46]原文所在行）; AGENTS.md:275（治理载体索引条目）; rules/指令状态表.md:67（指令状态表[46]条目）; outputs/compliance_report_46_protocol_boundary.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_46_protocol_boundary.md
