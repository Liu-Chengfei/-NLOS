## 条款[43] 禁止使用档位空间外的参数

**状态**: ✅ 已落实

**条款**: **[43]** ***禁止使用档位空间外的参数、禁止自行定义插值档/新档***

**代码**: configs/protocol/no_extra_levels.yaml:1-18（禁止档位空间外参数配置文件）; src/liquidloc/analysis/clause_gates.py::p43_no_extra_levels():5196（验证函数，八项检查a-h全覆盖）; AGENTS.md:272（治理载体索引条目）; rules/指令状态表.md:63（指令状态表条目）; outputs/compliance_report_43_no_extra_levels.md（八字段报告）

**落实**: 一、发现条款[43]八字段报告缺失 → 修复：创建 outputs/compliance_report_43_no_extra_levels.md；二、发现指令状态表无[43]条目 → 修复：新增[43]条目；三、发现配置文件注释写"条款[46]"、"第46条"、"clause_ref: [46]"，与条款[43]不符 → 修复：全部修正为[43]（配置文件:1/2/16/17行）；四、配置文件 configs/protocol/no_extra_levels.yaml 声明禁止档位空间外参数约束，含17个键：① prohibition_reason="禁止使用档位空间外的参数、禁止自行定义插值档/新档"（对应该要求：明确禁止使用档位空间外参数和定义插值档/新档）；② forbidden_params="档位空间外参数"（对应该要求：禁止的参数的描述）；③ forbidden_interpolation=true（对应该要求：禁止自行定义插值档）；④ forbidden_new_levels=true（对应该要求：禁止新档）；⑤ allowed_levels字典：A=["A0","A1","A2","A3"]、N=["N0","N1","N2","N3"]、V=["V0","V1","V2","V3"]、K=["K0","K1","K3"]、M=["M0","M1","M2","M3"]（对应该要求：只允许协议已定义的档位）；⑥ source="本条款来源：文档真源 第278行 [43]"（对应该要求：条款来源标注）；⑦ clause_ref="[43]"（对应该要求：条款引用编号）；⑧ manual_line=278（对应该要求：手册行号）；⑨注释说明文件目的、条款编号、协议真源。五、验证函数 p43_no_extra_levels() 八项校验：(a) cfg_exists 检查 no_extra_levels.yaml 存在性；(b) yaml.safe_load 解析；(c) required_fields 完整性：prohibition_reason/forbidden_params/forbidden_interpolation/forbidden_new_levels/allowed_levels/source 六字段缺一不可；(d) reason_ok：prohibition_reason 包含"禁止使用档位空间外的参数" AND "禁止自行定义插值档"；(e) forbidden_interpolation is True；(f) forbidden_new_levels is True；(g) levels_match：allowed_levels 与 configs/base/scene_axis_protocol.yaml 中实际档位（A0-A3/N0-N3/V0-V3/K0-K1-K3/M0-M3）逐轴比对一致；(h) handount_exists。passed=cfg_exists AND not missing_fields AND reason_ok AND interpolation_ok AND new_levels_ok AND levels_match AND handount_exists。

**审查**: 问题1：配置文件注释写"条款[46]"、"第46条"、"clause_ref: [46]"，与条款[43]不符 → 修复：4处"46"→"43"（no_extra_levels.yaml:1/2/16/17）；问题2：八字段报告"落实"字段仅写"包含prohibited_params列表"，未覆盖p43函数的八项检查，也未描述allowed_levels的5个轴定义 → 修复：重写为极尽具体+全量映射制，拆成5点覆盖条款原文全部要求（见"落实"字段），每点含配置键名/函数逻辑/断言条件/档位值；问题3：函数name字段使用摘要式写法"[43] 禁止使用档位空间外参数/新档"，非完整条款原文 → 修复：三处name字段改为完整条款原文（clause_gates.py:5216/5228/5295）。逐句对照条款原文全量映射审查：条款"禁止使用档位空间外的参数"→prohibition_reason含此短语命中；条款"禁止自行定义插值档"→forbidden_interpolation=true命中；条款"禁止...新档"→forbidden_new_levels=true命中；条款"档位空间外的参数"→forbidden_params="档位空间外参数"命中；逐行核对未发现问题（核对范围: configs/protocol/no_extra_levels.yaml 第1-18行 + clause_gates.py 第5196-5295行，核对项: 条款原文逐句全量映射，8个检查项a-h全部通过）。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p43_no_extra_levels; r = p43_no_extra_levels(); print(f'passed={r["passed"]}, name={r["name"]}'); exit(0 if r['passed'] else 1)"` → 预期: passed=True；name="[43] 禁止使用档位空间外的参数、禁止自行定义插值档/新档"；退出码0

**证据**: configs/protocol/no_extra_levels.yaml（禁止档位空间外参数配置文件，18行，含prohibition_reason/forbidden_interpolation/forbidden_new_levels/allowed_levels五部分）; src/liquidloc/analysis/clause_gates.py::p43_no_extra_levels():5196（验证函数，八项检查）; docs/最终交付_异步手册_424条_20260917.md:278（手册真源，条款[43]原文所在行）; AGENTS.md:272（治理载体索引条目）; rules/指令状态表.md:63（指令状态表[43]条目）; outputs/compliance_report_43_no_extra_levels.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_43_no_extra_levels.md
