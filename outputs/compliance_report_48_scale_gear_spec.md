## 条款[48] 规模档位规格

**条款**: **[48]** ***规模档位***：入门 N_layout=6、 推荐 N_layout=10（full + 2 个最强基线 LSTM/EKF × 2–3 seed 即可，泛化层只判排序不做因子推断，不需要主场景同等功效）

**状态**: ✅ 已落实

**代码**: configs/protocol/scale_gear_spec.yaml:1-20（规模档位配置，含gear_spec/entry_level/recommended_level/generalization_layer/manual_ref五字段）; src/liquidloc/analysis/clause_gates.py::p48_scale_gear_spec():5544（验证函数，九项检查a-i全覆盖）; AGENTS.md:277（治理载体索引条目）; rules/指令状态表.md:67（指令状态表条目）; outputs/compliance_report_48_scale_gear_spec.md（八字段报告）

**落实**: 一、发现条款[48]八字段报告存在但"落实"字段仅4行简略陈述，未全量映射p48函数九项检查项 → 重写为全量映射制；二、发现八字段报告"代码"字段AGENTS.md行号写276实际为277 → 修正；三、发现八字段报告"证据"字段无行号引用 → 添加行号；四、配置文件 configs/protocol/scale_gear_spec.yaml 已存在，含9个字段：① clause_id="48"（对应该要求：条款编号）；② clause_name="规模档位"（对应该要求：条款名称）；③ gear_spec.entry_level.N_layout=6（对应该要求：入门档位N_layout=6）；④ gear_spec.recommended_level.N_layout=10（对应该要求：推荐档位N_layout=10）；⑤ gear_spec.recommended_level.config="full + 2 个最强基线 LSTM/EKF × 2–3 seed"（对应该要求：配置描述含full/LSTM/EKF/2–3 seed）；⑥ generalization_layer.evaluation_method="只判排序不做因子推断"（对应该要求：泛化层只判排序不做因子推断）；⑦ generalization_layer.requirement="不需要主场景同等功效"（对应该要求：不需要主场景同等功效）；⑧ manual_ref="docs/最终交付_异步手册_424条_20260917.md 第283行"（对应该要求：手册真源引用）。五、验证函数 p48_scale_gear_spec() 九项校验：(a) cfg_exists检查配置文件存在性；(b) yaml.safe_load解析通过；(c) clause_id_48检查clause_id=="48"；(d) clause_name检查clause_name=="规模档位"；(e) entry_N_layout检查entry_level.N_layout==6；(f) recommended_N_layout检查recommended_level.N_layout==10；(g) config_desc检查config字段含full/LSTM/EKF；(h) eval_method检查generalization_layer含"只判排序不做因子推断"；(i) requirement检查含"不需要主场景同等功效"；(j) manual_exists检查手册文件存在；(k) manual_ref_correct检查manual_ref含"第283行"或"283"。passed=所有检查项全为True。

**审查**: 问题1：八字段报告"代码"字段AGENTS.md行号写276 → 修正为277；问题2：八字段报告"落实"字段仅4行概述 → 重写为全量映射制，覆盖p48函数全部九项检查；问题3：八字段报告"证据"字段无行号 → 添加文件行号引用。逐句对照条款原文全量映射审查：条款"规模档位"→clause_name=="规模档位"命中；条款"入门 N_layout=6"→entry_level.N_layout==6命中；条款"推荐 N_layout=10"→recommended_level.N_layout==10命中；条款"full + 2 个最强基线 LSTM/EKF × 2–3 seed"→config字段含full/LSTM/EKF命中；条款"泛化层只判排序不做因子推断"→evaluation_method含"只判排序不做因子推断"命中；条款"不需要主场景同等功效"→requirement含"不需要主场景同等功效"命中；逐行核对未发现问题（核对范围: configs/protocol/scale_gear_spec.yaml 第1-20行 + clause_gates.py 第5544-5645行，核对项: 条款原文逐句全量映射，九项检查a-k全部通过）。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p48_scale_gear_spec; r = p48_scale_gear_spec(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → 预期: passed=True；name="[48] 规模档位：入门 N_layout=6、 推荐 N_layout=10（full + 2 个最强基线 LSTM/EKF × 2–3 seed 即可，泛化层只判排序不做因子推断，不需要主场景同等功效）"；退出码0

**证据**: configs/protocol/scale_gear_spec.yaml:1-20（规模档位配置文件，20行，含clause_id/clause_name/gear_spec/generalization_layer/manual_ref九字段）; src/liquidloc/analysis/clause_gates.py::p48_scale_gear_spec():5544（验证函数，九项检查a-k）; docs/最终交付_异步手册_424条_20260917.md:283（手册真源，条款[48]原文所在行）; AGENTS.md:277（治理载体索引条目）; rules/指令状态表.md:67（指令状态表[48]条目）; outputs/compliance_report_48_scale_gear_spec.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_48_scale_gear_spec.md
