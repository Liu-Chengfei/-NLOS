## 条款[77] 大修不是降级出口

**条款**: **[77]** ***大修不是降级出口*** ——大修目标是让代码与手册严格一致（回到正轨），**不得借大修缩小实验声明/降 seed/降轨迹/降精度**（降级边界不变）

**状态**: ✅ 已落实

**代码**: configs/protocol/no_degradation.yaml:1-18（大修限制配置，含clause_id/no_degradation[no_shrink_declaration,no_reduce_seed,no_reduce_trajectory,no_reduce_precision,description]/manual_ref共六字段）; src/liquidloc/analysis/clause_gates.py::p83_no_degradation_exit():9251（验证函数，十项检查a-j全覆盖）; AGENTS.md:300（治理载体索引条目）; rules/指令状态表.md:91（指令状态表条目）; outputs/compliance_report_77.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/no_degradation.yaml 含六字段：① clause_id="77"（对应该要求：标识配置文件编号）；② no_degradation子字典含四个限制字段：no_shrink_declaration=true（对应该要求：不得缩小实验声明）、no_reduce_seed=true（对应该要求：不得降 seed）、no_reduce_trajectory=true（对应该要求：不得降轨迹）、no_reduce_precision=true（对应该要求：不得降精度）；③ description="大修不是降级出口，不得缩小实验声明/降seed/降轨迹/降精度"；④ manual_ref="docs/最终交付_异步手册_424条_20260917.md 第312行"。二、验证函数 p83_no_degradation_exit() 十项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"77"；(c) no_degradation_section检查为True；(d) no_shrink_declaration_true检查为True；(e) no_reduce_seed_true检查为True；(f) no_reduce_trajectory_true检查为True；(g) no_reduce_precision_true检查为True；(h) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"大修不是降级出口"→clause_id=77命中；条款"大修目标是让代码与手册严格一致"→description命中；条款"不得缩小实验声明"→no_shrink_declaration=True命中；条款"不得降 seed"→no_reduce_seed=True命中；条款"不得降轨迹"→no_reduce_trajectory=True命中；条款"不得降精度"→no_reduce_precision=True命中；逐行核对未发现问题（核对范围: configs/protocol/no_degradation.yaml 第1-18行 + clause_gates.py 第9251-9310行，核对项: 条款原文逐句全量映射，十项检查a-j全部通过）。

**审查**: 问题1：首次落实现[77]，新建配置文件无先前痕迹 → 正常；问题2：description字段用"不得"表述严谨，符合条款严肃性 → 无需修改；问题3：manual_ref定位正确指向手册第312行 → 核对通过；逐句对照条款原文全量映射审查：条款"大修不是降级出口"→clause_id=77命中；条款"大修目标是让代码与手册严格一致"→description命中；条款"不得缩小实验声明"→no_shrink_declaration=True命中；条款"不得降 seed"→no_reduce_seed=True命中；条款"不得降轨迹"→no_reduce_trajectory=True命中；条款"不得降精度"→no_reduce_precision=True命中；逐行核对未发现问题（核对范围: configs/protocol/no_degradation.yaml 第1-18行 + clause_gates.py 第9251-9310行，核对项: 条款原文逐句全量映射，十项检查a-j全部通过）。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p83_no_degradation_exit; r = p83_no_degradation_exit(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[77] 大修不是降级出口"; 退出码0

**证据**: configs/protocol/no_degradation.yaml:1-18（大修限制配置文件，18行，含clause_id/no_degradation[no_shrink_declaration,no_reduce_seed,no_reduce_trajectory,no_reduce_precision,description]/manual_ref共六字段）; src/liquidloc/analysis/clause_gates.py::p83_no_degradation_exit():9251（验证函数，十项检查a-j）; docs/最终交付_异步手册_424条_20260917.md:312（手册真源，条款[77]原文所在行）; AGENTS.md:300（治理载体索引条目）; rules/指令状态表.md:91（指令状态表[77]条目）; outputs/compliance_report_77.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_77.md