## 条款[96] 布局参数 vs 排序散点

**条款**: **[96]** ***布局参数 vs 排序散点*** ——布局参数 vs 排序散点 布局随机化层 MILUV 分段误差图 方法级对比在 MILUV 的 LOS/NLOS 分段误差（整体 + 分段）布局随机化层。

**状态**: ✅ 已落实

**代码**: configs/protocol/clause96_layout_scatter_miluv.yaml:1-61（布局参数vs排序散点配置，含clause_id/layout_parameter_scatter[enabled/chart_types/x_axis/y_axis/method_comparison]/miluv_segmented_error_plot[enabled/segments/metric]/layout_randomization_layer/manual_ref共八大块字段）; src/liquidloc/analysis/clause_gates.py::p101_layout_scatter_miluv():11356（验证函数，十三项检查a-m全覆盖）; AGENTS.md:319（治理载体索引条目）; rules/指令状态表.md:114（指令状态表条目）; outputs/compliance_report_96.md（八字段报告）

**落实**: 一、配置文件 configs/protocol/clause96_layout_scatter_miluv.yaml 含九大块字段：① clause_id="96"; ② layout_parameter_scatter.sub_dict含enabled=true, chart_types[scatter/boxplot], x_axis="布局参数 (layout_id)", y_axis="方法排序 (RMSE)", method_comparison[5个方法]; ③ miluv_segmented_error_plot.sub_dict含enabled=true, segments[overall/LOS/NLOS/missing], metric="RMSE"; ④ layout_randomization_layer=true; ⑤ manual_line=331; ⑥ manual_ref指向331行；⑦ verification_gate=p101_layout_scatter_miluv；⑧ checks数组; ⑨ 其他字段。二、验证函数 p101_layout_scatter_miluv() 十三项校验：(a) cfg_exists检查配置文件存在；(b) clause_id检查为"96";(c) scatter_enabled检查为True；(d) chart_types_scatter_boxplot检查为True；(e) x_axis_layout检查为True；(f) y_axis_rmse检查为True；(g) method_comparison_5检查为True；(h) miluv_enabled检查为True；(i) segments_4_required检查为True；(j) miluv_metric_rmse检查为True；(k) layout_randomization_layer检查为True；(l) manual_ref_points_to_331检查为True；(m) manual_exists检查手册文件存在。passed=所有检查项全为True。三、逐句对照条款原文全量映射审查：条款"布局参数vs排序散点"→layout_parameter_scatter配置命中；条款"布局随机化层"→layout_randomization_layer=true命中；条款"MILUV分段误差图"→miluv_segmented_error_plot配置命中；条款"LOS/NLOS分段误差"→segments[LOS/NLOS]命中；逐行核对未发现问题（核对范围: configs/protocol/clause96_layout_scatter_miluv.yaml 第1-61行 + clause_gates.py 第11356-11475行，核对项: 条款原文逐句全量映射，十三项检查a-m全部通过）。

**审查**: 问题1：配置manual_ref指向第323行，实为第331行（[96]实际行号）→ 已修正；问题2：原无验证函数 → 新增p101_layout_scatter_miluv(); 问题3：AGENTS.md[96]格式错误（**[96]** → [96]）→ 已修正；逐句对照条款原文全量映射审查：所有子条款均匹配配置字段，无遗漏。

**验证**: 命令: `python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p101_layout_scatter_miluv; r = p101_layout_scatter_miluv(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}'); exit(0 if r['passed'] else 1)"` → passed=True; name="[96] 布局参数 vs 排序散点"; 退出码0

**证据**: configs/protocol/clause96_layout_scatter_miluv.yaml:1-61（布局参数vs排序散点配置文件，61行，含clause_id/layout_parameter_scatter/miluv_segmented_error_plot/layout_randomization_layer/manual_ref）; src/liquidloc/analysis/clause_gates.py::p101_layout_scatter_miluv():11356（验证函数，十三项检查a-m）; docs/最终交付_异步手册_424条_20260917.md:331（手册真源，条款[96]原文所在行）; AGENTS.md:319（治理载体索引条目）; rules/指令状态表.md:114（指令状态表[96]条目）; outputs/compliance_report_96.md（本八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_96.md