# 第 1 阶段 · 条款 [50] 落实报告

**批次**: Stage 1 条款 [50]
**日期**: 2026-09-17
**阶段**: 第 1 阶段（代码落实期）

> 唯一真源: `docs/异步实验_执行手册_终版v15_总纲条款版.md`。格式 = 手册八字段逐字段规范（逐字载体: `.agents/skills/execution-report-gate/SKILL.md` §1）。
> 本条一行一条；「条款:」字段为对应条款正文全文逐字抄录。本批全程未跑任何实验、未改任何实验代码/口径。

---

## 八字段落实记录

### [50] ✅

**条款**: **[50]** **GT 真值精度声明（硬约束）**：`gt_pos`/`gt_yaw` 为本仿真数据集的**无噪声理想真值**（由轨迹几何精确生成，GT 不确定度 = 0）， 评估时**不引入 GT 不确定度传播**——误差预算（0-4）只含传感器噪声×GDOP 项，不含 GT 噪声项

**代码**: `configs/protocol/gt_truth_precision.yaml:1-27`（GT 真值精度声明配置文件，含 statement_name / constraint_level / gt_pos_nature / gt_yaw_nature / generation_method / gt_uncertainty / no_gt_uncertainty_propagation / error_budget_components / no_gt_noise_in_budget / gt_pos_field / gt_yaw_field / manual_ref / clause_id / clause_name / rationale / effective 字段）；`src/liquidloc/analysis/clause_gates.py::p50_gt_truth_precision():5163-5228`（Gate 验证函数，11 项校验：yaml 存在性 + statement_name 关键短语匹配 + constraint_level 为硬约束 + gt_pos_nature 关键短语匹配 + gt_yaw_nature 关键短语匹配 + generation_method 关键短语匹配 + gt_uncertainty 为 0 + no_gt_uncertainty_propagation 为 true + error_budget_components 含传感器噪声和 GDOP + no_gt_noise_in_budget 为 true + 手册真源存在）；`src/liquidloc/analysis/clause_gates.py::_ALL:5197`（注册到 _ALL 列表，p50 为第 50 项）

**落实**:
①条款原文要求"gt_pos/gt_yaw 为无噪声理想真值" → 落点 `configs/protocol/gt_truth_precision.yaml::gt_pos_nature:9` + `gt_yaw_nature:10` → 实现 `gt_pos_nature: "无噪声理想真值"` + `gt_yaw_nature: "无噪声理想真值"`（逐字匹配条款原文）→ 满足：gt_pos/gt_yaw 真值性质已明确声明
②条款原文要求"由轨迹几何精确生成" → 落点 `configs/protocol/gt_truth_precision.yaml::generation_method:12` → 实现 `generation_method: "由轨迹几何精确生成"`（逐字匹配条款原文）→ 满足：生成方法已声明
③条款原文要求"GT 不确定度 = 0" → 落点 `configs/protocol/gt_truth_precision.yaml::gt_uncertainty:14` → 实现 `gt_uncertainty: 0`（逐字匹配条款原文）→ 满足：GT 不确定度已限定为零
④条款原文要求"评估时不引入 GT 不确定度传播" → 落点 `configs/protocol/gt_truth_precision.yaml::no_gt_uncertainty_propagation:15` → 实现 `no_gt_uncertainty_propagation: true` → 满足：GT 不确定度传播禁令已声明
⑤条款原文要求"误差预算（0-4）只含传感器噪声×GDOP 项" → 落点 `configs/protocol/gt_truth_precision.yaml::error_budget_components:17` → 实现 `error_budget_components: "只含传感器噪声×GDOP 项"`（逐字匹配条款原文）→ 满足：误差预算成分已限定
⑥条款原文要求"不含 GT 噪声项" → 落点 `configs/protocol/gt_truth_precision.yaml::no_gt_noise_in_budget:18` → 实现 `no_gt_noise_in_budget: true` → 满足：GT 噪声项禁令已声明
⑦条款原文要求声明硬约束等级 → 落点 `configs/protocol/gt_truth_precision.yaml::constraint_level:6` → 实现 `constraint_level: "硬约束"`（逐字匹配条款原文）→ 满足：约束等级已声明
⑧条款原文要求声明真值字段名 → 落点 `configs/protocol/gt_truth_precision.yaml::gt_pos_field:20` + `gt_yaw_field:21` → 实现 `gt_pos_field: "gt_pos"` + `gt_yaw_field: "gt_yaw"` → 满足：真值字段名已对齐
⑨条款原文要求手册引用 → 落点 `configs/protocol/gt_truth_precision.yaml::manual_ref:23` → 实现 `manual_ref: "docs/异步实验_执行手册_终版v15_总纲条款版.md 第310行"`（指向手册第310行）→ 满足：手册真源引用已标注
⑩Gate 验证函数静态校验 → 落点 `src/liquidloc/analysis/clause_gates.py::p50_gt_truth_precision():5163-5228` → 实现：11 项校验（yaml 存在性 + statement_name 关键短语匹配 + constraint_level 为硬约束 + gt_pos_nature 关键短语匹配 + gt_yaw_nature 关键短语匹配 + generation_method 关键短语匹配 + gt_uncertainty 为 0 + no_gt_uncertainty_propagation 为 true + error_budget_components 含传感器噪声和 GDOP + no_gt_noise_in_budget 为 true + 手册真源存在）→ 满足：全部校验通过（PASSED）
⑪已有基础核查 → `configs/protocol/` 目录下已有 axis_a_sync.yaml、axis_n_nlos.yaml、protocol_reference_boundary.yaml 等协议配置文件（第1批目录调研确认）；本文件补充 GT 真值精度这一硬约束声明，与已有协议文件体系一致；`configs/base/sim_noise_spec.yaml` 仅有传感器噪声参数（uwb_range_std_m / imu_* / vio_*），无任何 GT 相关字段，说明 GT 真值精度约束是声明在 protocol 层而非 base 层，符合架构分工
⑫已有代码逻辑无需改动 → 现有 `src/liquidloc/metrics/metric_runner.py:919-929` 已正确使用 gt_pos/gt_yaw 进行误差计算，且不引入任何 GT 噪声项；S9 校验脚本已能对照 manifest 中的 gt_pos/gt_yaw 字段验证 schema 完整性；本条 Gate 仅作声明式验证，不对运行时行为做干预

**审查**: 逐行核对 `configs/protocol/gt_truth_precision.yaml`（11 个必填字段命中条款要求）：statement_name="GT 真值精度声明"逐字匹配条款标题；constraint_level="硬约束"逐字匹配条款约束等级；gt_pos_nature="无噪声理想真值"、gt_yaw_nature="无噪声理想真值"逐字匹配条款原文；generation_method="由轨迹几何精确生成"逐字匹配条款原文；gt_uncertainty=0（数值零）逐字匹配条款原文；no_gt_uncertainty_propagation=true、no_gt_noise_in_budget=true 满足禁令要求；error_budget_components="只含传感器噪声×GDOP 项"逐字匹配条款原文；gt_pos_field="gt_pos"、gt_yaw_field="gt_yaw" 对齐已有字段名；manual_ref 指向手册第310行满足真源引用要求；clause_id/clause_name/rationale/effective 完整。Gate 函数 11 项校验全部通过（PASSED）。问题: 无。修复: 无需修复。所有字段与条款原文逐字对应，GT 真值精度声明约束完整。

**验证**: 命令: `python -c "import sys; sys.path.insert(0,'src'); from liquidloc.analysis.clause_gates import p50_gt_truth_precision; r=p50_gt_truth_precision(); print('PASSED' if r['passed'] else 'FAILED'); print('code:', r['code']); print('detail:', r['detail'])"` → 预期: PASSED; code: P50-gt-truth-precision; detail: cfg_exists=True; checks=statement_name=True; constraint_level=True; gt_pos_nature=True; gt_yaw_nature=True; generation_method=True; gt_uncertainty=True; no_gt_uncertainty_propagation=True; error_budget_components=True; no_gt_noise_in_budget=True; manual_exists=True → 实际输出: PASSED; code: P50-gt-truth-precision; detail: cfg_exists=True; checks=statement_name=True; constraint_level=True; gt_pos_nature=True; gt_yaw_nature=True; generation_method=True; gt_uncertainty=True; no_gt_uncertainty_propagation=True; error_budget_components=True; no_gt_noise_in_budget=True; manual_exists=True

**证据**: `configs/protocol/gt_truth_precision.yaml`; `src/liquidloc/analysis/clause_gates.py`（p50_gt_truth_precision 函数定义行 5163-5228、_ALL 注册行 5197）; `configs/protocol/` 目录（已有档位协议配置文件：axis_a_sync.yaml、axis_n_nlos.yaml、protocol_reference_boundary.yaml 等）; `configs/base/sim_noise_spec.yaml`（仅含传感器噪声参数，无 GT 相关字段，架构分工验证）; `.audit/report_stage1_clause50.md`; `docs/异步实验_执行手册_终版v15_总纲条款版.md` 第 310 行（条款 [50] 原文）
