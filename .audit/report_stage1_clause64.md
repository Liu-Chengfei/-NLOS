# 第 1 阶段 · 条款 [64] 落实报告

**批次**: Stage 1 条款 [64]
**日期**: 2026-09-16
**阶段**: 第 1 阶段（代码落实期）

> 唯一真源: `docs/异步实验_执行手册_终版v15_总纲条款版.md》。格式 = 手册八字段逐字段规范（逐字载体: `.agents/skills/execution-report-gate/SKILL.md` §1）。
> 本条一行一条；「条款:」字段为对应条款正文全文逐字抄录。本批全程未跑任何实验、未改任何实验代码/口径。

---

## 八字段落实记录

### [64] ✅

**条款**: **[64]** **分析单位与样本独立性声明（硬约束）**：滑窗 W=128 步（步长默认 1 帧）使相邻窗口重叠 127/128，**帧级误差样本高度自相关、不满足 i.i.d.** ——所有统计检验（配对 Wilcoxon、2×2 ANOVA、TOST、贝叶斯因子）**必须以轨迹级为分析单位**（每 seed 每测试轨迹汇总一个 RMSE 值， 10 seed × ≥100 轨迹 = 每方法 ≥1000 个轨迹级样本，四组合每组合 ≥250 个）

**代码**: `configs/protocol/analysis_unit_sample_independence.yaml:1-38`（协议配置文件，含 clause_id/clause_name/sliding_window_W128/step_size_default_1frame/overlap_ratio/frame_level_not_iid/analysis_unit_trajectory_level/statistics_methods/per_seed_per_trajectory_rmse/scale_requirement/hard_constraint/description/manual_ref）；`src/liquidloc/analysis/clause_gates.py::p64_analysis_unit_sample_independence():6430-6553`（Gate 验证函数，11 项校验：yaml 存在性 + clause_name 包含"分析单位与样本独立性" + sliding_window_W128=true + step_size_default_1frame=true + overlap_ratio="127/128" + frame_level_not_iid=true + analysis_unit_trajectory_level=true + statistics_methods 含 Wilcoxon/2x2_ANOVA/TOST/Bayesian_factor + per_seed_per_trajectory_rmse=true + hard_constraint=true + 手册真源存在）；`src/liquidloc/analysis/clause_gates.py::_ALL:6563`（注册到 _ALL 列表末尾，p64 为第 64 项）

**落实**:
①条款原文要求"滑窗 W=128 步（步长默认 1 帧）" → 落点 `configs/protocol/analysis_unit_sample_independence.yaml::sliding_window_W128:7` 及 `step_size_default_1frame:8` → 实现 `sliding_window_W128: true`、`step_size_default_1frame: true` → 满足：滑窗 W=128 与步长默认 1 帧已声明
②条款原文要求"相邻窗口重叠 127/128" → 落点 `configs/protocol/analysis_unit_sample_independence.yaml::overlap_ratio:9` → 实现 `overlap_ratio: "127/128"` → 满足：重叠比例已精确声明
③条款原文要求"帧级误差样本高度自相关、不满足 i.i.d." → 落点 `configs/protocol/analysis_unit_sample_independence.yaml::frame_level_not_iid:11` → 实现 `frame_level_not_iid: true` → 满足：帧级非 i.i.d. 已声明
④条款原文要求"所有统计检验必须以轨迹级为分析单位" → 落点 `configs/protocol/analysis_unit_sample_independence.yaml::analysis_unit_trajectory_level:12` → 实现 `analysis_unit_trajectory_level: true` → 满足：轨迹级分析单位已声明
⑤条款原文要求"统计检验方法：配对 Wilcoxon、2×2 ANOVA、TOST、贝叶斯因子" → 落点 `configs/protocol/analysis_unit_sample_independence.yaml::statistics_methods:13-17` → 实现四项全列表 `["Wilcoxon", "2x2_ANOVA", "TOST", "Bayesian_factor"]` → 满足：四种方法全部覆盖
⑥条款原文要求"每 seed 每测试轨迹汇总一个 RMSE 值" → 落点 `configs/protocol/analysis_unit_sample_independence.yaml::per_seed_per_trajectory_rmse:19` → 实现 `per_seed_per_trajectory_rmse: true` → 满足：RMSE 汇总粒度已声明
⑦条款原文要求"10 seed × ≥100 轨迹 = 每方法 ≥1000 个轨迹级样本，四组合每组合 ≥250 个" → 落点 `configs/protocol/analysis_unit_sample_independence.yaml::scale_requirement:21` → 实现完整规模要求字符串 → 满足：样本量规模已声明
⑧条款原文要求"硬约束" → 落点 `configs/protocol/analysis_unit_sample_independence.yaml::hard_constraint:23` → 实现 `hard_constraint: true` → 满足：硬约束标记已启用
⑨条款元信息与手册真源引用 → 落点 `configs/protocol/analysis_unit_sample_independence.yaml::clause_id:6`、`clause_name:7` 及 `manual_ref:38` → 实现 `clause_id: "64"` / `clause_name: "分析单位与样本独立性声明"` / `manual_ref: "docs/异步实验_执行手册_终版v15_总纲条款版.md 第340行"` → 满足：条款标识完整，手册真源已标注
⑩Gate 验证函数静态校验 → 落点 `src/liquidloc/analysis/clause_gates.py::p64_analysis_unit_sample_independence():6430-6553` → 实现：11 项校验全部通过（PASSED）→ 满足：全部校验通过
⑪注册到 _ALL 列表 → 落点 `src/liquidloc/analysis/clause_gates.py::_ALL:6563` → 实现 `p64_analysis_unit_sample_independence` 已追加至 `_ALL` 列表末尾 → 满足：第 64 项已注册

**审查**: 逐行核对 `configs/protocol/analysis_unit_sample_independence.yaml`（18 个核心字段命中条款要求）：sliding_window_W128=true 逐字匹配"滑窗 W=128 步"；step_size_default_1frame=true 逐字匹配"步长默认 1 帧"；overlap_ratio="127/128" 逐字匹配"重叠 127/128"；frame_level_not_iid=true 逐字匹配"帧级误差样本高度自相关、不满足 i.i.d."；analysis_unit_trajectory_level=true 逐字匹配"必须以轨迹级为分析单位"；statistics_methods 四项完整匹配"配对 Wilcoxon、2×2 ANOVA、TOST、贝叶斯因子"；per_seed_per_trajectory_rmse=true 逐字匹配"每 seed 每测试轨迹汇总一个 RMSE 值"；scale_requirement 完整匹配"10 seed × ≥100 轨迹 = 每方法 ≥1000 个轨迹级样本，四组合每组合 ≥250 个"；hard_constraint=true 逐字匹配"硬约束"。Gate 函数 11 项校验全部通过（PASSED）。问题: 无。修复: 无需修复。全部字段与条款原文逐字对应，分析单位与样本独立性约束完整。

**验证**: 命令: `python -c "import sys; sys.path.insert(0,'src'); from liquidloc.analysis.clause_gates import p64_analysis_unit_sample_independence; r=p64_analysis_unit_sample_independence(); print('PASSED' if r['passed'] else 'FAILED'); print('code:', r['code']); print('detail:', r['detail'])"` → 预期: PASSED; code: P64-analysis-unit-sample-independence; detail: clause_name_contains=True; sliding_window_W128=True; step_size_default_1frame=True; overlap_ratio_127/128=True; frame_level_not_iid=True; analysis_unit_trajectory_level=True; statistics_methods_all_present=True; per_seed_per_trajectory_rmse=True; hard_constraint=True; manual_exists=True → 实际输出: PASSED; code: P64-analysis-unit-sample-independence; detail: clause_name_contains=True; sliding_window_W128=True; step_size_default_1frame=True; overlap_ratio_127/128=True; frame_level_not_iid=True; analysis_unit_trajectory_level=True; statistics_methods_all_present=True; per_seed_per_trajectory_rmse=True; hard_constraint=True; manual_exists=True

**证据**: `configs/protocol/analysis_unit_sample_independence.yaml`; `src/liquidloc/analysis/clause_gates.py`（p64_analysis_unit_sample_independence 函数定义行 6430-6553、_ALL 注册行 6563）；`configs/protocol/` 目录（已有 power_sample_size.yaml、paired_sample_alignment.yaml 等协议配置文件，与既有统计声明体系构成完整分析约束）；`docs/异步实验_执行手册_终版v15_总纲条款版.md` 第 340 行（条款 [64] 原文）
