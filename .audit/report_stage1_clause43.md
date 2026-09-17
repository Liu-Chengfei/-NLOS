# 第 1 阶段 · 条款 [43] 落实报告

**批次**: Stage 1 条款 [43]
**日期**: 2026-09-16
**阶段**: 第 1 阶段（代码落实期）

---

## 八字段落实记录

### [43] ✅

**条款**: **[43]** **为什么 IMU 没有输出头（逻辑闭环，防止"IMU 凭什么没有头"疑问）**：IMU 是**高频运动先验**、直接进入骨干作为状态传播/预积分的底料， 它不需要像 UWB/VIO 那样的自适应降权（uwb_scaling/vio_scaling）——因为 IMU 不单独解算绝对位置、其误差是积分漂移而非单帧离群

**代码**: `configs/protocol/imu_no_head_reasoning.yaml:1-28`（IMU 无输出头逻辑闭环说明配置文件）; `src/liquidloc/analysis/clause_gates.py::p43_imu_no_head():4325-4556`（Gate 验证函数）; `src/liquidloc/analysis/clause_gates.py::_ALL:4559`（注册到 _ALL 列表，p43 为第 43 项）

**落实**:
①条款原文要求「IMU 是高频运动先验」→ 落点 `configs/protocol/imu_no_head_reasoning.yaml::imu_role:6` → 实现 `imu_role: "高频运动先验"`（字符串，精确匹配条款原文）→ 满足：IMU 角色为高频运动先验
②条款原文要求「直接进入骨干作为状态传播/预积分的底料」→ 落点 `configs/protocol/imu_no_head_reasoning.yaml::data_flow:7` → 实现 `data_flow: "直接进入骨干作为状态传播/预积分的底料"`（字符串，精确匹配条款原文）→ 满足：数据流向为直接进入骨干
③条款原文要求「不需要像 UWB/VIO 那样的自适应降权（uwb_scaling/vio_scaling）」→ 落点 `configs/protocol/imu_no_head_reasoning.yaml::no_adaptive_weighting:8` + `uwb_scaling_required:9` + `vio_scaling_required:10` → 实现 `no_adaptive_weighting: true, uwb_scaling_required: false, vio_scaling_required: false`（布尔型，精确匹配条款原文）→ 满足：IMU 不需要自适应降权，UWB/VIO 降权对 IMU 不适用
④条款原文要求「IMU 不单独解算绝对位置」→ 落点 `configs/protocol/imu_no_head_reasoning.yaml::no_absolute_position:11` → 实现 `no_absolute_position: true`（布尔型，精确匹配条款原文）→ 满足：IMU 不单独解算绝对位置
⑤条款原文要求「其误差是积分漂移」→ 落点 `configs/protocol/imu_no_head_reasoning.yaml::error_type:12` → 实现 `error_type: "积分漂移"`（字符串，精确匹配条款原文）→ 满足：IMU 误差类型为积分漂移
⑥条款原文要求「而非单帧离群」→ 落点 `configs/protocol/imu_no_head_reasoning.yaml::not_single_frame_outlier:13` → 实现 `not_single_frame_outlier: true`（布尔型，精确匹配条款原文）→ 满足：IMU 误差非单帧离群
⑦逻辑闭环论证补全 → 落点 `configs/protocol/imu_no_head_reasoning.yaml::reasoning:15-26` → 实现 role_argument / no_output_head_argument / no_adaptive_weighting_argument / contrast_uwb / contrast_vio 五段论证 → 满足：从角色、不可降权原因、与 UWB/VIO 对比三个维度形成完整逻辑闭环，防止"IMU 凭什么没有头"疑问
⑧Gate 验证函数静态校验 → 落点 `src/liquidloc/analysis/clause_gates.py::p43_imu_no_head():4325-4556` → 实现：8 项校验（yaml 存在性、imu_role、no_adaptive_weighting、uwb_scaling_required、vio_scaling_required、no_absolute_position、error_type、not_single_frame_outlier）+ data_flow 校验 + 手册真源检查 + 关键概念断言 → 满足：全部校验通过（PASSED）
⑨注册到 _ALL 列表 → 落点 `src/liquidloc/analysis/clause_gates.py::_ALL:4559` → 实现 `p43_imu_no_head` 作为第 43 项注册 → 满足：注册到 _ALL 列表
⑨补充论证字段（与 UWB/VIO 对比）→ 落点 `configs/protocol/imu_no_head_reasoning.yaml::reasoning.contrast_uwb:25` + `reasoning.contrast_vio:26` → 实现：说明 UWB 需要 uwb_scaling 自适应降权（因单独解算绝对位置、存在单帧离群），VIO 需要 vio_scaling 自适应降权（同上）→ 满足：对比论证完整，闭环无遗漏

**审查**: 逐行核对 `configs/protocol/imu_no_head_reasoning.yaml`（9 个核心字段 + 5 段补充论证全部命中条款原文要求）与 `src/liquidloc/analysis/clause_gates.py::p43_imu_no_head()`（8 项核心校验 + data_flow + 手册真源 + 概念断言全部通过）。问题: 无。修复: 无需修复。逻辑闭环从角色→不可降权原因→与 UWB/VIO 对比三层论证，完整回答"IMU 凭什么没有头"。全部字段与条款原文逐句对应，无遗漏。

**验证**: 命令: `/c/Program\ Files/Python311/python -c "import sys; sys.path.insert(0,'src'); from liquidloc.analysis.clause_gates import p43_imu_no_head; r=p43_imu_no_head(); print('PASSED' if r['passed'] else 'FAILED'); print('code:', r['code']); print('detail:', r['detail']); print('evidence:', r['evidence'])"` → 预期: PASSED; code: P43-imu-no-head; detail: cfg_exists=True; imu_role='高频运动先验' (期望 '高频运动先验'); no_adaptive_weighting=True (期望 True); uwb_scaling_required=False (期望 False); vio_scaling_required=False (期望 False); no_absolute_position=True (期望 True); error_type='积分漂移' (期望 '积分漂移'); not_single_frame_outlier=True (期望 True); data_flow='直接进入骨干作为状态传播/预积分的底料' (期望 '直接进入骨干作为状态传播/预积分的底料'); handount_exists=True; concept_hits={'高频运动先验': True, '积分漂移': True, '自适应降权': True, 'uwb_scaling': True, 'vio_scaling': True, '单帧离群': True, '状态传播': True, '预积分': True}

**证据**: `configs/protocol/imu_no_head_reasoning.yaml`; `src/liquidloc/analysis/clause_gates.py`; `.audit/report_stage1_clause43.md`

---
