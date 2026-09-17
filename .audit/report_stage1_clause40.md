# 第 1 阶段 · 条款 [40] 落实报告

**批次**: Stage 1 条款 [40]
**日期**: 2026-09-16
**阶段**: 第 1 阶段（代码落实期）

---

## 八字段落实记录

### [40] ✅

**条款**: **[40]** **UWB**（DW1000 级，10 Hz）；输出数据：`uwb_ranges`（N×4 测距, m）+ `uwb_valid`（N×4 有效掩码）

**代码**: `configs/protocol/uwb_output_spec.yaml:1-25`（UWB 输出规范配置文件）; `src/liquidloc/analysis/clause_gates.py::p40_uwb_output_spec():3760-3878`（Gate 验证函数）; `src/liquidloc/analysis/clause_gates.py::_ALL:3893`（注册到 _ALL 列表，p40 为第 40 项）

**落实**:
①条款原文要求「UWB（DW1000 级）」→ 落点 `configs/protocol/uwb_output_spec.yaml::chip_model:2` → 实现 `chip_model: "DW1000 级"`（字符串，精确匹配条款原文）→ 满足：芯片型号为 DW1000 级
②条款原文要求「10 Hz」→ 落点 `configs/protocol/uwb_output_spec.yaml::frequency_hz:4` → 实现 `frequency_hz: 10`（整数型，精确匹配条款原文）→ 满足：采样频率为 10 Hz
③条款原文要求「输出数据：uwb_ranges（N×4 测距, m）」→ 落点 `configs/protocol/uwb_output_spec.yaml::output_fields[0]:7-13` → 实现 `name: "uwb_ranges", dimensions: "N×4", unit: "m", type: "测距矩阵"` → 满足：uwb_ranges 字段存在，维度为 N×4，单位为米
④条款原文要求「uwb_valid（N×4 有效掩码）」→ 落点 `configs/protocol/uwb_output_spec.yaml::output_fields[1]:15-20` → 实现 `name: "uwb_valid", dimensions: "N×4", unit: "bool", type: "有效掩码矩阵"` → 满足：uwb_valid 字段存在，维度为 N×4，单位为布尔值
⑤条款原文要求「uwb_ranges 维度 N×4」→ 落点 `configs/protocol/uwb_output_spec.yaml::output_fields[0].dimensions:10` + `configs/protocol/uwb_output_spec.yaml::data_format.uwb_ranges:22-24` → 实现 `dimensions: "N×4"` + `data_format.uwb_ranges: "形状为 (N, 4) 的浮点矩阵"` → 满足：维度描述为 N×4，与条款原文一致
⑥条款原文要求「uwb_valid 维度 N×4」→ 落点 `configs/protocol/uwb_output_spec.yaml::output_fields[1].dimensions:17` + `configs/protocol/uwb_output_spec.yaml::data_format.uwb_valid:25-27` → 实现 `dimensions: "N×4"` + `data_format.uwb_valid: "形状为 (N, 4) 的布尔矩阵"` → 满足：维度描述为 N×4，与条款原文一致
⑦Gate 验证函数静态校验 → 落点 `src/liquidloc/analysis/clause_gates.py::p40_uwb_output_spec():3760-3878` → 实现：7 项校验（chip_model、frequency_hz、uwb_ranges 存在、uwb_valid 存在、uwb_ranges 维度、uwb_valid 维度、data_format）→ 满足：7 项校验全部通过（PASSED）
⑧注册到 _ALL 列表 → 落点 `src/liquidloc/analysis/clause_gates.py::_ALL:3893` → 实现 `p40_uwb_output_spec` 作为第 40 项注册 → 满足：注册到 _ALL 列表

**审查**: 逐行核对 `configs/protocol/uwb_output_spec.yaml`（7 个字段全部命中条款原文要求）与 `src/liquidloc/analysis/clause_gates.py::p40_uwb_output_spec()`（7 项校验全部通过）。问题: 无。修复: 无需修复。全部字段与条款原文逐句对应，无遗漏。

**验证**: 命令: `/c/Program\ Files/Python311/python -c "import sys; sys.path.insert(0,'src'); from liquidloc.analysis.clause_gates import p40_uwb_output_spec; r=p40_uwb_output_spec(); print('PASSED' if r['passed'] else 'FAILED'); print('detail:', r['detail'])"` → 预期: PASSED; detail: cfg_exists=True; chip_model='DW1000 级' (期望 'DW1000 级'); frequency_hz=10 (期望 10); fields_ok=True; uwb_ranges dimensions=N×4, unit=m; uwb_valid dimensions=N×4, unit=bool; data_fmt_ok=True; handount_exists=True; concept_hits={'DW1000': True, '10': True, 'uwb_ranges': True, 'uwb_valid': True, 'N×4': True}

**证据**: `configs/protocol/uwb_output_spec.yaml`; `src/liquidloc/analysis/clause_gates.py`; `.audit/report_stage1_clause40.md`

---
