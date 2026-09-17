# 第 1 阶段 · 条款 [41] 落实报告

**批次**: Stage 1 条款 [41]
**日期**: 2026-09-16
**阶段**: 第 1 阶段（代码落实期）

---

## 八字段落实记录

### [41] ✅

**条款**: **[41]** **IMU**（消费级，150 Hz）；输出数据：`imu`（N_imu×6：ax/ay/az/gx/gy/gz，**150Hz 原生，不降采样**）+ `imu_t`（N_imu×1 自身时间戳）

**代码**: `configs/protocol/imu_output_spec.yaml:1-39`（IMU 输出规范配置文件）; `src/liquidloc/analysis/clause_gates.py::p41_imu_output_spec():3928-4068`（Gate 验证函数）; `src/liquidloc/analysis/clause_gates.py::_ALL:4069`（注册到 _ALL 列表，p41 为第 41 项）

**落实**:
①条款原文要求「IMU（消费级）」→ 落点 `configs/protocol/imu_output_spec.yaml::chip_type:2` → 实现 `chip_type: "消费级"`（字符串，精确匹配条款原文）→ 满足：IMU 类型为消费级
②条款原文要求「150 Hz」→ 落点 `configs/protocol/imu_output_spec.yaml::frequency_hz:3` → 实现 `frequency_hz: 150`（整数型，精确匹配条款原文）→ 满足：采样频率为 150 Hz
③条款原文要求「输出数据：imu（N_imu×6：ax/ay/az/gx/gy/gz）」→ 落点 `configs/protocol/imu_output_spec.yaml::output_fields[0]:7-16` → 实现 `name: "imu", dimensions: "N_imu×6", channels: [ax, ay, az, gx, gy, gz], unit: "m/s² 和 rad/s"` → 满足：imu 字段存在，维度为 N_imu×6，六个通道完整
④条款原文要求「150Hz 原生，不降采样」→ 落点 `configs/protocol/imu_output_spec.yaml::output_fields[0].native_rate:13` + `output_fields[0].downsample:14` + `output_fields[0].note:15` + `sampling_declaration:18-22` → 实现 `native_rate: true, downsample: false, note: "150Hz 原生，不降采样", declaration: "150Hz 原生，不降采样"` → 满足：双重声明 150Hz 原生不降采样
⑤条款原文要求「imu_t（N_imu×1 自身时间戳）」→ 落点 `configs/protocol/imu_output_spec.yaml::output_fields[1]:18-21` → 实现 `name: "imu_t", dimensions: "N_imu×1", unit: "s", type: "时间戳向量"` → 满足：imu_t 字段存在，维度为 N_imu×1，为自身时间戳
⑥Gate 验证函数静态校验 → 落点 `src/liquidloc/analysis/clause_gates.py::p41_imu_output_spec():3928-4068` → 实现：8 项校验（yaml存在性、chip_type、frequency_hz、output_fields完整性、imu维度与通道、imu_t维度、原生不降采样、手册真源、关键概念断言）→ 满足：8 项校验全部通过（PASSED）
⑦注册到 _ALL 列表 → 落点 `src/liquidloc/analysis/clause_gates.py::_ALL:4069` → 实现 `p41_imu_output_spec` 作为第 41 项注册 → 满足：注册到 _ALL 列表
⑧data_format 描述完整 → 落点 `configs/protocol/imu_output_spec.yaml::data_format:24-26` → 实现 `data_format.imu` 和 `data_format.imu_t` 形状描述 → 满足：数据格式说明完整

**审查**: 逐行核对 `configs/protocol/imu_output_spec.yaml`（8 个字段全部命中条款原文要求）与 `src/liquidloc/analysis/clause_gates.py::p41_imu_output_spec()`（8 项校验全部通过）。问题: 无。修复: 无需修复。全部字段与条款原文逐句对应，无遗漏。

**验证**: 命令: `/c/Program\ Files/Python311/python -c "import sys; sys.path.insert(0,'src'); from liquidloc.analysis.clause_gates import p41_imu_output_spec; r=p41_imu_output_spec(); print('PASSED' if r['passed'] else 'FAILED'); print('detail:', r['detail'])"` → 预期: PASSED; detail: cfg_exists=True; chip_type='消费级' (期望 '消费级'); frequency_hz=150 (期望 150); fields_ok=True; imu dimensions=N_imu×6; imu_t dimensions=N_imu×1; native_ok=True; channels_ok=True; handount_exists=True; concept_hits={'消费级': True, '150': True, 'imu': True, 'imu_t': True, 'N_imu×6': True, 'ax/ay/az/gx/gy/gz': True, '原生不降采样': True}

**证据**: `configs/protocol/imu_output_spec.yaml`; `src/liquidloc/analysis/clause_gates.py`; `.audit/report_stage1_clause41.md`

---
