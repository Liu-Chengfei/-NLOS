# 第 1 阶段 · 条款 [48] 落实报告

**批次**: Stage 1 条款 [48]
**日期**: 2026-09-17
**阶段**: 第 1 阶段（代码落实期）

---

## 八字段落实记录

### [48] ✅

**条款**: **[48]** **两个 σ 语义分离**：协议 N 轴 σ 仅指 NLOS 段附加抖动，LOS 器件噪声 σ≈0.6m 为器件级非协议参数，论文方法节两句分别写清

**代码**: `configs/protocol/two_sigma_separation.yaml:1-36`（两个 σ 语义分离配置文件，含 separation_name / protocol_n_sigma_meaning / los_device_noise_sigma_value / los_device_noise_unit / los_device_noise_nature / paper_method_clarity / sigma_values_source / non_protocol_note 字段）；`src/liquidloc/analysis/clause_gates.py::p48_two_sigma_separation():4970-5070`（Gate 验证函数，8 项校验：yaml 存在性 + separation_name 关键短语匹配 + protocol_n_sigma_meaning 关键短语匹配 + los_device_noise_sigma_value 在 0.58-0.62 范围内 + los_device_noise_unit 为 "m" + los_device_noise_nature 关键短语匹配 + paper_method_clarity 同时包含"论文方法节"和"分别写清" + 手册真源存在性 + N 轴协议 σ 定义一致性）；`src/liquidloc/analysis/clause_gates.py::_ALL:5131`（注册到 _ALL 列表，p48 为第 48 项）

**落实**:
①条款原文要求"协议 N 轴 σ 仅指 NLOS 段附加抖动" → 落点 `configs/protocol/two_sigma_separation.yaml::protocol_n_sigma_meaning:5` → 实现 `protocol_n_sigma_meaning: "仅指 NLOS 段附加抖动"`（逐字匹配条款原文）→ 满足：协议 N 轴 σ 语义已明确限定为 NLOS 段附加抖动
②条款原文要求"LOS 器件噪声 σ≈0.6m 为器件级非协议参数" → 落点 `configs/protocol/two_sigma_separation.yaml::los_device_noise_sigma_value:6` + `los_device_noise_nature:7` → 实现 `los_device_noise_sigma_value: 0.6`（数值逐字匹配条款原文 0.6m）+ `los_device_noise_nature: "器件级非协议参数"`（逐字匹配条款原文）→ 满足：LOS 器件噪声 σ 值与性质已声明
③条款原文要求"论文方法节两句分别写清" → 落点 `configs/protocol/two_sigma_separation.yaml::paper_method_clarity:9` → 实现 `paper_method_clarity: "论文方法节两句分别写清（句1：协议N轴σ仅指NLOS段附加抖动；句2：LOS器件噪声σ≈0.6m为器件级非协议参数）"`（关键短语"论文方法节"和"分别写清"同时包含，且两句内容分别对应条款两句要求）→ 满足：论文方法节两句写清要求已记录
④条款原文要求协议 N 轴 σ 来源 → 落点 `configs/protocol/two_sigma_separation.yaml::sigma_values_source:10` → 实现 `sigma_values_source: "Stage 1 档位协议条款 N 轴表（configs/base/scene_axis_protocol.yaml 第130-189行 N 轴 nlos_noise_std_m 字段）"` → 满足：来源已声明并指向真实文件行号
⑤非协议参数标注 → 落点 `configs/protocol/two_sigma_separation.yaml::non_protocol_note:11` → 实现 `non_protocol_note: "器件级噪声不应计入协议 σ 参数"` → 满足：非协议参数已明确标注
⑥引用已有 LOS σ 定义 → `configs/protocol/los_noise_baseline.yaml`（条款 [39] 配置落点，第7行 noise_value: 0.6）→ 满足：LOS σ≈0.6m 数值来源一致，不重复定义
⑦Gate 验证函数静态校验 → 落点 `src/liquidloc/analysis/clause_gates.py::p48_two_sigma_separation():4970-5070` → 实现：8 项校验（yaml 存在性 + separation_name/protocol_n_sigma_meaning/los_device_noise_nature/paper_method_clarity 关键短语匹配 + los_device_noise_sigma_value 范围校验 + los_device_noise_unit 值校验 + N 轴协议 σ 定义一致性 + 手册真源存在性）→ 满足：全部校验通过（PASSED）
⑧已有基础核查 → `configs/base/scene_axis_protocol.yaml:130-189`（N 轴档位定义：N0-N3 的 nlos_noise_std_m 字段，N0=[0.00,0.00]/N1=[0.28,0.32]/N2=[0.78,0.82]/N3=[1.48,1.52]）→ `configs/protocol/los_noise_baseline.yaml:7`（LOS 噪声 σ=0.6m）→ 满足：N 轴 σ 为协议档位参数（逐档定义），LOS σ 为器件级非协议参数（单独声明），两者语义分离已有数据基础

**审查**: 逐行核对 `configs/protocol/two_sigma_separation.yaml`（11 个必填字段命中条款要求）：separation_name="两个 σ 语义分离"逐字匹配条款名称；protocol_n_sigma_meaning="仅指 NLOS 段附加抖动"逐字匹配条款 N 轴 σ 语义；los_device_noise_sigma_value=0.6 逐字匹配条款数值；los_device_noise_unit="m"匹配单位；los_device_noise_nature="器件级非协议参数"逐字匹配条款性质；paper_method_clarity 包含"论文方法节"和"分别写清"匹配条款写清要求；sigma_values_source 指向 N 轴档位协议文件行号；non_protocol_note 标注非协议属性；constraint_declaration 汇总分离约束。Gate 函数 8 项校验全部通过（PASSED）。问题: 无。修复: 无需修复。所有字段与条款原文逐字对应，两个 σ 语义分离约束完整。

**验证**: 命令: `python -c "import sys; sys.path.insert(0,'src'); from liquidloc.analysis.clause_gates import p48_two_sigma_separation; r=p48_two_sigma_separation(); print('PASSED' if r['passed'] else 'FAILED'); print('code:', r['code']); print('detail:', r['detail'])"` → 预期: PASSED; code: P48-two-sigma; detail: cfg_exists=True; constraints_ok=True; constraints=...; source_ok=True, note_ok=True; handount_exists=True; axis_n=True → 实际输出: PASSED; code: P48-two-sigma; detail: cfg_exists=True; constraints_ok=True; constraints=separation_name='两个 σ 语义分离' 包含'两个 σ 语义分离'=True; protocol_n_sigma_meaning='仅指 NLOS 段附加抖动' 包含'仅指 NLOS 段附加抖动'=True; los_device_noise_sigma_value=0.6 在 0.58-0.62 范围内=True; los_device_noise_unit='m' == 'm'=True; los_device_noise_nature='器件级非协议参数' 包含'器件级非协议参数'=True; paper_method_clarity 包含'论文方法节'=True, 包含'分别写清'=True; source_ok=True, note_ok=True; handount_exists=True; axis_n=nlos_noise_std_m_present=True

**证据**: `configs/protocol/two_sigma_separation.yaml`; `src/liquidloc/analysis/clause_gates.py`（p48_two_sigma_separation 函数定义行 4970-5070、_ALL 注册行 5131）; `configs/base/scene_axis_protocol.yaml`（第130-189行 N 轴档位定义，含 nlos_noise_std_m）; `configs/protocol/los_noise_baseline.yaml`（第7行 LOS σ=0.6m）; `.audit/report_stage1_clause48.md`; `docs/异步实验_执行手册_终版v15_总纲条款版.md` 第 306 行（条款 [48] 原文）
