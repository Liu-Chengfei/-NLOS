# 第 1 阶段 · 条款 [45] 落实报告

**批次**: Stage 1 条款 [45]
**日期**: 2026-09-17
**阶段**: 第 1 阶段（代码落实期）

---

## 八字段落实记录

### [45] ✅

**条款**: **[45]** **注：N1.5（ρ=20–25%/μ=1.5–2.5m/σ=0.5m）不在本手册档位协议条款档位表内（本手册引用协议 N 轴为 N0–N3）** ——N1.5 属于同机另一实验体系按**其各自对应协议文件**执行的档位（该协议文件 N 轴含 N1.5 标准档）

**代码**: `configs/protocol/n15_boundary.yaml:1-15`（N1.5 边界协议配置文件，含 scope_note / n15_params / current_manual_n_axis / n15_belongs_to_other_system / n15_executed_by_own_protocol / other_system_protocol / other_system_n_axis_includes_n15 / exclusion_reason / manual_ref）；`src/liquidloc/analysis/clause_gates.py::p45_n15_boundary():4663-4756`（Gate 验证函数，7 项校验）；`src/liquidloc/analysis/clause_gates.py::_ALL:4757`（注册到 _ALL 列表，p45 为第 45 项）

**落实**:
①条款原文要求"N1.5 不在本手册档位协议条款档位表内" → 落点 `configs/protocol/n15_boundary.yaml::scope_note:2` → 实现 `scope_note: "N1.5 不在本手册档位协议条款档位表内"`（逐字匹配条款原文）→ 满足：边界声明到位
②条款原文要求 N1.5 参数"ρ=20–25%/μ=1.5–2.5m/σ=0.5m" → 落点 `configs/protocol/n15_boundary.yaml::n15_params:3-6`（rho / mu / sigma）→ 实现 `rho: "20-25%"`, `mu: "1.5-2.5m"`, `sigma: "0.5m"` → 满足：参数完全匹配
③条款原文要求"本手册引用协议 N 轴为 N0–N3" → 落点 `configs/protocol/n15_boundary.yaml::current_manual_n_axis:7` + `exclusion_reason:13` + `docs/异步实验_执行手册_终版v15_总纲条款版.md:300`（条款 [45] 原文）→ 实现 `current_manual_n_axis: [N0, N1, N2, N3]`（不含 N1.5）+ `exclusion_reason: "本手册引用协议 N 轴为 N0–N3，不包含 N1.5"` → 满足：N 轴范围精确限定
④条款原文要求"N1.5 属于同机另一实验体系" → 落点 `configs/protocol/n15_boundary.yaml::n15_belongs_to_other_system:8` → 实现 `n15_belongs_to_other_system: true` → 满足：归属声明到位
⑤条款原文要求"按其各自对应协议文件执行" → 落点 `configs/protocol/n15_boundary.yaml::n15_executed_by_own_protocol:9` + `other_system_protocol:10` → 实现 `n15_executed_by_own_protocol: true` + `other_system_protocol: "其各自对应协议文件"` → 满足：执行方式精确匹配
⑥条款原文要求"该协议文件 N 轴含 N1.5 标准档" → 落点 `configs/protocol/n15_boundary.yaml::other_system_n_axis_includes_n15:11` → 实现 `other_system_n_axis_includes_n15: true` → 满足：N1.5 包含声明到位
⑦Gate 验证函数静态校验 → 落点 `src/liquidloc/analysis/clause_gates.py::p45_n15_boundary():4663-4756` → 实现：7 项校验（yaml 存在性 + scope_note 必含短语 + current_manual_n_axis 精确匹配 [N0,N1,N2,N3] + n15_params 三参数逐一匹配 + n15_belongs_to_other_system 为 true + other_system_protocol 含"各自对应协议文件" + other_system_n_axis_includes_n15 为 true）→ 满足：全部校验通过（PASSED）
⑧已有基础核查 → `configs/base/scene_axis_protocol.yaml:136-188`（仅定义 N0/N1/N2/N3 四档，无 N1.5）；`configs/base/scene_axis_protocol.yaml:436`（`N: ["N0", "N1", "N2", "N3"]`，N 轴不含 N1.5）；`src/liquidloc/analysis/clause_gates.py` _ALL 列表此前末项为 p44_four_output_heads → 满足：基础已确认 N0–N3 无 N1.5，未重复创建

**审查**: 逐行核对 `configs/protocol/n15_boundary.yaml`（15 行，9 个字段全部命中条款原文要求）：scope_note 逐字匹配"N1.5 不在本手册档位协议条款档位表内"；n15_params 三参数（rho=20-25% / mu=1.5-2.5m / sigma=0.5m）精确匹配；current_manual_n_axis=[N0,N1,N2,N3] 不含 N1.5；n15_belongs_to_other_system=true；other_system_protocol="其各自对应协议文件"；other_system_n_axis_includes_n15=true；exclusion_reason 完整说明排除原因。与 `docs/异步实验_执行手册_终版v15_总纲条款版.md:300` 条款 [45] 原文逐句对应。与 `configs/base/scene_axis_protocol.yaml:136-188,436` 一致（仅 N0–N3，无 N1.5）。问题: 无。修复: 无需修复。全部字段与条款原文逐字对应，无遗漏。

**验证**: 命令: `python -c "import sys; sys.path.insert(0,'src'); from liquidloc.analysis.clause_gates import p45_n15_boundary; r=p45_n15_boundary(); print('PASSED' if r['passed'] else 'FAILED'); print('code:', r['code']); print('detail:', r['detail'])"` → 预期: PASSED; code: P45-n15-boundary; detail: scope_note 包含必含短语=True; current_manual_n_axis=['N0','N1','N2','N3'] 不含 N1.5=True; n15_params 全部匹配=True; n15_belongs_to_other_system=True; other_system_protocol 包含"各自对应协议文件"=True; other_system_n_axis_includes_n15=True → 实际输出: PASSED; code: P45-n15-boundary; detail: scope_note='N1.5 不在本手册档位协议条款档位表内', 包含必含短语=True; current_manual_n_axis=['N0', 'N1', 'N2', 'N3'], 期望=['N0', 'N1', 'N2', 'N3'], 不含N1.5, 匹配=True; n15_params=rho=20-25%, mu=1.5-2.5m, sigma=0.5m, 全部匹配=True; n15_belongs_to_other_system=True, 为 true=True; other_system_protocol='其各自对应协议文件', 包含'各自对应协议文件'=True; other_system_n_axis_includes_n15=True, 为 true=True

**证据**: `configs/protocol/n15_boundary.yaml`; `configs/base/scene_axis_protocol.yaml`（第136-188行 N0-N3 定义、第436行 N 轴列表）；`src/liquidloc/analysis/clause_gates.py`; `.audit/report_stage1_clause45.md`; `docs/异步实验_执行手册_终版v15_总纲条款版.md` 第 300 行（条款 [45] 原文）
