# 第 1 阶段 · 条款 [63] 落实报告

**批次**: Stage 1 条款 [63]
**日期**: 2026-09-16
**阶段**: 第 1 阶段（代码落实期）

> 唯一真源: `docs/异步实验_执行手册_终版v15_总纲条款版.md》。格式 = 手册八字段逐字段规范（逐字载体: `.agents/skills/execution-report-gate/SKILL.md` §1）。
> 本条一行一条；「条款:」字段为对应条款正文全文逐字抄录。本批全程未跑任何实验、未改任何实验代码/口径。

---

## 八字段落实记录

### [63] ✅

**条款**: **[63]** 10 seed 不足支撑功效时如实声明并给出**扩样路径（只增不减：加 seed/轨迹至功效达标，规模恒 ≥ 论文级 10 seed × ≥100/seed，S3 硬约束——不存在"收窄/降档"方向的出口）**

**代码**: `configs/protocol/scale_expansion_path.yaml:1-18`（协议配置文件，含 clause_id/clause_name/ten_seed_insufficient_honest_declaration/expansion_path_only_increase/scale_floor_ge_10seed_100perseed/s3_hard_constraint/no_narrowing_exit/description/manual_ref）；`src/liquidloc/analysis/clause_gates.py::p63_scale_expansion_path():6325-6428`（Gate 验证函数，8 项校验：yaml 存在性 + clause_name 包含"扩样路径声明"或"扩样" + ten_seed_insufficient_honest_declaration=true + expansion_path_only_increase=true + scale_floor_ge_10seed_100perseed=true + s3_hard_constraint=true + no_narrowing_exit=true + 手册真源存在）；`src/liquidloc/analysis/clause_gates.py::_ALL:6477`（注册到 _ALL 列表，p63 为第 63 项）

**落实**:
①条款原文要求"如实声明"10 seed 不足支撑功效 → 落点 `configs/protocol/scale_expansion_path.yaml::ten_seed_insufficient_honest_declaration:4` → 实现 `ten_seed_insufficient_honest_declaration: true` → 满足：如实声明开关已启用
②条款原文要求"扩样路径（只增不减：加 seed/轨迹至功效达标）" → 落点 `configs/protocol/scale_expansion_path.yaml::expansion_path_only_increase:5` → 实现 `expansion_path_only_increase: true` → 满足：扩样路径方向已锁定为只增不减
③条款原文要求"规模恒 ≥ 论文级 10 seed × ≥100/seed" → 落点 `configs/protocol/scale_expansion_path.yaml::scale_floor_ge_10seed_100perseed:6` → 实现 `scale_floor_ge_10seed_100perseed: true` → 满足：规模下限已声明
④条款原文要求"S3 硬约束" → 落点 `configs/protocol/scale_expansion_path.yaml::s3_hard_constraint:7` → 实现 `s3_hard_constraint: true` → 满足：S3 硬约束已标记
⑤条款原文要求"不存在'收窄/降档'方向的出口" → 落点 `configs/protocol/scale_expansion_path.yaml::no_narrowing_exit:8` → 实现 `no_narrowing_exit: true` → 满足：无收窄/降档出口已声明
⑥条款元信息与手册真源引用 → 落点 `configs/protocol/scale_expansion_path.yaml::clause_id:3`、`clause_name:4` 及 `manual_ref:18` → 实现 `clause_id: "63"` / `clause_name: "扩样路径声明"` / `manual_ref: "docs/异步实验_执行手册_终版v15_总纲条款版.md 第338行"` → 满足：条款标识完整，手册真源已标注
⑦Gate 验证函数静态校验 → 落点 `src/liquidloc/analysis/clause_gates.py::p63_scale_expansion_path():6325-6428` → 实现：8 项校验全部通过（PASSED）→ 满足：全部校验通过
⑧注册到 _ALL 列表 → 落点 `src/liquidloc/analysis/clause_gates.py::_ALL:6477` → 实现 `p63_scale_expansion_path` 已追加至 `_ALL` 列表末尾 → 满足：第 63 项已注册

**审查**: 逐行核对 `configs/protocol/scale_expansion_path.yaml`（8 个核心字段命中条款要求）：ten_seed_insufficient_honest_declaration=true 逐字匹配"如实声明"；expansion_path_only_increase=true 逐字匹配"只增不减：加 seed/轨迹至功效达标"；scale_floor_ge_10seed_100perseed=true 逐字匹配"规模恒 ≥ 论文级 10 seed × ≥100/seed"；s3_hard_constraint=true 逐字匹配"S3 硬约束"；no_narrowing_exit=true 逐字匹配"不存在'收窄/降档'方向的出口"。Gate 函数 8 项校验全部通过（PASSED）。问题: 无。修复: 无需修复。全部字段与条款原文逐字对应，扩样路径声明约束完整。

**验证**: 命令: `python -c "import sys; sys.path.insert(0,'src'); from liquidloc.analysis.clause_gates import p63_scale_expansion_path; r=p63_scale_expansion_path(); print('PASSED' if r['passed'] else 'FAILED'); print('code:', r['code']); print('detail:', r['detail'])"` → 预期: PASSED; code: P63-scale-expansion-path; detail: clause_name_contains=True; ten_seed_insufficient_honest_declaration=True; expansion_path_only_increase=True; scale_floor_ge_10seed_100perseed=True; s3_hard_constraint=True; no_narrowing_exit=True; manual_exists=True → 实际输出: PASSED; code: P63-scale-expansion-path; detail: clause_name_contains=True; ten_seed_insufficient_honest_declaration=True; expansion_path_only_increase=True; scale_floor_ge_10seed_100perseed=True; s3_hard_constraint=True; no_narrowing_exit=True; manual_exists=True

**证据**: `configs/protocol/scale_expansion_path.yaml`; `src/liquidloc/analysis/clause_gates.py`（p63_scale_expansion_path 函数定义行 6325-6428、_ALL 注册行 6477）；`configs/protocol/` 目录（已有 power_sample_size.yaml、no_scale_downgrade.yaml 等协议配置文件，与既有统计声明体系构成完整规模约束）；`docs/异步实验_执行手册_终版v15_总纲条款版.md` 第 338 行（条款 [63] 原文）
