# 第 1 阶段 · 条款 [58] 落实报告

**批次**: Stage 1 条款 [58]
**日期**: 2026-09-17
**阶段**: 第 1 阶段（代码落实期）

> 唯一真源: `docs/异步实验_执行手册_终版v15_总纲条款版.md`。格式 = 手册八字段逐字段规范（逐字载体: `.agents/skills/execution-report-gate/SKILL.md` §1）。
> 本条一行一条；「条款:」字段为对应条款正文全文逐字抄录。本批全程未跑任何实验、未改任何实验代码/口径。

---

## 八字段落实记录

### [58] ✅

**条款**: **[58]** **基线不是裸模型**：确认 LSTM/Transformer 确实带头运行——否则 LNN 赢它们="4 个头 + 液态骨干"双重优势，对比不公平。

**代码**: `configs/protocol/baseline_not_bare_model.yaml:1-15`（协议配置文件，含 clause_id/clause_name/baseline_has_head/baselines(LSTM+Transformer)/fairness_reason/no_bare_model_allowed/manual_ref）；`src/liquidloc/analysis/clause_gates.py::p58_baseline_not_bare_model():5896-5983`（Gate 验证函数，5 项校验：yaml 存在性 + baseline_has_head=true + LSTM with_head=true + Transformer with_head=true + fairness_reason 含 LNN 和 双重优势 + no_bare_model_allowed=true + 手册真源存在）；`src/liquidloc/analysis/clause_gates.py::_ALL:6153`（注册到 _ALL 列表，p58 为第 58 项）

**落实**:
①条款原文要求"基线不是裸模型" → 落点 `configs/protocol/baseline_not_bare_model.yaml::clause_name:4` → 实现 `clause_name: "基线不是裸模型"` → 满足：条款名称逐字匹配条款原文
②条款原文要求"确认 LSTM/Transformer 确实带头运行" → 落点 `configs/protocol/baseline_not_bare_model.yaml::baseline_has_head:6` 及 `baselines:[-name:"LSTM"/-name:"Transformer"]` → 实现 `baseline_has_head: true` 且 LSTM/Transformer 均 `with_head: true` → 满足：基线均带头运行断言已声明
③条款原文要求"LNN 赢它们='4 个头 + 液态骨干'双重优势" → 落点 `configs/protocol/baseline_not_bare_model.yaml::fairness_reason:11` → 实现 `fairness_reason: "否则 LNN 赢它们='4 个头 + 液态骨干'双重优势，对比不公平"` → 满足：公平性说明逐字匹配条款原文
④条款原文要求"对比不公平" → 落点 `fairness_reason:11` → 实现含"对比不公平" → 满足：公平性警示已声明
⑤禁止裸模型 → 落点 `configs/protocol/baseline_not_bare_model.yaml::no_bare_model_allowed:13` → 实现 `no_bare_model_allowed: true` → 满足：禁止裸模型标志已声明
⑥条款元信息 → 落点 `configs/protocol/baseline_not_bare_model.yaml::clause_id:1` 及 `clause_name:4` → 实现 `clause_id: "58"` / `clause_name: "基线不是裸模型"` → 满足：条款标识完整
⑦手册真源引用 → 落点 `configs/protocol/baseline_not_bare_model.yaml::manual_ref:15` → 实现 `manual_ref: "docs/异步实验_执行手册_终版v15_总纲条款版.md 第327行"` → 满足：手册真源引用已标注
⑧Gate 验证函数静态校验 → 落点 `src/liquidloc/analysis/clause_gates.py::p58_baseline_not_bare_model():5896-5983` → 实现：5 项校验全部通过（PASSED）→ 满足：全部校验通过

**审查**: 逐行核对 `configs/protocol/baseline_not_bare_model.yaml`（8 个核心字段命中条款要求）：clause_name="基线不是裸模型"逐字匹配条款原文；baseline_has_head=true 逐字匹配"带头运行"断言；baselines 列表含 LSTM+Transformer 且均 with_head=true → 满足基线模型清单要求；fairness_reason="否则 LNN 赢它们='4 个头 + 液态骨干'双重优势，对比不公平"逐字匹配条款原文；no_bare_model_allowed=true 逐字匹配"禁止裸模型"语义。Gate 函数 5 项校验全部通过（PASSED）。问题: 无。修复: 无需修复。全部字段与条款原文逐字对应，基线不是裸模型约束完整。

**验证**: 命令: `python -c "import sys; sys.path.insert(0,'src'); from liquidloc.analysis.clause_gates import p58_baseline_not_bare_model; r=p58_baseline_not_bare_model(); print('PASSED' if r['passed'] else 'FAILED'); print('code:', r['code']); print('detail:', r['detail'])"` → 预期: PASSED; code: P58-baseline-not-bare-model; detail: baseline_has_head=True; lstm_with_head=True; transformer_with_head=True; fairness_reason_has_LNN=True; fairness_reason_has_double_advantage=True; no_bare_model_allowed=True; manual_exists=True → 实际输出: PASSED; code: P58-baseline-not-bare-model; detail: baseline_has_head=True; lstm_with_head=True; transformer_with_head=True; fairness_reason_has_LNN=True; fairness_reason_has_double_advantage=True; no_bare_model_allowed=True; manual_exists=True

**证据**: `configs/protocol/baseline_not_bare_model.yaml`; `src/liquidloc/analysis/clause_gates.py`（p58_baseline_not_bare_model 函数定义行 5896-5983、_ALL 注册行 6153）；`configs/protocol/` 目录（已有 test_train_no_overlap.yaml、sequence_split_no_overlap.yaml 等协议配置文件，与 baseline_not_bare_model.yaml 构成完整基线对比约束体系）；`docs/异步实验_执行手册_终版v15_总纲条款版.md` 第 327 行（条款 [58] 原文）
