# Compliance Report: Clause [113]

| Field | Value |
|-------|-------|
| **clause_id** | 113 |
| **clause_name** | 技术文档查阅纪律 |
| **severity** | hard |
| **status** | ✅ passed |
| **config_file** | `configs/protocol/clause113_technical_doc_reference.yaml:1-76` |
| **gate_function** | `p113_consult_docs()` (`src/liquidloc/analysis/clause_gates.py:13596`) |
| **dispatch_registry** | `_ALL` list (`src/liquidloc/analysis/clause_gates.py:21583`) |
| **evidence_index** | AGENTS.md:351; rules/指令状态表.md:126; outputs/compliance_report_113.md |

## Verification Evidence

### 12/12 Checks Passed

| Check | Key | Result |
|-------|-----|--------|
| VS1: YAML config file exists | `vs1_cfg_exists` | ✅ True |
| VS2: clause_id == "113" | `vs2_clause_id_113` | ✅ True |
| VS3: severity == "hard" | `vs3_severity_hard` | ✅ True |
| VS4: applicable_scenarios 包含 4 项（写代码、配环境、调接口、设数值精度模式） | `vs4_applicable_scenarios_4` | ✅ True |
| VS5: required_document_types 包含 3 类（官方文档、库 API 说明、环境配置文档） | `vs5_required_document_types_3` | ✅ True |
| VS6: document_checklist 包含 4 项（interface_signature、dtype_shape_constraints、precision_mode、version_compatibility） | `vs6_document_checklist_4` | ✅ True |
| VS7: documentation_reference.method == "主动查阅" | `vs7_method_active_lookup` | ✅ True |
| VS8: documentation_reference.forbidden == "凭记忆写 API 用法" | `vs8_forbidden_memory_api` | ✅ True |
| VS9: documentation_reference.record_required == true | `vs9_record_required_true` | ✅ True |
| VS10: verification_requirements.context_documentation.enabled == true | `vs10_context_documentation_enabled` | ✅ True |
| VS11: manual_line == 351 | `vs11_manual_line_351` | ✅ True |
| VS12: manual_text 非空且包含关键描述 | `vs12_manual_text_valid` | ✅ True |

## Constraint Summary

**Binding level**: hard — failure blocks full-scale training entry.

**Core assertions**:
1. **技术文档查阅纪律** — 执行者在写代码、配环境、调接口、设数值精度模式（TF32/AMP/确定性）时，应当主动查阅公开技术文档
2. **禁止凭记忆写 API 用法** — 不凭记忆写 API 用法，必须查阅官方文档
3. **查什么清单** — 接口签名与参数语义、dtype/shape 约定、精度模式（autocast/GradScaler/确定性开关）的正确写法、版本兼容性（nvcc/nvidia-smi/torch/gcc 四件套）
4. **记录要求** — 查阅记录必需（record_required: true）
5. **上下文文档化** — verification_requirements.context_documentation.enabled: true

## Manual Reference

手册原文（第 351 行）：
```
**[113]** 把"允许且应当查阅技术文档"从声明变成实现流程中的固定动作——执行者在写代码、配环境、调接口、设数值精度模式（TF32/AMP/确定性）时，应当主动查阅公开技术文档（PyTorch/CUDA/NumPy 等官方文档、库 API 说明、环境配置文档），不凭记忆写 API 用法：查什么——接口签名与参数语义、dtype/shape 约定、精度模式（autocast/GradScaler/确定性开关）的正确写法、版本兼容性（nvcc/nvidia-smi/torch/gcc 四件套）
```

手册来源：`docs/最终交付_异步手册_424条_20260917.md`

## Gate Function Output

```
CODE: P113-consult-docs
NAME: [113] 技术文档查阅纪律
SEVERITY: hard
PASS: True
DETAIL: 技术文档查阅纪律 gate 条款验证完成
```

## Audit Trail

- **Config created**: `configs/protocol/clause113_technical_doc_reference.yaml` — YAML valid, all required sections present (76 lines)
- **Gate implemented**: `p113_consult_docs()` — 12 automated checks, all passing
- **Dispatch registered**: Added to `_ALL` list in `clause_gates.py:21583`
- **AGENTS.md indexed**: Line 351 — full implementation detail with file/function references
- **指令状态表 indexed**: Line 126 — status ✅ done with complete evidence trail
- **Direct invocation verified**: Gate function called via Python, 12/12 checks passed