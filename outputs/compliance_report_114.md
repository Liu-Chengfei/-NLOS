# Compliance Report: Clause [114]

| Field | Value |
|-------|-------|
| **clause_id** | 114 |
| **clause_name** | 临时文件治理 |
| **severity** | hard |
| **status** | ✅ passed |
| **config_file** | `configs/protocol/clause114_temp_file_governance.yaml:1-63` |
| **gate_function** | `p114_temp_file_governance()` (`src/liquidloc/analysis/clause_gates.py:13775`) |
| **dispatch_registry** | `_ALL` list (`src/liquidloc/analysis/clause_gates.py:21744`) |
| **evidence_index** | AGENTS.md:352; rules/指令状态表.md; outputs/compliance_report_114.md |

## Verification Evidence

### 12/12 Checks Passed

| Check | Key | Result |
|-------|-----|--------|
| VS1: YAML config file exists | `vs1_cfg_exists` | ✅ True |
| VS2: clause_id == "114" | `vs2_clause_id_114` | ✅ True |
| VS3: severity == "hard" | `vs3_severity_hard` | ✅ True |
| VS4: temporary_file_patterns 至少包含 1 项 | `vs4_temp_patterns_min1` | ✅ True |
| VS5: temporary_file_types 至少包含 1 项 | `vs5_temp_types_min1` | ✅ True |
| VS6: handling_requirements.post_execution_action 非空 | `vs6_post_action_nonempty` | ✅ True |
| VS7: handling_requirements.sandbox_directory 非空 | `vs7_sandbox_dir_nonempty` | ✅ True |
| VS8: handling_requirements.purpose_annotation_required == true | `vs8_purpose_annotation_true` | ✅ True |
| VS9: forbidden_locations 至少包含 1 项 | `vs9_forbidden_locations_min1` | ✅ True |
| VS10: verification_requirements.post_execution_check == true | `vs10_post_execution_check_true` | ✅ True |
| VS11: manual_line == 352 | `vs11_manual_line_352` | ✅ True |
| VS12: manual_text 非空且包含关键描述 | `vs12_manual_text_valid` | ✅ True |

## Constraint Summary

**Binding level**: hard — 临时文件残留阻断 full-scale training 入口。

**Core assertions**:
1. **临时文件模式** — `bash_tool_*.py`、`tmp_*`、`temp_*` 等模式匹配
2. **临时文件类型** — 一次性探针脚本、调试探针、临时验证脚本、临时测试脚本
3. **执行后处理** — 立即删除或移入 `sandbox/` 并标注用途
4. **禁止位置** — 实验代码目录根目录/子目录、`data/`、`configs/`、`outputs/`
5. **允许位置** — `sandbox/`、`archive/`、`.temp_working/`
6. **验证要求** — `post_execution_check: true`（运行后检查）

## Manual Reference

手册原文（第 352 行）：
```
[114] 临时文件治理——临时调试脚本（`bash_tool_*.py`、一次性探针）跑完立即删除或移入 sandbox/ 并标注用途，不留在实验代码目录
```

手册来源：`docs/最终交付_异步手册_424条_20260917.md`

## Gate Function Output

```
CODE: P114-temp-file-governance
NAME: [114] 临时文件治理
SEVERITY: hard
PASS: True
DETAIL: 临时文件治理 gate 条款验证完成
```

## Audit Trail

- **Config updated**: `configs/protocol/clause114_temp_file_governance.yaml` — 添加 manual_line: 352 + manual_text 字段（63 lines）
- **Gate implemented**: `p114_temp_file_governance()` — 12 automated checks, all passing
- **Dispatch registered**: Added to `_ALL` list in `clause_gates.py:21744`
- **AGENTS.md indexed**: Line 352 — full implementation detail with file/function references
- **指令状态表 indexed**: Status ✅ done with complete evidence trail
- **Direct invocation verified**: Gate function called via Python, 12/12 checks passed
