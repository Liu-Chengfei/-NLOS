# [115] checkpoint 落盘格式细则 合规报告

**条款编号**：[115]
**条款标题**：checkpoint 落盘格式细则（硬约束，结果持久化的落盘实现细则）
**严重级别**：hard
**验证函数**：`src/liquidloc/analysis/clause_gates.py::p115_checkpoint_format()`
**配置文件**：`configs/protocol/clause115_checkpoint_format.yaml`
**验证项**：12/12 通过
**验证时间**：2026-07-05T00:00:00Z
**Git commit**：pending
**验证函数行号**：clause_gates.py:13909-14045
**验证结果**：PASS

## 验证项清单

| 编号 | 验证内容 | 状态 | 证据 |
|------|---------|------|------|
| VS-1 | 配置文件存在 | PASS | configs/protocol/clause115_checkpoint_format.yaml |
| VS-2 | clause_id == "115" | PASS | YAML 解析: clause_id='115' |
| VS-3 | severity == "hard" | PASS | YAML 解析: severity='hard' |
| VS-4 | required_payload_fields ≥ 3 项 | PASS | 6 项: model_state, optimizer_state, config_hash, checkpoint_format, best_epoch, best_loss |
| VS-5 | atomic_save.required == true | PASS | YAML: atomic_save.required=true |
| VS-6 | atomic_save.method 匹配 | PASS | method=temp_then_rename |
| VS-7 | config_hash_validation.required == true | PASS | YAML: config_hash_validation.required=true |
| VS-8 | mismatch_action 匹配拒绝 | PASS | mismatch_action=reject |
| VS-9 | forbidden_patterns ≥ 1 项 | PASS | 2 项: write_directly_to_final_path, partial_save_without_cleanup |
| VS-10 | code_audit == true | PASS | YAML: verification_requirements.code_audit=true |
| VS-11 | manual_line == 353 | PASS | 手册第 353 行: **\[115] checkpoint 落盘格式细则 |
| VS-12 | manual_text 非空且含关键描述 | PASS | 含 "115"+"checkpoint"+"落盘" |

## 落盘细则与实现对应

| 配置要求 | 实现位置 | 状态 |
|---------|---------|------|
| required_payload_fields.model_state | _build_checkpoint_payload: model_state key | PASS |
| required_payload_fields.optimizer_state | _build_checkpoint_payload: optimizer_state key | PASS |
| required_payload_fields.config_hash | _add_config_hash_to_payload: payload["config_hash"] | PASS |
| required_payload_fields.checkpoint_format | _build_checkpoint_payload: "checkpoint_format": "liquid_real_v1" | PASS |
| atomic_save.method=temp_then_rename | _atomic_save_checkpoint: tmp_path + os.replace | PASS |
| atomic_save.on_failure_behavior | 崩溃时仅留 tmp，恢复器只认完整文件 | PASS |
| config_hash_validation.mismatch_action=reject | _validate_checkpoint_config_hash: 不匹配→return False | PASS |
