> 本 Stage 从五轴档位协议条款开始:档位参数为本实验场景规格唯一权威定义;随后为场景规格代码条款,逐条落实。
> 
> **五铁律**：
> 1. 报告先于实验；
> 2. 未清场不跑实验；
> 3. 落实说明必须含函数名/改动点/断言，不得空话；
> 4. 证据路径必须真实存在；
> 5. 一旦本实验或另一实验进入论文级区间（重生成至验收），两实验不同时各跑论文级。
> 
> **唯一真源=本手册;状态只认磁盘;原文执行;不做解释性改编;一步一验失败即停、不确定不跑**

## 治理载体索引

|载体|落点|
|---|---|
|`AGENTS.md` 本文件|五铁律 + 治理载体索引|
|`rules/`|条款归属总则.md、判定铁律.md、阶段执行地图.md、手册目标.md、底线清单.md、指令状态表.md、状态只认磁盘.md、治理载体分散布局.md、落实报告格式规范.md、checklist_gate.md|
|`skills/`|逐条落实工序/SKILL.md、落实报告书写/SKILL.md、核验流程/SKILL.md、.agents/skills/execution-report-gate/SKILL.md|
|`.git/hooks/pre-commit`|0 清场、提交前自查、残留审计|
|`outputs/`|落实报告（compliance_report_*.md）|
|`archive/`|清场归档 + 决策日志|
## [5] 依赖声明（治理层·指令模式）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause05_dependency_declaration.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py:15259-15507` — p5_dependency_declaration() 验证函数
- `.git/hooks/pre-commit:111,229` — check_dependencies 前置依赖链硬拦截（定义@111、调用@229）
- `outputs/compliance_report_5.md` — 合规报告

**内容**: 每条指令头部标「前置:[N]」；执行前先查前置是否done，未done → 标blocked停下问，不硬跑、不跳序。pre-commit钩子check_dependencies在[2]-[6]依赖链上实施硬拦截（exit 1阻止提交）。



## [14] 底线：防统计作弊（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录防统计作弊规则文件 — 防统计作弊规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十四防统计作弊配置文件 — 防统计作弊配置（硬约束声明）
- 条款门禁源码 — 验证函数
- 版本控制预提交钩子 — 防统计作弊检查调用点
- 输出目录八字段落实报告 — 八字段落实报告
**判据**: 禁止选择性补跑/改数据到满意/旧数据复用/诊断轨并入验收轨

## [13] 底线：语言铁律（GLM 5.3 专属）
**状态**: ✅ 已落地
**文件**:
-  — 语言铁律规则文件（核心判据+禁止项+允许例外+违规处置）
-  — 语言铁律配置（clause_id=13, constraint_level=硬约束）
-  — 验证函数 
- [pre-commit] 前置检查 [2] 依赖 [1]: OK
[pre-commit] 前置检查 [3] 依赖 [1][2]: OK
[pre-commit] 前置检查 [4] 依赖 [1][2][3]: OK
[pre-commit] 前置检查 [5] 依赖 [1][2][3][4]: OK
[pre-commit] 前置检查 [6] 依赖 [1][2][3][4][5]: OK
[pre-commit] [8] 提交前自查: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | cfg_exists=True; cfg_parse_ok=True; clause_name_contains_8=True; constraint_level_is_hard=True; self_check_before_submit=True; check_all_reports_pass=True; check_status_table_done=True; check_manual_goals_met=True; check_eight_fields_complete=True; manual_exists=True; manual_contains_8=True; manual_contains_self_check=True; manual_contains_core_criteria=True; manual_contains_acceptance_table=True; hook_exists=True; hook_contains_p8_call=True; agents_md_exists=True; agents_md_has_clause_8_section=True; status_table_exists=True; status_table_has_clause_8=True; status_table_clause_8_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [9] 落实报告格式: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_all_eight_fields=True; rules_all_fields_have_three_levels=True; one_sentence_template=True; governance_homology=True; pre_submit_self_check=True; complete_examples_code=True; complete_examples_governance=True; judgment_iron_rule=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_9=True; eight_fields_spec=True; one_sentence_template_cfg=True; governance_homology_cfg=True; pre_submit_self_check_cfg=True; complete_examples_cfg=True; judgment_iron_rule_cfg=True; manual_exists=True; manual_contains_9=True; manual_contains_format=True; manual_contains_eight_fields=True; hook_exists=True; hook_contains_p9_call=True; agents_md_exists=True; agents_md_has_clause_9_section=True; status_table_exists=True; status_table_has_clause_9=True; status_table_clause_9_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [10] 逐条落实工序: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | skill_exists=True; step1_check=True; step2_implementation=True; step3_verification=True; step4_reporting=True; eight_fields_spec=True; self_check_list=True; failure_makes_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_10=True; agents_md_exists=True; agents_md_has_clause_10=True; status_table_exists=True; status_table_has_clause_10=True; status_table_clause_10_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p10_call=True
[pre-commit] [11] 条款归属总则: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_phase0=True; rules_has_phase1_6=True; rules_has_criterion_merged=True; rules_has_phase7_8=True; rules_has_final_check=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_11=True; agents_md_exists=True; agents_md_has_clause_11=True; status_table_exists=True; status_table_has_clause_11=True; status_table_clause_11_field_valid=True; manual_exists=True; manual_contains_11=True; manual_contains_attribution=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p11_call=True
[pre-commit] [12] 手册目标: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | manual_exists=True; manual_has_criterion=True; manual_has_line_a=True; manual_has_line_b=True; manual_has_aux=True; manual_has_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_12=True; agents_md_exists=True; agents_md_has_clause_12=True; status_table_exists=True; status_table_has_clause_12=True; status_table_clause_12_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p12_call=True — 语言铁律检查调用点（仅检查落实报告文件）
-  — 八字段落实报告
**判据**: 全程简体中文；禁止英文回答/禁止中英混杂；3处及以上违规=报告作废重做

## [6] 代码与文件冗余治理（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause06_redundancy_governance.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p6_redundancy_governance() 验证函数
- `outputs/compliance_report_6.md` — 合规报告

**内容**: 禁止为同一逻辑留多份拷贝、禁止复制旧文件再改的备份文化，历史版本走 git，工作目录只保留当前生效版本；发现 _backup/_old/_copy 类冗余文件 → 移入 archive 或删除；禁止"注释掉一大段代码留作备用"；清理动作只作用于本体系目录，禁止跨体系删改；清理任何进程前必须先判定归属。

## [7] 违反即处置（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause07_violation_disposition.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p7_violation_disposition() 验证函数
- `rules/判定铁律.md` — 违反即处置章节（五、违反即处置）
- `outputs/compliance_report_7.md` — 合规报告

**内容**: 执行中任何环节发现违反本文档（未清场/审计失败/口径不符/数据或权重污染）→ 隔离作废相关产物（archive 留痕）→ 修复根因 → 回到对应回归点重跑，处置证据随结果一并提交；任何"未达预期"的第一解释永远是执行偏差，不是方法不行、不是数据集不行；窗口外先修根因再谈结论。

## [8] 提交前自查（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause08_submit_self_check.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p8_submit_self_check() 验证函数
- `.git/hooks/pre-commit` — pre-commit 自查调用点（p8调用在check_dependencies之后）
- `outputs/compliance_report_8.md` — 合规报告

**内容**: 向用户提交论文级结果前先逐项自查是否符合手册目标（核心判据 + 验收表），全部符合才整理提交。

## [10] 逐条落实工序（治理层·四步闭环）
**状态**: ✅ 已落地
**文件**:
- `skills/逐条落实工序/SKILL.md` — 四步闭环工序唯一权威载体（查→落→验→报）
- `configs/protocol/clause10_four_step_closure.yaml` — 约束配置（硬约束声明、四步标志）
- `src/liquidloc/analysis/clause_gates.py` — p10_four_step_closure() 验证函数
- `.git/hooks/pre-commit` — pre-commit 四步工序验证调用点（p10调用在p9之后）
- `outputs/compliance_report_10.md` — 合规报告

**内容**: 每条指令按「查 → 落 → 验 → 报」四步闭环：查（读条款原文+查磁盘现状，禁止凭记忆）→ 落（翻译为文件路径+函数/类+行号落点）→ 验（逐行核对条款逐句对应，不落实当场改）→ 报（按八字段写落实记录，未写记录视为未落实）。

## [11] 条款归属总则（治理层）
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 三阶段归属总则（治理层/Stage1-6落实类/Stage7-8执行决策类）
- `configs/protocol/clause11_clause_attribution.yaml` — 约束配置（硬约束声明、归属标志）
- `src/liquidloc/analysis/clause_gates.py` — p11_clause_attribution() 验证函数
- `.git/hooks/pre-commit` — pre-commit 归属总则验证调用点（p11调用在p10之后）
- `outputs/compliance_report_11.md` — 合规报告

**内容**: 治理层条款→第0阶段逐条落实并写报告，证据写对应治理载体（按『治理载体分散布局』分散落点：AGENTS/rules/skill/hook/plugin/MCP）；Stage 1-6落实类→第1阶段代码落实，判据条款的判据已并入对应代码条款；Stage 7-8执行决策类→第2阶段执行，每条同样单独写执行落实报告。

## [12] 手册目标（治理层·最终检查只看这个）
**状态**: ✅ 已落地
**文件**:
- `rules/手册目标.md` — 核心判据唯一判据（主线A/B必须同时达成）
- `configs/protocol/clause12_manual_objectives.yaml` — 约束配置（硬约束声明、判据标志）
- `src/liquidloc/analysis/clause_gates.py` — p12_manual_objectives() 验证函数
- `.git/hooks/pre-commit` — pre-commit 手册目标验证调用点（p12调用在p11之后）
- `outputs/compliance_report_12.md` — 合规报告

**内容**: 防异步高NLOS实验RMSE整体过大（异步+NLOS下误差过大时所有方法一起高）；防LNN（Liquid液态骨干网络，proposed）的RMSE不是最低；两项判据统一为唯一判据，跑前冒烟/预达标与跑后判定共用；验收分级与降级出口：验收表每一项必须在完整验收路径下逐项回查修通，诊断轨/冒烟数据不得以任何形式并入、替代或补充验收轨数据；数据/代码/结果可追溯（manifest + git commit + 验收留痕）；核验通过后最终只看目标是否达成，不看流程勾选。

## [9] 落实报告格式（治理层·八字段）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause09_compliance_report_format.yaml` — 约束配置
- `rules/落实报告格式规范.md` — 八字段格式规范（唯一权威副本，含八字段逐字段说明/一句话模板/治理类同构/提交前自检/完整示例/判定铁律）
- `.agents/skills/execution-report-gate/SKILL.md` — §1 为八字段格式唯一逐字权威副本（入口）
- `src/liquidloc/analysis/clause_gates.py` — p9_compliance_report_format() 验证函数
- `.git/hooks/pre-commit` — pre-commit 八字段验证调用点（p9调用在p8之后）
- `outputs/compliance_report_9.md` — 合规报告

**内容**: 每条指令输出 = 落实报告格式一行，八字段逐字段规范（条款号/状态/条款逐字抄录/代码绝对定位/落实极尽具体+全量映射/审查问题→修复+循环轨迹/验证命令→预期/证据真实路径+stdout落盘），含一句话模板、治理类同构、提交前自检、完整示例、判定铁律；缺任一字段或字段不合格 = 该条未落实。细则进 `rules/落实报告格式规范.md`，不堆进 AGENTS.md。

## [36] git 是最后防线
**状态**: ✅ 已落地
**文件**: 
- `configs/protocol/git_is_last_resort.yaml` — 约束配置
- `.git/hooks/pre-commit` — pre-commit 防线（第 1-161 行）
- `.gitignore:30-31` — *.log 和 *.bak 防护规则
- `outputs/compliance_report_36.md` — 合规报告

**内容**: git 仓库与工作区必须一致，任何版本回退/文件损坏/误改恢复一律走 git（checkout/revert），禁止复制 _backup/_old 副本。

## [37] 档位定义声明
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/gear_definition_declaration.yaml` — 约束配置
- `configs/protocol/axis_a_sync.yaml` — A 轴档位协议
- `configs/protocol/axis_n_nlos.yaml` — N 轴档位协议
- `configs/protocol/axis_v_visual.yaml` — V 轴档位协议
- `configs/protocol/axis_k_anchor.yaml` — K 轴档位协议
- `configs/protocol/axis_m_missing.yaml` — M 轴档位协议
- `src/liquidloc/analysis/clause_gates.py:3242-3395` — 验证函数 `p37_gear_definition_declaration()`
- `outputs/compliance_report_37.md` — 合规报告

**内容**: A/N/V/K/M 各轴档位划分、参数与具体数值以 Stage 1 档位协议条款为唯一权威副本。

## [38] LOS 段测距噪声 σ≈0.6m
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/los_noise_baseline.yaml` — LOS 段测距噪声基准配置（noise_value: 0.6, clause_id: 38）
- `configs/models/ekf.yaml:30-31` — EKF UWB 量测噪声 σ=0.6m（S4 LOS 基准 σ0.6m）
- `configs/models/robust_ekf.yaml:34-35` — Robust-EKF UWB 量测噪声 σ=0.6m（S4 LOS σ0.6m）
- `outputs/compliance_report_38.md` — 合规报告

**内容**: LOS 段测距噪声 σ≈0.6m（器件事例级均值误差，作为 LOS 基准，非协议档位参数）。LOS 段高斯噪声（商用 UWB 模块室内实测测距误差标准差在亚米级、LOS 与 NLOS 统计分布差异显著）。
## [39] UWB(DW1000级, 10Hz) 输出数据规范
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/uwb_output_spec.yaml` — UWB 输出数据规范配置（chip_model: DW1000级, frequency_hz: 10Hz）
- `src/liquidloc/analysis/clause_gates.py:3761-3927` — 验证函数 `p39_uwb_output_spec()`
- `outputs/compliance_report_39.md` — 合规报告

**内容**: UWB(DW1000级, 10Hz)；输出数据：`uwb_ranges`(N×4 测距, m) + `uwb_valid`(N×4 有效掩码)。UWB 传感器输出数据规范，包含测距矩阵与有效掩码矩阵。

## [40] IMU（消费级，150Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/imu_output_spec.yaml` — IMU 输出数据规范配置（chip_type: 消费级，frequency_hz: 150Hz）
- `src/liquidloc/analysis/clause_gates.py:3930-4110` — 验证函数 `p40_imu_output_spec()`
- `outputs/compliance_report_40.md` — 合规报告

**内容**: IMU（消费级，150Hz）；输出数据：`imu`（N_imu×6：ax/ay/az/gx/gy/gz，150Hz 原生，不降采样）+ `imu_t`（N_imu×1 自身时间戳）。IMU 传感器输出数据规范。

## [41] VIO（单目+IMU 前端，20 Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/vio_output_spec.yaml` — VIO 输出数据规范配置（frequency_hz: 20Hz, sensor_type: 单目+IMU前端）
- `src/liquidloc/analysis/clause_gates.py:4111-4324` — 验证函数 `p41_vio_output_spec()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_41.md` — 合规报告

**内容**: VIO（单目+IMU 前端，20 Hz）；输出数据：`vio_pos`（N×2/3）+ `vio_yaw`（N×1）+ `vio_valid`（N×1）+ 协方差（V0 健康档有效帧率 100%）。VIO 传感器输出数据规范。

## [42] 四个输出头
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/four_output_heads.yaml` — 四个输出头机制链配置（uwb_scaling/bias/vio_scaling/risk）
- `configs/protocol/imu_no_head_reasoning.yaml` — IMU 不设头推理说明配置
- `src/liquidloc/analysis/clause_gates.py:4326-4518` — 验证函数 `p42_imu_no_head_reasoning()`
- `src/liquidloc/analysis/clause_gates.py:4521-4665` — 验证函数 `p42b_four_output_heads()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_42.md` — 合规报告

**内容**: 四个输出头（uwb_scaling / bias / vio_scaling / risk）对三源量测做自适应加权、偏差补偿与门控，是"NLOS 识别 → 协方差调制/降权 → 鲁棒门控"机制链的落地（与 S4.2 鲁棒融合组合拳一致）。IMU 作为高频运动先验不参与头调制（惯性量测不经数据驱动加权，走 EKF 运动模型传播），故不设 IMU 头；risk 头为通道级联合风险。

## [43] 禁止使用档位空间外的参数、禁止自行定义插值档/新档
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_extra_levels.yaml` — 禁止额外档位配置（forbidden_interpolation: true, forbidden_new_levels: true）
- `src/liquidloc/analysis/clause_gates.py:4859-4960` — 验证函数 `p43_no_extra_levels()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_43.md` — 合规报告

**内容**: 档位参数严格取自档位协议条款档位表（A/N/V/K/M 五轴），禁止使用档位空间外的参数值，禁止自行定义插值档（如 N1.5 类）或任何新档位。

## [44] 档位注入纪律
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause44_injection_discipline.yaml` — 档位注入纪律配置（静态口径/禁止漂移档位）
- `src/liquidloc/analysis/clause_gates.py:4668-4770` — 验证函数 `p44_gear_injection_discipline()`
- `outputs/compliance_report_44.md` — 合规报告

**内容**: A轴为静态时钟偏移/延迟统计口径（数值见 Stage 1 档位协议条款 A 轴表），严格按静态口径注入，不引入随时间线性增长的时钟漂移作为额外档位（如需评估 → 协议裁决项 + 敏感性分析）。

## [45] N1.5 边界协议
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/n15_boundary.yaml` — N1.5 边界协议配置
- `src/liquidloc/analysis/clause_gates.py:4759-4855` — 验证函数 `p45_n15_boundary()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_45.md` — 合规报告

**内容**: N1.5（ρ=20–25%/μ=1.5–2.5m/σ=0.5m）不在本手册档位协议条款档位表内；本手册引用协议 N 轴为 N0–N3；N1.5 属于同机另一实验体系，按其各自对应协议文件执行。

## [46] 协议引用边界
**状态**: ✅ 已落地（由 p49_protocol_reference_boundary() 验证）
**文件**:
- `configs/protocol/protocol_reference_boundary.yaml` — 协议引用边界配置
- `src/liquidloc/analysis/clause_gates.py:5098` 起 — 验证函数 `p49_protocol_reference_boundary()`
- `outputs/compliance_report_46.md` — 合规报告

**内容**: 档位以 Stage 1 档位协议条款为唯一权威副本；其他文件引用档位时必须标明出自该条款，不得复制参数数值形成第二权威副本。注意：验证函数沿用 p49 命名（历史实现），实际验证手册 [46] 协议引用边界条款。

## [47] 三层证据结构总表
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/three_layer_evidence_structure.yaml` — 三层证据结构总表配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p47_three_layer_evidence()`
- `outputs/compliance_report_47.md` — 合规报告

**内容**: 三层证据结构总表（论文级数据架构，与 S3 轨迹规格衔接）：主场景（受控）S1–S3固定布局（20m×20m、4锚、4遮挡区）10seed×5方法=50单元；布局随机化泛化层；统计推断与主场景共享层。

## [48] 规模档位
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/scale_gear_spec.yaml` — 规模档位配置（入门 N_layout=6、推荐 N_layout=10）
- `src/liquidloc/analysis/clause_gates.py:5189-5290` — 验证函数 `p48_two_sigma_separation()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_48.md` — 合规报告

**内容**: 规模档位：入门 N_layout=6、推荐 N_layout=10（full + 2 个最强基线 LSTM/EKF × 2–3 seed 即可，泛化层只判排序不做因子推断，不需要主场景同等功效）。

## [49] 布局参数与方法排序的散点/箱线
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_scatter_plot_spec.yaml` — 布局参数散点配置
- `src/liquidloc/analysis/clause_gates.py`（p54_layout_parameter_scatter 验证函数）
- `outputs/compliance_report_49.md` — 合规报告

**内容**: 布局参数变化与方法排序关系以散点/箱线图呈现，禁止只用汇总统计量。

## [50] 测试轨迹与训练无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause50_test_train_no_overlap.yaml` — 测试轨迹与训练无重叠配置
- `src/liquidloc/analysis/clause_gates.py`（p50_test_train_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_50.md` — 合规报告

**内容**: 测试轨迹与训练数据在序列级别严格分离，不存在任何重叠。所有评估指标均在独立于训练集的数据上计算。


## [51] 划分按整条序列、训练与测试无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause51_sequence_split_no_overlap.yaml` — 序列级划分配置
- `src/liquidloc/analysis/clause_gates.py`（p51_sequence_split_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_51.md` — 合规报告

**内容**: 划分按整条序列进行，训练集与测试集之间不存在任何序列级别的重叠。禁止在同一序列内随机采样部分帧/窗口作为训练、部分帧/窗口作为测试，防止数据泄漏。

## [52] 基线不是裸模型
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/baseline_not_bare_model.yaml` — 基线不是裸模型配置
- `src/liquidloc/analysis/clause_gates.py`（p52_baseline_not_bare_model 验证函数）
- `outputs/compliance_report_52.md` — 合规报告

**内容**: 确认 LSTM/Transformer 基线确实带头运行——否则 LNN 赢它们="4 个头 + 液态骨干"双重优势，对比不公平。

## [53] 配对样本对齐
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/paired_sample_alignment.yaml` — 配对样本对齐配置
- `src/liquidloc/analysis/clause_gates.py:5658-5729` — 验证函数 `p53_paired_sample_alignment()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_53.md` — 合规报告

**内容**: 各方法评估按"同一测试轨迹 × 同 seed"配对。

## [54] 两两比较附Holm-Bonferroni多重比较校正
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/holm_bonferroni_correction.yaml` — Holm-Bonferroni校正配置
- `src/liquidloc/analysis/clause_gates.py`（p60_holm_bonferroni_correction 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_54.md` — 合规报告

**内容**: 五方法两两比较共10对，使用Holm-Bonferroni校正控制族错误率。多方法对比不做校正会放大假阳性；Holm-Bonferroni为稳健默认；ML比较应配套统计检验+CI+校正。

## [55] 扩样路径
**状态**: 落实中
**文件**:
- `configs/protocol/scale_expansion_path.yaml` — 扩样路径配置
- `src/liquidloc/analysis/clause_gates.py`（p55_scale_expansion_path 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_55.md` — 合规报告

**内容**: 10 seed 不足支撑功效时如实声明并给出扩样路径（只增不减：加 seed/轨迹至功效达标，规模恒 ≥ 论文级 10 seed × ≥100/seed，论文级硬约束——不存在"收窄/降档"方向的出口）。

## [56] 分析单位与样本独立性声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/analysis_unit_sample_independence.yaml` — 分析单位配置
- `src/liquidloc/analysis/clause_gates.py`（p64_analysis_unit_sample_independence 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_56.md` — 合规报告

**内容**: 滑窗W=128步（步长默认1帧）使相邻窗口重叠127/128，帧级误差样本高度自相关、不满足i.i.d.，所有统计检验必须以轨迹级为分析单位。

## [57] 先导实验（pilot）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/pilot_experiment.yaml` — 先导实验配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p57_sequence_split_no_overlap()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_57.md` — 合规报告

**内容**: 先导实验单独用于方案冻结前估计效应量/方差/可行性，规模<=1 seed×四组合各1-2条测试轨迹。

## [58] A2/A3 的偏移与抖动数值按协议生效
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/a2a3_shift_jitter.yaml` — A2/A3 偏移与抖动协议配置（clause_id="58"）
- `src/liquidloc/analysis/clause_gates.py::p68_a2a3_shift_jitter()` — 验证函数
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_58.md` — 合规报告
- `checkpoints/compliance_report_clause58.md` — 验证快照

**内容**: A2/A3的偏移与抖动数值按协议生效，已按A2/A3全参数逐项核对。A2：UWB时钟偏移20-50ms、UWB往返延迟40-80ms、IMU时钟偏移5-15ms、IMU采样抖动±3ms、VIO时钟偏移20-50ms、VIO处理延迟30-80ms。

## [59] 重采样方法声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/resampling_method_declaration.yaml` — 重采样方法声明配置
- `src/liquidloc/analysis/clause_gates.py`（p69_resampling_method 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_59.md` — 合规报告

**内容**: UWB/VIO重采样采用最近邻（nearest-neighbor）对齐到UWB帧时刻，不使用线性/样条插值生成测距或VIO值；声明写入论文方法节。

## [60] 边界/极端情况
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/edge_case_extreme.yaml` — 边界/极端情况配置
- `src/liquidloc/analysis/clause_gates.py`（p60_edge_case_extreme 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_60.md` — 合规报告

**内容**: 边界极端情况：空轨迹、极短序列、M1长间隙落在首尾，不产生人工尖峰/除零。

## [61] 同 seed 初始化
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/same_seed_init.yaml` — 同seed初始化配置
- `src/liquidloc/analysis/clause_gates.py`（p71_same_seed_init 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_61.md` — 合规报告

**内容**: 同seed初始化：LNN/LSTM/Transformer用同一seed的权重初始化。

## [62] 论文级规模参数
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/power_sample_size.yaml` — 规模参数配置
- `src/liquidloc/analysis/clause_gates.py`（p62_power_sample_size 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_62.md` — 合规报告

**内容**: 论文级规模参数：训练12-15条/seed，测试>=100条/seed，四组合各>=25，120s基准，10 seed（论文级基线）。

## [63] 布局随机化泛化层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_randomization.yaml` — 布局随机化配置
- `src/liquidloc/analysis/clause_gates.py`（p72_layout_randomization 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_63.md` — 合规报告
- `checkpoints/compliance_report_clause63.md` — checkpoint 验证报告

**内容**: 布局随机化泛化层：N_layout>=6(入门)/10(推荐)，每布局5方法x2-3 seed。

## [64] 重新生成全部 seed 数据
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/regenerate_all_seeds.yaml` — 重新生成全部 seed 配置
- `src/liquidloc/analysis/clause_gates.py`（p73_regenerate_all_seeds 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_64.md` — 合规报告
- `checkpoints/compliance_report_clause64.md` — checkpoint 验证报告

**内容**: 主场景10 seed全部重新生成，禁止新旧数据混用。

## [65] 新旧数据混用禁令（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_old_new_mixing.yaml` — 新旧数据混用禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p74_no_old_new_mixing 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_65.md` — 合规报告
- `checkpoints/compliance_report_clause65.md` — checkpoint 验证报告

**内容**: 禁止部分seed新+部分seed旧混用，硬约束。

## [66] 禁止把冒烟数据并入全量测试集
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_smoke_merge.yaml` — 冒烟数据禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p75_no_smoke_merge 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_66.md` — 合规报告
- `checkpoints/compliance_report_clause66.md` — checkpoint 验证报告

**内容**: 禁止冒烟>=12条测试数据并入全量测试集，整批化执行，硬约束。

## [67] 全量单元
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/full_scale_unit.yaml` — 全量单元配置
- `src/liquidloc/analysis/clause_gates.py`（p76_full_scale_unit 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_67.md` — 合规报告
- `checkpoints/compliance_report_clause67.md` — checkpoint 验证报告

**内容**: 全量单元：主场景10 seed×5方法=50单元全部用新数据训练。

## [68] 执行偏差（deviation）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/execution_deviation.yaml` — 执行偏差配置（含 deviation 表、E 系列审计、目录命名规范）
- `src/liquidloc/analysis/clause_gates.py:7103-7203` — 验证函数 `p68_execution_deviation()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_68.md` — 合规报告
- `checkpoints/compliance_report_clause68.md` — 验证快照

**内容**: 执行层面与方案的偏差（漏跑/换机/中断恢复）→ 记 execution deviation 表（E 系列审计覆盖），不影响协议与分析定义。实验目录结构 run 目录命名规范：`run-<日期>-<seed>-<method>-<config_hash>`。

## [69] CPU 角色分层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/cpu_role_description.yaml` — CPU 角色分层配置（cpu_non_training_only/cpu_idle_training/no_conflict/priority_rule）
- `src/liquidloc/analysis/clause_gates.py:7418-7519` — 验证函数 `p69_cpu_role_hierarchy()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_69.md` — 合规报告
- `checkpoints/compliance_report_clause69.md` — 验证快照

**内容**: "CPU 仅限非训练负载"与"CPU 空闲协助训练"的关系：前者是 GPU 空闲时的默认分工（CPU 只做数据生成/评估/预处理），后者是 GPU 排满后的追加自由度（CPU 可承接剩余训练单元）——两者不冲突，GPU 有空的按前者，GPU 排满的按后者。

## [70] 评估并行
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause70_evaluation_parallel.yaml` — 评估并行配置（gpu_execution=true, trajectory_batch_size=16, no_batch_mixing=true, evaluation_protocol_unchanged=true）
- `src/liquidloc/analysis/clause_gates.py`（p70_evaluation_parallel 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_70.md` — 合规报告
- `checkpoints/compliance_report_clause70.md` — checkpoint 验证报告

**内容**: 评估并行：测试轨迹推理在 GPU 上按轨迹独立并行执行（轨迹内窗口批大小 16，轨迹间不混批，评估协议不变）。

## [71] 父进程链可回溯
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/parent_process_traceable.yaml` — 父进程链可回溯配置（traceable=true, command="ps -o ppid= -p <pid>"）
- `src/liquidloc/analysis/clause_gates.py:8337-8417` — 验证函数 `p77_parent_process_traceable()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_71.md` — 合规报告
- `checkpoints/compliance_report_clause71.md` — checkpoint 验证报告

**内容**: 父进程链可回溯到本体系的启动器（`ps -o ppid= -p <pid>` 逐级上溯，最终父进程是本体系的调度/启动脚本）。

## [72] Transformer 滑窗同轨迹内合批
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/transformer_sliding_window.yaml` — Transformer 滑窗合批配置（enabled=true, method="同轨迹内窗口合批", log_batch_composition=true）
- `src/liquidloc/analysis/clause_gates.py:8420-8505` — 验证函数 `p78_transformer_sliding_window()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_72.md` — 合规报告
- `checkpoints/compliance_report_clause72.md` — checkpoint 验证报告

**内容**: Transformer 滑窗同轨迹内合批 评估日志打印批构成（同轨迹内窗口）。

## [73] GPU 报错停
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/gpu_error_handling.yaml` — GPU 报错处理配置（verified=true, cpu_idle_assistance=true, cpu_only_for_full_gpu=true）
- `src/liquidloc/analysis/clause_gates.py:8508-8592` — 验证函数 `p79_no_gpu_error_stop()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_73.md` — 合规报告
- `checkpoints/compliance_report_clause73.md` — checkpoint 验证报告

**内容**: 在无 GPU 环境实测 → 确认报错停而非静默 CPU 训练。CPU 空闲协助（执行模式/执行模式·CPU 空闲协助）CPU 承接仅限 GPU 排满后启用的剩余训练单元。

## [74] RAM 上限 12G
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/ram_limit_hard_cap.yaml` — RAM 上限配置（max_gb=12, verification_method="冒烟实测单单元峰值内存（psutil 探针）"）
- `src/liquidloc/analysis/clause_gates.py:8595-8676` — 验证函数 `p80_ram_limit_12g()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_74.md` — 合规报告
- `checkpoints/compliance_report_clause74.md` — checkpoint 验证报告

**内容**: RAM ≤12G 硬上限（执行模式/执行模式·RAM ≤12G 硬上限）；验证方法（细化，自测留痕）：冒烟实测单单元峰值内存（psutil 探针）。

## [75] 禁止借大修变口径
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_refactor_changes.yaml` — 禁止变口径配置（no_change_paper_level_values=true, no_change_data_gen=true, no_change_eval=true, no_change_protocol=true）
- `src/liquidloc/analysis/clause_gates.py:8679-8768` — 验证函数 `p81_no_refactor_changes()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_75.md` — 合规报告
- `checkpoints/compliance_report_clause75.md` — checkpoint 验证报告

**内容**: 禁止借大修变口径——大修只改代码组织与可维护性（模块拆分/常量收敛/断言补齐/命名统一），不得改变任何论文级数值、不得改数据生成/评估/统计口径、不得为了好写而改协议。

## [76]

## [15]


## [16] 底线：第1阶段附加义务（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录第1阶段附加义务规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十六第1阶段附加义务配置文件（硬约束声明，含条款名加约束级别加治理阶段零等标志）
- 源码目录条款门禁验证模块（p一六资源冲突错峰排队调度函数）
- 版本控制预提交钩子（第1阶段附加义务检查调用点）
- 输出目录八字段落实报告（纯中文表述）
**判据**: 落实中发现缺陷→顺手修根因（禁止留待办事项/绕开/注释旧实现）；缺陷检查范围=阶段一至六条款映射代码；每条条款落实完成时做一次缺陷检查（静态审查/运行断言/单测至少一种）；发现即修根因；修复检查结果写入八字段报告"审查:"字段并留痕
## [5] 依赖声明（治理层·指令模式）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause05_dependency_declaration.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py:15259-15507` — p5_dependency_declaration() 验证函数
- `.git/hooks/pre-commit:111,229` — check_dependencies 前置依赖链硬拦截（定义@111、调用@229）
- `outputs/compliance_report_5.md` — 合规报告

**内容**: 每条指令头部标「前置:[N]」；执行前先查前置是否done，未done → 标blocked停下问，不硬跑、不跳序。pre-commit钩子check_dependencies在[2]-[6]依赖链上实施硬拦截（exit 1阻止提交）。



## [14] 底线：防统计作弊（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录防统计作弊规则文件 — 防统计作弊规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十四防统计作弊配置文件 — 防统计作弊配置（硬约束声明）
- 条款门禁源码 — 验证函数
- 版本控制预提交钩子 — 防统计作弊检查调用点
- 输出目录八字段落实报告 — 八字段落实报告
**判据**: 禁止选择性补跑/改数据到满意/旧数据复用/诊断轨并入验收轨

## [13] 底线：语言铁律（GLM 5.3 专属）
**状态**: ✅ 已落地
**文件**:
-  — 语言铁律规则文件（核心判据+禁止项+允许例外+违规处置）
-  — 语言铁律配置（clause_id=13, constraint_level=硬约束）
-  — 验证函数 
- [pre-commit] 前置检查 [2] 依赖 [1]: OK
[pre-commit] 前置检查 [3] 依赖 [1][2]: OK
[pre-commit] 前置检查 [4] 依赖 [1][2][3]: OK
[pre-commit] 前置检查 [5] 依赖 [1][2][3][4]: OK
[pre-commit] 前置检查 [6] 依赖 [1][2][3][4][5]: OK
[pre-commit] [8] 提交前自查: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | cfg_exists=True; cfg_parse_ok=True; clause_name_contains_8=True; constraint_level_is_hard=True; self_check_before_submit=True; check_all_reports_pass=True; check_status_table_done=True; check_manual_goals_met=True; check_eight_fields_complete=True; manual_exists=True; manual_contains_8=True; manual_contains_self_check=True; manual_contains_core_criteria=True; manual_contains_acceptance_table=True; hook_exists=True; hook_contains_p8_call=True; agents_md_exists=True; agents_md_has_clause_8_section=True; status_table_exists=True; status_table_has_clause_8=True; status_table_clause_8_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [9] 落实报告格式: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_all_eight_fields=True; rules_all_fields_have_three_levels=True; one_sentence_template=True; governance_homology=True; pre_submit_self_check=True; complete_examples_code=True; complete_examples_governance=True; judgment_iron_rule=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_9=True; eight_fields_spec=True; one_sentence_template_cfg=True; governance_homology_cfg=True; pre_submit_self_check_cfg=True; complete_examples_cfg=True; judgment_iron_rule_cfg=True; manual_exists=True; manual_contains_9=True; manual_contains_format=True; manual_contains_eight_fields=True; hook_exists=True; hook_contains_p9_call=True; agents_md_exists=True; agents_md_has_clause_9_section=True; status_table_exists=True; status_table_has_clause_9=True; status_table_clause_9_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [10] 逐条落实工序: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | skill_exists=True; step1_check=True; step2_implementation=True; step3_verification=True; step4_reporting=True; eight_fields_spec=True; self_check_list=True; failure_makes_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_10=True; agents_md_exists=True; agents_md_has_clause_10=True; status_table_exists=True; status_table_has_clause_10=True; status_table_clause_10_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p10_call=True
[pre-commit] [11] 条款归属总则: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_phase0=True; rules_has_phase1_6=True; rules_has_criterion_merged=True; rules_has_phase7_8=True; rules_has_final_check=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_11=True; agents_md_exists=True; agents_md_has_clause_11=True; status_table_exists=True; status_table_has_clause_11=True; status_table_clause_11_field_valid=True; manual_exists=True; manual_contains_11=True; manual_contains_attribution=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p11_call=True
[pre-commit] [12] 手册目标: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | manual_exists=True; manual_has_criterion=True; manual_has_line_a=True; manual_has_line_b=True; manual_has_aux=True; manual_has_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_12=True; agents_md_exists=True; agents_md_has_clause_12=True; status_table_exists=True; status_table_has_clause_12=True; status_table_clause_12_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p12_call=True — 语言铁律检查调用点（仅检查落实报告文件）
-  — 八字段落实报告
**判据**: 全程简体中文；禁止英文回答/禁止中英混杂；3处及以上违规=报告作废重做

## [6] 代码与文件冗余治理（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause06_redundancy_governance.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p6_redundancy_governance() 验证函数
- `outputs/compliance_report_6.md` — 合规报告

**内容**: 禁止为同一逻辑留多份拷贝、禁止复制旧文件再改的备份文化，历史版本走 git，工作目录只保留当前生效版本；发现 _backup/_old/_copy 类冗余文件 → 移入 archive 或删除；禁止"注释掉一大段代码留作备用"；清理动作只作用于本体系目录，禁止跨体系删改；清理任何进程前必须先判定归属。

## [7] 违反即处置（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause07_violation_disposition.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p7_violation_disposition() 验证函数
- `rules/判定铁律.md` — 违反即处置章节（五、违反即处置）
- `outputs/compliance_report_7.md` — 合规报告

**内容**: 执行中任何环节发现违反本文档（未清场/审计失败/口径不符/数据或权重污染）→ 隔离作废相关产物（archive 留痕）→ 修复根因 → 回到对应回归点重跑，处置证据随结果一并提交；任何"未达预期"的第一解释永远是执行偏差，不是方法不行、不是数据集不行；窗口外先修根因再谈结论。

## [8] 提交前自查（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause08_submit_self_check.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p8_submit_self_check() 验证函数
- `.git/hooks/pre-commit` — pre-commit 自查调用点（p8调用在check_dependencies之后）
- `outputs/compliance_report_8.md` — 合规报告

**内容**: 向用户提交论文级结果前先逐项自查是否符合手册目标（核心判据 + 验收表），全部符合才整理提交。

## [10] 逐条落实工序（治理层·四步闭环）
**状态**: ✅ 已落地
**文件**:
- `skills/逐条落实工序/SKILL.md` — 四步闭环工序唯一权威载体（查→落→验→报）
- `configs/protocol/clause10_four_step_closure.yaml` — 约束配置（硬约束声明、四步标志）
- `src/liquidloc/analysis/clause_gates.py` — p10_four_step_closure() 验证函数
- `.git/hooks/pre-commit` — pre-commit 四步工序验证调用点（p10调用在p9之后）
- `outputs/compliance_report_10.md` — 合规报告

**内容**: 每条指令按「查 → 落 → 验 → 报」四步闭环：查（读条款原文+查磁盘现状，禁止凭记忆）→ 落（翻译为文件路径+函数/类+行号落点）→ 验（逐行核对条款逐句对应，不落实当场改）→ 报（按八字段写落实记录，未写记录视为未落实）。

## [11] 条款归属总则（治理层）
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 三阶段归属总则（治理层/Stage1-6落实类/Stage7-8执行决策类）
- `configs/protocol/clause11_clause_attribution.yaml` — 约束配置（硬约束声明、归属标志）
- `src/liquidloc/analysis/clause_gates.py` — p11_clause_attribution() 验证函数
- `.git/hooks/pre-commit` — pre-commit 归属总则验证调用点（p11调用在p10之后）
- `outputs/compliance_report_11.md` — 合规报告

**内容**: 治理层条款→第0阶段逐条落实并写报告，证据写对应治理载体（按『治理载体分散布局』分散落点：AGENTS/rules/skill/hook/plugin/MCP）；Stage 1-6落实类→第1阶段代码落实，判据条款的判据已并入对应代码条款；Stage 7-8执行决策类→第2阶段执行，每条同样单独写执行落实报告。

## [12] 手册目标（治理层·最终检查只看这个）
**状态**: ✅ 已落地
**文件**:
- `rules/手册目标.md` — 核心判据唯一判据（主线A/B必须同时达成）
- `configs/protocol/clause12_manual_objectives.yaml` — 约束配置（硬约束声明、判据标志）
- `src/liquidloc/analysis/clause_gates.py` — p12_manual_objectives() 验证函数
- `.git/hooks/pre-commit` — pre-commit 手册目标验证调用点（p12调用在p11之后）
- `outputs/compliance_report_12.md` — 合规报告

**内容**: 防异步高NLOS实验RMSE整体过大（异步+NLOS下误差过大时所有方法一起高）；防LNN（Liquid液态骨干网络，proposed）的RMSE不是最低；两项判据统一为唯一判据，跑前冒烟/预达标与跑后判定共用；验收分级与降级出口：验收表每一项必须在完整验收路径下逐项回查修通，诊断轨/冒烟数据不得以任何形式并入、替代或补充验收轨数据；数据/代码/结果可追溯（manifest + git commit + 验收留痕）；核验通过后最终只看目标是否达成，不看流程勾选。

## [9] 落实报告格式（治理层·八字段）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause09_compliance_report_format.yaml` — 约束配置
- `rules/落实报告格式规范.md` — 八字段格式规范（唯一权威副本，含八字段逐字段说明/一句话模板/治理类同构/提交前自检/完整示例/判定铁律）
- `.agents/skills/execution-report-gate/SKILL.md` — §1 为八字段格式唯一逐字权威副本（入口）
- `src/liquidloc/analysis/clause_gates.py` — p9_compliance_report_format() 验证函数
- `.git/hooks/pre-commit` — pre-commit 八字段验证调用点（p9调用在p8之后）
- `outputs/compliance_report_9.md` — 合规报告

**内容**: 每条指令输出 = 落实报告格式一行，八字段逐字段规范（条款号/状态/条款逐字抄录/代码绝对定位/落实极尽具体+全量映射/审查问题→修复+循环轨迹/验证命令→预期/证据真实路径+stdout落盘），含一句话模板、治理类同构、提交前自检、完整示例、判定铁律；缺任一字段或字段不合格 = 该条未落实。细则进 `rules/落实报告格式规范.md`，不堆进 AGENTS.md。

## [36] git 是最后防线
**状态**: ✅ 已落地
**文件**: 
- `configs/protocol/git_is_last_resort.yaml` — 约束配置
- `.git/hooks/pre-commit` — pre-commit 防线（第 1-161 行）
- `.gitignore:30-31` — *.log 和 *.bak 防护规则
- `outputs/compliance_report_36.md` — 合规报告

**内容**: git 仓库与工作区必须一致，任何版本回退/文件损坏/误改恢复一律走 git（checkout/revert），禁止复制 _backup/_old 副本。

## [37] 档位定义声明
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/gear_definition_declaration.yaml` — 约束配置
- `configs/protocol/axis_a_sync.yaml` — A 轴档位协议
- `configs/protocol/axis_n_nlos.yaml` — N 轴档位协议
- `configs/protocol/axis_v_visual.yaml` — V 轴档位协议
- `configs/protocol/axis_k_anchor.yaml` — K 轴档位协议
- `configs/protocol/axis_m_missing.yaml` — M 轴档位协议
- `src/liquidloc/analysis/clause_gates.py:3242-3395` — 验证函数 `p37_gear_definition_declaration()`
- `outputs/compliance_report_37.md` — 合规报告

**内容**: A/N/V/K/M 各轴档位划分、参数与具体数值以 Stage 1 档位协议条款为唯一权威副本。

## [38] LOS 段测距噪声 σ≈0.6m
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/los_noise_baseline.yaml` — LOS 段测距噪声基准配置（noise_value: 0.6, clause_id: 38）
- `configs/models/ekf.yaml:30-31` — EKF UWB 量测噪声 σ=0.6m（S4 LOS 基准 σ0.6m）
- `configs/models/robust_ekf.yaml:34-35` — Robust-EKF UWB 量测噪声 σ=0.6m（S4 LOS σ0.6m）
- `outputs/compliance_report_38.md` — 合规报告

**内容**: LOS 段测距噪声 σ≈0.6m（器件事例级均值误差，作为 LOS 基准，非协议档位参数）。LOS 段高斯噪声（商用 UWB 模块室内实测测距误差标准差在亚米级、LOS 与 NLOS 统计分布差异显著）。
## [39] UWB(DW1000级, 10Hz) 输出数据规范
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/uwb_output_spec.yaml` — UWB 输出数据规范配置（chip_model: DW1000级, frequency_hz: 10Hz）
- `src/liquidloc/analysis/clause_gates.py:3761-3927` — 验证函数 `p39_uwb_output_spec()`
- `outputs/compliance_report_39.md` — 合规报告

**内容**: UWB(DW1000级, 10Hz)；输出数据：`uwb_ranges`(N×4 测距, m) + `uwb_valid`(N×4 有效掩码)。UWB 传感器输出数据规范，包含测距矩阵与有效掩码矩阵。

## [40] IMU（消费级，150Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/imu_output_spec.yaml` — IMU 输出数据规范配置（chip_type: 消费级，frequency_hz: 150Hz）
- `src/liquidloc/analysis/clause_gates.py:3930-4110` — 验证函数 `p40_imu_output_spec()`
- `outputs/compliance_report_40.md` — 合规报告

**内容**: IMU（消费级，150Hz）；输出数据：`imu`（N_imu×6：ax/ay/az/gx/gy/gz，150Hz 原生，不降采样）+ `imu_t`（N_imu×1 自身时间戳）。IMU 传感器输出数据规范。

## [41] VIO（单目+IMU 前端，20 Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/vio_output_spec.yaml` — VIO 输出数据规范配置（frequency_hz: 20Hz, sensor_type: 单目+IMU前端）
- `src/liquidloc/analysis/clause_gates.py:4111-4324` — 验证函数 `p41_vio_output_spec()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_41.md` — 合规报告

**内容**: VIO（单目+IMU 前端，20 Hz）；输出数据：`vio_pos`（N×2/3）+ `vio_yaw`（N×1）+ `vio_valid`（N×1）+ 协方差（V0 健康档有效帧率 100%）。VIO 传感器输出数据规范。

## [42] 四个输出头
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/four_output_heads.yaml` — 四个输出头机制链配置（uwb_scaling/bias/vio_scaling/risk）
- `configs/protocol/imu_no_head_reasoning.yaml` — IMU 不设头推理说明配置
- `src/liquidloc/analysis/clause_gates.py:4326-4518` — 验证函数 `p42_imu_no_head_reasoning()`
- `src/liquidloc/analysis/clause_gates.py:4521-4665` — 验证函数 `p42b_four_output_heads()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_42.md` — 合规报告

**内容**: 四个输出头（uwb_scaling / bias / vio_scaling / risk）对三源量测做自适应加权、偏差补偿与门控，是"NLOS 识别 → 协方差调制/降权 → 鲁棒门控"机制链的落地（与 S4.2 鲁棒融合组合拳一致）。IMU 作为高频运动先验不参与头调制（惯性量测不经数据驱动加权，走 EKF 运动模型传播），故不设 IMU 头；risk 头为通道级联合风险。

## [43] 禁止使用档位空间外的参数、禁止自行定义插值档/新档
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_extra_levels.yaml` — 禁止额外档位配置（forbidden_interpolation: true, forbidden_new_levels: true）
- `src/liquidloc/analysis/clause_gates.py:4859-4960` — 验证函数 `p43_no_extra_levels()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_43.md` — 合规报告

**内容**: 档位参数严格取自档位协议条款档位表（A/N/V/K/M 五轴），禁止使用档位空间外的参数值，禁止自行定义插值档（如 N1.5 类）或任何新档位。

## [44] 档位注入纪律
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause44_injection_discipline.yaml` — 档位注入纪律配置（静态口径/禁止漂移档位）
- `src/liquidloc/analysis/clause_gates.py:4668-4770` — 验证函数 `p44_gear_injection_discipline()`
- `outputs/compliance_report_44.md` — 合规报告

**内容**: A轴为静态时钟偏移/延迟统计口径（数值见 Stage 1 档位协议条款 A 轴表），严格按静态口径注入，不引入随时间线性增长的时钟漂移作为额外档位（如需评估 → 协议裁决项 + 敏感性分析）。

## [45] N1.5 边界协议
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/n15_boundary.yaml` — N1.5 边界协议配置
- `src/liquidloc/analysis/clause_gates.py:4759-4855` — 验证函数 `p45_n15_boundary()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_45.md` — 合规报告

**内容**: N1.5（ρ=20–25%/μ=1.5–2.5m/σ=0.5m）不在本手册档位协议条款档位表内；本手册引用协议 N 轴为 N0–N3；N1.5 属于同机另一实验体系，按其各自对应协议文件执行。

## [46] 协议引用边界
**状态**: ✅ 已落地（由 p49_protocol_reference_boundary() 验证）
**文件**:
- `configs/protocol/protocol_reference_boundary.yaml` — 协议引用边界配置
- `src/liquidloc/analysis/clause_gates.py:5098` 起 — 验证函数 `p49_protocol_reference_boundary()`
- `outputs/compliance_report_46.md` — 合规报告

**内容**: 档位以 Stage 1 档位协议条款为唯一权威副本；其他文件引用档位时必须标明出自该条款，不得复制参数数值形成第二权威副本。注意：验证函数沿用 p49 命名（历史实现），实际验证手册 [46] 协议引用边界条款。

## [47] 三层证据结构总表
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/three_layer_evidence_structure.yaml` — 三层证据结构总表配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p47_three_layer_evidence()`
- `outputs/compliance_report_47.md` — 合规报告

**内容**: 三层证据结构总表（论文级数据架构，与 S3 轨迹规格衔接）：主场景（受控）S1–S3固定布局（20m×20m、4锚、4遮挡区）10seed×5方法=50单元；布局随机化泛化层；统计推断与主场景共享层。

## [48] 规模档位
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/scale_gear_spec.yaml` — 规模档位配置（入门 N_layout=6、推荐 N_layout=10）
- `src/liquidloc/analysis/clause_gates.py:5189-5290` — 验证函数 `p48_two_sigma_separation()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_48.md` — 合规报告

**内容**: 规模档位：入门 N_layout=6、推荐 N_layout=10（full + 2 个最强基线 LSTM/EKF × 2–3 seed 即可，泛化层只判排序不做因子推断，不需要主场景同等功效）。

## [49] 布局参数与方法排序的散点/箱线
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_scatter_plot_spec.yaml` — 布局参数散点配置
- `src/liquidloc/analysis/clause_gates.py`（p54_layout_parameter_scatter 验证函数）
- `outputs/compliance_report_49.md` — 合规报告

**内容**: 布局参数变化与方法排序关系以散点/箱线图呈现，禁止只用汇总统计量。

## [50] 测试轨迹与训练无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause50_test_train_no_overlap.yaml` — 测试轨迹与训练无重叠配置
- `src/liquidloc/analysis/clause_gates.py`（p50_test_train_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_50.md` — 合规报告

**内容**: 测试轨迹与训练数据在序列级别严格分离，不存在任何重叠。所有评估指标均在独立于训练集的数据上计算。


## [51] 划分按整条序列、训练与测试无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause51_sequence_split_no_overlap.yaml` — 序列级划分配置
- `src/liquidloc/analysis/clause_gates.py`（p51_sequence_split_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_51.md` — 合规报告

**内容**: 划分按整条序列进行，训练集与测试集之间不存在任何序列级别的重叠。禁止在同一序列内随机采样部分帧/窗口作为训练、部分帧/窗口作为测试，防止数据泄漏。

## [52] 基线不是裸模型
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/baseline_not_bare_model.yaml` — 基线不是裸模型配置
- `src/liquidloc/analysis/clause_gates.py`（p52_baseline_not_bare_model 验证函数）
- `outputs/compliance_report_52.md` — 合规报告

**内容**: 确认 LSTM/Transformer 基线确实带头运行——否则 LNN 赢它们="4 个头 + 液态骨干"双重优势，对比不公平。

## [53] 配对样本对齐
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/paired_sample_alignment.yaml` — 配对样本对齐配置
- `src/liquidloc/analysis/clause_gates.py:5658-5729` — 验证函数 `p53_paired_sample_alignment()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_53.md` — 合规报告

**内容**: 各方法评估按"同一测试轨迹 × 同 seed"配对。

## [54] 两两比较附Holm-Bonferroni多重比较校正
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/holm_bonferroni_correction.yaml` — Holm-Bonferroni校正配置
- `src/liquidloc/analysis/clause_gates.py`（p60_holm_bonferroni_correction 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_54.md` — 合规报告

**内容**: 五方法两两比较共10对，使用Holm-Bonferroni校正控制族错误率。多方法对比不做校正会放大假阳性；Holm-Bonferroni为稳健默认；ML比较应配套统计检验+CI+校正。

## [55] 扩样路径
**状态**: 落实中
**文件**:
- `configs/protocol/scale_expansion_path.yaml` — 扩样路径配置
- `src/liquidloc/analysis/clause_gates.py`（p55_scale_expansion_path 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_55.md` — 合规报告

**内容**: 10 seed 不足支撑功效时如实声明并给出扩样路径（只增不减：加 seed/轨迹至功效达标，规模恒 ≥ 论文级 10 seed × ≥100/seed，论文级硬约束——不存在"收窄/降档"方向的出口）。

## [56] 分析单位与样本独立性声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/analysis_unit_sample_independence.yaml` — 分析单位配置
- `src/liquidloc/analysis/clause_gates.py`（p64_analysis_unit_sample_independence 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_56.md` — 合规报告

**内容**: 滑窗W=128步（步长默认1帧）使相邻窗口重叠127/128，帧级误差样本高度自相关、不满足i.i.d.，所有统计检验必须以轨迹级为分析单位。

## [57] 先导实验（pilot）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/pilot_experiment.yaml` — 先导实验配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p57_sequence_split_no_overlap()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_57.md` — 合规报告

**内容**: 先导实验单独用于方案冻结前估计效应量/方差/可行性，规模<=1 seed×四组合各1-2条测试轨迹。

## [58] A2/A3 的偏移与抖动数值按协议生效
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/a2a3_shift_jitter.yaml` — A2/A3 偏移与抖动协议配置（clause_id="58"）
- `src/liquidloc/analysis/clause_gates.py::p68_a2a3_shift_jitter()` — 验证函数
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_58.md` — 合规报告
- `checkpoints/compliance_report_clause58.md` — 验证快照

**内容**: A2/A3的偏移与抖动数值按协议生效，已按A2/A3全参数逐项核对。A2：UWB时钟偏移20-50ms、UWB往返延迟40-80ms、IMU时钟偏移5-15ms、IMU采样抖动±3ms、VIO时钟偏移20-50ms、VIO处理延迟30-80ms。

## [59] 重采样方法声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/resampling_method_declaration.yaml` — 重采样方法声明配置
- `src/liquidloc/analysis/clause_gates.py`（p69_resampling_method 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_59.md` — 合规报告

**内容**: UWB/VIO重采样采用最近邻（nearest-neighbor）对齐到UWB帧时刻，不使用线性/样条插值生成测距或VIO值；声明写入论文方法节。

## [60] 边界/极端情况
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/edge_case_extreme.yaml` — 边界/极端情况配置
- `src/liquidloc/analysis/clause_gates.py`（p60_edge_case_extreme 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_60.md` — 合规报告

**内容**: 边界极端情况：空轨迹、极短序列、M1长间隙落在首尾，不产生人工尖峰/除零。

## [61] 同 seed 初始化
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/same_seed_init.yaml` — 同seed初始化配置
- `src/liquidloc/analysis/clause_gates.py`（p71_same_seed_init 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_61.md` — 合规报告

**内容**: 同seed初始化：LNN/LSTM/Transformer用同一seed的权重初始化。

## [62] 论文级规模参数
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/power_sample_size.yaml` — 规模参数配置
- `src/liquidloc/analysis/clause_gates.py`（p62_power_sample_size 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_62.md` — 合规报告

**内容**: 论文级规模参数：训练12-15条/seed，测试>=100条/seed，四组合各>=25，120s基准，10 seed（论文级基线）。

## [63] 布局随机化泛化层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_randomization.yaml` — 布局随机化配置
- `src/liquidloc/analysis/clause_gates.py`（p72_layout_randomization 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_63.md` — 合规报告
- `checkpoints/compliance_report_clause63.md` — checkpoint 验证报告

**内容**: 布局随机化泛化层：N_layout>=6(入门)/10(推荐)，每布局5方法x2-3 seed。

## [64] 重新生成全部 seed 数据
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/regenerate_all_seeds.yaml` — 重新生成全部 seed 配置
- `src/liquidloc/analysis/clause_gates.py`（p73_regenerate_all_seeds 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_64.md` — 合规报告
- `checkpoints/compliance_report_clause64.md` — checkpoint 验证报告

**内容**: 主场景10 seed全部重新生成，禁止新旧数据混用。

## [65] 新旧数据混用禁令（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_old_new_mixing.yaml` — 新旧数据混用禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p74_no_old_new_mixing 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_65.md` — 合规报告
- `checkpoints/compliance_report_clause65.md` — checkpoint 验证报告

**内容**: 禁止部分seed新+部分seed旧混用，硬约束。

## [66] 禁止把冒烟数据并入全量测试集
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_smoke_merge.yaml` — 冒烟数据禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p75_no_smoke_merge 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_66.md` — 合规报告
- `checkpoints/compliance_report_clause66.md` — checkpoint 验证报告

**内容**: 禁止冒烟>=12条测试数据并入全量测试集，整批化执行，硬约束。

## [67] 全量单元
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/full_scale_unit.yaml` — 全量单元配置
- `src/liquidloc/analysis/clause_gates.py`（p76_full_scale_unit 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_67.md` — 合规报告
- `checkpoints/compliance_report_clause67.md` — checkpoint 验证报告

**内容**: 全量单元：主场景10 seed×5方法=50单元全部用新数据训练。

## [68] 执行偏差（deviation）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/execution_deviation.yaml` — 执行偏差配置（含 deviation 表、E 系列审计、目录命名规范）
- `src/liquidloc/analysis/clause_gates.py:7103-7203` — 验证函数 `p68_execution_deviation()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_68.md` — 合规报告
- `checkpoints/compliance_report_clause68.md` — 验证快照

**内容**: 执行层面与方案的偏差（漏跑/换机/中断恢复）→ 记 execution deviation 表（E 系列审计覆盖），不影响协议与分析定义。实验目录结构 run 目录命名规范：`run-<日期>-<seed>-<method>-<config_hash>`。

## [69] CPU 角色分层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/cpu_role_description.yaml` — CPU 角色分层配置（cpu_non_training_only/cpu_idle_training/no_conflict/priority_rule）
- `src/liquidloc/analysis/clause_gates.py:7418-7519` — 验证函数 `p69_cpu_role_hierarchy()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_69.md` — 合规报告
- `checkpoints/compliance_report_clause69.md` — 验证快照

**内容**: "CPU 仅限非训练负载"与"CPU 空闲协助训练"的关系：前者是 GPU 空闲时的默认分工（CPU 只做数据生成/评估/预处理），后者是 GPU 排满后的追加自由度（CPU 可承接剩余训练单元）——两者不冲突，GPU 有空的按前者，GPU 排满的按后者。

## [70] 评估并行
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause70_evaluation_parallel.yaml` — 评估并行配置（gpu_execution=true, trajectory_batch_size=16, no_batch_mixing=true, evaluation_protocol_unchanged=true）
- `src/liquidloc/analysis/clause_gates.py`（p70_evaluation_parallel 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_70.md` — 合规报告
- `checkpoints/compliance_report_clause70.md` — checkpoint 验证报告

**内容**: 评估并行：测试轨迹推理在 GPU 上按轨迹独立并行执行（轨迹内窗口批大小 16，轨迹间不混批，评估协议不变）。

## [71] 父进程链可回溯
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/parent_process_traceable.yaml` — 父进程链可回溯配置（traceable=true, command="ps -o ppid= -p <pid>"）
- `src/liquidloc/analysis/clause_gates.py:8337-8417` — 验证函数 `p77_parent_process_traceable()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_71.md` — 合规报告
- `checkpoints/compliance_report_clause71.md` — checkpoint 验证报告

**内容**: 父进程链可回溯到本体系的启动器（`ps -o ppid= -p <pid>` 逐级上溯，最终父进程是本体系的调度/启动脚本）。

## [72] Transformer 滑窗同轨迹内合批
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/transformer_sliding_window.yaml` — Transformer 滑窗合批配置（enabled=true, method="同轨迹内窗口合批", log_batch_composition=true）
- `src/liquidloc/analysis/clause_gates.py:8420-8505` — 验证函数 `p78_transformer_sliding_window()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_72.md` — 合规报告
- `checkpoints/compliance_report_clause72.md` — checkpoint 验证报告

**内容**: Transformer 滑窗同轨迹内合批 评估日志打印批构成（同轨迹内窗口）。

## [73] GPU 报错停
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/gpu_error_handling.yaml` — GPU 报错处理配置（verified=true, cpu_idle_assistance=true, cpu_only_for_full_gpu=true）
- `src/liquidloc/analysis/clause_gates.py:8508-8592` — 验证函数 `p79_no_gpu_error_stop()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_73.md` — 合规报告
- `checkpoints/compliance_report_clause73.md` — checkpoint 验证报告

**内容**: 在无 GPU 环境实测 → 确认报错停而非静默 CPU 训练。CPU 空闲协助（执行模式/执行模式·CPU 空闲协助）CPU 承接仅限 GPU 排满后启用的剩余训练单元。

## [74] RAM 上限 12G
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/ram_limit_hard_cap.yaml` — RAM 上限配置（max_gb=12, verification_method="冒烟实测单单元峰值内存（psutil 探针）"）
- `src/liquidloc/analysis/clause_gates.py:8595-8676` — 验证函数 `p80_ram_limit_12g()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_74.md` — 合规报告
- `checkpoints/compliance_report_clause74.md` — checkpoint 验证报告

**内容**: RAM ≤12G 硬上限（执行模式/执行模式·RAM ≤12G 硬上限）；验证方法（细化，自测留痕）：冒烟实测单单元峰值内存（psutil 探针）。

## [75] 禁止借大修变口径
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_refactor_changes.yaml` — 禁止变口径配置（no_change_paper_level_values=true, no_change_data_gen=true, no_change_eval=true, no_change_protocol=true）
- `src/liquidloc/analysis/clause_gates.py:8679-8768` — 验证函数 `p81_no_refactor_changes()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_75.md` — 合规报告
- `checkpoints/compliance_report_clause75.md` — checkpoint 验证报告

**内容**: 禁止借大修变口径——大修只改代码组织与可维护性（模块拆分/常量收敛/断言补齐/命名统一），不得改变任何论文级数值、不得改数据生成/评估/统计口径、不得为了好写而改协议。

## [76]

## [15]

## [16]


## [17] 底线：落实证据必须引用文件:行号（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录落实证据引用规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十七落实证据引用配置文件（硬约束声明，含条款名加约束级别加治理阶段零等标志）
- 源码目录条款门禁验证模块（p一七拆解复核函数）
- 版本控制预提交钩子（落实证据引用检查调用点）
- 输出目录八字段落实报告（纯中文表述）
**判据**: 第0阶段与第1阶段每条落实记录的证据字段必须写真实文件路径 + 行号/章节锚点；只写文件名不写行号或行号与文件实际不符 = 未落实
## [5] 依赖声明（治理层·指令模式）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause05_dependency_declaration.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py:15259-15507` — p5_dependency_declaration() 验证函数
- `.git/hooks/pre-commit:111,229` — check_dependencies 前置依赖链硬拦截（定义@111、调用@229）
- `outputs/compliance_report_5.md` — 合规报告

**内容**: 每条指令头部标「前置:[N]」；执行前先查前置是否done，未done → 标blocked停下问，不硬跑、不跳序。pre-commit钩子check_dependencies在[2]-[6]依赖链上实施硬拦截（exit 1阻止提交）。



## [14] 底线：防统计作弊（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录防统计作弊规则文件 — 防统计作弊规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十四防统计作弊配置文件 — 防统计作弊配置（硬约束声明）
- 条款门禁源码 — 验证函数
- 版本控制预提交钩子 — 防统计作弊检查调用点
- 输出目录八字段落实报告 — 八字段落实报告
**判据**: 禁止选择性补跑/改数据到满意/旧数据复用/诊断轨并入验收轨

## [13] 底线：语言铁律（GLM 5.3 专属）
**状态**: ✅ 已落地
**文件**:
-  — 语言铁律规则文件（核心判据+禁止项+允许例外+违规处置）
-  — 语言铁律配置（clause_id=13, constraint_level=硬约束）
-  — 验证函数 
- [pre-commit] 前置检查 [2] 依赖 [1]: OK
[pre-commit] 前置检查 [3] 依赖 [1][2]: OK
[pre-commit] 前置检查 [4] 依赖 [1][2][3]: OK
[pre-commit] 前置检查 [5] 依赖 [1][2][3][4]: OK
[pre-commit] 前置检查 [6] 依赖 [1][2][3][4][5]: OK
[pre-commit] [8] 提交前自查: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | cfg_exists=True; cfg_parse_ok=True; clause_name_contains_8=True; constraint_level_is_hard=True; self_check_before_submit=True; check_all_reports_pass=True; check_status_table_done=True; check_manual_goals_met=True; check_eight_fields_complete=True; manual_exists=True; manual_contains_8=True; manual_contains_self_check=True; manual_contains_core_criteria=True; manual_contains_acceptance_table=True; hook_exists=True; hook_contains_p8_call=True; agents_md_exists=True; agents_md_has_clause_8_section=True; status_table_exists=True; status_table_has_clause_8=True; status_table_clause_8_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [9] 落实报告格式: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_all_eight_fields=True; rules_all_fields_have_three_levels=True; one_sentence_template=True; governance_homology=True; pre_submit_self_check=True; complete_examples_code=True; complete_examples_governance=True; judgment_iron_rule=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_9=True; eight_fields_spec=True; one_sentence_template_cfg=True; governance_homology_cfg=True; pre_submit_self_check_cfg=True; complete_examples_cfg=True; judgment_iron_rule_cfg=True; manual_exists=True; manual_contains_9=True; manual_contains_format=True; manual_contains_eight_fields=True; hook_exists=True; hook_contains_p9_call=True; agents_md_exists=True; agents_md_has_clause_9_section=True; status_table_exists=True; status_table_has_clause_9=True; status_table_clause_9_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [10] 逐条落实工序: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | skill_exists=True; step1_check=True; step2_implementation=True; step3_verification=True; step4_reporting=True; eight_fields_spec=True; self_check_list=True; failure_makes_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_10=True; agents_md_exists=True; agents_md_has_clause_10=True; status_table_exists=True; status_table_has_clause_10=True; status_table_clause_10_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p10_call=True
[pre-commit] [11] 条款归属总则: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_phase0=True; rules_has_phase1_6=True; rules_has_criterion_merged=True; rules_has_phase7_8=True; rules_has_final_check=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_11=True; agents_md_exists=True; agents_md_has_clause_11=True; status_table_exists=True; status_table_has_clause_11=True; status_table_clause_11_field_valid=True; manual_exists=True; manual_contains_11=True; manual_contains_attribution=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p11_call=True
[pre-commit] [12] 手册目标: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | manual_exists=True; manual_has_criterion=True; manual_has_line_a=True; manual_has_line_b=True; manual_has_aux=True; manual_has_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_12=True; agents_md_exists=True; agents_md_has_clause_12=True; status_table_exists=True; status_table_has_clause_12=True; status_table_clause_12_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p12_call=True — 语言铁律检查调用点（仅检查落实报告文件）
-  — 八字段落实报告
**判据**: 全程简体中文；禁止英文回答/禁止中英混杂；3处及以上违规=报告作废重做

## [6] 代码与文件冗余治理（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause06_redundancy_governance.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p6_redundancy_governance() 验证函数
- `outputs/compliance_report_6.md` — 合规报告

**内容**: 禁止为同一逻辑留多份拷贝、禁止复制旧文件再改的备份文化，历史版本走 git，工作目录只保留当前生效版本；发现 _backup/_old/_copy 类冗余文件 → 移入 archive 或删除；禁止"注释掉一大段代码留作备用"；清理动作只作用于本体系目录，禁止跨体系删改；清理任何进程前必须先判定归属。

## [7] 违反即处置（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause07_violation_disposition.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p7_violation_disposition() 验证函数
- `rules/判定铁律.md` — 违反即处置章节（五、违反即处置）
- `outputs/compliance_report_7.md` — 合规报告

**内容**: 执行中任何环节发现违反本文档（未清场/审计失败/口径不符/数据或权重污染）→ 隔离作废相关产物（archive 留痕）→ 修复根因 → 回到对应回归点重跑，处置证据随结果一并提交；任何"未达预期"的第一解释永远是执行偏差，不是方法不行、不是数据集不行；窗口外先修根因再谈结论。

## [8] 提交前自查（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause08_submit_self_check.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p8_submit_self_check() 验证函数
- `.git/hooks/pre-commit` — pre-commit 自查调用点（p8调用在check_dependencies之后）
- `outputs/compliance_report_8.md` — 合规报告

**内容**: 向用户提交论文级结果前先逐项自查是否符合手册目标（核心判据 + 验收表），全部符合才整理提交。

## [10] 逐条落实工序（治理层·四步闭环）
**状态**: ✅ 已落地
**文件**:
- `skills/逐条落实工序/SKILL.md` — 四步闭环工序唯一权威载体（查→落→验→报）
- `configs/protocol/clause10_four_step_closure.yaml` — 约束配置（硬约束声明、四步标志）
- `src/liquidloc/analysis/clause_gates.py` — p10_four_step_closure() 验证函数
- `.git/hooks/pre-commit` — pre-commit 四步工序验证调用点（p10调用在p9之后）
- `outputs/compliance_report_10.md` — 合规报告

**内容**: 每条指令按「查 → 落 → 验 → 报」四步闭环：查（读条款原文+查磁盘现状，禁止凭记忆）→ 落（翻译为文件路径+函数/类+行号落点）→ 验（逐行核对条款逐句对应，不落实当场改）→ 报（按八字段写落实记录，未写记录视为未落实）。

## [11] 条款归属总则（治理层）
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 三阶段归属总则（治理层/Stage1-6落实类/Stage7-8执行决策类）
- `configs/protocol/clause11_clause_attribution.yaml` — 约束配置（硬约束声明、归属标志）
- `src/liquidloc/analysis/clause_gates.py` — p11_clause_attribution() 验证函数
- `.git/hooks/pre-commit` — pre-commit 归属总则验证调用点（p11调用在p10之后）
- `outputs/compliance_report_11.md` — 合规报告

**内容**: 治理层条款→第0阶段逐条落实并写报告，证据写对应治理载体（按『治理载体分散布局』分散落点：AGENTS/rules/skill/hook/plugin/MCP）；Stage 1-6落实类→第1阶段代码落实，判据条款的判据已并入对应代码条款；Stage 7-8执行决策类→第2阶段执行，每条同样单独写执行落实报告。

## [12] 手册目标（治理层·最终检查只看这个）
**状态**: ✅ 已落地
**文件**:
- `rules/手册目标.md` — 核心判据唯一判据（主线A/B必须同时达成）
- `configs/protocol/clause12_manual_objectives.yaml` — 约束配置（硬约束声明、判据标志）
- `src/liquidloc/analysis/clause_gates.py` — p12_manual_objectives() 验证函数
- `.git/hooks/pre-commit` — pre-commit 手册目标验证调用点（p12调用在p11之后）
- `outputs/compliance_report_12.md` — 合规报告

**内容**: 防异步高NLOS实验RMSE整体过大（异步+NLOS下误差过大时所有方法一起高）；防LNN（Liquid液态骨干网络，proposed）的RMSE不是最低；两项判据统一为唯一判据，跑前冒烟/预达标与跑后判定共用；验收分级与降级出口：验收表每一项必须在完整验收路径下逐项回查修通，诊断轨/冒烟数据不得以任何形式并入、替代或补充验收轨数据；数据/代码/结果可追溯（manifest + git commit + 验收留痕）；核验通过后最终只看目标是否达成，不看流程勾选。

## [9] 落实报告格式（治理层·八字段）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause09_compliance_report_format.yaml` — 约束配置
- `rules/落实报告格式规范.md` — 八字段格式规范（唯一权威副本，含八字段逐字段说明/一句话模板/治理类同构/提交前自检/完整示例/判定铁律）
- `.agents/skills/execution-report-gate/SKILL.md` — §1 为八字段格式唯一逐字权威副本（入口）
- `src/liquidloc/analysis/clause_gates.py` — p9_compliance_report_format() 验证函数
- `.git/hooks/pre-commit` — pre-commit 八字段验证调用点（p9调用在p8之后）
- `outputs/compliance_report_9.md` — 合规报告

**内容**: 每条指令输出 = 落实报告格式一行，八字段逐字段规范（条款号/状态/条款逐字抄录/代码绝对定位/落实极尽具体+全量映射/审查问题→修复+循环轨迹/验证命令→预期/证据真实路径+stdout落盘），含一句话模板、治理类同构、提交前自检、完整示例、判定铁律；缺任一字段或字段不合格 = 该条未落实。细则进 `rules/落实报告格式规范.md`，不堆进 AGENTS.md。

## [36] git 是最后防线
**状态**: ✅ 已落地
**文件**: 
- `configs/protocol/git_is_last_resort.yaml` — 约束配置
- `.git/hooks/pre-commit` — pre-commit 防线（第 1-161 行）
- `.gitignore:30-31` — *.log 和 *.bak 防护规则
- `outputs/compliance_report_36.md` — 合规报告

**内容**: git 仓库与工作区必须一致，任何版本回退/文件损坏/误改恢复一律走 git（checkout/revert），禁止复制 _backup/_old 副本。

## [37] 档位定义声明
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/gear_definition_declaration.yaml` — 约束配置
- `configs/protocol/axis_a_sync.yaml` — A 轴档位协议
- `configs/protocol/axis_n_nlos.yaml` — N 轴档位协议
- `configs/protocol/axis_v_visual.yaml` — V 轴档位协议
- `configs/protocol/axis_k_anchor.yaml` — K 轴档位协议
- `configs/protocol/axis_m_missing.yaml` — M 轴档位协议
- `src/liquidloc/analysis/clause_gates.py:3242-3395` — 验证函数 `p37_gear_definition_declaration()`
- `outputs/compliance_report_37.md` — 合规报告

**内容**: A/N/V/K/M 各轴档位划分、参数与具体数值以 Stage 1 档位协议条款为唯一权威副本。

## [38] LOS 段测距噪声 σ≈0.6m
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/los_noise_baseline.yaml` — LOS 段测距噪声基准配置（noise_value: 0.6, clause_id: 38）
- `configs/models/ekf.yaml:30-31` — EKF UWB 量测噪声 σ=0.6m（S4 LOS 基准 σ0.6m）
- `configs/models/robust_ekf.yaml:34-35` — Robust-EKF UWB 量测噪声 σ=0.6m（S4 LOS σ0.6m）
- `outputs/compliance_report_38.md` — 合规报告

**内容**: LOS 段测距噪声 σ≈0.6m（器件事例级均值误差，作为 LOS 基准，非协议档位参数）。LOS 段高斯噪声（商用 UWB 模块室内实测测距误差标准差在亚米级、LOS 与 NLOS 统计分布差异显著）。
## [39] UWB(DW1000级, 10Hz) 输出数据规范
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/uwb_output_spec.yaml` — UWB 输出数据规范配置（chip_model: DW1000级, frequency_hz: 10Hz）
- `src/liquidloc/analysis/clause_gates.py:3761-3927` — 验证函数 `p39_uwb_output_spec()`
- `outputs/compliance_report_39.md` — 合规报告

**内容**: UWB(DW1000级, 10Hz)；输出数据：`uwb_ranges`(N×4 测距, m) + `uwb_valid`(N×4 有效掩码)。UWB 传感器输出数据规范，包含测距矩阵与有效掩码矩阵。

## [40] IMU（消费级，150Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/imu_output_spec.yaml` — IMU 输出数据规范配置（chip_type: 消费级，frequency_hz: 150Hz）
- `src/liquidloc/analysis/clause_gates.py:3930-4110` — 验证函数 `p40_imu_output_spec()`
- `outputs/compliance_report_40.md` — 合规报告

**内容**: IMU（消费级，150Hz）；输出数据：`imu`（N_imu×6：ax/ay/az/gx/gy/gz，150Hz 原生，不降采样）+ `imu_t`（N_imu×1 自身时间戳）。IMU 传感器输出数据规范。

## [41] VIO（单目+IMU 前端，20 Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/vio_output_spec.yaml` — VIO 输出数据规范配置（frequency_hz: 20Hz, sensor_type: 单目+IMU前端）
- `src/liquidloc/analysis/clause_gates.py:4111-4324` — 验证函数 `p41_vio_output_spec()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_41.md` — 合规报告

**内容**: VIO（单目+IMU 前端，20 Hz）；输出数据：`vio_pos`（N×2/3）+ `vio_yaw`（N×1）+ `vio_valid`（N×1）+ 协方差（V0 健康档有效帧率 100%）。VIO 传感器输出数据规范。

## [42] 四个输出头
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/four_output_heads.yaml` — 四个输出头机制链配置（uwb_scaling/bias/vio_scaling/risk）
- `configs/protocol/imu_no_head_reasoning.yaml` — IMU 不设头推理说明配置
- `src/liquidloc/analysis/clause_gates.py:4326-4518` — 验证函数 `p42_imu_no_head_reasoning()`
- `src/liquidloc/analysis/clause_gates.py:4521-4665` — 验证函数 `p42b_four_output_heads()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_42.md` — 合规报告

**内容**: 四个输出头（uwb_scaling / bias / vio_scaling / risk）对三源量测做自适应加权、偏差补偿与门控，是"NLOS 识别 → 协方差调制/降权 → 鲁棒门控"机制链的落地（与 S4.2 鲁棒融合组合拳一致）。IMU 作为高频运动先验不参与头调制（惯性量测不经数据驱动加权，走 EKF 运动模型传播），故不设 IMU 头；risk 头为通道级联合风险。

## [43] 禁止使用档位空间外的参数、禁止自行定义插值档/新档
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_extra_levels.yaml` — 禁止额外档位配置（forbidden_interpolation: true, forbidden_new_levels: true）
- `src/liquidloc/analysis/clause_gates.py:4859-4960` — 验证函数 `p43_no_extra_levels()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_43.md` — 合规报告

**内容**: 档位参数严格取自档位协议条款档位表（A/N/V/K/M 五轴），禁止使用档位空间外的参数值，禁止自行定义插值档（如 N1.5 类）或任何新档位。

## [44] 档位注入纪律
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause44_injection_discipline.yaml` — 档位注入纪律配置（静态口径/禁止漂移档位）
- `src/liquidloc/analysis/clause_gates.py:4668-4770` — 验证函数 `p44_gear_injection_discipline()`
- `outputs/compliance_report_44.md` — 合规报告

**内容**: A轴为静态时钟偏移/延迟统计口径（数值见 Stage 1 档位协议条款 A 轴表），严格按静态口径注入，不引入随时间线性增长的时钟漂移作为额外档位（如需评估 → 协议裁决项 + 敏感性分析）。

## [45] N1.5 边界协议
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/n15_boundary.yaml` — N1.5 边界协议配置
- `src/liquidloc/analysis/clause_gates.py:4759-4855` — 验证函数 `p45_n15_boundary()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_45.md` — 合规报告

**内容**: N1.5（ρ=20–25%/μ=1.5–2.5m/σ=0.5m）不在本手册档位协议条款档位表内；本手册引用协议 N 轴为 N0–N3；N1.5 属于同机另一实验体系，按其各自对应协议文件执行。

## [46] 协议引用边界
**状态**: ✅ 已落地（由 p49_protocol_reference_boundary() 验证）
**文件**:
- `configs/protocol/protocol_reference_boundary.yaml` — 协议引用边界配置
- `src/liquidloc/analysis/clause_gates.py:5098` 起 — 验证函数 `p49_protocol_reference_boundary()`
- `outputs/compliance_report_46.md` — 合规报告

**内容**: 档位以 Stage 1 档位协议条款为唯一权威副本；其他文件引用档位时必须标明出自该条款，不得复制参数数值形成第二权威副本。注意：验证函数沿用 p49 命名（历史实现），实际验证手册 [46] 协议引用边界条款。

## [47] 三层证据结构总表
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/three_layer_evidence_structure.yaml` — 三层证据结构总表配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p47_three_layer_evidence()`
- `outputs/compliance_report_47.md` — 合规报告

**内容**: 三层证据结构总表（论文级数据架构，与 S3 轨迹规格衔接）：主场景（受控）S1–S3固定布局（20m×20m、4锚、4遮挡区）10seed×5方法=50单元；布局随机化泛化层；统计推断与主场景共享层。

## [48] 规模档位
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/scale_gear_spec.yaml` — 规模档位配置（入门 N_layout=6、推荐 N_layout=10）
- `src/liquidloc/analysis/clause_gates.py:5189-5290` — 验证函数 `p48_two_sigma_separation()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_48.md` — 合规报告

**内容**: 规模档位：入门 N_layout=6、推荐 N_layout=10（full + 2 个最强基线 LSTM/EKF × 2–3 seed 即可，泛化层只判排序不做因子推断，不需要主场景同等功效）。

## [49] 布局参数与方法排序的散点/箱线
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_scatter_plot_spec.yaml` — 布局参数散点配置
- `src/liquidloc/analysis/clause_gates.py`（p54_layout_parameter_scatter 验证函数）
- `outputs/compliance_report_49.md` — 合规报告

**内容**: 布局参数变化与方法排序关系以散点/箱线图呈现，禁止只用汇总统计量。

## [50] 测试轨迹与训练无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause50_test_train_no_overlap.yaml` — 测试轨迹与训练无重叠配置
- `src/liquidloc/analysis/clause_gates.py`（p50_test_train_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_50.md` — 合规报告

**内容**: 测试轨迹与训练数据在序列级别严格分离，不存在任何重叠。所有评估指标均在独立于训练集的数据上计算。


## [51] 划分按整条序列、训练与测试无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause51_sequence_split_no_overlap.yaml` — 序列级划分配置
- `src/liquidloc/analysis/clause_gates.py`（p51_sequence_split_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_51.md` — 合规报告

**内容**: 划分按整条序列进行，训练集与测试集之间不存在任何序列级别的重叠。禁止在同一序列内随机采样部分帧/窗口作为训练、部分帧/窗口作为测试，防止数据泄漏。

## [52] 基线不是裸模型
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/baseline_not_bare_model.yaml` — 基线不是裸模型配置
- `src/liquidloc/analysis/clause_gates.py`（p52_baseline_not_bare_model 验证函数）
- `outputs/compliance_report_52.md` — 合规报告

**内容**: 确认 LSTM/Transformer 基线确实带头运行——否则 LNN 赢它们="4 个头 + 液态骨干"双重优势，对比不公平。

## [53] 配对样本对齐
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/paired_sample_alignment.yaml` — 配对样本对齐配置
- `src/liquidloc/analysis/clause_gates.py:5658-5729` — 验证函数 `p53_paired_sample_alignment()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_53.md` — 合规报告

**内容**: 各方法评估按"同一测试轨迹 × 同 seed"配对。

## [54] 两两比较附Holm-Bonferroni多重比较校正
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/holm_bonferroni_correction.yaml` — Holm-Bonferroni校正配置
- `src/liquidloc/analysis/clause_gates.py`（p60_holm_bonferroni_correction 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_54.md` — 合规报告

**内容**: 五方法两两比较共10对，使用Holm-Bonferroni校正控制族错误率。多方法对比不做校正会放大假阳性；Holm-Bonferroni为稳健默认；ML比较应配套统计检验+CI+校正。

## [55] 扩样路径
**状态**: 落实中
**文件**:
- `configs/protocol/scale_expansion_path.yaml` — 扩样路径配置
- `src/liquidloc/analysis/clause_gates.py`（p55_scale_expansion_path 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_55.md` — 合规报告

**内容**: 10 seed 不足支撑功效时如实声明并给出扩样路径（只增不减：加 seed/轨迹至功效达标，规模恒 ≥ 论文级 10 seed × ≥100/seed，论文级硬约束——不存在"收窄/降档"方向的出口）。

## [56] 分析单位与样本独立性声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/analysis_unit_sample_independence.yaml` — 分析单位配置
- `src/liquidloc/analysis/clause_gates.py`（p64_analysis_unit_sample_independence 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_56.md` — 合规报告

**内容**: 滑窗W=128步（步长默认1帧）使相邻窗口重叠127/128，帧级误差样本高度自相关、不满足i.i.d.，所有统计检验必须以轨迹级为分析单位。

## [57] 先导实验（pilot）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/pilot_experiment.yaml` — 先导实验配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p57_sequence_split_no_overlap()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_57.md` — 合规报告

**内容**: 先导实验单独用于方案冻结前估计效应量/方差/可行性，规模<=1 seed×四组合各1-2条测试轨迹。

## [58] A2/A3 的偏移与抖动数值按协议生效
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/a2a3_shift_jitter.yaml` — A2/A3 偏移与抖动协议配置（clause_id="58"）
- `src/liquidloc/analysis/clause_gates.py::p68_a2a3_shift_jitter()` — 验证函数
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_58.md` — 合规报告
- `checkpoints/compliance_report_clause58.md` — 验证快照

**内容**: A2/A3的偏移与抖动数值按协议生效，已按A2/A3全参数逐项核对。A2：UWB时钟偏移20-50ms、UWB往返延迟40-80ms、IMU时钟偏移5-15ms、IMU采样抖动±3ms、VIO时钟偏移20-50ms、VIO处理延迟30-80ms。

## [59] 重采样方法声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/resampling_method_declaration.yaml` — 重采样方法声明配置
- `src/liquidloc/analysis/clause_gates.py`（p69_resampling_method 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_59.md` — 合规报告

**内容**: UWB/VIO重采样采用最近邻（nearest-neighbor）对齐到UWB帧时刻，不使用线性/样条插值生成测距或VIO值；声明写入论文方法节。

## [60] 边界/极端情况
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/edge_case_extreme.yaml` — 边界/极端情况配置
- `src/liquidloc/analysis/clause_gates.py`（p60_edge_case_extreme 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_60.md` — 合规报告

**内容**: 边界极端情况：空轨迹、极短序列、M1长间隙落在首尾，不产生人工尖峰/除零。

## [61] 同 seed 初始化
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/same_seed_init.yaml` — 同seed初始化配置
- `src/liquidloc/analysis/clause_gates.py`（p71_same_seed_init 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_61.md` — 合规报告

**内容**: 同seed初始化：LNN/LSTM/Transformer用同一seed的权重初始化。

## [62] 论文级规模参数
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/power_sample_size.yaml` — 规模参数配置
- `src/liquidloc/analysis/clause_gates.py`（p62_power_sample_size 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_62.md` — 合规报告

**内容**: 论文级规模参数：训练12-15条/seed，测试>=100条/seed，四组合各>=25，120s基准，10 seed（论文级基线）。

## [63] 布局随机化泛化层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_randomization.yaml` — 布局随机化配置
- `src/liquidloc/analysis/clause_gates.py`（p72_layout_randomization 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_63.md` — 合规报告
- `checkpoints/compliance_report_clause63.md` — checkpoint 验证报告

**内容**: 布局随机化泛化层：N_layout>=6(入门)/10(推荐)，每布局5方法x2-3 seed。

## [64] 重新生成全部 seed 数据
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/regenerate_all_seeds.yaml` — 重新生成全部 seed 配置
- `src/liquidloc/analysis/clause_gates.py`（p73_regenerate_all_seeds 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_64.md` — 合规报告
- `checkpoints/compliance_report_clause64.md` — checkpoint 验证报告

**内容**: 主场景10 seed全部重新生成，禁止新旧数据混用。

## [65] 新旧数据混用禁令（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_old_new_mixing.yaml` — 新旧数据混用禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p74_no_old_new_mixing 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_65.md` — 合规报告
- `checkpoints/compliance_report_clause65.md` — checkpoint 验证报告

**内容**: 禁止部分seed新+部分seed旧混用，硬约束。

## [66] 禁止把冒烟数据并入全量测试集
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_smoke_merge.yaml` — 冒烟数据禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p75_no_smoke_merge 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_66.md` — 合规报告
- `checkpoints/compliance_report_clause66.md` — checkpoint 验证报告

**内容**: 禁止冒烟>=12条测试数据并入全量测试集，整批化执行，硬约束。

## [67] 全量单元
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/full_scale_unit.yaml` — 全量单元配置
- `src/liquidloc/analysis/clause_gates.py`（p76_full_scale_unit 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_67.md` — 合规报告
- `checkpoints/compliance_report_clause67.md` — checkpoint 验证报告

**内容**: 全量单元：主场景10 seed×5方法=50单元全部用新数据训练。

## [68] 执行偏差（deviation）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/execution_deviation.yaml` — 执行偏差配置（含 deviation 表、E 系列审计、目录命名规范）
- `src/liquidloc/analysis/clause_gates.py:7103-7203` — 验证函数 `p68_execution_deviation()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_68.md` — 合规报告
- `checkpoints/compliance_report_clause68.md` — 验证快照

**内容**: 执行层面与方案的偏差（漏跑/换机/中断恢复）→ 记 execution deviation 表（E 系列审计覆盖），不影响协议与分析定义。实验目录结构 run 目录命名规范：`run-<日期>-<seed>-<method>-<config_hash>`。

## [69] CPU 角色分层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/cpu_role_description.yaml` — CPU 角色分层配置（cpu_non_training_only/cpu_idle_training/no_conflict/priority_rule）
- `src/liquidloc/analysis/clause_gates.py:7418-7519` — 验证函数 `p69_cpu_role_hierarchy()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_69.md` — 合规报告
- `checkpoints/compliance_report_clause69.md` — 验证快照

**内容**: "CPU 仅限非训练负载"与"CPU 空闲协助训练"的关系：前者是 GPU 空闲时的默认分工（CPU 只做数据生成/评估/预处理），后者是 GPU 排满后的追加自由度（CPU 可承接剩余训练单元）——两者不冲突，GPU 有空的按前者，GPU 排满的按后者。

## [70] 评估并行
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause70_evaluation_parallel.yaml` — 评估并行配置（gpu_execution=true, trajectory_batch_size=16, no_batch_mixing=true, evaluation_protocol_unchanged=true）
- `src/liquidloc/analysis/clause_gates.py`（p70_evaluation_parallel 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_70.md` — 合规报告
- `checkpoints/compliance_report_clause70.md` — checkpoint 验证报告

**内容**: 评估并行：测试轨迹推理在 GPU 上按轨迹独立并行执行（轨迹内窗口批大小 16，轨迹间不混批，评估协议不变）。

## [71] 父进程链可回溯
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/parent_process_traceable.yaml` — 父进程链可回溯配置（traceable=true, command="ps -o ppid= -p <pid>"）
- `src/liquidloc/analysis/clause_gates.py:8337-8417` — 验证函数 `p77_parent_process_traceable()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_71.md` — 合规报告
- `checkpoints/compliance_report_clause71.md` — checkpoint 验证报告

**内容**: 父进程链可回溯到本体系的启动器（`ps -o ppid= -p <pid>` 逐级上溯，最终父进程是本体系的调度/启动脚本）。

## [72] Transformer 滑窗同轨迹内合批
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/transformer_sliding_window.yaml` — Transformer 滑窗合批配置（enabled=true, method="同轨迹内窗口合批", log_batch_composition=true）
- `src/liquidloc/analysis/clause_gates.py:8420-8505` — 验证函数 `p78_transformer_sliding_window()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_72.md` — 合规报告
- `checkpoints/compliance_report_clause72.md` — checkpoint 验证报告

**内容**: Transformer 滑窗同轨迹内合批 评估日志打印批构成（同轨迹内窗口）。

## [73] GPU 报错停
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/gpu_error_handling.yaml` — GPU 报错处理配置（verified=true, cpu_idle_assistance=true, cpu_only_for_full_gpu=true）
- `src/liquidloc/analysis/clause_gates.py:8508-8592` — 验证函数 `p79_no_gpu_error_stop()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_73.md` — 合规报告
- `checkpoints/compliance_report_clause73.md` — checkpoint 验证报告

**内容**: 在无 GPU 环境实测 → 确认报错停而非静默 CPU 训练。CPU 空闲协助（执行模式/执行模式·CPU 空闲协助）CPU 承接仅限 GPU 排满后启用的剩余训练单元。

## [74] RAM 上限 12G
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/ram_limit_hard_cap.yaml` — RAM 上限配置（max_gb=12, verification_method="冒烟实测单单元峰值内存（psutil 探针）"）
- `src/liquidloc/analysis/clause_gates.py:8595-8676` — 验证函数 `p80_ram_limit_12g()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_74.md` — 合规报告
- `checkpoints/compliance_report_clause74.md` — checkpoint 验证报告

**内容**: RAM ≤12G 硬上限（执行模式/执行模式·RAM ≤12G 硬上限）；验证方法（细化，自测留痕）：冒烟实测单单元峰值内存（psutil 探针）。

## [75] 禁止借大修变口径
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_refactor_changes.yaml` — 禁止变口径配置（no_change_paper_level_values=true, no_change_data_gen=true, no_change_eval=true, no_change_protocol=true）
- `src/liquidloc/analysis/clause_gates.py:8679-8768` — 验证函数 `p81_no_refactor_changes()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_75.md` — 合规报告
- `checkpoints/compliance_report_clause75.md` — checkpoint 验证报告

**内容**: 禁止借大修变口径——大修只改代码组织与可维护性（模块拆分/常量收敛/断言补齐/命名统一），不得改变任何论文级数值、不得改数据生成/评估/统计口径、不得为了好写而改协议。

## [76]

## [15]


## [16] 底线：第1阶段附加义务（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录第1阶段附加义务规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十六第1阶段附加义务配置文件（硬约束声明，含条款名加约束级别加治理阶段零等标志）
- 源码目录条款门禁验证模块（p一六资源冲突错峰排队调度函数）
- 版本控制预提交钩子（第1阶段附加义务检查调用点）
- 输出目录八字段落实报告（纯中文表述）
**判据**: 落实中发现缺陷→顺手修根因（禁止留待办事项/绕开/注释旧实现）；缺陷检查范围=阶段一至六条款映射代码；每条条款落实完成时做一次缺陷检查（静态审查/运行断言/单测至少一种）；发现即修根因；修复检查结果写入八字段报告"审查:"字段并留痕
## [5] 依赖声明（治理层·指令模式）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause05_dependency_declaration.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py:15259-15507` — p5_dependency_declaration() 验证函数
- `.git/hooks/pre-commit:111,229` — check_dependencies 前置依赖链硬拦截（定义@111、调用@229）
- `outputs/compliance_report_5.md` — 合规报告

**内容**: 每条指令头部标「前置:[N]」；执行前先查前置是否done，未done → 标blocked停下问，不硬跑、不跳序。pre-commit钩子check_dependencies在[2]-[6]依赖链上实施硬拦截（exit 1阻止提交）。



## [14] 底线：防统计作弊（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录防统计作弊规则文件 — 防统计作弊规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十四防统计作弊配置文件 — 防统计作弊配置（硬约束声明）
- 条款门禁源码 — 验证函数
- 版本控制预提交钩子 — 防统计作弊检查调用点
- 输出目录八字段落实报告 — 八字段落实报告
**判据**: 禁止选择性补跑/改数据到满意/旧数据复用/诊断轨并入验收轨

## [13] 底线：语言铁律（GLM 5.3 专属）
**状态**: ✅ 已落地
**文件**:
-  — 语言铁律规则文件（核心判据+禁止项+允许例外+违规处置）
-  — 语言铁律配置（clause_id=13, constraint_level=硬约束）
-  — 验证函数 
- [pre-commit] 前置检查 [2] 依赖 [1]: OK
[pre-commit] 前置检查 [3] 依赖 [1][2]: OK
[pre-commit] 前置检查 [4] 依赖 [1][2][3]: OK
[pre-commit] 前置检查 [5] 依赖 [1][2][3][4]: OK
[pre-commit] 前置检查 [6] 依赖 [1][2][3][4][5]: OK
[pre-commit] [8] 提交前自查: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | cfg_exists=True; cfg_parse_ok=True; clause_name_contains_8=True; constraint_level_is_hard=True; self_check_before_submit=True; check_all_reports_pass=True; check_status_table_done=True; check_manual_goals_met=True; check_eight_fields_complete=True; manual_exists=True; manual_contains_8=True; manual_contains_self_check=True; manual_contains_core_criteria=True; manual_contains_acceptance_table=True; hook_exists=True; hook_contains_p8_call=True; agents_md_exists=True; agents_md_has_clause_8_section=True; status_table_exists=True; status_table_has_clause_8=True; status_table_clause_8_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [9] 落实报告格式: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_all_eight_fields=True; rules_all_fields_have_three_levels=True; one_sentence_template=True; governance_homology=True; pre_submit_self_check=True; complete_examples_code=True; complete_examples_governance=True; judgment_iron_rule=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_9=True; eight_fields_spec=True; one_sentence_template_cfg=True; governance_homology_cfg=True; pre_submit_self_check_cfg=True; complete_examples_cfg=True; judgment_iron_rule_cfg=True; manual_exists=True; manual_contains_9=True; manual_contains_format=True; manual_contains_eight_fields=True; hook_exists=True; hook_contains_p9_call=True; agents_md_exists=True; agents_md_has_clause_9_section=True; status_table_exists=True; status_table_has_clause_9=True; status_table_clause_9_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [10] 逐条落实工序: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | skill_exists=True; step1_check=True; step2_implementation=True; step3_verification=True; step4_reporting=True; eight_fields_spec=True; self_check_list=True; failure_makes_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_10=True; agents_md_exists=True; agents_md_has_clause_10=True; status_table_exists=True; status_table_has_clause_10=True; status_table_clause_10_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p10_call=True
[pre-commit] [11] 条款归属总则: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_phase0=True; rules_has_phase1_6=True; rules_has_criterion_merged=True; rules_has_phase7_8=True; rules_has_final_check=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_11=True; agents_md_exists=True; agents_md_has_clause_11=True; status_table_exists=True; status_table_has_clause_11=True; status_table_clause_11_field_valid=True; manual_exists=True; manual_contains_11=True; manual_contains_attribution=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p11_call=True
[pre-commit] [12] 手册目标: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | manual_exists=True; manual_has_criterion=True; manual_has_line_a=True; manual_has_line_b=True; manual_has_aux=True; manual_has_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_12=True; agents_md_exists=True; agents_md_has_clause_12=True; status_table_exists=True; status_table_has_clause_12=True; status_table_clause_12_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p12_call=True — 语言铁律检查调用点（仅检查落实报告文件）
-  — 八字段落实报告
**判据**: 全程简体中文；禁止英文回答/禁止中英混杂；3处及以上违规=报告作废重做

## [6] 代码与文件冗余治理（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause06_redundancy_governance.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p6_redundancy_governance() 验证函数
- `outputs/compliance_report_6.md` — 合规报告

**内容**: 禁止为同一逻辑留多份拷贝、禁止复制旧文件再改的备份文化，历史版本走 git，工作目录只保留当前生效版本；发现 _backup/_old/_copy 类冗余文件 → 移入 archive 或删除；禁止"注释掉一大段代码留作备用"；清理动作只作用于本体系目录，禁止跨体系删改；清理任何进程前必须先判定归属。

## [7] 违反即处置（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause07_violation_disposition.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p7_violation_disposition() 验证函数
- `rules/判定铁律.md` — 违反即处置章节（五、违反即处置）
- `outputs/compliance_report_7.md` — 合规报告

**内容**: 执行中任何环节发现违反本文档（未清场/审计失败/口径不符/数据或权重污染）→ 隔离作废相关产物（archive 留痕）→ 修复根因 → 回到对应回归点重跑，处置证据随结果一并提交；任何"未达预期"的第一解释永远是执行偏差，不是方法不行、不是数据集不行；窗口外先修根因再谈结论。

## [8] 提交前自查（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause08_submit_self_check.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p8_submit_self_check() 验证函数
- `.git/hooks/pre-commit` — pre-commit 自查调用点（p8调用在check_dependencies之后）
- `outputs/compliance_report_8.md` — 合规报告

**内容**: 向用户提交论文级结果前先逐项自查是否符合手册目标（核心判据 + 验收表），全部符合才整理提交。

## [10] 逐条落实工序（治理层·四步闭环）
**状态**: ✅ 已落地
**文件**:
- `skills/逐条落实工序/SKILL.md` — 四步闭环工序唯一权威载体（查→落→验→报）
- `configs/protocol/clause10_four_step_closure.yaml` — 约束配置（硬约束声明、四步标志）
- `src/liquidloc/analysis/clause_gates.py` — p10_four_step_closure() 验证函数
- `.git/hooks/pre-commit` — pre-commit 四步工序验证调用点（p10调用在p9之后）
- `outputs/compliance_report_10.md` — 合规报告

**内容**: 每条指令按「查 → 落 → 验 → 报」四步闭环：查（读条款原文+查磁盘现状，禁止凭记忆）→ 落（翻译为文件路径+函数/类+行号落点）→ 验（逐行核对条款逐句对应，不落实当场改）→ 报（按八字段写落实记录，未写记录视为未落实）。

## [11] 条款归属总则（治理层）
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 三阶段归属总则（治理层/Stage1-6落实类/Stage7-8执行决策类）
- `configs/protocol/clause11_clause_attribution.yaml` — 约束配置（硬约束声明、归属标志）
- `src/liquidloc/analysis/clause_gates.py` — p11_clause_attribution() 验证函数
- `.git/hooks/pre-commit` — pre-commit 归属总则验证调用点（p11调用在p10之后）
- `outputs/compliance_report_11.md` — 合规报告

**内容**: 治理层条款→第0阶段逐条落实并写报告，证据写对应治理载体（按『治理载体分散布局』分散落点：AGENTS/rules/skill/hook/plugin/MCP）；Stage 1-6落实类→第1阶段代码落实，判据条款的判据已并入对应代码条款；Stage 7-8执行决策类→第2阶段执行，每条同样单独写执行落实报告。

## [12] 手册目标（治理层·最终检查只看这个）
**状态**: ✅ 已落地
**文件**:
- `rules/手册目标.md` — 核心判据唯一判据（主线A/B必须同时达成）
- `configs/protocol/clause12_manual_objectives.yaml` — 约束配置（硬约束声明、判据标志）
- `src/liquidloc/analysis/clause_gates.py` — p12_manual_objectives() 验证函数
- `.git/hooks/pre-commit` — pre-commit 手册目标验证调用点（p12调用在p11之后）
- `outputs/compliance_report_12.md` — 合规报告

**内容**: 防异步高NLOS实验RMSE整体过大（异步+NLOS下误差过大时所有方法一起高）；防LNN（Liquid液态骨干网络，proposed）的RMSE不是最低；两项判据统一为唯一判据，跑前冒烟/预达标与跑后判定共用；验收分级与降级出口：验收表每一项必须在完整验收路径下逐项回查修通，诊断轨/冒烟数据不得以任何形式并入、替代或补充验收轨数据；数据/代码/结果可追溯（manifest + git commit + 验收留痕）；核验通过后最终只看目标是否达成，不看流程勾选。

## [9] 落实报告格式（治理层·八字段）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause09_compliance_report_format.yaml` — 约束配置
- `rules/落实报告格式规范.md` — 八字段格式规范（唯一权威副本，含八字段逐字段说明/一句话模板/治理类同构/提交前自检/完整示例/判定铁律）
- `.agents/skills/execution-report-gate/SKILL.md` — §1 为八字段格式唯一逐字权威副本（入口）
- `src/liquidloc/analysis/clause_gates.py` — p9_compliance_report_format() 验证函数
- `.git/hooks/pre-commit` — pre-commit 八字段验证调用点（p9调用在p8之后）
- `outputs/compliance_report_9.md` — 合规报告

**内容**: 每条指令输出 = 落实报告格式一行，八字段逐字段规范（条款号/状态/条款逐字抄录/代码绝对定位/落实极尽具体+全量映射/审查问题→修复+循环轨迹/验证命令→预期/证据真实路径+stdout落盘），含一句话模板、治理类同构、提交前自检、完整示例、判定铁律；缺任一字段或字段不合格 = 该条未落实。细则进 `rules/落实报告格式规范.md`，不堆进 AGENTS.md。

## [36] git 是最后防线
**状态**: ✅ 已落地
**文件**: 
- `configs/protocol/git_is_last_resort.yaml` — 约束配置
- `.git/hooks/pre-commit` — pre-commit 防线（第 1-161 行）
- `.gitignore:30-31` — *.log 和 *.bak 防护规则
- `outputs/compliance_report_36.md` — 合规报告

**内容**: git 仓库与工作区必须一致，任何版本回退/文件损坏/误改恢复一律走 git（checkout/revert），禁止复制 _backup/_old 副本。

## [37] 档位定义声明
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/gear_definition_declaration.yaml` — 约束配置
- `configs/protocol/axis_a_sync.yaml` — A 轴档位协议
- `configs/protocol/axis_n_nlos.yaml` — N 轴档位协议
- `configs/protocol/axis_v_visual.yaml` — V 轴档位协议
- `configs/protocol/axis_k_anchor.yaml` — K 轴档位协议
- `configs/protocol/axis_m_missing.yaml` — M 轴档位协议
- `src/liquidloc/analysis/clause_gates.py:3242-3395` — 验证函数 `p37_gear_definition_declaration()`
- `outputs/compliance_report_37.md` — 合规报告

**内容**: A/N/V/K/M 各轴档位划分、参数与具体数值以 Stage 1 档位协议条款为唯一权威副本。

## [38] LOS 段测距噪声 σ≈0.6m
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/los_noise_baseline.yaml` — LOS 段测距噪声基准配置（noise_value: 0.6, clause_id: 38）
- `configs/models/ekf.yaml:30-31` — EKF UWB 量测噪声 σ=0.6m（S4 LOS 基准 σ0.6m）
- `configs/models/robust_ekf.yaml:34-35` — Robust-EKF UWB 量测噪声 σ=0.6m（S4 LOS σ0.6m）
- `outputs/compliance_report_38.md` — 合规报告

**内容**: LOS 段测距噪声 σ≈0.6m（器件事例级均值误差，作为 LOS 基准，非协议档位参数）。LOS 段高斯噪声（商用 UWB 模块室内实测测距误差标准差在亚米级、LOS 与 NLOS 统计分布差异显著）。
## [39] UWB(DW1000级, 10Hz) 输出数据规范
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/uwb_output_spec.yaml` — UWB 输出数据规范配置（chip_model: DW1000级, frequency_hz: 10Hz）
- `src/liquidloc/analysis/clause_gates.py:3761-3927` — 验证函数 `p39_uwb_output_spec()`
- `outputs/compliance_report_39.md` — 合规报告

**内容**: UWB(DW1000级, 10Hz)；输出数据：`uwb_ranges`(N×4 测距, m) + `uwb_valid`(N×4 有效掩码)。UWB 传感器输出数据规范，包含测距矩阵与有效掩码矩阵。

## [40] IMU（消费级，150Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/imu_output_spec.yaml` — IMU 输出数据规范配置（chip_type: 消费级，frequency_hz: 150Hz）
- `src/liquidloc/analysis/clause_gates.py:3930-4110` — 验证函数 `p40_imu_output_spec()`
- `outputs/compliance_report_40.md` — 合规报告

**内容**: IMU（消费级，150Hz）；输出数据：`imu`（N_imu×6：ax/ay/az/gx/gy/gz，150Hz 原生，不降采样）+ `imu_t`（N_imu×1 自身时间戳）。IMU 传感器输出数据规范。

## [41] VIO（单目+IMU 前端，20 Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/vio_output_spec.yaml` — VIO 输出数据规范配置（frequency_hz: 20Hz, sensor_type: 单目+IMU前端）
- `src/liquidloc/analysis/clause_gates.py:4111-4324` — 验证函数 `p41_vio_output_spec()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_41.md` — 合规报告

**内容**: VIO（单目+IMU 前端，20 Hz）；输出数据：`vio_pos`（N×2/3）+ `vio_yaw`（N×1）+ `vio_valid`（N×1）+ 协方差（V0 健康档有效帧率 100%）。VIO 传感器输出数据规范。

## [42] 四个输出头
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/four_output_heads.yaml` — 四个输出头机制链配置（uwb_scaling/bias/vio_scaling/risk）
- `configs/protocol/imu_no_head_reasoning.yaml` — IMU 不设头推理说明配置
- `src/liquidloc/analysis/clause_gates.py:4326-4518` — 验证函数 `p42_imu_no_head_reasoning()`
- `src/liquidloc/analysis/clause_gates.py:4521-4665` — 验证函数 `p42b_four_output_heads()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_42.md` — 合规报告

**内容**: 四个输出头（uwb_scaling / bias / vio_scaling / risk）对三源量测做自适应加权、偏差补偿与门控，是"NLOS 识别 → 协方差调制/降权 → 鲁棒门控"机制链的落地（与 S4.2 鲁棒融合组合拳一致）。IMU 作为高频运动先验不参与头调制（惯性量测不经数据驱动加权，走 EKF 运动模型传播），故不设 IMU 头；risk 头为通道级联合风险。

## [43] 禁止使用档位空间外的参数、禁止自行定义插值档/新档
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_extra_levels.yaml` — 禁止额外档位配置（forbidden_interpolation: true, forbidden_new_levels: true）
- `src/liquidloc/analysis/clause_gates.py:4859-4960` — 验证函数 `p43_no_extra_levels()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_43.md` — 合规报告

**内容**: 档位参数严格取自档位协议条款档位表（A/N/V/K/M 五轴），禁止使用档位空间外的参数值，禁止自行定义插值档（如 N1.5 类）或任何新档位。

## [44] 档位注入纪律
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause44_injection_discipline.yaml` — 档位注入纪律配置（静态口径/禁止漂移档位）
- `src/liquidloc/analysis/clause_gates.py:4668-4770` — 验证函数 `p44_gear_injection_discipline()`
- `outputs/compliance_report_44.md` — 合规报告

**内容**: A轴为静态时钟偏移/延迟统计口径（数值见 Stage 1 档位协议条款 A 轴表），严格按静态口径注入，不引入随时间线性增长的时钟漂移作为额外档位（如需评估 → 协议裁决项 + 敏感性分析）。

## [45] N1.5 边界协议
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/n15_boundary.yaml` — N1.5 边界协议配置
- `src/liquidloc/analysis/clause_gates.py:4759-4855` — 验证函数 `p45_n15_boundary()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_45.md` — 合规报告

**内容**: N1.5（ρ=20–25%/μ=1.5–2.5m/σ=0.5m）不在本手册档位协议条款档位表内；本手册引用协议 N 轴为 N0–N3；N1.5 属于同机另一实验体系，按其各自对应协议文件执行。

## [46] 协议引用边界
**状态**: ✅ 已落地（由 p49_protocol_reference_boundary() 验证）
**文件**:
- `configs/protocol/protocol_reference_boundary.yaml` — 协议引用边界配置
- `src/liquidloc/analysis/clause_gates.py:5098` 起 — 验证函数 `p49_protocol_reference_boundary()`
- `outputs/compliance_report_46.md` — 合规报告

**内容**: 档位以 Stage 1 档位协议条款为唯一权威副本；其他文件引用档位时必须标明出自该条款，不得复制参数数值形成第二权威副本。注意：验证函数沿用 p49 命名（历史实现），实际验证手册 [46] 协议引用边界条款。

## [47] 三层证据结构总表
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/three_layer_evidence_structure.yaml` — 三层证据结构总表配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p47_three_layer_evidence()`
- `outputs/compliance_report_47.md` — 合规报告

**内容**: 三层证据结构总表（论文级数据架构，与 S3 轨迹规格衔接）：主场景（受控）S1–S3固定布局（20m×20m、4锚、4遮挡区）10seed×5方法=50单元；布局随机化泛化层；统计推断与主场景共享层。

## [48] 规模档位
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/scale_gear_spec.yaml` — 规模档位配置（入门 N_layout=6、推荐 N_layout=10）
- `src/liquidloc/analysis/clause_gates.py:5189-5290` — 验证函数 `p48_two_sigma_separation()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_48.md` — 合规报告

**内容**: 规模档位：入门 N_layout=6、推荐 N_layout=10（full + 2 个最强基线 LSTM/EKF × 2–3 seed 即可，泛化层只判排序不做因子推断，不需要主场景同等功效）。

## [49] 布局参数与方法排序的散点/箱线
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_scatter_plot_spec.yaml` — 布局参数散点配置
- `src/liquidloc/analysis/clause_gates.py`（p54_layout_parameter_scatter 验证函数）
- `outputs/compliance_report_49.md` — 合规报告

**内容**: 布局参数变化与方法排序关系以散点/箱线图呈现，禁止只用汇总统计量。

## [50] 测试轨迹与训练无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause50_test_train_no_overlap.yaml` — 测试轨迹与训练无重叠配置
- `src/liquidloc/analysis/clause_gates.py`（p50_test_train_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_50.md` — 合规报告

**内容**: 测试轨迹与训练数据在序列级别严格分离，不存在任何重叠。所有评估指标均在独立于训练集的数据上计算。


## [51] 划分按整条序列、训练与测试无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause51_sequence_split_no_overlap.yaml` — 序列级划分配置
- `src/liquidloc/analysis/clause_gates.py`（p51_sequence_split_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_51.md` — 合规报告

**内容**: 划分按整条序列进行，训练集与测试集之间不存在任何序列级别的重叠。禁止在同一序列内随机采样部分帧/窗口作为训练、部分帧/窗口作为测试，防止数据泄漏。

## [52] 基线不是裸模型
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/baseline_not_bare_model.yaml` — 基线不是裸模型配置
- `src/liquidloc/analysis/clause_gates.py`（p52_baseline_not_bare_model 验证函数）
- `outputs/compliance_report_52.md` — 合规报告

**内容**: 确认 LSTM/Transformer 基线确实带头运行——否则 LNN 赢它们="4 个头 + 液态骨干"双重优势，对比不公平。

## [53] 配对样本对齐
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/paired_sample_alignment.yaml` — 配对样本对齐配置
- `src/liquidloc/analysis/clause_gates.py:5658-5729` — 验证函数 `p53_paired_sample_alignment()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_53.md` — 合规报告

**内容**: 各方法评估按"同一测试轨迹 × 同 seed"配对。

## [54] 两两比较附Holm-Bonferroni多重比较校正
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/holm_bonferroni_correction.yaml` — Holm-Bonferroni校正配置
- `src/liquidloc/analysis/clause_gates.py`（p60_holm_bonferroni_correction 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_54.md` — 合规报告

**内容**: 五方法两两比较共10对，使用Holm-Bonferroni校正控制族错误率。多方法对比不做校正会放大假阳性；Holm-Bonferroni为稳健默认；ML比较应配套统计检验+CI+校正。

## [55] 扩样路径
**状态**: 落实中
**文件**:
- `configs/protocol/scale_expansion_path.yaml` — 扩样路径配置
- `src/liquidloc/analysis/clause_gates.py`（p55_scale_expansion_path 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_55.md` — 合规报告

**内容**: 10 seed 不足支撑功效时如实声明并给出扩样路径（只增不减：加 seed/轨迹至功效达标，规模恒 ≥ 论文级 10 seed × ≥100/seed，论文级硬约束——不存在"收窄/降档"方向的出口）。

## [56] 分析单位与样本独立性声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/analysis_unit_sample_independence.yaml` — 分析单位配置
- `src/liquidloc/analysis/clause_gates.py`（p64_analysis_unit_sample_independence 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_56.md` — 合规报告

**内容**: 滑窗W=128步（步长默认1帧）使相邻窗口重叠127/128，帧级误差样本高度自相关、不满足i.i.d.，所有统计检验必须以轨迹级为分析单位。

## [57] 先导实验（pilot）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/pilot_experiment.yaml` — 先导实验配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p57_sequence_split_no_overlap()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_57.md` — 合规报告

**内容**: 先导实验单独用于方案冻结前估计效应量/方差/可行性，规模<=1 seed×四组合各1-2条测试轨迹。

## [58] A2/A3 的偏移与抖动数值按协议生效
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/a2a3_shift_jitter.yaml` — A2/A3 偏移与抖动协议配置（clause_id="58"）
- `src/liquidloc/analysis/clause_gates.py::p68_a2a3_shift_jitter()` — 验证函数
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_58.md` — 合规报告
- `checkpoints/compliance_report_clause58.md` — 验证快照

**内容**: A2/A3的偏移与抖动数值按协议生效，已按A2/A3全参数逐项核对。A2：UWB时钟偏移20-50ms、UWB往返延迟40-80ms、IMU时钟偏移5-15ms、IMU采样抖动±3ms、VIO时钟偏移20-50ms、VIO处理延迟30-80ms。

## [59] 重采样方法声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/resampling_method_declaration.yaml` — 重采样方法声明配置
- `src/liquidloc/analysis/clause_gates.py`（p69_resampling_method 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_59.md` — 合规报告

**内容**: UWB/VIO重采样采用最近邻（nearest-neighbor）对齐到UWB帧时刻，不使用线性/样条插值生成测距或VIO值；声明写入论文方法节。

## [60] 边界/极端情况
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/edge_case_extreme.yaml` — 边界/极端情况配置
- `src/liquidloc/analysis/clause_gates.py`（p60_edge_case_extreme 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_60.md` — 合规报告

**内容**: 边界极端情况：空轨迹、极短序列、M1长间隙落在首尾，不产生人工尖峰/除零。

## [61] 同 seed 初始化
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/same_seed_init.yaml` — 同seed初始化配置
- `src/liquidloc/analysis/clause_gates.py`（p71_same_seed_init 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_61.md` — 合规报告

**内容**: 同seed初始化：LNN/LSTM/Transformer用同一seed的权重初始化。

## [62] 论文级规模参数
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/power_sample_size.yaml` — 规模参数配置
- `src/liquidloc/analysis/clause_gates.py`（p62_power_sample_size 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_62.md` — 合规报告

**内容**: 论文级规模参数：训练12-15条/seed，测试>=100条/seed，四组合各>=25，120s基准，10 seed（论文级基线）。

## [63] 布局随机化泛化层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_randomization.yaml` — 布局随机化配置
- `src/liquidloc/analysis/clause_gates.py`（p72_layout_randomization 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_63.md` — 合规报告
- `checkpoints/compliance_report_clause63.md` — checkpoint 验证报告

**内容**: 布局随机化泛化层：N_layout>=6(入门)/10(推荐)，每布局5方法x2-3 seed。

## [64] 重新生成全部 seed 数据
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/regenerate_all_seeds.yaml` — 重新生成全部 seed 配置
- `src/liquidloc/analysis/clause_gates.py`（p73_regenerate_all_seeds 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_64.md` — 合规报告
- `checkpoints/compliance_report_clause64.md` — checkpoint 验证报告

**内容**: 主场景10 seed全部重新生成，禁止新旧数据混用。

## [65] 新旧数据混用禁令（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_old_new_mixing.yaml` — 新旧数据混用禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p74_no_old_new_mixing 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_65.md` — 合规报告
- `checkpoints/compliance_report_clause65.md` — checkpoint 验证报告

**内容**: 禁止部分seed新+部分seed旧混用，硬约束。

## [66] 禁止把冒烟数据并入全量测试集
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_smoke_merge.yaml` — 冒烟数据禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p75_no_smoke_merge 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_66.md` — 合规报告
- `checkpoints/compliance_report_clause66.md` — checkpoint 验证报告

**内容**: 禁止冒烟>=12条测试数据并入全量测试集，整批化执行，硬约束。

## [67] 全量单元
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/full_scale_unit.yaml` — 全量单元配置
- `src/liquidloc/analysis/clause_gates.py`（p76_full_scale_unit 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_67.md` — 合规报告
- `checkpoints/compliance_report_clause67.md` — checkpoint 验证报告

**内容**: 全量单元：主场景10 seed×5方法=50单元全部用新数据训练。

## [68] 执行偏差（deviation）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/execution_deviation.yaml` — 执行偏差配置（含 deviation 表、E 系列审计、目录命名规范）
- `src/liquidloc/analysis/clause_gates.py:7103-7203` — 验证函数 `p68_execution_deviation()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_68.md` — 合规报告
- `checkpoints/compliance_report_clause68.md` — 验证快照

**内容**: 执行层面与方案的偏差（漏跑/换机/中断恢复）→ 记 execution deviation 表（E 系列审计覆盖），不影响协议与分析定义。实验目录结构 run 目录命名规范：`run-<日期>-<seed>-<method>-<config_hash>`。

## [69] CPU 角色分层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/cpu_role_description.yaml` — CPU 角色分层配置（cpu_non_training_only/cpu_idle_training/no_conflict/priority_rule）
- `src/liquidloc/analysis/clause_gates.py:7418-7519` — 验证函数 `p69_cpu_role_hierarchy()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_69.md` — 合规报告
- `checkpoints/compliance_report_clause69.md` — 验证快照

**内容**: "CPU 仅限非训练负载"与"CPU 空闲协助训练"的关系：前者是 GPU 空闲时的默认分工（CPU 只做数据生成/评估/预处理），后者是 GPU 排满后的追加自由度（CPU 可承接剩余训练单元）——两者不冲突，GPU 有空的按前者，GPU 排满的按后者。

## [70] 评估并行
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause70_evaluation_parallel.yaml` — 评估并行配置（gpu_execution=true, trajectory_batch_size=16, no_batch_mixing=true, evaluation_protocol_unchanged=true）
- `src/liquidloc/analysis/clause_gates.py`（p70_evaluation_parallel 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_70.md` — 合规报告
- `checkpoints/compliance_report_clause70.md` — checkpoint 验证报告

**内容**: 评估并行：测试轨迹推理在 GPU 上按轨迹独立并行执行（轨迹内窗口批大小 16，轨迹间不混批，评估协议不变）。

## [71] 父进程链可回溯
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/parent_process_traceable.yaml` — 父进程链可回溯配置（traceable=true, command="ps -o ppid= -p <pid>"）
- `src/liquidloc/analysis/clause_gates.py:8337-8417` — 验证函数 `p77_parent_process_traceable()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_71.md` — 合规报告
- `checkpoints/compliance_report_clause71.md` — checkpoint 验证报告

**内容**: 父进程链可回溯到本体系的启动器（`ps -o ppid= -p <pid>` 逐级上溯，最终父进程是本体系的调度/启动脚本）。

## [72] Transformer 滑窗同轨迹内合批
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/transformer_sliding_window.yaml` — Transformer 滑窗合批配置（enabled=true, method="同轨迹内窗口合批", log_batch_composition=true）
- `src/liquidloc/analysis/clause_gates.py:8420-8505` — 验证函数 `p78_transformer_sliding_window()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_72.md` — 合规报告
- `checkpoints/compliance_report_clause72.md` — checkpoint 验证报告

**内容**: Transformer 滑窗同轨迹内合批 评估日志打印批构成（同轨迹内窗口）。

## [73] GPU 报错停
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/gpu_error_handling.yaml` — GPU 报错处理配置（verified=true, cpu_idle_assistance=true, cpu_only_for_full_gpu=true）
- `src/liquidloc/analysis/clause_gates.py:8508-8592` — 验证函数 `p79_no_gpu_error_stop()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_73.md` — 合规报告
- `checkpoints/compliance_report_clause73.md` — checkpoint 验证报告

**内容**: 在无 GPU 环境实测 → 确认报错停而非静默 CPU 训练。CPU 空闲协助（执行模式/执行模式·CPU 空闲协助）CPU 承接仅限 GPU 排满后启用的剩余训练单元。

## [74] RAM 上限 12G
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/ram_limit_hard_cap.yaml` — RAM 上限配置（max_gb=12, verification_method="冒烟实测单单元峰值内存（psutil 探针）"）
- `src/liquidloc/analysis/clause_gates.py:8595-8676` — 验证函数 `p80_ram_limit_12g()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_74.md` — 合规报告
- `checkpoints/compliance_report_clause74.md` — checkpoint 验证报告

**内容**: RAM ≤12G 硬上限（执行模式/执行模式·RAM ≤12G 硬上限）；验证方法（细化，自测留痕）：冒烟实测单单元峰值内存（psutil 探针）。

## [75] 禁止借大修变口径
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_refactor_changes.yaml` — 禁止变口径配置（no_change_paper_level_values=true, no_change_data_gen=true, no_change_eval=true, no_change_protocol=true）
- `src/liquidloc/analysis/clause_gates.py:8679-8768` — 验证函数 `p81_no_refactor_changes()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_75.md` — 合规报告
- `checkpoints/compliance_report_clause75.md` — checkpoint 验证报告

**内容**: 禁止借大修变口径——大修只改代码组织与可维护性（模块拆分/常量收敛/断言补齐/命名统一），不得改变任何论文级数值、不得改数据生成/评估/统计口径、不得为了好写而改协议。

## [76]

## [15]

## [16]

## [17]


## [18] 底线：档位引用纪律（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录档位引用纪律规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十八档位引用纪律配置文件（硬约束声明，含条款名加约束级别加治理阶段零等标志）
- 源码目录条款门禁验证模块（p一八执行者模板建出函数）
- 版本控制预提交钩子（档位引用纪律检查调用点）
- 智能体指南（治理载体索引一十八条目的）
- 输出目录八字段落实报告（纯中文表述）
**判据**: 本手册只引用档位名（A二/N一点五/V零/K一/M一）与"是否符合协议"的判定基准；档位参数数值不重复罗列、一律以 第一阶段 基准条款为准
## [5] 依赖声明（治理层·指令模式）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause05_dependency_declaration.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py:15259-15507` — p5_dependency_declaration() 验证函数
- `.git/hooks/pre-commit:111,229` — check_dependencies 前置依赖链硬拦截（定义@111、调用@229）
- `outputs/compliance_report_5.md` — 合规报告

**内容**: 每条指令头部标「前置:[N]」；执行前先查前置是否done，未done → 标blocked停下问，不硬跑、不跳序。pre-commit钩子check_dependencies在[2]-[6]依赖链上实施硬拦截（exit 1阻止提交）。



## [14] 底线：防统计作弊（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录防统计作弊规则文件 — 防统计作弊规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十四防统计作弊配置文件 — 防统计作弊配置（硬约束声明）
- 条款门禁源码 — 验证函数
- 版本控制预提交钩子 — 防统计作弊检查调用点
- 输出目录八字段落实报告 — 八字段落实报告
**判据**: 禁止选择性补跑/改数据到满意/旧数据复用/诊断轨并入验收轨

## [13] 底线：语言铁律（GLM 5.3 专属）
**状态**: ✅ 已落地
**文件**:
-  — 语言铁律规则文件（核心判据+禁止项+允许例外+违规处置）
-  — 语言铁律配置（clause_id=13, constraint_level=硬约束）
-  — 验证函数 
- [pre-commit] 前置检查 [2] 依赖 [1]: OK
[pre-commit] 前置检查 [3] 依赖 [1][2]: OK
[pre-commit] 前置检查 [4] 依赖 [1][2][3]: OK
[pre-commit] 前置检查 [5] 依赖 [1][2][3][4]: OK
[pre-commit] 前置检查 [6] 依赖 [1][2][3][4][5]: OK
[pre-commit] [8] 提交前自查: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | cfg_exists=True; cfg_parse_ok=True; clause_name_contains_8=True; constraint_level_is_hard=True; self_check_before_submit=True; check_all_reports_pass=True; check_status_table_done=True; check_manual_goals_met=True; check_eight_fields_complete=True; manual_exists=True; manual_contains_8=True; manual_contains_self_check=True; manual_contains_core_criteria=True; manual_contains_acceptance_table=True; hook_exists=True; hook_contains_p8_call=True; agents_md_exists=True; agents_md_has_clause_8_section=True; status_table_exists=True; status_table_has_clause_8=True; status_table_clause_8_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [9] 落实报告格式: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_all_eight_fields=True; rules_all_fields_have_three_levels=True; one_sentence_template=True; governance_homology=True; pre_submit_self_check=True; complete_examples_code=True; complete_examples_governance=True; judgment_iron_rule=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_9=True; eight_fields_spec=True; one_sentence_template_cfg=True; governance_homology_cfg=True; pre_submit_self_check_cfg=True; complete_examples_cfg=True; judgment_iron_rule_cfg=True; manual_exists=True; manual_contains_9=True; manual_contains_format=True; manual_contains_eight_fields=True; hook_exists=True; hook_contains_p9_call=True; agents_md_exists=True; agents_md_has_clause_9_section=True; status_table_exists=True; status_table_has_clause_9=True; status_table_clause_9_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [10] 逐条落实工序: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | skill_exists=True; step1_check=True; step2_implementation=True; step3_verification=True; step4_reporting=True; eight_fields_spec=True; self_check_list=True; failure_makes_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_10=True; agents_md_exists=True; agents_md_has_clause_10=True; status_table_exists=True; status_table_has_clause_10=True; status_table_clause_10_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p10_call=True
[pre-commit] [11] 条款归属总则: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_phase0=True; rules_has_phase1_6=True; rules_has_criterion_merged=True; rules_has_phase7_8=True; rules_has_final_check=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_11=True; agents_md_exists=True; agents_md_has_clause_11=True; status_table_exists=True; status_table_has_clause_11=True; status_table_clause_11_field_valid=True; manual_exists=True; manual_contains_11=True; manual_contains_attribution=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p11_call=True
[pre-commit] [12] 手册目标: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | manual_exists=True; manual_has_criterion=True; manual_has_line_a=True; manual_has_line_b=True; manual_has_aux=True; manual_has_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_12=True; agents_md_exists=True; agents_md_has_clause_12=True; status_table_exists=True; status_table_has_clause_12=True; status_table_clause_12_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p12_call=True — 语言铁律检查调用点（仅检查落实报告文件）
-  — 八字段落实报告
**判据**: 全程简体中文；禁止英文回答/禁止中英混杂；3处及以上违规=报告作废重做

## [6] 代码与文件冗余治理（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause06_redundancy_governance.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p6_redundancy_governance() 验证函数
- `outputs/compliance_report_6.md` — 合规报告

**内容**: 禁止为同一逻辑留多份拷贝、禁止复制旧文件再改的备份文化，历史版本走 git，工作目录只保留当前生效版本；发现 _backup/_old/_copy 类冗余文件 → 移入 archive 或删除；禁止"注释掉一大段代码留作备用"；清理动作只作用于本体系目录，禁止跨体系删改；清理任何进程前必须先判定归属。

## [7] 违反即处置（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause07_violation_disposition.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p7_violation_disposition() 验证函数
- `rules/判定铁律.md` — 违反即处置章节（五、违反即处置）
- `outputs/compliance_report_7.md` — 合规报告

**内容**: 执行中任何环节发现违反本文档（未清场/审计失败/口径不符/数据或权重污染）→ 隔离作废相关产物（archive 留痕）→ 修复根因 → 回到对应回归点重跑，处置证据随结果一并提交；任何"未达预期"的第一解释永远是执行偏差，不是方法不行、不是数据集不行；窗口外先修根因再谈结论。

## [8] 提交前自查（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause08_submit_self_check.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p8_submit_self_check() 验证函数
- `.git/hooks/pre-commit` — pre-commit 自查调用点（p8调用在check_dependencies之后）
- `outputs/compliance_report_8.md` — 合规报告

**内容**: 向用户提交论文级结果前先逐项自查是否符合手册目标（核心判据 + 验收表），全部符合才整理提交。

## [10] 逐条落实工序（治理层·四步闭环）
**状态**: ✅ 已落地
**文件**:
- `skills/逐条落实工序/SKILL.md` — 四步闭环工序唯一权威载体（查→落→验→报）
- `configs/protocol/clause10_four_step_closure.yaml` — 约束配置（硬约束声明、四步标志）
- `src/liquidloc/analysis/clause_gates.py` — p10_four_step_closure() 验证函数
- `.git/hooks/pre-commit` — pre-commit 四步工序验证调用点（p10调用在p9之后）
- `outputs/compliance_report_10.md` — 合规报告

**内容**: 每条指令按「查 → 落 → 验 → 报」四步闭环：查（读条款原文+查磁盘现状，禁止凭记忆）→ 落（翻译为文件路径+函数/类+行号落点）→ 验（逐行核对条款逐句对应，不落实当场改）→ 报（按八字段写落实记录，未写记录视为未落实）。

## [11] 条款归属总则（治理层）
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 三阶段归属总则（治理层/Stage1-6落实类/Stage7-8执行决策类）
- `configs/protocol/clause11_clause_attribution.yaml` — 约束配置（硬约束声明、归属标志）
- `src/liquidloc/analysis/clause_gates.py` — p11_clause_attribution() 验证函数
- `.git/hooks/pre-commit` — pre-commit 归属总则验证调用点（p11调用在p10之后）
- `outputs/compliance_report_11.md` — 合规报告

**内容**: 治理层条款→第0阶段逐条落实并写报告，证据写对应治理载体（按『治理载体分散布局』分散落点：AGENTS/rules/skill/hook/plugin/MCP）；Stage 1-6落实类→第1阶段代码落实，判据条款的判据已并入对应代码条款；Stage 7-8执行决策类→第2阶段执行，每条同样单独写执行落实报告。

## [12] 手册目标（治理层·最终检查只看这个）
**状态**: ✅ 已落地
**文件**:
- `rules/手册目标.md` — 核心判据唯一判据（主线A/B必须同时达成）
- `configs/protocol/clause12_manual_objectives.yaml` — 约束配置（硬约束声明、判据标志）
- `src/liquidloc/analysis/clause_gates.py` — p12_manual_objectives() 验证函数
- `.git/hooks/pre-commit` — pre-commit 手册目标验证调用点（p12调用在p11之后）
- `outputs/compliance_report_12.md` — 合规报告

**内容**: 防异步高NLOS实验RMSE整体过大（异步+NLOS下误差过大时所有方法一起高）；防LNN（Liquid液态骨干网络，proposed）的RMSE不是最低；两项判据统一为唯一判据，跑前冒烟/预达标与跑后判定共用；验收分级与降级出口：验收表每一项必须在完整验收路径下逐项回查修通，诊断轨/冒烟数据不得以任何形式并入、替代或补充验收轨数据；数据/代码/结果可追溯（manifest + git commit + 验收留痕）；核验通过后最终只看目标是否达成，不看流程勾选。

## [9] 落实报告格式（治理层·八字段）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause09_compliance_report_format.yaml` — 约束配置
- `rules/落实报告格式规范.md` — 八字段格式规范（唯一权威副本，含八字段逐字段说明/一句话模板/治理类同构/提交前自检/完整示例/判定铁律）
- `.agents/skills/execution-report-gate/SKILL.md` — §1 为八字段格式唯一逐字权威副本（入口）
- `src/liquidloc/analysis/clause_gates.py` — p9_compliance_report_format() 验证函数
- `.git/hooks/pre-commit` — pre-commit 八字段验证调用点（p9调用在p8之后）
- `outputs/compliance_report_9.md` — 合规报告

**内容**: 每条指令输出 = 落实报告格式一行，八字段逐字段规范（条款号/状态/条款逐字抄录/代码绝对定位/落实极尽具体+全量映射/审查问题→修复+循环轨迹/验证命令→预期/证据真实路径+stdout落盘），含一句话模板、治理类同构、提交前自检、完整示例、判定铁律；缺任一字段或字段不合格 = 该条未落实。细则进 `rules/落实报告格式规范.md`，不堆进 AGENTS.md。

## [36] git 是最后防线
**状态**: ✅ 已落地
**文件**: 
- `configs/protocol/git_is_last_resort.yaml` — 约束配置
- `.git/hooks/pre-commit` — pre-commit 防线（第 1-161 行）
- `.gitignore:30-31` — *.log 和 *.bak 防护规则
- `outputs/compliance_report_36.md` — 合规报告

**内容**: git 仓库与工作区必须一致，任何版本回退/文件损坏/误改恢复一律走 git（checkout/revert），禁止复制 _backup/_old 副本。

## [37] 档位定义声明
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/gear_definition_declaration.yaml` — 约束配置
- `configs/protocol/axis_a_sync.yaml` — A 轴档位协议
- `configs/protocol/axis_n_nlos.yaml` — N 轴档位协议
- `configs/protocol/axis_v_visual.yaml` — V 轴档位协议
- `configs/protocol/axis_k_anchor.yaml` — K 轴档位协议
- `configs/protocol/axis_m_missing.yaml` — M 轴档位协议
- `src/liquidloc/analysis/clause_gates.py:3242-3395` — 验证函数 `p37_gear_definition_declaration()`
- `outputs/compliance_report_37.md` — 合规报告

**内容**: A/N/V/K/M 各轴档位划分、参数与具体数值以 Stage 1 档位协议条款为唯一权威副本。

## [38] LOS 段测距噪声 σ≈0.6m
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/los_noise_baseline.yaml` — LOS 段测距噪声基准配置（noise_value: 0.6, clause_id: 38）
- `configs/models/ekf.yaml:30-31` — EKF UWB 量测噪声 σ=0.6m（S4 LOS 基准 σ0.6m）
- `configs/models/robust_ekf.yaml:34-35` — Robust-EKF UWB 量测噪声 σ=0.6m（S4 LOS σ0.6m）
- `outputs/compliance_report_38.md` — 合规报告

**内容**: LOS 段测距噪声 σ≈0.6m（器件事例级均值误差，作为 LOS 基准，非协议档位参数）。LOS 段高斯噪声（商用 UWB 模块室内实测测距误差标准差在亚米级、LOS 与 NLOS 统计分布差异显著）。
## [39] UWB(DW1000级, 10Hz) 输出数据规范
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/uwb_output_spec.yaml` — UWB 输出数据规范配置（chip_model: DW1000级, frequency_hz: 10Hz）
- `src/liquidloc/analysis/clause_gates.py:3761-3927` — 验证函数 `p39_uwb_output_spec()`
- `outputs/compliance_report_39.md` — 合规报告

**内容**: UWB(DW1000级, 10Hz)；输出数据：`uwb_ranges`(N×4 测距, m) + `uwb_valid`(N×4 有效掩码)。UWB 传感器输出数据规范，包含测距矩阵与有效掩码矩阵。

## [40] IMU（消费级，150Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/imu_output_spec.yaml` — IMU 输出数据规范配置（chip_type: 消费级，frequency_hz: 150Hz）
- `src/liquidloc/analysis/clause_gates.py:3930-4110` — 验证函数 `p40_imu_output_spec()`
- `outputs/compliance_report_40.md` — 合规报告

**内容**: IMU（消费级，150Hz）；输出数据：`imu`（N_imu×6：ax/ay/az/gx/gy/gz，150Hz 原生，不降采样）+ `imu_t`（N_imu×1 自身时间戳）。IMU 传感器输出数据规范。

## [41] VIO（单目+IMU 前端，20 Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/vio_output_spec.yaml` — VIO 输出数据规范配置（frequency_hz: 20Hz, sensor_type: 单目+IMU前端）
- `src/liquidloc/analysis/clause_gates.py:4111-4324` — 验证函数 `p41_vio_output_spec()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_41.md` — 合规报告

**内容**: VIO（单目+IMU 前端，20 Hz）；输出数据：`vio_pos`（N×2/3）+ `vio_yaw`（N×1）+ `vio_valid`（N×1）+ 协方差（V0 健康档有效帧率 100%）。VIO 传感器输出数据规范。

## [42] 四个输出头
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/four_output_heads.yaml` — 四个输出头机制链配置（uwb_scaling/bias/vio_scaling/risk）
- `configs/protocol/imu_no_head_reasoning.yaml` — IMU 不设头推理说明配置
- `src/liquidloc/analysis/clause_gates.py:4326-4518` — 验证函数 `p42_imu_no_head_reasoning()`
- `src/liquidloc/analysis/clause_gates.py:4521-4665` — 验证函数 `p42b_four_output_heads()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_42.md` — 合规报告

**内容**: 四个输出头（uwb_scaling / bias / vio_scaling / risk）对三源量测做自适应加权、偏差补偿与门控，是"NLOS 识别 → 协方差调制/降权 → 鲁棒门控"机制链的落地（与 S4.2 鲁棒融合组合拳一致）。IMU 作为高频运动先验不参与头调制（惯性量测不经数据驱动加权，走 EKF 运动模型传播），故不设 IMU 头；risk 头为通道级联合风险。

## [43] 禁止使用档位空间外的参数、禁止自行定义插值档/新档
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_extra_levels.yaml` — 禁止额外档位配置（forbidden_interpolation: true, forbidden_new_levels: true）
- `src/liquidloc/analysis/clause_gates.py:4859-4960` — 验证函数 `p43_no_extra_levels()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_43.md` — 合规报告

**内容**: 档位参数严格取自档位协议条款档位表（A/N/V/K/M 五轴），禁止使用档位空间外的参数值，禁止自行定义插值档（如 N1.5 类）或任何新档位。

## [44] 档位注入纪律
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause44_injection_discipline.yaml` — 档位注入纪律配置（静态口径/禁止漂移档位）
- `src/liquidloc/analysis/clause_gates.py:4668-4770` — 验证函数 `p44_gear_injection_discipline()`
- `outputs/compliance_report_44.md` — 合规报告

**内容**: A轴为静态时钟偏移/延迟统计口径（数值见 Stage 1 档位协议条款 A 轴表），严格按静态口径注入，不引入随时间线性增长的时钟漂移作为额外档位（如需评估 → 协议裁决项 + 敏感性分析）。

## [45] N1.5 边界协议
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/n15_boundary.yaml` — N1.5 边界协议配置
- `src/liquidloc/analysis/clause_gates.py:4759-4855` — 验证函数 `p45_n15_boundary()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_45.md` — 合规报告

**内容**: N1.5（ρ=20–25%/μ=1.5–2.5m/σ=0.5m）不在本手册档位协议条款档位表内；本手册引用协议 N 轴为 N0–N3；N1.5 属于同机另一实验体系，按其各自对应协议文件执行。

## [46] 协议引用边界
**状态**: ✅ 已落地（由 p49_protocol_reference_boundary() 验证）
**文件**:
- `configs/protocol/protocol_reference_boundary.yaml` — 协议引用边界配置
- `src/liquidloc/analysis/clause_gates.py:5098` 起 — 验证函数 `p49_protocol_reference_boundary()`
- `outputs/compliance_report_46.md` — 合规报告

**内容**: 档位以 Stage 1 档位协议条款为唯一权威副本；其他文件引用档位时必须标明出自该条款，不得复制参数数值形成第二权威副本。注意：验证函数沿用 p49 命名（历史实现），实际验证手册 [46] 协议引用边界条款。

## [47] 三层证据结构总表
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/three_layer_evidence_structure.yaml` — 三层证据结构总表配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p47_three_layer_evidence()`
- `outputs/compliance_report_47.md` — 合规报告

**内容**: 三层证据结构总表（论文级数据架构，与 S3 轨迹规格衔接）：主场景（受控）S1–S3固定布局（20m×20m、4锚、4遮挡区）10seed×5方法=50单元；布局随机化泛化层；统计推断与主场景共享层。

## [48] 规模档位
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/scale_gear_spec.yaml` — 规模档位配置（入门 N_layout=6、推荐 N_layout=10）
- `src/liquidloc/analysis/clause_gates.py:5189-5290` — 验证函数 `p48_two_sigma_separation()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_48.md` — 合规报告

**内容**: 规模档位：入门 N_layout=6、推荐 N_layout=10（full + 2 个最强基线 LSTM/EKF × 2–3 seed 即可，泛化层只判排序不做因子推断，不需要主场景同等功效）。

## [49] 布局参数与方法排序的散点/箱线
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_scatter_plot_spec.yaml` — 布局参数散点配置
- `src/liquidloc/analysis/clause_gates.py`（p54_layout_parameter_scatter 验证函数）
- `outputs/compliance_report_49.md` — 合规报告

**内容**: 布局参数变化与方法排序关系以散点/箱线图呈现，禁止只用汇总统计量。

## [50] 测试轨迹与训练无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause50_test_train_no_overlap.yaml` — 测试轨迹与训练无重叠配置
- `src/liquidloc/analysis/clause_gates.py`（p50_test_train_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_50.md` — 合规报告

**内容**: 测试轨迹与训练数据在序列级别严格分离，不存在任何重叠。所有评估指标均在独立于训练集的数据上计算。


## [51] 划分按整条序列、训练与测试无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause51_sequence_split_no_overlap.yaml` — 序列级划分配置
- `src/liquidloc/analysis/clause_gates.py`（p51_sequence_split_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_51.md` — 合规报告

**内容**: 划分按整条序列进行，训练集与测试集之间不存在任何序列级别的重叠。禁止在同一序列内随机采样部分帧/窗口作为训练、部分帧/窗口作为测试，防止数据泄漏。

## [52] 基线不是裸模型
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/baseline_not_bare_model.yaml` — 基线不是裸模型配置
- `src/liquidloc/analysis/clause_gates.py`（p52_baseline_not_bare_model 验证函数）
- `outputs/compliance_report_52.md` — 合规报告

**内容**: 确认 LSTM/Transformer 基线确实带头运行——否则 LNN 赢它们="4 个头 + 液态骨干"双重优势，对比不公平。

## [53] 配对样本对齐
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/paired_sample_alignment.yaml` — 配对样本对齐配置
- `src/liquidloc/analysis/clause_gates.py:5658-5729` — 验证函数 `p53_paired_sample_alignment()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_53.md` — 合规报告

**内容**: 各方法评估按"同一测试轨迹 × 同 seed"配对。

## [54] 两两比较附Holm-Bonferroni多重比较校正
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/holm_bonferroni_correction.yaml` — Holm-Bonferroni校正配置
- `src/liquidloc/analysis/clause_gates.py`（p60_holm_bonferroni_correction 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_54.md` — 合规报告

**内容**: 五方法两两比较共10对，使用Holm-Bonferroni校正控制族错误率。多方法对比不做校正会放大假阳性；Holm-Bonferroni为稳健默认；ML比较应配套统计检验+CI+校正。

## [55] 扩样路径
**状态**: 落实中
**文件**:
- `configs/protocol/scale_expansion_path.yaml` — 扩样路径配置
- `src/liquidloc/analysis/clause_gates.py`（p55_scale_expansion_path 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_55.md` — 合规报告

**内容**: 10 seed 不足支撑功效时如实声明并给出扩样路径（只增不减：加 seed/轨迹至功效达标，规模恒 ≥ 论文级 10 seed × ≥100/seed，论文级硬约束——不存在"收窄/降档"方向的出口）。

## [56] 分析单位与样本独立性声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/analysis_unit_sample_independence.yaml` — 分析单位配置
- `src/liquidloc/analysis/clause_gates.py`（p64_analysis_unit_sample_independence 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_56.md` — 合规报告

**内容**: 滑窗W=128步（步长默认1帧）使相邻窗口重叠127/128，帧级误差样本高度自相关、不满足i.i.d.，所有统计检验必须以轨迹级为分析单位。

## [57] 先导实验（pilot）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/pilot_experiment.yaml` — 先导实验配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p57_sequence_split_no_overlap()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_57.md` — 合规报告

**内容**: 先导实验单独用于方案冻结前估计效应量/方差/可行性，规模<=1 seed×四组合各1-2条测试轨迹。

## [58] A2/A3 的偏移与抖动数值按协议生效
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/a2a3_shift_jitter.yaml` — A2/A3 偏移与抖动协议配置（clause_id="58"）
- `src/liquidloc/analysis/clause_gates.py::p68_a2a3_shift_jitter()` — 验证函数
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_58.md` — 合规报告
- `checkpoints/compliance_report_clause58.md` — 验证快照

**内容**: A2/A3的偏移与抖动数值按协议生效，已按A2/A3全参数逐项核对。A2：UWB时钟偏移20-50ms、UWB往返延迟40-80ms、IMU时钟偏移5-15ms、IMU采样抖动±3ms、VIO时钟偏移20-50ms、VIO处理延迟30-80ms。

## [59] 重采样方法声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/resampling_method_declaration.yaml` — 重采样方法声明配置
- `src/liquidloc/analysis/clause_gates.py`（p69_resampling_method 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_59.md` — 合规报告

**内容**: UWB/VIO重采样采用最近邻（nearest-neighbor）对齐到UWB帧时刻，不使用线性/样条插值生成测距或VIO值；声明写入论文方法节。

## [60] 边界/极端情况
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/edge_case_extreme.yaml` — 边界/极端情况配置
- `src/liquidloc/analysis/clause_gates.py`（p60_edge_case_extreme 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_60.md` — 合规报告

**内容**: 边界极端情况：空轨迹、极短序列、M1长间隙落在首尾，不产生人工尖峰/除零。

## [61] 同 seed 初始化
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/same_seed_init.yaml` — 同seed初始化配置
- `src/liquidloc/analysis/clause_gates.py`（p71_same_seed_init 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_61.md` — 合规报告

**内容**: 同seed初始化：LNN/LSTM/Transformer用同一seed的权重初始化。

## [62] 论文级规模参数
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/power_sample_size.yaml` — 规模参数配置
- `src/liquidloc/analysis/clause_gates.py`（p62_power_sample_size 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_62.md` — 合规报告

**内容**: 论文级规模参数：训练12-15条/seed，测试>=100条/seed，四组合各>=25，120s基准，10 seed（论文级基线）。

## [63] 布局随机化泛化层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_randomization.yaml` — 布局随机化配置
- `src/liquidloc/analysis/clause_gates.py`（p72_layout_randomization 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_63.md` — 合规报告
- `checkpoints/compliance_report_clause63.md` — checkpoint 验证报告

**内容**: 布局随机化泛化层：N_layout>=6(入门)/10(推荐)，每布局5方法x2-3 seed。

## [64] 重新生成全部 seed 数据
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/regenerate_all_seeds.yaml` — 重新生成全部 seed 配置
- `src/liquidloc/analysis/clause_gates.py`（p73_regenerate_all_seeds 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_64.md` — 合规报告
- `checkpoints/compliance_report_clause64.md` — checkpoint 验证报告

**内容**: 主场景10 seed全部重新生成，禁止新旧数据混用。

## [65] 新旧数据混用禁令（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_old_new_mixing.yaml` — 新旧数据混用禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p74_no_old_new_mixing 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_65.md` — 合规报告
- `checkpoints/compliance_report_clause65.md` — checkpoint 验证报告

**内容**: 禁止部分seed新+部分seed旧混用，硬约束。

## [66] 禁止把冒烟数据并入全量测试集
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_smoke_merge.yaml` — 冒烟数据禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p75_no_smoke_merge 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_66.md` — 合规报告
- `checkpoints/compliance_report_clause66.md` — checkpoint 验证报告

**内容**: 禁止冒烟>=12条测试数据并入全量测试集，整批化执行，硬约束。

## [67] 全量单元
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/full_scale_unit.yaml` — 全量单元配置
- `src/liquidloc/analysis/clause_gates.py`（p76_full_scale_unit 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_67.md` — 合规报告
- `checkpoints/compliance_report_clause67.md` — checkpoint 验证报告

**内容**: 全量单元：主场景10 seed×5方法=50单元全部用新数据训练。

## [68] 执行偏差（deviation）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/execution_deviation.yaml` — 执行偏差配置（含 deviation 表、E 系列审计、目录命名规范）
- `src/liquidloc/analysis/clause_gates.py:7103-7203` — 验证函数 `p68_execution_deviation()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_68.md` — 合规报告
- `checkpoints/compliance_report_clause68.md` — 验证快照

**内容**: 执行层面与方案的偏差（漏跑/换机/中断恢复）→ 记 execution deviation 表（E 系列审计覆盖），不影响协议与分析定义。实验目录结构 run 目录命名规范：`run-<日期>-<seed>-<method>-<config_hash>`。

## [69] CPU 角色分层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/cpu_role_description.yaml` — CPU 角色分层配置（cpu_non_training_only/cpu_idle_training/no_conflict/priority_rule）
- `src/liquidloc/analysis/clause_gates.py:7418-7519` — 验证函数 `p69_cpu_role_hierarchy()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_69.md` — 合规报告
- `checkpoints/compliance_report_clause69.md` — 验证快照

**内容**: "CPU 仅限非训练负载"与"CPU 空闲协助训练"的关系：前者是 GPU 空闲时的默认分工（CPU 只做数据生成/评估/预处理），后者是 GPU 排满后的追加自由度（CPU 可承接剩余训练单元）——两者不冲突，GPU 有空的按前者，GPU 排满的按后者。

## [70] 评估并行
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause70_evaluation_parallel.yaml` — 评估并行配置（gpu_execution=true, trajectory_batch_size=16, no_batch_mixing=true, evaluation_protocol_unchanged=true）
- `src/liquidloc/analysis/clause_gates.py`（p70_evaluation_parallel 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_70.md` — 合规报告
- `checkpoints/compliance_report_clause70.md` — checkpoint 验证报告

**内容**: 评估并行：测试轨迹推理在 GPU 上按轨迹独立并行执行（轨迹内窗口批大小 16，轨迹间不混批，评估协议不变）。

## [71] 父进程链可回溯
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/parent_process_traceable.yaml` — 父进程链可回溯配置（traceable=true, command="ps -o ppid= -p <pid>"）
- `src/liquidloc/analysis/clause_gates.py:8337-8417` — 验证函数 `p77_parent_process_traceable()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_71.md` — 合规报告
- `checkpoints/compliance_report_clause71.md` — checkpoint 验证报告

**内容**: 父进程链可回溯到本体系的启动器（`ps -o ppid= -p <pid>` 逐级上溯，最终父进程是本体系的调度/启动脚本）。

## [72] Transformer 滑窗同轨迹内合批
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/transformer_sliding_window.yaml` — Transformer 滑窗合批配置（enabled=true, method="同轨迹内窗口合批", log_batch_composition=true）
- `src/liquidloc/analysis/clause_gates.py:8420-8505` — 验证函数 `p78_transformer_sliding_window()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_72.md` — 合规报告
- `checkpoints/compliance_report_clause72.md` — checkpoint 验证报告

**内容**: Transformer 滑窗同轨迹内合批 评估日志打印批构成（同轨迹内窗口）。

## [73] GPU 报错停
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/gpu_error_handling.yaml` — GPU 报错处理配置（verified=true, cpu_idle_assistance=true, cpu_only_for_full_gpu=true）
- `src/liquidloc/analysis/clause_gates.py:8508-8592` — 验证函数 `p79_no_gpu_error_stop()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_73.md` — 合规报告
- `checkpoints/compliance_report_clause73.md` — checkpoint 验证报告

**内容**: 在无 GPU 环境实测 → 确认报错停而非静默 CPU 训练。CPU 空闲协助（执行模式/执行模式·CPU 空闲协助）CPU 承接仅限 GPU 排满后启用的剩余训练单元。

## [74] RAM 上限 12G
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/ram_limit_hard_cap.yaml` — RAM 上限配置（max_gb=12, verification_method="冒烟实测单单元峰值内存（psutil 探针）"）
- `src/liquidloc/analysis/clause_gates.py:8595-8676` — 验证函数 `p80_ram_limit_12g()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_74.md` — 合规报告
- `checkpoints/compliance_report_clause74.md` — checkpoint 验证报告

**内容**: RAM ≤12G 硬上限（执行模式/执行模式·RAM ≤12G 硬上限）；验证方法（细化，自测留痕）：冒烟实测单单元峰值内存（psutil 探针）。

## [75] 禁止借大修变口径
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_refactor_changes.yaml` — 禁止变口径配置（no_change_paper_level_values=true, no_change_data_gen=true, no_change_eval=true, no_change_protocol=true）
- `src/liquidloc/analysis/clause_gates.py:8679-8768` — 验证函数 `p81_no_refactor_changes()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_75.md` — 合规报告
- `checkpoints/compliance_report_clause75.md` — checkpoint 验证报告

**内容**: 禁止借大修变口径——大修只改代码组织与可维护性（模块拆分/常量收敛/断言补齐/命名统一），不得改变任何论文级数值、不得改数据生成/评估/统计口径、不得为了好写而改协议。

## [76]

## [15]


## [16] 底线：第1阶段附加义务（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录第1阶段附加义务规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十六第1阶段附加义务配置文件（硬约束声明，含条款名加约束级别加治理阶段零等标志）
- 源码目录条款门禁验证模块（p一六资源冲突错峰排队调度函数）
- 版本控制预提交钩子（第1阶段附加义务检查调用点）
- 输出目录八字段落实报告（纯中文表述）
**判据**: 落实中发现缺陷→顺手修根因（禁止留待办事项/绕开/注释旧实现）；缺陷检查范围=阶段一至六条款映射代码；每条条款落实完成时做一次缺陷检查（静态审查/运行断言/单测至少一种）；发现即修根因；修复检查结果写入八字段报告"审查:"字段并留痕
## [5] 依赖声明（治理层·指令模式）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause05_dependency_declaration.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py:15259-15507` — p5_dependency_declaration() 验证函数
- `.git/hooks/pre-commit:111,229` — check_dependencies 前置依赖链硬拦截（定义@111、调用@229）
- `outputs/compliance_report_5.md` — 合规报告

**内容**: 每条指令头部标「前置:[N]」；执行前先查前置是否done，未done → 标blocked停下问，不硬跑、不跳序。pre-commit钩子check_dependencies在[2]-[6]依赖链上实施硬拦截（exit 1阻止提交）。



## [14] 底线：防统计作弊（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录防统计作弊规则文件 — 防统计作弊规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十四防统计作弊配置文件 — 防统计作弊配置（硬约束声明）
- 条款门禁源码 — 验证函数
- 版本控制预提交钩子 — 防统计作弊检查调用点
- 输出目录八字段落实报告 — 八字段落实报告
**判据**: 禁止选择性补跑/改数据到满意/旧数据复用/诊断轨并入验收轨

## [13] 底线：语言铁律（GLM 5.3 专属）
**状态**: ✅ 已落地
**文件**:
-  — 语言铁律规则文件（核心判据+禁止项+允许例外+违规处置）
-  — 语言铁律配置（clause_id=13, constraint_level=硬约束）
-  — 验证函数 
- [pre-commit] 前置检查 [2] 依赖 [1]: OK
[pre-commit] 前置检查 [3] 依赖 [1][2]: OK
[pre-commit] 前置检查 [4] 依赖 [1][2][3]: OK
[pre-commit] 前置检查 [5] 依赖 [1][2][3][4]: OK
[pre-commit] 前置检查 [6] 依赖 [1][2][3][4][5]: OK
[pre-commit] [8] 提交前自查: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | cfg_exists=True; cfg_parse_ok=True; clause_name_contains_8=True; constraint_level_is_hard=True; self_check_before_submit=True; check_all_reports_pass=True; check_status_table_done=True; check_manual_goals_met=True; check_eight_fields_complete=True; manual_exists=True; manual_contains_8=True; manual_contains_self_check=True; manual_contains_core_criteria=True; manual_contains_acceptance_table=True; hook_exists=True; hook_contains_p8_call=True; agents_md_exists=True; agents_md_has_clause_8_section=True; status_table_exists=True; status_table_has_clause_8=True; status_table_clause_8_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [9] 落实报告格式: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_all_eight_fields=True; rules_all_fields_have_three_levels=True; one_sentence_template=True; governance_homology=True; pre_submit_self_check=True; complete_examples_code=True; complete_examples_governance=True; judgment_iron_rule=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_9=True; eight_fields_spec=True; one_sentence_template_cfg=True; governance_homology_cfg=True; pre_submit_self_check_cfg=True; complete_examples_cfg=True; judgment_iron_rule_cfg=True; manual_exists=True; manual_contains_9=True; manual_contains_format=True; manual_contains_eight_fields=True; hook_exists=True; hook_contains_p9_call=True; agents_md_exists=True; agents_md_has_clause_9_section=True; status_table_exists=True; status_table_has_clause_9=True; status_table_clause_9_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [10] 逐条落实工序: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | skill_exists=True; step1_check=True; step2_implementation=True; step3_verification=True; step4_reporting=True; eight_fields_spec=True; self_check_list=True; failure_makes_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_10=True; agents_md_exists=True; agents_md_has_clause_10=True; status_table_exists=True; status_table_has_clause_10=True; status_table_clause_10_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p10_call=True
[pre-commit] [11] 条款归属总则: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_phase0=True; rules_has_phase1_6=True; rules_has_criterion_merged=True; rules_has_phase7_8=True; rules_has_final_check=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_11=True; agents_md_exists=True; agents_md_has_clause_11=True; status_table_exists=True; status_table_has_clause_11=True; status_table_clause_11_field_valid=True; manual_exists=True; manual_contains_11=True; manual_contains_attribution=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p11_call=True
[pre-commit] [12] 手册目标: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | manual_exists=True; manual_has_criterion=True; manual_has_line_a=True; manual_has_line_b=True; manual_has_aux=True; manual_has_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_12=True; agents_md_exists=True; agents_md_has_clause_12=True; status_table_exists=True; status_table_has_clause_12=True; status_table_clause_12_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p12_call=True — 语言铁律检查调用点（仅检查落实报告文件）
-  — 八字段落实报告
**判据**: 全程简体中文；禁止英文回答/禁止中英混杂；3处及以上违规=报告作废重做

## [6] 代码与文件冗余治理（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause06_redundancy_governance.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p6_redundancy_governance() 验证函数
- `outputs/compliance_report_6.md` — 合规报告

**内容**: 禁止为同一逻辑留多份拷贝、禁止复制旧文件再改的备份文化，历史版本走 git，工作目录只保留当前生效版本；发现 _backup/_old/_copy 类冗余文件 → 移入 archive 或删除；禁止"注释掉一大段代码留作备用"；清理动作只作用于本体系目录，禁止跨体系删改；清理任何进程前必须先判定归属。

## [7] 违反即处置（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause07_violation_disposition.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p7_violation_disposition() 验证函数
- `rules/判定铁律.md` — 违反即处置章节（五、违反即处置）
- `outputs/compliance_report_7.md` — 合规报告

**内容**: 执行中任何环节发现违反本文档（未清场/审计失败/口径不符/数据或权重污染）→ 隔离作废相关产物（archive 留痕）→ 修复根因 → 回到对应回归点重跑，处置证据随结果一并提交；任何"未达预期"的第一解释永远是执行偏差，不是方法不行、不是数据集不行；窗口外先修根因再谈结论。

## [8] 提交前自查（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause08_submit_self_check.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p8_submit_self_check() 验证函数
- `.git/hooks/pre-commit` — pre-commit 自查调用点（p8调用在check_dependencies之后）
- `outputs/compliance_report_8.md` — 合规报告

**内容**: 向用户提交论文级结果前先逐项自查是否符合手册目标（核心判据 + 验收表），全部符合才整理提交。

## [10] 逐条落实工序（治理层·四步闭环）
**状态**: ✅ 已落地
**文件**:
- `skills/逐条落实工序/SKILL.md` — 四步闭环工序唯一权威载体（查→落→验→报）
- `configs/protocol/clause10_four_step_closure.yaml` — 约束配置（硬约束声明、四步标志）
- `src/liquidloc/analysis/clause_gates.py` — p10_four_step_closure() 验证函数
- `.git/hooks/pre-commit` — pre-commit 四步工序验证调用点（p10调用在p9之后）
- `outputs/compliance_report_10.md` — 合规报告

**内容**: 每条指令按「查 → 落 → 验 → 报」四步闭环：查（读条款原文+查磁盘现状，禁止凭记忆）→ 落（翻译为文件路径+函数/类+行号落点）→ 验（逐行核对条款逐句对应，不落实当场改）→ 报（按八字段写落实记录，未写记录视为未落实）。

## [11] 条款归属总则（治理层）
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 三阶段归属总则（治理层/Stage1-6落实类/Stage7-8执行决策类）
- `configs/protocol/clause11_clause_attribution.yaml` — 约束配置（硬约束声明、归属标志）
- `src/liquidloc/analysis/clause_gates.py` — p11_clause_attribution() 验证函数
- `.git/hooks/pre-commit` — pre-commit 归属总则验证调用点（p11调用在p10之后）
- `outputs/compliance_report_11.md` — 合规报告

**内容**: 治理层条款→第0阶段逐条落实并写报告，证据写对应治理载体（按『治理载体分散布局』分散落点：AGENTS/rules/skill/hook/plugin/MCP）；Stage 1-6落实类→第1阶段代码落实，判据条款的判据已并入对应代码条款；Stage 7-8执行决策类→第2阶段执行，每条同样单独写执行落实报告。

## [12] 手册目标（治理层·最终检查只看这个）
**状态**: ✅ 已落地
**文件**:
- `rules/手册目标.md` — 核心判据唯一判据（主线A/B必须同时达成）
- `configs/protocol/clause12_manual_objectives.yaml` — 约束配置（硬约束声明、判据标志）
- `src/liquidloc/analysis/clause_gates.py` — p12_manual_objectives() 验证函数
- `.git/hooks/pre-commit` — pre-commit 手册目标验证调用点（p12调用在p11之后）
- `outputs/compliance_report_12.md` — 合规报告

**内容**: 防异步高NLOS实验RMSE整体过大（异步+NLOS下误差过大时所有方法一起高）；防LNN（Liquid液态骨干网络，proposed）的RMSE不是最低；两项判据统一为唯一判据，跑前冒烟/预达标与跑后判定共用；验收分级与降级出口：验收表每一项必须在完整验收路径下逐项回查修通，诊断轨/冒烟数据不得以任何形式并入、替代或补充验收轨数据；数据/代码/结果可追溯（manifest + git commit + 验收留痕）；核验通过后最终只看目标是否达成，不看流程勾选。

## [9] 落实报告格式（治理层·八字段）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause09_compliance_report_format.yaml` — 约束配置
- `rules/落实报告格式规范.md` — 八字段格式规范（唯一权威副本，含八字段逐字段说明/一句话模板/治理类同构/提交前自检/完整示例/判定铁律）
- `.agents/skills/execution-report-gate/SKILL.md` — §1 为八字段格式唯一逐字权威副本（入口）
- `src/liquidloc/analysis/clause_gates.py` — p9_compliance_report_format() 验证函数
- `.git/hooks/pre-commit` — pre-commit 八字段验证调用点（p9调用在p8之后）
- `outputs/compliance_report_9.md` — 合规报告

**内容**: 每条指令输出 = 落实报告格式一行，八字段逐字段规范（条款号/状态/条款逐字抄录/代码绝对定位/落实极尽具体+全量映射/审查问题→修复+循环轨迹/验证命令→预期/证据真实路径+stdout落盘），含一句话模板、治理类同构、提交前自检、完整示例、判定铁律；缺任一字段或字段不合格 = 该条未落实。细则进 `rules/落实报告格式规范.md`，不堆进 AGENTS.md。

## [36] git 是最后防线
**状态**: ✅ 已落地
**文件**: 
- `configs/protocol/git_is_last_resort.yaml` — 约束配置
- `.git/hooks/pre-commit` — pre-commit 防线（第 1-161 行）
- `.gitignore:30-31` — *.log 和 *.bak 防护规则
- `outputs/compliance_report_36.md` — 合规报告

**内容**: git 仓库与工作区必须一致，任何版本回退/文件损坏/误改恢复一律走 git（checkout/revert），禁止复制 _backup/_old 副本。

## [37] 档位定义声明
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/gear_definition_declaration.yaml` — 约束配置
- `configs/protocol/axis_a_sync.yaml` — A 轴档位协议
- `configs/protocol/axis_n_nlos.yaml` — N 轴档位协议
- `configs/protocol/axis_v_visual.yaml` — V 轴档位协议
- `configs/protocol/axis_k_anchor.yaml` — K 轴档位协议
- `configs/protocol/axis_m_missing.yaml` — M 轴档位协议
- `src/liquidloc/analysis/clause_gates.py:3242-3395` — 验证函数 `p37_gear_definition_declaration()`
- `outputs/compliance_report_37.md` — 合规报告

**内容**: A/N/V/K/M 各轴档位划分、参数与具体数值以 Stage 1 档位协议条款为唯一权威副本。

## [38] LOS 段测距噪声 σ≈0.6m
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/los_noise_baseline.yaml` — LOS 段测距噪声基准配置（noise_value: 0.6, clause_id: 38）
- `configs/models/ekf.yaml:30-31` — EKF UWB 量测噪声 σ=0.6m（S4 LOS 基准 σ0.6m）
- `configs/models/robust_ekf.yaml:34-35` — Robust-EKF UWB 量测噪声 σ=0.6m（S4 LOS σ0.6m）
- `outputs/compliance_report_38.md` — 合规报告

**内容**: LOS 段测距噪声 σ≈0.6m（器件事例级均值误差，作为 LOS 基准，非协议档位参数）。LOS 段高斯噪声（商用 UWB 模块室内实测测距误差标准差在亚米级、LOS 与 NLOS 统计分布差异显著）。
## [39] UWB(DW1000级, 10Hz) 输出数据规范
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/uwb_output_spec.yaml` — UWB 输出数据规范配置（chip_model: DW1000级, frequency_hz: 10Hz）
- `src/liquidloc/analysis/clause_gates.py:3761-3927` — 验证函数 `p39_uwb_output_spec()`
- `outputs/compliance_report_39.md` — 合规报告

**内容**: UWB(DW1000级, 10Hz)；输出数据：`uwb_ranges`(N×4 测距, m) + `uwb_valid`(N×4 有效掩码)。UWB 传感器输出数据规范，包含测距矩阵与有效掩码矩阵。

## [40] IMU（消费级，150Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/imu_output_spec.yaml` — IMU 输出数据规范配置（chip_type: 消费级，frequency_hz: 150Hz）
- `src/liquidloc/analysis/clause_gates.py:3930-4110` — 验证函数 `p40_imu_output_spec()`
- `outputs/compliance_report_40.md` — 合规报告

**内容**: IMU（消费级，150Hz）；输出数据：`imu`（N_imu×6：ax/ay/az/gx/gy/gz，150Hz 原生，不降采样）+ `imu_t`（N_imu×1 自身时间戳）。IMU 传感器输出数据规范。

## [41] VIO（单目+IMU 前端，20 Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/vio_output_spec.yaml` — VIO 输出数据规范配置（frequency_hz: 20Hz, sensor_type: 单目+IMU前端）
- `src/liquidloc/analysis/clause_gates.py:4111-4324` — 验证函数 `p41_vio_output_spec()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_41.md` — 合规报告

**内容**: VIO（单目+IMU 前端，20 Hz）；输出数据：`vio_pos`（N×2/3）+ `vio_yaw`（N×1）+ `vio_valid`（N×1）+ 协方差（V0 健康档有效帧率 100%）。VIO 传感器输出数据规范。

## [42] 四个输出头
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/four_output_heads.yaml` — 四个输出头机制链配置（uwb_scaling/bias/vio_scaling/risk）
- `configs/protocol/imu_no_head_reasoning.yaml` — IMU 不设头推理说明配置
- `src/liquidloc/analysis/clause_gates.py:4326-4518` — 验证函数 `p42_imu_no_head_reasoning()`
- `src/liquidloc/analysis/clause_gates.py:4521-4665` — 验证函数 `p42b_four_output_heads()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_42.md` — 合规报告

**内容**: 四个输出头（uwb_scaling / bias / vio_scaling / risk）对三源量测做自适应加权、偏差补偿与门控，是"NLOS 识别 → 协方差调制/降权 → 鲁棒门控"机制链的落地（与 S4.2 鲁棒融合组合拳一致）。IMU 作为高频运动先验不参与头调制（惯性量测不经数据驱动加权，走 EKF 运动模型传播），故不设 IMU 头；risk 头为通道级联合风险。

## [43] 禁止使用档位空间外的参数、禁止自行定义插值档/新档
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_extra_levels.yaml` — 禁止额外档位配置（forbidden_interpolation: true, forbidden_new_levels: true）
- `src/liquidloc/analysis/clause_gates.py:4859-4960` — 验证函数 `p43_no_extra_levels()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_43.md` — 合规报告

**内容**: 档位参数严格取自档位协议条款档位表（A/N/V/K/M 五轴），禁止使用档位空间外的参数值，禁止自行定义插值档（如 N1.5 类）或任何新档位。

## [44] 档位注入纪律
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause44_injection_discipline.yaml` — 档位注入纪律配置（静态口径/禁止漂移档位）
- `src/liquidloc/analysis/clause_gates.py:4668-4770` — 验证函数 `p44_gear_injection_discipline()`
- `outputs/compliance_report_44.md` — 合规报告

**内容**: A轴为静态时钟偏移/延迟统计口径（数值见 Stage 1 档位协议条款 A 轴表），严格按静态口径注入，不引入随时间线性增长的时钟漂移作为额外档位（如需评估 → 协议裁决项 + 敏感性分析）。

## [45] N1.5 边界协议
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/n15_boundary.yaml` — N1.5 边界协议配置
- `src/liquidloc/analysis/clause_gates.py:4759-4855` — 验证函数 `p45_n15_boundary()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_45.md` — 合规报告

**内容**: N1.5（ρ=20–25%/μ=1.5–2.5m/σ=0.5m）不在本手册档位协议条款档位表内；本手册引用协议 N 轴为 N0–N3；N1.5 属于同机另一实验体系，按其各自对应协议文件执行。

## [46] 协议引用边界
**状态**: ✅ 已落地（由 p49_protocol_reference_boundary() 验证）
**文件**:
- `configs/protocol/protocol_reference_boundary.yaml` — 协议引用边界配置
- `src/liquidloc/analysis/clause_gates.py:5098` 起 — 验证函数 `p49_protocol_reference_boundary()`
- `outputs/compliance_report_46.md` — 合规报告

**内容**: 档位以 Stage 1 档位协议条款为唯一权威副本；其他文件引用档位时必须标明出自该条款，不得复制参数数值形成第二权威副本。注意：验证函数沿用 p49 命名（历史实现），实际验证手册 [46] 协议引用边界条款。

## [47] 三层证据结构总表
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/three_layer_evidence_structure.yaml` — 三层证据结构总表配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p47_three_layer_evidence()`
- `outputs/compliance_report_47.md` — 合规报告

**内容**: 三层证据结构总表（论文级数据架构，与 S3 轨迹规格衔接）：主场景（受控）S1–S3固定布局（20m×20m、4锚、4遮挡区）10seed×5方法=50单元；布局随机化泛化层；统计推断与主场景共享层。

## [48] 规模档位
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/scale_gear_spec.yaml` — 规模档位配置（入门 N_layout=6、推荐 N_layout=10）
- `src/liquidloc/analysis/clause_gates.py:5189-5290` — 验证函数 `p48_two_sigma_separation()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_48.md` — 合规报告

**内容**: 规模档位：入门 N_layout=6、推荐 N_layout=10（full + 2 个最强基线 LSTM/EKF × 2–3 seed 即可，泛化层只判排序不做因子推断，不需要主场景同等功效）。

## [49] 布局参数与方法排序的散点/箱线
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_scatter_plot_spec.yaml` — 布局参数散点配置
- `src/liquidloc/analysis/clause_gates.py`（p54_layout_parameter_scatter 验证函数）
- `outputs/compliance_report_49.md` — 合规报告

**内容**: 布局参数变化与方法排序关系以散点/箱线图呈现，禁止只用汇总统计量。

## [50] 测试轨迹与训练无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause50_test_train_no_overlap.yaml` — 测试轨迹与训练无重叠配置
- `src/liquidloc/analysis/clause_gates.py`（p50_test_train_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_50.md` — 合规报告

**内容**: 测试轨迹与训练数据在序列级别严格分离，不存在任何重叠。所有评估指标均在独立于训练集的数据上计算。


## [51] 划分按整条序列、训练与测试无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause51_sequence_split_no_overlap.yaml` — 序列级划分配置
- `src/liquidloc/analysis/clause_gates.py`（p51_sequence_split_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_51.md` — 合规报告

**内容**: 划分按整条序列进行，训练集与测试集之间不存在任何序列级别的重叠。禁止在同一序列内随机采样部分帧/窗口作为训练、部分帧/窗口作为测试，防止数据泄漏。

## [52] 基线不是裸模型
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/baseline_not_bare_model.yaml` — 基线不是裸模型配置
- `src/liquidloc/analysis/clause_gates.py`（p52_baseline_not_bare_model 验证函数）
- `outputs/compliance_report_52.md` — 合规报告

**内容**: 确认 LSTM/Transformer 基线确实带头运行——否则 LNN 赢它们="4 个头 + 液态骨干"双重优势，对比不公平。

## [53] 配对样本对齐
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/paired_sample_alignment.yaml` — 配对样本对齐配置
- `src/liquidloc/analysis/clause_gates.py:5658-5729` — 验证函数 `p53_paired_sample_alignment()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_53.md` — 合规报告

**内容**: 各方法评估按"同一测试轨迹 × 同 seed"配对。

## [54] 两两比较附Holm-Bonferroni多重比较校正
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/holm_bonferroni_correction.yaml` — Holm-Bonferroni校正配置
- `src/liquidloc/analysis/clause_gates.py`（p60_holm_bonferroni_correction 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_54.md` — 合规报告

**内容**: 五方法两两比较共10对，使用Holm-Bonferroni校正控制族错误率。多方法对比不做校正会放大假阳性；Holm-Bonferroni为稳健默认；ML比较应配套统计检验+CI+校正。

## [55] 扩样路径
**状态**: 落实中
**文件**:
- `configs/protocol/scale_expansion_path.yaml` — 扩样路径配置
- `src/liquidloc/analysis/clause_gates.py`（p55_scale_expansion_path 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_55.md` — 合规报告

**内容**: 10 seed 不足支撑功效时如实声明并给出扩样路径（只增不减：加 seed/轨迹至功效达标，规模恒 ≥ 论文级 10 seed × ≥100/seed，论文级硬约束——不存在"收窄/降档"方向的出口）。

## [56] 分析单位与样本独立性声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/analysis_unit_sample_independence.yaml` — 分析单位配置
- `src/liquidloc/analysis/clause_gates.py`（p64_analysis_unit_sample_independence 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_56.md` — 合规报告

**内容**: 滑窗W=128步（步长默认1帧）使相邻窗口重叠127/128，帧级误差样本高度自相关、不满足i.i.d.，所有统计检验必须以轨迹级为分析单位。

## [57] 先导实验（pilot）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/pilot_experiment.yaml` — 先导实验配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p57_sequence_split_no_overlap()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_57.md` — 合规报告

**内容**: 先导实验单独用于方案冻结前估计效应量/方差/可行性，规模<=1 seed×四组合各1-2条测试轨迹。

## [58] A2/A3 的偏移与抖动数值按协议生效
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/a2a3_shift_jitter.yaml` — A2/A3 偏移与抖动协议配置（clause_id="58"）
- `src/liquidloc/analysis/clause_gates.py::p68_a2a3_shift_jitter()` — 验证函数
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_58.md` — 合规报告
- `checkpoints/compliance_report_clause58.md` — 验证快照

**内容**: A2/A3的偏移与抖动数值按协议生效，已按A2/A3全参数逐项核对。A2：UWB时钟偏移20-50ms、UWB往返延迟40-80ms、IMU时钟偏移5-15ms、IMU采样抖动±3ms、VIO时钟偏移20-50ms、VIO处理延迟30-80ms。

## [59] 重采样方法声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/resampling_method_declaration.yaml` — 重采样方法声明配置
- `src/liquidloc/analysis/clause_gates.py`（p69_resampling_method 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_59.md` — 合规报告

**内容**: UWB/VIO重采样采用最近邻（nearest-neighbor）对齐到UWB帧时刻，不使用线性/样条插值生成测距或VIO值；声明写入论文方法节。

## [60] 边界/极端情况
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/edge_case_extreme.yaml` — 边界/极端情况配置
- `src/liquidloc/analysis/clause_gates.py`（p60_edge_case_extreme 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_60.md` — 合规报告

**内容**: 边界极端情况：空轨迹、极短序列、M1长间隙落在首尾，不产生人工尖峰/除零。

## [61] 同 seed 初始化
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/same_seed_init.yaml` — 同seed初始化配置
- `src/liquidloc/analysis/clause_gates.py`（p71_same_seed_init 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_61.md` — 合规报告

**内容**: 同seed初始化：LNN/LSTM/Transformer用同一seed的权重初始化。

## [62] 论文级规模参数
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/power_sample_size.yaml` — 规模参数配置
- `src/liquidloc/analysis/clause_gates.py`（p62_power_sample_size 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_62.md` — 合规报告

**内容**: 论文级规模参数：训练12-15条/seed，测试>=100条/seed，四组合各>=25，120s基准，10 seed（论文级基线）。

## [63] 布局随机化泛化层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_randomization.yaml` — 布局随机化配置
- `src/liquidloc/analysis/clause_gates.py`（p72_layout_randomization 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_63.md` — 合规报告
- `checkpoints/compliance_report_clause63.md` — checkpoint 验证报告

**内容**: 布局随机化泛化层：N_layout>=6(入门)/10(推荐)，每布局5方法x2-3 seed。

## [64] 重新生成全部 seed 数据
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/regenerate_all_seeds.yaml` — 重新生成全部 seed 配置
- `src/liquidloc/analysis/clause_gates.py`（p73_regenerate_all_seeds 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_64.md` — 合规报告
- `checkpoints/compliance_report_clause64.md` — checkpoint 验证报告

**内容**: 主场景10 seed全部重新生成，禁止新旧数据混用。

## [65] 新旧数据混用禁令（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_old_new_mixing.yaml` — 新旧数据混用禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p74_no_old_new_mixing 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_65.md` — 合规报告
- `checkpoints/compliance_report_clause65.md` — checkpoint 验证报告

**内容**: 禁止部分seed新+部分seed旧混用，硬约束。

## [66] 禁止把冒烟数据并入全量测试集
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_smoke_merge.yaml` — 冒烟数据禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p75_no_smoke_merge 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_66.md` — 合规报告
- `checkpoints/compliance_report_clause66.md` — checkpoint 验证报告

**内容**: 禁止冒烟>=12条测试数据并入全量测试集，整批化执行，硬约束。

## [67] 全量单元
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/full_scale_unit.yaml` — 全量单元配置
- `src/liquidloc/analysis/clause_gates.py`（p76_full_scale_unit 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_67.md` — 合规报告
- `checkpoints/compliance_report_clause67.md` — checkpoint 验证报告

**内容**: 全量单元：主场景10 seed×5方法=50单元全部用新数据训练。

## [68] 执行偏差（deviation）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/execution_deviation.yaml` — 执行偏差配置（含 deviation 表、E 系列审计、目录命名规范）
- `src/liquidloc/analysis/clause_gates.py:7103-7203` — 验证函数 `p68_execution_deviation()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_68.md` — 合规报告
- `checkpoints/compliance_report_clause68.md` — 验证快照

**内容**: 执行层面与方案的偏差（漏跑/换机/中断恢复）→ 记 execution deviation 表（E 系列审计覆盖），不影响协议与分析定义。实验目录结构 run 目录命名规范：`run-<日期>-<seed>-<method>-<config_hash>`。

## [69] CPU 角色分层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/cpu_role_description.yaml` — CPU 角色分层配置（cpu_non_training_only/cpu_idle_training/no_conflict/priority_rule）
- `src/liquidloc/analysis/clause_gates.py:7418-7519` — 验证函数 `p69_cpu_role_hierarchy()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_69.md` — 合规报告
- `checkpoints/compliance_report_clause69.md` — 验证快照

**内容**: "CPU 仅限非训练负载"与"CPU 空闲协助训练"的关系：前者是 GPU 空闲时的默认分工（CPU 只做数据生成/评估/预处理），后者是 GPU 排满后的追加自由度（CPU 可承接剩余训练单元）——两者不冲突，GPU 有空的按前者，GPU 排满的按后者。

## [70] 评估并行
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause70_evaluation_parallel.yaml` — 评估并行配置（gpu_execution=true, trajectory_batch_size=16, no_batch_mixing=true, evaluation_protocol_unchanged=true）
- `src/liquidloc/analysis/clause_gates.py`（p70_evaluation_parallel 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_70.md` — 合规报告
- `checkpoints/compliance_report_clause70.md` — checkpoint 验证报告

**内容**: 评估并行：测试轨迹推理在 GPU 上按轨迹独立并行执行（轨迹内窗口批大小 16，轨迹间不混批，评估协议不变）。

## [71] 父进程链可回溯
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/parent_process_traceable.yaml` — 父进程链可回溯配置（traceable=true, command="ps -o ppid= -p <pid>"）
- `src/liquidloc/analysis/clause_gates.py:8337-8417` — 验证函数 `p77_parent_process_traceable()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_71.md` — 合规报告
- `checkpoints/compliance_report_clause71.md` — checkpoint 验证报告

**内容**: 父进程链可回溯到本体系的启动器（`ps -o ppid= -p <pid>` 逐级上溯，最终父进程是本体系的调度/启动脚本）。

## [72] Transformer 滑窗同轨迹内合批
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/transformer_sliding_window.yaml` — Transformer 滑窗合批配置（enabled=true, method="同轨迹内窗口合批", log_batch_composition=true）
- `src/liquidloc/analysis/clause_gates.py:8420-8505` — 验证函数 `p78_transformer_sliding_window()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_72.md` — 合规报告
- `checkpoints/compliance_report_clause72.md` — checkpoint 验证报告

**内容**: Transformer 滑窗同轨迹内合批 评估日志打印批构成（同轨迹内窗口）。

## [73] GPU 报错停
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/gpu_error_handling.yaml` — GPU 报错处理配置（verified=true, cpu_idle_assistance=true, cpu_only_for_full_gpu=true）
- `src/liquidloc/analysis/clause_gates.py:8508-8592` — 验证函数 `p79_no_gpu_error_stop()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_73.md` — 合规报告
- `checkpoints/compliance_report_clause73.md` — checkpoint 验证报告

**内容**: 在无 GPU 环境实测 → 确认报错停而非静默 CPU 训练。CPU 空闲协助（执行模式/执行模式·CPU 空闲协助）CPU 承接仅限 GPU 排满后启用的剩余训练单元。

## [74] RAM 上限 12G
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/ram_limit_hard_cap.yaml` — RAM 上限配置（max_gb=12, verification_method="冒烟实测单单元峰值内存（psutil 探针）"）
- `src/liquidloc/analysis/clause_gates.py:8595-8676` — 验证函数 `p80_ram_limit_12g()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_74.md` — 合规报告
- `checkpoints/compliance_report_clause74.md` — checkpoint 验证报告

**内容**: RAM ≤12G 硬上限（执行模式/执行模式·RAM ≤12G 硬上限）；验证方法（细化，自测留痕）：冒烟实测单单元峰值内存（psutil 探针）。

## [75] 禁止借大修变口径
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_refactor_changes.yaml` — 禁止变口径配置（no_change_paper_level_values=true, no_change_data_gen=true, no_change_eval=true, no_change_protocol=true）
- `src/liquidloc/analysis/clause_gates.py:8679-8768` — 验证函数 `p81_no_refactor_changes()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_75.md` — 合规报告
- `checkpoints/compliance_report_clause75.md` — checkpoint 验证报告

**内容**: 禁止借大修变口径——大修只改代码组织与可维护性（模块拆分/常量收敛/断言补齐/命名统一），不得改变任何论文级数值、不得改数据生成/评估/统计口径、不得为了好写而改协议。

## [76]

## [15]

## [16]


## [17] 底线：落实证据必须引用文件:行号（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录落实证据引用规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十七落实证据引用配置文件（硬约束声明，含条款名加约束级别加治理阶段零等标志）
- 源码目录条款门禁验证模块（p一七拆解复核函数）
- 版本控制预提交钩子（落实证据引用检查调用点）
- 输出目录八字段落实报告（纯中文表述）
**判据**: 第0阶段与第1阶段每条落实记录的证据字段必须写真实文件路径 + 行号/章节锚点；只写文件名不写行号或行号与文件实际不符 = 未落实
## [5] 依赖声明（治理层·指令模式）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause05_dependency_declaration.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py:15259-15507` — p5_dependency_declaration() 验证函数
- `.git/hooks/pre-commit:111,229` — check_dependencies 前置依赖链硬拦截（定义@111、调用@229）
- `outputs/compliance_report_5.md` — 合规报告

**内容**: 每条指令头部标「前置:[N]」；执行前先查前置是否done，未done → 标blocked停下问，不硬跑、不跳序。pre-commit钩子check_dependencies在[2]-[6]依赖链上实施硬拦截（exit 1阻止提交）。



## [14] 底线：防统计作弊（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录防统计作弊规则文件 — 防统计作弊规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十四防统计作弊配置文件 — 防统计作弊配置（硬约束声明）
- 条款门禁源码 — 验证函数
- 版本控制预提交钩子 — 防统计作弊检查调用点
- 输出目录八字段落实报告 — 八字段落实报告
**判据**: 禁止选择性补跑/改数据到满意/旧数据复用/诊断轨并入验收轨

## [13] 底线：语言铁律（GLM 5.3 专属）
**状态**: ✅ 已落地
**文件**:
-  — 语言铁律规则文件（核心判据+禁止项+允许例外+违规处置）
-  — 语言铁律配置（clause_id=13, constraint_level=硬约束）
-  — 验证函数 
- [pre-commit] 前置检查 [2] 依赖 [1]: OK
[pre-commit] 前置检查 [3] 依赖 [1][2]: OK
[pre-commit] 前置检查 [4] 依赖 [1][2][3]: OK
[pre-commit] 前置检查 [5] 依赖 [1][2][3][4]: OK
[pre-commit] 前置检查 [6] 依赖 [1][2][3][4][5]: OK
[pre-commit] [8] 提交前自查: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | cfg_exists=True; cfg_parse_ok=True; clause_name_contains_8=True; constraint_level_is_hard=True; self_check_before_submit=True; check_all_reports_pass=True; check_status_table_done=True; check_manual_goals_met=True; check_eight_fields_complete=True; manual_exists=True; manual_contains_8=True; manual_contains_self_check=True; manual_contains_core_criteria=True; manual_contains_acceptance_table=True; hook_exists=True; hook_contains_p8_call=True; agents_md_exists=True; agents_md_has_clause_8_section=True; status_table_exists=True; status_table_has_clause_8=True; status_table_clause_8_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [9] 落实报告格式: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_all_eight_fields=True; rules_all_fields_have_three_levels=True; one_sentence_template=True; governance_homology=True; pre_submit_self_check=True; complete_examples_code=True; complete_examples_governance=True; judgment_iron_rule=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_9=True; eight_fields_spec=True; one_sentence_template_cfg=True; governance_homology_cfg=True; pre_submit_self_check_cfg=True; complete_examples_cfg=True; judgment_iron_rule_cfg=True; manual_exists=True; manual_contains_9=True; manual_contains_format=True; manual_contains_eight_fields=True; hook_exists=True; hook_contains_p9_call=True; agents_md_exists=True; agents_md_has_clause_9_section=True; status_table_exists=True; status_table_has_clause_9=True; status_table_clause_9_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [10] 逐条落实工序: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | skill_exists=True; step1_check=True; step2_implementation=True; step3_verification=True; step4_reporting=True; eight_fields_spec=True; self_check_list=True; failure_makes_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_10=True; agents_md_exists=True; agents_md_has_clause_10=True; status_table_exists=True; status_table_has_clause_10=True; status_table_clause_10_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p10_call=True
[pre-commit] [11] 条款归属总则: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_phase0=True; rules_has_phase1_6=True; rules_has_criterion_merged=True; rules_has_phase7_8=True; rules_has_final_check=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_11=True; agents_md_exists=True; agents_md_has_clause_11=True; status_table_exists=True; status_table_has_clause_11=True; status_table_clause_11_field_valid=True; manual_exists=True; manual_contains_11=True; manual_contains_attribution=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p11_call=True
[pre-commit] [12] 手册目标: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | manual_exists=True; manual_has_criterion=True; manual_has_line_a=True; manual_has_line_b=True; manual_has_aux=True; manual_has_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_12=True; agents_md_exists=True; agents_md_has_clause_12=True; status_table_exists=True; status_table_has_clause_12=True; status_table_clause_12_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p12_call=True — 语言铁律检查调用点（仅检查落实报告文件）
-  — 八字段落实报告
**判据**: 全程简体中文；禁止英文回答/禁止中英混杂；3处及以上违规=报告作废重做

## [6] 代码与文件冗余治理（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause06_redundancy_governance.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p6_redundancy_governance() 验证函数
- `outputs/compliance_report_6.md` — 合规报告

**内容**: 禁止为同一逻辑留多份拷贝、禁止复制旧文件再改的备份文化，历史版本走 git，工作目录只保留当前生效版本；发现 _backup/_old/_copy 类冗余文件 → 移入 archive 或删除；禁止"注释掉一大段代码留作备用"；清理动作只作用于本体系目录，禁止跨体系删改；清理任何进程前必须先判定归属。

## [7] 违反即处置（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause07_violation_disposition.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p7_violation_disposition() 验证函数
- `rules/判定铁律.md` — 违反即处置章节（五、违反即处置）
- `outputs/compliance_report_7.md` — 合规报告

**内容**: 执行中任何环节发现违反本文档（未清场/审计失败/口径不符/数据或权重污染）→ 隔离作废相关产物（archive 留痕）→ 修复根因 → 回到对应回归点重跑，处置证据随结果一并提交；任何"未达预期"的第一解释永远是执行偏差，不是方法不行、不是数据集不行；窗口外先修根因再谈结论。

## [8] 提交前自查（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause08_submit_self_check.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p8_submit_self_check() 验证函数
- `.git/hooks/pre-commit` — pre-commit 自查调用点（p8调用在check_dependencies之后）
- `outputs/compliance_report_8.md` — 合规报告

**内容**: 向用户提交论文级结果前先逐项自查是否符合手册目标（核心判据 + 验收表），全部符合才整理提交。

## [10] 逐条落实工序（治理层·四步闭环）
**状态**: ✅ 已落地
**文件**:
- `skills/逐条落实工序/SKILL.md` — 四步闭环工序唯一权威载体（查→落→验→报）
- `configs/protocol/clause10_four_step_closure.yaml` — 约束配置（硬约束声明、四步标志）
- `src/liquidloc/analysis/clause_gates.py` — p10_four_step_closure() 验证函数
- `.git/hooks/pre-commit` — pre-commit 四步工序验证调用点（p10调用在p9之后）
- `outputs/compliance_report_10.md` — 合规报告

**内容**: 每条指令按「查 → 落 → 验 → 报」四步闭环：查（读条款原文+查磁盘现状，禁止凭记忆）→ 落（翻译为文件路径+函数/类+行号落点）→ 验（逐行核对条款逐句对应，不落实当场改）→ 报（按八字段写落实记录，未写记录视为未落实）。

## [11] 条款归属总则（治理层）
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 三阶段归属总则（治理层/Stage1-6落实类/Stage7-8执行决策类）
- `configs/protocol/clause11_clause_attribution.yaml` — 约束配置（硬约束声明、归属标志）
- `src/liquidloc/analysis/clause_gates.py` — p11_clause_attribution() 验证函数
- `.git/hooks/pre-commit` — pre-commit 归属总则验证调用点（p11调用在p10之后）
- `outputs/compliance_report_11.md` — 合规报告

**内容**: 治理层条款→第0阶段逐条落实并写报告，证据写对应治理载体（按『治理载体分散布局』分散落点：AGENTS/rules/skill/hook/plugin/MCP）；Stage 1-6落实类→第1阶段代码落实，判据条款的判据已并入对应代码条款；Stage 7-8执行决策类→第2阶段执行，每条同样单独写执行落实报告。

## [12] 手册目标（治理层·最终检查只看这个）
**状态**: ✅ 已落地
**文件**:
- `rules/手册目标.md` — 核心判据唯一判据（主线A/B必须同时达成）
- `configs/protocol/clause12_manual_objectives.yaml` — 约束配置（硬约束声明、判据标志）
- `src/liquidloc/analysis/clause_gates.py` — p12_manual_objectives() 验证函数
- `.git/hooks/pre-commit` — pre-commit 手册目标验证调用点（p12调用在p11之后）
- `outputs/compliance_report_12.md` — 合规报告

**内容**: 防异步高NLOS实验RMSE整体过大（异步+NLOS下误差过大时所有方法一起高）；防LNN（Liquid液态骨干网络，proposed）的RMSE不是最低；两项判据统一为唯一判据，跑前冒烟/预达标与跑后判定共用；验收分级与降级出口：验收表每一项必须在完整验收路径下逐项回查修通，诊断轨/冒烟数据不得以任何形式并入、替代或补充验收轨数据；数据/代码/结果可追溯（manifest + git commit + 验收留痕）；核验通过后最终只看目标是否达成，不看流程勾选。

## [9] 落实报告格式（治理层·八字段）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause09_compliance_report_format.yaml` — 约束配置
- `rules/落实报告格式规范.md` — 八字段格式规范（唯一权威副本，含八字段逐字段说明/一句话模板/治理类同构/提交前自检/完整示例/判定铁律）
- `.agents/skills/execution-report-gate/SKILL.md` — §1 为八字段格式唯一逐字权威副本（入口）
- `src/liquidloc/analysis/clause_gates.py` — p9_compliance_report_format() 验证函数
- `.git/hooks/pre-commit` — pre-commit 八字段验证调用点（p9调用在p8之后）
- `outputs/compliance_report_9.md` — 合规报告

**内容**: 每条指令输出 = 落实报告格式一行，八字段逐字段规范（条款号/状态/条款逐字抄录/代码绝对定位/落实极尽具体+全量映射/审查问题→修复+循环轨迹/验证命令→预期/证据真实路径+stdout落盘），含一句话模板、治理类同构、提交前自检、完整示例、判定铁律；缺任一字段或字段不合格 = 该条未落实。细则进 `rules/落实报告格式规范.md`，不堆进 AGENTS.md。

## [36] git 是最后防线
**状态**: ✅ 已落地
**文件**: 
- `configs/protocol/git_is_last_resort.yaml` — 约束配置
- `.git/hooks/pre-commit` — pre-commit 防线（第 1-161 行）
- `.gitignore:30-31` — *.log 和 *.bak 防护规则
- `outputs/compliance_report_36.md` — 合规报告

**内容**: git 仓库与工作区必须一致，任何版本回退/文件损坏/误改恢复一律走 git（checkout/revert），禁止复制 _backup/_old 副本。

## [37] 档位定义声明
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/gear_definition_declaration.yaml` — 约束配置
- `configs/protocol/axis_a_sync.yaml` — A 轴档位协议
- `configs/protocol/axis_n_nlos.yaml` — N 轴档位协议
- `configs/protocol/axis_v_visual.yaml` — V 轴档位协议
- `configs/protocol/axis_k_anchor.yaml` — K 轴档位协议
- `configs/protocol/axis_m_missing.yaml` — M 轴档位协议
- `src/liquidloc/analysis/clause_gates.py:3242-3395` — 验证函数 `p37_gear_definition_declaration()`
- `outputs/compliance_report_37.md` — 合规报告

**内容**: A/N/V/K/M 各轴档位划分、参数与具体数值以 Stage 1 档位协议条款为唯一权威副本。

## [38] LOS 段测距噪声 σ≈0.6m
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/los_noise_baseline.yaml` — LOS 段测距噪声基准配置（noise_value: 0.6, clause_id: 38）
- `configs/models/ekf.yaml:30-31` — EKF UWB 量测噪声 σ=0.6m（S4 LOS 基准 σ0.6m）
- `configs/models/robust_ekf.yaml:34-35` — Robust-EKF UWB 量测噪声 σ=0.6m（S4 LOS σ0.6m）
- `outputs/compliance_report_38.md` — 合规报告

**内容**: LOS 段测距噪声 σ≈0.6m（器件事例级均值误差，作为 LOS 基准，非协议档位参数）。LOS 段高斯噪声（商用 UWB 模块室内实测测距误差标准差在亚米级、LOS 与 NLOS 统计分布差异显著）。
## [39] UWB(DW1000级, 10Hz) 输出数据规范
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/uwb_output_spec.yaml` — UWB 输出数据规范配置（chip_model: DW1000级, frequency_hz: 10Hz）
- `src/liquidloc/analysis/clause_gates.py:3761-3927` — 验证函数 `p39_uwb_output_spec()`
- `outputs/compliance_report_39.md` — 合规报告

**内容**: UWB(DW1000级, 10Hz)；输出数据：`uwb_ranges`(N×4 测距, m) + `uwb_valid`(N×4 有效掩码)。UWB 传感器输出数据规范，包含测距矩阵与有效掩码矩阵。

## [40] IMU（消费级，150Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/imu_output_spec.yaml` — IMU 输出数据规范配置（chip_type: 消费级，frequency_hz: 150Hz）
- `src/liquidloc/analysis/clause_gates.py:3930-4110` — 验证函数 `p40_imu_output_spec()`
- `outputs/compliance_report_40.md` — 合规报告

**内容**: IMU（消费级，150Hz）；输出数据：`imu`（N_imu×6：ax/ay/az/gx/gy/gz，150Hz 原生，不降采样）+ `imu_t`（N_imu×1 自身时间戳）。IMU 传感器输出数据规范。

## [41] VIO（单目+IMU 前端，20 Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/vio_output_spec.yaml` — VIO 输出数据规范配置（frequency_hz: 20Hz, sensor_type: 单目+IMU前端）
- `src/liquidloc/analysis/clause_gates.py:4111-4324` — 验证函数 `p41_vio_output_spec()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_41.md` — 合规报告

**内容**: VIO（单目+IMU 前端，20 Hz）；输出数据：`vio_pos`（N×2/3）+ `vio_yaw`（N×1）+ `vio_valid`（N×1）+ 协方差（V0 健康档有效帧率 100%）。VIO 传感器输出数据规范。

## [42] 四个输出头
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/four_output_heads.yaml` — 四个输出头机制链配置（uwb_scaling/bias/vio_scaling/risk）
- `configs/protocol/imu_no_head_reasoning.yaml` — IMU 不设头推理说明配置
- `src/liquidloc/analysis/clause_gates.py:4326-4518` — 验证函数 `p42_imu_no_head_reasoning()`
- `src/liquidloc/analysis/clause_gates.py:4521-4665` — 验证函数 `p42b_four_output_heads()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_42.md` — 合规报告

**内容**: 四个输出头（uwb_scaling / bias / vio_scaling / risk）对三源量测做自适应加权、偏差补偿与门控，是"NLOS 识别 → 协方差调制/降权 → 鲁棒门控"机制链的落地（与 S4.2 鲁棒融合组合拳一致）。IMU 作为高频运动先验不参与头调制（惯性量测不经数据驱动加权，走 EKF 运动模型传播），故不设 IMU 头；risk 头为通道级联合风险。

## [43] 禁止使用档位空间外的参数、禁止自行定义插值档/新档
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_extra_levels.yaml` — 禁止额外档位配置（forbidden_interpolation: true, forbidden_new_levels: true）
- `src/liquidloc/analysis/clause_gates.py:4859-4960` — 验证函数 `p43_no_extra_levels()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_43.md` — 合规报告

**内容**: 档位参数严格取自档位协议条款档位表（A/N/V/K/M 五轴），禁止使用档位空间外的参数值，禁止自行定义插值档（如 N1.5 类）或任何新档位。

## [44] 档位注入纪律
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause44_injection_discipline.yaml` — 档位注入纪律配置（静态口径/禁止漂移档位）
- `src/liquidloc/analysis/clause_gates.py:4668-4770` — 验证函数 `p44_gear_injection_discipline()`
- `outputs/compliance_report_44.md` — 合规报告

**内容**: A轴为静态时钟偏移/延迟统计口径（数值见 Stage 1 档位协议条款 A 轴表），严格按静态口径注入，不引入随时间线性增长的时钟漂移作为额外档位（如需评估 → 协议裁决项 + 敏感性分析）。

## [45] N1.5 边界协议
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/n15_boundary.yaml` — N1.5 边界协议配置
- `src/liquidloc/analysis/clause_gates.py:4759-4855` — 验证函数 `p45_n15_boundary()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_45.md` — 合规报告

**内容**: N1.5（ρ=20–25%/μ=1.5–2.5m/σ=0.5m）不在本手册档位协议条款档位表内；本手册引用协议 N 轴为 N0–N3；N1.5 属于同机另一实验体系，按其各自对应协议文件执行。

## [46] 协议引用边界
**状态**: ✅ 已落地（由 p49_protocol_reference_boundary() 验证）
**文件**:
- `configs/protocol/protocol_reference_boundary.yaml` — 协议引用边界配置
- `src/liquidloc/analysis/clause_gates.py:5098` 起 — 验证函数 `p49_protocol_reference_boundary()`
- `outputs/compliance_report_46.md` — 合规报告

**内容**: 档位以 Stage 1 档位协议条款为唯一权威副本；其他文件引用档位时必须标明出自该条款，不得复制参数数值形成第二权威副本。注意：验证函数沿用 p49 命名（历史实现），实际验证手册 [46] 协议引用边界条款。

## [47] 三层证据结构总表
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/three_layer_evidence_structure.yaml` — 三层证据结构总表配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p47_three_layer_evidence()`
- `outputs/compliance_report_47.md` — 合规报告

**内容**: 三层证据结构总表（论文级数据架构，与 S3 轨迹规格衔接）：主场景（受控）S1–S3固定布局（20m×20m、4锚、4遮挡区）10seed×5方法=50单元；布局随机化泛化层；统计推断与主场景共享层。

## [48] 规模档位
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/scale_gear_spec.yaml` — 规模档位配置（入门 N_layout=6、推荐 N_layout=10）
- `src/liquidloc/analysis/clause_gates.py:5189-5290` — 验证函数 `p48_two_sigma_separation()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_48.md` — 合规报告

**内容**: 规模档位：入门 N_layout=6、推荐 N_layout=10（full + 2 个最强基线 LSTM/EKF × 2–3 seed 即可，泛化层只判排序不做因子推断，不需要主场景同等功效）。

## [49] 布局参数与方法排序的散点/箱线
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_scatter_plot_spec.yaml` — 布局参数散点配置
- `src/liquidloc/analysis/clause_gates.py`（p54_layout_parameter_scatter 验证函数）
- `outputs/compliance_report_49.md` — 合规报告

**内容**: 布局参数变化与方法排序关系以散点/箱线图呈现，禁止只用汇总统计量。

## [50] 测试轨迹与训练无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause50_test_train_no_overlap.yaml` — 测试轨迹与训练无重叠配置
- `src/liquidloc/analysis/clause_gates.py`（p50_test_train_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_50.md` — 合规报告

**内容**: 测试轨迹与训练数据在序列级别严格分离，不存在任何重叠。所有评估指标均在独立于训练集的数据上计算。


## [51] 划分按整条序列、训练与测试无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause51_sequence_split_no_overlap.yaml` — 序列级划分配置
- `src/liquidloc/analysis/clause_gates.py`（p51_sequence_split_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_51.md` — 合规报告

**内容**: 划分按整条序列进行，训练集与测试集之间不存在任何序列级别的重叠。禁止在同一序列内随机采样部分帧/窗口作为训练、部分帧/窗口作为测试，防止数据泄漏。

## [52] 基线不是裸模型
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/baseline_not_bare_model.yaml` — 基线不是裸模型配置
- `src/liquidloc/analysis/clause_gates.py`（p52_baseline_not_bare_model 验证函数）
- `outputs/compliance_report_52.md` — 合规报告

**内容**: 确认 LSTM/Transformer 基线确实带头运行——否则 LNN 赢它们="4 个头 + 液态骨干"双重优势，对比不公平。

## [53] 配对样本对齐
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/paired_sample_alignment.yaml` — 配对样本对齐配置
- `src/liquidloc/analysis/clause_gates.py:5658-5729` — 验证函数 `p53_paired_sample_alignment()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_53.md` — 合规报告

**内容**: 各方法评估按"同一测试轨迹 × 同 seed"配对。

## [54] 两两比较附Holm-Bonferroni多重比较校正
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/holm_bonferroni_correction.yaml` — Holm-Bonferroni校正配置
- `src/liquidloc/analysis/clause_gates.py`（p60_holm_bonferroni_correction 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_54.md` — 合规报告

**内容**: 五方法两两比较共10对，使用Holm-Bonferroni校正控制族错误率。多方法对比不做校正会放大假阳性；Holm-Bonferroni为稳健默认；ML比较应配套统计检验+CI+校正。

## [55] 扩样路径
**状态**: 落实中
**文件**:
- `configs/protocol/scale_expansion_path.yaml` — 扩样路径配置
- `src/liquidloc/analysis/clause_gates.py`（p55_scale_expansion_path 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_55.md` — 合规报告

**内容**: 10 seed 不足支撑功效时如实声明并给出扩样路径（只增不减：加 seed/轨迹至功效达标，规模恒 ≥ 论文级 10 seed × ≥100/seed，论文级硬约束——不存在"收窄/降档"方向的出口）。

## [56] 分析单位与样本独立性声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/analysis_unit_sample_independence.yaml` — 分析单位配置
- `src/liquidloc/analysis/clause_gates.py`（p64_analysis_unit_sample_independence 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_56.md` — 合规报告

**内容**: 滑窗W=128步（步长默认1帧）使相邻窗口重叠127/128，帧级误差样本高度自相关、不满足i.i.d.，所有统计检验必须以轨迹级为分析单位。

## [57] 先导实验（pilot）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/pilot_experiment.yaml` — 先导实验配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p57_sequence_split_no_overlap()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_57.md` — 合规报告

**内容**: 先导实验单独用于方案冻结前估计效应量/方差/可行性，规模<=1 seed×四组合各1-2条测试轨迹。

## [58] A2/A3 的偏移与抖动数值按协议生效
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/a2a3_shift_jitter.yaml` — A2/A3 偏移与抖动协议配置（clause_id="58"）
- `src/liquidloc/analysis/clause_gates.py::p68_a2a3_shift_jitter()` — 验证函数
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_58.md` — 合规报告
- `checkpoints/compliance_report_clause58.md` — 验证快照

**内容**: A2/A3的偏移与抖动数值按协议生效，已按A2/A3全参数逐项核对。A2：UWB时钟偏移20-50ms、UWB往返延迟40-80ms、IMU时钟偏移5-15ms、IMU采样抖动±3ms、VIO时钟偏移20-50ms、VIO处理延迟30-80ms。

## [59] 重采样方法声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/resampling_method_declaration.yaml` — 重采样方法声明配置
- `src/liquidloc/analysis/clause_gates.py`（p69_resampling_method 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_59.md` — 合规报告

**内容**: UWB/VIO重采样采用最近邻（nearest-neighbor）对齐到UWB帧时刻，不使用线性/样条插值生成测距或VIO值；声明写入论文方法节。

## [60] 边界/极端情况
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/edge_case_extreme.yaml` — 边界/极端情况配置
- `src/liquidloc/analysis/clause_gates.py`（p60_edge_case_extreme 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_60.md` — 合规报告

**内容**: 边界极端情况：空轨迹、极短序列、M1长间隙落在首尾，不产生人工尖峰/除零。

## [61] 同 seed 初始化
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/same_seed_init.yaml` — 同seed初始化配置
- `src/liquidloc/analysis/clause_gates.py`（p71_same_seed_init 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_61.md` — 合规报告

**内容**: 同seed初始化：LNN/LSTM/Transformer用同一seed的权重初始化。

## [62] 论文级规模参数
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/power_sample_size.yaml` — 规模参数配置
- `src/liquidloc/analysis/clause_gates.py`（p62_power_sample_size 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_62.md` — 合规报告

**内容**: 论文级规模参数：训练12-15条/seed，测试>=100条/seed，四组合各>=25，120s基准，10 seed（论文级基线）。

## [63] 布局随机化泛化层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_randomization.yaml` — 布局随机化配置
- `src/liquidloc/analysis/clause_gates.py`（p72_layout_randomization 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_63.md` — 合规报告
- `checkpoints/compliance_report_clause63.md` — checkpoint 验证报告

**内容**: 布局随机化泛化层：N_layout>=6(入门)/10(推荐)，每布局5方法x2-3 seed。

## [64] 重新生成全部 seed 数据
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/regenerate_all_seeds.yaml` — 重新生成全部 seed 配置
- `src/liquidloc/analysis/clause_gates.py`（p73_regenerate_all_seeds 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_64.md` — 合规报告
- `checkpoints/compliance_report_clause64.md` — checkpoint 验证报告

**内容**: 主场景10 seed全部重新生成，禁止新旧数据混用。

## [65] 新旧数据混用禁令（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_old_new_mixing.yaml` — 新旧数据混用禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p74_no_old_new_mixing 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_65.md` — 合规报告
- `checkpoints/compliance_report_clause65.md` — checkpoint 验证报告

**内容**: 禁止部分seed新+部分seed旧混用，硬约束。

## [66] 禁止把冒烟数据并入全量测试集
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_smoke_merge.yaml` — 冒烟数据禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p75_no_smoke_merge 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_66.md` — 合规报告
- `checkpoints/compliance_report_clause66.md` — checkpoint 验证报告

**内容**: 禁止冒烟>=12条测试数据并入全量测试集，整批化执行，硬约束。

## [67] 全量单元
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/full_scale_unit.yaml` — 全量单元配置
- `src/liquidloc/analysis/clause_gates.py`（p76_full_scale_unit 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_67.md` — 合规报告
- `checkpoints/compliance_report_clause67.md` — checkpoint 验证报告

**内容**: 全量单元：主场景10 seed×5方法=50单元全部用新数据训练。

## [68] 执行偏差（deviation）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/execution_deviation.yaml` — 执行偏差配置（含 deviation 表、E 系列审计、目录命名规范）
- `src/liquidloc/analysis/clause_gates.py:7103-7203` — 验证函数 `p68_execution_deviation()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_68.md` — 合规报告
- `checkpoints/compliance_report_clause68.md` — 验证快照

**内容**: 执行层面与方案的偏差（漏跑/换机/中断恢复）→ 记 execution deviation 表（E 系列审计覆盖），不影响协议与分析定义。实验目录结构 run 目录命名规范：`run-<日期>-<seed>-<method>-<config_hash>`。

## [69] CPU 角色分层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/cpu_role_description.yaml` — CPU 角色分层配置（cpu_non_training_only/cpu_idle_training/no_conflict/priority_rule）
- `src/liquidloc/analysis/clause_gates.py:7418-7519` — 验证函数 `p69_cpu_role_hierarchy()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_69.md` — 合规报告
- `checkpoints/compliance_report_clause69.md` — 验证快照

**内容**: "CPU 仅限非训练负载"与"CPU 空闲协助训练"的关系：前者是 GPU 空闲时的默认分工（CPU 只做数据生成/评估/预处理），后者是 GPU 排满后的追加自由度（CPU 可承接剩余训练单元）——两者不冲突，GPU 有空的按前者，GPU 排满的按后者。

## [70] 评估并行
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause70_evaluation_parallel.yaml` — 评估并行配置（gpu_execution=true, trajectory_batch_size=16, no_batch_mixing=true, evaluation_protocol_unchanged=true）
- `src/liquidloc/analysis/clause_gates.py`（p70_evaluation_parallel 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_70.md` — 合规报告
- `checkpoints/compliance_report_clause70.md` — checkpoint 验证报告

**内容**: 评估并行：测试轨迹推理在 GPU 上按轨迹独立并行执行（轨迹内窗口批大小 16，轨迹间不混批，评估协议不变）。

## [71] 父进程链可回溯
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/parent_process_traceable.yaml` — 父进程链可回溯配置（traceable=true, command="ps -o ppid= -p <pid>"）
- `src/liquidloc/analysis/clause_gates.py:8337-8417` — 验证函数 `p77_parent_process_traceable()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_71.md` — 合规报告
- `checkpoints/compliance_report_clause71.md` — checkpoint 验证报告

**内容**: 父进程链可回溯到本体系的启动器（`ps -o ppid= -p <pid>` 逐级上溯，最终父进程是本体系的调度/启动脚本）。

## [72] Transformer 滑窗同轨迹内合批
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/transformer_sliding_window.yaml` — Transformer 滑窗合批配置（enabled=true, method="同轨迹内窗口合批", log_batch_composition=true）
- `src/liquidloc/analysis/clause_gates.py:8420-8505` — 验证函数 `p78_transformer_sliding_window()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_72.md` — 合规报告
- `checkpoints/compliance_report_clause72.md` — checkpoint 验证报告

**内容**: Transformer 滑窗同轨迹内合批 评估日志打印批构成（同轨迹内窗口）。

## [73] GPU 报错停
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/gpu_error_handling.yaml` — GPU 报错处理配置（verified=true, cpu_idle_assistance=true, cpu_only_for_full_gpu=true）
- `src/liquidloc/analysis/clause_gates.py:8508-8592` — 验证函数 `p79_no_gpu_error_stop()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_73.md` — 合规报告
- `checkpoints/compliance_report_clause73.md` — checkpoint 验证报告

**内容**: 在无 GPU 环境实测 → 确认报错停而非静默 CPU 训练。CPU 空闲协助（执行模式/执行模式·CPU 空闲协助）CPU 承接仅限 GPU 排满后启用的剩余训练单元。

## [74] RAM 上限 12G
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/ram_limit_hard_cap.yaml` — RAM 上限配置（max_gb=12, verification_method="冒烟实测单单元峰值内存（psutil 探针）"）
- `src/liquidloc/analysis/clause_gates.py:8595-8676` — 验证函数 `p80_ram_limit_12g()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_74.md` — 合规报告
- `checkpoints/compliance_report_clause74.md` — checkpoint 验证报告

**内容**: RAM ≤12G 硬上限（执行模式/执行模式·RAM ≤12G 硬上限）；验证方法（细化，自测留痕）：冒烟实测单单元峰值内存（psutil 探针）。

## [75] 禁止借大修变口径
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_refactor_changes.yaml` — 禁止变口径配置（no_change_paper_level_values=true, no_change_data_gen=true, no_change_eval=true, no_change_protocol=true）
- `src/liquidloc/analysis/clause_gates.py:8679-8768` — 验证函数 `p81_no_refactor_changes()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_75.md` — 合规报告
- `checkpoints/compliance_report_clause75.md` — checkpoint 验证报告

**内容**: 禁止借大修变口径——大修只改代码组织与可维护性（模块拆分/常量收敛/断言补齐/命名统一），不得改变任何论文级数值、不得改数据生成/评估/统计口径、不得为了好写而改协议。

## [76]

## [15]


## [16] 底线：第1阶段附加义务（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录第1阶段附加义务规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十六第1阶段附加义务配置文件（硬约束声明，含条款名加约束级别加治理阶段零等标志）
- 源码目录条款门禁验证模块（p一六资源冲突错峰排队调度函数）
- 版本控制预提交钩子（第1阶段附加义务检查调用点）
- 输出目录八字段落实报告（纯中文表述）
**判据**: 落实中发现缺陷→顺手修根因（禁止留待办事项/绕开/注释旧实现）；缺陷检查范围=阶段一至六条款映射代码；每条条款落实完成时做一次缺陷检查（静态审查/运行断言/单测至少一种）；发现即修根因；修复检查结果写入八字段报告"审查:"字段并留痕
## [5] 依赖声明（治理层·指令模式）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause05_dependency_declaration.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py:15259-15507` — p5_dependency_declaration() 验证函数
- `.git/hooks/pre-commit:111,229` — check_dependencies 前置依赖链硬拦截（定义@111、调用@229）
- `outputs/compliance_report_5.md` — 合规报告

**内容**: 每条指令头部标「前置:[N]」；执行前先查前置是否done，未done → 标blocked停下问，不硬跑、不跳序。pre-commit钩子check_dependencies在[2]-[6]依赖链上实施硬拦截（exit 1阻止提交）。



## [14] 底线：防统计作弊（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录防统计作弊规则文件 — 防统计作弊规则文件（核心判据加禁止项加执行保障）
- 配置文件目录条款十四防统计作弊配置文件 — 防统计作弊配置（硬约束声明）
- 条款门禁源码 — 验证函数
- 版本控制预提交钩子 — 防统计作弊检查调用点
- 输出目录八字段落实报告 — 八字段落实报告
**判据**: 禁止选择性补跑/改数据到满意/旧数据复用/诊断轨并入验收轨

## [13] 底线：语言铁律（GLM 5.3 专属）
**状态**: ✅ 已落地
**文件**:
-  — 语言铁律规则文件（核心判据+禁止项+允许例外+违规处置）
-  — 语言铁律配置（clause_id=13, constraint_level=硬约束）
-  — 验证函数 
- [pre-commit] 前置检查 [2] 依赖 [1]: OK
[pre-commit] 前置检查 [3] 依赖 [1][2]: OK
[pre-commit] 前置检查 [4] 依赖 [1][2][3]: OK
[pre-commit] 前置检查 [5] 依赖 [1][2][3][4]: OK
[pre-commit] 前置检查 [6] 依赖 [1][2][3][4][5]: OK
[pre-commit] [8] 提交前自查: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | cfg_exists=True; cfg_parse_ok=True; clause_name_contains_8=True; constraint_level_is_hard=True; self_check_before_submit=True; check_all_reports_pass=True; check_status_table_done=True; check_manual_goals_met=True; check_eight_fields_complete=True; manual_exists=True; manual_contains_8=True; manual_contains_self_check=True; manual_contains_core_criteria=True; manual_contains_acceptance_table=True; hook_exists=True; hook_contains_p8_call=True; agents_md_exists=True; agents_md_has_clause_8_section=True; status_table_exists=True; status_table_has_clause_8=True; status_table_clause_8_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [9] 落实报告格式: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_all_eight_fields=True; rules_all_fields_have_three_levels=True; one_sentence_template=True; governance_homology=True; pre_submit_self_check=True; complete_examples_code=True; complete_examples_governance=True; judgment_iron_rule=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_9=True; eight_fields_spec=True; one_sentence_template_cfg=True; governance_homology_cfg=True; pre_submit_self_check_cfg=True; complete_examples_cfg=True; judgment_iron_rule_cfg=True; manual_exists=True; manual_contains_9=True; manual_contains_format=True; manual_contains_eight_fields=True; hook_exists=True; hook_contains_p9_call=True; agents_md_exists=True; agents_md_has_clause_9_section=True; status_table_exists=True; status_table_has_clause_9=True; status_table_clause_9_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True
[pre-commit] [10] 逐条落实工序: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | skill_exists=True; step1_check=True; step2_implementation=True; step3_verification=True; step4_reporting=True; eight_fields_spec=True; self_check_list=True; failure_makes_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_10=True; agents_md_exists=True; agents_md_has_clause_10=True; status_table_exists=True; status_table_has_clause_10=True; status_table_clause_10_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p10_call=True
[pre-commit] [11] 条款归属总则: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | rules_exists=True; rules_has_phase0=True; rules_has_phase1_6=True; rules_has_criterion_merged=True; rules_has_phase7_8=True; rules_has_final_check=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_11=True; agents_md_exists=True; agents_md_has_clause_11=True; status_table_exists=True; status_table_has_clause_11=True; status_table_clause_11_field_valid=True; manual_exists=True; manual_contains_11=True; manual_contains_attribution=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p11_call=True
[pre-commit] [12] 手册目标: [配置] ===== bridge_thresholds.py 常量 =====
[配置] cov_jitter_eps = 1e-09
[配置] flow_missing_dyaw_quality_penalty = 0.5
[配置] imu_missing_inflation = 10.0
[配置] max_consecutive_skip_count = 1000.0
[配置] non_current_scaling_ceiling = 2.5
[配置] risk_hard_skip_threshold = 0.05
[配置] risk_max = 1.0
[配置] risk_min = 0.0
[配置] scaling_max = 50.0
[配置] scaling_min = 1.0
[配置] uwb_bias_absolute_max = 5.0
[配置] uwb_bias_max_ratio = 0.5
[配置] uwb_hard_skip_quality_floor = 0.1
[配置] uwb_noise_multiplier_ceiling = 5000.0
[配置] uwb_risk_hard_skip_threshold = 0.95
[配置] vio_dyaw_max = 6.283185307179586
[配置] vio_dyaw_min = -6.283185307179586
[配置] vio_hard_skip_quality_floor = 0.12
[配置] vio_noise_multiplier_ceiling = 5000.0
[配置] vio_risk_hard_skip_threshold = 0.05
[配置] ===== bridge_thresholds.py 常量 结束 =====
[配置] ===== load_sensor_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_sensor_contract 入口参数 结束 =====
[配置] ===== version.py 常量 =====
[配置] CONFIG_VERSION = 1
[配置] OUTPUT_CONTRACT_VERSION = 1
[配置] PROTOCOL_VERSION = 27
[配置] SNAPSHOT_VERSION = 1
[配置] ===== version.py 常量 结束 =====
[配置] ===== load_task_contract 入口参数 =====
[配置] config_path = None
[配置] ===== load_task_contract 入口参数 结束 =====
[配置] ===== metric_schema.py 常量 =====
[配置] _METRICS_CONFIG_PATH = 'E:\异步高NLOS\configs\base\metrics.yaml'
[配置] _METRIC_GROUP_FIELD_MAP:
[配置]   mechanism = 'mechanism_metrics'
[配置]   primary = 'primary_metrics'
[配置]   runtime = 'runtime_metrics'
[配置]   secondary = 'secondary_metrics'
[配置] _METRIC_META:
[配置]   ate = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   bias_alignment = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   corr_scaling_error = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   coverage = {'unit': 'ratio', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   failure_rate = {'unit': 'ratio', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   latency_mean = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p50 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   latency_p95 = {'unit': 'ms', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   mae = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   mean_rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p95 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   p99 = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   params = {'unit': 'count', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   ram_peak = {'unit': 'MB', 'direction': 'lower_is_better', 'group': 'runtime'}
[配置]   risk_error_corr = {'unit': 'corr', 'direction': 'higher_is_better', 'group': 'mechanism'}
[配置]   rmse = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'primary'}
[配置]   rpe = {'unit': 'm', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   std_rmse = {'unit': 'm', 'direction': 'neutral', 'group': 'primary'}
[配置]   yaw_p95 = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置]   yaw_rmse = {'unit': 'rad', 'direction': 'lower_is_better', 'group': 'secondary'}
[配置] _METRIC_ORDER = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate', 'p99', 'mae', 'ate', 'rpe', 'yaw_rmse', 'yaw_p95', 'risk_error_corr', 'coverage', 'bias_alignment', 'corr_scaling_error', 'latency_mean', 'latency_p50', 'latency_p95', 'params', 'ram_peak']
[配置] _PRIMARY_METRICS = ['rmse', 'mean_rmse', 'std_rmse', 'p95', 'failure_rate']
[配置] ===== metric_schema.py 常量 结束 =====
PASSED: True | manual_exists=True; manual_has_criterion=True; manual_has_line_a=True; manual_has_line_b=True; manual_has_aux=True; manual_has_void=True; cfg_exists=True; cfg_parse_ok=True; clause_name_contains_12=True; agents_md_exists=True; agents_md_has_clause_12=True; status_table_exists=True; status_table_has_clause_12=True; status_table_clause_12_field_valid=True; report_exists=True; report_has_eight_fields=True; report_has_done_status=True; hook_exists=True; hook_contains_p12_call=True — 语言铁律检查调用点（仅检查落实报告文件）
-  — 八字段落实报告
**判据**: 全程简体中文；禁止英文回答/禁止中英混杂；3处及以上违规=报告作废重做

## [6] 代码与文件冗余治理（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause06_redundancy_governance.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p6_redundancy_governance() 验证函数
- `outputs/compliance_report_6.md` — 合规报告

**内容**: 禁止为同一逻辑留多份拷贝、禁止复制旧文件再改的备份文化，历史版本走 git，工作目录只保留当前生效版本；发现 _backup/_old/_copy 类冗余文件 → 移入 archive 或删除；禁止"注释掉一大段代码留作备用"；清理动作只作用于本体系目录，禁止跨体系删改；清理任何进程前必须先判定归属。

## [7] 违反即处置（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause07_violation_disposition.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p7_violation_disposition() 验证函数
- `rules/判定铁律.md` — 违反即处置章节（五、违反即处置）
- `outputs/compliance_report_7.md` — 合规报告

**内容**: 执行中任何环节发现违反本文档（未清场/审计失败/口径不符/数据或权重污染）→ 隔离作废相关产物（archive 留痕）→ 修复根因 → 回到对应回归点重跑，处置证据随结果一并提交；任何"未达预期"的第一解释永远是执行偏差，不是方法不行、不是数据集不行；窗口外先修根因再谈结论。

## [8] 提交前自查（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause08_submit_self_check.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p8_submit_self_check() 验证函数
- `.git/hooks/pre-commit` — pre-commit 自查调用点（p8调用在check_dependencies之后）
- `outputs/compliance_report_8.md` — 合规报告

**内容**: 向用户提交论文级结果前先逐项自查是否符合手册目标（核心判据 + 验收表），全部符合才整理提交。

## [10] 逐条落实工序（治理层·四步闭环）
**状态**: ✅ 已落地
**文件**:
- `skills/逐条落实工序/SKILL.md` — 四步闭环工序唯一权威载体（查→落→验→报）
- `configs/protocol/clause10_four_step_closure.yaml` — 约束配置（硬约束声明、四步标志）
- `src/liquidloc/analysis/clause_gates.py` — p10_four_step_closure() 验证函数
- `.git/hooks/pre-commit` — pre-commit 四步工序验证调用点（p10调用在p9之后）
- `outputs/compliance_report_10.md` — 合规报告

**内容**: 每条指令按「查 → 落 → 验 → 报」四步闭环：查（读条款原文+查磁盘现状，禁止凭记忆）→ 落（翻译为文件路径+函数/类+行号落点）→ 验（逐行核对条款逐句对应，不落实当场改）→ 报（按八字段写落实记录，未写记录视为未落实）。

## [11] 条款归属总则（治理层）
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 三阶段归属总则（治理层/Stage1-6落实类/Stage7-8执行决策类）
- `configs/protocol/clause11_clause_attribution.yaml` — 约束配置（硬约束声明、归属标志）
- `src/liquidloc/analysis/clause_gates.py` — p11_clause_attribution() 验证函数
- `.git/hooks/pre-commit` — pre-commit 归属总则验证调用点（p11调用在p10之后）
- `outputs/compliance_report_11.md` — 合规报告

**内容**: 治理层条款→第0阶段逐条落实并写报告，证据写对应治理载体（按『治理载体分散布局』分散落点：AGENTS/rules/skill/hook/plugin/MCP）；Stage 1-6落实类→第1阶段代码落实，判据条款的判据已并入对应代码条款；Stage 7-8执行决策类→第2阶段执行，每条同样单独写执行落实报告。

## [12] 手册目标（治理层·最终检查只看这个）
**状态**: ✅ 已落地
**文件**:
- `rules/手册目标.md` — 核心判据唯一判据（主线A/B必须同时达成）
- `configs/protocol/clause12_manual_objectives.yaml` — 约束配置（硬约束声明、判据标志）
- `src/liquidloc/analysis/clause_gates.py` — p12_manual_objectives() 验证函数
- `.git/hooks/pre-commit` — pre-commit 手册目标验证调用点（p12调用在p11之后）
- `outputs/compliance_report_12.md` — 合规报告

**内容**: 防异步高NLOS实验RMSE整体过大（异步+NLOS下误差过大时所有方法一起高）；防LNN（Liquid液态骨干网络，proposed）的RMSE不是最低；两项判据统一为唯一判据，跑前冒烟/预达标与跑后判定共用；验收分级与降级出口：验收表每一项必须在完整验收路径下逐项回查修通，诊断轨/冒烟数据不得以任何形式并入、替代或补充验收轨数据；数据/代码/结果可追溯（manifest + git commit + 验收留痕）；核验通过后最终只看目标是否达成，不看流程勾选。

## [9] 落实报告格式（治理层·八字段）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause09_compliance_report_format.yaml` — 约束配置
- `rules/落实报告格式规范.md` — 八字段格式规范（唯一权威副本，含八字段逐字段说明/一句话模板/治理类同构/提交前自检/完整示例/判定铁律）
- `.agents/skills/execution-report-gate/SKILL.md` — §1 为八字段格式唯一逐字权威副本（入口）
- `src/liquidloc/analysis/clause_gates.py` — p9_compliance_report_format() 验证函数
- `.git/hooks/pre-commit` — pre-commit 八字段验证调用点（p9调用在p8之后）
- `outputs/compliance_report_9.md` — 合规报告

**内容**: 每条指令输出 = 落实报告格式一行，八字段逐字段规范（条款号/状态/条款逐字抄录/代码绝对定位/落实极尽具体+全量映射/审查问题→修复+循环轨迹/验证命令→预期/证据真实路径+stdout落盘），含一句话模板、治理类同构、提交前自检、完整示例、判定铁律；缺任一字段或字段不合格 = 该条未落实。细则进 `rules/落实报告格式规范.md`，不堆进 AGENTS.md。

## [36] git 是最后防线
**状态**: ✅ 已落地
**文件**: 
- `configs/protocol/git_is_last_resort.yaml` — 约束配置
- `.git/hooks/pre-commit` — pre-commit 防线（第 1-161 行）
- `.gitignore:30-31` — *.log 和 *.bak 防护规则
- `outputs/compliance_report_36.md` — 合规报告

**内容**: git 仓库与工作区必须一致，任何版本回退/文件损坏/误改恢复一律走 git（checkout/revert），禁止复制 _backup/_old 副本。

## [37] 档位定义声明
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/gear_definition_declaration.yaml` — 约束配置
- `configs/protocol/axis_a_sync.yaml` — A 轴档位协议
- `configs/protocol/axis_n_nlos.yaml` — N 轴档位协议
- `configs/protocol/axis_v_visual.yaml` — V 轴档位协议
- `configs/protocol/axis_k_anchor.yaml` — K 轴档位协议
- `configs/protocol/axis_m_missing.yaml` — M 轴档位协议
- `src/liquidloc/analysis/clause_gates.py:3242-3395` — 验证函数 `p37_gear_definition_declaration()`
- `outputs/compliance_report_37.md` — 合规报告

**内容**: A/N/V/K/M 各轴档位划分、参数与具体数值以 Stage 1 档位协议条款为唯一权威副本。

## [38] LOS 段测距噪声 σ≈0.6m
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/los_noise_baseline.yaml` — LOS 段测距噪声基准配置（noise_value: 0.6, clause_id: 38）
- `configs/models/ekf.yaml:30-31` — EKF UWB 量测噪声 σ=0.6m（S4 LOS 基准 σ0.6m）
- `configs/models/robust_ekf.yaml:34-35` — Robust-EKF UWB 量测噪声 σ=0.6m（S4 LOS σ0.6m）
- `outputs/compliance_report_38.md` — 合规报告

**内容**: LOS 段测距噪声 σ≈0.6m（器件事例级均值误差，作为 LOS 基准，非协议档位参数）。LOS 段高斯噪声（商用 UWB 模块室内实测测距误差标准差在亚米级、LOS 与 NLOS 统计分布差异显著）。
## [39] UWB(DW1000级, 10Hz) 输出数据规范
**状态**: ✅ 已落地
**文件**:
- `rules/条款归属总则.md` — 条款治理规则文件
- `configs/protocol/uwb_output_spec.yaml` — UWB 输出数据规范配置（chip_model: DW1000级, frequency_hz: 10Hz）
- `src/liquidloc/analysis/clause_gates.py:3761-3927` — 验证函数 `p39_uwb_output_spec()`
- `outputs/compliance_report_39.md` — 合规报告

**内容**: UWB(DW1000级, 10Hz)；输出数据：`uwb_ranges`(N×4 测距, m) + `uwb_valid`(N×4 有效掩码)。UWB 传感器输出数据规范，包含测距矩阵与有效掩码矩阵。

## [40] IMU（消费级，150Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/imu_output_spec.yaml` — IMU 输出数据规范配置（chip_type: 消费级，frequency_hz: 150Hz）
- `src/liquidloc/analysis/clause_gates.py:3930-4110` — 验证函数 `p40_imu_output_spec()`
- `outputs/compliance_report_40.md` — 合规报告

**内容**: IMU（消费级，150Hz）；输出数据：`imu`（N_imu×6：ax/ay/az/gx/gy/gz，150Hz 原生，不降采样）+ `imu_t`（N_imu×1 自身时间戳）。IMU 传感器输出数据规范。

## [41] VIO（单目+IMU 前端，20 Hz）输出数据规范
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/vio_output_spec.yaml` — VIO 输出数据规范配置（frequency_hz: 20Hz, sensor_type: 单目+IMU前端）
- `src/liquidloc/analysis/clause_gates.py:4111-4324` — 验证函数 `p41_vio_output_spec()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_41.md` — 合规报告

**内容**: VIO（单目+IMU 前端，20 Hz）；输出数据：`vio_pos`（N×2/3）+ `vio_yaw`（N×1）+ `vio_valid`（N×1）+ 协方差（V0 健康档有效帧率 100%）。VIO 传感器输出数据规范。

## [42] 四个输出头
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/four_output_heads.yaml` — 四个输出头机制链配置（uwb_scaling/bias/vio_scaling/risk）
- `configs/protocol/imu_no_head_reasoning.yaml` — IMU 不设头推理说明配置
- `src/liquidloc/analysis/clause_gates.py:4326-4518` — 验证函数 `p42_imu_no_head_reasoning()`
- `src/liquidloc/analysis/clause_gates.py:4521-4665` — 验证函数 `p42b_four_output_heads()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_42.md` — 合规报告

**内容**: 四个输出头（uwb_scaling / bias / vio_scaling / risk）对三源量测做自适应加权、偏差补偿与门控，是"NLOS 识别 → 协方差调制/降权 → 鲁棒门控"机制链的落地（与 S4.2 鲁棒融合组合拳一致）。IMU 作为高频运动先验不参与头调制（惯性量测不经数据驱动加权，走 EKF 运动模型传播），故不设 IMU 头；risk 头为通道级联合风险。

## [43] 禁止使用档位空间外的参数、禁止自行定义插值档/新档
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_extra_levels.yaml` — 禁止额外档位配置（forbidden_interpolation: true, forbidden_new_levels: true）
- `src/liquidloc/analysis/clause_gates.py:4859-4960` — 验证函数 `p43_no_extra_levels()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_43.md` — 合规报告

**内容**: 档位参数严格取自档位协议条款档位表（A/N/V/K/M 五轴），禁止使用档位空间外的参数值，禁止自行定义插值档（如 N1.5 类）或任何新档位。

## [44] 档位注入纪律
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause44_injection_discipline.yaml` — 档位注入纪律配置（静态口径/禁止漂移档位）
- `src/liquidloc/analysis/clause_gates.py:4668-4770` — 验证函数 `p44_gear_injection_discipline()`
- `outputs/compliance_report_44.md` — 合规报告

**内容**: A轴为静态时钟偏移/延迟统计口径（数值见 Stage 1 档位协议条款 A 轴表），严格按静态口径注入，不引入随时间线性增长的时钟漂移作为额外档位（如需评估 → 协议裁决项 + 敏感性分析）。

## [45] N1.5 边界协议
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/n15_boundary.yaml` — N1.5 边界协议配置
- `src/liquidloc/analysis/clause_gates.py:4759-4855` — 验证函数 `p45_n15_boundary()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_45.md` — 合规报告

**内容**: N1.5（ρ=20–25%/μ=1.5–2.5m/σ=0.5m）不在本手册档位协议条款档位表内；本手册引用协议 N 轴为 N0–N3；N1.5 属于同机另一实验体系，按其各自对应协议文件执行。

## [46] 协议引用边界
**状态**: ✅ 已落地（由 p49_protocol_reference_boundary() 验证）
**文件**:
- `configs/protocol/protocol_reference_boundary.yaml` — 协议引用边界配置
- `src/liquidloc/analysis/clause_gates.py:5098` 起 — 验证函数 `p49_protocol_reference_boundary()`
- `outputs/compliance_report_46.md` — 合规报告

**内容**: 档位以 Stage 1 档位协议条款为唯一权威副本；其他文件引用档位时必须标明出自该条款，不得复制参数数值形成第二权威副本。注意：验证函数沿用 p49 命名（历史实现），实际验证手册 [46] 协议引用边界条款。

## [47] 三层证据结构总表
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/three_layer_evidence_structure.yaml` — 三层证据结构总表配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p47_three_layer_evidence()`
- `outputs/compliance_report_47.md` — 合规报告

**内容**: 三层证据结构总表（论文级数据架构，与 S3 轨迹规格衔接）：主场景（受控）S1–S3固定布局（20m×20m、4锚、4遮挡区）10seed×5方法=50单元；布局随机化泛化层；统计推断与主场景共享层。

## [48] 规模档位
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/scale_gear_spec.yaml` — 规模档位配置（入门 N_layout=6、推荐 N_layout=10）
- `src/liquidloc/analysis/clause_gates.py:5189-5290` — 验证函数 `p48_two_sigma_separation()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_48.md` — 合规报告

**内容**: 规模档位：入门 N_layout=6、推荐 N_layout=10（full + 2 个最强基线 LSTM/EKF × 2–3 seed 即可，泛化层只判排序不做因子推断，不需要主场景同等功效）。

## [49] 布局参数与方法排序的散点/箱线
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_scatter_plot_spec.yaml` — 布局参数散点配置
- `src/liquidloc/analysis/clause_gates.py`（p54_layout_parameter_scatter 验证函数）
- `outputs/compliance_report_49.md` — 合规报告

**内容**: 布局参数变化与方法排序关系以散点/箱线图呈现，禁止只用汇总统计量。

## [50] 测试轨迹与训练无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause50_test_train_no_overlap.yaml` — 测试轨迹与训练无重叠配置
- `src/liquidloc/analysis/clause_gates.py`（p50_test_train_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_50.md` — 合规报告

**内容**: 测试轨迹与训练数据在序列级别严格分离，不存在任何重叠。所有评估指标均在独立于训练集的数据上计算。


## [51] 划分按整条序列、训练与测试无重叠
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause51_sequence_split_no_overlap.yaml` — 序列级划分配置
- `src/liquidloc/analysis/clause_gates.py`（p51_sequence_split_no_overlap 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_51.md` — 合规报告

**内容**: 划分按整条序列进行，训练集与测试集之间不存在任何序列级别的重叠。禁止在同一序列内随机采样部分帧/窗口作为训练、部分帧/窗口作为测试，防止数据泄漏。

## [52] 基线不是裸模型
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/baseline_not_bare_model.yaml` — 基线不是裸模型配置
- `src/liquidloc/analysis/clause_gates.py`（p52_baseline_not_bare_model 验证函数）
- `outputs/compliance_report_52.md` — 合规报告

**内容**: 确认 LSTM/Transformer 基线确实带头运行——否则 LNN 赢它们="4 个头 + 液态骨干"双重优势，对比不公平。

## [53] 配对样本对齐
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/paired_sample_alignment.yaml` — 配对样本对齐配置
- `src/liquidloc/analysis/clause_gates.py:5658-5729` — 验证函数 `p53_paired_sample_alignment()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_53.md` — 合规报告

**内容**: 各方法评估按"同一测试轨迹 × 同 seed"配对。

## [54] 两两比较附Holm-Bonferroni多重比较校正
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/holm_bonferroni_correction.yaml` — Holm-Bonferroni校正配置
- `src/liquidloc/analysis/clause_gates.py`（p60_holm_bonferroni_correction 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_54.md` — 合规报告

**内容**: 五方法两两比较共10对，使用Holm-Bonferroni校正控制族错误率。多方法对比不做校正会放大假阳性；Holm-Bonferroni为稳健默认；ML比较应配套统计检验+CI+校正。

## [55] 扩样路径
**状态**: 落实中
**文件**:
- `configs/protocol/scale_expansion_path.yaml` — 扩样路径配置
- `src/liquidloc/analysis/clause_gates.py`（p55_scale_expansion_path 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_55.md` — 合规报告

**内容**: 10 seed 不足支撑功效时如实声明并给出扩样路径（只增不减：加 seed/轨迹至功效达标，规模恒 ≥ 论文级 10 seed × ≥100/seed，论文级硬约束——不存在"收窄/降档"方向的出口）。

## [56] 分析单位与样本独立性声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/analysis_unit_sample_independence.yaml` — 分析单位配置
- `src/liquidloc/analysis/clause_gates.py`（p64_analysis_unit_sample_independence 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_56.md` — 合规报告

**内容**: 滑窗W=128步（步长默认1帧）使相邻窗口重叠127/128，帧级误差样本高度自相关、不满足i.i.d.，所有统计检验必须以轨迹级为分析单位。

## [57] 先导实验（pilot）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/pilot_experiment.yaml` — 先导实验配置
- `src/liquidloc/analysis/clause_gates.py` — 验证函数 `p57_sequence_split_no_overlap()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_57.md` — 合规报告

**内容**: 先导实验单独用于方案冻结前估计效应量/方差/可行性，规模<=1 seed×四组合各1-2条测试轨迹。

## [58] A2/A3 的偏移与抖动数值按协议生效
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/a2a3_shift_jitter.yaml` — A2/A3 偏移与抖动协议配置（clause_id="58"）
- `src/liquidloc/analysis/clause_gates.py::p68_a2a3_shift_jitter()` — 验证函数
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_58.md` — 合规报告
- `checkpoints/compliance_report_clause58.md` — 验证快照

**内容**: A2/A3的偏移与抖动数值按协议生效，已按A2/A3全参数逐项核对。A2：UWB时钟偏移20-50ms、UWB往返延迟40-80ms、IMU时钟偏移5-15ms、IMU采样抖动±3ms、VIO时钟偏移20-50ms、VIO处理延迟30-80ms。

## [59] 重采样方法声明（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/resampling_method_declaration.yaml` — 重采样方法声明配置
- `src/liquidloc/analysis/clause_gates.py`（p69_resampling_method 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_59.md` — 合规报告

**内容**: UWB/VIO重采样采用最近邻（nearest-neighbor）对齐到UWB帧时刻，不使用线性/样条插值生成测距或VIO值；声明写入论文方法节。

## [60] 边界/极端情况
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/edge_case_extreme.yaml` — 边界/极端情况配置
- `src/liquidloc/analysis/clause_gates.py`（p60_edge_case_extreme 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_60.md` — 合规报告

**内容**: 边界极端情况：空轨迹、极短序列、M1长间隙落在首尾，不产生人工尖峰/除零。

## [61] 同 seed 初始化
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/same_seed_init.yaml` — 同seed初始化配置
- `src/liquidloc/analysis/clause_gates.py`（p71_same_seed_init 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_61.md` — 合规报告

**内容**: 同seed初始化：LNN/LSTM/Transformer用同一seed的权重初始化。

## [62] 论文级规模参数
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/power_sample_size.yaml` — 规模参数配置
- `src/liquidloc/analysis/clause_gates.py`（p62_power_sample_size 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_62.md` — 合规报告

**内容**: 论文级规模参数：训练12-15条/seed，测试>=100条/seed，四组合各>=25，120s基准，10 seed（论文级基线）。

## [63] 布局随机化泛化层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/layout_randomization.yaml` — 布局随机化配置
- `src/liquidloc/analysis/clause_gates.py`（p72_layout_randomization 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_63.md` — 合规报告
- `checkpoints/compliance_report_clause63.md` — checkpoint 验证报告

**内容**: 布局随机化泛化层：N_layout>=6(入门)/10(推荐)，每布局5方法x2-3 seed。

## [64] 重新生成全部 seed 数据
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/regenerate_all_seeds.yaml` — 重新生成全部 seed 配置
- `src/liquidloc/analysis/clause_gates.py`（p73_regenerate_all_seeds 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_64.md` — 合规报告
- `checkpoints/compliance_report_clause64.md` — checkpoint 验证报告

**内容**: 主场景10 seed全部重新生成，禁止新旧数据混用。

## [65] 新旧数据混用禁令（硬约束）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_old_new_mixing.yaml` — 新旧数据混用禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p74_no_old_new_mixing 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_65.md` — 合规报告
- `checkpoints/compliance_report_clause65.md` — checkpoint 验证报告

**内容**: 禁止部分seed新+部分seed旧混用，硬约束。

## [66] 禁止把冒烟数据并入全量测试集
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_smoke_merge.yaml` — 冒烟数据禁令配置
- `src/liquidloc/analysis/clause_gates.py`（p75_no_smoke_merge 验证函数）
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_66.md` — 合规报告
- `checkpoints/compliance_report_clause66.md` — checkpoint 验证报告

**内容**: 禁止冒烟>=12条测试数据并入全量测试集，整批化执行，硬约束。

## [67] 全量单元
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/full_scale_unit.yaml` — 全量单元配置
- `src/liquidloc/analysis/clause_gates.py`（p76_full_scale_unit 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_67.md` — 合规报告
- `checkpoints/compliance_report_clause67.md` — checkpoint 验证报告

**内容**: 全量单元：主场景10 seed×5方法=50单元全部用新数据训练。

## [68] 执行偏差（deviation）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/execution_deviation.yaml` — 执行偏差配置（含 deviation 表、E 系列审计、目录命名规范）
- `src/liquidloc/analysis/clause_gates.py:7103-7203` — 验证函数 `p68_execution_deviation()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_68.md` — 合规报告
- `checkpoints/compliance_report_clause68.md` — 验证快照

**内容**: 执行层面与方案的偏差（漏跑/换机/中断恢复）→ 记 execution deviation 表（E 系列审计覆盖），不影响协议与分析定义。实验目录结构 run 目录命名规范：`run-<日期>-<seed>-<method>-<config_hash>`。

## [69] CPU 角色分层
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/cpu_role_description.yaml` — CPU 角色分层配置（cpu_non_training_only/cpu_idle_training/no_conflict/priority_rule）
- `src/liquidloc/analysis/clause_gates.py:7418-7519` — 验证函数 `p69_cpu_role_hierarchy()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_69.md` — 合规报告
- `checkpoints/compliance_report_clause69.md` — 验证快照

**内容**: "CPU 仅限非训练负载"与"CPU 空闲协助训练"的关系：前者是 GPU 空闲时的默认分工（CPU 只做数据生成/评估/预处理），后者是 GPU 排满后的追加自由度（CPU 可承接剩余训练单元）——两者不冲突，GPU 有空的按前者，GPU 排满的按后者。

## [70] 评估并行
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause70_evaluation_parallel.yaml` — 评估并行配置（gpu_execution=true, trajectory_batch_size=16, no_batch_mixing=true, evaluation_protocol_unchanged=true）
- `src/liquidloc/analysis/clause_gates.py`（p70_evaluation_parallel 验证函数）
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_70.md` — 合规报告
- `checkpoints/compliance_report_clause70.md` — checkpoint 验证报告

**内容**: 评估并行：测试轨迹推理在 GPU 上按轨迹独立并行执行（轨迹内窗口批大小 16，轨迹间不混批，评估协议不变）。

## [71] 父进程链可回溯
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/parent_process_traceable.yaml` — 父进程链可回溯配置（traceable=true, command="ps -o ppid= -p <pid>"）
- `src/liquidloc/analysis/clause_gates.py:8337-8417` — 验证函数 `p77_parent_process_traceable()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_71.md` — 合规报告
- `checkpoints/compliance_report_clause71.md` — checkpoint 验证报告

**内容**: 父进程链可回溯到本体系的启动器（`ps -o ppid= -p <pid>` 逐级上溯，最终父进程是本体系的调度/启动脚本）。

## [72] Transformer 滑窗同轨迹内合批
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/transformer_sliding_window.yaml` — Transformer 滑窗合批配置（enabled=true, method="同轨迹内窗口合批", log_batch_composition=true）
- `src/liquidloc/analysis/clause_gates.py:8420-8505` — 验证函数 `p78_transformer_sliding_window()`
- `rules/条款归属总则.md` — 条款治理规则文件
- `outputs/compliance_report_72.md` — 合规报告
- `checkpoints/compliance_report_clause72.md` — checkpoint 验证报告

**内容**: Transformer 滑窗同轨迹内合批 评估日志打印批构成（同轨迹内窗口）。

## [73] GPU 报错停
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/gpu_error_handling.yaml` — GPU 报错处理配置（verified=true, cpu_idle_assistance=true, cpu_only_for_full_gpu=true）
- `src/liquidloc/analysis/clause_gates.py:8508-8592` — 验证函数 `p79_no_gpu_error_stop()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_73.md` — 合规报告
- `checkpoints/compliance_report_clause73.md` — checkpoint 验证报告

**内容**: 在无 GPU 环境实测 → 确认报错停而非静默 CPU 训练。CPU 空闲协助（执行模式/执行模式·CPU 空闲协助）CPU 承接仅限 GPU 排满后启用的剩余训练单元。

## [74] RAM 上限 12G
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/ram_limit_hard_cap.yaml` — RAM 上限配置（max_gb=12, verification_method="冒烟实测单单元峰值内存（psutil 探针）"）
- `src/liquidloc/analysis/clause_gates.py:8595-8676` — 验证函数 `p80_ram_limit_12g()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_74.md` — 合规报告
- `checkpoints/compliance_report_clause74.md` — checkpoint 验证报告

**内容**: RAM ≤12G 硬上限（执行模式/执行模式·RAM ≤12G 硬上限）；验证方法（细化，自测留痕）：冒烟实测单单元峰值内存（psutil 探针）。

## [75] 禁止借大修变口径
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/no_refactor_changes.yaml` — 禁止变口径配置（no_change_paper_level_values=true, no_change_data_gen=true, no_change_eval=true, no_change_protocol=true）
- `src/liquidloc/analysis/clause_gates.py:8679-8768` — 验证函数 `p81_no_refactor_changes()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_75.md` — 合规报告
- `checkpoints/compliance_report_clause75.md` — checkpoint 验证报告

**内容**: 禁止借大修变口径——大修只改代码组织与可维护性（模块拆分/常量收敛/断言补齐/命名统一），不得改变任何论文级数值、不得改数据生成/评估/统计口径、不得为了好写而改协议。

## [76]

## [15]

## [16]

## [17]

## [18]

## [19]

## [20] 底线：评估口径（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录评估口径规则文件 — 评估口径规则文件（核心判据加禁止项加允许范围）
- 配置文件目录条款二十评估口径配置文件 — 评估口径配置（条款_id=20, 约束_级别=硬约束）
- 源码目录条款门禁验证模块 — 验证函数 p20_N轴非直线传播偏差档位协议
- 版本控制预提交钩子 — N轴非直线传播偏差检查调用点
- 智能体指南 — 治理载体索引二十条目
- 输出目录八字段落实报告 — 八字段落实报告
**判据**: 同步定位与地图构建轨迹对齐 → 二维平面坐标世界系误差 → 预热期前十秒剔除 → 轨迹级汇总；图表口径硬约束

## [21] 底线：视觉退化验证（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录视觉退化验证规则文件 — 视觉退化验证规则文件（核心判据加禁止项加允许范围）
- 配置文件目录条款二十一视觉退化验证配置文件 — 视觉退化验证配置（条款_id=21, 约束_级别=硬约束）
- 源码目录条款门禁验证模块 — 验证函数 p21_V轴视觉退化档位协议
- 版本控制预提交钩子 — 视觉退化验证检查调用点
- 智能体指南 — 治理载体索引二十一条目
- 输出目录八字段落实报告 — 八字段落实报告
**判据**: 视觉退化测试必做 → 光照变化场景验证 → 能见度降低场景验证 → 对比实验组与对照组差异

## [22] 底线：锚点配置验证（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录锚点配置验证规则文件 — 锚点配置验证规则文件（核心判据加禁止项加允许范围）
- 配置文件目录条款二十二锚点配置验证配置文件 — 锚点配置验证配置（条款_id=22, 约束_级别=硬约束）
- 源码目录条款门禁验证模块 — 验证函数 p22_K轴锚点配置
- 版本控制预提交钩子 — 锚点配置验证检查调用点
- 智能体指南 — 治理载体索引二十二条目
- 输出目录八字段落实报告 — 八字段落实报告
**判据**: 锚点配置必做 → 档位协议条款校验 → K轴配置参数验证 → 与手册定义一致性检查

## [23] 底线：数据缺失档位协议（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录数据缺失档位协议规则文件 — 数据缺失档位协议规则文件（核心判据加禁止项加允许范围）
- 配置文件目录条款二十三数据缺失档位协议配置文件 — 数据缺失档位协议配置（条款_id=23, 约束_级别=硬约束）
- 源码目录条款门禁验证模块 — 验证函数 p23_M轴数据缺失档位协议
- 版本控制预提交钩子 — 数据缺失档位协议检查调用点
- 智能体指南 — 治理载体索引二十三条目
- 输出目录八字段落实报告 — 八字段落实报告
**判据**: 数据缺失必做 → 档位协议条款校验 → M轴缺失数据参数验证 → 与手册定义一致性检查

## [24] 底线：五轴独立性说明（治理层）
**状态**: ✅ 已落实
**文件**:
- 规则目录五轴独立性说明规则文件 — 五轴独立性说明规则文件（核心判据加禁止项加允许范围）
- 配置文件目录条款二十四五轴独立性验证配置文件 — 五轴独立性说明配置（条款_id=24, 约束_级别=硬约束）
- 源码目录条款门禁验证模块 — 验证函数 p24五轴独立性验证
- 版本控制预提交钩子 — 五轴独立性检查调用点
- 智能体指南 — 治理载体索引二十四条目
- 输出目录八字段落实报告 — 八字段落实报告
**判据**: 五轴独立性必做 → 档位协议条款校验 → 五轴配置参数验证 → 与手册定义一致性检查