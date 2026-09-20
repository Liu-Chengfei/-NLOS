# 执行报告 gate 规则勾选清单（与手册条款编号一一对应）

**格式说明**：每行格式为 `条款号(验证点序号)` + `；检查内容：...`，对应手册条款[333]规则勾选清单格式模板。
通过项打勾 `☑`，未通过项打叉 `☒`，未验证项留空。

---

## 第 0 阶段 · 治理层条款 [1-23]

### [1] 0 清场
- [1](1)；检查内容：clear_workspace.py 存在且可执行
- [1](2)；检查内容：旧仿真数据集/旧权重/旧结果已清空或归档

### [2] 执行总纲固化
- [2](1)；检查内容：AGENTS.md 存在且五铁律逐字一致
- [2](2)；检查内容：rules/ 目录下 9 个治理载体文件全部存在
- [2](3)；检查内容：skills/ 目录下 3 个 SKILL.md 全部存在
- [2](4)；检查内容：.git/hooks/pre-commit 存在且断言逐字一致
- [2](5)；检查内容：.git/hooks/pre-commit 存在且断言逐字一致（0 清场 + 提交前自查 + 残留审计）
- [2](6)；检查内容：.git/hooks/pre-commit 中 pre-commit 钩子存在且拦截有效

### [3] 状态只认磁盘
- [3](1)；检查内容：rules/状态只认磁盘.md 存在且铁律细则完整
- [3](2)；检查内容：磁盘锚点文件存在且内容与预期一致

### [4] 指令状态表与逐条队列
- [4](1)；检查内容：rules/指令状态表.md 存在且条款编号同步

### [5] 依赖声明
- [5](1)；检查内容：依赖声明文件存在（requirements.txt / environment.yml / pyproject.toml 之一）

### [6] 代码与文件冗余治理
- [6](1)；检查内容：无冗余旧文件名（*_backup / *_old / *_temp）
- [6](2)；检查内容：.gitignore 防护 *.log 和 *.bak

### [7] 违反即处置
- [7](1)；检查内容：违反即处置规则已写入治理载体

### [8] 提交前自查
- [8](1)；检查内容：pre-commit 钩子含提交前自查断言

### [9] 落实报告格式
- [9](1)；检查内容：八字段格式规范已写入 rules/落实报告格式规范.md
- [9](2)；检查内容：八字段格式已写入 .agents/skills/execution-report-gate/SKILL.md §1
- [9](3)；检查内容：compliance_report_*.md 全部含 8 个字段

### [10] 逐条落实工序
- [10](1)；检查内容：skills/逐条落实工序/SKILL.md 存在且四步闭环完整

### [11] 条款归属总则
- [11](1)；检查内容：rules/条款归属总则.md 存在且三阶段归属表完整

### [12] 手册目标
- [12](1)；检查内容：rules/手册目标.md 存在且主线A/B判据完整

### [13] 底线：语言铁律
- [13](1)；检查内容：语言铁律已写入 AGENTS.md 和 rules/底线清单.md

### [14] 底线：防 p-hacking
- [14](1)；检查内容：防 p-hacking 规则已写入 rules/底线清单.md

### [15] 底线：解冻纪律与数据纯净性
- [15](1)；检查内容：解冻纪律已写入 rules/底线清单.md

### [16] 底线：第 1 阶段附加义务
- [16](1)；检查内容：第1阶段附加义务已写入 rules/条款归属总则.md L24-33

### [17] 底线：落实证据必须引用文件:行号
- [17](1)；检查内容：判定铁律已写入 rules/判定铁律.md
- [17](2)；检查内容：所有落实报告的 代码/证据 字段含文件:行号

### [18] 档位引用纪律
- [18](1)；检查内容：档位引用规则已写入 rules/阶段执行地图.md 或 rules/条款归属总则.md

### [19] 全量定义
- [19](1)；检查内容：全量定义已写入 rules/手册目标.md 或 Stage 1 条款

### [20] 评估口径
- [20](1)；检查内容：评估口径已写入 rules/手册目标.md 或 Stage 1 条款

### [21] 治理载体分散布局
- [21](1)；检查内容：rules/治理载体分散布局.md 存在且硬约束完整
- [21](2)；检查内容：AGENTS.md L17 索引仅列出磁盘实际存在的文件（无虚标）

### [22] 阶段执行地图
- [22](1)；检查内容：rules/阶段执行地图.md 存在且执行顺序+阶段速查表完整

### [23] 底线：第 2 阶段附加义务
- [23](1)；检查内容：第2阶段附加义务已写入 rules/条款归属总则.md L24-33

---

## Stage 1 · 代码落实类条款 [36-76]

### [36] git 是最后防线
- [36](1)；检查内容：configs/protocol/git_is_last_resort.yaml 存在
- [36](2)；检查内容：.git/hooks/pre-commit 含 0 清场检查
- [36](3)；检查内容：.gitignore:30-31 含 *.log 和 *.bak 防护规则
- [36](4)；检查内容：outputs/compliance_report_36.md 存在且含八字段

### [37] 档位定义声明
- [37](1)；检查内容：configs/protocol/gear_definition.yaml 存在
- [37](2)；检查内容：outputs/compliance_report_37.md 存在且含八字段

### [38] LOS段测距噪声基准
- [38](1)；检查内容：configs/protocol/los_noise_baseline.yaml 存在
- [38](2)；检查内容：outputs/compliance_report_38.md 存在且含八字段

### [39] UWB输出规范
- [39](1)；检查内容：configs/protocol/uwb_output_spec.yaml 存在
- [39](2)；检查内容：outputs/compliance_report_39.md 存在且含八字段

### [40] IMU输出规范
- [40](1)；检查内容：configs/protocol/imu_output_spec.yaml 存在
- [40](2)；检查内容：outputs/compliance_report_40.md 存在且含八字段

### [41] 五轴独立性
- [41](1)；检查内容：五轴档位协议文件全部存在
- [41](2)；检查内容：outputs/compliance_report_41.md 存在且含八字段

### [42] 桥接参数阈值
- [42](1)；检查内容：src/liquidloc/config/bridge_thresholds.py 存在
- [42](2)；检查内容：outputs/compliance_report_42.md 存在且含八字段

### [43] 传感器合同版本
- [43](1)；检查内容：src/liquidloc/config/sensor_contract.py 存在
- [43](2)；检查内容：configs/base/sensor_contract.yaml 存在
- [43](3)；检查内容：outputs/compliance_report_43.md 存在且含八字段

### [44] 指标schema版本
- [44](1)；检查内容：src/liquidloc/config/metric_schema.py 存在
- [44](2)；检查内容：configs/base/metrics.yaml 存在
- [44](3)；检查内容：outputs/compliance_report_44.md 存在且含八字段

### [45] 任务合同版本
- [45](1)；检查内容：src/liquidloc/config/task_contract.py 存在
- [45](2)；检查内容：configs/base/task_contract.yaml 存在
- [45](3)；检查内容：outputs/compliance_report_45.md 存在且含八字段

### [46] 协议引用边界
- [46](1)；检查内容：configs/protocol/protocol_reference_boundary.yaml 存在
- [46](2)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p49_protocol_reference_boundary
- [46](3)；检查内容：outputs/compliance_report_46.md 存在且含八字段

### [47] 确定性重放
- [47](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p47_deterministic_replay
- [47](2)；检查内容：outputs/compliance_report_47.md 存在且含八字段

### [48] 冻结期防线
- [48](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p48_freeze_period_defense
- [48](2)；检查内容：outputs/compliance_report_48.md 存在且含八字段

### [49] 协议引用边界（验证函数命名沿用 p49）
- [49](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p49_protocol_reference_boundary
- [49](2)；检查内容：outputs/compliance_report_46.md 已覆盖条款 [46]

### [50] 算法锚定
- [50](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p50_algorithm_anchoring
- [50](2)；检查内容：outputs/compliance_report_50.md 存在且含八字段

### [51] 序列级划分
- [51](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p51_sequence_level_split
- [51](2)；检查内容：outputs/compliance_report_51.md 存在且含八字段

### [52] 去标识化
- [52](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p52_deidentification
- [52](2)；检查内容：outputs/compliance_report_52.md 存在且含八字段

### [53] 数据冻结锁
- [53](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p53_data_freeze_lock
- [53](2)；检查内容：outputs/compliance_report_53.md 存在且含八字段

### [54] 基准发布
- [54](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p54_baseline_release
- [54](2)；检查内容：outputs/compliance_report_54.md 存在且含八字段

### [55] 指标冻结
- [55](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p55_metric_freeze
- [55](2)；检查内容：outputs/compliance_report_55.md 存在且含八字段

### [56] 超参锁定
- [56](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p56_hyperparam_lock
- [56](2)；检查内容：outputs/compliance_report_56.md 存在且含八字段

### [57] 训练规模防缩水
- [57](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p57_training_scale_guard
- [57](2)；检查内容：outputs/compliance_report_57.md 存在且含八字段

### [58] 基线不可降级
- [58](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p58_baseline_immutable
- [58](2)；检查内容：outputs/compliance_report_58.md 存在且含八字段

### [59] 论文级再训练
- [59](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p59_paper_grade_retrain
- [59](2)；检查内容：outputs/compliance_report_59.md 存在且含八字段

### [60] 论文级报告
- [60](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p60_paper_grade_report
- [60](2)；检查内容：outputs/compliance_report_60.md 存在且含八字段

### [61] 收尾共识
- [61](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p61_wrapup_consensus
- [61](2)；检查内容：outputs/compliance_report_61.md 存在且含八字段

### [62] 功效样本量
- [62](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p62_power_sample_size
- [62](2)；检查内容：outputs/compliance_report_62.md 存在且含八字段

### [63] 扩样路径
- [63](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p63_augmentation_path
- [63](2)；检查内容：outputs/compliance_report_63.md 存在且含八字段

### [64] 分析单位
- [64](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p64_analysis_unit
- [64](2)；检查内容：outputs/compliance_report_64.md 存在且含八字段

### [65] 先导实验
- [65](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p65_pilot
- [65](2)；检查内容：configs/protocol/pilot.yaml 存在
- [65](3)；检查内容：outputs/compliance_report_65.md 存在且含八字段

### [66] 无NaN/Inf/梯度爆炸
- [66](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p66_no_nan_inf_grad_explosion
- [66](2)；检查内容：训练监控含 loss 收敛断言 + 权重范数断言
- [66](3)；检查内容：outputs/compliance_report_66.md 存在且含八字段

### [67] 运行时内存上限
- [67](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p67_runtime_memory_ceiling
- [67](2)；检查内容：outputs/compliance_report_67.md 存在且含八字段

### [68] 设备调度纪律
- [68](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p68_device_scheduling_discipline
- [68](2)；检查内容：outputs/compliance_report_68.md 存在且含八字段

### [69] 数据流完整
- [69](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p69_data_flow_integrity
- [69](2)；检查内容：outputs/compliance_report_69.md 存在且含八字段

### [70] 全局种子
- [70](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p70_global_seed
- [70](2)；检查内容：outputs/compliance_report_70.md 存在且含八字段

### [71] 梯度裁剪
- [71](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p71_gradient_clipping
- [71](2)；检查内容：outputs/compliance_report_71.md 存在且含八字段

### [72] 早停策略
- [72](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p72_early_stopping_policy
- [72](2)；检查内容：outputs/compliance_report_72.md 存在且含八字段

### [73] 日志格式
- [73](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p73_log_format
- [73](2)；检查内容：outputs/compliance_report_73.md 存在且含八字段

### [74] 指标可追溯
- [74](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p74_metric_traceability
- [74](2)；检查内容：outputs/compliance_report_74.md 存在且含八字段

### [75] 结果可复现
- [75](1)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p75_reproducibility
- [75](2)；检查内容：outputs/compliance_report_75.md 存在且含八字段

### [76] 大修范围界定
- [76](1)；检查内容：configs/protocol/ 下配置文件全部存在
- [76](2)；检查内容：src/liquidloc/analysis/clause_gates.py 含 p76_scope_boundary
- [76](3)；检查内容：rules/ + skills/ + .git/hooks/ 配置文件全部存在
- [76](4)；检查内容：outputs/compliance_report_76.md 存在且含八字段

---

## 使用说明

1. **每过 gate 逐项勾选**：通过打 `☑`，未通过打 `☒`，未验证留空
2. **条款 [46] 与 [49]**：共享同一验证函数 `p49_protocol_reference_boundary`，[49] 项仅检查函数存在性，实际覆盖由 [46] 项负责
3. **扩展方式**：新增条款时在对应阶段区块追加 `### [N] 条款名` + 检查项
4. **与本清单不一致时**：以本清单为准，手册条款[333]为格式来源
