## 条款[38] LOS 段测距噪声 σ≈0.6m

**条款**: LOS 段测距噪声 σ≈0.6m（器件事例级均值误差，作为 LOS 基准，非协议档位参数）， LOS 段高斯噪声（商用 UWB 模块室内实测测距误差标准差在亚米级、LOS 与 NLOS 统计分布差异显著 ——）

**状态**: ✅ 已落实

**代码**: configs/protocol/los_noise_baseline.yaml:1-20（LOS噪声配置文件）; src/liquidloc/analysis/clause_gates.py::p39_los_noise_baseline():3891（验证函数，检查噪声值≈0.6m、声明为器件事例级均值误差、作为LOS基准、非协议档位参数、ekf.yaml一致性）; AGENTS.md:267（治理载体索引条目）; rules/指令状态表.md:58（指令状态表条目）; outputs/compliance_report_38.md（八字段报告）

**落实**: 一、发现条款[38]八字段报告缺失——修复：创建outputs/compliance_report_38.md；二、发现指令状态表无[38]条目——修复：新增[38]条目

**审查**: 问题：条款[38]八字段报告缺失——修复：创建八字段报告；问题：指令状态表无[38]条目——修复：新增[38]条目

**验证**: 命令: python -c "import sys; sys.path.insert(0, \"src\"); from liquidloc.analysis.clause_gates import p39_los_noise_baseline; r = p39_los_noise_baseline(); exit(0 if r[\"passed\"] else 1)" → 预期: 0（exit code 0 表示通过，detail含 cfg_exists=True; missing_fields=none; noise_value=0.6 (期望 ≈0.6, 偏差=0.0000); declaration_ok=True (basis='器事例级均值误差' (期望 '器事例级均值误差'); purpose='作为 LOS 基准' (期望 '作为 LOS 基准'); not_a='协议档位参数' (期望 '协议档位参数')); exclusion='非协议档位参数' (期望 '非协议档位参数'); noise_type='LOS 段高斯噪声' (期望 'LOS 段高斯噪声'); role='LOS 基准' (期望 'LOS 基准'); ekf_ok=True (ekf.yaml uwb=0.6 (期望 ≈0.6), match=True; LOS注释=存在); handount_exists=True; concept_hits={'LOS': True, '0.6': True, '高斯噪声': True, '基准': True, '非协议': True, '器事例': True}）

**证据**: 配置文件目录LOS噪声配置文件（二十行）；源码目录条款门禁验证模块（p39函数）；configs/models/ekf.yaml（噪声配置一致性）；智能体指南（第267行）；指令状态表（第58行）；输出目录八字段落实报告。

---

**附件**: 完整八字段报告见 outputs/compliance_report_38.md
