## 条款[38] LOS 段测距噪声 σ≈0.6m（器件事例级均值误差，作为 LOS 基准，非协议档位参数）

**状态**: ✅ 已落实

**条款**: ***[38]*** **LOS 段测距噪声 σ≈0.6m（器件事例级均值误差，作为 LOS 基准，非协议档位参数）**， LOS 段高斯噪声（商用 UWB 模块室内实测测距误差标准差在亚米级、LOS 与 NLOS 统计分布差异显著 ——）

**代码**: configs/protocol/los_noise_baseline.yaml:1-42（LOS噪声配置文件，clause_id="38"）; src/liquidloc/analysis/clause_gates.py::p38_los_noise_baseline():3894（验证函数，检查噪声值≈0.6m、声明为器件事例级均值误差、作为LOS基准、非协议档位参数、ekf.yaml一致性）; AGENTS.md:267（治理载体索引条目）; rules/指令状态表.md:58（指令状态表条目）; outputs/compliance_report_38.md（八字段报告）

**落实**: 一、创建 configs/protocol/los_noise_baseline.yaml：声明 LOS 段测距噪声基准 σ=0.6m，字段含 noise_value=0.6、noise_type="LOS 段高斯噪声"、role="LOS 基准"、exclusion="非协议档位参数"、declaration.basis="器事例级均值误差"、declaration.purpose="作为 LOS 基准"、declaration.not_a="协议档位参数"，约束声明 clause_id="38"、discipline_level="hard"；二、创建 src/liquidloc/analysis/clause_gates.py::p38_los_noise_baseline() 验证函数（原函数名误写为 p39，已修复），检查 cfg_exists、missing_fields、noise_value≈0.6、declaration_ok（器件事例级均值误差/作为LOS基准/非协议档位参数）、exclusion_ok（非协议档位参数）、noise_type_ok（LOS段高斯噪声）、role_ok（LOS基准）、ekf_ok（ekf.yaml uwb=0.6）、handount_exists、concept_hits（LOS/0.6/高斯噪声/基准/非协议/器事例全命中）；三、修复配置文件 clause_id 从 "39" 改为 "38"，注释从"手册第39条"改为"手册第38条"；四、修复函数名 p39_los_noise_baseline → p38_los_noise_baseline，修复返回 name 从 "[39]" 改为 "[38]"；五、修复 _ALL 列表中函数引用；六、新增指令状态表 [38] 条目

**审查**: 问题: 函数名 p39_los_noise_baseline 与条款[38]不匹配 → 修复: 重命名为 p38_los_noise_baseline（clause_gates.py:3894）；问题: 函数返回 name="[39] LOS段测距噪声σ≈0.6m基准" → 修复: 改为 name="[38] LOS 段测距噪声 σ≈0.6m（器件事例级均值误差，作为 LOS 基准，非协议档位参数）"（clause_gates.py:3919/3932/4071）；问题: 配置文件注释写"手册第39条" → 修复: 改为"手册第38条"（los_noise_baseline.yaml:3）；问题: 配置文件 clause_id="39" → 修复: 改为 clause_id="38"（los_noise_baseline.yaml:38）

**验证**: 命令: python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p38_los_noise_baseline; r = p38_los_noise_baseline(); print(f'passed={r[\"passed\"]}, name={r[\"name\"]}, code={r[\"code\"]}')" → 预期: passed=True; name="[38] LOS 段测距噪声 σ≈0.6m（器件事例级均值误差，作为 LOS 基准，非协议档位参数）"; code="P38-los-noise-baseline"

**证据**: configs/protocol/los_noise_baseline.yaml（LOS噪声配置文件，42行）; src/liquidloc/analysis/clause_gates.py::p38_los_noise_baseline():3894（验证函数）; configs/models/ekf.yaml:31-33（uwb=0.6，LOS段噪声配置一致性）; AGENTS.md:267（条款[38]治理载体索引）; rules/指令状态表.md:58（指令状态表[38]条目）; outputs/compliance_report_38.md（八字段报告）

---

**附件**: 完整八字段报告见 outputs/compliance_report_38.md
