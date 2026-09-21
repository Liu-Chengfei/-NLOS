## 条款[29] 档位协议条款 · A 轴 —— 同步精度

**条款**: 档位协议条款 · A 轴 —— 同步精度（UWB时钟偏移/往返延迟、IMU时钟偏移/采样抖动、VIO时钟偏移/处理延迟的A0-A3四档定义）

**状态**: ✅ 已落实

**代码**: configs/protocol/axis_a_sync.yaml:1-50（A轴同步精度配置文件）; src/liquidloc/analysis/clause_gates.py::p29_axis_a_sync_verification():17280-17380（验证函数，含A轴六个参数四个档位的配置文件存在性检查+参数验证）; AGENTS.md:187（治理载体索引条目）; rules/指令状态表.md:49（指令状态表条目）; .git/hooks/pre-commit（预提交钩子调用点）

**落实**: 一、发现A轴同步精度条款八字段报告缺失——修复：创建八字段报告；二、发现配置文件缺失——修复：创建configs/protocol/axis_a_sync.yaml；三、发现验证函数缺失——修复：创建p29_axis_a_sync_verification函数；四、发现AGENTS.md无[29]条目——修复：新增[29]治理载体条目；五、发现指令状态表无[29]条目——修复：新增[29]条目

**审查**: 问题：A轴同步精度条款八字段报告缺失——修复：创建八字段报告；问题：配置文件缺失——修复：创建配置文件；问题：验证函数语法错误——修复：修复字典键引用语法

**验证**: 命令: python -c "import sys; sys.path.insert(0, "src"); from liquidloc.analysis.clause_gates import p29_axis_a_sync_verification; r = p29_axis_a_sync_verification(); exit(0 if r["passed"] else 1)" → 预期: 0（exit code 0 表示通过，detail含 cfg_exists=True; mismatches=[]; missing_gears=[]; missing_fields=[]; handount_exists=True）

**证据**: 配置文件目录A轴同步精度配置文件（五十行）；源码目录条款门禁验证模块（p29函数）；版本控制预提交钩子；智能体指南（第187行）；指令状态表（第49行）；输出目录八字段落实报告。

---

**附件**: 完整八字段报告见 outputs/compliance_report_29.md
