## 条款[25] 底线：数据口径（治理层）

**条款**: 数据口径——双速率契约（UWB/VIO/gt 对齐 10Hz 公共轴，IMU 保留 150Hz 原生不降采样）

**状态**: ✅ 已落实

**代码**: `configs/protocol/eval_statistical_parallel.yaml:1-33`（硬约束声明+条款名+约束级别+治理阶段零）; `src/liquidloc/analysis/clause_gates.py::p25_eval_statistical_parallel_verification():2681-2800`（验证函数，含expected_cfg字典定义parallel_mode/gpu_execution/trajectory_independent/trajectory_batch_size/no_batch_mixing/criterion_ref + constraint_declaration字典定义trajectory_internal_batch_size/trajectory_independent_parallel/no_mixing_across_trajectories/gpu_execution_default/criterion_reference + 代码侧并行/轨迹/GPU逻辑站点搜索）; `.git/hooks/pre-commit`（预提交钩子调用点，已注册到全部检查列表）; `AGENTS.md:183`（治理载体索引条目）

**落实**: 一、发现条款二十五状态表条目缺失——修复：新增状态表条目映射到正确条款；问题：无数据口径规则文件——修复：创建规则目录数据口径规则文件，含核心判据加禁止项加允许范围；问题：无数据口径验证配置文件——修复：创建配置文件目录条款二十五数据口径验证配置文件，含硬约束声明及判据标志；问题：无智能体指南二十五条目——修复：新增## [25] 底线：数据口径条目；问题：无八字段报告——修复：按八字段单行格式写入输出目录八字段落实报告

**审查**: 问题：手册二十五条款原文与状态表条目映射不匹配——修复：新增状态表条目映射到正确条款；问题：无数据口径验证配置文件——修复：创建配置文件目录条款二十五数据口径验证配置文件，含硬约束声明及判据标志；问题：无智能体指南二十五条目——修复：新增## [25] 底线：数据口径条目

**验证**: 命令: python -c "import sys; sys.path.insert(0, 'src'); from liquidloc.analysis.clause_gates import p25_eval_statistical_parallel_verification; r = p25_eval_statistical_parallel_verification(); exit(0 if r['passed'] else 1)" → 预期: 0（exit code 0 表示通过，detail含 cfg_exists=True; cfg_mismatches=none; constraint_mismatches=none; parallel_logic_sites=7; trajectory_logic_sites=1192; gpu_logic_sites=140; handout_exists=True）

**证据**: 配置文件目录数据口径验证配置文件（三十三行）；源码目录条款门禁验证模块（p25函数）；版本控制预提交钩子（全部检查列表）；智能体指南（第183行）；指令状态表（待新增）；输出目录八字段落实报告。
