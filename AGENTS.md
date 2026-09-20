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

## [6] 代码与文件冗余治理（治理层）
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/clause06_redundancy_governance.yaml` — 约束配置
- `src/liquidloc/analysis/clause_gates.py` — p6_redundancy_governance() 验证函数
- `outputs/compliance_report_6.md` — 合规报告

**内容**: 禁止为同一逻辑留多份拷贝、禁止复制旧文件再改的备份文化，历史版本走 git，工作目录只保留当前生效版本；发现 _backup/_old/_copy 类冗余文件 → 移入 archive 或删除；禁止"注释掉一大段代码留作备用"；清理动作只作用于本体系目录，禁止跨体系删改；清理任何进程前必须先判定归属。

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

## [76] 大修范围界定
**状态**: ✅ 已落地
**文件**:
- `configs/protocol/refactor_scope_definition.yaml` — 大修范围界定配置（clause_id=76, scope_definition, decision_log, refactor_restriction）
- `src/liquidloc/analysis/clause_gates.py:8771-8882` — 验证函数 `p82_refactor_scope_definition()`
- `rules/判定铁律.md` — 条款治理规则文件
- `outputs/compliance_report_76.md` — 合规报告
- `checkpoints/compliance_report_clause76.md` — checkpoint 验证报告

**内容**: 大修范围界定——明确本次大修的范围（哪个模块/哪个脚本/哪些接口），写入 decision log（时间/原因/影响范围/git commit）。
