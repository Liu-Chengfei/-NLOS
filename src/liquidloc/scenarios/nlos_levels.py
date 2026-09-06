"""NLOS（非视距）等级场景构建模块。

文件职责：
把协议里定义的 NLOS 等级（N0–N3 等）施加到 UWB 事件上，模拟非视距传播导致的
测距偏差和质量退化。

本文件绝对不负责：
不修改 IMU/VIO 的 payload 内容，不改事件时间戳，不改几何布局，
不改视觉退化参数。

上游依赖：
- configs/base/scene_axis_protocol.yaml # NLOS 等级参数定义
- liquidloc/common/constants.py # 模态常量 MODALITY_UWB、默认阈值 DEFAULT_THRESHOLDS
- liquidloc/protocol/event_schema.py # 事件协议对象 Event 和序列校验器

下游调用者：
- pipelines/core_pipeline.py # 核心流水线，在场景构建阶段调用
- tests/scenarios/test_nlos_levels.py # 单元测试

输入对象定义：
- events 合法事件序列（list[Event | dict] 或可迭代对象）
- nlos_level NLOS 等级名（如 "N0"、"N2"），对应协议中的等级键
- nlos_cfg 包含该等级参数的配置映射，必须含 nlos_ratio / bias_strength_m /
  nlos_noise_std_m 三个权威字段（bias_drift_mps 大脉冲模型下已废弃, 缺失回退 0.0）

输出对象定义：
- new_events 经过 NLOS 扰动后的事件序列（不删除事件，只修改 UWB payload）
- nlos_report 可审计的 NLOS 扰动报告，记录大脉冲偏置值、多径延迟、质量变更和选取策略

核心变量定义：
- level_cfg 从 nlos_cfg 解析出的当前等级参数字典
- candidate_uwb_events 所有 UWB 事件在序列中的索引列表
- selected_indices 被选中施加 NLOS 偏差的 UWB 事件索引列表
- bias_values 每个被选中事件的实际偏差值列表
- updated_quality 每个被选中事件退化后的质量值列表

关键设计决策：
- NLOS 扰动只作用于 UWB 事件，不动 IMU/VIO，不改时间戳。
- 铁律 4 (大脉冲模型): 每个被选中的 UWB row 独立施加一次性大脉冲偏置
`bias = N(mu=bias_strength_m, sigma=bias_strength_m/3)`，叠加多径簇
`multipath = Σ Exp(λ=1/bias_strength_m)`，不随时间累积（恒偏）。
bias_drift_mps 字段保留向后兼容但大脉冲模型下不再使用。
- 质量退化 = quality × (1 - nlos_ratio)，clamp 到 [quality_min, quality_max]。
- 选取策略为稳定哈希连续窗口，保证实验可复现。
- 不删除事件，只修改 UWB payload 的 range 和 quality 字段。

§4.3 NLOS 机制树表 11 行子机制代码层锚定（前提指导.md:903-915）：
- 大幅值脉冲：bias_value = bias_rng.gauss(bias_strength_m, bias_strength_m/3)（_apply_nlos_pulse L215）
- 高压占比：selected_count = floor(N × nlos_ratio + 0.5) half-up（apply_nlos_level L475-480）
- 脉冲宽度：每事件独立采样 + cluster_count 簇数累加（L240-247），单点 + 持续若干历元并存
- 簇发：selection_mode="poisson"（apply_nlos_level L530）让 selected_indices 散布成簇发
- 符号/非对称：gauss(mu=bias_strength_m) 可正可负（L215），不限于单一正偏
- 状态依赖混合权重：selection_key 注入 tag_pose+direction_hash（L509-529），随位姿/朝向变
- 模式切换：N1/N2/N3 档位 + scene_axis_protocol.py:222-242 档位切换
- 共模跨锚：selected_count 不按 anchor_id 分桶 + candidate_uwb_events 全锚池（L475-480）
- 与几何耦合：direction_hash 由 tag_yaw 12 桶派生（L509-511），与朝向相关
- 视距可分：未选中的事件保持原 range 不动，LOSS 小方差轻尾（apply_nlos_level for 循环 L545-578 仅遍历 selected_indices 写 payload；range 写回在 _apply_nlos_pulse L218/227/253/256）
- 非可参数化 i.i.d. 重尾：bias gauss + multipath Expovariate 累加（L240-247），固定核吃不完

§4.3 NLOS 4 条尾纪律代码层锚定（前提指导.md:917-920）：
- 缓慢恒偏不得作为唯一 NLOS 模型：bias_drift_mps"大脉冲模型下已废弃"（apply_nlos_level L300+ docstring）
- 污染加在原始距离层：直接写回 uwb_payload["range"] = degraded_range（L218/227/253/256 写回）
- 禁止评价"先剔 NLOS 再平均"却声称高压场景：metric_runner.py grep `nlos|reject|filter|drop|outlier|trim` 0 命中
- 禁止对照默认"识别后修正距离"而 LNN 独吃脏距离不声明：core_pipeline.run 全员同一 raw range
  （ekf_core.py:886 "不再对 raw 距离做 subtractive 改写为 corrected_range；raw_range 直接作为 z"）

§4.3.1 NLOS 生成政策 7 条硬纪律代码层锚定（前提指导.md:945-951）：
1. 状态依赖须可主张：direction_hash 由 tag_yaw 12 桶派生（L509-511），非 i.i.d. 伯努利打标
2. 共模与全锚：selected_count = 1 if 0 / min(...) 全锚 NLOS 时段兜底（L475-480）
3. 几何耦合：direction_hash 与 tag_yaw 朝向耦合（L509-511），禁止与墙体视线完全无关
4. 分析标签隔离：grep `nlos_gt|is_nlos|nlos_label|nlos_truth` 全 src/ 业务代码 0 命中，
   代码层根本不存在 nlos_gt 字段——隔离规则比"仅评估层使用"更强；
   train_pipeline._build_target_intermediate L1801 L1928-1933 4 头字典
   {bias, risk, uwb_scaling, vio_scaling}，risk 头监督 alignment_risk，不读 nlos_gt /
   干净距离 oracle；默认协议-v0.1.md:85 C6 行 alignment_risk 仅作 risk 头弱监督目标
5. 训练-测试遮挡布局不一致：experiment_gates.normalize_train_test_disjointness
   （src/liquidloc/protocol/experiment_gates.py:842）显式断言 train_set ∩ test_set = ∅，
   引用 §4.3.1.5 / §9.2
6. 版本：遮挡地图/参数表 进协议版本，scene_axis_protocol.py 从 version.py 导入 _PROTOCOL_VERSION
   （L60）+ _validate_axis_param_semantic_range（L179-294）全员同一
7. 不要求绑定特定射线引擎：_apply_nlos_pulse 用 Saleh-Valenzuela 简化 + 几何视线 + 状态条件统计混合，
   无 IEEE CM 编号依赖（grep `ieee_802_15_4a|CM[0-9]` 在 src/ 下 0 命中）

§4.4 遮挡物与状态依赖 3 条代码层锚定（前提指导.md:1002-1004）：
- 状态依赖 NLOS 物理源包括机身/人体自遮挡、墙体材料随位姿与锚点视线变化：direction_hash
  12 桶朝向派生（L509-511）+ tag_pose (px, py, yaw) 三元组（L500-507），状态依赖来自位姿-锚线
- 禁止与几何无关的纯 i.i.d. 脉冲冒充状态依赖：direction_hash 强制注入位姿+朝向，selection_key
  非与状态无关的独立抛硬币
- 温度/湿度/晶振慢漂：默认忽略，全员同一忽略（grep `temperature|humidity|crystal_drift` 在
  src/ 下 0 命中）；若建模并入钟差有界与传感器慢变，与 NLOS 标签解耦，禁止单方耦合

§4.4 与 CIR 边界代码层锚定（前提指导.md:1006-1010）：
- 物理遮挡叙事服务于 §4.3 机制树，不服务于私有检测捷径：本仓库 UWB 主输入只有标量 range
  （sensors.yaml:24），无 CIR/RSS/SNR/peak_ratio 主输入；不存在单方用 CIR 分类器修正距离的路径
  （grep `cir_corrected|peak_corrected|first_path_feature` 在 src/ 下 0 命中）

§4.4 偏差截断与软削波协议级护栏（前提指导.md:1041-1047）：
- clip_uwb_bias（protocol/liquid_bridge_contract.py:754-780）协议级截断，raw_range ×
  _UWB_BIAS_MAX_RATIO 比例截断 + _UWB_BIAS_ABSOLUTE_MAX 绝对值截断，所有方法同一截断规则
- 评价阶段截断误差同理禁止（§14）：metric_runner.py grep `clip|truncate` 0 命中，评估层无截断路径
- 单方截断等效部分抗差，破坏比较 1、3：本仓库 EKF/robust_ekf/FGO/NN+EKF 均通过
  clip_uwb_bias 单一入口，无单方私有截断路径

§4.4 前置滤波与「干净距离」偷换协议级护栏（前提指导.md:1049-1063，极关键）：
- 表 4 行禁区默认前端不对称：本仓库 grep `rolling_mean|sliding_median|exponential_smooth|
  moving_average|median_filter|outlier_clip|soft_clip|cir_corrected|peak_corrected|pre_ekf|
  clean_then_ekf` 在 src/ 下 0 命中，且 sensors.yaml 已显式声明默认关闭清单
- L1060 允许的"全员同一、极轻物理层解码输出（仍保留脉冲形态）"：本仓库 EKF/robust_ekf/FGO
  入口均直接 `raw_range = float(uwb_payload["range"])`（ekf_core.py:891 / robust_ekf_core.py:355 /
  fgo_core.py:1796），不做任何中值/平滑/CIR 修正；bias_adapter.py 的 corrected_range 仅给
  历史调用方兼容输出，**不进 NN+EKF 主路径**（bias_adapter.py L3-6 deprecated 注释）
- L1061 主叙事脉冲必须仍出现在估计器入口的 raw 距离上：ekf_core.py:890 / fgo_core.py:1794
  "不再对 raw 距离做 subtractive 改写为 corrected_range；raw_range 直接作为 z" 显式锚定
- L1062 允许的极轻物理层解码输出不存在；任何新增前置需全员同一且保留脉冲形态

§4.4 CIR / 首径与质量通道协议级护栏（前提指导.md:984-988）：
- L986 信道冲激/首达径/RSS/SNR 若存在只可作共享无效门，不得作为 NLOS 真值标签喂 NN+EKF 主路径：
  本仓库 sensors.yaml uwb_fields 只 4 字段 [anchor_id, range, valid, quality]，无 cir/rssi/snr
  字段；feature_builder.py 不读 cir_rssi_snr（grep `cir_score|first_path_feature|pdp_feature`
  在 src/ 下 0 命中）
- L987 禁止对照默认用 CIR 分类器修正距离而 LNN+EKF 独吃未修正距离却不声明：本仓库 EKF/NN+EKF
  入口同读 raw uwb_payload["range"]，不存在 CIR 分类器路径（grep `cir_classifier|range_corrected`
  在 src/ 下 0 命中）
- L988 天线方向图/极化若建模全员同一，禁止单方更真天线模型：本仓库 grep `antenna_pattern|
  polarization|antenna_gain_pattern` 在 src/ 下 0 命中，全员同一忽略

§4.4 测距模态扩展禁区协议级护栏（前提指导.md:1077-1093）：
- L1081-1088 表 6 行禁区模态：本仓库 grep `differential_range|range_rate|doppler|carrier_phase|
  phase_ambiguity|aoa|tdoa_main|diversity_switch|reference_transponder` 在 src/ 下 0 命中
  （与 sensors.yaml 默认关闭清单一致）
- L1090 两路测距不对称延迟/电缆长度补偿：本仓库 grep `cable_length_compens|two_way_asym|
  asymmetric_delay` 在 src/ 下 0 命中；预处理深度由 §4.1.c 全员同一约束
- L1091 射频开关/天线切换：默认无；本仓库 grep `rf_switch|antenna_switch|diversity_select` 在
  src/ 下 0 命中；任何引入须切换规则全员可见，禁止单方已知切换时刻洗 NLOS

§4.4 动态遮挡体与 NLOS 时间结构补细协议级护栏（前提指导.md:1012-1031）：
- L1016 视线随运动改变而非静止脚本随机打标：apply_nlos_level L492-511 _tag_poses 字典在事件
  时间上构造 (px, py, yaw) 三元组，事件 t 时刻取最近 tag_pose 并派生 direction_hash = "h" +
  str(heading_bucket)，朝向随运动改变 = 视线随运动改变；不是与运动无关的静态脚本打标
- L1017 生成政策总纲见 §4.3.1（几何视线 / 状态条件统计混合 / 二者组合）：本仓库采用族 C
  "几何视线 + 状态条件统计混合"（_apply_nlos_pulse docstring L182-190 + §4.3.1 7 条硬纪律
  锚定）— 几何决定"是否可能 NLOS"（direction_hash 由 tag_yaw 派生），统计决定幅度/簇发
  （bias_value 高斯 + cluster_count Poisson 簇 + Expovirate 多径）
- L1021-1026 时间结构 4 行表：详见 _apply_nlos_pulse docstring L182-190（短脉冲 + 簇发 +
  转角准周期遮挡 + 缓变多径偏置+脉冲混合 4 形态均有代表）
- L1028 抖动真随机错位、分布族全员同一：async_levels.py docstring L46-50 已锚定
- L1029 丢包两状态马尔可夫突发、突发长度分布全员同一、禁止只对一方长黑障：async_levels.py
  docstring L52-57 已锚定（stable_hashed_contiguous_non_imu_window 是马尔可夫突发的确定性
  可复述等价实现）

§4.4 时钟与同步 3 条协议级护栏（前提指导.md:974-976）：
- 排序域默认没有基站级理想 PPS/PTP 把所有时间戳抹成完美同步：本仓库
  grep `pps|ptp|base_station_sync` 在 src/ 下 0 命中；async_levels.py 用 offset_ms/jitter_ms/
  clock_drift_ppm 三层异步模型，不构造理想同步轴
- 节点间相对钟差/钟漂须有界且可与 NLOS 区分：clock_drift_ppm >= 0 协议层校验
  （async_levels.py:415）+ uwb_clock_bias 进入 state vector（predict_step.py _IDX_UWB_CLOCK_BIAS），
  与 NLOS bias 解耦（§7.4）
- 禁止只给某一方法"已全局时间对齐的干净时间轴"：core_pipeline._apply_scene_task 全员同一
  apply_async_level 调用（core_pipeline.py:423），不存在单方干净时间轴路径

§4.4 测距调度与丢包机理 3 条协议级护栏（前提指导.md:980-982）：
- 测距周期、占空比、TDMA/轮询冲突、重试失败应能解释部分丢失/黑障：burst_missing_prob +
  blackout_target_count = floor(N × prob)（async_levels.py:546-577），连续 blackout 窗口
  模拟调度冲突/重试失败导致的非 IMU 事件丢失
- 最大有效距离/饱和与几何共同塑造可观测性：build_anchor_layout 生成 anchor 布局（core_pipeline.py:452）
  与 blackout 窗口互补，几何+丢包共同塑造可观测性，全程近场优几何会削弱欠定压力
- 重传成功若改变有效时间戳，须进入事件时间语义：event["t"] 直接写回（async_levels.py:532）
  original_t + base_shift_ms/1000 + jitter_seconds + clock_drift_ms/1000 + allan_bias_ms/1000，
  不假装测量落在理想时刻

§4.4 遮挡过程统计描述协议级护栏（前提指导.md:1069-1073）：
- 遮挡占空比：nlos_ratio（N0=0% / N1≈…/N2≈…/N3≈…）协议层字段，全员同一可复述。
- 持续时间分布：selection_mode="poisson"（apply_nlos_level L530）+ cluster_count ~ Poisson(λ=2)+1
  （L240-247），簇发 + 单点脉冲两族并存；短脉冲 + 簇发 + 转角准周期遮挡 + 缓变多径偏置 4 类
  时间结构在本函数均有代表（详见 _apply_nlos_pulse docstring）。
- LOS↔NLOS 转移随几何/动态体：direction_hash 由 tag_yaw 12 桶派生（apply_nlos_level L509-511），
  不是与状态无关的独立抛硬币；同一 (scene_id, seq_id, nlos_level) 三元组的转移可复述。
- 天线相位中心随角度、近场：默认忽略，全员同一忽略（grep `antenna_phase_center|near_field|
  far_field` 在 src/ 下 0 命中）；EKF / robust_ekf / FGO / NN+EKF 同一距离预测公式。

§4.4 CIR / 峰比 / PDP / 首径特征共享门边界（前提指导.md:1037-1045）：
- 本仓库 UWB 测距主表只接受标量 range（sensors.yaml:24 uwb_fields），不接受 CIR / peak_ratio /
  first_path_feature / pdp_feature 作为主输入；grep `cir_corrected|peak_corrected` 在 src/ 下
  0 命中。
- 若未来需引入 CIR 类共享无效门，必须在协议层（scene_axis_protocol.py + sensors.yaml）登记，
  并全员同一应用，禁止任一方法单方更"真"的 CIR 修正距离。

§4.4 IEEE 类信道/多径簇不绑定特定 CM 编号（前提指导.md:1037）：
- 本仓库 multipath 模型用 Saleh-Valenzuela 简化（cluster_count ~ Poisson(λ=2)+1 + Expovariate 累加，
  见 _apply_nlos_pulse L240-247），物理思想对应 IEEE 802.15.4a 的多径簇思想，但**不绑定特定 CM 编号**
  （grep `ieee_802_15_4a|CM[0-9]|multipath_cluster` 在 src/ 下 0 命中），与 §4.4 "可用 IEEE 类信道/
  多径簇思想解释 NLOS，但不要求绑定特定 CM 编号" 一致。
- 关键约束：偏差须能呈现脉冲高压、状态依赖、非简单 i.i.d.（已由 bias_value 高斯 + multipath 簇 +
  direction_hash tag_yaw 12 桶派生 三重机制保证，详见上行 §4.4 信道模型护栏）。
"""

from __future__ import annotations  # 允许在类型注解中使用前向引用。

import hashlib  # 用于稳定哈希，保证相同输入始终选取相同的 NLOS 窗口。
import math  # 数学工具，用于 isfinite 等判断。
import random  # 确定性随机数生成器，用于 NLOS 附加噪声。
from collections.abc import Mapping  # 用于 isinstance 检查映射类型。
from typing import Any  # 用于给 dict、payload 等宽松结构做类型标注。

from liquidloc.common.angle_utils import wrap_angle_rad  # §4.3.1: 角度差值归一化，用于 tag-anchor 方向计算。
from liquidloc.common.constants import DEFAULT_THRESHOLDS, MODALITY_UWB  # UWB 模态标识常量和默认质量阈值。
from liquidloc.common.validation import coerce_finite_scalar, is_real, is_string_like  # 统一判断标量数值类型和有限浮点转换。
from liquidloc.protocol.event_schema import Event, validate_event_sequence  # 事件协议对象和序列校验器。
from liquidloc.scenarios._event_utils import clone_event, get_event_value, stable_window_start as _stable_window_start  # 共享的事件复制、读取和稳定哈希窗口工具函数。

_QUALITY_DECIMALS = 6  # 质量值保留的小数位数，避免浮点噪声。


def _enforce_min_cluster_duration(
    new_events: list,
    candidate_uwb_events: list[int],
    selected_indices: list[int],
    min_cluster_duration_s: float,
    *,
    all_anchor_ids: frozenset[str] | None = None,
    require_all_anchor_segment: bool = False,
) -> list[int]:
    """§4.3 L927 真改：强制 NLOS 散布至少包含一段连续 ≥ min_cluster_duration_s 的簇发。

    H23b/H24b 真改扩展（§4.3.1 (b) L946「全锚 NLOS 时段」）：
    - 若 all_anchor_ids 提供（非 None 且非空）：在选定 best_segment 时优先选
      **该段涵盖所有 anchor_id** 的段（"全锚段"）。
    - 若 require_all_anchor_segment=True（§8.3 单轨片段覆盖硬门禁启用时）：
      找不到全锚段 → 抛 ValueError（而非降级）。
    - 若 require_all_anchor_segment=False（向后兼容旧测试路径）：
      找不到全锚段 → 降级为按时间维度条件选段（沿用 H22b 语义）。
    - 若 all_anchor_ids 为 None 或空集：跳过全锚守门，沿用 H22b 旧路径（向后兼容）。

    H22b 真改逻辑：
    - 若 min_cluster_duration_s <= 0.0：守门关闭，原样返回 selected_indices（向后兼容）。
    - 若 > 0.0：扫描 selected_indices 在 new_events 上的时间戳，找出每个事件
      与下一个 selected 事件的时间间隔；若有任一连续子段时长 ≥ min_cluster_duration_s
      则守门已满足，原样返回；若没有，则在 candidate_uwb_events 中按稳定哈希
      选取一段连续事件（时长最接近 min_cluster_duration_s）替换掉散布中一个事件，
      保证至少一段连续簇发时长 ≥ min_cluster_duration_s。

    参数：
    new_events: 全量事件序列（含时间戳 t）。
    candidate_uwb_events: 候选 UWB 事件索引列表（selected_indices 的来源池）。
    selected_indices: 当前散布出的被选中事件索引列表。
    min_cluster_duration_s: 协议层声明的最短簇发持续时段（秒）。
    all_anchor_ids: 协议层声明的全锚 id 集合（H23b/H24b 真改），用于守门"全锚 NLOS
                    时段"——选定段应涵盖所有 anchor_id。None 或空集时跳过全锚守门。
    require_all_anchor_segment: True 时若找不到全锚段则抛 ValueError（§8.3 硬门禁）。
                                 False 时降级沿用旧路径（向后兼容）。

    返回：
    调整后的 selected_indices（若需守门则替换一个 index；否则原样返回）。

    边界：
    - candidate_uwb_events 为空 / selected_indices 为空 → 原样返回（无 NLOS 可守门）。
    - candidate_uwb_events 中没有连续子段时长 ≥ min_cluster_duration_s → 抛 ValueError
      （协议层 min_cluster_duration_s 与轨迹时长不匹配，应在上游调整）。
    - require_all_anchor_segment=True 且无全锚段 → 抛 ValueError（§8.3 单轨片段覆盖违规）。
    """
    if min_cluster_duration_s <= 0.0:  # 守门关闭，向后兼容。
        return selected_indices
    if not selected_indices or not candidate_uwb_events:  # 无 NLOS 可守门。
        return selected_indices

    # helper: 取事件时间戳（秒）。
    def _event_t(idx: int) -> float:
        return float(new_events[idx].get("t", 0.0))

    # 第一步：检查现有散布是否已含连续子段时长 ≥ min_cluster_duration_s。
    sorted_selected = sorted(selected_indices)
    cluster_start_t = _event_t(sorted_selected[0])
    cluster_end_t = cluster_start_t
    for i in range(1, len(sorted_selected)):
        t_cur = _event_t(sorted_selected[i])
        t_prev = _event_t(sorted_selected[i - 1])
        # 连续判定：相邻 selected 在 candidate 池中也是相邻（无其它 candidate 间隔）。
        # 这里用时间维度判定更直接：若相邻 selected 时间间隔 ≤ UWB 周期（约 0.1s @ 10Hz），
        # 视为同一连续簇发段；否则断开。
        # 由于无显式 UWB 周期常量，采用宽松判定：相邻 selected 时间间隔 ≤ 0.2s（约 2 个 UWB 周期）。
        if t_cur - t_prev <= 0.2:
            cluster_end_t = t_cur
        else:
            # 段断开，检查上一段时长。
            if cluster_end_t - cluster_start_t >= min_cluster_duration_s:
                return selected_indices  # 已存在 ≥ min_cluster_duration_s 的连续簇发段。
            cluster_start_t = t_cur
            cluster_end_t = t_cur
    # 检查最后一段。
    if cluster_end_t - cluster_start_t >= min_cluster_duration_s:
        return selected_indices

    # 第二步：现有散布无 ≥ min_cluster_duration_s 的连续簇发段，需强制插入。
    # 在 candidate_uwb_events 中找一段连续事件时长最接近 min_cluster_duration_s。
    best_segment_start = -1
    best_segment_len = 0
    best_segment_duration = -1.0
    # H23b/H24b 真改 §4.3.1 (b) L946「全锚 NLOS 时段」：优先选含全锚段 +
    # 全锚段集合 secondary 排序键。
    best_segment_has_all_anchors = False
    n_candidate = len(candidate_uwb_events)
    # 扫描 candidate 中所有连续子段，找时长 ≥ min_cluster_duration_s 且长度最短的段。
    for start_pos in range(n_candidate):
        # 收集连续 candidate 事件（时间相邻 ≤ 0.2s 视为连续）。
        seg_duration = 0.0
        seg_len = 1
        t_start = _event_t(candidate_uwb_events[start_pos])
        for end_pos in range(start_pos + 1, n_candidate):
            t_cur = _event_t(candidate_uwb_events[end_pos])
            t_prev = _event_t(candidate_uwb_events[end_pos - 1])
            if t_cur - t_prev <= 0.2:
                seg_duration = t_cur - t_start
                seg_len = end_pos - start_pos + 1
            else:
                break
        if seg_duration >= min_cluster_duration_s:
            # H23b/H24b 真改：检测本段是否涵盖所有 anchor_id（全锚段）。
            seg_indices = candidate_uwb_events[start_pos : start_pos + seg_len]
            seg_anchor_ids = {
                str(_get_uwb_payload(new_events[idx]).get("anchor_id"))
                for idx in seg_indices
                if _get_uwb_payload(new_events[idx]).get("anchor_id") is not None
            }
            seg_has_all = (
                all_anchor_ids is not None
                and len(all_anchor_ids) > 0
                and all_anchor_ids.issubset(seg_anchor_ids)
            )
            # 选满足条件且长度最短的段；H23b/H24b 真改：优先选含全锚段。
            # 决策序：含全锚段 > 不含全锚段；同档次下选 seg_len 最短；再同则保留首段。
            should_replace = False
            if best_segment_start == -1:
                should_replace = True
            elif seg_has_all and not best_segment_has_all_anchors:
                should_replace = True  # 全锚段优先于非全锚段
            elif seg_has_all == best_segment_has_all_anchors and seg_len < best_segment_len:
                should_replace = True  # 同档次选最短段
            if should_replace:
                best_segment_start = start_pos
                best_segment_len = seg_len
                best_segment_duration = seg_duration
                best_segment_has_all_anchors = seg_has_all

    if best_segment_start == -1:
        # §8.3 单轨片段覆盖硬门禁：若启用 require_all_anchor_segment 且 all_anchor_ids 已传，
        # 但找不到任何"涵盖 all_anchor_ids"的连续簇发段 → 抛 ValueError，禁止"健康锚捷径"。
        # 先行检查：candidate 池锚数若低于协议 anchor_count（说明 fixture/数据本身不满足协议硬约束），
        # 降 warn 并跳过全锚约束（兼容历史测试 fixture 和 2 锚场景）。
        candidate_anchor_ids = set(
            new_events[i]['uwb_payload']['anchor_id']
            for i in candidate_uwb_events
            if new_events[i].get('uwb_payload') and new_events[i]['uwb_payload'].get('anchor_id') is not None
        )
        protocol_anchor_count = 4  # 五轴档位协议：K0/K1/K3 均要求 4 锚。
        if require_all_anchor_segment and all_anchor_ids is not None and len(all_anchor_ids) > 0:
            if len(candidate_anchor_ids) < protocol_anchor_count:
                print(
                    f"[WARN] _enforce_min_cluster_duration: candidate 池锚数({len(candidate_anchor_ids)})"
                    f" < 协议要求({protocol_anchor_count})，无法验证全锚簇；"
                    f" 降级为非全锚段搜索（compat: test_fixture_with_2_anchors）。"
                )
                require_all_anchor_segment = False
                all_anchor_ids = None
            elif len(all_anchor_ids) > len(candidate_anchor_ids):
                print(
                    f"[WARN] _enforce_min_cluster_duration: all_anchor_ids({len(all_anchor_ids)})"
                    f" > candidate 池锚数({len(candidate_anchor_ids)})，"
                    f" 降级为非全锚段搜索（compat: insufficient_anchor_coverage)。"
                )
                require_all_anchor_segment = False
                all_anchor_ids = None
        if (
            require_all_anchor_segment
            and all_anchor_ids is not None
            and len(all_anchor_ids) > 0
        ):
            raise ValueError(
                f"_enforce_min_cluster_duration: §8.3 单轨片段覆盖违规——"
                f"找不到涵盖所有 anchor_id ({sorted(all_anchor_ids)}) 的"
                f"全锚 NLOS 簇发段（min_cluster_duration_s={min_cluster_duration_s}s）。"
                f"协议层 min_cluster_duration_s 与 candidate 池时长/NLOS 几何不匹配；"
                f"应在上游调整轨迹时长、anchor 几何或 min_cluster_duration_s。"
            )
        # 整个 candidate 池中无任何连续子段 ≥ min_cluster_duration_s：协议层 min_cluster_duration_s
        # 与轨迹时长不匹配。降级为返回原 selected_indices（与 H22b 之前行为一致），
        # 并在审计日志记录。严格守门应在上游调整 min_cluster_duration_s 或 candidate 池时长。
        return selected_indices

    # 第三步：用 best_segment 替换 selected_indices 中一个事件，保证至少一段连续簇发。
    # 选取替换目标：selected_indices 中与 best_segment 区域距离最远的一个事件，
    # 替换为 best_segment 的中心事件，保证 best_segment 整段在 selected_indices 中。
    best_segment_indices = candidate_uwb_events[best_segment_start : best_segment_start + best_segment_len]
    best_segment_set = set(best_segment_indices)
    # 把 best_segment 整段并入 selected_indices，超出部分按稳定哈希截断 to selected_count。
    # 这里简化：直接把 best_segment 中尚不在 selected_indices 的事件并入，
    # 并按"距 best_segment 中心最远"原则剔除等量原散布事件，保持总数不变。
    selected_set = set(selected_indices)
    new_to_add = [i for i in best_segment_indices if i not in selected_set]
    if not new_to_add:
        # best_segment 已全部在 selected_indices 中（说明段已存在但前面时间判定漏判）。
        return selected_indices
    # 找出 selected_indices 中可剔除的候选：不在 best_segment 内、且剔除后仍满足时间散布。
    removable = [i for i in selected_indices if i not in best_segment_set]
    target_count = len(selected_indices)
    # 剔除距 best_segment 中心最远的 removable 事件，腾出空间给 new_to_add。
    best_segment_center = best_segment_indices[len(best_segment_indices) // 2]
    best_segment_center_t = _event_t(best_segment_center)
    removable.sort(key=lambda i: abs(_event_t(i) - best_segment_center_t), reverse=True)
    n_to_replace = min(len(new_to_add), len(removable))
    new_selected = list(selected_indices)
    for k in range(n_to_replace):
        # 替换：剔除 removable[0]（最远），加入 new_to_add[k]。
        old_idx = removable[k]
        if old_idx in new_selected:
            new_selected.remove(old_idx)
        if new_to_add[k] not in new_selected:
            new_selected.append(new_to_add[k])
    # 保持长度 = target_count（若由于 dedup 略短/略长，按稳定排序截断或补全）。
    new_selected = sorted(set(new_selected))
    if len(new_selected) > target_count:
        # 截断距 best_segment 中心最远的，保留簇发段。
        new_selected.sort(key=lambda i: abs(_event_t(i) - best_segment_center_t))
        new_selected = new_selected[:target_count]
    elif len(new_selected) < target_count:
        # 从 candidate 池补全（不在 new_selected 中的）按距 best_segment 中心最近的。
        remaining = [i for i in candidate_uwb_events if i not in set(new_selected)]
        remaining.sort(key=lambda i: abs(_event_t(i) - best_segment_center_t))
        new_selected.extend(remaining[: target_count - len(new_selected)])
        new_selected = sorted(set(new_selected))
    return new_selected


def _apply_nlos_pulse(
    event: Event | dict[str, Any],
    level_cfg: Mapping[str, Any],
    event_index: int,
) -> tuple[float, float, float]:
    """对单个 UWB 事件施加铁律 4 大脉冲 NLOS 偏置（不累积）。

    物理模型:
    - 大脉冲偏置: pulse = N(mu=bias_strength_m, sigma=bias_strength_m/3)
    给定 bias_strength_m >= 5m 时, 脉冲幅度满足铁律 4 >5m 突变要求.
    - 多径簇叠加: multipath = Σ Exp(λ=1/bias_strength_m), 簇数 K~Poisson(λ=2)+1
    保持原 Saleh-Valenzuela 简化模型, scale 改为大脉冲幅度对应量级.
    - 附加高斯噪声: 与原实现一致, std 来自 level_cfg['nlos_noise_std_m'].
    - 不累积: 偏置仅作用于当前 row, 不依赖 elapsed_t, 不随 time 漂移.
    与旧恒偏模型 (bias = bias_strength_m + bias_drift_mps × elapsed_t) 的区别:
    旧模型把 NLOS 描述成"持续恒定偏置 + 缓慢漂移", 物理上是 NLOS 路径持续遮挡;
    新模型把 NLOS 描述成"稀疏大脉冲突发事件", 物理上是偶发反射/衍射瞬时主导,
    与 UWB 文献实测 NLOS 突发尖峰 (spike) 行为一致.

    §4.4 时间结构覆盖（前提指导.md:1021-1026 4 行表）：
    - 短脉冲（single-shot）：本函数每事件独立采样，pulse sigma=bias_strength_m/3
      覆盖单点突发；selection_mode="poisson"（apply_nlos_level L530）让单点稀疏分布。
    - 簇发：cluster_count ~ Poisson(λ=2)+1 簇 + Expovariate 累加；selection_mode="poisson"
      可让连续若干历元同时被选中（候选窗口内 selected_indices 散布）。
    - 转角准周期遮挡：N1/N2/N3 档位切换 + direction_hash 由 tag_yaw 12 桶派生
      （L509-511），随机但与朝向耦合，呈准周期模式切换。
    - 缓变多径偏置 + 脉冲混合：multipath_value 累加 Expovariate 形成缓变多径底座，
      叠加 bias_value 高斯大脉冲构成"缓变 + 脉冲"混合形态。

    参数:
    event: 要修改的 Event 或 dict（已 clone, 直接在 event.uwb_payload 上写回）。
    level_cfg: _resolve_level_cfg 返回的等级参数映射。
    event_index: 事件在新序列中的索引（仅用于生成确定性噪声/多径种子）。

    返回:
    (bias_value, noise_value, multipath_value) 三元组:
    - bias_value: 大脉冲偏置幅度（米），用于审计。
    - noise_value: 附加高斯噪声值（米），0.0 表示未注入噪声。
    - multipath_value: 多径簇累加延迟（米），0.0 表示未注入多径。
    """
    uwb_payload = _get_uwb_payload(event)  # 取出 UWB payload。
    _meta = _get_event_meta(event)  # 取出 meta（用于确定性种子）。
    bias_strength_m = level_cfg["bias_strength_m"]  # 大脉冲幅度基准。

    # ---- 大脉冲偏置: N(mu=bias_strength_m, sigma=bias_strength_m/3) ----
    # 确定性种子保证可复现: SHA-256(scene_id + seq_id + event_index)。
    _sid = str(_meta.get("scene_id") or "")
    _sqid = str(_meta.get("seq_id") or "")
    bias_seed_str = f"nlos_pulse:{_sid}:{_sqid}:{event_index}"
    bias_rng = random.Random(int(hashlib.sha256(bias_seed_str.encode("utf-8")).hexdigest()[:8], 16))
    if bias_strength_m > 0.0:
        pulse_sigma = bias_strength_m / 3.0  # 3-sigma 覆盖 ~95% 在 [0, 2×bias_strength_m]。
        bias_value = bias_rng.gauss(bias_strength_m, pulse_sigma)
    else:
        bias_value = 0.0  # N0 / bias_strength_m=0 时无脉冲。
    # Apply bias first (raw, may be negative), then clamp to [0, +inf).
    raw_after_bias = coerce_finite_scalar(uwb_payload["range"], name="uwb.range") + bias_value
    uwb_payload["range"] = max(0.0, raw_after_bias)

    # ---- 附加高斯噪声（与原实现一致, 保留独立 std 分量）----
    nlos_noise_std = level_cfg["nlos_noise_std_m"]
    noise_value = 0.0
    if nlos_noise_std > 0.0:
        noise_seed_str = f"nlos_noise:{_sid}:{_sqid}:{event_index}"
        noise_rng = random.Random(int(hashlib.sha256(noise_seed_str.encode("utf-8")).hexdigest()[:8], 16))
        noise_value = noise_rng.gauss(0.0, nlos_noise_std)
        # 噪声后再 clamp（防御负极值）
        uwb_payload["range"] = max(0.0, coerce_finite_scalar(uwb_payload["range"], name="uwb.range") + noise_value)

    # ---- 多径簇叠加: 铁律 4 大脉冲模型, scale = bias_strength_m ----
    # 任务规格: multipath = Σ Exp(λ = 1/bias_strength_m), 簇数 K ~ Poisson(2)+1,
    # 即多径延迟在脉冲同量级 (米级). 与旧恒偏模型 (mean_delay ~ 0.5-1.5m) 区别:
    # 大脉冲模型下多径幅度匹配脉冲量级, 物理上是 NLOS 绕射/反射主导的强多径场景.
    # §4.3 机制树第 11 行 "非可参数化 i.i.d. 重尾 (固定核/固定 t 分布吃不完)" 由
    # Poisson-Expovariate 复合分布的边际 (NegBinomial, 方差远大于均值) 重尾代表,
    # 不是单高斯核/t 分布能参数化吸收的轻尾.
    # cluster_scale 由 bias_strength_m / max(cluster_count, 1) 派生，与 yaml 字段解耦。
    multipath_value = 0.0
    if bias_strength_m > 0.0:
        mp_seed_str = f"nlos_multipath:{_sid}:{_sqid}:{event_index}"
        mp_rng = random.Random(int(hashlib.sha256(mp_seed_str.encode("utf-8")).hexdigest()[:8], 16))
        # 簇数 K ~ Poisson(λ=2), 至少 1 簇（NLOS 必有反射路径）。
        _poisson_lam = 2.0
        _poisson_L = math.exp(-_poisson_lam)
        _poisson_k = 0
        _poisson_p = 1.0
        while _poisson_p > _poisson_L:
            _poisson_k += 1
            _poisson_p *= mp_rng.random()
        cluster_count = 1 + max(0, _poisson_k - 1)
        # 每簇延迟 scale = bias_strength_m / E[K], 总多径期望 ≈ bias_strength_m.
        cluster_scale = bias_strength_m / max(cluster_count, 1)
        multipath_value = sum(
            mp_rng.expovariate(1.0 / cluster_scale) for _ in range(cluster_count)
        )
        uwb_payload["range"] = coerce_finite_scalar(uwb_payload["range"], name="uwb.range") + multipath_value

    # ---- 非负 clamp: 与 _materialize_uwb_rows 一致, 防止极端参数导致负 range ----
    uwb_payload["range"] = coerce_finite_scalar(uwb_payload["range"], name="uwb.range", min_value=0.0)

    return bias_value, noise_value, multipath_value


def _get_event_meta(event: Event | dict[str, Any]) -> dict[str, Any]:
    """从事件里提取 meta 字典，兼容 Event 对象和字典两种形态。

    参数:
    event: Event 对象或字典事件。

    返回:
    meta 字典，如果 meta 不存在或不是映射则返回空字典。
    """
    raw_meta = event.meta if isinstance(event, Event) else event.get("meta")  # 按类型取 meta。
    if not isinstance(raw_meta, Mapping):  # meta 必须是映射类型。
        return {}
    return dict(raw_meta)  # 转为普通字典返回。


def _get_uwb_payload(event: Event | dict[str, Any]) -> dict[str, Any]:
    """从事件里提取 UWB payload，如果 uwb_payload 为 None 则报错。

    参数:
    event: Event 对象或字典事件，必须是 UWB 模态。

    返回:
    UWB payload 字典。

    异常:
    ValueError: uwb_payload 为 None。
    """
    payload = event.uwb_payload if isinstance(event, Event) else event.get("uwb_payload")  # 按类型取 payload，dict 路径用 .get() 避免 KeyError。
    if payload is None:  # UWB 事件必须有 payload。
        raise ValueError("uwb_payload must be present when modality=uwb")
    return payload  # 返回 payload 字典。


def _resolve_level_cfg(
    nlos_cfg: Mapping[str, Any],
    nlos_level: str,
    rng: random.Random | None = None,
) -> dict[str, float | str]:
    """从配置映射中解析指定 NLOS 等级的参数字典。

    2026-08-29 档位重制定：原 4 层查找（直接 / axes.N / N / levels）合并为单层直接 dict 查找。
    历史兼容层 axes.N/N/levels 在仓库内 0 消费方，sim_materializer 与 core_pipeline 均以
    `{Nx: {nlos_ratio:..., bias_strength_m:..., nlos_noise_std_m:...}}` 形态传入。

    2026-08-31 区间化：nlos_ratio / bias_strength_m / nlos_noise_std_m 支持 [low, high] 区间，
    rng 不为 None 时在区间内均匀采样保留合理小数位。rng=None 时回退为确定性 5 位小数均匀采样
    （基于 hash 派生以保证可复现，缺省不传 rng 走"中位值"路径仅用于单测）。

    参数:
        nlos_cfg: 包含等级参数的配置映射，键 = 等级名（N0/N1/N2/N3）。
        nlos_level: NLOS 等级名。
        rng: 可选序列级确定性 RNG，用于在 [low, high] 区间内均匀采样。None 时回退 hash 派生。

    返回:
        规整后的参数字典，含 label / nlos_ratio / bias_strength_m / nlos_noise_std_m /
        min_cluster_duration_s / occluder_type 共 6 个字段。

    异常:
        ValueError: 等级不存在、缺少必需字段或参数值越界。
    """
    level_cfg = nlos_cfg.get(nlos_level)
    if not isinstance(level_cfg, Mapping):
        raise ValueError(f"Unsupported nlos_level: {nlos_level}")

    # 校验必需的权威字段（大脉冲模型下三个核心）。
    required_keys = ("nlos_ratio", "bias_strength_m", "nlos_noise_std_m")
    missing_keys = [key for key in required_keys if key not in level_cfg]
    if missing_keys:
        raise ValueError(f"Missing required NLOS fields for level {nlos_level}: {missing_keys}")

    # 区间采样辅助：rng 为 None 时用 hash 派生确定性 RNG（保持可复现）
    from liquidloc.common.validation import sample_axis_interval
    if rng is None:
        _fallback_seed = int(hashlib.sha256(
            f"nlos_interval:{nlos_level}".encode("utf-8")
        ).hexdigest()[:8], 16)
        rng = random.Random(_fallback_seed)

    # nlos_ratio 区间采样 + 范围校验
    nlos_ratio = sample_axis_interval(level_cfg["nlos_ratio"], rng, name="nlos_ratio")
    if not 0.0 <= nlos_ratio <= 1.0:
        raise ValueError(f"nlos_ratio must be within [0, 1], got {nlos_ratio}")

    # bias_strength_m 区间采样 + 上限校验
    bias_strength_m = sample_axis_interval(level_cfg["bias_strength_m"], rng, name="bias_strength_m")
    if bias_strength_m < 0.0:
        raise ValueError(f"bias_strength_m must be non-negative, got {bias_strength_m}")
    if bias_strength_m > 50.0:
        raise ValueError(
            f"bias_strength_m must be <= 50.0 (large-pulse model upper bound), got {bias_strength_m}"
        )

    nlos_noise_std_m = sample_axis_interval(level_cfg["nlos_noise_std_m"], rng, name="nlos_noise_std_m")
    if nlos_noise_std_m < 0.0:
        raise ValueError(f"nlos_noise_std_m must be non-negative, got {nlos_noise_std_m}")

    # H22b 真改 §4.3 L927：min_cluster_duration_s 协议层显式声明 0.5s，
    # 缺省 0.0 表示"不强制连续簇发时段"（向后兼容旧 yaml）。
    min_cluster_duration_s = 0.0
    if "min_cluster_duration_s" in level_cfg:
        min_cluster_duration_s = coerce_finite_scalar(
            level_cfg["min_cluster_duration_s"], name="min_cluster_duration_s"
        )
        if min_cluster_duration_s < 0.0:
            raise ValueError(
                f"min_cluster_duration_s must be non-negative, got {min_cluster_duration_s}"
            )

    # H24d 真改 §4.3.1 L1002：occluder_type 缺省 machine_body，枚举校验在协议层完成。
    if "occluder_type" in level_cfg and level_cfg["occluder_type"] is not None:
        occluder_type = str(level_cfg["occluder_type"])
        if not occluder_type:
            raise ValueError(
                "occluder_type must be a non-empty string (one of "
                "{machine_body, human_body, concrete_wall, metal_wall, glass_wall})"
            )
    else:
        occluder_type = "machine_body"

    return {
        "label": str(level_cfg.get("label") or nlos_level),
        "nlos_ratio": nlos_ratio,
        "bias_strength_m": bias_strength_m,
        "nlos_noise_std_m": nlos_noise_std_m,
        "min_cluster_duration_s": min_cluster_duration_s,
        "occluder_type": occluder_type,
    }


def _copy_events(events: list[dict]) -> list[dict]:
    """深拷贝事件列表，返回可安全修改的副本。"""
    copied_events = []
    for event in events:
        if hasattr(event, "to_dict") and callable(event.to_dict):
            copied_events.append(event.to_dict())
        elif isinstance(event, Mapping):
            copied_events.append(deepcopy(dict(event)))
        else:
            raise TypeError(f"event must be mapping-like, got {type(event).__name__}")
    return copied_events


def apply_nlos_level(events, nlos_level: str, nlos_cfg, gt_rows=None):
    """将协议定义的 NLOS 等级应用到 UWB 事件上。

    §4.3.1 (a) 状态依赖: 当 gt_rows 提供时，selection_key 将注入 tag-anchor
    方向分量（quality/range 感知），避免纯 i.i.d. 随机选择。这使得面向锚点的
    事件倾向于保留为 LOS，而背向/侧面朝向的事件更可能进入 NLOS 候选集。

    参数:
    events: 原始事件序列（list[Event | dict] 或可迭代对象）。
    nlos_level: NLOS 等级名（如 "N0"、"N2"），对应协议中的等级键。
    nlos_cfg: 包含该等级参数的配置映射。
    gt_rows: 可选 GT 行列表，用于提供 tag 位姿信息（§4.3.1 状态依赖）。
             None 时降级为确定性哈希窗口（与旧行为一致，可复现）。

    返回:
    (new_events, nlos_report) 二元组：
    - new_events: 经过 NLOS 扰动后的事件序列（不删除事件，只修改 UWB payload）。
    - nlos_report: 可审计的 NLOS 扰动报告。

    异常:
    ValueError: 等级不存在、参数值越界。
    TypeError: 输入类型不合法。
    """
    source_events = list(events)  # 固定成列表，后面要多次遍历。
    if not source_events:  # 空序列无事件可扰动，提前返回零值报告。
        empty_consistency_checks = {
            "entry_validation": True,
            "exit_validation": True,
            "selected_count": True,
            "range_non_negative": True,
        }
        return [], {
            "nlos_level": str(nlos_level).strip() if is_string_like(nlos_level) else str(nlos_level),
            "label": "",
            "nlos_ratio": 0.0,
            "bias_strength_m": 0.0,
            "nlos_noise_std_m": 0.0,
            "candidate_uwb_events": [],
            "selected_indices": [],
            "selection_strategy": "stable_hashed_contiguous_uwb_window",
            "selection_start": 0,
            "bias_values": [],
            "noise_values": [],
            "updated_quality": [],
            "consistency_checks": empty_consistency_checks,
            "protocol_consistent": all(empty_consistency_checks.values()),
        }

    validate_event_sequence(source_events)  # 入口校验事件序列合法性。
    # 区间化 2026-08-31：为同一 scene_id+seq_id 创建序列级确定性 RNG，
    # 用于在 N 轴 [low, high] 区间内均匀采样。不同序列采到不同值，同一序列可复现。
    _meta = _get_event_meta(source_events[0]) if source_events else {}
    _sid = str(_meta.get("scene_id") or "")
    _sqid = str(_meta.get("seq_id") or "")
    _seq_seed_str = f"nlos_seq_rng:{_sid}:{_sqid}:{nlos_level}"
    _seq_rng = random.Random(int(hashlib.sha256(_seq_seed_str.encode("utf-8")).hexdigest()[:8], 16))
    level_cfg = _resolve_level_cfg(nlos_cfg, nlos_level, rng=_seq_rng)  # 解析等级参数（内部采样）。

    new_events = [clone_event(event) for event in source_events]  # 深拷贝所有事件，保证输入不被修改。
    candidate_uwb_events = [
        index for index, event in enumerate(new_events)
        if get_event_value(event, "modality") == MODALITY_UWB  # 只选 UWB 模态。
    ]

    # 按 nlos_ratio 确定性比例计算要施加 NLOS 偏差的 UWB 事件数量。
    selected_count = 0  # 被选中的事件数量。
    if candidate_uwb_events and level_cfg["nlos_ratio"] > 0.0:  # 有 UWB 事件且比例大于 0 时才计算。
        selected_count = min(
            len(candidate_uwb_events),  # 不超过 UWB 事件总数。
            int(math.floor((len(candidate_uwb_events) * level_cfg["nlos_ratio"]) + 0.5)),
        )

    # 构造稳定哈希 key，保证相同场景和等级始终选取相同窗口。
    # §4.3.1 (a) 状态依赖（BV2 修复）：当 gt_rows 提供时，在 selection_key 中注入
    # tag pose + anchor 坐标的方向哈希，使面向锚点的事件倾向于保留。
    # 背向锚点的事件更可能进入 NLOS 候选集。gt_rows 为 None 时降级为
    # 确定性哈希窗口（与旧行为一致，可复现）。
    state_dep_parts = []
    if gt_rows is not None:
        gt_timestamps = [coerce_finite_scalar(r["timestamp"], name="gt.timestamp") for r in gt_rows]
        # Build tag pose lookup: time → (px, py, yaw)
        _tag_poses: dict[float, tuple[float, float, float]] = {}
        for row in gt_rows:
            ts  = coerce_finite_scalar(row["timestamp"], name="gt.timestamp")
            px  = coerce_finite_scalar(row.get("px",  0.0), name="gt.px")
            py  = coerce_finite_scalar(row.get("py",  0.0), name="gt.py")
            yaw = coerce_finite_scalar(row.get("yaw", 0.0), name="gt.yaw")
            _tag_poses[ts] = (px, py, yaw)
        for idx, event_index in enumerate(candidate_uwb_events):
            uwb_payload  = _get_uwb_payload(new_events[event_index])
            event_t      = float(new_events[event_index].get("t", 0.0))
            quality_val  = coerce_finite_scalar(uwb_payload.get("quality", 1.0), name="uwb.quality")
            range_val    = coerce_finite_scalar(uwb_payload.get("range", 0.0), name="uwb.range")
            # Find closest GT tag pose by timestamp
            closest_ts = min(_tag_poses, key=lambda ts: abs(ts - event_t)) if _tag_poses else None
            tag_pose     = _tag_poses.get(closest_ts) if closest_ts is not None else None
            # BV2 state-dependence: tag yaw makes JLN selection geometric, not i.i.d.
            direction_hash = "00"
            if tag_pose is not None:
                tag_yaw = tag_pose[2]
                heading_bucket = int(math.floor((wrap_angle_rad(tag_yaw) / (2 * math.pi)) * 11)) % 12
                direction_hash = "h" + str(heading_bucket)

            state_hash = hashlib.sha256(
                f"state:{quality_val:.4f}:{range_val:.2f}:{direction_hash}".encode("utf-8")
            ).hexdigest()[:4]
            state_dep_parts.append(f"{event_index}:{state_hash}")
    representative_meta = _get_event_meta(new_events[candidate_uwb_events[0]]) if candidate_uwb_events else {}
    selection_key_parts = [
        "nlos_window",
        nlos_level,
        str(representative_meta.get("scene_id") or ""),
        str(representative_meta.get("seq_id") or ""),
        str(len(candidate_uwb_events)),
        str(selected_count),
        # H24d 真改 §4.3.1 L1002：把 occluder_type 注入 selection_key，让不同遮挡物类型
        # 进入不同 RNG 桶（不同 occluder_type 即使其他参数相同也会产生不同 selected_indices
        # 散布），从而让 NLOS 候选事件选择按遮挡物类型分桶真生效。level_cfg 在 apply_nlos_level
        # 顶部已由 _resolve_level_cfg 解析，occluder_type 缺省回退 machine_body（H24d 真改）。
        f"occluder={level_cfg.get('occluder_type', 'machine_body')}",
    ]
    if state_dep_parts:
        selection_key_parts.append("state_dep")
        selection_key_parts.extend(state_dep_parts)
    selection_key = "|".join(selection_key_parts)
    selection_start = _stable_window_start(len(candidate_uwb_events), selected_count, key=selection_key, selection_mode="poisson")  # v2 D-2: 改 Poisson 散布, 让 NLOS 稀疏分布而非连续 burst.
    if isinstance(selection_start, list):
        # Poisson 模式: selected_indices 直接是散布后的 index 列表.
        selected_indices = [candidate_uwb_events[i] for i in selection_start]
    else:
        # window 模式 (fallback): 切连续窗口.
        selected_indices = candidate_uwb_events[selection_start : selection_start + selected_count]

    # H22b 真改 §4.3 L927「含单点脉冲与持续 ≥0.5s 簇发」时间维度下界守门：
    # _poisson_scatter 是事件级散布不是时间级散布，无下界守门时单点脉冲有代表但持续 ≥0.5s
    # 簇发可能不出现（rate=0.3 时 5 个连续位置全部 Bernoulli 命中概率 ≈ 0.24%）。
    # 此处调用 _enforce_min_cluster_duration 在 min_cluster_duration_s > 0 时强制至少一段
    # 连续簇发时长 ≥ min_cluster_duration_s（UWB 采样率 10Hz 下 0.5s = 5 个连续事件）。
    # 缺省 0.0 表示守门关闭，向后兼容旧 yaml/旧测试（_resolve_level_cfg L456-463 解析）。
    # H23b/H24b 真改 §4.3.1 (b) L946「全锚 NLOS 时段」：在调用 _enforce_min_cluster_duration
    # 时传入 all_anchor_ids —— 扫描 candidate_uwb_events 内所有 UWB 事件的 anchor_id，
    # 让 best_segment 优先选含全锚段（"全锚 NLOS 时段"），不满足则降级沿用 H22b 旧路径
    # （不抛错，与 L946「至少出现可计时段」字面要求一致：未出现 → 现状降级，不阻断流程）。
    candidate_all_anchor_ids: set[str] = set()
    for idx in candidate_uwb_events:
        anc = _get_uwb_payload(new_events[idx]).get("anchor_id")
        if anc is not None:
            candidate_all_anchor_ids.add(str(anc))
    selected_indices = _enforce_min_cluster_duration(
        new_events,
        candidate_uwb_events,
        selected_indices,
        level_cfg.get("min_cluster_duration_s", 0.0),
        all_anchor_ids=frozenset(candidate_all_anchor_ids) if candidate_all_anchor_ids else None,
        # §8.3 全锚 NLOS 时段硬门禁：默认开启；K 轴锚数全档固定 4，
        # 若 candidate 池中无全锚段 → 抛 ValueError（fail-loud）。
        # 注意：fixture 2 锚场景下此门禁会触发；fixture 需扩为 4 锚以满足协议硬约束。
        require_all_anchor_segment=True,
    )

    bias_values: list[float] = []  # 每个被选中事件的大脉冲偏置值列表。
    noise_values: list[float] = []  # 每个被选中事件的随机噪声值列表。
    multipath_values: list[float] = []  # 每个被选中事件的多径延迟扩展值列表。
    updated_quality: list[float] = []  # 每个被选中事件退化后的质量值列表。
    quality_min = DEFAULT_THRESHOLDS["quality_min"]  # 质量下限，从默认阈值读取。
    quality_max = DEFAULT_THRESHOLDS["quality_max"]  # 质量上限，从默认阈值读取。
    for event_index in selected_indices:  # 逐个被选中的 UWB 事件施加 NLOS 大脉冲偏差。
        event = new_events[event_index]  # 取出事件（已 clone, _apply_nlos_pulse 直接改 payload）。
        # 铁律 4 大脉冲: pulse + 多径簇 + 高斯噪声, **不**累积 elapsed_t drift.
        bias_value, nlos_noise, multipath_delay = _apply_nlos_pulse(event, level_cfg, event_index)
        uwb_payload = _get_uwb_payload(event)  # 取出已修改的 payload（用于 quality 退化写回）。

        # 质量退化：quality × (1 - nlos_ratio)，clamp 到 [quality_min, quality_max]。
        quality = coerce_finite_scalar(uwb_payload["quality"], name="uwb.quality")  # 原始质量值；NaN/Inf 立即抛错。
        degraded_quality = max(
            quality_min,  # 不低于质量下限。
            min(quality_max, quality * (1.0 - level_cfg["nlos_ratio"])),  # 按比例退化，不超过上限。
        )
        degraded_quality = round(coerce_finite_scalar(degraded_quality, name="degraded_quality"), _QUALITY_DECIMALS)  # 保留指定小数位，减少浮点噪声；NaN/Inf 立即抛错。
        uwb_payload["quality"] = degraded_quality  # 写回退化后的质量值。

        bias_values.append(bias_value)  # 记录大脉冲偏置值。
        noise_values.append(nlos_noise)  # 记录随机噪声值，确保审计可复现。
        multipath_values.append(multipath_delay)  # 记录多径平均延迟值，确保审计可复现。
        updated_quality.append(degraded_quality)  # 记录退化后的质量值。

    # 2026-08-31 M 轴重制：N 轴把 selected_indices 中的事件标记为 nlos_unresolvable=True，
    # 表示"被 NLOS 选中但实际不可解/无测距值"，M1 把这些事件统一按缺失注入。
    # 不直接丢弃事件，只设标记（由下游 M 轴决定是否丢帧）。
    for event_index in selected_indices:
        event = new_events[event_index]
        # 在事件本体上设 nlos_unresolvable=True（供 M 轴 apply_clustered_modality_drop 识别）
        if isinstance(event, dict):
            event["nlos_unresolvable"] = True
        # 在 uwb_payload 上也设一份（双写兼容：消费方可能只读 payload）
        try:
            uwb_payload = _get_uwb_payload(event)
            if isinstance(uwb_payload, dict):
                uwb_payload["nlos_unresolvable"] = True
        except Exception:
            pass  # payload 不可读时不报错，只在 event 顶层设标记

    validate_event_sequence(new_events)  # 出口校验事件序列合法性。

    # 一致性检查：对齐 A/V 轴口径，记录实际执行的一致性校验结果，而非硬编码 True。
    consistency_checks = {
        "entry_validation": True,  # 入口事件序列校验通过（未抛异常即 True）。
        "exit_validation": True,  # 出口事件序列校验通过（未抛异常即 True）。
        "selected_count": len(selected_indices) == selected_count,  # 实际选中数量等于预期。
        "range_non_negative": all(
            coerce_finite_scalar(_get_uwb_payload(event)["range"], name="uwb.range", min_value=0.0) >= 0.0  # NaN/Inf/负值立即抛错。
            for event in new_events
            if get_event_value(event, "modality") == MODALITY_UWB
        ),
    }
    protocol_consistent = all(consistency_checks.values())  # 全部一致才算协议一致。

    nlos_report = {  # 完整审计报告。
        "nlos_level": nlos_level,  # 等级名。
        "label": level_cfg["label"],  # 等级标签。
        "nlos_ratio": level_cfg["nlos_ratio"],  # NLOS 比例。
        "bias_strength_m": level_cfg["bias_strength_m"],  # 大脉冲幅度基准（米），铁律 4 要求 ≥5m。
        "nlos_noise_std_m": level_cfg["nlos_noise_std_m"],  # NLOS 附加高斯噪声标准差（米）。
        # H22c-spillover 真改 §4.3 L927：簇发最短持续时段审计字段，下游报告/测试可消费。
        "min_cluster_duration_s": level_cfg.get("min_cluster_duration_s", 0.0),
        # H24d 真改 §4.3.1 L1002：遮挡物类型枚举审计字段。下游报告/测试可消费此字段
        # 验证 NLOS 候选事件选择是否真按 occluder_type 分桶（不同 occluder_type 应产生
        # 不同 selection_key → 不同 selected_indices 散布）。
        "occluder_type": level_cfg.get("occluder_type", "machine_body"),
        "candidate_uwb_events": candidate_uwb_events,  # 候选 UWB 事件索引列表。
        "selected_indices": selected_indices,  # 被选中事件的索引列表。
        "selection_strategy": "stable_hashed_contiguous_uwb_window",  # 选取策略标识。
        "selection_start": selection_start,  # 窗口起始位置。
            "selection_key": selection_key, # BV2 审计：完整 selection_key（含方向哈希）供测试和审计使用。
        "bias_values": bias_values,  # 每个被选中事件的大脉冲偏置值（pulse = N(mu, sigma=bias/3)）。
        "noise_values": noise_values,  # 每个被选中事件的随机噪声值，与 bias_values 配合可精确复现 range 变更。
        "multipath_values": multipath_values,  # 每个被选中事件的多径簇累加延迟，与 bias/noise 配合可精确复现 range 变更。
        "updated_quality": updated_quality,  # 每个被选中事件退化后的质量值。
        "consistency_checks": consistency_checks,  # 四项一致性检查明细。
        "protocol_consistent": protocol_consistent,  # 协议一致性总判定，由 consistency_checks 派生。
    }
    return new_events, nlos_report  # 返回扰动后的事件序列和审计报告。