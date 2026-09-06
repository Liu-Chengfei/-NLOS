"""公共常量定义。

职责：
    集中放跨模块复用的模态名称、事件键、默认输出目录和阈值，
    避免上层代码到处重复定义同一套字面量。

上游依赖：
    - 无（本模块是 common 层最底层的纯常量定义，不依赖其他内部模块）

下游调用者：
    - liquidloc.common.paths         — 复用 DEFAULT_OUTPUT_DIRS 构造输出子目录
    - liquidloc.common.__init__      — 统一导出常量给上层
    - liquidloc.common.prepared_inputs — 复用 normalize_run_mode 做模式归一化
    - liquidloc.protocol.*           — 协议层通过常量做模态校验与键名对齐
    - liquidloc.pipelines.*          — 流水线层通过常量做输出目录与阈值检查
    - liquidloc.estimators.*         — 估计器层通过 VIO_MEASUREMENT_ITEMS 等常量做字段校验
    - configs/base/*.yaml            — 配置文件中的阈值与常量保持语义一致（非代码调用关系）

核心变量：
    - MODALITY_IMU / MODALITY_UWB / MODALITY_VIO / MODALITY_FLOW / MODALITY_TOF  — 五种模态的字符串标识
    - ALLOWED_MODALITIES             — 允许的模态元组，用于校验
    - PRIMARY_EVENT_KEYS             — 主事件层必须保留的键名
    - META_KEYS / OPTIONAL_META_KEYS — meta 中必须/可选的键名
    - PAYLOAD_KEYS                   — 各模态对应的载荷键名映射
    - DEFAULT_OUTPUT_DIRS            — 默认要创建的输出目录名列表
    - DEFAULT_REQUIRED_OUTPUT_FILES  — 最小运行时必须能看到的产物路径
    - DEFAULT_THRESHOLDS             — 全局默认阈值字典
    - PRIMARY_KEYS                   — 按职责分组的主键注册表
    - ASYNC_GAP_FULL_SCALE_S 等      — 信号/阈值常量（详见代码注释）
    - BRIDGE_RISK_MIN/MAX/SCALING_MAX — 桥接层阈值常量（单源真相）
    - ALLOWED_RUN_MODES / RUN_MODE_ALIASES — 运行模式常量（单源真相）
    - DEVICE_AUTO/CPU/CUDA/ALLOWED_TRAIN_DEVICES — 训练设备请求常量（单源真相）
    - VIO_MEASUREMENT_ITEMS          — VIO 测量项常量（单源真相）
"""

from __future__ import annotations  # 保持类型注解写法一致。

import math  # 用于风险先验 logit 计算。

MODALITY_IMU = "imu"  # IMU 模态字符串常量。
MODALITY_UWB = "uwb"  # UWB 模态字符串常量。
MODALITY_VIO = "vio"  # VIO 模态字符串常量。
MODALITY_FLOW = "flow"  # 光流/流量类模态字符串常量（UTIL 数据集）。
MODALITY_TOF = "tof"  # ToF 测距类模态字符串常量（UTIL 数据集）。
ALLOWED_MODALITIES = (MODALITY_IMU, MODALITY_UWB, MODALITY_VIO, MODALITY_FLOW, MODALITY_TOF)  # 允许的模态集合。

PRIMARY_EVENT_KEYS = ("t", "dt", "modality", "meta")  # 主事件层必须保留的键。

# 案例引用核心字段名（单源真相，D9 配置表面漂移根因修复）。
# analysis/case_selector.py _extract_case_ref（权威定义）、analysis/summary_builder.py、
# plotting/plot_cases.py 均从此处引用，禁止在各模块本地重复定义字面量。
# 风险：若任一处漂移，会导致 case_ref 提取逻辑不一致，破坏 selected_cases 与 manifest 的引用合同。
SCENE_ID_KEY: str = "scene_id"  # 场景轴字段名（meta 必选键 + case_ref 联合引用组成部分）。
SEQ_ID_KEY: str = "seq_id"  # 序列轴字段名（meta 必选键 + case_ref 兜底引用）。

META_KEYS = (SCENE_ID_KEY, SEQ_ID_KEY)  # meta 中必须存在的键。source_t 为可选字段，不在必选列表中。
OPTIONAL_META_KEYS = ("source_t",)  # meta 中可选的键，存在时用于溯源和异步对齐。
PAYLOAD_KEYS = {  # 每个模态对应的载荷键名。
    MODALITY_IMU: "imu_payload",  # IMU 载荷键。
    MODALITY_UWB: "uwb_payload",  # UWB 载荷键。
    MODALITY_VIO: "vio_payload",  # VIO 载荷键。
    MODALITY_FLOW: "flow_payload",  # 光流/流量类载荷键（UTIL 数据集）。
    MODALITY_TOF: "tof_payload",  # ToF 测距类载荷键（UTIL 数据集）。
}  # 载荷键映射结束。

DEFAULT_OUTPUT_DIRS = (  # 默认要创建或识别的输出目录。
    "checkpoints",  # 模型权重目录。
    "predictions",  # 预测结果目录。
    "metrics",  # 指标结果目录。
    "statistics",  # 统计结果目录。
    "figures",  # 图表结果目录。
    "cases",  # 案例结果目录。
    "logs",  # 日志目录。
    "audits",  # 审计目录。
)  # 默认输出目录列表结束。

DEFAULT_REQUIRED_OUTPUT_FILES = (  # 最小运行时必须能看到的产物路径。
    "predictions/mini_predictions.json",  # 预测包最小检查文件。
    "metrics/mini_metrics.csv",  # 指标最小检查文件。
    "audits/protocol_snapshot.json",  # 协议快照检查文件。
)  # 必需输出文件列表结束。

# 五轴档位协议下，public_benchmark 合同与 default 合同目录差异：public_benchmark 仅要求
# public_benchmarks 单一目录（嵌套结构，不平铺其他子目录）。
PUBLIC_BENCHMARK_OUTPUT_DIRS = ("public_benchmarks",)  # 公开基准输出合同要求的目录集合。
PUBLIC_BENCHMARK_REQUIRED_OUTPUT_FILES = (  # 公开基准合同最小必须看到的产物路径。
    "public_benchmarks/manifest.json",  # 公开基准 manifest 清单。
)  # 公开基准要求文件列表结束。

# 真值文件名字常量（单源真相，D9 配置表面漂移根因修复）。
# pipelines/eval_pipeline.py、pipelines/core_pipeline.py、pipelines/train_pipeline.py、
# dataio/*_reader.py、dataio/manifests/dataset_checks.py、common/prepared_inputs.py 均从此处引用，
# 禁止在各模块本地重复定义 'gt.json' 字面量。
GT_FILE_NAME: str = "gt.json"

DEFAULT_THRESHOLDS = {  # 全局默认阈值表（纯算法/协议级阈值）。
    "time_tolerance": 1e-6,  # 时间对齐容差（浮点比较容差，用于判断两时间戳数值相等，非物理时钟同步精度）。
    "quality_min": 0.0,  # 质量下限。
    "quality_max": 1.0,  # 质量上限。
    "vio_displacement_min": -10.0,  # VIO 位移增量下界（米），dx/dy 防溢出阈值（典型室内 VIO 单帧 <0.5m）。
    "vio_displacement_max": 10.0,  # VIO 位移增量上界（米），dx/dy 防溢出阈值（典型室内 VIO 单帧 <0.5m）。
    "flow_displacement_min": -10.0,  # 光流位移增量下界（米），dx/dy 防溢出阈值。
    "flow_displacement_max": 10.0,  # 光流位移增量上界（米），dx/dy 防溢出阈值。
    "imu_accel_min": -150.0,  # IMU 加速度下界（m/s²），典型消费级 IMU 量程 ±50 m/s²。
    "imu_accel_max": 150.0,  # IMU 加速度上界（m/s²）。
    "imu_gyro_min": -200.0, # IMU 角速度下界（rad/s）。sim 数据覆盖到 ~181.6（含 sim_turn_02 / sim_long_50m_01 系列的尖峰），故放宽到 ±200 保留 10% 余量。
    "imu_gyro_max": 200.0, # IMU 角速度上界（rad/s）。sim 数据覆盖到 ~181.6（含 sim_turn_02 / sim_long_50m_01 系列的尖峰），故放宽到 ±200 保留 10% 余量。
}  # 阈值表结束。桥接层业务阈值已迁移至 liquidloc.protocol.bridge_thresholds.BRIDGE_THRESHOLDS。

ASYNC_GAP_FULL_SCALE_S = 0.30  # 异步时间间隔归一化满量程（秒），训练侧与推理侧共享。
VIO_REF_POSE_STALE_SECONDS = 0.5  # VIO 参考位姿过时阈值（秒）。前提指导 §1.5 可观积累。
# 真实语义：当上一参考位姿距今超过此阈值，认为中间可能有 VIO 事件被 blackout
# 移除或门控跳过，导致相邻帧增量语义不匹配（参考帧 ≠ 相邻帧），此时重置参考位姿
# 并跳过本次更新（ekf_core.py:1044 / fgo_core.py:970,1949 / robust_ekf_core.py:558
# 三处同口径拒识）。
# 此阈值是「参考位姿 persistence 时长」而非「N 帧间隔」——sim 端实际 VIO 帧率
# 30Hz（默认）或 60Hz（覆盖）（见 sim_materializer.py:206-207 / scene_axis_protocol.yaml
# vio_hz），0.5s 在 30Hz 下等价于 15 帧间隔、60Hz 下等价于 30 帧间隔，远超 5 帧。
# 旧版本注释「5 帧 VIO@100ms」数学错误（100ms 帧间隔既非 30Hz 也非 60Hz），已修正。
UWB_INVALID_SIGNAL_FLOOR = 0.25  # 无效 UWB 附加风险下限。
VIO_LOW_FEATURES_SIGNAL_FLOOR = 0.20  # VIO 低特征数风险下限。
VIO_HIGH_REPROJ_ERR_SIGNAL_FLOOR = 0.15  # VIO 高重投影误差风险下限。
VIO_TRACKED_FEATURES_FLOOR = 30  # VIO 跟踪特征数阈值，低于此值视为低质量。
VIO_HIGH_REPROJ_ERR_THRESHOLD = 1.0  # VIO 重投影误差阈值（像素），>=此值视为高误差。
VIO_TRACKED_FEATURES_FULL_SCALE = 200.0  # VIO 特征数归一化满量程（预留），与 scene_axis_protocol.yaml V0 上界对齐。当前无消费方，待特征数归一化链路完善后启用。
VIO_TRACKED_FEATURES_SAFE_FLOOR = 100.0  # VIO 特征数安全下界，与 scene_axis_protocol.yaml V0 下界对齐。特征数达到此值时 VIO 工作正常，feature_floor_risk 为 0。
VIO_REPROJ_ERR_NORM_FLOOR = 0.5  # VIO 重投影误差归一化下限（像素）。
VIO_REPROJ_ERR_FULL_SCALE = 4.0  # VIO 重投影误差归一化满量程（像素）。
QUALITY_FLOOR_EPSILON = 1e-9  # 质量门槛比较浮点容差。
RISK_PARTIAL_DAMPING_COEFF = 0.5  # 安全模式下 bias 衰减系数：damping = 1.0 - coeff * risk。

# V 轴漂移惩罚饱和阈值（单源真相，scenarios.visual_levels._degrade_quality 与 protocol.scene_axis_protocol 共享）。
# 此值是 drift_penalty 的满惩罚门槛：drift_bias_m >= 此值时 drift_penalty=1.0（满惩罚）。
# protocol.scene_axis_protocol.py L252-253 用此值作为 drift_bias_m 的上限校验（错误信息 "degrade_quality saturation limit"）。
# 修改此值必须同步修改 protocol 层校验（需用户授权，protocol 层为冻结真相）。
DRIFT_PENALTY_SATURATION_M: float = 1.0  # 漂移惩罚饱和阈值（米）。

# 桥接层阈值常量（单源真相，protocol.bridge_thresholds 从此处引用）。
# 这些阈值是 risk/scaling 范围校验的底层真相，common 层和 protocol 层共享。
BRIDGE_RISK_MIN: float = 0.0  # 风险最小值。
BRIDGE_RISK_MAX: float = 1.05  # 风险最大值。v3 提高至 1.05 以容纳 risk_hard_skip_threshold=1.05, 完全解除硬跳过 (诊断修复 2026-08-02). 实际网络 risk 输出仍受 normalize_risk 限制在 [0, 1.0].
BRIDGE_SCALING_MIN: float = 1.0  # 与 bridge_thresholds.py scaling_min 保持单源一致。v3 回退到 1.0：v2 放宽至 0.5 的 soft-mask 在 e9 场景下让 Liquid 不当降权好测量，不利于高 NLOS + 异步场景。
BRIDGE_SCALING_MAX: float = 50.0  # 缩放最大值（方差倍数，防止极端值导致数值溢出）。
BRIDGE_BIAS_MAX: float = 10.0  # 模型输出偏置上界（与 model_factory torch.clamp 和 fusion_runner 裁剪对齐）。
BRIDGE_BIAS_ABSOLUTE_MAX: float = 5.0  # 施加偏置绝对值上限（米），放宽至 5.0m 以释放 Liquid 调节空间。
# ceiling 与 scaling_max 关系：ceiling = scaling_max² × (1+risk_max) = 50² × 2 = 5000。
# D11-R2 修复：从 800.0 提高到 5000.0，释放 scaling_max 在 risk>0 时的调节空间
# （原 800.0 使 scaling_max=50 在 risk>0 时成为死代码，因为 50²×2=5000>800 会被截断）。
BRIDGE_NOISE_MULTIPLIER_CEILING: float = 5000.0  # 噪声倍数硬上限（scaling^2*(1+risk) 封顶值）。

# 非当前模态 scaling 上界（§12.3-C2a 三网同一写入口要求）。
# 唯一写入口：factories/model_factory.py apply_liquid_modality_output_contract。
# 历史：第十八轮穷举自审发现此值曾硬编码在 model_factory.py:240，违反 §12.3-C2a
# 三网同一写入口精神（孤立的硬编码常量未注册到 BRIDGE_THRESHOLDS 单源）。
# 现改为单源真相，model_factory.py 从 BRIDGE_THRESHOLDS 读，未来若 LSTM/Transformer
# 写本体 scaling clamp 必须从此常量再引用，禁止各自硬编码。
# 取值 2.5：v3 改造放宽非当前模态 scaling 上界从 1.0 → 2.5，让 4 头在 NLOS 场景下可以
# 把非当前模态的 R 矩阵放大（noise_multiplier = scaling^2*(1+risk)），允许 EKF 对非当前
# 模态观测做更强降权。仍受 scaling_max=50 顶层封顶。
BRIDGE_NON_CURRENT_SCALING_CEILING: float = 2.5  # 非当前模态 scaling 上界（单源真相）。

# 风险先验常量（单源真相，LSTM 和 Liquid 共享，保证公平性）。
# §7.4 修复 (对齐 docs/trainer.md §10 L871 [0.4, 0.6] 健康区间):
# RISK_PRIOR_PROB=0.10 使 sigmoid(logit)≈0.100 偏离 [0.4,0.6]；
# 改为 0.5 使 sigmoid(log(0.5/0.5))=sigmoid(0)=0.5 精确落入区间中点。
# LSTM 与 Liquid 同步使用此常量 (LSTM network.py L502 / Liquid model_factory.py L577)，
# 单源修改即双端生效，公平性保持不变。
RISK_PRIOR_PROB: float = 0.50  # 默认风险先验概率（50%），LSTM 与 Liquid 共享；sigmoid 映射到 0.5 落入健康区间中点。
RISK_PRIOR_LOGIT: float = math.log(RISK_PRIOR_PROB / (1.0 - RISK_PRIOR_PROB))  # 默认风险先验偏置（logit），等于 0.0。

# 运行模式常量（单源真相，protocol.experiment_gates 从此处引用）。
ALLOWED_RUN_MODES: frozenset[str] = frozenset({'quick', 'full'})  # 允许的运行模式，只有 quick（冒烟）和 full（完整）两种。
RUN_MODE_ALIASES: dict[str, str] = {  # 运行模式别名映射，将常见缩写映射到标准模式名。
    "q": "quick",
    "quick": "quick",
    "f": "full",
    "full": "full",
}

# 模型名字常量（单源真相，D9 配置表面漂移根因修复）。
# factories/model_factory.py _SUPPORTED、create_model 路由、
# pipelines/core_pipeline.py、pipelines/train_pipeline.py、trainers 均从此处引用，
# 禁止在各模块本地重复定义相同字面量。
MODEL_NAME_LSTM: str = "lstm_ekf"  # LSTM-EKF 基线模型名字。
MODEL_NAME_LIQUID: str = "liquid_ekf"  # Liquid-EKF 增强模型名字。
MODEL_NAME_TRANSFORMER: str = "transformer_ekf"  # Transformer-EKF 模型名字（§10.2 第 2 行 + §10.4 序列截断对等口径，仓库尚未实现模型本体，但 _NEURAL_METHODS / parity watchdog / model_factory 前瞻守卫已就位）。

# 设备请求常量（单源真相，D9 配置表面漂移根因修复）。
# pipelines/train_pipeline.py、models/{lstm,liquid}/trainer.py 与 factories/model_factory.py
# 均从此处引用，禁止在各模块本地重复定义 "auto"/"cpu"/"cuda" 字面量。
DEVICE_AUTO: str = "auto"  # 训练设备请求：自动选择（由 allow_auto_cuda 与 CUDA 可用性决定）。
DEVICE_CPU: str = "cpu"  # 训练/推理设备：CPU。
DEVICE_CUDA: str = "cuda"  # 训练/推理设备：CUDA。
DEVICE_CUDA_PREFIX: str = "cuda:"  # CUDA 设备索引前缀（如 "cuda:0"），trainer/model_factory 用于解析指定 GPU。
ALLOWED_TRAIN_DEVICES: frozenset[str] = frozenset({DEVICE_AUTO, DEVICE_CPU, DEVICE_CUDA})  # 训练管线允许的设备请求集合（不含 cuda:N，cuda:N 在 trainer 层解析）。

# 估计器名字常量（单源真相，D9 配置表面漂移根因修复）。
# factories/estimator_factory.py _SUPPORTED、_validate_estimator_cfg 专属校验分支、
# create_estimator 路由均从此处引用，禁止在各模块本地重复定义相同字面量。
ESTIMATOR_NAME_EKF: str = "ekf"  # 标准 EKF 估计器名字。
ESTIMATOR_NAME_ROBUST_EKF: str = "robust_ekf"  # 鲁棒 EKF 估计器名字。
ESTIMATOR_NAME_FGO: str = "fgo"  # 滑窗因子图优化估计器名字。
ESTIMATOR_NAME_SGPR: str = "sgpr"  # 纯 SGPR 估计器名字（§16.1 封闭对手集，当前无真实本体，仅占位）。

# 数据集名字常量（单源真相，D9 配置表面漂移根因修复）。
# pipelines/prepare_pipeline.py dataset_name 合法集校验与读取器分支、
# dataio/readers/sim_reader.py read_report 写入端、pipelines/train_pipeline.py、
# pipelines/public_benchmark_pipeline.py、scenarios/scene_sampler.py 均从此处引用，
# 禁止在各模块本地重复定义相同字面量。
DATASET_NAME_SIM: str = "sim"  # 仿真数据集名字。
DATASET_NAME_MILUV: str = "miluv"  # MILUV 数据集名字。
DATASET_NAME_NTU_VIRAL: str = "ntu_viral"  # NTU VIRAL 数据集名字。
DATASET_NAME_UTIL: str = "util"  # UTIL 数据集名字。

# MILUV 官方锚点元数据来源标识（单源真相，D9 配置表面漂移根因修复）。
# dataio/readers/miluv_reader.py 写入端（_discover_official_anchor_layout_candidate 返回值、
# _assess_teacher_projection_audit 投影路径校验）、pipelines/miluv_pipeline.py
# _resolve_sequence_anchor_layout 来源校验、pipelines/train_pipeline.py teacher 投影来源校验
# 均从此处引用，禁止在各模块本地重复定义相同字面量。
# 风险：若任一处漂移，会导致官方 3D 元数据来源比较静默失败，anchor_layout 被静默丢弃，
# 进而破坏 MILUV 公开基准（AGENTS.md §6）的锚点几何合同。
MILUV_OFFICIAL_ANCHOR_METADATA_SOURCE: str = "miluv_official_experiments_csv+anchors_yaml"

# VIO 测量项常量：与 protocol/task_contract 中冻结的 VIO measurement_items 一致。
# 此常量是 common 层唯一真相源，protocol 层从此处消费，避免 common 反向依赖 protocol。
VIO_MEASUREMENT_ITEMS = ("dx", "dy", "dyaw")

# VIO 共享噪声方案键名（单源真相，D9 配置表面漂移根因修复）。
# factories/estimator_factory.py 与 estimators/{fgo_core,robust_ekf_core,vision_update_step}
# 均从此处引用，禁止在各模块本地重复定义 ("pos", "yaw") 字面量。
# shared 方案：位置 dx/dy 共享同一噪声量级 pos，航向独立用 yaw。
VIO_SHARED_NOISE_KEYS = ("pos", "yaw")

# VIO 噪声键方案名称常量（单源真相，D9 配置表面漂移根因修复）。
# estimator_factory._resolve_vio_noise_key_scheme 返回值与调用方比较均从此处引用。
VIO_NOISE_SCHEME_SHARED = "shared"  # 共享方案：pos/yaw 两键。
VIO_NOISE_SCHEME_PER_AXIS = "per_axis"  # 逐轴方案：dx/dy/dyaw 三键。

# 可靠性状态常量（单源真相，D9 配置表面漂移根因修复）。
# metrics/reliability_metrics.py compute_reliability_metrics 与 metrics/metric_runner.py
# _build_support_report 均从此处引用，禁止在各模块本地重复定义 "ok"/"insufficient_overlap" 字面量。
# 风险：若任一处漂移，会导致 reliability_status 字段值不一致，破坏 metric_schema 与下游 analysis 的状态判断。
HARD_REJECT_REASON_MISSING_GT: str = "missing_gt" # 硬拒绝：真值缺失，无法对齐。
RELIABILITY_STATUS_OK: str = "ok"  # 可靠性状态：支持充分且可以信任该结果。
RELIABILITY_STATUS_INSUFFICIENT: str = "insufficient_overlap"  # 可靠性状态：对齐长度太短或有效样本不足。

# 评估对比状态常量（单源真相，D9 配置表面漂移根因修复）。
# analysis/statistics_runner.py build_smoke_only_statistics_payload 与
# pipelines/eval_pipeline.py audit_payload 的 comparison_status 字段均从此处引用，
# 禁止在各模块本地重复定义 "smoke_only_self_ground_truth"/"ground_truth_backed" 字面量。
# 风险：若任一处漂移，会导致 comparison_status 字段值不一致，破坏下游审计与统计载荷的对比资格判断，
# 进而误判冒烟自评与真值背书两种语义（影响 protocol/experiment_gates require_ground_truth_unless_smoke 口径）。
COMPARISON_STATUS_SMOKE_ONLY_SELF_GROUND_TRUTH: str = "smoke_only_self_ground_truth"  # 对比状态：冒烟自评（无外部真值，不可跨方法对比）。
COMPARISON_STATUS_GROUND_TRUTH_BACKED: str = "ground_truth_backed"  # 对比状态：真值背书（有外部真值，可跨方法对比）。
YAW_ZERO_NORTH: float = 0.0 # 航向零点定义：北向为0弧度（前提指导 §2.1 航向零点一致）。全链路角度差用 angle_delta_rad 计算，本常量仅作为参考原点标记。

# 状态项常量：与 protocol/task_contract 中冻结的 state_definition.state_items 一致。
# 此常量是 common 层唯一真相源，protocol 层和 estimator 层从此处消费。
# 前提指导 §1.1 默认主表：平面 2D + 位置/速度/航向 + 加计/陀螺偏置（8 维）。
# 紧耦合扩维：uwb_clock_bias（UWB 钟差，m）+ vio_scale（VIO 尺度因子，无量纲），
# 全体同增并重定义排序身份，进入主状态向量（10 维）。
STATE_ITEMS = ("px", "py", "vx", "vy", "yaw", "bax", "bay", "bg", "uwb_clock_bias", "vio_scale")

# 模型中间量输出头键名（单源真相，D9 配置表面漂移根因修复）。
# inference.py / trainer.py / lstm/network.py / model_factory.py 均从此处引用，
# 禁止在各模块本地重复定义相同元组。顺序固定为 bias → risk → uwb_scaling → vio_scaling，
# 与 ModelIntermediate 数据类字段顺序和 protocol/liquid_bridge_contract.py required_keys 一致。
MODEL_INTERMEDIATE_KEYS = ("bias", "risk", "uwb_scaling", "vio_scaling")

# 上下文特征键名（单源真相，§11.3 / 铁律 9）。
# LSTM network.py 与 Liquid model_factory.py 均从此处引用，禁止各模块本地重复定义。
# 字段语义：valid=共享无效硬标志，modality_gap_dt=当前模态距上次同模态事件的时间差，
# uwb_range_residual=UWB 距离残差（观测几何量，非质量标签），
# anchor_dx/dy=锚点相对位移，geom_score=几何一致性评分。
# 禁止 quality / uwb_quality_min / uwb_invalid_rate 等 sim 派生质量标签进 NN 上下文。
CONTEXT_FEATURE_KEYS = (
    "valid",
    "modality_gap_dt",
    "uwb_range_residual",
    "anchor_dx",
    "anchor_dy",
    "geom_score",
)
CONTEXT_DIM = 2 + (2 * len(CONTEXT_FEATURE_KEYS))  # 两个模态位 + 每个上下文特征的值与是否观测到标记。

# Liquid 读出上下文键名（单源真相，统一 network.py / output_head.py / fusion_runner.py 三处重复定义）。
# 这些字段从估计器状态/缓存中提取，注入模型特征窗口，让模型"看到"估计器实时状态。
# output_head.py 公开名 LIQUID_READOUT_CONTEXT_KEYS 从此处引用；
# network.py 和 fusion_runner.py 通过别名 _READOUT_CONTEXT_KEYS 引用，保持下游引用不变。
LIQUID_READOUT_CONTEXT_KEYS = (
    "state_cov_trace",         # 状态协方差矩阵的迹，反映整体不确定性。
    "pos_cov",                 # 位置协方差，反映位置估计不确定度。
    "last_innovation_norm",    # 上一次新息范数，反映量测与预测的偏差。
    "last_gate_skip_flag",     # 上一次门控跳过标志，反映量测是否被拒绝。
    "consecutive_skip_count",  # 连续跳过更新次数，异步/NLOS 场景关键信号。
    "time_since_last_update",  # 距上次成功更新的时间间隔，异步场景关键信号。
    # ---- 以下 8 键为 v2 扩展，释放 readout 候选面 ----
    "pos_cov_trace",           # 位置子块协方差迹（位置不确定性独立度量）。
    "vel_cov_trace",           # 速度子块协方差迹（速度不确定性独立度量）。
    "uwb_residual_norm",       # UWB 残差范数（量测与预测偏差幅度）。
    "vio_cov_summary",         # VIO 协方差摘要（VIO 测量噪声聚合指标）。
    "geometry_dop",            # 几何 DOP（锚点几何分布质量）。
    "cross_modal_consistency", # 跨模态一致性（UWB 与 VIO 估计一致程度）。
    "voxel_feature",           # 体素特征（空间占用稠密表示）。
    "last_update_skip_flag",   # 上次更新是否跳过（与 last_gate_skip_flag 互补）。
)

# Liquid 读出上下文模态缓存键名及默认值（单源真相，D9 配置表面漂移根因修复 + D7 训练/推理同口径）。
# 这些字段在事件循环中持续跟踪每个模态的门控状态、新息范数、连续跳过次数等"观测状态"，
# 与上方 LIQUID_READOUT_CONTEXT_KEYS（模型特征面 v2 共 14 键）是不同概念：模态缓存为每个键
# 同时保存 `_observed` 标志位（何时被实际写入），并包含 last_successful_update_timestamp 时间戳。
# v2 起 cache 与 readout context keys 数量保持一致（每键一对 cache 槽位 + observed 标志）。
# state_cov_trace/pos_cov/time_since_last_update 仍由 _build_readout_context 从估计器状态实时计算，
# 不进入 readout context cache（避免与估计器状态双重单一真相）。
# fusion_runner.py（推理侧 _init_readout_context_cache）与 train_pipeline.py
# （训练侧 _init_training_readout_context_cache）均从此处引用，禁止在两侧重复字面量定义，
# 防止训练/推理键名静默漂移（AGENTS.md §6 公平性同口径、§4 fusion 桥接层不偷带业务规则）。
LIQUID_READOUT_CONTEXT_CACHE_DEFAULTS = {
    "last_innovation_norm": 0.0,               # 上次新息范数，初始为 0
    "last_gate_skip_flag": 0.0,                # 上次门控跳过标志，初始为 0（未跳过）
    "innovation_observed": False,              # 是否已观察到至少一次新息，初始为 False
    "skip_flag_observed": False,               # 是否已观察到至少一次门控结果，初始为 False
    "consecutive_skip_count": 0.0,             # 连续跳过次数，初始为 0
    "consecutive_skip_observed": False,        # 是否已观察到连续跳过，初始为 False
    "last_successful_update_timestamp": None,  # 上次成功更新的时间戳，初始为 None
    # ---- 以下 8 键为 v2 扩展，与 LIQUID_READOUT_CONTEXT_KEYS 同口径 ----
    # 默认值与 observed 标志保留 v1 语义（0.0/False），上游 _update_readout_context_cache
    # 在加入实际观测源前 seen=False 保持现有行为，避免破坏推理/训练侧旧合同的可见状态。
    "pos_cov_trace": 0.0,                     # 位置子块协方差迹，初始为 0
    "pos_cov_trace_observed": False,           # 是否已观察到 pos_cov_trace，初始为 False
    "vel_cov_trace": 0.0,                      # 速度子块协方差迹，初始为 0
    "vel_cov_trace_observed": False,           # 是否已观察到 vel_cov_trace，初始为 False
    "uwb_residual_norm": 0.0,                 # UWB 残差范数，初始为 0
    "uwb_residual_norm_observed": False,       # 是否已观察到 uwb_residual_norm，初始为 False
    "vio_cov_summary": 0.0,                    # VIO 协方差摘要，初始为 0
    "vio_cov_summary_observed": False,          # 是否已观察到 vio_cov_summary，初始为 False
    "geometry_dop": 0.0,                      # 几何 DOP，初始为 0
    "geometry_dop_observed": False,            # 是否已观察到 geometry_dop，初始为 False
    "cross_modal_consistency": 0.0,            # 跨模态一致性，初始为 0
    "cross_modal_consistency_observed": False,  # 是否已观察到 cross_modal_consistency，初始为 False
    "voxel_feature": 0.0,                      # 体素特征，初始为 0
    "voxel_feature_observed": False,            # 是否已观察到 voxel_feature，初始为 False
    "last_update_skip_flag": 0.0,              # 上次更新跳过标志，初始为 0
    "last_update_skip_flag_observed": False,    # 是否已观察到 last_update_skip_flag，初始为 False
}

GROUND_TRUTH_BY_TASK_ID = "ground_truth_by_task_id"
GROUND_TRUTH_BY_SCENE_VARIANT_ID = "ground_truth_by_scene_variant_id"
GROUND_TRUTH_BY_SCENE_ID = "ground_truth_by_scene_id"
GROUND_TRUTH_BY_SEQ_ID = "ground_truth_by_seq_id"

# LiquidNetwork.forward 返回包装字典的键名（单源真相，D9 配置表面漂移根因修复）。
# network.py forward 写入端与 inference.py _unwrap_shared_features/_extract_shared_features
# 读取端均从此处引用，禁止在两侧重复字面量。键名固定为 "shared_features"，
# 与 _unwrap_shared_features 的 Mapping/属性双路径识别合同一致。
SHARED_FEATURES_KEY = "shared_features"

# 模型维度 OOM 上界（单源真相，D9 漂移根因修复 + D7 公平性）。
# input_dim / hidden_dim 的 OOM 守卫上界，cell.py _require_int、liquid/network.py _build_cell、
# lstm/network.py _coerce_positive_int 三处共用，保证 LSTM 与 Liquid 对维度配置的拒绝口径一致。
# 修改此值必须同步三处消费点；common 层非冻结真相，但禁止单点漂移。
MAX_MODEL_DIM: int = 1_000_000  # input_dim/hidden_dim 上界，超过此值拒绝以避免 nn.Linear/LSTM 权重分配 OOM。

# 运行时资源估算换算因子（单源真相，D9 配置表面漂移根因修复）。
# ram_peak（MB）= max(RAM_PEAK_FLOOR_MB, params / RAM_PEAK_PARAMS_PER_MB)。
# 此换算口径在 estimators/ekf_core.py _build_runtime_resource_meta 与
# factories/model_factory.py _build_module_runtime_resource_meta 两处共用，
# 保证 EKF 与神经模型对峰值内存的估算口径一致（D7 公平性）。
# 注：metric_schema.py 已冻结 ram_peak 单位为 "MB"，此因子是冻结的实现口径，
# 修改会改变所有方法的 ram_peak 绝对值并破坏与历史 benchmark 记录的可比性；
# common 层非冻结真相，但禁止单点漂移。
RAM_PEAK_PARAMS_PER_MB: float = 256.0  # params 计数到 ram_peak（MB）的冻结换算因子。
RAM_PEAK_FLOOR_MB: float = 1.0  # ram_peak 最小下限（MB），空模型或极小模型也至少占 1 MB。

# 指标长表专用字段名集合（单源真相，D9 配置表面漂移根因修复）。
# pipelines/eval_pipeline.py _LONG_FORM_KEYS、plotting/plot_runtime.py _LONG_FORM_KEYS、
# plotting/plot_calibration.py _LONG_FORM_KEYS 三处从此处引用，
# analysis/summary_builder.py _is_long_form_metric_row 也从此处引用核心键 "metric"/"value"。
# 禁止在各模块本地重复定义 {"metric", "value", "unit", "direction", "group"} 集合。
# 风险：若任一处漂移，会导致长表控制列剥离不一致，破坏 main_table 聚合视图与 statistics_table 的字段合同。
LONG_FORM_METRIC_KEYS = frozenset({"metric", "value", "unit", "direction", "group"})  # 长表模式专用 5 字段。
LONG_FORM_METRIC_KEY: str = "metric"  # 长表指标名列。
LONG_FORM_VALUE_KEY: str = "value"  # 长表指标值列。
LONG_FORM_GROUP_KEY: str = "group"  # 长表指标分组名列（与 metric_schema 元信息 group 键一致）。

# 指标分组名常量（单源真相，D9 配置表面漂移根因修复）。
# plotting/plot_calibration.py _MECHANISM_METRIC_NAMES 构造与 _pivot_mechanism_rows 过滤均从此处引用，
# 禁止在 plotting 层本地重复定义 "mechanism" 字面量。
# 注意：protocol/metric_schema.py 当前仍使用字面量（冻结层，需授权才能改为消费此常量）；
# 修改此值必须同步检查 protocol/metric_schema.py 中的 group 值一致性。
METRIC_GROUP_MECHANISM: str = "mechanism"  # 机制指标分组名。

# 运行时指标分组名常量（单源真相，D9 配置表面漂移根因修复）。
# plotting/plot_runtime.py _RUNTIME_METRIC_NAMES 构造与 _coerce_runtime_rows 长表过滤均从此处引用，
# 禁止在 plotting 层本地重复定义 "runtime" 字面量。
# 注意：protocol/metric_schema.py 当前仍使用字面量（冻结层，需授权才能改为消费此常量）；
# 修改此值必须同步检查 protocol/metric_schema.py 中的 group 值一致性。
METRIC_GROUP_RUNTIME: str = "runtime"  # 运行时指标分组名。

# 案例引用字段名与分隔符（单源真相，D9 配置表面漂移根因修复）。
# analysis/case_selector.py _extract_case_ref（权威定义）、analysis/summary_builder.py、
# plotting/plot_cases.py 均从此处引用，禁止在各模块本地重复定义字面量。
# 风险：若任一处漂移，会导致 case_ref 提取与 scene_id::seq_id 拼接逻辑不一致，破坏 selected_cases 引用合同。
CASE_REF_KEY: str = "case_ref"  # 案例显式引用字段名。
CASE_REF_SEPARATOR: str = "::"  # scene_id 与 seq_id 联合引用的分隔符。

# 绘图标签轴优先字段名（单源真相，D9 配置表面漂移根因修复）。
# plotting/plot_calibration.py _resolve_label_axis、plotting/plot_runtime.py _resolve_label_axis
# 两处从此处引用，禁止在各模块本地重复定义 ("case_ref", "task_id", "scene_id", "seq_id", "method_name") 字面量。
# 风险：若任一处漂移，会导致两图标签列选择不一致，破坏跨图对比的可读性合同。
PREFERRED_LABEL_KEYS: tuple[str, ...] = (CASE_REF_KEY, "task_id", SCENE_ID_KEY, SEQ_ID_KEY, "method_name")  # 标签轴候选字段优先级。

# 案例分组名常量（单源真相，D9 配置表面漂移根因修复）。
# analysis/case_selector.py select_cases、analysis/case_selector_runner.py build_selected_cases/
# build_empty_selected_cases、analysis/summary_builder.py _collect_case_refs 均从此处引用，
# 禁止在各模块本地重复定义 ("main_cases", "failure_cases", "boundary_cases") 字面量。
# 风险：若任一处漂移，会导致 selected_cases 字典键名不一致，破坏 case_refs 收集与 summary 协议校验。
CASE_GROUP_MAIN: str = "main_cases"  # 主案例分组名。
CASE_GROUP_FAILURE: str = "failure_cases"  # 失败案例分组名。
CASE_GROUP_BOUNDARY: str = "boundary_cases"  # 边界案例分组名。
CASE_GROUP_NAMES: tuple[str, ...] = (CASE_GROUP_MAIN, CASE_GROUP_FAILURE, CASE_GROUP_BOUNDARY)  # 三个必需分组的有序元组。

# 案例图默认输出后缀（单源真相，D9 配置表面漂移根因修复）。
# plotting/plot_cases.py _resolve_group_figure_path 从此处引用，禁止在绘图层本地重复定义 ".png" 字面量。
# 风险：若本地重复定义，会导致无后缀 figure_path 的默认输出格式漂移，破坏 case figure 文件名合同与下游 manifest 路径一致性。
DEFAULT_CASE_FIGURE_SUFFIX: str = ".png"  # 案例图默认输出文件后缀（Pillow 渲染的栅格图格式）。

# 绘图层 figure_path 键名（单源真相，D9 配置表面漂移根因修复）。
# plotting 层 7 个模块（plot_sweeps/plot_runtime/plot_calibration/plot_training_trends/
# plot_trajectories/plot_main_table/plot_cases）均使用 "figure_path" 字面量作为
# (1) figure_cfg 输入配置键、(2) figure_spec 内部规范键、(3) manifest 输出清单键。
# 风险：若本地重复定义，会导致输入/输出合同键名漂移，破坏下游 manifest 消费者路径读取一致性。
FIGURE_PATH_KEY: str = "figure_path"  # 绘图层统一的输出路径键名。

# 时间字段候选键名（单源真相，D1 协议一致性 + D9 配置表面漂移根因修复）。
# metrics 层（metric_runner.py、trajectory_metrics.py）与 plotting 层（plot_trajectories.py）
# 必须使用同一候选集合，避免"index"被误当时间键导致按值而非按位置对齐。
# 风险：若本地重复定义且包含 "index"，会把位置序号误当时间键，导致轨迹对齐错位。
TIME_KEY_CANDIDATES: tuple[str, ...] = ("t", "timestamp", "time")  # 时间字段候选键名（不含 "index"）。

# 训练态标签构造模式常量（单源真相，L5 追溯字段根因修复，对齐 docs/loss_function.md §实现约束 8）。
# 这些常量用于训练报告与 training_flow_contract 中显式记录所使用的标签构造模式，
# 保证不同损失版本在复现实验、加载 checkpoint、对比指标与论文撰写时可追溯、可复核、可审计。
# 当前代码默认事实（来自 docs/loss_function.md §一.2/§一.3/§一.4 "当前代码默认实现" 段落）：
# - risk 标签：基于位置误差与偏航误差的 alignment proxy
# - uwb_scaling 标签：teacher-free 启发式代理标签
# - vio_scaling 标签：teacher-free 启发式代理标签
# 若未来切换到推荐收紧版本（NIS 卡方监督、稳健尺度统计等），需同步更新此处常量与训练报告写入逻辑。
RISK_LABEL_MODE_DEFAULT: str = "alignment_proxy"  # risk 标签构造模式默认值（对齐 docs/loss_function.md §一.2 当前代码默认事实）。
UWB_SCALING_LABEL_MODE_DEFAULT: str = "teacher_free_heuristic"  # uwb_scaling 标签构造模式默认值（对齐 §一.3 当前代码默认实现）。
VIO_SCALING_LABEL_MODE_DEFAULT: str = "teacher_free_heuristic"  # vio_scaling 标签构造模式默认值（对齐 §一.4 当前代码默认实现）。

PRIMARY_KEYS = {  # 按职责分组的主键注册表。
    "event": PRIMARY_EVENT_KEYS,  # 主事件键。
    "meta": META_KEYS,  # 元信息键。
    "payload": PAYLOAD_KEYS,  # 载荷键。
}  # 主键注册表结束。


def validate_constant_registry() -> dict:  # 检查常量注册表内部一致性。
    """生成常量注册表检查报告。

    检查内容包括：
    1. 关键常量名是否都存在于当前模块全局作用域。
    2. PAYLOAD_KEYS 的键集合是否与 ALLOWED_MODALITIES 完全对齐。
    3. DEFAULT_REQUIRED_OUTPUT_FILES 所在目录是否都在 DEFAULT_OUTPUT_DIRS 中注册。
    4. DEFAULT_THRESHOLDS 中质量/VIO位移/IMU区间的最小值是否不超过最大值。
    5. 正值约束：time_tolerance 必须为正。

    注意：桥接层业务阈值（risk/scaling/bias/noise_multiplier 等）的校验
    已迁移至 liquidloc.protocol.bridge_thresholds.validate_bridge_thresholds()。

    Returns:
        dict: 包含以下键的检查报告字典：
            - required_names (list[str]): 应该存在的常量名列表。
            - missing_names (list[str]): 实际缺失的常量名列表。
            - contract_errors (list[str]): 契约错误描述列表。
            - is_complete (bool): 所有检查是否全部通过。
    """
    required_names = [  # 先列出这个模块应该存在的关键常量名。
        "MODALITY_IMU",  # IMU 模态名。
        "MODALITY_UWB",  # UWB 模态名。
        "MODALITY_VIO",  # VIO 模态名。
        "MODALITY_FLOW",  # 光流/流量类模态名。
        "MODALITY_TOF",  # ToF 测距类模态名。
        "ALLOWED_MODALITIES",  # 允许模态集合。
        "DEFAULT_OUTPUT_DIRS",  # 默认输出目录集合。
        "DEFAULT_REQUIRED_OUTPUT_FILES",  # 默认必需输出文件。
        "DEFAULT_THRESHOLDS",  # 默认阈值表。
        "PRIMARY_KEYS",  # 主键注册表。
        "PAYLOAD_KEYS",  # 载荷键表。
        "ASYNC_GAP_FULL_SCALE_S",  # 异步时间间隔归一化满量程。
        "VIO_REF_POSE_STALE_SECONDS",  # VIO 参考位姿过时阈值。
        "QUALITY_FLOOR_EPSILON",  # 质量门槛比较浮点容差。
        "RISK_PARTIAL_DAMPING_COEFF",  # 安全模式衰减系数。
        "DRIFT_PENALTY_SATURATION_M",  # V 轴漂移惩罚饱和阈值。
        "BRIDGE_RISK_MIN",  # 桥接层风险最小值。
        "BRIDGE_RISK_MAX",  # 桥接层风险最大值。
        "BRIDGE_SCALING_MAX",  # 桥接层缩放最大值。
        "BRIDGE_BIAS_MAX",  # 桥接层偏置上界。
        "BRIDGE_BIAS_ABSOLUTE_MAX",  # 施加偏置绝对值上限。
        "BRIDGE_NOISE_MULTIPLIER_CEILING",  # 噪声倍数硬上限。
        "RISK_PRIOR_PROB",  # 风险先验概率。
        "RISK_PRIOR_LOGIT",  # 风险先验 logit 偏置。
        "ALLOWED_RUN_MODES",  # 允许的运行模式集合。
        "RUN_MODE_ALIASES",  # 运行模式别名映射。
        "MODEL_NAME_LSTM",  # LSTM 模型名字常量（D9 漂移根因修复）。
        "MODEL_NAME_LIQUID",  # Liquid 模型名字常量（D9 漂移根因修复）。
        "MODEL_NAME_TRANSFORMER",  # Transformer-EKF 模型名字常量（§10.2 第 2 行 + §10.4 序列截断对等口径前瞻，仓库尚未实现模型本体）。
        "ESTIMATOR_NAME_EKF",  # EKF 估计器名字常量（D9 漂移根因修复）。
        "ESTIMATOR_NAME_ROBUST_EKF",  # 鲁棒 EKF 估计器名字常量（D9 漂移根因修复）。
        "ESTIMATOR_NAME_FGO",  # FGO 估计器名字常量（D9 漂移根因修复）。
        "DATASET_NAME_SIM",  # 仿真数据集名字常量（D9 漂移根因修复）。
        "DATASET_NAME_MILUV",  # MILUV 数据集名字常量（D9 漂移根因修复）。
        "DATASET_NAME_NTU_VIRAL",  # NTU VIRAL 数据集名字常量（D9 漂移根因修复）。
        "DATASET_NAME_UTIL",  # UTIL 数据集名字常量（D9 漂移根因修复）。
        "MILUV_OFFICIAL_ANCHOR_METADATA_SOURCE",  # MILUV 官方锚点元数据来源标识（D9 漂移根因修复）。
        "VIO_MEASUREMENT_ITEMS",  # VIO 测量项常量。
        "VIO_SHARED_NOISE_KEYS",  # VIO 共享噪声方案键名常量（D9 单源）。
        "VIO_NOISE_SCHEME_SHARED",  # VIO 噪声共享方案名常量（D9 单源）。
        "VIO_NOISE_SCHEME_PER_AXIS",  # VIO 噪声逐轴方案名常量（D9 单源）。
        "RELIABILITY_STATUS_OK",  # 可靠性状态正常常量（D9 单源）。
        "RELIABILITY_STATUS_INSUFFICIENT",  # 可靠性状态不足常量（D9 单源）。
        "STATE_ITEMS",  # 状态项常量。
        "MODEL_INTERMEDIATE_KEYS",  # 模型中间量输出头键名。
        "LIQUID_READOUT_CONTEXT_KEYS",  # Liquid 读出上下文键名。
        "LIQUID_READOUT_CONTEXT_CACHE_DEFAULTS",  # Liquid 读出上下文模态缓存键名及默认值（D9 单源）。
        "GROUND_TRUTH_BY_TASK_ID",  # 真值覆盖表键名常量（D9 单源）。
        "GROUND_TRUTH_BY_SCENE_VARIANT_ID",  # 真值覆盖表键名常量（D9 单源）。
        "GROUND_TRUTH_BY_SCENE_ID",  # 真值覆盖表键名常量（D9 单源）。
        "GROUND_TRUTH_BY_SEQ_ID",  # 真值覆盖表键名常量（D9 单源）。
        "SHARED_FEATURES_KEY",  # LiquidNetwork.forward 返回包装键名。
        "MAX_MODEL_DIM",  # 模型维度 OOM 上界。
        "RAM_PEAK_PARAMS_PER_MB",  # ram_peak 换算因子（D9 漂移根因修复）。
        "RAM_PEAK_FLOOR_MB",  # ram_peak 最小下限（D9 漂移根因修复）。
        "RISK_LABEL_MODE_DEFAULT",  # risk 标签构造模式默认值（L5 追溯字段）。
        "UWB_SCALING_LABEL_MODE_DEFAULT",  # uwb_scaling 标签构造模式默认值（L5 追溯字段）。
        "VIO_SCALING_LABEL_MODE_DEFAULT",  # vio_scaling 标签构造模式默认值（L5 追溯字段）。
        "DEVICE_AUTO",  # 训练设备请求：auto 常量（D9 漂移根因修复）。
        "DEVICE_CPU",  # 训练设备请求：cpu 常量（D9 漂移根因修复）。
        "DEVICE_CUDA",  # 训练设备请求：cuda 常量（D9 漂移根因修复）。
        "DEVICE_CUDA_PREFIX",  # CUDA 设备索引前缀常量（D9 漂移根因修复）。
        "ALLOWED_TRAIN_DEVICES",  # 训练管线允许的设备请求集合（D9 漂移根因修复）。
        "CASE_REF_KEY",  # 案例显式引用字段名（D9 漂移根因修复）。
        "SCENE_ID_KEY",  # 场景轴字段名（D9 漂移根因修复）。
        "SEQ_ID_KEY",  # 序列轴字段名（D9 漂移根因修复）。
        "CASE_REF_SEPARATOR",  # case_ref 联合引用分隔符（D9 漂移根因修复）。
        "FIGURE_PATH_KEY",  # 绘图层 figure_path 键名（D9 漂移根因修复）。
        "TIME_KEY_CANDIDATES",  # 时间字段候选键名（D1+D9 漂移根因修复）。
    ]  # required_names 结束。
    missing_names = [name for name in required_names if name not in globals()]  # 找出当前模块里缺失的常量名。
    contract_errors = []  # 用这个列表收集契约错误。
    if not missing_names:  # 只有基础常量齐全时才继续做内部一致性检查。
        # 模态枚举与 payload 键必须一一对应，避免某个模态有名字却没载荷入口。
        if set(PAYLOAD_KEYS) != set(ALLOWED_MODALITIES):  # key 集合必须完全相同。
            _pk = set(PAYLOAD_KEYS)
            _am = set(ALLOWED_MODALITIES)
            contract_errors.append(  # 不匹配就记一条错误。
                f"payload_keys_do_not_match_allowed_modalities: payload_keys={sorted(_pk)}, allowed_modalities={sorted(_am)}, "
                f"missing_in_payload={sorted(_am - _pk)}, extra_in_payload={sorted(_pk - _am)}"
            )
        # 必需输出文件所在目录必须在标准输出目录集中注册。
        missing_output_dirs = sorted(  # 把缺失的目录名整理成稳定顺序。
            {  # 用集合去重。
                output_file.split("/", 1)[0]  # 取每个必需文件的顶层目录。
                for output_file in DEFAULT_REQUIRED_OUTPUT_FILES  # 遍历必需文件列表。
                if output_file.split("/", 1)[0] not in DEFAULT_OUTPUT_DIRS  # 只留下未注册目录。
            }  # 集合结束。
        )  # 排序结束。
        if missing_output_dirs:  # 如果有缺失目录。
            contract_errors.append(  # 记一条目录缺失错误。
                f"required_output_dirs_missing:{','.join(missing_output_dirs)}"  # 错误里写清楚缺了哪些。
            )  # append 结束。
        # 阈值区间要保持有序，避免上层把最小值和最大值写反。
        if DEFAULT_THRESHOLDS["quality_min"] > DEFAULT_THRESHOLDS["quality_max"]:  # 质量区间不能反着写。
            contract_errors.append(
                f"quality_thresholds_out_of_order: quality_min={DEFAULT_THRESHOLDS['quality_min']} > quality_max={DEFAULT_THRESHOLDS['quality_max']}"
            )
        if DEFAULT_THRESHOLDS["vio_displacement_min"] > DEFAULT_THRESHOLDS["vio_displacement_max"]:  # VIO 位移区间不能反着写。
            contract_errors.append(
                f"vio_displacement_thresholds_out_of_order: vio_displacement_min={DEFAULT_THRESHOLDS['vio_displacement_min']} > vio_displacement_max={DEFAULT_THRESHOLDS['vio_displacement_max']}"
            )
        if DEFAULT_THRESHOLDS["flow_displacement_min"] > DEFAULT_THRESHOLDS["flow_displacement_max"]:  # 光流位移区间不能反着写。
            contract_errors.append(
                f"flow_displacement_thresholds_out_of_order: flow_displacement_min={DEFAULT_THRESHOLDS['flow_displacement_min']} > flow_displacement_max={DEFAULT_THRESHOLDS['flow_displacement_max']}"
            )
        if DEFAULT_THRESHOLDS["imu_accel_min"] > DEFAULT_THRESHOLDS["imu_accel_max"]:  # IMU 加速度区间不能反着写。
            contract_errors.append(
                f"imu_accel_thresholds_out_of_order: imu_accel_min={DEFAULT_THRESHOLDS['imu_accel_min']} > imu_accel_max={DEFAULT_THRESHOLDS['imu_accel_max']}"
            )
        if DEFAULT_THRESHOLDS["imu_gyro_min"] > DEFAULT_THRESHOLDS["imu_gyro_max"]:  # IMU 角速度区间不能反着写。
            contract_errors.append(
                f"imu_gyro_thresholds_out_of_order: imu_gyro_min={DEFAULT_THRESHOLDS['imu_gyro_min']} > imu_gyro_max={DEFAULT_THRESHOLDS['imu_gyro_max']}"
            )
        # 正值约束：time_tolerance 必须严格为正。
        if DEFAULT_THRESHOLDS["time_tolerance"] <= 0:
            contract_errors.append(f"time_tolerance_must_be_positive: value={DEFAULT_THRESHOLDS['time_tolerance']}")
    return {  # 返回一个统一的检查报告。
        "required_names": required_names,  # 需要存在的常量名。
        "missing_names": missing_names,  # 缺失的常量名。
        "contract_errors": contract_errors,  # 契约错误列表。
        "is_complete": not missing_names and not contract_errors,  # 是否完全通过检查。
    }  # 报告字典结束。


def normalize_run_mode(
    mode: object,
    *,
    default_mode: object,
) -> str:
    """纯字符串级运行模式归一化，不依赖协议层。

    将用户输入的运行模式（可能是缩写、大小写混合等）归一化为标准
    模式名（'quick' 或 'full'）。此函数不执行协议校验，仅做字符串
    映射和合法性检查，适合 common 层使用。

    protocol 层的 normalize_run_mode 在此基础上叠加冻结协议校验。

    参数：
        mode: 用户指定的运行模式，可以是任意类型，内部会转为字符串。
        default_mode: 默认运行模式，必须是 'quick' 或 'full'。

    返回：
        str: 归一化后的运行模式（'quick' 或 'full'）。

    异常：
        ValueError: 模式不在允许集合中时抛出。
    """
    normalized_default = str(default_mode).strip().lower()  # 规范化默认模式。
    if normalized_default not in ALLOWED_RUN_MODES:  # 默认模式也必须合法。
        raise ValueError(f'default_mode must be one of {sorted(ALLOWED_RUN_MODES)}')
    if mode is None:  # 未指定时使用默认模式。
        return normalized_default
    normalized_mode = str(mode).strip().lower()  # 规范化用户指定的模式。
    # 先查别名映射，再检查是否直接是合法模式名。
    if normalized_mode in RUN_MODE_ALIASES:  # 别名映射命中。
        return RUN_MODE_ALIASES[normalized_mode]
    if normalized_mode in ALLOWED_RUN_MODES:  # 直接命中合法模式名。
        return normalized_mode
    raise ValueError(
        f"mode must be one of {sorted(ALLOWED_RUN_MODES)} or aliases {sorted(RUN_MODE_ALIASES)}, got: {mode!r}"
    )
