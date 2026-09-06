"""train_pipeline.py —— 训练流水线总调度层。

本模块负责把原始序列、协议门控、特征构造、teacher 目标、样本切分和训练前端串起来。
它既支持轻量 smoke，也支持真实的 Liquid/LSTM 训练前端，是训练链路的总调度层。
文件存在的目的，是把所有训练entry收拢到一处，方便上层统一调用和审计。

上游依赖：
- liquidloc.protocol.experiment_gates: 实验协议门控
- liquidloc.dataio: 数据读取和事件构建
- liquidloc.factories: 估计器和模型工厂
- liquidloc.models: Liquid/LSTM 训练器
- liquidloc.sensors: UWB 测量和锚点模型

下游调用者：
- liquidloc.pipelines.__init__.py: 通过 TrainPipeline 类暴露 run 接口
- 外部脚本: 通过 run() 便捷函数调用

核心变量：
- _TARGET_KEYS: 训练目标四头名称 (bias, risk, uwb_scaling, vio_scaling)
- _DEFAULT_BRIDGE_THRESHOLDS: 桥接阈值默认值
- _DEFAULT_EKF_CFG: 默认 EKF 估计器配置
"""

from __future__ import annotations  # 启用延迟注解求值，允许类型标注前向引用

from copy import deepcopy  # 深拷贝估计器配置，避免污染默认 EKF 配置缓存或调用方对象。
import json  # JSON 序列化/反序列化，读取 gt.json 等文件
import math  # 数学运算：sqrt、isfinite、pi 等
from functools import lru_cache  # 缓存装饰器，避免重复加载协议配置
from collections.abc import Callable, Iterable, Mapping  # 抽象容器类型，用于 isinstance 检查与类型标注；Callable 用于 train_model_fn 签名注解（D7）
from pathlib import Path  # 路径操作，统一处理文件路径拼接
from typing import Any  # 任意类型标注，用于灵活接口签名

import torch  # PyTorch 核心，CUDA 设备检测和张量运算
import yaml  # YAML 解析，读取 anchors.yaml 等配置文件

from liquidloc.common.angle_utils import angle_delta_rad, wrap_angle_rad  # 角度差计算和角度归一化
from liquidloc.common.gt_utils import align_ground_truth, normalize_gt_rows, resolve_anchor_position, GT_TIME_TOLERANCE  # 真值归一化、对齐与锚点解析的规范实现
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_integer, is_string_like, validate_path_component  # 集中判断 bool / np.bool_ 和整数类型；validate_path_component 用于 seq_id 路径穿越校验（D8/D10 漂移根因修复，对齐 eval_pipeline.py L35）。
from liquidloc.common.constants import ASYNC_GAP_FULL_SCALE_S  # 异步满量程共享常量
from liquidloc.common.constants import DEFAULT_THRESHOLDS  # 全局默认阈值常量
from liquidloc.common.constants import MODALITY_UWB, MODALITY_VIO
from liquidloc.common.constants import ESTIMATOR_NAME_EKF  # EKF 估计器名字单源常量（D9 漂移根因修复，禁止本地 'ekf' 字面量）
from liquidloc.common.constants import MODEL_NAME_LIQUID, MODEL_NAME_LSTM, MODEL_NAME_TRANSFORMER  # 模型名字单源常量（D9 漂移根因修复，禁止本地 'liquid_ekf'/'lstm_ekf' 字面量）
from liquidloc.common.constants import MODEL_INTERMEDIATE_KEYS  # 四头输出头键名单源常量（D9 漂移根因修复，禁止本地重复定义四头元组）
from liquidloc.common.constants import DEVICE_AUTO, DEVICE_CPU, DEVICE_CUDA, ALLOWED_TRAIN_DEVICES  # 训练设备请求单源常量（D9 漂移根因修复，禁止本地 'auto'/'cpu'/'cuda' 字面量）
from liquidloc.common.constants import ALLOWED_RUN_MODES  # 运行模式单源常量（D9 漂移根因修复，禁止本地 {'quick','full'} 字面量集合）
from liquidloc.common.constants import PAYLOAD_KEYS  # 每种模态对应的 payload 键名注册表
from liquidloc.common.constants import RISK_PARTIAL_DAMPING_COEFF  # 安全模式衰减系数
from liquidloc.common.constants import UWB_INVALID_SIGNAL_FLOOR  # 无效 UWB 附加风险下限
from liquidloc.common.constants import VIO_LOW_FEATURES_SIGNAL_FLOOR  # VIO 低特征数风险下限
from liquidloc.common.constants import VIO_HIGH_REPROJ_ERR_SIGNAL_FLOOR  # VIO 高重投影误差风险下限
from liquidloc.common.constants import VIO_TRACKED_FEATURES_FLOOR  # VIO 跟踪特征数阈值
from liquidloc.common.constants import VIO_HIGH_REPROJ_ERR_THRESHOLD  # VIO 重投影误差阈值
from liquidloc.common.constants import VIO_TRACKED_FEATURES_SAFE_FLOOR  # VIO 特征数安全下界
from liquidloc.common.constants import VIO_REPROJ_ERR_NORM_FLOOR  # VIO 重投影误差归一化下限
from liquidloc.common.constants import VIO_REPROJ_ERR_FULL_SCALE  # VIO 重投影误差归一化满量程
from liquidloc.common.constants import LIQUID_READOUT_CONTEXT_KEYS as _READOUT_CONTEXT_KEYS  # 读出上下文键名单源真相，别名保留以避免改下游引用
from liquidloc.common.constants import LIQUID_READOUT_CONTEXT_CACHE_DEFAULTS as _READOUT_CONTEXT_CACHE_DEFAULTS  # 读出上下文模态缓存键名及默认值单源真相（D9 单源，训练/推理同口径）
from liquidloc.common.constants import RISK_LABEL_MODE_DEFAULT, UWB_SCALING_LABEL_MODE_DEFAULT, VIO_SCALING_LABEL_MODE_DEFAULT  # 标签构造模式默认值（L5 追溯字段根因修复，对齐 docs/loss_function.md §实现约束 8）
from liquidloc.common.constants import MILUV_OFFICIAL_ANCHOR_METADATA_SOURCE  # MILUV 官方锚点元数据来源标识（单源真相，D9 漂移根因修复）
from liquidloc.common.constants import DATASET_NAME_MILUV  # MILUV 数据集名字单源常量（D9 漂移根因修复，禁止本地 'miluv' 字面量）
from liquidloc.common.constants import GT_FILE_NAME  # 真值文件名单源常量（D9 漂移根因修复，禁止本地 'gt.json' 字面量）
from liquidloc.common.io_utils import dumps_json_text, read_json  # 严格标准 JSON 读写，拒绝 NaN/Infinity；D10 漂移根因修复，提升到模块顶层，避免函数内延迟 import（与 core_pipeline.py L53 / eval_pipeline.py L37 同口径顶层导入，dumps_json_text 供 _write_json 落盘使用）
from liquidloc.common.config_utils import find_project_root
from liquidloc.common.paths import get_standard_dirs, resolve_output_root  # 获取项目标准目录结构和统一输出根目录解析
from liquidloc.common.seed_utils import cuda_runtime_usable  # 检测 CUDA 运行时是否真正可用
from liquidloc.common.types import ModelIntermediate, StageResult  # 模型中间输出类型和流水线阶段结果
from liquidloc.dataio.adapters.event_builder import (  # 事件构建器：原始数据 -> 标准化事件
    build_imu_events,  # 构建 IMU 事件列表
    build_uwb_events,  # 构建 UWB 事件列表
    build_vio_events,  # 构建 VIO 事件列表
    merge_and_finalize_events,  # 合并并排序所有模态事件，生成最终事件序列
)
from liquidloc.dataio.adapters.field_mapper import map_external_fields  # 外部字段映射到内部字段名
from liquidloc.dataio.readers.miluv_reader import read_miluv_sequence  # 读取 MILUV 序列 bundle
from liquidloc.factories.estimator_factory import create_estimator  # 估计器工厂，按名称创建实例
from liquidloc.factories.model_factory import create_model  # 模型工厂，按名称创建或加载模型
from liquidloc.interfaces.pipeline_api import PipelineAPI, normalize_pipeline_cfg  # 流水线基类接口与统一配置规整 helper
from liquidloc.models.features.feature_builder import build_feature_state_history, build_feature_vector  # 特征构造：事件+状态 -> 特征向量
from liquidloc.models.liquid.trainer import train_model as train_liquid_model  # Liquid 模型训练器
from liquidloc.models.liquid.trainer import _CROSS_TRAINER_PARITY_DECLARATION as _LIQUID_PARITY_DECLARATION  # §13.6.2.8 cross-trainer parity 声明
from liquidloc.models.lstm.trainer import train_model as train_lstm_model  # LSTM 模型训练器
from liquidloc.models.lstm.trainer import _CROSS_TRAINER_PARITY_DECLARATION as _LSTM_PARITY_DECLARATION  # §13.6.2.8 cross-trainer parity 声明
from liquidloc.models.transformer.trainer import train_model as train_transformer_model  # Transformer 模型训练器
from liquidloc.models.transformer.trainer import _CROSS_TRAINER_PARITY_DECLARATION as _TRANSFORMER_PARITY_DECLARATION  # §13.6.2.8 cross-trainer parity 声明
from liquidloc.protocol.experiment_gates import check_test_set_not_in_training_scores, get_default_failure_threshold_m, load_experiment_protocol, normalize_run_mode, normalize_train_request  # 实验协议门控；D10：normalize_run_mode 提升到顶层，避免 _normalize_train_mode 函数内延迟 import。
from liquidloc.protocol.liquid_bridge_contract import UWB_BIAS_MAX_RATIO, _coerce_safe_mode_enabled_flag  # UWB bias 比例截断单源常量；安全模式开关严格解析器。D10 漂移根因修复：提升到模块顶层，避免 _build_geometric_bias_target 与 _build_target_intermediate 函数内延迟 import。
from liquidloc.protocol.scene_axis_protocol import load_scene_axis_protocol, get_nominal_levels  # 读取冻结场景轴协议，用于轴级退化风险下界；获取正常等级名，避免硬编码。
from liquidloc.scenarios.nlos_levels import apply_nlos_level  # §A 修复: NLOS 训练注入entry（与 core_pipeline 推理侧同源，避免数据不对称）。
from liquidloc.protocol.scene_schema import SceneSpec, decode_scene  # 场景规格对象；解析 scene_id，恢复 A/N/V 轴级上下文（D7 返回类型精确化）。

from liquidloc.sensors.anchor_model import build_anchor_lookup, project_anchor_layout_xy  # 锚点查找表构建和3D→2D投影
from liquidloc.sensors.uwb_model import extract_uwb_measurement, predict_range_to_anchor  # UWB 测量提取和几何距离预测


_TARGET_KEYS = ('bias', 'risk', 'uwb_scaling', 'vio_scaling')  # 训练目标的四个输出头名称：bias/risk/uwb_scaling/vio_scaling

_TEACHER_PROJECTION_AUDIT_KEYS = (  # teacher 投影审计键，记录 3D→2D 投影的关键元数据
    'original_anchor_position_dim',  # 原始锚点位置维度（通常为 3）
    'teacher_anchor_position_dim',  # teacher 使用的锚点位置维度（投影后为 2）
    'projection',  # 投影方式（如 'xy'）
    'ignored_axis',  # 被忽略的轴（如 'z'）
)
_ROBUST_SUPPLEMENT_QUALITY_THRESHOLD = 0.55  # robust 补充的质量阈值，观测质量得分低于此值视为低质量
_ROBUST_SUPPLEMENT_ALIGNMENT_THRESHOLD = 0.45  # robust 补充的对齐风险阈值，高于此值视为高风险
_ROBUST_SUPPLEMENT_MAX_BOOST = 0.20  # robust 补充的最大增量上限，防止 scaling 过度膨胀
_ROBUST_SUPPLEMENT_STRENGTH_FACTOR = 0.5  # robust 补充强度相对原始风险信号 max(quality_risk, modality_signal, geometry_risk) 的衰减系数，避免补充强度直接等于风险峰值；最终再受 _ROBUST_SUPPLEMENT_MAX_BOOST 截断（D7 单源常量，替代 _apply_robust_teacher_supplement 内硬编码 0.5）
_OBSERVATION_RISK_BLEND = 0.35  # 观测风险混合系数，用于审计口径
_UWB_GEOMETRIC_BIAS_FULL_SCALE = 0.50  # UWB 几何 bias 归一化满量程
_UWB_SCALING_ALIGNMENT_COEFF = 0.50  # alignment_risk 对 UWB scaling 的贡献系数
_UWB_SCALING_QUALITY_COEFF = 0.25  # quality_risk 对 UWB scaling 的贡献系数
_UWB_SCALING_GEOMETRY_COEFF = 0.35  # geometry_risk 对 UWB scaling 的贡献系数
_UWB_INVALID_EXTRA_BOOST = 0.15  # 无效 UWB 事件额外 scaling 增量
_VIO_SCALING_ALIGNMENT_COEFF = 0.50  # alignment_risk 对 VIO scaling 的贡献系数
_VIO_SCALING_QUALITY_COEFF = 0.25  # quality_risk 对 VIO scaling 的贡献系数
_VIO_LOW_FEATURES_SCALING_BOOST = 0.10  # VIO 低特征数额外 scaling 增量
_VIO_HIGH_REPROJ_SCALING_BOOST = 0.10  # VIO 高重投影误差额外 scaling 增量，bias 除以此值得到 0~1 风险
_ASYNC_GAP_FULL_SCALE = ASYNC_GAP_FULL_SCALE_S  # 异步时间间隔 dt 归一化满量程，dt 除以此值得到 0~1 风险
_ASYNC_UWB_SCALING_COEFF = 0.20  # UWB 异步间隔对 scaling 的影响系数
_ASYNC_VIO_SCALING_COEFF = 0.20  # VIO 异步间隔对 scaling 的影响系数
_AXIS_LEVEL_RISK_KEYS = ('async_axis_risk', 'nlos_axis_risk', 'visual_axis_risk')  # 轴级退化风险审计键。
_READOUT_CONTEXT_MODALITIES = (MODALITY_UWB, MODALITY_VIO)  # readout 上下文涉及的模态类型，与 fusion_runner 共享常量
_DEFAULT_TRAIN_OUTPUT_DIR_NAME: str = 'train_pipeline'  # 训练流水线默认输出子目录名（D9 单源常量，禁止在 _resolve_output_root 调用点重复 'train_pipeline' 字面量，防止两处输出根目录默认名静默漂移）
_DEFAULT_BRIDGE_THRESHOLDS = {  # 桥接阈值默认值字典，用于 teacher-free 目标构造
    'robust_supplement_quality_threshold': _ROBUST_SUPPLEMENT_QUALITY_THRESHOLD,  # robust 补充的质量阈值
    'robust_supplement_alignment_threshold': _ROBUST_SUPPLEMENT_ALIGNMENT_THRESHOLD,  # robust 补充的对齐风险阈值
    'robust_supplement_max_boost': _ROBUST_SUPPLEMENT_MAX_BOOST,  # robust 补充的最大增量
    'observation_risk_blend': _OBSERVATION_RISK_BLEND,  # 观测风险混合系数
    'uwb_geometric_bias_full_scale': _UWB_GEOMETRIC_BIAS_FULL_SCALE,  # UWB 几何 bias 归一化满量程
    'uwb_scaling_alignment_coeff': _UWB_SCALING_ALIGNMENT_COEFF,  # alignment_risk 对 UWB scaling 的贡献系数
    'uwb_scaling_quality_coeff': _UWB_SCALING_QUALITY_COEFF,  # quality_risk 对 UWB scaling 的贡献系数
    'uwb_scaling_geometry_coeff': _UWB_SCALING_GEOMETRY_COEFF,  # geometry_risk 对 UWB scaling 的贡献系数
    'uwb_invalid_extra_boost': _UWB_INVALID_EXTRA_BOOST,  # 无效 UWB 事件额外 scaling 增量
    'vio_scaling_alignment_coeff': _VIO_SCALING_ALIGNMENT_COEFF,  # alignment_risk 对 VIO scaling 的贡献系数
    'vio_scaling_quality_coeff': _VIO_SCALING_QUALITY_COEFF,  # quality_risk 对 VIO scaling 的贡献系数
    'vio_low_features_scaling_boost': _VIO_LOW_FEATURES_SCALING_BOOST,  # VIO 低特征数额外 scaling 增量
    'vio_high_reproj_scaling_boost': _VIO_HIGH_REPROJ_SCALING_BOOST,  # VIO 高重投影误差额外 scaling 增量
    'async_gap_full_scale': _ASYNC_GAP_FULL_SCALE,  # 异步间隔归一化满量程
    'async_uwb_scaling_coeff': _ASYNC_UWB_SCALING_COEFF,  # UWB 异步 scaling 系数
    'async_vio_scaling_coeff': _ASYNC_VIO_SCALING_COEFF,  # VIO 异步 scaling 系数
}
_DEFAULT_EKF_CFG = None  # 延迟加载，首次使用时从 configs/models/ekf.yaml 读取。


@lru_cache(maxsize=1)
def _load_default_ekf_cfg() -> dict:
    """从 configs/models/ekf.yaml 加载默认 EKF 配置，避免硬编码。

    注意：返回值由 ``@lru_cache`` 缓存，是模块级共享对象，调用方必须自行
    ``deepcopy`` 后再修改嵌套结构，否则会污染缓存。当前唯一调用点
    ``_run_real_frontend_pipeline`` 已用 ``deepcopy`` 隔离，与
    ``core_pipeline._resolve_estimator_cfg`` L769 同口径。
    """
    ekf_cfg_path = find_project_root() / 'configs' / 'models' / 'ekf.yaml'
    if ekf_cfg_path.exists():
        with open(ekf_cfg_path, encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    # YAML 不存在时回退到硬编码默认值（保持向后兼容）。
    # D9：name 引用 ESTIMATOR_NAME_EKF 单源常量，禁止本地 'ekf' 字面量漂移。
    # D1：fallback 必须与 configs/models/ekf.yaml / 前提指导 §1.1 主表 8 维 + §2.3 紧耦合扩维
    #（uwb_clock_bias / vio_scale，全体同增至 10 维）严格对齐。
    # 任何维度滞后会让 estimator_factory._validate_process_noise / _validate_init_state /
    # _validate_init_cov 直接拒绝创建估计器（P5 修复后工厂校验已 7 项过程噪声、10 维 init_state、
    # 10 项 init_cov），fallback 路径将在 yaml 缺失场景（CI 容器、解包部署、移动测试）破启动。
    return {
        'name': ESTIMATOR_NAME_EKF,
        'process_noise': {
            'pos': 0.05,
            'vel': 0.10,
            'yaw': 0.02,
            'accel_bias': 0.001,
            'gyro_bias': 0.001,
            'uwb_clock_bias': 0.001,  # UWB 钟差随机游走（m/√s），§2.3 在线辨识要求 P 必须随时间增长。
            'vio_scale': 0.001,       # VIO 尺度因子随机游走（无量纲/√s），围绕标准尺度 1.0 缓慢漂移；与 configs/models/ekf.yaml 严格一致。
        },
        'measurement_noise': {'uwb': 0.25, 'vio': {'pos': 0.08, 'yaw': 0.03}},
        'init_state': {
            'px': 0.0,
            'py': 0.0,
            'vx': 0.0,
            'vy': 0.0,
            'yaw': 0.0,
            'bax': 0.0,
            'bay': 0.0,
            'bg': 0.0,
            'uwb_clock_bias': 0.0,  # UWB 钟差初值（m），0.0 表示无初始钟差。
            'vio_scale': 1.0,       # VIO 尺度因子初值（无量纲），1.0 表示标准尺度（紧耦合 §2.3）。
        },
        'init_cov': [25.0, 25.0, 1.0, 1.0, 1.0, 0.05, 0.05, 0.02, 0.05, 0.01],
        # 与 configs/models/ekf.yaml init_cov 同口径：uwb_clock_bias σ≈0.22 m、vio_scale σ≈0.1。
        'gate': {
            'mahalanobis_sq': {
                'uwb': 3.841,  # χ²(1, 0.95)
                'vio': 7.815,  # χ²(3, 0.95)
            },
        },
    }


def _resolve_alignment_risk_scales() -> tuple[float, float]:
    """解析 alignment_risk 使用的平移/偏航归一化尺度。

    默认走协议层 failure_threshold_m=1.0m，但在大跨度 sim 场景下 pose_error 几乎一直 > 1m
    会让 risk label 天然 saturate 到 1.0，网络学到常量退化函数。
    支持 LIQUIDLOC_ALIGNMENT_POSE_FULL_SCALE_M 环境变量覆盖（必须 > 0，
    典型取轨迹最大跨度量级如 25.0/50.0m）以解决该 saturation 问题。
    """
    import os
    env_pose_scale = os.environ.get('LIQUIDLOC_ALIGNMENT_POSE_FULL_SCALE_M')
    if env_pose_scale is not None and str(env_pose_scale).strip() != '':
        # 走环境变量 override 路径，仍强制正有限
        pose_full_scale_m = coerce_finite_scalar(
            float(env_pose_scale),
            name='LIQUIDLOC_ALIGNMENT_POSE_FULL_SCALE_M',
            min_value=0.0,
            inclusive=False,
        )
        return pose_full_scale_m, math.pi
    # 默认路径：协议层 get_default_failure_threshold_m = 1.0m
    # 统一走 coerce_finite_scalar，与 _resolve_bridge_thresholds 同口径；
    # 协议层 get_default_failure_threshold_m 已保证正有限，此处为防御性深度校验，
    # 违例直接抛 ValueError，禁止静默回退 1.0 以免污染 alignment_pose_full_scale_m 审计口径。
    pose_full_scale_m = coerce_finite_scalar(
        get_default_failure_threshold_m(),  # 使用缓存路径，避免绕过 _resolve_protocol_cfg 的 lru_cache。
        name='alignment_pose_full_scale_m',
        min_value=0.0,
        inclusive=False,  # 必须 > 0，避免除零；与原 <= 0.0 拒绝语义一致。
    )
    return pose_full_scale_m, math.pi


def _resolve_bridge_thresholds(cfg: Mapping[str, Any] | None = None) -> dict[str, float]:
    """解析桥接阈值配置，合并默认值和用户覆盖。

    从 _DEFAULT_BRIDGE_THRESHOLDS 出发，用 cfg 中的 bridge_thresholds 覆盖，
    然后逐个验证每个阈值的类型和范围。

    Args:
        cfg: 模型配置字典，可包含 bridge_thresholds 子字典。

    Returns:
        解析后的完整桥接阈值字典。

    Raises:
        ValueError: bridge_thresholds 不是 Mapping 类型、包含未知键，或某个阈值超出允许范围。
    """
    # 从默认值开始
    resolved = dict(_DEFAULT_BRIDGE_THRESHOLDS)  # 以默认桥接阈值为基底
    if cfg is not None:  # 如果提供了配置字典
        raw_overrides = cfg.get('bridge_thresholds')  # 读取用户覆盖的桥接阈值
        if raw_overrides is not None:  # 如果存在用户覆盖
            if not isinstance(raw_overrides, Mapping):  # 检查覆盖值是否为 Mapping 类型
                raise ValueError('bridge_thresholds must be a mapping when provided')  # 非映射类型直接报错
            unknown_keys = sorted(set(raw_overrides) - set(_DEFAULT_BRIDGE_THRESHOLDS))  # 检查是否有未知键
            if unknown_keys:  # 存在未知键时报错
                raise ValueError(f'bridge_thresholds contains unknown keys: {unknown_keys}')  # 未知键报错
            resolved.update({str(key): value for key, value in raw_overrides.items()})  # 用用户覆盖值更新默认值

    resolved['robust_supplement_quality_threshold'] = coerce_finite_scalar(  # 验证 quality_threshold 的范围
        resolved['robust_supplement_quality_threshold'],  # quality_threshold 值
        name='bridge_thresholds.robust_supplement_quality_threshold',  # quality_threshold 参数名
        min_value=0.0,  # 最小值 0
        max_value=1.0,  # 最大值 1
    )
    resolved['robust_supplement_alignment_threshold'] = coerce_finite_scalar(  # 验证 alignment_threshold 的范围
        resolved['robust_supplement_alignment_threshold'],  # alignment_threshold 值
        name='bridge_thresholds.robust_supplement_alignment_threshold',  # alignment_threshold 参数名
        min_value=0.0,  # 最小值 0
        max_value=1.0,  # 最大值 1
    )
    resolved['robust_supplement_max_boost'] = coerce_finite_scalar(  # 验证 max_boost 的范围
        resolved['robust_supplement_max_boost'],  # max_boost 值
        name='bridge_thresholds.robust_supplement_max_boost',  # max_boost 参数名
        min_value=0.0,  # 最小值 0，无上限
    )
    resolved['observation_risk_blend'] = coerce_finite_scalar(  # 验证 observation_risk_blend 的范围 0~1
        resolved['observation_risk_blend'],  # observation_risk_blend 值
        name='bridge_thresholds.observation_risk_blend',  # observation_risk_blend 参数名
        min_value=0.0,  # 最小值 0
        max_value=1.0,  # 最大值 1
    )
    resolved['uwb_geometric_bias_full_scale'] = coerce_finite_scalar(  # 验证 UWB 几何 bias 满量程
        resolved['uwb_geometric_bias_full_scale'],  # uwb_geometric_bias_full_scale 值
        name='bridge_thresholds.uwb_geometric_bias_full_scale',  # 参数名
        min_value=1e-6,  # 最小值接近 0，避免除零
    )
    resolved['async_gap_full_scale'] = coerce_finite_scalar(  # 验证异步间隔满量程
        resolved['async_gap_full_scale'],  # async_gap_full_scale 值
        name='bridge_thresholds.async_gap_full_scale',  # 参数名
        min_value=1e-6,  # 最小值接近 0，避免除零
    )
    resolved['async_uwb_scaling_coeff'] = coerce_finite_scalar(  # 验证 UWB 异步 scaling 系数
        resolved['async_uwb_scaling_coeff'],  # async_uwb_scaling_coeff 值
        name='bridge_thresholds.async_uwb_scaling_coeff',  # 参数名
        min_value=0.0,  # 最小值 0，允许无贡献
    )
    resolved['async_vio_scaling_coeff'] = coerce_finite_scalar(  # 验证 VIO 异步 scaling 系数
        resolved['async_vio_scaling_coeff'],  # async_vio_scaling_coeff 值
        name='bridge_thresholds.async_vio_scaling_coeff',  # 参数名
        min_value=0.0,  # 最小值 0，允许无贡献
    )
    # 验证 teacher-free scaling 系数
    resolved['uwb_scaling_alignment_coeff'] = coerce_finite_scalar(
        resolved['uwb_scaling_alignment_coeff'],
        name='bridge_thresholds.uwb_scaling_alignment_coeff',
        min_value=0.0,
    )
    resolved['uwb_scaling_quality_coeff'] = coerce_finite_scalar(
        resolved['uwb_scaling_quality_coeff'],
        name='bridge_thresholds.uwb_scaling_quality_coeff',
        min_value=0.0,
    )
    resolved['uwb_scaling_geometry_coeff'] = coerce_finite_scalar(
        resolved['uwb_scaling_geometry_coeff'],
        name='bridge_thresholds.uwb_scaling_geometry_coeff',
        min_value=0.0,
    )
    resolved['uwb_invalid_extra_boost'] = coerce_finite_scalar(
        resolved['uwb_invalid_extra_boost'],
        name='bridge_thresholds.uwb_invalid_extra_boost',
        min_value=0.0,
    )
    resolved['vio_scaling_alignment_coeff'] = coerce_finite_scalar(
        resolved['vio_scaling_alignment_coeff'],
        name='bridge_thresholds.vio_scaling_alignment_coeff',
        min_value=0.0,
    )
    resolved['vio_scaling_quality_coeff'] = coerce_finite_scalar(
        resolved['vio_scaling_quality_coeff'],
        name='bridge_thresholds.vio_scaling_quality_coeff',
        min_value=0.0,
    )
    resolved['vio_low_features_scaling_boost'] = coerce_finite_scalar(
        resolved['vio_low_features_scaling_boost'],
        name='bridge_thresholds.vio_low_features_scaling_boost',
        min_value=0.0,
    )
    resolved['vio_high_reproj_scaling_boost'] = coerce_finite_scalar(
        resolved['vio_high_reproj_scaling_boost'],
        name='bridge_thresholds.vio_high_reproj_scaling_boost',
        min_value=0.0,
    )
    return resolved  # 返回解析后的完整桥接阈值字典


def _model_intermediate_to_dict(intermediate: ModelIntermediate) -> dict[str, float]:
    """将 ModelIntermediate 转换为纯字典，便于 JSON 序列化。

    Args:
        intermediate: 模型中间输出对象。

    Returns:
        包含 bias/risk/uwb_scaling/vio_scaling 的字典。
    """
    # 字段名引用 _TARGET_KEYS 单源常量，避免字面量漂移；数值经 coerce_finite_scalar
    # 校验有限性，与 ModelIntermediate.__post_init__ 的统一模式保持一致。
    return {  # 构建输出字典
        key: coerce_finite_scalar(getattr(intermediate, key), name=key)
        for key in _TARGET_KEYS
    }


def _normalize_train_mode(mode: str | None) -> str:
    """通过共享协议合同路由训练模式归一化。

    将训练模式归一化委托给 protocol 层的 ``normalize_run_mode``，确保与
    冻结的 quick/full 合同保持一致；空值默认为 ``full``。protocol 层会
    校验 ``quick_full_rule`` 未被篡改，并在模式非法时抛出 ``ValueError``。

    Args:
        mode: 训练模式字符串，可为 None。

    Returns:
        归一化后的小写字符串（``quick`` 或 ``full``）。

    Raises:
        ValueError: 模式不在允许集合中或协议 quick_full_rule 被篡改。
        TypeError: quick_full_rule 类型异常时由协议层抛出。
    """
    return normalize_run_mode(mode, default_mode='full')


def _resolve_train_device(
    device_request: str | None,
    *,
    allow_auto_cuda: bool,
) -> dict[str, Any]:
    """解析训练设备请求，决定最终使用 CPU 还是 CUDA。

    Args:
        device_request: 设备请求字符串，可为 auto/cpu/cuda 或 None。
        allow_auto_cuda: auto 模式下是否允许自动选择 CUDA。

    Returns:
        包含 requested_device/selected_device/cuda_available/cuda_runtime_available 的字典。

    Raises:
        ValueError: device_request 不是合法设备选项。
        TypeError: device_request 既不是 None 也不是字符串。
    """
    # 解析设备请求，决定使用 CPU 还是 CUDA
    if device_request is None:  # 未指定时默认 auto
        requested_device = DEVICE_AUTO
    elif is_string_like(device_request):  # 字符串请求：归一化大小写与空格
        requested_device = str(device_request).strip().lower()
    else:  # 非 None 非字符串：拒绝，避免 str() 静默掩盖非法类型（D8 根因修复）
        raise TypeError(
            f"device must be a string or None, got {type(device_request).__name__}"
        )
    if requested_device not in ALLOWED_TRAIN_DEVICES:  # 检查是否为合法设备选项
        raise ValueError(
            f"device must be one of {sorted(ALLOWED_TRAIN_DEVICES)}, got: {device_request!r}"
        )

    cuda_available = bool(torch.cuda.is_available())  # 检查 PyTorch 是否检测到 CUDA
    cuda_runtime_available = bool(cuda_available and cuda_runtime_usable())  # 检查 CUDA 运行时是否真正可用
    if requested_device == DEVICE_CPU:  # 明确请求 CPU 时，忽略 CUDA 可用性
        selected_device = DEVICE_CPU  # 选择 CPU
    elif requested_device == DEVICE_CUDA and cuda_runtime_available:  # 请求 CUDA 且可用时
        selected_device = DEVICE_CUDA  # 选择 CUDA
    elif requested_device == DEVICE_CUDA:  # 请求 CUDA 但不可用时，降级到 CPU
        selected_device = DEVICE_CPU  # 降级到 CPU
    else:  # auto 模式：根据条件自动选择
        selected_device = DEVICE_CUDA if allow_auto_cuda and cuda_runtime_available else DEVICE_CPU  # auto 模式下，允许时优先 CUDA

    return {  # 构建设备报告字典
        'requested_device': requested_device,  # 用户请求的设备
        'selected_device': selected_device,  # 最终选择的设备
        'cuda_available': cuda_available,  # PyTorch 是否检测到 CUDA
        'cuda_runtime_available': cuda_runtime_available,  # CUDA 运行时是否真正可用
    }


def _build_quick_closure_note() -> dict[str, Any]:
    """构建 quick 模式的执行说明。

    quick 仅作为冒烟规模标签，不引入新的科学语义；与 full 共享同一协议、
    数据边界、模型数学与验证协议（见 experiment_gates.quick_full_rule）。

    Returns:
        quick 模式执行说明字典。
    """
    # quick 仅是冒烟规模标签，不得写入科学语义差异
    return {  # quick 模式执行说明
        'scope': 'smoke-scale run label only',  # quick 仅是运行规模标签
        'changed': [  # quick 与 full 的差异列表
            'run scale overrides at entry (epochs / batch_size / repeats / max_sequences)',  # 仅规模覆盖
        ],
        'unchanged': [  # quick 与 full 相同的部分
            'target contract',  # 目标合同不变
            'model math',  # 模型数学不变
            'validation protocol',  # 验证协议不变
            'device policy',  # 设备策略不变
            'data boundary',  # 数据边界不变
        ],
        'supports_target_scene_claim': False,  # 训练阶段不支持目标场景声明
        'claim_action': 'no scientific claim difference from full',  # 声明动作：与 full 无科学差异
    }


def _build_full_execution_note() -> dict[str, Any]:
    """构建 full 模式的执行说明。

    full 仅作为真实执行标签，不引入新的科学语义；与 quick 共享同一协议、
    数据边界、模型数学与验证协议（见 experiment_gates.quick_full_rule）。

    Returns:
        full 模式执行说明字典。
    """
    # full 仅是真实执行标签，不得写入科学语义差异
    return {  # full 模式执行说明
        'scope': 'real execution run label',  # full 仅是真实执行标签
        'changed': [  # full 与 quick 的差异列表
            'run scale per protocol defaults',  # 使用协议默认规模
        ],
        'unchanged': [  # full 与 quick 相同的部分
            'target contract',  # 目标合同不变
            'model math',  # 模型数学不变
            'validation protocol',  # 验证协议不变
            'device policy',  # 设备策略不变
            'data boundary',  # 数据边界不变
        ],
        'supports_target_scene_claim': False,  # 训练阶段不支持目标场景声明
        'claim_action': 'requires downstream evaluation evidence',  # 声明动作：需要下游评估证据
    }


def _build_execution_note(mode: str | None) -> dict[str, Any]:
    """根据训练模式构建对应的执行说明。

    Args:
        mode: 训练模式字符串，可为 None。

    Returns:
        quick 或 full 模式的执行说明字典。
    """
    # 根据训练模式构建对应的执行说明
    normalized_mode = _normalize_train_mode(mode)  # 归一化训练模式
    if normalized_mode == 'full':  # full 模式返回 full 执行说明
        return _build_full_execution_note()  # full 模式构建完整执行说明
    return _build_quick_closure_note()  # 其他模式（quick）返回 quick 执行说明


def _enforce_train_device_constraints(cfg: dict[str, Any], device_report: dict[str, Any]) -> None:
    """强制执行训练设备约束，防止不合法的设备/模式组合。

    检查规则：
    - 请求 CUDA 但不可用时，抛出 RuntimeError
    - **断点续跑契约（手册 P25）：full / quick 模式均允许使用 checkpoint_path
      作"断点续跑"，不再强制从头训练**。但在冒烟 → 全量 gate 阶段（smoke_check=True）
      下仍要求从头训练，以保证冒烟阶段不被 checkpoint 状态污染（P19 可复现）。
    - 调用方必须保证：冒烟（全量 gate ①）→ 全量 code freeze → 全量；code freeze 后
      checkpoint_path 用于断点续跑前必须先调用 check_code_freeze 验证 commit 一致。

    quick/full 仅作为运行规模标签，不引入新的科学语义；二者共享同一设备
    策略与算力口径（见 experiment_gates.quick_full_rule 与 AGENTS.md §6）。

    Args:
        cfg: 流水线配置字典。
        device_report: 设备解析报告字典。

    Raises:
        RuntimeError: 设备约束不满足（CUDA 请求但运行时不可用）。
        ValueError: checkpoint_path 类型非法或与冒烟模式不兼容。
    """
    # 强制执行训练设备约束
    model_cfg = cfg.get('model_cfg') or {}  # 读取模型配置（只读访问，无需拷贝；D3 浅拷贝根因修复）
    normalized_mode = _normalize_train_mode(cfg.get('mode'))  # 归一化训练模式
    if device_report['requested_device'] == DEVICE_CUDA and device_report['selected_device'] != DEVICE_CUDA:  # 请求 CUDA 但实际未选中 CUDA
        raise RuntimeError(  # 报错：请求的 CUDA 不可用（运行时环境条件，非值合同违例）
            f'{normalized_mode} training requires an available CUDA GPU and selected_device="cuda"'  # 错误消息：需要可用的 CUDA GPU
        )
    raw_checkpoint_path = model_cfg.get('checkpoint_path')  # 读取 checkpoint_path 配置
    if raw_checkpoint_path is None:  # 未配置 checkpoint_path
        checkpoint_path = ''
    elif isinstance(raw_checkpoint_path, (str, Path)):  # 合法类型：str 或 Path，与 model_factory.load_checkpoint 口径一致
        checkpoint_path = str(raw_checkpoint_path).strip()
    else:  # 类型非法，拒绝静默转换（D8 根因修复）
        raise ValueError(
            f'checkpoint_path must be a str, Path, or None, got {type(raw_checkpoint_path).__name__}'
        )

    # 手册 P25 断点续跑契约：full / quick 均允许 checkpoint_path（不强制从头训练）。
    # 仅在显式声明 smoke_check=True 时要求从头训练，保证冒烟不被 checkpoint 污染。
    # 不带 smoke_check 时保持旧约束（full/quick 必须从头训练），保证旧测试与契约不变。
    smoke_check = bool(cfg.get('smoke_check', False))
    if not smoke_check:
        if normalized_mode == 'full' and checkpoint_path:
            raise ValueError('full training must start from scratch and cannot reuse checkpoint_path')
        if normalized_mode == 'quick' and checkpoint_path:
            raise ValueError('quick training must start from scratch and cannot reuse checkpoint_path')
        return
    if checkpoint_path:
        raise ValueError(
            'smoke_check mode forbids checkpoint_path (must start from scratch); '
            'disable smoke_check for 断点续跑 (handbook P25).'
        )


def _should_allow_auto_cuda(cfg: Mapping[str, Any]) -> bool:
    """判断当前配置是否允许 auto 模式自动选择 CUDA。

    Args:
        cfg: 流水线配置字典。

    Returns:
        quick 或 full 模式返回 True，否则 False。
    """
    # 判断 auto 模式是否允许自动选择 CUDA
    mode = _normalize_train_mode(cfg.get('mode'))  # 归一化训练模式（normalize_run_mode 已保证返回值在 ALLOWED_RUN_MODES 内，非法模式抛 ValueError）
    return mode in ALLOWED_RUN_MODES  # 所有合法训练模式（quick/full）均允许 auto CUDA，引用单源常量避免本地字面量漂移


# 模块内部别名，保持 _前缀 名称的向后兼容
_normalize_gt_rows = normalize_gt_rows
_align_ground_truth = align_ground_truth




def _load_ground_truth_rows_from_root(ground_truth_root, seq_id: str) -> list[dict[str, float]]:
    """从 ground_truth_root 目录加载真值行列表。

    读取 ground_truth_root/seq_id/gt.json 文件并归一化。

    Args:
        ground_truth_root: 真值根目录路径。
        seq_id: 序列 ID。

    Returns:
        归一化后的真值行列表。

    Raises:
        FileNotFoundError: gt.json 文件不存在。
    """
    validate_path_component(seq_id, name='seq_id')  # D8：校验 seq_id 不含路径穿越字符，防止目录逃逸。
    gt_path = Path(ground_truth_root) / seq_id / GT_FILE_NAME  # 按序列 ID 拼接真值文件路径（D9：使用单源常量）。
    if not gt_path.is_file():  # 真值文件不存在时报错。
        raise FileNotFoundError(f'ground-truth file not found: {gt_path}')
    return _normalize_gt_rows(read_json(gt_path))  # 读取并归一化真值行（D10：read_json 已提升到模块顶层）。


class _SeqIdNotFound(Exception):
    """直接输入模式下未找到 seq_id 的事件，作为 direct→miluv 回退链的控制流信号。

    与 ValueError（值合同违例，例如真值缺失）区分：本异常仅表示“直接输入中未找到
    该 seq_id”，应由 _resolve_sequence_payload 捕获并回退到数据集模式；真值缺失等
    配置错误仍以 ValueError 传播，避免静默回退掩盖用户配置错误（D7/D8 行为修复）。
    """


def _load_direct_sequence_payload(seq_id: str, cfg: Mapping[str, Any]) -> tuple[list[Any], list[dict[str, float]], dict[str, Any]]:
    """加载直接输入模式的序列数据（事件 + 真值）。

    从 cfg 中的 events_by_seq_id 或 events 字段获取事件列表，
    从 ground_truth_by_seq_id 或 ground_truth_root 获取真值。

    Args:
        seq_id: 序列 ID。
        cfg: 配置字典，须包含 events 和 ground_truth 相关字段。

    Returns:
        (events, gt_rows, source_report) 元组。

    Raises:
        _SeqIdNotFound: 找不到指定 seq_id 的事件（控制流信号，由 _resolve_sequence_payload 捕获回退到数据集模式）。
        ValueError: 缺少真值数据源（值合同违例，不回退，直接传播）。
    """
    events_by_seq_id = cfg.get('events_by_seq_id') or {}  # 按序列编号索引的事件映射。
    events = events_by_seq_id.get(seq_id)  # 查找当前序列的事件列表。
    if events is None and isinstance(cfg.get('events'), list):  # 回退到单一事件列表。
        split_ids = list(cfg.get('split_ids') or [])  # 获取切分 ID。
        if len(split_ids) == 1 and split_ids[0] == seq_id:  # 单序列时直接使用 events 字段。
            events = deepcopy(cfg['events'])  # 深拷贝单一事件列表，避免内部 dict 与 cfg 共享引用（D3）。
    if events is None:  # 事件列表仍然找不到时发控制流信号（非值合同违例，由上层 _resolve_sequence_payload 捕获并回退到数据集模式）。
        raise _SeqIdNotFound(f'no events found for seq_id={seq_id!r}; expected events_by_seq_id[{seq_id!r}] or events with matching split_ids, got none')

    ground_truth_by_seq_id = cfg.get('ground_truth_by_seq_id') or {}  # 按序列编号索引的真值映射。
    gt_rows = ground_truth_by_seq_id.get(seq_id)  # 查找当前序列的真值。
    if gt_rows is None and cfg.get('ground_truth_root'):  # 回退到磁盘真值目录。
        gt_rows = _load_ground_truth_rows_from_root(cfg['ground_truth_root'], seq_id)  # 从磁盘加载真值。
    elif gt_rows is not None:  # 内存中找到真值时先归一化。
        gt_rows = _normalize_gt_rows(gt_rows)  # 归一化真值行。
    else:  # 既没有内存真值也没有磁盘真值目录时报错。
        raise ValueError('ground_truth_by_seq_id or ground_truth_root is required for direct liquid training inputs')  # 缺少真值数据源。

    source_report_by_seq_id = cfg.get('source_report_by_seq_id') or {}  # 按序列编号索引的数据源报告。
    source_report = deepcopy(source_report_by_seq_id.get(seq_id) or {})  # 深拷贝数据源报告，避免嵌套结构与 cfg 共享引用（D3）。
    source_report.setdefault('source', 'direct_inputs')  # 默认来源标记为直接输入。
    return deepcopy(events), deepcopy(gt_rows), source_report  # 深拷贝事件与真值后返回，避免内部 dict 与上游引用共享（D3）。


def _load_miluv_sequence_payload(seq_id: str, cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, float]], dict[str, Any]]:
    """加载 MILUV 数据集的序列数据（事件 + 真值 + 锚点布局）。

    读取 MILUV 原始数据，执行字段映射和事件构建；
    锚点布局（含 experiments.csv + anchors.yaml 解析）由 dataio 层
    read_miluv_sequence 统一发现并放入 raw_bundle，本函数仅在 reader
    返回 3D 元数据时通过 project_anchor_layout_xy 投影到 2D。

    Args:
        seq_id: 序列 ID（实验名称）。
        cfg: 配置字典，须包含 raw_root 和 field_mapping。

    Returns:
        (events, gt_rows, source_report) 元组，
        source_report 包含锚点布局和元数据信息。

    Raises:
        ValueError: field_mapping 缺失或为空，或 raw_root 缺失。
    """
    field_mapping = cfg.get('field_mapping')
    if not isinstance(field_mapping, Mapping) or not field_mapping:  # MILUV 训练必须提供非空字段映射。
        raise ValueError('miluv liquid training requires a non-empty field_mapping')

    raw_root = cfg.get('raw_root')  # 取原始数据根目录。
    if not raw_root:  # raw_root 是 MILUV 训练必需配置，缺失时显式报 ValueError 而非 KeyError（D8 路由安全）。
        raise ValueError('miluv liquid training requires a non-empty raw_root')
    raw_bundle, read_report = read_miluv_sequence(seq_id, raw_root)  # 读取 MILUV 原始 bundle（含锚点布局发现，dataio 层为权威源）。
    internal_bundle, mapping_report = map_external_fields(raw_bundle, dict(field_mapping))  # 执行外部字段到内部字段的映射。
    events = merge_and_finalize_events(  # 构建并合并所有模态事件。
        [
            build_imu_events(internal_bundle.get('imu_raw', []), f'miluv:{seq_id}', seq_id),  # 构建 IMU 事件。
            build_uwb_events(internal_bundle.get('uwb_raw', []), f'miluv:{seq_id}', seq_id),  # 构建 UWB 事件。
            build_vio_events(internal_bundle.get('vio_raw', []), f'miluv:{seq_id}', seq_id),  # 构建 VIO 事件。
        ]
    )
    gt_rows = _normalize_gt_rows(internal_bundle.get('gt_raw', []))  # 归一化真值行。
    anchor_layout = raw_bundle.get('anchor_layout_raw')  # 从原始 bundle 获取锚点布局。
    anchor_layout_metadata = raw_bundle.get('anchor_layout_metadata_raw')  # 从原始 bundle 获取锚点布局元数据。
    anchor_layout_source = None  # 初始化锚点布局来源标识。
    anchor_layout_metadata_source = None  # 初始化锚点元数据来源标识。
    if anchor_layout is None and anchor_layout_metadata is not None:  # reader 返回 3D 元数据但未投影时，由 sensors 层投影到 2D；不在 pipeline 层重复解析 experiments.csv/anchors.yaml（D2 分层边界：锚点发现权威源在 dataio/readers/miluv_reader；D1 协议一致：避免 int/str 星座解析与 reader 分歧；D5 数值安全：投影内部已拒绝 NaN/Inf；D7 可复现：anchor_id 排序与归一化由 reader 统一处理）。
        anchor_layout = project_anchor_layout_xy(anchor_layout_metadata)  # 3D→2D 投影，内部已做有限性校验（拒绝 NaN/Inf/布尔坐标）。
    if isinstance(anchor_layout_metadata, Mapping):  # 从元数据中解析来源标识。
        anchor_layout_source = str(anchor_layout_metadata.get('source') or '') or None  # 提取来源标识。
        if anchor_layout_source == 'fixture_local_anchor_layout':  # 归一化 fixture 来源名称。
            anchor_layout_source = 'miluv_fixture_anchor_layout'  # 重命名为 MILUV 专用标识。
        anchor_layout_metadata_source = anchor_layout_source  # 元数据来源与布局来源一致。
    anchor_layout_position_dim = (  # 解析位置维度，优先用 read_report，其次从元数据推断。
        read_report.get('anchor_layout_position_dim')
        or _resolve_anchor_layout_position_dim(anchor_layout_metadata)
        or _resolve_anchor_layout_position_dim(anchor_layout)
    )
    return events, gt_rows, {  # 返回事件、真值和数据源报告。
        'source': 'miluv_raw',  # 来源标识为 MILUV 原始数据。
        'read_report': deepcopy(read_report),  # 读取报告深拷贝，断开与 reader 内部引用，避免调用方修改污染（D3 数据合同保真）。
        'mapping_report': deepcopy(mapping_report),  # 映射报告深拷贝。
        'anchor_layout': deepcopy(anchor_layout),  # 锚点布局深拷贝（可能为 None，deepcopy(None)=None）。
        'anchor_layout_source': anchor_layout_source,  # 锚点布局来源标识。
        'anchor_layout_metadata': deepcopy(anchor_layout_metadata),  # 锚点布局元数据深拷贝（3D 原始信息）。
        'anchor_layout_metadata_source': anchor_layout_metadata_source or read_report.get('anchor_layout_metadata_source'),  # 元数据来源标识。
        'anchor_layout_teacher_ready': bool(anchor_layout is not None or read_report.get('anchor_layout_teacher_ready')),  # 布局是否可供 teacher 使用。
        'anchor_layout_blockers': list(read_report.get('anchor_layout_blockers') or []),  # 布局解析阻塞原因列表。
        'anchor_layout_position_dim': anchor_layout_position_dim,  # 位置维度（2 或 3）。
    }


def _extract_teacher_projection_audit(anchor_layout: Any) -> dict[str, Any]:
    """提取锚点布局中的投影审计信息。

    Args:
        anchor_layout: 锚点布局字典。

    Returns:
        包含 _TEACHER_PROJECTION_AUDIT_KEYS 中存在的键值对的字典。
    """
    if not isinstance(anchor_layout, Mapping):  # 非映射型布局无法提取审计信息。
        return {}
    projection_audit: dict[str, Any] = {}  # 收集投影审计字段。
    for key in _TEACHER_PROJECTION_AUDIT_KEYS:  # 只提取预定义的审计键。
        if key in anchor_layout:  # 键存在时复制到审计字典。
            # D3：深拷贝隔离，避免审计字典与原 anchor_layout 共享可变引用（list/dict），
            # 当前审计键值均为 int/str 标量，但与 L2563 estimator_cfg 同口径保持防御性一致。
            projection_audit[key] = deepcopy(anchor_layout[key])
    return projection_audit


def _resolve_anchor_layout_position_dim(anchor_layout: Any) -> int | None:
    """解析锚点布局的位置维度（2D 或 3D）。

    优先使用 position_dim 字段，其次从 anchor_positions 推断。

    Args:
        anchor_layout: 锚点布局字典。

    Returns:
        位置维度（2 或 3），无法确定时返回 None。
    """
    if not isinstance(anchor_layout, Mapping):  # 非映射型布局无法推断维度。
        return None
    declared_dim = anchor_layout.get('position_dim')  # 优先使用显式声明的维度字段。
    if declared_dim is not None and not is_bool_like(declared_dim):  # 有显式声明且非布尔时尝试解析（bool 是 int 子类，int(True)=1 会误判为 dim=1，D3 数据合同保真，与 is_bool_like 排除口径一致）。
        try:  # 尝试转为整数。
            resolved_dim = int(declared_dim)
        except (TypeError, ValueError, OverflowError):  # 转换失败时置空；OverflowError 对应 int(float('inf'))，与 coerce_finite_scalar except 口径一致（D5 数值安全）。
            resolved_dim = None
        else:  # 转换成功时检查有效性。
            if resolved_dim > 0:  # 正整数维度才有效。
                return resolved_dim

    raw_positions = anchor_layout.get('anchor_positions')  # 从锚点坐标推断维度。
    if isinstance(raw_positions, (str, bytes)):  # 字符串/字节无法推断。
        return None
    try:  # 尝试转为列表。
        anchor_positions = list(raw_positions or [])
    except TypeError:  # 不可迭代时无法推断。
        return None
    if not anchor_positions:  # 空坐标列表无法推断。
        return None

    first_position = anchor_positions[0]  # 取第一个坐标推断维度。
    if isinstance(first_position, (str, bytes)):  # 字符串/字节无法推断。
        return None
    try:  # 从坐标元素数量推断维度。
        return len(list(first_position))  # 维度 = 坐标分量数。
    except TypeError:  # 不可迭代时无法推断。
        return None


def _coerce_teacher_ready_anchor_layout(
    anchor_layout: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """将锚点布局强制转换为 teacher 可用的 2D 格式。

    Args:
        anchor_layout: 原始锚点布局。

    Returns:
        (resolved_layout, report) 元组。
        resolved_layout 为 2D 锚点布局，或 None（无法转换）。
        report 包含 position_dim 和 blockers 信息。
    """

    if not isinstance(anchor_layout, Mapping):
        return None, {
            'position_dim': None,
            'blockers': ['anchor_layout_mapping_required'],
        }

    position_dim = _resolve_anchor_layout_position_dim(anchor_layout)  # 解析位置维度
    if position_dim == 2:  # 已经是 2D，teacher 可直接使用
        return deepcopy(anchor_layout), {  # 深拷贝隔离嵌套结构（anchor_positions/anchor_ids 等），避免调用方修改污染原 anchor_layout（D3 数据合同保真）
            'position_dim': 2,
            'blockers': [],
        }
    if position_dim is None:  # 维度无法确定
        return None, {  # 无法判定维度，返回 None
            'position_dim': None,
            'blockers': ['anchor_layout_position_dim_unresolved'],
        }
    return None, {  # 维度非 2（如 1D/3D），需要投影到 2D
        'position_dim': position_dim,
        'blockers': [f'anchor_layout_position_dim_{position_dim}_requires_2d_teacher'],
    }


def _materialize_teacher_anchor_layout_from_source_report(
    source_report: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """从 source_report 中物化 teacher 可用的锚点布局。

    优先使用直接的 anchor_layout，其次从 3D 元数据投影为 2D。

    Args:
        source_report: 数据源报告字典。

    Returns:
        (anchor_layout, report) 元组。
    """

    metadata_available = bool(source_report.get('anchor_layout_metadata') is not None)
    metadata_source = source_report.get('anchor_layout_metadata_source')
    metadata_position_dim = source_report.get('anchor_layout_position_dim')  # 获取元数据位置维度
    teacher_input_position_dim: int | None = None  # teacher 输入位置维度
    direct_layout_blockers: list[str] = []  # 直接布局的阻塞原因

    source_anchor_layout = source_report.get('anchor_layout')  # 获取直接锚点布局
    if isinstance(source_anchor_layout, Mapping):  # 尝试直接布局强制转换
        resolved_anchor_layout, direct_layout_report = _coerce_teacher_ready_anchor_layout(source_anchor_layout)  # 强制转换为 2D
        teacher_input_position_dim = direct_layout_report.get('position_dim')  # 获取位置维度
        if resolved_anchor_layout is not None:  # 直接布局已满足 teacher 要求
            return resolved_anchor_layout, {  # 返回已解析的布局
                'source': str(source_report.get('anchor_layout_source') or 'sequence_anchor_layout'),
                # metadata_available/metadata_source 报告元数据事实，不与 layout source 混淆；
                # 与 _resolve_sequence_anchor_layout 显式布局分支（L909-910）保持一致，避免 or True 恒真与回退污染。
                'metadata_available': metadata_available,
                'metadata_source': metadata_source,
                'metadata_position_dim': metadata_position_dim,
                'teacher_input_position_dim': teacher_input_position_dim,
                'blockers': [],
            }
        direct_layout_blockers = list(direct_layout_report.get('blockers') or [])  # 记录阻塞原因

    anchor_layout_metadata = source_report.get('anchor_layout_metadata')  # 尝试从元数据投影
    if (  # 检查元数据是否为 MILUV 官方 3D
        isinstance(anchor_layout_metadata, Mapping)
        and metadata_source == MILUV_OFFICIAL_ANCHOR_METADATA_SOURCE
        and metadata_position_dim == 3
    ):
        teacher_anchor_layout = project_anchor_layout_xy(anchor_layout_metadata)  # 将 3D 投影到 2D
        if teacher_anchor_layout is not None:  # 投影成功
            return teacher_anchor_layout, {  # 返回投影后的布局
                'source': MILUV_OFFICIAL_ANCHOR_METADATA_SOURCE,
                'metadata_available': True,
                'metadata_source': metadata_source,
                'metadata_position_dim': metadata_position_dim,
                'teacher_input_position_dim': _resolve_anchor_layout_position_dim(teacher_anchor_layout),
                'blockers': [],
            }

    return None, {  # 无可用布局
        'source': None,
        'metadata_available': metadata_available,
        'metadata_source': metadata_source,
        'metadata_position_dim': metadata_position_dim,
        'teacher_input_position_dim': teacher_input_position_dim,
        'blockers': direct_layout_blockers + list(source_report.get('anchor_layout_blockers') or []),
    }


def _resolve_sequence_payload(seq_id: str, cfg: Mapping[str, Any]) -> tuple[list[Any], list[dict[str, float]], dict[str, Any]]:
    """解析序列的数据载荷，自动选择直接输入或 MILUV 数据集。

    Args:
        seq_id: 序列 ID。
        cfg: 配置字典。

    Returns:
        (events, gt_rows, source_report) 元组。

    Raises:
        ValueError: 直接输入真值缺失，或 MILUV 配置不完整（field_mapping/raw_root 缺失），
            或既无直接输入也不是 MILUV 数据集。
    """
    try:
        return _load_direct_sequence_payload(seq_id, cfg)
    except _SeqIdNotFound:  # 直接输入未找到 seq_id（控制流信号）；真值缺失等 ValueError 不被捕获，直接传播（D7/D8 行为修复：避免静默回退掩盖配置错误）
        pass  # 继续尝试数据集模式

    dataset_name = cfg.get('dataset_name')  # 获取数据集名称
    if dataset_name == DATASET_NAME_MILUV:  # MILUV 数据集（单源常量，D9 漂移根因修复）
        return _load_miluv_sequence_payload(seq_id, cfg)  # 加载 MILUV 数据

    raise ValueError(  # 无有效数据源
        'liquid real training requires either direct events/ground-truth inputs or '
        "dataset_name='miluv' with raw_root and field_mapping"
    )


def _describe_geometry_bias_teacher(
    anchor_layout: Any,
    *,
    source: str | None = None,
    metadata_available: bool = False,
    metadata_source: str | None = None,
    blockers: list[str] | None = None,
    metadata_position_dim: int | None = None,
    teacher_input_position_dim: int | None = None,
) -> dict[str, Any]:
    """描述几何 bias teacher 的可用性和配置状态。

    Args:
        anchor_layout: 锚点布局字典，可为 None。
        source: 锚点布局来源标识。
        metadata_available: 元数据是否可用。
        metadata_source: 元数据来源。
        blockers: 阻塞原因列表。
        metadata_position_dim: 元数据中的位置维度。
        teacher_input_position_dim: teacher 输入的位置维度。

    Returns:
        包含 status/available/observed_inputs/missing_fields 等字段的字典。
    """

    missing_fields: list[str] = []
    if anchor_layout is None:
        missing_fields = ['estimator_cfg.anchor_layout.anchor_ids', 'estimator_cfg.anchor_layout.anchor_positions']  # 两个字段均缺失
    elif not isinstance(anchor_layout, Mapping):  # 不是映射类型
        missing_fields = ['estimator_cfg.anchor_layout']  # 布局本身缺失
    else:  # 检查各个字段
        if 'anchor_ids' not in anchor_layout:
            missing_fields.append('estimator_cfg.anchor_layout.anchor_ids')
        if 'anchor_positions' not in anchor_layout:
            missing_fields.append('estimator_cfg.anchor_layout.anchor_positions')
    return {  # 返回几何 bias teacher 描述
        'status': 'fallback_teacher_only' if missing_fields else 'geometric_teacher_available',
        'available': not missing_fields,
        'source': source,
        'metadata_available': metadata_available or anchor_layout is not None,
        'metadata_source': metadata_source or source,
        'metadata_position_dim': metadata_position_dim,
        'teacher_input_position_dim': (
            teacher_input_position_dim
            if teacher_input_position_dim is not None
            else _resolve_anchor_layout_position_dim(anchor_layout)
        ),
        'blockers': list(blockers or []),
        **_extract_teacher_projection_audit(anchor_layout),
        'observed_inputs': {
            'measured_range': True,
            'anchor_id': True,
            'aligned_ground_truth_pose': True,
            'anchor_position': not missing_fields,
            'geometric_true_range': not missing_fields,
        },
        'missing_fields': missing_fields,
        'formula_if_enabled': 'bias = max(0.0, measured_range - geometric_true_range)',
    }


def _resolve_sequence_anchor_layout(
    estimator_cfg: Mapping[str, Any],
    source_report: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """解析序列的锚点布局，优先使用 estimator_cfg 中的显式配置。

    Args:
        estimator_cfg: 估计器配置字典。
        source_report: 数据源报告字典。

    Returns:
        (anchor_layout, report) 元组。
    """

    explicit_anchor_layout = estimator_cfg.get('anchor_layout')  # 直接读取字段，避免 dict() 浅拷贝整个 estimator_cfg 只为读一个字段（D3 数据合同保真）；嵌套引用隔离由 _coerce_teacher_ready_anchor_layout 内部 deepcopy（L734）保证。
    if explicit_anchor_layout is not None:
        resolved_layout, explicit_layout_report = _coerce_teacher_ready_anchor_layout(explicit_anchor_layout)  # 强制转换为 2D
        return resolved_layout, {  # 返回显式布局
            'source': 'estimator_cfg.anchor_layout',
            'metadata_available': bool(source_report.get('anchor_layout_metadata') is not None),
            'metadata_source': source_report.get('anchor_layout_metadata_source'),
            'metadata_position_dim': source_report.get('anchor_layout_position_dim'),
            'teacher_input_position_dim': explicit_layout_report.get('position_dim'),
            'blockers': list(explicit_layout_report.get('blockers') or []),
        }
    return _materialize_teacher_anchor_layout_from_source_report(source_report)  # 回退到 source_report 物化



def _build_geometric_bias_target(
    event: Mapping[str, Any],
    gt_state: Mapping[str, float],
    anchor_lookup: Mapping[Any, tuple[float, float]],
    *,
    anchor_layout: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """构建几何 bias teacher 目标值。

    计算 bias = max(0.0, measured_range - geometric_true_range)，
    即测量距离减去几何真实距离的正部分。

    Args:
        event: UWB 事件字典。
        gt_state: 对齐后的真值状态字典。
        anchor_lookup: 锚点查找表。
        anchor_layout: 锚点布局字典，用于审计。

    Returns:
        (clamped_bias, trace) 元组，trace 包含完整的审计信息。
    """
    measurement = extract_uwb_measurement(event)
    anchor_id = measurement['anchor_id']
    anchor_position = resolve_anchor_position(anchor_id, anchor_lookup)  # 解析锚点位置
    geometric_true_range = predict_range_to_anchor(gt_state, anchor_position)  # 计算几何真实距离
    measured_range = coerce_finite_scalar(measurement['measured_range'], name='measured_range')  # D5：防御性有限性校验
    raw_bias = measured_range - geometric_true_range  # 计算原始 bias
    clamped_bias = max(0.0, raw_bias)  # 截断为非负值
    if measured_range > 0.0 and clamped_bias > measured_range * UWB_BIAS_MAX_RATIO:
        clamped_bias = measured_range * UWB_BIAS_MAX_RATIO
    return clamped_bias, {
        'anchor_id': anchor_id,
        'anchor_position': [coerce_finite_scalar(anchor_position[0], name='anchor_position[0]'), coerce_finite_scalar(anchor_position[1], name='anchor_position[1]')],
        'measured_range': measured_range,
        'geometric_true_range': coerce_finite_scalar(geometric_true_range, name='geometric_true_range'),
        'raw_bias': coerce_finite_scalar(raw_bias, name='raw_bias'),
        'clamped_bias': coerce_finite_scalar(clamped_bias, name='clamped_bias'),
        **_extract_teacher_projection_audit(anchor_layout),
    }


# 违反项 12 修复: 删除 _maybe_inject_nlos_for_training 函数定义（已无调用者），防止未来误调用注入 N3 引入训练/测试不对称。
# 原函数在 train_pipeline.py 内部强制注入 N3（nlos_ratio=0.37, μ=5.0m, σ=1.5m），
# 与 sim_materializer 已统一注入的 N1.5（ρ=0.225, μ=2.0m, σ=0.5m）分布不对称，
# bias 头学的是 N3 而非测试 N1.5。已删除该函数定义，仅保留此注释作为审计痕迹。


def _snapshot_proxy_estimator_state(estimator: Any) -> dict[str, float]:
    """提取训练样本构造阶段可复用的代理估计器状态快照。"""
    if not hasattr(estimator, 'get_state'):  # 检查估计器是否有 get_state 方法
        return {}  # 无 get_state 方法，返回空字典
    state_estimate = estimator.get_state()  # 从估计器获取状态估计
    state_payload = getattr(state_estimate, 'state', None)  # 提取 state 载荷
    if not isinstance(state_payload, Mapping):  # 检查 state 载荷是否为映射类型
        return {}  # 不是映射类型，返回空字典
    snapshot: dict[str, float] = {}  # 初始化状态快照字典
    for key, value in state_payload.items():  # 遍历 state 载荷项（state_payload 已是 Mapping，无需 dict() 物化）
        try:
            # coerce_finite_scalar 统一完成 float() + isfinite + 拒绝 NaN/Inf/非数值（D5 数值安全根因修复，对齐 Round 1-4 模式）。
            scalar_value = coerce_finite_scalar(value, name=f"state[{key!r}]")
        except (TypeError, ValueError):
            continue  # 跳过非数值或非有限值（TypeError=非数值/bool，ValueError=NaN/Inf/越界）
        snapshot[str(key)] = scalar_value  # 添加到快照
    return snapshot  # 返回状态快照


def _advance_proxy_estimator_for_training_sample(
    estimator: Any,
    event: Mapping[str, Any],
    *,
    allow_uwb_update: bool,
) -> dict[str, Any] | None:
    """按真实事件顺序推进训练样本代理估计器，保持 prediction_state 因果语义。"""
    if not hasattr(estimator, 'step'):
        return None
    modality = event.get('modality')  # 获取事件模态（保持原始类型，避免 str(None) 静默转换为 'None'）
    if modality == MODALITY_UWB and (not allow_uwb_update or not _resolve_event_valid_flag(event)):  # 不允许或无效 UWB 更新
        return {  # 返回 UWB 跳过报告
            'modality': MODALITY_UWB,
            'update_applied': False,
            'reason': f'{MODALITY_UWB}_skip_update',  # 与 _update_training_readout_context_cache 的 expected_skip_reason 同源
        }
    estimator.step(event)  # 用事件推进估计器
    return getattr(estimator, 'last_update_report', None)  # 返回上次更新报告


def _init_training_readout_context_cache() -> dict[str, dict[str, float | bool | None]]:
    """初始化训练 readout 上下文缓存，按模态存储新息和跳过标志。

    Returns:
        按 modality 索引的缓存字典。
    """
    cache: dict[str, dict[str, float | bool | None]] = {}
    for modality in _READOUT_CONTEXT_MODALITIES:
        # 键名与默认值来自单源常量 _READOUT_CONTEXT_CACHE_DEFAULTS，与推理侧
        # fusion_runner._init_readout_context_cache 同口径（D9 单源 + D7 公平性）。
        # dict() 浅拷贝足够：值均为不可变（0.0/False/None），无共享引用风险。
        cache[modality] = dict(_READOUT_CONTEXT_CACHE_DEFAULTS)
    return cache  # 返回初始化后的缓存


def _coerce_training_residual_norm(residual: Any, *, modality: str) -> float | None:
    """将训练残差强制转换为范数（标量或 L2 范数）。

    Args:
        residual: 残差值，可为标量或向量。
        modality: 模态名称，uwb 按标量处理，其他按向量处理。

    Returns:
        残差范数，无效时返回 None。
    """
    if modality == MODALITY_UWB:
        # UWB 残差是标量，取绝对值。统一走 coerce_finite_scalar 完成 float()+isfinite+
        # 拒绝 bool/NaN/Inf（D5 数值安全，与 _resolve_bridge_thresholds 同口径；D9 单源常量）。
        # 本函数对无效值返回 None（不抛错），用 try/except 兜底；coerce_finite_scalar 仅抛
        # TypeError/ValueError，覆盖原 float() 失败路径（require_not_none 对 None 抛 ValueError）。
        try:
            return abs(coerce_finite_scalar(residual, name='residual'))  # abs(有限)=有限，无需再校验
        except (TypeError, ValueError):  # 转换失败或非有限
            return None  # 无效残差
    # 其他模态残差按向量处理，计算 L2 范数。
    try:
        # coerce_finite_scalar 已保证每个元素有限，等价于原 float()+isfinite 两段式（D5）。
        values = [coerce_finite_scalar(value, name='residual') for value in list(residual)]
    except (TypeError, ValueError):  # 转换失败或 residual 不可迭代
        return None  # 无效残差
    if not values:  # 空残差
        return None  # 无效残差
    # 各元素已有限，但平方和可能溢出为 inf（如大数平方），需对最终范数做有限性校验（D5）。
    norm = math.sqrt(math.fsum(value * value for value in values))  # L2 范数，fsum 提高平方和精度
    return norm if math.isfinite(norm) else None  # inf 视为不可用


def _update_training_readout_context_cache(
    cache: dict[str, dict[str, float | bool | None]],
    report: Any,
    *,
    current_timestamp: float | None = None,
) -> None:
    """更新训练 readout 上下文缓存。

    根据估计器更新报告，更新对应模态的新息范数和跳过标志。

    Args:
        cache: readout 上下文缓存字典。
        report: 估计器更新报告。
        current_timestamp: 当前事件的时间戳，记录为上次成功更新的时间。
    """
    if not isinstance(report, Mapping):
        return
    modality = report.get('modality')  # 保持原始类型，避免 str(None or '') 静默为 ''，None not in cache 自然跳过
    if modality not in cache:  # 模态不在缓存中
        return  # 跳过更新
    modality_cache = cache[modality]  # 获取模态缓存
    reason = report.get('reason')  # 保持原始类型，None 与 f-string 比较自然为 False，无需 str() 兜底
    residual = report.get('residual')  # 获取残差
    if bool(report.get('update_applied')) and residual is not None:  # 更新已应用
        residual_norm = _coerce_training_residual_norm(residual, modality=modality)  # 计算残差范数
        if residual_norm is not None:  # 残差范数有效
            modality_cache['last_innovation_norm'] = residual_norm  # 更新新息范数
            modality_cache['innovation_observed'] = True  # 标记新息已观测
            modality_cache['last_gate_skip_flag'] = 0.0  # 重置跳过标志
            modality_cache['skip_flag_observed'] = True  # 标记跳过标志已观测
            modality_cache['consecutive_skip_count'] = 0.0  # 重置连续跳过计数
            modality_cache['consecutive_skip_observed'] = True  # 标记连续跳过已观测
            modality_cache['last_successful_update_timestamp'] = current_timestamp  # 记录成功更新时间戳
        return  # 处理完毕
    expected_skip_reason = f'{modality}_skip_update'  # 期望的跳过原因模式
    if reason == expected_skip_reason:  # 匹配跳过原因
        modality_cache['last_gate_skip_flag'] = 1.0  # 设置跳过标志
        modality_cache['skip_flag_observed'] = True  # 标记跳过标志已观测
        modality_cache['consecutive_skip_count'] = float(modality_cache.get('consecutive_skip_count', 0.0) or 0.0) + 1.0
        modality_cache['consecutive_skip_observed'] = True  # 标记连续跳过已观测
    elif not bool(report.get('update_applied')) and isinstance(report.get('gate'), Mapping) and report.get('gate', {}).get('passed') is False:
        # 门控未通过（非显式 skip_update 原因），同样计入跳过
        modality_cache['last_gate_skip_flag'] = 1.0
        modality_cache['skip_flag_observed'] = True
        modality_cache['consecutive_skip_count'] = float(modality_cache.get('consecutive_skip_count', 0.0) or 0.0) + 1.0
        modality_cache['consecutive_skip_observed'] = True  # 标记连续跳过已观测


def _build_training_readout_context(
    estimator: Any,
    modality_cache: dict[str, dict[str, float | bool | None]],
    *,
    modality: str,
    current_timestamp: float | None = None,
) -> tuple[dict[str, float], dict[str, bool]]:
    """构建训练 readout 上下文，从估计器状态和模态缓存中提取。

    Args:
        estimator: 估计器实例。
        modality_cache: 模态缓存字典。
        modality: 当前模态名称。

    Returns:
        (values, observed) 元组，values 为数值字典，observed 为是否观测到的标志字典。
    """
    values = {key: 0.0 for key in _READOUT_CONTEXT_KEYS}
    observed = {key: False for key in _READOUT_CONTEXT_KEYS}
    if hasattr(estimator, 'get_state'):  # 估计器有 get_state 方法
        state_estimate = estimator.get_state()  # 获取状态估计
        # 显式 None 检查：避免对 numpy 数组等多元素对象调用 bool() 触发歧义异常
        cov_diag_attr = getattr(state_estimate, 'covariance_diag', None)
        covariance_diag = list(cov_diag_attr) if cov_diag_attr is not None else []  # 获取协方差对角线
        if len(covariance_diag) >= 2:  # 至少有 2 个对角线元素
            # D5/D8 数值安全：统一转换+异常守卫，避免 float() 对 None/字符串抛未捕获异常中断构建。
            # 一次性转换避免重复 float() 调用（原检查/求迹/取位置方差各转一次）。
            try:
                diag_floats = [float(value) for value in covariance_diag]
            except (TypeError, ValueError):
                diag_floats = None
            if diag_floats is not None:
                if all(math.isfinite(d) for d in diag_floats):  # 所有值有限
                    values['state_cov_trace'] = math.fsum(diag_floats)  # 协方差迹，用 fsum 提高精度
                    observed['state_cov_trace'] = True  # 标记已观测
                    # D2-5 v2 redesign: 位置/速度协方差迹让 filter_context 区分两种不确定性
                    values['pos_cov_trace'] = max(diag_floats[0], 0.0) + max(diag_floats[1], 0.0)
                    observed['pos_cov_trace'] = True
                    if len(diag_floats) >= 4:
                        values['vel_cov_trace'] = max(diag_floats[2], 0.0) + max(diag_floats[3], 0.0)
                        observed['vel_cov_trace'] = True
                # pos_cov 仅依赖前两个元素，尾部元素非有限不影响 pos_cov 计算
                # （对齐 test_..._keeps_pos_cov_when_tail_covariance_entries_are_non_finite）。
                px_var = max(diag_floats[0], 0.0)  # x 位置方差
                py_var = max(diag_floats[1], 0.0)  # y 位置方差
                if math.isfinite(px_var) and math.isfinite(py_var):  # 两者均有限
                    values['pos_cov'] = math.sqrt(px_var + py_var)  # 位置协方差
                    observed['pos_cov'] = True  # 标记已观测
    modality_state = modality_cache.get(modality)  # 获取模态缓存
    if modality_state is None:  # 该模态无缓存
        return values, observed  # 返回默认值
    # D5 数值安全：缓存值用 coerce_finite_scalar 统一转换+有限性校验，
    # 避免 float() 对 None/字符串/inf/nan 静默通过；任何异常跳过该字段保持默认值。
    if bool(modality_state.get('innovation_observed', False)):  # 新息已观测
        try:
            values['last_innovation_norm'] = coerce_finite_scalar(
                modality_state['last_innovation_norm'], name='last_innovation_norm')
            observed['last_innovation_norm'] = True  # 标记已观测
        except (TypeError, ValueError, KeyError):
            pass
    if bool(modality_state.get('skip_flag_observed', False)):  # 跳过标志已观测
        try:
            skip_value = coerce_finite_scalar(
                modality_state['last_gate_skip_flag'], name='last_gate_skip_flag')
            values['last_gate_skip_flag'] = skip_value
            observed['last_gate_skip_flag'] = True  # 标记已观测
            # D2-5 v2 redesign: last_update_skip_flag 与 last_gate_skip_flag 同义，
            # 为 VIO filter context 子集提供"上次更新是否跳过"信号。
            values['last_update_skip_flag'] = skip_value
            observed['last_update_skip_flag'] = True
        except (TypeError, ValueError, KeyError):
            pass
    if bool(modality_state.get('consecutive_skip_observed', False)):
        try:
            # D9 移除冗余 `or 0.0`：observed 标志已守卫键被观测过，
            # None/缺失应触发异常而非静默掩为 0.0（掩盖缓存不一致）。
            values['consecutive_skip_count'] = coerce_finite_scalar(
                modality_state['consecutive_skip_count'], name='consecutive_skip_count')
            observed['consecutive_skip_count'] = True
        except (TypeError, ValueError, KeyError):
            pass
    last_successful_update_timestamp = modality_state.get('last_successful_update_timestamp')
    if last_successful_update_timestamp is not None and current_timestamp is not None:
        try:
            values['time_since_last_update'] = max(
                0.0,
                coerce_finite_scalar(current_timestamp, name='current_timestamp')
                - coerce_finite_scalar(last_successful_update_timestamp, name='last_successful_update_timestamp'),
            )
            observed['time_since_last_update'] = True
        except (TypeError, ValueError):
            pass
    return values, observed  # 返回 readout 上下文


def _apply_robust_teacher_supplement(
    target_intermediate: dict[str, float],
    target_trace: dict[str, Any],
    *,
    event: Mapping[str, Any],
    bridge_thresholds: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    """应用 robust teacher 补充：在低质量+高风险条件下增强 scaling。

    当观测质量低且对齐风险高时，额外增大对应模态的 scaling，
    使训练器对这些样本给予更多关注。

    Args:
        target_intermediate: 目标中间值字典。
        target_trace: 目标审计 trace 字典。
        event: 当前事件。
        bridge_thresholds: 桥接阈值字典。

    Returns:
        (robust_intermediate, robust_trace) 元组。
    """

    observation_signal = _resolve_modality_observation_signal(event)
    quality_score = observation_signal['quality_score']  # 质量得分（0~1）
    quality_risk = observation_signal['quality_risk']  # 质量风险（1 - quality_score）
    modality_signal = observation_signal['modality_signal']  # 模态特定信号
    observation_risk = observation_signal['observation_risk']  # 观测风险
    # D8：不使用 str() 静默转换；modality 缺失或非字符串时与 MODALITY_* 比较为 False，行为正确。
    modality = event.get('modality')  # 事件模态
    geometry_risk = 0.0  # 初始化几何风险
    if modality == MODALITY_UWB:  # UWB：包含几何 bias 风险
        # D5：用 coerce_finite_scalar 统一数值安全，替代 float(... or 0.0) + isfinite 两段式；
        # 原写法对 'abc' 等非数值字符串会抛未捕获 ValueError，且 `or 0.0` 会误吃 falsy 字符串。
        try:
            geometry_risk = coerce_finite_scalar(
                target_trace.get('uwb_geometry_risk'),
                name='uwb_geometry_risk',
            )
        except (TypeError, ValueError):
            geometry_risk = 0.0  # 缺失/非数值/非有限值统一回退为 0

    effective_observation_risk = max(observation_risk, geometry_risk)  # 有效观测风险
    # bridge_thresholds 由 _resolve_bridge_thresholds 上游逐键 coerce_finite_scalar 校验，
    # target_trace/target_intermediate 的 alignment_risk/observation_risk/risk 在上游均为有限浮点，
    # 此处无需重复 float()（D5：移除冗余防御性转换）。
    low_quality = effective_observation_risk >= bridge_thresholds['robust_supplement_quality_threshold']  # 检查低质量条件
    high_risk = max(  # 检查高风险条件
        target_trace.get('alignment_risk', 0.0),
        target_trace.get('observation_risk', 0.0),
        target_intermediate.get('risk', 0.0),
    ) >= bridge_thresholds['robust_supplement_alignment_threshold']
    # D7：补充强度衰减系数提取为单源常量 _ROBUST_SUPPLEMENT_STRENGTH_FACTOR，替代硬编码 0.5。
    supplement_strength = max(quality_risk, modality_signal, geometry_risk) * _ROBUST_SUPPLEMENT_STRENGTH_FACTOR  # 计算补充强度
    supplement_strength = min(  # 限制补充强度不超过最大增量
        bridge_thresholds['robust_supplement_max_boost'],
        max(0.0, supplement_strength),
    )
    if not (low_quality and high_risk and supplement_strength > 0.0):  # 无需补充
        return target_intermediate, target_trace  # 返回未修改的值

    robust_intermediate = dict(target_intermediate)  # 复制目标中间值（值为 float 不可变，浅拷贝安全）
    if modality == MODALITY_UWB:  # UWB：增强 UWB scaling
        robust_intermediate['uwb_scaling'] = max(robust_intermediate['uwb_scaling'], 1.0 + supplement_strength)  # 提升 UWB scaling
    elif modality == MODALITY_VIO:  # 增强 VIO scaling
        robust_intermediate['vio_scaling'] = max(robust_intermediate['vio_scaling'], 1.0 + supplement_strength)  # 提升 VIO scaling

    # D3：target_trace 含嵌套结构（prediction_state dict、geometry_bias_trace 展开项等），
    # dict() 浅拷贝会共享嵌套对象，后续改动会污染原 trace；用 deepcopy 隔离。
    robust_trace = deepcopy(target_trace)  # 深拷贝目标 trace
    robust_trace['confidence_source'] = 'modality_specific_scaling_with_neutral_floor+robust_low_quality_supplement'  # 更新置信度来源
    robust_trace['robust_teacher_supplement'] = 'active'  # 标记补充已激活
    robust_trace['robust_teacher_quality_score'] = quality_score  # 记录质量得分
    robust_trace['robust_teacher_supplement_strength'] = supplement_strength  # 记录补充强度
    return robust_intermediate, robust_trace  # 返回 robust 结果

def _resolve_event_payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """解析事件的 payload 字典。

    Args:
        event: 事件字典。

    Returns:
        对应模态的 payload 字典，不存在时返回空字典。
    """

    modality = event.get('modality', 'unknown')
    # .get() 在 key 不存在时返回 None，无需冗余 `in` 检查（D8 代码清晰性）。
    payload_key = PAYLOAD_KEYS.get(modality)
    # 用 `is not None` 精确判断，避免空字符串等 falsy 值的语义混淆（D8 精确性）。
    payload = event.get(payload_key) if payload_key is not None else None
    if isinstance(payload, Mapping):
        return payload  # 返回 payload
    return {}  # 无 payload，返回空字典


def _resolve_event_field(event: Mapping[str, Any], field_name: str) -> Any | None:
    """从事件中解析指定字段，仅从 payload 中获取，与推理侧一致。

    推理侧（liquid_bridge_contract.py）仅从 uwb_payload/vio_payload 中
    读取字段，不回退到 event 顶层。训练侧也必须遵循相同路径，否则当字段
    仅存在于 event 顶层时，训练侧能读到而推理侧读不到，导致训练-推理不一致。

    Args:
        event: 事件字典。
        field_name: 字段名称。

    Returns:
        字段值，payload 中不存在或值为 None 时返回 None。
    """

    payload = _resolve_event_payload(event)
    if field_name in payload and payload[field_name] is not None:
        return payload[field_name]  # 从 payload 中返回字段值
    return None  # 不回退到 event 顶层，与推理侧一致


def _resolve_event_valid_flag(event: Mapping[str, Any]) -> bool:
    """解析事件的有效标志。

    与推理侧 ``_resolve_uwb_valid_flag``（liquid_bridge_contract.py）语义一致：
    仅 bool-like False（含 numpy.bool_(False)）视为无效，0/空字符串/None 等
    非 bool-like 值均视为有效。缺失时默认为 True。

    Args:
        event: 事件字典。

    Returns:
        有效标志布尔值，未设置时默认为 True。
    """

    value = _resolve_event_field(event, 'valid')
    if value is None:
        return True  # 默认为有效
    # 与推理侧 _resolve_uwb_valid_flag 一致：仅 bool-like False 视为无效。
    # 必须用 is_bool_like 覆盖 numpy.bool_，因为 np.bool_(False) is False 为 False，
    # 直接用 `is False` 会导致训练-推理在 numpy bool 数据上语义分叉。
    if is_bool_like(value):
        return bool(value)
    return True  # 非 bool-like 值（0, "", None在payload中等）均视为有效


def _resolve_event_quality_score(event: Mapping[str, Any]) -> float:
    """解析事件的质量得分，归一化到 0~1 范围。

    Args:
        event: 事件字典。

    Returns:
        质量得分（0~1），1 表示最高质量。
    """

    quality = _resolve_event_field(event, 'quality')
    # 不做 reproj_err fallback：推理侧 build_measurement_control 在 quality 缺失时
    # 默认 1.0，训练侧必须一致。reproj_err 在推理侧是独立的模态信号
    # （reproj_err >= 1.0 时加 0.15 风险），而非 quality 的替代。
    quality_score = 1.0  # 默认质量得分（最高），与推理侧一致
    if quality is not None:  # quality 值可用
        if is_bool_like(quality):  # bool 不能静默当成 0/1，否则会篡改风险标签语义。
            raise TypeError("quality must be numeric, got bool")
        quality_value = coerce_finite_scalar(quality, name='quality')  # 一次性完成 float 转换 + isfinite 校验（D5 数值安全，与 quality_model.py / robust_ekf_core.py 同口径）
        # 与推理侧 _resolve_quality 语义一致：直接 clamp 到 [0, 1]。
        # 推理侧对 >1.0 的值直接截断为 1.0（最高质量），不做 /10 反转归一化。
        quality_score = min(1.0, max(0.0, quality_value))  # 截断到 0~1
    return quality_score  # 返回质量得分 0~1


def _resolve_modality_observation_signal(event: Mapping[str, Any]) -> dict[str, float]:
    """解析模态观测信号，计算质量风险、模态信号和观测风险。

    Args:
        event: 事件字典。

    Returns:
        包含 quality_score/quality_risk/modality_signal/observation_risk 的字典。
    """

    quality_score = _resolve_event_quality_score(event)
    quality_risk = 1.0 - quality_score
    modality = event.get('modality')  # D8：不做 str() 静默转换，modality 缺失或非字符串时与 MODALITY_* 比较为 False
    modality_signal = 0.0  # 初始化模态信号
    if modality == MODALITY_UWB and not _resolve_event_valid_flag(event):  # D9：单源常量替代 'uwb' 字面量；无效 UWB

        modality_signal = UWB_INVALID_SIGNAL_FLOOR  # 无效 UWB 信号提升
    elif modality == MODALITY_VIO:  # D9：单源常量替代 'vio' 字面量；VIO 模态检查

        tracked_features = _resolve_event_field(event, 'tracked_features')  # 获取跟踪特征数
        if tracked_features is not None:  # D5：bool 排除 + OverflowError 守卫
            if is_bool_like(tracked_features):  # bool 是 int 子类，int(True)=1 会误判低特征数，篡改风险标签
                raise TypeError("tracked_features must be numeric, got bool")
            try:  # int(float('inf')) 抛 OverflowError，需守卫
                tracked_features_int = int(tracked_features)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"tracked_features must be a finite numeric, got {tracked_features!r}") from exc
            if tracked_features_int < VIO_TRACKED_FEATURES_FLOOR:  # 特征数过低
                modality_signal = max(modality_signal, VIO_LOW_FEATURES_SIGNAL_FLOOR)  # 低特征数信号提升
        reproj_err = _resolve_event_field(event, 'reproj_err')  # 获取重投影误差
        if reproj_err is not None:  # D5：coerce_finite_scalar 统一 float()+isfinite，拒绝 NaN/Inf/bool
            reproj_err_value = coerce_finite_scalar(reproj_err, name='reproj_err')
            if reproj_err_value >= VIO_HIGH_REPROJ_ERR_THRESHOLD:  # 高重投影误差
                modality_signal = max(modality_signal, VIO_HIGH_REPROJ_ERR_SIGNAL_FLOOR)  # 高重投影误差信号提升

    observation_risk = min(1.0, max(quality_risk, modality_signal))  # 观测风险
    return {
        'quality_score': quality_score,
        'quality_risk': quality_risk,
        'modality_signal': modality_signal,
        'observation_risk': observation_risk,
    }


@lru_cache(maxsize=1)
def _load_scene_axis_protocol_cached() -> dict[str, Any]:
    """缓存加载冻结场景轴协议，避免训练样本构造重复 IO。

    注意：返回值由 ``@lru_cache`` 缓存，是模块级共享对象，调用方必须自行
    ``deepcopy`` 后再修改嵌套结构，否则会污染缓存。当前唯一调用点
    ``_resolve_scene_axis_risk_floor`` 仅只读访问（dict() 浅拷贝 + .get()
    读取，不写入原对象），故未 deepcopy，与 ``_load_default_ekf_cfg``
    同口径（见该函数文档字符串）。
    """

    return load_scene_axis_protocol()


def _decode_event_scene_spec(event: Mapping[str, Any]) -> SceneSpec | None:
    """从 event/meta 中解析 scene_id 对应的 SceneSpec；缺失或非法时返回 None。"""

    def _pick_scene_code(payload: Mapping[str, Any] | None) -> str | None:
        """从 payload 中提取 scene_id 或 scene_code 字段。"""
        if not isinstance(payload, Mapping):
            return None
        for field_name in ('scene_id', 'scene_code'):
            raw_value = payload.get(field_name)
            if not is_string_like(raw_value):
                continue
            candidate = str(raw_value)
            if not candidate.strip():  # 空白占位跳过，回退到下一个字段（对齐协议层 _try_decode_scene_code 失败回退语义）
                continue
            return candidate  # 不 strip，保持与 decode_scene fullmatch 严格口径一致（D1 协议一致性）
        return None

    raw_scene_id = None
    meta = event.get('meta')
    raw_scene_id = _pick_scene_code(meta)  # 优先从 meta 中提取
    if raw_scene_id is None:
        raw_scene_id = _pick_scene_code(event)  # 回退到 event 顶层
    if raw_scene_id is None:
        return None
    try:
        return decode_scene(raw_scene_id)  # 解码场景规格
    except (ValueError, TypeError):  # 与协议层 _try_decode_scene_code 同口径（D1/D8 异常类型对齐）
        return None


def _resolve_scene_axis_risk_floor(event: Mapping[str, Any]) -> dict[str, float]:
    """从 scene_id 解析 A/N/V 轴级退化下界风险。

    这些下界不替代观测字段退化，只在默认主链中提供一致的最小风险压力：
    - A 轴：由 cross_modal_skew_ms / _ASYNC_GAP_FULL_SCALE 归一化
    - N 轴：由 nlos_ratio 直接提供 UWB 退化下界
    - V 轴：由 tracked_features_range 和 reproj_err_max 联合提供 VIO 退化下界
    """

    scene_spec = _decode_event_scene_spec(event)
    if scene_spec is None:
        # 场景规格缺失，所有轴风险下界为 0
        return {
            'async_axis_risk': 0.0,
            'nlos_axis_risk': 0.0,
            'visual_axis_risk': 0.0,
        }

    protocol_cfg = _load_scene_axis_protocol_cached()
    axes_cfg = dict(protocol_cfg.get('axes') or {})
    async_cfg = dict((axes_cfg.get('A') or {}).get(scene_spec.A_level) or {})
    nlos_cfg = dict((axes_cfg.get('N') or {}).get(scene_spec.N_level) or {})
    visual_cfg = dict((axes_cfg.get('V') or {}).get(scene_spec.V_level) or {})

    # A 轴：异步间隔风险，由 cross_modal_skew_ms 归一化到 0~1。
    # A1/A2/A3 在 scene_axis_protocol.yaml 中定义为 [lo, hi] 范围，
    # 与 async_levels.py 生成时采样逻辑对齐：取中位值作为场景级风险基准。
    async_axis_risk = 0.0
    cross_modal_skew_ms = async_cfg.get('cross_modal_skew_ms')
    if cross_modal_skew_ms is not None:
        # D5：coerce_finite_scalar 统一 float()+isfinite+拒绝 NaN/Inf/bool，与协议层
        # _resolve_scene_axis_observation_floor 同口径（Round 8 / U37 数值安全根因修复）。
        # 范围 [lo, hi] → 取中位值，与 async_levels.py L398-403 口径一致。
        if isinstance(cross_modal_skew_ms, (list, tuple)) and len(cross_modal_skew_ms) == 2:
            skew_lo, skew_hi = cross_modal_skew_ms
            skew_val = float(skew_lo) + 0.5 * (float(skew_hi) - float(skew_lo))
        else:
            skew_val = coerce_finite_scalar(cross_modal_skew_ms, name='cross_modal_skew_ms')
        async_axis_risk = min(1.0, max(0.0, skew_val / (_ASYNC_GAP_FULL_SCALE * 1000.0)))

    # N 轴：NLOS 风险，由 nlos_ratio 直接提供
    nlos_axis_risk = 0.0
    nlos_ratio = nlos_cfg.get('nlos_ratio')
    if nlos_ratio is not None:
        # 协议 yaml 中 nlos_ratio 可以是范围 [min, max]（生成时均匀采样）
        # 或直接 scalar。取 midpoint 作为典型风险水平。
        if isinstance(nlos_ratio, (list, tuple)) and len(nlos_ratio) >= 2:
            raw_val = (float(nlos_ratio[0]) + float(nlos_ratio[1])) / 2.0
        else:
            raw_val = float(nlos_ratio)
        nlos_val = coerce_finite_scalar(raw_val, name='nlos_ratio')
        nlos_axis_risk = min(1.0, max(0.0, nlos_val))

    # V 轴：视觉退化风险，由 tracked_features_range 和 reproj_err_max 联合提供
    visual_axis_risk = 0.0
    tracked_features_range = visual_cfg.get('tracked_features_range')
    reproj_err_max = visual_cfg.get('reproj_err_max')
    feature_floor_risk = 0.0
    reproj_floor_risk = 0.0
    if isinstance(tracked_features_range, (list, tuple)) and len(tracked_features_range) == 2:
        feature_lower = max(0.0, coerce_finite_scalar(tracked_features_range[0], name='tracked_features_range[0]'))
        # 特征数下界越低，说明该等级下可能出现的最坏视觉退化越强，风险下界越高。
        # 以 VIO_TRACKED_FEATURES_SAFE_FLOOR（V0 下界 100）为零风险参考点，
        # 特征数达到此值时 VIO 工作正常，风险为 0；低于此值风险线性增加。
        feature_floor_risk = min(1.0, max(0.0, (VIO_TRACKED_FEATURES_SAFE_FLOOR - min(VIO_TRACKED_FEATURES_SAFE_FLOOR, feature_lower)) / VIO_TRACKED_FEATURES_SAFE_FLOOR))
    if reproj_err_max is not None:
        # 消融实验口径：reproj_err_max 在协议中允许 [low, high] 区间；先 unwrap 到标量中值。
        if isinstance(reproj_err_max, (list, tuple)) and len(reproj_err_max) == 2:
            reproj_err_max = float((reproj_err_max[0] + reproj_err_max[1]) / 2.0)
        # D5：coerce_finite_scalar 统一 float()+isfinite+拒绝 NaN/Inf/bool，与协议层同口径。
        reproj_val = coerce_finite_scalar(reproj_err_max, name='reproj_err_max')
        # 重投影误差上限越高，视觉退化风险越高
        reproj_floor_risk = min(1.0, max(0.0, (reproj_val - VIO_REPROJ_ERR_NORM_FLOOR) / (VIO_REPROJ_ERR_FULL_SCALE - VIO_REPROJ_ERR_NORM_FLOOR)))
    visual_axis_risk = max(feature_floor_risk, reproj_floor_risk)

    return {
        'async_axis_risk': async_axis_risk,
        'nlos_axis_risk': nlos_axis_risk,
        'visual_axis_risk': visual_axis_risk,
    }
def _resolve_window_feature_value(window_tensor: Mapping[str, Any], feature_name: str) -> float | None:
    """从窗口张量中解析指定特征值。

    Args:
        window_tensor: 窗口特征字典，包含 feature_order/feature_values/missing_mask。
        feature_name: 特征名称。

    Returns:
        特征值的浮点数，特征不存在或缺失时返回 None。
    """

    feature_order = list(window_tensor.get('feature_order') or [])
    if feature_name not in feature_order:
        return None  # 特征不存在
    feature_index = feature_order.index(feature_name)  # 获取特征索引
    # 修复: feature_values/missing_mask 可能是 torch 张量, `x or []` 会对多元素张量
    # 触发布尔真值判断而 RuntimeError; 改为显式 None 判断后再 list() 迭代。
    _feature_values_raw = window_tensor.get('feature_values')
    feature_values = list(_feature_values_raw) if _feature_values_raw is not None else []
    _missing_mask_raw = window_tensor.get('missing_mask')
    missing_mask = list(_missing_mask_raw) if _missing_mask_raw is not None else []
    if feature_index >= len(feature_values) or feature_index >= len(missing_mask):  # 索引越界
        return None
    if bool(missing_mask[feature_index]):  # 特征缺失
        return None
    # D5：用 coerce_finite_scalar 统一数值安全，替代 float() + math.isfinite 两段式；
    # 原写法 float(True)=1.0 会静默接受 bool，float(None)/float('abc') 抛未捕获异常中断调用链。
    try:
        scalar_value = coerce_finite_scalar(
            feature_values[feature_index],
            name=f'feature_values[{feature_name!r}]',
        )
    except (TypeError, ValueError):
        return None  # 非数值/NaN/Inf/bool 统一返回 None
    return scalar_value  # 返回有效特征值


def _resolve_uwb_geometry_risk(
    geometric_bias_trace: Mapping[str, Any] | None,
    *,
    bridge_thresholds: Mapping[str, float],
) -> float:
    """解析 UWB 几何 bias 风险，将 bias 值归一化为 0~1。

    Args:
        geometric_bias_trace: 几何 bias 审计 trace。
        bridge_thresholds: 桥接阈值字典。

    Returns:
        归一化后的 UWB 几何 bias 风险（0~1），无效输入返回 0.0。
    """

    if not isinstance(geometric_bias_trace, Mapping):
        return 0.0
    raw_bias = geometric_bias_trace.get('raw_bias')  # 获取原始 bias
    if raw_bias is None:  # 无原始 bias
        return 0.0  # 无 bias，零风险
    try:  # 统一完成 float() + isfinite + 拒绝 bool/NaN/Inf/非数值（D5 数值安全根因修复，对齐 Round 1-7 coerce_finite_scalar 修复模式）。
        raw_bias_value = coerce_finite_scalar(raw_bias, name='raw_bias')
    except (TypeError, ValueError):  # 非数值、bool、NaN、Inf 等非法输入
        return 0.0  # 非有限值，零风险
    positive_bias = max(0.0, raw_bias_value)  # 截断为非负值
    return min(1.0, positive_bias / float(bridge_thresholds['uwb_geometric_bias_full_scale']))  # 归一化到 0~1


def _resolve_async_gap_risk(
    window_tensor: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    bridge_thresholds: Mapping[str, float],
) -> float:
    """解析异步间隔风险，将 modality_gap_dt 归一化为 0~1。

    Args:
        window_tensor: 窗口特征字典。
        event: 当前事件。
        bridge_thresholds: 桥接阈值字典。

    Returns:
        归一化后的异步间隔风险（0~1），特征不存在时返回 0.0。
    """

    feature_order = list(window_tensor.get('feature_order') or [])
    if 'modality_gap_dt' not in feature_order:
        return 0.0  # 无间隔特征，零风险
    gap_value = _resolve_window_feature_value(window_tensor, 'modality_gap_dt')  # 从特征中获取间隔值
    if gap_value is None:  # 特征缺失，零风险
        # D1/D3：不回退到 event['dt']——语义不同（dt=相邻事件间隔，modality_gap_dt=同模态间隔），
        # 与 feature_builder.py L245 及本文件 L1917 同口径（缺失即 0.0，禁止 dt 代偿）。
        return 0.0
    if not math.isfinite(gap_value):  # 非有限间隔（防御性深度，_resolve_window_feature_value 已校验）
        return 0.0  # 非有限值，零风险
    return min(1.0, max(0.0, float(gap_value) / float(bridge_thresholds['async_gap_full_scale'])))  # 归一化到 0~1


def _build_teacher_free_uwb_scaling(
    event: Mapping[str, Any],
    alignment_risk: float,
    *,
    geometry_bias_risk: float = 0.0,
    async_gap_risk: float = 0.0,
    bridge_thresholds: Mapping[str, float] | None = None,
) -> float:
    """构建 teacher-free 的 UWB scaling 值。

    基础值为 1.0，叠加 alignment_risk/quality_risk/geometry_risk/async_gap_risk 的贡献。
    无效 UWB 事件额外增加 0.15。

    Args:
        event: 当前事件。
        alignment_risk: 对齐风险（0~1）。
        geometry_bias_risk: 几何 bias 风险（0~1）。
        async_gap_risk: 异步间隔风险（0~1）。
        bridge_thresholds: 桥接阈值字典。

    Returns:
        UWB scaling 值（>= 1.0）。
    """

    resolved_thresholds = dict(bridge_thresholds or _DEFAULT_BRIDGE_THRESHOLDS)
    quality_score = _resolve_event_quality_score(event)
    quality_risk = 1.0 - quality_score  # 质量风险
    geometry_risk = min(1.0, max(0.0, float(geometry_bias_risk)))  # 截断几何风险

    scaling = (  # 计算 UWB scaling：基础 1.0 + 加权风险
        1.0  # 基础 scaling
        + (float(resolved_thresholds['uwb_scaling_alignment_coeff']) * min(1.0, max(0.0, float(alignment_risk))))  # 对齐风险贡献（D5/D7：与 geometry/async 同口径截断到 [0,1]）
        + (float(resolved_thresholds['uwb_scaling_quality_coeff']) * quality_risk)  # 质量风险贡献
        + (float(resolved_thresholds['uwb_scaling_geometry_coeff']) * geometry_risk)  # 几何风险贡献
        + (float(resolved_thresholds['async_uwb_scaling_coeff']) * min(1.0, max(0.0, float(async_gap_risk))))  # 异步间隔贡献
    )
    if not _resolve_event_valid_flag(event):  # 无效 UWB 事件
        scaling += float(resolved_thresholds['uwb_invalid_extra_boost'])  # 无效 UWB 额外增量
    return _ensure_positive_finite_scaling(max(1.0, scaling), name='uwb_scaling')  # 确保正有限


def _build_teacher_free_vio_scaling(
    event: Mapping[str, Any],
    alignment_risk: float,
    *,
    async_gap_risk: float = 0.0,
    bridge_thresholds: Mapping[str, float] | None = None,
) -> float:
    """构建 teacher-free 的 VIO scaling 值。

    基础值为 1.0，叠加 alignment_risk/quality_risk/async_gap_risk 的贡献。
    跟踪特征数 < 30 或重投影误差 >= 1.0 时额外增加 0.10。

    Args:
        event: 当前事件。
        alignment_risk: 对齐风险（0~1）。
        async_gap_risk: 异步间隔风险（0~1）。
        bridge_thresholds: 桥接阈值字典。

    Returns:
        VIO scaling 值（>= 1.0）。
    """

    resolved_thresholds = dict(bridge_thresholds or _DEFAULT_BRIDGE_THRESHOLDS)
    quality_score = _resolve_event_quality_score(event)
    quality_risk = 1.0 - quality_score  # 质量风险
    scaling = (  # 计算 VIO scaling：基础 1.0 + 加权风险
        1.0  # 基础 scaling
        + (float(resolved_thresholds['vio_scaling_alignment_coeff']) * alignment_risk)  # 对齐风险贡献
        + (float(resolved_thresholds['vio_scaling_quality_coeff']) * quality_risk)  # 质量风险贡献
        + (float(resolved_thresholds['async_vio_scaling_coeff']) * min(1.0, max(0.0, float(async_gap_risk))))  # 异步间隔贡献
    )

    tracked_features = _resolve_event_field(event, 'tracked_features')  # 获取跟踪特征数
    if tracked_features is not None:  # D5：bool 排除 + OverflowError 守卫，对齐 L1379-L1390 已修复模式
        if is_bool_like(tracked_features):  # bool 是 int 子类，int(True)=1 会误判低特征数，篡改 vio_scaling
            raise TypeError("tracked_features must be numeric, got bool")
        try:  # int(float('inf')) 抛 OverflowError，int(float('nan')) 抛 ValueError，需守卫
            tracked_features_int = int(tracked_features)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"tracked_features must be a finite numeric, got {tracked_features!r}") from exc
        if tracked_features_int < VIO_TRACKED_FEATURES_FLOOR:  # 特征数过低
            scaling += float(resolved_thresholds['vio_low_features_scaling_boost'])  # 低特征数增量
    reproj_err = _resolve_event_field(event, 'reproj_err')  # 获取重投影误差
    if reproj_err is not None:  # D5：coerce_finite_scalar 统一 float()+isfinite，拒绝 NaN/Inf/bool，对齐 L1391-L1395
        reproj_err_value = coerce_finite_scalar(reproj_err, name='reproj_err')
        if reproj_err_value >= VIO_HIGH_REPROJ_ERR_THRESHOLD:  # 高重投影误差
            scaling += float(resolved_thresholds['vio_high_reproj_scaling_boost'])  # 高重投影误差增量

    return _ensure_positive_finite_scaling(max(1.0, scaling), name='vio_scaling')  # 确保正有限


def _ensure_positive_finite_scaling(value: Any, *, name: str) -> float:
    """确保 scaling 值为正有限数。

    Args:
        value: 待检查的值，必须是数值类型（int、float 等）或
            单元素 PyTorch 张量，不接受 bool。
        name: 参数名称，用于错误消息。

    Returns:
        正有限浮点数。

    Raises:
        TypeError: 值不是数值类型（bool 也不算）。
        ValueError: 值是 inf/nan 或非正数（<= 0.0）。
    """
    # D5：统一走 coerce_finite_scalar，复用 is_real 排除 bool 与非数值类型，
    # 与 fusion_runner L132/L133 推理侧 scaling 校验同口径。
    return coerce_finite_scalar(value, name=name, min_value=0.0, inclusive=False)  # 正有限校验，排除 bool


def _resolve_safe_mode_cfg_for_training(model_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """从模型配置中提取安全模式配置，与推理侧 core_pipeline 行为一致。

    推理侧从 model_cfg['safe_mode'] 读取安全模式配置，
    训练侧应使用同一来源，而非从 event.scene_parameters 读取
    （训练数据中事件不携带 scene_parameters.safe_mode 字段）。
    """
    if isinstance(model_cfg, Mapping):
        safe_mode = model_cfg.get('safe_mode')
        if isinstance(safe_mode, Mapping):
            return deepcopy(dict(safe_mode))  # 深拷贝：隔离嵌套可变引用，避免污染调用方 model_cfg['safe_mode']（D3 根因修复，对齐 core_pipeline L618 deepcopy 口径）
    return {'enabled': False, 'risk_threshold': 0.5}


def _is_normal_scene_for_training(event: Mapping[str, Any]) -> bool | None:
    """判断当前事件是否属于正常场景（A0/N0/V0/K0/M0），与推理侧安全模式逻辑一致。

    使用与推理侧 ``_resolve_safe_mode_scene`` 相同的 ``_decode_event_scene_spec`` 解析路径，
    从 ``event/meta/scene_id`` 解码场景规格，而非从 ``scene_parameters.axes`` 读取。

    返回：
        True: 正常场景（A0/N0/V0/K0/M0）
        False: 非正常场景
        None: 场景信息缺失，无法判定（推理侧此时跳过安全模式，训练侧也应跳过）
    """
    scene_spec = _decode_event_scene_spec(event)
    if scene_spec is None:
        return None  # 场景信息缺失，与推理侧行为一致：跳过安全模式
    _nominal = get_nominal_levels()  # 从协议动态获取正常等级名，避免硬编码。
    # 直接比较，与推理侧 liquid_bridge_contract.adjust_intermediate_for_safe_mode 一致；
    # SceneSpec 各字段已为 str（scene_schema.py），str() 静默转换会掩盖类型漂移（D8）。
    # 2026-08-31：G 轴已并入 K，nominal 中无 G 轴。
    return (
        scene_spec.A_level == _nominal["A"]
        and scene_spec.N_level == _nominal["N"]
        and scene_spec.V_level == _nominal["V"]
        and scene_spec.K_value == _nominal["K"]
        and scene_spec.M_level == _nominal["M"]
    )


def _build_target_intermediate(
    *,
    teacher_model: Any | None = None,
    window_tensor: Mapping[str, Any],
    event: Mapping[str, Any],
    prediction_state: Mapping[str, Any],
    gt_state: Mapping[str, float],
    gt_alignment: Mapping[str, Any],
    anchor_lookup: Mapping[Any, tuple[float, float]] | None,
    anchor_layout: Mapping[str, Any] | None,
    bridge_thresholds: Mapping[str, float] | None = None,
    model_cfg: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """构建训练目标中间值（teacher-free 模式）。

    计算 bias/risk/uwb_scaling/vio_scaling 四个输出头的目标值，
    以及完整的审计 trace。

    Args:
        teacher_model: 保留参数，当前未使用（teacher-free 模式）。
        window_tensor: 窗口特征字典。
        event: 当前事件。
        prediction_state: 代理估计器的预测状态。
        gt_state: 对齐后的真值状态。
        gt_alignment: 真值对齐信息。
        anchor_lookup: 锚点查找表。
        anchor_layout: 锚点布局字典。
        bridge_thresholds: 桥接阈值字典。

    Returns:
        (target_intermediate, target_trace) 元组。
    """

    del teacher_model  # 当前为 teacher-free 模式，显式忽略 teacher_model 参数
    resolved_bridge_thresholds = dict(bridge_thresholds or _DEFAULT_BRIDGE_THRESHOLDS)  # 解析桥接阈值

    # 计算预测状态与真值之间的位置偏差和航向偏差
    # D5：prediction_state/gt_state 来自动态运行时数据，float() 会把 bool→0/1、NaN/inf 静默通过，
    # 直接污染 pose_error → alignment_risk → risk 训练目标；coerce_finite_scalar 排除 bool 并校验有限性。
    delta_px = coerce_finite_scalar(prediction_state.get('px', 0.0), name='prediction_state.px') - coerce_finite_scalar(gt_state['px'], name='gt_state.px')  # x 方向位置偏差
    delta_py = coerce_finite_scalar(prediction_state.get('py', 0.0), name='prediction_state.py') - coerce_finite_scalar(gt_state['py'], name='gt_state.py')  # y 方向位置偏差
    delta_yaw = angle_delta_rad(coerce_finite_scalar(prediction_state.get('yaw', 0.0), name='prediction_state.yaw'), coerce_finite_scalar(gt_state['yaw'], name='gt_state.yaw'))  # 航向角偏差（考虑环绕）

    # 计算位置误差的欧几里得范数和航向误差的绝对值
    pose_error = math.sqrt(delta_px * delta_px + delta_py * delta_py)  # 位置误差（米）
    yaw_error = abs(delta_yaw)  # 航向误差（弧度）
    # 获取归一化尺度：位置满量程（米）和航向满量程（弧度，通常为 pi）
    pose_full_scale_m, yaw_full_scale_rad = _resolve_alignment_risk_scales()
    # 将位置误差和航向误差归一化到 0~1 范围
    normalized_pose_error = min(1.0, max(0.0, pose_error / pose_full_scale_m))
    normalized_yaw_error = min(1.0, max(0.0, yaw_error / yaw_full_scale_rad))
    # 对齐风险取位置和航向归一化误差的最大值
    alignment_risk = min(1.0, max(0.0, max(normalized_pose_error, normalized_yaw_error)))
    # 解析观测信号（质量得分、质量风险、模态信号、观测风险）
    observation_signal = _resolve_modality_observation_signal(event)
    scene_axis_risk_floor = _resolve_scene_axis_risk_floor(event)

    # 根据模态确定 bias 目标值
    modality = event['modality']  # 直接使用原始模态值，str() 静默转换会掩盖类型漂移（D8），与 _is_normal_scene_for_training 同口径
    bias = 0.0  # 默认 bias 为 0（中性基线）
    bias_source = 'neutral_baseline'  # 默认 bias 来源为中性基线
    geometric_bias_trace: dict[str, Any] = {}  # 几何 bias 审计 trace
    if modality == MODALITY_UWB:
        # UWB 模态：如果有锚点查找表，计算几何 bias teacher 目标
        if anchor_lookup:
            bias, geometric_bias_trace = _build_geometric_bias_target(
                event,
                gt_state,
                anchor_lookup,
                anchor_layout=anchor_layout,
            )
            bias_source = 'measured_minus_geometric_true_range'  # bias 来源为测量值减几何真值
    # 计算 UWB 几何 bias 风险（归一化到 0~1）
    uwb_geometry_risk = _resolve_uwb_geometry_risk(
        geometric_bias_trace,
        bridge_thresholds=resolved_bridge_thresholds,
    )
    # 计算异步间隔风险（归一化到 0~1）
    async_gap_risk = _resolve_async_gap_risk(
        window_tensor,
        event,
        bridge_thresholds=resolved_bridge_thresholds,
    )
    effective_async_risk = max(async_gap_risk, scene_axis_risk_floor['async_axis_risk'])
    axis_observation_floor = 0.0
    if modality == MODALITY_UWB:
        axis_observation_floor = max(
            scene_axis_risk_floor['async_axis_risk'],
            scene_axis_risk_floor['nlos_axis_risk'],
        )
    elif modality == MODALITY_VIO:
        axis_observation_floor = max(
            scene_axis_risk_floor['async_axis_risk'],
            scene_axis_risk_floor['visual_axis_risk'],
        )
    # 综合观测风险取各信号的最大值
    observation_risk = max(
        observation_signal['observation_risk'],
        uwb_geometry_risk,
        effective_async_risk,
        axis_observation_floor,
    )
    # risk 头只监督 pre-bridge base risk / alignment risk。
    # 观测退化信号保留在 observation_risk trace 和 scaling 链路里，
    # 由 bridge 侧与 quality / modality / axis floor 共同消费。
    risk = alignment_risk

    # 只为当前模态生成 teacher-free scaling 标签；非当前模态保持中性值 1.0。
    uwb_scaling = 1.0
    vio_scaling = 1.0
    if modality == MODALITY_UWB:
        uwb_scaling = _build_teacher_free_uwb_scaling(
            event,
            alignment_risk,
            geometry_bias_risk=uwb_geometry_risk,
            async_gap_risk=effective_async_risk,
            bridge_thresholds=resolved_bridge_thresholds,
        )
    elif modality == MODALITY_VIO:
        vio_scaling = _build_teacher_free_vio_scaling(
            event,
            alignment_risk,
            async_gap_risk=effective_async_risk,
            bridge_thresholds=resolved_bridge_thresholds,
        )

    # 组装四头目标中间值
    target_intermediate = {
        'bias': bias,  # bias 目标值：UWB 为几何 bias，其他为 0
        'risk': risk,  # risk 目标值：对齐风险
        'uwb_scaling': uwb_scaling,  # UWB scaling 目标值
        'vio_scaling': vio_scaling,  # VIO scaling 目标值
    }

    # 模拟推理时安全模式对目标值的调整（仅审计，不写入 target_intermediate）。
    # 推理时 adjust_intermediate_for_safe_mode 会根据 risk 和场景上下文
    # 修改 bias/scaling；如果训练目标也做同样缩放，bias label 会被 risk_value
    # (1e-3 量级) 压成接近 0 的尘埃，导致 bias_head 永远学不出非零权重。
    # 训练目标保持几何 teacher 真值，让 bias_head 拿到有意义的监督信号。
    safe_mode_cfg = _resolve_safe_mode_cfg_for_training(model_cfg)
    safe_mode_enabled = _coerce_safe_mode_enabled_flag(safe_mode_cfg.get('enabled', False))
    if safe_mode_enabled:
        risk_value = max(0.0, min(1.0, risk))
        is_normal_scene = _is_normal_scene_for_training(event)
        converge_flag = None
        partial_damping = None
        if is_normal_scene is not None:
            risk_threshold = coerce_finite_scalar(safe_mode_cfg.get('risk_threshold', 0.5), name='safe_mode_cfg.risk_threshold')
            converge_flag = risk_value <= risk_threshold
            if not converge_flag and not is_normal_scene:
                partial_damping = 1.0 - RISK_PARTIAL_DAMPING_COEFF * risk_value
                if partial_damping < 0.0:
                    partial_damping = 0.0
    else:
        risk_value = None
        is_normal_scene = None
        converge_flag = None
        partial_damping = None
    # 组装完整的审计 trace，记录目标构造过程中的所有中间量
    target_trace = {
        'modality': modality,  # 当前事件模态
        'baseline_risk': 0.0,  # 基线风险（当前固定为 0）
        'alignment_risk': alignment_risk,  # 对齐风险
        'prediction_state': {  # 预测状态快照
            'px': float(prediction_state.get('px', 0.0)),
            'py': float(prediction_state.get('py', 0.0)),
            'yaw': float(prediction_state.get('yaw', 0.0)),
        },
        'observation_risk': observation_risk,  # 综合观测风险
        'quality_risk': observation_signal['quality_risk'],  # 质量风险
        'safe_mode_simulation': {  # D10：训练时 safe-mode 模拟的审计记录，仅审计不写入 target_intermediate。
            'enabled': safe_mode_enabled,
            'risk_value': risk_value,
            'is_normal_scene': is_normal_scene,
            'converge_flag': converge_flag,
            'partial_damping': partial_damping,
            'pre_adjustment_bias': bias,
            'pre_adjustment_uwb_scaling': uwb_scaling,
            'pre_adjustment_vio_scaling': vio_scaling,
        },
        'modality_signal': observation_signal['modality_signal'],  # 模态信号
        'uwb_geometry_risk': uwb_geometry_risk,  # UWB 几何 bias 风险
        'async_gap_risk': async_gap_risk,  # 异步间隔风险
        'effective_async_risk': effective_async_risk,  # 事件级 gap 与场景 A 轴共同决定的有效异步风险
        'async_axis_risk': scene_axis_risk_floor['async_axis_risk'],  # A 轴退化下界风险
        'nlos_axis_risk': scene_axis_risk_floor['nlos_axis_risk'],  # N 轴退化下界风险
        'visual_axis_risk': scene_axis_risk_floor['visual_axis_risk'],  # V 轴退化下界风险
        'axis_observation_floor': axis_observation_floor,  # 由场景轴直接提供的观测风险下界
        'observation_risk_blend': float(resolved_bridge_thresholds['observation_risk_blend']),  # 观测风险混合系数
        'current_modality_gap_dt': float(_resolve_window_feature_value(window_tensor, 'modality_gap_dt') or 0.0),  # 当前模态间隔 dt
        'pose_error': pose_error,  # 位置误差（米）
        'normalized_pose_error': normalized_pose_error,  # 归一化位置误差
        'alignment_pose_full_scale_m': pose_full_scale_m,  # 位置归一化满量程
        'yaw_error': yaw_error,  # 航向误差（弧度）
        'normalized_yaw_error': normalized_yaw_error,  # 归一化航向误差
        'alignment_yaw_full_scale_rad': yaw_full_scale_rad,  # 航向归一化满量程
        'gt_alignment_mode': gt_alignment.get('mode', 'unknown'),  # 真值对齐模式；D8：str() 静默转换会把 None→'None'、int→str 掩盖数据合同漂移，直接使用原始值使合同违例可被审计追溯
        'gt_alignment_gap': float(gt_alignment.get('time_gap') or 0.0),  # 真值对齐时间差
        'gt_alignment_support_count': len(list(gt_alignment.get('support_timestamps') or [])),  # 支撑真值数
        'bias_source': bias_source,  # bias 来源标识
        'risk_source': 'pre_bridge_base_risk_alignment_only',  # risk 来源标识
        'confidence_source': 'modality_specific_scaling_with_neutral_floor',  # confidence 来源标识
        'geometry_bias_teacher': 'active' if geometric_bias_trace else 'unavailable_missing_anchor_layout',  # 几何 bias teacher 状态
        **geometric_bias_trace,  # 展开几何 bias 审计 trace
    }
    # 应用 robust teacher 补充：低质量+高风险条件下增强 scaling
    target_intermediate, target_trace = _apply_robust_teacher_supplement(
        target_intermediate,
        target_trace,
        event=event,
        bridge_thresholds=resolved_bridge_thresholds,
    )

    return (
        target_intermediate,
        target_trace,
    )


def _build_feature_window_builder(model_cfg: Mapping[str, Any], estimator: Any):
    """构建 Liquid 训练样本的窗口特征提取器。

    这个工厂函数把历史事件、当前事件、状态历史和锚点布局整理成模型真正消费的
    窗口字典。它单独封装出来，是为了让训练前端在不改原始样本语义的前提下，统一
    生成特征矩阵、缺失掩码和时间窗上下文。

    Args:
        model_cfg: 模型配置字典，须包含 feature_order 和 window 参数。
        estimator: 估计器实例，用于获取锚点查找表。

    Returns:
        _builder 闭包函数，接受 (history, event, cfg) 参数。
    """


    # 从模型配置中提取特征顺序列表
    feature_order = list(model_cfg.get('feature_order') or [])
    if not feature_order:
        raise ValueError('model_cfg.feature_order must be non-empty for liquid training')

    # 从模型配置中提取窗口参数
    window_cfg = dict(model_cfg.get('window') or {})
    if 'size' not in window_cfg:
        raise KeyError(
            "§10.4 序列截断长度守卫: model_cfg.window.size 必须显式声明, "
            "禁止默认值 1 兜底（spec §10.4 记忆深度对等前提, 训推同侧）."
        )  # §10.4 序列截断长度属于记忆深度，训推对等必须显式写死；默认 1 兜底会静默破坏对等前提。
    _window_size_raw = window_cfg['size']
    if not is_integer(_window_size_raw):
        raise TypeError(f"window.size must be an integer, got {type(_window_size_raw).__name__}")
    if int(_window_size_raw) <= 0:
        raise ValueError(f"window.size must be a positive integer, got {_window_size_raw!r}")
    window_size = int(_window_size_raw)  # 窗口大小：每个窗口包含的历史步数。D7：移除未使用的 step_size（window_cfg.get('step') 全程未被消费，属死代码；当前 trailing window 实现不依赖步长，仅取尾部 window_size 条事件，对齐 core_pipeline.py L793 同口径修复）。
    # 违反项 29 修复：手册 P29 硬约束——窗口 W=128 步 @150Hz = 0.85s。
    # 显式校验 window_size=128；非 128 时给出明确报错（不让步长偷偷漂移），
    # 与 precheck_orchestrator.check_P20_window / check_P29_windowing 口径一致。
    # 例外：单元/冒烟 fixture 只有 2~13 个事件，永远填不满 128 窗（P31 会返回空窗）。
    # 仿照 allow_cross_set_leakage 先例，允许 window.allow_smoke_window=true 显式声明
    # 偏离（仅限 smoke/单元场景）；真实配置（lstm_ekf.yaml/liquid_ekf.yaml）不设此旗标，
    # P29 硬门对真实 run 保持 fail-loud。
    _allow_smoke_window = bool(window_cfg.get('allow_smoke_window', False))
    _EXPECTED_WINDOW_SIZE = 128
    if window_size != _EXPECTED_WINDOW_SIZE and not _allow_smoke_window:
        raise ValueError(
            f"违反项 29 (手册 P29): window.size must be exactly {_EXPECTED_WINDOW_SIZE} steps "
            f"@150Hz = 0.85s, got {window_size}; W != 128 会破坏离线消融的窗口公平性。"
        )
    # 违反项 29 修复：手册 P29 硬约束——warm-up 前 10s 不进评估。
    # warmup_s 必须显式声明（默认 10.0），用于前端在窗口切片时剔除前 10s。
    warmup_s_raw = window_cfg.get('warmup_s', 10.0)
    try:
        warmup_s = float(warmup_s_raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"window.warmup_s must be a finite number (default 10.0), got {warmup_s_raw!r}"
        )
    if warmup_s < 0.0 or not math.isfinite(warmup_s):
        raise ValueError(
            f"违反项 29 (手册 P29): window.warmup_s must be finite and non-negative, got {warmup_s!r}"
        )
    # 从估计器中获取锚点查找表
    anchor_lookup = dict(getattr(estimator, '_anchor_lookup', {}) or {})
    if not anchor_lookup:
        # 如果估计器没有 _anchor_lookup，尝试从 cfg.anchor_layout 构建
        anchor_layout = getattr(estimator, 'cfg', {}).get('anchor_layout') if hasattr(estimator, 'cfg') else None
        if anchor_layout is not None:
            anchor_lookup = build_anchor_lookup(anchor_layout)

    def _builder(history, event, cfg):
        """窗口特征提取闭包：将历史事件序列转换为模型消费的窗口字典。

        依次完成：状态历史对齐 → 特征向量构建 → 窗口切分 → 取最新窗口 → 组装输出。

        Args:
            history: 历史事件列表，每个元素为事件字典。
            event: 当前事件字典。
            cfg: 配置字典，可包含 state_history 键。

        Returns:
            窗口特征字典，包含 current_modality/feature_order/feature_values/
            missing_mask/dt/feature_window/missing_mask_window/
            window_index_map/event_time_window 等字段。

        Raises:
            ValueError: state_history 长度与 history 不一致。
        """
        # 从配置中提取状态历史列表（warmup 过滤前；后面会随 history 一起切到 filtered_state_history）
        state_history = list((cfg or {}).get('state_history') or [])
        # 违反项 29 修复：手册 P29 硬约束——warm-up 前 10s 不进评估。
        # 对 history 和 state_history 做时间过滤，只保留 warmup_s 之后的事件。
        # event['t'] 是事件时间戳（浮点秒），过滤后确保训练窗口全在稳态区。
        if warmup_s > 0.0:
            # 找出第一个 t >= warmup_s 的索引：warmup_s 之后的事件才进训练。
            seq_t0 = None
            for _idx, _ev in enumerate(history):
                _t = float(_ev.get('t', 0.0))
                if _t >= warmup_s:
                    seq_t0 = _idx
                    break
            # seq_t0 == None 表示整条序列都在 warmup 内，此时过滤后为空列表。
            if seq_t0 is None:
                seq_t0 = len(history)
            filtered_history = history[seq_t0:]   # 违反项 29 修复：保留 warmup_s 之后的事件。
            filtered_state_history = state_history[seq_t0:]
        else:
            filtered_history = history
            filtered_state_history = state_history
        # 后续 state_history 由 filtered_state_history 取代（保持长度对齐）
        state_history = filtered_state_history
        if state_history and len(state_history) != len(filtered_history):
            raise ValueError('state_history must align with filtered_history length')
        # 如果没有提供状态历史，使用估计器当前状态填充
        if not state_history:
            state_ctx = {}
            if hasattr(estimator, 'get_state'):
                state_estimate = estimator.get_state()
                _state_payload = getattr(state_estimate, 'state', None)  # 取 state 载荷。
                if isinstance(_state_payload, Mapping):  # 仅当 state 是映射类型时才转字典，避免 list/tuple/str 触发 dict() 异常。
                    for _k, _v in dict(_state_payload).items():  # D5：逐字段用 coerce_finite_scalar 一步到位完成数值转换+有限性校验+tensor/.item() 处理，替代 float()+isfinite 两步式写法；过滤非数值/非有限值，防止 NaN/inf 或 tensor 泄漏（对齐 core_pipeline.py L687 同口径修复）。
                        try:
                            _sv = coerce_finite_scalar(_v, name=f"state.{_k}")
                        except (TypeError, ValueError):
                            continue
                        state_ctx[str(_k)] = _sv
            # 用同一个状态填充整个历史长度
            state_history = [state_ctx] * len(filtered_history)
        row_count = len(filtered_history)  # 违反项 29 修复：用 warmup 过滤后的历史长度（移除前 10s 事件）。
        window_start = max(0, row_count - window_size)  # 当前样本只消费尾部 trailing window。
        window_index_map = list(range(window_start, row_count))  # 保证当前事件总在窗口最后一行。
        window_history = [filtered_history[index] for index in window_index_map]  # 只保留当前窗口对应的事件子序列。
        window_state_history = [state_history[index] for index in window_index_map]  # 只保留当前窗口对应的状态子序列。
        # 构建特征状态历史：将事件历史和状态历史结合，加入锚点信息
        feature_state_history = build_feature_state_history(
            window_history,
            window_state_history,
            anchor_lookup=anchor_lookup or None,
        )

        # 违反项 31 修复：手册 P31 硬约束——空轨迹、极短序列、M1 长间隙在轨迹首尾。
        # 不产生人工尖峰；长间隙不触发除零/NaN。
        # 当 filtered_history 不足一个完整 window_size 时，跳过此次构窗（返回空特征），
        # 让上游 caller 自然丢弃该样本，不让极短序列污染训练。
        # 例外：window.allow_smoke_window=true（单元/冒烟显式声明）时保持历史行为——
        # 用现有历史填充窗口（pad-to-history），否则微型 fixture 产出的所有样本都是空窗，
        # 单元链路无法验证任何窗口语义（真实 run 不设此旗标，P31 保持 fail-loud）。
        if row_count < window_size and not _allow_smoke_window:
            empty_window = torch.zeros((0, len(feature_order)), dtype=torch.float32)
            empty_missing = torch.zeros((0, len(feature_order)), dtype=torch.float32)
            empty_event_t = torch.zeros((0,), dtype=torch.float32)
            return {
                'current_modality': event.get('modality'),
                'feature_order': list(feature_order),
                'feature_values': torch.zeros(len(feature_order), dtype=torch.float32),
                'missing_mask': torch.ones(len(feature_order), dtype=torch.float32),
                'dt': 0.0,
                'feature_window': empty_window,
                'missing_mask_window': empty_missing,
                'window_index_map': [],
                'event_time_window': empty_event_t,
                'warmup_excluded_short_seq': True,  # 违反项 29/31: 序列过短，标记样本丢弃。
            }
        # 违反项 29/31 修复：手册 P29/P31 硬约束——窗口不跨序列边界；
        # IMU 高频段不跨序列拼接（窗口内必须是同一序列的事件）。
        # event['meta']['seq_id'] 与 window_history[0]['meta']['seq_id'] 必一致，
        # 否则说明上游 caller 误把多条序列事件拼到同一 history，禁止构窗。
        event_seq_id = (event.get('meta') or {}).get('seq_id') if isinstance(event.get('meta'), dict) else None
        first_hist_seq_id = None
        if filtered_history and isinstance(filtered_history[0].get('meta'), dict):
            first_hist_seq_id = filtered_history[0].get('meta', {}).get('seq_id')
        if (event_seq_id is not None and first_hist_seq_id is not None
                and str(event_seq_id) != str(first_hist_seq_id)):
            raise ValueError(
                f"违反项 29 (手册 P29): 窗口不跨序列边界，但 event.seq_id={event_seq_id!r} "
                f"与 history[0].seq_id={first_hist_seq_id!r} 不一致；禁止跨序列构窗。"
            )
        # 为每个历史事件构建特征向量和缺失掩码
        feature_rows = [
            build_feature_vector(hist_event, hist_state_ctx, feature_order)
            for hist_event, hist_state_ctx in zip(window_history, feature_state_history, strict=True)
        ]
        # 提取特征值矩阵和缺失掩码矩阵
        feature_matrix = [list(row['feature_values']) for row in feature_rows]  # D3：逐行 list() 拷贝隔离 feature_rows 内部列表，避免返回的 feature_window 与 build_feature_vector 返回值共享可变引用，防止下游消费者修改窗口行时反向污染 current_row['feature_values']（对齐 core_pipeline.py L709 同口径修复）。
        missing_matrix = [list(row['missing_mask']) for row in feature_rows]  # D3：同上，缺失掩码窗口也逐行隔离。
        # 取最新一行特征作为当前特征
        current_row = feature_rows[-1]
        feature_window = feature_matrix  # 当前 builder 只返回当前事件实际消费的 trailing window。
        missing_window = missing_matrix  # 缺失掩码窗口与特征窗口逐行对齐。
        # 提取窗口内每个事件的时间戳
        event_time_window = [coerce_finite_scalar(window_history[index]['t'], name='event.t') for index in range(len(window_history))]  # D5：用 coerce_finite_scalar 严格校验时间戳有限性，禁止 NaN/inf 泄漏到窗口时间轴和后续 dt 推导（对齐 core_pipeline.py L714 同口径修复）。
        if len(event_time_window) >= 2:  # 训练构窗与推理构窗保持同一首步 dt 口径。
            window_dt = max(0.0, event_time_window[1] - event_time_window[0])
        elif window_history:  # 单步窗口没有相邻时间差时，才回退到当前唯一事件自带的 dt。
            window_dt = coerce_finite_scalar(window_history[0]['dt'], name='event.dt')  # D5：单步窗口仍沿用事件自带的显式步长，并严格校验有限性，禁止 NaN/inf 进入 Liquid step_dts 首步 fallback（对齐 core_pipeline.py L718 同口径修复）。
        else:
            window_dt = coerce_finite_scalar(event['dt'], name='event.dt')  # D5：理论兜底——空窗口时退回当前事件步长，并严格校验有限性，与上游 validate_event 的 dt 非负+有限约束对齐（对齐 core_pipeline.py L720 同口径修复）。

        # §13 OOM 修复：把 Python float list 矩阵直接转成 torch.float32 tensor，
        # 单样本从 ~13 KB 降到 ~1.5 KB，全量 255 万样本总内存从 33 GB 降到 ~3.8 GB。
        # trainer 的 _materialize_samples 已有 fast path（tensor 已存在时跳过 normalize），
        # 不会再做重复转化。
        feature_window_tensor = torch.as_tensor(feature_window, dtype=torch.float32)
        missing_window_tensor = torch.as_tensor(missing_window, dtype=torch.float32)
        feature_values_tensor = torch.as_tensor(list(current_row['feature_values']), dtype=torch.float32)
        missing_mask_tensor = torch.as_tensor(list(current_row['missing_mask']), dtype=torch.float32)
        event_time_window_tensor = torch.as_tensor(event_time_window, dtype=torch.float32)
        return {
            'current_modality': event['modality'],
            'feature_order': list(feature_order),
            'feature_values': feature_values_tensor,
            'missing_mask': missing_mask_tensor,
            'dt': window_dt,
            'feature_window': feature_window_tensor,
            'missing_mask_window': missing_window_tensor,
            'window_index_map': window_index_map,
            'event_time_window': event_time_window_tensor,
        }
    return _builder
def _build_liquid_samples(
    split_ids: list[str],
    cfg: dict[str, Any],
    model_cfg: Mapping[str, Any],
    estimator_cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """构建 Liquid 训练样本列表。

    遍历所有序列，为每个 UWB/VIO 事件构造训练样本，
    包含窗口特征、目标中间值和审计 trace。

    Args:
        split_ids: 序列 ID 列表。
        cfg: 流水线配置字典。
        model_cfg: 模型配置字典。
        estimator_cfg: 估计器配置字典。

    Returns:
        (samples, sample_report) 元组。
    """
    def _init_teacher_quality_audit() -> dict[str, int]:
        """初始化几何 teacher 质量审计计数字典。"""
        return {
            'uwb_teacher_enabled_sample_count': 0,  # UWB teacher 启用的样本数
            'uwb_teacher_fallback_sample_count': 0,  # UWB teacher 回退的样本数
            'teacher_anchor_layout_available_sequence_count': 0,  # 锚点布局可用的序列数
            'teacher_anchor_layout_missing_sequence_count': 0,  # 锚点布局缺失的序列数
            'teacher_projection_applied_sequence_count': 0,  # 应用了 3D→2D 投影的序列数
            'teacher_projection_required_but_unavailable_sequence_count': 0,  # 需要投影但无法执行的序列数
            'teacher_geometry_bias_trace_available_count': 0,  # 几何 bias trace 可用的样本数
            'teacher_geometry_bias_trace_missing_count': 0,  # 几何 bias trace 缺失的样本数
        }

    def _accumulate_teacher_quality_audit(
        teacher_quality_audit: dict[str, int],
        *,
        resolved_anchor_layout: Mapping[str, Any] | None,
        anchor_layout_report: Mapping[str, Any],
        target_trace: Mapping[str, Any] | None = None,
        is_sequence_init: bool = False,
        modality: str | None = None,
    ) -> None:
        """累加几何 teacher 质量审计计数。"""
        if is_sequence_init:  # 序列初始化阶段：统计锚点布局可用性
            if resolved_anchor_layout is not None:
                teacher_quality_audit['teacher_anchor_layout_available_sequence_count'] += 1
                if anchor_layout_report.get('teacher_input_position_dim') == 2 and anchor_layout_report.get('metadata_position_dim') == 3:
                    teacher_quality_audit['teacher_projection_applied_sequence_count'] += 1  # 3D→2D 投影已应用
            else:
                teacher_quality_audit['teacher_anchor_layout_missing_sequence_count'] += 1
                blockers = list(anchor_layout_report.get('blockers') or [])
                if any(is_string_like(blocker) and 'requires_2d_teacher' in blocker for blocker in blockers):  # D8：is_string_like 守卫，避免 str() 静默转换非字符串 blocker
                    teacher_quality_audit['teacher_projection_required_but_unavailable_sequence_count'] += 1  # 需要投影但无法执行
            return

        if modality != MODALITY_UWB or target_trace is None:  # 仅统计 UWB 模态的样本（D9：单源常量）
            return

        geometry_teacher_status = target_trace.get('geometry_bias_teacher', 'unavailable_missing_anchor_layout')  # D8：保持原始类型，避免 str() 静默转换（值由 _build_target_intermediate L2000 保证为字符串）
        if geometry_teacher_status == 'active':
            teacher_quality_audit['uwb_teacher_enabled_sample_count'] += 1  # teacher 启用
            teacher_quality_audit['teacher_geometry_bias_trace_available_count'] += 1  # bias trace 可用
        else:
            teacher_quality_audit['uwb_teacher_fallback_sample_count'] += 1  # teacher 回退
            teacher_quality_audit['teacher_geometry_bias_trace_missing_count'] += 1  # bias trace 缺失

    # 解析桥接阈值配置
    bridge_thresholds = _resolve_bridge_thresholds(model_cfg)
    all_samples: list[dict[str, Any]] = []  # 收集所有序列的全部训练样本
    init_state_cfg = dict(estimator_cfg.get('init_state') or {})  # 估计器初始状态配置
    # OOM 诊断：函数entry内存基线
    try:
        import psutil as _psutil_diag
        _diag_proc = _psutil_diag.Process()
        _diag_mem_baseline = _diag_proc.memory_info().rss / 1024 / 1024
        try:
            print(f"[OOM_DIAG] _build_liquid_samples entry: PID RSS={_diag_mem_baseline:.0f} MB, system_available={_psutil_diag.virtual_memory().available/1024/1024:.0f} MB, split_ids={len(split_ids)}")
        except OSError:
            pass  # subprocess stdout closed
    except Exception as _e:
        try:
            print(f"[OOM_DIAG] psutil unavailable: {_e}")
        except OSError:
            pass
    # 初始化样本报告字典，记录全局统计信息
    sample_report: dict[str, Any] = {
        'usable_sample_count': 0,  # 可用样本总数
        'usable_sample_count_by_modality': {MODALITY_UWB: 0, MODALITY_VIO: 0},  # 按模态统计的可用样本数（D9：单源常量）
        'train_window_count': 0,  # 训练窗口数（后续填充）
        'val_window_count': 0,  # 验证窗口数（后续填充）
        'geometry_bias_teacher': _describe_geometry_bias_teacher(None),  # 几何 bias teacher 状态描述
        'teacher_quality_audit': _init_teacher_quality_audit(),  # 几何 teacher 质量审计
        'bias_source_counts': {'measured_minus_geometric_true_range': 0, 'neutral_baseline': 0},  # bias 来源统计
        'sequences': {},  # 按序列 ID 索引的子报告
        'bridge_thresholds': dict(bridge_thresholds),  # 使用的桥接阈值
        # §13.6.2.6 激活架构前提 fail-loud 统计：真异步 Δt 激活样本数 / 状态依赖脉冲样本数
        # 训练集必须含真异步 Δt（modality_gap_dt > 0）和状态依赖脉冲（proxy_report 非空）
        # 的样本，否则 §13.6.2.6 "全程或至少末段激活真异步 Δt + 状态依赖脉冲" 前提不成立。
        'async_dt_activated_sample_count': 0,  # 真异步 Δt 激活样本数（modality_gap_dt > 0）
        'state_dependent_pulse_sample_count': 0,  # 状态依赖脉冲样本数（proxy_report 非空）
        'async_dt_activated_modality_counts': {MODALITY_UWB: 0, MODALITY_VIO: 0},  # 按模态统计
    }

    # 遍历所有序列（去重并保持顺序）
    _diag_seq_idx = 0
    for seq_id in _unique_in_order(split_ids):
        # OOM 诊断：每 50 个序列打印一次内存状态
        if _diag_seq_idx % 50 == 0:
            try:
                import psutil as _psutil_diag
                _diag_proc_now = _psutil_diag.Process()
                _diag_rss_now = _diag_proc_now.memory_info().rss / 1024 / 1024
                _diag_avail_now = _psutil_diag.virtual_memory().available / 1024 / 1024
                try:
                    print(f"[OOM_DIAG] seq_idx={_diag_seq_idx}/{len(split_ids)} seq_id={seq_id} RSS={_diag_rss_now:.0f} MB, system_available={_diag_avail_now:.0f} MB, all_samples={len(all_samples)}")
                except OSError:
                    pass  # subprocess stdout closed
                # 提前预警：剩余可用内存低于阈值时立即报错，避免 OOM 崩溃。
                # 阈值默认 500MB；可用环境变量 LIQUIDLOC_OOM_MIN_AVAIL_MB 覆盖——
                # 长进程场景（如全套件 pytest 单进程累积 RSS 数 GB）可显式调低，
                # 真实训练进程不设该变量则保持 500MB fail-loud 保护不变。
                import os as _os

                _oom_min_avail_mb = float(_os.environ.get("LIQUIDLOC_OOM_MIN_AVAIL_MB", "500"))
                if _diag_avail_now < _oom_min_avail_mb:
                    raise MemoryError(f"[OOM_DIAG] 系统可用内存仅 {_diag_avail_now:.0f} MB（阈值 {_oom_min_avail_mb:.0f} MB），将在加载更多序列时崩溃。当前已处理 {_diag_seq_idx}/{len(split_ids)} 个序列，all_samples={len(all_samples)}")
            except Exception as _e:
                if isinstance(_e, MemoryError):
                    raise
        _diag_seq_idx += 1
        # 加载序列的事件列表、真值行和数据源报告
        events, gt_rows, source_report = _resolve_sequence_payload(seq_id, cfg)
        # 按时间戳排序事件
        events = sorted(list(events or []), key=lambda item: coerce_finite_scalar(item.get('t', item.get('timestamp', 0.0)), name='event.t'))  # D5：coerce_finite_scalar 拒绝 NaN/Inf/非数值，避免排序未定义行为
        # 归一化真值行
        gt_rows = _normalize_gt_rows(gt_rows)

        # §A 修复: 训练时按序列 scene_parameters.axes.N.level 注入 NLOS 脉冲。
        # 违反项 12 修复: 手册 P12 硬约束——训练/测试注入必须分布对称（ρ/μ/σ 同分布），
        # 且按整条序列划分。原状：sim_materializer 已按协议档位（如 N1.5）注入 NLOS，
        # 训练侧再强制注入 N3 协议 (nlos_ratio=0.37, μ=5.0m, σ=1.5m)，
        # 导致训练分布 (N3) 与测试分布 (N1.5) 不对称，bias 头学到的偏差量级是 N3 而非测试 N1.5，
        # 评估时反而被 N1.5 撞主结论。修复：删除训练侧强制注入，材料器已统一注入 N1.5 协议。
        # 训练数据应与测试数据走同一注入协议（同一 sim_materializer 路径），不允许再叠加。
        # events 保留原状，不再调用 _maybe_inject_nlos_for_training。
        # （原代码：events = _maybe_inject_nlos_for_training(events, gt_rows)）

        # 解析序列的锚点布局
        resolved_anchor_layout, anchor_layout_report = _resolve_sequence_anchor_layout(estimator_cfg, source_report)
        # 从锚点布局构建查找表
        anchor_lookup = build_anchor_lookup(resolved_anchor_layout) if resolved_anchor_layout else {}
        # 如果有锚点布局，更新全局报告中的 geometry_bias_teacher 描述
        if resolved_anchor_layout is not None:
            sample_report['geometry_bias_teacher'] = _describe_geometry_bias_teacher(
                resolved_anchor_layout,
                source=anchor_layout_report.get('source') or source_report.get('anchor_layout_source') or source_report.get('source'),
                metadata_available=bool(anchor_layout_report.get('metadata_available')),
                metadata_source=anchor_layout_report.get('metadata_source') or source_report.get('anchor_layout_metadata_source'),
                blockers=list(anchor_layout_report.get('blockers') or []),
                metadata_position_dim=anchor_layout_report.get('metadata_position_dim'),
                teacher_input_position_dim=anchor_layout_report.get('teacher_input_position_dim'),
            )

        # 创建估计器桩对象，仅携带锚点查找表，供窗口构建器使用
        estimator_stub = type('_EstimatorStub', (), {'_anchor_lookup': anchor_lookup})()
        # 构建窗口特征提取器
        window_builder = _build_feature_window_builder(model_cfg, estimator_stub)
        # 构建代理估计器配置，加入锚点布局
        proxy_estimator_cfg = deepcopy(dict(estimator_cfg))  # D3：深拷贝避免嵌套结构与 estimator_cfg 共享引用，防止 create_estimator 污染调用方配置
        if resolved_anchor_layout is not None:
            proxy_estimator_cfg['anchor_layout'] = resolved_anchor_layout
        # 创建代理估计器，用于在样本构造过程中模拟估计器状态推进
        proxy_estimator = create_estimator(proxy_estimator_cfg.get('name') or ESTIMATOR_NAME_EKF, proxy_estimator_cfg)  # D8：移除 str() 静默转换，name 应为字符串或由 ESTIMATOR_NAME_EKF 兜底；D9：fallback 引用单源常量 ESTIMATOR_NAME_EKF。
        # 2026-09-02 R-1 修复：EKF 冷启动 init=(0,0) 导致 pre_update_proxy_state 与 gt 偏差 ~16m，
        # delta_px = prediction_state.px - gt_state.px 被锁在 16m 偏移；模型被迫学错误位置。
        # P16 协议要求序列初始化时用首批 UWB 事件做三边测量解（≥3 锚），关闭 R-1 偏差。
        if events:
            uwb_init_events = [e for e in events if e.get('modality') == MODALITY_UWB][:32]
            vio_init_events = [e for e in events if e.get('modality') == MODALITY_VIO][:4]
            if len(uwb_init_events) >= 3 and hasattr(proxy_estimator, 'apply_p16_init_from_first_frame'):
                try:
                    init_report = proxy_estimator.apply_p16_init_from_first_frame(
                        uwb_init_events,
                        vio_init_events,
                    )
                    _ir = init_report or {}
                    _p16 = _ir.get('trilaterated_position_xy') or (None, None)
                    try:
                        print(f'[R-1 fix] P16 triggered: seq={events[0].get("meta", {}).get("seq_id","?")} '
                              f'anchors={_ir.get("n_anchors_used","?")} '
                              f'tril_pos=({_p16[0]!r},{_p16[1]!r})')
                    except OSError:
                        pass  # subprocess stdout closed
                except Exception as _p16_exc:  # noqa: BLE001
                    # P16 失败回退到 (0,0) 冷启动（与原行为一致），保证 train 不崩。
                    try:
                        print(f'[R-1 fix] P16 init failed, fallback cold start: {_p16_exc}')
                    except OSError:
                        pass  # subprocess stdout closed
        # 初始化当前序列的子报告
        seq_report = {
            'source_report': source_report,  # D3：引用共享，避免每样本深拷贝
            'geometry_bias_teacher': deepcopy(sample_report['geometry_bias_teacher']),  # D3：深拷贝避免嵌套结构与全局报告共享引用
            'teacher_quality_audit': _init_teacher_quality_audit(),
            'usable_sample_count': 0,
            'usable_sample_count_by_modality': {MODALITY_UWB: 0, MODALITY_VIO: 0},  # D9：单源常量
            'bias_source_counts': {'measured_minus_geometric_true_range': 0, 'neutral_baseline': 0},
            'ground_truth_alignment_modes': {},  # 真值对齐模式统计
            'skipped_outside_ground_truth_span': 0,  # 因超出真值时间跨度而跳过的事件数
            'skipped_uwb_updates_without_anchor_layout': 0,  # 因缺少锚点布局而跳过的 UWB 更新数
        }
        _accumulate_teacher_quality_audit(
            sample_report['teacher_quality_audit'],
            resolved_anchor_layout=resolved_anchor_layout,
            anchor_layout_report=anchor_layout_report,
            is_sequence_init=True,
        )
        _accumulate_teacher_quality_audit(
            seq_report['teacher_quality_audit'],
            resolved_anchor_layout=resolved_anchor_layout,
            anchor_layout_report=anchor_layout_report,
            is_sequence_init=True,
        )

        # 初始化历史事件列表和状态历史列表
        history: list[dict[str, Any]] = []
        state_history: list[dict[str, float]] = []
        # 初始化 readout 上下文缓存
        readout_context_cache = _init_training_readout_context_cache()
        # 遍历序列中的每个事件
        for event in events:
            timestamp = coerce_finite_scalar(event.get('t', event.get('timestamp', 0.0)), name='event.t')  # D5：coerce_finite_scalar 拒绝 NaN/Inf/非数值
            gt_state, gt_alignment = _align_ground_truth(gt_rows, timestamp)  # 将事件时间戳与真值对齐
            if gt_state is None:
                # 真值对齐失败，跳过该事件
                seq_report['skipped_outside_ground_truth_span'] += 1
                continue

            # 记录代理估计器更新前的状态快照，作为 prediction_state
            pre_update_proxy_state = _snapshot_proxy_estimator_state(proxy_estimator)
            if not pre_update_proxy_state:
                # 如果代理估计器没有状态，使用初始状态配置
                pre_update_proxy_state = {
                    'px': coerce_finite_scalar(init_state_cfg.get('px', 0.0), name='init_state.px'),  # D5：拒绝 NaN/Inf
                    'py': coerce_finite_scalar(init_state_cfg.get('py', 0.0), name='init_state.py'),  # D5：拒绝 NaN/Inf
                    'yaw': coerce_finite_scalar(init_state_cfg.get('yaw', 0.0), name='init_state.yaw'),  # D5：拒绝 NaN/Inf
                }
            # 将事件和状态加入历史
            history.append(event)
            state_history.append(dict(pre_update_proxy_state))

            modality = event.get('modality')  # D8：保持原始类型，避免 str(None) 静默为 'None'（与 _advance_proxy_estimator_for_training_sample L1006、fusion_runner L252 同口径）
            if modality not in {MODALITY_UWB, MODALITY_VIO}:  # D9：单源常量
                # 非 UWB/VIO 事件（如 IMU），仅推进代理估计器，不构造训练样本
                proxy_report = _advance_proxy_estimator_for_training_sample(
                    proxy_estimator,
                    event,
                    allow_uwb_update=resolved_anchor_layout is not None,
                )
                _update_training_readout_context_cache(
                    readout_context_cache,
                    proxy_report,
                    current_timestamp=timestamp,
                )
                continue
            if modality == MODALITY_UWB and resolved_anchor_layout is None:  # D9：单源常量
                # UWB 事件但缺少锚点布局，记录跳过原因
                seq_report['skipped_uwb_updates_without_anchor_layout'] += 1

            # 构建窗口特征张量
            window_tensor = window_builder(history, event, {'state_history': state_history})
            # 构建 readout 上下文
            readout_context_by_name, readout_context_observed_by_name = _build_training_readout_context(
                proxy_estimator,
                readout_context_cache,
                modality=modality,
                current_timestamp=timestamp,
            )
            # 将 readout 上下文注入窗口张量
            window_tensor = dict(window_tensor)
            window_tensor['readout_context_by_name'] = readout_context_by_name
            window_tensor['readout_context_observed_by_name'] = readout_context_observed_by_name
            # prediction_state 使用更新前的状态，保证因果语义
            prediction_state = dict(pre_update_proxy_state)
            # 构建目标中间值和审计 trace
            target_intermediate, target_trace = _build_target_intermediate(
                teacher_model=None,
                window_tensor=window_tensor,
                event=event,
                prediction_state=prediction_state,
                gt_state=gt_state,
                gt_alignment=gt_alignment,
                anchor_lookup=anchor_lookup or None,
                anchor_layout=resolved_anchor_layout,
                bridge_thresholds=bridge_thresholds,
                model_cfg=model_cfg,
            )

            # 组装训练样本
            # H24c 真改 §0.2「近重复检查」v3 / §27 scene_id 不可泄漏：
            # 把 scene_id 写入 sample dict，让 _split_train_and_val_samples 在切分时
            # 守门 train/val 之间不存在 shared scene_id（即同一 scene_id 不可同时出现
            # 在 train 与 val，否则 cross-set 泄漏 → 评估高层 RMSE 失公平性）。
            # scene_id 提取自 event.meta['scene_id']（decode_scene 严格解析口径），
            # 不存在时为 None；沿用 _decode_event_scene_spec 的提取口径。
            _sample_scene_id = None
            _sample_meta = event.get('meta')
            if isinstance(_sample_meta, Mapping):
                _raw_sid = _sample_meta.get('scene_id')
                if is_string_like(_raw_sid) and str(_raw_sid).strip():
                    _sample_scene_id = str(_raw_sid)
            sample = {
                'seq_id': seq_id,  # 序列 ID
                # H24c 真改：scene_id 写入 sample dict，供 _split_train_and_val_samples
                # 守门 train/val 不存在 shared scene_id（cross-set scene_id disjoint 守门）。
                'scene_id': _sample_scene_id,
                'event_time': timestamp,  # 事件时间戳
                'modality': modality,  # 事件模态
                'window_tensor': window_tensor,  # 窗口特征张量
                'target_intermediate': target_intermediate,  # 目标中间值（四头）
                # §13.7 内存压缩 + 审计合同折中：trainer 仅读 5 个标量，但样本级审计
                # （P24 可追溯）仍要求 bias_source 与几何 teacher 投影字段（
                # test_liquid_official_anchor_projection_smoke 断言）。这里保留
                # 5 标量 + bias_source + 几何 teacher 审计键（均为小标量），
                # 其余重组段不随样本保留，兼顾 §13.7 内存与样本级可审计性。
                'target_trace': {
                    'alignment_risk': target_trace.get('alignment_risk'),
                    'observation_risk': target_trace.get('observation_risk'),
                    'quality_risk': target_trace.get('quality_risk'),
                    'modality_signal': target_trace.get('modality_signal'),
                    'current_modality_gap_dt': target_trace.get('current_modality_gap_dt'),
                    'bias_source': target_trace.get('bias_source'),
                    'anchor_position': target_trace.get('anchor_position'),
                    'geometric_true_range': target_trace.get('geometric_true_range'),
                    'original_anchor_position_dim': target_trace.get('original_anchor_position_dim'),
                    'teacher_anchor_position_dim': target_trace.get('teacher_anchor_position_dim'),
                    'projection': target_trace.get('projection'),
                    'ignored_axis': target_trace.get('ignored_axis'),
                },
                'source_report': source_report,  # D3：引用共享，避免每样本深拷贝造成 N×11KB 的内存浪费
            }
            all_samples.append(sample)
            # 更新序列级和全局级样本统计
            seq_report['usable_sample_count'] += 1
            seq_report['usable_sample_count_by_modality'][modality] += 1
            sample_report['usable_sample_count'] += 1
            sample_report['usable_sample_count_by_modality'][modality] += 1
            # 统计 bias 来源
            bias_source = target_trace.get('bias_source', 'neutral_baseline')  # D8：保持原始类型，避免 str() 静默转换（值由 _build_target_intermediate L1861 保证为字符串）
            seq_report['bias_source_counts'][bias_source] = seq_report['bias_source_counts'].get(bias_source, 0) + 1
            sample_report['bias_source_counts'][bias_source] = sample_report['bias_source_counts'].get(bias_source, 0) + 1
            _accumulate_teacher_quality_audit(
                sample_report['teacher_quality_audit'],
                resolved_anchor_layout=resolved_anchor_layout,
                anchor_layout_report=anchor_layout_report,
                target_trace=target_trace,
                modality=modality,
            )
            _accumulate_teacher_quality_audit(
                seq_report['teacher_quality_audit'],
                resolved_anchor_layout=resolved_anchor_layout,
                anchor_layout_report=anchor_layout_report,
                target_trace=target_trace,
                modality=modality,
            )
            # 统计真值对齐模式
            alignment_mode = gt_alignment.get('mode', 'unknown')  # D8：保持原始类型，避免 str() 静默转换
            seq_report['ground_truth_alignment_modes'][alignment_mode] = seq_report['ground_truth_alignment_modes'].get(alignment_mode, 0) + 1
            # 推进代理估计器（构造样本后更新状态）
            proxy_report = _advance_proxy_estimator_for_training_sample(
                proxy_estimator,
                event,
                allow_uwb_update=resolved_anchor_layout is not None,
            )
            _update_training_readout_context_cache(
                readout_context_cache,
                proxy_report,
                current_timestamp=timestamp,
            )
            # §13.6.2.6 激活架构前提统计：真异步 Δt + 状态依赖脉冲
            # 真异步 Δt = modality_gap_dt > 0（训练集必须含异步间隔事件，否则 §13.5 架构前提未激活）
            # 状态依赖脉冲 = estimator.step 被实际调用（proxy_report 非 None 且不是 UWB 跳过报告）
            _async_gap_dt = float(target_trace.get('current_modality_gap_dt') or 0.0)
            if _async_gap_dt > 0.0:
                sample_report['async_dt_activated_sample_count'] += 1
                sample_report['async_dt_activated_modality_counts'][modality] += 1
            _is_uwb_skip = (
                isinstance(proxy_report, Mapping)
                and proxy_report.get('modality') == MODALITY_UWB
                and proxy_report.get('update_applied') is False
            )
            if proxy_report is not None and not _is_uwb_skip:
                sample_report['state_dependent_pulse_sample_count'] += 1

        # 将当前序列的子报告加入全局报告
        sample_report['sequences'][seq_id] = seq_report

    # OOM 诊断：函数exit内存
    try:
        import psutil as _psutil_diag
        _diag_proc = _psutil_diag.Process()
        _diag_mem_end = _diag_proc.memory_info().rss / 1024 / 1024
        _mem_delta = _diag_mem_end - _diag_mem_baseline
        try:
            print(f"[OOM_DIAG] _build_liquid_samples exit: RSS={_diag_mem_end:.0f} MB, delta={_mem_delta:.0f} MB, samples={len(all_samples)}")
        except OSError:
            pass  # subprocess stdout closed
        # OOM 检查
        if _mem_delta > 3000:
            import warnings
            warnings.warn(f"[OOM_DIAG] 函数内增长 {_mem_delta:.0f} MB! 可能在 all_samples 中物化了 {len(all_samples)} 个样本，每个样本约 {_mem_delta / len(all_samples):.0f} MB")
    except Exception as _e:
        pass

    return all_samples, sample_report


def _unique_in_order(values: Iterable[str] | None) -> list[str]:
    """去重并保持原始顺序。

    Args:
        values: 可迭代对象，元素须为可哈希的字符串。

    Returns:
        去重后的列表，保持首次出现的顺序。
    """
    ordered_values = list(values or [])
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in ordered_values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _split_train_and_val_samples(
    samples: list[dict[str, Any]],
    *,
    split_ids: list[str] | None = None,
    train_split_ids: list[str] | None = None,
    val_split_ids: list[str] | None = None,
    val_sequences_per_seed: int = 2,  # 手册 P33 硬约束：每 seed 从训练池中独立划分 2 条 val 序列
    allow_cross_set_leakage: bool = False,  # 2026-09-05 RZ-2 smoke bypass: K1 数据仅 1 scene, 同 scene 必需 (论文级需 N1.5+N2 多 N scene 才能消, 本机资源受约束, deviated R-1)
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    """将样本列表切分为训练集和验证集。

    切分策略：
    1. 如果显式指定了 train_split_ids 或 val_split_ids，按指定切分
    2. 如果只有 1 个序列，按时间顺序前半训练后半验证
    3. 如果有多个序列：手册 P33 硬约束 — 每 seed 独立划分 val_sequences_per_seed
       （默认 2）条 val 序列，其余做训练；7 变体共用同一划分（通过共享
       split_ids 列表传入实现；本函数本身只决定单次切分结果）

    Args:
        samples: 样本列表。
        split_ids: 全部序列 ID 列表。
        train_split_ids: 显式指定的训练序列 ID。
        val_split_ids: 显式指定的验证序列 ID。
        val_sequences_per_seed: 每 seed 隐式切分时 val 序列条数（默认 2，手册 P33）。

    Returns:
        (train_samples, val_samples, train_ids, val_ids) 元组。
    """
    ordered_samples = sorted(samples, key=lambda item: (str(item['seq_id']), float(item['event_time'])))
    unique_seq_ids = _unique_in_order(split_ids or [str(sample['seq_id']) for sample in ordered_samples])
    requested_train_ids = [str(seq_id) for seq_id in _unique_in_order(train_split_ids)]
    requested_val_ids = [str(seq_id) for seq_id in _unique_in_order(val_split_ids)]
    known_seq_id_set = set(unique_seq_ids)
    unknown_train_ids = [seq_id for seq_id in requested_train_ids if seq_id not in known_seq_id_set]
    if unknown_train_ids:
        raise ValueError(f'train_split_ids contains unknown sequence ids: {unknown_train_ids}')
    unknown_val_ids = [seq_id for seq_id in requested_val_ids if seq_id not in known_seq_id_set]
    if unknown_val_ids:
        raise ValueError(f'val_split_ids contains unknown sequence ids: {unknown_val_ids}')
    overlap_ids = sorted(set(requested_train_ids) & set(requested_val_ids))
    if overlap_ids:
        raise ValueError(f'train_split_ids and val_split_ids must be disjoint, got overlap: {overlap_ids}')
    if requested_train_ids or requested_val_ids:
        train_ids = requested_train_ids or [seq_id for seq_id in unique_seq_ids if seq_id not in requested_val_ids]
        val_ids = requested_val_ids or [seq_id for seq_id in unique_seq_ids if seq_id not in train_ids]
    elif len(unique_seq_ids) <= 1:
        train_ids = list(unique_seq_ids)
        val_ids = list(unique_seq_ids)
    else:
        # 手册 P33 硬约束：每 seed 隐式切分 val_sequences_per_seed=2 条 val 序列。
        # 7 变体共用同一 val 划分：上游调用方传入相同的 split_ids 列表
        # （来自 _resolve_split_for_variant 的统一切分entry），本函数决定单次切分结果。
        # 7 变体entry须显式传 val_split_ids（由"按 seed 划分 2 条"统一生成），
        # 避免每个变体独立随机造成 7 套不同 val 划分（破坏 P33「共用同一验证集」）。
        # 这里仍保留 fallback：用户未传 val_split_ids 时取最后 2 条作为 val。
        if len(unique_seq_ids) >= val_sequences_per_seed + 1:
            val_ids = list(unique_seq_ids[-val_sequences_per_seed:])
            train_ids = [seq_id for seq_id in unique_seq_ids if seq_id not in set(val_ids)]
        else:
            # 序列数不足时退回原行为：最后 1 条为 val
            train_ids = list(unique_seq_ids[:-1])
            val_ids = list(unique_seq_ids[-1:])

    if len(unique_seq_ids) <= 1 and not (requested_train_ids or requested_val_ids):
        split_index = max(1, len(ordered_samples) - max(1, len(ordered_samples) // 2))
        train_samples = ordered_samples[:split_index]
        val_samples = ordered_samples[split_index:]
        if not val_samples and ordered_samples:
            val_samples = [deepcopy(ordered_samples[-1])]  # D3：深拷贝避免与 train_samples 共享同一 dict（len==1 时 train 与 val fallback 指向同一对象，训练期就地修改会污染验证集，对齐 L2633 estimator_cfg 深拷贝同口径）。
        if not train_samples and ordered_samples:
            train_samples = ordered_samples[:-1] or ordered_samples[-1:]
    else:
        train_set = set(train_ids)
        val_set = set(val_ids)
        train_samples = [sample for sample in ordered_samples if str(sample['seq_id']) in train_set]
        val_samples = [sample for sample in ordered_samples if str(sample['seq_id']) in val_set]
    # H24c 真改 §0.2 / §27「scene_id 不可同时出现 train 与 val」cross-set 守门：
    # 显式 split_ids 切分下 train/val 不应共享 scene_id（否则 cross-set 泄漏 → 评估
    # 高层 RMSE 失公平性）。fall-through 路径（len(unique_seq_ids) <= 1 单序列切分）下
    # 不守门（同 seq_id 内时间切片允许共享 scene_id，因为 seq_id ≠ scene_id）。本守门
    # 只在 multi-sequence 显式切分下生效；缺 scene_id 字段的 sample 视为 None 跳过守门
    # （向后兼容旧 sample 无 scene_id 字段）。allow_cross_set_leakage=True 时绕过此守门
    # （用于 smoke 验证，K1 数据仅 1 scene 不可避免）。
    if requested_train_ids or requested_val_ids:
        train_scene_ids = {
            str(sample['scene_id']) for sample in train_samples
            if sample.get('scene_id') is not None
        }
        val_scene_ids = {
            str(sample['scene_id']) for sample in val_samples
            if sample.get('scene_id') is not None
        }
        shared_scene_ids = sorted(train_scene_ids & val_scene_ids)
        if shared_scene_ids and not allow_cross_set_leakage:
            raise ValueError(
                f"train/val split shares scene_id across sets (cross-set leakage): "
                f"{shared_scene_ids}. Adjust train_split_ids/val_split_ids so that "
                f"no scene_id appears in both train and val (§0.2 / §27). "
                f"Set allow_cross_set_leakage=True in cfg to bypass (smoke/dev only)."
            )
    return train_samples, val_samples, train_ids, val_ids


def _build_split_audit(
    *,
    split_ids: list[str],
    requested_train_ids: list[str],
    requested_val_ids: list[str],
    resolved_train_ids: list[str],
    resolved_val_ids: list[str],
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """构建训练/验证切分审计，显式记录序列级与时间级边界。"""

    normalized_split_ids = [str(seq_id) for seq_id in split_ids]  # 归一化全部序列 ID
    normalized_requested_train_ids = [str(seq_id) for seq_id in requested_train_ids]  # 归一化请求的训练 ID
    normalized_requested_val_ids = [str(seq_id) for seq_id in requested_val_ids]  # 归一化请求的验证 ID
    normalized_resolved_train_ids = [str(seq_id) for seq_id in resolved_train_ids]  # 归一化解析后的训练 ID
    normalized_resolved_val_ids = [str(seq_id) for seq_id in resolved_val_ids]  # 归一化解析后的验证 ID
    shared_seq_ids = sorted(set(normalized_resolved_train_ids) & set(normalized_resolved_val_ids))  # 训练/验证集重叠的序列 ID

    if normalized_requested_train_ids or normalized_requested_val_ids:
        split_strategy = 'explicit_sequence_split'  # 显式序列切分
    elif len(_unique_in_order(normalized_split_ids)) <= 1:
        split_strategy = 'single_sequence_time_split'  # 单序列时间切分
    else:
        split_strategy = 'default_last_sequence_holdout'  # 默认：最后一个序列做验证

    def _time_span(samples_: list[dict[str, Any]]) -> dict[str, Any] | None:
        """计算样本列表的时间跨度。"""
        if not samples_:
            return None
        event_times = [coerce_finite_scalar(sample['event_time'], name='split_audit.event_time') for sample in samples_]  # D5：用 coerce_finite_scalar 严格校验事件时间有限性，禁止 NaN/inf 泄漏到时间跨度和 chronological_nonoverlap 判定（对齐 L2068 event.t 同口径修复）。
        return {
            'start': float(min(event_times)),  # 最早事件时间
            'end': float(max(event_times)),  # 最晚事件时间
        }

    train_time_span = _time_span(train_samples)  # 训练集时间跨度
    val_time_span = _time_span(val_samples)  # 验证集时间跨度
    chronological_nonoverlap = None  # 时间序不重叠标志
    if train_time_span is not None and val_time_span is not None:
        chronological_nonoverlap = bool(train_time_span['end'] <= val_time_span['start'])  # 训练集结束 <= 验证集开始

    # H24c 真改 §0.2 / §27「scene_id 不可泄漏」cross-set 守门审计字段：
    # 训练/验证 sample 列表中各自提取 scene_id 集合，shared_scene_ids 为交集；
    # scene_id_disjoint = True 表示 train/val 之间无共享 scene_id（满足 §0.2 / §27）。
    # 缺 scene_id 字段的 sample 视为 None，不入集合（向后兼容旧 sample 无 scene_id 字段）。
    train_scene_ids = {
        str(sample['scene_id']) for sample in train_samples
        if sample.get('scene_id') is not None
    }
    val_scene_ids = {
        str(sample['scene_id']) for sample in val_samples
        if sample.get('scene_id') is not None
    }
    shared_scene_ids = sorted(train_scene_ids & val_scene_ids)
    scene_id_disjoint = not shared_scene_ids

    return {
        'split_strategy': split_strategy,
        'requested_split_ids': list(normalized_split_ids),
        'requested_train_split_ids': list(normalized_requested_train_ids),
        'requested_val_split_ids': list(normalized_requested_val_ids),
        'resolved_train_split_ids': list(normalized_resolved_train_ids),
        'resolved_val_split_ids': list(normalized_resolved_val_ids),
        'shared_seq_ids': shared_seq_ids,
        'sequence_disjoint': not shared_seq_ids,
        # H24c 真改：scene_id cross-set 守门审计字段
        'shared_scene_ids': shared_scene_ids,
        'scene_id_disjoint': scene_id_disjoint,
        'train_scene_ids': sorted(train_scene_ids),
        'val_scene_ids': sorted(val_scene_ids),
        'time_ordered_nonoverlap': chronological_nonoverlap,
        'train_window_count': len(train_samples),
        'val_window_count': len(val_samples),
        'train_event_time_span': train_time_span,
        'val_event_time_span': val_time_span,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    """将 JSON 数据写入文件，自动创建父目录。

    Args:
        path: 目标文件路径。
        payload: 待序列化的数据。

    Returns:
        文件路径的字符串形式。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json_text(payload), encoding='utf-8')  # dumps_json_text 已在 L64 顶层导入（D10），拒绝 NaN/Infinity
    return str(path)


def _build_training_flow_contract(model_name: str, train_report: Mapping[str, Any]) -> dict[str, Any]:
    """构建训练流程合同，记录模型名称、训练模式和四头输出合同。

    Args:
        model_name: 模型名称（liquid_ekf 或 lstm_ekf）。
        train_report: 训练报告字典。

    Returns:
        训练流程合同字典。
    """
    trainer_mode = train_report.get('trainer_mode') or ('single_phase_baseline' if model_name == MODEL_NAME_LSTM else 'phase_scheduled_liquid')  # 推断训练器模式
    contract: dict[str, Any] = {
        'model_name': model_name,  # 模型名称
        'trainer_mode': trainer_mode,  # 训练器模式
        'four_head_contract': list(MODEL_INTERMEDIATE_KEYS),  # 四头输出合同（单源常量，禁止本地重复定义）
        'shared_target_chain': True,  # 共享目标链
        'shared_split_chain': True,  # 共享切分链
        'shared_sample_report_chain': True,  # 共享样本报告链
        'risk_semantics': 'alignment_risk_only_pre_bridge_base_risk',  # risk 语义：仅对齐风险
        'scaling_semantics': 'teacher_free_modality_specific_noise_inflation_only',  # scaling 语义：teacher-free 模态特定噪声膨胀
        # L5 追溯字段根因修复：显式记录标签构造模式，对齐 docs/loss_function.md §实现约束 8。
        # 优先从 train_report 读取（trainer 写入），缺失时回退到单源真相默认值；LSTM 与 Liquid 共享同一标签链与口径。
        'risk_label_mode': train_report.get('risk_label_mode') or RISK_LABEL_MODE_DEFAULT,  # risk 标签构造模式（当前默认 alignment proxy）。
        'uwb_scaling_label_mode': train_report.get('uwb_scaling_label_mode') or UWB_SCALING_LABEL_MODE_DEFAULT,  # uwb_scaling 标签构造模式（当前默认 teacher-free 启发式）。
        'vio_scaling_label_mode': train_report.get('vio_scaling_label_mode') or VIO_SCALING_LABEL_MODE_DEFAULT,  # vio_scaling 标签构造模式（当前默认 teacher-free 启发式）。
        'optimizer_weight_decay_applied': float(train_report.get('optimizer_weight_decay_applied', 0.0)),  # 优化器权重衰减
        'loss_diagnostics_path': train_report.get('loss_diagnostics_path') or '',  # 损失诊断路径
        'epoch_predictions_vs_targets_path': train_report.get('epoch_predictions_vs_targets_path') or '',  # epoch 预测 vs 目标路径
        # §13.6.2.8 cross-trainer parity declaration: 把 trainer 写入的 cross_trainer_parity_declaration
        # 持久化到 training_flow_contract, 让离线审计 / 跨 run 比对能直接读 contract 文件
        # 验证 calibration / mono 等不对称是否合规声明.
        'cross_trainer_parity_declaration': deepcopy(train_report.get('cross_trainer_parity_declaration') or {}),
    }
    if 'tail_selection_observation_coeff' in train_report:
        contract['tail_selection_observation_coeff'] = float(train_report['tail_selection_observation_coeff'])
    if model_name == MODEL_NAME_LIQUID:  # Liquid 模型特有的合同字段
        contract.update(
            {
                'phase_schedule': deepcopy(train_report.get('phase_schedule') or {}),  # 阶段调度（深拷贝，审计快照独立于 train_report）
                'epoch_phase_names': deepcopy(train_report.get('epoch_phase_names') or []),  # epoch 阶段名称（深拷贝）
                'phase_trainability_contract': deepcopy(train_report.get('phase_trainability_contract') or {}),  # 阶段可训练性合同（深拷贝）
                'phase_effective_lrs': deepcopy(train_report.get('phase_effective_lrs') or {}),  # 阶段有效学习率（深拷贝）
                'filter_aware_readout_context': True,  # 滤波感知 readout 上下文
                # 从 train_report 动态读取，反映模型实例真实状态（trainer 从 model.risk_calibration 是否存在读取）。
                'risk_calibration_enabled': bool(train_report.get('risk_calibration_enabled', True)),
            }
        )
    else:  # LSTM 模型：无阶段调度
        contract.update(
            {
                'phase_schedule': {},
                'epoch_phase_names': [],
                'phase_trainability_contract': {},
                'phase_effective_lrs': {},
                'filter_aware_readout_context': False,  # LSTM 无滤波感知 readout 上下文
                # 从 train_report 动态读取，反映模型实例真实状态。
                # 当前代码事实：LSTM 也拥有 risk_calibration（见 model_factory._LSTMModel._build_modules），
                # 故默认回退为 True；若未来删除 LSTM 的 risk_calibration，trainer 会写入 False。
                'risk_calibration_enabled': bool(train_report.get('risk_calibration_enabled', True)),
            }
        )
    return contract


def _assert_cross_trainer_parity_declared_and_aligned() -> None:
    """校验 Liquid / LSTM / Transformer 三网 _CROSS_TRAINER_PARITY_DECLARATION 跨 trainer 对等性。

    §13.6.2.8 要求三网公平：选模权重 / calibration / mono / L1 / 选模口径等
    跨 trainer 必须对称；若存在结构性合法不对称（如 calibration weight 不同），
    必须在三边同时声明完全相同的 asymmetry_reason 字符串，否则 fail-loud。
    三网校验扩展（2026-08-31）：原实现只对 Liquid↔LSTM 校验，
    Transformer 导入但未参与 parity 比对，会让 Transformer 静默漂移。
    本轮扩展到三网同时校验，防止 Transformer 偏差污染 §14.1 主表公平性结论。
    bridge_thresholds 扩展（2026-08-31）：原 parity 声明只覆盖 trainer 内部超参，
    未覆盖 yaml 顶层 bridge_thresholds 块（9/16 字段 Liquid vs LSTM 漂移是
    RMSE 偏大根因之一）。本函数在 trainer 声明校验后，再加载三网 yaml 做硬比对。

    校验规则（按声明键逐一比对）：
    - 三方 asymmetry_reason 均为空 → value 必须字面相等，否则 fail-loud
    - 三方 asymmetry_reason 均非空 → 值可不同，但 asymmetry_reason 字符串必须完全一致，否则 fail-loud
    - 仅部分方声明非对称 → fail-loud（不对称必须三边共同声明）
    - 仅部分方声明了该键 → fail-loud（声明必须覆盖相同键集）
    - yaml bridge_thresholds 块三边字面相等 → 否则 fail-loud

    Raises:
        AssertionError: 违反上述任一条规则时 fail-loud，附带具体键名与差异描述。
    """
    liquid_keys = set(_LIQUID_PARITY_DECLARATION.keys())
    lstm_keys = set(_LSTM_PARITY_DECLARATION.keys())
    transformer_keys = set(_TRANSFORMER_PARITY_DECLARATION.keys())
    # 声明键集必须三边一致
    if liquid_keys != lstm_keys or liquid_keys != transformer_keys:
        only_liquid = liquid_keys - lstm_keys - transformer_keys
        only_lstm = lstm_keys - liquid_keys - transformer_keys
        only_transformer = transformer_keys - liquid_keys - lstm_keys
        # 「仅一方缺键、另两方都有」时三个 only_* 差集全为空（如 Liquid/LSTM/Transformer 中
        # LSTM 单独缺 gate_l1_weight），只报 only_* 无法定位缺失键；补报每方相对并集缺失的键。
        _union_keys = liquid_keys | lstm_keys | transformer_keys
        missing_in_liquid = sorted(_union_keys - liquid_keys)
        missing_in_lstm = sorted(_union_keys - lstm_keys)
        missing_in_transformer = sorted(_union_keys - transformer_keys)
        raise AssertionError(
            f"§13.6.2.8 cross-trainer parity declaration key mismatch: "
            f"missing keys: Liquid missing={missing_in_liquid}, LSTM missing={missing_in_lstm}, "
            f"Transformer missing={missing_in_transformer}; "
            f"Liquid-only keys={sorted(only_liquid)}, LSTM-only keys={sorted(only_lstm)}, "
            f"Transformer-only keys={sorted(only_transformer)}"
        )
    for key in sorted(liquid_keys):
        liquid_entry = _LIQUID_PARITY_DECLARATION[key]
        lstm_entry = _LSTM_PARITY_DECLARATION[key]
        transformer_entry = _TRANSFORMER_PARITY_DECLARATION[key]
        liquid_reason = str(liquid_entry.get("asymmetry_reason", ""))
        lstm_reason = str(lstm_entry.get("asymmetry_reason", ""))
        transformer_reason = str(transformer_entry.get("asymmetry_reason", ""))
        liquid_value = liquid_entry.get("value")
        lstm_value = lstm_entry.get("value")
        transformer_value = transformer_entry.get("value")
        # 提取声明非对称的 trainer 集合
        asymmetry_declarers = []
        if liquid_reason:
            asymmetry_declarers.append("Liquid")
        if lstm_reason:
            asymmetry_declarers.append("LSTM")
        if transformer_reason:
            asymmetry_declarers.append("Transformer")
        if len(asymmetry_declarers) == 0:
            # 对称项：三边值必须字面相等
            if liquid_value != lstm_value or liquid_value != transformer_value:
                raise AssertionError(
                    f"§13.6.2.8 cross-trainer parity violation (symmetric key): "
                    f"{key}: Liquid={liquid_value!r} != LSTM={lstm_value!r} != "
                    f"Transformer={transformer_value!r}; all asymmetry_reason are empty "
                    f"so values must match"
                )
        elif len(asymmetry_declarers) == 3:
            # 三边均声明非对称：reason 字符串必须完全一致
            if not (liquid_reason == lstm_reason == transformer_reason):
                raise AssertionError(
                    f"§13.6.2.8 cross-trainer parity violation (asymmetry reason mismatch): "
                    f"{key}: Liquid='{liquid_reason[:80]}' LSTM='{lstm_reason[:80]}' "
                    f"Transformer='{transformer_reason[:80]}'"
                )
        else:
            # 仅部分方声明非对称：不对称必须三边共同声明
            raise AssertionError(
                f"§13.6.2.8 cross-trainer parity violation (asymmetry declared by only some trainers): "
                f"{key}: declared by {asymmetry_declarers}, but must be declared by all three"
            )

    # §13.6.2.8 bridge_thresholds yaml 硬检查 (2026-08-31 扩展):
    # 原 parity 声明只覆盖 trainer 内部超参 (calibration_weight / mono_weight 等),
    # 未覆盖 yaml 顶层的 bridge_thresholds 块。9/16 字段 Liquid vs LSTM 漂移
    # 已被确认为 RMSE 偏大根因之一。本节直接加载三网 yaml，对 bridge_thresholds
    # 子字典做逐键字面相等检查 — 任何字段在三网间不一致即 fail-loud。
    # 若需测试临时漂移某字段，注释本段即可（不推荐）。
    try:
        import yaml as _yaml  # 局部 import 避免模块加载开销
        _model_cfg_dir = find_project_root() / "configs" / "models"
        _liq_yaml = _yaml.safe_load((_model_cfg_dir / "liquid_ekf.yaml").read_text(encoding="utf-8"))
        _lstm_yaml = _yaml.safe_load((_model_cfg_dir / "lstm_ekf.yaml").read_text(encoding="utf-8"))
        _trans_yaml = _yaml.safe_load((_model_cfg_dir / "transformer_ekf.yaml").read_text(encoding="utf-8"))
        _liq_bt = _liq_yaml.get("bridge_thresholds", {}) or {}
        _lstm_bt = _lstm_yaml.get("bridge_thresholds", {}) or {}
        _trans_bt = _trans_yaml.get("bridge_thresholds", {}) or {}
        _all_bt_keys = sorted(set(_liq_bt) | set(_lstm_bt) | set(_trans_bt))
        for _k in _all_bt_keys:
            _lv, _sv, _tv = _liq_bt.get(_k), _lstm_bt.get(_k), _trans_bt.get(_k)
            if _lv != _sv or _lv != _tv:
                raise AssertionError(
                    f"§13.6.2.8 yaml bridge_thresholds mismatch: "
                    f"key={_k!r} Liquid={_lv!r} != LSTM={_sv!r} != Transformer={_tv!r}; "
                    f"all three model yamls must use identical bridge_thresholds values"
                )
    except FileNotFoundError as _fnf_exc:
        # yaml 缺失时跳过此硬检查（CI / fixture 场景），仅记录 warning。
        import warnings as _w
        _w.warn(
            f"_assert_cross_trainer_parity_declared_and_aligned: yaml bridge_thresholds "
            f"check skipped ({_fnf_exc})",
            stacklevel=2,
        )


def _assert_async_pipeline_activated_in_train_samples(
    train_samples: list[dict[str, Any]],
    *,
    async_dt_min_activation_ratio: float = 0.05,
    state_dependent_pulse_min_count: int = 1,
    strict_fail_loud: bool = False,
) -> dict[str, Any]:
    """§13.6.2.6 激活架构前提 audit gate (第三轮精读补 + 可单元测试纯函数).

    §13.6.2.6 要求: "全程或至少最后进入定序的阶段, 数据须含真异步 Δt 与状态依赖脉冲
    (§13.5); 禁止仅在干净同步数据上预热并直接拿该 checkpoint 上高压测试定比较 1."

    本函数在训练前对 train_samples 做检查, 默认 audit-only (写入 sample_report 但不
    raise, 允许 fixture / 同步预热水暖运行); 当 `strict_fail_loud=True` 时再 raise,
    用于真实训练前显式启用 fail-loud 守门 (避免 fixture 测试 / 单序列同步预热被
    fail-loud 误伤).

    检查项:
    1. **真异步 Δt 激活**: train_samples 中 `target_trace['current_modality_gap_dt'] > 0`
       的样本占比 >= `async_dt_min_activation_ratio` (默认 5%).
    2. **状态依赖脉冲激活**: train_samples 至少 `state_dependent_pulse_min_count` 个样本
       的 `target_trace` 中存在 `current_modality_gap_dt` 字段 (即 target_trace 已被
       构造, 标志状态依赖脉冲链路已被激活).

    参数:
        train_samples: 训练样本列表, 每个样本是 dict 含 `target_trace` 字段.
        async_dt_min_activation_ratio: 真异步 Δt 激活样本占比的下限 (默认 0.05).
        state_dependent_pulse_min_count: 状态依赖脉冲样本最少数量 (默认 1).
        strict_fail_loud: True 时违反阈值会 raise RuntimeError; False (默认) 时仅返回
            audit 字典含 `violation` 字段而不 raise, 用于 fixture 测试 / 同步预热水暖
            场景的 backward-compat.

    返回:
        字典包含 `async_dt_activated_count` / `total_train_samples` /
        `async_dt_activation_ratio` / `state_dependent_pulse_count` / `violation`
        (None 表示通过, 否则为字符串描述违规) 用于审计 trace.
    """
    if not train_samples:
        return {
            'async_dt_activated_count': 0,
            'total_train_samples': 0,
            'async_dt_activation_ratio': 0.0,
            'state_dependent_pulse_count': 0,
            'violation': None,
            'skipped_no_samples': True,
        }
    async_dt_count = 0
    state_pulse_count = 0
    for sample in train_samples:
        if not isinstance(sample, Mapping):
            continue
        target_trace = sample.get('target_trace') or {}
        if not isinstance(target_trace, Mapping):
            continue
        # 真异步 Δt: modality_gap_dt > 0
        try:
            gap_dt = float(target_trace.get('current_modality_gap_dt') or 0.0)
        except (TypeError, ValueError):
            gap_dt = 0.0
        if gap_dt > 0.0:
            async_dt_count += 1
        # 状态依赖脉冲: target_trace 中存在 current_modality_gap_dt 字段 (含 0),
        # 即样本构造链路已被激活, 不是 silent absence.
        if 'current_modality_gap_dt' in target_trace:
            state_pulse_count += 1
    total = len(train_samples)
    ratio = (async_dt_count / total) if total > 0 else 0.0
    violation: str | None = None
    # 先检查 state_dependent_pulse (更基本的"链路是否被激活"维度),
    # 因为 state pulse 为 0 必然伴随 async_dt 也为 0, 提前触发更精准的根因诊断.
    if state_pulse_count < state_dependent_pulse_min_count:
        violation = (
            f"§13.6.2.6 state-dependent pulse activation fail-loud: "
            f"state_dependent_pulse_count={state_pulse_count} < required_min={state_dependent_pulse_min_count}. "
            f"target_trace['current_modality_gap_dt'] field absence indicates the sample builder "
            f"did not propagate state-dependent pulse information into the training set (§13.5)."
        )
    elif ratio < async_dt_min_activation_ratio:
        violation = (
            f"§13.6.2.6 async_dt activation ratio fail-loud: "
            f"async_dt_activated_count={async_dt_count} / total_train_samples={total} = "
            f"{ratio:.4f} < required_min={async_dt_min_activation_ratio}. "
            f"True async Δt must be activated in training (§13.5 architecture precondition); "
            f"if train split is dominated by synchronous grid pre-warm, §13.6.2.6 is violated."
        )
    if violation is not None and strict_fail_loud:
        raise RuntimeError(violation)
    return {
        'async_dt_activated_count': async_dt_count,
        'total_train_samples': total,
        'async_dt_activation_ratio': ratio,
        'state_dependent_pulse_count': state_pulse_count,
        'async_dt_min_activation_ratio': async_dt_min_activation_ratio,
        'state_dependent_pulse_min_count': state_dependent_pulse_min_count,
        'violation': violation,
        'strict_fail_loud': strict_fail_loud,
    }


def _resolve_output_root(cfg: Mapping[str, Any], *, default_name: str) -> Path:
    """解析输出根目录，委托给 common.paths.resolve_output_root 统一处理。"""
    return resolve_output_root(dict(cfg), default_name)  # shallow copy: resolve_output_root only reads, never mutates


def _run_scaffold_pipeline(cfg: dict[str, Any], gate_report: dict[str, Any]) -> StageResult:
    """运行 scaffold 模式的训练流水线（仅验证模型可创建）。

    Args:
        cfg: 流水线配置字典。
        gate_report: 协议门控报告。

    Returns:
        StageResult，包含 scaffold 模式的元数据。
    """
    model_name = cfg['model_name']  # 模型名称
    model_cfg = dict(cfg.get('model_cfg') or {})  # 模型配置
    create_model(model_name, model_cfg)  # 验证模型可创建（scaffold 模式，不训练）
    device_report = _resolve_train_device(cfg.get('device'), allow_auto_cuda=False)  # 解析设备（scaffold 禁止自动 CUDA）
    output_root = _resolve_output_root(cfg, default_name=_DEFAULT_TRAIN_OUTPUT_DIR_NAME)  # 解析输出根目录
    output_root.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    metadata = {
        'status': 'scaffold',
        'model_name': model_name,
        'protocol_gate': gate_report,
        'requested_device': device_report['requested_device'],
        'selected_device': device_report['selected_device'],
        'cuda_available': device_report['cuda_available'],
        'cuda_runtime_available': device_report['cuda_runtime_available'],
    }
    return StageResult(
        stage_name='train_pipeline',
        artifacts=[],
        metadata=metadata,
    )


def _run_real_frontend_pipeline(
    cfg: dict[str, Any],
    gate_report: dict[str, Any],
    *,
    train_model_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]], tuple[str, dict[str, Any]]],
) -> StageResult:
    """运行真实的 Liquid/LSTM 训练前端流水线。

    完整流程：解析配置 → 构建样本 → 切分训练/验证集 →
    调用训练器 → 保存报告和审计文件 → 验证 checkpoint。

    Args:
        cfg: 流水线配置字典。
        gate_report: 协议门控报告。
        train_model_fn: 训练器函数（train_liquid_model 或 train_lstm_model）。

    Returns:
        StageResult，包含训练报告、样本报告、目标合同和 checkpoint smoke。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "model_name": cfg.get("model_name"),
        "split_ids": cfg.get("split_ids"),
        "train_split_ids": cfg.get("train_split_ids"),
        "val_split_ids": cfg.get("val_split_ids"),
        "device": cfg.get("device"),
        "output_root": str(cfg.get("output_root")) if cfg.get("output_root") else None,
        "model_cfg_keys": list(cfg.get("model_cfg", {}).keys()) if isinstance(cfg.get("model_cfg"), dict) else None,
        "estimator_cfg_keys": list(cfg.get("estimator_cfg", {}).keys()) if isinstance(cfg.get("estimator_cfg"), dict) else None,
    }, "_run_real_frontend_pipeline entry参数")
    model_name = cfg['model_name']  # 模型名称（取值见 common.constants.MODEL_NAME_LIQUID / MODEL_NAME_LSTM，D9：禁止本地重复 'liquid_ekf'/'lstm_ekf' 字面量）
    model_cfg = dict(cfg.get('model_cfg') or {})  # 模型配置
    estimator_cfg = deepcopy(cfg.get('estimator_cfg') or _load_default_ekf_cfg())  # 估计器配置，默认从 ekf.yaml 加载。D3：深拷贝隔离嵌套结构，避免修改 process_noise/measurement_noise 等时污染 _load_default_ekf_cfg 的 lru_cache 或调用方 cfg['estimator_cfg']，与 core_pipeline._resolve_estimator_cfg L769 同口径。
    split_ids = list(gate_report.get('split_ids') or cfg.get('split_ids') or ['mini_seq'])  # 归一化后的序列 ID 列表
    device_report = _resolve_train_device(
        cfg.get('device'),
        allow_auto_cuda=_should_allow_auto_cuda(cfg),
    )
    output_root = _resolve_output_root(cfg, default_name=_DEFAULT_TRAIN_OUTPUT_DIR_NAME)  # 解析输出根目录

    samples, sample_report = _build_liquid_samples(
        split_ids,
        cfg,
        model_cfg,
        estimator_cfg,
    )
    train_samples, val_samples, resolved_train_split_ids, resolved_val_split_ids = _split_train_and_val_samples(
        samples,
        split_ids=split_ids,
        train_split_ids=list(cfg.get('train_split_ids') or []),
        val_split_ids=list(cfg.get('val_split_ids') or []),
        allow_cross_set_leakage=bool(cfg.get('allow_cross_set_leakage', False)),
    )
    if not train_samples:
        raise ValueError(
            f'train split resolved to zero usable windows: train_split_ids={resolved_train_split_ids}, '
            f'available_seq_ids={split_ids}'
        )
    if not val_samples:
        raise ValueError(
            f'val split resolved to zero usable windows: val_split_ids={resolved_val_split_ids}, '
            f'available_seq_ids={split_ids}'
        )
    _enforce_train_device_constraints(cfg, device_report)  # 强制设备约束（如禁止 checkpoint_path 复用）
    sample_report['train_window_count'] = len(train_samples)
    sample_report['val_window_count'] = len(val_samples)
    sample_report['train_split_ids'] = list(resolved_train_split_ids)
    sample_report['val_split_ids'] = list(resolved_val_split_ids)
    split_audit = _build_split_audit(
        split_ids=split_ids,
        requested_train_ids=list(cfg.get('train_split_ids') or []),
        requested_val_ids=list(cfg.get('val_split_ids') or []),
        resolved_train_ids=list(resolved_train_split_ids),
        resolved_val_ids=list(resolved_val_split_ids),
        train_samples=train_samples,
        val_samples=val_samples,
    )
    sample_report['split_audit'] = deepcopy(split_audit)  # D3：深拷贝隔离嵌套结构（train_event_time_span/val_event_time_span 为嵌套 dict），避免与 train_report['split_audit'] 及 target_contract['split_audit'] 共享引用
    sample_report['bridge_threshold_source'] = 'model_cfg.bridge_thresholds_or_defaults'

    frontend_train_cfg = dict(model_cfg)  # 构建训练器前端配置
    frontend_train_cfg['train'] = dict(model_cfg.get('train') or {})  # 训练超参数
    frontend_train_cfg['name'] = model_name  # 模型名称
    frontend_train_cfg['output_root'] = output_root  # 输出目录
    frontend_train_cfg['device'] = device_report['selected_device']  # 训练设备
    # §13.6.2.8 cross-trainer parity gate: 在 train_model_fn 调用前 fail-loud 校验两边
    # _CROSS_TRAINER_PARITY_DECLARATION 是否合规声明对称 / 非对称, 防 silent drift.
    _assert_cross_trainer_parity_declared_and_aligned()
    # §13.6.2.6 激活架构前提 audit gate: 训练前校验 train_samples 含真异步 Δt +
    # 状态依赖脉冲 (§13.5 架构前提在训练中实质激活), 防同步网格预热 checkpoint 直接
    # 用于高压测试或与比较 1 名次挂钩. 默认 audit-only (strict_fail_loud=False),
    # 仅当 cfg['strict_async_activation_fail_loud'] 显式启用时才 raise,
    # 避免 fixture / 单序列同步预热测试被误伤.
    _async_pipeline_activation_audit = _assert_async_pipeline_activated_in_train_samples(
        train_samples,
        strict_fail_loud=bool(cfg.get('strict_async_activation_fail_loud', False)),
    )
    sample_report['async_pipeline_activation_audit'] = dict(_async_pipeline_activation_audit)
    checkpoint_path, train_report = train_model_fn(train_samples, val_samples, frontend_train_cfg)  # 调用训练器函数
    train_report = dict(train_report)  # 转换训练报告为可变字典
    train_report['requested_device'] = device_report['requested_device']  # 请求的设备
    train_report['device'] = device_report['selected_device']  # 选定的设备
    train_report['cuda_available'] = device_report['cuda_available']  # CUDA 是否可用
    train_report['cuda_runtime_available'] = device_report['cuda_runtime_available']  # CUDA 运行时是否可用
    train_report['mode'] = _normalize_train_mode(cfg.get('mode'))  # 归一化训练模式
    train_report['closure_note'] = _build_execution_note(train_report['mode'])  # 执行备注
    train_report['train_split_ids'] = list(resolved_train_split_ids)  # 训练序列 ID
    train_report['val_split_ids'] = list(resolved_val_split_ids)  # 验证序列 ID
    # §13.6.2.7 反例 gate: 校验测试集序列 ID 不渗入 train_report 的 val_split_ids / train_split_ids / val_selection_scores 字段。
    # 训练 pipeline 无 test_split_ids（仅 train/val 切分），传空列表使 gate 为 no-op 但确保协议层可见。
    check_test_set_not_in_training_scores(train_report, [])
    train_report['split_audit'] = deepcopy(split_audit)  # D3：深拷贝隔离嵌套结构，避免与 sample_report['split_audit'] 及 target_contract['split_audit'] 共享引用
    train_report['bridge_thresholds'] = dict(sample_report['bridge_thresholds'])  # 桥接阈值
    training_flow_contract = _build_training_flow_contract(model_name, train_report)
    training_flow_contract_path = _write_json(
        output_root / 'audits' / f'{model_name}_training_flow_contract.json',
        training_flow_contract,
    )
    train_report['training_flow_contract_path'] = training_flow_contract_path
    train_report['report_path'] = _write_json(
        output_root / 'reports' / f'{model_name}_train_report.json',
        train_report,
    )
    training_flow_contract['train_report_path'] = train_report['report_path']
    training_flow_contract_path = _write_json(
        output_root / 'audits' / f'{model_name}_training_flow_contract.json',
        training_flow_contract,
    )

    loaded_model = create_model(model_name, {'checkpoint_path': checkpoint_path})  # 加载训练好的 checkpoint 模型
    checkpoint_outputs = loaded_model.infer_intermediate(val_samples[0]['window_tensor'])  # 在第一个验证样本上运行推理
    checkpoint_smoke = {
        'status': 'ok',  # smoke 测试状态
        'checkpoint_path': checkpoint_path,  # checkpoint 文件路径
        'val_seq_id': val_samples[0]['seq_id'],  # 验证样本序列 ID
        'val_event_time': coerce_finite_scalar(val_samples[0]['event_time'], name='checkpoint_smoke.val_event_time'),  # D5：coerce_finite_scalar 严格校验事件时间有限性，禁止 NaN/inf 写入 checkpoint_smoke 审计（对齐 _build_split_audit L2476 event_time 同口径）
        'inference_outputs': _model_intermediate_to_dict(checkpoint_outputs),  # 推理四头输出
    }

    target_contract = {
        'required_heads': list(_TARGET_KEYS),  # 必需的输出头名称列表
        'risk_semantics': 'alignment_risk_only_pre_bridge_base_risk',  # risk 语义：仅桥接前基础对齐风险
        'confidence_materialization': 'trainer consumes confidence as modality-specific scaling heads with a neutral floor, plus a conditional robust low-quality supplement',  # 置信度物化方式
        'head_sources': {  # 各头的来源说明
            'bias': 'UWB samples use max(0.0, measured_range - geometric_true_range) from anchor_layout metadata when available; all other samples use a neutral baseline',  # bias 来源：UWB 使用几何 bias，其他使用中性基线
            'risk': 'alignment_risk only; base risk comes from max(normalized pose_error / failure_threshold_m, normalized yaw_error / pi) against exact, interpolated, or trailing-tolerant ground truth, while observation_risk remains an audit trace that is consumed by modality-specific scaling and downstream bridge noise logic instead of the risk head',  # risk 来源：仅对齐风险
            'uwb_scaling': 'UWB-only extra noise inflation from quality, validity, geometric bias severity, modality_gap_dt staleness, and alignment_risk with a neutral floor, plus a conditional robust low-quality supplement',  # uwb_scaling 来源
            'vio_scaling': 'VIO-only extra noise inflation from quality, tracked_features, reproj_err, modality_gap_dt staleness, and alignment_risk with a neutral floor, plus a conditional robust low-quality supplement',  # vio_scaling 来源
        },
        'bridge_thresholds': dict(sample_report['bridge_thresholds']),  # 桥接阈值（扁平 {str: float} 字典，浅拷贝安全）
        'geometry_bias_teacher': deepcopy(sample_report['geometry_bias_teacher']),  # D3：深拷贝隔离嵌套结构（observed_inputs/blockers/projection_audit），避免与 sample_report 共享引用
        'teacher_quality_audit': dict(sample_report['teacher_quality_audit']),  # teacher 质量审计（扁平 {str: int} 字典，浅拷贝安全）
        'split_audit': deepcopy(sample_report['split_audit']),  # D3：深拷贝隔离嵌套结构（train/val_event_time_span），避免与 sample_report/train_report 共享引用
        'ground_truth_alignment': {  # 真值对齐策略
            'in_span': 'exact_or_linear_interpolation',  # 跨度内：精确匹配或线性插值
            'trailing_strategy': 'reuse_last_ground_truth_within_one_trailing_gt_interval',  # 尾部策略：在一个尾部间隔内复用最后真值
            'leading_strategy': 'disabled_to_avoid_future_leakage',  # 前导策略：禁用以避免未来泄漏
        },
    }

    audits_dir = output_root / 'audits'  # 审计文件输出目录
    artifacts = [checkpoint_path, train_report['report_path']]  # 核心产物：checkpoint 和训练报告
    artifacts.append(_write_json(audits_dir / f'{model_name}_protocol_gate.json', gate_report))  # 协议门控审计
    artifacts.append(_write_json(audits_dir / f'{model_name}_sample_report.json', sample_report))  # 样本报告审计
    artifacts.append(_write_json(audits_dir / f'{model_name}_target_contract.json', target_contract))  # 目标合同审计
    artifacts.append(_write_json(audits_dir / f'{model_name}_checkpoint_smoke.json', checkpoint_smoke))  # checkpoint smoke 审计
    artifacts.append(training_flow_contract_path)  # 训练流程合同审计

    return StageResult(
        stage_name='train_pipeline',
        artifacts=artifacts,
        metadata={
            'train_report': train_report,
            'protocol_gate': gate_report,
            'sample_report': sample_report,
            'target_contract': target_contract,
            'checkpoint_smoke': checkpoint_smoke,
            'training_flow_contract': training_flow_contract,
        },
    )


def _run_liquid_real_pipeline(cfg: dict[str, Any], gate_report: dict[str, Any]) -> StageResult:
    """运行 Liquid 模型的真实训练流水线。

    当前全链路为 teacher-free 模式，不接受 teacher_checkpoint_path。

    Args:
        cfg: 流水线配置字典。
        gate_report: 协议门控报告。

    Returns:
        StageResult。

    Raises:
        ValueError: 提供了 teacher_checkpoint_path。
    """
    teacher_checkpoint_path = cfg.get('teacher_checkpoint_path')  # 读取 teacher checkpoint 路径
    if teacher_checkpoint_path is not None:
        # D8：禁止 str() 静默转换非字符串值；非字符串即配置合同违例
        if not is_string_like(teacher_checkpoint_path):
            raise ValueError(f'{MODEL_NAME_LIQUID}: teacher_checkpoint_path 必须为字符串或 None，实际类型 {type(teacher_checkpoint_path).__name__}')
        if teacher_checkpoint_path.strip():
            raise ValueError(f'{MODEL_NAME_LIQUID} does not accept teacher_checkpoint_path because the current full chain is teacher-free')  # teacher-free 模式禁止 teacher checkpoint
    return _run_real_frontend_pipeline(
        cfg,
        gate_report,
        train_model_fn=train_liquid_model,
    )


def _run_lstm_real_pipeline(cfg: dict[str, Any], gate_report: dict[str, Any]) -> StageResult:
    """运行 LSTM 模型的真实训练流水线。

    当前全链路为 teacher-free 模式，不接受 teacher_checkpoint_path。

    Args:
        cfg: 流水线配置字典。
        gate_report: 协议门控报告。

    Returns:
        StageResult。

    Raises:
        ValueError: 提供了 teacher_checkpoint_path。
    """
    teacher_checkpoint_path = cfg.get('teacher_checkpoint_path')  # 读取 teacher checkpoint 路径
    if teacher_checkpoint_path is not None:
        # D8：禁止 str() 静默转换非字符串值；非字符串即配置合同违例
        if not is_string_like(teacher_checkpoint_path):
            raise ValueError(f'{MODEL_NAME_LSTM}: teacher_checkpoint_path 必须为字符串或 None，实际类型 {type(teacher_checkpoint_path).__name__}')
        if teacher_checkpoint_path.strip():
            raise ValueError(f'{MODEL_NAME_LSTM} does not accept teacher_checkpoint_path because the current full chain is teacher-free')  # teacher-free 模式禁止 teacher checkpoint
    return _run_real_frontend_pipeline(
        cfg,
        gate_report,
        train_model_fn=train_lstm_model,
    )


def _run_transformer_real_pipeline(cfg: dict[str, Any], gate_report: dict[str, Any]) -> StageResult:
    """Transformer 模型训练entry：与 Liquid/LSTM 同口径的 real frontend pipeline 路径。

    与 _run_lstm_real_pipeline 对等：仅校验 teacher_checkpoint_path 必须为 None 或空字符串
    （当前全链路为 teacher-free），然后委托给 _run_real_frontend_pipeline，传入 transformer trainer。
    """
    teacher_checkpoint_path = cfg.get('teacher_checkpoint_path')
    if teacher_checkpoint_path is not None:
        if not is_string_like(teacher_checkpoint_path):
            raise ValueError(f'{MODEL_NAME_TRANSFORMER}: teacher_checkpoint_path 必须为字符串或 None，实际类型 {type(teacher_checkpoint_path).__name__}')
        if teacher_checkpoint_path.strip():
            raise ValueError(f'{MODEL_NAME_TRANSFORMER} does not accept teacher_checkpoint_path because the current full chain is teacher-free')
    return _run_real_frontend_pipeline(
        cfg,
        gate_report,
        train_model_fn=train_transformer_model,
    )


class TrainPipeline(PipelineAPI):
    """训练流水线类，支持 scaffold、Liquid 和 LSTM 三种模式。"""

    def run(self, pipeline_cfg: dict | None = None, runtime_context: dict | None = None) -> StageResult:
        """运行训练流水线，根据模型名称自动选择执行路径。

        Args:
            pipeline_cfg: 流水线配置字典，须包含 model_name。
            runtime_context: 运行时上下文（当前未使用）。

        Returns:
            StageResult，包含训练报告和审计文件。

        Raises:
            ValueError: model_name 未提供。
        """
        # P19 硬约束：train_pipeline entry处设置确定性种子。
        # P19-2 必须在所有 PYTHONHASHSEED 启动脚本（run_train.sh / conda activate）中
        # 预先设 PYTHONHASHSEED=0；此处仅做审计提示。
        from liquidloc.common.seed_protocol import (
            ensure_pythonhashseed, set_pipeline_seed, TRAIN_SEEDS, get_full_seed_grid,
        )
        ensure_pythonhashseed(seed=0)
        # P19-3: pipeline entry设种子：取训练 seed（若无则用 TRAIN_SEEDS[0]）。
        train_seed_raw = (pipeline_cfg or {}).get("seed")
        if train_seed_raw is None:
            effective_seed = TRAIN_SEEDS[0]
        elif isinstance(train_seed_raw, int):
            effective_seed = train_seed_raw
        else:
            effective_seed = int(str(train_seed_raw))
        seed_audit = set_pipeline_seed(int(effective_seed), deterministic=True)

        cfg = normalize_pipeline_cfg(pipeline_cfg)  # 流水线配置字典，并拒绝非映射输入
        from liquidloc.common.tee_logger import print_dict
        print_dict({
            "model_name": cfg.get("model_name"),
            "split_ids": cfg.get("split_ids"),
            "device": cfg.get("device"),
            "mode": cfg.get("mode"),
            "has_model_cfg": "model_cfg" in cfg,
            "has_estimator_cfg": "estimator_cfg" in cfg,
            "output_root": str(cfg.get("output_root")) if cfg.get("output_root") else None,
            "experiment_protocol_path": str(cfg.get("experiment_protocol_path")) if cfg.get("experiment_protocol_path") else None,
            "p19_seed_audit": seed_audit,
            "p19_seed_grid_size": len(get_full_seed_grid()),
        }, "TrainPipeline.run entry参数")
        model_name = cfg.get('model_name')  # 获取模型名称
        if not model_name:
            raise ValueError('model_name must be provided')  # 缺少模型名称错误

        split_ids = list(cfg.get('split_ids') or ['mini_seq'])  # 序列 ID 列表
        protocol_cfg = load_experiment_protocol(cfg.get('experiment_protocol_path'))  # 加载实验协议配置
        gate_report = normalize_train_request(cfg | {'split_ids': split_ids}, protocol_cfg)  # 验证训练请求合法性
        normalized_gate_report = dict(gate_report)
        normalized_split_ids = list(normalized_gate_report.pop('split_ids', split_ids))
        normalized_train_split_ids = list(normalized_gate_report.pop('train_split_ids', []))
        normalized_val_split_ids = list(normalized_gate_report.pop('val_split_ids', []))
        normalized_cfg = cfg | {
            'split_ids': normalized_split_ids,
            'train_split_ids': normalized_train_split_ids,
            'val_split_ids': normalized_val_split_ids,
        }

        # P39 持久化：git commit + config hash 一并记入 gate_report，供下游 stage_result 写出。
        try:
            # 修复: _resolve_git_commit/_resolve_config_hash 定义在 core_pipeline, 此前未导入
            # 导致 NameError 被 except 吞掉, p39_error 污染 train_report。
            from liquidloc.pipelines.core_pipeline import _resolve_config_hash as _p39_config_hash
            from liquidloc.pipelines.core_pipeline import _resolve_git_commit as _p39_git_commit
            normalized_gate_report['p39_git_commit'] = _p39_git_commit() or "no-git"
            normalized_gate_report['p39_config_hash'] = _p39_config_hash(normalized_cfg)
        except Exception as _p39_exc:
            normalized_gate_report['p39_error'] = str(_p39_exc)

        if model_name == MODEL_NAME_LIQUID:
            return _run_liquid_real_pipeline(normalized_cfg, normalized_gate_report)  # Liquid 模型训练
        if model_name == MODEL_NAME_LSTM:
            return _run_lstm_real_pipeline(normalized_cfg, normalized_gate_report)  # LSTM 模型训练
        if model_name == MODEL_NAME_TRANSFORMER:
            return _run_transformer_real_pipeline(normalized_cfg, normalized_gate_report)  # Transformer 模型训练
        return _run_scaffold_pipeline(normalized_cfg, normalized_gate_report)  # scaffold 模式（仅验证模型可创建）


def run(pipeline_cfg: dict | None = None) -> StageResult:
    """便捷函数：创建 TrainPipeline 实例并调用 run。

    Args:
        pipeline_cfg: 流水线配置字典，须包含 model_name。

    Returns:
        StageResult，包含训练报告和审计文件。

    Raises:
        ValueError: model_name 未提供。
    """
    return TrainPipeline().run(pipeline_cfg)
