"""协议层统一导出入口。

文件职责：
  把 protocol 子模块中最常用的常量、数据结构和校验函数集中导出，
  让上层代码只需 ``from liquidloc.protocol import ...`` 即可获取所有协议要素，
  无需关心内部文件划分。

本文件绝对不负责：
  不定义任何协议内容本身，只做转发和重导出。

上游依赖：
  liquidloc.common.constants（常量）、
  liquidloc.protocol.event_schema（事件校验）、
  liquidloc.protocol.experiment_gates（实验闸门）、
  liquidloc.protocol.liquid_bridge_contract（桥接契约）、
  liquidloc.protocol.metric_schema（指标元数据）、
  liquidloc.protocol.output_contract_schema（输出契约）、
  liquidloc.protocol.result_schema（结果载荷）、
  liquidloc.protocol.scene_axis_protocol（场景轴协议）、
  liquidloc.protocol.scene_schema（场景编码）、
  liquidloc.protocol.version（版本号）

下游调用者：
  scenarios/、pipelines/、fusion/、estimators/、dataio/、analysis/ 等几乎所有上层模块。

核心变量：
  __all__：对外导出名称元组，控制 ``from liquidloc.protocol import *`` 的范围。
"""

from liquidloc.common.constants import ALLOWED_MODALITIES  # 允许的模态集合。
from liquidloc.common.constants import DEFAULT_THRESHOLDS  # 默认阈值集合（纯算法/协议级）。
from liquidloc.common.constants import META_KEYS  # 事件 meta 需要的键。
from liquidloc.common.constants import MODALITY_IMU  # IMU 模态常量。
from liquidloc.common.constants import MODALITY_UWB  # UWB 模态常量。
from liquidloc.common.constants import MODALITY_VIO  # VIO 模态常量。
from liquidloc.common.constants import MODALITY_FLOW  # 光流/流量类模态常量（UTIL 数据集）。
from liquidloc.common.constants import MODALITY_TOF  # ToF 测距类模态常量（UTIL 数据集）。
from liquidloc.common.constants import PAYLOAD_KEYS  # 各模态 payload 键映射。
from liquidloc.common.constants import PRIMARY_EVENT_KEYS  # 事件主键集合。
from liquidloc.protocol.bridge_thresholds import BRIDGE_THRESHOLDS  # 桥接层业务阈值集合。
from liquidloc.protocol.event_schema import Event  # 事件数据结构。
from liquidloc.protocol.event_schema import coerce_event_for_feature_extraction  # 事件宽松校验函数。
from liquidloc.protocol.event_schema import validate_event  # 事件严格校验函数。
from liquidloc.protocol.event_schema import validate_event_sequence  # 事件序列校验函数。
from liquidloc.protocol.experiment_gates import get_default_failure_threshold_m  # 获取协议默认失效阈值（米）。
from liquidloc.protocol.experiment_gates import get_public_benchmark_allowed_datasets  # 获取公开基准允许的数据集列表。
from liquidloc.protocol.experiment_gates import load_experiment_protocol  # 加载实验协议。
from liquidloc.protocol.experiment_gates import normalize_eval_request  # 规范化评测请求。
from liquidloc.protocol.experiment_gates import normalize_eval_request_shape  # 规范化评测请求参数形状（不依赖文件系统）。
from liquidloc.protocol.experiment_gates import normalize_public_benchmark_request  # 规范化公开基准请求。
from liquidloc.protocol.experiment_gates import normalize_public_dataset_name  # 规范化公开数据集名称。
from liquidloc.protocol.experiment_gates import normalize_run_mode  # 规范化运行模式。
from liquidloc.protocol.experiment_gates import normalize_train_request  # 规范化训练请求。
from liquidloc.protocol.experiment_gates import validate_eval_request  # 兼容旧公开接口名的评测请求校验。
from liquidloc.protocol.experiment_gates import validate_public_benchmark_request  # 兼容旧公开接口名的公开基准请求校验。
from liquidloc.protocol.experiment_gates import validate_train_request  # 兼容旧公开接口名的训练请求校验。
from liquidloc.protocol.liquid_bridge_contract import LiquidBridgeDecision  # 桥接决策对象。
from liquidloc.protocol.liquid_bridge_contract import apply_safe_mode  # 安全模式动作判断。
from liquidloc.protocol.liquid_bridge_contract import build_measurement_control  # 构建测量控制对象。
from liquidloc.protocol.liquid_bridge_contract import normalize_risk  # 风险归一化。
from liquidloc.protocol.metric_schema import get_metric_meta  # 获取指标元数据。
from liquidloc.protocol.metric_schema import get_metric_order  # 获取指标顺序。
from liquidloc.protocol.metric_schema import get_primary_metrics  # 获取 primary 指标列表。
from liquidloc.protocol.metric_schema import MetricMeta  # 指标元数据 TypedDict。
from liquidloc.protocol.output_contract_schema import check_output_contract  # 输出结构检查函数。
from liquidloc.protocol.result_schema import ExperimentResult  # 实验结果对象。
from liquidloc.protocol.result_schema import SequenceResult  # 单序列结果对象。
from liquidloc.protocol.result_schema import SummaryResult  # 汇总结果对象。
from liquidloc.protocol.result_schema import validate_experiment_result  # 校验实验结果。
from liquidloc.protocol.result_schema import validate_sequence_result  # 校验单序列结果。
from liquidloc.protocol.result_schema import validate_summary_result  # 校验汇总结果。
from liquidloc.protocol.risk_projection import project_risk_to_protocol_range  # 风险映射到协议区间的权威公式（协议层唯一实现，禁止其他层重实现）。
from liquidloc.protocol.scene_axis_protocol import SceneParameters  # 场景参数数据结构。
from liquidloc.protocol.scene_axis_protocol import attach_scene_parameters  # 展开场景参数。
from liquidloc.protocol.scene_axis_protocol import load_scene_axis_protocol  # 加载场景轴协议。
from liquidloc.protocol.scene_axis_protocol import resolve_axis_level  # 解析单轴层级。
from liquidloc.protocol.scene_schema import SceneSpec  # 场景规格对象。
from liquidloc.protocol.scene_schema import axis_levels  # 获取轴层级表。
from liquidloc.protocol.scene_schema import coerce_axis_value  # 单轴值校验函数。
from liquidloc.protocol.scene_schema import decode_scene  # 解码场景字符串。
from liquidloc.protocol.scene_schema import encode_scene  # 编码场景字符串。
from liquidloc.protocol.version import CONFIG_VERSION  # 配置版本号。
from liquidloc.protocol.version import OUTPUT_CONTRACT_VERSION  # 输出契约版本号。
from liquidloc.protocol.version import PROTOCOL_VERSION  # 协议版本号。
from liquidloc.protocol.version import SNAPSHOT_VERSION  # 快照版本号。
from liquidloc.protocol.version import summarize_versions  # 汇总版本号。

__all__ = (  # 对外导出列表开始。
    "ALLOWED_MODALITIES",  # 允许的模态集合。
    "BRIDGE_THRESHOLDS",  # 桥接层业务阈值集合。
    "CONFIG_VERSION",  # 配置版本号。
    "DEFAULT_THRESHOLDS",  # 默认阈值集合（纯算法/协议级）。
    "Event",  # 事件数据结构。
    "ExperimentResult",  # 实验结果对象。
    "LiquidBridgeDecision",  # 桥接决策对象。
    "META_KEYS",  # 事件 meta 需要的键。
    "MetricMeta",  # 指标元数据 TypedDict。
    "MODALITY_IMU",  # IMU 模态常量。
    "MODALITY_UWB",  # UWB 模态常量。
    "MODALITY_VIO",  # VIO 模态常量。
    "MODALITY_FLOW",  # 光流/流量类模态常量。
    "MODALITY_TOF",  # ToF 测距类模态常量。
    "OUTPUT_CONTRACT_VERSION",  # 输出契约版本号。
    "PAYLOAD_KEYS",  # 各模态 payload 键映射。
    "PRIMARY_EVENT_KEYS",  # 事件主键集合。
    "PROTOCOL_VERSION",  # 协议版本号。
    "SceneParameters",  # 场景参数数据结构。
    "SceneSpec",  # 场景规格对象。
    "SequenceResult",  # 单序列结果对象。
    "SNAPSHOT_VERSION",  # 快照版本号。
    "SummaryResult",  # 汇总结果对象。
    "attach_scene_parameters",  # 展开场景参数。
    "apply_safe_mode",  # 安全模式动作判断。
    "axis_levels",  # 获取轴层级表。
    "build_measurement_control",  # 构建测量控制对象。
    "coerce_axis_value",  # 单轴值校验函数。
    "coerce_event_for_feature_extraction",  # 事件宽松校验函数。
    "decode_scene",  # 解码场景字符串。
    "encode_scene",  # 编码场景字符串。
    "get_default_failure_threshold_m",  # 获取协议默认失效阈值（米）。
    "get_metric_meta",  # 获取指标元数据。
    "get_metric_order",  # 获取指标顺序。
    "get_primary_metrics",  # 获取 primary 指标列表。
    "get_public_benchmark_allowed_datasets",  # 获取公开基准允许的数据集列表。
    "load_experiment_protocol",  # 加载实验协议。
    "load_scene_axis_protocol",  # 加载场景轴协议。
    "normalize_eval_request",  # 规范化评测请求。
    "normalize_eval_request_shape",  # 规范化评测请求参数形状（不依赖文件系统）。
    "normalize_public_benchmark_request",  # 规范化公开基准请求。
    "normalize_public_dataset_name",  # 规范化公开数据集名称。
    "normalize_run_mode",  # 规范化运行模式。
    "normalize_train_request",  # 规范化训练请求。
    "normalize_risk",  # 风险归一化。
    "project_risk_to_protocol_range",  # 风险映射到协议区间的权威公式。
    "resolve_axis_level",  # 解析单轴层级。
    "summarize_versions",  # 汇总版本号。
    "validate_eval_request",  # 兼容旧公开接口名的评测请求校验。
    "validate_event",  # 事件严格校验函数。
    "validate_event_sequence",  # 事件序列校验函数。
    "validate_experiment_result",  # 校验实验结果。
    "check_output_contract",  # 输出结构检查函数。
    "validate_public_benchmark_request",  # 兼容旧公开接口名的公开基准请求校验。
    "validate_sequence_result",  # 校验单序列结果。
    "validate_summary_result",  # 校验汇总结果。
    "validate_train_request",  # 兼容旧公开接口名的训练请求校验。
)  # 对外导出列表结束。
