"""common 子包的统一导出入口。

职责：
    把角度工具、常量、路径工具、轻量类型和校验函数集中导出，
    让上层模块可以从一个稳定入口拿到最基础的公共能力。

上游依赖：
    - liquidloc.common.angle_utils   — 角度归一化与角差计算
    - liquidloc.common.constants     — 模态名、目录名、阈值等跨模块常量
    - liquidloc.common.paths         — 项目根目录与标准输出路径构造
    - liquidloc.common.types         — 类型别名与 dataclass 容器
    - liquidloc.common.validation    — 非空、键集合、范围、形状校验

下游调用者：
    - liquidloc.protocol.*           — 协议层通过 common 获取常量与类型
    - liquidloc.pipelines.*          — 流水线层通过 common 获取路径与校验
    - liquidloc.estimators.*         — 估计器层通过 common 获取角度工具与类型
    - scripts / notebooks            — 脚本和笔记本通过 common 获取公共能力

核心变量：
    - ALLOWED_MODALITIES             — 允许的模态集合
    - DEFAULT_OUTPUT_DIRS            — 默认输出目录集合
    - DEFAULT_THRESHOLDS             — 默认阈值表
    - PRIMARY_KEYS                   — 主键注册表
"""

from liquidloc.common.angle_utils import angle_delta_rad, wrap_angle_rad  # 导出角度归一化和角差工具。
from liquidloc.common.constants import (  # 导出跨模块共享的模态名、目录名和阈值常量。
    ALLOWED_MODALITIES,  # 允许的模态集合。
    DEFAULT_OUTPUT_DIRS,  # 默认输出目录集合。
    DEFAULT_REQUIRED_OUTPUT_FILES,  # 默认必需产物列表。
    DEFAULT_THRESHOLDS,  # 默认阈值表。
    MODALITY_IMU,  # IMU 模态名。
    MODALITY_UWB,  # UWB 模态名。
    MODALITY_VIO,  # VIO 模态名。
    MODALITY_FLOW,  # 光流/流量类模态名（UTIL 数据集）。
    MODALITY_TOF,  # ToF 测距类模态名（UTIL 数据集）。
    PAYLOAD_KEYS,  # 各模态对应的载荷键名。
    PRIMARY_EVENT_KEYS,  # 主事件键名。
    PRIMARY_KEYS,  # 主键注册表。
)  # 常量导入结束。
from liquidloc.common.paths import build_output_path, get_project_root, get_standard_dirs, resolve_output_root  # 导出路径相关工具。
from liquidloc.common.types import (  # 导出常用类型别名和协议容器类型。
    MeasurementControl,  # 观测控制类型。
    MetricDirection,  # 指标方向类型。
    MetricRow,  # 指标行类型。
    ModelIntermediate,  # 模型中间态类型。
    PathLike,  # 路径类类型别名。
    PredictionBundle,  # 预测包类型。
    SceneCode,  # 场景编码类型。
    SeqId,  # 序列 ID 类型。
    StageResult,  # 阶段结果类型。
    StateEstimate,  # 状态估计类型。
    summarize_types,  # 类型层摘要函数。
)  # 类型导入结束。
from liquidloc.common.validation import (  # 导出基础校验函数。第 12 轮审查 LOW-1 修复（R12-E L1，继承 R11-E LOW-1）：删除 L56 重复的 dedupe_preserve_order 单行导入，仅保留多行块内 L71 的导入。
    coerce_finite_scalar,  # 有限浮点转换。
    is_bool_like,  # 布尔类型检查（含 numpy.bool_）。
    is_integer,  # 整数类型检查（排除 bool）。
    is_numeric,  # 数值类型检查（排除 bool）。
    is_real,  # 实数类型检查（排除 bool，基于 numbers.Real）。
    is_string_like,  # 字符串类型检查（含 numpy.str_）。
    normalize_optional_string,  # 可选字符串规整。
    quality_below_floor,  # 质量门槛比较（含浮点容差）。
    require_in_range,  # 范围校验。
    require_iterable,  # 可迭代校验。
    require_keys,  # 键存在校验。
    require_not_none,  # 非空校验。
    require_shape,  # 形状校验。
    dedupe_preserve_order,  # 顺序保留去重。
    validate_path_component,  # 路径组件校验。
)  # 校验导入结束。
from liquidloc.common.config_utils import (  # 导出配置读取与合并工具。
    collect_tbd_paths,  # 递归收集 TBD 占位路径。
    find_project_root,  # 基于 marker 文件查找项目根目录。
    load_dataset_config,  # 读取数据集配置并校验必需键。
    load_yaml_config,  # 读取 YAML 配置文件。
    merge_configs,  # 递归合并多个配置字典。
)  # 配置工具导入结束。
from liquidloc.common.io_utils import (  # 导出严格 JSON IO 工具。
    dumps_json_text,  # 序列化为严格标准 JSON 文本。
    loads_json_text,  # 解析严格标准 JSON 文本。
    read_json,  # 从磁盘读取 JSON 文件。
    read_json_records,  # 读取必须存在的 JSON 记录文件。
    read_optional_json_object,  # 读取可选 JSON 对象文件。
    write_json,  # 将对象写入 JSON 文件。
)  # JSON IO 工具导入结束。
from liquidloc.common.covariance_utils import build_effective_cov  # 导出协方差缩放工具。
from liquidloc.common.gt_utils import align_ground_truth, normalize_gt_rows, resolve_anchor_position, GT_TIME_TOLERANCE  # 导出真值归一化、对齐与锚点解析工具。
from liquidloc.common.statistics_utils import compute_trace_correlation  # 导出轨迹相关系数计算工具。

__all__ = (  # 明确 common 子包对外允许导出的名字。
    "ALLOWED_MODALITIES",  # 允许模态集合。
    "align_ground_truth",  # 真值时间对齐函数。
    "angle_delta_rad",  # 角差计算函数。
    "build_effective_cov",  # 协方差缩放工具。
    "build_output_path",  # 输出路径构建函数。
    "coerce_finite_scalar",  # 有限浮点转换函数。
    "collect_tbd_paths",  # TBD 占位路径收集函数。
    "compute_trace_correlation",  # 轨迹相关系数计算函数。
    "dumps_json_text",  # JSON 序列化函数。
    "DEFAULT_OUTPUT_DIRS",  # 默认输出目录。
    "DEFAULT_REQUIRED_OUTPUT_FILES",  # 默认必需输出文件。
    "DEFAULT_THRESHOLDS",  # 默认阈值。
    "find_project_root",  # 项目根目录查找函数。
    "get_project_root",  # 项目根目录函数。
    "get_standard_dirs",  # 标准目录获取函数。
    "GT_TIME_TOLERANCE",  # 真值时间容差常量。
    "is_bool_like",  # 布尔类型检查函数。
    "is_integer",  # 整数类型检查函数。
    "is_numeric",  # 数值类型检查函数。
    "is_real",  # 实数类型检查函数。
    "is_string_like",  # 字符串类型检查函数。
    "normalize_optional_string",  # 可选字符串规整函数。
    "load_dataset_config",  # 数据集配置加载函数。
    "load_yaml_config",  # YAML 配置加载函数。
    "loads_json_text",  # JSON 解析函数。
    "MeasurementControl",  # 观测控制类型。
    "merge_configs",  # 配置合并函数。
    "MetricDirection",  # 指标方向类型。
    "MetricRow",  # 指标行类型。
    "MODALITY_IMU",  # IMU 模态名。
    "MODALITY_UWB",  # UWB 模态名。
    "MODALITY_VIO",  # VIO 模态名。
    "MODALITY_FLOW",  # 光流/流量类模态名。
    "MODALITY_TOF",  # ToF 测距类模态名。
    "ModelIntermediate",  # 模型中间态类型。
    "normalize_gt_rows",  # 真值行归一化函数。
    "PAYLOAD_KEYS",  # 载荷键注册表。
    "PathLike",  # 路径类类型别名。
    "PredictionBundle",  # 预测包类型。
    "PRIMARY_EVENT_KEYS",  # 主事件键。
    "PRIMARY_KEYS",  # 主键注册表。
    "quality_below_floor",  # 质量门槛比较函数。
    "require_in_range",  # 范围校验函数。
    "require_iterable",  # 可迭代校验函数。
    "require_keys",  # 键校验函数。
    "require_not_none",  # 非空校验函数。
    "require_shape",  # 形状校验函数。
    "read_json",  # JSON 文件读取函数。
    "resolve_anchor_position",  # 锚点位置解析函数。
    "resolve_output_root",  # 输出根目录解析函数。
    "SceneCode",  # 场景编码类型。
    "SeqId",  # 序列 ID 类型。
    "StageResult",  # 阶段结果类型。
    "StateEstimate",  # 状态估计类型。
    "summarize_types",  # 类型层摘要函数。
    "dedupe_preserve_order",  # 顺序保留去重函数。
    "validate_path_component",  # 路径组件校验函数。
    "wrap_angle_rad",  # 角度包裹函数。
    "write_json",  # JSON 文件写入函数。
)  # 导出列表结束。
