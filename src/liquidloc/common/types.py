"""轻量共享类型定义。

职责：
    放跨模块复用的别名和小型数据容器，避免上层到处重复声明同一类
    状态结构，也让合同检查更容易统一口径。

上游依赖：
    - Python 标准库 dataclasses / pathlib / typing  — 类型系统基础

下游调用者：
    - liquidloc.common.__init__      — 统一导出类型给上层
    - liquidloc.protocol.*           — 协议层用类型做字段约束与合同检查
    - liquidloc.estimators.*         — 估计器层用 ModelIntermediate / StateEstimate 传递中间结果
    - liquidloc.pipelines.*          — 流水线层用 PredictionBundle / MetricRow / StageResult 组织产物
    - liquidloc.analysis.*           — 分析层用 MetricRow 做指标汇总

核心变量：
    - SceneCode                     — 场景编码类型别名（str）
    - SeqId                         — 序列 ID 类型别名（str）
    - PathLike                      — 路径类型别名（str | Path）
    - MetricDirection               — 指标优劣方向枚举（Literal）
    - ModelIntermediate             — 模型前端输出的中间态容器
    - MeasurementControl            — 单个测量通道的控制结果
    - StateEstimate                 — 时刻状态估计摘要
    - MetricRow                     — 单行指标记录
    - StageResult                   — 阶段性产物摘要
    - PredictionBundle              — 单序列预测包
"""

from __future__ import annotations  # 允许本文件内的类型注解更灵活。

from dataclasses import asdict, dataclass, field  # 用 dataclass 定义轻量数据容器。
from pathlib import Path  # 路径类型别名会用到 Path。
from typing import Any, Literal, TypeAlias, get_args  # 类型别名和标注工具。

from liquidloc.common.constants import ALLOWED_MODALITIES  # 模态校验用。
from liquidloc.common.constants import BRIDGE_RISK_MIN, BRIDGE_RISK_MAX, BRIDGE_SCALING_MAX  # 桥接层阈值常量（单源真相）。
from liquidloc.common.constants import BRIDGE_BIAS_MAX, BRIDGE_BIAS_ABSOLUTE_MAX, BRIDGE_NOISE_MULTIPLIER_CEILING  # 桥接层偏置/噪声阈值常量。
from liquidloc.common.constants import BRIDGE_SCALING_MIN  # 缩放下界单源真相，与 bridge_thresholds 同源。
from liquidloc.common.constants import STATE_ITEMS  # 状态项冻结集合（单源真相）。
from liquidloc.common.validation import coerce_finite_scalar  # 有限数值校验与转换的集中入口。

_ALLOWED_MODALITIES_SET = frozenset(ALLOWED_MODALITIES)  # 运行时校验用的模态集合。

SceneCode: TypeAlias = str  # 场景编码本质上是字符串。
SeqId: TypeAlias = str  # 序列 ID 也按字符串处理。
PathLike: TypeAlias = str | Path  # 路径既可以是字符串也可以是 Path。
MetricDirection: TypeAlias = Literal["lower_is_better", "higher_is_better", "neutral"]  # 指标优劣方向的有限枚举。
_METRIC_DIRECTIONS = frozenset(get_args(MetricDirection))  # 从 Literal 自动派生，避免手动同步漂移。
_VALID_GATE_ACTIONS = frozenset({
    'pass_through', 'uwb_skip_update', 'uwb_bias_and_noise_scale',
    'vio_skip_update', 'vio_confidence_scale',
})  # 合法门控动作集合。


@dataclass(slots=True)
class ModelIntermediate:  # 模型前端输出的中间态容器。
    """模型中间输出。

    记录前端模型对 bias、risk 和 scaling 的中间估计，供下游控制层使用。

    Attributes:
        bias (float): 偏置估计值，默认 0.0，范围 [0.0, bias_max]（非负约束，与全链路裁剪对齐）。
        risk (float): 风险评分，默认 0.0，范围 [0.0, 1.0]。
        uwb_scaling (float): UWB 量测缩放系数，默认 1.0（不缩放），范围 [1.0, scaling_max]。
        vio_scaling (float): VIO 量测缩放系数，默认 1.0（不缩放），范围 [1.0, scaling_max]。
    """
    bias: float = 0.0  # 偏置默认从 0 开始。
    risk: float = 0.0  # 风险默认从 0 开始。
    uwb_scaling: float = 1.0  # UWB 缩放默认不改变量测。
    vio_scaling: float = 1.0  # VIO 缩放默认不改变量测。

    def __post_init__(self) -> None:  # 实例化后做范围校验。
        """校验字段值在合理范围内，并将 numpy 标量强制转为 Python float。

        与 MeasurementControl 的校验标准对齐：
        - bias: 有限数值，范围 [0.0, bias_max]（非负约束，与全链路裁剪对齐）
        - risk: 有限数值，范围 [risk_min, risk_max]（取自 BRIDGE_THRESHOLDS）
        - uwb_scaling / vio_scaling: 有限数值，范围 [1.0, scaling_max]
        """
        # coerce_finite_scalar 统一完成数值类型校验、有限性校验与 float 转换；
        # 边界校验保留原消息格式，避免对消费者造成口径漂移。
        self.bias = coerce_finite_scalar(self.bias, name="bias")
        _bias_max = BRIDGE_BIAS_MAX
        if not (0.0 <= self.bias <= _bias_max):
            raise ValueError(f"bias must be in [0.0, {_bias_max}], got {self.bias}")

        self.risk = coerce_finite_scalar(self.risk, name="risk")
        _risk_min = BRIDGE_RISK_MIN
        _risk_max = BRIDGE_RISK_MAX
        if not (_risk_min <= self.risk <= _risk_max):
            raise ValueError(f"risk must be in [{_risk_min}, {_risk_max}], got {self.risk}")

        self.uwb_scaling = coerce_finite_scalar(self.uwb_scaling, name="uwb_scaling")
        _scaling_min = BRIDGE_SCALING_MIN
        _scaling_max = BRIDGE_SCALING_MAX
        if self.uwb_scaling < _scaling_min or self.uwb_scaling > _scaling_max:
            raise ValueError(f"uwb_scaling must be in [{_scaling_min}, {_scaling_max}], got {self.uwb_scaling}")

        self.vio_scaling = coerce_finite_scalar(self.vio_scaling, name="vio_scaling")
        if self.vio_scaling < _scaling_min or self.vio_scaling > _scaling_max:
            raise ValueError(f"vio_scaling must be in [{_scaling_min}, {_scaling_max}], got {self.vio_scaling}")


@dataclass(slots=True)
class MeasurementControl:  # 单个测量通道的控制结果。
    """单个测量通道的控制结果。

    这个结构描述某个模态经过门控、偏置修正和缩放后，最终如何进入后续
    估计器或融合器。
    """
    modality: str  # 当前控制对象所属模态。
    bias_applied: float = 0.0  # 实际施加的偏置。
    scaling: float = 1.0  # 实际使用的缩放系数。
    risk: float = 0.0  # 风险评分。
    noise_multiplier: float = 1.0  # 噪声放大倍数。
    gate_action: str = 'pass_through'  # 门控动作默认是直接通过。

    def __post_init__(self) -> None:  # 实例化后做范围校验。
        """校验字段值在合理范围内，并将 numpy 标量强制转为 Python float。"""
        if self.modality not in _ALLOWED_MODALITIES_SET:
            raise ValueError(f"modality must be one of {sorted(_ALLOWED_MODALITIES_SET)}, got {self.modality!r}")
        if self.gate_action not in _VALID_GATE_ACTIONS:
            raise ValueError(f"gate_action must be one of {sorted(_VALID_GATE_ACTIONS)}, got {self.gate_action!r}")
        # coerce_finite_scalar 统一完成数值类型校验、有限性校验与 float 转换；
        # 边界校验保留原消息格式与原始顺序，避免对消费者造成口径漂移。
        self.bias_applied = coerce_finite_scalar(self.bias_applied, name="bias_applied")
        _bias_abs_max = BRIDGE_BIAS_ABSOLUTE_MAX
        if not (0.0 <= self.bias_applied <= _bias_abs_max):
            raise ValueError(f"bias_applied must be in [0.0, {_bias_abs_max}], got {self.bias_applied}")
        self.scaling = coerce_finite_scalar(self.scaling, name="scaling")
        _scaling_min = BRIDGE_SCALING_MIN
        _scaling_max = BRIDGE_SCALING_MAX
        if self.scaling < _scaling_min or self.scaling > _scaling_max:
            raise ValueError(f"scaling must be in [{_scaling_min}, {_scaling_max}], got {self.scaling}")
        self.risk = coerce_finite_scalar(self.risk, name="risk")
        _risk_min = BRIDGE_RISK_MIN
        _risk_max = BRIDGE_RISK_MAX
        if not (_risk_min <= self.risk <= _risk_max):
            raise ValueError(f"risk must be in [{_risk_min}, {_risk_max}], got {self.risk}")
        self.noise_multiplier = coerce_finite_scalar(self.noise_multiplier, name="noise_multiplier")
        _nm_floor = _scaling_min * _scaling_min  # noise_multiplier = scaling^2 * (1+risk) >= scaling_min^2
        if self.noise_multiplier < _nm_floor:
            raise ValueError(f"noise_multiplier must be >= {_nm_floor}, got {self.noise_multiplier}")
        _nm_ceiling = BRIDGE_NOISE_MULTIPLIER_CEILING
        if self.noise_multiplier > _nm_ceiling:
            raise ValueError(f"noise_multiplier must be <= {_nm_ceiling}, got {self.noise_multiplier}")


@dataclass(slots=True)
class StateEstimate:  # 时刻状态估计摘要。
    """时刻状态估计和协方差摘要。"""
    state: dict[str, float] = field(default_factory=dict)  # 状态向量按字段名保存。
    covariance_diag: list[float] = field(default_factory=list)  # 协方差对角线摘要。
    timestamp: float | None = None  # 对应时间戳。

    def __post_init__(self) -> None:  # 校验字段并将 numpy 标量强制转为 Python 原生类型。
        """校验字段合法性，并将 numpy 标量强制转为 Python float，防止序列化失败。

        检查 timestamp 为有限数值（若非 None）、covariance_diag 各元素为非负有限数值、
        state 各值为有限数值、state 键名属于 STATE_ITEMS 冻结集合、
        covariance_diag 长度与 state 维度一致。
        """
        _allowed_state_keys = frozenset(STATE_ITEMS)
        # coerce_finite_scalar 统一完成数值类型校验、有限性校验与 float 转换；
        # 边界校验（covariance_diag 非负）保留原消息格式。
        _validated_state: dict[str, float] = {}
        for k, v in self.state.items():
            if k not in _allowed_state_keys:
                raise ValueError(f"state key {k!r} is not in STATE_ITEMS {STATE_ITEMS}")
            _validated_state[k] = coerce_finite_scalar(v, name=f"state[{k!r}]")
        self.state = _validated_state
        _validated_cov: list[float] = []
        for i, v in enumerate(self.covariance_diag):
            _scalar = coerce_finite_scalar(v, name=f"covariance_diag[{i}]")
            if _scalar < 0.0:
                raise ValueError(f"covariance_diag[{i}] must be non-negative, got {_scalar}")
            _validated_cov.append(_scalar)
        self.covariance_diag = _validated_cov
        if self.covariance_diag and self.state and len(self.covariance_diag) != len(self.state):
            raise ValueError(
                f"covariance_diag length ({len(self.covariance_diag)}) must match "
                f"state dimension ({len(self.state)}), got {len(self.covariance_diag)} vs {len(self.state)}"
            )
        if self.timestamp is not None:
            self.timestamp = coerce_finite_scalar(self.timestamp, name="timestamp")


@dataclass(slots=True)
class MetricRow:  # 单行指标记录。
    """单行指标记录。

    用于统一表达指标名称、数值、单位、优劣方向和所属分组。
    """
    metric: str  # 指标名。
    value: float  # 指标值。
    unit: str  # 单位。
    direction: MetricDirection  # 指标优劣方向。
    group: str  # 所属分组。

    def __post_init__(self) -> None:  # 实例化后做完整校验。
        """校验指标记录的完整性和合法性。

        检查 metric/unit/group 为非空字符串、value 为有限数值、direction 属于已知集合。
        与 ModelIntermediate/MeasurementControl 的严格校验标准对齐。

        Raises:
            ValueError: 当 direction 不合法、value 非有限、或字符串字段为空时抛出。
            TypeError: 当 value 为布尔类型时抛出。
        """
        if self.direction not in _METRIC_DIRECTIONS:  # 方向必须是预定义集合之一。
            raise ValueError(f"direction must be one of {sorted(_METRIC_DIRECTIONS)}, got {self.direction!r}")  # 不合法就报错。
        # coerce_finite_scalar 统一完成数值类型校验、有限性校验与 float 转换。
        self.value = coerce_finite_scalar(self.value, name="MetricRow.value")
        if not self.metric or not isinstance(self.metric, str):  # 指标名必须是非空字符串。
            raise ValueError(f"MetricRow.metric must be a non-empty string, got {self.metric!r}")
        if not self.unit or not isinstance(self.unit, str):  # 单位必须是非空字符串。
            raise ValueError(f"MetricRow.unit must be a non-empty string, got {self.unit!r}")
        if not self.group or not isinstance(self.group, str):  # 分组必须是非空字符串。
            raise ValueError(f"MetricRow.group must be a non-empty string, got {self.group!r}")


@dataclass(slots=True)
class StageResult:  # 阶段性产物摘要。
    """阶段性产物摘要。

    Attributes:
        stage_name (str): 阶段名称，如 "prepare_pipeline"、"train_pipeline"、"eval_pipeline"、"core_pipeline"。
        artifacts (list[str]): 产物路径列表，默认为空列表。
        metadata (dict[str, Any]): 附加元数据字典，默认为空字典。
    """
    stage_name: str  # 阶段名。
    artifacts: list[str] = field(default_factory=list)  # 产物列表。
    metadata: dict[str, Any] = field(default_factory=dict)  # 附加元数据。

    def __post_init__(self) -> None:  # 校验阶段名非空。
        """校验 stage_name 为非空字符串（拒绝纯空白字符串）。"""
        if not isinstance(self.stage_name, str) or not self.stage_name.strip():
            raise ValueError(f"StageResult.stage_name must be a non-empty string, got {self.stage_name!r}")


@dataclass(slots=True)
class PredictionBundle:  # 单序列预测包。
    """单序列预测包。

    保存序列 ID、场景 ID、状态序列和对应时间戳，方便后续 metrics /
    plotting / audit 直接读取。

    Attributes:
        seq_id (SeqId): 序列 ID，本质为字符串。
        scene_id (SceneCode): 场景 ID，本质为字符串。
        states (list[dict[str, float]]): 状态序列，每个元素是一个时刻的状态字典，
            默认为空列表。
        timestamps (list[float]): 时间戳序列，与 states 一一对应，默认为空列表。
    """
    seq_id: SeqId  # 序列 ID。
    scene_id: SceneCode  # 场景 ID。
    states: list[dict[str, float]] = field(default_factory=list)  # 状态序列。
    timestamps: list[float] = field(default_factory=list)  # 时间戳序列。

    def __post_init__(self) -> None:  # 基本校验。
        """校验序列/场景 ID 非空，状态和时间戳长度一致，并将 numpy 标量强制转为 Python 原生类型。

        与 StateEstimate 的校验标准对齐：每个状态值和时间戳都必须是有限数值，
        排除 bool、NaN 和 Inf。
        """
        if not isinstance(self.seq_id, str) or not self.seq_id.strip():  # seq_id 必须是非空字符串（拒绝纯空白）。
            raise ValueError(f"PredictionBundle.seq_id must be a non-empty string, got {self.seq_id!r}")
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():  # scene_id 必须是非空字符串（拒绝纯空白）。
            raise ValueError(f"PredictionBundle.scene_id must be a non-empty string, got {self.scene_id!r}")
        if len(self.states) != len(self.timestamps):  # 长度必须一致（含双方都为空的情况）。
            raise ValueError(
                f"PredictionBundle states and timestamps must have equal length, "
                f"got {len(self.states)} vs {len(self.timestamps)}"
            )
        # 逐个校验状态值，与 StateEstimate 对齐（含键名白名单）。
        # coerce_finite_scalar 统一完成数值类型校验、有限性校验与 float 转换。
        _allowed_state_keys = frozenset(STATE_ITEMS)
        _validated_states: list[dict[str, float]] = []
        for si, state in enumerate(self.states):
            _validated_state: dict[str, float] = {}
            for k, v in state.items():
                if k not in _allowed_state_keys:
                    raise ValueError(f"states[{si}] key {k!r} is not in STATE_ITEMS {STATE_ITEMS}")
                _validated_state[k] = coerce_finite_scalar(v, name=f"states[{si}][{k!r}]")
            _validated_states.append(_validated_state)
        self.states = _validated_states
        # 逐个校验时间戳。
        self.timestamps = [
            coerce_finite_scalar(t, name=f"timestamps[{ti}]")
            for ti, t in enumerate(self.timestamps)
        ]


def summarize_types() -> dict[str, Any]:  # 返回类型层的稳定摘要。
    """返回类型层的稳定摘要，供轻量测试和审计使用。

    构造一份包含类型别名说明、dataclass 名称列表和示例序列化结果的字典，
    方便下游在不导入完整类型的情况下做合同检查。

    Returns:
        dict[str, Any]: 包含以下键的摘要字典：
            - aliases (dict[str, str]): 类型别名到其底层类型的映射。
            - dataclasses (list[str]): 本模块定义的 dataclass 名称列表。
            - example_metric_row (dict): MetricRow 示例的序列化结果。
            - example_measurement_control (dict): MeasurementControl 示例的序列化结果。
            - example_model_intermediate (dict): ModelIntermediate 示例的序列化结果。
            - example_state_estimate (dict): StateEstimate 示例的序列化结果。
    """
    # 用真实 dataclass 造一个示例，方便检查字段名与默认值是否保持稳定。
    example_metric_row = asdict(  # 把示例 MetricRow 转成普通字典。
        MetricRow(  # 先构造一个标准指标行。
            metric="rmse",  # 示例指标名。
            value=0.0,  # 示例值。
            unit="m",  # 示例单位。
            direction="lower_is_better",  # 示例优劣方向。
            group="primary",  # 示例分组。
        )  # MetricRow 结束。
    )  # asdict 结束。
    # 同理给测量控制准备一个可序列化样例，便于审计输出结构。
    example_measurement_control = asdict(  # 把示例 MeasurementControl 转成普通字典。
        MeasurementControl(modality='uwb', bias_applied=0.1, scaling=1.5, risk=0.4, noise_multiplier=3.15, gate_action='uwb_bias_and_noise_scale')  # 构造一个示例控制记录；noise_multiplier 与 bridge 合同 scaling^2 * (1 + risk) 一致。
    )  # asdict 结束。
    # 模型中间态示例，覆盖非零 bias 和 scaling。
    example_model_intermediate = asdict(
        ModelIntermediate(bias=0.5, risk=0.3, uwb_scaling=2.0, vio_scaling=1.5)
    )
    # 状态估计示例，使用 STATE_ITEMS 冻结集合中的键。
    _example_state = {k: 0.0 for k in STATE_ITEMS}
    _example_state["px"] = 1.0
    _example_state["py"] = 2.0
    example_state_estimate = asdict(
        StateEstimate(state=_example_state, covariance_diag=[0.1] * len(STATE_ITEMS), timestamp=1.0)
    )
    # 自动收集本模块中所有 dataclass 名称，避免手动维护列表导致遗漏。
    import sys as _sys  # 延迟导入，避免污染模块命名空间。
    _current_module = _sys.modules[__name__]
    _dataclass_names = sorted(
        name for name, obj in vars(_current_module).items()
        if isinstance(obj, type) and hasattr(obj, '__dataclass_fields__')
    )
    return {  # 返回一个摘要字典。
        "aliases": {  # 先列出类型别名的解释。
            "SceneCode": "str",  # 场景编码是字符串。
            "SeqId": "str",  # 序列 ID 是字符串。
            "PathLike": "str | pathlib.Path",  # 路径既可以是字符串也可以是 Path。
            "MetricDirection": "Literal[lower_is_better|higher_is_better|neutral]",  # 指标方向是有限枚举。
        },  # aliases 结束。
        "dataclasses": _dataclass_names,  # 自动收集的 dataclass 名称列表。
        "example_metric_row": example_metric_row,  # 示例指标行。
        "example_measurement_control": example_measurement_control,  # 示例测量控制。
        "example_model_intermediate": example_model_intermediate,  # 示例模型中间态。
        "example_state_estimate": example_state_estimate,  # 示例状态估计。
    }  # 摘要字典结束。


__all__ = (
    "SceneCode",
    "SeqId",
    "PathLike",
    "MetricDirection",
    "ModelIntermediate",
    "MeasurementControl",
    "StateEstimate",
    "MetricRow",
    "StageResult",
    "PredictionBundle",
    "summarize_types",
)
