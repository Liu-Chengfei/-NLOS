"""
文件：src/liquidloc/models/features/feature_builder.py

【文件职责】
从事件和中间状态构造学习前端输入特征，字段顺序必须冻结。

【本文件绝对不负责】
不负责训练和模型推理。

【上游依赖】
protocol/event_schema.py、state_definition.py、configs/models/*.yaml。

【下游调用者】
window_builder.py、lstm/liquid inference、trainers。

【输入对象定义】
- event
- state_ctx
- feature_order

【输出对象定义】
- feature_vector

【核心变量定义】
- feature_order
- feature_vector
- state_ctx
- missing_fill_value

【推荐编写顺序】
1. 先在说明里列出字段顺序。
2. 再写 build_feature_vector。
3. 最后写批量构造函数（如需要）。

【建议先写的函数 / 类】
- 构造单步特征
  签名：def build_feature_vector(event, state_ctx, feature_order)
  作用：按冻结字段顺序构造单步特征向量。
  输入：event、state_ctx、feature_order
  输出：feature_vector
  关键局部变量：
    - feature_order
    - feature_vector
    - field_value
    - missing_fill_value
  伪代码：
    1) 遍历 feature_order。
    2) 从 event 或 state_ctx 取字段。
    3) 缺失时按协议填充。
    4) 按顺序拼成 feature_vector。
    5) 返回 feature_vector。

【最容易让 Codex 理解错的地方】
- 字段顺序一漂，训练和推理都会错。
- 不要在不同文件里重复定义 feature_order。

【最小手工测试步骤】
1. 对同一 event 多次提取特征，检查顺序完全一致。

【完成标准】
- 特征顺序冻结，批量构造不再猜字段。

【实现要求】
- 当前阶段保持空白实现，只保留细化编写说明。
- 真正实现时先写签名和 docstring，再写前置检查，再写主体逻辑，最后补测试。
- 任何字段名和变量名优先服从 protocol 和 configs，不能临时发明。

上游校验依赖：
  ``coerce_event_for_feature_extraction`` 是宽松模式校验，不检查字段值范围。
  正常流程中，事件在到达 feature_builder 之前已通过
  ``validate_event_sequence``（严格校验），非法值不会静默通过。
  如果绕过 fusion_runner 直接调用 feature_builder，需自行确保事件合法性。
"""

from __future__ import annotations  # 允许在注解里引用后面可能才出现的类型名。

from collections.abc import Iterable, Mapping  # 用于检查序列型输入和映射型输入。
from dataclasses import asdict, is_dataclass  # 支持把 dataclass 对象转成普通字典。
import math  # 用于有限性检查和几何相关计算。

from liquidloc.common.constants import MODALITY_UWB, MODALITY_VIO, PAYLOAD_KEYS  # 模态常量与模态到载荷键的固定映射，保证字段解释一致。
from liquidloc.common.gt_utils import resolve_anchor_position  # 锚点位置解析的规范实现
from liquidloc.common.validation import is_bool_like, is_numeric, is_string_like  # 统一判断数值类型。
from liquidloc.protocol.event_schema import coerce_event_for_feature_extraction  # 统一事件结构，避免各处自己猜字段。
from liquidloc.protocol.sensor_contract import get_feature_missing_policy  # 统一读取冻结缺失策略合同。
from liquidloc.sensors.anchor_model import compute_geometry_report  # 计算锚点几何质量指标。
from liquidloc.sensors.uwb_model import predict_range_to_anchor  # 预测 UWB 到锚点的几何距离。


_FEATURE_MISSING_POLICY = get_feature_missing_policy()  # 缺失填充值和来源优先级都从冻结合同读取。
_MISSING_SOURCE_PRECEDENCE = _FEATURE_MISSING_POLICY["source_precedence"]
_MISSING_FILL_VALUE = float(_FEATURE_MISSING_POLICY["numeric_fill_value"])
if not _FEATURE_MISSING_POLICY["require_missing_mask"]:
    raise ValueError("feature_missing_policy.require_missing_mask must stay true for feature extraction")
if _FEATURE_MISSING_POLICY["missing_semantics_carrier"] != "missing_mask":
    raise ValueError("feature_missing_policy.missing_semantics_carrier must stay 'missing_mask'")


# 这个辅助函数负责把各种“像映射”的对象统一成普通字典，便于后续按键读取。
def _coerce_mapping_like(payload, *, name: str) -> dict:
    """把输入统一转成字典，方便后续按字段读取。

    参数:
    `payload` 是待转换对象，可以是映射、dataclass、带 `to_dict` 的对象，或
    带 `__dict__` 的普通对象。
    `name` 是错误提示里使用的字段名，方便快速定位是谁传错了。

    返回值:
    返回一个普通 `dict`，供后续特征解析统一使用。

    关键局部变量:
    `payload_to_dict` 用来承接对象自带的 `to_dict` 方法。
    """
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    payload_to_dict = getattr(payload, "to_dict", None)
    if callable(payload_to_dict):
        # to_dict() 是动态调用，返回值必须校验为映射后再转 dict，
        # 否则非字典返回值（如 None/list/str）会静默破坏 -> dict 契约，
        # 导致下游 event_payload["modality"] / .get() 等访问出错或行为异常。
        coerced = payload_to_dict()
        if not isinstance(coerced, Mapping):
            raise TypeError(
                f"{name}.to_dict() must return a mapping, got {type(coerced).__name__}"
            )
        return dict(coerced)
    if is_dataclass(payload) and not isinstance(payload, type):
        return asdict(payload)
    if hasattr(payload, "__dict__") and not isinstance(payload, type):
        # vars() 返回的是对象内部 __dict__ 的直接引用，必须复制一份，
        # 否则下游对返回字典的写操作会回写原对象，造成状态污染。
        return dict(vars(payload))
    raise TypeError(f"{name} must be a mapping, dataclass, or object with __dict__")


def _safe_float(value) -> float | None:
    """安全转换为有限浮点数，非数值或非有限值返回 None。

    与 _resolve_feature_value 直接来源路径的 try/except + isfinite 检查对齐，
    确保派生特征解析不会因单个非数值或非有限字段崩溃整个特征向量构造。
    捕获口径与 L177 直接值路径一致：TypeError/ValueError/OverflowError。
    """
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    return result


# 这个函数负责先从事件、载荷或状态里找字段值，找不到再尝试派生值。
def _resolve_feature_value(event_payload: dict, state_payload: dict, field_name: str) -> tuple[float, int]:
    """按固定优先级解析单个特征值，并返回缺失标记。

    参数:
    `event_payload` 是已经标准化后的事件字典。
    `state_payload` 是该时间步对应的状态上下文字典。
    `field_name` 是当前正在解析的特征名。

    返回值:
    返回 `(field_value, missing_flag)`，其中 `field_value` 是数值特征，
    `missing_flag` 为 0 表示已找到，1 表示最终没找到。

    关键局部变量:
    `payload_key` 指向当前模态对应的载荷键。
    `payload` 是当前模态下的具体载荷内容。
    `derived_value` 是从上下文和几何关系推导出来的备选值。
    """
    payload_key = PAYLOAD_KEYS[event_payload["modality"]]
    payload = event_payload.get(payload_key)
    value_sources = {
        "event_root": event_payload,
        "modality_payload": payload if isinstance(payload, Mapping) else {},
        "state_ctx": state_payload,
    }
    for source_name in _MISSING_SOURCE_PRECEDENCE:
        source_payload = value_sources[source_name]
        if field_name not in source_payload or source_payload[field_name] is None:
            continue
        raw = source_payload[field_name]
        # 布尔载荷字段（如 UWB valid）需转为 0.0/1.0 数值特征，
        # 否则 is_numeric 会排除布尔值，导致该字段永远被标记为缺失，
        # 违反 sensors.yaml 中 valid 字段"0/1"语义和 missing_mask 协议。
        if is_bool_like(raw):
            return float(bool(raw)), 0
        if not is_numeric(raw):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            continue  # OverflowError 对应超大 int（如 10**1000），与 coerce_finite_scalar 的捕获口径对齐。
        if not math.isfinite(value):
            continue  # 当前来源值无效时继续按 source_precedence 回退，而不是提前判缺失。
        return value, 0

    # 派生值同样需要有限性检查：_resolve_derived_feature_value 多数返回路径
    # （modality_gap_dt/uwb_quality_min/anchor_dx/anchor_dy/geom_score 等）未做
    # math.isfinite 校验，非有限派生值（inf/nan）若直接返回会污染特征向量。
    # OverflowError 来自派生函数内部的 float() 转换（如 state_payload 含超大 int），
    # 与上方直接值处理口径一致，视为无效并回退到缺失填充。
    try:
        derived_value = _resolve_derived_feature_value(event_payload, state_payload, field_name)
    except OverflowError:
        derived_value = None
    if derived_value is not None and math.isfinite(derived_value):
        return derived_value, 0

    return _MISSING_FILL_VALUE, 1


# 这个函数只负责派生特征，避免主解析函数变得过长。
def _resolve_derived_feature_value(event_payload: dict, state_payload: dict, field_name: str) -> float | None:
    """根据模态和状态上下文推导那些不能直接取到的特征。

    参数:
    `event_payload` 是标准化事件字典。
    `state_payload` 是标准化状态字典。
    `field_name` 是要推导的特征名。

    返回值:
    成功时返回浮点数，失败时返回 `None`，表示这个派生值暂时拿不到。

    关键局部变量:
    `payload_key` 是当前模态对应的载荷字段名。
    `payload` 是当前模态载荷内容，缺失时用空字典兜底。
    `state_value` 用来承接状态里已有的候选值。
    """
    payload_key = PAYLOAD_KEYS[event_payload["modality"]]
    payload = event_payload.get(payload_key) if isinstance(event_payload.get(payload_key), Mapping) else {}

    if field_name in {"anchor_dx", "anchor_dy", "uwb_range_residual"}:
        return _resolve_uwb_geometry_feature(event_payload, state_payload, payload, field_name)

    if field_name == "geom_score":
        return _resolve_global_geometry_feature(state_payload)

    if field_name == "modality_gap_dt":
        state_value = state_payload.get("modality_gap_dt")
        if state_value is None:
            state_value = state_payload.get("time_since_last_modality")
        if state_value is None:
            return None  # 不应回退到 dt，语义不同：dt 是相邻事件间隔，modality_gap_dt 是同模态间隔。
        return _safe_float(state_value)

    if field_name == "uwb_quality_min":
        state_value = state_payload.get("uwb_quality_min")
        if state_value is None and event_payload["modality"] == MODALITY_UWB:
            state_value = payload.get("quality")
        if state_value is None:
            return None
        return _safe_float(state_value)

    if field_name == "uwb_invalid_rate":
        state_value = state_payload.get("uwb_invalid_rate")
        if state_value is None:
            if event_payload["modality"] == MODALITY_UWB:
                valid = payload.get("valid")
                if is_bool_like(valid):
                    state_value = 0.0 if valid else 1.0
            # 窗口级计数统计是模态无关的窗口指标，UWB 模态在 valid 缺失时
            # 也应回退到计数统计，不应因 elif 互斥而跳过。
            if state_value is None and state_payload.get("uwb_invalid_count") is not None and state_payload.get("uwb_sample_count"):
                invalid_count = _safe_float(state_payload["uwb_invalid_count"])
                sample_count = _safe_float(state_payload["uwb_sample_count"])
                if invalid_count is not None and sample_count is not None and sample_count > 0.0:
                    state_value = invalid_count / sample_count
        if state_value is None:
            return None
        return _safe_float(state_value)

    # 铁律 3 (Stage A1 下游修复, 2026-07-23): 删除 vio_reproj_err_slope / tracked_features_drop
    # 计算分支 — VIO 紧耦合不再输出 reproj_err / tracked_features 字段, 这两个特征
    # 无法再计算. model_factory.py 已从 LIQUID_CONTEXT_FEATURE_KEYS /
    # LIQUID_VIO_FAST_CONTEXT_FEATURE_KEYS 中删除这两个键, 因此这里不会进入.
    # 保留 fallback 分支防御, 但不再依赖已删除的字段.

    return None


# 这个函数负责只在 UWB 模态下推导几何相关特征。
def _resolve_uwb_geometry_feature(
    event_payload: dict,
    state_payload: dict,
    payload: Mapping,
    field_name: str,
) -> float | None:
    """推导 UWB 模态下的锚点相对位移和距离残差。

    参数:
    `event_payload` 是当前事件字典。
    `state_payload` 是当前状态字典。
    `payload` 是当前 UWB 载荷内容。
    `field_name` 指明当前要返回的是位移分量还是距离残差。

    返回值:
    返回 `anchor_dx`、`anchor_dy` 或 `uwb_range_residual` 对应的浮点值；
    如果缺少必要上下文则返回 `None`。

    关键局部变量:
    `anchor_lookup` 是锚点查找表。
    `anchor_position` 是锚点坐标。
    `px`、`py` 是当前状态下的位置信息。
    `anchor_dx`、`anchor_dy` 是锚点相对当前位置的坐标差。
    `measured_range` 和 `predicted_range` 用于计算残差。
    `residual` 是最终的距离误差。
    """
    if event_payload["modality"] != MODALITY_UWB:
        return None

    anchor_lookup = state_payload.get("anchor_lookup")
    if not isinstance(anchor_lookup, Mapping):
        return None

    anchor_id = payload.get("anchor_id")
    if anchor_id is None:
        return None
    try:
        anchor_position = resolve_anchor_position(anchor_id, anchor_lookup)
    except (TypeError, ValueError):
        return None
    if anchor_position is None:
        return None

    px = state_payload.get("px")
    py = state_payload.get("py")
    if px is None or py is None:
        return None
    try:
        px = float(px)
        py = float(py)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(px) and math.isfinite(py)):
        return None
    anchor_dx = float(anchor_position[0]) - px
    anchor_dy = float(anchor_position[1]) - py
    if not (math.isfinite(anchor_dx) and math.isfinite(anchor_dy)):
        return None

    if field_name == "anchor_dx":
        return anchor_dx
    if field_name == "anchor_dy":
        return anchor_dy

    measured_range = payload.get("range")
    if measured_range is None:
        return None
    try:
        predicted_range = predict_range_to_anchor({"px": px, "py": py}, anchor_position)
    except (TypeError, ValueError):
        return None
    try:
        residual = float(measured_range) - predicted_range
    except (TypeError, ValueError):
        return None
    if not math.isfinite(residual):
        return None
    return residual


# 这个函数负责汇总全局锚点几何质量。
def _resolve_global_geometry_feature(state_payload: dict) -> float | None:
    """从当前锚点布局里计算全局几何分数。

    参数:
    `state_payload` 是包含锚点布局和布局 ID 的状态字典。

    返回值:
    成功时返回几何分数，失败时返回 `None`。

    关键局部变量:
    `anchor_lookup` 是锚点查找表。
    `anchor_ids` 保存用于几何计算的锚点标识。
    `anchor_positions` 保存对应的二维坐标。
    `geometry_report` 是几何计算函数返回的报告。
    `anchor_count` 是锚点数量校验结果。
    """
    anchor_lookup = state_payload.get("anchor_lookup")
    if not isinstance(anchor_lookup, Mapping) or not anchor_lookup:
        return None

    anchor_ids = []
    anchor_positions = []
    for anchor_id, anchor_position in anchor_lookup.items():
        if isinstance(anchor_position, (str, bytes)):
            continue  # 跳过非坐标条目（如 layout_id、reference 等元数据键）。
        try:
            coords = list(anchor_position)
        except TypeError:
            continue  # 跳过不可迭代的条目。
        if len(coords) != 2:
            continue  # 跳过非二维坐标。
        # 与 _normalize_anchor_layout L195 / project_anchor_layout_xy L364 口径对齐：
        # 显式拒绝布尔坐标，避免 True/False 静默当成 1.0/0.0 进入几何评分。
        if is_bool_like(coords[0]) or is_bool_like(coords[1]):
            continue
        try:
            x_coord = float(coords[0])
            y_coord = float(coords[1])
        except (TypeError, ValueError, OverflowError):
            continue  # 跳过非数值坐标，OverflowError 与 _normalize_anchor_layout L200 口径对齐。
        if not math.isfinite(x_coord) or not math.isfinite(y_coord):
            continue  # 跳过 NaN/Inf 坐标，与 _normalize_anchor_layout L202 / project_anchor_layout_xy L371 口径对齐。
        anchor_positions.append((x_coord, y_coord))
        anchor_ids.append(anchor_id)

    try:
        geometry_report = compute_geometry_report(
            {
                "anchor_ids": anchor_ids,
                "anchor_positions": anchor_positions,
                "layout_id": state_payload.get("layout_id"),
            }
        )
        anchor_count = int(geometry_report["anchor_count"])
        if anchor_count < 3:
            return None
        return float(geometry_report["geom_score"])
    except (TypeError, ValueError, KeyError):
        return None


# 这个函数把历史事件和历史状态逐步拼成特征状态历史。
def build_feature_state_history(history, state_history, *, anchor_lookup: Mapping | None = None) -> list[dict]:
    """构造每个时间步对应的特征状态快照。

    参数:
    `history` 是事件序列。
    `state_history` 是与事件一一对应的状态序列。
    `anchor_lookup` 是可选的锚点查找表，如果给了就会复制进每一帧状态。

    返回值:
    返回一个列表，列表中的每个元素都是某个时间步的特征状态字典。

    关键局部变量:
    `feature_state_history` 收集每一步的状态快照。
    `last_timestamp_by_modality` 记录各模态上一次出现的时间。
    `uwb_quality_min` 记录迄今为止最差的 UWB 质量。
    `uwb_invalid_count` 和 `uwb_sample_count` 用来计算无效率。
    `feature_state` 是当前时间步复制出来的状态字典。
    `modality_gap_dt` 是当前模态距离上次出现的时间差。
    """
    # 偷懒审视 Round 4 真修 (audit #17): 删 last_vio_reproj_err / last_tracked_features
    # 相关 docstring — VIO 紧耦合不再输出 reproj_err / tracked_features, 计算分支已删, 上行注释也清.
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "history_length": len(history) if hasattr(history, "__len__") else None,
        "state_history_length": len(state_history) if hasattr(state_history, "__len__") else None,
        "has_anchor_lookup": anchor_lookup is not None,
    }, "build_feature_state_history")
    history = list(history or [])
    state_history = list(state_history or [])
    if len(state_history) != len(history):
        raise ValueError("state_history must align with history length")

    normalized_anchor_lookup = dict(anchor_lookup or {})
    feature_state_history: list[dict] = []
    last_timestamp_by_modality: dict[str, float] = {}
    uwb_quality_min: float | None = None
    uwb_invalid_count = 0
    uwb_sample_count = 0
    # 偷懒审视 Round 4 真修 (audit #17): 删 last_vio_reproj_err / last_tracked_features
    # 及其 slope/drop 跟踪变量 — VIO 紧耦合不再输出 reproj_err / tracked_features 字段
    # (见 model_factory.py LIQUID_CONTEXT_FEATURE_KEYS 已删 vio_reproj_err_slope /
    # tracked_features_drop), 上游再算下游也被 model_factory 过滤掉, 真死代码.

    # 这里逐步扫描事件序列，并把状态派生量填进每一步的 feature_state。
    for event, state_ctx in zip(history, state_history, strict=True):
        event_payload = coerce_event_for_feature_extraction(event)
        state_payload = _coerce_mapping_like(state_ctx, name="state_ctx")
        feature_state = dict(state_payload)
        if normalized_anchor_lookup:
            feature_state["anchor_lookup"] = dict(normalized_anchor_lookup)

        modality = str(event_payload["modality"])
        current_t = float(event_payload["t"])
        previous_modality_t = last_timestamp_by_modality.get(modality)
        # 首次出现该模态时，modality_gap_dt 设为 0.0 表示"无上次同模态事件"，
        # 而非用相邻事件间隔（dt），后者语义不符。
        modality_gap_dt = 0.0 if previous_modality_t is None else max(0.0, current_t - previous_modality_t)
        feature_state["modality_gap_dt"] = modality_gap_dt
        feature_state["time_since_last_modality"] = modality_gap_dt

        if modality == MODALITY_UWB:
            payload = event_payload.get(PAYLOAD_KEYS[MODALITY_UWB]) if isinstance(event_payload.get(PAYLOAD_KEYS[MODALITY_UWB]), Mapping) else {}
            quality = payload.get("quality")
            if quality is not None:
                quality_value = _safe_float(quality)
                if quality_value is not None:
                    uwb_quality_min = quality_value if uwb_quality_min is None else min(uwb_quality_min, quality_value)
            valid = payload.get("valid")
            if is_bool_like(valid):
                uwb_sample_count += 1
                if not valid:
                    uwb_invalid_count += 1

        # 偷懒审视 Round 4 真修 (audit #17): 删 current_vio_reproj_err /
        # current_tracked_features 整段 VIO payload 读 + reproj_err / tracked_features
        # 计算 — VIO 紧耦合不再输出此两字段, 且 LIQUID_CONTEXT_FEATURE_KEYS 已删
        # vio_reproj_err_slope / tracked_features_drop / vio_reproj_err_prev /
        # tracked_features_prev, 写入的 feature_state 字段也被 model_factory 过滤掉, 真死代码.

        if uwb_quality_min is not None:
            feature_state["uwb_quality_min"] = uwb_quality_min
        if uwb_sample_count > 0:
            feature_state["uwb_invalid_count"] = float(uwb_invalid_count)
            feature_state["uwb_sample_count"] = float(uwb_sample_count)
            feature_state["uwb_invalid_rate"] = float(uwb_invalid_count) / float(uwb_sample_count)
        # 偷懒审视 Round 4 真修 (audit #17): 删 vio_reproj_err_prev / tracked_features_prev /
        # vio_reproj_err_slope / tracked_features_drop 写入 + last_vio_reproj_err /
        # last_tracked_features 状态更新 — 同上, 真死代码.

        feature_state_history.append(feature_state)
        last_timestamp_by_modality[modality] = current_t

    return feature_state_history


# 这个函数负责把单个事件和状态上下文按冻结顺序展开成特征向量。
def build_feature_vector(event, state_ctx, feature_order):
    """按冻结的 `feature_order` 构造单步特征向量和缺失掩码。

    参数:
    `event` 是单个事件对象，可以是映射、dataclass 或带 `to_dict` 的对象。
    `state_ctx` 是当前事件对应的状态上下文。
    `feature_order` 是冻结的字段顺序，决定输出向量每一列代表什么。

    返回值:
    返回一个字典，里面包含 `feature_values` 和 `missing_mask` 两个列表。

    关键局部变量:
    `feature_order` 这里会被转成列表，确保后续能多次遍历。
    `event_payload` 是统一后的事件字典。
    `state_payload` 是统一后的状态字典。
    `feature_values` 收集按顺序排列的浮点特征值。
    `missing_mask` 收集每个字段是否缺失。
    `field_value` 和 `is_missing` 是逐字段解析得到的中间结果。
    """
    # 特征顺序必须是真正的可迭代字段名列表。
    # 字符串/字节串虽可迭代但会被拆成字符/整数序列，必须拒绝；
    # 映射（dict）会静默用键当字段顺序，掩盖调用方传错意图，也必须拒绝。
    if (
        is_string_like(feature_order)
        or isinstance(feature_order, (bytes, bytearray))
        or isinstance(feature_order, Mapping)
    ):
        raise TypeError("feature_order must be an iterable of field names, not a string/bytes/mapping")
    if not isinstance(feature_order, Iterable):
        raise TypeError("feature_order must be an iterable of field names")

    # 先把顺序固定成列表，后面才能稳定地按位置构造向量。
    feature_order = list(feature_order)
    # 空顺序无法构成有效特征向量，与下游 network/model_factory 的非空校验对齐。
    if not feature_order:
        raise ValueError("feature_order must be non-empty")
    # 每个字段名都必须是非空字符串，否则会破坏列语义。
    for index, field_name in enumerate(feature_order):
        if not is_string_like(field_name) or not field_name:
            raise ValueError(f"feature_order[{index}] must be a non-empty string")
    # 重复字段名会产生重复列，导致训练和推理特征错位，与 sensor_contract
    # _validate_field_list 的去重校验口径对齐。
    if len(set(feature_order)) != len(feature_order):
        raise ValueError("feature_order must not contain duplicate field names")

    # 事件本体可以是映射，也可以是带转换方法的对象。
    if isinstance(event, Mapping):
        event_payload = coerce_event_for_feature_extraction(event)
    else:
        # 如果对象自己能转字典，就先转成字典再交给统一事件校验。
        event_to_dict = getattr(event, "to_dict", None)
        if callable(event_to_dict):
            event_dict = event_to_dict()
            # to_dict() 返回值必须校验为映射后再转 dict，否则非字典返回值
            # （如 None/list/str）会在 coerce_event_for_feature_extraction 内部
            # 抛出不够明确的错误，与 _coerce_mapping_like 的校验口径对齐。
            if not isinstance(event_dict, Mapping):
                raise TypeError(
                    f"event.to_dict() must return a mapping, got {type(event_dict).__name__}"
                )
            event_payload = coerce_event_for_feature_extraction(dict(event_dict))
        # dataclass 也允许直接转成字典，避免调用方手工拆字段。
        # 排除 dataclass 类型本身（is_dataclass 对类和实例都返回 True），
        # asdict() 只接受实例，传入类会抛出不够明确的 TypeError。
        elif is_dataclass(event) and not isinstance(event, type):
            event_payload = coerce_event_for_feature_extraction(asdict(event))
        else:
            raise TypeError(f"event must be an Event or dict, got {type(event).__name__}")

    # 状态上下文也要统一成普通字典，后面才能按字段名读取。
    state_payload = _coerce_mapping_like(state_ctx, name="state_ctx")

    feature_values = []  # 这里收集按顺序排列的特征值。
    missing_mask = []  # 这里收集每个特征是否缺失的标记。
    # 逐个字段按冻结顺序提取，保证训练和推理看到的是同一列定义。
    for field_name in feature_order:
        # _resolve_feature_value 按合同返回 (value, missing_flag)，不会因
        # 单个字段非数值而抛异常（非数值会回退到缺失填充）。
        # 若发生结构性异常（如 modality 缺失），应让原始异常传播，不掩盖。
        field_value, is_missing = _resolve_feature_value(event_payload, state_payload, field_name)
        feature_values.append(field_value)
        missing_mask.append(bool(is_missing))

    # 返回值里同时带特征值和缺失掩码，方便后续窗口构造。
    return {
        "feature_values": feature_values,  # 按 feature_order 排列的特征值列表。
        "missing_mask": missing_mask,  # 与 feature_values 一一对应的缺失标记列表。
    }
