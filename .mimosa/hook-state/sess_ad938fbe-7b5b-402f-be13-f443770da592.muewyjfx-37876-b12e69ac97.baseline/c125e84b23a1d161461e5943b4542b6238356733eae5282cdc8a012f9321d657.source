"""质量值读取与归一化模块。

核心数据流：
    raw_quality → normalize_quality_value → [0, 1] float
    event → get_event_quality → [0, 1] float

上游依赖：
    liquidloc.common.constants（DEFAULT_THRESHOLDS, MODALITY_UWB, MODALITY_VIO, PAYLOAD_KEYS）

下游调用者：
    liquidloc.sensors.__init__（对外导出 get_event_quality, normalize_quality_value）

历史说明：
    早期 vision_model/uwb_model 曾调用 get_event_quality，但已改为直接使用
    coerce_finite_scalar 做严格范围校验（超界直接报错）。get_event_quality
    当前作为独立公共 API 保留，使用裁剪模式（越界裁剪到 [0, 1] 并警告），
    供外部消费者按需调用。两者口径不同，调用方应按需选择。

关键设计决策：
    - normalize_quality_value 是底层归一化函数，处理 None 回退、类型拒绝、
      NaN/inf 拒绝和 [0, 1] 裁剪。拒绝 bool（包括 np.bool_）、字符串和 bytes。
    - get_event_quality 是高层事件读取函数，只接受 UWB/VIO 两种模态。
      它先读取 modality → 读取 payload → 读取 payload.quality → 调用 normalize_quality_value。
    - getattr 对没有 modality/payload 属性的对象返回 None，后续校验会兜底，
      但错误消息可能不够精确（如报"Unsupported modality for quality extraction: None"而非"Missing modality"）。
    - 当 raw_quality 为 None 时用 default_quality 替换，之后 float(default_quality)
      可能失败，错误消息统一报"quality must be numeric or None"——实际可能是
      default_quality 本身非法，调用方应确保 default_quality 是合法浮点数。
"""

from __future__ import annotations  # 允许前向类型标注。

from collections.abc import Mapping  # 用于判断 event 或 payload 是否按映射结构提供字段。
import warnings  # 质量越界警告。

from liquidloc.common.constants import DEFAULT_THRESHOLDS, MODALITY_UWB, MODALITY_VIO, PAYLOAD_KEYS  # 读取协议和阈值常量。
from liquidloc.common.validation import coerce_finite_scalar, is_bool_like  # 集中判断 bool / np.bool_ 和有限浮点转换。


def _validate_quality_value(raw_quality, *, strict_mode: bool = False) -> float:
    """校验并归一化 quality 值的共享内核。

    统一 sensors 层与下游消费者（uwb_model/vision_model 的 extract_*_measurement、
    estimators 的 _quality_value 等）对 quality 字段的校验口径，消除"裁剪模式"
    与"严格模式"两套独立实现导致的口径漂移风险。

    校验流程：
        1. 拒绝 bool（包括 np.bool_）、字符串、bytes 类型（两种模式一致）
        2. 通过 coerce_finite_scalar 转为有限 float，拒绝 NaN/inf（两种模式一致）
        3. 范围处理由 strict_mode 切换：
           * strict_mode=False（裁剪模式）：越界时裁剪到 [quality_min, quality_max]
             并发出 RuntimeWarning。normalize_quality_value/get_event_quality 使用。
           * strict_mode=True（严格模式）：越界时直接抛 ValueError。
             extract_*_measurement 等严格路径使用。

    参数：
        raw_quality: 原始质量值，必须非 None。调用者负责处理 None → default_quality
            的回退逻辑（由 normalize_quality_value 完成）。
        strict_mode: True 时严格报错，False 时裁剪 + 警告。默认 False 保持
            与原 normalize_quality_value 行为兼容。

    返回：
        校验后的有限浮点数。strict_mode=False 时保证落在
        [quality_min, quality_max]；strict_mode=True 时若返回则同样落在该区间。

    异常：
        TypeError: raw_quality 是 bool、字符串、bytes 或无法转 float。
        ValueError: raw_quality 是 NaN 或无穷大；或 strict_mode=True 且 raw_quality 越界。

    注意：
        - float(True) = 1.0 和 float(False) = 0.0 在技术上是合法的，
          但本函数显式拒绝 bool（包括 np.bool_），因为布尔语义与质量值不兼容。
        - 兼容 numpy 数值标量（np.float64、np.int64 等），因为它们可以通过 float() 转换。
        - warnings.warn 的 stacklevel=2 指向本函数的直接调用方；若调用链为
          get_event_quality → normalize_quality_value → _validate_quality_value，
          警告位置会指向 normalize_quality_value（比原实现深一帧），属可接受偏差。
    """
    if is_bool_like(raw_quality) or isinstance(raw_quality, (str, bytes, bytearray)):  # 布尔值和字符串都不是合法质量。
        raise TypeError("quality must be numeric or None")  # 直接报错，与原 normalize_quality_value 口径一致。

    working_quality = coerce_finite_scalar(raw_quality, name="quality")  # 统一转有限 float，非数值/bool 报 TypeError，非有限报 ValueError。

    quality_min = DEFAULT_THRESHOLDS["quality_min"]  # 质量下界。
    quality_max = DEFAULT_THRESHOLDS["quality_max"]  # 质量上界。
    if working_quality < quality_min:  # 小于下界时按模式处理。
        if strict_mode:  # 严格模式直接报错，与 extract_*_measurement 的 coerce_finite_scalar(min_value=...) 口径一致。
            raise ValueError(
                f"quality={working_quality} < quality_min={quality_min}"
            )
        warnings.warn(
            f"quality={working_quality} < quality_min={quality_min}, "
            f"clipped to {quality_min}. "
            f"Upstream should ensure quality ∈ [0, 1].",
            RuntimeWarning,
            stacklevel=2,
        )  # 与 MATLAB 行为对齐：越界不应静默，至少发出警告。
        working_quality = quality_min  # 裁到最小阈值。
    if working_quality > quality_max:  # 大于上界时按模式处理。用独立 if 而非 elif，兼容 quality_min > quality_max 的极端配置。
        if strict_mode:  # 严格模式直接报错。
            raise ValueError(
                f"quality={working_quality} > quality_max={quality_max}"
            )
        warnings.warn(
            f"quality={working_quality} > quality_max={quality_max}, "
            f"clipped to {quality_max}. "
            f"Upstream should ensure quality ∈ [0, 1].",
            RuntimeWarning,
            stacklevel=2,
        )  # 与 MATLAB 行为对齐：越界不应静默，至少发出警告。
        working_quality = quality_max  # 裁到最大阈值。
    return working_quality  # 返回校验后的质量值。


def normalize_quality_value(raw_quality: float | int | None, default_quality: float = 0.0) -> float:
    """把原始 quality 规整成 [0, 1] 区间内的浮点数。

    处理流程：
        1. 如果 raw_quality 为 None，回退到 default_quality
        2. 调用 _validate_quality_value(strict_mode=False) 做类型拒绝、有限性
           校验和裁剪归一化（越界裁剪到 [0, 1] 并发出 RuntimeWarning）

    参数：
        raw_quality: 原始质量值，可为 None（缺失时用 default_quality 替代）。
            接受 int、float、numpy 数值标量，拒绝 bool、字符串、bytes。
        default_quality: 当 raw_quality 为 None 时使用的默认值，应为合法浮点数。
            默认 0.0。如果 default_quality 本身非法（如字符串），
            错误消息会报"quality must be numeric or None"。

    返回：
        裁剪到 [0, 1] 区间的有限浮点数。

    异常：
        TypeError: raw_quality（或 default_quality）是 bool、字符串、bytes 或无法转 float。
        ValueError: raw_quality（或 default_quality）是 NaN 或无穷大。

    注意：
        - 类型拒绝、有限性校验和范围裁剪均委托给 _validate_quality_value，
          与严格模式（extract_*_measurement）共享同一校验内核，仅范围处理
          行为不同（裁剪+警告 vs 严格报错）。
        - float(True) = 1.0 和 float(False) = 0.0 在技术上是合法的，
          但本函数显式拒绝 bool（包括 np.bool_），因为布尔语义与质量值不兼容。
        - 兼容 numpy 数值标量（np.float64、np.int64 等），因为它们可以通过 float() 转换。
    """
    is_missing = raw_quality is None  # 判断是否缺失。
    # 当 raw_quality 为 None 时用 default_quality 替换。
    # 如果 default_quality 本身非法（如字符串），后续 _validate_quality_value 会失败，
    # 错误消息统一报 "quality must be numeric or None"——
    # 调用方应确保 default_quality 是合法浮点数。
    working_quality = default_quality if is_missing else raw_quality  # 缺失则回退到默认值。
    return _validate_quality_value(working_quality, strict_mode=False)  # 委托给共享内核：类型+有限+裁剪归一化。


def get_event_quality(event: Mapping | object, default_quality: float = 0.0) -> float:
    """从 UWB 或 VIO 事件里读取并归一化 quality 字段。

    处理流程：
        1. 读取 modality 字段（dict 用 get，对象用 getattr）
        2. 校验 modality 必须是 UWB 或 VIO
        3. 通过 PAYLOAD_KEYS 找到对应 payload 键名
        4. 读取 payload（dict 用 get，对象用 getattr）
        5. 校验 payload 不为 None
        6. 从 payload 中读取 quality（dict 用 get，对象用 getattr）
        7. 调用 normalize_quality_value 归一化

    参数：
        event: 字典或对象，必须包含 ``"modality"`` 和对应 payload 字段。
            - 字典事件：通过 ``event["modality"]`` 和 ``event.get(payload_key)`` 读取
            - 对象事件：通过 ``getattr(event, "modality")`` 和 ``getattr(event, payload_key)`` 读取
        default_quality: 当 quality 为 None 时使用的默认值，传递给 normalize_quality_value。
            默认 0.0。

    返回：
        裁剪到 [0, 1] 区间的有限浮点数。

    异常：
        ValueError: modality 不是 UWB/VIO，或 payload 缺失。
        TypeError: payload 不支持 quality 字段读取（既非 Mapping 又无 quality 属性），
            或 quality/default_quality 类型非法。
        ValueError: quality 或 default_quality 是 NaN 或无穷大。

    注意：
        - getattr 对没有 modality/payload 属性的对象返回 None，
          后续 modality not in (UWB, VIO) 检查会抛 ValueError，
          错误消息为 "Unsupported modality for quality extraction: None"，
          不如"缺失 modality"精确——这是已知限制，由测试用例固定为预期行为。
        - 只接受 UWB 和 VIO 两种模态，IMU 事件不支持质量字段。
        - 本函数使用裁剪模式（越界裁剪到 [0, 1] 并发出 RuntimeWarning），
          与 vision_model/uwb_model 中 extract_*_measurement 的严格报错模式
          （越界直接抛 ValueError）不同。两者口径不一致，调用方应按需选择：
          需要容错归一化时用 get_event_quality，需要严格校验时用 extract_*_measurement。
        - dict payload 缺失 quality 键时返回 None → 走 default_quality；
          对象 payload 没有 quality 属性时抛 TypeError。这是 dict/object 路径
          的语义差异，由测试用例固定为预期行为。
    """
    if isinstance(event, Mapping):  # 映射事件按键读取。
        modality = event.get("modality")  # 读取模态字段。
    else:  # 对象事件按属性读取。
        # getattr 对没有 modality 属性的对象返回 None，而非 AttributeError。
        # 后续 modality not in (UWB, VIO) 检查会兜底，
        # 但错误消息报 "Unsupported modality for quality extraction: None" 而非 "Missing modality"——已知限制。
        modality = getattr(event, "modality", None)  # 没有属性时返回 None。

    if modality not in (MODALITY_UWB, MODALITY_VIO):  # 这里只接受 UWB 和 VIO 两种模态。
        raise ValueError(f"Unsupported modality for quality extraction: {modality!r}")  # 模态不支持就报错。

    payload_key = PAYLOAD_KEYS[modality]  # 通过模态找到对应 payload 的标准键名。
    if isinstance(event, Mapping):  # 映射事件按键读取 payload。
        payload = event.get(payload_key)  # 取出 payload。
    else:  # 对象事件按属性读取 payload。
        # 与 modality 读取同理，getattr 对没有 payload 属性的对象返回 None。
        # 后续 payload is None 检查会兜底。
        payload = getattr(event, payload_key, None)  # 没有属性时返回 None。

    if payload is None:  # payload 缺失时不能继续。
        raise ValueError(f"Missing payload for modality={modality}: {payload_key}")  # 明确指出缺的是哪个 payload。
    if isinstance(payload, Mapping):  # payload 是字典时，直接取 quality 键。
        raw_quality = payload.get("quality")  # 读取原始 quality。
    elif hasattr(payload, "quality"):  # payload 是对象时，读取同名属性。
        raw_quality = getattr(payload, "quality")  # 读取对象上的 quality。
    else:  # 其他类型都不符合质量字段协议。
        raise TypeError(f"{payload_key} must be a mapping or expose a quality attribute")  # 明确说明 payload 类型不对。
    clipped_quality = normalize_quality_value(raw_quality, default_quality=default_quality)  # 统一裁剪和归一化。
    return clipped_quality  # 返回标准化后的质量值。
