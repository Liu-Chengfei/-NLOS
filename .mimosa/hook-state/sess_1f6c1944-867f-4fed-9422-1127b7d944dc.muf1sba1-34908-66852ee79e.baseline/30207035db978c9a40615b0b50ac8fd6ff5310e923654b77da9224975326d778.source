"""场景编码与解码协议模块。

文件职责：
  定义场景编码格式 S(A_level,N_level,V_level,K_value,M_level)，
  提供 SceneSpec 结构化对象与字符串之间的双向转换，并校验每个轴值
  是否属于当前协议允许的层级集合。

  2026-08-31：G 轴并入 K 轴（G_level → K_value），SceneSpec 升级为
  A/N/V/K/M 五轴，与 SCENE_AXES 保持一致。

本文件绝对不负责：
  不定义协议内容本身（轴名、层级名、参数表）。
  不修改协议文件。

上游依赖：protocol/scene_axis_protocol.py（提供协议加载和轴校验）
下游调用者：protocol/__init__.py、protocol/liquid_bridge_contract.py

输入对象定义：
  - SceneSpec   五轴场景规格对象（frozen dataclass）
  - scene_code  格式为 S(A,N,V,K,M) 的字符串

输出对象定义：
  - encode_scene   SceneSpec → scene_code
  - decode_scene   scene_code → SceneSpec

核心变量定义：
  - _AXIS_ORDER          冻结的轴遍历顺序 (A, N, V, K, M)，复用 scene_axis_protocol.AXES
  - _AXIS_FIELDS         轴名到 SceneSpec 字段名的映射（当前覆盖 A/N/V/K/M）
  - _SCENE_CODE_PATTERN  scene_code 的正则解析模式，格式 S(A,N,V,K,M)

关键设计决策：
  - 编码/解码时都会校验轴值是否在当前协议允许集合内。
  - 默认协议快照不做缓存，每次从 load_scene_axis_protocol() 重新加载；
    原因是 Python 的 id() 在对象回收后可复用，导致 monkeypatch 场景下
    缓存键碰撞。encode_scene/decode_scene 不是热路径，性能影响可忽略。
  - SceneSpec 是 frozen dataclass，保证不可变和可哈希。
"""

from __future__ import annotations  # 允许前向类型标注。

from dataclasses import dataclass  # 用于定义场景规格对象。
import re  # 用于解析 scene_code。

from liquidloc.common.validation import is_string_like  # 统一字符串类型校验函数。
from liquidloc.protocol.scene_axis_protocol import AXES as _AXIS_ORDER, AXIS_METADATA_KEYS, get_nominal_levels, get_protocol_axes, load_scene_axis_protocol  # 复用场景轴协议。

_AXIS_FIELDS = {  # 轴名到 SceneSpec 字段名的映射。
    "A": "A_level",  # A 轴对应字段名。
    "N": "N_level",  # N 轴对应字段名。
    "V": "V_level",  # V 轴对应字段名。
    "K": "K_value",  # K 轴对应字段名。
    "M": "M_level",  # M 轴对应字段名（G 轴并入 K，M 正式进入 scene_code）。
}  # 轴名到字段名的映射结束。

_SCENE_CODE_PATTERN = re.compile(  # 场景编码正则，格式为 S(A,N,V,K,M)；M 段可省略（省略时回退协议名义档 M0）。
    r"^S\("  # 开头 S(。
    r"([^,\s)]+),"  # A 轴 token，不含逗号、空白和右括号。
    r"([^,\s)]+),"  # N 轴 token。
    r"([^,\s)]+),"  # V 轴 token。
    r"([^,\s)]+)"  # K 轴 token。
    r"(?:,([^,\s)]+))?"  # M 轴 token（可选；2026-08-31 五轴协议后大量事件/夹具使用 S(A,N,V,K) 四段简写）。
    r"\)$"  # 结尾 )。
)  # 场景编码正则结束。


@dataclass(slots=True, frozen=True)  # 场景规格对象：只读且使用 slots，节省内存。
class SceneSpec:
    """场景规格对象，封装五个场景轴的取值。

    每个实例代表一个完整的场景描述，由五个轴的层级值唯一确定。
    该对象是 frozen dataclass，创建后不可修改，且可哈希、可做字典键。

    属性：
        A_level：A 轴层级（异步程度），如 "A0"、"A1" 等。
        N_level：N 轴层级（非视距程度），如 "N0"、"N1" 等。
        V_level：V 轴层级（视觉退化程度），如 "V0"、"V1" 等。
        K_value：K 轴取值（锚点数量），如 "K0"、"K1"、"K3" 等（五轴档位协议仅保留三档）。
        M_level：M 轴层级（模态完整性），如 "M0"、"M1" 等。
    """

    A_level: str  # A 轴层级（异步程度）。
    N_level: str  # N 轴层级（非视距程度）。
    V_level: str  # V 轴层级（视觉退化程度）。
    K_value: str  # K 轴取值（锚点数量）。
    M_level: str  # M 轴层级（模态完整性；G 轴于 2026-08-31 并入 K，M 正式进入 scene_code）。


def _get_default_snapshot() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """获取默认协议的轴层级快照。

    每次都从 load_scene_axis_protocol() 重新加载，不做缓存。
    原因：用 id() 做缓存键在 monkeypatch 场景下不可靠（Python id 可复用），
    而 encode_scene/decode_scene 不是热路径，重复加载的性能开销可忽略。
    """
    protocol_cfg = load_scene_axis_protocol()  # 读取默认协议。
    axes = get_protocol_axes(protocol_cfg)  # 提取并校验 axes。
    return tuple(  # 返回冻结的轴快照。
        (  # 每个轴一项。
            axis,  # 轴名。
            tuple(k for k in axes[axis].keys() if k not in AXIS_METADATA_KEYS),  # 该轴的层级键元组，排除元数据键。
        )  # 单项结束。
        for axis in _AXIS_ORDER  # 按固定轴顺序遍历。
    )  # 元组结束。


def axis_levels(protocol_cfg: dict | None = None) -> dict[str, tuple[str, ...]]:  # 获取轴层级表。
    """获取每个轴可用的层级集合。

    参数：
        protocol_cfg: 外部协议对象；为 None 时使用默认协议快照。
    返回：
        轴名到层级元组的映射。
    """
    if protocol_cfg is None:  # 没给协议时，使用默认快照。
        return dict(_get_default_snapshot())  # 从快照转换为 dict。
    axes = get_protocol_axes(protocol_cfg)  # 对外部协议做完整校验。
    return {  # 返回层级表。
        axis: tuple(k for k in axes[axis].keys() if k not in AXIS_METADATA_KEYS)  # 每个轴对应的层级元组，排除元数据键。
        for axis in _AXIS_ORDER  # 按固定顺序构造。
    }  # 层级表结束。


def coerce_axis_value(  # 单轴值校验+返回函数。
    axis: str,  # 轴名。
    value: str,  # 轴值。
    levels: dict[str, tuple[str, ...]] | None = None,  # 可选层级表。
) -> str:  # 返回校验后的字符串。
    """校验单个轴值是否属于允许集合。"""
    # 第 8 轮审查 MEDIUM-1 修复：coerce_axis_value 在 encode_scene/decode_scene 中
    # 每场景被调用 5 次（A/N/V/K/M），高频 print_dict 违反工程规范
    # "Recursive functions and entry points should avoid print_dict calls"。
    if axis not in _AXIS_FIELDS:  # 轴名必须合法。
        raise ValueError(f"axis must be one of {sorted(_AXIS_FIELDS.keys())}, got {axis!r}")
    field_name = _AXIS_FIELDS[axis]  # 找到字段名。
    if not is_string_like(value):  # 值必须是字符串。
        raise TypeError(f"{field_name} must be a str, got {type(value).__name__}")  # 类型不对就报错。
    if levels is None:  # 没有层级表时。
        levels = axis_levels()  # 自动加载默认层级表。
    if axis not in levels:  # 层级表缺少当前轴。
        raise ValueError(f"levels dict is missing axis {axis!r}")  # 报出缺失轴名。
    # 五轴档位协议：K 轴仅 K0/K1/K3；严格校验，不做旧等级兼容。
    if value not in levels[axis]:  # 值不在允许范围内。
        raise ValueError(f"{field_name} must be one of {levels[axis]}, got {value!r}")  # 报出范围。
    return str(value)  # 归一化为 Python 原生 str，防止 numpy.str_ 等子类型穿透。


def encode_scene(scene_spec: SceneSpec) -> str:  # 编码函数。
    """把 SceneSpec 编码成冻结的 scene_code 字符串。

    编码时校验每个轴值是否属于当前协议允许的层级集合。
    """
    # 第 8 轮审查 MEDIUM-1 修复：encode_scene 是 scene_sampler 热路径入口，
    # 每场景调用一次，大规模实验下 print_dict 产生大量日志，违反工程规范。
    if not isinstance(scene_spec, SceneSpec):  # 编码器只接受结构化对象。
        raise TypeError(f"scene_spec must be a SceneSpec, got {type(scene_spec).__name__}")  # 类型不对拒绝。
    _levels = axis_levels()  # 先拿允许集合（只加载一次，五轴共享；encode_scene 校验 A/N/V/K/M）。
    A_level = coerce_axis_value("A", scene_spec.A_level, _levels)  # 校验 A。
    N_level = coerce_axis_value("N", scene_spec.N_level, _levels)  # 校验 N。
    V_level = coerce_axis_value("V", scene_spec.V_level, _levels)  # 校验 V。
    K_value = coerce_axis_value("K", scene_spec.K_value, _levels)  # 校验 K。
    M_level = coerce_axis_value("M", scene_spec.M_level, _levels)  # 校验 M。
    scene_code = (  # 拼接场景编码。
        f"S("  # 前缀。
        f"{A_level},"  # A 轴值。
        f"{N_level},"  # N 轴值。
        f"{V_level},"  # V 轴值。
        f"{K_value},"  # K 轴值。
        f"{M_level}"  # M 轴值。
        f")"  # 后缀。
    )  # 拼接结束。
    return scene_code  # 返回编码结果。


def decode_scene(scene_code: str) -> SceneSpec:  # 解码函数。
    """把冻结 scene_code 字符串解码回 SceneSpec。

    解码时校验格式和每个轴值是否属于当前协议允许的层级集合。
    """
    # 第 8 轮审查 MEDIUM-1 修复：decode_scene 是 scene_sampler 热路径入口，
    # 每场景调用一次，大规模实验下 print_dict 产生大量日志，违反工程规范。
    if not is_string_like(scene_code):  # 输入必须是字符串。
        raise TypeError(f"scene_code must be a str, got {type(scene_code).__name__}")  # 类型不对就中止。
    parsed_tokens = _SCENE_CODE_PATTERN.fullmatch(scene_code)  # 做全量匹配。
    if parsed_tokens is None:  # 格式不对。
        raise ValueError("scene_code must match S(A,N,V,K,M) or S(A,N,V,K) where each field is a valid axis level (e.g. S(A0,N0,V0,K3,M0))")  # 报出格式要求。
    _levels = axis_levels()  # 再次取允许集合（只加载一次）。
    A_level, N_level, V_level, K_value, M_level = parsed_tokens.groups()  # 按固定顺序拆 token（M 可能为 None）。
    if M_level is None:  # 四段简写 S(A,N,V,K)：M 缺省回退协议名义档，不硬编码 M0。
        M_level = get_nominal_levels()["M"]  # 从协议动态读取正常等级名。
    return SceneSpec(  # 返回结构化对象。
        A_level=coerce_axis_value("A", A_level, _levels),  # 校验 A 轴值。
        N_level=coerce_axis_value("N", N_level, _levels),  # 校验 N 轴值。
        V_level=coerce_axis_value("V", V_level, _levels),  # 校验 V 轴值。
        K_value=coerce_axis_value("K", K_value, _levels),  # 校验 K 值。
        M_level=coerce_axis_value("M", M_level, _levels),  # 校验 M 轴值。
    )  # 返回结构化对象结束。
