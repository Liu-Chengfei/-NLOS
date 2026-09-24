"""几何等级场景构建模块。

职责：
    根据几何档位（K0/K1/K3，五轴档位协议 K 轴含几何分布语义）生成不同的 anchor 布局，并输出几何质量报告。
    模块本身不做定位，也不直接参与滤波，只是把"锚点怎么摆"这件事规整成
    上层可以直接使用的场景任务。

    核心输出有两类：
    1. anchor_layout：描述锚点位置和布局标识，供 UWB 预测和仿真使用。
    2. geometry_report：描述几何条件、几何评分和布局强度，供实验日志和审计使用。

本文件绝对不负责：
    不修改时间轴，不制造 NLOS，不做视觉退化，不改事件 payload。

上游依赖：
    - liquidloc.sensors.anchor_model（validate_anchor_layout）— 锚点布局合法性校验

下游调用者：
    - liquidloc.pipelines.core_pipeline       — 核心流水线在场景构建阶段调用 build_anchor_layout
    - liquidloc.scenarios.scene_sampler       — 场景采样器在双轴交叉模式下间接使用
    - tests/scenarios/test_geometry_levels.py — 单元测试

核心变量：
    - 无模块级变量，全部通过函数返回

关键设计决策：
    - 几何条件（geom_condition）从协议配置中读取，归一化到 [0, 1] 后映射为布局强度（strength）。
    - 布局强度决定锚点分布形状：strength=0 时锚点排成一条直线（最弱几何），
      strength=1 时锚点均匀分布在圆周上（最强几何）。
    - 几何评分（geom_score）用协方差矩阵的最小/最大特征值比衡量分布均匀程度，
      越接近 1 表示分布越均匀。
    - project_anchor_layout_to_reference 把合成布局投影到参考布局的尺度和质心上，
      保证不同几何等级之间的比较只反映形状差异，不受尺度和位置影响。
"""

from __future__ import annotations  # 允许本模块内部类型在注解里直接引用。

import math  # 用来计算几何分布、均值、方差和平方根。
from collections.abc import Mapping, Sequence  # 用来判断配置结构是不是映射或序列。
from typing import Any  # 用于宽松类型标注。

from liquidloc.sensors.anchor_model import _compute_geometry_score_core, validate_anchor_layout  # 用来校验锚点布局是否合法，以及共享几何评分内核。
from liquidloc.common.validation import coerce_finite_scalar, is_integer, is_string_like, sample_axis_interval  # 统一实数类型校验函数和有限浮点转换，以及轴参数区间采样。


# 退化布局半径判定容差：RMS 半径低于此值视为退化布局，投影时所有点收敛到参考质心。
# 与 visual_levels._is_close 默认容差 1e-9 同量级，但用途不同（此处是退化判定，非近似比较）。
_DEGENERATE_RADIUS_TOL = 1e-9

# 浮点零容差：用于 strength == 0.0、scale == 0.0 等浮点等比判断，避免精度误差导致分支误判。
_FLOAT_ZERO_TOL = 1e-12

# 投影缩放因子上界：防止退化布局导致 scale_to_reference 爆炸到极大值。
_SCALE_CLAMP_UPPER = 1e6


def _read_cfg_value(obj, field_name: str) -> Any:
    """从映射或对象属性里读取配置字段。

    兼容两种配置形态：字典按键读取，对象按属性读取，
    统一上层调用逻辑。

    Args:
        obj: 配置对象，可以是映射（dict）或带属性的轻量对象。
        field_name: 要读取的字段名。

    Returns:
        字段值，缺失时返回 None。
    """
    if isinstance(obj, Mapping):  # 如果是映射，就按 key 取。
        return obj.get(field_name)
    return getattr(obj, field_name, None)  # 否则尝试按属性名取，兼容轻量配置对象。


def _normalize_geometry_levels(geometry_cfg) -> dict[str, dict[str, float]]:
    """把几何配置规整成统一的 level 字典。

    支持两种配置写法（顶层必须是映射）：
    1. 映射式：{level_name: {geom_condition: ...}, ...}
       可嵌套在 levels 子项下，也可直接作为顶层。
    2. 列表式：geometry_cfg["levels"] 为 [{name: ..., geom_condition: ...}, ...]
       （顶层 geometry_cfg 仍必须是映射，仅 levels 子项可为列表）

    无论哪种写法，最终都统一成 {LEVEL_NAME: {geom_condition: float}} 的
    标准结构，level 名统一去首尾空白（不强制大写，与协议层 resolve_axis_level
    仅 .strip() 口径对齐）。

    Args:
        geometry_cfg: 原始几何配置，顶层必须是映射（支持 levels 子项或直接作为顶层）。

    Returns:
        dict[str, dict[str, float]]: 规整后的 level 字典，
            键为去空白后的 level 名，值为包含 geom_condition 的字典。

    Raises:
        TypeError: 配置结构不符合协议要求时抛出。
        ValueError: geom_condition 不满足协议约束（必须严格正）或 level 名为空白时抛出。
    """
    if not isinstance(geometry_cfg, Mapping):  # 主配置必须是映射，与 A/N/V 轴入口校验口径对齐。
        raise TypeError(f"geometry_cfg must be a mapping, got {type(geometry_cfg).__name__}")
    raw_levels = geometry_cfg.get("levels", geometry_cfg)  # 先尝试从 levels 子项读取，没有就直接用自身。
    if isinstance(raw_levels, Mapping):  # levels 如果是映射，就按 level 名索引。
        normalized_levels = {}  # 这里收集规整后的 level。
        for level_name, level_payload in raw_levels.items():  # 逐个 level 做规整。
            if not is_string_like(level_name):  # level 名必须是字符串，与 V/A 轴 is_string_like 入口校验对齐。
                raise TypeError(f"geometry level name must be a string, got {type(level_name).__name__}")
            if not level_name.strip():  # level 名不能为空白，与 build_anchor_layout L369-370 和 A/N/V 轴空白校验口径对齐。
                raise ValueError("geometry level name must not be blank")
            if not isinstance(level_payload, Mapping):  # 非映射的 level 定义是配置错误。
                raise TypeError(f"geometry level '{level_name}' must be a mapping, got {type(level_payload).__name__}")
            geom_condition = _read_cfg_value(level_payload, "geom_condition")  # 读取这个 level 的几何条件。
            if geom_condition is None:  # 缺失 geom_condition 应保持 ValueError 语义（而非 TypeError 逃出 try-catch）。
                raise ValueError(f"geometry level '{level_name}' must define geom_condition, got None")
            geom_condition = sample_axis_interval(geom_condition, rng=None, name=f"geometry level '{level_name}'.geom_condition")  # 区间标量化（支持 [min, max] 区间均匀采样，标量直接返回）。
            geom_condition = coerce_finite_scalar(
                geom_condition,
                name="geometry level geom_condition",
            )  # 几何条件必须是有限数字。
            if geom_condition <= 0.0:  # 协议 G 轴约束：geom_condition 必须严格正（scene_axis_protocol.py L257-258）。
                raise ValueError(f"geometry level geom_condition must be positive, got {geom_condition}")
            normalized_levels[level_name.strip()] = {"geom_condition": geom_condition}  # 统一 level 名空白，与协议层 resolve_axis_level 仅 .strip() 口径对齐（不强制大写，保持大小写敏感）。
        if normalized_levels:  # 只要有合法 level，就可以直接返回。
            return normalized_levels
    elif isinstance(raw_levels, Sequence) and not isinstance(raw_levels, (str, bytes, bytearray, memoryview)):  # 也兼容列表式配置，排除集与 V/A 轴对齐。
        normalized_levels = {}  # 重新收集列表式 level。
        for level_payload in raw_levels:  # 列表里每项都应描述一个 level。
            if not isinstance(level_payload, Mapping):  # 列表项必须是映射，与映射分支口径对齐。
                raise TypeError(f"geometry level entry must be a mapping, got {type(level_payload).__name__}")
            template_name = _read_cfg_value(level_payload, "name")  # 列表式配置必须带名字。
            if not is_string_like(template_name):  # 名字必须是字符串。
                raise TypeError("geometry level entry must define a string name")
            if not template_name.strip():  # level 名不能为空白，与映射分支 L106 口径对齐。
                raise ValueError("geometry level entry name must not be blank")
            geom_condition = _read_cfg_value(level_payload, "geom_condition")  # 也必须带几何条件。
            if geom_condition is None:  # 缺失应保持 ValueError 语义。
                raise ValueError(f"geometry level '{template_name}' must define geom_condition, got None")
            geom_condition = sample_axis_interval(geom_condition, rng=None, name=f"geometry level '{template_name}'.geom_condition")  # 区间标量化。
            geom_condition = coerce_finite_scalar(
                geom_condition,
                name="geometry level geom_condition",
            )  # 几何条件也必须是有限数字。
            if geom_condition <= 0.0:  # 协议 G 轴约束：geom_condition 必须严格正。
                raise ValueError(f"geometry level geom_condition must be positive, got {geom_condition}")
            normalized_levels[template_name.strip()] = {"geom_condition": geom_condition}  # 统一成同一种 level 字典结构，与映射分支口径对齐（仅 .strip()，不强制大写）。
        if normalized_levels:  # 只要成功规整，就直接返回。
            return normalized_levels
    raise TypeError("geometry_cfg must expose authority geom_condition definitions")  # 否则说明配置结构不符合协议要求。


def _load_geometry_template(geometry_level: str, geometry_cfg) -> tuple[str, float, float, float]:
    """从配置里读取指定 geometry level 的模板参数。

    先把配置规整成统一 level 表，再按 level 名查找，
    同时返回所有 level 的条件范围，供后续归一化使用。

    Args:
        geometry_level: 几何档位名（如 "K0"、"K3"，五轴协议 K 轴含几何分布语义）。
        geometry_cfg: 几何配置对象。

    Returns:
        tuple[str, float, float, float]: (归一化等级名, 几何条件值, 最小条件值, 最大条件值)。

    Raises:
        TypeError: geometry_level 不是字符串时抛出。
        ValueError: 请求的等级不存在时抛出。
    """
    if not is_string_like(geometry_level):  # 入口校验等级名类型，与 A/N/V 轴 is_string_like 入口校验对齐。
        raise TypeError(f"geometry_level must be a string, got {type(geometry_level).__name__}")
    levels = _normalize_geometry_levels(geometry_cfg)  # 先把配置规整成统一 level 表。
    normalized_level = str(geometry_level).strip()  # level 名统一去空白，与协议层 resolve_axis_level 仅 .strip() 口径对齐（不强制大写，保持大小写敏感）。
    if normalized_level not in levels:  # 请求的 level 如果不存在，就直接报错。
        raise ValueError(f"unsupported geometry_level: {geometry_level}")

    geom_condition = levels[normalized_level]["geom_condition"]  # 取出该 level 对应的几何条件。
    condition_values = [payload["geom_condition"] for payload in levels.values()]  # 收集所有条件，后面计算归一化范围。
    return normalized_level, geom_condition, min(condition_values), max(condition_values)  # 返回 level、条件和归一化边界。


def _build_anchor_positions(anchor_count: int, strength: float) -> list[list[float]]:
    """根据锚点数量和退化强度生成锚点坐标。

    2026-08-31 K 轴改造：strength 语义反转。
    - strength=0（K0 好几何）：矩形对称四角，最优几何。
    - strength=1（K3 严重退化）：3 锚共线 + 1 锚孤立，最差几何。
    - 0 < strength < 1：椭圆短轴随 strength 增大而缩小，分布越来越偏向一条直线。

    旧语义"strength=0 共线"是"几何质量=强度"映射，已不适用新协议。
    新协议 geom_condition 越大几何质量越差（K3=10, K0=1），因此
    axis_strength = (geom_condition-1)/(10-1) 越大 = 越退化。

    Args:
        anchor_count: 锚点数量，本内部函数要求 >= 1。
        strength: 退化强度，范围 [0, 1]，越大表示几何越退化。

    Returns:
        list[list[float]]: 锚点坐标列表，每个坐标为 [x, y] 二维浮点列表。

    Raises:
        TypeError: anchor_count 不是整数或 strength 不是数值时抛出。
        ValueError: anchor_count < 1、strength 不在 [0, 1] 或非有限时抛出。
    """
    if not is_integer(anchor_count):  # 锚点数量必须是整数。
        raise TypeError(f"anchor_count must be an integer, got {type(anchor_count).__name__}")
    anchor_count = int(anchor_count)  # 确保 Python 原生 int，防止 np.int64 泄漏。
    if anchor_count < 1:  # 至少要有 1 个锚点。
        raise ValueError(f"anchor_count must be >= 1, got {anchor_count}")
    strength = coerce_finite_scalar(strength, name="strength")  # 强度必须是有限数字，拒绝 NaN/Inf。
    if not 0.0 <= strength <= 1.0:  # 强度范围必须在 [0, 1]，与 docstring 声明对齐。
        raise ValueError(f"strength must be within [0, 1], got {strength}")

    if anchor_count == 1:  # 单锚点时直接放在原点。
        return [[0.0, 0.0]]

    # K3 特殊处理：4 锚近共线退化（3 锚近似共线 + 1 锚孤立）
    if anchor_count == 4 and strength >= 1.0 - _FLOAT_ZERO_TOL:
        return [
            [-1.0, 0.0],
            [-0.33, 0.0],
            [0.33, 0.0],
            [0.0, 1.5],
        ]

    # K0 特殊处理：4 锚对称矩形四角
    if anchor_count == 4 and strength <= _FLOAT_ZERO_TOL:
        return [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]

    # 中间档：椭圆分布，strength 越大短轴越短（越退化）
    minor_radius = 1.0 - 0.85 * strength
    if minor_radius < 0.1:
        minor_radius = 0.1
    anchor_positions = []
    for index in range(anchor_count):
        angle = 2.0 * math.pi * index / anchor_count
        x_coord = round(math.cos(angle), 6) + 0.0
        y_coord = round(minor_radius * math.sin(angle), 6) + 0.0
        anchor_positions.append([x_coord, y_coord])
    return anchor_positions


def _compute_geom_score(anchor_positions: Sequence[Sequence[float]]) -> float:
    """根据锚点分布计算几何评分。

    通过协方差矩阵的特征值比衡量锚点分布的均匀程度：
    - geom_score = min_eigenvalue / max_eigenvalue
    - 越接近 1 表示分布越均匀（圆），越接近 0 表示越偏向一个方向（直线）。
    - 少于 2 个锚点时评分为 0，因为无法形成有效几何分布。

    本函数只负责对输入坐标做有限性校验（拒绝 NaN/Inf），核心评分计算
    委托给 ``liquidloc.sensors.anchor_model._compute_geometry_score_core``，
    与 sensors 层 ``compute_geometry_report`` 共享同一实现，避免两侧口径漂移。
    方差/协方差使用总体方差公式（除以 n）、特征值比、clamp 与 round 等口径
    均由共享内核统一定义。

    Args:
        anchor_positions: 锚点坐标列表，每个坐标为二维序列 [x, y]。

    Returns:
        float: 几何评分，范围 [0.0, 1.0]。

    Raises:
        ValueError: 坐标包含 NaN/Inf 时抛出。
    """
    # 校验坐标有限性，避免 NaN/Inf 静默传播破坏协方差计算。
    # 校验后转为 (x, y) 浮点元组列表，与共享内核的输入契约对齐。
    normalized_positions: list[tuple[float, float]] = []
    for position in anchor_positions:  # 逐个校验坐标有限性，与共享内核输入契约对齐。
        x_coord = coerce_finite_scalar(position[0], name="anchor x_coord")  # NaN/Inf 会破坏后续协方差计算。
        y_coord = coerce_finite_scalar(position[1], name="anchor y_coord")
        normalized_positions.append((x_coord, y_coord))
    # 委托给 sensors 层共享内核，保证与 compute_geometry_report 口径一致。
    return _compute_geometry_score_core(normalized_positions)


def _layout_centroid(anchor_positions: Sequence[Sequence[float]]) -> tuple[float, float]:
    """计算锚点布局的质心坐标。

    质心是所有锚点坐标的算术平均，用于后续的尺度对齐和
    投影操作。

    Args:
        anchor_positions: 锚点坐标列表。

    Returns:
        tuple[float, float]: 质心坐标 (x, y)。无锚点时返回 (0.0, 0.0)。

    Raises:
        ValueError: 坐标包含 NaN/Inf 时抛出。
    """
    if not anchor_positions:  # 没有点时质心默认是原点。
        return 0.0, 0.0
    xs = []  # 收集 x 坐标。
    ys = []  # 收集 y 坐标。
    for position in anchor_positions:  # 逐个校验坐标有限性，与 _compute_geom_score/_layout_rms_radius 口径对齐。
        x_coord = coerce_finite_scalar(position[0], name="anchor x_coord")  # NaN/Inf 会破坏后续质心计算。
        y_coord = coerce_finite_scalar(position[1], name="anchor y_coord")
        xs.append(x_coord)
        ys.append(y_coord)
    return math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)  # 返回平均位置，用 fsum 提高精度。


def _layout_rms_radius(
    anchor_positions: Sequence[Sequence[float]],
    *,
    centroid_xy: tuple[float, float],
) -> float:
    """计算锚点到质心的均方根半径。

    RMS 半径衡量布局的整体尺度，用于在不同布局之间做尺度归一化。
    计算方式：先求每个锚点到质心的平方距离，再取均值后开方。

    Args:
        anchor_positions: 锚点坐标列表。
        centroid_xy: 质心坐标 (x, y)。

    Returns:
        float: RMS 半径。无锚点时返回 0.0。

    Raises:
        ValueError: 坐标包含 NaN/Inf 时抛出。
    """
    if not anchor_positions:  # 没有点时半径为 0。
        return 0.0
    cx = coerce_finite_scalar(centroid_xy[0], name="centroid_xy x")  # 质心 NaN/Inf 会破坏后续半径计算，必须显式拒绝。
    cy = coerce_finite_scalar(centroid_xy[1], name="centroid_xy y")
    radius_sq = []  # 收集每个点到质心的平方距离。
    for position in anchor_positions:  # 逐个校验坐标有限性，避免 NaN/Inf 静默传播破坏半径计算。
        x_coord = coerce_finite_scalar(position[0], name="anchor x_coord")  # NaN/Inf 会破坏后续半径计算。
        y_coord = coerce_finite_scalar(position[1], name="anchor y_coord")
        dx = x_coord - cx
        dy = y_coord - cy
        radius_sq.append(dx * dx + dy * dy)
    return math.sqrt(max(0.0, math.fsum(radius_sq) / len(radius_sq)))  # max(0.0, ...) 防御浮点误差导致负数开方，用 fsum 提高精度。


def build_anchor_layout(
    anchor_count: int,
    geometry_level: Any,
    geometry_cfg = None,
    *,
    workspace_span_m: float | None = None,
):
    """构建锚点布局和对应的几何报告。

    处理流程：
    1. 校验输入参数（anchor_count >= 3，geometry_cfg/level 合法）
    2. 解析 geom_condition：
       - 若 geometry_level 是字符串且在 geometry_cfg 中匹配到 level，提取该 level 的 geom_condition
       - 若 geometry_level 是数值（int/float），直接作为 geom_condition 使用（2026-08-31 新协议 K 轴格式）
       - 若 geometry_cfg 是 Mapping 且顶层有 geom_condition 键，直接读取
    3. 将 geom_condition 归一化为布局强度（strength）
    4. 根据锚点数量和强度生成锚点坐标
    5. 可选：按 workspace_span_m 缩放到与轨迹 L_xy 同量级（§8.1/§8.2.1）
    6. 计算几何评分
    7. 组装布局对象和几何报告

    Args:
        anchor_count: 锚点数量，必须 >= 3 的整数（与 K 轴协议 anchor_count >= 3 对齐）。
        geometry_level: 几何条件。可以是：
            - str（如 "K0"、"K1"、"K3"，五轴协议 K 轴档位）：从 geometry_cfg 中按 level 名查找 geom_condition
            - 数值（int/float）：直接作为 geom_condition 使用
        geometry_cfg: 包含几何等级参数的配置映射，可为 None（数值模式时不需要）。
        workspace_span_m: 可选工作空间平面跨度 (m)。

    Returns:
        tuple[dict, dict]: (anchor_layout, geometry_report) 二元组。
        - anchor_layout: 布局对象，包含 anchor_ids、anchor_positions、
          anchor_count 和 layout_id。
        - geometry_report: 几何报告，包含 anchor_count、layout_id、
          geom_score、geom_condition、layout_strength、geometry_level、
          gdop_value、gdop_floor、gdop_above_floor_ratio、
          consistency_checks 和 protocol_consistent。

    Raises:
        TypeError: anchor_count 不是整数、geometry_level 不是字符串或 geometry_cfg 不是映射时抛出。
        ValueError: anchor_count < 3、geometry_level 为空或 geom_condition 不满足协议约束时抛出。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "anchor_count": anchor_count,
        "geometry_level": geometry_level,
        "geom_condition": (geometry_cfg.get("geom_condition") if isinstance(geometry_cfg, Mapping) else None),
        "workspace_span_m": workspace_span_m,
    }, "build_anchor_layout 入口参数")
    if not is_integer(anchor_count):  # 锚点数量必须是整数。
        raise TypeError("anchor_count must be an integer")
    anchor_count = int(anchor_count)  # 确保 Python 原生 int，防止 np.int64 泄漏。
    if anchor_count < 3:  # 与 K 轴协议 anchor_count >= 3 对齐（scene_axis_protocol.py L265-266），2D 定位至少需要 3 个锚点。
        raise ValueError(f"anchor_count must be >= 3 for 2D localization, got {anchor_count}")

    # 解析 geom_condition：支持 3 种模式
    if isinstance(geometry_level, (int, float)) and not isinstance(geometry_level, bool):
        # 模式 1：数值直接作为 geom_condition（2026-08-31 新协议 K 轴格式）
        geom_condition = float(geometry_level)
        normalized_level = f"GC{geom_condition}"
        min_condition = 1.0
        max_condition = 10.0
        axis_strength = (geom_condition - min_condition) / (max_condition - min_condition)  # 归一化到 0-1
    elif is_string_like(geometry_level) and str(geometry_level).strip():
        # 模式 2：字符串从 geometry_cfg 中查找 level
        if not isinstance(geometry_cfg, Mapping):
            raise TypeError(f"geometry_cfg must be a mapping for string-level mode, got {type(geometry_cfg).__name__}")
        normalized_level, geom_condition, min_condition, max_condition = _load_geometry_template(geometry_level, geometry_cfg)  # 读取模板并拿到归一化边界。
        if math.isclose(min_condition, max_condition, rel_tol=0.0, abs_tol=_FLOAT_ZERO_TOL):  # 所有 level 条件相同，说明无法做归一化映射。用显式容差替代默认相对容差，避免大数值时的相对容差误判。
            axis_strength = 0.0
        else:
            axis_strength = (geom_condition - min_condition) / (max_condition - min_condition)  # 归一化到 0-1，保持和 sweep strength 语义一致：越大越差。
    elif isinstance(geometry_cfg, Mapping) and "geom_condition" in geometry_cfg:
        # 模式 3：geometry_cfg 顶层直接有 geom_condition 键（简化配置）
        geom_condition = sample_axis_interval(geometry_cfg["geom_condition"], rng=None, name="geometry_cfg.geom_condition")  # 2026-09-03：支持 [min,max] 区间格式。
        geom_condition = float(geom_condition)
        normalized_level = f"GC{geom_condition}"
        min_condition = 1.0
        max_condition = 10.0
        axis_strength = (geom_condition - min_condition) / (max_condition - min_condition)
    else:
        raise TypeError(f"geometry_level must be a string level name or numeric geom_condition, got {geometry_level!r} with geometry_cfg type {type(geometry_cfg).__name__}")
    layout_uniformity = axis_strength  # K 轴新协议：axis_strength 越大=越退化，直接传给 _build_anchor_positions（已按新语义实现 K0/K3 特殊模板）
    anchor_positions = _build_anchor_positions(anchor_count, layout_uniformity)  # 生成锚点坐标。
    # §8.1/§8.2.1：单位圆布局默认直径 ~2m；主表 L_xy~20m 时必须缩放到同量级基线。
    workspace_scale = 1.0
    if workspace_span_m is not None:
        span = coerce_finite_scalar(workspace_span_m, name="workspace_span_m", min_value=1e-3, inclusive=False)
        # 单位布局外接跨度约 2.0；目标基线 ≈ workspace_span（与 L_xy 同量级）。
        unit_span = 2.0
        workspace_scale = float(span) / unit_span
        anchor_positions = [
            [round(float(p[0]) * workspace_scale, 6) + 0.0, round(float(p[1]) * workspace_scale, 6) + 0.0]
            for p in anchor_positions
        ]
    geom_score = _compute_geom_score(anchor_positions)  # 计算几何评分。
    layout_id = f"{normalized_level}_K{anchor_count}"  # 用 level + 锚点数拼出布局标识。

    # V1 修复（§4.2.2 GDOP 占比显式断言）：从协议顶层读取 gdop_floor 阈值，按
    # `前提指导.md:894` "差 GDOP、病态观测时段达到足够占比" + `前提指导.md:1410`
    # "欠定/临界 GDOP 时段足够" 字面要求，把 geom_condition 当作 GDOP 量级（协议
    # `configs/base/scene_axis_protocol.yaml` 注释已固化此语义："数值越大 GDOP 越大"），
    # 在单布局层显式暴露 `gdop_value` + `gdop_floor` + `gdop_above_floor_ratio`，
    # 让评估层可时序统计 "差 GDOP 时段占比" 以满足 §4.2.2 显式守门。
    gdop_floor_raw = geometry_cfg.get("gdop_floor") if isinstance(geometry_cfg, Mapping) else None
    if gdop_floor_raw is None:
        # 协议未显式给 floor 时取 1.5：K0=1.0 < floor 表示 K0 好几何，K3=3.5 ≥ floor
        # 表示 K3 差 GDOP，与「欠定/差 GDOP 须落到具体退化族」的语义对齐。
        gdop_floor = 1.5
    else:
        gdop_floor = float(coerce_finite_scalar(
            gdop_floor_raw, name="gdop_floor", min_value=0.0, inclusive=True,
        ))
    gdop_value = float(geom_condition)  # 协议注释固化：geom_condition 即 GDOP 量级。
    gdop_above_floor_ratio = 1.0 if gdop_value >= gdop_floor else 0.0  # 单布局层取 0/1；评估层按轨迹时序聚合。
    gdop_occupancy_meets_floor = gdop_above_floor_ratio >= 1.0  # 单布局层显式断言：本布局是否落入"差 GDOP"族。

    anchor_layout = {  # 布局对象本体，供下游直接使用。
        "anchor_ids": [f"A{index}" for index in range(anchor_count)],  # 每个锚点的稳定编号。
        "anchor_positions": anchor_positions,  # 锚点坐标列表。
        "anchor_count": anchor_count,  # 锚点总数。
        "layout_id": layout_id,  # 布局标识。
        "workspace_span_m": None if workspace_span_m is None else float(workspace_span_m),
        "workspace_scale": float(workspace_scale),
    }
    # 一致性检查：与 A/N/V 轴报告口径对齐，记录实际执行的一致性校验结果，而非硬编码 True。
    # 这些检查是后置断言（post-condition assertions）：上游 _normalize_geometry_levels 已保证
    # geom_condition > 0，build_anchor_layout 已保证 anchor_count >= 3，_compute_geom_score 已保证
    # geom_score ∈ [0, 1]。此处显式记录断言结果，供审计日志追溯，而非独立于上游的二次校验。
    # V1 修复：补 `gdop_floor_loaded` + `gdop_occupancy_meets_floor` 两条 GDOP 占比断言，把
    # §4.2.2 "差 GDOP 时段达到足够占比" 字面守门落实到 consistency_checks 显式字段。
    consistency_checks = {
        "geom_condition_positive": geom_condition > 0.0,  # 协议 G 轴约束：geom_condition 必须严格正。
        "anchor_count_sufficient": anchor_count >= 3,  # 协议 K 轴约束：2D 定位至少需要 3 个锚点。
        "geom_score_in_range": 0.0 <= geom_score <= 1.0,  # 几何评分输出范围约束。
        "gdop_floor_loaded": isinstance(gdop_floor, float) and gdop_floor >= 0.0,  # §4.2.2 GDOP 阈值已读入并归一化为非负浮点。
        "gdop_occupancy_meets_floor": gdop_occupancy_meets_floor,  # §4.2.2 本布局是否落入"差 GDOP"族（评估层按轨迹聚合占比）。
    }
    protocol_consistent = all(consistency_checks.values())  # 全部一致才算协议一致。
    geometry_report = {  # 几何报告，供测试和日志记录。
        "geometry_level": normalized_level,  # 几何等级名（已 strip 归一化，与 A/N/V 轴报告口径对齐）。
        "anchor_count": anchor_count,  # 锚点数量。
        "layout_id": layout_id,  # 布局标识。
        "geom_score": geom_score,  # 几何评分。
        "geom_condition": geom_condition,  # 原始几何条件。
        "layout_strength": round(axis_strength, 6),  # 对外保持和 G 轴 sweep strength 一致：越大表示几何压力越高。
        "workspace_span_m": None if workspace_span_m is None else float(workspace_span_m),
        "workspace_scale": float(workspace_scale),
        "gdop_value": gdop_value,  # V1 修复：GDOP 量级（= geom_condition，协议注释固化语义）。
        "gdop_floor": gdop_floor,  # V1 修复：GDOP 差几何阈值，供评估层时序聚合 "差 GDOP 占比"。
        "gdop_above_floor_ratio": gdop_above_floor_ratio,  # V1 修复：本布局 GDOP > floor 的占比（单布局层 0/1）。
        "consistency_checks": consistency_checks,  # 一致性检查明细。
        "protocol_consistent": protocol_consistent,  # 协议一致性总判定，由 consistency_checks 派生。
    }
    return anchor_layout, geometry_report  # 返回布局和报告两个对象。


def project_anchor_layout_to_reference(anchor_layout, reference_layout):
    """把一个合成布局投影到参考布局的尺度和质心上。

    投影过程：
    1. 分别计算待投影布局和参考布局的质心和 RMS 半径
    2. 计算尺度缩放因子 = reference_radius / projected_radius
    3. 对每个锚点：先去中心化（减去待投影质心），再缩放，再平移到参考质心
    4. 退化布局（半径接近 0）时所有点收敛到参考质心

    这个操作保证不同几何等级之间的比较只反映形状差异，
    不受尺度和位置影响。

    Args:
        anchor_layout: 待投影的合成布局，必须包含 anchor_positions。
        reference_layout: 参考布局，必须包含 anchor_positions。

    Returns:
        dict: 投影后的布局对象，新增 reference_layout_id、
            reference_anchor_count、reference_centroid_xy 和
            scale_to_reference 等元数据字段。

    Raises:
        TypeError: anchor_layout 或 reference_layout 不是映射类型时抛出。
        与 validate_anchor_layout 相同（布局不合法时抛出）。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "anchor_layout_type": type(anchor_layout).__name__,
        "reference_layout_type": type(reference_layout).__name__,
    }, "project_anchor_layout_to_reference 入口参数")
    # 输入必须是映射类型：后续代码使用 dict 下标赋值（projected_layout["anchor_positions"] = ...），
    # 非 Mapping 输入（如对象）无法通过下标赋值，会触发 TypeError。显式校验避免下游隐式失败。
    if not isinstance(anchor_layout, Mapping):
        raise TypeError(f"anchor_layout must be a mapping, got {type(anchor_layout).__name__}")
    if not isinstance(reference_layout, Mapping):
        raise TypeError(f"reference_layout must be a mapping, got {type(reference_layout).__name__}")
    # 用浅拷贝替代 deepcopy：anchor_layout 是 dict，只有顶层字段会被修改，
    # anchor_positions 会被替换为新列表，不需要深拷贝原始坐标。
    projected_layout = dict(anchor_layout)
    # reference_layout 同样用浅拷贝优化 dict 输入；validate_anchor_layout 对 dict 输入内部已做浅拷贝，不会修改原对象。
    reference_layout_copy = dict(reference_layout)
    # 显式物化 anchor_positions 为 list：validate_anchor_layout 内部会调用 list(anchor_positions)
    # 消耗生成器，若不预先物化，后续 projected_layout["anchor_positions"] 会得到空生成器。
    projected_layout["anchor_positions"] = list(projected_layout["anchor_positions"])
    reference_layout_copy["anchor_positions"] = list(reference_layout_copy["anchor_positions"])
    validate_anchor_layout(projected_layout)  # 先校验投影对象本身是合法的。
    validate_anchor_layout(reference_layout_copy)  # 再校验参考布局也合法（使用副本）。

    projected_positions = list(projected_layout["anchor_positions"])  # 取出待投影坐标。
    reference_positions = list(reference_layout_copy["anchor_positions"])  # 取出参考坐标。
    projected_centroid = _layout_centroid(projected_positions)  # 计算待投影布局质心。
    reference_centroid = _layout_centroid(reference_positions)  # 计算参考布局质心。
    projected_radius = _layout_rms_radius(projected_positions, centroid_xy=projected_centroid)  # 计算待投影布局半径。
    reference_radius = _layout_rms_radius(reference_positions, centroid_xy=reference_centroid)  # 计算参考布局半径。
    if projected_radius > _DEGENERATE_RADIUS_TOL and reference_radius > _DEGENERATE_RADIUS_TOL:  # 两边都不是退化布局时才做尺度对齐。用常量替代硬编码 1e-9。
        scale_to_reference = reference_radius / projected_radius
        # 防御性 clamp：避免退化布局导致 scale 爆炸到极大值。
        scale_to_reference = min(scale_to_reference, _SCALE_CLAMP_UPPER)
    else:
        scale_to_reference = 0.0  # 退化布局直接投到参考质心上。

    aligned_positions: list[list[float]] = []  # 这里收集对齐后的坐标。
    for x_coord, y_coord in projected_positions:  # 逐个点做平移缩放。
        if scale_to_reference <= _FLOAT_ZERO_TOL:  # 退化时所有点都收敛到参考质心。用 <= 容差替代 == 0.0 避免浮点等比误判。
            aligned_x = float(reference_centroid[0])
            aligned_y = float(reference_centroid[1])
        else:
            aligned_x = float(reference_centroid[0]) + (float(x_coord) - float(projected_centroid[0])) * scale_to_reference  # x 方向先去中心再缩放。
            aligned_y = float(reference_centroid[1]) + (float(y_coord) - float(projected_centroid[1])) * scale_to_reference  # y 方向同理。
        aligned_positions.append([round(aligned_x, 6) + 0.0, round(aligned_y, 6) + 0.0])  # 保留 6 位小数，加 0.0 消除 -0.0 负零。

    projected_layout["anchor_positions"] = aligned_positions  # 把新坐标写回布局。
    projected_layout["reference_layout_id"] = reference_layout_copy.get("layout_id")  # 记录参考布局标识，从副本读取，与"使用副本"的意图一致（validate_anchor_layout 不修改 layout_id，但统一从副本读取避免歧义）。
    projected_layout["reference_anchor_count"] = len(reference_positions)  # 记录参考锚点数。
    projected_layout["reference_centroid_xy"] = [
        round(float(reference_centroid[0]), 6) + 0.0,
        round(float(reference_centroid[1]), 6) + 0.0,
    ]  # 记录参考质心，方便调试和审计，加 0.0 消除 -0.0 负零。
    projected_layout["scale_to_reference"] = round(float(scale_to_reference), 6) + 0.0  # 记录实际缩放比例，加 0.0 消除 -0.0 负零，与 aligned_positions 负零归一化口径对齐。
    return projected_layout  # 返回投影后的布局对象。
