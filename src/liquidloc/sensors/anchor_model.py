"""锚点几何与查表工具模块。

这个模块专门处理 anchor 布局的校验、归一化、查表构建和几何质量估计。
它本身不做定位，也不直接参与滤波，只是把上层传进来的 anchor 布局整理成统一格式，
供 UWB 预测、几何评分和实验日志复用。

核心数据流：
    anchor_layout → validate_anchor_layout（纯校验）
    anchor_layout → build_anchor_lookup → {anchor_id: (x, y)}
    anchor_layout → compute_geometry_report → {anchor_count, layout_id, geom_score}

上游依赖：
    无外部模块依赖（仅用 math 标准库）。

下游调用者：
    liquidloc.sensors.__init__（对外导出 validate_anchor_layout, build_anchor_lookup, compute_geometry_report）
    liquidloc.dataio.sim_materializer（build_anchor_lookup）
    liquidloc.pipelines.core_pipeline（build_anchor_lookup）
    liquidloc.pipelines.train_pipeline（build_anchor_lookup）
    liquidloc.estimators.ekf_core（build_anchor_lookup）
    liquidloc.scenarios.geometry_levels（validate_anchor_layout）
    liquidloc.models.features.feature_builder（compute_geometry_report）

关键设计决策：
    - _normalize_anchor_layout 提供 writeback 参数控制是否写回归一化结果：
      writeback=True（默认）时对字典输入返回归一化后的副本（不修改原字典），
      对对象输入会尝试 setattr，失败则发出 RuntimeWarning；
      writeback=False 时完全跳过写回，适用于纯校验/纯查询场景（validate_anchor_layout、build_anchor_lookup）。
    - 方差和协方差使用总体方差公式（除以 n），而非样本方差（除以 n-1），
      因为所有 anchor 构成总体而非样本。
    - 几何评分 geom_score = min_eigenvalue / max_eigenvalue，越接近 1 表示分布越均匀。
    - 坐标必须是有限实数，显式拒绝布尔值，避免把 True/False 静默当成 1.0/0.0。
"""

import math  # 几何评分里要用到方差、特征值和有限性检查。
import warnings  # 对象写回失败时发出 RuntimeWarning。
from typing import Any  # 用于宽松类型标注。

from liquidloc.common.validation import coerce_finite_scalar, is_bool_like, is_integer  # 集中判断 bool / np.bool_ 和整数类型，以及有限浮点转换。


def _get_anchor_layout_fields(anchor_layout) -> tuple[Any, Any, Any]:
    """从 dict 或对象里读取 anchor 布局的基础字段。

    参数：
        anchor_layout: 字典或对象，应包含 anchor_positions、anchor_ids 和 layout_id。

    返回：
        (anchor_positions, anchor_ids, layout_id)。字段缺失时返回 None，
        不做任何规范化，由 _normalize_anchor_layout 统一校验。

    注意：
        getattr 对没有对应属性的对象返回 None，而非 AttributeError。
        后续 _normalize_anchor_layout 的 None 检查会捕获此情况。
    """
    if isinstance(anchor_layout, dict):  # 先兼容最常见的字典输入。
        anchor_positions = anchor_layout.get("anchor_positions")  # 提取坐标列表，后续要做长度和数值校验。
        anchor_ids = anchor_layout.get("anchor_ids")  # 提取 id 列表，后续要构建查表。
        layout_id = anchor_layout.get("layout_id")  # 布局标识只用于报告，不影响计算。
    else:  # 如果不是字典，就按对象属性读取。
        anchor_positions = getattr(anchor_layout, "anchor_positions", None)  # 对象没这个属性时返回 None，统一交给后续报错。
        anchor_ids = getattr(anchor_layout, "anchor_ids", None)  # 同上，id 列表也走属性读取。
        layout_id = getattr(anchor_layout, "layout_id", None)  # 布局标识同样允许缺省。
    return anchor_positions, anchor_ids, layout_id  # 这里只做字段读取，不做任何规范化。


def _store_normalized_anchor_layout(anchor_layout, anchor_positions, anchor_ids) -> None:
    """把归一化后的 anchor 布局写回原对象。

    参数：
        anchor_layout: 字典或对象。
        anchor_positions: 归一化后的坐标列表。
        anchor_ids: 规范化后的 id 列表。

    副作用：
        对字典输入原地修改 anchor_positions 和 anchor_ids 字段。
        对对象输入尝试 setattr，失败则发出 RuntimeWarning。
        若第一个 setattr 成功而第二个失败，会尝试回滚已修改的属性，
        避免对象处于部分更新的不一致状态。
    """
    if isinstance(anchor_layout, dict):  # 字典最容易原地回写。
        anchor_layout["anchor_positions"] = anchor_positions  # 用归一化后的 2D 浮点坐标覆盖原值。
        anchor_layout["anchor_ids"] = anchor_ids  # 用规范化后的 id 列表覆盖原值。
        return  # 字典已经写回完毕，直接结束。

    # 保存原始值以便回滚，避免第一个 setattr 成功而第二个失败时对象处于部分更新状态。
    original_positions = getattr(anchor_layout, "anchor_positions", None)
    original_ids = getattr(anchor_layout, "anchor_ids", None)
    try:  # 对象型输入也尽量回写，方便调用方复用。
        setattr(anchor_layout, "anchor_positions", anchor_positions)  # 这里要求对象允许设置同名属性。
        setattr(anchor_layout, "anchor_ids", anchor_ids)  # 同理，把 id 列表也写回去。
    except (AttributeError, TypeError):  # 如果对象不支持写属性，尝试回滚后发出警告。
        try:  # 回滚已修改的属性，避免对象处于部分更新的不一致状态。
            setattr(anchor_layout, "anchor_positions", original_positions)
            setattr(anchor_layout, "anchor_ids", original_ids)
        except (AttributeError, TypeError):
            pass  # 回滚失败，对象可能仍处于部分修改状态，警告已提示调用方使用返回值。
        warnings.warn(
            f"Cannot write normalized anchor layout back to {type(anchor_layout).__name__}; "
            "caller must use the returned values directly.",
            RuntimeWarning,
            stacklevel=4,
        )


def _coerce_anchor_id(aid: Any) -> Any:
    """将 numpy 标量 id 转为 Python 原生类型，防止 np.int64/np.float64 泄漏到下游查表键。

    只做类型归一化，不做有限性或数值范围校验：
    - numpy 整数标量 -> Python int
    - numpy 浮点标量 -> Python float
    - 其余类型（含 bool、str、Python 原生数值）保持原样

    注意：bool 和 numpy.bool_ 不在此函数处理，因为 is_integer 已先排除 bool。
    """
    if is_bool_like(aid):
        return aid
    if is_integer(aid):
        return int(aid)
    try:
        import numpy as np
        if isinstance(aid, np.floating):
            return float(aid)
    except ImportError:
        pass
    return aid


def _normalize_anchor_layout(anchor_layout, writeback: bool = True) -> tuple[list[tuple[float, float]], list[Any], Any]:
    """把 anchor 布局标准化成可直接使用的 Python 结构。

    处理流程：
        1. 读取 anchor_positions、anchor_ids、layout_id
        2. 校验 positions/ids 不为 None、不是字符串
        3. 校验数量 >= 1、ids 与 positions 长度一致
        4. 校验 ids 唯一且可哈希
        5. 逐个校验坐标：2D、可转 float、有限
        6. 当 writeback=True 时，将归一化结果写回原对象（副作用）
        7. 返回归一化后的结构

    副作用：
        对字典输入不修改原字典（函数开头做了浅拷贝），归一化结果写入副本。
        对对象输入会尝试 setattr，失败则发出 RuntimeWarning。
        当 writeback=False 时，完全跳过写回步骤，函数退化为纯校验+归一化返回。

    参数：
        anchor_layout: 字典或对象，必须提供 anchor_positions 和 anchor_ids。
        writeback: 是否将归一化结果写回原对象，默认 True 保持向后兼容。
            纯校验场景（如 validate_anchor_layout）应传 False，避免副作用。

    返回：
        (normalized_positions, anchor_ids, layout_id)
        - normalized_positions: list of (x, y) 浮点元组
        - anchor_ids: list of hashable id
        - layout_id: 任意类型，可为 None

    异常：
        ValueError: 字段缺失、数量非法、id 重复、坐标维度/值非法。
        TypeError: 字段类型非法（如字符串）、id 不可哈希、坐标非数值。
    """
    anchor_layout = dict(anchor_layout) if isinstance(anchor_layout, dict) else anchor_layout
    anchor_positions, anchor_ids, layout_id = _get_anchor_layout_fields(anchor_layout)  # 先读原始字段，再统一校验。
    if anchor_positions is None:  # 坐标缺失时不能继续，因为后续所有几何量都依赖它。
        raise ValueError("anchor_layout must provide anchor_positions")
    if anchor_ids is None:  # id 缺失时不能构建查表。
        raise ValueError("anchor_layout must provide anchor_ids")
    if isinstance(anchor_positions, (str, bytes)):  # 字符串不是坐标集合。
        raise TypeError("anchor_positions must be a coordinate collection")
    if isinstance(anchor_ids, (str, bytes)):  # 字符串也不是合法 id 集合。
        raise TypeError("anchor_ids must be a collection of ids")

    anchor_positions = list(anchor_positions)  # 先转成列表，后面需要做长度、重复和逐项数值校验。
    anchor_ids = list(anchor_ids)  # id 也统一成列表，方便统计和 zip 配对。
    # 将 numpy 标量转为 Python 原生类型，防止 np.int64/np.float64 泄漏到下游查表键。
    anchor_ids = [_coerce_anchor_id(aid) for aid in anchor_ids]
    anchor_count = len(anchor_positions)  # anchor 总数会影响后续的几何评分。
    if anchor_count < 1:  # 没有 anchor 时布局没有意义。
        raise ValueError("anchor_count must be >= 1")
    if len(anchor_ids) != anchor_count:  # id 数量和坐标数量必须一一对应。
        raise ValueError("anchor_positions and anchor_ids must have the same length")

    try:  # id 必须可哈希，才能构建 lookup。
        if len(set(anchor_ids)) != anchor_count:  # 去重后数量变化说明 id 重复。
            raise ValueError("anchor_ids must be unique")
    except TypeError as exc:  # 不可哈希的 id 也不能用于查表。
        raise TypeError("anchor_ids must be hashable") from exc

    normalized_positions = []  # 这里收集最终规范化后的 2D 坐标。
    for position in anchor_positions:  # 逐个 anchor 校验坐标形状。
        if isinstance(position, (str, bytes)):  # 字符串不是合法坐标容器。
            raise TypeError("each anchor position must be a 2D coordinate")
        coords = list(position)  # 先展开成列表，便于检查维度。
        if len(coords) != 2:  # anchor 必须是二维平面坐标。
            raise ValueError("each anchor position must contain exactly two coordinates")
        if is_bool_like(coords[0]) or is_bool_like(coords[1]):  # 布尔坐标会把配置错误伪装成 0/1 数值。
            raise TypeError("anchor coordinates must be numeric")
        x_coord = coerce_finite_scalar(coords[0], name="anchor x_coord")  # 无穷大和 NaN 都不接受。
        y_coord = coerce_finite_scalar(coords[1], name="anchor y_coord")
        normalized_positions.append((x_coord, y_coord))  # 统一存成二维浮点元组，避免后续再猜类型。

    if writeback:  # 仅在需要写回时才触发副作用，纯校验场景（validate_anchor_layout）传 False。
        _store_normalized_anchor_layout(anchor_layout, normalized_positions, anchor_ids)  # 归一化结果尽量写回原对象。
    return normalized_positions, anchor_ids, layout_id  # 返回后续查表和报告都能直接用的规范结构。


def validate_anchor_layout(anchor_layout) -> None:
    """只校验 anchor 布局是否满足最小协议要求。

    此函数是纯校验函数，不修改原输入对象（无论字典还是对象输入）。
    内部通过 writeback=False 调用 _normalize_anchor_layout，跳过写回副作用。

    参数：
        anchor_layout: 字典或对象，必须提供 anchor_positions 和 anchor_ids。

    异常：
        与 _normalize_anchor_layout 相同。

    注意：
        此函数不修改原输入：对字典输入，_normalize_anchor_layout 做了浅拷贝；
        对对象输入，通过 writeback=False 跳过 setattr 写回，原对象保持不变。
    """
    _normalize_anchor_layout(anchor_layout, writeback=False)  # 纯校验，不触发写回副作用。


def build_anchor_lookup(anchor_layout) -> dict[Any, tuple[float, float]]:
    """把 anchor 布局转换成 id -> position 的查找表。

    此函数是纯查询函数，不修改原输入对象（无论字典还是对象输入）。
    内部通过 writeback=False 调用 _normalize_anchor_layout，跳过写回副作用。

    参数：
        anchor_layout: 字典或对象，必须提供 anchor_positions 和 anchor_ids。

    返回：
        ``{anchor_id: (x, y)}`` 查找表。anchor_id 的类型由输入决定，
        坐标保证为有限浮点元组。

    异常：
        与 _normalize_anchor_layout 相同。

    注意：
        此函数不修改原输入：对字典输入，_normalize_anchor_layout 做了浅拷贝；
        对对象输入，通过 writeback=False 跳过 setattr 写回，原对象保持不变。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "anchor_count": (anchor_layout.get("anchor_count") if isinstance(anchor_layout, dict) else None),
        "layout_id": (anchor_layout.get("layout_id") if isinstance(anchor_layout, dict) else None),
    }, "build_anchor_lookup 入口参数")
    anchor_positions, anchor_ids, _ = _normalize_anchor_layout(anchor_layout, writeback=False)  # 纯查询，不触发写回副作用。
    anchor_lookup = {}  # 输出查表用字典。
    for anchor_id, position in zip(anchor_ids, anchor_positions):  # id 和坐标严格一一对应。
        anchor_lookup[anchor_id] = position  # 这里用 id 作为键，用二维坐标作为值。
    return anchor_lookup  # 返回后续 UWB 预测和日志排查都能直接用的查表结果。


def _compute_geometry_score_core(normalized_positions: list[tuple[float, float]]) -> float:
    """计算几何评分核心逻辑（协方差矩阵特征值比）。

    这是 sensors 层与 scenarios 层共享的几何评分内核，消除两侧重复实现
    导致的口径漂移风险。任何一侧修改本函数都会同时影响
    ``compute_geometry_report``（sensors 层）和
    ``scenarios.geometry_levels._compute_geom_score``（scenarios 层）。

    输入要求：
        normalized_positions: 已校验为有限浮点的 2D 坐标列表。
        调用者负责保证输入坐标的有限性和类型合法性（例如通过
        ``_normalize_anchor_layout`` 或 ``coerce_finite_scalar`` 完成校验）。

    计算口径：
        - 少于 2 个 anchor 时返回 0.0（无法形成有效几何分布）。
        - 方差和协方差使用总体方差公式（除以 n），而非样本方差（除以 n-1），
          因为所有 anchor 构成总体而非样本。
        - geom_score = min_eigenvalue / max_eigenvalue，越接近 1 表示分布越均匀。
        - max_eigenvalue <= 0.0 时返回 0.0（用 <= 0.0 替代 == 0.0 避免浮点等比误判）。
        - 数值误差下 |geom_score - 1.0| < 1e-6 直接归一到 1.0。
        - 防御性 clamp 确保 geom_score ∈ [0, 1]。
        - 保留 6 位小数，方便日志比较。

    参数：
        normalized_positions: 已校验的 2D 坐标列表，每个元素为 (x, y) 浮点元组。

    返回：
        几何评分 ∈ [0.0, 1.0]。
    """
    anchor_count = len(normalized_positions)  # anchor 数量会直接决定是否可以计算特征值。
    if anchor_count < 2:  # 少于 2 个点时无法形成有效几何分布。
        return 0.0  # 这里直接给 0 分，表示没有几何信息。

    xs = [position[0] for position in normalized_positions]  # 单独收集 x 轴坐标做统计。
    ys = [position[1] for position in normalized_positions]  # 单独收集 y 轴坐标做统计。

    mean_x = math.fsum(xs) / anchor_count  # x 轴均值，用 fsum 提高精度。
    mean_y = math.fsum(ys) / anchor_count  # y 轴均值，用 fsum 提高精度。
    var_x = math.fsum((x_coord - mean_x) ** 2 for x_coord in xs) / anchor_count  # x 轴方差。
    var_y = math.fsum((y_coord - mean_y) ** 2 for y_coord in ys) / anchor_count  # y 轴方差。
    cov_xy = math.fsum(  # 协方差反映两个坐标轴是否一起变化，用 fsum 提高精度。
        (x_coord - mean_x) * (y_coord - mean_y)  # 每个点对协方差的贡献。
        for x_coord, y_coord in zip(xs, ys)  # 同步遍历 x/y 坐标。
    ) / anchor_count

    trace = var_x + var_y  # 协方差矩阵的迹。
    delta = math.sqrt(max((var_x - var_y) ** 2 + 4.0 * cov_xy * cov_xy, 0.0))  # 特征值解析式里的判别项。
    max_eigenvalue = (trace + delta) / 2.0  # 较大的特征值。
    min_eigenvalue = max((trace - delta) / 2.0, 0.0)  # 较小的特征值，不允许掉到负数。

    if max_eigenvalue <= 0.0:  # 如果最大特征值非正（浮点误差可能产生极小负值），说明所有点几乎重合。
        return 0.0  # 直接记为 0 分。
    geom_score = min_eigenvalue / max_eigenvalue  # 这里越接近 1，表示分布越均匀。
    if abs(geom_score - 1.0) < 1e-6:  # 数值误差下的 1.0 直接归一回 1.0。
        geom_score = 1.0
    geom_score = min(max(geom_score, 0.0), 1.0)  # 防御性 clamp：确保 geom_score ∈ [0, 1]。
    return round(geom_score, 6)  # 把分数保留到 6 位小数，方便日志比较。


def compute_geometry_report(anchor_layout) -> dict[str, Any]:
    """计算布局的几何报告和粗略几何评分。

    参数：
        anchor_layout: 字典或对象，必须提供 anchor_positions 和 anchor_ids。

    返回：
        ``{"anchor_count": int, "layout_id": Any, "geom_score": float}``。
        - anchor_count: anchor 数量
        - layout_id: 布局标识，可为 None
        - geom_score: 几何评分 ∈ [0.0, 1.0]，
          小于 2 个 anchor 时为 0.0，
          否则 = min_eigenvalue / max_eigenvalue

    异常：
        与 _normalize_anchor_layout 相同。

    注意：
        - 此函数不修改原输入对象（无论字典还是对象输入）。
          对字典输入，_normalize_anchor_layout 做了浅拷贝；
          对对象输入，通过 writeback=False 跳过 setattr 写回，原对象保持不变。
        - 几何评分核心计算委托给 _compute_geometry_score_core，与
          scenarios.geometry_levels._compute_geom_score 共享同一实现，
          避免两侧口径漂移。
        - 方差和协方差使用总体方差公式（除以 n），而非样本方差（除以 n-1），
          因为所有 anchor 构成总体而非样本。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "anchor_count": (anchor_layout.get("anchor_count") if isinstance(anchor_layout, dict) else None),
        "layout_id": (anchor_layout.get("layout_id") if isinstance(anchor_layout, dict) else None),
    }, "compute_geometry_report 入口参数")
    normalized_positions, _, layout_id = _normalize_anchor_layout(anchor_layout, writeback=False)  # 纯计算，不触发写回副作用。
    anchor_count = len(normalized_positions)  # anchor 数量会直接进入报告。
    geom_score = _compute_geometry_score_core(normalized_positions)  # 委托给共享内核，保证与 scenarios 层口径一致。

    geometry_report = {  # 统一输出给上层的几何报告字典。
        "anchor_count": anchor_count,  # anchor 数量。
        "layout_id": layout_id,  # 布局标识。
        "geom_score": geom_score,  # 几何评分。
    }
    return geometry_report  # 返回统一报告结构，方便上层记录和展示。


def compute_anchor_gdop_report(
    anchor_layout,
    *,
    x_min: float = 0.5,
    x_max: float = 19.5,
    y_min: float = 0.5,
    y_max: float = 19.5,
    grid_step: float = 0.5,
) -> dict[str, float]:
    """计算锚点布局在活动区内的 GDOP 统计报告（避免 RMSE 过大准则 §4）。

    参数：
        anchor_layout: 字典或对象，必须提供 anchor_positions。
        x_min/x_max/y_min/y_max: 活动区边界（默认 0.5-19.5m，对应 paper site_xy_m=20m）。
        grid_step: 网格采样步长（米）。

    返回：
        dict，包含 mean_gdop/min_gdop/max_gdop/grid_points_evaluated 字段。
        mean ∈ [3, 5]、max ≤ 6 为合规。
    """
    import math as _math
    normalized_positions, _, _ = _normalize_anchor_layout(anchor_layout, writeback=False)
    if len(normalized_positions) < 3:
        return {"mean_gdop": float("inf"), "min_gdop": float("inf"), "max_gdop": float("inf"),
                "grid_points_evaluated": 0, "compliant": False, "reason": "fewer than 3 anchors"}

    xs = []
    x = x_min
    while x <= x_max + 1e-9:
        xs.append(x)
        x += grid_step
    ys = []
    y = y_min
    while y <= y_max + 1e-9:
        ys.append(y)
        y += grid_step

    anchors_xy = [(float(p[0]), float(p[1])) for p in normalized_positions]
    gdops: list[float] = []
    for gx in xs:
        for gy in ys:
            H: list[list[float]] = []
            for ax, ay in anchors_xy:
                dx = ax - gx
                dy = ay - gy
                d = _math.sqrt(dx * dx + dy * dy)
                if d < 1e-6:
                    continue
                H.append([dx / d, dy / d, 1.0])
            if len(H) < 3:
                continue
            # H^T H
            HtH = [[0.0] * 3 for _ in range(3)]
            for row in H:
                for i in range(3):
                    for j in range(3):
                        HtH[i][j] += row[i] * row[j]
            # 3x3 行列式（用于求逆）
            det = (
                HtH[0][0] * (HtH[1][1] * HtH[2][2] - HtH[1][2] * HtH[2][1])
                - HtH[0][1] * (HtH[1][0] * HtH[2][2] - HtH[1][2] * HtH[2][0])
                + HtH[0][2] * (HtH[1][0] * HtH[2][1] - HtH[1][1] * HtH[2][0])
            )
            if abs(det) < 1e-12:
                continue
            inv00 = (HtH[1][1] * HtH[2][2] - HtH[1][2] * HtH[2][1]) / det
            inv11 = (HtH[0][0] * HtH[2][2] - HtH[0][2] * HtH[2][0]) / det
            inv22 = (HtH[0][0] * HtH[1][1] - HtH[0][1] * HtH[1][0]) / det
            gdop_sq = inv00 + inv11 + inv22
            if gdop_sq > 0:
                gdops.append(_math.sqrt(gdop_sq))
    if not gdops:
        return {"mean_gdop": float("inf"), "min_gdop": float("inf"), "max_gdop": float("inf"),
                "grid_points_evaluated": 0, "compliant": False, "reason": "no valid grid points"}
    mean_g = sum(gdops) / len(gdops)
    max_g = max(gdops)
    min_g = min(gdops)
    compliant = (3.0 <= mean_g <= 5.0) and (max_g <= 6.0)
    return {
        "mean_gdop": float(mean_g),
        "min_gdop": float(min_g),
        "max_gdop": float(max_g),
        "grid_points_evaluated": int(len(gdops)),
        "compliant": bool(compliant),
        "reason": "compliant" if compliant else (
            f"mean {mean_g:.3f} ∉ [3, 5]" if not (3.0 <= mean_g <= 5.0)
            else f"max {max_g:.3f} > 6"
        ),
    }


def project_anchor_layout_xy(anchor_layout_metadata):
    """将 3D 锚点布局投影为 2D（丢弃 z 轴）。

    MILUV 官方数据提供 3D 锚点坐标，但训练和推理使用 2D 平面，
    因此需要把 (x, y, z) 投影为 (x, y)，同时保留投影审计信息。

    参数:
        anchor_layout_metadata: 包含 anchor_ids 和 anchor_positions 的元数据字典。

    返回值:
        投影后的 2D 锚点布局字典，包含投影审计信息；
        输入无效时返回 None。

    注意:
        坐标校验口径与 _normalize_anchor_layout 对齐：显式拒绝布尔值
        （避免 True/False 静默当成 1.0/0.0），要求可转 float 且有限
        （拒绝 NaN/Inf），与模块整体数值安全口径一致。
    """
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "anchor_count": (anchor_layout_metadata.get("anchor_count") if isinstance(anchor_layout_metadata, dict) else None),
        "layout_id": (anchor_layout_metadata.get("layout_id") if isinstance(anchor_layout_metadata, dict) else None),
    }, "project_anchor_layout_xy 入口参数")
    if not isinstance(anchor_layout_metadata, dict):  # 非 dict 输入无法读取字段，统一返回 None。
        return None
    raw_anchor_ids = anchor_layout_metadata.get("anchor_ids")  # 读取原始锚点 ID 列表。
    raw_anchor_positions = anchor_layout_metadata.get("anchor_positions")  # 读取原始 3D 坐标列表。
    if isinstance(raw_anchor_ids, (str, bytes)) or isinstance(raw_anchor_positions, (str, bytes)):  # 字符串/字节类型无法解析。
        return None
    try:
        anchor_ids = list(raw_anchor_ids or [])  # 转成列表。
        anchor_positions_3d = list(raw_anchor_positions or [])  # 转成列表。
    except TypeError:  # 不可迭代时无法处理。
        return None
    if not anchor_ids or len(anchor_ids) != len(anchor_positions_3d):  # ID 和坐标数量必须一致。
        return None

    anchor_positions_2d: list[list[float]] = []  # 收集投影后的 2D 坐标。
    for anchor_position_3d in anchor_positions_3d:  # 逐个锚点投影。
        if isinstance(anchor_position_3d, (str, bytes)):  # 字符串/字节类型无法解析。
            return None
        try:
            coords = list(anchor_position_3d)  # 转成坐标列表。
        except TypeError:  # 不可迭代时无法处理。
            return None
        if len(coords) != 3:  # 必须是 3D 坐标。
            return None
        # 与 _normalize_anchor_layout L195 对齐：显式拒绝布尔坐标，避免 True/False 静默当成 1.0/0.0。
        if is_bool_like(coords[0]) or is_bool_like(coords[1]):  # 布尔坐标会把配置错误伪装成数值。
            return None
        try:  # 坐标必须能转成浮点数，与 _normalize_anchor_layout L197-198 口径对齐。
            x_coord = float(coords[0])  # x 轴坐标。
            y_coord = float(coords[1])  # y 轴坐标。
        except (TypeError, ValueError, OverflowError):  # 非数值或溢出坐标直接判无效。
            return None
        if not math.isfinite(x_coord) or not math.isfinite(y_coord):  # 无穷大和 NaN 不接受，与 _normalize_anchor_layout L197-198 对齐。
            return None
        anchor_positions_2d.append([x_coord, y_coord])  # 只保留 x 和 y，丢弃 z。

    return {  # 返回投影后的 2D 锚点布局，附带投影审计信息。
        "anchor_ids": list(anchor_ids),  # 锚点 ID 列表。
        "anchor_positions": anchor_positions_2d,  # 投影后的 2D 坐标列表。
        "source": anchor_layout_metadata.get("source"),  # 数据来源标识。
        "layout_id": anchor_layout_metadata.get("layout_id"),  # 布局 ID。
        "experiment": anchor_layout_metadata.get("experiment"),  # 实验名称。
        "anchor_constellation": anchor_layout_metadata.get("anchor_constellation"),  # 锚点星座编号。
        "source_paths": list(anchor_layout_metadata.get("source_paths") or []),  # 原始文件路径。
        "original_anchor_position_dim": 3,  # 原始维度为 3D。
        "teacher_anchor_position_dim": 2,  # 投影后为 2D。
        "projection": "xy",  # 投影方式：保留 xy 平面。
        "ignored_axis": "z",  # 被忽略的轴：z。
    }
