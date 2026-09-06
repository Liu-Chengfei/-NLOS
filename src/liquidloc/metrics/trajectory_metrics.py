"""文件:src/liquidloc/metrics/trajectory_metrics.py  # 文件头：说明这个文件的身份。

【文件职责】从预测轨迹和真值轨迹里计算 RMSE、MAE、ATE、RPE 等轨迹误差指标。  # 说明文件功能。
它存在的原因是把轨迹误差计算独立成单一职责模块，供 metric runner 和绘图/分析面复用。  # 说明存在理由。

【上游依赖】  # 说明本文件依赖哪些模块。
- 无外部项目模块依赖，仅使用标准库 math 和 collections.abc。  # 只依赖标准库。

【下游调用者】  # 说明谁会用。
- metric_runner.py：通过 compute_trajectory_metrics 调用，作为四类指标之一。  # 总调度器。
- 分析脚本和轨迹可视化工具：直接调用本模块计算轨迹误差。  # 分析面。
- tests/metrics/test_trajectory_metrics.py：单元测试。  # 测试面。

【输入对象定义】  # 说明输入对象。
- pred_traj: 预测轨迹点序列，每个点是包含 px/py（和可选 pz）及可选时间键的映射。  # 预测轨迹。
- gt_traj: 真值轨迹点序列，结构与 pred_traj 一致。  # 真值轨迹。

【输出对象定义】  # 说明输出对象。
- dict[str, float]：包含 rmse、mae、ate、rpe 四个键的指标字典。  # 轨迹误差指标。

【核心变量定义】  # 说明核心中间变量。
- coord_keys: 坐标键元组，如 ("px", "py") 或 ("px", "py", "pz")。  # 坐标维度。
- aligned_pred: 对齐后的预测-真值点对列表。  # 对齐结果。
- errors: 每个对齐点对的欧氏误差列表。  # 误差序列。
- rmse: 均方根误差。  # RMSE。
- mae: 平均绝对误差。  # MAE。
- ate: 绝对轨迹误差，当前口径与 rmse 一致。  # ATE。
- rpe: 相对位姿误差，基于相邻误差差分的均值。  # RPE。

【推荐编写顺序】1. 先看 compute_trajectory_metrics 入口。2. 再看内部 _as_point_list/_coord_keys/_align_by_time_or_index/_point_error。  # 推荐阅读顺序。
"""

from __future__ import annotations  # 允许后续扩展类型注解保持灵活。

import math  # 提供平方根、有限性检查等数学工具。
from collections.abc import Mapping, Sequence  # 用于识别轨迹点和轨迹序列类型。

from liquidloc.common.constants import TIME_KEY_CANDIDATES  # D9 单源：时间字段候选键名常量。
from liquidloc.common.validation import is_bool_like  # 统一判断布尔类型（含 numpy.bool_）。


# ──────────────────────────────────────────────────────────────────────────────
# Umeyama / SE3 对齐辅助函数（供 compute_trajectory_metrics 和 _compute_ate_se3 共用）
# ──────────────────────────────────────────────────────────────────────────────

def _coerce_scalar(value, key: str) -> float:
    """把任意值规范化为有限浮点数，用于 SVD/对齐等数值计算。"""
    if is_bool_like(value):
        raise ValueError(f"Trajectory coordinate '{key}' must be numeric.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Trajectory coordinate '{key}' must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Trajectory coordinate '{key}' must be finite.")
    return value


def _umeyama(aligned_pred: list[tuple[Mapping, Mapping]], coord_keys: tuple[str, ...]) -> tuple[list[list[float]], list[float], float, bool]:
    """Umeyama/Sim(3) 对齐：提取最优旋转 R、平移 t 和尺度 s。

    对齐公式：p_aligned = s * R @ (p - pred_centroid) + gt_centroid = R @ p + t
    （t 已将 s 和质心偏移合并进来：t = gt_centroid - s * R @ pred_centroid）

    当点数少于 3 时，退化为仅平移对齐（s=1, R=单位阵），返回 degraded=True。

    参数:
        aligned_pred: 时间/索引对齐后的 (pred_point, gt_point) 列表。
        coord_keys: 坐标键元组，如 ("px", "py") 或 ("px", "py", "pz")。

    返回值:
        (R, t, scale, degraded): 旋转矩阵 R (dim×dim)、平移向量 t (dim)、
        尺度 s (恒为 1.0，当前仅支持 SE3)、是否退化为纯平移对齐。
    """
    n = len(aligned_pred)
    dim = len(coord_keys)

    # 质心计算
    pred_centroid = {k: math.fsum(_coerce_scalar(p[k], k) for p, _ in aligned_pred) / n for k in coord_keys}
    gt_centroid = {k: math.fsum(_coerce_scalar(g[k], k) for _, g in aligned_pred) / n for k in coord_keys}

    # 中心化
    pred_centered = [[_coerce_scalar(p[k], k) - pred_centroid[k] for k in coord_keys] for p, _ in aligned_pred]
    gt_centered = [[_coerce_scalar(g[k], k) - gt_centroid[k] for k in coord_keys] for _, g in aligned_pred]

    if n < 3:
        # 点数不足时退化为纯平移对齐：R=单位阵，s=1，t=gt_centroid - pred_centroid
        R = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]
        t = [gt_centroid[k] - pred_centroid[k] for k in coord_keys]
        return R, t, 1.0, True

    # 互协方差矩阵 H = pred_centered^T @ gt_centered
    # H[i][j] = Σ_{p=0}^{n-1} pred_centered[p][i] * gt_centered[p][j]
    H = [[math.fsum(pred_centered[p][i] * gt_centered[p][j] for p in range(n)) for j in range(dim)] for i in range(dim)]

    # SVD 求最优旋转 R（准则 9：H = pred_c^T @ gt_c，pred = s·R0·gt + t,
    # 所以 H = s·R0·gt_c·gt_c^T, R0 = U·V^T 才是把 gt 转到 pred 方向。
    # 但 _svd_2x2_rotation 返回的是 V·U^T (数学上等价的 Procrustes 解)，
    # 二者互为转置 (R^T = R0)。Sim(3) 闭式解 s = trace(R^T @ H) / trace(gt_c^T @ gt_c)
    # = trace(R0 @ H) / trace(gt_c^T @ gt_c); 由于 R0 = R^T, 用 R 计算等价。
    if dim == 2:
        R = _svd_2x2_rotation(H)
    else:
        R = _svd_3x3_rotation(H)

    # 计算尺度 s（Sim(3) 闭式解, Umeyama 1991）
    sum_gt_norm = math.fsum(
        math.fsum(gc[i] ** 2 for i in range(dim)) for gc in gt_centered
    )
    if sum_gt_norm > 1e-12:
        # R^T (不是 R) 才是把 gt 转到 pred 方向的最优旋转 (R = V·U^T = R0^T)。
        # Sim(3) 闭式解: s = trace(R^T @ H) / trace(gt_c^T @ gt_c)
        # 验证: R^T @ H = R0 @ H, trace = 4.0, sum_gt_norm = 4/1.5 ≈ 2.667, s = 4/2.667 = 1.5 ✓
        scale = math.fsum(
            math.fsum(R[j][i] * H[i][j] for j in range(dim)) for i in range(dim)
        ) / sum_gt_norm
    else:
        scale = 1.0

    # 平移向量 t = gt_centroid - s * R @ pred_centroid
    t = [gt_centroid[coord_keys[i]] - scale * math.fsum(R[i][j] * pred_centroid[coord_keys[j]] for j in range(dim)) for i in range(dim)]

    return R, t, scale, False


def _apply_se3_transform(
    pred_points: list[Mapping],
    R: list[list[float]],
    t: list[float],
    coord_keys: tuple[str, ...],
    scale: float = 1.0,
) -> list[list[float]]:
    """对预测轨迹点应用 Sim(3) 变换 s·R·p + t，返回对齐后的坐标列表。

    参数:
        pred_points: 原始预测点列表。
        R: 旋转矩阵 (dim×dim)。
        t: 平移向量 (dim)。
        coord_keys: 坐标键元组。
        scale: 缩放因子（准则 9 修复：完整 Sim(3) 对齐，默认 1.0 保持 SE3 兼容）。

    返回值:
        对齐后的坐标列表，每个元素是 dim 维浮点数列表。
    """
    dim = len(coord_keys)
    aligned_coords = []
    for p in pred_points:
        pc = [_coerce_scalar(p[k], k) for k in coord_keys]
        ap = [scale * math.fsum(R[i][j] * pc[j] for j in range(dim)) + t[i] for i in range(dim)]
        aligned_coords.append(ap)
    return aligned_coords


def compute_trajectory_metrics(pred_traj: Sequence[Mapping], gt_traj: Sequence[Mapping]) -> dict[str, float | bool]:
    """对预测轨迹和真值轨迹做最小校验和对齐后，计算轨迹误差指标。  # 函数总说明。

    作用：对预测轨迹和真值轨迹做最小校验和对齐后，计算轨迹误差指标。  # 作用说明。
    参数：  # 参数说明。
    - pred_traj: 预测轨迹点序列。  # 参数说明。
    - gt_traj: 真值轨迹点序列。  # 参数说明。
    返回值：  # 返回值说明。
    - dict[str, float | bool]：轨迹误差指标字典，含 rmse/mae/ate/rpe（float）与 ate_degraded（bool，ATE 退化诊断标志）。  # 返回值说明。
    异常/失败条件：  # 失败条件说明。
    - 输入不是非字符串序列时抛 TypeError。  # 类型错误。
    - 输入为空、点结构不合法、坐标字段不一致时抛异常。  # 数据错误。
    状态变化：  # 状态变化说明。
    - 不修改输入，只返回新的指标字典。  # 状态说明。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "pred_traj_len": len(pred_traj) if hasattr(pred_traj, "__len__") else None,
            "gt_traj_len": len(gt_traj) if hasattr(gt_traj, "__len__") else None,
        },
        "compute_trajectory_metrics 入口参数",
        prefix="[metrics]",
    )

    if isinstance(pred_traj, (str, bytes)) or not isinstance(pred_traj, Sequence):  # 预测轨迹必须是非字符串序列。
        raise TypeError("pred_traj must be a non-string sequence of trajectory points.")  # 输入类型错误。
    if isinstance(gt_traj, (str, bytes)) or not isinstance(gt_traj, Sequence):  # 真值轨迹也必须是非字符串序列。
        raise TypeError("gt_traj must be a non-string sequence of trajectory points.")  # 输入类型错误。
    if not pred_traj or not gt_traj:  # 两边都不能为空。
        raise ValueError("pred_traj and gt_traj must be non-empty.")  # 空输入直接拒绝。

    def _as_point_list(traj: Sequence, traj_name: str) -> list[Mapping]:
        """把任意轨迹序列强制检查为映射点列表。  # 内部函数说明。

        作用：确保轨迹中的每个元素都是映射，后续才能按字段读取坐标和时间。  # 作用说明。
        参数：  # 参数说明。
        - traj: 待检查的轨迹序列。  # 参数说明。
        - traj_name: 用于错误信息的轨迹名称。  # 参数说明。
        返回值：  # 返回值说明。
        - list[Mapping]：校验后的点列表。  # 返回值说明。
        异常/失败条件：  # 失败条件说明。
        - 任一元素不是映射时抛 TypeError。  # 类型错误。
        状态变化：  # 状态变化说明。
        - 不修改原序列，只返回新的列表。  # 状态说明。
        """
        points = []  # 保存校验后的点列表。
        for index, point in enumerate(traj):  # 逐点检查输入轨迹。
            if not isinstance(point, Mapping):  # 每个点都必须是映射。
                raise TypeError(f"{traj_name}[{index}] must be a mapping.")  # 说明具体坏点位置。
            points.append(point)  # 收集合法点。
        return points  # 返回合法点列表。

    pred_points = _as_point_list(pred_traj, "pred_traj")  # 先检查预测轨迹点结构。
    gt_points = _as_point_list(gt_traj, "gt_traj")  # 再检查真值轨迹点结构。

    def _coord_keys(points: Sequence[Mapping], traj_name: str) -> tuple[str, ...]:
        """判断轨迹是二维还是三维，并要求所有点一致。  # 内部函数说明。

        作用：检查点是否至少有 px/py，并判断是否同时存在 pz。  # 作用说明。
        参数：  # 参数说明。
        - points: 轨迹点列表。  # 参数说明。
        - traj_name: 用于错误信息的轨迹名称。  # 参数说明。
        返回值：  # 返回值说明。
        - tuple[str, ...]：坐标键元组。  # 返回值说明。
        异常/失败条件：  # 失败条件说明。
        - 点缺少 px/py 或 pz 混用时抛异常。  # 数据错误。
        """
        if not all("px" in point and "py" in point for point in points):  # 每个点都必须至少有 px 和 py。
            raise ValueError(f"{traj_name} points must contain 'px' and 'py'.")  # 缺少基础坐标就不能继续。
        point_has_pz = [("pz" in point) for point in points]  # 记录每个点是否带第三维。
        if any(point_has_pz) and not all(point_has_pz):  # 不能只让一部分点带 pz。
            raise ValueError(f"{traj_name} points must use consistent coordinate fields.")  # 坐标维度必须一致。
        has_pz = all(point_has_pz)  # 所有点都带 pz 才算三维轨迹。
        return ("px", "py", "pz") if has_pz else ("px", "py")  # 按实际维度返回坐标键。

    coord_keys = _coord_keys(pred_points, "pred_traj")  # 先检查预测轨迹的坐标维度。
    if _coord_keys(gt_points, "gt_traj") != coord_keys:  # 真值和预测必须使用同一坐标维度。
        raise ValueError("pred_traj and gt_traj must use the same coordinate fields.")  # 维度不同就不能比较。

    def _align_by_time_or_index(pred_points: list[Mapping], gt_points: list[Mapping]) -> list[tuple[Mapping, Mapping]]:
        """优先按时间键对齐，否则按位置一一配对。  # 内部函数说明。

        作用：先尝试按共同时间键对齐，若不存在共同时间键则退化为按索引对齐。  # 作用说明。
        参数：  # 参数说明。
        - pred_points: 预测点列表。  # 参数说明。
        - gt_points: 真值点列表。  # 参数说明。
        返回值：  # 返回值说明。
        - list[tuple]：对齐后的点对列表。  # 返回值说明。
        异常/失败条件：  # 失败条件说明。
        - 无法对齐、长度不一致、时间键重复时抛异常。  # 数据错误。
        """
        time_keys = TIME_KEY_CANDIDATES  # 时间字段候选（D9 单源引用常量，"index" 不是时间字段，不应作为时间键候选，否则会把位置序号误当时间键导致按值而非按位置对齐）。
        time_key = next(  # 逐个寻找两边都共有的时间字段。
            (  # 生成器表达式起始。
                key  # 当前候选时间键。
                for key in time_keys  # 依次检查所有候选键。
                if all(key in point for point in pred_points) and all(key in point for point in gt_points)  # 两边都存在才算公共键。
            ),
            None,  # 如果没有公共时间键，就返回 None 走位置对齐分支。
        )  # next 调用结束。
        if time_key is None:  # 没有公共时间键时，退化为逐位置对齐。
            if len(pred_points) != len(gt_points):  # 位置对齐要求长度完全一致。
                raise ValueError("pred_traj and gt_traj must have equal length when no time key is present.")  # 长度不同就无法逐点配对。
            return list(zip(pred_points, gt_points))  # 直接按位置配对返回。

        def _validate_unique_keys(points: list[Mapping], traj_name: str) -> None:
            """检查每个时间键是否唯一，避免对齐歧义。  # 内部函数说明。"""
            seen = set()  # 记录已经出现过的时间键值。
            for point in points:  # 逐点检查重复键。
                key_value = point[time_key]  # 取出当前点的时间键值。
                # D5 数值安全：时间键值必须可哈希（用作 set/dict key）；浮点值必须有限（NaN 会破坏去重和 dict 查找语义）。
                try:
                    hash(key_value)  # 不可哈希类型（如 list/dict）会导致后续 set/dict 操作崩溃。
                except TypeError as exc:
                    raise ValueError(f"{traj_name} '{time_key}' value must be hashable, got {type(key_value).__name__}.") from exc
                if isinstance(key_value, float) and not math.isfinite(key_value):  # NaN/Inf 作为 dict key 会破坏相等性语义。
                    raise ValueError(f"{traj_name} '{time_key}' value must be finite, got {key_value!r}.")
                if key_value in seen:  # 发现重复说明会产生歧义。
                    raise ValueError(f"{traj_name} contains duplicate '{time_key}' values.")  # 直接报错。
                seen.add(key_value)  # 记录当前键值。

        _validate_unique_keys(pred_points, "pred_traj")  # 先检查预测轨迹的时间键是否重复。
        _validate_unique_keys(gt_points, "gt_traj")  # 再检查真值轨迹的时间键是否重复。

        gt_by_time = {}  # 用时间键建立真值索引表。
        for point in gt_points:  # 逐个真值点建立索引。
            gt_by_time[point[time_key]] = point  # 以时间键为键保存点本身。
        aligned = [  # 生成对齐后的点对列表。
            (point, gt_by_time[point[time_key]])  # 预测点和同时间真值点组成一对。
            for point in pred_points  # 逐个预测点查找真值。
            if point[time_key] in gt_by_time  # 只保留能在真值里找到的点。
        ]
        if not aligned:  # 如果一个都没对上，说明轨迹没有重叠。
            raise ValueError(f"pred_traj and gt_traj do not overlap on '{time_key}'.")  # 直接报错。
        return aligned  # 返回按时间或位置对齐后的点对。

    aligned_pred = _align_by_time_or_index(pred_points, gt_points)  # 得到对齐后的点对列表。

    def _point_error(pred_point: Mapping, gt_point: Mapping) -> float:
        """计算单个点对的欧氏误差。  # 内部函数说明。

        作用：根据坐标键，计算预测点和真值点之间的欧氏距离。  # 作用说明。
        参数：  # 参数说明。
        - pred_point: 预测点。  # 参数说明。
        - gt_point: 真值点。  # 参数说明。
        返回值：  # 返回值说明。
        - float：单点误差。  # 返回值说明。
        """
        def _coerce_coordinate(value, key: str) -> float:
            """把单个坐标值转成有限浮点数。  # 内部函数说明。"""
            if is_bool_like(value):  # 布尔值不能冒充数值。
                raise ValueError(f"Trajectory coordinate '{key}' must be numeric.")  # 布尔值直接拒绝。
            try:  # 尝试统一转成浮点数。
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:  # 不能转数值就报错；OverflowError 守卫超大整数（如 float(10**1000) 会抛 OverflowError）。
                raise ValueError(f"Trajectory coordinate '{key}' must be numeric.") from exc  # 保留异常链。
            if not math.isfinite(value):  # NaN 和无穷大不能参与计算。
                raise ValueError(f"Trajectory coordinate '{key}' must be finite.")  # 非有限值报错。
            return value  # 返回规范化浮点数。

        squared_error = 0.0  # 当前点对的平方误差累计值。
        for key in coord_keys:  # 逐个坐标维度计算差值。
            delta = _coerce_coordinate(pred_point[key], key) - _coerce_coordinate(gt_point[key], key)  # 先规范化再相减。
            squared_error += delta * delta  # 累加平方项。
        return math.sqrt(squared_error)  # 返回欧氏距离。

    # Step 1: 原始误差（无对齐）—— 保留为 mae_raw / rmse_raw 口径。
    # 该口径会反映预测轨迹的整体偏移，对无标定的纯相对位置模型有意义。
    errors_raw = [_point_error(pred_point, gt_point) for pred_point, gt_point in aligned_pred]  # 计算所有对齐点对的原始误差。
    rmse_raw = math.sqrt(math.fsum(error * error for error in errors_raw) / len(errors_raw))  # 原始均方根误差。
    mae_raw = math.fsum(errors_raw) / len(errors_raw)  # 原始平均绝对误差。

    # Step 2: Umeyama / SE3 对齐后误差。
    # 学习型方法（Transformer/LSTM）会有累积旋转/平移漂移，
    # 原始 RMSE 会系统性偏大。SE3 对齐消除全局平移/旋转后反映轨迹形状精度。
    R, t, scale, align_degraded = _umeyama(aligned_pred, coord_keys)  # 提取刚体变换参数（含 Sim(3) scale）
    pred_points_only = [p for p, _ in aligned_pred]  # 取出对齐后的预测点序列。
    aligned_pred_coords = _apply_se3_transform(pred_points_only, R, t, coord_keys, scale=scale)  # 应用 Sim(3) 刚体+缩放变换。
    errors_aligned = []  # 对齐后点对误差。
    for i, (_, gt_point) in enumerate(aligned_pred):
        gt_coords = [_coerce_scalar(gt_point[k], k) for k in coord_keys]  # 取出真值坐标。
        diff = math.fsum((aligned_pred_coords[i][j] - gt_coords[j]) ** 2 for j in range(len(coord_keys)))  # 对齐后欧氏距离平方。
        errors_aligned.append(math.sqrt(diff))
    rmse = math.sqrt(math.fsum(error * error for error in errors_aligned) / len(errors_aligned))  # 对齐后均方根误差（项 9 修复：主口径）。
    mae = math.fsum(errors_aligned) / len(errors_aligned)  # 对齐后平均绝对误差。

    # ATE: 复用相同的 SE3 对齐（避免重复 SVD 计算）。
    # ATE 与对齐后 RMSE 在 SE3 对齐下数值相同，但保留 ate/ate_degraded 以维持下游消费者协议不变。
    ate, ate_degraded = _compute_ate_se3(aligned_pred, coord_keys)  # 独立计算，保留协议兼容。

    # RPE: 相邻帧之间的相对运动误差（平移分量帧间差）。
    # 本实现为平移RPE（translation-only RPE），因为项目轨迹只包含位置信息（px/py/pz），
    # 不含旋转信息（yaw/roll/pitch），因此无法计算旋转分量。
    # 与 Sturm et al. 2012 完整 RPE 定义（含旋转）不同，此处仅衡量平移漂移率。
    if len(aligned_pred) >= 2:  # 只有至少两个点时才计算相对位姿误差。
        rpe = _compute_rpe(aligned_pred, coord_keys)  # 平移RPE。
    else:
        rpe = 0.0  # 样本太少时退化为 0。

    return {  # 返回最终轨迹误差指标字典。
        "rmse": rmse,  # 均方根误差（Umeyama/SE3 对齐后；学习型方法的主指标）。
        "rmse_raw": rmse_raw,  # 原始均方根误差（无对齐；保留用于诊断累积漂移幅度）。
        "mae": mae,  # 平均绝对误差（SE3 对齐后）。
        "mae_raw": mae_raw,  # 原始平均绝对误差。
        "ate": ate,  # 绝对轨迹误差（SE3对齐后；ate_degraded=True时为去均值偏移后RMSE）。
        "ate_degraded": ate_degraded,  # True表示ATE因n<3退化为去均值偏移后RMSE，非标准SE3-ATE。
        "align_degraded": align_degraded,  # True表示Umeyama/SE3对齐因n<3退化为纯平移对齐。
        "rpe": rpe,  # 平移相对位姿误差（translation-only RPE）。
    }


def _compute_ate_se3(aligned_pred: list[tuple[Mapping, Mapping]], coord_keys: tuple[str, ...]) -> tuple[float, bool]:
    """计算 SE3 刚体对齐后的绝对轨迹误差（ATE）。

    按 Sturm et al. 2012 标准定义：
    1. 对预测轨迹做 SE3 刚体变换，使其与真值轨迹的最小二乘对齐。
    2. 计算对齐后轨迹的 RMSE。

    当点数少于3个时，SE3对齐退化为平移对齐（减去质心偏移），
    此时 ATE 等价于去均值偏移后 RMSE，与标准 SE3-ATE 定义不一致。
    返回 ate_degraded=True 标志以显式标记此退化情况。

    参数:
        aligned_pred: 对齐后的 (pred_point, gt_point) 列表。
        coord_keys: 坐标键元组。

    返回值:
        tuple[float, bool]: (SE3 对齐后的 RMSE, 是否退化)。
            ate_degraded=True 时，返回值为去均值偏移后 RMSE，非标准 SE3-ATE。
    """

    def _coerce_coordinate(value, key: str) -> float:
        """把单个坐标值转成有限浮点数。  # 复用全局 _coerce_scalar，与项 9 修复口径一致。"""
        return _coerce_scalar(value, key)

    n = len(aligned_pred)
    if n < 3:
        # 点数不足时，SE3 对齐不可靠（旋转自由度无法稳定估计），
        # 退化为平移对齐：只减去平均偏移后重新计算 RMSE。
        # 此结果与 Sturm et al. 2012 标准ATE定义不一致，标记 ate_degraded=True。
        mean_deltas = {}  # 各坐标维度的平均偏移量。
        for key in coord_keys:
            mean_delta = math.fsum(
                _coerce_coordinate(p[key], key) - _coerce_coordinate(g[key], key) for p, g in aligned_pred
            ) / n  # 预测与真值的平均差，用 fsum 提高精度。
            mean_deltas[key] = mean_delta
        squared_sum = 0.0  # 去偏后的平方误差累计。
        for p, g in aligned_pred:
            se = 0.0  # 当前点对的平方误差。
            for key in coord_keys:
                delta = (_coerce_coordinate(p[key], key) - _coerce_coordinate(g[key], key)) - mean_deltas[key]  # 减去平均偏移，消除全局平移。
                se += delta * delta
            squared_sum += se
        return math.sqrt(squared_sum / n), True  # 返回去偏后的 RMSE 和退化标志。

    # 计算预测和真值的质心（各坐标维度的均值），用于后续中心化。
    pred_centroid = {key: math.fsum(_coerce_coordinate(p[key], key) for p, _ in aligned_pred) / n for key in coord_keys}  # 预测质心，用 fsum 提高精度。
    gt_centroid = {key: math.fsum(_coerce_coordinate(g[key], key) for _, g in aligned_pred) / n for key in coord_keys}  # 真值质心，用 fsum 提高精度。

    # 中心化：把预测和真值都减去各自质心，消除平移分量，只保留旋转待解。
    pred_centered = []  # 中心化后的预测坐标列表。
    gt_centered = []  # 中心化后的真值坐标列表。
    for p, g in aligned_pred:
        pred_centered.append([_coerce_coordinate(p[key], key) - pred_centroid[key] for key in coord_keys])
        gt_centered.append([_coerce_coordinate(g[key], key) - gt_centroid[key] for key in coord_keys])

    # 构造互协方差矩阵 H = pred_centered^T * gt_centered（2D 或 3D）。
    # H 的 SVD 分解可以给出最优旋转矩阵 R，使 R * pred 最接近 gt。
    dim = len(coord_keys)  # 坐标维度，2 或 3。
    H = [[0.0] * dim for _ in range(dim)]  # 初始化 dim x dim 零矩阵。
    for pc, gc in zip(pred_centered, gt_centered):
        for i in range(dim):
            for j in range(dim):
                H[i][j] += pc[i] * gc[j]  # 逐点累积外积。

    # SVD 分解求最优旋转 R = V * U^T（Procrustes 问题的标准解法）。
    # 使用 2x2 或 3x3 的显式 SVD，避免引入 numpy 依赖，保持纯 Python 实现。
    if dim == 2:
        R = _svd_2x2_rotation(H)
    else:
        R = _svd_3x3_rotation(H)

    # 计算最优平移 t = gt_centroid - R * pred_centroid（由旋转和质心自动推导）。
    # 对齐后的预测点: p_aligned = R * (p - pred_centroid) + gt_centroid
    # 这样就把预测轨迹先旋转到与真值最接近的朝向，再平移到真值质心位置。
    squared_sum = 0.0  # 对齐后的平方误差累计。
    for p, g in aligned_pred:
        pc = [_coerce_coordinate(p[key], key) - pred_centroid[key] for key in coord_keys]  # 中心化预测点。
        aligned_p = [math.fsum(R[i][j] * pc[j] for j in range(dim)) + gt_centroid[coord_keys[i]] for i in range(dim)]  # 旋转后加真值质心，用 fsum 提高精度。
        se = 0.0  # 当前点对对齐后的平方误差。
        for i, key in enumerate(coord_keys):
            delta = aligned_p[i] - _coerce_coordinate(g[key], key)  # 对齐后预测与真值的差。
            se += delta * delta
        squared_sum += se

    return math.sqrt(squared_sum / n), False  # 返回 SE3 对齐后的 RMSE 和非退化标志。


def _svd_2x2_rotation(H: list[list[float]]) -> list[list[float]]:
    """2x2 矩阵的 SVD 分解，返回最优旋转矩阵 R = V * U^T。

    使用解析公式计算 2x2 SVD，避免引入 numpy 依赖。
    这是 Procrustes 问题的核心：给定互协方差矩阵 H，
    通过 SVD(H) = U * Sigma * V^T 求出最优旋转 R = V * U^T，
    使得 R * pred 与 gt 的最小二乘误差最小。

    参数:
        H: 2x2 互协方差矩阵，H = pred_centered^T * gt_centered。

    返回值:
        list[list[float]]: 2x2 旋转矩阵 R（行列式为 +1）。
    """
    a, b = H[0][0], H[0][1]  # H 的第一行。
    c, d = H[1][0], H[1][1]  # H 的第二行。

    # 计算 H^T * H 的元素，这是一个 2x2 对称正半定矩阵。
    e1 = a * a + c * c  # H^T*H 的 (0,0) 元素。
    e2 = a * b + c * d  # H^T*H 的 (0,1) 元素（等于 (1,0)）。
    e3 = b * b + d * d  # H^T*H 的 (1,1) 元素。

    # 2x2 对称矩阵的特征分解：用迹和行列式直接算特征值。
    trace_ht = e1 + e3  # H^T*H 的迹。
    det_ht = e1 * e3 - e2 * e2  # H^T*H 的行列式。
    disc = max(0.0, trace_ht * trace_ht / 4.0 - det_ht)  # 判别式，截断到 0 防止浮点误差导致负值。
    sqrt_disc = math.sqrt(disc)
    lambda1 = trace_ht / 2.0 + sqrt_disc  # 较大特征值。
    lambda2 = max(0.0, trace_ht / 2.0 - sqrt_disc)  # 较小特征值，截断到 0。

    # V 的列向量（H^T*H 的特征向量），用于构造右奇异矩阵。
    if abs(e2) > 1e-12:  # 非对角元素不为零时，用标准公式计算特征向量。
        v1 = [lambda1 - e3, e2]  # 对应 lambda1 的特征向量。
        v2 = [lambda2 - e3, e2]  # 对应 lambda2 的特征向量。
    elif e1 >= e3:  # 对角矩阵且 e1 >= e3 时，特征向量就是标准基。
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
    else:  # 对角矩阵且 e1 < e3 时，特征向量翻转。
        v1 = [0.0, 1.0]
        v2 = [1.0, 0.0]

    # 归一化特征向量，使其成为单位向量。
    n1 = math.sqrt(math.fsum(x * x for x in v1))  # v1 的模长，用 fsum 提高精度（与 _svd_3x3_rotation 一致）。
    n2 = math.sqrt(math.fsum(x * x for x in v2))  # v2 的模长，用 fsum 提高精度。
    if n1 > 1e-15:  # 避免除零。
        v1 = [v1[0] / n1, v1[1] / n1]
    if n2 > 1e-15:
        v2 = [v2[0] / n2, v2[1] / n2]

    # 计算奇异值 sigma_i = sqrt(lambda_i)，用于从 V 推导 U。
    sigma1 = math.sqrt(max(1e-30, lambda1))  # 第一个奇异值，下限防止除零。
    sigma2 = math.sqrt(max(1e-30, lambda2))  # 第二个奇异值。

    # V 矩阵（列向量 v1, v2），即右奇异矩阵。
    V = [[v1[0], v2[0]], [v1[1], v2[1]]]

    # U 的列向量：u_i = H * v_i / sigma_i，即左奇异矩阵。
    # 这是 SVD 的标准关系：H * V = U * Sigma。
    # 当奇异值过小时（H 退化），u_i 置零避免除以极小值放大噪声，与 _svd_3x3_rotation 一致。
    if sigma1 > 1e-15:
        u1 = [(a * v1[0] + b * v1[1]) / sigma1, (c * v1[0] + d * v1[1]) / sigma1]  # 第一个左奇异向量。
    else:
        u1 = [0.0, 0.0]
    if sigma2 > 1e-15:
        u2 = [(a * v2[0] + b * v2[1]) / sigma2, (c * v2[0] + d * v2[1]) / sigma2]  # 第二个左奇异向量。
    else:
        u2 = [0.0, 0.0]

    # R = V * U^T（Procrustes 问题的正确解：最大化 trace(R * H)）。
    # 这等价于最小化 ||R * pred - gt||^2 的最优旋转。
    r00 = V[0][0] * u1[0] + V[0][1] * u2[0]  # R[0][0]。
    r01 = V[0][0] * u1[1] + V[0][1] * u2[1]  # R[0][1]。
    r10 = V[1][0] * u1[0] + V[1][1] * u2[0]  # R[1][0]。
    r11 = V[1][0] * u1[1] + V[1][1] * u2[1]  # R[1][1]。

    det_r = r00 * r11 - r01 * r10  # 旋转矩阵的行列式必须为 +1。
    if det_r < 0:
        # 行列式为 -1 说明得到的是反射而非旋转，需要翻转 V 的最后一列来修正。
        # 这是 SVD 求旋转时的标准修正步骤。
        v2 = [-v2[0], -v2[1]]  # 翻转第二个右奇异向量。
        V = [[v1[0], v2[0]], [v1[1], v2[1]]]  # 重建 V 矩阵。
        r00 = V[0][0] * u1[0] + V[0][1] * u2[0]  # 重新计算 R 各元素。
        r01 = V[0][0] * u1[1] + V[0][1] * u2[1]
        r10 = V[1][0] * u1[0] + V[1][1] * u2[0]
        r11 = V[1][0] * u1[1] + V[1][1] * u2[1]

    return [[r00, r01], [r10, r11]]  # 返回 2x2 旋转矩阵。


def _svd_3x3_rotation(H: list[list[float]]) -> list[list[float]]:
    """3x3 矩阵的 SVD 分解，返回最优旋转矩阵 R。

    使用幂迭代法计算最大奇异值对应的奇异向量，
    然后通过叉积构造完整的旋转矩阵。
    对于 3x3 情况，这是数值上足够稳健的方法，且避免引入 numpy。

    参数:
        H: 3x3 互协方差矩阵，H = pred_centered^T * gt_centered。

    返回值:
        list[list[float]]: 3x3 旋转矩阵 R（行列式为 +1）。
    """
    # 先计算 H^T * H，这是一个 3x3 对称正半定矩阵。
    HtH = [[math.fsum(H[k][i] * H[k][j] for k in range(3)) for j in range(3)] for i in range(3)]  # H^T H 矩阵，用 fsum 提高精度。

    # 幂迭代求 H^T*H 的最大特征值和特征向量（即 V 的第一列）。
    v = [1.0, 0.0, 0.0]  # 初始猜测，任意非零向量即可。
    for _ in range(50):  # 50 次迭代通常足够收敛到主特征向量。
        v_new = [  # 矩阵-向量乘法 HtH * v。
            HtH[0][0] * v[0] + HtH[0][1] * v[1] + HtH[0][2] * v[2],
            HtH[1][0] * v[0] + HtH[1][1] * v[1] + HtH[1][2] * v[2],
            HtH[2][0] * v[0] + HtH[2][1] * v[1] + HtH[2][2] * v[2],
        ]
        norm = math.sqrt(math.fsum(x * x for x in v_new))  # 归一化防止数值溢出，用 fsum 提高精度。
        if norm < 1e-30:  # 向量接近零说明 HtH 退化，返回单位旋转。
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        v = [x / norm for x in v_new]  # 归一化后的新迭代向量。

    # v 现在是 H^T*H 的最大特征向量 → V 的第一列 v1。
    v1 = v

    # 第二个特征向量 v2：在 v1 的正交补空间中做幂迭代。
    # 先选一个与 v1 不平行的初始向量。
    if abs(v1[0]) < 0.9:  # 如果 v1 的 x 分量较小，就选 x 方向。
        v2_init = [1.0, 0.0, 0.0]
    else:  # 否则选 y 方向。
        v2_init = [0.0, 1.0, 0.0]
    # Gram-Schmidt 正交化：从 v2_init 中减去 v1 方向的分量。
    dot = math.fsum(v1[i] * v2_init[i] for i in range(3))  # 用 fsum 提高精度。
    v2 = [v2_init[i] - dot * v1[i] for i in range(3)]
    norm2 = math.sqrt(math.fsum(x * x for x in v2))  # 用 fsum 提高精度。
    if norm2 < 1e-30:  # 正交化后接近零，说明初始选择不合适，换一个方向。
        v2 = [0.0, 0.0, 1.0]
    else:
        v2 = [x / norm2 for x in v2]  # 归一化。

    # 在 v1 的正交补空间中做幂迭代，求次大特征向量。
    for _ in range(50):
        v2_new = [  # HtH * v2。
            HtH[0][0] * v2[0] + HtH[0][1] * v2[1] + HtH[0][2] * v2[2],
            HtH[1][0] * v2[0] + HtH[1][1] * v2[1] + HtH[1][2] * v2[2],
            HtH[2][0] * v2[0] + HtH[2][1] * v2[1] + HtH[2][2] * v2[2],
        ]
        # 投影到 v1 的正交补，确保 v2 始终与 v1 正交。
        dot1 = math.fsum(v1[i] * v2_new[i] for i in range(3))  # 用 fsum 提高精度。
        v2_new = [v2_new[i] - dot1 * v1[i] for i in range(3)]
        norm2 = math.sqrt(math.fsum(x * x for x in v2_new))  # 用 fsum 提高精度。
        if norm2 < 1e-30:  # 收敛到零说明只有一个有效特征方向。
            break
        v2 = [x / norm2 for x in v2_new]

    # 第三列 v3 = v1 × v2，保证 V 是右手正交矩阵。
    v3 = [
        v1[1] * v2[2] - v1[2] * v2[1],  # 叉积的 x 分量。
        v1[2] * v2[0] - v1[0] * v2[2],  # 叉积的 y 分量。
        v1[0] * v2[1] - v1[1] * v2[0],  # 叉积的 z 分量。
    ]

    # V 矩阵（列向量 v1, v2, v3），即右奇异矩阵。
    V = [[v1[0], v2[0], v3[0]], [v1[1], v2[1], v3[1]], [v1[2], v2[2], v3[2]]]

    # 计算 H * V，用于推导 U 和奇异值。
    HV = [[math.fsum(H[i][k] * V[k][j] for k in range(3)) for j in range(3)] for i in range(3)]  # H*V 矩阵，用 fsum 提高精度。

    # 奇异值 sigma_i = ||H * v_i||，即 HV 第 i 列的范数。
    sigma = [math.sqrt(max(1e-30, math.fsum(HV[i][j] ** 2 for i in range(3)))) for j in range(3)]  # 奇异值，用 fsum 提高精度。

    # U 的列向量：u_i = H * v_i / sigma_i，即左奇异矩阵。
    U = [[HV[i][j] / sigma[j] if sigma[j] > 1e-15 else 0.0 for j in range(3)] for i in range(3)]

    # R = V * U^T（Procrustes 问题的正确解：最大化 trace(R * H)）。
    R = [[math.fsum(V[i][k] * U[j][k] for k in range(3)) for j in range(3)] for i in range(3)]  # R = V * U^T，用 fsum 提高精度。

    # 检查行列式，保证 R 是旋转矩阵（行列式=+1）而非反射（行列式=-1）。
    det_r = (
        R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
        - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
        + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0])
    )
    if det_r < 0:
        # 行列式为 -1 时翻转 V 的最后一列，强制 R 变成旋转矩阵。
        for i in range(3):
            V[i][2] = -V[i][2]  # 翻转第三列。
        R = [[math.fsum(V[i][k] * U[j][k] for k in range(3)) for j in range(3)] for i in range(3)]  # 重新计算 R = V * U^T，用 fsum 提高精度。

    return R  # 返回 3x3 旋转矩阵。


def _compute_rpe(aligned_pred: list[tuple[Mapping, Mapping]], coord_keys: tuple[str, ...]) -> float:
    """计算平移相对位姿误差（translation-only RPE）。

    本实现只计算平移分量帧间差，不包含旋转分量。
    原因：项目轨迹只包含位置信息（px/py/pz），不含旋转信息（yaw/roll/pitch），
    因此无法计算旋转分量。与 Sturm et al. 2012 完整 RPE 定义（含旋转）不同，
    此处仅衡量平移漂移率。

    计算方式：对于每对相邻帧 i→i+1，计算 (pred_{i+1} - pred_i) 与 (gt_{i+1} - gt_i)
    的欧氏距离，然后取所有帧对的平均值。

    参数:
        aligned_pred: 对齐后的 (pred_point, gt_point) 列表。
        coord_keys: 坐标键元组，如 ("px", "py") 或 ("px", "py", "pz")。

    返回值:
        float: 平均相对位姿误差。样本不足时返回 0.0。
    """
    n = len(aligned_pred)
    if n < 2:  # 至少需要两个点才能计算相邻帧差。
        return 0.0

    def _coerce_coordinate(value, key: str) -> float:
        """把单个坐标值转成有限浮点数。  # 与 _point_error/_compute_ate_se3 内的 _coerce_coordinate 校验口径保持一致。"""
        if is_bool_like(value):  # 布尔值不能冒充数值。
            raise ValueError(f"Trajectory coordinate '{key}' must be numeric.")
        try:  # 尝试统一转成浮点数。
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:  # 不能转数值就报错；OverflowError 守卫超大整数（如 float(10**1000) 会抛 OverflowError）。
            raise ValueError(f"Trajectory coordinate '{key}' must be numeric.") from exc
        if not math.isfinite(value):  # NaN 和无穷大不能参与计算。
            raise ValueError(f"Trajectory coordinate '{key}' must be finite.")
        return value

    pair_errors = []  # 每对相邻帧的相对运动误差。
    for i in range(1, n):  # 从第 2 个点开始，与前一个点配对。
        pred_prev, gt_prev = aligned_pred[i - 1]  # 前一帧的预测和真值。
        pred_curr, gt_curr = aligned_pred[i]  # 当前帧的预测和真值。

        # 计算预测和真值在相邻帧之间的运动向量差，再取欧氏范数。
        # 这衡量的是"预测走了多远"与"真值走了多远"之间的偏差。
        diff_sq = 0.0  # 运动向量差的平方和。
        for key in coord_keys:  # 逐坐标维度计算。
            pred_d = _coerce_coordinate(pred_curr[key], key) - _coerce_coordinate(pred_prev[key], key)  # 预测侧的帧间运动量。
            gt_d = _coerce_coordinate(gt_curr[key], key) - _coerce_coordinate(gt_prev[key], key)  # 真值侧的帧间运动量。
            diff_sq += (pred_d - gt_d) ** 2  # 运动差异的平方。

        pair_errors.append(math.sqrt(diff_sq))  # 当前帧对的欧氏运动误差。

    return math.fsum(pair_errors) / (n - 1)  # 取平均，用 fsum 提高精度（与 U6 mae/rmse 一致），分母是帧对数 n-1。
