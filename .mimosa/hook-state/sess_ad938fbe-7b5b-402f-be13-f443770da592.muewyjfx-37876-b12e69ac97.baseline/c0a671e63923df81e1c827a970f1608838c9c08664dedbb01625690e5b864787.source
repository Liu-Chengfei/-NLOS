"""文件:src/liquidloc/metrics/metric_runner.py  # 文件头说明：这是指标总调度器文件。

【文件职责】统一调度 trajectory/tail/reliability/runtime 四类指标计算，并输出固定顺序的 metric_table。  # 说明本文件的主职责：把多个指标模块的结果合并成统一表。
【本文件绝对不负责】不挑案例，不做统计检验，不画图。  # 明确边界：这里只做指标汇总，不做分析决策和可视化。
【上游依赖】trajectory_metrics.py、tail_metrics.py、reliability_metrics.py、runtime_metrics.py、protocol/metric_schema.py。  # 说明它依赖哪些模块来产出各类指标。
【下游调用者】analysis/*、scripts/11_compute_metrics.py、tests/metrics/test_metric_runner.py。  # 说明哪些脚本/测试会直接使用这个文件。
【输入对象定义】- prediction_bundle  # 说明输入之一：预测包。
- gt_bundle  # 说明输入之二：真值包。

【输出对象定义】- metric_table  # 说明输出对象：统一指标表。

【核心变量定义】- trajectory_block  # trajectory 类指标结果块。
- tail_block  # tail 类指标结果块。
- reliability_block  # reliability 类指标结果块。
- runtime_block  # runtime 类指标结果块。
- metric_order  # 指标顺序来源。
- metric_table  # 最终合并结果。

【推荐编写顺序】1. 先写 compute_metrics。2. 按四类指标分块。3. 最后按 metric_schema 顺序输出表。  # 说明推荐的阅读顺序。
【建议先写的函数 / 类】- 统一计算指标  # 当前文件最核心的公共入口。
  签名：def compute_metrics(prediction_bundle, gt_bundle)  # 先固定入口签名，方便读者对照调用方式。
  作用：输出单序列或多序列统一指标表。  # 说明该入口会把多类指标整理成表。
  输入：prediction_bundle、gt_bundle  # 列出两个输入。
  输出：metric_table  # 列出一个输出。
  关键局部变量：  # 这里列出后面会反复出现的中间结果。
    - trajectory_block
    - tail_block
    - reliability_block
    - runtime_block
    - metric_table
  伪代码：  # 用自然语言解释执行流程。
    1) 对齐 prediction_bundle 和 gt_bundle。2) 分别调用四类指标函数。3) 把四类结果按固定字段顺序合并。4) 返回 metric_table。
【最容易误读 Codex 理解错的地方】- 不要把指标顺序硬编码成与 metric_schema 不一致。- 不要在这里插入案例选择字段。  # 提示最容易误读的风险点。
【最小手工测试步骤】1. 构造最小 prediction/gt。2. 检查四类指标合并后字段顺序稳定。  # 提示最小验证方式。
【完成标准】- metric_table 作为 analysis 和 plotting 的唯一入口表。  # 说明完成后这文件承担的输出合同。
【实现要求】- 当前阶段保持空白实现，只保留细化编写说明。- 真正实现时先写签名和 docstring，再写前置检查，再写主体逻辑，最后补测试。- 任何字段名和变量名优先服从 protocol 和 configs，不能临时发明。  # 明确本文件的约束。
"""

from __future__ import annotations  # 允许前向类型注解，减少运行期类型依赖。

import math  # 提供平方根、有限性检查和高精度求和等数学工具。
import warnings  # 用于 §9.3 脉冲/异步量级门违反时显式 Warning, 不阻断评估流程.
from collections.abc import Mapping, Sequence  # 用于识别映射和序列类型，避免把字符串误当序列。
from typing import Any  # 用于宽松类型标注。

from liquidloc.common.angle_utils import angle_delta_rad, wrap_angle_rad  # 角度差计算和角度归一化。
from liquidloc.common.constants import (  # 单源真相常量，禁止本地硬编码字面量。
    MODALITY_UWB,
    MODALITY_VIO,
    RELIABILITY_STATUS_OK,
    RELIABILITY_STATUS_INSUFFICIENT,
    HARD_REJECT_REASON_MISSING_GT,
    TIME_KEY_CANDIDATES,
)
from liquidloc.common.gt_utils import align_ground_truth, normalize_gt_rows, GT_TIME_TOLERANCE  # 真值归一化与对齐的规范实现。
from liquidloc.common.statistics_utils import compute_trace_correlation  # 公共轨迹相关性计算。
from liquidloc.common.validation import is_bool_like, is_real  # 统一判断布尔类型（含 numpy.bool_）与实数类型（排除 bool）。
from liquidloc.metrics.reliability_metrics import compute_reliability_metrics  # 复用可靠性指标实现。
from liquidloc.metrics.runtime_metrics import compute_runtime_metrics  # 复用运行时指标实现。
from liquidloc.metrics.tail_metrics import compute_tail_metrics  # 复用尾部指标实现。
from liquidloc.metrics.trajectory_metrics import compute_trajectory_metrics  # 复用轨迹误差指标实现。
from liquidloc.protocol.experiment_gates import get_default_failure_threshold_m  # 从实验协议读取默认失败阈值。
# 注: experiment_gates.get_cold_start_offset_s 函数已彻底删除 (Round 4 §9 穷举审视):
# 全 src/scripts/tests 0 个调用点, 文档 0 引用; 重新接入会破坏 metric_runner 的 partial-cfg 路径
# (partial cfg 缺 scene_scale 时 helper 会回退默认值 5.0s, 而现行内联读保持 0.0 = 不启用划除).
# 冷启动偏移由 _collect_sequence_data 直接读 protocol_cfg['scene_scale']['cold_start_offset_s']
# 与 protocol_cfg['scene_scale']['cold_start_global_enforced'] (详见 L839-848), 单一真相源就此归位.
from liquidloc.protocol.metric_schema import get_metric_order  # 读取协议冻结的指标顺序。

_GT_TIME_TOLERANCE = GT_TIME_TOLERANCE  # 向后兼容别名：使用 gt_utils 的规范值。

# 可靠性状态字符串统一引用 common/constants.py 单源真相（D9 配置表面漂移根因修复），
# 禁止本地硬编码 "ok"/"insufficient_overlap" 字面量，与 reliability_metrics.py 保持同步。

# 模块内部别名：metric_runner 使用 yaw_default=None 保留缺失 yaw 的语义，
# 由下游 yaw 指标计算过滤。
_normalize_gt_rows = lambda gt_rows: normalize_gt_rows(gt_rows, yaw_default=None)
_align_ground_truth = align_ground_truth


def _normalize_bundle_group(bundle_obj: Any, *, name: str) -> list[Mapping]:
    """把单个 bundle 或 bundle 序列统一成“bundle 列表”。

    作用：把 prediction_bundle / gt_bundle 统一为同一种内部表示，避免后续逻辑分别处理单个对象和多对象。
    参数：
    - bundle_obj: 可能是一个映射，也可能是多个映射组成的序列。
    - name: 用于错误信息，帮助指出是 prediction_bundle 还是 gt_bundle 出错。
    返回值：
    - list[Mapping]：统一后的 bundle 列表，每个元素都是一个映射。
    异常/失败条件：
    - 输入不是映射也不是映射序列时抛 TypeError。
    - 空序列抛 ValueError。
    状态变化：
    - 不修改原始对象，只构造新的列表视图或包装列表。
    """
    if isinstance(bundle_obj, Mapping):  # 单个映射时，直接包装成单元素列表，便于统一后续处理。
        return [bundle_obj]  # 保持原对象不变，只在外面套一层列表。
    if isinstance(bundle_obj, Sequence) and not isinstance(bundle_obj, (str, bytes)):  # 允许序列，但拒绝字符串和字节串。
        bundle_group = list(bundle_obj)  # 显式转成列表，后续会多次遍历。
        if not bundle_group:  # 空列表没有任何序列可处理。
            raise ValueError(f"{name} must be non-empty")  # 用 name 告知到底是哪类输入为空。
        if not all(isinstance(item, Mapping) for item in bundle_group):  # 序列中的每一项都必须是映射。
            raise TypeError(f"{name} must be a mapping or a sequence of mappings")  # 提醒输入类型不符合约定。

        bundle_keys = {"states", "timestamps", "diagnostics", "runtime_log", "seq_id", "scene_id"}  # 这些键表明该项已经是完整 bundle。
        if any(bundle_keys & set(item.keys()) for item in bundle_group):  # 只要某项带有 bundle 级字段，就认为它已经是 bundle。
            return bundle_group  # 直接返回原语义序列，不再包装成 states。
        return [{"states": bundle_group}]  # 否则把这组映射当成 states 序列封装成单个 bundle。
    raise TypeError(f"{name} must be a mapping or a sequence of mappings")  # 其他输入类型一律拒绝。


def _extract_traj(bundle_obj: Mapping[str, Any], *, name: str) -> list[dict]:
    """从 bundle 中提取轨迹点，并为缺失时间戳的点补上 timestamp。

    作用：把 bundle 内的 states 规范成轨迹点列表，确保每个点都可以继续用于对齐和误差计算。
    参数：
    - bundle_obj: 包含 states / timestamps 的映射。
    - name: 用于错误信息，指出当前是哪个 bundle。
    返回值：
    - list[dict]：每个元素都是一个轨迹点字典副本。
    异常/失败条件：
    - states 不是非字符串序列抛 TypeError。
    - states 为空抛 ValueError。
    - timestamps 提供但不是非字符串序列抛 TypeError。
    - timestamps 长度与 states 不一致抛 ValueError。
    - states 内有非映射元素抛 TypeError。
    状态变化：
    - 不修改原 bundle，只复制每个 state 到新 dict，并按需补充 timestamp。
    """
    states = bundle_obj.get("states")  # 从 bundle 里取出状态序列。
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):  # states 必须是非字符串序列。
        raise TypeError(f"{name}.states must be a non-string sequence")  # 说明 states 类型不对。
    if not states:  # 空轨迹没有后续计算意义。
        raise ValueError(f"{name}.states must be non-empty")  # 直接拒绝空输入。

    timestamps = bundle_obj.get("timestamps")  # 可选时间戳序列。
    if timestamps is not None:  # 只有显式提供时间戳时才做校验。
        if not isinstance(timestamps, Sequence) or isinstance(timestamps, (str, bytes)):  # 时间戳也必须是序列。
            raise TypeError(f"{name}.timestamps must be a non-string sequence when provided")  # 提示 timestamps 类型不对。
        if len(timestamps) != len(states):  # 时间戳必须与状态一一对应。
            raise ValueError(f"{name}.timestamps must align with {name}.states")  # 长度不一致就无法对齐。

    traj = []  # 用于保存规范后的轨迹点列表。
    for index, state in enumerate(states):  # 逐点检查并复制。
        if not isinstance(state, Mapping):  # 每个 state 都应是映射。
            raise TypeError(f"{name}.states[{index}] must be a mapping")  # 报出具体位置。
        point = dict(state)  # 复制一份，避免改动原始输入。
        if timestamps is not None and not any(key in point for key in TIME_KEY_CANDIDATES):  # 没有现成时间键时才补 timestamp（D9 单源引用常量，"index" 不是时间字段，与 trajectory_metrics.py 保持一致）。
            point["timestamp"] = timestamps[index]  # 从并行 timestamps 里补齐时间信息。
        traj.append(point)  # 收集规范后的轨迹点。
    return traj  # 返回可直接用于后续对齐的点列表。


def _align_prediction_and_ground_truth(pred_traj: Sequence[Mapping], gt_traj: Sequence[Mapping]) -> tuple[tuple[str, ...], list[tuple[int, Mapping, Mapping]], list[int]]:
    """把预测轨迹和真值轨迹按坐标与时间对齐。

    作用：先确认两个轨迹的坐标维度一致，再尽量按共同时间键对齐；如果没有共同时间键，就退化成位置对齐。
    参数：
    - pred_traj: 预测轨迹点列表。
    - gt_traj: 真值轨迹点列表。
    返回值：
    - (coord_keys, aligned_points) 其中 coord_keys 表示坐标键元组，aligned_points 保存对齐后的三元组列表，三元组结构为 (原始索引, 预测点, 对齐后的真值点)。
        - rejected_indices 保存因真值不可用而被硬拒绝的预测点原始索引列表。
    异常/失败条件：
    - 坐标键不一致、时间键重复、长度不匹配或无重叠时都会抛异常。
    状态变化：
    - 不修改输入，只构造新的对齐结果。
    """
    def _coerce_finite_float(value, *, name: str) -> float:
        """把值转成有限浮点数，守卫类型错误、溢出和非有限值。"""
        try:  # 尝试把值转成浮点数。
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:  # 不能转数值就报错；OverflowError 守卫超大整数（如 float(10**1000) 会抛 OverflowError）。
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(value):  # NaN 和无穷大不能参与对齐计算。
            raise ValueError(f"{name} must be finite")
        return value

    def _coord_keys(points: Sequence[Mapping], *, traj_name: str) -> tuple[str, ...]:
        """检查点集的坐标维度，并返回坐标键集合。"""
        if not all("px" in point and "py" in point for point in points):  # 每个点都必须至少有 px 和 py。
            raise ValueError(f"{traj_name} points must contain 'px' and 'py'")  # 缺少基础坐标时不能继续。
        point_has_pz = [("pz" in point) for point in points]  # 记录每个点是否带第三维 z。
        if any(point_has_pz) and not all(point_has_pz):  # 不能一部分点有 pz 一部分点没有。
            raise ValueError(f"{traj_name} points must use consistent coordinate fields")  # 坐标维度必须一致。
        has_pz = all(point_has_pz)  # 所有点都带 pz 时才算三维轨迹。
        return ("px", "py", "pz") if has_pz else ("px", "py")  # 按实际维度返回坐标键。

    coord_keys = _coord_keys(pred_traj, traj_name="prediction trajectory")  # 先检查预测轨迹的坐标维度。
    if _coord_keys(gt_traj, traj_name="ground-truth trajectory") != coord_keys:  # 真值和预测必须用同一坐标维度。
        raise ValueError("prediction and ground-truth trajectories must use the same coordinate fields")  # 维度不同会导致误差不可比。

    time_keys = TIME_KEY_CANDIDATES  # 时间字段候选（D9 单源引用常量，"index" 不是时间字段，不应作为时间键候选，否则会把位置序号误当时间键导致按值而非按位置对齐；与 trajectory_metrics.py U4 保持一致）。
    time_key = next(  # 逐个寻找两个轨迹都共有的时间字段。
        (
            key  # 当前候选时间键。
            for key in time_keys  # 依次检查所有候选键。
            if all(key in point for point in pred_traj) and all(key in point for point in gt_traj)  # 必须在两边都存在。
        ),
        None,  # 如果没有公共时间键，就返回 None 走位置对齐分支。
    )  # next 的返回值要么是公共时间键，要么是 None。

    if time_key is None:  # 没有公共时间键时，退化为逐位置对齐。
        if len(pred_traj) != len(gt_traj):  # 位置对齐要求长度完全一致。
            raise ValueError(
                "prediction_bundle and gt_bundle must have equal trajectory length when no shared time key exists"  # 长度不一致时不能靠索引对齐。
            )  # 长度不一致时不能靠索引对齐。
        # Phase 6 回归修复：早返回路径也按 L289 主路径 3 元组口径返回（coord_keys, aligned_points, rejected_indices=[]），
        # 与 L804 调用方 `coord_keys, aligned_points, rejected_indices = _align_prediction_and_ground_truth(...)` 严格匹配。
        return (
            coord_keys,
            [(index, pred_point, gt_point) for index, (pred_point, gt_point) in enumerate(zip(pred_traj, gt_traj))],  # 直接按位置配对返回。
            [],  # 早返回路径无 rejected_indices，返回空列表与 L289 主路径 3 元组口径一致。
        )

    def _validate_unique_keys(points: Sequence[Mapping], *, traj_name: str) -> None:
        """检查轨迹中的时间键是否重复，避免对齐歧义。"""
        seen = set()  # 记录已经见过的时间键值。
        for point in points:  # 逐点检查是否出现重复时间键。
            key_value = point[time_key]  # 取出当前点的时间键值。
            # D5 数值安全：时间键值必须可哈希（用作 set/dict key）；浮点值必须有限（NaN 会破坏去重和 dict 查找语义）。
            try:
                hash(key_value)  # 不可哈希类型（如 list/dict）会导致后续 set/dict 操作崩溃。
            except TypeError as exc:
                raise ValueError(f"{traj_name} '{time_key}' value must be hashable, got {type(key_value).__name__}.") from exc
            if isinstance(key_value, float) and not math.isfinite(key_value):  # NaN/Inf 作为 dict key 会破坏相等性语义。
                raise ValueError(f"{traj_name} '{time_key}' value must be finite, got {key_value!r}.")
            if key_value in seen:  # 已出现过就说明会造成对齐歧义。
                raise ValueError(f"{traj_name} contains duplicate '{time_key}' values")  # 直接拒绝重复键。
            seen.add(key_value)  # 记录当前键值，供后续重复检测。

    _validate_unique_keys(gt_traj, traj_name="ground-truth trajectory")  # 真值轨迹时间键必须唯一，否则对齐歧义。

    normalized_gt = _normalize_gt_rows(  # 把真值点规范成训练链路一致的行格式。
        {  # 单个真值点转成标准行字典。
            "timestamp": point[time_key],  # 用当前时间字段作为 timestamp。
            "px": point["px"],  # 保留 x 坐标。
            "py": point["py"],  # 保留 y 坐标。
            **({"yaw": point["yaw"]} if "yaw" in point else {}),  # 有 yaw 就一并保留，没有就不加。
        }
        for point in gt_traj  # 逐个真值点做标准化。
    )  # 规范化后的真值会继续喂给训练链路一致的对齐函数。

    # _normalize_gt_rows 只保留 timestamp/px/py/yaw，3D 轨迹的 pz 需要手动补回。
    if "pz" in coord_keys:
        gt_pz_by_timestamp = {  # 建立时间戳到 pz 值的映射。
            _coerce_finite_float(point[time_key], name=time_key): _coerce_finite_float(point["pz"], name="pz")  # 以时间戳为键，pz 值为值；用守卫函数防止类型/溢出/非有限值。
            for point in gt_traj  # 逐个真值点提取。
            if "pz" in point  # 只提取有 pz 的点。
        }
        for normalized_row in normalized_gt:  # 逐行补回 pz。
            ts = normalized_row["timestamp"]  # 取出当前行的时间戳。
            if ts in gt_pz_by_timestamp:  # 如果该时间戳有对应的 pz 值。
                normalized_row["pz"] = gt_pz_by_timestamp[ts]  # 把 pz 写回规范化行。

    # 为 pz 插值建立时间戳到规范化行的快速查找表。
    gt_row_by_timestamp = {}  # 时间戳到规范化行的映射。
    if "pz" in coord_keys:
        for row in normalized_gt:  # 逐行建立索引。
            gt_row_by_timestamp[row["timestamp"]] = row  # 以时间戳为键保存整行。

    aligned_points = []  # 保存对齐后的点三元组。
    rejected_indices: list[int] = []  # §2.4 硬拒绝索引列表。
    for index, pred_point in enumerate(pred_traj):  # 逐个预测点去真值里找对应位置。
        aligned_gt_row, alignment_info = _align_ground_truth(normalized_gt, _coerce_finite_float(pred_point[time_key], name=time_key))  # 按时间插值/对齐真值行，同时保留对齐信息；用守卫函数防止类型/溢出/非有限值。
        if aligned_gt_row is None:  # 对齐失败时（预测时间超出真值范围），§2.4 硬拒绝。
            reject_violation_if_missing_gt(pred_point)  # 硬拒绝标记。
            rejected_indices.append(index)  # 记录拒绝索引。
            continue  # 跳过无法对齐的点。

        # _align_ground_truth 的线性插值只输出 timestamp/px/py/yaw，3D 时需手动插值 pz。
        if "pz" in coord_keys and "pz" not in aligned_gt_row:
            mode = alignment_info.get("mode", "")  # 读取对齐模式。
            if mode == "linear_interpolation":  # 线性插值模式下需要手动补 pz。
                support_ts = alignment_info.get("support_timestamps", [])  # 取出支撑时间戳。
                if len(support_ts) == 2:  # 需要两个支撑点才能插值。
                    lower_row = gt_row_by_timestamp.get(support_ts[0])  # 下界行。
                    upper_row = gt_row_by_timestamp.get(support_ts[1])  # 上界行。
                    if (  # 两行都存在且都有 pz。
                        lower_row is not None
                        and upper_row is not None
                        and "pz" in lower_row
                        and "pz" in upper_row
                    ):
                        span = support_ts[1] - support_ts[0]  # 时间跨度。
                        if span > 0.0:  # 跨度大于 0 才能插值。
                            alpha = (_coerce_finite_float(pred_point[time_key], name=time_key) - support_ts[0]) / span  # 插值权重；用守卫函数防止类型/溢出/非有限值。
                            aligned_gt_row["pz"] = lower_row["pz"] + (upper_row["pz"] - lower_row["pz"]) * alpha  # 线性插值 pz。

        aligned_gt_point = dict(gt_traj[0])  # 以原始真值点结构作为模板，避免丢失同类字段。
        for key in coord_keys:  # 只替换坐标相关字段。
            aligned_gt_point[key] = aligned_gt_row[key]  # 把对齐后的坐标写回模板。
        if "yaw" in aligned_gt_row:  # 对齐后的 yaw 也需要写回，否则会保留模板的 yaw 值。
            aligned_gt_point["yaw"] = aligned_gt_row["yaw"]
        aligned_gt_point[time_key] = pred_point[time_key]  # 让对齐点保留预测侧时间键。
        if "timestamp" in aligned_gt_row:  # 如果标准化结果里有 timestamp，就同步保留。
            aligned_gt_point["timestamp"] = aligned_gt_row["timestamp"]
        aligned_points.append((index, pred_point, aligned_gt_point))  # 保存原索引、预测点、对齐后的真值点。
    if not aligned_points:  # 如果一个点都没对上，说明两条轨迹没有重叠。
        raise ValueError(f"prediction_bundle and gt_bundle do not overlap on '{time_key}'")
    return coord_keys, aligned_points, rejected_indices


def _compute_error_trace(
    coord_keys: tuple[str, ...],
    aligned_points: list[tuple[int, Mapping, Mapping]],
) -> tuple[list[float], list[int]]:
    """计算每个对齐点对的欧氏误差，并返回误差轨迹与原始索引。"""
    def _coerce_coordinate(value: Any, *, key: str) -> float:
        """把单个坐标值强制转成有限浮点数。"""
        if is_bool_like(value):  # 布尔值不能冒充数值。
            raise ValueError(f"trajectory coordinate '{key}' must be numeric")  # 布尔值直接拒绝。
        try:  # 尝试把值转成浮点数，统一后续计算口径。
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:  # 转换失败说明这个坐标不是合法数值；OverflowError 守卫超大整数（如 float(10**1000) 会抛 OverflowError）。
            raise ValueError(f"trajectory coordinate '{key}' must be numeric") from exc  # 把原始异常串起来方便定位。
        if not math.isfinite(value):  # 无穷大和 NaN 不能参与误差计算。
            raise ValueError(f"trajectory coordinate '{key}' must be finite")  # 非有限数值直接报错。
        return value  # 返回已经规范化的浮点数。

    error_trace = []  # 误差轨迹：每个对齐点对对应一个误差值。
    aligned_indices = []  # 保存这些误差对应的预测侧原始索引。
    for index, pred_point, gt_point in aligned_points:  # 逐个对齐点对计算误差。
        squared_error = math.fsum(  # 用 fsum 提高精度，避免 += 累加误差。
            (_coerce_coordinate(pred_point[key], key=key) - _coerce_coordinate(gt_point[key], key=key)) ** 2
            for key in coord_keys
        )  # 累加各坐标维度的平方差，保持欧氏距离定义。
        error_trace.append(math.sqrt(squared_error))  # 开方得到欧氏误差。
        aligned_indices.append(index)  # 记录该误差对应的原始预测索引。
    return error_trace, aligned_indices  # 一个给后续指标，一个给轨迹提取函数复用，后者用于所有按索引抽取的诊断轨迹。


def _extract_aligned_trace(
    prediction_obj: Mapping[str, Any],
    *,
    trace_name: str,
    aligned_indices: Sequence[int],
) -> list:
    """按对齐索引从预测对象里提取某条诊断轨迹。

    参数：
    - prediction_obj: 预测对象映射，可能包含 diagnostics 子映射或顶层轨迹字段。
    - trace_name: 要提取的诊断轨迹字段名。
    - aligned_indices: 与误差轨迹一致的对齐索引列表。
    返回值：
    - list：按 aligned_indices 顺序提取的轨迹值列表。
    异常/失败条件：
    - trace_name 在 diagnostics 和顶层都缺失时抛 KeyError。
    - trace_source 不是非字符串序列时抛 TypeError。
    - 轨迹为空时抛 ValueError。
    - 对齐索引越界（含负索引）时抛 ValueError。
    """
    diagnostics = prediction_obj.get("diagnostics")
    trace_source = None  # 先假设没有找到，后面逐层查找。
    if isinstance(diagnostics, Mapping) and trace_name in diagnostics:  # 优先从 diagnostics 里找。
        trace_source = diagnostics[trace_name]
    elif trace_name in prediction_obj:  # diagnostics 没有时，再退到对象顶层找。
        trace_source = prediction_obj[trace_name]

    if trace_source is None:  # 两处都没找到就直接报错。
        raise KeyError(f"prediction_bundle is missing '{trace_name}'")
    if not isinstance(trace_source, Sequence) or isinstance(trace_source, (str, bytes)):  # 轨迹必须是非字符串序列。
        raise TypeError(f"{trace_name} must be a non-string sequence")

    trace = list(trace_source)  # 统一转成列表，方便按索引抽取。
    if not trace:  # 空轨迹没有可提取数据。
        raise ValueError(f"{trace_name} must be non-empty")
    trace_length = len(trace)
    for index in aligned_indices:  # 逐个检查对齐索引是否越界。
        if index < 0 or index >= trace_length:  # 负索引和上界越界都拒绝，避免 Python 负索引静默从末尾取值。
            raise ValueError(
                f"{trace_name} aligned index {index} out of range for trace length {trace_length}"
            )
    return [trace[index] for index in aligned_indices]  # 只返回与对齐点一一对应的轨迹值。


def _extract_preferred_aligned_trace(
    prediction_obj: Mapping[str, Any],  # 当前预测对象，里面可能包含 diagnostics 和顶层轨迹字段。
    *,  # 强制这些参数只能按关键字传入，避免位置参数误配。
    preferred_trace_names: Sequence[str],  # 候选轨迹名的优先级列表。
    fallback_trace_name: str,  # 找不到候选时使用的默认轨迹名。
    aligned_indices: Sequence[int],  # 要抽取的对齐索引列表。
) -> list:  # 结束函数签名，下面开始写函数体。
    """优先从多个候选轨迹名里取诊断轨迹，取不到时回退到默认轨迹名。

    参数：
    - prediction_obj: 预测对象映射，可能包含 diagnostics 子映射或顶层轨迹字段。
    - preferred_trace_names: 候选诊断轨迹字段名的优先级列表，按顺序在 diagnostics 中试探。
    - fallback_trace_name: 所有候选都未命中时使用的兜底轨迹字段名。
    - aligned_indices: 与误差轨迹一致的对齐索引列表。
    返回值：
    - list：按 aligned_indices 顺序提取的轨迹值列表。
    异常/失败条件：
    - 候选命中 diagnostics 时，转交 _extract_aligned_trace 抛出 KeyError/TypeError/ValueError。
    - 所有候选均未命中且兜底轨迹名在 prediction_obj 中缺失时抛 KeyError。
    """
    diagnostics = prediction_obj.get("diagnostics")
    if isinstance(diagnostics, Mapping):  # 只有 diagnostics 是映射时才允许逐个候选查找。
        for trace_name in preferred_trace_names:  # 按优先级顺序试探各个轨迹名。
            if trace_name in diagnostics:  # 找到第一个可用候选就立即返回。
                return _extract_aligned_trace(
                    {"diagnostics": diagnostics},  # 只传 diagnostics，避免误读顶层同名字段。
                    trace_name=trace_name,  # 当前优先命中的候选轨迹名。
                    aligned_indices=aligned_indices,  # 只抽取和对齐点一致的索引。
                )
    return _extract_aligned_trace(
        prediction_obj,  # 所有候选都找不到时，使用默认回退轨迹名。
        trace_name=fallback_trace_name,  # 兜底轨迹名。
        aligned_indices=aligned_indices,  # 对齐后的索引列表。
    )


def _extract_scaling_trace(
    prediction_obj: Mapping[str, Any],
    *,
    aligned_indices: Sequence[int],
    prediction_length: int,
) -> list:
    """提取与对齐位置对应的缩放轨迹，优先使用已应用的缩放痕迹。"""
    diagnostics = prediction_obj.get("diagnostics")
    if not isinstance(diagnostics, Mapping):  # 没有 diagnostics 时无法重建缩放轨迹。
        return [None for _ in aligned_indices]

    if "applied_scaling_trace" in diagnostics:  # 优先使用统一的已应用缩放轨迹。
        return _extract_aligned_trace(
            {"diagnostics": diagnostics},  # 只把 diagnostics 交给统一提取函数。
            trace_name="applied_scaling_trace",  # 优先使用已应用的统一缩放轨迹名。
            aligned_indices=aligned_indices,  # 对齐后的索引列表。
        )

    if "scaling_trace" in diagnostics:  # 再回退到旧字段名 scaling_trace。
        return _extract_aligned_trace(
            {"diagnostics": diagnostics},  # 旧字段同样交给统一提取函数处理。
            trace_name="scaling_trace",  # 旧版统一缩放轨迹名。
            aligned_indices=aligned_indices,  # 对齐后的索引列表。
        )

    modalities_source = diagnostics.get("modalities")  # 每个点对应的模态来源。
    uwb_source = diagnostics.get("applied_uwb_scaling_trace")  # UWB 模态的应用后缩放轨迹。
    vio_source = diagnostics.get("applied_vio_scaling_trace")  # VIO 模态的应用后缩放轨迹。
    if uwb_source is None and vio_source is None:  # 如果没有应用后字段，就尝试旧字段。
        uwb_source = diagnostics.get("uwb_scaling_trace")
        vio_source = diagnostics.get("vio_scaling_trace")
    if modalities_source is None and uwb_source is None and vio_source is None:  # 三者都没有说明无法恢复缩放轨迹。
        return [None for _ in aligned_indices]

    if not isinstance(modalities_source, Sequence) or isinstance(modalities_source, (str, bytes)):  # modalities 必须是非字符串序列。
        raise TypeError("diagnostics.modalities must be a non-string sequence when scaling traces are provided")
    modalities = list(modalities_source)  # 转成列表便于按索引读取。
    if len(modalities) != prediction_length:  # modalities 长度必须和预测轨迹长度一致。
        raise ValueError("diagnostics.modalities must align with prediction trajectory length")

    def _optional_trace(trace_name: str) -> list | None:
        """读取一个可选轨迹字段，缺失时返回 None。"""
        trace_source = diagnostics.get(trace_name)
        if trace_source is None:  # 没有这个字段就返回 None。
            return None
        if not isinstance(trace_source, Sequence) or isinstance(trace_source, (str, bytes)):  # 有字段时也必须是序列。
            raise TypeError(f"{trace_name} must be a non-string sequence when provided")
        trace = list(trace_source)  # 先拷贝成列表，避免后续重复遍历原对象。
        if len(trace) != prediction_length:  # 所有缩放轨迹都必须与预测长度对齐。
            raise ValueError(f"{trace_name} must align with prediction trajectory length")
        return trace

    uwb_trace = _optional_trace("applied_uwb_scaling_trace")  # 优先尝试应用后 UWB 轨迹。
    if uwb_trace is None:  # 如果没有应用后字段，再尝试旧字段。
        uwb_trace = _optional_trace("uwb_scaling_trace")
    vio_trace = _optional_trace("applied_vio_scaling_trace")  # 优先尝试应用后 VIO 轨迹。
    if vio_trace is None:  # 如果没有应用后字段，再尝试旧字段。
        vio_trace = _optional_trace("vio_scaling_trace")
    scaling_trace = []  # 最终输出的对齐缩放轨迹。
    for index in aligned_indices:  # 只提取和误差轨迹一致的索引。
        modality = modalities[index]  # 读取这个点使用的模态。
        if modality == MODALITY_UWB:  # UWB 模态时取 UWB 轨迹。
            scaling_trace.append(None if uwb_trace is None else uwb_trace[index])
        elif modality == MODALITY_VIO:  # VIO 模态时取 VIO 轨迹。
            scaling_trace.append(None if vio_trace is None else vio_trace[index])
        else:  # 其他模态不参与缩放轨迹提取。
            scaling_trace.append(None)
    return scaling_trace


def _extract_measurement_mask(
    prediction_obj: Mapping[str, Any],
    *,
    aligned_indices: Sequence[int],
    prediction_length: int,
) -> list[bool]:
    """提取与对齐位置对应的测量级掩码，只保留真正可产生机制输出的模态。"""
    diagnostics = prediction_obj.get("diagnostics")
    if not isinstance(diagnostics, Mapping):  # 旧 bundle 没有 diagnostics 时保持全量兼容。
        return [True for _ in aligned_indices]
    modalities_source = diagnostics.get("modalities")
    if modalities_source is None:  # 缺少 modalities 时回退到旧行为，避免历史产物硬失败。
        return [True for _ in aligned_indices]
    if not isinstance(modalities_source, Sequence) or isinstance(modalities_source, (str, bytes)):
        raise TypeError("diagnostics.modalities must be a non-string sequence when provided")
    modalities = list(modalities_source)
    if len(modalities) != prediction_length:
        raise ValueError("diagnostics.modalities must align with prediction trajectory length")
    return [modalities[index] in {MODALITY_UWB, MODALITY_VIO} for index in aligned_indices]


def _mask_trace_with_measurement_support(
    trace: Sequence[float | None],
    measurement_mask: Sequence[bool],
) -> list[float | None]:
    """把非测量事件从机制轨迹里清空，保持长度不变以便和误差轨迹逐点对齐。"""
    if len(trace) != len(measurement_mask):
        raise ValueError("measurement_mask must align with the target trace length")
    return [value if is_measurement else None for value, is_measurement in zip(trace, measurement_mask)]


def _build_mechanism_support_context(measurement_mask: Sequence[bool]) -> dict[str, int]:
    """为 measurement-level 机制指标单独构造支持上下文。

    作用：把 measurement_mask 里 True 的数量作为 prediction/ground_truth/aligned
    三个支持长度，供 reliability_metrics._normalize_support_context 消费。

    参数：
        measurement_mask: 对齐后的测量事件掩码，元素为 bool（True 表示该位是
            真正的测量输出）。由 _extract_measurement_mask 产出，实际为 list[bool]。

    返回值：
        dict[str, int]：包含 prediction_length/ground_truth_length/aligned_length
        三个字段，三者均等于 measurement_mask 中 True 的数量。字段名与下游
        _normalize_support_context 的必需字段一致。
    """
    measurement_length = sum(1 for is_measurement in measurement_mask if is_measurement)
    return {
        "prediction_length": measurement_length,
        "ground_truth_length": measurement_length,
        "aligned_length": measurement_length,
    }


def _compact_measurement_trace(
    trace: Sequence[float | None],
    measurement_mask: Sequence[bool],
) -> list[float | None]:
    """把轨迹压缩成只包含 measurement-level 事件的序列。"""
    if len(trace) != len(measurement_mask):
        raise ValueError("measurement_mask must align with the target trace length")
    return [value for value, is_measurement in zip(trace, measurement_mask) if is_measurement]


def _compute_yaw_metrics(yaw_error_trace: Sequence[float | None]) -> dict[str, float]:
    """从航向角误差轨迹计算 yaw_rmse 和 yaw_p95 指标。

    参数：
        yaw_error_trace: 航向角绝对误差列表，元素为 float 或 None（缺失时）。

    返回值：
        dict[str, float]: 包含 yaw_rmse 和 yaw_p95 的指标字典。
    """
    valid_errors = [e for e in yaw_error_trace if e is not None and math.isfinite(e)]  # 过滤缺失值与非有限值（NaN/Inf），避免 NaN 污染 fsum 与破坏 sorted 顺序。
    if not valid_errors:  # 没有有效 yaw 误差样本时返回 0.0。
        return {"yaw_rmse": 0.0, "yaw_p95": 0.0}
    yaw_rmse = math.sqrt(math.fsum(e * e for e in valid_errors) / len(valid_errors))  # 航向角均方根误差。
    sorted_errors = sorted(valid_errors)  # 排序后计算分位数。
    yaw_p95 = _percentile_yaw(sorted_errors, 0.95)  # 航向角 95 百分位误差。
    return {"yaw_rmse": yaw_rmse, "yaw_p95": yaw_p95}


def _percentile_yaw(sorted_values: Sequence[float], percent: float) -> float:
    """线性插值分位数计算，与 tail_metrics._percentile 口径一致。

    参数:
        sorted_values: 已排序的数值列表（升序）。
        percent: 0 到 1 之间的分位比例，如 0.95 表示第 95 百分位。

    返回值:
        float: 分位数估计值。空列表返回 0.0，单元素列表返回该元素。
    """
    if not 0.0 <= percent <= 1.0:  # 分位比例必须在 [0, 1] 内，越界会导致位置越界或负索引外推。
        raise ValueError(f"percent must be in [0, 1], got {percent}")  # 越界直接拒绝。
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    position = (n - 1) * percent
    lower_index = int(position)
    upper_index = min(lower_index + 1, n - 1)
    weight = position - lower_index
    return sorted_values[lower_index] + (sorted_values[upper_index] - sorted_values[lower_index]) * weight


def _aggregate_gdop_occupancy(
    sequence_data_group: Sequence[Mapping[str, Any]],
    *,
    protocol_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """H24e 真改 §4.2 L894「差 GDOP 占比聚合」：跨序列聚合 gdop_occupancy_above_floor。

    H25c 真改：加 protocol_cfg kwarg 协议层显式冻结下界守门。
    若 protocol_cfg.evaluation.gdop_occupancy_min_ratio 设定（缺省 0.30）：
    - 当 count_measured > 0 且 ratio_above_floor < gdop_occupancy_min_ratio 时抛
      ValueError 阻断评估（与 §4.2 L894 协议层声明一致）。
    - 当 count_measured == 0（无序列携带 geometry_report）时跳过守门（向后兼容
      旧 prediction bundle 未携带 scenario_context.geometry_report 路径）。
    - protocol_cfg 为 None 或缺 evaluation.gdop_occupancy_min_ratio 字段时跳过守门
      （向后兼容旧调用方）。

    作用：把每个序列 support_context['gdop_occupancy_above_floor'] 收集起来，
    计算多序列聚合指标写入 support_report['gdop_occupancy_aggregation']。

    聚合口径：
    - per_sequence_values: list[float] —— 每个序列的 gdop_occupancy_above_floor（None 跳过）
    - mean: float —— 跨序列均值（仅统计非 None 序列）
    - min / max: float —— 跨序列极值
    - count_total: int —— 序列总数
    - count_measured: int —— 非 None 序列数（有 geometry_report 的序列）
    - count_above_floor: int —— gdop_occupancy_above_floor >= 1.0 的序列数
    - ratio_above_floor: float —— count_above_floor / count_measured
    - measured_ratio: float —— count_measured / count_total（覆盖率）
    - min_ratio_threshold: float | None —— protocol_cfg.evaluation.gdop_occupancy_min_ratio
      H25c 真改显式回写阈值到聚合报告，便于审计触发守门时的「阈值-观测值」对照

    返回值：
    - dict[str, Any]：聚合字段。若所有序列均无 gdop_occupancy_above_floor（count_measured=0），
      返回 None 让下游消费者视作"未测量"而非 0.0（与 §0 协议层缺省语义一致）。

    异常：
    - ValueError：当 protocol_cfg.evaluation.gdop_occupancy_min_ratio 设定且
      ratio_above_floor < 该阈值且 count_measured > 0 时抛出（H25c 真改协议层下界守门）。

    状态变化：
    - 不修改输入，只构造新的聚合字典返回。
    """
    if not sequence_data_group:
        return None
    per_sequence_values: list[float] = []
    count_total = len(sequence_data_group)
    count_measured = 0
    count_above_floor = 0
    for sequence_data in sequence_data_group:
        if not isinstance(sequence_data, Mapping):
            continue
        support_context = sequence_data.get("support_context")
        if not isinstance(support_context, Mapping):
            continue
        raw_value = support_context.get("gdop_occupancy_above_floor")
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(value):
            continue
        per_sequence_values.append(value)
        count_measured += 1
        if value >= 1.0:
            count_above_floor += 1
    if count_measured == 0:
        return None  # 无序列携带 geometry_report，下游视作"未测量"
    ratio_above_floor = count_above_floor / count_measured
    # H25c 真改：协议层显式冻结下界守门
    min_ratio_threshold: float | None = None
    if isinstance(protocol_cfg, Mapping):
        eval_cfg = protocol_cfg.get("evaluation")
        if isinstance(eval_cfg, Mapping):
            raw_threshold = eval_cfg.get("gdop_occupancy_min_ratio")
            if raw_threshold is not None:
                try:
                    min_ratio_threshold = float(raw_threshold)
                except (TypeError, ValueError, OverflowError):
                    min_ratio_threshold = None
    if (
        min_ratio_threshold is not None
        and math.isfinite(min_ratio_threshold)
        and min_ratio_threshold > 0.0
        and ratio_above_floor < min_ratio_threshold
    ):
        raise ValueError(
            f"gdop_occupancy ratio_above_floor={ratio_above_floor:.4f} < "
            f"protocol_cfg.evaluation.gdop_occupancy_min_ratio={min_ratio_threshold:.4f} "
            f"(count_measured={count_measured}, count_above_floor={count_above_floor}). "
            f"§4.2 L894 协议层下界守门触发：评估数据集 GDOP 占比未达协议层下界，"
            f"应回溯检查 anchor_layout 选取或扩充 evaluated scene 集。"
        )
    return {
        "per_sequence_values": per_sequence_values,
        "mean": math.fsum(per_sequence_values) / count_measured,
        "min": min(per_sequence_values),
        "max": max(per_sequence_values),
        "count_total": count_total,
        "count_measured": count_measured,
        "count_above_floor": count_above_floor,
        "ratio_above_floor": ratio_above_floor,
        "measured_ratio": count_measured / count_total,
        # H25c 真改：协议层下界阈值显式回写到聚合报告，便于审计触发守门时的「阈值-观测值」对照
        "min_ratio_threshold": min_ratio_threshold,
    }


def _build_support_report(
    prediction_length: int,
    ground_truth_length: int,
    aligned_length: int,
    valid_pair_count: int,
) -> dict[str, int | float | str]:
    """汇总支持信息，给外部检查样本覆盖情况和可靠性状态。

    作用：把预测/真值/对齐长度与有效配对数整理成固定字段的报告字典，
    供上层检查覆盖率与可靠性状态使用。

    参数：
        prediction_length: 原始预测轨迹长度（非负整数）。
        ground_truth_length: 原始真值轨迹长度（非负整数）。
        aligned_length: 实际对齐上的样本长度（非负整数）。
        valid_pair_count: 有效成对样本数（非负整数）。

    返回值：
        dict[str, int | float | str]：包含 prediction_length/ground_truth_length/
        aligned_length/valid_pair_count（int）、overlap_ratio（float，对齐长度与原始
        最大长度的比值）、reliability_status（str，"ok" 或 "insufficient_overlap"）
        六个字段的支持报告字典。

    异常/失败条件：
        - prediction_length 与 ground_truth_length 同时为零（即分母为零）时抛 ValueError。
    """
    denominator = max(prediction_length, ground_truth_length)  # 分母取两边长度里的较大值，避免覆盖率被放大。
    if denominator <= 0:  # 上游保证至少有一方非零；若双方都为零说明上游出了问题。
        raise ValueError(
            f"_build_support_report requires at least one non-zero length, "
            f"got prediction_length={prediction_length}, ground_truth_length={ground_truth_length}"
        )
    reliability_status = (  # 根据样本支持量决定可靠性状态。
        RELIABILITY_STATUS_INSUFFICIENT  # 对齐太短或有效对太少时，说明支持不足。
        if aligned_length < 2 or valid_pair_count < 2
        else RELIABILITY_STATUS_OK  # 其余情况下标记为正常。
    )  # 分支表达式结束。
    return {
        "prediction_length": prediction_length,  # 原预测长度。
        "ground_truth_length": ground_truth_length,  # 原真值长度。
        "aligned_length": aligned_length,  # 实际对齐上的长度。
        "valid_pair_count": valid_pair_count,  # 有效成对样本数。
        "overlap_ratio": aligned_length / denominator,  # 对齐长度与原始最大长度的比值。
        "reliability_status": reliability_status,  # 可靠性状态字符串。
    }  # 返回支持报告字典。


def reject_violation_if_missing_gt(pred_point):
    """§2.4 前提指导：真值不可用时硬拒绝预测点。"""
    pred_point["_hard_rejected"] = True
    pred_point["rejected_by"] = HARD_REJECT_REASON_MISSING_GT


def _resolve_runtime_source(prediction_obj: Mapping[str, Any]) -> Mapping[str, Any]:
    """从预测对象里找到运行时日志，优先顶层，其次 diagnostics。

    参数：
    - prediction_obj: 预测对象映射，可能包含顶层 runtime_log 或 diagnostics.runtime_log。
    返回值：
    - Mapping[str, Any]：找到的 runtime_log 映射，供 runtime 指标计算使用。
    异常/失败条件：
    - 顶层和 diagnostics 两处都没有 runtime_log 时抛 KeyError。
    """
    runtime_source = prediction_obj.get("runtime_log")  # 先读取顶层 runtime_log。
    diagnostics = prediction_obj.get("diagnostics")  # 再读取 diagnostics 作为备用来源。
    if runtime_source is None and isinstance(diagnostics, Mapping):  # 只有 diagnostics 是映射时才尝试备用路径。
        runtime_source = diagnostics.get("runtime_log")
    if runtime_source is None:  # 两条路径都没有就直接报错。
        raise KeyError("prediction_bundle is missing 'runtime_log'")
    return runtime_source  # 返回可供 runtime 指标计算使用的原始日志对象。


def _safe_get_timestamp(point: Mapping) -> float | None:
    """安全地从轨迹点提取 timestamp 浮点值, 缺失或非法时返回 None."""
    ts = point.get("timestamp")
    if ts is None:
        return None
    try:
        return float(ts)
    except (TypeError, ValueError, OverflowError):
        return None


def _collect_sequence_data(
    prediction_obj: Mapping[str, Any],
    gt_obj: Mapping[str, Any],
    *,
    bundle_index: int,
    protocol_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """把单个预测/真值配对整理成后续指标计算所需的中间数据。

    作用：对齐预测与真值轨迹，计算误差/航向角误差轨迹，并提取风险/偏差/缩放/测量掩码等中间字段，
    供 _compute_single_sequence_metrics 与多序列聚合消费者按字段名读取。

    参数：
    - prediction_obj: 单条预测 bundle 映射，需含 seq_id/scene_id/轨迹/runtime_log 等字段。
    - gt_obj: 单条真值 bundle 映射，需含 seq_id/轨迹等字段。
    - bundle_index: 当前配对在 bundle 列表中的位置，仅用于错误信息定位。

    返回值：
    - dict[str, Any]: 中间数据字典，包含 pred_traj/gt_traj/error_trace/yaw_error_trace/
      measurement_mask/mechanism_*_trace 系列/mechanism_error_trace_compact/risk_trace/
      bias_trace/scaling_trace/runtime_source/prediction_obj/support_context/
      mechanism_support_context 等字段，字段名与下游 sequence_data["..."] 消费者一致。

    异常/失败条件：
    - prediction_obj 或 gt_obj 不是 Mapping 时抛 TypeError。
    - 两侧 seq_id 都存在但不一致时抛 ValueError。
    - yaw 字段存在但不可转数值或非有限时抛 ValueError（D5 守卫，与 _coerce_finite_float 同口径）。
    - 轨迹提取/对齐/掩码等下游 helper 抛出的异常会向上传播。

    状态变化：
    - 不修改输入 prediction_obj/gt_obj，只构造新的中间数据字典返回。
    """
    if not isinstance(prediction_obj, Mapping) or not isinstance(gt_obj, Mapping):  # 两边都必须是映射。
        raise TypeError("prediction_bundle and gt_bundle entries must be mappings")

    prediction_seq_id = prediction_obj.get("seq_id")  # 读取预测序列 id。
    gt_seq_id = gt_obj.get("seq_id")  # 读取真值序列 id。
    if prediction_seq_id is not None and gt_seq_id is not None and prediction_seq_id != gt_seq_id:  # 两边都写了 id 且不一致时拒绝。
        raise ValueError(
            f"prediction_bundle[{bundle_index}] and gt_bundle[{bundle_index}] have different seq_id values"
        )

    # 提取 scene_id 用于场景变体级标注（不参与配对校验，仅透传）。
    prediction_scene_id = prediction_obj.get("scene_id")

    pred_traj = _extract_traj(prediction_obj, name=f"prediction_bundle[{bundle_index}]")  # 提取预测轨迹并补齐时间信息。
    gt_traj = _extract_traj(gt_obj, name=f"gt_bundle[{bundle_index}]")  # 提取真值轨迹并补齐时间信息。

    # §9.1 冷启动偏移统一划除: 协议要求每轨前 cold_start_offset_s 秒不计入有效计分段
    # (T_eff ≥ 20 s 量级). 全员统一, 禁止只删对某方法不利的起止段.
    # 移除 pred_traj/gt_traj 中 t < cold_start_threshold 的条目, 保持对齐.
    # 严格语义: 调用方传 partial protocol_cfg (缺 scene_scale 块) 时, 视为
    # 不启用冷启动划除 — 调用方只配置了 gdop/evaluation 等字段, 没有声明 §9.1.
    # 仅在 protocol_cfg 显式含 scene_scale.cold_start_offset_s 字段时启用划除.
    # 软约束保护: 若删除后任一轨迹将被清空 (例如合成 fixture 仅含 t<offset 的
    # 短段), 不执行删除并发出警告; 真实 §9 量级数据 (T_eff≥20s) 不会触发此保护.
    # §9.1 「全员同一 + 冷启动计入事先固定」守门: 协议字段 cold_start_global_enforced
    # 为 False 但 cold_start_offset_s > 0 时, 调用方须明确声明不参与统一划除并自行
    # 审计; 此处发出 warning 而非 raise, 以允许 §9.1 protocol-defined 软关闭路径.
    _cold_start_offset_s = 10.0  # §9 准则 37 默认值: 10 秒 warm-up.
    _cold_start_global_enforced = True  # §9.1 默认值: 全员统一划除.
    if protocol_cfg is not None and isinstance(protocol_cfg, Mapping):
        _scene_scale_partial = protocol_cfg.get('scene_scale')
        if isinstance(_scene_scale_partial, Mapping):
            if 'cold_start_offset_s' in _scene_scale_partial:
                try:
                    _cold_start_offset_s = float(_scene_scale_partial['cold_start_offset_s'])
                except (TypeError, ValueError, OverflowError):
                    _cold_start_offset_s = 10.0  # 解析失败时回退到准则 37 默认值.
            if 'cold_start_global_enforced' in _scene_scale_partial:
                _cold_start_global_enforced = bool(_scene_scale_partial['cold_start_global_enforced'])
    if _cold_start_offset_s > 0.0 and not _cold_start_global_enforced:
        import warnings as _warnings
        _warnings.warn(
            f"§9.1 cold_start_offset_s [{_cold_start_offset_s}s] 启用但 "
            f"cold_start_global_enforced=False — 调用方须明确声明不参与全员统一划除并自行审计, "
            f"按 §9.1 「禁止只删对某方法不利的起止段」, 上层须保证该 run 内所有方法均不接受统一划除.",
            UserWarning,
            stacklevel=2,
        )
    if _cold_start_offset_s > 0.0:
        _has_post_cold_start_pred = any(
            _safe_get_timestamp(_p) is not None and _safe_get_timestamp(_p) >= _cold_start_offset_s
            for _p in pred_traj
        )
        _has_post_cold_start_gt = any(
            _safe_get_timestamp(_p) is not None and _safe_get_timestamp(_p) >= _cold_start_offset_s
            for _p in gt_traj
        )
        if _has_post_cold_start_pred and _has_post_cold_start_gt:
            pred_traj, gt_traj = _trim_cold_start_segment(pred_traj, gt_traj, offset_s=_cold_start_offset_s)
        else:
            import warnings as _warnings
            _warnings.warn(
                f"§9.1 冷启动偏移 [{_cold_start_offset_s}s] 会清空轨迹 "
                f"(pred_traj 时长 < offset 或 gt_traj 时长 < offset); "
                f"跳过划除以保留链路可跑通. 真实 §9 量级数据 (T_eff≥20s) 不会触发此保护. "
                f"若 run 旨在主结论比较, 须将轨迹时长提升到 cold_start_offset_s 之后仍 ≥ t_eff_min_s.",
                UserWarning,
                stacklevel=2,
            )

    # §9.1 「T_eff ≥ 20s 量级」守门: 在冷启动划除之后,
    # 检查剩余段时长是否 ≥ t_eff_min_s (协议 scene_scale.t_eff_min_s, 默认 20.0).
    # 守门仅在 protocol_cfg 显式含 scene_scale.t_eff_min_s 时启用 (避免短 fixture 误触发).
    # 软约束: 短轨 fixture 触发时不 raise, 仅警告; 真实 §9 量级数据须满足该门.
    _t_eff_min_s = _resolve_t_eff_min_s(protocol_cfg)
    if _t_eff_min_s is not None and _t_eff_min_s > 0.0 and len(pred_traj) > 0:
        _t_eff_actual = _compute_traj_duration_s(pred_traj)
        if _t_eff_actual is not None and _t_eff_actual < _t_eff_min_s:
            import warnings as _warnings
            _warnings.warn(
                f"§9.1 T_eff 守门违反: pred_traj 实际有效段时长 [{_t_eff_actual:.3f}s] < "
                f"t_eff_min_s [{_t_eff_min_s}s] (已扣除冷启动); 真实 §9 主结论 run 须 ≥ "
                f"{_t_eff_min_s}s. 软约束: 仅警告, 不阻断; 若 run 旨在主结论比较, 须延长轨迹时长.",
                UserWarning,
                stacklevel=2,
            )

    coord_keys, aligned_points, rejected_indices = _align_prediction_and_ground_truth(pred_traj, gt_traj)  # 做轨迹对齐。
    if rejected_indices:
        from liquidloc.common.tee_logger import print_dict
        print_dict(
            {
                "rejected_count": len(rejected_indices),
                "sample_rejected_indices": rejected_indices[:5],
            },
            "GT hard-drop (§2.4) enforcement",
            prefix="[§2.4]",
        )
    aligned_pred_traj = [pred_point for _, pred_point, _ in aligned_points]  # 从三元组里取出预测侧点。
    aligned_gt_traj = [gt_point for _, _, gt_point in aligned_points]  # 从三元组里取出真值侧点。
    error_trace, aligned_indices = _compute_error_trace(coord_keys, aligned_points)  # 计算误差轨迹和原始索引。

    # 计算 yaw 误差轨迹：使用 angle_delta_rad 处理角度环绕。
    yaw_error_trace = []  # 保存每个对齐点的航向角绝对误差。
    for _, pred_point, gt_point in aligned_points:  # 逐个对齐点计算航向角误差。
        pred_yaw = pred_point.get("yaw")
        gt_yaw = gt_point.get("yaw")
        if pred_yaw is not None and gt_yaw is not None:  # 两侧都有 yaw 时才计算。
            # D5 数值安全：yaw 可能是字符串/超大整数等，裸 float() 会抛 TypeError/ValueError/OverflowError 或产生 NaN/Inf；与 _coerce_finite_float 同口径守卫，先转浮点再校验有限性。
            try:
                pred_yaw_f = float(pred_yaw)
                gt_yaw_f = float(gt_yaw)
            except (TypeError, ValueError, OverflowError) as exc:  # 非数值类型或超大整数无法转换。
                raise ValueError(f"prediction_bundle[{bundle_index}] yaw must be numeric") from exc
            if not (math.isfinite(pred_yaw_f) and math.isfinite(gt_yaw_f)):  # NaN/Inf 会污染 angle_delta_rad 并向下游传播。
                raise ValueError(f"prediction_bundle[{bundle_index}] yaw must be finite")
            yaw_delta = abs(angle_delta_rad(gt_yaw_f, pred_yaw_f))  # 角度差绝对值，正确处理 [-π, π) 环绕。
            yaw_error_trace.append(yaw_delta)
        else:  # 缺少 yaw 时无法计算航向误差。
            yaw_error_trace.append(None)

    prediction_length = len(prediction_obj.get('states') or prediction_obj.get('timestamps') or pred_traj)  # 记录原始预测轨迹长度 (剔除前).
    ground_truth_length = len(gt_traj)  # 记录真值轨迹长度。
    aligned_length = len(aligned_points)  # 记录实际对齐成功的长度。
    measurement_mask = _extract_measurement_mask(
        prediction_obj,
        aligned_indices=aligned_indices,
        prediction_length=prediction_length,
    )
    # §9 细节：起飞/降落与室内外/楼层切换制度 (§9 「全员同一处理或排除出主计分段」).
    # 在 measurement_mask 层级按协议政策把对应帧设为 False，使该帧不计入计分段.
    # exclude_unified + mask=True 的帧从 measurement_mask 中剔除；include_unified /
    # manual_guard 不修改 (调用方需自行审计). 缺 mask 字段时不修改.
    measurement_mask = _apply_takeoff_landing_floor_transition_to_measurement_mask(
        measurement_mask,
        aligned_indices=aligned_indices,
        prediction_obj=prediction_obj,
        protocol_cfg=protocol_cfg,
    )
    risk_trace = _extract_preferred_aligned_trace(  # 提取风险轨迹，优先用已应用版本。
        prediction_obj,  # 当前预测对象。
        preferred_trace_names=("applied_risk_trace",),  # 优先候选风险轨迹名。
        fallback_trace_name="risk_trace",  # 默认兜底风险轨迹名。
        aligned_indices=aligned_indices,  # 与误差轨迹一致的索引。
    )
    bias_trace = _extract_preferred_aligned_trace(  # 提取偏差轨迹，优先用已应用版本。
        prediction_obj,  # 当前预测对象。
        preferred_trace_names=("applied_bias_trace",),  # 优先候选偏差轨迹名。
        fallback_trace_name="bias_trace",  # 默认兜底偏差轨迹名。
        aligned_indices=aligned_indices,  # 与误差轨迹一致的索引。
    )
    scaling_trace = _extract_scaling_trace(  # 提取缩放轨迹，内部会处理多种来源字段。
        prediction_obj,  # 当前预测对象。
        aligned_indices=aligned_indices,  # 与误差轨迹一致的索引。
        prediction_length=prediction_length,  # 预测轨迹总长度，用于校验缩放轨迹对齐。
    )
    mechanism_error_trace = _mask_trace_with_measurement_support(error_trace, measurement_mask)
    risk_trace = _mask_trace_with_measurement_support(risk_trace, measurement_mask)
    bias_trace = _mask_trace_with_measurement_support(bias_trace, measurement_mask)
    scaling_trace = _mask_trace_with_measurement_support(scaling_trace, measurement_mask)
    mechanism_risk_trace = _compact_measurement_trace(risk_trace, measurement_mask)
    mechanism_bias_trace = _compact_measurement_trace(bias_trace, measurement_mask)
    mechanism_scaling_trace = _compact_measurement_trace(scaling_trace, measurement_mask)
    mechanism_error_trace_compact = _compact_measurement_trace(mechanism_error_trace, measurement_mask)
    runtime_source = _resolve_runtime_source(prediction_obj)  # 统一找出 runtime_log 的来源。

    # H24e 真改 §4.2 L894「差 GDOP 占比聚合」：从 prediction_obj 提取 geometry_report
    # 的 gdop_above_floor_ratio 字段。 prediction_obj 在 core_pipeline.py:1033 通过
    # bundle['scenario_context'] = deepcopy(scenario_context) 注入，scenario_context
    # 由 _resolve_scene_context_for_task（core_pipeline.py:192）从 quick_scene_context_*
    # 拷贝而来，含 geometry_report（core_pipeline.py:521 / 真改路径）。
    # gdop_above_floor_ratio 在 geometry_levels.py:410 由 gdop_value >= gdop_floor 计算
    # （单布局层 0/1），评估层按轨迹时序聚合为占比（0.0~1.0）。
    # 缺失时为 None，下游 _aggregate_gdop_occupancy 跳过该序列不参与聚合（向后兼容）。
    gdop_occupancy_above_floor = None
    _scenario_ctx = prediction_obj.get('scenario_context') if isinstance(prediction_obj, Mapping) else None
    if isinstance(_scenario_ctx, Mapping):
        _geom_report = _scenario_ctx.get('geometry_report')
        if isinstance(_geom_report, Mapping):
            _raw_ratio = _geom_report.get('gdop_above_floor_ratio')
            if _raw_ratio is not None:
                try:
                    gdop_occupancy_above_floor = float(_raw_ratio)
                except (TypeError, ValueError, OverflowError):
                    gdop_occupancy_above_floor = None

    return {
        "prediction_obj": prediction_obj,  # 原始预测对象，后续需要时可追溯。
        "pred_traj": aligned_pred_traj,  # 对齐后的预测轨迹。
        "gt_traj": aligned_gt_traj,  # 对齐后的真值轨迹。
        "error_trace": error_trace,  # 对齐后的误差轨迹。
        "yaw_error_trace": yaw_error_trace,  # 对齐后的航向角误差轨迹。
        "measurement_mask": measurement_mask,  # 对齐后的测量事件掩码。
        "mechanism_error_trace": mechanism_error_trace,  # 仅在测量事件上保留误差，供机制指标消费。
        "mechanism_error_trace_compact": mechanism_error_trace_compact,  # 压缩到 measurement-only 的误差轨迹。
        "mechanism_risk_trace": mechanism_risk_trace,  # 压缩到 measurement-only 的风险轨迹。
        "mechanism_bias_trace": mechanism_bias_trace,  # 压缩到 measurement-only 的偏差轨迹。
        "mechanism_scaling_trace": mechanism_scaling_trace,  # 压缩到 measurement-only 的缩放轨迹。
        "risk_trace": risk_trace,  # 对齐后的风险轨迹。
        "bias_trace": bias_trace,  # 对齐后的偏差轨迹。
        "scaling_trace": scaling_trace,  # 对齐后的缩放轨迹。
        "runtime_source": runtime_source,  # 原始 runtime 日志。
        "support_context": {
            "prediction_length": prediction_length,  # 预测侧原始长度。
            "ground_truth_length": ground_truth_length,  # 真值侧原始长度，不被 max 放大。
            "aligned_length": aligned_length,  # 实际对齐长度。
            "scene_id": prediction_scene_id,  # 场景变体标注，供上层消费者做场景级聚合。
            # H24e 真改 §4.2 L894「差 GDOP 占比聚合」：本序列的 gdop_above_floor_ratio
            # （单布局层 0/1），多序列聚合由 _aggregate_gdop_occupancy 计算占比。
            # None 表示该序列无 geometry_report，下游聚合时跳过（向后兼容旧 bundle）。
            "gdop_occupancy_above_floor": gdop_occupancy_above_floor,
        },  # support_context 小字典结束。
        "mechanism_support_context": _build_mechanism_support_context(measurement_mask),  # 机制指标单独的 measurement-level 支持口径。
    }  # 返回支持报告字典。


def _build_metric_table(
    merged_blocks: Mapping[str, Any],
    metric_order: Sequence[str],
) -> dict[str, Any]:
    """按 metric_order 固定顺序拼出最终 metric_table。

    作用：把已合并的各模块指标块按协议冻结顺序重建为最终指标表，确保输出字段顺序稳定。

    参数：
        merged_blocks: 已合并的各模块指标块映射，键为指标名，值为指标值。
        metric_order: 协议冻结的指标字段顺序列表，用于保证输出表字段顺序一致。

    返回值：
        dict[str, Any]：按 metric_order 排列的最终指标表。

    异常：
        ValueError: 当 merged_blocks 缺少 metric_order 中要求的指标时抛出。
    """
    missing_metrics = [metric_name for metric_name in metric_order if metric_name not in merged_blocks]  # 找出协议要求但还没合并到的指标。
    if missing_metrics:  # 任何必需指标缺失都不允许继续。
        raise ValueError(f"metric blocks are missing required schema metrics: {missing_metrics}")
    return {metric_name: merged_blocks[metric_name] for metric_name in metric_order}  # 按协议顺序重建最终表。


def _project_aligned_traj_for_trajectory_metrics(traj: Sequence[Mapping]) -> list[dict]:
    """从已对齐的轨迹点中只保留坐标字段，去掉时间戳等对齐键。  # 函数作用总述。

    作用：metric_runner 已经完成了预测与真值的时间对齐，传给 trajectory_metrics 时  # 说明为什么需要这个函数。
    不应再触发 trajectory_metrics 内部的二次对齐逻辑，因此只保留 px/py/pz 坐标字段，  # 说明投影的目的。
    让后续的 trajectory_metrics 按索引逐点比较即可。  # 说明投影后的使用方式。
    参数：  # 参数说明。
    - traj: 已对齐的轨迹点列表，每个点是包含 px/py（和可选 pz）及其他字段的映射。  # 参数说明。
    返回值：  # 返回值说明。
    - list[dict]：只包含坐标字段的新轨迹点列表，与输入按索引一一对应。  # 返回值说明。
    状态变化：  # 状态变化说明。
    - 不修改输入轨迹，只构造新的字典列表。  # 不改状态。
    """
    return [{key: point[key] for key in ("px", "py", "pz") if key in point} for point in traj]  # 逐点只保留存在的坐标键，丢弃时间戳等字段。


def _compute_single_sequence_metrics(
    sequence_data: Mapping[str, Any],
    *,
    metric_order: Sequence[str],
    failure_threshold: float,
    protocol_cfg: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """计算单序列场景下的所有指标块，并返回最终表和支持信息。  # 函数作用总述。

    作用：把单序列的中间数据分别喂给四类指标计算函数，再把结果按协议顺序合并成统一指标表。  # 详细作用。
    参数：  # 参数说明。
    - sequence_data: 由 _collect_sequence_data 返回的中间数据字典，包含 pred_traj/gt_traj/error_trace/risk_trace/bias_trace/scaling_trace/runtime_source/support_context。  # 参数说明。
    - metric_order: 协议冻结的指标字段顺序列表，用于保证输出表字段顺序一致。  # 参数说明。
    - failure_threshold: 尾部指标的失败阈值，单位米。  # 参数说明。
    返回值：  # 返回值说明。
    - tuple[dict[str, float], dict[str, Any]]：第一个是按 metric_order 排列的指标表，第二个是支持信息报告。  # 返回值说明。
    状态变化：  # 状态变化说明。
    - 不修改输入，只构造新的指标表和支持报告。  # 不改状态。
    """
    trajectory_block = compute_trajectory_metrics(  # 先算轨迹误差块。
        _project_aligned_traj_for_trajectory_metrics(sequence_data["pred_traj"]),  # 已对齐轨迹按索引比较，避免重复事件时间戳触发二次对齐。
        _project_aligned_traj_for_trajectory_metrics(sequence_data["gt_traj"]),  # 对真值侧做同样投影，保持与上面一致的比较口径。
    )
    ate_degraded = trajectory_block.pop("ate_degraded", False)  # 从指标字典中取出退化标志，不进入metric_table。
    # 准则 36：单序列路径下 mean_rmse = rmse（无跨序列聚合意义），std_rmse = 0.0。
    trajectory_block["mean_rmse"] = trajectory_block["rmse"]
    trajectory_block["std_rmse"] = 0.0
    tail_block, tail_support = compute_tail_metrics(sequence_data["error_trace"], failure_threshold=failure_threshold)  # 再算尾部指标。
    reliability_block, reliability_support = compute_reliability_metrics(  # 计算风险与误差之间的可靠性指标。
        sequence_data["mechanism_risk_trace"],  # 风险轨迹作为左侧输入，只保留 measurement-level 样本。
        sequence_data["mechanism_error_trace_compact"],  # 误差轨迹作为右侧输入，只保留 measurement-level 样本。
        support_context=sequence_data["mechanism_support_context"],  # 机制指标使用 measurement-level 支持口径。
    )
    reliability_block["bias_alignment"] = compute_trace_correlation(  # 额外补上偏差对误差的相关性。
        sequence_data["mechanism_bias_trace"],  # 偏差轨迹作为相关性左侧输入，只保留 measurement-level 样本。
        sequence_data["mechanism_error_trace_compact"],  # 误差轨迹作为相关性右侧输入，只保留 measurement-level 样本。
        lhs_name="bias_trace",  # 用于错误信息的轨迹名。
    )
    reliability_block["corr_scaling_error"] = compute_trace_correlation(  # 再补上缩放对误差的相关性。
        sequence_data["mechanism_scaling_trace"],  # 缩放轨迹作为相关性左侧输入，只保留 measurement-level 样本。
        sequence_data["mechanism_error_trace_compact"],  # 误差轨迹作为相关性右侧输入，只保留 measurement-level 样本。
        lhs_name="scaling_trace",  # 用于错误信息的轨迹名。
    )
    runtime_block = compute_runtime_metrics(sequence_data["runtime_source"])  # 直接把运行时日志转成运行时指标。

    # 计算 yaw 误差指标：yaw_rmse 和 yaw_p95。
    yaw_block = _compute_yaw_metrics(sequence_data["yaw_error_trace"])  # 从航向角误差轨迹计算航向指标。

    support_report = _build_support_report(  # 汇总样本支持信息，供上层检查覆盖率。
        sequence_data["support_context"]["prediction_length"],  # 预测长度。
        sequence_data["support_context"]["ground_truth_length"],  # 真值长度。
        sequence_data["support_context"]["aligned_length"],  # 对齐长度。
        reliability_support["valid_pair_count"],  # 有效成对样本数（从support_report取）。
    )
    support_report["measurement_mask"] = list(sequence_data.get("measurement_mask", []))  # §9 制度过滤后的测量级掩码，供上层审计哪些帧计入计分段。
    support_report["ate_degraded"] = ate_degraded  # ATE退化标志放入support_report，不进入metric_table。
    support_report["long_failure_segments"] = tail_support["long_failure_segments"]  # 连续失败段放入support_report。
    support_report["reliability_status"] = reliability_support["reliability_status"]  # 可靠性状态放入support_report。
    # H24e 真改 §4.2 L894「差 GDOP 占比聚合」：单序列场景下也加 gdop_occupancy_aggregation
    # 字段（只含本序列 1 个值），保持 multi_sequence 与 single_sequence 字段口径一致。
    support_report["gdop_occupancy_aggregation"] = _aggregate_gdop_occupancy(
        [sequence_data], protocol_cfg=protocol_cfg
    )
    # §9.3 单轨脉冲/异步量级门检查：n_pulse≥30 / n_async≥20。
    # measurement_mask 中 True 的数量即为有效 UWB 脉冲数；
    # n_async 从 prediction_obj 的 scenario_context.async_report.dropped_event_count
    # 读取（异步扰动删除的事件数即显著跨模态异步事件数）。
    _emit_section9_pulse_async_warnings(support_report, sequence_data, protocol_cfg=protocol_cfg)

    merged_blocks = {}  # 用一个总字典承接各模块输出。
    merged_blocks.update(trajectory_block)  # 先合并轨迹指标。
    merged_blocks.update(tail_block)  # 再合并尾部指标。
    merged_blocks.update(yaw_block)  # 合并航向角指标。
    merged_blocks.update(reliability_block)  # 再合并可靠性指标。
    merged_blocks.update(runtime_block)  # 最后合并运行时指标。
    return _build_metric_table(merged_blocks, metric_order), support_report


def _compute_pooled_runtime_metrics(sequence_data_group: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """把多序列的运行时日志汇总成一个统一 runtime 指标块。  # 函数作用总述。

    作用：将多个序列的 latency 样本拼接，params 要求所有序列一致，ram_peak 取最大值，  # 详细作用。
    然后交给 compute_runtime_metrics 统一计算运行时指标。  # 说明最终输出方式。
    参数：  # 参数说明。
    - sequence_data_group: 多序列中间数据列表，每个元素包含 runtime_source 字段。  # 参数说明。
    返回值：  # 返回值说明。
    - dict[str, float]：汇总后的运行时指标字典。  # 返回值说明。
    异常/失败条件：  # 失败条件说明。
    - 不同序列的 params 不一致时抛 ValueError。  # 一致性校验失败。
    状态变化：  # 状态变化说明。
    - 不修改输入，只构造新的汇总 runtime_log 并计算指标。  # 不改状态。
    """
    pooled_latency = []  # 汇总所有序列的延迟样本。
    params_values = []  # 收集每个序列的参数量，后面用来检查一致性。
    ram_peak_values = []  # 收集每个序列的峰值内存。
    for sequence_data in sequence_data_group:  # 逐序列累积运行时信息。
        runtime_source = sequence_data["runtime_source"]  # 读取当前序列的原始 runtime 日志。
        compute_runtime_metrics(runtime_source)  # 先让单序列格式校验通过，确保数据结构合法。
        pooled_latency.extend(runtime_source["latency"])  # 把每个序列的 latency 拼接起来。
        try:  # 极大的 Real 转 float 可能溢出，与 compute_runtime_metrics 守卫口径对齐。
            params_value = float(runtime_source["params"])  # 统一转成浮点数。
        except (OverflowError, TypeError, ValueError) as exc:  # 与 coerce_finite_scalar 守卫口径对齐。
            raise ValueError(f"runtime params must be a finite numeric value, got {runtime_source['params']!r}") from exc
        params_values.append(params_value)  # 记录参数量，后面要求所有序列一致。
        try:  # 极大的 Real 转 float 可能溢出，与 compute_runtime_metrics 守卫口径对齐。
            ram_peak_value = float(runtime_source["ram_peak"])  # 统一转成浮点数。
        except (OverflowError, TypeError, ValueError) as exc:  # 与 coerce_finite_scalar 守卫口径对齐。
            raise ValueError(f"runtime ram_peak must be a finite numeric value, got {runtime_source['ram_peak']!r}") from exc
        ram_peak_values.append(ram_peak_value)  # 记录峰值内存，用于取总体最大值。

    reference_params = params_values[0]  # 以第一条序列的参数量作为比较基准。
    if any(param != reference_params for param in params_values[1:]):  # 任何序列参数量不同都不允许合并。
        raise ValueError(f"runtime params must match across sequences, got {params_values}")

    return compute_runtime_metrics(
        {
            "latency": pooled_latency,  # 汇总后的总延迟样本。
            "params": reference_params,  # 所有序列必须共享同一个参数量。
            "ram_peak": max(ram_peak_values),  # 总体峰值内存取最大值。
        }  # 构造新的 runtime_log 供统一校验。
    )  # 再交给单序列 runtime 指标计算。


def _compute_pooled_tail_metrics(sequence_data_group: Sequence[Mapping[str, Any]], *, failure_threshold: float) -> tuple[dict[str, float], dict[str, list[tuple[int, int]]]]:
    """汇总多序列的尾部指标，但不跨序列边界合并失败段。  # 函数作用总述。

    作用：把每个序列的 error_trace 拼成全局轨迹后统一计算尾部指标，  # 详细作用。
    长失败段则按各自序列内部的 (start, end) 加上累计偏移后汇总，  # 说明长失败段处理方式。
    避免跨序列边界把首尾失败段误连成更长的失败段。  # 说明为何不复用全局轨迹。
    参数：  # 参数说明。
    - sequence_data_group: 多序列中间数据列表，每个元素需包含 error_trace 字段。  # 参数说明。
    - failure_threshold: 尾部指标的失败阈值，单位米。  # 参数说明。
    返回值：  # 返回值说明。
    - tuple[dict[str, float], dict[str, list[tuple[int, int]]]]：(尾部指标字典, 支撑报告)；尾部指标字典含 p95/p99/failure_rate，支撑报告含 long_failure_segments。  # 返回值说明。
    异常/失败条件：  # 失败条件说明。
    - failure_threshold 或 error_trace 非法时由 compute_tail_metrics 抛出对应异常。  # 委托校验。
    状态变化：  # 状态变化说明。
    - 不修改输入，只构造新的尾部指标字典和支撑报告。  # 不改状态。
    """
    pooled_error_trace = []  # 拼接所有序列的误差轨迹。
    pooled_failure_segments = []  # 收集各序列内部的失败段（已加偏移）。
    offset = 0  # 累计偏移，用于把各序列局部下标换算成全局下标。
    for sequence_data in sequence_data_group:
        error_trace = sequence_data["error_trace"]  # 当前序列的误差轨迹。
        pooled_error_trace.extend(error_trace)  # 拼接到全局轨迹。
        _, sequence_tail_support = compute_tail_metrics(error_trace, failure_threshold=failure_threshold)  # 只需长失败段，丢弃序列级尾部指标块。
        pooled_failure_segments.extend(
            (segment_start + offset, segment_end + offset)
            for segment_start, segment_end in sequence_tail_support["long_failure_segments"]
        )
        offset += len(error_trace)  # 整数索引累加，无需浮点精度保护。
    pooled_tail_block, _ = compute_tail_metrics(pooled_error_trace, failure_threshold=failure_threshold)  # 全局尾部指标，丢弃全局失败段。
    # long_failure_segments 通过 support_report 传递，不放入 metric_table。
    pooled_tail_support = {"long_failure_segments": pooled_failure_segments}
    return pooled_tail_block, pooled_tail_support


def _compute_multi_sequence_metrics(
    sequence_data_group: Sequence[Mapping[str, Any]],
    *,
    metric_order: Sequence[str],
    failure_threshold: float,
    protocol_cfg: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """计算多序列场景下的聚合指标，并保持协议顺序输出。  # 函数作用总述。

    作用：把多个序列的误差/风险/偏差/缩放轨迹展平后统一计算四类指标，  # 详细作用。
    轨迹指标从全局误差平方和直接计算，尾部和可靠性指标用展平后的轨迹，  # 说明各块的聚合方式。
    runtime 指标走 _compute_pooled_runtime_metrics 汇总。  # runtime 的特殊处理。
    参数：  # 参数说明。
    - sequence_data_group: 多序列中间数据列表。  # 参数说明。
    - metric_order: 协议冻结的指标字段顺序列表。  # 参数说明。
    - failure_threshold: 尾部指标的失败阈值，单位米。  # 参数说明。
    - protocol_cfg: H25c 真改 §4.2 L894 配置层下界守门配置（gdop_occupancy_min_ratio），
      None 时跳过守门（向后兼容）。  # 参数说明。
    返回值：  # 返回值说明。
    - tuple[dict[str, float], dict[str, Any]]：第一个是按 metric_order 排列的指标表，第二个是支持信息报告。  # 返回值说明。
    状态变化：  # 状态变化说明。
    - 不修改输入，只构造新的指标表和支持报告。  # 不改状态。
    """
    # 先为每个序列计算 trajectory_metrics，供 ATE/RPE 聚合使用。结果存入局部列表，不修改输入。
    any_ate_degraded = False  # 跟踪是否有任何序列的ATE退化。
    sequence_trajectory_metrics: list[dict[str, Any]] = []  # 逐序列缓存轨迹指标，避免修改输入。
    for sequence_data in sequence_data_group:
        seq_metrics = compute_trajectory_metrics(
            _project_aligned_traj_for_trajectory_metrics(sequence_data["pred_traj"]),
            _project_aligned_traj_for_trajectory_metrics(sequence_data["gt_traj"]),
        )
        # 读取 ate_degraded 标志但不修改指标字典，避免 ate_degraded 进入 metric_table。
        if seq_metrics.get("ate_degraded", False):
            any_ate_degraded = True
        sequence_trajectory_metrics.append(seq_metrics)

    pooled_error_trace = [error for sequence_data in sequence_data_group for error in sequence_data["error_trace"]]  # 展平所有误差轨迹。
    pooled_mechanism_error_trace = [
        error for sequence_data in sequence_data_group for error in sequence_data["mechanism_error_trace_compact"]
    ]  # 展平 measurement-level 误差轨迹。
    pooled_risk_trace = [
        risk for sequence_data in sequence_data_group for risk in sequence_data["mechanism_risk_trace"]
    ]  # 展平 measurement-level 风险轨迹。
    pooled_bias_trace = [
        bias for sequence_data in sequence_data_group for bias in sequence_data["mechanism_bias_trace"]
    ]  # 展平 measurement-level 偏差轨迹。
    pooled_scaling_trace = [
        value for sequence_data in sequence_data_group for value in sequence_data["mechanism_scaling_trace"]
    ]  # 展平 measurement-level 缩放轨迹。
    support_context = {
        "prediction_length": sum(
            sequence_data["support_context"]["prediction_length"] for sequence_data in sequence_data_group
        ),  # 所有预测长度累加。
        "ground_truth_length": sum(
            sequence_data["support_context"]["ground_truth_length"] for sequence_data in sequence_data_group
        ),  # 所有真值长度累加。
        "aligned_length": sum(
            sequence_data["support_context"]["aligned_length"] for sequence_data in sequence_data_group
        ),  # 所有对齐长度累加。
    }  # 多序列支持上下文。
    mechanism_support_context = {
        "prediction_length": sum(
            sequence_data["mechanism_support_context"]["prediction_length"] for sequence_data in sequence_data_group
        ),
        "ground_truth_length": sum(
            sequence_data["mechanism_support_context"]["ground_truth_length"] for sequence_data in sequence_data_group
        ),
        "aligned_length": sum(
            sequence_data["mechanism_support_context"]["aligned_length"] for sequence_data in sequence_data_group
        ),
    }  # 多序列 measurement-level 支持上下文。
    squared_error_sum = math.fsum(error * error for error in pooled_error_trace)  # 误差平方和。
    error_sum = math.fsum(pooled_error_trace)  # 误差求和。
    error_count = len(pooled_error_trace)  # 总样本数。
    if error_count == 0:  # 无有效对齐点时返回零值指标表，防止除零。
        metric_table = {key: 0.0 for key in metric_order}  # 所有指标置零。
        support_report = _build_support_report(  # 用标准函数构造完整 support_report，确保字段不缺失。
            prediction_length=support_context["prediction_length"],
            ground_truth_length=support_context["ground_truth_length"],
            aligned_length=0,
            valid_pair_count=0,
        )
        # H25c 真改：早返回路径也走 _aggregate_gdop_occupancy，保持 support_report 字段一致
        # （None 表示无序列携带 geometry_report，下游视作"未测量"）。
        support_report["gdop_occupancy_aggregation"] = _aggregate_gdop_occupancy(
            sequence_data_group, protocol_cfg=protocol_cfg
        )
        return metric_table, support_report
    trajectory_block = {
        "rmse": math.sqrt(squared_error_sum / error_count),  # RMSE 直接从误差平方和计算。
        "mae": error_sum / error_count,  # MAE 直接从误差均值计算。
        "ate": 0.0,  # 先放默认值，后面逐序列做 SE3 对齐后计算。
        "rpe": 0.0,  # 先放默认值，后面逐序列计算标准 RPE。
    }  # 多序列轨迹指标块。

    # 准则 36：跨序列 RMSE 帧级 mean ± std 聚合。
    # 在 sequence_trajectory_metrics 里已逐序列算出 per-seq rmse（frame-level mean 口径），
    # 跨序列聚合成一个均值与样本标准差（Bessel 校正），写到 trajectory_block。
    # 这与 pooled RMSE 互为补充：pooled 看「全局拼接点云」，跨序列 mean±std 看「每轨先 RMSE 再聚合」。
    per_seq_rmse_values: list[float] = []  # 收集每序列 RMSE（来自 per-seq compute_trajectory_metrics，已含 Umeyama 对齐）。
    for seq_metrics in sequence_trajectory_metrics:  # 遍历逐序列结果。
        seq_rmse = seq_metrics.get("rmse", None)
        if seq_rmse is None:
            continue
        try:
            seq_rmse_f = float(seq_rmse)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(seq_rmse_f):
            per_seq_rmse_values.append(seq_rmse_f)  # 仅追加有限值，防止污染 std。
    if per_seq_rmse_values:  # 至少有一个序列的 RMSE 时才算 mean±std。
        cross_seq_mean = math.fsum(per_seq_rmse_values) / len(per_seq_rmse_values)  # 跨序列 mean_rmse.
        if len(per_seq_rmse_values) >= 2:  # 至少 2 个序列才计算样本标准差（Bessel 校正）。
            cross_seq_variance = math.fsum(
                (value - cross_seq_mean) ** 2 for value in per_seq_rmse_values
            ) / (len(per_seq_rmse_values) - 1)
            cross_seq_std = math.sqrt(cross_seq_variance)
        else:
            cross_seq_std = 0.0  # 单序列时 std 退化为 0.0.
        trajectory_block["mean_rmse"] = cross_seq_mean  # 帧级跨序列 RMSE 均值。
        trajectory_block["std_rmse"] = cross_seq_std  # 帧级跨序列 RMSE 样本标准差（Bessel 校正）.
    # else：单序列或全 NaN 路径，mean_rmse/std_rmse 不写；下游消费者读 .get("mean_rmse") 拿 None 安全降级。

    # ATE: 逐序列做 SE3 对齐后计算 RMSE，再取所有序列的加权平均。
    ate_squared_terms: list[float] = []  # 收集各序列加权平方误差，用 math.fsum 累加以保证精度。
    ate_count = 0
    for sequence_data, seq_metrics in zip(sequence_data_group, sequence_trajectory_metrics):  # 逐序列累积 SE3 对齐后的误差。
        seq_ate = seq_metrics.get("ate", None)
        if seq_ate is not None:
            seq_len = sequence_data["support_context"].get("aligned_length", 0)
            if seq_len >= 2:  # 单点序列 ATE=0.0 无意义，跳过以避免稀释加权平均。
                ate_squared_terms.append(seq_ate * seq_ate * seq_len)  # 收集加权平方误差项。
                ate_count += seq_len
    ate_squared_sum = math.fsum(ate_squared_terms)  # 高精度累加加权平方误差。
    if ate_count > 0:
        trajectory_block["ate"] = math.sqrt(ate_squared_sum / ate_count)  # 加权平均 ATE。
    else:
        trajectory_block["ate"] = 0.0  # 无序列级 ATE 时置零（单点序列完美平移对齐），不用 RMSE 代替以避免语义混淆。

    # RPE: 逐序列计算标准相对位姿误差（相邻帧相对运动差异），再取加权平均。
    rpe_terms: list[float] = []  # 收集各序列加权 RPE，用 math.fsum 累加以保证精度。
    rpe_count = 0
    for sequence_data, seq_metrics in zip(sequence_data_group, sequence_trajectory_metrics):  # 逐序列累积标准 RPE。
        seq_rpe = seq_metrics.get("rpe", None)
        if seq_rpe is not None:
            seq_len = sequence_data["support_context"].get("aligned_length", 0)
            if seq_len >= 2:  # 单点序列 RPE=0.0 无意义，跳过以避免稀释加权平均。
                rpe_terms.append(seq_rpe * (seq_len - 1))  # 收集加权 RPE 项（n-1 个帧对）。
                rpe_count += seq_len - 1
    rpe_sum = math.fsum(rpe_terms)  # 高精度累加加权 RPE。
    if rpe_count > 0:
        trajectory_block["rpe"] = rpe_sum / rpe_count  # 加权平均 RPE。

    tail_block, tail_support = _compute_pooled_tail_metrics(  # 保持长失败段只在单条序列内部连续。
        sequence_data_group,
        failure_threshold=failure_threshold,
    )
    reliability_block, reliability_support = compute_reliability_metrics(  # 计算整体可靠性指标。
        pooled_risk_trace,  # 汇总后的风险轨迹。
        pooled_mechanism_error_trace,  # 汇总后的 measurement-level 误差轨迹。
        support_context=mechanism_support_context,  # 多序列 measurement-level 支持上下文。
    )
    reliability_block["bias_alignment"] = compute_trace_correlation(  # 补充偏差与误差相关性。
        pooled_bias_trace,  # 汇总后的偏差轨迹。
        pooled_mechanism_error_trace,  # 汇总后的 measurement-level 误差轨迹。
        lhs_name="bias_trace",  # 用于错误信息的轨迹名。
    )
    reliability_block["corr_scaling_error"] = compute_trace_correlation(  # 补充缩放与误差相关性。
        pooled_scaling_trace,  # 汇总后的缩放轨迹。
        pooled_mechanism_error_trace,  # 汇总后的 measurement-level 误差轨迹。
        lhs_name="scaling_trace",  # 用于错误信息的轨迹名。
    )
    runtime_block = _compute_pooled_runtime_metrics(sequence_data_group)  # 计算多序列 runtime 汇总块。
    # 计算 yaw 误差指标：展平所有序列的航向角误差轨迹。
    pooled_yaw_error_trace = [
        yaw_error for sequence_data in sequence_data_group for yaw_error in sequence_data["yaw_error_trace"]
    ]
    yaw_block = _compute_yaw_metrics(pooled_yaw_error_trace)
    support_report = _build_support_report(  # 汇总多序列支持信息。
        support_context["prediction_length"],  # 汇总预测长度。
        support_context["ground_truth_length"],  # 汇总真值长度。
        support_context["aligned_length"],  # 汇总对齐长度。
        reliability_support["valid_pair_count"],  # 汇总后的有效成对样本数（从support_report取）。
    )
    support_report["ate_degraded"] = any_ate_degraded  # ATE退化标志放入support_report，不进入metric_table。
    support_report["long_failure_segments"] = tail_support["long_failure_segments"]  # 连续失败段放入support_report。
    support_report["reliability_status"] = reliability_support["reliability_status"]  # 可靠性状态放入support_report。
    # H24e 真改 §4.2 L894「差 GDOP 占比聚合」：调 _aggregate_gdop_occupancy 跨序列
    # 聚合 gdop_occupancy_above_floor，写入 support_report 供上层审计。None 表示
    # 该评估组中所有序列均无 geometry_report，下游消费者视作"未测量"而非 0.0。
    support_report["gdop_occupancy_aggregation"] = _aggregate_gdop_occupancy(
        sequence_data_group, protocol_cfg=protocol_cfg
    )
    # §9.3 多序列场景下：每轨脉冲/异步量级门检查 + 聚合报告。
    support_report["section9_pulse_async_aggregation"] = _aggregate_section9_pulse_async(
        sequence_data_group, protocol_cfg=protocol_cfg
    )
    # 单轨警告写入 support_report（供 12_run_statistics / 15_build_six_cmp_aggregate 消费）。
    for i, sd in enumerate(sequence_data_group):
        _emit_section9_pulse_async_warnings(support_report, sd, protocol_cfg=protocol_cfg)

    merged_blocks = {}  # 用总字典承接各块结果。
    merged_blocks.update(trajectory_block)  # 合并轨迹块。
    merged_blocks.update(tail_block)  # 合并尾部块。
    merged_blocks.update(yaw_block)  # 合并航向角指标。
    merged_blocks.update(reliability_block)  # 合并可靠性块。
    merged_blocks.update(runtime_block)  # 合并运行时块。
    return _build_metric_table(merged_blocks, metric_order), support_report


def compute_metrics(
    prediction_bundle: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    gt_bundle: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    failure_threshold: float | None = None,
    *,
    return_support: bool = False,
    protocol_cfg: Mapping[str, Any] | None = None,
) -> dict[str, float] | tuple[dict[str, float], dict[str, Any]]:
    """对外统一入口：根据单序列/多序列输入返回最终指标表。  # 函数作用总述。

    作用：接收预测和真值 bundle，自动判断单序列/多序列，分别走对应的指标计算路径，  # 详细作用。
    最终按协议冻结顺序输出统一指标表。  # 说明输出保证。
    参数：  # 参数说明。
    - prediction_bundle: 预测结果，可以是单个映射或映射序列。  # 参数说明。
    - gt_bundle: 真值结果，可以是单个映射或映射序列，必须与 prediction_bundle 序列数一致。  # 参数说明。
    - failure_threshold: 尾部指标的失败阈值；若不显式传入，则默认跟随实验协议 evaluation.default_failure_threshold_m，必须为正有限实数。  # 参数说明。
    - return_support: 是否同时返回支持信息报告，默认 False。  # 参数说明。
    - protocol_cfg: H25c 真改 §4.2 L894「差 GDOP 占比守门」协议层显式冻结 —— 协议层 yaml
      顶层 gdop_occupancy_min_ratio（缺省 0.30），评估层 _aggregate_gdop_occupancy 计算
      ratio_above_floor 后，< gdop_occupancy_min_ratio 时抛 ValueError 阻断评估。
      None 时跳过守门（向后兼容旧调用方）。  # 参数说明。
    返回值：  # 返回值说明。
    - dict[str, float]：按协议顺序排列的指标表（return_support=False 时）。  # 默认返回值。
    - tuple[dict[str, float], dict[str, Any]]：(metric_table, support_report)（return_support=True 时）。  # 带支持信息的返回值。
    异常/失败条件：  # 失败条件说明。
    - failure_threshold 非实数时抛 TypeError。  # 阈值类型校验失败。
    - failure_threshold 非有限或非正时抛 ValueError。  # 阈值范围校验失败。
    - 两边序列数不一致时抛 ValueError。  # 配对校验失败。
    - 内部对齐或指标计算失败时会向上传播异常。  # 内部异常透传。
    - H25c 真改：protocol_cfg.evaluation.gdop_occupancy_min_ratio 守门触发时抛 ValueError。  # 协议层下界守门。
    状态变化：  # 状态变化说明。
    - 不修改输入，只返回新的指标表和可选的支持报告。  # 不改状态。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "failure_threshold": failure_threshold,
            "return_support": return_support,
            "prediction_bundle_type": type(prediction_bundle).__name__,
            "gt_bundle_type": type(gt_bundle).__name__,
        },
        "compute_metrics 入口参数",
        prefix="[metrics]",
    )
    if failure_threshold is None:  # 未显式指定时统一使用实验协议默认阈值。
        failure_threshold = get_default_failure_threshold_m()
    if not is_real(failure_threshold):  # 阈值必须是实数，不能是布尔值或字符串等。
        raise TypeError("failure_threshold must be a real number")  # 阈值类型不对。
    if not math.isfinite(failure_threshold):  # 阈值必须是有限值，拒绝 NaN 和无穷大。
        raise ValueError("failure_threshold must be finite")  # 非有限阈值拒绝。
    if failure_threshold <= 0.0:  # 阈值必须是正数。
        raise ValueError("failure_threshold must be positive")

    prediction_group = _normalize_bundle_group(prediction_bundle, name="prediction_bundle")  # 先把预测输入归一成 bundle 列表。
    gt_group = _normalize_bundle_group(gt_bundle, name="gt_bundle")  # 再把真值输入归一成 bundle 列表。
    if len(prediction_group) != len(gt_group):  # 两边序列数必须一致，才能一一配对。
        raise ValueError("prediction_bundle and gt_bundle must contain the same number of sequences")

    metric_order = get_metric_order()  # 读取协议冻结的指标顺序。
    sequence_data_group = [
        _collect_sequence_data(prediction_obj, gt_obj, bundle_index=bundle_index, protocol_cfg=protocol_cfg)  # 逐对整理每个序列的中间数据。
        for bundle_index, (prediction_obj, gt_obj) in enumerate(zip(prediction_group, gt_group))  # 逐序列配对遍历。
    ]  # 逐对整理每个序列的中间数据。

    if len(sequence_data_group) == 1:  # 单序列走单序列路径，避免不必要的聚合误差。
        metric_table, support_report = _compute_single_sequence_metrics(
            sequence_data_group[0],  # 只有一个序列时直接取第 0 项。
            metric_order=metric_order,  # 协议冻结的指标顺序。
            failure_threshold=failure_threshold,  # 尾部指标阈值。
            protocol_cfg=protocol_cfg,  # H25c 真改：协议层下界守门配置透传
        )
    else:  # 多序列走聚合路径。
        metric_table, support_report = _compute_multi_sequence_metrics(
            sequence_data_group,  # 多序列中间数据列表。
            metric_order=metric_order,  # 协议冻结的指标顺序。
            failure_threshold=failure_threshold,  # 尾部指标阈值。
            protocol_cfg=protocol_cfg,  # H25c 真改：协议层下界守门配置透传
        )

    if return_support:  # 调用者如果需要支持信息，就返回二元组。
        return metric_table, support_report
    return metric_table  # 默认只返回指标表本体。


# =============================================================================
# §9.3 单轨脉冲/异步量级门 (前提指导 §0.2 B07-B08 / §9.3 量级「n_pulse≥30 / n_async≥20」).
#
# 本节提供两类辅助函数:
#   1. _emit_section9_pulse_async_warnings: 单轨违反量级门时把违规条目写入
#      support_report['section9_pulse_async_violations'] 并在 protocol_cfg 启用
#      时通过 warnings.warn 显式提示; 不阻断评估 (与 §9 「放松则伤」同口径).
#   2. _aggregate_section9_pulse_async: 多序列场景下汇总每轨 n_pulse/n_async 与
#      违规频数, 写入 support_report['section9_pulse_async_aggregation'].
#
# n_pulse 推导: measurement_mask 中 True 的数量 = 有效 UWB 测量脉冲数.
# n_async 推导: 通过 prediction_obj.scenario_context.async_report.dropped_event_count
#   读取异步扰动删除的事件数 (即显著异步错位事件). 缺失时记为 None (向后兼容).
# =============================================================================


def _extract_n_async_from_scenario_context(prediction_obj: Any) -> int | None:
    """从 prediction_obj.scenario_context.async_report.dropped_event_count 解析 n_async.

    参数:
        prediction_obj: 单条预测 bundle.

    返回:
        int | None: 异步扰动删除事件数; 缺失或类型错时返回 None.
    """
    if not isinstance(prediction_obj, Mapping):
        return None
    scenario_ctx = prediction_obj.get('scenario_context')
    if not isinstance(scenario_ctx, Mapping):
        return None
    async_report = scenario_ctx.get('async_report')
    if not isinstance(async_report, Mapping):
        return None
    raw = async_report.get('dropped_event_count')
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if not isinstance(raw, int):
        try:
            if isinstance(raw, float) and math.isfinite(float(raw)):
                return int(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        return None
    if raw < 0:
        return None
    return int(raw)


def _emit_section9_pulse_async_warnings(
    support_report: dict[str, Any],
    sequence_data: Mapping[str, Any],
    *,
    protocol_cfg: Mapping[str, Any] | None = None,
) -> None:
    """§9.3 单轨脉冲/异步量级门检查并写入 support_report.

    参数:
        support_report: 待写入的 support_report 字典 (原地修改).
        sequence_data: _collect_sequence_data 返回的中间数据.
        protocol_cfg: 可选协议配置.

    返回:
        None; 违规条目追加到 support_report['section9_pulse_async_violations'].
    """
    try:
        from liquidloc.protocol.experiment_gates import check_per_trajectory_pulse_async
    except Exception as exc:  # pragma: no cover - 协议层 import 失败时降级
        warnings.warn(
            f'_emit_section9_pulse_async_warnings: 协议层 import 失败 ({exc!r}), §9.3 脉冲/异步量级门跳过',
            stacklevel=2,
        )
        return

    measurement_mask = sequence_data.get('measurement_mask') if isinstance(sequence_data, Mapping) else None
    if not isinstance(measurement_mask, (list, tuple)):
        return
    n_pulse = sum(1 for is_measured in measurement_mask if is_measured)
    prediction_obj = sequence_data.get('prediction_obj') if isinstance(sequence_data, Mapping) else None
    n_async = _extract_n_async_from_scenario_context(prediction_obj)
    if n_async is None:
        return  # 缺失场景下不参与检查 (向后兼容旧 bundle)

    try:
        report = check_per_trajectory_pulse_async(
            n_pulse=int(n_pulse),
            n_async=int(n_async),
            protocol_cfg=protocol_cfg,
        )
    except (TypeError, ValueError) as exc:
        warnings.warn(
            f'_emit_section9_pulse_async_warnings: check_per_trajectory_pulse_async 异常 ({exc!r})',
            stacklevel=2,
        )
        return

    # 单轨违规追加到 support_report['section9_pulse_async_violations']
    violations = support_report.setdefault('section9_pulse_async_violations', [])
    if report.get('pulse_violated') or report.get('async_violated'):
        seq_id = None
        if isinstance(prediction_obj, Mapping):
            seq_id = prediction_obj.get('seq_id')
        violations.append({
            'seq_id': seq_id,
            'n_pulse': int(report['n_pulse']),
            'n_pulse_min': int(report['n_pulse_min']),
            'n_async': int(report['n_async']),
            'n_async_min': int(report['n_async_min']),
            'pulse_violated': bool(report['pulse_violated']),
            'async_violated': bool(report['async_violated']),
            'cmp1_cmp5_at_risk': bool(report['cmp1_cmp5_at_risk']),
            'message': str(report.get('message', '')),
        })
        # §9 「放松则伤」: 显式 warning 便于审计
        warnings.warn(
            f"§9.3 {report['message']} (seq_id={seq_id})",
            stacklevel=2,
        )


def _aggregate_section9_pulse_async(
    sequence_data_group: Sequence[Mapping[str, Any]],
    *,
    protocol_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """§9.3 多序列场景下汇总每轨 n_pulse/n_async 与违规频数.

    参数:
        sequence_data_group: 多序列中间数据列表.
        protocol_cfg: 可选协议配置.

    返回:
        dict 含 sequence_count / cmmp1_cmp5_at_risk_count / total_n_pulse /
        total_n_async / per_trajectory 列表.
    """
    per_trajectory: list[dict[str, Any]] = []
    total_n_pulse = 0
    total_n_async = 0
    cmp1_cmp5_at_risk_count = 0
    missing_async_count = 0
    for sequence_data in sequence_data_group:
        if not isinstance(sequence_data, Mapping):
            continue
        measurement_mask = sequence_data.get('measurement_mask')
        if not isinstance(measurement_mask, (list, tuple)):
            continue
        n_pulse = sum(1 for is_measured in measurement_mask if is_measured)
        prediction_obj = sequence_data.get('prediction_obj')
        n_async = _extract_n_async_from_scenario_context(prediction_obj)
        seq_id = None
        if isinstance(prediction_obj, Mapping):
            seq_id = prediction_obj.get('seq_id')
        per_trajectory.append({
            'seq_id': seq_id,
            'n_pulse': int(n_pulse),
            'n_async': n_async,
        })
        total_n_pulse += int(n_pulse)
        if n_async is None:
            missing_async_count += 1
            continue
        total_n_async += int(n_async)
        try:
            from liquidloc.protocol.experiment_gates import check_per_trajectory_pulse_async
            report = check_per_trajectory_pulse_async(
                n_pulse=int(n_pulse),
                n_async=int(n_async),
                protocol_cfg=protocol_cfg,
            )
            if report.get('cmp1_cmp5_at_risk'):
                cmp1_cmp5_at_risk_count += 1
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f'_aggregate_section9_pulse_async: 异常跳过 seq_id={seq_id}: {exc!r}',
                stacklevel=2,
            )

    return {
        'sequence_count': len(per_trajectory),
        'cmp1_cmp5_at_risk_count': int(cmp1_cmp5_at_risk_count),
        'missing_async_count': int(missing_async_count),
        'total_n_pulse': int(total_n_pulse),
        'total_n_async': int(total_n_async),
        'per_trajectory': per_trajectory,
    }


# =============================================================================
# §9.1 冷启动偏移统一划除 (前提指导 §9 / §0.2 B06 / §9.1 「冷启动是否计入事先固定」).
#
# _trim_cold_start_segment: 在 _collect_sequence_data 中调用, 从 pred_traj/gt_traj
# 中移除 t < cold_start_threshold 的条目, 实现全员统一划除 (不偏向任何方法).
#
# 参数:
#   pred_traj: 预测轨迹列表, 每个元素含 timestamp 字段.
#   gt_traj: 真值轨迹列表, 每个元素含 timestamp 字段.
#   offset_s: 冷启动偏移秒数 (从 protocol_cfg.scene_scale.cold_start_offset_s 读取).
#
# 返回:
#   tuple[list[dict], list[dict]]: 去除冷启动段后的预测/真值轨迹列表.
#
# 注意:
#   - 保留第一条时间戳 >= offset_s 的条目作为新起点, 保持对齐.
#   - 若偏移后无有效条目, 返回空列表 (下游 compute_metrics 会返回零值指标).
#   - 不修改输入, 只构造新列表返回.
# =============================================================================

def _trim_cold_start_segment(
    pred_traj: Sequence[Mapping],
    gt_traj: Sequence[Mapping],
    *,
    offset_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """§9.1 冷启动偏移统一划除: 移除 pred_traj/gt_traj 中 t < offset_s 的条目.

    按 §9 「冷启动是否计入事先固定」全员统一划除, 禁止只删对某方法不利的起止段.
    保留第一条时间戳 >= offset_s 的条目作为新起点, 保持对齐.

    参数:
        pred_traj: 预测轨迹列表, 每个元素含 timestamp 字段 (秒).
        gt_traj: 真值轨迹列表, 每个元素含 timestamp 字段 (秒).
        offset_s: 冷启动偏移秒数 (§9.1 / §0.2 B06).

    返回:
        tuple[list[dict], list[dict]]: 去除冷启动段后的预测/真值轨迹列表.

    异常:
        TypeError: 输入不是序列类型.
        ValueError: offset_s 非有限或非正.
    """
    if not isinstance(offset_s, (int, float)) or isinstance(offset_s, bool):
        raise TypeError(f"offset_s must be a finite real number, got {type(offset_s).__name__}")
    if not math.isfinite(offset_s) or offset_s < 0:
        raise ValueError(f"offset_s must be a non-negative finite real, got {offset_s}")
    if offset_s == 0:
        return list(pred_traj), list(gt_traj)  # 无偏移时直接返回副本.

    from liquidloc.common.tee_logger import print_dict
    print_dict(
        {"pred_traj_len": len(pred_traj), "gt_traj_len": len(gt_traj), "offset_s": offset_s},
        "§9.1 冷启动偏移划除",
        prefix="[§9.1]",
    )

    # 找第一条 pred_traj 时间戳 >= offset_s 的索引.
    pred_start_idx = None
    for i, point in enumerate(pred_traj):
        ts = point.get("timestamp")
        if ts is None:
            continue
        try:
            ts_f = float(ts)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(ts_f) and ts_f >= float(offset_s):
            pred_start_idx = i
            break

    # 找第一条 gt_traj 时间戳 >= offset_s 的索引.
    gt_start_idx = None
    for i, point in enumerate(gt_traj):
        ts = point.get("timestamp")
        if ts is None:
            continue
        try:
            ts_f = float(ts)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(ts_f) and ts_f >= float(offset_s):
            gt_start_idx = i
            break

    if pred_start_idx is None and gt_start_idx is None:
        # 所有条目都在冷启动段内, 返回空列表.
        return [], []

    # 取较大索引作为起点, 保证对齐.
    start_idx = max(pred_start_idx or 0, gt_start_idx or 0)

    trimmed_pred = list(pred_traj[start_idx:]) if pred_start_idx is not None else []
    trimmed_gt = list(gt_traj[start_idx:]) if gt_start_idx is not None else []

    print_dict(
        {
            "pred_start_idx": pred_start_idx,
            "gt_start_idx": gt_start_idx,
            "start_idx": start_idx,
            "trimmed_pred_len": len(trimmed_pred),
            "trimmed_gt_len": len(trimmed_gt),
        },
        "§9.1 冷启动偏移划除 结果",
        prefix="[§9.1]",
    )

    return trimmed_pred, trimmed_gt


def _apply_takeoff_landing_floor_transition_to_measurement_mask(
    measurement_mask: list[bool],
    *,
    aligned_indices: Sequence[int],
    prediction_obj: Mapping[str, Any] | None = None,
    protocol_cfg: Mapping[str, Any] | None = None,
) -> list[bool]:
    """按 §9 细节制度把起飞/降落/楼层切换帧从 measurement_mask 中剔除.

    协议 scene_scale.takeoff_landing_policy / floor_transition_policy 决定如何
    排除或保留起飞/降落段与室内外切换段。在 measurement_mask 层级（已对齐到
    pred_traj 原始索引）应用 mask：exclude_unified + mask=True 的对应帧设为 False，
    使该帧不计入计分段；include_unified / manual_guard 不修改（调用方审计）.

    参数:
        measurement_mask: 来自 _extract_measurement_mask 的对齐后 measurement mask.
        aligned_indices: 与 measurement_mask 行序对齐的 pred_traj 原始索引.
        prediction_obj: 原始预测 bundle，含 scenario_context.takeoff_landing_mask
            与 floor_transition_mask (per-frame bool list, True = 该帧属于制度段).
        protocol_cfg: 协议配置，含 scene_scale.takeoff_landing_policy 与
            scene_scale.floor_transition_policy.

    返回:
        list[bool]: 修改后的 measurement_mask (mask=True 的制度段被设为 False)。
    """
    if len(measurement_mask) != len(aligned_indices):
        raise ValueError(
            "measurement_mask must align with aligned_indices length"
        )

    takeoff_policy = _resolve_takeoff_landing_policy(protocol_cfg)
    floor_policy = _resolve_floor_transition_policy(protocol_cfg)
    takeoff_mask = _get_scenario_bool_mask(prediction_obj, 'takeoff_landing_mask')
    floor_mask = _get_scenario_bool_mask(prediction_obj, 'floor_transition_mask')

    # 长度不匹配时降级为 None (由调用方跳过过滤).
    pred_len = _safe_prediction_length(prediction_obj)
    if takeoff_mask is not None and pred_len is not None and len(takeoff_mask) != pred_len:
        takeoff_mask = None
    if floor_mask is not None and pred_len is not None and len(floor_mask) != pred_len:
        floor_mask = None

    new_mask = list(measurement_mask)
    for i, axis_idx in enumerate(aligned_indices):
        if not (isinstance(axis_idx, int) and not isinstance(axis_idx, bool)):
            continue
        if (
            takeoff_policy == 'exclude_unified'
            and takeoff_mask is not None
            and 0 <= axis_idx < len(takeoff_mask)
            and takeoff_mask[axis_idx]
        ):
            new_mask[i] = False
        if (
            floor_policy == 'exclude_unified'
            and floor_mask is not None
            and 0 <= axis_idx < len(floor_mask)
            and floor_mask[axis_idx]
        ):
            new_mask[i] = False
    return new_mask


def _resolve_takeoff_landing_policy(
    protocol_cfg: Mapping[str, Any] | None,
) -> str:
    """从协议配置读取 takeoff_landing_policy，缺省返回 'exclude_unified'."""
    if protocol_cfg is None or not isinstance(protocol_cfg, Mapping):
        return 'exclude_unified'
    scene_scale = protocol_cfg.get('scene_scale')
    if not isinstance(scene_scale, Mapping):
        return 'exclude_unified'
    return scene_scale.get('takeoff_landing_policy', 'exclude_unified')


def _resolve_floor_transition_policy(
    protocol_cfg: Mapping[str, Any] | None,
) -> str:
    """从协议配置读取 floor_transition_policy，缺省返回 'exclude_unified'."""
    if protocol_cfg is None or not isinstance(protocol_cfg, Mapping):
        return 'exclude_unified'
    scene_scale = protocol_cfg.get('scene_scale')
    if not isinstance(scene_scale, Mapping):
        return 'exclude_unified'
    return scene_scale.get('floor_transition_policy', 'exclude_unified')


def _get_scenario_bool_mask(
    prediction_obj: Mapping[str, Any] | None,
    field: str,
) -> list[bool] | None:
    """从 prediction_obj.scenario_context 读取 per-frame bool 掩码.

    缺失、类型错误时返回 None（调用方按政策降级处理）.
    """
    if prediction_obj is None or not isinstance(prediction_obj, Mapping):
        return None
    scenario_ctx = prediction_obj.get('scenario_context')
    if not isinstance(scenario_ctx, Mapping):
        return None
    raw = scenario_ctx.get(field)
    if raw is None or not isinstance(raw, list) or not raw:
        return None
    try:
        return [bool(v) for v in raw]
    except (TypeError, ValueError):
        return None


def _safe_prediction_length(prediction_obj: Mapping[str, Any] | None) -> int | None:
    """从 prediction_obj.states 或 timestamps 推断预测轨迹长度，缺失返回 None."""
    if prediction_obj is None or not isinstance(prediction_obj, Mapping):
        return None
    states = prediction_obj.get('states')
    if isinstance(states, list):
        return len(states)
    timestamps = prediction_obj.get('timestamps')
    if isinstance(timestamps, list):
        return len(timestamps)
    return None


def _resolve_t_eff_min_s(
    protocol_cfg: Mapping[str, Any] | None,
) -> float | None:
    """从协议 scene_scale.t_eff_min_s 读取, 缺失或类型错时返回 None (不启用守门)."""
    if protocol_cfg is None or not isinstance(protocol_cfg, Mapping):
        return None
    scene_scale = protocol_cfg.get('scene_scale')
    if not isinstance(scene_scale, Mapping):
        return None
    if 't_eff_min_s' not in scene_scale:
        return None
    try:
        val = float(scene_scale['t_eff_min_s'])
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(val) or val <= 0.0:
        return None
    return val


def _compute_traj_duration_s(traj: list[dict[str, Any]]) -> float | None:
    """计算轨迹的有效时长 (max_timestamp - min_timestamp), 缺时间戳或单点时返回 None."""
    if not traj:
        return None
    timestamps = [
        _safe_get_timestamp(p) for p in traj
        if _safe_get_timestamp(p) is not None
    ]
    if len(timestamps) < 2:
        return None
    return max(timestamps) - min(timestamps)
