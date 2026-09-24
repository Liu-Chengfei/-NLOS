"""
文件：`src/liquidloc/plotting/plot_trajectories.py`

这个模块负责把预测轨迹和真实轨迹渲染成图。它不计算误差，也不重新对齐轨迹，
只负责把已经准备好的 trajectory bundle 整理成绘图规格，再交给 matplotlib 画图。

上游通常会传入 prediction_bundle 和 gt_bundle。
下游通常是 figures 生成脚本、测试，以及需要轨迹图路径的报告流程。

最容易看错的地方：
1. 预测轨迹和真实轨迹必须成对出现，而且序列数必须一致。
2. 如果有 z 维，就画 3D；否则画 2D。
3. 这里只做绘图，不做 ATE、RPE 或其他轨迹评估。
"""

from collections.abc import Mapping, Sequence  # 兼容映射和序列输入。
import math  # 用于 isfinite 检查。
from pathlib import Path  # 输出路径统一用 Path 处理.
from typing import Any  # 用于类型注解。

from liquidloc.common.constants import FIGURE_PATH_KEY, TIME_KEY_CANDIDATES  # D9 单源：绘图层 figure_path 键名与时间字段候选键名常量。
from liquidloc.common.validation import is_bool_like  # 统一判断布尔类型（含 numpy.bool_）。


# bundle 归一器：把单个 mapping 或多项序列统一成可迭代组。
def _normalize_bundle_group(bundle_obj: Any, *, name: str) -> list[Any]:  # 把一个 bundle 统一成序列列表。
    # bundle_obj 可能是单个 mapping，也可能是一个 sequence；name 用于报错定位。
    # 返回值永远是 list，后面可以直接 zip 预测和真实轨迹。
    if isinstance(bundle_obj, Mapping):  # 单个 mapping 也视为一组。
        return [bundle_obj]  # 直接包成单元素列表。
    if isinstance(bundle_obj, Sequence) and not isinstance(bundle_obj, (str, bytes)):  # 序列型输入直接展开。
        bundle_group = list(bundle_obj)  # 先冻结成列表。
        if not bundle_group:  # 空组不允许。
            raise ValueError(f"{name} must be non-empty.")
        return bundle_group  # 返回统一后的列表形式。
    raise TypeError(f"{name} must be a mapping or a non-string sequence of mappings.")  # 其他类型直接拒绝。


# 轨迹提取器：从 bundle 中取 states 和可选 timestamps，并补齐时间字段。
def _extract_traj(bundle_obj: Any, *, name: str) -> list[dict[str, Any]]:  # 从 bundle 中抽取轨迹点列表。
    # 这里把 states 和 timestamps 统一成后续能直接画图的点列表。
    # 返回的 traj 是一串 dict，每个 dict 表示一个轨迹点。
    if not isinstance(bundle_obj, Mapping):  # bundle 本身必须是 mapping。
        raise TypeError(f"{name} must be a mapping.")

    # D3 value contract：states 是必需键，用 "in" 守卫 + [] 直接访问，避免 .get() 静默返回 None。
    if "states" not in bundle_obj:  # states 是必需字段。
        raise ValueError(f"{name} is missing 'states'.")  # 缺键属于值合同违规，按项目硬约束用 ValueError 而非 KeyError。
    states = bundle_obj["states"]  # 轨迹状态序列。
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):  # states 必须是非字符串序列。
        raise TypeError(f"{name}.states must be a non-string sequence.")
    if not states:  # 空轨迹不能画。
        raise ValueError(f"{name}.states must be non-empty.")

    timestamps = bundle_obj.get("timestamps")  # 可选时间戳序列，用 .get() 合理。
    if timestamps is not None:  # 如果提供了时间戳，就校验长度和类型。
        if not isinstance(timestamps, Sequence) or isinstance(timestamps, (str, bytes)):  # 时间戳也必须是非字符串序列。
            raise TypeError(f"{name}.timestamps must be a non-string sequence when provided.")
        if len(timestamps) != len(states):  # 时间戳数量必须和状态数量一致。
            raise ValueError(f"{name}.timestamps must align with {name}.states.")

    traj = []  # 保存整理后的轨迹点。
    for index, state in enumerate(states):  # 逐状态处理。
        if not isinstance(state, Mapping):  # 每个状态都必须是 mapping。
            raise TypeError(f"{name}.states[{index}] must be a mapping.")
        point = dict(state)  # 复制一份，避免修改原对象。
        # D1 协议一致性 + D9 单源：时间字段候选引用 TIME_KEY_CANDIDATES 常量（不含 "index"），
        # 与 metric_runner.py / trajectory_metrics.py 保持一致，避免 "index" 被误当时间键导致按值而非按位置对齐。
        if timestamps is not None and not any(key in point for key in TIME_KEY_CANDIDATES):  # 如果没时间字段就补一个。
            point["timestamp"] = timestamps[index]
        traj.append(point)  # 收集轨迹点。
    return traj  # 返回轨迹点列表。


# 坐标维度判断器：检查点集是否全都有 pz，从而决定 2D/3D。
def _coord_keys(points: list[dict[str, Any]], *, traj_name: str) -> tuple[str, ...]:  # 判断轨迹是 2D 还是 3D。
    # traj_name 只是报错时说明是哪一组轨迹。
    # 这里返回的坐标键顺序会直接决定后面 plot 的参数顺序。
    if not all("px" in point and "py" in point for point in points):  # px 和 py 是必需字段。
        raise ValueError(f"{traj_name} points must contain 'px' and 'py'.")
    has_any_pz = any("pz" in point for point in points)  # 看有没有 z 维。
    has_all_pz = all("pz" in point for point in points)  # 必须全有或全无。
    if has_any_pz and not has_all_pz:  # 不能一部分点有 z，另一部分没有。
        raise ValueError(f"{traj_name} points must either all include 'pz' or all omit it.")
    has_pz = has_all_pz  # 如果全部有 z，就视为 3D。
    return ("px", "py", "pz") if has_pz else ("px", "py")  # 返回坐标键顺序。


# 坐标序列整理器：把每个点拆成按坐标分组的浮点序列。
def _point_series(points: list[dict[str, Any]], *, coord_keys: tuple[str, ...], traj_name: str) -> dict[str, list[float]]:  # 把轨迹点转换成按坐标分组的序列。
    # coord_keys 由上一步决定是 2D 还是 3D，这里只负责把值转成浮点数序列。
    # 返回的 dict 形如 {"px": [...], "py": [...], "pz": [...]}。
    series = {key: [] for key in coord_keys}  # 每个坐标键一组序列。
    for index, point in enumerate(points):  # 逐点处理。
        for key in coord_keys:  # 每个坐标都要检查。
            try:  # 逐点逐字段读取坐标值。
                raw_value = point[key]  # 取当前坐标值。
            except KeyError as exc:
                raise ValueError(f"{traj_name}.states[{index}] is missing '{key}'.") from exc
            if is_bool_like(raw_value):  # bool 不当作数值。
                raise ValueError(f"{traj_name}.states[{index}].{key} must be numeric.")
            try:  # 尝试把值转成浮点数。
                value = float(raw_value)  # 尝试转成浮点数。
            except (TypeError, ValueError, OverflowError) as exc:  # D5 数值安全：OverflowError 守卫，防止超大整数 float() 溢出。
                raise ValueError(f"{traj_name}.states[{index}].{key} must be numeric.") from exc
            if not math.isfinite(value):  # NaN/Inf 会导致轨迹图渲染异常（线断裂或轴无限拉伸）。
                raise ValueError(f"{traj_name}.states[{index}].{key} must be finite, got {value!r}.")
            series[key].append(value)  # 收集每个坐标序列。
    return series  # 返回按坐标分组的序列。


# 规范构建函数：把预测/真实轨迹整理成可直接绘制的规范对象。
def build_trajectory_figure_spec(trajectory_bundle: Any, figure_cfg: Mapping[str, Any]) -> dict[str, Any]:  # 组装轨迹图规范对象。
    """将预测轨迹和真实轨迹整理成可渲染的图形规范对象。

    作用：从 trajectory_bundle 中提取预测和真实轨迹，判断维度（2D/3D），
    把坐标序列整理成绘图用的浮点数组，返回供渲染函数直接消费的规范对象。

    参数:
        trajectory_bundle: 轨迹数据映射，必须包含 prediction_bundle 和 gt_bundle。
        figure_cfg: 渲染配置映射，必须包含 figure_path。

    返回值:
        dict[str, Any]: 包含 figure_path、coord_keys、dimension、sequence_specs 的规范对象。

    异常:
        TypeError: 输入类型不正确。
        ValueError: 缺少必需字段、序列数不一致、坐标维度不一致或字段缺失。
    """

    # trajectory_bundle 里必须同时包含 prediction_bundle 和 gt_bundle。
    # 返回的 figure_path 会在 render 阶段真正写入图片。
    if not isinstance(trajectory_bundle, Mapping):  # 顶层必须是 mapping。
        raise TypeError("trajectory_bundle must be a mapping.")
    if not isinstance(figure_cfg, Mapping):  # 配置也必须是 mapping。
        raise TypeError("figure_cfg must be a mapping.")

    # D3 value contract：必需字段缺失属于值合同违规，按项目硬约束用 ValueError 而非 KeyError。
    if "prediction_bundle" not in trajectory_bundle:  # 预测轨迹是必需项。
        raise ValueError("trajectory_bundle is missing 'prediction_bundle'.")
    if "gt_bundle" not in trajectory_bundle:  # 真实轨迹是必需项。
        raise ValueError("trajectory_bundle is missing 'gt_bundle'.")
    if FIGURE_PATH_KEY not in figure_cfg:  # 输出路径是必需项（D9 单源引用常量）。
        raise ValueError(f"figure_cfg is missing '{FIGURE_PATH_KEY}'.")

    figure_path = Path(figure_cfg[FIGURE_PATH_KEY])  # 输出路径转成 Path（D9 单源引用常量）。
    prediction_group = _normalize_bundle_group(trajectory_bundle["prediction_bundle"], name="prediction_bundle")  # 规范化预测组。
    gt_group = _normalize_bundle_group(trajectory_bundle["gt_bundle"], name="gt_bundle")  # 规范化真实组。
    if len(prediction_group) != len(gt_group):  # 两边序列数必须一致。
        raise ValueError("prediction_bundle and gt_bundle must contain the same number of sequences.")

    sequence_specs = []  # 每个序列一个绘图规格。
    dimension = None  # 轨迹维度先不确定。
    for bundle_index, (prediction_obj, gt_obj) in enumerate(zip(prediction_group, gt_group)):  # 逐对处理。
        if not isinstance(prediction_obj, Mapping) or not isinstance(gt_obj, Mapping):  # 每一项都必须是 mapping。
            raise TypeError("prediction_bundle and gt_bundle entries must be mappings.")

        prediction_seq_id = prediction_obj.get("seq_id")  # 取预测序列 ID。
        gt_seq_id = gt_obj.get("seq_id")  # 取真实序列 ID。
        if prediction_seq_id is not None and gt_seq_id is not None and prediction_seq_id != gt_seq_id:  # 如果两个 ID 都存在且不一致，就说明配对错了。
            raise ValueError(
                f"prediction_bundle[{bundle_index}] and gt_bundle[{bundle_index}] have different seq_id values."
            )

        pred_traj = _extract_traj(prediction_obj, name=f"prediction_bundle[{bundle_index}]")  # 抽取预测轨迹点。
        gt_traj = _extract_traj(gt_obj, name=f"gt_bundle[{bundle_index}]")  # 抽取真实轨迹点。

        coord_keys = _coord_keys(pred_traj, traj_name=f"prediction_bundle[{bundle_index}]")  # 决定轨迹维度。
        if _coord_keys(gt_traj, traj_name=f"gt_bundle[{bundle_index}]") != coord_keys:  # 两边坐标维度必须一致。
            raise ValueError("prediction and ground-truth trajectories must use the same coordinate fields.")

        current_dimension = len(coord_keys)  # 当前轨迹是 2D 还是 3D。
        if dimension is None:  # 第一对轨迹先设定全局维度。
            dimension = current_dimension
        elif dimension != current_dimension:  # 后续所有序列维度都要一致。
            raise ValueError("All trajectory pairs must use the same coordinate dimensionality.")

        sequence_specs.append(  # 保存当前序列的绘图规格。
            {
                "seq_label": prediction_seq_id or gt_seq_id or f"sequence_{bundle_index}",  # 序列标签优先用 ID。
                "pred_traj": _point_series(  # 预测轨迹的坐标序列。
                    pred_traj,  # 当前预测轨迹点列表。
                    coord_keys=coord_keys,  # 这一对轨迹共同使用的坐标键。
                    traj_name=f"prediction_bundle[{bundle_index}]",  # 报错时标明是预测哪一组。
                ),
                "gt_traj": _point_series(  # 真实轨迹的坐标序列。
                    gt_traj,  # 当前真实轨迹点列表。
                    coord_keys=coord_keys,  # 这一对轨迹共同使用的坐标键。
                    traj_name=f"gt_bundle[{bundle_index}]",  # 报错时标明是真实哪一组。
                ),
            }  # 当前序列字典结束。
        )  # 当前序列绘图规格追加完成。

    return {  # 返回完整的绘图规范，供渲染阶段直接消费。
        FIGURE_PATH_KEY: figure_path,  # 输出路径（D9 单源引用常量）。
        "coord_keys": coord_keys if sequence_specs else ("px", "py"),  # 没有序列时默认当作 2D。
        "dimension": dimension if dimension is not None else 2,  # 默认维度为 2。
        "sequence_specs": sequence_specs,  # 每个序列的绘图规格。
    }  # 绘图规范对象结束。


# 真实渲染函数：用 matplotlib 画出轨迹对比图并保存。
def render_trajectory_figure(trajectory_bundle: Any, figure_cfg: Mapping[str, Any]) -> str:  # 把轨迹图写到磁盘。
    """渲染预测轨迹和真实轨迹对比图，并返回输出路径。

    作用：先构建规范对象，再按 2D/3D 选择不同的 matplotlib 画法，
    真实轨迹用实线、预测轨迹用虚线，最后保存图片并返回路径。

    参数:
        trajectory_bundle: 轨迹数据映射，必须包含 prediction_bundle 和 gt_bundle。
        figure_cfg: 渲染配置映射，必须包含 figure_path。

    返回值:
        str: 输出图片的路径字符串。

    异常:
        RuntimeError: matplotlib 未安装时抛出。
        TypeError: 输入类型不正确。
        ValueError: 缺少必需字段、序列数不一致、坐标维度不一致或字段缺失。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "figure_cfg_keys": list(figure_cfg.keys()) if isinstance(figure_cfg, Mapping) else None,
            "trajectory_bundle_type": type(trajectory_bundle).__name__,
        },
        "render_trajectory_figure 入口参数",
        prefix="[plotting]",
    )

    # 先构建规范对象，再按 2D/3D 选择不同的 matplotlib 画法。
    # 这里的 figure_cfg 决定输出路径，而 trajectory_bundle 决定画哪些线。
    figure_spec = build_trajectory_figure_spec(trajectory_bundle, figure_cfg)  # 先生成规范对象。

    try:  # matplotlib 是绘图依赖。
        import matplotlib  # 导入 matplotlib 主包。

        matplotlib.use("Agg")  # 使用无界面后端，适合脚本环境。
        from matplotlib import pyplot as plt  # 再导入 pyplot 画图接口。
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render trajectory figures.") from exc

    figure_path = figure_spec[FIGURE_PATH_KEY]  # 取输出路径（D9 单源引用常量）。
    figure_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在。

    fig = plt.figure()  # 创建画布。
    if figure_spec["dimension"] == 3:  # 3D 轨迹走三维坐标轴。
        axis = fig.add_subplot(111, projection="3d")  # 创建三维子图。
        axis.set_zlabel("pz")  # z 轴标签。
    else:  # 2D 情况下不需要三维投影。
        axis = fig.add_subplot(111)  # 2D 轨迹用普通坐标轴。

    for sequence_spec in figure_spec["sequence_specs"]:  # 逐序列画曲线。
        seq_label = sequence_spec["seq_label"]  # 序列标签（含 method + combo）。
        method_name = sequence_spec.get("method_name", "method")  # F-3: 方法名 (LNN/EKF/GT)
        gt_traj = sequence_spec["gt_traj"]  # 真实轨迹。
        pred_traj = sequence_spec["pred_traj"]  # 预测轨迹。
        if figure_spec["dimension"] == 3:  # 3D 时画三维曲线。
            # 这段先画真实轨迹，再画预测轨迹，方便视觉上直接比较偏差。
            axis.plot(  # 先画真实轨迹。
                gt_traj["px"],  # 真实轨迹的 x 坐标序列。
                gt_traj["py"],  # 真实轨迹的 y 坐标序列。
                gt_traj["pz"],  # 真实轨迹的 z 坐标序列。
                linestyle="-",  # 真实轨迹用实线表示。
                label=f"{seq_label}: GT",  # F-3: GT label
            )  # 真实轨迹绘制结束。
            axis.plot(  # 再画预测轨迹。
                pred_traj["px"],  # 预测轨迹的 x 坐标序列。
                pred_traj["py"],  # 预测轨迹的 y 坐标序列。
                pred_traj["pz"],  # 预测轨迹的 z 坐标序列。
                linestyle="--",  # 预测轨迹用虚线表示。
                label=f"{seq_label}: {method_name}",  # F-3: LNN/EKF label
            )  # 预测轨迹绘制结束。
        else:  # 2D 时只画平面曲线。
            # 2D 路径使用同样的顺序，保证图例和视觉对照一致。
            axis.plot(  # 先画真实轨迹。
                gt_traj["px"],  # 真实轨迹的 x 坐标序列。
                gt_traj["py"],  # 真实轨迹的 y 坐标序列。
                linestyle="-",  # 真实轨迹用实线表示。
                label=f"{seq_label}: GT",  # F-3: GT label
            )  # 真实轨迹绘制结束。
            axis.plot(  # 再画预测轨迹。
                pred_traj["px"],  # 预测轨迹的 x 坐标序列。
                pred_traj["py"],  # 预测轨迹的 y 坐标序列。
                linestyle="--",  # 预测轨迹用虚线表示。
                label=f"{seq_label}: {method_name}",  # F-3: LNN/EKF label
            )  # 预测轨迹绘制结束。

    # J-2: x/y axes have units [m]
    axis.set_xlabel("px [m]")  # x-axis world-frame with units (J-2)
    axis.set_ylabel("py [m]")  # y-axis world-frame with units (J-2)
    axis.legend()  # 显示图例，区分 gt 和 prediction。
    # J-3: Self-contained caption with data layer + method info
    title = figure_spec.get("caption")
    if title is None:
        title = (
            f"Figure F-3: Trajectory Overlay (LNN vs EKF vs GT) — "
            f"{figure_spec.get('data_layer', '①')} {seq_label}"
        )
    axis.set_title(title)  # J-3: self-contained caption with data layer
    fig.tight_layout()  # 自动调整布局。
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")  # J-8: ≥300 dpi per handbook
    plt.close(fig)  # 关闭画布，释放资源。
    return str(figure_path)  # 返回路径字符串。
