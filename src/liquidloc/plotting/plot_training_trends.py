"""
文件：`src/liquidloc/plotting/plot_training_trends.py`

这个模块负责把训练趋势报告渲染成 SVG 图。它不负责重新跑训练，
也不负责生成训练日志，只负责把已有报告整理成规范对象，再拼出 SVG。

训练趋势报告通常包含多个 split（如 train/val）的逐 epoch 指标，
以及每个 head（如 bias、risk、uwb_scaling、vio_scaling）的 RMSE。
渲染时会生成多个面板：顶部是总体指标趋势（如 selection_score 或 mean_loss），
下面每个 head 一个 RMSE 面板，最后可选地追加 fixed_probe 面板。

上游依赖：
- liquidloc.common.paths.build_output_path：统一输出目录策略。

下游调用者：
- scripts/13_generate_figures.py 等脚本通过 build_training_trend_figure_spec /
  render_training_trend_figure 渲染训练趋势图。
- 测试文件通过同一入口验证输出。

最容易看错的地方：
1. 报告里每个 split 的 entries 必须是 list，每个 entry 必须是 mapping。
2. metric_key 可以自动推断（优先 selection_score，其次 mean_loss），也可以手工指定。
3. head_keys 可以自动推断，也可以手工指定。
4. 输出只支持 SVG，其他后缀会直接报错。
5. 这里只做绘图，不做训练指标的计算或重算。
"""

from __future__ import annotations  # 支持前向类型注解，避免循环引用问题。

import math  # 用于数值比较、有限性判断和线性缩放。
from collections.abc import Mapping, Sequence  # 兼容字典式和序列式输入。
from html import escape  # 生成 SVG 时转义文本，防止破坏标签结构。
from pathlib import Path  # 输出路径统一用 Path 处理。

from liquidloc.common.constants import FIGURE_PATH_KEY  # D9 单源：绘图层 figure_path 键名常量。
from liquidloc.common.paths import build_output_path  # 统一使用项目内的输出目录规则。

# 默认的 split 名称列表，通常训练趋势报告包含 train 和 val 两个 split。
_DEFAULT_SPLITS = ("train", "val")
# 默认的输出 head 名称列表，按常见顺序排列。
_DEFAULT_HEADS = ("bias", "risk", "uwb_scaling", "vio_scaling")
# 内部面板键集合，这些字段不当作普通指标列。
_INTERNAL_PANEL_KEYS = {"epoch_index", "mean_loss", "selection_score"}


def _coerce_report(report) -> dict:
    """把训练趋势报告强制转成普通字典。

    Args:
        report: 原始训练趋势报告，必须是 mapping 类型。

    Returns:
        复制后的普通字典。

    Raises:
        TypeError: 如果 report 不是 mapping。
    """
    if not isinstance(report, Mapping):  # 报告必须是 mapping。
        raise TypeError("training_trend_report must be a mapping.")
    return dict(report)  # 复制一份，避免修改外部对象。


def _coerce_figure_cfg(figure_cfg) -> dict:
    """把可选配置复制成普通字典，避免后续修改外部对象。

    Args:
        figure_cfg: 可选的配置映射，可以为 None。

    Returns:
        复制后的普通字典；如果输入为 None 则返回空字典。

    Raises:
        TypeError: 如果 figure_cfg 不是 mapping 类型。
    """
    if figure_cfg is None:  # 没传配置就用空配置。
        return {}
    if not isinstance(figure_cfg, Mapping):  # 配置必须可按键访问。
        raise TypeError("figure_cfg must be a mapping when provided.")
    return dict(figure_cfg)  # 复制一份，避免修改外部对象。


def _resolve_figure_path(cfg: Mapping[str, object]) -> Path:
    """从配置中解析输出文件路径，确保是 SVG 格式且目录存在。

    Args:
        cfg: 已归一化的配置字典，可能包含 figure_path。

    Returns:
        解析后的绝对 Path 对象。

    Raises:
        ValueError: 如果路径后缀不是 .svg。
    """
    requested_path = cfg.get(FIGURE_PATH_KEY)  # 读取 figure_path 键（D9 单源引用常量）。
    if requested_path is None:  # 没指定就用默认输出路径。
        requested_path = build_output_path("figures", "training_trends.svg")  # 默认落到 figures/training_trends.svg。
    figure_path = Path(requested_path)  # 统一转成 Path。
    if figure_path.suffix.lower() != ".svg":  # 这里只支持 SVG。
        raise ValueError("render_training_trend_figure currently supports SVG output only.")
    figure_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在。
    return figure_path.resolve()  # 返回绝对路径，避免后续歧义。


def _resolve_splits(report: Mapping[str, object], cfg: Mapping[str, object]) -> list[str]:
    """决定要画哪些 split 的趋势。

    优先使用配置中显式指定的 splits 列表；其次从报告中自动检测
    实际存在的默认 split（train、val）。

    Args:
        report: 已归一化的训练趋势报告字典。
        cfg: 已归一化的配置字典。

    Returns:
        要绘制的 split 名称列表。

    Raises:
        TypeError: 如果配置中的 splits 不是序列类型。
        ValueError: 如果没有可用的 split，或报告中缺少某个 split。
    """
    configured_splits = cfg.get("splits")  # 如果用户显式指定，就按这个来。
    if configured_splits is None:  # 没显式指定时，从报告中自动检测默认 split。
        splits = [split for split in _DEFAULT_SPLITS if split in report]  # 只保留报告中实际存在的默认 split。
    else:  # 显式指定时，就按配置的顺序来。
        if not isinstance(configured_splits, Sequence) or isinstance(configured_splits, (str, bytes, bytearray)):  # 配置必须是非字符串序列。
            raise TypeError("figure_cfg['splits'] must be a sequence when provided.")
        splits = [str(item) for item in configured_splits]  # 统一转成字符串列表。
    if not splits:  # 一个都没有就无法画图。
        raise ValueError("training_trend_report must contain at least one split.")
    for split in splits:  # 逐个检查报告里是否真的有这个 split。
        if split not in report:  # 报告里缺少某个 split 就报错。
            raise ValueError(f"training_trend_report is missing split {split!r}.")
    return splits  # 返回最终决定要画的 split 列表。


def _resolve_metric_key(entry: Mapping[str, object], configured_metric_key: object) -> str:
    """决定用哪个字段作为总体指标趋势的纵轴。

    优先使用配置中显式指定的 metric_key；其次自动推断
    （优先 selection_score，其次 mean_loss）。

    Args:
        entry: 第一个 split 的第一条记录，用于自动推断。
        configured_metric_key: 配置中显式指定的 metric_key，可以为 None。

    Returns:
        指标键名。

    Raises:
        ValueError: 如果无法自动推断且未显式指定。
    """
    if configured_metric_key is None:  # 没显式指定时自动推断。
        if "selection_score" in entry and entry.get("selection_score") is not None:  # 优先用 selection_score。
            return "selection_score"
        if "mean_loss" in entry and entry.get("mean_loss") is not None:  # 其次用 mean_loss。
            return "mean_loss"
        raise ValueError("training trend entries must contain 'selection_score' or 'mean_loss'.")  # 两个都没有就无法推断。
    metric_key = str(configured_metric_key)  # 显式指定时转成字符串。
    if metric_key not in entry:  # 但指定的键必须存在于记录中。
        raise ValueError(f"training trend entry is missing metric_key={metric_key!r}.")
    return metric_key  # 返回指标键名。


def _resolve_head_keys(report: Mapping[str, object], cfg: Mapping[str, object], splits: Sequence[str]) -> list[str]:
    """决定要画哪些 head 的 RMSE 面板。

    优先使用配置中显式指定的 head_keys 列表；其次从报告中自动检测
    实际出现的默认 head；最后兜底到 head_metrics 里出现的所有键。

    Args:
        report: 已归一化的训练趋势报告字典。
        cfg: 已归一化的配置字典。
        splits: 已确定要绘制的 split 列表。

    Returns:
        要绘制的 head 键名列表。

    Raises:
        TypeError: 如果配置中的 head_keys 不是序列类型，或 entries 不是 list/mapping。
        ValueError: 如果没有任何可用的 head。
    """
    configured_heads = cfg.get("head_keys")  # 如果用户显式指定，就按这个来。
    if configured_heads is not None:  # 显式指定时，就按配置的顺序来。
        if not isinstance(configured_heads, Sequence) or isinstance(configured_heads, (str, bytes, bytearray)):  # 配置必须是非字符串序列。
            raise TypeError("figure_cfg['head_keys'] must be a sequence when provided.")
        head_keys = [str(item) for item in configured_heads]  # 统一转成字符串列表。
    else:  # 没显式指定时自动推断。
        head_keys: list[str] = []  # 保存推断出的 head 键名。
        for candidate in _DEFAULT_HEADS:  # 先按默认 head 顺序尝试。
            for split in splits:  # 在每个 split 中查找。
                entries = report.get(split) or []  # 取出当前 split 的 entries。
                if not isinstance(entries, list):  # entries 必须是 list。
                    raise TypeError(f"training_trend_report[{split!r}] must be a list.")
                if any(  # 如果任意一条 entry 的 head_metrics 包含这个 candidate。
                    isinstance(entry, Mapping)  # entry 必须是 mapping。
                    and isinstance(entry.get("head_metrics"), Mapping)  # head_metrics 也必须是 mapping。
                    and candidate in entry["head_metrics"]  # candidate 存在于 head_metrics 中。
                    for entry in entries  # 遍历所有 entry。
                ):
                    head_keys.append(candidate)  # 找到就加入列表。
                    break  # 只要一个 split 里有就够了，不需要继续检查其他 split。
        if not head_keys:  # 默认 head 都没找到，就兜底到 head_metrics 里出现的所有键。
            for split in splits:  # 在每个 split 中查找。
                entries = report.get(split) or []  # 取出当前 split 的 entries。
                if not isinstance(entries, list):  # entries 必须是 list。
                    raise TypeError(f"training_trend_report[{split!r}] must be a list.")
                for entry in entries:  # 逐条 entry 扫描。
                    if not isinstance(entry, Mapping):  # entry 必须是 mapping。
                        raise TypeError(f"training_trend_report[{split!r}] entries must be mappings.")
                    head_metrics = entry.get("head_metrics")  # 取出 head_metrics。
                    if isinstance(head_metrics, Mapping):  # head_metrics 必须是 mapping。
                        for key in head_metrics:  # 遍历 head_metrics 的所有键。
                            key_str = str(key)  # 统一转成字符串。
                            if key_str not in head_keys:  # 去重，避免重复加入。
                                head_keys.append(key_str)  # 加入列表。
    if not head_keys:  # 一个都没有就无法生成面板。
        raise ValueError("training_trend_report must contain at least one output head.")
    return head_keys  # 返回最终决定要画的 head 键名列表。


def _normalize_series_points(
    report: Mapping[str, object],
    *,
    splits: Sequence[str],
    metric_key: str,
    head_keys: Sequence[str],
) -> tuple[list[int], dict[str, list[dict[str, object]]], dict[str, list[dict[str, object]]]]:
    """把训练趋势报告整理成绘图用的序列点数据。

    遍历所有 split 和 entry，提取 epoch_index、总体指标值和每个 head 的 RMSE，
    返回 epoch 索引列表、总体指标序列和 head RMSE 序列。

    Args:
        report: 已归一化的训练趋势报告字典。
        splits: 已确定要绘制的 split 列表。
        metric_key: 总体指标键名（如 selection_score 或 mean_loss）。
        head_keys: 已确定要绘制的 head 键名列表。

    Returns:
        三元组：(epoch_indices, metric_series, head_series)
        - epoch_indices: 排序后的 epoch 索引列表。
        - metric_series: 以 split 为键、点列表为值的总体指标序列。
        - head_series: 以 head_key 为键、点列表为值的 RMSE 序列。

    Raises:
        TypeError: 如果 entries 不是 list/mapping，或 head_metrics 不是 mapping。
        ValueError: 如果没有任何 epoch entry。
    """
    epoch_index_set: set[int] = set()  # 收集所有出现过的 epoch 索引。
    metric_series: dict[str, list[dict[str, object]]] = {}  # 以 split 为键的总体指标序列。
    head_series: dict[str, list[dict[str, object]]] = {head_key: [] for head_key in head_keys}  # 以 head_key 为键的 RMSE 序列。
    for split in splits:  # 逐个 split 处理。
        raw_entries = report.get(split) or []  # 取出当前 split 的 entries。
        if not isinstance(raw_entries, list):  # entries 必须是 list。
            raise TypeError(f"training_trend_report[{split!r}] must be a list.")
        split_metric_points: list[dict[str, object]] = []  # 当前 split 的总体指标点列表。
        for entry in raw_entries:  # 逐条 entry 处理。
            if not isinstance(entry, Mapping):  # entry 必须是 mapping。
                raise TypeError(f"training_trend_report[{split!r}] entries must be mappings.")
            epoch_index = int(entry["epoch_index"])  # 取出 epoch 索引。
            epoch_index_set.add(epoch_index)  # 收集 epoch 索引。
            head_metrics = entry.get("head_metrics")  # 取出 head_metrics。
            if not isinstance(head_metrics, Mapping):  # head_metrics 必须是 mapping。
                raise TypeError("training trend entry head_metrics must be a mapping.")
            metric_value = entry.get(metric_key)  # 取出总体指标值。
            if metric_value is not None:  # 总体指标缺失时只跳过总体趋势点，不应连带丢掉 head RMSE。
                split_metric_points.append({"epoch_index": epoch_index, "value": float(metric_value)})  # 只在当前指标存在时追加总体指标点。
            for head_key in head_keys:  # 逐个 head 提取 RMSE。
                metric_row = head_metrics.get(head_key)  # 取出当前 head 的指标行。
                if not isinstance(metric_row, Mapping):  # 指标行必须是 mapping，缺失就跳过。
                    continue
                rmse_value = metric_row.get("rmse")  # 取出 RMSE 值。
                if rmse_value is None:  # RMSE 可以为 None，跳过。
                    continue
                head_series[head_key].append(  # 追加 RMSE 点。
                    {
                        "epoch_index": epoch_index,  # epoch 索引。
                        "value": float(rmse_value),  # RMSE 值。
                        "split": split,  # 所属 split，用于区分 train/val 曲线。
                    }
                )
        metric_series[split] = split_metric_points  # 保存当前 split 的总体指标序列。
    epoch_indices = sorted(epoch_index_set)  # 排序后的 epoch 索引列表。
    if not epoch_indices:  # 没有 epoch 就无法画图。
        raise ValueError("training_trend_report must contain at least one epoch entry.")
    return epoch_indices, metric_series, head_series  # 返回三元组。


def _resolve_fixed_probe_panels(
    report: Mapping[str, object],
    cfg: Mapping[str, object],
    *,
    head_keys: Sequence[str],
) -> list[dict[str, object]]:
    """从报告中提取 fixed_probe_trends 并整理成面板列表。

    fixed_probe 面板显示每个 probe 样本在不同 epoch 的 signed error 变化，
    每条线代表一个 split 中的一个固定样本。

    Args:
        report: 已归一化的训练趋势报告字典。
        cfg: 已归一化的配置字典。
        head_keys: 已确定要绘制的 head 键名列表。

    Returns:
        面板字典列表，每个面板包含 key、title 和 series。
        如果报告中没有 fixed_probe_trends，返回空列表。

    Raises:
        TypeError: 如果 fixed_probe_trends 的格式不正确。
    """
    raw_fixed_probe = report.get("fixed_probe_trends")  # 取出 fixed_probe_trends 数据。
    if not isinstance(raw_fixed_probe, Mapping):  # 如果不存在或不是 mapping，就跳过。
        return []
    configured_probe_keys = cfg.get("fixed_probe_keys")  # 如果用户显式指定 probe 键，就按这个来。
    if configured_probe_keys is not None:  # 显式指定时，就按配置的顺序来。
        if not isinstance(configured_probe_keys, Sequence) or isinstance(configured_probe_keys, (str, bytes, bytearray)):  # 配置必须是非字符串序列。
            raise TypeError("figure_cfg['fixed_probe_keys'] must be a sequence when provided.")
        probe_keys = [str(item) for item in configured_probe_keys]  # 统一转成字符串列表。
    else:  # 没显式指定时，默认使用 head_keys 作为 probe 键。
        probe_keys = list(head_keys)  # 复制一份 head_keys。
    panels: list[dict[str, object]] = []  # 保存面板列表。
    for probe_key in probe_keys:  # 逐个 probe 键处理。
        panel_series: list[dict[str, object]] = []  # 当前 probe 的序列点列表。
        for split in _DEFAULT_SPLITS:  # 只在默认 split 中查找。
            raw_entries = raw_fixed_probe.get(split) or []  # 取出当前 split 的 entries。
            if not isinstance(raw_entries, list):  # entries 必须是 list。
                raise TypeError(f"fixed_probe_trends[{split!r}] must be a list.")
            for entry in raw_entries:  # 逐条 entry 处理。
                if not isinstance(entry, Mapping):  # entry 必须是 mapping。
                    raise TypeError(f"fixed_probe_trends[{split!r}] entries must be mappings.")
                epoch_index = int(entry["epoch_index"])  # 取出 epoch 索引。
                fixed_probe_rows = entry.get("fixed_probe_rows") or []  # 取出 fixed_probe_rows。
                if not isinstance(fixed_probe_rows, list):  # fixed_probe_rows 必须是 list。
                    raise TypeError("fixed_probe_rows must be a list.")
                for probe_row in fixed_probe_rows:  # 逐个 probe 行处理。
                    if not isinstance(probe_row, Mapping):  # probe 行必须是 mapping。
                        raise TypeError("fixed_probe_rows entries must be mappings.")
                    sample_index = int(probe_row.get("sample_index_in_split", 0))  # 取出样本索引，默认为 0。
                    signed_error_by_head = probe_row.get("signed_error_by_head")  # 取出按 head 分组的 signed error。
                    if not isinstance(signed_error_by_head, Mapping):  # signed_error_by_head 必须是 mapping，缺失就跳过。
                        continue
                    probe_value = signed_error_by_head.get(probe_key)  # 取出当前 probe 键的 signed error。
                    if probe_value is None:  # 值为 None 就跳过。
                        continue
                    panel_series.append(  # 追加序列点。
                        {
                            "epoch_index": epoch_index,  # epoch 索引。
                            "value": float(probe_value),  # signed error 值。
                            "series_label": f"{split}:sample_{sample_index}",  # 序列标签，区分不同 split 和样本。
                        }
                    )
        if panel_series:  # 只有有数据时才创建面板。
            panels.append(  # 追加面板。
                {
                    "key": f"fixed_probe_signed_error:{probe_key}",  # 面板键名。
                    "title": f"Fixed probe signed error ({probe_key})",  # 面板标题。
                    "series": panel_series,  # 序列点列表。
                }
            )
    return panels  # 返回面板列表。


def _build_panel_specs(
    report: Mapping[str, object],
    cfg: Mapping[str, object],
) -> tuple[list[int], str, list[dict[str, object]]]:
    """把训练趋势报告整理成面板规格列表。

    面板顺序：先总体指标面板，再每个 head 的 RMSE 面板，最后 fixed_probe 面板。
    空面板会被过滤掉。

    Args:
        report: 已归一化的训练趋势报告字典。
        cfg: 已归一化的配置字典。

    Returns:
        三元组：(epoch_indices, metric_key, panels)
        - epoch_indices: 排序后的 epoch 索引列表。
        - metric_key: 总体指标键名。
        - panels: 面板字典列表，每个面板包含 key、title 和 series。

    Raises:
        TypeError: 如果输入格式不正确。
        ValueError: 如果没有可绘制的面板。
    """
    splits = _resolve_splits(report, cfg)  # 决定要画哪些 split。
    first_split_entries = report[splits[0]]  # 取第一个 split 的 entries。
    if not isinstance(first_split_entries, list) or not first_split_entries:  # entries 必须是非空 list。
        raise ValueError(f"training_trend_report[{splits[0]!r}] must be a non-empty list.")
    first_entry = first_split_entries[0]  # 取第一条 entry。
    if not isinstance(first_entry, Mapping):  # entry 必须是 mapping。
        raise TypeError(f"training_trend_report[{splits[0]!r}] entries must be mappings.")
    metric_key = _resolve_metric_key(first_entry, cfg.get("metric_key"))  # 决定总体指标键名。
    head_keys = _resolve_head_keys(report, cfg, splits)  # 决定要画哪些 head。
    epoch_indices, metric_series, head_series = _normalize_series_points(  # 整理成绘图序列。
        report,
        splits=splits,
        metric_key=metric_key,
        head_keys=head_keys,
    )
    panels: list[dict[str, object]] = [  # 第一个面板是总体指标趋势。
        {
            "key": metric_key,  # 面板键名就是指标键名。
            "title": metric_key.replace("_", " ").title(),  # 标题用人类可读格式。
            "series": [  # 序列点列表，每个 split 一条线。
                {"epoch_index": point["epoch_index"], "value": point["value"], "series_label": split}  # 点对象。
                for split, points in metric_series.items()  # 遍历每个 split。
                for point in points  # 遍历每个点。
            ],
        }
    ]
    for head_key in head_keys:  # 逐个 head 生成 RMSE 面板。
        panels.append(  # 追加 RMSE 面板。
            {
                "key": f"rmse:{head_key}",  # 面板键名格式为 rmse:head_key。
                "title": f"RMSE ({head_key})",  # 面板标题。
                "series": [  # 序列点列表。
                    {
                        "epoch_index": point["epoch_index"],  # epoch 索引。
                        "value": point["value"],  # RMSE 值。
                        "series_label": str(point["split"]),  # 序列标签，区分 train/val。
                    }
                    for point in head_series[head_key]  # 遍历当前 head 的所有点。
                ],
            }
        )
    panels.extend(_resolve_fixed_probe_panels(report, cfg, head_keys=head_keys))  # 追加 fixed_probe 面板。
    panels = [panel for panel in panels if panel["series"]]  # 过滤掉空面板。
    if not panels:  # 所有面板都空就无法画图。
        raise ValueError("training_trend_report did not yield any drawable panels.")
    return epoch_indices, metric_key, panels  # 返回三元组。


def build_training_trend_figure_spec(training_trend_report, figure_cfg=None) -> dict:
    """把训练趋势报告整理成可渲染的规范对象。

    这个函数只做输入校验、面板整理和文本规范化，不做绘图。

    Args:
        training_trend_report: 训练趋势报告 mapping，包含 train/val 等 split 数据。
        figure_cfg: 可选的配置映射，控制标题、尺寸、输出路径等。

    Returns:
        包含 figure_path、title、metric_key、epoch_indices、panels、
        width、panel_height 的绘图规格字典。

    Raises:
        TypeError: 如果输入类型不正确。
        ValueError: 如果没有可绘制的面板。
    """
    report = _coerce_report(training_trend_report)  # 先统一报告格式。
    cfg = _coerce_figure_cfg(figure_cfg)  # 再统一配置格式。
    epoch_indices, metric_key, panels = _build_panel_specs(report, cfg)  # 整理成面板规格。
    return {  # 返回后续 SVG 渲染直接使用的规范对象。
        FIGURE_PATH_KEY: _resolve_figure_path(cfg),  # 最终输出文件路径（D9 单源引用常量）。
        "title": str(cfg.get("title") or "Training trends"),  # 默认标题。
        "metric_key": metric_key,  # 总体指标键名。
        "epoch_indices": epoch_indices,  # 排序后的 epoch 索引列表。
        "panels": panels,  # 面板字典列表。
        "width": max(int(cfg.get("width", 1280)), 800),  # 画布宽度有下限，避免太窄。
        "panel_height": max(int(cfg.get("panel_height", 220)), 180),  # 面板高度也有下限。
    }


def _scale_linear(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    """把数值从数据域线性映射到像素区间。

    当 domain_min 和 domain_max 相等时（区间没有跨度），返回目标区间中点。

    Args:
        value: 要映射的数值。
        domain_min: 数据域最小值。
        domain_max: 数据域最大值。
        range_min: 目标区间最小值（像素）。
        range_max: 目标区间最大值（像素）。

    Returns:
        映射后的像素坐标值。
    """
    if math.isclose(domain_min, domain_max):  # 区间没有跨度时返回中点。
        return (range_min + range_max) / 2.0
    ratio = (value - domain_min) / (domain_max - domain_min)  # 先算归一化比例。
    return range_min + (range_max - range_min) * ratio  # 再映射到目标区间。


def _build_svg(spec: Mapping[str, object]) -> str:
    """根据规范对象拼出 SVG 字符串。

    每个面板画成一个子图，包含标题、Y 轴刻度、零轴、X 轴刻度、
    折线和图例。面板按垂直方向堆叠。

    Args:
        spec: 绘图规格字典，包含 width、panel_height、title、
              epoch_indices、panels 等。

    Returns:
        完整的 SVG 字符串。
    """
    width = int(spec["width"])  # 画布宽度。
    panel_height = int(spec["panel_height"])  # 单个面板高度。
    title_height = 56  # 标题区高度。
    footer_height = 28  # 底部留白。
    panel_gap = 20  # 面板之间的间距。
    panel_count = len(spec["panels"])  # 面板数量。
    height = title_height + footer_height + panel_count * panel_height + max(panel_count - 1, 0) * panel_gap  # 总高度计算。
    margin_left = 84  # 左侧给 Y 轴标签留空间。
    margin_right = 40  # 右侧留白。
    panel_title_height = 22  # 面板标题所占空间。
    plot_padding_top = 10  # 绘图区上边距。
    plot_padding_bottom = 30  # 绘图区下边距。
    plot_width = width - margin_left - margin_right  # 真正用于画折线的宽度。
    colors = ("#2563eb", "#dc2626", "#0f766e", "#9333ea", "#ea580c", "#0891b2")  # 每条线循环使用不同颜色。
    svg_parts = [  # SVG 片段列表，最后统一拼接。
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',  # SVG 根标签。
        '<rect width="100%" height="100%" fill="white"/>',  # 白色背景。
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" font-size="22" font-family="Arial">{escape(str(spec["title"]))}</text>',  # 顶部标题。
    ]
    epoch_indices = list(spec["epoch_indices"])  # 取出 epoch 索引列表。
    epoch_min = float(min(epoch_indices))  # 最小 epoch 值。
    epoch_max = float(max(epoch_indices))  # 最大 epoch 值。
    for panel_index, panel in enumerate(spec["panels"]):  # 逐个面板绘制。
        panel_top = title_height + panel_index * (panel_height + panel_gap)  # 这个面板的顶部坐标。
        plot_top = panel_top + panel_title_height + plot_padding_top  # 实际绘图区域顶部。
        plot_height = panel_height - panel_title_height - plot_padding_top - plot_padding_bottom  # 绘图区高度。
        svg_parts.append(  # 追加面板标题文本。
            f'<text x="{margin_left}" y="{panel_top + 16:.1f}" text-anchor="start" font-size="15" font-family="Arial">{escape(str(panel["title"]))}</text>'
        )
        all_values = [float(point["value"]) for point in panel["series"]]  # 收集当前面板所有数值。
        value_min = min(all_values)  # 数值最小值。
        value_max = max(all_values)  # 数值最大值。
        if math.isclose(value_min, value_max):  # 如果所有值一样，就强行拉开一点。
            delta = 1.0 if math.isclose(value_min, 0.0) else abs(value_min) * 0.1  # 拉开的幅度。
            value_min -= delta  # 下限下移。
            value_max += delta  # 上限上移。
        for tick_index in range(5):  # 5 个 Y 轴刻度。
            tick_value = value_min + (value_max - value_min) * (tick_index / 4.0)  # 刻度值按线性插值算。
            tick_y = _scale_linear(tick_value, value_min, value_max, plot_top + plot_height, plot_top)  # 映射到像素坐标（注意 Y 轴方向翻转）。
            svg_parts.append(  # 追加 Y 轴网格线。
                f'<line x1="{margin_left:.2f}" y1="{tick_y:.2f}" x2="{margin_left + plot_width:.2f}" y2="{tick_y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
            )
            svg_parts.append(  # 追加 Y 轴刻度文本。
                f'<text x="{margin_left - 8:.2f}" y="{tick_y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial" fill="#4b5563">{tick_value:.3g}</text>'
            )
        if value_min <= 0.0 <= value_max:  # 如果零点在值域内，就画零轴。
            zero_y = _scale_linear(0.0, value_min, value_max, plot_top + plot_height, plot_top)  # 零轴 Y 像素坐标。
            svg_parts.append(  # 追加零轴横线。
                f'<line x1="{margin_left:.2f}" y1="{zero_y:.2f}" x2="{margin_left + plot_width:.2f}" y2="{zero_y:.2f}" stroke="#111827" stroke-width="1.2"/>'
            )
        for epoch_value in epoch_indices:  # 逐个 epoch 画 X 轴刻度。
            tick_x = _scale_linear(float(epoch_value), epoch_min, epoch_max, margin_left, margin_left + plot_width)  # epoch 映射到像素坐标。
            svg_parts.append(  # 追加 X 轴网格线。
                f'<line x1="{tick_x:.2f}" y1="{plot_top:.2f}" x2="{tick_x:.2f}" y2="{plot_top + plot_height:.2f}" stroke="#f3f4f6" stroke-width="1"/>'
            )
            svg_parts.append(  # 追加 X 轴刻度文本。
                f'<text x="{tick_x:.2f}" y="{plot_top + plot_height + 18:.2f}" text-anchor="middle" font-size="11" font-family="Arial" fill="#4b5563">{epoch_value}</text>'
            )
        grouped_series: dict[str, list[dict[str, object]]] = {}  # 按序列标签分组，同一标签的点画一条折线。
        for point in panel["series"]:  # 遍历面板所有点。
            grouped_series.setdefault(str(point["series_label"]), []).append(dict(point))  # 按标签分组追加。
        for series_index, (series_label, series_points) in enumerate(grouped_series.items()):  # 逐条折线绘制。
            series_points = sorted(series_points, key=lambda item: int(item["epoch_index"]))  # 按 epoch 排序，保证折线顺序正确。
            path_segments: list[str] = []  # SVG path 指令片段。
            color = colors[series_index % len(colors)]  # 循环选色。
            for point_index, point in enumerate(series_points):  # 逐点计算像素坐标。
                point_x = _scale_linear(float(point["epoch_index"]), epoch_min, epoch_max, margin_left, margin_left + plot_width)  # X 像素坐标。
                point_y = _scale_linear(float(point["value"]), value_min, value_max, plot_top + plot_height, plot_top)  # Y 像素坐标（方向翻转）。
                command = "M" if point_index == 0 else "L"  # 第一个点用 M（移动到），后续用 L（连线到）。
                path_segments.append(f"{command} {point_x:.2f} {point_y:.2f}")  # 追加路径指令。
                svg_parts.append(  # 追加数据点圆圈。
                    f'<circle cx="{point_x:.2f}" cy="{point_y:.2f}" r="3.2" fill="{color}"/>'
                )
            if path_segments:  # 有路径指令才画折线。
                svg_parts.append(  # 追加折线路径。
                    f'<path d="{" ".join(path_segments)}" fill="none" stroke="{color}" stroke-width="2"/>'
                )
            legend_x = margin_left + (series_index * 160)  # 图例 X 坐标，每条线偏移 160 像素。
            legend_y = panel_top + 16  # 图例 Y 坐标，和面板标题对齐。
            svg_parts.append(  # 追加图例线段。
                f'<line x1="{legend_x:.2f}" y1="{legend_y:.2f}" x2="{legend_x + 18:.2f}" y2="{legend_y:.2f}" stroke="{color}" stroke-width="2"/>'
            )
            svg_parts.append(  # 追加图例文本。
                f'<text x="{legend_x + 24:.2f}" y="{legend_y + 4:.2f}" text-anchor="start" font-size="11" font-family="Arial" fill="#111827">{escape(series_label)}</text>'
            )
    svg_parts.append("</svg>")  # SVG 结束标签。
    return "".join(svg_parts)  # 拼成完整字符串并返回。


def render_training_trend_figure(training_trend_report, figure_cfg=None):
    """把训练趋势报告渲染成 SVG 文件，并返回路径或清单。

    这个函数是模块对外的主要渲染入口：它先把输入规范化，
    再生成规范对象，然后拼出 SVG 文本，最后写入磁盘。

    Args:
        training_trend_report: 训练趋势报告 mapping，包含 train/val 等 split 数据。
        figure_cfg: 可选的配置映射，控制标题、尺寸、输出路径和清单输出行为。

    Returns:
        默认返回输出路径字符串。如果 figure_cfg 中 return_manifest 为真，
        则返回包含 figure_path、metric_key 和 panel_keys 的字典。

    Raises:
        TypeError: 如果输入类型不正确。
        ValueError: 如果没有可绘制的面板。
    """
    from liquidloc.common.tee_logger import print_dict  # 入口参数日志（D8 print_dict 增强）
    print_dict(
        {
            "figure_cfg_keys": list(figure_cfg.keys()) if isinstance(figure_cfg, Mapping) else None,
            "training_trend_report_type": type(training_trend_report).__name__,
        },
        "render_training_trend_figure 入口参数",
        prefix="[plotting]",
    )
    cfg = _coerce_figure_cfg(figure_cfg)  # 先统一配置格式。
    spec = build_training_trend_figure_spec(training_trend_report, cfg)  # 再生成规范对象。
    figure_path = spec[FIGURE_PATH_KEY]  # 输出路径（D9 单源引用常量）。
    figure_path.write_text(_build_svg(spec), encoding="utf-8")  # 写出 SVG 内容。
    if cfg.get("return_manifest"):  # 如果调用者要求清单，就返回描述对象。
        return {  # 返回清单对象。
            FIGURE_PATH_KEY: str(figure_path),  # 输出路径字符串（D9 单源引用常量）。
            "metric_key": spec["metric_key"],  # 使用的总体指标键名。
            "panel_keys": [str(panel["key"]) for panel in spec["panels"]],  # 所有面板的键名列表。
        }
    return str(figure_path)  # 默认返回路径字符串，方便外部直接使用。
