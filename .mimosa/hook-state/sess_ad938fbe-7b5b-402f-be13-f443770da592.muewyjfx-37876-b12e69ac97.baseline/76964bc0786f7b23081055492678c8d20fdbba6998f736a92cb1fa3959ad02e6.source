"""总结训练诊断里各输出头的逐 epoch 误差趋势。

这个脚本读取训练报告、损失诊断和可选的 epoch 预测对照文件，然后按输出头
聚合每个 epoch 的误差、RMSE、加权 RMSE 等指标，最后写成一个可供后续分析
或汇报使用的 JSON 文件。

上游依赖：05_train_lstm.py 或 06_train_liquid.py 产出的训练报告和损失诊断文件。
下游调用者：13_generate_figures.py 会消费训练趋势报告来渲染训练趋势图。
核心变量：output_heads 列表，定义了当前模型有哪些输出头需要统计。
"""

from __future__ import annotations  # 允许使用更现代的类型注解写法。

import argparse  # 解析命令行参数。
import json  # 读写 JSON 文件。
import math  # 计算 RMSE 时需要开平方。
from collections.abc import Mapping  # 判断映射类型。
from pathlib import Path  # 统一处理文件路径。
from typing import Any  # 标注任意 JSON 风格结构。

_DEFAULT_OUTPUT_HEADS = ("bias", "risk", "uwb_scaling", "vio_scaling")  # 默认输出头列表，当 epoch 预测文件里没有指定时使用。
_HEAD_UNITS = {  # 每个输出头的物理单位，用于报告里标注量纲。
    "bias": "m",  # 偏置头的单位是米。
    "risk": "unitless",  # 风险头是无量纲的。
    "uwb_scaling": "unitless",  # UWB 缩放头是无量纲的。
    "vio_scaling": "unitless",  # VIO 缩放头是无量纲的。
}  # 输出头单位映射结束。
_SNAPSHOT_TOLERANCE = 1e-5  # 快照对照时的浮点容差，超过这个差值就认为不一致。


def _round_float(value: float | None) -> float | None:
    """把浮点数统一四舍五入到固定精度。"""
    if value is None:  # 空值直接返回。
        return None  # 保留空值。
    return round(float(value), 8)  # 统一保留 8 位小数。


def _load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    from liquidloc.common.io_utils import read_json
    return read_json(path)  # 按 UTF-8 读取并解析 JSON。


def _resolve_output_heads(epoch_predictions_payload: Mapping[str, Any] | None) -> list[str]:
    """从 epoch 预测文件里解析输出头列表，失败时回退默认值。"""
    raw_heads = epoch_predictions_payload.get("output_heads") if epoch_predictions_payload else None  # 尝试从预测文件里取输出头。
    if isinstance(raw_heads, list) and raw_heads and all(isinstance(item, str) and item for item in raw_heads):  # 验证列表非空且每个元素都是非空字符串。
        return list(raw_heads)  # 验证通过就用文件里指定的输出头。
    return list(_DEFAULT_OUTPUT_HEADS)  # 否则回退到默认输出头列表。


def _default_output_path(loss_diagnostics_path: Path) -> Path:
    """根据输入文件名推导默认输出路径。"""
    if loss_diagnostics_path.name.endswith("_loss_diagnostics.json"):  # 如果文件名以 _loss_diagnostics.json 结尾。
        stem = loss_diagnostics_path.name[: -len("_loss_diagnostics.json")]  # 去掉后缀得到文件名主干。
        return loss_diagnostics_path.with_name(f"{stem}_head_error_trends.json")  # 用主干拼出趋势文件名。
    return loss_diagnostics_path.with_name(f"{loss_diagnostics_path.stem}_head_error_trends.json")  # 其他情况直接在文件名后追加。


def _resolve_input_paths(
    *,
    train_report_path: Path | None,
    loss_diagnostics_path: Path | None,
    epoch_predictions_path: Path | None,
) -> tuple[Path, Path | None]:
    """统一整理输入路径，优先从训练报告里补缺省路径。"""
    resolved_loss_path = loss_diagnostics_path  # 先用命令行传入的损失诊断路径。
    resolved_epoch_predictions_path = epoch_predictions_path  # 先用命令行传入的 epoch 预测路径。
    if train_report_path is not None:  # 如果提供了训练报告，就从中补全缺失路径。
        train_report_payload = _load_json(train_report_path)  # 读取训练报告。
        if resolved_loss_path is None:  # 如果命令行没给损失诊断路径。
            raw_loss_path = str(train_report_payload.get("loss_diagnostics_path") or "").strip()  # 从训练报告里取。
            if not raw_loss_path:  # 训练报告里也没有就报错。
                raise KeyError("train_report is missing loss_diagnostics_path")  # 缺失关键字段。
            resolved_loss_path = Path(raw_loss_path)  # 转成路径对象。
        if resolved_epoch_predictions_path is None:  # 如果命令行没给 epoch 预测路径。
            raw_epoch_path = str(train_report_payload.get("epoch_predictions_vs_targets_path") or "").strip()  # 从训练报告里取。
            if raw_epoch_path:  # 训练报告里有就补上。
                resolved_epoch_predictions_path = Path(raw_epoch_path)  # 转成路径对象。
    if resolved_loss_path is None:  # 最终损失诊断路径不能为空。
        raise ValueError("one of --loss-diagnostics or --train-report is required")  # 必须提供其中一个。
    return resolved_loss_path, resolved_epoch_predictions_path  # 返回解析后的路径元组。


def _index_epoch_snapshots(
    epoch_predictions_payload: Mapping[str, Any] | None,
) -> dict[tuple[str, int], dict[str, Any]]:
    """把 epoch 预测记录索引成 (split, epoch) 键，便于对照。"""
    if epoch_predictions_payload is None:  # 没有 epoch 预测数据就返回空索引。
        return {}  # 空索引。
    index: dict[tuple[str, int], dict[str, Any]] = {}  # 用 (split, epoch_index) 作为键的索引字典。
    for split in ("train", "val"):  # 遍历训练集和验证集。
        raw_entries = epoch_predictions_payload.get(split) or []  # 取出当前 split 的条目列表。
        if not isinstance(raw_entries, list):  # 条目必须是列表。
            raise TypeError(f"epoch_predictions_payload[{split!r}] must be a list")  # 类型不对就报错。
        for entry in raw_entries:  # 逐条处理。
            if not isinstance(entry, Mapping):  # 每条必须是映射。
                raise TypeError(f"epoch_predictions_payload[{split!r}] entries must be mappings")  # 类型不对就报错。
            epoch_index = int(entry["epoch_index"])  # 取出 epoch 编号。
            index[(split, epoch_index)] = dict(entry)  # 以 (split, epoch_index) 为键存入索引。
    return index  # 返回构建好的索引。


def _empty_head_aggregate() -> dict[str, float | int | list[float]]:
    """构造一个输出头聚合桶的初始状态。"""
    return {  # 所有累加器初始化为零，浮点累加器用列表收集以避免精度漂移。
        "active_sample_count": 0,  # 活跃样本计数。
        "sample_weight_sum": 0.0,  # 样本权重总和。
        "prediction_sum": 0.0,  # 预测值总和。
        "target_sum": 0.0,  # 目标值总和。
        "signed_error_sum": 0.0,  # 有符号误差总和。
        "abs_error_sum": 0.0,  # 绝对误差总和。
        "squared_error_sum": [],  # 平方误差收集列表，用 math.fsum 求和以避免浮点漂移。
        "weighted_prediction_sum": 0.0,  # 加权预测值总和。
        "weighted_target_sum": 0.0,  # 加权目标值总和。
        "weighted_signed_error_sum": 0.0,  # 加权有符号误差总和。
        "weighted_abs_error_sum": 0.0,  # 加权绝对误差总和。
        "weighted_squared_error_sum": [],  # 加权平方误差收集列表，用 math.fsum 求和以避免浮点漂移。
    }  # 聚合桶初始化结束。


def _safe_divide(numerator: float, denominator: float) -> float | None:
    """安全除法，分母非正时返回空值。"""
    if denominator <= 0.0:  # 分母为零或负数时不做除法。
        return None  # 返回空值避免除零错误。
    return numerator / denominator  # 正常计算除法。


def _build_head_metric(
    *,
    aggregate: Mapping[str, float | int],
    unit: str,
) -> dict[str, Any]:
    """把一个输出头的聚合桶转成可读指标。"""
    count = int(aggregate["active_sample_count"])  # 活跃样本数。
    weight_sum = float(aggregate["sample_weight_sum"])  # 权重总和。
    mean_prediction = _safe_divide(float(aggregate["prediction_sum"]), float(count))  # 平均预测值。
    mean_target = _safe_divide(float(aggregate["target_sum"]), float(count))  # 平均目标值。
    mean_signed_error = _safe_divide(float(aggregate["signed_error_sum"]), float(count))  # 平均有符号误差。
    mae = _safe_divide(float(aggregate["abs_error_sum"]), float(count))  # 平均绝对误差。
    squared_error_total = math.fsum(aggregate["squared_error_sum"]) if isinstance(aggregate["squared_error_sum"], list) else float(aggregate["squared_error_sum"])  # 用 math.fsum 精确求和。
    rmse = _safe_divide(squared_error_total, float(count))  # 均方误差（开方前）。
    weighted_mean_prediction = _safe_divide(float(aggregate["weighted_prediction_sum"]), weight_sum)  # 加权平均预测值。
    weighted_mean_target = _safe_divide(float(aggregate["weighted_target_sum"]), weight_sum)  # 加权平均目标值。
    weighted_mean_signed_error = _safe_divide(float(aggregate["weighted_signed_error_sum"]), weight_sum)  # 加权平均有符号误差。
    weighted_mae = _safe_divide(float(aggregate["weighted_abs_error_sum"]), weight_sum)  # 加权平均绝对误差。
    weighted_squared_error_total = math.fsum(aggregate["weighted_squared_error_sum"]) if isinstance(aggregate["weighted_squared_error_sum"], list) else float(aggregate["weighted_squared_error_sum"])  # 用 math.fsum 精确求和。
    weighted_rmse = _safe_divide(weighted_squared_error_total, weight_sum)  # 加权均方误差（开方前）。
    return {  # 组装指标字典。
        "unit": unit,  # 物理单位。
        "active_sample_count": count,  # 活跃样本数。
        "sample_weight_sum": _round_float(weight_sum),  # 权重总和。
        "prediction_mean": _round_float(mean_prediction),  # 平均预测值。
        "target_mean": _round_float(mean_target),  # 平均目标值。
        "mean_signed_error": _round_float(mean_signed_error),  # 平均有符号误差。
        "mae": _round_float(mae),  # 平均绝对误差。
        "rmse": _round_float(math.sqrt(rmse) if rmse is not None else None),  # 均方根误差（对 MSE 开方）。
        "weighted_prediction_mean": _round_float(weighted_mean_prediction),  # 加权平均预测值。
        "weighted_target_mean": _round_float(weighted_mean_target),  # 加权平均目标值。
        "weighted_mean_signed_error": _round_float(weighted_mean_signed_error),  # 加权平均有符号误差。
        "weighted_mae": _round_float(weighted_mae),  # 加权平均绝对误差。
        "weighted_rmse": _round_float(math.sqrt(weighted_rmse) if weighted_rmse is not None else None),  # 加权均方根误差。
    }  # 指标字典结束。


def _validate_against_snapshot(
    *,
    split: str,
    epoch_index: int,
    snapshot_entry: Mapping[str, Any] | None,
    metrics_by_head: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """把当前统计结果和训练快照对照，检查是否一致。"""
    if snapshot_entry is None:  # 没有快照数据就标记为不可用。
        return {"available": False}  # 快照不可用。
    if "active_sample_count_by_head" not in snapshot_entry:  # 旧格式快照没有按头拆分的计数。
        return {  # 标记为旧格式，不能严格比较。
            "available": True,  # 快照存在。
            "validation_mode": "legacy_snapshot_not_strictly_comparable",  # 但格式不兼容，不能严格对照。
        }  # 旧格式标记结束。
    counts_match = True  # 样本数是否一致，默认为是。
    means_match = True  # 均值是否一致，默认为是。
    snapshot_counts = dict(snapshot_entry.get("active_sample_count_by_head") or {})  # 快照中的按头样本计数。
    snapshot_prediction_mean = dict(snapshot_entry.get("prediction_mean") or {})  # 快照中的按头预测均值。
    snapshot_target_mean = dict(snapshot_entry.get("target_mean") or {})  # 快照中的按头目标均值。
    for head_key, metric_row in metrics_by_head.items():  # 逐个输出头比对。
        if int(snapshot_counts.get(head_key, 0)) != int(metric_row["active_sample_count"]):  # 样本数不一致。
            counts_match = False  # 标记为不一致。
        prediction_mean = metric_row["prediction_mean"]  # 当前预测均值。
        target_mean = metric_row["target_mean"]  # 当前目标均值。
        snapshot_prediction_value = snapshot_prediction_mean.get(head_key)  # 快照预测均值。
        snapshot_target_value = snapshot_target_mean.get(head_key)  # 快照目标均值。
        if prediction_mean is None:  # 当前预测均值为空。
            if snapshot_prediction_value is not None:  # 但快照有值，说明不一致。
                means_match = False  # 标记为不一致。
        elif snapshot_prediction_value is None or abs(float(prediction_mean) - float(snapshot_prediction_value)) > _SNAPSHOT_TOLERANCE:  # 预测均值差异超容差。
            means_match = False  # 标记为不一致。
        if target_mean is None:  # 当前目标均值为空。
            if snapshot_target_value is not None:  # 但快照有值，说明不一致。
                means_match = False  # 标记为不一致。
        elif snapshot_target_value is None or abs(float(target_mean) - float(snapshot_target_value)) > _SNAPSHOT_TOLERANCE:  # 目标均值差异超容差。
            means_match = False  # 标记为不一致。
    if not counts_match or not means_match:  # 如果有任何不一致就报错。
        raise ValueError(  # 抛出快照不匹配异常。
            f"snapshot mismatch for split={split!r} epoch={epoch_index}: "  # 包含 split 和 epoch 信息。
            f"counts_match={counts_match}, means_match={means_match}"  # 包含具体不一致项。
        )  # 异常构造结束。
    return {  # 返回验证通过的结果。
        "available": True,  # 快照可用。
        "validation_mode": "active_head_snapshot_strict",  # 严格对照模式。
        "counts_match": True,  # 样本数一致。
        "means_match": True,  # 均值一致。
    }  # 验证结果结束。


def _build_epoch_head_metrics(
    *,
    epoch_entry: Mapping[str, Any],
    output_heads: list[str],
    snapshot_entry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """把某个 epoch 的批次明细聚合成每个输出头的指标。"""
    aggregates = {head_key: _empty_head_aggregate() for head_key in output_heads}  # 为每个输出头初始化聚合桶。
    batches = epoch_entry.get("batches") or []  # 取出批次列表。
    if not isinstance(batches, list):  # 批次必须是列表。
        raise TypeError("epoch_entry['batches'] must be a list")  # 类型不对就报错。
    for batch in batches:  # 逐批次累加。
        if not isinstance(batch, Mapping):  # 每个批次必须是映射。
            raise TypeError("loss diagnostics batch entries must be mappings")  # 类型不对就报错。
        predictions = batch.get("prediction") or []  # 取出预测值矩阵。
        targets = batch.get("target") or []  # 取出目标值矩阵。
        active_keys_rows = batch.get("active_keys") or []  # 取出每个样本的活跃输出头列表。
        sample_weights = batch.get("sample_weights") or [1.0] * len(predictions)  # 取出样本权重，缺失时默认全 1。
        if not (len(predictions) == len(targets) == len(active_keys_rows) == len(sample_weights)):  # 行数必须一致。
            raise ValueError("prediction/target/active_keys/sample_weights row counts must match per batch")  # 不一致就报错。
        for prediction_row, target_row, active_keys, sample_weight in zip(  # 逐样本累加。
            predictions,  # 预测行。
            targets,  # 目标行。
            active_keys_rows,  # 活跃键行。
            sample_weights,  # 权重行。
            strict=True,  # 严格模式，长度不一致就报错。
        ):
            if len(prediction_row) != len(output_heads) or len(target_row) != len(output_heads):  # 列数必须和输出头数一致。
                raise ValueError("prediction/target row width must match output head count")  # 不一致就报错。
            active_key_set = set(active_keys)  # 把活跃键转成集合，加速查找。
            weight = float(sample_weight)  # 样本权重转浮点数。
            for head_index, head_key in enumerate(output_heads):  # 逐个输出头累加。
                if head_key not in active_key_set:  # 不活跃的头跳过。
                    continue  # 跳过不活跃头。
                prediction_value = float(prediction_row[head_index])  # 当前头的预测值。
                target_value = float(target_row[head_index])  # 当前头的目标值。
                error = prediction_value - target_value  # 有符号误差。
                aggregate = aggregates[head_key]  # 取出当前头的聚合桶。
                aggregate["active_sample_count"] = int(aggregate["active_sample_count"]) + 1  # 活跃样本数加一。
                aggregate["sample_weight_sum"] = float(aggregate["sample_weight_sum"]) + weight  # 累加权重。
                aggregate["prediction_sum"] = float(aggregate["prediction_sum"]) + prediction_value  # 累加预测值。
                aggregate["target_sum"] = float(aggregate["target_sum"]) + target_value  # 累加目标值。
                aggregate["signed_error_sum"] = float(aggregate["signed_error_sum"]) + error  # 累加有符号误差。
                aggregate["abs_error_sum"] = float(aggregate["abs_error_sum"]) + abs(error)  # 累加绝对误差。
                aggregate["squared_error_sum"].append(error * error)  # 收集平方误差，后续用 math.fsum 求和。
                aggregate["weighted_prediction_sum"] = float(aggregate["weighted_prediction_sum"]) + weight * prediction_value  # 累加加权预测值。
                aggregate["weighted_target_sum"] = float(aggregate["weighted_target_sum"]) + weight * target_value  # 累加加权目标值。
                aggregate["weighted_signed_error_sum"] = float(aggregate["weighted_signed_error_sum"]) + weight * error  # 累加加权有符号误差。
                aggregate["weighted_abs_error_sum"] = float(aggregate["weighted_abs_error_sum"]) + weight * abs(error)  # 累加加权绝对误差。
                aggregate["weighted_squared_error_sum"].append(weight * error * error)  # 收集加权平方误差，后续用 math.fsum 求和。
    metrics_by_head = {  # 把每个头的聚合桶转成指标字典。
        head_key: _build_head_metric(aggregate=aggregate, unit=_HEAD_UNITS.get(head_key, "unitless"))  # 构建指标并标注单位。
        for head_key, aggregate in aggregates.items()  # 遍历所有输出头。
    }  # 指标字典构建结束。
    snapshot_validation = _validate_against_snapshot(  # 和快照对照验证。
        split=str(epoch_entry["split"]),  # 当前 split。
        epoch_index=int(epoch_entry["epoch_index"]),  # 当前 epoch。
        snapshot_entry=snapshot_entry,  # 对应的快照条目。
        metrics_by_head=metrics_by_head,  # 当前计算的指标。
    )  # 快照验证结束。
    return {  # 返回当前 epoch 的完整指标。
        "split": str(epoch_entry["split"]),  # split 名称。
        "epoch_index": int(epoch_entry["epoch_index"]),  # epoch 编号。
        "mean_loss": _round_float(float(epoch_entry["mean_loss"])),  # 平均损失。
        "selection_score": _round_float(float(epoch_entry["selection_score"])) if epoch_entry.get("selection_score") is not None else None,  # 选择分数（可选）。
        "batch_count": int(epoch_entry.get("batch_count") or 0),  # 批次数。
        "head_metrics": metrics_by_head,  # 按输出头拆分的指标。
        "snapshot_validation": snapshot_validation,  # 快照验证结果。
    }  # epoch 指标结束。


def _build_fixed_probe_trends(
    epoch_predictions_payload: Mapping[str, Any] | None,
    *,
    output_heads: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """从 epoch 预测文件里提取固定探针样本的逐 epoch 趋势。"""
    if epoch_predictions_payload is None:  # 没有 epoch 预测数据就返回空趋势。
        return {"train": [], "val": []}  # 两个 split 都为空。
    report: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}  # 保存趋势报告。
    for split in ("train", "val"):  # 遍历训练集和验证集。
        raw_entries = epoch_predictions_payload.get(split) or []  # 取出当前 split 的条目。
        if not isinstance(raw_entries, list):  # 条目必须是列表。
            raise TypeError(f"epoch_predictions_payload[{split!r}] must be a list")  # 类型不对就报错。
        split_rows: list[dict[str, Any]] = []  # 保存当前 split 的趋势行。
        for entry in raw_entries:  # 逐条处理。
            if not isinstance(entry, Mapping):  # 每条必须是映射。
                raise TypeError(f"epoch_predictions_payload[{split!r}] entries must be mappings")  # 类型不对就报错。
            fixed_probe_rows = entry.get("fixed_probe_rows") or []  # 取出固定探针行。
            if not isinstance(fixed_probe_rows, list):  # 探针行必须是列表。
                raise TypeError("fixed_probe_rows must be a list when provided")  # 类型不对就报错。
            probe_rows_payload: list[dict[str, Any]] = []  # 保存规范化后的探针行。
            for probe_row in fixed_probe_rows:  # 逐行处理探针数据。
                if not isinstance(probe_row, Mapping):  # 每行必须是映射。
                    raise TypeError("fixed_probe_rows entries must be mappings")  # 类型不对就报错。
                prediction_by_head = dict(probe_row.get("prediction_by_head") or {})  # 按头的预测值。
                target_by_head = dict(probe_row.get("target_by_head") or {})  # 按头的目标值。
                signed_error_by_head = dict(probe_row.get("signed_error_by_head") or {})  # 按头的有符号误差。
                probe_rows_payload.append(  # 把规范化后的探针行加入列表。
                    {  # 构造探针行字典。
                        "sample_index_in_split": int(probe_row.get("sample_index_in_split", 0)),  # 样本在 split 中的序号。
                        "modality": str(probe_row.get("modality") or ""),  # 模态名称。
                        "seq_len": int(probe_row.get("seq_len", 0)),  # 序列长度。
                        "sample_weight": _round_float(float(probe_row.get("sample_weight", 0.0))),  # 样本权重。
                        "active_keys": [  # 活跃输出头列表。
                            str(key) for key in list(probe_row.get("active_keys") or [])  # 转成字符串列表。
                        ],  # 活跃键列表结束。
                        "prediction_by_head": {  # 按头的预测值。
                            head_key: _round_float(  # 四舍五入。
                                float(prediction_by_head[head_key])  # 转浮点数。
                            ) if head_key in prediction_by_head and prediction_by_head[head_key] is not None else None  # 缺失则为空。
                            for head_key in output_heads  # 遍历所有输出头。
                        },  # 预测值映射结束。
                        "target_by_head": {  # 按头的目标值。
                            head_key: _round_float(  # 四舍五入。
                                float(target_by_head[head_key])  # 转浮点数。
                            ) if head_key in target_by_head and target_by_head[head_key] is not None else None  # 缺失则为空。
                            for head_key in output_heads  # 遍历所有输出头。
                        },  # 目标值映射结束。
                        "signed_error_by_head": {  # 按头的有符号误差。
                            head_key: _round_float(  # 四舍五入。
                                float(signed_error_by_head[head_key])  # 转浮点数。
                            ) if head_key in signed_error_by_head and signed_error_by_head[head_key] is not None else None  # 缺失则为空。
                            for head_key in output_heads  # 遍历所有输出头。
                        },  # 有符号误差映射结束。
                    }  # 探针行字典结束。
                )  # 探针行添加结束。
            split_rows.append(  # 把当前 epoch 的趋势行加入 split 列表。
                {  # 构造 epoch 趋势行。
                    "epoch_index": int(entry["epoch_index"]),  # epoch 编号。
                    "mean_loss": _round_float(float(entry.get("mean_loss", 0.0))),  # 平均损失。
                    "selection_score": _round_float(float(entry["selection_score"])) if entry.get("selection_score") is not None else None,  # 选择分数。
                    "fixed_probe_source": str(entry.get("fixed_probe_source") or ""),  # 探针来源标签。
                    "fixed_probe_sample_count": int(entry.get("fixed_probe_sample_count") or len(probe_rows_payload)),  # 探针样本数。
                    "fixed_probe_rows": probe_rows_payload,  # 探针行列表。
                }  # epoch 趋势行结束。
            )  # 趋势行添加结束。
        report[split] = split_rows  # 把当前 split 的趋势行写入报告。
    return report  # 返回趋势报告。


def build_head_error_trend_report(
    *,
    loss_diagnostics_payload: Mapping[str, Any],
    epoch_predictions_payload: Mapping[str, Any] | None = None,
    source_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """构建完整的输出头误差趋势报告。"""
    output_heads = _resolve_output_heads(epoch_predictions_payload)  # 解析输出头列表。
    snapshot_index = _index_epoch_snapshots(epoch_predictions_payload)  # 构建 epoch 快照索引。
    report: dict[str, Any] = {  # 组装报告头部信息。
        "model_name": str(loss_diagnostics_payload.get("model_name") or ""),  # 模型名称。
        "checkpoint_format": str(loss_diagnostics_payload.get("checkpoint_format") or ""),  # 检查点格式。
        "output_heads": output_heads,  # 输出头列表。
        "head_units": {head_key: _HEAD_UNITS.get(head_key, "unitless") for head_key in output_heads},  # 按头的单位映射。
        "source_paths": dict(source_paths or {}),  # 输入文件路径记录。
        "fixed_probe_trends": _build_fixed_probe_trends(  # 固定探针趋势。
            epoch_predictions_payload,  # epoch 预测数据。
            output_heads=output_heads,  # 输出头列表。
        ),  # 探针趋势构建结束。
    }  # 报告头部结束。
    for split in ("train", "val"):  # 遍历训练集和验证集。
        raw_entries = loss_diagnostics_payload.get(split) or []  # 取出当前 split 的损失诊断条目。
        if not isinstance(raw_entries, list):  # 条目必须是列表。
            raise TypeError(f"loss_diagnostics_payload[{split!r}] must be a list")  # 类型不对就报错。
        report[split] = [  # 逐条构建 epoch 指标。
            _build_epoch_head_metrics(  # 构建单个 epoch 的指标。
                epoch_entry=entry,  # 当前 epoch 条目。
                output_heads=output_heads,  # 输出头列表。
                snapshot_entry=snapshot_index.get((split, int(entry["epoch_index"]))),  # 对应的快照条目。
            )  # epoch 指标构建结束。
            for entry in raw_entries  # 遍历所有条目。
        ]  # epoch 指标列表结束。
    return report  # 返回完整报告。


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)  # 用模块 docstring 作为描述。
    parser.add_argument("--train-report", type=Path, default=None, help="Path to *_train_report.json")  # 训练报告路径。
    parser.add_argument("--loss-diagnostics", type=Path, default=None, help="Path to *_loss_diagnostics.json")  # 损失诊断路径。
    parser.add_argument(  # epoch 预测对照文件路径（可选）。
        "--epoch-predictions",  # 参数名。
        type=Path,  # 参数类型。
        default=None,  # 默认不提供。
        help="Optional path to *_epoch_predictions_vs_targets.json for consistency checks",  # 帮助文本。
    )  # epoch 预测参数定义结束。
    parser.add_argument("--output-json", type=Path, default=None, help="Path to write *_head_error_trends.json")  # 输出路径。
    return parser  # 返回解析器。


def _print_summary(payload: dict[str, Any]) -> int:
    """打印结构化摘要并返回退出码。"""
    from liquidloc.common.io_utils import dumps_json_text
    print(dumps_json_text(payload, indent=None))
    return int(payload["exit_code"])


def _safe_write_report(output_json_path: Path, report: Mapping[str, Any]) -> tuple[bool, str | None, str | None]:
    """尽力写出趋势报告，失败时返回错误类型和详情。"""
    try:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
        output_json_path.write_text(dumps_json_text(report), encoding="utf-8")  # 写出 JSON 文件。
    except OSError as exc:
        return False, type(exc).__name__, str(exc)
    return True, None, None


def main() -> int:
    """脚本主入口，读入诊断文件并写出趋势报告。"""
    args = _build_arg_parser().parse_args()  # 解析命令行参数。

    from liquidloc.common.tee_logger import print_args, print_dict
    print_args(args, "21_analyze_head_training_trends")

    print("[21_head_trends] 开始 | train_report=" + str(args.train_report), flush=True)
    try:
        loss_diagnostics_path, epoch_predictions_path = _resolve_input_paths(  # 解析输入路径。
            train_report_path=args.train_report,  # 训练报告路径。
            loss_diagnostics_path=args.loss_diagnostics,  # 损失诊断路径。
            epoch_predictions_path=args.epoch_predictions,  # epoch 预测路径。
        )  # 路径解析结束。
        loss_diagnostics_payload = _load_json(loss_diagnostics_path)  # 读取损失诊断数据。
        epoch_predictions_payload = _load_json(epoch_predictions_path) if epoch_predictions_path is not None else None  # 读取 epoch 预测数据（可选）。
        print("[21_head_trends] 输入已加载 | 正在构建趋势报告", flush=True)
        report = build_head_error_trend_report(  # 构建趋势报告。
            loss_diagnostics_payload=loss_diagnostics_payload,  # 损失诊断数据。
            epoch_predictions_payload=epoch_predictions_payload,  # epoch 预测数据。
            source_paths={  # 记录输入文件路径。
                "loss_diagnostics": str(loss_diagnostics_path),  # 损失诊断路径。
                "epoch_predictions": str(epoch_predictions_path) if epoch_predictions_path is not None else "",  # epoch 预测路径。
                "train_report": str(args.train_report) if args.train_report is not None else "",  # 训练报告路径。
            },  # 源路径记录结束。
        )  # 报告构建结束。
        output_json_path = args.output_json or _default_output_path(loss_diagnostics_path)  # 决定输出路径。
        print_dict({"loss_diagnostics_path": str(loss_diagnostics_path), "epoch_predictions_path": str(epoch_predictions_path) if epoch_predictions_path else None, "output_json_path": str(output_json_path)}, "路径配置")
        print("[21_head_trends] 正在写入报告 | output=" + str(output_json_path), flush=True)
    except Exception as exc:
        print("[21_head_trends] 完成 | 返回码=1 (exception)", flush=True)
        return _print_summary(
            {
                "status": "failed",
                "stage": "head_training_trends",
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "exit_code": 1,
            }
        )
    write_ok, error_type, error_detail = _safe_write_report(output_json_path, report)  # 尽力写出趋势报告。
    if not write_ok:
        print("[21_head_trends] 完成 | 返回码=1 (write_failed)", flush=True)
        return _print_summary(
            {
                "status": "failed",
                "stage": "head_training_trends",
                "output_json": str(output_json_path),
                "error_type": error_type,
                "detail": error_detail,
                "exit_code": 1,
            }
        )
    print("[21_head_trends] 完成 | 返回码=0", flush=True)
    return _print_summary(
        {
            "status": "ok",
            "stage": "head_training_trends",
            "output_json": str(output_json_path),
            "exit_code": 0,
        }
    )


if __name__ == "__main__":  # 只有直接执行脚本时才走这里。
    raise SystemExit(main())  # 用 main 的返回码结束进程。
