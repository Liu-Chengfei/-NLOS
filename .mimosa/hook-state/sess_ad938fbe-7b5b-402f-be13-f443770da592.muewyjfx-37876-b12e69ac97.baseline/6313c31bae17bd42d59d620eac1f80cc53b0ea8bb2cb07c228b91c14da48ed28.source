"""Diagnostics utilities submodule for Liquid trainer.

Handles batch-level and epoch-level diagnostics collection,
including prediction/target snapshots, probe rows, loss breakdowns, and artifact generation.
"""
from __future__ import annotations

import math
import torch

from typing import Any, Mapping

from ._constants import (  # type: ignore[import]
    _ACTIVE_HEADS_BY_MODALITY,
    _BATCH_DIAGNOSTIC_PROBE_LIMIT,
    _BIAS_HUBER_DELTA,
    _EPOCH_ARTIFACT_BATCH_LIMIT,
    _EPOCH_FIXED_PROBE_LIMIT,
    _OUTPUT_KEYS,
    _RISK_CALIBRATION_BIN_COUNT,
    _SCALING_LOG_EPS,
    _SELECTION_WEIGHTS,
    _TAIL_SELECTION_OBSERVATION_COEFF,
    _TAIL_SELECTION_TAIL_COEFF,
)
from liquidloc.common.types import ModelIntermediate
from liquidloc.common.validation import coerce_finite_scalar

# Direct imports (no circular dependency detected)
from liquidloc.models.liquid._data_utils import (
    _resolve_current_modality,
    _resolve_sample_cache_device,
    _build_head_mask_tensor,
    _move_materialized_samples,
    _build_sample_weight_tensor,
    _materialize_samples,
)
from liquidloc.models.liquid._loss_compute import (
    _masked_component_loss_stats,
    _compute_liquid_auxiliary_loss_terms,
    _normalized_squared_error,
    _predict_batch,
    _semantic_loss_matrix,
    _masked_supervised_loss_stats,
)
from liquidloc.models.liquid._checkpoint_utils import _build_epoch_score_bundle, _build_export_score

def _collect_batch_loss_diagnostics(
    prediction_batch: torch.Tensor,
    target_batch: torch.Tensor,
    *,
    modalities: list[str],
    sample_weights: list[float],
    seq_lens: list[int],
    bias_huber_delta: float = _BIAS_HUBER_DELTA,
    scaling_log_eps: float = _SCALING_LOG_EPS,
) -> dict[str, Any]:
    """收集一个批次的损失诊断信息。

    包括批次大小、损失分母/分子、模态列表、序列长度。
    样本权重、每个样本的濢、活头、头掩码、预测、目标。
    语义损失和平方误差的逐头明细。

    参数：
        prediction_batch: 模型预测批次，形状(B, 4)。
        target_batch: 目标值批次，形状 (B, 4)。
        modalities: 当前批次每个样本的模态名列表。
        sample_weights: 样本权重列表。
        seq_lens: 每个样本的序列长度列表。

    返回值：
        包含所有诊断信息的字典。
    """
    mask = _build_head_mask_tensor(modalities, reference_tensor=prediction_batch)  # 构、输出头掩码。
    sample_weight_tensor = _build_sample_weight_tensor(sample_weights, reference_tensor=prediction_batch).unsqueeze(1)  # 样本权重张量，增加列维。
    semantic_loss = _semantic_loss_matrix(prediction_batch, target_batch, bias_huber_delta=bias_huber_delta, scaling_log_eps=scaling_log_eps)  # 语义损失矩阵。
    squared_error = _normalized_squared_error(prediction_batch, target_batch)  # 平方误差矩阵。
    weighted_mask = mask * sample_weight_tensor  # 加权掩码 = 头掩码× 样本权重。
    per_row_active_keys = [  # 每个样本的激活头列表。
        [
            key
            for key_index, key in enumerate(_OUTPUT_KEYS)
            if float(mask[row_index, key_index].item()) > 0.0  # 掩码 > 0 表示该头活。
        ]
        for row_index in range(int(mask.shape[0]))  # 遍历每个样本。
    ]
    probe_row_count = min(int(prediction_batch.shape[0]), _BATCH_DIAGNOSTIC_PROBE_LIMIT)  # 每个 batch 只保留少量探针行。
    component_loss_sums = {  # 每个输出头的逐样本语义损失和。
        key: float((semantic_loss[:, key_index] * mask[:, key_index]).sum().detach().item())
        for key_index, key in enumerate(_OUTPUT_KEYS)
    }
    component_loss_squared_sums = {  # 每个输出头的逐样本语义损失平方和。
        key: float(((semantic_loss[:, key_index] * mask[:, key_index]) ** 2).sum().detach().item())
        for key_index, key in enumerate(_OUTPUT_KEYS)
    }
    component_loss_counts = {  # 每个输出头的濢、活样本数。
        key: int(mask[:, key_index].sum().detach().item())
        for key_index, key in enumerate(_OUTPUT_KEYS)
    }
    modality_counts: dict[str, int] = {  # D7：与 lstm/trainer.py 同名函数口径对齐：键名走 str() 转换、按 sorted(set(...)) 排序，禁止本地插入序导致下游消费者字典序漂移。
        str(modality_key): int(sum(1 for modality in modalities if modality == modality_key))
        for modality_key in sorted(set(modalities))
    }
    return {
        "batch_size": int(prediction_batch.shape[0]),  # 批次大小。
        "loss_numerator": float((semantic_loss * weighted_mask).sum().detach().item()),  # 加权损失分子。
        "loss_denominator": float(weighted_mask.sum().detach().item()),  # 有效权重分母。
        "mean_loss": float(((semantic_loss * weighted_mask).sum() / weighted_mask.sum()).detach().item()) if weighted_mask.sum().item() > 0 else 0.0,  # 平均损失，防止除零。
        "modalities": list(modalities[:probe_row_count]),  # 探针行模态列表。
        "seq_lens": [int(seq_len) for seq_len in seq_lens[:probe_row_count]],  # 探针行序列长度列表。
        "sample_weights": [float(weight) for weight in sample_weights[:probe_row_count]],  # 探针行样本权重列表。
        "active_keys": per_row_active_keys[:probe_row_count],  # 探针行激活头列表。
        "head_mask": mask[:probe_row_count].detach().cpu().tolist(),  # 探针行头掩码矩阵。
        "prediction": prediction_batch[:probe_row_count].detach().cpu().tolist(),  # 探针行，预览前几行预测值。
        "target": target_batch[:probe_row_count].detach().cpu().tolist(),  # 探针行，预览前几行目标值。
        "semantic_loss_by_head": (semantic_loss[:probe_row_count] * mask[:probe_row_count]).detach().cpu().tolist(),  # 探针行、头语义损失。
        "squared_error_by_head": (squared_error[:probe_row_count] * mask[:probe_row_count]).detach().cpu().tolist(),  # 探针行、头平方误差。
        "probe_row_count": probe_row_count,
        "truncated_row_count": max(0, int(prediction_batch.shape[0]) - probe_row_count),
        "modality_counts": modality_counts,
        "seq_len_stats": {
            "min": min(int(seq_len) for seq_len in seq_lens) if seq_lens else None,  # D7：与 lstm/trainer.py 同名函数口径对齐：空列表默认 None 而非 0，避免下游消费、误判存在零长度序列。
            "max": max(int(seq_len) for seq_len in seq_lens) if seq_lens else None,  # D7：同上，空列表默认None 而非 0。
        },
        "sample_weight_stats": {
            "min": min(float(weight) for weight in sample_weights) if sample_weights else None,  # D7：与 lstm/trainer.py 同名函数口径对齐：空列表默认 None 而非 0.0，避免下游消费、误判存在零权重样本。
            "max": max(float(weight) for weight in sample_weights) if sample_weights else None,  # D7：同上，空列表默认None 而非 0.0。
            "mean": (sum(float(weight) for weight in sample_weights) / float(len(sample_weights)))  # D7/D3：与 lstm/trainer.py 同名函数口径对齐：补mean 字段，下游消费、（analysis/plotting/verify）依赖该字段，缺失会导致 KeyError 或静默漂移。
            if sample_weights
            else None,
        },
        "component_loss_sums": component_loss_sums,
        "component_loss_squared_sums": component_loss_squared_sums,
        "component_loss_counts": component_loss_counts,
    }


def _summarize_component_losses_compact(
    batches: list[dict[str, Any]],
    *,
    include_auxiliary: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float | None]]:
    """基于聚合统计量汇总、头损失，避免保留全量、样地"""
    summary: dict[str, dict[str, float | None]] = {}
    for key in _OUTPUT_KEYS:
        total_sum = 0.0
        total_squared_sum = 0.0
        total_count = 0
        for batch in batches:
            component_sums = batch.get("component_loss_sums", {})
            component_squared_sums = batch.get("component_loss_squared_sums", {})
            component_counts = batch.get("component_loss_counts", {})
            if not (
                isinstance(component_sums, Mapping)
                and isinstance(component_squared_sums, Mapping)
                and isinstance(component_counts, Mapping)
            ):
                continue
            key_count = int(component_counts.get(key, 0) or 0)
            if key_count <= 0:
                continue
            raw_sum = float(component_sums.get(key, 0.0) or 0.0)
            raw_squared_sum = float(component_squared_sums.get(key, 0.0) or 0.0)
            if not (math.isfinite(raw_sum) and math.isfinite(raw_squared_sum)):
                continue
            total_sum += raw_sum
            total_squared_sum += raw_squared_sum
            total_count += key_count
        if total_count > 0:
            mean_value = total_sum / float(total_count)
            variance = max(0.0, (total_squared_sum / float(total_count)) - (mean_value * mean_value))
            summary[key] = {"mean": mean_value, "variance": variance}
        else:
            summary[key] = {"mean": None, "variance": None}
    if include_auxiliary is not None:
        for key, value in include_auxiliary.items():
            if key in _OUTPUT_KEYS:
                raise ValueError(f"auxiliary key collides with output head: {key!r}")
            summary[key] = {"mean": float(value), "variance": 0.0}
    return summary


def _summarize_component_losses(
    batches: list[dict[str, Any]],
    *,
    include_auxiliary: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float | None]]:
    """汇、多个批次的逐头损失统计。

    对每个输出头，收集所有批次中的、样本损失、，
    计算均、和方差。可选地追加辅助损失项的统计。

    参数：
        batches: 批次诊断信息列表，每个元素是 _collect_batch_loss_diagnostics 的返回。
        include_auxiliary: 可、的辅助损失项映射，键为损失名，值为标量。

    返回值：
        字典，键为输出头名或辅助损失名，值为 {"mean": float, "variance": float}。
    """
    summary: dict[str, dict[str, float | None]] = {}  # 汇结果字典。
    for key in _OUTPUT_KEYS:  # 遍历每个输出头。
        total_sum = 0.0  # 该头所有批次的损失总和。
        total_squared_sum = 0.0  # 该头所有批次的损失平方总和。
        total_count = 0  # 该头所有批次的样本计数。
        for batch in batches:  # 遍历每个批次。
            component_sums = batch.get("component_loss_sums", {})  # 取该批次的、头损失总和。
            component_squared_sums = batch.get("component_loss_squared_sums", {})  # 取该批次的、头损失平方总和。
            component_counts = batch.get("component_loss_counts", {})  # 取该批次的、头样本计数。
            if not (  # 三字段必须同为映射，任一非映射即跳过整个批次，避免部分字段损坏导致计数与求和口径不一致。
                isinstance(component_sums, Mapping)
                and isinstance(component_squared_sums, Mapping)
                and isinstance(component_counts, Mapping)
            ):
                continue
            key_count = int(component_counts.get(key, 0) or 0)  # 该头在该批次的样本计数。
            if key_count <= 0: # 计数非正时跳过，避免 count=0 ?sum。 的损坏数据稀释均值。
                continue
            raw_sum = float(component_sums.get(key, 0.0) or 0.0)  # 该头在该批次的损失、和。
            raw_squared_sum = float(component_squared_sums.get(key, 0.0) or 0.0)  # 该头在该批次的损失平方、和。
            if not (math.isfinite(raw_sum) and math.isfinite(raw_squared_sum)):  # NaN/Inf 守卫，避免静默穿透污染统计。
                continue
            total_sum += raw_sum
            total_squared_sum += raw_squared_sum
            total_count += key_count
        if total_count > 0:  # 有、时计算统计量。
            mean_value = total_sum / float(total_count)  # 鍧囧€笺€。
            variance = max(0.0, (total_squared_sum / float(total_count)) - (mean_value ** 2))  # 方差。
            summary[key] = {"mean": mean_value, "variance": variance}  # 记录统计量。
        else:  # 没有值时标记None。
            summary[key] = {"mean": None, "variance": None}  # 无数据。
    if include_auxiliary is not None:  # 有辅助损失项时追加。
        for key, value in include_auxiliary.items():  # 遍历辅助损失项。
            if key in _OUTPUT_KEYS:  # 辅助键不得与输出头同名，避免静默覆盖输出头统计。
                raise ValueError(f"auxiliary key collides with output head: {key!r}")
            summary[key] = {"mean": float(value), "variance": 0.0}  # 辅助项方差为 0。
    return summary  # 返回汇结果。


def _collect_epoch_loss_diagnostics(
    *,
    split: str,
    epoch_index: int,
    batches: list[dict[str, Any]],
    selection_score: float | None,
    supervised_loss: float | None = None,
    auxiliary_summary: Mapping[str, float] | None = None,
    phase_switch_shock: Mapping[str, float | None] | None = None,
) -> dict[str, Any]:
    """汇、一epoch 的损失诊断信息。

    将所有批次的损失分子和分母累加，计算 epoch 级平均损失，
    并附带头损失汇和辅助损失项。

    参数：
    split: 数据集划分名train" ?"val"）。
        epoch_index: 当前 epoch 索引。
        batches: 批次诊断信息列表。
        selection_score: 选择分数，None 表示不记录。
        supervised_loss: 监督损失，None 表示不单独记录。        auxiliary_summary: 辅助损失项汇总，None 表示不记录。        phase_switch_shock: 相位切换冲击摘要，None 表示不记录。
    返回值：
        包含 split。乪poch_index。乵ean_loss。乴oss_numerator。乴oss_denominator。。
        batch_count。乥atches。乻election_score。乤uxiliary_losses 和。
        component_loss_summary 的字典。
    """
    loss_numerator = math.fsum(float(batch["loss_numerator"]) for batch in batches)  # 累加所有批次的损失分子，用 fsum 提高精度。
    loss_denominator = math.fsum(float(batch["loss_denominator"]) for batch in batches)  # 累加所有批次的损失分母，用 fsum 提高精度。
    stored_batches = list(batches[:_EPOCH_ARTIFACT_BATCH_LIMIT])
    epoch_payload = {  # 构建 epoch 级载荷。
        "split": split,  # 数据集划分。
        "epoch_index": int(epoch_index),  # epoch 绱㈠紩。。
        "mean_loss": loss_numerator / loss_denominator if loss_denominator > 0.0 else 0.0,  # epoch 平均损失。
        "loss_numerator": loss_numerator,  # 总损失分子。
        "loss_denominator": loss_denominator,  # 总损失分母。
        "batch_count": len(batches),  # 批次数量。
        "stored_batch_count": len(stored_batches),  # artifact 中实际保留的批次数量。
        "truncated_batch_count": max(0, len(batches) - len(stored_batches)),  # 被裁剪的批次数量。
        "batches": stored_batches,  # 批次诊断列表（只保留前若干批次明细）。
    }
    if selection_score is not None:
        epoch_payload["selection_score"] = coerce_finite_scalar(selection_score, name="selection_score")
    if supervised_loss is not None:
        epoch_payload["supervised_loss"] = coerce_finite_scalar(supervised_loss, name="supervised_loss")
    if auxiliary_summary is not None:
        epoch_payload["auxiliary_losses"] = {
            str(key): coerce_finite_scalar(value, name=f"auxiliary_losses.{key}")
            for key, value in auxiliary_summary.items()
        }
    if phase_switch_shock is not None:
        epoch_payload["phase_switch_shock"] = {
            str(key): (
                None
                if value is None
                else coerce_finite_scalar(value, name=f"phase_switch_shock.{key}")
            )
            for key, value in phase_switch_shock.items()
        }
    epoch_payload["component_loss_summary"] = _summarize_component_losses_compact(
        batches,  # 逐头损失汇总基于聚合统计量，避免保存全量逐样本值。
        include_auxiliary=auxiliary_summary,
    )
    return epoch_payload  # 返回 epoch 级诊断载荷。


def _collect_epoch_prediction_target_snapshot(
    *,
    model: Any,
    split: str,
    epoch_index: int,
    cached_windows: list[tuple[dict[str, Any], torch.Tensor, int, float]],
    batch_diagnostics: list[dict[str, Any]],
    prediction_target_summary: Mapping[str, Any],
    mean_loss: float,
    selection_score: float | None,
    supervised_loss: float | None = None,
    component_losses: Mapping[str, float] | None = None,
    auxiliary_losses: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """收集丢epoch 的预目标快照，用于训练日志。

    包括每个输出头的濢、活样本数、预测均值、目标均值，
    第一个批次的逐样本明细，以及固定探针样本的预测结果。

    参数：
        model: 液体模型实例。
        split: 数据集划分名train" ?"val"）。
        epoch_index: 当前 epoch 索引。
        cached_windows: 材料化窗口列表。
        batch_diagnostics: 批次诊断信息列表。
        mean_loss: epoch 平均损失。
        selection_score: 选择分数，None 表示不记录。        supervised_loss: 监督损失，None 表示不记录。        component_losses: 逐头损失映射，None 表示不记录。        auxiliary_losses: 辅助损失映射，None 表示不记录。
    返回值：
        包含 split。乪poch_index。乵ean_loss。乻election_score。乻ample_count。。
        各头统计量、探针批次明细和固定探针行的字典。
    """
    sample_count = int(prediction_target_summary.get("sample_count", 0) or 0)
    active_sample_count_by_head = {
        key: int((prediction_target_summary.get("active_sample_count_by_head", {}) or {}).get(key, 0) or 0)
        for key in _OUTPUT_KEYS
    }
    prediction_mean_active_samples = {
        key: (prediction_target_summary.get("prediction_mean_active_samples", {}) or {}).get(key)
        for key in _OUTPUT_KEYS
    }
    target_mean_active_samples = {
        key: (prediction_target_summary.get("target_mean_active_samples", {}) or {}).get(key)
        for key in _OUTPUT_KEYS
    }
    prediction_mean_all_samples = {
        key: (prediction_target_summary.get("prediction_mean_all_samples", {}) or {}).get(key)
        for key in _OUTPUT_KEYS
    }
    target_mean_all_samples = {
        key: (prediction_target_summary.get("target_mean_all_samples", {}) or {}).get(key)
        for key in _OUTPUT_KEYS
    }
    probe_batch = batch_diagnostics[0] if batch_diagnostics else {
        "modalities": [],
        "active_keys": [],
        "prediction": [],
        "target": [],
    }
    probe_batch_rows = []
    for row_index, (modality, active_keys, prediction_row, target_row) in enumerate(
        zip(
            probe_batch["modalities"],
            probe_batch["active_keys"],
            probe_batch["prediction"],
            probe_batch["target"],
            strict=True,
        )
    ):
        probe_batch_rows.append(
            {
                "row_index": row_index,
                "modality": modality,
                "active_keys": list(active_keys),
                # D5/D7：与 lstm/trainer.py 同名函数口径对齐，统丢coerce_finite_scalar 中心入口。
                # 拒绝 NaN/Inf 静默穿、至 snapshot（裸 float() 会把 nan/inf 静默写入消费者日志）。
                "prediction_by_head": {
                    key: coerce_finite_scalar(prediction_row[key_index], name=f"probe_batch.prediction.{key}")
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
                "target_by_head": {
                    key: coerce_finite_scalar(target_row[key_index], name=f"probe_batch.target.{key}")
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
                "signed_error_by_head": {
                    key: coerce_finite_scalar(
                        prediction_row[key_index] - target_row[key_index],
                        name=f"probe_batch.signed_error.{key}",
                    )
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
            }
        )
    fixed_probe_count = min(len(cached_windows), _EPOCH_FIXED_PROBE_LIMIT)
    fixed_probe_rows = _collect_fixed_probe_rows(
        model=model,
        cached_windows=cached_windows[:fixed_probe_count],
    )
    return {
        "split": split,
        "epoch_index": int(epoch_index),
        # D5/D7：epoch 级标量走 coerce_finite_scalar 中心入口，拒NaN/Inf 静默穿、至 snapshot。
        # ?_summarize_run ?train_loss/val_loss/selection_score 口径致（D7）。
        "mean_loss": coerce_finite_scalar(mean_loss, name="mean_loss"),
        "selection_score": None if selection_score is None else coerce_finite_scalar(selection_score, name="selection_score"),
        "supervised_loss": None if supervised_loss is None else coerce_finite_scalar(supervised_loss, name="supervised_loss"),
        "component_losses": {
            str(key): coerce_finite_scalar(value, name=f"component_losses.{key}")
            for key, value in (component_losses or {}).items()
        },
        "auxiliary_losses": {
            str(key): coerce_finite_scalar(value, name=f"auxiliary_losses.{key}")
            for key, value in (auxiliary_losses or {}).items()
        },
        "sample_count": sample_count,
        "active_sample_count_by_head": active_sample_count_by_head,
        "prediction_mean": dict(prediction_mean_active_samples),
        "target_mean": dict(target_mean_active_samples),
        "prediction_mean_active_samples": dict(prediction_mean_active_samples),
        "target_mean_active_samples": dict(target_mean_active_samples),
        "prediction_mean_all_samples": dict(prediction_mean_all_samples),
        "target_mean_all_samples": dict(target_mean_all_samples),
        "mean_scope": "fixed_split_active_sample_mean",
        "mean_scope_note": "target means can remain constant across epochs because the split is fixed and means are computed over active samples.",
        "probe_batch_modalities": list(probe_batch["modalities"]),
        "probe_batch_active_keys": [list(keys) for keys in probe_batch["active_keys"]],
        "probe_batch_prediction": probe_batch["prediction"],
        "probe_batch_target": probe_batch["target"],
        "probe_batch_rows": probe_batch_rows,
        "fixed_probe_source": "cached_split_prefix",
        "fixed_probe_sample_count": fixed_probe_count,
        "fixed_probe_rows": fixed_probe_rows,
    }


def _collect_fixed_probe_rows(
    *,
    model: Any,
    cached_windows: list[tuple[dict[str, Any], torch.Tensor, int, float]],
) -> list[dict[str, Any]]:
    """对固定探针样本做前向预测，收集、样本的预测-目标明细。

    固定探针用于epoch 追踪同一组样本的预测变化趋势。

    参数：
        model: 液体模型实例。
        cached_windows: 固定探针的材料化窗口列表。

    返回值：
        逐样本明细列表，每个元素包含 sample_index_in_split、modality。
        seq_len。乻ample_weight。乤ctive_keys。乸rediction_by_head。。
        target_by_head 和signed_error_by_head。。
    """
    if not cached_windows:  # 没有探针样本时返回空列表。
        return []  # 空列表。
    # D7/D4：与 lstm/trainer.py 同名函数口径对齐—探针前向在 no_grad 下执行，
    # 避免构、梯度图浪费显存（AGENTS.md：high-level consumer 不占 GPU）。
    with torch.no_grad():
        prediction_batch = _predict_batch(model, [sample[0] for sample in cached_windows])  # 对探针样本做前向预测。
    fixed_probe_rows: list[dict[str, Any]] = []  # 存、样本明细。
    for sample_index, sample in enumerate(cached_windows):  # 遍历每个探针样本。
        # D5/D7：与 lstm/trainer.py 同名函数口径对齐，统丢coerce_finite_scalar 中心入口，拒NaN/Inf 静默穿、（target 来自缓存窗口，未在上游做有限性校验）。
        prediction_row = [
            coerce_finite_scalar(value, name=f"fixed_probe_prediction[{sample_index}][{key_index}]")
            for key_index, value in enumerate(prediction_batch[sample_index].detach().cpu().tolist())
        ]
        target_row = [
            coerce_finite_scalar(value, name=f"fixed_probe_target[{sample_index}][{key_index}]")
            for key_index, value in enumerate(sample[1].detach().cpu().tolist())
        ]
        modality = _resolve_current_modality(sample[0])
        active_keys = list(_ACTIVE_HEADS_BY_MODALITY[modality])
        fixed_probe_rows.append(
            {
                "sample_index_in_split": sample_index,
                "modality": modality,
                "seq_len": int(sample[2]),
                "sample_weight": coerce_finite_scalar(sample[3], name=f"fixed_probe_sample_weight[{sample_index}]"),
                "active_keys": active_keys,
                "prediction_by_head": {
                    key: prediction_row[key_index]
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
                "target_by_head": {
                    key: target_row[key_index]
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
                "signed_error_by_head": {
                    key: prediction_row[key_index] - target_row[key_index]
                    for key_index, key in enumerate(_OUTPUT_KEYS)
                },
            }
        )
    return fixed_probe_rows


def _evaluate_model(
    model: Any,
    val_windows: list[Any],
    *,
    epoch_index: int | None = None,
    split: str = "val",
    collect_diagnostics: bool = False,
) -> float | tuple[float, list[dict[str, Any]], dict[str, Any]]:
    """评估模型在验训练集上的损失。

    ?no_grad 模式下、批次做前向预测，计算监督损失和辅助损失。
    得到选择损失（selection_loss = supervised_loss + auxiliary_loss。
    和、择分数。可选择收集详细的批次和 epoch 级诊断信息。

    参数：
        model: 液体模型实例。
        val_windows: 验证/训练窗口列表。
        epoch_index: 当前 epoch 索引，仅用于诊断日志。
        split: 数据集划分名val" ?"train"）。
        collect_diagnostics: 是否收集详细诊断信息。

    返回值：
        - collect_diagnostics=False: 返回 selection_loss 浮点数。
        - collect_diagnostics=True: 返回三元。
          (selection_loss, batch_diagnostics, epoch_snapshot)。。
    """
    model.eval()
    cache_attr = "_cached_train_windows" if split == "train" else "_cached_val_windows"
    cached_windows = getattr(model, cache_attr, None)
    if cached_windows is None:
        cache_device = _resolve_sample_cache_device(model)  # §13.8: 与 model 同设备，释放 CPU 内存
        cached_windows = _move_materialized_samples(
            _materialize_samples(
                val_windows,
                tail_selection_observation_coeff=coerce_finite_scalar(
                    getattr(model, "_tail_selection_observation_coeff", _TAIL_SELECTION_OBSERVATION_COEFF),
                    name="tail_selection_observation_coeff",
                    min_value=0.0,
                ),
            ),
            cache_device,
        )
        setattr(model, cache_attr, cached_windows)
    batch_attr = "_train_batch_size" if split == "train" else "_eval_batch_size"
    eval_batch_size = int(getattr(model, batch_attr, 0) or len(cached_windows))
    component_numerator_sums = {key: 0.0 for key in _OUTPUT_KEYS}
    component_denominator_sums = {key: 0.0 for key in _OUTPUT_KEYS}
    loss_numerator_sum = 0.0
    loss_denominator_sum = 0.0
    batch_diagnostics: list[dict[str, Any]] = []
    prediction_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    all_modalities: list[str] = []
    all_sample_weights: list[float] = []
    sample_count = 0
    active_sample_count_by_head = {key: 0 for key in _OUTPUT_KEYS}
    prediction_sum_all_samples = {key: 0.0 for key in _OUTPUT_KEYS}
    target_sum_all_samples = {key: 0.0 for key in _OUTPUT_KEYS}
    prediction_sum_active_samples = {key: 0.0 for key in _OUTPUT_KEYS}
    target_sum_active_samples = {key: 0.0 for key in _OUTPUT_KEYS}
    position_mse_scalar_sum = 0.0  # §24.1 执行点：位置误差 MSE 累加器（用于选模决策）。
    position_mse_scalar_count = 0    # §24.1 执行点：位置误差 MSE 样本计数。
    with torch.no_grad():
        # D5/D7：与 lstm/trainer.py._evaluate_model 口径对齐—LSTM 侧、过参数传入已在
        # LSTMTrainer.__init__ ?coerce_finite_scalar 校验bias_huber_delta/scaling_log_eps。
        # Liquid 侧从 model 实例属、读取，故在此统丢coerce_finite_scalar 中心入口。
        # 拒绝 NaN/Inf/bool/负、静默穿透到 eval 损失（与 _compute_batch_loss L2030-2031 同口径）。
        _bias_huber_delta_val = coerce_finite_scalar(
            getattr(model, "_bias_huber_delta", _BIAS_HUBER_DELTA),
            name="model._bias_huber_delta", min_value=0.0)
        _scaling_log_eps_val = coerce_finite_scalar(
            getattr(model, "_scaling_log_eps", _SCALING_LOG_EPS),
            name="model._scaling_log_eps", min_value=0.0)
        for start_index in range(0, len(cached_windows), eval_batch_size):
            end_index = min(len(cached_windows), start_index + eval_batch_size)
            batch_samples = cached_windows[start_index:end_index]
            metadata = [sample[0] for sample in batch_samples]
            prediction_batch = _predict_batch(model, metadata)
            target_batch = torch.stack([sample[1] for sample in batch_samples], dim=0)
            target_batch = target_batch.to(device=prediction_batch.device, dtype=prediction_batch.dtype)
            modalities = [_resolve_current_modality(sample[0]) for sample in batch_samples]
            sample_weights = [sample[3] for sample in batch_samples]
            numerators, denominators, active_keys, position_mse_scalar = _masked_component_loss_stats(
                prediction_batch,
                target_batch,
                modalities=modalities,
                sample_weights=sample_weights,
                bias_huber_delta=_bias_huber_delta_val,
                scaling_log_eps=_scaling_log_eps_val,
            )
            for key in active_keys:
                component_numerator_sums[key] += float(numerators[key].detach().item())
                component_denominator_sums[key] += float(denominators[key].detach().item())
            loss_numerator, loss_denominator = _masked_supervised_loss_stats(
                prediction_batch,
                target_batch,
                modalities=modalities,
                sample_weights=sample_weights,
                bias_huber_delta=_bias_huber_delta_val,
                scaling_log_eps=_scaling_log_eps_val,
            )
            loss_numerator_sum += float(loss_numerator.detach().item())
            loss_denominator_sum += float(loss_denominator.detach().item())
            prediction_batches.append(prediction_batch)
            target_batches.append(target_batch)
            all_modalities.extend(modalities)
            all_sample_weights.extend(float(weight) for weight in sample_weights)
            sample_count += int(prediction_batch.shape[0])
            # §24.1 执行点：累加位置误差 MSE 执行点值（用于最终选模决策与审计追踪）。
            position_mse_scalar_sum += float(position_mse_scalar.detach().item())
            position_mse_scalar_count += 1
            for row_index, modality in enumerate(modalities):
                active_keys = set(_ACTIVE_HEADS_BY_MODALITY[modality])
                for key_index, key in enumerate(_OUTPUT_KEYS):
                    prediction_value = float(prediction_batch[row_index, key_index].detach().item())
                    target_value = float(target_batch[row_index, key_index].detach().item())
                    prediction_sum_all_samples[key] += prediction_value
                    target_sum_all_samples[key] += target_value
                    if key in active_keys:
                        active_sample_count_by_head[key] += 1
                        prediction_sum_active_samples[key] += prediction_value
                        target_sum_active_samples[key] += target_value
            if collect_diagnostics:
                batch_payload = _collect_batch_loss_diagnostics(
                    prediction_batch,
                    target_batch,
                    modalities=modalities,
                    sample_weights=sample_weights,
                    seq_lens=[sample[2] for sample in batch_samples],
                    bias_huber_delta=_bias_huber_delta_val,
                    scaling_log_eps=_scaling_log_eps_val,
                )
                batch_diagnostics.append(batch_payload)
    active_keys = [key for key in _OUTPUT_KEYS if component_denominator_sums[key] > 0.0]
    component_losses = {
        key: torch.tensor(component_numerator_sums[key] / component_denominator_sums[key], dtype=torch.float32)
        for key in active_keys
    }
    supervised_loss = loss_numerator_sum / loss_denominator_sum if loss_denominator_sum > 0.0 else 0.0
    auxiliary_losses = {"calibration": 0.0, "mono": 0.0, "reg_l2": 0.0, "gate_l1": 0.0, "smooth": 0.0, "total": 0.0}
    if prediction_batches:
        full_prediction_batch = torch.cat(prediction_batches, dim=0)
        full_target_batch = torch.cat(target_batches, dim=0)
        auxiliary_terms = _compute_liquid_auxiliary_loss_terms(
            model,
            full_prediction_batch,
            full_target_batch,
            modalities=all_modalities,
            sample_weights=all_sample_weights,
        )
        auxiliary_losses = {
            key: float(auxiliary_terms[key].detach().item())
            for key in ("calibration", "mono", "reg_l2", "gate_l1", "smooth", "total")
        }
    auxiliary_loss = auxiliary_losses["total"]
    # §24.1 执行点：计算平均位置误差 MSE 并传入选模函数（保证终序仅 raw 位置 RMSE 精神执行）。
    avg_position_mse_scalar = torch.tensor(
        position_mse_scalar_sum / position_mse_scalar_count if position_mse_scalar_count > 0 else 0.0,
        dtype=torch.float32,
    )
    export_score = _build_export_score(
        component_losses=component_losses,
        active_keys=active_keys,
        supervised_loss=float(supervised_loss),
        position_mse_scalar=avg_position_mse_scalar,  # §24.1 执行点：位置误差强制纳入选模。
    )
    score_bundle = _build_epoch_score_bundle(
        supervised_loss=float(supervised_loss),
        auxiliary_loss=float(auxiliary_loss),
        export_score=float(export_score),
    )
    optimization_loss = float(score_bundle["optimization_loss"])
    phase_control_score = float(score_bundle["phase_control_score"])
    export_score = float(score_bundle["export_score"])
    selection_score = float(score_bundle["selection_score"])
    model._optimization_loss = optimization_loss
    model._phase_control_score = phase_control_score
    model._export_score = export_score
    model._selection_score = selection_score
    model._selection_loss = optimization_loss
    component_losses_payload = {
        str(key): float(value.detach().item())
        for key, value in component_losses.items()
    }
    model._component_losses = dict(component_losses_payload)
    model._supervised_loss = float(supervised_loss)
    model._auxiliary_losses = dict(auxiliary_losses)
    if not collect_diagnostics:
        return optimization_loss
    for batch_payload in batch_diagnostics:
        batch_payload["auxiliary_losses"] = dict(auxiliary_losses)
    prediction_target_summary = {
        "sample_count": sample_count,
        "active_sample_count_by_head": dict(active_sample_count_by_head),
        "prediction_mean_active_samples": {
            key: (
                prediction_sum_active_samples[key] / float(active_sample_count_by_head[key])
                if int(active_sample_count_by_head[key]) > 0
                else None
            )
            for key in _OUTPUT_KEYS
        },
        "target_mean_active_samples": {
            key: (
                target_sum_active_samples[key] / float(active_sample_count_by_head[key])
                if int(active_sample_count_by_head[key]) > 0
                else None
            )
            for key in _OUTPUT_KEYS
        },
        "prediction_mean_all_samples": {
            key: (prediction_sum_all_samples[key] / float(sample_count)) if sample_count > 0 else None
            for key in _OUTPUT_KEYS
        },
        "target_mean_all_samples": {
            key: (target_sum_all_samples[key] / float(sample_count)) if sample_count > 0 else None
            for key in _OUTPUT_KEYS
        },
    }
    epoch_payload = _collect_epoch_prediction_target_snapshot(
        model=model,
        split=split,
        epoch_index=int(epoch_index or 0),
        cached_windows=cached_windows,
        batch_diagnostics=batch_diagnostics,
        prediction_target_summary=prediction_target_summary,
        mean_loss=optimization_loss,
        selection_score=selection_score,
        supervised_loss=supervised_loss,
        component_losses=component_losses_payload,
        auxiliary_losses=auxiliary_losses,
    )
    epoch_payload["optimization_loss"] = optimization_loss
    epoch_payload["phase_control_score"] = phase_control_score
    epoch_payload["export_score"] = export_score
    epoch_payload["component_losses"] = dict(component_losses_payload)
    epoch_payload["auxiliary_losses"] = dict(auxiliary_losses)
    return optimization_loss, batch_diagnostics, epoch_payload


