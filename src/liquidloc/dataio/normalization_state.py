"""异步高NLOS实验全流程保障手册 P11 训练归一化状态追踪模块。

P11 硬约束（手册 line 168-180）：
- 模型输入（测距/IMU/VIO/时间戳）的归一化统计量（均值/方差/min-max 等）**只用训练集计算**，
  测试集一律用训练集统计量变换；禁止用全量或测试集统计量（滑动窗口归一化时窗口内
  不得含未来/测试帧信息）——时序数据归一化泄漏是评估虚高的常见来源。
- 归一化状态写入 ``im_meta.json`` 落盘，便于跨机/跨 seed 复现。

模块入口:
    compute_normalization_state(train_events) → NormalizationState
    write_im_meta_json(state, output_root) → 写出 im_meta.json
    load_im_meta_json(output_root) → 加载复现归一化状态
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from liquidloc.common.validation import coerce_finite_scalar


# 归一化统计字段定义（与 S6 数据契约对应）
NORMALIZATION_FIELDS: tuple[tuple[str, str], ...] = (
    # (modality, field_name)
    ("uwb", "measured_range"),
    ("imu", "ax"),
    ("imu", "ay"),
    ("imu", "gz"),
    ("vio", "dx"),
    ("vio", "dy"),
    ("vio", "dyaw"),
)


@dataclass
class FieldStatistics:
    """单字段归一化统计（mean/std/min/max/count/non_finite）。"""

    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    non_finite_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FieldStatistics:
        return cls(
            count=int(payload.get("count", 0)),
            mean=float(payload.get("mean", 0.0)),
            std=float(payload.get("std", 0.0)),
            min=float(payload.get("min", 0.0)),
            max=float(payload.get("max", 0.0)),
            non_finite_count=int(payload.get("non_finite_count", 0)),
        )


@dataclass
class NormalizationState:
    """P11 训练集归一化统计状态（落盘到 im_meta.json）。"""

    n_train_events: int = 0
    n_train_splits: int = 0
    by_modality_field: dict[str, FieldStatistics] = field(default_factory=dict)
    train_split_ids: list[str] = field(default_factory=list)
    dataset_name: str = ""
    computed_at_unix: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "p11_train_only_normalization",
            "n_train_events": self.n_train_events,
            "n_train_splits": self.n_train_splits,
            "train_split_ids": list(self.train_split_ids),
            "dataset_name": self.dataset_name,
            "computed_at_unix": self.computed_at_unix,
            "by_modality_field": {
                key: stats.to_dict() for key, stats in self.by_modality_field.items()
            },
            "notes": (
                "P11 硬约束：均值/方差仅由 train split 计算，"
                "test/val 必须用同一组 stats 变换，禁止泄漏"
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NormalizationState:
        by_field = {}
        for key, stats_payload in (payload.get("by_modality_field") or {}).items():
            by_field[key] = FieldStatistics.from_dict(stats_payload)
        return cls(
            n_train_events=int(payload.get("n_train_events", 0)),
            n_train_splits=int(payload.get("n_train_splits", 0)),
            train_split_ids=list(payload.get("train_split_ids") or []),
            dataset_name=str(payload.get("dataset_name", "")),
            computed_at_unix=float(payload.get("computed_at_unix", 0.0)),
            by_modality_field=by_field,
        )


def _extract_payload_value(event: Mapping[str, Any], modality: str, field_name: str) -> float | None:
    """从事件 payload 中取出指定字段的数值（失败返回 None 不抛错）。"""
    if not isinstance(event, Mapping):
        return None
    event_modality = event.get("modality")
    if event_modality != modality:
        return None
    payload = event.get("payload") or event.get(f"{modality}_payload")
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(field_name)
    if value is None:
        return None
    try:
        return coerce_finite_scalar(value, name=f"{modality}.{field_name}")
    except (TypeError, ValueError):
        return None


def compute_normalization_state(
    train_events: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str = "",
) -> NormalizationState:
    """从训练事件序列计算归一化统计（P11 硬约束：仅用训练集）。

    参数:
        train_events: 训练事件序列（不可含 val/test 帧，避免泄漏）。
        dataset_name: 数据集名（写入 im_meta.json 用于审计）。

    返回:
        NormalizationState：每个 (modality, field) 的 mean/std/min/max/count/non_finite_count。
    """
    import time

    per_field_values: dict[str, list[float]] = {f"{m}.{f}": [] for m, f in NORMALIZATION_FIELDS}
    per_field_non_finite: dict[str, int] = {key: 0 for key in per_field_values}
    n_events = 0
    train_split_ids: set[str] = set()
    for event in train_events:
        if not isinstance(event, Mapping):
            continue
        n_events += 1
        split_id = event.get("split_id")
        if isinstance(split_id, str):
            train_split_ids.add(split_id)
        for key in per_field_values:
            modality, field_name = key.split(".", 1)
            value = _extract_payload_value(event, modality, field_name)
            if value is None:
                per_field_non_finite[key] += 1
            else:
                per_field_values[key].append(value)
    by_modality_field: dict[str, FieldStatistics] = {}
    for key, values in per_field_values.items():
        n = len(values)
        if n == 0:
            by_modality_field[key] = FieldStatistics(
                count=0, mean=0.0, std=0.0, min=0.0, max=0.0,
                non_finite_count=per_field_non_finite[key],
            )
            continue
        mean = statistics.fmean(values)
        if n >= 2:
            std = statistics.stdev(values)
        else:
            std = 0.0
        by_modality_field[key] = FieldStatistics(
            count=n,
            mean=mean,
            std=std,
            min=min(values),
            max=max(values),
            non_finite_count=per_field_non_finite[key],
        )
    return NormalizationState(
        n_train_events=n_events,
        n_train_splits=len(train_split_ids),
        train_split_ids=sorted(train_split_ids),
        dataset_name=dataset_name,
        computed_at_unix=time.time(),
        by_modality_field=by_modality_field,
    )


def write_im_meta_json(state: NormalizationState, output_root: str | Path) -> Path:
    """把归一化状态写到 ``output_root/im_meta.json``（P11 数据契约）。"""
    out_root = Path(output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "im_meta.json"
    meta_path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta_path


def load_im_meta_json(im_meta_path: str | Path) -> NormalizationState:
    """从 im_meta.json 加载归一化状态。"""
    payload = json.loads(Path(im_meta_path).read_text(encoding="utf-8"))
    return NormalizationState.from_dict(payload)


def validate_no_test_leak(
    train_events: Sequence[Mapping[str, Any]],
    *,
    allowed_split_ids: set[str] | None = None,
) -> tuple[int, int]:
    """P11 反泄漏审计：返回 (n_train_events, n_suspect_events)。

    n_suspect_events：事件携带的 split_id 不在 allowed_split_ids 内的数量。
    若 n_suspect > 0 → 调用方应 raise，禁止进入 normalize 计算。
    """
    if allowed_split_ids is None:
        allowed_split_ids = {"train"}
    n_train = 0
    n_suspect = 0
    for event in train_events:
        if not isinstance(event, Mapping):
            continue
        split_id = event.get("split_id") or "train"
        if split_id in allowed_split_ids:
            n_train += 1
        else:
            n_suspect += 1
    return n_train, n_suspect


def compute_z_score_normalization(
    state: NormalizationState,
    *,
    epsilon: float = 1e-6,
) -> dict[str, tuple[float, float]]:
    """从 NormalizationState 导出 z-score 归一化参数 (mean, std)。

    用于测试/val 阶段：用训练集 mean/std 把新数据变换到 N(0, 1)。
    返回: {key: (mean, std)}；std=0 时返回 (mean, max(std, epsilon))。
    """
    out: dict[str, tuple[float, float]] = {}
    for key, stats in state.by_modality_field.items():
        std = max(float(stats.std), epsilon)
        out[key] = (float(stats.mean), std)
    return out


__all__ = [
    "NORMALIZATION_FIELDS",
    "FieldStatistics",
    "NormalizationState",
    "compute_normalization_state",
    "write_im_meta_json",
    "load_im_meta_json",
    "validate_no_test_leak",
    "compute_z_score_normalization",
]