"""论文推理入口共享的 checkpoint 解析。

该模块只构造传给 ``CorePipeline`` 的 ``model_cfgs`` 内存覆盖，绝不修改
``configs/models`` 下的共享 YAML。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path


DEFAULT_CHECKPOINT_FILES: Mapping[str, str] = {
    "lstm_ekf": "lstm_ekf_best_checkpoint.pt",
    "liquid_ekf": "liquid_ekf_resume_checkpoint.pt",
    "transformer_ekf": "transformer_ekf_best_checkpoint.pt",
}


def _checkpoint_candidates(checkpoint_dir: Path, method: str, preferred_name: str) -> tuple[Path, ...]:
    """按稳定优先级返回目录内可能的 checkpoint。"""
    ordered: list[Path] = [checkpoint_dir / preferred_name]
    for pattern in (
        f"{method}_best_checkpoint.pt",
        f"{method}_resume_checkpoint.pt",
        f"{method}_epoch_*.pt",
        "*.pt",
    ):
        ordered.extend(sorted(checkpoint_dir.glob(pattern), reverse=True))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in ordered:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return tuple(unique)


def resolve_checkpoint_overrides(
    methods: Iterable[str],
    train_root: str | Path,
    *,
    seed: int = 0,
    preferred_files: Mapping[str, str] = DEFAULT_CHECKPOINT_FILES,
) -> dict[str, dict[str, str]]:
    """解析神经方法 checkpoint，返回 ``CorePipeline.model_cfgs`` 覆盖。

    经典方法不需要 checkpoint，会被忽略。请求的神经方法若没有任何可用
    checkpoint，则立即报错，避免静默使用未训练模型生成论文结果。
    """
    root = Path(train_root)
    overrides: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for method in methods:
        preferred_name = preferred_files.get(method)
        if preferred_name is None:
            continue
        checkpoint_dir = root / f"seed{seed}" / method / "checkpoints"
        checkpoint_path = next(
            (candidate for candidate in _checkpoint_candidates(checkpoint_dir, method, preferred_name) if candidate.is_file()),
            None,
        )
        if checkpoint_path is None:
            missing.append(f"{method}: {checkpoint_dir}")
            continue
        overrides[method] = {"checkpoint_path": checkpoint_path.resolve().as_posix()}

    if missing:
        raise FileNotFoundError(
            "missing trained checkpoint for requested neural method(s): " + "; ".join(missing)
        )
    return overrides


__all__ = ("DEFAULT_CHECKPOINT_FILES", "resolve_checkpoint_overrides")
