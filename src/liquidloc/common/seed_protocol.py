"""P19 deterministic 种子体系实现（手册 Part 2 P19 硬约束）。

本模块是种子体系的单源真相（SSOT），包含：
- TRAIN_SEEDS × DATA_SEED_MULTIPLIER = N × 3 个完整种子组合（手册要求 5×3=15）。
- PYTHONHASHSEED=0 在所有启动路径的置顶设置。
- pipeline 入口设种子（train_pipeline / eval_pipeline / inference_pipeline）。

用法：
    from liquidloc.common.seed_protocol import (
        TRAIN_SEEDS, DATA_SEED_MULTIPLIER, get_full_seed_grid,
        ensure_pythonhashseed, set_pipeline_seed, seed_everything,
    )
"""

from __future__ import annotations

import os
import random
import sys

# ============================================================================
# P19-1: 训练种子常量（手册 line ~120，要求 5×3=15 个种子组合）
# ============================================================================
# 5 个训练种子 × 3 个数据种子乘数 = 15 个独立种子组合（可完整填满 paper_main.yaml 的 seed 维度）。
TRAIN_SEEDS: tuple[int, ...] = (42, 2024, 7, 314, 999)
DATA_SEED_MULTIPLIER: tuple[int, ...] = (1, 1000, 1000000)


def get_full_seed_grid() -> list[int]:
    """返回 P19 规定的完整种子网格（5×3=15 个种子）。"""
    return [s * m for s in TRAIN_SEEDS for m in DATA_SEED_MULTIPLIER]


# ============================================================================
# P19-2: PYTHONHASHSEED=0（手册 line ~121）
# 说明：此变量必须在进程启动时设置（import 前），模块无法在运行时修改它。
# 因此本函数在启动脚本（run_train.sh / conda activate hook）调用，模块内提供
# 审计函数供入口脚本调用。
# ============================================================================
_PYTHONHASHSEED_ENFORCED: bool = False


def ensure_pythonhashseed(seed: int = 0) -> bool:
    """确保 PYTHONHASHSEED 环境变量被设置为目标值（通常为 0）。

    P19 硬约束要求所有启动路径设 PYTHONHASHSEED=0。
    此函数在启动脚本（如 run_train.sh / activate hook）顶层调用；
    在 Python 模块内部调用时，如果当前进程没有正确设置，会发出警告。

    Returns:
        bool: 是否已正确设置（True = 进程启动时就已设置，False = 警告但继续）。
    """
    global _PYTHONHASHSEED_ENFORCED
    current = os.environ.get("PYTHONHASHSEED")
    if current is None:
        import warnings

        warnings.warn(
            "[P19 WARNING] PYTHONHASHSEED is not set. "
            "Set PYTHONHASHSEED=0 in the shell before launching Python. "
            "Hash-ordered data structures (dicts, sets) may differ across runs.",
            RuntimeWarning,
            stacklevel=2,
        )
        _PYTHONHASHSEED_ENFORCED = False
        return False
    elif int(current) != seed:
        import warnings

        warnings.warn(
            f"[P19 WARNING] PYTHONHASHSEED={current}, expected {seed}. "
            "Restart the process with PYTHONHASHSEED=0.",
            RuntimeWarning,
            stacklevel=2,
        )
        _PYTHONHASHSEED_ENFORCED = False
        return False
    else:
        _PYTHONHASHSEED_ENFORCED = True
        return True


def is_pythonhashseed_enforced() -> bool:
    """返回 PYTHONHASHSEED 是否已在进程启动时正确设置。"""
    return _PYTHONHASHSEED_ENFORCED


# ============================================================================
# P19-3: pipeline 入口设种子（手册 line ~122）
# ============================================================================
def set_pipeline_seed(seed: int, deterministic: bool = True) -> dict:
    """在 pipeline 入口设种子（调用 set_global_seed 并额外设置 CUBLAS）。

    来自 liquidloc.common.seed_utils.set_global_seed。
    这是 train / eval / inference 入口的唯一种子设置路径。

    Args:
        seed: 随机种子。
        deterministic: 是否开启确定性（影响 cuDNN / cuBLAS）。

    Returns:
        dict: 种子审计报告。
    """
    from liquidloc.common.seed_utils import set_global_seed

    report = set_global_seed(seed, deterministic=deterministic)

    # P19 额外设置：CUBLAS workspace（确定性运算需要）。
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 审计：记录种子体系元数据。
    report["pythonhashseed_enforced"] = is_pythonhashseed_enforced()
    report["p19_seed_protocol"] = True
    report["seed_grid_description"] = "TRAIN_SEEDS × DATA_SEED_MULTIPLIER (5×3=15)"
    return report


def seed_everything(seed: int, deterministic: bool = True) -> dict:
    """P19 的对外别名（兼容旧调用方）。"""
    return set_pipeline_seed(seed, deterministic=deterministic)


# ============================================================================
# P19 快速审计辅助
# ============================================================================
def audit_deterministic_setup(seed: int) -> dict[str, bool | str]:
    """P19 deterministic 设置完整审计。

    返回各组件的确定性状态（供实验元数据记录）。
    """
    status: dict[str, bool | str] = {}

    # PYTHONHASHSEED
    phs = os.environ.get("PYTHONHASHSEED")
    status["pythonhashseed_correct"] = phs == "0"
    status["pythonhashseed_value"] = phs or "NOT SET"

    # CUBLAS workspace
    status["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "NOT SET")

    # Python hash is deterministic
    # dict iteration order is consistent within a session
    status["python_version"] = sys.version.split()[0]
    return status


__all__ = [
    "TRAIN_SEEDS",
    "DATA_SEED_MULTIPLIER",
    "get_full_seed_grid",
    "ensure_pythonhashseed",
    "is_pythonhashseed_enforced",
    "set_pipeline_seed",
    "seed_everything",
    "audit_deterministic_setup",
]
