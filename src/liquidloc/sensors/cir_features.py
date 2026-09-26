"""CIR (Channel Impulse Response) 统计特征计算模块。

文件职责：提供 CIR 幅度序列的首径幅度、峰值幅度、延迟扩展等统计特征计算工具，
以及 LOS/NLOS 型 CIR 合成函数和批量计算接口。

输出规范：
- 每个 CIR 帧 → 3 个标量特征：(first_path_amplitude, peak_amplitude, delay_spread)
- 批量矩阵 (N, L) → 特征矩阵 (N, 3)
- 格式：npz 文件含 first_path_amplitude / peak_amplitude / delay_spread 三个数组

计算定义：
    first_path_amplitude: 首个超过 amplitude_threshold 的样本幅度（归一化后）。
        若所有样本均未达阈值，则取全局最大样本位置处的归一化幅度。
    peak_amplitude: CIR 全局最大归一化幅度（即 ||CIR||_inf / ||CIR||_inf = 1.0）。
        若全零则返回 0.0。
    delay_spread: RMS 延迟扩展，以采样点索引为单位，衡量多径散布程度。
        ds = sqrt(sum_i w_i * (i - i_mean)^2) / max(w)，其中 w = abs(CIR)^2。

合成模型：
    LOS: 高首径峰值 + 低多径尾（指数扩散慢）
    NLOS: 低首径 + 高多径尾（指数扩散快），随 NLOS 档位增强

依赖：
    numpy（必须）
"""

from __future__ import annotations

import random as _random
from typing import Sequence

import numpy as np


# ─── CIR 特征计算 ───────────────────────────────────────────────────────────

def compute_cir_features(
    cir: np.ndarray,
    *,
    amplitude_threshold: float = 0.01,
) -> tuple[float, float, float]:
    """计算单帧 CIR 的三项统计特征。

    参数：
        cir: 1D 复数或实数 CIR 幅度序列，形状 (L,)。
        amplitude_threshold: 首径检测幅度阈值，低于此值被视为噪声。

    返回：
        (first_path_amplitude, peak_amplitude, delay_spread)
        - first_path_amplitude: 归一化首径幅度 ∈ [0, 1]
        - peak_amplitude: 归一化峰值幅度 ∈ [0, 1]
        - delay_spread: RMS 延迟扩展（采样点索引）≥ 0

    异常：
        ValueError: cir 不是 1D 时抛出。
    """
    if cir.ndim != 1:
        raise ValueError(f"CIR must be 1D, got shape {cir.shape}")
    l = len(cir)
    if l == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    abs_cir = np.abs(cir).astype(np.float32)
    max_val = float(np.max(abs_cir))
    if max_val <= 1e-12:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # 归一化
    norm_cir = abs_cir / max_val

    # 峰值幅度 = 1.0（归一化后），但保留原始位置供验证
    peak_amp = 1.0

    # 首径：首个超过阈值的样本
    threshold_idx = None
    for i in range(l):
        if norm_cir[i] >= amplitude_threshold:
            threshold_idx = i
            break
    if threshold_idx is None:
        # 无样本超过阈值，用全局最大位置
        threshold_idx = int(np.argmax(norm_cir))

    first_path_amp = float(norm_cir[threshold_idx])

    # 延迟扩展：RMS 加权标准差
    weights = abs_cir ** 2
    w_sum = float(np.sum(weights))
    if w_sum <= 1e-12:
        delay_spread = 0.0
    else:
        mean_idx = float(np.sum(weights * np.arange(l, dtype=np.float32))) / w_sum
        variance = float(np.sum(weights * (np.arange(l, dtype=np.float32) - mean_idx) ** 2)) / w_sum
        delay_spread = float(np.sqrt(max(variance, 0.0)))

    return np.array([first_path_amp, peak_amp, delay_spread], dtype=np.float32)


def compute_all_cir_features(
    cir_matrix: np.ndarray,
    *,
    amplitude_threshold: float = 0.01,
) -> np.ndarray:
    """批量计算 CIR 矩阵的特征，返回 (N, 3) 数组。

    参数：
        cir_matrix: 2D CIR 幅度矩阵，形状 (N, L)。
        amplitude_threshold: 同 compute_cir_features。

    返回：
        features: (N, 3) 数组，列分别为
            first_path_amplitude, peak_amplitude, delay_spread。

    异常：
        ValueError: cir_matrix 不是 2D 时抛出。
    """
    if cir_matrix.ndim != 2:
        raise ValueError(f"CIR matrix must be 2D, got shape {cir_matrix.shape}")
    n, _ = cir_matrix.shape
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    features = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        features[i] = compute_cir_features(cir_matrix[i], amplitude_threshold=amplitude_threshold)
    return features


def validate_cir_features_array(
    features: np.ndarray,
    *,
    expected_frames: int | None = None,
) -> bool:
    """验证 CIR 特征数组的形状与取值区间。

    参数：
        features: (N, 3) 特征数组。
        expected_frames: 若指定，断言行数与此值相等。

    返回：
        True 若验证通过。

    异常：
        ValueError: 形状或取值区间不符合约束时抛出。
    """
    if features.ndim != 2:
        raise ValueError(f"CIR features must be 2D array, got shape {features.shape}")
    if features.shape[1] != 3:
        raise ValueError(
            f"CIR features second dim must be 3 (first_path, peak, delay_spread), "
            f"got {features.shape[1]}"
        )
    n = features.shape[0]
    if n < 1:
        raise ValueError(f"CIR features must have at least 1 frame, got {n}")
    if expected_frames is not None and n != expected_frames:
        raise ValueError(
            f"CIR features frame count mismatch: got {n}, expected {expected_frames}"
        )

    first_path = features[:, 0]
    peak = features[:, 1]
    delay = features[:, 2]

    if float(np.min(first_path)) < 0.0 or float(np.max(first_path)) > 1.0:
        raise ValueError(
            f"first_path_amplitude out of range [0, 1]: "
            f"[{float(np.min(first_path)):.4f}, {float(np.max(first_path)):.4f}]"
        )
    if float(np.min(peak)) < 0.0 or float(np.max(peak)) > 1.0:
        raise ValueError(
            f"peak_amplitude out of range [0, 1]: "
            f"[{float(np.min(peak)):.4f}, {float(np.max(peak)):.4f}]"
        )
    if float(np.min(delay)) < 0.0:
        raise ValueError(
            f"delay_spread must be >= 0, got min={float(np.min(delay)):.4f}"
        )
    return True


# ─── CIR 合成模型 ───────────────────────────────────────────────────────────

def generate_los_cir(
    length: int = 64,
    *,
    first_path_amp: float = 0.85,
    seed: int | None = None,
) -> np.ndarray:
    """生成 LOS 型 CIR 幅度序列。

    特征：高首径峰值（位置靠前）、低多径尾、小延迟扩展。

    参数：
        length: CIR 长度（采样点数）。
        first_path_amp: 首径峰值归一化幅度，应在 (0, 1] 范围。
        seed: 随机种子，用于可复现性。

    返回：
        cir: 1D float32 数组，形状 (length,)。
    """
    rng = _random.Random(seed)
    cir = np.zeros(length, dtype=np.float32)

    # 首径脉冲放在 index 4（靠前位置），便于 first_path 检测
    fp_idx = 4
    # 先叠加噪声基底（从 fp_idx 开始，fp_idx 之前保持零值）
    for i in range(fp_idx, length):
        cir[i] = 0.008 * rng.uniform(0, 1)

    # 首径峰值，强度远高于噪声基底（归一化后仍远高于阈值）
    cir[fp_idx] = first_path_amp
    # 在首径附近叠加微小扰动（仅影响 fp_idx 及之后，不碰前向零值区域）
    for j in range(fp_idx, min(length, fp_idx + 3)):
        if j != fp_idx:
            cir[j] += first_path_amp * 0.03 * rng.uniform(0, 1)  # 只加不减，确保低于首径

    # 低强度多径尾（指数衰减，强度远低于首径）
    tail_amp = 0.08 * first_path_amp
    for i in range(fp_idx + 1, length):
        dist = i - fp_idx
        cir[i] += tail_amp * np.exp(-dist / (length * 0.15))

    # 归一化到 [0, 1]
    mx = float(np.max(np.abs(cir)))
    if mx > 1e-12:
        cir = cir / mx
    else:
        cir[fp_idx] = 1.0

    return cir.astype(np.float32)


def generate_nlos_cir(
    length: int = 64,
    *,
    first_path_amp: float = 0.2,
    nlos_level: int = 2,
    seed: int | None = None,
) -> np.ndarray:
    """生成 NLOS 型 CIR 幅度序列。

    特征：低首径峰值、强多径尾、大延迟扩展，随 NLOS 档位增强。

    参数：
        length: CIR 长度。
        first_path_amp: 首径归一化幅度（NLOS 下较弱）。
        nlos_level: NLOS 档位 (0-3)，越大则多径越严重。
        seed: 随机种子。

    返回：
        cir: 1D float32 数组，形状 (length,)。
    """
    rng = _random.Random(seed)
    cir = np.zeros(length, dtype=np.float32)

    # 弱首径
    fp_idx = length // 6 + rng.randint(-2, 2)
    fp_idx = max(1, min(length - 1, fp_idx))
    cir[fp_idx] = first_path_amp

    # 强多径尾（随 NLOS 档位增强）
    tail_strength = 0.1 + nlos_level * 0.15
    for i in range(length):
        dist = abs(i - fp_idx)
        cir[i] += tail_strength * np.exp(-dist / (length * 0.15 / max(1, nlos_level + 1)))
        cir[i] += tail_strength * 0.3 * rng.uniform(-1, 1)

    # 散斑噪声
    cir += 0.03 * np.array([rng.uniform(0, 1) for _ in range(length)], dtype=np.float32)

    # 归一化
    mx = float(np.max(np.abs(cir)))
    if mx > 1e-12:
        cir = cir / mx
    else:
        cir[fp_idx] = 1.0

    return cir.astype(np.float32)


# ─── 公开导出 ──────────────────────────────────────────────────────────────

__all__ = (
    "compute_cir_features",
    "compute_all_cir_features",
    "validate_cir_features_array",
    "generate_los_cir",
    "generate_nlos_cir",
)
