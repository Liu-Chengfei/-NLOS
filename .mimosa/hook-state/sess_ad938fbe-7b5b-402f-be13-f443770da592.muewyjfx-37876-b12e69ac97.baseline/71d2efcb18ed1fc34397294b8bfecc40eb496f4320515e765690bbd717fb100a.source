"""Paper 数据集专用连续段 NLOS 注入器。

与 `liquidloc.scenarios.nlos_levels.apply_nlos_level` 的区别：
- apply_nlos_level 用 Poisson scatter 选事件 + 大脉冲（每事件独立），适合论文
  "主表"基线（铁律 4 大脉冲模型）。
- paper_nlos_injection 用用户指定的时间段（2-5s 连续段）整体加正偏差 + 高斯
  噪声，匹配主实验数据集 spec §4"遮挡连续持续 2-5s"语义。

入参:
    uwb_rows: 已物化的 UWB 行（list[dict]，含 timestamp / anchor_id / range /
        valid / quality 等字段）
    occlusion_segments: 连续遮挡段 list[(start_s, end_s)]
    nlos_level: "N0" / "N2" / "N3"
    level_cfg: 对应 nlos_levels.apply_nlos_level 的入参 cfg（含
        bias_strength_m / nlos_noise_std_m 等）
    seq_id: 用于生成确定性种子
    rng: 已 seed 的 random.Random

返回:
    注入后的 uwb_rows（in-place 修改），同时返回 nlos_report 字典供审计
"""
from __future__ import annotations

import hashlib
import random
from typing import Any


def _is_in_outage(t: float, segments: list[tuple[float, float]]) -> bool:
    for s, e in segments:
        if s <= t <= e:
            return True
    return False


def apply_paper_nlos_blocks(
    uwb_rows: list[dict[str, Any]],
    occlusion_segments: list[tuple[float, float]],
    nlos_level: str,
    level_cfg: dict[str, Any],
    seq_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """对 occlusion_segments 内的 UWB 行注入连续 NLOS 偏差 + 噪声。

    偏差与噪声从 level_cfg 读取（与 nlos_levels 字段一致）：
      bias_strength_m: 段内每次测距的正向偏差中心（米）
      nlos_noise_std_m: 附加高斯噪声标准差（米）
    """
    if nlos_level == "N0" or not occlusion_segments:
        return uwb_rows, {
            "nlos_level": nlos_level,
            "segment_count": len(occlusion_segments),
            "injected_row_count": 0,
        }

    # 区间化 2026-08-31：bias_strength_m / nlos_noise_std_m 支持 [low, high] 区间，
    # rng 在 level_cfg 读取之前创建（seq_id 派生，与事件级 RNG 独立）。
    rng = random.Random(
        int.from_bytes(
            hashlib.sha256(f"paper_nlos:{seq_id}:{nlos_level}".encode("utf-8")).digest()[:8],
            "big",
            signed=False,
        )
    )
    from liquidloc.common.validation import sample_axis_interval
    bias_strength_m = sample_axis_interval(level_cfg.get("bias_strength_m", 0.0), rng, name="bias_strength_m")
    nlos_noise_std_m = sample_axis_interval(level_cfg.get("nlos_noise_std_m", 0.0), rng, name="nlos_noise_std_m")
    if bias_strength_m <= 0.0 and nlos_noise_std_m <= 0.0:
        return uwb_rows, {
            "nlos_level": nlos_level,
            "segment_count": len(occlusion_segments),
            "injected_row_count": 0,
            "note": "level_cfg has zero bias/noise; no-op",
        }
    injected = 0
    biased_rows: list[int] = []
    for i, row in enumerate(uwb_rows):
        t = float(row.get("timestamp", 0.0))
        if not _is_in_outage(t, occlusion_segments):
            continue
        # 段内：bias + noise
        bias = bias_strength_m
        if nlos_noise_std_m > 0.0:
            bias += rng.gauss(0.0, nlos_noise_std_m)
        new_range = float(row.get("range", 0.0)) + bias
        if new_range < 0.0:
            new_range = 0.0
        row["range"] = new_range
        injected += 1
        biased_rows.append(i)

    report = {
        "nlos_level": nlos_level,
        "segment_count": len(occlusion_segments),
        "injected_row_count": injected,
        "biased_row_indices": biased_rows[:20] + (["..."] if len(biased_rows) > 20 else []),
        "bias_strength_m": bias_strength_m,
        "nlos_noise_std_m": nlos_noise_std_m,
    }
    return uwb_rows, report


__all__ = ["apply_paper_nlos_blocks"]
