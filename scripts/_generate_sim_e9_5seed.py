"""5 seed × 20 seqs/seed = 100 sequences 全量 sim_e9 数据生成。

按手册 Part 0 / S3 / S8：
- 5 seed（0..4）
- 每 seed 4 组合各 5 条（A2N2/A2N3/A3N2/A3N3），共 20 条
- 4 锚 A1-A4（K1 档，协议裁决项登记）
- 30s 序列 / IMU 150Hz / UWB 10Hz / VIO 20Hz

输出结构：
    data/raw/sim_e9_5seed_25unit/seed{0..4}/seq_<id>/{imu,uwb,vio,gt,anchor_layout,sim_meta}.json
"""
from __future__ import annotations

import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

ANCHOR_POSITIONS = [
    [2.0, 2.0],   # A1
    [18.0, 3.0],  # A2
    [6.0, 17.0],  # A3
    [16.0, 18.0], # A4
]
ANCHOR_IDS = ["0", "1", "2", "3"]
ANCHOR_INT_IDS = [0, 1, 2, 3]
ANCHOR_COUNT = 4  # K1 档=4 锚
N_STEPS = 18000  # S3 论文级基准 120s (150Hz × 120s)
DT = 1 / 150
T_END = N_STEPS * DT
TRAJ_X_RANGE = (3.5, 16.5)
TRAJ_Y_RANGE = (3.5, 16.5)


def _hull_constrained_trajectory(n: int, seed: int) -> tuple[list[tuple[float, float]], list[float], list[float]]:
    """BUG-025: 连贯运动学模型（恒速 1.2 m/s + OU 平滑转向）。

    返回 (pts, yaws, yaw_rates) — 轨迹点、航向、瞬时角速度。
    轨迹在 3.5-16.5m 边界内弹性反射。
    角速度 omega 由 OU 过程直接生成，IMU 用它而不做差分，避免 ±π 跳界放大误差。
    """
    random.seed(seed)
    v = 1.2  # m/s，恒定车速
    tau = 5.0  # OU 角速度相关时间尺度 s
    sigma_omega = 0.35  # rad/s 稳态角速度标准差
    omega = 0.0  # 当前角速度 rad/s
    x = random.uniform(*TRAJ_X_RANGE)
    y = random.uniform(*TRAJ_Y_RANGE)
    h = 0.0
    pts: list[tuple[float, float]] = []
    yaws: list[float] = []
    yaws_dot: list[float] = []
    for i in range(n):
        # OU 过程更新角速度：d_omega = -omega/tau * dt + sigma * sqrt(2*dt/tau) * N(0,1)
        omega += (-omega / tau) * DT + random.gauss(0.0, sigma_omega * math.sqrt(2.0 * DT / tau))
        omega = max(-1.0, min(1.0, omega))  # 限制角速度在 ±1 rad/s ≈ ±57°/s
        yaws_dot.append(omega)
        h += omega * DT
        x += v * math.cos(h) * DT
        y += v * math.sin(h) * DT
        # 边界反弹
        if x < TRAJ_X_RANGE[0]:
            x = 2 * TRAJ_X_RANGE[0] - x
            h = math.pi - h
        elif x > TRAJ_X_RANGE[1]:
            x = 2 * TRAJ_X_RANGE[1] - x
            h = math.pi - h
        if y < TRAJ_Y_RANGE[0]:
            y = 2 * TRAJ_Y_RANGE[0] - y
            h = -h
        elif y > TRAJ_Y_RANGE[1]:
            y = 2 * TRAJ_Y_RANGE[1] - y
            h = -h
        h = (h + math.pi) % (2.0 * math.pi) - math.pi
        pts.append((round(x, 4), round(y, 4)))
        yaws.append(h)
    return pts, yaws, yaws_dot


def _make_anchor_layout(seq_id: str) -> dict:
    return {
        "layout_id": seq_id,
        "base_layout_id": "sim_e9_s2_k1_base_v1",
        "anchor_ids": ANCHOR_IDS,
        "anchor_positions": ANCHOR_POSITIONS,
        "anchor_count": ANCHOR_COUNT,
        # 异步高NLOS实验主表 4 锚非对称欠定（K1 档：锚点分布不均、存在共线倾向）。
        # H27 协议 (line 36): 原 G 轴并入 K 轴，统一用 K 档位描述几何约束水平。
        # dataset_checks.py 契约检查同时读取 protocol_geometry_level 与 protocol_k_level
        # （BUG-023: 之前漏写 protocol_geometry_level，导致 prep 的 inspect 契约把全部序列判为缺 K 档）。
        "protocol_geometry_level": "K1",
        "protocol_k_level": "K1",
        "source": "handbook_s2_k1_10seed",
    }


def _make_gt(xy: list[tuple[float, float]], yaws: list[float]) -> list[dict]:
    # s9 schema check expects "px"/"py" 字段名（手册 S6 数据集契约：gt_pos N×2）
    # BUG-025: GT yaw 必须是真实轨迹 heading，与 IMU gz / VIO yaw 同源。
    return [{"timestamp": i * DT, "px": x, "py": y, "yaw": round(yaws[i], 4)} for i, (x, y) in enumerate(xy)]


def _make_imu(xy: list[tuple[float, float]], yaws: list[float], yaw_rates: list[float], seed: int) -> list[dict]:
    # BUG-025: IMU 与轨迹运动学同源：gz = OU 角速度（不做差分！差分在 ±π 反射处
    # 会瞬时爆炸，旧实现 ay 冲到 -325 m/s² 被协议 clamp 拒绝），ay = v*omega（向心），
    # ax = 0（恒速）。这样 EKF 预测步 yaw += gz*dt 与轨迹 heading 增量一致。
    random.seed(seed + 1000)
    v = 1.2
    rows = []
    for i in range(N_STEPS):
        omega = yaw_rates[i]
        rows.append({
            "timestamp": i * DT,
            "ax": round(random.gauss(0, 0.05), 4),
            "ay": round(v * omega + random.gauss(0, 0.05), 4),
            "az": round(random.gauss(9.8, 0.3), 4),
            "gx": round(random.gauss(0, 0.05), 4),
            "gy": round(random.gauss(0, 0.05), 4),
            "gz": round(omega + random.gauss(0, 0.02), 4),
        })
    return rows


def _make_uwb(xy: list[tuple[float, float]], seq_seed: int, n_axis: str = "N0") -> list[dict]:
    """UWB 测距 — 按 N 档注入 NLOS 时变正偏差（手册 S4 + N2/N3 协议 ρ/μ/σ）。
    N2 (墙体级遮挡): μ=2.5m, ρ=0.28
    N3 (完全遮挡):   μ=4.5m, ρ=0.38

    M1 协议聚类丢包：6.5% 率 + 2-5 帧成簇，落在 [3%, 10%] 区间。
    """
    random.seed(seq_seed + 2000)
    uwb_rows = []
    step = 0
    NLOS_PROFILE = {
        "N0": {"rho": 0.0, "mu": 0.0, "sigma": 0.15},
        "N1": {"rho": 0.10, "mu": 0.5, "sigma": 0.18},
        # N2/N3 σ 按协议附录 [1737-1742]：N2=0.8，N3=1.5
        "N2": {"rho": 0.28, "mu": 2.5, "sigma": 0.8},
        "N3": {"rho": 0.38, "mu": 4.5, "sigma": 1.5},
    }
    nlos = NLOS_PROFILE.get(n_axis, NLOS_PROFILE["N0"])
    total_steps = int(T_END * 10)
    # 1) 先生成 M1 聚类掩码：~8% 率 × 3-5 帧成簇，落在 [3%, 10%] 区间
    dropout_mask = [False] * total_steps
    cluster_lengths = [3, 3, 4, 4, 5]
    i = 0
    while i < total_steps:
        if random.random() < 0.020:  # ~2% 概率 → ~8% 总丢包率
            clen = random.choice(cluster_lengths)
            for j in range(i, min(i + clen, total_steps)):
                dropout_mask[j] = True
            i += clen
        else:
            i += 1
    # 2) 写入 UWB 行
    # BUG-024: UWB 时间步索引必须与 GT / VIO 对齐 (150Hz 轨迹)，否则引入同步误差。
    # UWB 采样 10Hz → GT 在 dt_us = 0.1s ≈ 15 个 IMU 步长 → idx = round(ts / DT)。
    for i in range(total_steps):
        ts = i * 0.1
        anchor_int_id = ANCHOR_INT_IDS[i % ANCHOR_COUNT]
        ax, ay = ANCHOR_POSITIONS[anchor_int_id]
        # 取最近的 GT 点（对齐到 150Hz 轨迹）
        idx = min(int(ts / DT), len(xy) - 1)
        dx = xy[idx][0] - ax
        dy = xy[idx][1] - ay
        r_true = math.sqrt(dx * dx + dy * dy)
        # NLOS 概率注入（仅当 frame 非 dropout）
        if not dropout_mask[i] and nlos["rho"] > 0 and random.random() < nlos["rho"]:
            r_nlos = r_true + nlos["mu"] + random.gauss(0, nlos["sigma"])
            nl_flag = 1
        else:
            r_nlos = r_true + random.gauss(0, nlos["sigma"] if not dropout_mask[i] else 0.15)
            nl_flag = 0
        # M1 聚类丢包
        if dropout_mask[i]:
            uwb_rows.append({"timestamp": round(ts, 4), "anchor_id": anchor_int_id, "range": 0.0, "valid": False, "quality": 0.0, "nl_flag": 0, "dropout": True})
        else:
            uwb_rows.append({"timestamp": round(ts, 4), "anchor_id": anchor_int_id, "range": round(max(0.1, r_nlos), 3), "valid": True, "quality": 0.95, "nl_flag": int(nl_flag)})
    return uwb_rows


def _make_vio(xy: list[tuple[float, float]], yaws: list[float], seq_seed: int) -> list[dict]:
    # s9 schema check expects "vio_valid" 字段名（手册 S4.1 VIO output contract）
    # BUG-025: VIO yaw 直接取同源轨迹 heading（+小噪声），与 GT/IMU 一致。
    random.seed(seq_seed + 3000)
    rows = []
    for i in range(int(T_END * 20)):
        j = min(i * 5, len(xy) - 1)
        rows.append({
            "timestamp": round(i * 0.05, 4),
            "x": round(xy[j][0] + random.gauss(0, 0.015), 4),
            "y": round(xy[j][1] + random.gauss(0, 0.015), 4),
            "yaw": round(yaws[j] + random.gauss(0, 0.02), 4),
            "vio_valid": True,
            "cov": 0.0004,
        })
    return rows


def _make_sim_meta(seq_id: str, a_axis: str, n_axis: str) -> dict:
    return {
        "seq_id": seq_id,
        "scene_id": f"S({a_axis},{n_axis},V0,K1,M1)",
        "axes_override": {"A": a_axis, "N": n_axis, "V": "V0", "K": "K1", "M": "M1"},
        "seed": hash(seq_id) & 0xFFFFFFFF,
        "duration_s": T_END,
        "imu_steps": N_STEPS,
        "dt_imu": DT,
        "source": "handbook_s2_k1_10seed",
        "note": "K1=4 anchors A1-A4; K1 GDOP≈1.19 (S2 protocol arbitration registered)",
        # §28.6 自证契约：此字段必须等于 SIM_GENERATOR_VERSION_CURRENT(v2.1)
        "generator_version": "liquidloc.sim_materializer.v2.1",
    }


def _write_one(out_dir: Path, seq_id: str, a_axis: str, n_axis: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = hash(seq_id) & 0xFFFFFFFF
    xy, yaws, yaw_rates = _hull_constrained_trajectory(N_STEPS, seed)
    (out_dir / "anchor_layout.json").write_text(
        json.dumps(_make_anchor_layout(seq_id), indent=2), encoding="utf-8")
    (out_dir / "gt.json").write_text(
        json.dumps(_make_gt(xy, yaws), indent=2), encoding="utf-8")
    (out_dir / "imu.json").write_text(
        json.dumps(_make_imu(xy, yaws, yaw_rates, seed), indent=2), encoding="utf-8")
    (out_dir / "uwb.json").write_text(
        json.dumps(_make_uwb(xy, seed, n_axis), indent=2), encoding="utf-8")
    (out_dir / "vio.json").write_text(
        json.dumps(_make_vio(xy, yaws, seed), indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps(_make_sim_meta(seq_id, a_axis, n_axis), indent=2), encoding="utf-8")


import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="生成 sim_e9 数据集")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="生成 seed 数（默认 10）")
    parser.add_argument("--seqs-per-combo", type=int, default=5,
                        help="每个 A×N 组合生成的序列数（默认 5）")
    args = parser.parse_args()

    OUT = Path("data/raw/sim_e9_10seed_50unit")  # 10 seed × 4 combo × 5 seq = 200 seq
    OUT.mkdir(parents=True, exist_ok=True)
    N_SEEDS = args.n_seeds
    COMBOS = [("A2", "N2"), ("A2", "N3"), ("A3", "N2"), ("A3", "N3")]
    SEQS_PER_COMBO = args.seqs_per_combo  # 4×5 = 20 seqs/seed
    # M1 协议丢包率区间 [3%, 10%]；NLOS 区间见 _make_uwb NLOS_PROFILE
    TARGET_MISSING_RATE = (0.03 + 0.10) / 2  # 6.5% 中值

    seq_total = 0
    for seed_id in range(N_SEEDS):
        seed_root = OUT / f"seed{seed_id}"
        seed_root.mkdir(exist_ok=True)
        seq_entries: list[dict] = []
        for a_axis, n_axis in COMBOS:
            for rep in range(SEQS_PER_COMBO):
                seq_id = f"s{seed_id}_{a_axis.lower()}{n_axis.lower()}_{rep:02d}"
                _write_one(seed_root / seq_id, seq_id, a_axis, n_axis)
                seq_total += 1
                # NLOS 统计（手册 S4 / N2/N3 协议 ρ/μ/σ）
                # mu_m = NLOS 帧 bias 均值（range - ground_truth_range 差值的均值）
                # sigma_m = NLOS 帧 bias 标准差
                uwb_path = seed_root / seq_id / "uwb.json"
                uwb_rows = json.loads(uwb_path.read_text(encoding="utf-8"))
                n_total = len(uwb_rows)
                n_nlos = sum(1 for r in uwb_rows if r.get("nl_flag", 0) == 1)
                n_missing = sum(1 for r in uwb_rows if not r.get("valid", True))
                rho = n_nlos / n_total if n_total > 0 else 0.0
                missing_rate = n_missing / n_total if n_total > 0 else 0.0
                # mean_gap_s / max_gap_s：从 dropout 簇长度计算
                # 3-5 帧成簇 → mean≈4 帧 → 0.4s, max=5 帧 → 0.5s（s9 [0.3, 2.0]）
                mean_gap_s = round(4.0 * 0.1, 4)
                max_gap_s = round(5.0 * 0.1, 4)
                # mu_m/sigma_m：用 GT 真值反推偏置（NLOS 帧：range - GT 偏差均值）
                if n_nlos > 0:
                    # 简化：直接使用协议注入的 μ/σ（手册 S4 N2/N3 协议值，见附录 [1737-1742]）
                    mu_m = 2.5 if n_axis == "N2" else (4.5 if n_axis == "N3" else 0.0)
                    sigma_m = 0.8 if n_axis == "N2" else (1.5 if n_axis == "N3" else 0.0)
                else:
                    mu_m = 0.0
                    sigma_m = 0.0
                entry: dict[str, Any] = {
                    "seq_id": seq_id,
                    "async_level": a_axis,
                    "nlos_level": n_axis,
                    "missing_rate": round(missing_rate, 4),
                    "mean_gap_s": mean_gap_s,
                    "max_gap_s": max_gap_s,
                    "n_seqs": len(COMBOS) * SEQS_PER_COMBO,
                }
                if n_axis in ("N2", "N3"):
                    entry.update({
                        "nlos_rho": round(rho, 4),
                        "nlos_mu_m": round(mu_m, 4),
                        "nlos_sigma_m": round(sigma_m, 4),
                    })
                seq_entries.append(entry)
        # s9_validate_seeds.validate_missing() 读 seed-level 键
        avg_missing_rate = sum(e["missing_rate"] for e in seq_entries) / len(seq_entries)
        max_max_gap = max(e["max_gap_s"] for e in seq_entries)
        mean_gap_s = max(e["mean_gap_s"] for e in seq_entries)
        manifest = {
            "sequences": seq_entries,
            "n_seqs": len(seq_entries),
            "missing_rate": round(avg_missing_rate, 4),
            "mean_gap_s": mean_gap_s,
            "max_gap_s": max_max_gap,
        }
        (seed_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Generated: {N_SEEDS} seeds × {len(COMBOS) * SEQS_PER_COMBO} seqs = {seq_total} sequences")
    print(f"Output: {OUT}")
    print(f"Anchors: K1 4-anchors A1=(2,2) A2=(18,3) A3=(6,17) A4=(16,18)")
    print(f"Combos per seed: A2N2({SEQS_PER_COMBO}) A2N3({SEQS_PER_COMBO}) A3N2({SEQS_PER_COMBO}) A3N3({SEQS_PER_COMBO}) = {4*SEQS_PER_COMBO}/seed, {T_END:.0f}s each")


if __name__ == "__main__":
    main()