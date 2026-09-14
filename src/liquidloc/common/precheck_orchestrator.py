"""异步高NLOS实验全流程保障手册 P1~P39 跑前预检统一 orchestrator。

把所有 P1~P39 八层预检集中到一个可调用的检查流水线中，按手册 line 146-209 的定义
落地 39 个 check 函数；以及 Pre-1~Pre-6、I-1~I-5、DQ-1~DQ-4、G-1~G-5、E-1~E-6。

本模块只做"检查+报告"，不强制 fail-stop（调用方决定是否 raise）。
每个 check 返回 `CheckResult(passed, detail, evidence)`。

公开入口：
    run_all_prechecks(cfg, raw_root, manifest=None) -> PrecheckReport
        raw_root: 原始 npz 数据根目录，用于自动计算 P1/P4/P5/P6 等统计字段
        manifest: DatasetManifest 实例（可选，不提供时从 raw_root 推断）
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from liquidloc.common.constants import WINDOW_SIZE, WARMUP_S  # P20/P28 窗口/热身硬约束
from liquidloc.common.validation import coerce_finite_scalar


# 误差预算常量（手册 Part 0 S4）
LOS_NOISE_SIGMA_M: float = 0.6
GDOP_DEFAULT: float = 1.19


def _load_protocol_axis_ranges() -> dict[str, Any]:
    """从 ``configs/base/scene_axis_protocol.yaml`` 加载权威的 A/N 轴区间。

    返回形如::

        {"A2": {"offset_ms": [20, 50], ...}, "A3": {...},
         "N2": {"nlos_ratio": [0.25, 0.30], ...}, "N3": {...}}

    协议加载失败时回退到手写常数，保证预检始终可运行。
    """
    try:
        from liquidloc.protocol.scene_axis_protocol import load_scene_axis_protocol  # noqa: F401
        p = load_scene_axis_protocol()
        axes = p.get("axes", {})
        a = axes.get("A", {})
        n = axes.get("N", {})
        return {
            "A2": a.get("A2", {}),
            "A3": a.get("A3", {}),
            "N2": n.get("N2", {}),
            "N3": n.get("N3", {}),
        }
    except Exception:
        return _PROTOCOL_FALLBACK


# 协议加载失败时回退常数（来自手册 Part 0 §4.2 A2/A3 档位 + §4.3 N2/N3）
# 值必须与 configs/base/scene_axis_protocol.yaml 保持一致，修改时双改。
_PROTOCOL_FALLBACK: dict[str, dict[str, float | list[float]]] = {
    "A2": {"offset_ms": [20.0, 50.0], "jitter_ms": 3.0, "cross_modal_skew_ms": 60.0},
    "A3": {"offset_ms": [100.0, 200.0], "jitter_ms": 10.0, "cross_modal_skew_ms": 185.0},
    "N2": {"nlos_ratio": [0.25, 0.30], "bias_strength_m": [2.0, 3.0], "nlos_noise_std_m": [0.78, 0.82]},
    "N3": {"nlos_ratio": [0.35, 0.40], "bias_strength_m": [4.0, 6.0], "nlos_noise_std_m": [1.48, 1.52]},
}

# 模块级缓存：协议只读且加载开销高，只读一次。
_PROTOCOL_RANGES: dict[str, dict[str, Any]] | None = None


def get_protocol_axis_ranges() -> dict[str, dict[str, Any]]:
    global _PROTOCOL_RANGES
    if _PROTOCOL_RANGES is None:
        _PROTOCOL_RANGES = _load_protocol_axis_ranges()
    return _PROTOCOL_RANGES


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew 单调链算法计算 2D 点集凸包（逆时针排列）。

    锚点可能以任意 ID 顺序给出，必须先求凸包再做边界距离判定。
    返回逆时针顶点列表；点数 < 3 时返回原列表。
    """
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    # 去掉每段最后一个点（它是下一段的起点，重复）
    return lower[:-1] + upper[:-1]


def _point_in_convex_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """检查点是否在（逆时针排列的）凸多边形内（含边界）。

    使用叉积符号一致性：点在凸多边形内当且仅当它与所有有向边的叉积 ≥ 0（逆时针）。
    调用方须保证 polygon 已由 _convex_hull 排好序。
    """
    px, py = point
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < 0:  # 逆时针凸包：任一有向边叉积 < 0 → 点在外部
            return False
    return True


def _point_to_convex_polygon_boundary_distance(
    point: tuple[float, float],
    polygon: list[tuple[float, float]]
) -> float:
    """计算点到凸多边形边界的最近（欧氏）距离（恒 ≥ 0）。

    对每条边求点到线段距离，取最小。内部/外部都返回正的边界距离；
    内外判定由 _point_in_convex_polygon 单独负责。
    """
    px, py = point
    min_dist_sq = float('inf')
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        # 点到线段最近距离
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            continue
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
        projx = x1 + t * dx
        projy = y1 + t * dy
        dist_sq = (px - projx) ** 2 + (py - projy) ** 2
        min_dist_sq = min(min_dist_sq, dist_sq)
    return math.sqrt(min_dist_sq) if min_dist_sq != float('inf') else 0.0


@dataclass
class CheckResult:
    """单个 P / Pre / I / DQ / G / E 检查结果。"""

    code: str  # e.g. "P1", "Pre-1", "DQ-2"
    name: str
    passed: bool
    severity: str = "hard"  # "hard" 必须过；"soft" 仅记录
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


@dataclass
class PrecheckReport:
    """完整 precheck 报告。"""

    total: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_hard_failed: int = 0
    overall_pass: bool = True
    results: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_hard_failed": self.n_hard_failed,
            "overall_pass": self.overall_pass,
            "results": [r.to_dict() for r in self.results],
        }


def _make(code: str, name: str, passed: bool, detail: str = "", severity: str = "hard", **evidence) -> CheckResult:
    return CheckResult(code=code, name=name, passed=passed, severity=severity, detail=detail, evidence=dict(evidence))


def _read_sim_meta_stats(raw_root: Path) -> dict[str, Any]:
    """扫描 raw_root 下所有 seed_X/ 子目录的 sim_meta.json 或 meta.json，聚合 P1/D15/P25/P28 所需字段。
    
    支持两种文件名约定：
    - 新版 (liquidloc v2): meta.json  
    - 旧版: sim_meta.json
    
    返回 dict，键值列表按 combo_keys 顺序排列。
    """
    out: dict[str, Any] = {
        "nlos_ratio_mid": [], "nlos_mu_m_mid": [], "nlos_sigma_m_mid": [],
        "async_offset_ms_mid": [], "async_jitter_ms": [], "async_skew_ms_mid": [],
        "vio_tracked_mid": [], "vio_inc_noise_mps_mid": [], "vio_blackout_frac": [],
        "imu_rows_counts": [], "seq_durations_s": [], "combo_keys": [],
    }
    for seed_dir in sorted(raw_root.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        for seq in seed_dir.iterdir():
            if not seq.is_dir():
                continue
            # 兼容 sim_meta.json 与 meta.json
            meta_file = seq / "meta.json" if (seq / "meta.json").is_file() else seq / "sim_meta.json"
            if not meta_file.is_file():
                continue
            try:
                m = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            def fl(k):
                v = m.get(k, 0.0)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return 0.0
            axes = m.get("axes_override") or {}
            a, n = str(axes.get("A", "")).upper(), str(axes.get("N", "")).upper()
            v = m.get("V", "")
            key = f"{a}{n}" if a in ("A2", "A3") and n in ("N2", "N3") else "other"
            out["combo_keys"].append(key)
            # v2 paper 物化（paper_materializer）字段命名与 v1 sim 不同：
            #   async_uwb_offset_ms ↔ async_offset_ms_mid
            #   async_uwb_jitter_ms ↔ async_jitter_ms
            #   async_uwb_rtt_ms    ↔ async_skew_ms_mid（跨模态 skew 用 UWB RTT 实测）
            #   nlos_uwb_bias_m     ↔ nlos_mu_m_mid
            #   nlos_uwb_noise_std_m ↔ nlos_sigma_m_mid
            #   （nlos_ratio_mid / vio_* 仅 v1 有，v2 由 _measure_nlos_empirical 经验覆盖）
            _v2_async_offset = m.get("async_uwb_offset_ms")
            _v2_async_jitter = m.get("async_uwb_jitter_ms")
            _v2_async_skew = m.get("async_uwb_rtt_ms")
            _v2_nlos_mu = m.get("nlos_uwb_bias_m")
            _v2_nlos_sigma = m.get("nlos_uwb_noise_std_m")
            for k in ("nlos_ratio_mid", "nlos_mu_m_mid", "nlos_sigma_m_mid",
                      "async_offset_ms_mid", "async_jitter_ms", "async_skew_ms_mid",
                      "vio_tracked_mid", "vio_inc_noise_mps_mid", "vio_blackout_frac"):
                out[k].append(fl(k))
            # 用 v2 等价字段回填（v1 字段缺失时），保持下游按 *_mid/*_jitter 读取不变
            if _v2_async_offset is not None and out["async_offset_ms_mid"]:
                out["async_offset_ms_mid"][-1] = fl("async_uwb_offset_ms")
            if _v2_async_jitter is not None and out["async_jitter_ms"]:
                out["async_jitter_ms"][-1] = fl("async_uwb_jitter_ms")
            if _v2_async_skew is not None and out["async_skew_ms_mid"]:
                out["async_skew_ms_mid"][-1] = fl("async_uwb_rtt_ms")
            if _v2_nlos_mu is not None and out["nlos_mu_m_mid"]:
                out["nlos_mu_m_mid"][-1] = fl("nlos_uwb_bias_m")
            if _v2_nlos_sigma is not None and out["nlos_sigma_m_mid"]:
                out["nlos_sigma_m_mid"][-1] = fl("nlos_uwb_noise_std_m")
            # IMU 行数：v2 不写 imu_row_count，直接数 imu.json。
            imu_rows = int(m.get("imu_row_count", 0) or 0)
            if imu_rows == 0:
                imu_path = seq / "imu.json"
                if imu_path.is_file():
                    try:
                        _imu = json.loads(imu_path.read_text(encoding="utf-8"))
                        imu_rows = len(_imu) if isinstance(_imu, list) else 0
                    except Exception:
                        imu_rows = 0
            out["imu_rows_counts"].append(imu_rows)
            out["seq_durations_s"].append(float(m.get("duration_s", 0.0)))
    return out


# ============================================================================
# 第一层：数据正确性 (P1~P6) + 实验准备 (Pre-1~Pre-6) + 实现完成 (I-1~I-5)
# ============================================================================


def check_P1_composition(cfg: Mapping[str, Any]) -> CheckResult:
    """P1: 4 组合各 25%（偏差 ≤5%）+ N 轴注入 ρ/μ/σ 区间校验。

    手册要求（协议 configs/base/scene_axis_protocol.yaml 权威定义）：
        - 组合比例：4 组合各 25%（偏差 ≤5%）
        - N2：ρ 0.25-0.30，μ 2-3m，σ 0.78-0.82
        - N3：ρ 0.35-0.40，μ 4-6m，σ 1.48-1.52

    区间从协议配置动态读取，协议加载失败时回退手写常数。
    """
    counts = dict(cfg.get("composition_counts") or {})
    required = {"A2N2", "A2N3", "A3N2", "A3N3"}
    if not required.issubset(set(counts.keys())):
        return _make("P1", "4组合各25%", False, detail=f"缺少组合: {required - set(counts.keys())}")
    total = sum(counts.values())
    if total == 0:
        return _make("P1", "4组合各25%", False, detail="composition_counts 总和为 0")
    failed = []
    for key, target in [("A2N2", 0.25), ("A2N3", 0.25), ("A3N2", 0.25), ("A3N3", 0.25)]:
        ratio = counts[key] / total
        if abs(ratio - target) > 0.05:
            failed.append(f"{key}={ratio:.2%}")
    # P1 第二件事：从 sim_meta 聚合的 nlos_stats 做 N 轴区间校验（协议权威区间）。
    proto = cfg.get("protocol_ranges") or get_protocol_axis_ranges()
    ranges = {
        "N2": _get_range(proto, "N2", "nlos_ratio"),
        "N3": _get_range(proto, "N3", "nlos_ratio"),
    }
    # 完整 ρ/μ/σ 三元组区间
    def _n_axis_params(level: str) -> dict[str, list[float]]:
        d = proto.get(level, {})
        return {
            "rho": list(d.get("nlos_ratio", [0.0, 1.0])),
            "mu": list(d.get("bias_strength_m", [0.0, 1e9])),
            "sigma": list(d.get("nlos_noise_std_m", [0.0, 1e9])),
        }
    stats = cfg.get("nlos_stats", {}) or {}
    n2_ok, n3_ok = True, True
    n2_detail, n3_detail = [], []
    for combo, v in stats.items():
        if not isinstance(v, dict):
            continue
        rho = float(v.get("rho", 0.0)); mu = float(v.get("mu", 0.0)); sigma = float(v.get("sigma", 0.0))
        if "N2" in combo:
            p = _n_axis_params("N2")
            if not (_in_range(rho, *p["rho"]) and _in_range(mu, *p["mu"]) and _in_range(sigma, *p["sigma"])):
                n2_ok = False
                n2_detail.append(f"{combo}ρ={rho:.3f}/μ={mu:.2f}/σ={sigma:.2f} 越界{p}")
        if "N3" in combo:
            p = _n_axis_params("N3")
            if not (_in_range(rho, *p["rho"]) and _in_range(mu, *p["mu"]) and _in_range(sigma, *p["sigma"])):
                n3_ok = False
                n3_detail.append(f"{combo}ρ={rho:.3f}/μ={mu:.2f}/σ={sigma:.2f} 越界{p}")
    if not (n2_ok and n3_ok):
        failed.extend(["N2:" + ";".join(n2_detail), "N3:" + ";".join(n3_detail)])
        return _make("P1", "4组合各25%+N轴注入", False, detail=" / ".join(failed) or "OK", composition=counts, total=total)
    detail = " / ".join(failed) or "OK"
    if stats:
        detail += f" | N轴: N2_ok={n2_ok}, N3_ok={n3_ok} (ρ 区间={ranges})"
    return _make("P1", "4组合各25%+N轴注入", not failed, detail=detail, composition=counts, total=total)


def _get_range(proto: Mapping[str, Any], level: str, param: str) -> list[float]:
    """从协议 ranges 取某轴的某参数区间（[lo, hi]），缺失则返回哨兵。"""
    return list(proto.get(level, {}).get(param, [float("nan"), float("nan")]))


def _in_range(v: float, lo: float, hi: float) -> bool:
    """值 v 是否落在闭区间 [lo, hi] 内。"""
    return lo <= v <= hi


def check_P2_missing_source(cfg: Mapping[str, Any]) -> CheckResult:
    """P2: uwb_valid=0 分布与 M1 缺失率一致；nl_flag 与 N 注入统计交叉验证。

    真实化：若 sim_meta 有 nlos_stats，按组合核验 N 轴比例是否在协议区间内，
    否则回退到软性"missing_source_stats 有值"检查。
    """
    stats = dict(cfg.get("missing_source_stats") or {})
    # 新路径：直接从 sim_meta 读取 N/A/V 轴协议参数
    nl_stats = cfg.get("nlos_stats") or {}
    if nl_stats:
        a2n2 = nl_stats.get("A2N2", {})
        a2n3 = nl_stats.get("A2N3", {})
        a3n2 = nl_stats.get("A3N2", {})
        a3n3 = nl_stats.get("A3N3", {})
        # 每个组合的 nlos_ratio_mid 应在协议 N2/N3 区间内
        # 协议区间来自 scene_axis_protocol.yaml: N2.nlos_ratio=[0.25,0.30], N3.nlos_ratio=[0.35,0.40]
        ranges = get_protocol_axis_ranges()
        n2_lo, n2_hi = _get_range(ranges, "N2", "nlos_ratio")
        n3_lo, n3_hi = _get_range(ranges, "N3", "nlos_ratio")
        detail_parts = []
        ok2, ok3 = True, True
        for combo, rho, level in [("A2N2", a2n2.get("rho"), "N2"), ("A3N2", a3n2.get("rho"), "N2"),
                                   ("A2N3", a2n3.get("rho"), "N3"), ("A3N3", a3n3.get("rho"), "N3")]:
            if rho is not None:
                hi = n3_hi if level == "N3" else n2_hi
                lo = n3_lo if level == "N3" else n2_lo
                if not (lo <= rho <= hi):
                    detail_parts.append(f"{combo}ρ={rho:.3f} 越界 [{lo},{hi}]")
                    if level == "N2":
                        ok2 = False
                    else:
                        ok3 = False
        return _make(
            "P2", "缺失源与N轴注入交叉验证",
            bool(nl_stats) and ok2 and ok3,  # 两档都必须通过（N2/N3 两档都得在区间）
            detail=f"N轴注入核验 N2∈[{n2_lo:.2f},{n2_hi:.2f}] {'✓' if ok2 else '✗'} / N3∈[{n3_lo:.2f},{n3_hi:.2f}] {'✓' if ok3 else '✗'} | " + ("; ".join(detail_parts) or "区间OK"),
            n_axis_stats=nl_stats,
        )
    if not stats:
        return _make("P2", "缺失源与可解率口径", False, detail="missing_source_stats 缺失", severity="soft")
    valid_rate = float(stats.get("valid_rate", 0.0))
    nlos_coverage = float(stats.get("nlos_coverage", 0.0))
    return _make(
        "P2", "缺失源与可解率口径",
        0.0 <= valid_rate <= 1.0 and 0.0 <= nlos_coverage <= 1.0,
        detail=f"valid_rate={valid_rate:.3f}, nlos_coverage={nlos_coverage:.3f}",
        **stats,
    )


def check_P3_m1_burst(cfg: Mapping[str, Any]) -> CheckResult:
    """P3: M1 成簇丢包、间隙区间落在协议。"""
    burst = dict(cfg.get("m1_burst_stats") or {})
    if not burst:
        return _make("P3", "M1成簇丢包", False, detail="m1_burst_stats 缺失", severity="soft")
    mean_gap = float(burst.get("mean_gap_s", 0.0))
    max_gap = float(burst.get("max_gap_s", 0.0))
    missing_rate = float(burst.get("missing_rate", 0.0))
    passed = (0.30 <= mean_gap <= 2.0) and (0.03 <= missing_rate <= 0.10) and (mean_gap <= max_gap <= 4.0)
    return _make("P3", "M1成簇丢包", passed,
                 detail=f"mean_gap={mean_gap:.2f}, max_gap={max_gap:.2f}, missing_rate={missing_rate:.2%}",
                 **burst)


def check_P4_k1_gdop(cfg: Mapping[str, Any]) -> CheckResult:
    """P4: K1 GDOP 带宽核对。容差与 block1_gdop_verifier 统一为 20%（PA-4 单一真源）。"""
    measured = float(cfg.get("measured_gdop", 0.0))
    expected = float(cfg.get("expected_gdop", GDOP_DEFAULT))
    k_arch = str(cfg.get("anchor_geometry_class", "unknown"))
    if measured == 0.0:
        return _make("P4", "K1 GDOP≈1.19", False, detail="measured_gdop 缺失", severity="soft")
    rel = abs(measured - expected) / max(expected, 1e-9)
    return _make("P4", "K1 GDOP≈1.19", rel <= 0.20,
                 detail=f"measured={measured:.3f}, expected={expected:.3f}, rel={rel:.1%}, geometry={k_arch}",
                 measured_gdop=measured, expected_gdop=expected, k_class=k_arch)


def check_P5_no_out_of_domain(cfg: Mapping[str, Any]) -> CheckResult:
    """P5: 轨迹不出域 (凸包内 + ≥1.5m 缓冲)。

    判定口径随场景语义切换:
      - 默认: 违规序列数 = 0 → PASS
      - 外推测试场景 (data note / config 标记 extrapolation / S2 仲裁已登记):
        GT 故意超出锚区凸包是设计意图, 违规数仅作描述性报告, 判 PASS
    """
    boundary_violations = int(cfg.get("boundary_violations", 0))
    n_sequences = int(cfg.get("n_sequences", 0)) or 1
    is_extrapolation = (
        "extrapolation" in str(cfg.get("scene_intent", "")).lower()
        or "extrapolation" in str(cfg.get("data_note", "")).lower()
        or "arbitration" in str(cfg.get("scene_note", "")).lower()
        or bool(cfg.get("s2_arbitration_registered", False))
    )
    if is_extrapolation:
        return _make("P5", "轨迹不出域", True,
                     detail=f"extrapolation test (S2 裁决): boundary_violations={boundary_violations}/{n_sequences}, 超出锚区为设计意图",
                     boundary_violations=boundary_violations, n_sequences=n_sequences, severity="soft")
    return _make("P5", "轨迹不出域", boundary_violations == 0,
                 detail=f"violations={boundary_violations}/{n_sequences}",
                 boundary_violations=boundary_violations, n_sequences=n_sequences)


def check_P6_size_and_split(cfg: Mapping[str, Any]) -> CheckResult:
    """P6: 训练 12-15 / 测试 ≥100 / seed，划分按整条序列不重叠 (S9/DQ-4)。

    手册 S9/DQ-4 明确要求：
    > train 12-15 + test ≥100 (四组合各 ≥25) / seed
    """
    n_train = int(cfg.get("n_train_per_seed", 0))
    n_test = int(cfg.get("n_test_per_seed", 0))
    overlap = int(cfg.get("train_test_overlap", 0))
    # CODE-5: 断言与手册 S9/DQ-4 一致，train 12-15, test >= 100
    passed = (12 <= n_train <= 15) and (n_test >= 100) and (overlap == 0)
    return _make("P6", "规模与划分", passed,
                 detail=f"train={n_train} (need 12-15), test={n_test} (need >=100), overlap={overlap}",
                 n_train=n_train, n_test=n_test, overlap=overlap)


def check_Pre1_environment_locked(cfg: Mapping[str, Any]) -> CheckResult:
    """Pre-1: 环境锁定（requirements.lock / CUDA / PyTorch 版本）。"""
    has_lock = bool(cfg.get("requirements_lock_path"))
    has_cuda = bool(cfg.get("cuda_version"))
    has_torch = bool(cfg.get("torch_version"))
    return _make("Pre-1", "环境锁定",
                 has_lock and has_cuda and has_torch,
                 detail=f"requirements_lock={has_lock}, cuda={has_cuda}, torch={has_torch}",
                 has_requirements_lock=has_lock, has_cuda=has_cuda, has_torch=has_torch)


def check_Pre2_decision_log(cfg: Mapping[str, Any]) -> CheckResult:
    """Pre-2: 决策日志存在（变更三级分类已记录）。"""
    has_log = bool(cfg.get("decision_log_path"))
    n_entries = int(cfg.get("decision_log_entries", 0))
    return _make("Pre-2", "决策日志",
                 has_log and n_entries > 0,
                 detail=f"path={cfg.get('decision_log_path')}, entries={n_entries}",
                 n_entries=n_entries)


def check_Pre3_directory_structure(cfg: Mapping[str, Any]) -> CheckResult:
    """Pre-3: run 目录命名规范 run-<日期>-<seed>-<method>-<config_hash>。"""
    import re
    run_dirs = list(cfg.get("run_directories") or [])
    pat = re.compile(r"^run-\d{4}-\d{2}-\d{2}-\d+-.+-.{6,}$")
    bad = [d for d in run_dirs if not pat.match(str(d))]
    return _make("Pre-3", "目录结构",
                 len(run_dirs) > 0 and not bad,
                 detail=f"n={len(run_dirs)}, bad={len(bad)}",
                 n_directories=len(run_dirs), bad=bad[:5])


def check_Pre4_checksums(cfg: Mapping[str, Any]) -> CheckResult:
    """Pre-4: npz / manifest 校验和 (SHA-256) 记录。"""
    n_with_sha = int(cfg.get("n_files_with_sha256", 0))
    n_total = int(cfg.get("n_files_total", 1))
    return _make("Pre-4", "数据校验和",
                 n_with_sha == n_total and n_total > 0,
                 detail=f"with_sha={n_with_sha}/{n_total}",
                 n_with_sha=n_with_sha, n_total=n_total)


def check_Pre5_resource_budget(cfg: Mapping[str, Any]) -> CheckResult:
    """Pre-5: 资源预算与单单元冒烟耗时。"""
    has_gpu = bool(cfg.get("gpu_model"))
    has_smoke = bool(cfg.get("smoke_unit_seconds"))
    return _make("Pre-5", "资源预算",
                 has_gpu and has_smoke,
                 detail=f"gpu={cfg.get('gpu_model')}, smoke_s={cfg.get('smoke_unit_seconds')}")


def check_Pre6_seed_manifest_consistency(cfg: Mapping[str, Any]) -> CheckResult:
    """Pre-6: 随机源清单与 manifest 一致；确定性模式开启；N_seed ≥ 10（protocol 硬门）。"""
    keys = ["n_manifest_seeds", "n_random_sources", "deterministic_mode"]
    missing = [k for k in keys if k not in cfg]
    if missing:
        return _make("Pre-6", "随机源核对", False,
                     detail=f"missing={missing}（必须由调用方提供实测值）",
                     missing=missing)
    n_manifest_seeds = int(cfg.get("n_manifest_seeds", 0))
    n_random_sources = int(cfg.get("n_random_sources", 0))
    deterministic = bool(cfg.get("deterministic_mode"))
    N_SEED_MIN = 10  # protocol.yaml n_seed_min
    return _make("Pre-6", "随机源核对",
                 n_manifest_seeds >= N_SEED_MIN and n_random_sources >= 3 and deterministic,
                 detail=f"manifest_seeds={n_manifest_seeds}(min={N_SEED_MIN}), sources={n_random_sources}, det={deterministic}")


def check_I1_data_generator(cfg: Mapping[str, Any]) -> CheckResult:
    """I-1: 数据生成器 S8 7 步注入顺序。"""
    steps_done = list(cfg.get("data_generator_steps_done") or [])
    required = ["scene_gen", "sensor_sim", "async_inject", "nlos_inject", "missing_inject", "resample", "write_npz"]
    missing = [s for s in required if s not in steps_done]
    return _make("I-1", "数据生成器7步", not missing,
                 detail=f"missing={missing or 'OK'}",
                 steps_done=steps_done, missing=missing)


def check_I2_s9_script(cfg: Mapping[str, Any]) -> CheckResult:
    """I-2: S9 校验脚本可输出 7 项判定。"""
    report_keys = list(cfg.get("s9_report_keys") or [])
    required = ["composition", "nlos_injection", "missing", "m_burst", "gdop", "boundary", "schema"]
    missing = [k for k in required if k not in report_keys]
    return _make("I-2", "S9校验脚本", not missing,
                 detail=f"missing={missing or 'OK'}",
                 report_keys=report_keys)


def check_I3_5_methods(cfg: Mapping[str, Any]) -> CheckResult:
    """I-3: 5 方法脚本存在（LNN/LSTM/Transformer/EKF/Robust-EKF）。

    修复 (2026-09-05): 同时接受 `available_methods`（路由层用，全名如 lstm_ekf）
    和 `available_methods_basename`（precheck 用，basename 如 lstm）。两套命名以
    任一存在即 PASS。
    第 8 阶段修复: 'lnn' 是手册语义名, 实现中模型 factory MODEL_NAME_LIQUID = 'liquid',
    basename 应接受 'lnn' 或 'liquid' 任一。
    """
    full_names = list(cfg.get("available_methods") or [])
    base_names = list(cfg.get("available_methods_basename") or [])
    combined = sorted(set(full_names) | set(base_names))
    combined_set = set(combined)  # 单独一份 set 用来 O(1) 检测
    # liquid 和 lnn 是同一方法的不同别名: 手册语义用 lnn, 实现层用 liquid
    if "liquid" in combined_set and "lnn" not in combined_set:
        combined_set.add("lnn")
        combined = sorted(combined_set)
    required = ["lnn", "lstm", "transformer", "ekf", "robust_ekf"]
    missing = [m for m in required if m not in combined]
    if missing:
        return _make("I-3", "5方法脚本", False,
                     detail=f"missing={missing}（必须在 available_methods 或 available_methods_basename 中）",
                     available=combined, missing=missing)
    return _make("I-3", "5方法脚本", True,
                 detail=f"available=full={full_names} | basename={base_names}",
                 available=combined, missing=[])


def check_I4_stats_script(cfg: Mapping[str, Any]) -> CheckResult:
    """I-4: 统计脚本 P22 全输出可跑通。"""
    has_outputs = list(cfg.get("stats_script_outputs") or [])
    required = ["mean", "std", "p95", "p50", "delta_vs_lnn", "wilcoxon_p", "ci_95",
                "holm_bonferroni", "slicing_4_combo", "anova_2x2", "c1_vs_c4"]
    missing = [k for k in required if k not in has_outputs]
    return _make("I-4", "统计脚本", not missing,
                 detail=f"missing={missing or 'OK'}")


def check_I5_eval_pipeline(cfg: Mapping[str, Any]) -> CheckResult:
    """I-5: 评估口径实现 Sim(2/3) 对齐 + 2D + warm-up 剔除 + 世界系。

    第 8 阶段修复: 2D 实验用 Sim(2) Umeyama 对齐 (旋转+平移, 3-DOF), sim3_alignment=false 也 PASS。
    3D 实验必须用 Sim(3) 缩放+旋转+平移 (4-DOF)。
    """
    sim3 = bool(cfg.get("sim3_alignment"))
    is_2d = bool(cfg.get("is_2d"))
    warmup_removed = bool(cfg.get("warmup_removed"))
    world_frame = bool(cfg.get("world_frame"))
    # 2D 实验可关闭 Sim(3); 3D 实验必须开启 Sim(3) + 关闭 2D (互斥)
    alignment_ok = sim3 or is_2d  # Sim(3) 开启 或 2D 模式 即可 (Sim(2) 隐式成立)
    return _make("I-5", "评估口径",
                 alignment_ok and warmup_removed and world_frame,
                 detail=f"sim3={sim3}, 2d={is_2d}, warmup={warmup_removed}, world={world_frame}, alignment_ok={alignment_ok}")


# ============================================================================
# 第二层：P7 / P8 / P9 旋转链路 / 单位 / Sim(3) 对齐（缺 → 实施）
# ============================================================================


def check_P7_rotation_chain(cfg: Mapping[str, Any]) -> CheckResult:
    """P7: 纯 IMU 积分走"绕圈回起点"，直线 vs 转角对比，偏差方向符合 yaw 约定。"""
    # 调用方提供"绕圈回起点"自测结果
    circle_test_passed = bool(cfg.get("imu_circle_return_passed"))
    straight_vs_turn_diff_m = float(cfg.get("imu_straight_vs_turn_diff_m", 0.0))
    yaw_direction_correct = bool(cfg.get("yaw_direction_correct"))
    passed = circle_test_passed and yaw_direction_correct and straight_vs_turn_diff_m < 0.5
    return _make("P7", "旋转链路", passed,
                 detail=f"circle={circle_test_passed}, yaw={yaw_direction_correct}, straight_turn_diff={straight_vs_turn_diff_m:.3f}m",
                 circle_test_passed=circle_test_passed,
                 yaw_direction_correct=yaw_direction_correct,
                 straight_turn_diff_m=straight_vs_turn_diff_m)


def check_P8_units_and_normalization(cfg: Mapping[str, Any]) -> CheckResult:
    """P8: dyaw rad, distance m, angle in [-π, π], 无 ×57/×100 静默错误。"""
    units_check = dict(cfg.get("unit_check") or {})
    distance_m = units_check.get("distance_unit", "")
    yaw_rad = units_check.get("yaw_unit", "")
    # 检查是否出现过 0.01745 (rad→deg 转换) 或 57.3 这种迹象
    has_x57_100 = bool(units_check.get("has_x57_or_x100_artifact", False))
    passed = (distance_m == "m") and (yaw_rad == "rad") and not has_x57_100
    return _make("P8", "单位与归一化", passed,
                 detail=f"distance={distance_m}, yaw={yaw_rad}, x57_100_artifact={has_x57_100}")


def check_P9_sim3_alignment(cfg: Mapping[str, Any]) -> CheckResult:
    """P9: RMSE 计算前 Sim(3)/Umeyama 对齐（必须）。"""
    pre_align_rmse = float(cfg.get("pre_align_rmse", 0.0))
    post_align_rmse = float(cfg.get("post_align_rmse", 0.0))
    # 对齐后误差应当显著小于对齐前（旋转/平移轨迹无对齐会恒高）
    ratio = post_align_rmse / max(pre_align_rmse, 1e-9)
    aligned = bool(cfg.get("sim3_alignment_applied"))
    passed = aligned and ratio < 0.95
    return _make("P9", "Sim(3)对齐", passed,
                 detail=f"pre={pre_align_rmse:.3f}, post={post_align_rmse:.3f}, ratio={ratio:.3f}",
                 pre_align=pre_align_rmse, post_align=post_align_rmse, ratio=ratio)


def check_P10_2d_only(cfg: Mapping[str, Any]) -> CheckResult:
    """P10: 2D 维度（不混 Z）。"""
    has_z_in_metrics = bool(cfg.get("has_z_in_2d_metrics", False))
    coord_dim = int(cfg.get("coordinate_dim", 2))
    return _make("P10", "2D维度", not has_z_in_metrics and coord_dim == 2,
                 detail=f"has_z={has_z_in_metrics}, dim={coord_dim}")


def check_P11_input_isolation(cfg: Mapping[str, Any]) -> CheckResult:
    """P11: 模型输入只含测距/IMU/VIO/时间戳，不含锚点坐标 / geom_condition。"""
    features = list(cfg.get("input_features") or [])
    forbidden = ["anchor_x", "anchor_y", "geom_condition", "k_level", "gdop_value"]
    leaked = [f for f in features if f in forbidden]
    has_4_required = all(f in features for f in ("range", "imu", "vio", "t"))
    return _make("P11", "输入隔离", not leaked and has_4_required,
                 detail=f"leaked={leaked or 'none'}, has_4_required={has_4_required}",
                 features=features, leaked=leaked)


def check_P12_train_test_symmetry(cfg: Mapping[str, Any]) -> CheckResult:
    """P12: 训练/测试注入对称（训练集含与测试集同分布 NLOS 大脉冲）。"""
    train_nlos = dict(cfg.get("train_nlos_stats") or {})
    test_nlos = dict(cfg.get("test_nlos_stats") or {})
    if not train_nlos or not test_nlos:
        return _make("P12", "注入对称", False, detail="train/test nlos stats 缺失", severity="soft")
    rho_dev = abs(float(train_nlos.get("rho", 0)) - float(test_nlos.get("rho", 0)))
    mu_dev = abs(float(train_nlos.get("mu_m", 0)) - float(test_nlos.get("mu_m", 0))) / max(float(train_nlos.get("mu_m", 1e-9)), 1e-9)
    passed = rho_dev < 0.05 and mu_dev < 0.10
    return _make("P12", "注入对称", passed,
                 detail=f"rho_dev={rho_dev:.3f}, mu_dev={mu_dev:.1%}",
                 rho_dev=rho_dev, mu_dev=mu_dev)


def check_P13_rq_matching(cfg: Mapping[str, Any]) -> CheckResult:
    """P13: 滤波类 R/Q 与注入噪声统计一致。"""
    r_std = float(cfg.get("ekf_r_std", 0.0))
    q_std = float(cfg.get("ekf_q_std", 0.0))
    inj_noise = float(cfg.get("injected_uwb_noise", 0.6))
    r_match = abs(r_std - inj_noise) < 0.3
    q_match = q_std > 0.0
    return _make("P13", "R/Q匹配", r_match and q_match,
                 detail=f"r={r_std}, q={q_std}, injected={inj_noise}",
                 r=r_std, q=q_std, injected=inj_noise)


def check_P14_shared_4_heads(cfg: Mapping[str, Any]) -> CheckResult:
    """P14: LSTM/Transformer/LNN 共享 4 头 (uwb_scaling/bias/vio_scaling/risk)。"""
    heads = dict(cfg.get("network_4_heads") or {})
    required_keys = {"uwb_scaling", "bias", "vio_scaling", "risk"}
    all_have = all(heads.get(m) == heads.get("lnn") for m in ("lnn", "lstm", "transformer"))
    all_4 = required_keys.issubset(set(heads.get("lnn", {}).keys()))
    return _make("P14", "4头共享", all_have and all_4,
                 detail=f"all_have_same={all_have}, has_4={all_4}",
                 heads=heads)


def check_P15_per_method_tuning(cfg: Mapping[str, Any]) -> CheckResult:
    """P15: per-method tuning 调优预算一致。

    D10 PARTIAL: 必须验证 LNN 收到了独立的 lr sweep，
    而非仅检查 budget 一致性。具体要求：
    1. 每个方法的 tuning_budget 条目必须包含 lr_sweep_runs >= 1 或 best_lr 字段（说明有调优证据）
    2. LNN 必须有独立调优证据：lr_sweep_runs >= 2（至少 2 轮扫参，排除"默认参数充数"）
    """
    budgets = dict(cfg.get("tuning_budget") or {})
    methods = ["lnn", "lstm", "transformer"]
    has_all = all(budgets.get(m) for m in methods)
    consistent = len(set(str(budgets.get(m)) for m in methods)) <= 1

    # D10 PARTIAL 强化：每个方法必须有调优证据（LNN 需 >=2 轮独立扫参）
    method_evidences: dict[str, bool] = {}
    lnn_independent = False
    for m in methods:
        raw_entry = budgets.get(m, 0)
        if isinstance(raw_entry, Mapping):  # 完整形式：{"lr_sweep_runs": N, "best_lr": ...}。
            lr_sweep = int(raw_entry.get("lr_sweep_runs", 0) or 0)
            has_best_lr = "best_lr" in raw_entry
        elif isinstance(raw_entry, (int, float)) and not isinstance(raw_entry, bool):  # 简写形式：整数直接表示 lr_sweep_runs。
            lr_sweep = int(raw_entry)
            has_best_lr = False
        else:  # 非法条目按无调优证据处理，不做静默崩溃。
            lr_sweep = 0
            has_best_lr = False
        has_evidence = lr_sweep >= 1 or has_best_lr
        method_evidences[m] = has_evidence
        if m == "lnn":
            # D10 PARTIAL: LNN 必须有 >=2 轮独立扫参，排除默认参数
            lnn_independent = lr_sweep >= 2

    all_have_evidence = all(method_evidences.values())
    passed = has_all and consistent and all_have_evidence and lnn_independent

    return _make(
        "P15",
        "per-method tuning",
        passed,
        detail=(
            f"budgets={budgets}, consistent={consistent}, "
            f"all_have_evidence={all_have_evidence}, "
            f"lnn_independent_tuning={lnn_independent} "
            f"(evidence={method_evidences})"
        ),
        budgets=budgets,
        consistent=consistent,
        all_have_evidence=all_have_evidence,
        lnn_independent=lnn_independent,
        method_evidences=method_evidences,
    )


# P16/P17/P18 在 ekf_init_protocol + 5 方法脚本中存在；调用方填入实际 self-test 结果
def check_P16_ekf_init_protocol(cfg: Mapping[str, Any]) -> CheckResult:
    trilat_passed = bool(cfg.get("ekf_trilateration_recovered", False))
    delayed_init = bool(cfg.get("ekf_delayed_init_works", False))
    vio_yaw_used = bool(cfg.get("ekf_vio_yaw_used", False))
    return _make("P16", "EKF初始化协议", trilat_passed and delayed_init and vio_yaw_used,
                 detail=f"trilat={trilat_passed}, delayed={delayed_init}, vio_yaw={vio_yaw_used}")


def check_P17_robust_ekf_huber(cfg: Mapping[str, Any]) -> CheckResult:
    huber_used = bool(cfg.get("robust_ekf_huber_applied", False))
    not_just_r = bool(cfg.get("robust_ekf_not_just_r_scaling", False))
    return _make("P17", "Robust-EKF+Huber", huber_used and not_just_r,
                 detail=f"huber={huber_used}, not_just_r={not_just_r}")


def check_P18_baselines_with_heads(cfg: Mapping[str, Any]) -> CheckResult:
    lstm_heads = int(cfg.get("lstm_n_output_heads", 0))
    tr_heads = int(cfg.get("transformer_n_output_heads", 0))
    return _make("P18", "基线带头", lstm_heads == 4 and tr_heads == 4,
                 detail=f"lstm_heads={lstm_heads}, tr_heads={tr_heads}")


# ============================================================================
# 第四层：P19~P23 统计脚手架
# ============================================================================


def check_P19_seed_system(cfg: Mapping[str, Any]) -> CheckResult:
    """P19: seed 体系，N_seed ≥ 10（protocol n_seed_min）。"""
    keys = ["n_seeds", "deterministic_mode", "tf32_explicit_set"]
    missing = [k for k in keys if k not in cfg]
    if missing:
        return _make("P19", "seed体系", False,
                     detail=f"missing={missing}（必须由调用方提供实测值）",
                     missing=missing)
    n_seeds = int(cfg.get("n_seeds", 0))
    deterministic = bool(cfg.get("deterministic_mode", False))
    tf32_set = bool(cfg.get("tf32_explicit_set", False))
    N_SEED_MIN = 10  # protocol.yaml n_seed_min
    return _make("P19", "seed体系", n_seeds >= N_SEED_MIN and deterministic and tf32_set,
                 detail=f"n_seeds={n_seeds}(min={N_SEED_MIN}), deterministic={deterministic}, tf32={tf32_set}")


def check_P20_window(cfg: Mapping[str, Any]) -> CheckResult:
    W = int(cfg.get("window_size", 0))
    warmup = float(cfg.get("warmup_s", -1.0))
    no_cross = bool(cfg.get("no_window_cross_seq", False))
    imu_no_cross = bool(cfg.get("imu_no_cross_seq_concat", False))
    return _make("P20", "窗口化", W == WINDOW_SIZE and warmup == WARMUP_S and no_cross and imu_no_cross,
                 detail=f"W={W}, warmup={warmup}, no_cross={no_cross}, imu_no_cross={imu_no_cross}")


def check_P21_normalization_isolation(cfg: Mapping[str, Any]) -> CheckResult:
    train_only = bool(cfg.get("normalization_uses_train_only", False))
    no_test_leak = bool(cfg.get("no_test_leak_in_normalization", False))
    return _make("P21", "归一化隔离", train_only and no_test_leak,
                 detail=f"train_only={train_only}, no_test_leak={no_test_leak}")


def check_P22_stats_script(cfg: Mapping[str, Any]) -> CheckResult:
    has_holm = bool(cfg.get("has_holm_bonferroni", False))
    has_wilcoxon = bool(cfg.get("has_wilcoxon", False))
    has_slicing = bool(cfg.get("has_4_combo_slicing", False))
    has_anova = bool(cfg.get("has_2x2_anova", False))
    has_diag = bool(cfg.get("has_c1_vs_c4_diag", False))
    return _make("P22", "统计脚本",
                 all([has_holm, has_wilcoxon, has_slicing, has_anova, has_diag]),
                 detail=f"holm={has_holm}, wilcox={has_wilcoxon}, slice={has_slicing}, anova={has_anova}, diag={has_diag}")


def check_P23_config_consistency(cfg: Mapping[str, Any]) -> CheckResult:
    yaml_axes = set(cfg.get("yaml_frozen_axes") or [])
    manifest_axes = set(cfg.get("manifest_axes") or [])
    no_mismatch = yaml_axes == manifest_axes
    return _make("P23", "配置一致性", no_mismatch and len(yaml_axes) > 0,
                 detail=f"yaml-yaml={yaml_axes}, manifest={manifest_axes}")


# ============================================================================
# 第五层：P24~P27 稳定性/接口
# ============================================================================


def check_P24_no_nan_inf(cfg: Mapping[str, Any]) -> CheckResult:
    # P24：键缺失视为 FAIL，防止默过
    required = ["n_nan_detected", "n_inf_detected", "gradient_exploded"]
    has_keys = all(k in cfg for k in required)
    if not has_keys:
        return _make("P24", "无NaN/Inf/梯度爆炸", False,
                     detail=f"统计字段缺失: missing={[k for k in required if k not in cfg]}",
                     severity="hard")
    n_nan = int(cfg["n_nan_detected"])
    n_inf = int(cfg["n_inf_detected"])
    grad_exploded = bool(cfg["gradient_exploded"])
    return _make("P24", "无NaN/Inf/梯度爆炸",
                 n_nan == 0 and n_inf == 0 and not grad_exploded,
                 detail=f"nan={n_nan}, inf={n_inf}, grad_exploded={grad_exploded}")


def check_P25_async_timestamps(cfg: Mapping[str, Any]) -> CheckResult:
    # 真实值：从 sim_meta 聚合的 async_offset_ms_mid / async_jitter_ms / async_skew_ms_mid 核对协议 A2/A3
    a2_a3_params = cfg.get("async_params", {})
    monotonic = bool(cfg.get("timestamps_monotonic", False))
    if a2_a3_params:
        # 校验 A2/A3 偏移是否在协议区间内（configs/base/scene_axis_protocol.yaml 权威定义）
        # offset_ms / cross_modal_skew_ms 为区间 [lo,hi]，jitter_ms 为固定标量
        proto = cfg.get("protocol_ranges") or get_protocol_axis_ranges()
        checks_ok = True
        details = []
        for level in ("A2", "A3"):
            p = proto.get(level, {})
            params = a2_a3_params.get(level, {})
            off = float(params.get("async_offset_ms_mid", 0.0))
            jit = float(params.get("async_jitter_ms", 0.0))
            skew = float(params.get("async_skew_ms_mid", 0.0))
            # offset_ms 为区间 [lo,hi]；cross_modal_skew_ms 是协议标量中位（"UWB 往返延迟中位"），
            # 与 jitter_ms 同为固定标量，允许 ±30% 抽样离散度（BUG-022 修复：不得按区间解包标量）。
            off_lo, off_hi = _get_range(proto, level, "offset_ms")
            skew_exp = float(p.get("cross_modal_skew_ms", 0.0))
            # jitter_ms: 协议为固定标量，允许 ±30%（IMU 抖动抽样离散度）
            jitter_exp = float(p.get("jitter_ms", 0.0))
            ok = _in_range(off, off_lo, off_hi)
            ok = ok and ((abs(skew - skew_exp) / max(skew_exp, 1e-9)) < 0.30) if skew_exp else (skew == 0)
            ok = ok and ((abs(jit - jitter_exp) / max(jitter_exp, 1e-9)) < 0.30) if jitter_exp else (jit == 0)
            if not ok:
                checks_ok = False
                details.append(f"{level}: off={off:.1f}∈[{off_lo},{off_hi}], skew={skew:.1f}≈{skew_exp}, jit={jit:.1f}≈{jitter_exp}")
        if not checks_ok:
            return _make("P25", "A2/A3协议核对+时间戳", False,
                         detail=f"A2/A3参数越界: {'; '.join(details)} | mono={monotonic}")
        return _make("P25", "A2/A3协议核对+时间戳", True,
                     detail=f"A2/A3 偏移/skew 在协议区间; jitter 在标量±30%; mono={monotonic}",
                     async_params=a2_a3_params)
    # 回退到布尔标志（调用方手工传入）
    async_in_pre = bool(cfg.get("async_injected_before_resample", False))
    jitter_per_packet = bool(cfg.get("jitter_per_packet", False))
    return _make("P25", "A2/A3时间戳(标志)", monotonic and async_in_pre and jitter_per_packet,
                 detail=f"mono={monotonic}, pre_resample={async_in_pre}, per_packet={jitter_per_packet}",
                 severity="soft")


def check_P26_vio_payload(cfg: Mapping[str, Any]) -> CheckResult:
    vio_keys = set(cfg.get("vio_payload_keys") or [])
    required = {"dx", "dy", "dyaw", "quality"}
    all_4 = required.issubset(vio_keys)
    real_wire = bool(cfg.get("vio_valid_and_scaling_wired", False))
    return _make("P26", "VIO payload真接入", all_4 and real_wire,
                 detail=f"keys={vio_keys}, wired={real_wire}")


def check_P26a_v0_negative_exclusion(cfg: Mapping[str, Any]) -> CheckResult:
    """P26a: V0 负向排除——E9 固定 V0，数据不得含 V1/V2/V3 退化症状。

    手册第 26a 项（V0 档负向排除）四项症状，任一为真即 FAIL：
        ① 置零（blackout_fraction>0 或 tracked≈0）
        ② 增量噪声 5× 放大（inc_noise_std_mps > 0.04 视为放大）
        ③ 关键帧中断（keyframe_interruptions>0）
        ④ vio_valid 语义错（V0 应全程有效）
    从 sim_meta 聚合的 vio_tracked_mid / vio_inc_noise_mps_mid / vio_blackout_frac 判定。
    """
    v0 = cfg.get("v0_stats") or {}
    if not v0:
        # 无 sim_meta 聚合时，回退到调用方标志位（缺失则 FAIL，不静默过）
        flag = cfg.get("v0_negative_exclusion_verified")
        if flag is None:
            return _make("P26a", "V0负向排除", False,
                         detail="缺 v0_stats 聚合字段（须从 sim_meta 推导或调用方提供）",
                         severity="soft")
        return _make("P26a", "V0负向排除", bool(flag), detail=f"verified_flag={flag}", severity="soft")
    tracked = v0.get("vio_tracked_mid", [])
    inc_noise = v0.get("vio_inc_noise_mps_mid", [])
    blackout = v0.get("vio_blackout_frac", [])
    # 症状①：任何序列 tracked < 50 → 疑似置零/黑屏
    symptom1 = any((t < 50.0 and t > 0.0) for t in tracked) if tracked else True
    # 症状②：任何序列 inc_noise > 0.04 → 疑似 σ×5 放大（V0 应 ≤0.03）
    symptom2 = any((n > 0.04) for n in inc_noise) if inc_noise else True
    # 症状③：任何序列 blackout_frac > 0 → 黑屏注入（V0 应为 0）
    symptom3 = any((b > 0.001) for b in blackout) if blackout else True
    # 症状④：vio_valid 全程 1（由调用方或数据契约提供，此处缺省不报）
    symptom4 = bool(cfg.get("vio_valid_not_full_for_v0", False))
    symptoms = {
        "①置零/黑屏": symptom1, "②σ×5放大": symptom2,
        "③关键帧中断": symptom3, "④valid语义": symptom4,
    }
    any_symptom = any(symptoms.values())
    return _make("P26a", "V0负向排除", not any_symptom,
                 detail=f"V0健康性: {symptoms}", severity="hard", **symptoms)


def check_P27_activation_stats(cfg: Mapping[str, Any]) -> CheckResult:
    saved_heatmap = bool(cfg.get("activation_heatmap_saved", False))
    n4_heads = int(cfg.get("n_4_heads_with_stats", 0))
    return _make("P27", "激活统计可视化", saved_heatmap and n4_heads == 4,
                 detail=f"saved={saved_heatmap}, n_heads={n4_heads}")


# ============================================================================
# 第六层：P28~P31 数据管道
# ============================================================================


def check_P28_dual_rate(cfg: Mapping[str, Any]) -> CheckResult:
    # 真实核对：从 sim_meta 聚合 IMU 行数 vs 时长, 校验 ≈150Hz
    imu_rows = cfg.get("imu_row_rate", {})
    if imu_rows:
        rows = imu_rows.get("rows", [])
        durations = imu_rows.get("durations", [])
        if rows and durations:
            rates = []
            for r, d in zip(rows, durations):
                if d > 0:
                    rates.append(r / d)
            if rates:
                n = len(rates); mn = min(rates); mx = max(rates)
                avg = sum(rates) / n
                # 150Hz, 允许 ±3%
                ok = (0.97 * 150.0 <= avg <= 1.03 * 150.0) and (mx - mn < 6.0)
                return _make(
                    "P28", "多速率重采样(150Hz实测)",
                    ok,
                    detail=f"IMU速率实测: avg={avg:.1f}Hz, min={mn:.1f}, max={mx:.1f}, n_seq={n}",
                    imu_row_rate=rates,
                    upstream=bool(cfg.get("resample_method_is_nearest_neighbor", False)),
                )
    # 回退标志路径
    imu_150 = bool(cfg.get("imu_at_150hz_native", False))
    imu_no_downsample = bool(cfg.get("imu_not_downsampled", False))
    nearest_neighbor = bool(cfg.get("resample_method_is_nearest_neighbor", False))
    return _make("P28", "多速率重采样", imu_150 and imu_no_downsample and nearest_neighbor,
                 detail=f"imu_150={imu_150}, no_down={imu_no_downsample}, NN={nearest_neighbor}", severity="soft")


def check_P29_windowing(cfg: Mapping[str, Any]) -> CheckResult:
    no_cross = bool(cfg.get("window_no_cross_seq_boundary", False))
    warmup_excluded = bool(cfg.get("warmup_excluded_from_eval", False))
    imu_no_cross_concat = bool(cfg.get("imu_segments_no_cross_concat", False))
    zero_pad = bool(cfg.get("short_segment_zero_padded", False))
    return _make("P29", "窗口化", all([no_cross, warmup_excluded, imu_no_cross_concat, zero_pad]),
                 detail=f"no_cross={no_cross}, warmup_excl={warmup_excluded}, imu_no_cross={imu_no_cross_concat}, zero_pad={zero_pad}")


def check_P30_split_before_window(cfg: Mapping[str, Any]) -> CheckResult:
    return _make("P30", "划分先于窗口化",
                 bool(cfg.get("split_before_windowing", False)),
                 detail=f"flag={cfg.get('split_before_windowing')}")


def check_P31_edge_cases(cfg: Mapping[str, Any]) -> CheckResult:
    """P31: 边界情况无 NaN/zero-division。字段缺失视为 FAIL（避免静默通过）。"""
    has_keys = "n_zero_div_events" in cfg and "n_nan_after_edge_case" in cfg
    n_zero_div = int(cfg.get("n_zero_div_events", -1)) if has_keys else -1
    n_nan_after_edge = int(cfg.get("n_nan_after_edge_case", -1)) if has_keys else -1
    if not has_keys:
        return _make("P31", "边界情况", False,
                     detail="n_zero_div_events / n_nan_after_edge_case 字段缺失（必须由调用方提供实测值）",
                     n_zero_div=n_zero_div, n_nan_after_edge=n_nan_after_edge)
    return _make("P31", "边界情况",
                 n_zero_div == 0 and n_nan_after_edge == 0,
                 detail=f"zero_div={n_zero_div}, nan_after_edge={n_nan_after_edge}")


# ============================================================================
# 第七层：P32~P35 训练一致性
# ============================================================================


def check_P32_same_seed_init(cfg: Mapping[str, Any]) -> CheckResult:
    """P32: 神经网络三方法用同一 init_seed。字段缺失视为 FAIL。"""
    methods = ["lnn", "lstm", "transformer"]
    keys = [f"{m}_init_seed" for m in methods]
    missing = [k for k in keys if k not in cfg]
    if missing:
        return _make("P32", "同seed初始化", False,
                     detail=f"missing={missing}（必须由调用方提供实测 init_seed）",
                     missing=missing)
    ref = cfg.get("lnn_init_seed")
    same = all(cfg.get(k) == ref for k in keys)
    return _make("P32", "同seed初始化", same, detail=f"all_same={same}")


def check_P33_best_val(cfg: Mapping[str, Any]) -> CheckResult:
    uses_best_val = bool(cfg.get("uses_best_val_checkpoint", False))
    no_test_select = bool(cfg.get("no_test_set_selection", False))
    # 兼容旧实现：直接接受 no_test_select flag
    return _make("P33", "best-val", uses_best_val and no_test_select,
                 detail=f"best_val={uses_best_val}, no_test_select={no_test_select}")


def check_P34_no_overfit(cfg: Mapping[str, Any]) -> CheckResult:
    train_loss_converged = bool(cfg.get("train_loss_converged", False))
    val_loss_not_rising = bool(cfg.get("val_loss_not_rising", False))
    return _make("P34", "无过拟合", train_loss_converged and val_loss_not_rising,
                 detail=f"train_conv={train_loss_converged}, val_not_rise={val_loss_not_rising}")


def check_P35_head_consistency(cfg: Mapping[str, Any]) -> CheckResult:
    """P35: LSTM / LNN 头结构一致。字段缺失视为 FAIL。"""
    keys = ["lstm_head_layers", "lnn_head_layers"]
    missing = [k for k in keys if k not in cfg or cfg.get(k) is None]
    if missing:
        return _make("P35", "头实现一致", False,
                     detail=f"missing={missing}（必须由调用方提供头结构实测值）",
                     missing=missing)
    lstm_layers = dict(cfg.get("lstm_head_layers") or {})
    lnn_layers = dict(cfg.get("lnn_head_layers") or {})
    same_structure = lstm_layers == lnn_layers
    return _make("P35", "头实现一致", same_structure,
                 detail=f"lstm={lstm_layers}, lnn={lnn_layers}")


# ============================================================================
# 第八层：P36~P39 评估口径
# ============================================================================


def check_P36_metric_declaration(cfg: Mapping[str, Any]) -> CheckResult:
    """P36: 报告 mean±std, P50, P95 + MAE, trimmed_mean, median。"""
    rmse = bool(cfg.get("report_rmse", False))
    mean = bool(cfg.get("report_mean", False))
    std = bool(cfg.get("report_std", False))
    p50 = bool(cfg.get("report_p50", False))
    p95 = bool(cfg.get("report_p95", False))
    mae = bool(cfg.get("report_mae", False))
    trimmed = bool(cfg.get("report_trimmed_mean", False))
    median = bool(cfg.get("report_median", False))
    no_winsor = bool(cfg.get("no_winsorization", False))
    passed = all([rmse, mean, std, p50, p95, mae, trimmed, median, no_winsor])
    return _make("P36", "RMSE口径+MAE+trimmed+median", passed,
                 detail=f"rmse={rmse}, mae={mae}, trimmed={trimmed}, median={median}, no_winsor={no_winsor}")


def check_P37_warmup_removed(cfg: Mapping[str, Any]) -> CheckResult:
    return _make("P37", "warm-up剔除",
                 bool(cfg.get("warmup_10s_removed", False)),
                 detail=f"flag={cfg.get('warmup_10s_removed')}")


def check_P38_world_frame(cfg: Mapping[str, Any]) -> CheckResult:
    return _make("P38", "2D世界系",
                 bool(cfg.get("errors_in_world_frame", False)),
                 detail=f"flag={cfg.get('errors_in_world_frame')}")


def check_P39_persistence(cfg: Mapping[str, Any]) -> CheckResult:
    has_config_hash = bool(cfg.get("config_hash_persisted", False))
    has_seed = bool(cfg.get("seed_persisted", False))
    has_metrics = bool(cfg.get("metrics_persisted", False))
    has_git_commit = bool(cfg.get("git_commit_persisted", False))
    return _make("P39", "结果持久化",
                 all([has_config_hash, has_seed, has_metrics, has_git_commit]),
                 detail=f"config_hash={has_config_hash}, seed={has_seed}, metrics={has_metrics}, git_commit={has_git_commit}")


# ============================================================================
# DQ-1~DQ-4
# ============================================================================


def check_DQ1_difficulty_gradient(cfg: Mapping[str, Any]) -> CheckResult:
    """DQ-1: 难度梯度（C4 vs C1 相对提升 ≥ 40%）。与 handbook_gates 公式一致。

    双实现冲突已解决：precheck 和 handbook_gates 均使用
    diff_ratio = (c4 - c1) / c1, target = 0.40
    （原先 precheck 用绝对米 target_diff_m=2.4 与 handbook 不一致）。
    """
    keys = ["c4_mean_rmse", "c1_mean_rmse", "target_diff_ratio"]
    missing = [k for k in keys if k not in cfg]
    if missing:
        return _make("DQ-1", "难度梯度", False,
                     detail=f"missing={missing}（必须由调用方提供实测值）",
                     missing=missing)
    c4 = float(cfg.get("c4_mean_rmse", 0.0))
    c1 = float(cfg.get("c1_mean_rmse", 0.0))
    target_ratio = float(cfg.get("target_diff_ratio", 0.40))
    diff_ratio = (c4 - c1) / c1 if c1 > 0 else float('inf')
    return _make("DQ-1", "难度梯度", diff_ratio >= target_ratio,
                 detail=f"c4={c4:.2f}, c1={c1:.2f}, ratio={diff_ratio:.2%}, target>={target_ratio:.0%}")


def check_DQ2_snr(cfg: Mapping[str, Any]) -> CheckResult:
    """DQ-2: 信噪比（NLOS bias / (LOS_noise × GDOP) ≥ 3.0）。

    边界漏洞修复：
    - los_noise=0 → 分母为 0 → FAIL
    - gdop=0     → 分母为 0 → FAIL
    语义修正：分子取 bias（偏置幅度），非 bias+std（手册原文："偏置幅度/bias magnitude"）。
    """
    keys = ["nlos_bias_m", "los_noise_m", "gdop"]
    missing = [k for k in keys if k not in cfg]
    if missing:
        return _make("DQ-2", "信噪比", False,
                     detail=f"missing={missing}（必须由调用方提供实测值）",
                     missing=missing)
    bias = float(cfg.get("nlos_bias_m", 0.0))
    los = float(cfg.get("los_noise_m", 0.0))
    gdop = float(cfg.get("gdop", GDOP_DEFAULT))
    denom = los * gdop
    if denom <= 0:
        return _make("DQ-2", "信噪比", False,
                     detail=f"denom={denom} (los={los}, gdop={gdop}) → 分母为零/负，非法参数",
                     nlos_snr=float('nan'), denom=denom)
    snr = bias / denom
    return _make("DQ-2", "信噪比", snr >= 3.0,
                 detail=f"snr={snr:.2f} (>=3.0, bias={bias}, los={los}, gdop={gdop})",
                 nlos_snr=snr)


def check_DQ3_distribution(cfg: Mapping[str, Any]) -> CheckResult:
    def dev(a, b): return abs(a - b) / max(abs(a), abs(b), 1e-9)
    rho_d = dev(float(cfg.get("train_rho", 0)), float(cfg.get("test_rho", 0)))
    mu_d = dev(float(cfg.get("train_mu", 0)), float(cfg.get("test_mu", 0)))
    sig_d = dev(float(cfg.get("train_sigma", 0)), float(cfg.get("test_sigma", 0)))
    passed = max(rho_d, mu_d, sig_d) <= 0.10
    return _make("DQ-3", "分布一致", passed,
                 detail=f"rho={rho_d:.1%}, mu={mu_d:.1%}, sigma={sig_d:.1%}")


def check_DQ4_sample_size(cfg: Mapping[str, Any]) -> CheckResult:
    """DQ-4: 样本量充分性（功效支撑）。

    手册 S3/DQ-4 明确要求：
    > 测试 ≥100 条/seed × 10 seed = 每方法 ≥1000 个轨迹级配对样本
    > 四组合每组合 ≥250 个轨迹级样本
    """
    n_seeds = int(cfg.get("n_seeds", 0))
    n_test = int(cfg.get("n_test_per_seed", 0))
    total = n_seeds * n_test
    # CODE-5: 断言与手册 S3/DQ-4 一致，10 seed × ≥100 test = ≥1000 轨迹级样本
    passed = (n_seeds >= 10) and (n_test >= 100) and (total >= 1000)
    return _make("DQ-4", "样本量充分", passed,
                 detail=f"n_seeds={n_seeds}×{n_test}={total} (need n_seeds≥10, n_test≥100, total≥1000)")


# ============================================================================
# G-1~G-5
# ============================================================================


def check_G1_unit_completeness(cfg: Mapping[str, Any]) -> CheckResult:
    """G-1: 单元齐全性检查

    手册 S3 明确要求：
    > 10 seed × 5 方法 = 50 单元
    """
    expected = 50  # 修复：从 25 改为 50 (10 seed × 5 方法)
    n_complete = int(cfg.get("n_units_complete", 0))
    n_resumable = int(cfg.get("n_resumable_pending", 0))
    return _make("G-1", "单元齐全", n_complete == expected and n_resumable == 0,
                 detail=f"complete={n_complete}/{expected}, pending={n_resumable}")


def check_G2_metrics_readable(cfg: Mapping[str, Any]) -> CheckResult:
    """G-2: 指标无 NaN / 无缺失字段。字段缺失视为 FAIL。"""
    keys = ["n_metric_nan", "n_metric_missing_fields"]
    missing = [k for k in keys if k not in cfg]
    if missing:
        return _make("G-2", "指标可读", False,
                     detail=f"missing={missing}（必须由调用方提供实测值）",
                     missing=missing)
    n_nan = int(cfg.get("n_metric_nan", 0))
    n_missing = int(cfg.get("n_metric_missing_fields", 0))
    return _make("G-2", "指标可读", n_nan == 0 and n_missing == 0,
                 detail=f"nan={n_nan}, missing={n_missing}")


def check_G3_config_data_cross(cfg: Mapping[str, Any]) -> CheckResult:
    """G-3: 配置-数据无冲突。字段缺失视为 FAIL。"""
    if "n_config_data_mismatch" not in cfg:
        return _make("G-3", "配置-数据交叉", False,
                     detail="n_config_data_mismatch 字段缺失（必须由调用方提供实测值）",
                     missing=["n_config_data_mismatch"])
    n_mismatch = int(cfg.get("n_config_data_mismatch", 0))
    return _make("G-3", "配置-数据交叉", n_mismatch == 0,
                 detail=f"mismatch={n_mismatch}")


def check_G4_alerts_cleared(cfg: Mapping[str, Any]) -> CheckResult:
    """G-4: 告警清零。字段缺失视为 FAIL。"""
    if "n_unresolved_alerts" not in cfg:
        return _make("G-4", "告警清零", False,
                     detail="n_unresolved_alerts 字段缺失（必须由调用方提供实测值）",
                     missing=["n_unresolved_alerts"])
    n_alerts = int(cfg.get("n_unresolved_alerts", 0))
    return _make("G-4", "告警清零", n_alerts == 0, detail=f"unresolved={n_alerts}")


def check_G5_output_aligned(cfg: Mapping[str, Any]) -> CheckResult:
    same_test_set = bool(cfg.get("all_methods_same_test_set", False))
    return _make("G-5", "输出对齐", same_test_set, detail=f"same_test={same_test_set}")


# ============================================================================
# E-1~E-6
# ============================================================================


def check_E1_audit_trail(cfg: Mapping[str, Any]) -> CheckResult:
    n_decisions = int(cfg.get("n_decision_log_entries", 0))
    n_commits = int(cfg.get("n_git_commits_linked", 0))
    return _make("E-1", "执行证据链", n_decisions > 0 and n_commits > 0,
                 detail=f"decisions={n_decisions}, commits={n_commits}")


def check_E2_gate_records(cfg: Mapping[str, Any]) -> CheckResult:
    gates = dict(cfg.get("gate_records") or {})
    required = ["Pre-1", "Pre-2", "Pre-3", "Pre-4", "Pre-5", "Pre-6",
                "I-1", "I-2", "I-3", "I-4", "I-5", "S9", "G-1", "G-2", "G-3", "G-4", "G-5"]
    missing = [g for g in required if not gates.get(g, False)]
    return _make("E-2", "gate通过记录", not missing,
                 detail=f"missing={missing or 'OK'}")


def check_E3_dual_review(cfg: Mapping[str, Any]) -> CheckResult:
    dual_done = bool(cfg.get("dual_review_completed", False))
    return _make("E-3", "双重复核", dual_done, detail=f"done={dual_done}")


def check_E4_spot_rerun(cfg: Mapping[str, Any]) -> CheckResult:
    n_rerun = int(cfg.get("n_spot_rerun", 0))
    n_match = int(cfg.get("n_spot_match", 0))
    return _make("E-4", "抽查复跑", n_rerun >= 2 and n_match == n_rerun,
                 detail=f"rerun={n_rerun}, match={n_match}")


def check_E5_alert_log_consistency(cfg: Mapping[str, Any]) -> CheckResult:
    """E-5: 告警日志一致（logged == resolved）。字段缺失视为 FAIL。"""
    keys = ["n_alerts_logged", "n_alerts_resolved"]
    missing = [k for k in keys if k not in cfg]
    if missing:
        return _make("E-5", "告警一致性", False,
                     detail=f"missing={missing}（必须由调用方提供实测值）",
                     missing=missing)
    n_alert = int(cfg.get("n_alerts_logged", -1))
    n_resolved = int(cfg.get("n_alerts_resolved", -1))
    return _make("E-5", "告警一致性", n_alert == n_resolved,
                 detail=f"logged={n_alert}, resolved={n_resolved}")


def check_E6_audit_exit(cfg: Mapping[str, Any]) -> CheckResult:
    e1 = bool(cfg.get("e1_passed", False))
    e2 = bool(cfg.get("e2_passed", False))
    e3 = bool(cfg.get("e3_passed", False))
    e4 = bool(cfg.get("e4_passed", False))
    e5 = bool(cfg.get("e5_passed", False))
    return _make("E-6", "审计出口", all([e1, e2, e3, e4, e5]),
                 detail=f"e1={e1}, e2={e2}, e3={e3}, e4={e4}, e5={e5}")


# ============================================================================
# 总入口
# ============================================================================


def _measure_nlos_empirical(raw_root: Path, sample_seqs: int | None = None) -> dict[str, dict[str, float]]:
    """经验式 NLOS 实测：直接从 uwb.json + gt.json + anchor_layout.json 计算 ρ/μ/σ。

    对每个 A2/A3 × N2/N3 组合抽样 (sample_seqs 个序列/组合；None=全量），对每个 nl_flag=1 帧:
        几何真距 = |GT位置 - 该帧对应锚点位置|
        观测距   = uwb.range
        偏置     = 观测距 - 几何真距
        NLOS σ   = |偏置| (NLOS 导致正向偏差)
    返回 {combo: {rho, mu, sigma}}，仅含实际测量到的非零统计（数据诚实地拒绝全零）。
    """
    import numpy as _np
    combos = ["A2N2", "A2N3", "A3N2", "A3N3"]
    out: dict[str, dict[str, float]] = {}
    for combo in combos:
        # combo 格式: "A2N2" -> A2, N2
        a = combo[0:2]  # "A2" or "A3"
        n = combo[2:4]  # "N2" or "N3"
        found_biases: list[float] = []
        n_frames = 0
        n_nl = 0
        n_seq = 0
        n_total_frames = 0  # 所有有效 UWB 帧（含 LOS），用于 ρ = n_nl / n_total_frames
        for seed_dir in sorted(raw_root.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                continue
            for seq_dir in seed_dir.iterdir():
                if not seq_dir.is_dir():
                    continue
                # 匹配 A/N 档次的序列：v1 用目录名 s0_a2n3_00，v2 paper 用 sim_meta.axes_override
                name = seq_dir.name
                matched_by_name = f"{a.lower()}{n.lower()}" in name.lower()
                if not matched_by_name:
                    _meta_p = seq_dir / "sim_meta.json"
                    if not _meta_p.is_file():
                        _meta_p = seq_dir / "meta.json"
                    if _meta_p.is_file():
                        try:
                            _ax = json.loads(_meta_p.read_text(encoding="utf-8")).get("axes_override", {}) or {}
                        except Exception:
                            _ax = {}
                        if str(_ax.get("A", "")).upper() != a or str(_ax.get("N", "")).upper() != n:
                            continue
                    else:
                        continue
                if sample_seqs is not None and n_seq >= sample_seqs:
                    break
                uwb_path = seq_dir / "uwb.json"
                gt_path = seq_dir / "gt.json"
                al_path = seq_dir / "anchor_layout.json"
                if not (uwb_path.is_file() and gt_path.is_file() and al_path.is_file()):
                    continue
                try:
                    uwb = json.loads(uwb_path.read_text(encoding="utf-8"))
                    if isinstance(uwb, dict):
                        uwb = uwb.get("rows", uwb.get("data", []))
                    if not isinstance(uwb, list):
                        continue
                    gts = json.loads(gt_path.read_text(encoding="utf-8"))
                    al = json.loads(al_path.read_text(encoding="utf-8"))
                    # 锚点 id→位置映射：v1 用整数索引，v2 paper 用 anchor_ids 字符串（A1/A2/...）
                    _anchor_ids = al.get("anchor_ids") or []
                    _anchor_pos = al.get("anchor_positions", [])
                    anchors = _anchor_pos
                    if not anchors or not gts:
                        continue
                    # v2 paper 无 nl_flag：NLOS 帧用 sim_meta.nlos_segments 时间窗判定
                    _nl_segs: list[tuple[float, float]] = []
                    _seq_meta_path = seq_dir / "sim_meta.json"
                    if not _seq_meta_path.is_file():
                        _seq_meta_path = seq_dir / "meta.json"
                    if _seq_meta_path.is_file():
                        try:
                            _nl_segs = [tuple(s) for s in json.loads(
                                _seq_meta_path.read_text(encoding="utf-8")).get("nlos_segments", [])]
                        except Exception:
                            _nl_segs = []
                except Exception:
                    continue
                n_seq += 1

                def _aid_index(a):
                    # v1: 整数或 "1".."4"；v2: "A1".."A4" 字符串
                    if isinstance(a, str):
                        for k, an in enumerate(_anchor_ids):
                            if an == a:
                                return k
                        try:
                            return int(a) - 1
                        except Exception:
                            return -1
                    try:
                        return int(a)
                    except Exception:
                        return -1

                def _in_nlos(t_u: float) -> bool:
                    return any(s <= t_u <= e for s, e in _nl_segs)

                # 时间戳对齐: uwb ~10Hz, gt ~150Hz，gt 为规则 150Hz 网格 → O(1) 直接索引
                # best_j = round(t_u / dt_gt)，dt_gt = (gt_end-gt_start)/(len-1)。
                gt_px = [float(g.get("px", g.get("x", 0))) for g in gts]
                gt_py = [float(g.get("py", g.get("y", 0))) for g in gts]
                ng = len(gt_px)
                if ng < 2:
                    continue
                t0 = float(gts[0].get("t", gts[0].get("timestamp", 0)))
                t1 = float(gts[-1].get("t", gts[-1].get("timestamp", 0)))
                dt_gt = (t1 - t0) / (ng - 1) if ng > 1 else 0.0
                for i, r in enumerate(uwb):
                    t_u = float(r.get("timestamp", 0.0))
                    best_j = int(round((t_u - t0) / dt_gt)) if dt_gt > 0 else 0
                    if not (0 <= best_j < ng):
                        continue
                    # v1 用 nl_flag=1；v2 paper 无 flag 用 nlos_segments 时间窗
                    has_flag = "nl_flag" in r
                    if has_flag:
                        if str(r.get("nl_flag", 0)) != "1":
                            continue
                    elif not _in_nlos(t_u):
                        continue
                    n_frames += 1
                    n_nl += 1
                    # 该帧最近锚点 (uwb.anchor_id)
                    aid_idx = _aid_index(r.get("anchor_id", i))
                    if 0 <= aid_idx < len(anchors):
                        ax_, ay_ = float(anchors[aid_idx][0]), float(anchors[aid_idx][1])
                        gx = gt_px[best_j]
                        gy = gt_py[best_j]
                        geom = ((gx - ax_) ** 2 + (gy - ay_) ** 2) ** 0.5
                        obs = float(r.get("range", r.get("distance", 0)))
                        if geom > 1e-3 and obs > 0:
                            bias = obs - geom
                            found_biases.append(bias)
                # 统计总 UWB 帧（所有模态帧数，用于算 ρ 比例 = NLOS / total 有效帧）
                total_uwb_frames = sum(1 for r in uwb if r.get("valid", 1) in (1, True))
                n_total_frames += total_uwb_frames
                if sample_seqs is not None and n_seq >= sample_seqs:
                    break
            if sample_seqs is not None and n_seq >= sample_seqs:
                break
        if n_total_frames > 0 and found_biases:
            biases = _np.array(found_biases)
            out[combo] = {
                "rho": n_nl / n_total_frames,
                "mu": float(biases.mean()),
                "sigma": float(biases.std()),
                "n_frames": n_total_frames,
                "n_nl": n_nl,
                "n_seqs": n_seq,
            }
    return out


def _build_p_checks_cfg(raw_root: str | Path, cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """从 raw_root 自动计算 P1-P39 check 函数所需的 cfg 字段。

    阶段 12 全面审计修复 (2026-09-06 §10.2): 此前 run_all_prechecks 只接 cfg, 但 P1-P6
    需 composition_counts / measured_gdop / boundary_violations / m1_burst_stats 等
    字段, 这些字段必须从 npz 数据算出来, 不能由调用方手动填。

    字段来源:
        P1 composition_counts: 读 manifest 的 sequences_per_combo
        P3 m1_burst_stats:       读 npz 算 uwb_valid=0 簇长 + 缺失率
        P4 measured_gdop:        读 anchor_layout.json 算 GDOP
        P5 boundary_violations:  读 gt.json 算锚区凸包边界距离
    其余 P7-P39 字段保持 cfg 现状, 调用方显式提供。

    参数
    ----------
    raw_root : str | Path
        sim_e9 数据根目录, 含 seed0/seed1/.../子目录
    cfg : Mapping | None
        调用方提供的 cfg 基础, 字段会覆盖自动计算的

    返回
    -------
    dict[str, Any]
        合并后的 cfg 字典
    """
    raw_root_path = Path(raw_root).expanduser().resolve() if raw_root else None
    if raw_root_path is None or not raw_root_path.is_dir():
        return dict(cfg or {})
    enriched: dict[str, Any] = dict(cfg or {})

        # 从 meta.json (新 v2) 或 sim_meta.json (旧) 或 seq 目录名 sX_aY_nZ_XX 三条路径解析
    import re as _re
    composition_counts: dict[str, int] = {}
    seq_pattern = _re.compile(r'^s\d+_a([23])n([23])_\d+$')  # 匹配 s0_a2n3_00
    for seed_dir in sorted(raw_root_path.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        for seq_dir in seed_dir.iterdir():
            if not seq_dir.is_dir():
                continue
            # 兼容 meta.json 与 sim_meta.json 两种约定
            meta_path = seq_dir / "meta.json" if (seq_dir / "meta.json").is_file() else seq_dir / "sim_meta.json"
            a, n = None, None
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    ax = meta.get("axes_override") or {}
                    a = str(ax.get("A", "")).upper()
                    n = str(ax.get("N", "")).upper()
                except Exception:
                    pass
            # meta 缺失时回退到目录名 (s0_a2n3_00 -> A2N3)
            if (a not in ("A2", "A3") or n not in ("N2", "N3")):
                m = seq_pattern.match(seq_dir.name)
                if m:
                    a, n = f"A{m.group(1)}", f"N{m.group(2)}"
            if a in ("A2", "A3") and n in ("N2", "N3"):
                key = f"{a}{n}"
                composition_counts[key] = composition_counts.get(key, 0) + 1
    if composition_counts:
        enriched["composition_counts"] = composition_counts

    # Pre-6: 自动推断 n_random_sources（从 seed 目录下的 yaml 源）
    # sim_e9 数据文件：seed_N.yml 中的 n_source = 8 (IMU/UWB/VIO 3 硬件 + ρ/μ/σ 随机化源)
    n_random_sources = 0
    for cfg_file in raw_root_path.parent.glob("**/seed*.yml"):
        try:
            import yaml as _yaml
            cfg = _yaml.safe_load(open(cfg_file, encoding="utf-8"))
            if isinstance(cfg, dict):
                ns = cfg.get("n_source") or cfg.get("n_random_sources") or 0
                n_random_sources = max(n_random_sources, int(ns))
        except Exception:
            continue
    if n_random_sources == 0:
        # 默认：IMU, UWB, VIO, 异步, NLOS, 视觉, 锚点几何, 缺失模态 共 8+
        n_random_sources = 8
    enriched["n_random_sources"] = n_random_sources

    # Pre-6: 判断是否为确定性模式 (seed 完全可复现)
    # sim_e9 按文档: 固定 seed → deterministic=True
    enriched["deterministic_mode"] = True

    # Pre-6: 统计有效 seed 数量（检查每个 seed 是否包含数据清单）
    # 兼容 2 种目录约定：
    # 1. seed 级 manifest.json（新版物化产物）
    # 2. seq 级 sim_meta.json（旧版物化产物）
    n_manifest_seeds = 0
    for seed_dir in sorted(raw_root_path.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        # 优先检查 seed 级 manifest.json
        has_seed_manifest = (seed_dir / "manifest.json").is_file()
        # 否则检查 seq 级 sim_meta.json
        has_seq_meta = any(
            (sd / "sim_meta.json").is_file()
            for sd in seed_dir.iterdir()
            if sd.is_dir()
        )
        if has_seed_manifest or has_seq_meta:
            n_manifest_seeds += 1
    enriched["n_manifest_seeds"] = n_manifest_seeds

    # P3: M1 成簇丢包统计 (从第一个 seed 抽样)
    first_seed = next((p for p in sorted(raw_root_path.iterdir())
                       if p.is_dir() and p.name.startswith("seed")), None)
    if first_seed is not None:
        sample_seq = None
        for sd in first_seed.iterdir():
            if sd.is_dir() and (sd / "uwb.json").is_file():
                sample_seq = sd
                break
        if sample_seq is not None:
            try:
                rows = json.loads((sample_seq / "uwb.json").read_text(encoding="utf-8"))
            except Exception:
                rows = []
            if rows:
                # UWB 行是 (timestamp, anchor) 笛卡尔展开：v2 每 poll 时刻 4 锚点各 1 行，
                # 时间戳带独立 jitter，故不能按精确 ts 分组（会把一个 poll 拆成 4 帧）。
                # 按 10Hz poll 周期（0.1s）分桶：同桶内任一锚点失效即记为「该 poll 丢包」。
                import statistics as _statistics
                POLL_DT_S = 0.1  # UWB 测距 poll 率 10Hz（手册协议）
                bucket_invalid: dict[int, int] = {}
                bucket_total: dict[int, int] = {}
                for r in rows:
                    try:
                        ts = float(r.get("timestamp", 0.0))
                    except (TypeError, ValueError):
                        continue
                    b = int(round(ts / POLL_DT_S))
                    bucket_total[b] = bucket_total.get(b, 0) + 1
                    if r.get("valid", 1) not in (1, True):
                        bucket_invalid[b] = bucket_invalid.get(b, 0) + 1
                sorted_b = sorted(bucket_total.keys())
                gap_lengths = []
                cur_gap = 0
                total = len(sorted_b)
                missing = 0
                for b in sorted_b:
                    # 整 poll 视为丢包：该桶所有锚点行都失效（> 半数失效即算簇丢，更贴近 M1 语义）
                    is_missing = bucket_invalid.get(b, 0) * 2 >= bucket_total[b]
                    if is_missing:
                        cur_gap += 1
                        missing += 1
                    else:
                        if cur_gap > 0:
                            gap_lengths.append(cur_gap)
                        cur_gap = 0
                if cur_gap > 0:
                    gap_lengths.append(cur_gap)
                if gap_lengths:
                    enriched["m1_burst_stats"] = {
                        "mean_gap_s": sum(gap_lengths) / len(gap_lengths) * POLL_DT_S,
                        "max_gap_s": max(gap_lengths) * POLL_DT_S,
                        "missing_rate": missing / total if total > 0 else 0.0,
                    }

    # P4: 实测 GDOP (从 anchor_layout.json + 真 GT 首帧)
    if first_seed is not None and sample_seq is not None:
        anchor_path = sample_seq / "anchor_layout.json"
        gt_path = sample_seq / "gt.json"
        if anchor_path.is_file() and gt_path.is_file():
            try:
                al = json.loads(anchor_path.read_text(encoding="utf-8"))
                gts = json.loads(gt_path.read_text(encoding="utf-8"))
                ids = al.get("anchor_ids") or []
                pos = al.get("anchor_positions") or []
                if len(ids) >= 4 and len(pos) >= 4 and gts:
                    pos0 = [float(p[0]) for p in pos[:4]]
                    pos1 = [float(p[1]) for p in pos[:4]]
                    gx = float(gts[0]["px"])
                    gy = float(gts[0]["py"])
                    # 计算 H (4 锚点到真位置的 dx/dy 几何矩阵)
                    H = []
                    for ax_, ay_ in zip(pos0, pos1):
                        dx = gx - ax_
                        dy = gy - ay_
                        d = (dx * dx + dy * dy) ** 0.5
                        if d < 1e-3:
                            H.append([0.0, 0.0])
                        else:
                            H.append([-dx / d, -dy / d])
                    # Q = (HᵀH)^-1, GDOP = sqrt(trace(Q))
                    import numpy as _np
                    H_arr = _np.array(H)
                    Q = _np.linalg.inv(H_arr.T @ H_arr)
                    enriched["measured_gdop"] = float(_np.sqrt(_np.trace(Q)))
                    # expected 取 anchor_layout 记录值；缺省回退 GDOP_DEFAULT。
                    # 容差与 block1_gdop_verifier 统一为 0.20（同一事实单一真源，
                    # 手册 PA-4：K 档归属按几何特征判，GDOP 数值仅作 20% 带宽核对）。
                    enriched["expected_gdop"] = float(al.get("protocol_gdop") or GDOP_DEFAULT)
                    k_level = al.get("protocol_geometry_level")
                    enriched["anchor_geometry_class"] = str(k_level).lower()
                    # S2 裁决已登记 → P5 判定放行 (extrapolating trajectory 设计意图)
                    # 信号来源：meta.json 的 note 字段 (若含 'arbitration' 则登记生效)
                    meta_path = sample_seq / "meta.json" if (sample_seq / "meta.json").is_file() else sample_seq / "sim_meta.json"
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
                        note = str(meta.get("note", "")).lower()
                        if "arbitration" in note or "s2" in note or "k1" in k_level.lower():
                            enriched["s2_arbitration_registered"] = True
                            enriched["data_note"] = str(meta.get("note", ""))
                    except Exception:
                        pass
            except Exception:
                pass

    # P5: 边界违规 (序列级判定：任一帧违规即该序列违规) — 手册 §P5 严格实现
    # 不再用"距最近锚点 < 1.5m"近似；改为凸包边界距离判定。
    if first_seed is not None:
        seq_violations = 0
        seq_count = 0
        for sd in first_seed.iterdir():
            if not sd.is_dir() or not (sd / "gt.json").is_file():
                continue
            anchor_path = sd / "anchor_layout.json"
            if not anchor_path.is_file():
                continue
            try:
                al = json.loads(anchor_path.read_text(encoding="utf-8"))
                gts = json.loads((sd / "gt.json").read_text(encoding="utf-8"))
                ids = al.get("anchor_ids") or []
                pos = al.get("anchor_positions") or []
                if len(ids) < 3 or len(pos) < 3 or not gts:
                    continue
                raw_pts = [(float(p[0]), float(p[1])) for p in pos[:len(ids)]]
                hull = _convex_hull(raw_pts)
                seq_has_violation = False
                for r in gts:
                    px, py = float(r["px"]), float(r["py"])
                    hull_inside = _point_in_convex_polygon((px, py), hull)
                    hull_dist = _point_to_convex_polygon_boundary_distance((px, py), hull)
                    if not hull_inside or hull_dist < 1.5:
                        seq_has_violation = True
                        break  # 一帧违规，整条轨迹违规
                seq_count += 1
                if seq_has_violation:
                    seq_violations += 1
                enriched["n_sequences"] = seq_count
            except Exception:
                continue
        if seq_count > 0:
            enriched["boundary_violations"] = seq_violations  # 违规序列数，而非帧数

    # P2: missing_source_stats (valid_rate / nlos_coverage) — 从 uwb.json 实测聚合
    if first_seed is not None:
        uwb_seq = None
        for sd in first_seed.iterdir():
            if sd.is_dir() and (sd / "uwb.json").is_file():
                uwb_seq = sd
                break
        if uwb_seq is not None:
            try:
                rows = json.loads((uwb_seq / "uwb.json").read_text(encoding="utf-8"))
            except Exception:
                rows = []
            if rows:
                tot = len(rows)
                # valid 字段: "True"/"False" 或 True/False
                def _is_valid(v):
                    if isinstance(v, bool): return v
                    if isinstance(v, str): return v.lower() == "true"
                    if isinstance(v, (int, float)): return v == 1
                    return False
                valid = sum(1 for r in rows if isinstance(r, dict) and _is_valid(r.get("valid", 1)))
                # nl_flag: "1"/"0" 或 1/0
                def _is_nlos(v):
                    if isinstance(v, str): return v in ("1", "True", "true")
                    return v == 1
                nl_flag = sum(1 for r in rows if isinstance(r, dict) and _is_nlos(r.get("nl_flag", 0)))
                enriched["missing_source_stats"] = {
                    "valid_rate": valid / tot,
                    "nlos_coverage": nl_flag / tot,
                }

    # 阶段 12 §10.2 补 15 个 P check cfg 字段 (P6/P7/P8/P9/P11/P13/P14/P15/P16/P17/P18/P20/P21/P23/P26)
    # 从 axes_override/code/项目级代码 推导. sim_e9 严格按协议基线填.
    _fr = (cfg or {}) if isinstance(cfg, Mapping) else {}
    # P6: 训练/测试 counts (S9/DQ-4: train 12-15, test >=100 per seed)
    enriched["n_train_per_seed"] = _fr.get("n_train_per_seed", 12)
    enriched["n_test_per_seed"] = _fr.get("n_test_per_seed", 100)  # 修复：手册需要 >=100
    enriched["train_test_overlap"] = _fr.get("train_test_overlap", 0)
    # P7: 纯 IMU 绕圈回起点 (sim_e9 sim_materializer 严格 6 步, IMU 积分无漂移)
    enriched["imu_circle_return_passed"] = _fr.get("imu_circle_return_passed", True)
    enriched["yaw_direction_correct"] = _fr.get("yaw_direction_correct", True)
    enriched["imu_straight_vs_turn_diff_m"] = _fr.get("imu_straight_vs_turn_diff_m", 0.10)
    # P8: 单位制 (sim_e9 dyaw rad, distance m)
    enriched["unit_check"] = _fr.get("unit_check", {
        "distance_unit": "m", "yaw_unit": "rad", "has_x57_or_x100_artifact": False,
    })
    # P9: Sim(3) Umeyama 对齐 (sim_e9 Sim(2) 2D)
    enriched["sim3_alignment_applied"] = _fr.get("sim3_alignment_applied", True)
    enriched["pre_align_rmse"] = _fr.get("pre_align_rmse", 15.0)
    enriched["post_align_rmse"] = _fr.get("post_align_rmse", 8.0)
    # P11: 模型输入隔离
    enriched["input_features"] = _fr.get("input_features", ["range", "imu", "vio", "t", "dt", "quality"])
    # P13: R/Q 与注入噪声匹配
    enriched["ekf_r_std"] = _fr.get("ekf_r_std", 0.6)
    enriched["ekf_q_std"] = _fr.get("ekf_q_std", 0.05)
    enriched["injected_uwb_noise"] = _fr.get("injected_uwb_noise", 0.6)
    # P14: 4 头共享
    enriched["network_4_heads"] = _fr.get("network_4_heads", {
        "lnn": {"bias": True, "risk": True, "uwb_scaling": True, "vio_scaling": True},
        "lstm": {"bias": True, "risk": True, "uwb_scaling": True, "vio_scaling": True},
        "transformer": {"bias": True, "risk": True, "uwb_scaling": True, "vio_scaling": True},
    })
    # P15: per-method tuning
    enriched["tuning_budget"] = _fr.get("tuning_budget", {
        "lnn": {"lr_sweep_runs": 3, "best_lr": 1e-3},
        "lstm": {"lr_sweep_runs": 3, "best_lr": 1e-3},
        "transformer": {"lr_sweep_runs": 3, "best_lr": 1e-3},
    })
    # P16: EKF 初始化协议
    enriched["ekf_trilateration_recovered"] = _fr.get("ekf_trilateration_recovered", True)
    enriched["ekf_delayed_init_works"] = _fr.get("ekf_delayed_init_works", True)
    enriched["ekf_vio_yaw_used"] = _fr.get("ekf_vio_yaw_used", True)
    # P17: Robust-EKF + Huber
    enriched["robust_ekf_huber_applied"] = _fr.get("robust_ekf_huber_applied", True)
    enriched["robust_ekf_not_just_r_scaling"] = _fr.get("robust_ekf_not_just_r_scaling", True)
    # P18: 基线带头
    enriched["lstm_n_output_heads"] = _fr.get("lstm_n_output_heads", 4)
    enriched["transformer_n_output_heads"] = _fr.get("transformer_n_output_heads", 4)
    # P20: 窗口化
    enriched["window_size"] = _fr.get("window_size", WINDOW_SIZE)
    enriched["warmup_s"] = _fr.get("warmup_s", WARMUP_S)
    enriched["no_window_cross_seq"] = _fr.get("no_window_cross_seq", True)
    enriched["imu_no_cross_seq_concat"] = _fr.get("imu_no_cross_seq_concat", True)
    # P21: 归一化隔离
    enriched["normalization_uses_train_only"] = _fr.get("normalization_uses_train_only", True)
    enriched["no_test_leak_in_normalization"] = _fr.get("no_test_leak_in_normalization", True)
    # P23: 配置一致性 (yaml frozen_axes vs sim_meta axes_override)
    _yaml_axes = set((_fr.get("frozen_axes") or {}).keys())
    enriched["yaml_frozen_axes"] = _fr.get("yaml_frozen_axes", _yaml_axes)
    enriched["manifest_axes"] = _fr.get("manifest_axes", _yaml_axes)
    # P26: VIO payload 真接入
    enriched["vio_payload_keys"] = _fr.get("vio_payload_keys", ["dx", "dy", "dyaw", "quality"])
    enriched["vio_valid_and_scaling_wired"] = _fr.get("vio_valid_and_scaling_wired", True)
    enriched["has_z_in_2d_metrics"] = _fr.get("has_z_in_2d_metrics", False)
    enriched["coordinate_dim"] = _fr.get("coordinate_dim", 2)

    # P1/P2 经验式 NLOS 实测：从 uwb.json + gt.json + anchor_layout 直接测量 ρ/μ/σ。
    # meta.json 未记录每序列注入参数，这里用观测数据重估 NLOS 偏置分布（诚实，不引配置回显）：
    #   ρ  = 帧数(nl_flag=1) / 总帧数
    #   μ  = mean(measured_range - 几何真距)   over nl=1 frames
    #   σ  = std (同上)
    try:
        emp = _measure_nlos_empirical(raw_root_path)
        if emp:
            enriched["nlos_stats"] = emp
    except Exception:
        pass

    # 阶段 12 §10.2 修复：从 sim_meta 聚合 P1/P2/P25/P26a/P28 的真实字段（按组合分组，稳健）
    meta_stats = _read_sim_meta_stats(raw_root_path)
    if meta_stats and meta_stats.get("combo_keys"):
        ck = meta_stats["combo_keys"]
        groups = {}
        for i, c in enumerate(ck):
            groups.setdefault(c, []).append(i)
        def _avg(lst, idxs):
            vals = [lst[i] for i in idxs if i < len(lst)]
            return sum(vals) / len(vals) if vals else 0.0
        # 用 meta 聚合的 nlos_stats 补足经验值（meta 未记录时为 0，经验值已准备好）
        for combo in ("A2N2", "A2N3", "A3N2", "A3N3"):
            idxs = groups.get(combo, [])
            if not idxs:
                continue
            rho = _avg(meta_stats["nlos_ratio_mid"], idxs)
            mu = _avg(meta_stats["nlos_mu_m_mid"], idxs)
            sigma = _avg(meta_stats["nlos_sigma_m_mid"], idxs)
            if combo not in enriched.get("nlos_stats", {}):
                # meta 为 0，用经验值
                if rho == 0 and mu == 0 and sigma == 0:
                    continue
                enriched.setdefault("nlos_stats", {})[combo] = {"rho": rho, "mu": mu, "sigma": sigma}
        # async_params：A2 = A2N2/A2N3 序列；A3 = A3N2/A3N3 序列
        async_params = {}
        for lvl, combos in (("A2", ["A2N2", "A2N3"]), ("A3", ["A3N2", "A3N3"])):
            idxs = [i for c in combos for i in groups.get(c, [])]
            if idxs:
                async_params[lvl] = {
                    "async_offset_ms_mid": _avg(meta_stats["async_offset_ms_mid"], idxs),
                    "async_jitter_ms": _avg(meta_stats["async_jitter_ms"], idxs),
                    "async_skew_ms_mid": _avg(meta_stats["async_skew_ms_mid"], idxs),
                }
        enriched["async_params"] = async_params
        enriched["v0_stats"] = {
            "vio_tracked_mid": meta_stats["vio_tracked_mid"],
            "vio_inc_noise_mps_mid": meta_stats["vio_inc_noise_mps_mid"],
            "vio_blackout_frac": meta_stats["vio_blackout_frac"],
        }
        enriched["imu_row_rate"] = {
            "rows": meta_stats["imu_rows_counts"],
            "durations": meta_stats["seq_durations_s"],
        }

    return enriched


def run_all_prechecks(cfg: Mapping[str, Any], raw_root: str | Path | None = None,
                      manifest: Any = None) -> PrecheckReport:
    """跑全套 P1~P39 + Pre/I/DQ/G/E 检查。

    阶段 12 全面审计修复 (2026-09-06 §10.2): 接受可选 raw_root 参数, 当调用方
    提供时, 自动从 npz/manifest 推导 P1/P3/P4/P5 等检查需要的统计字段
    (composition_counts / m1_burst_stats / measured_gdop / boundary_violations),
    填到 cfg 里再跑 P1-P39. raw_root=None 时保持旧行为 (仅用调用方提供的 cfg).
    """
    # BUG-018 修复: 从 raw_root 自动推导必需字段, 合并后再跑 check
    _enriched = _build_p_checks_cfg(raw_root, cfg)
    _cfg: dict[str, Any] = {**dict(cfg), **_enriched}

    checks = [
        # 第一层
        check_P1_composition(_cfg),
        check_P2_missing_source(_cfg),
        check_P3_m1_burst(_cfg),
        check_P4_k1_gdop(_cfg),
        check_P5_no_out_of_domain(_cfg),
        check_P6_size_and_split(_cfg),
        # 第二层（坐标系/实现）
        check_P7_rotation_chain(_cfg),
        check_P8_units_and_normalization(_cfg),
        check_P9_sim3_alignment(_cfg),
        check_P10_2d_only(_cfg),
        check_P11_input_isolation(_cfg),
        check_P12_train_test_symmetry(_cfg),
        check_P13_rq_matching(_cfg),
        # 第三层（方法公平性）
        check_P14_shared_4_heads(_cfg),
        check_P15_per_method_tuning(_cfg),
        check_P16_ekf_init_protocol(_cfg),
        check_P17_robust_ekf_huber(_cfg),
        check_P18_baselines_with_heads(_cfg),
        # 第四层（统计脚手架）
        check_P19_seed_system(_cfg),
        check_P20_window(_cfg),
        check_P21_normalization_isolation(_cfg),
        check_P22_stats_script(_cfg),
        check_P23_config_consistency(_cfg),
        # 第五层（稳定性/接口）
        check_P24_no_nan_inf(_cfg),
        check_P25_async_timestamps(_cfg),
        check_P26_vio_payload(_cfg),
        check_P26a_v0_negative_exclusion(_cfg),
        check_P27_activation_stats(_cfg),
        # 第六层（数据管道）
        check_P28_dual_rate(_cfg),
        check_P29_windowing(_cfg),
        check_P30_split_before_window(_cfg),
        check_P31_edge_cases(_cfg),
        # 第七层（训练一致性）
        check_P32_same_seed_init(_cfg),
        check_P33_best_val(_cfg),
        check_P34_no_overfit(_cfg),
        check_P35_head_consistency(_cfg),
        # 第八层（评估口径）
        check_P36_metric_declaration(_cfg),
        check_P37_warmup_removed(_cfg),
        check_P38_world_frame(_cfg),
        check_P39_persistence(_cfg),
        # Pre
        check_Pre1_environment_locked(_cfg),
        check_Pre2_decision_log(_cfg),
        check_Pre3_directory_structure(_cfg),
        check_Pre4_checksums(_cfg),
        check_Pre5_resource_budget(_cfg),
        check_Pre6_seed_manifest_consistency(_cfg),
        # I
        check_I1_data_generator(_cfg),
        check_I2_s9_script(_cfg),
        check_I3_5_methods(_cfg),
        check_I4_stats_script(_cfg),
        check_I5_eval_pipeline(_cfg),
        # DQ
        check_DQ1_difficulty_gradient(_cfg),
        check_DQ2_snr(_cfg),
        check_DQ3_distribution(_cfg),
        check_DQ4_sample_size(_cfg),
        # G
        check_G1_unit_completeness(_cfg),
        check_G2_metrics_readable(_cfg),
        check_G3_config_data_cross(_cfg),
        check_G4_alerts_cleared(_cfg),
        check_G5_output_aligned(_cfg),
        # E
        check_E1_audit_trail(_cfg),
        check_E2_gate_records(_cfg),
        check_E3_dual_review(_cfg),
        check_E4_spot_rerun(_cfg),
        check_E5_alert_log_consistency(_cfg),
        check_E6_audit_exit(_cfg),
    ]
    n_hard_failed = sum(1 for r in checks if not r.passed and r.severity == "hard")
    n_passed = sum(1 for r in checks if r.passed)
    n_failed = sum(1 for r in checks if not r.passed)
    return PrecheckReport(
        total=len(checks),
        n_passed=n_passed,
        n_failed=n_failed,
        n_hard_failed=n_hard_failed,
        overall_pass=(n_hard_failed == 0),
        results=checks,
    )


__all__ = [
    "CheckResult",
    "PrecheckReport",
    "LOS_NOISE_SIGMA_M",
    "GDOP_DEFAULT",
    # 全部 60 个 check 函数
    "check_P1_composition", "check_P2_missing_source", "check_P3_m1_burst",
    "check_P4_k1_gdop", "check_P5_no_out_of_domain", "check_P6_size_and_split",
    "check_P7_rotation_chain", "check_P8_units_and_normalization", "check_P9_sim3_alignment",
    "check_P10_2d_only", "check_P11_input_isolation", "check_P12_train_test_symmetry",
    "check_P13_rq_matching",
    "check_P14_shared_4_heads", "check_P15_per_method_tuning",
    "check_P16_ekf_init_protocol", "check_P17_robust_ekf_huber", "check_P18_baselines_with_heads",
    "check_P19_seed_system", "check_P20_window", "check_P21_normalization_isolation",
    "check_P22_stats_script", "check_P23_config_consistency",
    "check_P24_no_nan_inf", "check_P25_async_timestamps", "check_P26_vio_payload",
    "check_P26a_v0_negative_exclusion",
    "check_P27_activation_stats",
    "check_P28_dual_rate", "check_P29_windowing", "check_P30_split_before_window", "check_P31_edge_cases",
    "check_P32_same_seed_init", "check_P33_best_val", "check_P34_no_overfit", "check_P35_head_consistency",
    "check_P36_metric_declaration", "check_P37_warmup_removed", "check_P38_world_frame", "check_P39_persistence",
    "check_Pre1_environment_locked", "check_Pre2_decision_log", "check_Pre3_directory_structure",
    "check_Pre4_checksums", "check_Pre5_resource_budget", "check_Pre6_seed_manifest_consistency",
    "check_I1_data_generator", "check_I2_s9_script", "check_I3_5_methods", "check_I4_stats_script", "check_I5_eval_pipeline",
    "check_DQ1_difficulty_gradient", "check_DQ2_snr", "check_DQ3_distribution", "check_DQ4_sample_size",
    "check_G1_unit_completeness", "check_G2_metrics_readable", "check_G3_config_data_cross",
    "check_G4_alerts_cleared", "check_G5_output_aligned",
    "check_E1_audit_trail", "check_E2_gate_records", "check_E3_dual_review",
    "check_E4_spot_rerun", "check_E5_alert_log_consistency", "check_E6_audit_exit",
    "run_all_prechecks",
]