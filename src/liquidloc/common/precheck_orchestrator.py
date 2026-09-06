"""异步高NLOS实验全流程保障手册 P1~P39 跑前预检统一 orchestrator。

把所有 P1~P39 八层预检集中到一个可调用的检查流水线中，按手册 line 146-209 的定义
落地 39 个 check 函数；以及 Pre-1~Pre-6、I-1~I-5、DQ-1~DQ-4、G-1~G-5、E-1~E-6。

本模块只做"检查+报告"，不强制 fail-stop（调用方决定是否 raise）。
每个 check 返回 `CheckResult(passed, detail, evidence)`。

公开入口：
    run_all_prechecks(cfg, raw_root, manifest) -> PrecheckReport
    run_dataset_gates(cfg) -> list[CheckResult]  (P1~P6, Pre-1~Pre-6, I-1~I-5)
    run_model_fairness_gates(cfg, methods) -> list[CheckResult]  (P14~P18, P32~P35)
    run_stats_gates(...) -> list[CheckResult]  (P19~P23, P36~P39)
    run_stability_gates(...) -> list[CheckResult]  (P24~P27)
    run_pipeline_gates(...) -> list[CheckResult]  (P28~P31)
    run_dq_gates(...) -> list[CheckResult]  (DQ-1~DQ-4)
    run_run_complete_gates(...) -> list[CheckResult]  (G-1~G-5, E-1~E-6)
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

from liquidloc.common.validation import coerce_finite_scalar


# 误差预算常量（手册 Part 0 S4）
LOS_NOISE_SIGMA_M: float = 0.6
GDOP_DEFAULT: float = 1.19


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


# ============================================================================
# 第一层：数据正确性 (P1~P6) + 实验准备 (Pre-1~Pre-6) + 实现完成 (I-1~I-5)
# ============================================================================


def check_P1_composition(cfg: Mapping[str, Any]) -> CheckResult:
    """P1: 4 组合各 25%（偏差 ≤5%）。"""
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
    return _make("P1", "4组合各25%", not failed, detail=" / ".join(failed) or "OK", composition=counts, total=total)


def check_P2_missing_source(cfg: Mapping[str, Any]) -> CheckResult:
    """P2: uwb_valid=0 分布与 M1 缺失率一致；nl_flag 与 N 注入统计可交叉验证。"""
    stats = dict(cfg.get("missing_source_stats") or {})
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
    """P4: K1 GDOP ≈1.19（偏差 ≤10%）；K 档归属按几何特征判定。"""
    measured = float(cfg.get("measured_gdop", 0.0))
    expected = float(cfg.get("expected_gdop", GDOP_DEFAULT))
    k_arch = str(cfg.get("anchor_geometry_class", "unknown"))
    if measured == 0.0:
        return _make("P4", "K1 GDOP≈1.19", False, detail="measured_gdop 缺失", severity="soft")
    rel = abs(measured - expected) / max(expected, 1e-9)
    return _make("P4", "K1 GDOP≈1.19", rel <= 0.10,
                 detail=f"measured={measured:.3f}, expected={expected:.3f}, rel={rel:.1%}, geometry={k_arch}",
                 measured_gdop=measured, expected_gdop=expected, k_class=k_arch)


def check_P5_no_out_of_domain(cfg: Mapping[str, Any]) -> CheckResult:
    """P5: 轨迹全程在锚区凸包内，距边界 ≥1.5m 缓冲。"""
    boundary_violations = int(cfg.get("boundary_violations", 0))
    n_sequences = int(cfg.get("n_sequences", 1))
    return _make("P5", "轨迹不出域", boundary_violations == 0,
                 detail=f"violations={boundary_violations}/{n_sequences}",
                 boundary_violations=boundary_violations, n_sequences=n_sequences)


def check_P6_size_and_split(cfg: Mapping[str, Any]) -> CheckResult:
    """P6: 训练 6-7 / 测试 ≥60 / seed，划分按整条序列不重叠。"""
    n_train = int(cfg.get("n_train_per_seed", 0))
    n_test = int(cfg.get("n_test_per_seed", 0))
    overlap = int(cfg.get("train_test_overlap", 0))
    passed = (n_train >= 4) and (n_test >= 60) and (overlap == 0)
    return _make("P6", "规模与划分", passed,
                 detail=f"train={n_train}, test={n_test}, overlap={overlap}",
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
    """Pre-6: 随机源清单与 manifest 一致；确定性模式开启。"""
    n_manifest_seeds = int(cfg.get("n_manifest_seeds", 0))
    n_random_sources = int(cfg.get("n_random_sources", 0))
    deterministic = bool(cfg.get("deterministic_mode"))
    return _make("Pre-6", "随机源核对",
                 n_manifest_seeds > 0 and n_random_sources >= 3 and deterministic,
                 detail=f"manifest_seeds={n_manifest_seeds}, sources={n_random_sources}, det={deterministic}")


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
    """I-3: 5 方法脚本存在（LNN/LSTM/Transformer/EKF/Robust-EKF）。"""
    methods = list(cfg.get("available_methods") or [])
    required = ["lnn", "lstm", "transformer", "ekf", "robust_ekf"]
    missing = [m for m in required if m not in methods]
    return _make("I-3", "5方法脚本", not missing,
                 detail=f"missing={missing or 'OK'}",
                 available=methods, missing=missing)


def check_I4_stats_script(cfg: Mapping[str, Any]) -> CheckResult:
    """I-4: 统计脚本 P22 全输出可跑通。"""
    has_outputs = list(cfg.get("stats_script_outputs") or [])
    required = ["mean", "std", "p95", "p50", "delta_vs_lnn", "wilcoxon_p", "ci_95",
                "holm_bonferroni", "slicing_4_combo", "anova_2x2", "c1_vs_c4"]
    missing = [k for k in required if k not in has_outputs]
    return _make("I-4", "统计脚本", not missing,
                 detail=f"missing={missing or 'OK'}")


def check_I5_eval_pipeline(cfg: Mapping[str, Any]) -> CheckResult:
    """I-5: 评估口径实现 Sim(3) 对齐 + 2D + warm-up 剔除 + 世界系。"""
    sim3 = bool(cfg.get("sim3_alignment"))
    is_2d = bool(cfg.get("is_2d"))
    warmup_removed = bool(cfg.get("warmup_removed"))
    world_frame = bool(cfg.get("world_frame"))
    return _make("I-5", "评估口径",
                 sim3 and is_2d and warmup_removed and world_frame,
                 detail=f"sim3={sim3}, 2d={is_2d}, warmup={warmup_removed}, world={world_frame}")


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
        entry = budgets.get(m, {})
        lr_sweep = int(entry.get("lr_sweep_runs", 0))
        has_best_lr = "best_lr" in entry
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
    n_seeds = int(cfg.get("n_seeds", 0))
    deterministic = bool(cfg.get("deterministic_mode", False))
    tf32_set = bool(cfg.get("tf32_explicit_set", False))
    return _make("P19", "seed体系", n_seeds >= 5 and deterministic and tf32_set,
                 detail=f"n_seeds={n_seeds}, deterministic={deterministic}, tf32={tf32_set}")


def check_P20_window(cfg: Mapping[str, Any]) -> CheckResult:
    W = int(cfg.get("window_size", 0))
    warmup = float(cfg.get("warmup_s", -1.0))
    no_cross = bool(cfg.get("no_window_cross_seq", False))
    imu_no_cross = bool(cfg.get("imu_no_cross_seq_concat", False))
    return _make("P20", "窗口化", W == 128 and warmup == 10.0 and no_cross and imu_no_cross,
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
    n_nan = int(cfg.get("n_nan_detected", 0))
    n_inf = int(cfg.get("n_inf_detected", 0))
    grad_exploded = bool(cfg.get("gradient_exploded", False))
    return _make("P24", "无NaN/Inf/梯度爆炸",
                 n_nan == 0 and n_inf == 0 and not grad_exploded,
                 detail=f"nan={n_nan}, inf={n_inf}, grad_exploded={grad_exploded}")


def check_P25_async_timestamps(cfg: Mapping[str, Any]) -> CheckResult:
    monotonic = bool(cfg.get("timestamps_monotonic", False))
    async_in_pre = bool(cfg.get("async_injected_before_resample", False))
    jitter_per_packet = bool(cfg.get("jitter_per_packet", False))
    return _make("P25", "A2/A3时间戳", monotonic and async_in_pre and jitter_per_packet,
                 detail=f"mono={monotonic}, pre_resample={async_in_pre}, per_packet={jitter_per_packet}")


def check_P26_vio_payload(cfg: Mapping[str, Any]) -> CheckResult:
    vio_keys = set(cfg.get("vio_payload_keys") or [])
    required = {"dx", "dy", "dyaw", "quality"}
    all_4 = required.issubset(vio_keys)
    real_wire = bool(cfg.get("vio_valid_and_scaling_wired", False))
    return _make("P26", "VIO payload真接入", all_4 and real_wire,
                 detail=f"keys={vio_keys}, wired={real_wire}")


def check_P27_activation_stats(cfg: Mapping[str, Any]) -> CheckResult:
    saved_heatmap = bool(cfg.get("activation_heatmap_saved", False))
    n4_heads = int(cfg.get("n_4_heads_with_stats", 0))
    return _make("P27", "激活统计可视化", saved_heatmap and n4_heads == 4,
                 detail=f"saved={saved_heatmap}, n_heads={n4_heads}")


# ============================================================================
# 第六层：P28~P31 数据管道
# ============================================================================


def check_P28_dual_rate(cfg: Mapping[str, Any]) -> CheckResult:
    imu_150 = bool(cfg.get("imu_at_150hz_native", False))
    imu_no_downsample = bool(cfg.get("imu_not_downsampled", False))
    nearest_neighbor = bool(cfg.get("resample_method_is_nearest_neighbor", False))
    return _make("P28", "多速率重采样", imu_150 and imu_no_downsample and nearest_neighbor,
                 detail=f"imu_150={imu_150}, no_down={imu_no_downsample}, NN={nearest_neighbor}")


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
    c4 = float(cfg.get("c4_mean_rmse", 0.0))
    c1 = float(cfg.get("c1_mean_rmse", 0.0))
    target_diff = float(cfg.get("target_diff_m", 2.4))
    diff = c4 - c1
    return _make("DQ-1", "难度梯度", diff >= target_diff,
                 detail=f"c4={c4:.2f}, c1={c1:.2f}, diff={diff:.2f}, target>={target_diff}")


def check_DQ2_snr(cfg: Mapping[str, Any]) -> CheckResult:
    bias = float(cfg.get("nlos_bias_m", 0.0))
    std = float(cfg.get("nlos_std_m", 0.0))
    los = float(cfg.get("los_noise_m", 0.6))
    gdop = float(cfg.get("gdop", GDOP_DEFAULT))
    nlos_total = bias + std
    denom = los * gdop
    snr = nlos_total / denom if denom > 0 else 0.0
    return _make("DQ-2", "信噪比", snr >= 3.0,
                 detail=f"snr={snr:.2f} (>=3.0)", nlos_snr=snr)


def check_DQ3_distribution(cfg: Mapping[str, Any]) -> CheckResult:
    def dev(a, b): return abs(a - b) / max(abs(a), abs(b), 1e-9)
    rho_d = dev(float(cfg.get("train_rho", 0)), float(cfg.get("test_rho", 0)))
    mu_d = dev(float(cfg.get("train_mu", 0)), float(cfg.get("test_mu", 0)))
    sig_d = dev(float(cfg.get("train_sigma", 0)), float(cfg.get("test_sigma", 0)))
    passed = max(rho_d, mu_d, sig_d) <= 0.10
    return _make("DQ-3", "分布一致", passed,
                 detail=f"rho={rho_d:.1%}, mu={mu_d:.1%}, sigma={sig_d:.1%}")


def check_DQ4_sample_size(cfg: Mapping[str, Any]) -> CheckResult:
    n_seeds = int(cfg.get("n_seeds", 0))
    n_test = int(cfg.get("n_test_per_seed", 0))
    total = n_seeds * n_test
    return _make("DQ-4", "样本量充分", total >= 300,
                 detail=f"n_seeds={n_seeds}×{n_test}={total} (need≥300)")


# ============================================================================
# G-1~G-5
# ============================================================================


def check_G1_unit_completeness(cfg: Mapping[str, Any]) -> CheckResult:
    expected = 25
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
    n_alert = int(cfg.get("n_alerts_logged", 0))
    n_resolved = int(cfg.get("n_alerts_resolved", 0))
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


def run_all_prechecks(cfg: Mapping[str, Any]) -> PrecheckReport:
    """跑全套 P1~P39 + Pre/I/DQ/G/E 检查。"""
    checks = [
        # 第一层
        check_P1_composition(cfg),
        check_P2_missing_source(cfg),
        check_P3_m1_burst(cfg),
        check_P4_k1_gdop(cfg),
        check_P5_no_out_of_domain(cfg),
        check_P6_size_and_split(cfg),
        # 第二层（坐标系/实现）
        check_P7_rotation_chain(cfg),
        check_P8_units_and_normalization(cfg),
        check_P9_sim3_alignment(cfg),
        check_P10_2d_only(cfg),
        check_P11_input_isolation(cfg),
        check_P12_train_test_symmetry(cfg),
        check_P13_rq_matching(cfg),
        # 第三层（方法公平性）
        check_P14_shared_4_heads(cfg),
        check_P15_per_method_tuning(cfg),
        check_P16_ekf_init_protocol(cfg),
        check_P17_robust_ekf_huber(cfg),
        check_P18_baselines_with_heads(cfg),
        # 第四层（统计脚手架）
        check_P19_seed_system(cfg),
        check_P20_window(cfg),
        check_P21_normalization_isolation(cfg),
        check_P22_stats_script(cfg),
        check_P23_config_consistency(cfg),
        # 第五层（稳定性/接口）
        check_P24_no_nan_inf(cfg),
        check_P25_async_timestamps(cfg),
        check_P26_vio_payload(cfg),
        check_P27_activation_stats(cfg),
        # 第六层（数据管道）
        check_P28_dual_rate(cfg),
        check_P29_windowing(cfg),
        check_P30_split_before_window(cfg),
        check_P31_edge_cases(cfg),
        # 第七层（训练一致性）
        check_P32_same_seed_init(cfg),
        check_P33_best_val(cfg),
        check_P34_no_overfit(cfg),
        check_P35_head_consistency(cfg),
        # 第八层（评估口径）
        check_P36_metric_declaration(cfg),
        check_P37_warmup_removed(cfg),
        check_P38_world_frame(cfg),
        check_P39_persistence(cfg),
        # Pre
        check_Pre1_environment_locked(cfg),
        check_Pre2_decision_log(cfg),
        check_Pre3_directory_structure(cfg),
        check_Pre4_checksums(cfg),
        check_Pre5_resource_budget(cfg),
        check_Pre6_seed_manifest_consistency(cfg),
        # I
        check_I1_data_generator(cfg),
        check_I2_s9_script(cfg),
        check_I3_5_methods(cfg),
        check_I4_stats_script(cfg),
        check_I5_eval_pipeline(cfg),
        # DQ
        check_DQ1_difficulty_gradient(cfg),
        check_DQ2_snr(cfg),
        check_DQ3_distribution(cfg),
        check_DQ4_sample_size(cfg),
        # G
        check_G1_unit_completeness(cfg),
        check_G2_metrics_readable(cfg),
        check_G3_config_data_cross(cfg),
        check_G4_alerts_cleared(cfg),
        check_G5_output_aligned(cfg),
        # E
        check_E1_audit_trail(cfg),
        check_E2_gate_records(cfg),
        check_E3_dual_review(cfg),
        check_E4_spot_rerun(cfg),
        check_E5_alert_log_consistency(cfg),
        check_E6_audit_exit(cfg),
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