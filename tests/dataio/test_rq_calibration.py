"""R/Q 标定模块测试 — 构造数据单测（[138] 硬约束验证）。

覆盖点：
- calibrate_rq() 返回字典含所有必需字段
- R 标定值在目标 σ≈0.6m 的 ±25% 容差内（构造数据确定性）
- Q 标定值在 [0.01, 0.03] m/步 范围内
- write_rq_calib_to_manifest() 正确写入 manifest
- print_rq_calib_report() 不抛出
- 不同 seed 产生确定性结果（可复现）
- n_samples >= 300 时判定通过性
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from liquidloc.dataio.rq_calibration import (
    R_TARGET_LOS_STD_M,
    R_TOLERANCE_RATIO,
    Q_TARGET_VIO_MIN_M,
    Q_TARGET_VIO_MAX_M,
    calibrate_rq,
    write_rq_calib_to_manifest,
    print_rq_calib_report,
)


# ─── 字段完整性 ─────────────────────────────────────────────────────────────

REQUIRED_FIELDS = [
    "r_uwb_std_m",
    "r_uwb_var_m2",
    "q_vio_dx_std_m",
    "q_vio_dy_std_m",
    "q_vio_combined_std_m",
    "sample_count",
    "los_level",
    "vio_level",
    "r_target_m",
    "q_target_range",
    "r_within_tolerance",
    "q_within_tolerance",
]


def test_calibrate_rq_returns_all_fields() -> None:
    """标定结果必须包含全部必需字段。"""
    result = calibrate_rq(n_samples=300, seed=20260605)
    for field_name in REQUIRED_FIELDS:
        assert field_name in result, f"缺失字段: {field_name}"


def test_calibrate_rq_los_level_is_n0() -> None:
    """R 标定使用 N0（LOS）档位。"""
    result = calibrate_rq()
    assert result["los_level"] == "N0"


def test_calibrate_rq_vio_level_is_v0() -> None:
    """Q 标定使用 V0 档位。"""
    result = calibrate_rq()
    assert result["vio_level"] == "V0"


# ─── R 标定值正确性 ─────────────────────────────────────────────────────────

def test_r_std_within_target_tolerance() -> None:
    """R 标定 σ 必须在目标 0.6m 的 ±25% 容差内。"""
    result = calibrate_rq(n_samples=300, seed=20260605)
    r_std = result["r_uwb_std_m"]
    tolerance = R_TOLERANCE_RATIO * R_TARGET_LOS_STD_M
    assert R_TARGET_LOS_STD_M - tolerance <= r_std <= R_TARGET_LOS_STD_M + tolerance, (
        f"R 标定值 {r_std:.4f} m 超出容差 [{R_TARGET_LOS_STD_M - tolerance:.4f}, "
        f"{R_TARGET_LOS_STD_M + tolerance:.4f}]"
    )


def test_r_within_tolerance_flag_is_true() -> None:
    """r_within_tolerance 标志必须为 True。"""
    result = calibrate_rq()
    assert result["r_within_tolerance"] is True


def test_r_var_matches_std_squared() -> None:
    """R² 应等于 σ²（允许 rounding 误差）。"""
    result = calibrate_rq()
    # 因 round 到 6 位小数，允许 1e-5 容差
    assert abs(result["r_uwb_var_m2"] - result["r_uwb_std_m"] ** 2) < 1e-5


# ─── Q 标定值正确性 ─────────────────────────────────────────────────────────

def test_q_combined_std_in_range() -> None:
    """Q 标定合成 σ 必须在 [0.01, 0.03] m/步 范围内（硬约束）。"""
    result = calibrate_rq(n_samples=1000, seed=20260605)
    q_std = result["q_vio_combined_std_m"]
    assert Q_TARGET_VIO_MIN_M <= q_std <= Q_TARGET_VIO_MAX_M, (
        f"Q 标定值 {q_std:.4f} m/步 超出硬约束范围 [{Q_TARGET_VIO_MIN_M}, {Q_TARGET_VIO_MAX_M}]"
    )


def test_q_within_tolerance_flag_is_true() -> None:
    """q_within_tolerance 标志必须为 True。"""
    result = calibrate_rq()
    assert result["q_within_tolerance"] is True


def test_q_dx_dy_individual_stds_in_range() -> None:
    """VIO dx/dy 各自 σ 必须在 [0.01, 0.03] m/步 范围内（硬约束）。"""
    result = calibrate_rq()
    assert Q_TARGET_VIO_MIN_M <= result["q_vio_dx_std_m"] <= Q_TARGET_VIO_MAX_M, (
        f"Q dx σ={result['q_vio_dx_std_m']:.4f} 超出硬约束 [{Q_TARGET_VIO_MIN_M}, {Q_TARGET_VIO_MAX_M}]"
    )
    assert Q_TARGET_VIO_MIN_M <= result["q_vio_dy_std_m"] <= Q_TARGET_VIO_MAX_M, (
        f"Q dy σ={result['q_vio_dy_std_m']:.4f} 超出硬约束 [{Q_TARGET_VIO_MIN_M}, {Q_TARGET_VIO_MAX_M}]"
    )


# ─── 可复现性 ────────────────────────────────────────────────────────────────

def test_calibrate_rq_deterministic_same_seed() -> None:
    """相同 seed 产生完全相同结果。"""
    r1 = calibrate_rq(n_samples=300, seed=20260605)
    r2 = calibrate_rq(n_samples=300, seed=20260605)
    assert r1 == r2


def test_calibrate_rq_different_seed_different_values() -> None:
    """不同 seed 的样本值不同（但判定仍应通过）。"""
    r1 = calibrate_rq(n_samples=300, seed=20260605)
    r2 = calibrate_rq(n_samples=300, seed=99999999)
    # std 值可能因采样差异而不同
    assert r1["r_uwb_std_m"] != r2["r_uwb_std_m"] or r1["q_vio_combined_std_m"] != r2["q_vio_combined_std_m"]
    # 但判定标志必须仍为 True
    assert r2["r_within_tolerance"] is True
    assert r2["q_within_tolerance"] is True


# ─── Manifest 写入 ──────────────────────────────────────────────────────────

def test_write_rq_calib_to_manifest() -> None:
    """write_rq_calib_to_manifest 将结果写入 manifest['rq_calib']。"""
    manifest: dict = {}
    result = calibrate_rq()
    written = write_rq_calib_to_manifest(result, manifest)
    assert "rq_calib" in written
    assert written["rq_calib"] == result


def test_write_rq_calib_overwrites_existing() -> None:
    """多次写入时 manifest['rq_calib'] 被覆盖为最新值。"""
    manifest: dict = {"rq_calib": {"old": True}}
    result = calibrate_rq()
    write_rq_calib_to_manifest(result, manifest)
    assert manifest["rq_calib"] == result
    assert "old" not in manifest["rq_calib"]


# ─── 冒烟打印 ───────────────────────────────────────────────────────────────

def test_print_rq_calib_report_no_exception() -> None:
    """print_rq_calib_report 不应抛出异常。"""
    result = calibrate_rq()
    print_rq_calib_report(result)  # 直接调用，捕获输出


# ─── 样本数鲁棒性 ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_samples", [300, 500, 1000, 2000])
def test_calibrate_rq_robust_to_sample_count(n_samples: int) -> None:
    """不同样本数下标定结果仍应通过判定（n >= 300 且 sigma 有 +15% 余量）。"""
    result = calibrate_rq(n_samples=n_samples, seed=20260605)
    assert result["r_within_tolerance"] is True
    assert result["q_within_tolerance"] is True
    assert result["sample_count"] == n_samples


def test_calibrate_rq_small_n_raises() -> None:
    """n_samples < 300 时应抛出 ValueError（样本不足导致 Q 标定不稳定）。"""
    with pytest.raises(ValueError, match="n_samples 必须 >= 300"):
        calibrate_rq(n_samples=100, seed=20260605)


# ─── 集成：完整流程（标定 → 写入 manifest → 验证） ────────────────────────

def test_full_calibration_pipeline() -> None:
    """完整标定流程：生成 → 判定 → 写入 → 验证。"""
    # 1. 标定（n_samples=1000 确保 Q 稳定落在 [0.01, 0.03] 内）
    result = calibrate_rq(n_samples=1000, seed=20260605)

    # 2. 写入 manifest
    manifest: dict[str, Any] = {"version": "1.0"}
    write_rq_calib_to_manifest(result, manifest)

    # 3. 验证
    assert manifest["rq_calib"]["r_within_tolerance"] is True
    assert manifest["rq_calib"]["q_within_tolerance"] is True
    assert manifest["rq_calib"]["los_level"] == "N0"
    assert manifest["rq_calib"]["vio_level"] == "V0"
    assert manifest["rq_calib"]["sample_count"] == 1000
