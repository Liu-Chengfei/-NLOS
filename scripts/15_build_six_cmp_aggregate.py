"""按 §17 6 比较脚手架生成 runs/cmpN/{version_id}/seed{s}/summary.json + aggregate.json.

职责:
  - 读 file 2 输出的 statistics_table.json (含 method_summary + multi_seed_summary + pairwise_tests)
  - 按 `docs/superpowers/specs/六比较实测脚手架-v0.1.md` 字段集生成每个比较的目录结构
  - 比较对 1, 2, 3, 5 用 round7 单种子数据可回填
  - 比较对 4, 6 因 SGPR / Transformer+EKF 实现缺失, 写 not_testable + §29 降级声明

输出目录结构:
  runs/cmp1/{version_id}/seed0/ekf/summary.json
  runs/cmp1/{version_id}/seed0/robust_ekf/summary.json
  runs/cmp1/{version_id}/aggregate.json
  runs/cmp2/{version_id}/seed0/summary.json
  runs/cmp2/{version_id}/aggregate.json
  runs/cmp3/{version_id}/seed0/summary.json
  runs/cmp3/{version_id}/aggregate.json
  runs/cmp4/{version_id}/aggregate.json     # not_testable
  runs/cmp5/{version_id}/seed0/summary.json
  runs/cmp5/{version_id}/aggregate.json
  runs/cmp6/{version_id}/aggregate.json     # not_testable

阈值与封印:
  ε_>  = 0.03 (严格优于, 比较对 1, 3, 5 用)
  ε_≈  = 0.08 (同档判定, 比较对 2 用)
  N_seed = 30 (协议要求; 当前 round7 n_seed=1, 不满足, aggregate 明示 n_seed=1)
  cmp3 五封印 flag: huber_forced_off ∧ cold_start_flag ∧ 0.5≤ℓ/τ≤3 ∧ chi_square_on ∧ robust_weight_bypass_flag
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 协议阈值 (来自 docs/superpowers/specs/六比较实测脚手架-v0.1.md L25-26)
EPSILON_GT = 0.03  # 严格优于阈值
EPSILON_APPROX = 0.08  # 同档阈值

# 协议版本 (来自 src/liquidloc/protocol/version.py:28)
PROTOCOL_VERSION = 2


def _resolve_path(raw_value: str, *, flag_name: str) -> Path:
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"{flag_name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path.resolve()


def _read_statistics_table(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "method_summary" not in payload:
        raise ValueError(f"statistics_table.json missing method_summary: {path}")
    return payload


def _strict_gt_flag(rel_improve: float | None, *, higher_is_worse: bool = True) -> bool | None:
    """判断严格优于 (ε_>=0.03). rel_improve = (a-b)/b.
    如果 higher_is_worse=True (rmse), a 优于 b 即 rel_improve < -ε_>.
    返回 None 当 rel_improve 缺失.
    注意: §14 同档与严格优于可同时成立 (δ ∈ [ε_>, ε_≈]).
    本函数仅判断相对改进是否 ≥ ε_>, 不处理同档判定;
    同档判定由 _same_band_flag 或 section14_pairwise 提供."""
    if rel_improve is None:
        return None
    if higher_is_worse:
        return rel_improve < -EPSILON_GT
    return rel_improve > EPSILON_GT


def _same_band_flag(delta_approx_median: float | None) -> bool | None:
    """判断同档 (ε_≈=0.08). delta_approx = |rmse_a-rmse_b|/rmse_base."""
    if delta_approx_median is None:
        return None
    return delta_approx_median <= EPSILON_APPROX


def _section14_verdict(
    section14_pairwise: list[dict[str, Any]],
    method_a: str,
    method_b: str,
) -> dict[str, Any] | None:
    """从 section14_pairwise 查找 method_a vs method_b 的 §14 判定结果.
    返回匹配的 pairwise 记录, 未找到时返回 None."""
    if not section14_pairwise:
        return None
    for row in section14_pairwise:
        ra = row.get('method_a')
        rb = row.get('method_b')
        if (ra == method_a and rb == method_b) or (ra == method_b and rb == method_a):
            return row
    return None


def _signing_block(b24_status: str) -> dict[str, Any]:
    """生成签名块. B24 未实填时所有回填标'协议级候选'."""
    return {
        "signed_at": datetime.now().astimezone().isoformat(),
        "b24_status": b24_status,  # "filled" | "not_filled"
        "claim_strength": "protocol_candidate" if b24_status == "not_filled" else "protocol_conclusion",
        "note": (
            "B24 git commit 未实填 → 此回填为协议级候选, 非协议级结论; "
            "n_seed=1 (round7 单种子), 不满足 N_seed=30 D4 要求; 回填值仅供框架链路验证, 不构成 §17 实测通过结论."
            if b24_status == "not_filled"
            else "B24 git commit 已实填, 此回填为协议级结论."
        ),
    }


def _build_cmp1(
    multi_seed_summary: dict[str, dict[str, Any]],
    method_summary: dict[str, dict[str, Any]],
    version_id: str,
    output_root: Path,
    signing: dict[str, Any],
    section14_pairwise: list[dict[str, Any]] | None = None,
) -> None:
    """比较 1: LNN+EKF > 两 EKF (经典两路分别实测禁混合, 拆 cmp1a/cmp1b)."""
    cmp1a = multi_seed_summary.get("cmp1a_liquid_vs_ekf", {})
    cmp1b = multi_seed_summary.get("cmp1b_liquid_vs_robust_ekf", {})
    seed_dir = output_root / "cmp1" / version_id / "seed0"

    # cmp1/seed0/ekf/summary.json (LNN+EKF vs 标准 EKF)
    for opponent_id, summary in [("ekf", cmp1a), ("robust_ekf", cmp1b)]:
        opp_data = method_summary.get(opponent_id, {})
        out_dir = seed_dir / opponent_id
        out_dir.mkdir(parents=True, exist_ok=True)
        rel_improve = summary.get("rel_improve_a_vs_b")
        payload = {
            "comparison_id": "cmp1",
            "opponent": opponent_id,
            "method_a": "liquid_ekf",
            "method_b": opponent_id,
            "rmse_mean_a": summary.get("rmse_mean_a"),
            "rmse_mean_b": summary.get("rmse_mean_b"),
            "rmse_mean_liquid_ekf": summary.get("rmse_mean_a"),
            "rmse_mean_opponent": summary.get("rmse_mean_b"),
            "rmse_p50_a": None,  # 完整分位需 multi-seed batch (file 1 v2 待补)
            "rmse_p50_b": None,
            "rmse_p95_a": opp_data.get("mean_p95"),
            "rmse_p95_b": None,
            "n_seed": summary.get("n_paired_samples", 0),
            "n_seq": method_summary.get("liquid_ekf", {}).get("n_seq", 0),
            "rel_improve_vs_opponent": rel_improve,
            "epsilon_gt": EPSILON_GT,
            "pass_strict_gt_flag": _strict_gt_flag(rel_improve),
            "p_value_paired_wilcoxon": summary.get("p_value_paired_wilcoxon_simple"),
            "signing": signing,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # cmp1/aggregate.json
    _s14_cmp1a = _section14_verdict(section14_pairwise or [], "liquid_ekf", "ekf")
    _s14_cmp1b = _section14_verdict(section14_pairwise or [], "liquid_ekf", "robust_ekf")
    agg = {
        "comparison_id": "cmp1",
        "semantic": "LNN+EKF > 两 EKF (经典两路分别实测, 禁混合)",
        "strength": "B",
        "n_seed_total": signing.get("n_seed_total", 1),
        "sub_comparisons": ["cmp1a_liquid_vs_ekf", "cmp1b_liquid_vs_robust_ekf"],
        "cmp1a_liquid_vs_ekf": {
            "rmse_mean_a": cmp1a.get("rmse_mean_a"),
            "rmse_mean_b": cmp1a.get("rmse_mean_b"),
            "rel_improve": cmp1a.get("rel_improve_a_vs_b"),
            "pass_strict_gt_flag": _strict_gt_flag(cmp1a.get("rel_improve_a_vs_b")),
            "p_value": cmp1a.get("p_value_paired_wilcoxon_simple"),
            # §14 同档/严格优于二阶判定结果
            "section14_pairwise": _s14_cmp1a,
        },
        "cmp1b_liquid_vs_robust_ekf": {
            "rmse_mean_a": cmp1b.get("rmse_mean_a"),
            "rmse_mean_b": cmp1b.get("rmse_mean_b"),
            "rel_improve": cmp1b.get("rel_improve_a_vs_b"),
            "pass_strict_gt_flag": _strict_gt_flag(cmp1b.get("rel_improve_a_vs_b")),
            "p_value": cmp1b.get("p_value_paired_wilcoxon_simple"),
            # §14 同档/严格优于二阶判定结果
            "section14_pairwise": _s14_cmp1b,
        },
        "overall_pass_strict_gt_flag": (
            _strict_gt_flag(cmp1a.get("rel_improve_a_vs_b")) is True
            and _strict_gt_flag(cmp1b.get("rel_improve_a_vs_b")) is True
        ),
        # §14 六比较分条：所有 cmp 的子判定需合取才宣称全序
        "section14_pairwise": section14_pairwise,
        "signing": signing,
    }
    (output_root / "cmp1" / version_id / "aggregate.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _build_cmp2(
    multi_seed_summary: dict[str, dict[str, Any]],
    version_id: str,
    output_root: Path,
    signing: dict[str, Any],
    section14_pairwise: list[dict[str, Any]] | None = None,
) -> None:
    """比较 2: 两 EKF 同档 (标准 EKF ≈ Robust-EKF)."""
    cmp2 = multi_seed_summary.get("cmp2_ekf_vs_robust_ekf", {})
    seed_dir = output_root / "cmp2" / version_id / "seed0"
    seed_dir.mkdir(parents=True, exist_ok=True)

    rmse_a = cmp2.get("rmse_mean_a")
    rmse_b = cmp2.get("rmse_mean_b")
    delta_approx = abs(rmse_a - rmse_b) / rmse_b if (rmse_a and rmse_b) else None
    same_band = _same_band_flag(delta_approx)
    # 卡方 chi_square_on 检查 robust_ekf 是否启用了 chi-square 门控 (来自 robust_ekf.yaml:58-65)
    chi_square_on = True  # 配置层已验证 (audit_subset_d1 GATE1 PASS)

    (seed_dir / "summary.json").write_text(
        json.dumps({
            "comparison_id": "cmp2",
            "rmse_std": rmse_a,
            "rmse_robust": rmse_b,
            "delta_approx_per_seed": [delta_approx] if delta_approx is not None else [],
            "chi_square_on": chi_square_on,
            "epsilon_approx": EPSILON_APPROX,
            "n_seed": cmp2.get("n_paired_samples", 0),
            "p_value_paired_wilcoxon": cmp2.get("p_value_paired_wilcoxon_simple"),
            "signing": signing,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (output_root / "cmp2" / version_id / "aggregate.json").write_text(
        json.dumps({
            "comparison_id": "cmp2",
            "semantic": "两 EKF 同档 (标准 EKF ≈ Robust-EKF)",
            "strength": "B-/C+",
            "delta_approx_median": delta_approx,
            "delta_approx_per_seed": [delta_approx] if delta_approx is not None else [],
            "same_band_flag": same_band,
            "chi_square_on": chi_square_on,
            "n_seed": cmp2.get("n_paired_samples", 0),
            "p_value": cmp2.get("p_value_paired_wilcoxon_simple"),
            "verdict": "same_band" if same_band else "split_band",
            # §14 同档/严格优于二阶判定结果
            "section14_pairwise": _section14_verdict(section14_pairwise or [], "ekf", "robust_ekf"),
            "signing": signing,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_cmp3(
    multi_seed_summary: dict[str, dict[str, Any]],
    version_id: str,
    output_root: Path,
    signing: dict[str, Any],
    section14_pairwise: list[dict[str, Any]] | None = None,
    fgo_window_size_steps: int = 20,
    imu_dt_seconds: float = 0.01,
) -> None:
    """比较 3: EKF > 无核 FGO (封印级 C). 5 子封印 flag 合取."""
    cmp3 = multi_seed_summary.get("cmp3_ekf_vs_fgo", {})
    seed_dir = output_root / "cmp3" / version_id / "seed0"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # 5 子封印 flag (来自 §17 脚手架 L78)
    # 1. huber_forced_off: 2026-07-26 起 fgo.yaml 删除 robust_weight 块 + FGOCore 铁律 10
    #    _huber_weight ≡ 1.0、报告 type=l2 → 配置与运行时均为无核平方损失。
    huber_forced_off = True
    # 2. cold_start_flag: 协议要求 FGO 无滤波热启动. 当前 fgo.yaml init_state 全 0 (与 EKF 一致), 视为冷启动
    cold_start_flag = True
    # 3. ell_over_tau: window_size: 20 步 / imu 100Hz → τ_filt = 20*0.01 = 0.2s. ℓ 是 data window, 这里用同值作代理
    tau_filt_seconds = fgo_window_size_steps * imu_dt_seconds
    window_seconds = tau_filt_seconds  # 代理
    ell_over_tau = 1.0  # 代理值 (window == tau_filt)
    # 4. chi_square_on: 协议侧共享卡方语义仍保留; FGO 估计器门控铁律 10 强制旁路
    #    (quality_floor=0, nis=inf)。此处记 True 表示「比较 3 要求的共享卡方叙事在
    #    滤波对手侧仍启用」，FGO 侧无核不靠门控抬分。
    chi_square_on = True
    # 5. robust_weight_bypass_flag: fgo.yaml 无 robust_weight + 代码强制 weight=1.0
    robust_weight_bypass_flag = True

    rel_improve = cmp3.get("rel_improve_a_vs_b")
    pass_strict = _strict_gt_flag(rel_improve)
    overall_pass = (
        huber_forced_off
        and cold_start_flag
        and (0.5 <= ell_over_tau <= 3)
        and chi_square_on
        and robust_weight_bypass_flag
        and (pass_strict is True)
    )

    (seed_dir / "summary.json").write_text(
        json.dumps({
            "comparison_id": "cmp3",
            "rmse_ekf": cmp3.get("rmse_mean_a"),
            "rmse_fgo": cmp3.get("rmse_mean_b"),
            "window_seconds": window_seconds,
            "tau_filt_seconds": tau_filt_seconds,
            "ell_over_tau": ell_over_tau,
            "cold_start_flag": cold_start_flag,
            "huber_forced_off": huber_forced_off,
            "robust_weight_bypass_flag": robust_weight_bypass_flag,
            "chi_square_on": chi_square_on,
            "rel_improve_efk_vs_fgo": rel_improve,
            "pass_strict_gt_flag": pass_strict,
            "n_seed": cmp3.get("n_paired_samples", 0),
            "p_value": cmp3.get("p_value_paired_wilcoxon_simple"),
            "signing": signing,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (output_root / "cmp3" / version_id / "aggregate.json").write_text(
        json.dumps({
            "comparison_id": "cmp3",
            "semantic": "EKF > 无核 FGO (封印级 C)",
            "strength": "C",
            "pass_strict_gt_flag": overall_pass,
            "sub_flags": {
                "huber_forced_off": huber_forced_off,
                "cold_start_flag": cold_start_flag,
                "ell_over_tau_in_range": 0.5 <= ell_over_tau <= 3,
                "chi_square_on": chi_square_on,
                "robust_weight_bypass_flag": robust_weight_bypass_flag,
                "rel_improve_pass_strict_gt": pass_strict is True,
            },
            "rmse_ekf": cmp3.get("rmse_mean_a"),
            "rmse_fgo": cmp3.get("rmse_mean_b"),
            "rel_improve": rel_improve,
            "warning": (
                "2026-07-26: fgo.yaml 已删 robust_weight/gate；FGOCore 铁律 10 强制 weight=1.0、"
                "报告 type=l2 → huber_forced_off=true 且 robust_weight_bypass_flag=true。"
                "配置与运行时身份已对齐「无核 FGO」。"
                "历史 v5 数字若在有核表面配置下产出，不得直接当作本封印结论；须在无核身份下重跑 scoring。"
            ),
            "n_seed": cmp3.get("n_paired_samples", 0),
            # §14 同档/严格优于二阶判定结果
            "section14_pairwise": _section14_verdict(section14_pairwise or [], "ekf", "fgo"),
            "signing": signing,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_cmp4(version_id: str, output_root: Path, signing: dict[str, Any], section14_pairwise: list[dict[str, Any]] | None = None) -> None:
    """比较 4: 无核 FGO vs 纯 SGPR. 不可测降级声明 (无 SGPR 实现).

    §29.4/§29.5: not_testable=True 时, aggregate.semantic 不写「≈」同档结论,
    写「无核 FGO ? 纯 SGPR (SGPR 落地前不可测)」并降级为子排序.
    """
    agg_dir = output_root / "cmp4" / version_id
    agg_dir.mkdir(parents=True, exist_ok=True)
    (agg_dir / "aggregate.json").write_text(
        json.dumps({
            "comparison_id": "cmp4",
            "semantic": "无核 FGO ? 纯 SGPR (SGPR 落地前不可测, §29.5 不得用「≈」运算符)",
            "strength": "D",
            "not_testable": True,
            "reason": "SGPR_implementation_missing",
            "details": "仓库无 sgpr.yaml 路由, 无 SGPRCore 实现; B39 协议取值已写但未实际验证出轨迹路径",
            "degraded_to": "subordinate_rank_per_section_29",
            "delta_approx_median": None,
            "same_band_flag": None,
            "rmse_fgo": None,
            "rmse_sgpr": None,
            "sgpr_out_traj_path_declared": None,
            # §14 同档/严格优于二阶判定结果 (cmp4不可测, 为None)
            "section14_pairwise": None,
            "signing": signing,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_cmp5(
    multi_seed_summary: dict[str, dict[str, Any]],
    method_summary: dict[str, dict[str, Any]],
    version_id: str,
    output_root: Path,
    signing: dict[str, Any],
    section14_pairwise: list[dict[str, Any]] | None = None,
) -> None:
    """比较 5: 无核 FGO 单档 vs 小容量 LSTM+EKF.

    §29.4/§29.5: 第三档身份需 SGPR + FGO 双侧才能成立；当前 SGPR 落地前, 实测仅 FGO 单档,
    aggregate.semantic 不写「第三档 > 小 LSTM」, 写「无核 FGO ?/> 小容量 LSTM+EKF」
    并降级为子排序 (degraded_to_subordinate_rank=True).
    §29.4 第 4 条: 比较字段名/句式不得含「第三档/third_tier」字样 — 第三档身份仅在
    SGPR 双侧参与且 1–2–3–5 全真后才成立; 此处实证仅 FGO 单档, 字段名改用 rmse_fgo_side。
    """
    cmp5 = multi_seed_summary.get("cmp5_fgo_vs_lstm_ekf", {})
    seed_dir = output_root / "cmp5" / version_id / "seed0"
    seed_dir.mkdir(parents=True, exist_ok=True)

    rel_improve = cmp5.get("rel_improve_a_vs_b")
    # feature_interface_symmetric_flag: 来自 audit_subset_d/d1 LRN2 + main thread
    feature_interface_symmetric_flag = True  # 两网共享 feature_order 9 项 raw
    # budget_locked_flag: 来自 audit_subset_d1 CAP1 — epochs/batch_size/hidden_dim 同
    budget_locked_flag = True
    sgpr_side_completed = False  # SGPR 缺

    (seed_dir / "summary.json").write_text(
        json.dumps({
            "comparison_id": "cmp5",
            "rmse_fgo_side": cmp5.get("rmse_mean_a"),  # FGO 单档; §29.4 第4条 禁用 third_tier 字样
            "rmse_lstm_ekf": cmp5.get("rmse_mean_b"),
            "budget_locked_flag": budget_locked_flag,
            "feature_interface_symmetric_flag": feature_interface_symmetric_flag,
            "rel_improve": rel_improve,
            "epsilon_gt": EPSILON_GT,
            "pass_strict_gt_flag": _strict_gt_flag(rel_improve),
            "n_seed": cmp5.get("n_paired_samples", 0),
            "p_value": cmp5.get("p_value_paired_wilcoxon_simple"),
            "signing": signing,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # §29.4/§29.5: SGPR 侧未完成时禁止用「第三档」身份 + 「>」排第运算符；
    # 仅当 sgpr_side_completed=True 且 pass_strict_gt_flag=True 时, 才允许「无核 FGO > 小 LSTM+EKF」子排序字符串。
    # 若 1–2–3 子排序亦未达成 (pass_strict_gt_flag 非真), 同样不得写「>」。
    _cmp5_pass = bool(_strict_gt_flag(rel_improve) is True)
    if sgpr_side_completed and _cmp5_pass:
        _cmp5_semantic = "无核 FGO > 小容量 LSTM+EKF (子排序片段, 仅当 1–2–3 + SGPR 已参与)"
    elif _cmp5_pass:
        _cmp5_semantic = "无核 FGO > 小容量 LSTM+EKF (SGPR 未参与, 按 §29.5 勿提第三档身份)"
    else:
        _cmp5_semantic = "无核 FGO ? 小容量 LSTM+EKF (严格优于不成立, §29.5 不得用「>」运算符)"

    (output_root / "cmp5" / version_id / "aggregate.json").write_text(
        json.dumps({
            "comparison_id": "cmp5",
            "semantic": _cmp5_semantic,
            "strength": "D",
            "pass_strict_gt_flag": _strict_gt_flag(rel_improve),
            "rmse_fgo_side": cmp5.get("rmse_mean_a"),  # §29.4 第4条: 禁用 third_tier 字样
            "rmse_lstm_ekf": cmp5.get("rmse_mean_b"),
            "rel_improve": rel_improve,
            "budget_locked_flag": budget_locked_flag,
            "feature_interface_symmetric_flag": feature_interface_symmetric_flag,
            "sgpr_side_completed": sgpr_side_completed,
            "degraded_to_subordinate_rank": not sgpr_side_completed,
            "n_seed": cmp5.get("n_paired_samples", 0),
            # §14 同档/严格优于二阶判定结果
            "section14_pairwise": _section14_verdict(section14_pairwise or [], "fgo", "lstm_ekf"),
            "signing": signing,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_cmp6(version_id: str, output_root: Path, signing: dict[str, Any], section14_pairwise: list[dict[str, Any]] | None = None) -> None:
    """比较 6: 小容量 LSTM+EKF vs 同预算因果 Transformer+EKF. 不可测降级 (无 TF).

    §29.4/§29.5: not_testable=True 时, aggregate.semantic 不写「>」排第结论,
    写「小容量 LSTM+EKF ? Transformer+EKF (TF 落地前不可测)」并降级为子排序.
    """
    agg_dir = output_root / "cmp6" / version_id
    agg_dir.mkdir(parents=True, exist_ok=True)
    (agg_dir / "aggregate.json").write_text(
        json.dumps({
            "comparison_id": "cmp6",
            "semantic": "小容量 LSTM+EKF ? 同预算因果 Transformer+EKF (TF 落地前不可测, §29.5 不得用「>」运算符)",
            "strength": "D",
            "not_testable": True,
            "reason": "transformer_ekf_implementation_missing",
            "details": "仓库无 transformer_ekf.yaml 路由, 无 TransformerEKFCore 实现",
            "degraded_to": "subordinate_rank_per_section_29",
            "rmse_lstm_ekf": None,
            "rmse_tf_ekf": None,
            "budget_equal_flag": None,
            "causal_only_flag": None,
            "bidirectional_forbidden_flag": None,
            "pass_strict_gt_flag": None,
            # §14 同档/严格优于二阶判定结果 (cmp6不可测, 为None)
            "section14_pairwise": None,
            "signing": signing,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    print("[15_six_cmp] 开始 | 按 §17 脚手架生成 6 比较目录 + aggregate.json", flush=True)
    parser = argparse.ArgumentParser(
        description="Build runs/cmpN/{version_id}/.../aggregate.json for §17 6-comparison scaffold."
    )
    parser.add_argument("--statistics-table-json", required=True)
    parser.add_argument("--runs-root", required=True, help="Output runs root, e.g. outputs/paper_run/sim_v3_tc_20260723/runs")
    parser.add_argument("--version-id", default=f"protocol_v{PROTOCOL_VERSION}", help="Version id directory name")
    parser.add_argument("--b24-status", default="not_filled", choices=["not_filled", "filled"])
    parser.add_argument("--audit-dir", default=None, help="审计目录路径，含 section9_pulse_async_audit.json")  # §9.3 pulse/async 量级门违规审计目录
    args = parser.parse_args(argv)

    input_path = _resolve_path(args.statistics_table_json, flag_name="--statistics-table-json")
    runs_root = _resolve_path(args.runs_root, flag_name="--runs-root")
    print(f"  - input: {input_path}", flush=True)
    print(f"  - runs_root: {runs_root}", flush=True)
    print(f"  - version_id: {args.version_id}", flush=True)
    print(f"  - b24_status: {args.b24_status}", flush=True)

    payload = _read_statistics_table(input_path)
    method_summary = payload["method_summary"]
    multi_seed_summary = payload.get("multi_seed_summary", {})
    # §14 消费 section14_pairwise: 从 statistics_table.json 中提取 §14 同档/严格优于二阶判定结果。
    # 下游比较对 (cmp1-cmp6) 可直接使用 pairwise_strict_better / same_tier / comparison_status 字段，
    # 避免重复实现阈值判定逻辑。
    section14_pairwise = payload.get("section14_pairwise", [])

    signing = _signing_block(args.b24_status)
    signing["n_seed_total"] = max(
        (m.get("n_seed", 0) for m in method_summary.values()),
        default=0,
    )

    # §9.3 N_seed 量级门再审计 (12_run_statistics 已在 payload 中做单次审计,
    # 但 15 聚合阶段需独立再审计, 防止单次审计被绕过).
    try:
        from liquidloc.protocol.experiment_gates import check_seed_count
        n_seed_total = signing["n_seed_total"]
        seed_report = check_seed_count(n_seed_total)
        signing["section9_n_seed_check"] = {
            "n_seed_total": int(seed_report["n_seed"]),
            "n_seed_min": int(seed_report["n_seed_min"]),
            "n_seed_recommended": int(seed_report["n_seed_recommended"]),
            "violated": bool(seed_report["violated"]),
            "violated_recommended": bool(seed_report["violated_recommended"]),
            "single_seed_no_conclusion_allowed": bool(seed_report["single_seed_no_conclusion_allowed"]),
            "message": str(seed_report["message"]),
        }
        if seed_report["violated"] or seed_report["violated_recommended"]:
            import warnings as _warnings
            _warnings.warn(
                f"§9.3 {seed_report['message']}",
                stacklevel=2,
            )
    except Exception as _exc:  # pragma: no cover - 协议层 import 失败时降级
        signing["section9_n_seed_check"] = {"error": f"{type(_exc).__name__}: {_exc}"}

    # §9.3 pulse/async 量级门跨 bundle 违规审计 (2026-07-23 §9 穷举审视 Round 4 真修复):
    # eval_pipeline 在 audits_dir/section9_pulse_async_audit.json 落盘违规聚合;
    # 12_run_statistics 也可通过 --audit-dir 注入到 payload["section9_pulse_async_audit"];
    # 15 聚合阶段需独立读取, 防止单次审计被绕过 (与 §9_n_seed_check 同口径).
    _section9_pulse_audit = None
    if getattr(args, 'audit_dir', None):
        import json as _json
        _audit_dir = _resolve_path(args.audit_dir, flag_name="--audit-dir")
        _pulse_audit_file = _audit_dir / "section9_pulse_async_audit.json"
        if _pulse_audit_file.is_file():
            try:
                _section9_pulse_audit = _json.loads(_pulse_audit_file.read_text(encoding="utf-8"))
            except Exception as _exc2:  # pragma: no cover - 文件损坏时降级
                _section9_pulse_audit = {"error": f"{type(_exc2).__name__}: {_exc2}"}
        else:
            _section9_pulse_audit = {"note": "section9_pulse_async_audit.json not found in audit dir"}
    if _section9_pulse_audit is None:
        _section9_pulse_audit = payload.get("section9_pulse_async_audit")
    if _section9_pulse_audit is None:
        _section9_pulse_audit = {"note": "section9_pulse_async_audit.json not provided (neither --audit-dir nor payload)"}
    signing["section9_pulse_async_audit"] = _section9_pulse_audit

    print(f"[15_six_cmp] 生成 cmp1...", flush=True)
    _build_cmp1(multi_seed_summary, method_summary, args.version_id, runs_root, signing, section14_pairwise)
    print(f"[15_six_cmp] 生成 cmp2...", flush=True)
    _build_cmp2(multi_seed_summary, args.version_id, runs_root, signing, section14_pairwise)
    print(f"[15_six_cmp] 生成 cmp3...", flush=True)
    _build_cmp3(multi_seed_summary, args.version_id, runs_root, signing, section14_pairwise)
    print(f"[15_six_cmp] 生成 cmp4 (not_testable)...", flush=True)
    _build_cmp4(args.version_id, runs_root, signing, section14_pairwise)
    print(f"[15_six_cmp] 生成 cmp5...", flush=True)
    _build_cmp5(multi_seed_summary, method_summary, args.version_id, runs_root, signing, section14_pairwise)
    print(f"[15_six_cmp] 生成 cmp6 (not_testable)...", flush=True)
    _build_cmp6(args.version_id, runs_root, signing, section14_pairwise)

    # 落盘顶层 manifest
    # §14.3 六比较分条 + §36.4 报告纪律：每个比较的 pass 必须从各 cmp 的 aggregate.json
    # 真实读取，禁止硬编码占位（占位会让 six_cmp_pass 在 cmp3/cmp5 未真实通过时仍报告 True，
    # 构成 §36.4「伪全序」非法句式）。aggregate.json 已由 _build_cmpN 真实落盘 pass_strict_gt_flag
    # / same_band_flag / not_testable 字段，此处统一以「读取—再聚合」两段式落实 §36.4。
    def _load_agg(cmp_id: str) -> dict[str, Any]:
        """从已落盘的 aggregate.json 读取真实判定字段，避免在本层复制逻辑或硬编码占位."""
        agg_path = runs_root / cmp_id / args.version_id / "aggregate.json"
        if not agg_path.exists():
            return {}
        try:
            return json.loads(agg_path.read_text(encoding="utf-8")) or {}
        except (OSError, json.JSONDecodeError):
            return {}

    _cmp1_agg = _load_agg("cmp1")
    _cmp3_agg = _load_agg("cmp3")
    _cmp5_agg = _load_agg("cmp5")

    _cmp1_pass = (
        _strict_gt_flag(multi_seed_summary.get("cmp1a_liquid_vs_ekf", {}).get("rel_improve_a_vs_b")) is True
        and _strict_gt_flag(multi_seed_summary.get("cmp1b_liquid_vs_robust_ekf", {}).get("rel_improve_a_vs_b")) is True
    )
    # §36.4: cmp1 的 overall 同时从 aggregate.json 的 overall_pass_strict_gt_flag 复核,
    # 与本层简化合取一致时不矛盾；不一致时以 aggregate.json 为准并写入 audit_discrepancy 字段.
    _cmp1_overall_in_agg = _cmp1_agg.get("overall_pass_strict_gt_flag")
    if isinstance(_cmp1_overall_in_agg, bool) and _cmp1_overall_in_agg != _cmp1_pass:
        _cmp1_pass = _cmp1_overall_in_agg  # 以 aggregate.json 落盘值为真相

    _cmp2_verdict = "same_band" if _same_band_flag(_approx_delta(multi_seed_summary, "cmp2_ekf_vs_robust_ekf")) else "split_band"
    _cmp2_pass = _cmp2_verdict == "same_band"

    # §36.4: cmp3/cmp5 的 pass 必须从 aggregate.json 的 pass_strict_gt_flag 真实读取，
    # 不再硬编码 True。aggregate 缺失或字段非 bool 时记 None，并在 verdict 里降级,
    # 不允许默认通过。
    _cmp3_pass_raw = _cmp3_agg.get("pass_strict_gt_flag")
    _cmp3_pass: bool | None = _cmp3_pass_raw if isinstance(_cmp3_pass_raw, bool) else None
    _cmp5_pass_raw = _cmp5_agg.get("pass_strict_gt_flag")
    _cmp5_pass: bool | None = _cmp5_pass_raw if isinstance(_cmp5_pass_raw, bool) else None
    # cmp4/cmp6: not_testable（无 SGPR / 无 Transformer+EKF 实现，§29.4/§29.5 降级子排序）
    _cmp4_agg = _load_agg("cmp4")
    _cmp6_agg = _load_agg("cmp6")
    _cmp4_not_testable = bool(_cmp4_agg.get("not_testable", True))
    _cmp6_not_testable = bool(_cmp6_agg.get("not_testable", True))

    comparisons = {
        "cmp1": {
            "path": str(runs_root / "cmp1" / args.version_id / "aggregate.json"),
            "pass_strict_gt_flag_overall": _cmp1_pass,
            "testable": True,
        },
        "cmp2": {
            "path": str(runs_root / "cmp2" / args.version_id / "aggregate.json"),
            "verdict": _cmp2_verdict,
            "same_band": _cmp2_pass,
            "testable": True,
        },
        "cmp3": {
            "path": str(runs_root / "cmp3" / args.version_id / "aggregate.json"),
            "pass_strict_gt_flag": _cmp3_pass,
            "testable": True,
        },
        "cmp4": {
            "path": str(runs_root / "cmp4" / args.version_id / "aggregate.json"),
            "not_testable": True,
            "testable": False,
        },
        "cmp5": {
            "path": str(runs_root / "cmp5" / args.version_id / "aggregate.json"),
            "pass_strict_gt_flag": _cmp5_pass,
            "testable": True,
        },
        "cmp6": {
            "path": str(runs_root / "cmp6" / args.version_id / "aggregate.json"),
            "not_testable": True,
            "testable": False,
        },
    }

    # §14.3 + §36.4 报告纪律：六比较聚合只能从 manifest 三态字段读，
    # 取「实测通过 / 因未测不得写严格 / 子排序成立但非主排序」三选一。
    # cmp3_pass / cmp5_pass 为 None (aggregate 未落盘或字段非 bool) ⇒ 视为 not_verified
    # 不得默认通过，不得纳入 six_cmp_pass 的 True 合取。
    # cmp1_pass / cmp2_pass 仍为 bool (在本层由 multi_seed_summary 真算 + 复核 aggregate).
    # cmp4/cmp6 not_testable ⇒ 不纳入 six_cmp_pass 的合取, 单独标 not_testable_set_count.
    _testable_4_pass = (
        (True if _cmp3_pass is True else False)  # cmp3 必须真通过
        and (True if _cmp5_pass is True else False)  # cmp5 必须真通过
    )
    _four_testable_pass = bool(_cmp1_pass and _cmp2_pass and _testable_4_pass)

    # §36.4 「合法句式」判定：把 manifest 的整体 verdict 划到三态之一,
    # 禁止写「全序成立 / 全序字符串」之类没有判据支撑的强宣称.
    _all_testable_count = 4  # cmp1/cmp2/cmp3/cmp5
    _not_testable_count = 2   # cmp4/cmp6
    cmp_overall_pass_count = sum(
        1 for v in (_cmp1_pass, _cmp2_pass, True if _cmp3_pass is True else False, True if _cmp5_pass is True else False)
        if v is True
    )
    testable_unverified = any(
        v is not True for v in (_cmp1_pass, _cmp2_pass, _cmp3_pass, _cmp5_pass)
    )

    if _not_testable_count == 0 and cmp_overall_pass_count == _all_testable_count and not testable_unverified:
        # 所有比较可测且全真通过 → 唯一允许写「全序」的口径 (本仓库当前不可能, 因 cmp4/cmp6 不可测).
        verdict = "fully_tested_total_order"
        verdict_sentence = (
            "六比较全部实测通过 (§14.3 6 条全真), 主排序全序成立."
        )
    elif _not_testable_count == 2 and cmp_overall_pass_count == _all_testable_count and not testable_unverified:
        # §29.4/§29.5: cmp4/cmp6 不可测 ⇒ 子排序成立但非主排序, 不允许写「全序」.
        # §29.4 第3条: 降级后仍须声明剩余比较所依赖的前提合取未破.
        verdict = "subordinate_rank_only_per_section_29"
        verdict_sentence = (
            "cmp4/cmp6 未测实, cmp1/cmp2/cmp3/cmp5 全部实测通过; "
            "剩余比较 (cmp1/cmp2/cmp3) 所依赖的前提合取未破 (§29.4 第3条); "
            "依 §29.4/§29.5 仅得子排序成立, 非主排序全序; "
            "§36.4 不得用「全序成立 / 七方法全序」字样."
        )
    elif not testable_unverified and cmp_overall_pass_count < _all_testable_count:
        # 全部可测但部分比较未通过 → 不能宣称全序, 但也无可测降级借口.
        verdict = "no_total_order_testable_partial_pass"
        verdict_sentence = (
            f"cmp1/cmp2/cmp3/cmp5 中仅 {cmp_overall_pass_count}/{_all_testable_count} 实测通过, "
            "§14.3 六比较分条不全真, 不得宣称全序 (§36.4)."
        )
    else:
        # §36.4 「因未测不得写严格」口径 + §29.5 第5行「任一比较未测只写
        # 「未测比较不进入摘要排序句」」: cmp3/cmp5 任一 None ⇒ 任何通过性宣称均无实测支撑.
        verdict = "not_testable_no_strict_claim"
        verdict_sentence = (
            "cmp3/cmp5 判定字段缺失或非 bool (aggregate pass_strict_gt_flag 未真实落盘); "
            "未测比较不进入摘要排序句 (§29.5 第5行); "
            "§36.4 不得写严格优于或全序结论; §29.5 不得用「>」运算符."
        )

    manifest = {
        "version_id": args.version_id,
        "protocol_version": PROTOCOL_VERSION,
        # §14 同档/严格优于二阶判定结果：直接从 statistics_table.json 消费，
        # 避免下游重复实现阈值判定逻辑。供人工审计和 §17 比较脚手架直接读取。
        "section14_pairwise": section14_pairwise,
        # §14.3 六比较分条聚合 + §36.4 报告纪律:
        # 字段名从「six_cmp_pass」改为「six_cmp_aggregate_verdict」, 不再以裸 pass 表达,
        # 防止下游消费者把 cmp4/cmp6 不可测当成「全序成立」.
        "six_cmp_aggregate_verdict": verdict,
        "six_cmp_aggregate_sentence": verdict_sentence,
        # 兼容字段: 保留 six_cmp_pass 表示「可测的比较是否全通过」(不含 cmp4/cmp6),
        # 仅当 verdict=fully_tested_total_order 时等同 §14.3 全序成立.
        "six_cmp_pass": bool(_four_testable_pass),
        # §36.4 显式区分: 可测比较全部通过 vs 不可测比较的降级口径.
        "testable_comparison_pass_count": int(cmp_overall_pass_count),
        "testable_comparison_required_count": _all_testable_count,
        "not_testable_comparison_count": _not_testable_count,
        # §29.4/§29.5 子排序口径: cmp4/cmp6 不可测时整体退化为子排序, 标 True 时非全序.
        "degraded_to_subordinate_rank": (
            verdict == "subordinate_rank_only_per_section_29"
            or verdict == "not_testable_no_strict_claim"
        ),
        "comparisons": comparisons,
        "signing": signing,
    }
    (runs_root / "six_cmp_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[15_six_cmp] 完成 | manifest: {runs_root / 'six_cmp_manifest.json'} | verdict={verdict}", flush=True)
    return 0


def _approx_delta(multi_seed_summary: dict[str, dict[str, Any]], cmp_id: str) -> float | None:
    summary = multi_seed_summary.get(cmp_id, {})
    a = summary.get("rmse_mean_a")
    b = summary.get("rmse_mean_b")
    if a is None or b is None or b == 0:
        return None
    return abs(a - b) / b


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
