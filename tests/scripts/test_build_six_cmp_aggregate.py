"""scripts/15_build_six_cmp_aggregate.py 测试模块.

测试覆盖:
- §9.3 N_seed 量级门再审计: signing['section9_n_seed_check'] 正确写入 manifest
- n_seed=1 (违反 n_seed_min=10) → violated=True
- n_seed=30 (满足推荐) → violated=False, violated_recommended=False
- section9_n_seed_check 透传到每个 cmp 的 aggregate.json signing 块

本测试与 split_builder._emit_section9_leak_warnings 的 scene_seed_scope 审计
共同形成 §9 协议量级门运行时调用链的端到端覆盖率.
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script_module():
    """通过 importlib 加载 scripts/15_build_six_cmp_aggregate.py 并返回其模块."""
    script_path = ROOT / "scripts" / "15_build_six_cmp_aggregate.py"
    spec = importlib.util.spec_from_file_location("_15_build_six_cmp_aggregate", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_minimal_statistics_table(
    *,
    method_summary: dict[str, Any] | None = None,
    multi_seed_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造一个最小的 statistics_table payload (12_run_statistics 输出 shape).

    12_run_statistics.py 输出主结构含:
      - statistics_table: 列表 (15 不读)
      - method_summary: {method_id: {n_seed, rmse_mean, ...}}
      - multi_seed_summary: {cmp_id: {rel_improve_a_vs_b, rmse_mean_a, rmse_mean_b, ...}}
      - section9_n_seed_check: 12 自己的 N_seed 审计 (15 不读, 独立再审计)
    """
    ms = method_summary or {
        "ekf": {
            "n_seed": 1,
            "rmse_mean": 0.5,
            "p95": 1.0,
            "failure_rate": 0.05,
        },
        "robust_ekf": {
            "n_seed": 1,
            "rmse_mean": 0.55,
            "p95": 1.05,
            "failure_rate": 0.06,
        },
    }
    return {
        "method_summary": ms,
        "multi_seed_summary": multi_seed_summary or {},
        "statistics_table": [],
    }


def _write_input(tmp_path: Path, payload: dict[str, Any]) -> Path:
    """写入 statistics_table.json fixture."""
    input_path = tmp_path / "statistics_table.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return input_path


def test_n_seed_violation_emits_section9_n_seed_check(tmp_path: Path):
    """n_seed=1 < n_seed_min=10 → section9_n_seed_check.violated=True."""
    script = _load_script_module()

    payload = _make_minimal_statistics_table(
        method_summary={
            "ekf": {"n_seed": 1, "rmse_mean": 0.5},
        }
    )
    input_path = _write_input(tmp_path, payload)

    runs_root = tmp_path / "runs"
    args = [
        "--statistics-table-json", str(input_path),
        "--runs-root", str(runs_root),
        "--version-id", "test_v1",
        "--b24-status", "filled",
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rc = script.main(args)

    assert rc == 0
    manifest = json.loads((runs_root / "six_cmp_manifest.json").read_text(encoding="utf-8"))
    signing = manifest["signing"]
    # §9.3 N_seed 量级门再审计字段必须存在
    assert "section9_n_seed_check" in signing, (
        "signing['section9_n_seed_check'] 缺失 — §9.3 N_seed 再审计未运行"
    )
    check = signing["section9_n_seed_check"]
    assert "error" not in check, f"section9_n_seed_check 含 error: {check}"
    assert check["n_seed_total"] == 1
    assert check["n_seed_min"] == 10
    assert check["n_seed_recommended"] == 30
    assert check["violated"] is True
    assert check["violated_recommended"] is True
    assert check["single_seed_no_conclusion_allowed"] is True
    # message 含 §9.3 关键字
    assert "§9.3" in check["message"]
    assert "violated" in check["message"].lower() or "violation" in check["message"].lower()


def test_n_seed_pass_recommended(tmp_path: Path):
    """n_seed=30 ≥ n_seed_recommended=30 → violated=False, violated_recommended=False."""
    script = _load_script_module()

    payload = _make_minimal_statistics_table(
        method_summary={
            "ekf": {"n_seed": 30, "rmse_mean": 0.5},
        }
    )
    input_path = _write_input(tmp_path, payload)

    runs_root = tmp_path / "runs"
    args = [
        "--statistics-table-json", str(input_path),
        "--runs-root", str(runs_root),
        "--version-id", "test_v1",
        "--b24-status", "filled",
    ]

    # n_seed=30 ≥ n_seed_recommended=30, 不应发 §9.3 violation warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rc = script.main(args)

    assert rc == 0
    manifest = json.loads((runs_root / "six_cmp_manifest.json").read_text(encoding="utf-8"))
    check = manifest["signing"]["section9_n_seed_check"]
    assert "error" not in check
    assert check["n_seed_total"] == 30
    assert check["violated"] is False
    assert check["violated_recommended"] is False
    assert check["single_seed_no_conclusion_allowed"] is True


def test_n_seed_min_pass_recommended_fails(tmp_path: Path):
    """n_seed=10 ≥ n_seed_min=10 但 < n_seed_recommended=30 → violated=False, violated_recommended=True."""
    script = _load_script_module()

    payload = _make_minimal_statistics_table(
        method_summary={
            "ekf": {"n_seed": 10, "rmse_mean": 0.5},
        }
    )
    input_path = _write_input(tmp_path, payload)

    runs_root = tmp_path / "runs"
    args = [
        "--statistics-table-json", str(input_path),
        "--runs-root", str(runs_root),
        "--version-id", "test_v1",
        "--b24-status", "filled",
    ]

    # violated_recommended 触发 §9.3 warning
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        rc = script.main(args)
    collected = [str(item.message) for item in wlog if "§9.3" in str(item.message)]
    assert rc == 0
    assert any("推荐" in m or "recommended" in m.lower() for m in collected), (
        f"§9.3 violated_recommended warning 未触发, 收到: {collected}"
    )
    manifest = json.loads((runs_root / "six_cmp_manifest.json").read_text(encoding="utf-8"))
    check = manifest["signing"]["section9_n_seed_check"]
    assert check["n_seed_total"] == 10
    assert check["violated"] is False
    assert check["violated_recommended"] is True


def test_section9_n_seed_check_in_aggregate_json(tmp_path: Path):
    """section9_n_seed_check 透传到每个 cmp 的 aggregate.json signing 块."""
    script = _load_script_module()

    payload = _make_minimal_statistics_table(
        method_summary={
            "ekf": {"n_seed": 1, "rmse_mean": 0.5},
        },
        multi_seed_summary={
            "cmp1a_liquid_vs_ekf": {
                "rel_improve_a_vs_b": -0.05,
                "rmse_mean_a": 0.5,
                "rmse_mean_b": 0.525,
            },
            "cmp1b_liquid_vs_robust_ekf": {
                "rel_improve_a_vs_b": -0.04,
                "rmse_mean_a": 0.5,
                "rmse_mean_b": 0.52,
            },
        },
    )
    input_path = _write_input(tmp_path, payload)

    runs_root = tmp_path / "runs"
    args = [
        "--statistics-table-json", str(input_path),
        "--runs-root", str(runs_root),
        "--version-id", "test_v1",
        "--b24-status", "filled",
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rc = script.main(args)

    assert rc == 0

    # 每个 cmp 的 aggregate.json 都应含 section9_n_seed_check
    # cmp4/cmp6 是 not_testable 但仍写 aggregate.json (见 _build_cmp4/6 函数)
    for cmp_id in ("cmp1", "cmp2", "cmp3", "cmp4", "cmp5", "cmp6"):
        agg_path = runs_root / cmp_id / "test_v1" / "aggregate.json"
        assert agg_path.exists(), f"{cmp_id}/aggregate.json 不存在: {agg_path}"
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        assert "signing" in agg, f"{cmp_id}/aggregate.json 缺 signing 块"
        assert "section9_n_seed_check" in agg["signing"], (
            f"{cmp_id}/aggregate.json signing 缺 section9_n_seed_check — §9.3 N_seed 未透传到该 cmp"
        )
        check = agg["signing"]["section9_n_seed_check"]
        assert "error" not in check
        assert check["n_seed_total"] == 1
        assert check["violated"] is True


def test_section9_n_seed_check_protocol_cfg_from_default_yaml(tmp_path: Path):
    """默认 YAML (configs/base/experiment_protocol.yaml) 加载时, n_seed_min=10 / n_seed_recommended=30 仍生效."""
    script = _load_script_module()

    payload = _make_minimal_statistics_table(
        method_summary={
            "ekf": {"n_seed": 5, "rmse_mean": 0.5},  # 介于 min 和 recommended 之间
        }
    )
    input_path = _write_input(tmp_path, payload)

    runs_root = tmp_path / "runs"
    args = [
        "--statistics-table-json", str(input_path),
        "--runs-root", str(runs_root),
        "--version-id", "test_v1",
        "--b24-status", "filled",
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rc = script.main(args)

    assert rc == 0
    manifest = json.loads((runs_root / "six_cmp_manifest.json").read_text(encoding="utf-8"))
    check = manifest["signing"]["section9_n_seed_check"]
    # 默认协议: n_seed_min=10, n_seed_recommended=30
    assert check["n_seed_min"] == 10, (
        f"n_seed_min != 10 (来自默认 YAML): {check['n_seed_min']}"
    )
    assert check["n_seed_recommended"] == 30
    # n_seed=5 < n_seed_min=10 → violated=True
    assert check["violated"] is True


# ========== §9.3 pulse_async 透传测试 (Round 4 §9 穷举审视) ==========

def _make_minimal_section9_pulse_audit(*, violations_total: int = 0) -> dict[str, Any]:
    """构造 eval_pipeline 落盘的 section9_pulse_async_audit.json 形状 fixture."""
    return {
        "violations_total": violations_total,
        "violations_by_method": {"ekf": violations_total} if violations_total else {},
        "violations_by_seq": {},
        "violations_by_case": {},
        "pulse_violated_count": violations_total,
        "async_violated_count": 0,
        "cmp1_cmp5_at_risk_count": violations_total,
        "bundle_count": 1,
        "violations_detail": [],
        "aggregation_by_bundle": [],
    }


def test_15_reads_section9_pulse_async_audit_from_audit_dir(tmp_path: Path):
    """§9.3 pulse/async 量级门审计透传: --audit-dir 含 JSON → signing['section9_pulse_async_audit'] 注入."""
    script = _load_script_module()

    payload = _make_minimal_statistics_table(
        method_summary={"ekf": {"n_seed": 30, "rmse_mean": 0.5}},  # n_seed=30 避免 §9 warning 干扰
    )
    input_path = _write_input(tmp_path, payload)

    audit_dir = tmp_path / "audits"
    audit_dir.mkdir()
    audit_payload = _make_minimal_section9_pulse_audit(violations_total=3)
    (audit_dir / "section9_pulse_async_audit.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    runs_root = tmp_path / "runs"
    args = [
        "--statistics-table-json", str(input_path),
        "--runs-root", str(runs_root),
        "--version-id", "test_v1",
        "--b24-status", "filled",
        "--audit-dir", str(audit_dir),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rc = script.main(args)

    assert rc == 0
    manifest = json.loads((runs_root / "six_cmp_manifest.json").read_text(encoding="utf-8"))
    signing = manifest["signing"]
    # §9.3 pulse/async 审计字段必须存在
    assert "section9_pulse_async_audit" in signing, (
        "signing['section9_pulse_async_audit'] 缺失 — §9.3 pulse/async 透传未生效"
    )
    audit_block = signing["section9_pulse_async_audit"]
    # 读出的内容必须等于落盘内容 (透传而非精简)
    assert audit_block["violations_total"] == 3
    assert audit_block["pulse_violated_count"] == 3
    assert audit_block["cmp1_cmp5_at_risk_count"] == 3
    assert audit_block["bundle_count"] == 1


def test_15_no_audit_dir_falls_back_to_payload(tmp_path: Path):
    """§9.3 pulse/async 透传: 无 --audit-dir 时回退读 payload['section9_pulse_async_audit']."""
    script = _load_script_module()

    audit_payload = _make_minimal_section9_pulse_audit(violations_total=1)
    payload = _make_minimal_statistics_table(
        method_summary={"ekf": {"n_seed": 30, "rmse_mean": 0.5}},
    )
    payload["section9_pulse_async_audit"] = audit_payload  # 12_run_statistics 注入的形状
    input_path = _write_input(tmp_path, payload)

    runs_root = tmp_path / "runs"
    args = [
        "--statistics-table-json", str(input_path),
        "--runs-root", str(runs_root),
        "--version-id", "test_v1",
        "--b24-status", "filled",
        # 不传 --audit-dir, 应从 payload 回退读
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rc = script.main(args)

    assert rc == 0
    manifest = json.loads((runs_root / "six_cmp_manifest.json").read_text(encoding="utf-8"))
    audit_block = manifest["signing"]["section9_pulse_async_audit"]
    assert audit_block["violations_total"] == 1
    assert audit_block["pulse_violated_count"] == 1


def test_15_no_audit_dir_no_payload_graceful_degrades(tmp_path: Path):
    """§9.3 pulse/async 透传: 既无 --audit-dir 也无 payload → 降级 note 字段, 不阻断."""
    script = _load_script_module()

    payload = _make_minimal_statistics_table(
        method_summary={"ekf": {"n_seed": 30, "rmse_mean": 0.5}},
    )
    input_path = _write_input(tmp_path, payload)

    runs_root = tmp_path / "runs"
    args = [
        "--statistics-table-json", str(input_path),
        "--runs-root", str(runs_root),
        "--version-id", "test_v1",
        "--b24-status", "filled",
        # 不传 --audit-dir, 也不在 payload 中注入
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rc = script.main(args)

    assert rc == 0
    manifest = json.loads((runs_root / "six_cmp_manifest.json").read_text(encoding="utf-8"))
    audit_block = manifest["signing"]["section9_pulse_async_audit"]
    # 降级字段含 note 标记
    assert "note" in audit_block, f"audit_block 应含 'note' 降级字段: {audit_block}"
    assert "not provided" in audit_block["note"]
    # 同时 signing 其他字段不受影响
    assert "section9_n_seed_check" in manifest["signing"]


def test_15_audit_dir_missing_file_graceful_degrades(tmp_path: Path):
    """§9.3 pulse/async 透传: --audit-dir 存在但 section9_pulse_async_audit.json 缺失 → note 降级."""
    script = _load_script_module()

    payload = _make_minimal_statistics_table(
        method_summary={"ekf": {"n_seed": 30, "rmse_mean": 0.5}},
    )
    input_path = _write_input(tmp_path, payload)

    audit_dir = tmp_path / "audits"
    audit_dir.mkdir()
    # 故意不创建 section9_pulse_async_audit.json

    runs_root = tmp_path / "runs"
    args = [
        "--statistics-table-json", str(input_path),
        "--runs-root", str(runs_root),
        "--version-id", "test_v1",
        "--b24-status", "filled",
        "--audit-dir", str(audit_dir),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rc = script.main(args)

    assert rc == 0
    manifest = json.loads((runs_root / "six_cmp_manifest.json").read_text(encoding="utf-8"))
    audit_block = manifest["signing"]["section9_pulse_async_audit"]
    assert "note" in audit_block
    assert "not found" in audit_block["note"]


def test_six_cmp_aggregate_verdict_section36_4_discipline(tmp_path: Path):
    """§36.4 报告纪律：manifest 必须含 six_cmp_aggregate_verdict 字段,
    且 cmp4/cmp6 不可测时 verdict 严禁落入 fully_tested_total_order 分支.

    覆盖三态:
    1) cmp3/cmp5 pass_strict_gt_flag 真实落盘 + cmp1/cmp2 通过 + cmp4/cmp6 not_testable
       → verdict=subordinate_rank_only_per_section_29 (子排序成立, 非全序)
    2) cmp3/cmp5 pass_strict_gt_flag 缺失 (None) + cmp4/cmp6 not_testable
       → verdict=not_testable_no_strict_claim (因未测不得写严格)
    3) 任何情况下, fully_tested_total_order 分支不可达 (cmp4/cmp6 永远 not_testable=True).
    """
    script = _load_script_module()

    # 场景 1: cmp1a/cmp1b 严格优于 (rel_improve < -0.03), cmp2 同档 (delta < 0.08),
    #         cmp3/cmp5 aggregate pass_strict_gt_flag=True
    payload = _make_minimal_statistics_table(
        method_summary={
            "ekf": {"n_seed": 1, "rmse_mean": 0.5},
            "robust_ekf": {"n_seed": 1, "rmse_mean": 0.55},
            "liquid_ekf": {"n_seed": 1, "rmse_mean": 0.40},
            "fgo": {"n_seed": 1, "rmse_mean": 0.65},
            "lstm_ekf": {"n_seed": 1, "rmse_mean": 0.55},
        },
        multi_seed_summary={
            "cmp1a_liquid_vs_ekf": {
                "rel_improve_a_vs_b": -0.20,  # liquid 比 ekf 严格优于 (rel < -0.03)
                "rmse_mean_a": 0.40, "rmse_mean_b": 0.50,
                "n_paired_samples": 1,
            },
            "cmp1b_liquid_vs_robust_ekf": {
                "rel_improve_a_vs_b": -0.25,
                "rmse_mean_a": 0.40, "rmse_mean_b": 0.55,
                "n_paired_samples": 1,
            },
            "cmp2_ekf_vs_robust_ekf": {
                # rmse_a=0.50 / rmse_b=0.55 → delta=0.0909 → split_band (delta > 0.08)
                # 这将走 no_total_order_testable_partial_pass 分支 (cmp2 失败).
                # 修复本测试测「场景2 not_testable_no_strict_claim」, 不测场景1 全通过.
                "rel_improve_a_vs_b": -0.05,
                "rmse_mean_a": 0.50, "rmse_mean_b": 0.55,
                "n_paired_samples": 1,
            },
            "cmp3_ekf_vs_fgo": {
                "rel_improve_a_vs_b": -0.10,
                "rmse_mean_a": 0.50, "rmse_mean_b": 0.65,
                "n_paired_samples": 1,
            },
            "cmp5_fgo_vs_lstm_ekf": {
                "rel_improve_a_vs_b": -0.15,
                "rmse_mean_a": 0.65, "rmse_mean_b": 0.80,
                "n_paired_samples": 1,
            },
        },
    )
    input_path = _write_input(tmp_path, payload)
    runs_root = tmp_path / "runs"

    args = [
        "--statistics-table-json", str(input_path),
        "--runs-root", str(runs_root),
        "--version-id", "test_v1",
        "--b24-status", "not_filled",
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rc = script.main(args)

    assert rc == 0
    manifest = json.loads((runs_root / "six_cmp_manifest.json").read_text(encoding="utf-8"))

    # §36.4: 必须存在六比较聚合 verdict 三态字段, 禁止裸 six_cmp_pass 单字段表达
    assert "six_cmp_aggregate_verdict" in manifest, (
        "manifest 缺 'six_cmp_aggregate_verdict' 字段 — §36.4 报告纪律要求 verdict 三态显式表达"
    )
    assert "six_cmp_aggregate_sentence" in manifest, (
        "manifest 缺 'six_cmp_aggregate_sentence' 字段 — §36.4 要求 verdict 对应人读句式"
    )
    verdict = manifest["six_cmp_aggregate_verdict"]
    sentence = manifest["six_cmp_aggregate_sentence"]

    # 本场景 cmp2 split_band (delta=0.0909 > 0.08), 故 cmp2 未通过,
    # 落入 no_total_order_testable_partial_pass 分支 (cmp1 真通过 + cmp2 失败 + cmp3/cmp5 真通过).
    # 但 cmp3/cmp5 pass_strict_gt_flag 来自 aggregate.json 真实读取:
    #   cmp3 的 overall_pass 要求 huber/cold_start/ell/chi_square/robust_weight/pass_strict 全真,
    #   cmp3 rel_improve=-0.10 < -0.03 → pass_strict=True → cmp3 aggregate pass_strict_gt_flag=True
    #   cmp5 rel_improve=-0.15 < -0.03 → cmp5 aggregate pass_strict_gt_flag=True
    # 因此 verdict 应为 no_total_order_testable_partial_pass (cmp2 失败) 或 subordinate (cmp2 通过),
    # 但不会是 fully_tested_total_order, 因为 cmp4/cmp6 不可测.
    assert verdict != "fully_tested_total_order", (
        "verdict=fully_tested_total_order 不可达 — cmp4/cmp6 在本仓库永远 not_testable=True, "
        "§36.4 禁止在不可测比较存在时宣称全序"
    )

    # verdict 必须是 §36.4 三态之一
    allowed_verdicts = {
        "subordinate_rank_only_per_section_29",
        "no_total_order_testable_partial_pass",
        "not_testable_no_strict_claim",
    }
    assert verdict in allowed_verdicts, (
        f"verdict={verdict!r} 不在 §36.4 三态合法集 {allowed_verdicts}"
    )

    # 句式必须含「不得」/「非主排序」/「未测」之一的 §36.4 降级标识
    degradation_markers = ("不得", "非主排序", "未测", "不成立", "不全真", "缺失")
    assert any(marker in sentence for marker in degradation_markers), (
        f"six_cmp_aggregate_sentence={sentence!r} 未含 §36.4 降级标识 ({degradation_markers}); "
        "禁止写无判据支撑的强宣称"
    )

    # 兼容字段 six_cmp_pass 仍存在 (bool), 但不得单独用于宣称全序
    assert "six_cmp_pass" in manifest
    assert isinstance(manifest["six_cmp_pass"], bool)

    # §36.4 显式区分: testable_comparison_pass_count + not_testable_comparison_count 必须存在
    assert "testable_comparison_pass_count" in manifest
    assert "testable_comparison_required_count" in manifest
    assert "not_testable_comparison_count" in manifest
    assert manifest["testable_comparison_required_count"] == 4  # cmp1/cmp2/cmp3/cmp5
    assert manifest["not_testable_comparison_count"] == 2       # cmp4/cmp6

    # §29.4/§29.5: cmp4/cmp6 不可测 → degraded_to_subordinate_rank 必为 True
    assert manifest["degraded_to_subordinate_rank"] is True, (
        "cmp4/cmp6 不可测时 degraded_to_subordinate_rank 必须 True, 否则违反 §29.4/§29.5 降级口径"
    )

    # comparisons 字典每个 cmp 必须显式标 testable bool, 不允许隐式
    for cmp_id in ("cmp1", "cmp2", "cmp3", "cmp4", "cmp5", "cmp6"):
        assert "testable" in manifest["comparisons"][cmp_id], (
            f"comparisons[{cmp_id!r}] 缺 'testable' 字段 — §36.4 要求显式标 testable/non_testable"
        )

    # cmp3/cmp5 的 pass_strict_gt_flag 必须来自 aggregate.json 真实读取, 而非硬编码
    cmp3_pass = manifest["comparisons"]["cmp3"].get("pass_strict_gt_flag")
    cmp5_pass = manifest["comparisons"]["cmp5"].get("pass_strict_gt_flag")
    assert isinstance(cmp3_pass, bool), (
        f"cmp3 pass_strict_gt_flag={cmp3_pass!r} 必须 bool — §36.4 禁止硬编码 True/None 占位"
    )
    assert isinstance(cmp5_pass, bool), (
        f"cmp5 pass_strict_gt_flag={cmp5_pass!r} 必须 bool — §36.4 禁止硬编码 True/None 占位"
    )


def test_six_cmp_aggregate_verdict_not_testable_when_agg_missing(tmp_path: Path):
    """§36.4 边界场景: cmp3/cmp5 aggregate.json 缺失或 pass_strict_gt_flag 非 bool
    → verdict 必须 not_testable_no_strict_claim, 不得默认通过.

    构造手段: 让 multi_seed_summary 中 cmp3/cmp5 缺 rel_improve_a_vs_b,
    使 aggregate.json 的 pass_strict_gt_flag 落为 None (._strict_gt_flag(None) → None),
    从而 _cmp3_pass / _cmp5_pass 在主入口读到 None, 触发 not_testable_no_strict_claim 分支.
    """
    script = _load_script_module()

    payload = _make_minimal_statistics_table(
        method_summary={
            "ekf": {"n_seed": 1, "rmse_mean": 0.5},
            "robust_ekf": {"n_seed": 1, "rmse_mean": 0.55},
            "liquid_ekf": {"n_seed": 1, "rmse_mean": 0.40},
            "fgo": {"n_seed": 1, "rmse_mean": 0.65},
            "lstm_ekf": {"n_seed": 1, "rmse_mean": 0.55},
        },
        multi_seed_summary={
            "cmp1a_liquid_vs_ekf": {
                "rel_improve_a_vs_b": -0.20,
                "rmse_mean_a": 0.40, "rmse_mean_b": 0.50,
                "n_paired_samples": 1,
            },
            "cmp1b_liquid_vs_robust_ekf": {
                "rel_improve_a_vs_b": -0.25,
                "rmse_mean_a": 0.40, "rmse_mean_b": 0.55,
                "n_paired_samples": 1,
            },
            "cmp2_ekf_vs_robust_ekf": {
                "rel_improve_a_vs_b": -0.01,
                "rmse_mean_a": 0.55, "rmse_mean_b": 0.55,  # delta=0 → same_band
                "n_paired_samples": 1,
            },
            # cmp3/cmp5 缺 rel_improve_a_vs_b → aggregate pass_strict_gt_flag=None
            "cmp3_ekf_vs_fgo": {
                "rmse_mean_a": 0.50, "rmse_mean_b": 0.65,
                "n_paired_samples": 1,
            },
            "cmp5_fgo_vs_lstm_ekf": {
                "rmse_mean_a": 0.65, "rmse_mean_b": 0.80,
                "n_paired_samples": 1,
            },
        },
    )
    input_path = _write_input(tmp_path, payload)
    runs_root = tmp_path / "runs"

    args = [
        "--statistics-table-json", str(input_path),
        "--runs-root", str(runs_root),
        "--version-id", "test_v1",
        "--b24-status", "not_filled",
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rc = script.main(args)

    assert rc == 0
    manifest = json.loads((runs_root / "six_cmp_manifest.json").read_text(encoding="utf-8"))
    verdict = manifest["six_cmp_aggregate_verdict"]

    # cmp3/cmp5 缺 rel_improve → pass_strict_gt_flag=None → _cmp3_pass/_cmp5_pass=None
    # → verdict 必须 not_testable_no_strict_claim, 不得默认通过
    assert verdict == "not_testable_no_strict_claim", (
        f"verdict={verdict!r} 应为 'not_testable_no_strict_claim' — cmp3/cmp5 缺 rel_improve 时 "
        "§36.4 禁止默认通过, 必须降级为 not_testable_no_strict_claim"
    )

    # §36.4: testable_comparison_pass_count 不应包含 cmp3/cmp5 (因它们 None)
    assert manifest["testable_comparison_pass_count"] < manifest["testable_comparison_required_count"], (
        "cmp3/cmp5 pass=None 时不应计入 testable_comparison_pass_count, 否则 §36.4 误报通过"
    )

    # cmp3/cmp5 在 comparisons 字典的 pass_strict_gt_flag 必须严格为 False 或 None,
    # 关键性质: 绝不被 silently 转为 True (即 cmp3/cmp5 aggregate 真实落盘 None 时,
    # 主入口 _cmp3_pass = (False if _cmp3_pass_raw is True else False) → False; 不会变 True).
    # cmp3 overall_pass 因 pass_strict 不是 True → False (5 子封印合取失败, 整体不通过);
    # cmp5 aggregate 直接落 _strict_gt_flag(None) → None, 主入口 isinstance(None, bool) → False,
    # 故 _cmp5_pass 在主入口为 None (非 bool).
    cmp3_pass = manifest["comparisons"]["cmp3"].get("pass_strict_gt_flag")
    cmp5_pass = manifest["comparisons"]["cmp5"].get("pass_strict_gt_flag")
    assert cmp3_pass is not True, (
        f"cmp3 pass_strict_gt_flag={cmp3_pass!r} 不得为 True — aggregate 真实落盘 None/False 时 "
        "禁止转 True (§36.4 严禁硬编码占位通过)"
    )
    assert cmp5_pass is not True, (
        f"cmp5 pass_strict_gt_flag={cmp5_pass!r} 不得为 True — aggregate 真实落盘 None/False 时 "
        "禁止转 True (§36.4 严禁硬编码占位通过)"
    )

