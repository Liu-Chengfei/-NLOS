"""statistics_runner §14 合约测试.

覆盖:
- §14.4 主报告统计: mean_rmse + std_rmse 分散报告.
- §14.3 同档/严格优于二阶判定: section14_pairwise 输出.
- §14.1 主指标 rmse 优先于 p95.
"""

from liquidloc.analysis.statistics_runner import build_statistics_payload


def _make_row(
    method_name,
    metric,
    value,
    scene_id="s1",
    seq_id="seq1",
    repeat=0,
    bundle=0,
):
    return {
        "method_name": method_name,
        "metric": metric,
        "value": value,
        "scene_id": scene_id,
        "seq_id": seq_id,
        "repeat": repeat,
        "bundle": bundle,
        "case_ref": f"{scene_id}_{seq_id}_{repeat}_{bundle}",
    }


def test_build_statistics_payload_includes_std_rmse():
    """§14.4 主报告统计至少报 raw 位置误差的均值与分散."""
    metric_rows = [
        _make_row("A", "rmse", 1.0),
        _make_row("A", "rmse", 1.1),
        _make_row("B", "rmse", 1.2),
        _make_row("B", "rmse", 1.3),
    ]
    payload = build_statistics_payload(metric_rows, metric_names=["rmse"])
    for method_name, summary in payload["method_summary"].items():
        assert "mean_rmse" in summary, f"missing mean_rmse for {method_name}"
        assert "std_rmse" in summary, f"missing std_rmse for {method_name}"
        assert summary["std_rmse"] >= 0.0


def test_build_statistics_payload_std_rmse_bessel_correction():
    """§14.4 std_rmse 使用 Bessel 校正 (n-1 分母)."""
    import math

    metric_rows = [
        _make_row("A", "rmse", 1.0, scene_id="s1", seq_id="q1"),
        _make_row("A", "rmse", 1.1, scene_id="s2", seq_id="q1"),
    ]
    payload = build_statistics_payload(metric_rows, metric_names=["rmse"])
    summary = payload["method_summary"]["A"]
    expected_std = math.sqrt(((1.0 - 1.05) ** 2 + (1.1 - 1.05) ** 2) / 1)
    assert abs(summary["std_rmse"] - expected_std) < 1e-10


def test_build_statistics_payload_std_rmse_single_sample_is_zero():
    """§14.4 n<2 时 std_rmse 退为 0.0."""
    metric_rows = [
        _make_row("A", "rmse", 1.0, scene_id="s1", seq_id="q1"),
    ]
    payload = build_statistics_payload(metric_rows, metric_names=["rmse"])
    summary = payload["method_summary"]["A"]
    assert summary["std_rmse"] == 0.0


def test_build_statistics_payload_section14_pairwise_strict_better():
    """§14.3 当 δ > ε_≈ (0.08) 时 section14_pairwise 报 strict_better."""
    metric_rows = [
        _make_row("A", "rmse", 1.0, scene_id="s1", seq_id="q1"),
        _make_row("A", "rmse", 1.0, scene_id="s2", seq_id="q1"),
        _make_row("B", "rmse", 1.2, scene_id="s1", seq_id="q1"),
        _make_row("B", "rmse", 1.2, scene_id="s2", seq_id="q1"),
    ]
    payload = build_statistics_payload(metric_rows, metric_names=["rmse"])
    pairwise = payload.get("section14_pairwise", [])
    assert len(pairwise) > 0
    row = next(r for r in pairwise if r["method_a"] == "A" and r["method_b"] == "B")
    assert row["comparison_status"] == "strict_better"
    assert row["same_tier"] is False
    assert row["strict_better"] is True


def test_build_statistics_payload_section14_pairwise_same_tier():
    """§14.3 当 δ ≤ ε_> (0.03) 时 section14_pairwise 报 same_tier（无 strict_better）."""
    metric_rows = [
        _make_row("A", "rmse", 1.0, scene_id="s1", seq_id="q1"),
        _make_row("A", "rmse", 1.0, scene_id="s2", seq_id="q1"),
        _make_row("B", "rmse", 1.01, scene_id="s1", seq_id="q1"),
        _make_row("B", "rmse", 1.01, scene_id="s2", seq_id="q1"),
    ]
    payload = build_statistics_payload(metric_rows, metric_names=["rmse"])
    pairwise = payload.get("section14_pairwise", [])
    assert len(pairwise) > 0
    row = next(r for r in pairwise if r["method_a"] == "A" and r["method_b"] == "B")
    assert row["comparison_status"] == "same_tier"
    assert row["same_tier"] is True
    assert row["strict_better"] is False


def test_build_statistics_payload_section14_pairwise_same_tier_and_strict_better():
    """§14.3 当 ε_> ≤ δ ≤ ε_≈ 时 section14_pairwise 报 same_tier_and_strict_better."""
    metric_rows = [
        _make_row("A", "rmse", 1.0, scene_id="s1", seq_id="q1"),
        _make_row("A", "rmse", 1.0, scene_id="s2", seq_id="q1"),
        _make_row("B", "rmse", 1.04, scene_id="s1", seq_id="q1"),
        _make_row("B", "rmse", 1.04, scene_id="s2", seq_id="q1"),
    ]
    payload = build_statistics_payload(metric_rows, metric_names=["rmse"])
    pairwise = payload.get("section14_pairwise", [])
    assert len(pairwise) > 0
    row = next(r for r in pairwise if r["method_a"] == "A" and r["method_b"] == "B")
    assert row["comparison_status"] == "same_tier_and_strict_better"
    assert row["same_tier"] is True
    assert row["strict_better"] is True


def test_build_statistics_payload_section14_pairwise_uses_rmse_not_p95():
    """§14.1 主指标 rmse 优先于 p95; section14_pairwise 只对 rmse 做判定."""
    metric_rows = [
        _make_row("A", "rmse", 1.0, scene_id="s1", seq_id="q1"),
        _make_row("A", "rmse", 1.0, scene_id="s2", seq_id="q1"),
        _make_row("B", "rmse", 1.2, scene_id="s1", seq_id="q1"),
        _make_row("B", "rmse", 1.2, scene_id="s2", seq_id="q1"),
        _make_row("A", "p95", 2.0, scene_id="s1", seq_id="q1"),
        _make_row("A", "p95", 2.0, scene_id="s2", seq_id="q1"),
        _make_row("B", "p95", 2.5, scene_id="s1", seq_id="q1"),
        _make_row("B", "p95", 2.5, scene_id="s2", seq_id="q1"),
    ]
    payload = build_statistics_payload(metric_rows, metric_names=["rmse", "p95"])
    pairwise = payload.get("section14_pairwise", [])
    for row in pairwise:
        assert row["metric"] == "rmse", (
            f"section14_pairwise should only cover rmse, got {row['metric']}"
        )


def test_build_statistics_payload_main_table_rmse_first():
    """§14.1 主表列序: rmse 优先于 p95."""
    metric_rows = [
        _make_row("A", "rmse", 1.0),
        _make_row("A", "rmse", 1.0),
        _make_row("B", "rmse", 1.2),
        _make_row("B", "rmse", 1.2),
    ]
    payload = build_statistics_payload(metric_rows, metric_names=["rmse"])
    main_table = payload.get("main_table", [])
    if main_table:
        columns = list(main_table[0].keys())
        assert columns[0] == "method_name"
        if "mean_p95" in columns:
            rmse_idx = columns.index("mean_rmse")
            p95_idx = columns.index("mean_p95")
            assert rmse_idx < p95_idx
