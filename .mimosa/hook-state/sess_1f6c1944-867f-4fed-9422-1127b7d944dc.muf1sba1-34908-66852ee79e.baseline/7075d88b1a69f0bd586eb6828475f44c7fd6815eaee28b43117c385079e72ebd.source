"""scripts/12_run_statistics.py 测试模块.

测试覆盖:
- --audit-dir 参数读取 section9_pulse_async_audit.json, 注入到 statistics_table.json
- 无 --audit-dir 时, payload 不含 section9_pulse_async_audit (向后兼容)
- section9_pulse_async_audit 写入 payload["section9_pulse_async_audit"]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "12_run_statistics.py"


def _load_run_12_main():
    """以 importlib 方式加载 12_run_statistics.py (避免名字以数字开头的 import 限制)."""
    spec = importlib.util.spec_from_file_location("_12_run_statistics_module", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # 把 src 加入 import 路径 (12 内部 import liquidloc 模块时需要)
    src_path = str(_REPO_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    spec.loader.exec_module(module)
    return module.main


def _write_metric_table_json(tmp_path: Path, payload: dict) -> Path:
    """写入一个最小 metrics table JSON (12 的 --metric-table 输入)."""
    input_path = tmp_path / "metric_table.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return input_path


def _write_section9_audit_json(tmp_path: Path, payload: dict) -> Path:
    """写入 section9_pulse_async_audit.json (eval_pipeline 落盘格式)."""
    audit_path = tmp_path / "section9_pulse_async_audit.json"
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit_path


def _make_minimal_metric_table(rows: list[dict]) -> dict:
    """构造 12 实际能消费的 metric_table JSON (来自 run_significance_tests 的输入格式).

    12 通过 _unwrap_metric_table_payload 解包, 期望顶层是 'metric_table' 列表 (或直接是 list).
    """
    return {"metric_table": rows}


def _minimal_rmse_row(method_name: str = "ekf") -> dict:
    return {
        "case_ref": f"scene_00::{method_name}::S(A2,N2,V2,K0,M0)",
        "seq_id": "mini_seq",
        "scene_id": "S(A2,N2,V2,K0,M0)",
        "method_name": method_name,
        "metric": "rmse",
        "value": 0.05,
        "unit": "m",
        "direction": "lower_is_better",
        "group": "primary",
    }


def test_12_reads_section9_pulse_async_audit_from_audit_dir(tmp_path: Path):
    """--audit-dir 含 section9_pulse_async_audit.json → payload 含 section9_pulse_async_audit."""
    run_12 = _load_run_12_main()

    metric_table_payload = _make_minimal_metric_table([_minimal_rmse_row()])
    metric_table_path = _write_metric_table_json(tmp_path, metric_table_payload)

    audit_payload = {
        "violations_total": 1,
        "violations_by_method": {"ekf": 1},
        "violations_by_seq": {"mini_seq": 1},
        "pulse_violated_count": 1,
        "async_violated_count": 1,
        "cmp1_cmp5_at_risk_count": 1,
        "violations_detail": [
            {
                "case_ref": "scene_00::ekf::S(A2,N2,V2,K0,M0)",
                "seq_id": "mini_seq",
                "method_name": "ekf",
                "n_pulse": 5,
                "n_pulse_min": 30,
                "n_async": 2,
                "n_async_min": 20,
                "pulse_violated": True,
                "async_violated": True,
                "cmp1_cmp5_at_risk": True,
                "message": "§9.3 pulse/async violation (seq_id=mini_seq)",
            }
        ],
        "aggregation_by_bundle": [],
        "bundle_count": 1,
    }
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    _write_section9_audit_json(audit_dir, audit_payload)

    output_path = tmp_path / "statistics_table.json"
    args = [
        "--metric-table", str(metric_table_path),
        "--group-keys", "method_name",
        "--metric-names", "rmse",
        "--output-path", str(output_path),
        "--audit-dir", str(audit_dir),
    ]

    rc = run_12(args)
    assert rc == 0, f"12 脚本运行失败, rc={rc}"
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert "section9_pulse_async_audit" in result, (
        "12 payload 缺 section9_pulse_async_audit — --audit-dir 注入失效"
    )
    check = result["section9_pulse_async_audit"]
    assert check["violations_total"] == 1
    assert check["pulse_violated_count"] == 1
    assert check["async_violated_count"] == 1
    assert check["cmp1_cmp5_at_risk_count"] == 1
    assert len(check["violations_detail"]) == 1
    detail = check["violations_detail"][0]
    assert detail["n_pulse"] == 5
    assert detail["n_pulse_min"] == 30
    assert detail["n_async"] == 2
    assert detail["n_async_min"] == 20
    assert detail["pulse_violated"] is True
    assert detail["async_violated"] is True


def test_12_no_audit_dir_does_not_inject_section9_field(tmp_path: Path):
    """无 --audit-dir 时, payload 不含 section9_pulse_async_audit (向后兼容)."""
    run_12 = _load_run_12_main()

    metric_table_payload = _make_minimal_metric_table([_minimal_rmse_row()])
    metric_table_path = _write_metric_table_json(tmp_path, metric_table_payload)
    output_path = tmp_path / "statistics_table.json"

    args = [
        "--metric-table", str(metric_table_path),
        "--group-keys", "method_name",
        "--metric-names", "rmse",
        "--output-path", str(output_path),
    ]

    rc = run_12(args)
    assert rc == 0, f"12 脚本运行失败, rc={rc}"
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert "section9_pulse_async_audit" not in result, (
        "无 --audit-dir 时 payload 不应含 section9_pulse_async_audit"
    )


def test_12_audit_dir_missing_file_gracefully_degrades(tmp_path: Path):
    """--audit-dir 存在但缺 section9_pulse_async_audit.json → payload 含 note, 不 crash."""
    run_12 = _load_run_12_main()

    metric_table_payload = _make_minimal_metric_table([_minimal_rmse_row()])
    metric_table_path = _write_metric_table_json(tmp_path, metric_table_payload)
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    # 故意不写 section9_pulse_async_audit.json
    output_path = tmp_path / "statistics_table.json"

    args = [
        "--metric-table", str(metric_table_path),
        "--group-keys", "method_name",
        "--metric-names", "rmse",
        "--output-path", str(output_path),
        "--audit-dir", str(audit_dir),
    ]

    rc = run_12(args)
    assert rc == 0, f"12 脚本运行失败, rc={rc}"
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert "section9_pulse_async_audit" in result
    assert result["section9_pulse_async_audit"].get("note") == (
        "section9_pulse_async_audit.json not found in audit dir"
    )

