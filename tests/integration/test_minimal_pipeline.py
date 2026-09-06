"""最小管线子进程冒烟测试模块。

本模块通过子进程方式运行 99_mini_smoke.py 脚本，验证最小管线
从端到端的产物生成和输出合同完整性。

测试覆盖范围：
  - 子进程运行 99_mini_smoke.py，验证退出码、产物文件和合同报告

被测模块：
  - scripts.99_mini_smoke（通过子进程调用）
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _extract_stdout_json(stdout: str):
    """从混合了诊断日志的 stdout 中提取脚本打印的 JSON 摘要。

    编排脚本会向 stdout 打印诊断日志（print_args/print_dict/阶段标记），
    机器可读的 JSON 摘要以独立的多行块形式打印（行首为 '{'）。这里扫描行首
    '{' 并用 raw_decode 解析，返回最后一个解析成功的 dict，兼容纯净 stdout。
    """
    decoder = json.JSONDecoder()
    payloads: list = []
    for idx in range(len(stdout)):
        if stdout[idx] != "{":
            continue
        if idx > 0 and stdout[idx - 1] != "\n":
            continue
        try:
            value, _end = decoder.raw_decode(stdout[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            payloads.append(value)
    assert payloads, f"no JSON object found in stdout: {stdout!r}"
    return payloads[-1]


def test_minimal_pipeline_smoke(tmp_path):
    """验证最小管线子进程运行的端到端产物生成。

    测试场景：通过子进程运行 99_mini_smoke.py 脚本，
    指定临时输出目录。
    预期行为：脚本正常退出（退出码 0），stdout 输出 JSON 包含
    stage_name="contract_smoke"，合同报告 is_complete=True，
    所有产物文件存在且位于输出目录下。
    """
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "99_mini_smoke.py"
    output_root = tmp_path / "mini_smoke"
    # 定义预期的产物文件路径
    expected_artifacts = [
        output_root / "predictions" / "mini_predictions.json",
        output_root / "metrics" / "mini_metrics.csv",
        output_root / "audits" / "protocol_snapshot.json",
        output_root / "logs" / "mini_smoke.log",
    ]

    # 通过子进程运行脚本，确保脚本在独立进程中执行
    completed = subprocess.run(
        [sys.executable, str(script_path), "--output-root", str(output_root)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    # 解析脚本 stdout 的 JSON 输出（脚本可能混合打印诊断日志，提取最后一个 JSON 对象）
    payload = _extract_stdout_json(completed.stdout)
    assert payload["pipeline_stage"] == "contract_smoke"

    # 验证合同报告完整性
    report = payload["metadata"]["contract_report"]
    assert report["is_complete"] is True
    assert report["output_root"] == str(output_root)

    # 验证产物路径与预期一致，且文件实际存在
    artifacts = [Path(path) for path in payload["artifacts"]]
    assert artifacts == expected_artifacts
    # 所有产物文件必须存在
    assert all(path.exists() for path in artifacts)
    # 所有产物文件必须位于输出目录下（无路径逃逸）
    assert all(path.is_relative_to(output_root) for path in artifacts)
