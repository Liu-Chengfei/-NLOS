"""协议快照回归测试模块。

本模块验证输出合同（output contract）的目录结构与文件完整性检查逻辑，
确保实验产物的目录布局符合协议规范。

测试覆盖范围：
  - 正常情况：所有必需目录和文件均存在时验证通过
  - 边界情况：缺少必需文件时报告缺失项
  - 异常情况：传入 None 时抛出 TypeError

被测模块：
  - liquidloc.protocol.output_contract_schema
  - liquidloc.common.constants（DEFAULT_OUTPUT_DIRS）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from liquidloc.common.constants import DEFAULT_OUTPUT_DIRS
from liquidloc.protocol.output_contract_schema import check_output_contract


def _build_snapshot_root(root: Path) -> Path:
    """在给定根目录下构建符合输出合同的完整目录结构和占位文件。

    Args:
        root: 输出根目录路径。

    Returns:
        构建完成的根目录路径。
    """
    # 创建所有协议要求的输出子目录
    for dirname in DEFAULT_OUTPUT_DIRS:
        (root / dirname).mkdir(parents=True, exist_ok=True)
    # 在各子目录下写入占位文件，模拟真实实验产物
    (root / "predictions" / "mini_predictions.json").write_text("{}", encoding="utf-8")
    (root / "metrics" / "mini_metrics.csv").write_text("metric,value\nrmse,0\n", encoding="utf-8")
    (root / "audits" / "protocol_snapshot.json").write_text("{}", encoding="utf-8")
    return root


def test_normal_case(tmp_path):
    """验证所有必需目录和文件均存在时，输出合同验证通过。

    测试场景：构建完整的快照根目录，调用 check_output_contract。
    预期行为：is_complete 为 True，missing_dirs 和 missing_files 均为空，
    required_dirs 与 DEFAULT_OUTPUT_DIRS 一致。
    """
    report = check_output_contract(_build_snapshot_root(tmp_path))

    assert report["is_complete"] is True
    assert report["missing_dirs"] == []
    assert report["missing_files"] == []
    # 验证报告的 required_dirs 与协议定义的默认输出目录一致
    assert report["required_dirs"] == list(DEFAULT_OUTPUT_DIRS)


def test_boundary_case(tmp_path):
    """验证缺少必需文件时，输出合同报告缺失项。

    测试场景：构建完整快照后删除 audits/protocol_snapshot.json，
    调用 check_output_contract。
    预期行为：is_complete 为 False，missing_files 包含被删除的文件路径，
    missing_dirs 仍为空（目录未被删除）。
    """
    root = _build_snapshot_root(tmp_path)
    # 删除协议快照文件，模拟产物不完整的情况
    (root / "audits" / "protocol_snapshot.json").unlink()

    report = check_output_contract(root)

    assert report["is_complete"] is False
    # 目录仍然存在，不报告缺失
    assert report["missing_dirs"] == []
    # 被删除的文件应出现在缺失列表中
    assert report["missing_files"] == ["audits/protocol_snapshot.json"]


def test_invalid_case(tmp_path):
    """验证传入 None 时抛出异常。

    测试场景：向 check_output_contract 传入 None。
    预期行为：抛出 ValueError 或 TypeError。
    """
    with pytest.raises((ValueError, TypeError)):
        check_output_contract(None)
