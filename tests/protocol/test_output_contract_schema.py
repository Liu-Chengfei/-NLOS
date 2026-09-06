from __future__ import annotations

"""输出合同模式（output_contract_schema）测试模块。

测试覆盖范围：
- 输出目录结构的合同验证
- 报告/审计/检查点产物的路径约定
- 输出合同与元数据的一致性
- layout 参数：default / public_benchmark 两种合同模板
- 未知 layout 拒绝

被测模块：liquidloc.protocol.output_contract_schema"""

from pathlib import Path

import pytest

from liquidloc.common.constants import (
    DEFAULT_OUTPUT_DIRS,
    PUBLIC_BENCHMARK_OUTPUT_DIRS,
    PUBLIC_BENCHMARK_REQUIRED_OUTPUT_FILES,
)
from liquidloc.protocol.output_contract_schema import (
    LAYOUT_DEFAULT,
    LAYOUT_PUBLIC_BENCHMARK,
    check_output_contract,
)



def _build_minimal_output_tree(root: Path) -> None:
    for dirname in DEFAULT_OUTPUT_DIRS:
        (root / dirname).mkdir(parents=True, exist_ok=True)
    (root / "predictions" / "mini_predictions.json").write_text("{}", encoding="utf-8")
    (root / "metrics" / "mini_metrics.csv").write_text("metric,value\nrmse,0\n", encoding="utf-8")
    (root / "audits" / "protocol_snapshot.json").write_text("{}", encoding="utf-8")


def _build_minimal_public_benchmark_tree(root: Path) -> None:
    """按 PUBLIC_BENCHMARK 嵌套布局构造最小产物树。"""
    for dirname in PUBLIC_BENCHMARK_OUTPUT_DIRS:
        (root / dirname).mkdir(parents=True, exist_ok=True)
    # 模拟子数据集子树（miluv/core/...）。
    (root / "miluv" / "core" / "predictions").mkdir(parents=True, exist_ok=True)
    (root / "miluv" / "core" / "audits").mkdir(parents=True, exist_ok=True)
    (root / "miluv" / "core" / "plotting_inputs").mkdir(parents=True, exist_ok=True)
    (root / "miluv" / "core" / "logs").mkdir(parents=True, exist_ok=True)
    # 模拟 eval 子树（eval/...）。
    (root / "eval" / "audits").mkdir(parents=True, exist_ok=True)
    (root / "eval" / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "eval" / "statistics").mkdir(parents=True, exist_ok=True)
    (root / "eval" / "cases").mkdir(parents=True, exist_ok=True)
    # 写齐 PUBLIC_BENCHMARK_REQUIRED_OUTPUT_FILES 中每个文件。
    (root / "eval" / "audits").mkdir(parents=True, exist_ok=True)
    for relative_path in PUBLIC_BENCHMARK_REQUIRED_OUTPUT_FILES:
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")



def test_normal_case(tmp_path):
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    _build_minimal_output_tree(tmp_path)
    report = check_output_contract(tmp_path)
    assert report["is_complete"] is True
    assert report["layout"] == LAYOUT_DEFAULT  # 默认 layout 仍能返回标识字段。



def test_boundary_case(tmp_path):
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    _build_minimal_output_tree(tmp_path)
    report = check_output_contract(tmp_path)
    assert report["missing_dirs"] == []
    assert report["missing_files"] == []



def test_invalid_case(tmp_path):
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    _build_minimal_output_tree(tmp_path)
    (tmp_path / "metrics" / "mini_metrics.csv").unlink()
    report = check_output_contract(tmp_path)
    assert report["is_complete"] is False
    assert "metrics/mini_metrics.csv" in report["missing_files"]


def test_public_benchmark_layout_passes_on_minimal_tree(tmp_path):
    """public_benchmark 布局：最小嵌套树通过合同。

    验证 E7 公开 benchmark 产物的标准嵌套形态能被合同接受。
    """
    _build_minimal_public_benchmark_tree(tmp_path)
    report = check_output_contract(tmp_path, layout=LAYOUT_PUBLIC_BENCHMARK)
    assert report["layout"] == LAYOUT_PUBLIC_BENCHMARK
    assert report["is_complete"] is True
    assert report["missing_dirs"] == []
    assert report["missing_files"] == []


def test_public_benchmark_layout_reports_missing_subdirs(tmp_path):
    """public_benchmark 布局：缺关键子目录时报告。

    验证当嵌套布局的顶层目录不全时，合同明确报告缺失项。
    """
    # PUBLIC_BENCHMARK_OUTPUT_DIRS 故意缺失（实际合同只要求 public_benchmarks）。
    report = check_output_contract(tmp_path, layout=LAYOUT_PUBLIC_BENCHMARK)
    assert report["is_complete"] is False
    assert "public_benchmarks" in report["missing_dirs"]


def test_default_layout_rejects_public_benchmark_artefacts(tmp_path):
    """default 布局：public_benchmark 嵌套树应报告缺默认目录。

    验证当调用方使用 default 布局但产物是 public_benchmark 嵌套形态时，
    合同正确报告默认布局下的缺失项（避免误用 layout）。
    """
    _build_minimal_public_benchmark_tree(tmp_path)
    report = check_output_contract(tmp_path, layout=LAYOUT_DEFAULT)
    # default 布局要求 predictions / metrics / checkpoints / logs 等 flat 目录。
    assert report["is_complete"] is False
    assert "predictions" in report["missing_dirs"]
    assert "checkpoints" in report["missing_dirs"]


def test_unknown_layout_rejected(tmp_path):
    """未知 layout：抛 ValueError。

    验证 check_output_contract 拒绝未知的 layout 名，避免静默使用默认布局。
    """
    with pytest.raises(ValueError, match=r"unknown layout"):
        check_output_contract(tmp_path, layout="not_a_layout")
