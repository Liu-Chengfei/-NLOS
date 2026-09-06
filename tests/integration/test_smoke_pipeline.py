"""合同冒烟管线集成测试模块。

本模块验证 ContractSmokePipeline 的端到端行为，包括产物生成、
路径解析、异常输入处理等。

测试覆盖范围：
  - 正常情况：指定 output_root 运行，生成 4 个产物文件，合同验证通过
  - 边界情况：指定 project_root 运行，产物路径基于 project_root/outputs 解析
  - 相对路径情况：output_root 为相对路径时，基于 project_root 解析
  - 异常情况：output_root 指向已存在的文件（非目录）时抛出异常
  - 异常情况：空白 output_root 被拒绝

被测模块：
  - liquidloc.pipelines.contract_smoke_pipeline
"""

from __future__ import annotations

from pathlib import Path

import pytest

from liquidloc.pipelines.contract_smoke_pipeline import ContractSmokePipeline


def test_normal_case(tmp_path):
    """验证指定 output_root 运行时，产物文件完整且合同验证通过。

    测试场景：传入 output_root 参数运行 ContractSmokePipeline。
    预期行为：stage_name 为 contract_smoke，生成 4 个产物文件，
    合同报告 is_complete=True，所有产物文件存在。
    """
    output_root = tmp_path / "outputs"

    result = ContractSmokePipeline().run({"output_root": str(output_root)})

    assert result.stage_name == "contract_smoke"
    # 验证生成 4 个产物文件
    assert len(result.artifacts) == 4
    assert result.metadata["contract_report"]["is_complete"] is True
    # 验证各产物文件存在
    assert (output_root / "predictions" / "mini_predictions.json").is_file()
    assert (output_root / "metrics" / "mini_metrics.csv").is_file()
    assert (output_root / "audits" / "protocol_snapshot.json").is_file()
    assert (output_root / "logs" / "mini_smoke.log").is_file()


def test_boundary_case(tmp_path):
    """验证指定 project_root 运行时，产物路径基于 project_root/outputs 解析。

    测试场景：传入 project_root 参数（含 outputs 子目录），
    不指定 output_root。
    预期行为：产物路径解析为 project_root/outputs/mini_smoke，
    合同报告的 output_root 指向该路径。
    """
    project_root = tmp_path / "project"
    (project_root / "outputs").mkdir(parents=True)

    result = ContractSmokePipeline().run({"project_root": str(project_root)})

    # 产物路径应基于 project_root/outputs/mini_smoke
    expected_root = project_root / "outputs" / "mini_smoke"
    assert result.stage_name == "contract_smoke"
    assert Path(result.artifacts[0]).parent == expected_root / "predictions"
    assert (expected_root / "metrics" / "mini_metrics.csv").is_file()
    assert result.metadata["contract_report"]["output_root"] == str(expected_root)


def test_relative_output_root_is_project_relative(tmp_path):
    """验证 output_root 为相对路径时，基于 project_root 解析。

    测试场景：传入 project_root 和相对路径 output_root="custom_smoke"。
    预期行为：产物路径解析为 project_root/custom_smoke，
    合同报告的 output_root 指向该路径。
    """
    project_root = tmp_path / "project"
    (project_root / "outputs").mkdir(parents=True)

    result = ContractSmokePipeline().run(
        {"project_root": str(project_root), "output_root": "custom_smoke"}
    )

    # 相对路径应基于 project_root 解析
    expected_root = project_root / "custom_smoke"
    assert result.metadata["contract_report"]["output_root"] == str(expected_root)
    assert (expected_root / "predictions" / "mini_predictions.json").is_file()


def test_invalid_case(tmp_path):
    """验证 output_root 指向已存在的文件（非目录）时抛出异常。

    测试场景：output_root 路径被一个普通文件占用。
    预期行为：抛出 NotADirectoryError、FileExistsError 或 OSError。
    """
    bad_input = tmp_path / "not_a_directory"
    # 创建一个文件来阻止目录创建
    bad_input.write_text("file blocks directory creation", encoding="utf-8")

    with pytest.raises((NotADirectoryError, FileExistsError, OSError)):
        ContractSmokePipeline().run({"output_root": str(bad_input)})


def test_blank_output_root_is_rejected():
    """验证空白 output_root 被拒绝。

    测试场景：output_root 为单个空格字符串。
    预期行为：抛出 ValueError，提示 output_root 必须为非空路径。
    """
    with pytest.raises(ValueError, match=r"output_root must not be blank"):
        ContractSmokePipeline().run({"output_root": " "})
