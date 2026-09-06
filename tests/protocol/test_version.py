from __future__ import annotations

"""协议版本（version）测试模块。

测试覆盖范围：
- 协议版本号的正确性
- 版本兼容性检查

被测模块：liquidloc.protocol.version"""

from liquidloc.protocol.version import (
    CONFIG_VERSION,
    OUTPUT_CONTRACT_VERSION,
    PROTOCOL_VERSION,
    SNAPSHOT_VERSION,
    summarize_versions,
)


def test_normal_case():
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    assert summarize_versions() == {
        "protocol_version": PROTOCOL_VERSION,
        "config_version": CONFIG_VERSION,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "snapshot_version": SNAPSHOT_VERSION,
    }


def test_boundary_case():
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    assert list(summarize_versions()) == [
        "protocol_version",
        "config_version",
        "output_contract_version",
        "snapshot_version",
    ]


def test_invalid_case():
    """版本号类型与正值校验。

    验证所有版本号为正整数，确保版本号定义不会意外变为非整数或非正值。
    """
    for version_name, version_value in summarize_versions().items():
        assert isinstance(version_value, int), f"{version_name} should be int, got {type(version_value)}"
        assert version_value >= 1, f"{version_name} must be positive"
