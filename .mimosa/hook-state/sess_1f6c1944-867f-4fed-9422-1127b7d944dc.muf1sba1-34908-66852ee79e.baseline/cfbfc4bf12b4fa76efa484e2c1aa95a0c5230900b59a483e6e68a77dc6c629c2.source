from __future__ import annotations

"""模拟原始数据生成脚本（generate_sim_raw）测试模块。

测试覆盖范围：
- 模拟原始数据的生成流程
- 噪声注入与序列规格
- 输出文件结构的验证

被测模块：scripts.generate_sim_raw"""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    script_path = ROOT / "scripts" / "02_generate_sim_raw.py"
    spec = importlib.util.spec_from_file_location("generate_sim_raw_script", script_path)
    script = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(script)
    return script


def _extract_stdout_json(stdout: str):
    """从混合了诊断日志的 stdout 中提取脚本打印的 JSON 摘要。

    02_generate_sim_raw.py 脚本会向 stdout 打印诊断日志（print_args/print_dict/
    阶段标记），机器可读的 JSON 摘要以独立的多行块形式打印（行首为 '{'）。
    这里扫描行首 '{' 并用 raw_decode 解析，返回最后一个解析成功的 dict，
    兼容纯净 stdout。
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


def test_script_materializes_default_contract(tmp_path, capsys):
    """合同测试：script materializes default。\n\n验证 script materializes default 的接口合同，\n确保输入输出符合协议约定。
    """
    script = _load_script()
    output_root = tmp_path / "sim_raw"
    fixture_root = ROOT / "tests" / "fixtures" / "datasets" / "miluv"

    assert script.main(["--output-root", str(output_root), "--fixture-root", str(fixture_root)]) == 0
    stdout_payload = _extract_stdout_json(capsys.readouterr().out)

    assert stdout_payload["status"] == "ok"
    assert "noise" in stdout_payload
    assert (output_root / "sim_line_01" / "imu.json").is_file()
    assert (output_root / "sim_rotate_01" / "anchor_layout.json").is_file()


def test_script_rejects_blank_paths():
    """拒绝测试：script。\n\n验证被测功能对 script 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    script = _load_script()

    with pytest.raises(ValueError, match=r"--output-root must be a non-empty path"):
        script.main(["--output-root", " "])
    with pytest.raises(ValueError, match=r"--fixture-root must be a non-empty path"):
        script.main(["--fixture-root", " "])


# ---- _resolve_path 单元测试 ----


def test_resolve_path_none_uses_default():
    """使用测试：resolve path none。\n\n验证被测功能正确使用 resolve path none，\n确保内部依赖被正确调用。
    """
    script = _load_script()
    default = ROOT / "data" / "raw" / "sim"
    result = script._resolve_path(None, flag_name="--output-root", default=default)
    assert result == default.resolve()


def test_resolve_path_relative_interpreted_from_root():
    script = _load_script()
    result = script._resolve_path("some/relative", flag_name="--output-root", default=ROOT / "default")
    assert result.is_absolute()
    # 相对路径应该按仓库根目录解释
    assert result == (ROOT / "some" / "relative").resolve()


def test_resolve_path_absolute_kept_as_is():
    script = _load_script()
    abs_path = "/tmp/sim_output"
    result = script._resolve_path(abs_path, flag_name="--output-root", default=ROOT / "default")
    assert result == Path(abs_path).resolve()


def test_resolve_path_blank_raises():
    """空白测试：resolve path。\n\n验证 resolve path 对空白输入的拒绝，\n确保空白字符串不被接受。
    """
    script = _load_script()
    with pytest.raises(ValueError, match="--flag must be a non-empty path"):
        script._resolve_path("  ", flag_name="--flag", default=ROOT)


def test_main_returns_zero_on_success(tmp_path, capsys):
    """零值测试：main returns。\n\n验证 main returns 在零值输入下的行为，\n确保边界情况正确处理。
    """
    script = _load_script()
    output_root = tmp_path / "sim_raw"
    fixture_root = ROOT / "tests" / "fixtures" / "datasets" / "miluv"
    exit_code = script.main(["--output-root", str(output_root), "--fixture-root", str(fixture_root)])
    assert exit_code == 0
    report = _extract_stdout_json(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert "noise" in report
    assert "sim_line_01" in report["sequences"]
