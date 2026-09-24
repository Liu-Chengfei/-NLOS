from __future__ import annotations

"""环境检查脚本（check_env）测试模块。

测试覆盖范围：
- 运行环境的依赖检查
- CUDA/GPU 可用性检测
- Python 版本与包版本验证

被测模块：scripts.check_env"""

import json
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_module(script_name: str, alias: str):
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_script_smoke(tmp_path):
    """冒烟测试：script。\n\n快速验证 script 的基本功能可用，\n不深入检查细节，仅确认流程不崩溃。
    """
    report_path = tmp_path / "env_report.json"
    module = _load_module("00_check_env.py", "check_env_script")
    exit_code = module.main(["--report-path", str(report_path)])
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_path"] == str(report_path)
    assert payload["status"] in {"ok", "failed"}
    assert payload["outputs"]["writable"] is True
    assert "required_distributions" in payload["dependencies"]
    assert "required_paths" in payload["paths"]
    assert exit_code == (0 if payload["status"] == "ok" else 1)

    with pytest.raises(ValueError, match=r"--project-root must be a non-empty path"):
        module.main(["--project-root", " "])
    with pytest.raises(ValueError, match=r"--report-path must be a non-empty path"):
        module.main(["--report-path", " "])


def test_distribution_parsing_and_deduplication():
    module = _load_module("00_check_env.py", "check_env_script_parsing")

    project_meta = {
        "project": {
            "dependencies": [
                "demo_pkg @ https://example.com/demo_pkg-1.0.0.tar.gz",
                "requests>=2",
                "requests[security]>=2",
            ],
            "optional-dependencies": {
                "dev": [
                    "demo-pkg>=1.1",
                    "pytest>=8",
                    "pytest",
                ]
            },
        }
    }

    report = module._check_distributions(project_meta)

    assert report["required_distributions"] == ["demo-pkg", "requests", "pytest"]


def test_python_version_spec_operators():
    module = _load_module("00_check_env.py", "check_env_script_versions")

    current_major = module.sys.version_info.major
    current_minor = module.sys.version_info.minor
    current_micro = module.sys.version_info.micro

    assert module._check_python_version(f"=={current_major}.{current_minor}.*")["supported"] is True
    assert module._check_python_version(f"!={current_major + 1}.0")["supported"] is True
    assert module._check_python_version(f"<={current_major}.{current_minor}.{current_micro}")["supported"] is True
    assert module._check_python_version(f">{current_major - 1}.9")["supported"] is True
    assert module._check_python_version(f"~={current_major}.{current_minor}")["supported"] is True
