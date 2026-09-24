from __future__ import annotations

"""模拟数据准备脚本（prepare_sim_data）测试模块。

测试覆盖范围：
- 模拟数据生成与写入
- 噪声规格的传递
- 序列规格的验证

被测模块：scripts.prepare_sim_data"""

import importlib.util
import json
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


def _write_records(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_script_smoke(tmp_path):
    """冒烟测试：script。\n\n快速验证 script 的基本功能可用，\n不深入检查细节，仅确认流程不崩溃。
    """
    from liquidloc.dataio.sim_materializer import materialize_sim_raw

    module = _load_module("02_prepare_sim_data.py", "prepare_sim_script")
    raw_root = tmp_path / "raw_sim"
    fixture_root = ROOT / "tests" / "fixtures" / "datasets" / "miluv"
    materialize_sim_raw(raw_root, fixture_root=fixture_root)

    output_root = tmp_path / "prepare_sim"
    exit_code = module.main(["--raw-root", str(raw_root), "--output-root", str(output_root)])

    assert exit_code == 0
    payload = json.loads((output_root / "prepare_manifest.json").read_text(encoding="utf-8"))
    assert len(payload["sequences"]) > 0
    first_seq = next(iter(payload["sequences"]))
    assert payload["sequences"][first_seq]["event_count"] > 0

    # main() catches exceptions and returns exit_code=1 instead of raising
    assert module.main(["--project-root", " "]) == 1
    assert module.main(["--raw-root", " "]) == 1
    assert module.main(["--output-root", " "]) == 1


def test_default_raw_root_uses_dataset_config(tmp_path):
    """使用测试：default raw root。\n\n验证被测功能正确使用 default raw root，\n确保内部依赖被正确调用。
    """
    from liquidloc.dataio.sim_materializer import materialize_sim_raw

    module = _load_module("02_prepare_sim_data.py", "prepare_sim_script_default_raw")
    project_root = tmp_path / "project"
    fixture_root = ROOT / "tests" / "fixtures" / "datasets" / "miluv"
    materialize_sim_raw(project_root / "data" / "raw" / "sim", fixture_root=fixture_root)
    cfg_dir = project_root / "configs" / "datasets"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "sim.yaml").write_text(
        "dataset_name: sim\nraw_root: data/raw/sim\ninterim_root: data/interim/sim\nprocessed_root: data/processed\nmanifests_root: data/manifests\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "prepare_sim_default"
    exit_code = module.main(["--project-root", str(project_root), "--output-root", str(output_root)])

    assert exit_code == 0
    payload = json.loads((output_root / "prepare_manifest.json").read_text(encoding="utf-8"))
    assert len(payload["sequences"]) > 0
    first_seq = next(iter(payload["sequences"]))
    assert payload["sequences"][first_seq]["event_count"] > 0


def test_default_raw_root_rejects_blank_config_path(tmp_path):
    """拒绝测试：default raw root。\n\n验证被测功能对 default raw root 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    module = _load_module("02_prepare_sim_data.py", "prepare_sim_script_blank_cfg")
    project_root = tmp_path / "project"
    cfg_dir = project_root / "configs" / "datasets"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "sim.yaml").write_text(
        "dataset_name: sim\nraw_root: ' '\ninterim_root: data/interim/sim\nprocessed_root: data/processed\nmanifests_root: data/manifests\n",
        encoding="utf-8",
    )

    # main() catches ValueError and returns exit_code=1 instead of raising
    exit_code = module.main(["--project-root", str(project_root)])
    assert exit_code == 1


def test_default_raw_root_falls_back_when_config_path_is_file(tmp_path):
    module = _load_module("02_prepare_sim_data.py", "prepare_sim_script_file_cfg")
    project_root = tmp_path / "project"
    cfg_path = project_root / "configs" / "datasets" / "sim.yaml"
    cfg_path.parent.mkdir(parents=True)
    raw_root_file = project_root / "data" / "raw" / "sim.json"
    raw_root_file.parent.mkdir(parents=True)
    raw_root_file.write_text("{}", encoding="utf-8")
    cfg_path.write_text(
        "dataset_name: sim\nraw_root: data/raw/sim.json\ninterim_root: data/interim/sim\nprocessed_root: data/processed\nmanifests_root: data/manifests\n",
        encoding="utf-8",
    )

    # 源代码现在不再回退到 fixtures，而是抛出 FileNotFoundError
    with pytest.raises(FileNotFoundError, match="sim raw_root is missing or empty"):
        module._resolve_default_raw_root(project_root, {"raw_root": "data/raw/sim.json"})


# ---- _resolve_non_empty_path 单元测试 ----


def test_resolve_non_empty_path_none_with_default():
    module = _load_module("02_prepare_sim_data.py", "prepare_sim_resolve_default")
    default = ROOT / "data" / "raw" / "sim"
    result = module._resolve_non_empty_path(None, "--raw-root", default)
    assert result == default.resolve()


def test_resolve_non_empty_path_none_without_default_raises():
    """无依赖测试：resolve non empty path none。\n\n验证 resolve non empty path none 在缺少依赖时的降级行为，\n确保回退策略正确。
    """
    module = _load_module("02_prepare_sim_data.py", "prepare_sim_resolve_no_default")
    with pytest.raises(ValueError, match="--flag must be a non-empty path"):
        module._resolve_non_empty_path(None, "--flag")


def test_resolve_non_empty_path_blank_raises():
    """空白测试：resolve non empty path。\n\n验证 resolve non empty path 对空白输入的拒绝，\n确保空白字符串不被接受。
    """
    module = _load_module("02_prepare_sim_data.py", "prepare_sim_resolve_blank")
    with pytest.raises(ValueError, match="--flag must be a non-empty path"):
        module._resolve_non_empty_path("   ", "--flag")


def test_resolve_non_empty_path_absolute_kept():
    module = _load_module("02_prepare_sim_data.py", "prepare_sim_resolve_abs")
    result = module._resolve_non_empty_path("/tmp/sim_output", "--raw-root")
    assert result == Path("/tmp/sim_output").resolve()


# ---- _resolve_cfg_path 单元测试 ----


def test_resolve_cfg_path_absolute_kept():
    module = _load_module("02_prepare_sim_data.py", "prepare_sim_cfg_abs")
    abs_path = ROOT / "data" / "raw" / "sim"  # 使用真实的绝对路径
    result = module._resolve_cfg_path(ROOT, str(abs_path))
    assert result == abs_path


def test_resolve_cfg_path_relative_interpreted_from_root():
    module = _load_module("02_prepare_sim_data.py", "prepare_sim_cfg_rel")
    result = module._resolve_cfg_path(ROOT, "data/raw/sim")
    assert result == ROOT / "data" / "raw" / "sim"


def test_resolve_cfg_path_blank_raises():
    """空白测试：resolve cfg path。\n\n验证 resolve cfg path 对空白输入的拒绝，\n确保空白字符串不被接受。
    """
    module = _load_module("02_prepare_sim_data.py", "prepare_sim_cfg_blank")
    with pytest.raises(ValueError, match="sim config raw_root must be a non-empty path"):
        module._resolve_cfg_path(ROOT, "   ")


# ---- _resolve_default_raw_root 更多边界 ----


def test_resolve_default_raw_root_prefers_config_dir(tmp_path):
    module = _load_module("02_prepare_sim_data.py", "prepare_sim_default_prefer")
    config_dir = tmp_path / "data" / "raw" / "sim"
    seq_dir = config_dir / "seq_a"
    seq_dir.mkdir(parents=True)
    result = module._resolve_default_raw_root(tmp_path, {"raw_root": str(config_dir)})
    assert result == config_dir


def test_resolve_default_raw_root_falls_back_when_empty(tmp_path):
    module = _load_module("02_prepare_sim_data.py", "prepare_sim_default_empty")
    config_dir = tmp_path / "data" / "raw" / "sim"
    config_dir.mkdir(parents=True)
    # 配置目录存在但没有子目录，源代码现在抛出 FileNotFoundError 而非回退
    with pytest.raises(FileNotFoundError, match="sim raw_root is missing or empty"):
        module._resolve_default_raw_root(tmp_path, {"raw_root": str(config_dir)})
