"""路径工具（paths）测试模块。

文件职责：验证 build_output_path、get_project_root 和
get_standard_dirs 的正确性。

测试覆盖范围：
- 显式 project_root 的标准目录
- 默认 project_root 为仓库根
- 空白字符串 project_root 拒绝
- 非空白字符串路径保留

被测模块：liquidloc.common.paths"""

from __future__ import annotations


from pathlib import Path

from liquidloc.common.constants import DEFAULT_OUTPUT_DIRS
from liquidloc.common.paths import build_output_path, get_project_root, get_standard_dirs
from liquidloc.project_layout import ProjectLayout, find_project_root


def test_get_standard_dirs_uses_explicit_project_root(tmp_path):
    dirs = get_standard_dirs(tmp_path / ".")

    assert dirs["project_root"] == tmp_path.resolve()
    assert dirs["configs"] == tmp_path / "configs"
    assert dirs["outputs"] == tmp_path / "outputs"
    assert dirs["scripts"] == tmp_path / "scripts"
    assert set(dirs["output_subdirs"]) == set(DEFAULT_OUTPUT_DIRS)
    assert dirs["output_subdirs"]["logs"] == tmp_path / "outputs" / "logs"
    assert build_output_path("figures", "plot.png", project_root=tmp_path) == (
        tmp_path / "outputs" / "figures" / "plot.png"
    )


def test_get_project_root_defaults_to_repository_root():
    assert get_project_root() == Path(__file__).resolve().parents[2]


def test_get_project_root_rejects_blank_string():
    try:
        get_project_root(" ")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for blank project_root")


def test_get_project_root_preserves_non_blank_string_path():
    path = "  relative-root  "

    assert get_project_root(path) == Path(path).resolve()


def test_project_layout_discovers_nearest_marker_from_file(tmp_path):
    project_root = tmp_path / "checkout"
    nested_file = project_root / "src" / "package" / "module.py"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert find_project_root(nested_file) == project_root.resolve()
    assert ProjectLayout.discover(nested_file) == ProjectLayout.from_root(project_root)


def test_project_layout_rejects_empty_marker_set(tmp_path):
    try:
        find_project_root(tmp_path, markers=())
    except ValueError as exc:
        assert "markers" in str(exc)
    else:
        raise AssertionError("expected ValueError for an empty marker set")
