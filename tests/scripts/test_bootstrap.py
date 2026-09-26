"""Contract tests for the shared script startup boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _bootstrap import _discover_root, bootstrap_project


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_bootstrap_discovers_checkout_and_prioritizes_source_tree():
    layout = bootstrap_project(PROJECT_ROOT / "scripts" / "00_check_env.py")

    assert layout.root == PROJECT_ROOT
    assert layout.scripts == PROJECT_ROOT / "scripts"
    assert sys.path[0] == str(PROJECT_ROOT / "src")
    assert sys.path.count(str(PROJECT_ROOT / "src")) == 1


def test_discover_root_accepts_nested_directory(tmp_path):
    root = tmp_path / "checkout"
    nested = root / "scripts" / "nested"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert _discover_root(nested) == root.resolve()


def test_direct_script_help_works_outside_checkout():
    script_path = PROJECT_ROOT / "scripts" / "00_check_env.py"
    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=PROJECT_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--project-root" in completed.stdout
