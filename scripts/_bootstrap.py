"""Shared startup boundary for directly executed project scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def _discover_root(script_file: str | Path) -> Path:
    origin = Path(script_file).expanduser().resolve()
    if origin.is_file():
        origin = origin.parent
    for candidate in (origin, *origin.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Cannot find project root from {origin}")


def bootstrap_project(script_file: str | Path):
    """Put this checkout first on ``sys.path`` and return its layout."""
    root = _discover_root(script_file)
    source_root = root / "src"
    source_text = str(source_root)
    sys.path[:] = [entry for entry in sys.path if entry != source_text]
    sys.path.insert(0, source_text)

    from liquidloc.project_layout import ProjectLayout

    return ProjectLayout.from_root(root)


__all__ = ("bootstrap_project",)
