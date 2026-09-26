"""Project topology discovery and immutable directory layout.

This module is deliberately independent from the rest of the package so that
CLI startup can locate the checkout before importing application modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable

PROJECT_MARKERS: Final[tuple[str, ...]] = ("pyproject.toml", ".git")


def _normalize_start_path(start: str | Path | None) -> Path:
    candidate = Path(__file__) if start is None else Path(start)
    candidate = candidate.expanduser().resolve()
    return candidate.parent if candidate.is_file() else candidate


def find_project_root(
    start: str | Path | None = None,
    *,
    markers: Iterable[str] = PROJECT_MARKERS,
) -> Path:
    """Find the nearest ancestor containing any configured project marker."""
    marker_names = tuple(str(marker).strip() for marker in markers)
    if not marker_names or any(not marker for marker in marker_names):
        raise ValueError("markers must contain at least one non-empty name")

    origin = _normalize_start_path(start)
    for candidate in (origin, *origin.parents):
        if any((candidate / marker).exists() for marker in marker_names):
            return candidate
    raise RuntimeError(
        f"Cannot find project root (looked for {marker_names}) starting from {origin}"
    )


@dataclass(frozen=True, slots=True)
class ProjectLayout:
    """Canonical directories belonging to one project checkout."""

    root: Path
    configs: Path
    data: Path
    outputs: Path
    scripts: Path
    src: Path
    tests: Path

    @classmethod
    def from_root(cls, project_root: str | Path) -> "ProjectLayout":
        root = Path(project_root).expanduser().resolve()
        return cls(
            root=root,
            configs=root / "configs",
            data=root / "data",
            outputs=root / "outputs",
            scripts=root / "scripts",
            src=root / "src",
            tests=root / "tests",
        )

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "ProjectLayout":
        return cls.from_root(find_project_root(start))

    def as_dict(self) -> dict[str, Path]:
        """Return the mapping used by existing pipelines."""
        return {
            "project_root": self.root,
            "configs": self.configs,
            "data": self.data,
            "outputs": self.outputs,
            "scripts": self.scripts,
            "src": self.src,
            "tests": self.tests,
        }


__all__ = ("PROJECT_MARKERS", "ProjectLayout", "find_project_root")
