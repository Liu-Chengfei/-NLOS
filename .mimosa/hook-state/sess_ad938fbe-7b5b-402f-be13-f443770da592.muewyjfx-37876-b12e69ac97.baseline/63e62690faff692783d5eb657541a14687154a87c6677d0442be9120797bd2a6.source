"""Git utilities for traceability."""

import subprocess
from pathlib import Path


def get_git_commit(repo_root: Path | None = None) -> str:
    """Get current git commit SHA (7-char short hash).
    
    Args:
        repo_root: Repository root path. Defaults to current working directory.
        
    Returns:
        Short git commit SHA (7 characters), or "unknown" if not a git repo.
    """
    try:
        if repo_root is None:
            repo_root = Path.cwd()
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:7]
        return "unknown"
    except Exception:
        return "unknown"


def get_git_branch(repo_root: Path | None = None) -> str:
    """Get current git branch name.
    
    Args:
        repo_root: Repository root path. Defaults to current working directory.
        
    Returns:
        Current branch name, or "unknown" if not a git repo.
    """
    try:
        if repo_root is None:
            repo_root = Path.cwd()
        result = subprocess.run(
            ["git", "-C", str(repo_root), "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            return branch if branch else "unknown"
        return "unknown"
    except Exception:
        return "unknown"
