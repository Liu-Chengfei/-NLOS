from __future__ import annotations

"""Compatibility CLI for the paper experiment orchestrator.

The implementation and compatibility hooks live under src. This file
keeps the historical numbered entry point importable while forwarding
attribute overrides used by legacy tests and callers.
"""

import inspect
from functools import wraps
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from liquidloc.pipelines import paper_experiments as _orchestrator

_orchestrator._bootstrap_impl_namespace()

_FACADE_NAMES = {
    "_orchestrator",
    "_sync_script_overrides",
    "_make_proxy",
    "_FACADE_NAMES",
    "main",
}


def _sync_script_overrides() -> None:
    """Propagate monkeypatches made on this legacy module into src."""
    target_ns = vars(_orchestrator)
    local_ns = globals()
    for name in tuple(target_ns):
        if name.startswith("__") or name in _FACADE_NAMES:
            continue
        if name not in local_ns:
            continue
        value = local_ns[name]
        if getattr(value, "__script_proxy_key__", None) == name:
            continue
        target_ns[name] = value


def _make_proxy(name: str, target: Any):
    @wraps(target)
    def _proxy(*args: Any, **kwargs: Any):
        _sync_script_overrides()
        return getattr(_orchestrator, name)(*args, **kwargs)

    _proxy.__script_proxy_key__ = name
    return _proxy


for _name, _value in tuple(vars(_orchestrator).items()):
    if _name.startswith("__") or _name in _FACADE_NAMES:
        continue
    if inspect.isfunction(_value):
        globals()[_name] = _make_proxy(_name, _value)
    else:
        globals()[_name] = _value


def main(argv: list[str] | None = None) -> int:
    _sync_script_overrides()
    return int(_orchestrator.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
