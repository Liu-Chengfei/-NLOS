"""Top-level package exports with lazy loading for startup isolation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_TYPE_EXPORTS = {
    "MeasurementControl",
    "MetricRow",
    "ModelIntermediate",
    "PredictionBundle",
    "StageResult",
    "StateEstimate",
}

if TYPE_CHECKING:
    from liquidloc.common.types import (
        MeasurementControl as MeasurementControl,
        MetricRow as MetricRow,
        ModelIntermediate as ModelIntermediate,
        PredictionBundle as PredictionBundle,
        StageResult as StageResult,
        StateEstimate as StateEstimate,
    )


def __getattr__(name: str) -> Any:
    """Load public data contracts only when callers request them."""
    if name not in _TYPE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from liquidloc.common import types as common_types

    value = getattr(common_types, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_TYPE_EXPORTS))


__all__ = tuple(sorted(_TYPE_EXPORTS))
