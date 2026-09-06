"""Standalone matrix-level sliding-window helpers.

These helpers slice dense feature matrices into fixed windows and preserve the
source row indices. They are reusable utilities, but they are not the current
structured-window contract used by the paper-grade main training/inference
pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping

from liquidloc.common.validation import is_integer


def build_windows(feature_matrix, window_size: int, step_size: int):
    """Build fixed-length windows and their source-row index mapping."""
    from liquidloc.common.tee_logger import print_dict
    print_dict({
        "window_size": window_size,
        "step_size": step_size,
        "feature_matrix_type": type(feature_matrix).__name__,
        "feature_matrix_len": len(feature_matrix) if hasattr(feature_matrix, "__len__") else None,
    }, "build_windows 入口参数")
    if not is_integer(window_size):
        raise TypeError("window_size must be an integer")
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if not is_integer(step_size):
        raise TypeError("step_size must be an integer")
    if step_size <= 0:
        raise ValueError("step_size must be a positive integer")

    if isinstance(feature_matrix, (str, bytes, bytearray)):
        raise TypeError("feature_matrix must be a 2D sequence, not a string/bytes object.")
    if isinstance(feature_matrix, Mapping):
        raise TypeError("feature_matrix must be a 2D sequence, not a mapping.")
    try:
        feature_rows = list(feature_matrix)
    except TypeError as exc:
        raise TypeError("feature_matrix must be a 2D sequence.") from exc

    if not feature_rows:
        raise ValueError("feature_matrix must contain at least one row.")

    try:
        feature_dim = len(feature_rows[0])
    except TypeError as exc:
        raise TypeError("feature_matrix rows must be sequence-like.") from exc

    if feature_dim == 0:
        raise ValueError("feature_matrix must contain at least one column.")

    for row in feature_rows:
        try:
            row_dim = len(row)
        except TypeError as exc:
            raise TypeError("feature_matrix rows must be sequence-like.") from exc
        if row_dim != feature_dim:
            raise ValueError("feature_matrix must be rectangular.")

    row_count = len(feature_rows)
    if window_size > row_count:
        return [], []

    window_tensor = []
    window_index_map = []
    for start_idx in range(0, row_count - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window_tensor.append(feature_rows[start_idx:end_idx])
        window_index_map.append(list(range(start_idx, end_idx)))
    return window_tensor, window_index_map
