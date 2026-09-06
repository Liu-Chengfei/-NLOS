"""Okabe-Ito color-blind-safe palette for paper-grade figures.

Handbook J-5 配色要求（异步高NLOS实验全流程保障手册 §期刊图表规范覆盖层）：
    "Okabe-Ito / colorblind-safe 序列，满足 IEEE 与一般期刊可读性"

The Okabe-Ito palette is a set of 8 colors designed to be distinguishable
for people with various types of color blindness. We expose 5 colors mapped
to the 5 methods (LNN / LSTM / Transformer / EKF / Robust-EKF) as required
by handbook Part 0 排序 LNN < LSTM <= Transformer < EKF <= Robust-EKF.

All paper-grade plotters should import METHOD_COLORS from this module instead
of defining their own palette, ensuring cross-figure consistency.
"""

from __future__ import annotations

# Okabe-Ito 5-method palette (J-5 color-blind-safe)
# Source: Okabe & Ito (2008) "Color Universal Design"
METHOD_COLORS: dict[str, str] = {
    "lnn":          "#E69F00",  # orange        — proposed method, furthest left
    "lstm":         "#56B4E9",  # sky blue      — baseline NN
    "transformer":  "#009E73",  # bluish green  — baseline NN
    "ekf":          "#D55E00",  # vermilion    — primary EKF baseline
    "robust_ekf":   "#0072B2",  # blue          — robust variant
}

# Full 8-color Okabe-Ito palette (extended use)
OKABE_ITO_8: dict[str, str] = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "sky_blue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermilion": "#D55E00",
    "purple":  "#CC79A7",
}

# High-contrast grayscale fallback (when printing monochrome)
GRAYSCALE: dict[str, str] = {
    "lnn":          "#000000",
    "lstm":         "#404040",
    "transformer":  "#808080",
    "ekf":          "#B0B0B0",
    "robust_ekf":   "#D0D0D0",
}


def get_method_color(method: str, *, palette: str = "okabe_ito") -> str:
    """Get the color for a given method name.

    Args:
        method: One of 'lnn', 'lstm', 'transformer', 'ekf', 'robust_ekf'.
        palette: 'okabe_ito' (default, color-blind-safe) or 'grayscale' (monochrome).

    Returns:
        Hex color string (e.g., '#E69F00').

    Raises:
        ValueError: If method is not in the palette.
    """
    palettes = {
        "okabe_ito": METHOD_COLORS,
        "grayscale": GRAYSCALE,
    }
    p = palettes.get(palette, METHOD_COLORS)
    if method not in p:
        raise ValueError(f"Unknown method {method!r}; expected one of {list(p.keys())}")
    return p[method]


def get_method_color_by_index(method_idx: int, *, palette: str = "okabe_ito") -> str:
    """Get the color for a given method index (0-4) in canonical order.

    Canonical order is (lnn, lstm, transformer, ekf, robust_ekf).
    """
    method_order = ["lnn", "lstm", "transformer", "ekf", "robust_ekf"]
    if not 0 <= method_idx < len(method_order):
        raise ValueError(f"method_idx {method_idx} out of range [0, {len(method_order)})")
    return get_method_color(method_order[method_idx], palette=palette)


__all__ = [
    "METHOD_COLORS",
    "OKABE_ITO_8",
    "GRAYSCALE",
    "get_method_color",
    "get_method_color_by_index",
]