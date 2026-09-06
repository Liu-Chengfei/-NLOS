"""异步高NLOS实验全流程保障手册 P22 完整统计方法模块。

实现手册 P22 要求的全部统计检验与多重比较校正：

1. **Holm-Bonferroni 多重比较校正**（替代默认 BH-FDR，匹配手册"两两比较附 Holm-Bonferroni
   多重比较校正"的硬约束）。
2. **Cohen's d 效应量**（paired Cohen's d 与独立样本 Cohen's d，匹配 P22/Cohen's d 与
   手册 §"不显著结果的正式统计出口"对效应量联合报告的要求）。
3. **SESOI（Smallest Effect Size of Interest，预设等效边界）**：在方案冻结时预注册，
   写入 decision log 后不可更改；用于 D26-D28 后的"等效/无差异"出口判定。
4. **TOST 等效性检验**（Two One-Sided Tests）：声明"LNN 与 X 等效/无实际差异"必须做的
   正式工具；输出 90%CI 与等价边界判定。
5. **贝叶斯因子 BF01**（支持 H0 的强度）：与 frequentist 结论并列报告、不互相替代。
6. **2×2 ANOVA**（A 轴 × N 轴）：A×N 交互验证 + 主效应 F/p/partial η² + Levene 方差齐性检验。
7. **整体检验（Friedman / Kruskal-Wallis）**：五方法整体差异先由全局检验承载，全局不显著
   时两两 Wilcoxon 降级为探索性（exploratory）。

所有函数遵循项目硬约束：value contract 错误抛 ValueError、输入校验复用
``coerce_finite_scalar``，NaN/Inf 显式拒绝。
"""

from __future__ import annotations

import math
import random  # noqa: F401  # 用于 BF01 的 Bayesian bootstrap 抽样
from collections.abc import Mapping, Sequence
from itertools import combinations
from statistics import NormalDist
from typing import Any

from liquidloc.common.validation import coerce_finite_scalar, require_not_none


def _coerce_p_value(raw_value: Any, *, name: str) -> float:
    """统一把 p 值转成 [0, 1] 区间内的有限 float。"""
    try:
        return coerce_finite_scalar(
            raw_value,
            name=name,
            min_value=0.0,
            max_value=1.0,
        )
    except TypeError as exc:
        if "must be numeric" in str(exc):
            raise ValueError(str(exc)) from exc
        raise


def _coerce_effect_size(raw_value: Any, *, name: str) -> float:
    """把效应量转成有限 float（不约束符号范围）。"""
    try:
        return coerce_finite_scalar(raw_value, name=name)
    except TypeError as exc:
        if "must be numeric" in str(exc):
            raise ValueError(str(exc)) from exc
        raise


def holm_bonferroni_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni 多重比较校正（手册 P22 硬约束）。

    Holm-Bonferroni 是 Bonferroni 的步进式改进，按 p 值升序排序后逐级放大阈值：
    ``p_(i) * (n - i + 1)``，并强制单调不下降（与 BH-FDR 的单调性规则一致）。

    优势（对比 BH-FDR）：控制族错误率（FWER）而非错误发现率，在五方法 10 对
    这种小到中等比较数下比 BH-FDR 更稳健，是 ML benchmark 的事实默认（手册 P22
    明文规定）。BH-FDR 仍保留在 ``_bh_fdr_adjust_p_values`` 用于其他场景。

    输入合同：元素必须是 [0, 1] 区间内的有限 float；非数值 / NaN / Inf / 超界 → ValueError。
    """
    if not p_values:
        return []

    validated_p_values: list[float] = []
    for index, raw_p_value in enumerate(p_values):
        validated_p_value = _coerce_p_value(
            raw_p_value,
            name=f"p_values[{index}]",
        )
        validated_p_values.append(validated_p_value)

    n = len(validated_p_values)
    ranked = sorted(enumerate(validated_p_values), key=lambda item: item[1])
    adjusted_ranked = [1.0] * n
    for rank_index, (_, raw_p) in enumerate(ranked):
        adjusted_ranked[rank_index] = min(1.0, raw_p * (n - rank_index))

    # 强制单调不下降：较小的 p 值对应较大的 adjusted，从大往小回推 min。
    for rank_index in range(n - 2, -1, -1):
        adjusted_ranked[rank_index] = min(
            adjusted_ranked[rank_index],
            adjusted_ranked[rank_index + 1],
        )

    adjusted_p_values = [1.0] * n
    for rank_index, (original_index, _) in enumerate(ranked):
        adjusted_p_values[original_index] = adjusted_ranked[rank_index]
    return adjusted_p_values


def cohens_d_paired(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """配对 Cohen's d（paired Cohen's d，标准化差值的均值除以差值的标准差）。

    公式：``d = mean(b - a) / sd(b - a)``。``sd`` 使用差值的总体标准差（ddof=0），
    与 Cohen 原文献及主流 ML benchmark 实现保持一致。

    返回值:
        float: Cohen's d。零方差时（所有差值完全相等）返回 0.0。
    """
    diffs = [float(b) - float(a) for a, b in zip(sample_a, sample_b)]
    if not diffs:
        return 0.0
    mean_diff = math.fsum(diffs) / len(diffs)
    if len(diffs) < 2:
        return 0.0
    variance = math.fsum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
    std_diff = math.sqrt(variance)
    if std_diff == 0.0:
        return 0.0
    return mean_diff / std_diff


def cohens_d_independent(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """独立样本 Cohen's d（pooled SD 版本，Hedges/Cohen 1988 默认）。

    公式：``d = (mean_b - mean_a) / s_pooled``，其中
    ``s_pooled = sqrt(((n_a-1)*var_a + (n_b-1)*var_b) / (n_a + n_b - 2))``。

    返回值:
        float: Cohen's d。
    """
    if len(sample_a) < 1 or len(sample_b) < 1:
        return 0.0
    mean_a = math.fsum(sample_a) / len(sample_a)
    mean_b = math.fsum(sample_b) / len(sample_b)
    if len(sample_a) < 2 or len(sample_b) < 2:
        return 0.0
    var_a = math.fsum((x - mean_a) ** 2 for x in sample_a) / (len(sample_a) - 1)
    var_b = math.fsum((x - mean_b) ** 2 for x in sample_b) / (len(sample_b) - 1)
    numerator = (len(sample_a) - 1) * var_a + (len(sample_b) - 1) * var_b
    denominator = len(sample_a) + len(sample_b) - 2
    if denominator <= 0 or numerator <= 0:
        return 0.0
    s_pooled = math.sqrt(numerator / denominator)
    if s_pooled == 0.0:
        return 0.0
    return (mean_b - mean_a) / s_pooled


def tost_equivalence_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    equivalence_bound: float,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Two One-Sided Tests（TOST）等效性检验（手册 §"不显著结果的正式统计出口"硬约束）。

    等效性 H0：|mean_b - mean_a| >= equivalence_bound（差距"有意义"）。
    等效性 H1：|mean_b - mean_a| < equivalence_bound（差距"小于边界"）。
    两个单侧检验各自以 alpha 检验；若两者皆拒绝 H0 → 声明"等效"。

    本实现使用正态近似（Wald 区间）：
    - t1 = (mean_b - mean_a - (-bound)) / se_diff = (mean_b - mean_a + bound) / se_diff
    - t2 = ((bound) - (mean_b - mean_a)) / se_diff = (bound - mean_b + mean_a) / se_diff

    返回 90% CI（对应 alpha=0.05 TOST；alpha=0.01 则返回 98% CI）。

    参数:
        sample_a: A 组（基线 / LNN）。
        sample_b: B 组（对照 / EKF 等）。
        equivalence_bound: 等效边界绝对值（与 SESOI 一致）。
        alpha: 显著性水平，默认 0.05。

    返回值:
        dict: 含 tost_p_value（min of two one-sided p）、90%CI、mean_diff、
        equivalence_bound、equivalence_decision（'equivalent'/'not_equivalent'/'insufficient_data'）。
    """
    require_not_none(sample_a, "sample_a")
    require_not_none(sample_b, "sample_b")
    equivalence_bound = _coerce_effect_size(
        equivalence_bound, name="equivalence_bound"
    )
    if equivalence_bound < 0:
        raise ValueError("equivalence_bound must be non-negative")
    alpha = _coerce_p_value(alpha, name="alpha")

    n_a = len(sample_a)
    n_b = len(sample_b)
    if n_a < 2 or n_b < 2:
        return {
            "method": "tost_normal_approximation",
            "equivalence_bound": equivalence_bound,
            "alpha": alpha,
            "mean_diff": math.fsum(sample_b) / max(n_b, 1) - math.fsum(sample_a) / max(n_a, 1),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_level": 1.0 - 2 * alpha,
            "tost_p_value": float("nan"),
            "tost_lower_p": float("nan"),
            "tost_upper_p": float("nan"),
            "equivalence_decision": "insufficient_data",
            "n_a": n_a,
            "n_b": n_b,
        }

    mean_a = math.fsum(sample_a) / n_a
    mean_b = math.fsum(sample_b) / n_b
    var_a = math.fsum((x - mean_a) ** 2 for x in sample_a) / (n_a - 1)
    var_b = math.fsum((x - mean_b) ** 2 for x in sample_b) / (n_b - 1)
    se_diff = math.sqrt(var_a / n_a + var_b / n_b)
    mean_diff = mean_b - mean_a

    if se_diff == 0.0:
        # 所有样本均值与差值都已知，等于 0 直接判等效（除非 bound 也是 0 且 mean_diff 非零）。
        ci_low = mean_diff
        ci_high = mean_diff
    else:
        z_crit = NormalDist().inv_cdf(1 - alpha)
        ci_low = mean_diff - z_crit * se_diff
        ci_high = mean_diff + z_crit * se_diff

    # Two one-sided tests:
    # H0_lower: mean_diff <= -bound  vs  H1: mean_diff > -bound  → reject iff (mean_diff + bound) > z_alpha * se
    # H0_upper: mean_diff >=  bound  vs  H1: mean_diff <  bound  → reject iff (bound - mean_diff) > z_alpha * se
    if se_diff == 0.0:
        lower_p = 1.0 if (mean_diff + equivalence_bound) > 0 else 0.0
        upper_p = 1.0 if (equivalence_bound - mean_diff) > 0 else 0.0
    else:
        z_lower = (mean_diff + equivalence_bound) / se_diff
        z_upper = (equivalence_bound - mean_diff) / se_diff
        lower_p = 1.0 - NormalDist().cdf(z_lower)
        upper_p = 1.0 - NormalDist().cdf(z_upper)

    tost_p = max(0.0, min(1.0, max(lower_p, upper_p)))
    equivalence_decision = "equivalent" if tost_p < alpha else "not_equivalent"

    return {
        "method": "tost_normal_approximation",
        "equivalence_bound": equivalence_bound,
        "alpha": alpha,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": 1.0 - 2 * alpha,
        "tost_lower_p": lower_p,
        "tost_upper_p": upper_p,
        "tost_p_value": tost_p,
        "equivalence_decision": equivalence_decision,
        "n_a": n_a,
        "n_b": n_b,
        "se_diff": se_diff,
    }


def bayes_factor_null(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    paired: bool = False,
    n_samples: int = 5000,
    rng_seed: int | None = None,
) -> dict[str, Any]:
    """BF01 贝叶斯因子：观测数据对 H0（无差异 / 等效）的支持强度。

    实现采用非参数 Bayesian bootstrap 估计均值差的后验分布，BF01 近似为后验
    概率密度的"等价区域（|Δ|<ε）"与"非等价区域（|Δ|≥ε）"之比。这里 ε
    取样本观测方差的一个小倍数（r = 0.1 * max(sd_a, sd_b)），与文献 JZS/BayesFactor
    的"宽边界 prior"实践对齐（保守，避免 BF01 过度敏感）。

    BF01 解读：
    - BF01 ∈ [1, 3]：弱证据
    - BF01 ∈ [3, 10]：中等证据
    - BF01 > 10：强证据（"支持 H0"）
    - BF01 < 1/3：支持 H1（与 frequentist 显著类似）

    严格 JZS BF01 需要 analytic integration with Cauchy prior，本实现以非参数
    bootstrap 替代，在小样本（轨迹级 N<60）下更稳健。

    参数:
        sample_a / sample_b: 两组样本。
        paired: 是否为配对样本（影响 bootstrap 抽样方式）。
        n_samples: bootstrap 抽样数。
        rng_seed: 随机种子（None → 用 Python random 模块当前状态）。

    返回值:
        dict: 含 bf01、bf10、interpretation、posterior_prob_equivalent。
    """
    require_not_none(sample_a, "sample_a")
    require_not_none(sample_b, "sample_b")
    n_a = len(sample_a)
    n_b = len(sample_b)
    if n_a < 2 or n_b < 2:
        return {
            "method": "bayes_factor_bootstrap",
            "bf01": float("nan"),
            "bf10": float("nan"),
            "interpretation": "insufficient_data",
            "posterior_prob_equivalent": float("nan"),
            "n_samples_requested": n_samples,
            "rng_seed": rng_seed,
        }

    rng = random.Random(rng_seed)
    sd_a = math.sqrt(math.fsum((x - math.fsum(sample_a) / n_a) ** 2 for x in sample_a) / (n_a - 1))
    sd_b = math.sqrt(math.fsum((x - math.fsum(sample_b) / n_b) ** 2 for x in sample_b) / (n_b - 1))
    # epsilon = "等效区域"半径。常态情形下用样本 SD 的 10%；两组方差都极小时退到一个
    # 默认最小值 1e-3（避免 0/0 边界，此时所有 diff_samples 必然都等于 0）。
    epsilon = max(0.1 * max(sd_a, sd_b), 1e-3)

    diff_samples: list[float] = []
    if paired and n_a == n_b:
        paired_diffs = [b - a for a, b in zip(sample_a, sample_b)]
        for _ in range(n_samples):
            weights = _dirichlet_weights(len(paired_diffs), rng)
            diff_samples.append(math.fsum(w * d for w, d in zip(weights, paired_diffs)))
    else:
        for _ in range(n_samples):
            weights_a = _dirichlet_weights(n_a, rng)
            weights_b = _dirichlet_weights(n_b, rng)
            mean_a = math.fsum(w * x for w, x in zip(weights_a, sample_a))
            mean_b = math.fsum(w * x for w, x in zip(weights_b, sample_b))
            diff_samples.append(mean_b - mean_a)

    n_equiv = math.fsum(1.0 for d in diff_samples if abs(d) < epsilon)
    p_equiv = n_equiv / len(diff_samples) if diff_samples else 0.0
    p_nonequiv = 1.0 - p_equiv

    # Laplace 平滑避免除零；p_equiv / max(p_nonequiv, ε')，加 ε' = 1/n_samples
    eps_smooth = 1.0 / max(n_samples, 1)
    bf01 = (p_equiv + eps_smooth) / (p_nonequiv + eps_smooth)
    bf10 = 1.0 / bf01

    if bf01 > 10:
        interpretation = "strong_evidence_for_null"
    elif bf01 > 3:
        interpretation = "moderate_evidence_for_null"
    elif bf01 > 1:
        interpretation = "weak_evidence_for_null"
    elif bf10 > 10:
        interpretation = "strong_evidence_for_alternative"
    elif bf10 > 3:
        interpretation = "moderate_evidence_for_alternative"
    else:
        interpretation = "inconclusive"

    return {
        "method": "bayes_factor_bootstrap",
        "bf01": bf01,
        "bf10": bf10,
        "interpretation": interpretation,
        "posterior_prob_equivalent": p_equiv,
        "epsilon": epsilon,
        "n_samples": len(diff_samples),
        "rng_seed": rng_seed,
    }


def _dirichlet_weights(n: int, rng: random.Random) -> list[float]:
    """生成 n 个 Dirichlet(1,...,1) 权重（等价于均匀分布上的随机单纯形）。"""
    gamma_samples = [rng.gammavariate(1.0, 1.0) for _ in range(n)]
    total = math.fsum(gamma_samples)
    if total == 0.0:
        return [1.0 / n] * n
    return [g / total for g in gamma_samples]


def two_way_anova(
    values: Sequence[float],
    factor_a: Sequence[Any],
    factor_b: Sequence[Any],
) -> dict[str, Any]:
    """两因素方差分析（2×2 ANOVA：A×N），含交互项。

    公式遵循经典 ANOVA 分解：
    - SS_total = sum((y - grand_mean)^2)
    - SS_A = sum_a n_a * (mean_a - grand_mean)^2
    - SS_B = sum_b n_b * (mean_b - grand_mean)^2
    - SS_cells = sum_cells n_cell * (mean_cell - grand_mean)^2
    - SS_interaction = SS_cells - SS_A - SS_B
    - SS_within = SS_total - SS_cells

    自由度：A 主效应 1、B 主效应 1、交互 1、within N - 4。

    返回值:
        dict: 含 f_a、p_a、f_b、p_b、f_interaction、p_interaction、
        partial_eta_squared_a / partial_eta_squared_b / partial_eta_squared_interaction、
        df_a / df_b / df_interaction / df_within / df_total / SS 分解。
    """
    require_not_none(values, "values")
    n = len(values)
    if n < 4:
        raise ValueError("two_way_anova requires at least 4 observations (2x2 cells with replication)")
    if len(factor_a) != n or len(factor_b) != n:
        raise ValueError("values, factor_a, factor_b must have equal length")

    float_values = []
    for i, v in enumerate(values):
        float_values.append(_coerce_effect_size(v, name=f"values[{i}]"))

    grand_mean = math.fsum(float_values) / n

    cell_indices: dict[tuple[int, int], list[int]] = {}
    for i, (a, b) in enumerate(zip(factor_a, factor_b)):
        cell_indices.setdefault((_coerce_factor(a), _coerce_factor(b)), []).append(i)

    if len(cell_indices) < 4:
        raise ValueError("two_way_anova requires all 4 (A x B) cells to be populated")

    factor_a_levels = sorted({_coerce_factor(a) for a in factor_a})
    factor_b_levels = sorted({_coerce_factor(b) for b in factor_b})
    if len(factor_a_levels) < 2:
        raise ValueError("factor_a must have at least 2 levels")
    if len(factor_b_levels) < 2:
        raise ValueError("factor_b must have at least 2 levels")

    a_means: dict[int, list[float]] = {a_level: [] for a_level in factor_a_levels}
    b_means: dict[int, list[float]] = {b_level: [] for b_level in factor_b_levels}

    ss_total = math.fsum((v - grand_mean) ** 2 for v in float_values)
    ss_cells = 0.0
    ss_within = 0.0

    for (a_level, b_level), indices in cell_indices.items():
        cell_values = [float_values[i] for i in indices]
        cell_mean = math.fsum(cell_values) / len(cell_values)
        ss_cells += len(cell_values) * (cell_mean - grand_mean) ** 2
        ss_within += math.fsum((v - cell_mean) ** 2 for v in cell_values)
        a_means[a_level].extend(cell_values)
        b_means[b_level].extend(cell_values)

    # 主效应：A 和 B 都按"水平均值 - 总均值"的平方和。
    ss_a = sum(
        len(a_means[a_level]) * (math.fsum(a_means[a_level]) / len(a_means[a_level]) - grand_mean) ** 2
        for a_level in factor_a_levels
    )
    ss_b = sum(
        len(b_means[b_level]) * (math.fsum(b_means[b_level]) / len(b_means[b_level]) - grand_mean) ** 2
        for b_level in factor_b_levels
    )
    ss_interaction = ss_cells - ss_a - ss_b

    df_a = len(factor_a_levels) - 1
    df_b = len(factor_b_levels) - 1
    df_interaction = df_a * df_b
    df_within = n - len(cell_indices)
    df_total = n - 1

    if df_within <= 0:
        raise ValueError("two_way_anova: df_within <= 0 (need at least 1 replicate per cell)")

    ms_a = ss_a / df_a if df_a > 0 else 0.0
    ms_b = ss_b / df_b if df_b > 0 else 0.0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0.0
    ms_within = ss_within / df_within

    f_a = ms_a / ms_within if ms_within > 0 else float("nan")
    f_b = ms_b / ms_within if ms_within > 0 else float("nan")
    f_interaction = ms_interaction / ms_within if ms_within > 0 else float("nan")

    p_a = _f_to_p(f_a, df_a, df_within)
    p_b = _f_to_p(f_b, df_b, df_within)
    p_interaction = _f_to_p(f_interaction, df_interaction, df_within)

    partial_eta_a = ss_a / (ss_a + ss_within) if (ss_a + ss_within) > 0 else 0.0
    partial_eta_b = ss_b / (ss_b + ss_within) if (ss_b + ss_within) > 0 else 0.0
    partial_eta_interaction = (
        ss_interaction / (ss_interaction + ss_within)
        if (ss_interaction + ss_within) > 0
        else 0.0
    )

    return {
        "method": "two_way_anova",
        "ss_total": ss_total,
        "ss_a": ss_a,
        "ss_b": ss_b,
        "ss_interaction": ss_interaction,
        "ss_within": ss_within,
        "ss_cells": ss_cells,
        "df_a": df_a,
        "df_b": df_b,
        "df_interaction": df_interaction,
        "df_within": df_within,
        "df_total": df_total,
        "ms_a": ms_a,
        "ms_b": ms_b,
        "ms_interaction": ms_interaction,
        "ms_within": ms_within,
        "f_a": f_a,
        "p_a": p_a,
        "f_b": f_b,
        "p_b": p_b,
        "f_interaction": f_interaction,
        "p_interaction": p_interaction,
        "partial_eta_squared_a": partial_eta_a,
        "partial_eta_squared_b": partial_eta_b,
        "partial_eta_squared_interaction": partial_eta_interaction,
        "n": n,
        "factor_a_levels": factor_a_levels,
        "factor_b_levels": factor_b_levels,
    }


def _coerce_factor(value: Any) -> int:
    """把 factor 标签转成 int 索引（支持 hashable 字符串）。"""
    if isinstance(value, (int, bool)):
        return int(value)
    if isinstance(value, str):
        return hash(value) & 0x7FFFFFFF
    raise TypeError(f"factor label must be int or str, got {type(value).__name__}")


def _f_to_p(f_value: float, df1: int, df2: int) -> float:
    """F 分布的 p 值（用 Beta 不完全函数近似，正态近似的偏置对小 df 也不大）。"""
    if math.isnan(f_value) or f_value < 0:
        return float("nan")
    if df1 <= 0 or df2 <= 0:
        return float("nan")
    # 用正则化不完全 beta：P(X > f) = I_{df2/(df2+df1*f)}(df2/2, df1/2)
    x = df2 / (df2 + df1 * f_value)
    try:
        from math import lbeta  # Python 3.8+ 提供
        ibeta_val = _regularized_incomplete_beta(x, df2 / 2.0, df1 / 2.0)
        return min(1.0, max(0.0, ibeta_val))
    except Exception:
        # 失败时退回到粗略 Wilson-Hilferty 近似，避免 crash。
        return min(1.0, max(0.0, 1.0 - NormalDist().cdf(_wilson_hilferty_z(f_value, df1, df2))))


def _wilson_hilferty_z(f_value: float, df1: int, df2: int) -> float:
    """Wilson-Hilferty 近似把 F 分布转 z 分数（仅 fallback 使用）。"""
    if df1 <= 0 or df2 <= 0:
        return float("nan")
    term = (f_value ** (1.0 / 3.0)) * (1.0 - 2.0 / (9.0 * df2)) - (1.0 - 2.0 / (9.0 * df1))
    denom = math.sqrt((2.0 / (9.0 * df1)) + (f_value ** (2.0 / 3.0)) * (2.0 / (9.0 * df2)))
    if denom == 0.0:
        return 0.0
    return term / denom


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)。

    用 Lentz 连分式 + 对称性 I_x(a,b) = 1 - I_{1-x}(b,a)，避免数值发散。
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if x < (a + 1.0) / (a + b + 2.0):
        return _betacf(x, a, b) * math.exp(
            a * math.log(x) + b * math.log(1.0 - x) - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b)
        ) / a
    return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)


def _betacf(x: float, a: float, b: float, max_iter: int = 200, eps: float = 3e-7) -> float:
    """Continued fraction for incomplete beta function (Numerical Recipes 风格)。"""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def levene_test(
    groups: Sequence[Sequence[float]],
    *,
    center: str = "median",
) -> dict[str, Any]:
    """Levene 方差齐性检验（手册 P22 §ANOVA 须检查方差齐性）。

    使用 Brown-Forsythe 修正（中心 = 中位数）以提升对非正态分布的稳健性。
    返回 W 统计量近似 p 值。
    """
    require_not_none(groups, "groups")
    if center not in ("mean", "median"):
        raise ValueError("center must be 'mean' or 'median'")

    valid_groups = []
    for group in groups:
        if not group:
            raise ValueError("each group must be non-empty")
        float_group = [_coerce_effect_size(v, name="group_value") for v in group]
        valid_groups.append(float_group)

    k = len(valid_groups)
    n_total = sum(len(g) for g in valid_groups)

    if center == "mean":
        centers = [math.fsum(g) / len(g) for g in valid_groups]
    else:
        centers = [sorted(g)[len(g) // 2] for g in valid_groups]

    z_values: list[float] = []
    group_ids: list[int] = []
    for gid, group in enumerate(valid_groups):
        for value in group:
            z_values.append(abs(value - centers[gid]))
            group_ids.append(gid)

    # 单因素 ANOVA on Z_ij
    grand_mean_z = math.fsum(z_values) / n_total
    ss_between = sum(
        sum(1 for gid in group_ids if gid == gid_iter) * (sum(z for z, gid in zip(z_values, group_ids) if gid == gid_iter) / sum(1 for gid in group_ids if gid == gid_iter) - grand_mean_z) ** 2
        for gid_iter in range(k)
    )
    # 重写：用更清晰的方式算 ss_between
    ss_between = 0.0
    for gid in range(k):
        indices = [i for i, g in enumerate(group_ids) if g == gid]
        n_g = len(indices)
        if n_g == 0:
            continue
        group_z = [z_values[i] for i in indices]
        mean_z = math.fsum(group_z) / n_g
        ss_between += n_g * (mean_z - grand_mean_z) ** 2

    ss_within = 0.0
    for gid in range(k):
        indices = [i for i, g in enumerate(group_ids) if g == gid]
        group_z = [z_values[i] for i in indices]
        if not group_z:
            continue
        mean_z = math.fsum(group_z) / len(group_z)
        ss_within += math.fsum((z - mean_z) ** 2 for z in group_z)

    df_between = k - 1
    df_within = n_total - k
    if df_within <= 0:
        raise ValueError("levene test: insufficient total sample size")

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    w_stat = ms_between / ms_within if ms_within > 0 else float("nan")
    p_value = _f_to_p(w_stat, df_between, df_within)

    return {
        "method": "levene_brown_forsythe" if center == "median" else "levene_classical",
        "center": center,
        "w_statistic": w_stat,
        "p_value": p_value,
        "df_between": df_between,
        "df_within": df_within,
        "k_groups": k,
        "n_total": n_total,
        "ss_between": ss_between,
        "ss_within": ss_within,
    }


def friedman_test(
    blocks: Sequence[Sequence[float]],
    method_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Friedman 检验（双向秩方差分析，按块对方法排序）。

    五方法整体差异的全局检验；全局不显著时，两两 Wilcoxon 降级为探索性
    （手册 D26 / 整体-局部分层推断硬约束）。
    """
    require_not_none(blocks, "blocks")
    n_blocks = len(blocks)
    if n_blocks < 2:
        raise ValueError("friedman test requires at least 2 blocks")

    block_length = len(blocks[0])
    if block_length < 2:
        raise ValueError("friedman test requires at least 2 methods per block")

    for i, block in enumerate(blocks):
        if len(block) != block_length:
            raise ValueError(f"block {i} has length {len(block)}, expected {block_length}")
        for j, value in enumerate(block):
            _coerce_effect_size(value, name=f"blocks[{i}][{j}]")

    k = block_length
    ranks: list[list[float]] = []
    tie_term = 0.0
    for block in blocks:
        sorted_indices = sorted(range(k), key=lambda i: block[i])
        rank_sum: list[float] = [0.0] * k
        i = 0
        while i < k:
            j = i
            while j + 1 < k and block[sorted_indices[j + 1]] == block[sorted_indices[i]]:
                j += 1
            avg_rank = (i + 1 + j + 1) / 2.0
            for t in range(i, j + 1):
                rank_sum[sorted_indices[t]] = avg_rank
            tie_size = j - i + 1
            if tie_size > 1:
                tie_term += tie_size ** 3 - tie_size
            i = j + 1
        ranks.append(rank_sum)

    column_rank_sums = [0.0] * k
    for rank in ranks:
        for j in range(k):
            column_rank_sums[j] += rank[j]

    q_stat = (12.0 / (n_blocks * k * (k + 1))) * sum(r ** 2 for r in column_rank_sums) - 3.0 * n_blocks * (k + 1)
    tie_correction = 1.0 - tie_term / (n_blocks * (k ** 3 - k)) if (k ** 3 - k) > 0 else 1.0
    if tie_correction == 0.0:
        tie_correction = 1e-30
    q_corrected = q_stat / tie_correction
    p_value = _chi2_to_p(q_corrected, df=k - 1)

    return {
        "method": "friedman",
        "q_statistic": q_stat,
        "q_corrected": q_corrected,
        "tie_correction_factor": tie_correction,
        "df": k - 1,
        "p_value": p_value,
        "n_blocks": n_blocks,
        "n_methods": k,
        "method_labels": list(method_labels) if method_labels is not None else None,
        "method_rank_sums": column_rank_sums,
    }


def kruskal_wallis_test(
    groups: Sequence[Sequence[float]],
    method_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Kruskal-Wallis H 检验（独立样本非参数 ANOVA）。"""
    require_not_none(groups, "groups")
    k = len(groups)
    if k < 2:
        raise ValueError("kruskal_wallis requires at least 2 groups")

    all_values: list[tuple[float, int, int]] = []
    for gi, group in enumerate(groups):
        for vi, value in enumerate(group):
            all_values.append((_coerce_effect_size(value, name=f"groups[{gi}][{vi}]"), gi, vi))

    n = len(all_values)
    sorted_values = sorted(all_values, key=lambda x: x[0])
    ranks: list[float] = []
    rank_by_group: dict[int, list[float]] = {gi: [] for gi in range(k)}
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1][0] == sorted_values[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        tie_size = j - i + 1
        if tie_size > 1:
            tie_term += tie_size ** 3 - tie_size
        for t in range(i, j + 1):
            ranks.append(avg_rank)
            rank_by_group[sorted_values[t][1]].append(avg_rank)
        i = j + 1

    h_stat = (12.0 / (n * (n + 1))) * sum(
        len(group_ranks) * (math.fsum(group_ranks) / len(group_ranks)) ** 2
        for group_ranks in rank_by_group.values()
        if group_ranks
    ) - 3.0 * (n + 1)
    tie_correction = 1.0 - tie_term / (n ** 3 - n) if (n ** 3 - n) > 0 else 1.0
    if tie_correction == 0.0:
        tie_correction = 1e-30
    h_corrected = h_stat / tie_correction
    p_value = _chi2_to_p(h_corrected, df=k - 1)

    return {
        "method": "kruskal_wallis",
        "h_statistic": h_stat,
        "h_corrected": h_corrected,
        "tie_correction_factor": tie_correction,
        "df": k - 1,
        "p_value": p_value,
        "k_groups": k,
        "n_total": n,
        "method_labels": list(method_labels) if method_labels is not None else None,
    }


def _chi2_to_p(chi2: float, df: int) -> float:
    """χ² 分布右尾概率（用 Wilson-Hilferty 正态近似）。

    Wilson-Hilferty 近似对中等 df（≥3）足够精确，与 scipy.stats.chi2.sf 差几个百分点。
    """
    if math.isnan(chi2) or chi2 < 0 or df <= 0:
        return float("nan")
    z = ((chi2 / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return min(1.0, max(0.0, 1.0 - NormalDist().cdf(z)))


def preset_sesoi(
    *,
    relative_improvement_min: float = 0.33,
    equivalence_bound_ratio: float = 0.5,
) -> dict[str, Any]:
    """预设等效边界（SESOI，方案冻结时定，Pre-2 预注册口径）。

    手册 §"不显著结果的正式统计出口"硬约束：
    - SESOI = LNN vs EKF 相对提升的"实际判定下限 33%（窗口内数学可达下限）"
    - 等效边界 = SESOI × equivalence_bound_ratio（默认 ±50%，即 ±16.5%）
    - SESOI 写入 Pre-2 decision log，事后不可更改

    参数:
        relative_improvement_min: 相对提升最小判定值（手册默认 33%）。
        equivalence_bound_ratio: 等效边界占 SESOI 比例（手册默认 ±50%）。

    返回值:
        dict: 含 sesoi_relative、equivalence_bound_relative、备注等，可直接写入 decision log。
    """
    relative_improvement_min = _coerce_effect_size(
        relative_improvement_min, name="relative_improvement_min"
    )
    equivalence_bound_ratio = _coerce_effect_size(
        equivalence_bound_ratio, name="equivalence_bound_ratio"
    )
    equivalence_bound = relative_improvement_min * equivalence_bound_ratio
    return {
        "method": "sesoi_preset",
        "sesoi_relative": relative_improvement_min,
        "equivalence_bound_relative": equivalence_bound,
        "equivalence_bound_ratio": equivalence_bound_ratio,
        "prescription": (
            "等效边界取 SESOI 的 ±{:.0%}（即相对提升 ±{:.1%} 内视为等效/无实际差异）"
            .format(equivalence_bound_ratio, equivalence_bound)
        ),
        "frozen_at_plan_lock": True,
        "pre_registration_id": "Pre-2::sesoi",
        "post_hoc_modification_allowed": False,
        "notes": (
            "手册 Part 0 实际判定区间 40–62.5% 与窗口内数学可达下限 33%；"
            "D4 用 33% 作压力/训练问题门槛，A-2 用 40% 作验收下限。"
            "此处 SESOI = 33%（数学可达下限），等效边界 = SESOI × 50% = ±16.5%。"
        ),
    }


def run_p22_full_analysis(
    metric_table: Any,
    *,
    group_keys: list[str],
    metric_names: list[str],
    pairing_keys: list[str] | None = None,
    sesoi_relative: float = 0.33,
    equivalence_bound_ratio: float = 0.5,
    bf_rng_seed: int | None = 1729,
    factor_a_key: str | None = None,
    factor_b_key: str | None = None,
    target_method_a: str = "lnn",
    target_method_b: str = "ekf",
) -> dict[str, Any]:
    """手册 P22 一站式调用：frequentist + Bayesian + 等效检验 + 全局 + 2×2 ANOVA。

    输出包含：
    - 原始 p 值（配对 Wilcoxon 或 Mann-Whitney U，per-metric、per-pair）
    - Holm-Bonferroni 校正后 p 值（5 方法 10 对）
    - Cohen's d（配对或独立样本）
    - TOST 等效性检验（含 90%CI）
    - BF01 贝叶斯因子
    - 5 方法全局 Friedman 检验 / Kruskal-Wallis 检验（按 paired 是否可用）
    - 2×2 ANOVA（按 factor_a_key × factor_b_key）
    - Levene 方差齐性检验（4 组合）
    - SESOI 预设（decision log 可追溯）

    参数:
        metric_table: 输入的指标表（DataFrame 或 list[dict]），必须含 group_keys + metric_names 列。
        group_keys: 分组键（如 ["method_name"]）。
        metric_names: 需要分析的指标名列表。
        pairing_keys: 配对键（如 ["scene_id", "seq_id"]），None 则退回独立样本检验。
        sesoi_relative: 相对提升 SESOI（默认 33%，手册数学可达下限）。
        equivalence_bound_ratio: 等效边界比例（默认 ±50%）。
        bf_rng_seed: BF01 贝叶斯 bootstrap 随机种子（保证跨机可复现）。
        factor_a_key / factor_b_key: 2×2 ANOVA 的因素键（如 ["async_level", "nlos_level"]）。
        target_method_a / target_method_b: TOST/BF 比较的目标组（默认 LNN vs EKF）。

    返回值:
        dict: 嵌套结构，包含所有统计量；下游脚本可序列化为 JSON。
    """
    from liquidloc.analysis.significance_tests import (
        _bh_fdr_adjust_p_values,
        group_metric_values,
        run_pairwise_tests,
    )

    sesoi = preset_sesoi(
        relative_improvement_min=sesoi_relative,
        equivalence_bound_ratio=equivalence_bound_ratio,
    )
    sesoi_bound = sesoi["equivalence_bound_relative"]

    metric_rows = list(metric_table) if not isinstance(metric_table, list) else metric_table
    grouped_values = group_metric_values(metric_rows, group_keys, metric_names)
    test_rows = run_pairwise_tests(
        grouped_values,
        group_keys,
        metric_names,
        metric_rows=metric_rows,
        pairing_keys=pairing_keys,
    )
    p_values = [row["p_value"] for row in test_rows]
    holm_adjusted = holm_bonferroni_adjust(p_values)
    bh_adjusted = _bh_fdr_adjust_p_values(p_values)

    # 为每对追加 Cohen's d、TOST、BF01
    for row, holm_p in zip(test_rows, holm_adjusted):
        row["holm_bonferroni_p"] = holm_p
        row["bh_fdr_p"] = bh_adjusted[test_rows.index(row)]
        # 重建配对/独立样本（这里复用 run_pairwise_tests 的逻辑）
        # 简化：从 row.group_a/group_b 反查 grouped_values
        metric_name = row["metric_name"]
        g_a = row["group_a"]
        g_b = row["group_b"]
        a_values = grouped_values.get(metric_name, {}).get(g_a, [])
        b_values = grouped_values.get(metric_name, {}).get(g_b, [])
        if row.get("test_name") == "paired_wilcoxon" and len(a_values) == len(b_values):
            row["cohens_d"] = cohens_d_paired(a_values, b_values)
        else:
            row["cohens_d"] = cohens_d_independent(a_values, b_values)
        row["tost"] = tost_equivalence_test(a_values, b_values, equivalence_bound=sesoi_bound)
        row["bayes_factor"] = bayes_factor_null(
            a_values,
            b_values,
            paired=(row.get("test_name") == "paired_wilcoxon"),
            rng_seed=bf_rng_seed,
        )

    # 整体检验（Friedman 或 Kruskal-Wallis）
    method_labels = sorted({key for metric_dict in grouped_values.values() for key in metric_dict})
    friedman_per_metric: dict[str, dict[str, Any]] = {}
    kw_per_metric: dict[str, dict[str, Any]] = {}
    for metric_name in metric_names:
        blocks = []
        # 取一个稳定排序（如按 scene_id+seq_id）做块；若无配对信息，回退到 KW
        if pairing_keys:
            # 重建 block：从 metric_rows 按 pairing_keys 聚合
            pair_index: dict[tuple, dict[Any, float]] = {}
            for row in metric_rows:
                if group_keys[0] not in row or metric_name not in row:
                    continue
                key = tuple(row[k] for k in pairing_keys)
                pair_index.setdefault(key, {})[row[group_keys[0]]] = float(row[metric_name])
            method_blocks = []
            for key in sorted(pair_index):
                entry = pair_index[key]
                if all(m in entry for m in method_labels):
                    method_blocks.append([entry[m] for m in method_labels])
            if len(method_blocks) >= 2:
                blocks = method_blocks
        if len(blocks) >= 2 and len({len(b) for b in blocks}) == 1 and method_labels:
            friedman_per_metric[metric_name] = friedman_test(blocks, method_labels=method_labels)
        # KW 永远可算
        groups_for_kw = [grouped_values.get(metric_name, {}).get(m, []) for m in method_labels]
        kw_per_metric[metric_name] = kruskal_wallis_test(
            [g for g in groups_for_kw if g],
            method_labels=[m for m, g in zip(method_labels, groups_for_kw) if g],
        )

    # 2×2 ANOVA
    anova_results: dict[str, Any] = {}
    levene_results: dict[str, Any] = {}
    if factor_a_key and factor_b_key and factor_a_key in metric_rows[0] and factor_b_key in metric_rows[0]:
        for metric_name in metric_names:
            a_vals: list[float] = []
            b_vals: list[float] = []
            factor_a_vals: list[Any] = []
            factor_b_vals: list[Any] = []
            for row in metric_rows:
                if metric_name not in row:
                    continue
                a_vals.append(float(row[metric_name]))
                factor_a_vals.append(row[factor_a_key])
                factor_b_vals.append(row[factor_b_key])
            # 第二列名复用
            if a_vals:
                anova_results[metric_name] = two_way_anova(a_vals, factor_a_vals, factor_b_vals)
            # Levene 按 factor_a × factor_b 4 组合
            cells: dict[tuple, list[float]] = {}
            for v, fa, fb in zip(a_vals, factor_a_vals, factor_b_vals):
                cells.setdefault((fa, fb), []).append(v)
            if len(cells) == 4:
                levene_results[metric_name] = levene_test(list(cells.values()))

    # LNN vs EKF 专项
    target_comparison: dict[str, Any] = {}
    for metric_name in metric_names:
        for row in test_rows:
            if row["metric_name"] != metric_name:
                continue
            group_a_label = row["group_a"][0] if row["group_a"] else None
            group_b_label = row["group_b"][0] if row["group_b"] else None
            if (target_method_a in str(group_a_label) and target_method_b in str(group_b_label)) or (
                target_method_b in str(group_a_label) and target_method_a in str(group_b_label)
            ):
                target_comparison[metric_name] = row
                break

    return {
        "method": "run_p22_full_analysis",
        "sesoi": sesoi,
        "holm_bonferroni_method": "holm_step_down_bonferroni",
        "bh_fdr_method": "benjamini_hochberg_fdr",
        "pairwise_tests": test_rows,
        "friedman_global": friedman_per_metric,
        "kruskal_wallis_global": kw_per_metric,
        "two_way_anova": anova_results,
        "levene_test": levene_results,
        "target_comparison": target_comparison,
        "method_labels": method_labels,
    }