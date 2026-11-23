"""Small, dependency-free statistics used across the kit.

Everything here exists to answer one question: is the number on the report
telling you something, or is it noise from a small golden set and a
non-deterministic model?
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Better behaved than the normal approximation at small n and near 0 or 1,
    which is exactly where golden-set pass rates live.
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} must be between 0 and n={n}")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on paired pass/fail outcomes.

    ``b`` is the number of cases that went pass -> fail, ``c`` the number that
    went fail -> pass. Cases that did not change carry no information about the
    difference and are ignored, which is the point of a paired test.
    """
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_mean_ci(
    values: Sequence[float],
    iters: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``.

    Used for paired score deltas and for pass rates when each case was run
    several times, where trials within a case are not independent and a plain
    binomial interval would be too narrow. Seeded so reports are reproducible.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    lo = means[int(alpha / 2 * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (lo, hi)


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    """Cohen's kappa for a 2x2 agreement table (judge vs human).

    Raw agreement flatters a judge when most cases pass: a judge that passes
    everything agrees 90% of the time with a set that is 90% passing. Kappa
    subtracts the agreement you would get by chance.
    """
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    observed = (tp + tn) / n
    judge_yes = (tp + fp) / n
    human_yes = (tp + fn) / n
    expected = judge_yes * human_yes + (1 - judge_yes) * (1 - human_yes)
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def pass_hat_k(successes: int, trials: int, k: int) -> float:
    """Unbiased estimate that ``k`` fresh trials of one case *all* pass.

    tau-bench's pass^k: C(c, k) / C(n, k) for c successes in n trials. Averaged
    over cases it gives a reliability curve from a single run. k=1 is the mean
    pass rate; k=n is "passed every trial".
    """
    if not 1 <= k <= trials:
        raise ValueError(f"k={k} must be between 1 and trials={trials}")
    return math.comb(successes, k) / math.comb(trials, k)


def pass_at_k(successes: int, trials: int, k: int) -> float:
    """Unbiased estimate that *at least one* of ``k`` fresh trials passes.

    Chen et al. (2021): 1 - C(n - c, k) / C(n, k).
    """
    if not 1 <= k <= trials:
        raise ValueError(f"k={k} must be between 1 and trials={trials}")
    return 1.0 - math.comb(trials - successes, k) / math.comb(trials, k)


def standard_error(values: Sequence[float]) -> float:
    """Standard error of the mean, one value per case (not per trial)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def clustered_standard_error(values: Sequence[float], clusters: Sequence[str]) -> float:
    """Cluster-robust standard error of the mean.

    Cases built from the same scenario or template are not independent; treating
    them as if they were understates the error. This is the estimator in Miller,
    "Adding Error Bars to Evals" (2024): sum residuals within each cluster,
    square, add up across clusters.
    """
    n = len(values)
    if n < 2 or len(clusters) != n:
        return 0.0
    mean = sum(values) / n
    totals: dict[str, float] = {}
    for v, c in zip(values, clusters, strict=True):
        totals[c] = totals.get(c, 0.0) + (v - mean)
    g = len(totals)
    if g < 2:
        return 0.0
    # small-sample correction g / (g - 1), standard for cluster-robust errors
    return math.sqrt(g / (g - 1) * sum(t * t for t in totals.values())) / n


Z_80_POWER = 0.8416212335729143


def min_detectable_effect(
    n_cases: int,
    diff_variance: float,
    within_variance_a: float = 0.0,
    within_variance_b: float = 0.0,
    trials_a: int = 1,
    trials_b: int = 1,
    z_alpha: float = Z_95,
    z_power: float = Z_80_POWER,
) -> float:
    """Smallest true change a paired comparison would detect with 80% power.

    From Miller (2024): n = (z_a + z_b)^2 (w^2 + s_A^2/K_A + s_B^2/K_B) / d^2,
    solved for d. ``diff_variance`` is the variance of per-case score
    differences; the within-case terms account for sampling noise in each run.
    """
    if n_cases < 1:
        return float("inf")
    total = diff_variance + within_variance_a / trials_a + within_variance_b / trials_b
    return (z_alpha + z_power) * math.sqrt(total / n_cases)


def cases_needed(
    effect: float,
    diff_variance: float,
    within_variance_a: float = 0.0,
    within_variance_b: float = 0.0,
    trials_a: int = 1,
    trials_b: int = 1,
    z_alpha: float = Z_95,
    z_power: float = Z_80_POWER,
) -> int:
    """Cases needed to detect ``effect`` with 80% power (same formula, solved for n)."""
    if effect <= 0:
        raise ValueError("effect must be positive")
    total = diff_variance + within_variance_a / trials_a + within_variance_b / trials_b
    return max(1, math.ceil((z_alpha + z_power) ** 2 * total / effect**2))


# Two-sided 97.5% quantiles of Student's t, df 1..30. Beyond 30 the normal is close.
_T975 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]  # fmt: skip


def t_critical_95(df: int) -> float:
    """Two-sided 95% critical value of Student's t (normal beyond df 30)."""
    if df < 1:
        raise ValueError("df must be at least 1")
    return _T975[df - 1] if df <= len(_T975) else Z_95
