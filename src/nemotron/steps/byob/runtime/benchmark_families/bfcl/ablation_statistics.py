"""Statistical primitives shared by the BFCL ablation reports.

Everything here is stdlib-only and deterministic, so a report hash does not
depend on which numerical library happened to be installed. Derived floats are
rounded through :func:`round_statistic` before they reach a report, because a
content-addressed report must not change with the platform's last bit.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Final

_FLOAT_PLACES: Final = 10


class StatisticsError(ValueError):
    """A statistic cannot be computed from the values supplied."""


def round_statistic(value: float) -> float:
    """Round a derived float so a report hash is platform-independent."""
    if not math.isfinite(value):
        raise StatisticsError("derived statistics must be finite")
    result = round(value, _FLOAT_PLACES)
    return 0.0 if result == 0 else result


def critical_z(confidence_level: float) -> float:
    if not 0.5 < confidence_level < 1.0:
        raise StatisticsError("confidence_level must be between 0.5 and 1.0")
    return statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence_level) / 2.0)


def normal_two_sided_p(z: float) -> float:
    return 2.0 * statistics.NormalDist().cdf(-abs(z))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for index in range(1, 301):
        m2 = 2 * index
        aa = index * (b - index) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        aa = -(a + index) * (qab + index) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 3.0e-14:
            return result
    raise StatisticsError("incomplete beta evaluation did not converge")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t: float, df: float) -> float:
    if df <= 0.0:
        raise StatisticsError("t-test degrees of freedom must be positive")
    return regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


def wilson_interval(successes: int, attempts: int, z: float) -> tuple[float, float]:
    if attempts <= 0:
        raise StatisticsError("a Wilson interval needs at least one attempt")
    if not 0 <= successes <= attempts:
        raise StatisticsError("successes must lie between zero and attempts")
    proportion = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = (proportion + z * z / (2.0 * attempts)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / attempts + z * z / (4.0 * attempts * attempts))
        / denominator
    )
    return center - spread, center + spread


def newcombe_difference_interval(
    arm: tuple[int, int],
    baseline: tuple[int, int],
    z: float,
) -> tuple[float, float]:
    """Score-based interval for a difference of independent proportions."""
    arm_low, arm_high = wilson_interval(arm[0], arm[1], z)
    base_low, base_high = wilson_interval(baseline[0], baseline[1], z)
    arm_proportion = arm[0] / arm[1]
    base_proportion = baseline[0] / baseline[1]
    delta = arm_proportion - base_proportion
    lower = delta - math.sqrt((arm_proportion - arm_low) ** 2 + (base_high - base_proportion) ** 2)
    upper = delta + math.sqrt((arm_high - arm_proportion) ** 2 + (base_proportion - base_low) ** 2)
    return lower, upper


def two_proportion_test(arm: tuple[int, int], baseline: tuple[int, int]) -> float:
    """Two-sided score test for a difference of independent proportions."""
    pooled = (arm[0] + baseline[0]) / (arm[1] + baseline[1])
    if pooled in (0.0, 1.0):
        # Both groups are saturated at the same value, so the difference is
        # exactly zero and a score test would divide by zero.
        return 1.0
    standard_error = math.sqrt(pooled * (1.0 - pooled) * (1.0 / arm[1] + 1.0 / baseline[1]))
    if standard_error == 0.0:
        return 1.0
    z = (arm[0] / arm[1] - baseline[0] / baseline[1]) / standard_error
    return normal_two_sided_p(z)


def welch_test(
    arm: Sequence[float],
    baseline: Sequence[float],
) -> tuple[float, float, float, float]:
    """Return (delta, p_value, ci_low, ci_high) for two independent samples."""
    if len(arm) < 2 or len(baseline) < 2:
        raise StatisticsError("Welch's test needs at least two observations per group")
    arm_mean = statistics.fmean(arm)
    base_mean = statistics.fmean(baseline)
    delta = arm_mean - base_mean
    arm_term = statistics.variance(arm) / len(arm)
    base_term = statistics.variance(baseline) / len(baseline)
    standard_error = math.sqrt(arm_term + base_term)
    if standard_error == 0.0:
        return delta, 1.0, delta, delta
    df = (arm_term + base_term) ** 2 / (
        arm_term**2 / (len(arm) - 1) + base_term**2 / (len(baseline) - 1)
    )
    t = delta / standard_error
    # A normal critical value avoids an inverse-t solve. It is slightly narrow
    # for small samples, which callers guard by refusing to conclude from them.
    critical = statistics.NormalDist().inv_cdf(0.975)
    return (
        delta,
        student_t_two_sided_p(t, df),
        delta - critical * standard_error,
        delta + critical * standard_error,
    )


def mcnemar_exact_p(discordant_a: int, discordant_b: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs only.

    Concordant pairs carry no information about a within-task change, so the
    test conditions on the discordant total. With no discordant pairs there is
    nothing to reject.
    """
    if discordant_a < 0 or discordant_b < 0:
        raise StatisticsError("discordant counts cannot be negative")
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    smaller = min(discordant_a, discordant_b)
    tail = sum(math.comb(total, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2.0**total))


__all__ = [
    "StatisticsError",
    "critical_z",
    "mcnemar_exact_p",
    "newcombe_difference_interval",
    "normal_two_sided_p",
    "regularized_incomplete_beta",
    "round_statistic",
    "student_t_two_sided_p",
    "two_proportion_test",
    "welch_test",
    "wilson_interval",
]
