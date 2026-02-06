"""Statistical power analysis, sample-size estimation, and effect-size
calculation."""

import math

from scipy import stats as sp_stats


def cohens_d(mean1: float, mean2: float, sd_pooled: float) -> float:
    """
    Cohen's d effect size.

    Parameters
    ----------
    mean1, mean2 : float
        Group means.
    sd_pooled : float
        Pooled standard deviation.

    Returns
    -------
    float
    """
    if sd_pooled <= 0:
        raise ValueError("sd_pooled must be positive")
    return (mean1 - mean2) / sd_pooled


def z_test_power(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """
    Power of a one-sample z-test.

    Parameters
    ----------
    effect_size : float
        Cohen's d (or standardized effect size).
    n : int
        Sample size.
    alpha : float
        Significance level.
    alternative : str
        ``'two-sided'``, ``'less'``, or ``'greater'``.

    Returns
    -------
    float
        Statistical power.
    """
    se = 1.0 / math.sqrt(n)
    match alternative:
        case "two-sided":
            z_crit = sp_stats.norm.ppf(1 - alpha / 2)
            power = (
                sp_stats.norm.sf(z_crit - effect_size / se)
                + sp_stats.norm.cdf(-z_crit - effect_size / se)
            )
        case "greater":
            z_crit = sp_stats.norm.ppf(1 - alpha)
            power = sp_stats.norm.sf(z_crit - effect_size / se)
        case "less":
            z_crit = sp_stats.norm.ppf(alpha)
            power = sp_stats.norm.cdf(z_crit - effect_size / se)
        case _:
            raise ValueError(f"Unknown alternative {alternative!r}")
    return float(power)


def sample_size_z(
    effect_size: float,
    power: float = 0.80,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> int:
    """
    Minimum sample size for a one-sample z-test to achieve the desired power.

    Parameters
    ----------
    effect_size : float
        Cohen's d.
    power : float
        Desired power (e.g. 0.80).
    alpha : float
        Significance level.
    alternative : str
        ``'two-sided'`` or ``'greater'``.

    Returns
    -------
    int
        Required sample size (rounded up).
    """
    if effect_size == 0:
        raise ValueError("effect_size must be non-zero")
    match alternative:
        case "two-sided":
            z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        case "greater" | "less":
            z_alpha = sp_stats.norm.ppf(1 - alpha)
        case _:
            raise ValueError(f"Unknown alternative {alternative!r}")
    z_beta = sp_stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / effect_size) ** 2
    return math.ceil(n)
