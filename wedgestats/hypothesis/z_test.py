"""Z-tests for means and proportions."""

import math

import numpy as np
from scipy import stats as sp_stats

from wedgestats._typing import ArrayLike
from wedgestats.hypothesis._result import TestResult


def one_sample_z(
    data: ArrayLike,
    mu0: float,
    sigma: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    One-sample z-test for a population mean (known sigma).

    Parameters
    ----------
    data : ArrayLike
        Sample data.
    mu0 : float
        Hypothesized population mean.
    sigma : float
        Known population standard deviation.
    alternative : str
        ``'two-sided'``, ``'less'``, or ``'greater'``.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    arr = np.asarray(data, dtype=float)
    n = arr.size
    z = (float(np.mean(arr)) - mu0) / (sigma / math.sqrt(n))
    p = _z_pvalue(z, alternative)
    return TestResult(
        statistic=z, p_value=p, test_name="One-sample z-test", reject=p < alpha, alpha=alpha
    )


def two_sample_z(
    data1: ArrayLike,
    data2: ArrayLike,
    sigma1: float,
    sigma2: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    Two-sample z-test for the difference of population means (known sigmas).

    Parameters
    ----------
    data1, data2 : ArrayLike
        Sample data for each group.
    sigma1, sigma2 : float
        Known population standard deviations.
    alternative : str
        ``'two-sided'``, ``'less'``, or ``'greater'``.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    a1 = np.asarray(data1, dtype=float)
    a2 = np.asarray(data2, dtype=float)
    z = (float(np.mean(a1)) - float(np.mean(a2))) / math.sqrt(
        sigma1**2 / a1.size + sigma2**2 / a2.size
    )
    p = _z_pvalue(z, alternative)
    return TestResult(
        statistic=z, p_value=p, test_name="Two-sample z-test", reject=p < alpha, alpha=alpha
    )


def proportion_z(
    count1: int,
    nobs1: int,
    count2: int,
    nobs2: int,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    Two-proportion z-test.

    Parameters
    ----------
    count1, count2 : int
        Number of successes in each sample.
    nobs1, nobs2 : int
        Sample sizes.
    alternative : str
        ``'two-sided'``, ``'less'``, or ``'greater'``.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    p_pool = (count1 + count2) / (nobs1 + nobs2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / nobs1 + 1 / nobs2))
    z = (p1 - p2) / se
    p = _z_pvalue(z, alternative)
    return TestResult(
        statistic=z,
        p_value=p,
        test_name="Two-proportion z-test",
        reject=p < alpha,
        alpha=alpha,
    )


def _z_pvalue(z: float, alternative: str) -> float:
    """Compute p-value from a z-statistic."""
    match alternative:
        case "two-sided":
            return float(2 * sp_stats.norm.sf(abs(z)))
        case "less":
            return float(sp_stats.norm.cdf(z))
        case "greater":
            return float(sp_stats.norm.sf(z))
        case _:
            raise ValueError(
                f"Unknown alternative {alternative!r}; choose 'two-sided', 'less', or 'greater'"
            )
