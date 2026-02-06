"""Confidence intervals."""

from dataclasses import dataclass
import math

import numpy as np
from scipy import stats as sp_stats

from wedgestats._typing import ArrayLike


@dataclass(frozen=True)
class ConfidenceInterval:
    """An interval estimate."""

    lower: float
    upper: float
    confidence_level: float


def ci_mean(
    data: ArrayLike,
    confidence: float = 0.95,
    sigma: float | None = None,
) -> ConfidenceInterval:
    """
    Confidence interval for a population mean.

    Parameters
    ----------
    data : ArrayLike
        Sample data.
    confidence : float
        Confidence level (e.g. 0.95).
    sigma : float or None
        If provided, use z-interval (known sigma). Otherwise, use
        t-interval.

    Returns
    -------
    ConfidenceInterval
    """
    arr = np.asarray(data, dtype=float)
    n = arr.size
    xbar = float(np.mean(arr))
    alpha_half = (1 - confidence) / 2

    if sigma is not None:
        z = sp_stats.norm.ppf(1 - alpha_half)
        me = z * sigma / math.sqrt(n)
    else:
        s = float(np.std(arr, ddof=1))
        t_crit = sp_stats.t.ppf(1 - alpha_half, df=n - 1)
        me = t_crit * s / math.sqrt(n)

    return ConfidenceInterval(
        lower=xbar - me, upper=xbar + me, confidence_level=confidence
    )


def ci_proportion(
    count: int,
    nobs: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """
    Wald confidence interval for a population proportion.

    Parameters
    ----------
    count : int
        Number of successes.
    nobs : int
        Sample size.
    confidence : float
        Confidence level.

    Returns
    -------
    ConfidenceInterval
    """
    p_hat = count / nobs
    alpha_half = (1 - confidence) / 2
    z = sp_stats.norm.ppf(1 - alpha_half)
    me = z * math.sqrt(p_hat * (1 - p_hat) / nobs)
    return ConfidenceInterval(
        lower=p_hat - me, upper=p_hat + me, confidence_level=confidence
    )


def ci_difference_of_means(
    data1: ArrayLike,
    data2: ArrayLike,
    confidence: float = 0.95,
    equal_var: bool = True,
) -> ConfidenceInterval:
    """
    Confidence interval for the difference of two population means.

    Parameters
    ----------
    data1, data2 : ArrayLike
        Sample data for each group.
    confidence : float
        Confidence level.
    equal_var : bool
        If True, use pooled variance. Otherwise, use Welch's approach.

    Returns
    -------
    ConfidenceInterval
    """
    a1 = np.asarray(data1, dtype=float)
    a2 = np.asarray(data2, dtype=float)
    n1, n2 = a1.size, a2.size
    diff = float(np.mean(a1)) - float(np.mean(a2))
    alpha_half = (1 - confidence) / 2

    if equal_var:
        sp2 = ((n1 - 1) * float(np.var(a1, ddof=1)) + (n2 - 1) * float(np.var(a2, ddof=1))) / (
            n1 + n2 - 2
        )
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
        df = n1 + n2 - 2
    else:
        s1_sq = float(np.var(a1, ddof=1))
        s2_sq = float(np.var(a2, ddof=1))
        se = math.sqrt(s1_sq / n1 + s2_sq / n2)
        num = (s1_sq / n1 + s2_sq / n2) ** 2
        denom = (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
        df = num / denom

    t_crit = sp_stats.t.ppf(1 - alpha_half, df=df)
    me = t_crit * se
    return ConfidenceInterval(
        lower=diff - me, upper=diff + me, confidence_level=confidence
    )
