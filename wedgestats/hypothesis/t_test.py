"""T-tests for means."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats._typing import ArrayLike
from wedgestats.hypothesis._result import TestResult


def one_sample_t(
    data: ArrayLike,
    mu0: float = 0,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    One-sample t-test.

    Parameters
    ----------
    data : ArrayLike
        Sample data.
    mu0 : float
        Hypothesized population mean.
    alternative : str
        ``'two-sided'``, ``'less'``, or ``'greater'``.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    arr = np.asarray(data, dtype=float)
    stat, p = sp_stats.ttest_1samp(arr, mu0, alternative=alternative)
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        test_name="One-sample t-test",
        reject=float(p) < alpha,
        alpha=alpha,
    )


def two_sample_t(
    data1: ArrayLike,
    data2: ArrayLike,
    equal_var: bool = True,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    Two-sample (independent) t-test.

    Parameters
    ----------
    data1, data2 : ArrayLike
        Sample data for each group.
    equal_var : bool
        If True, assume equal variances (Student's). If False, use Welch's
        t-test.
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
    stat, p = sp_stats.ttest_ind(a1, a2, equal_var=equal_var, alternative=alternative)
    name = "Two-sample t-test" if equal_var else "Welch's t-test"
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        test_name=name,
        reject=float(p) < alpha,
        alpha=alpha,
    )


def paired_t(
    data1: ArrayLike,
    data2: ArrayLike,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    Paired (dependent) t-test.

    Parameters
    ----------
    data1, data2 : ArrayLike
        Paired observations.
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
    stat, p = sp_stats.ttest_rel(a1, a2, alternative=alternative)
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        test_name="Paired t-test",
        reject=float(p) < alpha,
        alpha=alpha,
    )
