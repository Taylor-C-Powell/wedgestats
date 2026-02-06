"""Nonparametric hypothesis tests."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats._typing import ArrayLike
from wedgestats.hypothesis._result import TestResult


def mann_whitney(
    x: ArrayLike,
    y: ArrayLike,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    Mann-Whitney U test (two independent samples).

    Parameters
    ----------
    x, y : ArrayLike
        Sample data.
    alternative : str
        ``'two-sided'``, ``'less'``, or ``'greater'``.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    stat, p = sp_stats.mannwhitneyu(ax, ay, alternative=alternative)
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        test_name="Mann-Whitney U test",
        reject=float(p) < alpha,
        alpha=alpha,
    )


def wilcoxon(
    x: ArrayLike,
    y: ArrayLike | None = None,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    Wilcoxon signed-rank test.

    If *y* is provided, tests the differences *x - y*; otherwise tests
    whether the median of *x* is zero.

    Parameters
    ----------
    x : ArrayLike
        Sample data (or first paired sample).
    y : ArrayLike or None
        Second paired sample.
    alternative : str
        ``'two-sided'``, ``'less'``, or ``'greater'``.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float) if y is not None else None
    stat, p = sp_stats.wilcoxon(ax, ay, alternative=alternative)
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        test_name="Wilcoxon signed-rank test",
        reject=float(p) < alpha,
        alpha=alpha,
    )


def kruskal_wallis(
    *groups: ArrayLike,
    alpha: float = 0.05,
) -> TestResult:
    """
    Kruskal-Wallis H test (non-parametric one-way ANOVA).

    Parameters
    ----------
    *groups : ArrayLike
        Two or more groups of observations.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    if len(groups) < 2:
        raise ValueError("Kruskal-Wallis requires at least two groups")
    arrays = [np.asarray(g, dtype=float) for g in groups]
    stat, p = sp_stats.kruskal(*arrays)
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        test_name="Kruskal-Wallis H test",
        reject=float(p) < alpha,
        alpha=alpha,
    )
