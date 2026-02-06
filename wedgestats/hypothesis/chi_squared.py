"""Chi-squared tests."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats._typing import ArrayLike
from wedgestats.hypothesis._result import TestResult


def goodness_of_fit(
    observed: ArrayLike,
    expected: ArrayLike | None = None,
    alpha: float = 0.05,
) -> TestResult:
    """
    Chi-squared goodness-of-fit test.

    Parameters
    ----------
    observed : ArrayLike
        Observed frequencies.
    expected : ArrayLike or None
        Expected frequencies. If None, uniform expectation is assumed.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float) if expected is not None else None
    stat, p = sp_stats.chisquare(obs, f_exp=exp)
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        test_name="Chi-squared goodness-of-fit",
        reject=float(p) < alpha,
        alpha=alpha,
    )


def independence(
    contingency_table: ArrayLike,
    alpha: float = 0.05,
) -> TestResult:
    """
    Chi-squared test of independence on a contingency table.

    Parameters
    ----------
    contingency_table : ArrayLike
        2-D array of observed frequencies.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    table = np.asarray(contingency_table, dtype=float)
    stat, p, _dof, _expected = sp_stats.chi2_contingency(table)
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        test_name="Chi-squared test of independence",
        reject=float(p) < alpha,
        alpha=alpha,
    )


def homogeneity(
    contingency_table: ArrayLike,
    alpha: float = 0.05,
) -> TestResult:
    """
    Chi-squared test of homogeneity (same computation as independence,
    different interpretation).

    Parameters
    ----------
    contingency_table : ArrayLike
        2-D array of observed frequencies.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    table = np.asarray(contingency_table, dtype=float)
    stat, p, _dof, _expected = sp_stats.chi2_contingency(table)
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        test_name="Chi-squared test of homogeneity",
        reject=float(p) < alpha,
        alpha=alpha,
    )
