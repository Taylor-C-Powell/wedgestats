"""ANOVA and post-hoc tests."""

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

from wedgestats._typing import ArrayLike
from wedgestats.hypothesis._result import TestResult


def one_way(*groups: ArrayLike, alpha: float = 0.05) -> TestResult:
    """
    One-way ANOVA (F-test).

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
        raise ValueError("ANOVA requires at least two groups")
    arrays = [np.asarray(g, dtype=float) for g in groups]
    stat, p = sp_stats.f_oneway(*arrays)
    return TestResult(
        statistic=float(stat),
        p_value=float(p),
        test_name="One-way ANOVA",
        reject=float(p) < alpha,
        alpha=alpha,
    )


@dataclass(frozen=True)
class TukeyResult:
    """Result of Tukey's HSD post-hoc test."""

    group1: int
    group2: int
    mean_diff: float
    p_value: float
    reject: bool


def tukey_hsd(*groups: ArrayLike, alpha: float = 0.05) -> list[TukeyResult]:
    """
    Tukey's Honest Significant Difference post-hoc test.

    Parameters
    ----------
    *groups : ArrayLike
        Two or more groups of observations.
    alpha : float
        Significance level.

    Returns
    -------
    list[TukeyResult]
        Pairwise comparison results.
    """
    if len(groups) < 2:
        raise ValueError("Tukey HSD requires at least two groups")
    arrays = [np.asarray(g, dtype=float) for g in groups]
    result = sp_stats.tukey_hsd(*arrays)
    comparisons = []
    k = len(arrays)
    for i in range(k):
        for j in range(i + 1, k):
            pval = float(result.pvalue[i][j])
            comparisons.append(
                TukeyResult(
                    group1=i,
                    group2=j,
                    mean_diff=float(np.mean(arrays[i]) - np.mean(arrays[j])),
                    p_value=pval,
                    reject=pval < alpha,
                )
            )
    return comparisons
