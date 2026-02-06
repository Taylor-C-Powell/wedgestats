"""Summary statistics."""

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

from wedgestats._typing import ArrayLike


def five_number_summary(data: ArrayLike) -> tuple[float, float, float, float, float]:
    """
    Tukey's five-number summary: (min, Q1, median, Q3, max).

    Parameters
    ----------
    data : ArrayLike
        Input data.

    Returns
    -------
    tuple[float, float, float, float, float]
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    mn = float(np.min(arr))
    q1, med, q3 = (float(v) for v in np.percentile(arr, [25, 50, 75]))
    mx = float(np.max(arr))
    return mn, q1, med, q3, mx


@dataclass(frozen=True)
class DescribeResult:
    """Container for descriptive statistics returned by :func:`describe`."""

    count: int
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float
    skewness: float
    kurtosis: float


def describe(data: ArrayLike) -> DescribeResult:
    """
    Compute a suite of descriptive statistics.

    Parameters
    ----------
    data : ArrayLike
        Input data.

    Returns
    -------
    DescribeResult
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    q1, med, q3 = (float(v) for v in np.percentile(arr, [25, 50, 75]))
    return DescribeResult(
        count=int(arr.size),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)),
        min=float(np.min(arr)),
        q1=q1,
        median=med,
        q3=q3,
        max=float(np.max(arr)),
        skewness=float(sp_stats.skew(arr, bias=False)),
        kurtosis=float(sp_stats.kurtosis(arr, bias=False)),
    )
