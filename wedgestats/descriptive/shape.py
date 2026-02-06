"""Shape statistics (skewness and kurtosis)."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats._typing import ArrayLike


def skewness(data: ArrayLike, bias: bool = True) -> float:
    """
    Skewness of the data.

    Parameters
    ----------
    data : ArrayLike
        Input data.
    bias : bool
        If False, apply bias correction (sample skewness). Default True
        computes the biased (population) estimator.

    Returns
    -------
    float
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    return float(sp_stats.skew(arr, bias=bias))


def kurtosis(data: ArrayLike, bias: bool = True, fisher: bool = True) -> float:
    """
    Kurtosis of the data.

    Parameters
    ----------
    data : ArrayLike
        Input data.
    bias : bool
        If False, apply bias correction.
    fisher : bool
        If True (default), return excess kurtosis (normal = 0).
        If False, return standard kurtosis (normal = 3).

    Returns
    -------
    float
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    return float(sp_stats.kurtosis(arr, bias=bias, fisher=fisher))
