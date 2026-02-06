"""Measures of dispersion (spread)."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats._typing import ArrayLike


def variance(data: ArrayLike, ddof: int = 1) -> float:
    """
    Sample variance (ddof=1) or population variance (ddof=0).

    Parameters
    ----------
    data : ArrayLike
        Input data.
    ddof : int
        Delta degrees of freedom. Default 1 gives sample variance.

    Returns
    -------
    float
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    return float(np.var(arr, ddof=ddof))


def std_dev(data: ArrayLike, ddof: int = 1) -> float:
    """
    Standard deviation.

    Parameters
    ----------
    data : ArrayLike
        Input data.
    ddof : int
        Delta degrees of freedom. Default 1 gives sample std dev.

    Returns
    -------
    float
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    return float(np.std(arr, ddof=ddof))


def data_range(data: ArrayLike) -> float:
    """
    Range (max - min).

    Parameters
    ----------
    data : ArrayLike
        Input data.

    Returns
    -------
    float
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    return float(np.ptp(arr))


def iqr(data: ArrayLike) -> float:
    """
    Interquartile range (Q3 - Q1).

    Parameters
    ----------
    data : ArrayLike
        Input data.

    Returns
    -------
    float
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    return float(sp_stats.iqr(arr))


def mad(data: ArrayLike) -> float:
    """
    Median absolute deviation.

    Parameters
    ----------
    data : ArrayLike
        Input data.

    Returns
    -------
    float
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    return float(sp_stats.median_abs_deviation(arr, scale=1.0))


def cv(data: ArrayLike, ddof: int = 1) -> float:
    """
    Coefficient of variation (std_dev / mean).

    Parameters
    ----------
    data : ArrayLike
        Input data.
    ddof : int
        Delta degrees of freedom for std_dev calculation.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the mean is zero.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    m = float(np.mean(arr))
    if m == 0:
        raise ValueError("coefficient of variation is undefined when mean is zero")
    return float(np.std(arr, ddof=ddof) / m)
