"""Measures of central tendency."""

from collections import Counter

import numpy as np

from wedgestats._typing import ArrayLike


def mean(data: ArrayLike) -> float:
    """
    Arithmetic mean.

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
    return float(np.mean(arr))


def median(data: ArrayLike) -> float:
    """
    Median (middle value).

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
    return float(np.median(arr))


def mode(data: ArrayLike) -> float:
    """
    Mode (most frequent value). If there are multiple modes, the smallest
    is returned.

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
    counts = Counter(arr.tolist())
    max_count = max(counts.values())
    modes = sorted(k for k, v in counts.items() if v == max_count)
    return float(modes[0])


def trimmed_mean(data: ArrayLike, proportion: float = 0.1) -> float:
    """
    Trimmed mean — arithmetic mean after removing a proportion of the
    smallest and largest values.

    Parameters
    ----------
    data : ArrayLike
        Input data.
    proportion : float
        Fraction of values to trim from *each* end (0 <= proportion < 0.5).

    Returns
    -------
    float
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    if not (0 <= proportion < 0.5):
        raise ValueError("proportion must satisfy 0 <= proportion < 0.5")
    from scipy import stats as sp_stats

    return float(sp_stats.trim_mean(arr, proportion))
