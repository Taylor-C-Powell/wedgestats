"""Positional statistics (percentiles, quartiles, z-scores)."""

import numpy as np

from wedgestats._typing import ArrayLike


def percentile(data: ArrayLike, q: float) -> float:
    """
    Compute the *q*-th percentile of the data.

    Parameters
    ----------
    data : ArrayLike
        Input data.
    q : float
        Percentile to compute (0 <= q <= 100).

    Returns
    -------
    float
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    if not (0 <= q <= 100):
        raise ValueError("q must be between 0 and 100")
    return float(np.percentile(arr, q))


def quartiles(data: ArrayLike) -> tuple[float, float, float]:
    """
    Compute Q1, Q2 (median), and Q3.

    Parameters
    ----------
    data : ArrayLike
        Input data.

    Returns
    -------
    tuple[float, float, float]
        (Q1, Q2, Q3)
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    q1, q2, q3 = np.percentile(arr, [25, 50, 75])
    return float(q1), float(q2), float(q3)


def z_score(data: ArrayLike, ddof: int = 0) -> np.ndarray:
    """
    Compute z-scores (standard scores) for each element.

    Parameters
    ----------
    data : ArrayLike
        Input data.
    ddof : int
        Delta degrees of freedom for std calculation. Default 0 gives
        population z-scores.

    Returns
    -------
    np.ndarray
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    m = np.mean(arr)
    s = np.std(arr, ddof=ddof)
    if s == 0:
        raise ValueError("standard deviation is zero; z-scores undefined")
    return (arr - m) / s
