"""Association statistics (correlation, covariance)."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats._typing import ArrayLike


def covariance(x: ArrayLike, y: ArrayLike, ddof: int = 1) -> float:
    """
    Sample covariance between *x* and *y*.

    Parameters
    ----------
    x, y : ArrayLike
        Input arrays of equal length.
    ddof : int
        Delta degrees of freedom (default 1 for sample covariance).

    Returns
    -------
    float
    """
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    if ax.size == 0 or ay.size == 0:
        raise ValueError("data must not be empty")
    if ax.shape != ay.shape:
        raise ValueError("x and y must have the same shape")
    return float(np.cov(ax, ay, ddof=ddof)[0, 1])


def correlation(
    x: ArrayLike,
    y: ArrayLike,
    method: str = "pearson",
) -> float:
    """
    Correlation coefficient between *x* and *y*.

    Parameters
    ----------
    x, y : ArrayLike
        Input arrays of equal length.
    method : str
        One of ``'pearson'``, ``'spearman'``, ``'kendall'``.

    Returns
    -------
    float
        Correlation coefficient.
    """
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    if ax.size == 0 or ay.size == 0:
        raise ValueError("data must not be empty")
    if ax.shape != ay.shape:
        raise ValueError("x and y must have the same shape")

    match method:
        case "pearson":
            r, _ = sp_stats.pearsonr(ax, ay)
        case "spearman":
            r, _ = sp_stats.spearmanr(ax, ay)
        case "kendall":
            r, _ = sp_stats.kendalltau(ax, ay)
        case _:
            raise ValueError(
                f"Unknown method {method!r}; choose 'pearson', 'spearman', or 'kendall'"
            )
    return float(r)
