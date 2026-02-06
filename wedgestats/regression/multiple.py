"""Multiple OLS regression."""

import numpy as np

from wedgestats._typing import ArrayLike
from wedgestats.regression._result import RegressionResult
from wedgestats.regression.linear import _ols_fit


def multiple_ols(X: ArrayLike, y: ArrayLike) -> RegressionResult:
    """
    Ordinary least-squares multiple regression.

    Parameters
    ----------
    X : ArrayLike
        Predictor matrix of shape (n, k). An intercept column is added
        automatically.
    y : ArrayLike
        Response variable of length n.

    Returns
    -------
    RegressionResult
    """
    aX = np.asarray(X, dtype=float)
    ay = np.asarray(y, dtype=float)
    if aX.ndim == 1:
        aX = aX.reshape(-1, 1)
    n = aX.shape[0]
    design = np.column_stack([np.ones(n), aX])
    return _ols_fit(design, ay)
