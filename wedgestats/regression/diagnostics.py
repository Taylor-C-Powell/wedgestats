"""Regression diagnostics."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats._typing import ArrayLike


def residuals(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """
    Raw residuals.

    Parameters
    ----------
    y_true, y_pred : ArrayLike
        Observed and predicted values.

    Returns
    -------
    np.ndarray
    """
    return np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)


def standardized_residuals(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """
    Standardized residuals (residuals / std(residuals)).

    Parameters
    ----------
    y_true, y_pred : ArrayLike
        Observed and predicted values.

    Returns
    -------
    np.ndarray
    """
    r = residuals(y_true, y_pred)
    s = np.std(r, ddof=1)
    if s == 0:
        raise ValueError("residual standard deviation is zero")
    return r / s


def vif(X: ArrayLike) -> np.ndarray:
    """
    Variance inflation factors for each predictor column.

    Parameters
    ----------
    X : ArrayLike
        Predictor matrix of shape (n, k), *without* an intercept column.

    Returns
    -------
    np.ndarray
        VIF for each of the k predictors.
    """
    aX = np.asarray(X, dtype=float)
    if aX.ndim == 1:
        aX = aX.reshape(-1, 1)
    k = aX.shape[1]
    vifs = np.empty(k)
    for j in range(k):
        y_j = aX[:, j]
        X_j = np.delete(aX, j, axis=1)
        if X_j.shape[1] == 0:
            vifs[j] = 1.0
            continue
        X_j = np.column_stack([np.ones(aX.shape[0]), X_j])
        beta, _, _, _ = np.linalg.lstsq(X_j, y_j, rcond=None)
        y_hat = X_j @ beta
        ss_res = float(np.sum((y_j - y_hat) ** 2))
        ss_tot = float(np.sum((y_j - np.mean(y_j)) ** 2))
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[j] = 1.0 / (1.0 - r_sq) if r_sq < 1.0 else np.inf
    return vifs


def cooks_distance(X: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Cook's distance for each observation.

    Parameters
    ----------
    X : ArrayLike
        Predictor matrix (n, k) *without* intercept.
    y : ArrayLike
        Response variable.

    Returns
    -------
    np.ndarray
        Cook's distance for each of the n observations.
    """
    aX = np.asarray(X, dtype=float)
    ay = np.asarray(y, dtype=float)
    if aX.ndim == 1:
        aX = aX.reshape(-1, 1)
    n, k = aX.shape
    design = np.column_stack([np.ones(n), aX])
    p = design.shape[1]

    H = design @ np.linalg.solve(design.T @ design, design.T)
    h = np.diag(H)

    beta = np.linalg.lstsq(design, ay, rcond=None)[0]
    e = ay - design @ beta
    mse = float(np.sum(e**2)) / (n - p)

    return (e**2 / (p * mse)) * (h / (1 - h) ** 2)


def durbin_watson(residuals_arr: ArrayLike) -> float:
    """
    Durbin-Watson statistic for autocorrelation in residuals.

    Parameters
    ----------
    residuals_arr : ArrayLike
        Regression residuals.

    Returns
    -------
    float
        Durbin-Watson statistic (range 0 to 4; 2 = no autocorrelation).
    """
    r = np.asarray(residuals_arr, dtype=float)
    diff = np.diff(r)
    return float(np.sum(diff**2) / np.sum(r**2))
