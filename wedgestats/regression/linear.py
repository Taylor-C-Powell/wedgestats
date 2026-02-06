"""Simple OLS linear regression."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats._typing import ArrayLike
from wedgestats.regression._result import RegressionResult


def simple_ols(x: ArrayLike, y: ArrayLike) -> RegressionResult:
    """
    Ordinary least-squares simple linear regression (one predictor).

    Parameters
    ----------
    x : ArrayLike
        Predictor variable.
    y : ArrayLike
        Response variable.

    Returns
    -------
    RegressionResult
    """
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    n = ax.size

    X = np.column_stack([np.ones(n), ax])
    return _ols_fit(X, ay)


def _ols_fit(X: np.ndarray, y: np.ndarray) -> RegressionResult:
    """Core OLS fitting given a design matrix *X* (with intercept column)."""
    n, p = X.shape
    beta, residuals_ss, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ beta
    resid = y - y_hat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    r_sq = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - p) if n > p else 0.0

    # Mean squared error
    mse = ss_res / (n - p) if n > p else 0.0

    # Standard errors of coefficients
    try:
        cov_matrix = mse * np.linalg.inv(X.T @ X)
        std_errors = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        std_errors = np.full(p, np.nan)

    t_stats = beta / std_errors
    p_values = np.array(
        [float(2 * sp_stats.t.sf(abs(t), df=n - p)) for t in t_stats]
    )

    # F-statistic
    ss_reg = ss_tot - ss_res
    df_reg = p - 1
    df_res = n - p
    if df_reg > 0 and df_res > 0 and mse > 0:
        f_stat = (ss_reg / df_reg) / mse
        f_p = float(sp_stats.f.sf(f_stat, df_reg, df_res))
    else:
        f_stat = 0.0
        f_p = 1.0

    return RegressionResult(
        coefficients=beta,
        residuals=resid,
        r_squared=r_sq,
        adj_r_squared=adj_r_sq,
        f_statistic=f_stat,
        f_p_value=f_p,
        std_errors=std_errors,
        t_statistics=t_stats,
        p_values=p_values,
    )
