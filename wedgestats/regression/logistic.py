"""Binary logistic regression via iteratively reweighted least squares."""

import numpy as np

from wedgestats._typing import ArrayLike
from wedgestats.regression._result import LogisticResult


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression(
    X: ArrayLike,
    y: ArrayLike,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> LogisticResult:
    """
    Binary logistic regression fitted with IRLS (Newton-Raphson).

    Parameters
    ----------
    X : ArrayLike
        Predictor matrix of shape (n, k). An intercept column is added.
    y : ArrayLike
        Binary response (0/1) of length n.
    max_iter : int
        Maximum number of IRLS iterations.
    tol : float
        Convergence tolerance on the coefficient change.

    Returns
    -------
    LogisticResult
    """
    aX = np.asarray(X, dtype=float)
    ay = np.asarray(y, dtype=float)
    if aX.ndim == 1:
        aX = aX.reshape(-1, 1)
    n = aX.shape[0]
    design = np.column_stack([np.ones(n), aX])
    p = design.shape[1]

    beta = np.zeros(p)
    for i in range(1, max_iter + 1):
        mu = _sigmoid(design @ beta)
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        W = np.diag(mu * (1 - mu))
        z = design @ beta + np.linalg.solve(W, ay - mu)
        try:
            beta_new = np.linalg.solve(design.T @ W @ design, design.T @ W @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    mu = _sigmoid(design @ beta)
    mu = np.clip(mu, 1e-15, 1 - 1e-15)
    ll = float(np.sum(ay * np.log(mu) + (1 - ay) * np.log(1 - mu)))

    return LogisticResult(
        coefficients=beta,
        log_likelihood=ll,
        iterations=i,
    )
