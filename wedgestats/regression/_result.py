"""Result containers for regression models."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionResult:
    """
    Result of a linear regression fit.

    Attributes
    ----------
    coefficients : np.ndarray
        Estimated coefficients (intercept first, then slopes).
    residuals : np.ndarray
        Residuals (observed - predicted).
    r_squared : float
        Coefficient of determination.
    adj_r_squared : float
        Adjusted R-squared.
    f_statistic : float
        Overall F-statistic.
    f_p_value : float
        p-value for the F-statistic.
    std_errors : np.ndarray
        Standard errors of the coefficients.
    t_statistics : np.ndarray
        t-statistics for each coefficient.
    p_values : np.ndarray
        p-values for each coefficient.
    """

    coefficients: np.ndarray
    residuals: np.ndarray
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_p_value: float
    std_errors: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray


@dataclass(frozen=True)
class LogisticResult:
    """
    Result of a logistic regression fit.

    Attributes
    ----------
    coefficients : np.ndarray
        Estimated coefficients (intercept first).
    log_likelihood : float
        Log-likelihood of the fitted model.
    iterations : int
        Number of IRLS iterations.
    """

    coefficients: np.ndarray
    log_likelihood: float
    iterations: int
