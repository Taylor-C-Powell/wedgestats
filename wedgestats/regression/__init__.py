"""Regression analysis for wedgestats."""

from wedgestats.regression._result import LogisticResult, RegressionResult
from wedgestats.regression.diagnostics import (
    cooks_distance,
    durbin_watson,
    residuals,
    standardized_residuals,
    vif,
)
from wedgestats.regression.linear import simple_ols
from wedgestats.regression.logistic import logistic_regression
from wedgestats.regression.multiple import multiple_ols

__all__ = [
    "LogisticResult",
    "RegressionResult",
    "cooks_distance",
    "durbin_watson",
    "logistic_regression",
    "multiple_ols",
    "residuals",
    "simple_ols",
    "standardized_residuals",
    "vif",
]
