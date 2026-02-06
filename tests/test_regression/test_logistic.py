"""Tests for logistic regression."""

import pytest
import numpy as np

from wedgestats.regression.logistic import logistic_regression
from wedgestats.regression._result import LogisticResult


class TestLogisticRegression:
    def setup_method(self):
        np.random.seed(42)
        n = 200
        x = np.random.normal(0, 1, n)
        prob = 1 / (1 + np.exp(-(1.0 + 2.0 * x)))
        y = (np.random.rand(n) < prob).astype(float)
        self.x = x
        self.y = y
        self.result = logistic_regression(x, y)

    def test_returns_result(self):
        assert isinstance(self.result, LogisticResult)

    def test_coefficients_reasonable(self):
        # intercept near 1, slope near 2
        assert self.result.coefficients[0] == pytest.approx(1.0, abs=1.0)
        assert self.result.coefficients[1] == pytest.approx(2.0, abs=1.0)

    def test_converges(self):
        assert self.result.iterations < 100

    def test_log_likelihood_negative(self):
        assert self.result.log_likelihood < 0

    def test_perfect_separation(self):
        x = np.array([1, 2, 3, 10, 11, 12], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        result = logistic_regression(x, y, max_iter=200)
        # Should converge (possibly with large coefficients)
        assert isinstance(result, LogisticResult)
