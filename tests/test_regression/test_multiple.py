"""Tests for multiple OLS regression."""

import pytest
import numpy as np

from wedgestats.regression.multiple import multiple_ols
from wedgestats.regression._result import RegressionResult


class TestMultipleOLS:
    def setup_method(self):
        np.random.seed(0)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        self.y = 5.0 + 2.0 * x1 + 3.0 * x2 + np.random.normal(0, 0.5, n)
        self.X = np.column_stack([x1, x2])
        self.result = multiple_ols(self.X, self.y)

    def test_returns_result(self):
        assert isinstance(self.result, RegressionResult)

    def test_coefficients_close(self):
        assert self.result.coefficients[0] == pytest.approx(5.0, abs=0.5)
        assert self.result.coefficients[1] == pytest.approx(2.0, abs=0.5)
        assert self.result.coefficients[2] == pytest.approx(3.0, abs=0.5)

    def test_r_squared_high(self):
        assert self.result.r_squared > 0.95

    def test_adj_r_squared(self):
        assert self.result.adj_r_squared <= self.result.r_squared

    def test_f_significant(self):
        assert self.result.f_p_value < 0.05

    def test_1d_input(self):
        x = np.array([1.0, 2, 3, 4, 5])
        y = np.array([2.0, 4, 6, 8, 10])
        r = multiple_ols(x, y)
        assert r.coefficients[1] == pytest.approx(2.0, abs=1e-10)
