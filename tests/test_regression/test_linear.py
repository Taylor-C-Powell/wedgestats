"""Tests for simple OLS regression."""

import pytest
import numpy as np
from scipy import stats as sp_stats

from wedgestats.regression.linear import simple_ols
from wedgestats.regression._result import RegressionResult


class TestSimpleOLS:
    def setup_method(self):
        np.random.seed(42)
        self.x = np.arange(1, 21, dtype=float)
        self.y = 3.0 + 2.0 * self.x + np.random.normal(0, 0.5, 20)
        self.result = simple_ols(self.x, self.y)

    def test_returns_result(self):
        assert isinstance(self.result, RegressionResult)

    def test_coefficients_close(self):
        # Intercept near 3, slope near 2
        assert self.result.coefficients[0] == pytest.approx(3.0, abs=1.5)
        assert self.result.coefficients[1] == pytest.approx(2.0, abs=0.2)

    def test_r_squared_high(self):
        assert self.result.r_squared > 0.99

    def test_residuals_shape(self):
        assert self.result.residuals.shape == (20,)

    def test_against_scipy_linregress(self):
        ref = sp_stats.linregress(self.x, self.y)
        assert self.result.coefficients[1] == pytest.approx(ref.slope, abs=1e-10)
        assert self.result.coefficients[0] == pytest.approx(ref.intercept, abs=1e-10)

    def test_perfect_fit(self):
        x = [1, 2, 3, 4, 5]
        y = [3, 5, 7, 9, 11]  # y = 1 + 2x exactly
        r = simple_ols(x, y)
        assert r.coefficients[0] == pytest.approx(1.0, abs=1e-10)
        assert r.coefficients[1] == pytest.approx(2.0, abs=1e-10)
        assert r.r_squared == pytest.approx(1.0, abs=1e-10)
