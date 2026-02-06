"""Tests for regression diagnostics."""

import pytest
import numpy as np

from wedgestats.regression.diagnostics import (
    cooks_distance,
    durbin_watson,
    residuals,
    standardized_residuals,
    vif,
)


class TestResiduals:
    def test_basic(self):
        y_true = [3, 5, 7]
        y_pred = [2.8, 5.1, 7.0]
        r = residuals(y_true, y_pred)
        expected = np.array([0.2, -0.1, 0.0])
        np.testing.assert_allclose(r, expected, atol=1e-10)


class TestStandardizedResiduals:
    def test_shape(self):
        y_true = [3, 5, 7, 9, 11]
        y_pred = [2.8, 5.1, 7.0, 8.7, 11.2]
        sr = standardized_residuals(y_true, y_pred)
        assert sr.shape == (5,)

    def test_zero_std_raises(self):
        with pytest.raises(ValueError):
            standardized_residuals([1, 2, 3], [1, 2, 3])


class TestVIF:
    def test_uncorrelated(self):
        np.random.seed(0)
        X = np.random.normal(0, 1, (100, 3))
        v = vif(X)
        # VIFs should all be close to 1 for uncorrelated predictors
        for val in v:
            assert val == pytest.approx(1.0, abs=0.5)

    def test_highly_correlated(self):
        np.random.seed(0)
        x1 = np.random.normal(0, 1, 100)
        x2 = x1 + np.random.normal(0, 0.01, 100)  # nearly identical
        X = np.column_stack([x1, x2])
        v = vif(X)
        assert v[0] > 10  # high VIF
        assert v[1] > 10

    def test_single_predictor(self):
        x = np.random.normal(0, 1, 50)
        v = vif(x)
        assert v[0] == pytest.approx(1.0)


class TestCooksDistance:
    def test_shape(self):
        np.random.seed(0)
        x = np.arange(1, 11, dtype=float)
        y = 2 * x + np.random.normal(0, 0.5, 10)
        cd = cooks_distance(x, y)
        assert cd.shape == (10,)

    def test_all_nonnegative(self):
        np.random.seed(0)
        x = np.arange(1, 11, dtype=float)
        y = 2 * x + np.random.normal(0, 0.5, 10)
        cd = cooks_distance(x, y)
        assert np.all(cd >= 0)


class TestDurbinWatson:
    def test_no_autocorrelation(self):
        np.random.seed(0)
        r = np.random.normal(0, 1, 100)
        dw = durbin_watson(r)
        # Should be near 2 for independent residuals
        assert 1.5 < dw < 2.5

    def test_positive_autocorrelation(self):
        # Residuals with strong positive autocorrelation
        r = np.cumsum(np.random.normal(0, 1, 100))
        dw = durbin_watson(r)
        assert dw < 1.0  # close to 0 for positive autocorrelation

    def test_range(self):
        dw = durbin_watson([1, -1, 1, -1, 1, -1])
        assert 0 <= dw <= 4
