"""Tests for association statistics."""

import pytest
import numpy as np
from scipy import stats as sp_stats

from wedgestats.descriptive.association import correlation, covariance


X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = [2, 4, 5, 4, 5, 7, 8, 9, 10, 12]


class TestCovariance:
    def test_basic(self):
        expected = float(np.cov(X, Y, ddof=1)[0, 1])
        assert covariance(X, Y) == pytest.approx(expected)

    def test_population(self):
        expected = float(np.cov(X, Y, ddof=0)[0, 1])
        assert covariance(X, Y, ddof=0) == pytest.approx(expected)

    def test_mismatched_shape_raises(self):
        with pytest.raises(ValueError):
            covariance([1, 2], [1, 2, 3])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            covariance([], [])


class TestCorrelation:
    def test_pearson(self):
        expected, _ = sp_stats.pearsonr(X, Y)
        assert correlation(X, Y, method="pearson") == pytest.approx(expected)

    def test_spearman(self):
        expected, _ = sp_stats.spearmanr(X, Y)
        assert correlation(X, Y, method="spearman") == pytest.approx(expected)

    def test_kendall(self):
        expected, _ = sp_stats.kendalltau(X, Y)
        assert correlation(X, Y, method="kendall") == pytest.approx(expected)

    def test_perfect_positive(self):
        assert correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert correlation([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            correlation(X, Y, method="bogus")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            correlation([], [])
