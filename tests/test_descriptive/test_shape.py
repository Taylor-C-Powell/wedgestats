"""Tests for shape statistics."""

import pytest
from scipy import stats as sp_stats

from wedgestats.descriptive.shape import kurtosis, skewness


DATA = [2, 4, 4, 4, 5, 5, 7, 9]


class TestSkewness:
    def test_biased(self):
        expected = float(sp_stats.skew(DATA, bias=True))
        assert skewness(DATA, bias=True) == pytest.approx(expected)

    def test_unbiased(self):
        expected = float(sp_stats.skew(DATA, bias=False))
        assert skewness(DATA, bias=False) == pytest.approx(expected)

    def test_symmetric(self):
        sym = [1, 2, 3, 4, 5]
        assert skewness(sym) == pytest.approx(0.0, abs=1e-10)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            skewness([])


class TestKurtosis:
    def test_fisher(self):
        expected = float(sp_stats.kurtosis(DATA, bias=True, fisher=True))
        assert kurtosis(DATA, bias=True, fisher=True) == pytest.approx(expected)

    def test_pearson(self):
        expected = float(sp_stats.kurtosis(DATA, bias=True, fisher=False))
        assert kurtosis(DATA, bias=True, fisher=False) == pytest.approx(expected)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            kurtosis([])
