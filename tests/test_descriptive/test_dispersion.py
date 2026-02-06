"""Tests for dispersion measures."""

import numpy as np
import pytest
from scipy import stats as sp_stats

from wedgestats.descriptive.dispersion import cv, data_range, iqr, mad, std_dev, variance


DATA = [2, 4, 4, 4, 5, 5, 7, 9]


class TestVariance:
    def test_sample(self):
        assert variance(DATA) == pytest.approx(np.var(DATA, ddof=1))

    def test_population(self):
        assert variance(DATA, ddof=0) == pytest.approx(np.var(DATA, ddof=0))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            variance([])


class TestStdDev:
    def test_sample(self):
        assert std_dev(DATA) == pytest.approx(np.std(DATA, ddof=1))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            std_dev([])


class TestRange:
    def test_basic(self):
        assert data_range(DATA) == pytest.approx(7.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            data_range([])


class TestIQR:
    def test_basic(self):
        assert iqr(DATA) == pytest.approx(sp_stats.iqr(DATA))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            iqr([])


class TestMAD:
    def test_basic(self):
        expected = float(sp_stats.median_abs_deviation(DATA, scale=1.0))
        assert mad(DATA) == pytest.approx(expected)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mad([])


class TestCV:
    def test_basic(self):
        expected = float(np.std(DATA, ddof=1) / np.mean(DATA))
        assert cv(DATA) == pytest.approx(expected)

    def test_zero_mean_raises(self):
        with pytest.raises(ValueError):
            cv([-1, 1])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cv([])
