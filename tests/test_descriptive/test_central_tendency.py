"""Tests for central tendency measures."""

import pytest
import numpy as np
from scipy import stats as sp_stats

from wedgestats.descriptive.central_tendency import mean, median, mode, trimmed_mean


DATA = [2, 4, 4, 4, 5, 5, 7, 9]


class TestMean:
    def test_basic(self):
        assert mean(DATA) == pytest.approx(5.0)

    def test_single(self):
        assert mean([42]) == pytest.approx(42.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mean([])


class TestMedian:
    def test_odd_length(self):
        assert median([1, 3, 5]) == pytest.approx(3.0)

    def test_even_length(self):
        assert median([1, 2, 3, 4]) == pytest.approx(2.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            median([])


class TestMode:
    def test_basic(self):
        assert mode(DATA) == pytest.approx(4.0)

    def test_unique_returns_smallest(self):
        assert mode([1, 2, 3]) == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mode([])


class TestTrimmedMean:
    def test_basic(self):
        data = list(range(1, 21))  # 1..20
        result = trimmed_mean(data, 0.1)
        expected = float(sp_stats.trim_mean(data, 0.1))
        assert result == pytest.approx(expected)

    def test_no_trim(self):
        assert trimmed_mean(DATA, 0.0) == pytest.approx(mean(DATA))

    def test_invalid_proportion(self):
        with pytest.raises(ValueError):
            trimmed_mean(DATA, 0.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            trimmed_mean([], 0.1)
