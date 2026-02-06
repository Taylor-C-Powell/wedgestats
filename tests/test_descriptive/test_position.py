"""Tests for positional statistics."""

import numpy as np
import pytest

from wedgestats.descriptive.position import percentile, quartiles, z_score


DATA = [2, 4, 4, 4, 5, 5, 7, 9]


class TestPercentile:
    def test_median(self):
        assert percentile(DATA, 50) == pytest.approx(np.median(DATA))

    def test_min(self):
        assert percentile(DATA, 0) == pytest.approx(2.0)

    def test_max(self):
        assert percentile(DATA, 100) == pytest.approx(9.0)

    def test_invalid_q(self):
        with pytest.raises(ValueError):
            percentile(DATA, -1)
        with pytest.raises(ValueError):
            percentile(DATA, 101)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            percentile([], 50)


class TestQuartiles:
    def test_basic(self):
        q1, q2, q3 = quartiles(DATA)
        assert q1 == pytest.approx(np.percentile(DATA, 25))
        assert q2 == pytest.approx(np.percentile(DATA, 50))
        assert q3 == pytest.approx(np.percentile(DATA, 75))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            quartiles([])


class TestZScore:
    def test_standardized(self):
        result = z_score(DATA)
        assert result.mean() == pytest.approx(0.0, abs=1e-10)
        assert np.std(result, ddof=0) == pytest.approx(1.0, abs=1e-10)

    def test_shape(self):
        result = z_score([1, 2, 3])
        assert result.shape == (3,)

    def test_zero_std_raises(self):
        with pytest.raises(ValueError):
            z_score([5, 5, 5])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            z_score([])
