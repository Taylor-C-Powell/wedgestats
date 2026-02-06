"""Tests for confidence intervals."""

import math

import pytest
import numpy as np
from scipy import stats as sp_stats

from wedgestats.hypothesis.confidence import (
    ConfidenceInterval,
    ci_difference_of_means,
    ci_mean,
    ci_proportion,
)


class TestCIMean:
    def test_z_interval(self):
        data = [52, 48, 55, 50, 47, 53, 49, 51]
        result = ci_mean(data, confidence=0.95, sigma=5)
        assert isinstance(result, ConfidenceInterval)
        assert result.confidence_level == 0.95
        xbar = float(np.mean(data))
        me = 1.96 * 5 / math.sqrt(len(data))
        assert result.lower == pytest.approx(xbar - me, abs=0.01)
        assert result.upper == pytest.approx(xbar + me, abs=0.01)

    def test_t_interval(self):
        data = [52, 48, 55, 50, 47, 53, 49, 51]
        result = ci_mean(data, confidence=0.95)
        assert result.lower < float(np.mean(data)) < result.upper


class TestCIProportion:
    def test_basic(self):
        result = ci_proportion(60, 100, confidence=0.95)
        assert result.lower < 0.6 < result.upper
        assert result.confidence_level == 0.95


class TestCIDiffMeans:
    def test_equal_var(self):
        a = [5.1, 4.9, 5.0, 5.2, 5.1, 5.0]
        b = [4.8, 4.7, 4.9, 5.0, 4.8, 4.9]
        result = ci_difference_of_means(a, b, confidence=0.95)
        diff = float(np.mean(a)) - float(np.mean(b))
        assert result.lower < diff < result.upper

    def test_welch(self):
        a = [5.1, 4.9, 5.0, 5.2, 5.1, 5.0]
        b = [4.8, 4.7, 4.9, 5.0, 4.8, 4.9]
        result = ci_difference_of_means(a, b, confidence=0.95, equal_var=False)
        assert isinstance(result, ConfidenceInterval)
