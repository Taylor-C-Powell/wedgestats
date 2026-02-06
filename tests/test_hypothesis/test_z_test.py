"""Tests for z-tests."""

import pytest
import numpy as np
from scipy import stats as sp_stats

from wedgestats.hypothesis.z_test import one_sample_z, proportion_z, two_sample_z
from wedgestats.hypothesis._result import TestResult


class TestOneSampleZ:
    def test_no_effect(self):
        np.random.seed(0)
        data = np.random.normal(100, 15, 50)
        result = one_sample_z(data, mu0=float(np.mean(data)), sigma=15)
        assert isinstance(result, TestResult)
        assert result.statistic == pytest.approx(0.0, abs=1e-10)
        assert result.p_value == pytest.approx(1.0, abs=1e-5)

    def test_large_effect(self):
        data = [105] * 100  # all values = 105
        result = one_sample_z(data, mu0=100, sigma=15)
        assert result.reject is True

    def test_one_sided_greater(self):
        data = [105] * 100
        result = one_sample_z(data, mu0=100, sigma=15, alternative="greater")
        assert result.p_value < 0.05

    def test_one_sided_less(self):
        data = [95] * 100
        result = one_sample_z(data, mu0=100, sigma=15, alternative="less")
        assert result.p_value < 0.05


class TestTwoSampleZ:
    def test_same_populations(self):
        np.random.seed(42)
        d1 = np.random.normal(100, 15, 50)
        d2 = np.random.normal(100, 15, 50)
        result = two_sample_z(d1, d2, sigma1=15, sigma2=15)
        assert isinstance(result, TestResult)
        # Should generally not reject when populations are the same
        assert result.test_name == "Two-sample z-test"

    def test_different_populations(self):
        d1 = [120] * 50
        d2 = [100] * 50
        result = two_sample_z(d1, d2, sigma1=15, sigma2=15)
        assert result.reject is True


class TestProportionZ:
    def test_equal_proportions(self):
        # 50/100 vs 50/100
        result = proportion_z(50, 100, 50, 100)
        assert result.statistic == pytest.approx(0.0, abs=1e-10)
        assert result.p_value == pytest.approx(1.0, abs=1e-5)

    def test_different_proportions(self):
        result = proportion_z(80, 100, 50, 100)
        assert result.reject is True
