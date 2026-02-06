"""Tests for t-tests."""

import pytest
import numpy as np
from scipy import stats as sp_stats

from wedgestats.hypothesis.t_test import one_sample_t, paired_t, two_sample_t
from wedgestats.hypothesis._result import TestResult


class TestOneSampleT:
    def test_against_scipy(self):
        data = [2.3, 1.9, 2.5, 2.1, 2.8, 2.0, 2.4]
        result = one_sample_t(data, mu0=2.0)
        ref_stat, ref_p = sp_stats.ttest_1samp(data, 2.0)
        assert result.statistic == pytest.approx(ref_stat, abs=1e-10)
        assert result.p_value == pytest.approx(ref_p, abs=1e-10)
        assert result.test_name == "One-sample t-test"

    def test_frozen_result(self):
        result = one_sample_t([1, 2, 3], mu0=0)
        assert isinstance(result, TestResult)
        with pytest.raises(AttributeError):
            result.statistic = 0  # type: ignore[misc]


class TestTwoSampleT:
    def test_equal_var(self):
        a = [5.1, 4.9, 5.0, 5.2, 5.1]
        b = [4.8, 4.7, 4.9, 5.0, 4.8]
        result = two_sample_t(a, b, equal_var=True)
        ref_stat, ref_p = sp_stats.ttest_ind(a, b, equal_var=True)
        assert result.statistic == pytest.approx(ref_stat, abs=1e-10)
        assert result.p_value == pytest.approx(ref_p, abs=1e-10)
        assert result.test_name == "Two-sample t-test"

    def test_welch(self):
        a = [5.1, 4.9, 5.0, 5.2, 5.1]
        b = [4.8, 4.7, 4.9, 5.0, 4.8]
        result = two_sample_t(a, b, equal_var=False)
        ref_stat, ref_p = sp_stats.ttest_ind(a, b, equal_var=False)
        assert result.statistic == pytest.approx(ref_stat, abs=1e-10)
        assert result.p_value == pytest.approx(ref_p, abs=1e-10)
        assert result.test_name == "Welch's t-test"


class TestPairedT:
    def test_against_scipy(self):
        before = [120, 130, 125, 140, 135]
        after = [115, 125, 120, 135, 130]
        result = paired_t(before, after)
        ref_stat, ref_p = sp_stats.ttest_rel(before, after)
        assert result.statistic == pytest.approx(ref_stat, abs=1e-10)
        assert result.p_value == pytest.approx(ref_p, abs=1e-10)
        assert result.test_name == "Paired t-test"
