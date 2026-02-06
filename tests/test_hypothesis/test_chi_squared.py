"""Tests for chi-squared tests."""

import pytest
from scipy import stats as sp_stats

from wedgestats.hypothesis.chi_squared import goodness_of_fit, homogeneity, independence


class TestGoodnessOfFit:
    def test_uniform(self):
        observed = [18, 22, 20, 25, 15]
        result = goodness_of_fit(observed)
        ref_stat, ref_p = sp_stats.chisquare(observed)
        assert result.statistic == pytest.approx(ref_stat, abs=1e-10)
        assert result.p_value == pytest.approx(ref_p, abs=1e-10)

    def test_expected(self):
        observed = [18, 22, 20, 25, 15]
        expected = [20, 20, 20, 20, 20]
        result = goodness_of_fit(observed, expected)
        ref_stat, ref_p = sp_stats.chisquare(observed, expected)
        assert result.statistic == pytest.approx(ref_stat, abs=1e-10)
        assert result.p_value == pytest.approx(ref_p, abs=1e-10)


class TestIndependence:
    def test_basic(self):
        table = [[10, 20, 30], [6, 9, 17]]
        result = independence(table)
        ref_stat, ref_p, _, _ = sp_stats.chi2_contingency(table)
        assert result.statistic == pytest.approx(ref_stat, abs=1e-10)
        assert result.p_value == pytest.approx(ref_p, abs=1e-10)
        assert result.test_name == "Chi-squared test of independence"


class TestHomogeneity:
    def test_basic(self):
        table = [[10, 20, 30], [6, 9, 17]]
        result = homogeneity(table)
        assert result.test_name == "Chi-squared test of homogeneity"
        # Same computation as independence
        ref = independence(table)
        assert result.statistic == pytest.approx(ref.statistic, abs=1e-10)
