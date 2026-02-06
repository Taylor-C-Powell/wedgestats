"""Tests for ANOVA."""

import pytest
from scipy import stats as sp_stats

from wedgestats.hypothesis.anova import one_way, tukey_hsd


G1 = [6.1, 5.8, 6.3, 5.9, 6.0]
G2 = [7.2, 6.8, 7.0, 7.1, 6.9]
G3 = [5.5, 5.3, 5.7, 5.6, 5.4]


class TestOneWay:
    def test_against_scipy(self):
        result = one_way(G1, G2, G3)
        ref_stat, ref_p = sp_stats.f_oneway(G1, G2, G3)
        assert result.statistic == pytest.approx(ref_stat, abs=1e-10)
        assert result.p_value == pytest.approx(ref_p, abs=1e-10)
        assert result.test_name == "One-way ANOVA"

    def test_needs_two_groups(self):
        with pytest.raises(ValueError):
            one_way(G1)


class TestTukeyHSD:
    def test_returns_pairwise(self):
        results = tukey_hsd(G1, G2, G3)
        # 3 groups → 3 pairwise comparisons
        assert len(results) == 3

    def test_needs_two_groups(self):
        with pytest.raises(ValueError):
            tukey_hsd(G1)
