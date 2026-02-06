"""Tests for FDistribution."""

import pytest
from scipy import stats as sp_stats

from tests.conftest import ABS_TOL
from wedgestats.distributions import FDistribution


class TestFDistribution:
    def setup_method(self):
        self.dist = FDistribution(df1=5, df2=10)
        self.ref = sp_stats.f(5, 10)

    def test_pdf(self):
        for x in [0.1, 0.5, 1, 2, 5]:
            assert self.dist.pdf(x) == pytest.approx(self.ref.pdf(x), abs=ABS_TOL)

    def test_cdf(self):
        for x in [0.1, 0.5, 1, 2, 5]:
            assert self.dist.cdf(x) == pytest.approx(self.ref.cdf(x), abs=ABS_TOL)

    def test_ppf(self):
        for q in [0.01, 0.25, 0.5, 0.75, 0.99]:
            assert self.dist.ppf(q) == pytest.approx(self.ref.ppf(q), abs=ABS_TOL)

    def test_mean(self):
        # Mean = df2 / (df2 - 2) = 10/8 = 1.25
        assert self.dist.mean() == pytest.approx(self.ref.mean(), abs=ABS_TOL)

    def test_variance(self):
        assert self.dist.variance() == pytest.approx(self.ref.var(), abs=ABS_TOL)

    def test_repr(self):
        assert repr(self.dist) == "FDistribution(df1=5, df2=10)"

    def test_invalid_df1(self):
        with pytest.raises(ValueError):
            FDistribution(0, 10)

    def test_invalid_df2(self):
        with pytest.raises(ValueError):
            FDistribution(5, 0)
