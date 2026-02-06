"""Tests for NegativeBinomial distribution."""

import pytest
from scipy import stats as sp_stats

from tests.conftest import ABS_TOL
from wedgestats.distributions import NegativeBinomial


class TestNegativeBinomial:
    def setup_method(self):
        self.dist = NegativeBinomial(r=5, p=0.4)
        self.ref = sp_stats.nbinom(5, 0.4)

    def test_pmf(self):
        for k in range(20):
            assert self.dist.pmf(k) == pytest.approx(self.ref.pmf(k), abs=ABS_TOL)

    def test_cdf(self):
        for k in range(20):
            assert self.dist.cdf(k) == pytest.approx(self.ref.cdf(k), abs=ABS_TOL)

    def test_mean(self):
        assert self.dist.mean() == pytest.approx(self.ref.mean(), abs=ABS_TOL)

    def test_variance(self):
        assert self.dist.variance() == pytest.approx(self.ref.var(), abs=ABS_TOL)

    def test_repr(self):
        assert repr(self.dist) == "NegativeBinomial(r=5, p=0.4)"

    def test_invalid_r(self):
        with pytest.raises(ValueError):
            NegativeBinomial(0, 0.5)

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            NegativeBinomial(5, 0)
        with pytest.raises(ValueError):
            NegativeBinomial(5, 1.5)
