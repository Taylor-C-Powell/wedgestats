"""Tests for Exponential distribution."""

import pytest
from scipy import stats as sp_stats

from tests.conftest import ABS_TOL
from wedgestats.distributions import Exponential


class TestExponential:
    def setup_method(self):
        self.dist = Exponential(lam=2.0)
        self.ref = sp_stats.expon(scale=0.5)

    def test_pdf(self):
        for x in [0, 0.5, 1, 2, 5]:
            assert self.dist.pdf(x) == pytest.approx(self.ref.pdf(x), abs=ABS_TOL)

    def test_cdf(self):
        for x in [0, 0.5, 1, 2, 5]:
            assert self.dist.cdf(x) == pytest.approx(self.ref.cdf(x), abs=ABS_TOL)

    def test_ppf(self):
        for q in [0.01, 0.25, 0.5, 0.75, 0.99]:
            assert self.dist.ppf(q) == pytest.approx(self.ref.ppf(q), abs=ABS_TOL)

    def test_mean(self):
        assert self.dist.mean() == pytest.approx(0.5, abs=ABS_TOL)

    def test_variance(self):
        assert self.dist.variance() == pytest.approx(0.25, abs=ABS_TOL)

    def test_repr(self):
        assert repr(self.dist) == "Exponential(lam=2.0)"

    def test_invalid_lam(self):
        with pytest.raises(ValueError):
            Exponential(0)
        with pytest.raises(ValueError):
            Exponential(-1)
