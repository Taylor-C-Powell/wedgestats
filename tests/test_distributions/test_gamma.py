"""Tests for Gamma distribution."""

import pytest
from scipy import stats as sp_stats

from tests.conftest import ABS_TOL
from wedgestats.distributions import Gamma


class TestGamma:
    def setup_method(self):
        self.dist = Gamma(alpha=3, beta=2)
        self.ref = sp_stats.gamma(a=3, scale=0.5)

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
        # Mean = alpha / beta = 3/2 = 1.5
        assert self.dist.mean() == pytest.approx(1.5, abs=ABS_TOL)

    def test_variance(self):
        # Var = alpha / beta^2 = 3/4 = 0.75
        assert self.dist.variance() == pytest.approx(0.75, abs=ABS_TOL)

    def test_repr(self):
        assert repr(self.dist) == "Gamma(alpha=3, beta=2)"

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            Gamma(0, 2)

    def test_invalid_beta(self):
        with pytest.raises(ValueError):
            Gamma(3, 0)
