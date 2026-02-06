"""Tests for Beta distribution."""

import pytest
from scipy import stats as sp_stats

from tests.conftest import ABS_TOL
from wedgestats.distributions import Beta


class TestBeta:
    def setup_method(self):
        self.dist = Beta(alpha=2, beta=5)
        self.ref = sp_stats.beta(2, 5)

    def test_pdf(self):
        for x in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert self.dist.pdf(x) == pytest.approx(self.ref.pdf(x), abs=ABS_TOL)

    def test_cdf(self):
        for x in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert self.dist.cdf(x) == pytest.approx(self.ref.cdf(x), abs=ABS_TOL)

    def test_ppf(self):
        for q in [0.01, 0.25, 0.5, 0.75, 0.99]:
            assert self.dist.ppf(q) == pytest.approx(self.ref.ppf(q), abs=ABS_TOL)

    def test_mean(self):
        # Mean = alpha / (alpha + beta) = 2/7
        assert self.dist.mean() == pytest.approx(2.0 / 7.0, abs=ABS_TOL)

    def test_variance(self):
        assert self.dist.variance() == pytest.approx(self.ref.var(), abs=ABS_TOL)

    def test_repr(self):
        assert repr(self.dist) == "Beta(alpha=2, beta=5)"

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            Beta(0, 5)

    def test_invalid_beta(self):
        with pytest.raises(ValueError):
            Beta(2, 0)
