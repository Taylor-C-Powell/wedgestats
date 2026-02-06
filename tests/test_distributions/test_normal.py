"""Tests for Normal distribution."""

import pytest
from scipy import stats as sp_stats

from tests.conftest import ABS_TOL
from wedgestats.distributions import Normal


class TestNormal:
    def setup_method(self):
        self.dist = Normal(mu=5, sigma=2)
        self.ref = sp_stats.norm(loc=5, scale=2)

    def test_pdf(self):
        for x in [-1, 0, 3, 5, 7, 10]:
            assert self.dist.pdf(x) == pytest.approx(self.ref.pdf(x), abs=ABS_TOL)

    def test_cdf(self):
        for x in [-1, 0, 3, 5, 7, 10]:
            assert self.dist.cdf(x) == pytest.approx(self.ref.cdf(x), abs=ABS_TOL)

    def test_ppf(self):
        for q in [0.01, 0.25, 0.5, 0.75, 0.99]:
            assert self.dist.ppf(q) == pytest.approx(self.ref.ppf(q), abs=ABS_TOL)

    def test_mean(self):
        assert self.dist.mean() == pytest.approx(5.0, abs=ABS_TOL)

    def test_variance(self):
        assert self.dist.variance() == pytest.approx(4.0, abs=ABS_TOL)

    def test_standard_normal(self):
        std = Normal()
        assert std.mean() == pytest.approx(0.0, abs=ABS_TOL)
        assert std.variance() == pytest.approx(1.0, abs=ABS_TOL)

    def test_repr(self):
        assert repr(self.dist) == "Normal(mu=5, sigma=2)"

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            Normal(0, 0)
        with pytest.raises(ValueError):
            Normal(0, -1)
