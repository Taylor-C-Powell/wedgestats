"""Tests for Hypergeometric distribution."""

import pytest
from scipy import stats as sp_stats

from tests.conftest import ABS_TOL
from wedgestats.distributions import Hypergeometric


class TestHypergeometric:
    def setup_method(self):
        self.dist = Hypergeometric(M=50, n=20, N=10)
        self.ref = sp_stats.hypergeom(50, 20, 10)

    def test_pmf(self):
        for k in range(11):
            assert self.dist.pmf(k) == pytest.approx(self.ref.pmf(k), abs=ABS_TOL)

    def test_cdf(self):
        for k in range(11):
            assert self.dist.cdf(k) == pytest.approx(self.ref.cdf(k), abs=ABS_TOL)

    def test_mean(self):
        assert self.dist.mean() == pytest.approx(self.ref.mean(), abs=ABS_TOL)

    def test_variance(self):
        assert self.dist.variance() == pytest.approx(self.ref.var(), abs=ABS_TOL)

    def test_repr(self):
        assert repr(self.dist) == "Hypergeometric(M=50, n=20, N=10)"

    def test_invalid_n_greater_than_M(self):
        with pytest.raises(ValueError):
            Hypergeometric(M=10, n=20, N=5)

    def test_invalid_N_greater_than_M(self):
        with pytest.raises(ValueError):
            Hypergeometric(M=10, n=5, N=20)

    def test_negative_M(self):
        with pytest.raises(ValueError):
            Hypergeometric(M=-1, n=5, N=5)
