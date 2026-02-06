"""Tests for Geometric distribution."""

import pytest
from scipy import stats as sp_stats

from tests.conftest import ABS_TOL
from wedgestats.distributions import Geometric


class TestGeometric:
    def setup_method(self):
        self.dist = Geometric(p=0.3)
        self.ref = sp_stats.geom(0.3)

    def test_pmf(self):
        for k in range(1, 15):
            assert self.dist.pmf(k) == pytest.approx(self.ref.pmf(k), abs=ABS_TOL)

    def test_cdf(self):
        for k in range(1, 15):
            assert self.dist.cdf(k) == pytest.approx(self.ref.cdf(k), abs=ABS_TOL)

    def test_mean(self):
        assert self.dist.mean() == pytest.approx(self.ref.mean(), abs=ABS_TOL)

    def test_variance(self):
        assert self.dist.variance() == pytest.approx(self.ref.var(), abs=ABS_TOL)

    def test_repr(self):
        assert repr(self.dist) == "Geometric(p=0.3)"

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            Geometric(0)
        with pytest.raises(ValueError):
            Geometric(-0.1)
        with pytest.raises(ValueError):
            Geometric(1.1)
