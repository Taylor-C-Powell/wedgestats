"""Tests for StudentT distribution."""

import pytest
from scipy import stats as sp_stats

from tests.conftest import ABS_TOL
from wedgestats.distributions import StudentT


class TestStudentT:
    def setup_method(self):
        self.dist = StudentT(df=10)
        self.ref = sp_stats.t(10)

    def test_pdf(self):
        for x in [-3, -1, 0, 1, 3]:
            assert self.dist.pdf(x) == pytest.approx(self.ref.pdf(x), abs=ABS_TOL)

    def test_cdf(self):
        for x in [-3, -1, 0, 1, 3]:
            assert self.dist.cdf(x) == pytest.approx(self.ref.cdf(x), abs=ABS_TOL)

    def test_ppf(self):
        for q in [0.01, 0.25, 0.5, 0.75, 0.99]:
            assert self.dist.ppf(q) == pytest.approx(self.ref.ppf(q), abs=ABS_TOL)

    def test_mean(self):
        assert self.dist.mean() == pytest.approx(0.0, abs=ABS_TOL)

    def test_variance(self):
        # Var(t_10) = 10 / (10 - 2) = 1.25
        assert self.dist.variance() == pytest.approx(1.25, abs=ABS_TOL)

    def test_repr(self):
        assert repr(self.dist) == "StudentT(df=10)"

    def test_invalid_df(self):
        with pytest.raises(ValueError):
            StudentT(0)
        with pytest.raises(ValueError):
            StudentT(-5)
