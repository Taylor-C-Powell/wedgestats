"""Tests for the reference-distribution registry and its estimators."""

import numpy as np
import pytest

from wedgestats.distributions import ContinuousDistribution

from wedgelab.models import (
    DISTRIBUTIONS,
    FIT_METHODS,
    MAD_CONSISTENCY,
    FitError,
    distribution_keys,
    fit,
    sample_skewness_hint,
)

RNG = np.random.default_rng(0)

SAMPLES = {
    "normal": RNG.normal(10.0, 3.0, 400),
    "exponential": RNG.exponential(4.0, 400),
    "gamma": RNG.gamma(3.0, 2.0, 400),
    "beta": RNG.beta(2.0, 5.0, 400),
    "t": RNG.standard_t(6, 400),
    "chi2": RNG.chisquare(5, 400),
    "uniform": RNG.uniform(-2.0, 5.0, 400),
    "f": RNG.f(5, 12, 400),
}


class TestRegistry:
    def test_keys_are_stable(self):
        assert distribution_keys() == tuple(DISTRIBUTIONS)

    def test_every_spec_key_matches_its_slot(self):
        for key, spec in DISTRIBUTIONS.items():
            assert spec.key == key

    def test_every_spec_is_documented(self):
        for spec in DISTRIBUTIONS.values():
            assert spec.label and spec.param_names

    def test_support_is_known(self):
        for spec in DISTRIBUTIONS.values():
            assert spec.support in {"real", "positive", "unit"}


class TestFitting:
    @pytest.mark.parametrize("key", list(DISTRIBUTIONS))
    def test_mle_produces_a_wedgestats_distribution(self, key):
        result = fit(SAMPLES[key], key, "mle")
        assert isinstance(result.dist, ContinuousDistribution)
        assert len(result.params) == len(result.spec.param_names)

    @pytest.mark.parametrize("key", list(DISTRIBUTIONS))
    @pytest.mark.parametrize("method", ["mle", "moments", "robust"])
    def test_every_method_returns_a_usable_fit(self, key, method):
        result = fit(SAMPLES[key], key, method)
        assert np.isfinite(result.dist.ppf(0.5))
        assert result.method in FIT_METHODS

    def test_unavailable_method_falls_back_and_says_so(self):
        result = fit(SAMPLES["gamma"], "gamma", "robust")
        assert result.method == "mle"
        assert result.fell_back
        assert any("not defined" in note for note in result.notes)

    def test_available_method_does_not_fall_back(self):
        assert not fit(SAMPLES["normal"], "normal", "robust").fell_back


class TestEstimatorCorrectness:
    def test_normal_mle_recovers_parameters(self):
        result = fit(np.random.default_rng(1).normal(50.0, 7.0, 5000), "normal", "mle")
        mu, sigma = result.params
        assert mu == pytest.approx(50.0, abs=0.5)
        assert sigma == pytest.approx(7.0, abs=0.4)

    def test_normal_moments_uses_bessel_correction(self):
        data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        _, sigma = fit(data, "normal", "moments").params
        assert sigma == pytest.approx(float(np.std(data, ddof=1)))

    def test_robust_uses_median_and_scaled_mad(self):
        data = np.random.default_rng(2).normal(0.0, 1.0, 4000)
        mu, sigma = fit(data, "normal", "robust").params
        assert mu == pytest.approx(float(np.median(data)))
        expected = MAD_CONSISTENCY * float(np.median(np.abs(data - np.median(data))))
        assert sigma == pytest.approx(expected)

    def test_robust_ignores_contamination(self):
        clean = np.random.default_rng(3).normal(0.0, 1.0, 500)
        dirty = np.concatenate([clean, np.full(25, 60.0)])
        mle_sigma = fit(dirty, "normal", "mle").params[1]
        robust_sigma = fit(dirty, "normal", "robust").params[1]
        assert robust_sigma == pytest.approx(1.0, abs=0.15)
        assert mle_sigma > 5 * robust_sigma

    def test_exponential_moments_inverts_the_mean(self):
        data = np.random.default_rng(4).exponential(5.0, 3000)
        assert fit(data, "exponential", "moments").params[0] == pytest.approx(
            1.0 / float(np.mean(data))
        )

    def test_exponential_robust_uses_the_median(self):
        data = np.random.default_rng(5).exponential(5.0, 3000)
        expected = np.log(2.0) / float(np.median(data))
        assert fit(data, "exponential", "robust").params[0] == pytest.approx(expected)

    def test_gamma_moments_match_closed_form(self):
        data = np.random.default_rng(6).gamma(4.0, 2.0, 4000)
        alpha, beta = fit(data, "gamma", "moments").params
        m, v = float(np.mean(data)), float(np.var(data, ddof=1))
        assert alpha == pytest.approx(m * m / v)
        assert beta == pytest.approx(m / v)

    def test_uniform_moments_span_the_right_width(self):
        data = np.random.default_rng(7).uniform(0.0, 10.0, 6000)
        low, high = fit(data, "uniform", "moments").params
        assert high - low == pytest.approx(10.0, abs=0.6)


class TestSupportValidation:
    def test_positive_support_rejects_negatives(self):
        with pytest.raises(FitError, match="support x > 0"):
            fit(np.array([-1.0, 2.0, 3.0]), "exponential", "mle")

    def test_unit_support_rejects_out_of_range(self):
        with pytest.raises(FitError, match="0 < x < 1"):
            fit(np.array([0.2, 0.5, 4.0]), "beta", "mle")

    def test_real_support_accepts_negatives(self):
        assert fit(np.array([-3.0, 0.0, 2.0, 5.0]), "normal", "mle").params[1] > 0


class TestErrorHandling:
    def test_rejects_tiny_sample(self):
        with pytest.raises(FitError, match="at least three"):
            fit(np.array([1.0, 2.0]), "normal", "mle")

    def test_rejects_unknown_distribution(self):
        with pytest.raises(FitError, match="unknown distribution"):
            fit(SAMPLES["normal"], "cauchy", "mle")

    def test_rejects_unknown_method(self):
        with pytest.raises(FitError, match="unknown fit method"):
            fit(SAMPLES["normal"], "normal", "eyeball")

    def test_manual_requires_the_right_count(self):
        with pytest.raises(FitError, match="needs 2 parameter"):
            fit(SAMPLES["normal"], "normal", "manual", manual_params=(1.0,))

    def test_manual_rejects_invalid_parameters(self):
        with pytest.raises(FitError, match="rejected the estimated parameters"):
            fit(SAMPLES["normal"], "normal", "manual", manual_params=(0.0, -1.0))

    def test_manual_bypasses_support_validation(self):
        """A specified reference is the user's claim, not an estimate."""
        result = fit(np.array([-1.0, 1.0, 2.0]), "exponential", "manual", (0.5,))
        assert result.method == "manual"

    def test_non_finite_values_are_dropped(self):
        data = np.concatenate([SAMPLES["normal"], [np.nan, np.inf, -np.inf]])
        assert np.isfinite(fit(data, "normal", "mle").params).all()


class TestPresentation:
    def test_summary_names_the_method(self):
        assert "by mle" in fit(SAMPLES["normal"], "normal", "mle").summary()

    def test_param_map_is_keyed_by_name(self):
        mapping = fit(SAMPLES["normal"], "normal", "mle").param_map()
        assert set(mapping) == {"mu", "sigma"}

    def test_describe_params_formats_pairs(self):
        spec = DISTRIBUTIONS["normal"]
        assert spec.describe_params((1.0, 2.0)) == "mu=1, sigma=2"


class TestShapeHint:
    def test_symmetric_hint(self):
        data = np.random.default_rng(8).normal(0.0, 1.0, 500)
        assert "symmetric" in sample_skewness_hint(data)

    def test_right_skewed_positive_hint(self):
        data = np.random.default_rng(9).exponential(2.0, 500)
        assert "Gamma" in sample_skewness_hint(data)

    def test_left_skewed_hint(self):
        data = -np.random.default_rng(10).exponential(2.0, 500)
        assert "reflect" in sample_skewness_hint(data)

    def test_tiny_sample_declines_to_guess(self):
        assert "Too few" in sample_skewness_hint(np.array([1.0, 2.0, 3.0]))
