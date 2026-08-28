"""Tests for the P-P, ECDF and two-sample diagnostics."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

import wedgelab as wl
from wedgelab.diagnostics import (
    ECDF_ENVELOPES,
    PP_ENVELOPES,
    TWO_SAMPLE_ENVELOPES,
    ECDFSpec,
    PPSpec,
    TwoSampleSpec,
    compute_ecdf,
    compute_pp,
    compute_two_sample,
)
from wedgelab.models import FitError
from wedgelab.plot import PlotOptions, render_diagnostic
from wedgelab.qq import QQSpec, compute


def sample(n=120, seed=0, mu=10.0, sigma=2.0):
    return np.random.default_rng(seed).normal(mu, sigma, n)


# ===========================================================================
# P-P plot
# ===========================================================================


class TestPPBasics:
    def setup_method(self):
        self.result = compute_pp(PPSpec(data=sample(), envelope="beta", label="test"))

    def test_axes_are_probabilities(self):
        for axis in (self.result.theoretical, self.result.empirical):
            assert np.all(axis >= 0.0) and np.all(axis <= 1.0)

    def test_both_axes_ascend(self):
        assert np.all(np.diff(self.result.theoretical) > 0)
        assert np.all(np.diff(self.result.empirical) >= 0)

    def test_correct_model_tracks_the_identity(self):
        assert self.result.max_deviation < 0.15
        assert self.result.correlation > 0.99

    def test_identity_line_is_the_default(self):
        assert (self.result.slope, self.result.intercept) == (1.0, 0.0)

    def test_ols_line_is_available(self):
        r = compute_pp(PPSpec(data=sample(), line="ols"))
        assert r.slope != 1.0 or r.intercept != 0.0

    def test_caption_names_the_band(self):
        assert "Beta" in self.result.caption()

    def test_lines_are_strings(self):
        assert all(isinstance(line, str) for line in self.result.lines())


class TestPPMirrorsQQ:
    """The two figures are one test seen from two angles."""

    @pytest.mark.parametrize("key", ["normal", "heavy", "bimodal", "lognormal"])
    def test_beta_band_flags_identical_points(self, key):
        """F is monotone, so the Q-Q band is the F-inverse image of the P-P band.

        This is an identity, not a coincidence: a point is outside one exactly
        when it is outside the other. If this test ever fails, one of the two
        envelopes has stopped being a transformation of the other.
        """
        data = wl.generate(key, 120, seed=3)
        shared = dict(fit_method="manual", manual_params=(50.0, 10.0), envelope="beta")
        pp = compute_pp(PPSpec(data=data, **shared))
        qq = compute(QQSpec(data=data, **shared))
        assert np.array_equal(pp.outside, qq.outside)

    def test_pp_is_blind_where_qq_is_sharp(self):
        """Heavy tails wreck the Q-Q correlation and barely dent the P-P one."""
        data = wl.generate("heavy", 200, seed=3)
        shared = dict(fit_method="robust", envelope="beta")
        pp = compute_pp(PPSpec(data=data, **shared))
        qq = compute(QQSpec(data=data, **shared))
        assert qq.diagnostics.ppcc < 0.95
        assert pp.correlation > 0.99


class TestPPEnvelopes:
    @pytest.mark.parametrize("envelope", PP_ENVELOPES)
    def test_every_envelope_computes(self, envelope):
        r = compute_pp(PPSpec(data=sample(60), envelope=envelope, bootstrap_reps=60))
        if envelope == "none":
            assert r.lower is None
        else:
            assert r.lower is not None
            assert np.all(r.lower <= r.upper)
            assert np.all(r.lower >= 0.0) and np.all(r.upper <= 1.0)

    def test_auto_resolves_like_the_qq_plot(self):
        estimated = compute_pp(PPSpec(data=sample(50), bootstrap_reps=60))
        assert estimated.envelope == "bootstrap"
        specified = compute_pp(
            PPSpec(data=sample(50), fit_method="manual", manual_params=(10.0, 2.0))
        )
        assert specified.envelope == "beta"

    def test_bootstrap_is_reproducible(self):
        spec = PPSpec(data=sample(50), envelope="bootstrap", bootstrap_reps=80, random_state=4)
        assert np.allclose(compute_pp(spec).lower, compute_pp(spec).lower)


class TestPPValidation:
    def test_rejects_unknown_envelope(self):
        with pytest.raises(ValueError, match="envelope must be"):
            PPSpec(data=sample(20), envelope="vibes")

    def test_rejects_unknown_line(self):
        with pytest.raises(ValueError, match="line must be"):
            PPSpec(data=sample(20), line="wiggly")

    def test_rejects_tiny_sample(self):
        with pytest.raises(ValueError, match="at least three"):
            compute_pp(PPSpec(data=np.array([1.0, 2.0])))

    def test_rejects_constant_sample(self):
        with pytest.raises(ValueError, match="constant"):
            compute_pp(PPSpec(data=np.full(20, 4.0)))

    def test_drops_non_finite(self):
        data = np.concatenate([sample(40), [np.nan, np.inf]])
        r = compute_pp(PPSpec(data=data))
        assert r.n == 40 and r.n_dropped == 2


# ===========================================================================
# ECDF
# ===========================================================================


class TestECDF:
    def setup_method(self):
        self.result = compute_ecdf(ECDFSpec(data=sample(), label="test"))

    def test_ecdf_ascends_to_one(self):
        assert self.result.ecdf[0] > 0
        assert self.result.ecdf[-1] == pytest.approx(1.0)
        assert np.all(np.diff(self.result.ecdf) > 0)

    def test_model_curve_is_a_cdf(self):
        m = self.result.model_cdf
        assert np.all(np.diff(m) >= -1e-12)
        assert np.all((m >= 0.0) & (m <= 1.0))

    def test_ks_matches_scipy(self):
        from scipy import stats as sp_stats

        data = sample(80, seed=2)
        r = compute_ecdf(ECDFSpec(data=data, fit_method="mle"))
        mu, sigma = r.fit.params
        reference = sp_stats.kstest(np.sort(data), sp_stats.norm(mu, sigma).cdf).statistic
        assert r.ks_statistic == pytest.approx(float(reference), abs=1e-9)

    def test_correct_model_stays_inside_the_band(self):
        assert self.result.band_contains_model
        assert self.result.ks_excursion == 0.0

    def test_a_badly_wrong_model_leaves_the_band(self):
        """A normal reference against strongly skewed data at a usable n."""
        r = compute_ecdf(
            ECDFSpec(data=wl.generate("lognormal", 400, seed=1), fit_method="mle")
        )
        assert not r.band_contains_model
        assert r.ks_excursion > 0.0

    @pytest.mark.parametrize("envelope", ECDF_ENVELOPES)
    def test_every_envelope_computes(self, envelope):
        r = compute_ecdf(ECDFSpec(data=sample(60), envelope=envelope))
        if envelope == "none":
            assert r.lower is None
        else:
            assert np.all(r.lower <= r.upper)

    def test_simultaneous_band_is_wider_than_pointwise(self):
        wide = compute_ecdf(ECDFSpec(data=sample(80), envelope="simultaneous"))
        narrow = compute_ecdf(ECDFSpec(data=sample(80), envelope="pointwise"))
        assert np.mean(wide.upper - wide.lower) > np.mean(narrow.upper - narrow.lower)

    def test_warns_that_the_band_is_loose_under_estimation(self):
        r = compute_ecdf(ECDFSpec(data=sample(60), fit_method="mle"))
        assert any("Lilliefors" in w for w in r.warnings)

    def test_rejects_unknown_envelope(self):
        with pytest.raises(ValueError, match="envelope must be"):
            ECDFSpec(data=sample(20), envelope="vibes")

    def test_caption_states_the_verdict(self):
        assert "band" in self.result.caption()


# ===========================================================================
# Two-sample
# ===========================================================================


class TestTwoSample:
    def test_recovers_a_scale_ratio(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(10, 2, 400), rng.normal(10, 3.0, 400)
        r = compute_two_sample(TwoSampleSpec(first=a, second=b, envelope="none"))
        assert r.slope == pytest.approx(1.5, rel=0.15)

    def test_recovers_a_location_shift(self):
        rng = np.random.default_rng(6)
        a, b = rng.normal(10, 2, 400), rng.normal(13, 2, 400)
        r = compute_two_sample(TwoSampleSpec(first=a, second=b, envelope="none"))
        assert r.slope == pytest.approx(1.0, abs=0.2)
        assert r.intercept == pytest.approx(3.0, abs=0.8)

    def test_identity_line_is_exactly_the_identity(self):
        rng = np.random.default_rng(7)
        r = compute_two_sample(
            TwoSampleSpec(first=rng.normal(0, 1, 60), second=rng.normal(0, 1, 60),
                          line="identity", envelope="none")
        )
        assert (r.slope, r.intercept) == (1.0, 0.0)

    def test_unequal_sizes_use_the_smaller(self):
        rng = np.random.default_rng(8)
        r = compute_two_sample(
            TwoSampleSpec(first=rng.normal(0, 1, 40), second=rng.normal(0, 1, 90),
                          envelope="none")
        )
        assert (r.n_first, r.n_second, r.n_common) == (40, 90, 40)

    def test_ks_matches_scipy(self):
        from scipy import stats as sp_stats

        rng = np.random.default_rng(9)
        a, b = rng.normal(0, 1, 70), rng.normal(0.6, 1, 90)
        r = compute_two_sample(TwoSampleSpec(first=a, second=b, envelope="none"))
        reference = sp_stats.ks_2samp(a, b)
        assert r.ks_statistic == pytest.approx(float(reference.statistic))
        assert r.ks_p_value == pytest.approx(float(reference.pvalue))

    def test_the_exact_band_is_refused_with_a_reason(self):
        with pytest.raises(ValueError, match="both axes are random"):
            TwoSampleSpec(first=sample(20), second=sample(20), envelope="beta")

    def test_rejects_tiny_samples(self):
        with pytest.raises(ValueError, match="at least three"):
            compute_two_sample(TwoSampleSpec(first=[1.0, 2.0], second=sample(20)))

    def test_rejects_a_constant_sample(self):
        with pytest.raises(ValueError, match="constant"):
            compute_two_sample(TwoSampleSpec(first=np.full(20, 3.0), second=sample(20)))

    def test_bootstrap_is_reproducible(self):
        rng = np.random.default_rng(10)
        spec = TwoSampleSpec(first=rng.normal(0, 1, 50), second=rng.normal(0, 1, 60),
                             bootstrap_reps=120, random_state=3)
        assert np.allclose(compute_two_sample(spec).lower, compute_two_sample(spec).lower)

    @pytest.mark.parametrize("line", ["ols", "quartile", "identity"])
    def test_band_is_not_wildly_miscalibrated(self, line):
        """Matched samples must not light up the band.

        The band is built under a pooled null with the line rule refitted in
        every replicate. Measured over 15 seeds it runs conservative -- 1 to 3
        percent against a nominal 5 -- so this asserts a generous ceiling that
        still catches the construction being wrong. Centring replicates on the
        observed line instead of refitting them put this near 20 percent.
        """
        flagged = total = 0
        for seed in range(6):
            rng = np.random.default_rng(700 + seed)
            r = compute_two_sample(
                TwoSampleSpec(first=rng.normal(10, 2, 60), second=rng.normal(10, 2, 90),
                              line=line, bootstrap_reps=200, random_state=seed)
            )
            flagged += r.outside_band
            total += r.n_common
        assert flagged / total < 0.15, flagged / total

    def test_ols_line_absorbs_location_and_scale(self):
        """So excursions report shape, and slope/intercept report the rest."""
        rng = np.random.default_rng(11)
        base = rng.normal(10, 2, 200)
        shifted = rng.normal(14, 2, 200)
        r = compute_two_sample(
            TwoSampleSpec(first=base, second=shifted, line="ols", bootstrap_reps=200)
        )
        assert r.intercept > 2.0
        assert r.outside_band / r.n_common < 0.15

    @pytest.mark.parametrize("envelope", TWO_SAMPLE_ENVELOPES)
    def test_every_envelope_computes(self, envelope):
        rng = np.random.default_rng(12)
        r = compute_two_sample(
            TwoSampleSpec(first=rng.normal(0, 1, 50), second=rng.normal(0, 1, 50),
                          envelope=envelope, bootstrap_reps=100)
        )
        assert (r.lower is None) == (envelope == "none")

    def test_caption_explains_the_slope(self):
        rng = np.random.default_rng(13)
        r = compute_two_sample(
            TwoSampleSpec(first=rng.normal(0, 1, 50), second=rng.normal(0, 1, 50),
                          bootstrap_reps=100)
        )
        assert "ratio of scales" in r.caption()


# ===========================================================================
# Rendering
# ===========================================================================


class TestRendering:
    @pytest.fixture(scope="class")
    def results(self):
        data = wl.generate("heavy", 100, seed=1)
        rng = np.random.default_rng(0)
        return [
            compute(QQSpec(data=data, fit_method="robust", bootstrap_reps=80)),
            compute_pp(PPSpec(data=data, fit_method="robust", bootstrap_reps=80)),
            compute_ecdf(ECDFSpec(data=data, fit_method="robust")),
            compute_two_sample(
                TwoSampleSpec(first=rng.normal(0, 1, 60), second=rng.standard_t(3, 80),
                              bootstrap_reps=100)
            ),
        ]

    def test_every_result_renders(self, results):
        for result in results:
            assert isinstance(render_diagnostic(result, "screen"), Figure)

    @pytest.mark.parametrize("theme", ["nature", "science", "ieee", "apa"])
    def test_every_theme_renders_every_type(self, results, theme):
        for result in results:
            fig = render_diagnostic(result, theme)
            assert fig.get_size_inches()[0] == pytest.approx(
                wl.get_theme(theme).width_in, abs=0.01
            )

    def test_options_are_respected(self, results):
        options = PlotOptions(show_band=False, show_legend=False, annotate=False)
        for result in results:
            assert isinstance(render_diagnostic(result, "screen", options), Figure)

    def test_pp_axes_are_the_unit_square(self, results):
        ax = render_diagnostic(results[1], "screen").axes[0]
        assert ax.get_xlim() == pytest.approx((0.0, 1.0))
        assert ax.get_ylim() == pytest.approx((0.0, 1.0))

    def test_ecdf_labels_its_axes(self, results):
        ax = render_diagnostic(results[2], "screen").axes[0]
        assert "Cumulative" in ax.get_ylabel()

    def test_two_sample_names_both_samples(self, results):
        ax = render_diagnostic(results[3], "screen").axes[0]
        assert "Quantiles of" in ax.get_xlabel()
        assert "Quantiles of" in ax.get_ylabel()

    def test_unknown_result_is_refused(self):
        with pytest.raises(TypeError, match="cannot render"):
            render_diagnostic({"not": "a result"})
