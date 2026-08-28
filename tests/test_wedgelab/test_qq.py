"""Tests for the Q-Q engine, its envelopes, and its diagnostics."""

import numpy as np
import pytest

from wedgelab.formula import FormulaError, Formula
from wedgelab.knowledge import KNOWLEDGE, symmetric_plotting_position
from wedgelab.models import FitError
from wedgelab.qq import (
    ENVELOPE_METHODS,
    LINE_METHODS,
    QQSpec,
    check_positions,
    compute,
    evaluate_positions,
    resolve_envelope,
)


def normal_sample(n: int = 120, seed: int = 0, mu: float = 10.0, sigma: float = 2.0):
    return np.random.default_rng(seed).normal(mu, sigma, n)


class TestSpecValidation:
    def test_rejects_unknown_line(self):
        with pytest.raises(ValueError, match="line must be"):
            QQSpec(data=normal_sample(), line="wiggly")

    def test_rejects_unknown_envelope(self):
        with pytest.raises(ValueError, match="envelope must be"):
            QQSpec(data=normal_sample(), envelope="vibes")

    def test_rejects_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            QQSpec(data=normal_sample(), alpha=1.5)

    def test_rejects_too_few_bootstrap_reps(self):
        with pytest.raises(ValueError, match="bootstrap_reps"):
            QQSpec(data=normal_sample(), bootstrap_reps=5)

    def test_replace_returns_a_copy(self):
        spec = QQSpec(data=normal_sample())
        other = spec.replace(alpha=0.10)
        assert spec.alpha == 0.05 and other.alpha == 0.10


class TestComputeBasics:
    def setup_method(self):
        self.result = compute(QQSpec(data=normal_sample(), label="test"))

    def test_arrays_are_aligned(self):
        r = self.result
        assert r.theoretical.shape == r.sample.shape == r.probabilities.shape

    def test_sample_is_sorted(self):
        assert np.all(np.diff(self.result.sample) >= 0)

    def test_theoretical_is_sorted(self):
        assert np.all(np.diff(self.result.theoretical) > 0)

    def test_probabilities_are_inside_unit_interval(self):
        p = self.result.probabilities
        assert np.all(p > 0) and np.all(p < 1)

    def test_ppcc_is_high_for_a_correct_model(self):
        assert self.result.diagnostics.ppcc > 0.99

    def test_slope_is_near_one(self):
        assert self.result.diagnostics.slope == pytest.approx(1.0, abs=0.1)

    def test_caption_names_every_choice(self):
        caption = self.result.caption()
        for fragment in ("Normal", "Plotting positions", "envelope", "r ="):
            assert fragment in caption

    def test_diagnostic_lines_are_strings(self):
        assert all(isinstance(line, str) for line in self.result.diagnostics.lines())


class TestErrorCases:
    def test_rejects_tiny_sample(self):
        with pytest.raises(ValueError, match="at least three"):
            compute(QQSpec(data=np.array([1.0, 2.0])))

    def test_rejects_constant_sample(self):
        with pytest.raises(ValueError, match="constant"):
            compute(QQSpec(data=np.full(20, 3.0)))

    def test_drops_non_finite_and_reports_it(self):
        data = np.concatenate([normal_sample(40), [np.nan, np.inf]])
        result = compute(QQSpec(data=data))
        assert result.n == 40
        assert result.diagnostics.n_dropped == 2
        assert any("non-finite" in w for w in result.warnings)

    def test_rejects_positions_outside_unit_interval(self):
        bad = Formula(name="bad", expression="i / n", lhs="p_i")
        with pytest.raises(FormulaError, match="strictly inside"):
            compute(QQSpec(data=normal_sample(20), position=bad))

    def test_rejects_decreasing_positions(self):
        bad = Formula(name="bad", expression="(n - i + 0.5) / n", lhs="p_i")
        with pytest.raises(FormulaError, match="non-decreasing"):
            compute(QQSpec(data=normal_sample(20), position=bad))

    def test_rejects_negative_data_for_positive_support(self):
        with pytest.raises(FitError, match="support"):
            compute(QQSpec(data=normal_sample(30, mu=0.0), dist_key="exponential"))


class TestEnvelopes:
    @pytest.mark.parametrize("envelope", ENVELOPE_METHODS)
    def test_every_envelope_computes(self, envelope):
        result = compute(
            QQSpec(data=normal_sample(60), envelope=envelope, bootstrap_reps=60)
        )
        if envelope == "none":
            assert result.lower is None
        else:
            assert result.lower is not None
            assert np.all(result.lower <= result.upper)

    def test_beta_band_brackets_the_median(self):
        """The exact band must contain the model's own median quantile."""
        result = compute(QQSpec(data=normal_sample(101), envelope="beta"))
        mid = 50
        centre = result.theoretical[mid]
        assert result.lower[mid] < centre < result.upper[mid]

    def test_beta_band_is_narrowest_in_the_middle(self):
        result = compute(QQSpec(data=normal_sample(101), envelope="beta"))
        width = result.upper - result.lower
        assert width[50] < width[5]
        assert width[50] < width[-5]

    def test_band_widens_as_alpha_shrinks(self):
        wide = compute(QQSpec(data=normal_sample(80), envelope="beta", alpha=0.01))
        narrow = compute(QQSpec(data=normal_sample(80), envelope="beta", alpha=0.20))
        assert np.mean(wide.upper - wide.lower) > np.mean(narrow.upper - narrow.lower)

    def test_simultaneous_band_is_wider_than_pointwise(self):
        point = compute(QQSpec(data=normal_sample(80), envelope="beta"))
        simul = compute(QQSpec(data=normal_sample(80), envelope="simultaneous"))
        finite = np.isfinite(simul.lower) & np.isfinite(simul.upper)
        assert np.mean((simul.upper - simul.lower)[finite]) > np.mean(
            (point.upper - point.lower)[finite]
        )

    def test_simultaneous_band_may_be_unbounded(self):
        result = compute(QQSpec(data=normal_sample(40), envelope="simultaneous"))
        assert not np.all(np.isfinite(result.lower)) or not np.all(
            np.isfinite(result.upper)
        )

    def test_bootstrap_is_reproducible(self):
        spec = QQSpec(
            data=normal_sample(50), envelope="bootstrap", bootstrap_reps=80, random_state=7
        )
        assert np.allclose(compute(spec).lower, compute(spec).lower)

    def test_bootstrap_differs_with_seed(self):
        base = QQSpec(data=normal_sample(50), envelope="bootstrap", bootstrap_reps=80)
        assert not np.allclose(
            compute(base.replace(random_state=1)).lower,
            compute(base.replace(random_state=2)).lower,
        )

    def test_pointwise_envelopes_agree_for_a_specified_reference(self):
        """With nothing estimated, all three answer the same question."""
        totals = {}
        for envelope in ("beta", "asymptotic", "bootstrap"):
            total = 0
            for seed in range(5):
                data = np.random.default_rng(seed).normal(0.0, 1.0, 150)
                total += compute(
                    QQSpec(
                        data=data,
                        envelope=envelope,
                        fit_method="manual",
                        manual_params=(0.0, 1.0),
                        bootstrap_reps=200,
                        random_state=1,
                    )
                ).diagnostics.outside_band
            totals[envelope] = total
        assert max(totals.values()) - min(totals.values()) <= 20, totals

    def test_beta_underflags_against_bootstrap_when_parameters_are_estimated(self):
        """The bands diverge once the reference is fitted -- and must.

        A Beta band built on estimated parameters is fitted to the data it is
        judging, so it flags far less than the calibrated bootstrap band.  An
        earlier version of this suite asserted the three bands agreed here;
        they only appeared to because the bootstrap was discarding its refit.
        """
        beta_total = boot_total = 0
        for seed in range(6):
            data = np.random.default_rng(seed).normal(10.0, 2.0, 150)
            beta_total += compute(
                QQSpec(data=data, envelope="beta", fit_method="mle")
            ).diagnostics.outside_band
            boot_total += compute(
                QQSpec(
                    data=data,
                    envelope="bootstrap",
                    fit_method="mle",
                    bootstrap_reps=300,
                    random_state=1,
                )
            ).diagnostics.outside_band
        assert boot_total > beta_total, (beta_total, boot_total)

    def test_bootstrap_restores_calibration_under_estimation(self):
        """The whole justification for the 'auto' default, asserted directly.

        Measured over 50 replicates at n=100: Beta+MLE averages 0.24
        excursions where alpha*n is 5.0; the pivotal bootstrap averages 5.38.
        The bounds here are wide enough to survive Monte Carlo noise at the
        smaller replicate count a test can afford.
        """
        rng = np.random.default_rng(31)
        beta_counts, boot_counts = [], []
        for _ in range(16):
            data = rng.normal(0.0, 1.0, 100)
            beta_counts.append(
                compute(
                    QQSpec(data=data, envelope="beta", fit_method="mle")
                ).diagnostics.outside_band
            )
            boot_counts.append(
                compute(
                    QQSpec(
                        data=data,
                        envelope="bootstrap",
                        fit_method="mle",
                        bootstrap_reps=300,
                    )
                ).diagnostics.outside_band
            )
        assert np.mean(beta_counts) < 1.5, np.mean(beta_counts)
        assert np.mean(boot_counts) > 2.0, np.mean(boot_counts)

    def test_exact_band_coverage_is_close_to_nominal(self):
        """Under a fully specified model, excursions should track alpha*n."""
        rng = np.random.default_rng(11)
        excursions = []
        for _ in range(40):
            data = rng.normal(0.0, 1.0, 100)
            result = compute(
                QQSpec(
                    data=data,
                    envelope="beta",
                    fit_method="manual",
                    manual_params=(0.0, 1.0),
                )
            )
            excursions.append(result.diagnostics.outside_band)
        # 5% of 100 points, pointwise; order statistics are strongly
        # correlated, so the mean count is well below 5 but must not be zero.
        assert 0.2 < np.mean(excursions) < 9.0


class TestReferenceLines:
    @pytest.mark.parametrize("line", LINE_METHODS)
    def test_every_line_computes(self, line):
        result = compute(QQSpec(data=normal_sample(60), line=line))
        assert np.isfinite(result.slope) and np.isfinite(result.intercept)

    def test_theoretical_line_is_the_identity(self):
        result = compute(QQSpec(data=normal_sample(60), line="theoretical"))
        assert (result.slope, result.intercept) == (1.0, 0.0)

    def test_ols_line_matches_least_squares(self):
        result = compute(QQSpec(data=normal_sample(60), line="ols"))
        coefficients = np.polyfit(result.theoretical, result.sample, 1)
        assert result.slope == pytest.approx(coefficients[0], rel=1e-9)
        assert result.intercept == pytest.approx(coefficients[1], rel=1e-9)

    def test_quartile_line_passes_through_the_quartiles(self):
        result = compute(QQSpec(data=normal_sample(101), line="quartile"))
        tq1, tq3 = np.percentile(result.theoretical, [25, 75])
        sq1, sq3 = np.percentile(result.sample, [25, 75])
        assert result.intercept + result.slope * tq1 == pytest.approx(sq1)
        assert result.intercept + result.slope * tq3 == pytest.approx(sq3)

    def test_quartile_line_resists_outliers(self):
        """With the reference held fixed, the quartile line moves less.

        The fit is pinned to the true parameters so that only the line rule
        varies; otherwise this would be measuring the estimator's robustness
        rather than the line's.
        """
        dirty = normal_sample(200, seed=2)
        dirty[:5] = 400.0
        pinned = QQSpec(data=dirty, fit_method="manual", manual_params=(10.0, 2.0))
        ols = compute(pinned.replace(line="ols")).slope
        quartile = compute(pinned.replace(line="quartile")).slope
        assert abs(quartile - 1.0) < abs(ols - 1.0)

    def test_line_points_span_the_axis(self):
        result = compute(QQSpec(data=normal_sample(40)))
        x, _ = result.line_points()
        assert x[0] == result.theoretical.min()
        assert x[1] == result.theoretical.max()


class TestTransforms:
    def test_standardize_rescales_both_axes(self):
        raw = compute(QQSpec(data=normal_sample(80, mu=100.0, sigma=15.0)))
        std = compute(
            QQSpec(data=normal_sample(80, mu=100.0, sigma=15.0), standardize=True)
        )
        assert abs(np.mean(std.sample)) < 1.0
        assert np.ptp(std.sample) < np.ptp(raw.sample)

    def test_standardize_preserves_ppcc(self):
        data = normal_sample(80, mu=100.0, sigma=15.0)
        raw = compute(QQSpec(data=data))
        std = compute(QQSpec(data=data, standardize=True))
        assert std.diagnostics.ppcc == pytest.approx(raw.diagnostics.ppcc, abs=1e-10)

    def test_detrend_centres_the_residuals(self):
        result = compute(QQSpec(data=normal_sample(80), detrend=True, line="ols"))
        assert abs(np.mean(result.sample)) < 1e-9

    def test_detrend_flattens_the_drawn_line(self):
        result = compute(QQSpec(data=normal_sample(80), detrend=True))
        assert (result.slope, result.intercept) == (0.0, 0.0)

    def test_detrend_keeps_the_real_slope_in_diagnostics(self):
        """The drawn line flattens; the reported one must not."""
        data = normal_sample(80, mu=5.0, sigma=3.0)
        plain = compute(QQSpec(data=data))
        detrended = compute(QQSpec(data=data, detrend=True))
        assert detrended.diagnostics.slope == pytest.approx(plain.diagnostics.slope)

    def test_detrend_preserves_ppcc(self):
        """PPCC describes the Q-Q relationship, not the residual view."""
        data = normal_sample(80)
        plain = compute(QQSpec(data=data))
        detrended = compute(QQSpec(data=data, detrend=True))
        assert detrended.diagnostics.ppcc == pytest.approx(
            plain.diagnostics.ppcc, abs=1e-12
        )
        assert detrended.diagnostics.ppcc > 0.9

    def test_detrend_keeps_outlier_flags(self):
        data = normal_sample(100, seed=6)
        data[0] = -50.0
        plain = compute(QQSpec(data=data, envelope="beta"))
        detrended = compute(QQSpec(data=data, envelope="beta", detrend=True))
        assert plain.diagnostics.outside_band == detrended.diagnostics.outside_band


class TestPlottingPositionSensitivity:
    def test_ppcc_is_invariant_to_the_fitted_location_and_scale(self):
        """Pearson r cannot referee the estimator; the band has to."""
        data = normal_sample(100, seed=8)
        mle = compute(QQSpec(data=data, fit_method="mle"))
        robust = compute(QQSpec(data=data, fit_method="robust"))
        assert mle.diagnostics.ppcc == pytest.approx(robust.diagnostics.ppcc, abs=1e-10)

    def test_exact_median_rank_runs(self):
        result = compute(
            QQSpec(
                data=normal_sample(60),
                position=KNOWLEDGE.get("pp_median_exact").formula,
            )
        )
        assert result.diagnostics.ppcc > 0.98

    def test_positions_summary_reports_bindings(self):
        spec = QQSpec(
            data=normal_sample(30),
            position=symmetric_plotting_position(0.375),
            position_bindings={"a": 0.44},
        )
        assert "a=0.44" in spec.positions_summary()


class TestAutoEnvelope:
    """'auto' must pick the band that is calibrated for the estimator in use."""

    def test_resolver_picks_beta_for_a_specified_reference(self):
        assert resolve_envelope("auto", "manual") == "beta"

    @pytest.mark.parametrize("method", ["mle", "moments", "robust"])
    def test_resolver_picks_bootstrap_for_an_estimated_reference(self, method):
        assert resolve_envelope("auto", method) == "bootstrap"

    @pytest.mark.parametrize("envelope", ["none", "beta", "asymptotic", "simultaneous"])
    def test_resolver_passes_through_explicit_choices(self, envelope):
        assert resolve_envelope(envelope, "mle") == envelope

    def test_auto_is_the_default(self):
        assert QQSpec(data=normal_sample(30)).envelope == "auto"

    def test_result_records_the_resolved_envelope(self):
        estimated = compute(QQSpec(data=normal_sample(50), bootstrap_reps=60))
        assert estimated.spec.envelope == "auto"
        assert estimated.envelope == "bootstrap"

        specified = compute(
            QQSpec(data=normal_sample(50), fit_method="manual", manual_params=(10.0, 2.0))
        )
        assert specified.envelope == "beta"

    def test_auto_explains_its_choice(self):
        result = compute(QQSpec(data=normal_sample(50), bootstrap_reps=60))
        assert any("resolved to 'bootstrap'" in w for w in result.warnings)

    def test_caption_names_the_band_actually_drawn(self):
        result = compute(QQSpec(data=normal_sample(50), bootstrap_reps=60))
        assert "bootstrap" in result.caption()
        assert "auto" not in result.caption()

    def test_beta_under_estimation_warns_with_numbers(self):
        """The 'conservative' caveat is a 15x effect and must say so."""
        result = compute(QQSpec(data=normal_sample(50), envelope="beta", fit_method="mle"))
        warning = next(w for w in result.warnings if "fully specified reference" in w)
        assert "0.4" in warning and "bootstrap" in warning

    def test_auto_selects_the_calibrated_band(self):
        """Calibration itself is asserted in TestEnvelopes; here just the wiring."""
        result = compute(QQSpec(data=normal_sample(80), fit_method="mle", bootstrap_reps=200))
        assert result.envelope == "bootstrap"
        assert result.diagnostics.expected_outside == pytest.approx(0.05 * 80)


class TestPositionSensitivityIsConstantInN:
    """Locks in a measured fact that an earlier narrative got backwards."""

    @staticmethod
    def _fan_ratio(n: int, seed: int) -> float:
        family = symmetric_plotting_position(0.375)
        data = np.random.default_rng(seed).normal(0.0, 1.0, n)
        base = QQSpec(
            data=data,
            position=family,
            envelope="beta",
            fit_method="manual",
            manual_params=(0.0, 1.0),
        )
        curves = [
            compute(base.replace(position_bindings={"a": a})).theoretical
            for a in (0.0, 0.25, 0.5)
        ]
        stack = np.vstack(curves)
        band = compute(base.replace(position_bindings={"a": 0.375}))
        return float(np.max((stack.max(0) - stack.min(0)) / (band.upper - band.lower)))

    def test_ratio_does_not_shrink_with_n(self):
        small = np.mean([self._fan_ratio(15, s) for s in range(4)])
        large = np.mean([self._fan_ratio(200, s) for s in range(4)])
        # Measured at 13-15% across n = 10..300; assert it stays in that band
        # rather than decaying, which is what the corrected narrative claims.
        assert 0.08 < small < 0.20
        assert 0.08 < large < 0.20
        assert large > 0.5 * small

    def test_ppcc_spread_does_shrink_with_n(self):
        """The other metric genuinely does shrink -- both facts are true."""
        family = symmetric_plotting_position(0.375)

        def spread(n: int) -> float:
            data = normal_sample(n, seed=3)
            values = [
                compute(
                    QQSpec(data=data, position=family, position_bindings={"a": a}, envelope="none")
                ).diagnostics.ppcc
                for a in (0.0, 0.5)
            ]
            return abs(values[0] - values[1])

        assert spread(15) > spread(400)


class TestPositionValidation:
    """The workbench and the engine must enforce the same contract."""

    def test_check_accepts_a_valid_rule(self):
        ok, message = check_positions(symmetric_plotting_position(0.375), 50)
        assert ok and message == ""

    def test_check_rejects_a_boundary_reaching_rule(self):
        ok, message = check_positions(Formula(name="f", expression="i / n"), 50)
        assert not ok
        assert "strictly inside" in message

    def test_check_rejects_a_decreasing_rule(self):
        ok, message = check_positions(
            Formula(name="f", expression="(n - i + 0.5) / n"), 50
        )
        assert not ok
        assert "non-decreasing" in message

    def test_check_rejects_an_unbound_symbol(self):
        ok, message = check_positions(Formula(name="f", expression="(i - b) / n"), 50)
        assert not ok
        assert "unbound symbol" in message

    def test_check_honours_bindings(self):
        family = symmetric_plotting_position(0.375)
        assert check_positions(family, 50, {"a": 0.5})[0]

    def test_evaluate_returns_the_right_length(self):
        p = evaluate_positions(symmetric_plotting_position(0.375), 37)
        assert p.shape == (37,)

    def test_check_and_compute_agree(self):
        """Anything check_positions accepts, compute must also accept."""
        for expression in (
            "(i - a) / (n + 1 - 2*a)",
            "beta_ppf(0.5, i, n - i + 1)",
            "(i - 0.44) / (n + 0.12)",
            "where(i == 1, 1 - 0.5**(1/n), (i - 0.3175) / (n + 0.365))",
        ):
            formula = Formula(name="f", expression=expression, lhs="p_i")
            bindings = {"a": 0.375} if "a" in formula.symbols else {}
            ok, _ = check_positions(formula, 40, bindings)
            assert ok, expression
            compute(
                QQSpec(
                    data=normal_sample(40),
                    position=formula,
                    position_bindings=bindings,
                )
            )


class TestDiagnosticsContent:
    def test_shapiro_only_for_normal(self):
        normal = compute(QQSpec(data=normal_sample(60)))
        exponential = compute(
            QQSpec(
                data=np.random.default_rng(1).exponential(2.0, 60), dist_key="exponential"
            )
        )
        assert normal.diagnostics.shapiro_w is not None
        assert exponential.diagnostics.shapiro_w is None

    def test_ks_p_value_is_flagged_as_invalid_when_fitted(self):
        result = compute(QQSpec(data=normal_sample(60), fit_method="mle"))
        assert any("Kolmogorov-Smirnov p-value is not valid" in w for w in result.warnings)

    def test_ks_p_value_is_not_flagged_when_specified(self):
        result = compute(
            QQSpec(data=normal_sample(60), fit_method="manual", manual_params=(10.0, 2.0))
        )
        assert not any("p-value is not valid" in w for w in result.warnings)

    def test_heavy_tails_lower_the_ppcc(self):
        rng = np.random.default_rng(5)
        light = compute(QQSpec(data=rng.normal(0, 1, 200)))
        heavy = compute(QQSpec(data=rng.standard_t(2, 200)))
        assert heavy.diagnostics.ppcc < light.diagnostics.ppcc

    def test_expected_outside_tracks_alpha_when_calibrated(self):
        result = compute(
            QQSpec(
                data=normal_sample(200),
                envelope="beta",
                alpha=0.10,
                fit_method="manual",
                manual_params=(10.0, 2.0),
            )
        )
        assert result.diagnostics.expected_outside == pytest.approx(20.0)

    def test_expected_outside_is_withheld_when_the_band_is_not_calibrated(self):
        """alpha*n is wrong for a Beta band fitted to its own data."""
        result = compute(
            QQSpec(data=normal_sample(200), envelope="beta", alpha=0.10, fit_method="mle")
        )
        assert result.diagnostics.expected_outside == 0.0
        assert any("not calibrated" in w or "calibrated band" in w for w in result.warnings)

    def test_bootstrap_reports_an_expected_count(self):
        result = compute(
            QQSpec(data=normal_sample(60), envelope="bootstrap", bootstrap_reps=60, alpha=0.10)
        )
        assert result.diagnostics.expected_outside == pytest.approx(6.0)
