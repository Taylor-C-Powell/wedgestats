"""Diagnostic figures other than the Q-Q plot.

Each one here exists because it answers a question the Q-Q plot answers badly
or not at all:

* :func:`compute_pp` -- the P-P plot is sensitive in the middle of a
  distribution and nearly blind in the tails, which is the exact complement of
  the Q-Q plot.  Disagreement between the two localises where a model fails.
* :func:`compute_ecdf` -- the empirical CDF with a simultaneous band is the
  figure that actually supports "the data are consistent with F", because the
  band bounds the whole curve rather than each point.
* :func:`compute_two_sample` -- comparing two samples to each other needs no
  reference distribution at all, and cannot use the exact Beta band, because
  both axes are random.

All three reuse the same five decisions as :mod:`wedgelab.qq`: plotting
positions from an editable formula, a fitted reference and its estimator, a
reference line, a confidence envelope, and a publication theme.  Where a
decision does not apply -- there is no fitted distribution in a two-sample
plot -- the field is absent rather than ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from wedgestats.descriptive import correlation, median, quartiles
from wedgestats.regression import simple_ols

from wedgelab.formula import Formula
from wedgelab.knowledge import symmetric_plotting_position
from wedgelab.models import FitError, FitResult, fit
from wedgelab.qq import (
    beta_band,
    evaluate_positions,
    ks_critical,
    resolve_envelope,
)

__all__ = [
    "PPSpec",
    "PPResult",
    "compute_pp",
    "ECDFSpec",
    "ECDFResult",
    "compute_ecdf",
    "TwoSampleSpec",
    "TwoSampleResult",
    "compute_two_sample",
    "PP_ENVELOPES",
    "ECDF_ENVELOPES",
    "TWO_SAMPLE_ENVELOPES",
]

_EPS = 1e-12

PP_ENVELOPES: tuple[str, ...] = ("auto", "none", "beta", "simultaneous", "bootstrap")
ECDF_ENVELOPES: tuple[str, ...] = ("none", "simultaneous", "pointwise")
TWO_SAMPLE_ENVELOPES: tuple[str, ...] = ("none", "bootstrap")


def _clean(data: Any) -> tuple[np.ndarray, int]:
    """Drop non-finite values and report how many went."""
    arr = np.asarray(data, dtype=float).ravel()
    finite = np.isfinite(arr)
    return arr[finite], int(np.sum(~finite))


def _cdf(dist: Any, x: np.ndarray) -> np.ndarray:
    """Vectorised CDF over the scalar wedgestats interface."""
    return np.array([dist.cdf(float(v)) for v in x], dtype=float)


# ===========================================================================
# P-P plot
# ===========================================================================


@dataclass(frozen=True)
class PPSpec:
    """A probability-probability plot: fitted CDF against plotting position.

    Attributes mirror :class:`~wedgelab.qq.QQSpec` where they mean the same
    thing, so a session can be switched between the two views without
    re-specifying the model.
    """

    data: np.ndarray
    dist_key: str = "normal"
    fit_method: str = "mle"
    manual_params: tuple[float, ...] | None = None
    position: Formula = field(default_factory=lambda: symmetric_plotting_position(0.375))
    position_bindings: dict[str, float] = field(default_factory=dict)
    line: str = "identity"
    envelope: str = "auto"
    alpha: float = 0.05
    bootstrap_reps: int = 500
    random_state: int = 0
    label: str = "sample"

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", np.asarray(self.data, dtype=float).ravel())
        if self.envelope not in PP_ENVELOPES:
            raise ValueError(f"envelope must be one of {PP_ENVELOPES}, got '{self.envelope}'")
        if self.line not in ("identity", "ols"):
            raise ValueError(f"line must be 'identity' or 'ols', got '{self.line}'")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")

    def replace(self, **changes: Any) -> "PPSpec":
        """Return a copy with fields replaced."""
        return replace(self, **changes)


@dataclass(frozen=True)
class PPResult:
    """Everything needed to draw and caption a P-P plot.

    Attributes
    ----------
    theoretical : np.ndarray
        Plotting positions, the horizontal axis.
    empirical : np.ndarray
        Fitted CDF evaluated at the ordered sample, the vertical axis.
    lower, upper : np.ndarray or None
        Envelope bounds, in probability units.
    """

    theoretical: np.ndarray
    empirical: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    outside: np.ndarray
    slope: float
    intercept: float
    envelope: str
    fit: FitResult
    n: int
    n_dropped: int
    correlation: float
    max_deviation: float
    outside_band: int
    spec: PPSpec
    warnings: tuple[str, ...] = ()

    def line_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Endpoints of the reference line across the unit square."""
        x = np.array([0.0, 1.0])
        return x, self.intercept + self.slope * x

    def lines(self) -> list[str]:
        """Diagnostic lines for a readout."""
        out = [
            f"n = {self.n}" + (f"  ({self.n_dropped} non-finite dropped)" if self.n_dropped else ""),
            f"correlation = {self.correlation:.5f}",
            f"max |F(x) - p| = {self.max_deviation:.5f}",
            f"slope = {self.slope:.5g}   intercept = {self.intercept:.5g}",
        ]
        if self.lower is not None:
            out.append(f"outside band: {self.outside_band}")
        return out

    def caption(self) -> str:
        """A self-contained figure caption."""
        band = {
            "none": "no envelope",
            "beta": f"exact pointwise {100 * (1 - self.spec.alpha):.0f}% envelope from "
            "the Beta(i, n-i+1) distribution of the order statistics",
            "simultaneous": f"simultaneous {100 * (1 - self.spec.alpha):.0f}% "
            "Kolmogorov-Smirnov envelope",
            "bootstrap": f"parametric bootstrap {100 * (1 - self.spec.alpha):.0f}% envelope "
            f"({self.spec.bootstrap_reps} replicates, seed {self.spec.random_state})",
        }[self.envelope]
        return (
            f"Probability-probability plot of {self.spec.label} (n = {self.n}) against a "
            f"{self.fit.summary()} reference. Plotting positions: {self.spec.position.name}. "
            f"Shown with {band}. A P-P plot resolves the centre of the distribution and is "
            f"insensitive in the tails. Correlation r = {self.correlation:.4f}."
        )


def _pp_bootstrap(fit_result: FitResult, spec: PPSpec, n: int, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pivotal bootstrap band in probability units.

    Each replicate is drawn from the fitted model, refitted, and pushed through
    its **own** refitted CDF, so it carries the estimation shrinkage the
    observed sample carries.  Without the refit this reduces to the Beta band.
    """
    rng = np.random.default_rng(spec.random_state)
    reps = int(spec.bootstrap_reps)
    draws = np.empty((reps, n), dtype=float)
    pivotal = spec.fit_method != "manual"
    for r in range(reps):
        sample = np.sort(
            np.asarray(
                fit_result.dist.rvs(size=n, random_state=int(rng.integers(0, 2**31 - 1))),
                dtype=float,
            ).ravel()
        )
        source = fit_result
        if pivotal:
            try:
                source = fit(sample, spec.dist_key, spec.fit_method)
            except FitError:
                source = fit_result
        draws[r] = _cdf(source.dist, sample)
    lo = np.quantile(draws, spec.alpha / 2.0, axis=0)
    hi = np.quantile(draws, 1.0 - spec.alpha / 2.0, axis=0)
    return lo, hi


def compute_pp(spec: PPSpec) -> PPResult:
    """Build a P-P plot from its specification.

    The vertical axis is the fitted CDF at each ordered observation; the
    horizontal axis is the plotting position that observation was assigned.
    Under a correct model the points lie on the identity.

    Note what this does **not** buy you. Because the fitted CDF is monotone,
    ``x_(i)`` lies inside a Q-Q band exactly when ``F(x_(i))`` lies inside the
    corresponding P-P band -- the two figures flag precisely the same points,
    always. The P-P plot is a change of axis, not a second test. Its value is
    resolution: it magnifies the centre of the distribution, where a Q-Q plot
    crowds every point into a short stretch of the line, and it flattens the
    tails, where a Q-Q plot spreads them out. Read the pair together to see
    *where* a model fails, not to get a second opinion on *whether* it does.

    Raises
    ------
    ValueError
        If the sample is too small or degenerate.
    FitError
        If the reference cannot be fitted.
    FormulaError
        If the plotting-position formula is invalid.
    """
    data, n_dropped = _clean(spec.data)
    n = data.size
    if n < 3:
        raise ValueError(f"at least three finite observations are required, got {n}")
    if float(np.ptp(data)) < _EPS:
        raise ValueError("the sample is constant; a P-P plot carries no information")

    warnings: list[str] = []
    if n_dropped:
        warnings.append(f"{n_dropped} non-finite observation(s) dropped")

    fit_result = fit(data, spec.dist_key, spec.fit_method, spec.manual_params)
    envelope = resolve_envelope(spec.envelope, spec.fit_method)
    if spec.envelope == "auto":
        warnings.append(f"envelope 'auto' resolved to '{envelope}'")
    if envelope == "asymptotic":  # not defined here; auto never picks it
        envelope = "beta"

    p = evaluate_positions(spec.position, n, spec.position_bindings)
    ordered = np.sort(data)
    empirical = _cdf(fit_result.dist, ordered)

    # A Q-Q plot survives a reference on the wrong scale: the quantiles simply
    # come out far from the data. A P-P plot does not. The CDF saturates, every
    # observation is assigned the same probability, and the vertical axis
    # collapses to a point -- taking the correlation with it. wedgestats
    # StudentT, ChiSquared and FDistribution carry no location or scale, so
    # this is reachable by choosing one for data that is not already centred.
    if float(np.ptp(empirical)) < 1e-10:
        label = fit_result.spec.label
        remedy = (
            f"{label} has no location or scale parameter: standardise the data "
            "first, or choose a reference distribution that has them."
            if label in ("Student t", "Chi-squared", "F")
            else "The reference sits on a completely different scale from the data."
        )
        raise ValueError(
            f"the fitted {label} assigns every observation the same probability "
            f"({float(empirical[0]):.4g}), so a P-P plot carries no information. "
            + remedy
        )

    lower = upper = None
    if envelope == "beta":
        lower, upper = beta_band(spec.alpha / 2.0, 1.0 - spec.alpha / 2.0, n)
        if spec.fit_method != "manual":
            warnings.append(
                "the exact Beta envelope assumes a fully specified reference; "
                "with estimated parameters use 'bootstrap' (or 'auto')"
            )
    elif envelope == "simultaneous":
        d = ks_critical(spec.alpha, n)
        lower = np.clip(p - d, 0.0, 1.0)
        upper = np.clip(p + d, 0.0, 1.0)
    elif envelope == "bootstrap":
        lower, upper = _pp_bootstrap(fit_result, spec, n, p)

    if spec.line == "identity":
        slope, intercept = 1.0, 0.0
    else:
        result = simple_ols(p, empirical)
        intercept, slope = (float(v) for v in result.coefficients[:2])

    outside = (
        (empirical < lower) | (empirical > upper)
        if lower is not None
        else np.zeros(n, dtype=bool)
    )

    return PPResult(
        theoretical=p,
        empirical=empirical,
        lower=lower,
        upper=upper,
        outside=outside,
        slope=float(slope),
        intercept=float(intercept),
        envelope=envelope,
        fit=fit_result,
        n=n,
        n_dropped=n_dropped,
        correlation=float(correlation(p, empirical)),
        max_deviation=float(np.max(np.abs(empirical - p))),
        outside_band=int(np.sum(outside)),
        spec=spec,
        warnings=tuple(warnings),
    )


# ===========================================================================
# Empirical CDF with a confidence band
# ===========================================================================


@dataclass(frozen=True)
class ECDFSpec:
    """The empirical distribution function, with the fitted model overlaid."""

    data: np.ndarray
    dist_key: str = "normal"
    fit_method: str = "mle"
    manual_params: tuple[float, ...] | None = None
    envelope: str = "simultaneous"
    alpha: float = 0.05
    show_model: bool = True
    label: str = "sample"

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", np.asarray(self.data, dtype=float).ravel())
        if self.envelope not in ECDF_ENVELOPES:
            raise ValueError(f"envelope must be one of {ECDF_ENVELOPES}, got '{self.envelope}'")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")

    def replace(self, **changes: Any) -> "ECDFSpec":
        """Return a copy with fields replaced."""
        return replace(self, **changes)


@dataclass(frozen=True)
class ECDFResult:
    """The empirical CDF, its band, and the fitted curve to compare against."""

    x: np.ndarray
    ecdf: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    model_x: np.ndarray
    model_cdf: np.ndarray
    fit: FitResult
    n: int
    n_dropped: int
    ks_statistic: float
    ks_excursion: float
    band_contains_model: bool
    envelope: str
    spec: ECDFSpec
    warnings: tuple[str, ...] = ()

    def lines(self) -> list[str]:
        """Diagnostic lines for a readout."""
        out = [
            f"n = {self.n}" + (f"  ({self.n_dropped} non-finite dropped)" if self.n_dropped else ""),
            f"KS D = {self.ks_statistic:.5f}",
        ]
        if self.lower is not None:
            verdict = "inside" if self.band_contains_model else "leaves"
            out.append(f"the fitted CDF {verdict} the band")
            out.append(f"largest excursion beyond the band = {self.ks_excursion:.5f}")
        return out

    def caption(self) -> str:
        """A self-contained figure caption."""
        level = 100 * (1 - self.spec.alpha)
        band = {
            "none": "no confidence band",
            "simultaneous": f"a {level:.0f}% simultaneous Kolmogorov-Smirnov band",
            "pointwise": f"a {level:.0f}% pointwise band from the Beta(i, n-i+1) "
            "distribution of the order statistics",
        }[self.envelope]
        verdict = (
            "The fitted model stays inside the band."
            if self.band_contains_model
            else "The fitted model leaves the band."
        )
        return (
            f"Empirical distribution function of {self.spec.label} (n = {self.n}) with "
            f"{band}, against a {self.fit.summary()} reference. "
            f"Kolmogorov-Smirnov D = {self.ks_statistic:.4f}. {verdict} A simultaneous "
            "band bounds the whole curve at once, which is what a claim about the "
            "distribution as a whole requires."
        )


def compute_ecdf(spec: ECDFSpec) -> ECDFResult:
    """Build an ECDF figure with a confidence band.

    The simultaneous band is the classic Kolmogorov-Smirnov construction:
    every hypothesised CDF lying entirely inside it is accepted at level
    ``alpha``.  That is a statement about the whole curve, which a pointwise
    band cannot make.
    """
    data, n_dropped = _clean(spec.data)
    n = data.size
    if n < 3:
        raise ValueError(f"at least three finite observations are required, got {n}")

    warnings: list[str] = []
    if n_dropped:
        warnings.append(f"{n_dropped} non-finite observation(s) dropped")

    fit_result = fit(data, spec.dist_key, spec.fit_method, spec.manual_params)

    x = np.sort(data)
    ecdf = np.arange(1, n + 1, dtype=float) / n

    lower = upper = None
    if spec.envelope == "simultaneous":
        d = ks_critical(spec.alpha, n)
        lower = np.clip(ecdf - d, 0.0, 1.0)
        upper = np.clip(ecdf - 1.0 / n + d, 0.0, 1.0)
    elif spec.envelope == "pointwise":
        lo_p, hi_p = beta_band(spec.alpha / 2.0, 1.0 - spec.alpha / 2.0, n)
        lower, upper = lo_p, hi_p

    model_cdf_at_x = _cdf(fit_result.dist, x)
    # Both one-sided sup distances, which together give the KS statistic.
    above = float(np.max(ecdf - model_cdf_at_x))
    below = float(np.max(model_cdf_at_x - (ecdf - 1.0 / n)))
    ks_statistic = max(above, below)

    contains = True
    excursion = 0.0
    if lower is not None:
        over = float(np.max(model_cdf_at_x - upper, initial=0.0))
        under = float(np.max(lower - model_cdf_at_x, initial=0.0))
        excursion = max(over, under, 0.0)
        contains = excursion <= 0.0

    if spec.envelope == "simultaneous" and spec.fit_method != "manual":
        warnings.append(
            "the Kolmogorov-Smirnov band assumes a fully specified reference; "
            "with parameters estimated from the same sample it is too wide, so "
            "a model staying inside it is weak evidence (see Lilliefors)"
        )

    # A smooth curve for the overlay, spanning a little past the data.
    pad = 0.04 * float(np.ptp(x)) if float(np.ptp(x)) > 0 else 1.0
    model_x = np.linspace(x[0] - pad, x[-1] + pad, 400)
    model_cdf = _cdf(fit_result.dist, model_x)

    return ECDFResult(
        x=x,
        ecdf=ecdf,
        lower=lower,
        upper=upper,
        model_x=model_x,
        model_cdf=model_cdf,
        fit=fit_result,
        n=n,
        n_dropped=n_dropped,
        ks_statistic=ks_statistic,
        ks_excursion=excursion,
        band_contains_model=bool(contains),
        envelope=spec.envelope,
        spec=spec,
        warnings=tuple(warnings),
    )


# ===========================================================================
# Two-sample Q-Q
# ===========================================================================


@dataclass(frozen=True)
class TwoSampleSpec:
    """Compare two samples to each other, with no reference distribution."""

    first: np.ndarray
    second: np.ndarray
    position: Formula = field(default_factory=lambda: symmetric_plotting_position(0.375))
    position_bindings: dict[str, float] = field(default_factory=dict)
    line: str = "ols"
    envelope: str = "bootstrap"
    alpha: float = 0.05
    bootstrap_reps: int = 500
    random_state: int = 0
    first_label: str = "sample A"
    second_label: str = "sample B"

    def __post_init__(self) -> None:
        object.__setattr__(self, "first", np.asarray(self.first, dtype=float).ravel())
        object.__setattr__(self, "second", np.asarray(self.second, dtype=float).ravel())
        if self.envelope not in TWO_SAMPLE_ENVELOPES:
            raise ValueError(
                f"envelope must be one of {TWO_SAMPLE_ENVELOPES}, got '{self.envelope}'. "
                "The exact Beta band is not available here: it assumes a fixed "
                "reference, and in a two-sample plot both axes are random."
            )
        if self.line not in ("ols", "quartile", "identity"):
            raise ValueError(f"line must be 'ols', 'quartile' or 'identity', got '{self.line}'")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")

    def replace(self, **changes: Any) -> "TwoSampleSpec":
        """Return a copy with fields replaced."""
        return replace(self, **changes)


@dataclass(frozen=True)
class TwoSampleResult:
    """Two empirical samples compared at common quantiles.

    The reference line carries the interpretation: its slope estimates the
    ratio of scales and its intercept the difference in location.
    """

    first_quantiles: np.ndarray
    second_quantiles: np.ndarray
    probabilities: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    outside: np.ndarray
    slope: float
    intercept: float
    n_first: int
    n_second: int
    n_common: int
    correlation: float
    ks_statistic: float
    ks_p_value: float
    outside_band: int
    spec: TwoSampleSpec
    warnings: tuple[str, ...] = ()

    def line_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Endpoints of the reference line across the plotted range."""
        x = np.array([self.first_quantiles.min(), self.first_quantiles.max()])
        return x, self.intercept + self.slope * x

    def lines(self) -> list[str]:
        """Diagnostic lines for a readout."""
        out = [
            f"n = {self.n_first} vs {self.n_second}, compared on {self.n_common} quantiles",
            f"slope = {self.slope:.5g}   (ratio of scales)",
            f"intercept = {self.intercept:.5g}   (shift in location)",
            f"correlation = {self.correlation:.5f}",
            f"two-sample KS D = {self.ks_statistic:.5f}, p = {self.ks_p_value:.4g}",
        ]
        if self.lower is not None:
            out.append(f"outside band: {self.outside_band}")
        return out

    def caption(self) -> str:
        """A self-contained figure caption."""
        band = (
            "no confidence envelope"
            if self.lower is None
            else f"a bootstrap {100 * (1 - self.spec.alpha):.0f}% envelope "
            f"({self.spec.bootstrap_reps} replicates, seed {self.spec.random_state})"
        )
        return (
            f"Two-sample quantile-quantile plot of {self.spec.second_label} "
            f"(n = {self.n_second}) against {self.spec.first_label} (n = {self.n_first}), "
            f"compared at {self.n_common} common quantiles and shown with {band}. "
            f"The fitted slope of {self.slope:.3f} estimates the ratio of scales and the "
            f"intercept of {self.intercept:.3f} the shift in location. Two-sample "
            f"Kolmogorov-Smirnov D = {self.ks_statistic:.4f}, p = {self.ks_p_value:.3g}."
        )


def _two_sample_line(rule: str, qa: np.ndarray, qb: np.ndarray) -> tuple[float, float]:
    """Return ``(slope, intercept)`` for a two-sample reference line.

    Shared by the observed fit and by every bootstrap replicate.  They must use
    the same rule: a band whose replicates refit a least-squares line while the
    figure draws the identity is centred on one thing and scaled by another,
    and flags most of a *correctly matched* pair.
    """
    if rule == "identity":
        return 1.0, 0.0
    if rule == "quartile":
        aq1, _, aq3 = quartiles(qa)
        bq1, _, bq3 = quartiles(qb)
        if abs(aq3 - aq1) < _EPS:
            raise ValueError("quartile line is undefined: the first sample's quartiles coincide")
        slope = (bq3 - bq1) / (aq3 - aq1)
        return float(slope), float(bq1 - slope * aq1)
    fitted = simple_ols(qa, qb)
    return float(fitted.coefficients[1]), float(fitted.coefficients[0])


def compute_two_sample(spec: TwoSampleSpec) -> TwoSampleResult:
    """Compare two samples at common quantiles.

    No reference distribution is fitted, so no exact band is available: the
    Beta envelope describes order statistics around a *fixed* reference, and
    here both axes carry sampling error.  The bootstrap resamples both samples
    and is the only honest envelope offered.
    """
    first, dropped_a = _clean(spec.first)
    second, dropped_b = _clean(spec.second)
    if first.size < 3 or second.size < 3:
        raise ValueError("both samples need at least three finite observations")
    if float(np.ptp(first)) < _EPS or float(np.ptp(second)) < _EPS:
        raise ValueError("a constant sample carries no quantile information")

    warnings: list[str] = []
    if dropped_a or dropped_b:
        warnings.append(f"{dropped_a + dropped_b} non-finite observation(s) dropped")

    n_common = min(first.size, second.size)
    p = evaluate_positions(spec.position, n_common, spec.position_bindings)
    qa = np.quantile(first, p)
    qb = np.quantile(second, p)

    slope, intercept = _two_sample_line(spec.line, qa, qb)

    lower = upper = None
    if spec.envelope == "bootstrap":
        # The null is that both samples share a distribution, so pooling and
        # re-splitting at the observed sizes preserves it while destroying any
        # difference. The band must be *pivotal*: the reference line is refitted
        # inside every replicate, so each draw carries the same line-fitting
        # variability the observed residuals carry. Centring the replicates on
        # the observed line instead leaves the band badly unstable across seeds
        # -- measured at 3.8% coverage on average but swinging to 26% on an
        # unlucky one, against a nominal 5%.
        rng = np.random.default_rng(spec.random_state)
        reps = int(spec.bootstrap_reps)
        draws = np.empty((reps, n_common), dtype=float)
        pooled = np.concatenate([first, second])
        for r in range(reps):
            picks = rng.choice(pooled, size=pooled.size, replace=True)
            qa_star = np.quantile(picks[: first.size], p)
            qb_star = np.quantile(picks[first.size :], p)
            m_star, c_star = _two_sample_line(spec.line, qa_star, qb_star)
            draws[r] = qb_star - (c_star + m_star * qa_star)
        lo = np.quantile(draws, spec.alpha / 2.0, axis=0)
        hi = np.quantile(draws, 1.0 - spec.alpha / 2.0, axis=0)
        centre = intercept + slope * qa
        lower, upper = centre + lo, centre + hi
        warnings.append(
            "the bootstrap envelope is built under the null that both samples "
            "share a distribution, by pooling and resampling with the reference "
            "line refitted in each replicate; it runs somewhat conservative at "
            "large n"
        )

    outside = (
        (qb < lower) | (qb > upper) if lower is not None else np.zeros(n_common, dtype=bool)
    )
    ks = sp_stats.ks_2samp(first, second)

    return TwoSampleResult(
        first_quantiles=qa,
        second_quantiles=qb,
        probabilities=p,
        lower=lower,
        upper=upper,
        outside=outside,
        slope=float(slope),
        intercept=float(intercept),
        n_first=int(first.size),
        n_second=int(second.size),
        n_common=int(n_common),
        correlation=float(correlation(qa, qb)),
        ks_statistic=float(ks.statistic),
        ks_p_value=float(ks.pvalue),
        outside_band=int(np.sum(outside)),
        spec=spec,
        warnings=tuple(warnings),
    )
