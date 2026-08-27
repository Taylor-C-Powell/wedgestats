"""The Q-Q plot, assembled from replaceable parts.

A Q-Q plot is not one calculation but five, and every one of them is a choice
a statistician can defend or get wrong:

1. **Plotting positions** -- what probability does the i-th order statistic
   get?  Supplied as an editable :class:`~wedgelab.formula.Formula`.
2. **Reference distribution and estimator** -- see :mod:`wedgelab.models`.
3. **Reference line** -- least squares, quartile, or the theoretical identity.
4. **Confidence envelope** -- exact Beta, asymptotic, simultaneous, bootstrap.
5. **Presentation** -- raw, standardised, or detrended.

This module makes each of the five an explicit field of :class:`QQSpec` and
returns everything the figure and its caption need in :class:`QQResult`.

The statistics come from :mod:`wedgestats`: quantiles from the fitted
distribution's ``ppf``, the exact envelope from ``Beta(i, n-i+1)``, the
reference line from ``simple_ols``, and the probability plot correlation
coefficient from ``correlation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from wedgestats.descriptive import correlation, describe, quartiles
from wedgestats.distributions import Beta, ContinuousDistribution
from wedgestats.regression import simple_ols

from wedgelab.formula import Formula, FormulaError
from wedgelab.knowledge import symmetric_plotting_position
from wedgelab.models import FitError, FitResult, fit

__all__ = [
    "QQSpec",
    "QQResult",
    "Diagnostics",
    "compute",
    "evaluate_positions",
    "check_positions",
    "ENVELOPE_METHODS",
    "LINE_METHODS",
]

ENVELOPE_METHODS: tuple[str, ...] = ("none", "beta", "asymptotic", "simultaneous", "bootstrap")
LINE_METHODS: tuple[str, ...] = ("ols", "quartile", "theoretical")

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QQSpec:
    """Every decision behind a Q-Q plot, in one immutable record.

    Attributes
    ----------
    data : np.ndarray
        The sample.  Non-finite values are dropped and counted.
    dist_key : str
        Reference distribution, a key of :data:`wedgelab.models.DISTRIBUTIONS`.
    fit_method : str
        ``"mle"``, ``"moments"``, ``"robust"`` or ``"manual"``.
    manual_params : tuple[float, ...] or None
        Parameters when ``fit_method == "manual"``.
    position : Formula
        Plotting-position formula in ``i`` and ``n``.
    position_bindings : dict[str, float]
        Values for the formula's tunable parameters.
    line : str
        Reference-line rule; one of :data:`LINE_METHODS`.
    envelope : str
        Confidence-envelope rule; one of :data:`ENVELOPE_METHODS`.
    alpha : float
        Envelope significance level.
    standardize : bool
        Put both axes on the fitted distribution's z-scale.
    detrend : bool
        Plot departures from the reference line instead of raw quantiles.
    bootstrap_reps : int
        Replicates for the bootstrap envelope.
    random_state : int
        Seed for the bootstrap, so a published figure is reproducible.
    label : str
        Name of the sample, used in titles and legends.
    """

    data: np.ndarray
    dist_key: str = "normal"
    fit_method: str = "mle"
    manual_params: tuple[float, ...] | None = None
    position: Formula = field(default_factory=lambda: symmetric_plotting_position(0.375))
    position_bindings: dict[str, float] = field(default_factory=dict)
    line: str = "ols"
    envelope: str = "beta"
    alpha: float = 0.05
    standardize: bool = False
    detrend: bool = False
    bootstrap_reps: int = 1000
    random_state: int = 0
    label: str = "sample"

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", np.asarray(self.data, dtype=float).ravel())
        if self.line not in LINE_METHODS:
            raise ValueError(f"line must be one of {LINE_METHODS}, got '{self.line}'")
        if self.envelope not in ENVELOPE_METHODS:
            raise ValueError(
                f"envelope must be one of {ENVELOPE_METHODS}, got '{self.envelope}'"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")
        if self.bootstrap_reps < 20:
            raise ValueError("bootstrap_reps must be at least 20")

    def replace(self, **changes: Any) -> "QQSpec":
        """Return a copy with fields replaced."""
        return replace(self, **changes)

    def positions_summary(self) -> str:
        """One-line description of the plotting-position rule in force."""
        bound = {**self.position.defaults(), **self.position_bindings}
        if bound:
            args = ", ".join(f"{k}={v:.4g}" for k, v in sorted(bound.items()))
            return f"{self.position.name} [{args}]"
        return self.position.name


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnostics:
    """Numbers that belong in the caption of a Q-Q figure."""

    n: int
    n_dropped: int
    ppcc: float
    slope: float
    intercept: float
    r_squared: float
    slope_se: float
    intercept_se: float
    skewness: float
    kurtosis: float
    outside_band: int
    expected_outside: float
    shapiro_w: float | None = None
    shapiro_p: float | None = None
    anderson_a2: float | None = None
    ks_d: float | None = None
    ks_p: float | None = None

    def lines(self) -> list[str]:
        """Human-readable diagnostic lines for the GUI readout."""
        out = [
            f"n = {self.n}" + (f"  ({self.n_dropped} non-finite dropped)" if self.n_dropped else ""),
            f"PPCC r = {self.ppcc:.5f}   (Filliben)",
            f"slope = {self.slope:.5g} +/- {self.slope_se:.3g}",
            f"intercept = {self.intercept:.5g} +/- {self.intercept_se:.3g}",
            f"R^2 = {self.r_squared:.5f}",
            f"skewness = {self.skewness:+.4f}   excess kurtosis = {self.kurtosis:+.4f}",
        ]
        if self.expected_outside > 0:
            out.append(
                f"outside band: {self.outside_band} "
                f"(expected about {self.expected_outside:.1f} under the model)"
            )
        if self.shapiro_w is not None:
            out.append(f"Shapiro-Wilk W = {self.shapiro_w:.5f}, p = {self.shapiro_p:.4g}")
        if self.anderson_a2 is not None:
            out.append(f"Anderson-Darling A^2 = {self.anderson_a2:.4f}")
        if self.ks_d is not None:
            out.append(f"Kolmogorov-Smirnov D = {self.ks_d:.5f}, p = {self.ks_p:.4g}")
        return out


@dataclass(frozen=True)
class QQResult:
    """Everything needed to draw and describe one Q-Q plot.

    Attributes
    ----------
    theoretical : np.ndarray
        Quantiles of the fitted reference distribution, ascending.
    sample : np.ndarray
        Ordered sample values on the plotted scale.
    probabilities : np.ndarray
        Plotting positions actually used.
    lower, upper : np.ndarray or None
        Envelope bounds on the plotted scale, or ``None``.
    outside : np.ndarray
        Boolean mask of points beyond the envelope.
    slope, intercept : float
        Reference line on the plotted scale.
    fit : FitResult
        The fitted reference distribution and its provenance.
    diagnostics : Diagnostics
        Summary statistics.
    spec : QQSpec
        The specification that produced this result.
    warnings : tuple[str, ...]
        Non-fatal issues the caller should surface.
    """

    theoretical: np.ndarray
    sample: np.ndarray
    probabilities: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    outside: np.ndarray
    slope: float
    intercept: float
    fit: FitResult
    diagnostics: Diagnostics
    spec: QQSpec
    warnings: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        """Number of plotted points."""
        return int(self.sample.size)

    def line_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Endpoints of the reference line, spanning the theoretical axis."""
        x = np.array([self.theoretical.min(), self.theoretical.max()], dtype=float)
        return x, self.intercept + self.slope * x

    def caption(self) -> str:
        """A defensible, self-contained figure caption."""
        d = self.diagnostics
        band = {
            "none": "no confidence envelope",
            "beta": f"exact pointwise {100 * (1 - self.spec.alpha):.0f}% envelope from "
            "the Beta(i, n-i+1) distribution of the order statistics",
            "asymptotic": f"asymptotic pointwise {100 * (1 - self.spec.alpha):.0f}% envelope "
            "from the delta-method standard error",
            "simultaneous": f"simultaneous {100 * (1 - self.spec.alpha):.0f}% "
            "Kolmogorov-Smirnov envelope",
            "bootstrap": f"parametric bootstrap pointwise "
            f"{100 * (1 - self.spec.alpha):.0f}% envelope "
            f"({self.spec.bootstrap_reps} replicates, seed {self.spec.random_state})",
        }[self.spec.envelope]
        line = {
            "ols": "least-squares reference line",
            "quartile": "quartile reference line",
            "theoretical": "theoretical identity line",
        }[self.spec.line]
        scale = " Both axes are standardised." if self.spec.standardize else ""
        detr = " Vertical axis shows departures from the reference line." if self.spec.detrend else ""
        return (
            f"Quantile-quantile plot of {self.spec.label} (n = {d.n}) against a "
            f"{self.fit.summary()} reference. Plotting positions: "
            f"{self.spec.positions_summary()}. Shown with a {line} and {band}. "
            f"Probability plot correlation coefficient r = {d.ppcc:.4f}.{scale}{detr}"
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def evaluate_positions(
    position: Formula, n: int, bindings: dict[str, float] | None = None
) -> np.ndarray:
    """Evaluate a plotting-position formula and enforce its contract.

    A plotting position must be finite, strictly inside ``(0, 1)`` -- a
    probability of exactly 0 or 1 maps to an infinite quantile -- and
    non-decreasing in the rank ``i``.

    Parameters
    ----------
    position : Formula
        Expression in ``i`` and ``n``.
    n : int
        Sample size.
    bindings : dict[str, float] or None
        Values for the formula's tunable parameters.

    Returns
    -------
    np.ndarray
        Probabilities of length *n*.

    Raises
    ------
    FormulaError
        If evaluation fails or the contract is violated.
    """
    i = np.arange(1, n + 1, dtype=float)
    env = {**(bindings or {}), "i": i, "n": float(n)}
    try:
        p = np.asarray(position.evaluate(**env), dtype=float)
    except FormulaError as exc:
        raise FormulaError(f"plotting position formula failed: {exc}") from exc

    p = np.broadcast_to(p, (n,)).astype(float, copy=True)
    if not np.all(np.isfinite(p)):
        raise FormulaError("plotting position formula produced non-finite values")
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise FormulaError(
            "plotting positions must lie strictly inside (0, 1); got range "
            f"[{p.min():.4g}, {p.max():.4g}]"
        )
    if np.any(np.diff(p) < 0):
        raise FormulaError("plotting positions must be non-decreasing in i")
    return p


def check_positions(
    position: Formula, n: int, bindings: dict[str, float] | None = None
) -> tuple[bool, str]:
    """Report whether a plotting-position formula is usable, without raising.

    The workbench calls this before adopting an edited expression, so that a
    formula the engine would reject is never installed in the first place.

    Returns
    -------
    tuple[bool, str]
        ``(ok, message)``; *message* is empty when ``ok`` is ``True``.
    """
    try:
        evaluate_positions(position, n, bindings)
    except FormulaError as exc:
        return False, str(exc)
    return True, ""


def _plotting_positions(spec: QQSpec, n: int) -> np.ndarray:
    """Evaluate the specification's plotting positions."""
    return evaluate_positions(spec.position, n, spec.position_bindings)


def _quantiles(dist: ContinuousDistribution, p: np.ndarray) -> np.ndarray:
    """Vectorised inverse CDF using the wedgestats scalar ``ppf``."""
    return np.array([dist.ppf(float(v)) for v in p], dtype=float)


def _beta_band(p_lo: float, p_hi: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact probability bounds for every order statistic.

    ``U_(i) ~ Beta(i, n - i + 1)`` exactly, so the bounds come straight from
    the wedgestats Beta quantile function.
    """
    lo = np.empty(n, dtype=float)
    hi = np.empty(n, dtype=float)
    for idx in range(n):
        b = Beta(alpha=float(idx + 1), beta=float(n - idx))
        lo[idx] = b.ppf(p_lo)
        hi[idx] = b.ppf(p_hi)
    return lo, hi


def _ks_critical(alpha: float, n: int) -> float:
    """Two-sided critical sup-norm distance for a simultaneous band.

    Uses the exact two-sided Kolmogorov distribution when SciPy can invert it,
    and the Dvoretzky-Kiefer-Wolfowitz asymptotic otherwise.
    """
    try:
        d = float(sp_stats.kstwo.ppf(1.0 - alpha, n))
        if np.isfinite(d) and d > 0:
            return d
    except Exception:
        pass
    return float(np.sqrt(-np.log(alpha / 2.0) / (2.0 * n)))


def _quantiles_unbounded(dist: ContinuousDistribution, p: np.ndarray) -> np.ndarray:
    """Inverse CDF that returns an infinite bound outside ``(0, 1)``.

    A simultaneous band genuinely has no finite bound where ``p_i +/- d``
    leaves the unit interval.  Reporting an infinity is honest; clipping to a
    tiny epsilon would invent an extreme finite number instead.
    """
    out = np.empty(p.size, dtype=float)
    for k, v in enumerate(p):
        if v <= 0.0:
            out[k] = -np.inf
        elif v >= 1.0:
            out[k] = np.inf
        else:
            out[k] = dist.ppf(float(v))
    return out


def _bootstrap_band(
    fit_result: FitResult,
    spec: QQSpec,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise band from order statistics of simulated samples.

    Each replicate is drawn from the fitted model and *refitted* with the same
    estimator, so the band reflects the parameter uncertainty that the exact
    Beta band ignores.
    """
    rng = np.random.default_rng(spec.random_state)
    reps = int(spec.bootstrap_reps)
    draws = np.empty((reps, n), dtype=float)
    refit_failures = 0
    for r in range(reps):
        sample = np.asarray(
            fit_result.dist.rvs(size=n, random_state=int(rng.integers(0, 2**31 - 1))),
            dtype=float,
        ).ravel()
        if spec.fit_method == "manual":
            draws[r] = np.sort(sample)
            continue
        try:
            refit = fit(sample, spec.dist_key, spec.fit_method)
            centred = _standardise(np.sort(sample), refit) if spec.standardize else np.sort(sample)
        except FitError:
            refit_failures += 1
            centred = np.sort(sample)
        draws[r] = centred
    lo = np.quantile(draws, spec.alpha / 2.0, axis=0)
    hi = np.quantile(draws, 1.0 - spec.alpha / 2.0, axis=0)
    return lo, hi


def _standardise(values: np.ndarray, fit_result: FitResult) -> np.ndarray:
    """Map values onto the fitted distribution's z-scale."""
    mu = float(fit_result.dist.mean())
    sigma = float(fit_result.dist.std_dev())
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        return values
    return (values - mu) / sigma


def _reference_line(
    method: str,
    theoretical: np.ndarray,
    sample: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Return ``(slope, intercept, r_squared, slope_se, intercept_se)``."""
    if method == "theoretical":
        return 1.0, 0.0, float("nan"), 0.0, 0.0

    if method == "quartile":
        tq1, _, tq3 = quartiles(theoretical)
        sq1, _, sq3 = quartiles(sample)
        spread = tq3 - tq1
        if abs(spread) < _EPS:
            raise ValueError("quartile line is undefined: theoretical quartiles coincide")
        slope = (sq3 - sq1) / spread
        return slope, sq1 - slope * tq1, float("nan"), 0.0, 0.0

    result = simple_ols(theoretical, sample)
    intercept, slope = (float(v) for v in result.coefficients[:2])
    se_intercept, se_slope = (float(v) for v in result.std_errors[:2])
    return slope, intercept, float(result.r_squared), se_slope, se_intercept


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute(spec: QQSpec) -> QQResult:
    """Turn a :class:`QQSpec` into everything needed to draw and caption a plot.

    Parameters
    ----------
    spec : QQSpec
        The full specification.

    Returns
    -------
    QQResult

    Raises
    ------
    FitError
        If the reference distribution cannot be fitted.
    FormulaError
        If the plotting-position formula is invalid or out of range.
    ValueError
        If the sample is too small or degenerate.
    """
    raw = np.asarray(spec.data, dtype=float).ravel()
    finite = np.isfinite(raw)
    n_dropped = int(np.sum(~finite))
    data = raw[finite]
    n = data.size
    if n < 3:
        raise ValueError(f"at least three finite observations are required, got {n}")

    # Checked before fitting, so a degenerate sample gets an explanation rather
    # than a downstream complaint about a zero scale parameter.
    if float(np.ptp(data)) < _EPS:
        raise ValueError(
            f"the sample is constant (every value is {data[0]:.6g}); a Q-Q plot "
            "carries no information"
        )

    warnings: list[str] = []
    if n_dropped:
        warnings.append(f"{n_dropped} non-finite observation(s) dropped")

    fit_result = fit(data, spec.dist_key, spec.fit_method, spec.manual_params)
    if fit_result.fell_back:
        warnings.append(
            f"requested '{fit_result.requested_method}' estimation; "
            f"used '{fit_result.method}' instead"
        )

    p = _plotting_positions(spec, n)
    ordered = np.sort(data)
    theoretical = _quantiles(fit_result.dist, p)

    if not np.all(np.isfinite(theoretical)):
        raise ValueError(
            "the reference distribution produced non-finite quantiles; try a "
            "plotting-position rule that keeps the extremes away from 0 and 1"
        )

    # ---- envelope, computed on the raw sample scale -----------------------
    lower_raw: np.ndarray | None = None
    upper_raw: np.ndarray | None = None

    if spec.envelope == "beta":
        p_lo, p_hi = _beta_band(spec.alpha / 2.0, 1.0 - spec.alpha / 2.0, n)
        lower_raw = _quantiles(fit_result.dist, p_lo)
        upper_raw = _quantiles(fit_result.dist, p_hi)
        if spec.fit_method != "manual":
            warnings.append(
                "the exact Beta envelope assumes a fully specified reference; "
                "with estimated parameters it is conservative -- use the "
                "bootstrap envelope for an exact-level band"
            )
    elif spec.envelope == "simultaneous":
        d = _ks_critical(spec.alpha, n)
        lower_raw = _quantiles_unbounded(fit_result.dist, p - d)
        upper_raw = _quantiles_unbounded(fit_result.dist, p + d)
        n_open = int(np.sum(~np.isfinite(lower_raw)) + np.sum(~np.isfinite(upper_raw)))
        if n_open:
            warnings.append(
                f"the simultaneous band is unbounded at {n_open} endpoint(s) "
                f"where p_i +/- {d:.4g} leaves (0, 1); those bounds are drawn "
                "to the axis edge"
            )
    elif spec.envelope == "bootstrap":
        lower_raw, upper_raw = _bootstrap_band(fit_result, spec, n)

    # ---- move onto the plotted scale --------------------------------------
    if spec.standardize:
        sample = _standardise(ordered, fit_result)
        theoretical = _standardise(theoretical, fit_result)
        if lower_raw is not None and spec.envelope != "bootstrap":
            lower_raw = _standardise(lower_raw, fit_result)
            upper_raw = _standardise(upper_raw, fit_result)
    else:
        sample = ordered

    if np.ptp(sample) < _EPS:
        raise ValueError("the sample is constant; a Q-Q plot carries no information")

    slope, intercept, r_squared, slope_se, intercept_se = _reference_line(
        spec.line, theoretical, sample
    )

    # The asymptotic band is centred on the fitted model's own quantiles, not
    # on the reference line.  The delta method states that
    # X_(i) is approximately N(F^-1(p_i), se_i^2), so F^-1(p_i) is the correct
    # centre -- and it keeps all four envelopes answering the same question,
    # which makes them comparable.
    if spec.envelope == "asymptotic":
        z = float(sp_stats.norm.ppf(1.0 - spec.alpha / 2.0))
        raw_quantiles = _quantiles(fit_result.dist, p)
        density = np.array(
            [max(float(fit_result.dist.pdf(float(q))), _EPS) for q in raw_quantiles]
        )
        # Delta-method standard error of the i-th order statistic, in the
        # units of the raw data.  Cramer (1946), section 28.5.
        se = np.sqrt(p * (1.0 - p) / n) / density
        centre = raw_quantiles
        if spec.standardize:
            sigma = float(fit_result.dist.std_dev())
            if np.isfinite(sigma) and sigma > 0:
                se = se / sigma
            centre = _standardise(raw_quantiles, fit_result)
        lower_raw = centre - z * se
        upper_raw = centre + z * se
        if np.any(se > 10.0 * np.ptp(sample)):
            warnings.append(
                "the asymptotic band is very wide in the tails, where the "
                "density in its denominator approaches zero; prefer the exact "
                "Beta or bootstrap envelope"
            )

    lower = lower_raw
    upper = upper_raw

    # The probability plot correlation coefficient is a property of the Q-Q
    # relationship itself, so it is measured before detrending.  Detrended
    # residuals are orthogonal to the theoretical quantiles by construction,
    # and reporting the resulting r of exactly zero would be meaningless.
    ppcc = float(correlation(theoretical, sample, method="pearson"))

    # The reference line stays in the diagnostics even after detrending, where
    # the *drawn* line flattens to y = 0.  Reporting a slope of exactly zero
    # would hide the fitted relationship the reader needs.
    line_slope, line_intercept = slope, intercept

    if spec.detrend:
        trend = intercept + slope * theoretical
        sample = sample - trend
        if lower is not None:
            lower = lower - trend
            upper = upper - trend
        slope, intercept = 0.0, 0.0

    if lower is not None and upper is not None:
        outside = (sample < lower) | (sample > upper)
    else:
        outside = np.zeros(n, dtype=bool)

    # ---- diagnostics ------------------------------------------------------
    summary = describe(ordered)
    expected_outside = float(spec.alpha * n) if spec.envelope in ("beta", "asymptotic", "bootstrap") else 0.0

    shapiro_w = shapiro_p = anderson_a2 = ks_d = ks_p = None
    if spec.dist_key == "normal" and 3 <= n <= 5000:
        try:
            w, pv = sp_stats.shapiro(ordered)
            shapiro_w, shapiro_p = float(w), float(pv)
        except Exception:
            pass
    if spec.dist_key == "normal":
        try:
            anderson_a2 = float(sp_stats.anderson(ordered, dist="norm").statistic)
        except Exception:
            pass
    try:
        ks = sp_stats.kstest(
            ordered,
            lambda x: np.array(
                [fit_result.dist.cdf(float(v)) for v in np.atleast_1d(x)]
            ),
        )
        ks_d, ks_p = float(ks.statistic), float(ks.pvalue)
        if spec.fit_method != "manual":
            # The Kolmogorov-Smirnov null distribution assumes a fully
            # specified reference.  Estimating the parameters from the same
            # data makes D stochastically smaller, so the p-value is
            # optimistic -- often badly so.  Lilliefors (1967) is the fix.
            warnings.append(
                "the Kolmogorov-Smirnov p-value is not valid here: its null "
                "distribution assumes fixed parameters, and these were "
                "estimated from the same sample, which makes it optimistic "
                "(see the Lilliefors entry in the knowledge base)"
            )
    except Exception:
        pass

    diagnostics = Diagnostics(
        n=n,
        n_dropped=n_dropped,
        ppcc=ppcc,
        slope=float(line_slope),
        intercept=float(line_intercept),
        r_squared=r_squared,
        slope_se=slope_se,
        intercept_se=intercept_se,
        skewness=float(summary.skewness),
        kurtosis=float(summary.kurtosis),
        outside_band=int(np.sum(outside)),
        expected_outside=expected_outside,
        shapiro_w=shapiro_w,
        shapiro_p=shapiro_p,
        anderson_a2=anderson_a2,
        ks_d=ks_d,
        ks_p=ks_p,
    )

    return QQResult(
        theoretical=theoretical,
        sample=sample,
        probabilities=p,
        lower=lower,
        upper=upper,
        outside=outside,
        slope=float(slope),
        intercept=float(intercept),
        fit=fit_result,
        diagnostics=diagnostics,
        spec=spec,
        warnings=tuple(warnings),
    )
