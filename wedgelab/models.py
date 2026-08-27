"""Reference-distribution registry and parameter estimation.

Q-Q plotting needs a *fitted* reference distribution, and the honest version
of that sentence includes which estimator produced it.  This module maps the
continuous :mod:`wedgestats` distributions onto three estimation strategies
(maximum likelihood, method of moments, robust median/MAD) plus a manual mode,
and records what actually happened -- including any fallback -- in the result.

Every fit returns a live :class:`wedgestats.distributions.Distribution`, so the
rest of the toolkit only ever talks to the wedgestats API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats as sp_stats

from wedgestats.descriptive import mad, mean, median, skewness, std_dev, variance
from wedgestats.distributions import (
    Beta,
    ChiSquared,
    ContinuousDistribution,
    ContinuousUniform,
    Exponential,
    FDistribution,
    Gamma,
    Normal,
    StudentT,
)

__all__ = [
    "DistributionSpec",
    "FitResult",
    "FitError",
    "DISTRIBUTIONS",
    "distribution_keys",
    "fit",
    "FIT_METHODS",
]


class FitError(ValueError):
    """Raised when a distribution cannot be fitted to the supplied data."""


FIT_METHODS: tuple[str, ...] = ("mle", "moments", "robust", "manual")

# Consistency constant making the MAD an unbiased scale estimate under
# normality: 1 / Phi^-1(0.75).  See Rousseeuw and Croux (1993).
MAD_CONSISTENCY = 1.4826022185056018


@dataclass(frozen=True)
class DistributionSpec:
    """Metadata binding a wedgestats distribution to its estimators.

    Attributes
    ----------
    key : str
        Stable identifier used in the GUI and exported scripts.
    label : str
        Display name.
    param_names : tuple[str, ...]
        Constructor keyword names, in order.
    support : str
        One of ``"real"``, ``"positive"``, ``"unit"``.
    build : Callable
        Maps a parameter tuple to a wedgestats distribution instance.
    mle : Callable
        ``(data) -> tuple`` of parameters by maximum likelihood.
    moments : Callable or None
        ``(data) -> tuple`` by method of moments, if one is defined.
    robust : Callable or None
        ``(data) -> tuple`` from resistant summaries, if one is defined.
    notes : str
        Anything the user should know before choosing this reference.
    """

    key: str
    label: str
    param_names: tuple[str, ...]
    support: str
    build: Callable[..., ContinuousDistribution]
    mle: Callable[[np.ndarray], tuple[float, ...]]
    moments: Callable[[np.ndarray], tuple[float, ...]] | None = None
    robust: Callable[[np.ndarray], tuple[float, ...]] | None = None
    notes: str = ""

    def make(self, params: tuple[float, ...]) -> ContinuousDistribution:
        """Instantiate the wedgestats distribution from a parameter tuple."""
        return self.build(*params)

    def describe_params(self, params: tuple[float, ...]) -> str:
        """Format parameters as ``name=value`` pairs."""
        return ", ".join(
            f"{n}={v:.6g}" for n, v in zip(self.param_names, params, strict=False)
        )

    def validate_support(self, data: np.ndarray) -> None:
        """Raise :class:`FitError` if *data* leaves the distribution's support."""
        if self.support == "positive" and np.any(data <= 0):
            raise FitError(
                f"{self.label} has support x > 0 but the data contain "
                f"{int(np.sum(data <= 0))} non-positive value(s)"
            )
        if self.support == "unit" and (np.any(data <= 0) or np.any(data >= 1)):
            raise FitError(
                f"{self.label} has support 0 < x < 1 but the data fall outside it"
            )


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _normal_mle(data: np.ndarray) -> tuple[float, float]:
    loc, scale = sp_stats.norm.fit(data)
    return float(loc), float(scale)


def _normal_moments(data: np.ndarray) -> tuple[float, float]:
    return float(mean(data)), float(std_dev(data))


def _normal_robust(data: np.ndarray) -> tuple[float, float]:
    scale = MAD_CONSISTENCY * float(mad(data))
    if scale <= 0:
        raise FitError("robust scale estimate is zero (more than half the data are tied)")
    return float(median(data)), scale


def _expon_mle(data: np.ndarray) -> tuple[float]:
    _, scale = sp_stats.expon.fit(data, floc=0.0)
    if scale <= 0:
        raise FitError("exponential scale estimate is zero")
    return (float(1.0 / scale),)


def _expon_moments(data: np.ndarray) -> tuple[float]:
    m = float(mean(data))
    if m <= 0:
        raise FitError("exponential mean must be positive")
    return (float(1.0 / m),)


def _expon_robust(data: np.ndarray) -> tuple[float]:
    # The exponential median is ln(2)/lambda, so lambda = ln(2)/median.
    med = float(median(data))
    if med <= 0:
        raise FitError("exponential median must be positive")
    return (float(np.log(2.0) / med),)


def _gamma_mle(data: np.ndarray) -> tuple[float, float]:
    a, _, scale = sp_stats.gamma.fit(data, floc=0.0)
    return float(a), float(1.0 / scale)


def _gamma_moments(data: np.ndarray) -> tuple[float, float]:
    m = float(mean(data))
    v = float(variance(data))
    if v <= 0 or m <= 0:
        raise FitError("gamma moments require positive mean and variance")
    return float(m * m / v), float(m / v)


def _beta_mle(data: np.ndarray) -> tuple[float, float]:
    a, b, _, _ = sp_stats.beta.fit(data, floc=0.0, fscale=1.0)
    return float(a), float(b)


def _beta_moments(data: np.ndarray) -> tuple[float, float]:
    m = float(mean(data))
    v = float(variance(data))
    if not 0.0 < m < 1.0 or v <= 0 or v >= m * (1 - m):
        raise FitError("beta moments are outside the admissible region")
    common = m * (1 - m) / v - 1.0
    return float(m * common), float((1 - m) * common)


def _t_mle(data: np.ndarray) -> tuple[float]:
    df, _, _ = sp_stats.t.fit(data)
    return (float(max(df, 1e-3)),)


def _t_moments(data: np.ndarray) -> tuple[float]:
    # Var(t_df) = df / (df - 2) for df > 2, so df = 2v / (v - 1) on the
    # standardised scale.  Falls back to a heavy-tailed default when the
    # sample variance of the z-scores is at or below 1.
    z = (np.asarray(data, dtype=float) - mean(data)) / std_dev(data)
    v = float(np.var(z, ddof=1))
    if v <= 1.0:
        return (30.0,)
    return (float(np.clip(2.0 * v / (v - 1.0), 2.1, 200.0)),)


def _chi2_mle(data: np.ndarray) -> tuple[float]:
    df, _, _ = sp_stats.chi2.fit(data, floc=0.0, fscale=1.0)
    return (float(max(df, 1e-3)),)


def _chi2_moments(data: np.ndarray) -> tuple[float]:
    m = float(mean(data))
    if m <= 0:
        raise FitError("chi-squared mean must be positive")
    return (float(m),)


def _uniform_mle(data: np.ndarray) -> tuple[float, float]:
    lo, scale = sp_stats.uniform.fit(data)
    return float(lo), float(lo + scale)


def _uniform_moments(data: np.ndarray) -> tuple[float, float]:
    m = float(mean(data))
    s = float(std_dev(data))
    half = s * np.sqrt(3.0)
    return float(m - half), float(m + half)


def _f_mle(data: np.ndarray) -> tuple[float, float]:
    df1, df2, _, _ = sp_stats.f.fit(data, floc=0.0, fscale=1.0)
    return float(max(df1, 1e-3)), float(max(df2, 1e-3))


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

DISTRIBUTIONS: dict[str, DistributionSpec] = {
    "normal": DistributionSpec(
        key="normal",
        label="Normal",
        param_names=("mu", "sigma"),
        support="real",
        build=lambda mu, sigma: Normal(mu=mu, sigma=sigma),
        mle=_normal_mle,
        moments=_normal_moments,
        robust=_normal_robust,
        notes="The default reference. Robust mode fits the bulk with "
        "median/MAD so that outliers stay visible instead of bending the line.",
    ),
    "exponential": DistributionSpec(
        key="exponential",
        label="Exponential",
        param_names=("lam",),
        support="positive",
        build=lambda lam: Exponential(lam=lam),
        mle=_expon_mle,
        moments=_expon_moments,
        robust=_expon_robust,
        notes="Location is pinned at zero. Robust mode uses the median, "
        "since the exponential median is ln(2)/lambda.",
    ),
    "gamma": DistributionSpec(
        key="gamma",
        label="Gamma",
        param_names=("alpha", "beta"),
        support="positive",
        build=lambda alpha, beta: Gamma(alpha=alpha, beta=beta),
        mle=_gamma_mle,
        moments=_gamma_moments,
        notes="Rate parameterisation: beta is a rate, not a scale. "
        "Location pinned at zero.",
    ),
    "beta": DistributionSpec(
        key="beta",
        label="Beta",
        param_names=("alpha", "beta"),
        support="unit",
        build=lambda alpha, beta: Beta(alpha=alpha, beta=beta),
        mle=_beta_mle,
        moments=_beta_moments,
        notes="Support fixed to (0, 1); rescale your data first if it lives "
        "on another interval.",
    ),
    "t": DistributionSpec(
        key="t",
        label="Student t",
        param_names=("df",),
        support="real",
        build=lambda df: StudentT(df=df),
        mle=_t_mle,
        moments=_t_moments,
        notes="wedgestats StudentT carries no location or scale, so the data "
        "are standardised before plotting. Use it to ask how heavy the tails "
        "are, not where the centre is.",
    ),
    "chi2": DistributionSpec(
        key="chi2",
        label="Chi-squared",
        param_names=("df",),
        support="positive",
        build=lambda df: ChiSquared(df=df),
        mle=_chi2_mle,
        moments=_chi2_moments,
        notes="Scale pinned at one. The natural reference for squared "
        "residuals and for likelihood-ratio statistics.",
    ),
    "uniform": DistributionSpec(
        key="uniform",
        label="Uniform",
        param_names=("low", "high"),
        support="real",
        build=lambda low, high: ContinuousUniform(low=low, high=high),
        mle=_uniform_mle,
        moments=_uniform_moments,
        notes="A uniform Q-Q plot of p-values is the standard check that a "
        "test is calibrated under the null.",
    ),
    "f": DistributionSpec(
        key="f",
        label="F",
        param_names=("df1", "df2"),
        support="positive",
        build=lambda df1, df2: FDistribution(df1=df1, df2=df2),
        mle=_f_mle,
        notes="Scale pinned at one. Method of moments is not offered; the "
        "second moment does not exist for df2 <= 4.",
    ),
}


def distribution_keys() -> tuple[str, ...]:
    """Registry keys in display order."""
    return tuple(DISTRIBUTIONS)


@dataclass(frozen=True)
class FitResult:
    """A fitted reference distribution plus its provenance.

    Attributes
    ----------
    dist : ContinuousDistribution
        The live wedgestats distribution.
    spec : DistributionSpec
        Registry entry it came from.
    params : tuple[float, ...]
        Estimated parameters in ``spec.param_names`` order.
    method : str
        Estimator actually used, which may differ from the one requested.
    requested_method : str
        Estimator the caller asked for.
    notes : tuple[str, ...]
        Anything worth carrying into the figure caption.
    """

    dist: ContinuousDistribution
    spec: DistributionSpec
    params: tuple[float, ...]
    method: str
    requested_method: str
    notes: tuple[str, ...] = field(default=())

    @property
    def fell_back(self) -> bool:
        """Whether the estimator used differs from the one requested."""
        return self.method != self.requested_method

    def summary(self) -> str:
        """One-line description of the fitted model."""
        return f"{self.spec.label}({self.spec.describe_params(self.params)}) by {self.method}"

    def param_map(self) -> dict[str, float]:
        """Parameters as a name-keyed mapping."""
        return dict(zip(self.spec.param_names, self.params, strict=False))


def fit(
    data: np.ndarray,
    dist_key: str = "normal",
    method: str = "mle",
    manual_params: tuple[float, ...] | None = None,
) -> FitResult:
    """Fit a reference distribution to *data*.

    Parameters
    ----------
    data : np.ndarray
        Sample values; must contain at least three finite observations.
    dist_key : str
        Key into :data:`DISTRIBUTIONS`.
    method : str
        One of :data:`FIT_METHODS`.  When an estimator is not defined for the
        chosen distribution the call falls back to maximum likelihood and says
        so in :attr:`FitResult.notes`.
    manual_params : tuple[float, ...] or None
        Required when ``method == "manual"``.

    Returns
    -------
    FitResult

    Raises
    ------
    FitError
        If the data leave the distribution's support, or fitting fails.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        raise FitError("at least three finite observations are required")

    try:
        spec = DISTRIBUTIONS[dist_key]
    except KeyError:
        raise FitError(
            f"unknown distribution '{dist_key}'; choose from {', '.join(DISTRIBUTIONS)}"
        ) from None

    if method not in FIT_METHODS:
        raise FitError(f"unknown fit method '{method}'; choose from {', '.join(FIT_METHODS)}")

    notes: list[str] = []

    if method == "manual":
        if manual_params is None or len(manual_params) != len(spec.param_names):
            raise FitError(
                f"manual fitting needs {len(spec.param_names)} parameter(s): "
                f"{', '.join(spec.param_names)}"
            )
        params = tuple(float(p) for p in manual_params)
        used = "manual"
    else:
        spec.validate_support(arr)
        estimator = {"mle": spec.mle, "moments": spec.moments, "robust": spec.robust}[method]
        used = method
        if estimator is None:
            notes.append(
                f"{method} estimation is not defined for the {spec.label} "
                "distribution; used maximum likelihood instead"
            )
            estimator = spec.mle
            used = "mle"
        try:
            params = tuple(float(p) for p in estimator(arr))
        except FitError:
            raise
        except Exception as exc:
            raise FitError(f"{spec.label} {used} fit failed: {exc}") from exc

    try:
        dist = spec.make(params)
    except ValueError as exc:
        raise FitError(
            f"{spec.label} rejected the estimated parameters "
            f"({spec.describe_params(params)}): {exc}"
        ) from exc

    if spec.notes:
        notes.append(spec.notes)

    return FitResult(
        dist=dist,
        spec=spec,
        params=params,
        method=used,
        requested_method=method,
        notes=tuple(notes),
    )


def sample_skewness_hint(data: np.ndarray) -> str:
    """Suggest a reference distribution from the sample's shape.

    A small convenience the GUI uses to nudge the user toward a sensible
    starting point; it is a hint, never an automatic choice.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return "Too few points for a shape hint."
    g1 = float(skewness(arr, bias=False))
    positive = bool(np.all(arr > 0))
    if abs(g1) < 0.5:
        return f"Sample skewness {g1:+.2f}: roughly symmetric, try Normal or Student t."
    if g1 > 0 and positive:
        return f"Sample skewness {g1:+.2f}: right-skewed and positive, try Gamma or Exponential."
    if g1 > 0:
        return f"Sample skewness {g1:+.2f}: right-skewed, consider a Box-Cox transform."
    return f"Sample skewness {g1:+.2f}: left-skewed, consider reflecting the data first."
