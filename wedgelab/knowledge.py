"""A curated, citable knowledge base of statistical formulas.

Every entry pairs a piece of textbook knowledge with an *executable*
:class:`~wedgelab.formula.Formula`, so the workbench can load a canonical
result, let the user edit it, and still show where the original came from.
The citation travels with the formula; that is the point.

The base is deliberately opinionated about Q-Q plotting because that is the
worked example the toolkit ships with, but the structure is general.

Examples
--------
>>> kb = KNOWLEDGE
>>> entry = kb.get("pp_blom")
>>> entry.citation
'Blom, G. (1958). Statistical Estimates and Transformed Beta-Variables. Wiley, New York.'
>>> float(entry.formula.evaluate(i=1, n=10))
0.06097560975609756
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from wedgelab.formula import Formula, Parameter

__all__ = [
    "KnowledgeEntry",
    "KnowledgeBase",
    "KNOWLEDGE",
    "PLOTTING_POSITION_ALPHAS",
    "symmetric_plotting_position",
    "CATEGORY_ORDER",
]


# ---------------------------------------------------------------------------
# Entry container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgeEntry:
    """One citable piece of statistical knowledge.

    Attributes
    ----------
    key : str
        Stable identifier, used by the GUI and by exported scripts.
    category : str
        Grouping shown in the browser, e.g. ``"Plotting positions"``.
    name : str
        Short display name.
    summary : str
        One sentence saying what it computes.
    formula : Formula or None
        Executable form.  ``None`` for entries that describe a procedure
        rather than a closed-form expression (e.g. "maximum likelihood").
    citation : str
        Full literature reference.
    when_to_use : str
        Practical guidance: the sentence a statistician would actually say.
    notes : str
        Caveats, equivalences, and derivation remarks.
    tags : tuple[str, ...]
        Free-text search tags.
    """

    key: str
    category: str
    name: str
    summary: str
    citation: str
    formula: Formula | None = None
    when_to_use: str = ""
    notes: str = ""
    tags: tuple[str, ...] = ()

    @property
    def has_formula(self) -> bool:
        """Whether this entry carries an executable formula."""
        return self.formula is not None

    def searchable(self) -> str:
        """Lowercased blob used by :meth:`KnowledgeBase.search`."""
        parts = [
            self.key,
            self.name,
            self.category,
            self.summary,
            self.when_to_use,
            self.notes,
            self.citation,
            " ".join(self.tags),
        ]
        if self.formula is not None:
            parts.append(self.formula.expression)
        return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# Plotting positions
#
# The one-parameter symmetric family
#
#     p_i = (i - a) / (n + 1 - 2a),    i = 1 .. n
#
# contains almost every plotting position in common use.  Choosing ``a`` is
# the single most consequential and least-discussed decision behind a Q-Q
# plot, which makes it the ideal thing to put on a slider.
# ---------------------------------------------------------------------------

PLOTTING_POSITION_ALPHAS: dict[str, float] = {
    "weibull": 0.0,
    "beard": 0.31,
    "filliben": 0.3175,
    "tukey": 1.0 / 3.0,
    "blom": 0.375,
    "cunnane": 0.40,
    "gringorten": 0.44,
    "hazen": 0.5,
}

_SYMMETRIC_EXPR = "(i - a) / (n + 1 - 2*a)"
_SYMMETRIC_LATEX = r"p_i = \frac{i - a}{n + 1 - 2a}"


def symmetric_plotting_position(
    a: float = 0.375,
    *,
    name: str = "Symmetric family",
    citation: str = "",
    description: str = "",
) -> Formula:
    """Build the one-parameter symmetric plotting-position formula.

    Parameters
    ----------
    a : float
        The family parameter; ``0`` gives Weibull, ``0.375`` Blom,
        ``0.5`` Hazen.
    name, citation, description : str
        Metadata attached to the returned formula.

    Returns
    -------
    Formula
        With a single tunable parameter ``a`` bounded to ``[0, 0.5]``.
    """
    return Formula(
        name=name,
        expression=_SYMMETRIC_EXPR,
        lhs="p_i",
        description=description or "Cumulative probability assigned to the i-th order statistic",
        parameters=(
            Parameter(
                "a",
                default=float(a),
                lower=0.0,
                upper=0.5,
                step=0.005,
                description="Family parameter; 0=Weibull, 3/8=Blom, 1/2=Hazen",
            ),
        ),
        citation=citation,
        latex=_SYMMETRIC_LATEX,
    )


def _pp_entry(
    key: str,
    name: str,
    a: float,
    summary: str,
    citation: str,
    when_to_use: str,
    notes: str = "",
    tags: tuple[str, ...] = (),
) -> KnowledgeEntry:
    """Construct a plotting-position entry pinned at a given ``a``."""
    return KnowledgeEntry(
        key=key,
        category="Plotting positions",
        name=name,
        summary=summary,
        citation=citation,
        formula=symmetric_plotting_position(a, name=name, citation=citation, description=summary),
        when_to_use=when_to_use,
        notes=notes,
        tags=("plotting position", "order statistics", *tags),
    )


_PLOTTING_POSITIONS: list[KnowledgeEntry] = [
    _pp_entry(
        "pp_weibull",
        "Weibull  (a = 0)",
        0.0,
        "p_i = i / (n + 1); the mean of the i-th uniform order statistic.",
        "Weibull, W. (1939). A Statistical Theory of the Strength of Materials. "
        "Ingeniors Vetenskaps Akademien Handlingar 151, 1-45.",
        "Use when you want an unbiased estimate of the exceedance probability "
        "itself rather than of the quantile. Standard in hydrology.",
        notes="Exactly E[U_(i)] = i/(n+1) for uniform order statistics. Because "
        "quantile transformation is nonlinear, unbiased in probability does not "
        "imply unbiased in quantile.",
        tags=("weibull", "hydrology", "mean rank"),
    ),
    _pp_entry(
        "pp_beard",
        "Beard  (a = 0.31)",
        0.31,
        "p_i = (i - 0.31) / (n + 0.38); a median-rank approximation.",
        "Beard, L. R. (1943). Statistical Analysis in Hydrology. "
        "Transactions of the ASCE 108, 1110-1160.",
        "A hydrological median-rank approximation; largely superseded by "
        "Cunnane and by exact Beta median ranks.",
        tags=("beard", "median rank"),
    ),
    _pp_entry(
        "pp_filliben",
        "Filliben  (a = 0.3175)",
        0.3175,
        "p_i = (i - 0.3175) / (n + 0.365); the interior median-rank approximation.",
        "Filliben, J. J. (1975). The Probability Plot Correlation Coefficient Test "
        "for Normality. Technometrics 17(1), 111-117.",
        "Use with the probability plot correlation coefficient test, whose "
        "critical values were tabulated against exactly this rule.",
        notes="Filliben's full rule replaces the endpoints with "
        "m_1 = 1 - 0.5^(1/n) and m_n = 0.5^(1/n); the interior approximation is "
        "what the symmetric family reproduces. Use the 'pp_filliben_exact' entry "
        "for the endpoint-corrected version.",
        tags=("filliben", "ppcc", "median rank"),
    ),
    _pp_entry(
        "pp_tukey",
        "Tukey  (a = 1/3)",
        1.0 / 3.0,
        "p_i = (i - 1/3) / (n + 1/3).",
        "Tukey, J. W. (1962). The Future of Data Analysis. "
        "Annals of Mathematical Statistics 33(1), 1-67.",
        "A distribution-free compromise; close to the exact median rank for "
        "most n and easy to state.",
        tags=("tukey",),
    ),
    _pp_entry(
        "pp_blom",
        "Blom  (a = 3/8)",
        0.375,
        "p_i = (i - 3/8) / (n + 1/4); approximates E[Z_(i)] under normality.",
        "Blom, G. (1958). Statistical Estimates and Transformed Beta-Variables. "
        "Wiley, New York.",
        "The default choice for a normal Q-Q plot: it makes the plotted "
        "abscissae track the expected normal order statistics closely.",
        notes="Accurate to about 0.001 in E[Z_(i)] for n >= 5. This is the rule "
        "behind R's qqnorm for n <= 10 and Minitab's default.",
        tags=("blom", "normal", "default"),
    ),
    _pp_entry(
        "pp_cunnane",
        "Cunnane  (a = 0.40)",
        0.40,
        "p_i = (i - 0.4) / (n + 0.2); approximately quantile-unbiased.",
        "Cunnane, C. (1978). Unbiased Plotting Positions - A Review. "
        "Journal of Hydrology 37(3-4), 205-222.",
        "A good general-purpose compromise across distributions when you do "
        "not want to commit to normality.",
        tags=("cunnane", "quantile unbiased"),
    ),
    _pp_entry(
        "pp_gringorten",
        "Gringorten  (a = 0.44)",
        0.44,
        "p_i = (i - 0.44) / (n + 0.12); tuned for Gumbel probability paper.",
        "Gringorten, I. I. (1963). A Plotting Rule for Extreme Probability Paper. "
        "Journal of Geophysical Research 68(3), 813-814.",
        "Use for extreme-value (Gumbel) analysis, where it is close to "
        "quantile-unbiased.",
        tags=("gringorten", "gumbel", "extreme value"),
    ),
    _pp_entry(
        "pp_hazen",
        "Hazen  (a = 1/2)",
        0.5,
        "p_i = (i - 1/2) / n; the midpoint rule.",
        "Hazen, A. (1914). Storage to be Provided in Impounding Reservoirs for "
        "Municipal Water Supply. Transactions of the ASCE 77, 1539-1640.",
        "The oldest and simplest rule; still the right answer when you want "
        "the empirical CDF evaluated at cell midpoints.",
        notes="Equivalent to using the midpoint of each step of the empirical "
        "CDF. Symmetric about 0.5 by construction.",
        tags=("hazen", "midpoint"),
    ),
]

_PLOTTING_POSITIONS_SPECIAL: list[KnowledgeEntry] = [
    KnowledgeEntry(
        key="pp_median_exact",
        category="Plotting positions",
        name="Exact median rank (Beta)",
        summary="p_i = median of Beta(i, n - i + 1), the exact i-th uniform order statistic.",
        citation="David, H. A. and Nagaraja, H. N. (2003). Order Statistics, 3rd ed. "
        "Wiley, Hoboken. Section 2.2.",
        formula=Formula(
            name="Exact median rank",
            expression="beta_ppf(0.5, i, n - i + 1)",
            lhs="p_i",
            description="Median of the exact sampling distribution of U_(i)",
            citation="David and Nagaraja (2003), Order Statistics, 3rd ed.",
            latex=r"p_i = F^{-1}_{\mathrm{Beta}(i,\,n-i+1)}(0.5)",
        ),
        when_to_use="When you want the exact median rank instead of an "
        "approximation, and n is small enough that the difference matters.",
        notes="U_(i) ~ Beta(i, n - i + 1) exactly. All the approximations above "
        "are closed-form stand-ins for this quantity. Costs one Beta quantile "
        "evaluation per point, which is negligible below a few thousand points.",
        tags=("exact", "beta", "order statistics", "median rank"),
    ),
    KnowledgeEntry(
        key="pp_mean_exact",
        category="Plotting positions",
        name="Exact mean rank",
        summary="p_i = E[U_(i)] = i / (n + 1), exactly.",
        citation="David, H. A. and Nagaraja, H. N. (2003). Order Statistics, 3rd ed. "
        "Wiley, Hoboken.",
        formula=Formula(
            name="Exact mean rank",
            expression="i / (n + 1)",
            lhs="p_i",
            description="Mean of the exact sampling distribution of U_(i)",
            latex=r"p_i = \frac{i}{n+1}",
        ),
        when_to_use="Identical to Weibull; listed separately because the "
        "derivation, not the arithmetic, is what justifies it.",
        tags=("exact", "mean rank", "weibull"),
    ),
    KnowledgeEntry(
        key="pp_filliben_exact",
        category="Plotting positions",
        name="Filliben with endpoint correction",
        summary="Interior points use (i - 0.3175)/(n + 0.365); the extremes use "
        "1 - 0.5^(1/n) and 0.5^(1/n).",
        citation="Filliben, J. J. (1975). The Probability Plot Correlation "
        "Coefficient Test for Normality. Technometrics 17(1), 111-117.",
        formula=Formula(
            name="Filliben (endpoint corrected)",
            expression=(
                "where(i == 1, 1 - 0.5**(1/n), "
                "where(i == n, 0.5**(1/n), (i - 0.3175) / (n + 0.365)))"
            ),
            lhs="p_i",
            description="Median ranks with exact treatment of the two extremes",
            citation="Filliben (1975), Technometrics 17(1), 111-117.",
            latex=(
                r"p_i = \begin{cases} 1 - 0.5^{1/n} & i = 1 \\ "
                r"(i - 0.3175)/(n + 0.365) & 1 < i < n \\ "
                r"0.5^{1/n} & i = n \end{cases}"
            ),
        ),
        when_to_use="Use when the tails drive the conclusion and n is small; "
        "the endpoint values are exact medians, not approximations.",
        notes="A worked demonstration of a piecewise formula in the workbench: "
        "the 'where' function is available in the expression namespace.",
        tags=("filliben", "piecewise", "tails", "endpoint"),
    ),
]


# ---------------------------------------------------------------------------
# Confidence envelopes
# ---------------------------------------------------------------------------

_ENVELOPES: list[KnowledgeEntry] = [
    KnowledgeEntry(
        key="env_beta_exact",
        category="Confidence envelopes",
        name="Exact pointwise band (order statistics)",
        summary="Bounds are F^-1 of the alpha/2 and 1-alpha/2 quantiles of Beta(i, n-i+1).",
        citation="David, H. A. and Nagaraja, H. N. (2003). Order Statistics, 3rd ed. "
        "Wiley, Hoboken. Chapter 2.",
        formula=Formula(
            name="Exact pointwise band (lower)",
            expression="beta_ppf(alpha/2, i, n - i + 1)",
            lhs="p_lower",
            description="Lower probability bound for the i-th order statistic",
            parameters=(
                Parameter(
                    "alpha",
                    default=0.05,
                    lower=0.001,
                    upper=0.5,
                    step=0.001,
                    description="Pointwise significance level",
                ),
            ),
            citation="David and Nagaraja (2003), Order Statistics, 3rd ed.",
            latex=r"p^{-}_i = F^{-1}_{\mathrm{Beta}(i,\,n-i+1)}(\alpha/2)",
        ),
        when_to_use="The default for a publication figure: exact, no asymptotics, "
        "and correctly asymmetric in the tails.",
        notes="Pointwise, not simultaneous. Roughly alpha*n points are expected "
        "to fall outside it even under a perfect fit, so do not read a single "
        "excursion as evidence of misfit. Valid as drawn when the reference "
        "distribution is fully specified; when parameters are estimated from the "
        "same data the band is conservative (too wide).",
        tags=("exact", "beta", "pointwise", "band"),
    ),
    KnowledgeEntry(
        key="env_normal_se",
        category="Confidence envelopes",
        name="Asymptotic pointwise band (delta method)",
        summary="SE of the i-th order statistic is sqrt(p(1-p)/n) / f(F^-1(p)).",
        citation="Cramer, H. (1946). Mathematical Methods of Statistics. "
        "Princeton University Press. Section 28.5.",
        formula=Formula(
            name="Order statistic standard error",
            expression="sqrt(p * (1 - p) / n) / normal_pdf(normal_ppf(p))",
            lhs="se_i",
            description="Large-sample standard error of the i-th order statistic "
            "on the standardised scale",
            latex=r"\mathrm{se}_i = \frac{1}{f(F^{-1}(p_i))}\sqrt{\frac{p_i(1-p_i)}{n}}",
        ),
        when_to_use="When you need a cheap band and n is large; it is the "
        "standard textbook derivation and reproduces what most software draws.",
        notes="Breaks down in the tails, where the density in the denominator "
        "goes to zero and the normal approximation to the order statistic is "
        "poor. Prefer the exact Beta band for small n or heavy tails.",
        tags=("asymptotic", "delta method", "standard error"),
    ),
    KnowledgeEntry(
        key="env_ks_simultaneous",
        category="Confidence envelopes",
        name="Simultaneous band (Kolmogorov-Smirnov)",
        summary="Shift p_i by the critical sup-norm distance d(alpha, n) to bound "
        "the whole curve at once.",
        citation="Conover, W. J. (1999). Practical Nonparametric Statistics, 3rd ed. "
        "Wiley, New York. Section 6.1.",
        formula=Formula(
            name="KS band half-width (asymptotic)",
            expression="sqrt(-log(alpha/2) / (2*n))",
            lhs="d",
            description="Asymptotic Kolmogorov-Smirnov critical distance",
            parameters=(
                Parameter(
                    "alpha",
                    default=0.05,
                    lower=0.001,
                    upper=0.5,
                    step=0.001,
                    description="Simultaneous significance level",
                ),
            ),
            citation="Conover (1999), Practical Nonparametric Statistics, 3rd ed.",
            latex=r"d_{\alpha,n} = \sqrt{\frac{-\ln(\alpha/2)}{2n}}",
        ),
        when_to_use="When the claim is about the whole plot ('the data are "
        "consistent with normality') rather than about individual points. "
        "This is the band that supports that sentence.",
        notes="Noticeably wider than a pointwise band. The closed form above is "
        "the Dvoretzky-Kiefer-Wolfowitz asymptotic; the exact Kolmogorov "
        "distribution is used when available.",
        tags=("simultaneous", "kolmogorov", "smirnov", "dkw"),
    ),
    KnowledgeEntry(
        key="env_bootstrap",
        category="Confidence envelopes",
        name="Parametric bootstrap band",
        summary="Simulate B samples from the fitted model and take pointwise "
        "quantiles of the resulting order statistics.",
        citation="Efron, B. and Tibshirani, R. J. (1993). An Introduction to the "
        "Bootstrap. Chapman and Hall, New York.",
        formula=None,
        when_to_use="The honest band when parameters were estimated from the "
        "same data, because the simulation reproduces that estimation step and "
        "therefore the shrinkage it induces.",
        notes="Costs B refits. Unlike the Beta band it is not conservative "
        "under parameter estimation, which is exactly why it is worth the cost "
        "for a headline figure.",
        tags=("bootstrap", "simulation", "estimated parameters"),
    ),
]


# ---------------------------------------------------------------------------
# Goodness-of-fit statistics
# ---------------------------------------------------------------------------

_GOODNESS: list[KnowledgeEntry] = [
    KnowledgeEntry(
        key="gof_ppcc",
        category="Goodness of fit",
        name="Probability plot correlation coefficient",
        summary="r = corr(theoretical quantiles, ordered data); the Q-Q plot's own "
        "straightness, expressed as a number.",
        citation="Filliben, J. J. (1975). The Probability Plot Correlation "
        "Coefficient Test for Normality. Technometrics 17(1), 111-117.",
        formula=None,
        when_to_use="Report alongside the figure. It is the statistic that "
        "measures precisely what the reader is looking at.",
        notes="Computed with wedgestats.correlation on the plotted pairs. "
        "Sensitive to the plotting-position rule, so state which one you used.",
        tags=("filliben", "ppcc", "correlation", "r"),
    ),
    KnowledgeEntry(
        key="gof_shapiro",
        category="Goodness of fit",
        name="Shapiro-Wilk W",
        summary="Ratio of the best linear unbiased estimate of sigma to the usual "
        "sum of squares.",
        citation="Shapiro, S. S. and Wilk, M. B. (1965). An Analysis of Variance "
        "Test for Normality (Complete Samples). Biometrika 52(3-4), 591-611.",
        formula=None,
        when_to_use="The most powerful general omnibus test of normality for "
        "small to moderate n. Pair it with the plot; never replace the plot.",
        notes="At large n it rejects on deviations too small to matter. That is "
        "a property of the test, not a defect of the data.",
        tags=("shapiro", "wilk", "normality"),
    ),
    KnowledgeEntry(
        key="gof_anderson_darling",
        category="Goodness of fit",
        name="Anderson-Darling A-squared",
        summary="Weighted quadratic distance between empirical and hypothesised "
        "CDF, weighted to emphasise the tails.",
        citation="Anderson, T. W. and Darling, D. A. (1954). A Test of Goodness of "
        "Fit. Journal of the American Statistical Association 49(268), 765-769.",
        formula=None,
        when_to_use="When tail behaviour is what matters - risk models, extreme "
        "value work, anything where the far quantiles carry the decision.",
        tags=("anderson", "darling", "tails"),
    ),
    KnowledgeEntry(
        key="gof_ks",
        category="Goodness of fit",
        name="Kolmogorov-Smirnov D",
        summary="Supremum distance between empirical and hypothesised CDF.",
        citation="Kolmogorov, A. N. (1933). Sulla determinazione empirica di una "
        "legge di distribuzione. Giornale dell'Istituto Italiano degli Attuari 4, 83-91.",
        formula=None,
        when_to_use="When you want the statistic that matches the simultaneous "
        "band you drew. Weak in the tails.",
        notes="Its null distribution assumes fully specified parameters. With "
        "estimated parameters use the Lilliefors correction.",
        tags=("kolmogorov", "smirnov", "sup norm"),
    ),
    KnowledgeEntry(
        key="gof_lilliefors",
        category="Goodness of fit",
        name="Lilliefors correction",
        summary="KS statistic with a null distribution recomputed for estimated "
        "mean and variance.",
        citation="Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov Test for "
        "Normality with Mean and Variance Unknown. Journal of the American "
        "Statistical Association 62(318), 399-402.",
        formula=None,
        when_to_use="Whenever you fit mu and sigma from the same data and still "
        "want a KS-style p-value.",
        tags=("lilliefors", "estimated parameters"),
    ),
]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

_FITTING: list[KnowledgeEntry] = [
    KnowledgeEntry(
        key="fit_mle",
        category="Estimation",
        name="Maximum likelihood",
        summary="Choose the parameters that maximise the likelihood of the sample.",
        citation="Fisher, R. A. (1922). On the Mathematical Foundations of "
        "Theoretical Statistics. Philosophical Transactions of the Royal Society A "
        "222, 309-368.",
        formula=None,
        when_to_use="The default. Efficient when the model is right.",
        notes="Not robust: a single outlier moves the fit, which then bends the "
        "whole reference line and can hide the outlier that caused it.",
        tags=("mle", "likelihood", "fisher"),
    ),
    KnowledgeEntry(
        key="fit_moments",
        category="Estimation",
        name="Method of moments",
        summary="Match sample moments to population moments.",
        citation="Pearson, K. (1894). Contributions to the Mathematical Theory of "
        "Evolution. Philosophical Transactions of the Royal Society A 185, 71-110.",
        formula=Formula(
            name="Sample standard deviation (unbiased variance)",
            expression="sqrt(sum((x - mean(x))**2) / (n - 1))",
            lhs="s",
            description="Bessel-corrected sample standard deviation",
            latex=r"s = \sqrt{\frac{1}{n-1}\sum_i (x_i - \bar{x})^2}",
        ),
        when_to_use="When the likelihood is awkward, or when you want an "
        "estimator that is trivially reproducible by hand.",
        notes="For the normal distribution this coincides with MLE up to the "
        "Bessel correction on the variance.",
        tags=("moments", "pearson"),
    ),
    KnowledgeEntry(
        key="fit_robust",
        category="Estimation",
        name="Robust location and scale (median / MAD)",
        summary="mu = median(x); sigma = 1.4826 * median(|x - median(x)|).",
        citation="Rousseeuw, P. J. and Croux, C. (1993). Alternatives to the "
        "Median Absolute Deviation. Journal of the American Statistical "
        "Association 88(424), 1273-1283.",
        formula=Formula(
            name="MAD scale estimate",
            expression="k * median(abs(x - median(x)))",
            lhs="sigma_hat",
            description="Consistency-corrected median absolute deviation",
            parameters=(
                Parameter(
                    "k",
                    default=1.4826,
                    lower=1.0,
                    upper=2.0,
                    step=0.0001,
                    description="Consistency constant; 1/Phi^-1(3/4) = 1.4826 for normal",
                ),
            ),
            citation="Rousseeuw and Croux (1993), JASA 88(424), 1273-1283.",
            latex=r"\hat{\sigma} = k \cdot \mathrm{median}_i |x_i - \mathrm{median}(x)|",
        ),
        when_to_use="Fit the bulk, then let the outliers show themselves. This "
        "is the right default when the Q-Q plot's job is outlier detection.",
        notes="The constant 1.4826 = 1/Phi^-1(0.75) makes the MAD consistent for "
        "sigma under normality; it is wrong for other distributions. Put it on a "
        "slider and see how the tails respond.",
        tags=("robust", "mad", "median", "outliers"),
    ),
    KnowledgeEntry(
        key="fit_lmoments",
        category="Estimation",
        name="L-moments",
        summary="Linear combinations of order statistics used as moment analogues.",
        citation="Hosking, J. R. M. (1990). L-moments: Analysis and Estimation of "
        "Distributions Using Linear Combinations of Order Statistics. Journal of "
        "the Royal Statistical Society B 52(1), 105-124.",
        formula=None,
        when_to_use="Heavy-tailed or small samples, where conventional moments "
        "have enormous sampling variance.",
        tags=("lmoments", "hosking", "heavy tails"),
    ),
]


# ---------------------------------------------------------------------------
# Reference lines and transformations
# ---------------------------------------------------------------------------

_LINES: list[KnowledgeEntry] = [
    KnowledgeEntry(
        key="line_ols",
        category="Reference lines",
        name="Least-squares line through all points",
        summary="Fit y = b0 + b1 * q by ordinary least squares on the plotted pairs.",
        citation="Draper, N. R. and Smith, H. (1998). Applied Regression Analysis, "
        "3rd ed. Wiley, New York.",
        formula=None,
        when_to_use="When you want the line the eye is already fitting. Computed "
        "with wedgestats.simple_ols so the slope and intercept come with "
        "standard errors.",
        notes="Outliers pull it, by construction. That is a feature when you are "
        "reporting fit and a bug when you are hunting outliers.",
        tags=("ols", "regression", "line"),
    ),
    KnowledgeEntry(
        key="line_quartile",
        category="Reference lines",
        name="Quartile line",
        summary="Line through the first and third quartile pairs.",
        citation="Chambers, J. M., Cleveland, W. S., Kleiner, B. and Tukey, P. A. "
        "(1983). Graphical Methods for Data Analysis. Wadsworth, Belmont.",
        formula=None,
        when_to_use="The R qqline default. Resistant to tail behaviour, so tail "
        "departures stay visible instead of being absorbed into the line.",
        tags=("quartile", "qqline", "robust", "r"),
    ),
    KnowledgeEntry(
        key="line_theoretical",
        category="Reference lines",
        name="Theoretical line from the fit",
        summary="Slope sigma-hat, intercept mu-hat, taken from the fitted model.",
        citation="Thode, H. C. (2002). Testing for Normality. Marcel Dekker, New York.",
        formula=None,
        when_to_use="When the question is whether the data match a *specified* "
        "model, not merely whether they are straight.",
        notes="The only line of the three that can be wrong in a useful way: if "
        "the points are straight but off this line, location or scale is off.",
        tags=("theoretical", "identity"),
    ),
]

_TRANSFORMS: list[KnowledgeEntry] = [
    KnowledgeEntry(
        key="tf_zscore",
        category="Transformations",
        name="Standardisation",
        summary="z = (x - mu) / sigma.",
        citation="Standard; see e.g. Casella, G. and Berger, R. L. (2002). "
        "Statistical Inference, 2nd ed. Duxbury, Pacific Grove.",
        formula=Formula(
            name="z-score",
            expression="(x - mu) / sigma",
            lhs="z",
            description="Centre and scale",
            latex=r"z = \frac{x - \mu}{\sigma}",
        ),
        when_to_use="Put the plot on standard-normal axes so several panels can "
        "share one scale.",
        tags=("zscore", "standardise"),
    ),
    KnowledgeEntry(
        key="tf_boxcox",
        category="Transformations",
        name="Box-Cox",
        summary="y = (x^lambda - 1)/lambda for lambda != 0, log(x) at lambda = 0.",
        citation="Box, G. E. P. and Cox, D. R. (1964). An Analysis of "
        "Transformations. Journal of the Royal Statistical Society B 26(2), 211-252.",
        formula=Formula(
            name="Box-Cox transform",
            expression="where(abs(lam) < 1e-9, log(x), (x**lam - 1) / where(abs(lam) < 1e-9, 1, lam))",
            lhs="y",
            description="Power transform toward normality; requires x > 0",
            parameters=(
                Parameter(
                    "lam",
                    default=1.0,
                    lower=-2.0,
                    upper=3.0,
                    step=0.01,
                    description="Power parameter; 0 is the log transform",
                ),
            ),
            citation="Box and Cox (1964), JRSS B 26(2), 211-252.",
            latex=r"y = \begin{cases} (x^{\lambda}-1)/\lambda & \lambda \neq 0 \\ "
            r"\ln x & \lambda = 0 \end{cases}",
        ),
        when_to_use="Drag lambda and watch the Q-Q plot straighten. This is the "
        "clearest demonstration in the toolkit of a formula parameter doing "
        "visible statistical work.",
        notes="Requires strictly positive data. The guard on the denominator "
        "keeps the expression finite at lambda = 0, where the limit is log(x).",
        tags=("boxcox", "power transform", "normality"),
    ),
    KnowledgeEntry(
        key="tf_detrend",
        category="Transformations",
        name="Detrended Q-Q (Tukey mean-difference)",
        summary="Plot the residual from the reference line instead of the raw quantile.",
        citation="Thode, H. C. (2002). Testing for Normality. Marcel Dekker, "
        "New York. Section 2.3.",
        formula=Formula(
            name="Detrended residual",
            expression="x - (b0 + b1 * q)",
            lhs="r_i",
            description="Vertical departure from the reference line",
            latex=r"r_i = x_{(i)} - (\hat{\beta}_0 + \hat{\beta}_1 q_i)",
        ),
        when_to_use="When the departure is small relative to the range. "
        "Removing the dominant linear trend rescales the vertical axis and makes "
        "curvature legible.",
        notes="Same information, better resolution. Always label it as detrended; "
        "a reader who assumes a raw Q-Q plot will misread the axis.",
        tags=("detrended", "residual", "tukey"),
    ),
]


# ---------------------------------------------------------------------------
# Interpretation guidance (no formulas, but the knowledge that matters most)
# ---------------------------------------------------------------------------

_INTERPRETATION: list[KnowledgeEntry] = [
    KnowledgeEntry(
        key="read_sshape",
        category="Reading the plot",
        name="S-shape means the tails disagree",
        summary="Points below the line at the left and above it at the right "
        "indicate heavier tails than the reference; the mirror image indicates "
        "lighter tails.",
        citation="Chambers, J. M., Cleveland, W. S., Kleiner, B. and Tukey, P. A. "
        "(1983). Graphical Methods for Data Analysis. Wadsworth, Belmont. Chapter 6.",
        formula=None,
        when_to_use="First thing to check on any Q-Q plot.",
        tags=("interpretation", "tails", "kurtosis"),
    ),
    KnowledgeEntry(
        key="read_curve",
        category="Reading the plot",
        name="Monotone curvature means skew",
        summary="A single convex or concave arc indicates the sample is skewed "
        "relative to the reference distribution.",
        citation="Thode, H. C. (2002). Testing for Normality. Marcel Dekker, New York.",
        formula=None,
        when_to_use="Distinguish skew (one arc) from tail weight (S-shape) "
        "before reaching for a transformation.",
        tags=("interpretation", "skew"),
    ),
    KnowledgeEntry(
        key="read_granularity",
        category="Reading the plot",
        name="Horizontal runs mean discreteness",
        summary="Flat steps indicate ties or a rounded measurement scale, not "
        "distributional misfit.",
        citation="Cleveland, W. S. (1993). Visualizing Data. Hobart Press, Summit.",
        formula=None,
        when_to_use="Before diagnosing non-normality on a coarsely recorded "
        "variable. Check the measurement resolution first.",
        tags=("interpretation", "ties", "discreteness"),
    ),
    KnowledgeEntry(
        key="read_pointwise",
        category="Reading the plot",
        name="A pointwise band is not a simultaneous band",
        summary="Under a correct model roughly alpha*n of n points fall outside a "
        "pointwise band by construction.",
        citation="Conover, W. J. (1999). Practical Nonparametric Statistics, 3rd ed. "
        "Wiley, New York.",
        formula=None,
        when_to_use="Before writing 'several points lie outside the confidence "
        "band, so the data are not normal' in a caption.",
        notes="With n = 100 and alpha = 0.05, five excursions are the "
        "expectation, not the exception. Use a simultaneous band if the claim "
        "is about the plot as a whole.",
        tags=("interpretation", "band", "multiplicity"),
    ),
    KnowledgeEntry(
        key="read_smalln",
        category="Reading the plot",
        name="Small samples look non-normal even when they are not",
        summary="At n below roughly 30 the sampling variability of order "
        "statistics dominates the visual impression.",
        citation="Thode, H. C. (2002). Testing for Normality. Marcel Dekker, New York.",
        formula=None,
        when_to_use="Simulate from the fitted model at the same n and compare. "
        "The toolkit's bootstrap band does exactly this.",
        tags=("interpretation", "small sample", "power"),
    ),
]


CATEGORY_ORDER: tuple[str, ...] = (
    "Plotting positions",
    "Confidence envelopes",
    "Reference lines",
    "Estimation",
    "Transformations",
    "Goodness of fit",
    "Reading the plot",
)


# ---------------------------------------------------------------------------
# The base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgeBase:
    """An immutable, searchable collection of :class:`KnowledgeEntry`."""

    entries: tuple[KnowledgeEntry, ...] = field(default=())

    def __post_init__(self) -> None:
        keys = [e.key for e in self.entries]
        if len(set(keys)) != len(keys):
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            raise ValueError(f"duplicate knowledge keys: {dupes}")

    def __iter__(self) -> Iterator[KnowledgeEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, key: str) -> KnowledgeEntry:
        """Return the entry with the given key.

        Raises
        ------
        KeyError
            If no entry has that key.
        """
        for entry in self.entries:
            if entry.key == key:
                return entry
        raise KeyError(f"no knowledge entry with key '{key}'")

    def categories(self) -> tuple[str, ...]:
        """Categories present, in the canonical display order."""
        present = {e.category for e in self.entries}
        ordered = [c for c in CATEGORY_ORDER if c in present]
        ordered.extend(sorted(present - set(CATEGORY_ORDER)))
        return tuple(ordered)

    def by_category(self, category: str) -> tuple[KnowledgeEntry, ...]:
        """All entries in *category*, in insertion order."""
        return tuple(e for e in self.entries if e.category == category)

    def search(self, text: str) -> tuple[KnowledgeEntry, ...]:
        """Entries matching every whitespace-separated term in *text*."""
        terms = [t for t in text.lower().split() if t]
        if not terms:
            return self.entries
        return tuple(
            e for e in self.entries if all(t in e.searchable() for t in terms)
        )

    def with_formulas(self) -> tuple[KnowledgeEntry, ...]:
        """Only the entries that carry an executable formula."""
        return tuple(e for e in self.entries if e.has_formula)


KNOWLEDGE = KnowledgeBase(
    entries=tuple(
        _PLOTTING_POSITIONS
        + _PLOTTING_POSITIONS_SPECIAL
        + _ENVELOPES
        + _LINES
        + _FITTING
        + _TRANSFORMS
        + _GOODNESS
        + _INTERPRETATION
    )
)
