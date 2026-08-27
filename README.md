# wedgestats

A comprehensive, pure-Python statistical library covering distributions, descriptive statistics, hypothesis testing, and regression analysis. Built on top of NumPy and SciPy with a clean, object-oriented API.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import wedgestats as ws

# Distributions
n = ws.Normal(mu=100, sigma=15)
n.pdf(100)       # 0.0266
n.cdf(115)       # 0.8413
n.ppf(0.975)     # 129.39...
n.mean()         # 100.0

b = ws.Binomial(n=20, p=0.3)
b.pmf(6)         # P(X = 6)
b.cdf(8)         # P(X <= 8)

# Descriptive statistics
data = [2, 4, 4, 4, 5, 5, 7, 9]
ws.mean(data)           # 5.0
ws.median(data)         # 4.5
ws.std_dev(data)        # 2.138...
ws.describe(data)       # full summary

# Correlation
ws.correlation(x, y, method="pearson")
ws.correlation(x, y, method="spearman")

# Hypothesis testing
result = ws.one_sample_t(data, mu0=4.0)
result.statistic   # t-statistic
result.p_value     # p-value
result.reject      # True/False at alpha=0.05

# Confidence intervals
ci = ws.ci_mean(data, confidence=0.95)
ci.lower, ci.upper

# ANOVA
ws.one_way(group1, group2, group3)

# Regression
result = ws.simple_ols(x, y)
result.coefficients   # [intercept, slope]
result.r_squared      # R^2
```

## API Overview

### Distributions

14 distribution classes with a unified API (`pmf`/`pdf`, `cdf`, `sf`, `mean()`, `variance()`, `std_dev()`, `rvs()`):

| Discrete | Continuous |
|---|---|
| `Binomial(n, p)` | `Normal(mu, sigma)` |
| `Hypergeometric(M, n, N)` | `Exponential(lam)` |
| `Poisson(lam)` | `ChiSquared(df)` |
| `Geometric(p)` | `StudentT(df)` |
| `NegativeBinomial(r, p)` | `FDistribution(df1, df2)` |
| `DiscreteUniform(low, high)` | `Beta(alpha, beta)` |
| | `Gamma(alpha, beta)` |
| | `ContinuousUniform(low, high)` |

### Descriptive Statistics

`mean`, `median`, `mode`, `trimmed_mean`, `variance`, `std_dev`, `data_range`, `iqr`, `mad`, `cv`, `skewness`, `kurtosis`, `percentile`, `quartiles`, `z_score`, `correlation`, `covariance`, `five_number_summary`, `describe`

### Hypothesis Testing

- **Z-tests:** `one_sample_z`, `two_sample_z`, `proportion_z`
- **T-tests:** `one_sample_t`, `two_sample_t` (Student's & Welch's), `paired_t`
- **Chi-squared:** `goodness_of_fit`, `independence`, `homogeneity`
- **ANOVA:** `one_way`, `tukey_hsd`
- **Nonparametric:** `mann_whitney`, `wilcoxon`, `kruskal_wallis`
- **Confidence intervals:** `ci_mean`, `ci_proportion`, `ci_difference_of_means`
- **Power analysis:** `cohens_d`, `z_test_power`, `sample_size_z`

All tests return a frozen `TestResult(statistic, p_value, test_name, reject, alpha)`.

### Regression

- `simple_ols(x, y)` - Simple linear regression
- `multiple_ols(X, y)` - Multiple regression
- `logistic_regression(X, y)` - Binary logistic regression (IRLS)
- **Diagnostics:** `residuals`, `standardized_residuals`, `vif`, `cooks_distance`, `durbin_watson`

## wedgelab - the interactive workbench

`wedgelab` is a standalone application built on top of `wedgestats`. It exists
to make the choices inside a statistical figure visible, editable, and citable,
using the publication-quality Q-Q plot as its worked example.

```bash
python sandbox.py          # open the workbench
python sandbox.py demo     # run the seven-stage Q-Q walkthrough
python -m wedgelab         # same as the first, once installed
```

### Why it exists

A Q-Q plot is not one calculation but five, and each is a choice you can
defend or get wrong:

| Decision | What varies | Where it lives |
|---|---|---|
| Plotting positions | `p_i = (i - a)/(n + 1 - 2a)` and its relatives | editable formula |
| Reference model | distribution and estimator (MLE, moments, robust, manual) | `wedgelab.models` |
| Reference line | least squares, quartile, theoretical identity | `wedgelab.qq` |
| Confidence envelope | exact Beta, asymptotic, simultaneous, bootstrap | `wedgelab.qq` |
| Presentation | raw, standardised, detrended; journal theme | `wedgelab.plot` |

Most software fixes all five and shows you the answer. The workbench puts each
one on a control and shows you what it costs.

### The three ideas

**Formulas are objects.** A `Formula` is plain-text mathematics with declared
parameters, evaluated through an AST whitelist rather than `eval`. Attributes,
indexing, comprehensions, lambdas and imports are all rejected, so an
expression typed into the GUI cannot reach outside the mathematical namespace.
Editing a formula returns a new one that remembers where it came from.

```python
import wedgelab as wl

blom = wl.KNOWLEDGE.get("pp_blom").formula
blom.evaluate(i=1, n=10)              # 0.0609...
blom.derivative("a").pretty()          # symbolic, via SymPy
blom.edit("beta_ppf(0.5, i, n - i + 1)", name="Exact median rank")
```

Anything you type that is not `i` or `n` becomes a slider, so an edit turns a
fixed rule into a family you can explore.

**Knowledge is citable and executable.** Thirty-five entries pair a piece of
textbook statistics with a runnable formula and a full literature reference --
Blom (1958), Filliben (1975), Cunnane (1978), Gringorten (1963), Hosking
(1990), Rousseeuw and Croux (1993), and the rest. Loading a rule into the
workbench brings its provenance with it, and the tests check the constants
against the papers.

```python
wl.KNOWLEDGE.search("tails")           # entries about tail behaviour
wl.KNOWLEDGE.get("env_beta_exact").citation
```

**A figure is a computation.** `to_script` writes a standalone Python file that
rebuilds the figure exactly, plotting-position formula included. Put it in the
supplementary material and the figure stops being a claim.

```python
result = wl.compute(wl.QQSpec(data=data, dist_key="normal", label="assay"))
fig = wl.render(result, "nature")                    # exactly 89 mm wide
wl.save_figure(fig, "figure3", formats=("pdf", "svg"))
print(result.caption())                              # states every choice made
```

### Publication themes

Ten presets fix the *physical* width in millimetres to the target journal's
column, so the exported file is placed at 1:1 and the type stays the size it
was set: Nature (89 / 183 mm), Science (55 mm), PLOS (132 mm), IEEE (88.9 mm),
Elsevier (90 mm), APA, thesis, presentation, and a screen theme for working.
Vector output keeps text as text (`pdf.fonttype 42`, `svg.fonttype none`), as
most journals require. Check the current author guidelines before submitting;
column widths do change.

### What it reuses from wedgestats

Not decoration -- load-bearing:

- `Normal`, `Gamma`, `Beta`, `StudentT`, `ChiSquared`, `Exponential`,
  `ContinuousUniform`, `FDistribution` supply every theoretical quantile
  through `ppf`, and generate every synthetic sample through `rvs`
- `Beta(i, n - i + 1)` **is** the exact confidence envelope: the i-th uniform
  order statistic's own distribution
- `simple_ols` fits the reference line, with standard errors
- `correlation` computes Filliben's probability plot correlation coefficient
- `describe`, `quartiles`, `mad`, `median`, `skewness` back the diagnostics
  and the robust estimator

### Honesty features

The engine reports what it knows to be shaky rather than letting a number
stand unqualified:

- the exact Beta envelope is flagged as conservative when parameters were
  estimated from the same data
- the Kolmogorov-Smirnov p-value is flagged as invalid under estimated
  parameters, pointing at the Lilliefors correction
- a simultaneous band that is genuinely unbounded is reported as unbounded,
  not clipped to a plausible-looking finite number
- the probability plot correlation coefficient is measured before detrending,
  because detrended residuals are orthogonal to the theoretical quantiles by
  construction and would report exactly zero

## Running Tests

```bash
pytest                        # 757 tests
pytest tests/test_wedgelab    # the workbench alone
```

The workbench tests include a GUI integration suite that builds the real Tk
window and drives the panel callbacks; it skips automatically when no display
is available.

## License

MIT
