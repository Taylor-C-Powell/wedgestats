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

## Running Tests

```bash
pytest
```

## License

MIT
