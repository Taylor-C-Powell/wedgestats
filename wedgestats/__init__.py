"""wedgestats - A comprehensive statistical library."""

__version__ = "0.1.0"

# Core math
from wedgestats._math import combination, factorial, permutation

# Distributions
from wedgestats.distributions import (
    Beta,
    Binomial,
    ChiSquared,
    ContinuousDistribution,
    ContinuousUniform,
    DiscreteDistribution,
    DiscreteUniform,
    Distribution,
    Exponential,
    FDistribution,
    Gamma,
    Geometric,
    Hypergeometric,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
)

# Descriptive statistics
from wedgestats.descriptive import (
    DescribeResult,
    correlation,
    covariance,
    cv,
    data_range,
    describe,
    five_number_summary,
    iqr,
    kurtosis,
    mad,
    mean,
    median,
    mode,
    percentile,
    quartiles,
    skewness,
    std_dev,
    trimmed_mean,
    variance,
    z_score,
)

# Hypothesis testing
from wedgestats.hypothesis import (
    ConfidenceInterval,
    TestResult,
    TukeyResult,
    ci_difference_of_means,
    ci_mean,
    ci_proportion,
    cohens_d,
    goodness_of_fit,
    homogeneity,
    independence,
    kruskal_wallis,
    mann_whitney,
    one_sample_t,
    one_sample_z,
    one_way,
    paired_t,
    proportion_z,
    sample_size_z,
    tukey_hsd,
    two_sample_t,
    two_sample_z,
    wilcoxon,
    z_test_power,
)

# Regression
from wedgestats.regression import (
    LogisticResult,
    RegressionResult,
    cooks_distance,
    durbin_watson,
    logistic_regression,
    multiple_ols,
    residuals,
    simple_ols,
    standardized_residuals,
    vif,
)
