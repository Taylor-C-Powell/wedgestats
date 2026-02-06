"""Hypothesis testing for wedgestats."""

from wedgestats.hypothesis._result import TestResult
from wedgestats.hypothesis.anova import TukeyResult, one_way, tukey_hsd
from wedgestats.hypothesis.chi_squared import goodness_of_fit, homogeneity, independence
from wedgestats.hypothesis.confidence import (
    ConfidenceInterval,
    ci_difference_of_means,
    ci_mean,
    ci_proportion,
)
from wedgestats.hypothesis.nonparametric import kruskal_wallis, mann_whitney, wilcoxon
from wedgestats.hypothesis.power import cohens_d, sample_size_z, z_test_power
from wedgestats.hypothesis.t_test import one_sample_t, paired_t, two_sample_t
from wedgestats.hypothesis.z_test import one_sample_z, proportion_z, two_sample_z

__all__ = [
    "ConfidenceInterval",
    "TestResult",
    "TukeyResult",
    "ci_difference_of_means",
    "ci_mean",
    "ci_proportion",
    "cohens_d",
    "goodness_of_fit",
    "homogeneity",
    "independence",
    "kruskal_wallis",
    "mann_whitney",
    "one_sample_t",
    "one_sample_z",
    "one_way",
    "paired_t",
    "proportion_z",
    "sample_size_z",
    "tukey_hsd",
    "two_sample_t",
    "two_sample_z",
    "wilcoxon",
    "z_test_power",
]
