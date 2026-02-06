"""Descriptive statistics for wedgestats."""

from wedgestats.descriptive.association import correlation, covariance
from wedgestats.descriptive.central_tendency import mean, median, mode, trimmed_mean
from wedgestats.descriptive.dispersion import cv, data_range, iqr, mad, std_dev, variance
from wedgestats.descriptive.position import percentile, quartiles, z_score
from wedgestats.descriptive.shape import kurtosis, skewness
from wedgestats.descriptive.summary import DescribeResult, describe, five_number_summary

__all__ = [
    "correlation",
    "covariance",
    "cv",
    "data_range",
    "describe",
    "DescribeResult",
    "five_number_summary",
    "iqr",
    "kurtosis",
    "mad",
    "mean",
    "median",
    "mode",
    "percentile",
    "quartiles",
    "skewness",
    "std_dev",
    "trimmed_mean",
    "variance",
    "z_score",
]
