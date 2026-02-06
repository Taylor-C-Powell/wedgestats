"""Immutable result container for hypothesis tests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TestResult:
    """
    Immutable result of a hypothesis test.

    Attributes
    ----------
    statistic : float
        Test statistic value.
    p_value : float
        p-value of the test.
    test_name : str
        Human-readable name of the test performed.
    reject : bool
        Whether to reject H0 at the given alpha.
    alpha : float
        Significance level used.
    """

    statistic: float
    p_value: float
    test_name: str
    reject: bool
    alpha: float = 0.05
