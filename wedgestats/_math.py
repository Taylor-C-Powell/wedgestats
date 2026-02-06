"""Core mathematical functions for wedgestats."""

import math


def factorial(n: int) -> int:
    """
    Compute the factorial of a non-negative integer.

    Parameters
    ----------
    n : int
        A non-negative integer.

    Returns
    -------
    int
        The factorial n!.

    Raises
    ------
    ValueError
        If *n* is negative.
    TypeError
        If *n* is not an integer.
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 0:
        raise ValueError("n must be non-negative")
    return math.factorial(n)


def combination(n: int, r: int) -> int:
    """
    Compute the number of combinations C(n, r) = n! / (r! (n - r)!).

    Parameters
    ----------
    n : int
        Total number of items.
    r : int
        Number of items to choose.

    Returns
    -------
    int
        The binomial coefficient C(n, r).

    Raises
    ------
    ValueError
        If *r* < 0 or *r* > *n*.
    """
    if r < 0 or r > n:
        raise ValueError(f"r must satisfy 0 <= r <= n, got r={r}, n={n}")
    return math.comb(n, r)


def permutation(n: int, r: int) -> int:
    """
    Compute the number of permutations P(n, r) = n! / (n - r)!.

    Parameters
    ----------
    n : int
        Total number of items.
    r : int
        Number of items to arrange.

    Returns
    -------
    int
        The number of permutations P(n, r).

    Raises
    ------
    ValueError
        If *r* < 0 or *r* > *n*.
    """
    if r < 0 or r > n:
        raise ValueError(f"r must satisfy 0 <= r <= n, got r={r}, n={n}")
    return math.perm(n, r)
