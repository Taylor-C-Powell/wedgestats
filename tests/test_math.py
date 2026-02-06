"""Tests for wedgestats._math."""

import math

import pytest

from wedgestats._math import combination, factorial, permutation


class TestFactorial:
    def test_zero(self):
        assert factorial(0) == 1

    def test_one(self):
        assert factorial(1) == 1

    def test_small(self):
        assert factorial(5) == 120

    def test_large(self):
        assert factorial(20) == math.factorial(20)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            factorial(-1)

    def test_float_raises(self):
        with pytest.raises(TypeError):
            factorial(3.5)


class TestCombination:
    def test_basic(self):
        assert combination(5, 2) == 10

    def test_zero_choose(self):
        assert combination(5, 0) == 1

    def test_n_choose_n(self):
        assert combination(5, 5) == 1

    def test_matches_math_comb(self):
        assert combination(20, 7) == math.comb(20, 7)

    def test_r_greater_than_n_raises(self):
        with pytest.raises(ValueError):
            combination(3, 5)

    def test_negative_r_raises(self):
        with pytest.raises(ValueError):
            combination(5, -1)


class TestPermutation:
    def test_basic(self):
        assert permutation(5, 2) == 20

    def test_zero_r(self):
        assert permutation(5, 0) == 1

    def test_n_equals_r(self):
        assert permutation(5, 5) == 120

    def test_matches_math_perm(self):
        assert permutation(10, 4) == math.perm(10, 4)

    def test_r_greater_than_n_raises(self):
        with pytest.raises(ValueError):
            permutation(3, 5)

    def test_negative_r_raises(self):
        with pytest.raises(ValueError):
            permutation(5, -1)
