"""Tests for power analysis."""

import pytest

from wedgestats.hypothesis.power import cohens_d, sample_size_z, z_test_power


class TestCohensD:
    def test_basic(self):
        d = cohens_d(105, 100, 15)
        assert d == pytest.approx(5 / 15)

    def test_zero_sd_raises(self):
        with pytest.raises(ValueError):
            cohens_d(100, 100, 0)


class TestZTestPower:
    def test_large_effect_high_power(self):
        power = z_test_power(effect_size=0.8, n=100)
        assert power > 0.99

    def test_small_effect_low_power(self):
        power = z_test_power(effect_size=0.2, n=10)
        assert power < 0.5

    def test_one_sided(self):
        power_two = z_test_power(effect_size=0.5, n=50)
        power_one = z_test_power(effect_size=0.5, n=50, alternative="greater")
        assert power_one > power_two  # one-sided is more powerful for correct direction


class TestSampleSizeZ:
    def test_medium_effect(self):
        n = sample_size_z(effect_size=0.5, power=0.80)
        assert isinstance(n, int)
        assert n > 0
        # Verify the computed n actually achieves the desired power
        achieved = z_test_power(effect_size=0.5, n=n)
        assert achieved >= 0.80

    def test_zero_effect_raises(self):
        with pytest.raises(ValueError):
            sample_size_z(effect_size=0, power=0.80)
