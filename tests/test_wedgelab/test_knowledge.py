"""Tests for the statistical knowledge base.

The point of these tests is that a citable entry stays *executable* and stays
*correct*: the closed-form plotting positions must reproduce the constants in
the papers they cite.
"""

import numpy as np
import pytest

from wedgelab.knowledge import (
    CATEGORY_ORDER,
    KNOWLEDGE,
    PLOTTING_POSITION_ALPHAS,
    KnowledgeBase,
    KnowledgeEntry,
    symmetric_plotting_position,
)


class TestIntegrity:
    def test_keys_are_unique(self):
        keys = [e.key for e in KNOWLEDGE]
        assert len(set(keys)) == len(keys)

    def test_duplicate_keys_are_rejected(self):
        entry = KNOWLEDGE.entries[0]
        with pytest.raises(ValueError, match="duplicate"):
            KnowledgeBase(entries=(entry, entry))

    def test_every_entry_is_cited(self):
        for entry in KNOWLEDGE:
            assert entry.citation.strip(), f"{entry.key} has no citation"

    def test_every_entry_has_a_summary(self):
        for entry in KNOWLEDGE:
            assert entry.summary.strip(), f"{entry.key} has no summary"

    def test_categories_are_ordered(self):
        assert KNOWLEDGE.categories()[0] == CATEGORY_ORDER[0]

    def test_categories_partition_entries(self):
        counted = sum(len(KNOWLEDGE.by_category(c)) for c in KNOWLEDGE.categories())
        assert counted == len(KNOWLEDGE)

    def test_get_unknown_key_raises(self):
        with pytest.raises(KeyError):
            KNOWLEDGE.get("no_such_entry")


class TestFormulasExecute:
    @pytest.mark.parametrize("entry", KNOWLEDGE.with_formulas(), ids=lambda e: e.key)
    def test_formula_renders_latex(self, entry: KnowledgeEntry):
        assert entry.formula.to_latex()

    @pytest.mark.parametrize(
        "entry",
        KNOWLEDGE.by_category("Plotting positions"),
        ids=lambda e: e.key,
    )
    def test_plotting_position_is_valid(self, entry: KnowledgeEntry):
        n = 25
        p = np.asarray(
            entry.formula.evaluate(i=np.arange(1, n + 1, dtype=float), n=float(n)),
            dtype=float,
        )
        p = np.broadcast_to(p, (n,))
        assert np.all(np.isfinite(p))
        assert np.all(p > 0.0) and np.all(p < 1.0)
        assert np.all(np.diff(p) > 0), "plotting positions must increase with rank"


class TestPublishedConstants:
    """Each named rule must reproduce the constants in its source."""

    def test_weibull_is_mean_rank(self):
        f = symmetric_plotting_position(0.0)
        n = 9
        p = f.evaluate(i=np.arange(1, n + 1, dtype=float), n=float(n), a=0.0)
        assert np.allclose(p, np.arange(1, n + 1) / (n + 1))

    def test_hazen_is_midpoint(self):
        n = 8
        p = symmetric_plotting_position(0.5).evaluate(
            i=np.arange(1, n + 1, dtype=float), n=float(n), a=0.5
        )
        assert np.allclose(p, (np.arange(1, n + 1) - 0.5) / n)

    def test_blom_denominator(self):
        n = 20
        p = KNOWLEDGE.get("pp_blom").formula.evaluate(i=1.0, n=float(n))
        assert float(p) == pytest.approx((1 - 0.375) / (n + 0.25))

    def test_cunnane_denominator(self):
        n = 20
        p = KNOWLEDGE.get("pp_cunnane").formula.evaluate(i=1.0, n=float(n))
        assert float(p) == pytest.approx((1 - 0.40) / (n + 0.20))

    def test_gringorten_denominator(self):
        n = 20
        p = KNOWLEDGE.get("pp_gringorten").formula.evaluate(i=1.0, n=float(n))
        assert float(p) == pytest.approx((1 - 0.44) / (n + 0.12))

    def test_registry_alphas_match_entries(self):
        for name, a in PLOTTING_POSITION_ALPHAS.items():
            entry = KNOWLEDGE.get(f"pp_{name}")
            assert entry.formula.defaults()["a"] == pytest.approx(a)

    def test_blom_approximates_expected_normal_order_statistic(self):
        """Blom's rule exists because it tracks E[Z_(i)]; check that it does."""
        from scipy import stats as sp_stats

        n = 20
        i = np.arange(1, n + 1, dtype=float)
        p = np.asarray(KNOWLEDGE.get("pp_blom").formula.evaluate(i=i, n=float(n)))
        approx = sp_stats.norm.ppf(p)
        # Monte Carlo estimate of the true expected order statistics.
        rng = np.random.default_rng(0)
        draws = np.sort(rng.standard_normal((40000, n)), axis=1)
        exact = draws.mean(axis=0)
        assert np.max(np.abs(approx - exact)) < 0.02

    def test_filliben_endpoints_are_exact_medians(self):
        """The endpoint correction should equal the exact Beta median rank."""
        n = 12
        i = np.arange(1, n + 1, dtype=float)
        filliben = np.asarray(
            KNOWLEDGE.get("pp_filliben_exact").formula.evaluate(i=i, n=float(n))
        )
        exact = np.asarray(
            KNOWLEDGE.get("pp_median_exact").formula.evaluate(i=i, n=float(n))
        )
        assert filliben[0] == pytest.approx(exact[0], abs=1e-9)
        assert filliben[-1] == pytest.approx(exact[-1], abs=1e-9)
        # The interior is an approximation, but a close one.
        assert np.max(np.abs(filliben[1:-1] - exact[1:-1])) < 5e-4

    def test_exact_median_rank_is_symmetric(self):
        n = 15
        p = np.asarray(
            KNOWLEDGE.get("pp_median_exact").formula.evaluate(
                i=np.arange(1, n + 1, dtype=float), n=float(n)
            )
        )
        assert np.allclose(p, 1.0 - p[::-1], atol=1e-10)
        assert p[n // 2] == pytest.approx(0.5, abs=1e-10)


class TestOtherFormulas:
    def test_boxcox_limit_at_zero_is_log(self):
        f = KNOWLEDGE.get("tf_boxcox").formula
        x = np.array([0.5, 1.0, 2.0, 7.0])
        assert np.allclose(f.evaluate(x=x, lam=0.0), np.log(x))

    def test_boxcox_identity_at_one(self):
        f = KNOWLEDGE.get("tf_boxcox").formula
        x = np.array([1.0, 2.0, 5.0])
        assert np.allclose(f.evaluate(x=x, lam=1.0), x - 1.0)

    def test_mad_consistency_constant(self):
        f = KNOWLEDGE.get("fit_robust").formula
        rng = np.random.default_rng(3)
        x = rng.normal(0.0, 4.0, 40000)
        assert float(f.evaluate(x=x)) == pytest.approx(4.0, rel=0.02)

    def test_sample_standard_deviation(self):
        f = KNOWLEDGE.get("fit_moments").formula
        x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert float(f.evaluate(x=x, n=float(x.size))) == pytest.approx(
            float(np.std(x, ddof=1))
        )

    def test_ks_band_halfwidth_shrinks_with_n(self):
        f = KNOWLEDGE.get("env_ks_simultaneous").formula
        assert float(f.evaluate(n=1000.0)) < float(f.evaluate(n=10.0))

    def test_order_statistic_se_peaks_in_the_tails(self):
        f = KNOWLEDGE.get("env_normal_se").formula
        centre = float(f.evaluate(p=0.5, n=100.0))
        tail = float(f.evaluate(p=0.01, n=100.0))
        assert tail > centre


class TestSearch:
    def test_empty_query_returns_everything(self):
        assert len(KNOWLEDGE.search("")) == len(KNOWLEDGE)

    def test_search_is_case_insensitive(self):
        assert KNOWLEDGE.search("BLOM") == KNOWLEDGE.search("blom")

    def test_search_requires_all_terms(self):
        both = KNOWLEDGE.search("beta order")
        assert all("beta" in e.searchable() and "order" in e.searchable() for e in both)

    def test_search_finds_by_citation(self):
        assert any(e.key == "pp_blom" for e in KNOWLEDGE.search("1958"))

    def test_search_misses_return_empty(self):
        assert KNOWLEDGE.search("zzzzznotarealterm") == ()
