"""Tests for the safe formula engine."""

import numpy as np
import pytest

from wedgelab.formula import (
    NAMESPACE,
    Formula,
    FormulaError,
    Parameter,
    namespace_summary,
)

BLOM = "(i - a) / (n + 1 - 2*a)"


def blom(a: float = 0.375) -> Formula:
    return Formula(
        name="Blom",
        expression=BLOM,
        lhs="p_i",
        parameters=(Parameter("a", a, 0.0, 0.5),),
    )


class TestParsing:
    def test_accepts_arithmetic(self):
        assert Formula(name="f", expression="1 + 2*3 - 4/5").evaluate() == pytest.approx(6.2)

    def test_rejects_empty(self):
        with pytest.raises(FormulaError, match="empty"):
            Formula(name="f", expression="   ")

    def test_rejects_syntax_error(self):
        with pytest.raises(FormulaError, match="syntax error"):
            Formula(name="f", expression="1 +")

    def test_rejects_unknown_function(self):
        with pytest.raises(FormulaError, match="unknown function"):
            Formula(name="f", expression="frobnicate(3)")

    def test_declared_parameter_must_appear(self):
        with pytest.raises(FormulaError, match="absent from the expression"):
            Formula(name="f", expression="i + n", parameters=(Parameter("q", 0.0, 0.0, 1.0),))


class TestSandbox:
    """The evaluator must not be reachable from a typed expression."""

    @pytest.mark.parametrize(
        "hostile",
        [
            '__import__("os").system("echo pwned")',
            "x.__class__",
            "().__class__.__bases__",
            "[q for q in range(3)]",
            "{1: 2}",
            "lambda: 1",
            "(yield)",
            "x[0]",
            "f'{x}'",
        ],
    )
    def test_rejects_hostile_input(self, hostile):
        with pytest.raises(FormulaError):
            Formula(name="hostile", expression=hostile)

    def test_rejects_enormous_exponent(self):
        with pytest.raises(FormulaError, match="safety limit"):
            Formula(name="f", expression="2**99999")

    def test_allows_bounded_exponent(self):
        assert Formula(name="f", expression="2**10").evaluate() == pytest.approx(1024)

    def test_builtins_are_unreachable(self):
        with pytest.raises(FormulaError):
            Formula(name="f", expression="open('x')")


class TestEvaluation:
    def test_scalar(self):
        assert float(blom().evaluate(i=1, n=10)) == pytest.approx(0.625 / 10.25)

    def test_vectorised(self):
        p = blom(0.5).evaluate(i=np.arange(1, 6), n=5)
        assert np.allclose(p, [0.1, 0.3, 0.5, 0.7, 0.9])

    def test_parameter_default_is_used(self):
        assert float(blom(0.5).evaluate(i=3, n=5)) == pytest.approx(0.5)

    def test_binding_overrides_default(self):
        assert float(blom(0.5).evaluate(i=1, n=10, a=0.0)) == pytest.approx(1 / 11)

    def test_unbound_symbol_reports_inputs(self):
        with pytest.raises(FormulaError, match="unbound symbol"):
            blom().evaluate(i=1)

    def test_check_reports_non_finite(self):
        f = Formula(name="f", expression="1 / (i - 1)")
        ok, message = f.check(i=np.array([1.0, 2.0]))
        assert not ok
        assert "non-finite" in message

    def test_check_succeeds(self):
        ok, message = blom().check(i=np.arange(1, 6), n=5)
        assert ok and message == ""


class TestIntrospection:
    def test_inputs_exclude_parameters(self):
        assert blom().inputs == ("i", "n")

    def test_parameter_names(self):
        assert blom().parameter_names == ("a",)

    def test_symbols_exclude_namespace(self):
        f = Formula(name="f", expression="sqrt(i) + pi")
        assert f.symbols == frozenset({"i"})

    def test_defaults(self):
        assert blom(0.4).defaults() == {"a": 0.4}

    def test_missing_parameter_raises(self):
        with pytest.raises(KeyError):
            blom().parameter("nope")


class TestEditing:
    def test_edit_keeps_surviving_parameters(self):
        edited = blom().edit("(i - a) / n")
        assert edited.parameter_names == ("a",)
        assert edited.derived_from == "Blom"

    def test_edit_drops_absent_parameters(self):
        edited = blom().edit("i / (n + 1)")
        assert edited.parameter_names == ()

    def test_freeze_pins_range(self):
        frozen = blom().freeze(a=0.4)
        p = frozen.parameter("a")
        assert (p.default, p.lower, p.upper) == (0.4, 0.4, 0.4)

    def test_with_parameter_changes_bounds(self):
        wider = blom().with_parameter("a", upper=0.9)
        assert wider.parameter("a").upper == 0.9

    def test_formula_is_immutable(self):
        f = blom()
        with pytest.raises(Exception):
            f.expression = "1"


class TestParameterValidation:
    def test_rejects_bad_identifier(self):
        with pytest.raises(FormulaError, match="identifier"):
            Parameter("2a", 0.0, 0.0, 1.0)

    def test_rejects_inverted_bounds(self):
        with pytest.raises(FormulaError, match="exceeds upper"):
            Parameter("a", 0.5, 1.0, 0.0)

    def test_rejects_default_outside_bounds(self):
        with pytest.raises(FormulaError, match="outside"):
            Parameter("a", 2.0, 0.0, 1.0)

    def test_rejects_non_positive_step(self):
        with pytest.raises(FormulaError, match="step"):
            Parameter("a", 0.5, 0.0, 1.0, step=0.0)

    def test_clamp(self):
        p = Parameter("a", 0.5, 0.0, 1.0)
        assert p.clamp(-3) == 0.0
        assert p.clamp(3) == 1.0
        assert p.clamp(0.25) == 0.25


class TestDistributionCallables:
    """The wedgestats-backed quantile functions must broadcast every argument."""

    def test_normal_ppf_scalar(self):
        f = Formula(name="f", expression="normal_ppf(0.975)")
        assert float(f.evaluate()) == pytest.approx(1.959963985, abs=1e-6)

    def test_normal_ppf_vector(self):
        f = Formula(name="f", expression="normal_ppf(p)")
        out = f.evaluate(p=np.array([0.025, 0.5, 0.975]))
        assert np.allclose(out, [-1.959963985, 0.0, 1.959963985], atol=1e-6)

    def test_beta_ppf_broadcasts_shape_parameters(self):
        # The exact median rank needs a different Beta for every i.
        f = Formula(name="f", expression="beta_ppf(0.5, i, n - i + 1)")
        out = f.evaluate(i=np.arange(1, 6), n=5)
        assert out.shape == (5,)
        assert np.all(np.diff(out) > 0)
        assert out[2] == pytest.approx(0.5, abs=1e-9)

    def test_beta_ppf_symmetric(self):
        f = Formula(name="f", expression="beta_ppf(0.5, i, n - i + 1)")
        out = f.evaluate(i=np.arange(1, 11), n=10)
        assert np.allclose(out, 1.0 - out[::-1], atol=1e-9)

    def test_t_and_chi2(self):
        assert float(
            Formula(name="f", expression="t_ppf(0.975, 10)").evaluate()
        ) == pytest.approx(2.2281388519, abs=1e-6)
        assert float(
            Formula(name="f", expression="chi2_ppf(0.95, 3)").evaluate()
        ) == pytest.approx(7.814727903, abs=1e-6)


class TestSymbolic:
    def test_latex_uses_hand_written_when_given(self):
        f = Formula(name="f", expression="i / n", latex=r"p = \frac{i}{n}")
        assert f.to_latex() == r"p = \frac{i}{n}"

    def test_latex_falls_back_to_sympy(self):
        assert "frac" in blom().to_latex()

    def test_derivative(self):
        d = blom().derivative("a")
        assert d.derived_from == "Blom"
        # Numerically check against a central difference.
        h = 1e-6
        f = blom()
        approx = (
            float(f.evaluate(i=3, n=10, a=0.375 + h))
            - float(f.evaluate(i=3, n=10, a=0.375 - h))
        ) / (2 * h)
        assert float(d.evaluate(i=3, n=10, a=0.375)) == pytest.approx(approx, abs=1e-5)

    @pytest.mark.parametrize("symbol", ["S", "E", "I", "N", "Q", "O", "beta"])
    def test_symbols_that_collide_with_sympy_still_differentiate(self, symbol):
        """Single capitals are ordinary variable names, not SymPy singletons."""
        f = Formula(name="f", expression=f"2 * {symbol}**2", lhs="y")
        d = f.derivative(symbol)
        assert float(d.evaluate(**{symbol: 3.0})) == pytest.approx(12.0)

    def test_colliding_symbol_renders_latex(self):
        f = Formula(name="MM", expression="Vmax * S / (Km + S)", lhs="v")
        latex = f.to_latex()
        assert "frac" in latex, latex

    def test_michaelis_menten_derivative_is_correct(self):
        f = Formula(name="MM", expression="Vmax * S / (Km + S)", lhs="v")
        d = f.derivative("S")
        # d/dS [Vmax*S/(Km+S)] = Vmax*Km/(Km+S)^2
        expected = 2.0 * 1.5 / (1.5 + 1.0) ** 2
        assert float(d.evaluate(S=1.0, Vmax=2.0, Km=1.5)) == pytest.approx(expected)

    def test_simplified_is_equivalent(self):
        f = Formula(name="f", expression="(i + i) / 2")
        assert float(f.simplified().evaluate(i=7)) == pytest.approx(7.0)


class TestNamespace:
    def test_summary_covers_namespace(self):
        assert len(namespace_summary()) == len(NAMESPACE)

    def test_summary_entries_are_pairs(self):
        assert all(len(entry) == 2 for entry in namespace_summary())

    def test_piecewise_via_where(self):
        f = Formula(name="f", expression="where(i == 1, 0.0, 1.0)")
        assert np.allclose(f.evaluate(i=np.array([1.0, 2.0])), [0.0, 1.0])
