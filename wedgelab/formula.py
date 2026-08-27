"""Editable mathematical formulas with safe evaluation.

The formula engine is the heart of the workbench.  A :class:`Formula` wraps a
plain-text mathematical expression, exposes the tunable parameters inside it,
and evaluates it against NumPy arrays without ever handing unrestricted input
to :func:`eval`.

Only a fixed whitelist of AST node types and callables is admitted, so an
expression typed into the GUI cannot import modules, reach attributes, index
into objects, or call anything outside the mathematical namespace.

Formulas are immutable.  Editing one returns a *new* :class:`Formula` that
remembers what it was derived from, which is what makes a published figure
reproducible: the derivation chain is the provenance record.

Examples
--------
>>> import numpy as np
>>> f = Formula(
...     name="Symmetric plotting position",
...     expression="(i - a) / (n + 1 - 2*a)",
...     parameters=(Parameter("a", 0.375, 0.0, 0.5),),
... )
>>> np.round(f.evaluate(i=np.arange(1, 6), n=5, a=0.5), 3)
array([0.1, 0.3, 0.5, 0.7, 0.9])
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
from scipy import special as _special

from wedgestats.distributions import Beta, ChiSquared, Normal, StudentT

__all__ = [
    "Formula",
    "FormulaError",
    "Parameter",
    "NAMESPACE",
    "namespace_summary",
]


class FormulaError(ValueError):
    """Raised when a formula cannot be parsed, validated, or evaluated."""


# ---------------------------------------------------------------------------
# Distribution-backed callables
#
# These wrap the wedgestats distribution classes so that an expression typed
# by the user can reach exact quantile functions.  wedgestats exposes scalar
# ``ppf``; the wrappers add array support without duplicating the statistics.
# ---------------------------------------------------------------------------


def _broadcast(fn: Callable[..., float], *args: Any) -> Any:
    """Apply a scalar function elementwise over broadcast arguments.

    The wedgestats distributions take scalar parameters and return scalars.
    Formulas routinely need them evaluated per point -- ``beta_ppf(0.5, i,
    n - i + 1)`` builds a different Beta for every ``i`` -- so every argument,
    not just the first, is broadcast.
    """
    arrays = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])
    if arrays[0].ndim == 0:
        return float(fn(*(float(a) for a in arrays)))
    flat = [a.ravel() for a in arrays]
    out = np.array(
        [fn(*(float(col[k]) for col in flat)) for k in range(flat[0].size)],
        dtype=float,
    )
    return out.reshape(arrays[0].shape)


def normal_ppf(q: Any, mu: Any = 0.0, sigma: Any = 1.0) -> Any:
    """Quantile function of the normal distribution (wedgestats.Normal)."""
    return _broadcast(lambda v, m, s: Normal(mu=m, sigma=s).ppf(v), q, mu, sigma)


def normal_cdf(x: Any, mu: Any = 0.0, sigma: Any = 1.0) -> Any:
    """CDF of the normal distribution (wedgestats.Normal)."""
    return _broadcast(lambda v, m, s: Normal(mu=m, sigma=s).cdf(v), x, mu, sigma)


def normal_pdf(x: Any, mu: Any = 0.0, sigma: Any = 1.0) -> Any:
    """PDF of the normal distribution (wedgestats.Normal)."""
    return _broadcast(lambda v, m, s: Normal(mu=m, sigma=s).pdf(v), x, mu, sigma)


def beta_ppf(q: Any, alpha: Any, beta: Any) -> Any:
    """Quantile function of the Beta distribution (wedgestats.Beta).

    This is the exact sampling distribution of the ``i``-th uniform order
    statistic when ``alpha = i`` and ``beta = n - i + 1``, which is why it
    underpins the exact Q-Q confidence envelope.  All three arguments
    broadcast, so a whole column of order statistics can be evaluated at once.
    """
    return _broadcast(lambda v, a, b: Beta(alpha=a, beta=b).ppf(v), q, alpha, beta)


def beta_cdf(x: Any, alpha: Any, beta: Any) -> Any:
    """CDF of the Beta distribution (wedgestats.Beta)."""
    return _broadcast(lambda v, a, b: Beta(alpha=a, beta=b).cdf(v), x, alpha, beta)


def t_ppf(q: Any, df: Any) -> Any:
    """Quantile function of Student's t distribution (wedgestats.StudentT)."""
    return _broadcast(lambda v, d: StudentT(df=d).ppf(v), q, df)


def chi2_ppf(q: Any, df: Any) -> Any:
    """Quantile function of the chi-squared distribution (wedgestats.ChiSquared)."""
    return _broadcast(lambda v, d: ChiSquared(df=d).ppf(v), q, df)


# ---------------------------------------------------------------------------
# The evaluation namespace
# ---------------------------------------------------------------------------

NAMESPACE: dict[str, Any] = {
    # constants
    "pi": float(np.pi),
    "e": float(np.e),
    "inf": float(np.inf),
    "nan": float(np.nan),
    # elementary
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "log1p": np.log1p,
    "expm1": np.expm1,
    "abs": np.abs,
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "power": np.power,
    "hypot": np.hypot,
    # trigonometric
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    # selection / shaping
    "where": np.where,
    "clip": np.clip,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "arange": np.arange,
    "linspace": np.linspace,
    "full_like": np.full_like,
    "ones_like": np.ones_like,
    "zeros_like": np.zeros_like,
    # reductions
    "sum": np.sum,
    "prod": np.prod,
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "var": np.var,
    "min": np.min,
    "max": np.max,
    "cumsum": np.cumsum,
    "sort": np.sort,
    "size": np.size,
    # special functions
    "gamma_fn": _special.gamma,
    "lgamma": _special.gammaln,
    "digamma": _special.digamma,
    "beta_fn": _special.beta,
    "erf": _special.erf,
    "erfc": _special.erfc,
    "erfinv": _special.erfinv,
    "comb": _special.comb,
    # distribution quantiles (wedgestats-backed)
    "normal_ppf": normal_ppf,
    "normal_cdf": normal_cdf,
    "normal_pdf": normal_pdf,
    "beta_ppf": beta_ppf,
    "beta_cdf": beta_cdf,
    "t_ppf": t_ppf,
    "chi2_ppf": chi2_ppf,
}

# Short descriptions used by the GUI's namespace reference panel.
_NAMESPACE_DOCS: dict[str, str] = {
    "normal_ppf": "normal_ppf(q, mu=0, sigma=1) - exact normal quantile",
    "normal_cdf": "normal_cdf(x, mu=0, sigma=1) - normal CDF",
    "normal_pdf": "normal_pdf(x, mu=0, sigma=1) - normal density",
    "beta_ppf": "beta_ppf(q, alpha, beta) - Beta quantile (order statistics)",
    "beta_cdf": "beta_cdf(x, alpha, beta) - Beta CDF",
    "t_ppf": "t_ppf(q, df) - Student t quantile",
    "chi2_ppf": "chi2_ppf(q, df) - chi-squared quantile",
    "gamma_fn": "gamma_fn(x) - gamma function",
    "lgamma": "lgamma(x) - log gamma",
    "digamma": "digamma(x) - derivative of log gamma",
    "beta_fn": "beta_fn(a, b) - beta function",
    "erfinv": "erfinv(x) - inverse error function",
    "where": "where(cond, a, b) - elementwise branch",
    "clip": "clip(x, lo, hi) - bound values",
    "comb": "comb(n, k) - binomial coefficient",
}


def namespace_summary() -> list[tuple[str, str]]:
    """Return ``(name, description)`` pairs for every entry in the namespace."""
    out: list[tuple[str, str]] = []
    for key in sorted(NAMESPACE):
        doc = _NAMESPACE_DOCS.get(key)
        if doc is None:
            value = NAMESPACE[key]
            doc = f"{key} - constant" if isinstance(value, float) else f"{key}(...)"
        out.append((key, doc))
    return out


# ---------------------------------------------------------------------------
# AST whitelist
# ---------------------------------------------------------------------------

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Call,
    ast.keyword,
    ast.Compare,
    ast.BoolOp,
    ast.IfExp,
    ast.Tuple,
    ast.List,
    # operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)

_MAX_EXPONENT = 1000.0


def _validate(node: ast.AST) -> None:
    """Recursively reject anything outside the whitelist."""
    if not isinstance(node, _ALLOWED_NODES):
        raise FormulaError(
            f"'{type(node).__name__}' is not permitted in a formula "
            "(attributes, indexing, comprehensions, assignment and imports are blocked)"
        )
    if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
        raise FormulaError("only direct calls to named functions are permitted")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        exponent = node.right
        if isinstance(exponent, ast.Constant) and isinstance(exponent.value, (int, float)):
            if abs(float(exponent.value)) > _MAX_EXPONENT:
                raise FormulaError(
                    f"exponent {exponent.value} exceeds the safety limit of {_MAX_EXPONENT}"
                )
    for child in ast.iter_child_nodes(node):
        _validate(child)


def _collect_names(tree: ast.AST) -> frozenset[str]:
    """Return every :class:`ast.Name` identifier used in *tree*."""
    return frozenset(n.id for n in ast.walk(tree) if isinstance(n, ast.Name))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameter:
    """A tunable scalar inside a formula.

    Attributes
    ----------
    name : str
        Identifier as it appears in the expression.
    default : float
        Starting value.
    lower, upper : float
        Inclusive bounds; the GUI renders these as a slider range.
    step : float
        Slider resolution.
    description : str
        Short human-readable meaning.
    """

    name: str
    default: float
    lower: float
    upper: float
    step: float = 0.005
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise FormulaError(f"parameter name '{self.name}' is not a valid identifier")
        if self.lower > self.upper:
            raise FormulaError(
                f"parameter '{self.name}': lower {self.lower} exceeds upper {self.upper}"
            )
        if not (self.lower <= self.default <= self.upper):
            raise FormulaError(
                f"parameter '{self.name}': default {self.default} is outside "
                f"[{self.lower}, {self.upper}]"
            )
        if self.step <= 0:
            raise FormulaError(f"parameter '{self.name}': step must be positive")

    def clamp(self, value: float) -> float:
        """Return *value* constrained to ``[lower, upper]``."""
        return float(min(max(float(value), self.lower), self.upper))


# ---------------------------------------------------------------------------
# Formula
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Formula:
    """An immutable, safely-evaluable mathematical expression.

    Parameters
    ----------
    name : str
        Display name, e.g. ``"Blom"``.
    expression : str
        The right-hand side, in Python infix syntax.
    lhs : str
        Symbol the expression defines, used for display and LaTeX output.
    description : str
        One-line explanation of what the formula computes.
    parameters : tuple[Parameter, ...]
        Tunable scalars appearing in *expression*.
    citation : str
        Literature reference for the formula.
    latex : str
        Hand-written LaTeX; when empty, :meth:`to_latex` derives one.
    derived_from : str
        Name of the formula this one was edited out of, or ``""``.
    """

    name: str
    expression: str
    lhs: str = "y"
    description: str = ""
    parameters: tuple[Parameter, ...] = ()
    citation: str = ""
    latex: str = ""
    derived_from: str = ""
    _tree: Any = field(default=None, repr=False, compare=False)
    _code: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        text = self.expression.strip()
        if not text:
            raise FormulaError("expression is empty")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise FormulaError(f"syntax error: {exc.msg}") from exc
        _validate(tree)

        used = _collect_names(tree)
        unknown_calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        } - set(NAMESPACE)
        if unknown_calls:
            raise FormulaError("unknown function(s): " + ", ".join(sorted(unknown_calls)))

        declared = {p.name for p in self.parameters}
        if len(declared) != len(self.parameters):
            raise FormulaError("duplicate parameter names")
        missing = declared - used
        if missing:
            raise FormulaError(
                "declared parameter(s) absent from the expression: "
                + ", ".join(sorted(missing))
            )

        object.__setattr__(self, "expression", text)
        object.__setattr__(self, "_tree", tree)
        object.__setattr__(self, "_code", compile(tree, "<formula>", "eval"))

    # -- introspection ----------------------------------------------------

    @property
    def symbols(self) -> frozenset[str]:
        """Free identifiers in the expression that are not namespace entries."""
        return frozenset(_collect_names(self._tree)) - set(NAMESPACE)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the declared tunable parameters."""
        return tuple(p.name for p in self.parameters)

    @property
    def inputs(self) -> tuple[str, ...]:
        """Symbols the caller must supply (data), excluding declared parameters."""
        return tuple(sorted(self.symbols - set(self.parameter_names)))

    def parameter(self, name: str) -> Parameter:
        """Return the declared :class:`Parameter` called *name*."""
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(name)

    def defaults(self) -> dict[str, float]:
        """Mapping of parameter name to default value."""
        return {p.name: p.default for p in self.parameters}

    # -- evaluation -------------------------------------------------------

    def evaluate(self, **bindings: Any) -> Any:
        """Evaluate the expression.

        Declared parameters fall back to their defaults; every other free
        symbol must be supplied.

        Raises
        ------
        FormulaError
            If a symbol is unbound or evaluation fails.
        """
        env: dict[str, Any] = dict(NAMESPACE)
        env.update(self.defaults())
        env.update(bindings)

        unbound = self.symbols - set(env)
        if unbound:
            raise FormulaError(
                "unbound symbol(s): "
                + ", ".join(sorted(unbound))
                + f" (expected inputs: {', '.join(self.inputs) or 'none'})"
            )

        try:
            with np.errstate(all="ignore"):
                result = eval(self._code, {"__builtins__": {}}, env)
        except FormulaError:
            raise
        except Exception as exc:
            raise FormulaError(f"evaluation failed: {exc}") from exc
        return result

    def check(self, **bindings: Any) -> tuple[bool, str]:
        """Try to evaluate and report success without raising.

        Returns
        -------
        tuple[bool, str]
            ``(ok, message)``; *message* is empty when ``ok`` is ``True``.
        """
        try:
            value = self.evaluate(**bindings)
        except FormulaError as exc:
            return False, str(exc)
        arr = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            return False, f"{n_bad} non-finite value(s) produced"
        return True, ""

    # -- editing ----------------------------------------------------------

    def edit(
        self,
        expression: str,
        *,
        name: str | None = None,
        parameters: Iterable[Parameter] | None = None,
        description: str | None = None,
    ) -> "Formula":
        """Return a new formula with a replaced expression.

        Declared parameters that survive in the new expression are carried
        over automatically unless *parameters* is given explicitly.
        """
        if parameters is None:
            try:
                probe = ast.parse(expression.strip() or "0", mode="eval")
                used = _collect_names(probe)
            except SyntaxError:
                used = frozenset()
            parameters = tuple(p for p in self.parameters if p.name in used)
        return Formula(
            name=name or f"{self.name} (edited)",
            expression=expression,
            lhs=self.lhs,
            description=self.description if description is None else description,
            parameters=tuple(parameters),
            citation="" if name is None else self.citation,
            derived_from=self.name,
        )

    def with_parameter(self, name: str, **changes: Any) -> "Formula":
        """Return a new formula with one parameter's metadata changed."""
        params: list[Parameter] = []
        for p in self.parameters:
            if p.name == name:
                fields = {
                    "name": p.name,
                    "default": p.default,
                    "lower": p.lower,
                    "upper": p.upper,
                    "step": p.step,
                    "description": p.description,
                }
                fields.update(changes)
                params.append(Parameter(**fields))
            else:
                params.append(p)
        return Formula(
            name=self.name,
            expression=self.expression,
            lhs=self.lhs,
            description=self.description,
            parameters=tuple(params),
            citation=self.citation,
            latex=self.latex,
            derived_from=self.derived_from,
        )

    def freeze(self, **values: float) -> "Formula":
        """Return a formula with the given parameters pinned to *values*.

        The parameters keep their names but their range collapses to a point,
        which is how the exporter records the exact settings behind a figure.
        """
        out = self
        for key, value in values.items():
            v = float(value)
            out = out.with_parameter(key, default=v, lower=v, upper=v)
        return out

    # -- presentation -----------------------------------------------------

    def to_latex(self) -> str:
        """Return a LaTeX rendering of the formula.

        Uses the hand-written :attr:`latex` when present, otherwise asks SymPy,
        otherwise falls back to a lightly cleaned copy of the expression.
        """
        if self.latex:
            return self.latex
        try:
            import sympy

            expr = sympy.sympify(self.expression)
            return f"{self.lhs} = {sympy.latex(expr)}"
        except Exception:
            body = self.expression.replace("*", " \\cdot ")
            return f"{self.lhs} = {body}"

    def simplified(self) -> "Formula":
        """Return an algebraically simplified copy (requires SymPy)."""
        try:
            import sympy

            simple = sympy.simplify(sympy.sympify(self.expression))
        except Exception as exc:
            raise FormulaError(f"simplification unavailable: {exc}") from exc
        return self.edit(str(simple), name=f"{self.name} (simplified)")

    def derivative(self, wrt: str) -> "Formula":
        """Return the partial derivative with respect to *wrt* (requires SymPy)."""
        try:
            import sympy

            d = sympy.diff(sympy.sympify(self.expression), sympy.Symbol(wrt))
            text = str(d)
            used = _collect_names(ast.parse(text, mode="eval"))
        except FormulaError:
            raise
        except Exception as exc:
            raise FormulaError(f"differentiation unavailable: {exc}") from exc
        return Formula(
            name=f"d({self.lhs})/d{wrt}",
            expression=text,
            lhs=f"d{self.lhs}/d{wrt}",
            description=f"Partial derivative of {self.name} with respect to {wrt}",
            parameters=tuple(p for p in self.parameters if p.name in used),
            derived_from=self.name,
        )

    def pretty(self) -> str:
        """Return a single-line human-readable rendering."""
        return f"{self.lhs} = {self.expression}"

    def __str__(self) -> str:
        return self.pretty()
