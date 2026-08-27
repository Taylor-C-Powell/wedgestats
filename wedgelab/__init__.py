"""wedgelab - an interactive workbench for statistical formulas.

A standalone application built on :mod:`wedgestats`.  It exists to make the
choices inside a statistical figure visible, editable, and citable, using the
publication-quality Q-Q plot as its worked example.

Three ideas hold it together:

**Formulas are objects.**  A :class:`~wedgelab.formula.Formula` is a
plain-text expression with declared parameters, evaluated through an
AST whitelist rather than :func:`eval`.  Editing one produces a new formula
that remembers what it came from.

**Knowledge is citable and executable.**  Every entry in
:data:`~wedgelab.knowledge.KNOWLEDGE` pairs a piece of textbook statistics
with a runnable formula and a full literature reference, so loading a rule
into the workbench brings its provenance with it.

**A figure is a computation.**  :func:`~wedgelab.export.to_script` writes a
standalone Python file that regenerates a figure exactly, including the
plotting-position formula that produced it.

Examples
--------
>>> import wedgelab as wl
>>> data = wl.generate("heavy", n=120, seed=1)
>>> result = wl.compute(wl.QQSpec(data=data, dist_key="normal", label="t(3) draws"))
>>> round(result.diagnostics.ppcc, 3) < 1.0
True

Launch the interface with ``python -m wedgelab``.
"""

__version__ = "1.0.0"

from wedgelab.datasets import DATASETS, Dataset, dataset_keys, generate, load_file
from wedgelab.export import (
    read_session,
    save_figure,
    to_json,
    to_script,
    write_session,
)
from wedgelab.formula import Formula, FormulaError, Parameter, namespace_summary
from wedgelab.knowledge import (
    KNOWLEDGE,
    KnowledgeBase,
    KnowledgeEntry,
    symmetric_plotting_position,
)
from wedgelab.models import DISTRIBUTIONS, DistributionSpec, FitError, FitResult, fit
from wedgelab.plot import PlotOptions, render
from wedgelab.qq import Diagnostics, QQResult, QQSpec, compute
from wedgelab.theme import THEMES, Theme, get_theme, theme_keys

__all__ = [
    "__version__",
    # formulas
    "Formula",
    "FormulaError",
    "Parameter",
    "namespace_summary",
    # knowledge
    "KNOWLEDGE",
    "KnowledgeBase",
    "KnowledgeEntry",
    "symmetric_plotting_position",
    # models
    "DISTRIBUTIONS",
    "DistributionSpec",
    "FitError",
    "FitResult",
    "fit",
    # q-q
    "QQSpec",
    "QQResult",
    "Diagnostics",
    "compute",
    # presentation
    "Theme",
    "THEMES",
    "get_theme",
    "theme_keys",
    "PlotOptions",
    "render",
    # data
    "Dataset",
    "DATASETS",
    "dataset_keys",
    "generate",
    "load_file",
    # export
    "save_figure",
    "to_script",
    "to_json",
    "write_session",
    "read_session",
    # interface
    "launch",
]


def launch() -> None:
    """Open the Tkinter workbench.

    Imported lazily so that the computational half of the package works in
    headless environments where Tk is unavailable.
    """
    from wedgelab.gui import launch as _launch

    _launch()
