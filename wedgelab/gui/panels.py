"""The workbench control panels.

Five stacked sections, one per decision behind a Q-Q plot: where the data come
from, what reference they are compared against, how order statistics are
mapped to probabilities, how the result is drawn, and what the literature says
about all of it.

Panels never compute statistics themselves.  They mutate
:class:`~wedgelab.gui.state.AppState` and call ``app.request_update()``; the
application owns the single recompute path.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

import numpy as np

from wedgelab.datasets import DATASETS, generate, load_file
from wedgelab.formula import Formula, FormulaError, Parameter
from wedgelab.knowledge import KNOWLEDGE
from wedgelab.models import DISTRIBUTIONS, FIT_METHODS, sample_skewness_hint
from wedgelab.qq import ENVELOPE_METHODS, LINE_METHODS, check_positions
from wedgelab.theme import THEMES, get_theme

from wedgelab.gui.widgets import (
    PALETTE,
    LabeledCombo,
    MathPreview,
    Section,
    SliderRow,
    StatusLine,
)

if TYPE_CHECKING:
    from wedgelab.gui.app import WedgeLabApp

__all__ = ["DataPanel", "ModelPanel", "FormulaPanel", "PresentationPanel", "KnowledgePanel"]

# Symbols the plotting-position formula receives from the engine rather than
# from a slider.
RESERVED_SYMBOLS = frozenset({"i", "n"})

_FIT_LABELS = {
    "mle": "Maximum likelihood",
    "moments": "Method of moments",
    "robust": "Robust (median / MAD)",
    "manual": "Manual parameters",
}

_LINE_LABELS = {
    "ols": "Least squares (all points)",
    "quartile": "Quartile line (resistant)",
    "theoretical": "Theoretical identity",
}

_ENVELOPE_LABELS = {
    "auto": "Auto (calibrated for the estimator)",
    "none": "None",
    "beta": "Exact pointwise (Beta)",
    "asymptotic": "Asymptotic pointwise",
    "simultaneous": "Simultaneous (KS)",
    "bootstrap": "Parametric bootstrap",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class DataPanel(Section):
    """Choose a synthetic sample or load a column from a file."""

    def __init__(self, parent: tk.Misc, app: "WedgeLabApp") -> None:
        super().__init__(parent, "1.  Data")
        self.app = app

        self._dataset = LabeledCombo(
            self,
            [(k, d.label) for k, d in DATASETS.items()],
            self._on_dataset,
            initial=app.state.dataset_key,
        )
        self.add(self._dataset, label="Sample")

        size = ttk.Frame(self)
        size.columnconfigure(1, weight=1)
        self._n = tk.IntVar(value=app.state.dataset_n)
        self._seed = tk.IntVar(value=app.state.dataset_seed)
        ttk.Label(size, text="n").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            size, from_=5, to=20000, increment=10, textvariable=self._n, width=8,
            command=self._on_regenerate,
        ).grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Label(size, text="seed").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(
            size, from_=0, to=99999, textvariable=self._seed, width=7,
            command=self._on_regenerate,
        ).grid(row=0, column=3, sticky="w", padx=(4, 0))
        self.add(size, span=True)

        buttons = ttk.Frame(self)
        ttk.Button(buttons, text="Redraw sample", command=self._on_regenerate).pack(
            side="left"
        )
        ttk.Button(buttons, text="Load file...", command=self._on_load).pack(
            side="left", padx=(6, 0)
        )
        self.add(buttons, span=True)

        self._summary = ttk.Label(self, font=("Consolas", 8), foreground=PALETTE["ink"])
        self.add(self._summary, span=True)
        self._expect = self.note("")
        self.refresh_summary()

    # -- callbacks --------------------------------------------------------

    def _on_dataset(self, key: str) -> None:
        spec = DATASETS[key]
        self.app.state.dataset_key = key
        self._dataset.set_key(key)
        self._n.set(spec.n)
        self.app.state.dist_key = spec.suggested_dist
        self.app.model_panel.sync()
        self._on_regenerate()

    def sync(self) -> None:
        """Push the current state out to the widgets."""
        state = self.app.state
        self._dataset.set_key(state.dataset_key)
        self._n.set(state.dataset_n)
        self._seed.set(state.dataset_seed)
        self.refresh_summary()

    def _on_regenerate(self) -> None:
        state = self.app.state
        try:
            n = max(5, int(self._n.get()))
            seed = int(self._seed.get())
        except (tk.TclError, ValueError):
            self.app.status.error("n and seed must be whole numbers")
            return
        state.dataset_n, state.dataset_seed = n, seed
        state.data = generate(state.dataset_key, n, seed)
        state.label = DATASETS[state.dataset_key].label
        state.source = f"dataset:{state.dataset_key}"
        self.refresh_summary()
        self.app.request_update()

    def _on_load(self) -> None:
        path = filedialog.askopenfilename(
            title="Load a numeric column",
            filetypes=[
                ("Data files", "*.csv *.tsv *.txt *.dat"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            values, name = load_file(path)
        except Exception as exc:
            messagebox.showerror("Could not load file", str(exc))
            return
        finite = values[np.isfinite(values)]
        if finite.size < 3:
            messagebox.showerror(
                "Could not load file", "the chosen column has fewer than three numbers"
            )
            return
        state = self.app.state
        state.data = values
        state.label = name
        state.source = path
        self.refresh_summary()
        self.app.request_update()

    # -- display ----------------------------------------------------------

    def refresh_summary(self) -> None:
        """Update the sample summary and the expectation note."""
        data = self.app.state.data
        finite = data[np.isfinite(data)]
        if finite.size:
            self._summary.configure(
                text=(
                    f"{self.app.state.label}\n"
                    f"n={finite.size}  min={finite.min():.4g}  "
                    f"median={np.median(finite):.4g}  max={finite.max():.4g}"
                )
            )
        key = self.app.state.dataset_key
        if self.app.state.source.startswith("dataset:"):
            self._expect.configure(text="Expect: " + DATASETS[key].expect)
        else:
            self._expect.configure(text=sample_skewness_hint(finite))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ModelPanel(Section):
    """Choose the reference distribution and how its parameters are estimated."""

    def __init__(self, parent: tk.Misc, app: "WedgeLabApp") -> None:
        super().__init__(parent, "2.  Reference model")
        self.app = app

        self._dist = LabeledCombo(
            self,
            [(k, s.label) for k, s in DISTRIBUTIONS.items()],
            self._on_dist,
            initial=app.state.dist_key,
        )
        self.add(self._dist, label="Distribution")

        self._method = LabeledCombo(
            self,
            [(m, _FIT_LABELS[m]) for m in FIT_METHODS],
            self._on_method,
            initial=app.state.fit_method,
        )
        self.add(self._method, label="Estimator")

        self._manual = ttk.Entry(self, font=("Consolas", 9))
        self._manual_row = self.add(self._manual, label="Parameters")
        self._manual.bind("<Return>", lambda _e: self._on_manual())
        self._manual.bind("<FocusOut>", lambda _e: self._on_manual())

        self._fit_label = ttk.Label(self, font=("Consolas", 8), foreground=PALETTE["accent"])
        self.add(self._fit_label, span=True)
        self._notes = self.note("")
        self._sync_manual_visibility()
        self.sync_distribution()

    def _on_dist(self, key: str) -> None:
        self.app.state.dist_key = key
        self.app.state.manual_params = None
        self.sync()
        self.app.request_update()

    def _on_method(self, key: str) -> None:
        self.app.state.fit_method = key
        if key == "manual" and self.app.state.manual_params is None:
            last = self.app.last_result
            if last is not None:
                self.app.state.manual_params = last.fit.params
                self._manual.delete(0, "end")
                self._manual.insert(0, ", ".join(f"{v:.6g}" for v in last.fit.params))
        self.sync()
        self.app.request_update()

    def _on_manual(self) -> None:
        if self.app.state.fit_method != "manual":
            return
        text = self._manual.get().replace(";", ",")
        try:
            values = tuple(float(part) for part in text.split(",") if part.strip())
        except ValueError:
            self.app.status.error("parameters must be a comma-separated list of numbers")
            return
        spec = DISTRIBUTIONS[self.app.state.dist_key]
        if len(values) != len(spec.param_names):
            self.app.status.error(
                f"{spec.label} needs {len(spec.param_names)}: {', '.join(spec.param_names)}"
            )
            return
        self.app.state.manual_params = values
        self.app.request_update()

    def _sync_manual_visibility(self) -> None:
        manual = self.app.state.fit_method == "manual"
        self._manual.configure(state="normal" if manual else "disabled")

    def sync(self) -> None:
        """Push the current state out to the widgets."""
        state = self.app.state
        spec = DISTRIBUTIONS[state.dist_key]
        self._dist.set_key(spec.key)
        self._method.set_key(state.fit_method)
        self._notes.configure(text=spec.notes)
        self._fit_label.configure(text=f"parameters: {', '.join(spec.param_names)}")
        if state.manual_params:
            self._manual.configure(state="normal")
            self._manual.delete(0, "end")
            self._manual.insert(0, ", ".join(f"{v:.6g}" for v in state.manual_params))
        self._sync_manual_visibility()

    # Retained under its original name for callers that only want the
    # distribution refreshed.
    sync_distribution = sync

    def show_fit(self, summary: str) -> None:
        """Display the fitted parameters returned by the engine."""
        self._fit_label.configure(text=summary)


# ---------------------------------------------------------------------------
# Formula workbench
# ---------------------------------------------------------------------------


class FormulaPanel(Section):
    """Edit the plotting-position formula and bind its parameters live.

    Typing a new symbol into the expression creates a slider for it, which is
    how an edited formula becomes an explorable one-parameter family rather
    than a fixed constant.
    """

    def __init__(self, parent: tk.Misc, app: "WedgeLabApp") -> None:
        super().__init__(parent, "3.  Plotting-position workbench")
        self.app = app
        self._sliders: list[SliderRow] = []
        self._debounce: str | None = None

        positions = KNOWLEDGE.by_category("Plotting positions")
        self._preset = LabeledCombo(
            self,
            [(e.key, e.name) for e in positions],
            self._on_preset,
            initial="pp_blom",
        )
        self.add(self._preset, label="Preset")

        ttk.Label(self, text="p_i =", font=("Consolas", 9)).grid(
            row=self.next_row(), column=0, sticky="w", pady=(6, 0)
        )
        self._entry = tk.Text(self, height=2, wrap="word", font=("Consolas", 10))
        self._entry.grid(row=self.next_row(), column=0, columnspan=2, sticky="ew", pady=(0, 3))
        self._entry.insert("1.0", app.state.position.expression)
        self._entry.bind("<KeyRelease>", self._on_typed)

        self._status = StatusLine(self, font=("Segoe UI", 8))
        self.add(self._status, span=True)

        self._preview = MathPreview(self)
        self.add(self._preview, span=True)

        buttons = ttk.Frame(self)
        ttk.Button(buttons, text="Apply", command=self._apply_now, width=7).pack(side="left")
        ttk.Button(buttons, text="Revert", command=self._revert, width=7).pack(
            side="left", padx=(4, 0)
        )
        ttk.Button(buttons, text="Calculus", command=self._calculus, width=9).pack(
            side="left", padx=(4, 0)
        )
        ttk.Button(buttons, text="Functions", command=self._show_namespace, width=10).pack(
            side="left", padx=(4, 0)
        )
        self.add(buttons, span=True)

        self._slider_host = ttk.Frame(self)
        self.add(self._slider_host, span=True)

        self._citation = self.note("")
        self.rebuild(app.state.position, app.state.position_bindings)

    # -- formula lifecycle ------------------------------------------------

    def set_expression(self, text: str) -> None:
        """Replace the editor contents and apply the result immediately."""
        self._entry.delete("1.0", "end")
        self._entry.insert("1.0", text)
        self._apply_now()

    @property
    def status_text(self) -> str:
        """Current validation message shown under the editor."""
        return str(self._status.cget("text"))

    def rebuild(self, formula: Formula, bindings: dict[str, float] | None = None) -> None:
        """Adopt *formula*, rebuild its sliders, and redraw the preview."""
        state = self.app.state
        state.position = formula
        merged = dict(formula.defaults())
        if bindings:
            merged.update({k: v for k, v in bindings.items() if k in merged})
        state.position_bindings = merged

        current = self._entry.get("1.0", "end").strip()
        if current != formula.expression:
            self._entry.delete("1.0", "end")
            self._entry.insert("1.0", formula.expression)

        for slider in self._sliders:
            slider.destroy()
        self._sliders = []
        for param in formula.parameters:
            row = SliderRow(
                self._slider_host,
                param.name,
                merged.get(param.name, param.default),
                param.lower,
                param.upper,
                param.step,
                param.description,
                self._on_slider,
            )
            row.pack(fill="x", expand=True)
            self._sliders.append(row)

        self._preview.show(formula.to_latex())
        cite = formula.citation or (
            f"derived from {formula.derived_from}" if formula.derived_from else ""
        )
        self._citation.configure(text=cite)
        self._status.ok("formula applied")

    def select_preset(self, key: str) -> None:
        """Sync the preset combobox to *key* without reloading the formula."""
        self._preset.set_key(key)

    def _on_preset(self, key: str) -> None:
        entry = KNOWLEDGE.get(key)
        if entry.formula is None:
            self._status.error(f"{entry.name} has no executable form")
            return
        self.app.state.position_source = key
        self.rebuild(entry.formula)
        self.app.request_update()

    def _on_slider(self, name: str, value: float) -> None:
        self.app.state.position_bindings[name] = value
        self.app.request_update()

    def _on_typed(self, _event: tk.Event) -> None:
        if self._debounce is not None:
            self.after_cancel(self._debounce)
        self._debounce = self.after(350, self._apply_now)

    def _revert(self) -> None:
        self._on_preset(self.app.state.position_source)

    def _apply_now(self) -> None:
        """Validate the typed expression and adopt it if it works."""
        self._debounce = None
        text = self._entry.get("1.0", "end").strip()
        if not text:
            self._status.error("expression is empty")
            return
        previous = self.app.state.position
        if text == previous.expression:
            return

        try:
            candidate = Formula(name="probe", expression=text)
        except FormulaError as exc:
            self._status.error(str(exc))
            return

        unknown = candidate.symbols - RESERVED_SYMBOLS
        known = {p.name: p for p in previous.parameters}
        parameters = tuple(
            known.get(name, self._invent(name)) for name in sorted(unknown)
        )

        try:
            formula = Formula(
                name=f"{previous.name} (edited)" if not previous.derived_from else previous.name,
                expression=text,
                lhs="p_i",
                description=previous.description,
                parameters=parameters,
                derived_from=previous.derived_from or previous.name,
            )
        except FormulaError as exc:
            self._status.error(str(exc))
            return

        # Validate against the same contract the engine enforces, so an
        # expression the engine would reject never gets installed here.
        n = int(np.sum(np.isfinite(self.app.state.data))) or 20
        defaults = formula.defaults()
        probe = {
            **defaults,
            **{k: v for k, v in self.app.state.position_bindings.items() if k in defaults},
        }
        ok, message = check_positions(formula, n, probe)
        if not ok:
            self._status.error(message)
            return

        self.rebuild(formula, self.app.state.position_bindings)
        self.app.request_update()

    @staticmethod
    def _invent(name: str) -> Parameter:
        """Create a slider for a symbol the user just introduced."""
        return Parameter(
            name=name,
            default=0.0,
            lower=-1.0,
            upper=1.0,
            step=0.005,
            description="introduced by editing; adjust the range as needed",
        )

    # -- symbolic helpers -------------------------------------------------

    def _calculus(self) -> None:
        """Show the simplified form and every partial derivative."""
        formula = self.app.state.position
        lines = [f"expression:  {formula.expression}", ""]
        try:
            lines.append(f"simplified:  {formula.simplified().expression}")
        except FormulaError as exc:
            lines.append(f"simplified:  unavailable ({exc})")
        lines.append("")
        targets = list(formula.parameter_names) or ["i"]
        for name in targets:
            try:
                lines.append(f"d p_i / d {name}:  {formula.derivative(name).expression}")
            except FormulaError as exc:
                lines.append(f"d p_i / d {name}:  unavailable ({exc})")
        lines += [
            "",
            "A derivative near zero over the slider range means the parameter",
            "is not doing visible work; a large one means the figure is",
            "sensitive to a choice the caption should state.",
        ]
        self.app.show_text("Formula calculus", "\n".join(lines))

    def _show_namespace(self) -> None:
        """List every callable available inside an expression."""
        from wedgelab.formula import namespace_summary

        lines = [
            "Symbols supplied by the engine:",
            "    i    rank, 1 .. n",
            "    n    sample size",
            "",
            "Any other symbol you type becomes a slider.",
            "",
            "Available functions and constants:",
        ]
        lines += [f"    {doc}" for _, doc in namespace_summary()]
        self.app.show_text("Expression namespace", "\n".join(lines))


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


class PresentationPanel(Section):
    """Reference line, confidence envelope, scale, and publication theme."""

    def __init__(self, parent: tk.Misc, app: "WedgeLabApp") -> None:
        super().__init__(parent, "4.  Line, envelope, and theme")
        self.app = app

        self._line = LabeledCombo(
            self,
            [(k, _LINE_LABELS[k]) for k in LINE_METHODS],
            self._on_line,
            initial=app.state.line,
        )
        self.add(self._line, label="Reference line")

        self._envelope = LabeledCombo(
            self,
            [(k, _ENVELOPE_LABELS[k]) for k in ENVELOPE_METHODS],
            self._on_envelope,
            initial=app.state.envelope,
        )
        self.add(self._envelope, label="Envelope")

        self._alpha = SliderRow(
            self,
            "alpha",
            app.state.alpha,
            0.001,
            0.30,
            0.001,
            "envelope significance level",
            self._on_alpha,
        )
        self.add(self._alpha, span=True)

        boot = ttk.Frame(self)
        self._reps = tk.IntVar(value=app.state.bootstrap_reps)
        self._seed = tk.IntVar(value=app.state.random_state)
        ttk.Label(boot, text="bootstrap reps").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            boot, from_=50, to=20000, increment=50, textvariable=self._reps, width=8,
            command=self._on_bootstrap,
        ).grid(row=0, column=1, padx=(4, 10))
        ttk.Label(boot, text="seed").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(
            boot, from_=0, to=99999, textvariable=self._seed, width=7,
            command=self._on_bootstrap,
        ).grid(row=0, column=3, padx=(4, 0))
        self.add(boot, span=True)

        self.separator()

        toggles = ttk.Frame(self)
        self._standardize = tk.BooleanVar(value=app.state.standardize)
        self._detrend = tk.BooleanVar(value=app.state.detrend)
        self._equal = tk.BooleanVar(value=app.state.options.equal_aspect)
        ttk.Checkbutton(
            toggles, text="Standardise axes", variable=self._standardize,
            command=self._on_toggle,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            toggles, text="Detrend", variable=self._detrend, command=self._on_toggle
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Checkbutton(
            toggles, text="Equal aspect", variable=self._equal, command=self._on_toggle
        ).grid(row=1, column=0, sticky="w")
        self.add(toggles, span=True)

        display = ttk.Frame(self)
        self._legend = tk.BooleanVar(value=app.state.options.show_legend)
        self._annotate = tk.BooleanVar(value=app.state.options.annotate)
        self._flag = tk.BooleanVar(value=app.state.options.mark_outliers)
        ttk.Checkbutton(
            display, text="Legend", variable=self._legend, command=self._on_toggle
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            display, text="Statistics box", variable=self._annotate, command=self._on_toggle
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Checkbutton(
            display, text="Flag outliers", variable=self._flag, command=self._on_toggle
        ).grid(row=1, column=0, sticky="w")
        self.add(display, span=True)

        self.separator()

        self._theme = LabeledCombo(
            self,
            [(k, t.label) for k, t in THEMES.items()],
            self._on_theme,
            initial=app.state.theme_key,
        )
        self.add(self._theme, label="Theme")

        self._geometry = ttk.Label(self, font=("Consolas", 8), foreground=PALETTE["accent"])
        self.add(self._geometry, span=True)
        self._theme_note = self.note("")
        self._refresh_theme_labels()

    def _on_line(self, key: str) -> None:
        self.app.state.line = key
        self._line.set_key(key)
        self.app.request_update()

    def _on_envelope(self, key: str) -> None:
        self.app.state.envelope = key
        self._envelope.set_key(key)
        self.app.request_update()

    def _on_alpha(self, _name: str, value: float) -> None:
        self.app.state.alpha = float(value)
        self.app.request_update()

    def _on_bootstrap(self) -> None:
        try:
            self.app.state.bootstrap_reps = max(50, int(self._reps.get()))
            self.app.state.random_state = int(self._seed.get())
        except (tk.TclError, ValueError):
            return
        if self.app.state.envelope == "bootstrap":
            self.app.request_update()

    def _on_toggle(self) -> None:
        state = self.app.state
        state.standardize = bool(self._standardize.get())
        state.detrend = bool(self._detrend.get())
        state.options = replace_options(
            state.options,
            equal_aspect=bool(self._equal.get()),
            show_legend=bool(self._legend.get()),
            annotate=bool(self._annotate.get()),
            mark_outliers=bool(self._flag.get()),
        )
        self.app.request_update()

    def _on_theme(self, key: str) -> None:
        self.app.state.theme_key = key
        self._theme.set_key(key)
        self._refresh_theme_labels()
        self.app.request_update()

    def _refresh_theme_labels(self) -> None:
        theme = get_theme(self.app.state.theme_key)
        self._geometry.configure(text=theme.geometry_note())
        self._theme_note.configure(text=theme.notes)

    def sync(self) -> None:
        """Push the current state out to the widgets."""
        state = self.app.state
        self._line.set_key(state.line)
        self._envelope.set_key(state.envelope)
        self._theme.set_key(state.theme_key)
        self._reps.set(state.bootstrap_reps)
        self._seed.set(state.random_state)
        self._standardize.set(state.standardize)
        self._detrend.set(state.detrend)
        self._equal.set(state.options.equal_aspect)
        self._legend.set(state.options.show_legend)
        self._annotate.set(state.options.annotate)
        self._flag.set(state.options.mark_outliers)
        self._refresh_theme_labels()


def replace_options(options, **changes):
    """Return a copy of a :class:`~wedgelab.plot.PlotOptions` with changes."""
    from dataclasses import replace

    return replace(options, **changes)


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


class KnowledgePanel(Section):
    """Search the knowledge base and pull formulas into the workbench."""

    def __init__(self, parent: tk.Misc, app: "WedgeLabApp") -> None:
        super().__init__(parent, "5.  Knowledge base")
        self.app = app

        self._query = ttk.Entry(self)
        self.add(self._query, label="Search")
        self._query.bind("<KeyRelease>", lambda _e: self.refresh())

        self._tree = ttk.Treeview(self, show="tree", height=11, selectmode="browse")
        self.add(self._tree, span=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        self._detail = tk.Text(
            self, height=9, wrap="word", font=("Segoe UI", 8), relief="flat",
            background=PALETTE["bg"], borderwidth=0,
        )
        self.add(self._detail, span=True)
        self._detail.configure(state="disabled")

        actions = ttk.Frame(self)
        self._load = ttk.Button(
            actions, text="Load into workbench", command=self._on_load, state="disabled"
        )
        self._load.pack(side="left")
        self._scratch = ttk.Button(
            actions, text="Open in scratchpad", command=self._on_scratch, state="disabled"
        )
        self._scratch.pack(side="left", padx=(6, 0))
        self.add(actions, span=True)

        self._keys: dict[str, str] = {}
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the tree from the current search text."""
        self._tree.delete(*self._tree.get_children())
        self._keys.clear()
        matches = KNOWLEDGE.search(self._query.get())
        by_category: dict[str, list] = {}
        for entry in matches:
            by_category.setdefault(entry.category, []).append(entry)
        for category in KNOWLEDGE.categories():
            entries = by_category.get(category)
            if not entries:
                continue
            parent = self._tree.insert("", "end", text=category, open=True)
            for entry in entries:
                mark = "  f(x)" if entry.has_formula else ""
                node = self._tree.insert("", "end", text=f"    {entry.name}{mark}")
                self._tree.move(node, parent, "end")
                self._keys[node] = entry.key

    def _selected_key(self) -> str | None:
        selection = self._tree.selection()
        if not selection:
            return None
        return self._keys.get(selection[0])

    def _on_select(self, _event: tk.Event) -> None:
        key = self._selected_key()
        if key is None:
            self._load.configure(state="disabled")
            self._scratch.configure(state="disabled")
            self._write_detail("")
            return
        entry = KNOWLEDGE.get(key)
        parts = [entry.name, "", entry.summary]
        if entry.formula is not None:
            parts += ["", f"    {entry.formula.pretty()}"]
        if entry.when_to_use:
            parts += ["", "WHEN TO USE", entry.when_to_use]
        if entry.notes:
            parts += ["", "NOTES", entry.notes]
        parts += ["", "SOURCE", entry.citation]
        self._write_detail("\n".join(parts))

        is_position = entry.category == "Plotting positions" and entry.has_formula
        self._load.configure(state="normal" if is_position else "disabled")
        self._scratch.configure(state="normal" if entry.has_formula else "disabled")

    def _write_detail(self, text: str) -> None:
        self._detail.configure(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.insert("1.0", text)
        self._detail.configure(state="disabled")

    def _on_load(self) -> None:
        key = self._selected_key()
        if key is None:
            return
        entry = KNOWLEDGE.get(key)
        if entry.formula is None:
            return
        self.app.state.position_source = key
        self.app.formula_panel.rebuild(entry.formula)
        self.app.formula_panel.select_preset(key)
        self.app.request_update()
        self.app.status.ok(f"loaded {entry.name} into the workbench")

    def _on_scratch(self) -> None:
        key = self._selected_key()
        if key is None:
            return
        entry = KNOWLEDGE.get(key)
        if entry.formula is None:
            return
        self.app.open_scratchpad(entry.formula)
