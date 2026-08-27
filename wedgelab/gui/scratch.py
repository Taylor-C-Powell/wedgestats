"""A general formula scratchpad, independent of the Q-Q workflow.

The Q-Q workbench binds one formula to one slot with fixed inputs.  This
window is the general case: take any expression, sweep one symbol across a
range, put every other free symbol on a slider, and watch the curve.  It is
where a formula gets developed before it is put to work.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from wedgelab.formula import Formula, FormulaError, Parameter
from wedgelab.theme import applied, get_theme

from wedgelab.gui.widgets import PALETTE, MathPreview, SliderRow, StatusLine

__all__ = ["ScratchWindow"]


class ScratchWindow(tk.Toplevel):
    """Sweep one symbol of a formula and plot the result.

    Parameters
    ----------
    parent : tk.Misc
        Owning window.
    formula : Formula
        Starting expression; it is copied, never mutated.
    """

    def __init__(self, parent: tk.Misc, formula: Formula) -> None:
        super().__init__(parent)
        self.title(f"Scratchpad - {formula.name}")
        self.geometry("980x620")
        self.minsize(820, 520)

        self._formula = formula
        self._bindings: dict[str, float] = dict(formula.defaults())
        self._sliders: list[SliderRow] = []
        self._scalars: dict[str, ttk.Entry] = {}
        self._debounce: str | None = None

        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="Expression", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self._entry = tk.Text(left, height=4, width=44, wrap="word", font=("Consolas", 10))
        self._entry.pack(fill="x", pady=(2, 4))
        self._entry.insert("1.0", formula.expression)
        self._entry.bind("<KeyRelease>", self._on_typed)

        self._status = StatusLine(left, font=("Segoe UI", 8))
        self._status.pack(anchor="w", fill="x")

        self._preview = MathPreview(left, height_px=64)
        self._preview.pack(fill="x", pady=(4, 6))

        sweep = ttk.LabelFrame(left, text="Sweep variable", padding=8)
        sweep.pack(fill="x", pady=(0, 6))
        self._sweep_var = tk.StringVar()
        self._sweep = ttk.Combobox(sweep, textvariable=self._sweep_var, state="readonly", width=10)
        self._sweep.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self._sweep.bind("<<ComboboxSelected>>", lambda _e: self._recompute())

        self._from = self._numeric_entry(sweep, "from", 0.0, row=1, column=0)
        self._to = self._numeric_entry(sweep, "to", 1.0, row=1, column=2)
        self._points = self._numeric_entry(sweep, "points", 200, row=2, column=0)
        sweep.columnconfigure(1, weight=1)
        sweep.columnconfigure(3, weight=1)

        self._scalar_host = ttk.LabelFrame(left, text="Other inputs", padding=8)
        self._scalar_host.pack(fill="x", pady=(0, 6))

        self._slider_host = ttk.LabelFrame(left, text="Parameters", padding=8)
        self._slider_host.pack(fill="x", pady=(0, 6))

        buttons = ttk.Frame(left)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Apply", command=self._apply, width=8).pack(side="left")
        ttk.Button(buttons, text="Simplify", command=self._simplify, width=9).pack(
            side="left", padx=(4, 0)
        )
        ttk.Button(buttons, text="Derivative", command=self._derivative, width=10).pack(
            side="left", padx=(4, 0)
        )

        self._readout = tk.Text(
            left, height=6, wrap="none", font=("Consolas", 8), relief="flat",
            background=PALETTE["bg"], borderwidth=0,
        )
        self._readout.pack(fill="both", expand=True, pady=(8, 0))
        self._readout.configure(state="disabled")

        self._figure = Figure(figsize=(6.0, 4.4), dpi=100)
        self._canvas = FigureCanvasTkAgg(self._figure, master=right)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        self._rebuild_controls()
        self._recompute()

    # -- construction helpers ---------------------------------------------

    @staticmethod
    def _numeric_entry(
        parent: tk.Misc, label: str, value: float, row: int, column: int
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 4))
        entry = ttk.Entry(parent, width=9, justify="right", font=("Consolas", 9))
        entry.insert(0, str(value))
        entry.grid(row=row, column=column + 1, sticky="ew", pady=1)
        return entry

    def _rebuild_controls(self) -> None:
        """Recreate sweep options, scalar inputs, and parameter sliders."""
        symbols = sorted(self._formula.symbols)
        params = set(self._formula.parameter_names)
        inputs = [s for s in symbols if s not in params]

        self._sweep.configure(values=inputs or ["(none)"])
        if self._sweep_var.get() not in inputs:
            self._sweep_var.set(inputs[0] if inputs else "(none)")

        for child in self._scalar_host.winfo_children():
            child.destroy()
        self._scalars = {}
        for row, name in enumerate(inputs):
            if name == self._sweep_var.get():
                continue
            ttk.Label(self._scalar_host, text=name, font=("Consolas", 9), width=6).grid(
                row=row, column=0, sticky="w"
            )
            entry = ttk.Entry(self._scalar_host, width=12, justify="right", font=("Consolas", 9))
            entry.insert(0, str(self._bindings.get(name, 1.0)))
            entry.grid(row=row, column=1, sticky="ew", pady=1)
            entry.bind("<Return>", lambda _e: self._recompute())
            entry.bind("<FocusOut>", lambda _e: self._recompute())
            self._scalars[name] = entry
        self._scalar_host.columnconfigure(1, weight=1)

        for slider in self._sliders:
            slider.destroy()
        self._sliders = []
        for param in self._formula.parameters:
            row = SliderRow(
                self._slider_host,
                param.name,
                self._bindings.get(param.name, param.default),
                param.lower,
                param.upper,
                param.step,
                param.description,
                self._on_slider,
            )
            row.pack(fill="x", expand=True)
            self._sliders.append(row)

        self._preview.show(self._formula.to_latex())

    # -- editing ----------------------------------------------------------

    def _on_typed(self, _event: tk.Event) -> None:
        if self._debounce is not None:
            self.after_cancel(self._debounce)
        self._debounce = self.after(400, self._apply)

    def _on_slider(self, name: str, value: float) -> None:
        self._bindings[name] = value
        self._recompute()

    def _apply(self) -> None:
        """Adopt the typed expression, inventing sliders for new symbols."""
        self._debounce = None
        text = self._entry.get("1.0", "end").strip()
        if not text or text == self._formula.expression:
            return
        try:
            probe = Formula(name="probe", expression=text)
        except FormulaError as exc:
            self._status.error(str(exc))
            return

        known = {p.name: p for p in self._formula.parameters}
        sweep = self._sweep_var.get()
        scalars = set(self._scalars)
        parameters = tuple(
            known[name]
            for name in sorted(probe.symbols)
            if name in known and name != sweep and name not in scalars
        )
        try:
            self._formula = Formula(
                name=f"{self._formula.name} (edited)",
                expression=text,
                lhs=self._formula.lhs,
                description=self._formula.description,
                parameters=parameters,
                derived_from=self._formula.derived_from or self._formula.name,
            )
        except FormulaError as exc:
            self._status.error(str(exc))
            return
        self._rebuild_controls()
        self._recompute()

    def _simplify(self) -> None:
        try:
            simplified = self._formula.simplified()
        except FormulaError as exc:
            self._status.error(str(exc))
            return
        self._entry.delete("1.0", "end")
        self._entry.insert("1.0", simplified.expression)
        self._formula = simplified
        self._rebuild_controls()
        self._recompute()
        self._status.ok("simplified")

    def _derivative(self) -> None:
        target = self._sweep_var.get()
        if target in ("", "(none)"):
            self._status.error("choose a sweep variable to differentiate against")
            return
        try:
            derived = self._formula.derivative(target)
        except FormulaError as exc:
            self._status.error(str(exc))
            return
        # Formula.derivative already keeps only the parameters that survive
        # differentiation, so the result can be adopted as-is.
        self._entry.delete("1.0", "end")
        self._entry.insert("1.0", derived.expression)
        self._formula = derived
        self._rebuild_controls()
        self._recompute()
        self._status.ok(f"differentiated with respect to {target}")

    # -- evaluation -------------------------------------------------------

    def _sweep_values(self) -> np.ndarray | None:
        try:
            lo = float(self._from.get())
            hi = float(self._to.get())
            count = int(float(self._points.get()))
        except ValueError:
            self._status.error("sweep range must be numeric")
            return None
        if count < 2:
            self._status.error("sweep needs at least two points")
            return None
        if hi <= lo:
            self._status.error("sweep 'to' must exceed 'from'")
            return None
        return np.linspace(lo, hi, min(count, 20000))

    def _recompute(self) -> None:
        """Evaluate the formula over the sweep and redraw."""
        sweep = self._sweep_var.get()
        bindings = dict(self._bindings)
        for name, entry in self._scalars.items():
            try:
                bindings[name] = float(entry.get())
            except ValueError:
                self._status.error(f"input '{name}' must be numeric")
                return

        if sweep in ("", "(none)"):
            try:
                value = self._formula.evaluate(**bindings)
            except FormulaError as exc:
                self._status.error(str(exc))
                return
            self._status.ok("evaluated")
            self._write_readout(f"{self._formula.lhs} = {np.asarray(value).ravel()[:8]}")
            self._draw(None, None, sweep)
            return

        x = self._sweep_values()
        if x is None:
            return
        bindings[sweep] = x
        try:
            y = np.asarray(self._formula.evaluate(**bindings), dtype=float)
        except FormulaError as exc:
            self._status.error(str(exc))
            return
        y = np.broadcast_to(y, x.shape).astype(float, copy=False)

        finite = np.isfinite(y)
        if not finite.any():
            self._status.error("every value is non-finite over this range")
            self._draw(None, None, sweep)
            return
        if not finite.all():
            self._status.warn(f"{int(np.sum(~finite))} of {y.size} values are non-finite")
        else:
            self._status.ok(
                f"range [{np.min(y):.6g}, {np.max(y):.6g}] over {sweep} in "
                f"[{x[0]:.6g}, {x[-1]:.6g}]"
            )

        rows = [f"{sweep:>12s}  {self._formula.lhs}"]
        idx = np.linspace(0, x.size - 1, min(8, x.size)).astype(int)
        rows += [f"{x[k]:12.6g}  {y[k]:.8g}" for k in idx]
        self._write_readout("\n".join(rows))
        self._draw(x, y, sweep)

    def _write_readout(self, text: str) -> None:
        self._readout.configure(state="normal")
        self._readout.delete("1.0", "end")
        self._readout.insert("1.0", text)
        self._readout.configure(state="disabled")

    def _draw(self, x: np.ndarray | None, y: np.ndarray | None, sweep: str) -> None:
        theme = get_theme("screen")
        with applied(theme):
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            if x is not None and y is not None:
                ax.plot(x, y, color=theme.color("point"), linewidth=1.6)
                ax.set_xlabel(sweep)
                ax.set_ylabel(self._formula.lhs)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(True, color=theme.color("grid"), linewidth=0.4)
            else:
                ax.axis("off")
                ax.text(
                    0.5, 0.5, "no sweep variable", ha="center", va="center",
                    color=theme.color("annotation"),
                )
            self._figure.tight_layout(pad=0.6)
        self._canvas.draw_idle()
