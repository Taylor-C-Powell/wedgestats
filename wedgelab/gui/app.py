"""The WedgeLab main window.

One window, one recompute path.  Panels mutate
:class:`~wedgelab.gui.state.AppState` and call :meth:`WedgeLabApp.request_update`;
that method debounces, freezes the spec, recomputes, and redraws.  No panel
ever touches the figure, which is why the on-screen figure and the exported
file cannot drift apart -- they are the same render call.

The application deliberately *holds* a :class:`tkinter.Tk` rather than
subclassing it, so that ``app.state`` can name the workbench state without
shadowing ``Tk.state``.
"""

from __future__ import annotations

import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from wedgelab.export import read_session, save_figure, to_script, write_session
from wedgelab.formula import Formula, FormulaError
from wedgelab.models import FitError
from wedgelab.diagnostics import compute_ecdf, compute_pp
from wedgelab.plot import render_diagnostic
from wedgelab.qq import QQResult, compute, resolve_envelope

from wedgelab.gui.panels import (
    DataPanel,
    FormulaPanel,
    KnowledgePanel,
    ModelPanel,
    PresentationPanel,
)
from wedgelab.gui.scratch import ScratchWindow
from wedgelab.gui.state import AppState
from wedgelab.gui.widgets import PALETTE, ScrollableFrame, StatusLine

__all__ = ["WedgeLabApp", "launch"]

_DEBOUNCE_MS = 90


class WedgeLabApp:
    """The statistician's workbench, with a live Q-Q plot as its worked example."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("WedgeLab - Statistician's Workbench")
        self.root.geometry("1500x940")
        self.root.minsize(1180, 760)

        self.state = AppState()
        self.last_result: QQResult | None = None
        self._pending: str | None = None
        self._busy = False

        self._configure_style()
        self._build_menu()
        self._build_layout()
        self.request_update(immediate=True)

    def run(self) -> None:
        """Enter the Tk event loop."""
        self.root.mainloop()

    # -- construction -----------------------------------------------------

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        names = style.theme_names()
        if "vista" in names:
            style.theme_use("vista")
        elif "clam" in names:
            style.theme_use("clam")
        self.root.configure(background=PALETTE["bg"])
        style.configure("TLabelframe", background=PALETTE["bg"])
        style.configure("TLabelframe.Label", font=("Segoe UI", 9, "bold"))
        style.configure("TFrame", background=PALETTE["bg"])
        style.configure("TLabel", background=PALETTE["bg"])
        style.configure("TCheckbutton", background=PALETTE["bg"])

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open session...", command=self._open_session)
        file_menu.add_command(label="Save session...", command=self._save_session)
        file_menu.add_separator()
        file_menu.add_command(label="Export figure...", command=self._export_figure)
        file_menu.add_command(
            label="Export reproducible script...", command=self._export_script
        )
        file_menu.add_command(label="Copy caption", command=self._copy_caption)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        tools = tk.Menu(menubar, tearoff=0)
        tools.add_command(
            label="Formula scratchpad", command=lambda: self.open_scratchpad(None)
        )
        tools.add_command(label="Full diagnostics", command=self._show_diagnostics)
        tools.add_command(label="Figure caption", command=self._show_caption)
        menubar.add_cascade(label="Tools", menu=tools)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="How to use this", command=self._show_guide)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.configure(menu=menubar)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        # -- control column ---------------------------------------------
        scroller = ScrollableFrame(outer, width=400)
        scroller.pack(side="left", fill="y")
        column = scroller.interior
        column.columnconfigure(0, weight=1)

        self.data_panel = DataPanel(column, self)
        self.model_panel = ModelPanel(column, self)
        self.formula_panel = FormulaPanel(column, self)
        self.presentation_panel = PresentationPanel(column, self)
        self.knowledge_panel = KnowledgePanel(column, self)
        panels = (
            self.data_panel,
            self.model_panel,
            self.formula_panel,
            self.presentation_panel,
            self.knowledge_panel,
        )
        for row, panel in enumerate(panels):
            panel.grid(row=row, column=0, sticky="ew", pady=(0, 8), padx=(0, 8))

        # -- figure and readout ------------------------------------------
        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        figure_frame = ttk.Frame(right)
        figure_frame.pack(fill="both", expand=True)
        self._canvas_host = tk.Frame(figure_frame, background="white")
        self._canvas_host.pack(fill="both", expand=True)
        self._canvas: FigureCanvasTkAgg | None = None
        self._toolbar: NavigationToolbar2Tk | None = None

        readout = ttk.LabelFrame(right, text="Diagnostics", padding=(8, 4, 8, 6))
        readout.pack(fill="x", pady=(8, 0))
        self._diagnostics = tk.Text(
            readout, height=8, wrap="word", font=("Consolas", 9), relief="flat",
            background=PALETTE["bg"], borderwidth=0,
        )
        self._diagnostics.pack(fill="both", expand=True)
        self._diagnostics.configure(state="disabled")

        bar = ttk.Frame(right)
        bar.pack(fill="x", pady=(6, 0))
        ttk.Button(bar, text="Export figure", command=self._export_figure).pack(side="left")
        ttk.Button(bar, text="Export script", command=self._export_script).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(bar, text="Copy caption", command=self._copy_caption).pack(
            side="left", padx=(6, 0)
        )
        self.status = StatusLine(bar, font=("Segoe UI", 9))
        self.status.pack(side="left", fill="x", expand=True, padx=(14, 0))

    # -- the single recompute path ----------------------------------------

    def request_update(self, immediate: bool = False) -> None:
        """Schedule a recompute, coalescing rapid changes such as slider drags."""
        if self._pending is not None:
            self.root.after_cancel(self._pending)
            self._pending = None
        if immediate:
            self._update()
        else:
            self._pending = self.root.after(_DEBOUNCE_MS, self._update)

    _COMPUTERS = {"qq": compute, "pp": compute_pp, "ecdf": compute_ecdf}

    @staticmethod
    def _readout(result) -> list[str]:
        """Diagnostic lines, wherever a given result type keeps them."""
        source = getattr(result, "diagnostics", result)
        return list(source.lines())

    @staticmethod
    def _headline(result) -> str:
        """One-line status, using whatever summary statistic the type has."""
        d = getattr(result, "diagnostics", result)
        n = getattr(result, "n", None) or getattr(d, "n", 0)
        for attr, name in (("ppcc", "r"), ("correlation", "r"), ("ks_statistic", "D")):
            if hasattr(d, attr):
                return f"n={n}   {name}={getattr(d, attr):.5f}"
        return f"n={n}"

    def _update(self) -> None:
        self._pending = None
        if self._busy:
            return
        self._busy = True
        try:
            spec = self.state.build_spec()
            if getattr(spec, "bootstrap_reps", None) and resolve_envelope(
                spec.envelope, getattr(spec, "fit_method", "mle")
            ) == "bootstrap":
                self.status.info(f"running {spec.bootstrap_reps} bootstrap replicates...")
                self.root.update_idletasks()
            result = self._COMPUTERS[self.state.figure_type](spec)
        except (FormulaError, FitError, ValueError) as exc:
            self.status.error(str(exc))
            return
        except Exception as exc:
            self.status.error(f"unexpected failure: {exc}")
            traceback.print_exc()
            return
        finally:
            self._busy = False

        self.last_result = result
        self._redraw(result)
        self._write_diagnostics(result)
        self.model_panel.show_fit(result.fit.summary())
        if result.warnings:
            self.status.warn(result.warnings[0])
        else:
            self.status.ok(self._headline(result))

    def _redraw(self, result: QQResult) -> None:
        """Replace the embedded figure with a freshly rendered one."""
        figure = render_diagnostic(result, self.state.theme_key, self.state.options)
        if self._toolbar is not None:
            self._toolbar.destroy()
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
        self._canvas = FigureCanvasTkAgg(figure, master=self._canvas_host)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._toolbar = NavigationToolbar2Tk(
            self._canvas, self._canvas_host, pack_toolbar=False
        )
        self._toolbar.update()
        self._toolbar.pack(side="bottom", fill="x")
        self._canvas.draw()

    def _write_diagnostics(self, result) -> None:
        lines = self._readout(result)
        lines.append(f"model: {result.fit.summary()}")
        if hasattr(result.spec, "positions_summary"):
            lines.append(f"positions: {result.spec.positions_summary()}")
        lines.extend(f"note: {warning}" for warning in result.warnings)
        self._diagnostics.configure(state="normal")
        self._diagnostics.delete("1.0", "end")
        self._diagnostics.insert("1.0", "\n".join(lines))
        self._diagnostics.configure(state="disabled")

    # -- dialogs ----------------------------------------------------------

    def show_text(self, title: str, body: str) -> None:
        """Open a read-only text window."""
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("780x540")
        text = tk.Text(window, wrap="word", font=("Consolas", 9), padx=12, pady=10)
        scroll = ttk.Scrollbar(window, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        text.insert("1.0", body)
        text.configure(state="disabled")

    def open_scratchpad(self, formula: Formula | None) -> None:
        """Open the general formula scratchpad."""
        ScratchWindow(self.root, formula or self.state.position)

    def _show_diagnostics(self) -> None:
        if self.last_result is None:
            return
        result = self.last_result
        body = "\n".join(
            result.diagnostics.lines()
            + ["", f"model: {result.fit.summary()}"]
            + [f"note: {note}" for note in result.fit.notes]
            + ["", "CAPTION", result.caption()]
        )
        self.show_text("Full diagnostics", body)

    def _show_caption(self) -> None:
        if self.last_result is not None:
            self.show_text("Figure caption", self.last_result.caption())

    def _copy_caption(self) -> None:
        if self.last_result is None:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.last_result.caption())
        self.status.ok("caption copied to the clipboard")

    def _show_guide(self) -> None:
        self.show_text("How to use this", GUIDE)

    # -- file actions -----------------------------------------------------

    def _export_figure(self) -> None:
        if self.last_result is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export figure",
            defaultextension=".pdf",
            filetypes=[
                ("PDF (vector)", "*.pdf"),
                ("SVG (vector)", "*.svg"),
                ("EPS (vector)", "*.eps"),
                ("PNG", "*.png"),
                ("TIFF", "*.tif"),
            ],
        )
        if not path:
            return
        suffix = Path(path).suffix.lstrip(".").lower() or "pdf"
        try:
            figure = render_diagnostic(self.last_result, self.state.theme_key, self.state.options)
            written = save_figure(figure, path, formats=(suffix,))
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status.ok(f"wrote {written[0].name}")

    def _export_script(self) -> None:
        if self.last_result is None:
            return
        if self.state.figure_type != "qq":
            messagebox.showinfo(
                "Not available yet",
                "Script export currently covers the Q-Q plot only. Switch the "
                "figure type back to Q-Q, or use Save session to record this "
                "specification.",
            )
            return
        path = filedialog.asksaveasfilename(
            title="Export reproducible script",
            defaultextension=".py",
            filetypes=[("Python", "*.py")],
        )
        if not path:
            return
        try:
            source = to_script(
                self.last_result,
                self.state.theme_key,
                self.state.options,
                stem=Path(path).with_suffix("").name,
            )
            Path(path).write_text(source, encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status.ok(f"wrote {Path(path).name}")

    def _save_session(self) -> None:
        if self.state.figure_type != "qq":
            messagebox.showinfo(
                "Not available yet",
                "Session files currently record a Q-Q specification only.",
            )
            return
        path = filedialog.asksaveasfilename(
            title="Save session", defaultextension=".json", filetypes=[("JSON", "*.json")]
        )
        if not path:
            return
        try:
            written = write_session(
                path, self.state.to_spec(), self.state.theme_key, self.state.options
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.status.ok(f"wrote {written.name}")

    def _open_session(self) -> None:
        path = filedialog.askopenfilename(
            title="Open session", filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            spec, theme_key, options = read_session(path)
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))
            return
        self.state.adopt(spec, theme_key, options)
        self.data_panel.sync()
        self.model_panel.sync()
        self.presentation_panel.sync()
        self.formula_panel.rebuild(spec.position, dict(spec.position_bindings))
        self.request_update(immediate=True)
        self.status.ok(f"loaded {Path(path).name}")


GUIDE = """WEDGELAB - STATISTICIAN'S WORKBENCH

The window is one long argument, read top to bottom on the left.

1.  DATA
    Pick a synthetic sample or load a column from a CSV, TSV, or text file.
    Each synthetic sample states what its Q-Q plot should look like, so you
    can check your reading of the plot against the truth that generated it.

2.  REFERENCE MODEL
    The distribution the sample is compared against, and how its parameters
    are estimated. Try 'Normal with 5% contamination' under maximum
    likelihood, then switch the estimator to robust: the line stops chasing
    the outliers and they appear outside the envelope where they belong.

3.  PLOTTING-POSITION WORKBENCH
    This is the formula engine. The expression maps rank i and sample size n
    to a probability. Every constant in it is a defensible choice somebody
    published, and the slider shows you what that choice costs.

    Type freely. Any symbol you introduce that is not i or n becomes a new
    slider, so an edit turns a fixed rule into a family you can explore.
    'Calculus' simplifies the expression and differentiates it with respect
    to each parameter; a derivative near zero means the parameter is not
    doing visible work. 'Functions' lists everything you may call, including
    exact quantile functions backed by wedgestats.

4.  LINE, ENVELOPE, AND THEME
    Four envelopes that mean genuinely different things:
      exact (Beta)   the order statistics' own distribution; exact, and
                     conservative when parameters were estimated
      asymptotic     the delta-method band; cheap, unreliable in the tails
      simultaneous   bounds the whole curve at once, which is what you need
                     to write 'consistent with normality' in a caption
      bootstrap      simulates and refits, so it accounts for the estimation
                     that the exact band ignores

    Themes fix the physical figure width in millimetres to the target
    journal's column, so the exported file drops in at 1:1 and the type
    stays the size you set it.

5.  KNOWLEDGE BASE
    Thirty-five citable entries. Plotting positions load straight into the
    workbench with their reference attached. Anything else opens in the
    scratchpad, where you can sweep one symbol and plot the result.

EXPORTING
    'Export figure' writes vector PDF, SVG, or EPS at the theme's exact size.
    'Export script' writes a standalone Python file that rebuilds the figure
    from scratch, plotting-position formula included -- put it in the
    supplementary material and the figure stops being a claim.
    'Copy caption' produces a caption that states every choice you made.

A WARNING WORTH REPEATING
    A pointwise envelope is not a simultaneous one. With n = 100 and
    alpha = 0.05, about five points falling outside a pointwise band is the
    expectation, not evidence of misfit.
"""


def launch() -> None:
    """Open the workbench."""
    WedgeLabApp().run()
