"""Small Tk widgets shared by the workbench panels.

Nothing here knows any statistics; these are the reusable pieces that keep
:mod:`wedgelab.gui.panels` readable.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, Sequence

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

__all__ = [
    "Section",
    "LabeledCombo",
    "SliderRow",
    "ScrollableFrame",
    "MathPreview",
    "StatusLine",
    "PALETTE",
]

# Muted, high-contrast interface palette.  Deliberately quiet: the figure is
# the only thing in the window allowed to be colourful.
PALETTE = {
    "bg": "#f4f5f7",
    "panel": "#ffffff",
    "ink": "#1b1f24",
    "muted": "#5c6672",
    "accent": "#1f3b57",
    "ok": "#1e7d44",
    "error": "#b02a1e",
    "warn": "#8a6100",
    "rule": "#d8dce1",
}


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable container.

    The control column is taller than most laptop screens once every panel is
    expanded, so it scrolls rather than truncating.
    """

    def __init__(self, parent: tk.Misc, width: int = 380) -> None:
        super().__init__(parent)
        self._canvas = tk.Canvas(
            self, borderwidth=0, highlightthickness=0, width=width, background=PALETTE["bg"]
        )
        self._bar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.interior = ttk.Frame(self._canvas)

        self._window = self._canvas.create_window((0, 0), window=self.interior, anchor="nw")
        self._canvas.configure(yscrollcommand=self._bar.set)

        self._canvas.pack(side="left", fill="both", expand=True)
        self._bar.pack(side="right", fill="y")

        self.interior.bind("<Configure>", self._on_interior)
        self._canvas.bind("<Configure>", self._on_canvas)
        self._canvas.bind("<Enter>", lambda _e: self._bind_wheel())
        self._canvas.bind("<Leave>", lambda _e: self._unbind_wheel())

    def _on_interior(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)

    def _bind_wheel(self) -> None:
        self._canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _unbind_wheel(self) -> None:
        self._canvas.unbind_all("<MouseWheel>")

    def _on_wheel(self, event: tk.Event) -> None:
        self._canvas.yview_scroll(int(-event.delta / 120), "units")


class Section(ttk.LabelFrame):
    """A titled group of controls."""

    def __init__(self, parent: tk.Misc, title: str) -> None:
        super().__init__(parent, text=title, padding=(9, 6, 9, 9))
        self.columnconfigure(1, weight=1)
        self._row = 0

    def next_row(self) -> int:
        """Return the next free grid row and advance the counter."""
        row = self._row
        self._row += 1
        return row

    def add(self, widget: tk.Widget, *, label: str = "", span: bool = False) -> int:
        """Grid *widget* into the section, optionally with a left-hand label."""
        row = self.next_row()
        if span or not label:
            widget.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        else:
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            widget.grid(row=row, column=1, sticky="ew", pady=2)
        return row

    def separator(self) -> None:
        """Draw a horizontal rule."""
        ttk.Separator(self, orient="horizontal").grid(
            row=self.next_row(), column=0, columnspan=2, sticky="ew", pady=(7, 5)
        )

    def note(self, text: str, *, wraplength: int = 330) -> ttk.Label:
        """Add a small muted explanatory label."""
        label = ttk.Label(
            self,
            text=text,
            wraplength=wraplength,
            justify="left",
            foreground=PALETTE["muted"],
            font=("Segoe UI", 8),
        )
        label.grid(row=self.next_row(), column=0, columnspan=2, sticky="w", pady=(2, 3))
        return label


class LabeledCombo(ttk.Combobox):
    """A read-only combobox that maps display labels to opaque keys."""

    def __init__(
        self,
        parent: tk.Misc,
        pairs: Sequence[tuple[str, str]],
        on_change: Callable[[str], None],
        initial: str | None = None,
    ) -> None:
        self._keys = [k for k, _ in pairs]
        self._labels = [v for _, v in pairs]
        self._on_change = on_change
        super().__init__(parent, values=self._labels, state="readonly", width=26)
        if initial is not None and initial in self._keys:
            self.current(self._keys.index(initial))
        elif self._labels:
            self.current(0)
        self.bind("<<ComboboxSelected>>", self._fire)

    def _fire(self, _event: tk.Event) -> None:
        self._on_change(self.key)

    @property
    def key(self) -> str:
        """Currently selected key."""
        idx = self.current()
        return self._keys[idx] if 0 <= idx < len(self._keys) else self._keys[0]

    def set_key(self, key: str) -> None:
        """Select *key* without firing the callback."""
        if key in self._keys:
            self.current(self._keys.index(key))

    def replace_pairs(self, pairs: Sequence[tuple[str, str]]) -> None:
        """Swap the whole option list."""
        self._keys = [k for k, _ in pairs]
        self._labels = [v for _, v in pairs]
        self.configure(values=self._labels)
        if self._labels:
            self.current(0)


class SliderRow(ttk.Frame):
    """A named slider with a live numeric entry, bounds, and a description.

    Dragging the slider and typing in the entry are kept in sync; both call
    back with the new value.
    """

    def __init__(
        self,
        parent: tk.Misc,
        name: str,
        value: float,
        lower: float,
        upper: float,
        step: float,
        description: str,
        on_change: Callable[[str, float], None],
    ) -> None:
        super().__init__(parent)
        self.columnconfigure(1, weight=1)
        self.name = name
        self._on_change = on_change
        self._lower = lower
        self._upper = upper
        self._step = step
        self._suppress = False

        ttk.Label(self, text=name, width=6, font=("Consolas", 9)).grid(
            row=0, column=0, sticky="w"
        )

        self._var = tk.DoubleVar(value=value)
        self._scale = ttk.Scale(
            self,
            from_=lower,
            to=upper,
            orient="horizontal",
            variable=self._var,
            command=self._on_slide,
        )
        self._scale.grid(row=0, column=1, sticky="ew", padx=(4, 6))
        if upper <= lower:
            self._scale.state(["disabled"])

        self._entry = ttk.Entry(self, width=9, justify="right", font=("Consolas", 9))
        self._entry.insert(0, self._format(value))
        self._entry.grid(row=0, column=2, sticky="e")
        self._entry.bind("<Return>", self._on_typed)
        self._entry.bind("<FocusOut>", self._on_typed)

        ttk.Label(
            self,
            text=f"[{lower:g}, {upper:g}]  {description}",
            font=("Segoe UI", 7),
            foreground=PALETTE["muted"],
            wraplength=320,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 4))

    def _format(self, value: float) -> str:
        digits = max(0, min(6, int(round(-math.log10(self._step))) + 1))
        return f"{value:.{digits}f}"

    def _quantise(self, value: float) -> float:
        snapped = round(value / self._step) * self._step
        return float(min(max(snapped, self._lower), self._upper))

    def _on_slide(self, raw: str) -> None:
        if self._suppress:
            return
        value = self._quantise(float(raw))
        self._suppress = True
        self._entry.delete(0, "end")
        self._entry.insert(0, self._format(value))
        self._suppress = False
        self._on_change(self.name, value)

    def _on_typed(self, _event: tk.Event) -> None:
        if self._suppress:
            return
        try:
            value = self._quantise(float(self._entry.get()))
        except ValueError:
            value = float(self._var.get())
        self._suppress = True
        self._var.set(value)
        self._entry.delete(0, "end")
        self._entry.insert(0, self._format(value))
        self._suppress = False
        self._on_change(self.name, value)

    @property
    def value(self) -> float:
        """Current value."""
        return float(self._var.get())


class MathPreview(ttk.Frame):
    """Renders a LaTeX string with matplotlib's mathtext.

    Mathtext covers ordinary expressions but not full LaTeX environments; when
    it cannot parse the string the widget falls back to showing the source,
    which is more useful than an exception.
    """

    def __init__(self, parent: tk.Misc, height_px: int = 58) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(3.6, height_px / 100.0), dpi=100)
        self._figure.patch.set_facecolor(PALETTE["panel"])
        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas.get_tk_widget().configure(height=height_px, background=PALETTE["panel"])
        self.show("")

    def show(self, latex: str) -> None:
        """Render *latex*, falling back to plain text when mathtext fails."""
        self._figure.clear()
        ax = self._figure.add_axes((0, 0, 1, 1))
        ax.axis("off")
        ax.set_facecolor(PALETTE["panel"])
        if not latex:
            self._canvas.draw_idle()
            return
        try:
            ax.text(
                0.5,
                0.5,
                f"${latex}$",
                ha="center",
                va="center",
                fontsize=13,
                color=PALETTE["ink"],
            )
            self._canvas.draw()
        except Exception:
            self._figure.clear()
            ax = self._figure.add_axes((0, 0, 1, 1))
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                latex if len(latex) < 70 else latex[:67] + "...",
                ha="center",
                va="center",
                fontsize=8,
                color=PALETTE["muted"],
                family="monospace",
            )
            self._canvas.draw_idle()


class StatusLine(ttk.Label):
    """A one-line status readout with severity colouring."""

    def __init__(self, parent: tk.Misc, **kwargs) -> None:
        super().__init__(parent, anchor="w", **kwargs)
        self.ok("")

    def ok(self, text: str) -> None:
        """Show a success or neutral message."""
        self.configure(text=text, foreground=PALETTE["ok"] if text else PALETTE["muted"])

    def warn(self, text: str) -> None:
        """Show a warning."""
        self.configure(text=text, foreground=PALETTE["warn"])

    def error(self, text: str) -> None:
        """Show an error."""
        self.configure(text=text, foreground=PALETTE["error"])

    def info(self, text: str) -> None:
        """Show a neutral message."""
        self.configure(text=text, foreground=PALETTE["muted"])


def grid_children(parent: tk.Misc, widgets: Iterable[tk.Widget]) -> None:
    """Stack *widgets* vertically in a single expanding column."""
    for row, widget in enumerate(widgets):
        widget.grid(row=row, column=0, sticky="ew", pady=(0, 8))
    parent.columnconfigure(0, weight=1)
