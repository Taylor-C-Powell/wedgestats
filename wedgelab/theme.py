"""Publication themes: journal geometry, typography, and colour.

"Publication quality" is mostly not an aesthetic judgement.  It is a set of
measurable constraints -- the figure is exactly one column wide, the smallest
type is at least seven points *after* the journal scales it, the file is
vector, the colours survive greyscale printing and the common forms of colour
vision deficiency.  A figure sized in pixels and scaled later fails all of
them at once.

Each :class:`Theme` therefore fixes the physical width in millimetres and
sizes type in points relative to it, so a figure exported at 89 mm is dropped
into a Nature column at 1:1 with 7 pt labels that stay 7 pt.

Column widths are taken from the publishers' current author guidelines; check
them against the target journal before submission, since they do change.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import matplotlib as mpl

__all__ = [
    "Theme",
    "THEMES",
    "theme_keys",
    "get_theme",
    "applied",
    "MM_PER_INCH",
]

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class Theme:
    """A complete, measurable specification for a publication figure.

    Attributes
    ----------
    key : str
        Stable identifier.
    label : str
        Display name.
    width_mm : float
        Physical figure width. This is the number that matters: the figure is
        placed at 1:1, never rescaled.
    aspect : float
        Height as a fraction of width.
    base_pt : float
        Body text size in points; tick labels sit slightly below it.
    family : str
        ``"serif"`` or ``"sans-serif"``.
    font_stack : tuple[str, ...]
        Preferred faces in order; the last is always an assured fallback.
    dpi : int
        Raster export resolution. Vector formats ignore it.
    spines : str
        ``"box"``, ``"lb"`` (left and bottom only), or ``"minimal"``.
    grid : bool
        Whether to draw a light grid.
    colors : dict[str, str]
        Semantic palette; see :data:`_BASE_COLORS` for the required keys.
    marker : str
        Matplotlib marker code for data points.
    marker_size : float
        Marker area in points squared.
    line_width : float
        Reference-line width in points.
    notes : str
        Guidance shown in the GUI.
    """

    key: str
    label: str
    width_mm: float
    aspect: float = 0.78
    base_pt: float = 8.0
    family: str = "sans-serif"
    font_stack: tuple[str, ...] = ("DejaVu Sans",)
    dpi: int = 600
    spines: str = "lb"
    grid: bool = False
    colors: dict[str, str] = field(default_factory=dict)
    marker: str = "o"
    marker_size: float = 9.0
    line_width: float = 0.9
    notes: str = ""

    # -- geometry ---------------------------------------------------------

    @property
    def width_in(self) -> float:
        """Figure width in inches."""
        return self.width_mm / MM_PER_INCH

    @property
    def height_in(self) -> float:
        """Figure height in inches."""
        return self.width_in * self.aspect

    def figsize(self) -> tuple[float, float]:
        """``(width, height)`` in inches, for ``plt.figure(figsize=...)``."""
        return (self.width_in, self.height_in)

    def geometry_note(self) -> str:
        """Human-readable statement of the physical size."""
        return (
            f"{self.width_mm:.0f} x {self.width_mm * self.aspect:.0f} mm "
            f"({self.width_in:.2f} x {self.height_in:.2f} in) at {self.base_pt:.0f} pt"
        )

    # -- style ------------------------------------------------------------

    def color(self, role: str) -> str:
        """Colour for a semantic role, falling back to the base palette."""
        return self.colors.get(role, _BASE_COLORS[role])

    def rc_params(self) -> dict[str, Any]:
        """Matplotlib rcParams implementing this theme."""
        tick_pt = self.base_pt - 0.5
        text = self.color("text")
        axis = self.color("axis")
        mathfont = "stix" if self.family == "serif" else "dejavusans"
        return {
            "figure.figsize": self.figsize(),
            "figure.dpi": 100,
            "savefig.dpi": self.dpi,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": self.family,
            f"font.{self.family}": list(self.font_stack),
            "font.size": self.base_pt,
            "mathtext.fontset": mathfont,
            "axes.titlesize": self.base_pt,
            "axes.labelsize": self.base_pt,
            "axes.titleweight": "regular",
            "axes.labelcolor": text,
            "axes.edgecolor": axis,
            "axes.linewidth": 0.6,
            "axes.facecolor": "white",
            "axes.grid": self.grid,
            "axes.axisbelow": True,
            "grid.color": self.color("grid"),
            "grid.linewidth": 0.4,
            "grid.alpha": 1.0,
            "xtick.labelsize": tick_pt,
            "ytick.labelsize": tick_pt,
            "xtick.color": axis,
            "ytick.color": axis,
            "xtick.labelcolor": text,
            "ytick.labelcolor": text,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.size": 1.4,
            "ytick.minor.size": 1.4,
            "legend.fontsize": tick_pt,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.borderaxespad": 0.3,
            "lines.linewidth": self.line_width,
            "lines.markersize": self.marker_size**0.5,
            "text.color": text,
            # Keep text as text in vector output so it stays editable and
            # searchable, which most journals require.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }


# ---------------------------------------------------------------------------
# Palette
#
# Point and band colours are chosen to hold up in greyscale and under
# deuteranopia: the band is a light neutral blue, the reference line is near
# black, and flagged points are a dark orange that stays distinguishable by
# lightness alone.
# ---------------------------------------------------------------------------

_BASE_COLORS: dict[str, str] = {
    "point": "#1f3b57",
    "point_edge": "#ffffff",
    "line": "#111111",
    "band": "#c9d8e6",
    "band_edge": "#8fa9c0",
    "outlier": "#b3520a",
    "text": "#111111",
    "axis": "#444444",
    "grid": "#e6e6e6",
    "annotation": "#555555",
}


def _palette(**overrides: str) -> dict[str, str]:
    """Base palette with selective overrides."""
    out = dict(_BASE_COLORS)
    out.update(overrides)
    return out


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

THEMES: dict[str, Theme] = {
    "nature": Theme(
        key="nature",
        label="Nature (single column, 89 mm)",
        width_mm=89.0,
        aspect=0.80,
        base_pt=7.0,
        family="sans-serif",
        font_stack=("Helvetica", "Arial", "DejaVu Sans"),
        dpi=600,
        spines="lb",
        grid=False,
        colors=_palette(),
        marker_size=7.0,
        line_width=0.8,
        notes="Nature asks for 89 mm single-column or 183 mm double-column "
        "figures, sans-serif labels, and a minimum type size of 5 pt. Seven "
        "point keeps you comfortably above the floor.",
    ),
    "nature_double": Theme(
        key="nature_double",
        label="Nature (double column, 183 mm)",
        width_mm=183.0,
        aspect=0.48,
        base_pt=7.0,
        family="sans-serif",
        font_stack=("Helvetica", "Arial", "DejaVu Sans"),
        dpi=600,
        spines="lb",
        colors=_palette(),
        marker_size=9.0,
        notes="Full-width variant. Keep the type size identical to the "
        "single-column theme so panels can sit side by side.",
    ),
    "science": Theme(
        key="science",
        label="Science (single column, 55 mm)",
        width_mm=55.0,
        aspect=0.90,
        base_pt=6.5,
        family="sans-serif",
        font_stack=("Helvetica", "Arial", "DejaVu Sans"),
        dpi=600,
        spines="lb",
        colors=_palette(),
        marker_size=5.0,
        line_width=0.7,
        notes="Science columns are narrow at 55 mm. Strip everything "
        "non-essential; at this width an annotation box will not fit.",
    ),
    "plos": Theme(
        key="plos",
        label="PLOS (132 mm)",
        width_mm=132.0,
        aspect=0.70,
        base_pt=9.0,
        family="sans-serif",
        font_stack=("Arial", "Helvetica", "DejaVu Sans"),
        dpi=300,
        spines="lb",
        colors=_palette(),
        marker_size=11.0,
        notes="PLOS accepts up to 190 mm wide and requires 300 dpi minimum "
        "for raster, with 8 to 12 pt type. Arial is explicitly recommended.",
    ),
    "ieee": Theme(
        key="ieee",
        label="IEEE (single column, 88.9 mm)",
        width_mm=88.9,
        aspect=0.76,
        base_pt=8.0,
        family="serif",
        font_stack=("Times New Roman", "Nimbus Roman", "DejaVu Serif"),
        dpi=600,
        spines="box",
        grid=True,
        colors=_palette(grid="#dddddd"),
        marker_size=9.0,
        notes="IEEE columns are 3.5 in. Times body text and a boxed axis with "
        "a light grid match the house style of most IEEE transactions.",
    ),
    "elsevier": Theme(
        key="elsevier",
        label="Elsevier (single column, 90 mm)",
        width_mm=90.0,
        aspect=0.78,
        base_pt=8.0,
        family="serif",
        font_stack=("Times New Roman", "Nimbus Roman", "DejaVu Serif"),
        dpi=600,
        spines="lb",
        colors=_palette(),
        notes="Elsevier single column is 90 mm, 1.5 column 140 mm, double "
        "190 mm. Submit vector EPS or PDF.",
    ),
    "apa": Theme(
        key="apa",
        label="APA manuscript (165 mm)",
        width_mm=165.0,
        aspect=0.62,
        base_pt=10.0,
        family="serif",
        font_stack=("Times New Roman", "Nimbus Roman", "DejaVu Serif"),
        dpi=300,
        spines="lb",
        colors=_palette(point="#333333", band="#d9d9d9", band_edge="#a0a0a0"),
        marker_size=14.0,
        line_width=1.0,
        notes="APA 7 wants a sans or serif face at a legible size, no "
        "gridlines, and greyscale-safe encoding. This preset is greyscale by "
        "construction.",
    ),
    "thesis": Theme(
        key="thesis",
        label="Thesis / report (140 mm)",
        width_mm=140.0,
        aspect=0.66,
        base_pt=9.5,
        family="serif",
        font_stack=("Times New Roman", "Nimbus Roman", "DejaVu Serif"),
        dpi=600,
        spines="lb",
        colors=_palette(),
        marker_size=12.0,
        notes="A comfortable width for a one-figure-per-page thesis chapter "
        "with generous margins.",
    ),
    "presentation": Theme(
        key="presentation",
        label="Presentation (200 mm)",
        width_mm=200.0,
        aspect=0.56,
        base_pt=14.0,
        family="sans-serif",
        font_stack=("Helvetica", "Arial", "DejaVu Sans"),
        dpi=200,
        spines="lb",
        colors=_palette(point="#1b4f72", band="#cfe2f3", outlier="#c0392b"),
        marker_size=26.0,
        line_width=1.8,
        notes="Large type and heavy strokes for projection. Not for "
        "submission; the aspect ratio is wrong for a journal column.",
    ),
    "screen": Theme(
        key="screen",
        label="Screen / exploration (170 mm)",
        width_mm=170.0,
        aspect=0.68,
        base_pt=10.0,
        family="sans-serif",
        font_stack=("DejaVu Sans",),
        dpi=150,
        spines="lb",
        grid=True,
        colors=_palette(),
        marker_size=16.0,
        notes="The working default while you explore. Switch to a journal "
        "theme before exporting anything you intend to submit.",
    ),
}


def theme_keys() -> tuple[str, ...]:
    """Available theme keys, in display order."""
    return tuple(THEMES)


def get_theme(key: str) -> Theme:
    """Look up a theme by key.

    Raises
    ------
    KeyError
        If no such theme exists.
    """
    try:
        return THEMES[key]
    except KeyError:
        raise KeyError(
            f"unknown theme '{key}'; choose from {', '.join(THEMES)}"
        ) from None


@contextmanager
def applied(theme: Theme) -> Iterator[Theme]:
    """Temporarily apply a theme's rcParams.

    Examples
    --------
    >>> with applied(get_theme("nature")) as t:
    ...     size = t.figsize()
    """
    with mpl.rc_context(theme.rc_params()):
        yield theme
