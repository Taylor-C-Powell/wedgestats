"""Render a :class:`~wedgelab.qq.QQResult` as a publication figure.

The renderer is deliberately separate from the computation.  Everything it
draws is already decided by the time it is called, so the same
:class:`~wedgelab.qq.QQResult` produces a byte-identical figure whether it is
drawn on the GUI canvas or written to a PDF by the exporter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from wedgelab.models import DISTRIBUTIONS
from wedgelab.qq import QQResult
from wedgelab.theme import Theme, applied, get_theme

__all__ = [
    "PlotOptions",
    "render",
    "render_diagnostic",
    "figure_for",
    "draw",
    "draw_pp",
    "draw_ecdf",
    "draw_two_sample",
]


@dataclass(frozen=True)
class PlotOptions:
    """Presentation choices that do not change any number in the figure.

    Attributes
    ----------
    title : str
        Axes title; empty for none (journals usually want none, because the
        caption carries the description).
    xlabel, ylabel : str
        Axis labels; empty strings fall back to sensible defaults.
    show_band : bool
        Draw the confidence envelope when the result carries one.
    show_line : bool
        Draw the reference line.
    show_legend : bool
        Draw a legend.
    legend_loc : str
        Matplotlib legend location.  The default, ``"best"``, lets matplotlib
        place it where it overlaps least, which matters because the shape of
        a Q-Q plot changes completely between raw and detrended views.
    annotate : bool
        Draw the diagnostic box inside the axes.
    annotation_fields : tuple[str, ...]
        Which diagnostics to include in that box.
    mark_outliers : bool
        Colour points that fall outside the envelope.
    label_outliers : int
        Label at most this many flagged points with their rank.
    equal_aspect : bool
        Force a 1:1 data aspect ratio, which makes departures from the line
        geometrically honest.
    zero_line : bool
        Draw a horizontal zero reference; only meaningful when detrended.
    """

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    show_band: bool = True
    show_line: bool = True
    show_legend: bool = True
    legend_loc: str = "best"
    annotate: bool = True
    annotation_fields: tuple[str, ...] = ("n", "ppcc", "slope")
    mark_outliers: bool = True
    label_outliers: int = 0
    equal_aspect: bool = False
    zero_line: bool = True


_ANNOTATION_LABELS: dict[str, str] = {
    "n": "n",
    "ppcc": "r",
    "slope": "slope",
    "intercept": "intercept",
    "r_squared": "R^2",
    "skewness": "g1",
    "kurtosis": "g2",
    "shapiro": "Shapiro-Wilk",
    "ks": "KS D",
    "anderson": "A^2",
}


def _annotation_text(result: QQResult, fields: tuple[str, ...]) -> str:
    """Assemble the in-axes diagnostic box."""
    d = result.diagnostics
    rows: list[str] = []
    for key in fields:
        if key == "n":
            rows.append(f"$n$ = {d.n}")
        elif key == "ppcc":
            rows.append(f"$r$ = {d.ppcc:.4f}")
        elif key == "slope":
            rows.append(f"slope = {d.slope:.3f}")
        elif key == "intercept":
            rows.append(f"intercept = {d.intercept:.3f}")
        elif key == "r_squared" and np.isfinite(d.r_squared):
            rows.append(f"$R^2$ = {d.r_squared:.4f}")
        elif key == "skewness":
            rows.append(f"$g_1$ = {d.skewness:+.3f}")
        elif key == "kurtosis":
            rows.append(f"$g_2$ = {d.kurtosis:+.3f}")
        elif key == "shapiro" and d.shapiro_w is not None:
            rows.append(f"$W$ = {d.shapiro_w:.4f}, $p$ = {d.shapiro_p:.3g}")
        elif key == "ks" and d.ks_d is not None:
            rows.append(f"$D$ = {d.ks_d:.4f}, $p$ = {d.ks_p:.3g}")
        elif key == "anderson" and d.anderson_a2 is not None:
            rows.append(f"$A^2$ = {d.anderson_a2:.3f}")
    return "\n".join(rows)


def _default_labels(result: QQResult) -> tuple[str, str]:
    """Axis labels that name the reference distribution honestly."""
    spec = result.spec
    label = DISTRIBUTIONS[spec.dist_key].label
    scale = " (standardised)" if spec.standardize else ""
    x = f"Theoretical quantiles, {label}{scale}"
    if spec.detrend:
        y = f"Sample quantiles minus reference line{scale}"
    else:
        y = f"Sample quantiles{scale}"
    return x, y


def _finite_band(
    lower: np.ndarray, upper: np.ndarray, sample: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Replace infinite bounds with a value beyond the plotted data.

    A simultaneous band is genuinely unbounded where ``p_i +/- d`` leaves the
    unit interval.  Extending it past the data preserves that meaning while
    keeping ``fill_between`` finite.
    """
    span = float(np.ptp(sample)) or 1.0
    floor = float(np.min(sample)) - span
    ceiling = float(np.max(sample)) + span
    lo = np.where(np.isfinite(lower), lower, floor)
    hi = np.where(np.isfinite(upper), upper, ceiling)
    return lo, hi


def _style_spines(ax: Axes, theme: Theme) -> None:
    """Apply the theme's spine treatment."""
    if theme.spines == "box":
        return
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if theme.spines == "minimal":
        ax.spines["left"].set_position(("outward", 3))
        ax.spines["bottom"].set_position(("outward", 3))


def draw(result: QQResult, ax: Axes, theme: Theme, options: PlotOptions) -> Axes:
    """Draw *result* onto an existing axes using *theme*.

    Assumes the theme's rcParams are already in force; use :func:`render`
    unless you are compositing several panels yourself.
    """
    handles: list[Line2D] = []

    x = result.theoretical
    y = result.sample

    # -- envelope ---------------------------------------------------------
    if options.show_band and result.lower is not None and result.upper is not None:
        lo, hi = _finite_band(result.lower, result.upper, y)
        ax.fill_between(
            x,
            lo,
            hi,
            facecolor=theme.color("band"),
            edgecolor="none",
            zorder=1,
            label=f"{100 * (1 - result.spec.alpha):.0f}% envelope",
        )
        ax.plot(x, lo, color=theme.color("band_edge"), linewidth=0.4, zorder=2)
        ax.plot(x, hi, color=theme.color("band_edge"), linewidth=0.4, zorder=2)
        handles.append(
            Line2D(
                [],
                [],
                color=theme.color("band"),
                linewidth=5,
                label=f"{100 * (1 - result.spec.alpha):.0f}% envelope",
            )
        )

    # -- reference line ---------------------------------------------------
    if options.show_line:
        if result.spec.detrend:
            if options.zero_line:
                ax.axhline(
                    0.0,
                    color=theme.color("line"),
                    linewidth=theme.line_width,
                    zorder=3,
                )
                handles.append(
                    Line2D([], [], color=theme.color("line"), label="Reference line")
                )
        else:
            lx, ly = result.line_points()
            ax.plot(
                lx,
                ly,
                color=theme.color("line"),
                linewidth=theme.line_width,
                zorder=3,
            )
            label = {
                "ols": "Least-squares line",
                "quartile": "Quartile line",
                "theoretical": "Identity line",
            }[result.spec.line]
            handles.append(Line2D([], [], color=theme.color("line"), label=label))

    # -- points -----------------------------------------------------------
    inside = ~result.outside if options.mark_outliers else np.ones_like(y, dtype=bool)
    ax.scatter(
        x[inside],
        y[inside],
        s=theme.marker_size,
        marker=theme.marker,
        facecolor=theme.color("point"),
        edgecolor=theme.color("point_edge"),
        linewidth=0.3,
        zorder=4,
        label=result.spec.label,
    )
    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker=theme.marker,
            markerfacecolor=theme.color("point"),
            markeredgecolor=theme.color("point_edge"),
            markersize=theme.marker_size**0.5,
            label=result.spec.label,
        )
    )

    n_out = int(np.sum(result.outside))
    if options.mark_outliers and n_out:
        ax.scatter(
            x[result.outside],
            y[result.outside],
            s=theme.marker_size * 1.25,
            marker=theme.marker,
            facecolor=theme.color("outlier"),
            edgecolor=theme.color("point_edge"),
            linewidth=0.3,
            zorder=5,
            label="Outside envelope",
        )
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=theme.marker,
                markerfacecolor=theme.color("outlier"),
                markeredgecolor=theme.color("point_edge"),
                markersize=theme.marker_size**0.5,
                label=f"Outside envelope ({n_out})",
            )
        )
        if options.label_outliers:
            idx = np.flatnonzero(result.outside)
            deviation = np.abs(y[idx] - (result.intercept + result.slope * x[idx]))
            worst = idx[np.argsort(deviation)[::-1][: options.label_outliers]]
            for k in worst:
                ax.annotate(
                    str(int(k) + 1),
                    (x[k], y[k]),
                    textcoords="offset points",
                    xytext=(3.5, 2.5),
                    fontsize=theme.base_pt - 1.5,
                    color=theme.color("outlier"),
                    zorder=6,
                )

    # -- frame ------------------------------------------------------------
    default_x, default_y = _default_labels(result)
    ax.set_xlabel(options.xlabel or default_x)
    ax.set_ylabel(options.ylabel or default_y)
    if options.title:
        ax.set_title(options.title, loc="left")

    if options.equal_aspect and not result.spec.detrend:
        ax.set_aspect("equal", adjustable="datalim")

    _style_spines(ax, theme)

    if options.annotate:
        text = _annotation_text(result, options.annotation_fields)
        if text:
            ax.text(
                0.035,
                0.965,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=theme.base_pt - 1.0,
                color=theme.color("annotation"),
                linespacing=1.45,
                zorder=6,
            )

    if options.show_legend and handles:
        ax.legend(handles=handles, loc=options.legend_loc, fontsize=theme.base_pt - 1.0)

    return ax


def render(
    result: QQResult,
    theme: Theme | str = "screen",
    options: PlotOptions | None = None,
) -> Figure:
    """Build a complete figure for *result*.

    Parameters
    ----------
    result : QQResult
        Output of :func:`wedgelab.qq.compute`.
    theme : Theme or str
        A theme, or a key into :data:`wedgelab.theme.THEMES`.
    options : PlotOptions or None
        Presentation choices; defaults are used when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
        Sized exactly to the theme's physical dimensions.
    """
    resolved = get_theme(theme) if isinstance(theme, str) else theme
    opts = options or PlotOptions()

    with applied(resolved):
        fig = Figure(figsize=resolved.figsize(), dpi=100)
        ax = fig.add_subplot(111)
        draw(result, ax, resolved, opts)
        fig.tight_layout(pad=0.4)
    return fig


def figure_for(result: QQResult, **kwargs: Any) -> Figure:
    """Convenience wrapper around :func:`render` taking keyword options."""
    theme = kwargs.pop("theme", "screen")
    return render(result, theme, PlotOptions(**kwargs))


# ---------------------------------------------------------------------------
# Renderers for the other diagnostics
#
# Each mirrors the Q-Q renderer: the figure is fully determined by the result,
# so the on-screen canvas and the exported file are the same draw call.
# ---------------------------------------------------------------------------


def _points_and_flags(ax, x, y, outside, theme, options, label):
    """Scatter the conforming points, then the flagged ones. Returns handles."""
    handles: list[Line2D] = []
    inside = ~outside if options.mark_outliers else np.ones_like(y, dtype=bool)
    ax.scatter(
        x[inside], y[inside], s=theme.marker_size, marker=theme.marker,
        facecolor=theme.color("point"), edgecolor=theme.color("point_edge"),
        linewidth=0.3, zorder=4,
    )
    handles.append(
        Line2D([], [], linestyle="none", marker=theme.marker,
               markerfacecolor=theme.color("point"),
               markeredgecolor=theme.color("point_edge"),
               markersize=theme.marker_size**0.5, label=label)
    )
    n_out = int(np.sum(outside))
    if options.mark_outliers and n_out:
        ax.scatter(
            x[outside], y[outside], s=theme.marker_size * 1.25, marker=theme.marker,
            facecolor=theme.color("outlier"), edgecolor=theme.color("point_edge"),
            linewidth=0.3, zorder=5,
        )
        handles.append(
            Line2D([], [], linestyle="none", marker=theme.marker,
                   markerfacecolor=theme.color("outlier"),
                   markeredgecolor=theme.color("point_edge"),
                   markersize=theme.marker_size**0.5,
                   label=f"Outside envelope ({n_out})")
        )
    return handles


def _band(ax, x, lower, upper, theme, alpha, step=False):
    """Fill a confidence envelope and return its legend handle."""
    kwargs = {"step": "post"} if step else {}
    ax.fill_between(x, lower, upper, facecolor=theme.color("band"),
                    edgecolor="none", zorder=1, **kwargs)
    return Line2D([], [], color=theme.color("band"), linewidth=5,
                  label=f"{100 * (1 - alpha):.0f}% envelope")


def _finish(ax, theme, options, handles, xlabel, ylabel, annotation):
    """Labels, spines, annotation box and legend, shared by every renderer."""
    ax.set_xlabel(options.xlabel or xlabel)
    ax.set_ylabel(options.ylabel or ylabel)
    if options.title:
        ax.set_title(options.title, loc="left")
    _style_spines(ax, theme)
    if options.annotate and annotation:
        ax.text(0.035, 0.965, annotation, transform=ax.transAxes, va="top", ha="left",
                fontsize=theme.base_pt - 1.0, color=theme.color("annotation"),
                linespacing=1.45, zorder=6)
    if options.show_legend and handles:
        ax.legend(handles=handles, loc=options.legend_loc, fontsize=theme.base_pt - 1.0)
    return ax


def draw_pp(result, ax: Axes, theme: Theme, options: PlotOptions) -> Axes:
    """Draw a P-P plot: fitted CDF against plotting position, on the unit square."""
    handles: list[Line2D] = []
    x, y = result.theoretical, result.empirical

    if options.show_band and result.lower is not None:
        handles.append(_band(ax, x, result.lower, result.upper, theme, result.spec.alpha))

    if options.show_line:
        lx, ly = result.line_points()
        ax.plot(lx, ly, color=theme.color("line"), linewidth=theme.line_width, zorder=3)
        handles.append(Line2D(
            [], [], color=theme.color("line"),
            label="Identity" if result.spec.line == "identity" else "Least squares",
        ))

    handles += _points_and_flags(ax, x, y, result.outside, theme, options, result.spec.label)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    return _finish(
        ax, theme, options, handles,
        "Theoretical probability",
        f"Fitted CDF, {DISTRIBUTIONS[result.spec.dist_key].label}",
        f"$n$ = {result.n}\n$r$ = {result.correlation:.4f}",
    )


def draw_ecdf(result, ax: Axes, theme: Theme, options: PlotOptions) -> Axes:
    """Draw an empirical CDF with its band and the fitted model overlaid."""
    handles: list[Line2D] = []

    if options.show_band and result.lower is not None:
        handle = _band(ax, result.x, result.lower, result.upper, theme,
                       result.spec.alpha, step=True)
        kind = "Simultaneous" if result.envelope == "simultaneous" else "Pointwise"
        handle.set_label(f"{kind} {100 * (1 - result.spec.alpha):.0f}% band")
        handles.append(handle)

    ax.step(result.x, result.ecdf, where="post", color=theme.color("point"),
            linewidth=theme.line_width * 1.2, zorder=4)
    handles.append(Line2D([], [], color=theme.color("point"),
                          label=f"ECDF, {result.spec.label}"))

    if result.spec.show_model:
        # The curve is drawn in the flag colour when it leaves the band, so the
        # verdict is legible without reading the caption.
        colour = theme.color("line") if result.band_contains_model else theme.color("outlier")
        ax.plot(result.model_x, result.model_cdf, color=colour, linestyle="--",
                linewidth=theme.line_width, zorder=5)
        handles.append(Line2D(
            [], [], color=colour, linestyle="--",
            label=f"Fitted {DISTRIBUTIONS[result.spec.dist_key].label}",
        ))

    ax.set_ylim(-0.02, 1.02)
    return _finish(
        ax, theme, options, handles, "Value", "Cumulative probability",
        f"$n$ = {result.n}\n$D$ = {result.ks_statistic:.4f}",
    )


def draw_two_sample(result, ax: Axes, theme: Theme, options: PlotOptions) -> Axes:
    """Draw a two-sample Q-Q plot, with no reference distribution involved."""
    handles: list[Line2D] = []
    x, y = result.first_quantiles, result.second_quantiles

    if options.show_band and result.lower is not None:
        handles.append(_band(ax, x, result.lower, result.upper, theme, result.spec.alpha))

    if options.show_line:
        lx, ly = result.line_points()
        ax.plot(lx, ly, color=theme.color("line"), linewidth=theme.line_width, zorder=3)
        handles.append(Line2D([], [], color=theme.color("line"), label={
            "ols": "Least squares", "quartile": "Quartile line", "identity": "Identity",
        }[result.spec.line]))

    handles += _points_and_flags(ax, x, y, result.outside, theme, options, "Common quantiles")
    if options.equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    return _finish(
        ax, theme, options, handles,
        f"Quantiles of {result.spec.first_label}",
        f"Quantiles of {result.spec.second_label}",
        f"slope = {result.slope:.3f}\nintercept = {result.intercept:.3f}",
    )


_DRAWERS = {
    "QQResult": draw,
    "PPResult": draw_pp,
    "ECDFResult": draw_ecdf,
    "TwoSampleResult": draw_two_sample,
}


def render_diagnostic(
    result: Any,
    theme: Theme | str = "screen",
    options: PlotOptions | None = None,
) -> Figure:
    """Render any diagnostic result, dispatching on its type.

    Accepts a :class:`~wedgelab.qq.QQResult` or any result from
    :mod:`wedgelab.diagnostics`, so a caller can switch figure type without
    switching function.

    Parameters
    ----------
    result : QQResult, PPResult, ECDFResult or TwoSampleResult
        What to draw.
    theme : Theme or str
        A theme, or a key into :data:`wedgelab.theme.THEMES`.
    options : PlotOptions or None
        Presentation choices.

    Returns
    -------
    matplotlib.figure.Figure
        Sized exactly to the theme's physical dimensions.

    Raises
    ------
    TypeError
        If *result* is not a recognised diagnostic result.
    """
    drawer = _DRAWERS.get(type(result).__name__)
    if drawer is None:
        raise TypeError(
            f"cannot render {type(result).__name__}; expected one of "
            + ", ".join(sorted(_DRAWERS))
        )
    resolved = get_theme(theme) if isinstance(theme, str) else theme
    opts = options or PlotOptions()
    with applied(resolved):
        fig = Figure(figsize=resolved.figsize(), dpi=100)
        ax = fig.add_subplot(111)
        drawer(result, ax, resolved, opts)
        fig.tight_layout(pad=0.4)
    return fig
