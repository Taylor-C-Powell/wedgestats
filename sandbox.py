"""WedgeLab sandbox: launch the workbench, or run the worked example.

    python sandbox.py            open the interactive workbench
    python sandbox.py demo       run the Q-Q development walkthrough
    python sandbox.py demo out   ... writing figures to ./out

The walkthrough builds one publication figure in seven stages and prints what
changed and why at each one.  It is the same code path the GUI uses; the GUI
just puts sliders on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wedgelab as wl
from wedgelab.export import to_script

RULE = "-" * 74


def _banner(number: int, title: str, why: str) -> None:
    print(f"\n{RULE}\nSTAGE {number}.  {title}\n{RULE}\n{why}\n")


def _report(result: wl.QQResult, note: str = "") -> None:
    d = result.diagnostics
    print(
        f"    r = {d.ppcc:.5f}    slope = {d.slope:7.4f}    "
        f"outside band = {d.outside_band:3d}"
        + (f"    (expected ~{d.expected_outside:.1f})" if d.expected_outside else "")
    )
    if note:
        print(f"    {note}")


def _save(result: wl.QQResult, theme: str, stem: Path) -> list[Path]:
    """Render and write a stage figure."""
    figure = wl.render(result, theme)
    return wl.save_figure(figure, stem, formats=("png",), dpi=140)


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap, so the caption prints inside the rule."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def walkthrough(outdir: Path) -> None:
    """Develop a publication-quality Q-Q plot, one decision at a time."""
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    print(RULE)
    print("DEVELOPING A PUBLICATION-QUALITY Q-Q PLOT")
    print(RULE)
    print(
        "\nThe sample: 100 assay readings, 95 drawn from N(50, 5) and 5 from a\n"
        "much wider N(50, 28). We are asked whether the assay is normally\n"
        "distributed. Every stage below changes exactly one decision."
    )

    data = wl.generate("contaminated", n=100, seed=11)
    label = "assay readings"

    # -- 1 ----------------------------------------------------------------
    _banner(
        1,
        "The naive plot",
        "Hazen positions, least-squares line, no envelope. Roughly what a\n"
        "one-line call gets you in most software. It is not wrong, but it\n"
        "answers no question: there is nothing to judge the scatter against.",
    )
    naive = wl.QQSpec(
        data=data,
        dist_key="normal",
        fit_method="mle",
        position=wl.KNOWLEDGE.get("pp_hazen").formula,
        envelope="none",
        label=label,
    )
    result = wl.compute(naive)
    _report(result)
    written += _save(result, "screen", outdir / "stage1_naive")

    # -- 2 ----------------------------------------------------------------
    _banner(
        2,
        "Choosing plotting positions",
        "p_i = (i - a) / (n + 1 - 2a) covers almost every published rule.\n"
        "Sweeping 'a' shows what the choice actually costs, which is the\n"
        "question nobody asks before making it.",
    )
    family = wl.symmetric_plotting_position(0.375)
    for key in ("pp_weibull", "pp_tukey", "pp_blom", "pp_cunnane", "pp_hazen"):
        entry = wl.KNOWLEDGE.get(key)
        a = entry.formula.defaults()["a"]
        r = wl.compute(naive.replace(position=family, position_bindings={"a": a}))
        print(f"    a = {a:<6.4f}  r = {r.diagnostics.ppcc:.6f}   {entry.name}")
    exact = wl.compute(naive.replace(position=wl.KNOWLEDGE.get("pp_median_exact").formula))
    print(f"    exact       r = {exact.diagnostics.ppcc:.6f}   Beta(i, n-i+1) median rank")
    print(
        "\n    Every rule in the literature lands within a whisker of the others\n"
        "    at this n, so the choice is defensible either way: state it and\n"
        "    move on. At n = 15 the same sweep separates them visibly.\n"
        "\n    Adopting Blom, which approximates E[Z_(i)] under normality."
    )
    blom = naive.replace(position=wl.KNOWLEDGE.get("pp_blom").formula)

    # -- 3 ----------------------------------------------------------------
    _banner(
        3,
        "An envelope that means something",
        "The i-th uniform order statistic is exactly Beta(i, n-i+1), so the\n"
        "band comes from the order statistics' own sampling behaviour rather\n"
        "than from an approximation. Now the scatter has a yardstick.",
    )
    banded = blom.replace(envelope="beta", alpha=0.05)
    result = wl.compute(banded)
    _report(result, "over half the points are outside, including the bulk")
    for warning in result.warnings:
        print(f"    note: {warning}")
    written += _save(result, "screen", outdir / "stage3_envelope")
    print(
        "\n    Worth understanding why so many: the exact band is *narrow* near\n"
        "    the median, because a central order statistic has little sampling\n"
        "    variance. A reference whose scale is wrong therefore fails across\n"
        "    the whole middle, not just at the ends. That is the band telling us\n"
        "    the fit is wrong everywhere -- which is the next stage's problem."
    )

    # -- 4 ----------------------------------------------------------------
    _banner(
        4,
        "The estimator decides what the band is measuring",
        "Maximum likelihood spends variance on the five contaminating points,\n"
        "inflating sigma until the reference describes neither the bulk nor the\n"
        "outliers. A median/MAD fit describes the bulk and lets the\n"
        "contamination fall outside, where it belongs.",
    )
    for method in ("mle", "robust"):
        r = wl.compute(banded.replace(fit_method=method))
        print(f"    {method:7s}  {r.fit.summary()}")
        _report(r)
    robust = banded.replace(fit_method="robust")
    result = wl.compute(robust)
    written += _save(result, "screen", outdir / "stage4_robust")
    print(
        "\n    MLE puts sigma at 9.78 -- nearly double the 5 the clean data were\n"
        "    drawn with -- and 53 points break the band. The robust fit recovers\n"
        "    sigma close to the truth and isolates the failure to 11 points in\n"
        "    the tails. Same data, same band rule; only the estimator changed.\n"
        "\n    Notice that r is identical under both fits. The probability plot\n"
        "    correlation coefficient is invariant to the fitted location and\n"
        "    scale, so it cannot referee this choice. The band can."
    )

    # -- 5 ----------------------------------------------------------------
    _banner(
        5,
        "A simultaneous band for a claim about the whole plot",
        "A pointwise band expects about alpha*n excursions even under a correct\n"
        "model. If the sentence you intend to write is about the sample as a\n"
        "whole, the band has to be simultaneous.",
    )
    for envelope in ("beta", "asymptotic", "simultaneous", "bootstrap"):
        r = wl.compute(robust.replace(envelope=envelope, bootstrap_reps=400))
        print(f"    {envelope:13s} outside = {r.diagnostics.outside_band:3d}")
    print(
        "\n    The three pointwise bands agree closely, which is reassuring: the\n"
        "    exact, asymptotic, and bootstrap derivations are answering the same\n"
        "    question and getting the same answer.\n"
        "\n    The simultaneous band flags nothing at all. That is not a bug. The\n"
        "    Kolmogorov-Smirnov band is weak exactly where this sample deviates,\n"
        "    in the tails, and it is unbounded at the extreme ranks. Shapiro-Wilk\n"
        "    on the same sample returns p = 5e-14. Choose the band to match the\n"
        "    claim, and do not let a silent band talk you out of a real finding."
    )

    # -- 6 ----------------------------------------------------------------
    _banner(
        6,
        "Detrending, when the departure is small next to the range",
        "Subtracting the reference line rescales the vertical axis to the\n"
        "residual. Same information, an order of magnitude more resolution.",
    )
    detrended = robust.replace(detrend=True, standardize=True)
    result = wl.compute(detrended)
    _report(result)
    written += _save(result, "screen", outdir / "stage6_detrended")

    # -- 7 ----------------------------------------------------------------
    _banner(
        7,
        "Sizing it for the journal, and making it reproducible",
        "The figure is written at the journal's exact column width, so it is\n"
        "placed at 1:1 and the type stays the size it was set. Then the whole\n"
        "specification is written out as a script that rebuilds it.",
    )
    final = robust
    result = wl.compute(final)
    options = wl.PlotOptions(
        annotation_fields=("n", "ppcc", "shapiro"),
        show_legend=True,
        label_outliers=3,
    )
    for theme_key in ("nature", "ieee", "plos"):
        theme = wl.get_theme(theme_key)
        figure = wl.render(result, theme_key, options)
        written += wl.save_figure(
            figure, outdir / f"stage7_{theme_key}", formats=("pdf", "png")
        )
        print(f"    {theme.label:38s} {theme.geometry_note()}")

    script_path = outdir / "reproduce_figure.py"
    script_path.write_text(
        to_script(
            result,
            "nature",
            options,
            stem=(outdir / "reproduced").as_posix(),
            formats=("pdf",),
        ),
        encoding="utf-8",
    )
    written.append(script_path)
    written.append(wl.write_session(outdir / "assay_qq", final, "nature", options))

    print(f"\n{RULE}\nCAPTION\n{RULE}\n")
    for line in _wrap(result.caption(), 74):
        print(line)

    print(f"\n{RULE}\nDIAGNOSTICS\n{RULE}")
    for line in result.diagnostics.lines():
        print(f"    {line}")

    print(f"\n{RULE}\nFILES WRITTEN\n{RULE}")
    for path in written:
        print(f"    {path.name}")
    print(f"\n    in {outdir}")
    print(
        f"\nRun 'python {Path(__file__).name}' with no arguments to open the\n"
        "workbench and do all of this with sliders instead."
    )


def main(argv: list[str]) -> int:
    """Dispatch between the GUI and the walkthrough."""
    if len(argv) > 1 and argv[1] == "demo":
        import matplotlib

        matplotlib.use("Agg")
        outdir = Path(argv[2]) if len(argv) > 2 else Path(__file__).resolve().parent / "out"
        walkthrough(outdir)
        return 0

    if len(argv) > 1 and argv[1] not in ("gui", "app"):
        print(__doc__)
        return 2

    wl.launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
