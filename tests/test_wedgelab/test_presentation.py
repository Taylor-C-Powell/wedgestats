"""Tests for themes, rendering, export, and the built-in datasets.

Rendering runs on the Agg backend, so these tests need no display.
"""

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from wedgelab.datasets import DATASETS, dataset_keys, generate, load_file
from wedgelab.export import (
    RASTER_FORMATS,
    VECTOR_FORMATS,
    from_json,
    read_session,
    save_figure,
    to_json,
    to_script,
    write_session,
)
from wedgelab.plot import PlotOptions, render
from wedgelab.qq import QQSpec, compute
from wedgelab.theme import MM_PER_INCH, THEMES, get_theme, theme_keys


@pytest.fixture(scope="module")
def result():
    return compute(QQSpec(data=generate("normal", 60, seed=1), label="fixture"))


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------


class TestThemes:
    def test_keys_are_stable(self):
        assert theme_keys() == tuple(THEMES)

    def test_unknown_theme_raises(self):
        with pytest.raises(KeyError, match="unknown theme"):
            get_theme("vogue")

    @pytest.mark.parametrize("key", list(THEMES))
    def test_geometry_is_physical(self, key):
        theme = get_theme(key)
        assert theme.width_mm > 0
        assert theme.width_in == pytest.approx(theme.width_mm / MM_PER_INCH)
        assert theme.figsize() == (theme.width_in, theme.height_in)

    @pytest.mark.parametrize("key", list(THEMES))
    def test_type_is_legible(self, key):
        """No journal accepts type below about five points."""
        assert get_theme(key).base_pt >= 6.0

    @pytest.mark.parametrize("key", list(THEMES))
    def test_rc_params_are_accepted_by_matplotlib(self, key):
        with matplotlib.rc_context(get_theme(key).rc_params()):
            assert matplotlib.rcParams["font.size"] == get_theme(key).base_pt

    @pytest.mark.parametrize("key", list(THEMES))
    def test_vector_text_stays_editable(self, key):
        """Journals require selectable text, not outlined glyphs."""
        rc = get_theme(key).rc_params()
        assert rc["pdf.fonttype"] == 42
        assert rc["svg.fonttype"] == "none"

    def test_nature_is_the_published_column_width(self):
        assert get_theme("nature").width_mm == pytest.approx(89.0)

    def test_ieee_is_three_and_a_half_inches(self):
        assert get_theme("ieee").width_in == pytest.approx(3.5, abs=0.01)

    def test_colour_falls_back_to_the_base_palette(self):
        assert get_theme("nature").color("point").startswith("#")

    def test_geometry_note_states_millimetres(self):
        assert "mm" in get_theme("nature").geometry_note()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    @pytest.mark.parametrize("key", list(THEMES))
    def test_every_theme_renders_at_its_declared_size(self, key, result):
        figure = render(result, key)
        theme = get_theme(key)
        width, height = figure.get_size_inches()
        assert width == pytest.approx(theme.width_in, abs=0.01)
        assert height == pytest.approx(theme.height_in, abs=0.01)

    def test_render_accepts_a_theme_object(self, result):
        assert isinstance(render(result, get_theme("nature")), Figure)

    @pytest.mark.parametrize("envelope", ["none", "beta", "simultaneous"])
    def test_renders_every_envelope(self, envelope):
        r = compute(QQSpec(data=generate("normal", 40, seed=2), envelope=envelope))
        assert isinstance(render(r, "screen"), Figure)

    def test_renders_detrended(self):
        r = compute(QQSpec(data=generate("normal", 40, seed=3), detrend=True))
        assert isinstance(render(r, "screen"), Figure)

    def test_renders_with_everything_switched_off(self, result):
        options = PlotOptions(
            show_band=False, show_line=False, show_legend=False, annotate=False
        )
        assert isinstance(render(result, "screen", options), Figure)

    def test_renders_with_outlier_labels(self):
        data = generate("contaminated", 60, seed=4)
        r = compute(QQSpec(data=data, envelope="beta"))
        figure = render(r, "screen", PlotOptions(label_outliers=3))
        assert isinstance(figure, Figure)

    def test_unbounded_band_still_renders(self):
        r = compute(QQSpec(data=generate("normal", 30, seed=5), envelope="simultaneous"))
        assert not np.all(np.isfinite(r.lower))
        assert isinstance(render(r, "screen"), Figure)

    def test_axis_labels_name_the_reference(self, result):
        ax = render(result, "screen").axes[0]
        assert "Normal" in ax.get_xlabel()

    def test_detrended_label_says_so(self):
        r = compute(QQSpec(data=generate("normal", 40, seed=6), detrend=True))
        assert "reference line" in render(r, "screen").axes[0].get_ylabel()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestSaveFigure:
    @pytest.mark.parametrize("fmt", VECTOR_FORMATS + RASTER_FORMATS)
    def test_every_format_writes_a_file(self, fmt, result, tmp_path):
        written = save_figure(render(result, "screen"), tmp_path / "fig", formats=(fmt,))
        assert written[0].exists() and written[0].stat().st_size > 0

    def test_rejects_unknown_format(self, result, tmp_path):
        with pytest.raises(ValueError, match="unsupported format"):
            save_figure(render(result, "screen"), tmp_path / "fig", formats=("jpeg2000",))

    def test_writes_several_formats_at_once(self, result, tmp_path):
        written = save_figure(
            render(result, "screen"), tmp_path / "fig", formats=("pdf", "png")
        )
        assert {p.suffix for p in written} == {".pdf", ".png"}

    def test_creates_missing_directories(self, result, tmp_path):
        target = tmp_path / "deep" / "nested" / "fig"
        assert save_figure(render(result, "screen"), target, formats=("png",))[0].exists()

    def test_pdf_is_a_real_pdf(self, result, tmp_path):
        path = save_figure(render(result, "nature"), tmp_path / "fig", formats=("pdf",))[0]
        assert path.read_bytes()[:4] == b"%PDF"


class TestScriptExport:
    def test_generated_script_is_valid_python(self, result):
        compile(to_script(result, "nature"), "<generated>", "exec")

    def test_script_embeds_the_data(self, result):
        assert "np.array([" in to_script(result, "nature")

    def test_script_records_the_plotting_position_formula(self, result):
        source = to_script(result, "nature")
        assert result.spec.position.expression in source

    def test_script_records_the_citation(self):
        from wedgelab.knowledge import KNOWLEDGE

        r = compute(
            QQSpec(
                data=generate("normal", 30, seed=7),
                position=KNOWLEDGE.get("pp_blom").formula,
            )
        )
        assert "Blom" in to_script(r, "nature")

    def test_script_references_a_data_file_when_asked(self, result):
        source = to_script(result, "nature", data_path="measurements.txt")
        assert "np.loadtxt('measurements.txt')" in source
        assert "np.array([" not in source

    def test_large_samples_are_not_inlined(self):
        r = compute(QQSpec(data=generate("normal", 400, seed=8)))
        source = to_script(r, "nature", max_inline=100)
        assert "exceeded the inline limit" in source

    def test_rejects_unknown_theme(self, result):
        with pytest.raises(KeyError):
            to_script(result, "vogue")

    def test_script_round_trips_the_settings(self):
        spec = QQSpec(
            data=generate("normal", 30, seed=9),
            envelope="simultaneous",
            line="quartile",
            alpha=0.01,
            detrend=True,
            label="round trip",
        )
        source = to_script(compute(spec), "ieee")
        for fragment in ("'simultaneous'", "'quartile'", "0.01", "detrend=True"):
            assert fragment in source


class TestSessions:
    def test_json_round_trip_preserves_the_spec(self):
        spec = QQSpec(
            data=generate("heavy", 40, seed=10),
            dist_key="normal",
            fit_method="robust",
            envelope="bootstrap",
            line="quartile",
            alpha=0.02,
            standardize=True,
            label="session test",
        )
        restored, theme, options = from_json(to_json(spec, "plos", PlotOptions()))
        assert restored.dist_key == spec.dist_key
        assert restored.fit_method == spec.fit_method
        assert restored.envelope == spec.envelope
        assert restored.line == spec.line
        assert restored.alpha == spec.alpha
        assert restored.standardize == spec.standardize
        assert restored.label == spec.label
        assert np.allclose(restored.data, spec.data)
        assert theme == "plos"
        assert isinstance(options, PlotOptions)

    def test_round_trip_preserves_the_formula_and_its_parameters(self):
        spec = QQSpec(
            data=generate("normal", 30, seed=11), position_bindings={"a": 0.44}
        )
        restored, _, _ = from_json(to_json(spec))
        assert restored.position.expression == spec.position.expression
        assert restored.position.parameter("a").upper == 0.5
        assert restored.position_bindings == {"a": 0.44}

    def test_round_trip_reproduces_the_numbers(self):
        spec = QQSpec(data=generate("right_skew", 50, seed=12), dist_key="gamma")
        restored, _, _ = from_json(to_json(spec))
        assert compute(restored).diagnostics.ppcc == pytest.approx(
            compute(spec).diagnostics.ppcc
        )

    def test_data_can_be_omitted_and_supplied_later(self):
        spec = QQSpec(data=generate("normal", 30, seed=13))
        text = to_json(spec, include_data=False)
        assert json.loads(text)["spec"]["data"] is None
        restored, _, _ = from_json(text, data=spec.data)
        assert np.allclose(restored.data, spec.data)

    def test_missing_data_without_replacement_raises(self):
        text = to_json(QQSpec(data=generate("normal", 30, seed=14)), include_data=False)
        with pytest.raises(ValueError, match="carries no data"):
            from_json(text)

    def test_rejects_foreign_json(self):
        with pytest.raises(ValueError, match="not a wedgelab session"):
            from_json('{"something": "else"}')

    def test_file_round_trip(self, tmp_path):
        spec = QQSpec(data=generate("normal", 30, seed=15), label="on disk")
        path = write_session(tmp_path / "session", spec, "thesis")
        assert path.suffix == ".json"
        restored, theme, _ = read_session(path)
        assert restored.label == "on disk"
        assert theme == "thesis"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class TestDatasets:
    def test_keys_are_stable(self):
        assert dataset_keys() == tuple(DATASETS)

    @pytest.mark.parametrize("key", list(DATASETS))
    def test_every_dataset_generates_usable_data(self, key):
        data = generate(key, seed=0)
        assert data.ndim == 1
        assert np.all(np.isfinite(data))
        assert np.ptp(data) > 0

    @pytest.mark.parametrize("key", list(DATASETS))
    def test_every_dataset_documents_its_signature(self, key):
        spec = DATASETS[key]
        assert spec.expect.strip()
        assert spec.suggested_dist in __import__(
            "wedgelab.models", fromlist=["DISTRIBUTIONS"]
        ).DISTRIBUTIONS

    def test_generation_is_reproducible(self):
        assert np.allclose(generate("normal", 40, 3), generate("normal", 40, 3))

    def test_different_seeds_differ(self):
        assert not np.allclose(generate("normal", 40, 1), generate("normal", 40, 2))

    def test_requested_size_is_honoured(self):
        assert generate("normal", 37, 0).size == 37

    def test_rejects_unknown_dataset(self):
        with pytest.raises(KeyError, match="unknown dataset"):
            generate("nonesuch")

    def test_rejects_tiny_size(self):
        with pytest.raises(ValueError, match="at least 3"):
            generate("normal", 2)

    def test_rounded_dataset_has_ties(self):
        data = generate("rounded", 150, seed=0)
        assert len(np.unique(data)) < data.size

    def test_heavy_dataset_has_heavy_tails(self):
        from scipy import stats as sp_stats

        assert sp_stats.kurtosis(generate("heavy", 400, seed=0)) > 1.0


class TestLoadFile:
    def test_reads_a_plain_column(self, tmp_path):
        path = tmp_path / "values.txt"
        path.write_text("1.0\n2.5\n3.5\n4.0\n", encoding="utf-8")
        values, name = load_file(path)
        assert np.allclose(values, [1.0, 2.5, 3.5, 4.0])
        assert name == "column 1"

    def test_reads_a_csv_header(self, tmp_path):
        path = tmp_path / "values.csv"
        path.write_text("id,weight\n1,10.5\n2,11.5\n3,9.5\n", encoding="utf-8")
        values, name = load_file(path, column="weight")
        assert name == "weight"
        assert np.allclose(values, [10.5, 11.5, 9.5])

    def test_first_numeric_column_is_the_default(self, tmp_path):
        path = tmp_path / "values.csv"
        path.write_text("a,b\n1,10\n2,11\n3,12\n", encoding="utf-8")
        values, name = load_file(path)
        assert name == "a" and np.allclose(values, [1, 2, 3])

    def test_reads_by_index(self, tmp_path):
        path = tmp_path / "values.tsv"
        path.write_text("1\t10\n2\t11\n3\t12\n", encoding="utf-8")
        values, _ = load_file(path, column=1)
        assert np.allclose(values, [10, 11, 12])

    def test_missing_values_become_nan(self, tmp_path):
        path = tmp_path / "values.csv"
        path.write_text("x\n1\n\n3\nNA\n5\n", encoding="utf-8")
        values, _ = load_file(path)
        assert np.isnan(values).sum() == 1

    def test_rejects_empty_file(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_file(path)

    def test_rejects_unknown_column_name(self, tmp_path):
        path = tmp_path / "values.csv"
        path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no column named"):
            load_file(path, column="missing")

    def test_rejects_out_of_range_index(self, tmp_path):
        path = tmp_path / "values.csv"
        path.write_text("1,2\n3,4\n5,6\n", encoding="utf-8")
        with pytest.raises(ValueError, match="out of range"):
            load_file(path, column=9)

    def test_rejects_a_file_with_no_numeric_column(self, tmp_path):
        path = tmp_path / "words.csv"
        path.write_text("a,b\nx,y\np,q\nm,n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no column with at least three"):
            load_file(path)

    def test_loaded_data_feeds_the_engine(self, tmp_path):
        path = tmp_path / "sample.txt"
        path.write_text(
            "\n".join(str(v) for v in generate("normal", 40, seed=0)), encoding="utf-8"
        )
        values, name = load_file(path)
        assert compute(QQSpec(data=values, label=name)).diagnostics.ppcc > 0.95
