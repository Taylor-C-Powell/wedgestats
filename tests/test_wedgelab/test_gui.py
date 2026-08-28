"""Integration tests that drive the real Tk workbench.

These construct the actual window and exercise the panel callbacks, which is
the only way to catch wiring mistakes between a panel and the application.
The whole module skips when no display or Tk installation is available.
"""

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture
def app():
    """A live workbench, destroyed after the test."""
    try:
        from wedgelab.gui.app import WedgeLabApp

        instance = WedgeLabApp()
    except tk.TclError as exc:
        pytest.skip(f"no Tk display available: {exc}")
    instance.root.withdraw()
    try:
        yield instance
    finally:
        instance.root.destroy()


def settle(app) -> None:
    """Flush pending callbacks and force the debounced recompute."""
    app.request_update(immediate=True)
    app.root.update_idletasks()


class TestStartup:
    def test_computes_on_launch(self, app):
        assert app.last_result is not None
        assert app.last_result.n > 0

    def test_default_is_a_good_fit(self, app):
        assert app.last_result.diagnostics.ppcc > 0.98

    def test_all_panels_exist(self, app):
        for name in (
            "data_panel",
            "model_panel",
            "formula_panel",
            "presentation_panel",
            "knowledge_panel",
        ):
            assert getattr(app, name) is not None


class TestDataPanel:
    def test_switching_dataset_changes_the_data(self, app):
        before = app.state.data.copy()
        app.data_panel._on_dataset("heavy")
        settle(app)
        assert not np.array_equal(before, app.state.data)
        assert app.state.dataset_key == "heavy"

    def test_switching_dataset_adopts_the_suggested_reference(self, app):
        app.data_panel._on_dataset("exponential")
        settle(app)
        assert app.state.dist_key == "exponential"
        assert app.last_result.fit.spec.key == "exponential"

    def test_heavy_tails_lower_the_ppcc(self, app):
        clean = app.last_result.diagnostics.ppcc
        app.data_panel._on_dataset("heavy")
        settle(app)
        assert app.last_result.diagnostics.ppcc < clean


class TestModelPanel:
    def test_changing_the_distribution_refits(self, app):
        app.data_panel._on_dataset("exponential")
        settle(app)
        app.model_panel._on_dist("gamma")
        settle(app)
        assert app.last_result.fit.spec.key == "gamma"

    def test_changing_the_estimator_changes_the_fit(self, app):
        app.data_panel._on_dataset("contaminated")
        settle(app)
        mle = app.last_result.fit.params
        app.model_panel._on_method("robust")
        settle(app)
        robust = app.last_result.fit.params
        assert robust[1] < mle[1], "robust scale should be smaller under contamination"

    def test_unsupported_data_reports_an_error_rather_than_crashing(self, app):
        """A positive-support reference on data containing negatives."""
        app.data_panel._on_dataset("heavy")  # Student t, straddles zero
        settle(app)
        surviving = app.last_result
        app.model_panel._on_dist("exponential")
        settle(app)
        assert "support" in app.status.cget("text")
        # The last good figure is still on screen; nothing crashed.
        assert app.last_result is surviving


class TestFormulaWorkbench:
    def test_preset_changes_the_formula(self, app):
        app.formula_panel._on_preset("pp_hazen")
        settle(app)
        assert app.state.position_bindings["a"] == pytest.approx(0.5)

    def test_slider_changes_the_positions(self, app):
        app.formula_panel._on_slider("a", 0.0)
        settle(app)
        assert app.last_result.probabilities[0] == pytest.approx(
            1.0 / (app.last_result.n + 1)
        )

    def test_editing_the_expression_takes_effect(self, app):
        app.formula_panel.set_expression("(i - 0.44) / (n + 0.12)")
        settle(app)
        n = app.last_result.n
        assert app.last_result.probabilities[0] == pytest.approx((1 - 0.44) / (n + 0.12))

    def test_a_new_symbol_becomes_a_slider(self, app):
        app.formula_panel.set_expression("(i - a) / (n + 1 - 2*a) + c")
        settle(app)
        assert "c" in app.state.position.parameter_names
        assert "c" in app.state.position_bindings

    def test_invalid_expression_reports_and_keeps_the_old_one(self, app):
        previous = app.state.position.expression
        app.formula_panel.set_expression("(i - a / ")
        assert "syntax error" in app.formula_panel.status_text
        assert app.state.position.expression == previous

    def test_hostile_expression_is_refused(self, app):
        previous = app.state.position.expression
        app.formula_panel.set_expression('__import__("os").system("echo pwned")')
        assert app.formula_panel.status_text
        assert app.state.position.expression == previous

    def test_out_of_range_expression_is_refused(self, app):
        previous = app.state.position.expression
        app.formula_panel.set_expression("i / n")
        assert app.state.position.expression == previous

    def test_exact_median_rank_expression_works(self, app):
        app.formula_panel.set_expression("beta_ppf(0.5, i, n - i + 1)")
        settle(app)
        p = app.last_result.probabilities
        assert np.allclose(p, 1.0 - p[::-1], atol=1e-9)


class TestPresentationPanel:
    @pytest.mark.parametrize(
        "envelope", ["none", "beta", "asymptotic", "simultaneous", "bootstrap"]
    )
    def test_every_envelope_draws(self, app, envelope):
        app.state.bootstrap_reps = 60
        app.presentation_panel._on_envelope(envelope)
        settle(app)
        assert app.last_result.spec.envelope == envelope

    @pytest.mark.parametrize("line", ["ols", "quartile", "theoretical"])
    def test_every_line_draws(self, app, line):
        app.presentation_panel._on_line(line)
        settle(app)
        assert app.last_result.spec.line == line

    @pytest.mark.parametrize(
        "theme", ["nature", "science", "ieee", "plos", "apa", "presentation"]
    )
    def test_every_theme_draws(self, app, theme):
        app.presentation_panel._on_theme(theme)
        settle(app)
        assert app.state.theme_key == theme

    def test_alpha_widens_the_band(self, app):
        app.presentation_panel._on_envelope("beta")
        app.presentation_panel._on_alpha("alpha", 0.20)
        settle(app)
        narrow = float(np.mean(app.last_result.upper - app.last_result.lower))
        app.presentation_panel._on_alpha("alpha", 0.01)
        settle(app)
        wide = float(np.mean(app.last_result.upper - app.last_result.lower))
        assert wide > narrow

    def test_toggles_reach_the_spec(self, app):
        app.presentation_panel._standardize.set(True)
        app.presentation_panel._detrend.set(True)
        app.presentation_panel._on_toggle()
        settle(app)
        assert app.last_result.spec.standardize
        assert app.last_result.spec.detrend


class TestKnowledgePanel:
    def test_search_narrows_the_tree(self, app):
        panel = app.knowledge_panel
        everything = len(panel._keys)
        panel._query.insert(0, "blom")
        panel.refresh()
        assert 0 < len(panel._keys) < everything

    def test_loading_a_position_reaches_the_workbench(self, app):
        panel = app.knowledge_panel
        node = next(k for k, v in panel._keys.items() if v == "pp_gringorten")
        panel._tree.selection_set(node)
        panel._on_select(None)
        panel._on_load()
        settle(app)
        assert app.state.position_bindings["a"] == pytest.approx(0.44)
        assert app.state.position_source == "pp_gringorten"

    def test_selecting_an_entry_shows_its_citation(self, app):
        panel = app.knowledge_panel
        node = next(k for k, v in panel._keys.items() if v == "pp_blom")
        panel._tree.selection_set(node)
        panel._on_select(None)
        assert "1958" in panel._detail.get("1.0", "end")


class TestScratchpad:
    def test_opens_and_evaluates(self, app):
        from wedgelab.gui.scratch import ScratchWindow

        window = ScratchWindow(app.root, app.state.position)
        try:
            window.update_idletasks()
            assert window._formula.expression == app.state.position.expression
        finally:
            window.destroy()

    def test_sweeps_a_variable(self, app):
        from wedgelab.gui.scratch import ScratchWindow
        from wedgelab.knowledge import KNOWLEDGE

        window = ScratchWindow(app.root, KNOWLEDGE.get("tf_boxcox").formula)
        try:
            window._sweep_var.set("x")
            window._from.delete(0, "end")
            window._from.insert(0, "0.5")
            window._to.delete(0, "end")
            window._to.insert(0, "5")
            window._recompute()
            window.update_idletasks()
            assert "range" in window._status.cget("text")
        finally:
            window.destroy()


class TestExportPaths:
    def test_caption_copies_to_the_clipboard(self, app):
        app._copy_caption()
        assert "Quantile-quantile plot" in app.root.clipboard_get()

    def test_figure_export_writes_a_file(self, app, tmp_path, monkeypatch):
        target = tmp_path / "figure.pdf"
        monkeypatch.setattr(
            "wedgelab.gui.app.filedialog.asksaveasfilename", lambda **_k: str(target)
        )
        app._export_figure()
        assert target.exists() and target.stat().st_size > 0

    def test_script_export_is_valid_python(self, app, tmp_path, monkeypatch):
        target = tmp_path / "repro.py"
        monkeypatch.setattr(
            "wedgelab.gui.app.filedialog.asksaveasfilename", lambda **_k: str(target)
        )
        app._export_script()
        compile(target.read_text(encoding="utf-8"), str(target), "exec")

    def test_session_round_trip_through_the_gui(self, app, tmp_path, monkeypatch):
        target = tmp_path / "session.json"
        monkeypatch.setattr(
            "wedgelab.gui.app.filedialog.asksaveasfilename", lambda **_k: str(target)
        )
        app.formula_panel._on_preset("pp_cunnane")
        app.presentation_panel._on_theme("ieee")
        settle(app)
        app._save_session()

        saved = target.with_suffix(".wedgelab.json")
        monkeypatch.setattr(
            "wedgelab.gui.app.filedialog.askopenfilename", lambda **_k: str(saved)
        )
        app.formula_panel._on_preset("pp_hazen")
        settle(app)
        app._open_session()
        assert app.state.theme_key == "ieee"
        assert app.state.position_bindings["a"] == pytest.approx(0.40)
        # The widgets must follow the restored state, not just the model.
        assert app.presentation_panel._theme.key == "ieee"

    def test_programmatic_changes_reach_the_widgets(self, app):
        """State set in code must show up in the controls, or the UI lies."""
        app.data_panel._on_dataset("contaminated")
        app.model_panel._on_method("robust")
        app.presentation_panel._on_envelope("simultaneous")
        settle(app)
        assert app.data_panel._dataset.key == "contaminated"
        assert app.model_panel._method.key == "robust"
        assert app.presentation_panel._envelope.key == "simultaneous"


class TestFigureTypes:
    """Switching figure type must keep every choice that still applies."""

    @pytest.mark.parametrize("kind", ["qq", "pp", "ecdf"])
    def test_every_type_computes_and_draws(self, app, kind):
        app.presentation_panel._on_figure_type(kind)
        settle(app)
        assert app.last_result is not None
        assert type(app.last_result).__name__ in (
            "QQResult", "PPResult", "ECDFResult",
        )

    def test_switching_keeps_the_data_and_model(self, app):
        app.data_panel._on_dataset("heavy")
        app.model_panel._on_method("robust")
        settle(app)
        for kind in ("pp", "ecdf", "qq"):
            app.presentation_panel._on_figure_type(kind)
            settle(app)
            assert app.state.dataset_key == "heavy"
            assert app.last_result.fit.method == "robust"

    def test_envelope_vocabulary_follows_the_figure(self, app):
        app.presentation_panel._on_figure_type("ecdf")
        settle(app)
        assert app.state.envelope in ("none", "simultaneous", "pointwise")
        app.presentation_panel._on_figure_type("qq")
        settle(app)
        assert app.state.envelope in (
            "auto", "none", "beta", "asymptotic", "simultaneous", "bootstrap",
        )

    def test_pp_and_qq_agree_on_the_flagged_count(self, app):
        """The two are one test on two axes; the readout must not imply otherwise."""
        app.data_panel._on_dataset("heavy")
        app.model_panel._on_dist("normal")
        app.presentation_panel._on_envelope("beta")
        settle(app)
        app.presentation_panel._on_figure_type("qq")
        settle(app)
        qq_flagged = app.last_result.diagnostics.outside_band
        app.presentation_panel._on_figure_type("pp")
        settle(app)
        assert app.last_result.outside_band == qq_flagged

    def test_script_export_declines_outside_qq(self, app, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            "wedgelab.gui.app.messagebox.showinfo",
            lambda title, message: seen.setdefault("title", title),
        )
        app.presentation_panel._on_figure_type("ecdf")
        settle(app)
        app._export_script()
        assert seen.get("title") == "Not available yet"

    def test_figure_export_works_for_every_type(self, app, tmp_path, monkeypatch):
        for kind in ("qq", "pp", "ecdf"):
            target = tmp_path / f"{kind}.png"
            monkeypatch.setattr(
                "wedgelab.gui.app.filedialog.asksaveasfilename", lambda **_k: str(target)
            )
            app.presentation_panel._on_figure_type(kind)
            settle(app)
            app._export_figure()
            assert target.exists() and target.stat().st_size > 0
