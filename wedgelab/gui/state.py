"""Mutable workbench state, and its translation into an immutable spec.

The GUI edits a :class:`AppState`; every recompute converts it into a frozen
:class:`~wedgelab.qq.QQSpec`.  Keeping the mutable and immutable halves apart
is what lets the exporter serialise exactly what was drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wedgelab.datasets import generate
from wedgelab.formula import Formula
from wedgelab.knowledge import symmetric_plotting_position
from wedgelab.plot import PlotOptions
from wedgelab.qq import QQSpec

__all__ = ["AppState"]


@dataclass
class AppState:
    """Everything the workbench is currently showing."""

    data: np.ndarray = field(default_factory=lambda: generate("normal", seed=7))
    label: str = "Normal, mu=100 sigma=15 (synthetic)"
    source: str = "dataset:normal"

    dataset_key: str = "normal"
    dataset_n: int = 80
    dataset_seed: int = 7

    dist_key: str = "normal"
    fit_method: str = "mle"
    manual_params: tuple[float, ...] | None = None

    position: Formula = field(
        default_factory=lambda: symmetric_plotting_position(
            0.375,
            name="Blom  (a = 3/8)",
            citation="Blom, G. (1958). Statistical Estimates and Transformed "
            "Beta-Variables. Wiley, New York.",
            description="Cumulative probability assigned to the i-th order statistic",
        )
    )
    position_bindings: dict[str, float] = field(default_factory=dict)
    position_source: str = "pp_blom"

    line: str = "ols"
    envelope: str = "beta"
    alpha: float = 0.05
    standardize: bool = False
    detrend: bool = False
    bootstrap_reps: int = 400
    random_state: int = 0

    theme_key: str = "screen"
    options: PlotOptions = field(default_factory=PlotOptions)

    def __post_init__(self) -> None:
        if not self.position_bindings:
            self.position_bindings = dict(self.position.defaults())

    def to_spec(self) -> QQSpec:
        """Freeze the current state into a :class:`~wedgelab.qq.QQSpec`."""
        return QQSpec(
            data=self.data,
            dist_key=self.dist_key,
            fit_method=self.fit_method,
            manual_params=self.manual_params,
            position=self.position,
            position_bindings=dict(self.position_bindings),
            line=self.line,
            envelope=self.envelope,
            alpha=self.alpha,
            standardize=self.standardize,
            detrend=self.detrend,
            bootstrap_reps=self.bootstrap_reps,
            random_state=self.random_state,
            label=self.label,
        )

    def adopt(self, spec: QQSpec, theme_key: str, options: PlotOptions) -> None:
        """Overwrite the state from a restored session."""
        self.data = np.asarray(spec.data, dtype=float)
        self.label = spec.label
        self.source = "session"
        self.dist_key = spec.dist_key
        self.fit_method = spec.fit_method
        self.manual_params = spec.manual_params
        self.position = spec.position
        self.position_bindings = dict(spec.position_bindings)
        self.position_source = spec.position.derived_from or spec.position.name
        self.line = spec.line
        self.envelope = spec.envelope
        self.alpha = spec.alpha
        self.standardize = spec.standardize
        self.detrend = spec.detrend
        self.bootstrap_reps = spec.bootstrap_reps
        self.random_state = spec.random_state
        self.theme_key = theme_key
        self.options = options
