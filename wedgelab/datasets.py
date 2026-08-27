"""Synthetic samples that exercise the diagnostic, plus a file loader.

Every generator here is synthetic and labelled as such -- no dataset in this
module is presented as measured data.  Each one is chosen to produce a Q-Q
signature worth recognising: the S-shape of heavy tails, the arc of skew, the
staircase of a rounded measurement scale, the isolated points of contamination.

Samples are drawn through the :mod:`wedgestats` distribution classes' ``rvs``,
so the toolkit generates and analyses data through the same API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from wedgestats.distributions import (
    ChiSquared,
    ContinuousUniform,
    Exponential,
    Gamma,
    Normal,
    StudentT,
)

__all__ = ["Dataset", "DATASETS", "dataset_keys", "generate", "load_file"]


@dataclass(frozen=True)
class Dataset:
    """A named synthetic sample generator.

    Attributes
    ----------
    key : str
        Stable identifier.
    label : str
        Display name.
    n : int
        Default sample size.
    expect : str
        What the Q-Q plot should look like, so the user can check their reading
        of the plot against the truth.
    suggested_dist : str
        Reference distribution that makes the example instructive.
    draw : Callable
        ``(n, rng) -> np.ndarray``.
    """

    key: str
    label: str
    n: int
    expect: str
    suggested_dist: str
    draw: Callable[[int, np.random.Generator], np.ndarray]


def _seed(rng: np.random.Generator) -> int:
    """Draw a seed for a wedgestats ``rvs`` call from *rng*."""
    return int(rng.integers(0, 2**31 - 1))


def _normal(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(Normal(mu=100.0, sigma=15.0).rvs(size=n, random_state=_seed(rng)))


def _heavy(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(StudentT(df=3).rvs(size=n, random_state=_seed(rng)))


def _light(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(
        ContinuousUniform(low=-1.0, high=1.0).rvs(size=n, random_state=_seed(rng))
    )


def _right_skew(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(Gamma(alpha=2.0, beta=0.35).rvs(size=n, random_state=_seed(rng)))


def _exponential(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(Exponential(lam=0.25).rvs(size=n, random_state=_seed(rng)))


def _chi_squared(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(ChiSquared(df=4).rvs(size=n, random_state=_seed(rng)))


def _contaminated(n: int, rng: np.random.Generator) -> np.ndarray:
    """Ninety-five percent clean normal, five percent from a wide normal."""
    clean = np.asarray(Normal(mu=50.0, sigma=5.0).rvs(size=n, random_state=_seed(rng)))
    n_bad = max(1, int(round(0.05 * n)))
    idx = rng.choice(n, size=n_bad, replace=False)
    clean[idx] = np.asarray(
        Normal(mu=50.0, sigma=28.0).rvs(size=n_bad, random_state=_seed(rng))
    )
    return clean


def _rounded(n: int, rng: np.random.Generator) -> np.ndarray:
    """Normal data recorded to the nearest whole unit."""
    return np.round(
        np.asarray(Normal(mu=20.0, sigma=2.2).rvs(size=n, random_state=_seed(rng)))
    )


def _bimodal(n: int, rng: np.random.Generator) -> np.ndarray:
    half = n // 2
    left = np.asarray(Normal(mu=-2.2, sigma=1.0).rvs(size=half, random_state=_seed(rng)))
    right = np.asarray(
        Normal(mu=2.2, sigma=1.0).rvs(size=n - half, random_state=_seed(rng))
    )
    return np.concatenate([left, right])


def _null_pvalues(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(
        ContinuousUniform(low=0.0, high=1.0).rvs(size=n, random_state=_seed(rng))
    )


def _lognormal(n: int, rng: np.random.Generator) -> np.ndarray:
    z = np.asarray(Normal(mu=0.0, sigma=0.65).rvs(size=n, random_state=_seed(rng)))
    return np.exp(z) * 12.0


DATASETS: dict[str, Dataset] = {
    "normal": Dataset(
        key="normal",
        label="Normal, mu=100 sigma=15 (synthetic)",
        n=80,
        expect="Points on the line, a handful straying outside a pointwise "
        "band. This is what a correct model looks like -- learn it first.",
        suggested_dist="normal",
        draw=_normal,
    ),
    "heavy": Dataset(
        key="heavy",
        label="Student t, df=3 (synthetic)",
        n=120,
        expect="A clear S-shape against a normal reference: low points below "
        "the line, high points above. Heavier tails than normal.",
        suggested_dist="normal",
        draw=_heavy,
    ),
    "light": Dataset(
        key="light",
        label="Uniform on (-1, 1) (synthetic)",
        n=120,
        expect="The reverse S against a normal reference. Tails lighter than "
        "normal, because a uniform has no tails at all.",
        suggested_dist="normal",
        draw=_light,
    ),
    "right_skew": Dataset(
        key="right_skew",
        label="Gamma, alpha=2 beta=0.35 (synthetic)",
        n=100,
        expect="A single upward arc against a normal reference. Switch the "
        "reference to Gamma and it straightens.",
        suggested_dist="normal",
        draw=_right_skew,
    ),
    "exponential": Dataset(
        key="exponential",
        label="Exponential, lambda=0.25 (synthetic)",
        n=100,
        expect="Strong right skew. The natural test case for the exponential "
        "reference and its robust median-based fit.",
        suggested_dist="exponential",
        draw=_exponential,
    ),
    "chi_squared": Dataset(
        key="chi_squared",
        label="Chi-squared, df=4 (synthetic)",
        n=120,
        expect="What a sum of squared residuals should look like when the "
        "model is right.",
        suggested_dist="chi2",
        draw=_chi_squared,
    ),
    "lognormal": Dataset(
        key="lognormal",
        label="Lognormal (synthetic)",
        n=120,
        expect="Pronounced upward curvature. Try Box-Cox at lambda = 0 in the "
        "knowledge base and watch it straighten.",
        suggested_dist="normal",
        draw=_lognormal,
    ),
    "contaminated": Dataset(
        key="contaminated",
        label="Normal with 5% contamination (synthetic)",
        n=100,
        expect="A straight body with a few points flung off both ends. Compare "
        "MLE against the robust median/MAD fit: robust keeps the line on the "
        "bulk so the outliers stay outside the band.",
        suggested_dist="normal",
        draw=_contaminated,
    ),
    "rounded": Dataset(
        key="rounded",
        label="Normal rounded to whole units (synthetic)",
        n=150,
        expect="A staircase. Horizontal runs are ties from the recording "
        "resolution, not evidence against normality.",
        suggested_dist="normal",
        draw=_rounded,
    ),
    "bimodal": Dataset(
        key="bimodal",
        label="Two separated normals (synthetic)",
        n=120,
        expect="A flat middle section between two steep ends -- the signature "
        "of a gap in the middle of the distribution.",
        suggested_dist="normal",
        draw=_bimodal,
    ),
    "null_pvalues": Dataset(
        key="null_pvalues",
        label="p-values under the null (synthetic)",
        n=200,
        expect="Against a Uniform reference these should be straight. This is "
        "the standard calibration check for a testing procedure.",
        suggested_dist="uniform",
        draw=_null_pvalues,
    ),
}


def dataset_keys() -> tuple[str, ...]:
    """Dataset keys in display order."""
    return tuple(DATASETS)


def generate(key: str, n: int | None = None, seed: int = 0) -> np.ndarray:
    """Draw a sample from a named generator.

    Parameters
    ----------
    key : str
        A key of :data:`DATASETS`.
    n : int or None
        Sample size; the dataset's default when ``None``.
    seed : int
        Seed, so an example in a tutorial is reproducible.

    Returns
    -------
    np.ndarray

    Raises
    ------
    KeyError
        If *key* is unknown.
    """
    try:
        spec = DATASETS[key]
    except KeyError:
        raise KeyError(
            f"unknown dataset '{key}'; choose from {', '.join(DATASETS)}"
        ) from None
    size = int(n) if n else spec.n
    if size < 3:
        raise ValueError("sample size must be at least 3")
    return np.asarray(spec.draw(size, np.random.default_rng(seed)), dtype=float).ravel()


def load_file(path: str | Path, column: int | str | None = None) -> tuple[np.ndarray, str]:
    """Read one numeric column from a text, CSV, or TSV file.

    Parameters
    ----------
    path : str or Path
        File to read.  A header row is detected and used for column names.
    column : int or str or None
        Column index or header name.  ``None`` takes the first numeric column.

    Returns
    -------
    tuple[np.ndarray, str]
        The values and the column's name.

    Raises
    ------
    ValueError
        If the file holds no usable numeric column.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"{p.name} is empty")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    delimiter = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else None)
    rows = [ln.split(delimiter) if delimiter else ln.split() for ln in lines]

    header: list[str] | None = None
    first = rows[0]
    try:
        [float(c) for c in first]
    except ValueError:
        header = [c.strip() for c in first]
        rows = rows[1:]
    if not rows:
        raise ValueError(f"{p.name} has a header but no data rows")

    width = max(len(r) for r in rows)
    columns: list[np.ndarray] = []
    for c in range(width):
        values: list[float] = []
        for r in rows:
            if c < len(r):
                try:
                    values.append(float(r[c].strip()))
                except ValueError:
                    values.append(np.nan)
            else:
                values.append(np.nan)
        columns.append(np.array(values, dtype=float))

    def name_of(index: int) -> str:
        if header and index < len(header):
            return header[index]
        return f"column {index + 1}"

    if column is None:
        for idx, col in enumerate(columns):
            if np.sum(np.isfinite(col)) >= 3:
                return col, name_of(idx)
        raise ValueError(f"{p.name} has no column with at least three numeric values")

    if isinstance(column, str):
        if not header or column not in header:
            raise ValueError(f"{p.name} has no column named '{column}'")
        idx = header.index(column)
    else:
        idx = int(column)
        if not 0 <= idx < width:
            raise ValueError(f"{p.name} has {width} column(s); index {idx} is out of range")
    return columns[idx], name_of(idx)
