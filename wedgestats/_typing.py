"""Shared type aliases for wedgestats."""

from typing import Sequence

import numpy as np

Numeric = int | float
ArrayLike = Sequence[Numeric] | np.ndarray
