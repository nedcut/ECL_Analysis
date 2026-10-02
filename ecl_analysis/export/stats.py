"""Summary statistics shared by the CSV summary and plot annotations."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def sample_std(values: Iterable[float]) -> Optional[float]:
    """Return the sample standard deviation (ddof=1) of a series.

    Returns None when fewer than two finite values are available, because a
    sample standard deviation is undefined for n < 2.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    return float(np.std(arr, ddof=1))


def format_std(std: Optional[float], digits: int) -> str:
    """Format a value from :func:`sample_std`, using 'n/a' when undefined."""
    return "n/a" if std is None else f"{std:.{digits}f}"
