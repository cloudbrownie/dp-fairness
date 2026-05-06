from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(v: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float, int]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = int(len(v))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(v))
    if n < 2:
        return mean, mean, mean, n
    se = float(np.std(v, ddof=1) / np.sqrt(n))
    h = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1)) * se
    return mean, mean - h, mean + h, n


def agg_factory(cols: Iterable[str], alpha: float = 0.05):
    cols = tuple(cols)

    def _agg(g: pd.DataFrame) -> pd.Series:
        out: dict[str, float | int] = {}
        for col in cols:
            mean, lo, hi, n = mean_ci(g[col].to_numpy(), alpha=alpha)
            out[f"{col}_mean"] = mean
            out[f"{col}_ci_lo"] = lo
            out[f"{col}_ci_hi"] = hi
            out[f"{col}_n"] = n
        return pd.Series(out)

    return _agg
