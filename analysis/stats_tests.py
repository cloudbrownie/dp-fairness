from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wilcoxon_two_sided(d: np.ndarray) -> tuple[float, float]:
    # Returns (median_delta, p_value). p is NaN when n<2 or all-zero.
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan")
    median = float(np.median(d))
    if len(d) < 2 or np.allclose(d, 0):
        return median, float("nan")
    try:
        stat = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        return median, float(stat.pvalue)
    except ValueError:
        return median, float("nan")


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    # Step-down Holm-Bonferroni adjustment. NaN p-values pass through as NaN
    # and do not consume a comparison slot.
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    out = np.full_like(p, np.nan, dtype=float)
    if finite.sum() == 0:
        return out
    idx = np.where(finite)[0]
    sub = p[idx]
    order = np.argsort(sub)
    m = len(sub)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, k in enumerate(order):
        val = (m - rank) * sub[k]
        running_max = max(running_max, val)
        adj[k] = min(running_max, 1.0)
    out[idx] = adj
    return out
