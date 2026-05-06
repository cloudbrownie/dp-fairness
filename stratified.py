"""Stratified subsampling indices for group-aware downsampling.

Used by expgrad oracle calls so every group that has at least `min_per_group`
rows is guaranteed that many in the fit subsample; groups smaller than
`min_per_group` contribute all their rows. Remaining budget is allocated
proportionally to each group's remaining pool (size minus floor).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_subsample_idx(
    groups: np.ndarray | pd.Series,
    target_size: int,
    min_per_group: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if isinstance(groups, pd.Series):
        groups_arr = groups.to_numpy()
    else:
        groups_arr = np.asarray(groups)

    unique = np.unique(groups_arr)
    idx_by_group: dict[object, np.ndarray] = {
        g: np.flatnonzero(groups_arr == g) for g in unique
    }
    sizes: dict[object, int] = {g: int(len(idx_by_group[g])) for g in unique}

    floors: dict[object, int] = {
        g: int(min(min_per_group, sizes[g])) for g in unique
    }
    floor_total = sum(floors.values())

    remaining_budget = max(0, int(target_size) - floor_total)
    remaining_pool = {g: max(0, sizes[g] - floors[g]) for g in unique}
    total_remaining_pool = sum(remaining_pool.values())

    if total_remaining_pool > 0 and remaining_budget > 0:
        extras = {
            g: int(round(remaining_budget * remaining_pool[g] / total_remaining_pool))
            for g in unique
        }
    else:
        extras = {g: 0 for g in unique}

    picked: list[np.ndarray] = []
    for g in unique:
        take = min(floors[g] + extras[g], sizes[g])
        if take == sizes[g]:
            chosen = idx_by_group[g]
        else:
            chosen = rng.choice(idx_by_group[g], size=take, replace=False)
        picked.append(chosen)

    return np.concatenate(picked)


def per_group_counts(
    groups: np.ndarray | pd.Series, idx: np.ndarray
) -> dict[object, int]:
    if isinstance(groups, pd.Series):
        groups_arr = groups.to_numpy()
    else:
        groups_arr = np.asarray(groups)
    taken = groups_arr[idx]
    unique, counts = np.unique(taken, return_counts=True)
    return {u: int(c) for u, c in zip(unique, counts)}
