from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.io import CLEAN_SYNTHESIS, load_all


GROUP_KEYS_SEED = ["model", "synthesis", "eps", "synth_seed"]
GROUP_KEYS_CELL = ["model", "synthesis", "eps", "intervention", "target_gap", "variant"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("data/results"))
    p.add_argument("--baseline-root", type=Path, default=Path("data/results"))
    p.add_argument(
        "--out", type=Path, default=Path("data/analysis/pareto/pareto.csv")
    )
    return p.parse_args()


def _pareto_mask(acc: np.ndarray, gap: np.ndarray) -> np.ndarray:
    # A point is on the front if no other point dominates it: another with
    # acc' >= acc and gap' <= gap with at least one strict.
    n = len(acc)
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not on_front[i]:
            continue
        dominated = (acc >= acc[i]) & (gap <= gap[i]) & ((acc > acc[i]) | (gap < gap[i]))
        if dominated.any():
            on_front[i] = False
    return on_front


def make_pareto(df: pd.DataFrame) -> pd.DataFrame:
    synth = df[(df["synthesis"] != CLEAN_SYNTHESIS) & (df["target_gap"].isin(("dp", "eo")))].copy()
    is_dp = synth["target_gap"] == "dp"
    synth["target_gap_value"] = np.where(
        is_dp, synth["dp_gap"], synth["eo_gap"]
    )

    seed_rows: list[dict] = []
    for keys, g in synth.groupby(GROUP_KEYS_SEED + ["target_gap"], dropna=False, sort=True):
        acc = g["accuracy"].to_numpy(dtype=float)
        gap = g["target_gap_value"].to_numpy(dtype=float)
        on_front = _pareto_mask(acc, gap)
        for is_front, (_, row) in zip(on_front, g.iterrows()):
            seed_rows.append({
                "model": row["model"],
                "synthesis": row["synthesis"],
                "eps": row["eps"],
                "synth_seed": row["synth_seed"],
                "intervention": row["intervention"],
                "target_gap": row["target_gap"],
                "variant": row["variant"],
                "on_front": bool(is_front),
            })
    seeds = pd.DataFrame(seed_rows)
    summary = (
        seeds.groupby(GROUP_KEYS_CELL, dropna=False, sort=True)
        .agg(n=("on_front", "size"), front_rate=("on_front", "mean"))
        .reset_index()
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_all(args.results_root, args.baseline_root, drop_failed=True)
    pareto = make_pareto(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pareto.to_csv(args.out, index=False)
    logging.info("wrote %s (%d rows)", args.out, len(pareto))


if __name__ == "__main__":
    main()
