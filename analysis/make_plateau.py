from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.io import CLEAN_SYNTHESIS, load_all
from analysis.stats_tests import wilcoxon_two_sided


EPS_PAIRS: tuple[tuple[float, float], ...] = ((1.0, 8.0), (2.0, 8.0), (4.0, 8.0))
METRICS: tuple[str, ...] = ("dp_gap", "eo_gap", "accuracy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("data/results"))
    p.add_argument("--baseline-root", type=Path, default=Path("data/results"))
    p.add_argument(
        "--out", type=Path, default=Path("data/analysis/plateau/plateau_tests.csv")
    )
    return p.parse_args()


def make_plateau(df: pd.DataFrame) -> pd.DataFrame:
    synth = df[df["synthesis"] != CLEAN_SYNTHESIS]
    rows: list[dict] = []
    cell_keys = ["model", "synthesis", "intervention", "target_gap", "variant"]
    for keys, g in synth.groupby(cell_keys, dropna=False, sort=True):
        for eps_a, eps_b in EPS_PAIRS:
            a = g[g["eps"] == eps_a].set_index("synth_seed")
            b = g[g["eps"] == eps_b].set_index("synth_seed")
            shared = a.index.intersection(b.index)
            if len(shared) == 0:
                continue
            a = a.loc[shared]
            b = b.loc[shared]
            for metric in METRICS:
                d = (b[metric].to_numpy() - a[metric].to_numpy()).astype(float)
                median, p = wilcoxon_two_sided(d)
                row = dict(zip(cell_keys, keys))
                row["eps_a"] = eps_a
                row["eps_b"] = eps_b
                row["metric"] = metric
                row["n"] = int(np.sum(np.isfinite(d)))
                row["median_delta"] = median
                row["wilcoxon_p"] = p
                rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_all(args.results_root, args.baseline_root, drop_failed=True)
    plateau = make_plateau(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plateau.to_csv(args.out, index=False)
    logging.info("wrote %s (%d rows)", args.out, len(plateau))


if __name__ == "__main__":
    main()
