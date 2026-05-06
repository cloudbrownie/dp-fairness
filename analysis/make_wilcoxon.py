from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.io import CLEAN_SYNTHESIS, load_all
from analysis.metrics import attach_unmitigated
from analysis.stats_tests import holm_bonferroni, wilcoxon_two_sided


GROUP_KEYS_CELL = ["model", "synthesis", "eps", "intervention", "target_gap", "variant"]
FAMILY_KEYS = ["model", "synthesis", "eps"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("data/results"))
    p.add_argument("--baseline-root", type=Path, default=Path("data/results"))
    p.add_argument(
        "--out", type=Path, default=Path("data/analysis/wilcoxon/wilcoxon.csv")
    )
    return p.parse_args()


def make_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    # Paired across seeds, tests target gap delta + accuracy delta. Holm-
    # Bonferroni adjustment is computed within each (model, synthesis, eps)
    # family of 9 cells.
    synth = df[df["synthesis"] != CLEAN_SYNTHESIS]
    paired = attach_unmitigated(synth)
    paired = paired[paired["target_gap"].isin(("dp", "eo"))]

    is_dp = paired["target_gap"] == "dp"
    paired["delta_target"] = np.where(
        is_dp,
        paired["dp_gap"] - paired["dp_gap_unm"],
        paired["eo_gap"] - paired["eo_gap_unm"],
    )
    paired["delta_acc"] = paired["accuracy"] - paired["accuracy_unm"]

    rows: list[dict] = []
    for keys, g in paired.groupby(GROUP_KEYS_CELL, dropna=False, sort=True):
        median_target, p_target = wilcoxon_two_sided(g["delta_target"].to_numpy())
        median_acc, p_acc = wilcoxon_two_sided(g["delta_acc"].to_numpy())
        row = dict(zip(GROUP_KEYS_CELL, keys))
        row["n"] = int(len(g))
        row["delta_target_median"] = median_target
        row["p_target"] = p_target
        row["delta_acc_median"] = median_acc
        row["p_acc"] = p_acc
        rows.append(row)
    out = pd.DataFrame(rows)

    out["p_target_holm"] = np.nan
    out["p_acc_holm"] = np.nan
    for _, idx in out.groupby(FAMILY_KEYS, dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "p_target_holm"] = holm_bonferroni(out.loc[idx, "p_target"].to_numpy())
        out.loc[idx, "p_acc_holm"] = holm_bonferroni(out.loc[idx, "p_acc"].to_numpy())
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_all(args.results_root, args.baseline_root, drop_failed=True)
    wil = make_wilcoxon(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wil.to_csv(args.out, index=False)
    logging.info("wrote %s (%d rows)", args.out, len(wil))


if __name__ == "__main__":
    main()
