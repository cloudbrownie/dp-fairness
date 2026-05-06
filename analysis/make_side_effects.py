from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.ci import agg_factory
from analysis.io import CLEAN_SYNTHESIS, load_all
from analysis.metrics import attach_unmitigated
from analysis.stats_tests import wilcoxon_two_sided


GROUP_KEYS_CELL = ["model", "synthesis", "eps", "intervention", "target_gap", "variant"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("data/results"))
    p.add_argument("--baseline-root", type=Path, default=Path("data/results"))
    p.add_argument(
        "--out-dir", type=Path, default=Path("data/analysis/side_effects")
    )
    return p.parse_args()


def make_side_effects(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # For each cell with target_gap in {dp, eo}, "off-target" is the *other*
    # gap. We report (a) summary of off-target gap, and (b) Wilcoxon delta vs
    # unmitigated on that off-target gap.
    synth = df[df["synthesis"] != CLEAN_SYNTHESIS]
    paired = attach_unmitigated(synth)
    paired = paired[paired["target_gap"].isin(("dp", "eo"))].copy()

    is_dp = paired["target_gap"] == "dp"
    paired["offtarget_metric"] = np.where(is_dp, "eo_gap", "dp_gap")
    paired["offtarget_value"] = np.where(is_dp, paired["eo_gap"], paired["dp_gap"])
    paired["offtarget_unm"] = np.where(
        is_dp, paired["eo_gap_unm"], paired["dp_gap_unm"]
    )
    paired["delta_offtarget"] = paired["offtarget_value"] - paired["offtarget_unm"]

    summary = (
        paired.groupby(GROUP_KEYS_CELL + ["offtarget_metric"], dropna=False, sort=True)[["offtarget_value"]]
        .apply(agg_factory(("offtarget_value",)))
        .reset_index()
    )

    rows: list[dict] = []
    for keys, g in paired.groupby(
        GROUP_KEYS_CELL + ["offtarget_metric"], dropna=False, sort=True
    ):
        median, p = wilcoxon_two_sided(g["delta_offtarget"].to_numpy())
        row = dict(zip(GROUP_KEYS_CELL + ["offtarget_metric"], keys))
        row["n"] = int(len(g))
        row["delta_offtarget_median"] = median
        row["p_offtarget"] = p
        rows.append(row)
    tests = pd.DataFrame(rows)
    return summary, tests


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_all(args.results_root, args.baseline_root, drop_failed=True)
    summary, tests = make_side_effects(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sum_path = args.out_dir / "side_effects_summary.csv"
    test_path = args.out_dir / "side_effects_wilcoxon.csv"
    summary.to_csv(sum_path, index=False)
    tests.to_csv(test_path, index=False)
    logging.info("wrote %s (%d rows)", sum_path, len(summary))
    logging.info("wrote %s (%d rows)", test_path, len(tests))


if __name__ == "__main__":
    main()
