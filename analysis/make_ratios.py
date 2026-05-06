from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.ci import agg_factory
from analysis.io import CLEAN_SYNTHESIS, load_all
from analysis.metrics import per_seed_ratios


GROUP_KEYS_CELL = ["model", "synthesis", "eps", "intervention", "target_gap", "variant"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("data/results"))
    p.add_argument("--baseline-root", type=Path, default=Path("data/results"))
    p.add_argument(
        "--out-dir", type=Path, default=Path("data/analysis/ratios")
    )
    return p.parse_args()


def make_ratios_long(df: pd.DataFrame) -> pd.DataFrame:
    # Excludes unmitigated rows (ratio undefined). Keeps the synth row +
    # broadcast clean benefit, plus the per-seed synth_target_benefit and
    # target_ratio.
    long = per_seed_ratios(df)
    keep = [
        "model", "synthesis", "eps", "synth_seed",
        "intervention", "target_gap", "variant",
        "accuracy", "auc", "dp_gap", "eo_gap",
        "dp_benefit", "eo_benefit",
        "synth_target_benefit", "clean_target_benefit", "target_ratio",
    ]
    return long[keep].reset_index(drop=True)


def make_ratios_summary(ratios_long: pd.DataFrame) -> pd.DataFrame:
    summary = (
        ratios_long.groupby(GROUP_KEYS_CELL, dropna=False, sort=True)[["target_ratio"]]
        .apply(agg_factory(("target_ratio",)))
        .reset_index()
    )
    return summary


def make_attenuation(ratios_long: pd.DataFrame) -> pd.DataFrame:
    # Mean synth target benefit per cell minus mean clean benefit per cell.
    # attenuation = 1 - synth_benefit_mean / clean_benefit_mean.
    grp = (
        ratios_long.groupby(GROUP_KEYS_CELL, dropna=False, sort=True)
        .agg(
            n=("synth_target_benefit", "count"),
            synth_target_benefit_mean=("synth_target_benefit", "mean"),
            clean_target_benefit_mean=("clean_target_benefit", "mean"),
        )
        .reset_index()
    )
    grp["attenuation"] = 1.0 - grp["synth_target_benefit_mean"] / grp["clean_target_benefit_mean"]
    return grp


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_all(args.results_root, args.baseline_root, drop_failed=True)
    ratios_long = make_ratios_long(df)
    ratios_summary = make_ratios_summary(ratios_long)
    atten = make_attenuation(ratios_long)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.out_dir / "ratios_long.csv"
    summary_path = args.out_dir / "ratios_summary.csv"
    atten_path = args.out_dir / "attenuation.csv"
    ratios_long.to_csv(long_path, index=False)
    ratios_summary.to_csv(summary_path, index=False)
    atten.to_csv(atten_path, index=False)
    logging.info("wrote %s (%d rows)", long_path, len(ratios_long))
    logging.info("wrote %s (%d rows)", summary_path, len(ratios_summary))
    logging.info("wrote %s (%d rows)", atten_path, len(atten))


if __name__ == "__main__":
    main()
