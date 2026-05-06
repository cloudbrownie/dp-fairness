from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from analysis.io import load_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("data/results"))
    p.add_argument("--baseline-root", type=Path, default=Path("data/results"))
    p.add_argument(
        "--out-dir", type=Path, default=Path("data/analysis/failures")
    )
    return p.parse_args()


def make_failures(df_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fails = df_all[df_all["error"] != ""].copy()
    long_cols = [
        "model", "synthesis", "eps", "synth_seed",
        "intervention", "target_gap", "variant", "error",
    ]
    long = fails[long_cols].sort_values(long_cols).reset_index(drop=True)

    summary = (
        fails.groupby(
            ["model", "synthesis", "intervention", "target_gap", "variant"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_failed"})
        .sort_values(["model", "synthesis", "intervention", "target_gap", "variant"])
        .reset_index(drop=True)
    )
    return long, summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df_all = load_all(args.results_root, args.baseline_root, drop_failed=False)
    long, summary = make_failures(df_all)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.out_dir / "failures_long.csv"
    summary_path = args.out_dir / "failures_summary.csv"
    long.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    logging.info("wrote %s (%d rows)", long_path, len(long))
    logging.info("wrote %s (%d rows)", summary_path, len(summary))
    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("(no failed cells)")


if __name__ == "__main__":
    main()
