from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.ci import agg_factory
from analysis.io import load_all


GROUP_KEYS = ["model", "synthesis", "eps", "intervention", "target_gap", "variant"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("data/results"))
    p.add_argument("--baseline-root", type=Path, default=Path("data/results"))
    p.add_argument(
        "--out-dir", type=Path, default=Path("data/analysis/raw_summary")
    )
    return p.parse_args()


def make_raw_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Per (model, synthesis, eps, intervention, target_gap, variant): mean ±
    # 95% CI for dp_gap, eo_gap, accuracy, auc. Both gaps included so this CSV
    # also serves as the unmitigated-reference table.
    cols = ("dp_gap", "eo_gap", "accuracy", "auc")
    summary = (
        df.groupby(GROUP_KEYS, dropna=False, sort=True)[list(cols)]
        .apply(agg_factory(cols))
        .reset_index()
    )
    return summary


def make_target_gap_summary(raw_summary: pd.DataFrame) -> pd.DataFrame:
    # Project the raw summary onto the *target* metric so plotting code can
    # consume one column instead of branching on target_gap.
    out = raw_summary.copy()
    is_dp = out["target_gap"] == "dp"
    is_eo = out["target_gap"] == "eo"
    target_cols = ("mean", "ci_lo", "ci_hi", "n")
    for tag in target_cols:
        out[f"target_gap_{tag}"] = np.where(
            is_dp, out[f"dp_gap_{tag}"],
            np.where(is_eo, out[f"eo_gap_{tag}"], np.nan),
        )
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_all(args.results_root, args.baseline_root, drop_failed=True)
    raw = make_raw_summary(df)
    target = make_target_gap_summary(raw)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw_summary.csv"
    target_path = args.out_dir / "raw_summary_target.csv"
    raw.to_csv(raw_path, index=False)
    target.to_csv(target_path, index=False)
    logging.info("wrote %s (%d rows)", raw_path, len(raw))
    logging.info("wrote %s (%d rows)", target_path, len(target))


if __name__ == "__main__":
    main()
