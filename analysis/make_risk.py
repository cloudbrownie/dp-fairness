from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.io import CLEAN_SYNTHESIS, load_all
from analysis.metrics import attach_unmitigated


GROUP_KEYS_CELL = ["model", "synthesis", "eps", "intervention", "target_gap", "variant"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("data/results"))
    p.add_argument("--baseline-root", type=Path, default=Path("data/results"))
    p.add_argument(
        "--out", type=Path, default=Path("data/analysis/risk/risk_summary.csv")
    )
    return p.parse_args()


def make_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Synth-only. Backfire / worst / p90 are computed on the *target gap*.
    synth = df[df["synthesis"] != CLEAN_SYNTHESIS]
    paired = attach_unmitigated(synth)

    is_dp = paired["target_gap"] == "dp"
    is_eo = paired["target_gap"] == "eo"
    paired["target_gap_value"] = np.where(
        is_dp, paired["dp_gap"], np.where(is_eo, paired["eo_gap"], np.nan)
    )
    paired["target_gap_unm"] = np.where(
        is_dp, paired["dp_gap_unm"], np.where(is_eo, paired["eo_gap_unm"], np.nan)
    )

    paired = paired[paired["target_gap"].isin(("dp", "eo"))]

    rows: list[dict] = []
    for keys, g in paired.groupby(GROUP_KEYS_CELL, dropna=False, sort=True):
        gap = g["target_gap_value"].to_numpy()
        unm = g["target_gap_unm"].to_numpy()
        ok = np.isfinite(gap) & np.isfinite(unm)
        gap = gap[ok]
        unm = unm[ok]
        n = len(gap)
        row = dict(zip(GROUP_KEYS_CELL, keys))
        row["n"] = n
        if n:
            row["target_worst"] = float(np.max(gap))
            row["target_p90"] = float(np.quantile(gap, 0.9))
            row["target_backfire_rate"] = float(np.mean(gap > unm))
        else:
            row["target_worst"] = float("nan")
            row["target_p90"] = float("nan")
            row["target_backfire_rate"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_all(args.results_root, args.baseline_root, drop_failed=True)
    risk = make_risk_summary(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    risk.to_csv(args.out, index=False)
    logging.info("wrote %s (%d rows)", args.out, len(risk))


if __name__ == "__main__":
    main()
