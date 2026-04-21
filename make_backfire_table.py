#!/usr/bin/env python3
"""EO backfire rate table: fraction of seeds where each intervention's EO gap > unmitigated."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_backfire_table(risk_summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(risk_summary_path)
    df = df[np.isfinite(df["eps"])]
    pivot = (
        df.pivot_table(
            index=["model", "eps"],
            columns="intervention",
            values="eo_backfire_rate",
        )
        .reset_index()
        [["model", "eps", "expgrad", "reweighting", "threshold"]]
        .rename(columns={
            "expgrad": "ExpGrad EO backfire",
            "reweighting": "Reweighting EO backfire",
            "threshold": "Threshold EO backfire",
        })
    )
    pivot["eps"] = pivot["eps"].astype(int)
    for c in pivot.columns[2:]:
        pivot[c] = (pivot[c] * 100).round(0).astype(int).astype(str) + "%"
    return pivot.rename(columns={"model": "model", "eps": "ε"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--risk-summary", type=Path, default=Path("data/results/risk_summary.csv"))
    p.add_argument("--out", type=Path, default=Path("data/results/backfire_table.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    table = make_backfire_table(args.risk_summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
