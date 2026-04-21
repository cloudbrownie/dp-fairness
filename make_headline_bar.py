#!/usr/bin/env python3
"""Bar chart: mean EO benefit preservation per (model, intervention), across ε × seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def make_headline_bar(ratios_long_path: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(ratios_long_path)
    agg = []
    for (model, inter), g in df.groupby(["model", "intervention"], sort=True):
        v = g["eo_ratio"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        n = len(v)
        mean = float(np.mean(v))
        se = float(np.std(v, ddof=1) / np.sqrt(n))
        h = float(stats.t.ppf(0.975, n - 1)) * se
        agg.append({"model": model, "intervention": inter, "mean": mean, "ci": h, "n": n})
    agg = pd.DataFrame(agg)

    interventions = ["reweighting", "expgrad", "threshold"]
    models = sorted(agg["model"].unique())
    colors = {"logreg": "#1f77b4", "xgboost": "#ff7f0e"}
    x = np.arange(len(interventions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        means = [
            float(agg[(agg["model"] == model) & (agg["intervention"] == inter)]["mean"].iloc[0])
            for inter in interventions
        ]
        cis = [
            float(agg[(agg["model"] == model) & (agg["intervention"] == inter)]["ci"].iloc[0])
            for inter in interventions
        ]
        ax.bar(
            x + (i - 0.5) * width, means, width, label=model, color=colors.get(model, "gray"),
            yerr=cis, capsize=5, edgecolor="black", linewidth=0.5,
        )

    ax.axhline(1.0, color="k", linestyle="--", alpha=0.6, linewidth=0.9, label="benefit preserved")
    ax.axhline(0.0, color="k", linestyle=":", alpha=0.7, linewidth=0.9, label="backfire threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(interventions)
    ax.set_ylabel("EO benefit preservation\n(synth benefit / clean benefit, mean ± 95% CI)")
    ax.set_title(
        "How much of each intervention's clean EO benefit survives DP?\n"
        "Averaged across ε ∈ {1,2,4,8} × 10 seeds"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ratios-long", type=Path, default=Path("data/results/ratios_long.csv"))
    p.add_argument("--out", type=Path, default=Path("data/results/headline_bar.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    make_headline_bar(args.ratios_long, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
