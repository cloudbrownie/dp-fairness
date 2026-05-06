from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import style as _style  # noqa: F401
from analysis.palette import PALETTE


CAPTION = (
    "Backfire rate per intervention cell × model: probability that the "
    "intervention's target gap exceeds the unmitigated baseline at the "
    "matching (model, synthesizer, ε, seed). Averaged across ε ∈ {1,2,4,8} "
    "and synthesizer ∈ {MST, PrivBayes}. Lower is better; 50% = coin flip."
)


CELL_ORDER = [
    "reweighing",
    "expgrad-dp-uniform",
    "expgrad-dp-stratified",
    "expgrad-eo-uniform",
    "expgrad-eo-stratified",
    "threshold-dp-naive",
    "threshold-dp-honest",
    "threshold-eo-naive",
    "threshold-eo-honest",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--risk-summary",
        type=Path,
        default=Path("data/analysis/risk/risk_summary.csv"),
    )
    p.add_argument(
        "--out", type=Path, default=Path("data/analysis/plots/headline_bar.png")
    )
    return p.parse_args()


def _add_cell(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cell"] = np.where(
        df["intervention"] == "reweighing",
        "reweighing",
        df["intervention"] + "-" + df["target_gap"] + "-" + df["variant"],
    )
    return df


def make_plot(risk_summary: Path, out_path: Path) -> None:
    # Headline figure: backfire rate per (intervention cell × model), averaged
    # across (ε, synth). Direct answer to "how often does this intervention
    # make the target gap worse than no intervention at all?".
    import matplotlib.pyplot as plt

    risk = pd.read_csv(risk_summary)
    risk = _add_cell(risk)

    models = sorted(risk["model"].unique())
    model_colors = {"logreg": PALETTE[0], "xgboost": PALETTE[4]}

    fig, ax = plt.subplots(figsize=(14, 6.0))

    width = 0.4
    x = np.arange(len(CELL_ORDER))
    for i, model in enumerate(models):
        rates: list[float] = []
        for cell in CELL_ORDER:
            r = risk.loc[
                (risk["model"] == model) & (risk["cell"] == cell), "target_backfire_rate"
            ].to_numpy()
            r = r[np.isfinite(r)]
            rates.append(float(np.mean(r)) if len(r) else float("nan"))
        ax.bar(
            x + (i - 0.5) * width,
            rates,
            width,
            label=model,
            color=model_colors.get(model, "gray"),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.axhline(0.5, color="k", linestyle=":", alpha=0.6, linewidth=0.9, label="coin flip")
    ax.set_xticks(x)
    ax.set_xticklabels(CELL_ORDER, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("P[intervention target gap > unmitigated]")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.subplots_adjust(left=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    make_plot(args.risk_summary, args.out)
    logging.info("wrote %s", args.out)
    print(f"\n[{args.out.name}] caption:\n{CAPTION}\n")


if __name__ == "__main__":
    main()
