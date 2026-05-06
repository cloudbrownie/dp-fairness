from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis import style as _style  # noqa: F401
from analysis.io import CLEAN_SYNTHESIS, load_all
from analysis.palette import PALETTE


CAPTION = (
    "Fairness degradation under DP synthesis per intervention cell × model: "
    "mean (synth target gap − clean target gap) ± 95% CI, averaged across "
    "ε ∈ {1,2,4,8} and synthesizer ∈ {MST, PrivBayes}. Positive bars = DP "
    "synthesis worsens the cell's target gap; negative bars = paradoxical "
    "improvement."
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
    p.add_argument("--results-root", type=Path, default=Path("data/results"))
    p.add_argument("--baseline-root", type=Path, default=Path("data/results"))
    p.add_argument(
        "--out", type=Path, default=Path("data/analysis/plots/headline_delta.png")
    )
    return p.parse_args()


def _ci_half(v: np.ndarray, alpha: float = 0.05) -> float:
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return 0.0
    se = float(np.std(v, ddof=1) / np.sqrt(n))
    return float(stats.t.ppf(1.0 - alpha / 2.0, n - 1)) * se


def _add_cell(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cell"] = np.where(
        df["intervention"] == "reweighing", "reweighing",
        df["intervention"] + "-" + df["target_gap"] + "-" + df["variant"],
    )
    return df


def make_plot(df: pd.DataFrame, out_path: Path) -> None:
    # Headline figure: per-cell, per-seed delta = synth_target_gap −
    # clean_target_gap, averaged across (ε, synth). Direct measure of "how
    # much fairness do you lose to DP synthesis?". Bounded, interpretable,
    # zero is meaningful (no degradation).
    import matplotlib.pyplot as plt

    df = _add_cell(df)
    df = df[df["intervention"] != "unmitigated"].copy()
    is_dp = df["target_gap"] == "dp"
    df["target_gap_value"] = np.where(is_dp, df["dp_gap"], df["eo_gap"])

    cell_keys = ["model", "intervention", "target_gap", "variant"]
    clean = (
        df[df["synthesis"] == CLEAN_SYNTHESIS]
        .groupby(cell_keys, as_index=False)["target_gap_value"]
        .mean()
        .rename(columns={"target_gap_value": "target_gap_clean"})
    )
    synth = df[df["synthesis"] != CLEAN_SYNTHESIS].merge(
        clean, on=cell_keys, how="left", validate="many_to_one"
    )
    synth["delta"] = synth["target_gap_value"] - synth["target_gap_clean"]
    synth = _add_cell(synth)

    models = sorted(synth["model"].unique())
    model_colors = {"logreg": PALETTE[0], "xgboost": PALETTE[4]}

    fig, ax = plt.subplots(figsize=(14, 6.0))

    width = 0.4
    x = np.arange(len(CELL_ORDER))
    for i, model in enumerate(models):
        means: list[float] = []
        cis: list[float] = []
        for cell in CELL_ORDER:
            v = synth.loc[
                (synth["model"] == model) & (synth["cell"] == cell), "delta"
            ].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v):
                means.append(float(np.mean(v)))
                cis.append(_ci_half(v))
            else:
                means.append(float("nan"))
                cis.append(0.0)
        ax.bar(
            x + (i - 0.5) * width, means, width,
            label=model, color=model_colors.get(model, "gray"),
            yerr=cis, capsize=4, edgecolor="black", linewidth=0.5,
        )

    ax.axhline(0.0, color="k", linestyle=":", alpha=0.6, linewidth=0.9,
               label="no degradation")
    ax.set_xticks(x)
    ax.set_xticklabels(CELL_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Synth target gap − clean target gap")
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
    df = load_all(args.results_root, args.baseline_root, drop_failed=True)
    make_plot(df, args.out)
    logging.info("wrote %s", args.out)
    print(f"\n[{args.out.name}] caption:\n{CAPTION}\n")


if __name__ == "__main__":
    main()
