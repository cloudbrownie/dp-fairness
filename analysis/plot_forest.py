from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import style as _style  # noqa: F401
from analysis.palette import PALETTE


CAPTION = (
    "Target gap per intervention cell across (model, synthesizer) panels — "
    "mean ± 95% CI across 10 seeds. Each row shows the four ε values "
    "(top→bottom: ε=8 best privacy budget, ε=1 worst); ★ marks the clean "
    "(non-DP) baseline; gray dotted vertical line is the unmitigated baseline "
    "at ε=8 for reference."
)


CELL_ORDER = [
    "unmitigated",
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
EPS_VALUES = [1.0, 2.0, 4.0, 8.0]
# Map ε to the palette's good→bad axis: ε=8 (largest budget, most utility) →
# teal end; ε=1 (smallest, worst utility) → yellow end.
EPS_COLORS = {
    8.0: PALETTE[0],
    4.0: PALETTE[1],
    2.0: PALETTE[3],
    1.0: PALETTE[4],
}
SYN_LABELS = {"mst": "MST", "pb": "PrivBayes"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-summary",
        type=Path,
        default=Path("data/analysis/raw_summary/raw_summary_target.csv"),
    )
    p.add_argument(
        "--out", type=Path, default=Path("data/analysis/plots/forest_target_gap.png")
    )
    return p.parse_args()


def _add_cell(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cell"] = np.where(
        df["intervention"] == "unmitigated", "unmitigated",
        np.where(
            df["intervention"] == "reweighing", "reweighing",
            df["intervention"] + "-" + df["target_gap"] + "-" + df["variant"],
        ),
    )
    return df


def _row_for(df: pd.DataFrame, model: str, synth: str, cell: str, eps: float):
    if synth == "clean":
        sub = df[(df["model"] == model) & (df["synthesis"] == "clean") & (df["cell"] == cell)]
    else:
        sub = df[
            (df["model"] == model)
            & (df["synthesis"] == synth)
            & (df["cell"] == cell)
            & (df["eps"] == eps)
        ]
    if sub.empty:
        return float("nan"), float("nan"), float("nan")
    r = sub.iloc[0]
    if "target_gap_mean" in r and not np.isnan(r["target_gap_mean"]):
        return float(r["target_gap_mean"]), float(r["target_gap_ci_lo"]), float(r["target_gap_ci_hi"])
    if cell == "unmitigated":
        return float(r["dp_gap_mean"]), float(r["dp_gap_ci_lo"]), float(r["dp_gap_ci_hi"])
    return float("nan"), float("nan"), float("nan")


def make_plot(raw_summary: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(raw_summary)
    df = _add_cell(df)

    models = sorted(df["model"].unique())
    syntheses = ["mst", "pb"]
    n_panels = len(models) * len(syntheses)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.6 * n_panels, 0.65 * len(CELL_ORDER) + 1.8),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    y_pos = list(range(len(CELL_ORDER)))[::-1]

    # Vertical jitter so the 4 ε dots fan out around the row center instead of
    # stacking. Order: ε=8 highest (good), ε=1 lowest (bad), preserving the
    # palette's good→bad axis vertically as well as in color.
    eps_jitter = {8.0: 0.27, 4.0: 0.09, 2.0: -0.09, 1.0: -0.27}

    for k, (model, synth) in enumerate([(m, s) for m in models for s in syntheses]):
        ax = axes[k]
        for i, cell in enumerate(CELL_ORDER):
            y = y_pos[i]
            for eps in EPS_VALUES:
                mean, lo, hi = _row_for(df, model, synth, cell, eps)
                if np.isnan(mean):
                    continue
                color = EPS_COLORS[eps]
                ax.errorbar(
                    mean, y + eps_jitter[eps],
                    xerr=[[mean - lo], [hi - mean]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.0,
                    capsize=2.5, markersize=5,
                )
            mean_c, lo_c, hi_c = _row_for(df, model, "clean", cell, np.nan)
            if not np.isnan(mean_c):
                ax.scatter(
                    mean_c, y, marker="*", s=180, color="black",
                    edgecolor="white", linewidth=0.6, zorder=3,
                )
            if i < len(CELL_ORDER) - 1:
                ax.axhline(y - 0.5, color="gray", linestyle="-", alpha=0.15, linewidth=0.5)

        # Reference line for unmitigated mean (averaged across eps for synth, or
        # clean value if no synth row available). We use the synth ε=8 value as
        # an approximate "unmitigated baseline at high privacy budget" reference.
        unm_mean, _, _ = _row_for(df, model, synth, "unmitigated", 8.0)
        if not np.isnan(unm_mean):
            ax.axvline(unm_mean, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(CELL_ORDER)
        ax.set_xlabel("Target gap (lower is fairer)")
        ax.set_title(f"{model} · {SYN_LABELS.get(synth, synth)}")
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(left=0.0)

    handles = [
        plt.Line2D([0], [0], marker="o", color=c, linestyle="",
                   markersize=6, label=f"ε={int(e)}") for e, c in EPS_COLORS.items()
    ]
    handles.append(
        plt.Line2D([0], [0], marker="*", color="black", linestyle="",
                   markersize=10, label="clean baseline")
    )
    handles.append(
        plt.Line2D([0], [0], color="gray", linestyle=":",
                   linewidth=0.8, label="unmitigated (ε=8)")
    )
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles),
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    make_plot(args.raw_summary, args.out)
    logging.info("wrote %s", args.out)
    print(f"\n[{args.out.name}] caption:\n{CAPTION}\n")


if __name__ == "__main__":
    main()
