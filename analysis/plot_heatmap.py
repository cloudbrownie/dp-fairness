from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import style as _style  # noqa: F401
from analysis.palette import cmap as palette_cmap

CAPTION = (
    "Target gap per (intervention cell × ε) — mean across 10 seeds. "
    "Darker = larger gap. Vertical line separates clean baseline from DP synth."
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
COL_ORDER = ["clean", 1.0, 2.0, 4.0, 8.0]
COL_LABELS = ["clean", "ε=1", "ε=2", "ε=4", "ε=8"]
SYN_LABELS = {"mst": "MST", "pb": "PrivBayes"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-summary",
        type=Path,
        default=Path("data/analysis/raw_summary/raw_summary_target.csv"),
    )
    p.add_argument(
        "--out", type=Path, default=Path("data/analysis/plots/heatmap_target_gap.png")
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


def _value_for(df: pd.DataFrame, model: str, synth: str, cell: str, col: str | float) -> float:
    if col == "clean":
        sub = df[(df["model"] == model) & (df["synthesis"] == "clean") & (df["cell"] == cell)]
    else:
        sub = df[
            (df["model"] == model)
            & (df["synthesis"] == synth)
            & (df["cell"] == cell)
            & (df["eps"] == col)
        ]
    if sub.empty:
        return float("nan")
    r = sub.iloc[0]
    val = r.get("target_gap_mean", np.nan)
    if isinstance(val, float) and np.isnan(val) and cell == "unmitigated":
        val = r["dp_gap_mean"]
    return float(val)


def make_plot(raw_summary: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(raw_summary)
    df = _add_cell(df)

    models = sorted(df["model"].unique())
    syntheses = ["mst", "pb"]

    matrices: dict[tuple[str, str], np.ndarray] = {}
    for model in models:
        for synth in syntheses:
            mat = np.full((len(CELL_ORDER), len(COL_ORDER)), np.nan)
            for i, cell in enumerate(CELL_ORDER):
                for j, col in enumerate(COL_ORDER):
                    mat[i, j] = _value_for(df, model, synth, cell, col)
            matrices[(model, synth)] = mat

    finite_vals = np.concatenate([m[np.isfinite(m)] for m in matrices.values()])
    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))

    fig, axes = plt.subplots(
        len(syntheses), len(models),
        figsize=(5.0 * len(models), 4.5 * len(syntheses)),
        squeeze=False,
    )

    for i, synth in enumerate(syntheses):
        for j, model in enumerate(models):
            ax = axes[i, j]
            mat = matrices[(model, synth)]
            im = ax.imshow(
                mat, aspect="auto", cmap=palette_cmap(), vmin=vmin, vmax=vmax,
            )
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    v = mat[r, c]
                    if np.isfinite(v):
                        # Palette runs dark-teal (low data value) → light-yellow
                        # (high). Use white text on dark cells (low rel), black
                        # on light cells (high rel).
                        rel = (v - vmin) / max(vmax - vmin, 1e-9)
                        text_color = "white" if rel < 0.45 else "black"
                        ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                                color=text_color, fontsize=8)
                    else:
                        ax.text(c, r, "—", ha="center", va="center",
                                color="gray", fontsize=8)
            ax.axvline(0.5, color="black", linestyle="-", linewidth=1.2)
            ax.set_xticks(range(len(COL_ORDER)))
            ax.set_xticklabels(COL_LABELS)
            if j == 0:
                ax.set_yticks(range(len(CELL_ORDER)))
                ax.set_yticklabels(CELL_ORDER, fontsize=8)
            else:
                ax.set_yticks([])
            ax.set_title(f"{model} · {SYN_LABELS.get(synth, synth)}")

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.014, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Target gap (lower = fairer)")

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
