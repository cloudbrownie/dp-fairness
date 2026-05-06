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


CAPTION_TEMPLATE = (
    "Accuracy vs {track} gap — per-(intervention cell, ε) means with 95% CI "
    "ellipses across 10 seeds. Marker shape = variant; fill color = "
    "intervention family; size encodes ε (small ε=1 → large ε=8). "
    "★ marks the clean (non-DP) baseline."
)


COLORS = {
    "unmitigated": "#7f7f7f",
    "reweighing": PALETTE[0],
    "expgrad": PALETTE[2],
    "threshold": PALETTE[4],
}
MARKERS_VARIANT = {
    "none": "o",
    "uniform": "s",
    "stratified": "D",
    "naive": "^",
    "honest": "v",
}
EPS_SIZES = {1.0: 40, 2.0: 70, 4.0: 110, 8.0: 160}
SYN_LABELS = {"mst": "MST", "pb": "PrivBayes"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("data/results"))
    p.add_argument("--baseline-root", type=Path, default=Path("data/results"))
    p.add_argument("--out-dir", type=Path, default=Path("data/analysis/plots"))
    return p.parse_args()


def _ci_half(v: np.ndarray, alpha: float = 0.05) -> float:
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return 0.0
    se = float(np.std(v, ddof=1) / np.sqrt(n))
    return float(stats.t.ppf(1.0 - alpha / 2.0, n - 1)) * se


def _plot_track(df: pd.DataFrame, track: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    sub = df[(df["target_gap"] == track) | (df["intervention"] == "unmitigated")].copy()
    if track == "dp":
        sub["gap"] = sub["dp_gap"]
    else:
        sub["gap"] = sub["eo_gap"]

    models = sorted(sub["model"].unique())
    syntheses = ["mst", "pb"]

    fig, axes = plt.subplots(
        len(syntheses), len(models),
        figsize=(6 * len(models), 5.5 * len(syntheses) + 0.8),
        sharex="col", sharey="row",
        squeeze=False,
    )

    seen_legend: set[str] = set()
    for i, synth in enumerate(syntheses):
        for j, model in enumerate(models):
            ax = axes[i, j]
            synth_rows = sub[(sub["model"] == model) & (sub["synthesis"] == synth)]
            clean_rows = sub[(sub["model"] == model) & (sub["synthesis"] == CLEAN_SYNTHESIS)]

            # Per-cell mean + 95% CI cross for synth points; size encodes ε.
            for keys, g in synth_rows.groupby(
                ["intervention", "target_gap", "variant", "eps"], sort=True
            ):
                inter, tg, variant, eps = keys
                color = COLORS.get(inter, "black")
                marker = MARKERS_VARIANT.get(variant, "x")
                size = EPS_SIZES.get(eps, 60)
                acc = g["accuracy"].to_numpy(dtype=float)
                gap = g["gap"].to_numpy(dtype=float)
                acc_mean = float(np.mean(acc))
                gap_mean = float(np.mean(gap))
                acc_ci = _ci_half(acc)
                gap_ci = _ci_half(gap)
                label = (
                    inter if variant == "none"
                    else f"{inter}-{variant}"
                )
                show = label not in seen_legend
                ax.scatter(
                    acc_mean, gap_mean, color=color, marker=marker,
                    s=size, edgecolor="black", linewidth=0.4, alpha=0.85,
                    label=label if show else None, zorder=2,
                )
                seen_legend.add(label)
                if acc_ci > 0 and gap_ci > 0:
                    e = Ellipse(
                        (acc_mean, gap_mean), width=2 * acc_ci, height=2 * gap_ci,
                        facecolor=color, alpha=0.12, edgecolor=color,
                        linewidth=0.5, zorder=1,
                    )
                    ax.add_patch(e)

            # Clean baseline as a single big star per (intervention, variant).
            for keys, g in clean_rows.groupby(
                ["intervention", "target_gap", "variant"], sort=True
            ):
                inter, tg, variant = keys
                color = COLORS.get(inter, "black")
                ax.scatter(
                    g["accuracy"], g["gap"],
                    color=color, marker="*", s=260,
                    edgecolor="black", linewidth=0.7, zorder=3,
                )

            if i == 0:
                ax.set_title(model)
            if j == 0:
                ax.set_ylabel(f"{SYN_LABELS.get(synth, synth)}\n{track.upper()} gap")
            if i == len(syntheses) - 1:
                ax.set_xlabel("Accuracy")
            ax.grid(alpha=0.3)

    handle_map: dict[str, object] = {}
    for ax_row in axes:
        for ax in ax_row:
            for h, lbl in zip(*ax.get_legend_handles_labels()):
                if lbl not in handle_map:
                    handle_map[lbl] = h
    fig.tight_layout()
    if handle_map:
        n = len(handle_map)
        ncol = min(n, 5)
        fig.subplots_adjust(bottom=0.12)
        fig.legend(
            list(handle_map.values()), list(handle_map.keys()),
            loc="lower center", ncol=ncol,
            bbox_to_anchor=(0.5, 0.0),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_all(args.results_root, args.baseline_root, drop_failed=True)
    for track in ("dp", "eo"):
        out = args.out_dir / f"accuracy_gap_scatter_{track}.png"
        _plot_track(df, track, out)
        logging.info("wrote %s", out)
        print(f"\n[{out.name}] caption:\n{CAPTION_TEMPLATE.format(track=track.upper())}\n")


if __name__ == "__main__":
    main()
