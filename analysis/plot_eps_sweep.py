from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import style as _style  # noqa: F401
from analysis.palette import PALETTE


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
# Group by intervention family along the palette axis. Reweighing (most
# reliable) sits at the teal end; expgrad and threshold get progressively
# warmer slots so the visual order reflects "good→bad" reliability seen in
# the data. Within a family, variants are distinguished by linestyle.
COLORS = {
    "unmitigated": "#7f7f7f",
    "reweighing": PALETTE[0],
    "expgrad-dp-uniform": PALETTE[1],
    "expgrad-dp-stratified": PALETTE[1],
    "expgrad-eo-uniform": PALETTE[2],
    "expgrad-eo-stratified": PALETTE[2],
    "threshold-dp-naive": PALETTE[3],
    "threshold-dp-honest": PALETTE[3],
    "threshold-eo-naive": PALETTE[4],
    "threshold-eo-honest": PALETTE[4],
}
LINESTYLES = {
    "unmitigated": "-",
    "reweighing": "-",
    "expgrad-dp-uniform": ":",
    "expgrad-dp-stratified": "-",
    "expgrad-eo-uniform": ":",
    "expgrad-eo-stratified": "-",
    "threshold-dp-naive": ":",
    "threshold-dp-honest": "-",
    "threshold-eo-naive": ":",
    "threshold-eo-honest": "-",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-summary",
        type=Path,
        default=Path("data/analysis/raw_summary/raw_summary_target.csv"),
    )
    p.add_argument(
        "--ratios-summary",
        type=Path,
        default=Path("data/analysis/ratios/ratios_summary.csv"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/analysis/plots"))
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


def _plot_eps_sweep(
    df: pd.DataFrame,
    metric_mean: str,
    metric_lo: str,
    metric_hi: str,
    title: str,
    out_path: Path,
    include_clean: bool,
    hline: tuple[float, ...] = (),
    track: str | None = None,
) -> None:
    # track in {"dp", "eo", None}. When set, only cells whose target_gap
    # matches the track are drawn (plus unmitigated as a thin gray reference
    # since it has no target).
    import matplotlib.pyplot as plt

    df = _add_cell(df)
    if track is not None:
        df = df[(df["target_gap"] == track) | (df["intervention"] == "unmitigated")]
    models = sorted(df["model"].unique())
    syntheses = ["mst", "pb"]
    syn_labels = {"mst": "MST", "pb": "PrivBayes"}

    if include_clean:
        eps_axis = [1.0, 2.0, 4.0, 8.0, np.inf]
        x_labels = ["1", "2", "4", "8", "clean"]
    else:
        eps_axis = [1.0, 2.0, 4.0, 8.0]
        x_labels = ["1", "2", "4", "8"]
    x_pos = list(range(len(eps_axis)))

    fig, axes = plt.subplots(
        len(syntheses), len(models),
        figsize=(6 * len(models), 4.2 * len(syntheses) + 0.8),
        sharey="row",
    )
    if len(syntheses) == 1:
        axes = np.array([axes])
    if len(models) == 1:
        axes = axes.reshape(-1, 1)

    for i, synth in enumerate(syntheses):
        for j, model in enumerate(models):
            ax = axes[i, j]
            sub = df[(df["model"] == model) & (df["synthesis"].isin([synth, "clean"]))]
            for cell in CELL_ORDER:
                if cell not in sub["cell"].unique():
                    continue
                color = COLORS.get(cell, "black")
                means, los, his = [], [], []
                for eps in eps_axis:
                    if np.isinf(eps):
                        rs = sub[(sub["cell"] == cell) & (sub["synthesis"] == "clean")]
                    else:
                        rs = sub[(sub["cell"] == cell) & (sub["eps"] == eps) & (sub["synthesis"] == synth)]
                    if rs.empty:
                        means.append(np.nan); los.append(np.nan); his.append(np.nan)
                    else:
                        means.append(float(rs.iloc[0][metric_mean]))
                        los.append(float(rs.iloc[0][metric_lo]))
                        his.append(float(rs.iloc[0][metric_hi]))
                n_synth = 4
                ls = LINESTYLES.get(cell, "-")
                is_unm = cell == "unmitigated"
                ax.plot(
                    x_pos[:n_synth], means[:n_synth],
                    marker="o" if not is_unm else None,
                    color=color, label=cell,
                    linewidth=0.9 if is_unm else 1.4,
                    linestyle="--" if is_unm else ls,
                    alpha=0.6 if is_unm else 1.0,
                )
                if not is_unm:
                    ax.fill_between(
                        x_pos[:n_synth], los[:n_synth], his[:n_synth],
                        color=color, alpha=0.15,
                    )
                if include_clean and not np.isnan(means[n_synth]):
                    ax.scatter(
                        x_pos[n_synth], means[n_synth], marker="*", s=180, color=color,
                        edgecolor="black", linewidth=0.5, zorder=3,
                    )

            for h in hline:
                ax.axhline(h, color="k", linestyle="--", alpha=0.4, linewidth=0.7)
            if include_clean:
                ax.axvline(3.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("ε")
            if j == 0:
                ax.set_ylabel(syn_labels.get(synth, synth))
            ax.set_title(f"{model}")
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
        # Reserve bottom space proportional to legend row count so axis labels
        # never collide with the legend.
        n_rows = (n + ncol - 1) // ncol
        fig.subplots_adjust(bottom=0.06 + 0.05 * n_rows)
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

    raw = pd.read_csv(args.raw_summary)
    for track in ("dp", "eo"):
        out = args.out_dir / f"eps_sweep_target_gap_{track}.png"
        _plot_eps_sweep(
            raw,
            metric_mean="target_gap_mean",
            metric_lo="target_gap_ci_lo",
            metric_hi="target_gap_ci_hi",
            title="",
            out_path=out,
            include_clean=True,
            track=track,
        )
        logging.info("wrote %s", out)
        cap = (
            f"{track.upper()}-target gap vs privacy budget ε for each "
            f"intervention cell (mean ± 95% CI across 10 seeds). Rows = "
            f"synthesizer (MST, PrivBayes); cols = model. ★ marks the clean "
            f"(non-DP) baseline; dashed gray line is unmitigated for reference. "
            f"Cells with target_gap={track.upper()} only."
        )
        print(f"\n[{out.name}] caption:\n{cap}\n")

    ratios = pd.read_csv(args.ratios_summary)
    for track in ("dp", "eo"):
        out = args.out_dir / f"eps_sweep_ratio_{track}.png"
        _plot_eps_sweep(
            ratios,
            metric_mean="target_ratio_mean",
            metric_lo="target_ratio_ci_lo",
            metric_hi="target_ratio_ci_hi",
            title="",
            out_path=out,
            include_clean=False,
            hline=(0.0, 1.0),
            track=track,
        )
        logging.info("wrote %s", out)
        cap = (
            f"{track.upper()} benefit-preservation ratio vs privacy budget ε "
            f"(synth target benefit / clean target benefit, mean ± 95% CI "
            f"across 10 seeds). 1 = benefit fully preserved; 0 = fully erased; "
            f"<0 = intervention backfires under DP."
        )
        print(f"\n[{out.name}] caption:\n{cap}\n")

    # Old all-cells plot kept as legacy artifact, no longer the headline.
    _plot_eps_sweep(
        ratios,
        metric_mean="target_ratio_mean",
        metric_lo="target_ratio_ci_lo",
        metric_hi="target_ratio_ci_hi",
        title="",
        out_path=args.out_dir / "eps_sweep_ratio.png",
        include_clean=False,
        hline=(0.0, 1.0),
    )
    logging.info("wrote %s", args.out_dir / "eps_sweep_ratio.png")


if __name__ == "__main__":
    main()
