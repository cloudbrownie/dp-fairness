#!/usr/bin/env python3
"""Intervention/unmitigated fairness gap ratios across (model, ε, seed)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MODELS = {
    "logreg": ("baseline.csv", "grid"),
    "xgboost": ("baseline_xgb.csv", "grid_xgb"),
}
GAPS = ("dp_gap", "eo_gap")
SEED_SENTINEL = -1.0  # for merging clean baseline (synth_seed is NaN)


def load_all(results_dir: Path, baseline_dir: Path = Path(".")) -> pd.DataFrame:
    rows = []
    for model, (base_name, grid_dir) in MODELS.items():
        base = pd.read_csv(baseline_dir / base_name)
        base["eps"] = np.inf
        base["synth_seed"] = SEED_SENTINEL
        base["model"] = model
        rows.append(base)
        for p in sorted((results_dir / grid_dir).glob("eps*_seed*.csv")):
            df = pd.read_csv(p)
            df["model"] = model
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    return out[["model", "eps", "synth_seed", "intervention", "accuracy", "auc", *GAPS]]


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Per-seed benefit preservation ratio: synth_benefit / clean_benefit.

    For each (model, intervention ≠ unmitigated, synth_seed):
        ratio = (unmitigated_synth_gap - intervention_synth_gap)
              / (unmitigated_clean_gap - intervention_clean_gap)
    Interpreted as the fraction of the intervention's clean-data fairness benefit
    that survives DP synthesis.
      ratio = 1  → DP fully preserved the benefit.
      ratio = 0  → DP fully erased the benefit.
      ratio < 0  → intervention backfires (synth gap worse than unmitigated).
      ratio > 1  → DP amplified the benefit (unusual).
    """
    unm = (
        df.query("intervention == 'unmitigated'")
        .rename(columns={g: f"{g}_unm" for g in GAPS})
        [["model", "eps", "synth_seed", *(f"{g}_unm" for g in GAPS)]]
    )
    inter = df.query("intervention != 'unmitigated'")
    paired = inter.merge(
        unm, on=["model", "eps", "synth_seed"], how="inner", validate="many_to_one"
    )

    clean = paired[~np.isfinite(paired["eps"])].copy()
    for g in GAPS:
        clean[f"{g}_clean_benefit"] = clean[f"{g}_unm"] - clean[g]
    clean_benefit = clean[["model", "intervention", *(f"{g}_clean_benefit" for g in GAPS)]]

    synth = paired[np.isfinite(paired["eps"])].copy()
    for g in GAPS:
        synth[f"{g}_synth_benefit"] = synth[f"{g}_unm"] - synth[g]

    merged = synth.merge(clean_benefit, on=["model", "intervention"], how="left", validate="many_to_one")
    for g in GAPS:
        key = g.split("_")[0]
        merged[f"{key}_ratio"] = merged[f"{g}_synth_benefit"] / merged[f"{g}_clean_benefit"]
    return merged


def _mean_ci(v: np.ndarray) -> tuple[float, float, float]:
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(v))
    if n < 2:
        return mean, mean, mean
    se = float(np.std(v, ddof=1) / np.sqrt(n))
    h = float(stats.t.ppf(0.975, n - 1)) * se
    return mean, mean - h, mean + h


def _agg_factory(cols: tuple[str, ...]):
    def _agg(g: pd.DataFrame) -> pd.Series:
        out = {}
        for col in cols:
            mean, lo, hi = _mean_ci(g[col].to_numpy())
            out[f"{col}_mean"] = mean
            out[f"{col}_ci_lo"] = lo
            out[f"{col}_ci_hi"] = hi
            out[f"{col}_n"] = int(np.sum(np.isfinite(g[col].to_numpy())))
        return pd.Series(out)

    return _agg


def summarize(ratios: pd.DataFrame) -> pd.DataFrame:
    return (
        ratios.groupby(["model", "eps", "intervention"], dropna=False, sort=True)
        .apply(_agg_factory(("dp_ratio", "eo_ratio")))
        .reset_index()
    )


def summarize_raw(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "eps", "intervention"], dropna=False, sort=True)
        .apply(_agg_factory(("dp_gap", "eo_gap")))
        .reset_index()
    )


def risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, eps, intervention): backfire rate and worst-case gap across seeds.

    backfire_rate = fraction of seeds where intervention gap > unmitigated gap.
    worst_gap = max across seeds; p90_gap = 90th percentile (tail-risk view).
    """
    unm = (
        df.query("intervention == 'unmitigated'")
        .rename(columns={"dp_gap": "dp_unm", "eo_gap": "eo_unm"})
        [["model", "eps", "synth_seed", "dp_unm", "eo_unm"]]
    )
    inter = df.query("intervention != 'unmitigated'")
    paired = inter.merge(unm, on=["model", "eps", "synth_seed"], how="inner")

    rows = []
    for (model, eps, inter_name), g in paired.groupby(
        ["model", "eps", "intervention"], dropna=False, sort=True
    ):
        row = {"model": model, "eps": eps, "intervention": inter_name, "n": int(len(g))}
        for metric in ("dp", "eo"):
            gap = g[f"{metric}_gap"].to_numpy()
            unm_gap = g[f"{metric}_unm"].to_numpy()
            row[f"{metric}_worst_gap"] = float(np.max(gap))
            row[f"{metric}_p90_gap"] = float(np.quantile(gap, 0.9))
            row[f"{metric}_backfire_rate"] = float(np.mean(gap > unm_gap))
        rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired signed-rank test per (model, eps, intervention) across seeds.

    Two-sided: H0 = intervention metric equals unmitigated metric. Sign of median
    delta indicates direction (negative delta on gap = intervention helps).
    Includes accuracy delta so the accuracy-cost test lives in the same table.
    Skips the test (p=NaN) when n<2 or all differences are zero.
    """
    unm = (
        df.query("intervention == 'unmitigated'")
        .rename(columns={"dp_gap": "dp_unm", "eo_gap": "eo_unm", "accuracy": "acc_unm"})
        [["model", "eps", "synth_seed", "dp_unm", "eo_unm", "acc_unm"]]
    )
    inter = df.query("intervention != 'unmitigated'")
    paired = inter.merge(unm, on=["model", "eps", "synth_seed"], how="inner")
    paired["delta_dp"] = paired["dp_gap"] - paired["dp_unm"]
    paired["delta_eo"] = paired["eo_gap"] - paired["eo_unm"]
    paired["delta_acc"] = paired["accuracy"] - paired["acc_unm"]

    rows = []
    for (model, eps, inter_name), g in paired.groupby(
        ["model", "eps", "intervention"], dropna=False, sort=True
    ):
        row = {"model": model, "eps": eps, "intervention": inter_name, "n": int(len(g))}
        for metric in ("dp", "eo", "acc"):
            d = g[f"delta_{metric}"].to_numpy()
            row[f"delta_{metric}_median"] = float(np.median(d))
            if len(d) >= 2 and not np.allclose(d, 0):
                try:
                    stat = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
                    row[f"wilcoxon_{metric}_p"] = float(stat.pvalue)
                except ValueError:
                    row[f"wilcoxon_{metric}_p"] = float("nan")
            else:
                row[f"wilcoxon_{metric}_p"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def attenuation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """How much of clean intervention benefit survives DP?

    clean_benefit = (unmitigated_gap - intervention_gap) at ε=∞.
    synth_benefit = mean over seeds of the same quantity at each synth ε.
    attenuation = 1 - synth_benefit / clean_benefit.
      0 = DP preserved the benefit; 1 = DP erased it; >1 = intervention now hurts.
    """
    unm = (
        df.query("intervention == 'unmitigated'")
        .rename(columns={"dp_gap": "dp_unm", "eo_gap": "eo_unm"})
        [["model", "eps", "synth_seed", "dp_unm", "eo_unm"]]
    )
    inter = df.query("intervention != 'unmitigated'")
    paired = inter.merge(unm, on=["model", "eps", "synth_seed"], how="inner")
    paired["dp_benefit"] = paired["dp_unm"] - paired["dp_gap"]
    paired["eo_benefit"] = paired["eo_unm"] - paired["eo_gap"]

    clean = (
        paired[~np.isfinite(paired["eps"])]
        .groupby(["model", "intervention"], as_index=False)[["dp_benefit", "eo_benefit"]]
        .mean()
        .rename(columns={"dp_benefit": "dp_clean_benefit", "eo_benefit": "eo_clean_benefit"})
    )
    synth = (
        paired[np.isfinite(paired["eps"])]
        .groupby(["model", "eps", "intervention"], as_index=False)[["dp_benefit", "eo_benefit"]]
        .mean()
        .rename(columns={"dp_benefit": "dp_synth_benefit", "eo_benefit": "eo_synth_benefit"})
    )
    out = synth.merge(clean, on=["model", "intervention"], how="left")
    out["dp_attenuation"] = 1 - out["dp_synth_benefit"] / out["dp_clean_benefit"]
    out["eo_attenuation"] = 1 - out["eo_synth_benefit"] / out["eo_clean_benefit"]
    return out[[
        "model", "eps", "intervention",
        "dp_clean_benefit", "dp_synth_benefit", "dp_attenuation",
        "eo_clean_benefit", "eo_synth_benefit", "eo_attenuation",
    ]]


def variance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Std across seeds per (model, eps, intervention); ratio vs. unmitigated at same (model, eps)."""
    grouped = (
        df.groupby(["model", "eps", "intervention"], dropna=False, as_index=False)
        .agg(
            n=("dp_gap", "count"),
            dp_std=("dp_gap", "std"),
            eo_std=("eo_gap", "std"),
            acc_std=("accuracy", "std"),
        )
    )
    unm = grouped[grouped["intervention"] == "unmitigated"][
        ["model", "eps", "dp_std", "eo_std", "acc_std"]
    ].rename(columns={"dp_std": "dp_std_unm", "eo_std": "eo_std_unm", "acc_std": "acc_std_unm"})
    out = grouped.merge(unm, on=["model", "eps"], how="left")
    for m in ("dp", "eo", "acc"):
        out[f"{m}_std_ratio"] = out[f"{m}_std"] / out[f"{m}_std_unm"]
    return out


RATIO_METRICS = [("dp_ratio", "DP gap ratio"), ("eo_ratio", "EO gap ratio")]
RAW_METRICS = [("dp_gap", "DP gap"), ("eo_gap", "EO gap")]
_COLORS = {
    "unmitigated": "tab:red",
    "reweighting": "tab:blue",
    "expgrad": "tab:orange",
    "threshold": "tab:green",
}
_EPS_ORDER = [1.0, 2.0, 4.0, 8.0, np.inf]
_X_LABELS = ["1", "2", "4", "8", "clean"]


def _render_grid(
    summary: pd.DataFrame, metrics, out_path: Path, suptitle: str,
    include_clean: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    models = sorted(summary["model"].unique())
    colors = _COLORS
    if include_clean:
        eps_order = _EPS_ORDER
        x_labels = _X_LABELS
    else:
        eps_order = _EPS_ORDER[:-1]
        x_labels = _X_LABELS[:-1]
    x_pos = list(range(len(eps_order)))

    fig, axes = plt.subplots(
        len(metrics), len(models), figsize=(5.5 * len(models), 4 * len(metrics)), sharey="row"
    )

    for i, (metric, ylabel) in enumerate(metrics):
        for j, model in enumerate(models):
            ax = axes[i, j]
            sub = summary[summary["model"] == model]
            for inter in sorted(sub["intervention"].unique()):
                c = colors.get(inter, "black")
                means, los, his = [], [], []
                for eps in eps_order:
                    mask = (sub["intervention"] == inter) & (
                        ~np.isfinite(sub["eps"]) if not np.isfinite(eps) else sub["eps"] == eps
                    )
                    r = sub[mask]
                    if r.empty:
                        means.append(np.nan); los.append(np.nan); his.append(np.nan)
                    else:
                        row = r.iloc[0]
                        means.append(row[f"{metric}_mean"])
                        los.append(row[f"{metric}_ci_lo"])
                        his.append(row[f"{metric}_ci_hi"])
                n_synth = 4
                ax.plot(x_pos[:n_synth], means[:n_synth], marker="o", color=c, label=inter)
                ax.fill_between(x_pos[:n_synth], los[:n_synth], his[:n_synth], alpha=0.2, color=c)
                if include_clean:
                    ax.scatter(
                        x_pos[n_synth], means[n_synth], marker="*", s=220, color=c,
                        edgecolor="black", linewidth=0.6, zorder=3,
                    )

            if metric.endswith("_ratio"):
                ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
                ax.axhline(0.0, color="k", linestyle=":", alpha=0.6, linewidth=0.8)
            if include_clean:
                ax.axvline(3.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("ε (privacy budget)")
            if j == 0:
                ax.set_ylabel(f"{ylabel} (mean ± 95% CI)")
            ax.set_title(f"{model}")
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(suptitle, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_plot(summary: pd.DataFrame, out_path: Path) -> None:
    _render_grid(
        summary, RATIO_METRICS, out_path,
        "Benefit preservation ratio: synth_benefit / clean_benefit "
        "(1 = benefit preserved; 0 = erased; <0 = backfire)",
        include_clean=False,
    )


def make_raw_plot(raw_summary: pd.DataFrame, out_path: Path) -> None:
    _render_grid(
        raw_summary, RAW_METRICS, out_path,
        "Absolute DP and EO gaps across ε, by intervention (★ = clean baseline)",
    )


def make_scatter_plot(df: pd.DataFrame, out_path: Path) -> None:
    """Accuracy vs. {DP, EO} gap; rows = gap type, cols = model, colored by intervention."""
    import matplotlib.pyplot as plt

    models = sorted(df["model"].unique())
    gaps = [("dp_gap", "DP gap"), ("eo_gap", "EO gap")]
    fig, axes = plt.subplots(
        len(gaps), len(models), figsize=(6 * len(models), 4.5 * len(gaps)), sharex="col"
    )

    for i, (gap_col, gap_label) in enumerate(gaps):
        for j, model in enumerate(models):
            ax = axes[i, j]
            sub = df[df["model"] == model]
            for inter in sorted(sub["intervention"].unique()):
                s = sub[sub["intervention"] == inter]
                synth = s[np.isfinite(s["eps"])]
                clean = s[~np.isfinite(s["eps"])]
                c = _COLORS.get(inter, "black")
                ax.scatter(synth["accuracy"], synth[gap_col], color=c, alpha=0.6, s=40, label=inter)
                ax.scatter(
                    clean["accuracy"], clean[gap_col], color=c, marker="*",
                    s=300, edgecolor="black", linewidth=0.8, zorder=3,
                )
            ax.set_ylabel(gap_label)
            if i == len(gaps) - 1:
                ax.set_xlabel("Accuracy")
            if i == 0:
                ax.set_title(model)
            ax.legend(fontsize=8, loc="best")
            ax.grid(alpha=0.3)

    fig.suptitle("Accuracy vs. DP and EO gap (dots = synth, ★ = clean baseline)", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("data/results"))
    p.add_argument("--baseline-dir", type=Path, default=Path("."))
    p.add_argument("--summary-out", type=Path, default=Path("data/results/summary.csv"))
    p.add_argument("--long-out", type=Path, default=Path("data/results/ratios_long.csv"))
    p.add_argument("--plot-out", type=Path, default=Path("data/results/ratios_plot.png"))
    p.add_argument("--raw-summary-out", type=Path, default=Path("data/results/raw_summary.csv"))
    p.add_argument("--raw-plot-out", type=Path, default=Path("data/results/raw_plot.png"))
    p.add_argument("--wilcoxon-out", type=Path, default=Path("data/results/wilcoxon.csv"))
    p.add_argument("--risk-out", type=Path, default=Path("data/results/risk_summary.csv"))
    p.add_argument("--attenuation-out", type=Path, default=Path("data/results/attenuation.csv"))
    p.add_argument("--variance-out", type=Path, default=Path("data/results/variance.csv"))
    p.add_argument("--scatter-out", type=Path, default=Path("data/results/accuracy_eo_scatter.png"))
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_all(args.results_dir, args.baseline_dir)
    logging.info("loaded %d rows (%s)", len(df), ", ".join(sorted(df["model"].unique())))

    ratios = compute_ratios(df)
    ratios.to_csv(args.long_out, index=False)
    logging.info("wrote %s (%d rows)", args.long_out, len(ratios))

    summary = summarize(ratios)
    summary.to_csv(args.summary_out, index=False)
    logging.info("wrote %s (%d rows)", args.summary_out, len(summary))

    raw_summary = summarize_raw(df)
    raw_summary.to_csv(args.raw_summary_out, index=False)
    logging.info("wrote %s (%d rows)", args.raw_summary_out, len(raw_summary))

    wilcoxon = wilcoxon_tests(df)
    wilcoxon.to_csv(args.wilcoxon_out, index=False)
    logging.info("wrote %s (%d rows)", args.wilcoxon_out, len(wilcoxon))

    risk = risk_summary(df)
    risk.to_csv(args.risk_out, index=False)
    logging.info("wrote %s (%d rows)", args.risk_out, len(risk))

    atten = attenuation_analysis(df)
    atten.to_csv(args.attenuation_out, index=False)
    logging.info("wrote %s (%d rows)", args.attenuation_out, len(atten))

    var = variance_summary(df)
    var.to_csv(args.variance_out, index=False)
    logging.info("wrote %s (%d rows)", args.variance_out, len(var))

    pretty = summary.copy()
    eps_display = pretty["eps"].map(lambda v: "inf" if not np.isfinite(v) else f"{v:g}")
    pretty.insert(1, "eps_", eps_display)
    pretty = pretty.drop(columns=["eps"]).rename(columns={"eps_": "eps"})
    for col in pretty.columns:
        if pretty[col].dtype.kind == "f":
            pretty[col] = pretty[col].round(3)
    print(pretty.to_string(index=False))

    if not args.no_plot:
        make_plot(summary, args.plot_out)
        logging.info("wrote %s", args.plot_out)
        make_raw_plot(raw_summary, args.raw_plot_out)
        logging.info("wrote %s", args.raw_plot_out)
        make_scatter_plot(df, args.scatter_out)
        logging.info("wrote %s", args.scatter_out)


if __name__ == "__main__":
    main()
