from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.io import CLEAN_SYNTHESIS, load_all
from analysis.stats_tests import wilcoxon_two_sided


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("data/results"))
    p.add_argument("--baseline-root", type=Path, default=Path("data/results"))
    p.add_argument(
        "--out-dir", type=Path, default=Path("data/analysis/variant_tests")
    )
    return p.parse_args()


def _target_gap_value(df: pd.DataFrame) -> pd.Series:
    is_dp = df["target_gap"] == "dp"
    return pd.Series(
        np.where(is_dp, df["dp_gap"].to_numpy(), df["eo_gap"].to_numpy()),
        index=df.index,
    )


def _paired_test(
    df: pd.DataFrame,
    group_keys: list[str],
    a_filter: dict[str, str],
    b_filter: dict[str, str],
    pair_keys: list[str],
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    a = df.copy()
    for k, v in a_filter.items():
        a = a[a[k] == v]
    b = df.copy()
    for k, v in b_filter.items():
        b = b[b[k] == v]
    a = a.copy()
    a["target_gap_value"] = _target_gap_value(a)
    b = b.copy()
    b["target_gap_value"] = _target_gap_value(b)

    a_keep = pair_keys + group_keys + ["target_gap_value", "accuracy"]
    b_keep = pair_keys + group_keys + ["target_gap_value", "accuracy"]
    a = a[a_keep].rename(
        columns={"target_gap_value": "target_a", "accuracy": "acc_a"}
    )
    b = b[b_keep].rename(
        columns={"target_gap_value": "target_b", "accuracy": "acc_b"}
    )
    merged = a.merge(b, on=pair_keys + group_keys, how="inner", validate="one_to_one")
    merged["delta_target"] = merged["target_a"] - merged["target_b"]
    merged["delta_acc"] = merged["acc_a"] - merged["acc_b"]

    rows: list[dict] = []
    for keys, g in merged.groupby(group_keys, dropna=False, sort=True):
        median_t, p_t = wilcoxon_two_sided(g["delta_target"].to_numpy())
        median_a, p_a = wilcoxon_two_sided(g["delta_acc"].to_numpy())
        row = dict(zip(group_keys, keys))
        row["n"] = int(len(g))
        row["a"] = label_a
        row["b"] = label_b
        row["delta_target_median"] = median_t
        row["p_target"] = p_t
        row["delta_acc_median"] = median_a
        row["p_acc"] = p_a
        rows.append(row)
    return pd.DataFrame(rows)


def make_expgrad_uniform_vs_stratified(df: pd.DataFrame) -> pd.DataFrame:
    synth = df[(df["synthesis"] != CLEAN_SYNTHESIS) & (df["intervention"] == "expgrad")]
    return _paired_test(
        synth,
        group_keys=["model", "synthesis", "eps", "target_gap"],
        a_filter={"variant": "uniform"},
        b_filter={"variant": "stratified"},
        pair_keys=["synth_seed"],
        label_a="uniform",
        label_b="stratified",
    )


def make_threshold_naive_vs_honest(df: pd.DataFrame) -> pd.DataFrame:
    synth = df[(df["synthesis"] != CLEAN_SYNTHESIS) & (df["intervention"] == "threshold")]
    return _paired_test(
        synth,
        group_keys=["model", "synthesis", "eps", "target_gap"],
        a_filter={"variant": "naive"},
        b_filter={"variant": "honest"},
        pair_keys=["synth_seed"],
        label_a="naive",
        label_b="honest",
    )


def make_mst_vs_pb(df: pd.DataFrame) -> pd.DataFrame:
    synth = df[df["synthesis"].isin(("mst", "pb"))]
    return _paired_test(
        synth,
        group_keys=["model", "eps", "intervention", "target_gap", "variant"],
        a_filter={"synthesis": "mst"},
        b_filter={"synthesis": "pb"},
        pair_keys=["synth_seed"],
        label_a="mst",
        label_b="pb",
    )


def make_logreg_vs_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    synth = df[df["synthesis"] != CLEAN_SYNTHESIS]
    return _paired_test(
        synth,
        group_keys=["synthesis", "eps", "intervention", "target_gap", "variant"],
        a_filter={"model": "logreg"},
        b_filter={"model": "xgboost"},
        pair_keys=["synth_seed"],
        label_a="logreg",
        label_b="xgboost",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_all(args.results_root, args.baseline_root, drop_failed=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "expgrad_uniform_vs_stratified.csv": make_expgrad_uniform_vs_stratified(df),
        "threshold_naive_vs_honest.csv": make_threshold_naive_vs_honest(df),
        "mst_vs_privbayes.csv": make_mst_vs_pb(df),
        "logreg_vs_xgboost.csv": make_logreg_vs_xgboost(df),
    }
    for name, table in artifacts.items():
        path = args.out_dir / name
        table.to_csv(path, index=False)
        logging.info("wrote %s (%d rows)", path, len(table))


if __name__ == "__main__":
    main()
