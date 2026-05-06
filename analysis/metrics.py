from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.io import CLEAN_SYNTHESIS, target_gap_metric

GAPS = ("dp_gap", "eo_gap")
PAIR_KEYS_SYNTH = ["model", "synthesis", "eps", "synth_seed"]


def attach_unmitigated(
    df: pd.DataFrame,
    pair_keys: list[str] | None = None,
) -> pd.DataFrame:
    # For each (model, synthesis, eps, synth_seed) cell, broadcast the
    # unmitigated row's gaps and accuracy back onto every other intervention
    # row in the same group, suffixed with "_unm".
    if pair_keys is None:
        pair_keys = PAIR_KEYS_SYNTH
    unm = (
        df[df["intervention"] == "unmitigated"]
        .rename(columns={"dp_gap": "dp_gap_unm", "eo_gap": "eo_gap_unm",
                         "accuracy": "accuracy_unm", "auc": "auc_unm"})
        [pair_keys + ["dp_gap_unm", "eo_gap_unm", "accuracy_unm", "auc_unm"]]
    )
    inter = df[df["intervention"] != "unmitigated"]
    return inter.merge(unm, on=pair_keys, how="inner", validate="many_to_one")


def add_target_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Add target_gap_metric, target_gap_value, target_gap_unm columns based on
    # each row's `target_gap` field. For target_gap=='none' the columns are NaN.
    df = df.copy()
    is_dp = df["target_gap"] == "dp"
    is_eo = df["target_gap"] == "eo"
    df["target_gap_metric"] = np.where(is_dp, "dp_gap", np.where(is_eo, "eo_gap", ""))
    df["target_gap_value"] = np.where(is_dp, df["dp_gap"], np.where(is_eo, df["eo_gap"], np.nan))
    if "dp_gap_unm" in df.columns and "eo_gap_unm" in df.columns:
        df["target_gap_unm"] = np.where(
            is_dp, df["dp_gap_unm"], np.where(is_eo, df["eo_gap_unm"], np.nan)
        )
    return df


def per_seed_benefits(df: pd.DataFrame) -> pd.DataFrame:
    # Compute per-row dp/eo benefit (= unmitigated_gap - intervention_gap) and
    # the target-gap benefit. Requires attach_unmitigated() to have been run
    # first (so dp_gap_unm / eo_gap_unm exist).
    df = df.copy()
    df["dp_benefit"] = df["dp_gap_unm"] - df["dp_gap"]
    df["eo_benefit"] = df["eo_gap_unm"] - df["eo_gap"]
    is_dp = df["target_gap"] == "dp"
    is_eo = df["target_gap"] == "eo"
    df["target_benefit"] = np.where(
        is_dp, df["dp_benefit"], np.where(is_eo, df["eo_benefit"], np.nan)
    )
    return df


def per_seed_ratios(df: pd.DataFrame) -> pd.DataFrame:
    # Two-track ratio: (synth_target_benefit) / (clean_target_benefit_for_same_cell).
    # Clean benefit is per-(model, intervention, target_gap, variant), averaged
    # across the (single) clean row per cell. Returns long-form with
    # `clean_target_benefit`, `synth_target_benefit`, `target_ratio` plus the
    # off-target dp/eo benefits for side-effect reporting.
    if df.empty:
        return df.assign(
            clean_target_benefit=pd.Series(dtype=float),
            synth_target_benefit=pd.Series(dtype=float),
            target_ratio=pd.Series(dtype=float),
        )
    with_unm = attach_unmitigated(df)
    with_benefits = per_seed_benefits(with_unm)

    cell_keys = ["model", "intervention", "target_gap", "variant"]
    clean_mask = with_benefits["synthesis"] == CLEAN_SYNTHESIS
    clean = (
        with_benefits[clean_mask]
        .groupby(cell_keys, as_index=False)["target_benefit"]
        .mean()
        .rename(columns={"target_benefit": "clean_target_benefit"})
    )
    synth = with_benefits[~clean_mask].rename(
        columns={"target_benefit": "synth_target_benefit"}
    )
    merged = synth.merge(clean, on=cell_keys, how="left", validate="many_to_one")
    merged["target_ratio"] = (
        merged["synth_target_benefit"] / merged["clean_target_benefit"]
    )
    return merged
