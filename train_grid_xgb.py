#!/usr/bin/env python3
"""Step 4b: Main experiment grid — DP synthetic data × fairness interventions — XGBoost.

One Slurm array task = one (epsilon, seed) pair. Runs all four interventions
on the corresponding synthetic training set and evaluates on the fixed clean
test split.

Writes results/grid_xgb/eps{e}_seed{s}.csv (one row per intervention). A
separate merge step combines these into results/grid_xgb.csv.

Columns: eps, synth_seed, intervention, accuracy, auc, dp_gap, eo_gap

Key differences from train_grid.py (LogisticRegression):
  - base_clf() returns XGBClassifier with hist tree method.
  - No dense materialization: XGBoost hist accepts sparse CSR directly.
    expgrad and threshold both receive the sparse matrix.
  - AUC for expgrad is computed from 0/1 predictions (predict_proba not
    available on the ExponentiatedGradient ensemble wrapper).
  - Output path: results/grid_xgb/ — never written to results/grid/.
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_baseline_xgb import (
    CAT_COLS,
    NUM_COLS,
    PROT_ATTR,
    TARGET,
    compute_metrics,
    find_prepared_table,
    make_preprocessor,
    run_expgrad,
    run_reweighting,
    run_threshold,
    run_unmitigated,
    set_seeds,
)


def load_clean_test(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    table_path = find_prepared_table(data_dir)
    df = pd.read_parquet(table_path) if table_path.suffix == ".parquet" else pd.read_pickle(table_path)
    idx_test = np.load(data_dir / "idx_test.npy")
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X.iloc[idx_test], y.iloc[idx_test], df[PROT_ATTR].iloc[idx_test]


def load_synth(synth_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_parquet(synth_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    A = df[PROT_ATTR]
    return X, y, A


def run_condition(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    A_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    A_test: pd.Series,
    interventions: list[str],
    expgrad_subsample: int,
    seed: int,
) -> list[dict[str, Any]]:
    preprocessor = make_preprocessor()
    # XGBoost hist accepts the dense output from make_preprocessor() directly.
    # No .toarray() call needed — sparse_output=False in OHE keeps memory
    # predictable and avoids the format-conversion branch in the LR grid.
    X_train_pre = preprocessor.fit_transform(X_train)  # dense ndarray, fit on synth
    X_test_pre = preprocessor.transform(X_test)        # dense ndarray, clean test

    logging.info(
        "synth train=%d  test=%d  features_after_enc=%d",
        len(y_train),
        len(y_test),
        X_train_pre.shape[1],
    )

    to_run = set(interventions)
    results: list[dict[str, Any]] = []

    if "unmitigated" in to_run:
        logging.info("running: unmitigated ...")
        t0 = time.perf_counter()
        r = run_unmitigated(X_train_pre, y_train, X_test_pre, y_test, A_test)
        logging.info("done: unmitigated (%.1fs)  %s", time.perf_counter() - t0, r)
        results.append(r)

    if "reweighting" in to_run:
        logging.info("running: reweighting ...")
        t0 = time.perf_counter()
        rw_preprocessor = make_preprocessor()
        rw_preprocessor.fit(X_train)
        X_test_pre_rw = rw_preprocessor.transform(X_test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_reweighting(X_train, y_train, A_train, rw_preprocessor, X_test_pre_rw, y_test, A_test)
        logging.info("done: reweighting (%.1fs)  %s", time.perf_counter() - t0, r)
        results.append(r)

    if "expgrad" in to_run:
        logging.info("running: expgrad (max_iter=50) ...")
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_expgrad(
                X_train_pre, y_train, A_train, X_test_pre, y_test, A_test,
                subsample=expgrad_subsample,
                seed=seed,
            )
        logging.info("done: expgrad (%.1fs)  %s", time.perf_counter() - t0, r)
        results.append(r)

    if "threshold" in to_run:
        logging.info("running: threshold ...")
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_threshold(X_train_pre, y_train, A_train, X_test_pre, y_test, A_test)
        logging.info("done: threshold (%.1fs)  %s", time.perf_counter() - t0, r)
        results.append(r)

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 4b: XGBoost experiment grid (one Slurm task = one eps/seed pair)."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory with acs_prepared.*, idx_test.npy.",
    )
    p.add_argument(
        "--synth-dir",
        type=Path,
        default=Path("data/synth"),
        help="Directory with eps{e}_seed{s}.parquet files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/grid_xgb"),
        help="Directory for per-task CSVs.",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="DP epsilon value for this task (e.g. 1, 2, 4, 8).",
    )
    p.add_argument(
        "--synth-seed",
        type=int,
        required=True,
        help="Synthetic data seed for this task (0-indexed).",
    )
    p.add_argument(
        "--interventions",
        nargs="+",
        choices=["unmitigated", "reweighting", "expgrad", "threshold"],
        default=["unmitigated", "reweighting", "expgrad", "threshold"],
    )
    p.add_argument(
        "--expgrad-subsample",
        type=int,
        default=20000,
        metavar="N",
        help="Rows subsampled per expgrad oracle call. 0 = full training set.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for model fitting / expgrad subsample.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        force=True,
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger().handlers[0].stream = open(  # noqa: SIM115
        "/dev/stderr", "w", buffering=1
    )
    set_seeds(args.seed)
    logging.info("config: %s", vars(args))

    eps_int = int(args.epsilon)
    synth_path = args.synth_dir / f"eps{eps_int}_seed{args.synth_seed}.parquet"
    if not synth_path.exists():
        raise FileNotFoundError(f"Synthetic file not found: {synth_path}")

    logging.info("loading synth: %s", synth_path)
    X_train, y_train, A_train = load_synth(synth_path)

    logging.info("loading clean test split from %s", args.data_dir)
    X_test, y_test, A_test = load_clean_test(args.data_dir)

    results = run_condition(
        X_train, y_train, A_train,
        X_test, y_test, A_test,
        interventions=args.interventions,
        expgrad_subsample=args.expgrad_subsample,
        seed=args.seed,
    )

    for r in results:
        r["eps"] = args.epsilon
        r["synth_seed"] = args.synth_seed

    df_out = pd.DataFrame(results)[["eps", "synth_seed", "intervention", "accuracy", "auc", "dp_gap", "eo_gap"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"eps{eps_int}_seed{args.synth_seed}.csv"
    df_out.to_csv(out_path, index=False)
    logging.info("wrote %s", out_path)
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
