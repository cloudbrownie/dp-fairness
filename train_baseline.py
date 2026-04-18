#!/usr/bin/env python3
"""Step 3: Clean baseline experiments on unperturbed ACS data.

Four conditions, all evaluated on the fixed held-out test split:
  - unmitigated: plain LogisticRegression
  - reweighting: aif360 Reweighing (pre-processing)
  - expgrad: fairlearn ExponentiatedGradient (in-processing)
  - threshold: fairlearn ThresholdOptimizer (post-processing)

Writes results/baseline.csv with columns:
  intervention, accuracy, auc, dp_gap, eo_gap
"""

from __future__ import annotations

import argparse
import logging
import random
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from aif360.sklearn.preprocessing import Reweighing
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_COLS = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]
NUM_COLS = ["AGEP", "WKHP"]
TARGET = "PINCP"
PROT_ATTR = "RAC1P"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_prepared_table(data_dir: Path) -> Path:
    for name in ("acs_prepared.parquet", "acs_prepared.pkl"):
        p = data_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No acs_prepared.parquet or acs_prepared.pkl under {data_dir}")


def load_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    table_path = find_prepared_table(data_dir)
    if table_path.suffix == ".parquet":
        df = pd.read_parquet(table_path)
    else:
        df = pd.read_pickle(table_path)

    idx_train = np.load(data_dir / "idx_train.npy")
    idx_test = np.load(data_dir / "idx_test.npy")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y, df[PROT_ATTR], idx_train, idx_test


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ],
        remainder="drop",
    )


def base_clf() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=0)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    sensitive: np.ndarray | pd.Series,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "dp_gap": float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)),
        "eo_gap": float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)),
    }


def run_unmitigated(
    X_train_pre: np.ndarray,
    y_train: pd.Series,
    X_test_pre: np.ndarray,
    y_test: pd.Series,
    A_test: pd.Series,
) -> dict[str, Any]:
    clf = base_clf()
    clf.fit(X_train_pre, y_train)
    y_pred = clf.predict(X_test_pre)
    y_score = clf.predict_proba(X_test_pre)[:, 1]
    return {"intervention": "unmitigated", **compute_metrics(y_test, y_pred, y_score, A_test)}


def run_reweighting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    A_train: pd.Series,
    preprocessor: ColumnTransformer,
    X_test_pre: np.ndarray,
    y_test: pd.Series,
    A_test: pd.Series,
) -> dict[str, Any]:
    # aif360 0.6.1 Reweighing.fit_transform(X, y) -> (X, sample_weight).
    # prot_attr must be present as a named index level, not a column.
    X_rw = X_train.reset_index(drop=True).copy()
    A_aligned = A_train.reset_index(drop=True).rename(PROT_ATTR)
    X_rw = X_rw.set_index(A_aligned)

    y_rw = y_train.reset_index(drop=True)
    rw = Reweighing(prot_attr=PROT_ATTR)
    X_rw_out, sample_weight = rw.fit_transform(X_rw, y_rw)
    X_rw_out = X_rw_out.reset_index(drop=True)

    X_rw_pre = preprocessor.fit_transform(X_rw_out)

    clf = base_clf()
    clf.fit(X_rw_pre, y_rw, sample_weight=sample_weight)
    y_pred = clf.predict(X_test_pre)
    y_score = clf.predict_proba(X_test_pre)[:, 1]
    return {"intervention": "reweighting", **compute_metrics(y_test, y_pred, y_score, A_test)}


def run_expgrad(
    X_train_pre: np.ndarray,
    y_train: pd.Series,
    A_train: pd.Series,
    X_test_pre: np.ndarray,
    y_test: pd.Series,
    A_test: pd.Series,
    subsample: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    if subsample and subsample < len(y_train):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_train), size=subsample, replace=False)
        X_fit = X_train_pre[idx]
        y_fit = y_train.iloc[idx]
        A_fit = A_train.iloc[idx]
        logging.info("expgrad oracle subsample: %d / %d rows", subsample, len(y_train))
    else:
        X_fit, y_fit, A_fit = X_train_pre, y_train, A_train

    constraint = EqualizedOdds()
    eg = ExponentiatedGradient(
        base_clf(),
        constraint,
        eps=0.01,
        max_iter=50,
    )
    eg.fit(X_fit, y_fit, sensitive_features=A_fit)
    logging.info("expgrad oracle calls: %d", eg.n_oracle_calls_)
    y_pred = eg.predict(X_test_pre)
    # ExponentiatedGradient does not expose predict_proba; use 0/1 as score.
    y_score = y_pred.astype(float)
    return {"intervention": "expgrad", **compute_metrics(y_test, y_pred, y_score, A_test)}


def run_threshold(
    X_train_pre: np.ndarray,
    y_train: pd.Series,
    A_train: pd.Series,
    X_test_pre: np.ndarray,
    y_test: pd.Series,
    A_test: pd.Series,
) -> dict[str, Any]:
    # Fit the base estimator first, then wrap.
    clf = base_clf()
    clf.fit(X_train_pre, y_train)

    to = ThresholdOptimizer(
        estimator=clf,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    to.fit(X_train_pre, y_train, sensitive_features=A_train)
    y_pred = to.predict(X_test_pre, sensitive_features=A_test)
    y_score = clf.predict_proba(X_test_pre)[:, 1]
    return {"intervention": "threshold", **compute_metrics(y_test, y_pred, y_score, A_test)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean baseline experiments (Step 3).")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory with acs_prepared.*, idx_train.npy, idx_test.npy.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for baseline.csv.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument(
        "--interventions",
        nargs="+",
        choices=["unmitigated", "reweighting", "expgrad", "threshold"],
        default=["unmitigated", "reweighting", "expgrad", "threshold"],
        help="Subset of interventions to run.",
    )
    p.add_argument(
        "--expgrad-subsample",
        type=int,
        default=20000,
        metavar="N",
        help="Rows subsampled per expgrad oracle call. 0 = full training set.",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="baseline.csv",
        help="Output filename under --output-dir (default: baseline.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    set_seeds(args.seed)
    logging.info("config: %s", vars(args))

    X, y, A_race, idx_train, idx_test = load_data(args.data_dir)

    X_train = X.iloc[idx_train]
    y_train = y.iloc[idx_train]
    A_train = A_race.iloc[idx_train]

    X_test = X.iloc[idx_test]
    y_test = y.iloc[idx_test]
    A_test = A_race.iloc[idx_test]

    preprocessor = make_preprocessor()
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    logging.info(
        "train=%d  test=%d  features_after_enc=%d",
        len(idx_train),
        len(idx_test),
        X_train_pre.shape[1],
    )

    results: list[dict[str, Any]] = []
    to_run = set(args.interventions)

    if "unmitigated" in to_run:
        logging.info("running: unmitigated")
        r = run_unmitigated(X_train_pre, y_train, X_test_pre, y_test, A_test)
        results.append(r)
        logging.info("  %s", r)

    if "reweighting" in to_run:
        logging.info("running: reweighting")
        # Reweighting refits preprocessor internally; pass a fresh clone.
        rw_preprocessor = make_preprocessor()
        rw_preprocessor.fit(X_train)  # fit once so test transform is valid
        X_test_pre_rw = rw_preprocessor.transform(X_test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_reweighting(X_train, y_train, A_train, rw_preprocessor, X_test_pre_rw, y_test, A_test)
        results.append(r)
        logging.info("  %s", r)

    if "expgrad" in to_run:
        logging.info("running: expgrad (max_iter=50) ...")
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_expgrad(
                X_train_pre, y_train, A_train, X_test_pre, y_test, A_test,
                subsample=args.expgrad_subsample,
                seed=args.seed,
            )
        logging.info("done: expgrad (%.1fs)  %s", time.perf_counter() - t0, r)
        results.append(r)

    if "threshold" in to_run:
        logging.info("running: threshold")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_threshold(X_train_pre, y_train, A_train, X_test_pre, y_test, A_test)
        results.append(r)
        logging.info("  %s", r)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_name
    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False)
    logging.info("wrote %s", out_path)
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
