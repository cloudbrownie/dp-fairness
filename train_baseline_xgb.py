#!/usr/bin/env python3
"""Clean baseline experiments on unperturbed ACS data (XGBoost).

Four conditions, all evaluated on the fixed held-out test split:
  - unmitigated: plain XGBClassifier
  - reweighing: aif360 Reweighing (pre-processing)
  - expgrad: fairlearn ExponentiatedGradient (in-processing, stratified oracle)
  - threshold: fairlearn ThresholdOptimizer (post-processing)

The expgrad oracle uses a STRATIFIED subsample of the training data. See
train_baseline.py for the rationale; using stratified on XGB as well
keeps the LR/XGB comparison apples-to-apples (same sampling scheme
across models).

Writes data/results/baseline_xgb.csv with columns:
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
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from stratified import per_group_counts, stratified_subsample_idx

CAT_COLS = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]
NUM_COLS = ["AGEP", "WKHP"]
TARGET = "PINCP"
PROT_ATTR = "RAC1P"

CELL_CHOICES = [
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


def base_clf() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=0,
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
    )


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


def run_reweighing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    A_train: pd.Series,
    preprocessor: ColumnTransformer,
    X_test_pre: np.ndarray,
    y_test: pd.Series,
    A_test: pd.Series,
) -> dict[str, Any]:
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
    return {"intervention": "reweighing", **compute_metrics(y_test, y_pred, y_score, A_test)}


def run_expgrad(
    X_train_pre: np.ndarray,
    y_train: pd.Series,
    A_train: pd.Series,
    X_test_pre: np.ndarray,
    y_test: pd.Series,
    A_test: pd.Series,
    subsample: int,
    min_per_group: int,
    seed: int,
    *,
    target_gap: str = "eo",
    sampler: str = "stratified",
) -> dict[str, Any]:
    if subsample and subsample < len(y_train):
        rng = np.random.default_rng(seed)
        if sampler == "stratified":
            idx = stratified_subsample_idx(
                A_train, target_size=subsample, min_per_group=min_per_group, rng=rng
            )
        elif sampler == "uniform":
            idx = rng.choice(len(y_train), size=subsample, replace=False)
        else:
            raise ValueError(f"sampler must be 'stratified' or 'uniform'; got {sampler!r}")
        X_fit = X_train_pre[idx]
        y_fit = y_train.iloc[idx]
        A_fit = A_train.iloc[idx]
        counts = per_group_counts(A_train, idx)
        logging.info(
            "expgrad %s subsample: %d rows across %d groups (min_per_group=%d)",
            sampler, len(idx), len(counts), min_per_group,
        )
        logging.info("expgrad per-group counts: %s", counts)
    else:
        X_fit, y_fit, A_fit = X_train_pre, y_train, A_train
        logging.info("expgrad subsample disabled or >= train size: using full train (%d rows)", len(y_train))

    if target_gap == "eo":
        constraint: Any = EqualizedOdds()
    elif target_gap == "dp":
        constraint = DemographicParity()
    else:
        raise ValueError(f"target_gap must be 'eo' or 'dp'; got {target_gap!r}")
    logging.info(
        "expgrad target_gap=%s constraint=%s sampler=%s",
        target_gap, type(constraint).__name__, sampler,
    )
    eg = ExponentiatedGradient(
        base_clf(),
        constraint,
        eps=0.01,
        max_iter=50,
    )
    fl_logger = logging.getLogger("fairlearn")
    fl_logger.setLevel(logging.DEBUG)
    eg.fit(X_fit, y_fit, sensitive_features=A_fit)
    fl_logger.setLevel(logging.WARNING)
    logging.info("expgrad oracle calls: %d", eg.n_oracle_calls_)
    y_pred = eg.predict(X_test_pre)
    y_score = y_pred.astype(float)
    return {"intervention": "expgrad", **compute_metrics(y_test, y_pred, y_score, A_test)}


def _stratified_fit_calib_split(
    n: int,
    y: pd.Series,
    A: pd.Series,
    calib_frac: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    strata = (A.astype(str) + "|" + y.astype(str)).to_numpy()
    strata_counts = pd.Series(strata).value_counts()
    n_calib_per_stratum = (strata_counts * calib_frac).round().astype(int)
    if (strata_counts - n_calib_per_stratum < 1).any() or (n_calib_per_stratum < 1).any():
        bad = strata_counts[(strata_counts - n_calib_per_stratum < 1) | (n_calib_per_stratum < 1)]
        raise ValueError(
            f"(y, A) stratified split with calib_frac={calib_frac} leaves a stratum empty. "
            f"Offending strata (count): {bad.to_dict()}"
        )
    all_idx = np.arange(n)
    fit_idx, calib_idx = train_test_split(
        all_idx,
        test_size=calib_frac,
        random_state=split_seed,
        shuffle=True,
        stratify=strata,
    )
    return fit_idx, calib_idx


def run_threshold(
    X_train_pre: np.ndarray,
    y_train: pd.Series,
    A_train: pd.Series,
    X_test_pre: np.ndarray,
    y_test: pd.Series,
    A_test: pd.Series,
    *,
    target_gap: str = "eo",
    mode: str = "naive",
    calib_frac: float = 0.2,
    split_seed: int | None = None,
) -> dict[str, Any]:
    if target_gap == "eo":
        constraints_str = "equalized_odds"
    elif target_gap == "dp":
        constraints_str = "demographic_parity"
    else:
        raise ValueError(f"target_gap must be 'eo' or 'dp'; got {target_gap!r}")
    if mode not in ("naive", "honest"):
        raise ValueError(f"mode must be 'naive' or 'honest'; got {mode!r}")
    logging.info(
        "threshold target_gap=%s constraints=%s mode=%s",
        target_gap, constraints_str, mode,
    )

    if mode == "naive":
        clf = base_clf()
        clf.fit(X_train_pre, y_train)
        X_cal_pre, y_cal, A_cal = X_train_pre, y_train, A_train
    else:
        n = X_train_pre.shape[0]
        fit_idx, calib_idx = _stratified_fit_calib_split(
            n,
            y_train,
            A_train,
            calib_frac=calib_frac,
            split_seed=split_seed if split_seed is not None else 0,
        )
        logging.info(
            "threshold honest split: n=%d fit=%d calibrate=%d calib_frac=%.3f",
            n, len(fit_idx), len(calib_idx), calib_frac,
        )
        clf = base_clf()
        clf.fit(X_train_pre[fit_idx], y_train.iloc[fit_idx])
        X_cal_pre = X_train_pre[calib_idx]
        y_cal = y_train.iloc[calib_idx]
        A_cal = A_train.iloc[calib_idx]

    to = ThresholdOptimizer(
        estimator=clf,
        constraints=constraints_str,
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    to.fit(X_cal_pre, y_cal, sensitive_features=A_cal)
    y_pred = to.predict(X_test_pre, sensitive_features=A_test)
    y_score = clf.predict_proba(X_test_pre)[:, 1]
    return {"intervention": "threshold", **compute_metrics(y_test, y_pred, y_score, A_test)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean XGB baseline (stratified expgrad).")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory with acs_prepared.*, idx_train.npy, idx_test.npy.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results"),
        help="Directory for the output CSV.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument(
        "--cells",
        nargs="+",
        choices=CELL_CHOICES,
        default=CELL_CHOICES,
        help=(
            "Subset of (intervention, target_gap) cells to run. "
            "Default: all six. Note: reweighing is DP-only by construction."
        ),
    )
    p.add_argument(
        "--expgrad-subsample",
        type=int,
        default=20000,
        metavar="N",
        help="Rows subsampled per expgrad oracle call. 0 = full training set.",
    )
    p.add_argument(
        "--min-per-group",
        type=int,
        default=200,
        metavar="N",
        help="Minimum rows per RAC1P group in the stratified expgrad subsample.",
    )
    p.add_argument(
        "--threshold-calib-frac",
        type=float,
        default=0.2,
        help="Fraction of synth train held out for ThresholdOptimizer.fit in "
             "honest-mode threshold cells. Stratified on (y, RAC1P).",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="baseline_xgb.csv",
        help="Output filename under --output-dir.",
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
    to_run = set(args.cells)

    # Re-seed before every cell so each (intervention, target_gap, variant)
    # starts from an identical RNG state regardless of cell ordering.
    # Mirrors the grid driver's behavior; see train_grid.py for the rationale.
    if "unmitigated" in to_run:
        set_seeds(args.seed)
        logging.info("running: unmitigated ...")
        t0 = time.perf_counter()
        r = run_unmitigated(X_train_pre, y_train, X_test_pre, y_test, A_test)
        r["target_gap"] = "none"
        r["variant"] = "none"
        logging.info("done: unmitigated (%.1fs)  %s", time.perf_counter() - t0, r)
        results.append(r)

    if "reweighing" in to_run:
        set_seeds(args.seed)
        logging.info("running: reweighing (target_gap=dp) ...")
        t0 = time.perf_counter()
        rw_preprocessor = make_preprocessor()
        rw_preprocessor.fit(X_train)
        X_test_pre_rw = rw_preprocessor.transform(X_test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_reweighing(X_train, y_train, A_train, rw_preprocessor, X_test_pre_rw, y_test, A_test)
        r["target_gap"] = "dp"
        r["variant"] = "none"
        logging.info("done: reweighing (%.1fs)  %s", time.perf_counter() - t0, r)
        results.append(r)

    for tg in ("dp", "eo"):
        for sampler in ("uniform", "stratified"):
            cell = f"expgrad-{tg}-{sampler}"
            if cell not in to_run:
                continue
            set_seeds(args.seed)
            logging.info("running: expgrad target_gap=%s sampler=%s ...", tg, sampler)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = run_expgrad(
                    X_train_pre, y_train, A_train, X_test_pre, y_test, A_test,
                    subsample=args.expgrad_subsample,
                    min_per_group=args.min_per_group,
                    seed=args.seed,
                    target_gap=tg,
                    sampler=sampler,
                )
            r["target_gap"] = tg
            r["variant"] = sampler
            logging.info("done: expgrad-%s-%s (%.1fs)  %s", tg, sampler, time.perf_counter() - t0, r)
            results.append(r)

    for tg in ("dp", "eo"):
        for mode in ("naive", "honest"):
            cell = f"threshold-{tg}-{mode}"
            if cell not in to_run:
                continue
            set_seeds(args.seed)
            logging.info("running: threshold target_gap=%s mode=%s ...", tg, mode)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = run_threshold(
                    X_train_pre, y_train, A_train, X_test_pre, y_test, A_test,
                    target_gap=tg,
                    mode=mode,
                    calib_frac=args.threshold_calib_frac,
                    split_seed=args.seed,
                )
            r["target_gap"] = tg
            r["variant"] = mode
            logging.info("done: threshold-%s-%s (%.1fs)  %s", tg, mode, time.perf_counter() - t0, r)
            results.append(r)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_name
    df_out = pd.DataFrame(results)[
        ["intervention", "target_gap", "variant", "accuracy", "auc", "dp_gap", "eo_gap"]
    ]
    df_out.to_csv(out_path, index=False)
    logging.info("wrote %s", out_path)
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
