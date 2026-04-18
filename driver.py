"""Experiment driver: (train_df, test_df, attr, model, intervention) -> metrics."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from interventions import (
    fit_baseline,
    fit_exp_gradient,
    fit_reweighing,
    fit_threshold_optimizer,
)
from metrics import compute_fairness

TARGET = "PINCP"
NUM_COLS = ["AGEP", "WKHP"]
CAT_COLS = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]

Intervention = Literal["baseline", "reweighing", "exp_gradient", "threshold"]
_FIT_FNS = {
    "baseline": fit_baseline,
    "reweighing": fit_reweighing,
    "exp_gradient": fit_exp_gradient,
    "threshold": fit_threshold_optimizer,
}


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )


def run_one(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protected_attr: str,
    estimator: BaseEstimator,
    intervention: Intervention,
) -> dict[str, float]:
    pre = build_preprocessor().fit(train_df[NUM_COLS + CAT_COLS])
    Xtr = pre.transform(train_df[NUM_COLS + CAT_COLS])
    Xte = pre.transform(test_df[NUM_COLS + CAT_COLS])
    ytr = train_df[TARGET].astype(int).to_numpy()
    yte = test_df[TARGET].astype(int).to_numpy()
    A_tr = train_df[protected_attr].reset_index(drop=True)
    A_te = test_df[protected_attr].reset_index(drop=True)

    predict = _FIT_FNS[intervention](estimator, Xtr, ytr, A_tr)
    y_pred, y_score = predict(Xte, A_te)
    return compute_fairness(yte, y_pred, A_te, y_score)
