"""Accuracy and fairness metrics for a single (y_true, y_pred, groups) triple."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score, roc_auc_score

ArrayLike = np.ndarray | pd.Series | Sequence[int]


def compute_fairness(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: pd.Series,
    y_score: ArrayLike | None = None,
) -> dict[str, float]:
    """Return overall accuracy/AUC plus pairwise group gaps for a binary task.

    Gaps are max pairwise differences across groups (`between_groups`).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc_by_group = MetricFrame(
        metrics=accuracy_score,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float("nan") if y_score is None else float(roc_auc_score(y_true, np.asarray(y_score))),
        "accuracy_parity_gap": float(acc_by_group.difference(method="between_groups")),
        "demographic_parity_gap": float(
            demographic_parity_difference(y_true, y_pred, sensitive_features=groups)
        ),
        "equalized_odds_gap": float(
            equalized_odds_difference(y_true, y_pred, sensitive_features=groups)
        ),
    }
