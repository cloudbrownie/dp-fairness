"""Fairness interventions: reweighing, ExponentiatedGradient, ThresholdOptimizer.

Each `fit_<name>(estimator, X, y, groups)` returns a predict function
    `(X_test, groups_test) -> (y_pred, y_score | None)`
compatible with metrics.compute_fairness. y_score is None for methods that do
not expose calibrated probabilities (ExpGrad, ThresholdOptimizer).
"""

from __future__ import annotations

import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from sklearn.base import clone


def _predict_fn(model):
    def fn(X_te, groups_te=None):
        y_pred = np.asarray(model.predict(X_te))
        y_score = (
            np.asarray(model.predict_proba(X_te)[:, 1])
            if hasattr(model, "predict_proba")
            else None
        )
        return y_pred, y_score

    return fn


def fit_baseline(estimator, X, y, groups=None):
    return _predict_fn(clone(estimator).fit(X, y))


def fit_reweighing(estimator, X, y, groups):
    sw = _kc_weights(np.asarray(y), np.asarray(groups))
    return _predict_fn(clone(estimator).fit(X, y, sample_weight=sw))


def fit_exp_gradient(estimator, X, y, groups):
    mit = ExponentiatedGradient(clone(estimator), constraints=EqualizedOdds())
    mit.fit(X, y, sensitive_features=groups)

    def predict(X_te, groups_te=None):
        return np.asarray(mit.predict(X_te)), None

    return predict


def fit_threshold_optimizer(estimator, X, y, groups):
    base = clone(estimator).fit(X, y)
    to = ThresholdOptimizer(
        estimator=base, constraints="equalized_odds", prefit=True
    )
    to.fit(X, y, sensitive_features=groups)

    def predict(X_te, groups_te):
        return np.asarray(to.predict(X_te, sensitive_features=groups_te)), None

    return predict


def _kc_weights(y: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Kamiran-Calders reweighing: w(a, y) = P(A=a) P(Y=y) / P(A=a, Y=y)."""
    n = len(y)
    w = np.empty(n, dtype=float)
    for ai in np.unique(a):
        for yi in np.unique(y):
            mask = (a == ai) & (y == yi)
            if not mask.any():
                continue
            p_a = (a == ai).mean()
            p_y = (y == yi).mean()
            p_ay = mask.mean()
            w[mask] = (p_a * p_y) / p_ay
    return w
