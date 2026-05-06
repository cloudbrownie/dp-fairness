from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

GRID_DIRS: dict[tuple[str, str], str] = {
    ("logreg", "mst"): "grid",
    ("logreg", "pb"): "grid_pb",
    ("xgboost", "mst"): "grid_xgb",
    ("xgboost", "pb"): "grid_xgb_pb",
}
BASELINES: dict[str, str] = {
    "logreg": "baseline.csv",
    "xgboost": "baseline_xgb.csv",
}

SEED_SENTINEL: int = -1
CLEAN_EPS: float = float("inf")
CLEAN_SYNTHESIS: str = "clean"

LONG_COLS: list[str] = [
    "model",
    "synthesis",
    "eps",
    "synth_seed",
    "intervention",
    "target_gap",
    "variant",
    "accuracy",
    "auc",
    "dp_gap",
    "eo_gap",
    "error",
]


def _ensure_error_col(df: pd.DataFrame) -> pd.DataFrame:
    if "error" not in df.columns:
        df = df.copy()
        df["error"] = ""
    else:
        df = df.copy()
        df["error"] = df["error"].fillna("").astype(str)
    return df


def load_baseline(path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _ensure_error_col(df)
    df["model"] = model
    df["synthesis"] = CLEAN_SYNTHESIS
    df["eps"] = CLEAN_EPS
    df["synth_seed"] = SEED_SENTINEL
    return df[LONG_COLS]


def load_grid(grid_dir: Path, model: str, synthesis: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for p in sorted(grid_dir.glob("eps*_seed*.csv")):
        df = pd.read_csv(p)
        df = _ensure_error_col(df)
        df["model"] = model
        df["synthesis"] = synthesis
        rows.append(df[LONG_COLS])
    if not rows:
        return pd.DataFrame(columns=LONG_COLS)
    return pd.concat(rows, ignore_index=True)


def load_all(
    results_root: Path = Path("data/results"),
    baseline_root: Path = Path("data/results"),
    drop_failed: bool = True,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for model, base_name in BASELINES.items():
        base_path = baseline_root / base_name
        if not base_path.exists():
            logging.warning("baseline missing: %s", base_path)
            continue
        parts.append(load_baseline(base_path, model))
    for (model, synthesis), grid_name in GRID_DIRS.items():
        grid_dir = results_root / grid_name
        if not grid_dir.exists():
            logging.warning("grid dir missing: %s", grid_dir)
            continue
        parts.append(load_grid(grid_dir, model, synthesis))

    if not parts:
        return pd.DataFrame(columns=LONG_COLS)
    df = pd.concat(parts, ignore_index=True)

    df["error"] = df["error"].fillna("").astype(str)
    df["target_gap"] = df["target_gap"].astype(str)
    df["variant"] = df["variant"].astype(str)
    df["intervention"] = df["intervention"].astype(str)
    df["model"] = df["model"].astype(str)
    df["synthesis"] = df["synthesis"].astype(str)

    n_failed = int((df["error"] != "").sum())
    if drop_failed and n_failed:
        logging.info("dropping %d failed rows (NaN metrics, error != '')", n_failed)
        df = df[df["error"] == ""].reset_index(drop=True)
    return df


def split_clean_synth(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = df[df["synthesis"] == CLEAN_SYNTHESIS].reset_index(drop=True)
    synth = df[df["synthesis"] != CLEAN_SYNTHESIS].reset_index(drop=True)
    return clean, synth


def cell_label(intervention: str, target_gap: str, variant: str) -> str:
    if intervention == "unmitigated":
        return "unmitigated"
    if intervention == "reweighing":
        return "reweighing"
    return f"{intervention}-{target_gap}-{variant}"


def add_cell_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cell"] = [
        cell_label(i, t, v)
        for i, t, v in zip(df["intervention"], df["target_gap"], df["variant"])
    ]
    return df


def target_gap_metric(target_gap: str) -> str | None:
    if target_gap == "dp":
        return "dp_gap"
    if target_gap == "eo":
        return "eo_gap"
    return None
