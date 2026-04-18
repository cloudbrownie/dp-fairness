#!/usr/bin/env python3
"""Full evaluation grid: (source, model, protected attr, intervention) -> metrics row."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from driver import run_one

ALL_INTERVENTIONS = ["baseline", "reweighing", "exp_gradient", "threshold"]
ALL_ATTRS = ["SEX", "RAC1P"]
MODELS: dict[str, Callable[[], object]] = {
    "logreg": lambda: LogisticRegression(max_iter=500),
    # "xgboost": lambda: XGBClassifier(n_estimators=200, max_depth=6, tree_method="hist", n_jobs=-1),
    # "nn": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=0),
}

RESULT_COLS = [
    "source", "epsilon", "seed", "model", "attr", "intervention",
    "accuracy", "auc", "accuracy_parity_gap", "demographic_parity_gap",
    "equalized_odds_gap", "elapsed_sec",
]


def load_clean_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_pickle(data_dir / "acs_prepared.pkl")
    idx_train = np.load(data_dir / "idx_train.npy")
    idx_test = np.load(data_dir / "idx_test.npy")
    return (
        df.iloc[idx_train].reset_index(drop=True),
        df.iloc[idx_test].reset_index(drop=True),
    )


def discover_synth(synth_dir: Path) -> list[tuple[float, int, Path]]:
    rows = []
    for p in sorted(synth_dir.glob("eps*_seed*.parquet")):
        eps_str, seed_str = p.stem.split("_")
        rows.append((float(eps_str.removeprefix("eps")), int(seed_str.removeprefix("seed")), p))
    return sorted(rows)


def existing_keys(out_path: Path) -> set[tuple]:
    if not out_path.is_file():
        return set()
    df = pd.read_csv(out_path)

    def _k(v):
        return "" if pd.isna(v) else str(v)

    return {
        (r.source, _k(r.epsilon), _k(r.seed), r.model, r.attr, r.intervention)
        for r in df.itertuples(index=False)
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fairness interventions on clean and DP-synth training sets.")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--synth-dir", type=Path, default=Path("data/synth"))
    p.add_argument("--output", type=Path, default=Path("data/results.csv"))
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--attrs", nargs="+", default=ALL_ATTRS, choices=ALL_ATTRS)
    p.add_argument("--interventions", nargs="+", default=ALL_INTERVENTIONS, choices=ALL_INTERVENTIONS)
    p.add_argument("--epsilons", type=float, nargs="+", default=None, help="Filter synth by ε; default all.")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Filter synth by seed; default all.")
    p.add_argument("--include-clean", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true", help="Ignore existing rows and rewrite file.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("config: %s", vars(args))

    sources: list[tuple[str, float | None, int | None, Path | None]] = []
    if args.include_clean:
        sources.append(("clean", None, None, None))
    for eps, seed, path in discover_synth(args.synth_dir):
        if args.epsilons is not None and eps not in args.epsilons:
            continue
        if args.seeds is not None and seed not in args.seeds:
            continue
        sources.append(("synth", eps, seed, path))

    seen = set() if args.overwrite else existing_keys(args.output)

    plan = []
    for src, eps, seed, path in sources:
        for model_name in args.models:
            for attr in args.attrs:
                for inter in args.interventions:
                    key = (
                        src,
                        "" if eps is None else str(eps),
                        "" if seed is None else str(seed),
                        model_name, attr, inter,
                    )
                    if key in seen:
                        continue
                    plan.append((src, eps, seed, path, model_name, attr, inter))

    logging.info("%d runs planned (%d already done, skipping)", len(plan), len(seen))
    if args.dry_run:
        for row in plan[:10]:
            logging.info("would run: %s", row)
        if len(plan) > 10:
            logging.info("... and %d more", len(plan) - 10)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.is_file():
        args.output.unlink()

    clean_train, test_df = load_clean_splits(args.data_dir)
    write_header = not args.output.is_file()

    for i, (src, eps, seed, path, model_name, attr, inter) in enumerate(plan, 1):
        train_df = clean_train if src == "clean" else pd.read_parquet(path)
        estimator = MODELS[model_name]()
        t0 = time.perf_counter()
        m = run_one(train_df, test_df, attr, estimator, inter)
        dt = time.perf_counter() - t0

        row = {
            "source": src, "epsilon": eps, "seed": seed,
            "model": model_name, "attr": attr, "intervention": inter,
            **m, "elapsed_sec": round(dt, 2),
        }
        pd.DataFrame([row], columns=RESULT_COLS).to_csv(
            args.output, mode="a", header=write_header, index=False
        )
        write_header = False
        logging.info(
            "[%d/%d] %s eps=%s seed=%s %s %s %s acc=%.3f EO=%.3f (%.1fs)",
            i, len(plan), src, eps, seed, model_name, attr, inter,
            m["accuracy"], m["equalized_odds_gap"], dt,
        )


if __name__ == "__main__":
    main()
