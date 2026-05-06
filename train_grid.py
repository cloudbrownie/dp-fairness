#!/usr/bin/env python3
"""Step 4: Main experiment grid — DP synthetic data × fairness interventions.

One Slurm array task = one (epsilon, seed) pair. Runs ten (intervention,
target_gap, variant) cells on the corresponding synthetic training set and
evaluates on the fixed clean test split:

  - unmitigated                                   (target_gap=none, variant=none)
  - reweighing                                   (target_gap=dp,   variant=none)
  - expgrad / {dp, eo} × {uniform, stratified}    (4 cells)
  - threshold / {dp, eo} × {naive, honest}        (4 cells)

Reweighing (Kamiran-Calders) has no EO analogue: its weights are derived to
satisfy A ⊥ Y in the training distribution, which is the demographic-parity
target.

expgrad variants:
  - uniform: 20k row uniform draw for the oracle subsample. Smallest RAC1P
    groups can end up severely under-sampled, which on LR makes EqualizedOdds
    duals ill-posed (see project.md, Task C1).
  - stratified: 20k draw with min_per_group=200 (preferred default).

threshold variants:
  - naive: fit base classifier on full synth, calibrate ThresholdOptimizer
    on the same rows. In-sample calibration; optimistic.
  - honest: stratified split on (y, RAC1P) with calib_frac=0.2; base clf on
    fit slice, ThresholdOptimizer.fit on disjoint calibrate slice.

Writes results/grid/eps{e}_seed{s}.csv with columns:
  eps, synth_seed, intervention, target_gap, variant, accuracy, auc, dp_gap, eo_gap

Both gaps are reported on every row regardless of target_gap, so off-target
gaps are recorded as side-effect measurements rather than as the metric the
method tried to control.
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

from train_baseline import (
    CAT_COLS,
    NUM_COLS,
    PROT_ATTR,
    TARGET,
    compute_metrics,
    find_prepared_table,
    make_preprocessor,
    run_expgrad,
    run_reweighing,
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


def _safe_call(
    cell_name: str,
    intervention: str,
    target_gap: str,
    variant: str,
    fn,
) -> dict[str, Any]:
    # Run a single cell's training callable. On any exception, log and return
    # a NaN row tagged with the error so the rest of the grid still completes.
    # Without this, one degenerate (eps, seed) cell (e.g. honest-split with a
    # near-empty stratum at small eps) would kill the whole task and lose the
    # other already-computed cells in the same run.
    t0 = time.perf_counter()
    try:
        r = fn()
        r.setdefault("intervention", intervention)
        r["target_gap"] = target_gap
        r["variant"] = variant
        r.setdefault("error", "")
        logging.info("done: %s (%.1fs)  %s", cell_name, time.perf_counter() - t0, r)
        return r
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logging.exception("FAILED: %s (%.1fs) -- %s", cell_name, time.perf_counter() - t0, msg)
        return {
            "intervention": intervention,
            "target_gap": target_gap,
            "variant": variant,
            "accuracy": float("nan"),
            "auc": float("nan"),
            "dp_gap": float("nan"),
            "eo_gap": float("nan"),
            "error": msg,
        }


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


def run_condition(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    A_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    A_test: pd.Series,
    cells: list[str],
    expgrad_subsample: int,
    min_per_group: int,
    seed: int,
    threshold_calib_frac: float,
) -> list[dict[str, Any]]:
    preprocessor = make_preprocessor()
    X_train_pre = preprocessor.fit_transform(X_train)  # sparse CSR, fit on synth
    X_test_pre = preprocessor.transform(X_test)        # sparse CSR, transform clean test

    nnz = X_train_pre.nnz if hasattr(X_train_pre, "nnz") else int(X_train_pre.size)
    logging.info(
        "synth train=%d  test=%d  features_after_enc=%d  sparse_nnz=%d",
        len(y_train),
        len(y_test),
        X_train_pre.shape[1],
        nnz,
    )

    to_run = set(cells)
    fairlearn_cells = {c for c in CELL_CHOICES if c.startswith(("expgrad-", "threshold-"))}
    needs_dense = bool(to_run & fairlearn_cells)
    if needs_dense and hasattr(X_train_pre, "toarray"):
        logging.info("materializing dense matrix for fairlearn (cells=%s)", sorted(to_run))
        X_train_dense = X_train_pre.toarray()
        X_test_dense = X_test_pre.toarray()
        logging.info("dense matrix: %.0f MB", X_train_dense.nbytes / 1e6)
    elif needs_dense:
        X_train_dense = X_train_pre
        X_test_dense = X_test_pre
        logging.info("preprocessor already dense: %.0f MB", X_train_dense.nbytes / 1e6)
    else:
        X_train_dense = X_test_dense = None

    results: list[dict[str, Any]] = []

    # Re-seed before every cell so each (intervention, target_gap) starts from
    # an identical RNG state regardless of cell ordering. Without this, calls
    # earlier in the loop perturb numpy / random / torch global state, which
    # leaks into the (otherwise deterministic) downstream cells via fairlearn's
    # internal RNG draws (LP iterates, threshold tie-breaks). This makes
    # `--cells expgrad-eo` produce the same numbers whether run alone or as
    # part of the full six, and restores bit-equivalence with the pre-DP/EO
    # split MST grid.
    if "unmitigated" in to_run:
        set_seeds(seed)
        logging.info("running: unmitigated ...")
        results.append(_safe_call(
            "unmitigated", "unmitigated", "none", "none",
            lambda: run_unmitigated(X_train_pre, y_train, X_test_pre, y_test, A_test),
        ))

    if "reweighing" in to_run:
        set_seeds(seed)
        logging.info("running: reweighing (target_gap=dp) ...")
        rw_preprocessor = make_preprocessor()
        rw_preprocessor.fit(X_train)
        X_test_pre_rw = rw_preprocessor.transform(X_test)
        def _rw_thunk():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return run_reweighing(
                    X_train, y_train, A_train, rw_preprocessor, X_test_pre_rw, y_test, A_test
                )
        results.append(_safe_call(
            "reweighing", "reweighing", "dp", "none", _rw_thunk,
        ))

    for tg in ("dp", "eo"):
        for sampler in ("uniform", "stratified"):
            cell = f"expgrad-{tg}-{sampler}"
            if cell not in to_run:
                continue
            set_seeds(seed)
            logging.info("running: expgrad target_gap=%s sampler=%s (max_iter=50) ...", tg, sampler)
            def _eg_thunk(tg=tg, sampler=sampler):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return run_expgrad(
                        X_train_dense, y_train, A_train, X_test_dense, y_test, A_test,
                        subsample=expgrad_subsample,
                        min_per_group=min_per_group,
                        seed=seed,
                        target_gap=tg,
                        sampler=sampler,
                    )
            results.append(_safe_call(cell, "expgrad", tg, sampler, _eg_thunk))

    for tg in ("dp", "eo"):
        for mode in ("naive", "honest"):
            cell = f"threshold-{tg}-{mode}"
            if cell not in to_run:
                continue
            set_seeds(seed)
            logging.info("running: threshold target_gap=%s mode=%s ...", tg, mode)
            def _th_thunk(tg=tg, mode=mode):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return run_threshold(
                        X_train_dense, y_train, A_train, X_test_dense, y_test, A_test,
                        target_gap=tg,
                        mode=mode,
                        calib_frac=threshold_calib_frac,
                        split_seed=seed,
                    )
            results.append(_safe_call(cell, "threshold", tg, mode, _th_thunk))

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 4: experiment grid (one Slurm task = one eps/seed pair).")
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
        default=Path("data/results/grid"),
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
        cells=args.cells,
        expgrad_subsample=args.expgrad_subsample,
        min_per_group=args.min_per_group,
        seed=args.seed,
        threshold_calib_frac=args.threshold_calib_frac,
    )

    for r in results:
        r["eps"] = args.epsilon
        r["synth_seed"] = args.synth_seed

    for r in results:
        r.setdefault("error", "")
    df_out = pd.DataFrame(results)[
        ["eps", "synth_seed", "intervention", "target_gap", "variant",
         "accuracy", "auc", "dp_gap", "eo_gap", "error"]
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"eps{eps_int}_seed{args.synth_seed}.csv"
    df_out.to_csv(out_path, index=False)
    logging.info("wrote %s", out_path)
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
