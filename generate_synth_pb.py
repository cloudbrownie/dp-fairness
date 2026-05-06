#!/usr/bin/env python3
"""Step 2 (PrivBayes variant): PrivBayes+PGM DP synthetic data at each epsilon × seed.

Mirrors generate_synth_mst.py but uses dpmm.pipelines.PrivBayesPipeline. The
PGM part is internal to dpmm's PrivBayes implementation (same mbi/PGM
machinery as MST/AIM) and needs no extra wiring at this layer.

Default output directory is data/synth_pb/ so PrivBayes outputs do not
collide with the MST or AIM grids.

PrivBayes-specific knobs (`--max-model-size`, `--no-compress`) are exposed
but only forwarded to PrivBayesPipeline when the user overrides them; by
default we fall through to the library defaults (compress=True,
max_model_size=None).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dpmm.pipelines import PrivBayesPipeline

TARGET = "PINCP"
DEFAULT_EPSILONS = [1, 2, 4, 8]
DEFAULT_N_SEEDS = 10


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
    raise FileNotFoundError(
        f"No acs_prepared.parquet or acs_prepared.pkl under {data_dir}"
    )


def load_full_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported table format: {path}")


def load_domain(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def align_domain(domain: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    d_keys = set(domain)
    c_set = set(columns)
    if d_keys != c_set:
        raise ValueError(
            f"domain keys != DataFrame columns. only_in_domain={d_keys - c_set}, "
            f"only_in_df={c_set - d_keys}"
        )
    return {c: domain[c] for c in columns}


def build_df_train(df: pd.DataFrame, idx_train: np.ndarray, cat_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if c != TARGET] + [TARGET]
    out = df.iloc[idx_train][cols].copy()
    for col in cat_cols:
        if col not in out.columns:
            raise KeyError(f"expected column {col!r} in prepared table")
        out[col] = pd.Categorical(out[col])
    if TARGET not in out.columns:
        raise KeyError(f"expected column {TARGET!r} in prepared table")
    out[TARGET] = pd.Categorical(out[TARGET])
    return out


def infer_cat_cols(columns: list[str]) -> list[str]:
    num = {"AGEP", "WKHP"}
    return [c for c in columns if c not in num]


def ensure_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "parquet output needs pyarrow or fastparquet (pip install pyarrow) "
                "or pass --format pickle"
            ) from e


def save_synth(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PrivBayes+PGM synthetic data grid: eps × seeds on ACS train split."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory with acs_prepared.*, idx_train.npy, domain.json.",
    )
    p.add_argument(
        "--prepared-path",
        type=Path,
        default=None,
        help="Override path to acs_prepared table; default: discover under --data-dir.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synth_pb"),
        help="Directory for eps{{e}}_seed{{s}} outputs.",
    )
    p.add_argument(
        "--format",
        choices=("parquet", "pickle"),
        default="parquet",
        help="Synthetic file format (parquet needs pyarrow or fastparquet).",
    )
    p.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=DEFAULT_EPSILONS,
        help="Privacy epsilon values.",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Number of runs per epsilon (seeds 0 .. n-1).",
    )
    p.add_argument(
        "--proc-epsilon",
        type=float,
        default=0.1,
        help="Processing (binner) epsilon for PrivBayesPipeline.",
    )
    p.add_argument(
        "--delta",
        type=float,
        default=None,
        help="DP delta; default 1/n^2 for n = len(train).",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="PrivBayes gen n_jobs (-1 = all cores).",
    )
    p.add_argument(
        "--max-model-size",
        type=int,
        default=None,
        help=(
            "Cap on PGM model size used by PrivBayes. If unset, falls "
            "through to PrivBayesPipeline's library default."
        ),
    )
    p.add_argument(
        "--no-compress",
        dest="compress",
        action="store_false",
        default=None,
        help=(
            "Disable PrivBayesPipeline's `compress` flag. If unset, falls "
            "through to the library default (compress=True)."
        ),
    )
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed (numpy, random, torch).")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and replace existing output files.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs and exit without fitting.",
    )
    p.add_argument(
        "--only-epsilon",
        type=float,
        default=None,
        help="With --only-seed, run a single (epsilon, seed) cell (for Slurm arrays).",
    )
    p.add_argument(
        "--only-seed",
        type=int,
        default=None,
        help="With --only-epsilon, run a single grid cell.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    set_seeds(args.seed)
    logging.info("config: %s", vars(args))

    data_dir = args.data_dir
    table_path = args.prepared_path if args.prepared_path is not None else find_prepared_table(data_dir)
    idx_path = data_dir / "idx_train.npy"
    domain_path = data_dir / "domain.json"

    if not idx_path.is_file():
        raise FileNotFoundError(idx_path)
    if not domain_path.is_file():
        raise FileNotFoundError(domain_path)

    df = load_full_table(table_path)
    idx_train = np.load(idx_path)
    domain_raw = load_domain(domain_path)

    cat_cols = [c for c in infer_cat_cols(list(df.columns)) if c != TARGET]
    cat_cols_with_target = cat_cols + [TARGET]
    df_train = build_df_train(df, idx_train, cat_cols_with_target)
    domain = align_domain(domain_raw, list(df_train.columns))

    n = len(idx_train)
    delta = args.delta if args.delta is not None else 1.0 / (n**2)
    logging.info(
        "train rows=%d, cols=%s, delta=%.4e, table=%s",
        n,
        list(df_train.columns),
        delta,
        table_path,
    )

    ext = "parquet" if args.format == "parquet" else "pkl"
    oe, os_ = args.only_epsilon, args.only_seed
    if (oe is None) ^ (os_ is None):
        raise SystemExit("use both --only-epsilon and --only-seed together, or neither.")
    planned: list[tuple[float, int, Path]] = []
    if oe is not None and os_ is not None:
        out_path = args.output_dir / f"eps{oe:g}_seed{os_}.{ext}"
        planned.append((float(oe), int(os_), out_path))
    else:
        for eps in args.epsilons:
            for run_seed in range(args.n_seeds):
                out_path = args.output_dir / f"eps{eps:g}_seed{run_seed}.{ext}"
                planned.append((eps, run_seed, out_path))

    if args.dry_run:
        for eps, run_seed, out_path in planned:
            logging.info("would run eps=%s seed=%s -> %s", eps, run_seed, out_path)
        return

    if args.format == "parquet":
        ensure_parquet_engine()

    extra_kwargs: dict[str, Any] = {}
    if args.max_model_size is not None:
        extra_kwargs["max_model_size"] = args.max_model_size
    if args.compress is not None:
        extra_kwargs["compress"] = args.compress

    for eps, run_seed, out_path in planned:
        if out_path.is_file() and not args.overwrite:
            logging.info("skip existing %s", out_path)
            continue
        t0 = time.perf_counter()
        model = PrivBayesPipeline(
            epsilon=float(eps),
            delta=delta,
            proc_epsilon=args.proc_epsilon,
            n_jobs=args.n_jobs,
            **extra_kwargs,
        )
        model.fit(df_train, domain=domain, random_state=run_seed)
        synth = model.generate(n_records=n, random_state=run_seed)
        save_synth(synth, out_path, args.format)
        elapsed = time.perf_counter() - t0
        logging.info("wrote %s (%.1fs)", out_path, elapsed)


if __name__ == "__main__":
    main()
