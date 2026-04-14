#!/usr/bin/env python3
"""Download ACS PUMS (ACSIncome), stratified train/test indices, and DPMM domain."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split

CAT_COLS = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]
NUM_COLS = ["AGEP", "WKHP"]
TARGET = "PINCP"


def _json_scalar(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


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


def load_acs_frame(
    survey_year: str,
    horizon: str,
    survey: str,
    states: list[str],
    folktables_root: Path,
    download: bool,
) -> pd.DataFrame:
    data_source = ACSDataSource(
        survey_year=survey_year,
        horizon=horizon,
        survey=survey,
        root_dir=str(folktables_root),
    )
    raw = data_source.get_data(states=states, download=download)
    features, label, _ = ACSIncome.df_to_pandas(raw)
    out = features.copy()
    out[TARGET] = label[TARGET].astype(int)
    return out


def make_split_indices(
    y: pd.Series, test_size: float, split_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=split_seed
    )
    return idx_train, idx_test


def build_domain(df_train: pd.DataFrame) -> dict[str, dict[str, Any]]:
    domain: dict[str, dict[str, Any]] = {}
    for col in NUM_COLS:
        domain[col] = {
            "lower": float(df_train[col].min()),
            "upper": float(df_train[col].max()),
        }
    for col in CAT_COLS + [TARGET]:
        cat = pd.Categorical(df_train[col])
        cats = sorted(cat.categories.tolist())
        domain[col] = {"categories": [_json_scalar(c) for c in cats]}
    return domain


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare ACSIncome data and DPMM domain.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for parquet, indices, and domain JSON.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed (numpy, random, torch if installed).")
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for train_test_split only; defaults to --seed.",
    )
    p.add_argument("--survey-year", type=str, default="2018")
    p.add_argument("--horizon", type=str, default="1-Year")
    p.add_argument("--survey", type=str, default="person")
    p.add_argument(
        "--states",
        nargs="+",
        default=["CA"],
        help="State postal codes, e.g. CA NY",
    )
    p.add_argument(
        "--folktables-root",
        type=Path,
        default=Path("data"),
        help="Folktables ACS cache root (passed to ACSDataSource root_dir).",
    )
    p.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download ACS if missing (default: true).",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument(
        "--format",
        choices=("pickle", "parquet"),
        default="pickle",
        help="Table serialization (parquet needs pyarrow or fastparquet).",
    )
    p.add_argument(
        "--table-name",
        type=str,
        default=None,
        help="Output filename under output-dir; default acs_prepared.pkl or .parquet from --format.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    set_seeds(args.seed)
    logging.info("config: %s", vars(args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.folktables_root.mkdir(parents=True, exist_ok=True)

    df = load_acs_frame(
        survey_year=args.survey_year,
        horizon=args.horizon,
        survey=args.survey,
        states=list(args.states),
        folktables_root=args.folktables_root,
        download=args.download,
    )
    logging.info("loaded %d rows, columns: %s", len(df), list(df.columns))

    y = df[TARGET]
    idx_train, idx_test = make_split_indices(y, args.test_size, split_seed)

    if args.table_name:
        table_path = args.output_dir / args.table_name
    else:
        ext = "parquet" if args.format == "parquet" else "pkl"
        table_path = args.output_dir / f"acs_prepared.{ext}"

    if args.format == "parquet":
        df.to_parquet(table_path, index=False)
    else:
        df.to_pickle(table_path)
    np.save(args.output_dir / "idx_train.npy", idx_train)
    np.save(args.output_dir / "idx_test.npy", idx_test)

    df_train = df.iloc[idx_train].copy()
    for col in CAT_COLS + [TARGET]:
        df_train[col] = pd.Categorical(df_train[col])

    domain = build_domain(df_train)
    domain_path = args.output_dir / "domain.json"
    with domain_path.open("w", encoding="utf-8") as f:
        json.dump(domain, f, indent=2, sort_keys=True)

    logging.info("wrote %s", table_path)
    logging.info("wrote %s, %s", args.output_dir / "idx_train.npy", args.output_dir / "idx_test.npy")
    logging.info("wrote %s", domain_path)


if __name__ == "__main__":
    main()
