# DP Synthetic Data × Fairness Interventions

Pipeline for studying how differentially private (DP) synthetic data interacts
with three fairness interventions (reweighing, exponentiated gradient,
threshold optimizer) on the Folktables `ACSIncome` task (California 2018,
~195k rows). The grid crosses two synthesizers (MST, PrivBayes) with two
model families (LogisticRegression, XGBoost) at ε ∈ {1, 2, 4, 8}, 10 seeds
each, and is post-processed by a battery of analysis drivers in
`analysis/`.

## Repo layout

```
.
├── prepare_acs_data.py        # Step 1: ACS download + train/test split
├── generate_synth_mst.py      # Step 2a: MST synth     -> data/synth/
├── generate_synth_pb.py       # Step 2b: PrivBayes synth -> data/synth_pb/
├── train_baseline.py          # Step 3a: clean LR baseline   -> data/results/baseline.csv
├── train_baseline_xgb.py      # Step 3b: clean XGB baseline  -> data/results/baseline_xgb.csv
├── train_grid.py              # Step 4a: LR  × MST       -> data/results/grid/
├── train_grid_xgb.py          # Step 4b: XGB × MST       -> data/results/grid_xgb/
│                              # (same scripts also drive _pb / _xgb_pb via --synth-dir)
├── analysis/                  # Step 5: summarisation, stats, plots
│   ├── make_*.py              #   tables (raw_summary, ratios, risk, wilcoxon, ...)
│   ├── plot_*.py              #   figures (headline_bar, eps_sweep, heatmap, ...)
│   └── io.py / metrics.py / stats_tests.py / palette.py / style.py
├── run_all.py                 # orchestrator for the analysis/ pipeline
├── interventions.py, metrics.py, driver.py, run_grid.py
│                              # legacy single-source driver (superseded by train_grid*.py)
├── stratified.py              # stratified expgrad oracle subsample helper
├── smoke_test.py              # checks dpmm/folktables/fairlearn/aif360 imports + tiny MST run
├── slurm/                     # one sbatch per (synth, model) cell
├── pyproject.toml / uv.lock   # pinned dependencies (uv)
└── data/                      # all inputs and outputs (see below)
```

## Data layout

```
data/
├── 2018/                       # raw Folktables download cache
├── raw/
│   ├── acs_prepared.pkl        # full DataFrame (features + PINCP)
│   ├── idx_train.npy           # 80% train indices (seed 42)
│   ├── idx_test.npy            # 20% held-out test indices
│   └── domain.json             # dpmm domain spec (no DP budget on schema)
├── synth/                      # MST synth, 40 files: eps{1,2,4,8}_seed{0..9}.parquet
├── synth_pb/                   # PrivBayes synth, same naming
├── results/
│   ├── baseline.csv            # clean LR  (4 interventions)
│   ├── baseline_xgb.csv        # clean XGB (4 interventions)
│   ├── grid/                   # LR  × MST       eps{e}_seed{s}.csv (10 cells/file)
│   ├── grid_pb/                # LR  × PrivBayes
│   ├── grid_xgb/               # XGB × MST
│   └── grid_xgb_pb/            # XGB × PrivBayes
└── analysis/                   # everything written by run_all.py
    ├── raw_summary/            # raw_summary.csv, raw_summary_target.csv
    ├── ratios/                 # ratios_long.csv, ratios_summary.csv, attenuation.csv
    ├── risk/                   # risk_summary.csv
    ├── wilcoxon/               # wilcoxon.csv
    ├── side_effects/           # side_effects_summary.csv, side_effects_wilcoxon.csv
    ├── variant_tests/          # mst_vs_privbayes, logreg_vs_xgboost,
    │                           # expgrad_uniform_vs_stratified, threshold_naive_vs_honest
    ├── plateau/  pareto/  failures/
    └── plots/                  # headline_bar, headline_delta, eps_sweep_*,
                                # forest_target_gap, heatmap_target_gap,
                                # accuracy_gap_scatter{,_dp,_eo}.png
```

Each grid CSV row is one (intervention, target_gap, variant) cell:

```
eps, synth_seed, intervention, target_gap, variant, accuracy, auc, dp_gap, eo_gap
```

with the ten cells per (eps, seed) being:

| intervention  | target_gap | variant            |
|---------------|------------|--------------------|
| unmitigated   | none       | none               |
| reweighing    | dp         | none               |
| expgrad       | dp / eo    | uniform / stratified |
| threshold     | dp / eo    | naive / honest     |

Both `dp_gap` and `eo_gap` are reported on every row regardless of
`target_gap`, so the off-target gap is a side-effect measurement.

## Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once per machine
uv sync                                           # creates .venv/ from uv.lock
source .venv/bin/activate                         # optional; `uv run <cmd>` also works
```

Python is pinned to 3.11; key deps: `dpmm==0.1.9`, `folktables==0.0.12`,
`fairlearn==0.13.0`, `aif360==0.6.1`, `xgboost==3.2.0`,
`scikit-learn==1.5.0`, `numpy==1.26.4`, `pandas==2.1.0`. Use
`uv add <pkg>` rather than bare `pip install` to keep the lockfile honest.

## Running the pipeline

### Step 1 — ACS prep (already cached in `data/raw/`)

```bash
python prepare_acs_data.py --output-dir data/raw --folktables-root data --format pickle --seed 42
```

### Step 2 — DP synthetic data

Single cell:

```bash
python generate_synth_mst.py --data-dir data/raw --output-dir data/synth    --only-epsilon 2 --only-seed 3
python generate_synth_pb.py  --data-dir data/raw --output-dir data/synth_pb --only-epsilon 2 --only-seed 3
```

Full 4 × 10 grid on Slurm:

```bash
sbatch slurm/generate_synth_mst_array.sbatch
sbatch slurm/generate_synth_pb_array.sbatch
```

Privacy parameters: ε ∈ {1, 2, 4, 8}, δ = 1/n², `proc_epsilon=0.1`.

### Step 3 — Clean baselines

```bash
python train_baseline.py     --data-dir data/raw --output-dir data/results --seed 42
python train_baseline_xgb.py --data-dir data/raw --output-dir data/results --seed 42
```

Or via Slurm: `sbatch slurm/run_baseline.sbatch` / `run_baseline_xgb.sbatch`.

### Step 4 — Synth × intervention grid

Per cell (one synth file → 10 result rows):

```bash
python train_grid.py     --synth-dir data/synth    --output-dir data/results/grid     --eps 2 --synth-seed 3
python train_grid_xgb.py --synth-dir data/synth_pb --output-dir data/results/grid_xgb_pb --eps 2 --synth-seed 3
```

Full grids on Slurm — one array task per (eps, seed) per (model, synth):

```bash
sbatch slurm/run_grid_array.sbatch          # LR  × MST
sbatch slurm/run_grid_pb_array.sbatch       # LR  × PrivBayes
sbatch slurm/run_grid_xgb_array.sbatch      # XGB × MST
sbatch slurm/run_grid_xgb_pb_array.sbatch   # XGB × PrivBayes
```

### Step 5 — Analysis (tables + plots)

```bash
python run_all.py
```

`run_all.py` is a flag-free orchestrator: each module under `analysis/`
uses its own argparse defaults (read from `data/results/`, write under
`data/analysis/<artifact>/`). Modules can also be invoked individually,
e.g. `python -m analysis.make_ratios` or `python -m analysis.plot_heatmap`.

Pipeline order (mirrors `PIPELINE` in `run_all.py`):

1. tables: `make_failures`, `make_raw_summary`, `make_ratios`, `make_risk`,
   `make_wilcoxon`, `make_side_effects`, `make_variant_tests`,
   `make_plateau`, `make_pareto`
2. plots: `plot_headline_bar`, `plot_headline_delta`, `plot_eps_sweep`,
   `plot_forest`, `plot_heatmap`, `plot_scatter`

## Smoke test

```bash
python smoke_test.py
```

Verifies that `dpmm`, `folktables`, `fairlearn`, and `aif360` import and
that a tiny MST fit + sample completes end-to-end.
