# DP Fairness Interventions Study

This project studies how **differentially private (DP) synthetic data generation** interacts with **algorithmic fairness interventions**. We generate synthetic versions of a real census-derived income dataset at multiple privacy budgets (ε), train classifiers on the synthetic data alone, and measure how well three fairness interventions (reweighting, exponentiated gradient, threshold adjustment) recover equalized odds compared to a clean-data baseline.

## Dataset and connection to the US Census

We use the **ACS PUMS Income** task from [Folktables](https://github.com/socialfoundations/folktables), constructed from the 2018 American Community Survey Public Use Microdata Sample (PUMS) for California. The ACS is conducted annually by the US Census Bureau as the primary source of detailed socioeconomic statistics between decennial censuses. The Folktables `ACSIncome` task replicates the adult income classification benchmark on this data:

- **Target**: whether personal income (`PINCP`) exceeds $50,000
- **Features**: age, class of worker, education, marital status, occupation, place of birth, relationship, usual hours worked, sex, race (10 columns)
- **Protected attributes**: `RAC1P` (race, 1–9) and `SEX` (1=Male, 2=Female)
- **Population**: adults in California who worked at least some hours (≈195k rows after the standard `adult_filter`)

**Synthetic data and the Census connection.** The synthetic datasets are produced by the **MST (Maximum Spanning Tree) mechanism** implemented in the [`dpmm`](https://github.com/ryan112358/private-pgm) library — the same class of graphical-model mechanisms underlying the Census Bureau's **TopDown algorithm** used for the 2020 Decennial Census. Step 5 of the pipeline validates this connection by comparing the group-level distortions in our MST synthetic data against the TopDown distortions published by NHGIS in their 2010 DP demonstration files.

## Environment setup

The project uses [uv](https://docs.astral.sh/uv/) for deterministic environment management. All dependencies and their pinned versions are declared in `pyproject.toml`; `uv.lock` ensures exact reproducibility.

```bash
# 1. Install uv (once per machine — skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the virtual environment and install all dependencies
#    (run from the project root)
uv sync

# 3. Activate (optional — uv run prefix works without activation)
source .venv/bin/activate
```

`uv sync` reads `pyproject.toml` and `uv.lock` and installs the exact pinned versions into `.venv/`. Do not use `pip install` directly — it will not respect the lockfile.

To add a new dependency:

```bash
uv add <package>          # adds to pyproject.toml and updates uv.lock
uv sync                   # installs into .venv
```

## Data pipeline

### Step 1 — ACS download and preparation (already done)

Outputs are in `data/raw/`. To regenerate from scratch:

```bash
python prepare_acs_data.py \
  --output-dir data/raw \
  --folktables-root data \
  --format pickle \
  --seed 42
```

This downloads the 2018 ACS PUMS for California (~195k rows), performs a stratified 80/20 train/test split (fixed at seed 42), and writes:

| File | Contents |
|------|----------|
| `data/raw/acs_prepared.pkl` | Full DataFrame (features + PINCP label) |
| `data/raw/idx_train.npy` | Training row indices |
| `data/raw/idx_test.npy` | Test row indices (never used for fitting) |
| `data/raw/domain.json` | dpmm domain spec (bounds for numeric cols, categories for categorical cols) |

The domain spec is passed to `MSTPipeline` so no DP budget is spent on domain estimation.

### Step 2 — DP synthetic data generation (already done)

Outputs are 40 parquet files in `data/synth/`: `eps{1,2,4,8}_seed{0..9}.parquet`. Each file contains a synthetic training set of the same size as the real training split, with the same schema (features + `PINCP`).

**To regenerate a single cell** (e.g. ε=2, seed=3):

```bash
python generate_synth_mst.py \
  --data-dir data/raw \
  --output-dir data/synth \
  --only-epsilon 2 \
  --only-seed 3
```

**To regenerate the full grid on a Slurm cluster:**

```bash
# Option A: array job (40 tasks × ~10 min each, runs in parallel)
sbatch slurm/generate_synth_mst_array_short.sbatch

# Option B: single job, all 40 runs sequentially on one node
sbatch slurm/generate_synth_mst_serial.sbatch
```

The array job (`_array_short`) is preferred — each task takes under 10 minutes and the whole grid finishes in a single scheduler slot per task. The serial job is simpler to monitor when debugging.

Privacy parameters: ε ∈ {1, 2, 4, 8}, δ = 1/n² where n is the number of training rows, `proc_epsilon=0.1`.

### Step 3 — Clean baseline

```bash
python train_baseline.py \
  --data-dir data/raw \
  --output-dir results \
  --seed 42
```

Trains and evaluates four conditions on clean data (unmitigated + reweighting + exponentiated gradient + threshold adjustment). Writes `results/baseline.csv`.

## Output structure

```
data/
  raw/            # ACS parquet/pkl, split indices, domain.json
  synth/          # eps{e}_seed{s}.parquet (40 files)
  nhgis/          # NHGIS 2010 DP demonstration files (Step 5)
results/
  baseline.csv    # clean baseline metrics (Step 3)
  grid.csv        # full (eps, seed, intervention) results (Step 4)
  nhgis_check.csv # distortion comparison (Step 5)
logs/             # Slurm stdout/stderr
```

## Smoke test

```bash
python smoke_test.py
```

Verifies that the dpmm, folktables, fairlearn, and aif360 imports work and that a small MST fit+generate cycle completes without errors.
