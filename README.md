# minkiPy

<p align="center">
  <img src="minkiPy/minkiPy_illust.png" alt="3D density distribution separated into level sets" width="520">
</p>

`minkiPy` is a Python package for differential analysis of gene spatial organisation in spatial transcriptomics data, using Minkowski functionals and tensors.

This repository accompanies the paper **"Differential Analysis of Gene Spatial Organisation with Minkowski Functionals and Tensors"** and includes:
- the `minkiPy` package,
- a command-line interface,
- an exploratory notebook to get started quickly on your own data,
- full workflow notebooks used for end-to-end analyses.

---

## Contents

1. [Input format](#input-format)
2. [Method summary](#method-summary)
3. [Installation](#installation)
4. [Quick start (Python)](#quick-start-python)
5. [Command-line usage](#command-line-usage)
6. [MPI usage patterns](#mpi-usage-patterns)
7. [Repository layout](#repository-layout)

---

## Input format

`minkiPy` expects a `pandas.DataFrame` with transcript-level coordinates and these columns:
- `gene`
- `global_x`
- `global_y`

```python
import pandas as pd

transcripts_df = pd.DataFrame({
    "gene": [...],
    "global_x": [...],
    "global_y": [...],
})
```

Notes:
- `gene` is a string identifier.
- `global_x` and `global_y` should share the same coordinate system (usually micrometres).
- Converting platform-specific files to this format is done upstream.

---

## Method summary

For each gene, `minkiPy` reconstructs a spatial density field and computes a profile across level sets.

Each profile contains:
- `W0` (area),
- `W1` (boundary length),
- `W2` (Euler-characteristic-related term),
- `beta` (anisotropy index from a Minkowski tensor).

Profiles are shaped `(4, LS)` per gene.

Optional Monte Carlo runs estimate covariance. Distances can then be covariance-aware Gaussian 2-Wasserstein, or Euclidean for fast exploration.

These profiles are the starting point for downstream analysis: sample and gene comparisons, condition-level ranking of spatial reorganisation, and embedding/graph analyses.

---

## Installation

> `mpi4py` needs an MPI runtime (`mpirun`/`mpiexec`) installed on your machine.

Before choosing an option:
- **Option A (pip from PyPI)** does **not** require cloning this repository.
- **Options B/C (YAML or local development)** require a local clone first:

```bash
git clone https://github.com/BAUDOTlab/minkiPy.git
cd minkiPy
```

### Option A (recommended): pip

1) Check MPI:

```bash
mpirun --version
```

If missing, install MPI first:

- **Ubuntu/Debian**
  ```bash
  sudo apt update
  sudo apt install -y openmpi-bin libopenmpi-dev
  ```
- **macOS (Homebrew)**
  ```bash
  brew install open-mpi
  ```
- **Conda-only**
  ```bash
  conda install -c conda-forge openmpi mpi4py
  ```

2) Update pip tooling:

```bash
python -m pip install --upgrade pip setuptools wheel
```

3) Install:

```bash
pip install minkipy-st
```

4) Verify:

```bash
python -c "import minkiPy; print('minkiPy import OK')"
python -m minkiPy --help
```

### Option B: Conda environment from YAML

Use this option from the repository root (after `git clone` and `cd minkiPy`).

1) Update Conda first:

```bash
conda update -n base -c defaults conda
```

2) Create the environment:

```bash
conda env create -f minkiPy_env.yaml
```

3) Activate it:

```bash
conda activate minkiPy
```

4) Install package from source (editable):

```bash
pip install -e .
```

5) (Optional) Add a Jupyter kernel:

```bash
python -m ipykernel install --user --name minkiPy --display-name "Python (minkiPy)"
```

### Option C: Local development install

Use this option from the repository root (after `git clone` and `cd minkiPy`).

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

### Troubleshooting

If installation fails:

1) Retry after updating pip tooling:

```bash
python -m pip install --upgrade pip setuptools wheel
```

2) For Conda setups, also update Conda:

```bash
conda update -n base -c defaults conda
```

3) Create a clean virtual environment and reinstall:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows (PowerShell): .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install minkipy-st
```

4) If MPI errors persist, re-check `mpirun --version` and ensure MPI + `mpi4py` are compatible.

---

## Quick start (Python)

```python
import minkiPy

h5_path = minkiPy.compute_Minkowski_profiles(
    transcripts_df,
    name="sample_A",
    output_path="results",
    resolution=20.0,
    nbr=25,
    n_cov_samples=None,  # default MC realisations; set 0 for faster exploratory runs
    # mpi_procs:
    # None -> auto-detect
    # 1    -> single process
    # >1   -> spawn MPI processes
)
```

Typical output file:

`results/minkiPy_merged_resolution_<resolution>_<name>.h5`

Example downstream loading:

```python
filepaths = [
    "results/minkiPy_merged_resolution_20.0_sample_A.h5",
    "results/minkiPy_merged_resolution_20.0_sample_B.h5",
]

ordered_conditions = ["sample_A", "sample_B"]

data = minkiPy.process_data(
    filepaths,
    ordered_conditions=ordered_conditions,
    verbose=True,
)
```

### Downstream analysis (beyond `process_data`)

After `process_data`, typical downstream steps include:
- condition-level averaging with `add_averaged_condition_datasets`,
- sample or gene distances with `compute_sample_distances` and `compute_gene_distances`,
- graph and embedding visualisations (`plot_dataset_graphs_from_data`, `plot_gene_graphs_from_data`, `plot_pca_grid_by_condition`),
- differential ranking and trend plots (`plot_top_changing_genes`, `plot_w2_abslog2fc_with_trend`),
- profile-level diagnostics (`plot_minkowski_profile`, `plot_w2_diag_vs_euclid_distributions`, `plot_w2_diag_vs_full_plus_euclid_distributions`).

To get started quickly with your **own** data, begin with `minkiPy_exploratory_workflow.ipynb`.

---

## Command-line usage

Run under MPI:

```bash
mpirun -n 8 python -m minkiPy \
  --input transcripts.csv \
  --name sample_A \
  --output-path results \
  --resolution 20 \
  --nbr 25
```

Custom column names:

```bash
mpirun -n 8 python -m minkiPy \
  --input transcripts.tsv \
  --sep '\t' \
  --gene-col gene_symbol \
  --x-col x \
  --y-col y \
  --name sample_A \
  --output-path results
```

Supported formats: `.csv`, `.txt`, `.tsv`, `.parquet`.

---

## MPI usage patterns

### 1) Standard MPI launch

Launch your script with `mpirun`/`mpiexec`. `compute_Minkowski_profiles(...)` uses the active MPI communicator.

### 2) Auto-MPI from Python or notebook

```python
h5_path = minkiPy.compute_Minkowski_profiles(
    transcripts_df,
    name="sample_A",
    output_path="results",
    resolution=20.0,
    nbr=25,
    mpi_procs=60,
    use_hwthreads=True,
)
```

Useful parameters:
- `mpi_procs` (`int | None`, default `None`)
- `use_hwthreads` (`bool`, default `False`)
- `oversubscribe` (`bool`, default `False`)
- `extra_mpirun_args` (`list[str] | None`)

---

## Repository layout

```text
minkiPy/
├── minkiPy/                              # Core package
│   ├── minkowski_core.py                 # Per-gene Minkowski profile computation
│   ├── mpi_driver.py                     # MPI distribution + auto-MPI wrapper
│   ├── cli.py                            # Command-line logic
│   ├── io.py                             # NPZ/HDF5 output writing and merge
│   └── downstream/                       # Post-processing, distances, visualisation
├── minkiPy_env.yaml                      # Conda environment definition
├── minkiPy_exploratory_workflow.ipynb    # Introductory exploratory workflow
├── minkiPy_FSHD_complete_workflow.ipynb  # Full FSHD workflow
├── minkiPy_CRC_complete_workflow.ipynb   # Full CRC workflow
└── examples/                             # Data staging for notebooks
```
