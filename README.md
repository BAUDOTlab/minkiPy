# minkiPy

<p align="center">
  <img src="minkiPy/minkiPy_illust.png" alt="3D density distribution separated into level sets" width="520">
</p>

`minkiPy` is a Python framework for the differential analysis of gene spatial organisation in spatial transcriptomics data using Minkowski functionals and tensors.

## Core input requirement

The core `minkiPy` workflow is technology-agnostic. It expects as input a `pandas.DataFrame` containing transcript-level spatial coordinates with the following required columns:

- `gene`
- `global_x`
- `global_y`

This input format is central to the package design. Once data are represented in this generic schema, the same workflow can be applied to both imaging-based and sequencing-based spatial transcriptomics data.

---

## Method overview

For each gene, `minkiPy` reconstructs a spatial density field from transcript coordinates and computes a Minkowski profile across multiple level sets.

Each Minkowski profile is based on three Minkowski functionals and one tensor-derived anisotropy index:

- `W0` (area)
- `W1` (boundary length)
- `W2` (Euler-characteristic-related quantity)
- `beta` (anisotropy index derived from a Minkowski tensor)

These four quantities are evaluated across a level-set grid and stacked as a `(4, LS)` Minkowski profile for each gene.

`minkiPy` can also generate Monte Carlo realisations to estimate profile covariance. When covariance is available, downstream comparisons can use covariance-aware Gaussian 2-Wasserstein distances. Otherwise, Euclidean distances between Minkowski profiles provide a simpler exploratory alternative.

---

## Repository structure

```text
minkiPy/
├── minkiPy/                              # Core package
│   ├── minkowski_core.py                 # Per-gene Minkowski profile computation
│   ├── mpi_driver.py                     # MPI distribution + auto-MPI wrapper
│   ├── cli.py                            # Command-line entry logic
│   ├── io.py                             # NPZ/HDF5 output writing and merge
│   └── downstream/                       # Post-processing, distances computation, data visualisation
├── minkiPy_env.yaml                      # Conda environment
├── minkiPy_exploratory_workflow.ipynb    # Lightweight introduction and exploratory workflow
├── minkiPy_FSHD_complete_workflow.ipynb  # Complete workflow reproducing the FSHD application figures of the article
├── minkiPy_CRC_complete_workflow.ipynb   # Complete workflow reproducing the CRC application figures of the article
└── examples/                             # Data staging directory used by notebooks
```

---

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/BAUDOTlab/minkiPy.git
cd minkiPy
```

### 2) Install an MPI implementation on your machine for parallelization (required)

`mpi4py` is a Python binding, but it still requires a system MPI runtime (`mpirun`/`mpiexec`) to be installed first.

Check whether MPI is already available:

```bash
mpirun --version
```

If this command is not found, install MPI first:

- **Ubuntu/Debian**
  ```bash
  sudo apt update
  sudo apt install -y openmpi-bin libopenmpi-dev
  ```
- **macOS (Homebrew)**
  ```bash
  brew install open-mpi
  ```
- **Conda-only setup (cross-platform)**
  ```bash
  conda install -c conda-forge openmpi mpi4py
  ```

On HPC clusters, MPI is often provided through environment modules (for example `module load openmpi` or `module load mpich`).

### 3) Create the conda environment

The repository provides the environment file `minkiPy_env.yaml`.

```bash
conda env create -f minkiPy_env.yaml
conda activate minkiPy
```

If you want to use the `minkiPy` conda environment from Jupyter Notebook or JupyterLab, you should also register it as a Jupyter kernel:

```bash
python -m ipykernel install --user --name minkiPy --display-name "Python (minkiPy)"
```

This step is necessary if you want the environment to appear as an available kernel in Jupyter.

### 4) Use the package from the repository root

The repository does not yet include packaging metadata such as `pyproject.toml` or `setup.py`, although this will be added in a future update. For now, `minkiPy` is typically used directly from the repository root, or with `PYTHONPATH` pointing to it.

---

## Input format

At minimum, supply a transcript table equivalent to:

```python
import pandas as pd

transcripts_df = pd.DataFrame({
    "gene": [...],
    "global_x": [...],
    "global_y": [...],
})
```

Notes:

- `gene` is treated as string identity.
- `global_x` and `global_y` are spatial coordinates, expressed in micrometres, in a common spatial reference frame.
- Upstream conversion from platform-specific files is intentionally left to the user.
- Minkowski profiles are computed independently for each sample. Comparisons between samples are performed during downstream analysis.

---

## Quick start (Python)

```python
import minkiPy

h5_path = minkiPy.compute_Minkowski_profiles(
    transcripts_df,
    name="sample_A",          # Name used to label the output sample
    output_path="results",    # Directory where output files will be written
    resolution=20.0,          # Spatial grid resolution, in micrometres
    nbr=25,                   # Number of level sets used to build the Minkowski profiles
    n_cov_samples=None,       # Use the default number of Monte Carlo realisations determined by minkiPy; set to 0 for a faster exploratory run without covariance estimation
    # mpi_procs is optional:
    # - if omitted, minkiPy automatically uses all available CPUs
    # - set mpi_procs=1 to force single-process execution
)

```

This computes per-gene profiles and writes a merged file:

`results/minkiPy_merged_resolution_<resolution>_<name>.h5`

After computing Minkowski profiles for one sample, the same step can be repeated for additional samples, for example `sample_B`, `sample_C`, and so on. Each run produces one merged HDF5 output file.

Once several samples have been processed, these merged outputs can be loaded together with `process_data` to start the downstream analysis. A typical workflow is to define the list of output files, specify the sample order, and optionally define groups of samples corresponding to biological conditions.

```python
filepaths = [
    "results/minkiPy_merged_resolution_20.0_sample_A.h5",
    "results/minkiPy_merged_resolution_20.0_sample_B.h5",
]

ordered_conditions = [
    "sample_A",
    "sample_B",
]

data = minkiPy.process_data(
    filepaths,
    ordered_conditions=ordered_conditions,
    verbose=True,
)
```

This creates a common data object that can then be used for downstream analyses. More complete examples are provided in the notebooks included in the repository.

---

## Command-line usage

The CLI is MPI-aware. Recommended invocation is via `python -m minkiPy` under `mpirun`.

```bash
mpirun -n 8 python -m minkiPy \
  --input transcripts.csv \
  --name sample_A \
  --output-path results \
  --resolution 20 \
  --nbr 25
```

If your file uses different column names:

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

Supported input formats in the CLI loader: `.csv`, `.txt`, `.tsv`, `.parquet`.

---

## Python usage patterns

### Standard MPI execution

`minkiPy` can be used in a standard MPI context, for example when a Python script is launched explicitly with `mpirun` or `mpiexec`. In this case, `compute_Minkowski_profiles(...)` runs within the MPI execution environment and distributes the computation across processes.

### MPI execution from a Python script or notebook

`minkiPy` also provides an integrated wrapper that makes MPI execution possible directly from a standard Python script or a Jupyter notebook, without manually launching Python under `mpirun`. In this case, it is sufficient to pass the desired number of MPI processes to `compute_Minkowski_profiles(...)`, for example:

```python
h5_path = minkiPy.compute_Minkowski_profiles(
    transcripts_df,
    name="sample_A",
    output_path="results",
    resolution=20.0,
    nbr=25,
    mpi_procs=60,          # Adapt this to the number of MPI processes you want to use
    use_hwthreads=True,
)
```

This is particularly convenient in notebook-based workflows and Python scripts, while still allowing efficient parallel execution on multi-core or multi-node systems.


### Optional MPI-related parameters in `compute_Minkowski_profiles`

`compute_Minkowski_profiles(...)` exposes a few MPI options that are useful when running from Python or notebooks:

- `mpi_procs` (`int | None`, default: `None`)  
  Number of MPI processes to launch when you are **not already under MPI**.  
  - `None` (default): automatically uses `SLURM_NTASKS` if defined, otherwise `os.cpu_count()`.
  - `1`: disables auto-MPI spawning and runs in a single Python process.
  - `>1`: launches `mpirun -n <mpi_procs> ...`.
- `use_hwthreads` (`bool`, default: `False`)  
  Adds `--use-hwthread-cpus` to `mpirun` (OpenMPI-style) to also use logical CPUs (hyper-threads).
- `oversubscribe` (`bool`, default: `False`)  
  Adds `--map-by :OVERSUBSCRIBE`, which can help when launching more ranks than available slots.
- `extra_mpirun_args` (`list[str] | None`, default: `None`)  
  Additional flags appended to the `mpirun` command (for scheduler/network tuning, binding policies, etc.).
- `tmp_dir` (`str | None`, default: `None`)  
  Temporary directory used to stage the input DataFrame and config for spawned MPI workers.

For users unfamiliar with MPI, the default behaviour is usually sufficient: install MPI once, call `compute_Minkowski_profiles(...)` normally, and let minkiPy use all detected CPUs automatically.


---

## Notebook overview

The repository includes three main notebooks:

- `minkiPy_FSHD_complete_workflow.ipynb`: complete end-to-end workflow for the FSHD application presented in the associated paper.
- `minkiPy_CRC_complete_workflow.ipynb`: complete end-to-end workflow for the CRC application presented in the associated paper.
- `minkiPy_exploratory_workflow.ipynb`: lightweight practical introduction to `minkiPy` for rapid exploratory use.

The two complete notebooks reproduce the full analysis pipelines used in the paper, from data download and preprocessing to Minkowski-profile computation and downstream analysis. They provide the full workflows required to reproduce the figures associated with the FSHD and CRC application sections of the manuscript, and illustrate the use of the downstream analysis functions in realistic end-to-end settings.

The exploratory notebook is intended as a faster entry point for new users. It shows how to prepare data, run `minkiPy`, and obtain a first exploratory analysis without Monte Carlo realisations or covariance estimation. In this setting, downstream distances are Euclidean rather than covariance-aware 2-Wasserstein distances. This notebook is useful for quickly understanding the package and visualising its main capabilities on the example data. For rigorous analyses intended for publication, the complete covariance-aware workflow is recommended.

---

## Citation

If you use `minkiPy`, please cite both the software repository and the associated manuscript.
