# minkiPy

**minkiPy** is a Python package for the computation and downstream analysis of **Minkowski profiles** from spatial transcriptomics data.

It is designed to be **technology-agnostic**. It can be used with both **sequencing-based** and **imaging-based** spatial transcriptomics platforms, provided that the input data can be represented as a table of transcript coordinates with the required columns:

- `gene`
- `global_x`
- `global_y`

The package can be used in several ways. It can be called directly from **Python code**, run interactively from a **Jupyter notebook**, launched from the **command line**, or executed under **MPI** for multi-process and cluster-scale computations.

---

## Overview

The purpose of `minkiPy` is to describe the spatial organisation of each gene by computing a set of geometric descriptors across a sequence of level sets. For each gene, the package computes a **Minkowski profile** composed of:

- `W0`: area functional
- `W1`: boundary length functional
- `W2`: curvature / topology-related functional
- `Beta`: global anisotropy measure

These profiles are computed from a smoothed spatial field reconstructed from transcript coordinates.

When requested, `minkiPy` can also generate **Monte Carlo shot-noise resamples**. These resamples are used to estimate **covariance matrices** for each gene profile. This enables covariance-aware downstream comparisons based on **Gaussian 2-Wasserstein distances**.

For faster and more exploratory work, `minkiPy` can also be run **without Monte Carlo resampling**. In that case, no covariance matrices are estimated, and downstream comparisons rely on **Euclidean distances between Minkowski profiles**.

---

## Main features

- computation of gene-level Minkowski profiles from spatial transcript coordinates
- optional Monte Carlo resampling for covariance estimation
- MPI-enabled execution for large datasets
- downstream analysis of sample-level and gene-level relationships
- graph-based and embedding-based visualisation tools
- support for both exploratory and publication-oriented workflows
- example notebooks reproducing the analyses presented in the manuscript

---

## Input data format

The core computation requires a single transcript-level table with the following columns:

- `gene`: gene name
- `global_x`: x coordinate
- `global_y`: y coordinate

This is the only mandatory raw input format expected by the core package.

Any upstream preprocessing is therefore left to the user. This includes, for example:

- loading vendor-specific files
- selecting a tissue region
- filtering low-quality spots or transcripts
- removing control probes
- converting pixel units to physical coordinates
- expanding spot-level counts into transcript-level coordinates where needed

This design keeps the core package independent of any specific technology or file format.
