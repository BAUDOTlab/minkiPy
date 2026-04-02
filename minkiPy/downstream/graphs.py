from __future__ import annotations

import os
import gc
import math

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import colors as mcolors
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatter, LogLocator

import tifffile as tiff
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from typing import Optional, Sequence, Union, Tuple, List, Dict, Any

__all__ = ["plot_minkowski_profile",
           "plot_random_mc_gaussian_overlay_grid",
           "plot_dataset_graphs_from_data",
           "plot_w2_diag_vs_full_plus_euclid_distributions",
           "plot_w2_diag_vs_euclid_distributions",
           "plot_gene_graphs_from_data",
           "plot_pca_grid_by_condition",
           "plot_top_changing_genes",
           "plot_w2_abslog2fc_with_trend",
           "plot_gene_density_over_dapi"]

def plot_minkowski_profile(
    data: dict,
    *,
    dataset: Union[int, str],
    gene: Union[int, str],
    use_scaled: bool = False,
    use_scaled_sn: Optional[bool] = None,
    n_sigma: float = 1.0,
    figsize: tuple[float, float] = (11.5, 4.2),
    color: str = "red",
    lw: float = 2.0,
    band_alpha: float = 0.25,
    facecolor: str = "none",
    xlabel_fs: float = 18.0,
    ylabel_fs: float = 16.0,
    tick_fs: float = 10.0,
    title_fs: float = 13.0,
    savepath: Optional[str] = None,
    dpi: int = 200,
    show: bool = True,
):
    """
    Plot the Minkowski profile of a single gene in a single dataset.

    The function displays the four components of the Minkowski profile,
    namely ``W0``, ``W1``, ``W2`` and ``beta``, across the level-set grid
    stored in ``data["LS"]``. If Monte Carlo resamples are available, the
    profile is shown together with a symmetric uncertainty band defined as
    ``mean ± n_sigma * standard_deviation``. If no Monte Carlo resamples are
    available, only the profile itself is plotted.

    Parameters
    ----------
    data:
        Dictionary containing at least

            - ``"LS"``
            - ``"gene_names"``
            - ``"conditions"``
            - ``"gene_to_index"``
            - ``"tensor_per_sample"`` or ``"tensor_scaled"``

        Optional Monte Carlo uncertainty bands are drawn if either
        ``"sn_samples"`` or ``"sn_samples_scaled"`` is present and compatible.

    dataset:
        Dataset selector, given either as an integer index or as a dataset
        name present in ``data["conditions"]``.

    gene:
        Gene selector, given either as an integer index or as a gene name
        present in ``data["gene_to_index"]``.

    use_scaled:
        If ``True``, plot the scaled Minkowski profile stored in
        ``data["tensor_scaled"]``. Otherwise use the raw profile stored in
        ``data["tensor_per_sample"]``.

    use_scaled_sn:
        Controls whether uncertainty bands should be computed from
        ``"sn_samples_scaled"`` or ``"sn_samples"``. If ``None``, this choice
        follows ``use_scaled``.

    n_sigma:
        Width of the uncertainty band in units of standard deviation.

    figsize:
        Figure size in inches.

    color:
        Line colour of the profile and fill colour of the uncertainty band.

    lw:
        Line width of the profile curve.

    band_alpha:
        Opacity of the uncertainty band.

    facecolor:
        Axes face colour.

    xlabel_fs, ylabel_fs, tick_fs, title_fs:
        Font sizes for axis labels, tick labels and panel titles.

    savepath:
        Optional output path for saving the figure.

    dpi:
        Resolution used for saving the figure.

    show:
        If ``True``, display the figure.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.

    Notes
    -----
    This function is robust to the absence of Monte Carlo resamples. In that
    case, the profile is plotted without uncertainty bands.
    """

    def _dataset_index(data_obj: dict, dataset_obj: Union[int, str]) -> int:
        """Resolve a dataset selector given as an index or dataset name."""
        if isinstance(dataset_obj, (int, np.integer)):
            return int(dataset_obj)

        names = list(map(str, data_obj["conditions"]))
        dataset_name = str(dataset_obj)
        if dataset_name not in names:
            raise ValueError(f"Unknown dataset {dataset_name!r}. Available datasets: {names}")
        return names.index(dataset_name)

    def _gene_index(data_obj: dict, gene_obj: Union[int, str]) -> int:
        """Resolve a gene selector given as an index or gene name."""
        if isinstance(gene_obj, (int, np.integer)):
            return int(gene_obj)

        gene_to_index = data_obj["gene_to_index"]
        gene_name = str(gene_obj)
        if gene_name not in gene_to_index:
            raise ValueError(f"Unknown gene {gene_name!r}.")
        return int(gene_to_index[gene_name])

    dataset_idx = _dataset_index(data, dataset)
    gene_idx = _gene_index(data, gene)

    if "LS" not in data:
        raise KeyError("`data` must contain 'LS'.")
    if "conditions" not in data:
        raise KeyError("`data` must contain 'conditions'.")
    if "gene_names" not in data:
        raise KeyError("`data` must contain 'gene_names'.")

    tensor_key = "tensor_scaled" if use_scaled else "tensor_per_sample"
    if tensor_key not in data:
        raise KeyError(f"`data` must contain '{tensor_key}'.")

    LS = np.asarray(data["LS"], dtype=float)
    T = np.asarray(data[tensor_key])[dataset_idx, gene_idx]  # (4, L)

    if T.ndim != 2 or T.shape[0] != 4:
        raise ValueError(
            f"Expected {tensor_key}[dataset, gene] to have shape (4, L), got {T.shape}."
        )

    sn_key = "sn_samples_scaled" if (use_scaled if use_scaled_sn is None else use_scaled_sn) else "sn_samples"
    SN = data.get(sn_key, None)

    has_sn = (
        isinstance(SN, np.ndarray)
        and SN.ndim == 5
        and dataset_idx < SN.shape[0]
        and gene_idx < SN.shape[1]
    )

    if has_sn:
        sn = np.asarray(SN)[dataset_idx, gene_idx]  # (4, R, L)
        if sn.ndim != 3 or sn.shape[0] != 4 or sn.shape[2] != LS.size:
            has_sn = False

    if has_sn:
        mu = np.nanmean(sn, axis=1)         # (4, L)
        sd = np.nanstd(sn, axis=1, ddof=1)  # (4, L)
    else:
        mu = T
        sd = None

    stat_titles = [r"$W_0$", r"$W_1$", r"$W_2$", r"$\beta$"]
    dataset_name = str(data["conditions"][dataset_idx])
    gene_name = str(data["gene_names"][gene_idx])

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = axes.ravel()

    for k, ax in enumerate(axes):
        ax.set_facecolor(facecolor)

        y = mu[k]
        ax.plot(LS, y, color=color, lw=lw)

        if has_sn and sd is not None:
            ax.fill_between(
                LS,
                y - n_sigma * sd[k],
                y + n_sigma * sd[k],
                color=color,
                alpha=band_alpha,
                linewidth=0,
            )

        ax.set_title(stat_titles[k], fontsize=title_fs)
        ax.tick_params(labelsize=tick_fs)

        if k in (2, 3):
            ax.set_xlabel("Level set", fontsize=xlabel_fs)

    axes[0].set_ylabel("Profile value", fontsize=ylabel_fs)
    axes[2].set_ylabel("Profile value", fontsize=ylabel_fs)

    fig.suptitle(f"{gene_name} in {dataset_name}", fontsize=title_fs + 1)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes

def plot_random_mc_gaussian_overlay_grid(
    data: dict,
    *,
    n_rows: int = 8,
    n_cols: int = 6,
    use_scaled: bool = True,                 # True -> sn_samples_scaled ; False -> sn_samples
    bins: int = 30,
    seed: int = 0,                           # deterministic random sampling
    min_finite_draws: int = 10,              # skip panels with too few finite MC draws
    allow_repeats: bool = True,              # if False, tries to avoid repeated (gene,dataset,stat,LS)
    figsize: tuple[float, float] = (18.0, 12.0),
    label_fontsize: int = 12,
    save_pdf: Optional[str] = None,
) -> plt.Figure:
    """
    Draw a compact grid of random Monte Carlo histograms with a moment-matched
    Gaussian overlay.

    Each panel randomly selects one dataset, one gene, one Minkowski statistic
    and one level-set value (excluding the first level set), then displays the
    empirical distribution of Monte Carlo resamples together with the Gaussian
    distribution having the same mean and variance.

    This diagnostic plot requires Monte Carlo resamples to be available in the
    input `data` object. It is therefore intended for outputs generated by
    minkiPy when Monte Carlo production is enabled.

    Parameters
    ----------
    data:
        Output dictionary from `process_data()`. It must contain either
        `sn_samples_scaled` or `sn_samples`, with shape `(F, G, 4, R, L)`,
        together with `gene_names` and `conditions`.

    n_rows, n_cols:
        Number of rows and columns in the panel grid.

    use_scaled:
        If `True`, use `data["sn_samples_scaled"]`. Otherwise use
        `data["sn_samples"]`.

    bins:
        Number of histogram bins in each panel.

    seed:
        Random seed used for reproducible panel sampling.

    min_finite_draws:
        Minimum number of finite Monte Carlo draws required to render a panel.

    allow_repeats:
        If `True`, the same `(dataset, gene, statistic, level set)` combination
        may appear more than once. If `False`, the function attempts to avoid
        exact repeats.

    figsize:
        Figure size in inches.

    label_fontsize:
        Font size of the in-panel annotation.

    save_pdf:
        Optional path for PDF export.

    Returns
    -------
    fig:
        The Matplotlib figure.
    """
    if data is None:
        raise ValueError("`data` is None.")

    key = "sn_samples_scaled" if use_scaled else "sn_samples"
    SN = data.get(key, None)

    if SN is None:
        raise ValueError(
            "Monte Carlo resamples are not available in the provided `data` object. "
            f"This plot requires `data['{key}']` with shape `(F, G, 4, R, L)`. "
            "To generate this diagnostic figure, run minkiPy with Monte Carlo "
            "production enabled so that resampled profiles are written to disk "
            "and loaded by `process_data()`."
        )

    SN = np.asarray(SN)
    if SN.ndim != 5 or SN.shape[2] != 4:
        raise ValueError(f"Expected {key} shape (F, G, 4, R, L), got {SN.shape}.")

    F, G, S, R, L = SN.shape
    if S != 4:
        raise ValueError(f"Expected 4 statistics, got S={S}.")
    if L <= 1:
        raise ValueError("Need at least 2 level sets to exclude LS=0.")

    gene_names = list(map(str, data.get("gene_names", [])))
    conditions = list(map(str, data.get("conditions", [])))
    stat_names = tuple(data.get("stat_names", ("W0", "W1", "W2", "BETA")))

    LS_grid = np.asarray(data.get("LS", np.arange(L)), dtype=np.float64)
    if LS_grid.shape[0] != L:
        LS_grid = np.arange(L, dtype=np.float64)

    if len(gene_names) != G:
        raise ValueError(f"len(gene_names)={len(gene_names)} does not match G={G}.")
    if len(conditions) != F:
        raise ValueError(f"len(conditions)={len(conditions)} does not match F={F}.")

    rng = np.random.default_rng(seed)

    def _norm_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        z = (x - mu) / sigma
        return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))

    n_panels = n_rows * n_cols

    # Pre-sample tuples (dataset, gene, stat, LS), excluding LS=0.
    seen = set()
    choices: List[Tuple[int, int, int, int]] = []
    max_tries_per_panel = 200

    for _ in range(n_panels):
        if allow_repeats:
            d_idx = int(rng.integers(0, F))
            g_idx = int(rng.integers(0, G))
            s_idx = int(rng.integers(0, 4))
            l_idx = int(rng.integers(1, L))
            choices.append((d_idx, g_idx, s_idx, l_idx))
        else:
            found = False
            for _try in range(max_tries_per_panel):
                d_idx = int(rng.integers(0, F))
                g_idx = int(rng.integers(0, G))
                s_idx = int(rng.integers(0, 4))
                l_idx = int(rng.integers(1, L))
                tpl = (d_idx, g_idx, s_idx, l_idx)
                if tpl not in seen:
                    seen.add(tpl)
                    choices.append(tpl)
                    found = True
                    break
            if not found:
                d_idx = int(rng.integers(0, F))
                g_idx = int(rng.integers(0, G))
                s_idx = int(rng.integers(0, 4))
                l_idx = int(rng.integers(1, L))
                choices.append((d_idx, g_idx, s_idx, l_idx))

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=figsize,
        constrained_layout=False,
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, n_cols)
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.0, hspace=0.0)

    skipped = 0

    for p, (d_idx, g_idx, s_idx, l_idx) in enumerate(choices):
        r = p // n_cols
        c = p % n_cols
        ax = axes[r, c]

        x = SN[d_idx, g_idx, s_idx, :, l_idx]
        x = x[np.isfinite(x)]

        if x.size < min_finite_draws:
            skipped += 1
            ax.text(0.5, 0.5, "Skipped", ha="center", va="center", fontsize=label_fontsize)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue

        mu = float(np.mean(x))
        var = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
        sigma_val = float(np.sqrt(var)) if var > 0 else 0.0

        ax.hist(x, bins=bins, density=True, alpha=0.65)

        if sigma_val > 0:
            xmin, xmax = float(np.min(x)), float(np.max(x))
            pad = 0.10 * (xmax - xmin) if xmax > xmin else 1.0
            grid = np.linspace(xmin - pad, xmax + pad, 300)
            ax.plot(grid, _norm_pdf(grid, mu, sigma_val), linewidth=2.0)
        else:
            ax.axvline(mu, linewidth=2.0)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", bottom=False, left=False)
        ax.set_title("")

        ds_name = conditions[d_idx]
        g_name = gene_names[g_idx]
        stat = stat_names[s_idx] if s_idx < len(stat_names) else f"stat{s_idx}"
        ls_val = float(LS_grid[l_idx])

        label = f"{ds_name}\n{g_name}\n{stat}\nLS={ls_val:.2g}"
        ax.text(
            0.03, 0.97,
            label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=label_fontsize,
        )

        for sp in ax.spines.values():
            sp.set_linewidth(0.4)

    plt.show()

    if save_pdf is not None:
        save_pdf = os.path.expanduser(save_pdf)
        if not save_pdf.lower().endswith(".pdf"):
            save_pdf += ".pdf"
        out_dir = os.path.dirname(save_pdf)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_pdf, format="pdf", bbox_inches="tight")
        print(f"[saved] {save_pdf} | skipped panels: {skipped}/{n_panels}")

    return fig

def add_averaged_condition_datasets(
    data: dict,
    groups: Dict[str, Sequence[Union[int, str]]],
    *,
    ordered_conditions: Optional[List[str]] = None,
    field_density: str = "gene_density",
    name_fmt: str = "{label}",
    verbose: bool = True,
) -> dict:
    """
    Add condition-level averaged datasets to a *minkiPy*-style data dictionary.

    This function constructs synthetic datasets by averaging selected datasets
    in the robust-scaled feature space. Each entry in ``groups`` defines a new
    averaged dataset, built from either dataset names or integer selectors.

    For a group containing ``K`` datasets, let ``X_k(g)`` denote the scaled
    feature tensor of gene ``g`` in dataset ``k``, with shape
    ``(4 statistics, L level sets)``. The averaged profile is defined as

        μ_mean(g) = (1 / K) * Σ_k X_k(g)

    If covariance matrices are available, the covariance of the averaged
    profile is computed as

        Cov_mean(g) = (1 / K^2) * Σ_k Cov_k(g)

    where the sum includes only datasets with finite covariance matrices for
    gene ``g``. The denominator nevertheless remains ``K^2``, not the number
    of valid covariance matrices.

    If gene density fields are present, they are averaged gene-wise using
    finite values only.

    The resulting averaged dataset is appended to the arrays stored in
    ``data``, unless a dataset with the same name already exists, in which
    case it is overwritten in place.

    Parameters
    ----------
    data:
        Dictionary produced by the main preprocessing pipeline. It must contain
        at least:

            - ``'conditions'``: sequence of dataset names of length ``F``
            - ``'tensor_scaled'``: array of shape ``(F, G, 4, L)``

        The following entries are optional and, if present, are extended with
        one row per averaged dataset:

            - ``'cov_matrices'``: array of shape ``(F, G, D, D)``, with
              ``D = 4 * L``
            - ``field_density``: raw gene densities, shape ``(F, G)``
            - ``'normalized_density'``: normalised gene densities, shape
              ``(F, G)``
            - ``'n_transcripts'``: transcript counts, shape ``(F, G)``
            - ``'min_val'``, ``'max_val'``, ``'area_LS0'``: arrays of shape
              ``(F, G)``
            - ``'area_mask'``: array of shape ``(F,)``
            - ``'resolutions'``: dataset-level metadata

    groups:
        Mapping from output dataset names, or labels, to the datasets that
        should be averaged together. Each selector can be either

            - an integer index, or
            - a dataset name present in ``data['conditions']``.

        This design is intentionally generic and does not assume any specific
        naming convention. It therefore works equally well for datasets such as
        ``FSHD1``, ``Control``, ``DEL5``, ``WT``, ``Reeler``, or any other
        user-defined labels.

    ordered_conditions:
        Optional reference ordering used when resolving integer selectors.
        If provided, integers in ``groups`` are first interpreted as indices
        into ``ordered_conditions``, then mapped to dataset names, and finally
        resolved against ``data['conditions']``.

        If omitted, integer selectors are interpreted directly as indices into
        ``data['conditions']``.

        String selectors always refer directly to dataset names and are
        unaffected by this argument.

    field_density:
        Name of the field in ``data`` containing the raw gene-wise density
        array. If present, the corresponding averaged density is computed and
        stored under the same key.

    name_fmt:
        Format string used to build the output dataset name from each group
        label. It is evaluated as

            ``name_fmt.format(label=<group_label>, K=<group_size>)``

        The default, ``"{label}"``, uses the group key directly.

    verbose:
        If ``True``, print a concise summary of the groups that are appended,
        overwritten or skipped, together with any relevant warnings.

    Returns
    -------
    dict
        The input dictionary, updated in place with the new averaged datasets.

        At minimum, ``'tensor_scaled'`` and ``'conditions'`` are updated.
        Optional arrays are extended when present. The entry
        ``'cov_feature_dim'`` is also set to ``D = 4 * L``.

    Notes
    -----
    This function is dataset-agnostic. It does not assume any biological
    condition names, batch structure or experimental design beyond what is
    encoded in ``groups``. The same implementation can therefore be reused
    across distinct projects without modification.
    """
    def _v(message: str) -> None:
        """Print a message only when verbose logging is enabled."""
        if verbose:
            print(message, flush=True)

    # ------------------------------------------------------------------
    # 1. Validate required inputs and inspect core shapes
    # ------------------------------------------------------------------
    if "conditions" not in data or "tensor_scaled" not in data:
        raise KeyError("`data` must contain at least 'conditions' and 'tensor_scaled'.")

    X = np.asarray(data["tensor_scaled"])  # (F, G, 4, L)
    F, G, S, L = X.shape
    if S != 4:
        raise ValueError(f"Expected 4 statistics in 'tensor_scaled'; got {S}.")
    d = 4 * L

    have_cov = ("cov_matrices" in data) and (data["cov_matrices"] is not None)
    C = np.asarray(data["cov_matrices"]) if have_cov else None
    if have_cov and C.shape != (F, G, d, d):
        raise ValueError(f"'cov_matrices' has shape {C.shape}, expected {(F, G, d, d)}.")

    have_density = (field_density in data) and (data[field_density] is not None)
    DENS = np.asarray(data[field_density]) if have_density else None
    if have_density and DENS.shape != (F, G):
        raise ValueError(f"'{field_density}' has shape {DENS.shape}, expected {(F, G)}.")

    have_norm_density = ("normalized_density" in data) and (data["normalized_density"] is not None)
    NDENS = np.asarray(data["normalized_density"]) if have_norm_density else None
    if have_norm_density and NDENS.shape != (F, G):
        raise ValueError(f"'normalized_density' has shape {NDENS.shape}, expected {(F, G)}.")
    if have_cov:
        _v(f"[avg] Covariance matrices detected; averaged covariances will be propagated when possible.")
    else:
        _v("[avg] No covariance matrices detected; only averaged profiles and optional density fields will be updated.")
    
    _v(
        f"[avg] Starting condition-level averaging from {F} datasets, "
        f"{G} genes, 4 statistics and {L} level sets."
    )

    # ------------------------------------------------------------------
    # 2. Resolve dataset names and optional reference ordering
    # ------------------------------------------------------------------
    cond_names = list(map(str, data["conditions"]))
    name2idx = {name: i for i, name in enumerate(cond_names)}

    if ordered_conditions is not None:
        ordered_conditions = list(map(str, ordered_conditions))
        remap: Dict[int, int] = {}
        for k, name in enumerate(ordered_conditions):
            if name in name2idx:
                remap[k] = name2idx[name]

        missing_names = [name for name in ordered_conditions if name not in name2idx]
        if missing_names:
            _v(
                "[avg][warning] The following entries in 'ordered_conditions' "
                f"were not found in data['conditions'] and will be ignored: {missing_names}"
            )
    else:
        remap = None

    # ------------------------------------------------------------------
    # 3. Helper to resolve selectors into dataset indices
    # ------------------------------------------------------------------
    def _resolve(sel: Sequence[Union[int, str]]) -> List[int]:
        """
        Resolve a sequence of dataset selectors into indices of data['conditions'].

        Selectors may be dataset names or integer indices. Duplicate selections
        are removed while preserving order.
        """
        out: List[int] = []

        for x in sel:
            if isinstance(x, (int, np.integer)):
                idx_int = int(x)
                if remap is not None:
                    assert ordered_conditions is not None  # for type checking
                    if idx_int < 0 or idx_int >= len(ordered_conditions):
                        raise IndexError(
                            f"Index {idx_int} out of range for ordered_conditions "
                            f"(len={len(ordered_conditions)})."
                        )
                    name = ordered_conditions[idx_int]
                    if name not in name2idx:
                        # Missing names were already reported above when verbose=True.
                        continue
                    out.append(name2idx[name])
                else:
                    if idx_int < 0 or idx_int >= F:
                        raise IndexError(
                            f"Index {idx_int} out of range for data['conditions'] (len={F})."
                        )
                    out.append(idx_int)
            else:
                name = str(x)
                if name not in name2idx:
                    raise KeyError(f"Dataset name {name!r} not found in data['conditions'].")
                out.append(name2idx[name])

        seen = set()
        resolved: List[int] = []
        for idx in out:
            if idx not in seen:
                seen.add(idx)
                resolved.append(idx)
        return resolved

    # ------------------------------------------------------------------
    # 4. Small utility helpers for extending arrays and metadata
    # ------------------------------------------------------------------
    def _append_row(arr: np.ndarray, row: np.ndarray) -> np.ndarray:
        """Append a single row along axis 0."""
        return np.concatenate([arr, row], axis=0)

    def _ensure_resolutions() -> None:
        """
        Ensure that data['resolutions'] exists and is aligned with conditions.

        If missing, placeholder values are created for existing datasets.
        """
        if "resolutions" not in data or data["resolutions"] is None:
            data["resolutions"] = ["?"] * len(cond_names)

    # ------------------------------------------------------------------
    # 5. Build each requested averaged dataset
    # ------------------------------------------------------------------
    n_appended = 0
    n_overwritten = 0
    n_skipped = 0

    for label, sel in groups.items():
        idxs = _resolve(sel)
        K = len(idxs)

        if K == 0:
            n_skipped += 1
            _v(f"[avg] Skipping '{label}': selection resolved to no datasets.")
            continue

        source_names = [cond_names[i] for i in idxs]

        # Mean scaled profile per gene: shape (G, 4, L).
        Xi = X[idxs]
        mu_gene = np.nanmean(Xi, axis=0)

        # Covariance of the averaged profile, if available.
        if have_cov:
            assert C is not None
            Ci = C[idxs]  # (K, G, d, d)
            Cav = np.full((G, d, d), np.nan, dtype=C.dtype)
            finite = np.isfinite(Ci).all(axis=(2, 3))  # (K, G)

            for g in range(G):
                valid_k = np.where(finite[:, g])[0]
                if valid_k.size:
                    Cav[g] = (
                        np.sum(Ci[valid_k, g], axis=0) / (K * K)
                    ).astype(C.dtype, copy=False)
        else:
            Cav = None

        # Mean raw densities, if available.
        if have_density:
            assert DENS is not None
            Di = DENS[idxs]
            with np.errstate(invalid="ignore"):
                Dav = np.nanmean(Di, axis=0).astype(DENS.dtype, copy=False)
        else:
            Dav = None

        # Mean normalised densities, if available.
        if have_norm_density:
            assert NDENS is not None
            NDi = NDENS[idxs]
            with np.errstate(invalid="ignore"):
                NDav = np.nanmean(NDi, axis=0).astype(NDENS.dtype, copy=False)
        else:
            NDav = None

        new_name = name_fmt.format(label=label, K=K)

        # --------------------------------------------------------------
        # 6. Overwrite an existing averaged dataset or append a new one
        # --------------------------------------------------------------
        if new_name in name2idx:
            tgt = name2idx[new_name]
            X[tgt] = mu_gene

            if have_cov and Cav is not None:
                C[tgt] = Cav
                data["cov_matrices"] = C
            if have_density and Dav is not None:
                DENS[tgt] = Dav
                data[field_density] = DENS
            if have_norm_density and NDav is not None:
                NDENS[tgt] = NDav
                data["normalized_density"] = NDENS

            _ensure_resolutions()
            data["resolutions"][tgt] = "avg"
            n_overwritten += 1

            if have_cov and Cav is not None:
                ok_cov = int(np.isfinite(Cav).all(axis=(1, 2)).sum())
                _v(
                    f"[avg] Overwrote '{new_name}' from {K} datasets "
                    f"({', '.join(source_names)}); finite covariances for {ok_cov}/{G} genes."
                )
            else:
                _v(
                    f"[avg] Overwrote '{new_name}' from {K} datasets "
                    f"({', '.join(source_names)})."
                )

        else:
            X = _append_row(X, mu_gene[np.newaxis, ...])

            if have_cov and Cav is not None:
                C = _append_row(C, Cav[np.newaxis, ...])  # type: ignore[arg-type]
                data["cov_matrices"] = C
            if have_density and Dav is not None:
                DENS = _append_row(DENS, Dav[np.newaxis, ...])  # type: ignore[arg-type]
                data[field_density] = DENS
            if have_norm_density and NDav is not None:
                NDENS = _append_row(NDENS, NDav[np.newaxis, ...])  # type: ignore[arg-type]
                data["normalized_density"] = NDENS

            cond_names.append(new_name)
            data["conditions"] = cond_names

            _ensure_resolutions()
            data["resolutions"].append("avg")

            # Extend ancillary arrays with neutral or missing values.
            if "n_transcripts" in data and isinstance(data["n_transcripts"], np.ndarray):
                nt = data["n_transcripts"]
                data["n_transcripts"] = _append_row(
                    nt,
                    np.full((1, nt.shape[1]), -1, dtype=nt.dtype),
                )

            for key in ("min_val", "max_val", "area_LS0"):
                if key in data and isinstance(data[key], np.ndarray):
                    arr = data[key]
                    data[key] = _append_row(
                        arr,
                        np.full((1, arr.shape[1]), np.nan, dtype=arr.dtype),
                    )

            if "area_mask" in data and isinstance(data["area_mask"], np.ndarray):
                am = data["area_mask"]
                data["area_mask"] = _append_row(
                    am,
                    np.full((1,), np.nan, dtype=am.dtype),
                )

            name2idx[new_name] = len(cond_names) - 1
            n_appended += 1

            if have_cov and Cav is not None:
                ok_cov = int(np.isfinite(Cav).all(axis=(1, 2)).sum())
                _v(
                    f"[avg] Appended '{new_name}' from {K} datasets "
                    f"({', '.join(source_names)}); finite covariances for {ok_cov}/{G} genes."
                )
            else:
                _v(
                    f"[avg] Appended '{new_name}' from {K} datasets "
                    f"({', '.join(source_names)})."
                )

        # Persist updated core arrays after each group.
        data["tensor_scaled"] = X
        if have_cov:
            data["cov_matrices"] = C
        if have_density:
            data[field_density] = DENS
        if have_norm_density:
            data["normalized_density"] = NDENS

    # ------------------------------------------------------------------
    # 7. Final bookkeeping
    # ------------------------------------------------------------------
    data["cov_feature_dim"] = d

    _v(
        f"[avg] Finished condition-level averaging: "
        f"{n_appended} appended, {n_overwritten} overwritten, {n_skipped} skipped."
    )

    return data

def plot_dataset_graphs_from_data(
    data: dict,
    *,
    ordered_conditions: list[str],
    groups: dict[str, list[int]],
    group_colors: dict[str, str] | None = None,
    avg_group_colors: dict[str, str] | None = None,
    avg_suffix: str = "_avg",
    avg_darken_factor: float = 0.7,
    annotate_edges: bool = False,
    # Display options
    show_others: bool = True,
    others_group_name: str = "OTHER",
    label_fontsize: float = 12.0,
    edge_label_fontsize: float = 9.0,
    legend_loc: str | None = None,
    legend_loc_w2: str = "best",
    legend_loc_pca: str = "best",
    legend_loc_umap: str = "best",
    legend_loc_pca_density: str = "best",
    marker_size: float = 140.0,
    # Figure titles
    title_w2: str = "Sample-level MDS based on distance matrix",
    title_pca: str = "Sample-level PCA based on Minkowski profiles",
    title_umap: str = "Sample-level UMAP based on Minkowski profiles",
    title_pca_density: str = "Sample-level PCA based on normalised densities",
    # Stochastic control
    random_seed: int = 0,
    # UMAP parameters
    umap_n_neighbors: int = 8,
    umap_min_dist: float = 0.05,
    umap_repulsion_strength: float = 1.0,
    umap_parallel: bool = True,
    # Optional density-only PCA
    include_density_pca: bool = False,
    density_field: str = "normalized_density",
    # Optional PDF outputs
    out_pdf_w2: str | None = None,
    out_pdf_pca: str | None = None,
    out_pdf_umap: str | None = None,
    out_pdf_pca_density: str | None = None,
):
    """
    Visualise sample-level relationships using a precomputed sample-distance
    matrix together with low-dimensional embeddings derived from Minkowski
    profiles, and optionally from normalised densities alone.
    """
    from matplotlib.lines import Line2D
    from matplotlib import colors as mcolors
    from sklearn.decomposition import PCA

    def _darken(color: str, factor: float = 0.7) -> str:
        """Return a darker version of a Matplotlib-compatible colour."""
        r, g, b = mcolors.to_rgb(color)
        return mcolors.to_hex((r * factor, g * factor, b * factor))

    def _impute_nan_columns(X: np.ndarray) -> np.ndarray:
        """
        Replace missing values by the corresponding column mean.

        If an entire column is missing, the imputed value is set to zero.
        """
        X = np.where(np.isfinite(X), X, np.nan)
        if np.isnan(X).any():
            col_mean = np.nanmean(X, axis=0)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
        return X

    if "tensor_scaled" not in data or "conditions" not in data:
        raise KeyError("data must contain 'tensor_scaled' and 'conditions'.")
    if "sample_distance_summary" not in data:
        raise KeyError(
            "Run compute_sample_distances(...) first to populate "
            "'sample_distance_summary'."
        )

    X_scaled = np.asarray(data["tensor_scaled"])
    labels_data = list(map(str, data["conditions"]))
    F, G, S, L = X_scaled.shape
    if S != 4:
        raise ValueError("tensor_scaled must have shape (F, G, 4, L).")

    Dsum = data["sample_distance_summary"]
    D = np.asarray(Dsum["matrix"], dtype=np.float64)
    counts = np.asarray(Dsum.get("counts", np.zeros((F, F), dtype=int)))

    if D.shape != (F, F):
        raise ValueError(
            f"sample_distance_summary['matrix'] has shape {D.shape}, "
            f"expected {(F, F)}."
        )
    if counts.shape != (F, F):
        raise ValueError(
            f"sample_distance_summary['counts'] has shape {counts.shape}, "
            f"expected {(F, F)}."
        )

    labs = labels_data
    group_of_name = {name: others_group_name for name in labs}

    for group_name, idxs in groups.items():
        for idx in idxs:
            name = ordered_conditions[idx]
            if name in group_of_name:
                group_of_name[name] = group_name

    avg_group_colors = {} if avg_group_colors is None else dict(avg_group_colors)

    is_avg = np.zeros(F, dtype=bool)
    if avg_suffix:
        for k, name in enumerate(labs):
            if name.endswith(avg_suffix):
                base = name[:-len(avg_suffix)]
                if (group_colors and base in group_colors) or (base in groups):
                    group_of_name[name] = base
                is_avg[k] = True

    if group_colors is None:
        group_colors = {others_group_name: "gray"}

    def color_for(idx: int) -> str:
        name = labs[idx]
        group = group_of_name.get(name, others_group_name)
        base_color = group_colors.get(group, "gray")

        if is_avg[idx]:
            if group in avg_group_colors:
                return avg_group_colors[group]
            return _darken(base_color, avg_darken_factor)

        return base_color

    node_colors_full = [color_for(i) for i in range(F)]

    keep_mask = np.array(
        [
            (group_of_name.get(name, others_group_name) != others_group_name) or show_others
            for name in labs
        ],
        dtype=bool,
    )

    if not np.any(keep_mask):
        raise ValueError(
            "All samples were filtered out "
            "(show_others=False and no grouped sample remains)."
        )

    labs_kept = [name for name, keep in zip(labs, keep_mask) if keep]
    is_avg_kept = is_avg[keep_mask]
    node_colors = [c for c, keep in zip(node_colors_full, keep_mask) if keep]

    D = D[np.ix_(keep_mask, keep_mask)]
    counts = counts[np.ix_(keep_mask, keep_mask)]
    X_sub = X_scaled[keep_mask]
    F_kept = X_sub.shape[0]

    X_flat = X_sub.reshape(F_kept, -1).astype(float, copy=False)
    X_flat = _impute_nan_columns(X_flat)

    pca = PCA(n_components=min(F_kept - 1, 10), svd_solver="full")
    X_pca = pca.fit_transform(X_flat)
    pca_evr = pca.explained_variance_ratio_

    if X_pca.shape[1] < 2:
        X_pca = np.pad(X_pca, ((0, 0), (0, 2 - X_pca.shape[1])), mode="constant")
        pca_evr = np.pad(pca_evr, (0, 2 - pca_evr.size), mode="constant")

    X_pca_density = None
    pca_density_evr = None
    fig_pca_density = None
    ax_pca_density = None

    if include_density_pca:
        if density_field not in data:
            raise KeyError(
                f"include_density_pca=True requires data['{density_field}']."
            )

        X_density = np.asarray(data[density_field], dtype=float)
        if X_density.shape != (F, G):
            raise ValueError(
                f"{density_field!r} has shape {X_density.shape}; expected {(F, G)}."
            )

        X_density_sub = X_density[keep_mask].astype(float, copy=False)
        X_density_sub = _impute_nan_columns(X_density_sub)

        pca_density = PCA(n_components=min(F_kept - 1, 10), svd_solver="full")
        X_pca_density = pca_density.fit_transform(X_density_sub)
        pca_density_evr = pca_density.explained_variance_ratio_

        if X_pca_density.shape[1] < 2:
            X_pca_density = np.pad(
                X_pca_density,
                ((0, 0), (0, 2 - X_pca_density.shape[1])),
                mode="constant",
            )
            pca_density_evr = np.pad(
                pca_density_evr,
                (0, 2 - pca_density_evr.size),
                mode="constant",
            )

    X_umap = None
    umap_err = None

    try:
        import umap

        um = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            repulsion_strength=umap_repulsion_strength,
            metric="euclidean",
            n_epochs=None,
            init="spectral",
            verbose=False,
            low_memory=True,
            transform_seed=random_seed,
            random_state=random_seed,
            n_jobs=-1 if umap_parallel else 1,
        )
        X_umap = um.fit_transform(X_flat)
    except Exception as exc:
        umap_err = exc
        X_umap = None

    try:
        from adjustText import adjust_text
        _have_adjust = True
    except Exception:
        _have_adjust = False

    def _deoverlap_labels(ax, texts, scatter_artist=None) -> None:
        if not _have_adjust or not texts:
            return

        kwargs = dict(
            only_move={"points": "", "texts": "xy", "objects": "xy"},
            expand_points=(1.6, 2.0),
            expand_text=(1.5, 2.2),
            force_points=0.5,
            force_text=2.0,
            autoalign="xy",
            lim=500,
            precision=0.001,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="gray", alpha=0.6),
        )

        if scatter_artist is not None:
            adjust_text(texts, ax=ax, add_objects=[scatter_artist], **kwargs)
        else:
            adjust_text(texts, ax=ax, **kwargs)

    def _classical_mds_with_evr(Dmat: np.ndarray, n_components: int = 2):
        Dmat = Dmat.copy()
        mask = np.isfinite(Dmat) & ~np.eye(Dmat.shape[0], dtype=bool)
        fill = np.nanmean(Dmat[mask]) if np.any(mask) else 0.0
        Dmat[~np.isfinite(Dmat)] = fill
        np.fill_diagonal(Dmat, 0.0)

        n = Dmat.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (Dmat ** 2) @ J

        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]

        pos = vals > 1e-12
        vals_pos = vals[pos]
        vecs_pos = vecs[:, pos]

        if vecs_pos.size == 0:
            coords = np.zeros((n, n_components))
            evr = np.zeros((n_components,))
            return coords, evr

        total = np.sum(vals_pos) if np.sum(vals_pos) > 0 else 1.0
        evr_full = vals_pos / total
        k = min(n_components, vecs_pos.shape[1])
        coords = vecs_pos[:, :k] * np.sqrt(vals_pos[:k][np.newaxis, :])
        evr = evr_full[:k]

        if coords.shape[1] < n_components:
            coords = np.pad(
                coords,
                ((0, 0), (0, n_components - coords.shape[1])),
                mode="constant",
            )
            evr = np.pad(evr, (0, n_components - evr.size), mode="constant")

        return coords, evr

    coords, mds_evr = _classical_mds_with_evr(D, n_components=2)

    denom = np.max(np.linalg.norm(coords, axis=1)) + 1e-12
    coords = coords / denom

    loc_w2 = legend_loc if legend_loc is not None else legend_loc_w2
    loc_pca = legend_loc if legend_loc is not None else legend_loc_pca
    loc_umap = legend_loc if legend_loc is not None else legend_loc_umap
    loc_pca_density = legend_loc if legend_loc is not None else legend_loc_pca_density

    present_groups = sorted(set(group_of_name.get(name, others_group_name) for name in labs_kept))
    legend_handles = []
    for group in present_groups:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                label=group,
                markerfacecolor=group_colors.get(group, "gray"),
                markeredgecolor="white",
                markersize=9,
            )
        )

    fig_w2, ax_w2 = plt.subplots(1, 1, figsize=(6.8, 6.2))

    for i in range(F_kept):
        for j in range(i + 1, F_kept):
            xi, yi = coords[i]
            xj, yj = coords[j]

            seg_len = np.hypot(xi - xj, yi - yj) + 1e-12
            lw = 2.6 / (1.0 + 3.0 * seg_len)

            ax_w2.plot(
                [xi, xj],
                [yi, yj],
                lw=lw,
                alpha=0.0,
                color="black",
                zorder=1,
            )

            if annotate_edges and np.isfinite(D[i, j]):
                xm, ym = (xi + xj) / 2, (yi + yj) / 2
                label_ij = f"{D[i, j]:.2f}\n(n={counts[i, j]})"
                ax_w2.text(
                    xm,
                    ym,
                    label_ij,
                    ha="center",
                    va="center",
                    fontsize=edge_label_fontsize,
                    alpha=0.9,
                )

    sc_w2 = ax_w2.scatter(
        coords[:, 0],
        coords[:, 1],
        s=marker_size,
        c=node_colors,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )

    texts = []
    for k, name in enumerate(labs_kept):
        texts.append(
            ax_w2.text(
                coords[k, 0],
                coords[k, 1],
                f"  {name}",
                va="center",
                ha="left",
                fontsize=label_fontsize,
                fontweight=("bold" if is_avg_kept[k] else "normal"),
                zorder=4,
            )
        )
    _deoverlap_labels(ax_w2, texts, scatter_artist=sc_w2)

    ax_w2.set_aspect("equal", adjustable="datalim")
    ax_w2.set_xticks([])
    ax_w2.set_yticks([])
    ax_w2.grid(True, alpha=0.2)
    ax_w2.legend(handles=legend_handles, loc=loc_w2, frameon=True, fontsize=15)

    mds1_pct = 100.0 * (mds_evr[0] if mds_evr.size > 0 else 0.0)
    mds2_pct = 100.0 * (mds_evr[1] if mds_evr.size > 1 else 0.0)
    ax_w2.set_xlabel(f"MDS 1 (EVR = {mds1_pct:.1f}%)", fontsize=17)
    ax_w2.set_ylabel(f"MDS 2 (EVR = {mds2_pct:.1f}%)", fontsize=17)
    ax_w2.set_title(title_w2, fontsize=17)
    fig_w2.tight_layout()

    if out_pdf_w2 is not None:
        path = out_pdf_w2 if out_pdf_w2.lower().endswith(".pdf") else (out_pdf_w2 + ".pdf")
        fig_w2.savefig(path, bbox_inches="tight", pad_inches=0.5)

    fig_pca, ax_pca = plt.subplots(1, 1, figsize=(6.8, 6.2))

    sc_pca = ax_pca.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        s=marker_size,
        c=node_colors,
        edgecolors="white",
        linewidths=1.2,
    )

    texts = []
    for i, name in enumerate(labs_kept):
        texts.append(
            ax_pca.text(
                X_pca[i, 0],
                X_pca[i, 1],
                f"  {name}",
                va="center",
                ha="left",
                fontsize=label_fontsize,
                fontweight=("bold" if is_avg_kept[i] else "normal"),
            )
        )
    _deoverlap_labels(ax_pca, texts, scatter_artist=sc_pca)

    evr1 = 100.0 * (pca_evr[0] if pca_evr.size > 0 else 0.0)
    evr2 = 100.0 * (pca_evr[1] if pca_evr.size > 1 else 0.0)
    ax_pca.set_xlabel(f"PC1 (EVR = {evr1:.1f}%)", fontsize=17)
    ax_pca.set_ylabel(f"PC2 (EVR = {evr2:.1f}%)", fontsize=17)
    ax_pca.legend(handles=legend_handles, loc=loc_pca, frameon=True, fontsize=15)
    ax_pca.set_title(title_pca, fontsize=17)
    fig_pca.tight_layout()

    if out_pdf_pca is not None:
        path = out_pdf_pca if out_pdf_pca.lower().endswith(".pdf") else (out_pdf_pca + ".pdf")
        fig_pca.savefig(path, bbox_inches="tight", pad_inches=0.5)

    fig_umap, ax_umap = plt.subplots(1, 1, figsize=(6.8, 6.2))

    if X_umap is not None and X_umap.shape[1] >= 2:
        sc_um = ax_umap.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            s=marker_size,
            c=node_colors,
            edgecolors="white",
            linewidths=1.2,
        )

        texts = []
        for i, name in enumerate(labs_kept):
            texts.append(
                ax_umap.text(
                    X_umap[i, 0],
                    X_umap[i, 1],
                    f"  {name}",
                    va="center",
                    ha="left",
                    fontsize=label_fontsize,
                    fontweight=("bold" if is_avg_kept[i] else "normal"),
                )
            )
        _deoverlap_labels(ax_umap, texts, scatter_artist=sc_um)

        ax_umap.set_title(title_umap, fontsize=17)
        ax_umap.legend(handles=legend_handles, loc=loc_umap, frameon=True, fontsize=15)
        ax_umap.set_xlabel("UMAP 1", fontsize=17)
        ax_umap.set_ylabel("UMAP 2", fontsize=17)
    else:
        ax_umap.text(
            0.5,
            0.5,
            "UMAP not available",
            ha="center",
            va="center",
            fontsize=12,
        )
        if umap_err is not None:
            ax_umap.text(
                0.5,
                0.42,
                f"{type(umap_err).__name__}: {umap_err}",
                ha="center",
                va="center",
                fontsize=8,
            )
        ax_umap.set_xticks([])
        ax_umap.set_yticks([])
        ax_umap.set_title(title_umap, fontsize=17)

    fig_umap.tight_layout()

    if out_pdf_umap is not None:
        path = out_pdf_umap if out_pdf_umap.lower().endswith(".pdf") else (out_pdf_umap + ".pdf")
        fig_umap.savefig(path, bbox_inches="tight", pad_inches=0.5)

    if include_density_pca:
        fig_pca_density, ax_pca_density = plt.subplots(1, 1, figsize=(6.8, 6.2))

        sc_den = ax_pca_density.scatter(
            X_pca_density[:, 0],
            X_pca_density[:, 1],
            s=marker_size,
            c=node_colors,
            edgecolors="white",
            linewidths=1.2,
        )

        texts = []
        for i, name in enumerate(labs_kept):
            texts.append(
                ax_pca_density.text(
                    X_pca_density[i, 0],
                    X_pca_density[i, 1],
                    f"  {name}",
                    va="center",
                    ha="left",
                    fontsize=label_fontsize,
                    fontweight=("bold" if is_avg_kept[i] else "normal"),
                )
            )
        _deoverlap_labels(ax_pca_density, texts, scatter_artist=sc_den)

        evr1_den = 100.0 * (pca_density_evr[0] if pca_density_evr.size > 0 else 0.0)
        evr2_den = 100.0 * (pca_density_evr[1] if pca_density_evr.size > 1 else 0.0)

        ax_pca_density.set_xlabel(f"PC1 (EVR = {evr1_den:.1f}%)", fontsize=17)
        ax_pca_density.set_ylabel(f"PC2 (EVR = {evr2_den:.1f}%)", fontsize=17)
        ax_pca_density.legend(handles=legend_handles, loc=loc_pca_density, frameon=True, fontsize=15)
        ax_pca_density.set_title(title_pca_density, fontsize=17)
        fig_pca_density.tight_layout()

        if out_pdf_pca_density is not None:
            path = (
                out_pdf_pca_density
                if out_pdf_pca_density.lower().endswith(".pdf")
                else (out_pdf_pca_density + ".pdf")
            )
            fig_pca_density.savefig(path, bbox_inches="tight", pad_inches=0.5)

    return {
        "fig_w2": fig_w2,
        "ax_w2": ax_w2,
        "fig_pca": fig_pca,
        "ax_pca": ax_pca,
        "fig_umap": fig_umap,
        "ax_umap": ax_umap,
        "fig_pca_density": fig_pca_density,
        "ax_pca_density": ax_pca_density,
        "mds_evr": mds_evr,
        "pca_evr": pca_evr,
        "pca_density_evr": pca_density_evr,
        "labels_kept": labs_kept,
        "is_avg_kept": is_avg_kept,
    }

def plot_w2_diag_vs_full_plus_euclid_distributions(
    data: dict,
    run_full: str,
    run_diag: str,
    *,
    bins: int = 100,
    density: bool = True,
    figsize: tuple = (10, 8),
    logy: bool = False,
    max_rel_pct: float | None = None,
    # axis limits (applied to both rows if provided)
    xlim_abs: tuple[float, float] | None = None,
    ylim_abs: tuple[float, float] | None = None,
    xlim_rel: tuple[float, float] | None = None,
    ylim_rel: tuple[float, float] | None = None,
    # optional output path (PDF, PNG, etc.)
    save_path: str | None = None,
    save_dpi: int = 300,
):
    """
    Compare (i) full-cov vs diagonal W2 and (ii) full-cov W2 vs Euclidean (means-only).

    Requires two runs in `data["w2_allpairs_runs"]`:
      - run_full: produced with diag_only=False  -> provides payload["D"]
      - run_diag: produced with diag_only=True and add_euclidean=True -> provides payload["D"] and payload["E"]

    The figure has 2 rows x 2 columns:
      Row 1: full - diag (absolute and relative)
      Row 2: full - euclid (absolute and relative)
    """
    runs = data.get("w2_allpairs_runs", {})
    if run_full not in runs:
        raise KeyError(f"Full-covariance run '{run_full}' not found in data['w2_allpairs_runs'].")
    if run_diag not in runs:
        raise KeyError(f"Diagonal run '{run_diag}' not found in data['w2_allpairs_runs'].")

    payload_full = runs[run_full]
    payload_diag = runs[run_diag]

    D_full = np.asarray(payload_full["D"], dtype=np.float64)
    D_diag = np.asarray(payload_diag["D"], dtype=np.float64)

    mask_full = np.asarray(payload_full["counts_mask"], dtype=bool)
    mask_diag = np.asarray(payload_diag["counts_mask"], dtype=bool)

    if D_full.shape != D_diag.shape:
        raise ValueError(f"Incompatible shapes: D_full {D_full.shape} vs D_diag {D_diag.shape}.")
    if mask_full.shape != mask_diag.shape:
        raise ValueError(f"Incompatible mask shapes: {mask_full.shape} vs {mask_diag.shape}.")

    # --- Euclidean distances (means only) ---
    if "E" not in payload_diag or payload_diag["E"] is None:
        raise KeyError(
            f"Run '{run_diag}' does not contain payload['E'].\n"
            "Re-run attach_w2_all_gene_pairs(..., diag_only=True, add_euclidean=True)."
        )
    E_eucl = np.asarray(payload_diag["E"], dtype=np.float64)
    if E_eucl.shape != D_full.shape:
        raise ValueError(f"Incompatible shapes: E {E_eucl.shape} vs D_full {D_full.shape}.")

    # ------------------------------------------------------------------
    # Extract common valid entries for each comparison
    # ------------------------------------------------------------------
    valid_fd = (
        mask_full & mask_diag
        & np.isfinite(D_full) & np.isfinite(D_diag)
    )
    if not np.any(valid_fd):
        raise RuntimeError("No common valid entries between full and diagonal runs.")

    W_full_fd = D_full[valid_fd]
    W_diag    = D_diag[valid_fd]
    diff_abs_fd = W_full_fd - W_diag
    denom_fd = np.maximum(W_full_fd, 1e-12)
    diff_rel_fd = 100.0 * diff_abs_fd / denom_fd

    valid_fe = (
        mask_full
        & np.isfinite(D_full) & np.isfinite(E_eucl)
    )
    if not np.any(valid_fe):
        raise RuntimeError("No common valid entries between full and Euclidean distances.")

    W_full_fe = D_full[valid_fe]
    W_eucl    = E_eucl[valid_fe]
    diff_abs_fe = W_full_fe - W_eucl
    denom_fe = np.maximum(W_full_fe, 1e-12)
    diff_rel_fe = 100.0 * diff_abs_fe / denom_fe

    if max_rel_pct is not None:
        keep_fd = np.abs(diff_rel_fd) <= max_rel_pct
        diff_rel_fd = diff_rel_fd[keep_fd]
        keep_fe = np.abs(diff_rel_fe) <= max_rel_pct
        diff_rel_fe = diff_rel_fe[keep_fe]

    # ------------------------------------------------------------------
    # Plotting (2 rows x 2 cols)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    (ax00, ax01), (ax10, ax11) = axes

    # Medians
    med_abs_fd = np.nanmedian(diff_abs_fd)
    med_rel_fd = np.nanmedian(diff_rel_fd)
    med_abs_fe = np.nanmedian(diff_abs_fe)
    med_rel_fe = np.nanmedian(diff_rel_fe)

    # Row 1: full - diag
    ax00.hist(diff_abs_fd, bins=bins, density=density, alpha=0.8)
    ax00.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax00.axvline(med_abs_fd, color="k", linestyle="-", linewidth=1)
    ax00.set_xlabel(r"$W^{\mathrm{full}} - W^{\mathrm{diag}}$", fontsize=14)
    ax00.set_ylabel("Density" if density else "Counts", fontsize=14)
    ax00.set_title("Full vs diagonal — absolute", fontsize=14)

    ax01.hist(diff_rel_fd, bins=bins, density=density, alpha=0.8)
    ax01.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax01.axvline(med_rel_fd, color="k", linestyle="-", linewidth=1)
    ax01.set_xlabel(
        r"$100 \times \dfrac{W^{\mathrm{full}} - W^{\mathrm{diag}}}{W^{\mathrm{full}}}$ (%)",
        fontsize=14,
    )
    ax01.set_ylabel("Density" if density else "Counts", fontsize=14)
    ax01.set_title("Full vs diagonal — relative (%)", fontsize=14)

    # Row 2: full - euclid
    ax10.hist(diff_abs_fe, bins=bins, density=density, alpha=0.8)
    ax10.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax10.axvline(med_abs_fe, color="k", linestyle="-", linewidth=1)
    ax10.set_xlabel(r"$W^{\mathrm{full}} - \|\mu_1-\mu_2\|_2$", fontsize=14)
    ax10.set_ylabel("Density" if density else "Counts", fontsize=14)
    ax10.set_title("Full vs Euclidean — absolute", fontsize=14)

    ax11.hist(diff_rel_fe, bins=bins, density=density, alpha=0.8)
    ax11.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax11.axvline(med_rel_fe, color="k", linestyle="-", linewidth=1)
    ax11.set_xlabel(
        r"$100 \times \dfrac{W^{\mathrm{full}} - \|\mu_1-\mu_2\|_2}{W^{\mathrm{full}}}$ (%)",
        fontsize=14,
    )
    ax11.set_ylabel("Density" if density else "Counts", fontsize=14)
    ax11.set_title("Full vs Euclidean — relative (%)", fontsize=14)

    # Optional limits (applied uniformly if provided)
    if xlim_abs is not None:
        ax00.set_xlim(*xlim_abs); ax10.set_xlim(*xlim_abs)
    if ylim_abs is not None:
        ax00.set_ylim(*ylim_abs); ax10.set_ylim(*ylim_abs)
    if xlim_rel is not None:
        ax01.set_xlim(*xlim_rel); ax11.set_xlim(*xlim_rel)
    if ylim_rel is not None:
        ax01.set_ylim(*ylim_rel); ax11.set_ylim(*ylim_rel)

    if logy:
        for ax in (ax00, ax01, ax10, ax11):
            ax.set_yscale("log")

    fig.tight_layout()
    plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")

    return fig, axes

def plot_w2_diag_vs_euclid_distributions(
    data: dict,
    run_diag: str,
    *,
    bins: int = 100,
    density: bool = True,
    figsize: tuple = (10, 4),
    logy: bool = False,
    max_rel_pct: float | None = None,
    # axis limits
    xlim_abs: tuple[float, float] | None = None,
    ylim_abs: tuple[float, float] | None = None,
    xlim_rel: tuple[float, float] | None = None,
    ylim_rel: tuple[float, float] | None = None,
    # optional output path
    save_path: str | None = None,
    save_dpi: int = 300,
):
    """
    Compare diagonal Gaussian 2-Wasserstein distances with Euclidean distances
    between mean feature vectors.

    This function expects a run stored in ``data["w2_allpairs_runs"]``,
    produced with

        attach_w2_all_gene_pairs(..., diag_only=True, add_euclidean=True)

    The resulting figure contains two panels:

      - left:  W_diag - ||mu_1 - mu_2||_2   (absolute difference)
      - right: 100 * (W_diag - ||mu_1 - mu_2||_2) / W_diag   (relative difference)

    Parameters
    ----------
    data:
        Data dictionary containing ``"w2_allpairs_runs"``.

    run_diag:
        Name of the diagonal-W2 run to use. This run must contain both
        ``payload["D"]`` and ``payload["E"]``.

    bins:
        Number of histogram bins.

    density:
        If ``True``, plot probability densities. Otherwise plot counts.

    figsize:
        Figure size passed to Matplotlib.

    logy:
        If ``True``, use a logarithmic y-axis on both panels.

    max_rel_pct:
        Optional symmetric clipping threshold for relative differences in
        percent. If provided, only entries satisfying
        ``abs(relative_difference) <= max_rel_pct`` are kept.

    xlim_abs, ylim_abs:
        Optional axis limits for the absolute-difference panel.

    xlim_rel, ylim_rel:
        Optional axis limits for the relative-difference panel.

    save_path:
        Optional output path for saving the figure.

    save_dpi:
        Resolution used when saving the figure.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """
    runs = data.get("w2_allpairs_runs", {})
    if run_diag not in runs:
        raise KeyError(f"Diagonal run {run_diag!r} not found in data['w2_allpairs_runs'].")

    payload_diag = runs[run_diag]

    if "D" not in payload_diag:
        raise KeyError(f"Run {run_diag!r} does not contain payload['D'].")
    if "E" not in payload_diag or payload_diag["E"] is None:
        raise KeyError(
            f"Run {run_diag!r} does not contain payload['E'].\n"
            "Re-run attach_w2_all_gene_pairs(..., diag_only=True, add_euclidean=True)."
        )

    D_diag = np.asarray(payload_diag["D"], dtype=np.float64)
    E_eucl = np.asarray(payload_diag["E"], dtype=np.float64)
    mask_diag = np.asarray(payload_diag["counts_mask"], dtype=bool)

    if D_diag.shape != E_eucl.shape:
        raise ValueError(f"Incompatible shapes: D {D_diag.shape} vs E {E_eucl.shape}.")
    if mask_diag.shape != D_diag.shape:
        raise ValueError(f"Incompatible mask shape: {mask_diag.shape} vs {D_diag.shape}.")

    # ------------------------------------------------------------------
    # Extract common valid entries
    # ------------------------------------------------------------------
    valid = (
        mask_diag
        & np.isfinite(D_diag)
        & np.isfinite(E_eucl)
    )
    if not np.any(valid):
        raise RuntimeError("No common valid entries between diagonal W2 and Euclidean distances.")

    W_diag = D_diag[valid]
    W_eucl = E_eucl[valid]

    diff_abs = W_diag - W_eucl
    denom = np.maximum(W_diag, 1e-12)
    diff_rel = 100.0 * diff_abs / denom

    if max_rel_pct is not None:
        keep = np.abs(diff_rel) <= max_rel_pct
        diff_rel = diff_rel[keep]

    # Medians
    med_abs = np.nanmedian(diff_abs)
    med_rel = np.nanmedian(diff_rel)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax0, ax1 = axes

    # Left: absolute difference
    ax0.hist(diff_abs, bins=bins, density=density, alpha=0.8)
    ax0.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax0.axvline(med_abs, color="k", linestyle="-", linewidth=1)
    ax0.set_xlabel(r"$W^{\mathrm{diag}} - \|\mu_1-\mu_2\|_2$", fontsize=14)
    ax0.set_ylabel("Density" if density else "Counts", fontsize=14)
    ax0.set_title("Diagonal vs Euclidean — absolute", fontsize=14)

    # Right: relative difference
    ax1.hist(diff_rel, bins=bins, density=density, alpha=0.8)
    ax1.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax1.axvline(med_rel, color="k", linestyle="-", linewidth=1)
    ax1.set_xlabel(
        r"$100 \times \dfrac{W^{\mathrm{diag}} - \|\mu_1-\mu_2\|_2}{W^{\mathrm{diag}}}$ (%)",
        fontsize=14,
    )
    ax1.set_ylabel("Density" if density else "Counts", fontsize=14)
    ax1.set_title("Diagonal vs Euclidean — relative (%)", fontsize=14)

    if xlim_abs is not None:
        ax0.set_xlim(*xlim_abs)
    if ylim_abs is not None:
        ax0.set_ylim(*ylim_abs)
    if xlim_rel is not None:
        ax1.set_xlim(*xlim_rel)
    if ylim_rel is not None:
        ax1.set_ylim(*ylim_rel)

    if logy:
        ax0.set_yscale("log")
        ax1.set_yscale("log")

    fig.tight_layout()
    plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")

    return fig, axes

def plot_gene_graphs_from_data(
    data: dict,
    datasets: Optional[Sequence[Union[int, str]]] = None,
    *,
    # ------------------------------------------------------------------
    # Similarity graph construction from pairwise gene distances
    # ------------------------------------------------------------------
    sigma: Union[float, str] = "auto",
    graph_mode: str = "knn",
    k: int = 12,
    mutual_knn: bool = False,
    epsilon: Optional[float] = None,
    add_mst: bool = True,

    # ------------------------------------------------------------------
    # Embedding and clustering
    # ------------------------------------------------------------------
    layout: str = "umap",
    resolution: float = 1.0,
    prefer_partition: str = "auto",
    random_seed: int = 0,

    # UMAP parameters applied to flattened Minkowski profiles
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.10,
    umap_spread: float = 1.0,
    umap_n_components: int = 2,
    umap_init: str = "spectral",
    umap_metric: str = "euclidean",
    umap_low_memory: bool = True,
    umap_kw: Optional[Dict[str, Any]] = None,

    # ------------------------------------------------------------------
    # Figure size and axis limits
    # ------------------------------------------------------------------
    figsize_umap: tuple[float, float] = (20.0, 7.0),
    figsize_mds: tuple[float, float] = (20.0, 7.0),
    figsize_pca: tuple[float, float] = (20.0, 7.0),

    umap_xlim: Optional[tuple[float, float]] = None,
    umap_ylim: Optional[tuple[float, float]] = None,
    mds_xlim: Optional[tuple[float, float]] = None,
    mds_ylim: Optional[tuple[float, float]] = None,
    pca_xlim: Optional[tuple[float, float]] = None,
    pca_ylim: Optional[tuple[float, float]] = None,

    node_size: float = 8.0,
    node_alpha: float = 0.9,
    edge_mode: str = "mst",
    thin_k: int = 3,
    edge_alpha: float = 0.25,

    # ------------------------------------------------------------------
    # Gene annotations
    # ------------------------------------------------------------------
    genes_to_label: Optional[Sequence[str]] = None,
    max_gene_labels: Optional[int] = None,
    goi_text_size: float = 11.0,
    goi_marker_scale: float = 1.6,
    goi_darken: float = 0.45,

    # ------------------------------------------------------------------
    # Links between copies of the same gene across datasets (UMAP left only)
    # ------------------------------------------------------------------
    link_same_gene_across_datasets: bool = False,
    link_only_for_annotated: bool = False,
    max_gene_links: Optional[int] = None,
    link_lw: float = 0.6,
    link_alpha: float = 0.35,

    # ------------------------------------------------------------------
    # Colours and titles
    # ------------------------------------------------------------------
    dataset_palette: Optional[Dict[str, Any]] = None,
    dataset_lighten: float = 0.35,
    title_umap_left: str = "Gene-level UMAP based on Minkowski profiles (coloured by dataset)",
    title_umap_right: str = "Gene-level UMAP based on Minkowski profiles (coloured by cluster)",
    title_mds_left: str = "Gene-level MDS based on distance matrix (coloured by dataset)",
    title_mds_right: str = "Gene-level MDS based on distance matrix (coloured by dataset)",
    title_pca_left: str = "Gene-level PCA based on Minkowski profiles (coloured by dataset)",
    title_pca_right: str = "Gene-level PCA based on Minkowski profiles (coloured by dataset)",

    # Legends
    legend_loc_datasets: str = "best",
    show_cluster_legend: bool = False,
    legend_loc_clusters: str = "best",
    cluster_legend_fontsize: float = 14.0,

    # ------------------------------------------------------------------
    # Cluster identifier annotations (UMAP right panel only)
    # ------------------------------------------------------------------
    show_cluster_ids_on_umap: bool = True,
    cluster_id_fontsize: float = 12.0,
    cluster_id_fontweight: str = "bold",
    cluster_id_text_color: str = "black",
    cluster_id_alpha: float = 1.0,
    cluster_id_bbox: bool = True,
    cluster_id_bbox_pad: float = 0.25,
    cluster_id_bbox_fc: str = "white",
    cluster_id_bbox_ec: str = "none",
    cluster_id_bbox_alpha: float = 0.75,

    # adjustText control for cluster identifiers on the UMAP right panel
    cluster_id_adjust: bool = True,
    cluster_id_adjust_expand_text: tuple[float, float] = (1.2, 1.4),
    cluster_id_adjust_force_text: float = 0.8,

    # ------------------------------------------------------------------
    # Optional PCA figure
    # ------------------------------------------------------------------
    make_pca_figure: bool = False,

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    out_umap_pdf: Optional[str] = None,
    out_mds_pdf: Optional[str] = None,
    out_pca_pdf: Optional[str] = None,

    # ------------------------------------------------------------------
    # Cluster summary and membership export
    # ------------------------------------------------------------------
    print_cluster_info: bool = True,
    out_cluster_membership_rds: Optional[str] = None,
    out_cluster_membership_csv: Optional[str] = None,

    # ------------------------------------------------------------------
    # Formatting of printed mean-density columns
    # ------------------------------------------------------------------
    mean_density_floatfmt: str = ".3e",
) -> dict:
    """
    Build a gene-level similarity graph from pairwise gene distances, cluster
    the graph, and generate UMAP, MDS and optional PCA visualisations.

    This function operates on one or more datasets for which pairwise
    gene-to-gene distances have already been computed. These distances may be

        - Gaussian 2-Wasserstein distances, when covariance matrices were
          available upstream, or
        - Euclidean distances between flattened Minkowski profiles, when
          covariance matrices were absent upstream.

    The function is therefore agnostic to the distance type. It only assumes
    that a valid pairwise gene-distance payload has been stored in
    ``data["gene_distances"]``. For backward compatibility, it also accepts
    ``data["w2_allpairs"]``.

    Importantly, this function always gives priority to the main distance
    tensor ``payload["D"]``. By construction, ``payload["D"]`` is the primary
    distance chosen upstream:

        - Wasserstein if covariance information was available,
        - Euclidean otherwise.

    If an auxiliary Euclidean tensor ``payload["E"]`` is also present, it is
    treated only as a sidecar and is never used here to override the main
    distance definition.

    From the selected distance blocks it assembles a global gene-by-gene
    distance matrix across datasets, converts this matrix into a sparse
    similarity graph, partitions the graph into clusters, and displays the
    result in several low-dimensional views.

    The function returns three possible geometric views.

    First, a two-panel UMAP figure is produced. The left panel is coloured by
    dataset, while the right panel is coloured by graph cluster. Optional gene
    annotations and optional links between copies of the same gene across
    datasets can be added to the left panel. Cluster identifiers and the
    cluster legend are shown only on the UMAP right panel.

    Secondly, a two-panel classical MDS figure is produced from the assembled
    pairwise gene-distance matrix. In this version, both MDS panels are
    coloured by dataset only.

    Thirdly, an optional PCA figure can be produced from flattened Minkowski
    profiles. As for MDS, both PCA panels are coloured by dataset only.

    Parameters
    ----------
    data:
        Dictionary containing at least

            - ``"gene_distances"`` or ``"w2_allpairs"``
            - ``"tensor_scaled"``: ndarray of shape ``(F, G, 4, L)``
            - ``"conditions"``: dataset labels

        Optional fields such as ``"gene_names"``, ``"gene_is_goi"``, and
        density arrays are used when available.

    datasets:
        Optional subset of datasets to include. Each selector may be either
        a dataset index or a dataset name. If ``None``, all datasets present
        in the distance payload are used.

    sigma:
        Width of the radial-basis similarity kernel applied to the assembled
        gene-distance matrix. If ``"auto"``, a robust median-based estimate
        is used.

    graph_mode:
        Graph sparsification mode. ``"knn"`` builds a k-nearest-neighbour graph;
        ``"epsilon"`` connects pairs with similarity above ``epsilon``.

    k:
        Number of neighbours used when ``graph_mode="knn"``.

    mutual_knn:
        If ``True``, retain only mutual nearest-neighbour edges. Otherwise use
        the symmetrised union.

    epsilon:
        Similarity threshold used when ``graph_mode="epsilon"``. If omitted,
        a heuristic quantile-based threshold is used.

    add_mst:
        If ``True``, augment the sparse graph with a minimum spanning tree to
        improve global connectivity.

    layout:
        Embedding used for the main figure. If ``"umap"``, UMAP is applied to
        flattened Minkowski profiles. If UMAP fails or is unavailable, a
        spectral embedding is used instead. If ``"spectral"``, the spectral
        embedding is used directly.

    resolution:
        Resolution parameter used by Leiden or Louvain clustering when those
        algorithms are available.

    prefer_partition:
        Preferred clustering backend. Allowed values are ``"auto"``,
        ``"leiden"``, ``"louvain"``, and ``"spectral-fallback"``.

    random_seed:
        Random seed used for embedding and clustering.

    umap_n_neighbors, umap_min_dist, umap_spread, umap_n_components,
    umap_init, umap_metric, umap_low_memory, umap_kw:
        Parameters controlling the optional UMAP embedding of flattened
        Minkowski profiles.

    figsize_umap, figsize_mds, figsize_pca:
        Figure sizes for the UMAP, MDS and PCA figures.

    umap_xlim, umap_ylim, mds_xlim, mds_ylim, pca_xlim, pca_ylim:
        Optional axis limits applied symmetrically to both panels of the
        corresponding figure.

    node_size, node_alpha:
        Marker size and transparency for gene nodes.

    edge_mode:
        Edge rendering mode. ``"none"`` suppresses graph edges,
        ``"mst"`` draws only the minimum spanning tree, and ``"thin-knn"``
        draws a thin subset of graph edges.

    thin_k:
        Number of strongest graph neighbours to draw per node when
        ``edge_mode="thin-knn"``.

    edge_alpha:
        Transparency of rendered edges.

    genes_to_label:
        Optional list of genes to annotate. If ``None``, the function uses
        ``data["gene_is_goi"]`` when available.

    max_gene_labels:
        Optional cap on the number of annotated base genes.

    goi_text_size, goi_marker_scale, goi_darken:
        Styling parameters for highlighted gene annotations.

    link_same_gene_across_datasets:
        If ``True``, connect copies of the same gene across datasets in the
        UMAP left panel.

    link_only_for_annotated:
        If ``True``, cross-dataset links are drawn only for annotated genes.

    max_gene_links:
        Optional cap on the number of base genes for which cross-dataset links
        are drawn.

    link_lw, link_alpha:
        Styling parameters for cross-dataset links.

    dataset_palette:
        Optional mapping from dataset names to base colours.

    dataset_lighten:
        Amount used to lighten dataset colours for the main dataset-coloured
        panels.

    title_umap_left, title_umap_right, title_mds_left, title_mds_right,
    title_pca_left, title_pca_right:
        Figure titles.

    legend_loc_datasets:
        Legend location for the dataset legend on the UMAP left panel.

    show_cluster_legend:
        If ``True``, show the cluster legend on the UMAP right panel.

    legend_loc_clusters:
        Legend location for the cluster legend on the UMAP right panel.

    cluster_legend_fontsize:
        Font size of the cluster legend on the UMAP right panel.

    show_cluster_ids_on_umap:
        If ``True``, annotate cluster identifiers on the UMAP right panel.

    cluster_id_fontsize, cluster_id_fontweight, cluster_id_text_color,
    cluster_id_alpha, cluster_id_bbox, cluster_id_bbox_pad,
    cluster_id_bbox_fc, cluster_id_bbox_ec, cluster_id_bbox_alpha:
        Styling parameters for cluster identifier annotations.

    cluster_id_adjust, cluster_id_adjust_expand_text,
    cluster_id_adjust_force_text:
        Parameters controlling ``adjustText`` for cluster identifiers on the
        UMAP right panel.

    make_pca_figure:
        If ``True``, generate the PCA figure.

    out_umap_pdf, out_mds_pdf, out_pca_pdf:
        Optional export paths. A ``.pdf`` suffix is appended if missing.

    print_cluster_info:
        If ``True``, compute and print a cluster summary table.

    out_cluster_membership_rds, out_cluster_membership_csv:
        Optional export paths for per-node cluster membership tables.

    mean_density_floatfmt:
        Float-format string used only for printing mean-density columns in the
        cluster summary.

    Returns
    -------
    dict
        Dictionary containing the graph, assembled distance matrix, embeddings,
        cluster assignments, summary tables, Matplotlib figure objects, and
        metadata describing the distance type used.

    Notes
    -----
    In this version, cluster colours, cluster legend, and cluster identifier
    annotations are shown only on the UMAP right panel. MDS and PCA panels are
    coloured by dataset only.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.sparse.linalg import eigsh
    from matplotlib.lines import Line2D
    from matplotlib import colors as mcolors

    try:
        import umap as _umap_mod
        _HAVE_UMAP = True
    except Exception:
        _HAVE_UMAP = False
        _umap_mod = None

    try:
        from adjustText import adjust_text as _adjust_text
        _HAVE_ADJUST = True
    except Exception:
        _HAVE_ADJUST = False
        _adjust_text = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _lighten(color, amount: float = 0.3):
        r, g, b = mcolors.to_rgb(color)
        return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

    def _darken(color, amount: float = 0.3):
        r, g, b = mcolors.to_rgb(color)
        return (r * (1 - amount), g * (1 - amount), b * (1 - amount))

    def _resolve_datasets(payload: dict, wanted):
        """Resolve dataset selectors against the datasets present in the payload."""
        labels = list(map(str, payload["labels"]))
        if wanted is None:
            idx = list(range(len(labels)))
            names = labels
        else:
            conds = list(map(str, data.get("conditions", [])))

            def _one(x):
                if isinstance(x, (int, np.integer)):
                    if 0 <= x < len(labels):
                        return int(x)
                    if 0 <= x < len(conds) and conds[int(x)] in labels:
                        return labels.index(conds[int(x)])
                    raise IndexError(f"Dataset index {x} out of range.")
                name = str(x)
                if name in labels:
                    return labels.index(name)
                if name in conds:
                    return labels.index(name)
                raise KeyError(f"Dataset {name!r} not found in distance payload['labels'].")

            idx = [_one(x) for x in list(wanted)]
            seen = set()
            idx = [i for i in idx if not (i in seen or seen.add(i))]
            names = [labels[i] for i in idx]
        return idx, names

    def _assemble_block_distance(payload: dict, idx_sel: Sequence[int]):
        """
        Assemble the full gene-by-gene distance matrix for the selected datasets.

        Priority is always given to the main distance tensor stored in
        ``payload["D"]`` and its associated block views ``payload["blocks"]``.
        By construction, this corresponds to the primary upstream distance:

            - Wasserstein when covariance information was available,
            - Euclidean otherwise.

        If an auxiliary Euclidean tensor ``payload["E"]`` is also present, it is
        ignored here and never overrides the main distance definition.
        """
        if "blocks" not in payload:
            raise KeyError("Distance payload must contain 'blocks' built from payload['D'].")

        blocks = payload["blocks"]
        any_key = next(iter(blocks.keys()))
        G = int(blocks[any_key].shape[0])
        K = len(idx_sel)
        N = K * G
        D = np.full((N, N), np.nan, dtype=np.float64)
        labels = list(map(str, payload["labels"]))

        for a, ia in enumerate(idx_sel):
            na = labels[ia]
            for b, ib in enumerate(idx_sel):
                nb = labels[ib]
                key = f"{na}|{nb}"
                if key in blocks:
                    B = np.asarray(blocks[key], dtype=np.float64)
                else:
                    rev_key = f"{nb}|{na}"
                    if rev_key not in blocks:
                        raise KeyError(
                            f"Could not find distance block '{key}' or '{rev_key}' in payload['blocks']."
                        )
                    B = np.asarray(blocks[rev_key], dtype=np.float64).T

                ra = slice(a * G, (a + 1) * G)
                rb = slice(b * G, (b + 1) * G)
                D[ra, rb] = B

        np.fill_diagonal(D, 0.0)
        return D, G

    def _rbf_from_dist(Dmat: np.ndarray, sigma_val: Union[float, str] = "auto"):
        """Convert a distance matrix into a radial-basis similarity matrix."""
        N = Dmat.shape[0]
        if isinstance(sigma_val, str) and sigma_val.lower() == "auto":
            meds = []
            for i in range(N):
                di = Dmat[i]
                m = (np.isfinite(di)) & (di > 0)
                if m.any():
                    meds.append(np.median(di[m]))
            if meds:
                sig = float(np.median(meds))
            else:
                off = Dmat[~np.eye(N, dtype=bool)]
                off = off[np.isfinite(off) & (off > 0)]
                sig = float(np.median(off)) if off.size else 1.0
        else:
            sig = float(sigma_val)

        sig = max(sig, 1e-8)
        S = np.exp(-(Dmat.astype(np.float64) ** 2) / (2.0 * sig * sig))
        np.fill_diagonal(S, 0.0)
        return S.astype(np.float32)

    def _knn(S: np.ndarray, k: int = 12, mutual: bool = True):
        """Build a sparse k-nearest-neighbour similarity graph."""
        N = S.shape[0]
        k = int(max(1, min(k, N - 1)))
        rows, cols, vals = [], [], []

        for i in range(N):
            row = S[i]
            nn = np.argpartition(-row, k + 1)[:k + 1]
            nn = nn[nn != i]
            if nn.size > k:
                nn = nn[:k]
            for j in nn:
                rows.append(i)
                cols.append(int(j))
                vals.append(float(row[j]))

        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float32)
        B = A.minimum(A.T) if mutual else A.maximum(A.T)
        B.eliminate_zeros()
        return B.tocsr()

    def _eps_graph(S: np.ndarray, eps: float):
        """Build a sparse epsilon-neighbourhood similarity graph."""
        mask = (S >= float(eps)) & np.isfinite(S)
        np.fill_diagonal(mask, False)
        r, c = np.where(mask)
        v = S[r, c].astype(np.float32)
        A = sp.coo_matrix((v, (r, c)), shape=S.shape, dtype=np.float32).tocsr()
        A = A.maximum(A.T)
        A.eliminate_zeros()
        return A

    def _add_MST(A: sp.csr_matrix, Dmat: np.ndarray):
        """Add a minimum spanning tree to the graph to improve connectivity."""
        D2 = Dmat.copy().astype(np.float64)
        finite = np.isfinite(D2)
        huge = (np.nanmax(D2[finite]) if finite.any() else 1.0) * 10.0 + 1.0
        D2[~finite] = huge
        np.fill_diagonal(D2, 0.0)

        T = minimum_spanning_tree(sp.csr_matrix(D2))
        T = T.maximum(T.T)
        r, c = T.nonzero()
        w = 1.0 / (1.0 + D2[r, c])
        MST = sp.csr_matrix((w, (r, c)), shape=D2.shape, dtype=np.float32)
        MST = MST.maximum(MST.T)

        out = A.maximum(MST)
        out.eliminate_zeros()
        return out.tocsr()

    def _spectral(A: sp.csr_matrix, n_components: int = 2, seed: int = 42):
        """Compute a spectral embedding of the graph."""
        A = A.tocsr().astype(np.float64)
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        Dm12 = sp.diags(1.0 / np.sqrt(deg))
        L = sp.eye(A.shape[0], dtype=np.float64) - Dm12 @ A @ Dm12

        k_ = min(n_components + 1, max(2, A.shape[0] - 1))
        vals, vecs = eigsh(
            L,
            k=k_,
            which="SM",
            tol=1e-4,
            maxiter=A.shape[0] * 20,
            v0=np.random.RandomState(seed).rand(A.shape[0]),
        )
        X = vecs[:, 1:n_components + 1]

        for j in range(X.shape[1]):
            if np.sum(X[:, j]) < 0:
                X[:, j] *= -1

        return X.astype(np.float32)

    def _umap_from_features(
        X: np.ndarray,
        *,
        n_neighbors: int = 15,
        min_dist: float = 0.10,
        spread: float = 1.0,
        n_components: int = 2,
        init: str = "spectral",
        metric: str = "euclidean",
        low_memory: bool = True,
        random_state: int = 42,
        extra_kw: Optional[Dict[str, Any]] = None,
    ):
        """Compute a UMAP embedding from flattened feature vectors."""
        if not _HAVE_UMAP:
            raise RuntimeError("UMAP is not installed.")

        metric_eff = "euclidean" if str(metric).lower() == "precomputed" else str(metric)
        kw = dict(
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            spread=float(spread),
            n_components=int(n_components),
            init=str(init),
            metric=metric_eff,
            low_memory=bool(low_memory),
            random_state=int(random_state),
            verbose=False,
        )
        if extra_kw:
            kw.update(dict(extra_kw))

        reducer = _umap_mod.UMAP(**kw)
        X = np.asarray(X, dtype=np.float64)
        Z = reducer.fit_transform(X)
        return Z.astype(np.float32)

    def _cmdscale_from_dist(D: np.ndarray):
        """Perform classical metric MDS from a distance matrix."""
        D = np.asarray(D, dtype=float)
        N = D.shape[0]
        J = np.eye(N) - np.ones((N, N)) / N
        B = -0.5 * J @ (D ** 2) @ J

        evals, evecs = np.linalg.eigh(B)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        pos = evals > (1e-12 * np.max(evals) if evals.size else 0.0)
        evals_pos = evals[pos]
        evecs_pos = evecs[:, pos]

        if evals_pos.size == 0:
            return np.zeros((N, 2), dtype=float), (0.0, 0.0)

        L12 = np.sqrt(evals_pos)
        Xfull = evecs_pos * L12[None, :]
        total = np.sum(evals_pos)
        evr_all = evals_pos / (total if total > 0 else 1.0)

        if Xfull.shape[1] >= 2:
            X2 = Xfull[:, :2]
            evr = (float(evr_all[0]), float(evr_all[1]))
        elif Xfull.shape[1] == 1:
            X2 = np.column_stack([Xfull[:, 0], np.zeros(N)])
            evr = (float(evr_all[0]), 0.0)
        else:
            X2 = np.zeros((N, 2), dtype=float)
            evr = (0.0, 0.0)

        for j in range(2):
            if np.sum(X2[:, j]) < 0:
                X2[:, j] *= -1

        return X2.astype(np.float32), evr

    def _pca_from_features(X: np.ndarray, n_components: int = 2):
        """Compute a PCA embedding from flattened feature vectors using SVD."""
        X = np.asarray(X, dtype=float)
        X = X - X.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        if s.size == 0:
            return np.zeros((X.shape[0], 2), dtype=float), (0.0, 0.0)

        expl_var = (s ** 2) / max(X.shape[0] - 1, 1)
        total = expl_var.sum() if expl_var.size else 0.0
        evr_all = expl_var / total if total > 0 else expl_var
        scores = U * s

        if scores.shape[1] >= 2:
            X2 = scores[:, :2]
            evr = (float(evr_all[0]), float(evr_all[1]) if evr_all.size > 1 else 0.0)
        elif scores.shape[1] == 1:
            X2 = np.column_stack([scores[:, 0], np.zeros(X.shape[0])])
            evr = (float(evr_all[0]), 0.0 if evr_all.size > 0 else 0.0)
        else:
            X2 = np.zeros((X.shape[0], 2), dtype=float)
            evr = (0.0, 0.0)

        for j in range(2):
            if np.sum(X2[:, j]) < 0:
                X2[:, j] *= -1

        return X2.astype(np.float32), evr

    def _resolve_gene_density_matrix(
        data_obj: dict,
        selected_dataset_names: Sequence[str],
        n_genes: int
    ) -> np.ndarray:
        """Recover a gene-density matrix for the selected datasets, if available."""
        Kloc = len(selected_dataset_names)
        out = np.full((Kloc, n_genes), np.nan, dtype=float)

        conds_all_local = list(map(str, data_obj.get("conditions", [])))
        cond_to_idx = {c: i for i, c in enumerate(conds_all_local)}

        candidate_keys = [
            "densities", "density", "gene_densities", "gene_density",
            "mean_densities", "density_norm", "gene_density_norm",
            "normalized_density",
        ]

        dens = None
        for kname in candidate_keys:
            if kname in data_obj:
                dens = data_obj[kname]
                break
        if dens is None:
            return out

        if isinstance(dens, dict):
            for k_idx, dname in enumerate(selected_dataset_names):
                if dname in dens:
                    vec = np.asarray(dens[dname], dtype=float).ravel()
                    if vec.size == n_genes:
                        out[k_idx] = vec
            return out

        arr = np.asarray(dens, dtype=float)
        if arr.ndim == 3:
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
            elif arr.shape[1] == 1:
                arr = arr[:, 0, :]
            else:
                return out
        if arr.ndim != 2:
            return out

        if arr.shape == (Kloc, n_genes):
            return arr.copy()

        if arr.shape[1] == n_genes and len(conds_all_local) == arr.shape[0]:
            for k_idx, dname in enumerate(selected_dataset_names):
                if dname in cond_to_idx:
                    out[k_idx] = arr[cond_to_idx[dname]]
            return out

        return out

    def _format_cluster_summary_for_print(df, density_cols, density_floatfmt: str):
        """Format density columns as strings for printed cluster summaries."""
        import pandas as pd
        out = df.copy()
        for col in density_cols:
            if col in out.columns:
                out[col] = out[col].map(
                    lambda x: (format(float(x), density_floatfmt) if pd.notna(x) else "nan")
                )
        return out

    # ------------------------------------------------------------------
    # 1. Resolve the distance payload
    # ------------------------------------------------------------------
    if "gene_distances" in data and isinstance(data["gene_distances"], dict):
        payload = data["gene_distances"]
    elif "w2_allpairs" in data and isinstance(data["w2_allpairs"], dict):
        payload = data["w2_allpairs"]
    else:
        raise KeyError(
            "No gene-distance payload found. Run compute_gene_distances(...) first."
        )

    distance_kind = str(payload.get("distance_kind", payload.get("kind", "unknown")))
    distance_label = str(payload.get("distance_label", "distance"))
    uses_covariance = bool(payload.get("uses_covariance", False))
    primary_distance_source = "wasserstein" if uses_covariance else "euclidean"

    # ------------------------------------------------------------------
    # 2. Assemble the global distance matrix for the selected datasets
    # ------------------------------------------------------------------
    idx_sel, dataset_names = _resolve_datasets(payload, datasets)
    D, G = _assemble_block_distance(payload, idx_sel)

    K = len(idx_sel)
    N = K * G

    # ------------------------------------------------------------------
    # 3. Build the flattened Minkowski feature matrix used by UMAP and PCA
    # ------------------------------------------------------------------
    if "tensor_scaled" not in data or "conditions" not in data:
        raise KeyError(
            "To build UMAP and PCA from Minkowski profiles, `data` must contain "
            "'tensor_scaled' (F×G×4×L) and 'conditions'."
        )

    X4 = np.asarray(data["tensor_scaled"])
    conds_all = list(map(str, data["conditions"]))
    _, G_check, S_stats, L = X4.shape

    if G_check != G:
        raise ValueError(
            f"Mismatch between gene dimension in distance payload (G={G}) and "
            f"tensor_scaled (G={G_check})."
        )

    d_feat = S_stats * L
    X_mink = np.empty((K, G, d_feat), dtype=np.float32)

    for a, name in enumerate(dataset_names):
        if name not in conds_all:
            raise KeyError(
                f"Dataset {name!r} from distance payload['labels'] not found in data['conditions']."
            )
        idx_cond = conds_all.index(name)
        prof = X4[idx_cond]
        X_mink[a] = prof.reshape(G, d_feat).astype(np.float32, copy=False)

    X_flat = X_mink.reshape(N, d_feat).astype(np.float32)

    if not np.isfinite(X_flat).all():
        X_flat = X_flat.astype(float)
        bad = ~np.isfinite(X_flat)
        col_means = np.nanmean(np.where(bad, np.nan, X_flat), axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        inds = np.where(bad)
        X_flat[inds] = col_means[inds[1]]

    # ------------------------------------------------------------------
    # 4. Construct the similarity graph
    # ------------------------------------------------------------------
    S = _rbf_from_dist(D, sigma_val=sigma)

    if graph_mode.lower() == "knn":
        A = _knn(S, k=int(k), mutual=bool(mutual_knn))
    elif graph_mode.lower() == "epsilon":
        if epsilon is None:
            nz = S[~np.eye(N, dtype=bool)]
            nz = nz[np.isfinite(nz) & (nz > 0)]
            eps = float(np.quantile(nz, 0.90)) if nz.size else 0.0
        else:
            eps = float(epsilon)
        A = _eps_graph(S, eps)
    else:
        raise ValueError("graph_mode must be 'knn' or 'epsilon'.")

    if add_mst:
        A = _add_MST(A, D)

    # ------------------------------------------------------------------
    # 5. Compute the main embedding
    # ------------------------------------------------------------------
    if layout.lower() == "umap":
        try:
            X_umap = _umap_from_features(
                X_flat,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                spread=umap_spread,
                n_components=umap_n_components,
                init=umap_init,
                metric=umap_metric,
                low_memory=umap_low_memory,
                random_state=random_seed,
                extra_kw=umap_kw,
            )
        except Exception:
            X_umap = _spectral(A, n_components=2, seed=random_seed)
    else:
        X_umap = _spectral(A, n_components=2, seed=random_seed)

    # ------------------------------------------------------------------
    # 6. Cluster the graph
    # ------------------------------------------------------------------
    def _eigengap_k(A: sp.csr_matrix, kmax: int = 15) -> int:
        """Estimate a cluster count from the eigengap of the graph Laplacian."""
        A_ = A.tocsr().astype(np.float64)
        deg = np.asarray(A_.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        Dm12 = sp.diags(1.0 / np.sqrt(deg))
        L_ = sp.eye(A_.shape[0], dtype=np.float64) - Dm12 @ A_ @ Dm12

        k_ = min(kmax + 1, max(2, A_.shape[0] - 1))
        vals, _ = eigsh(
            L_,
            k=k_,
            which="SM",
            tol=1e-4,
            maxiter=A_.shape[0] * 10,
            v0=np.random.RandomState(random_seed).rand(A_.shape[0]),
        )
        vals = np.sort(vals)
        gaps = np.diff(vals[:k_])
        if gaps.size == 0:
            return 1
        idx = int(np.argmax(gaps)) + 1
        return max(1, idx)

    def _partition(A: sp.csr_matrix, resolution: float = 1.0,
                   prefer: str = "auto", seed: int = 42) -> np.ndarray:
        """Partition the graph using Leiden, Louvain, or a spectral fallback."""
        Nloc = A.shape[0]

        if prefer in ("auto", "leiden"):
            try:
                import igraph as _ig_local
                import leidenalg as _la_local
                r, c = A.nonzero()
                w = A[r, c].A1
                g = _ig_local.Graph(
                    n=Nloc,
                    edges=list(zip(map(int, r), map(int, c))),
                    directed=False
                )
                g.es["weight"] = w
                part = _la_local.find_partition(
                    g,
                    _la_local.CPMVertexPartition,
                    weights=w,
                    resolution_parameter=float(resolution),
                    seed=seed,
                )
                return np.asarray(part.membership, dtype=int)
            except Exception:
                pass

        if prefer in ("auto", "louvain"):
            try:
                import networkx as nx
                import community as community_louvain_local
                Gx = nx.Graph()
                r, c = A.nonzero()
                for i, j, ww in zip(r, c, A[r, c].A1):
                    if i < j:
                        Gx.add_edge(int(i), int(j), weight=float(ww))
                lab = community_louvain_local.best_partition(
                    Gx,
                    weight="weight",
                    resolution=float(resolution),
                    random_state=seed,
                )
                return np.array([lab[i] for i in range(Nloc)], dtype=int)
            except Exception:
                pass

        from sklearn.cluster import KMeans
        k_ = _eigengap_k(A, kmax=min(25, max(2, int(np.sqrt(Nloc)))))
        Z = _spectral(A, n_components=max(2, k_), seed=seed)
        km = KMeans(n_clusters=k_, n_init=10, random_state=seed)
        return km.fit_predict(Z).astype(int)

    z = _partition(A, resolution=float(resolution), prefer=prefer_partition, seed=random_seed).astype(int)
    n_clusters = int(np.max(z)) + 1 if z.size else 0

    node_dataset = np.repeat(np.arange(K, dtype=int), G)
    node_gene = np.tile(np.arange(G), K)

    if z.size != K * G:
        raise ValueError(f"Unexpected z size: got {z.size}, expected {K * G}.")

    z_mat = z.reshape(K, G)
    base_gene_names = list(map(str, data.get("gene_names", [f"g{i}" for i in range(G)])))
    node_names = [base_gene_names[g] for g in node_gene]

    # ------------------------------------------------------------------
    # 7. Determine which genes should be annotated
    # ------------------------------------------------------------------
    label_mask = np.zeros(G, dtype=bool)
    if genes_to_label is not None:
        wanted = set(map(str, genes_to_label))
        name2idx = {nm: i for i, nm in enumerate(base_gene_names)}
        for nm in wanted:
            if nm in name2idx:
                label_mask[name2idx[nm]] = True
    elif "gene_is_goi" in data:
        gm = np.asarray(data["gene_is_goi"], dtype=bool)
        if gm.shape[0] == G:
            label_mask = gm

    def _select_goi_nodes() -> np.ndarray:
        """Expand the selected base-gene mask into node indices across datasets."""
        genes_idx = np.where(label_mask)[0]
        if max_gene_labels is not None:
            genes_idx = genes_idx[:int(max_gene_labels)]
        sel = []
        for g in genes_idx:
            for k_ in range(K):
                sel.append(k_ * G + g)
        return np.array(sel, dtype=int)

    # ------------------------------------------------------------------
    # 8. Define colours
    # ------------------------------------------------------------------
    if dataset_palette is None:
        cmap = plt.get_cmap("tab20")
        dataset_palette = {nm: cmap(i % 20) for i, nm in enumerate(dataset_names)}
    for nm in dataset_names:
        dataset_palette.setdefault(nm, "C0")

    base_colors = {nm: _lighten(dataset_palette[nm], dataset_lighten) for nm in dataset_names}
    goi_colors_dataset = {nm: _darken(base_colors[nm], goi_darken) for nm in dataset_names}

    node_colors_by_dataset = np.array([base_colors[dataset_names[k]] for k in node_dataset], dtype=object)

    cmap_mod = plt.get_cmap("tab20")
    cluster_colors = np.array([cmap_mod(i % 20) for i in range(max(1, n_clusters))], dtype=object)
    node_colors_by_cluster = cluster_colors[z] if z.size else "C0"

    # ------------------------------------------------------------------
    # 9. Compute MDS and optional PCA views
    # ------------------------------------------------------------------
    X_mds, evr_mds = _cmdscale_from_dist(D)
    mds_evr1 = f"{100.0 * evr_mds[0]:.0f}%" if np.isfinite(evr_mds[0]) else "0%"
    mds_evr2 = f"{100.0 * evr_mds[1]:.0f}%" if np.isfinite(evr_mds[1]) else "0%"

    X_pca = None
    pca_evr = (np.nan, np.nan)
    pca_evr1 = "0%"
    pca_evr2 = "0%"
    if make_pca_figure or (out_pca_pdf is not None):
        X_pca, pca_evr = _pca_from_features(X_flat, n_components=2)
        pca_evr1 = f"{100.0 * pca_evr[0]:.0f}%" if np.isfinite(pca_evr[0]) else "0%"
        pca_evr2 = f"{100.0 * pca_evr[1]:.0f}%" if np.isfinite(pca_evr[1]) else "0%"

    # ------------------------------------------------------------------
    # 10. Plotting helpers
    # ------------------------------------------------------------------
    def _style_axes_box(ax):
        """Apply a consistent boxed axis style without grid lines."""
        ax.grid(False)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.0)

    def _draw_edges(ax, mode: str, coords: np.ndarray):
        """Draw a sparse subset of graph edges in the chosen embedding."""
        if mode.lower() == "none":
            return

        if mode.lower() == "mst":
            T = minimum_spanning_tree(sp.csr_matrix(D))
            T = T.maximum(T.T)
            r, c = T.nonzero()
            for i, j in zip(r, c):
                if i < j:
                    ax.plot(
                        [coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        lw=0.6,
                        alpha=float(edge_alpha),
                        color="k",
                        zorder=1,
                    )

        elif mode.lower() == "thin-knn":
            for i in range(A.shape[0]):
                row = A.getrow(i)
                if row.nnz == 0:
                    continue
                kk = min(thin_k, row.nnz)
                idx = np.argpartition(-row.data, kk - 1)[:kk]
                js = row.indices[idx]
                for j in js:
                    if i < j:
                        ax.plot(
                            [coords[i, 0], coords[j, 0]],
                            [coords[i, 1], coords[j, 1]],
                            lw=0.4,
                            alpha=float(edge_alpha),
                            color="k",
                            zorder=1,
                        )

    def _annotate_goi(ax, coords: np.ndarray, use_dataset_colors: bool, title: Optional[str]):
        """Highlight selected genes of interest and optionally annotate them."""
        _style_axes_box(ax)
        sel_nodes = _select_goi_nodes()

        if sel_nodes.size:
            if use_dataset_colors:
                cols = [goi_colors_dataset[dataset_names[k]] for k in node_dataset[sel_nodes]]
                text_colors = cols
            else:
                cols = node_colors_by_cluster[sel_nodes]
                text_colors = ["black"] * len(sel_nodes)

            ax.scatter(
                coords[sel_nodes, 0],
                coords[sel_nodes, 1],
                s=node_size * (goi_marker_scale ** 2),
                c=cols,
                alpha=1.0,
                linewidths=0,
                zorder=6,
            )

            texts = []
            for i, n in enumerate(sel_nodes):
                texts.append(
                    ax.text(
                        coords[n, 0],
                        coords[n, 1],
                        node_names[n],
                        fontsize=goi_text_size,
                        color=text_colors[i],
                        ha="left",
                        va="bottom",
                        zorder=7,
                    )
                )

            if _HAVE_ADJUST and texts:
                _adjust_text(
                    texts,
                    ax=ax,
                    only_move={"points": "", "texts": "xy", "objects": "xy"},
                    expand_points=(1.4, 1.8),
                    expand_text=(1.1, 1.6),
                    force_points=0.2,
                    force_text=1.0,
                    autoalign="y",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="gray", alpha=0.6),
                )

        if title:
            ax.set_title(title, fontsize=19)

    def _annotate_cluster_ids_umap(ax, coords: np.ndarray):
        """Annotate cluster identifiers on the UMAP right panel."""
        if not show_cluster_ids_on_umap:
            return
        if n_clusters <= 0:
            return

        texts = []
        for c in range(n_clusters):
            idx = np.where(z == c)[0]
            if idx.size == 0:
                continue
            xy = coords[idx, :2].mean(axis=0)

            bbox = None
            if cluster_id_bbox:
                bbox = dict(
                    boxstyle=f"round,pad={cluster_id_bbox_pad}",
                    facecolor=cluster_id_bbox_fc,
                    edgecolor=cluster_id_bbox_ec,
                    alpha=cluster_id_bbox_alpha,
                )

            t = ax.text(
                float(xy[0]),
                float(xy[1]),
                str(c),
                fontsize=float(cluster_id_fontsize),
                fontweight=str(cluster_id_fontweight),
                color=str(cluster_id_text_color),
                alpha=float(cluster_id_alpha),
                ha="center",
                va="center",
                bbox=bbox,
                zorder=200,
            )
            texts.append(t)

        if _HAVE_ADJUST and cluster_id_adjust and texts:
            _adjust_text(
                texts,
                ax=ax,
                only_move={"texts": "xy"},
                expand_text=cluster_id_adjust_expand_text,
                force_text=float(cluster_id_adjust_force_text),
                autoalign="xy",
                arrowprops=None,
            )

    # ------------------------------------------------------------------
    # 11. Figure 1: UMAP
    # ------------------------------------------------------------------
    fig_umap, axs_umap = plt.subplots(1, 2, figsize=figsize_umap)
    ax0, ax1 = axs_umap

    _style_axes_box(ax0)
    _draw_edges(ax0, edge_mode, X_umap)
    ax0.scatter(
        X_umap[:, 0], X_umap[:, 1],
        s=node_size,
        c=node_colors_by_dataset,
        alpha=node_alpha,
        linewidths=0,
        zorder=5,
    )

    if link_same_gene_across_datasets and K >= 2:
        genes_to_link = np.where(label_mask)[0] if link_only_for_annotated else np.arange(G)
        if max_gene_links is not None:
            genes_to_link = genes_to_link[:int(max_gene_links)]
        for g in genes_to_link:
            idx_nodes = [k_ * G + g for k_ in range(K)]
            for a in range(K - 1):
                i = idx_nodes[a]
                j = idx_nodes[a + 1]
                ax0.plot(
                    [X_umap[i, 0], X_umap[j, 0]],
                    [X_umap[i, 1], X_umap[j, 1]],
                    lw=link_lw,
                    alpha=link_alpha,
                    color="k",
                    zorder=2,
                )

    _annotate_goi(ax0, X_umap, use_dataset_colors=True, title=title_umap_left)
    ax0.set_xlabel("UMAP 1", fontsize=17)
    ax0.set_ylabel("UMAP 2", fontsize=19)

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            label=nm,
            markerfacecolor=base_colors[nm],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=9,
        )
        for nm in dataset_names
    ]
    if handles:
        leg = ax0.legend(handles=handles, loc=legend_loc_datasets, frameon=True, fontsize=17)
        leg.set_zorder(100)
        frame = leg.get_frame()
        frame.set_alpha(0.8)
        frame.set_facecolor("white")

    _style_axes_box(ax1)
    _draw_edges(ax1, edge_mode, X_umap)
    ax1.scatter(
        X_umap[:, 0], X_umap[:, 1],
        s=node_size,
        c=node_colors_by_cluster,
        alpha=node_alpha,
        linewidths=0,
        zorder=5,
    )
    _annotate_goi(ax1, X_umap, use_dataset_colors=False, title=title_umap_right)
    ax1.set_xlabel("UMAP 1", fontsize=17)
    ax1.set_ylabel("UMAP 2", fontsize=19)

    if show_cluster_legend and n_clusters > 0:
        counts_cluster = np.bincount(z, minlength=n_clusters)
        ch = [
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                label=f"C{i} : {int(counts_cluster[i])} genes",
                markerfacecolor=cluster_colors[i],
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=9,
            )
            for i in range(n_clusters)
        ]
        leg = ax1.legend(
            handles=ch,
            loc=legend_loc_clusters,
            frameon=True,
            fontsize=cluster_legend_fontsize,
        )
        leg.set_zorder(100)
        frame = leg.get_frame()
        frame.set_alpha(0.8)
        frame.set_facecolor("white")

    _annotate_cluster_ids_umap(ax1, X_umap)

    if umap_xlim is not None:
        ax0.set_xlim(umap_xlim)
        ax1.set_xlim(umap_xlim)
    if umap_ylim is not None:
        ax0.set_ylim(umap_ylim)
        ax1.set_ylim(umap_ylim)

    fig_umap.tight_layout()
    if out_umap_pdf is not None:
        path_umap = out_umap_pdf if str(out_umap_pdf).lower().endswith(".pdf") else (str(out_umap_pdf) + ".pdf")
        fig_umap.savefig(path_umap, bbox_inches="tight", pad_inches=0.5)

    # ------------------------------------------------------------------
    # 12. Figure 2: MDS
    # ------------------------------------------------------------------
    fig_mds, axs_mds = plt.subplots(1, 2, figsize=figsize_mds)
    ax2, ax3 = axs_mds

    _style_axes_box(ax2)
    _draw_edges(ax2, edge_mode, X_mds)
    ax2.scatter(
        X_mds[:, 0], X_mds[:, 1],
        s=node_size,
        c=node_colors_by_dataset,
        alpha=node_alpha,
        linewidths=0,
        zorder=5,
    )
    _annotate_goi(ax2, X_mds, use_dataset_colors=True, title=title_mds_left)
    ax2.set_xlabel(f"MDS 1 (EVR = {mds_evr1})", fontsize=17)
    ax2.set_ylabel(f"MDS 2 (EVR = {mds_evr2})", fontsize=19)

    _style_axes_box(ax3)
    _draw_edges(ax3, edge_mode, X_mds)
    ax3.scatter(
        X_mds[:, 0], X_mds[:, 1],
        s=node_size,
        c=node_colors_by_dataset,
        alpha=node_alpha,
        linewidths=0,
        zorder=5,
    )
    _annotate_goi(ax3, X_mds, use_dataset_colors=True, title=title_mds_right)
    ax3.set_xlabel(f"MDS 1 (EVR = {mds_evr1})", fontsize=17)
    ax3.set_ylabel(f"MDS 2 (EVR = {mds_evr2})", fontsize=19)

    if mds_xlim is not None:
        ax2.set_xlim(mds_xlim)
        ax3.set_xlim(mds_xlim)
    if mds_ylim is not None:
        ax2.set_ylim(mds_ylim)
        ax3.set_ylim(mds_ylim)

    fig_mds.tight_layout()
    if out_mds_pdf is not None:
        path_mds = out_mds_pdf if str(out_mds_pdf).lower().endswith(".pdf") else (str(out_mds_pdf) + ".pdf")
        fig_mds.savefig(path_mds, bbox_inches="tight", pad_inches=0.5)

    # ------------------------------------------------------------------
    # 13. Figure 3: PCA (optional)
    # ------------------------------------------------------------------
    fig_pca = None
    axs_pca = None
    if X_pca is not None and (make_pca_figure or out_pca_pdf is not None):
        fig_pca, axs_pca = plt.subplots(1, 2, figsize=figsize_pca)

        ax = axs_pca[0]
        _style_axes_box(ax)
        _draw_edges(ax, edge_mode, X_pca)
        ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            s=node_size,
            c=node_colors_by_dataset,
            alpha=node_alpha,
            linewidths=0,
            zorder=5,
        )
        _annotate_goi(ax, X_pca, use_dataset_colors=True, title=title_pca_left)
        ax.set_xlabel(f"PC1 (EVR = {pca_evr1})", fontsize=17)
        ax.set_ylabel(f"PC2 (EVR = {pca_evr2})", fontsize=19)

        ax = axs_pca[1]
        _style_axes_box(ax)
        _draw_edges(ax, edge_mode, X_pca)
        ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            s=node_size,
            c=node_colors_by_dataset,
            alpha=node_alpha,
            linewidths=0,
            zorder=5,
        )
        _annotate_goi(ax, X_pca, use_dataset_colors=True, title=title_pca_right)
        ax.set_xlabel(f"PC1 (EVR = {pca_evr1})", fontsize=17)
        ax.set_ylabel(f"PC2 (EVR = {pca_evr2})", fontsize=19)

        if pca_xlim is not None:
            axs_pca[0].set_xlim(pca_xlim)
            axs_pca[1].set_xlim(pca_xlim)
        if pca_ylim is not None:
            axs_pca[0].set_ylim(pca_ylim)
            axs_pca[1].set_ylim(pca_ylim)

        fig_pca.tight_layout()
        if out_pca_pdf is not None:
            path_pca = out_pca_pdf if str(out_pca_pdf).lower().endswith(".pdf") else (str(out_pca_pdf) + ".pdf")
            fig_pca.savefig(path_pca, bbox_inches="tight", pad_inches=0.5)

    # ------------------------------------------------------------------
    # 14. Cluster summary and optional membership export
    # ------------------------------------------------------------------
    cluster_info_df = None
    membership_df = None

    try:
        import pandas as pd

        density_by_condition_gene = _resolve_gene_density_matrix(data, dataset_names, G)
        is_goi_gene = label_mask[node_gene]

        membership_df = pd.DataFrame({
            "node_id": np.arange(N, dtype=int),
            "condition": [dataset_names[i] for i in node_dataset],
            "condition_id": node_dataset.astype(int),
            "gene": [base_gene_names[i] for i in node_gene],
            "gene_id": node_gene.astype(int),
            "cluster": z.astype(int),
            "is_goi": is_goi_gene.astype(bool),
        })

        goi_gene_ids = np.where(label_mask)[0]
        n_goi_total = int(goi_gene_ids.size)
        has_any_goi = n_goi_total > 0

        rows = []
        for c in range(n_clusters):
            nodes_c = np.where(z == c)[0]
            n_nodes = int(nodes_c.size)
            if n_nodes == 0:
                continue

            genes_c = np.unique(node_gene[nodes_c])
            n_genes = int(genes_c.size)

            if genes_c.size > 0:
                stays_all = np.all(z_mat[:, genes_c] == c, axis=0)
                retention_pct = 100.0 * float(np.mean(stays_all))
            else:
                retention_pct = 0.0

            row = {
                "Cluster": c,
                "n_nodes": n_nodes,
                "n_genes": n_genes,
                "Retention [%]": retention_pct,
            }

            if has_any_goi:
                goi_nodes_pct = 100.0 * float(is_goi_gene[nodes_c].mean()) if n_nodes > 0 else 0.0
                goi_genes_covered = np.intersect1d(genes_c, goi_gene_ids).size
                goi_genes_covered_pct = 100.0 * float(goi_genes_covered) / float(n_goi_total)
                row["GOI nodes [%]"] = goi_nodes_pct
                row["GOI genes covered [%]"] = goi_genes_covered_pct

            cond_counts = np.bincount(node_dataset[nodes_c], minlength=K).astype(float)
            cond_pct = 100.0 * cond_counts / float(n_nodes)

            for kk in range(K):
                row[f"{dataset_names[kk]} [%]"] = cond_pct[kk]
                dens_vals = density_by_condition_gene[kk, genes_c] if genes_c.size > 0 else np.array([], dtype=float)
                dens_vals = dens_vals[np.isfinite(dens_vals)]
                row[f"{dataset_names[kk]} mean density"] = float(np.mean(dens_vals)) if dens_vals.size > 0 else np.nan

            rows.append(row)

        cluster_info_df = pd.DataFrame(rows)
        if cluster_info_df is not None and not cluster_info_df.empty:
            cluster_info_df = cluster_info_df.sort_values("Cluster").reset_index(drop=True)

        if print_cluster_info and cluster_info_df is not None:
            density_cols = [f"{nm} mean density" for nm in dataset_names]
            dfp = _format_cluster_summary_for_print(cluster_info_df, density_cols, mean_density_floatfmt)
            print("\n=== Cluster summary (nodes/genes/retention/dataset composition/mean densities) ===")
            print(dfp.to_string(index=False))

        if out_cluster_membership_rds is not None:
            path_rds = str(out_cluster_membership_rds)
            if not path_rds.lower().endswith(".rds"):
                path_rds += ".rds"
            try:
                import pyreadr
                pyreadr.write_rds(path_rds, membership_df)
            except Exception as e:
                if out_cluster_membership_csv is not None:
                    path_csv = str(out_cluster_membership_csv)
                    if not path_csv.lower().endswith(".csv"):
                        path_csv += ".csv"
                    membership_df.to_csv(path_csv, index=False)
                    print(f"[WARN] Could not write RDS ({e}). Wrote CSV instead: {path_csv}")
                else:
                    raise RuntimeError(
                        "Requested out_cluster_membership_rds but could not write RDS. "
                        "Install `pyreadr` or provide out_cluster_membership_csv."
                    ) from e

        if out_cluster_membership_csv is not None:
            path_csv = str(out_cluster_membership_csv)
            if not path_csv.lower().endswith(".csv"):
                path_csv += ".csv"
            membership_df.to_csv(path_csv, index=False)

    except Exception as _e:
        if print_cluster_info:
            print(f"[WARN] Could not compute/print cluster info or export membership: {_e}")

    return {
        "A": A.tocsr(),
        "D": D,
        "S": S,
        "layout_umap": X_umap,
        "layout_mds": X_mds,
        "mds_evr": evr_mds,
        "layout_pca": X_pca,
        "pca_evr": pca_evr,
        "clusters": z,
        "node_dataset": node_dataset,
        "node_gene": node_gene,
        "node_names": node_names,
        "dataset_names": dataset_names,
        "cluster_info_df": cluster_info_df,
        "membership_df": membership_df,
        "distance_kind": distance_kind,
        "distance_label": distance_label,
        "uses_covariance": uses_covariance,
        "primary_distance_source": primary_distance_source,
        "fig_umap": fig_umap,
        "axs_umap": axs_umap,
        "fig_mds": fig_mds,
        "axs_mds": axs_mds,
        "fig_pca": fig_pca,
        "axs_pca": axs_pca,
        "fig": fig_umap,
        "axs": axs_umap,
    }


def plot_pca_grid_by_condition(
    data: dict,
    *,
    group_order: Tuple[str, ...] = ("Control", "FSHD1", "DEL5"),
    group_palette: Optional[Dict[str, Any]] = None,
    avg_suffix: str = "_avg",
    use_scaled: bool = True,
    # Gene annotation
    genes_to_label: Optional[Sequence[str]] = None,
    max_gene_labels: Optional[int] = None,
    label_goi: bool = False,
    goi_text_size: float = 8.5,
    goi_marker_scale: float = 1.7,
    goi_darken: float = 0.50,
    # Plot controls
    node_size: float = 12.0,
    node_alpha: float = 0.90,
    figsize: tuple[float, float] = (12.5, 14.0),
    save_pdf: Optional[str] = None,
):
    """
    Plot per-dataset PCA projections of gene-level Minkowski profiles in a grid
    organised by dataset group.

    Each subplot corresponds to one dataset. Within a given subplot, rows of the
    feature matrix represent genes and columns represent flattened Minkowski
    features. PCA is therefore performed independently for each dataset, across
    genes, and the first two principal components are displayed.

    The layout is organised column-wise according to ``group_order``. For each
    group, the first row is reserved for the corresponding averaged dataset,
    identified by ``avg_suffix``, and the following rows contain the remaining
    datasets assigned to that group.

    This function is designed to be generic. It does not assume any particular
    biological context or naming convention beyond what is provided in
    ``group_order``, ``avg_suffix``, and the metadata already present in
    ``data``.

    Parameters
    ----------
    data:
        Dictionary containing at least

            - ``"conditions"``: sequence of dataset labels
            - ``"gene_names"``: sequence of gene names
            - ``"tensor_scaled"`` or ``"tensor_per_sample"``:
              ndarray of shape ``(F, G, 4, L)``

        Optional metadata such as ``"dataset_condition"``,
        ``data["extra_knowledge"]["condition_mapping"]``, and
        ``"gene_is_goi"`` are used when available.

    group_order:
        Ordered tuple defining the grid columns. Each entry corresponds to one
        dataset group.

    group_palette:
        Optional mapping from group name to plotting colour. If omitted,
        Matplotlib default colours are assigned automatically.

    avg_suffix:
        Suffix used to identify averaged datasets.

    use_scaled:
        If ``True``, use ``data["tensor_scaled"]``. Otherwise use
        ``data["tensor_per_sample"]``.

    genes_to_label:
        Optional list of genes to highlight. If omitted, ``data["gene_is_goi"]``
        is used when available.

    max_gene_labels:
        Optional cap on the number of highlighted base genes.

    label_goi:
        If ``True``, write the names of highlighted genes next to their points.

    goi_text_size, goi_marker_scale, goi_darken:
        Styling parameters for highlighted genes.

    node_size, node_alpha:
        Marker size and transparency for the background gene cloud.

    figsize:
        Figure size in inches.

    save_pdf:
        Optional output path for saving the figure as a PDF. A ``.pdf`` suffix
        is appended automatically if missing.

    Returns
    -------
    fig, axes, results
        Matplotlib figure, axes array, and a dictionary containing per-dataset
        PCA coordinates, explained variance ratios, and dataset indices.

    Notes
    -----
    The function is robust to cases where ``data["conditions"]`` includes
    averaged datasets but ``data["dataset_condition"]`` has not been updated
    accordingly. In that case, group labels are inferred from available
    metadata and dataset names.
    """
    if data is None:
        raise ValueError("`data` is None.")

    for key in ("conditions", "gene_names"):
        if key not in data:
            raise KeyError(f"`data` must contain '{key}'.")

    Xkey = "tensor_scaled" if use_scaled else "tensor_per_sample"
    if Xkey not in data:
        raise KeyError(f"`data['{Xkey}']` is missing.")

    X4_all = np.asarray(data[Xkey])  # (F, G, 4, L)
    if X4_all.ndim != 4 or X4_all.shape[2] != 4:
        raise ValueError(f"Expected {Xkey} with shape (F, G, 4, L), got {X4_all.shape}.")

    conds_all = list(map(str, data["conditions"]))
    gene_names = list(map(str, data["gene_names"]))

    F, G, S, L = X4_all.shape
    d_feat = S * L

    if len(conds_all) != F:
        raise ValueError(
            f"len(data['conditions']) = {len(conds_all)} must match F = {F} in {Xkey}."
        )

    # ------------------------------------------------------------------
    # Colour palette
    # ------------------------------------------------------------------
    if group_palette is None:
        cmap = plt.get_cmap("tab10")
        group_palette = {grp: cmap(i % 10) for i, grp in enumerate(group_order)}

    # ------------------------------------------------------------------
    # Gene-of-interest mask
    # ------------------------------------------------------------------
    goi_mask = np.zeros(G, dtype=bool)
    if genes_to_label is not None:
        wanted = set(map(str, genes_to_label))
        name2idx = {nm: i for i, nm in enumerate(gene_names)}
        for nm in wanted:
            if nm in name2idx:
                goi_mask[name2idx[nm]] = True
    elif "gene_is_goi" in data:
        gm = np.asarray(data["gene_is_goi"], dtype=bool)
        if gm.shape[0] == G:
            goi_mask = gm.copy()

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _lighten(color, amount: float = 0.35):
        """Return a lighter version of a Matplotlib-compatible colour."""
        r, g, b = mcolors.to_rgb(color)
        return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

    def _darken(color, amount: float = 0.35):
        """Return a darker version of a Matplotlib-compatible colour."""
        r, g, b = mcolors.to_rgb(color)
        return (r * (1 - amount), g * (1 - amount), b * (1 - amount))

    def _pca_2d(X: np.ndarray):
        """
        Compute the first two principal components using an SVD-based PCA.

        The input matrix is centred column-wise. The returned explained
        variance ratios correspond to the first two components.
        """
        X = np.asarray(X, dtype=float)
        X = X - X.mean(axis=0, keepdims=True)
        U, s, _ = np.linalg.svd(X, full_matrices=False)

        if s.size == 0:
            return np.zeros((X.shape[0], 2), dtype=np.float32), (0.0, 0.0)

        expl_var = (s ** 2) / max(X.shape[0] - 1, 1)
        total = float(expl_var.sum()) if expl_var.size else 0.0
        evr_all = expl_var / (total if total > 0 else 1.0)

        scores = U * s
        if scores.shape[1] >= 2:
            Z = scores[:, :2]
            evr = (float(evr_all[0]), float(evr_all[1]))
        elif scores.shape[1] == 1:
            Z = np.column_stack([scores[:, 0], np.zeros(scores.shape[0])])
            evr = (float(evr_all[0]), 0.0)
        else:
            Z = np.zeros((scores.shape[0], 2), dtype=float)
            evr = (0.0, 0.0)

        # Enforce a deterministic sign convention.
        for j in range(2):
            if np.sum(Z[:, j]) < 0:
                Z[:, j] *= -1

        return Z.astype(np.float32), evr

    # ------------------------------------------------------------------
    # Build dataset-to-group mapping
    # ------------------------------------------------------------------
    ds_groups_raw = data.get("dataset_condition", None)
    condition_mapping = None
    if isinstance(data.get("extra_knowledge", None), dict):
        condition_mapping = data["extra_knowledge"].get("condition_mapping", None)

    if ds_groups_raw is not None:
        ds_groups_raw = list(map(str, ds_groups_raw))

    def _infer_group(ds_name: str) -> str:
        """
        Infer a dataset group from available metadata and dataset naming.

        This fallback is used when explicit dataset-group annotations are
        missing or inconsistent with the current list of datasets.
        """
        if isinstance(condition_mapping, dict) and ds_name in condition_mapping:
            return str(condition_mapping[ds_name])

        if ds_name.endswith(avg_suffix):
            base = ds_name[:-len(avg_suffix)]
            if isinstance(condition_mapping, dict) and base in condition_mapping:
                return str(condition_mapping[base])

            for gname in group_order:
                if base.lower() == gname.lower():
                    return gname
            for gname in group_order:
                if gname.lower() in base.lower():
                    return gname

        for gname in group_order:
            if gname.lower() in ds_name.lower():
                return gname

        return "UNKNOWN"

    if ds_groups_raw is not None and len(ds_groups_raw) == len(conds_all):
        ds_groups = ds_groups_raw
    else:
        ds_groups = [_infer_group(ds) for ds in conds_all]

    # ------------------------------------------------------------------
    # Collect datasets by group and split averaged vs non-averaged datasets
    # ------------------------------------------------------------------
    group_to_ds = {g: [] for g in group_order}
    for ds_name, grp in zip(conds_all, ds_groups):
        if grp in group_to_ds:
            group_to_ds[grp].append(ds_name)

    group_to_avg = {g: None for g in group_order}
    group_to_rep = {g: [] for g in group_order}

    for g in group_order:
        for ds in group_to_ds[g]:
            if ds.endswith(avg_suffix):
                if group_to_avg[g] is None:
                    group_to_avg[g] = ds
            else:
                group_to_rep[g].append(ds)

    max_rep = max(len(group_to_rep[g]) for g in group_order) if group_order else 0
    nrows = 1 + max_rep
    ncols = len(group_order)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )

    results = {}

    # ------------------------------------------------------------------
    # Per-panel plotting function
    # ------------------------------------------------------------------
    def _plot_one(ax, ds_name: str, group_name: str):
        """
        Plot the gene-level PCA for a single dataset in one subplot.
        """
        f_idx = conds_all.index(ds_name)
        prof = X4_all[f_idx]        # (G, 4, L)
        X = prof.reshape(G, d_feat) # (G, 4*L)

        # Replace non-finite values by column means.
        if not np.isfinite(X).all():
            X = X.astype(float)
            bad = ~np.isfinite(X)
            col_means = np.nanmean(np.where(bad, np.nan, X), axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            ii = np.where(bad)
            X[ii] = col_means[ii[1]]

        Z, evr = _pca_2d(X)

        base = group_palette.get(group_name, "gray")
        col_non = _lighten(base, 0.35)
        col_goi = _darken(col_non, goi_darken)

        ax.scatter(
            Z[:, 0],
            Z[:, 1],
            s=node_size,
            c=[col_non],
            alpha=node_alpha,
            linewidths=0,
        )

        goi_idx = np.where(goi_mask)[0]
        if max_gene_labels is not None:
            goi_idx = goi_idx[:int(max_gene_labels)]

        if goi_idx.size:
            ax.scatter(
                Z[goi_idx, 0],
                Z[goi_idx, 1],
                s=node_size * (goi_marker_scale ** 2),
                c=[col_goi],
                alpha=1.0,
                linewidths=0,
                zorder=5,
            )

            if label_goi:
                for gidx in goi_idx:
                    ax.text(
                        Z[gidx, 0],
                        Z[gidx, 1],
                        gene_names[gidx],
                        fontsize=float(goi_text_size),
                        ha="left",
                        va="bottom",
                        color="black",
                        zorder=6,
                    )

        evr1, evr2 = evr
        ax.set_title(ds_name, fontsize=11.5)
        ax.set_xlabel(f"PC1 (EVR {100 * evr1:.0f}%)", fontsize=13)
        ax.set_ylabel(f"PC2 (EVR {100 * evr2:.0f}%)", fontsize=13)
        ax.grid(False)

        # Keep independent autoscaling for each subplot.
        ax.relim()
        ax.autoscale_view()

        results[ds_name] = {
            "coords": Z,
            "evr": (evr1, evr2),
            "dataset_index": f_idx,
        }

    # ------------------------------------------------------------------
    # Fill the grid
    # ------------------------------------------------------------------
    for col, grp in enumerate(group_order):
        # First row: averaged dataset, if present.
        ax = axes[0, col]
        ds_avg = group_to_avg.get(grp, None)
        if ds_avg is None:
            ax.set_axis_off()
        else:
            _plot_one(ax, ds_avg, grp)

        # Remaining rows: non-averaged datasets.
        reps = group_to_rep[grp]
        for r in range(max_rep):
            ax = axes[1 + r, col]
            if r >= len(reps):
                ax.set_axis_off()
            else:
                _plot_one(ax, reps[r], grp)

    # ------------------------------------------------------------------
    # Column headers
    # ------------------------------------------------------------------
    for col, grp in enumerate(group_order):
        if axes[0, col].axison:
            axes[0, col].text(
                0.5,
                1.15,
                grp,
                transform=axes[0, col].transAxes,
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
            )

    if save_pdf is not None:
        path = save_pdf if str(save_pdf).lower().endswith(".pdf") else (str(save_pdf) + ".pdf")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.35)

    plt.show()
    return fig, axes, results

def plot_top_changing_genes(
    data: dict,
    dataset_A: str,
    dataset_B: str,
    *,
    density_field: str = "normalized_density",
    top_k: int = 30,

    # Dot and bubble encoding
    dot_min_size: float = 30.0,
    dot_max_size: float = 550.0,
    size_pctl: float = 95.0,

    # Gene-of-interest handling
    goi_field: str = "gene_is_goi",
    goi_names: Optional[Sequence[str]] = None,

    # Title and export
    title: Optional[str] = None,
    save_csv: Optional[str] = None,
    save_pdf: Optional[str] = None,

    # Optional KDE band showing the same-gene distance distribution
    show_distance_kde: bool = True,
    kde_band_height: float = 3.0,
    kde_gap: float = 1.0,
    kde_bins: int = 256,
    kde_sigma_bins: float = 2.0,
    kde_alpha: float = 0.35,
    kde_facecolor: str = "0.25",

    # Percentile lines drawn on the KDE band
    percentile_lines: tuple = (50, 90, 95),
    p50_style: dict | None = None,
    p90_style: dict | None = None,
    p95_style: dict | None = None,

    # Axes styling
    xlim_max: Optional[float] = None,

    # Marker styling
    marker_edgecolor: str = "black",
    marker_linewidth: float = 0.6,
    marker_alpha: float = 0.95,

    # log2 fold-change computation
    logfc_eps: float = 1e-12,

    # Optional log2FC filtering
    log2fc_range: Optional[Tuple[float, float]] = None,

    # Annotation formatting
    annotate_log2fc: bool = True,
    annotate_fontsize: float = 11,
    annotate_sep: str = ", ",

    # Printing of percentile values
    print_percentile_values: bool = True,
) -> Dict[str, Any]:
    """
    Rank genes by same-gene distance between two datasets and visualise the
    strongest changes as a bubble plot.

    For each gene g, the ranking statistic is the distance between gene g in
    `dataset_A` and the same gene g in `dataset_B`. Distances are extracted
    from a precomputed gene-by-gene distance tensor stored in the input
    dictionary. If Wasserstein distances are available, they are used by
    priority. Otherwise, Euclidean distances are used.

    The bubble plot uses:
        - x-position: same-gene distance
        - bubble colour: sign of the percentage change in `density_field`
        - bubble size: magnitude of the percentage change, clipped at
          percentile `size_pctl`

    Percentage change and log2 fold-change are always computed from
    `density_field`, independently of the distance metric used for ranking.

    Parameters
    ----------
    data:
        Dictionary containing at least:
            - a gene-distance payload under `data["gene_distances"]`
              or, for backward compatibility, `data["w2_allpairs"]`
            - `gene_names`
            - `conditions`
            - `density_field`

    dataset_A, dataset_B:
        Names of the two datasets to compare.

    density_field:
        Name of the 2D array in `data` used to compute both percentage change
        and log2 fold-change.

    top_k:
        Number of top-ranked genes to display.

    dot_min_size, dot_max_size:
        Minimum and maximum bubble sizes.

    size_pctl:
        Percentile used to clip the absolute percentage change when mapping
        values to bubble sizes.

    goi_field:
        Name of the boolean field in `data` indicating genes of interest.

    goi_names:
        Optional explicit list of genes of interest. If provided, it overrides
        `goi_field`.

    title:
        Optional figure title. If omitted, a default title is constructed.

    save_csv:
        Optional path for exporting the filtered ranking table as CSV.

    save_pdf:
        Optional path for exporting the figure as PDF.

    show_distance_kde:
        If True, draw a one-dimensional KDE-style band representing the
        same-gene distance distribution over all retained genes.

    kde_band_height, kde_gap, kde_bins, kde_sigma_bins, kde_alpha, kde_facecolor:
        Parameters controlling the KDE band and its histogram-based fallback.

    percentile_lines:
        Percentiles of the same-gene distance distribution to display on the
        KDE band.

    p50_style, p90_style, p95_style:
        Optional Matplotlib style dictionaries for the 50th, 90th and 95th
        percentile lines.

    xlim_max:
        Optional upper bound for the x-axis.

    marker_edgecolor, marker_linewidth, marker_alpha:
        Bubble styling parameters.

    logfc_eps:
        Pseudo-count added when computing log2 fold-change.

    log2fc_range:
        Optional interval `(lo, hi)`. If provided, only genes with
        `lo <= log2FC <= hi` are retained for plotting and CSV export.

    annotate_log2fc:
        If True, include log2FC in the per-dot annotation text.

    annotate_fontsize:
        Font size for per-dot annotations.

    annotate_sep:
        Separator string used between annotation components.

    print_percentile_values:
        If True, print the numerical values of the percentile lines drawn on
        the KDE band.

    Returns
    -------
    dict
        Dictionary containing the same-gene distances, density-based
        statistics, the applied mask, the filtered rank order, and the
        percentile values used for the KDE band.
    """
    # ------------------------------------------------------------------
    # Resolve the distance payload, prioritising the current API
    # ------------------------------------------------------------------
    payload = None
    payload_name = None

    if "gene_distances" in data and isinstance(data["gene_distances"], dict):
        payload = data["gene_distances"]
        payload_name = "gene_distances"
    elif "w2_allpairs" in data and isinstance(data["w2_allpairs"], dict):
        payload = data["w2_allpairs"]
        payload_name = "w2_allpairs"
    else:
        raise KeyError(
            "No gene-distance payload found. Expected either "
            "`data['gene_distances']` or `data['w2_allpairs']`."
        )

    for key in ("D", "labels", "shapes"):
        if key not in payload:
            raise KeyError(
                f"`data['{payload_name}']` must contain 'D', 'labels', and 'shapes'."
            )

    # ------------------------------------------------------------------
    # Determine which distance metric is stored
    # ------------------------------------------------------------------
    distance_kind_raw = str(payload.get("distance_kind", "")).lower()
    kind_raw = str(payload.get("kind", "")).lower()
    uses_covariance = payload.get("uses_covariance", None)

    if distance_kind_raw:
        if "wasserstein" in distance_kind_raw or distance_kind_raw == "w2":
            distance_kind = "wasserstein"
        elif "euclidean" in distance_kind_raw:
            distance_kind = "euclidean"
        else:
            distance_kind = "wasserstein" if uses_covariance else "euclidean"
    else:
        if "euclidean" in kind_raw or uses_covariance is False:
            distance_kind = "euclidean"
        else:
            distance_kind = "wasserstein"

    if distance_kind == "wasserstein":
        distance_label = "2-Wasserstein distance"
        short_distance_label = "W2"
    else:
        distance_label = "Euclidean distance"
        short_distance_label = "Euclidean"

    # ------------------------------------------------------------------
    # Validate required auxiliary inputs
    # ------------------------------------------------------------------
    if "gene_names" not in data:
        raise KeyError("`data` must contain 'gene_names'.")
    genes = list(map(str, data["gene_names"]))
    G = len(genes)

    if density_field not in data:
        raise KeyError(f"{density_field!r} not found in data.")
    dens = np.asarray(data[density_field], dtype=float)
    if dens.ndim != 2 or dens.shape[1] != G:
        raise ValueError(f"{density_field!r} must have shape (n_datasets, G). Got {dens.shape}.")

    if "conditions" not in data:
        raise KeyError("`data` must contain 'conditions'.")
    conds_all = list(map(str, data["conditions"]))

    labels = list(map(str, payload["labels"]))
    try:
        aK = labels.index(dataset_A)
        bK = labels.index(dataset_B)
    except ValueError as exc:
        raise ValueError(
            f"Both datasets must be present in data['{payload_name}']['labels']: {labels}"
        ) from exc

    try:
        aF = conds_all.index(dataset_A)
        bF = conds_all.index(dataset_B)
    except ValueError as exc:
        raise ValueError(f"Dataset not found in data['conditions']: {exc}") from exc

    D4 = np.asarray(payload["D"], dtype=float)
    if int(payload["shapes"]["G"]) != G:
        raise ValueError("Mismatch between the gene-distance tensor and data['gene_names'].")

    # ------------------------------------------------------------------
    # Extract same-gene distances
    # ------------------------------------------------------------------
    D_between = D4[aK, bK]
    same_gene_distance = np.diag(D_between).astype(float)

    # ------------------------------------------------------------------
    # Extract densities and compute percentage change and log2FC
    # ------------------------------------------------------------------
    rho_A = dens[aF].astype(float, copy=False)
    rho_B = dens[bF].astype(float, copy=False)

    eps_pct = 0.0
    denom = rho_B.copy()
    pct_gain = np.full(G, np.nan, dtype=float)

    both_zero = (np.abs(rho_A) <= eps_pct) & (np.abs(rho_B) <= eps_pct)
    safe = ~both_zero & (np.abs(denom) > eps_pct)

    pct_gain[both_zero] = 0.0
    pct_gain[safe] = 100.0 * (rho_A[safe] - rho_B[safe]) / denom[safe]

    rho_A_pos = rho_A.copy()
    rho_B_pos = rho_B.copy()
    rho_A_pos[rho_A_pos < 0] = np.nan
    rho_B_pos[rho_B_pos < 0] = np.nan
    log2fc = np.log2((rho_A_pos + float(logfc_eps)) / (rho_B_pos + float(logfc_eps)))

    # ------------------------------------------------------------------
    # Define the gene-of-interest mask
    # ------------------------------------------------------------------
    if goi_names is not None:
        goi_set = set(map(str, goi_names))
        goi_mask = np.array([g in goi_set for g in genes], dtype=bool)
    elif goi_field in data and isinstance(data[goi_field], np.ndarray):
        mask = np.asarray(data[goi_field]).astype(bool)
        if mask.shape != (G,):
            raise ValueError(f"{goi_field} must have shape ({G},).")
        goi_mask = mask
    else:
        goi_mask = np.zeros(G, dtype=bool)

    # ------------------------------------------------------------------
    # Apply optional log2FC filtering
    # ------------------------------------------------------------------
    keep = np.isfinite(same_gene_distance) & np.isfinite(log2fc)

    if log2fc_range is not None:
        lo, hi = float(log2fc_range[0]), float(log2fc_range[1])
        if lo > hi:
            lo, hi = hi, lo
        keep &= (log2fc >= lo) & (log2fc <= hi)

    kept_idx = np.where(keep)[0]
    if kept_idx.size == 0:
        msg = "No genes left after filtering."
        if log2fc_range is not None:
            msg += f" (log2fc_range={log2fc_range})"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Rank genes by decreasing same-gene distance
    # ------------------------------------------------------------------
    dist_kept = same_gene_distance[kept_idx]
    order_kept_desc = kept_idx[np.argsort(dist_kept)[::-1]]

    k = min(int(top_k), int(order_kept_desc.size))
    top_idx = order_kept_desc[:k]

    top_genes = [genes[i] for i in top_idx]
    top_vals = same_gene_distance[top_idx]
    top_pct = pct_gain[top_idx]
    top_log2fc = log2fc[top_idx]
    top_goi = goi_mask[top_idx]

    finite_all = same_gene_distance[kept_idx]
    finite_all = finite_all[np.isfinite(finite_all)]

    # ------------------------------------------------------------------
    # Map percentage change to bubble colour and size
    # ------------------------------------------------------------------
    colors = np.full(k, "gray", dtype=object)
    colors[np.isfinite(top_pct) & (top_pct > 0)] = "blue"
    colors[np.isfinite(top_pct) & (top_pct < 0)] = "red"

    abs_pg = np.abs(top_pct)
    finite_abs = abs_pg[np.isfinite(abs_pg)]

    if finite_abs.size > 0:
        ref = np.percentile(finite_abs, float(size_pctl))
        if not (np.isfinite(ref) and ref > 0):
            ref = float(np.nanmax(finite_abs)) if np.nanmax(finite_abs) > 0 else 1.0
    else:
        ref = 1.0

    t = np.clip(abs_pg / ref, 0.0, 1.0)
    sizes = dot_min_size + t * (dot_max_size - dot_min_size)

    if finite_abs.size > 0:
        med = float(np.nanmedian(finite_abs))
        t_med = 0.0 if ref == 0 else np.clip(med / ref, 0.0, 1.0)
        s_med = dot_min_size + t_med * (dot_max_size - dot_min_size)
    else:
        s_med = 0.5 * (dot_min_size + dot_max_size)

    sizes[~np.isfinite(abs_pg)] = s_med

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig_h = max(4.2, 0.28 * k + 2.0)
    fig, ax = plt.subplots(figsize=(11.0, fig_h * 1.5))

    if xlim_max is not None:
        xlim_right = float(xlim_max)
    else:
        xmax_all = float(np.nanmax(finite_all)) if finite_all.size else 1.0
        xlim_right = xmax_all * 1.10 if xmax_all > 0 else 1.0

    y = np.arange(k)

    ax.scatter(
        top_vals,
        y,
        s=sizes,
        c=colors,
        alpha=marker_alpha,
        edgecolors=marker_edgecolor,
        linewidths=marker_linewidth,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(top_genes, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel(distance_label, fontsize=17)
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0.0, xlim_right)

    # ------------------------------------------------------------------
    # Annotate each displayed gene
    # ------------------------------------------------------------------
    dx = 0.03 * xlim_right
    for yy, (v, pg, lf) in enumerate(zip(top_vals, top_pct, top_log2fc)):
        if np.isfinite(pg):
            txt = f"Δρ: {pg:+.0f}%"
        else:
            txt = "Δρ: NA"

        if annotate_log2fc:
            if np.isfinite(lf):
                txt += f"{annotate_sep}log2FC: {lf:+.2f}"
            else:
                txt += f"{annotate_sep}log2FC: NA"

        ax.text(v + dx, yy, txt, va="center", ha="left", fontsize=annotate_fontsize)

    for txt, is_goi in zip(ax.get_yticklabels(), top_goi):
        if bool(is_goi):
            txt.set_fontweight("bold")

    # ------------------------------------------------------------------
    # Optional KDE band for the same-gene distance distribution
    # ------------------------------------------------------------------
    pct_values = None
    if show_distance_kde and finite_all.size >= 1:
        y0 = -(kde_gap + kde_band_height)
        xgrid = np.linspace(0.0, xlim_right, 512)

        try:
            from scipy.stats import gaussian_kde
            if np.allclose(finite_all, finite_all[0]):
                raise RuntimeError("Degenerate KDE")
            kde = gaussian_kde(finite_all)
            raw = kde(xgrid)
            m = raw.max() if raw.size and np.isfinite(raw).any() else 1.0
            dens_y = (raw / m) if (m > 0) else np.zeros_like(raw)
        except Exception:
            hist, edges = np.histogram(
                finite_all,
                bins=int(kde_bins),
                range=(0.0, xlim_right),
                density=True,
            )
            centers = 0.5 * (edges[:-1] + edges[1:])
            sigma = max(1.0, float(kde_sigma_bins))
            rad = int(4 * sigma)
            tt = np.arange(-rad, rad + 1, dtype=float)
            kernel = np.exp(-0.5 * (tt / sigma) ** 2)
            kernel /= kernel.sum()
            smoothed = np.convolve(hist, kernel, mode="same")
            raw = np.interp(xgrid, centers, smoothed, left=0.0, right=0.0)
            m = raw.max() if raw.size and np.isfinite(raw).any() else 1.0
            dens_y = (raw / m) if (m > 0) else np.zeros_like(raw)

        y_curve = y0 + kde_band_height * dens_y
        ax.fill_between(
            xgrid,
            y0,
            y_curve,
            color=kde_facecolor,
            alpha=kde_alpha,
            linewidth=0,
            zorder=1,
        )

        ax.set_ylim(k - 0.5, y0 - 0.1)

        if p50_style is None:
            p50_style = dict(color="black", linestyle="-", linewidth=1.2, alpha=0.75)
        if p90_style is None:
            p90_style = dict(color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        if p95_style is None:
            p95_style = dict(color="black", linestyle=":", linewidth=1.0, alpha=0.6)

        pct_values = {}
        for p in percentile_lines:
            try:
                pct_values[int(p)] = float(np.percentile(finite_all, float(p)))
            except Exception:
                pct_values[int(p)] = np.nan

        if print_percentile_values:
            parts = []
            for p in percentile_lines:
                xv = pct_values[int(p)]
                parts.append(f"p{int(p)}={xv:.6g}" if np.isfinite(xv) else f"p{int(p)}=NA")
            print(
                f"[plot_top_changing_genes] {short_distance_label} percentile lines: "
                + ", ".join(parts)
            )

        if 50 in percentile_lines:
            xval = pct_values.get(50, np.nan)
            if np.isfinite(xval):
                ax.axvline(xval, **p50_style)
        if 90 in percentile_lines:
            xval = pct_values.get(90, np.nan)
            if np.isfinite(xval):
                ax.axvline(xval, **p90_style)
        if 95 in percentile_lines:
            xval = pct_values.get(95, np.nan)
            if np.isfinite(xval):
                ax.axvline(xval, **p95_style)

    # ------------------------------------------------------------------
    # Legend and title
    # ------------------------------------------------------------------
    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markeredgecolor=marker_edgecolor,
            markersize=9,
            label="density variation > 0",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markeredgecolor=marker_edgecolor,
            markersize=9,
            label="density variation < 0",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor=marker_edgecolor,
            markersize=9,
            label="density variation NA/0",
        ),
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.2, label="median"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="90th pct"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.0, label="95th pct"),
    ]

    if title is None:
        title = f"{dataset_A} vs {dataset_B}"
        if log2fc_range is not None:
            title += f" | log2FC in [{float(log2fc_range[0]):g}, {float(log2fc_range[1]):g}]"

    ax.set_title(title, fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.4)

    fig.legend(
        handles=legend_elems,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=1,
        frameon=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="0.6",
        fancybox=True,
        fontsize=12,
    )

    if save_pdf is not None:
        pdf_path = save_pdf if str(save_pdf).lower().endswith(".pdf") else (str(save_pdf) + ".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.5)

    plt.show()

    # ------------------------------------------------------------------
    # Optional CSV export
    # ------------------------------------------------------------------
    if save_csv is not None:
        try:
            import pandas as pd
        except Exception as exc:
            raise ImportError("Saving CSV requires pandas. Install with `pip install pandas`.") from exc

        kept_order = kept_idx[np.argsort(same_gene_distance[kept_idx])[::-1]]
        ranks_full = np.full(G, -1, dtype=int)
        ranks_full[kept_order] = np.arange(1, kept_order.size + 1, dtype=int)

        df = pd.DataFrame({
            "rank": ranks_full,
            "gene": genes,
            "same_gene_distance": same_gene_distance,
            "distance_kind": distance_kind,
            "pct_gain": pct_gain,
            "log2FC": log2fc,
            f"{density_field}_{dataset_A}": rho_A,
            f"{density_field}_{dataset_B}": rho_B,
            "is_GOI": goi_mask.astype(bool),
            "kept_by_log2fc_range": keep.astype(bool),
        })

        if distance_kind == "wasserstein":
            df["W2_same_gene"] = same_gene_distance

        df = df[df["kept_by_log2fc_range"]].copy()
        df = df.sort_values("rank", ascending=True)
        df.to_csv(save_csv, index=False)

    out = {
        "genes": genes,
        "same_gene_distance": same_gene_distance,
        "distance_kind": distance_kind,
        "distance_label": distance_label,
        "pct_gain": pct_gain,
        "log2fc": log2fc,
        "kept_mask": keep,
        "idx_A": aK,
        "idx_B": bK,
        "order_desc_filtered": kept_idx[np.argsort(same_gene_distance[kept_idx])[::-1]],
        "distance_percentiles": pct_values,
    }

    if distance_kind == "wasserstein":
        out["W2_same_gene"] = same_gene_distance
        out["w2_percentiles"] = pct_values

    return out

def plot_w2_abslog2fc_with_trend(
    data: dict,
    dataset_A: str,
    dataset_B: str,
    *,
    density_field: str = "normalized_density",
    logfc_eps: float = 1e-12,
    drop_w2_nonpositive: bool = True,
    xscale: str = "log",
    nbins: int = 30,
    binning: str = "log",
    show_iqr: bool = True,
    show_regression: bool = False,
    title: Optional[str] = None,
    save_pdf: Optional[str] = None,
    point_size: float = 10.0,
    point_alpha: float = 0.50,
) -> Dict[str, Any]:
    """
    Plot same-gene 2-Wasserstein distance against absolute log2 fold-change,
    together with a binned trend curve and an optional interquartile band.

    For each gene, the x-coordinate is the same-gene 2-Wasserstein distance
    between ``dataset_A`` and ``dataset_B``, extracted from
    ``data["w2_allpairs"]["D"]`` as the diagonal of the corresponding
    between-dataset block. The y-coordinate is

        |log2FC| = |log2((rho_A + eps) / (rho_B + eps))|,

    where ``rho_A`` and ``rho_B`` are taken from ``data[density_field]`` and
    ``eps = logfc_eps`` is a pseudo-count used to avoid numerical instability
    near zero.

    The plot contains three main elements.

    First, a scatter plot of all retained genes.

    Secondly, a binned trend curve showing the median ``|log2FC|`` as a
    function of the Wasserstein distance. When ``show_iqr=True``, the
    interquartile range within each bin is displayed as a shaded band.

    Thirdly, an optional regression line can be added for the linear model

        |log2FC| = intercept + slope * log10(W2),

    which is often less informative than the binned trend when the spread
    increases with Wasserstein distance.

    The function also reports the Pearson correlation between ``W2`` and
    ``|log2FC|``.

    Parameters
    ----------
    data:
        Dictionary containing at least

            - ``"w2_allpairs"``
            - ``"gene_names"``
            - ``density_field``
            - ``"conditions"``

    dataset_A, dataset_B:
        Names of the two datasets to compare.

    density_field:
        Name of the array in ``data`` used to compute log2 fold-changes.

    logfc_eps:
        Pseudo-count added to both densities when computing log2 fold-change.

    drop_w2_nonpositive:
        If ``True``, remove genes with non-positive Wasserstein distance before
        plotting and computing statistics.

    xscale:
        Scale of the x-axis. Allowed values are ``"log"``, ``"linear"``, and
        ``"symlog"``.

    nbins:
        Number of bins used for the trend curve.

    binning:
        Binning scheme for the x-axis. Allowed values are ``"log"`` and
        ``"linear"``.

    show_iqr:
        If ``True``, display the interquartile range around the binned median.

    show_regression:
        If ``True``, overlay a linear regression of ``|log2FC|`` against
        ``log10(W2)``.

    title:
        Optional figure title. If omitted, a default title based on the two
        dataset names is used.

    save_pdf:
        Optional output path for saving the figure as a PDF.

    point_size, point_alpha:
        Marker size and transparency for the scatter plot.

    Returns
    -------
    dict
        Dictionary containing the Pearson correlation, the number of retained
        genes, the binned trend statistics, and the optional regression
        coefficients.

    Notes
    -----
    This function is generic and does not assume any particular dataset naming
    convention or biological context.
    """
    # ------------------------------------------------------------------
    # Validate required inputs and extract the relevant arrays
    # ------------------------------------------------------------------
    if "w2_allpairs" not in data or not isinstance(data["w2_allpairs"], dict):
        raise KeyError("`data['w2_allpairs']` not found.")

    W = data["w2_allpairs"]
    for key in ("D", "labels", "shapes"):
        if key not in W:
            raise KeyError("`data['w2_allpairs']` must contain 'D', 'labels', and 'shapes'.")

    if "gene_names" not in data:
        raise KeyError("`data` must contain 'gene_names'.")
    genes = list(map(str, data["gene_names"]))
    G = len(genes)

    if density_field not in data:
        raise KeyError(f"{density_field!r} not found in data.")
    dens = np.asarray(data[density_field], dtype=float)
    if dens.ndim != 2 or dens.shape[1] != G:
        raise ValueError(f"{density_field!r} must have shape (n_datasets, G). Got {dens.shape}.")

    if "conditions" not in data:
        raise KeyError("`data` must contain 'conditions'.")
    conds = list(map(str, data["conditions"]))

    labels = list(map(str, W["labels"]))
    try:
        aK = labels.index(dataset_A)
        bK = labels.index(dataset_B)
    except ValueError as exc:
        raise ValueError(
            f"Datasets must exist in data['w2_allpairs']['labels']: {labels}"
        ) from exc

    try:
        aF = conds.index(dataset_A)
        bF = conds.index(dataset_B)
    except ValueError as exc:
        raise ValueError(
            f"Datasets must exist in data['conditions']: {conds}"
        ) from exc

    D4 = np.asarray(W["D"], dtype=float)
    if int(W["shapes"]["G"]) != G:
        raise ValueError("Mismatch: w2_allpairs['shapes']['G'] != len(data['gene_names']).")

    # ------------------------------------------------------------------
    # Compute same-gene W2 and absolute log2 fold-change
    # ------------------------------------------------------------------
    w2 = np.diag(D4[aK, bK]).astype(float)

    rho_A = dens[aF].astype(float, copy=False)
    rho_B = dens[bF].astype(float, copy=False)

    # Negative densities are treated as invalid for fold-change computation.
    rho_A_pos = rho_A.copy()
    rho_B_pos = rho_B.copy()
    rho_A_pos[rho_A_pos < 0] = np.nan
    rho_B_pos[rho_B_pos < 0] = np.nan

    log2fc = np.log2((rho_A_pos + float(logfc_eps)) / (rho_B_pos + float(logfc_eps)))
    abslog2fc = np.abs(log2fc)

    m = np.isfinite(w2) & np.isfinite(abslog2fc)
    if drop_w2_nonpositive:
        m &= (w2 > 0)

    x = w2[m]
    y = abslog2fc[m]
    if x.size == 0:
        raise ValueError("No valid points remain after filtering.")

    # ------------------------------------------------------------------
    # Pearson correlation between W2 and |log2FC|
    # ------------------------------------------------------------------
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    pearson_r = (np.sum(x0 * y0) / denom) if denom > 0 else np.nan

    # ------------------------------------------------------------------
    # Define bin edges for the trend curve
    # ------------------------------------------------------------------
    nbins = int(max(5, nbins))
    binning = str(binning).lower()
    if binning not in ("log", "linear"):
        raise ValueError("binning must be 'log' or 'linear'.")

    if binning == "log":
        lx = np.log10(x)
        edges = np.linspace(lx.min(), lx.max(), nbins + 1)
        bin_id = np.digitize(lx, edges) - 1
    else:
        edges = np.linspace(x.min(), x.max(), nbins + 1)
        bin_id = np.digitize(x, edges) - 1

    # ------------------------------------------------------------------
    # Per-bin summaries
    # ------------------------------------------------------------------
    x_mid = []
    y_med = []
    y_q25 = []
    y_q75 = []
    counts = []

    for b in range(nbins):
        sel = bin_id == b
        if not np.any(sel):
            continue

        xb = x[sel]
        yb = y[sel]

        # The x-coordinate of the trend is taken as the bin-wise median W2,
        # which is generally more stable than the geometric or arithmetic bin
        # centre when occupancy is uneven.
        x_mid.append(float(np.median(xb)))
        y_med.append(float(np.median(yb)))
        y_q25.append(float(np.percentile(yb, 25)))
        y_q75.append(float(np.percentile(yb, 75)))
        counts.append(int(sel.sum()))

    x_mid = np.array(x_mid, dtype=float)
    y_med = np.array(y_med, dtype=float)
    y_q25 = np.array(y_q25, dtype=float)
    y_q75 = np.array(y_q75, dtype=float)
    counts = np.array(counts, dtype=int)

    # ------------------------------------------------------------------
    # Optional regression of |log2FC| against log10(W2)
    # ------------------------------------------------------------------
    reg = None
    if show_regression:
        z = np.log10(x)
        z0 = z - z.mean()
        denom2 = np.sum(z0 * z0)
        slope = (np.sum(z0 * (y - y.mean())) / denom2) if denom2 > 0 else np.nan
        intercept = float(y.mean() - slope * z.mean()) if np.isfinite(slope) else np.nan
        reg = (intercept, float(slope))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if title is None:
        title = f"{dataset_A} vs {dataset_B}"

    fig, ax = plt.subplots(figsize=(17, 6.2))

    ax.scatter(
        x,
        y,
        s=float(point_size),
        alpha=float(point_alpha),
        linewidths=0,
        zorder=2,
    )

    # Binned median trend.
    ax.plot(x_mid, y_med, linewidth=2.0, zorder=5)

    if show_iqr:
        ax.fill_between(x_mid, y_q25, y_q75, alpha=0.20, zorder=4)

    if show_regression and reg is not None and np.all(np.isfinite(reg)):
        intercept, slope = reg
        xline = np.linspace(x.min(), x.max(), 300)
        yline = intercept + slope * np.log10(xline)
        ax.plot(xline, yline, linestyle="--", linewidth=1.6, zorder=6)

    ax.set_xlabel("2-Wasserstein distance", fontsize=17)
    ax.set_ylabel("|log2 fold-change|", fontsize=17)
    ax.grid(alpha=0.25)

    xscale = str(xscale).lower()
    if xscale == "log":
        ax.set_xscale("log")
    elif xscale == "symlog":
        ax.set_xscale("symlog", linthresh=1e-6)
    elif xscale == "linear":
        pass
    else:
        raise ValueError("xscale must be one of {'log', 'linear', 'symlog'}.")

    ax.set_title(
        f"{title}\nPearson corr(W2, |log2FC|) = {pearson_r:.3f}   (n={x.size})",
        fontsize=17,
    )

    plt.tight_layout()

    if save_pdf is not None:
        path = save_pdf if str(save_pdf).lower().endswith(".pdf") else (str(save_pdf) + ".pdf")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)

    plt.show()

    return {
        "pearson_r_w2_abslog2fc": float(pearson_r) if np.isfinite(pearson_r) else np.nan,
        "n": int(x.size),
        "binned_x_median": x_mid,
        "binned_abslog2fc_median": y_med,
        "binned_abslog2fc_q25": y_q25,
        "binned_abslog2fc_q75": y_q75,
        "binned_counts": counts,
        "regression_abslog2fc_vs_log10w2": reg,
    }

def plot_gene_density_over_dapi(
    genes,
    transcript_files,
    image_files,
    *,
    titles=None,
    associated_rows: list[int] | None = None,
    # Transcript table columns
    gene_col: str = "gene",
    x_col: str = "global_x",
    y_col: str = "global_y",
    # Density rendering
    cmap: str = "magma",
    image_alpha: float = 1.0,
    overlay_alpha: float = 0.7,
    sigma: float | None = 20.0,
    # Spatial binning
    bins: int | tuple[int, int] | None = None,
    # Optional square crop in coordinate units
    crop_side_um: float | None = None,
    # Layout, used when associated_rows is not provided
    max_cols: int = 2,
    figsize_per_panel: tuple[float, float] = (6.0, 5.0),
    black_background: bool = True,
    # Shared colour scale and colour bar
    color_scale: str = "global_log",
    global_clip_percentiles: tuple[float, float] = (1.0, 99.5),
    threshold_frac: float = 0.03,
    show_colorbar: bool = True,
    colorbar_label: str = "Transcripts per μm²",
    colorbar_ticks: list[float] | None = None,
    colorbar_row_ratio: float = 0.06,
    # Scale bar
    draw_scalebar: bool = True,
    scalebar_um: float = 500.0,
    scalebar_color: str = "white",
    scalebar_width_px: float = 3.0,
    scalebar_margin_um: float = 200.0,
    scalebar_label: bool = True,
    scalebar_label_position: str = "above",
    scalebar_label_offset_um: float | None = None,
    scalebar_label_fontsize: float = 12.0,
    # Row alignment when associated_rows is provided
    assoc_row_align: str = "center",
    assoc_last_row_align: str = "right",
    # Export
    save_pdf: str | None = None,
    # Verbosity
    verbose: bool = True,
):
    """
    Plot smoothed transcript-density maps over DAPI images for one or more genes,
    using a shared colour scale across all panels.

    For each dataset, transcript coordinates are binned in the spatial frame of
    the corresponding image, optionally smoothed with a Gaussian kernel, and
    converted to transcript density in transcripts per square unit of the input
    coordinates. The resulting density map is overlaid on the DAPI image with a
    common colour scale, either linear or logarithmic, and an optional global
    horizontal colour bar.

    The function supports two layout modes. If ``associated_rows`` is omitted,
    panels are arranged automatically in a regular grid. If ``associated_rows``
    is provided, datasets are grouped by row according to the supplied integer
    labels, with configurable alignment within each row.

    Parameters
    ----------
    genes:
        Gene name or sequence of gene names to display. When several genes are
        provided, their transcripts are pooled into a single density map.

    transcript_files:
        Path or sequence of paths to transcript tables.

    image_files:
        Path or sequence of paths to the corresponding DAPI images.

    titles:
        Optional panel titles. If omitted, titles are derived from transcript
        file names.

    associated_rows:
        Optional list assigning each dataset to a row in the final figure.

    gene_col, x_col, y_col:
        Column names in the transcript tables containing the gene identifier and
        the x/y coordinates.

    cmap:
        Matplotlib colormap used for the density overlay.

    image_alpha:
        Opacity of the background DAPI image.

    overlay_alpha:
        Opacity of the density overlay.

    sigma:
        Standard deviation of the Gaussian smoothing kernel, expressed in binned
        pixel units. If ``None`` or non-positive, no smoothing is applied.

    bins:
        Spatial binning of transcript coordinates. If ``None``, use the image
        resolution. If an integer, downsample the image grid by that factor. If
        a tuple, interpret it as ``(nx, ny)``.

    crop_side_um:
        Optional side length of a centred square crop in the coordinate system
        of the transcript data. If omitted, use the full field of view.

    max_cols:
        Maximum number of columns when automatic layout is used.

    figsize_per_panel:
        Size of each panel in inches.

    black_background:
        If ``True``, render the figure on a black background.

    color_scale:
        Shared colour scale for all panels. Allowed values are
        ``"global_log"`` and ``"global_linear"``.

    global_clip_percentiles:
        Lower and upper percentiles used to define the global colour range from
        the full set of density values before thresholding.

    threshold_frac:
        Fraction of the global maximum used to mask low-density pixels before
        plotting.

    show_colorbar:
        If ``True``, draw a shared horizontal colour bar.

    colorbar_label:
        Label of the colour bar.

    colorbar_ticks:
        Optional explicit ticks for the colour bar.

    colorbar_row_ratio:
        Relative height of the colour-bar row in the GridSpec layout.

    draw_scalebar:
        If ``True``, draw a scale bar in each panel.

    scalebar_um:
        Length of the scale bar in the coordinate system of the input data.

    scalebar_color, scalebar_width_px, scalebar_margin_um:
        Visual parameters of the scale bar.

    scalebar_label:
        If ``True``, add a text label to the scale bar.

    scalebar_label_position:
        Position of the scale-bar label, either ``"above"`` or ``"below"``.

    scalebar_label_offset_um:
        Offset of the scale-bar label from the bar. If omitted, a default value
        proportional to the panel height is used.

    scalebar_label_fontsize:
        Font size of the scale-bar label.

    assoc_row_align, assoc_last_row_align:
        Alignment of grouped panels when ``associated_rows`` is provided. Each
        must be one of ``{"left", "center", "right"}``.

    save_pdf:
        Optional output path for PDF export.

    verbose:
        If ``True``, print a concise loading log.

    Returns
    -------
    dict
        Dictionary containing the figure, axes, panel extents, transcript
        counts, and the effective colour-scale parameters.

    Notes
    -----
    This function is generic and does not assume any specific dataset naming
    scheme or biological context.
    """

    def _v(message: str) -> None:
        """Print a message only when verbose logging is enabled."""
        if verbose:
            print(message, flush=True)

    # ------------------------------------------------------------------
    # Harmonise inputs
    # ------------------------------------------------------------------
    if isinstance(genes, str):
        genes = [genes]
    else:
        genes = list(map(str, genes))

    if isinstance(transcript_files, str):
        transcript_files = [transcript_files]
    if isinstance(image_files, str):
        image_files = [image_files]

    if len(transcript_files) != len(image_files):
        raise ValueError("`transcript_files` and `image_files` must have the same length.")

    n = len(transcript_files)
    if n == 0:
        raise ValueError("No transcript files were provided.")

    if titles is None:
        titles = [os.path.basename(path).replace(".csv", "") for path in transcript_files]
    if len(titles) != n:
        raise ValueError("`titles` must have the same length as `transcript_files` and `image_files`.")

    # ------------------------------------------------------------------
    # Optional row grouping
    # ------------------------------------------------------------------
    assoc_arr = None
    if associated_rows is not None:
        if len(associated_rows) != n:
            raise ValueError("`associated_rows` must have the same length as `transcript_files`.")
        assoc_arr = np.asarray(associated_rows, dtype=int)

        unique_rows = []
        for value in assoc_arr:
            ivalue = int(value)
            if ivalue not in unique_rows:
                unique_rows.append(ivalue)

        row_to_indices = {
            row_value: [i for i, a in enumerate(assoc_arr) if a == row_value]
            for row_value in unique_rows
        }
        rows_assoc = len(unique_rows)
        counts_per_row = [len(row_to_indices[row_value]) for row_value in unique_rows]
        cols_assoc = max(counts_per_row) if counts_per_row else 1

        valid_align = {"left", "center", "right"}
        if assoc_row_align not in valid_align:
            raise ValueError("`assoc_row_align` must be one of {'left', 'center', 'right'}.")
        if assoc_last_row_align not in valid_align:
            raise ValueError("`assoc_last_row_align` must be one of {'left', 'center', 'right'}.")
    else:
        unique_rows = None
        row_to_indices = None
        rows_assoc = None
        cols_assoc = None

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _genes_suptitle(glist, max_names: int = 6) -> str:
        """Build a compact figure title from the selected genes."""
        if len(glist) == 1:
            return f"Gene: {glist[0]}"
        head = ", ".join(glist[:max_names])
        if len(glist) > max_names:
            head += f", +{len(glist) - max_names} others"
        return f"Genes: {head}"

    def _load_dapi_and_edges(img_path, x_all, y_all):
        """
        Load a DAPI image and define the spatial bin edges used for density
        estimation in the coordinate system of the transcript table.
        """
        with tiff.TiffFile(img_path) as tif:
            dapi = tif.asarray()

        if dapi.ndim == 3 and dapi.shape[2] == 3:
            dapi_gray = np.dot(dapi[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            dapi_gray = dapi

        H, W = int(dapi_gray.shape[0]), int(dapi_gray.shape[1])

        x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))

        if bins is None:
            nx, ny = W, H
        elif isinstance(bins, int):
            factor = max(1, int(bins))
            nx = max(16, W // factor)
            ny = max(16, H // factor)
        else:
            nx, ny = int(bins[0]), int(bins[1])

        x_edges = np.linspace(x_min, x_max, nx + 1)
        y_edges = np.linspace(y_min, y_max, ny + 1)
        return dapi_gray, (x_min, x_max, y_min, y_max), (x_edges, y_edges)

    def _crop_by_world_square_arrays(dapi, dens, extent_xy, side_um):
        """
        Apply a centred square crop in the original coordinate system to both
        the DAPI image and the density array.
        """
        x_min, x_max, y_min, y_max = extent_xy
        H, W = dapi.shape[:2]
        ny, nx = dens.shape[:2]

        if (side_um is None) or (side_um <= 0):
            return dapi, dens, (x_min, x_max, y_min, y_max)

        width_w = x_max - x_min
        height_w = y_max - y_min
        if (side_um >= width_w) or (side_um >= height_w):
            return dapi, dens, (x_min, x_max, y_min, y_max)

        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        half = 0.5 * side_um

        x0w = max(x_min, cx - half)
        x1w = min(x_max, cx + half)
        y0w = max(y_min, cy - half)
        y1w = min(y_max, cy + half)

        sx_dapi = W / width_w
        sy_dapi = H / height_w
        x0i_d = int(np.floor((x0w - x_min) * sx_dapi))
        x1i_d = int(np.ceil((x1w - x_min) * sx_dapi))
        y_top_i_d = int(np.floor((y_max - y1w) * sy_dapi))
        y_bottom_i_d = int(np.ceil((y_max - y0w) * sy_dapi))

        x0i_d = np.clip(x0i_d, 0, W - 1)
        x1i_d = np.clip(x1i_d, 1, W)
        y_top_i_d = np.clip(y_top_i_d, 0, H - 1)
        y_bottom_i_d = np.clip(y_bottom_i_d, 1, H)

        dapi_crop = dapi[y_top_i_d:y_bottom_i_d, x0i_d:x1i_d]

        sx_den = nx / width_w
        sy_den = ny / height_w
        x0i_dn = int(np.floor((x0w - x_min) * sx_den))
        x1i_dn = int(np.ceil((x1w - x_min) * sx_den))
        y_top_i_dn = int(np.floor((y_max - y1w) * sy_den))
        y_bottom_i_dn = int(np.ceil((y_max - y0w) * sy_den))

        x0i_dn = np.clip(x0i_dn, 0, nx - 1)
        x1i_dn = np.clip(x1i_dn, 1, nx)
        y_top_i_dn = np.clip(y_top_i_dn, 0, ny - 1)
        y_bottom_i_dn = np.clip(y_bottom_i_dn, 1, ny)

        dens_crop = dens[y_top_i_dn:y_bottom_i_dn, x0i_dn:x1i_dn]

        return dapi_crop, dens_crop, (x0w, x1w, y0w, y1w)

    def _start_col(n_in_row: int, n_cols: int, align: str) -> int:
        """Return the starting column index for a row-aligned block of panels."""
        if n_in_row >= n_cols:
            return 0
        if align == "left":
            return 0
        if align == "right":
            return n_cols - n_in_row
        return (n_cols - n_in_row) // 2

    # ------------------------------------------------------------------
    # First pass: load data and compute full-field density maps
    # ------------------------------------------------------------------
    dens_maps_full = []
    extents_xy = []
    counts_total = []
    counts_selected = []

    for i, (tx_path, img_path) in enumerate(zip(transcript_files, image_files)):
        _v(
            f"[load] {i + 1}/{n}  "
            f"tx='{os.path.basename(tx_path)}'  "
            f"img='{os.path.basename(img_path)}'"
        )

        had_sel = False
        x = y = H2 = None

        usecols = [gene_col, x_col, y_col]
        df = pd.read_csv(tx_path, usecols=usecols)

        n_total = int(df.shape[0])
        df_sel = df[df[gene_col].isin(genes)]
        n_sel = int(df_sel.shape[0])
        had_sel = (n_sel > 0)

        x_all = df[x_col].to_numpy()
        y_all = df[y_col].to_numpy()

        dapi_gray, ext_xy, (x_edges, y_edges) = _load_dapi_and_edges(img_path, x_all, y_all)

        ny = len(y_edges) - 1
        nx = len(x_edges) - 1

        if not had_sel:
            dens_count = np.zeros((ny, nx), dtype=np.float32)
        else:
            x = df_sel[x_col].to_numpy()
            y = df_sel[y_col].to_numpy()
            H2, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
            dens_count = H2.T.astype(np.float32)

        if sigma is not None and sigma > 0:
            dens_count = gaussian_filter(dens_count, sigma=float(sigma), mode="nearest")

        dx = float(np.diff(x_edges).mean())
        dy = float(np.diff(y_edges).mean())
        area_um2 = max(dx * dy, 1e-12)
        dens_um2 = dens_count / area_um2

        dens_maps_full.append((dapi_gray, dens_um2))
        extents_xy.append(ext_xy)
        counts_total.append(n_total)
        counts_selected.append(n_sel)

        try:
            del df, df_sel, x_all, y_all, dens_count
            if had_sel:
                del x, y, H2
        except Exception:
            pass
        gc.collect()

    # ------------------------------------------------------------------
    # Define the shared colour scale and density threshold
    # ------------------------------------------------------------------
    finite_all = (
        np.concatenate([panel[1].ravel() for panel in dens_maps_full])
        if dens_maps_full else np.array([])
    )
    finite_all = finite_all[np.isfinite(finite_all) & (finite_all >= 0)]

    if finite_all.size == 0:
        vmin_global, vmax_global = 0.0, 1.0
    else:
        p_lo, p_hi = global_clip_percentiles
        vmin_global = float(np.percentile(finite_all, p_lo))
        vmax_global = float(np.percentile(finite_all, p_hi))
        vmin_global = max(vmin_global, 0.0)
        if vmax_global <= max(vmin_global, 1e-12):
            vmax_global = float(np.max(finite_all))

    threshold_abs = float(threshold_frac) * float(vmax_global)

    # ------------------------------------------------------------------
    # Define the figure layout
    # ------------------------------------------------------------------
    if assoc_arr is None:
        cols_eff = min(max_cols, max(1, n))
        rows_eff = math.ceil(n / cols_eff)
    else:
        rows_eff = rows_assoc
        cols_eff = cols_assoc

    extra_rows = 1 if show_colorbar else 0
    height_ratios = [1.0] * rows_eff + ([colorbar_row_ratio] if show_colorbar else [])

    fig_w = figsize_per_panel[0] * cols_eff
    fig_h = figsize_per_panel[1] * rows_eff + (
        figsize_per_panel[0] * colorbar_row_ratio if show_colorbar else 0.0
    )

    fig = plt.figure(figsize=(fig_w, fig_h))
    if black_background:
        fig.patch.set_facecolor("black")

    fig.suptitle(
        f"Smoothed transcript density on DAPI — {_genes_suptitle(genes)}",
        fontsize=14,
        color=("white" if black_background else "black"),
        y=0.98,
    )

    gs = fig.add_gridspec(
        rows_eff + extra_rows,
        cols_eff,
        wspace=0.02,
        hspace=0.06,
        height_ratios=height_ratios,
    )

    # ------------------------------------------------------------------
    # Create panel axes
    # ------------------------------------------------------------------
    if assoc_arr is None:
        axes = []
        k = 0
        for r in range(rows_eff):
            remaining = n - k
            if (r == rows_eff - 1) and (remaining == 1) and (cols_eff == 2):
                ax = fig.add_subplot(gs[r, :])
                axes.append(ax)
                k += 1
            else:
                for c in range(min(cols_eff, remaining)):
                    ax = fig.add_subplot(gs[r, c])
                    axes.append(ax)
                    k += 1
    else:
        axes_by_dataset = [None] * n
        for r_idx, row_val in enumerate(unique_rows):
            idx_list = row_to_indices[row_val]
            n_in_row = len(idx_list)

            align = assoc_last_row_align if (r_idx == rows_eff - 1) else assoc_row_align
            start = _start_col(n_in_row, cols_eff, align)

            for j, dataset_idx in enumerate(idx_list):
                ax = fig.add_subplot(gs[r_idx, start + j])
                axes_by_dataset[dataset_idx] = ax

        axes = axes_by_dataset

    # ------------------------------------------------------------------
    # Precompute cropped panels and the effective displayed intensity range
    # ------------------------------------------------------------------
    shown_min, shown_max = np.inf, 0.0
    precomputed_panels = []

    for i in range(n):
        dapi_gray_full, dens_full = dens_maps_full[i]
        x_min, x_max, y_min, y_max = extents_xy[i]

        if crop_side_um is not None and crop_side_um > 0:
            dapi_img, dens_img, (xc0, xc1, yc0, yc1) = _crop_by_world_square_arrays(
                dapi_gray_full,
                dens_full,
                (x_min, x_max, y_min, y_max),
                crop_side_um,
            )
            extent_imshow = (xc0, xc1, yc1, yc0)
            width_w, height_w = (xc1 - xc0), (yc1 - yc0)
        else:
            dapi_img, dens_img = dapi_gray_full, dens_full
            extent_imshow = (x_min, x_max, y_max, y_min)
            width_w, height_w = (x_max - x_min), (y_max - y_min)

        mask = np.isfinite(dens_img) & (dens_img >= threshold_abs)
        dens_plot = np.where(mask, dens_img, np.nan).astype(np.float32)

        if np.isfinite(dens_plot).any():
            shown_min = min(shown_min, float(np.nanmin(dens_plot)))
            shown_max = max(shown_max, float(np.nanmax(dens_plot)))

        precomputed_panels.append((dapi_img, dens_plot, extent_imshow, (width_w, height_w)))

    if not np.isfinite(shown_min):
        shown_min, shown_max = max(vmin_global, 1e-12), max(vmax_global, 1.0)
    else:
        shown_min = max(shown_min, 1e-12 if color_scale == "global_log" else 0.0)
        shown_max = max(shown_max, shown_min * (1.0 + 1e-12))

    if color_scale == "global_log":
        norm = LogNorm(vmin=shown_min, vmax=shown_max, clip=True)
    elif color_scale == "global_linear":
        norm = Normalize(vmin=shown_min, vmax=shown_max, clip=True)
    else:
        raise ValueError("`color_scale` must be one of {'global_log', 'global_linear'}.")

    # ------------------------------------------------------------------
    # Draw panels
    # ------------------------------------------------------------------
    label_sel = "N_gene" if len(genes) == 1 else "N_genes"

    for i, ax in enumerate(axes):
        if black_background:
            ax.set_facecolor("black")

        dapi_img, dens_plot, extent_imshow, (width_w, height_w) = precomputed_panels[i]
        panel_title = titles[i]

        ax.imshow(dapi_img, cmap="gray", alpha=image_alpha, extent=extent_imshow)
        ax.imshow(
            dens_plot,
            cmap=cmap,
            alpha=overlay_alpha,
            extent=extent_imshow,
            interpolation="nearest",
            norm=norm,
        )

        inside_title = (
            f"{panel_title} — N_tot={counts_total[i]:,} | {label_sel}={counts_selected[i]:,}"
            .replace(",", " ")
        )
        ax.text(
            0.5, 0.02,
            inside_title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            color=("white" if black_background else "black"),
            bbox=dict(
                facecolor=("black" if black_background else "white"),
                alpha=0.28,
                edgecolor="none",
                boxstyle="round,pad=0.25",
            ),
        )

        if draw_scalebar and (scalebar_um is not None) and (scalebar_um > 0):
            x0 = extent_imshow[0] + scalebar_margin_um
            x1 = x0 + float(scalebar_um)
            y0 = extent_imshow[3] + scalebar_margin_um

            ax.plot(
                [x0, x1], [y0, y0],
                color=scalebar_color,
                lw=scalebar_width_px,
                solid_capstyle="butt",
            )

            if scalebar_label:
                dy = (0.02 * height_w) if (scalebar_label_offset_um is None) else float(scalebar_label_offset_um)
                if str(scalebar_label_position).lower() == "below":
                    y_lab, va = y0 - dy, "top"
                else:
                    y_lab, va = y0 + dy, "bottom"

                ax.text(
                    0.5 * (x0 + x1),
                    y_lab,
                    f"{int(round(float(scalebar_um)))} μm",
                    ha="center",
                    va=va,
                    fontsize=scalebar_label_fontsize,
                    color=scalebar_color,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="black", alpha=0.6)],
                )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    # ------------------------------------------------------------------
    # Draw the shared horizontal colour bar
    # ------------------------------------------------------------------
    if show_colorbar:
        cax = fig.add_subplot(gs[rows_eff, :])

        if isinstance(norm, LogNorm):
            if colorbar_ticks is None:
                locator = LogLocator(numticks=8)
                formatter = LogFormatter(10, labelOnlyBase=False)
                cb = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cax,
                    orientation="horizontal",
                    format=formatter,
                    ticks=locator,
                )
            else:
                cb = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cax,
                    orientation="horizontal",
                    ticks=colorbar_ticks,
                )
        else:
            cb = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax,
                orientation="horizontal",
                ticks=colorbar_ticks,
            )

        cb.set_label(colorbar_label, fontsize=12)

        if black_background:
            cax.set_facecolor("black")
            for spine in cax.spines.values():
                spine.set_color("white")
            cb.ax.xaxis.label.set_color("white")
            cb.ax.tick_params(colors="white")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    if save_pdf is not None:
        pdf_path = save_pdf if str(save_pdf).lower().endswith(".pdf") else (str(save_pdf) + ".pdf")
        outdir = os.path.dirname(pdf_path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        fig.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            dpi=300,
            facecolor=("black" if black_background else "white"),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return {
        "fig": fig,
        "axes": axes,
        "extents": [(e[0], e[1], e[2], e[3]) for e in extents_xy],
        "genes": genes,
        "transcript_files": list(transcript_files),
        "image_files": list(image_files),
        "counts_total": counts_total,
        "counts_selected": counts_selected,
        "vmin_shown": float(shown_min),
        "vmax_shown": float(shown_max),
        "color_scale": color_scale,
        "threshold_abs": float(threshold_abs),
        "save_pdf": save_pdf,
    }
