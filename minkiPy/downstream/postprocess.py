from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from typing import Dict, List, Optional, Sequence, Union, Tuple

__all__ = ["process_data",
           "add_averaged_condition_datasets",
           "compute_sample_distances",
           "compute_gene_distances"]

def process_data(
    filepaths: Sequence[str],
    *,
    # Robust scaling and covariance estimation
    by_level: bool = False,
    cov_use_scaled: bool = True,
    min_mc: int | None = None,
    min_samples_per_feature: float = 1.5,
    density_field: str = "gene_density",
    dtype=np.float32,

    # Gene filtering
    min_transcripts_any_sample: Optional[int] = None,

    # Optional annotations
    GOI: Optional[List[str]] = None,
    ordered_conditions: Optional[List[str]] = None,
    groups: Optional[Dict[str, List[int]]] = None,
    default_label: str = "UNKNOWN",

    # Verbosity
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Load non-subsampled merged *minkiPy* HDF5 files and assemble a single
    in-memory data structure for downstream comparative analysis.

    Parameters
    ----------
    filepaths:
        Sequence of HDF5 file paths. Each file must follow the naming pattern

            minkiPy_merged_resolution_<reso>_<name>.h5

        and must correspond to a non-subsampled merged output. All files are
        required to share the same level-set grid (``LS``). Only genes present
        in all files are retained.

    by_level:
        Controls how robust scaling is fitted.

        If ``False`` (default), one robust scaler is fitted per Minkowski
        statistic across all files, genes and level sets.

        If ``True``, an independent scaler is fitted for each
        (statistic, level-set) pair.

    cov_use_scaled:
        If ``True`` (default), covariance matrices are estimated from the
        robust-scaled Monte Carlo resamples. Otherwise, raw resamples are used.

    min_mc:
        Optional absolute minimum number of Monte Carlo samples required for a
        covariance estimate to be considered sufficiently supported. If
        ``None``, the threshold is derived from ``min_samples_per_feature``.

    min_samples_per_feature:
        Relative rule used when ``min_mc`` is ``None``. With
        ``d = 4 * L`` features, where ``L`` is the number of level sets, a
        covariance estimate is considered sufficiently supported when the
        number of valid Monte Carlo draws ``m`` satisfies

            m >= min_samples_per_feature * d + 1

    density_field:
        Key used to store the raw gene density array in the returned
        dictionary.

    dtype:
        NumPy dtype used for floating-point arrays.

    min_transcripts_any_sample:
        Optional post-intersection gene filter. A gene is retained if

            max_i number_of_transcripts(dataset_i, gene)
                >= min_transcripts_any_sample

        This removes genes that are weakly expressed in all datasets while
        retaining genes that are highly expressed in at least one dataset.

    GOI:
        Optional list of genes of interest. If provided, the returned object
        includes a boolean mask ``gene_is_goi`` and a summary in
        ``extra_knowledge``.

    ordered_conditions:
        Optional reference order used when interpreting the integer indices
        supplied in ``groups``. If omitted, the order inferred from the file
        names is used.

    groups:
        Optional mapping from condition labels to dataset indices, for example
        ``{"CRC": [...], "NAT": [...]}``. This is used only to attach labels to
        datasets and does not affect numerical computations.

    default_label:
        Label assigned to datasets not covered by ``groups``.

    verbose:
        If ``True``, print a concise processing log.

    Returns
    -------
    dict
        Dictionary containing the loaded tensors, Monte Carlo resamples,
        scaling information, covariance matrices, density estimates, and
        optional dataset or gene annotations.

    Notes
    -----
    This function deliberately enforces a common gene set and a common level-set
    grid across all input files. That restriction simplifies downstream
    comparisons and avoids silent inconsistencies between datasets.
    """
    import re
    import h5py

    def _v(message: str) -> None:
        """Print a message only when verbose logging is enabled."""
        if verbose:
            print(message, flush=True)

    # ------------------------------------------------------------------
    # 1. Validate input files and recover dataset metadata from filenames
    # ------------------------------------------------------------------
    if not filepaths:
        raise ValueError("process_data() received an empty 'filepaths' sequence.")

    norm_paths = [str(Path(fp).expanduser().resolve()) for fp in filepaths]

    merged_re = re.compile(
        r"^minkiPy_merged_resolution_(?P<reso>[^_]+)_(?!subsampled_data_)(?P<name>.+)\.h5$"
    )

    listed: list[tuple[str, str, str]] = []
    for fp in norm_paths:
        base = os.path.basename(fp)
        match = merged_re.match(base)
        if not match:
            raise ValueError(
                f"File {fp!r} does not match the expected pattern "
                "'minkiPy_merged_resolution_<reso>_<name>.h5'."
            )
        listed.append((fp, match.group("reso"), match.group("name")))

    if not listed:
        raise FileNotFoundError(
            "No valid non-subsampled merged files were found in 'filepaths'."
        )

    # Deterministic ordering: dataset name, then resolution, then full path.
    listed.sort(key=lambda item: (item[2], item[1], item[0]))

    filepaths = [item[0] for item in listed]
    resolutions = [item[1] for item in listed]
    conditions = [item[2] for item in listed]
    n_files = len(filepaths)

    parent_dirs = [os.path.dirname(fp) for fp in filepaths]
    root_dir = parent_dirs[0] if len(parent_dirs) == 1 else os.path.commonpath(parent_dirs)

    _v(f"[load] Found {n_files} non-subsampled merged files.")
    if verbose and n_files <= 8:
        for i, fp in enumerate(filepaths):
            _v(
                f"  - #{i:02d} | dataset={conditions[i]} | resolution={resolutions[i]} "
                f"| file={os.path.basename(fp)}"
            )
    elif verbose:
        for i, fp in enumerate(filepaths[:8]):
            _v(
                f"  - #{i:02d} | dataset={conditions[i]} | resolution={resolutions[i]} "
                f"| file={os.path.basename(fp)}"
            )
        _v("  - ...")

    # ------------------------------------------------------------------
    # 2. Inspect files: level sets, gene overlap, tensor shapes, MC status
    # ------------------------------------------------------------------
    LS_ref = None
    gene_sets: list[set[str]] = []
    n_stats = None
    n_levels = None
    any_sn = False
    area_mask = np.full((n_files,), np.nan, dtype=dtype)

    for i, fp in enumerate(filepaths):
        with h5py.File(fp, "r") as f:
            if "LS" not in f:
                raise KeyError(f"{fp} is missing dataset 'LS'.")
            LS = f["LS"][()]
            if LS_ref is None:
                LS_ref = LS
            else:
                if not (LS.shape == LS_ref.shape and np.allclose(LS, LS_ref, rtol=0, atol=1e-12)):
                    raise ValueError(f"Level-set grid mismatch detected in {fp}.")

            if "area_mask" in f:
                try:
                    area_mask[i] = np.float32(f["area_mask"][()])
                except Exception:
                    _v(
                        f"[load][warning] Could not read 'area_mask' in "
                        f"{os.path.basename(fp)}; storing NaN instead."
                    )

            if "genes" not in f:
                raise KeyError(f"{fp} is missing group 'genes'.")
            genes_here = set(map(str, f["genes"].keys()))
            if not genes_here:
                raise ValueError(f"No genes were found in {fp}.")
            gene_sets.append(genes_here)

            # Use a single gene to infer the expected tensor shape.
            g0 = next(iter(genes_here))
            MT = f[f"genes/{g0}/Minkowski_tensor"]
            if MT.ndim != 2:
                raise ValueError(
                    f"{fp}: 'Minkowski_tensor' must be 2D, got shape {MT.shape}."
                )
            s0, s1 = int(MT.shape[0]), int(MT.shape[1])
            if n_stats is None:
                n_stats, n_levels = s0, s1
            else:
                if (s0, s1) != (n_stats, n_levels):
                    raise ValueError(
                        f"Minkowski tensor shape mismatch in {fp}: "
                        f"{(s0, s1)} versus {(n_stats, n_levels)}."
                    )

            if "SN_respl_samples" in f[f"genes/{g0}"]:
                any_sn = True

    assert LS_ref is not None
    assert n_levels is not None
    if n_stats != 4:
        raise ValueError(f"Expected 4 statistics (W0, W1, W2, BETA), got {n_stats}.")

    common_genes_all = sorted(set.intersection(*gene_sets))
    if not common_genes_all:
        raise ValueError("No gene is shared across all input files.")
    n_genes_before = len(common_genes_all)

    _v(
        f"[inspect] Common level-set grid with {n_levels} levels; "
        f"{n_genes_before} genes shared across all datasets."
    )

    # ------------------------------------------------------------------
    # 3. Optionally filter genes using transcript counts across datasets
    # ------------------------------------------------------------------
    if min_transcripts_any_sample is not None:
        if min_transcripts_any_sample <= 0:
            raise ValueError(
                "min_transcripts_any_sample must be a positive integer or None."
            )

        _v(
            "[gene-filter] Applying transcript-count filter: "
            f"retain genes with max_i n_transcripts[i, g] >= {min_transcripts_any_sample}."
        )

        max_ntr = np.zeros((n_genes_before,), dtype=np.int32)
        missing_ntr_counter = 0

        for fp in filepaths:
            with h5py.File(fp, "r") as f:
                ggrp = f["genes"]
                for j, g in enumerate(common_genes_all):
                    G = ggrp[g]
                    if "number_of_transcripts" in G:
                        try:
                            ntr_val = int(G["number_of_transcripts"][()])
                        except Exception:
                            ntr_val = 0
                            missing_ntr_counter += 1
                    else:
                        ntr_val = 0
                        missing_ntr_counter += 1
                    if ntr_val > max_ntr[j]:
                        max_ntr[j] = ntr_val

        keep_mask = max_ntr >= int(min_transcripts_any_sample)
        common_genes = [g for g, keep in zip(common_genes_all, keep_mask) if keep]
        n_genes_after = len(common_genes)

        _v(
            f"[gene-filter] Retained {n_genes_after}/{n_genes_before} genes "
            f"after transcript-count filtering."
        )
        if missing_ntr_counter > 0:
            _v(
                "[gene-filter][warning] Missing or unreadable "
                f"'number_of_transcripts' entries were treated as 0 "
                f"({missing_ntr_counter} occurrences across all file-gene pairs)."
            )

        if n_genes_after == 0:
            raise ValueError(
                "Gene filtering removed all genes. Lower "
                "'min_transcripts_any_sample' or verify that "
                "'number_of_transcripts' exists in the HDF5 files."
            )
    else:
        common_genes = common_genes_all
        n_genes_after = n_genes_before

    n_genes = len(common_genes)
    gene_to_index = {g: i for i, g in enumerate(common_genes)}

    # ------------------------------------------------------------------
    # 4. Determine the common number of MC resamples, if available
    # ------------------------------------------------------------------
    nreal = None
    if any_sn:
        nreal_min = None
        for fp in filepaths:
            with h5py.File(fp, "r") as f:
                for g in common_genes:
                    G = f[f"genes/{g}"]
                    if "SN_respl_samples" in G:
                        shp = G["SN_respl_samples"].shape  # expected: (4, r, L)
                        if len(shp) != 3 or shp[0] != 4 or shp[2] != n_levels:
                            raise ValueError(f"{fp}:{g} has invalid SN_respl_samples shape {shp}.")
                        r = int(shp[1])
                        if r > 0:
                            nreal_min = r if nreal_min is None else min(nreal_min, r)
        if nreal_min and nreal_min > 0:
            nreal = nreal_min

    if nreal is None:
        _v("[inspect] No Monte Carlo resamples detected in the input files.")
    else:
        _v(f"[inspect] Common Monte Carlo depth: {nreal}.")

    # ------------------------------------------------------------------
    # 5. Load tensors, transcript counts and optional per-gene attributes
    # ------------------------------------------------------------------
    T = np.empty((n_files, n_genes, 4, n_levels), dtype=dtype)
    ntr = np.empty((n_files, n_genes), dtype=np.int32)
    vmin = np.full((n_files, n_genes), np.nan, dtype=dtype)
    vmax = np.full((n_files, n_genes), np.nan, dtype=dtype)
    area_LS0 = np.full((n_files, n_genes), np.nan, dtype=dtype)
    SN = None
    if nreal is not None:
        SN = np.full((n_files, n_genes, 4, nreal, n_levels), np.nan, dtype=dtype)

    missing_ntr = 0
    missing_area_LS0 = 0

    for i, fp in enumerate(filepaths):
        with h5py.File(fp, "r") as f:
            ggrp = f["genes"]
            for g, j in gene_to_index.items():
                G = ggrp[g]
                MT = G["Minkowski_tensor"][()]
                if MT.shape != (4, n_levels):
                    raise ValueError(f"{fp}:{g} has tensor shape {MT.shape}, expected (4, {n_levels}).")
                T[i, j] = MT.astype(dtype, copy=False)

                if "number_of_transcripts" in G:
                    ntr[i, j] = int(G["number_of_transcripts"][()])
                else:
                    ntr[i, j] = -1
                    missing_ntr += 1

                if "min_val" in G:
                    vmin[i, j] = np.float32(G["min_val"][()])
                if "max_val" in G:
                    vmax[i, j] = np.float32(G["max_val"][()])
                if "area_LS0" in G:
                    try:
                        area_LS0[i, j] = np.float32(G["area_LS0"][()])
                    except Exception:
                        missing_area_LS0 += 1
                else:
                    missing_area_LS0 += 1

                if SN is not None and "SN_respl_samples" in G:
                    sn = G["SN_respl_samples"][()]  # shape: (4, r, L)
                    SN[i, j, :, :nreal, :] = sn[:, :nreal, :].astype(dtype, copy=False)

    _v(
        f"[load] Loaded arrays for {n_files} datasets, {n_genes} genes, "
        f"4 statistics and {n_levels} level sets."
    )
    _v(f"[load] Missing 'number_of_transcripts' entries: {missing_ntr}.")
    _v(f"[load] Missing or unreadable 'area_LS0' entries: {missing_area_LS0}.")
    present_mask = ~np.isnan(area_mask)
    _v(f"[load] 'area_mask' available for {int(present_mask.sum())}/{n_files} datasets.")

    data: Dict[str, object] = {
        "root_dir": str(Path(root_dir).resolve()),
        "folder": str(Path(root_dir).resolve()),  # Backward-compatible alias.
        "files": filepaths,
        "conditions": conditions,
        "resolutions": resolutions,
        "gene_names": common_genes,
        "gene_to_index": gene_to_index,
        "LS": LS_ref,
        "stat_names": ("W0", "W1", "W2", "BETA"),
        "n_stats": 4,
        "n_levels": int(n_levels),
        "nreal": nreal,
        "tensor_per_sample": T,
        "sn_samples": SN,
        "n_transcripts": ntr,
        "min_val": vmin,
        "max_val": vmax,
        "area_LS0": area_LS0,
        "area_mask": area_mask,
        "min_transcripts_any_sample": min_transcripts_any_sample,
        "n_genes_before_filter": int(n_genes_before),
        "n_genes_after_filter": int(n_genes_after),
    }

    # ------------------------------------------------------------------
    # 6. Apply robust scaling to tensors and MC resamples
    # ------------------------------------------------------------------
    T = data["tensor_per_sample"]            # type: ignore[assignment]
    SN = data.get("sn_samples", None)        # type: ignore[assignment]
    F, G, S, L = T.shape                     # type: ignore[union-attr]
    if S != 4:
        raise ValueError("Expected 4 statistics (W0, W1, W2, BETA).")

    if SN is not None:
        if (
            not isinstance(SN, np.ndarray)
            or SN.ndim != 5
            or SN.shape[2] != 4
            or SN.shape[-1] != L
        ):
            raise ValueError(
                "sn_samples must have shape (F, G, 4, R, L), "
                f"got {None if not isinstance(SN, np.ndarray) else SN.shape}."
            )

    if not by_level:
        # One robust scaler per statistic, pooling over files, genes and levels.
        reduce_axes = (0, 1, 3)
        centres = np.nanmedian(T, axis=reduce_axes).astype(dtype)
        q25 = np.nanpercentile(T, 25, axis=reduce_axes).astype(dtype)
        q75 = np.nanpercentile(T, 75, axis=reduce_axes).astype(dtype)
        scales = (q75 - q25).astype(dtype)
        scales[scales == 0.0] = 1.0

        T_scaled = (T - centres[None, None, :, None]) / scales[None, None, :, None]
        SN_scaled = None
        if SN is not None:
            SN_scaled = (
                SN - centres[None, None, :, None, None]
            ) / scales[None, None, :, None, None]
        mode_desc = "per statistic"
    else:
        # One robust scaler per (statistic, level-set) pair.
        reduce_axes = (0, 1)
        centres = np.nanmedian(T, axis=reduce_axes).astype(dtype)
        q25 = np.nanpercentile(T, 25, axis=reduce_axes).astype(dtype)
        q75 = np.nanpercentile(T, 75, axis=reduce_axes).astype(dtype)
        scales = (q75 - q25).astype(dtype)
        scales[scales == 0.0] = 1.0

        T_scaled = (T - centres[None, None, :, :]) / scales[None, None, :, :]
        SN_scaled = None
        if SN is not None:
            SN_scaled = (
                SN - centres[None, None, :, None, :]
            ) / scales[None, None, :, None, :]
        mode_desc = "per statistic and level set"

    data["tensor_scaled"] = T_scaled.astype(dtype, copy=False)
    data["sn_samples_scaled"] = None if SN is None else SN_scaled.astype(dtype, copy=False)
    data["scaler"] = {
        "type": "robust",
        "by_level": bool(by_level),
        "centers": centres,
        "scales": scales,
        "stat_names": tuple(data.get("stat_names", ("W0", "W1", "W2", "BETA"))),  # type: ignore[arg-type]
    }
    _v(f"[scale] Applied robust scaling ({mode_desc}).")

    # ------------------------------------------------------------------
    # 7. Estimate covariance matrices from Monte Carlo resamples
    # ------------------------------------------------------------------
    SN_for_cov = (
        data["sn_samples_scaled"] if cov_use_scaled else data.get("sn_samples", None)
    )  # type: ignore[assignment]

    if SN_for_cov is None:
        d = data["n_stats"] * data["n_levels"]  # type: ignore[index]
        data["cov_matrices"] = None
        data["cov_counts"] = None
        data["cov_feature_dim"] = int(d)
        data["cov_rank_deficient"] = None
        data["cov_insufficient_mc"] = None
    
        _v("[cov] No Monte Carlo resamples were provided; covariance estimation was skipped.")
        _v("[cov] The returned data object therefore does not contain covariance matrices.")
        _v("[cov] Covariance-aware downstream analyses, including Gaussian Wasserstein distances, are unavailable.")
        _v("[cov] Downstream comparisons must therefore rely on profile-only representations, for example Euclidean distances computed from Minkowski profiles.")
    else:
        if (
            not isinstance(SN_for_cov, np.ndarray)
            or SN_for_cov.ndim != 5
            or SN_for_cov.shape[2] != 4
        ):
            raise ValueError(
                "sn_samples has invalid shape "
                f"{None if not isinstance(SN_for_cov, np.ndarray) else SN_for_cov.shape}; "
                "expected (F, G, 4, R, L)."
            )

        F, G, _, R, L = SN_for_cov.shape
        d = 4 * L
        threshold = min_mc if min_mc is not None else int(np.ceil(min_samples_per_feature * d) + 1)
        _v(f"[cov] Estimating covariance matrices in D={d} dimensions; MC threshold = {threshold}.")

        cov_mats = np.full((F, G, d, d), np.nan, dtype=dtype)
        counts = np.zeros((F, G), dtype=np.int32)
        rank_def = np.zeros((F, G), dtype=bool)
        insuf = np.zeros((F, G), dtype=bool)

        for i in range(F):
            for j in range(G):
                X = SN_for_cov[i, j].transpose(1, 0, 2).reshape(R, d)
                mask = np.isfinite(X).all(axis=1)
                X = X[mask]
                m = X.shape[0]
                counts[i, j] = m
                insuf_here = m < threshold
                insuf[i, j] = insuf_here

                if m < 2:
                    rank_def[i, j] = True
                    continue

                X = X - X.mean(axis=0, keepdims=True)
                C = (X.T @ X) / (m - 1)
                C = 0.5 * (C + C.T)
                cov_mats[i, j] = C.astype(dtype, copy=False)
                rank_def[i, j] = (m - 1) < d

        n_insuf = int(insuf.sum())
        n_rankd = int(rank_def.sum())

        _v(
            f"[cov] Completed covariance estimation for {F * G} dataset-gene pairs."
        )
        if n_insuf > 0:
            _v(
                f"[cov] {n_insuf}/{F * G} pairs did not meet the MC support "
                f"threshold ({threshold})."
            )
        if n_rankd > 0:
            _v(
                f"[cov] {n_rankd}/{F * G} covariance matrices are potentially "
                f"rank-deficient because m - 1 < D."
            )

        data["cov_matrices"] = cov_mats
        data["cov_counts"] = counts
        data["cov_feature_dim"] = int(d)
        data["cov_rank_deficient"] = rank_def
        data["cov_insufficient_mc"] = insuf

    # ------------------------------------------------------------------
    # 8. Compute raw and normalised gene densities
    # ------------------------------------------------------------------
    if "n_transcripts" not in data:
        raise KeyError("`data` must contain 'n_transcripts' with shape (F, G).")
    if "area_mask" not in data:
        raise KeyError("`data` must contain 'area_mask' with shape (F,).")

    NTR = np.asarray(data["n_transcripts"], dtype=np.float64)
    A = np.asarray(data["area_mask"], dtype=np.float64)
    if A.ndim != 1:
        raise ValueError(f"area_mask must be 1D with shape (n_files,), got {A.shape}.")
    if NTR.ndim != 2:
        raise ValueError(f"n_transcripts must be 2D with shape (n_files, n_genes), got {NTR.shape}.")
    F, G = NTR.shape
    if A.shape[0] != F:
        raise ValueError(f"len(area_mask) = {A.shape[0]} but n_files = {F}.")

    # Guard against invalid areas.
    A_safe = A.copy()
    invalid_area = ~np.isfinite(A_safe) | (A_safe <= 0)
    A_safe[invalid_area] = np.nan

    # Guard against invalid transcript counts.
    N_safe = NTR.copy()
    N_safe[(~np.isfinite(N_safe)) | (N_safe < 0)] = np.nan

    # Raw density: transcripts per unit area.
    density = (N_safe / A_safe[:, None]).astype(dtype)
    data[density_field] = density

    # Across-dataset normalisation based on the total transcript count per dataset.
    totals = np.nansum(N_safe, axis=1)
    mean_total = np.nanmean(totals) if np.isfinite(totals).any() else np.nan
    scale = np.ones((F,), dtype=dtype)

    valid_tot = np.isfinite(totals) & (totals > 0)
    if np.isfinite(mean_total) and valid_tot.any():
        scale[valid_tot] = (mean_total / totals[valid_tot]).astype(dtype)
    else:
        _v(
            "[density][warning] Could not compute a valid mean transcript total "
            "across datasets; normalisation factors were left at 1."
        )

    normalized_density = (density * scale[:, None]).astype(dtype)

    data["normalized_density"] = normalized_density
    data["transcripts_total_per_dataset"] = totals.astype(np.float64, copy=False)
    data["transcripts_total_mean"] = np.array(mean_total, dtype=dtype)
    data["transcripts_scale_factors"] = scale

    if verbose:
        n_total = F * G
        n_finite = int(np.isfinite(density).sum())
        _v(
            f"[density] Computed {n_finite}/{n_total} finite raw densities; "
            f"invalid dataset areas: {int(invalid_area.sum())}/{F}."
        )

        with np.errstate(invalid="ignore"):
            per_ds_med = np.nanmedian(density, axis=1)
            per_ds_mean = np.nanmean(density, axis=1)
            per_ds_med_norm = np.nanmedian(normalized_density, axis=1)
            per_ds_mean_norm = np.nanmean(normalized_density, axis=1)

        _v(f"[density] Mean total transcript count across datasets: {mean_total:.6g}.")
        _v("[density] Per-dataset scaling summary:")
        for i, name in enumerate(data.get("conditions", [f"D{i}" for i in range(F)])):  # type: ignore[arg-type]
            s = float(scale[i])
            t = float(totals[i]) if np.isfinite(totals[i]) else np.nan
            med_raw = per_ds_med[i]
            mean_raw = per_ds_mean[i]
            med_norm = per_ds_med_norm[i]
            mean_norm = per_ds_mean_norm[i]
            _v(
                f"  - {name}: total={t:.6g}, scale={s:.4f}, "
                f"raw(median={med_raw:.3g}, mean={mean_raw:.3g}), "
                f"normalised(median={med_norm:.3g}, mean={mean_norm:.3g})"
            )

    # ------------------------------------------------------------------
    # 9. Attach optional dataset-condition labels and GOI annotations
    # ------------------------------------------------------------------
    if (groups is not None) or (GOI is not None):
        ds_names_data = list(map(str, data["conditions"]))  # type: ignore[index]
        gene_names = list(map(str, data["gene_names"]))     # type: ignore[index]
        n_files = len(ds_names_data)

        if groups is not None:
            ref_order = ordered_conditions if ordered_conditions is not None else ds_names_data
            name_by_index = {i: name for i, name in enumerate(ref_order)}
            label_by_name = {name: default_label for name in ds_names_data}

            for label, idxs in groups.items():
                for idx in idxs:
                    if idx not in name_by_index:
                        raise IndexError(
                            f"Index {idx} in groups['{label}'] is out of bounds "
                            f"(len={len(ref_order)})."
                        )
                    name = name_by_index[idx]
                    if name in label_by_name:
                        if label_by_name[name] != default_label and label_by_name[name] != label:
                            raise ValueError(
                                f"Dataset '{name}' appears in more than one group: "
                                f"'{label_by_name[name]}' and '{label}'."
                            )
                        label_by_name[name] = label

            dataset_condition = [label_by_name[name] for name in ds_names_data]
        else:
            label_by_name = {name: default_label for name in ds_names_data}
            dataset_condition = [default_label for _ in ds_names_data]

        if GOI is not None:
            goi_set = set(map(str, GOI))
            gene_is_goi = np.array([g in goi_set for g in gene_names], dtype=bool)
            goi_indices = np.nonzero(gene_is_goi)[0].tolist()
            goi_found = [gene_names[i] for i in goi_indices]
            goi_missing = sorted(list(goi_set.difference(set(gene_names))))
        else:
            gene_is_goi = None
            goi_indices, goi_found, goi_missing = [], [], []

        data["dataset_condition"] = dataset_condition
        if gene_is_goi is not None:
            data["gene_is_goi"] = gene_is_goi

        data["extra_knowledge"] = {
            "condition_mapping": label_by_name,
            "groups_input": None if groups is None else {k: list(v) for k, v in groups.items()},
            "reference_order": (
                list(ordered_conditions) if ordered_conditions is not None else list(ds_names_data)
            ),
            "default_label": default_label,
            "counts_by_condition": {
                label: dataset_condition.count(label) for label in sorted(set(dataset_condition))
            },
            "goi_list_input": None if GOI is None else list(GOI),
            "goi_found": goi_found,
            "goi_missing_in_data": goi_missing,
            "goi_indices": goi_indices,
            "n_datasets": n_files,
            "n_genes": len(gene_names),
            "min_transcripts_any_sample": min_transcripts_any_sample,
            "n_genes_before_filter": int(n_genes_before),
            "n_genes_after_filter": int(n_genes_after),
        }

        if verbose:
            cond_counts = data["extra_knowledge"]["counts_by_condition"]  # type: ignore[index]
            _v(f"[annotate] Datasets per condition: {cond_counts}.")

            if GOI is not None:
                n_found = len(goi_indices)
                n_total = len(GOI)
                n_missing = len(goi_missing)
                _v(
                    f"[annotate] GOI matched in data: {n_found}/{n_total}; "
                    f"missing: {n_missing}."
                )
                if n_missing > 0:
                    _v(f"[annotate] GOI missing from data: {', '.join(goi_missing)}")

    _v(
        f"[done] Processing complete. Final gene set: {n_genes} genes "
        f"(before filtering: {n_genes_before}, after filtering: {n_genes_after})."
    )
    return data

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

def compute_sample_distances(
    data: dict,
    *,
    diag_only: bool = False,
    ridge: float = 1e-6,
    dataset_pairs: Union[str, Sequence[Tuple[Union[int, str], Union[int, str]]]] = "ALL",
    weight_by_transcripts: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Compute and attach averaged sample-to-sample distances.

    This function compares datasets represented by gene-level Minkowski
    profiles stored in ``data["tensor_scaled"]``. When covariance matrices are
    available in ``data["cov_matrices"]``, distances are computed using
    Gaussian 2-Wasserstein geometry. When covariance matrices are absent, the
    function falls back to Euclidean distances between mean feature vectors.

    For each dataset ``i`` and gene ``g``, let ``μ_{i,g}`` denote the
    robust-scaled Minkowski profile flattened into a vector in ``R^d``, with
    ``d = 4 * L`` where ``L`` is the number of level sets.

    If covariance matrices are available, let ``Σ_{i,g}`` denote the
    covariance matrix in ``R^{d x d}``.

    For each selected dataset pair ``(i, j)``, the function first computes a
    per-gene distance between the same gene in the two datasets. These
    gene-level distances are then aggregated into a sample-level distance.
    By default, aggregation is weighted by the total number of transcripts of
    each gene in the two datasets.

    Two cases are supported.

    1. Covariance matrices available
       ------------------------------
       If ``diag_only=False``, the exact Gaussian 2-Wasserstein distance is
       used:

           W2^2 = ||μ1 - μ2||^2
                  + tr(Σ1) + tr(Σ2)
                  - 2 tr((Σ2^{1/2} Σ1 Σ2^{1/2})^{1/2})

       If ``diag_only=True``, a diagonal approximation is used:

           W2^2 ≈ ||μ1 - μ2||^2
                  + Σ_k (sqrt(v1_k) - sqrt(v2_k))^2

       where ``v1_k`` and ``v2_k`` are the diagonal entries of the covariance
       matrices. A small ridge is added to the diagonal for numerical
       stabilisation.

    2. No covariance matrices available
       --------------------------------
       Distances are computed as Euclidean distances between the flattened mean
       profiles:

           d_E(μ1, μ2) = ||μ1 - μ2||_2

       In this case, ``diag_only`` and ``ridge`` are ignored numerically, but
       are still recorded in the output summary for transparency.

    Results are written back into ``data`` in three forms.

    First, a generic summary dictionary is stored in
    ``data["sample_distance_summary"]``.

    Secondly, one scalar entry is stored for each computed dataset pair using
    the key pattern

        "<label_i>_<label_j>_averaged_sample_distance"

    Thirdly, if all unordered off-diagonal dataset pairs are requested, the
    per-gene distances are cached in ``data["sample_distance_per_gene"]`` with
    shape ``(F, F, G)``.

    For backward compatibility, the function also populates
    ``data["wasserstein_summary"]`` and ``data["wasserstein_D_gene"]`` when
    covariance matrices are available.

    Parameters
    ----------
    data:
        Dictionary produced by the upstream preprocessing pipeline. It must
        contain at least

            - ``"tensor_scaled"``: ndarray of shape ``(F, G, 4, L)``
            - ``"conditions"``: sequence of dataset labels of length ``F``

        If covariance-aware distances are desired, it should also contain

            - ``"cov_matrices"``: ndarray of shape ``(F, G, d, d)``

        If transcript-weighted aggregation is desired, it should also contain

            - ``"n_transcripts"``: ndarray of shape ``(F, G)``

    diag_only:
        If covariance matrices are available, choose between the full Gaussian
        2-Wasserstein distance (``False``) and its diagonal approximation
        (``True``). Ignored when covariance matrices are absent.

    ridge:
        Small non-negative value added to covariance diagonals for numerical
        stabilisation in the Wasserstein case. Ignored when covariance matrices
        are absent.

    dataset_pairs:
        Dataset pairs to compare. This can be either

            - ``"ALL"``, meaning all unique unordered pairs ``(i, j)`` with
              ``i < j``, or
            - a sequence of pairs ``(a, b)``, where each element is either a
              dataset index or a dataset name present in ``data["conditions"]``.

        Pairs are normalised internally so that order does not matter,
        duplicates are removed, and self-comparisons are ignored.

    weight_by_transcripts:
        If ``True`` (default), aggregate per-gene distances into a sample-level
        distance using transcript-count weights proportional to the sum of
        transcript counts in the two datasets. If transcript counts are not
        available, the function falls back to an unweighted mean.

    verbose:
        If ``True``, print a concise summary of the computation, optionally
        including a single progress bar when available.

    Returns
    -------
    dict
        The input dictionary, updated in place.

    Notes
    -----
    This function is dataset-agnostic. It does not assume any application-
    specific condition names, experimental design, or biological context.
    """
    def _v(message: str) -> None:
        if verbose:
            print(message, flush=True)

    # ------------------------------------------------------------------
    # 1. Validate required inputs and inspect array shapes
    # ------------------------------------------------------------------
    if "tensor_scaled" not in data:
        raise KeyError("Expected 'tensor_scaled' in data.")
    if "conditions" not in data:
        raise KeyError("Expected 'conditions' in data (dataset labels).")

    X = np.asarray(data["tensor_scaled"])   # (F, G, 4, L)
    labels = list(map(str, data["conditions"]))

    F, G, S, L = X.shape
    if S != 4:
        raise ValueError(f"'tensor_scaled' has {S} statistics; expected 4.")
    d = S * L

    C_raw = data.get("cov_matrices", None)
    have_cov = C_raw is not None
    C = None if not have_cov else np.asarray(C_raw)

    if have_cov and C.shape != (F, G, d, d):
        raise ValueError(
            f"'cov_matrices' has shape {C.shape}; expected {(F, G, d, d)}."
        )

    NTR_raw = data.get("n_transcripts", None)
    have_ntr = NTR_raw is not None
    NTR = None if not have_ntr else np.asarray(NTR_raw)

    if have_ntr and NTR.shape != (F, G):
        raise ValueError(
            f"'n_transcripts' has shape {NTR.shape}; expected {(F, G)}."
        )

    # ------------------------------------------------------------------
    # 2. Resolve dataset selectors and define the set of dataset pairs
    # ------------------------------------------------------------------
    def _name_to_index(x: Union[int, str]) -> int:
        if isinstance(x, (int, np.integer)):
            idx = int(x)
            if not (0 <= idx < F):
                raise IndexError(f"Dataset index {idx} out of range [0, {F - 1}].")
            return idx

        if isinstance(x, str):
            try:
                return labels.index(x)
            except ValueError as exc:
                raise KeyError(
                    f"Dataset name {x!r} not found in data['conditions']."
                ) from exc

        raise TypeError(
            "Dataset selectors must be integers (indices) or strings (dataset names)."
        )

    if isinstance(dataset_pairs, str):
        if dataset_pairs.upper() != "ALL":
            raise ValueError("dataset_pairs must be 'ALL' or a sequence of pairs.")
        pairs = [(i, j) for i in range(F) for j in range(i + 1, F)]
        compute_all = True
    else:
        pairs_idx = [(_name_to_index(a), _name_to_index(b)) for (a, b) in dataset_pairs]
        pairs = sorted({(min(i, j), max(i, j)) for (i, j) in pairs_idx if i != j})
        compute_all = (len(pairs) == (F * (F - 1)) // 2)

    n_pairs = len(pairs)

    if have_cov:
        mode = "diagonal Wasserstein approximation" if diag_only else "full-covariance Wasserstein"
        _v(
            f"[distance] Computing averaged sample distances for {n_pairs} "
            f"dataset pairs (datasets={F}, genes={G}, feature_dim={d}, "
            f"mode={mode}, ridge={ridge:g}, "
            f"transcript_weighting={'on' if weight_by_transcripts else 'off'})."
        )
    else:
        _v("[distance] No covariance matrices were found in `data['cov_matrices']`.")
        _v("[distance] Falling back to Euclidean distances between flattened Minkowski profiles.")
        _v(
            f"[distance] Computing averaged sample distances for {n_pairs} "
            f"dataset pairs (datasets={F}, genes={G}, feature_dim={d}, "
            f"mode=Euclidean profile distance, "
            f"transcript_weighting={'on' if weight_by_transcripts else 'off'})."
        )

    if weight_by_transcripts and not have_ntr:
        _v(
            "[distance] Transcript-count weighting was requested, but "
            "`data['n_transcripts']` is unavailable. Falling back to "
            "unweighted averaging across genes."
        )

    # ------------------------------------------------------------------
    # 3. Numerical helpers
    # ------------------------------------------------------------------
    def _sym(A: np.ndarray) -> np.ndarray:
        return 0.5 * (A + A.T)

    def _sqrtm_psd(A: np.ndarray) -> np.ndarray:
        A = _sym(A)
        w, V = np.linalg.eigh(A)
        w = np.clip(w, 0.0, None)
        return (V * np.sqrt(w)) @ V.T

    def _w2_full(mu1: np.ndarray, C1: np.ndarray, mu2: np.ndarray, C2: np.ndarray) -> float:
        C1s = _sym(C1) + float(ridge) * np.eye(C1.shape[0], dtype=C1.dtype)
        C2s = _sym(C2) + float(ridge) * np.eye(C2.shape[0], dtype=C2.dtype)

        diff = mu1 - mu2
        dm2 = float(np.dot(diff, diff))

        S2 = _sqrtm_psd(C2s)
        inner = _sqrtm_psd(S2 @ C1s @ S2)
        trpart = float(np.trace(C1s) + np.trace(C2s) - 2.0 * np.trace(inner))

        w2sq = dm2 + max(trpart, 0.0)
        return float(np.sqrt(max(w2sq, 0.0)))

    def _w2_diag(mu1: np.ndarray, C1: np.ndarray, mu2: np.ndarray, C2: np.ndarray) -> float:
        v1 = np.clip(np.diag(C1), 0.0, None) + float(ridge)
        v2 = np.clip(np.diag(C2), 0.0, None) + float(ridge)

        diff = mu1 - mu2
        dm2 = float(np.dot(diff, diff))
        var_term = float(np.sum((np.sqrt(v1) - np.sqrt(v2)) ** 2))

        return float(np.sqrt(max(dm2 + max(var_term, 0.0), 0.0)))

    def _euclidean(mu1: np.ndarray, mu2: np.ndarray) -> float:
        diff = mu1 - mu2
        return float(np.sqrt(max(float(np.dot(diff, diff)), 0.0)))

    def _sanitize(label: str) -> str:
        return "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in label
        )

    # Flatten mean profiles once so that each gene profile becomes a vector in R^d.
    Mflat = X.reshape(F, G, d)

    # ------------------------------------------------------------------
    # 4. Allocate outputs
    # ------------------------------------------------------------------
    D_gene = np.full((F, F, G), np.nan, dtype=np.float64) if compute_all else None
    counts = np.zeros((F, F), dtype=np.int32)
    D_matrix = np.full((F, F), np.nan, dtype=np.float64)

    if verbose and n_pairs > 0:
        try:
            from tqdm.auto import tqdm
            pbar: Optional["tqdm"] = tqdm(
                total=n_pairs,
                desc="sample-distance(dataset pairs)",
                leave=True,
            )
        except Exception:
            pbar = None
    else:
        pbar = None

    # ------------------------------------------------------------------
    # 5. Main loop over dataset pairs
    # ------------------------------------------------------------------
    n_pairs_with_data = 0
    n_pairs_all_nan = 0

    for i, j in pairs:
        mui = Mflat[i]   # (G, d)
        muj = Mflat[j]   # (G, d)

        m_ok = np.isfinite(mui).all(axis=1) & np.isfinite(muj).all(axis=1)

        if have_cov:
            assert C is not None
            Ci = C[i]
            Cj = C[j]

            if diag_only:
                diag_Ci = np.diagonal(Ci, axis1=1, axis2=2)
                diag_Cj = np.diagonal(Cj, axis1=1, axis2=2)
                c_ok = np.isfinite(diag_Ci).all(axis=1) & np.isfinite(diag_Cj).all(axis=1)
            else:
                c_ok = np.isfinite(Ci).all(axis=(1, 2)) & np.isfinite(Cj).all(axis=(1, 2))

            ok = m_ok & c_ok
        else:
            ok = m_ok

        idxs = np.where(ok)[0]

        if idxs.size == 0:
            avg = np.nan
            n_pairs_all_nan += 1
        else:
            if have_cov:
                assert C is not None
                Ci = C[i]
                Cj = C[j]

                if diag_only:
                    vals = np.fromiter(
                        (_w2_diag(mui[g], Ci[g], muj[g], Cj[g]) for g in idxs),
                        dtype=np.float64,
                        count=idxs.size,
                    )
                else:
                    vals = np.fromiter(
                        (_w2_full(mui[g], Ci[g], muj[g], Cj[g]) for g in idxs),
                        dtype=np.float64,
                        count=idxs.size,
                    )
            else:
                vals = np.fromiter(
                    (_euclidean(mui[g], muj[g]) for g in idxs),
                    dtype=np.float64,
                    count=idxs.size,
                )

            if weight_by_transcripts and have_ntr:
                assert NTR is not None
                w = (
                    np.clip(NTR[i, idxs], 0.0, None) +
                    np.clip(NTR[j, idxs], 0.0, None)
                ).astype(np.float64)

                w_ok = np.isfinite(w) & (w > 0) & np.isfinite(vals)
                if np.any(w_ok):
                    avg = float(np.dot(w[w_ok], vals[w_ok]) / np.sum(w[w_ok]))
                    n_used = int(np.sum(w_ok))
                else:
                    avg = np.nan
                    n_used = 0
            else:
                vals_ok = np.isfinite(vals)
                if np.any(vals_ok):
                    avg = float(np.nanmean(vals[vals_ok]))
                    n_used = int(np.sum(vals_ok))
                else:
                    avg = np.nan
                    n_used = 0

            counts[i, j] = counts[j, i] = n_used
            D_matrix[i, j] = D_matrix[j, i] = avg

            if np.isfinite(avg):
                n_pairs_with_data += 1
            else:
                n_pairs_all_nan += 1

            if compute_all and D_gene is not None:
                D_gene[i, j, idxs] = vals
                D_gene[j, i, idxs] = vals

        key = f"{_sanitize(labels[i])}_{_sanitize(labels[j])}_averaged_sample_distance"
        data[key] = avg

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # ------------------------------------------------------------------
    # 6. Finalise pairwise summary matrices
    # ------------------------------------------------------------------
    np.fill_diagonal(D_matrix, 0.0)

    finite_mask = np.isfinite(D_matrix)
    finite_mask[np.eye(F, dtype=bool)] = False
    n_finite_pairs = int(finite_mask.sum() // 2)
    n_nan_pairs = n_pairs - n_finite_pairs

    if have_cov:
        summary_kind = "W2_diag" if diag_only else "W2_full"
        summary_mode = "diag-only Wasserstein" if diag_only else "full-covariance Wasserstein"
    else:
        summary_kind = "euclidean"
        summary_mode = "Euclidean"

    _v(
        f"[distance] Completed {n_pairs} dataset pairs "
        f"(mode={summary_mode}, "
        f"pairs_with_values={n_pairs_with_data}, all_nan={n_pairs_all_nan}, "
        f"finite_averages={n_finite_pairs}, nan_averages={n_nan_pairs})."
    )

    if compute_all and D_gene is not None:
        _v(
            "[distance] Per-gene distances cached in 'sample_distance_per_gene' "
            f"with shape {D_gene.shape}."
        )

    # ------------------------------------------------------------------
    # 7. Store summary outputs in the data dictionary
    # ------------------------------------------------------------------
    data["sample_distance_summary"] = {
        "kind": summary_kind,
        "uses_covariance": bool(have_cov),
        "diag_only": bool(diag_only),
        "ridge": float(ridge),
        "weighted_by_transcripts": bool(weight_by_transcripts and have_ntr),
        "labels": labels,
        "matrix": D_matrix,
        "counts": counts,
        "per_gene_available": bool(compute_all),
    }

    if compute_all and D_gene is not None:
        data["sample_distance_per_gene"] = D_gene

    # Backward-compatible aliases in the covariance-aware case only.
    if have_cov:
        data["wasserstein_summary"] = {
            "kind": "W2",
            "diag_only": bool(diag_only),
            "ridge": float(ridge),
            "labels": labels,
            "matrix": D_matrix,
            "counts": counts,
            "per_gene_available": bool(compute_all),
        }
        if compute_all and D_gene is not None:
            data["wasserstein_D_gene"] = D_gene

        for i, j in pairs:
            old_key = f"{_sanitize(labels[i])}_{_sanitize(labels[j])}_averaged_Wasserstein_distance"
            new_key = f"{_sanitize(labels[i])}_{_sanitize(labels[j])}_averaged_sample_distance"
            data[old_key] = data[new_key]

    return data

def compute_gene_distances(
    data: dict,
    datasets: Optional[Sequence[Union[int, str]]] = None,
    *,
    diag_only: bool = True,
    ridge: float = 1e-8,
    dtype: str = "float32",
    low_rank_r: Optional[int] = None,
    add_euclidean: bool = False,
    n_jobs: int = 1,
    block_size: int = 256,
    show_progress: bool = True,
    verbose: bool = True,
    run_name: Optional[str] = None,
    keep_history: bool = True,
) -> dict:
    """
    Compute all gene-pair distances within and across a selected subset of datasets.

    Each dataset ``k`` and gene ``g`` is represented by a flattened Minkowski
    profile ``μ_{k,g}`` in ``R^d``, where ``d = 4 * L`` and ``L`` is the number
    of level sets. These vectors are obtained from ``data["tensor_scaled"]``,
    which has shape ``(F, G, 4, L)``.

    If covariance matrices are available in ``data["cov_matrices"]``, each gene
    is further represented by a covariance matrix ``Σ_{k,g}`` in ``R^{d x d}``,
    and distances are computed in Gaussian 2-Wasserstein geometry.

    If covariance matrices are absent, distances are computed as Euclidean
    distances between flattened Minkowski profiles.

    For a selected set of ``K`` datasets and ``G`` genes, the function returns
    a tensor of shape ``(K, K, G, G)``, where entry ``[a, b, g1, g2]`` is the
    distance between gene ``g1`` in dataset ``a`` and gene ``g2`` in dataset
    ``b``.

    Two covariance-aware modes are supported.

    If ``diag_only=True``, the Gaussian 2-Wasserstein distance is approximated
    by retaining only marginal variances:

        W2^2 ≈ ||μ1 − μ2||^2
               + Σ_i (sqrt(v1_i) − sqrt(v2_i))^2

    where ``v1_i`` and ``v2_i`` are the diagonal entries of the covariance
    matrices.

    If ``diag_only=False``, the exact full-covariance Gaussian 2-Wasserstein
    distance is used:

        W2^2 = ||μ1 − μ2||^2
               + tr(Σ1) + tr(Σ2)
               − 2 tr((Σ2^{1/2} Σ1 Σ2^{1/2})^{1/2})

    In the absence of covariance matrices, the Euclidean distance is used:

        d_E(μ1, μ2) = ||μ1 − μ2||_2
    """
    def _v(message: str) -> None:
        if verbose:
            print(message, flush=True)

    # ------------------------------------------------------------------
    # 1. Validate required inputs and inspect core array shapes
    # ------------------------------------------------------------------
    if "tensor_scaled" not in data:
        raise KeyError("Expected 'tensor_scaled' in data.")
    if "conditions" not in data:
        raise KeyError("Expected 'conditions' in data.")

    X4 = np.asarray(data["tensor_scaled"])  # (F, G, 4, L)
    conds = list(map(str, data["conditions"]))
    LS = np.asarray(data.get("LS", np.arange(X4.shape[-1])))

    F, G, S, L = X4.shape
    if S != 4:
        raise ValueError(f"'tensor_scaled' has {S} statistics; expected 4.")
    d = 4 * L

    C_raw = data.get("cov_matrices", None)
    have_cov = C_raw is not None
    C = None if not have_cov else np.asarray(C_raw)

    if have_cov and C.shape != (F, G, d, d):
        raise ValueError(
            f"'cov_matrices' has shape {C.shape}; expected {(F, G, d, d)}."
        )

    # ------------------------------------------------------------------
    # 2. Resolve dataset selection and remove duplicates
    # ------------------------------------------------------------------
    if datasets is None:
        idx_sel = list(range(F))
    else:
        idx_sel = []
        for item in datasets:
            if isinstance(item, (int, np.integer)):
                idx = int(item)
                if idx < 0 or idx >= F:
                    raise IndexError(f"Dataset index {idx} out of range [0, {F - 1}].")
                idx_sel.append(idx)
            else:
                name = str(item)
                if name not in conds:
                    raise KeyError(f"Dataset name {name!r} not found in data['conditions'].")
                idx_sel.append(conds.index(name))

    seen = set()
    idx_sel = [i for i in idx_sel if not (i in seen or seen.add(i))]

    K = len(idx_sel)
    if K < 1:
        raise ValueError("Please specify at least one dataset to process.")

    labels = [conds[i] for i in idx_sel]

    # ------------------------------------------------------------------
    # 3. Cast arrays and restrict to selected datasets
    # ------------------------------------------------------------------
    np_dtype = np.float32 if str(dtype).lower().endswith("32") else np.float64

    X = X4[idx_sel].reshape(K, G, d).astype(np_dtype, copy=False)
    Csel = None if not have_cov else C[idx_sel].astype(np_dtype, copy=False)

    if have_cov:
        distance_kind = "wasserstein"
        distance_label = "2-Wasserstein distance"
        mode = "diag-only Wasserstein" if diag_only else "full-covariance Wasserstein"
        _v(
            f"[gene-distance] Computing {mode} distances for "
            f"{K} datasets, {G} genes, feature_dim={d}, ridge={ridge:g}."
        )
        if add_euclidean:
            _v("[gene-distance] Euclidean gene-pair distances will also be stored.")
        if low_rank_r is not None and not diag_only:
            _v("[gene-distance] Note: 'low_rank_r' is currently ignored.")
    else:
        distance_kind = "euclidean"
        distance_label = "Euclidean distance"
        _v("[gene-distance] No covariance matrices were found in `data['cov_matrices']`.")
        _v("[gene-distance] Falling back to Euclidean distances between Minkowski profiles.")
        _v(
            f"[gene-distance] Computing Euclidean distances for "
            f"{K} datasets, {G} genes, feature_dim={d}."
        )

    if n_jobs != 1:
        _v("[gene-distance] Note: 'n_jobs' is currently kept for API compatibility only.")

    # ------------------------------------------------------------------
    # 4. Allocate outputs and define block structure
    # ------------------------------------------------------------------
    D = np.full((K, K, G, G), np.nan, dtype=np.float32)
    mask = np.zeros((K, K, G, G), dtype=bool)

    store_euclidean_sidecar = bool(add_euclidean and have_cov)
    E = np.full((K, K, G, G), np.nan, dtype=np.float32) if store_euclidean_sidecar else None

    within_pairs: List[Tuple[int, int]] = [(a, a) for a in range(K)]
    between_pairs: List[Tuple[int, int]] = [(a, b) for a in range(K) for b in range(a + 1, K)]

    if show_progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(
                total=len(within_pairs) + len(between_pairs),
                desc="gene distances",
                leave=True,
            )
        except Exception:
            pbar = None
    else:
        pbar = None

    # ------------------------------------------------------------------
    # 5. Numerical helpers
    # ------------------------------------------------------------------
    def _euclid_block(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        na = np.sum(A * A, axis=1)[:, None]
        nb = np.sum(B * B, axis=1)[None, :]
        M2 = na + nb - 2.0 * (A @ B.T)
        return np.sqrt(np.clip(M2, 0.0, None)).astype(np.float32)

    def _sqrtm_psd(A: np.ndarray) -> np.ndarray:
        A = 0.5 * (A + A.T)
        A = A + ridge * np.eye(A.shape[0], dtype=A.dtype)
        w, V = np.linalg.eigh(A)
        w = np.clip(w, 0.0, None)
        return (V * np.sqrt(w)) @ V.T

    def _trace_sqrt_middle(C1: np.ndarray, C2: np.ndarray) -> float:
        S2 = _sqrtm_psd(C2)
        M = S2 @ C1 @ S2
        M = 0.5 * (M + M.T) + ridge * np.eye(M.shape[0], dtype=M.dtype)
        w, _ = np.linalg.eigh(M)
        w = np.clip(w, 0.0, None)
        return float(np.sum(np.sqrt(w)))

    # ------------------------------------------------------------------
    # 6. Euclidean-only path when no covariance is available
    # ------------------------------------------------------------------
    if not have_cov:
        for a in range(K):
            A = X[a]
            Dab = _euclid_block(A, A)
            D[a, a] = Dab
            mask[a, a] = np.isfinite(Dab)

            if pbar is not None:
                pbar.update(1)

        for a in range(K):
            for b in range(a + 1, K):
                A = X[a]
                B = X[b]
                Dab = _euclid_block(A, B)

                D[a, b] = Dab
                D[b, a] = Dab.T
                mask[a, b] = np.isfinite(Dab)
                mask[b, a] = mask[a, b].transpose(1, 0)

                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        tot = (K * K) * (G * G)
        _v(f"[gene-distance] Euclidean path filled {int(mask.sum())}/{tot} entries.")

    # ------------------------------------------------------------------
    # 7. Diagonal Wasserstein approximation path
    # ------------------------------------------------------------------
    elif diag_only:
        assert Csel is not None

        var = np.diagonal(Csel, axis1=2, axis2=3)
        sv = np.sqrt(np.clip(var, 0.0, None) + ridge)

        mu_norm2 = np.sum(X * X, axis=2)
        sv_norm2 = np.sum(sv * sv, axis=2)

        for a in range(K):
            A, Sroot = X[a], sv[a]

            Dab2 = (
                mu_norm2[a][:, None]
                + mu_norm2[a][None, :]
                - 2.0 * (A @ A.T)
            ) + (
                sv_norm2[a][:, None]
                + sv_norm2[a][None, :]
                - 2.0 * (Sroot @ Sroot.T)
            )

            Dab = np.sqrt(np.clip(Dab2, 0.0, None)).astype(np.float32)
            D[a, a] = Dab
            mask[a, a] = np.isfinite(Dab)

            if E is not None:
                E[a, a] = _euclid_block(A, A)

            if pbar is not None:
                pbar.update(1)

        for a in range(K):
            for b in range(a + 1, K):
                A, Sa = X[a], sv[a]
                B, Sb = X[b], sv[b]

                Dab2 = (
                    mu_norm2[a][:, None]
                    + mu_norm2[b][None, :]
                    - 2.0 * (A @ B.T)
                ) + (
                    sv_norm2[a][:, None]
                    + sv_norm2[b][None, :]
                    - 2.0 * (Sa @ Sb.T)
                )

                Dab = np.sqrt(np.clip(Dab2, 0.0, None)).astype(np.float32)

                D[a, b] = Dab
                D[b, a] = Dab.T
                mask[a, b] = np.isfinite(Dab)
                mask[b, a] = mask[a, b].transpose(1, 0)

                if E is not None:
                    Eb = _euclid_block(A, B)
                    E[a, b] = Eb
                    E[b, a] = Eb.T

                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        tot = (K * K) * (G * G)
        _v(f"[gene-distance] Diagonal Wasserstein path filled {int(mask.sum())}/{tot} entries.")

    # ------------------------------------------------------------------
    # 8. Exact full-covariance Wasserstein path
    # ------------------------------------------------------------------
    else:
        assert Csel is not None

        def _compute_block(a: int, b: int) -> None:
            nonlocal D, mask, E

            A = X[a]
            B = X[b]

            na = np.sum(A * A, axis=1)[:, None]
            nb = np.sum(B * B, axis=1)[None, :]
            mpart = na + nb - 2.0 * (A @ B.T)

            idx = np.arange(G)
            for s in range(0, G, block_size):
                bs = idx[s:s + block_size]
                for t in range(0, G, block_size):
                    bt = idx[t:t + block_size]

                    for gi in bs:
                        for gj in bt:
                            if a == b and gi == gj:
                                D[a, b, gi, gj] = 0.0
                                mask[a, b, gi, gj] = True
                                if E is not None:
                                    E[a, b, gi, gj] = 0.0
                                continue

                            C1 = Csel[a, gi]
                            C2 = Csel[b, gj]

                            if not (
                                np.isfinite(C1).all()
                                and np.isfinite(C2).all()
                                and np.isfinite(A[gi]).all()
                                and np.isfinite(B[gj]).all()
                            ):
                                continue

                            tr1 = float(np.trace(C1))
                            tr2 = float(np.trace(C2))
                            cross = _trace_sqrt_middle(C1, C2)
                            bterm = tr1 + tr2 - 2.0 * cross
                            val2 = mpart[gi, gj] + bterm

                            if val2 < 0 and val2 > -1e-8:
                                val2 = 0.0

                            if val2 >= 0:
                                D[a, b, gi, gj] = np.sqrt(val2).astype(np.float32)
                                mask[a, b, gi, gj] = True

                                if E is not None:
                                    E[a, b, gi, gj] = np.sqrt(max(mpart[gi, gj], 0.0)).astype(np.float32)

        for a, _ in within_pairs:
            _compute_block(a, a)
            if pbar is not None:
                pbar.update(1)

        for a, b in between_pairs:
            _compute_block(a, b)
            if pbar is not None:
                pbar.update(1)

        for a in range(K):
            for b in range(a + 1, K):
                D[b, a] = D[a, b].transpose(1, 0)
                mask[b, a] = mask[a, b].transpose(1, 0)
                if E is not None:
                    E[b, a] = E[a, b].transpose(1, 0)

        if pbar is not None:
            pbar.close()

        tot = (K * K) * (G * G)
        _v(f"[gene-distance] Full-covariance Wasserstein path filled {int(mask.sum())}/{tot} entries.")

    # ------------------------------------------------------------------
    # 9. Build convenience block views and payload
    # ------------------------------------------------------------------
    blocks: Dict[str, np.ndarray] = {}
    order: List[str] = []
    euclid_blocks: Dict[str, np.ndarray] | None = {} if E is not None else None

    for a in range(K):
        key = f"{labels[a]}|{labels[a]}"
        blocks[key] = D[a, a]
        if euclid_blocks is not None and E is not None:
            euclid_blocks[key] = E[a, a]
        order.append(key)

    for a in range(K):
        for b in range(a + 1, K):
            key = f"{labels[a]}|{labels[b]}"
            blocks[key] = D[a, b]
            if euclid_blocks is not None and E is not None:
                euclid_blocks[key] = E[a, b]
            order.append(key)

    payload: Dict[str, object] = {
        "kind": "W2_allpairs" if have_cov else "euclidean_allpairs",
        "distance_kind": distance_kind,
        "distance_label": distance_label,
        "uses_covariance": bool(have_cov),
        "diag_only": bool(diag_only) if have_cov else False,
        "ridge": float(ridge),
        "labels": labels,
        "LS": LS,
        "D": D,
        "counts_mask": mask,
        "shapes": {"K": K, "G": G, "d": d},
        "blocks": blocks,
        "block_order": order,
    }

    if E is not None:
        payload["E"] = E
        payload["euclid_blocks"] = euclid_blocks

    # ------------------------------------------------------------------
    # 10. Store latest run and update history
    # ------------------------------------------------------------------
    data["gene_distances"] = payload
    data["gene_distance_allpairs"] = payload  # convenient explicit alias

    if run_name is None:
        run_name = "datasets:" + ",".join(labels)

    if keep_history:
        runs = data.get("gene_distance_runs")
        if runs is None or not isinstance(runs, dict):
            runs = {}
        runs[run_name] = payload
        data["gene_distance_runs"] = runs

    # Backward-compatible aliases for existing Wasserstein workflows
    if have_cov:
        data["w2_allpairs"] = payload
        if keep_history:
            runs_old = data.get("w2_allpairs_runs")
            if runs_old is None or not isinstance(runs_old, dict):
                runs_old = {}
            runs_old[run_name] = payload
            data["w2_allpairs_runs"] = runs_old

    # ------------------------------------------------------------------
    # 11. Final logging summary
    # ------------------------------------------------------------------
    _v(
        f"[gene-distance] Stored run '{run_name}' "
        f"(distance_kind={distance_kind}) for datasets: {', '.join(labels)}."
    )

    if K == 1:
        _v(f"[gene-distance] Available block: {labels[0]}|{labels[0]}")
    else:
        within_block_names = ", ".join(f"{label}|{label}" for label in labels)
        between_block_names = ", ".join(order[K:]) if len(order) > K else "none"
        _v(f"[gene-distance] Within-dataset blocks: {within_block_names}")
        _v(f"[gene-distance] Between-dataset blocks: {between_block_names}")

    if E is not None:
        _v("[gene-distance] Euclidean block tensors were stored alongside the main distance tensor.")

    return data
