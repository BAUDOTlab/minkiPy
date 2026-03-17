import os
import glob
import numpy as np


def gene_npz_path(output_path: str, resolution: float, name: str, gene: str) -> str:
    fname = f"minkiPy_Minkowski_resolution_{resolution}_{name}_{gene}.npz"
    return os.path.join(output_path, fname)


def save_gene_npz(
    output_path: str,
    resolution: float,
    name: str,
    gene: str,
    *,
    LS,
    Minkowski_tensor,
    number_of_transcripts: int,
    min_val: float,
    max_val: float,
    area_LS0: float,
    SN_respl_samples=None,
) -> str:
    os.makedirs(output_path, exist_ok=True)
    out_path = gene_npz_path(output_path, resolution, name, gene)

    save_kwargs = dict(
        LS=LS,
        Minkowski_tensor=Minkowski_tensor,
        gene_name=gene,
        number_of_transcripts=int(number_of_transcripts),
        min_val=float(min_val),
        max_val=float(max_val),
        area_LS0=float(area_LS0),
    )
    if SN_respl_samples is not None:
        save_kwargs["SN_respl_samples"] = SN_respl_samples

    np.savez_compressed(out_path, **save_kwargs)
    return out_path


def list_gene_npz(output_path: str, resolution: float, name: str) -> list[str]:
    pattern = os.path.join(output_path, f"minkiPy_Minkowski_resolution_{resolution}_{name}_*.npz")
    return sorted(glob.glob(pattern))


def merge_npz_to_h5(output_path: str, resolution: float, name: str, area_mask: float) -> str | None:
    """
    Merge per-gene .npz into one HDF5 file. Removes .npz files after merge.
    Returns merged path, or None if nothing to merge.
    """
    import h5py

    npz_files = list_gene_npz(output_path, resolution, name)
    if len(npz_files) == 0:
        print("[Rank 0] No .npz files found! Exiting merge step.", flush=True)
        return None

    merged_h5_path = os.path.join(output_path, f"minkiPy_merged_resolution_{resolution}_{name}.h5")

    with h5py.File(merged_h5_path, "w") as h5f:
        print(f"[Rank 0] Creating merged HDF5 file: {merged_h5_path}", flush=True)

        first = np.load(npz_files[0])
        h5f.create_dataset("LS", data=first["LS"])
        first.close()
        h5f.create_dataset("area_mask", data=float(area_mask))

        genes_group = h5f.create_group("genes")

        for npz_path in npz_files:
            with np.load(npz_path, allow_pickle=True) as data:
                gene_name = data["gene_name"].item()
                Minkowski_tensor = data["Minkowski_tensor"]
                n_trans = int(data["number_of_transcripts"])
                min_val = float(data["min_val"])
                max_val = float(data["max_val"])
                area_LS0 = float(data["area_LS0"])

                has_respl = "SN_respl_samples" in data.files
                SN_respl_samples = data["SN_respl_samples"] if has_respl else None

            g = genes_group.create_group(gene_name)
            g.create_dataset("Minkowski_tensor", data=Minkowski_tensor, compression="gzip", compression_opts=4)
            g.create_dataset("min_val", data=min_val)
            g.create_dataset("max_val", data=max_val)
            g.create_dataset("area_LS0", data=float(area_LS0 * (resolution ** 2)))
            g.create_dataset("number_of_transcripts", data=n_trans)

            if SN_respl_samples is not None:
                g.create_dataset("SN_respl_samples", data=SN_respl_samples, compression="gzip", compression_opts=4)

            os.remove(npz_path)

    print(f"[Rank 0] Merged results into {merged_h5_path}, removed individual .npz files.", flush=True)
    return merged_h5_path
