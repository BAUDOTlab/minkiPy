from . import io
import os
import sys
import json
import tempfile
import subprocess
import pandas as pd
import numpy as np

from mpi4py import MPI

from . import minkowski_core


def redistribute_data_amongs_ranks(processed_df: pd.DataFrame, comm=MPI.COMM_WORLD):
    """
    Split the gene list across MPI ranks and send each subset to the correct rank.
    Rank 0 only.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank != 0:
        raise RuntimeError("redistribute_data_amongs_ranks must be called on rank 0 only.")

    gene_list = np.unique(
        processed_df[~processed_df["gene"].astype(str).str.startswith("Blank")]["gene"].values
    )

    total_genes = len(gene_list)
    chunk_size = total_genes // size
    remainder = total_genes % size

    send_info = []
    for r in range(size):
        start = r * chunk_size + min(r, remainder)
        end = start + chunk_size + (1 if r < remainder else 0)
        send_info.append((start, end))

    s0, e0 = send_info[0]
    local_genes = gene_list[s0:e0]
    local_data = processed_df[processed_df["gene"].isin(local_genes)]

    for r in range(1, size):
        s, e = send_info[r]
        r_local_genes = gene_list[s:e]
        df_sub_r = processed_df[processed_df["gene"].isin(r_local_genes)]

        comm.send(df_sub_r, dest=r, tag=100 + r)
        comm.send(r_local_genes, dest=r, tag=200 + r)

    return local_genes, local_data





def compute_Minkowski_profiles(
    transcripts_df,   # <- accepte None sur ranks != 0
    *,
    name: str,
    output_path: str,
    resolution: float = 20.0,
    nbr: int = 25,
    n_cov_samples: int | None = None,
    area_mask: float | None = None,
    overwrite: bool = False,
    comm=None,
    mpi_procs: int | None = None,
    tmp_dir: str | None = None,
    use_hwthreads: bool = False,
    oversubscribe: bool = False,
    extra_mpirun_args: list[str] | None = None,

):
    """
    Compute Minkowski profiles for one sample from a user-provided DataFrame.
    Required columns: ['gene','global_x','global_y'].
    Writes results to disk and returns merged HDF5 path on rank 0, else None.
    """
    # ---- AUTO-MPI mode: spawn mpirun from a single-process context ----
    if mpi_procs is not None and mpi_procs > 1:
        import json, tempfile, subprocess
        import pandas as pd

        # if user is already under MPI, do NOT respawn
        from mpi4py import MPI
        _comm = MPI.COMM_WORLD if comm is None else comm
        if _comm.Get_size() != 1:
            raise ValueError("mpi_procs>1 requested but already running under MPI.")

        if transcripts_df is None:
            raise ValueError("transcripts_df cannot be None in auto-MPI mode.")

        # keep only required columns
        required = ["gene", "global_x", "global_y"]
        missing = [c for c in required if c not in transcripts_df.columns]
        if missing:
            raise ValueError(f"transcripts_df is missing columns: {missing}")

        os.makedirs(output_path, exist_ok=True)

        # write dataframe to disk (parquet is fast + preserves types)
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="minkipy_auto_mpi_")
        os.makedirs(tmp_dir, exist_ok=True)

        df_path = os.path.join(tmp_dir, f"transcripts_{name}.parquet")
        transcripts_df[required].to_parquet(df_path, index=False)

        cfg = {
            "input": df_path,
            "input_format": "parquet",
            "gene_col": "gene",
            "x_col": "global_x",
            "y_col": "global_y",
            "name": name,
            "output_path": output_path,
            "resolution": float(resolution),
            "nbr": int(nbr),
            "n_cov_samples": None if n_cov_samples is None else int(n_cov_samples),
            "area_mask": None if area_mask is None else float(area_mask),
            "overwrite": bool(overwrite),
        }

        cfg_path = os.path.join(tmp_dir, f"config_{name}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        cmd = ["mpirun", "-n", str(int(mpi_procs))]

        if use_hwthreads:
            cmd += ["--use-hwthread-cpus"]
        
        if oversubscribe:
            cmd += ["--map-by", ":OVERSUBSCRIBE"]
        
        if extra_mpirun_args:
            cmd += list(extra_mpirun_args)
        
        cmd += [sys.executable, "-m", "minkiPy", "--run-config", cfg_path]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        captured_lines = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            captured_lines.append(line)
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(
                "Auto-MPI subprocess failed.\n"
                f"Command: {' '.join(cmd)}\n\n"
                "Combined STDOUT/STDERR:\n"
                f"{''.join(captured_lines)}\n"
            )




        merged_file = os.path.join(output_path, f"minkiPy_merged_resolution_{resolution}_{name}.h5")
        if not os.path.exists(merged_file):
            raise RuntimeError(f"Auto-MPI finished but merged file not found: {merged_file}")

        return merged_file
     
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    required_cols = {"gene", "global_x", "global_y"}
    if rank == 0:
        if transcripts_df is None:
            raise ValueError("Rank 0 requires transcripts_df (got None).")
        missing = required_cols - set(transcripts_df.columns)
        if missing:
            raise ValueError(f"transcripts_df is missing columns: {sorted(missing)}")

        os.makedirs(output_path, exist_ok=True)

        merged_file = os.path.join(
            output_path,
            f"minkiPy_merged_resolution_{resolution}_{name}.h5",
        )
        if (not overwrite) and os.path.exists(merged_file):
            print(f"[Rank 0] Found '{merged_file}', skipping '{name}'.", flush=True)
            skip_current = True
        else:
            skip_current = False
    else:
        skip_current = None

    skip_current = comm.bcast(skip_current, root=0)
    if skip_current:
        comm.Barrier()
        return merged_file if rank == 0 else None

    if n_cov_samples is None:
        n_cov_samples = int(1.5 * (4 * nbr) + 2)

    if rank == 0:
        processed_df = transcripts_df[["gene", "global_x", "global_y"]].copy()
        processed_df["gene"] = processed_df["gene"].astype(str)
        processed_df = processed_df.dropna(subset=["gene", "global_x", "global_y"])

        processed_df["global_x"] -= float(processed_df["global_x"].min())
        processed_df["global_y"] -= float(processed_df["global_y"].min())

        xmin, grid_size, edges = minkowski_core.grid_definition(resolution, processed_df=processed_df)
        edges = minkowski_core.build_grid(xmin, grid_size, resolution)

        if area_mask is None:
            area_mask_local = minkowski_core.compute_area_mask_from_transcripts(processed_df, edges)
        else:
            area_mask_local = float(area_mask)

        area_plane = (edges[-1] - edges[0]) ** 2
        frac_area = (area_mask_local / area_plane) * (grid_size**2)

        local_genes, local_data = redistribute_data_amongs_ranks(processed_df, comm=comm)

        param_dict = {
            "resolution": float(resolution),
            "nbr": int(nbr),
            "xmin": float(xmin),
            "grid_size": int(grid_size),
            "frac_area": float(frac_area),
            "output_path": output_path,
            "name": name,
            "n_cov_samples": int(n_cov_samples),
            "edges": edges.astype(np.float32),
            "area_mask": float(area_mask_local),
        }
        param_dict = comm.bcast(param_dict, root=0)
        del processed_df
    else:
        local_data = comm.recv(source=0, tag=100 + rank)
        local_genes = comm.recv(source=0, tag=200 + rank)
        param_dict = comm.bcast(None, root=0)

    resolution = param_dict["resolution"]
    nbr = param_dict["nbr"]
    edges = param_dict["edges"]
    frac_area = param_dict["frac_area"]
    output_path = param_dict["output_path"]
    name = param_dict["name"]
    n_cov_samples = param_dict["n_cov_samples"]
    area_mask_local = param_dict["area_mask"]

    print(
        f"[Rank {rank}] local_data shape = {local_data.shape}, #genes = {len(local_genes)}",
        flush=True,
    )

    for gene in local_genes:
        minkowski_core.process_gene(
            gene,
            local_data,
            edges,
            nbr,
            frac_area,
            resolution,
            output_path,
            name,
            n_cov_samples,
        )

    del local_data, local_genes
    comm.Barrier()

    if rank == 0:
        npz_files = io.list_gene_npz(output_path, resolution, name)
        if len(npz_files) >= 1:
            merged_h5 = io.merge_npz_to_h5(output_path, resolution, name, area_mask_local)
            return merged_h5

        else:
            print("[Rank 0] No .npz found at merge time (unexpected).", flush=True)
            return None

    return None


def _write_transcripts_df(df: pd.DataFrame, path_prefix: str) -> tuple[str, str]:
    df = df[["gene", "global_x", "global_y"]].copy()
    df["gene"] = df["gene"].astype(str)

    parquet_path = path_prefix + ".parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path, "parquet"
    except Exception:
        pkl_path = path_prefix + ".pkl"
        df.to_pickle(pkl_path)
        return pkl_path, "pickle"


def _read_transcripts_df(filepath: str, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(filepath)
    if fmt == "pickle":
        return pd.read_pickle(filepath)
    raise ValueError(f"Unknown df format: {fmt}")


def _run_from_config(config_path: str, comm=None) -> int:
    from mpi4py import MPI
    import os, json
    import pandas as pd

    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Rank 0 checks file exists, then everybody syncs
    if rank == 0 and not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    comm.Barrier()

    # Everybody reads the same config (cheap, avoids broadcast complexity)
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Only rank 0 loads the dataframe; others pass None (your compute_* expects that)
    df = None
    if rank == 0:
        input_path = cfg["input"]
        fmt = cfg.get("input_format", None)
        if fmt is None:
            # fallback from extension
            fmt = os.path.splitext(input_path)[1].lower().lstrip(".")

        if fmt in ("parquet", "pq"):
            df = pd.read_parquet(input_path)
        elif fmt in ("csv", "txt"):
            sep = cfg.get("sep", ",")
            df = pd.read_csv(input_path, sep=sep)
        elif fmt in ("tsv",):
            df = pd.read_csv(input_path, sep="\t")
        else:
            raise ValueError(f"Unsupported input_format={fmt!r} for input={input_path!r}")

        # Normalize column names to required ones
        gene_col = cfg.get("gene_col", "gene")
        x_col = cfg.get("x_col", "global_x")
        y_col = cfg.get("y_col", "global_y")
        df = df[[gene_col, x_col, y_col]].copy()
        df.columns = ["gene", "global_x", "global_y"]

    merged = compute_Minkowski_profiles(
        df,  # None on non-zero ranks
        name=cfg["name"],
        output_path=cfg["output_path"],
        resolution=float(cfg.get("resolution", 20.0)),
        nbr=int(cfg.get("nbr", 25)),
        n_cov_samples=cfg.get("n_cov_samples", None),
        area_mask=cfg.get("area_mask", None),
        overwrite=bool(cfg.get("overwrite", False)),
        comm=comm,
    )

    if rank == 0 and merged is not None:
        print(f"MERGED_H5_PATH={merged}", flush=True)

    return 0



def compute_Minkowski_profiles_auto_mpi(
    transcripts_df: pd.DataFrame,
    *,
    name: str,
    output_path: str,
    resolution: float = 20.0,
    nbr: int = 25,
    n_cov_samples: int | None = None,
    area_mask: float | None = None,
    overwrite: bool = False,
    n_ranks: int | None = None,
    mpirun: str = "mpirun",
    extra_mpirun_args: list[str] | None = None,
) -> str | None:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    if size > 1:
        return compute_Minkowski_profiles(
            transcripts_df,
            name=name,
            output_path=output_path,
            resolution=resolution,
            nbr=nbr,
            n_cov_samples=n_cov_samples,
            area_mask=area_mask,
            overwrite=overwrite,
            comm=comm,
        )

    os.makedirs(output_path, exist_ok=True)

    if n_ranks is None:
        n_ranks = int(os.environ.get("SLURM_NTASKS") or os.cpu_count() or 1)
    if n_ranks < 1:
        n_ranks = 1

    with tempfile.TemporaryDirectory(dir=output_path) as tmpdir:
        prefix = os.path.join(tmpdir, f"minkipy_{name}")
        df_path, df_fmt = _write_transcripts_df(transcripts_df, prefix)

        cfg = {
            "df_path": df_path,
            "df_format": df_fmt,
            "name": name,
            "output_path": output_path,
            "resolution": float(resolution),
            "nbr": int(nbr),
            "n_cov_samples": n_cov_samples,
            "area_mask": area_mask,
            "overwrite": overwrite,
        }
        cfg_path = os.path.join(tmpdir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        cmd = [mpirun, "-n", str(n_ranks)]
        if extra_mpirun_args:
            cmd += list(extra_mpirun_args)
        cmd += [sys.executable, "-m", "minkiPy", "--run-config", cfg_path]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(
                "MPI subprocess failed.\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"STDOUT:\n{proc.stdout}\n\n"
                f"STDERR:\n{proc.stderr}\n"
            )

        merged = None
        for line in proc.stdout.splitlines():
            if line.startswith("MERGED_H5_PATH="):
                merged = line.split("=", 1)[1].strip()
                break

        return merged
