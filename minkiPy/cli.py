from __future__ import annotations

import os
import argparse
import pandas as pd

from .mpi_driver import compute_Minkowski_profiles


def _load_transcripts(
    path: str,
    *,
    gene_col: str,
    x_col: str,
    y_col: str,
    sep: str = ",",
) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path, sep=sep)
    elif ext in [".tsv"]:
        df = pd.read_csv(path, sep="\t")
    elif ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported input extension {ext!r}. Use .csv/.tsv/.parquet"
        )

    # normalise to required names
    missing = [c for c in (gene_col, x_col, y_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in input file: {missing}. "
            f"Available columns: {list(df.columns)[:30]}{'...' if len(df.columns) > 30 else ''}"
        )

    out = df[[gene_col, x_col, y_col]].copy()
    out.columns = ["gene", "global_x", "global_y"]
    return out



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="minkiPy",
        description="Compute Minkowski profiles from a transcript table (MPI).",
    )

    p.add_argument("--input", required=True, help="Path to CSV/TSV/Parquet with transcripts.")
    p.add_argument("--name", required=True, help="Sample name used in output filename.")
    p.add_argument("--output-path", required=True, help="Directory for outputs.")
    p.add_argument("--resolution", type=float, default=20.0)
    p.add_argument("--nbr", type=int, default=25)
    p.add_argument("--n-cov-samples", type=int, default=None)
    p.add_argument("--mc-seed", type=int, default=None)
    p.add_argument("--area-mask", type=float, default=None)
    p.add_argument("--overwrite", action="store_true")

    # column mapping
    p.add_argument("--gene-col", default="gene")
    p.add_argument("--x-col", default="global_x")
    p.add_argument("--y-col", default="global_y")
    p.add_argument("--sep", default=",", help="CSV separator (default ','). Ignored for parquet/tsv.")

    args = p.parse_args(argv)

    # only rank 0 loads data; others pass None
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    transcripts_df = None
    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
        transcripts_df = _load_transcripts(
            args.input,
            gene_col=args.gene_col,
            x_col=args.x_col,
            y_col=args.y_col,
            sep=args.sep,
        )

    h5 = compute_Minkowski_profiles(
        transcripts_df,
        name=args.name,
        output_path=args.output_path,
        resolution=args.resolution,
        nbr=args.nbr,
        n_cov_samples=args.n_cov_samples,
        mc_seed=args.mc_seed,
        area_mask=args.area_mask,
        overwrite=args.overwrite,
        comm=comm,
    )

    if rank == 0:
        print("merged:", h5, flush=True)

    return 0
