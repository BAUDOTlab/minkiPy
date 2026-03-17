import argparse
from mpi4py import MPI

from .mpi_driver import _run_from_config
from . import cli


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--run-config",
        type=str,
        default=None,
        help="Run compute_Minkowski_profiles from a JSON config (used by auto MPI wrapper).",
    )

    args, remaining = parser.parse_known_args(argv)

    if args.run_config is not None:
        return _run_from_config(args.run_config, comm=MPI.COMM_WORLD)

    return cli.main(remaining)


if __name__ == "__main__":
    ret = main()
    raise SystemExit(0 if ret is None else int(ret))
