""":mod:`mpi4py` utilities."""
from typing import Annotated as An

from mpi4py import MPI

from cneuromax.utils.beartype import ge


def get_mpi_variables() -> tuple[MPI.Comm, An[int, ge(0)], An[int, ge(1)]]:
    """Retrieves MPI variables from the MPI runtime.

    Returns:
        * The MPI communicator.
        * The rank of the current process.
        * The total number of processes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size
