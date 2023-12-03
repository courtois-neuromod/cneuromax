""":mod:`mpi4py` utilities."""

from typing import Annotated as An

from mpi4py import MPI

from cneuromax.utils.annotations import ge


def retrieve_mpi_variables() -> (
    tuple[MPI.Comm, An[int, ge(0)], An[int, ge(1)]]
):
    """Retrieves MPI variables from the MPI runtime.

    Returns:
        comm: The MPI communicator.
        rank: The rank of the current process.
        size: The total number of processes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size
