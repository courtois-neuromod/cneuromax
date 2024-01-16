""":mod:`wandb` utilities for Neuroevolution fitting."""
from collections.abc import Callable
from typing import Any

from wandb.util import generate_id

from cneuromax.utils.mpi4py import get_mpi_variables


def setup_wandb(wandb_init: Callable[..., Any]) -> None:
    """Sets up :mod:`wandb` logging for all MPI processes.

    Args:
        wandb_init: See :func:`~.wandb.init`.
    """
    comm, rank, _ = get_mpi_variables()
    wandb_group_id = generate_id() if rank == 0 else None
    wandb_group_id = comm.bcast(wandb_group_id)
    wandb_init(group=wandb_group_id)
