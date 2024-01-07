""":mod:`wandb` utilities for Neuroevolution fitting."""
import wandb
from wandb.util import generate_id

from cneuromax.utils.mpi4py import retrieve_mpi_variables


def setup_wandb(entity: None | str) -> None:
    """Reads the W&B key, logs in, creates a group and initializes.

    Args:
        entity: Name of the account or team to use for the current run.
    """
    if not entity:
        return
    comm, rank, _ = retrieve_mpi_variables()
    wandb_group_id = generate_id() if rank == 0 else None
    wandb_group_id = comm.bcast(wandb_group_id)
    wandb.init(entity=entity, group=wandb_group_id)
