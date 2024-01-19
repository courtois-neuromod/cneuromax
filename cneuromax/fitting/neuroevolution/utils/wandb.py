""":mod:`wandb` utilities for Neuroevolution fitting."""
from collections.abc import Callable
from typing import Any

import numpy as np
import wandb

from cneuromax.utils.mpi4py import get_mpi_variables


def setup_wandb(logger: Callable[..., Any]) -> None:
    """Sets up :mod:`wandb` logging for all MPI processes.

    Args:
        logger: See :func:`~.wandb.init`.
    """
    comm, rank, _ = get_mpi_variables()
    if rank != 0:
        return
    logger()


def terminate_wandb() -> None:
    """Terminates :mod:`wandb` logging."""
    comm, rank, _ = get_mpi_variables()
    if rank != 0:
        return
    wandb.finish()


def gather(logged_score: float | None, curr_gen: int) -> None:
    """Gathers logged scores from all MPI processes.

    Args:
        logged_score: A value logged during evaluation. If ``None``,\
            then no value was logged during evaluation.
        curr_gen: See :paramref:`~.BaseSpace.curr_gen`.
    """
    comm, rank, _ = get_mpi_variables()
    logged_scores: list[float | None] | None = comm.gather(
        sendobj=logged_score,
    )
    if rank != 0:
        return
    # `logged_scores` is only `None` when `rank != 0`. The following
    # `assert` statement is for static type checking reasons and has no
    # execution purposes.
    assert logged_scores is not None  # noqa: S101
    non_none_logged_scores: list[float] = list(filter(None, logged_scores))
    non_none_logged_scores_mean = np.mean(non_none_logged_scores)
    wandb.log(data={"score": non_none_logged_scores_mean, "gen": curr_gen})
