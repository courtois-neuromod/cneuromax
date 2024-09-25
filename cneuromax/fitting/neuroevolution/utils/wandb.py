"""`W&B <https://wandb.ai/>`_ utilities for Neuroevolution fitting."""

from collections.abc import Callable
from typing import Any

import numpy as np
import wandb
from omegaconf import OmegaConf

from cneuromax.utils.mpi4py import get_mpi_variables


def setup_wandb(logger: Callable[..., Any], output_dir: str) -> None:
    """Sets up `W&B <https://wandb.ai/>`_ logging for all MPI processes.

    Args:
        logger: See :func:`~.wandb.init`.
        output_dir: See :paramref:`~.BaseSubtaskConfig.output_dir`.
    """
    comm, rank, _ = get_mpi_variables()
    if rank != 0:
        return
    logger(
        config=OmegaConf.to_container(
            OmegaConf.load(f"{output_dir}/.hydra/config.yaml"),
            resolve=True,
            throw_on_missing=True,
        ),
    )


def terminate_wandb() -> None:
    """Terminates `W&B <https://wandb.ai/>`_ logging."""
    comm, rank, _ = get_mpi_variables()
    if rank != 0:
        return
    wandb.finish()


def gather(
    logged_score: float | None,
    curr_gen: int,
    agent_total_num_steps: int,
) -> None:
    """Gathers logged scores from all MPI processes.

    Args:
        logged_score: A value logged during evaluation. If ``None``,
            then no value was logged during evaluation.
        curr_gen: See :paramref:`~.BaseSpace.curr_gen`.
        agent_total_num_steps: See
            :attr:`~.BaseAgent.total_num_steps`.
    """
    comm, rank, _ = get_mpi_variables()
    logged_scores: list[float | None] | None = comm.gather(
        sendobj=logged_score,
    )
    logged_agent_total_num_steps: list[int] | None = comm.gather(
        sendobj=agent_total_num_steps,
    )
    if rank != 0:
        return
    # `logged_scores` & `logged_agent_total_num_steps` are only `None`
    # when `rank != 0`. The following `assert` statements are for static
    # type checking reasons and have no execution purposes.
    assert logged_scores is not None  # noqa: S101
    assert logged_agent_total_num_steps is not None  # noqa: S101
    non_none_logged_scores: list[float] = list(filter(None, logged_scores))
    non_none_logged_scores_mean = np.mean(a=non_none_logged_scores)
    logged_agent_total_num_steps_mean = np.mean(
        a=logged_agent_total_num_steps,
    )
    wandb.log(
        data={
            "score": non_none_logged_scores_mean,
            "num_steps": logged_agent_total_num_steps_mean,
            "gen": curr_gen,
        },
    )
