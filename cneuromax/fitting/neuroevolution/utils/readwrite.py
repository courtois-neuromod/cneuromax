"""File reading and writing utilities for Neuroevolution fitting."""

from pathlib import Path
from typing import Annotated as An

import numpy as np

from cneuromax.fitting.neuroevolution.utils.type import (
    agents_batch_type,
    agents_type,
    generation_results_batch_type,
    generation_results_type,
)
from cneuromax.utils.annotations import ge
from cneuromax.utils.mpi import retrieve_mpi_variables


def load_state(
    agents_batch: agents_batch_type,
    prev_num_gens: An[int, ge(0)],
) -> tuple[
    generation_results_type | None,  # generation_results
    An[int, ge(0)] | None,  # total_num_env_steps
]:
    """Load a previous experiment state from disk.

    Args:
        agents_batch: See return value `agents_batch` in\
            :func:`cneuromax.fitting.neuroevolution.utils.initialize`.
        prev_num_gens: See\
            :paramref:`~cneuromax.fitting.neuroevolution.fit.NeuroevolutionFittingHydraConfig.prev_num_gens`.

    Returns:
        agents_batch: See return value `agents_batch` in\
            :func:`cneuromax.fitting.neuroevolution.utils.initialize`.
        generation_results: See return value `generation_results` in\
            :func:`cneuromax.fitting.neuroevolution.utils.initialize`.
        total_num_env_steps: See return value `total_num_env_steps` in\
            :func:`cneuromax.fitting.neuroevolution.utils.initialize`.
    """
    comm, rank, size = retrieve_mpi_variables()
    if rank == 0:
        path = Path.cwd() / f"/{prev_num_gens}/state.npz"
        if not path.exists():
            error_msg = f"No saved state found at {path}."
            raise FileNotFoundError(error_msg)
        state = np.load(file=path, allow_pickle=True)
        agents: agents_type = state["agents"]
        generation_results: generation_results_type = state[
            "generation_results"
        ]
        total_num_env_steps = int(state["total_num_env_steps"])
    comm.Scatter(sendbuf=None if rank != 0 else agents, recvbuf=agents_batch)
    return (
        None if rank != 0 else generation_results,
        None if rank != 0 else total_num_env_steps,
    )


def save_state(
    agents: agents_type | None,
    agents_batch: agents_batch_type,
    generation_results: generation_results_batch_type | None,
    total_num_env_steps: An[int, ge(0)] | None,
    curr_gen: int,
) -> None:
    """Dump the current experiment state to disk.

    Args:
        agents: See return value `agents` in\
            :func:`cneuromax.fitting.neuroevolution.utils.initialize`.
        agents_batch: See return value `agents_batch` in\
            :func:`cneuromax.fitting.neuroevolution.utils.initialize`.
        generation_results: See return value `generation_results` in\
            :func:`cneuromax.fitting.neuroevolution.utils.initialize`.
        total_num_env_steps: See return value `total_num_env_steps` in\
            :func:`cneuromax.fitting.neuroevolution.utils.initialize`.
        curr_gen: The current generation number.
    """
    comm, rank, _ = retrieve_mpi_variables()
    comm.Gather(sendbuf=agents_batch, recvbuf=agents)
    if rank != 0:
        return
    # `agents`, `generation_results`, and `total_num_env_steps` are only
    # `None` when `rank != 0`. The following `assert` statements are for
    # static type checking reasons and have no execution purposes.
    assert agents  # noqa: S101
    assert generation_results  # noqa: S101
    assert total_num_env_steps  # noqa: S101
    path = Path.cwd() / f"/{curr_gen}/"
    if not path.exists():
        path.mkdir(parents=True)
    np.savez(
        file=path / "state",
        agents=agents,
        generation_results=generation_results,
        total_num_env_steps=total_num_env_steps,
    )
