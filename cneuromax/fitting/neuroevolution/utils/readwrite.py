"""File reading and writing utilities for Neuroevolution fitting."""

import pickle
from pathlib import Path
from typing import Annotated as An

from cneuromax.fitting.neuroevolution.agent.singular import (
    BaseSingularAgent,
)
from cneuromax.fitting.neuroevolution.utils.type import (
    Generation_results_batch_type,
    Generation_results_type,
)
from cneuromax.utils.annotations import ge
from cneuromax.utils.mpi import retrieve_mpi_variables


def load_state(
    prev_num_gens: An[int, ge(0)],
    len_agents_batch: An[int, ge(1)],
) -> tuple[
    list[list[BaseSingularAgent]],  # agents_batch
    Generation_results_type | None,  # generation_results
    An[int, ge(0)] | None,  # total_num_env_steps
]:
    """Load a previous experiment state from disk.

    Args:
        prev_num_gens: See\
            :paramref:`~.NeuroevolutionFittingHydraConfig.prev_num_gens`.
        len_agents_batch: See\
            :paramref:`~.initialize_agents.len_agents_batch`.

    Returns:
        * See ~.compute_generation_results.agents_batch`.
        * See\
            :paramref:`~.compute_generation_results.generation_results`.
        * See\
            :paramref:`~.compute_total_num_env_steps_and_process_fitnesses.total_num_env_steps`.
    """
    comm, rank, size = retrieve_mpi_variables()
    if rank == 0:
        path = Path.cwd() / f"/{prev_num_gens}/state.pkl"
        if not path.exists():
            error_msg = f"No saved state found at {path}."
            raise FileNotFoundError(error_msg)
        with path.open(mode="rb") as f:
            state = pickle.load(file=f)
        agents: list[list[BaseSingularAgent]] = state[0]
        generation_results: Generation_results_type = state[1]
        total_num_env_steps: int = state[2]
        batched_agents: list[list[list[BaseSingularAgent]]] = [
            agents[i * len_agents_batch : (i + 1) * len_agents_batch]
            for i in range(size)
        ]
    # `comm.scatter` argument `sendobj` is wrongly typed. `[]` is the
    # workaround for not being able to set it to `None`.
    # See https://github.com/mpi4py/mpi4py/issues/434
    agents_batch = comm.scatter(sendobj=[] if rank != 0 else batched_agents)
    return (
        agents_batch,
        None if rank != 0 else generation_results,
        None if rank != 0 else total_num_env_steps,
    )


def save_state(
    agents_batch: list[list[BaseSingularAgent]],
    generation_results: Generation_results_batch_type | None,
    total_num_env_steps: An[int, ge(0)] | None,
    curr_gen: An[int, ge(1)],
) -> None:
    """Dump the current experiment state to disk.

    Args:
        agents_batch: See\
            :paramref:`~.compute_generation_results.agents_batch`.
        generation_results: See\
            :paramref:`~.compute_generation_results.generation_results`.
        total_num_env_steps: See\
            :paramref:`~.compute_total_num_env_steps_and_process_fitnesses.total_num_env_steps`.
        curr_gen: See :paramref:`~.BaseSpace.curr_gen`.
    """
    comm, rank, _ = retrieve_mpi_variables()
    batched_agents: list[list[list[BaseSingularAgent]]] | None = comm.gather(
        sendobj=agents_batch,
    )
    if rank != 0:
        return
    # `batched_agents`, `generation_results`, and `total_num_env_steps`
    # are only  `None` when `rank != 0`. The following `assert`
    # statements are for static type checking reasons and have no
    # execution purposes.
    assert batched_agents  # noqa: S101
    assert generation_results  # noqa: S101
    assert total_num_env_steps  # noqa: S101
    agents: list[list[BaseSingularAgent]] = []
    for agent_batch in batched_agents:
        agents = agents + agent_batch
    path = Path.cwd() / f"/{curr_gen}/"
    if not path.exists():
        path.mkdir(parents=True)
    with path.open(mode="wb") as f:
        pickle.dump(
            obj=[agents, generation_results, total_num_env_steps],
            file=f,
        )
