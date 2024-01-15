"""Process agent exchange for Neuroevolution fitting."""
from typing import Annotated as An

import numpy as np
from mpi4py import MPI

from cneuromax.fitting.neuroevolution.agent import (
    BaseAgent,
)
from cneuromax.fitting.neuroevolution.utils.type import (
    Exchange_and_mutate_info_batch_type,
    Exchange_and_mutate_info_type,
    Generation_results_type,
    Seeds_type,
)
from cneuromax.utils.beartype import ge, le
from cneuromax.utils.mpi4py import get_mpi_variables


def update_exchange_and_mutate_info(
    num_pops: An[int, ge(1), le(2)],
    pop_size: An[int, ge(1)],
    exchange_and_mutate_info: Exchange_and_mutate_info_type | None,
    generation_results: Generation_results_type | None,
    seeds: Seeds_type | None,
) -> None:
    """Update the exchange and mutate information.

    The selection process of the algorithm is in some sense implicit in
    :mod:`cneuromax`. We make use of 50% truncation selection, which is
    reflected in the information stored inside
    :paramref:`exchange_and_mutate_info`.

    In some sense, the selection process of the algorithm is performed
    in this function.

    Args:
        num_pops: See :meth:`~.BaseSpace.num_pops`.
        pop_size: See\
            :paramref:`~.compute_start_time_and_seeds.pop_size`.
        exchange_and_mutate_info: An array maintained only by\
            the primary process (secondary processes set this to\
            ``None``) containing information for all processes on\
            how to exchange and mutate agents. Precisions on the 3rd\
            dimension: 0) The size of the agent when serialized, 1)\
            The position of the agent paired for with the current\
            agent, 2) Whether to send or receive the agent, 3) The\
            seed to randomize the mutation and evaluation of the\
            agent.
        generation_results: See\
            :paramref:`~.compute_generation_results.generation_results`.
        seeds: The seeds to set the mutation and evaluation randomness\
            for the current generation.
    """
    _, rank, _ = get_mpi_variables()
    if rank != 0:
        return
    # `exchange_and_mutate_info`, `generation_results`, and `seeds`are
    # only `None` when `rank != 0`. The following `assert` statements
    # are for static type checking reasons and have no execution
    # purposes.
    assert exchange_and_mutate_info  # noqa: S101
    assert generation_results  # noqa: S101
    assert seeds  # noqa: S101
    serialized_agent_sizes = generation_results[:, :, 2]
    fitnesses = generation_results[:, :, 0]
    # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
    # for a full example execution of the genetic algorithm.
    # The following block is examplified in section 10.
    fitnesses_sorting_indices = fitnesses.argsort(axis=0)
    fitnesses_index_ranking = fitnesses_sorting_indices.argsort(axis=0)
    # 0) MPI buffer size
    exchange_and_mutate_info[:, :, 0] = np.max(serialized_agent_sizes)
    for j in range(num_pops):
        # Each selected/non-selected agent is paired with a
        # corresponding non-selected/selected agent. Both agents are
        # placed in the same position in the ranking sub-leaderboard of
        # selected and non-selected agents.
        # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
        # for a full example execution of the genetic algorithm.
        # The following block is examplified in section 12.
        # (Section 11 is performed in
        # :func:`.compute_start_time_and_seeds`)
        paired_agent_ranking = (
            fitnesses_index_ranking[:, j] + pop_size // 2
        ) % pop_size
        paired_agent_position = fitnesses_sorting_indices[
            :,
            j,
        ][paired_agent_ranking]
        # 1) Agent pair position
        exchange_and_mutate_info[:, j, 1] = paired_agent_position
    # 2) Sending (1 means sending, 0 means receiving)
    exchange_and_mutate_info[:, :, 2] = np.greater_equal(
        fitnesses_index_ranking,
        pop_size // 2,
    )  # Also section 12 (send)
    # 3) Seeds to set randomness for mutation & evaluation.
    exchange_and_mutate_info[:, :, 3] = seeds


def exchange_agents(
    num_pops: An[int, ge(1), le(2)],
    pop_size: An[int, ge(1)],
    agents_batch: list[list[BaseAgent]],
    exchange_and_mutate_info_batch: Exchange_and_mutate_info_batch_type,
) -> None:
    """Exchange agents between processes.

    Args:
        num_pops: See :meth:`~.BaseSpace.num_pops`.
        pop_size: See\
            :paramref:`~.compute_start_time_and_seeds.pop_size`.
        agents_batch: See\
            :paramref:`~.compute_generation_results.agents_batch`.
        exchange_and_mutate_info_batch: See\
            :paramref:`~.mutate.exchange_and_mutate_info_batch`.
    """
    comm, rank, _ = get_mpi_variables()
    mpi_buffer_size = exchange_and_mutate_info_batch[:, :, 0]
    paired_agent_position = exchange_and_mutate_info_batch[:, :, 1]
    sending = exchange_and_mutate_info_batch[:, :, 2]
    len_agents_batch = len(agents_batch)
    # List to contain the `len_agents_batch` * `num_pops` number of
    # MPI requests created with the `isend` and `irecv` methods.
    req: list[MPI.Request] = []
    # Iterate over all agents in the batch.
    for i in range(len_agents_batch):
        for j in range(num_pops):
            # Can determine the rank of the paired process rank from the
            # `paired_agent_position` and `len_agents_batch` variables.
            # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
            # for a full example execution of the genetic algorithm.
            # The following block is examplified in section 14.
            paired_process_rank = int(
                paired_agent_position[i, j] // len_agents_batch,
            )
            if sending[i, j] == 1:  # 1 means sending
                # Give a unique tag for this agent that the receiving
                # process will be able to match.
                tag = int(pop_size * j + len_agents_batch * rank + i)
                # Send (non-blocking) the agent and append the MPI
                # request.
                # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
                # for a full example execution of the genetic algorithm.
                # The following block is examplified in section 15.
                req.append(
                    comm.isend(
                        obj=agents_batch[i][j],
                        dest=paired_process_rank,
                        tag=tag,
                    ),
                )
            else:  # not 1 (0) means receiving
                # Give a unique tag for this agent that the sending
                # process will be able to match.
                tag = int(pop_size * j + paired_agent_position[i, j])
                # Receive (non-blocking) the agent and append the MPI
                # request.
                # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
                # for a full example execution of the genetic algorithm.
                # The following block is examplified in section 15.
                req.append(
                    comm.irecv(
                        buf=mpi_buffer_size[i, j, 0],
                        source=paired_process_rank,
                        tag=tag,
                    ),
                )
    # Wait for all MPI requests and retrieve a list composed of the
    # agents received from the other processes and `None` for the
    # agents that were sent.
    agent_or_none_list: list[BaseAgent] = MPI.Request.waitall(req)
    # Replacing existing agents with the received agents.
    for i, agent_or_none in enumerate(agent_or_none_list):
        if agent_or_none:
            agents_batch[i // num_pops][i % num_pops] = agent_or_none
