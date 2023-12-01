"""Process agent exchange for Neuroevolution fitting."""

from typing import Annotated as An

from mpi4py import MPI

from cneuromax.fitting.neuroevolution.agent.singular.base import (
    BaseSingularAgent,
)
from cneuromax.fitting.neuroevolution.utils.type import (
    exchange_and_mutate_info_batch_type,
)
from cneuromax.utils.annotations import ge, le
from cneuromax.utils.mpi import retrieve_mpi_variables


def exchange_agents(
    num_pops: An[int, ge(1), le(2)],
    pop_size: An[int, ge(1)],
    agents_batch: list[list[BaseSingularAgent]],
    exchange_and_mutate_info_batch: exchange_and_mutate_info_batch_type,
) -> None:
    """Exchange agents between processes.

    Args:
        num_pops: See\
            :meth:`~cneuromax.fitting.neuroevolution.space.base.BaseSpace.num_pops`.
        pop_size: See return value `pop_size` from\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
        agents_batch: See return value `agents_batch` from\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
        exchange_and_mutate_info_batch: See return value\
            `exchange_and_mutate_info_batch` from\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
    """
    comm, rank, _ = retrieve_mpi_variables()
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
            paired_process_rank = int(
                paired_agent_position[i, j] // len_agents_batch,
            )
            if sending[i, j] == 1:  # 1 means sending
                # Give a unique tag for this agent that the receiving
                # process will be able to match.
                tag = int(pop_size * j + len_agents_batch * rank + i)
                # Send (non-blocking) the agent and append the MPI
                # request.
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
    agent_or_none_list: list[BaseSingularAgent] = MPI.Request.waitall(req)
    # Replacing existing agents with the received agents.
    for i, agent_or_none in enumerate(agent_or_none_list):
        if agent_or_none:
            agents_batch[i // num_pops][i % num_pops] = agent_or_none


# def update_exchange_and_mutate_info():
#     pass
