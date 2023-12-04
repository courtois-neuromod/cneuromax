"""Evolutionary operations for Neuroevolution fitting.

The selection operation is implicit in `cneuromax`, see\
:func:`~cneuromax.fitting.neuroevolution.utils.exchange.update_exchange_and_mutate_info`
for more details.
"""

from typing import Annotated as An

import numpy as np
from mpi4py import MPI

from cneuromax.fitting.neuroevolution.agent.singular.base import (
    BaseSingularAgent,
)
from cneuromax.fitting.neuroevolution.space.base import BaseSpace
from cneuromax.fitting.neuroevolution.utils.type import (
    exchange_and_mutate_info_batch_type,
    fitnesses_and_num_env_steps_batch_type,
)
from cneuromax.utils.annotations import ge
from cneuromax.utils.mpi import retrieve_mpi_variables


def run_mutation(
    agents_batch: list[list[BaseSingularAgent]],
    exchange_and_mutate_info_batch: exchange_and_mutate_info_batch_type,
    num_pops: int,
) -> None:
    """Mutate all agents maintained by this process.

    Args:
        agents_batch: See return value of ``agents_batch`` from\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
        exchange_and_mutate_info_batch: See return value of\
            ``exchange_and_mutate_info_batch`` from\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
        num_pops: See\
            :meth:`~cneuromax.fitting.neuroevolution.space.base.BaseSpace.num_pops`.
    """
    seeds = exchange_and_mutate_info_batch[:, :, 3]
    for i in range(num_pops):
        for j in range(len(agents_batch)):
            agents_batch[i][j].curr_seed = seeds[i, j]
            agents_batch[i][j].mutate()


def run_evaluation_cpu(
    agents_batch: list[list[BaseSingularAgent]],
    space: BaseSpace,
    curr_gen: An[int, ge(1)],
) -> (
    fitnesses_and_num_env_steps_batch_type  # fitnesses_and_num_env_steps_batch
):
    """Evaluate all agents maintained by this process.

    Args:
        agents_batch: See return value of ``agents_batch`` from\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
        space: The experiment's pace, see\
            :class:`~cneuromax.fitting.neuroevolution.space.base.BaseSpace`.
        curr_gen: Current generation number.

    Returns:
        fitnesses_and_num_env_steps_batch: The output of\
            agent evaluation by this process. See\
            :meth:`~cneuromax.fitting.neuroevolution.space.base.BaseSpace.evaluate`.
    """
    fitnesses_and_num_env_steps_batch = np.zeros(
        shape=(len(agents_batch), space.num_pops, 2),
        dtype=np.float32,
    )
    for i in range(len(agents_batch)):
        fitnesses_and_num_env_steps_batch[i] = space.evaluate(
            agent_s=[agents_batch[i]],
            curr_gen=curr_gen,
        )
    return fitnesses_and_num_env_steps_batch


def run_evaluation_gpu(
    ith_gpu_comm: MPI.Comm,
    agents_batch: list[list[BaseSingularAgent]],
    space: BaseSpace,
    curr_gen: An[int, ge(1)],
    *,
    transfer: bool,
) -> (
    fitnesses_and_num_env_steps_batch_type  # fitnesses_and_num_env_steps_batch
):
    """Evaluate all agents maintained by this process.

    Args:
        ith_gpu_comm: See return value of ``ith_gpu_comm`` from\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_gpu_comm`.
        agents_batch: See return value of ``agents_batch`` from\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
        space: The experiment's pace, see\
            :class:`~cneuromax.fitting.neuroevolution.space.base.BaseSpace`.
        curr_gen: Current generation number.
        transfer: Whether any of\
            :paramref:`~cneuromax.fitting.neuroevolution.fit.NeuroevolutionFittingHydraConfig.env_transfer`,\
            :paramref:`~cneuromax.fitting.neuroevolution.fit.NeuroevolutionFittingHydraConfig.fit_transfer`\
            or\
            :paramref:`~cneuromax.fitting.neuroevolution.fit.NeuroevolutionFittingHydraConfig.mem_transfer`\
            is `True`.

    Returns:
        fitnesses_and_num_env_steps_batch: The output of\
            agent evaluation by this process. See\
            :meth:`~cneuromax.fitting.neuroevolution.space.base.BaseSpace.evaluate`.
    """
    comm, rank, size = retrieve_mpi_variables()
    ith_gpu_comm_rank = ith_gpu_comm.Get_rank()
    ith_gpu_batched_agents: list[
        list[list[BaseSingularAgent]]
    ] | None = ith_gpu_comm.gather(sendobj=agents_batch)
    if ith_gpu_comm_rank == 0:
        # `ith_gpu_agents_batch` is only `None` when
        # `ith_gpu_comm_rank != 0`. The following `assert` statement
        # is for static type checking reasons and has no execution
        # purposes.
        assert ith_gpu_batched_agents  # noqa: S101
        ith_gpu_agents_batch: list[list[BaseSingularAgent]] = []
        for agent_batch in ith_gpu_batched_agents:
            ith_gpu_agents_batch = ith_gpu_agents_batch + agent_batch
        ith_gpu_fitnesses_and_num_env_steps_batch = space.evaluate(
            ith_gpu_agents_batch,
            curr_gen,
        )
    fitnesses_and_num_env_steps_batch = np.empty(
        shape=(len(agent_batch), space.num_pops, 2),
        dtype=np.float32,
    )
    ith_gpu_comm.Scatter(
        sendbuf=None
        if ith_gpu_comm_rank != 0
        else ith_gpu_fitnesses_and_num_env_steps_batch,
        recvbuf=fitnesses_and_num_env_steps_batch,
    )
    # Send back the agents to their corresponding processes if
    # `transfer` is `True` as the agents have been modified by the
    # evaluation process.
    if transfer:
        # Prevents `agents_batch` from being overwritten.
        temp_agents_batch = ith_gpu_comm.scatter(sendobj=ith_gpu_agents_batch)
        for i in range(len(agent_batch)):
            agents_batch[i] = temp_agents_batch[i]
    return fitnesses_and_num_env_steps_batch_type
