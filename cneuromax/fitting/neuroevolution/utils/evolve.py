"""Evolutionary operations for Neuroevolution fitting.

The selection operation is implicit in :mod:`cneuromax`, see
:func:`.update_exchange_and_mutate_info` for more details.
"""

from typing import Annotated as An

import numpy as np
from mpi4py import MPI

from cneuromax.fitting.neuroevolution.agent import (
    BaseAgent,
)
from cneuromax.fitting.neuroevolution.space.base import BaseSpace
from cneuromax.fitting.neuroevolution.utils.type import (
    Exchange_and_mutate_info_batch_type,
    Fitnesses_and_num_env_steps_batch_type,
)
from cneuromax.utils.beartype import ge
from cneuromax.utils.misc import seed_all
from cneuromax.utils.mpi4py import get_mpi_variables


def mutate(
    agents_batch: list[list[BaseAgent]],
    exchange_and_mutate_info_batch: Exchange_and_mutate_info_batch_type,
    num_pops: int,
) -> None:
    """Mutate :paramref:`agents_batch`.

    Args:
        agents_batch: See
            :paramref:`~.compute_generation_results.agents_batch`.
        exchange_and_mutate_info_batch: A sub-array of
            :paramref:`~.update_exchange_and_mutate_info.exchange_and_mutate_info`
            maintained by this process.
        num_pops: See :meth:`~.BaseSpace.num_pops`.
    """
    seeds = exchange_and_mutate_info_batch[:, :, 3]
    for i in range(len(agents_batch)):
        for j in range(num_pops):
            seed_all(seed=seeds[i, j])
            # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
            # for a full example execution of the genetic algorithm.
            # The following block is examplified in section 4 & 16.
            agents_batch[i][j].mutate()


def evaluate_on_cpu(
    agents_batch: list[list[BaseAgent]],
    space: BaseSpace,
    curr_gen: An[int, ge(1)],
) -> (
    Fitnesses_and_num_env_steps_batch_type  # fitnesses_and_num_env_steps_batch
):
    """Evaluate :paramref:`agents_batch`.

    Args:
        agents_batch: See
            :paramref:`~.compute_generation_results.agents_batch`.
        space: The :class:`~.BaseSpace` instance used throughout the
            execution.
        curr_gen: See :paramref:`~.BaseSpace.curr_gen`.

    Returns:
        The output of agent evaluation performed by the process calling
            this function on the agents it maintains
            (:paramref:`agents_batch`). See
            :meth:`~.BaseSpace.evaluate`.
    """
    fitnesses_and_num_env_steps_batch = np.zeros(
        shape=(len(agents_batch), space.num_pops, 2),
        dtype=np.float32,
    )
    seed_all(seed=curr_gen)
    # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
    # for a full example execution of the genetic algorithm.
    # The following block is examplified in section 5.
    for i in range(len(agents_batch)):
        fitnesses_and_num_env_steps_batch[i] = space.evaluate(
            agents=[agents_batch[i]],
            curr_gen=curr_gen,
        )
    return fitnesses_and_num_env_steps_batch


def evaluate_on_gpu(
    ith_gpu_comm: MPI.Comm,
    agents_batch: list[list[BaseAgent]],
    space: BaseSpace,
    curr_gen: An[int, ge(1)],
    *,
    transfer: bool,
) -> (
    Fitnesses_and_num_env_steps_batch_type  # fitnesses_and_num_env_steps_batch
):
    """Gather :paramref:`agents_batch` on process subset & evaluate.

    Args:
        ith_gpu_comm: A :mod:`mpi4py` communicator used by existing CPU
            processes to exchange agents for GPU work queueing.
        agents_batch: See
            :paramref:`~.compute_generation_results.agents_batch`.
        space: See :paramref:`~.evaluate_on_cpu.space`.
        curr_gen: See :paramref:`~.BaseSpace.curr_gen`.
        transfer: Whether any of
            :paramref:`~.NeuroevolutionSubtaskConfig.env_transfer`,
            :paramref:`~.NeuroevolutionSubtaskConfig.fit_transfer`
            or
            :paramref:`~.NeuroevolutionSubtaskConfig.mem_transfer`
            is ``True``.

    Returns:
        The output of agent evaluation by this process. See
            :meth:`~.BaseSpace.evaluate`.
    """
    comm, rank, size = get_mpi_variables()
    ith_gpu_comm_rank = ith_gpu_comm.Get_rank()
    # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
    # for a full example execution of the genetic algorithm.
    # The following block is examplified in section 5.
    # As opposed to the CPU evaluation, agents are not evaluated on the
    # process that mutates them but instead gathered on a single process
    # that evaluates them on the GPU, before sending back their
    # fitnesses to the process that mutated them.
    ith_gpu_batched_agents: list[list[list[BaseAgent]]] | None = (
        ith_gpu_comm.gather(sendobj=agents_batch)
    )
    if ith_gpu_comm_rank == 0:
        # `ith_gpu_agents_batch` is only `None` when
        # `ith_gpu_comm_rank != 0`. The following `assert` statement
        # is for static type checking reasons and has no execution
        # purposes.
        assert ith_gpu_batched_agents is not None  # noqa: S101
        ith_gpu_agents_batch: list[list[BaseAgent]] = []
        for agent_batch in ith_gpu_batched_agents:
            ith_gpu_agents_batch = ith_gpu_agents_batch + agent_batch
        seed_all(seed=curr_gen)
        ith_gpu_fitnesses_and_num_env_steps_batch = space.evaluate(
            ith_gpu_agents_batch,
            curr_gen,
        )
    fitnesses_and_num_env_steps_batch = np.empty(
        shape=(len(agent_batch), space.num_pops, 2),
        dtype=np.float32,
    )
    ith_gpu_comm.Scatter(
        sendbuf=(
            None
            if ith_gpu_comm_rank != 0
            else ith_gpu_fitnesses_and_num_env_steps_batch
        ),
        recvbuf=fitnesses_and_num_env_steps_batch,
    )
    # Send back the agents to their corresponding processes if
    # `transfer == True` as the agents have been modified by the
    # evaluation process.
    if transfer:
        # Prevents `agents_batch` from being overwritten.
        temp_agents_batch = ith_gpu_comm.scatter(sendobj=ith_gpu_agents_batch)
        for i in range(len(agent_batch)):
            agents_batch[i] = temp_agents_batch[i]
    return fitnesses_and_num_env_steps_batch
