"""Variable initialization for Neuroevolution fitting."""

from typing import Annotated as An

import numpy as np
from mpi4py import MPI

from cneuromax.fitting.neuroevolution.agent.singular import (
    BaseSingularAgent,
)
from cneuromax.fitting.neuroevolution.utils.type import (
    exchange_and_mutate_info_batch_type,
    exchange_and_mutate_info_type,
    generation_results_batch_type,
    generation_results_type,
)
from cneuromax.utils.annotations import ge, le
from cneuromax.utils.hydra import get_launcher_config
from cneuromax.utils.mpi import retrieve_mpi_variables


def initialize_common_variables(
    agents_per_task: An[int, ge(1)],
    num_pops: An[int, ge(1), le(2)],
) -> tuple[
    An[int, ge(1)],  # pop_size
    list[list[BaseSingularAgent]],  # agents_batch
    exchange_and_mutate_info_type | None,  # exchange_and_mutate_info
    exchange_and_mutate_info_batch_type,  # exchange_and_mutate_info_batch
    generation_results_type | None,  # generation_results
    generation_results_batch_type,  # generation_results_batch
    An[int, ge(0)] | None,  # total_num_env_steps
]:
    """Initializes variables common to all execution modes.

    Args:
        agents_per_task: See\
            :paramref:`~cneuromax.fitting.neuroevolution.fit.NeuroevolutionFittingHydraConfig.agents_per_task`.
        num_pops: See\
            :meth:`~cneuromax.fitting.neuroevolution.space.base.BaseSpace.num_pops`.

    Returns:
        pop_size: Number of agents per population, computed from\
            :paramref:`agents_per_task` and Hydra launcher values\
            `nodes` and `tasks_per_node`.
        agents_batch: A 2D list of agents maintained by this process
            during a given generation.
        exchange_and_mutate_info: An array maintained only by the\
            primary process (secondary processes set this to\
            `None`) containing information for all processes on\
            how to exchange and mutate agents. Precisions on the 3rd\
            dimension: 0) The size of the agent when serialized, 1)\
            The position of the agent paired for with the current\
            agent, 2) Whether to send or receive the agent, 3) The\
            seed to randomize the mutation and evaluation of the\
            agent.
        exchange_and_mutate_info_batch: A sub-array of\
            :paramref:`exchange_and_mutate_info` maintained by this\
            process.
        generation_results: An array maintained only by the primary\
            process (secondary processes set this to `None`)\
            containing several pieces of information about the\
            results of a given generation. Precisions on the 3rd\
            dimension: 0) Agent fitness, 1) Number of environment\
            steps taken by the agent during the evaluation, 2) Size\
            of the agent when serialized.
        generation_results_batch: A sub-array of\
            :paramref:`generation_results` maintained by this\
            process.
    """
    comm, rank, size = retrieve_mpi_variables()
    launcher_config = get_launcher_config()
    pop_size = (
        launcher_config.nodes
        * launcher_config.tasks_per_node
        * (agents_per_task)
    )
    len_agents_batch = pop_size // size
    agents_batch: list[list[BaseSingularAgent]] = []
    exchange_and_mutate_info = (
        None
        if rank != 0
        else np.empty(
            shape=(num_pops, pop_size, 4),
            dtype=np.uint32,
        )
    )
    exchange_and_mutate_info_batch = np.empty(
        shape=(num_pops, len_agents_batch, 4),
        dtype=np.uint32,
    )
    generation_results_batch = np.empty(
        shape=(num_pops, len_agents_batch, 3),
        dtype=np.float32,
    )
    generation_results = (
        None
        if rank != 0
        else np.empty(
            shape=(num_pops, pop_size, 3),
            dtype=np.float32,
        )
    )
    total_num_env_steps = None if rank != 0 else 0
    return (
        pop_size,
        agents_batch,
        exchange_and_mutate_info,
        exchange_and_mutate_info_batch,
        generation_results,
        generation_results_batch,
        total_num_env_steps,
    )


def initialize_gpu_comm() -> MPI.Comm:
    """Initializes a communicator for GPU work queueing.

    Assuming the experiment is ran with `N` MPI processes &
    `M` GPUs, this function will create `M` communicators, each
    containing `N/M` processes. Each communicator will be used to
    gather mutated agents onto one process, which will then
    evaluate them on the GPU.

    Returns:
        ith_gpu_comm: A communicator for GPU work queueing.
    """
    comm, rank, size = retrieve_mpi_variables()
    launcher_config = get_launcher_config()
    # `gpus_per_node` will have already been narrowed to a
    # positive integer by the
    # :func:`cneuromax.fitting.neuroevolution.fit.validate_config`
    # function.
    assert launcher_config.gpus_per_node  # noqa: S101
    tasks_per_gpu = size // launcher_config.gpus_per_node
    gpu_idx = rank // tasks_per_gpu
    ith_gpu_comm_task_list = np.arange(
        start=gpu_idx * tasks_per_gpu,
        stop=(gpu_idx + 1) * tasks_per_gpu,
    ).tolist()
    return comm.Create_group(comm.group.Incl(ith_gpu_comm_task_list))
