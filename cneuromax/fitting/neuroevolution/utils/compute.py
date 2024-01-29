"""Not agent-based computation functions for Neuroevolution fitting."""

import logging
import pickle
import time
from typing import Annotated as An

import numpy as np
import wandb

from cneuromax.fitting.neuroevolution.agent import BaseAgent
from cneuromax.fitting.neuroevolution.utils.type import (
    Fitnesses_and_num_env_steps_batch_type,
    Generation_results_batch_type,
    Generation_results_type,
    Seeds_type,
)
from cneuromax.utils.beartype import ge
from cneuromax.utils.mpi4py import get_mpi_variables


def compute_generation_results(
    generation_results: Generation_results_type | None,
    generation_results_batch: Generation_results_batch_type,
    fitnesses_and_num_env_steps_batch: Fitnesses_and_num_env_steps_batch_type,
    agents_batch: list[list[BaseAgent]],
    num_pops: An[int, ge(1)],
) -> None:
    """Fills the :paramref:`generation_results` array with results.

    Extracts the fitnesses & number of environment steps from
    :paramref:`fitnesses_and_num_env_steps_batch`, computes the
    pickled agent sizes and stores all of this information in
    :paramref:`generation_results`.

    Args:
        generation_results: An array maintained solely by the\
            primary process (secondary processes have this variable\
            set to ``None``) containing several pieces of information\
            about the results of a given generation. The 3rd\
            dimension contains the following information at the\
            following indices: 0) Agent fitness, 1) Number of\
            environment steps taken by the agent during the\
            evaluation, 2) Size of the agent when serialized.
        generation_results_batch: A sub-array of\
            :paramref:`generation_results` maintained by the process\
            calling this function.
        fitnesses_and_num_env_steps_batch: The output values of\
            the evaluation performed in :func:`.evaluate_on_cpu`\
            or :func:`.evaluate_on_gpu` on the agents maintained\
            by the process calling this function.
        agents_batch: A 2D list of agents maintained by the process\
            calling this function.
        num_pops: See :meth:`~.BaseSpace.num_pops`.
    """
    comm, _, _ = get_mpi_variables()
    # Store the fitnesses and number of environment steps
    generation_results_batch[:, :, 0:2] = fitnesses_and_num_env_steps_batch
    # Store the size of the agents
    for i in range(len(agents_batch)):
        for j in range(num_pops):
            generation_results_batch[i, j, 2] = len(
                pickle.dumps(obj=agents_batch[i][j]),
            )
    # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
    # for a full example execution of the genetic algorithm.
    # The following block is examplified in section 6.
    comm.Gather(
        sendbuf=generation_results_batch,
        recvbuf=generation_results,
    )


def compute_save_points(
    prev_num_gens: An[int, ge(0)],
    total_num_gens: An[int, ge(0)],
    save_interval: An[int, ge(0)],
    *,
    save_first_gen: bool,
) -> list[int]:  # save_points
    """Compute generations at which to save the state.

    Args:
        prev_num_gens: See\
            :paramref:`~.NeuroevolutionSubtaskConfig.prev_num_gens`.
        total_num_gens: See\
            :paramref:`~.NeuroevolutionSubtaskConfig.total_num_gens`.
        save_interval: See\
            :paramref:`~.NeuroevolutionSubtaskConfig.save_interval`.
        save_first_gen: See\
            :paramref:`~.NeuroevolutionSubtaskConfig.save_first_gen`.

    Returns:
        A list of generations at which to save the state.
    """
    save_points: list[int] = list(
        range(
            prev_num_gens + save_interval,
            total_num_gens + 1,
            save_interval,
        ),
    )
    if save_first_gen and save_interval != 1:
        save_points = [prev_num_gens + 1, *save_points]
    return save_points


def compute_start_time_and_seeds(
    generation_results: Generation_results_type | None,
    curr_gen: An[int, ge(1)],
    num_pops: An[int, ge(1)],
    pop_size: An[int, ge(1)],
    *,
    pop_merge: bool,
) -> tuple[float | None, Seeds_type | None]:  # start_time, seeds
    """Compute the start time and seeds for the current generation.

    Fetches the start time and generates the seeds for the current\
    generation. If :paramref:`pop_merge` is ``True``, the seeds are\
    shared between the populations.

    Args:
        generation_results: See\
            :paramref:`~compute_generation_results.generation_results`.
        curr_gen: See :paramref:`~.BaseSpace.curr_gen`.
        num_pops: See :meth:`~.BaseSpace.num_pops`.
        pop_size: Total number of agent per population.
        pop_merge: See\
            :paramref:`~.NeuroevolutionSubtaskConfig.pop_merge`.

    Returns:
        * The start time for the current generation.
        * See\
            :paramref:`~.update_exchange_and_mutate_info.seeds`.
    """
    comm, rank, size = get_mpi_variables()
    np.random.seed(seed=curr_gen)
    if rank != 0:
        return None, None
    start_time = time.time()
    # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
    # for a full example execution of the genetic algorithm.
    # The following block is examplified in section 1 & 8.
    seeds = np.random.randint(
        low=0,
        high=2**32,
        size=(
            pop_size,
            1 if pop_merge else num_pops,
        ),
        dtype=np.uint32,
    )
    if pop_merge:
        # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
        # for a full example execution of the genetic algorithm.
        # The following block is examplified in section 2 & 9.
        seeds = np.repeat(a=seeds, repeats=2, axis=1)
        if curr_gen == 1:
            seeds[:, 1] = seeds[:, 1][::-1]
    if curr_gen > 1:
        # `generation_results` is only `None` when `rank != 0`. The
        # following `assert` statement is for static type checking
        # reasons and has no execution purposes.
        assert generation_results is not None  # noqa: S101
        fitnesses = generation_results[:, :, 0]
        # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
        # for a full example execution of the genetic algorithm.
        # The following block is examplified in section 10.
        fitnesses_sorting_indices = fitnesses.argsort(axis=0)
        fitnesses_index_ranking = fitnesses_sorting_indices.argsort(axis=0)
        # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
        # for a full example execution of the genetic algorithm.
        # The following block is examplified in section 11.
        for j in range(num_pops):
            seeds[:, j] = seeds[:, j][fitnesses_index_ranking[:, j]]
    return start_time, seeds


def compute_total_num_env_steps_and_process_fitnesses(
    generation_results: Generation_results_type | None,
    total_num_env_steps: An[int, ge(0)] | None,
    curr_gen: An[int, ge(1)],
    start_time: float | None,
    *,
    pop_merge: bool,
) -> An[int, ge(0)] | None:  # total_num_env_steps
    """Processes the generation results.

    Args:
        generation_results: See\
            :paramref:`~.compute_generation_results.generation_results`.
        total_num_env_steps: The total number of environment\
            steps taken by all agents during the entire experiment.\
            This variable is maintained solely by the primary process\
            (secondary processes set this to ``None``).
        curr_gen: See :paramref:`~.BaseSpace.curr_gen`.
        start_time: Generation start time.
        pop_merge: See\
            :paramref:`~.NeuroevolutionSubtaskConfig.pop_merge`.

    Returns:
        The updated total number of environment steps.
    """
    _, rank, _ = get_mpi_variables()
    if rank != 0:
        return None
    # `generation_results`, `total_num_env_steps` & `start_time` are
    # only `None` when `rank != 0`. The following `assert` statements
    # are for static type checking reasons and have no execution
    # purposes.
    assert generation_results is not None  # noqa: S101
    assert total_num_env_steps is not None  # noqa: S101
    assert start_time is not None  # noqa: S101
    fitnesses = generation_results[:, :, 0]
    if pop_merge:
        # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
        # for a full example execution of the genetic algorithm.
        # The following block is examplified in section 7.
        fitnesses[:, 0] += fitnesses[:, 1][::-1]
        fitnesses[:, 1] = fitnesses[:, 0][::-1]
    num_env_steps = generation_results[:, :, 1]
    total_num_env_steps += int(num_env_steps.sum())
    elapsed_time = time.time() - start_time
    fitnesses_mean = fitnesses.mean(axis=0)
    fitnesses_max = fitnesses.max(axis=0)
    logging.info(f"{curr_gen}: {elapsed_time}")
    logging.info(f"{fitnesses_mean}\n{fitnesses_max}\n")
    wandb.log(
        {
            "gen": curr_gen,
            "fitnesses_mean": fitnesses_mean,
            "fitnesses_max": fitnesses_max,
            "elapsed_time": elapsed_time,
            "total_num_env_steps": total_num_env_steps,
        },
    )
    return total_num_env_steps
