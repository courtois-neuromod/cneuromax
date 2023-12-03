"""Functions requiring some computation for Neuroevolution fitting."""

import pickle
import time
from typing import Annotated as An

import numpy as np

from cneuromax.fitting.neuroevolution.agent.singular import BaseSingularAgent
from cneuromax.fitting.neuroevolution.utils.type import (
    exchange_and_mutate_info_batch_type,
    exchange_and_mutate_info_type,
    fitnesses_and_num_env_steps_batch_type,
    generation_results_batch_type,
    generation_results_type,
    seeds_type,
)
from cneuromax.utils.annotations import ge
from cneuromax.utils.mpi import retrieve_mpi_variables


def compute_pickled_agents_sizes(
    generation_results_batch: generation_results_batch_type,
    agents_batch: list[list[BaseSingularAgent]],
    num_pops: An[int, ge(1)],
) -> None:
    """Compute this process' maintained agents' serialized sizes.

    Args:
        generation_results_batch: See return `generation_results_batch`\
            from\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
        agents_batch: See return value of `agents_batch` from\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
        num_pops: See\
            :meth:`~cneuromax.fitting.neuroevolution.space.base.BaseSpace.num_pops`.
    """
    for i in range(len(agents_batch)):
        for j in range(num_pops):
            generation_results_batch[i, j, 2] = len(
                pickle.dumps(agents_batch[i][j]),
            )


def compute_save_points(
    prev_num_gens: An[int, ge(0)],
    total_num_gens: An[int, ge(0)],
    save_interval: An[int, ge(0)],
    *,
    save_first_gen: bool,
) -> list[int]:
    """Compute generations at which to save the state.

    Args:
        prev_num_gens: See\
            :paramref:`~cneuromax.fitting.neuroevolution.fit.NeuroevolutionFittingHydraConfig.prev_num_gens`.
        total_num_gens: See\
            :paramref:`~cneuromax.fitting.neuroevolution.fit.NeuroevolutionFittingHydraConfig.total_num_gens`.
        save_interval: See\
            :paramref:`~cneuromax.fitting.neuroevolution.fit.NeuroevolutionFittingHydraConfig.save_interval`.
        save_first_gen: See\
            :paramref:`~cneuromax.fitting.neuroevolution.fit.NeuroevolutionFittingHydraConfig.save_first_gen`.

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
    generation_results: generation_results_type,
    curr_gen: An[int, ge(1)],
    num_pops: An[int, ge(1)],
    pop_size: An[int, ge(1)],
    *,
    pop_merge: bool,
) -> tuple[float | None, seeds_type | None]:  # start_time, seeds
    """Compute the start time and seeds for the current generation.

    Args:
        generation_results: See return value of `generation_results`\
            from\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
        curr_gen: Current generation number.
        num_pops: See\
            :meth:`~cneuromax.fitting.neuroevolution.space.base.BaseSpace.num_pops`.
        pop_size: See return value of `pop_size` from\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
        pop_merge: See\
            :paramref:`~cneuromax.fitting.neuroevolution.fit.NeuroevolutionFittingHydraConfig.pop_merge`.

    Returns:
        * **start_time** - The start time for the current generation.
        * **seeds** - The seeds for the current generation.
    """
    comm, rank, size = retrieve_mpi_variables()
    np.random.seed(seed=curr_gen)
    if rank != 0:
        return None, None
    start_time = time.time()
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
        seeds = np.repeat(a=seeds, repeats=2, axis=1)
        if curr_gen == 1:
            seeds[:, 1] = seeds[:, 1][::-1]
    fitnesses = generation_results[:, :, 0]
    fitness_sorting_indices = np.argsort(a=fitnesses, axis=0)
    fitnesses_rankings = np.argsort(a=fitness_sorting_indices, axis=0)
    if curr_gen != 0:
        for j in range(num_pops):
            seeds[:, j] = seeds[:, j][fitnesses_rankings[:, j]]
    return start_time, seeds


def compute_total_num_env_steps_and_process_fitnesses(
    generation_results: generation_results_type,
    total_num_env_steps: An[int, ge(0)] | None,
    curr_gen: An[int, ge(1)],
    start_time: float,
    pop_merge: bool,
) -> An[int, ge(0)]:  # total_num_env_steps
    """Processes the generation results

    Args:
        generation_results: See return value of\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
        total_num_env_steps: See return value of\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
        curr_gen: Current generation number.
        start_time: See return value of\
            :func:`~cneuromax.fitting.neuroevolution.utils.initialize.initialize_common_variables`.
        pop_merge: See\
            :paramref:`~cneuromax.fitting.neuroevolution.fit.NeuroevolutionFittingHydraConfig.pop_merge`.

    Returns:
        total_num_env_steps: The updated number of environment steps.
    """
    comm, rank, size = retrieve_mpi_variables()
    if pop_merge:
        # Primary process merges populations.
        if rank == 0:
            generation_results = generation_results.reshape(
                (
                    generation_results.shape[0] * generation_results.shape[1],
                    generation_results.shape[2],
                )
            )
    # Primary process selects the next generation of agents.
    if rank == 0:
        total_num_env_steps = total_num_env_steps + np.sum(
            generation_results[:, :, 1]
        )
        generation_results = generation_results[
            np.argsort(generation_results[:, :, 0])[::-1]
        ]
        generation_results = generation_results[
            : generation_results.shape[0] // 2
        ]
        generation_results = generation_results.reshape(
            (
                generation_results.shape[0] // generation_results.shape[1],
                generation_results.shape[1],
                generation_results.shape[2],
            )
        )
