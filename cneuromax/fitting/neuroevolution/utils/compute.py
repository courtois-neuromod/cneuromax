"""Functions requiring some computation for Neuroevolution fitting."""

import logging
import pickle
import time
from typing import Annotated as An

import numpy as np

from cneuromax.fitting.neuroevolution.agent.singular import BaseSingularAgent
from cneuromax.fitting.neuroevolution.utils.type import (
    Fitnesses_and_num_env_steps_batch_type,
    Generation_results_batch_type,
    Generation_results_type,
    Seeds_type,
)
from cneuromax.utils.annotations import ge
from cneuromax.utils.mpi import retrieve_mpi_variables


def compute_generation_results(
    generation_results: Generation_results_type | None,
    generation_results_batch: Generation_results_batch_type,
    fitnesses_and_num_env_steps_batch: Fitnesses_and_num_env_steps_batch_type,
    agents_batch: list[list[BaseSingularAgent]],
    num_pops: An[int, ge(1)],
) -> None:
    """Computes generation results across processes and gathers them.

    Args:
        generation_results: See return value of ``generation_results``\
            from\
            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
        generation_results_batch: See return value of\
            ``generation_results_batch`` from\
            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
        fitnesses_and_num_env_steps_batch: See return value of\
            ``fitnesses_and_num_env_steps_batch`` from\
            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
        agents_batch: See return value of ``agents_batch`` from\
            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
        num_pops: See\
            :meth:`~.neuroevolution.space.base.BaseSpace.num_pops`.
    """
    comm, _, _ = retrieve_mpi_variables()
    # Store the fitnesses and number of environment steps
    generation_results_batch[:, :, 0:2] = fitnesses_and_num_env_steps_batch
    # Store the size of the agents
    for i in range(len(agents_batch)):
        for j in range(num_pops):
            generation_results_batch[i, j, 2] = len(
                pickle.dumps(obj=agents_batch[i][j]),
            )
    # Gather the results on the primary process
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
) -> list[int]:
    """Compute generations at which to save the state.

    Args:
        prev_num_gens: See\
            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.prev_num_gens`.
        total_num_gens: See\
            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.total_num_gens`.
        save_interval: See\
            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.save_interval`.
        save_first_gen: See\
            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.save_first_gen`.

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

    Args:
        generation_results: See return value of ``generation_results``\
            from\
            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
        curr_gen: Current generation number.
        num_pops: See\
            :meth:`~.neuroevolution.space.base.BaseSpace.num_pops`.
        pop_size: See return value of ``pop_size`` from\
            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
        pop_merge: See\
            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.pop_merge`.

    Returns:
        * The start time for the current generation.
        * The seeds for the current generation.
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
    # `generation_results` is only `None` when `rank != 0`. The
    # following `assert` statement is for static type checking reasons
    # and has no execution purposes.
    assert generation_results  # noqa: S101
    fitnesses = generation_results[:, :, 0]
    fitness_sorting_indices = np.argsort(a=fitnesses, axis=0)
    fitnesses_rankings = np.argsort(a=fitness_sorting_indices, axis=0)
    if curr_gen != 0:
        for j in range(num_pops):
            seeds[:, j] = seeds[:, j][fitnesses_rankings[:, j]]
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
        generation_results: See return value of ``generation_results``\
            from\
            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
        total_num_env_steps: See return value of\
            ``total_num_env_steps`` from\
            :func:`~.neuroevolution.utils.initialize.initialize_common_variables`.
        curr_gen: Current generation number.
        start_time: Generation start time.
        pop_merge: See\
            :paramref:`~.neuroevolution.config.NeuroevolutionFittingHydraConfig.pop_merge`.

    Returns:
        The updated total number of environment steps.
    """
    _, rank, _ = retrieve_mpi_variables()
    if rank != 0:
        return None
    # `generation_results`, `total_num_env_steps` & `start_time` are
    # only `None` when `rank != 0`. The following `assert` statements
    # are for static type checking reasons and have no execution
    # purposes.
    assert generation_results  # noqa: S101
    assert total_num_env_steps  # noqa: S101
    assert start_time  # noqa: S101
    fitnesses = generation_results[:, :, 0]
    if pop_merge:
        fitnesses[:, 0] += fitnesses[:, 1][::-1]
        fitnesses[:, 1] = fitnesses[:, 0][::-1]
    num_env_steps = generation_results[:, :, 1]
    total_num_env_steps += int(num_env_steps.sum())
    logging.info(f"{curr_gen+1}: {int(time.time() - start_time)}")
    logging.info(
        f"{np.mean(a=fitnesses, axis=0)}\n{np.max(a=fitnesses, axis=0)}",
    )
    return total_num_env_steps
