""":func:`fit`, :func:`train` and :func:`test`."""
import logging
import pickle
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from cneuromax.fitting.neuroevolution.agent import BaseAgent
from cneuromax.fitting.neuroevolution.space import BaseSpace
from cneuromax.fitting.neuroevolution.utils.compute import (
    compute_generation_results,
    compute_save_points,
    compute_start_time_and_seeds,
    compute_total_num_env_steps_and_process_fitnesses,
)
from cneuromax.fitting.neuroevolution.utils.evolve import (
    evaluate_on_cpu,
    evaluate_on_gpu,
    mutate,
)
from cneuromax.fitting.neuroevolution.utils.exchange import (
    exchange_agents,
    update_exchange_and_mutate_info,
)
from cneuromax.fitting.neuroevolution.utils.initialize import (
    initialize_agents,
    initialize_common_variables,
    initialize_gpu_comm,
)
from cneuromax.fitting.neuroevolution.utils.readwrite import (
    load_state,
    save_state,
)
from cneuromax.fitting.neuroevolution.utils.type import Generation_results_type
from cneuromax.fitting.neuroevolution.utils.validate import validate_space
from cneuromax.fitting.neuroevolution.utils.wandb import (
    setup_wandb,
    terminate_wandb,
)
from cneuromax.utils.misc import seed_all
from cneuromax.utils.mpi4py import get_mpi_variables

from .config import (
    NeuroevolutionSubtaskConfig,
    NeuroevolutionSubtaskTestConfig,
)

MAX_INT = 2**31 - 1


def fit(
    space: BaseSpace,
    agent: partial[BaseAgent],
    logger: Callable[..., Any],
    config: NeuroevolutionSubtaskConfig,
) -> None:
    """Neuroevolution + testing.

    Note that this function and all of its sub-functions will be called
    by ``num_nodes * tasks_per_node`` MPI processes/tasks. These two
    variables are set in the Hydra launcher configuration.

    Args:
        space: See :class:`.BaseSpace`.
        agent: See :class:`~.BaseAgent`.
        logger: See :func:`~.utils.wandb.setup_wandb`.
        config: See :class:`.NeuroevolutionSubtaskConfig`.
    """
    if not isinstance(config, NeuroevolutionSubtaskTestConfig):
        evolve(space=space, agent=agent, logger=logger, config=config)
    else:
        test(space=space, config=config)


def evolve(
    space: BaseSpace,
    agent: partial[BaseAgent],
    logger: Callable[..., Any],
    config: NeuroevolutionSubtaskConfig,
) -> None:
    """Neuroevolution.

    Args:
        space: See :class:`.BaseSpace`.
        agent: See :class:`~.BaseAgent`.
        logger: See :func:`~.utils.wandb.setup_wandb`.
        config: See :class:`.NeuroevolutionSubtaskConfig`.
    """
    comm, _, _ = get_mpi_variables()
    validate_space(space=space, pop_merge=config.pop_merge)
    save_points = compute_save_points(
        prev_num_gens=config.prev_num_gens,
        total_num_gens=config.total_num_gens,
        save_interval=config.save_interval,
        save_first_gen=config.save_first_gen,
    )
    (
        pop_size,
        len_agents_batch,
        exchange_and_mutate_info,
        exchange_and_mutate_info_batch,
        seeds_batch,
        generation_results,
        generation_results_batch,
        total_num_env_steps,
    ) = initialize_common_variables(
        agents_per_task=config.agents_per_task,
        num_pops=space.num_pops,
    )
    if space.evaluates_on_gpu:
        ith_gpu_comm = initialize_gpu_comm()
    if config.prev_num_gens > 0:
        (
            agents_batch,
            generation_results,
            total_num_env_steps,
        ) = load_state(
            prev_num_gens=config.prev_num_gens,
            len_agents_batch=len_agents_batch,
            output_dir=config.output_dir,
        )
    else:
        agents_batch = initialize_agents(
            agent=agent,
            len_agents_batch=len_agents_batch,
            num_pops=space.num_pops,
            pop_merge=config.pop_merge,
        )
    setup_wandb(logger=logger)
    for curr_gen in range(config.prev_num_gens + 1, config.total_num_gens + 1):
        start_time, seeds = compute_start_time_and_seeds(
            generation_results=generation_results,
            curr_gen=curr_gen,
            num_pops=space.num_pops,
            pop_size=pop_size,
            pop_merge=config.pop_merge,
        )
        if curr_gen == 1:
            # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
            # for a full example execution of the genetic algorithm.
            # The following block is examplified in section 3.
            comm.Scatter(
                sendbuf=seeds,
                recvbuf=seeds_batch,
            )
            exchange_and_mutate_info_batch[:, :, 3] = seeds_batch
        else:
            update_exchange_and_mutate_info(
                num_pops=space.num_pops,
                pop_size=pop_size,
                exchange_and_mutate_info=exchange_and_mutate_info,
                generation_results=generation_results,
                seeds=seeds,
            )
            # See https://github.com/courtois-neuromod/cneuromax/blob/main/docs/genetic.pdf
            # for a full example execution of the genetic algorithm.
            # The following block is examplified in section 13.
            comm.Scatter(
                sendbuf=exchange_and_mutate_info,
                recvbuf=exchange_and_mutate_info_batch,
            )
            exchange_agents(
                num_pops=space.num_pops,
                pop_size=pop_size,
                agents_batch=agents_batch,
                exchange_and_mutate_info_batch=exchange_and_mutate_info_batch,
            )
        mutate(
            agents_batch=agents_batch,
            exchange_and_mutate_info_batch=exchange_and_mutate_info_batch,
            num_pops=space.num_pops,
        )
        fitnesses_and_num_env_steps_batch = (
            (
                evaluate_on_gpu(
                    ith_gpu_comm=ith_gpu_comm,
                    agents_batch=agents_batch,
                    space=space,
                    curr_gen=curr_gen,
                    transfer=config.env_transfer
                    or config.fit_transfer
                    or config.mem_transfer,
                )
            )
            if space.evaluates_on_gpu
            else evaluate_on_cpu(
                agents_batch=agents_batch,
                space=space,
                curr_gen=curr_gen,
            )
        )
        compute_generation_results(
            generation_results=generation_results,
            generation_results_batch=generation_results_batch,
            fitnesses_and_num_env_steps_batch=fitnesses_and_num_env_steps_batch,
            agents_batch=agents_batch,
            num_pops=space.num_pops,
        )
        # Primary process gathers fitnesses, number of environment steps
        # and pickled agent sizes
        comm.Gather(
            sendbuf=generation_results_batch,
            recvbuf=generation_results,
        )
        total_num_env_steps = (
            compute_total_num_env_steps_and_process_fitnesses(
                generation_results=generation_results,
                total_num_env_steps=total_num_env_steps,
                curr_gen=curr_gen,
                start_time=start_time,
                pop_merge=config.pop_merge,
            )
        )
        # State saving.
        if curr_gen in save_points:
            save_state(
                agents_batch=agents_batch,
                generation_results=generation_results,
                total_num_env_steps=total_num_env_steps,
                curr_gen=curr_gen,
                output_dir=config.output_dir,
            )
    terminate_wandb()


def test(
    space: BaseSpace,
    config: NeuroevolutionSubtaskTestConfig,
) -> None:
    """Neuroevolution testing.

    Args:
        space: See :class:`.BaseReinforcementSpace`.
        config: See :paramref:`.NeuroevolutionTestingSubtaskConfig`.
    """
    # Get MPI info
    comm, rank, size = get_mpi_variables()
    # Setup work distribution across MPI processes
    save_points = compute_save_points(
        prev_num_gens=config.prev_num_gens,
        total_num_gens=config.total_num_gens,
        save_interval=config.save_interval,
        save_first_gen=config.save_first_gen,
    )
    assigned_save_points = [
        save_points[i] for i in range(len(save_points)) if i % size == rank
    ]
    # Loop over work assigned to this MPI process
    for gen in assigned_save_points:
        # Validate path & load state
        path = Path(f"{config.output_dir}/{gen}/")
        if not (path / "state.pkl").is_file():
            logging.info(f"No saved state found at {path}.")
            continue
        if (path / "evaluation.pkl").is_file():
            logging.info(f"Already evaluated generation {gen}.")
            continue
        with (path / "state.pkl").open(mode="rb") as f:
            state = pickle.load(f)
        # Extract state information for testing
        agents: list[list[BaseAgent]] = state[0]
        pop_size: int = len(agents)
        generation_results: Generation_results_type = state[1]
        total_num_env_steps: int = state[2]
        fitnesses = generation_results[:, :, 0]
        fitnesses_sorting_indices = fitnesses.argsort(axis=0)
        fitnesses_index_ranking = fitnesses_sorting_indices.argsort(axis=0)
        selected = np.greater_equal(fitnesses_index_ranking, pop_size // 2)
        selected_indices = np.where(selected[:, 0])[0]
        # Setup evaluation
        scores = np.empty((pop_size // 2, config.num_tests))
        # Setting `eval_num_steps` to `test_num_steps` to have
        # later evaluate the agent for this potentially different
        # number of steps.
        space.config.eval_num_steps = config.test_num_steps
        # Loop over selected agents
        for i in range(pop_size // 2):
            agent: BaseAgent = agents[selected_indices[i]][0]
            for j in range(config.num_tests):
                logging.info(f"Test #{j}, agent #{i}, generation #{gen}.")
                seed_all(MAX_INT - j)
                # env,fit,env+fit,env+fit+mem: reset
                # mem,mem+fit: no reset
                if not (
                    agent.config.mem_transfer
                    or (
                        agent.config.mem_transfer and agent.config.fit_transfer
                    )
                ):
                    agent.reset()
                # Setting `fit_transfer` to `False` to have
                # `space.evaluate` return the evaluation score rather
                # than the continual fitness.
                agent.config.fit_transfer = False
                # Setting `env_transfer` to `False` to not have
                # `space.evaluate` loop forever.
                agent.config.env_transfer = False
                fitnesses_and_num_env_steps = space.evaluate(
                    agents=[[agent]],
                    curr_gen=MAX_INT - j,
                )
                scores[i][j] = fitnesses_and_num_env_steps[0]
        with (path / "evaluation.pkl").open(mode="wb") as f:
            pickle.dump([scores, total_num_env_steps], f)
