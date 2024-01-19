""":func:`evolve`."""
from collections.abc import Callable
from functools import partial
from typing import Any

from cneuromax.fitting.neuroevolution.agent import BaseAgent
from cneuromax.fitting.neuroevolution.config import (
    NeuroevolutionSubtaskConfig,
)
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
from cneuromax.fitting.neuroevolution.utils.validate import validate_space
from cneuromax.fitting.neuroevolution.utils.wandb import (
    setup_wandb,
    terminate_wandb,
)
from cneuromax.utils.mpi4py import get_mpi_variables


def evolve(
    space: BaseSpace,
    agent: partial[BaseAgent],
    logger: Callable[..., Any],
    config: NeuroevolutionSubtaskConfig,
) -> None:
    """Neuroevolution.

    Note that this function and all of its sub-functions will be called
    by ``num_nodes * tasks_per_node`` MPI processes/tasks. These two
    variables are set in the Hydra launcher configuration.

    Args:
        space: See :class:`~.space.BaseSpace`.
        agent: See :class:`~.agent.BaseAgent`.
        logger: See :func:`~.utils.wandb.setup_wandb`.
        config: See :paramref:`~.post_process_base_config.config`.
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
