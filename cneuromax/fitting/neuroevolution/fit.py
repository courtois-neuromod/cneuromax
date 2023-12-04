"""Neuroevolution fitting."""

import wandb
from hydra.utils import instantiate

from cneuromax.fitting.neuroevolution.config import (
    NeuroevolutionFittingHydraConfig,
)
from cneuromax.fitting.neuroevolution.space import BaseSpace
from cneuromax.fitting.neuroevolution.utils.compute import (
    compute_pickled_agents_sizes,
    compute_save_points,
    compute_start_time_and_seeds,
    compute_total_num_env_steps_and_process_fitnesses,
)
from cneuromax.fitting.neuroevolution.utils.evolve import (
    run_evaluation_cpu,
    run_evaluation_gpu,
    run_mutation,
)
from cneuromax.fitting.neuroevolution.utils.exchange import (
    exchange_agents,
    update_exchange_and_mutate_info,
)
from cneuromax.fitting.neuroevolution.utils.initialize import (
    initialize_common_variables,
    initialize_gpu_comm,
)
from cneuromax.fitting.neuroevolution.utils.readwrite import (
    load_state,
    save_state,
)
from cneuromax.fitting.neuroevolution.utils.validate import (
    validate_space,
)
from cneuromax.fitting.neuroevolution.utils.wandb import setup_wandb
from cneuromax.utils.mpi import retrieve_mpi_variables


def fit(config: NeuroevolutionFittingHydraConfig) -> None:
    """Fitting function for Neuroevolution algorithms.

    This function is the main entry point of the Neuroevolution module.
    It acts as an interface between Hydra (configuration + launcher +
    sweeper), Spaces, Agents and MPI resource scheduling for
    Neuroevolution algorithms.

    Note that this function and all of its sub-functions will be called
    by `num_nodes * tasks_per_node` MPI processes/tasks. These two
    variables are set in the Hydra launcher configuration.

    Args:
        config: The run's :mod:`hydra-core` structured config, see\
            :class:`cneuromax.fitting.neuroevolution.config.NeuroevolutionFittingHydraConfig`.
    """
    comm, _, _ = retrieve_mpi_variables()
    space: BaseSpace = instantiate(config=config.space)
    validate_space(space, pop_merge=config.pop_merge)
    save_points = compute_save_points(
        prev_num_gens=config.prev_num_gens,
        total_num_gens=config.total_num_gens,
        save_interval=config.save_interval,
        save_first_gen=config.save_first_gen,
    )
    (
        pop_size,
        agents_batch,
        exchange_and_mutate_info,
        exchange_and_mutate_info_batch,
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
            len_agents_batch=len(agents_batch),
        )
    setup_wandb(entity=config.wandb_entity)
    for curr_gen in range(config.prev_num_gens + 1, config.total_num_gens + 1):
        start_time, seeds = compute_start_time_and_seeds(
            generation_results=generation_results,
            curr_gen=curr_gen,
            num_pops=space.num_pops,
            pop_size=pop_size,
            pop_merge=config.pop_merge,
        )
        if curr_gen > 1:
            update_exchange_and_mutate_info(
                num_pops=space.num_pops,
                pop_size=pop_size,
                exchange_and_mutate_info=exchange_and_mutate_info,
                generation_results=generation_results,
                seeds=seeds,
            )
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
        run_mutation(
            agents_batch=agents_batch,
            exchange_and_mutate_info_batch=exchange_and_mutate_info_batch,
            num_pops=space.num_pops,
        )
        fitnesses_and_num_env_steps_batch = (
            (
                run_evaluation_gpu(
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
            else run_evaluation_cpu(
                agents_batch=agents_batch,
                space=space,
                curr_gen=curr_gen,
            )
        )
        generation_results_batch[:, :, 0:2] = fitnesses_and_num_env_steps_batch
        compute_pickled_agents_sizes(
            generation_results_batch=generation_results_batch,
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
            )
    wandb.finish()
