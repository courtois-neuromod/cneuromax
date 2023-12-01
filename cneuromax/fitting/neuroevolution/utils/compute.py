"""Functions requiring some computation for Neuroevolution fitting."""


def compute_pickled_agents_sizes():
    pass


def compute_save_points():
    pass


def compute_start_time_and_seeds():
    pass


def process_fitnesses_and_total_num_env_steps(
    generation_results: generation_results_type,
    total_num_env_steps: An[int, ge(0)] | None,
    curr_gen: An[int, ge(0)],
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
