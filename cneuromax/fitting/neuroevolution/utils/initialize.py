"""Variable initialization for Neuroevolution fitting."""

from typing import Annotated as An

from cneuromax.fitting.neuroevolution.utils.type import (
    agents_batch_type,
    exchange_and_mutate_info_batch_type,
    exchange_and_mutate_info_type,
    generation_results_batch_type,
    generation_results_type,
)
from cneuromax.utils.annotations import ge, le


def initialize_common_variables(
    agents_per_task: An[int, ge(1)],
    num_pops: An[int, ge(1), le(2)],
) -> tuple[
    An[int, ge(1)],  # pop_size
    agents_batch_type,  # agents_batch
    generation_results_type | None,  # generation_results
    generation_results_batch_type,  # generation_results_batch
    exchange_and_mutate_info_type | None,  # exchange_and_mutate_info
    exchange_and_mutate_info_batch_type,  # exchange_and_mutate_info_batch
    An[int, ge(0)],  # total_num_env_steps
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
        agents_batch: The array of\
            :class:`cneuromax.fitting.neuroevolution.agent.singular.base.BaseSingularAgent`\
            instances maintained by the given process.
        generation_results: The array of\
    """
    pass
