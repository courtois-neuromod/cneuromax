"""Run validation for Neuroevolution fitting."""

from cneuromax.fitting.neuroevolution.space.base import BaseSpace
from cneuromax.fitting.utils.hydra import get_launcher_config


def validate_space(space: BaseSpace, *, pop_merge: bool) -> None:
    """Makes sure that the Space is valid given config values.

    Args:
        space: See :paramref:`~.evaluate_on_cpu.space`.
        pop_merge: See\
            :paramref:`~.NeuroevolutionSubtaskConfig.pop_merge`.
    """
    launcher_config = get_launcher_config()
    if pop_merge and space.num_pops != 2:  # noqa: PLR2004
        error_msg = "`pop_merge = True` requires `num_pops = 2`."
        raise ValueError(error_msg)
    if not launcher_config.gpus_per_node and space.evaluates_on_gpu:
        error_msg = (
            "GPU evaluation is not supported when `gpus_per_node` is not "
            "specified in the launcher config or set to 0."
        )
        raise ValueError(error_msg)
