""":class:`NeuroevolutionTaskRunner`."""
from typing import Any

import wandb
from hydra_zen import ZenStore

from cneuromax.fitting.neuroevolution.config import (
    NeuroevolutionSubtaskConfig,
)
from cneuromax.fitting.neuroevolution.fit import fit
from cneuromax.fitting.runner import FittingTaskRunner
from cneuromax.utils.hydra_zen import store_wandb_logger_configs


class NeuroevolutionTaskRunner(FittingTaskRunner):
    """Neuroevolution ``task`` runner.

    Attr:
        subtask_config: See :attr:`~.BaseTaskRunner.subtask_config`.
    """

    subtask_config: type[
        NeuroevolutionSubtaskConfig
    ] = NeuroevolutionSubtaskConfig

    @staticmethod
    def store_configs(store: ZenStore) -> None:
        """Stores structured configs.

        .. warning::

            Make sure to call this method if you are overriding it.

        Args:
            store:\
                See :paramref:`~.FittingTaskRunner.store_configs.store`.
        """
        FittingTaskRunner.store_configs(store)
        store_wandb_logger_configs(store, clb=wandb.init)
        store(NeuroevolutionSubtaskConfig, name="neuroevolution")

    @staticmethod
    def validate_subtask_config(config: NeuroevolutionSubtaskConfig) -> None:
        """See :meth:`BaseTaskRunner.validate_subtask_config`.

        Args:
            config: See :attr:`subtask_config`

        Raises:
            RuntimeError: If ``.NeuroevolutionSubtaskConfig.device`` is
                set to ``gpu`` but CUDA is not available.
        """
        if config.eval_num_steps == 0 and config.env_transfer:
            error_msg = "`env_transfer = True` requires `eval_num_steps > 0`."
            raise ValueError(error_msg)
        if (
            config.total_num_gens - config.prev_num_gens
        ) % config.save_interval != 0:
            error_msg = (
                "`save_interval` must be a multiple of "
                "`total_num_gens - prev_num_gens`."
            )
            raise ValueError(error_msg)

    @staticmethod
    def run_subtask(config: subtask_config) -> Any:  # noqa: ANN401
        """Run the ``subtask`` given the :paramref:`config`.

        This method is meant to hold the ``subtask`` execution logic.

        Args:
            config: See :attr:`subtask_config`.
        """
        return fit(config)
