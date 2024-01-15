""":class:`NeuroevolutionTaskRunner`."""
from functools import partial
from typing import Any

import wandb
from hydra_zen import ZenStore

from cneuromax.fitting.neuroevolution.agent import BaseAgent
from cneuromax.fitting.neuroevolution.config import (
    NeuroevolutionSubtaskConfig,
)
from cneuromax.fitting.neuroevolution.evolve import evolve
from cneuromax.fitting.neuroevolution.space import BaseSpace
from cneuromax.fitting.runner import FittingTaskRunner
from cneuromax.store import store_wandb_logger_configs


class NeuroevolutionTaskRunner(FittingTaskRunner):
    """Neuroevolution ``task`` runner.

    Attributes:
        subtask_config: See :attr:`~.BaseTaskRunner.subtask_config`.
    """

    subtask_config: type[
        NeuroevolutionSubtaskConfig
    ] = NeuroevolutionSubtaskConfig

    @classmethod
    def store_configs(
        cls: type["NeuroevolutionTaskRunner"],
        store: ZenStore,
    ) -> None:
        """Stores structured configs.

        .. warning::

            Make sure to call this method if you are overriding it.

        Args:
            store:\
                See :paramref:`~.FittingTaskRunner.store_configs.store`.
        """
        cls.store_configs(store=store)
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

    @classmethod
    def run_subtask(
        cls: type["NeuroevolutionTaskRunner"],
        space: BaseSpace,
        agent: partial[BaseAgent],
        config: NeuroevolutionSubtaskConfig,
    ) -> Any:  # noqa: ANN401
        """Runs the ``subtask``."""
        return evolve(space=space, agent=agent, config=config)
