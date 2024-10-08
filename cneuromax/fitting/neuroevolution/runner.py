""":class:`.NeuroevolutionTaskRunner`."""

from collections.abc import Callable
from functools import partial
from typing import Any

import wandb
from hydra_zen import ZenStore

from cneuromax.fitting.neuroevolution.agent import BaseAgent
from cneuromax.fitting.neuroevolution.config import (
    NeuroevolutionSubtaskConfig,
    NeuroevolutionSubtaskTestConfig,
    NeuroevolutionTaskConfig,
)
from cneuromax.fitting.neuroevolution.fit import fit
from cneuromax.fitting.neuroevolution.space import BaseSpace
from cneuromax.fitting.runner import FittingTaskRunner
from cneuromax.store import store_wandb_logger_configs


class NeuroevolutionTaskRunner(FittingTaskRunner):
    """Neuroevolution ``task`` runner."""

    @classmethod
    def store_configs(
        cls: type["NeuroevolutionTaskRunner"],
        store: ZenStore,
    ) -> None:
        """Stores structured configs.

        .. warning::

            Make sure to call this method if you are overriding it.

        Args:
            store:
                See :paramref:`~.FittingTaskRunner.store_configs.store`.
        """
        super().store_configs(store=store)
        store_wandb_logger_configs(store, clb=wandb.init)
        store(NeuroevolutionTaskConfig, name="config")
        store(NeuroevolutionSubtaskTestConfig, group="config", name="test")

    @staticmethod
    def validate_subtask_config(config: NeuroevolutionSubtaskConfig) -> None:
        """Validates the ``subtask`` config.

        Args:
            config

        Raises:
            RuntimeError: If
                :paramref:`~.NeuroevolutionSubtaskConfig.device` is
                set to ``gpu`` but CUDA is not available.
        """
        if config.eval_num_steps == 0 and config.env_transfer:
            error_msg = "`env_transfer = True` requires `eval_num_steps > 0`."
            raise ValueError(error_msg)

    @classmethod
    def run_subtask(
        cls: type["NeuroevolutionTaskRunner"],
        space: BaseSpace,
        agent: partial[BaseAgent],
        logger: Callable[..., Any],
        config: NeuroevolutionSubtaskConfig,
    ) -> Any:  # noqa: ANN401
        """Runs the ``subtask``."""
        return fit(space=space, agent=agent, logger=logger, config=config)
