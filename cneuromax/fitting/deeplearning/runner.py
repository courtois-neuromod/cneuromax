""":class:`DeepLearningTaskRunner`."""
from typing import Any

from hydra_zen import ZenStore
from lightning.pytorch.loggers.wandb import WandbLogger

from cneuromax.fitting.deeplearning.config import (
    DeepLearningSubtaskConfig,
)
from cneuromax.fitting.deeplearning.train import train
from cneuromax.fitting.runner import FittingTaskRunner
from cneuromax.utils.wandb import store_wandb_logger_configs


class DeepLearningTaskRunner(FittingTaskRunner):
    """Deep Learning ``task`` runner.

    Attr:
        subtask_config: See :attr:`~.BaseTaskRunner.subtask_config`.
    """

    subtask_config: type[DeepLearningSubtaskConfig] = DeepLearningSubtaskConfig

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
        store_wandb_logger_configs(store, clb=WandbLogger)
        store(DeepLearningSubtaskConfig, name="deeplearning")

    @staticmethod
    def run_subtask(
        config: DeepLearningSubtaskConfig,  # type: ignore[override]
    ) -> Any:  # noqa: ANN401
        """Run the ``subtask`` given the :paramref:`config`.

        This method is meant to hold the ``subtask`` execution logic.

        Args:
            config: See :attr:`subtask_config`.
        """
        return train(config)
