""":class:`DeepLearningTaskRunner`."""
from typing import Any, ClassVar

from hydra_zen import ZenStore
from lightning.pytorch.loggers.wandb import WandbLogger

from cneuromax.fitting.deeplearning.config import (
    DeepLearningSubtaskConfig,
)
from cneuromax.fitting.deeplearning.train import train
from cneuromax.fitting.deeplearning.utils.zen import (
    store_basic_deeplearning_configs,
)
from cneuromax.fitting.runner import FittingTaskRunner
from cneuromax.utils.zen import store_wandb_logger_configs


class DeepLearningTaskRunner(FittingTaskRunner):
    """Deep Learning ``task`` runner.

    Attr:
        subtask_config: See :attr:`~.BaseTaskRunner.subtask_config`.
    """

    subtask_config: type[DeepLearningSubtaskConfig] = DeepLearningSubtaskConfig

    @classmethod
    def store_configs(
        cls: type["DeepLearningTaskRunner"],
        store: ZenStore,
    ) -> None:
        """Stores structured configs.

        .. warning::

            Make sure to call this method if you are overriding it.

        Args:
            store:\
                See :paramref:`~.FittingTaskRunner.store_configs.store`.
        """
        super().store_configs(store)
        store_basic_deeplearning_configs(store)
        store_wandb_logger_configs(store, clb=WandbLogger)
        store(
            fs_builds(cls.run_subtask, config=cls.subtask_config()),
            name="config",
        )

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
