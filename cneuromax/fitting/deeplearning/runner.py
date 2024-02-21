""":class:`DeepLearningTaskRunner`."""

from functools import partial
from typing import Any

from hydra_zen import ZenStore
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from cneuromax.fitting.deeplearning.config import (
    DeepLearningSubtaskConfig,
    DeepLearningTaskConfig,
)
from cneuromax.fitting.deeplearning.datamodule import (
    BaseDataModule,
)
from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.fitting.deeplearning.litmodule.store import (
    store_basic_optimizer_configs,
    store_basic_scheduler_configs,
    store_mlp_config,
)
from cneuromax.fitting.deeplearning.store import (
    store_basic_trainer_config,
)
from cneuromax.fitting.deeplearning.train import train
from cneuromax.fitting.runner import FittingTaskRunner
from cneuromax.store import store_wandb_logger_configs


class DeepLearningTaskRunner(FittingTaskRunner):
    """Deep Learning ``task`` runner."""

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
        store_basic_optimizer_configs(store)
        store_basic_scheduler_configs(store)
        store_mlp_config(store)
        store_basic_trainer_config(store)
        store_wandb_logger_configs(
            store,
            clb=WandbLogger,
        )
        store(DeepLearningTaskConfig, name="config")

    @classmethod
    def run_subtask(  # noqa: PLR0913
        cls: type["DeepLearningTaskRunner"],
        trainer: partial[Trainer],
        datamodule: BaseDataModule,
        litmodule: BaseLitModule,
        logger: partial[WandbLogger],
        config: DeepLearningSubtaskConfig,
    ) -> Any:  # noqa: ANN401
        """Runs the ``subtask``."""
        return train(
            trainer=trainer,
            datamodule=datamodule,
            litmodule=litmodule,
            logger=logger,
            config=config,
        )
