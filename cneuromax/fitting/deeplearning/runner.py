""":class:`DeepLearningTaskRunner`."""
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from hydra_zen import ZenStore, make_config
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.optim import SGD, Adam, AdamW
from transformers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)

from cneuromax.fitting.config import (
    FittingSubtaskConfig,
)
from cneuromax.fitting.deeplearning.datamodule import (
    BaseDataModule,
    BaseDataModuleConfig,
)
from cneuromax.fitting.deeplearning.litmodule import BaseLitModule
from cneuromax.fitting.deeplearning.train import train
from cneuromax.fitting.runner import FittingTaskRunner
from cneuromax.utils.hydra_zen import (
    fs_builds,
    pfs_builds,
    store_wandb_logger_configs,
)


def store_basic_optimizer_configs(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` ``litmodule/optimizer`` group configs.

    Config names: ``adam``, ``adamw``, ``sgd``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(pfs_builds(Adam), name="adam", group="litmodule/optimizer")
    store(pfs_builds(AdamW), name="adamw", group="litmodule/optimizer")
    store(pfs_builds(SGD), name="sgd", group="litmodule/optimizer")


def store_basic_scheduler_configs(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` ``litmodule/scheduler`` group configs.

    Config names: ``constant``, ``linear_warmup``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        pfs_builds(get_constant_schedule),
        name="constant",
        group="litmodule/scheduler",
    )
    store(
        pfs_builds(get_constant_schedule_with_warmup),
        name="linear_warmup",
        group="litmodule/scheduler",
    )


def store_basic_trainer_config(store: ZenStore) -> None:
    """Stores :mod:`hydra-core` ``trainer`` group configs.

    Config name: ``base``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        pfs_builds(
            Trainer,
            accelerator="${config.device}",
            default_root_dir="${config.output_dir}/lightning/",
        ),
        name="base",
        group="trainer",
    )


@dataclass
class DeepLearningTaskConfig(
    make_config(  # type: ignore[misc]
        trainer=pfs_builds(Trainer),
        logger=pfs_builds(WandbLogger),
        datamodule=fs_builds(BaseDataModule, config=BaseDataModuleConfig()),
        litmodule=fs_builds(BaseLitModule),
        config=fs_builds(FittingSubtaskConfig),
    ),
):
    """Deep Learning ``task`` config.

    Args:
        trainer: See :class:`~lightning.pytorch.Trainer`.
        logger: See\
            :class:`~lightning.pytorch.loggers.wandb.WandbLogger`.
        datamodule: See :class:`.BaseDataModule`.
        litmodule: See :class:`.BaseLitModule`.
        config: See :class:`.FittingSubtaskConfig`.
        defaults: Hydra defaults.
    """

    defaults: list[Any] = field(
        default_factory=lambda: [
            {"trainer": "base"},
            {"litmodule/scheduler": "constant"},
            {"litmodule/optimizer": "adamw"},
            {"logger": "wandb"},
            "_self_",
            {"task": None},
        ],
    )


class DeepLearningTaskRunner(FittingTaskRunner):
    """Deep Learning ``task`` runner.

    Attr:
        subtask_config: See :attr:`~.BaseTaskRunner.subtask_config`.
    """

    task_config = DeepLearningTaskConfig

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
        store_basic_trainer_config(store)
        store_wandb_logger_configs(store, clb=WandbLogger)
        store(DeepLearningTaskConfig, name="config")

    @classmethod
    def run_subtask(
        cls: type["DeepLearningTaskRunner"],
        trainer: partial[Trainer],
        logger: partial[WandbLogger],
        datamodule: BaseDataModule,
        litmodule: BaseLitModule,
        config: FittingSubtaskConfig,
    ) -> Any:  # noqa: ANN401
        """Run the ``subtask``.

        This method is meant to hold the ``subtask`` execution logic.
        """
        return train(trainer, logger, datamodule, litmodule, config)
