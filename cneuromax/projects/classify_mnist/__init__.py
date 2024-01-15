"""MNIST classification ``project``."""
from hydra_zen import ZenStore

from cneuromax.fitting.deeplearning.runner import DeepLearningTaskRunner
from cneuromax.utils.hydra_zen import fs_builds

from .datamodule import (
    MNISTClassificationDataModule,
    MNISTClassificationDataModuleConfig,
)
from .litmodule import MNISTClassificationLitModule

__all__ = [
    "TaskRunner",
    "MNISTClassificationDataModuleConfig",
    "MNISTClassificationDataModule",
    "MNISTClassificationLitModule",
]


class TaskRunner(DeepLearningTaskRunner):
    """MNIST classification ``task`` runner."""

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        """Stores :mod:`hydra-core` MNIST classification configs.

        Args:
            store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        """
        super().store_configs(store)
        store(
            fs_builds(
                MNISTClassificationDataModule,
                config=MNISTClassificationDataModuleConfig(),
            ),
            name="classify_mnist",
            group="datamodule",
        )
        store(
            fs_builds(MNISTClassificationLitModule),
            name="classify_mnist",
            group="litmodule",
        )
